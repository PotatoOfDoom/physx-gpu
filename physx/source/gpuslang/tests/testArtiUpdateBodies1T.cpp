// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

// P9.43b: `updateBodiesLaunch1T` (forwardDynamic2.cu:3047-3075, body
// updateBodiesInternal :2984-3045) on the engine ABI: block (32, 2, 1), grid
// ceil(130 / 64) = 3, the {desc, dt, bool} arg pack, BOTH values of `integrate`.
//
// THE ORACLE is a transcription of the .cu over the engine's own host-callable
// helpers (PxMat33(PxQuat), Cm::transformInertiaTensor, PxMat33::getInverse, the
// PxgArticulation.h block loads) plus the three hoisted mirrors this kernel
// composes -- aiMirror (P8.94c), fvMirror (P8.94b), refUpdateRootBody /
// refPropagateLink / refSphericalJointPositions (P8.94a) -- and two new
// transcriptions: computeAndEnforceJointPositions and conserveMomentum.
//
// WHAT IS EXACT AND WHAT IS NOT.  conserveMomentum is add / mul / div / sqrt:
// bit-exact when the association is kept.  The flush and the averaging are exact.
// The transcendentals live in the POSE paths: on integrate == false only the
// spherical re-derivation (atan2) touches mJointPositions; on integrate == true
// the poses go through PxExp / sin / cos and everything conserveMomentum derives
// from them (link velocities, positions) inherits llvmpipe's ~2-ULP
// transcendental noise.  The tolerance is therefore scoped PER PATH:
//   integrate == false: dof mJointPositions and the record jointPositions;
//   integrate == true:  those, plus link mAccumulatedPose and mMotionVelocity.
// Every other word -- the flushed joint velocities, the accelerations, the state
// clear, the untouched blocks, all the poison -- must match bit for bit.
//
// SHAPED TO SEE
//   * 130 articulations: three X blocks, five live warp slices, a sixth that must
//     come back poisoned; the record index a PERMUTATION of the fleet slot;
//   * fixed AND free bases in every warp (conserveMomentum is a no-op for a
//     fixed base; updatePoses skips the root; both still walk the links);
//   * stateDirty 0..3, dirty slab pairs on both alignments (the P8.94d seeds);
//   * every joint type, with revolute positions seeded beyond +-2pi (the wrap),
//     limited prismatic dofs whose positions sit below AND above their limits
//     (5.12 dropped the clamp: they must stay put), and spherical dofs (the
//     atan2 re-derivation);
//   * a free-base articulation AT REST -- every velocity zero, no dirty flags --
//     which drives conserveMomentum's aDenom == 0 guard and makes every
//     accumulator's first term a signed zero;
//   * a free-base articulation NEAR rest, velocities around 1e-25, which drives the
//     same guard with a momentum vector that is NOT zero: at rest the guard's value
//     (0, where the acceleration kernel's twin uses 1) scales a zero and cannot be
//     observed at all -- the mutation sweep found that blind spot;
//   * per-record joint arrays at the packed stride with a canary past the last dof
//     that is NOT the dof blocks' poison, so a writeback that runs past mTotalDofs is
//     seen (it was the same bit pattern, and the extra write was therefore invisible).

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"
#include "pxgArtiRefHelpers.h"

#include "foundation/PxMat33.h"
#include "foundation/PxMathUtils.h"
#include "foundation/PxQuat.h"
#include "foundation/PxTransform.h"
#include "foundation/PxVec3.h"
#include "CmSpatialVector.h"
#include "CmUtils.h"
#include "PxgArticulation.h"
#include "PxgArticulationCoreDesc.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::PxMat33;
using physx::PxQuat;
using physx::PxReal;
using physx::PxTransform;
using physx::PxU32;
using physx::PxVec3;
using physx::Cm::UnAlignedSpatialVector;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;
using physx::loadSpatialVector;
using physx::storeSpatialVector;
using physx::loadSpatialTransform;
using physx::storeSpatialTransform;
using physx::storeSpatialMatrix;
using physx::loadPxMat33;
using physx::storePxMat33;
using physx::loadPxVec3;
using pxgref::J_FIX;
using pxgref::J_PRISMATIC;
using pxgref::J_REVOLUTE;
using pxgref::J_SPHERICAL;
using pxgref::AI_NONE;

static std::vector<uint8_t> a1Loader(const char* file) { return loadShaderBlob(file); }

static uint32_t a1F2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static uint32_t a1UlpDiff(float a, float b)
{
    if (a1F2u(a) == a1F2u(b)) return 0;
    if (std::isnan(a) || std::isnan(b)) return 0xFFFFFFFFu;
    int32_t ia, ib;
    std::memcpy(&ia, &a, 4);
    std::memcpy(&ib, &b, 4);
    if (ia < 0) ia = int32_t(0x80000000) - ia;
    if (ib < 0) ib = int32_t(0x80000000) - ib;
    return uint32_t(ia > ib ? ia - ib : ib - ia);
}

const uint32_t A1_ARTS = 130;
const uint32_t A1_WARPS = 6;          // ceil(130/64) blocks x 2 warps; slice 5 stays poisoned
const uint32_t A1_MAX_LINKS = 6;
const uint32_t A1_MAX_DOFS = 13;
const uint32_t A1_NB_SLABS = 3;
const float A1_DT = 0.02734375f;      // 7/256

// PxArticulationMotion (PxSolverDefs.h:287-291)
const uint8_t A1_MOTION_LOCKED = 0, A1_MOTION_LIMITED = 1, A1_MOTION_FREE = 2;

static uint32_t a1NumLinks(uint32_t a) { return 2u + (a % 5u); }
static uint8_t a1JointType(uint32_t a, uint32_t i) { return uint8_t((a + i) % 5u); }
static uint8_t a1Dofs(uint8_t jt) { return jt == J_SPHERICAL ? uint8_t(3) : (jt == J_FIX ? uint8_t(0) : uint8_t(1)); }
static bool a1FixBase(uint32_t a) { return (a % 3u) == 1u; }
// the free articulations at rest: no velocity anywhere, nothing dirty
static bool a1AtRest(uint32_t a) { return (a % 13u) == 8u && !a1FixBase(a); }
// NEAR rest, which is not the same experiment.  conserveMomentum guards its angular
// ratio on `aDenom == 0` and yields ZERO there (the CUDA's 0, not the 1 of the
// acceleration kernel's twin).  At rest the guard fires but cannot be observed: the
// vector it scales is exactly zero, so 0 and 1 both give zero.  These articulations
// carry velocities around 1e-25 instead: every component is a perfectly normal float,
// but its SQUARE (1e-50) is below the smallest denormal, so the magnitude underflows
// to zero while the momentum vector itself is not zero -- the guard fires AND the
// factor it produces is visible.  Nothing else in the fixture reaches denormal range.
static const float A1_TINY = 1e-25f;
static bool a1NearRest(uint32_t a) { return (a % 13u) == 5u && !a1FixBase(a); }
static uint32_t a1StateDirty(uint32_t a) { return (a1AtRest(a) || a1NearRest(a)) ? 0u : (a * 7u + 3u) % 4u; }
static uint32_t a1DirtySlabs(uint32_t a) { return a % 5u; }
// the record index is a PERMUTATION of the fleet slot (gcd(53, 130) = 1)
static uint32_t a1ArtiIndex(uint32_t a) { return (a * 53u + 11u) % A1_ARTS; }

static float a1V(uint32_t a, uint32_t f)
{
    return (float((a * 31u + f * 13u) % 421u) + 1.0f) * 0.015625f - 3.25f;
}

static uint32_t a1TotalDofs(uint32_t a)
{
    uint32_t t = 0;
    for (uint32_t i = 1; i < a1NumLinks(a); ++i) t += a1Dofs(a1JointType(a, i));
    return t;
}

// ---------------------------------------------------------------------------
// the fill: everything the four sections read, seeded; the rest poison
// ---------------------------------------------------------------------------
static void a1Fill(physx::PxgArticulationBlockData* ab, physx::PxgArticulationBlockLinkData* lk,
                   physx::PxgArticulationBlockDofData* df, uint32_t a, uint32_t lane)
{
    const uint32_t numLinks = a1NumLinks(a);
    const bool rest = a1AtRest(a);
    const bool nearRest = a1NearRest(a);
    auto v = [&](uint32_t i, uint32_t f) { return a1V(a, i * 61u + f); };
    auto unitQ = [&](uint32_t i, uint32_t f) {
        PxQuat q(v(i, f), v(i, f + 1), v(i, f + 2), v(i, f + 3) + 2.5f);
        return q.getNormalized();
    };
    auto unitV = [&](uint32_t i, uint32_t f) {
        PxVec3 x(v(i, f), v(i, f + 1), v(i, f + 2) + 1.5f);
        return x.getNormalized();
    };
    auto sv = [&](uint32_t i, uint32_t f) {
        return UnAlignedSpatialVector(PxVec3(v(i, f), v(i, f + 1), v(i, f + 2)),
                                      PxVec3(v(i, f + 3), v(i, f + 4), v(i, f + 5)));
    };
    auto svScaled = [&](uint32_t i, uint32_t f, float s) {
        return UnAlignedSpatialVector(PxVec3(v(i, f) * s, v(i, f + 1) * s, v(i, f + 2) * s),
                                      PxVec3(v(i, f + 3) * s, v(i, f + 4) * s, v(i, f + 5) * s));
    };
    // NEGATIVE zeros, deliberately: the at-rest articulation is where every momentum
    // accumulator's first term is a signed zero, and where slangc's `x + 0.0` fold
    // (npArtiSpatialInertia.slang) would leave a -0.0 the engine's real add turns
    // into +0.0.  The mirror performs the adds, so it expects the IEEE answer.
    const UnAlignedSpatialVector zeroSV(PxVec3(-0.f, -0.f, -0.f), PxVec3(-0.f, -0.f, -0.f));

    ab->mNumLinks[lane] = numLinks;
    ab->mFlags[lane] = a1FixBase(a) ? 1u : 0u;
    ab->mStateDirty[lane] = uint8_t(a1StateDirty(a));
    ab->mTotalDofs[lane] = a1TotalDofs(a);
    ab->mArticulationIndex[lane] = a1ArtiIndex(a);
    ab->mLinkWithDeferredImpulse[lane] = ((a % 4u) == 2u) ? (numLinks - 1u) : 0u;
    storeSpatialVector(ab->mRootDeferredZ, sv(30, 0) * 0.125f, lane);
    storeSpatialVector(ab->mCommonLinkDeltaVelocity, sv(30, 6), lane);
    // the P8.94d magnitude discipline: response terms ~0.03, impulses ~0.05
    physx::Dy::SpatialMatrix inv;
    inv.topLeft = PxMat33(PxVec3(v(31, 0), v(31, 1), v(31, 2)), PxVec3(v(31, 3), v(31, 4), v(31, 5)),
                          PxVec3(v(31, 6), v(31, 7), v(31, 8))) * 0.03125f;
    inv.topRight = PxMat33(PxVec3(v(32, 0), v(32, 1), v(32, 2)), PxVec3(v(32, 3), v(32, 4), v(32, 5)),
                           PxVec3(v(32, 6), v(32, 7), v(32, 8))) * 0.03125f;
    inv.bottomLeft = PxMat33(PxVec3(v(33, 0), v(33, 1), v(33, 2)), PxVec3(v(33, 3), v(33, 4), v(33, 5)),
                             PxVec3(v(33, 6), v(33, 7), v(33, 8))) * 0.03125f;
    storeSpatialMatrix(ab->mInvSpatialArticulatedInertia, lane, inv);

    float totalMass = 0.f;
    PxVec3 comAccum(0.f);
    uint32_t cursor = 0;
    for (uint32_t i = 0; i < numLinks; ++i)
    {
        physx::PxgArticulationBlockLinkData& L = lk[i];
        const uint8_t jt = (i == 0) ? uint8_t(J_FIX) : a1JointType(a, i);
        const uint32_t dofs = (i == 0) ? 0u : a1Dofs(jt);
        L.mJointType[lane] = jt;
        L.mDofs[lane] = uint8_t(dofs);
        L.mParents[lane] = (i == 0) ? 0u : ((i >= 3u && (i % 3u) == 0u) ? 1u : i - 1u);
        L.mJointOffset[lane] = cursor;
        L.mRw_x[lane] = v(i, 3);
        L.mRw_y[lane] = v(i, 4);
        L.mRw_z[lane] = v(i, 5);
        storeSpatialVector(L.mMotionVelocity, rest ? zeroSV : (nearRest ? svScaled(i, 6, A1_TINY) : sv(i, 6)), lane);
        storeSpatialVector(L.mPosMotionVelocity, rest ? zeroSV : (nearRest ? svScaled(i, 12, A1_TINY) : sv(i, 12)), lane);
        storeSpatialVector(L.mDeltaMotion, sv(i, 18), lane);
        storeSpatialVector(L.mScratchImpulse, sv(i, 24), lane);
        storeSpatialVector(L.mSolverSpatialImpulse, sv(i, 30), lane);
        storeSpatialVector(L.mSolverSpatialDeltaVel, sv(i, 36), lane);
        storeSpatialVector(L.mScratchDeltaV, sv(i, 42), lane);
        storeSpatialVector(L.mConstraintForces, sv(i, 48), lane);
        const PxTransform pre(PxVec3(v(i, 54), v(i, 55), v(i, 56)), unitQ(i, 57));
        const PxTransform acc(PxVec3(v(i, 20), v(i, 21), v(i, 22)), unitQ(i, 23));
        storeSpatialTransform(L.mPreTransform, lane, pre);
        storeSpatialTransform(L.mAccumulatedPose, lane, acc);
        storeSpatialTransform(L.mChildPose, lane, PxTransform(PxVec3(v(i, 27), v(i, 28), v(i, 29)), unitQ(i, 44)));
        storeSpatialTransform(L.mParentPose, lane, PxTransform(PxVec3(v(i, 33), v(i, 34), v(i, 35)), unitQ(i, 48)));
        const PxQuat rq = unitQ(i, 39);
        { float4 f; f.x = rq.x; f.y = rq.y; f.z = rq.z; f.w = rq.w; L.mRelativeQuat[lane] = f; }

        // the momentum inputs: a positive mass, a symmetric positive isolated inertia,
        // a positive inverse-inertia diagonal (no zero: 1/0 is the engine's +inf and
        // this fixture wants finite bits to compare)
        const float mass = 0.5f + std::fabs(v(i, 62)) * 0.75f;
        L.mMass[lane] = mass;
        totalMass += mass;
        comAccum += acc.p * mass;
        {
            const float d0 = 0.6f + std::fabs(v(i, 63)), d1 = 0.6f + std::fabs(v(i, 64)), d2 = 0.6f + std::fabs(v(i, 65));
            const float o01 = v(i, 66) * 0.05f, o02 = v(i, 67) * 0.05f, o12 = v(i, 68) * 0.05f;
            const PxMat33 I(PxVec3(d0, o01, o02), PxVec3(o01, d1, o12), PxVec3(o02, o12, d2));
            storePxMat33(L.mIsolatedInertia, lane, I);
        }
        { float4 f; f.x = 0.25f + std::fabs(v(i, 69)) * 0.5f; f.y = 0.25f + std::fabs(v(i, 70)) * 0.5f;
          f.z = 0.25f + std::fabs(v(i, 71)) * 0.5f; f.w = 1.0f / mass; L.mInvInertiaXYZ_invMassW[lane] = f; }

        for (uint32_t dd = 0; dd < dofs; ++dd)
        {
            physx::PxgArticulationBlockDofData& D = df[cursor + dd];
            const uint32_t u = (cursor + dd) * 7u;
            float jp = v(10 + dd, u);
            // revolute positions beyond +-2pi on every third such joint: the WRAP
            if (jt == J_REVOLUTE && ((a + i) % 3u) == 0u) jp += (jp < 0.f) ? -7.0f : 7.0f;
            D.mJointPositions[lane] = jp;
            // near rest scales the JOINT velocities too: conserveMomentum rebuilds every
            // link velocity by forward kinematics from the root, so a normal joint
            // velocity would put a normal number back into the momentum sum and the
            // magnitude would not underflow.
            D.mJointVelocities[lane] = rest ? 0.f : (v(11 + dd, u) * 2.0f) * (nearRest ? A1_TINY : 1.0f);
            D.mPosJointVelocities[lane] = rest ? 0.f : (v(12 + dd, u) * 0.5f - 1.25f) * (nearRest ? A1_TINY : 1.0f);
            D.mJointUnconstrainedVelocities[lane] = rest ? 0.f : v(22 + dd, u) * 1.5f;
            D.mJointAccel[lane] = v(23 + dd, u);
            D.mDeferredQstZ[lane] = v(13 + dd, u) * 0.125f;
            D.mInvStIsT_x[lane] = v(14 + dd, u) * 0.03125f;
            D.mInvStIsT_y[lane] = v(15 + dd, u) * 0.03125f;
            D.mInvStIsT_z[lane] = v(16 + dd, u) * 0.03125f;
            storeSpatialVector(D.mLocalMotionMatrix,
                               UnAlignedSpatialVector(unitV(17 + dd, u),
                                                      PxVec3(v(18 + dd, u), v(18 + dd, u + 1), v(18 + dd, u + 2))),
                               lane);
            storeSpatialVector(D.mWorldMotionMatrix, sv(19 + dd, u), lane);
            storeSpatialVector(D.mIsInvDW, sv(20 + dd, u) * 0.03125f, lane);
            storeSpatialVector(D.mIsW, sv(21 + dd, u) * 0.03125f, lane);
            // prismatic motion: LIMITED on two of every three, with limits placed so the
            // position sits below the low limit, above the high one, or between them
            if (jt == J_PRISMATIC)
            {
                const uint32_t k = (a + i) % 3u;
                D.mMotion[lane] = (k == 2u) ? A1_MOTION_FREE : A1_MOTION_LIMITED;
                const uint32_t c = (a + 2u * i) % 3u;
                float2 lim;
                if (c == 0u)      { lim.x = jp + 0.25f; lim.y = jp + 1.0f; }   // below -> clamp up
                else if (c == 1u) { lim.x = jp - 1.0f; lim.y = jp - 0.25f; }   // above -> clamp down
                else              { lim.x = jp - 0.5f; lim.y = jp + 0.5f; }    // inside -> untouched
                D.mConstraintData.mLimits_LowLimitX_highLimitY[lane] = lim;
            }
            else
                D.mMotion[lane] = (dd == 0u) ? A1_MOTION_FREE : A1_MOTION_LOCKED;
        }
        cursor += dofs;
    }
    PXG_CHECK_MSG(cursor <= A1_MAX_DOFS, "articulation %u needs %u dofs", a, cursor);
    PXG_CHECK_MSG(cursor == a1TotalDofs(a), "dof cursor %u vs total %u", cursor, a1TotalDofs(a));

    // the pre-step COM and the total inverse mass, offset a little from the mass-weighted
    // mean so the correction is non-trivial
    { float4 f; const PxVec3 c = comAccum / totalMass + PxVec3(v(34, 0), v(34, 1), v(34, 2)) * 0.05f;
      f.x = c.x; f.y = c.y; f.z = c.z; f.w = 1.0f / totalMass; ab->mCOM_TotalInvMassW[lane] = f; }
}

// ---------------------------------------------------------------------------
// the two new transcriptions
// ---------------------------------------------------------------------------
struct A1Coverage
{
    uint32_t revoluteWraps = 0, prismaticLow = 0, prismaticHigh = 0, prismaticInside = 0, spherical = 0;
    // zeroDenom counts every lane whose angular denominator underflows; zeroDenomLive
    // counts the ones where the vector it scales is NOT zero, which is the only case in
    // which the guard's value (0, not 1) can be observed at all.
    uint32_t zeroDenom = 0, zeroDenomLive = 0;
};

// forwardDynamic2.cu:2452-2512
static void refComputeAndEnforceJointPositions(physx::PxgArticulationBlockLinkData* lk,
                                               physx::PxgArticulationBlockDofData* df, uint32_t linkCount,
                                               uint32_t lane, A1Coverage& cov)
{
    for (uint32_t linkID = 1; linkID < linkCount; ++linkID)
    {
        physx::PxgArticulationBlockLinkData& link = lk[linkID];
        const uint8_t jointType = link.mJointType[lane];
        const uint8_t jointOffset = uint8_t(link.mJointOffset[lane]); // :2463, the PxU8 local
        physx::PxgArticulationBlockDofData* dof = &df[jointOffset];
        if (jointType == J_REVOLUTE)
        {
            PxReal jPos = dof->mJointPositions[lane];
            if (jPos > physx::PxTwoPi) { jPos -= physx::PxTwoPi * 2.f; ++cov.revoluteWraps; }
            else if (jPos < -physx::PxTwoPi) { jPos += physx::PxTwoPi * 2.f; ++cov.revoluteWraps; }
            jPos = physx::PxClamp(jPos, -physx::PxTwoPi * 2.f, physx::PxTwoPi * 2.f);
            dof->mJointPositions[lane] = jPos;
        }
        else if (jointType == J_PRISMATIC)
        {
            // 5.12 removed the limit clamp: the positions outside the limits must
            // come back untouched (the coverage counts still classify them)
            if (dof->mMotion[lane] == A1_MOTION_LIMITED)
            {
                const float2 lim = dof->mConstraintData.mLimits_LowLimitX_highLimitY[lane];
                const PxReal jPosition = dof->mJointPositions[lane];
                if (jPosition < lim.x) ++cov.prismaticLow;
                else if (jPosition > lim.y) ++cov.prismaticHigh;
                else ++cov.prismaticInside;
            }
        }
        else if (jointType == J_SPHERICAL)
        {
            const PxU32 parent = link.mParents[lane];
            const float4 pRot = lk[parent].mAccumulatedPose.q[lane];
            const float4 newRot = link.mAccumulatedPose.q[lane];
            const float4 rq = link.mRelativeQuat[lane];
            pxgref::refSphericalJointPositions(PxQuat(rq.x, rq.y, rq.z, rq.w), PxQuat(newRot.x, newRot.y, newRot.z, newRot.w),
                                               PxQuat(pRot.x, pRot.y, pRot.z, pRot.w), dof, link.mDofs[lane], lane);
            ++cov.spherical;
        }
    }
}

// forwardDynamic2.cu:2688-2704 + :2673-2685
static void refUpdatePoses(physx::PxgArticulationBlockLinkData* lk, physx::PxgArticulationBlockDofData* df,
                           uint32_t linkCount, bool fixBase, float dt, uint32_t lane)
{
    if (!fixBase)
    {
        const UnAlignedSpatialVector posVel = loadSpatialVector(lk[0].mPosMotionVelocity, lane);
        const PxTransform preTrans = loadSpatialTransform(lk[0].mAccumulatedPose, lane);
        const PxTransform trans = pxgref::refUpdateRootBody(posVel, preTrans, dt);
        storeSpatialTransform(lk[0].mAccumulatedPose, lane, trans);
    }
    for (uint32_t linkID = 1; linkID < linkCount; ++linkID)
    {
        const PxTransform body2World = pxgref::refPropagateLink(lk, df, linkID, dt, /*isSubstep*/ false, lane);
        storeSpatialTransform(lk[linkID].mAccumulatedPose, lane, body2World);
    }
}

// forwardDynamic2.cu:1569-1577
static PxMat33 refTranslateInertiaMass(const PxMat33& inertia, PxReal mass, const PxVec3& t)
{
    PxMat33 s(PxVec3(0, t.z, -t.y), PxVec3(-t.z, 0, t.x), PxVec3(t.y, -t.x, 0));
    PxMat33 translatedIT = s.getTranspose() * s * mass + inertia;
    return translatedIT;
}

// forwardDynamic2.cu:2783-2981
static void refConserveMomentum(physx::PxgArticulationBlockData* ab, physx::PxgArticulationBlockLinkData* lk,
                                physx::PxgArticulationBlockDofData* df, uint32_t linkCount, bool fixBase,
                                uint32_t lane, float dt, bool doIntegrate, A1Coverage& cov)
{
    if (fixBase) return;

    const float4 COM_invMassW = ab->mCOM_TotalInvMassW[lane];
    const PxVec3 preCOM(COM_invMassW.x, COM_invMassW.y, COM_invMassW.z);
    PxVec3 postCOM(0.f);
    UnAlignedSpatialVector preMomentum(PxVec3(0.f), PxVec3(0.f));
    UnAlignedSpatialVector postMomentum(PxVec3(0.f), PxVec3(0.f));
    PxVec3 posMomentum(0.f);

    for (uint32_t linkID = 0; linkID < linkCount; ++linkID)
    {
        const PxReal mass = lk[linkID].mMass[lane];
        preMomentum.top += loadSpatialVector(lk[linkID].mMotionVelocity, lane).bottom * mass;
        posMomentum += loadSpatialVector(lk[linkID].mPosMotionVelocity, lane).bottom * mass;
    }

    PxReal mass = lk[0].mMass[lane];
    const PxVec3 comPreLinVel = preMomentum.top * COM_invMassW.w;
    PxMat33 inertia;
    loadPxMat33(lk[0].mIsolatedInertia, lane, inertia);
    UnAlignedSpatialVector vel = loadSpatialVector(lk[0].mMotionVelocity, lane);
    preMomentum.bottom = inertia * vel.top
                       + (loadPxVec3(lk[0].mPreTransform.p, lane) - preCOM).cross(vel.bottom - comPreLinVel) * mass;
    postMomentum.top = vel.bottom * mass;
    postCOM += loadPxVec3(lk[0].mAccumulatedPose.p, lane) * mass;

    PxU32 nextParent = (linkCount > 1) ? lk[1].mParents[lane] : 0u;
    for (uint32_t linkID = 1; linkID < linkCount; ++linkID)
    {
        const PxU32 parent = nextParent;
        if (linkID != linkCount - 1) nextParent = lk[linkID + 1].mParents[lane];

        const PxTransform childPose = loadSpatialTransform(lk[linkID].mAccumulatedPose, lane);
        const PxVec3 parentPos = loadPxVec3(lk[parent].mAccumulatedPose.p, lane);
        mass = lk[linkID].mMass[lane];
        const PxVec3 rw = childPose.p - parentPos;
        const PxU32 dofs = lk[linkID].mDofs[lane];
        const PxU32 jointOffset = lk[linkID].mJointOffset[lane];

        loadPxMat33(lk[linkID].mIsolatedInertia, lane, inertia);
        UnAlignedSpatialVector lvel = loadSpatialVector(lk[linkID].mMotionVelocity, lane);
        preMomentum.bottom += inertia * lvel.top
                            + (loadPxVec3(lk[linkID].mPreTransform.p, lane) - preCOM).cross(lvel.bottom - comPreLinVel) * mass;

        // FeatherstoneArticulation::translateSpatialVector(-rw, parentVel)
        const UnAlignedSpatialVector pv = loadSpatialVector(lk[parent].mMotionVelocity, lane);
        lvel = UnAlignedSpatialVector(pv.top, pv.bottom + (-rw).cross(pv.top));
        for (PxU32 ind = 0; ind < 3; ++ind)
        {
            if (ind < dofs)
            {
                const PxReal jVel = df[jointOffset + ind].mJointVelocities[lane];
                lvel += loadSpatialVector(df[jointOffset + ind].mLocalMotionMatrix, lane).rotate(childPose) * jVel;
            }
        }
        postCOM += childPose.p * mass;
        storeSpatialVector(lk[linkID].mMotionVelocity, lvel, lane);
        postMomentum.top += lvel.bottom * mass;
    }
    postCOM *= COM_invMassW.w;
    const PxVec3 comPostLinVel = postMomentum.top * COM_invMassW.w;

    PxMat33 compoundInertia(physx::PxZero);
    float4 nextInvInertiaDiag = lk[0].mInvInertiaXYZ_invMassW[lane];
    PxReal nextMass = lk[0].mMass[lane];
    PxTransform nextPose = loadSpatialTransform(lk[0].mAccumulatedPose, lane);
    for (uint32_t linkID = 0; linkID < linkCount; ++linkID)
    {
        const UnAlignedSpatialVector lvel = loadSpatialVector(lk[linkID].mMotionVelocity, lane);
        const PxVec3 inertiaDiag(1.f / nextInvInertiaDiag.x, 1.f / nextInvInertiaDiag.y, 1.f / nextInvInertiaDiag.z);
        const PxReal lmass = nextMass;
        const PxTransform pose = nextPose;
        if (linkID != linkCount - 1)
        {
            nextInvInertiaDiag = lk[linkID + 1].mInvInertiaXYZ_invMassW[lane];
            nextMass = lk[linkID + 1].mMass[lane];
            nextPose = loadSpatialTransform(lk[linkID + 1].mAccumulatedPose, lane);
        }
        const PxVec3 offset = pose.p - postCOM;
        PxMat33 R(pose.q);
        PxMat33 linkInertia;
        physx::Cm::transformInertiaTensor(inertiaDiag, R, linkInertia);
        compoundInertia += refTranslateInertiaMass(linkInertia, lmass, offset);
        postMomentum.bottom += linkInertia * lvel.top + offset.cross(lvel.bottom - comPostLinVel) * lmass;
    }

    const PxMat33 invCompoundInertia = compoundInertia.getInverse();
    const PxReal aDenom = postMomentum.bottom.magnitude();
    if (aDenom == 0.f)
    {
        ++cov.zeroDenom;
        if (postMomentum.bottom.x != 0.f || postMomentum.bottom.y != 0.f || postMomentum.bottom.z != 0.f)
            ++cov.zeroDenomLive;
    }
    const PxReal angRatio = aDenom == 0.f ? 0.f : preMomentum.bottom.magnitude() / aDenom;
    const PxVec3 angMomDelta = postMomentum.bottom * (angRatio - 1.f);
    const PxVec3 angDelta = invCompoundInertia * angMomDelta;

    for (uint32_t linkID = 0; linkID < linkCount; ++linkID)
    {
        const UnAlignedSpatialVector lvel = loadSpatialVector(lk[linkID].mMotionVelocity, lane);
        const PxReal lmass = lk[linkID].mMass[lane];
        const PxVec3 offset = loadPxVec3(lk[linkID].mAccumulatedPose.p, lane) - postCOM;
        const UnAlignedSpatialVector velChange(angDelta, -offset.cross(angDelta));
        storeSpatialVector(lk[linkID].mMotionVelocity, lvel + velChange, lane);
        postMomentum.top += velChange.bottom * lmass;
    }

    const UnAlignedSpatialVector delta(PxVec3(0.f), (preMomentum.top - postMomentum.top) * COM_invMassW.w);
    PxVec3 predictedCOM;
    if (doIntegrate) predictedCOM = preCOM + posMomentum * COM_invMassW.w * dt;
    else             predictedCOM = preCOM + posMomentum * COM_invMassW.w;
    const PxVec3 correction = predictedCOM - postCOM;

    for (uint32_t linkID = 0; linkID < linkCount; ++linkID)
    {
        // addSpatialVector (PxgArticulation.h:214-218), component-wise as cutil_math's
        // float4 / float2 operator+= are: the top xyz get a REAL `+ 0.f`
        float4& t = lk[linkID].mMotionVelocity.mTopxyz_bx[lane];
        t.x += delta.top.x; t.y += delta.top.y; t.z += delta.top.z; t.w += delta.bottom.x;
        float2& b = lk[linkID].mMotionVelocity.mbyz[lane];
        b.x += delta.bottom.y; b.y += delta.bottom.z;
        float4& p = lk[linkID].mAccumulatedPose.p[lane];
        p.x += correction.x; p.y += correction.y; p.z += correction.z; p.w += 0.f;
    }
}

// the record arrays of ONE articulation: three float arrays of A1_MAX_DOFS + 1 (canary)
struct A1Record
{
    std::vector<float> jointP, jointV, jointA;
};

// updateBodiesInternal (forwardDynamic2.cu:2984-3045)
static void a1Mirror(uint32_t* isSlabDirty, float* impulses, physx::PxgArticulationBlockData* ab,
                     physx::PxgArticulationBlockLinkData* lk, physx::PxgArticulationBlockDofData* df,
                     A1Record& rec, uint32_t a, float dt, bool integrate, uint32_t lane, A1Coverage& cov,
                     uint32_t& alignedHits, uint32_t& unalignedHits)
{
    const uint32_t linkCount = ab->mNumLinks[lane];
    const bool fixBase = (ab->mFlags[lane] & 1u) != 0u;
    const float invDt = 1.f / dt;

    if (!integrate) refComputeAndEnforceJointPositions(lk, df, linkCount, lane, cov);
    else refUpdatePoses(lk, df, linkCount, fixBase, dt, lane);

    const uint32_t stateDirty = ab->mStateDirty[lane];
    if (stateDirty)
    {
        if (stateDirty & 2u)
            pxgref::aiMirror(isSlabDirty, impulses, ab, lk, df, a, A1_MAX_LINKS, A1_ARTS, A1_NB_SLABS, lane,
                             alignedHits, unalignedHits);
        pxgref::fvMirror(ab, lk, df, linkCount, fixBase, lane);
        ab->mStateDirty[lane] = 0;
    }

    refConserveMomentum(ab, lk, df, linkCount, fixBase, lane, dt, integrate, cov);

    const uint32_t totalDofs = ab->mTotalDofs[lane];
    for (uint32_t i = 0; i < totalDofs; ++i)
    {
        rec.jointP[i] = df[i].mJointPositions[lane];
        rec.jointV[i] = df[i].mJointVelocities[lane];
        rec.jointA[i] = df[i].mJointAccel[lane]
                      + (df[i].mJointVelocities[lane] - df[i].mJointUnconstrainedVelocities[lane]) * invDt;
    }
}

// accumulateExternalZAcrossChainW (forwardDynamic2.cu:2993, new in 5.12): the
// leaves-to-root sweep updateBodiesLaunch1T runs under
// isExternalForcesEveryTgsIterationEnabled, over the record's zAForces, AFTER
// updateBodiesInternal.  propagateImpulseW<WriteToProvidedQstZ> with
// jointForce == NULL and providedQstZ == NULL: per dof
// `stZ = 0.0f - sas.innerProduct(Z)` (jf is the 0.0f literal), nothing written
// back to the dofs; zAForces[parent] += translateSpatialVector(rw, temp).
static void refAccumulateExternalZAcrossChainW(UnAlignedSpatialVector* zAForces,
                                               const physx::PxgArticulationBlockLinkData* lk,
                                               const physx::PxgArticulationBlockDofData* df,
                                               uint32_t numLinks, uint32_t lane)
{
    for (uint32_t linkID = numLinks - 1u; linkID > 0u; --linkID)
    {
        const physx::PxgArticulationBlockLinkData& L = lk[linkID];
        const uint32_t parent = L.mParents[lane];
        const PxVec3 rw(L.mRw_x[lane], L.mRw_y[lane], L.mRw_z[lane]);
        const uint32_t dofCount = L.mDofs[lane];
        const physx::PxgArticulationBlockDofData* dofData = &df[L.mJointOffset[lane]];

        const UnAlignedSpatialVector Z = zAForces[linkID];
        UnAlignedSpatialVector temp = Z;
        for (uint32_t ind = 0; ind < 3; ++ind)
        {
            if (ind < dofCount)
            {
                const UnAlignedSpatialVector sas = loadSpatialVector(dofData[ind].mWorldMotionMatrix, lane);
                const UnAlignedSpatialVector isInvD = loadSpatialVector(dofData[ind].mIsInvDW, lane);
                const PxReal jf = 0.0f;
                const PxReal stZ = jf - sas.innerProduct(Z);
                temp += isInvD * stZ;
            }
        }
        zAForces[parent] += physx::Dy::FeatherstoneArticulation::translateSpatialVector(rw, temp);
    }
}

// ---------------------------------------------------------------------------
// one dispatch per `integrate` value; extEvery (5.12) adds the external-Z chain
// sweep into each record's zAForces
// ---------------------------------------------------------------------------
static void a1Run(PxgCudaShim& shim, bool integrate, bool extEvery = false)
{
    std::vector<physx::PxgArticulationBlockData> abImg(A1_WARPS);
    std::vector<physx::PxgArticulationBlockLinkData> lkImg(size_t(A1_WARPS) * A1_MAX_LINKS);
    std::vector<physx::PxgArticulationBlockDofData> dfImg(size_t(A1_WARPS) * A1_MAX_DOFS);
    std::memset(abImg.data(), 0xAB, abImg.size() * sizeof(physx::PxgArticulationBlockData));
    std::memset(lkImg.data(), 0xAB, lkImg.size() * sizeof(physx::PxgArticulationBlockLinkData));
    std::memset(dfImg.data(), 0xAB, dfImg.size() * sizeof(physx::PxgArticulationBlockDofData));

    std::vector<uint32_t> dirty(size_t(A1_NB_SLABS) * A1_ARTS * 2, AI_NONE);
    std::vector<float> imp(size_t(A1_NB_SLABS) * A1_MAX_LINKS * A1_ARTS * 6);
    for (size_t k = 0; k < imp.size(); ++k)
        imp[k] = a1V(uint32_t(k % 997u), uint32_t(k / 997u)) * 0.015625f;

    for (uint32_t a = 0; a < A1_ARTS; ++a)
    {
        const uint32_t w = a / 32u, lane = a & 31u;
        a1Fill(&abImg[w], &lkImg[size_t(w) * A1_MAX_LINKS], &dfImg[size_t(w) * A1_MAX_DOFS], a, lane);
        const uint32_t nd = a1DirtySlabs(a);
        const uint32_t nl = a1NumLinks(a);
        for (uint32_t s = 0; s < nd && s < A1_NB_SLABS; ++s)
        {
            const size_t di = size_t(s) * A1_ARTS + a;
            const uint32_t mode = (a + s) % 3u;
            const uint32_t x = 1u + ((a + s) % (nl - 1u));
            const uint32_t y = 1u + ((a + s + 1u) % (nl - 1u));
            if (mode != 1u) dirty[di * 2 + 0] = x;
            if (mode != 0u) dirty[di * 2 + 1] = y;
        }
    }

    // records: poisoned PxgArticulation each, the three joint arrays planted, one
    // canary float past the last dof
    std::vector<physx::PxgArticulation> arts(A1_ARTS);
    std::memset(arts.data(), 0xAB, arts.size() * sizeof(physx::PxgArticulation));
    std::vector<A1Record> recs(A1_ARTS);
    // The record canary must NOT be the dof blocks' 0xAB poison.  It was, and that
    // made an out-of-range writeback invisible: the mutant that runs the dof loop one
    // iteration too far reads dof block [totalDofs] -- inside the allocation, never
    // seeded, so still 0xABABABAB -- and stores it into jointP/V/A[totalDofs], where
    // the expectation held the SAME bit pattern.  (The acceleration is
    // P + (P - P) * invDt, and P - P is +0, so it comes back as P bit for bit too.)
    // A canary a poisoned dof block cannot produce closes that.
    const float poisonF = -8675309.5f;
    for (uint32_t r = 0; r < A1_ARTS; ++r)
    {
        recs[r].jointP.assign(A1_MAX_DOFS + 1, poisonF);
        recs[r].jointV.assign(A1_MAX_DOFS + 1, poisonF);
        recs[r].jointA.assign(A1_MAX_DOFS + 1, poisonF);
    }

    // ---- the expectation ----
    std::vector<uint32_t> wantDirty = dirty;
    std::vector<float> wantImp = imp;
    std::vector<physx::PxgArticulationBlockData> wantAb = abImg;
    std::vector<physx::PxgArticulationBlockLinkData> wantLk = lkImg;
    std::vector<physx::PxgArticulationBlockDofData> wantDf = dfImg;
    std::vector<A1Record> wantRecs = recs;
    // 5.12: each record's zAForces (the isolated external Z per link), seeded;
    // one canary spatial vector past the last link
    std::vector<std::vector<UnAlignedSpatialVector>> zaf(A1_ARTS);
    for (uint32_t r = 0; r < A1_ARTS; ++r)
    {
        zaf[r].resize(A1_MAX_LINKS + 1);
        for (uint32_t l = 0; l <= A1_MAX_LINKS; ++l)
            zaf[r][l] = UnAlignedSpatialVector(PxVec3(a1V(r, 600u + 6u * l), a1V(r, 601u + 6u * l),
                                                      a1V(r, 602u + 6u * l)),
                                               PxVec3(a1V(r, 603u + 6u * l), a1V(r, 604u + 6u * l),
                                                      a1V(r, 605u + 6u * l)) * 0.5f);
    }
    std::vector<std::vector<UnAlignedSpatialVector>> wantZaf = zaf;
    A1Coverage cov;
    uint32_t alignedHits = 0, unalignedHits = 0, chainHops = 0;
    for (uint32_t a = 0; a < A1_ARTS; ++a)
    {
        a1Mirror(wantDirty.data(), wantImp.data(), &wantAb[a / 32u], &wantLk[size_t(a / 32u) * A1_MAX_LINKS],
                 &wantDf[size_t(a / 32u) * A1_MAX_DOFS], wantRecs[a1ArtiIndex(a)], a, A1_DT, integrate, a & 31u, cov,
                 alignedHits, unalignedHits);
        if (extEvery)
        {
            // updateBodiesLaunch1T: the record by the BLOCK's mArticulationIndex,
            // the link count by its mNumLinks, after updateBodiesInternal
            const uint32_t lane = a & 31u;
            const physx::PxgArticulationBlockData& B = wantAb[a / 32u];
            refAccumulateExternalZAcrossChainW(wantZaf[B.mArticulationIndex[lane]].data(),
                                               &wantLk[size_t(a / 32u) * A1_MAX_LINKS],
                                               &wantDf[size_t(a / 32u) * A1_MAX_DOFS], B.mNumLinks[lane], lane);
            chainHops += B.mNumLinks[lane] - 1u;
        }
    }
    if (extEvery)
        PXG_CHECK_MSG(chainHops > 100, "the external-Z chain sweep runs (%u hops)", chainHops);
    std::printf("[ info ] au1 int=%d: impulse reads 16-aligned %u, 8-aligned %u; revolute wraps %u, prismatic "
                "clamps low %u / high %u / inside %u, spherical re-derivations %u, zero-denominator %u "
                "(of which %u with a non-zero momentum vector)\n",
                int(integrate), alignedHits, unalignedHits, cov.revoluteWraps, cov.prismaticLow, cov.prismaticHigh,
                cov.prismaticInside, cov.spherical, cov.zeroDenom, cov.zeroDenomLive);
    PXG_CHECK_MSG(alignedHits > 5 && unalignedHits > 5, "both impulse alignments occur (%u / %u)", alignedHits,
                  unalignedHits);
    if (!integrate)
    {
        PXG_CHECK_MSG(cov.revoluteWraps > 3, "the revolute wrap fires (%u)", cov.revoluteWraps);
        PXG_CHECK_MSG(cov.prismaticLow > 3 && cov.prismaticHigh > 3 && cov.prismaticInside > 3,
                      "both prismatic clamps and the inside case occur (%u / %u / %u)", cov.prismaticLow,
                      cov.prismaticHigh, cov.prismaticInside);
        PXG_CHECK_MSG(cov.spherical > 10, "spherical joints are re-derived (%u)", cov.spherical);
    }
    PXG_CHECK_MSG(cov.zeroDenom > 0, "the at-rest articulations reach aDenom == 0 (%u)", cov.zeroDenom);
    PXG_CHECK_MSG(cov.zeroDenomLive > 0,
                  "the NEAR-rest articulations reach aDenom == 0 with a non-zero momentum vector (%u) -- "
                  "without that the guard's value is unobservable, because it scales a zero", cov.zeroDenomLive);

    // ---- device ----
    PxgDevicePtr dAb = 0, dLk = 0, dDf = 0, dDirty = 0, dImp = 0, dArts = 0, dDesc = 0;
    shim.memAlloc(&dAb, abImg.size() * sizeof(physx::PxgArticulationBlockData));
    shim.memAlloc(&dLk, lkImg.size() * sizeof(physx::PxgArticulationBlockLinkData));
    shim.memAlloc(&dDf, dfImg.size() * sizeof(physx::PxgArticulationBlockDofData));
    shim.memAlloc(&dDirty, dirty.size() * sizeof(uint32_t));
    shim.memAlloc(&dImp, imp.size() * sizeof(float));
    shim.memAlloc(&dArts, arts.size() * sizeof(physx::PxgArticulation));
    shim.memAlloc(&dDesc, sizeof(physx::PxgArticulationCoreDesc));
    if (!dAb || !dLk || !dDf || !dDirty || !dImp || !dArts || !dDesc)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    struct RecPtrs { PxgDevicePtr p, v, a, za; };
    std::vector<RecPtrs> rp(A1_ARTS);
    auto up = [&](const std::vector<float>& src) {
        PxgDevicePtr dp = 0;
        shim.memAlloc(&dp, src.size() * 4);
        if (dp) shim.memcpyHtoD(dp, src.data(), src.size() * 4);
        return dp;
    };
    for (uint32_t r = 0; r < A1_ARTS; ++r)
    {
        rp[r].p = up(recs[r].jointP);
        rp[r].v = up(recs[r].jointV);
        rp[r].a = up(recs[r].jointA);
        arts[r].jointPositions = reinterpret_cast<PxReal*>(rp[r].p);
        arts[r].jointVelocities = reinterpret_cast<PxReal*>(rp[r].v);
        arts[r].jointAccelerations = reinterpret_cast<PxReal*>(rp[r].a);
        rp[r].za = 0;
        shim.memAlloc(&rp[r].za, zaf[r].size() * sizeof(UnAlignedSpatialVector));
        if (rp[r].za)
            shim.memcpyHtoD(rp[r].za, zaf[r].data(), zaf[r].size() * sizeof(UnAlignedSpatialVector));
        arts[r].zAForces = reinterpret_cast<UnAlignedSpatialVector*>(rp[r].za);
    }
    shim.memcpyHtoD(dAb, abImg.data(), abImg.size() * sizeof(physx::PxgArticulationBlockData));
    shim.memcpyHtoD(dLk, lkImg.data(), lkImg.size() * sizeof(physx::PxgArticulationBlockLinkData));
    shim.memcpyHtoD(dDf, dfImg.data(), dfImg.size() * sizeof(physx::PxgArticulationBlockDofData));
    shim.memcpyHtoD(dDirty, dirty.data(), dirty.size() * sizeof(uint32_t));
    shim.memcpyHtoD(dImp, imp.data(), imp.size() * sizeof(float));
    shim.memcpyHtoD(dArts, arts.data(), arts.size() * sizeof(physx::PxgArticulation));

    physx::PxgArticulationCoreDesc desc;
    std::memset(&desc, 0xAB, sizeof(desc));
    desc.nbArticulations = A1_ARTS;
    desc.mMaxLinksPerArticulation = A1_MAX_LINKS;
    desc.mMaxDofsPerArticulation = A1_MAX_DOFS;
    desc.nbSlabs = A1_NB_SLABS;
    desc.mArticulationBlocks = reinterpret_cast<physx::PxgArticulationBlockData*>(dAb);
    desc.mArticulationLinkBlocks = reinterpret_cast<physx::PxgArticulationBlockLinkData*>(dLk);
    desc.mArticulationDofBlocks = reinterpret_cast<physx::PxgArticulationBlockDofData*>(dDf);
    desc.slabHasChanges = reinterpret_cast<uint2*>(dDirty);
    desc.impulses = reinterpret_cast<UnAlignedSpatialVector*>(dImp);
    desc.articulations = reinterpret_cast<physx::PxgArticulation*>(dArts);
    // 5.12: the external-Z chain sweep runs only under this flag (poison reads true)
    desc.isExternalForcesEveryTgsIterationEnabled = extEvery;
    desc.isSleepingDisabled = false;
    shim.memcpyHtoD(dDesc, &desc, sizeof(desc));

    // the bool in ONE byte at offset 12, its padding poisoned (the P8.95b pattern)
    struct Params { uint64_t desc; float dt; uint32_t integrateWord; } prm = {
        uint64_t(dDesc), A1_DT, integrate ? 0xA7A5A301u : 0xB6B4B200u };
    PXG_CHECK_MSG(shim.launchKernel(0, 3, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch updateBodiesLaunch1T (grid 3, block 32x2, integrate=%d)", int(integrate));
    shim.streamSynchronize();

    // ---- compare ----
    std::vector<physx::PxgArticulationBlockData> gotAb(abImg.size());
    std::vector<physx::PxgArticulationBlockLinkData> gotLk(lkImg.size());
    std::vector<physx::PxgArticulationBlockDofData> gotDf(dfImg.size());
    std::vector<uint32_t> gotDirty(dirty.size());
    std::vector<float> gotImp(imp.size());
    std::vector<physx::PxgArticulation> gotArts(arts.size());
    shim.memcpyDtoH(gotAb.data(), dAb, gotAb.size() * sizeof(physx::PxgArticulationBlockData));
    shim.memcpyDtoH(gotLk.data(), dLk, gotLk.size() * sizeof(physx::PxgArticulationBlockLinkData));
    shim.memcpyDtoH(gotDf.data(), dDf, gotDf.size() * sizeof(physx::PxgArticulationBlockDofData));
    shim.memcpyDtoH(gotDirty.data(), dDirty, gotDirty.size() * sizeof(uint32_t));
    shim.memcpyDtoH(gotImp.data(), dImp, gotImp.size() * sizeof(float));
    shim.memcpyDtoH(gotArts.data(), dArts, gotArts.size() * sizeof(physx::PxgArticulation));

    size_t bad = 0, signedZeros = 0, tolerated = 0, loud = 0;
    uint32_t maxUlp = 0;
    float maxAbs = 0.f;
    const size_t linkWords = sizeof(physx::PxgArticulationBlockLinkData) / 4;
    const size_t dofWords = sizeof(physx::PxgArticulationBlockDofData) / 4;
    auto inRange = [](size_t w, size_t structWords, size_t off, size_t bytes) {
        const size_t k = w % structWords;
        return k >= off / 4 && k < (off + bytes) / 4;
    };
    // kind: 0 exact; 1 link block; 2 dof block; 3 record jointPositions
    auto cmp = [&](const char* what, const void* wp, const void* gp, size_t bytes, int kind) {
        const uint32_t* w = reinterpret_cast<const uint32_t*>(wp);
        const uint32_t* g = reinterpret_cast<const uint32_t*>(gp);
        for (size_t k = 0; k < bytes / 4; ++k)
        {
            if (w[k] == g[k]) continue;
            bool tol = false;
            if (kind == 1 && integrate)
                tol = inRange(k, linkWords, offsetof(physx::PxgArticulationBlockLinkData, mAccumulatedPose),
                              sizeof(physx::PxgSpatialTransformBlock))
                   || inRange(k, linkWords, offsetof(physx::PxgArticulationBlockLinkData, mMotionVelocity),
                              sizeof(physx::PxgSpatialVectorBlock));
            else if (kind == 2)
                tol = inRange(k, dofWords, offsetof(physx::PxgArticulationBlockDofData, mJointPositions),
                              sizeof(PxReal) * 32);
            else if (kind == 3)
                tol = true;
            // A signed-zero mismatch is only TOLERATED where the ulp budget applies.
            // It used to be tolerated everywhere, which made every +-0 mutation in this
            // kernel unkillable on the integrate == true path -- including deleting the
            // au1PlusZero canonicalisation the port depends on.  Outside the
            // transcendental scope it is now an ordinary mismatch.
            if ((w[k] == 0u && g[k] == 0x80000000u) || (w[k] == 0x80000000u && g[k] == 0u))
            {
                if (signedZeros < 12)
                {
                    const size_t structWords = kind == 1 ? linkWords : (kind == 2 ? dofWords : 0);
                    std::printf("SZ au1 int=%d %s word %zu (struct %zu, byte %zu) ref %08x gpu %08x%s\n",
                                int(integrate), what, k, structWords ? k / structWords : 0,
                                structWords ? (k % structWords) * 4 : k * 4, w[k], g[k],
                                tol ? "" : "  (outside the tolerance scope: counted as a mismatch)");
                }
                ++signedZeros;
                if (tol) continue;
                ++bad;
                continue;
            }
            if (tol)
            {
                float x, y;
                std::memcpy(&x, &w[k], 4);
                std::memcpy(&y, &g[k], 4);
                const uint32_t u = a1UlpDiff(x, y);
                const float e = std::fabs(x - y);
                maxUlp = maxUlp > u ? maxUlp : u;
                maxAbs = maxAbs > e ? maxAbs : e;
                // the P8.94a/P8.94d budget: ULP-close OR absolutely close
                if (u > 384 && e > 4e-6f)
                {
                    if (loud < 8)
                        std::printf("MM au1 int=%d %s tol word %zu: %u ulp, abs %.3g\n", int(integrate), what, k, u,
                                    double(e));
                    ++loud;
                }
                ++tolerated;
                continue;
            }
            if (bad < 12)
                std::printf("MM au1 int=%d %s word %zu  ref %08x  gpu %08x\n", int(integrate), what, k, w[k], g[k]);
            ++bad;
        }
    };
    cmp("arti", wantAb.data(), gotAb.data(), wantAb.size() * sizeof(physx::PxgArticulationBlockData), 0);
    cmp("link", wantLk.data(), gotLk.data(), wantLk.size() * sizeof(physx::PxgArticulationBlockLinkData), 1);
    cmp("dof", wantDf.data(), gotDf.data(), wantDf.size() * sizeof(physx::PxgArticulationBlockDofData), 2);
    cmp("dirty", wantDirty.data(), gotDirty.data(), wantDirty.size() * sizeof(uint32_t), 0);
    cmp("impulses", wantImp.data(), gotImp.data(), wantImp.size() * sizeof(float), 0);
    cmp("records", arts.data(), gotArts.data(), arts.size() * sizeof(physx::PxgArticulation), 0);
    std::vector<float> tmp(A1_MAX_DOFS + 1);
    for (uint32_t r = 0; r < A1_ARTS; ++r)
    {
        shim.memcpyDtoH(tmp.data(), rp[r].p, tmp.size() * 4);
        cmp("jointPositions", wantRecs[r].jointP.data(), tmp.data(), tmp.size() * 4, 3);
        shim.memcpyDtoH(tmp.data(), rp[r].v, tmp.size() * 4);
        cmp("jointVelocities", wantRecs[r].jointV.data(), tmp.data(), tmp.size() * 4, 0);
        shim.memcpyDtoH(tmp.data(), rp[r].a, tmp.size() * 4);
        cmp("jointAccelerations", wantRecs[r].jointA.data(), tmp.data(), tmp.size() * 4, 0);
        // the record's zAForces: untouched unless extEvery, then the chain (exact)
        std::vector<UnAlignedSpatialVector> gotZa(zaf[r].size());
        shim.memcpyDtoH(gotZa.data(), rp[r].za, gotZa.size() * sizeof(UnAlignedSpatialVector));
        cmp("zAForces", wantZaf[r].data(), gotZa.data(), gotZa.size() * sizeof(UnAlignedSpatialVector), 0);
    }

    std::printf("[ info ] au1 int=%d: signed zeros %zu, tolerated %zu words, max %u ulp, max abs %.3g\n",
                int(integrate), signedZeros, tolerated, maxUlp, double(maxAbs));
    PXG_CHECK_MSG(bad == 0, "integrate=%d: every word outside the transcendental scope is bit-exact (%zu are not)",
                  int(integrate), bad);
    PXG_CHECK_MSG(loud == 0, "integrate=%d: %zu words are outside both the ulp and the absolute bound",
                  int(integrate), loud);
    PXG_CHECK_MSG(maxAbs <= 1e-3f, "integrate=%d: the pose drifted grossly: %.3g absolute", int(integrate),
                  double(maxAbs));
    // integrate == false is the TGS path the scenes run: nothing there is a
    // transcendental except the spherical atan2, and the signed-zero class must be
    // empty -- conserveMomentum's accumulators start from a runtime zero for that.
    if (!integrate)
        PXG_CHECK_MSG(signedZeros == 0, "integrate=0: %zu signed-zero mismatches", signedZeros);

    for (uint32_t r = 0; r < A1_ARTS; ++r)
        for (PxgDevicePtr p : { rp[r].p, rp[r].v, rp[r].a, rp[r].za }) if (p) shim.memFree(p);
    for (PxgDevicePtr p : { dAb, dLk, dDf, dDirty, dImp, dArts, dDesc }) shim.memFree(p);
}

// ---------------------------------------------------------------------------
// TWO NARROW DISPATCHES beside the main fleet.  Each closes a blind spot the
// mutation sweep found, and each needs a geometry the fleet cannot have, because
// maxLinks and maxDofs are properties of the DESCRIPTOR, not of one articulation.
//
//  a1RunSingleLink: every articulation has exactly ONE link, so maxLinks is 1 and
//    the link-block allocation is exactly (warps * 1) blocks.  The kernel prefetches
//    link[1]'s parent before its walk, which the CUDA does unconditionally
//    (forwardDynamic2.cu:2821) and this port guards on linkCount > 1.  With maxLinks
//    of 2 or more the unguarded read lands in the warp's own slice -- garbage, but
//    mapped, and the value is dead either way, so no comparison can see it.  With
//    maxLinks == 1 it lands 28 KB PAST the buffer, and the last warp has no slice
//    behind it.  A scene of single-link articulations gives exactly that
//    (PxgSimulationController.cpp:922 takes the fleet maximum).
//
//  a1RunWideDofs: one articulation of 88 links and 259 dofs, so a joint offset
//    exceeds 255.  computeAndEnforceJointPositions reads that offset through a PxU8
//    local (forwardDynamic2.cu:2463) -- the port reproduces the truncation, upstream
//    candidate #44 -- and below 256 dofs the truncation is invisible.  Here link 87's
//    offset is 258, so the truncated read lands on dof 2 and the untruncated one on
//    dof 258: the two write different dofs, and both are compared.
// ---------------------------------------------------------------------------
static const uint32_t A1S_ARTS = 33;   // two warp slices; slice 1 is the last, so its
static const uint32_t A1S_WARPS = 2;   // link[1] read has nothing behind it
static const uint32_t A1S_MAX_LINKS = 1;
static const uint32_t A1S_MAX_DOFS = 1;

static void a1RunSingleLink(PxgCudaShim& shim, bool integrate)
{
    std::vector<physx::PxgArticulationBlockData> abImg(A1S_WARPS);
    std::vector<physx::PxgArticulationBlockLinkData> lkImg(size_t(A1S_WARPS) * A1S_MAX_LINKS);
    std::vector<physx::PxgArticulationBlockDofData> dfImg(size_t(A1S_WARPS) * A1S_MAX_DOFS);
    std::memset(abImg.data(), 0xAB, abImg.size() * sizeof(physx::PxgArticulationBlockData));
    std::memset(lkImg.data(), 0xAB, lkImg.size() * sizeof(physx::PxgArticulationBlockLinkData));
    std::memset(dfImg.data(), 0xAB, dfImg.size() * sizeof(physx::PxgArticulationBlockDofData));

    for (uint32_t a = 0; a < A1S_ARTS; ++a)
    {
        const uint32_t w = a / 32u, lane = a & 31u;
        physx::PxgArticulationBlockData& ab = abImg[w];
        physx::PxgArticulationBlockLinkData& L = lkImg[size_t(w) * A1S_MAX_LINKS];
        auto v = [&](uint32_t i, uint32_t f) { return a1V(a + 7u, i * 61u + f); };
        auto unitQ = [&](uint32_t i, uint32_t f) {
            PxQuat q(v(i, f), v(i, f + 1), v(i, f + 2), v(i, f + 3) + 2.5f);
            return q.getNormalized();
        };
        auto sv = [&](uint32_t i, uint32_t f) {
            return UnAlignedSpatialVector(PxVec3(v(i, f), v(i, f + 1), v(i, f + 2)),
                                          PxVec3(v(i, f + 3), v(i, f + 4), v(i, f + 5)));
        };
        ab.mNumLinks[lane] = 1u;
        ab.mFlags[lane] = 0u;            // FREE base: a fixed one returns before the prefetch
        ab.mStateDirty[lane] = 0u;       // no flush, so the impulse buffers stay out of it
        ab.mTotalDofs[lane] = 0u;
        ab.mArticulationIndex[lane] = a;
        storeSpatialVector(ab.mRootDeferredZ, sv(30, 0) * 0.125f, lane);
        storeSpatialVector(ab.mCommonLinkDeltaVelocity, sv(30, 6), lane);

        L.mJointType[lane] = uint8_t(J_FIX);
        L.mDofs[lane] = 0u;
        L.mParents[lane] = 0u;           // the root is its own parent
        L.mJointOffset[lane] = 0u;
        L.mRw_x[lane] = v(0, 3); L.mRw_y[lane] = v(0, 4); L.mRw_z[lane] = v(0, 5);
        storeSpatialVector(L.mMotionVelocity, sv(0, 6), lane);
        storeSpatialVector(L.mPosMotionVelocity, sv(0, 12), lane);
        storeSpatialVector(L.mDeltaMotion, sv(0, 18), lane);
        storeSpatialVector(L.mScratchImpulse, sv(0, 24), lane);
        storeSpatialVector(L.mSolverSpatialImpulse, sv(0, 30), lane);
        storeSpatialVector(L.mSolverSpatialDeltaVel, sv(0, 36), lane);
        storeSpatialVector(L.mScratchDeltaV, sv(0, 42), lane);
        storeSpatialVector(L.mConstraintForces, sv(0, 48), lane);
        const PxTransform acc(PxVec3(v(0, 20), v(0, 21), v(0, 22)), unitQ(0, 23));
        storeSpatialTransform(L.mPreTransform, lane, PxTransform(PxVec3(v(0, 54), v(0, 55), v(0, 56)), unitQ(0, 57)));
        storeSpatialTransform(L.mAccumulatedPose, lane, acc);
        storeSpatialTransform(L.mChildPose, lane, PxTransform(PxVec3(v(0, 27), v(0, 28), v(0, 29)), unitQ(0, 44)));
        storeSpatialTransform(L.mParentPose, lane, PxTransform(PxVec3(v(0, 33), v(0, 34), v(0, 35)), unitQ(0, 48)));
        { const PxQuat rq = unitQ(0, 39); float4 f; f.x = rq.x; f.y = rq.y; f.z = rq.z; f.w = rq.w;
          L.mRelativeQuat[lane] = f; }
        const float mass = 0.5f + std::fabs(v(0, 62)) * 0.75f;
        L.mMass[lane] = mass;
        {
            const float d0 = 0.6f + std::fabs(v(0, 63)), d1 = 0.6f + std::fabs(v(0, 64)),
                        d2 = 0.6f + std::fabs(v(0, 65));
            const float o01 = v(0, 66) * 0.05f, o02 = v(0, 67) * 0.05f, o12 = v(0, 68) * 0.05f;
            storePxMat33(L.mIsolatedInertia, lane,
                         PxMat33(PxVec3(d0, o01, o02), PxVec3(o01, d1, o12), PxVec3(o02, o12, d2)));
        }
        { float4 f; f.x = 0.25f + std::fabs(v(0, 69)) * 0.5f; f.y = 0.25f + std::fabs(v(0, 70)) * 0.5f;
          f.z = 0.25f + std::fabs(v(0, 71)) * 0.5f; f.w = 1.0f / mass; L.mInvInertiaXYZ_invMassW[lane] = f; }
        { float4 f; const PxVec3 c = acc.p + PxVec3(v(34, 0), v(34, 1), v(34, 2)) * 0.05f;
          f.x = c.x; f.y = c.y; f.z = c.z; f.w = 1.0f / mass; ab.mCOM_TotalInvMassW[lane] = f; }
    }

    // the mirror: stateDirty is 0 and there are no dofs, so only the pose section and
    // conserveMomentum run, and a1Mirror's impulse/dirty arguments are never touched
    std::vector<physx::PxgArticulationBlockData> wantAb = abImg;
    std::vector<physx::PxgArticulationBlockLinkData> wantLk = lkImg;
    std::vector<physx::PxgArticulationBlockDofData> wantDf = dfImg;
    std::vector<A1Record> wantRecs(A1S_ARTS);
    for (uint32_t r = 0; r < A1S_ARTS; ++r)
    {
        wantRecs[r].jointP.assign(2, -8675309.5f);
        wantRecs[r].jointV.assign(2, -8675309.5f);
        wantRecs[r].jointA.assign(2, -8675309.5f);
    }
    A1Coverage cov;
    uint32_t alignedHits = 0, unalignedHits = 0;
    std::vector<uint32_t> noDirty(1, AI_NONE);
    std::vector<float> noImp(1, 0.f);
    for (uint32_t a = 0; a < A1S_ARTS; ++a)
        a1Mirror(noDirty.data(), noImp.data(), &wantAb[a / 32u], &wantLk[size_t(a / 32u) * A1S_MAX_LINKS],
                 &wantDf[size_t(a / 32u) * A1S_MAX_DOFS], wantRecs[a], a, A1_DT, integrate, a & 31u, cov,
                 alignedHits, unalignedHits);

    PxgDevicePtr dAb = 0, dLk = 0, dDf = 0, dArts = 0, dDesc = 0;
    shim.memAlloc(&dAb, abImg.size() * sizeof(physx::PxgArticulationBlockData));
    // EXACTLY the two slices: the tight allocation is what puts the prefetch outside
    shim.memAlloc(&dLk, lkImg.size() * sizeof(physx::PxgArticulationBlockLinkData));
    shim.memAlloc(&dDf, dfImg.size() * sizeof(physx::PxgArticulationBlockDofData));
    std::vector<physx::PxgArticulation> arts(A1S_ARTS);
    std::memset(arts.data(), 0xAB, arts.size() * sizeof(physx::PxgArticulation));
    shim.memAlloc(&dArts, arts.size() * sizeof(physx::PxgArticulation));
    shim.memAlloc(&dDesc, sizeof(physx::PxgArticulationCoreDesc));
    if (!dAb || !dLk || !dDf || !dArts || !dDesc) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    struct RecPtrs { PxgDevicePtr p, v, a; };
    std::vector<RecPtrs> rp(A1S_ARTS);
    auto up = [&](const std::vector<float>& src) {
        PxgDevicePtr dp = 0;
        shim.memAlloc(&dp, src.size() * 4);
        if (dp) shim.memcpyHtoD(dp, src.data(), src.size() * 4);
        return dp;
    };
    const std::vector<float> canary(2, -8675309.5f);
    for (uint32_t r = 0; r < A1S_ARTS; ++r)
    {
        rp[r].p = up(canary); rp[r].v = up(canary); rp[r].a = up(canary);
        arts[r].jointPositions = reinterpret_cast<PxReal*>(rp[r].p);
        arts[r].jointVelocities = reinterpret_cast<PxReal*>(rp[r].v);
        arts[r].jointAccelerations = reinterpret_cast<PxReal*>(rp[r].a);
    }
    shim.memcpyHtoD(dAb, abImg.data(), abImg.size() * sizeof(physx::PxgArticulationBlockData));
    shim.memcpyHtoD(dLk, lkImg.data(), lkImg.size() * sizeof(physx::PxgArticulationBlockLinkData));
    shim.memcpyHtoD(dDf, dfImg.data(), dfImg.size() * sizeof(physx::PxgArticulationBlockDofData));
    shim.memcpyHtoD(dArts, arts.data(), arts.size() * sizeof(physx::PxgArticulation));

    physx::PxgArticulationCoreDesc desc;
    std::memset(&desc, 0xAB, sizeof(desc));
    desc.nbArticulations = A1S_ARTS;
    desc.mMaxLinksPerArticulation = A1S_MAX_LINKS;
    desc.mMaxDofsPerArticulation = A1S_MAX_DOFS;
    desc.nbSlabs = 0;
    desc.mArticulationBlocks = reinterpret_cast<physx::PxgArticulationBlockData*>(dAb);
    desc.mArticulationLinkBlocks = reinterpret_cast<physx::PxgArticulationBlockLinkData*>(dLk);
    desc.mArticulationDofBlocks = reinterpret_cast<physx::PxgArticulationBlockDofData*>(dDf);
    desc.articulations = reinterpret_cast<physx::PxgArticulation*>(dArts);
    // 5.12: the external-Z chain sweep runs only under this flag (poison reads true)
    desc.isExternalForcesEveryTgsIterationEnabled = false;
    desc.isSleepingDisabled = false;
    shim.memcpyHtoD(dDesc, &desc, sizeof(desc));

    struct Params { uint64_t desc; float dt; uint32_t integrateWord; } prm = {
        uint64_t(dDesc), A1_DT, integrate ? 0xA7A5A301u : 0xB6B4B200u };
    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch updateBodiesLaunch1T single-link (integrate=%d)", int(integrate));
    shim.streamSynchronize();

    std::vector<physx::PxgArticulationBlockLinkData> gotLk(lkImg.size());
    shim.memcpyDtoH(gotLk.data(), dLk, gotLk.size() * sizeof(physx::PxgArticulationBlockLinkData));
    size_t bad = 0;
    {
        const uint32_t* w = reinterpret_cast<const uint32_t*>(wantLk.data());
        const uint32_t* g = reinterpret_cast<const uint32_t*>(gotLk.data());
        const size_t n = gotLk.size() * sizeof(physx::PxgArticulationBlockLinkData) / 4;
        for (size_t k = 0; k < n; ++k)
        {
            if (w[k] == g[k]) continue;
            if (integrate) continue;   // the pose path is transcendental; this run is about the ACCESS
            if (bad < 6) std::printf("MM au1-single %s word %zu ref %08x gpu %08x\n", "link", k, w[k], g[k]);
            ++bad;
        }
    }
    std::printf("[ info ] au1 single-link int=%d: %u articulations, maxLinks 1, link buffer %zu B -- the "
                "prefetch of link[1] would read %u B past it\n", int(integrate), A1S_ARTS,
                lkImg.size() * sizeof(physx::PxgArticulationBlockLinkData),
                uint32_t(sizeof(physx::PxgArticulationBlockLinkData)));
    PXG_CHECK_MSG(bad == 0, "single-link integrate=%d: %zu link words differ", int(integrate), bad);

    for (uint32_t r = 0; r < A1S_ARTS; ++r)
        for (PxgDevicePtr p : { rp[r].p, rp[r].v, rp[r].a }) if (p) shim.memFree(p);
    for (PxgDevicePtr p : { dAb, dLk, dDf, dArts, dDesc }) shim.memFree(p);
}

// 88 links: one fixed root, 86 spherical (3 dofs each) and a revolute probe, so the
// dof cursor runs 0, 3, ... 255 and the probe's joint offset is 258.
static const uint32_t A1W_LINKS = 88;
static const uint32_t A1W_DOFS = 259;

static void a1RunWideDofs(PxgCudaShim& shim)
{
    std::vector<physx::PxgArticulationBlockData> abImg(1);
    std::vector<physx::PxgArticulationBlockLinkData> lkImg(A1W_LINKS);
    std::vector<physx::PxgArticulationBlockDofData> dfImg(A1W_DOFS);
    std::memset(abImg.data(), 0xAB, abImg.size() * sizeof(physx::PxgArticulationBlockData));
    std::memset(lkImg.data(), 0xAB, lkImg.size() * sizeof(physx::PxgArticulationBlockLinkData));
    std::memset(dfImg.data(), 0xAB, dfImg.size() * sizeof(physx::PxgArticulationBlockDofData));

    const uint32_t lane = 0;
    physx::PxgArticulationBlockData& ab = abImg[0];
    auto v = [&](uint32_t i, uint32_t f) { return a1V(i * 3u + 11u, i * 61u + f); };
    auto unitQ = [&](uint32_t i, uint32_t f) {
        PxQuat q(v(i, f), v(i, f + 1), v(i, f + 2), v(i, f + 3) + 2.5f);
        return q.getNormalized();
    };
    auto unitV = [&](uint32_t i, uint32_t f) {
        PxVec3 x(v(i, f), v(i, f + 1), v(i, f + 2) + 1.5f);
        return x.getNormalized();
    };
    auto sv = [&](uint32_t i, uint32_t f) {
        return UnAlignedSpatialVector(PxVec3(v(i, f), v(i, f + 1), v(i, f + 2)),
                                      PxVec3(v(i, f + 3), v(i, f + 4), v(i, f + 5)));
    };

    ab.mNumLinks[lane] = A1W_LINKS;
    ab.mFlags[lane] = 1u;          // FIXED base: conserveMomentum returns, the test is about the offset
    ab.mStateDirty[lane] = 0u;
    ab.mTotalDofs[lane] = A1W_DOFS;
    ab.mArticulationIndex[lane] = 0u;

    uint32_t cursor = 0;
    for (uint32_t i = 0; i < A1W_LINKS; ++i)
    {
        physx::PxgArticulationBlockLinkData& L = lkImg[i];
        const uint8_t jt = (i == 0) ? uint8_t(J_FIX)
                                    : (i == A1W_LINKS - 1 ? uint8_t(J_REVOLUTE) : uint8_t(J_SPHERICAL));
        const uint32_t dofs = (i == 0) ? 0u : a1Dofs(jt);
        L.mJointType[lane] = jt;
        L.mDofs[lane] = uint8_t(dofs);
        L.mParents[lane] = (i == 0) ? 0u : i - 1u;
        L.mJointOffset[lane] = cursor;
        L.mRw_x[lane] = v(i, 3); L.mRw_y[lane] = v(i, 4); L.mRw_z[lane] = v(i, 5);
        storeSpatialVector(L.mMotionVelocity, sv(i, 6), lane);
        storeSpatialVector(L.mPosMotionVelocity, sv(i, 12), lane);
        storeSpatialTransform(L.mPreTransform, lane, PxTransform(PxVec3(v(i, 54), v(i, 55), v(i, 56)), unitQ(i, 57)));
        storeSpatialTransform(L.mAccumulatedPose, lane,
                              PxTransform(PxVec3(v(i, 20), v(i, 21), v(i, 22)), unitQ(i, 23)));
        storeSpatialTransform(L.mChildPose, lane, PxTransform(PxVec3(v(i, 27), v(i, 28), v(i, 29)), unitQ(i, 44)));
        storeSpatialTransform(L.mParentPose, lane, PxTransform(PxVec3(v(i, 33), v(i, 34), v(i, 35)), unitQ(i, 48)));
        { const PxQuat rq = unitQ(i, 39); float4 f; f.x = rq.x; f.y = rq.y; f.z = rq.z; f.w = rq.w;
          L.mRelativeQuat[lane] = f; }
        const float mass = 0.5f + std::fabs(v(i, 62)) * 0.75f;
        L.mMass[lane] = mass;
        { float4 f; f.x = 0.25f; f.y = 0.35f; f.z = 0.45f; f.w = 1.0f / mass; L.mInvInertiaXYZ_invMassW[lane] = f; }

        for (uint32_t dd = 0; dd < dofs; ++dd)
        {
            physx::PxgArticulationBlockDofData& D = dfImg[cursor + dd];
            const uint32_t u = (cursor + dd) * 7u;
            // the probe: a revolute position past 2pi, so the WRAP writes it back and the
            // written dof index is the only thing that separates truncated from not
            float jp = (jt == J_REVOLUTE) ? 7.5f : v(10 + (dd % 3u), u);
            D.mJointPositions[lane] = jp;
            D.mJointVelocities[lane] = v(11 + (dd % 3u), u) * 2.0f;
            D.mPosJointVelocities[lane] = v(12 + (dd % 3u), u) * 0.5f - 1.25f;
            D.mJointUnconstrainedVelocities[lane] = v(22 + (dd % 3u), u) * 1.5f;
            D.mJointAccel[lane] = v(23 + (dd % 3u), u);
            D.mMotion[lane] = A1_MOTION_FREE;
            storeSpatialVector(D.mLocalMotionMatrix,
                               UnAlignedSpatialVector(unitV(17 + (dd % 3u), u),
                                                      PxVec3(v(18, u), v(18, u + 1), v(18, u + 2))),
                               lane);
            storeSpatialVector(D.mWorldMotionMatrix, sv(19, u), lane);
        }
        cursor += dofs;
    }
    PXG_CHECK_MSG(cursor == A1W_DOFS, "wide fleet has %u dofs, expected %u", cursor, A1W_DOFS);
    PXG_CHECK_MSG(lkImg[A1W_LINKS - 1].mJointOffset[lane] == 258u,
                  "the probe link's joint offset is %u, and only an offset above 255 separates the "
                  "truncated read from the full one", lkImg[A1W_LINKS - 1].mJointOffset[lane]);

    // mirror
    std::vector<physx::PxgArticulationBlockData> wantAb = abImg;
    std::vector<physx::PxgArticulationBlockLinkData> wantLk = lkImg;
    std::vector<physx::PxgArticulationBlockDofData> wantDf = dfImg;
    std::vector<A1Record> wantRecs(1);
    wantRecs[0].jointP.assign(A1W_DOFS + 1, -8675309.5f);
    wantRecs[0].jointV.assign(A1W_DOFS + 1, -8675309.5f);
    wantRecs[0].jointA.assign(A1W_DOFS + 1, -8675309.5f);
    A1Coverage cov;
    uint32_t alignedHits = 0, unalignedHits = 0;
    std::vector<uint32_t> noDirty(1, AI_NONE);
    std::vector<float> noImp(1, 0.f);
    a1Mirror(noDirty.data(), noImp.data(), &wantAb[0], wantLk.data(), wantDf.data(), wantRecs[0], 0, A1_DT,
             /*integrate*/ false, lane, cov, alignedHits, unalignedHits);

    PxgDevicePtr dAb = 0, dLk = 0, dDf = 0, dArts = 0, dDesc = 0, dP = 0, dV = 0, dA = 0;
    shim.memAlloc(&dAb, abImg.size() * sizeof(physx::PxgArticulationBlockData));
    shim.memAlloc(&dLk, lkImg.size() * sizeof(physx::PxgArticulationBlockLinkData));
    shim.memAlloc(&dDf, dfImg.size() * sizeof(physx::PxgArticulationBlockDofData));
    std::vector<physx::PxgArticulation> arts(1);
    std::memset(arts.data(), 0xAB, sizeof(physx::PxgArticulation));
    shim.memAlloc(&dArts, sizeof(physx::PxgArticulation));
    shim.memAlloc(&dDesc, sizeof(physx::PxgArticulationCoreDesc));
    const std::vector<float> canary(A1W_DOFS + 1, -8675309.5f);
    shim.memAlloc(&dP, canary.size() * 4);
    shim.memAlloc(&dV, canary.size() * 4);
    shim.memAlloc(&dA, canary.size() * 4);
    if (!dAb || !dLk || !dDf || !dArts || !dDesc || !dP || !dV || !dA)
    { std::printf("  NOTE: no BDA -> skip\n"); return; }
    shim.memcpyHtoD(dP, canary.data(), canary.size() * 4);
    shim.memcpyHtoD(dV, canary.data(), canary.size() * 4);
    shim.memcpyHtoD(dA, canary.data(), canary.size() * 4);
    arts[0].jointPositions = reinterpret_cast<PxReal*>(dP);
    arts[0].jointVelocities = reinterpret_cast<PxReal*>(dV);
    arts[0].jointAccelerations = reinterpret_cast<PxReal*>(dA);
    shim.memcpyHtoD(dAb, abImg.data(), abImg.size() * sizeof(physx::PxgArticulationBlockData));
    shim.memcpyHtoD(dLk, lkImg.data(), lkImg.size() * sizeof(physx::PxgArticulationBlockLinkData));
    shim.memcpyHtoD(dDf, dfImg.data(), dfImg.size() * sizeof(physx::PxgArticulationBlockDofData));
    shim.memcpyHtoD(dArts, arts.data(), sizeof(physx::PxgArticulation));

    physx::PxgArticulationCoreDesc desc;
    std::memset(&desc, 0xAB, sizeof(desc));
    desc.nbArticulations = 1;
    desc.mMaxLinksPerArticulation = A1W_LINKS;
    desc.mMaxDofsPerArticulation = A1W_DOFS;
    desc.nbSlabs = 0;
    desc.mArticulationBlocks = reinterpret_cast<physx::PxgArticulationBlockData*>(dAb);
    desc.mArticulationLinkBlocks = reinterpret_cast<physx::PxgArticulationBlockLinkData*>(dLk);
    desc.mArticulationDofBlocks = reinterpret_cast<physx::PxgArticulationBlockDofData*>(dDf);
    desc.articulations = reinterpret_cast<physx::PxgArticulation*>(dArts);
    // 5.12: the external-Z chain sweep runs only under this flag (poison reads true)
    desc.isExternalForcesEveryTgsIterationEnabled = false;
    desc.isSleepingDisabled = false;
    shim.memcpyHtoD(dDesc, &desc, sizeof(desc));

    struct Params { uint64_t desc; float dt; uint32_t integrateWord; } prm = {
        uint64_t(dDesc), A1_DT, 0xB6B4B200u };
    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch updateBodiesLaunch1T wide-dofs");
    shim.streamSynchronize();

    std::vector<physx::PxgArticulationBlockDofData> gotDf(dfImg.size());
    std::vector<float> gotP(canary.size()), gotV(canary.size()), gotA(canary.size());
    shim.memcpyDtoH(gotDf.data(), dDf, gotDf.size() * sizeof(physx::PxgArticulationBlockDofData));
    shim.memcpyDtoH(gotP.data(), dP, gotP.size() * 4);
    shim.memcpyDtoH(gotV.data(), dV, gotV.size() * 4);
    shim.memcpyDtoH(gotA.data(), dA, gotA.size() * 4);

    size_t bad = 0;
    float maxAbs = 0.f;
    auto cmpW = [&](const char* what, const float* w, const float* g, size_t n) {
        for (size_t k = 0; k < n; ++k)
        {
            if (a1F2u(w[k]) == a1F2u(g[k])) continue;
            const uint32_t u = a1UlpDiff(w[k], g[k]);
            const float e = std::fabs(w[k] - g[k]);
            maxAbs = maxAbs > e ? maxAbs : e;
            if (u <= 384 || e <= 4e-6f) continue;   // the atan2 budget of the main fleet
            if (bad < 8)
                std::printf("MM au1-wide %s [%zu] ref %.9g gpu %.9g (%u ulp)\n", what, k, double(w[k]),
                            double(g[k]), u);
            ++bad;
        }
    };
    std::vector<float> wantJp(A1W_DOFS), gotJp(A1W_DOFS);
    for (uint32_t d = 0; d < A1W_DOFS; ++d)
    {
        wantJp[d] = wantDf[d].mJointPositions[lane];
        gotJp[d] = gotDf[d].mJointPositions[lane];
    }
    cmpW("dof jointPositions", wantJp.data(), gotJp.data(), A1W_DOFS);
    cmpW("record jointPositions", wantRecs[0].jointP.data(), gotP.data(), canary.size());
    cmpW("record jointVelocities", wantRecs[0].jointV.data(), gotV.data(), canary.size());
    cmpW("record jointAccelerations", wantRecs[0].jointA.data(), gotA.data(), canary.size());
    std::printf("[ info ] au1 wide-dofs: %u links, %u dofs, probe offset 258 -> truncated to %u; dof[2] %.6g, "
                "dof[258] %.6g; max abs %.3g\n", A1W_LINKS, A1W_DOFS, 258u & 0xFFu, double(gotJp[2]),
                double(gotJp[258]), double(maxAbs));
    PXG_CHECK_MSG(bad == 0, "wide-dofs: %zu words differ", bad);

    for (PxgDevicePtr p : { dAb, dLk, dDf, dArts, dDesc, dP, dV, dA }) shim.memFree(p);
}

PXG_TEST(artiUpdateBodies1T)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiUpdateBodies1T.spv", "updateBodiesLaunch1T", 16u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &a1Loader);

    uint32_t sdSeen[4] = { 0, 0, 0, 0 };
    uint32_t fixSeen = 0, freeSeen = 0, restSeen = 0;
    for (uint32_t a = 0; a < A1_ARTS; ++a)
    {
        sdSeen[a1StateDirty(a)] += 1u;
        if (a1FixBase(a)) ++fixSeen; else ++freeSeen;
        if (a1AtRest(a)) ++restSeen;
    }
    std::printf("[ info ] au1: stateDirty 0/1/2/3 on %u/%u/%u/%u lanes, %u fixed, %u free, %u at rest\n",
                sdSeen[0], sdSeen[1], sdSeen[2], sdSeen[3], fixSeen, freeSeen, restSeen);
    for (uint32_t c = 0; c < 4; ++c)
        PXG_CHECK_MSG(sdSeen[c] > 10, "stateDirty %u occurs only %u times", c, sdSeen[c]);
    PXG_CHECK(fixSeen > 20 && freeSeen > 20 && restSeen > 2);

    a1Run(shim, /*integrate*/ false); // TGS: the path every scene runs
    a1Run(shim, /*integrate*/ true);  // PGS
    // 5.12: eENABLE_EXTERNAL_FORCES_EVERY_ITERATION_TGS -- the external-Z chain
    // sweep (a TGS flag, so the integrate == false path)
    a1Run(shim, /*integrate*/ false, /*extEvery*/ true);
    a1RunSingleLink(shim, /*integrate*/ false);
    a1RunWideDofs(shim);
}

} // namespace
