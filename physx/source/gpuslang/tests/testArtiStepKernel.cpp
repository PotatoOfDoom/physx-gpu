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


// P8.94d: the assembled `stepArticulation1TTGS` kernel, driven over the REAL
// engine ABI — a device-resident PxgArticulationCoreDesc, the (32, 2, 1) x
// ceil(nbArticulations / 64) launch shape, and the {ptr, float} arg pack.
//
// ===========================================================================
// THE ORACLE
// ===========================================================================
//
// COMPOSITION OF THE THREE HOISTED MIRRORS, plus the deltaQ / deltaMotion
// bookkeeping this kernel adds around them: `aiMirror` (P8.94c), `fvMirror`
// (P8.94b), `refUpdateRootBody` / `refPropagateLink` (P8.94a) — all in
// pxgArtiRefHelpers.h now, moved there verbatim so this fixture is their
// fourth USER, not their fourth transcription.
//
// The comparison inherits P8.94a's measured reality: the integration goes
// through PxExp / PxQuat(angle, axis) / atan2, and llvmpipe's transcendentals
// are within ~2 ULP of glibc's but not equal.  The tolerance is scoped to the
// three fields that carry a transcendental result — `mAccumulatedPose`,
// `mJointPositions` and `mDeltaQ` (dq is BUILT from the accumulated pose) —
// and every other word, poison included, must match bit for bit.  That exact
// remainder covers everything P8.94b/c produce: the flush and the averaging
// are pure arithmetic, so their outputs tolerate NOTHING.
//
// ===========================================================================
// WHAT THE FIXTURE IS SHAPED TO SEE
// ===========================================================================
//
//   * 130 articulations -> grid 3, FIVE live warp slices with the fifth only
//     2/32 occupied, and a sixth allocated slice that must come back poisoned
//     (P8.85 measured that two blocks can alias into idempotence; three
//     cannot);
//   * `mStateDirty` takes 0, 1 (eVEL_DIRTY), 2 (eHAS_IMPULSES) and 3 across
//     lanes — 0 skips both flush and averaging, 1 flushes WITHOUT averaging,
//     2 and 3 average then flush; the clear-to-zero is checked on all three
//     non-zero cases while the 0 case must leave neighbouring lanes' bytes
//     in the shared word untouched;
//   * fix-base and free-base in the same warp, so the root branch's guard and
//     the flush's fixBase parameter are both exercised per-lane;
//   * dirty slab pairs with only .x, only .y, both and neither, impulse
//     entries on BOTH 16- and 8-byte alignments (counted and required);
//   * every link and dof field any of the three paths reads is seeded from
//     the deterministic generator; everything else is 0xAB poison.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"
#include "pxgArtiRefHelpers.h"

#include "foundation/PxQuat.h"
#include "foundation/PxTransform.h"
#include "foundation/PxVec3.h"
#include "foundation/PxMathUtils.h"
#include "CmSpatialVector.h"
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
using pxgref::JointType;
using pxgref::J_FIX;
using pxgref::J_SPHERICAL;
using pxgref::AI_NONE;

static std::vector<uint8_t> askLoader(const char* file) { return loadShaderBlob(file); }

static uint32_t askF2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static uint32_t askUlpDiff(float a, float b)
{
    if (askF2u(a) == askF2u(b)) return 0;
    if (std::isnan(a) || std::isnan(b)) return 0xFFFFFFFFu;
    int32_t ia, ib;
    std::memcpy(&ia, &a, 4);
    std::memcpy(&ib, &b, 4);
    if (ia < 0) ia = int32_t(0x80000000) - ia;
    if (ib < 0) ib = int32_t(0x80000000) - ib;
    return uint32_t(ia > ib ? ia - ib : ib - ia);
}

const uint32_t ASK_ARTS = 130;
const uint32_t ASK_WARPS = 6;         // ceil(130/64) blocks x 2 warps; slice 5 stays poisoned
const uint32_t ASK_MAX_LINKS = 6;
const uint32_t ASK_MAX_DOFS = 13;
const uint32_t ASK_NB_SLABS = 3;
const float ASK_STEP_DT = 0.02734375f;   // 7/256: exact in binary, not 1, not 1/60

// 2..6, never equal to the LANE index (a % 32): f(a) - (a%32) = 2 + (a%5) - (a%32),
// zero only when a%32 - a%5 == 2 ... enumerated below and checked at runtime
// rather than argued.
static uint32_t askNumLinks(uint32_t a) { return 2u + (a % 5u); }
static uint8_t askJointType(uint32_t a, uint32_t i) { return uint8_t((a + i) % 5u); }
static uint8_t askDofs(uint8_t jt) { return jt == J_SPHERICAL ? uint8_t(3) : (jt == J_FIX ? uint8_t(0) : uint8_t(1)); }
static uint32_t askStateDirty(uint32_t a) { return (a * 7u + 3u) % 4u; }
static bool askFixBase(uint32_t a) { return (a % 3u) == 1u; }
// mod 5, NOT mod 4: askStateDirty is mod 4, and two predicates on the same
// modulus interlock -- the first run of this fixture keyed both on a % 4 and
// every lane that both averaged AND had a dirty slab was a == 1 (mod 4), i.e.
// ODD, so every impulse read landed on one alignment class and the aligned
// counter read zero.  The P8.94a lesson again, now between two predicates
// rather than between a predicate and a branch.
static uint32_t askDirtySlabs(uint32_t a) { return a % 5u; }

static float askV(uint32_t a, uint32_t f)
{
    return (float((a * 31u + f * 13u) % 421u) + 1.0f) * 0.015625f - 3.25f;
}

static uint32_t askTotalDofs(uint32_t a, uint32_t numLinks)
{
    uint32_t t = 0;
    for (uint32_t i = 1; i < numLinks; ++i) t += askDofs(askJointType(a, i));
    return t;
}

// ---------------------------------------------------------------------------
// The fill: every field ANY of the three paths reads, seeded; the rest poison.
// ---------------------------------------------------------------------------
static void askFill(physx::PxgArticulationBlockData* ab,
                    physx::PxgArticulationBlockLinkData* lk,
                    physx::PxgArticulationBlockDofData* df, uint32_t a, uint32_t lane)
{
    const uint32_t numLinks = askNumLinks(a);
    auto v = [&](uint32_t i, uint32_t f) { return askV(a, i * 61u + f); };
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

    ab->mNumLinks[lane] = numLinks;
    ab->mFlags[lane] = askFixBase(a) ? 1u : 0u;
    ab->mStateDirty[lane] = uint8_t(askStateDirty(a));
    // above every dirty index on every fourth lane, zero otherwise
    ab->mLinkWithDeferredImpulse[lane] = ((a % 4u) == 2u) ? (numLinks - 1u) : 0u;
    storeSpatialVector(ab->mRootDeferredZ, sv(30, 0) * 0.125f, lane);
    storeSpatialVector(ab->mCommonLinkDeltaVelocity, sv(30, 6), lane);
    // THE MAGNITUDE DISCIPLINE, and it was learned the hard way.  This
    // fixture's first version seeded every response term at +-3, and the
    // impulse -> deferredQstZ -> flush chain amplified that geometrically
    // across six links: the post-flush angular velocity reached 2.97e9 --
    // BIT-IDENTICAL on both sides -- so the integration computed
    // sin(|w| * dt) at an argument of 1e8 radians, where llvmpipe's argument
    // reduction and glibc's are DIFFERENT ALGORITHMS and the results are
    // uncorrelated O(1).  That is not a port bug and no ULP bound survives
    // it; no engine scene reaches it either (velocities are clamped
    // upstream).  So the INVERSE-INERTIA-LIKE seeds are scaled to ~0.1 and
    // the impulses to ~0.05, keeping every substep rotation at a few radians
    // -- the regime the P8.94a budget was measured in.
    physx::Dy::SpatialMatrix inv;
    inv.topLeft = physx::PxMat33(PxVec3(v(31, 0), v(31, 1), v(31, 2)),
                                 PxVec3(v(31, 3), v(31, 4), v(31, 5)),
                                 PxVec3(v(31, 6), v(31, 7), v(31, 8))) * 0.03125f;
    inv.topRight = physx::PxMat33(PxVec3(v(32, 0), v(32, 1), v(32, 2)),
                                  PxVec3(v(32, 3), v(32, 4), v(32, 5)),
                                  PxVec3(v(32, 6), v(32, 7), v(32, 8))) * 0.03125f;
    inv.bottomLeft = physx::PxMat33(PxVec3(v(33, 0), v(33, 1), v(33, 2)),
                                    PxVec3(v(33, 3), v(33, 4), v(33, 5)),
                                    PxVec3(v(33, 6), v(33, 7), v(33, 8))) * 0.03125f;
    storeSpatialMatrix(ab->mInvSpatialArticulatedInertia, lane, inv);

    uint32_t cursor = 0;
    for (uint32_t i = 0; i < numLinks; ++i)
    {
        physx::PxgArticulationBlockLinkData& L = lk[i];
        const uint8_t jt = (i == 0) ? uint8_t(J_FIX) : askJointType(a, i);
        const uint32_t dofs = (i == 0) ? 0u : askDofs(jt);
        L.mJointType[lane] = jt;
        L.mDofs[lane] = uint8_t(dofs);
        L.mParents[lane] = (i == 0) ? 0u : ((i >= 3u && (i % 3u) == 0u) ? 1u : i - 1u);
        L.mJointOffset[lane] = cursor;
        L.mRw_x[lane] = v(i, 3);
        L.mRw_y[lane] = v(i, 4);
        L.mRw_z[lane] = v(i, 5);
        storeSpatialVector(L.mMotionVelocity, sv(i, 6), lane);
        storeSpatialVector(L.mPosMotionVelocity, sv(i, 12), lane);
        storeSpatialVector(L.mDeltaMotion, sv(i, 18), lane);
        storeSpatialVector(L.mScratchImpulse, sv(i, 24), lane);
        storeSpatialVector(L.mSolverSpatialImpulse, sv(i, 30), lane);
        storeSpatialVector(L.mSolverSpatialDeltaVel, sv(i, 36), lane);
        storeSpatialVector(L.mScratchDeltaV, sv(i, 42), lane);
        storeSpatialVector(L.mConstraintForces, sv(i, 48), lane);
        storeSpatialTransform(L.mPreTransform, lane,
                              PxTransform(PxVec3(v(i, 54), v(i, 55), v(i, 56)), unitQ(i, 57)));
        storeSpatialTransform(L.mAccumulatedPose, lane,
                              PxTransform(PxVec3(v(i, 20), v(i, 21), v(i, 22)), unitQ(i, 23)));
        // only the positions of the child/parent poses are read by the step,
        // but seed the quats too so a port that starts reading them is caught
        // by a real rotation rather than by poison
        storeSpatialTransform(L.mChildPose, lane,
                              PxTransform(PxVec3(v(i, 27), v(i, 28), v(i, 29)), unitQ(i, 44)));
        storeSpatialTransform(L.mParentPose, lane,
                              PxTransform(PxVec3(v(i, 33), v(i, 34), v(i, 35)), unitQ(i, 48)));
        const PxQuat rq = unitQ(i, 39);
        { float4 f; f.x = rq.x; f.y = rq.y; f.z = rq.z; f.w = rq.w;
          L.mRelativeQuat[lane] = f; }

        for (uint32_t dd = 0; dd < dofs; ++dd)
        {
            physx::PxgArticulationBlockDofData& D = df[cursor + dd];
            const uint32_t u = (cursor + dd) * 7u;
            D.mJointPositions[lane] = v(10 + dd, u);
            D.mJointVelocities[lane] = v(11 + dd, u) * 2.0f;
            D.mPosJointVelocities[lane] = v(12 + dd, u) * 0.5f - 1.25f;
            D.mDeferredQstZ[lane] = v(13 + dd, u) * 0.125f;
            D.mInvStIsT_x[lane] = v(14 + dd, u) * 0.03125f;
            D.mInvStIsT_y[lane] = v(15 + dd, u) * 0.03125f;
            D.mInvStIsT_z[lane] = v(16 + dd, u) * 0.03125f;
            storeSpatialVector(D.mLocalMotionMatrix,
                               UnAlignedSpatialVector(unitV(17 + dd, u),
                                                      PxVec3(v(18 + dd, u), v(18 + dd, u + 1),
                                                             v(18 + dd, u + 2))),
                               lane);
            storeSpatialVector(D.mWorldMotionMatrix, sv(19 + dd, u), lane);
            storeSpatialVector(D.mIsInvDW, sv(20 + dd, u) * 0.03125f, lane);
            storeSpatialVector(D.mIsW, sv(21 + dd, u) * 0.03125f, lane);
        }
        cursor += dofs;
    }
    PXG_CHECK_MSG(cursor <= ASK_MAX_DOFS, "articulation %u needs %u dofs", a, cursor);
}

// ---------------------------------------------------------------------------
// The composed mirror: the kernel's structure over the three hoisted mirrors.
// ---------------------------------------------------------------------------
static void askMirror(uint32_t* isSlabDirty, float* impulses,
                      physx::PxgArticulationBlockData* ab,
                      physx::PxgArticulationBlockLinkData* lk,
                      physx::PxgArticulationBlockDofData* df, uint32_t a, float stepDt,
                      uint32_t lane, uint32_t& alignedHits, uint32_t& unalignedHits)
{
    const uint32_t numLinks = ab->mNumLinks[lane];
    const uint32_t stateDirty = ab->mStateDirty[lane];
    const bool fixBase = (ab->mFlags[lane] & 1u) != 0u;

    if (stateDirty)
    {
        if (stateDirty & 2u)
            pxgref::aiMirror(isSlabDirty, impulses, ab, lk, df, a, ASK_MAX_LINKS, ASK_ARTS,
                             ASK_NB_SLABS, lane, alignedHits, unalignedHits);
        pxgref::fvMirror(ab, lk, df, numLinks, fixBase, lane);
        ab->mStateDirty[lane] = 0;
    }

    if (!fixBase)
    {
        physx::PxgArticulationBlockLinkData& link = lk[0];
        const PxQuat prevPoseConjugate =
            loadSpatialTransform(link.mPreTransform, lane).q.getConjugate();
        const UnAlignedSpatialVector motionVelocity = loadSpatialVector(link.mMotionVelocity,
                                                                        lane);
        const UnAlignedSpatialVector deltaMotion = loadSpatialVector(link.mDeltaMotion, lane);
        const UnAlignedSpatialVector posMotionVelocity =
            loadSpatialVector(link.mPosMotionVelocity, lane);

        const PxTransform body2World = pxgref::refUpdateRootBody(
            motionVelocity, loadSpatialTransform(link.mAccumulatedPose, lane), stepDt);

        PxQuat dq = body2World.q * prevPoseConjugate;
        if (dq.w < 0.f)
            dq = -dq;
        { float4 f; f.x = dq.x; f.y = dq.y; f.z = dq.z; f.w = dq.w;
          link.mDeltaQ[lane] = f; }

        const UnAlignedSpatialVector delta = motionVelocity * stepDt;
        storeSpatialVector(link.mDeltaMotion, deltaMotion + delta, lane);
        storeSpatialVector(link.mPosMotionVelocity, posMotionVelocity + delta, lane);
        storeSpatialTransform(link.mAccumulatedPose, lane, body2World);
    }

    for (uint32_t linkID = 1; linkID < numLinks; ++linkID)
    {
        physx::PxgArticulationBlockLinkData& link = lk[linkID];
        const PxQuat prevPoseConjugate =
            loadSpatialTransform(link.mPreTransform, lane).q.getConjugate();
        UnAlignedSpatialVector deltaMotionVel = loadSpatialVector(link.mMotionVelocity, lane);
        UnAlignedSpatialVector deltaMotion = loadSpatialVector(link.mDeltaMotion, lane);
        const UnAlignedSpatialVector posMotionVelocity =
            loadSpatialVector(link.mPosMotionVelocity, lane);

        const PxTransform body2World = pxgref::refPropagateLink(lk, df, linkID, stepDt,
                                                                /*isSubstep*/ true, lane);

        PxQuat dq = body2World.q * prevPoseConjugate;
        if (dq.w < 0.f)
            dq = -dq;
        { float4 f; f.x = dq.x; f.y = dq.y; f.z = dq.z; f.w = dq.w;
          link.mDeltaQ[lane] = f; }

        deltaMotionVel *= stepDt;
        deltaMotion += deltaMotionVel;
        storeSpatialVector(link.mDeltaMotion, deltaMotion, lane);
        storeSpatialVector(link.mPosMotionVelocity, deltaMotionVel + posMotionVelocity, lane);
        storeSpatialTransform(link.mAccumulatedPose, lane, body2World);
    }
}

PXG_TEST(artiStepKernel)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiStepKernel.spv", "stepArticulation1TTGS", 16u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &askLoader);

    // coverage, computed from the same predicates the fill uses
    uint32_t sdSeen[4] = { 0, 0, 0, 0 };
    uint32_t fixSeen = 0, freeSeen = 0, laneCollisions = 0;
    for (uint32_t a = 0; a < ASK_ARTS; ++a)
    {
        sdSeen[askStateDirty(a)] += 1u;
        if (askFixBase(a)) ++fixSeen; else ++freeSeen;
        if (askNumLinks(a) == (a & 31u)) ++laneCollisions;
    }
    std::printf("[ info ] ask: stateDirty 0/1/2/3 on %u/%u/%u/%u lanes, %u fixed, %u free\n",
                sdSeen[0], sdSeen[1], sdSeen[2], sdSeen[3], fixSeen, freeSeen);
    for (uint32_t c = 0; c < 4; ++c)
        PXG_CHECK_MSG(sdSeen[c] > 10, "stateDirty %u occurs only %u times", c, sdSeen[c]);
    PXG_CHECK(fixSeen > 20 && freeSeen > 20);
    PXG_CHECK_MSG(laneCollisions == 0, "%u lanes have numLinks == lane index", laneCollisions);

    // ---- images ----
    std::vector<physx::PxgArticulationBlockData> abImg(ASK_WARPS);
    std::vector<physx::PxgArticulationBlockLinkData> lkImg(size_t(ASK_WARPS) * ASK_MAX_LINKS);
    std::vector<physx::PxgArticulationBlockDofData> dfImg(size_t(ASK_WARPS) * ASK_MAX_DOFS);
    std::memset(abImg.data(), 0xAB, abImg.size() * sizeof(physx::PxgArticulationBlockData));
    std::memset(lkImg.data(), 0xAB, lkImg.size() * sizeof(physx::PxgArticulationBlockLinkData));
    std::memset(dfImg.data(), 0xAB, dfImg.size() * sizeof(physx::PxgArticulationBlockDofData));

    std::vector<uint32_t> dirty(size_t(ASK_NB_SLABS) * ASK_ARTS * 2, AI_NONE);
    std::vector<float> imp(size_t(ASK_NB_SLABS) * ASK_MAX_LINKS * ASK_ARTS * 6);
    for (size_t k = 0; k < imp.size(); ++k)
        imp[k] = askV(uint32_t(k % 997u), uint32_t(k / 997u)) * 0.015625f;

    for (uint32_t a = 0; a < ASK_ARTS; ++a)
    {
        const uint32_t w = a / 32u, lane = a & 31u;
        askFill(&abImg[w], &lkImg[size_t(w) * ASK_MAX_LINKS], &dfImg[size_t(w) * ASK_MAX_DOFS],
                a, lane);
        // dirty slabs, only meaningful where the averaging actually runs
        const uint32_t nd = askDirtySlabs(a);
        const uint32_t nl = askNumLinks(a);
        for (uint32_t s = 0; s < nd && s < ASK_NB_SLABS; ++s)
        {
            const size_t di = size_t(s) * ASK_ARTS + a;
            const uint32_t mode = (a + s) % 3u;
            const uint32_t x = 1u + ((a + s) % (nl - 1u));
            const uint32_t y = 1u + ((a + s + 1u) % (nl - 1u));
            if (mode != 1u) dirty[di * 2 + 0] = x;
            if (mode != 0u) dirty[di * 2 + 1] = y;
        }
    }

    // ---- the expectation ----
    std::vector<uint32_t> wantDirty = dirty;
    std::vector<float> wantImp = imp;
    std::vector<physx::PxgArticulationBlockData> wantAb = abImg;
    std::vector<physx::PxgArticulationBlockLinkData> wantLk = lkImg;
    std::vector<physx::PxgArticulationBlockDofData> wantDf = dfImg;
    uint32_t alignedHits = 0, unalignedHits = 0;
    for (uint32_t a = 0; a < ASK_ARTS; ++a)
        askMirror(wantDirty.data(), wantImp.data(), &wantAb[a / 32u],
                  &wantLk[size_t(a / 32u) * ASK_MAX_LINKS], &wantDf[size_t(a / 32u) * ASK_MAX_DOFS],
                  a, ASK_STEP_DT, a & 31u, alignedHits, unalignedHits);
    std::printf("[ info ] ask: impulse reads 16-aligned %u, 8-aligned %u\n", alignedHits,
                unalignedHits);
    PXG_CHECK_MSG(alignedHits > 5 && unalignedHits > 5,
                  "both impulse alignments must occur (%u / %u)", alignedHits, unalignedHits);

    // ---- device ----
    PxgDevicePtr dAb = 0, dLk = 0, dDf = 0, dDirty = 0, dImp = 0, dDesc = 0;
    shim.memAlloc(&dAb, abImg.size() * sizeof(physx::PxgArticulationBlockData));
    shim.memAlloc(&dLk, lkImg.size() * sizeof(physx::PxgArticulationBlockLinkData));
    shim.memAlloc(&dDf, dfImg.size() * sizeof(physx::PxgArticulationBlockDofData));
    shim.memAlloc(&dDirty, dirty.size() * sizeof(uint32_t));
    shim.memAlloc(&dImp, imp.size() * sizeof(float));
    shim.memAlloc(&dDesc, sizeof(physx::PxgArticulationCoreDesc));
    if (!dAb || !dLk || !dDf || !dDirty || !dImp || !dDesc)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    shim.memcpyHtoD(dAb, abImg.data(), abImg.size() * sizeof(physx::PxgArticulationBlockData));
    shim.memcpyHtoD(dLk, lkImg.data(), lkImg.size() * sizeof(physx::PxgArticulationBlockLinkData));
    shim.memcpyHtoD(dDf, dfImg.data(), dfImg.size() * sizeof(physx::PxgArticulationBlockDofData));
    shim.memcpyHtoD(dDirty, dirty.data(), dirty.size() * sizeof(uint32_t));
    shim.memcpyHtoD(dImp, imp.data(), imp.size() * sizeof(float));

    // the REAL descriptor, device-resident, poison outside the fields this
    // kernel reads
    physx::PxgArticulationCoreDesc desc;
    std::memset(&desc, 0xAB, sizeof(desc));
    desc.nbArticulations = ASK_ARTS;
    desc.mMaxLinksPerArticulation = ASK_MAX_LINKS;
    desc.mMaxDofsPerArticulation = ASK_MAX_DOFS;
    desc.nbSlabs = ASK_NB_SLABS;
    desc.mArticulationBlocks = reinterpret_cast<physx::PxgArticulationBlockData*>(dAb);
    desc.mArticulationLinkBlocks =
        reinterpret_cast<physx::PxgArticulationBlockLinkData*>(dLk);
    desc.mArticulationDofBlocks = reinterpret_cast<physx::PxgArticulationBlockDofData*>(dDf);
    desc.slabHasChanges = reinterpret_cast<uint2*>(dDirty);
    desc.impulses = reinterpret_cast<UnAlignedSpatialVector*>(dImp);
    shim.memcpyHtoD(dDesc, &desc, sizeof(desc));

    struct Params { uint64_t desc; float stepDt; uint32_t pad0; } prm = { uint64_t(dDesc),
                                                                          ASK_STEP_DT, 0 };
    // grid ceil(130 / 64) = 3, block (32, 2, 1) comes from the shader
    PXG_CHECK_MSG(shim.launchKernel(0, 3, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch stepArticulation1TTGS");
    shim.streamSynchronize();

    // ---- compare ----
    std::vector<physx::PxgArticulationBlockData> gotAb(abImg.size());
    std::vector<physx::PxgArticulationBlockLinkData> gotLk(lkImg.size());
    std::vector<physx::PxgArticulationBlockDofData> gotDf(dfImg.size());
    std::vector<uint32_t> gotDirty(dirty.size());
    std::vector<float> gotImp(imp.size());
    shim.memcpyDtoH(gotAb.data(), dAb, gotAb.size() * sizeof(physx::PxgArticulationBlockData));
    shim.memcpyDtoH(gotLk.data(), dLk, gotLk.size() * sizeof(physx::PxgArticulationBlockLinkData));
    shim.memcpyDtoH(gotDf.data(), dDf, gotDf.size() * sizeof(physx::PxgArticulationBlockDofData));
    shim.memcpyDtoH(gotDirty.data(), dDirty, gotDirty.size() * sizeof(uint32_t));
    shim.memcpyDtoH(gotImp.data(), dImp, gotImp.size() * sizeof(float));

    size_t bad = 0, signedZeros = 0, tolerated = 0, loud = 0;
    uint32_t maxUlp = 0;
    float maxAbs = 0.f;
    const size_t linkWords = sizeof(physx::PxgArticulationBlockLinkData) / 4;
    const size_t dofWords = sizeof(physx::PxgArticulationBlockDofData) / 4;
    auto inRange = [](size_t w, size_t structWords, size_t off, size_t bytes) {
        const size_t k = w % structWords;
        return k >= off / 4 && k < (off + bytes) / 4;
    };
    auto cmp = [&](const char* what, const void* wp, const void* gp, size_t bytes, int kind) {
        const uint32_t* w = reinterpret_cast<const uint32_t*>(wp);
        const uint32_t* g = reinterpret_cast<const uint32_t*>(gp);
        for (size_t k = 0; k < bytes / 4; ++k)
        {
            if (w[k] == g[k]) continue;
            if ((w[k] == 0u && g[k] == 0x80000000u) || (w[k] == 0x80000000u && g[k] == 0u))
            { ++signedZeros; continue; }
            // the transcendental scope: accumulated pose + deltaQ on links,
            // joint positions on dofs -- see the header
            bool tol = false;
            if (kind == 1)
                tol = inRange(k, linkWords,
                              offsetof(physx::PxgArticulationBlockLinkData, mAccumulatedPose),
                              sizeof(physx::PxgSpatialTransformBlock))
                   || inRange(k, linkWords,
                              offsetof(physx::PxgArticulationBlockLinkData, mDeltaQ),
                              sizeof(float4) * 32);
            else if (kind == 2)
                tol = inRange(k, dofWords,
                              offsetof(physx::PxgArticulationBlockDofData, mJointPositions),
                              sizeof(PxReal) * 32);
            if (tol)
            {
                float x, y;
                std::memcpy(&x, &w[k], 4);
                std::memcpy(&y, &g[k], 4);
                const uint32_t u = askUlpDiff(x, y);
                const float e = std::fabs(x - y);
                maxUlp = maxUlp > u ? maxUlp : u;
                maxAbs = maxAbs > e ? maxAbs : e;
                // The P8.94a rotvec-probe bound, for the same reason: dq and
                // the spherical joint positions have components that are
                // legitimately NEAR ZERO, where a 2-ULP transcendental input
                // error becomes thousands of RELATIVE ulps on an absolute
                // error of ~1e-7.  ULP-close OR absolutely-close, both
                // reported, loud only when neither holds.
                if (u > 384 && e > 4e-6f)
                {
                    if (loud < 8)
                        std::printf("MM ask %s tol word %zu: %u ulp, abs %.3g\n", what, k, u,
                                    double(e));
                    ++loud;
                }
                ++tolerated;
                continue;
            }
            if (bad < 12)
                std::printf("MM ask %s word %zu  ref %08x  gpu %08x\n", what, k, w[k], g[k]);
            ++bad;
        }
    };
    cmp("arti", wantAb.data(), gotAb.data(),
        wantAb.size() * sizeof(physx::PxgArticulationBlockData), 0);
    cmp("link", wantLk.data(), gotLk.data(),
        wantLk.size() * sizeof(physx::PxgArticulationBlockLinkData), 1);
    cmp("dof", wantDf.data(), gotDf.data(),
        wantDf.size() * sizeof(physx::PxgArticulationBlockDofData), 2);
    cmp("dirty", wantDirty.data(), gotDirty.data(), wantDirty.size() * sizeof(uint32_t), 0);
    cmp("impulses", wantImp.data(), gotImp.data(), wantImp.size() * sizeof(float), 0);

    std::printf("[ info ] ask: signed zeros %zu, transcendental-tolerated %zu words, max %u ulp, "
                "max abs %.3g\n", signedZeros, tolerated, maxUlp, double(maxAbs));
    // Everything the flush and the averaging write, the state-dirty clear, the
    // delta-motion integrals and all the poison: bit-exact.
    PXG_CHECK_MSG(bad == 0, "every word outside the pose/deltaQ/jointPosition scope is bit-exact");
    // Every tolerated word must be ULP-close (the P8.94a chained-pose budget)
    // OR absolutely close (the near-zero-component regime); the count that is
    // neither must be zero, and the gross-drift ceiling stands guard over the
    // absolute prong.
    PXG_CHECK_MSG(loud == 0, "%zu words are outside both the ulp and the absolute bound", loud);
    PXG_CHECK_MSG(maxAbs <= 1e-3f, "the pose drifted grossly: %.3g absolute", double(maxAbs));

    shim.memFree(dAb);
    shim.memFree(dLk);
    shim.memFree(dDf);
    shim.memFree(dDirty);
    shim.memFree(dImp);
    shim.memFree(dDesc);
}

}  // namespace
