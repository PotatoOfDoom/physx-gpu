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

// P8.94a: npArtiStep.slang — updateRootBody, propagateLink and the two
// quaternion helpers.
//
// ===========================================================================
// THE ORACLE
// ===========================================================================
//
// `PxExp`, the `PxQuat(angle, axis)` constructor, `PxQuat::rotate`,
// `getConjugate`, `getNormalized`, `PxVec3::normalize` and `PxTransform` are all
// PX_CUDA_CALLABLE, so the expectation is the engine's own arithmetic on the
// host, compared bit for bit.  `toRotationVector`, `computeSphericalJointPositions`
// and `propagateLink` are `static __device__`, so only their structure is
// transcribed.
//
// ===========================================================================
// WHAT THE FIXTURE IS SHAPED TO SEE
// ===========================================================================
//
//   * **all five joint types in the same warp**, and the two revolute variants
//     side by side so the wrap-and-clamp is distinguishable from its absence;
//   * a revolute joint driven PAST 4pi in one step, so the wrap fires, and a
//     second driven past the clamp so both operations are separately visible —
//     with the two on DIFFERENT lanes, because one lane cannot show both;
//   * a spherical joint whose angular velocity is exactly ZERO, so
//     `normalize()`'s `m > 0` guard fires and `dist` misses the 1e-6 test;
//   * `isSubstep` both ways, in two dispatches, against joint velocity fields
//     seeded with DIFFERENT values — so the four reads it selects cannot be
//     confused;
//   * link counts distinct from the lane index, and every parent chain a real
//     chain, so the parent-pose dependency is exercised;
//   * the whole link and dof block region compared byte for byte against a 0xAB
//     poison image.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "foundation/PxQuat.h"
#include "foundation/PxTransform.h"
#include "foundation/PxVec3.h"
#include "foundation/PxMathUtils.h"
#include "CmSpatialVector.h"
#include "PxgArticulation.h"

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

static std::vector<uint8_t> artiStepLoader(const char* file) { return loadShaderBlob(file); }

static uint32_t stF2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

// The ULP distance between two floats, as a monotone integer mapping.
static uint32_t stUlpDiff(float a, float b)
{
    if (stF2u(a) == stF2u(b))
        return 0;
    if (std::isnan(a) || std::isnan(b))
        return 0xFFFFFFFFu;
    int32_t ia, ib;
    std::memcpy(&ia, &a, 4);
    std::memcpy(&ib, &b, 4);
    if (ia < 0) ia = int32_t(0x80000000) - ia;
    if (ib < 0) ib = int32_t(0x80000000) - ib;
    return uint32_t(ia > ib ? ia - ib : ib - ia);
}


enum JointType : uint8_t { J_FIX = 0, J_PRISMATIC, J_REVOLUTE, J_REVOLUTE_UNWRAPPED, J_SPHERICAL };

const uint32_t ST_LANES = 25;
const uint32_t ST_MAX_LINKS = 6;
const uint32_t ST_MAX_DOFS = 13;
const float ST_DT = 0.0234375f;   // not 1, and not a power of two times 1/60

// 2..6, and distinct from the lane index for EVERY lane: with `2 + (k % 5)` the
// difference `f(k) - k` is `2 - 5*floor(k/5)`, never zero.
static uint32_t stNumLinks(uint32_t k) { return 2u + (k % 5u); }

// all five types reachable, and the two revolute variants adjacent
static uint8_t stJointType(uint32_t k, uint32_t i) { return uint8_t((k + i) % 5u); }
static uint8_t stDofs(uint8_t jt) { return jt == J_SPHERICAL ? uint8_t(3) : (jt == J_FIX ? uint8_t(0) : uint8_t(1)); }

// The two revolute extremes.  These are keyed on `(lane + link)`, i.e. on the
// same quantity `stJointType` is keyed on, and NOT on the lane alone -- which is
// how P8.94a's first sweep lost S22 (the dropped clamp).  With `k % 7 == 5` over
// 25 lanes the clamp fired on lanes 5, 12 and 19, and `stJointType(k, 1)` made
// those PRISMATIC, REVOLUTE_UNWRAPPED and FIX respectively: the case existed
// three times over and landed on a revolute exactly never.  Since
// `gcd(3, 5) == 1`, a revolute link -- one with `(k + i) == 2 (mod 5)` -- takes
// all three residues of `(k + i) % 3`, so both extremes are reached by
// construction rather than by a coincidence between two moduli.
static bool stPastWrap(uint32_t k, uint32_t i) { return ((k + i) % 3u) == 0u; }
static bool stPastClamp(uint32_t k, uint32_t i) { return ((k + i) % 3u) == 1u; }
static bool stZeroAngVel(uint32_t k) { return (k % 9u) == 4u; }
// |w| * dt lands strictly inside (0, 1e-6], so the small-angle guard is the only
// thing that decides -- the normalize() guard does not fire here.
// ...and it must land on a SPHERICAL link 1, which needs k == 3 (mod 5).
static bool stTinyAngVel(uint32_t k) { return (k % 5u) == 3u && (k % 2u) == 0u; }

static uint32_t stTotalDofs(uint32_t k, uint32_t numLinks)
{
    uint32_t t = 0;
    for (uint32_t i = 1; i < numLinks; ++i)
        t += stDofs(stJointType(k, i));
    return t;
}

static void stFill(physx::PxgArticulationBlockLinkData* lk, physx::PxgArticulationBlockDofData* df,
                   uint32_t k, uint32_t numLinks, uint32_t lane)
{
    auto v = [&](uint32_t i, uint32_t f) {
        return (float((k * 29u + i * 17u + f) % 419u) + 1.0f) * 0.015625f - 3.25f;
    };
    auto unitQ = [&](uint32_t i, uint32_t f) {
        PxQuat q(v(i, f), v(i, f + 1), v(i, f + 2), v(i, f + 3) + 2.5f);
        return q.getNormalized();
    };
    auto unitV = [&](uint32_t i, uint32_t f) {
        PxVec3 a(v(i, f), v(i, f + 1), v(i, f + 2) + 1.5f);
        return a.getNormalized();
    };

    uint32_t cursor = 0;
    for (uint32_t i = 0; i < numLinks; ++i)
    {
        physx::PxgArticulationBlockLinkData& b = lk[i];
        const uint8_t jt = (i == 0) ? uint8_t(J_FIX) : stJointType(k, i);
        const uint8_t dofs = (i == 0) ? uint8_t(0) : stDofs(jt);
        b.mParents[lane] = (i == 0) ? 0u : (i - 1u);
        b.mDofs[lane] = dofs;
        b.mJointType[lane] = jt;
        b.mJointOffset[lane] = cursor;

        const PxQuat q = unitQ(i, 0);
        b.mAccumulatedPose.q[lane] = make_float4(q.x, q.y, q.z, q.w);
        b.mAccumulatedPose.p[lane] = make_float4(v(i, 10), v(i, 11), v(i, 12), 0.f);

        const PxQuat rq = unitQ(i, 20);
        b.mRelativeQuat[lane] = make_float4(rq.x, rq.y, rq.z, rq.w);

        const PxQuat cq = unitQ(i, 30), pq = unitQ(i, 40);
        b.mChildPose.q[lane] = make_float4(cq.x, cq.y, cq.z, cq.w);
        b.mChildPose.p[lane] = make_float4(v(i, 50), v(i, 51), v(i, 52), 0.f);
        b.mParentPose.q[lane] = make_float4(pq.x, pq.y, pq.z, pq.w);
        b.mParentPose.p[lane] = make_float4(v(i, 53), v(i, 54), v(i, 55), 0.f);

        // a zero-angular-velocity link for the normalize() guard, and a
        // separate one whose angular velocity is tiny but NON-zero, so
        // `dist` lands strictly inside (0, 1e-6] and only the small-angle
        // guard decides
        const bool zeroAng = (i == 1) && stZeroAngVel(k);
        const bool tinyAng = (i == 1) && stTinyAngVel(k);
        const PxVec3 tiny(1e-6f, 0.f, 0.f);
        storeSpatialVector(b.mMotionVelocity,
                           UnAlignedSpatialVector(zeroAng ? PxVec3(0.f)
                                                          : (tinyAng ? tiny
                                                                     : PxVec3(v(i, 60), v(i, 61),
                                                                              v(i, 62))),
                                                  PxVec3(v(i, 63), v(i, 64), v(i, 65))),
                           lane);
        storeSpatialVector(b.mPosMotionVelocity,
                           UnAlignedSpatialVector(zeroAng ? PxVec3(0.f)
                                                          : (tinyAng ? tiny
                                                                     : PxVec3(v(i, 66) * 2.f,
                                                                              v(i, 67) * 2.f,
                                                                              v(i, 68) * 2.f)),
                                                  PxVec3(v(i, 69), v(i, 70), v(i, 71))),
                           lane);

        for (uint32_t dd = 0; dd < dofs; ++dd)
        {
            physx::PxgArticulationBlockDofData& d = df[cursor + dd];
            const PxVec3 axis = unitV(i, 80 + dd * 4);
            storeSpatialVector(d.mLocalMotionMatrix,
                               UnAlignedSpatialVector(axis, PxVec3(v(i, 90 + dd * 3),
                                                                  v(i, 91 + dd * 3),
                                                                  v(i, 92 + dd * 3))),
                               lane);
            // a joint position that the step drives past 4pi / past the clamp
            float jp = v(i, 100 + dd);
            float jv = v(i, 105 + dd) * 3.0f;
            if (jt == J_REVOLUTE && stPastWrap(k, i))
            {
                jp = 6.5f;          // just above 2pi: the WRAP fires and the clamp does not
                jv = 8.0f;
            }
            else if (jt == J_REVOLUTE && stPastClamp(k, i))
            {
                // A wrap alone can never reach the clamp -- it subtracts exactly
                // 4pi once.  Only a step large enough to leave |jPos| above 4pi
                // AFTER the wrap makes the clamp observable.
                jp = 1.0f;
                jv = 10000.0f;
            }
            else if (jt == J_REVOLUTE_UNWRAPPED && stPastClamp(k, i))
            {
                // the same magnitude on an UNWRAPPED joint, where neither the
                // wrap nor the clamp may fire
                jp = 1.0f;
                jv = 10000.0f;
            }
            d.mJointPositions[lane] = jp;
            // the two velocity fields are DIFFERENT, so `isSubstep` is observable
            d.mJointVelocities[lane] = jv;
            d.mPosJointVelocities[lane] = jv * 0.5f - 1.25f;
        }
        cursor += dofs;
    }
}

// ---------------------------------------------------------------------------
// The mirror: the CUDA's structure, the engine's arithmetic
// ---------------------------------------------------------------------------

static PxVec3 refToRotationVector(const PxQuat& q)
{
    const float quatEpsilon = 1.0e-8f;
    const float s2 = q.x * q.x + q.y * q.y + q.z * q.z;
    PxReal angle;
    PxVec3 axis;
    if (s2 < quatEpsilon * quatEpsilon)
    {
        angle = 0.0f;
        axis = PxVec3(1.0f, 0.0f, 0.0f);
    }
    else
    {
        const float s = physx::PxRecipSqrt(s2);
        axis = PxVec3(q.x, q.y, q.z) * s;
        angle = physx::PxAbs(q.w) < quatEpsilon ? physx::PxPi
                                                : physx::PxAtan2(s2 * s, q.w) * 2.0f;
    }
    return axis * angle;
}

static PxTransform refUpdateRootBody(const UnAlignedSpatialVector& motionVelocity,
                                     const PxTransform& preTransform, PxReal dt)
{
    const PxVec3 comLinVel = motionVelocity.bottom;
    const PxVec3 newP = (preTransform.p) + comLinVel * dt;
    const PxQuat deltaQ = physx::PxExp(motionVelocity.top * dt);
    return PxTransform(newP, (deltaQ * preTransform.q).getNormalized());
}

static PxQuat refSphericalJointPositions(const PxQuat& relativeQuat, const PxQuat& newRot,
                                         const PxQuat& pBody2WorldRot,
                                         physx::PxgArticulationBlockDofData* dofData, PxU32 dofs,
                                         uint32_t lane)
{
    PxQuat newParentToChild = (newRot.getConjugate() * pBody2WorldRot).getNormalized();
    if (newParentToChild.w < 0.f)
        newParentToChild = -newParentToChild;

    PxQuat jointRotation = newParentToChild * relativeQuat.getConjugate();
    if (jointRotation.w < 0)
        jointRotation = -jointRotation;

    const PxVec3 axis = refToRotationVector(jointRotation);
    for (PxU32 dd = 0; dd < dofs; ++dd)
    {
        const float4 top = dofData[dd].mLocalMotionMatrix.mTopxyz_bx[lane];
        const PxReal ang = -PxVec3(top.x, top.y, top.z).dot(axis);
        dofData[dd].mJointPositions[lane] = ang;
    }
    return newParentToChild;
}

static PxTransform refPropagateLink(physx::PxgArticulationBlockLinkData* lk,
                                    physx::PxgArticulationBlockDofData* df, uint32_t linkID,
                                    PxReal dt, bool isSubstep, uint32_t lane)
{
    physx::PxgArticulationBlockLinkData& link = lk[linkID];
    const PxU32 jointOffset = link.mJointOffset[lane];
    const PxU32 jointType = link.mJointType[lane];
    const PxU32 parent = link.mParents[lane];
    physx::PxgArticulationBlockDofData* dof = df + jointOffset;

    const PxTransform pBody2World = loadSpatialTransform(lk[parent].mAccumulatedPose, lane);

    const float4 co = link.mChildPose.p[lane];
    const float4 po = link.mParentPose.p[lane];
    const float4 rq = link.mRelativeQuat[lane];
    const PxVec3 childOffset(-co.x, -co.y, -co.z);
    const PxVec3 parentOffset(po.x, po.y, po.z);
    const PxQuat relativeQuat(rq.x, rq.y, rq.z, rq.w);

    PxQuat newParentToChild = relativeQuat;
    PxVec3 offset(0.f);

    if (jointType == J_PRISMATIC)
    {
        const PxReal delta = (isSubstep ? dof[0].mJointVelocities[lane]
                                        : dof[0].mPosJointVelocities[lane]) * dt;
        const PxReal pos = dof[0].mJointPositions[lane] + delta;
        dof[0].mJointPositions[lane] = pos;
        newParentToChild = relativeQuat;
        const UnAlignedSpatialVector mm = loadSpatialVector(dof[0].mLocalMotionMatrix, lane);
        offset = mm.bottom * pos;
    }
    else if (jointType == J_REVOLUTE)
    {
        const PxReal delta = (isSubstep ? dof[0].mJointVelocities[lane]
                                        : dof[0].mPosJointVelocities[lane]) * dt;
        PxReal jPos = dof[0].mJointPositions[lane] + delta;
        if (jPos > physx::PxTwoPi)
            jPos -= physx::PxTwoPi * 2.f;
        else if (jPos < -physx::PxTwoPi)
            jPos += physx::PxTwoPi * 2.f;
        jPos = physx::PxClamp(jPos, -physx::PxTwoPi * 2.f, physx::PxTwoPi * 2.f);
        dof[0].mJointPositions[lane] = jPos;
        const float4 u = dof[0].mLocalMotionMatrix.mTopxyz_bx[lane];
        PxQuat jointRotation = PxQuat(-jPos, PxVec3(u.x, u.y, u.z));
        if (jointRotation.w < 0)
            jointRotation = -jointRotation;
        newParentToChild = (jointRotation * relativeQuat).getNormalized();
    }
    else if (jointType == J_REVOLUTE_UNWRAPPED)
    {
        const PxReal delta = (isSubstep ? dof[0].mJointVelocities[lane]
                                        : dof[0].mPosJointVelocities[lane]) * dt;
        const PxReal jPos = dof[0].mJointPositions[lane] + delta;
        dof[0].mJointPositions[lane] = jPos;
        const float4 u = dof[0].mLocalMotionMatrix.mTopxyz_bx[lane];
        PxQuat jointRotation = PxQuat(-jPos, PxVec3(u.x, u.y, u.z));
        if (jointRotation.w < 0)
            jointRotation = -jointRotation;
        newParentToChild = (jointRotation * relativeQuat).getNormalized();
    }
    else if (jointType == J_SPHERICAL)
    {
        const PxU32 nbDofs = link.mDofs[lane];
        const PxTransform oldTransform = loadSpatialTransform(link.mAccumulatedPose, lane);
        UnAlignedSpatialVector worldVel = isSubstep
                                              ? loadSpatialVector(link.mMotionVelocity, lane)
                                              : loadSpatialVector(link.mPosMotionVelocity, lane);
        PxVec3 worldAngVel = worldVel.top;
        const PxReal dist = worldAngVel.normalize() * dt;
        PxQuat newWorldQ;
        if (dist > 1e-6f)
            newWorldQ = PxQuat(dist, worldAngVel) * oldTransform.q;
        else
            newWorldQ = oldTransform.q;
        newParentToChild =
            refSphericalJointPositions(relativeQuat, newWorldQ, pBody2World.q, dof, nbDofs, lane);
    }
    else
    {
        newParentToChild = relativeQuat;
    }

    const PxVec3 e = newParentToChild.rotate(parentOffset);
    const PxVec3 d = childOffset;
    const PxVec3 r = e + d + offset;

    PxTransform body2World;
    body2World.q = (pBody2World.q * newParentToChild.getConjugate()).getNormalized();
    body2World.p = pBody2World.p + body2World.q.rotate(r);
    return body2World;
}

static void stMirror(physx::PxgArticulationBlockLinkData* lk,
                     physx::PxgArticulationBlockDofData* df, uint32_t numLinks, PxReal dt,
                     bool isSubstep, uint32_t lane)
{
    const UnAlignedSpatialVector mv = loadSpatialVector(lk[0].mMotionVelocity, lane);
    const PxTransform root =
        refUpdateRootBody(mv, loadSpatialTransform(lk[0].mAccumulatedPose, lane), dt);
    storeSpatialTransform(lk[0].mAccumulatedPose, lane, root);

    for (uint32_t linkID = 1; linkID < numLinks; ++linkID)
    {
        const PxTransform b = refPropagateLink(lk, df, linkID, dt, isSubstep, lane);
        storeSpatialTransform(lk[linkID].mAccumulatedPose, lane, b);
    }
}

PXG_TEST(artiStepAlgebra)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiStep.spv", "artiStepAlgebra", 48u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &artiStepLoader);

    std::vector<physx::PxgArticulationBlockLinkData> lkImg(ST_MAX_LINKS);
    std::vector<physx::PxgArticulationBlockDofData> dfImg(ST_MAX_DOFS);
    std::memset(lkImg.data(), 0xAB, lkImg.size() * sizeof(physx::PxgArticulationBlockLinkData));
    std::memset(dfImg.data(), 0xAB, dfImg.size() * sizeof(physx::PxgArticulationBlockDofData));

    std::vector<uint32_t> numLinks(32, 0u);
    uint32_t seen[5] = { 0, 0, 0, 0, 0 };
    uint32_t wrap = 0, clampd = 0, zeroAng = 0, tinyAng = 0, sphZero = 0, sphTiny = 0;
    for (uint32_t lane = 0; lane < ST_LANES; ++lane)
    {
        numLinks[lane] = stNumLinks(lane);
        PXG_CHECK_MSG(numLinks[lane] <= ST_MAX_LINKS, "lane %u has %u links", lane,
                      numLinks[lane]);
        PXG_CHECK_MSG(numLinks[lane] != lane, "lane %u's link count equals its index", lane);
        PXG_CHECK_MSG(stTotalDofs(lane, numLinks[lane]) <= ST_MAX_DOFS, "lane %u needs %u dofs",
                      lane, stTotalDofs(lane, numLinks[lane]));
        stFill(lkImg.data(), dfImg.data(), lane, numLinks[lane], lane);
        for (uint32_t i = 1; i < numLinks[lane]; ++i)
        {
            const uint8_t jt = stJointType(lane, i);
            seen[jt] += 1u;
            // count the extremes exactly as `stFill` applies them -- on a
            // REVOLUTE link, not on a lane
            if (jt == J_REVOLUTE && stPastWrap(lane, i)) ++wrap;
            if (jt == J_REVOLUTE && stPastClamp(lane, i)) ++clampd;
        }
        if (stZeroAngVel(lane)) ++zeroAng;
        if (stTinyAngVel(lane)) ++tinyAng;
        // both only matter on a SPHERICAL link 1 -- count the ones that land there
        if (numLinks[lane] > 1 && stJointType(lane, 1) == J_SPHERICAL)
        {
            if (stZeroAngVel(lane)) ++sphZero;
            if (stTinyAngVel(lane)) ++sphTiny;
        }
    }
    std::printf("[ info ] st: joint types %u/%u/%u/%u/%u, %u wrap, %u clamp, %u zero-ang "
                "(%u spherical), %u tiny-ang (%u spherical)\n",
                seen[0], seen[1], seen[2], seen[3], seen[4], wrap, clampd, zeroAng, sphZero,
                tinyAng, sphTiny);
    for (uint32_t t = 0; t < 5; ++t)
        PXG_CHECK_MSG(seen[t] > 4, "joint type %u occurs only %u times", t, seen[t]);
    PXG_CHECK(wrap > 1 && clampd > 1 && zeroAng > 1 && tinyAng > 1);
    // the two spherical guards are only reachable on a spherical link
    PXG_CHECK_MSG(sphZero > 0, "no lane puts a zero angular velocity on a spherical link");
    PXG_CHECK_MSG(sphTiny > 0, "no lane puts a tiny angular velocity on a spherical link");

    PxgDevicePtr dLk = 0, dDf = 0, dCounts = 0;
    shim.memAlloc(&dLk, lkImg.size() * sizeof(physx::PxgArticulationBlockLinkData));
    shim.memAlloc(&dDf, dfImg.size() * sizeof(physx::PxgArticulationBlockDofData));
    shim.memAlloc(&dCounts, numLinks.size() * sizeof(uint32_t));
    if (!dLk || !dDf || !dCounts)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    shim.memcpyHtoD(dCounts, numLinks.data(), numLinks.size() * sizeof(uint32_t));

    size_t bad = 0, signedZeros = 0, tolerated = 0;
    uint32_t maxUlp = 0, maxUlpRoot = 0;
    for (uint32_t pass = 0; pass < 2; ++pass)
    {
        const bool isSubstep = (pass == 0);

        shim.memcpyHtoD(dLk, lkImg.data(),
                        lkImg.size() * sizeof(physx::PxgArticulationBlockLinkData));
        shim.memcpyHtoD(dDf, dfImg.data(),
                        dfImg.size() * sizeof(physx::PxgArticulationBlockDofData));

        struct Params
        {
            uint64_t linkBlocks, dofBlocks, numLinksPer;
            uint32_t nbLanes, isSubstep;
            float dt;
            uint32_t pad0, pad1, pad2;
        } prm = { uint64_t(dLk), uint64_t(dDf), uint64_t(dCounts), ST_LANES,
                  isSubstep ? 1u : 0u, ST_DT, 0, 0, 0 };
        PXG_CHECK_MSG(sizeof(prm) == 48, "push-constant blob is %zu bytes", sizeof(prm));

        PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                      "launch artiStepAlgebra");
        shim.streamSynchronize();

        std::vector<physx::PxgArticulationBlockLinkData> wantLk = lkImg;
        std::vector<physx::PxgArticulationBlockDofData> wantDf = dfImg;
        for (uint32_t lane = 0; lane < ST_LANES; ++lane)
            stMirror(wantLk.data(), wantDf.data(), numLinks[lane], ST_DT, isSubstep, lane);

        std::vector<physx::PxgArticulationBlockLinkData> gotLk(lkImg.size());
        std::vector<physx::PxgArticulationBlockDofData> gotDf(dfImg.size());
        shim.memcpyDtoH(gotLk.data(), dLk,
                        gotLk.size() * sizeof(physx::PxgArticulationBlockLinkData));
        shim.memcpyDtoH(gotDf.data(), dDf,
                        gotDf.size() * sizeof(physx::PxgArticulationBlockDofData));

        // ---- the comparison, and why it is NOT bit-exact ----
        //
        // `artiStepQuatProbe` measures the reason: llvmpipe's `sin`/`cos` and
        // glibc's agree to within 2 ULP but not exactly, and PxExp and the
        // PxQuat(angle, axis) constructor both go through them.  Everything
        // downstream inherits that.  So the tolerance is scoped to the fields
        // that can carry a transcendental result -- `mAccumulatedPose` and
        // `mJointPositions` -- and EVERY OTHER WORD, poison included, must match
        // exactly.  The worst ULP seen is printed and bounded.
        auto inRange = [](size_t wordIdx, size_t structWords, size_t fieldOff, size_t fieldBytes) {
            const size_t w = wordIdx % structWords;
            return w >= fieldOff / 4 && w < (fieldOff + fieldBytes) / 4;
        };
        const size_t linkWords = sizeof(physx::PxgArticulationBlockLinkData) / 4;
        const size_t dofWords = sizeof(physx::PxgArticulationBlockDofData) / 4;

        auto cmp = [&](const char* what, const void* wp, const void* gp, size_t bytes,
                       bool isLink) {
            const uint32_t* w = reinterpret_cast<const uint32_t*>(wp);
            const uint32_t* g = reinterpret_cast<const uint32_t*>(gp);
            for (size_t k = 0; k < bytes / 4; ++k)
            {
                if (w[k] == g[k])
                    continue;
                if ((w[k] == 0u && g[k] == 0x80000000u) || (w[k] == 0x80000000u && g[k] == 0u))
                {
                    ++signedZeros;
                    continue;
                }
                const bool tolerable =
                    isLink ? inRange(k, linkWords,
                                     offsetof(physx::PxgArticulationBlockLinkData,
                                              mAccumulatedPose),
                                     sizeof(physx::PxgSpatialTransformBlock))
                           : inRange(k, dofWords,
                                     offsetof(physx::PxgArticulationBlockDofData, mJointPositions),
                                     sizeof(PxReal) * 32);
                if (tolerable)
                {
                    float a, b;
                    std::memcpy(&a, &w[k], 4);
                    std::memcpy(&b, &g[k], 4);
                    const uint32_t u = stUlpDiff(a, b);
                    // split by link, because the amplification is the whole
                    // point: the ROOT goes through exactly one PxExp, every other
                    // link inherits its parent's pose and adds a quaternion
                    // multiply, a normalize and a rotate of its own.
                    if (isLink && (k / linkWords) == 0)
                        maxUlpRoot = maxUlpRoot > u ? maxUlpRoot : u;
                    else
                        maxUlp = maxUlp > u ? maxUlp : u;
                    ++tolerated;
                    continue;
                }
                if (bad < 12)
                    std::printf("MM st pass %u %s word %zu  ref %08x  gpu %08x\n", pass, what, k,
                                w[k], g[k]);
                ++bad;
            }
        };
        cmp("link", wantLk.data(), gotLk.data(),
            wantLk.size() * sizeof(physx::PxgArticulationBlockLinkData), true);
        cmp("dof", wantDf.data(), gotDf.data(),
            wantDf.size() * sizeof(physx::PxgArticulationBlockDofData), false);
    }

    std::printf("[ info ] st: signed zeros %zu, transcendental-tolerated %zu words, "
                "max %u ulp (root %u)\n", signedZeros, tolerated, maxUlp, maxUlpRoot);
    // The structure is pinned by this one: every word outside the pose and the
    // joint positions -- the joint-type dispatch, the wrap and the clamp, the
    // offsets, the isSubstep field selection, and all the poison -- is bit-exact.
    PXG_CHECK_MSG(bad == 0, "every word outside the pose and joint-position fields is bit-exact");
    // The ROOT goes through exactly ONE PxExp, so it must stay within the 2 ulp
    // `artiStepQuatProbe` measures, plus room for the normalize.
    PXG_CHECK_MSG(maxUlpRoot <= 8, "the root pose drifted: %u ulp", maxUlpRoot);
    // Every other link inherits its parent's pose and adds a quaternion multiply,
    // a normalize and a rotate.  Measured at 256 over a six-link chain; the bound
    // is set just above so drift is visible rather than absorbed.
    PXG_CHECK_MSG(maxUlp <= 384, "the chained pose drifted past its budget: %u ulp", maxUlp);

    shim.memFree(dLk);
    shim.memFree(dDf);
    shim.memFree(dCounts);
}


// ---------------------------------------------------------------------------
// The quaternion probe: is the difference transcendental or algebraic?
// ---------------------------------------------------------------------------
//
// `PxExp` and the `PxQuat(angle, axis)` constructor both go through `sin` and
// `cos`.  Everything downstream of them inherits whatever they do.  This test
// isolates them so the answer is measured rather than assumed, and it reports the
// worst ULP it sees either way.

PXG_TEST(artiStepQuatProbe)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiStep.spv", "artiStepQuatProbe", 32u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &artiStepLoader);

    const uint32_t N = 257;
    auto val = [](uint32_t i, uint32_t k) {
        return (float((i * 37u + k * 11u) % 641u) + 1.0f) * 0.009765625f - 3.0f;
    };

    std::vector<float> in(size_t(N) * 7);
    uint32_t tiny = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        float* c = in.data() + size_t(i) * 7;
        c[0] = val(i, 0); c[1] = val(i, 1); c[2] = val(i, 2);
        if (i % 23 == 4)
        {
            // below PxExp's 1e-24 magnitude-squared threshold: the identity path,
            // which touches no transcendental at all
            c[0] = c[1] = c[2] = 0.0f;
            ++tiny;
        }
        c[3] = val(i, 3) * 2.0f;
        const PxVec3 axis = PxVec3(val(i, 4), val(i, 5), val(i, 6) + 1.5f).getNormalized();
        c[4] = axis.x; c[5] = axis.y; c[6] = axis.z;
    }
    PXG_CHECK(tiny > 8);

    PxgDevicePtr dIn = 0, dOut = 0;
    shim.memAlloc(&dIn, in.size() * sizeof(float));
    shim.memAlloc(&dOut, size_t(N) * 8 * sizeof(float));
    if (!dIn || !dOut)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    shim.memcpyHtoD(dIn, in.data(), in.size() * sizeof(float));

    struct Params { uint64_t inF, outF; uint32_t n, p0, p1, p2; } prm = { uint64_t(dIn),
                                                                          uint64_t(dOut), N, 0, 0,
                                                                          0 };
    PXG_CHECK_MSG(shim.launchKernel(0, (N + 63) / 64, 1, 1, &prm, sizeof(prm))
                      == PxgResult::Success,
                  "launch artiStepQuatProbe");
    shim.streamSynchronize();

    std::vector<float> got(size_t(N) * 8);
    shim.memcpyDtoH(got.data(), dOut, got.size() * sizeof(float));

    uint32_t maxUlpExp = 0, maxUlpAxis = 0, exactExp = 0, exactAxis = 0, identityExact = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* c = in.data() + size_t(i) * 7;
        const PxQuat e = physx::PxExp(PxVec3(c[0], c[1], c[2]));
        const PxQuat a(c[3], PxVec3(c[4], c[5], c[6]));
        const float* g = got.data() + size_t(i) * 8;

        uint32_t ue = 0, ua = 0;
        for (uint32_t k = 0; k < 4; ++k)
        {
            const uint32_t d0 = stUlpDiff((&e.x)[k], g[k]);
            const uint32_t d1 = stUlpDiff((&a.x)[k], g[4 + k]);
            ue = ue > d0 ? ue : d0;
            ua = ua > d1 ? ua : d1;
        }
        if (ue == 0) ++exactExp;
        if (ua == 0) ++exactAxis;
        if (c[0] == 0.0f && c[1] == 0.0f && c[2] == 0.0f && ue == 0) ++identityExact;
        maxUlpExp = maxUlpExp > ue ? maxUlpExp : ue;
        maxUlpAxis = maxUlpAxis > ua ? maxUlpAxis : ua;
    }

    std::printf("[ info ] quat probe: PxExp exact %u/%u (max %u ulp), PxQuat(angle,axis) exact "
                "%u/%u (max %u ulp), identity path exact %u/%u\n",
                exactExp, N, maxUlpExp, exactAxis, N, maxUlpAxis, identityExact, tiny);

    // The identity path touches NO transcendental, so it must be exact.
    PXG_CHECK_MSG(identityExact == tiny,
                  "PxExp's identity path is not bit-exact (%u of %u)", identityExact, tiny);
    // sin/cos are llvmpipe's polynomials against glibc's -- a few ULP, not zero.
    PXG_CHECK_MSG(maxUlpExp <= 4 && maxUlpAxis <= 4,
                  "the quaternion helpers drifted: %u / %u ulp", maxUlpExp, maxUlpAxis);

    shim.memFree(dIn);
    shim.memFree(dOut);
}


// `toRotationVector` probed directly, because its tiny-axis guard is not
// reachable from the block fixture: it needs a spherical joint whose relative
// quaternion exactly cancels its parent's computed pose, and the parent's pose is
// itself an output of the mirror.  The table therefore includes EXACT identity
// quaternions, which is the only input that takes that branch.
PXG_TEST(artiStepRotVecProbe)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiStep.spv", "artiStepRotVecProbe", 32u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &artiStepLoader);

    const uint32_t N = 193;
    auto val = [](uint32_t i, uint32_t k) {
        return (float((i * 43u + k * 13u) % 509u) + 1.0f) * 0.0078125f - 2.0f;
    };

    std::vector<float> in(size_t(N) * 4);
    uint32_t identity = 0, nearHalf = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        float* c = in.data() + size_t(i) * 4;
        PxQuat q(val(i, 0), val(i, 1), val(i, 2), val(i, 3) + 2.0f);
        q = q.getNormalized();
        if (i % 17 == 3)
        {
            // EXACTLY the identity: s2 is zero, which is the only way into the
            // "cannot extract a sensible axis" branch
            q = PxQuat(physx::PxIdentity);
            ++identity;
        }
        else if (i % 23 == 5)
        {
            // w exactly zero: the OTHER branch, where the angle is PxPi rather
            // than an atan2
            q = PxQuat(0.f, 0.f, 1.f, 0.f);
            ++nearHalf;
        }
        c[0] = q.x; c[1] = q.y; c[2] = q.z; c[3] = q.w;
    }
    PXG_CHECK(identity > 5 && nearHalf > 3);

    PxgDevicePtr dIn = 0, dOut = 0;
    shim.memAlloc(&dIn, in.size() * sizeof(float));
    shim.memAlloc(&dOut, size_t(N) * 3 * sizeof(float));
    if (!dIn || !dOut)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    shim.memcpyHtoD(dIn, in.data(), in.size() * sizeof(float));

    struct Params { uint64_t inF, outF; uint32_t n, p0, p1, p2; } prm = { uint64_t(dIn),
                                                                          uint64_t(dOut), N, 0, 0,
                                                                          0 };
    PXG_CHECK_MSG(shim.launchKernel(0, (N + 63) / 64, 1, 1, &prm, sizeof(prm))
                      == PxgResult::Success,
                  "launch artiStepRotVecProbe");
    shim.streamSynchronize();

    std::vector<float> got(size_t(N) * 3);
    shim.memcpyDtoH(got.data(), dOut, got.size() * sizeof(float));

    uint32_t maxUlp = 0, exact = 0, identityExact = 0, loud = 0;
    float maxAbs = 0.f;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* c = in.data() + size_t(i) * 4;
        const PxVec3 r = refToRotationVector(PxQuat(c[0], c[1], c[2], c[3]));
        const float* g = got.data() + size_t(i) * 3;
        // A pure ULP bound is the wrong metric here.  For a NEAR-IDENTITY
        // quaternion the angle is `atan2(tiny, ~1)`, so a few ULP of absolute
        // error in atan2 becomes a large RELATIVE error -- and the rotation
        // vector it scales is correspondingly tiny.  So the test asks for ULP-
        // close OR absolutely-close, and reports both.
        uint32_t u = 0;
        float absErr = 0.f;
        for (uint32_t k = 0; k < 3; ++k)
        {
            const uint32_t d = stUlpDiff((&r.x)[k], g[k]);
            u = u > d ? u : d;
            const float e = std::fabs((&r.x)[k] - g[k]);
            absErr = absErr > e ? absErr : e;
        }
        if (u == 0) ++exact;
        if (i % 17 == 3 && u == 0) ++identityExact;
        if (u > 8 && absErr > 4e-6f)
        {
            if (loud < 8)
                std::printf("MM rotvec case %u: %u ulp, abs %.9g\n", i, u, absErr);
            ++loud;
        }
        maxAbs = maxAbs > absErr ? maxAbs : absErr;
        maxUlp = maxUlp > u ? maxUlp : u;
    }
    std::printf("[ info ] rotvec probe: exact %u/%u, max %u ulp, max abs %.3g, identity branch "
                "exact %u/%u, beyond both bounds %u\n",
                exact, N, maxUlp, double(maxAbs), identityExact, identity, loud);

    // The identity branch touches no transcendental at all.
    PXG_CHECK_MSG(identityExact == identity, "the tiny-axis branch is not bit-exact (%u of %u)",
                  identityExact, identity);
    // The rest goes through atan2 -- llvmpipe's polynomial against glibc's.
    // Every case must be within 8 ulp OR within 4e-6 absolute; the count that is
    // neither must be zero.  4e-6 is chosen just above the MEASURED worst case
    // (2.64e-6, printed above so drift stays visible), not picked to be safe.
    // It is loose in ULP terms -- on a component of order 1 it is ~33 ulp -- but
    // it is still three orders of magnitude tighter than any algebraic error the
    // sweep can produce here: the cheapest wrong answer in this function (w vs
    // -w, a missing factor of two, the half-angle dropped) moves a component by
    // O(0.1), not by O(1e-6).
    PXG_CHECK_MSG(loud == 0, "%u cases are outside both the ulp and the absolute bound", loud);
    PXG_CHECK_MSG(maxAbs <= 1e-5f, "toRotationVector drifted: %.3g absolute", double(maxAbs));

    shim.memFree(dIn);
    shim.memFree(dOut);
}


// The spherical small-angle guard, probed directly.
//
// WHY IT NEEDS ITS OWN TEST.  P8.94a's first sweep left S31 ("the spherical
// small-angle guard is dropped") alive even though the block fixture DOES put a
// sub-threshold angular velocity on a spherical link -- `sphTiny` counts them.
// The reason is arithmetic, not coverage: below the threshold the rotation the
// guard skips is at most 1e-6 radians, which moves a unit quaternion by ~4 ULP,
// and `artiStepAlgebra` has to tolerate 384 ULP on a chained pose because
// llvmpipe's `sin`/`cos` are not glibc's (`artiStepQuatProbe` measures that).
// The mutation is real but it is smaller than the noise floor the platform
// forces on that comparison, so no bound `artiStepAlgebra` can carry will
// separate them.
//
// Probing the function directly removes the noise entirely.  Below the threshold
// the engine does not approximate the rotation -- it returns the OLD quaternion,
// untouched.  That is a bit-exact claim with no transcendental anywhere in it,
// so the check is `==` on the raw words, and dropping the guard fails it the
// moment `sin(dist/2)` is non-zero.
PXG_TEST(artiStepSphGuardProbe)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiStep.spv", "artiStepSphGuardProbe", 32u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &artiStepLoader);

    // `dist` is `|w| * dt`.  The cases straddle the 1e-6 threshold deliberately,
    // and the sub-threshold ones are pushed as CLOSE to it as the guard allows:
    // that is where the skipped rotation is largest, hence where a dropped guard
    // is most visible.  `dt` varies per case so `dist` is not a property of `|w|`
    // alone.
    const float dists[] = { 0.0f,      1e-9f,    1e-8f,     1e-7f,   3.7e-7f,
                            5e-7f,     9e-7f,    9.99e-7f,  1e-6f,        // <= threshold
                            1.02e-6f,  2e-6f,    1e-5f,     3.3e-4f, 0.01f,
                            0.37f,     1.5f,     3.0f };
    const uint32_t D = uint32_t(sizeof(dists) / sizeof(dists[0]));
    const uint32_t N = D * 11;   // 11 orientations per distance

    auto val = [](uint32_t i, uint32_t k) {
        return (float((i * 37u + k * 29u) % 401u) + 1.0f) * 0.0078125f - 1.5f;
    };

    std::vector<float> in(size_t(N) * 8);
    uint32_t below = 0, above = 0, zeroAxis = 0;
    std::vector<PxQuat> qs(N);
    std::vector<PxVec3> ws(N);
    std::vector<float> dts(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        const float dist = dists[i % D];
        // a `dt` that is neither 1 nor equal across cases, so `dist` is a
        // PRODUCT rather than a magnitude read straight out of the buffer
        const float dt = 0.0234375f * (1.0f + float(i % 5));
        PxQuat q(val(i, 0), val(i, 1), val(i, 2), val(i, 3) + 1.75f);
        q = q.getNormalized();

        PxVec3 dir(val(i, 4), val(i, 5), val(i, 6));
        if (i % 13 == 6)
        {
            // the OTHER guard in the same expression: `normalize` leaves a zero
            // vector alone, so `dist` is zero and the small-angle branch is the
            // one that must save it
            dir = PxVec3(0.f, 0.f, 0.f);
            ++zeroAxis;
        }
        else
        {
            dir = dir.getNormalized();
        }
        const PxVec3 w = dir * (dist / dt);

        qs[i] = q;
        ws[i] = w;
        dts[i] = dt;
        float* c = in.data() + size_t(i) * 8;
        c[0] = q.x; c[1] = q.y; c[2] = q.z; c[3] = q.w;
        c[4] = w.x; c[5] = w.y; c[6] = w.z;
        c[7] = dt;

        // classify by the value the SHADER will compute, not by the nominal
        // `dist` -- `|w| * dt` need not round back to it exactly
        PxVec3 wn = w;
        const float actual = wn.normalize() * dt;
        if (actual > 1e-6f) ++above; else ++below;
    }
    PXG_CHECK_MSG(below > 20 && above > 20, "the threshold is not straddled (%u below, %u above)",
                  below, above);
    PXG_CHECK(zeroAxis > 5);

    PxgDevicePtr dIn = 0, dOut = 0;
    shim.memAlloc(&dIn, in.size() * sizeof(float));
    shim.memAlloc(&dOut, size_t(N) * 4 * sizeof(float));
    if (!dIn || !dOut)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    shim.memcpyHtoD(dIn, in.data(), in.size() * sizeof(float));

    struct Params { uint64_t inF, outF; uint32_t n, p0, p1, p2; } prm = { uint64_t(dIn),
                                                                          uint64_t(dOut), N, 0, 0,
                                                                          0 };
    PXG_CHECK_MSG(shim.launchKernel(0, (N + 63) / 64, 1, 1, &prm, sizeof(prm))
                      == PxgResult::Success,
                  "launch artiStepSphGuardProbe");
    shim.streamSynchronize();

    std::vector<float> got(size_t(N) * 4);
    shim.memcpyDtoH(got.data(), dOut, got.size() * sizeof(float));

    uint32_t heldExact = 0, heldBad = 0, moved = 0, stuck = 0, maxUlpAbove = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        PxVec3 wn = ws[i];
        const float dist = wn.normalize() * dts[i];
        const float* g = got.data() + size_t(i) * 4;
        const PxQuat& q = qs[i];

        if (dist > 1e-6f)
        {
            // the engine's own arithmetic; `sin`/`cos` puts this a couple of ULP
            // out, which `artiStepQuatProbe` has already measured
            const PxQuat want = PxQuat(dist, wn) * q;
            uint32_t u = 0;
            for (uint32_t k = 0; k < 4; ++k)
            {
                const uint32_t d = stUlpDiff((&want.x)[k], g[k]);
                u = u > d ? u : d;
            }
            maxUlpAbove = maxUlpAbove > u ? maxUlpAbove : u;
            // and the case must actually DO something, or it proves nothing
            if (g[0] != q.x || g[1] != q.y || g[2] != q.z || g[3] != q.w)
                ++moved;
            else
                ++stuck;
        }
        else
        {
            // THE EXACT CLAIM: the guard returns the input untouched.
            if (g[0] == q.x && g[1] == q.y && g[2] == q.z && g[3] == q.w)
                ++heldExact;
            else
            {
                if (heldBad < 8)
                    std::printf("MM sph guard case %u: dist %.6g  in %.9g got %.9g\n", i,
                                double(dist), double(q.x), double(g[0]));
                ++heldBad;
            }
        }
    }
    std::printf("[ info ] sph guard: %u below threshold held bit-exact (%u failed), %u above "
                "moved (%u did not), max %u ulp above\n",
                heldExact, heldBad, moved, stuck, maxUlpAbove);

    PXG_CHECK_MSG(heldBad == 0, "%u sub-threshold cases were rotated anyway", heldBad);
    // Without this the previous check is vacuous: a shader that never rotates at
    // all would pass it.
    PXG_CHECK_MSG(stuck == 0, "%u above-threshold cases were NOT rotated", stuck);
    PXG_CHECK_MSG(maxUlpAbove <= 8, "the above-threshold rotation drifted: %u ulp", maxUlpAbove);

    shim.memFree(dIn);
    shim.memFree(dOut);
}

}  // namespace
