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


// npArtiSetupTendons.slang
//
// P8.97b: `setupInternalSpatialTendonConstraints` (internalConstraints2.cu:
// 824-1012), `setupInternalFixedTendonConstraints` (:1015-1198), and the
// link-pair impulse-response family they call -- the dispatcher (:787), the
// all-dof fast path (:95), and `getImpulseSelfResponseSlow` (:658), the
// common-ancestor walk over the dof blocks' mTmpQstZ scratch.
//
// ZERO new ABI beyond the P8.97b groundwork commit: the attachment /
// tendon-joint / tendon blocks are P8.77/78's maps, the ITC output block and
// the two descriptor pointers were generated and pinned in the groundwork,
// and the propagate algebra is P8.90's module (WriteToTempQstZ/
// ReadFromTempQstZ become a load of mTmpQstZ into the qstZ[3] array around
// the SAME ad* calls -- the .cuh policies accumulate the identical
// jf - innerProduct term, verified against articulationDynamic.cuh:47/129).
//
// TRAPS, from the reading (each a fixture case / mutant):
//  1. The bit-stack DFS: child = 63 - clz64(bitStack) (HIGHEST set bit
//     first); at the loop bottom the current child bit is cleared, the
//     children mask OR-ed in, the next child popped, and parent re-read from
//     the NEXT child's mParents -- all only under stackCount > 0.
//  2. SPATIAL: only LEAF attachments (mChildrens == 0) emit a constraint;
//     non-leaves store accumLength[child] = u instead.  FIXED: EVERY tendon
//     joint emits one, and writes its mConstraintId BEFORE numConstraints++.
//  3. SPATIAL cfm = the CHILD link's mCfm alone; FIXED cfm =
//     max(parent link, child link) -- deliberately different.
//  4. SPATIAL recipResponse is guarded by DY_ARTICULATION_MIN_RESPONSE
//     (1e-12) and yields 0 below it; FIXED recipResponse is UNGUARDED.
//  5. SPATIAL stores mDeltaVA = r0 and NO mDeltaVB (the commented-out stores
//     in the CUDA are the trap); FIXED stores mDeltaVA = r0 AND mDeltaVB.
//  6. unitResponse = r0 - r1 (minus), with r0/r1 spelled as the explicit
//     bottom-dot + top-dot pairs against the UNNEGATED axes.
//  7. The root axis state (rAxis/rRaXn resp. sAxis/sRaXn) is LOOP-CARRIED:
//     written only on parent == 0 iterations, read at every emission.  The
//     first popped child always has parent 0, so it is initialized before
//     any read; a leaf under a deeper parent reuses the LAST root-child's
//     axis -- transcribed, not tidied.
//  8. accumLength[] entries are written root-first (index 0 at entry, each
//     non-leaf when visited), so accumLength[parent] is always initialized
//     before its children read it -- the zero-init here is a dead store for
//     the compiler's benefit.
//  9. The spatial axis guards are `distance > 0.001f`, not > 0.
// 10. FIXED root branch: angular tjAxis (< eX(3)) gives sAxis = 0, sRaXn =
//     worldMotion.top; linear gives the FULL transform compose
//     (sBody2World.transform(parentPose)) and ang0 = (cA2w.p - sBody2World.p)
//     x worldMotion.bottom.  The per-joint branch uses cBody2World and
//     mChildPose the same way.
// 11. The fixed root's sBody2World is loaded ONCE from tendon joint 0's link
//     before the loop, even though the per-joint branch reloads cLink's pose.
// 12. FIXED reads the dof through mInvDofIds[tjAxis] (a 6x32 PxU8 table,
//     row stride 32) added to the link's mJointOffset.
// 13. getImpulseSelfResponseSlow zeroes mTmpQstZ for EVERY dof of EVERY link
//     first, pushes both root-ward paths onto ONE stack (i0 entries then
//     i1 - i0), applies the COMMON link's 6x6 response to -(Z0 + Z1), and
//     walks the two stack slices in reverse -- deltaV1 from the [i0, i1)
//     slice, deltaV0 from [0, i0).
// 14. The dispatcher takes the fast path only when link1's parent == linkId0.
// 15. Slow negates BOTH impulses into Z0/Z1 (the dof-aligned fast path of
//     P8.97a negates only impulse1).
//
// The tendon scalar family (a, x, velMultiplier, biasCoefficient, the TGS
// impulse-multiplier select) is spelled inline exactly as the CUDA spells
// it -- it is NOT the joint-drive family lifted in P8.97a (different guard:
// unitResponse > 0 on x, stepDt everywhere, no simDt bias).

import pxgSlangCommon;
import pxgSlangArticulationBlocks;
import pxgSlangSpatialMatrix;
import pxgSlangArticulationDynamic;

static const uint ST_AXIS_EX = 3u;   // PxArticulationAxis::eX

uint stPopc64(uint64_t v)
{
    return countbits(uint(v)) + countbits(uint(v >> 32));
}

// 63 - __clzll(v) == the index of the highest set bit
uint stHighestBit(uint64_t v)
{
    const uint hi = uint(v >> 32);
    return hi != 0u ? 32u + firstbithigh(hi) : firstbithigh(uint(v));
}

AbSpatialVector stLoadSV(uint64_t base, uint fieldOffset, uint lane)
{
    float3 top, bottom;
    abLoadSpatialVector(base, fieldOffset, lane, top, bottom);
    return asmSVec(top, bottom);
}

PxgTransform28 stLoadPose(uint64_t link, uint fieldOffset, uint lane)
{
    const float4 q = abF4(link, fieldOffset + PXG_AB_STB_Q, lane);
    const float4 p = abF4(link, fieldOffset + PXG_AB_STB_P, lane);
    return makeTransform28(q, float3(p.x, p.y, p.z));
}

// getImpulseSelfResponse (internalConstraints2.cu:95): the all-dof fast path
// for a direct parent-child pair.  dofBase = the CHILD link's first dof.
void stImpulseSelfResponseFast(AbSpatialVector impulse0, AbSpatialVector impulse1,
                               out AbSpatialVector deltaV0, out AbSpatialVector deltaV1,
                               uint64_t dofBase, float3 childToParent,
                               AbMatrix6x6 parentResponse, uint dofCount, uint lane)
{
    const AbSpatialVector Z1W = asmSVec(-impulse1.top, -impulse1.bottom);

    AbSpatialVector isInvD[3];
    AbSpatialVector motionMatrix[3];
    AbSpatialVector isW[3];
    float3 invStIsT[3];
    for (uint ind = 0u; ind < 3u; ++ind)
    {
        isInvD[ind] = asmSVec(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f));
        motionMatrix[ind] = isInvD[ind];
        isW[ind] = isInvD[ind];
        invStIsT[ind] = float3(0.0f, 0.0f, 0.0f);
        if (ind < dofCount)
        {
            const uint64_t d = dofBase + uint64_t(ind) * PXG_AB_DOF_SIZE;
            isInvD[ind] = stLoadSV(d, PXG_AB_DOF_mIsInvDW, lane);
            motionMatrix[ind] = stLoadSV(d, PXG_AB_DOF_mWorldMotionMatrix, lane);
            isW[ind] = stLoadSV(d, PXG_AB_DOF_mIsW, lane);
            invStIsT[ind] = float3(abF32(d, PXG_AB_DOF_mInvStIsT_x, lane),
                                   abF32(d, PXG_AB_DOF_mInvStIsT_y, lane),
                                   abF32(d, PXG_AB_DOF_mInvStIsT_z, lane));
        }
    }

    float qstZ[3] = { 0.0f, 0.0f, 0.0f };
    const AbSpatialVector Z0W =
        adPropagateImpulseW(isInvD, childToParent, motionMatrix, Z1W, dofCount, qstZ);

    const AbSpatialVector impulseDifW = asmSVecSub(impulse0, Z0W);
    const AbSpatialVector deltaV0W = asmM6MulVec(parentResponse, impulseDifW);

    deltaV1 = adPropagateAccelerationW(childToParent, invStIsT, motionMatrix, deltaV0W,
                                       dofCount, isW, qstZ);
    deltaV0 = deltaV0W;
}

// one hop of propagateImpulseW<WriteToTempQstZ>: the ad* call accumulates
// into a local array seeded from mTmpQstZ, stored back after -- the same
// jf(=0) - innerProduct term the .cuh policy adds
AbSpatialVector stPropagateImpulseTempQstZ(uint64_t linkBase, uint64_t dofBlocks, uint linkID,
                                           AbSpatialVector Z, uint lane)
{
    const uint64_t link = linkBase + uint64_t(linkID) * PXG_AB_LINK_SIZE;
    const uint jointOffset = abU32(link, PXG_AB_LINK_mJointOffset, lane);
    const uint dofCount = abU8(link, PXG_AB_LINK_mDofs, lane);
    const uint64_t dofBase = dofBlocks + uint64_t(jointOffset) * PXG_AB_DOF_SIZE;
    const float3 childToParent = float3(abF32(link, PXG_AB_LINK_mRw_x, lane),
                                        abF32(link, PXG_AB_LINK_mRw_y, lane),
                                        abF32(link, PXG_AB_LINK_mRw_z, lane));

    AbSpatialVector isInvD[3];
    AbSpatialVector motionMatrix[3];
    float qstZ[3] = { 0.0f, 0.0f, 0.0f };
    for (uint ind = 0u; ind < 3u; ++ind)
    {
        isInvD[ind] = asmSVec(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f));
        motionMatrix[ind] = isInvD[ind];
        if (ind < dofCount)
        {
            const uint64_t d = dofBase + uint64_t(ind) * PXG_AB_DOF_SIZE;
            isInvD[ind] = stLoadSV(d, PXG_AB_DOF_mIsInvDW, lane);
            motionMatrix[ind] = stLoadSV(d, PXG_AB_DOF_mWorldMotionMatrix, lane);
            qstZ[ind] = abF32(d, PXG_AB_DOF_mTmpQstZ, lane);
        }
    }

    const AbSpatialVector outZ =
        adPropagateImpulseW(isInvD, childToParent, motionMatrix, Z, dofCount, qstZ);

    for (uint ind = 0u; ind < 3u; ++ind)
        if (ind < dofCount)
        {
            const uint64_t d = dofBase + uint64_t(ind) * PXG_AB_DOF_SIZE;
            abStoreF32(d, PXG_AB_DOF_mTmpQstZ, lane, qstZ[ind]);
        }

    return outZ;
}

// one hop of propagateAccelerationW_child2ParentPreLoaded<ReadFromTempQstZ>
AbSpatialVector stPropagateAccelTempQstZ(uint64_t linkBase, uint64_t dofBlocks, uint linkID,
                                         AbSpatialVector deltaV, uint lane)
{
    const uint64_t link = linkBase + uint64_t(linkID) * PXG_AB_LINK_SIZE;
    const uint jointOffset = abU32(link, PXG_AB_LINK_mJointOffset, lane);
    const uint dofCount = abU8(link, PXG_AB_LINK_mDofs, lane);
    const uint64_t dofBase = dofBlocks + uint64_t(jointOffset) * PXG_AB_DOF_SIZE;
    const float3 childToParent = float3(abF32(link, PXG_AB_LINK_mRw_x, lane),
                                        abF32(link, PXG_AB_LINK_mRw_y, lane),
                                        abF32(link, PXG_AB_LINK_mRw_z, lane));

    AbSpatialVector isW[3];
    AbSpatialVector motionMatrix[3];
    float3 invStIsT[3];
    float qstZ[3] = { 0.0f, 0.0f, 0.0f };
    for (uint ind = 0u; ind < 3u; ++ind)
    {
        isW[ind] = asmSVec(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f));
        motionMatrix[ind] = isW[ind];
        invStIsT[ind] = float3(0.0f, 0.0f, 0.0f);
        if (ind < dofCount)
        {
            const uint64_t d = dofBase + uint64_t(ind) * PXG_AB_DOF_SIZE;
            isW[ind] = stLoadSV(d, PXG_AB_DOF_mIsW, lane);
            motionMatrix[ind] = stLoadSV(d, PXG_AB_DOF_mWorldMotionMatrix, lane);
            invStIsT[ind] = float3(abF32(d, PXG_AB_DOF_mInvStIsT_x, lane),
                                   abF32(d, PXG_AB_DOF_mInvStIsT_y, lane),
                                   abF32(d, PXG_AB_DOF_mInvStIsT_z, lane));
            qstZ[ind] = abF32(d, PXG_AB_DOF_mTmpQstZ, lane);
        }
    }

    return adPropagateAccelerationW(childToParent, invStIsT, motionMatrix, deltaV,
                                    dofCount, isW, qstZ);
}

// getImpulseSelfResponseSlow (internalConstraints2.cu:658)
void stImpulseSelfResponseSlow(uint linkID0_, uint linkID1_,
                               AbSpatialVector impulse0, AbSpatialVector impulse1,
                               out AbSpatialVector deltaV0, out AbSpatialVector deltaV1,
                               uint64_t artiBlock, uint64_t linkBase, uint64_t dofBlocks,
                               uint lane)
{
    uint stack[PXG_DY_ARTICULATION_TENDON_MAX_SIZE];

    uint linkID0 = linkID0_;
    uint linkID1 = linkID1_;

    // find the common ancestor (parents are strictly smaller than children)
    uint i0 = linkID0, i1 = linkID1;
    while (i0 != i1)
    {
        if (i0 < i1)
            i1 = abU32(linkBase + uint64_t(i1) * PXG_AB_LINK_SIZE, PXG_AB_LINK_mParents, lane);
        else
            i0 = abU32(linkBase + uint64_t(i0) * PXG_AB_LINK_SIZE, PXG_AB_LINK_mParents, lane);
    }
    const uint common = i0;

    // trap 15: BOTH impulses negated here
    AbSpatialVector Z0 = asmSVec(-impulse0.top, -impulse0.bottom);
    AbSpatialVector Z1 = asmSVec(-impulse1.top, -impulse1.bottom);

    // zero the mTmpQstZ scratch of every dof of every link
    const uint numLinks = abU32(artiBlock, PXG_AB_mNumLinks, lane);
    for (uint i = 0u; i < numLinks; ++i)
    {
        const uint64_t link = linkBase + uint64_t(i) * PXG_AB_LINK_SIZE;
        const uint jointOffset = abU32(link, PXG_AB_LINK_mJointOffset, lane);
        const uint dofCount = abU8(link, PXG_AB_LINK_mDofs, lane);
        for (uint j = 0u; j < dofCount; ++j)
            abStoreF32(dofBlocks + uint64_t(jointOffset + j) * PXG_AB_DOF_SIZE,
                       PXG_AB_DOF_mTmpQstZ, lane, 0.0f);
    }

    i0 = 0u;
    while (linkID0 != common)
    {
        Z0 = stPropagateImpulseTempQstZ(linkBase, dofBlocks, linkID0, Z0, lane);
        stack[i0++] = linkID0;
        linkID0 = abU32(linkBase + uint64_t(linkID0) * PXG_AB_LINK_SIZE,
                        PXG_AB_LINK_mParents, lane);
    }

    i1 = i0;
    while (linkID1 != common)
    {
        Z1 = stPropagateImpulseTempQstZ(linkBase, dofBlocks, linkID1, Z1, lane);
        stack[i1++] = linkID1;
        linkID1 = abU32(linkBase + uint64_t(linkID1) * PXG_AB_LINK_SIZE,
                        PXG_AB_LINK_mParents, lane);
    }

    const AbSpatialVector ZZ = asmSVecAdd(Z0, Z1);

    const AbMatrix6x6 spatialResponse =
        abLoadSpatialMatrix6x6(linkBase + uint64_t(common) * PXG_AB_LINK_SIZE,
                               PXG_AB_LINK_mSpatialResponseMatrix, lane);
    const AbSpatialVector v = asmM6MulVec(spatialResponse, asmSVec(-ZZ.top, -ZZ.bottom));

    AbSpatialVector dv1 = v;
    for (uint index = i1; index > i0;)
    {
        --index;
        dv1 = stPropagateAccelTempQstZ(linkBase, dofBlocks, stack[index], dv1, lane);
    }

    AbSpatialVector dv0 = v;
    for (uint index = i0; index > 0u;)
    {
        --index;
        dv0 = stPropagateAccelTempQstZ(linkBase, dofBlocks, stack[index], dv0, lane);
    }

    deltaV0 = dv0;
    deltaV1 = dv1;
}

// the dispatcher (internalConstraints2.cu:787)
void stImpulseSelfResponse(uint linkId0, uint linkId1,
                           AbSpatialVector impulse0, AbSpatialVector impulse1,
                           out AbSpatialVector deltaV0, out AbSpatialVector deltaV1,
                           uint64_t artiBlock, uint64_t linkBase, uint64_t dofBlocks,
                           uint lane)
{
    const uint64_t link1 = linkBase + uint64_t(linkId1) * PXG_AB_LINK_SIZE;
    if (abU32(link1, PXG_AB_LINK_mParents, lane) == linkId0)
    {
        const uint64_t link0 = linkBase + uint64_t(linkId0) * PXG_AB_LINK_SIZE;
        const AbMatrix6x6 parentResponse =
            abLoadSpatialMatrix6x6(link0, PXG_AB_LINK_mSpatialResponseMatrix, lane);
        const uint jointOffset = abU32(link1, PXG_AB_LINK_mJointOffset, lane);
        const uint dofCount = abU8(link1, PXG_AB_LINK_mDofs, lane);
        const float3 childToParent = float3(abF32(link1, PXG_AB_LINK_mRw_x, lane),
                                            abF32(link1, PXG_AB_LINK_mRw_y, lane),
                                            abF32(link1, PXG_AB_LINK_mRw_z, lane));
        stImpulseSelfResponseFast(impulse0, impulse1, deltaV0, deltaV1,
                                  dofBlocks + uint64_t(jointOffset) * PXG_AB_DOF_SIZE,
                                  childToParent, parentResponse, dofCount, lane);
    }
    else
    {
        stImpulseSelfResponseSlow(linkId0, linkId1, impulse0, impulse1, deltaV0, deltaV1,
                                  artiBlock, linkBase, dofBlocks, lane);
    }
}

// setupInternalSpatialTendonConstraints (internalConstraints2.cu:824)
public void stSetupSpatialTendons(uint64_t artiBlock, uint64_t linkBase, uint64_t dofBlocks,
                                  uint64_t tendonBase, uint64_t constraintBase,
                                  uint64_t attachmentBase, uint maxAttachments,
                                  float stepDt, float dt, float invDt, bool isTGSSolver,
                                  uint lane)
{
    const uint numTendons = abU32(artiBlock, PXG_AB_mNumSpatialTendons, lane);

    // trap 8: entries are written before any read; the init is a dead store
    float accumLength[PXG_DY_ARTICULATION_TENDON_MAX_SIZE];
    for (uint k = 0u; k < PXG_DY_ARTICULATION_TENDON_MAX_SIZE; ++k)
        accumLength[k] = 0.0f;

    for (uint i = 0u; i < numTendons; ++i)
    {
        const uint64_t tendon = tendonBase + uint64_t(i) * PXG_AB_ST_SIZE;
        const float stiffness = abF32(tendon, PXG_AB_ST_mStiffness, lane);
        const float damping = abF32(tendon, PXG_AB_ST_mDamping, lane);
        const float limitStiffness = abF32(tendon, PXG_AB_ST_mLimitStiffness, lane);
        const float offset = abF32(tendon, PXG_AB_ST_mOffset, lane);

        const uint64_t attachmentBlock = attachmentBase
                                       + uint64_t(i) * uint64_t(maxAttachments) * PXG_AB_AT_SIZE;
        const uint64_t constraintBlock = constraintBase
                                       + uint64_t(i) * uint64_t(maxAttachments) * PXG_AB_ITC_SIZE;

        const float coefficient = abF32(attachmentBlock, PXG_AB_AT_mCoefficient, lane);
        uint64_t bitStack = abU64(attachmentBlock, PXG_AB_AT_mChildrens, lane);
        uint stackCount = stPopc64(bitStack);

        uint parent = 0u;
        uint numConstraints = 0u;

        accumLength[parent] = offset * coefficient;

        const uint rAttachmentLinkIndex = abU32(attachmentBlock, PXG_AB_AT_mLinkIndex, lane);
        const uint64_t rAttachmentLink = linkBase
                                       + uint64_t(rAttachmentLinkIndex) * PXG_AB_LINK_SIZE;
        const PxgTransform28 rB2w = stLoadPose(rAttachmentLink, PXG_AB_LINK_mAccumulatedPose, lane);
        const float3 rRelOffset = abF3(attachmentBlock, PXG_AB_AT_mRelativeOffset, lane);
        const float3 rRa = quatRotate(rB2w.q(), rRelOffset);
        const float3 rAttachPoint = rB2w.p() + rRa;
        // trap 7: loop-carried root axis state
        float3 rAxis = float3(0.0f, 0.0f, 0.0f);
        float3 rRaXn = float3(0.0f, 0.0f, 0.0f);

        uint child = stHighestBit(bitStack);

        while (stackCount != 0u)
        {
            stackCount--;

            const uint64_t attachment = attachmentBlock + uint64_t(child) * PXG_AB_AT_SIZE;

            const uint linkInd = abU32(attachment, PXG_AB_AT_mLinkIndex, lane);
            const uint64_t cLink = linkBase + uint64_t(linkInd) * PXG_AB_LINK_SIZE;

            // trap 3: the CHILD's cfm alone
            const float cfm = abF32(cLink, PXG_AB_LINK_mCfm, lane);

            const PxgTransform28 cB2w = stLoadPose(cLink, PXG_AB_LINK_mAccumulatedPose, lane);
            const float3 rb = quatRotate(cB2w.q(), abF3(attachment, PXG_AB_AT_mRelativeOffset, lane));
            const float3 cAttachPoint = cB2w.p() + rb;

            float distance = 0.0f;
            float3 dif = float3(0.0f, 0.0f, 0.0f);
            if (parent == 0u)
            {
                dif = rAttachPoint - cAttachPoint;
                const float distanceSq = dot(dif, dif);
                distance = sqrt(distanceSq);

                // trap 9: the 0.001 threshold
                rAxis = distance > 0.001f ? (dif / distance) : float3(0.0f, 0.0f, 0.0f);
                rRaXn = cross(rRa, rAxis);
            }
            else
            {
                const uint64_t pAttachment = attachmentBlock + uint64_t(parent) * PXG_AB_AT_SIZE;
                const uint pLinkInd = abU32(pAttachment, PXG_AB_AT_mLinkIndex, lane);
                const uint64_t pLink = linkBase + uint64_t(pLinkInd) * PXG_AB_LINK_SIZE;

                const PxgTransform28 pB2w = stLoadPose(pLink, PXG_AB_LINK_mAccumulatedPose, lane);
                const float3 ra = quatRotate(pB2w.q(), abF3(pAttachment, PXG_AB_AT_mRelativeOffset, lane));
                const float3 pAttachPoint = pB2w.p() + ra;

                dif = pAttachPoint - cAttachPoint;
                const float distanceSq = dot(dif, dif);
                distance = sqrt(distanceSq);
            }

            const float u = distance * abF32(attachment, PXG_AB_AT_mCoefficient, lane)
                          + accumLength[parent];

            const uint64_t children = abU64(attachment, PXG_AB_AT_mChildrens, lane);

            if (children != 0uL)
            {
                stackCount += stPopc64(children);
                accumLength[child] = u;
            }
            else
            {
                // trap 2: only leaves emit constraints
                const float3 axis = distance > 0.001f ? (dif / distance)
                                                      : float3(0.0f, 0.0f, 0.0f);
                const float3 rbXn = cross(rb, axis);

                const AbSpatialVector axis0 = asmSVec(rAxis, rRaXn);
                const AbSpatialVector axis1 = asmSVec(-axis, -rbXn);

                AbSpatialVector deltaV0, deltaV1;
                stImpulseSelfResponse(rAttachmentLinkIndex, linkInd, axis0, axis1,
                                      deltaV0, deltaV1, artiBlock, linkBase, dofBlocks, lane);

                // trap 6: explicit dot pairs, minus
                const float r0 = dot(deltaV0.bottom, rAxis) + dot(deltaV0.top, rRaXn);
                const float r1 = dot(deltaV1.bottom, axis) + dot(deltaV1.top, rbXn);

                const float unitResponse = r0 - r1;

                // trap 4: the MIN_RESPONSE guard, 0 below it
                const float recipResponse = unitResponse > PXG_DY_ARTICULATION_MIN_RESPONSE
                                                ? (1.0f / (unitResponse + cfm)) : 0.0f;

                const uint64_t constraint = constraintBlock
                                          + uint64_t(numConstraints) * PXG_AB_ITC_SIZE;
                numConstraints++;

                abStoreSpatialVector(constraint, PXG_AB_ITC_mRow0, lane, axis0.top, axis0.bottom);
                abStoreSpatialVector(constraint, PXG_AB_ITC_mRow1, lane, -axis1.top, -axis1.bottom);

                // trap 5: mDeltaVA = r0, and NO mDeltaVB store
                abStoreF32(constraint, PXG_AB_ITC_mDeltaVA, lane, r0);

                abStoreF32(constraint, PXG_AB_ITC_mRecipResponse, lane, recipResponse);

                const float a = stepDt * (stepDt * stiffness + damping);
                const float a2 = stepDt * (stepDt * limitStiffness + damping);
                const float x = unitResponse > 0.0f ? 1.0f / (1.0f + a * unitResponse) : 0.0f;
                const float x2 = unitResponse > 0.0f ? 1.0f / (1.0f + a2 * unitResponse) : 0.0f;

                abStoreF32(constraint, PXG_AB_ITC_mVelMultiplier, lane, -x * a);
                abStoreF32(constraint, PXG_AB_ITC_mImpulseMultiplier, lane,
                           isTGSSolver ? 1.0f : 1.0f - x);
                abStoreF32(constraint, PXG_AB_ITC_mBiasCoefficient, lane,
                           -stiffness * x * stepDt);
                abStoreF32(constraint, PXG_AB_ITC_mAppliedForce, lane, 0.0f);

                abStoreF32(constraint, PXG_AB_ITC_mAccumulatedLength, lane, u);
                abStoreU32(constraint, PXG_AB_ITC_mLink0, lane, rAttachmentLinkIndex);
                abStoreU32(constraint, PXG_AB_ITC_mLink1, lane, linkInd);

                abStoreF32(constraint, PXG_AB_ITC_mLimitImpulseMultiplier, lane,
                           isTGSSolver ? 1.0f : 1.0f - x2);
                abStoreF32(constraint, PXG_AB_ITC_mLimitBiasCoefficient, lane,
                           -limitStiffness * x2 * stepDt);
                abStoreF32(constraint, PXG_AB_ITC_mLimitAppliedForce, lane, 0.0f);

                abStoreF32(constraint, PXG_AB_ITC_mRestDistance, lane,
                           abF32(attachment, PXG_AB_AT_mRestDistance, lane));
                abStoreF32(constraint, PXG_AB_ITC_mLowLimit, lane,
                           abF32(attachment, PXG_AB_AT_mLowLimit, lane));
                abStoreF32(constraint, PXG_AB_ITC_mHighLimit, lane,
                           abF32(attachment, PXG_AB_AT_mHighLimit, lane));
            }

            if (stackCount > 0u)
            {
                // trap 1: clear, merge, pop, re-read parent -- in this order
                bitStack &= ~(uint64_t(1) << child);
                bitStack |= children;
                child = stHighestBit(bitStack);
                parent = abU32(attachmentBlock + uint64_t(child) * PXG_AB_AT_SIZE,
                               PXG_AB_AT_mParents, lane);
            }
        }

        abStoreU32(tendon, PXG_AB_ST_mNumConstraints, lane, numConstraints);
    }
}

// setupInternalFixedTendonConstraints (internalConstraints2.cu:1015)
public void stSetupFixedTendons(uint64_t artiBlock, uint64_t linkBase, uint64_t dofBlocks,
                                uint64_t tendonBase, uint64_t constraintBase,
                                uint64_t tendonJointBase, uint maxTendonJoints,
                                float stepDt, float dt, float invDt, bool isTGSSolver,
                                uint lane)
{
    const uint numTendons = abU32(artiBlock, PXG_AB_mNumFixedTendons, lane);

    for (uint i = 0u; i < numTendons; ++i)
    {
        const uint64_t tendon = tendonBase + uint64_t(i) * PXG_AB_FT_SIZE;
        const float stiffness = abF32(tendon, PXG_AB_FT_mStiffness, lane);
        const float damping = abF32(tendon, PXG_AB_FT_mDamping, lane);
        const float limitStiffness = abF32(tendon, PXG_AB_FT_mLimitStiffness, lane);

        const uint64_t tjBlock = tendonJointBase
                               + uint64_t(i) * uint64_t(maxTendonJoints) * PXG_AB_TJ_SIZE;
        const uint64_t constraintBlock = constraintBase
                                       + uint64_t(i) * uint64_t(maxTendonJoints) * PXG_AB_ITC_SIZE;

        uint64_t bitStack = abU64(tjBlock, PXG_AB_TJ_mChildrens, lane);
        uint stackCount = stPopc64(bitStack);

        uint parent = 0u;
        uint numConstraints = 0u;

        // trap 11: the root link pose, loaded once before the loop
        const uint sLinkIndex = abU32(tjBlock, PXG_AB_TJ_mLinkIndex, lane);
        const uint64_t sLink = linkBase + uint64_t(sLinkIndex) * PXG_AB_LINK_SIZE;
        const PxgTransform28 sB2w = stLoadPose(sLink, PXG_AB_LINK_mAccumulatedPose, lane);

        // trap 7: loop-carried root axis state
        float3 sAxis = float3(0.0f, 0.0f, 0.0f);
        float3 sRaXn = float3(0.0f, 0.0f, 0.0f);

        uint child = stHighestBit(bitStack);

        while (stackCount != 0u)
        {
            stackCount--;

            const uint64_t tj = tjBlock + uint64_t(child) * PXG_AB_TJ_SIZE;

            const uint tjAxis = abU32(tj, PXG_AB_TJ_mAxis, lane);

            const uint cLinkInd = abU32(tj, PXG_AB_TJ_mLinkIndex, lane);
            const uint64_t cLink = linkBase + uint64_t(cLinkInd) * PXG_AB_LINK_SIZE;

            // trap 3: max(parent link, child link) here
            const uint parentLink = abU32(cLink, PXG_AB_LINK_mParents, lane);
            const float pCfm = abF32(linkBase + uint64_t(parentLink) * PXG_AB_LINK_SIZE,
                                     PXG_AB_LINK_mCfm, lane);
            const float lCfm = abF32(cLink, PXG_AB_LINK_mCfm, lane);
            const float cfm = pCfm > lCfm ? pCfm : lCfm;

            const uint jointOffset = abU32(cLink, PXG_AB_LINK_mJointOffset, lane);
            // trap 12: the 6x32 PxU8 inverse-dof table, row stride 32
            const uint dofIndex = abU8(cLink, PXG_AB_LINK_mInvDofIds + tjAxis * 32u, lane);
            const uint64_t dofData = dofBlocks
                                   + uint64_t(jointOffset + dofIndex) * PXG_AB_DOF_SIZE;

            const AbSpatialVector worldMotionVector =
                stLoadSV(dofData, PXG_AB_DOF_mWorldMotionMatrix, lane);

            if (parent == 0u)
            {
                if (tjAxis < ST_AXIS_EX)
                {
                    sAxis = float3(0.0f, 0.0f, 0.0f);
                    sRaXn = worldMotionVector.top;
                }
                else
                {
                    // trap 10: full transform compose for the linear root
                    const PxgTransform28 parentPose =
                        stLoadPose(cLink, PXG_AB_LINK_mParentPose, lane);
                    const PxgTransform28 cA2w = transformTransform(sB2w, parentPose);
                    const float3 ang0 = cross(cA2w.p() - sB2w.p(), worldMotionVector.bottom);
                    sAxis = worldMotionVector.bottom;
                    sRaXn = ang0;
                }
            }

            const uint64_t children = abU64(tj, PXG_AB_TJ_mChildrens, lane);

            float3 axis, rbXn;
            if (tjAxis < ST_AXIS_EX)
            {
                axis = float3(0.0f, 0.0f, 0.0f);
                rbXn = worldMotionVector.top;
            }
            else
            {
                const PxgTransform28 cB2w = stLoadPose(cLink, PXG_AB_LINK_mAccumulatedPose, lane);
                const PxgTransform28 childPose = stLoadPose(cLink, PXG_AB_LINK_mChildPose, lane);
                const PxgTransform28 cB2wJoint = transformTransform(cB2w, childPose);
                axis = worldMotionVector.bottom;
                rbXn = cross(cB2wJoint.p() - cB2w.p(), axis);
            }

            const AbSpatialVector axis0 = asmSVec(sAxis, sRaXn);
            const AbSpatialVector axis1 = asmSVec(-axis, -rbXn);

            AbSpatialVector deltaV0, deltaV1;
            stImpulseSelfResponse(sLinkIndex, cLinkInd, axis0, axis1,
                                  deltaV0, deltaV1, artiBlock, linkBase, dofBlocks, lane);

            const float r0 = dot(deltaV0.bottom, sAxis) + dot(deltaV0.top, sRaXn);
            const float r1 = dot(deltaV1.bottom, axis) + dot(deltaV1.top, rbXn);

            const float unitResponse = r0 - r1;

            // trap 4: UNGUARDED here
            const float recipResponse = 1.0f / (unitResponse + cfm);

            const uint64_t constraint = constraintBlock
                                      + uint64_t(numConstraints) * PXG_AB_ITC_SIZE;

            // trap 5: BOTH mDeltaVA and mDeltaVB here
            abStoreF32(constraint, PXG_AB_ITC_mDeltaVA, lane, r0);
            abStoreSpatialVector(constraint, PXG_AB_ITC_mDeltaVB, lane,
                                 deltaV1.top, deltaV1.bottom);
            abStoreSpatialVector(constraint, PXG_AB_ITC_mRow0, lane, axis0.top, axis0.bottom);
            abStoreSpatialVector(constraint, PXG_AB_ITC_mRow1, lane, -axis1.top, -axis1.bottom);

            abStoreF32(constraint, PXG_AB_ITC_mRecipResponse, lane, recipResponse);

            const float a = stepDt * (stepDt * stiffness + damping);
            const float a2 = stepDt * (stepDt * limitStiffness + damping);
            const float x = unitResponse > 0.0f ? 1.0f / (1.0f + a * unitResponse) : 0.0f;
            const float x2 = unitResponse > 0.0f ? 1.0f / (1.0f + a2 * unitResponse) : 0.0f;

            abStoreF32(constraint, PXG_AB_ITC_mVelMultiplier, lane, -x * a);
            abStoreF32(constraint, PXG_AB_ITC_mImpulseMultiplier, lane,
                       isTGSSolver ? 1.0f : 1.0f - x);
            abStoreF32(constraint, PXG_AB_ITC_mBiasCoefficient, lane, -stiffness * x * stepDt);
            abStoreF32(constraint, PXG_AB_ITC_mAppliedForce, lane, 0.0f);

            abStoreU32(constraint, PXG_AB_ITC_mLink0, lane, sLinkIndex);
            abStoreU32(constraint, PXG_AB_ITC_mLink1, lane, cLinkInd);

            abStoreF32(constraint, PXG_AB_ITC_mLimitImpulseMultiplier, lane,
                       isTGSSolver ? 1.0f : 1.0f - x2);
            abStoreF32(constraint, PXG_AB_ITC_mLimitBiasCoefficient, lane,
                       -limitStiffness * x2 * stepDt);
            abStoreF32(constraint, PXG_AB_ITC_mLimitAppliedForce, lane, 0.0f);

            // trap 2: mConstraintId assigned BEFORE the increment, every joint
            abStoreU32(tj, PXG_AB_TJ_mConstraintId, lane, numConstraints);
            numConstraints++;

            if (children != 0uL)
                stackCount += stPopc64(children);

            if (stackCount > 0u)
            {
                bitStack &= ~(uint64_t(1) << child);
                bitStack |= children;
                child = stHighestBit(bitStack);
                parent = abU32(tjBlock + uint64_t(child) * PXG_AB_TJ_SIZE,
                               PXG_AB_TJ_mParents, lane);
            }
        }

        abStoreU32(tendon, PXG_AB_FT_mNumConstraints, lane, numConstraints);
    }
}

// ---------------------------------------------------------------------------
// fixture entry points: one articulation per lane, one warp per group; block
// bases handed in directly (the wrapper's index chain is P8.97c's job)
// ---------------------------------------------------------------------------

struct StParams
{
    uint64_t artiBlocks;      // @0
    uint64_t linkBlocks;      // @8
    uint64_t dofBlocks;       // @16
    uint64_t tendonBlocks;    // @24
    uint64_t constraintBlocks;// @32
    uint64_t elementBlocks;   // @40 attachments (spatial) / tendon joints (fixed)
    float stepDt;             // @48
    float dt;                 // @52
    float invDt;              // @56
    uint isTGS;               // @60
    uint nbArticulations;     // @64
    uint maxLinks;            // @68
    uint maxDofs;             // @72
    uint maxTendons;          // @76
    uint maxElements;         // @80
    uint pad0;                // @84 -> 88
};

[[vk::push_constant]]
ConstantBuffer<StParams> gSt;

[shader("compute")]
[numthreads(32, 1, 1)]
void stSetupSpatialTest(uint3 groupId : SV_GroupID, uint3 groupThreadId : SV_GroupThreadID)
{
    const uint w = groupId.x;
    const uint lane = groupThreadId.x;
    if (w * 32u + lane >= gSt.nbArticulations)
        return;

    stSetupSpatialTendons(gSt.artiBlocks + uint64_t(w) * PXG_AB_BLOCK_SIZE,
                          gSt.linkBlocks + uint64_t(w) * uint64_t(gSt.maxLinks) * PXG_AB_LINK_SIZE,
                          gSt.dofBlocks + uint64_t(w) * uint64_t(gSt.maxDofs) * PXG_AB_DOF_SIZE,
                          gSt.tendonBlocks + uint64_t(w) * uint64_t(gSt.maxTendons) * PXG_AB_ST_SIZE,
                          gSt.constraintBlocks + uint64_t(w) * uint64_t(gSt.maxTendons)
                              * uint64_t(gSt.maxElements) * PXG_AB_ITC_SIZE,
                          gSt.elementBlocks + uint64_t(w) * uint64_t(gSt.maxTendons)
                              * uint64_t(gSt.maxElements) * PXG_AB_AT_SIZE,
                          gSt.maxElements, gSt.stepDt, gSt.dt, gSt.invDt, gSt.isTGS != 0u, lane);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void stSetupFixedTest(uint3 groupId : SV_GroupID, uint3 groupThreadId : SV_GroupThreadID)
{
    const uint w = groupId.x;
    const uint lane = groupThreadId.x;
    if (w * 32u + lane >= gSt.nbArticulations)
        return;

    stSetupFixedTendons(gSt.artiBlocks + uint64_t(w) * PXG_AB_BLOCK_SIZE,
                        gSt.linkBlocks + uint64_t(w) * uint64_t(gSt.maxLinks) * PXG_AB_LINK_SIZE,
                        gSt.dofBlocks + uint64_t(w) * uint64_t(gSt.maxDofs) * PXG_AB_DOF_SIZE,
                        gSt.tendonBlocks + uint64_t(w) * uint64_t(gSt.maxTendons) * PXG_AB_FT_SIZE,
                        gSt.constraintBlocks + uint64_t(w) * uint64_t(gSt.maxTendons)
                            * uint64_t(gSt.maxElements) * PXG_AB_ITC_SIZE,
                        gSt.elementBlocks + uint64_t(w) * uint64_t(gSt.maxTendons)
                            * uint64_t(gSt.maxElements) * PXG_AB_TJ_SIZE,
                        gSt.maxElements, gSt.stepDt, gSt.dt, gSt.invDt, gSt.isTGS != 0u, lane);
}
