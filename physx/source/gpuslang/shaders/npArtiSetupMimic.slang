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


// npArtiSetupMimic.slang
//
// P8.97c: `setupInternalMimicJointConstraints` (internalConstraints2.cu:617)
// with `computeMimicJointSelfResponse` (:411), `computeMimicJointCrossResponse`
// (:480) and the three load helpers (:354-399) -- the last setup body of
// `setupInternalConstraintLaunch1T`.
//
// ZERO new ABI: the mimic blocks are P8.77/78's (PXG_AB_MJ_* / PXG_AB_IMJ_*),
// the path-to-root bitfields P8.73's, mDeferredQstZ P8.80's, and mInvDofIds
// P8.68's.
//
// THE ORACLE IS FULLY HOST-CALLABLE: the propagate pair the mimic path uses
// is the SHARED DyCpuGpuArticulation.h pair (propagateImpulseW :494,
// propagateAccelerationW :549, both PX_CUDA_CALLABLE), and
// computeRecipMimicJointEffectiveInertia (:603) too -- the fixture computes
// every response with the engine's own arithmetic.
//
// TRAPS, from the reading:
//  1. The SHARED propagateImpulseW SUMS IN A DIFFERENT ORDER than the block
//     variant P8.90 ported: YParentW starts at ZERO, accumulates the per-dof
//     terms, and adds YChildW at the END -- (t1+t2)+Z, where the ad* module
//     computes Z+t1+t2.  Different float rounding: the shared spelling is
//     its OWN function here (mjPropagateImpulseShared), not a reuse.  The
//     shared propagateAccelerationW IS bit-equal to adPropagateAccelerationWEx
//     for zero-seeded accumulators (the extra full-dot terms multiply zeros,
//     and +-0 addition is exact), so the acceleration direction reuses it.
//  2. The cross-response uses mDeferredQstZ as its persistent scratch --
//     zeroed over mTotalDofs BEFORE the walk and AGAIN before returning
//     (the solver accumulates there later; the second zeroing is part of
//     the contract).  The self-response touches NO block state at all.
//  3. The A->root walk iterates bitfield words TOP-DOWN taking the HIGHEST
//     set bit first (root-ward order); the root->B walk iterates BOTTOM-UP
//     taking the LOWEST set bit first.  Both skip linkIndex 0 (the root is
//     on every path but cannot propagate upward).
//  4. The first A-walk iteration is always linkA itself (the highest bit of
//     its own path), so Zp is never read before its first write; the
//     jointImpulse is the unit test impulse THERE and NULL everywhere else,
//     while the link impulse is zero THERE and the running Zp elsewhere.
//  5. jointVelocity is OVERWRITTEN per visited link on the way down --
//     after the loop it holds the LAST link's (linkB's) childJointSpeed.
//  6. Cross-response stores the local QMinusSTZ into mDeferredQstZ by
//     ASSIGNMENT after each impulse hop (the propagate itself accumulates
//     into the zero-seeded local).
//  7. dofA/dofB resolve through mInvDofIds (PxU8, row stride 32) and are
//     stored into mInternalData BEFORE the responses are computed.
//  8. recipEffectiveInertia = rAA + g*(rAB + rBA) + g*g*rBB, spelled exactly
//     (DyCpuGpuArticulation.h:603).

import pxgSlangCommon;
import pxgSlangArticulationBlocks;
import pxgSlangSpatialMatrix;
import pxgSlangArticulationDynamic;

uint mjHighestBit64(uint64_t v)
{
    const uint hi = uint(v >> 32);
    return hi != 0u ? 32u + firstbithigh(hi) : firstbithigh(uint(v));
}

uint mjLowestBit64(uint64_t v)
{
    const uint lo = uint(v);
    return lo != 0u ? firstbitlow(lo) : 32u + firstbitlow(uint(v >> 32));
}

AbSpatialVector mjLoadSV(uint64_t base, uint fieldOffset, uint lane)
{
    float3 top, bottom;
    abLoadSpatialVector(base, fieldOffset, lane, top, bottom);
    return asmSVec(top, bottom);
}

// the per-link dof arrays every propagation hop needs
void mjLoadDofArrays(uint64_t dofBlocks, uint jointOffset, uint dofCount, uint lane,
                     out AbSpatialVector motion[3], out AbSpatialVector isInvD[3],
                     out AbSpatialVector isW[3], out float3 invStIsT[3])
{
    for (uint i = 0u; i < 3u; ++i)
    {
        motion[i] = asmSVec(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f));
        isInvD[i] = motion[i];
        isW[i] = motion[i];
        invStIsT[i] = float3(0.0f, 0.0f, 0.0f);
        if (i < dofCount)
        {
            const uint64_t d = dofBlocks + uint64_t(jointOffset + i) * PXG_AB_DOF_SIZE;
            motion[i] = mjLoadSV(d, PXG_AB_DOF_mWorldMotionMatrix, lane);
            isInvD[i] = mjLoadSV(d, PXG_AB_DOF_mIsInvDW, lane);
            isW[i] = mjLoadSV(d, PXG_AB_DOF_mIsW, lane);
            invStIsT[i] = float3(abF32(d, PXG_AB_DOF_mInvStIsT_x, lane),
                                 abF32(d, PXG_AB_DOF_mInvStIsT_y, lane),
                                 abF32(d, PXG_AB_DOF_mInvStIsT_z, lane));
        }
    }
}

// the SHARED propagateImpulseW (DyCpuGpuArticulation.h:494) -- trap 1: the
// child impulse joins the sum at the END, unlike the P8.90 block variant
AbSpatialVector mjPropagateImpulseShared(float3 parentToChild, AbSpatialVector YChildW,
                                         bool hasJointImpulse, float jointImpulse[3],
                                         AbSpatialVector isInvD[3], AbSpatialVector motion[3],
                                         uint dofCount, inout float qMinusStY[3])
{
    AbSpatialVector YParentW = asmSVec(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f));
    for (uint ind = 0u; ind < 3u; ++ind)
    {
        if (ind < dofCount)
        {
            const float Q = hasJointImpulse ? jointImpulse[ind] : 0.0f;
            const float QMinusStY = Q - asmInnerProduct(motion[ind], YChildW);
            YParentW = asmSVecAdd(YParentW, asmSVecScale(isInvD[ind], QMinusStY));
            qMinusStY[ind] = qMinusStY[ind] + QMinusStY;
        }
    }
    YParentW = asmSVecAdd(YParentW, YChildW);
    return adTranslateSpatialVector(parentToChild, YParentW);
}

// computeMimicJointSelfResponse (internalConstraints2.cu:411): the deltaQDot
// of a joint dof to a unit impulse on that same dof.  Touches NO block state.
float mjSelfResponse(uint linkIndex, uint dof, uint64_t linkBase, uint64_t dofBlocks, uint lane)
{
    const uint64_t link = linkBase + uint64_t(linkIndex) * PXG_AB_LINK_SIZE;
    const uint parentLinkIndex = abU32(link, PXG_AB_LINK_mParents, lane);
    const float3 rw = float3(abF32(link, PXG_AB_LINK_mRw_x, lane),
                             abF32(link, PXG_AB_LINK_mRw_y, lane),
                             abF32(link, PXG_AB_LINK_mRw_z, lane));
    const uint jointOffset = abU32(link, PXG_AB_LINK_mJointOffset, lane);
    const uint dofCount = abU8(link, PXG_AB_LINK_mDofs, lane);

    float testJointImpulse[3] = { 0.0f, 0.0f, 0.0f };
    testJointImpulse[dof] = 1.0f;

    AbSpatialVector motion[3];
    AbSpatialVector isInvD[3];
    AbSpatialVector isW[3];
    float3 invStIsT[3];
    mjLoadDofArrays(dofBlocks, jointOffset, dofCount, lane, motion, isInvD, isW, invStIsT);

    float qMinusStZ[3] = { 0.0f, 0.0f, 0.0f };
    const AbSpatialVector zero = asmSVec(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f));
    const AbSpatialVector Zp = mjPropagateImpulseShared(rw, zero, true, testJointImpulse,
                                                        isInvD, motion, dofCount, qMinusStZ);

    const AbMatrix6x6 mat =
        abLoadSpatialMatrix6x6(linkBase + uint64_t(parentLinkIndex) * PXG_AB_LINK_SIZE,
                               PXG_AB_LINK_mSpatialResponseMatrix, lane);
    const AbSpatialVector deltaVParent = asmM6MulVec(mat, asmSVec(-Zp.top, -Zp.bottom));

    float jointDeltaQDot[3];
    adPropagateAccelerationWEx(rw, invStIsT, motion, deltaVParent, dofCount, isW, qMinusStZ,
                               jointDeltaQDot);
    return jointDeltaQDot[dof];
}

// computeMimicJointCrossResponse (internalConstraints2.cu:480): the deltaQDot
// at linkB/dofB from a unit impulse on linkA/dofA, via the path-to-root
// bitfields and the mDeferredQstZ scratch (zeroed on entry AND exit).
float mjCrossResponse(uint linkA, uint dofA, uint linkB, uint dofB,
                      uint64_t artiBlock, uint64_t linkBase, uint64_t dofBlocks,
                      uint64_t bitFields, uint wordSize, uint lane)
{
    float testJointImpulse[3] = { 0.0f, 0.0f, 0.0f };
    testJointImpulse[dofA] = 1.0f;

    const uint totalDofs = abU32(artiBlock, PXG_AB_mTotalDofs, lane);
    for (uint i = 0u; i < totalDofs; ++i)
        abStoreF32(dofBlocks + uint64_t(i) * PXG_AB_DOF_SIZE, PXG_AB_DOF_mDeferredQstZ,
                   lane, 0.0f);

    // trap 3/4: linkA -> root, top word down, highest bit first
    AbSpatialVector Zp = asmSVec(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f));
    for (uint jj = 0u; jj < wordSize; ++jj)
    {
        const uint j = wordSize - 1u - jj;
        const uint bitOffset = j * 64u;
        uint64_t word = abU64(bitFields
                                  + uint64_t(linkA * wordSize + j) * PXG_AB_BITFIELD_STACK_SIZE,
                              0u, lane);
        while (word != 0uL)
        {
            const uint bitIndex = mjHighestBit64(word);
            const uint linkIndex = bitIndex + bitOffset;
            word &= ~(uint64_t(1) << bitIndex);

            if (linkIndex != 0u)
            {
                const bool isA = linkA == linkIndex;
                const AbSpatialVector linkImpulse =
                    isA ? asmSVec(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f)) : Zp;

                const uint64_t link = linkBase + uint64_t(linkIndex) * PXG_AB_LINK_SIZE;
                const float3 rw = float3(abF32(link, PXG_AB_LINK_mRw_x, lane),
                                         abF32(link, PXG_AB_LINK_mRw_y, lane),
                                         abF32(link, PXG_AB_LINK_mRw_z, lane));
                const uint jointOffset = abU32(link, PXG_AB_LINK_mJointOffset, lane);
                const uint dofCount = abU8(link, PXG_AB_LINK_mDofs, lane);

                AbSpatialVector motion[3];
                AbSpatialVector isInvD[3];
                AbSpatialVector isW[3];
                float3 invStIsT[3];
                mjLoadDofArrays(dofBlocks, jointOffset, dofCount, lane,
                                motion, isInvD, isW, invStIsT);

                float qMinusStZ[3] = { 0.0f, 0.0f, 0.0f };
                Zp = mjPropagateImpulseShared(rw, linkImpulse, isA, testJointImpulse,
                                              isInvD, motion, dofCount, qMinusStZ);

                // trap 6: assignment into the persistent scratch
                for (uint i = 0u; i < dofCount; ++i)
                    abStoreF32(dofBlocks + uint64_t(jointOffset + i) * PXG_AB_DOF_SIZE,
                               PXG_AB_DOF_mDeferredQstZ, lane, qMinusStZ[i]);
            }
        }
    }

    const AbMatrix6x6 rootMat =
        abLoadSpatialMatrix6x6(linkBase, PXG_AB_LINK_mSpatialResponseMatrix, lane);
    const AbSpatialVector deltaVRoot = asmM6MulVec(rootMat, asmSVec(-Zp.top, -Zp.bottom));

    // trap 3/5: root -> linkB, word 0 up, lowest bit first; jointVelocity is
    // overwritten per link and ends as linkB's
    float jointVelocity[3] = { 0.0f, 0.0f, 0.0f };
    AbSpatialVector deltaVParent = deltaVRoot;
    for (uint j = 0u; j < wordSize; ++j)
    {
        const uint bitOffset = j * 64u;
        uint64_t word = abU64(bitFields
                                  + uint64_t(linkB * wordSize + j) * PXG_AB_BITFIELD_STACK_SIZE,
                              0u, lane);
        while (word != 0uL)
        {
            const uint bitIndex = mjLowestBit64(word);
            const uint linkIndex = bitIndex + bitOffset;
            word &= ~(uint64_t(1) << bitIndex);

            if (linkIndex != 0u)
            {
                const uint64_t link = linkBase + uint64_t(linkIndex) * PXG_AB_LINK_SIZE;
                const float3 rw = float3(abF32(link, PXG_AB_LINK_mRw_x, lane),
                                         abF32(link, PXG_AB_LINK_mRw_y, lane),
                                         abF32(link, PXG_AB_LINK_mRw_z, lane));
                const uint jointOffset = abU32(link, PXG_AB_LINK_mJointOffset, lane);
                const uint dofCount = abU8(link, PXG_AB_LINK_mDofs, lane);

                AbSpatialVector motion[3];
                AbSpatialVector isInvD[3];
                AbSpatialVector isW[3];
                float3 invStIsT[3];
                mjLoadDofArrays(dofBlocks, jointOffset, dofCount, lane,
                                motion, isInvD, isW, invStIsT);

                float qMinusStZ[3] = { 0.0f, 0.0f, 0.0f };
                for (uint i = 0u; i < dofCount; ++i)
                    qMinusStZ[i] = abF32(dofBlocks + uint64_t(jointOffset + i) * PXG_AB_DOF_SIZE,
                                         PXG_AB_DOF_mDeferredQstZ, lane);

                float childJointSpeed[3];
                deltaVParent = adPropagateAccelerationWEx(rw, invStIsT, motion, deltaVParent,
                                                          dofCount, isW, qMinusStZ,
                                                          childJointSpeed);
                jointVelocity[0] = childJointSpeed[0];
                jointVelocity[1] = childJointSpeed[1];
                jointVelocity[2] = childJointSpeed[2];
            }
        }
    }

    // trap 2: the exit zeroing is part of the contract
    for (uint i = 0u; i < totalDofs; ++i)
        abStoreF32(dofBlocks + uint64_t(i) * PXG_AB_DOF_SIZE, PXG_AB_DOF_mDeferredQstZ,
                   lane, 0.0f);

    return jointVelocity[dofB];
}

// setupInternalMimicJointConstraints (internalConstraints2.cu:617)
public void mjSetupMimicJoints(uint64_t artiBlock, uint64_t linkBase, uint64_t dofBlocks,
                               uint64_t bitFields, uint wordSize, uint64_t mimicBase,
                               uint lane)
{
    const uint nbMimicJoints = abU32(artiBlock, PXG_AB_mNumMimicJoints, lane);
    for (uint i = 0u; i < nbMimicJoints; ++i)
    {
        const uint64_t mj = mimicBase + uint64_t(i) * PXG_AB_MJ_SIZE;
        const uint linkA = abU32(mj, PXG_AB_MJ_mLinkA, lane);
        const uint linkB = abU32(mj, PXG_AB_MJ_mLinkB, lane);
        const uint axisA = abU32(mj, PXG_AB_MJ_mAxisA, lane);
        const uint axisB = abU32(mj, PXG_AB_MJ_mAxisB, lane);
        // trap 7: PxU8 rows, stride 32
        const uint dofA = abU8(linkBase + uint64_t(linkA) * PXG_AB_LINK_SIZE,
                               PXG_AB_LINK_mInvDofIds + axisA * 32u, lane);
        const uint dofB = abU8(linkBase + uint64_t(linkB) * PXG_AB_LINK_SIZE,
                               PXG_AB_LINK_mInvDofIds + axisB * 32u, lane);
        abStoreU32(mj + PXG_AB_MJ_mInternalData, PXG_AB_IMJ_mDofA, lane, dofA);
        abStoreU32(mj + PXG_AB_MJ_mInternalData, PXG_AB_IMJ_mDofB, lane, dofB);

        const float rAA = mjSelfResponse(linkA, dofA, linkBase, dofBlocks, lane);
        const float rBB = mjSelfResponse(linkB, dofB, linkBase, dofBlocks, lane);
        const float rBA = mjCrossResponse(linkA, dofA, linkB, dofB,
                                          artiBlock, linkBase, dofBlocks,
                                          bitFields, wordSize, lane);
        const float rAB = mjCrossResponse(linkB, dofB, linkA, dofA,
                                          artiBlock, linkBase, dofBlocks,
                                          bitFields, wordSize, lane);

        const float gearRatio = abF32(mj, PXG_AB_MJ_mGearRatio, lane);
        // trap 8: DyCpuGpuArticulation.h:603, spelled exactly
        abStoreF32(mj + PXG_AB_MJ_mInternalData, PXG_AB_IMJ_recipEffectiveInertia, lane,
                   rAA + gearRatio * (rAB + rBA) + gearRatio * gearRatio * rBB);
    }
}

// ---------------------------------------------------------------------------
// fixture entry point: one articulation per lane, one warp per group
// ---------------------------------------------------------------------------

struct MjParams
{
    uint64_t artiBlocks;   // @0
    uint64_t linkBlocks;   // @8
    uint64_t dofBlocks;    // @16
    uint64_t bitFields;    // @24
    uint64_t mimicBlocks;  // @32
    uint nbArticulations;  // @40
    uint maxLinks;         // @44
    uint maxDofs;          // @48
    uint maxMimicJoints;   // @52
    uint pad0;             // @56 -> 64
    uint pad1;             // @60
};

[[vk::push_constant]]
ConstantBuffer<MjParams> gMj;

[shader("compute")]
[numthreads(32, 1, 1)]
void mjSetupMimicTest(uint3 groupId : SV_GroupID, uint3 groupThreadId : SV_GroupThreadID)
{
    const uint w = groupId.x;
    const uint lane = groupThreadId.x;
    if (w * 32u + lane >= gMj.nbArticulations)
        return;

    const uint wordSize = (gMj.maxLinks + 63u) / 64u;
    mjSetupMimicJoints(gMj.artiBlocks + uint64_t(w) * PXG_AB_BLOCK_SIZE,
                       gMj.linkBlocks + uint64_t(w) * uint64_t(gMj.maxLinks) * PXG_AB_LINK_SIZE,
                       gMj.dofBlocks + uint64_t(w) * uint64_t(gMj.maxDofs) * PXG_AB_DOF_SIZE,
                       gMj.bitFields + uint64_t(w) * uint64_t(gMj.maxLinks)
                           * uint64_t(wordSize) * PXG_AB_BITFIELD_STACK_SIZE,
                       wordSize,
                       gMj.mimicBlocks + uint64_t(w) * uint64_t(gMj.maxMimicJoints)
                           * PXG_AB_MJ_SIZE,
                       lane);
}
