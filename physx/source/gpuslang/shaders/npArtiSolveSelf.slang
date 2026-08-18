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


// npArtiSolveSelf.slang
//
// P8.99b: the self-constraint solve block of
// artiPropagateRigidImpulsesAndSolveSelfConstraints1T
// (internalConstraints2.cu:3870-4093, the TGS instantiation) plus the one
// propagation hop it needs that the tendon-solve module does not carry:
//
//   * `svPropagateAccelNoJVel` <- propagateAccelerationWNoJVelUpdate
//     (articulationImpulseResponse.cuh:59) — the DEFERRED acceleration hop in
//     the DIRECT (everythingPreLoaded) summation order: Q-stZ seeded from
//     mDeferredQstZ, per-dof contributions accumulated straight onto pDeltaV.
//     NOT svPropagateAccelDeferred: that is the child2ParentPreLoaded order
//     (separate jointSpatialDeltaV, added at the end), 1-2 ULP apart on
//     multi-dof joints — the P8.98a lesson, and here the .cuh uses the OTHER
//     order.  The body is adPropagateAccelerationWEx over block loads.
//   * `ssSolveSelfConstraints` — the per-batch loop: per active lane the two
//     link IDs from the batch's PxNodeIndex pairs, the three temp bitfield
//     buffers (A, B, common = A&B) staged from the per-link path-to-root
//     words, velocity walks root->link, the ext-body TGS solve dispatch, and
//     the impulse walks leaf->root into mDeferredQstZ with rootDeferredZ
//     carried across batches in a register.
//
// ===========================================================================
// TRAPS (each carries a mutant in the sweep unless marked equivalent)
// ===========================================================================
//
//  1. **commonDelta is computed PER CONSTRAINT, inside the active branch**,
//     from the REGISTER rootDeferredZ — which the previous batch may have
//     just accumulated into.  Hoisting it out of the loop solves batch i > 0
//     against a stale root response.
//  2. **The velocity walks are word-0-up, LOWEST-bit-first with the root
//     skipped after the offset add (`index != 0`); the impulse walks are
//     TOP-word-down, HIGHEST-bit-first** — opposite directions, and the
//     leaf->root impulse order is load-bearing because every hop writes
//     mDeferredQstZ before the next hop reads its own dof slice.
//  3. **deltaVB seeds from deltaVA AFTER the common walk** — the common-path
//     propagation is shared, then the A\common and B\common walks diverge.
//  4. **The common bits are STRIPPED from A and B in the temp buffers**
//     (read-modify-write) before the split walks; the impulse walks then
//     re-read the stripped buffers.
//  5. **The common impulse walk BREAKS at the root** (`index + wordOffset ==
//     0`) — root is bit 0 of word 0, the last position a highest-first walk
//     visits, so break-vs-skip is equivalent HERE; the mutant that matters is
//     propagating AT the root (no inbound joint, poisoned dof slice).
//  6. **`impulse0 += impulse1` happens between the A walk and the common
//     walk** — B's remainder joins A's before the shared path propagates.
//  7. **mTotalSelfConstraintCount / mSelfConstraintOffset are per-block
//     SCALARS** (no [lane] dimension) — read at lane 0, not at `lane`.
//  8. **The batch's node arrays are indexed by the lane's exclusive-popcount
//     offset within batch.mask**, not by the lane itself
//     (warpScanExclusive = popc(mask & ((1<<lane)-1)), reduction.cuh:592 —
//     a pure bit op, no cross-lane traffic).
//  9. **constraintType is the HIGH half of descStride_constraintType**
//     (PxU16 mDescStride at +0, PxU16 constraintType at +2);
//     eARTICULATION_CONTACT = 3, everything else is the 1D path.
// 10. `__syncwarp(mask)` before the solve is a lane-local no-op in this port:
//     the solve bodies exchange nothing across lanes, and the temp bitfield
//     buffers are lane-indexed ([32] per word entry).  Dropped, deliberately.
// 11. The PxgErrorAccumulator is function-local in the CUDA and NEVER
//     flushed to memory by this kernel — observable only through the 1D
//     path's per-row residual writes.  Mirrored: a local SvxError, discarded.
// 12. The final mRootDeferredZ store happens only under
//     `nbSelfConstraints != 0` — with zero batches the CUDA skips it.  A
//     port that stores unconditionally writes back the unmodified loaded
//     value, byte-identical: EQUIVALENT, not swept, noted here.
// 13. velA/velB (mMotionVelocity + deltas) are NEVER stored back — the
//     velocity walks matter only through the solve's deltaF chain.

import pxgSlangCommon;
import pxgSlangArticulationBlocks;
import pxgSlangSpatialMatrix;
import pxgSlangArticulationDynamic;
import pxgSlangSolverBlocks;
import npArtiSolveTendons;
import npArtiSolveExtTGS;

static const uint SS_ARTI_CONTACT = 3u;      // PxgSolverConstraintDesc::eARTICULATION_CONTACT
static const uint SS_BATCH_STRIDE = 1536u;   // sizeof(PxgBlockConstraintBatch), pinned by the P8.99a fixture

// propagateAccelerationWNoJVelUpdate (articulationImpulseResponse.cuh:59):
// the deferred acceleration hop in the DIRECT summation order (see header).
public AbSpatialVector svPropagateAccelNoJVel(uint64_t artiLinks, uint index,
                                              uint64_t dofBlocks, AbSpatialVector hDeltaV,
                                              uint lane)
{
    const uint64_t link = artiLinks + uint64_t(index) * PXG_AB_LINK_SIZE;
    const uint jointOffset = abU32(link, PXG_AB_LINK_mJointOffset, lane);
    const uint dofCount = abU8(link, PXG_AB_LINK_mDofs, lane);
    const float3 rw = float3(abF32(link, PXG_AB_LINK_mRw_x, lane),
                             abF32(link, PXG_AB_LINK_mRw_y, lane),
                             abF32(link, PXG_AB_LINK_mRw_z, lane));

    AbSpatialVector motion[3];
    AbSpatialVector isInvD[3];
    AbSpatialVector isW[3];
    float3 invStIsT[3];
    svLoadDofArrays(dofBlocks, jointOffset, dofCount, lane, motion, isInvD, isW, invStIsT);

    float qstz[3] = { 0.0f, 0.0f, 0.0f };
    for (uint i = 0u; i < 3u; ++i)
    {
        if (i < dofCount)
            qstz[i] = abF32(dofBlocks + uint64_t(jointOffset + i) * PXG_AB_DOF_SIZE,
                            PXG_AB_DOF_mDeferredQstZ, lane);
    }

    float jDeltaOut[3];
    return adPropagateAccelerationWEx(rw, invStIsT, motion, hDeltaV, dofCount, isW, qstz, jDeltaOut);
}

// the leaf->root deferred impulse hop this kernel uses: WriteToDeferredQstZ
// with NO joint force (the template's NULL path, `0.0f - dot`).
AbSpatialVector ssPropagateImpulseHop(uint64_t artiLinks, uint index, uint64_t dofBlocks,
                                      AbSpatialVector Z, uint lane)
{
    const uint64_t link = artiLinks + uint64_t(index) * PXG_AB_LINK_SIZE;
    const uint jointOffset = abU32(link, PXG_AB_LINK_mJointOffset, lane);
    const uint dofCount = abU8(link, PXG_AB_LINK_mDofs, lane);
    const float3 c2p = float3(abF32(link, PXG_AB_LINK_mRw_x, lane),
                              abF32(link, PXG_AB_LINK_mRw_y, lane),
                              abF32(link, PXG_AB_LINK_mRw_z, lane));
    float jfNone[3] = { 0.0f, 0.0f, 0.0f };
    return svPropagateImpulseDeferred(c2p, dofBlocks, jointOffset, dofCount, Z, lane,
                                      jfNone, 1.0f, false);
}

// internalConstraints2.cu:3861-4093 — the self-constraint block.  The state
// dirty / countActiveSlabs / averageLinkImpulsesAndPropagate prologue is the
// WRAPPER's (P8.99c); this function owns everything from the
// mTotalSelfConstraintCount read to the mRootDeferredZ store.
public void ssSolveSelfConstraints(
    uint64_t artiBlock, uint64_t artiLinks, uint64_t artiDofs,
    uint64_t linkBitFields, uint64_t tmpFieldA, uint64_t tmpFieldB, uint64_t tmpFieldCommon,
    uint wordSize,
    uint64_t batches,
    uint64_t contactHeaders, uint64_t frictionHeaders,
    uint64_t contactPoints, uint64_t frictionPoints,
    uint64_t jointHeaders, uint64_t jointRowsCon, uint64_t responses,
    float elapsedTime, float minPen, bool accumulateError,
    uint lane)
{
    // trap 7: per-block scalars, lane 0
    const uint nbSelfConstraints = abU32(artiBlock, PXG_AB_mTotalSelfConstraintCount, 0u);
    const uint startIndex = abU32(artiBlock, PXG_AB_mSelfConstraintOffset, 0u);

    AbSpatialVector rootDeferredZ = svLoadSV(artiBlock, PXG_AB_mRootDeferredZ, lane);
    const AbSpatialMatrix rootInvInertia =
        asmLoadSpatialMatrix(artiBlock, PXG_AB_mInvSpatialArticulatedInertia, lane);

    if (nbSelfConstraints != 0u)
    {
        SvxError error = { 0.0f, 0, 0.0f };

        for (uint b = 0u; b < nbSelfConstraints; ++b)
        {
            Ptr<PxgBlockConstraintBatch> batch =
                Ptr<PxgBlockConstraintBatch>(batches + uint64_t(startIndex + b) * SS_BATCH_STRIDE);

            const uint mask = batch[0].mask;

            if ((mask & (1u << lane)) != 0u)
            {
                // trap 1: per constraint, from the REGISTER rootDeferredZ
                const AbSpatialVector commonDelta = asmMulVec(rootInvInertia, asmSVec(-rootDeferredZ.top, -rootDeferredZ.bottom));

                // trap 8: the exclusive-popcount offset, not the lane
                const uint offset = countbits(mask & ((1u << lane) - 1u));

                const uint linkIDA = nodeIndexArticulationLinkId(batch[0].bodyANodeIndex[offset]);
                const uint linkIDB = nodeIndexArticulationLinkId(batch[0].bodyBNodeIndex[offset]);

                const uint64_t linkA = artiLinks + uint64_t(linkIDA) * PXG_AB_LINK_SIZE;
                const uint64_t linkB = artiLinks + uint64_t(linkIDB) * PXG_AB_LINK_SIZE;

                // stage the three temp bitfield buffers (lane-indexed words)
                for (uint j = 0u; j < wordSize; ++j)
                {
                    const uint64_t wordA = abU64(linkBitFields + uint64_t(linkIDA * wordSize + j) * PXG_AB_BITFIELD_STACK_SIZE, 0u, lane);
                    const uint64_t wordB = abU64(linkBitFields + uint64_t(linkIDB * wordSize + j) * PXG_AB_BITFIELD_STACK_SIZE, 0u, lane);
                    abStoreU64(tmpFieldA + uint64_t(j) * PXG_AB_BITFIELD_STACK_SIZE, 0u, lane, wordA);
                    abStoreU64(tmpFieldB + uint64_t(j) * PXG_AB_BITFIELD_STACK_SIZE, 0u, lane, wordB);
                    abStoreU64(tmpFieldCommon + uint64_t(j) * PXG_AB_BITFIELD_STACK_SIZE, 0u, lane, wordA & wordB);
                }

                AbSpatialVector velA = svLoadSV(linkA, PXG_AB_LINK_mMotionVelocity, lane);
                AbSpatialVector velB = svLoadSV(linkB, PXG_AB_LINK_mMotionVelocity, lane);

                AbSpatialVector deltaVA = commonDelta;

                // velocity walk 1: the common path, root->down (trap 2)
                for (uint j = 0u, wordOffset = 0u; j < wordSize; ++j, wordOffset += 64u)
                {
                    uint64_t word = abU64(tmpFieldCommon + uint64_t(j) * PXG_AB_BITFIELD_STACK_SIZE, 0u, lane);
                    while (word != 0uL)
                    {
                        const uint index = svLowestBit64(word) + wordOffset;
                        if (index != 0u)   // root skip, AFTER the offset add
                        {
                            deltaVA = svPropagateAccelNoJVel(artiLinks, index, artiDofs, deltaVA, lane);
                        }
                        word &= (word - 1uL);
                    }
                }

                // trap 3: B branches from the post-common-walk deltaVA
                AbSpatialVector deltaVB = deltaVA;

                // trap 4: strip the common bits out of A and B
                for (uint j = 0u; j < wordSize; ++j)
                {
                    const uint64_t common = abU64(tmpFieldCommon + uint64_t(j) * PXG_AB_BITFIELD_STACK_SIZE, 0u, lane);
                    abStoreU64(tmpFieldA + uint64_t(j) * PXG_AB_BITFIELD_STACK_SIZE, 0u, lane, abU64(tmpFieldA + uint64_t(j) * PXG_AB_BITFIELD_STACK_SIZE, 0u, lane) & (~common));
                    abStoreU64(tmpFieldB + uint64_t(j) * PXG_AB_BITFIELD_STACK_SIZE, 0u, lane, abU64(tmpFieldB + uint64_t(j) * PXG_AB_BITFIELD_STACK_SIZE, 0u, lane) & (~common));
                }

                // velocity walk 2: A \ common (never contains the root)
                for (uint j = 0u, wordOffset = 0u; j < wordSize; ++j, wordOffset += 64u)
                {
                    uint64_t word = abU64(tmpFieldA + uint64_t(j) * PXG_AB_BITFIELD_STACK_SIZE, 0u, lane);
                    while (word != 0uL)
                    {
                        const uint index = svLowestBit64(word) + wordOffset;
                        deltaVA = svPropagateAccelNoJVel(artiLinks, index, artiDofs, deltaVA, lane);
                        word &= (word - 1uL);
                    }
                }

                // velocity walk 3: B \ common
                for (uint j = 0u, wordOffset = 0u; j < wordSize; ++j, wordOffset += 64u)
                {
                    uint64_t word = abU64(tmpFieldB + uint64_t(j) * PXG_AB_BITFIELD_STACK_SIZE, 0u, lane);
                    while (word != 0uL)
                    {
                        const uint index = svLowestBit64(word) + wordOffset;
                        deltaVB = svPropagateAccelNoJVel(artiLinks, index, artiDofs, deltaVB, lane);
                        word &= (word - 1uL);
                    }
                }

                velA = asmSVecAdd(velA, deltaVA);
                velB = asmSVecAdd(velB, deltaVB);


                AbSpatialVector impulse0 = asmSVec(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f));
                AbSpatialVector impulse1 = asmSVec(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f));

                // __syncwarp(mask) here in the CUDA — lane-local no-op (trap 10)

                // solveConstraints (:3774, the TGS overload)
                const AbSpatialVector delta0 = svLoadSV(linkA, PXG_AB_LINK_mDeltaMotion, lane);
                const AbSpatialVector delta1 = svLoadSV(linkB, PXG_AB_LINK_mDeltaMotion, lane);

                const uint constraintType = batch[0].descStride_constraintType >> 16u;
                if (constraintType == SS_ARTI_CONTACT)
                {
                    svxSolveExtContactBlockTGS(batch, velA, velB, delta0, delta1, offset,
                                               Ptr<PxgTGSBlockSolverContactHeader>(contactHeaders),
                                               Ptr<PxgTGSBlockSolverFrictionHeader>(frictionHeaders),
                                               Ptr<PxgTGSBlockSolverContactPoint>(contactPoints),
                                               Ptr<PxgTGSBlockSolverContactFriction>(frictionPoints),
                                               Ptr<PxgArticulationBlockResponse>(responses),
                                               elapsedTime, minPen, impulse0, impulse1,
                                               error, accumulateError, 1.0f, 1.0f);
                }
                else
                {
                    const float4 deltaQ0 = abF4(linkA, PXG_AB_LINK_mDeltaQ, lane);
                    const float4 deltaQ1 = abF4(linkB, PXG_AB_LINK_mDeltaQ, lane);
                    svxSolveExt1DBlockTGS(batch, velA, velB, delta0, delta1, offset,
                                          Ptr<PxgTGSBlockSolverConstraint1DHeader>(jointHeaders),
                                          Ptr<PxgTGSBlockSolverConstraint1DCon>(jointRowsCon),
                                          Ptr<PxgArticulationBlockResponse>(responses),
                                          deltaQ0, deltaQ1, elapsedTime, impulse0, impulse1,
                                          accumulateError, 1.0f, 1.0f);
                }

                // impulse walk 1: B \ common, leaf->root (trap 2)
                for (int j = int(wordSize) - 1; j >= 0; --j)
                {
                    const uint wordOffset = uint(j) * 64u;
                    uint64_t word = abU64(tmpFieldB + uint64_t(j) * PXG_AB_BITFIELD_STACK_SIZE, 0u, lane);
                    while (word != 0uL)
                    {
                        const uint index = svHighestBit64(word);
                        word &= ~(uint64_t(1) << index);
                        impulse1 = ssPropagateImpulseHop(artiLinks, index + wordOffset, artiDofs, impulse1, lane);
                    }
                }

                // impulse walk 2: A \ common
                for (int j = int(wordSize) - 1; j >= 0; --j)
                {
                    const uint wordOffset = uint(j) * 64u;
                    uint64_t word = abU64(tmpFieldA + uint64_t(j) * PXG_AB_BITFIELD_STACK_SIZE, 0u, lane);
                    while (word != 0uL)
                    {
                        const uint index = svHighestBit64(word);
                        word &= ~(uint64_t(1) << index);
                        impulse0 = ssPropagateImpulseHop(artiLinks, index + wordOffset, artiDofs, impulse0, lane);
                    }
                }

                // trap 6: join before the shared path
                impulse0 = asmSVecAdd(impulse0, impulse1);

                // impulse walk 3: the common path, with the root BREAK (trap 5)
                for (int j = int(wordSize) - 1; j >= 0; --j)
                {
                    const uint wordOffset = uint(j) * 64u;
                    uint64_t word = abU64(tmpFieldCommon + uint64_t(j) * PXG_AB_BITFIELD_STACK_SIZE, 0u, lane);
                    while (word != 0uL)
                    {
                        const uint index = svHighestBit64(word);
                        if ((index + wordOffset) == 0u)
                            break;
                        word &= ~(uint64_t(1) << index);
                        impulse0 = ssPropagateImpulseHop(artiLinks, index + wordOffset, artiDofs, impulse0, lane);
                    }
                }

                rootDeferredZ = asmSVecAdd(rootDeferredZ, impulse0);
            }
        }

        abStoreSpatialVector(artiBlock, PXG_AB_mRootDeferredZ, lane, rootDeferredZ.top, rootDeferredZ.bottom);
    }
}

// ===========================================================================
// test entry (the P8.98b arrangement: one warp, params blob, byte-compare on
// the host side)
// ===========================================================================

struct SsParams
{
    uint64_t artiBlock;        // @0
    uint64_t artiLinks;        // @8
    uint64_t artiDofs;         // @16
    uint64_t linkBitFields;    // @24
    uint64_t tmpFieldA;        // @32  mTempSharedBitFieldBlocks slice
    uint64_t tmpFieldB;        // @40  mTempRootBitFieldBlocks slice
    uint64_t tmpFieldCommon;   // @48  mTempPathToRootBitFieldBlocks slice
    uint64_t batches;          // @56
    uint64_t contactHeaders;   // @64
    uint64_t frictionHeaders;  // @72
    uint64_t contactPoints;    // @80
    uint64_t frictionPoints;   // @88
    uint64_t jointHeaders;     // @96
    uint64_t jointRowsCon;     // @104
    uint64_t responses;        // @112
    float elapsedTime;         // @120
    float minPen;              // @124
    uint wordSize;             // @128
    uint accumulateError;      // @132
    uint nbLanes;              // @136
    uint pad0;                 // @140
}

[[vk::push_constant]]
ConstantBuffer<SsParams> gSs;

[shader("compute")]
[numthreads(32, 1, 1)]
void ssSolveSelfTest(uint3 groupThreadId : SV_GroupThreadID)
{
    const uint lane = groupThreadId.x;
    if (lane >= gSs.nbLanes)
        return;

    ssSolveSelfConstraints(gSs.artiBlock, gSs.artiLinks, gSs.artiDofs,
                           gSs.linkBitFields, gSs.tmpFieldA, gSs.tmpFieldB, gSs.tmpFieldCommon,
                           gSs.wordSize,
                           gSs.batches,
                           gSs.contactHeaders, gSs.frictionHeaders,
                           gSs.contactPoints, gSs.frictionPoints,
                           gSs.jointHeaders, gSs.jointRowsCon, gSs.responses,
                           gSs.elapsedTime, gSs.minPen, gSs.accumulateError != 0u,
                           lane);
}
