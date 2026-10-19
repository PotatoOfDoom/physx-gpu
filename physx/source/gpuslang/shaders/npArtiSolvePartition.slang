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

// npArtiSolvePartition.slang
//
// P8.104: artiSolveBlockPartitionTGSInternal (solverMultiBlockTGS.cu:756-996)
// -- the articulation branch of solveBlockUnified (blockIdx.y != 0), which
// per solver iteration solves one partition's ARTICULATION contact/joint
// batches: mixed arti/rigid body pairs, the ext-solve pair from P8.99a, the
// deferredZ impulse scatter, and the slab/state-dirty bookkeeping the
// propagate kernels (P8.99c/P8.101) consume.  Kept as its own module so
// solverTGS.slang (whose file-local block-struct mirrors would collide with
// pxgSlangSolverBlocks' public ones) imports ONE public function and nothing
// else becomes visible.
//
// ===========================================================================
// TRAPS (each carries a sweep mutant unless marked)
// ===========================================================================
//  1. **startIndex is partition-0-special**: island.mArtiBatchStartIndex for
//     partition 0, else artiConstraintsPerPartition[partitionIndex +
//     startPartitionIndex - 1]; endIndex adds the batch offset, startIndex
//     does NOT (it is added into k separately).
//  2. **articulationBatchOffset and nbArticulations read islandContextPool[0]**
//     (the CUDA derefs the POINTER: `islandContextPool->mBatchCount` /
//     `->mArticulationCount`), while island itself is pool[islandIndex].
//  3. **The lane guard must not skip the error reduction**: the CUDA's
//     `return` for lanes >= mDescStride never reaches the accumulate, which
//     contributes exactly zero -- the port guards the body instead so the
//     caller's groupshared butterfly stays in uniform control flow.
//  4. **The 3-slot velocity staging** at readIndex = k*192 + lane, slots
//     0/32/64 (body A) and 96/128/160 (body B), TGS packing
//     (vb.xyz|vt.x, vt.yz|db.xy, db.z|dt.xyz).
//  5. **curRef counts active slabs** for articulations at remappedBodyIndex +
//     numDynamicBodies + bodyOffset, for rigids only when solverBodyId >=
//     bodyOffset; both default to 1.
//  6. **deltaQ loads only happen for eARTICULATION_CONSTRAINT_1D batches**;
//     arti links read the BLOCK's mDeltaQ at column solverBodyId & 31, rigids
//     read txIData[solverBodyId].deltaBody2World.q (offset 0 of the 64-byte
//     struct).
//  7. **The rigid writeback goes to remappedBodyAIndex/remappedBodyBIndex**
//     (NOT the readIndex) in the same 3-slot packing.
//  8. **The arti writeback scatters the IMPULSE** (not the velocity) to
//     deferredZ[computeDeltaVIndex(nbArticulations, maxLinks, artiId, linkID,
//     slabId)] (artiId + linkID*nbArti + slabId*nbArti*maxLinks, 24-byte
//     UnAlignedSpatialVector, top first), writes isSlabDirty[artiId +
//     slabId*nbArticulations].x = linkIndexA (.y = linkIndexB for body B),
//     and sets mStateDirty = eHAS_IMPULSES | eVEL_DIRTY (= 3) through the
//     sub-word atomic pair.
//  9. constraintType discriminates eARTICULATION_CONTACT (3) vs
//     eARTICULATION_CONSTRAINT_1D (2) from the packed high 16 bits.
// ===========================================================================

import pxgSlangCommon;
import pxgSlangSpatialMatrix;
import pxgSlangArticulationBlocks;
import pxgSlangArticulationDesc;
import pxgSlangSolverBlocks;
import pxgSlangSolverDesc;
import npArtiSolveExtTGS;

static const uint ASP_ARTI_CONSTRAINT_1D = 2u;   // PxgSolverConstraintDesc::eARTICULATION_CONSTRAINT_1D
static const uint ASP_ARTI_CONTACT = 3u;         // PxgSolverConstraintDesc::eARTICULATION_CONTACT
static const uint ASP_DIRTY = 3u;                // eHAS_IMPULSES | eVEL_DIRTY

// computeDeltaVIndex (PxgArticulation.h:967)
public uint aspDeltaVIndex(uint maxArticulations, uint maxLinks, uint articulationID,
                           uint linkID, uint slabID)
{
    return articulationID + linkID * maxArticulations + slabID * maxArticulations * maxLinks;
}

public void aspSolveBlockPartitionTGS(
    uint64_t solverDescA, uint64_t sharedDescA,
    uint islandIndex, uint partitionIndex,
    float elapsedTime, float minPen, uint64_t artiDescA,
    uint globalWarpIndex, uint lane,
    out float errSumSq, out float errMax, out uint errCount, out bool accumulateError)
{
    errSumSq = 0.0f;
    errMax = 0.0f;
    errCount = 0u;

    Ptr<PxgSolverCoreDescGpu> solverDesc = Ptr<PxgSolverCoreDescGpu>(solverDescA);
    Ptr<PxgSolverSharedDescTGSGpu> sharedDesc = Ptr<PxgSolverSharedDescTGSGpu>(sharedDescA);
    Ptr<PxgIslandContextGpu> islandPool = Ptr<PxgIslandContextGpu>(solverDesc[0].islandContextPool);

    // 5.12 removed the solver residual reporting and PxgSolverCoreDesc's
    // contactErrorAccumulator: nothing accumulates any more (the error outputs
    // stay until the articulation area strips them; solveBlockUnified ignores them).
    accumulateError = false;

    const uint64_t d = artiDescA;
    const uint64_t gArticulations = artiPtrField(d, PXG_ARTI_CORE_ARTICULATION_BLOCKS);
    const uint64_t gArticulationLinks = artiPtrField(d, PXG_ARTI_CORE_LINK_BLOCKS);
    const uint maxLinks = artiU32Field(d, PXG_ARTI_CORE_MAX_LINKS);
    const uint64_t deferredZ = sharedDesc[0].articulationDeferredZ;
    const uint64_t isSlabDirty = artiPtrField(d, PXG_ARTI_CORE_SLAB_HAS_CHANGES);

    const uint startPartitionIndex = islandPool[islandIndex].mStartPartitionIndex;
    Ptr<uint> artiConstraintsPerPartition = Ptr<uint>(solverDesc[0].artiConstraintsPerPartition);

    // trap 1
    const uint startIndex = partitionIndex == 0u
        ? islandPool[islandIndex].mArtiBatchStartIndex
        : artiConstraintsPerPartition[partitionIndex + startPartitionIndex - 1u];

    // trap 2: the POINTER deref -- element 0
    const uint articulationBatchOffset = islandPool[0].mBatchCount;
    const uint endIndex =
        artiConstraintsPerPartition[partitionIndex + startPartitionIndex] + articulationBatchOffset;

    const uint k = startIndex + globalWarpIndex + articulationBatchOffset;

    Ptr<float> iData = Ptr<float>(solverDesc[0].solverBodyTxIDataPool);   // 64-byte stride

    if (k >= endIndex)
        return;

    Ptr<PxgBlockConstraintBatch> batches =
        Ptr<PxgBlockConstraintBatch>(sharedDesc[0].iter_blockConstraintBatch);
    Ptr<PxgBlockConstraintBatch> batch = batches + k;

    const uint descStride = batch[0].descStride_constraintType & 0xFFFFu;
    const uint constraintType = batch[0].descStride_constraintType >> 16u;

    // trap 3: guard, not return -- the caller's butterfly needs every lane
    if (lane >= descStride)
        return;

    const uint readIndex = k * 192u + lane;

    // trap 2 again: nbArticulations from pool[0]
    const uint nbArticulations = islandPool[0].mArticulationCount;

    const uint2 igNodeIndexA = batch[0].bodyANodeIndex[lane];
    const uint2 igNodeIndexB = batch[0].bodyBNodeIndex[lane];
    const uint slabId = batch[0].slabId[lane];
    const uint solverBodyId0 = batch[0].bodyAIndex[lane];
    const uint solverBodyId1 = batch[0].bodyBIndex[lane];
    const uint linkIndexA = nodeIndexArticulationLinkId(igNodeIndexA);
    const uint linkIndexB = nodeIndexArticulationLinkId(igNodeIndexB);
    const bool isArtiA = nodeIndexIsArticulation(igNodeIndexA);
    const bool isArtiB = nodeIndexIsArticulation(igNodeIndexB);

    Ptr<float4> bodyVelocities = Ptr<float4>(sharedDesc[0].iter_solverBodyVelPool);

    // trap 4: the 3-slot staging
    AbSpatialVector vel0, vel1, delta0, delta1;
    {
        const float4 vec0 = bodyVelocities[readIndex];
        const float4 vec1 = bodyVelocities[readIndex + 32u];
        const float4 vec2 = bodyVelocities[readIndex + 64u];
        vel0 = asmSVec(float3(vec0.w, vec1.x, vec1.y), float3(vec0.x, vec0.y, vec0.z));
        delta0 = asmSVec(float3(vec2.y, vec2.z, vec2.w), float3(vec1.z, vec1.w, vec2.x));
    }
    {
        const float4 vec0 = bodyVelocities[readIndex + 96u];
        const float4 vec1 = bodyVelocities[readIndex + 128u];
        const float4 vec2 = bodyVelocities[readIndex + 160u];
        vel1 = asmSVec(float3(vec0.w, vec1.x, vec1.y), float3(vec0.x, vec0.y, vec0.z));
        delta1 = asmSVec(float3(vec2.y, vec2.z, vec2.w), float3(vec1.z, vec1.w, vec2.x));
    }

    // trap 5: reference counts
    float curRef0 = 1.0f;
    float curRef1 = 1.0f;
    {
        const uint bodyOffset = islandPool[0].mBodyStartIndex;
        const uint numDynamicBodies = islandPool[0].mBodyCount;
        Ptr<uint> encodedReferenceCount = Ptr<uint>(sharedDesc[0].iter_solverEncodedReferenceCount);
        const uint numTotalBodies = bodyOffset + numDynamicBodies + nbArticulations;
        const uint numSlabs = solverDesc[0].numSlabs;

        if (isArtiA)
        {
            const uint globalBodyIdA =
                batch[0].remappedBodyAIndex[lane] + numDynamicBodies + bodyOffset;
            curRef0 = float(pxgCountActiveSlabsBda(globalBodyIdA, numSlabs, numTotalBodies,
                                                   encodedReferenceCount));
        }
        else if (solverBodyId0 >= bodyOffset)
        {
            curRef0 = float(pxgCountActiveSlabsBda(solverBodyId0, numSlabs, numTotalBodies,
                                                   encodedReferenceCount));
        }

        if (isArtiB)
        {
            const uint globalBodyIdB =
                batch[0].remappedBodyBIndex[lane] + numDynamicBodies + bodyOffset;
            curRef1 = float(pxgCountActiveSlabsBda(globalBodyIdB, numSlabs, numTotalBodies,
                                                   encodedReferenceCount));
        }
        else if (solverBodyId1 >= bodyOffset)
        {
            curRef1 = float(pxgCountActiveSlabsBda(solverBodyId1, numSlabs, numTotalBodies,
                                                   encodedReferenceCount));
        }
    }

    // trap 6: deltaQ only for 1D batches
    float4 deltaQ0 = float4(0.0f, 0.0f, 0.0f, 1.0f);
    float4 deltaQ1 = float4(0.0f, 0.0f, 0.0f, 1.0f);
    if (constraintType == ASP_ARTI_CONSTRAINT_1D)
    {
        if (isArtiA)
        {
            const uint articBlockId = solverBodyId0 / 32u;
            const uint linkID = articBlockId * maxLinks + linkIndexA;
            deltaQ0 = abF4(gArticulationLinks + uint64_t(linkID) * PXG_AB_LINK_SIZE,
                           PXG_AB_LINK_mDeltaQ, solverBodyId0 & 31u);
        }
        else
        {
            Ptr<float> q = Ptr<float>(solverDesc[0].solverBodyTxIDataPool
                                      + uint64_t(solverBodyId0) * 64u);
            deltaQ0 = float4(q[0], q[1], q[2], q[3]);   // deltaBody2World.q at offset 0
        }

        if (isArtiB)
        {
            const uint articBlockId = solverBodyId1 / 32u;
            const uint linkID = articBlockId * maxLinks + linkIndexB;
            deltaQ1 = abF4(gArticulationLinks + uint64_t(linkID) * PXG_AB_LINK_SIZE,
                           PXG_AB_LINK_mDeltaQ, solverBodyId1 & 31u);
        }
        else
        {
            Ptr<float> q = Ptr<float>(solverDesc[0].solverBodyTxIDataPool
                                      + uint64_t(solverBodyId1) * 64u);
            deltaQ1 = float4(q[0], q[1], q[2], q[3]);
        }
    }

    AbSpatialVector impulse0 = asmSVec(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f));
    AbSpatialVector impulse1 = asmSVec(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f));

    SvxError error = { 0.0f, 0, 0.0f };

    if (constraintType == ASP_ARTI_CONTACT)
    {
        svxSolveExtContactBlockTGS(batch, vel0, vel1, delta0, delta1, lane,
                                   Ptr<PxgTGSBlockSolverContactHeader>(sharedDesc[0].iter_blockContactHeaders),
                                   Ptr<PxgTGSBlockSolverFrictionHeader>(sharedDesc[0].iter_blockFrictionHeaders),
                                   Ptr<PxgTGSBlockSolverContactPoint>(sharedDesc[0].iter_blockContactPoints),
                                   Ptr<PxgTGSBlockSolverContactFriction>(sharedDesc[0].iter_blockFrictions),
                                   Ptr<PxgArticulationBlockResponse>(sharedDesc[0].iter_artiResponse),
                                   elapsedTime, minPen, impulse0, impulse1,
                                   error, accumulateError, curRef0, curRef1);
    }
    else
    {
        svxSolveExt1DBlockTGS(batch, vel0, vel1, delta0, delta1, lane,
                              Ptr<PxgTGSBlockSolverConstraint1DHeader>(sharedDesc[0].iter_blockJointConstraintHeaders),
                              Ptr<PxgTGSBlockSolverConstraint1DCon>(sharedDesc[0].iter_blockJointConstraintRowsCon),
                              Ptr<PxgArticulationBlockResponse>(sharedDesc[0].iter_artiResponse),
                              deltaQ0, deltaQ1, elapsedTime, impulse0, impulse1,
                              accumulateError, curRef0, curRef1);
    }

    // trap 7: rigid writeback to the REMAPPED index
    if (!isArtiA)
    {
        const uint outIndex = batch[0].remappedBodyAIndex[lane];
        bodyVelocities[outIndex] = float4(vel0.bottom, vel0.top.x);
        bodyVelocities[outIndex + 32u] = float4(vel0.top.y, vel0.top.z, delta0.bottom.x, delta0.bottom.y);
        bodyVelocities[outIndex + 64u] = float4(delta0.bottom.z, delta0.top.x, delta0.top.y, delta0.top.z);
    }
    else
    {
        // trap 8: the impulse scatter + slab/state bookkeeping
        const uint index = aspDeltaVIndex(nbArticulations, maxLinks, solverBodyId0, linkIndexA, slabId);
        Ptr<float> z = Ptr<float>(deferredZ + uint64_t(index) * 24u);
        z[0] = impulse0.top.x; z[1] = impulse0.top.y; z[2] = impulse0.top.z;
        z[3] = impulse0.bottom.x; z[4] = impulse0.bottom.y; z[5] = impulse0.bottom.z;

        (Ptr<uint>(isSlabDirty))[(solverBodyId0 + slabId * nbArticulations) * 2u] = linkIndexA;

        const uint articBlockId = solverBodyId0 / 32u;
        abStoreU8(gArticulations + uint64_t(articBlockId) * PXG_AB_BLOCK_SIZE,
                  PXG_AB_mStateDirty, solverBodyId0 & 31u, ASP_DIRTY);
    }

    if (!isArtiB)
    {
        const uint outIndex = batch[0].remappedBodyBIndex[lane];
        bodyVelocities[outIndex] = float4(vel1.bottom, vel1.top.x);
        bodyVelocities[outIndex + 32u] = float4(vel1.top.y, vel1.top.z, delta1.bottom.x, delta1.bottom.y);
        bodyVelocities[outIndex + 64u] = float4(delta1.bottom.z, delta1.top.x, delta1.top.y, delta1.top.z);
    }
    else
    {
        const uint index = aspDeltaVIndex(nbArticulations, maxLinks, solverBodyId1, linkIndexB, slabId);
        Ptr<float> z = Ptr<float>(deferredZ + uint64_t(index) * 24u);
        z[0] = impulse1.top.x; z[1] = impulse1.top.y; z[2] = impulse1.top.z;
        z[3] = impulse1.bottom.x; z[4] = impulse1.bottom.y; z[5] = impulse1.bottom.z;

        (Ptr<uint>(isSlabDirty))[(solverBodyId1 + slabId * nbArticulations) * 2u + 1u] = linkIndexB;

        const uint articBlockId = solverBodyId1 / 32u;
        abStoreU8(gArticulations + uint64_t(articBlockId) * PXG_AB_BLOCK_SIZE,
                  PXG_AB_mStateDirty, solverBodyId1 & 31u, ASP_DIRTY);
    }

    errSumSq = error.sumOfSquares;
    errMax = error.maxError;
    errCount = uint(error.counter);
}
