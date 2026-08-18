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


// npArtiPropagate2.slang
//
// P8.101b/c: the two per-partition propagate kernels the solver's
// averageDeltaV step launches back to back (PxgArticulationCore.cpp:852-900):
//
//   artiPropagateImpulses2TGS  (forwardDynamic2.cu:3643, body :3560-3633)
//   artiPropagateVelocityTGS   (forwardDynamic2.cu:3759, body :3652-3749)
//
// Both: block (32, 2, 1) -- TWO warps per block (COMPUTE_UNCONSTRAINED_
// VELOCITES = 64, numWarpsPerBlock = 2, :854-856), so globalWarpIndex =
// blockIdx.x * 2 + threadIdx.y, NOT the one-warp shape of the solve kernels.
// The velocity kernel's grid is 2D: blockIdx.y IS the slab id.
//
// ===========================================================================
// artiPropagateImpulses2 -- wrapper-level facts, each a sweep mutant:
//  1. the whole body sits under `if (dirtyFlag)` (any dirty bit);
//  2. partitionId == 0 -> PxcFsFlushVelocity + mStateDirty = 0 (ZERO, not
//     eVEL_DIRTY);
//  3. partitionId != 0 runs ONLY under eHAS_IMPULSES, then ONLY when
//     mPartitionAverageScale[gti + partitionId*nb] > 0;
//  4. the per-partition path base: offset2 = globalWarpIndex + partitionId *
//     ceil(nb/32) * wordSize, in units of PxgArticulationBitFieldStackData;
//  5. scale = 1/countActiveSlabs(...) -- the P8.99c helper over the shared
//     desc's encodedReferenceCount, offset by islandContextPool[0].mBodyCount
//     + mBodyStartIndex;
//  6. the dirty flags are deliberately NOT cleared afterwards (the CUDA's
//     own comment: partial work).
//
// artiPropagateVelocity -- wrapper-level facts, each a sweep mutant:
//  7. the early-out is partitionAverageScale == 0 (a WRITE-side skip: nobody
//     reads the output for this articulation in this partition);
//  8. slabDirtyMasks advances by nbArticulations*nbSlabs uint4 entries per
//     PARTITION, then indexes [gti + slabId*nbArticulations];
//  9. two (dirtyIndex, writeIndex) pairs per uint4 -- x/y then z/w;
// 10. the deltaV correction runs only when partitionId != 0 AND dirtyLink !=
//     0xFFFFFFFF: pathToRoot = wordA & ~wordB (dirtyIndex's path minus
//     dirtyLink's), walked LOWEST-first with the DIRECT-order NoJVel hop
//     seeded from mCommonLinkDeltaVelocity;
// 11. the TGS output packs velocity + deltaMotion into THREE float4 slots at
//     writeIndex, +32, +64 -- (vb.x vb.y vb.z vt.x), (vt.y vt.z db.x db.y),
//     (db.z dt.x dt.y dt.z).
// ===========================================================================

import pxgSlangCommon;
import pxgSlangArticulationBlocks;
import pxgSlangSpatialMatrix;
import pxgSlangArticulationDynamic;
import pxgSlangArticulationDesc;
import pxgSlangSolverDesc;
import npArtiSolveTendons;
import npArtiAvgImpulse;
import npArtiFlushVel;
import npArtiSolveSelf;

static const uint P2_NONE = 0xFFFFFFFFu;
static const uint P2_VEL_DIRTY = 1u;      // PxgArtiStateDirtyFlag::eVEL_DIRTY
static const uint P2_HAS_IMPULSES = 2u;   // PxgArtiStateDirtyFlag::eHAS_IMPULSES

// ---------------------------------------------------------------------------
// artiPropagateImpulses2TGS
// ---------------------------------------------------------------------------

struct P2iParams
{
    uint64_t scDesc;      // @0  PxgArticulationCoreDesc*
    uint64_t solverDesc;  // @8  PxgSolverCoreDesc*
    uint64_t sharedDesc;  // @16 PxgSolverSharedDesc<IterativeSolveDataTGS>*
    uint partitionId;     // @24
    uint pad0;            // @28 -> 32
}

[[vk::push_constant]]
ConstantBuffer<P2iParams> gP2i;

// P8.109: the body, shared by the TGS and PGS entries.  The ONLY
// type-dependent value in the template (forwardDynamic2.cu:3560) is
// iterativeData.solverEncodedReferenceCount, which both mirrors carry at the
// same offset (@176) -- each entry resolves it through its OWN typed mirror
// and passes the pointer in.
void p2iPropagateImpulses2(uint64_t d, uint64_t solverDescA, uint64_t encodedRefCountA,
                           uint partitionId, uint lane, uint globalWarpIndex)
{
    const uint globalThreadIndex = globalWarpIndex * 32u + lane;

    const uint nbArticulations = artiU32Field(d, PXG_ARTI_CORE_NB_ARTICULATIONS);
    if (globalThreadIndex >= nbArticulations)
        return;

    const uint64_t w = uint64_t(globalWarpIndex);
    const uint64_t artiBlock = artiPtrField(d, PXG_ARTI_CORE_ARTICULATION_BLOCKS)
                             + w * PXG_AB_BLOCK_SIZE;
    const uint maxLinks = artiU32Field(d, PXG_ARTI_CORE_MAX_LINKS);
    const uint maxDofs = artiU32Field(d, PXG_ARTI_CORE_MAX_DOFS);

    const uint dirtyFlag = abU8(artiBlock, PXG_AB_mStateDirty, lane);
    if (dirtyFlag == 0u)
        return;

    const uint64_t linkBase = artiPtrField(d, PXG_ARTI_CORE_LINK_BLOCKS)
                            + w * uint64_t(maxLinks) * PXG_AB_LINK_SIZE;
    const uint64_t dofBase = artiPtrField(d, PXG_ARTI_CORE_DOF_BLOCKS)
                           + w * uint64_t(maxDofs) * PXG_AB_DOF_SIZE;
    const uint numLinks = abU32(artiBlock, PXG_AB_mNumLinks, lane);

    if (partitionId == 0u)
    {
        const bool fixBase =
            (abU32(artiBlock, PXG_AB_mFlags, lane) & PXG_ARTI_FLAG_FIX_BASE) != 0u;
        fvFlushVelocity(artiBlock, linkBase, dofBase, numLinks, fixBase, lane);
        abStoreU8(artiBlock, PXG_AB_mStateDirty, lane, 0u);   // ZERO, not eVEL_DIRTY
    }
    else if ((dirtyFlag & P2_HAS_IMPULSES) != 0u)
    {
        const uint offset = globalThreadIndex + partitionId * nbArticulations;
        const float partitionAverageScale =
            (Ptr<float>(artiPtrField(d, PXG_ARTI_CORE_PARTITION_AVERAGE_SCALE)))[offset];

        if (partitionAverageScale > 0.0f)
        {
            const uint nbSlabs = artiU32Field(d, PXG_ARTI_CORE_NB_SLABS);
            const uint64_t slabHasChanges = artiPtrField(d, PXG_ARTI_CORE_SLAB_HAS_CHANGES);
            const uint dirtyLink =
                (Ptr<uint>(artiPtrField(d, PXG_ARTI_CORE_IMPULSE_HOLDING_LINK)))[offset];

            const uint nbArticulationBatchPerPartition = (nbArticulations + 31u) / 32u;
            const uint wordSize = (maxLinks + 63u) / 64u;
            const uint offset2 = globalWarpIndex
                               + partitionId * nbArticulationBatchPerPartition * wordSize;
            const uint64_t pathToRootPerPartition =
                artiPtrField(d, PXG_ARTI_CORE_PATH_TO_ROOTS_PER_PARTITION)
                + uint64_t(offset2) * PXG_AB_BITFIELD_STACK_SIZE;

            Ptr<PxgSolverCoreDescGpu> solverDesc = Ptr<PxgSolverCoreDescGpu>(solverDescA);
            Ptr<PxgIslandContextGpu> island = Ptr<PxgIslandContextGpu>(solverDesc[0].islandContextPool);
            const uint articulationReferenceCountOffset = island[0].mBodyCount + island[0].mBodyStartIndex;
            const uint numSolverBodies = articulationReferenceCountOffset + nbArticulations;

            Ptr<uint> encodedRefCount = Ptr<uint>(encodedRefCountA);
            const uint referenceCount = pxgCountActiveSlabsBda(
                articulationReferenceCountOffset + globalThreadIndex, nbSlabs, numSolverBodies,
                encodedRefCount);
            const float scale = 1.0f / float(referenceCount);

            const uint64_t impulses = artiPtrField(d, PXG_ARTI_CORE_IMPULSES);

            aiAverageAndPropagate2(slabHasChanges, impulses, artiBlock, linkBase, dofBase,
                                   globalThreadIndex, maxLinks, nbArticulations, nbSlabs,
                                   scale, pathToRootPerPartition, wordSize,
                                   dirtyLink, dirtyFlag, lane);

            // the dirty flags are deliberately NOT cleared (:3627)
        }
    }
}

[shader("compute")]
[numthreads(32, 2, 1)]
void artiPropagateImpulses2TGS(uint3 groupThreadId : SV_GroupThreadID,
                               uint3 groupId : SV_GroupID)
{
    Ptr<PxgSolverSharedDescTGSGpu> sharedDesc =
        Ptr<PxgSolverSharedDescTGSGpu>(gP2i.sharedDesc);
    p2iPropagateImpulses2(gP2i.scDesc, gP2i.solverDesc,
                          sharedDesc[0].iter_solverEncodedReferenceCount, gP2i.partitionId,
                          groupThreadId.x, groupId.x * 2u + groupThreadId.y);
}

// P8.109: the PGS twin (forwardDynamic2.cu:3635) -- the same body over
// PxgSolverSharedDesc<IterativeSolveData>.
[shader("compute")]
[numthreads(32, 2, 1)]
void artiPropagateImpulses2PGS(uint3 groupThreadId : SV_GroupThreadID,
                               uint3 groupId : SV_GroupID)
{
    Ptr<PxgSolverSharedDescPGSGpu> sharedDesc =
        Ptr<PxgSolverSharedDescPGSGpu>(gP2i.sharedDesc);
    p2iPropagateImpulses2(gP2i.scDesc, gP2i.solverDesc,
                          sharedDesc[0].iter_solverEncodedReferenceCount, gP2i.partitionId,
                          groupThreadId.x, groupId.x * 2u + groupThreadId.y);
}

// ---------------------------------------------------------------------------
// artiPropagateVelocityTGS
// ---------------------------------------------------------------------------

struct P2vParams
{
    uint64_t scDesc;          // @0  PxgArticulationCoreDesc*
    uint partitionId;         // @8
    uint pad0;                // @12
    uint64_t velocityOutput;  // @16 float4*
}

[[vk::push_constant]]
ConstantBuffer<P2vParams> gP2v;

// P8.109: the body, shared by the TGS and PGS entries
// (artiPropagateVelocityInternal<isTGS>, forwardDynamic2.cu:3652) -- they
// differ ONLY in the output tail.
void p2vPropagateVelocity(uint64_t d, uint partitionId, uint64_t velocityOutput, bool isTGS,
                          uint lane, uint globalWarpIndex, uint slabIdIn)
{
    const uint globalThreadIndex = globalWarpIndex * 32u + lane;

    const uint nbArticulations = artiU32Field(d, PXG_ARTI_CORE_NB_ARTICULATIONS);
    if (globalThreadIndex >= nbArticulations)
        return;

    const uint slabId = slabIdIn;   // the 2D grid: blockIdx.y IS the slab

    const uint offset = globalThreadIndex + partitionId * nbArticulations;
    const float partitionAverageScale =
        (Ptr<float>(artiPtrField(d, PXG_ARTI_CORE_PARTITION_AVERAGE_SCALE)))[offset];
    if (partitionAverageScale == 0.0f)
        return;   // nobody reads this articulation's output in this partition

    const uint64_t w = uint64_t(globalWarpIndex);
    const uint64_t artiBlock = artiPtrField(d, PXG_ARTI_CORE_ARTICULATION_BLOCKS)
                             + w * PXG_AB_BLOCK_SIZE;
    const uint maxLinks = artiU32Field(d, PXG_ARTI_CORE_MAX_LINKS);
    const uint maxDofs = artiU32Field(d, PXG_ARTI_CORE_MAX_DOFS);
    const uint wordSize = (maxLinks + 63u) / 64u;
    const uint nbSlabs = artiU32Field(d, PXG_ARTI_CORE_NB_SLABS);
    const uint slabPartitionOffset = nbArticulations * nbSlabs;

    const uint dirtyLink =
        (Ptr<uint>(artiPtrField(d, PXG_ARTI_CORE_IMPULSE_HOLDING_LINK)))[offset];

    const uint64_t linkBase = artiPtrField(d, PXG_ARTI_CORE_LINK_BLOCKS)
                            + w * uint64_t(maxLinks) * PXG_AB_LINK_SIZE;
    const uint64_t dofBase = artiPtrField(d, PXG_ARTI_CORE_DOF_BLOCKS)
                           + w * uint64_t(maxDofs) * PXG_AB_DOF_SIZE;
    const uint64_t linkBitFields = artiPtrField(d, PXG_ARTI_CORE_PATH_TO_ROOT_BITFIELD_BLOCKS)
        + w * uint64_t(maxLinks) * uint64_t(wordSize) * PXG_AB_BITFIELD_STACK_SIZE;

    // one uint4 per (articulation, slab), stepped per PARTITION
    const uint64_t slabDirtyMasks = artiPtrField(d, PXG_ARTI_CORE_SLAB_DIRTY_MASKS)
        + uint64_t(slabPartitionOffset) * uint64_t(partitionId) * 16u;
    const uint4 dirtyIndex4 =
        (Ptr<uint4>(slabDirtyMasks))[globalThreadIndex + slabId * nbArticulations];

    for (uint i = 0u; i < 2u; ++i)
    {
        const uint dirtyIndex = (i == 0u) ? dirtyIndex4.x : dirtyIndex4.z;
        const uint writeIndex = (i == 0u) ? dirtyIndex4.y : dirtyIndex4.w;
        if (dirtyIndex != P2_NONE)
        {
            const uint64_t dLink = linkBase + uint64_t(dirtyIndex) * PXG_AB_LINK_SIZE;
            AbSpatialVector velocity = svLoadSV(dLink, PXG_AB_LINK_mMotionVelocity, lane);

            if (partitionId != 0u && dirtyLink != P2_NONE)
            {
                AbSpatialVector deltaV = svLoadSV(artiBlock, PXG_AB_mCommonLinkDeltaVelocity, lane);

                for (uint j = 0u; j < wordSize; ++j)
                {
                    const uint wordOffset = j * 64u;
                    const uint64_t wordA = (Ptr<uint64_t>(linkBitFields
                        + (uint64_t(dirtyIndex) * wordSize + j) * PXG_AB_BITFIELD_STACK_SIZE))[lane];
                    const uint64_t wordB = (Ptr<uint64_t>(linkBitFields
                        + (uint64_t(dirtyLink) * wordSize + j) * PXG_AB_BITFIELD_STACK_SIZE))[lane];
                    uint64_t pathToRoot = wordA & (~wordB);

                    while (pathToRoot != 0uL)
                    {
                        const uint index = svLowestBit64(pathToRoot) + wordOffset;
                        deltaV = svPropagateAccelNoJVel(linkBase, index, dofBase, deltaV, lane);
                        pathToRoot &= (pathToRoot - 1uL);   // clears the lowest set bit
                    }
                }

                velocity = asmSVecAdd(velocity, deltaV);
            }

            Ptr<float4> outp = Ptr<float4>(velocityOutput);
            if (isTGS)
            {
                // the TGS packing: three float4 slots at writeIndex, +32, +64
                const AbSpatialVector deltaMotion =
                    svLoadSV(dLink, PXG_AB_LINK_mDeltaMotion, lane);
                outp[writeIndex] = float4(velocity.bottom.x, velocity.bottom.y,
                                          velocity.bottom.z, velocity.top.x);
                outp[writeIndex + 32u] = float4(velocity.top.y, velocity.top.z,
                                                deltaMotion.bottom.x, deltaMotion.bottom.y);
                outp[writeIndex + 64u] = float4(deltaMotion.bottom.z, deltaMotion.top.x,
                                                deltaMotion.top.y, deltaMotion.top.z);
            }
            else
            {
                // the PGS packing: two slots, w = 0, NO deltaMotion
                outp[writeIndex] = float4(velocity.bottom, 0.0f);
                outp[writeIndex + 32u] = float4(velocity.top, 0.0f);
            }
        }
    }
}

[shader("compute")]
[numthreads(32, 2, 1)]
void artiPropagateVelocityTGS(uint3 groupThreadId : SV_GroupThreadID,
                              uint3 groupId : SV_GroupID)
{
    p2vPropagateVelocity(gP2v.scDesc, gP2v.partitionId, gP2v.velocityOutput, true,
                         groupThreadId.x, groupId.x * 2u + groupThreadId.y, groupId.y);
}

// P8.109: the PGS twin (forwardDynamic2.cu:3752) --
// artiPropagateVelocityInternal<false>.
[shader("compute")]
[numthreads(32, 2, 1)]
void artiPropagateVelocity(uint3 groupThreadId : SV_GroupThreadID,
                           uint3 groupId : SV_GroupID)
{
    p2vPropagateVelocity(gP2v.scDesc, gP2v.partitionId, gP2v.velocityOutput, false,
                         groupThreadId.x, groupId.x * 2u + groupThreadId.y, groupId.y);
}
