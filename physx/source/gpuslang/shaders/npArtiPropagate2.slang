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

// ===========================================================================
// P8.110: artiSaveVelocity1TPGS (forwardDynamic2.cu:3255-3313).  Grid
// ceil(nb/64), block (32,2,1).  The PGS "save velocities" step, launched
// once per PGS solve from PxgCudaSolverCore::solveContactMultiBlockParallel:
// flush any pending impulses, then snapshot the motion and joint velocities
// into their mPos* twins.
//
// TRAPS, each a sweep mutant:
//  1. The average runs ONLY under the eHAS_IMPULSES bit (&2); a lane with
//     eVEL_DIRTY alone flushes WITHOUT averaging.
//  2. The flush runs for ANY non-zero dirty value, and the flag is cleared
//     to 0 inside that branch.
//  3. The two snapshot loops run UNCONDITIONALLY -- clean lanes copy too.
//  4. The link snapshot starts at link 0 (the root included); the dof
//     snapshot is bounded by the HEADER's mTotalDofs field, not by a
//     recomputed per-link sum (the fixture seeds them apart on some lanes).
//  5. The copies are RAW -- load/store with no arithmetic.
// ===========================================================================

struct SvpParams
{
    uint64_t scDesc;   // @0  PxgArticulationCoreDesc*
}

[[vk::push_constant]]
ConstantBuffer<SvpParams> gSvp;

[shader("compute")]
[numthreads(32, 2, 1)]
void artiSaveVelocity1TPGS(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint64_t d = gSvp.scDesc;
    const uint lane = groupThreadId.x;
    const uint globalWarpIndex = groupId.x * 2u + groupThreadId.y;
    const uint globalThreadIndex = globalWarpIndex * 32u + lane;

    const uint nbArticulations = artiU32Field(d, PXG_ARTI_CORE_NB_ARTICULATIONS);
    if (globalThreadIndex >= nbArticulations)
        return;

    const uint maxLinks = artiU32Field(d, PXG_ARTI_CORE_MAX_LINKS);
    const uint maxDofs = artiU32Field(d, PXG_ARTI_CORE_MAX_DOFS);

    const uint64_t w = uint64_t(globalWarpIndex);
    const uint64_t artiBlock = artiPtrField(d, PXG_ARTI_CORE_ARTICULATION_BLOCKS)
                             + w * PXG_AB_BLOCK_SIZE;
    const uint64_t linkBase = artiPtrField(d, PXG_ARTI_CORE_LINK_BLOCKS)
                            + w * uint64_t(maxLinks) * PXG_AB_LINK_SIZE;
    const uint64_t dofBase = artiPtrField(d, PXG_ARTI_CORE_DOF_BLOCKS)
                           + w * uint64_t(maxDofs) * PXG_AB_DOF_SIZE;

    const uint numLinks = abU32(artiBlock, PXG_AB_mNumLinks, lane);
    const uint totalDofs = abU32(artiBlock, PXG_AB_mTotalDofs, lane);   // trap 4
    const bool fixBase = (abU32(artiBlock, PXG_AB_mFlags, lane) & PXG_ARTI_FLAG_FIX_BASE) != 0u;

    const uint stateDirty = abU8(artiBlock, PXG_AB_mStateDirty, lane);
    if (stateDirty != 0u)
    {
        if ((stateDirty & P2_HAS_IMPULSES) != 0u)   // trap 1
        {
            aiAverageAndPropagate(artiPtrField(d, PXG_ARTI_CORE_SLAB_HAS_CHANGES),
                                  artiPtrField(d, PXG_ARTI_CORE_IMPULSES), artiBlock, linkBase,
                                  dofBase, globalThreadIndex, maxLinks, nbArticulations,
                                  artiU32Field(d, PXG_ARTI_CORE_NB_SLABS), lane);
        }
        fvFlushVelocity(artiBlock, linkBase, dofBase, numLinks, fixBase, lane);   // trap 2
        abStoreU8(artiBlock, PXG_AB_mStateDirty, lane, 0u);
    }

    // trap 3 + 5: unconditional raw snapshots
    for (uint linkID = 0u; linkID < numLinks; ++linkID)
    {
        const uint64_t tLink = linkBase + uint64_t(linkID) * PXG_AB_LINK_SIZE;
        const AbSpatialVector mv = svLoadSV(tLink, PXG_AB_LINK_mMotionVelocity, lane);
        abStoreSpatialVector(tLink, PXG_AB_LINK_mPosMotionVelocity, lane, mv.top, mv.bottom);
    }
    for (uint dofId = 0u; dofId < totalDofs; ++dofId)
    {
        const uint64_t dd = dofBase + uint64_t(dofId) * PXG_AB_DOF_SIZE;
        abStoreF32(dd, PXG_AB_DOF_mPosJointVelocities, lane,
                   abF32(dd, PXG_AB_DOF_mJointVelocities, lane));
    }
}

// ===========================================================================
// P8.111: artiComputeDependencies (forwardDynamic2.cu:3315-3480).  Grid
// ceil(nb/64), block (32,2,1), one THREAD per articulation.  Precomputes,
// per partition: the common node (mImpulseHoldingLink), the slab-count
// average scale (mPartitionAverageScale), and the per-partition
// propagate-up path (mPathToRootsPerPartition) that artiPropagateImpulses2
// and artiPropagateVelocity consume.  Pure per-thread bitfield algebra --
// no warp ops, no block SoA (only the desc-level bitfield pools).
//
// TRAPS, each a sweep mutant unless marked:
//  1. The zero loop clears FOUR pools: the three mTemp* scratches AND the
//     partition-0 slot of mPathToRootsPerPartition (at w * wordSize).
//  2. The common-link search walks words HIGHEST-first and takes the
//     highest set bit; the sharedPathEmpty mode searches the RAW link path,
//     the other mode the INTERSECTION with sharedBitField.
//  3. commonNode = min(commonLink, commonNode) accumulates ACROSS
//     partitions -- never reset.
//  4. rootBitField reloads from the common node's path ONLY when commonNode
//     changed (initially 0xFFFFFFFF == old, so no reload until a dirty link
//     exists -- which is what keeps the indexed read in bounds).
//  5. The per-partition output goes to the NEXT partition's slot, and only
//     when nextPartition < nbPartitions.
//  6. After each partition: sharedBitField := rootBitField, sharedPathEmpty
//     keeps accumulating (it is MONOTONIC -- once false, false forever),
//     pathToRootBitField is zeroed.
//  7. count counts SLABS with at least one dirty index (x or z);
//     scale = count < 1 ? 0 : 1/count.
//
// SEVENTH UPSTREAM-BUG CANDIDATE (transcribed verbatim, not fixed): the
// zero loop indexes mPathToRootsPerPartition at `w * wordSize`, but the
// per-partition write (and the propagate2 READ, :3609) at
// `w + partition * nbArticulationBatchPerPartition * wordSize` -- for
// wordSize >= 2 (more than 64 links) adjacent warps' slots OVERLAP, a
// cross-warp data race on the same (slot, lane) words.  Invisible at
// wordSize == 1, which is every scene today.  The fixture's rounds are
// chosen race-free (multi-warp only at wordSize 1; wordSize 2 single-warp).
// ===========================================================================

struct CdParams
{
    uint64_t scDesc;      // @0  PxgArticulationCoreDesc*
    uint nbPartitions;    // @8
    uint pad0;            // @12 -> 16
}

[[vk::push_constant]]
ConstantBuffer<CdParams> gCd;

[shader("compute")]
[numthreads(32, 2, 1)]
void artiComputeDependencies(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint64_t d = gCd.scDesc;
    const uint nbPartitions = gCd.nbPartitions;
    const uint lane = groupThreadId.x;
    const uint globalWarpIndex = groupId.x * 2u + groupThreadId.y;
    const uint globalThreadIndex = globalWarpIndex * 32u + lane;

    const uint nbArticulations = artiU32Field(d, PXG_ARTI_CORE_NB_ARTICULATIONS);
    if (globalThreadIndex >= nbArticulations)
        return;

    const uint articulationId = globalThreadIndex;
    const uint maxLinks = artiU32Field(d, PXG_ARTI_CORE_MAX_LINKS);
    const uint nbSlabs = artiU32Field(d, PXG_ARTI_CORE_NB_SLABS);
    const uint64_t slabDirtyMasksBase = artiPtrField(d, PXG_ARTI_CORE_SLAB_DIRTY_MASKS);
    Ptr<uint> dirtyLink = Ptr<uint>(artiPtrField(d, PXG_ARTI_CORE_IMPULSE_HOLDING_LINK));
    Ptr<float> partitionAverageScale =
        Ptr<float>(artiPtrField(d, PXG_ARTI_CORE_PARTITION_AVERAGE_SCALE));

    const uint slabPartitionStride = nbArticulations * nbSlabs;
    const uint wordSize = (maxLinks + 63u) / 64u;
    const uint64_t w = uint64_t(globalWarpIndex);

    const uint64_t pathToRootBitField = artiPtrField(d, PXG_ARTI_CORE_TEMP_PATH_TO_ROOT_BITFIELD_BLOCKS)
                                      + w * wordSize * PXG_AB_BITFIELD_STACK_SIZE;
    const uint64_t sharedBitField = artiPtrField(d, PXG_ARTI_CORE_TEMP_SHARED_BITFIELD_BLOCKS)
                                  + w * wordSize * PXG_AB_BITFIELD_STACK_SIZE;
    const uint64_t rootBitField = artiPtrField(d, PXG_ARTI_CORE_TEMP_ROOT_BITFIELD_BLOCKS)
                                + w * wordSize * PXG_AB_BITFIELD_STACK_SIZE;
    const uint64_t linkBitFieldBlocks = artiPtrField(d, PXG_ARTI_CORE_PATH_TO_ROOT_BITFIELD_BLOCKS)
        + w * uint64_t(maxLinks) * wordSize * PXG_AB_BITFIELD_STACK_SIZE;
    const uint64_t pathsPerPartition = artiPtrField(d, PXG_ARTI_CORE_PATH_TO_ROOTS_PER_PARTITION);

    // trap 1: four pools zeroed, incl. the partition-0 slot at w * wordSize
    {
        const uint64_t p0 = pathsPerPartition + w * wordSize * PXG_AB_BITFIELD_STACK_SIZE;
        for (uint j = 0u; j < wordSize; ++j)
        {
            abStoreU64(pathToRootBitField + uint64_t(j) * PXG_AB_BITFIELD_STACK_SIZE, 0u, lane, 0uL);
            abStoreU64(rootBitField + uint64_t(j) * PXG_AB_BITFIELD_STACK_SIZE, 0u, lane, 0uL);
            abStoreU64(sharedBitField + uint64_t(j) * PXG_AB_BITFIELD_STACK_SIZE, 0u, lane, 0uL);
            abStoreU64(p0 + uint64_t(j) * PXG_AB_BITFIELD_STACK_SIZE, 0u, lane, 0uL);
        }
    }

    uint commonNode = 0xFFFFFFFFu;
    uint oldCommonNode = commonNode;
    bool sharedPathEmpty = true;

    const uint nbArticulationBatchPerPartition = (nbArticulations + 31u) / 32u;

    for (uint partition = 0u; partition < nbPartitions; ++partition)
    {
        const uint partitionOffset = articulationId + nbArticulations * partition;
        const uint64_t slabDirtyMasks = slabDirtyMasksBase
                                      + uint64_t(slabPartitionStride) * partition * 16u;

        float count = 0.0f;

        for (uint s = 0u; s < nbSlabs; ++s)
        {
            const uint slabDirtyIndex = articulationId + s * nbArticulations;
            const uint4 dirtyIndex2 = (Ptr<uint4>(slabDirtyMasks))[slabDirtyIndex];

            if (dirtyIndex2.x != 0xFFFFFFFFu || dirtyIndex2.z != 0xFFFFFFFFu)
                count += 1.0f;   // trap 7

            for (uint i = 0u; i < 2u; ++i)
            {
                const uint dirtyIndex = (i == 0u) ? dirtyIndex2.x : dirtyIndex2.z;
                if (dirtyIndex != 0xFFFFFFFFu)
                {
                    const uint64_t linkPathToRoot = linkBitFieldBlocks
                        + uint64_t(dirtyIndex) * wordSize * PXG_AB_BITFIELD_STACK_SIZE;
                    uint commonLink = 0u;
                    if (sharedPathEmpty)
                    {
                        // trap 2: the raw path, highest word first
                        for (int j = int(wordSize) - 1; j >= 0; --j)
                        {
                            const uint64_t word = abU64(
                                linkPathToRoot + uint64_t(j) * PXG_AB_BITFIELD_STACK_SIZE, 0u,
                                lane);
                            if (word != 0uL)
                            {
                                commonLink = svHighestBit64(word) + uint(j) * 64u;
                                break;
                            }
                        }
                    }
                    else
                    {
                        // trap 2: the INTERSECTION with the shared path
                        for (int j = int(wordSize) - 1; j >= 0; --j)
                        {
                            const uint64_t word0 = abU64(
                                sharedBitField + uint64_t(j) * PXG_AB_BITFIELD_STACK_SIZE, 0u,
                                lane);
                            const uint64_t word1 = abU64(
                                linkPathToRoot + uint64_t(j) * PXG_AB_BITFIELD_STACK_SIZE, 0u,
                                lane);
                            const uint64_t word = word0 & word1;
                            if (word != 0uL)
                            {
                                commonLink = svHighestBit64(word) + uint(j) * 64u;
                                break;
                            }
                        }
                    }

                    commonNode = min(commonLink, commonNode);   // trap 3

                    for (uint j = 0u; j < wordSize; ++j)
                    {
                        const uint64_t jOff = uint64_t(j) * PXG_AB_BITFIELD_STACK_SIZE;
                        const uint64_t word = abU64(linkPathToRoot + jOff, 0u, lane);
                        abStoreU64(pathToRootBitField + jOff, 0u, lane,
                                   abU64(pathToRootBitField + jOff, 0u, lane) | word);
                        abStoreU64(sharedBitField + jOff, 0u, lane,
                                   abU64(sharedBitField + jOff, 0u, lane) | word);
                        sharedPathEmpty = sharedPathEmpty && (word == 0uL);
                    }
                }
            }
        }

        // trap 4: reload only on change
        if (commonNode != oldCommonNode)
        {
            const uint64_t linkPathToRoot = linkBitFieldBlocks
                + uint64_t(commonNode) * wordSize * PXG_AB_BITFIELD_STACK_SIZE;
            for (uint j = 0u; j < wordSize; ++j)
            {
                const uint64_t jOff = uint64_t(j) * PXG_AB_BITFIELD_STACK_SIZE;
                abStoreU64(rootBitField + jOff, 0u, lane, abU64(linkPathToRoot + jOff, 0u, lane));
            }
            oldCommonNode = commonNode;
        }

        // trap 5: the NEXT partition's slot (the engine's own addressing --
        // see the seventh-candidate note above)
        const uint nextPartition = partition + 1u;
        if (nextPartition < nbPartitions)
        {
            const uint64_t outSlot = pathsPerPartition
                + uint64_t(globalWarpIndex
                           + nextPartition * nbArticulationBatchPerPartition * wordSize)
                      * PXG_AB_BITFIELD_STACK_SIZE;
            for (uint j = 0u; j < wordSize; ++j)
            {
                const uint64_t jOff = uint64_t(j) * PXG_AB_BITFIELD_STACK_SIZE;
                abStoreU64(outSlot + jOff, 0u, lane,
                           abU64(pathToRootBitField + jOff, 0u, lane));
            }
        }

        // trap 6
        for (uint j = 0u; j < wordSize; ++j)
        {
            const uint64_t jOff = uint64_t(j) * PXG_AB_BITFIELD_STACK_SIZE;
            const uint64_t word = abU64(rootBitField + jOff, 0u, lane);
            abStoreU64(sharedBitField + jOff, 0u, lane, word);
            sharedPathEmpty = sharedPathEmpty && (word == 0uL);
            abStoreU64(pathToRootBitField + jOff, 0u, lane, 0uL);
        }

        const float scale = count < 1.0f ? 0.0f : 1.0f / count;
        dirtyLink[partitionOffset] = commonNode;
        partitionAverageScale[partitionOffset] = scale;
    }
}
