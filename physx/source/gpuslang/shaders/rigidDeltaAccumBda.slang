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

// rigidDeltaAccumBda.slang -- P9.44: the deformable/particle -> RIGID impulse
// coupling, the five kernels of gpusimulationcontroller/src/CUDA/rigidDeltaAccum.cu:
//
//   accumulateDeltaVRigidFirstLaunch            (.cu:55-220)
//   accumulateDeltaVRigidSecondLaunch           (.cu:224-429)
//   clearDeltaVRigidSecondLaunchMulti           (.cu:433-516)
//   accumulateDeltaVRigidSecondLaunchMultiStage1 (.cu:521-668)
//   accumulateDeltaVRigidSecondLaunchMultiStage2 (.cu:671-874)
//
// WHAT THE CHAIN DOES.  Every deformable-vs-rigid and particle-vs-rigid solve
// kernel writes one per-contact rigid delta (linear at deltaV[i], angular at
// deltaV[i + n]; the linear .w carries the contact's relaxation weight) next to a
// PxNodeIndex list SORTED by rigid body.  First turns deltaV in place into a
// per-block segmented inclusive scan and emits each block's trailing run (id +
// partial sum); Second finds the last contact of every run, adds the previous
// block's carry when the run crosses the block boundary, and applies the total,
// divided by max(1, weight), to the solver body's delta velocity (or, for an
// articulation link, stores the negated impulse into the link's scratch impulse
// and marks the articulation eHAS_IMPULSES).  The Multi trio is the same
// reduction for particle-rigid ATTACHMENTS, whose id list is sorted only per
// attachment buffer, so one body can head several runs: Stage1 atomically sums
// the relaxation denominators into a scratch (tempDenom / mDeltaScale) that the
// clear kernel zeroes first, Stage2 atomically adds the scaled sums.
//
// Launch sites: PxgParticleSystemCore.cpp:1838-1923 and PxgFEMCore.cpp:260-335
// for First/Second (both grid 32 x block 512: PxgParticleSystemKernelGridDim /
// PxgSoftBodyKernelGridDim ACCUMULATE_DELTA), PxgParticleSystemCore.cpp
// solveRigidAttachments for the Multi trio (same shape).  P9.40's guard found
// First/Second launched in all 23 deformable scenes and all 10 particle scenes
// (x240..x3100 per run): every rigid body a cloth, softbody or particle touched
// was pushed by nothing.
//
// PORTABILITY (P2.1 contract: workgroup-uniform barriers, no lockstep assumed):
//   * The three float segmented Hillis-Steele scans (32 lanes in a warp, 16 warp
//     totals in warp 0, 32 block totals) are shuffle-emulated through groupshared
//     arrays, one barrier pair per round, so the ASSOCIATION of the float sums is
//     the CUDA's exactly -- a serialized scan would land different ulps.
//   * The .cu closes First's tile without a barrier (lane-31 writes of the next
//     tile vs. this tile's reads of the warp arrays rely on warp convergence); the
//     tile is closed explicitly here, as frozenHistogram.slang / broadphase*.slang.
//   * The warp-0 second-level scan only ever reads lanes 0..15 (lanes 16..31 hold
//     the sentinel and are written to slots nobody reads); it runs on tid < 16.
//   * Second's PGS `linearVelocity += linVel * ratio` adds a PROVABLE zero to .w
//     (the .cu sets linVel.w = 0 first).  llvmpipe folds x + 0 -> x for such a
//     product (P9.43b, fifth platform rule), which keeps a -0 the CUDA turns into
//     +0; the .w lane is canonicalised in integer space instead (rdaPlusZero).
//   * Float atomics (Multi stages) are a compare-exchange loop on the raw bits, the
//     shape skinning.slang uses.  The retry is not a formality: with 256 lanes of one
//     block aimed at a single word, dropping it loses adds on this platform (the
//     fixture's atomicCollision case).  The ORDER of concurrent adds is not
//     deterministic, so every fixture case keeps the summands order-independent --
//     at most two contributions per body, or 256 equal powers of two.
//
// ENGINE ABI (PxgVulkanKernelAbi.cpp): First {8x5} = 40 B; Second {8x9, 1} = 80 B
// (isTGS byte @72); clear {8x6} = 48 B; Stage1/2 {8x10, 1, 4, 1} = 96 B
// (useLocalRelax byte @80, globalRelaxationCoefficient @84, isTGS byte @88).

import pxgSlangPrepDesc;           // PxgPrePrepDescGpu (@272 solverBodyIndices)
import pxgSlangSolverDesc;         // PxgSolverCoreDescGpu, PxgSolverSharedDescPGSGpu
import pxgSlangArticulationDesc;   // PXG_ARTI_CORE_*, artiPtrField, artiU32Field
import pxgSlangArticulationBlocks; // PXG_AB_*, abStoreU8, abStoreSpatialVector

static const uint AD_BLOCK = 512u;                    // ACCUMULATE_DELTA / SB_ACCUMULATE_DELTA block
static const uint AD_GRID = 32u;                      // ACCUMULATE_DELTA / SB_ACCUMULATE_DELTA grid
static const uint WARP = 32u;
static const uint WARPS_PER_BLOCK = AD_BLOCK / WARP;  // numWarpsPerBlock (.cu:66) = 16
static const uint STATIC_NODE = 0xffffffffu;          // PX_INVALID_NODE
// 0x8fffffffffffffff (.cu:90): never a valid PxNodeIndex (mID 0xffffffff is the static
// body, but mLinkID 0x8fffffff is neither 0 nor a link id anyone hands out).
static const uint64_t RDA_SENTINEL = (uint64_t(0x8fffffffu) << 32) | uint64_t(0xffffffffu);
static const uint ARTI_STATE_HAS_IMPULSES = 2u;       // PxgArtiStateDirtyFlag::eHAS_IMPULSES (PxgArticulation.h:442)

// PxNodeIndex (PxNodeIndex.h:53): { PxU32 mID (low), PxU32 mLinkID (high) }.
uint rdaNodeIndex(uint64_t id) { return uint(id & uint64_t(0xffffffffu)); }
bool rdaIsArticulation(uint64_t id) { return (uint(id >> 32) & 1u) != 0u; }
uint rdaLinkId(uint64_t id) { return uint(id >> 32) >> 1; }

// PxMax on FLOAT is not the generic template (PxMath.h:75, `a < b ? b : a`): the
// float overload (PxMath.h:82) forwards to intrinsics::selectMax, which is
// `a > b ? a : b` (PxUnixMathIntrinsics.h:122).  The two differ for a NaN operand
// and for PxMax(+0, -0) -- the first mutation sweep of this module caught the
// generic order here, with the correct code as the surviving mutant.
float rdaMax(float a, float b) { return (a > b) ? a : b; }

// x + (+0.0f) in IEEE: only a negative zero changes.  See the header.
float rdaPlusZero(float x)
{
    const uint u = asuint(x);
    return asfloat(u == 0x80000000u ? 0u : u);
}

// float atomicAdd via compare-exchange on the raw bits (skinning.slang:132).
void rdaAtomicAddF32(Ptr<uint> words, uint index, float value)
{
    uint expected = words[index];
    for (;;)
    {
        const uint desired = asuint(asfloat(expected) + value);
        uint original;
        InterlockedCompareExchange(words[index], expected, desired, original);
        if (original == expected)
            break;
        expected = original;
    }
}

// ---------------------------------------------------------------------------
// Shared memory (the .cu's __shared__ arrays plus the shuffle scratch)
// ---------------------------------------------------------------------------
groupshared uint64_t gsRigidId[AD_BLOCK + 1];   // sRigidId (.cu:63/240/442/539/690)
groupshared float4 gsLin[AD_BLOCK];             // shuffle scratch, linear
groupshared float4 gsAng[AD_BLOCK];             // shuffle scratch, angular
groupshared float gsW[AD_BLOCK];                // shuffle scratch, Stage1's .w only
groupshared float4 gsLinWarp[WARP];             // sLinWarpAccumulator (.cu:68)
groupshared float4 gsAngWarp[WARP];             // sAngWarpAccumulator (.cu:69)
groupshared uint64_t gsWarpRigidId[WARP];       // sWarpRigidId (.cu:70)
groupshared float4 gsLinBlock;                  // sLinBlockAccumulator (.cu:71)
groupshared float4 gsAngBlock;                  // sAngBlockAccumulator (.cu:72)
groupshared uint64_t gsBlockRigidId;            // sBlockRigidId (.cu:73)
groupshared float4 gsBlockLin[AD_GRID];         // sBlockLinDeltaV (.cu:237)
groupshared float4 gsBlockAng[AD_GRID];         // sBlockAngDeltaV (.cu:238)
groupshared uint64_t gsBlockRid[AD_GRID];       // sBlockRigidId[] (.cu:239)
groupshared float gsBlockW[AD_GRID];            // sBlockLinDeltaVW (.cu:537)

// ===========================================================================
// 1. accumulateDeltaVRigidFirstLaunch (.cu:55-220)
//    (const PxU64* sortedRigidIds, const PxU32* numContacts, float4* deltaV,
//     float4* blockDeltaV, PxU64* blockRigidId) -> {8x5} = 40 B
// ===========================================================================
struct AccumFirstParams
{
    uint64_t sortedRigidIds; // @0  input
    uint64_t numContacts;    // @8  input
    uint64_t deltaV;         // @16 input/output
    uint64_t blockDeltaV;    // @24 output (32 linear + 32 angular float4)
    uint64_t blockRigidId;   // @32 output (32 PxU64) -> 40
};
[[vk::push_constant]]
ConstantBuffer<AccumFirstParams> gF;

[shader("compute")]
[numthreads(AD_BLOCK, 1, 1)]
void accumulateDeltaVRigidFirstLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const uint lane = tid & (WARP - 1u);      // threadIndexInWarp (.cu:78)
    const uint warpIndex = tid / WARP;        // (.cu:79)

    Ptr<uint64_t> sortedRigidIds = Ptr<uint64_t>(gF.sortedRigidIds);
    Ptr<float4> deltaV = Ptr<float4>(gF.deltaV);
    Ptr<float4> blockDeltaV = Ptr<float4>(gF.blockDeltaV);
    Ptr<uint64_t> blockRigidId = Ptr<uint64_t>(gF.blockRigidId);

    const uint tNumContacts = Ptr<uint>(gF.numContacts)[0];
    const uint nbBlocksRequired = (tNumContacts + AD_BLOCK - 1u) / AD_BLOCK;
    const uint gridSize = WorkgroupCount().x;                                  // gridDim.x (.cu:77)
    const uint nbIterationsPerBlock = (nbBlocksRequired + gridSize - 1u) / gridSize;

    // .cu:81-91: threads 0..3 zero the block accumulators, thread 0 the id.
    if (tid == 0)
    {
        gsLinBlock = float4(0.0f, 0.0f, 0.0f, 0.0f);
        gsAngBlock = float4(0.0f, 0.0f, 0.0f, 0.0f);
        gsBlockRigidId = RDA_SENTINEL;
    }
    GroupMemoryBarrierWithGroupSync();  // .cu:92

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        // .cu:96 -- block b owns the CONTIGUOUS chunk of nbIterationsPerBlock tiles.
        const uint workIndex = AD_BLOCK * (blockIdx * nbIterationsPerBlock + i) + tid;
        const bool valid = workIndex < tNumContacts;

        uint64_t rigidId = RDA_SENTINEL;
        float4 linDeltaV = float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 angDeltaV = float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (valid)
        {
            rigidId = sortedRigidIds[workIndex];
            gsRigidId[tid] = rigidId;
            linDeltaV = deltaV[workIndex];
            angDeltaV = deltaV[workIndex + tNumContacts];
        }
        gsLin[tid] = linDeltaV;
        gsAng[tid] = angDeltaV;
        GroupMemoryBarrierWithGroupSync();  // .cu:108

        // .cu:110-125: segmented inclusive scan over the warp.  shuffle(v, lane - r)
        // is read from the scratch; a lane adds it iff it is >= r, is a live
        // contact and shares the rigid id of lane - r (which is then live too, since
        // the contacts are contiguous -- .cu:117-118).
        for (uint r = 1; r < WARP; r <<= 1)
        {
            float4 linVal = float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 angVal = float4(0.0f, 0.0f, 0.0f, 0.0f);
            bool add = false;
            if (lane >= r)
            {
                linVal = gsLin[tid - r];
                angVal = gsAng[tid - r];
                if (valid)
                    add = (rigidId == gsRigidId[tid - r]);
            }
            GroupMemoryBarrierWithGroupSync();
            if (add)
            {
                linDeltaV += linVal;
                angDeltaV += angVal;
            }
            gsLin[tid] = linDeltaV;
            gsAng[tid] = angDeltaV;
            GroupMemoryBarrierWithGroupSync();
        }

        // .cu:127-132: lane 31 publishes the warp's trailing run.
        if (lane == WARP - 1u)
        {
            gsLinWarp[warpIndex] = linDeltaV;
            gsAngWarp[warpIndex] = angDeltaV;
            gsWarpRigidId[warpIndex] = rigidId;
        }

        // .cu:134-136: every thread snapshots the carry of the PREVIOUS tiles
        // before warp 0 updates it.
        const float4 prevLinBlockAccumulator = gsLinBlock;
        const float4 prevAngBlockAccumulator = gsAngBlock;
        const uint64_t prevBlockRigidId = gsBlockRigidId;
        GroupMemoryBarrierWithGroupSync();  // .cu:139

        // .cu:141-186: warp 0 scans the 16 warp totals (segmented by the warps'
        // trailing ids), lane 15 folds the tile's trailing run into the block
        // carry, and the scanned totals are written back over the warp arrays.
        float4 tLin = float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 tAng = float4(0.0f, 0.0f, 0.0f, 0.0f);
        uint64_t warpRigidId = RDA_SENTINEL;
        if (tid < WARPS_PER_BLOCK)
        {
            tLin = gsLinWarp[tid];
            tAng = gsAngWarp[tid];
            warpRigidId = gsWarpRigidId[tid];
        }
        gsLin[tid] = tLin;   // scratch reuse: the level-1 reads all sit before .cu:139
        gsAng[tid] = tAng;
        GroupMemoryBarrierWithGroupSync();
        for (uint r = 1; r < WARPS_PER_BLOCK; r <<= 1)
        {
            float4 linVal = float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 angVal = float4(0.0f, 0.0f, 0.0f, 0.0f);
            bool add = false;
            if (tid < WARPS_PER_BLOCK && tid >= r)
            {
                linVal = gsLin[tid - r];
                angVal = gsAng[tid - r];
                add = (warpRigidId == gsWarpRigidId[tid - r]);
            }
            GroupMemoryBarrierWithGroupSync();
            if (add)
            {
                tLin += linVal;
                tAng += angVal;
            }
            gsLin[tid] = tLin;
            gsAng[tid] = tAng;
            GroupMemoryBarrierWithGroupSync();
        }
        if (tid == WARPS_PER_BLOCK - 1u)
        {
            // .cu:170-182: a new trailing id resets the carry first.
            if (gsBlockRigidId != warpRigidId)
            {
                gsLinBlock = float4(0.0f, 0.0f, 0.0f, 0.0f);
                gsAngBlock = float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            gsLinBlock += tLin;
            gsAngBlock += tAng;
            gsBlockRigidId = warpRigidId;
        }
        if (tid < WARPS_PER_BLOCK)
        {
            gsLinWarp[tid] = tLin;   // .cu:184-185
            gsAngWarp[tid] = tAng;
        }
        GroupMemoryBarrierWithGroupSync();  // .cu:187

        // .cu:189-211: add the previous warp's (scanned) trailing run and the
        // previous tiles' carry when the ids match; write the scan back in place.
        if (valid)
        {
            float4 accumLin = float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 accumAng = float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (warpIndex > 0 && rigidId == gsWarpRigidId[warpIndex - 1])
            {
                accumLin = gsLinWarp[warpIndex - 1];
                accumAng = gsAngWarp[warpIndex - 1];
            }
            if (i != 0 && rigidId == prevBlockRigidId)
            {
                accumLin += prevLinBlockAccumulator;
                accumAng += prevAngBlockAccumulator;
            }
            deltaV[workIndex] = linDeltaV + accumLin;
            deltaV[workIndex + tNumContacts] = angDeltaV + accumAng;
        }
        // The .cu closes the tile on warp convergence; close it explicitly (header).
        GroupMemoryBarrierWithGroupSync();
    }

    // .cu:214-219: written for every block, sentinel + zeros when it had no tiles.
    if (tid == 0)
    {
        blockDeltaV[blockIdx] = gsLinBlock;
        blockDeltaV[blockIdx + AD_GRID] = gsAngBlock;   // the CONSTANT 32, not gridDim.x
        blockRigidId[blockIdx] = gsBlockRigidId;
    }
}

// ---------------------------------------------------------------------------
// Shared by Second / Stage1 / Stage2: the 32-wide segmented scan of the block
// carries (.cu:247-289 / 548-586 / 699-743).  In the .cu every warp of the block
// runs the shuffle loop and only tid < 32 writes back; it runs on tid < 32 here.
// On return gsBlockLin/gsBlockAng/gsBlockW hold the scanned carries and
// gsBlockRid the raw block ids (re-read from memory, .cu:286).
// ---------------------------------------------------------------------------
void rdaScanBlockCarries(uint tid, Ptr<float4> blockDeltaV, Ptr<uint64_t> blockRigidId, bool withVectors)
{
    float4 linBlockDeltaV = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 angBlockDeltaV = float4(0.0f, 0.0f, 0.0f, 0.0f);
    uint64_t tBlockRigidId = RDA_SENTINEL;
    if (tid < AD_GRID)
    {
        linBlockDeltaV = blockDeltaV[tid];
        if (withVectors)
            angBlockDeltaV = blockDeltaV[tid + AD_GRID];
        tBlockRigidId = blockRigidId[tid];
        gsBlockLin[tid] = linBlockDeltaV;
        gsBlockAng[tid] = angBlockDeltaV;
        gsBlockW[tid] = linBlockDeltaV.w;
        gsBlockRid[tid] = tBlockRigidId;
    }
    GroupMemoryBarrierWithGroupSync();  // .cu:262

    float4 tLin = linBlockDeltaV;
    float4 tAng = angBlockDeltaV;
    gsLin[tid] = tLin;
    gsAng[tid] = tAng;
    GroupMemoryBarrierWithGroupSync();
    for (uint r = 1; r < AD_GRID; r <<= 1)
    {
        float4 linVal = float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 angVal = float4(0.0f, 0.0f, 0.0f, 0.0f);
        bool add = false;
        if (tid < AD_GRID && tid >= r)
        {
            linVal = gsLin[tid - r];
            angVal = gsAng[tid - r];
            add = (tBlockRigidId == gsBlockRid[tid - r]);
        }
        GroupMemoryBarrierWithGroupSync();
        if (add)
        {
            tLin += linVal;
            tAng += angVal;
        }
        gsLin[tid] = tLin;
        gsAng[tid] = tAng;
        GroupMemoryBarrierWithGroupSync();
    }

    // .cu:280-289
    if (tid < AD_GRID)
    {
        gsBlockLin[tid] = tLin;
        gsBlockAng[tid] = tAng;
        gsBlockW[tid] = tLin.w;
        gsBlockRid[tid] = blockRigidId[tid];
    }
    GroupMemoryBarrierWithGroupSync();
}

// Shared by Second / clear / Stage1 / Stage2 (.cu:306-324 etc.): load this tile's
// ids so that gsRigidId[tid] is the id of the NEXT contact (sentinel past the end),
// which makes `rigidId != gsRigidId[tid]` "last contact of its run".  Returns the
// lane's id (sentinel when the lane is past the end).
// Both reads are CLAMPED and the guard applied to the VALUE, not the address
// (countAt, npRigidSumBatches.slang:143, the P9.22 shape): llvmpipe turns the `if`
// into a mask, so a guarded-but-unclamped index still forms and dereferences the
// out-of-range address.  Here the index is a function PARAMETER, so nothing inside
// this function bounds it even in principle -- the static checker flags exactly that.
// The clamps are semantically free: `workIndex < tNumContacts` on the first, and on
// the second the enclosing branch has already excluded workIndex == tNumContacts - 1.
uint64_t rdaLoadRunEnds(uint tid, uint workIndex, uint tNumContacts, Ptr<uint64_t> sortedRigidIds)
{
    uint64_t rigidId = RDA_SENTINEL;
    if (tNumContacts != 0u)
    {
        const uint last = tNumContacts - 1u;
        const bool inRange = workIndex < tNumContacts;
        const uint64_t raw = sortedRigidIds[inRange ? workIndex : 0u];
        if (inRange)
        {
            rigidId = raw;
            if (tid > 0)
                gsRigidId[tid - 1] = rigidId;

            if (workIndex == last)
                gsRigidId[tid] = RDA_SENTINEL;
            else if (tid == AD_BLOCK - 1u)
                gsRigidId[tid] = sortedRigidIds[min(workIndex + 1u, last)];
        }
    }
    GroupMemoryBarrierWithGroupSync();
    return rigidId;
}

// ===========================================================================
// 2. accumulateDeltaVRigidSecondLaunch (.cu:224-429)
//    (const PxU64* sortedRigidIds, const PxU32* numContacts, const float4* deltaV,
//     const float4* blockDeltaV, const PxU64* blockRigidId, PxgPrePrepDesc*,
//     PxgSolverCoreDesc*, PxgArticulationCoreDesc*,
//     PxgSolverSharedDesc<IterativeSolveData>*, const bool isTGS)
//    -> {8x9, 1} = 80 B (isTGS byte @72)
// ===========================================================================
struct AccumSecondParams
{
    uint64_t sortedRigidIds; // @0
    uint64_t numContacts;    // @8
    uint64_t deltaV;         // @16
    uint64_t blockDeltaV;    // @24
    uint64_t blockRigidId;   // @32
    uint64_t prePrepDesc;    // @40
    uint64_t solverCoreDesc; // @48
    uint64_t artiCoreDesc;   // @56
    uint64_t sharedDesc;     // @64
    uint isTGSWord;          // @72 (bool byte0) -> 80
};
[[vk::push_constant]]
ConstantBuffer<AccumSecondParams> gS;

[shader("compute")]
[numthreads(AD_BLOCK, 1, 1)]
void accumulateDeltaVRigidSecondLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;

    Ptr<uint64_t> sortedRigidIds = Ptr<uint64_t>(gS.sortedRigidIds);
    Ptr<float4> deltaV = Ptr<float4>(gS.deltaV);

    const uint tNumContacts = Ptr<uint>(gS.numContacts)[0];
    const uint nbBlocksRequired = (tNumContacts + AD_BLOCK - 1u) / AD_BLOCK;
    const uint gridSize = WorkgroupCount().x;                                  // gridDim.x (.cu:244)
    const uint nbIterationsPerBlock = (nbBlocksRequired + gridSize - 1u) / gridSize;

    rdaScanBlockCarries(tid, Ptr<float4>(gS.blockDeltaV), Ptr<uint64_t>(gS.blockRigidId), true);

    // .cu:291-298: hoisted descriptor state, exactly as the .cu (the engine always
    // hands real descriptors here; the pointers they hold are dereferenced only on
    // their own branch).
    Ptr<PxgSolverCoreDescGpu> solverCoreDesc = Ptr<PxgSolverCoreDescGpu>(gS.solverCoreDesc);
    Ptr<PxgSolverSharedDescPGSGpu> sharedDesc = Ptr<PxgSolverSharedDescPGSGpu>(gS.sharedDesc);
    Ptr<PxgPrePrepDescGpu> prePrepDesc = Ptr<PxgPrePrepDescGpu>(gS.prePrepDesc);
    const uint accOffset = solverCoreDesc[0].accumulatedBodyDeltaVOffset;
    Ptr<float4> solverBodyVelPool = Ptr<float4>(sharedDesc[0].iter_solverBodyVelPool);
    const uint numSolverBodies = solverCoreDesc[0].numSolverBodies;
    Ptr<uint> solverBodyIndices = Ptr<uint>(prePrepDesc[0].solverBodyIndices);
    const uint64_t artiBlocks = artiPtrField(gS.artiCoreDesc, PXG_ARTI_CORE_ARTICULATION_BLOCKS);
    const uint64_t artiLinkBlocks = artiPtrField(gS.artiCoreDesc, PXG_ARTI_CORE_LINK_BLOCKS);
    const uint maxLinks = artiU32Field(gS.artiCoreDesc, PXG_ARTI_CORE_MAX_LINKS);
    const bool isTGS = (gS.isTGSWord & 0xffu) != 0u;

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        GroupMemoryBarrierWithGroupSync();  // .cu:302
        const uint workIndex = AD_BLOCK * (blockIdx * nbIterationsPerBlock + i) + tid;
        const uint64_t rigidId = rdaLoadRunEnds(tid, workIndex, tNumContacts, sortedRigidIds);

        if (workIndex < tNumContacts && rigidId != gsRigidId[tid])
        {
            float4 linVel = deltaV[workIndex];
            float4 angVel = deltaV[workIndex + tNumContacts];

            // .cu:336-342: a run that started in the previous block carries in.
            const uint64_t preBlockRigidId = blockIdx > 0 ? gsBlockRid[blockIdx - 1] : RDA_SENTINEL;
            if (rigidId == preBlockRigidId)
            {
                linVel += gsBlockLin[blockIdx - 1];
                angVel += gsBlockAng[blockIdx - 1];
            }

            const uint nodeIndex = rdaNodeIndex(rigidId);
            if (nodeIndex != STATIC_NODE)
            {
                const uint solverBodyIndex = solverBodyIndices[nodeIndex];
                if (rdaIsArticulation(rigidId))
                {
                    // .cu:354-385: solverBodyIndex is the articulation's global thread
                    // index in the block format.
                    const uint blockIndex = solverBodyIndex / WARP;
                    const uint artiIndexInBlock = solverBodyIndex % WARP;
                    const uint64_t artiBlock = artiBlocks + uint64_t(blockIndex) * uint64_t(PXG_AB_BLOCK_SIZE);
                    abStoreU8(artiBlock, PXG_AB_mStateDirty, artiIndexInBlock, ARTI_STATE_HAS_IMPULSES);

                    const uint linkID = rdaLinkId(rigidId);
                    const uint64_t link = artiLinkBlocks
                                        + uint64_t(blockIndex * maxLinks + linkID) * uint64_t(PXG_AB_LINK_SIZE);

                    const float denom = rdaMax(1.0f, linVel.w);
                    const float ratio = 1.0f / denom;
                    // linVel and angVel accumulate an IMPULSE here; stored negated.
                    const float3 top = float3(linVel.x * ratio, linVel.y * ratio, linVel.z * ratio);
                    const float3 bottom = float3(angVel.x * ratio, angVel.y * ratio, angVel.z * ratio);
                    abStoreSpatialVector(link, PXG_AB_LINK_mScratchImpulse, artiIndexInBlock, -top, -bottom);
                }
                else
                {
                    float4 linearVelocity = solverBodyVelPool[accOffset + solverBodyIndex];
                    float4 angularVelocity = solverBodyVelPool[accOffset + solverBodyIndex + numSolverBodies];

                    const float denom = rdaMax(1.0f, linVel.w);
                    const float ratio = 1.0f / denom;
                    // .cu:393 linVel.w = 0.f -- see rdaPlusZero for the .w lane below.

                    if (isTGS)
                    {
                        // .cu:395-404: the TGS pool packs (linVel.xyz, angVel.x) /
                        // (angVel.yz, ...) -- the rest is the delta position buffer.
                        linearVelocity.x += linVel.x * ratio;
                        linearVelocity.y += linVel.y * ratio;
                        linearVelocity.z += linVel.z * ratio;
                        linearVelocity.w += angVel.x * ratio;
                        angularVelocity.x += angVel.y * ratio;
                        angularVelocity.y += angVel.z * ratio;
                    }
                    else
                    {
                        // .cu:414-415: linearVelocity += linVel * ratio (with linVel.w == 0,
                        // so .w gains +0.0f exactly); angularVelocity += angVel * ratio.
                        linearVelocity.x += linVel.x * ratio;
                        linearVelocity.y += linVel.y * ratio;
                        linearVelocity.z += linVel.z * ratio;
                        linearVelocity.w = rdaPlusZero(linearVelocity.w);
                        angularVelocity.x += angVel.x * ratio;
                        angularVelocity.y += angVel.y * ratio;
                        angularVelocity.z += angVel.z * ratio;
                        angularVelocity.w += angVel.w * ratio;
                    }

                    solverBodyVelPool[accOffset + solverBodyIndex] = linearVelocity;
                    solverBodyVelPool[accOffset + solverBodyIndex + numSolverBodies] = angularVelocity;
                }
            }
        }
    }
}

// ===========================================================================
// 3. clearDeltaVRigidSecondLaunchMulti (.cu:433-516)
//    (PxU64* sortedRigidIds, PxU32* numContacts, PxgPrePrepDesc*, PxgSolverCoreDesc*,
//     PxgArticulationCoreDesc*, PxReal* tempDenom) -> {8x6} = 48 B
// ===========================================================================
struct ClearMultiParams
{
    uint64_t sortedRigidIds; // @0
    uint64_t numContacts;    // @8
    uint64_t prePrepDesc;    // @16
    uint64_t solverCoreDesc; // @24 (unread by the .cu)
    uint64_t artiCoreDesc;   // @32
    uint64_t tempDenom;      // @40 -> 48
};
[[vk::push_constant]]
ConstantBuffer<ClearMultiParams> gC;

[shader("compute")]
[numthreads(AD_BLOCK, 1, 1)]
void clearDeltaVRigidSecondLaunchMulti(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;

    Ptr<uint64_t> sortedRigidIds = Ptr<uint64_t>(gC.sortedRigidIds);
    Ptr<float> tempDenom = Ptr<float>(gC.tempDenom);

    const uint tNumContacts = Ptr<uint>(gC.numContacts)[0];
    // .cu:447-448: the CONSTANT grid here, not gridDim.x.
    const uint totalBlockRequired = (tNumContacts + AD_BLOCK - 1u) / AD_BLOCK;
    const uint numIterationPerBlock = (totalBlockRequired + AD_GRID - 1u) / AD_GRID;

    Ptr<uint> solverBodyIndices = Ptr<uint>(Ptr<PxgPrePrepDescGpu>(gC.prePrepDesc)[0].solverBodyIndices);
    const uint64_t artiLinkBlocks = artiPtrField(gC.artiCoreDesc, PXG_ARTI_CORE_LINK_BLOCKS);
    const uint maxLinks = artiU32Field(gC.artiCoreDesc, PXG_ARTI_CORE_MAX_LINKS);

    for (uint i = 0; i < numIterationPerBlock; ++i)
    {
        const uint workIndex = i * AD_BLOCK + tid + numIterationPerBlock * blockIdx * AD_BLOCK;
        const uint64_t rigidId = rdaLoadRunEnds(tid, workIndex, tNumContacts, sortedRigidIds);

        if (workIndex < tNumContacts && rigidId != gsRigidId[tid])
        {
            const uint nodeIndex = rdaNodeIndex(rigidId);
            if (nodeIndex != STATIC_NODE)
            {
                const uint solverBodyIndex = solverBodyIndices[nodeIndex];
                if (rdaIsArticulation(rigidId))
                {
                    const uint blockIndex = solverBodyIndex / WARP;
                    const uint artiIndexInBlock = solverBodyIndex % WARP;
                    const uint linkID = rdaLinkId(rigidId);
                    const uint64_t link = artiLinkBlocks
                                        + uint64_t(blockIndex * maxLinks + linkID) * uint64_t(PXG_AB_LINK_SIZE);
                    abStoreF32(link, PXG_AB_LINK_mDeltaScale, artiIndexInBlock, 0.0f);
                }
                else
                {
                    tempDenom[solverBodyIndex] = 0.0f;
                }
            }
        }
        // The .cu's loop body ends without a barrier: the next tile's gsRigidId
        // writes vs. this tile's reads rely on convergence.  Close explicitly.
        GroupMemoryBarrierWithGroupSync();
    }
}

// ===========================================================================
// 4./5. accumulateDeltaVRigidSecondLaunchMultiStage1 / Stage2 (.cu:521-668 / 671-874)
//    (PxU64* sortedRigidIds, PxU32* numContacts, float4* deltaV, float4* blockDeltaV,
//     PxU64* blockRigidId, PxgPrePrepDesc*, PxgSolverCoreDesc*, PxgArticulationCoreDesc*,
//     PxgSolverSharedDesc<IterativeSolveData>*, PxReal* tempDenom,
//     const bool useLocalRelax, const float globalRelaxationCoefficient, bool isTGS)
//    -> {8x10, 1, 4, 1} = 96 B (useLocalRelax byte @80, coefficient @84, isTGS byte @88)
// ===========================================================================
struct AccumMultiParams
{
    uint64_t sortedRigidIds;           // @0
    uint64_t numContacts;              // @8
    uint64_t deltaV;                   // @16
    uint64_t blockDeltaV;              // @24
    uint64_t blockRigidId;             // @32
    uint64_t prePrepDesc;              // @40
    uint64_t solverCoreDesc;           // @48
    uint64_t artiCoreDesc;             // @56
    uint64_t sharedDesc;               // @64
    uint64_t tempDenom;                // @72
    uint useLocalRelaxWord;            // @80 (bool byte0)
    float globalRelaxationCoefficient; // @84
    uint isTGSWord;                    // @88 (bool byte0) -> 96
};
[[vk::push_constant]]
ConstantBuffer<AccumMultiParams> gM;

[shader("compute")]
[numthreads(AD_BLOCK, 1, 1)]
void accumulateDeltaVRigidSecondLaunchMultiStage1(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;

    Ptr<uint64_t> sortedRigidIds = Ptr<uint64_t>(gM.sortedRigidIds);
    Ptr<float4> deltaV = Ptr<float4>(gM.deltaV);
    Ptr<uint> tempDenomWords = Ptr<uint>(gM.tempDenom);

    const uint tNumContacts = Ptr<uint>(gM.numContacts)[0];
    const uint totalBlockRequired = (tNumContacts + AD_BLOCK - 1u) / AD_BLOCK;

    // .cu:548-586: the .w-only carry scan (blockDeltaV[idx].w; the angular half is
    // not read here).
    rdaScanBlockCarries(tid, Ptr<float4>(gM.blockDeltaV), Ptr<uint64_t>(gM.blockRigidId), false);

    const uint numIterationPerBlock = (totalBlockRequired + AD_GRID - 1u) / AD_GRID;   // .cu:584

    Ptr<uint> solverBodyIndices = Ptr<uint>(Ptr<PxgPrePrepDescGpu>(gM.prePrepDesc)[0].solverBodyIndices);
    const uint64_t artiLinkBlocks = artiPtrField(gM.artiCoreDesc, PXG_ARTI_CORE_LINK_BLOCKS);
    const uint maxLinks = artiU32Field(gM.artiCoreDesc, PXG_ARTI_CORE_MAX_LINKS);
    const bool useLocalRelax = (gM.useLocalRelaxWord & 0xffu) != 0u;
    const float globalRelaxationCoefficient = gM.globalRelaxationCoefficient;

    for (uint i = 0; i < numIterationPerBlock; ++i)
    {
        const uint workIndex = i * AD_BLOCK + tid + numIterationPerBlock * blockIdx * AD_BLOCK;
        const uint64_t rigidId = rdaLoadRunEnds(tid, workIndex, tNumContacts, sortedRigidIds);

        if (workIndex < tNumContacts && rigidId != gsRigidId[tid])
        {
            float linVelW = deltaV[workIndex].w;

            const uint64_t preBlockRigidId = blockIdx > 0 ? gsBlockRid[blockIdx - 1] : RDA_SENTINEL;
            if (rigidId == preBlockRigidId)
                linVelW += gsBlockW[blockIdx - 1];

            const uint nodeIndex = rdaNodeIndex(rigidId);
            if (nodeIndex != STATIC_NODE)
            {
                const uint solverBodyIndex = solverBodyIndices[nodeIndex];

                float denom = globalRelaxationCoefficient;
                if (useLocalRelax)
                    denom = rdaMax(denom, linVelW);

                if (rdaIsArticulation(rigidId))
                {
                    const uint blockIndex = solverBodyIndex / WARP;
                    const uint artiIndexInBlock = solverBodyIndex % WARP;
                    const uint linkID = rdaLinkId(rigidId);
                    const uint64_t link = artiLinkBlocks
                                        + uint64_t(blockIndex * maxLinks + linkID) * uint64_t(PXG_AB_LINK_SIZE);
                    rdaAtomicAddF32(Ptr<uint>(link + uint64_t(PXG_AB_LINK_mDeltaScale)), artiIndexInBlock, denom);
                }
                else
                {
                    rdaAtomicAddF32(tempDenomWords, solverBodyIndex, denom);
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();   // close the tile (see clear)
    }
}

[shader("compute")]
[numthreads(AD_BLOCK, 1, 1)]
void accumulateDeltaVRigidSecondLaunchMultiStage2(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;

    Ptr<uint64_t> sortedRigidIds = Ptr<uint64_t>(gM.sortedRigidIds);
    Ptr<float4> deltaV = Ptr<float4>(gM.deltaV);
    Ptr<float> tempDenom = Ptr<float>(gM.tempDenom);

    const uint tNumContacts = Ptr<uint>(gM.numContacts)[0];
    const uint totalBlockRequired = (tNumContacts + AD_BLOCK - 1u) / AD_BLOCK;

    rdaScanBlockCarries(tid, Ptr<float4>(gM.blockDeltaV), Ptr<uint64_t>(gM.blockRigidId), true);

    const uint numIterationPerBlock = (totalBlockRequired + AD_GRID - 1u) / AD_GRID;   // .cu:741

    // .cu:745-752
    Ptr<PxgSolverCoreDescGpu> solverCoreDesc = Ptr<PxgSolverCoreDescGpu>(gM.solverCoreDesc);
    Ptr<PxgSolverSharedDescPGSGpu> sharedDesc = Ptr<PxgSolverSharedDescPGSGpu>(gM.sharedDesc);
    const uint accOffset = solverCoreDesc[0].accumulatedBodyDeltaVOffset;
    const uint numSolverBodies = solverCoreDesc[0].numSolverBodies;
    // The pool as 32-bit words: float4 index k, component c -> word 4k + c.
    Ptr<uint> velPoolWords = Ptr<uint>(sharedDesc[0].iter_solverBodyVelPool);
    Ptr<uint> solverBodyIndices = Ptr<uint>(Ptr<PxgPrePrepDescGpu>(gM.prePrepDesc)[0].solverBodyIndices);
    const uint64_t artiBlocks = artiPtrField(gM.artiCoreDesc, PXG_ARTI_CORE_ARTICULATION_BLOCKS);
    const uint64_t artiLinkBlocks = artiPtrField(gM.artiCoreDesc, PXG_ARTI_CORE_LINK_BLOCKS);
    const uint maxLinks = artiU32Field(gM.artiCoreDesc, PXG_ARTI_CORE_MAX_LINKS);
    const bool isTGS = (gM.isTGSWord & 0xffu) != 0u;

    for (uint i = 0; i < numIterationPerBlock; ++i)
    {
        const uint workIndex = i * AD_BLOCK + tid + numIterationPerBlock * blockIdx * AD_BLOCK;
        const uint64_t rigidId = rdaLoadRunEnds(tid, workIndex, tNumContacts, sortedRigidIds);

        if (workIndex < tNumContacts && rigidId != gsRigidId[tid])
        {
            float4 linVel = deltaV[workIndex];
            float4 angVel = deltaV[workIndex + tNumContacts];

            const uint64_t preBlockRigidId = blockIdx > 0 ? gsBlockRid[blockIdx - 1] : RDA_SENTINEL;
            if (rigidId == preBlockRigidId)
            {
                linVel += gsBlockLin[blockIdx - 1];
                angVel += gsBlockAng[blockIdx - 1];
            }

            const uint nodeIndex = rdaNodeIndex(rigidId);
            if (nodeIndex != STATIC_NODE)
            {
                const uint solverBodyIndex = solverBodyIndices[nodeIndex];
                if (rdaIsArticulation(rigidId))
                {
                    const uint blockIndex = solverBodyIndex / WARP;
                    const uint artiIndexInBlock = solverBodyIndex % WARP;
                    const uint64_t artiBlock = artiBlocks + uint64_t(blockIndex) * uint64_t(PXG_AB_BLOCK_SIZE);
                    abStoreU8(artiBlock, PXG_AB_mStateDirty, artiIndexInBlock, ARTI_STATE_HAS_IMPULSES);

                    const uint linkID = rdaLinkId(rigidId);
                    const uint64_t link = artiLinkBlocks
                                        + uint64_t(blockIndex * maxLinks + linkID) * uint64_t(PXG_AB_LINK_SIZE);

                    // .cu:823-824: the denominator Stage1 summed into mDeltaScale.
                    const float denom = abF32(link, PXG_AB_LINK_mDeltaScale, artiIndexInBlock);
                    const float ratio = 1.0f / denom;
                    const float3 top = float3(linVel.x * ratio, linVel.y * ratio, linVel.z * ratio);
                    const float3 bottom = float3(angVel.x * ratio, angVel.y * ratio, angVel.z * ratio);

                    // atomicAddSpatialVector(mScratchImpulse, -impulse, lane): the block
                    // stores (top.xyz, bottom.x) as a float4[32] and (bottom.yz) as a
                    // float2[32]; six word atomics.
                    Ptr<uint> topWords = Ptr<uint>(link + uint64_t(PXG_AB_LINK_mScratchImpulse + PXG_AB_SVB_TOPXYZ_BX));
                    Ptr<uint> byzWords = Ptr<uint>(link + uint64_t(PXG_AB_LINK_mScratchImpulse + PXG_AB_SVB_BYZ));
                    rdaAtomicAddF32(topWords, artiIndexInBlock * 4u + 0u, -top.x);
                    rdaAtomicAddF32(topWords, artiIndexInBlock * 4u + 1u, -top.y);
                    rdaAtomicAddF32(topWords, artiIndexInBlock * 4u + 2u, -top.z);
                    rdaAtomicAddF32(topWords, artiIndexInBlock * 4u + 3u, -bottom.x);
                    rdaAtomicAddF32(byzWords, artiIndexInBlock * 2u + 0u, -bottom.y);
                    rdaAtomicAddF32(byzWords, artiIndexInBlock * 2u + 1u, -bottom.z);
                }
                else
                {
                    const float denom = tempDenom[solverBodyIndex];   // .cu:841
                    const float ratio = 1.0f / denom;

                    const uint linWord = (accOffset + solverBodyIndex) * 4u;
                    const uint angWord = (accOffset + solverBodyIndex + numSolverBodies) * 4u;
                    if (isTGS)
                    {
                        // .cu:845-854
                        rdaAtomicAddF32(velPoolWords, linWord + 0u, linVel.x * ratio);
                        rdaAtomicAddF32(velPoolWords, linWord + 1u, linVel.y * ratio);
                        rdaAtomicAddF32(velPoolWords, linWord + 2u, linVel.z * ratio);
                        rdaAtomicAddF32(velPoolWords, linWord + 3u, angVel.x * ratio);
                        rdaAtomicAddF32(velPoolWords, angWord + 0u, angVel.y * ratio);
                        rdaAtomicAddF32(velPoolWords, angWord + 1u, angVel.z * ratio);
                    }
                    else
                    {
                        // .cu:856-865
                        rdaAtomicAddF32(velPoolWords, linWord + 0u, linVel.x * ratio);
                        rdaAtomicAddF32(velPoolWords, linWord + 1u, linVel.y * ratio);
                        rdaAtomicAddF32(velPoolWords, linWord + 2u, linVel.z * ratio);
                        rdaAtomicAddF32(velPoolWords, angWord + 0u, angVel.x * ratio);
                        rdaAtomicAddF32(velPoolWords, angWord + 1u, angVel.y * ratio);
                        rdaAtomicAddF32(velPoolWords, angWord + 2u, angVel.z * ratio);
                    }
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();   // close the tile (see clear)
    }
}
