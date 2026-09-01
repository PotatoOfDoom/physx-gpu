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

// frozenHistogram.slang -- P9.42: the frozen/unfrozen shape compaction, three
// kernels of gpusimulationcontroller/src/CUDA/updateTransformAndBoundArray.cu:
//
//   computeFrozenAndUnfrozenHistogramLaunch  (.cu:245-341)  pass 1
//   outputFrozenAndUnfrozenHistogram         (.cu:343-403)  pass 2
//   createFrozenAndUnfrozenArray             (.cu:405-428)  pass 3
//
// WHAT THE CHAIN DOES.  updateTransformCacheAndBoundArrayLaunch (updateBounds.slang)
// marks every shape whose body froze or unfroze this frame with a 1 in
// scDesc->mFrozen[i] / mUnfrozen[i] (both memset to 0 each frame,
// PxgSimulationCore.cpp:2166-2167).  Pass 1 turns both arrays IN PLACE into
// block-local exclusive prefix counts and writes each block's total into the
// first 32 entries of mFrozenBlockAndRes / mUnfrozenBlockAndRes.  Pass 2 scans the
// 32 block totals, adds each block's base back into its partition (the prefixes
// are now device-wide) and writes the two grand totals INTO THE DESCRIPTOR
// (mTotalFrozenShapes / mTotalUnfrozenShapes, bytes 144 / 148), which the host
// copies back whole (PxgSimulationCore.cpp:2224) and reads in syncDmaback
// (:2306-2307).  Pass 3 binary-searches the prefixes to write the compact list of
// frozen / unfrozen shape indices over the SAME BlockAndRes buffers -- the block
// totals have been consumed by then -- and the host hands those lists to the
// scene-query tree update (ScScene.cpp:504-524).
//
// Launched only when !enableDirectGPUAPI (PxgSimulationCore.cpp:2693-2707), every
// frame, on every rigid scene.  P9.40's guard found all three silently skipped in
// all 33 scenes (x25..x300 per run).
//
// ENGINE ABI: one device pointer ({8} blob) for all three -- the same
// PxgSimulationCoreDesc the mapped siblings in updateBounds.slang take.  Grids:
// pass 1 and 2 are EXACTLY 32 blocks of 256 (PxgSimulationCoreKernelIndices.h:78-79,
// "this has to be 32"), pass 3 is 64 blocks of 256 and grid-strides.
//
// PORTABILITY.  Every quantity here is a PxU32 sum, so any evaluation order is
// bit-exact and the CUDA warp shuffles can be replaced freely:
//   * the per-lane `popc(ballot(v) & threadMask)` becomes a groupshared 0/1 array
//     and a 31-step summation below the lane;
//   * the 8-wide (pass 1) and 32-wide (pass 2) `warpScan - value` exclusive scans,
//     which the .cu runs in warp 0 only, are serialized in thread 0 so that no
//     barrier sits in divergent control flow (the P2.1 emulation contract);
//   * the .cu's block-0-lane-31 write of the totals (`globalThreadIndex == 31`) is
//     thread 0 of block 0 here -- same single writer, same values.
// The partition formula (contiguous per-block chunks of nbIterationsPerBlock * 256,
// NOT grid-stride) is transcribed verbatim in both passes; pass 2 relies on pass 1
// having used exactly it.  Model: broadphaseEndPtHistogram.slang /
// broadphaseRegionHistogram.slang, the ports of the structurally identical
// broadphase.cu histograms.

import pxgSlangCommon;
import pxgSlangSimCoreDesc;

static const uint BLOCK = 256;                    // PxgSimulationCoreKernelBlockDim::*_FROZEN_UNFROZEN_*
static const uint NUM_BLOCKS = 32;                // PxgSimulationCoreKernelGridDim::COMPUTE/OUTPUT_FROZEN_UNFROZEN_HISTOGRAM
static const uint WARP = 32;
static const uint WARPS_PER_BLOCK = BLOCK / WARP; // 8 (WARP_PERBLOCK_SIZE, .cu:247)

// ENGINE launch ABI: the CUDA parameter list is one device pointer.
struct SimCoreAbiParams
{
    uint64_t scDesc; // PxgSimulationCoreDesc* (non-const for pass 2: it writes the totals)
};
[[vk::push_constant]]
ConstantBuffer<SimCoreAbiParams> gParams;

groupshared uint gsLaneFrozen[BLOCK];             // 0/1 per lane: the ballot's bits
groupshared uint gsLaneUnfrozen[BLOCK];
groupshared uint gsWarpFrozen[WARPS_PER_BLOCK];   // sFrozenWarpAccumulator (.cu:252)
groupshared uint gsWarpUnfrozen[WARPS_PER_BLOCK]; // sUnFrozenWarpAccumulator (.cu:253)
groupshared uint gsBlockFrozen;                   // sFrozenBlockAccumulator (.cu:255)
groupshared uint gsBlockUnfrozen;                 // sUnfrozenBlockAccumulator (.cu:256)
groupshared uint gsBlockPrefixFrozen[NUM_BLOCKS];   // sFrozenBlockAccum (.cu:348)
groupshared uint gsBlockPrefixUnfrozen[NUM_BLOCKS]; // sUnfrozenBlockAccum (.cu:349)

// ---------------------------------------------------------------------------
// pass 1 (.cu:245-341): block-local exclusive prefixes in place + 32 block totals
// engine grid (32, 1, 1) x 256
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void computeFrozenAndUnfrozenHistogramLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;                 // idx (.cu:271)
    const uint blockIdx = gid.x;
    const uint lane = tid & (WARP - 1);      // threadIndexInWarp (.cu:269)
    const uint warpIndex = tid / WARP;       // (.cu:270)
    const uint laneBase = tid & ~(WARP - 1);

    Ptr<PxgSimulationCoreDescGpu> desc = Ptr<PxgSimulationCoreDescGpu>(gParams.scDesc);
    Ptr<uint> gFrozen = Ptr<uint>(desc[0].mFrozen);
    Ptr<uint> gUnfrozen = Ptr<uint>(desc[0].mUnfrozen);
    Ptr<uint> gFrozenBlock = Ptr<uint>(desc[0].mFrozenBlockAndRes);
    Ptr<uint> gUnfrozenBlock = Ptr<uint>(desc[0].mUnfrozenBlockAndRes);
    const uint nbTotalShapes = desc[0].mNbTotalShapes;

    // .cu:265-267 with blockDim.x = 256 and gridDim.x = 32, the engine's launch.
    // Zero shapes -> zero iterations: the data pointers may then be null (the host
    // allocates 0 bytes, PxgSimulationCore.cpp:2144-2145) and are never touched.
    const uint nbBlocksRequired = (nbTotalShapes + BLOCK - 1) / BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + NUM_BLOCKS - 1) / NUM_BLOCKS;

    if (tid == 0)
    {
        gsBlockFrozen = 0;
        gsBlockUnfrozen = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint it = 0; it < nbIterationsPerBlock; ++it)
    {
        // .cu:283 -- block b owns the CONTIGUOUS chunk [b*nbIter*256, (b+1)*nbIter*256).
        const uint workIndex = it * BLOCK + tid + nbIterationsPerBlock * blockIdx * BLOCK;

        // .cu:286-291: loads strictly under the guard, never hoisted.
        uint frozen = 0, unfrozen = 0;
        if (workIndex < nbTotalShapes)
        {
            frozen = gFrozen[workIndex];
            unfrozen = gUnfrozen[workIndex];
        }

        // .cu:293-296: popc(__ballot_sync(FULL_MASK, v) & ((1 << lane) - 1)) -- the
        // number of lanes BELOW this one whose value is nonzero.  The ballot tests
        // `v != 0`; the value itself is added raw at :300.  Both kept as written.
        gsLaneFrozen[tid] = frozen != 0u ? 1u : 0u;
        gsLaneUnfrozen[tid] = unfrozen != 0u ? 1u : 0u;
        GroupMemoryBarrierWithGroupSync();
        uint frozenAccumVal = 0, unfrozenAccumVal = 0;
        for (uint i = 0; i < lane; ++i)
        {
            frozenAccumVal += gsLaneFrozen[laneBase + i];
            unfrozenAccumVal += gsLaneUnfrozen[laneBase + i];
        }

        // .cu:298-302: lane 31 stores the warp's INCLUSIVE total (exclusive count + own
        // RAW value).
        if (lane == WARP - 1)
        {
            gsWarpFrozen[warpIndex] = frozenAccumVal + frozen;
            gsWarpUnfrozen[warpIndex] = unfrozenAccumVal + unfrozen;
        }

        // .cu:304-305: read by EVERY thread BEFORE the barrier -- the running total
        // of the PREVIOUS tiles of this block, not including this one.
        const uint prevFrozenBlockAccumulator = gsBlockFrozen;
        const uint prevUnfrozenBlockAccumulator = gsBlockUnfrozen;
        GroupMemoryBarrierWithGroupSync();

        // .cu:309-324: warp 0 lanes 0..7 rewrite the 8 warp totals as EXCLUSIVE
        // prefixes (warpScan<Add,PxU32,3> - value) and lane 7 adds the tile's inclusive
        // total to the block accumulator.  Serialized in thread 0: integer sums are
        // bit-exact in any order, and the barrier stays workgroup-uniform.
        if (tid == 0)
        {
            uint accFrozen = 0, accUnfrozen = 0;
            for (uint w = 0; w < WARPS_PER_BLOCK; ++w)
            {
                const uint vF = gsWarpFrozen[w];
                gsWarpFrozen[w] = accFrozen;
                accFrozen += vF;
                const uint vU = gsWarpUnfrozen[w];
                gsWarpUnfrozen[w] = accUnfrozen;
                accUnfrozen += vU;
            }
            gsBlockFrozen += accFrozen;      // .cu:321 (frozenOutput + frozenValue at lane 7)
            gsBlockUnfrozen += accUnfrozen;  // .cu:322
        }
        GroupMemoryBarrierWithGroupSync();

        // .cu:328-333: in-place overwrite of the 0/1 flag by the thread that read it.
        if (workIndex < nbTotalShapes)
        {
            gFrozen[workIndex] = frozenAccumVal + prevFrozenBlockAccumulator + gsWarpFrozen[warpIndex];
            gUnfrozen[workIndex] = unfrozenAccumVal + prevUnfrozenBlockAccumulator + gsWarpUnfrozen[warpIndex];
        }

        // The .cu closes the tile without a barrier and relies on warp convergence
        // between this tile's reads of the warp arrays (:331-332) and the next tile's
        // lane-31 writes (:300).  An emulation over 8-wide subgroups has no lockstep to
        // lean on, so the tile is closed explicitly (broadphaseEndPtHistogram.slang:208).
        GroupMemoryBarrierWithGroupSync();
    }

    // .cu:336-340: written for ALL 32 blocks, 0 for blocks whose chunk lies past the
    // end (and when nbTotalShapes == 0) -- pass 2 reads all 32 unconditionally.
    if (tid == 0)
    {
        gFrozenBlock[blockIdx] = gsBlockFrozen;
        gUnfrozenBlock[blockIdx] = gsBlockUnfrozen;
    }
}

// ---------------------------------------------------------------------------
// pass 2 (.cu:343-403): exclusive prefix over the 32 block totals, grand totals
// into the descriptor, block base folded into each block's partition
// engine grid (32, 1, 1) x 256
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void outputFrozenAndUnfrozenHistogram(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;

    Ptr<PxgSimulationCoreDescGpu> desc = Ptr<PxgSimulationCoreDescGpu>(gParams.scDesc);
    Ptr<uint> gFrozen = Ptr<uint>(desc[0].mFrozen);
    Ptr<uint> gUnfrozen = Ptr<uint>(desc[0].mUnfrozen);
    Ptr<uint> gFrozenBlock = Ptr<uint>(desc[0].mFrozenBlockAndRes);
    Ptr<uint> gUnfrozenBlock = Ptr<uint>(desc[0].mUnfrozenBlockAndRes);
    const uint nbTotalShapes = desc[0].mNbTotalShapes;

    // .cu:366-382: warp 0 loads the 32 block totals UNCONDITIONALLY (the host
    // allocates PxMax(32, n) entries, PxgSimulationCore.cpp:2150-2151, and pass 1
    // wrote all 32), exclusive-scans them, and block 0's lane 31 writes the grand
    // totals (exclusive prefix at 31 + its own value) into the desc's own members.
    // Serialized in thread 0; block 0 is the single writer, as in the .cu.
    if (tid == 0)
    {
        uint accFrozen = 0, accUnfrozen = 0;
        for (uint b = 0; b < NUM_BLOCKS; ++b)
        {
            gsBlockPrefixFrozen[b] = accFrozen;
            accFrozen += gFrozenBlock[b];
            gsBlockPrefixUnfrozen[b] = accUnfrozen;
            accUnfrozen += gUnfrozenBlock[b];
        }
        if (blockIdx == 0)
        {
            desc[0].mTotalFrozenShapes = accFrozen;      // .cu:380, byte 144
            desc[0].mTotalUnfrozenShapes = accUnfrozen;  // .cu:381, byte 148
        }
    }

    // .cu:384-386: the CONSTANT 32 here, not gridDim.x -- both passes must agree so
    // block b's base lands on exactly the elements block b scanned.
    const uint totalBlockRequired = (nbTotalShapes + BLOCK - 1) / BLOCK;
    const uint numIterationPerBlock = (totalBlockRequired + NUM_BLOCKS - 1) / NUM_BLOCKS;

    GroupMemoryBarrierWithGroupSync();       // .cu:388

    const uint frozenBlockAccum = gsBlockPrefixFrozen[blockIdx];
    const uint unfrozenBlockAccum = gsBlockPrefixUnfrozen[blockIdx];

    // .cu:393-402: plain read-modify-write; each element has exactly one owner
    // because the partitions are disjoint and identical to pass 1's.
    for (uint it = 0; it < numIterationPerBlock; ++it)
    {
        const uint workIndex = it * BLOCK + tid + numIterationPerBlock * blockIdx * BLOCK;
        if (workIndex < nbTotalShapes)
        {
            gFrozen[workIndex] = gFrozen[workIndex] + frozenBlockAccum;
            gUnfrozen[workIndex] = gUnfrozen[workIndex] + unfrozenBlockAccum;
        }
    }
}

// reduction.cuh:728-752 binarySearch<PxU32>: the largest index into the sorted
// array with data[index] <= value; 0 when there is none.  Transcribed, not
// re-derived: `left ? left - 1 : 0` is what turns a lower-bound search into the
// "last element <= value" the compaction relies on.
uint fhBinarySearch(Ptr<uint> data, uint numElements, uint value)
{
    uint left = 0;
    uint right = numElements;
    while (left < right)
    {
        const uint pos = (left + right) / 2;
        const uint element = data[pos];
        if (element <= value)
            left = pos + 1;
        else
            right = pos;
    }
    return left != 0u ? left - 1u : 0u;
}

// ---------------------------------------------------------------------------
// pass 3 (.cu:405-428): compact the frozen / unfrozen shape indices over the
// BlockAndRes buffers (their 32 block totals are consumed by now)
// engine grid (64, 1, 1) x 256, grid-stride
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void createFrozenAndUnfrozenArray(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    Ptr<PxgSimulationCoreDescGpu> desc = Ptr<PxgSimulationCoreDescGpu>(gParams.scDesc);
    Ptr<uint> gFrozen = Ptr<uint>(desc[0].mFrozen);
    Ptr<uint> gUnfrozen = Ptr<uint>(desc[0].mUnfrozen);
    Ptr<uint> gFrozenRes = Ptr<uint>(desc[0].mFrozenBlockAndRes);
    Ptr<uint> gUnfrozenRes = Ptr<uint>(desc[0].mUnfrozenBlockAndRes);

    const uint nbTotalShapes = desc[0].mNbTotalShapes;
    const uint nbFrozenTotalShapes = desc[0].mTotalFrozenShapes;      // written by pass 2
    const uint nbUnfrozenTotalShapes = desc[0].mTotalUnfrozenShapes;

    const uint idx = gtid.x + gid.x * BLOCK;         // .cu:417
    const uint stride = BLOCK * WorkgroupCount().x;   // .cu:419/424: blockDim.x * gridDim.x

    for (uint i = idx; i < nbFrozenTotalShapes; i += stride)
        gFrozenRes[i] = fhBinarySearch(gFrozen, nbTotalShapes, i);

    for (uint i = idx; i < nbUnfrozenTotalShapes; i += stride)
        gUnfrozenRes[i] = fhBinarySearch(gUnfrozen, nbTotalShapes, i);
}
