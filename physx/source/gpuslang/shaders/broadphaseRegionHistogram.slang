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

// broadphaseRegionHistogram.slang
//
// Slang port of the broad-phase region prefix-sum scan pair
// (gpubroadphase/src/CUDA/broadphase.cu, the templated device helpers
// computeHistogram / outputHistogram, L868-968).  These implement a plain
// device-wide exclusive prefix sum over a flat PxU32[nbProjections] array plus
// its grand total; the CUDA exposes them as four kernels —
// computeStartRegionsHistogram / outputStartRegionsHistogram (over
// startRegionAccum / blockStartRegionAccum / startRegionAccumTotal) and
// computeRegionsHistogram / outputRegionsHistogram (over regionAccum /
// blockRegionAccum / regionAccumTotal) — that are byte-identical modulo which
// descriptor array they scan, so the two generic entry points below cover all
// four dispatches (bind the start- or region-arrays).
//
//   - computeRegionsHistogram: per-block segmented exclusive prefix (the block
//     owns a contiguous partition of the array), writing the block-local
//     prefixes back in place and the per-block grand total to blockHistogram.
//   - outputRegionsHistogram: exclusive prefix over the 32 per-block totals
//     (grand total -> the scalar total buffer), added back into each block's
//     range so the array becomes a device-wide exclusive prefix sum.
//
// This is the plain (single-value, no packing, no scatter) instance of the same
// two-pass scan used by broadphaseIncrementalHistogram.  Portability: the CUDA
// warpScanAdd / warpScanAddWriteToSharedMem __shfl scans become groupshared
// 32-lane scans under uniform GroupMemoryBarrierWithGroupSync(); the fixed
// 32-block device grid (BLOCK = 256, 8 warps, gridDim == BLOCK_SIZE == 32) and
// its sequential between-tile / between-block accumulation are preserved exactly.

static const uint BLOCK = 256;
static const uint NUM_BLOCKS = 32; // gridDim.x == BLOCK_SIZE == 32 (load-bearing)
static const uint WARP = 32;
static const uint WARPS_PER_BLOCK = BLOCK / WARP; // 8

struct RegionHistParams
{
    uint maxWorkIndex; // = nbProjections (scan length)
}

[[vk::push_constant]]
ConstantBuffer<RegionHistParams> gParams;

RWStructuredBuffer<uint> gHistogram;      // binding 0  startRegionAccum / regionAccum [nbProj] (in/out)
RWStructuredBuffer<uint> gBlockHistogram; // binding 1  blockStartRegionAccum / blockRegionAccum [32]
RWStructuredBuffer<uint> gTotal;          // binding 2  {start,region}RegionAccumTotal [1] (output only)

groupshared uint gsLane[BLOCK];          // per-lane scan scratch
groupshared uint gsWarp[WARPS_PER_BLOCK];
groupshared uint gsBlockAccum;           // running total over this block's tiles
groupshared uint gsBlockPrefix[NUM_BLOCKS];

// inclusive scan of gsLane within each 32-lane warp (all warps in parallel)
void warpScanLane(uint tid)
{
    for (uint off = 1; off < WARP; off <<= 1)
    {
        const uint lane = tid & (WARP - 1);
        const uint add = (lane >= off) ? gsLane[tid - off] : 0u;
        GroupMemoryBarrierWithGroupSync();
        gsLane[tid] += add;
        GroupMemoryBarrierWithGroupSync();
    }
}

// exclusive scan over the WARPS_PER_BLOCK warp totals in gsWarp; folds the grand
// total into gsBlockAccum.  Uniform control flow (all threads reach every
// barrier); only lanes < 8 touch gsWarp.
void scanWarpTotals(uint tid)
{
    const bool active = tid < WARPS_PER_BLOCK;
    uint p0 = active ? gsWarp[tid] : 0u;
    uint pInc = p0;
    for (uint off = 1; off < WARPS_PER_BLOCK; off <<= 1)
    {
        const bool take = active && tid >= off;
        const uint pa = take ? gsWarp[tid - off] : 0u;
        GroupMemoryBarrierWithGroupSync();
        pInc += pa;
        if (active) gsWarp[tid] = pInc;
        GroupMemoryBarrierWithGroupSync();
    }
    GroupMemoryBarrierWithGroupSync();
    if (active) gsWarp[tid] = pInc - p0;
    if (tid == WARPS_PER_BLOCK - 1) gsBlockAccum += pInc;
}

// ---------------------------------------------------------------------------
// pass 1: per-block segmented exclusive scan + block totals
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void computeRegionsHistogram(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const uint warpIndex = tid / WARP;
    const uint lane = tid & (WARP - 1);

    const uint maxWorkIndex = gParams.maxWorkIndex;
    const uint nbBlocksRequired = (maxWorkIndex + BLOCK - 1) / BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + NUM_BLOCKS - 1) / NUM_BLOCKS;

    if (tid == 0)
        gsBlockAccum = 0;
    GroupMemoryBarrierWithGroupSync();

    for (uint it = 0; it < nbIterationsPerBlock; ++it)
    {
        const uint workIndex = it * BLOCK + tid + nbIterationsPerBlock * blockIdx * BLOCK;

        const uint histVal = (workIndex < maxWorkIndex) ? gHistogram[workIndex] : 0u;

        gsLane[tid] = histVal;
        GroupMemoryBarrierWithGroupSync();
        warpScanLane(tid);
        const uint accumVal = gsLane[tid] - histVal; // exclusive prefix within warp
        if (lane == WARP - 1)
            gsWarp[warpIndex] = gsLane[tid]; // inclusive warp total

        const uint prevBlockAccum = gsBlockAccum; // block base from prior tiles
        GroupMemoryBarrierWithGroupSync();

        scanWarpTotals(tid); // gsWarp -> exclusive prefixes; gsBlockAccum += tile total

        GroupMemoryBarrierWithGroupSync();

        if (workIndex < maxWorkIndex)
            gHistogram[workIndex] = accumVal + prevBlockAccum + gsWarp[warpIndex];

        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0)
        gBlockHistogram[blockIdx] = gsBlockAccum;
}

// ---------------------------------------------------------------------------
// pass 2: exclusive prefix over the per-block totals + fold into each block
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void outputRegionsHistogram(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;

    const uint maxWorkIndex = gParams.maxWorkIndex;
    const uint nbBlocksRequired = (maxWorkIndex + BLOCK - 1) / BLOCK;
    const uint numIterationPerBlock = (nbBlocksRequired + NUM_BLOCKS - 1) / NUM_BLOCKS;

    // exclusive prefix over the 32 per-block totals
    if (tid == 0)
    {
        uint acc = 0;
        for (uint b = 0; b < NUM_BLOCKS; ++b)
        {
            const uint v = gBlockHistogram[b];
            gsBlockPrefix[b] = acc; // exclusive prefix -> block base
            acc += v;
        }
        if (blockIdx == 0)
            gTotal[0] = acc; // grand total
    }
    GroupMemoryBarrierWithGroupSync();

    const uint blockAccum = gsBlockPrefix[blockIdx];

    for (uint it = 0; it < numIterationPerBlock; ++it)
    {
        const uint workIndex = it * BLOCK + tid + numIterationPerBlock * blockIdx * BLOCK;
        if (workIndex < maxWorkIndex)
            gHistogram[workIndex] = gHistogram[workIndex] + blockAccum;
    }
}
