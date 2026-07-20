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
// P5 Step 4 (BDA, engine launch path): the broad-phase region prefix-sum scan
// pair (gpubroadphase/src/CUDA/broadphase.cu, the templated device helpers
// computeHistogram / outputHistogram instantiated as computeRegionsHistogram /
// outputRegionsHistogram, launched as BP_COMPUTE_REGION_HISTOGRAM /
// BP_OUTPUT_REGION_HISTOGRAM), driven exactly as the engine launches them:
// ONE PxgBroadPhaseDesc device pointer on a (32, 1, 1) grid.  The aggregate is
// walked through the byte-exact PxgBroadPhaseDescGpu mirror
// (pxgSlangBpDesc.slang, proven by testBpDescLayout): the scan runs over
// regionAccum / blockRegionAccum with nbProjections = 2 * (numPreviousHandles
// + numCreatedHandles - numRemovedHandles), and the grand total lands in the
// desc's own regionAccumTotal member (the .cu's by-reference write — hence its
// non-const PxgBroadPhaseDesc* signature).
//
//   - computeRegionsHistogram: per-block segmented exclusive prefix (the block
//     owns a contiguous partition of the array), writing the block-local
//     prefixes back in place and the per-block grand total to blockRegionAccum.
//   - outputRegionsHistogram: exclusive prefix over the 32 per-block totals
//     (grand total -> bpDesc->regionAccumTotal), added back into each block's
//     range so the array becomes a device-wide exclusive prefix sum.
//
// The .cu's computeStartRegionsHistogram / outputStartRegionsHistogram are
// byte-identical modulo scanning startRegionAccum / blockStartRegionAccum /
// startRegionAccumTotal instead; they remain unmapped manifest rows.
//
// Portability: the CUDA warpScanAdd / warpScanAddWriteToSharedMem __shfl scans
// become groupshared 32-lane scans under uniform
// GroupMemoryBarrierWithGroupSync(); computeHistogram's iteration count uses
// gridDim.x (WorkgroupCount().x) exactly like the .cu, while outputHistogram
// uses the BLOCK_SIZE (= 32) constant, also exactly like the .cu.

import pxgSlangBpDesc;

static const uint BLOCK = 256;
static const uint NUM_BLOCKS = 32; // broadphase.cu BLOCK_SIZE == engine grid width (load-bearing)
static const uint WARP = 32;
static const uint WARPS_PER_BLOCK = BLOCK / WARP; // 8

// Engine ABI: one device pointer to the PxgBroadPhaseDesc aggregate.
struct RegionHistParams
{
    uint64_t bpDesc;
}
[[vk::push_constant]]
ConstantBuffer<RegionHistParams> gParams;

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
// (BP_COMPUTE_REGION_HISTOGRAM, engine grid (32, 1, 1))
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void computeRegionsHistogram(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const uint warpIndex = tid / WARP;
    const uint lane = tid & (WARP - 1);

    Ptr<PxgBroadPhaseDescGpu> desc = Ptr<PxgBroadPhaseDescGpu>(gParams.bpDesc);
    Ptr<uint> histogram = Ptr<uint>(desc[0].regionAccum);
    Ptr<uint> blockHistogram = Ptr<uint>(desc[0].blockRegionAccum);

    const uint numHandles = desc[0].numPreviousHandles + desc[0].numCreatedHandles - desc[0].numRemovedHandles;
    const uint maxWorkIndex = numHandles * 2;

    const uint nbBlocksRequired = (maxWorkIndex + BLOCK - 1) / BLOCK;
    // the .cu derives this from gridDim.x (== 32 in the engine launch)
    const uint gridDimX = WorkgroupCount().x;
    const uint nbIterationsPerBlock = (nbBlocksRequired + gridDimX - 1) / gridDimX;

    if (tid == 0)
        gsBlockAccum = 0;
    GroupMemoryBarrierWithGroupSync();

    for (uint it = 0; it < nbIterationsPerBlock; ++it)
    {
        const uint workIndex = it * BLOCK + tid + nbIterationsPerBlock * blockIdx * BLOCK;

        const uint histVal = (workIndex < maxWorkIndex) ? histogram[workIndex] : 0u;

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
            histogram[workIndex] = accumVal + prevBlockAccum + gsWarp[warpIndex];

        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0)
        blockHistogram[blockIdx] = gsBlockAccum;
}

// ---------------------------------------------------------------------------
// pass 2: exclusive prefix over the per-block totals + fold into each block
// (BP_OUTPUT_REGION_HISTOGRAM, engine grid (32, 1, 1))
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void outputRegionsHistogram(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;

    Ptr<PxgBroadPhaseDescGpu> desc = Ptr<PxgBroadPhaseDescGpu>(gParams.bpDesc);
    Ptr<uint> histogram = Ptr<uint>(desc[0].regionAccum);
    Ptr<uint> blockHistogram = Ptr<uint>(desc[0].blockRegionAccum);

    const uint numHandles = desc[0].numPreviousHandles + desc[0].numCreatedHandles - desc[0].numRemovedHandles;
    const uint maxWorkIndex = numHandles * 2;

    // exclusive prefix over the 32 per-block totals; the grand total goes to
    // the desc's own regionAccumTotal member (the .cu writes it from
    // globalThreadIndex == BLOCK_SIZE-1 == 31, i.e. block 0)
    if (tid == 0)
    {
        uint acc = 0;
        for (uint b = 0; b < NUM_BLOCKS; ++b)
        {
            gsBlockPrefix[b] = acc; // exclusive prefix -> block base
            acc += blockHistogram[b];
        }
        if (blockIdx == 0)
            desc[0].regionAccumTotal = acc; // grand total
    }
    GroupMemoryBarrierWithGroupSync();

    const uint blockAccum = gsBlockPrefix[blockIdx];

    // the .cu uses the BLOCK_SIZE (= 32) constant here, NOT gridDim.x
    const uint totalBlockRequired = (maxWorkIndex + BLOCK - 1) / BLOCK;
    const uint numIterationPerBlock = (totalBlockRequired + NUM_BLOCKS - 1) / NUM_BLOCKS;

    for (uint it = 0; it < numIterationPerBlock; ++it)
    {
        const uint workIndex = it * BLOCK + tid + numIterationPerBlock * blockIdx * BLOCK;
        if (workIndex < maxWorkIndex)
            histogram[workIndex] = histogram[workIndex] + blockAccum;
    }
}
