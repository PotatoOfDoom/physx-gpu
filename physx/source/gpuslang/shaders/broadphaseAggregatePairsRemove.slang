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

// broadphaseAggregatePairsRemove.slang
//
// Slang port of the aggregate-pair removal compaction (aggregate.cu):
//   - removeAggPairsStage1: per-block count of the aggregate pairs flagged for
//     removal (a non-zero entry in the remove bitmap).
//   - removeAggPairsStage2: exclusive prefix over the per-block counts, then a
//     stable compaction that appends each removed pair's index to the free list
//     (freeIndices[numFreeIndices + rank] = pairIndex) in increasing index
//     order, marks the pair dead, and records the total removed count.
//
// The CUDA kernels use a fixed 32-block x 1024-thread grid with warp
// __ballot_sync/__popc counting, __shfl-based between-block prefix and a
// per-warp/-block/-lane offset decomposition.  Because the result is a stable
// compaction keyed on the pair index, it is independent of the grid geometry;
// this port reuses the verified broadphaseReports two-pass structure (32 blocks
// of 256 threads, groupshared 32-lane scans) which visits the pairs in index
// order and produces the identical output.

static const uint BLOCK = 256;
static const uint NUM_BLOCKS = 32;
static const uint WARP = 32;
static const uint WARPS_PER_BLOCK = BLOCK / WARP; // 8

// PxgAggregatePair modelled as 4 uints {actorHandle0, actorHandle1, isNew, isDead}
static const uint AGGPAIR_STRIDE = 4u;
static const uint AGGPAIR_ISDEAD = 3u;

struct RemoveParams
{
    uint nbAggregatePairs;  // *aggPairCount
    uint numFreeIndices;    // freeBufferList->numFreeIndices (base for appended slots)
}

[[vk::push_constant]]
ConstantBuffer<RemoveParams> gParams;

RWStructuredBuffer<uint> gRemoveBitmap; // binding 0  removeBitmap [nbPairs] (non-zero = remove)
RWStructuredBuffer<uint> gHistogram;    // binding 1  per-block remove counts [NUM_BLOCKS]
RWStructuredBuffer<uint> gFreeIndices;  // binding 2  free-index list (append at numFreeIndices+rank)
RWStructuredBuffer<uint> gAggPairs;     // binding 3  aggPairs [nbPairs][4] (isDead written)
RWStructuredBuffer<uint> gNbRemoved;    // binding 4  total removed count [1]

groupshared uint gsLane[BLOCK];
groupshared uint gsWarp[WARPS_PER_BLOCK];
groupshared uint gsBlockAccum;
groupshared uint gsBlockHistogram[NUM_BLOCKS];

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

// exclusive scan over the warp totals in gsWarp; folds the grand total into
// gsBlockAccum.  Uniform control flow (all threads reach every barrier).
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
// removeAggPairsStage1: per-block remove count
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void removeAggPairsStage1(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;

    const uint nbPairs = gParams.nbAggregatePairs;
    const uint totalBlockRequired = (nbPairs + (BLOCK - 1)) / BLOCK;
    const uint numIterationPerBlock = (totalBlockRequired + (NUM_BLOCKS - 1)) / NUM_BLOCKS;

    uint localCount = 0;
    for (uint it = 0; it < numIterationPerBlock; ++it)
    {
        const uint workIndex = it * BLOCK + tid + numIterationPerBlock * blockIdx * BLOCK;
        if (workIndex < nbPairs && gRemoveBitmap[workIndex] != 0u)
            ++localCount;
    }

    if (tid == 0)
        gsBlockAccum = 0;
    gsLane[tid] = localCount;
    GroupMemoryBarrierWithGroupSync();
    warpScanLane(tid);
    if ((tid & (WARP - 1)) == WARP - 1)
        gsWarp[tid / WARP] = gsLane[tid];
    GroupMemoryBarrierWithGroupSync();
    scanWarpTotals(tid);
    GroupMemoryBarrierWithGroupSync();

    if (tid == 0)
        gHistogram[blockIdx] = gsBlockAccum;
}

// ---------------------------------------------------------------------------
// removeAggPairsStage2: between-block prefix + stable compaction
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void removeAggPairsStage2(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const uint warpIndex = tid / WARP;

    const uint nbPairs = gParams.nbAggregatePairs;
    const uint numFreeIndices = gParams.numFreeIndices;

    if (tid == 0)
        gsBlockAccum = 0;

    if (tid < NUM_BLOCKS)
        gsLane[tid] = gHistogram[tid];
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0)
    {
        uint acc = 0;
        for (uint i = 0; i < NUM_BLOCKS; ++i)
        {
            const uint v = gsLane[i];
            gsBlockHistogram[i] = acc;
            acc += v;
        }
        if (blockIdx == 0)
            gNbRemoved[0] = acc; // total removed
    }
    GroupMemoryBarrierWithGroupSync();

    const uint totalBlockRequired = (nbPairs + (BLOCK - 1)) / BLOCK;
    const uint numIterationPerBlock = (totalBlockRequired + (NUM_BLOCKS - 1)) / NUM_BLOCKS;
    const uint blockStartIndex = gsBlockHistogram[blockIdx];

    for (uint it = 0; it < numIterationPerBlock; ++it)
    {
        const uint workIndex = it * BLOCK + tid + numIterationPerBlock * blockIdx * BLOCK;
        const bool matches = (workIndex < nbPairs) && (gRemoveBitmap[workIndex] != 0u);

        gsLane[tid] = matches ? 1u : 0u;
        GroupMemoryBarrierWithGroupSync();
        warpScanLane(tid);
        const uint offset = gsLane[tid] - (matches ? 1u : 0u);
        if ((tid & (WARP - 1)) == WARP - 1)
            gsWarp[warpIndex] = gsLane[tid];

        const uint prevAccum = gsBlockAccum;
        GroupMemoryBarrierWithGroupSync();

        scanWarpTotals(tid);
        GroupMemoryBarrierWithGroupSync();

        if (matches)
        {
            const uint rank = offset + prevAccum + gsWarp[warpIndex] + blockStartIndex;
            gFreeIndices[numFreeIndices + rank] = workIndex;
            gAggPairs[workIndex * AGGPAIR_STRIDE + AGGPAIR_ISDEAD] = 1u; // isDead = true
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
