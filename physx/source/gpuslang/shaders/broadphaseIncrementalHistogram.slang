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

// broadphaseIncrementalHistogram.slang
//
// Slang port of the incremental-SAP comparison-count stage of the broad phase
// (gpubroadphase/src/CUDA/broadphase.cu):
//   - computeIncrementalComparisonHistograms_Stage1: for every projection on
//     each of the three axes, count how many overlap comparisons the
//     incremental sweep-and-prune must perform for that projection, then run a
//     per-block segmented exclusive prefix over those counts (writing the
//     within-block running total to incrementalComparisons and the block total
//     to incrementalBlockComparisons).
//   - computeIncrementalComparisonHistograms_Stage2: exclusive prefix over the
//     32 per-block totals (producing totalIncrementalComparisons) and adding
//     each block's offset back into incrementalComparisons, yielding a
//     device-wide exclusive prefix sum of the per-projection comparison counts.
//
// Comparison count for a projection is a difference of two entries of a
// prefix-sum histogram, selected from a 2x2x3 table indexed by
// [isStartHandle][isDownPass][axis]:
//   isStartHandle -> end-point histogram (a start projection overlaps end
//                    points), otherwise start-point histogram;
//   isDownPass    -> last frame's histogram (readIdx moved down/stayed), else
//                    this frame's histogram.
// Deleted projections contribute 0.  The [start,end) window is derived from the
// projection's post-sort rank; rank is clamped to the previous list size on the
// down pass to avoid overrunning the histogram.
//
// Portability: the CUDA warpScanAdd/__shfl_sync scans become groupshared
// 32-lane scans; the fixed 32-block device grid (BLOCK = 256, 8 warps) and its
// sequential between-tile / between-block accumulation are preserved exactly.
//
// P5 Step 4 (BDA, engine launch path): both stages take ONE PxgBroadPhaseDesc
// device pointer (bpDescd) and walk the byte-exact PxgBroadPhaseDescGpu mirror
// (pxgSlangBpDesc.slang, proven by testBpDescLayout).  Every per-axis /
// per-(frame,axis) array that the bound-buffer form packed axis-major into one
// flat buffer is a distinct device pointer inside the desc here — so the flat
// axisBase / (frame*3+axis)*nbProj offsets collapse to per-pointer indexing.
// The engine launches both stages with grid.x = 32 (BP_COMPUTE_INCREMENTAL_
// CMP_COUNTS1/2), so NUM_BLOCKS = gridDim.x is a fixed device-grid constant
// exactly as the .cu assumes (the per-block totals array is sized 32 and
// stage 2 scans those 32 entries).

import pxgSlangBpDesc;

static const uint BLOCK = 256;
static const uint NUM_BLOCKS = 32;
static const uint WARP = 32;
static const uint WARPS_PER_BLOCK = BLOCK / WARP; // 8

// engine ABI: the PxgBroadPhaseDesc device pointer only
struct BpDescParams
{
    uint64_t bpDesc;
}

[[vk::push_constant]]
ConstantBuffer<BpDescParams> gBp;

bool isStartProjection(uint sortedHandle)
{
    return (sortedHandle & 1u) != 0u;
}

bool isDeletedProjection(uint sortedHandle)
{
    return (sortedHandle & (1u << 2)) != 0u;
}

groupshared uint gsLane[BLOCK];         // per-lane scan scratch
groupshared uint gsWarp[WARPS_PER_BLOCK];
groupshared uint gsBlockAccum;          // running total over this block's tiles

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
// stage 1: per-projection comparison count + per-block segmented exclusive scan
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void computeIncrementalComparisonHistograms_Stage1(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const uint warpIndex = tid / WARP;
    const uint lane = tid & (WARP - 1);

    Ptr<PxgBroadPhaseDescGpu> desc = Ptr<PxgBroadPhaseDescGpu>(gBp.bpDesc);
    const uint numHandles = desc[0].numPreviousHandles;
    const uint nbProjections = numHandles * 2;

    const uint nbBlocksRequired = (nbProjections + BLOCK - 1) / BLOCK;
    const uint numIterationPerBlock = (nbBlocksRequired + NUM_BLOCKS - 1) / NUM_BLOCKS;

    for (uint axis = 0; axis < 3; ++axis)
    {
        if (tid == 0)
            gsBlockAccum = 0;
        GroupMemoryBarrierWithGroupSync();

        // per-axis device pointers (boxProjectionRanks[axis], boxHandles[0][axis],
        // incrementalComparisons[axis]); the this-frame/last-frame start & end
        // histograms are boxwise [frame*3+axis].
        Ptr<uint> ranks = Ptr<uint>(desc[0].boxProjectionRanks[axis]);
        Ptr<uint> handles = Ptr<uint>(desc[0].boxHandles[axis]);
        Ptr<uint> incCmp = Ptr<uint>(desc[0].incrementalComparisons[axis]);

        for (uint it = 0; it < numIterationPerBlock; ++it)
        {
            const uint blockId = blockIdx * numIterationPerBlock + it;
            const uint readIdx = blockId * BLOCK + tid;

            uint comparisons = 0;
            if (readIdx < nbProjections)
            {
                const uint rank = ranks[readIdx];
                const uint sortedHandle = handles[readIdx];
                const bool isDownPass = readIdx <= rank;
                const bool isStartHandle = isStartProjection(sortedHandle);

                // clamp rank into the previous list on the down pass so the
                // histogram index cannot overrun the buffer
                const uint start = isDownPass ? readIdx : (rank + 1);
                const uint end = isDownPass ? min(rank, nbProjections - 1) : readIdx;

                if (!isDeletedProjection(sortedHandle))
                {
                    // sStartEndAccum[isStartHandle][isDownPass][axis]:
                    //   isStartHandle -> end-point histogram, else start-point
                    //   isDownPass    -> last frame (1), else this frame (0)
                    const uint frame = isDownPass ? 1u : 0u;
                    const uint fa = frame * 3u + axis;
                    Ptr<uint> hist = isStartHandle ? Ptr<uint>(desc[0].endPtHistogram[fa])
                                                   : Ptr<uint>(desc[0].startPtHistogram[fa]);
                    comparisons = hist[end] - hist[start];
                }
            }

            // intra-warp exclusive prefix (res) + warp total
            gsLane[tid] = comparisons;
            GroupMemoryBarrierWithGroupSync();
            warpScanLane(tid);
            const uint res = gsLane[tid] - comparisons;
            if (lane == WARP - 1)
                gsWarp[warpIndex] = gsLane[tid];

            const uint accum = gsBlockAccum; // running total of previous tiles
            GroupMemoryBarrierWithGroupSync();

            scanWarpTotals(tid); // gsWarp -> exclusive prefixes; gsBlockAccum += tile total

            GroupMemoryBarrierWithGroupSync();

            if (readIdx < nbProjections)
                incCmp[readIdx] = accum + res + gsWarp[warpIndex];

            GroupMemoryBarrierWithGroupSync();
        }

        if (tid == 0)
            (Ptr<uint>(desc[0].incrementalBlockComparisons[axis]))[blockIdx] = gsBlockAccum;

        GroupMemoryBarrierWithGroupSync();
    }
}

// ---------------------------------------------------------------------------
// stage 2: exclusive prefix over the per-block totals + fold into each block
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void computeIncrementalComparisonHistograms_Stage2(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;

    Ptr<PxgBroadPhaseDescGpu> desc = Ptr<PxgBroadPhaseDescGpu>(gBp.bpDesc);
    const uint numHandles = desc[0].numPreviousHandles;
    const uint nbProjections = numHandles * 2;

    const uint nbBlocksRequired = (nbProjections + BLOCK - 1) / BLOCK;
    const uint numIterationPerBlock = (nbBlocksRequired + NUM_BLOCKS - 1) / NUM_BLOCKS;

    for (uint axis = 0; axis < 3; ++axis)
    {
        Ptr<uint> incBlock = Ptr<uint>(desc[0].incrementalBlockComparisons[axis]);
        Ptr<uint> incCmp = Ptr<uint>(desc[0].incrementalComparisons[axis]);

        // exclusive prefix over the 32 per-block comparison totals
        if (tid == 0)
        {
            uint acc = 0;
            for (uint b = 0; b < NUM_BLOCKS; ++b)
            {
                const uint v = incBlock[b];
                gsLane[b] = acc; // exclusive prefix -> block offset
                acc += v;
            }
            if (blockIdx == 0)
                desc[0].totalIncrementalComparisons[axis] = acc; // grand total (inline desc field)
        }
        GroupMemoryBarrierWithGroupSync();

        const uint blockAccumulation = gsLane[blockIdx];

        for (uint it = 0; it < numIterationPerBlock; ++it)
        {
            const uint blockId = blockIdx * numIterationPerBlock + it;
            const uint readIdx = blockId * BLOCK + tid;
            if (readIdx < nbProjections)
                incCmp[readIdx] = incCmp[readIdx] + blockAccumulation;
        }

        GroupMemoryBarrierWithGroupSync();
    }
}
