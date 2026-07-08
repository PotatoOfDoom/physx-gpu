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

// broadphaseEndPtHistogram.slang
//
// Slang port of the broad-phase start/end-point projection histogram builders
// (gpubroadphase/src/CUDA/broadphase.cu, firstPassEndPtsHistogram /
// secondPassEndPtsHistogram, launched as BP_COMPUTE_ENDPT_HISTOGRAM /
// BP_OUTPUT_ENDPT_HISTOGRAM).  These produce the startPtHistogram /
// endPtHistogram prefix-sum arrays (and the ordered start/end point handles)
// that the incremental comparison-count stage (broadphaseIncrementalHistogram)
// consumes verbatim.
//
//   - computeEndPtsHistogram: for every projection on each axis, classify it as
//     a (non-deleted) start, a (non-deleted) end, and/or any end (end2, which
//     includes deleted ends), then run a per-block segmented exclusive prefix
//     producing block-local prefixes in {start,end,totalEnd}PtHistogram and the
//     block grand totals in block{Start,End,TotalEnd}PtHistogram.  The three
//     0/1 counters are packed into one uint (10 bits each: per-warp sum <= 32,
//     per-block sum <= 256 < 1024) so a single scan yields all three.
//   - outputEndPtsHistogram: exclusive prefix over the 32 per-block totals, add
//     each block's offset back into the histograms (promoting them to a
//     device-wide exclusive prefix), and compactly scatter each non-deleted
//     handle into its class array (start/end point handles) at its global slot.
//
// Two frames are processed in incremental mode: frame 0 (current, writes the
// total-end histogram) always, frame 1 (previous, no total) only when handles
// were removed.  Non-incremental mode processes only frame 0 with
// numHandles = numPreviousHandles + numCreatedHandles - numRemovedHandles.
//
// Portability: the CUDA __ballot_sync/__popc warp scans become groupshared
// 32-lane scans under uniform GroupMemoryBarrierWithGroupSync(); the fixed
// 32-block device grid (BLOCK = 256, 8 warps) and its sequential
// between-tile / between-block accumulation are preserved exactly.

static const uint BLOCK = 256;
static const uint NUM_BLOCKS = 32;
static const uint WARP = 32;
static const uint WARPS_PER_BLOCK = BLOCK / WARP; // 8
static const uint FIELD_MASK = 0x3FFu;            // 10-bit packed field

struct EndPtHistParams
{
    uint numPreviousHandles;
    uint numCreatedHandles;
    uint numRemovedHandles;
    uint isIncremental; // 0/1, uniform across the launch
}

[[vk::push_constant]]
ConstantBuffer<EndPtHistParams> gParams;

// per-(frame,axis) arrays packed flat: [(frame*3 + axis) * stride + i], frame 0
// = current, frame 1 = previous.  Histograms / boxHandles use stride P =
// nbProjections; ordered handles use stride H = numHandles; block companions
// use stride NUM_BLOCKS.  The total-end arrays have no frame dimension (axis
// only), matching totalEndPtHistogram[3] / blockTotalEndPtHistogram[3].
RWStructuredBuffer<uint> gBoxHandles;      // binding 0  boxHandles[frame][axis]        (R)
RWStructuredBuffer<uint> gEndHist;         // binding 1  endPtHistogram[frame][axis]     (R/W)
RWStructuredBuffer<uint> gStartHist;       // binding 2  startPtHistogram[frame][axis]   (R/W)
RWStructuredBuffer<uint> gTotalEndHist;    // binding 3  totalEndPtHistogram[axis]       (R/W, frame0)
RWStructuredBuffer<uint> gBlockEndHist;    // binding 4  blockEndPtHistogram[frame][axis](R/W)
RWStructuredBuffer<uint> gBlockStartHist;  // binding 5  blockStartPtHistogram[frame][axis](R/W)
RWStructuredBuffer<uint> gBlockTotalEnd;   // binding 6  blockTotalEndPtHistogram[axis]  (R/W, frame0)
RWStructuredBuffer<uint> gEndHandles;      // binding 7  endPointHandles[frame][axis]    (W)
RWStructuredBuffer<uint> gStartHandles;    // binding 8  startPointHandles[frame][axis]  (W)

bool isStartProjection(uint sortedHandle)
{
    return (sortedHandle & 1u) != 0u;
}

bool isDeletedProjection(uint sortedHandle)
{
    return (sortedHandle & (1u << 2)) != 0u;
}

uint getHandle(uint sortedHandle)
{
    return sortedHandle >> 3;
}

groupshared uint gsLane[BLOCK];          // per-lane packed scan scratch (pass 1)
groupshared uint gsWarp[WARPS_PER_BLOCK]; // packed warp totals (pass 1)
groupshared uint gsAccumStart;           // running block totals across tiles (pass 1)
groupshared uint gsAccumEnd;
groupshared uint gsAccumEnd2;
groupshared uint gsBlockEnd[NUM_BLOCKS];  // exclusive prefixes of block totals (pass 2)
groupshared uint gsBlockStart[NUM_BLOCKS];
groupshared uint gsBlockTotalEnd[NUM_BLOCKS];

// inclusive scan of gsLane within each 32-lane warp (all warps in parallel).
// Packed fields never carry across the 10-bit boundaries (per-warp sum <= 32).
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

// exclusive scan over the WARPS_PER_BLOCK packed warp totals in gsWarp; unpacks
// and folds the grand total into gsAccumStart/End/End2.  Uniform control flow
// (all threads reach every barrier); only lanes < 8 touch gsWarp.
void scanWarpTotalsPacked(uint tid)
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
    if (tid == WARPS_PER_BLOCK - 1)
    {
        gsAccumEnd2 += pInc & FIELD_MASK;
        gsAccumEnd += (pInc >> 10) & FIELD_MASK;
        gsAccumStart += pInc >> 20;
    }
}

uint numHandles0()
{
    if (gParams.isIncremental != 0u)
        return gParams.numPreviousHandles;
    return gParams.numPreviousHandles + gParams.numCreatedHandles - gParams.numRemovedHandles;
}

// ---------------------------------------------------------------------------
// pass 1: per-block segmented prefixes + block totals
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void computeEndPtsHistogram(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const uint warpIndex = tid / WARP;
    const uint lane = tid & (WARP - 1);

    const uint nH = numHandles0();
    const uint P = nH * 2;
    const uint nbBlocksRequired = (P + BLOCK - 1) / BLOCK;
    const uint numIterationPerBlock = (nbBlocksRequired + NUM_BLOCKS - 1) / NUM_BLOCKS;

    const bool doFrame1 = (gParams.isIncremental != 0u) && (gParams.numRemovedHandles != 0u);
    const uint numFrames = doFrame1 ? 2u : 1u;

    for (uint frame = 0; frame < numFrames; ++frame)
    {
        const bool writeTotal = (frame == 0u);
        for (uint axis = 0; axis < 3; ++axis)
        {
            const uint faBase = (frame * 3u + axis) * P;
            const uint taBase = axis * P;
            const uint blkBase = (frame * 3u + axis) * NUM_BLOCKS;
            const uint tblkBase = axis * NUM_BLOCKS;

            if (tid == WARPS_PER_BLOCK - 1)
            {
                gsAccumStart = 0;
                gsAccumEnd = 0;
                gsAccumEnd2 = 0;
            }
            GroupMemoryBarrierWithGroupSync();

            for (uint it = 0; it < numIterationPerBlock; ++it)
            {
                const uint workIndex = it * BLOCK + tid + numIterationPerBlock * blockIdx * BLOCK;

                uint start = 0, end = 0, end2 = 0;
                if (workIndex < P)
                {
                    const uint sortedHandle = gBoxHandles[faBase + workIndex];
                    end2 = 1u - (isStartProjection(sortedHandle) ? 1u : 0u);
                    if (!isDeletedProjection(sortedHandle))
                    {
                        start = 1u - end2;
                        end = end2;
                    }
                }

                const uint val = (start << 20) | (end << 10) | end2;
                gsLane[tid] = val;
                GroupMemoryBarrierWithGroupSync();
                warpScanLane(tid);
                const uint endPtAccum = gsLane[tid] - val; // exclusive prefix within warp (packed)
                if (lane == WARP - 1)
                    gsWarp[warpIndex] = gsLane[tid]; // inclusive warp total (packed)

                const uint prevEnd = gsAccumEnd;
                const uint prevEnd2 = gsAccumEnd2;
                const uint prevStart = gsAccumStart;
                GroupMemoryBarrierWithGroupSync();

                scanWarpTotalsPacked(tid); // gsWarp -> exclusive prefixes; gsAccum* += tile total

                GroupMemoryBarrierWithGroupSync();

                if (workIndex < P)
                {
                    const uint accumulation = endPtAccum + gsWarp[warpIndex];
                    const uint endCount = ((accumulation >> 10) & FIELD_MASK) + prevEnd;
                    const uint startCount = (accumulation >> 20) + prevStart;
                    gEndHist[faBase + workIndex] = endCount;
                    gStartHist[faBase + workIndex] = startCount;
                    if (writeTotal)
                        gTotalEndHist[taBase + workIndex] = (accumulation & FIELD_MASK) + prevEnd2;
                }

                GroupMemoryBarrierWithGroupSync();
            }

            if (tid == WARPS_PER_BLOCK - 1)
            {
                gBlockEndHist[blkBase + blockIdx] = gsAccumEnd;
                gBlockStartHist[blkBase + blockIdx] = gsAccumStart;
                if (writeTotal)
                    gBlockTotalEnd[tblkBase + blockIdx] = gsAccumEnd2;
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
}

// ---------------------------------------------------------------------------
// pass 2: promote block prefixes to device-wide prefixes + scatter handles
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void outputEndPtsHistogram(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;

    const uint nH = numHandles0();
    const uint P = nH * 2;
    const uint nbBlocksRequired = (P + BLOCK - 1) / BLOCK;
    const uint numIterationPerBlock = (nbBlocksRequired + NUM_BLOCKS - 1) / NUM_BLOCKS;

    const bool doFrame1 = (gParams.isIncremental != 0u) && (gParams.numRemovedHandles != 0u);
    const uint numFrames = doFrame1 ? 2u : 1u;

    for (uint frame = 0; frame < numFrames; ++frame)
    {
        const bool writeTotal = (frame == 0u);
        for (uint axis = 0; axis < 3; ++axis)
        {
            const uint faBase = (frame * 3u + axis) * P;
            const uint taBase = axis * P;
            const uint blkBase = (frame * 3u + axis) * NUM_BLOCKS;
            const uint tblkBase = axis * NUM_BLOCKS;
            const uint hBase = (frame * 3u + axis) * nH;

            // exclusive prefix over the 32 per-block totals
            if (tid == 0)
            {
                uint accE = 0, accS = 0, accT = 0;
                for (uint b = 0; b < NUM_BLOCKS; ++b)
                {
                    gsBlockEnd[b] = accE;
                    accE += gBlockEndHist[blkBase + b];
                    gsBlockStart[b] = accS;
                    accS += gBlockStartHist[blkBase + b];
                    if (writeTotal)
                    {
                        gsBlockTotalEnd[b] = accT;
                        accT += gBlockTotalEnd[tblkBase + b];
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();

            const uint endBase = gsBlockEnd[blockIdx];
            const uint startBase = gsBlockStart[blockIdx];
            const uint totalEndBase = writeTotal ? gsBlockTotalEnd[blockIdx] : 0u;

            for (uint it = 0; it < numIterationPerBlock; ++it)
            {
                const uint workIndex = it * BLOCK + tid + numIterationPerBlock * blockIdx * BLOCK;
                if (workIndex < P)
                {
                    const uint sortedHandle = gBoxHandles[faBase + workIndex];
                    const bool startProjection = isStartProjection(sortedHandle);
                    const bool isDeleted = isDeletedProjection(sortedHandle);

                    const uint eIdx = faBase + workIndex;
                    const uint endGlobal = gEndHist[eIdx] + endBase;
                    const uint startGlobal = gStartHist[eIdx] + startBase;
                    gEndHist[eIdx] = endGlobal;
                    gStartHist[eIdx] = startGlobal;

                    if (!isDeleted)
                    {
                        const uint handle = getHandle(sortedHandle);
                        if (startProjection)
                            gStartHandles[hBase + startGlobal] = handle;
                        else
                            gEndHandles[hBase + endGlobal] = handle;
                    }

                    if (writeTotal)
                    {
                        const uint tIdx = taBase + workIndex;
                        gTotalEndHist[tIdx] = gTotalEndHist[tIdx] + totalEndBase;
                    }
                }
            }

            GroupMemoryBarrierWithGroupSync();
        }
    }
}
