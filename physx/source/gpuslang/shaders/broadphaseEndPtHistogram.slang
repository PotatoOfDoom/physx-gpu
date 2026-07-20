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
// P5 Step 4 (BDA, engine launch path): the broad-phase start/end-point
// projection histogram builders (gpubroadphase/src/CUDA/broadphase.cu,
// firstPassEndPtsHistogram / secondPassEndPtsHistogram, launched as
// BP_COMPUTE_ENDPT_HISTOGRAM / BP_OUTPUT_ENDPT_HISTOGRAM), driven exactly as
// the engine launches them: (const PxgBroadPhaseDesc*, bool isIncremental) on
// a (32, 3, 1) grid where blockIdx.y is the axis.  The PxgBroadPhaseDesc
// aggregate arrives BY DEVICE POINTER and is walked through the byte-exact
// PxgBroadPhaseDescGpu mirror (pxgSlangBpDesc.slang, proven by
// testBpDescLayout).
//
//   - computeEndPtsHistogram: for every projection on this block's axis,
//     classify it as a (non-deleted) start, a (non-deleted) end, and/or any
//     end (end2, which includes deleted ends), then run a per-block segmented
//     exclusive prefix producing block-local prefixes in {start,end,totalEnd}
//     PtHistogram and the block grand totals in block{Start,End,TotalEnd}
//     PtHistogram.  The three 0/1 counters are packed into one uint (10 bits
//     each: per-warp sum <= 32, per-block sum <= 256 < 1024) so a single scan
//     yields all three.
//   - outputEndPtsHistogram: exclusive prefix over the 32 per-block totals,
//     add each block's offset back into the histograms (promoting them to a
//     device-wide exclusive prefix), and compactly scatter each non-deleted
//     handle into its class array (start/end point handles) at its global slot.
//
// Incremental mode processes frame 0 (current, numHandles =
// numPreviousHandles, writes the total-end histogram) and, when handles were
// removed, re-runs on frame 1 (previous) with the total-end pointers NULL —
// the .cu passes endPtsHistogram2 = NULL / blocksEndPtsAccum2 = NULL,
// represented here as address == 0 checks.  Non-incremental mode processes
// only frame 0 with numHandles = numPreviousHandles + numCreatedHandles -
// numRemovedHandles.
//
// Portability: the CUDA __ballot_sync/__popc warp scans become groupshared
// 32-lane scans under uniform GroupMemoryBarrierWithGroupSync(); the fixed
// 32-block device grid (BLOCK = 256, 8 warps, BLOCK_SIZE = 32) and its
// sequential between-tile / between-block accumulation are preserved exactly.

import pxgSlangBpDesc;

static const uint BLOCK = 256;
static const uint NUM_BLOCKS = 32;                // broadphase.cu BLOCK_SIZE (grid width)
static const uint WARP = 32;
static const uint WARPS_PER_BLOCK = BLOCK / WARP; // 8
static const uint FIELD_MASK = 0x3FFu;            // 10-bit packed field

// Engine ABI: (const PxgBroadPhaseDesc* bpDesc, const bool isIncremental).
// The bool is byte 8 of the 16-byte CUDA parameter blob (bytes 9..15 are
// zero-filled padding), so the masked word read is exact.
struct EndPtHistParams
{
    uint64_t bpDesc;        // @0
    uint isIncrementalWord; // @8: bool in byte 0
    uint pad0;              // @12 (blob rounds to 16)
}
[[vk::push_constant]]
ConstantBuffer<EndPtHistParams> gParams;

groupshared uint gsLane[BLOCK];           // per-lane packed scan scratch (pass 1)
groupshared uint gsWarp[WARPS_PER_BLOCK]; // packed warp totals (pass 1)
groupshared uint gsAccumStart;            // running block totals across tiles (pass 1)
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

// mirrors the .cu's firstPassEndPtsHistogram device helper.  All arrays arrive
// as raw device addresses from the bpDesc mirror; endPtsHistogram2A /
// blocksEndPtsAccum2A may be 0 (the .cu's NULL second-pass arguments) — the
// total-end outputs are skipped exactly like the .cu's pointer checks.
void firstPassEndPtsHistogram(uint64_t boxHandlesA, uint numHandles,
                              uint64_t blocksEndPtsAccumA, uint64_t blocksStartPtsAccumA,
                              uint64_t endPtsHistogramA, uint64_t startPtsHistogramA,
                              uint64_t endPtsHistogram2A, uint64_t blocksEndPtsAccum2A,
                              uint tid, uint blockIdx)
{
    Ptr<uint> boxHandles = Ptr<uint>(boxHandlesA);
    Ptr<uint> endPtsHistogram = Ptr<uint>(endPtsHistogramA);
    Ptr<uint> startPtsHistogram = Ptr<uint>(startPtsHistogramA);

    const uint warpIndex = tid / WARP;
    const uint lane = tid & (WARP - 1);

    const uint numProjections = numHandles * 2;
    const uint totalBlockRequired = (numProjections + BLOCK - 1) / BLOCK;
    const uint numIterationPerBlock = (totalBlockRequired + NUM_BLOCKS - 1) / NUM_BLOCKS;

    if (tid == WARPS_PER_BLOCK - 1)
    {
        gsAccumEnd = 0;
        gsAccumEnd2 = 0;
        gsAccumStart = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint it = 0; it < numIterationPerBlock; ++it)
    {
        const uint workIndex = it * BLOCK + tid + numIterationPerBlock * blockIdx * BLOCK;

        uint start = 0, end = 0, end2 = 0;
        if (workIndex < numProjections)
        {
            const uint sortedHandle = boxHandles[workIndex];
            end2 = 1u - (bpIsStartProjection(sortedHandle) ? 1u : 0u);
            if (!bpIsDeletedProjection(sortedHandle))
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

        if (workIndex < numProjections)
        {
            const uint accumulation = endPtAccum + gsWarp[warpIndex];
            const uint endCount = ((accumulation >> 10) & FIELD_MASK) + prevEnd;
            const uint startCount = (accumulation >> 20) + prevStart;
            if (endPtsHistogram2A != 0)
                (Ptr<uint>(endPtsHistogram2A))[workIndex] = (accumulation & FIELD_MASK) + prevEnd2;
            endPtsHistogram[workIndex] = endCount;
            startPtsHistogram[workIndex] = startCount;
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == WARPS_PER_BLOCK - 1)
    {
        if (blocksEndPtsAccum2A != 0)
            (Ptr<uint>(blocksEndPtsAccum2A))[blockIdx] = gsAccumEnd2;
        (Ptr<uint>(blocksEndPtsAccumA))[blockIdx] = gsAccumEnd;
        (Ptr<uint>(blocksStartPtsAccumA))[blockIdx] = gsAccumStart;
    }
    GroupMemoryBarrierWithGroupSync(); // separates back-to-back frame passes
}

// mirrors the .cu's secondPassEndPtsHistogram device helper.  The .cu's three
// concurrent 32-lane warp scans of the per-block totals (warp 2 gated on a
// non-NULL total-end pointer) are serialized in thread 0; the tiled
// promote/scatter loop is verbatim.  totalEndPtsHistogramA /
// blockTotalEndPtHistogramA may be 0 (the .cu's NULL second-pass arguments).
void secondPassEndPtsHistogram(uint64_t blockEndPtHistogramA, uint64_t endPtsHistogramA,
                               uint64_t orderedEndPtsHandlesA, uint64_t orderedHandlesA,
                               uint64_t blockStartPtHistogramA, uint64_t startPtsHistogramA,
                               uint64_t orderedStartPtsHandlesA, uint numHandles,
                               uint64_t totalEndPtsHistogramA, uint64_t blockTotalEndPtHistogramA,
                               uint tid, uint blockIdx)
{
    Ptr<uint> blockEndPtHistogram = Ptr<uint>(blockEndPtHistogramA);
    Ptr<uint> blockStartPtHistogram = Ptr<uint>(blockStartPtHistogramA);
    Ptr<uint> endPtsHistogram = Ptr<uint>(endPtsHistogramA);
    Ptr<uint> startPtsHistogram = Ptr<uint>(startPtsHistogramA);
    Ptr<uint> orderedHandles = Ptr<uint>(orderedHandlesA);
    Ptr<uint> orderedEndPtsHandles = Ptr<uint>(orderedEndPtsHandlesA);
    Ptr<uint> orderedStartPtsHandles = Ptr<uint>(orderedStartPtsHandlesA);
    const bool hasTotal = totalEndPtsHistogramA != 0;

    // exclusive prefix over the 32 per-block totals
    if (tid == 0)
    {
        uint accE = 0, accS = 0, accT = 0;
        for (uint b = 0; b < NUM_BLOCKS; ++b)
        {
            gsBlockEnd[b] = accE;
            accE += blockEndPtHistogram[b];
            gsBlockStart[b] = accS;
            accS += blockStartPtHistogram[b];
            if (hasTotal)
            {
                gsBlockTotalEnd[b] = accT;
                accT += (Ptr<uint>(blockTotalEndPtHistogramA))[b];
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    const uint endBase = gsBlockEnd[blockIdx];
    const uint startBase = gsBlockStart[blockIdx];
    const uint totalEndBase = hasTotal ? gsBlockTotalEnd[blockIdx] : 0u;

    const uint numProjections = numHandles * 2;
    const uint totalBlockRequired = (numProjections + BLOCK - 1) / BLOCK;
    const uint numIterationPerBlock = (totalBlockRequired + NUM_BLOCKS - 1) / NUM_BLOCKS;

    for (uint it = 0; it < numIterationPerBlock; ++it)
    {
        const uint workIndex = it * BLOCK + tid + numIterationPerBlock * blockIdx * BLOCK;
        if (workIndex < numProjections)
        {
            const uint sortedHandle = orderedHandles[workIndex];
            const bool startProjection = bpIsStartProjection(sortedHandle);
            const bool isDeleted = bpIsDeletedProjection(sortedHandle);

            // promote BOTH class histograms in place (the .cu writes
            // sPtsHistograms[startProjection] and [1-startProjection])
            const uint endGlobal = endPtsHistogram[workIndex] + endBase;
            const uint startGlobal = startPtsHistogram[workIndex] + startBase;
            endPtsHistogram[workIndex] = endGlobal;
            startPtsHistogram[workIndex] = startGlobal;

            if (!isDeleted)
            {
                if (startProjection)
                    orderedStartPtsHandles[startGlobal] = bpGetHandle(sortedHandle);
                else
                    orderedEndPtsHandles[endGlobal] = bpGetHandle(sortedHandle);
            }

            if (hasTotal)
            {
                Ptr<uint> totalEndPtsHistogram = Ptr<uint>(totalEndPtsHistogramA);
                totalEndPtsHistogram[workIndex] = totalEndPtsHistogram[workIndex] + totalEndBase;
            }
        }
    }
    GroupMemoryBarrierWithGroupSync(); // separates back-to-back frame passes
}

// ---------------------------------------------------------------------------
// pass 1: per-block segmented prefixes + block totals
// (BP_COMPUTE_ENDPT_HISTOGRAM, engine grid (32, 3, 1), blockIdx.y = axis)
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void computeEndPtsHistogram(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const uint axis = gid.y;

    Ptr<PxgBroadPhaseDescGpu> desc = Ptr<PxgBroadPhaseDescGpu>(gParams.bpDesc);
    const bool isIncremental = (gParams.isIncrementalWord & 0xffu) != 0u;

    // endPtHistogram[2][3] etc. are flattened [frame*3 + axis] in the mirror:
    // frame 0 = [axis], frame 1 = [3 + axis].
    if (isIncremental)
    {
        // incremental sap won't generate pairs for the newly inserted pairs, so
        // insertion is deferred: scan only the previous handles.
        const uint numHandles = desc[0].numPreviousHandles;

        firstPassEndPtsHistogram(desc[0].boxHandles[axis], numHandles,
                                 desc[0].blockEndPtHistogram[axis], desc[0].blockStartPtHistogram[axis],
                                 desc[0].endPtHistogram[axis], desc[0].startPtHistogram[axis],
                                 desc[0].totalEndPtHistogram[axis], desc[0].blockTotalEndPtHistogram[axis],
                                 tid, blockIdx);

        // we need to re-scan the end point for removal (total-end args NULL)
        if (desc[0].numRemovedHandles != 0u)
        {
            firstPassEndPtsHistogram(desc[0].boxHandles[3 + axis], desc[0].numPreviousHandles,
                                     desc[0].blockEndPtHistogram[3 + axis], desc[0].blockStartPtHistogram[3 + axis],
                                     desc[0].endPtHistogram[3 + axis], desc[0].startPtHistogram[3 + axis],
                                     0, 0, tid, blockIdx);
        }
    }
    else
    {
        const uint numHandles = desc[0].numPreviousHandles + desc[0].numCreatedHandles - desc[0].numRemovedHandles;

        firstPassEndPtsHistogram(desc[0].boxHandles[axis], numHandles,
                                 desc[0].blockEndPtHistogram[axis], desc[0].blockStartPtHistogram[axis],
                                 desc[0].endPtHistogram[axis], desc[0].startPtHistogram[axis],
                                 desc[0].totalEndPtHistogram[axis], desc[0].blockTotalEndPtHistogram[axis],
                                 tid, blockIdx);
    }
}

// ---------------------------------------------------------------------------
// pass 2: promote block prefixes to device-wide prefixes + scatter handles
// (BP_OUTPUT_ENDPT_HISTOGRAM, engine grid (32, 3, 1), blockIdx.y = axis)
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void outputEndPtsHistogram(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const uint axis = gid.y;

    Ptr<PxgBroadPhaseDescGpu> desc = Ptr<PxgBroadPhaseDescGpu>(gParams.bpDesc);
    const bool isIncremental = (gParams.isIncrementalWord & 0xffu) != 0u;

    if (isIncremental)
    {
        const uint numHandles = desc[0].numPreviousHandles;

        secondPassEndPtsHistogram(desc[0].blockEndPtHistogram[axis], desc[0].endPtHistogram[axis],
                                  desc[0].endPointHandles[axis], desc[0].boxHandles[axis],
                                  desc[0].blockStartPtHistogram[axis], desc[0].startPtHistogram[axis],
                                  desc[0].startPointHandles[axis], numHandles,
                                  desc[0].totalEndPtHistogram[axis], desc[0].blockTotalEndPtHistogram[axis],
                                  tid, blockIdx);

        // we need to re-scan the end point for removal (total-end args NULL)
        if (desc[0].numRemovedHandles != 0u)
        {
            secondPassEndPtsHistogram(desc[0].blockEndPtHistogram[3 + axis], desc[0].endPtHistogram[3 + axis],
                                      desc[0].endPointHandles[3 + axis], desc[0].boxHandles[3 + axis],
                                      desc[0].blockStartPtHistogram[3 + axis], desc[0].startPtHistogram[3 + axis],
                                      desc[0].startPointHandles[3 + axis], desc[0].numPreviousHandles,
                                      0, 0, tid, blockIdx);
        }
    }
    else
    {
        const uint numHandles = desc[0].numPreviousHandles + desc[0].numCreatedHandles - desc[0].numRemovedHandles;

        secondPassEndPtsHistogram(desc[0].blockEndPtHistogram[axis], desc[0].endPtHistogram[axis],
                                  desc[0].endPointHandles[axis], desc[0].boxHandles[axis],
                                  desc[0].blockStartPtHistogram[axis], desc[0].startPtHistogram[axis],
                                  desc[0].startPointHandles[axis], numHandles,
                                  desc[0].totalEndPtHistogram[axis], desc[0].blockTotalEndPtHistogram[axis],
                                  tid, blockIdx);
    }
}
