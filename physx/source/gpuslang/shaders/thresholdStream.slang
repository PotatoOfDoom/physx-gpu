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

// thresholdStream.slang
//
// Slang port of the force-threshold reporting accumulation pipeline
// (gpusolver/src/CUDA/accumulateThresholdStream.cu).  After the solver
// writeback appends per-contact-patch threshold elements, this pipeline:
//   1. extracts the sort keys (nodeIndexB then nodeIndexA) for a two-pass
//      radix sort (initialRanksAndBodyIndexB / initialRanksAndBodyIndexA) and
//      reorders the stream into (nodeIndexA, nodeIndexB) order
//      (reorganizeThresholdElements);
//   2. runs a device-wide segmented scan that accumulates each body pair's
//      normal force and assigns a per-pair write index
//      (computeAccumulateThresholdStream + outputAccumulateThresholdStream +
//       writeoutAccumulatedForcePerObject);
//   3. detects the pairs whose accumulated force exceeds threshold*dt and
//      compacts them into the exceeded-force element array
//      (computeExceededForceThresholdElementIndice +
//       outputExceededForceThresholdElementIndice).
//
// The cross-frame found/lost/persistent force-change event diff
// (setThresholdElementsMask onward) is left for a later tranche.
//
// Portability: the CUDA warp shuffles (warpScan / __ballot_sync+__popc) become
// groupshared 32-lane Hillis-Steele scans; the block-partitioned device grid and
// its between-block accumulation are preserved (BLOCK = 256 threads, 8 warps).
//
// P9.63 -- THIS FILE IS LAUNCHED WITH TWO DIFFERENT GRIDS AND THE PORT HAD ONE
// CONSTANT.  The CUDA writes its stride as the runtime product
// `blockDim.x * gridDim.x`, which is right at every launch site by construction.
// The bridge drops blockDim, so the port must name the product per launch site --
// and PxgSolverKernelIndices.h gives two distinct grids for the entries here:
//
//   grid 32:  COMPUTE_ACCUMULATED_THRESHOLDSTREAM, OUTPUT_ACCUMULATED_THRESHOLDSTREAM,
//             COMPUTE_EXCEEDEDFORCE_..., OUTPUT_EXCEEDEDFORCE_...   -> NUM_BLOCKS
//   grid 64:  INITIALIZE_INPUT_AND_RANKS, REORGANIZE_THRESHOLDSTREAM,
//             WRITEOUT_ACCUMULATEDFORCEPEROBJECT                    -> NUM_BLOCKS_WIDE
//
// Before this the four grid-64 entries strided by 256*32 while the engine gave
// them 256*64 threads, so every index at or above the stride was visited by TWO
// threads.  That was OUTPUT-EQUIVALENT and not a coverage hole -- all four bodies
// are pure gathers whose destination is a different buffer from their source, so
// the duplicate simply stores the same bytes again -- which is why nothing ever
// failed.  It is corrected anyway: the equivalence rests entirely on the bodies
// staying idempotent, and nothing in the file said so.  Note also that this is NOT
// the case the grid-stride checker can decide (it finds missing loops, not wrong
// constants); it took reading the four launch sites.
//
// The fixture must dispatch the ENGINE's grid per entry, and with more elements
// than one pass covers, or the loop never iterates and the constant is unobserved
// (it dispatched 32 for all eight, with N <= 2001, and saw nothing).

import pxgSlangCommon;

static const uint BLOCK = 256;
static const uint NUM_BLOCKS = 32;       // PxgKernelGridDim, the block-partitioned entries
static const uint NUM_BLOCKS_WIDE = 64;  // PxgKernelGridDim, the plain grid-stride entries
static const uint WARP = 32;
static const uint WARPS_PER_BLOCK = BLOCK / WARP; // 8

// mirrors Dy::ThresholdStreamElement (40 bytes) — words:
//   0,1 shapeInteraction | 2 normalForce | 3 threshold |
//   4,5 nodeIndexA(x,y)  | 6,7 nodeIndexB(x,y) | 8 accumulatedForce | 9 pad
struct ThresholdStreamElement
{
    uint2 shapeInteraction;
    float normalForce;
    float threshold;
    uint2 nodeIndexA;
    uint2 nodeIndexB;
    float accumulatedForce;
    uint pad;
}

struct ThresholdParams
{
    uint nbThresholdElements;
    uint nbPrevExceeded;      // unused here; reserved for the cross-frame diff
    float dt;
    float pad;
}

[[vk::push_constant]]
ConstantBuffer<ThresholdParams> gParams;

RWStructuredBuffer<ThresholdStreamElement> gThresholdStream;          // binding 0
RWStructuredBuffer<uint> gInputKeys;                                  // binding 1
RWStructuredBuffer<uint> gInputRanks;                                 // binding 2
RWStructuredBuffer<uint> gThresholdStreamWords;                       // binding 3 (word view of gThresholdStream)
RWStructuredBuffer<uint> gTmpThresholdStreamWords;                    // binding 4 (word view of the unsorted copy)
RWStructuredBuffer<float> gAccumulatedForce;                          // binding 5
RWStructuredBuffer<float> gAccumulatedForceBetweenBlocks;             // binding 6
RWStructuredBuffer<uint> gWriteIndex;                                 // binding 7
RWStructuredBuffer<uint> gWriteIndexBetweenBlocks;                    // binding 8
RWStructuredBuffer<uint> gWriteable;                                  // binding 9 (bool as uint)
RWStructuredBuffer<float> gAccumulatedForceObjectPairs;              // binding 10
RWStructuredBuffer<ThresholdStreamElement> gExceededForceElements;   // binding 11
RWStructuredBuffer<uint> gNbExceeded;                                // binding 12 (single counter)

bool nodeEq(uint2 a, uint2 b) { return a.x == b.x && a.y == b.y; }

// ---------------------------------------------------------------------------
// sort-key extraction + reorganize
// ---------------------------------------------------------------------------

// mirrors initialRanksAndBodyIndexB(): keys = nodeIndexB.index(), ranks = i,
// padding the tail up to a multiple of 4 with 0xffffffff keys / identity ranks
[shader("compute")]
[numthreads(256, 1, 1)]
void initialRanksAndBodyIndexB(uint3 dtid : SV_DispatchThreadID, uint3 gdim : SV_GroupID)
{
    const uint n = gParams.nbThresholdElements;
    const uint stride = BLOCK * NUM_BLOCKS_WIDE;
    for (uint i = dtid.x; i < n; i += stride)
    {
        gInputKeys[i] = gThresholdStream[i].nodeIndexB.x; // index()
        gInputRanks[i] = i;
    }
    const uint remaining = (4 - (n & 3)) & 3;
    for (uint i = dtid.x; i < remaining; i += stride)
    {
        const uint index = i + n;
        gInputKeys[index] = 0xffffffffu;
        gInputRanks[index] = index;
    }
}

// mirrors initialRanksAndBodyIndexA(): after the B sort, keys[i] =
// thresholdStream[ranks[i]].nodeIndexA.index() (ranks hold the B permutation)
[shader("compute")]
[numthreads(256, 1, 1)]
void initialRanksAndBodyIndexA(uint3 dtid : SV_DispatchThreadID)
{
    const uint n = gParams.nbThresholdElements;
    const uint stride = BLOCK * NUM_BLOCKS_WIDE;
    for (uint i = dtid.x; i < n; i += stride)
        gInputKeys[i] = gThresholdStream[gInputRanks[i]].nodeIndexA.x;
    const uint remaining = (4 - (n & 3)) & 3;
    for (uint i = dtid.x; i < remaining; i += stride)
        gInputKeys[i + n] = 0xffffffffu;
}

// mirrors reorganizeThresholdElements(): gather element ranks[i] of the
// unsorted copy into slot i of the sorted stream, copying words 0..7 (all
// input fields; accumulatedForce/pad are produced later) with 8 lanes/element
[shader("compute")]
[numthreads(256, 1, 1)]
void reorganizeThresholdElements(uint3 dtid : SV_DispatchThreadID)
{
    const uint n = gParams.nbThresholdElements;
    const uint stride = (BLOCK * NUM_BLOCKS_WIDE) / 8;
    const uint lane8 = dtid.x & 7;
    for (uint i = dtid.x / 8; i < n; i += stride)
    {
        const uint rank = gInputRanks[i];
        gThresholdStreamWords[i * 10 + lane8] = gTmpThresholdStreamWords[rank * 10 + lane8];
    }
}

// ---------------------------------------------------------------------------
// device-wide segmented force accumulation
// ---------------------------------------------------------------------------

groupshared float gsForce[BLOCK];       // per-lane force scan scratch
groupshared uint gsPairs[BLOCK];        // per-lane new-pair scan scratch
groupshared float gsWarpForce[WARPS_PER_BLOCK];
groupshared uint gsWarpPairs[WARPS_PER_BLOCK];
groupshared float gsBlockForce;
groupshared uint gsBlockPairs;

// inclusive scan of gsForce within each 32-lane warp (all warps in parallel)
void warpScanForce(uint tid)
{
    for (uint off = 1; off < WARP; off <<= 1)
    {
        const uint lane = tid & (WARP - 1);
        float add = (lane >= off) ? gsForce[tid - off] : 0.0f;
        GroupMemoryBarrierWithGroupSync();
        gsForce[tid] += add;
        GroupMemoryBarrierWithGroupSync();
    }
}

void warpScanPairs(uint tid)
{
    for (uint off = 1; off < WARP; off <<= 1)
    {
        const uint lane = tid & (WARP - 1);
        uint add = (lane >= off) ? gsPairs[tid - off] : 0u;
        GroupMemoryBarrierWithGroupSync();
        gsPairs[tid] += add;
        GroupMemoryBarrierWithGroupSync();
    }
}

// exclusive scan over the WARPS_PER_BLOCK warp totals; folds the grand total
// into gsBlockForce/gsBlockPairs.  All barriers are in uniform control flow
// (every thread of the workgroup reaches each barrier); only lanes < 8 touch
// the groupshared warp-total arrays.
void scanWarpTotals(uint tid)
{
    const bool active = tid < WARPS_PER_BLOCK;
    float f0 = active ? gsWarpForce[tid] : 0.0f;
    uint p0 = active ? gsWarpPairs[tid] : 0u;
    float fInc = f0;
    uint pInc = p0;
    for (uint off = 1; off < WARPS_PER_BLOCK; off <<= 1)
    {
        const bool take = active && tid >= off;
        const float fa = take ? gsWarpForce[tid - off] : 0.0f;
        const uint pa = take ? gsWarpPairs[tid - off] : 0u;
        GroupMemoryBarrierWithGroupSync();
        fInc += fa;
        pInc += pa;
        if (active) { gsWarpForce[tid] = fInc; gsWarpPairs[tid] = pInc; }
        GroupMemoryBarrierWithGroupSync();
    }
    GroupMemoryBarrierWithGroupSync();
    if (active) { gsWarpForce[tid] = fInc - f0; gsWarpPairs[tid] = pInc - p0; }
    if (tid == WARPS_PER_BLOCK - 1) { gsBlockForce += fInc; gsBlockPairs += pInc; }
}

// pairs-only variant (exceeded-force detection); folds into gsBlockPairs
void scanWarpTotalsPairs(uint tid)
{
    const bool active = tid < WARPS_PER_BLOCK;
    uint p0 = active ? gsWarpPairs[tid] : 0u;
    uint pInc = p0;
    for (uint off = 1; off < WARPS_PER_BLOCK; off <<= 1)
    {
        const bool take = active && tid >= off;
        const uint pa = take ? gsWarpPairs[tid - off] : 0u;
        GroupMemoryBarrierWithGroupSync();
        pInc += pa;
        if (active) gsWarpPairs[tid] = pInc;
        GroupMemoryBarrierWithGroupSync();
    }
    GroupMemoryBarrierWithGroupSync();
    if (active) gsWarpPairs[tid] = pInc - p0;
    if (tid == WARPS_PER_BLOCK - 1) gsBlockPairs += pInc;
}

// mirrors computeAccumulateThresholdStream(): one workgroup per device block
[shader("compute")]
[numthreads(256, 1, 1)]
void computeAccumulateThresholdStream(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const uint n = gParams.nbThresholdElements;
    const uint warpIndex = tid / WARP;

    const uint totalBlockRequired = (n + (BLOCK - 1)) / BLOCK;
    const uint numIterationPerBlock = (totalBlockRequired + (NUM_BLOCKS - 1)) / NUM_BLOCKS;

    if (tid == 0)
    {
        gsBlockForce = 0.0f;
        gsBlockPairs = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint it = 0; it < numIterationPerBlock; ++it)
    {
        const uint workIndex = tid + it * BLOCK + BLOCK * blockIdx * numIterationPerBlock;

        float val = 0.0f;
        bool isNewPair = false;
        if (workIndex < n)
        {
            val = gThresholdStream[workIndex].normalForce;
            const uint2 nA = gThresholdStream[workIndex].nodeIndexA;
            const uint2 nB = gThresholdStream[workIndex].nodeIndexB;
            if (workIndex + 1 < n)
            {
                const uint2 mA = gThresholdStream[workIndex + 1].nodeIndexA;
                const uint2 mB = gThresholdStream[workIndex + 1].nodeIndexB;
                if (!(nodeEq(nA, mA) && nodeEq(nB, mB)))
                    isNewPair = true;
            }
            else
            {
                isNewPair = true;
            }
        }

        // 32-lane inclusive scans -> exclusive prefixes within the warp
        gsForce[tid] = val;
        gsPairs[tid] = isNewPair ? 1u : 0u;
        GroupMemoryBarrierWithGroupSync();
        warpScanForce(tid);
        warpScanPairs(tid);
        const float accumVal = gsForce[tid] - val;
        const uint accumPairs = gsPairs[tid] - (isNewPair ? 1u : 0u);

        if ((tid & (WARP - 1)) == WARP - 1)
        {
            gsWarpForce[warpIndex] = accumVal + val;
            gsWarpPairs[warpIndex] = accumPairs + (isNewPair ? 1u : 0u);
        }

        const float prevBlockForce = gsBlockForce;
        const uint prevBlockPairs = gsBlockPairs;

        GroupMemoryBarrierWithGroupSync();

        scanWarpTotals(tid);

        GroupMemoryBarrierWithGroupSync();

        if (workIndex < n)
        {
            gAccumulatedForce[workIndex] = val + accumVal + prevBlockForce + gsWarpForce[warpIndex];
            gWriteIndex[workIndex] = accumPairs + prevBlockPairs + gsWarpPairs[warpIndex];
            gWriteable[workIndex] = isNewPair ? 1u : 0u;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0)
    {
        gAccumulatedForceBetweenBlocks[blockIdx] = gsBlockForce;
        gWriteIndexBetweenBlocks[blockIdx] = gsBlockPairs;
    }
}

groupshared float gsBlockForceAccum[NUM_BLOCKS];
groupshared uint gsBlockWriteAccum[NUM_BLOCKS];

// mirrors outputAccumulateThresholdStream(): add the exclusive prefix of the
// per-block totals to every element of each block's region
[shader("compute")]
[numthreads(256, 1, 1)]
void outputAccumulateThresholdStream(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const uint n = gParams.nbThresholdElements;

    // warp 0 scans the 32 between-block totals into exclusive prefixes
    if (tid < NUM_BLOCKS)
    {
        gsBlockForceAccum[tid] = gAccumulatedForceBetweenBlocks[tid];
        gsBlockWriteAccum[tid] = gWriteIndexBetweenBlocks[tid];
    }
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0)
    {
        float fAcc = 0.0f;
        uint pAcc = 0u;
        for (uint i = 0; i < NUM_BLOCKS; ++i)
        {
            const float f = gsBlockForceAccum[i];
            const uint p = gsBlockWriteAccum[i];
            gsBlockForceAccum[i] = fAcc;
            gsBlockWriteAccum[i] = pAcc;
            fAcc += f;
            pAcc += p;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    const uint totalBlockRequired = (n + (BLOCK - 1)) / BLOCK;
    const uint numIterationPerBlock = (totalBlockRequired + (NUM_BLOCKS - 1)) / NUM_BLOCKS;
    const float blockForceAccum = gsBlockForceAccum[blockIdx];
    const uint blockWriteAccum = gsBlockWriteAccum[blockIdx];

    for (uint it = 0; it < numIterationPerBlock; ++it)
    {
        const uint workIndex = it * BLOCK + tid + numIterationPerBlock * blockIdx * BLOCK;
        if (workIndex < n)
        {
            gWriteIndex[workIndex] = gWriteIndex[workIndex] + blockWriteAccum;
            gAccumulatedForce[workIndex] = gAccumulatedForce[workIndex] + blockForceAccum;
        }
    }
}

// mirrors writeoutAccumulatedForcePerObject(): store each pair's inclusive
// force prefix (at its last element) into the per-pair array
[shader("compute")]
[numthreads(256, 1, 1)]
void writeoutAccumulatedForcePerObject(uint3 dtid : SV_DispatchThreadID)
{
    const uint n = gParams.nbThresholdElements;
    const uint stride = BLOCK * NUM_BLOCKS_WIDE;
    for (uint workIndex = dtid.x; workIndex < n; workIndex += stride)
    {
        if (gWriteable[workIndex] != 0u)
            gAccumulatedForceObjectPairs[gWriteIndex[workIndex]] = gAccumulatedForce[workIndex];
    }
}

// ---------------------------------------------------------------------------
// exceeded-force detection + compaction
// ---------------------------------------------------------------------------

// mirrors computeExceededForceThresholdElementIndice(): recover each pair's
// segment force (prefix difference), store it on the element, mark the pairs
// whose force exceeds threshold*dt and compute their compaction indices
[shader("compute")]
[numthreads(256, 1, 1)]
void computeExceededForceThresholdElementIndice(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const uint n = gParams.nbThresholdElements;
    const float dt = gParams.dt;
    const uint warpIndex = tid / WARP;

    const uint totalBlockRequired = (n + (BLOCK - 1)) / BLOCK;
    const uint numIterationPerBlock = (totalBlockRequired + (NUM_BLOCKS - 1)) / NUM_BLOCKS;

    if (tid == 0)
        gsBlockPairs = 0u;
    GroupMemoryBarrierWithGroupSync();

    for (uint it = 0; it < numIterationPerBlock; ++it)
    {
        const uint workIndex = tid + it * BLOCK + BLOCK * blockIdx * numIterationPerBlock;

        bool isExceededForce = false;
        if (workIndex < n)
        {
            const uint writeIndex = gWriteIndex[workIndex];
            float accumulatedForce = gAccumulatedForceObjectPairs[writeIndex];
            if (writeIndex > 0)
                accumulatedForce -= gAccumulatedForceObjectPairs[writeIndex - 1];
            gThresholdStream[workIndex].accumulatedForce = accumulatedForce;
            isExceededForce = accumulatedForce > (gThresholdStream[workIndex].threshold * dt);
        }

        gsPairs[tid] = isExceededForce ? 1u : 0u;
        GroupMemoryBarrierWithGroupSync();
        warpScanPairs(tid);
        const uint accumPairs = gsPairs[tid] - (isExceededForce ? 1u : 0u);

        if ((tid & (WARP - 1)) == WARP - 1)
            gsWarpPairs[warpIndex] = accumPairs + (isExceededForce ? 1u : 0u);

        const uint prevBlockPairs = gsBlockPairs;
        GroupMemoryBarrierWithGroupSync();

        scanWarpTotalsPairs(tid);

        GroupMemoryBarrierWithGroupSync();

        if (workIndex < n)
        {
            gWriteIndex[workIndex] = accumPairs + prevBlockPairs + gsWarpPairs[warpIndex];
            gWriteable[workIndex] = isExceededForce ? 1u : 0u;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0)
        gWriteIndexBetweenBlocks[blockIdx] = gsBlockPairs;
}

// mirrors outputExceededForceThresholdElementIndice(): compact the exceeded
// elements into gExceededForceElements and record the total count
[shader("compute")]
[numthreads(256, 1, 1)]
void outputExceededForceThresholdElementIndice(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const uint n = gParams.nbThresholdElements;

    if (tid < NUM_BLOCKS)
        gsBlockWriteAccum[tid] = gWriteIndexBetweenBlocks[tid];
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0)
    {
        uint acc = 0u;
        for (uint i = 0; i < NUM_BLOCKS; ++i)
        {
            const uint p = gsBlockWriteAccum[i];
            gsBlockWriteAccum[i] = acc;
            acc += p;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    const uint totalBlockRequired = (n + (BLOCK - 1)) / BLOCK;
    const uint numIterationPerBlock = (totalBlockRequired + (NUM_BLOCKS - 1)) / NUM_BLOCKS;
    const uint blockWriteAccum = gsBlockWriteAccum[blockIdx];

    for (uint it = 0; it < numIterationPerBlock; ++it)
    {
        const uint workIndex = it * BLOCK + tid + numIterationPerBlock * blockIdx * BLOCK;
        if (workIndex < n)
        {
            const uint writeIndex = gWriteIndex[workIndex] + blockWriteAccum;
            if (gWriteable[workIndex] != 0u)
            {
                ThresholdStreamElement e = gThresholdStream[workIndex];
                ThresholdStreamElement outE;
                outE.shapeInteraction = e.shapeInteraction;
                outE.nodeIndexA = e.nodeIndexA;
                outE.nodeIndexB = e.nodeIndexB;
                outE.normalForce = e.normalForce;
                outE.accumulatedForce = e.accumulatedForce;
                outE.threshold = e.threshold;
                outE.pad = 0u;
                gExceededForceElements[writeIndex] = outE;
            }
            if (workIndex == n - 1)
                gNbExceeded[0] = (gWriteable[workIndex] != 0u) ? (writeIndex + 1) : writeIndex;
        }
    }
}
