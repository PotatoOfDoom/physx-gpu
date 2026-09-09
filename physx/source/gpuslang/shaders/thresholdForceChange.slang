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

// thresholdForceChange.slang
//
// Slang port of the cross-frame force-change event pipeline
// (gpusolver/src/CUDA/accumulateThresholdStream.cu, from setThresholdElementsMask
// onward) — the tail that thresholdStream.slang deferred.  Once the current and
// previous frame's exceeded-force elements are known, this pipeline emits the
// found / lost / persistent force-change events by diffing the two sorted sets:
//
//   1. setThresholdElementsMask: for each current exceeded element, binary-search
//      the previous set (sorted by (nodeIndexA,nodeIndexB)) and, if a full match
//      (nodes + shapeInteraction) is found, clear that pair's previous- and
//      current-mask entries -> it is a *persistent* pair, not a force-change.
//   2. setPersistentForceElementMask: persistentMask[i] = !previousMask[i]
//      (a persistent pair, whose previous mask was cleared above, is written to
//       the force-change stream with its accumulated force).
//   3. computeThresholdElementMaskIndices: exclusive scan of the 0/1 mask array
//      (previous | current | persistent regions) into per-element write indices,
//      copying the original mask into gWriteable; per-block totals to gMaskBetweenBlocks.
//   4. outputThresholdPairsMaskIndices: add the cross-block exclusive prefix.
//   5. createForceChangeThresholdElements: compact each flagged element into
//      gForceChangeElements[writeIndex] and record the total count.
//
// Mask array layout (thresholdStreamWriteIndex), total = nbPrev*2 + nbExceeded:
//   [0, nbPrev)                          previous-exceeded mask (1 = lost pair)
//   [nbPrev, nbPrev+nbExceeded)          current-exceeded  mask (1 = found pair)
//   [nbPrev+nbExceeded, +nbPrev)         persistent        mask (1 = persistent)
//
// Restructuring: the CUDA setPersistentForceElementMask() runs as a device
// helper at the head of computeThresholdElementMaskIndices, then the same launch
// scans a persistent region that other blocks are concurrently writing — a
// cross-block read-after-write the original tolerates.  Here it is split into its
// own kernel launch so the write is globally ordered before the scan reads it;
// the emitted values (persistentMask = !previousMask) are identical.
//
// Portability: the CUDA warp shuffles (warpScan / __ballot_sync+__popc) become
// groupshared 32-lane Hillis-Steele scans; the block-partitioned device grid and
// its between-block accumulation are preserved (BLOCK = 256 threads, 8 warps).
//
// P9.64 -- LIKE thresholdStream.slang, THIS FILE IS LAUNCHED WITH TWO GRIDS.
// Found by tools/check_fixture_grid.py, the twelfth checker, on its first run:
// PxgSolverKernelIndices.h gives
//
//   grid 64:  SET_THRESHOLDELEMENT_MASK          -> setThresholdElementsMask
//             CREATE_FORCECHANGE_THRESHOLDELEMENTS -> createForceChangeThresholdElements
//   grid 32:  COMPUTE_/OUTPUT_THRESHOLDELEMENT_MASK_INDICES
//
// and `setPersistentForceElementMask` belongs to the grid-32 side: the CUDA has it
// as a __device__ helper called from computeThresholdElementMaskIndices, whose own
// body asserts `PxgKernelGridDim::COMPUTE_THRESHOLDELEMENT_MASK_INDICES == 32`.
// It is a separate entry here only because the original scans a persistent region
// other blocks are concurrently writing (see above), so the split is about
// ordering, not about the grid.
//
// The two grid-64 entries strided by 256*32 against 256*64 threads, so every index
// at or above the stride was visited twice.  Both bodies are pure gathers into a
// buffer other than their source, so this was output-equivalent and no coverage
// hole -- the same shape, and the same reason nothing failed, as in
// thresholdStream.slang.  Corrected for the same reason: the equivalence depends
// on those bodies staying idempotent and nothing said so.

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

struct ForceChangeParams
{
    uint nbExceeded;  // nbExceededThresholdElements  (current frame)
    uint nbPrev;      // nbPrevExceededThresholdElements (previous frame)
}

[[vk::push_constant]]
ConstantBuffer<ForceChangeParams> gParams;

RWStructuredBuffer<ThresholdStreamElement> gExceeded;       // binding 0  current exceeded
RWStructuredBuffer<ThresholdStreamElement> gPrev;           // binding 1  previous exceeded (sorted)
RWStructuredBuffer<ThresholdStreamElement> gForceChange;    // binding 2  output events
RWStructuredBuffer<uint> gMask;                             // binding 3  thresholdStreamWriteIndex
RWStructuredBuffer<uint> gMaskBetweenBlocks;               // binding 4
RWStructuredBuffer<uint> gWriteable;                       // binding 5  thresholdStreamWriteable (bool)
RWStructuredBuffer<uint> gNbForceChange;                   // binding 6  single counter

// PxNodeIndex is a 64-bit value (x = index/low word, y = linkID/high word);
// its ordering is the numeric order of the 64-bit word.
bool node_lt(uint2 a, uint2 b) { return a.y < b.y || (a.y == b.y && a.x < b.x); }
bool node_le(uint2 a, uint2 b) { return a.y < b.y || (a.y == b.y && a.x <= b.x); }
bool node_eq(uint2 a, uint2 b) { return a.x == b.x && a.y == b.y; }

// Dy::ThresholdStreamElement::operator<= (compares nodeIndexA then nodeIndexB)
bool elem_le(ThresholdStreamElement a, ThresholdStreamElement b)
{
    return node_lt(a.nodeIndexA, b.nodeIndexA) ||
           (node_eq(a.nodeIndexA, b.nodeIndexA) && node_le(a.nodeIndexB, b.nodeIndexB));
}

// mirrors gpucommon reduction.cuh binarySearch(): largest index s.t.
// data[index] <= value, or 0 if data[0] > value
uint binarySearchPrev(uint numElements, ThresholdStreamElement value)
{
    uint left = 0u;
    uint right = numElements;
    while (left < right)
    {
        const uint pos = (left + right) / 2u;
        if (elem_le(gPrev[pos], value))
            left = pos + 1u;
        else
            right = pos;
    }
    return left != 0u ? left - 1u : 0u;
}

// ---------------------------------------------------------------------------
// 1) setThresholdElementsMask
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void setThresholdElementsMask(uint3 dtid : SV_DispatchThreadID)
{
    const uint nbExceeded = gParams.nbExceeded;
    const uint nbPrev = gParams.nbPrev;
    const uint stride = BLOCK * NUM_BLOCKS_WIDE;

    for (uint workIndex = dtid.x; workIndex < nbExceeded; workIndex += stride)
    {
        ThresholdStreamElement element = gExceeded[workIndex];

        // last previous element whose (nodeIndexA,nodeIndexB) <= element
        uint pos = binarySearchPrev(nbPrev, element);

        bool done = false;
        while (!done)
        {
            done = true;
            ThresholdStreamElement prePair = gPrev[pos];
            if (node_eq(prePair.nodeIndexA, element.nodeIndexA) &&
                node_eq(prePair.nodeIndexB, element.nodeIndexB))
            {
                if (prePair.shapeInteraction.x == element.shapeInteraction.x &&
                    prePair.shapeInteraction.y == element.shapeInteraction.y)
                {
                    // persistent pair: clear both its previous- and current-mask entries
                    gMask[pos] = 0u;
                    gMask[nbPrev + workIndex] = 0u;
                }
                else if (pos > 1u)
                {
                    pos = pos - 1u;
                    done = false;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 2) setPersistentForceElementMask (own launch — see file header)
//    persistentMask[i] = !previousMask[i]
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void setPersistentForceElementMask(uint3 dtid : SV_DispatchThreadID)
{
    const uint nbExceeded = gParams.nbExceeded;
    const uint nbPrev = gParams.nbPrev;
    const uint stride = BLOCK * NUM_BLOCKS;
    const uint persistentStart = nbPrev + nbExceeded;

    for (uint workIndex = dtid.x; workIndex < nbPrev; workIndex += stride)
        gMask[persistentStart + workIndex] = (gMask[workIndex] != 0u) ? 0u : 1u;
}

// ---------------------------------------------------------------------------
// exclusive-scan scaffolding (pairs-only, uint)
// ---------------------------------------------------------------------------
groupshared uint gsPairs[BLOCK];
groupshared uint gsWarpPairs[WARPS_PER_BLOCK];
groupshared uint gsBlockPairs;
groupshared uint gsBlockWriteAccum[NUM_BLOCKS];

// inclusive scan of gsPairs within each 32-lane warp (all warps in parallel)
void warpScanPairs(uint tid)
{
    for (uint off = 1; off < WARP; off <<= 1)
    {
        const uint lane = tid & (WARP - 1);
        const uint add = (lane >= off) ? gsPairs[tid - off] : 0u;
        GroupMemoryBarrierWithGroupSync();
        gsPairs[tid] += add;
        GroupMemoryBarrierWithGroupSync();
    }
}

// exclusive scan over the WARPS_PER_BLOCK warp totals; folds the grand total
// into gsBlockPairs.  All barriers are in uniform control flow.
void scanWarpTotalsPairs(uint tid)
{
    const bool active = tid < WARPS_PER_BLOCK;
    const uint p0 = active ? gsWarpPairs[tid] : 0u;
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

// ---------------------------------------------------------------------------
// 3) computeThresholdElementMaskIndices: per-block exclusive scan of the mask
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void computeThresholdElementMaskIndices(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const uint nbExceeded = gParams.nbExceeded;
    const uint nbPrev = gParams.nbPrev;
    const uint total = nbExceeded + nbPrev * 2u;
    const uint warpIndex = tid / WARP;

    const uint totalBlockRequired = (total + (BLOCK - 1)) / BLOCK;
    const uint numIterationPerBlock = (totalBlockRequired + (NUM_BLOCKS - 1)) / NUM_BLOCKS;

    if (tid == 0)
        gsBlockPairs = 0u;
    GroupMemoryBarrierWithGroupSync();

    for (uint it = 0; it < numIterationPerBlock; ++it)
    {
        const uint workIndex = tid + it * BLOCK + BLOCK * blockIdx * numIterationPerBlock;

        uint forceChangeMask = 0u;
        if (workIndex < total)
            forceChangeMask = gMask[workIndex];

        gsPairs[tid] = forceChangeMask;
        GroupMemoryBarrierWithGroupSync();
        warpScanPairs(tid);
        const uint accumPairs = gsPairs[tid] - forceChangeMask; // exclusive within warp

        if ((tid & (WARP - 1)) == WARP - 1)
            gsWarpPairs[warpIndex] = accumPairs + forceChangeMask;

        const uint prevBlockPairs = gsBlockPairs;
        GroupMemoryBarrierWithGroupSync();

        scanWarpTotalsPairs(tid);

        GroupMemoryBarrierWithGroupSync();

        if (workIndex < total)
        {
            gMask[workIndex] = accumPairs + prevBlockPairs + gsWarpPairs[warpIndex];
            gWriteable[workIndex] = (forceChangeMask != 0u) ? 1u : 0u;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0)
        gMaskBetweenBlocks[blockIdx] = gsBlockPairs;
}

// ---------------------------------------------------------------------------
// 4) outputThresholdPairsMaskIndices: add cross-block exclusive prefix
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void outputThresholdPairsMaskIndices(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const uint nbExceeded = gParams.nbExceeded;
    const uint nbPrev = gParams.nbPrev;
    const uint total = nbExceeded + nbPrev * 2u;

    if (tid < NUM_BLOCKS)
        gsBlockWriteAccum[tid] = gMaskBetweenBlocks[tid];
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

    const uint totalBlockRequired = (total + (BLOCK - 1)) / BLOCK;
    const uint numIterationPerBlock = (totalBlockRequired + (NUM_BLOCKS - 1)) / NUM_BLOCKS;
    const uint blockWriteAccum = gsBlockWriteAccum[blockIdx];

    for (uint it = 0; it < numIterationPerBlock; ++it)
    {
        const uint workIndex = it * BLOCK + tid + numIterationPerBlock * blockIdx * BLOCK;
        if (workIndex < total)
            gMask[workIndex] = gMask[workIndex] + blockWriteAccum;
    }
}

// ---------------------------------------------------------------------------
// 5) createForceChangeThresholdElements: compaction
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void createForceChangeThresholdElements(uint3 dtid : SV_DispatchThreadID)
{
    const uint nbExceeded = gParams.nbExceeded;
    const uint nbPrev = gParams.nbPrev;
    const uint total = nbExceeded + nbPrev * 2u;
    const uint persistentStart = nbPrev + nbExceeded;
    const uint stride = BLOCK * NUM_BLOCKS_WIDE;

    for (uint workIndex = dtid.x; workIndex < total; workIndex += stride)
    {
        const bool hasForceChangeOrPersistent = gWriteable[workIndex] != 0u;
        const uint writeIndex = gMask[workIndex];

        if (hasForceChangeOrPersistent)
        {
            const bool lostPair = workIndex < nbPrev;
            const bool foundPair = (workIndex < persistentStart) && !lostPair;

            ThresholdStreamElement pair;
            if (lostPair)
                pair = gPrev[workIndex];
            else if (foundPair)
                pair = gExceeded[workIndex - nbPrev];
            else
                pair = gPrev[workIndex - persistentStart];

            ThresholdStreamElement tempPair;
            tempPair.shapeInteraction = pair.shapeInteraction;
            tempPair.nodeIndexA = pair.nodeIndexA;
            tempPair.nodeIndexB = pair.nodeIndexB;
            tempPair.normalForce = pair.normalForce;
            tempPair.accumulatedForce = lostPair ? 0.0f : pair.accumulatedForce;
            tempPair.threshold = pair.threshold;
            tempPair.pad = 0u;

            gForceChange[writeIndex] = tempPair;
        }

        if (workIndex == total - 1u)
            gNbForceChange[0] = hasForceChangeOrPersistent ? (writeIndex + 1u) : writeIndex;
    }
}
