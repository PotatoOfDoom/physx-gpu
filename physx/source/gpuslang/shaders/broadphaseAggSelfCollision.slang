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

// broadphaseAggSelfCollision.slang
//
// Slang port of doSelfCollision (gpubroadphase/src/CUDA/aggregate.cu): for each
// aggregate that is valid and has self-collision enabled (one 32-thread group),
// count for every member box how many other members' start projections fall
// strictly within its X span (from sapBox1D + startMasks), prefix-sum those
// counts to per-box offsets and a grand total, then generate the found pairs
// (members that now overlap but did not last frame) and — for non-new
// aggregates — the lost pairs (members that overlapped last frame but no longer
// do) on the previous frame's structure.  Finally record the total as the next
// frame's previous-comparison count.
//
// Structures for both frames (sapBox1D, startMasks, sortedHandles, boundIndices,
// comparisons) live in shared pools indexed per aggregate and per frame.

static const uint WARP = 32u;
static const uint PX_INVALID_U32 = 0xffffffffu;

// gAggMeta per aggregate (stride 16)
static const uint AM_STRIDE = 16u;
static const uint M_VALID = 0u, M_FILTER = 1u, M_ISNEW = 2u, M_SIZE = 3u, M_PREVSIZE = 4u, M_PREVCOMP = 5u;
static const uint M_SAPC = 6u, M_MASKC = 7u, M_HANDC = 8u, M_BIC = 9u, M_COMPC = 10u;
static const uint M_SAPP = 11u, M_MASKP = 12u, M_HANDP = 13u, M_BIP = 14u, M_COMPP = 15u;

struct SelfParams
{
    uint nbAggregates;
    uint maxFoundLostPairs;
}

[[vk::push_constant]]
ConstantBuffer<SelfParams> gParams;

RWStructuredBuffer<uint> gNewBounds;   // binding 0  new integer AABB [n][6]
RWStructuredBuffer<uint> gOldBounds;   // binding 1  old integer AABB [n][6]
RWStructuredBuffer<uint> gGroupIds;    // binding 2  group id per bound handle
RWStructuredBuffer<uint> gAggMeta;     // binding 3  [nbAgg][16]
RWStructuredBuffer<uint> gSapPool;     // binding 4  sapBox1D flat [box][2]
RWStructuredBuffer<uint> gMaskPool;    // binding 5  start masks
RWStructuredBuffer<uint> gHandlePool;  // binding 6  sorted handles
RWStructuredBuffer<uint> gBiPool;      // binding 7  box -> bound handle
RWStructuredBuffer<uint> gCompPool;    // binding 8  per-box comparison arrays (curr scratch, prev input)
RWStructuredBuffer<uint> gFoundReport; // binding 9  found pairs [maxFL][2]
RWStructuredBuffer<uint> gLostReport;  // binding 10 lost pairs [maxFL][2]
RWStructuredBuffer<uint> gSharedIndex; // binding 11 [0]=found,[1]=lost
RWStructuredBuffer<uint> gPrevCompOut; // binding 12 output prevComparisons per aggregate

uint getHandle(uint sortedHandle) { return sortedHandle >> 3; }

bool aabbIntersects(uint a, uint b, bool useNew)
{
    const uint ab = a * 6u, bb = b * 6u;
    uint aMinX, aMinY, aMinZ, aMaxX, aMaxY, aMaxZ, bMinX, bMinY, bMinZ, bMaxX, bMaxY, bMaxZ;
    if (useNew)
    {
        aMinX = gNewBounds[ab + 0u]; aMinY = gNewBounds[ab + 1u]; aMinZ = gNewBounds[ab + 2u];
        aMaxX = gNewBounds[ab + 3u]; aMaxY = gNewBounds[ab + 4u]; aMaxZ = gNewBounds[ab + 5u];
        bMinX = gNewBounds[bb + 0u]; bMinY = gNewBounds[bb + 1u]; bMinZ = gNewBounds[bb + 2u];
        bMaxX = gNewBounds[bb + 3u]; bMaxY = gNewBounds[bb + 4u]; bMaxZ = gNewBounds[bb + 5u];
    }
    else
    {
        aMinX = gOldBounds[ab + 0u]; aMinY = gOldBounds[ab + 1u]; aMinZ = gOldBounds[ab + 2u];
        aMaxX = gOldBounds[ab + 3u]; aMaxY = gOldBounds[ab + 4u]; aMaxZ = gOldBounds[ab + 5u];
        bMinX = gOldBounds[bb + 0u]; bMinY = gOldBounds[bb + 1u]; bMinZ = gOldBounds[bb + 2u];
        bMaxX = gOldBounds[bb + 3u]; bMaxY = gOldBounds[bb + 4u]; bMaxZ = gOldBounds[bb + 5u];
    }
    return !(bMinX > aMaxX || aMinX > bMaxX || bMinY > aMaxY || aMinY > bMaxY || bMinZ > aMaxZ || aMinZ > bMaxZ);
}

uint getIthBit(uint mask, uint i)
{
    uint lower = i, upper = 32u;
    const uint bitTarget = i + 1u;
    while ((upper - lower) > 1u)
    {
        const uint midPoint = (lower + upper) / 2u;
        const uint nbBits = countbits(mask & ((2u << midPoint) - 1u));
        const uint active = mask & (1u << midPoint);
        if (nbBits > bitTarget || (active == 0u && nbBits == bitTarget)) upper = midPoint;
        else lower = midPoint;
    }
    return lower;
}

// getIndex over the start-mask sub-array at maskBase
uint getIndex(uint maskBase, uint startIdx, uint targetIndex, uint baseIndex)
{
    const uint startWord = startIdx / WARP;
    const uint mask = ~((1u << (startIdx & 31u)) - 1u);
    uint count = 0u;
    const uint target = targetIndex - baseIndex;
    uint w = startWord;
    uint maskedResult = gMaskPool[maskBase + w] & mask;
    for (;; ++w)
    {
        const uint newCount = count + countbits(maskedResult);
        if (newCount > target) break;
        count = newCount;
        maskedResult = gMaskPool[maskBase + w + 1u];
    }
    return w * WARP + getIthBit(maskedResult, target - count);
}

// largest box in [0,nbBounds) with comparisons[box] <= value, else 0
uint binarySearchComp(uint compBase, uint nbBounds, uint value)
{
    uint left = 0u, right = nbBounds;
    while (left < right)
    {
        const uint pos = (left + right) / 2u;
        if (gCompPool[compBase + pos] <= value) left = pos + 1u; else right = pos;
    }
    return left == 0u ? 0u : left - 1u;
}

groupshared uint gsScan[WARP];
groupshared uint gsTotal;
groupshared uint gsOutIndex;

void warpScan32(uint lane)
{
    for (uint off = 1u; off < WARP; off <<= 1)
    {
        const uint add = (lane >= off) ? gsScan[lane - off] : 0u;
        GroupMemoryBarrierWithGroupSync();
        gsScan[lane] += add;
        GroupMemoryBarrierWithGroupSync();
    }
}

// one generatePairs test for comparison index i on the given frame
bool generatePairs(uint aggBase, uint frame, uint i, uint nbBounds, bool useNew, out uint pairA, out uint pairB)
{
    pairA = 0u; pairB = 0u;
    const uint sapBase = frame == 1u ? gAggMeta[aggBase + M_SAPP] : gAggMeta[aggBase + M_SAPC];
    const uint maskBase = frame == 1u ? gAggMeta[aggBase + M_MASKP] : gAggMeta[aggBase + M_MASKC];
    const uint handBase = frame == 1u ? gAggMeta[aggBase + M_HANDP] : gAggMeta[aggBase + M_HANDC];
    const uint biBase = frame == 1u ? gAggMeta[aggBase + M_BIP] : gAggMeta[aggBase + M_BIC];
    const uint compBase = frame == 1u ? gAggMeta[aggBase + M_COMPP] : gAggMeta[aggBase + M_COMPC];

    const uint idx = binarySearchComp(compBase, nbBounds, i);
    const uint startIdx = gSapPool[sapBase + idx * 2u + 0u];
    const uint index = getIndex(maskBase, startIdx + 1u, i, gCompPool[compBase + idx]);
    const uint otherSortedId = getHandle(gHandlePool[handBase + index]);
    const uint boundsIdx = gBiPool[biBase + idx];
    const uint otherBoundsIdx = gBiPool[biBase + otherSortedId];

    if (gGroupIds[otherBoundsIdx] != gGroupIds[boundsIdx])
    {
        if (aabbIntersects(otherBoundsIdx, boundsIdx, useNew))
        {
            if (!aabbIntersects(otherBoundsIdx, boundsIdx, !useNew))
            {
                pairA = min(boundsIdx, otherBoundsIdx);
                pairB = max(boundsIdx, otherBoundsIdx);
                return true;
            }
        }
    }
    return false;
}

void generateAllPairs(uint lane, uint aggBase, uint frame, uint totalComparisons, uint nbBounds, bool useNew,
                      bool lostReport, uint maxPairs)
{
    for (uint base = 0u; base < totalComparisons; base += WARP)
    {
        const uint idx = base + lane;
        bool gen = false;
        uint pairA = 0u, pairB = 0u;
        if (idx < totalComparisons)
            gen = generatePairs(aggBase, frame, idx, nbBounds, useNew, pairA, pairB);

        gsScan[lane] = gen ? 1u : 0u;
        GroupMemoryBarrierWithGroupSync();
        warpScan32(lane);
        const uint nbPairs = gsScan[WARP - 1u];
        const uint myOffset = gsScan[lane] - (gen ? 1u : 0u);
        if (lane == 0u && nbPairs > 0u)
        {
            uint orig;
            InterlockedAdd(gSharedIndex[lostReport ? 1u : 0u], nbPairs, orig);
            gsOutIndex = orig;
        }
        GroupMemoryBarrierWithGroupSync();

        if (gen)
        {
            const uint offset = myOffset + gsOutIndex;
            if (offset < maxPairs)
            {
                if (lostReport) { gLostReport[offset * 2u + 0u] = pairA; gLostReport[offset * 2u + 1u] = pairB; }
                else { gFoundReport[offset * 2u + 0u] = pairA; gFoundReport[offset * 2u + 1u] = pairB; }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

// ---------------------------------------------------------------------------
// doSelfCollision: one 32-thread group per aggregate
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(32, 1, 1)]
void doSelfCollision(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    const uint aggIdx = gid.x;
    const uint lane = gtid.x;
    if (aggIdx >= gParams.nbAggregates)
        return;

    const uint gAggBase = aggIdx * AM_STRIDE;
    if (lane == 0u)
        gsTotal = 0u;
    GroupMemoryBarrierWithGroupSync();

    const uint valid = gAggMeta[gAggBase + M_VALID];
    const uint selfBit = gAggMeta[gAggBase + M_FILTER] & 1u;

    if (valid != 0u && selfBit != 0u)
    {
        const uint numBounds = gAggMeta[gAggBase + M_SIZE];
        const uint sapBaseC = gAggMeta[gAggBase + M_SAPC];
        const uint maskBaseC = gAggMeta[gAggBase + M_MASKC];
        const uint compBaseC = gAggMeta[gAggBase + M_COMPC];
        const uint maxPairs = gParams.maxFoundLostPairs;

        // per-box comparison count (start projections strictly within the span)
        for (uint i = lane; i < numBounds; i += WARP)
        {
            const uint startIdx = gSapPool[sapBaseC + i * 2u + 0u];
            const uint endIdx = gSapPool[sapBaseC + i * 2u + 1u];
            const uint startWord = startIdx / WARP;
            const uint endWord = endIdx / WARP;
            uint mask = ~((2u << (startIdx & 31u)) - 1u);
            uint count = 0u;
            for (uint w = startWord; w < endWord; ++w)
            {
                count += countbits(gMaskPool[maskBaseC + w] & mask);
                mask = 0xffffffffu;
            }
            mask &= ((2u << (endIdx & 31u)) - 1u);
            count += countbits(gMaskPool[maskBaseC + endWord] & mask);
            gCompPool[compBaseC + i] = count;
        }
        GroupMemoryBarrierWithGroupSync();

        // prefix sum the counts into per-box offsets, accumulating the total
        for (uint base = 0u; base < numBounds; base += WARP)
        {
            const uint idx = base + lane;
            const uint count = (idx < numBounds) ? gCompPool[compBaseC + idx] : 0u;
            gsScan[lane] = count;
            GroupMemoryBarrierWithGroupSync();
            warpScan32(lane);
            const uint inclusiveSum = gsScan[lane];
            const uint total = gsScan[WARP - 1u];
            if (idx < numBounds)
                gCompPool[compBaseC + idx] = inclusiveSum - count + gsTotal;
            GroupMemoryBarrierWithGroupSync();
            if (lane == 0u) gsTotal += total;
            GroupMemoryBarrierWithGroupSync();
        }

        const uint totalCount = gsTotal;

        // lost pairs (previous frame's structure)
        if (gAggMeta[gAggBase + M_ISNEW] == 0u)
        {
            generateAllPairs(lane, gAggBase, 1u, gAggMeta[gAggBase + M_PREVCOMP], gAggMeta[gAggBase + M_PREVSIZE],
                             false, true, maxPairs);
        }
        // found pairs (current frame's structure)
        generateAllPairs(lane, gAggBase, 0u, totalCount, numBounds, true, false, maxPairs);
    }

    if (lane == 0u)
        gPrevCompOut[aggIdx] = gsTotal;
}
