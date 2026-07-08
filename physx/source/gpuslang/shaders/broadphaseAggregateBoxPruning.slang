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

// broadphaseAggregateBoxPruning.slang
//
// Slang port of the aggregate collision SAP sweep primitive (aggregate.cu):
// bipartiteBoxPruning (two boxPruningProjection passes) plus its helpers
// (getIthBit / getIndex / countComparisons / bSearchFirstProjection /
// binarySearch1 / PxgIntegerAABB::intersects).  This is the shared inner loop
// of doSelfCollision and doAggPairCollisions: given two lists of boxes and one
// list's sorted X-projection SAP structure, it emits every cross-list pair
// whose new integer bounds intersect but whose old bounds do not (and which
// pass the group filter), deduplicated across the two projection directions.
//
// The CUDA kernel runs one warp per aggregate pair with __ballot_sync/__popc/
// __shfl/warpScan; this port runs one 32-thread group per bipartite sweep and
// replaces the warp intrinsics with groupshared 32-lane scans and InterlockedAdd.

static const uint WARP = 32u;

struct BoxPruneParams
{
    uint nbObj0;
    uint nbObj1;
    uint maxPairs;
}

[[vk::push_constant]]
ConstantBuffer<BoxPruneParams> gParams;

RWStructuredBuffer<uint> gBounds;         // binding 0  new integer AABB [n][6]
RWStructuredBuffer<uint> gOldBounds;      // binding 1  old integer AABB [n][6]
RWStructuredBuffer<uint> gGroupIds;       // binding 2  group id per bound handle
RWStructuredBuffer<uint> gSortedProj0;    // binding 3  list0 sorted X projections [nbObj0*2]
RWStructuredBuffer<uint> gSortedHandles0; // binding 4  list0 sorted handles [nbObj0*2]
RWStructuredBuffer<uint> gStartMask0;     // binding 5  list0 start-projection bitmask
RWStructuredBuffer<uint> gBoundIndices0;  // binding 6  list0 obj -> bound handle [nbObj0]
RWStructuredBuffer<uint> gSortedProj1;    // binding 7  list1 sorted X projections [nbObj1*2]
RWStructuredBuffer<uint> gSortedHandles1; // binding 8  list1 sorted handles [nbObj1*2]
RWStructuredBuffer<uint> gStartMask1;     // binding 9  list1 start-projection bitmask
RWStructuredBuffer<uint> gBoundIndices1;  // binding 10 list1 obj -> bound handle [nbObj1]
RWStructuredBuffer<uint> gPairs;          // binding 11 output pairs [maxPairs][2] {mVolA,mVolB}
RWStructuredBuffer<uint> gSharedIndex;    // binding 12 atomic output counter [1]

// projection-list accessors (proj1 == true selects list1 as the projection list)
uint pProj(bool proj1, uint k)    { return proj1 ? gSortedProj1[k]    : gSortedProj0[k]; }
uint pHandle(bool proj1, uint k)  { return proj1 ? gSortedHandles1[k] : gSortedHandles0[k]; }
uint pMask(bool proj1, uint w)    { return proj1 ? gStartMask1[w]     : gStartMask0[w]; }
uint pBoundIdx(bool proj1, uint k){ return proj1 ? gBoundIndices1[k]  : gBoundIndices0[k]; }
// object-list accessors (the objects being projected)
uint oBoundIdx(bool proj1, uint k){ return proj1 ? gBoundIndices0[k]  : gBoundIndices1[k]; }

uint getHandle(uint sortedHandle) { return sortedHandle >> 3; }

// PxgIntegerAABB::intersects (3D) over a flat {minX,minY,minZ,maxX,maxY,maxZ} store
bool aabbIntersects(uint a, uint b, bool useOld)
{
    const uint ab = a * 6u, bb = b * 6u;
    uint aMinX, aMinY, aMinZ, aMaxX, aMaxY, aMaxZ;
    uint bMinX, bMinY, bMinZ, bMaxX, bMaxY, bMaxZ;
    if (useOld)
    {
        aMinX = gOldBounds[ab + 0u]; aMinY = gOldBounds[ab + 1u]; aMinZ = gOldBounds[ab + 2u];
        aMaxX = gOldBounds[ab + 3u]; aMaxY = gOldBounds[ab + 4u]; aMaxZ = gOldBounds[ab + 5u];
        bMinX = gOldBounds[bb + 0u]; bMinY = gOldBounds[bb + 1u]; bMinZ = gOldBounds[bb + 2u];
        bMaxX = gOldBounds[bb + 3u]; bMaxY = gOldBounds[bb + 4u]; bMaxZ = gOldBounds[bb + 5u];
    }
    else
    {
        aMinX = gBounds[ab + 0u]; aMinY = gBounds[ab + 1u]; aMinZ = gBounds[ab + 2u];
        aMaxX = gBounds[ab + 3u]; aMaxY = gBounds[ab + 4u]; aMaxZ = gBounds[ab + 5u];
        bMinX = gBounds[bb + 0u]; bMinY = gBounds[bb + 1u]; bMinZ = gBounds[bb + 2u];
        bMaxX = gBounds[bb + 3u]; bMaxY = gBounds[bb + 4u]; bMaxZ = gBounds[bb + 5u];
    }
    return !(bMinX > aMaxX || aMinX > bMaxX ||
             bMinY > aMaxY || aMinY > bMaxY ||
             bMinZ > aMaxZ || aMinZ > bMaxZ);
}

// lower_bound: first index with projections[pos] >= proj
uint bSearchFirstProjection(bool proj1, uint numElements, uint proj)
{
    uint left = 0u, right = numElements;
    while (left < right)
    {
        const uint pos = (left + right) / 2u;
        if (pProj(proj1, pos) < proj) left = pos + 1u;
        else right = pos;
    }
    return left;
}

// largest index with data[index] <= value, else 0 (PxU32 binarySearch1)
uint binarySearch1(bool proj1, uint numElements, uint value)
{
    uint left = 0u, right = numElements;
    while (left < right)
    {
        const uint pos = (left + right) / 2u;
        if (pProj(proj1, pos) <= value) left = pos + 1u;
        else right = pos;
    }
    if (left == 0u)
    {
        left = 0u; right = numElements;
        while (left < right)
        {
            const uint pos = (left + right) / 2u;
            if (pProj(proj1, pos) <= value) left = pos + 1u;
            else right = pos;
        }
        return left;
    }
    return left - 1u;
}

// index of the (i)-th set bit of mask (0-based)
uint getIthBit(uint mask, uint i)
{
    uint lower = i, upper = 32u;
    const uint bitTarget = i + 1u;
    while ((upper - lower) > 1u)
    {
        const uint midPoint = (lower + upper) / 2u;
        const uint nbBits = countbits(mask & ((2u << midPoint) - 1u));
        const uint active = mask & (1u << midPoint);
        if (nbBits > bitTarget || (active == 0u && nbBits == bitTarget))
            upper = midPoint;
        else
            lower = midPoint;
    }
    return lower;
}

// resolve the (targetIndex - baseIndex)-th start projection at/after startIdx
uint getIndex(bool proj1, uint startIdx, uint targetIndex, uint baseIndex)
{
    const uint startWord = startIdx / WARP;
    const uint mask = ~((1u << (startIdx & 31u)) - 1u);
    uint count = 0u;
    const uint target = targetIndex - baseIndex;
    uint w = startWord;
    uint maskedResult = pMask(proj1, w) & mask;
    for (;; ++w)
    {
        const uint newCount = count + countbits(maskedResult);
        if (newCount > target) break;
        count = newCount;
        maskedResult = pMask(proj1, w + 1u);
    }
    const uint offset = target - count;
    const uint bit = getIthBit(maskedResult, offset);
    return w * WARP + bit;
}

// count start projections of the projection list within [obj.minX, obj.maxX]
uint countComparisons(bool proj1, uint objMinX, uint objMaxX, uint nbProjections,
                      out uint outStartIdx, out uint outEndIdx)
{
    const uint startIdx = bSearchFirstProjection(proj1, nbProjections, objMinX);
    const uint endIdx = binarySearch1(proj1, nbProjections, objMaxX);
    const uint startWordId = startIdx / 32u;
    const uint endWordId = endIdx / 32u;

    uint w = pMask(proj1, startWordId) & (~((1u << (startIdx & 31u)) - 1u));
    uint total = 0u;
    uint i = startWordId;
    for (; i < endWordId; )
    {
        total += countbits(w);
        ++i;
        w = pMask(proj1, i);
    }
    w &= ((2u << (endIdx & 31u)) - 1u);
    total += countbits(w);

    outStartIdx = startIdx;
    outEndIdx = endIdx;
    return total;
}

groupshared uint gsScan[WARP];         // 32-lane inclusive scan scratch
groupshared uint shComparisons[WARP];  // per-object exclusive comparison prefix
groupshared uint shStartIdx[WARP];     // per-object projection start index
groupshared uint gsOutIndex;           // broadcast of the atomic reservation

// inclusive scan of gsScan over the 32 lanes
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

// largest index in [0,32) with shComparisons[index] <= value (binarySearch1 semantics)
uint searchComparisons(uint value)
{
    uint left = 0u, right = WARP;
    while (left < right)
    {
        const uint pos = (left + right) / 2u;
        if (shComparisons[pos] <= value) left = pos + 1u;
        else right = pos;
    }
    if (left == 0u) return 0u;
    return left - 1u;
}

// one boxPruningProjection pass; proj1 selects which list supplies the projections
void boxPruningProjection(bool proj1, uint lane)
{
    const uint nbObj = proj1 ? gParams.nbObj0 : gParams.nbObj1;
    const uint nbProjections = (proj1 ? gParams.nbObj1 : gParams.nbObj0) * 2u;
    const bool isFirst = proj1;
    const uint maxPairs = gParams.maxPairs;

    for (uint i = 0u; i < nbObj; i += WARP)
    {
        const uint objId = i + lane;

        uint startIdx = 0u, endIdx = 0u, count = 0u;
        if (objId < nbObj)
        {
            const uint handle = oBoundIdx(proj1, objId);
            count = countComparisons(proj1, gBounds[handle * 6u + 0u], gBounds[handle * 6u + 3u],
                                     nbProjections, startIdx, endIdx);
        }

        gsScan[lane] = count;
        GroupMemoryBarrierWithGroupSync();
        warpScan32(lane);
        const uint inclusiveSum = gsScan[lane];
        const uint totalCount = gsScan[WARP - 1u];
        shComparisons[lane] = inclusiveSum - count; // exclusive prefix
        shStartIdx[lane] = startIdx;
        GroupMemoryBarrierWithGroupSync();

        for (uint b = 0u; b < totalCount; b += WARP)
        {
            const uint cmpId = b + lane;

            bool hasContact = false;
            uint pairA = 0u, pairB = 0u;
            if (cmpId < totalCount)
            {
                const uint idx = searchComparisons(cmpId);
                const uint objId2 = i + idx;
                const uint handle = oBoundIdx(proj1, objId2);
                const uint sIdx = shStartIdx[idx];
                const uint index = getIndex(proj1, sIdx, cmpId, shComparisons[idx]);
                const uint otherId = getHandle(pHandle(proj1, index));
                const uint otherHandle = pBoundIdx(proj1, otherId);

                if (gGroupIds[handle] != gGroupIds[otherHandle])
                {
                    if (aabbIntersects(handle, otherHandle, false))
                    {
                        if (!aabbIntersects(handle, otherHandle, true))
                        {
                            if (isFirst || gBounds[handle * 6u + 0u] != gBounds[otherHandle * 6u + 0u])
                            {
                                hasContact = true;
                                pairA = min(handle, otherHandle);
                                pairB = max(handle, otherHandle);
                            }
                        }
                    }
                }
            }

            // compact-emit the contacts found by this 32-lane group
            gsScan[lane] = hasContact ? 1u : 0u;
            GroupMemoryBarrierWithGroupSync();
            warpScan32(lane);
            const uint nbPairs = gsScan[WARP - 1u];
            const uint myOffset = gsScan[lane] - (hasContact ? 1u : 0u);
            if (lane == 0u && nbPairs > 0u)
            {
                uint orig;
                InterlockedAdd(gSharedIndex[0], nbPairs, orig);
                gsOutIndex = orig;
            }
            GroupMemoryBarrierWithGroupSync();

            if (hasContact)
            {
                const uint offset = myOffset + gsOutIndex;
                if (offset < maxPairs)
                {
                    gPairs[offset * 2u + 0u] = pairA;
                    gPairs[offset * 2u + 1u] = pairB;
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
}

// ---------------------------------------------------------------------------
// bipartiteBoxPruning: project list0 onto list1, then list1 onto list0
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(32, 1, 1)]
void aggregateBipartiteBoxPruning(uint3 gtid : SV_GroupThreadID)
{
    const uint lane = gtid.x;
    boxPruningProjection(true, lane);
    GroupMemoryBarrierWithGroupSync();
    boxPruningProjection(false, lane);
}
