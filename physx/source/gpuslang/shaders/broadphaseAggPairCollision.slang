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

// broadphaseAggPairCollision.slang
//
// Slang port of doAggPairCollisions (gpubroadphase/src/CUDA/aggregate.cu): for
// each aggregate pair, skip dead pairs; mark the pair removed if either of its
// bounds was deleted this frame; otherwise decide whether it should still be
// removed (its aggregate-level bounds no longer intersect), run the bipartite
// SAP sweep (broadphaseAggregateBoxPruning) between the two sides' member boxes
// to emit found pairs (and, for non-new pairs, lost pairs on the previous
// frame's structure), and clear the pair's new flag.
//
// Each side is either an aggregate (member boxes with a precomputed sorted
// X-projection SAP structure, stored in shared pools indexed per aggregate and
// per frame) or a single actor (a synthesized 1-box structure, built into
// groupshared scratch from its own integer bounds).  One 32-thread group runs
// one pair; the sweep's warp intrinsics are groupshared 32-lane scans.

static const uint WARP = 32u;
static const uint PX_INVALID_U32 = 0xffffffffu;

// PxgAggregatePair (flat 4 uints)
static const uint AP_H0 = 0u, AP_H1 = 1u, AP_ISNEW = 2u, AP_ISDEAD = 3u, AP_STRIDE = 4u;
// gAggMeta per aggregate: [projBaseCurr, maskBaseCurr, biBaseCurr, sizeCurr,
//                          projBasePrev, maskBasePrev, biBasePrev, sizePrev]
static const uint AM_STRIDE = 8u;

struct AggPairParams
{
    uint nbAggregatePairs;
    uint maxFoundLostPairs;
}

[[vk::push_constant]]
ConstantBuffer<AggPairParams> gParams;

RWStructuredBuffer<uint> gNewBounds;     // binding 0  new integer AABB [n][6]
RWStructuredBuffer<uint> gOldBounds;     // binding 1  old integer AABB [n][6]
RWStructuredBuffer<uint> gGroupIds;      // binding 2  group id per bound handle
RWStructuredBuffer<uint> gVolumeData;    // binding 3  mAggregate encoding per handle
RWStructuredBuffer<uint> gRemovedMap;    // binding 4  removed-handle bitmap
RWStructuredBuffer<uint> gAggPairs;      // binding 5  aggPairs [nbPairs][4]
RWStructuredBuffer<uint> gAggBitmap;     // binding 6  per-pair remove bitmap [nbPairs]
RWStructuredBuffer<uint> gAggMeta;       // binding 7  [nbAgg][8]
RWStructuredBuffer<uint> gProjPool;      // binding 8  concatenated sorted projections
RWStructuredBuffer<uint> gHandlePool;    // binding 9  concatenated sorted handles
RWStructuredBuffer<uint> gMaskPool;      // binding 10 concatenated start masks
RWStructuredBuffer<uint> gBiPool;        // binding 11 concatenated bound-index lists
RWStructuredBuffer<uint> gFoundReport;   // binding 12 found pairs [maxFL][2]
RWStructuredBuffer<uint> gLostReport;    // binding 13 lost pairs [maxFL][2]
RWStructuredBuffer<uint> gSharedIndex;   // binding 14 [0]=found count, [1]=lost count

// a side of a pair: an aggregate's pool slice or a synthesized actor (slot 0/1)
struct SideRef
{
    bool isAgg;
    uint projBase;
    uint handleBase;
    uint maskBase;
    uint biBase;
    uint nbObj;
    uint slot; // groupshared actor scratch slot when !isAgg
}

uint getHandle(uint sortedHandle) { return sortedHandle >> 3; }

uint testBitmap(uint handle) { return (gRemovedMap[handle / 32u] & (1u << (handle & 31u))) != 0u ? 1u : 0u; }

// integer-AABB intersects over new (useNew) or old bounds
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

// groupshared actor scratch: [slot][frame][pos]; boundIndex per slot
groupshared uint gsActorProj[2][2][2];
groupshared uint gsActorBi[2];

// projection-list value / mask / handle-objId / boundIndex accessors
uint sProj(SideRef s, uint frame, uint k) { return s.isAgg ? gProjPool[s.projBase + k] : gsActorProj[s.slot][frame][k]; }
uint sMask(SideRef s, uint w) { return s.isAgg ? gMaskPool[s.maskBase + w] : (w == 0u ? 1u : 0u); }
uint sObjId(SideRef s, uint index) { return s.isAgg ? getHandle(gHandlePool[s.handleBase + index]) : 0u; }
uint sBoundIdx(SideRef s, uint objId) { return s.isAgg ? gBiPool[s.biBase + objId] : gsActorBi[s.slot]; }

uint bnd(uint handle, uint i, bool useNew) { return useNew ? gNewBounds[handle * 6u + i] : gOldBounds[handle * 6u + i]; }

// ---- SAP helpers (see broadphaseAggregateBoxPruning.slang) ----
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

uint getIndex(SideRef s, uint startIdx, uint targetIndex, uint baseIndex)
{
    const uint startWord = startIdx / WARP;
    const uint mask = ~((1u << (startIdx & 31u)) - 1u);
    uint count = 0u;
    const uint target = targetIndex - baseIndex;
    uint w = startWord;
    uint maskedResult = sMask(s, w) & mask;
    for (;; ++w)
    {
        const uint newCount = count + countbits(maskedResult);
        if (newCount > target) break;
        count = newCount;
        maskedResult = sMask(s, w + 1u);
    }
    return w * WARP + getIthBit(maskedResult, target - count);
}

uint bSearchFirstProjection(SideRef s, uint frame, uint numElements, uint proj)
{
    uint left = 0u, right = numElements;
    while (left < right)
    {
        const uint pos = (left + right) / 2u;
        if (sProj(s, frame, pos) < proj) left = pos + 1u; else right = pos;
    }
    return left;
}

uint binarySearchProj(SideRef s, uint frame, uint numElements, uint value)
{
    uint left = 0u, right = numElements;
    while (left < right)
    {
        const uint pos = (left + right) / 2u;
        if (sProj(s, frame, pos) <= value) left = pos + 1u; else right = pos;
    }
    return left == 0u ? 0u : left - 1u;
}

uint countComparisons(SideRef s, uint frame, uint objMinX, uint objMaxX, uint nbProjections,
                      out uint outStartIdx, out uint outEndIdx)
{
    const uint startIdx = bSearchFirstProjection(s, frame, nbProjections, objMinX);
    const uint endIdx = binarySearchProj(s, frame, nbProjections, objMaxX);
    const uint startWordId = startIdx / 32u, endWordId = endIdx / 32u;
    uint w = sMask(s, startWordId) & (~((1u << (startIdx & 31u)) - 1u));
    uint total = 0u, i = startWordId;
    for (; i < endWordId; ) { total += countbits(w); ++i; w = sMask(s, i); }
    w &= ((2u << (endIdx & 31u)) - 1u);
    total += countbits(w);
    outStartIdx = startIdx; outEndIdx = endIdx;
    return total;
}

groupshared uint gsScan[WARP];
groupshared uint shComparisons[WARP];
groupshared uint shStartIdx[WARP];
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

uint searchComparisons(uint value)
{
    uint left = 0u, right = WARP;
    while (left < right)
    {
        const uint pos = (left + right) / 2u;
        if (shComparisons[pos] <= value) left = pos + 1u; else right = pos;
    }
    return left == 0u ? 0u : left - 1u;
}

// one boxPruningProjection pass: project objSide's objects onto projSide's
// projections; useNew selects which bounds are "current" (found) vs "old".
void boxPruningProjection(uint lane, SideRef objSide, SideRef projSide, uint frame,
                          bool isFirst, bool useNew, bool lostReport, uint maxPairs)
{
    const uint nbObj = objSide.nbObj;
    const uint nbProjections = projSide.nbObj * 2u;

    for (uint i = 0u; i < nbObj; i += WARP)
    {
        const uint objId = i + lane;
        uint startIdx = 0u, endIdx = 0u, count = 0u;
        if (objId < nbObj)
        {
            const uint handle = sBoundIdx(objSide, objId);
            count = countComparisons(projSide, frame, bnd(handle, 0u, useNew), bnd(handle, 3u, useNew),
                                     nbProjections, startIdx, endIdx);
        }

        gsScan[lane] = count;
        GroupMemoryBarrierWithGroupSync();
        warpScan32(lane);
        const uint totalCount = gsScan[WARP - 1u];
        shComparisons[lane] = gsScan[lane] - count;
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
                const uint handle = sBoundIdx(objSide, i + idx);
                const uint index = getIndex(projSide, shStartIdx[idx], cmpId, shComparisons[idx]);
                const uint otherId = sObjId(projSide, index);
                const uint otherHandle = sBoundIdx(projSide, otherId);

                if (gGroupIds[handle] != gGroupIds[otherHandle])
                {
                    if (aabbIntersects(handle, otherHandle, useNew))
                    {
                        if (!aabbIntersects(handle, otherHandle, !useNew))
                        {
                            if (isFirst || bnd(handle, 0u, useNew) != bnd(otherHandle, 0u, useNew))
                            {
                                hasContact = true;
                                pairA = min(handle, otherHandle);
                                pairB = max(handle, otherHandle);
                            }
                        }
                    }
                }
            }

            gsScan[lane] = hasContact ? 1u : 0u;
            GroupMemoryBarrierWithGroupSync();
            warpScan32(lane);
            const uint nbPairs = gsScan[WARP - 1u];
            const uint myOffset = gsScan[lane] - (hasContact ? 1u : 0u);
            if (lane == 0u && nbPairs > 0u)
            {
                uint orig;
                InterlockedAdd(gSharedIndex[lostReport ? 1u : 0u], nbPairs, orig);
                gsOutIndex = orig;
            }
            GroupMemoryBarrierWithGroupSync();

            if (hasContact)
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
}

// project list0 objects onto list1, then list1 objects onto list0
void bipartiteBoxPruning(uint lane, SideRef s0, SideRef s1, uint frame, bool useNew, bool lostReport, uint maxPairs)
{
    boxPruningProjection(lane, s0, s1, frame, true, useNew, lostReport, maxPairs);
    GroupMemoryBarrierWithGroupSync();
    boxPruningProjection(lane, s1, s0, frame, false, useNew, lostReport, maxPairs);
}

// build a SideRef for one side of the pair; synthesize actor scratch if needed
SideRef makeSide(uint lane, uint actorHandle, uint slot, uint frame)
{
    SideRef s;
    s.slot = slot;
    const uint vd = gVolumeData[actorHandle];
    const bool isAgg = (vd != PX_INVALID_U32) && ((vd & 1u) != 0u);
    s.isAgg = isAgg;
    if (isAgg)
    {
        const uint agg = vd >> 1u;
        const uint m = agg * AM_STRIDE + (frame == 1u ? 4u : 0u);
        s.projBase = gAggMeta[m + 0u];
        s.maskBase = gAggMeta[m + 1u];
        s.biBase = gAggMeta[m + 2u];
        s.handleBase = s.projBase; // handles share the projection ordering/base
        s.nbObj = gAggMeta[m + 3u];
    }
    else
    {
        // synthesize a single-box structure for both frames from this actor's bounds
        if (lane < 2u)
        {
            gsActorProj[slot][0][lane] = gNewBounds[actorHandle * 6u + lane * 3u]; // [0]=minX,[1]=maxX
            gsActorProj[slot][1][lane] = gOldBounds[actorHandle * 6u + lane * 3u];
        }
        if (lane == 0u) gsActorBi[slot] = actorHandle;
        s.projBase = 0u; s.maskBase = 0u; s.biBase = 0u; s.handleBase = 0u;
        s.nbObj = 1u;
    }
    return s;
}

// ---------------------------------------------------------------------------
// doAggPairCollisions: one 32-thread group per aggregate pair
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(32, 1, 1)]
void doAggPairCollisions(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    const uint i = gid.x;
    const uint lane = gtid.x;
    if (i >= gParams.nbAggregatePairs)
        return;

    const uint isDead = gAggPairs[i * AP_STRIDE + AP_ISDEAD];
    const uint isNew = gAggPairs[i * AP_STRIDE + AP_ISNEW];
    const uint h0 = gAggPairs[i * AP_STRIDE + AP_H0];
    const uint h1 = gAggPairs[i * AP_STRIDE + AP_H1];

    uint shouldRemove = 0u;

    if (isDead == 0u)
    {
        const bool removed = (testBitmap(h0) != 0u) || (testBitmap(h1) != 0u);
        if (removed)
        {
            if (lane == 0u) gAggBitmap[i] = 1u;
            return;
        }

        shouldRemove = aabbIntersects(h0, h1, true) ? 0u : 1u;

        const uint maxPairs = gParams.maxFoundLostPairs;
        SideRef s0c = makeSide(lane, h0, 0u, 0u);
        SideRef s1c = makeSide(lane, h1, 1u, 0u);
        GroupMemoryBarrierWithGroupSync(); // actor scratch visible

        bipartiteBoxPruning(lane, s0c, s1c, 0u, true, false, maxPairs); // found (curr frame)

        if (isNew == 0u)
        {
            SideRef s0p = makeSide(lane, h0, 0u, 1u);
            SideRef s1p = makeSide(lane, h1, 1u, 1u);
            GroupMemoryBarrierWithGroupSync();
            bipartiteBoxPruning(lane, s0p, s1p, 1u, false, true, maxPairs); // lost (prev frame)
        }

        if (lane == 0u) gAggPairs[i * AP_STRIDE + AP_ISNEW] = 0u;
    }

    if (lane == 0u) gAggBitmap[i] = shouldRemove;
}
