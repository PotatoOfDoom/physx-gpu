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

// broadphaseAggregateBounds.slang
//
// Slang port of the aggregate bounds/bitmap refresh kernels
// (gpubroadphase/src/CUDA/aggregate.cu):
//   - markAggregateBoundsUpdatedBitmap: for every valid aggregate, mark the
//     aggregate's own AABB-manager handle as changed (atomic OR) and clear the
//     changed bit of each of its member shape bounds (atomic AND ~bit), so the
//     member shapes are not independently re-broadphased.
//   - updateAggregateBounds: for every valid aggregate, recompute the union of
//     its member shapes' contact-distance-inflated bounds and store it back at
//     the aggregate's own bound slot.
//
// The CUDA kernels run one warp per aggregate, iterating the aggregate's
// member-bound pointer list (PxgAggregate::updateBoundIndices).  This port
// runs one 32-thread group per aggregate (so the block-wide barrier used by
// the bounds reduction is warp-wide and the isValid() guard stays uniform),
// and flattens the per-aggregate pointer arrays into shared pool buffers
// indexed by a per-aggregate base offset:
//   updateBoundIndices[i]  ->  gBoundIndices[gAggBoundBase[agg] + i]
//
// PxBounds3 arrays are stored flat as {minX,minY,minZ,maxX,maxY,maxZ} per slot,
// matching translateAABBsLaunch's convention.

static const uint AGG_WARP = 32u;
static const uint BITMAP_WORD_SIZE = 32u;
static const uint PX_INVALID_AGG_INDEX = 0xffffffffu;
// PxBounds3::setEmpty(): minimum = PX_MAX_BOUNDS_EXTENTS, maximum = -that.
// PX_MAX_BOUNDS_EXTENTS = FLT_MAX * 0.25f == asfloat(0x7e7fffff).
static const float PX_MAX_BOUNDS_EXTENTS = asfloat(0x7e7fffffu);

struct AggBoundsParams
{
    uint numAggregates;
}

[[vk::push_constant]]
ConstantBuffer<AggBoundsParams> gParams;

RWStructuredBuffer<uint>  gAggMIndex;    // binding 0  PxgAggregate::mIndex per aggregate
RWStructuredBuffer<uint>  gAggSize;      // binding 1  PxgAggregate::size  per aggregate
RWStructuredBuffer<uint>  gAggBoundBase; // binding 2  flattened base of updateBoundIndices
RWStructuredBuffer<uint>  gBoundIndices; // binding 3  shared member-bound-index pool
RWStructuredBuffer<uint>  gChangedMap;   // binding 4  aabbMngr changed-handle bitmap (atomic)
RWStructuredBuffer<float> gBounds;       // binding 5  PxBounds3[] flat [n][6]
RWStructuredBuffer<float> gContactDist;  // binding 6  contact distance per bound slot

bool aggIsValid(uint size, uint mIndex)
{
    return (size > 0u) && (mIndex != PX_INVALID_AGG_INDEX);
}

// ---------------------------------------------------------------------------
// markAggregateBoundsUpdatedBitmap
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(32, 1, 1)]
void markAggregateBoundsUpdatedBitmap(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    const uint warpIdx = gid.x;   // one aggregate per group
    const uint lane = gtid.x;

    if (warpIdx < gParams.numAggregates)
    {
        const uint size = gAggSize[warpIdx];
        const uint mIndex = gAggMIndex[warpIdx];
        if (aggIsValid(size, mIndex))
        {
            if (lane == 0u)
            {
                const uint bit = 1u << (mIndex % BITMAP_WORD_SIZE);
                InterlockedOr(gChangedMap[mIndex / BITMAP_WORD_SIZE], bit);
            }

            const uint base = gAggBoundBase[warpIdx];
            for (uint id = lane; id < size; id += AGG_WARP)
            {
                const uint idx = gBoundIndices[base + id];
                const uint bit = 1u << (idx % BITMAP_WORD_SIZE);
                InterlockedAnd(gChangedMap[idx / BITMAP_WORD_SIZE], ~bit);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// updateAggregateBounds
// ---------------------------------------------------------------------------
groupshared float sMinX[AGG_WARP];
groupshared float sMinY[AGG_WARP];
groupshared float sMinZ[AGG_WARP];
groupshared float sMaxX[AGG_WARP];
groupshared float sMaxY[AGG_WARP];
groupshared float sMaxZ[AGG_WARP];

[shader("compute")]
[numthreads(32, 1, 1)]
void updateAggregateBounds(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    const uint warpIdx = gid.x;   // one aggregate per group
    const uint lane = gtid.x;

    if (warpIdx < gParams.numAggregates)
    {
        const uint size = gAggSize[warpIdx];
        const uint mIndex = gAggMIndex[warpIdx];
        if (aggIsValid(size, mIndex))   // uniform across the group
        {
            const uint base = gAggBoundBase[warpIdx];

            float mnx = PX_MAX_BOUNDS_EXTENTS, mny = PX_MAX_BOUNDS_EXTENTS, mnz = PX_MAX_BOUNDS_EXTENTS;
            float mxx = -PX_MAX_BOUNDS_EXTENTS, mxy = -PX_MAX_BOUNDS_EXTENTS, mxz = -PX_MAX_BOUNDS_EXTENTS;

            for (uint id = lane; id < size; id += AGG_WARP)
            {
                const uint idx = gBoundIndices[base + id];
                const uint b = idx * 6u;
                const float inflate = gContactDist[idx];
                mnx = min(mnx, gBounds[b + 0u] - inflate);
                mny = min(mny, gBounds[b + 1u] - inflate);
                mnz = min(mnz, gBounds[b + 2u] - inflate);
                mxx = max(mxx, gBounds[b + 3u] + inflate);
                mxy = max(mxy, gBounds[b + 4u] + inflate);
                mxz = max(mxz, gBounds[b + 5u] + inflate);
            }

            sMinX[lane] = mnx; sMinY[lane] = mny; sMinZ[lane] = mnz;
            sMaxX[lane] = mxx; sMaxY[lane] = mxy; sMaxZ[lane] = mxz;
            GroupMemoryBarrierWithGroupSync();

            for (uint s = AGG_WARP / 2u; s > 0u; s >>= 1)
            {
                if (lane < s)
                {
                    sMinX[lane] = min(sMinX[lane], sMinX[lane + s]);
                    sMinY[lane] = min(sMinY[lane], sMinY[lane + s]);
                    sMinZ[lane] = min(sMinZ[lane], sMinZ[lane + s]);
                    sMaxX[lane] = max(sMaxX[lane], sMaxX[lane + s]);
                    sMaxY[lane] = max(sMaxY[lane], sMaxY[lane + s]);
                    sMaxZ[lane] = max(sMaxZ[lane], sMaxZ[lane + s]);
                }
                GroupMemoryBarrierWithGroupSync();
            }

            if (lane == 0u)
            {
                const uint o = mIndex * 6u;
                gBounds[o + 0u] = sMinX[0];
                gBounds[o + 1u] = sMinY[0];
                gBounds[o + 2u] = sMinZ[0];
                gBounds[o + 3u] = sMaxX[0];
                gBounds[o + 4u] = sMaxY[0];
                gBounds[o + 5u] = sMaxZ[0];
            }
        }
    }
}
