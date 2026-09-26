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
                // P9.86: PxVec3::minimum/maximum are PxMin/PxMax selects (acc < v ? acc : v,
                // acc > v ? acc : v: the NEW value on a +-0 tie and on NaN; aggregate.cu:204-205).
                // llvmpipe's stamped FMin/FMax order the zeros and return the other on NaN.
                const float vnx = gBounds[b + 0u] - inflate;
                const float vny = gBounds[b + 1u] - inflate;
                const float vnz = gBounds[b + 2u] - inflate;
                const float vxx = gBounds[b + 3u] + inflate;
                const float vxy = gBounds[b + 4u] + inflate;
                const float vxz = gBounds[b + 5u] + inflate;
                mnx = (mnx < vnx) ? mnx : vnx;
                mny = (mny < vny) ? mny : vny;
                mnz = (mnz < vnz) ? mnz : vnz;
                mxx = (mxx > vxx) ? mxx : vxx;
                mxy = (mxy > vxy) ? mxy : vxy;
                mxz = (mxz > vxz) ? mxz : vxz;
            }

            sMinX[lane] = mnx; sMinY[lane] = mny; sMinZ[lane] = mnz;
            sMaxX[lane] = mxx; sMaxY[lane] = mxy; sMaxZ[lane] = mxz;
            GroupMemoryBarrierWithGroupSync();

            // P9.86: the CUDA reduces with a butterfly `PxMin(__shfl_xor_sync(.., v, r), v)` /
            // `PxMax(..)` (aggregate.cu:209-216): selects that keep the lane's OWN value on a
            // tie, so lane 0's result is the adjacent-pair tree r = 1,2,4,8,16 in which the
            // LOWER half wins a +0/-0 tie (and the NaN operand order is fixed).  GLSL min/max
            // on a lane+s tree picks the other zero (llvmpipe min(+0,-0) = -0), so replay
            // that exact tree with the PxMin/PxMax select spellings.
            for (uint r = 1u; r < AGG_WARP; r <<= 1)
            {
                if ((lane & (2u * r - 1u)) == 0u)
                {
                    const uint p = lane + r;
                    sMinX[lane] = (sMinX[p] < sMinX[lane]) ? sMinX[p] : sMinX[lane];
                    sMinY[lane] = (sMinY[p] < sMinY[lane]) ? sMinY[p] : sMinY[lane];
                    sMinZ[lane] = (sMinZ[p] < sMinZ[lane]) ? sMinZ[p] : sMinZ[lane];
                    sMaxX[lane] = (sMaxX[p] > sMaxX[lane]) ? sMaxX[p] : sMaxX[lane];
                    sMaxY[lane] = (sMaxY[p] > sMaxY[lane]) ? sMaxY[p] : sMaxY[lane];
                    sMaxZ[lane] = (sMaxZ[p] > sMaxZ[lane]) ? sMaxZ[p] : sMaxZ[lane];
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
