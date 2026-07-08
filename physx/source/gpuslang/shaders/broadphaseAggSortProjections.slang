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

// broadphaseAggSortProjections.slang
//
// Slang port of sortAndUpdateAggregateProjections (aggregate.cu): for each valid
// aggregate (one 32-thread group) build the per-member min/max X projections and
// their start/end handles from updateBoundIndices + newIntegerBounds, sort the
// 2*size projections by value, then compute the per-box SAP-box positions
// (sapBox1D), the start-projection bitmask (startMasks) and the local bound-index
// list — the sorted structure that doSelfCollision and doAggPairCollisions
// consume.
//
// The CUDA kernel sorts with a warp radix sort (radixSortSingleWarp, stable);
// because the output depends only on the resulting stable order, this port sorts
// with an output-identical stable rank sort (each projection's rank = the number
// of projections with a strictly smaller value, or an equal value and a smaller
// original index).  The double-buffer pointer flip is a host-side frame-
// management detail; this port computes the new (current-frame) structure.

static const uint WARP = 32u;
static const uint MAX_PROJ = 512u; // supports aggregates up to 256 member boxes

// gAggMeta per aggregate (stride 8)
static const uint AM_STRIDE = 8u;
static const uint M_VALID = 0u, M_SIZE = 1u, M_UBIBASE = 2u, M_PROJBASE = 3u,
                  M_SAPBASE = 4u, M_MASKBASE = 5u, M_BIBASE = 6u;

struct SortParams
{
    uint nbAggregates;
}

[[vk::push_constant]]
ConstantBuffer<SortParams> gParams;

RWStructuredBuffer<uint> gNewBounds;          // binding 0  new integer AABB [n][6]
RWStructuredBuffer<uint> gUpdateBoundIndices; // binding 1  per-aggregate member handles (pool)
RWStructuredBuffer<uint> gAggMeta;            // binding 2  [nbAgg][8]
RWStructuredBuffer<uint> gProjOut;            // binding 3  sorted projections (pool)
RWStructuredBuffer<uint> gHandleOut;          // binding 4  sorted handles (pool)
RWStructuredBuffer<uint> gSapOut;             // binding 5  sapBox1D [box][2] (pool)
RWStructuredBuffer<uint> gMaskOut;            // binding 6  start masks (pool)
RWStructuredBuffer<uint> gBiOut;              // binding 7  local bound indices (pool)

uint getHandle(uint sortedHandle) { return sortedHandle >> 3; }
bool isStartProjection(uint sortedHandle) { return (sortedHandle & 1u) != 0u; }
// createHandle(id, isStart, false)
uint createHandle(uint id, bool isStart) { return (id << 3) | (isStart ? 1u : 0u); }

groupshared uint gsProj[MAX_PROJ];
groupshared uint gsHandle[MAX_PROJ];
groupshared uint gsBallot;

// ---------------------------------------------------------------------------
// sortAndUpdateAggregateProjections: one 32-thread group per aggregate
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(32, 1, 1)]
void sortAndUpdateAggregateProjections(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    const uint aggIdx = gid.x;
    const uint lane = gtid.x;
    if (aggIdx >= gParams.nbAggregates)
        return;

    const uint base = aggIdx * AM_STRIDE;
    if (gAggMeta[base + M_VALID] == 0u)
        return;

    const uint numBounds = gAggMeta[base + M_SIZE];
    const uint nbProj = numBounds * 2u;
    const uint ubiBase = gAggMeta[base + M_UBIBASE];
    const uint projBase = gAggMeta[base + M_PROJBASE];
    const uint sapBase = gAggMeta[base + M_SAPBASE];
    const uint maskBase = gAggMeta[base + M_MASKBASE];
    const uint biBase = gAggMeta[base + M_BIBASE];

    // build the unsorted projections/handles into groupshared
    for (uint i = lane; i < nbProj; i += WARP)
    {
        const uint id = i / 2u;
        const uint handle = gUpdateBoundIndices[ubiBase + id];
        const uint off = (i & 1u) * 3u; // even -> minX (0), odd -> maxX (3)
        gsProj[i] = gNewBounds[handle * 6u + off];
        gsHandle[i] = createHandle(id, (i & 1u) == 0u); // even = start
        if ((i & 1u) == 0u)
            gBiOut[biBase + id] = handle;
    }
    GroupMemoryBarrierWithGroupSync();

    // stable sort by projection value (ties broken by original index) via ranks
    for (uint e = lane; e < nbProj; e += WARP)
    {
        const uint v = gsProj[e];
        uint rank = 0u;
        for (uint f = 0u; f < nbProj; ++f)
        {
            const uint vf = gsProj[f];
            if (vf < v || (vf == v && f < e))
                ++rank;
        }
        gProjOut[projBase + rank] = v;
        gHandleOut[projBase + rank] = gsHandle[e];
    }
    GroupMemoryBarrierWithGroupSync();

    // per-box SAP positions + start-projection bitmask
    for (uint b = 0u; b < nbProj; b += WARP)
    {
        const uint idx = b + lane;
        bool isStart = false;
        if (idx < nbProj)
        {
            const uint sortedHandle = gHandleOut[projBase + idx];
            const uint box = getHandle(sortedHandle);
            isStart = isStartProjection(sortedHandle);
            gSapOut[sapBase + box * 2u + (isStart ? 0u : 1u)] = idx;
        }

        if (lane == 0u) gsBallot = 0u;
        GroupMemoryBarrierWithGroupSync();
        if (isStart) InterlockedOr(gsBallot, 1u << lane);
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0u) gMaskOut[maskBase + b / 32u] = gsBallot;
        GroupMemoryBarrierWithGroupSync();
    }
}
