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

// broadphaseAggregateDirty.slang
//
// Slang port of the dirty-aggregate double-buffering kernels
// (gpubroadphase/src/CUDA/aggregate.cu):
//   - updateDirtyAggregate: for each dirty (CPU-updated) aggregate, copy its
//     new member-bound indices into the aggregate's bound-index storage, then
//     merge the staging aggregate into the persistent one — a full struct copy
//     if the persistent aggregate is brand new, or a selective field update
//     (preserving prevSize as the previous frame's size) otherwise.
//   - clearDirtyAggregates: at the end of the step, clear the new flag and fix
//     prevSize to the current size for every dirty aggregate.
//
// PxgAggregate is modelled as a flat AGG_STRIDE-uint record (the scalar tail
// plus a base-offset that stands in for the updateBoundIndices device pointer,
// and two representative "other" fields that the full copy transfers but the
// selective update must leave untouched).  The CUDA warp-per-aggregate layout
// maps to one 32-thread group per dirty aggregate.

static const uint AGG_STRIDE = 8u;
// field offsets within the flat aggregate record
static const uint A_BOUNDBASE = 0u; // updateBoundIndices pointer -> pool base offset
static const uint A_PREVSIZE  = 1u;
static const uint A_MINDEX    = 2u;
static const uint A_SIZE      = 3u;
static const uint A_FILTER    = 4u;
static const uint A_MENV      = 5u; // mEnvID          (full copy only)
static const uint A_PREVCMP   = 6u; // prevComparisons (full copy only)
static const uint A_ISNEW     = 7u;

struct DirtyParams
{
    uint numDirtyAggregates;
}

[[vk::push_constant]]
ConstantBuffer<DirtyParams> gParams;

RWStructuredBuffer<uint> gAgg;               // binding 0  persistent aggregates [nbAgg][AGG_STRIDE]
RWStructuredBuffer<uint> gDirtyAgg;          // binding 1  staging aggregates [numDirty][AGG_STRIDE]
RWStructuredBuffer<uint> gDirtyAggIndices;   // binding 2  dirtyAggregateIndices [numDirty]
RWStructuredBuffer<uint> gDirtyBoundIndices; // binding 3  source new-bound-index pool
RWStructuredBuffer<uint> gDirtyBoundStart;   // binding 4  dirtyBoundStartIndices [numDirty]
RWStructuredBuffer<uint> gDirtyAggBoundPool; // binding 5  destination updateBoundIndices storage

// ---------------------------------------------------------------------------
// updateDirtyAggregate
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(32, 1, 1)]
void updateDirtyAggregate(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    const uint warp = gid.x;   // one dirty aggregate per group
    const uint lane = gtid.x;

    if (warp < gParams.numDirtyAggregates)
    {
        const uint index = gDirtyAggIndices[warp];
        const uint aggBase = index * AGG_STRIDE;
        const uint dAggBase = warp * AGG_STRIDE;

        const uint dBoundBase = gDirtyAgg[dAggBase + A_BOUNDBASE];
        const uint startIndex = gDirtyBoundStart[warp];
        const uint size = gDirtyAgg[dAggBase + A_SIZE];

        // copy the aggregate's new member-bound indices into its storage
        for (uint i = lane; i < size; i += 32u)
            gDirtyAggBoundPool[dBoundBase + i] = gDirtyBoundIndices[startIndex + i];

        // ensure every lane has read the (uniform) isNew flag before the full
        // copy below can overwrite it
        const uint isNew = gAgg[aggBase + A_ISNEW];
        GroupMemoryBarrierWithGroupSync();

        if (isNew != 0u)
        {
            // full struct copy: staging -> persistent (warpCopy of sizeof(PxgAggregate))
            for (uint j = lane; j < AGG_STRIDE; j += 32u)
                gAgg[aggBase + j] = gDirtyAgg[dAggBase + j];
        }
        else if (lane == 0u)
        {
            // selective update — preserve the double-buffered prevSize (old size)
            gAgg[aggBase + A_PREVSIZE]  = gAgg[aggBase + A_SIZE];
            gAgg[aggBase + A_BOUNDBASE] = gDirtyAgg[dAggBase + A_BOUNDBASE];
            gAgg[aggBase + A_MINDEX]    = gDirtyAgg[dAggBase + A_MINDEX];
            gAgg[aggBase + A_SIZE]      = gDirtyAgg[dAggBase + A_SIZE];
            gAgg[aggBase + A_FILTER]    = gDirtyAgg[dAggBase + A_FILTER];
        }
    }
}

// ---------------------------------------------------------------------------
// clearDirtyAggregates
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(32, 1, 1)]
void clearDirtyAggregates(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    const uint warp = gid.x;   // one dirty aggregate per group
    const uint lane = gtid.x;

    if (warp < gParams.numDirtyAggregates)
    {
        const uint index = gDirtyAggIndices[warp];
        const uint aggBase = index * AGG_STRIDE;

        if (lane == 0u)
        {
            gAgg[aggBase + A_ISNEW] = 0u;
            gAgg[aggBase + A_PREVSIZE] = gAgg[aggBase + A_SIZE];
        }
    }
}
