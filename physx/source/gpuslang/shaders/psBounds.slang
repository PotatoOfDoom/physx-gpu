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

// psBounds.slang
//
// Slang port of the particle-vs-primitive bound/cell bookkeeping kernels of
// gpunarrowphase/src/CUDA/cudaParticleSystem.cu:
//
// - ps_primitivesBoundFirstPassLaunch: per contact-manager pair, intersect
//   the rigid and particle AABBs (combine = component max of minima / min of
//   maxima), fattenFast by the summed contact distance, count the wrapped
//   grid cells the overlap covers (calcGridRange ->
//   calcWrappedGridRangeSize, product), and produce a segmented EXCLUSIVE
//   scan of those counts: per-warp scan, per-block warp-total scan and a
//   running block accumulator over the block's tiles, with the per-block
//   totals written to blockOffsets.  BLOCK = 512 threads
//   (PxgParticleSystemKernelBlockDim::BOUNDCELLUPDATE), 32 blocks
//   (PxgParticleSystemKernelGridDim::BOUNDCELLUPDATE).
// - ps_primitivesBoundSecondPassLaunch: exclusive scan over the 32 block
//   totals, folded into every offset, grand total to totalNumPairs.
// - ps_reorderPrimitiveContactsLaunch: gather PxgParticlePrimitiveContact
//   by the rigid- and particle-sorted remap tables (radix-sort ranks).  The
//   CUDA iteration-count expression `(total + bd*gd - 1) / bd * gd`
//   over-counts by a factor of gridDim (missing parentheses); it is
//   harmless — the loop exits on index >= total — and is preserved.
// - calcGridHashInBounds + combine (the helpers the collision drivers use
//   to enumerate cells): exposed as a per-thread entry for verification.
//
// The engine-side LoadShapePair / PxgParticleSystem plumbing is driver glue;
// work items here carry the two bounds, the cell width, the summed contact
// distance and the wrapped grid size directly.  Grid helpers are the
// gridCal.cuh ports (see gridCal.slang).

static const uint BLOCK = 512;      // PxgParticleSystemKernelBlockDim::BOUNDCELLUPDATE
static const uint NUM_BLOCKS = 32;  // PxgParticleSystemKernelGridDim::BOUNDCELLUPDATE
static const uint WARP = 32;
static const uint WARPS_PER_BLOCK = BLOCK / WARP; // 16

struct PbParams { uint numTests; uint pad0; uint pad1; uint pad2; }

struct PbWork
{
    float4 b0min; // w = cellWidth
    float4 b0max; // w = cDistance (summed contact distance)
    float4 b1min; // w = cell offset (ctPsGridHash only)
    float4 b1max;
    uint4 grid;   // xyz = wrapped grid size (powers of two)
}

// PxgParticlePrimitiveContact (32 bytes)
struct PsContact
{
    float4 normal_pen;
    uint2 rigidId;    // PxU64
    uint2 particleId; // PxU64
}

[[vk::push_constant]]
ConstantBuffer<PbParams> gParams;

RWStructuredBuffer<PbWork> gWork;          // binding 0
RWStructuredBuffer<uint> gOffsets;         // binding 1 (first/second pass in/out)
RWStructuredBuffer<uint> gBlockOffsets;    // binding 2 [32]
RWStructuredBuffer<uint> gTotal;           // binding 3 [1]
RWStructuredBuffer<PsContact> gContactsIn; // binding 4 (reorder)
RWStructuredBuffer<uint> gRemapA;          // binding 5 (remapByRigid)
RWStructuredBuffer<uint> gRemapB;          // binding 6 (remapByParticle)
RWStructuredBuffer<PsContact> gSortedA;    // binding 7 (sortedByRigid)
RWStructuredBuffer<PsContact> gSortedB;    // binding 8 (sortedByParticle)

groupshared uint gsLane[BLOCK];
groupshared uint gsWarp[WARPS_PER_BLOCK];
groupshared uint gsBlockAccum;
groupshared uint gsBlockPrefix[NUM_BLOCKS];
groupshared uint gsTotalPairs;

// ---- gridCal.cuh helpers (see gridCal.slang) ------------------------------
int3 calcGridPos(float3 p, float cellWidth)
{
    return int3(int(floor(p.x / cellWidth)), int(floor(p.y / cellWidth)), int(floor(p.z / cellWidth)));
}

uint calcGridHash(int3 gridPos, uint3 gridSize)
{
    gridPos.x = gridPos.x & int(gridSize.x - 1);
    gridPos.y = gridPos.y & int(gridSize.y - 1);
    gridPos.z = gridPos.z & int(gridSize.z - 1);
    return uint((gridPos.z * int(gridSize.y)) * int(gridSize.x) + gridPos.y * int(gridSize.x) + gridPos.x);
}

uint3 calcWrappedGridRangeSize(int3 mn, int3 mx, uint3 gridSize)
{
    return uint3(uint(clamp(mx.x - mn.x + 1, 0, int(gridSize.x))),
                 uint(clamp(mx.y - mn.y + 1, 0, int(gridSize.y))),
                 uint(clamp(mx.z - mn.z + 1, 0, int(gridSize.z))));
}

uint3 calcGridOffsetInRange(uint3 rangeSize, uint offset)
{
    return uint3(offset % rangeSize.x,
                 (offset / rangeSize.x) % rangeSize.y,
                 offset / (rangeSize.x * rangeSize.y));
}

void calcGridRange(out int3 gridPosMin, out int3 gridPosMax, float3 bmin, float3 bmax, float cellWidth)
{
    gridPosMin = calcGridPos(bmin, cellWidth);
    gridPosMax = calcGridPos(bmax, cellWidth);
}

// combine: intersection box (component max of minima, min of maxima)
void combineBounds(float3 mn0, float3 mx0, float3 mn1, float3 mx1, out float3 mn, out float3 mx)
{
    mn = max(mn0, mn1);
    mx = min(mx0, mx1);
}

uint calcGridHashInBounds(float3 bmin, float3 bmax, float cellWidth, uint cellOffset, uint3 gridSize)
{
    int3 gridPosMin, gridPosMax;
    calcGridRange(gridPosMin, gridPosMax, bmin, bmax, cellWidth);
    const uint3 rangeSize = calcWrappedGridRangeSize(gridPosMin, gridPosMax, gridSize);
    const uint3 offset = calcGridOffsetInRange(rangeSize, cellOffset);
    const int3 gridPos = int3(gridPosMin.x + int(offset.x), gridPosMin.y + int(offset.y), gridPosMin.z + int(offset.z));
    return calcGridHash(gridPos, gridSize);
}

// ---- scan plumbing (same shapes as broadphaseRegionHistogram.slang) -------
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

void scanWarpTotals(uint tid)
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
    if (tid == WARPS_PER_BLOCK - 1) gsBlockAccum += pInc;
}

// ---------------------------------------------------------------------------
// pass 1
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(512, 1, 1)]
void ctPsBoundFirstPass(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const uint warpIndex = tid / WARP;
    const uint lane = tid & (WARP - 1);

    const uint numTests = gParams.numTests;
    const uint nbBlocksRequired = (numTests + BLOCK - 1) / BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + NUM_BLOCKS - 1) / NUM_BLOCKS;

    if (tid == 0)
        gsBlockAccum = 0;
    GroupMemoryBarrierWithGroupSync();

    for (uint it = 0; it < nbIterationsPerBlock; ++it)
    {
        const uint workIndex = it * BLOCK + tid + nbIterationsPerBlock * blockIdx * BLOCK;

        uint cellNumCount = 0;
        if (workIndex < numTests)
        {
            const PbWork w = gWork[workIndex];
            const float cellWidth = w.b0min.w;
            const float cDistance = w.b0max.w;

            float3 mn, mx;
            combineBounds(w.b0min.xyz, w.b0max.xyz, w.b1min.xyz, w.b1max.xyz, mn, mx);
            mn -= float3(cDistance, cDistance, cDistance); // PxBounds3::fattenFast
            mx += float3(cDistance, cDistance, cDistance);

            int3 gridPosMin, gridPosMax;
            calcGridRange(gridPosMin, gridPosMax, mn, mx, cellWidth);
            const uint3 rangeSize = calcWrappedGridRangeSize(gridPosMin, gridPosMax, w.grid.xyz);
            cellNumCount = rangeSize.x * rangeSize.y * rangeSize.z;
        }

        gsLane[tid] = cellNumCount;
        GroupMemoryBarrierWithGroupSync();
        warpScanLane(tid);
        const uint offset = gsLane[tid] - cellNumCount; // exclusive prefix within warp
        if (lane == WARP - 1)
            gsWarp[warpIndex] = gsLane[tid];

        const uint prevBlockAccumulator = gsBlockAccum;
        GroupMemoryBarrierWithGroupSync();

        scanWarpTotals(tid);

        GroupMemoryBarrierWithGroupSync();

        if (workIndex < numTests)
            gOffsets[workIndex] = offset + prevBlockAccumulator + gsWarp[warpIndex];

        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0)
        gBlockOffsets[blockIdx] = gsBlockAccum;
}

// ---------------------------------------------------------------------------
// pass 2
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(512, 1, 1)]
void ctPsBoundSecondPass(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;

    const uint numTests = gParams.numTests;

    // exclusive scan over the 32 per-block totals (the CUDA does this with
    // one warp; a serial lane-0 scan is output-identical)
    if (tid == 0)
    {
        uint acc = 0;
        for (uint b = 0; b < NUM_BLOCKS; ++b)
        {
            const uint v = gBlockOffsets[b];
            gsBlockPrefix[b] = acc;
            acc += v;
        }
        gsTotalPairs = acc;
    }
    GroupMemoryBarrierWithGroupSync();

    const uint totalBlockRequired = (numTests + (BLOCK - 1)) / BLOCK;
    const uint numIterationPerBlock = (totalBlockRequired + (NUM_BLOCKS - 1)) / NUM_BLOCKS;

    const uint blockAccum = gsBlockPrefix[blockIdx];

    for (uint i = 0; i < numIterationPerBlock; ++i)
    {
        const uint workIndex = i * BLOCK + tid + numIterationPerBlock * blockIdx * BLOCK;
        if (workIndex < numTests)
            gOffsets[workIndex] = gOffsets[workIndex] + blockAccum;
    }

    if (blockIdx == 0 && tid == 0)
        gTotal[0] = gsTotalPairs;
}

// ---------------------------------------------------------------------------
// reorder (gather by the two radix-sort remap tables)
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(512, 1, 1)]
void ctPsReorderContacts(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    // real launch geometry: 32 blocks x 512 (BOUNDCELLUPDATE dims)
    const uint totalNumContacts = gTotal[0];
    const uint blockDimX = 512;
    const uint gridDimX = 32;

    // literal CUDA expression (missing parentheses over-count is harmless)
    const uint numIternations = (totalNumContacts + blockDimX * gridDimX - 1) / blockDimX * gridDimX;

    for (uint i = 0; i < numIternations; ++i)
    {
        const uint index = gtid.x + gid.x * blockDimX + i * blockDimX * gridDimX;
        if (index >= totalNumContacts)
            return;

        gSortedB[index] = gContactsIn[gRemapB[index]];
        gSortedA[index] = gContactsIn[gRemapA[index]];
    }
}

// ---------------------------------------------------------------------------
// per-thread helper verification: combine + calcGridHashInBounds
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void ctPsGridHash(uint3 tid : SV_DispatchThreadID)
{
    const uint i = tid.x;
    if (i >= gParams.numTests) return;
    const PbWork w = gWork[i];

    // every CUDA call site composes combine -> fattenFast -> hash, so the
    // fatten is part of the verified composition here
    float3 mn, mx;
    combineBounds(w.b0min.xyz, w.b0max.xyz, w.b1min.xyz, w.b1max.xyz, mn, mx);
    const float cd = w.b0max.w;
    mn -= float3(cd, cd, cd);
    mx += float3(cd, cd, cd);
    const uint hash = calcGridHashInBounds(mn, mx, w.b0min.w, uint(w.b1min.w), w.grid.xyz);
    gOffsets[i] = hash;
}
