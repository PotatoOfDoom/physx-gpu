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

// broadphaseActiveRegionHistogram.slang
//
// Slang port of the broad-phase active-region histogram pipeline
// (gpubroadphase/src/CUDA/broadphase.cu, L1015-1244), the stage that compacts,
// per coarse region, the handles whose bounds touch that region:
//   - writeOutStartAndActiveRegionHistogram (L1015): for each region cell a box
//     covers (the box's fixed min-X region times its [minY,maxY]x[minZ,maxZ]
//     span), stamp a 1 into that region's slice of startRegionsHistogram at the
//     box's projection index, and isNewProjection into activeRegionsHistogram.
//   - computeStartAndActiveRegionHistogram (L1075): turn each region slice's 0/1
//     flags into in-place exclusive-prefix-sum offset tables (start and active
//     counts packed 16/16 into one lane and scanned together, one region per
//     block, block=512).
//   - outputOrderedActiveRegionHistogram (L1171): scatter each box's handle into
//     the compacted ordered start/active handle arrays at the slot the prefix
//     table assigned to the box's min projection (idempotent collisions).
//
// Portability: the CUDA warpScanAdd / warpScanAddWriteToSharedMem __shfl scans
// (block=512 = 16 warps) become groupshared 32-lane scans under uniform
// GroupMemoryBarrierWithGroupSync(), so no subgroup-width assumption is made;
// the per-region contiguous nbProjections stride and the fixed 32/64-block
// grids are preserved.  Each region slice's start/active counts stay < 2^16, so
// the 16/16 packing never carries across the field boundary.

static const uint REGION_SIZE_PER_AXIS = 4;
static const uint TOTAL_REGION_SIZE = 64; // 4^3
static const uint WARP = 32;
static const uint BLOCK1 = 256;   // writeout / output
static const uint BLOCK2 = 512;   // compute (16 warps)
static const uint WARPS2 = BLOCK2 / WARP; // 16

struct ActiveRegionParams
{
    uint numHandles;       // nbProjections = numHandles * 2
    uint totalComparison;  // startRegionAccumTotal
    uint gridDimX;         // groups launched (32 for writeout/output)
}

[[vk::push_constant]]
ConstantBuffer<ActiveRegionParams> gParams;

RWStructuredBuffer<uint> gStartRegionAccum; // binding 0  startRegionAccum [nbProj] (exclusive prefix)  (R)
RWStructuredBuffer<uint> gHandlesX;         // binding 1  boxHandles[0][AXIS_X] [nbProj]                (R)
RWStructuredBuffer<uint> gRegionRange;      // binding 2  regionRange[handle] flat [numHandles][8]       (R)
RWStructuredBuffer<uint> gStartHist;        // binding 3  startRegionsHistogram [64][nbProj]  (W/RMW/R)
RWStructuredBuffer<uint> gActiveHist;       // binding 4  activeRegionsHistogram [64][nbProj] (W/RMW/R)
RWStructuredBuffer<uint> gSapBoxX;          // binding 5  boxSapBox1D[AXIS_X] flat [numHandles][2]        (R)
RWStructuredBuffer<uint> gOrderedStart;     // binding 6  orderedStartRegionHandles [64][nbProj]          (W)
RWStructuredBuffer<uint> gOrderedActive;    // binding 7  orderedActiveRegionHandles [64][nbProj]         (W)

uint getHandle(uint sortedHandle) { return sortedHandle >> 3; }
uint isNewProjection(uint sortedHandle) { return (sortedHandle >> 1) & 1u; }

// largest index into gStartRegionAccum s.t. value <= workIndex, else 0
uint binarySearch(uint numElements, uint value)
{
    uint left = 0;
    uint right = numElements;
    while (left < right)
    {
        const uint pos = (left + right) / 2u;
        if (gStartRegionAccum[pos] <= value)
            left = pos + 1u;
        else
            right = pos;
    }
    return left != 0u ? left - 1u : 0u;
}

// Decode a work item into (pos, handle, regionBase) shared by writeout/output.
struct Decoded
{
    uint pos;
    uint handle;
    uint sortedHandle;
    uint regionBase; // regionIndex * nbProjections
    uint sRegionZ;
    uint eRegionZ;
}

Decoded decode(uint workIndex, uint nbProjections)
{
    Decoded d;
    d.pos = binarySearch(nbProjections, workIndex);
    d.sortedHandle = gHandlesX[d.pos];
    d.handle = getHandle(d.sortedHandle);
    const uint offset = workIndex - gStartRegionAccum[d.pos];

    const uint rr = d.handle * 8u;
    const uint sRegionX = gRegionRange[rr + 0u];
    const uint sRegionY = gRegionRange[rr + 1u];
    d.sRegionZ = gRegionRange[rr + 2u];
    d.eRegionZ = gRegionRange[rr + 6u]; // maxRange.z

    const uint nbZRegions = (d.eRegionZ - d.sRegionZ) + 1u;
    const uint zRegion = (offset % nbZRegions) + d.sRegionZ;
    const uint yRegion = (offset / nbZRegions) + sRegionY;
    const uint regionIndex = sRegionX + yRegion * REGION_SIZE_PER_AXIS
                             + zRegion * (REGION_SIZE_PER_AXIS * REGION_SIZE_PER_AXIS);
    d.regionBase = regionIndex * nbProjections;
    return d;
}

// ---------------------------------------------------------------------------
// pass 1: stamp start/active 0-1 flags into each region slice
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void writeOutStartAndActiveRegionHistogram(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const uint nbProjections = gParams.numHandles * 2u;
    const uint totalComparison = gParams.totalComparison;
    const uint gridDimX = gParams.gridDimX;

    const uint nbBlocksRequired = (totalComparison + BLOCK1 - 1u) / BLOCK1;
    const uint nbIterationsPerBlock = (nbBlocksRequired + gridDimX - 1u) / gridDimX;

    for (uint it = 0; it < nbIterationsPerBlock; ++it)
    {
        const uint workIndex = it * BLOCK1 + tid + nbIterationsPerBlock * blockIdx * BLOCK1;
        if (workIndex < totalComparison)
        {
            const Decoded d = decode(workIndex, nbProjections);
            gStartHist[d.regionBase + d.pos] = 1u;
            gActiveHist[d.regionBase + d.pos] = isNewProjection(d.sortedHandle);
        }
    }
}

// ---------------------------------------------------------------------------
// pass 2: per-region packed (start<<16|active) exclusive prefix, in place
// ---------------------------------------------------------------------------
groupshared uint gsLane[BLOCK2];   // packed per-lane scan
groupshared uint gsWarp[WARPS2];   // packed warp totals -> exclusive prefixes
groupshared uint gsStartBlockAccum;
groupshared uint gsActiveBlockAccum;

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

// exclusive scan over the 16 packed warp totals; unpacks + folds the grand total
// into the two block accumulators.  Uniform control flow; only lanes < 16 touch
// gsWarp.
void scanWarpTotals(uint tid)
{
    const bool active = tid < WARPS2;
    uint p0 = active ? gsWarp[tid] : 0u;
    uint pInc = p0;
    for (uint off = 1; off < WARPS2; off <<= 1)
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
    if (tid == WARPS2 - 1)
    {
        gsStartBlockAccum += pInc >> 16;
        gsActiveBlockAccum += pInc & 0xFFFFu;
    }
}

[shader("compute")]
[numthreads(512, 1, 1)]
void computeStartAndActiveRegionHistogram(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint warpIndex = tid / WARP;
    const uint lane = tid & (WARP - 1);
    const uint nbProjections = gParams.numHandles * 2u;
    const uint nbIterationsPerBlock = (nbProjections + BLOCK2 - 1u) / BLOCK2;

    for (uint a = gid.x; a < TOTAL_REGION_SIZE; a += TOTAL_REGION_SIZE)
    {
        if (tid == 0)
        {
            gsStartBlockAccum = 0;
            gsActiveBlockAccum = 0;
        }
        GroupMemoryBarrierWithGroupSync();

        const uint regionBase = a * nbProjections;

        for (uint it = 0; it < nbIterationsPerBlock; ++it)
        {
            const uint workIndex = it * BLOCK2 + tid;

            uint activeHistVal = 0, startHistVal = 0;
            if (workIndex < nbProjections)
            {
                activeHistVal = gActiveHist[regionBase + workIndex];
                startHistVal = gStartHist[regionBase + workIndex];
            }
            const uint combinedVal = (startHistVal << 16) | activeHistVal;

            gsLane[tid] = combinedVal;
            GroupMemoryBarrierWithGroupSync();
            warpScanLane(tid);
            const uint startAccumVal = gsLane[tid] - combinedVal; // exclusive within warp (packed)
            if (lane == WARP - 1)
                gsWarp[warpIndex] = gsLane[tid]; // inclusive warp total (packed)

            const uint prevActive = gsActiveBlockAccum;
            const uint prevStart = gsStartBlockAccum;
            GroupMemoryBarrierWithGroupSync();

            scanWarpTotals(tid); // gsWarp -> exclusive prefixes; block accums += tile total

            GroupMemoryBarrierWithGroupSync();

            if (workIndex < nbProjections)
            {
                const uint warpAccum = startAccumVal + gsWarp[warpIndex];
                const uint activeIndex = (warpAccum & 0xFFFFu) + prevActive;
                const uint startIndex = (warpAccum >> 16) + prevStart;
                gActiveHist[regionBase + workIndex] = activeIndex;
                gStartHist[regionBase + workIndex] = startIndex;
            }

            GroupMemoryBarrierWithGroupSync();
        }
    }
}

// ---------------------------------------------------------------------------
// pass 3: scatter handles into the compacted ordered start/active arrays
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void outputOrderedActiveRegionHistogram(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const uint numHandles = gParams.numHandles;
    const uint nbProjections = numHandles * 2u;
    const uint totalComparison = gParams.totalComparison;
    const uint gridDimX = gParams.gridDimX;

    const uint nbBlocksRequired = (totalComparison + BLOCK1 - 1u) / BLOCK1;
    const uint nbIterationsPerBlock = (nbBlocksRequired + gridDimX - 1u) / gridDimX;

    for (uint it = 0; it < nbIterationsPerBlock; ++it)
    {
        const uint workIndex = it * BLOCK1 + tid + nbIterationsPerBlock * blockIdx * BLOCK1;
        if (workIndex < totalComparison)
        {
            const Decoded d = decode(workIndex, nbProjections);

            const uint startProjIndex = gSapBoxX[d.handle * 2u + 0u]; // box min projection
            const uint startIndex = gStartHist[d.regionBase + startProjIndex];

            if (d.eRegionZ >= d.sRegionZ)
            {
                gOrderedStart[d.regionBase + startIndex] = d.handle;
                if (isNewProjection(d.sortedHandle) != 0u)
                {
                    const uint activeIndex = gActiveHist[d.regionBase + startProjIndex];
                    gOrderedActive[d.regionBase + activeIndex] = d.handle;
                }
            }
        }
    }
}
