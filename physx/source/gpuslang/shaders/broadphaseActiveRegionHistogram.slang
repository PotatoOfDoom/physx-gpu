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
// P5 Step 4 (BDA, engine launch path): the broad-phase active-region histogram
// pipeline (gpubroadphase/src/CUDA/broadphase.cu, L1015-1244), the stage that
// compacts, per coarse region, the handles whose bounds touch that region.
// All three kernels take ONE PxgBroadPhaseDesc device pointer, walked through
// the byte-exact PxgBroadPhaseDescGpu mirror (pxgSlangBpDesc.slang, proven by
// testBpDescLayout), exactly as the engine launches them (computeStartAnd-
// ActiveHistogramKernel, PxgCudaBroadPhaseSap.cpp:881-894):
//   - writeOutStartAndActiveRegionHistogram (BP_WRITEOUT_ACTIVE_HISTOGRAM,
//     grid 32x256): for each region cell a box covers (the box's fixed min-X
//     region times its [minY,maxY]x[minZ,maxZ] span), stamp a 1 into that
//     region's slice of startRegionsHistogram at the box's projection index,
//     and isNewProjection into activeRegionsHistogram.
//   - computeStartAndActiveRegionHistogram (BP_COMPUTE_ACTIVE_HISTOGRAM, grid
//     64x512): turn each region slice's 0/1 flags into in-place exclusive-
//     prefix-sum offset tables (start and active counts packed 16/16 into one
//     lane and scanned together, one region slice per block iteration).
//   - outputOrderedActiveRegionHistogram (BP_OUTPUT_ACTIVE_HISTOGRAM, grid
//     64x256): scatter each box's handle into the compacted ordered start/
//     active handle arrays at the slot the prefix table assigned to the box's
//     min projection (idempotent collisions).
//
// numHandles = numPreviousHandles + numCreatedHandles - numRemovedHandles and
// totalComparision = startRegionAccumTotal are read from the desc, as the .cu
// does; every gridDim.x read becomes WorkgroupCount().x (the region loop's
// a += gridDim.x stride included).  Portability: the CUDA warpScanAdd /
// warpScanAddWriteToSharedMem __shfl scans (block=512 = 16 warps) become
// groupshared 32-lane scans under uniform GroupMemoryBarrierWithGroupSync(),
// so no subgroup-width assumption is made.  Each region slice's start/active
// counts stay < 2^16, so the 16/16 packing never carries across the field
// boundary.  PxgIntegerRegion = { uint4 minRange; uint4 maxRange } = 8 flat
// uints; PxgSapBox1D = { PxU32 mMinMax[2] } = 2 flat uints.

import pxgSlangBpDesc;

static const uint REGION_SIZE_PER_AXIS = 4;
static const uint TOTAL_REGION_SIZE = 64; // 4^3
static const uint WARP = 32;
static const uint BLOCK1 = 256;   // writeout / output (BP_WRITEOUT/BP_OUTPUT_ACTIVE_HISTOGRAM)
static const uint BLOCK2 = 512;   // compute (WARP_SIZE * WARP_PERBLOCK_SIZE_16)
static const uint WARPS2 = BLOCK2 / WARP; // 16

struct BpDescParams
{
    uint64_t bpDesc;
}
[[vk::push_constant]]
ConstantBuffer<BpDescParams> gBp;

// mirrors binarySearch (gpucommon/src/CUDA/reduction.cuh): largest index into
// data s.t. data[index] <= value, else 0
uint binarySearch(Ptr<uint> data, uint numElements, uint value)
{
    uint left = 0;
    uint right = numElements;
    while (left < right)
    {
        const uint pos = (left + right) / 2u;
        if (data[pos] <= value)
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

Decoded decode(Ptr<uint> startRegionAccum, Ptr<uint> sortedHandles, Ptr<uint> regionRange,
               uint workIndex, uint nbProjections)
{
    Decoded d;
    d.pos = binarySearch(startRegionAccum, nbProjections, workIndex);
    d.sortedHandle = sortedHandles[d.pos];
    d.handle = bpGetHandle(d.sortedHandle);
    const uint offset = workIndex - startRegionAccum[d.pos];

    const uint rr = d.handle * 8u;
    const uint sRegionX = regionRange[rr + 0u]; // minRange.x
    const uint sRegionY = regionRange[rr + 1u]; // minRange.y
    d.sRegionZ = regionRange[rr + 2u];          // minRange.z
    d.eRegionZ = regionRange[rr + 6u];          // maxRange.z

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
// (BP_WRITEOUT_ACTIVE_HISTOGRAM, grid 32 x 256)
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void writeOutStartAndActiveRegionHistogram(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    Ptr<PxgBroadPhaseDescGpu> desc = Ptr<PxgBroadPhaseDescGpu>(gBp.bpDesc);

    Ptr<uint> startRegionAccum = Ptr<uint>(desc[0].startRegionAccum);
    Ptr<uint> regionRange = Ptr<uint>(desc[0].regionRange);
    Ptr<uint> sortedHandles = Ptr<uint>(desc[0].boxHandles[0]); // boxHandles[0][AXIS_X], current frame
    Ptr<uint> activeRegionsHistogram = Ptr<uint>(desc[0].activeRegionsHistogram);
    Ptr<uint> startRegionsHistogram = Ptr<uint>(desc[0].startRegionsHistogram);

    const uint numHandles = desc[0].numPreviousHandles + desc[0].numCreatedHandles - desc[0].numRemovedHandles;
    const uint nbProjections = numHandles * 2u;

    const uint totalComparision = desc[0].startRegionAccumTotal;

    const uint nbBlocksRequired = (totalComparision + BLOCK1 - 1u) / BLOCK1;
    const uint nbIterationsPerBlock = (nbBlocksRequired + WorkgroupCount().x - 1u) / WorkgroupCount().x;

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * BLOCK1 + gtid.x + nbIterationsPerBlock * gid.x * BLOCK1;

        if (workIndex < totalComparision)
        {
            const Decoded d = decode(startRegionAccum, sortedHandles, regionRange, workIndex, nbProjections);
            startRegionsHistogram[d.regionBase + d.pos] = 1u;
            activeRegionsHistogram[d.regionBase + d.pos] = bpIsNewProjection(d.sortedHandle) ? 1u : 0u;
        }
    }
}

// ---------------------------------------------------------------------------
// pass 2: per-region packed (start<<16|active) exclusive prefix, in place
// (BP_COMPUTE_ACTIVE_HISTOGRAM, grid 64 x 512)
// ---------------------------------------------------------------------------
groupshared uint gsLane[BLOCK2];   // packed per-lane scan (sStartHistogram)
groupshared uint gsWarp[WARPS2];   // packed warp totals -> exclusive prefixes (sStartWarpAccumulator)
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
    Ptr<PxgBroadPhaseDescGpu> desc = Ptr<PxgBroadPhaseDescGpu>(gBp.bpDesc);

    const uint tid = gtid.x;
    const uint warpIndex = tid / WARP;
    const uint lane = tid & (WARP - 1);

    const uint numHandles = desc[0].numPreviousHandles + desc[0].numCreatedHandles - desc[0].numRemovedHandles;
    const uint nbProjections = numHandles * 2u;

    const uint nbIterationsPerBlock = (nbProjections + BLOCK2 - 1u) / BLOCK2;

    for (uint a = gid.x; a < TOTAL_REGION_SIZE; a += WorkgroupCount().x)
    {
        if (tid == 0)
        {
            gsActiveBlockAccum = 0;
            gsStartBlockAccum = 0;
        }
        GroupMemoryBarrierWithGroupSync();

        const uint regionStartOffset = a * nbProjections;
        Ptr<uint> activeRegionsHistogram = Ptr<uint>(desc[0].activeRegionsHistogram);
        Ptr<uint> startRegionsHistogram = Ptr<uint>(desc[0].startRegionsHistogram);

        for (uint i = 0; i < nbIterationsPerBlock; ++i)
        {
            const uint workIndex = i * BLOCK2 + tid;

            uint activeHistVal = 0, startHistVal = 0;
            if (workIndex < nbProjections)
            {
                activeHistVal = activeRegionsHistogram[regionStartOffset + workIndex];
                startHistVal = startRegionsHistogram[regionStartOffset + workIndex];
            }
            const uint combinedVal = (startHistVal << 16) | activeHistVal;

            gsLane[tid] = combinedVal;
            GroupMemoryBarrierWithGroupSync();
            warpScanLane(tid);
            const uint startAccumVal = gsLane[tid] - combinedVal; // exclusive within warp (packed)
            if (lane == WARP - 1)
                gsWarp[warpIndex] = gsLane[tid]; // inclusive warp total (packed)

            const uint prevActiveBlockAccumulator = gsActiveBlockAccum;
            const uint prevStartBlockAccumulator = gsStartBlockAccum;
            GroupMemoryBarrierWithGroupSync();

            scanWarpTotals(tid); // gsWarp -> exclusive prefixes; block accums += tile total

            GroupMemoryBarrierWithGroupSync();

            if (workIndex < nbProjections)
            {
                const uint warpAccum = startAccumVal + gsWarp[warpIndex];
                const uint activeIndex = (warpAccum & 0xFFFFu) + prevActiveBlockAccumulator;
                const uint startIndex = (warpAccum >> 16) + prevStartBlockAccumulator;
                activeRegionsHistogram[regionStartOffset + workIndex] = activeIndex;
                startRegionsHistogram[regionStartOffset + workIndex] = startIndex;
            }

            GroupMemoryBarrierWithGroupSync();
        }
    }
}

// ---------------------------------------------------------------------------
// pass 3: scatter handles into the compacted ordered start/active arrays
// (BP_OUTPUT_ACTIVE_HISTOGRAM, grid 64 x 256)
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void outputOrderedActiveRegionHistogram(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    Ptr<PxgBroadPhaseDescGpu> desc = Ptr<PxgBroadPhaseDescGpu>(gBp.bpDesc);

    Ptr<uint> startRegionAccum = Ptr<uint>(desc[0].startRegionAccum);
    Ptr<uint> regionRange = Ptr<uint>(desc[0].regionRange);
    Ptr<uint> sortedHandles = Ptr<uint>(desc[0].boxHandles[0]); // boxHandles[0][AXIS_X], current frame
    Ptr<uint> sapBoxes = Ptr<uint>(desc[0].boxSapBox1D[0]);     // boxSapBox1D[AXIS_X], 2 uints/box
    Ptr<uint> startRegionsHistogram = Ptr<uint>(desc[0].startRegionsHistogram);
    Ptr<uint> activeRegionsHistogram = Ptr<uint>(desc[0].activeRegionsHistogram);
    Ptr<uint> orderedStartHandles = Ptr<uint>(desc[0].orderedStartRegionHandles);
    Ptr<uint> orderedActiveHandles = Ptr<uint>(desc[0].orderedActiveRegionHandles);

    const uint numHandles = desc[0].numPreviousHandles + desc[0].numCreatedHandles - desc[0].numRemovedHandles;
    const uint nbProjections = numHandles * 2u;

    const uint totalComparision = desc[0].startRegionAccumTotal;

    const uint nbBlocksRequired = (totalComparision + BLOCK1 - 1u) / BLOCK1;
    const uint nbIterationsPerBlock = (nbBlocksRequired + WorkgroupCount().x - 1u) / WorkgroupCount().x;

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * BLOCK1 + gtid.x + nbIterationsPerBlock * gid.x * BLOCK1;

        if (workIndex < totalComparision)
        {
            const Decoded d = decode(startRegionAccum, sortedHandles, regionRange, workIndex, nbProjections);

            const uint startProjIndex = sapBoxes[d.handle * 2u + 0u]; // sapBox.mMinMax[0]
            const uint startIndex = startRegionsHistogram[d.regionBase + startProjIndex];

            if (d.eRegionZ >= d.sRegionZ)
            {
                // Output to orderedStartHandles
                orderedStartHandles[d.regionBase + startIndex] = d.handle;
                if (bpIsNewProjection(d.sortedHandle))
                {
                    const uint activeIndex = activeRegionsHistogram[d.regionBase + startProjIndex];
                    orderedActiveHandles[d.regionBase + activeIndex] = d.handle;
                }
            }
        }
    }
}
