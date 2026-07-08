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

// broadphaseOverlapCheckHistogram.slang
//
// Slang port of the broad-phase overlap-check region histogram trio
// (gpubroadphase/src/CUDA/broadphase.cu, L1246-1478):
//   - writeOutOverlapChecksForInsertedBoundsRegionsHistogram (L1246): for each
//     region cell a box's full 3D span covers, count the overlap-check
//     candidates in that region (a difference of two entries of the region's
//     ordered-handle offset table, selected start/active by isNewProjection),
//     and record (handleIndex, regionIndex) per comparison slot.
//   - computeOverlapChecksForRegionsHistogram (L1328): per-block exclusive
//     prefix over the overlap-check counts with a running block accumulator,
//     writing the per-block grand total to blockOverlapChecksRegion.
//   - outputOverlapChecksForRegionHistogram (L1425): exclusive prefix over the
//     32 per-block totals (grand total -> overlapChecksTotalRegion) added back
//     into each block's range -> a device-wide exclusive prefix sum.
//
// The counts sum to the frame's total overlap-check count, which overflows
// 32 bits (PhysX types these buffers PxU64), so the scan pair carries 64-bit
// accumulators — mapped to Slang uint64_t (the Vulkan device enables
// shaderInt64).  This is the 64-bit instance of the broadphaseRegionHistogram
// two-pass scan.  The CUDA warp-shuffle scans become groupshared 32-lane scans
// under uniform GroupMemoryBarrierWithGroupSync(); the fixed 32-block grid
// (gridDim == BLOCK_SIZE == 32) is preserved.

static const uint BLOCK = 256;
static const uint NUM_BLOCKS = 32; // gridDim.x == BLOCK_SIZE == 32
static const uint WARP = 32;
static const uint WARPS_PER_BLOCK = BLOCK / WARP; // 8
static const uint REGION_SIZE_PER_AXIS = 4;

struct OverlapCheckParams
{
    uint numHandles;      // nbProjections = numHandles * 2
    uint totalComparison; // regionAccumTotal
    uint gridDimX;        // groups launched (32)
}

[[vk::push_constant]]
ConstantBuffer<OverlapCheckParams> gParams;

RWStructuredBuffer<uint> gRegionAccum;        // binding 0  regionAccum [nbProj] (exclusive prefix)  (R)
RWStructuredBuffer<uint> gHandlesX;           // binding 1  boxHandles[0][AXIS_X] [nbProj]           (R)
RWStructuredBuffer<uint> gSapBoxX;            // binding 2  boxSapBox1D[AXIS_X] flat [numHandles][2]  (R)
RWStructuredBuffer<uint> gRegionRange;        // binding 3  regionRange[handle] flat [numHandles][8]  (R)
RWStructuredBuffer<uint> gStartHist;          // binding 4  startRegionsHistogram offset table [64][nbProj] (R)
RWStructuredBuffer<uint> gActiveHist;         // binding 5  activeRegionsHistogram offset table [64][nbProj] (R)
RWStructuredBuffer<uint64_t> gOverlapChecks;  // binding 6  overlapChecksRegion [total]  (W / RMW)
RWStructuredBuffer<uint> gHandleRegiones;     // binding 7  overlapChecksHandleRegiones [total][2] (W)
RWStructuredBuffer<uint64_t> gBlockOverlap;   // binding 8  blockOverlapChecksRegion [32]  (W/R)
RWStructuredBuffer<uint64_t> gTotalOverlap;   // binding 9  overlapChecksTotalRegion [1]   (W)

uint getHandle(uint sortedHandle) { return sortedHandle >> 3; }
uint isNewProjection(uint sortedHandle) { return (sortedHandle >> 1) & 1u; }

uint binarySearch(uint numElements, uint value)
{
    uint left = 0;
    uint right = numElements;
    while (left < right)
    {
        const uint pos = (left + right) / 2u;
        if (gRegionAccum[pos] <= value)
            left = pos + 1u;
        else
            right = pos;
    }
    return left != 0u ? left - 1u : 0u;
}

// ---------------------------------------------------------------------------
// pass 1: per-comparison overlap-check count + (handle, region) record
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void writeOutOverlapChecksForInsertedBoundsRegionsHistogram(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const uint nbProjections = gParams.numHandles * 2u;
    const uint totalComparison = gParams.totalComparison;
    const uint gridDimX = gParams.gridDimX;

    const uint nbBlocksRequired = (totalComparison + BLOCK - 1u) / BLOCK;
    const uint numIterationPerBlock = (nbBlocksRequired + gridDimX - 1u) / gridDimX;

    for (uint it = 0; it < numIterationPerBlock; ++it)
    {
        const uint workIndex = it * BLOCK + tid + numIterationPerBlock * blockIdx * BLOCK;
        if (workIndex < totalComparison)
        {
            const uint boxStartIndexX = binarySearch(nbProjections, workIndex);
            const uint offset = workIndex - gRegionAccum[boxStartIndexX];
            const uint sortedHandle = gHandlesX[boxStartIndexX];
            const uint handle = getHandle(sortedHandle);
            const uint isNew = isNewProjection(sortedHandle);

            const uint boxEndIndexX = gSapBoxX[handle * 2u + 1u];

            const uint rr = handle * 8u;
            const uint sRegionX = gRegionRange[rr + 0u];
            const uint sRegionY = gRegionRange[rr + 1u];
            const uint sRegionZ = gRegionRange[rr + 2u];
            const uint eRegionY = gRegionRange[rr + 5u];
            const uint eRegionZ = gRegionRange[rr + 6u];

            const uint nbZRegions = (eRegionZ - sRegionZ) + 1u;
            const uint nbYRegions = (eRegionY - sRegionY) + 1u;
            const uint zRegion = (offset % nbZRegions) + sRegionZ;
            const uint yRegion = ((offset / nbZRegions) % nbYRegions) + sRegionY;
            const uint xRegion = (offset / (nbZRegions * nbYRegions)) + sRegionX;

            const uint regionIndex = (xRegion + yRegion * REGION_SIZE_PER_AXIS
                                      + zRegion * (REGION_SIZE_PER_AXIS * REGION_SIZE_PER_AXIS))
                                     * nbProjections;

            // isNew -> start-region table, else active-region table
            uint regionStartIndex, regionEndIndex;
            if (isNew != 0u)
            {
                regionStartIndex = gStartHist[regionIndex + boxStartIndexX + 1u];
                regionEndIndex = gStartHist[regionIndex + boxEndIndexX];
            }
            else
            {
                regionStartIndex = gActiveHist[regionIndex + boxStartIndexX + 1u];
                regionEndIndex = gActiveHist[regionIndex + boxEndIndexX];
            }

            gHandleRegiones[workIndex * 2u + 0u] = handle;
            gHandleRegiones[workIndex * 2u + 1u] = regionIndex;
            gOverlapChecks[workIndex] = uint64_t(regionEndIndex - regionStartIndex);
        }
    }
}

// ---------------------------------------------------------------------------
// pass 2: per-block segmented exclusive scan (64-bit) + block totals
// ---------------------------------------------------------------------------
groupshared uint64_t gsLane[BLOCK];
groupshared uint64_t gsWarp[WARPS_PER_BLOCK];
groupshared uint64_t gsBlockAccum;
groupshared uint64_t gsBlockPrefix[NUM_BLOCKS];

void warpScanLane(uint tid)
{
    for (uint off = 1; off < WARP; off <<= 1)
    {
        const uint lane = tid & (WARP - 1);
        const uint64_t add = (lane >= off) ? gsLane[tid - off] : uint64_t(0);
        GroupMemoryBarrierWithGroupSync();
        gsLane[tid] += add;
        GroupMemoryBarrierWithGroupSync();
    }
}

void scanWarpTotals(uint tid)
{
    const bool active = tid < WARPS_PER_BLOCK;
    uint64_t p0 = active ? gsWarp[tid] : uint64_t(0);
    uint64_t pInc = p0;
    for (uint off = 1; off < WARPS_PER_BLOCK; off <<= 1)
    {
        const bool take = active && tid >= off;
        const uint64_t pa = take ? gsWarp[tid - off] : uint64_t(0);
        GroupMemoryBarrierWithGroupSync();
        pInc += pa;
        if (active) gsWarp[tid] = pInc;
        GroupMemoryBarrierWithGroupSync();
    }
    GroupMemoryBarrierWithGroupSync();
    if (active) gsWarp[tid] = pInc - p0;
    if (tid == WARPS_PER_BLOCK - 1) gsBlockAccum += pInc;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void computeOverlapChecksForRegionsHistogram(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const uint warpIndex = tid / WARP;
    const uint lane = tid & (WARP - 1);
    const uint totalComparison = gParams.totalComparison;
    const uint gridDimX = gParams.gridDimX;

    const uint nbBlocksRequired = (totalComparison + BLOCK - 1u) / BLOCK;
    const uint numIterationPerBlock = (nbBlocksRequired + gridDimX - 1u) / gridDimX;

    if (tid == 0)
        gsBlockAccum = uint64_t(0);
    GroupMemoryBarrierWithGroupSync();

    for (uint it = 0; it < numIterationPerBlock; ++it)
    {
        const uint workIndex = it * BLOCK + tid + numIterationPerBlock * blockIdx * BLOCK;
        const uint64_t nbComparison = (workIndex < totalComparison) ? gOverlapChecks[workIndex] : uint64_t(0);

        gsLane[tid] = nbComparison;
        GroupMemoryBarrierWithGroupSync();
        warpScanLane(tid);
        const uint64_t pairCount = gsLane[tid] - nbComparison; // exclusive within warp
        if (lane == WARP - 1)
            gsWarp[warpIndex] = gsLane[tid]; // inclusive warp total

        const uint64_t prevAccum = gsBlockAccum;
        GroupMemoryBarrierWithGroupSync();

        scanWarpTotals(tid); // gsWarp -> exclusive prefixes; gsBlockAccum += tile total

        GroupMemoryBarrierWithGroupSync();

        if (workIndex < totalComparison)
            gOverlapChecks[workIndex] = pairCount + gsWarp[warpIndex] + prevAccum;

        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0)
        gBlockOverlap[blockIdx] = gsBlockAccum;
}

// ---------------------------------------------------------------------------
// pass 3: exclusive prefix over the per-block totals + fold into each block
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void outputOverlapChecksForRegionHistogram(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const uint totalComparison = gParams.totalComparison;
    const uint gridDimX = gParams.gridDimX;

    const uint nbBlocksRequired = (totalComparison + BLOCK - 1u) / BLOCK;
    const uint numIterationPerBlock = (nbBlocksRequired + gridDimX - 1u) / gridDimX;

    if (tid == 0)
    {
        uint64_t acc = uint64_t(0);
        for (uint b = 0; b < NUM_BLOCKS; ++b)
        {
            const uint64_t v = gBlockOverlap[b];
            gsBlockPrefix[b] = acc; // exclusive prefix
            acc += v;
        }
        if (blockIdx == 0)
            gTotalOverlap[0] = acc; // grand total
    }
    GroupMemoryBarrierWithGroupSync();

    const uint64_t blockAccum = gsBlockPrefix[blockIdx];

    for (uint it = 0; it < numIterationPerBlock; ++it)
    {
        const uint workIndex = it * BLOCK + tid + numIterationPerBlock * blockIdx * BLOCK;
        if (workIndex < totalComparison)
            gOverlapChecks[workIndex] = gOverlapChecks[workIndex] + blockAccum;
    }
}
