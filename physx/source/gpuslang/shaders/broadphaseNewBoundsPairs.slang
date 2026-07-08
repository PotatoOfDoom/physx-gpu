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

// broadphaseNewBoundsPairs.slang
//
// Slang port of generateFoundPairsForNewBoundsRegion (BP_GENERATE_FOUNDPAIR_
// NEWBOUNDS) from gpubroadphase/src/CUDA/broadphase.cu (L1723-1908) — the final
// kernel of the one-shot region pipeline.  It consumes everything the region
// stages produced (the overlap-check prefix + handle/region records, the
// per-region ordered-handle offset tables and compacted handle lists) and emits
// the found (newly overlapping) pairs among the created bounds.
//
// Grid 64 x block 256; a 64-bit grid-stride loop over overlapChecksTotalRegion.
// Each work item:
//   1. binary-searches the overlap-check prefix to find its (handle, region)
//      entry and the offset within that entry's candidate slice;
//   2. resolves the candidate otherHandle from the region's compacted ordered
//      handle list (start or active list, selected by isNewProjection);
//   3. dedups by keeping the pair only in the region at the max of the two
//      boxes' min corners, applies the group/environment filter, and does the
//      integer-AABB intersection test on the new bounds;
//   4. block-compacts the hits (groupshared 32-lane scan) and atomically
//      reserves a contiguous output range in the found-pair report.
//
// Portability: the CUDA warpScanAdd/__ballot scans become groupshared 32-lane
// scans under uniform barriers; the 64-bit work indices are carried in
// uint64_t (the device enables shaderInt64); the atomicAdd to the 32-bit
// sharedFoundPairIndex counter becomes InterlockedAdd.

static const uint BLOCK = 256;
static const uint WARP = 32;
static const uint WARPS_PER_BLOCK = BLOCK / WARP; // 8
static const uint REGION_SIZE_PER_AXIS = 4;
static const uint PX_INVALID_U32 = 0xffffffffu;

struct FoundPairParams
{
    uint numHandles;        // nbProjections = numHandles * 2
    uint overlCheckesSize;  // regionAccumTotal (binarySearch length)
    uint gridDimX;          // 64
    uint maxFoundLostPairs;
    uint useEnvIds;         // 0/1
    uint64_t totalPair;     // overlapChecksTotalRegion (work count)
}

[[vk::push_constant]]
ConstantBuffer<FoundPairParams> gParams;

RWStructuredBuffer<uint64_t> gOverlapChecks; // binding 0  overlapChecksRegion [overlCheckesSize] (R)
RWStructuredBuffer<uint> gHandleRegiones;    // binding 1  overlapChecksHandleRegiones [·][2]       (R)
RWStructuredBuffer<uint> gRegionRange;       // binding 2  regionRange[handle] flat [numHandles][8]  (R)
RWStructuredBuffer<uint> gSapBoxX;           // binding 3  boxSapBox1D[AXIS_X] flat [numHandles][2]   (R)
RWStructuredBuffer<uint> gHandlesX;          // binding 4  boxHandles[0][AXIS_X] [nbProj]             (R)
RWStructuredBuffer<uint> gStartHist;         // binding 5  startRegionsHistogram offset table [64][nbProj] (R)
RWStructuredBuffer<uint> gActiveHist;        // binding 6  activeRegionsHistogram offset table [64][nbProj] (R)
RWStructuredBuffer<uint> gOrderedStart;      // binding 7  orderedStartRegionHandles [64][nbProj]     (R)
RWStructuredBuffer<uint> gOrderedActive;     // binding 8  orderedActiveRegionHandles [64][nbProj]    (R)
RWStructuredBuffer<uint> gNewBounds;         // binding 9  newIntegerBounds [numHandles][6]           (R)
RWStructuredBuffer<uint> gGroups;            // binding 10 updateData_groups [numHandles]             (R)
RWStructuredBuffer<uint> gEnvs;              // binding 11 updateData_envIDs [numHandles]             (R)
RWStructuredBuffer<uint> gFoundReport;       // binding 12 foundPairReport [maxPairs][2]              (W)
RWStructuredBuffer<uint> gSharedFound;       // binding 13 sharedFoundPairIndex [1] (atomic)          (W)

uint isNewProjection(uint sortedHandle) { return (sortedHandle >> 1) & 1u; }

// PxgIntegerAABB.intersects on a flat [6] view {minX,minY,minZ,maxX,maxY,maxZ}
bool intersects(uint bi, uint bj)
{
    const uint a = bi * 6u, b = bj * 6u;
    return !(gNewBounds[b + 0u] > gNewBounds[a + 3u] || gNewBounds[a + 0u] > gNewBounds[b + 3u] ||
             gNewBounds[b + 1u] > gNewBounds[a + 4u] || gNewBounds[a + 1u] > gNewBounds[b + 4u] ||
             gNewBounds[b + 2u] > gNewBounds[a + 5u] || gNewBounds[a + 2u] > gNewBounds[b + 5u]);
}

bool filtering(uint groupId, uint otherGroupId, uint envId, uint otherEnvId)
{
    return (groupId != otherGroupId) &&
           ((envId == otherEnvId) || (envId == PX_INVALID_U32) || (otherEnvId == PX_INVALID_U32));
}

// largest index into gOverlapChecks s.t. value <= workIndex, else 0 (64-bit compares)
uint binarySearch(uint numElements, uint64_t value)
{
    uint left = 0;
    uint right = numElements;
    while (left < right)
    {
        const uint pos = (left + right) / 2u;
        if (gOverlapChecks[pos] <= value)
            left = pos + 1u;
        else
            right = pos;
    }
    return left != 0u ? left - 1u : 0u;
}

groupshared uint gsAcc[BLOCK];
groupshared uint gsWarp[WARPS_PER_BLOCK];
groupshared uint gsBase;

void warpScanAcc(uint tid)
{
    for (uint off = 1; off < WARP; off <<= 1)
    {
        const uint lane = tid & (WARP - 1);
        const uint add = (lane >= off) ? gsAcc[tid - off] : 0u;
        GroupMemoryBarrierWithGroupSync();
        gsAcc[tid] += add;
        GroupMemoryBarrierWithGroupSync();
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void generateFoundPairsForNewBoundsRegion(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const uint warpIndex = tid / WARP;
    const uint lane = tid & (WARP - 1);

    const uint numHandles = gParams.numHandles;
    const uint nbProjections = numHandles * 2u;
    const uint overlCheckesSize = gParams.overlCheckesSize;
    const uint gridDimX = gParams.gridDimX;
    const uint maxPairs = gParams.maxFoundLostPairs;
    const bool useEnvIds = gParams.useEnvIds != 0u;
    const uint64_t totalPair = gParams.totalPair;

    const uint64_t totalBlockRequired = (totalPair + uint64_t(BLOCK - 1u)) / uint64_t(BLOCK);
    const uint64_t numIterationPerBlock = (totalBlockRequired + uint64_t(gridDimX - 1u)) / uint64_t(gridDimX);

    for (uint64_t i = 0; i < numIterationPerBlock; ++i)
    {
        const uint64_t workIndex = i * uint64_t(BLOCK) + uint64_t(tid)
                                   + numIterationPerBlock * uint64_t(blockIdx) * uint64_t(BLOCK);

        uint foundCount = 0;
        uint handle = PX_INVALID_U32;
        uint otherHandle = PX_INVALID_U32;

        if (workIndex < totalPair)
        {
            const uint index = binarySearch(overlCheckesSize, workIndex);
            handle = gHandleRegiones[index * 2u + 0u];
            const uint regionIndex = gHandleRegiones[index * 2u + 1u]; // = cell * nbProjections

            const uint boxStartIndexX = gSapBoxX[handle * 2u + 0u];
            const uint boxEndIndexX = gSapBoxX[handle * 2u + 1u];
            const uint isNew = isNewProjection(gHandlesX[boxStartIndexX]);

            const uint64_t offset = workIndex - gOverlapChecks[index];

            const uint regionStart = (isNew != 0u) ? gStartHist[regionIndex + boxStartIndexX + 1u]
                                                   : gActiveHist[regionIndex + boxStartIndexX + 1u];
            const uint regionEnd = (isNew != 0u) ? gStartHist[regionIndex + boxEndIndexX]
                                                 : gActiveHist[regionIndex + boxEndIndexX];

            if (regionEnd > regionStart)
            {
                const uint64_t otherIndex64 = uint64_t(regionStart) + offset;
                if (otherIndex64 < uint64_t(regionEnd))
                {
                    const uint otherIndex = uint(otherIndex64);
                    const uint group = gGroups[handle];
                    const uint envId = useEnvIds ? gEnvs[handle] : PX_INVALID_U32;

                    otherHandle = (isNew != 0u) ? gOrderedStart[regionIndex + otherIndex]
                                                : gOrderedActive[regionIndex + otherIndex];

                    // dedup: report the pair only in the region at the max of the
                    // two boxes' min corners
                    const uint rr = handle * 8u;
                    const uint orr = otherHandle * 8u;
                    const uint x = max(gRegionRange[rr + 0u], gRegionRange[orr + 0u]);
                    const uint y = max(gRegionRange[rr + 1u], gRegionRange[orr + 1u]);
                    const uint z = max(gRegionRange[rr + 2u], gRegionRange[orr + 2u]);
                    const uint otherRegionIndex = (x + y * REGION_SIZE_PER_AXIS
                                                   + z * (REGION_SIZE_PER_AXIS * REGION_SIZE_PER_AXIS))
                                                  * nbProjections;

                    if (regionIndex == otherRegionIndex)
                    {
                        const uint otherGroup = gGroups[otherHandle];
                        const uint otherEnvId = useEnvIds ? gEnvs[otherHandle] : PX_INVALID_U32;
                        if (filtering(group, otherGroup, envId, otherEnvId))
                            foundCount = intersects(handle, otherHandle) ? 1u : 0u;
                    }
                }
            }
        }

        // block stream-compaction of the found hits
        gsAcc[tid] = foundCount;
        GroupMemoryBarrierWithGroupSync();
        warpScanAcc(tid);
        const uint res = gsAcc[tid] - foundCount; // exclusive prefix within warp
        if (lane == WARP - 1)
            gsWarp[warpIndex] = gsAcc[tid]; // inclusive warp total
        GroupMemoryBarrierWithGroupSync();

        if (tid == 0)
        {
            uint acc = 0;
            [unroll]
            for (uint w = 0; w < WARPS_PER_BLOCK; ++w)
            {
                const uint t = gsWarp[w];
                gsWarp[w] = acc; // exclusive prefix of warp totals
                acc += t;
            }
            uint base = 0;
            if (acc > 0u)
                InterlockedAdd(gSharedFound[0], acc, base);
            gsBase = base;
        }
        GroupMemoryBarrierWithGroupSync();

        if (foundCount != 0u)
        {
            const uint position = gsBase + res + gsWarp[warpIndex];
            if (position < maxPairs)
            {
                // PxgBroadPhasePair canonicalizes to (min, max) so mVolA < mVolB
                gFoundReport[position * 2u + 0u] = min(handle, otherHandle);
                gFoundReport[position * 2u + 1u] = max(handle, otherHandle);
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
