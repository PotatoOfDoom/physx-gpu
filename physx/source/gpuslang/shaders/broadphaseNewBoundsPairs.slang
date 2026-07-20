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
// P5 Step 4 (BDA, engine launch path): generateFoundPairsForNewBoundsRegion
// (BP_GENERATE_FOUNDPAIR_NEWBOUNDS) from gpubroadphase/src/CUDA/broadphase.cu
// (L1723-1908) — the final kernel of the one-shot region pipeline, driven
// exactly as the engine launches it: ONE PxgBroadPhaseDesc device pointer,
// walked through the byte-exact PxgBroadPhaseDescGpu mirror.  It consumes
// everything the region stages produced (the overlap-check prefix + handle/
// region records, the per-region ordered-handle offset tables and compacted
// handle lists) and emits the found (newly overlapping) pairs among the
// created bounds.
//
// Grid 64 x block 256; a 64-bit grid-stride loop over the desc's
// overlapChecksTotalRegion (u64 VALUE).  Each work item:
//   1. binary-searches the overlap-check prefix to find its (handle, region)
//      entry and the offset within that entry's candidate slice;
//   2. resolves the candidate otherHandle from the region's compacted ordered
//      handle list (start or active list, selected by isNewProjection);
//   3. dedups by keeping the pair only in the region at the max of the two
//      boxes' min corners, applies the group/environment filter (the .cu's
//      USE_ENV_IDS path with its boxEnvIDs NULL-pointer branch), and does the
//      integer-AABB intersection test on the new bounds;
//   4. block-compacts the hits (groupshared 32-lane scan) and atomically
//      reserves a contiguous output range in the found-pair report — a BDA
//      atomic (InterlockedAdd) directly on the desc's sharedFoundPairIndex
//      field, exactly the .cu's atomicAdd(&bpDesc->sharedFoundPairIndex, n).
//
// Portability: the CUDA warpScanAdd/__ballot scans become groupshared 32-lane
// scans under uniform barriers; the 64-bit work indices are carried in
// uint64_t (the device enables shaderInt64); numIterationPerBlock divides by
// gridDim.x = WorkgroupCount().x as the .cu does.

import pxgSlangBpDesc;

static const uint BLOCK = 256;
static const uint WARP = 32;
static const uint WARPS_PER_BLOCK = BLOCK / WARP; // 8
static const uint REGION_SIZE_PER_AXIS = 4;
static const uint PX_INVALID_U32 = 0xffffffffu;

// engine ABI: the PxgBroadPhaseDesc device pointer only
struct BpDescParams
{
    uint64_t bpDesc;
}
[[vk::push_constant]]
ConstantBuffer<BpDescParams> gBp;

// PxgIntegerAABB.intersects on a flat [6] view {minX,minY,minZ,maxX,maxY,maxZ}
bool intersects(Ptr<uint> bounds, uint bi, uint bj)
{
    const uint a = bi * 6u, b = bj * 6u;
    return !(bounds[b + 0u] > bounds[a + 3u] || bounds[a + 0u] > bounds[b + 3u] ||
             bounds[b + 1u] > bounds[a + 4u] || bounds[a + 1u] > bounds[b + 4u] ||
             bounds[b + 2u] > bounds[a + 5u] || bounds[a + 2u] > bounds[b + 5u]);
}

bool filtering(uint groupId, uint otherGroupId, uint envId, uint otherEnvId)
{
    return (groupId != otherGroupId) &&
           ((envId == otherEnvId) || (envId == PX_INVALID_U32) || (otherEnvId == PX_INVALID_U32));
}

// largest index into the 64-bit prefix s.t. value <= workIndex, else 0
uint binarySearch64(Ptr<uint64_t> data, uint numElements, uint64_t value)
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
    Ptr<PxgBroadPhaseDescGpu> desc = Ptr<PxgBroadPhaseDescGpu>(gBp.bpDesc);
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const uint warpIndex = tid / WARP;
    const uint lane = tid & (WARP - 1);

    const uint numHandles = desc[0].numPreviousHandles + desc[0].numCreatedHandles - desc[0].numRemovedHandles;
    const uint nbProjections = numHandles * 2u;

    Ptr<uint> newBounds = Ptr<uint>(desc[0].newIntegerBounds);                 // PxgIntegerAABB = 6 uints
    Ptr<uint> reports = Ptr<uint>(desc[0].foundPairReport);                    // PxgBroadPhasePair = 2 uints

    Ptr<uint> regionRange = Ptr<uint>(desc[0].regionRange);                    // PxgIntegerRegion = 8 uints
    Ptr<uint> boxGroups = Ptr<uint>(desc[0].updateData_groups);
    const uint64_t boxEnvIDsAddr = desc[0].updateData_envIDs;                  // may be NULL (.cu ptr branch)
    Ptr<uint> boxSapBox1DX = Ptr<uint>(desc[0].boxSapBox1D[0]);                // PxgSapBox1D = 2 uints, AXIS_X
    Ptr<uint> boxHandles = Ptr<uint>(desc[0].boxHandles[0]);                   // boxHandles[0][AXIS_X]

    // sStartEndPtHistograms[0]=active/[1]=start, sOrderedStartRegionHandles likewise
    Ptr<uint> activeHist = Ptr<uint>(desc[0].activeRegionsHistogram);
    Ptr<uint> startHist = Ptr<uint>(desc[0].startRegionsHistogram);
    Ptr<uint> orderedActive = Ptr<uint>(desc[0].orderedActiveRegionHandles);
    Ptr<uint> orderedStart = Ptr<uint>(desc[0].orderedStartRegionHandles);

    Ptr<uint64_t> overlapChecks = Ptr<uint64_t>(desc[0].overlapChecksRegion);  // regionOverlapType = PxU64
    const uint overlCheckesSize = desc[0].regionAccumTotal;
    Ptr<uint> handleRegions = Ptr<uint>(desc[0].overlapChecksHandleRegiones);  // PxgHandleRegion = 2 uints

    const uint64_t totalPair = desc[0].overlapChecksTotalRegion;               // u64 VALUE in the desc
    const uint maxPairs = desc[0].max_found_lost_pairs;

    const uint gridDimX = WorkgroupCount().x;
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
            const uint index = binarySearch64(overlapChecks, overlCheckesSize, workIndex);
            handle = handleRegions[index * 2u + 0u];               // .handleIndex
            const uint regionIndex = handleRegions[index * 2u + 1u]; // .regionIndex (= cell * nbProjections)

            const uint boxStartIndexX = boxSapBox1DX[handle * 2u + 0u];
            const uint boxEndIndexX = boxSapBox1DX[handle * 2u + 1u];
            const bool isNew = bpIsNewProjection(boxHandles[boxStartIndexX]);

            const uint64_t offset = workIndex - overlapChecks[index];

            Ptr<uint> startEndPtHistogram = isNew ? startHist : activeHist;
            Ptr<uint> orderedStartHandles = isNew ? orderedStart : orderedActive;

            const uint regionStartIndex = startEndPtHistogram[regionIndex + boxStartIndexX + 1u];
            const uint regionEndIndex = startEndPtHistogram[regionIndex + boxEndIndexX];

            if (regionEndIndex > regionStartIndex)
            {
                const uint64_t otherIndex64 = uint64_t(regionStartIndex) + offset;
                if (otherIndex64 < uint64_t(regionEndIndex))
                {
                    const uint otherIndex = uint(otherIndex64);
                    const uint group = boxGroups[handle];
                    const uint envId = boxEnvIDsAddr != 0 ? (Ptr<uint>(boxEnvIDsAddr))[handle] : PX_INVALID_U32;

                    otherHandle = orderedStartHandles[regionIndex + otherIndex];

                    // dedup: report the pair only in the region at the max of the
                    // two boxes' min corners
                    const uint rr = handle * 8u;
                    const uint orr = otherHandle * 8u;
                    const uint x = max(regionRange[rr + 0u], regionRange[orr + 0u]);
                    const uint y = max(regionRange[rr + 1u], regionRange[orr + 1u]);
                    const uint z = max(regionRange[rr + 2u], regionRange[orr + 2u]);
                    const uint otherRegionIndex = (x + y * REGION_SIZE_PER_AXIS
                                                   + z * (REGION_SIZE_PER_AXIS * REGION_SIZE_PER_AXIS))
                                                  * nbProjections;

                    if (regionIndex == otherRegionIndex)
                    {
                        const uint otherGroup = boxGroups[otherHandle];
                        const uint otherEnvId = boxEnvIDsAddr != 0 ? (Ptr<uint>(boxEnvIDsAddr))[otherHandle]
                                                                   : PX_INVALID_U32;
                        if (filtering(group, otherGroup, envId, otherEnvId))
                            foundCount = intersects(newBounds, handle, otherHandle) ? 1u : 0u;
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
                InterlockedAdd(desc[0].sharedFoundPairIndex, acc, base); // BDA atomic on the desc field
            gsBase = base;
        }
        GroupMemoryBarrierWithGroupSync();

        if (foundCount != 0u)
        {
            const uint position = gsBase + res + gsWarp[warpIndex];
            if (position < maxPairs)
            {
                // PxgBroadPhasePair's ctor canonicalizes to (min, max) so mVolA < mVolB
                reports[position * 2u + 0u] = min(handle, otherHandle);
                reports[position * 2u + 1u] = max(handle, otherHandle);
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
