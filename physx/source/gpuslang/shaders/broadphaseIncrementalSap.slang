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

// broadphaseIncrementalSap.slang
//
// Slang port of performIncrementalSAP (BP_INCREMENTAL_SAP) from
// gpubroadphase/src/CUDA/broadphase.cu.  This is the sweep-and-prune core of the
// incremental broad phase: it consumes the per-projection comparison prefix sum
// (incrementalComparisons / totalIncrementalComparisons, produced by
// broadphaseIncrementalHistogram) and the start/end-point histograms + ordered
// handles (produced by broadphaseEndPtHistogram), and emits the found (newly
// overlapping) and lost (newly separated) volume pairs.
//
// Grid: (gridDimX, 3, 1) — SV_GroupID.y selects the axis; block = 256 threads.
// For each axis, totalComparisons[axis] units of work are distributed over the
// grid via a grid-stride loop of 256-wide logical blocks.  Each work item:
//   1. binary-searches incrementalComparisons to find its projection index and
//      the offset within that projection's comparison range;
//   2. derives the swap partner (otherHandle) from the ordered start/end handle
//      arrays, selected by the projection's start/end kind and up/down pass
//      (down pass reads last frame's data, up pass this frame's);
//   3. tests, with the group/environment filter, whether the pair's overlap
//      status changed between the old and new integer AABBs — reporting it on
//      the first axis of old (found) / new (lost) separation for dedup;
//   4. compacts the found/lost hits within the block (groupshared 32-lane
//      scans) and atomically reserves a contiguous output range per category.
//
// Portability: the CUDA warpScanAddWriteToSharedMem / __ballot scans become
// groupshared 32-lane scans under uniform GroupMemoryBarrierWithGroupSync();
// the two per-thread category counts (found, lost) are packed into one uint
// (16-bit fields; per-block sum <= 256) so a single scan serves both.  The
// atomicAdd to the shared pair counters becomes InterlockedAdd.

static const uint BLOCK = 256;
static const uint WARP = 32;
static const uint WARPS_PER_BLOCK = BLOCK / WARP; // 8
static const uint PX_INVALID_U32 = 0xffffffffu;

struct SapParams
{
    uint numPreviousHandles; // nbProjections = numPreviousHandles * 2; #boxes = numPreviousHandles
    uint gridDimX;           // number of groups launched per axis
    uint maxFoundLostPairs;
    uint useEnvIds;          // 0/1
}

[[vk::push_constant]]
ConstantBuffer<SapParams> gParams;

// inputs (axis-major flat packing; frame 0 = current, frame 1 = last frame)
RWStructuredBuffer<uint> gTotalIncComparisons; // binding 0  [3]
RWStructuredBuffer<uint> gIncComparisons;      // binding 1  [3][nbProj]  exclusive prefix
RWStructuredBuffer<uint> gRanks;               // binding 2  [3][nbProj]  boxProjectionRanks
RWStructuredBuffer<uint> gBoxHandles;          // binding 3  [3][nbProj]  boxHandles[0][axis]
RWStructuredBuffer<uint> gStartHist;           // binding 4  [2][3][nbProj]
RWStructuredBuffer<uint> gEndHist;             // binding 5  [2][3][nbProj]
RWStructuredBuffer<uint> gStartHandles;        // binding 6  [2][3][nH]
RWStructuredBuffer<uint> gEndHandles;          // binding 7  [2][3][nH]
RWStructuredBuffer<uint> gNewBounds;           // binding 8  [nH][6]  newIntegerBounds
RWStructuredBuffer<uint> gOldBounds;           // binding 9  [nH][6]  oldIntegerBounds
RWStructuredBuffer<uint> gGroups;              // binding 10 [nH]
RWStructuredBuffer<uint> gEnvs;                // binding 11 [nH]
// outputs
RWStructuredBuffer<uint> gFoundReport;         // binding 12 [maxPairs][2]  (mVolA, mVolB)
RWStructuredBuffer<uint> gLostReport;          // binding 13 [maxPairs][2]
RWStructuredBuffer<uint> gSharedFound;         // binding 14 [1]  atomic counter
RWStructuredBuffer<uint> gSharedLost;          // binding 15 [1]  atomic counter

bool isStartProjection(uint sortedHandle) { return (sortedHandle & 1u) != 0u; }
uint getHandle(uint sortedHandle) { return sortedHandle >> 3; }

// PxgIntegerAABB: mMinMax[6] = {minX,minY,minZ,maxX,maxY,maxZ}
struct IAABB
{
    uint m[6];
}

IAABB loadAABB(RWStructuredBuffer<uint> buf, uint idx)
{
    IAABB r;
    const uint base = idx * 6u;
    [unroll]
    for (uint i = 0; i < 6; ++i)
        r.m[i] = buf[base + i];
    return r;
}

bool intersects(IAABB a, IAABB b)
{
    return !(b.m[0] > a.m[3] || a.m[0] > b.m[3] ||
             b.m[1] > a.m[4] || a.m[1] > b.m[4] ||
             b.m[2] > a.m[5] || a.m[2] > b.m[5]);
}

bool intersects1D(IAABB a, IAABB b, uint axis)
{
    const uint maxAxis = axis + 3u;
    return !(b.m[axis] > a.m[maxAxis] || a.m[axis] > b.m[maxAxis]);
}

// separated on any axis strictly before `axis` (dedup: report a changed pair on
// its first axis of separation)
bool isSeparatedBeforeAxis(uint axis, IAABB a0, IAABB a1)
{
    if (axis == 2u)
    {
        if (!intersects1D(a0, a1, 1u)) return true;
        if (!intersects1D(a0, a1, 0u)) return true;
    }
    else if (axis == 1u)
    {
        if (!intersects1D(a0, a1, 0u)) return true;
    }
    return false;
}

bool filtering(uint groupId, uint otherGroupId, uint envId, uint otherEnvId)
{
    return (groupId != otherGroupId) &&
           ((envId == otherEnvId) || (envId == PX_INVALID_U32) || (otherEnvId == PX_INVALID_U32));
}

// largest index into the sorted `data` (base-offset) array s.t. data[index] <= value;
// 0 if data[0] > value
uint binarySearch(uint base, uint numElements, uint value)
{
    uint left = 0;
    uint right = numElements;
    while (left < right)
    {
        const uint pos = (left + right) / 2u;
        const uint element = gIncComparisons[base + pos];
        if (element <= value)
            left = pos + 1u;
        else
            right = pos;
    }
    return left != 0u ? left - 1u : 0u;
}

groupshared uint gsAcc[BLOCK];          // per-lane packed (found | lost<<16) scan
groupshared uint gsWarp[WARPS_PER_BLOCK]; // packed warp totals -> exclusive prefixes
groupshared uint gsBaseFound;
groupshared uint gsBaseLost;

// inclusive scan of gsAcc within each 32-lane warp (packed; each 16-bit field's
// per-warp sum <= 32 so no carry crosses the boundary)
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
void performIncrementalSAP(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint axis = gid.y;
    const uint warpIndex = tid / WARP;
    const uint lane = tid & (WARP - 1);

    const uint numHandles = gParams.numPreviousHandles;
    const uint nbProjections = numHandles * 2u;
    const uint maxPairs = gParams.maxFoundLostPairs;
    const uint gridDimX = gParams.gridDimX;
    const bool useEnvIds = gParams.useEnvIds != 0u;

    const uint axisProjBase = axis * nbProjections;
    const uint totalCmpsThisAxis = gTotalIncComparisons[axis];
    const uint nbBlocksRequired = (totalCmpsThisAxis + BLOCK - 1u) / BLOCK;

    // grid-stride loop over 256-wide logical blocks assigned to this group
    for (uint blockId = gid.x; blockId < nbBlocksRequired; blockId += gridDimX)
    {
        const uint workId = blockId * BLOCK + tid;

        uint handle = PX_INVALID_U32;
        uint otherHandle = PX_INVALID_U32;
        uint foundOrLostID = 0;   // 0 = found, 1 = lost
        uint foundOrLostPair = 0; // 0/1

        if (workId < totalCmpsThisAxis)
        {
            const uint index = binarySearch(axisProjBase, nbProjections, workId);
            const uint sortedHandle = gBoxHandles[axisProjBase + index];
            const uint isStartHandle = isStartProjection(sortedHandle) ? 1u : 0u;
            handle = getHandle(sortedHandle);

            const uint groupId = gGroups[handle];
            const uint envId = useEnvIds ? gEnvs[handle] : PX_INVALID_U32;

            const uint offset = workId - gIncComparisons[axisProjBase + index];
            const uint prevIndex = gRanks[axisProjBase + index];
            const uint downPass = prevIndex > index ? 1u : 0u;
            const uint startIndex = (downPass != 0u) ? index : (prevIndex + 1u);
            const uint willFindPair = (1u - isStartHandle) ^ downPass;

            // startEndAccum / startEndHandles = [isStartHandle][downPass]:
            //   start handle -> end-point arrays, else start-point arrays;
            //   down pass -> last frame (1), up pass -> this frame (0)
            const uint frame = downPass;
            const uint fa = (frame * 3u + axis);
            const uint accumVal = (isStartHandle != 0u)
                ? gEndHist[fa * nbProjections + startIndex]
                : gStartHist[fa * nbProjections + startIndex];
            const uint handleIndex = accumVal + offset;
            otherHandle = (isStartHandle != 0u)
                ? gEndHandles[fa * numHandles + handleIndex]
                : gStartHandles[fa * numHandles + handleIndex];

            const uint otherGroupId = gGroups[otherHandle];
            const uint otherEnvId = useEnvIds ? gEnvs[otherHandle] : PX_INVALID_U32;

            if (filtering(groupId, otherGroupId, envId, otherEnvId))
            {
                // invert new/old bounds based on up/down sweep so the overlap
                // logic reports the change on the correct frame
                const bool wfp = willFindPair != 0u;
                const IAABB aabb = wfp ? loadAABB(gNewBounds, handle) : loadAABB(gOldBounds, handle);
                const IAABB oldAABB = wfp ? loadAABB(gOldBounds, handle) : loadAABB(gNewBounds, handle);
                const IAABB otherAABB = wfp ? loadAABB(gNewBounds, otherHandle) : loadAABB(gOldBounds, otherHandle);
                const IAABB otherOldAABB = wfp ? loadAABB(gOldBounds, otherHandle) : loadAABB(gNewBounds, otherHandle);

                if (intersects(aabb, otherAABB) && !intersects1D(oldAABB, otherOldAABB, axis))
                {
                    if (!isSeparatedBeforeAxis(axis, oldAABB, otherOldAABB))
                    {
                        foundOrLostID = 1u - willFindPair;
                        foundOrLostPair = 1u;
                    }
                }
            }
        }

        // pack the two category counts: bits [0,16) = found, [16,32) = lost
        const uint val = (foundOrLostPair != 0u)
            ? ((foundOrLostID == 0u) ? 1u : (1u << 16))
            : 0u;

        gsAcc[tid] = val;
        GroupMemoryBarrierWithGroupSync();
        warpScanAcc(tid);
        const uint res = gsAcc[tid] - val; // exclusive prefix within warp (packed)
        if (lane == WARP - 1)
            gsWarp[warpIndex] = gsAcc[tid]; // inclusive warp total (packed)
        GroupMemoryBarrierWithGroupSync();

        // exclusive prefix over the 8 warp totals + atomic reservation
        if (tid == 0)
        {
            uint acc = 0;
            [unroll]
            for (uint w = 0; w < WARPS_PER_BLOCK; ++w)
            {
                const uint t = gsWarp[w];
                gsWarp[w] = acc; // exclusive prefix (packed)
                acc += t;
            }
            const uint totalFound = acc & 0xffffu;
            const uint totalLost = acc >> 16;
            uint baseFound = 0, baseLost = 0;
            if (totalFound > 0u)
                InterlockedAdd(gSharedFound[0], totalFound, baseFound);
            if (totalLost > 0u)
                InterlockedAdd(gSharedLost[0], totalLost, baseLost);
            gsBaseFound = baseFound;
            gsBaseLost = baseLost;
        }
        GroupMemoryBarrierWithGroupSync();

        if (foundOrLostPair != 0u)
        {
            uint writeIndex;
            if (foundOrLostID == 0u)
                writeIndex = gsBaseFound + (res & 0xffffu) + (gsWarp[warpIndex] & 0xffffu);
            else
                writeIndex = gsBaseLost + (res >> 16) + (gsWarp[warpIndex] >> 16);

            if (writeIndex < maxPairs)
            {
                if (foundOrLostID == 0u)
                {
                    gFoundReport[writeIndex * 2u] = handle;
                    gFoundReport[writeIndex * 2u + 1u] = otherHandle;
                }
                else
                {
                    gLostReport[writeIndex * 2u] = handle;
                    gLostReport[writeIndex * 2u + 1u] = otherHandle;
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
