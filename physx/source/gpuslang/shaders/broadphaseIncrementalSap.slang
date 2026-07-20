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
//
// P5 Step 4 (BDA, engine launch path): the kernel takes ONE PxgBroadPhaseDesc
// device pointer (bpDescd) and walks the byte-exact PxgBroadPhaseDescGpu mirror
// (pxgSlangBpDesc.slang, proven by testBpDescLayout).  Every axis-major flat
// input the bound-buffer form bound as a StructuredBuffer is a distinct device
// pointer inside the desc here (boxProjectionRanks[axis], boxHandles[0][axis],
// startPtHistogram/endPtHistogram/startPointHandles/endPointHandles[frame*3+
// axis], incrementalComparisons[axis], ...).  totalIncrementalComparisons is an
// inline desc field array.  The atomicAdd reservations become BDA InterlockedAdd
// directly on the desc's sharedFoundPairIndex/sharedLostPairIndex fields.  Grid
// is (WorkgroupCount().x, 3): gid.y selects the axis, gridDimX = WorkgroupCount()
// .x drives the grid-stride, and useEnvIds mirrors the .cu's `envIds ?` NULL
// check on updateData_envIDs.

import pxgSlangBpDesc;

static const uint BLOCK = 256;
static const uint WARP = 32;
static const uint WARPS_PER_BLOCK = BLOCK / WARP; // 8
static const uint PX_INVALID_U32 = 0xffffffffu;

// engine ABI: the PxgBroadPhaseDesc device pointer only
struct BpDescParams
{
    uint64_t bpDesc;
}

[[vk::push_constant]]
ConstantBuffer<BpDescParams> gBp;

bool isStartProjection(uint sortedHandle) { return (sortedHandle & 1u) != 0u; }
uint getHandle(uint sortedHandle) { return sortedHandle >> 3; }

// PxgIntegerAABB: mMinMax[6] = {minX,minY,minZ,maxX,maxY,maxZ}
struct IAABB
{
    uint m[6];
}

IAABB loadAABB(Ptr<uint> buf, uint idx)
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

// largest index into the sorted `data` array s.t. data[index] <= value;
// 0 if data[0] > value
uint binarySearch(Ptr<uint> data, uint numElements, uint value)
{
    uint left = 0;
    uint right = numElements;
    while (left < right)
    {
        const uint pos = (left + right) / 2u;
        const uint element = data[pos];
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

    Ptr<PxgBroadPhaseDescGpu> desc = Ptr<PxgBroadPhaseDescGpu>(gBp.bpDesc);
    const uint numHandles = desc[0].numPreviousHandles;
    const uint nbProjections = numHandles * 2u;
    const uint maxPairs = desc[0].max_found_lost_pairs;
    const uint gridDimX = WorkgroupCount().x;              // groups launched per axis
    const bool useEnvIds = desc[0].updateData_envIDs != 0; // .cu's `envIds ?` NULL check

    // per-axis device pointers (axis = gid.y)
    Ptr<uint> incComparisons = Ptr<uint>(desc[0].incrementalComparisons[axis]);
    Ptr<uint> ranksArr = Ptr<uint>(desc[0].boxProjectionRanks[axis]);
    Ptr<uint> boxHandlesArr = Ptr<uint>(desc[0].boxHandles[axis]); // boxHandles[0][axis]
    Ptr<uint> newBounds = Ptr<uint>(desc[0].newIntegerBounds);
    Ptr<uint> oldBounds = Ptr<uint>(desc[0].oldIntegerBounds);
    Ptr<uint> groupsArr = Ptr<uint>(desc[0].updateData_groups);
    Ptr<uint> envsArr = Ptr<uint>(desc[0].updateData_envIDs);
    Ptr<uint> foundReport = Ptr<uint>(desc[0].foundPairReport); // PxgBroadPhasePair = 2 uints
    Ptr<uint> lostReport = Ptr<uint>(desc[0].lostPairReport);

    const uint totalCmpsThisAxis = desc[0].totalIncrementalComparisons[axis];
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
            const uint index = binarySearch(incComparisons, nbProjections, workId);
            const uint sortedHandle = boxHandlesArr[index];
            const uint isStartHandle = isStartProjection(sortedHandle) ? 1u : 0u;
            handle = getHandle(sortedHandle);

            const uint groupId = groupsArr[handle];
            const uint envId = useEnvIds ? envsArr[handle] : PX_INVALID_U32;

            const uint offset = workId - incComparisons[index];
            const uint prevIndex = ranksArr[index];
            const uint downPass = prevIndex > index ? 1u : 0u;
            const uint startIndex = (downPass != 0u) ? index : (prevIndex + 1u);
            const uint willFindPair = (1u - isStartHandle) ^ downPass;

            // startEndAccum / startEndHandles = [isStartHandle][downPass]:
            //   start handle -> end-point arrays, else start-point arrays;
            //   down pass -> last frame (1), up pass -> this frame (0)
            const uint frame = downPass;
            const uint fa = (frame * 3u + axis);
            const uint accumVal = (isStartHandle != 0u)
                ? (Ptr<uint>(desc[0].endPtHistogram[fa]))[startIndex]
                : (Ptr<uint>(desc[0].startPtHistogram[fa]))[startIndex];
            const uint handleIndex = accumVal + offset;
            otherHandle = (isStartHandle != 0u)
                ? (Ptr<uint>(desc[0].endPointHandles[fa]))[handleIndex]
                : (Ptr<uint>(desc[0].startPointHandles[fa]))[handleIndex];

            const uint otherGroupId = groupsArr[otherHandle];
            const uint otherEnvId = useEnvIds ? envsArr[otherHandle] : PX_INVALID_U32;

            if (filtering(groupId, otherGroupId, envId, otherEnvId))
            {
                // invert new/old bounds based on up/down sweep so the overlap
                // logic reports the change on the correct frame
                const bool wfp = willFindPair != 0u;
                const IAABB aabb = wfp ? loadAABB(newBounds, handle) : loadAABB(oldBounds, handle);
                const IAABB oldAABB = wfp ? loadAABB(oldBounds, handle) : loadAABB(newBounds, handle);
                const IAABB otherAABB = wfp ? loadAABB(newBounds, otherHandle) : loadAABB(oldBounds, otherHandle);
                const IAABB otherOldAABB = wfp ? loadAABB(oldBounds, otherHandle) : loadAABB(newBounds, otherHandle);

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
                InterlockedAdd(desc[0].sharedFoundPairIndex, totalFound, baseFound); // BDA atomic on desc field
            if (totalLost > 0u)
                InterlockedAdd(desc[0].sharedLostPairIndex, totalLost, baseLost);
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
                // PxgBroadPhasePair canonicalizes to (min, max) so mVolA < mVolB
                const uint volA = min(handle, otherHandle);
                const uint volB = max(handle, otherHandle);
                if (foundOrLostID == 0u)
                {
                    foundReport[writeIndex * 2u] = volA;
                    foundReport[writeIndex * 2u + 1u] = volB;
                }
                else
                {
                    lostReport[writeIndex * 2u] = volA;
                    lostReport[writeIndex * 2u + 1u] = volB;
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
