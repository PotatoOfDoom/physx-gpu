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

// aggregateBda.slang -- P9.92: the GPU broadphase's AGGREGATE path on the engine launch
// ABI, all fourteen kernels of gpubroadphase/src/CUDA/aggregate.cu as PxgAABBManager
// launches them (void** form, KERNEL_PARAM_TYPE packs):
//
//   updateDirtyAggregate / clearDirtyAggregates        (16 warps per block)
//   markAggregateBoundsUpdatedBitmap                    (8 warps)
//   updateAggregateBounds                               (16 warps)
//   markAddedAndDeletedAggregatedBounds                 (128 x 2 threads)
//   sortAndUpdateAggregateProjections                   (BP_AGGREGATE_SORT / 32 = 2 warps)
//   doSelfCollision                                     (16 warps)
//   addAggPairsStage1 / addAggPairsStage2               (64 x 1024 threads)
//   doAggPairCollisions                                 (2 warps, grid 8192)
//   removeAggPairsStage1 / 2 (32 x 1024) / 3 (1 x 32)
//   aggCopyReports                                      (64 x 256)
//
// Every buffer is reached through the engine's own PxgAggregateDesc / PxgBroadPhaseDesc
// (pxgSlangBpDesc.slang) and PxgAggregate records by buffer device address.  The Phase-1
// binding ports (broadphaseAgg*.slang, broadphaseAggregate*.slang) proved the algorithms
// on flat pooled layouts; this file is what the manifest maps.
//
// ONE WORKGROUP PER CUDA WARP.  The CUDA kernels are warp-per-item with blockDim.y warps
// per block and only warp-level synchronization (__syncwarp, ballots, shuffles; per-warp
// __shared__ slots, no __syncthreads).  Here a workgroup is ONE warp of 32 lanes.  The
// block's warps used to be walked in turn inside one group; the launch sites now
// (PxgAABBManager.cpp, PXG_NO_CUDA_TOOLKIT) launch one group per CUDA warp,
// numBlocks * blockDim.y of them, so the group index IS the CUDA's globalWarpIndex (the
// pattern of softBodyMeshMidphaseBda / convexTrimeshMidphaseBda).  The block's warps never
// talk to each other in these kernels, so running them as separate groups changes nothing
// but the order of the global atomics that allocate report slots -- which is unordered
// across warps in the CUDA too.  The kernels without a CUDA grid-stride loop take their
// items in a group-stride loop (stride WorkgroupCount().x): one pass at the port's grid,
// and every item at any smaller grid (at the CUDA's own grid a group takes blockDim.y
// items one grid-width apart, not its own block's).  The loop index is clamped for
// the loads: llvmpipe can run a barrier-bearing loop body once more with an EMPTY mask
// and still issue them (P9.22, P9.49), and an unclamped index would read an aggregate
// record past the array and follow its pointers.  Every warp primitive is
// the groupshared emulation, so every loop that contains one has a warp-uniform trip
// count, exactly as the CUDA requires of its FULL_MASK primitives.
//
// Sub-word stores (the PxgAggregate::isNew, PxgAggregatePair::isNew/isDead and the
// aggregate desc's three overflow bools) are the InterlockedAnd/InterlockedOr pair
// (check_subword_store.py).

import pxgSlangBpDesc;
import pxgSlangScan;
import pxgSlangRadixWarp;

static const uint WARP = 32u;
// The CUDA blockDim.y of each launch; the port's launch sites multiply their grid by it
// (one group per CUDA warp, see above).
static const uint W_DIRTY = 16u;      // updateDirtyAggregate / clearDirtyAggregates
static const uint W_BITMAP = 8u;      // markAggregateBoundsUpdatedBitmap
static const uint W_BOUNDS = 16u;     // updateAggregateBounds
static const uint W_SORT = 2u;        // BP_AGGREGATE_SORT (64) / WARP_SIZE
static const uint W_SELF = 16u;       // doSelfCollision
static const uint W_PAIR = 2u;        // doAggPairCollisions: BP_AGGREGATE_SORT / WARP_SIZE
static const uint ADD_GRID = 64u;     // addAggPairsStage1 grid
static const uint ADD_BLOCK = 1024u;  // addAggPairsStage1 block
static const uint REMOVE_GRID = 32u;  // PxgBPKernelGridDim::BP_AGGREGATE_REMOVE
static const uint REMOVE_BLOCK = 1024u; // PxgBPKernelBlockDim::BP_AGGREGATE_REMOVE
static const uint COPY_GRID = 64u;    // aggCopyReports
static const uint COPY_BLOCK = 256u;

// ---------------------------------------------------------------------------
// the engine structs (gpubroadphase/include/PxgAggregate.h, PxgAggregateDesc.h)
// ---------------------------------------------------------------------------

// PxgAggregate, 136 bytes
struct PxgAggregateGpu
{
    uint64_t updateBoundIndices;   // @0
    uint64_t boundIndices0;        // @8
    uint64_t boundIndices1;        // @16
    uint64_t sortedProjections0;   // @24
    uint64_t sortedProjections1;   // @32
    uint64_t sortedHandles0;       // @40
    uint64_t sortedHandles1;       // @48
    uint64_t sapBox1D0;            // @56
    uint64_t sapBox1D1;            // @64
    uint64_t startMasks0;          // @72
    uint64_t startMasks1;          // @80
    uint64_t comparisons0;         // @88
    uint64_t comparisons1;         // @96
    uint prevComparisons;          // @104
    uint prevSize;                 // @108
    uint mIndex;                   // @112
    uint size;                     // @116
    uint filterHint;               // @120
    uint mEnvID;                   // @124
    uint isNewWord;                // @128  bool isNew in byte 0, then 3 pad bytes
    uint pad132;                   // @132
}
static const uint AGG_WORDS = 34u;       // sizeof(PxgAggregate) / 4
static const uint AGG_ISNEW_OFF = 128u;

// PxgAggregateDesc, 136 bytes
struct PxgAggregateDescGpu
{
    uint64_t foundPairReport;      // @0   PxgBroadPhasePair*
    uint64_t lostPairReport;       // @8
    uint64_t foundPairReportMap;   // @16
    uint64_t lostPairReportMap;    // @24
    uint sharedFoundPairIndex;     // @32
    uint sharedLostPairIndex;      // @36
    uint64_t aggregates;           // @40  PxgAggregate*
    uint numAgregates;             // @48
    uint pad52;
    uint64_t freeBufferList;       // @56  PxgFreeBufferList* {numFreeIndices, maxIndex}
    uint64_t freeIndices;          // @64
    uint64_t removeBitmap;         // @72  PxU32*
    uint64_t removeHistogram;      // @80
    uint nbRemoved;                // @88
    uint pad92;
    uint64_t aggPairs;             // @96  PxgAggregatePair*
    uint64_t aggPairCount;         // @104 PxU32*
    uint max_found_lost_pairs;     // @112
    uint max_agg_pairs;            // @116
    uint aggPairOverflowCount;     // @120
    uint foundCandidatePairOverflowCount; // @124
    uint flagsWord;                // @128 found_pairs_overflow_flags, lost_..., agg_pairs_overflow_flags (bytes 0..2)
    uint pad132;
}
static const uint DESC_FLAGS_OFF = 128u;

// PxgAggregatePair, 12 bytes: actorHandle0, actorHandle1, bool isNew (byte 8),
// bool isDead (byte 9), PxU16 pad
struct PxgAggregatePairGpu
{
    uint actorHandle0;
    uint actorHandle1;
    uint flagsWord;
}

// Bp::VolumeData, 16 bytes: void* mUserData, AggregateHandle mAggregate
struct VolumeDataGpu
{
    uint64_t userData;
    uint aggregate;
    uint pad;
}

// PxgIntegerAABB: mMinMax[6] = min x, y, z, max x, y, z
struct IAABB
{
    uint v[6];
}

bool iaabbIntersects(IAABB a, IAABB b)
{
    return !(b.v[0] > a.v[3] || a.v[0] > b.v[3] ||
             b.v[1] > a.v[4] || a.v[1] > b.v[4] ||
             b.v[2] > a.v[5] || a.v[2] > b.v[5]);
}

IAABB loadIAABB(uint64_t base, uint i) { return Ptr<IAABB>(base)[i]; }

// PxgIntegerAABB::setEmpty
void storeEmptyIAABB(uint64_t base, uint i)
{
    Ptr<uint> w = Ptr<uint>(base + uint64_t(i) * 24UL);
    w[0] = 0xff7fffffu; w[1] = 0xff7fffffu; w[2] = 0xff7fffffu;
    w[3] = 0x00800000u; w[4] = 0x00800000u; w[5] = 0x00800000u;
}

// ---------------------------------------------------------------------------
// the byte stores (bool fields) and loads
// ---------------------------------------------------------------------------
void storeByte(uint64_t addr, uint value)
{
    Ptr<uint> w = Ptr<uint>(addr & ~uint64_t(3));
    const uint shift = uint(addr & 3UL) * 8u;
    uint prev;
    InterlockedAnd(w[0], ~(0xffu << shift), prev);
    if ((value & 0xffu) != 0u)
        InterlockedOr(w[0], (value & 0xffu) << shift, prev);
}

bool loadBool(uint64_t addr)
{
    const uint word = Ptr<uint>(addr & ~uint64_t(3))[0];
    return ((word >> (uint(addr & 3UL) * 8u)) & 0xffu) != 0u;
}

uint64_t aggAddr(uint64_t aggregates, uint i) { return aggregates + uint64_t(i) * 136UL; }
bool aggIsNew(uint64_t a) { return loadBool(a + uint64_t(AGG_ISNEW_OFF)); }
bool aggIsValid(PxgAggregateGpu agg) { return agg.size > 0u && agg.mIndex != 0xffffffffu; }

// the [2] member arrays, by index
uint64_t aggBoundIndices(PxgAggregateGpu a, uint k)     { return k == 0u ? a.boundIndices0 : a.boundIndices1; }
uint64_t aggSortedProjections(PxgAggregateGpu a, uint k) { return k == 0u ? a.sortedProjections0 : a.sortedProjections1; }
uint64_t aggSortedHandles(PxgAggregateGpu a, uint k)    { return k == 0u ? a.sortedHandles0 : a.sortedHandles1; }
uint64_t aggSapBox1D(PxgAggregateGpu a, uint k)         { return k == 0u ? a.sapBox1D0 : a.sapBox1D1; }
uint64_t aggStartMasks(PxgAggregateGpu a, uint k)       { return k == 0u ? a.startMasks0 : a.startMasks1; }
uint64_t aggComparisons(PxgAggregateGpu a, uint k)      { return k == 0u ? a.comparisons0 : a.comparisons1; }

// ---------------------------------------------------------------------------
// warp-32 primitives for a 32-lane workgroup (every call warp-uniform)
// ---------------------------------------------------------------------------
groupshared uint gsLane[WARP];

// __ballot_sync(FULL_MASK, pred)
uint warpBallot(uint lane, bool pred)
{
    gsLane[lane] = pred ? 1u : 0u;
    GroupMemoryBarrierWithGroupSync();
    uint mask = 0u;
    for (uint l = 0u; l < WARP; ++l)
        mask |= gsLane[l] << l;
    GroupMemoryBarrierWithGroupSync();
    return mask;
}

// __shfl_sync(FULL_MASK, v, src)
uint warpShfl(uint lane, uint v, uint src)
{
    gsLane[lane] = v;
    GroupMemoryBarrierWithGroupSync();
    const uint r = gsLane[src & (WARP - 1u)];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

// __shfl_xor_sync(FULL_MASK, v, r) on a float, bit-exact
float warpShflXorF(uint lane, float v, uint r)
{
    return asfloat(warpShfl(lane, asuint(v), lane ^ r));
}

// warpScan<AddOpPxU32>: inclusive over the 32 lanes (reduction.cuh up-shuffle scan)
uint warpScanAdd(uint lane, uint v)
{
    return warp32InclusiveScan(lane, v);
}

uint popc(uint v) { return countbits(v); }

// PxMin / PxMax on floats: selectMin(a, b) = a < b ? a : b, selectMax(a, b) = a > b ? a : b
float selMin(float a, float b) { return (a < b) ? a : b; }
float selMax(float a, float b) { return (a > b) ? a : b; }

// reduction.cuh:729 binarySearch: the largest index with data[index] <= value, or 0
uint binarySearchU(Ptr<uint> data, uint numElements, uint value)
{
    uint left = 0u;
    uint right = numElements;
    while (left < right)
    {
        const uint pos = (left + right) / 2u;
        if (data[pos] <= value) left = pos + 1u; else right = pos;
    }
    return left != 0u ? left - 1u : 0u;
}

// ---------------------------------------------------------------------------
// the bit searches (aggregate.cu:371-465)
// ---------------------------------------------------------------------------
// `2 << b` for b = 31 is 0 in PTX (shl.b32 of 2 by 31 overflows out of the word), so
// `(2 << b) - 1` is all ones; the same holds for uint arithmetic here.
uint maskUpTo(uint b) { return (2u << b) - 1u; }

uint getIthBit(uint mask, uint i)
{
    uint lower = i;
    uint upper = 32u;
    const uint bitTarget = i + 1u;
    while ((upper - lower) > 1u)
    {
        const uint midPoint = (lower + upper) / 2u;
        const uint nbBits = popc(mask & maskUpTo(midPoint));
        const uint active = mask & (1u << midPoint);
        if (nbBits > bitTarget || (active == 0u && nbBits == bitTarget))
            upper = midPoint;
        else
            lower = midPoint;
    }
    return lower;
}

uint getIndex(Ptr<uint> startMasks, uint startIdx, uint targetIndex, uint baseIndex)
{
    const uint startWord = startIdx / WARP;
    const uint mask = ~((1u << (startIdx & 31u)) - 1u);
    uint count = 0u;
    const uint target = targetIndex - baseIndex;
    uint w = startWord;
    uint maskedResult = startMasks[w] & mask;
    for (;;)
    {
        const uint newCount = count + popc(maskedResult);
        if (newCount > target)
            break;
        count = newCount;
        maskedResult = startMasks[w + 1u];
        ++w;
    }
    const uint offset = target - count;
    const uint bit = getIthBit(maskedResult, offset);
    return w * WARP + bit;
}

// ===========================================================================
// the dirty-aggregate upload (PxgAABBManager::updateDescriptor / clearDirtyAggs)
// ===========================================================================

// updateDirtyAggregate(PxgAggregate* aggregates, PxU32 numAggregates,
//   PxU32* dirtyAggregateIndices, PxgAggregate* dirtyAggregates, PxU32 numDirtyAggregates,
//   const PxU32* dirtyBoundIndices, const PxU32* dirtyBoundStartIndices)
//   -> {8,4,8,8,4,8,8} = 56-byte blob
struct UpdateDirtyParams
{
    uint64_t aggregates;
    uint numAggregates;
    uint pad0;
    uint64_t dirtyAggregateIndices;
    uint64_t dirtyAggregates;
    uint numDirtyAggregates;
    uint pad1;
    uint64_t dirtyBoundIndices;
    uint64_t dirtyBoundStartIndices;
}
[[vk::push_constant]] ConstantBuffer<UpdateDirtyParams> gUd;

// aggregate.cu:54 -- one warp per dirty aggregate
[shader("compute")]
[numthreads(32, 1, 1)]
void updateDirtyAggregate(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;
    for (uint itemLoop = gid.x; itemLoop < gUd.numDirtyAggregates; itemLoop += WorkgroupCount().x)
    {
        // The P9.22 clamp (see the header): live iterations have globalWarpIndex == itemLoop.
        const uint globalWarpIndex = min(itemLoop, max(gUd.numDirtyAggregates, 1u) - 1u);

        const uint index = Ptr<uint>(gUd.dirtyAggregateIndices)[globalWarpIndex];
        const uint64_t agg = aggAddr(gUd.aggregates, index);
        const uint64_t dirtyAgg = aggAddr(gUd.dirtyAggregates, globalWarpIndex);
        const PxgAggregateGpu d = Ptr<PxgAggregateGpu>(dirtyAgg)[0];

        const uint startIndex = Ptr<uint>(gUd.dirtyBoundStartIndices)[globalWarpIndex];
        for (uint i = lane; i < d.size; i += WARP)
            Ptr<uint>(d.updateBoundIndices)[i] = Ptr<uint>(gUd.dirtyBoundIndices)[startIndex + i];

        // every lane reads isNew before lane 0 may change the record below
        const bool isNew = aggIsNew(agg);
        GroupMemoryBarrierWithGroupSync();
        if (isNew)
        {
            // warpCopy<uint>(dAgg, sAgg, sizeof(PxgAggregate)): all 34 words, pad included
            for (uint i = lane; i < AGG_WORDS; i += WARP)
                Ptr<uint>(agg)[i] = Ptr<uint>(dirtyAgg)[i];
        }
        else if (lane == 0u)
        {
            Ptr<PxgAggregateGpu> a = Ptr<PxgAggregateGpu>(agg);
            a[0].prevSize = a[0].size;
            a[0].updateBoundIndices = d.updateBoundIndices;
            a[0].mIndex = d.mIndex;
            a[0].size = d.size;
            a[0].filterHint = d.filterHint;
        }
        AllMemoryBarrierWithGroupSync();
    }
}

// clearDirtyAggregates(PxgAggregate* aggregates, PxU32 numAggregates,
//   PxU32* dirtyAggregateIndices, PxU32 numDirtyAggregates) -> {8,4,8,4} = 32-byte blob
struct ClearDirtyParams
{
    uint64_t aggregates;
    uint numAggregates;
    uint pad0;
    uint64_t dirtyAggregateIndices;
    uint numDirtyAggregates;
    uint pad1;
}
[[vk::push_constant]] ConstantBuffer<ClearDirtyParams> gCd;

// aggregate.cu:117
[shader("compute")]
[numthreads(32, 1, 1)]
void clearDirtyAggregates(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    if (gtid.x != 0u)
        return;
    for (uint itemLoop = gid.x; itemLoop < gCd.numDirtyAggregates; itemLoop += WorkgroupCount().x)
    {
        // The P9.22 clamp (see the header): live iterations have globalWarpIndex == itemLoop.
        const uint globalWarpIndex = min(itemLoop, max(gCd.numDirtyAggregates, 1u) - 1u);
        const uint index = Ptr<uint>(gCd.dirtyAggregateIndices)[globalWarpIndex];
        const uint64_t agg = aggAddr(gCd.aggregates, index);
        storeByte(agg + uint64_t(AGG_ISNEW_OFF), 0u);
        Ptr<PxgAggregateGpu> a = Ptr<PxgAggregateGpu>(agg);
        a[0].prevSize = a[0].size;
    }
}

// ===========================================================================
// the aggregate bounds (computeAggregateBounds / markAggregateBoundsBitmap)
// ===========================================================================

// markAggregateBoundsUpdatedBitmap(PxgAggregateDesc* aggDesc, PxU32* changedAABBMgrHandles)
struct TwoPtrParams
{
    uint64_t p0;
    uint64_t p1;
}
[[vk::push_constant]] ConstantBuffer<TwoPtrParams> gTwo;

// aggregate.cu:143 -- the aggregate's own bound becomes changed, its members' bits clear
[shader("compute")]
[numthreads(32, 1, 1)]
void markAggregateBoundsUpdatedBitmap(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;
    const PxgAggregateDescGpu desc = Ptr<PxgAggregateDescGpu>(gTwo.p0)[0];
    Ptr<uint> changed = Ptr<uint>(gTwo.p1);
    for (uint itemLoop = gid.x; itemLoop < desc.numAgregates; itemLoop += WorkgroupCount().x)
    {
        // The P9.22 clamp (see the header): live iterations have warpIdx == itemLoop.
        const uint warpIdx = min(itemLoop, max(desc.numAgregates, 1u) - 1u);
        const PxgAggregateGpu agg = Ptr<PxgAggregateGpu>(aggAddr(desc.aggregates, warpIdx))[0];
        if (!aggIsValid(agg))
            continue;
        uint prev;
        if (lane == 0u)
            InterlockedOr(changed[agg.mIndex / 32u], 1u << (agg.mIndex % 32u), prev);
        for (uint id = lane; id < agg.size; id += WARP)
        {
            const uint idx = Ptr<uint>(agg.updateBoundIndices)[id];
            InterlockedAnd(changed[idx / 32u], ~(1u << (idx % 32u)), prev);
        }
    }
}

// updateAggregateBounds(PxgAggregateDesc* aggDesc, PxBounds3* bounds, PxReal* contactDistance)
struct ThreePtrParams
{
    uint64_t p0;
    uint64_t p1;
    uint64_t p2;
}
[[vk::push_constant]] ConstantBuffer<ThreePtrParams> gThree;

// aggregate.cu:177 -- the union of the members' inflated bounds.  Each lane folds its
// members with PxVec3::minimum/maximum (PxMin/PxMax selects, running value first), then
// the xor butterfly (radius 1, 2, 4, 8, 16) folds PxMin(partner, own) -- the partner's
// value is the FIRST operand, so on a +-0 tie or a NaN the lane keeps its own -- and lane
// 0 stores.
[shader("compute")]
[numthreads(32, 1, 1)]
void updateAggregateBounds(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;
    const PxgAggregateDescGpu desc = Ptr<PxgAggregateDescGpu>(gThree.p0)[0];
    Ptr<float> bounds = Ptr<float>(gThree.p1);
    Ptr<float> contactDistance = Ptr<float>(gThree.p2);
    for (uint itemLoop = gid.x; itemLoop < desc.numAgregates; itemLoop += WorkgroupCount().x)
    {
        // The P9.22 clamp (see the header): live iterations have warpIdx == itemLoop.
        const uint warpIdx = min(itemLoop, max(desc.numAgregates, 1u) - 1u);
        const PxgAggregateGpu agg = Ptr<PxgAggregateGpu>(aggAddr(desc.aggregates, warpIdx))[0];
        if (!aggIsValid(agg))
            continue;

        // PxBounds3::setEmpty: (PX_MAX_BOUNDS_EXTENTS, -PX_MAX_BOUNDS_EXTENTS) = 0x7e7fffff
        const float ext = asfloat(0x7e7fffffu);
        float mn[3] = { ext, ext, ext };
        float mx[3] = { -ext, -ext, -ext };
        for (uint id = lane; id < agg.size; id += WARP)
        {
            const uint idx = Ptr<uint>(agg.updateBoundIndices)[id];
            const float inflate = contactDistance[idx];
            for (uint c = 0u; c < 3u; ++c)
            {
                mn[c] = selMin(mn[c], bounds[idx * 6u + c] - inflate);
                mx[c] = selMax(mx[c], bounds[idx * 6u + 3u + c] + inflate);
            }
        }
        for (uint r = 1u; r < WARP; r *= 2u)
        {
            for (uint c = 0u; c < 3u; ++c)
            {
                mn[c] = selMin(warpShflXorF(lane, mn[c], r), mn[c]);
                mx[c] = selMax(warpShflXorF(lane, mx[c], r), mx[c]);
            }
        }
        if (lane == 0u)
        {
            for (uint c = 0u; c < 3u; ++c)
            {
                bounds[agg.mIndex * 6u + c] = mn[c];
                bounds[agg.mIndex * 6u + 3u + c] = mx[c];
            }
        }
    }
}

// markAddedAndDeletedAggregatedBounds(PxgBroadPhaseDesc* bpDesc, PxgAggregateDesc* aggDesc,
//   PxU32* deletedAggregatedBounds, PxU32 numDeletedAggregatedBounds,
//   PxU32* addedAggregatedBounds, PxU32 numAddedAggregatedBounds) -> {8,8,8,4,8,4} = 48 bytes
struct MarkAddedDeletedParams
{
    uint64_t bpDesc;
    uint64_t aggDesc;
    uint64_t deleted;
    uint numDeleted;
    uint pad0;
    uint64_t added;
    uint numAdded;
    uint pad1;
}
[[vk::push_constant]] ConstantBuffer<MarkAddedDeletedParams> gMad;

// aggregate.cu:1427, block (128, 2): row 0 the added bounds, row 1 the deleted ones.  A
// deleted bound loses both its new and its old integer bounds, an added one only its old.
[shader("compute")]
[numthreads(128, 2, 1)]
void markAddedAndDeletedAggregatedBounds(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint globalThreadIdx = gtid.x + gid.x * 128u;
    const bool deletedRow = gtid.y != 0u;
    const uint maxCount = deletedRow ? gMad.numDeleted : gMad.numAdded;
    if (globalThreadIdx >= maxCount)
        return;
    const PxgBroadPhaseDescGpu bp = Ptr<PxgBroadPhaseDescGpu>(gMad.bpDesc)[0];
    const uint handle = Ptr<uint>(deletedRow ? gMad.deleted : gMad.added)[globalThreadIdx];
    if (deletedRow)
        storeEmptyIAABB(bp.newIntegerBounds, handle);
    storeEmptyIAABB(bp.oldIntegerBounds, handle);
}

// ===========================================================================
// the per-aggregate SAP structure and self-collision
// ===========================================================================

// (PxgBroadPhaseDesc* desc, PxgAggregateDesc* aggDesc) -> kTwoPtr; reuses gTwo:
//   p0 = the broadphase desc, p1 = the aggregate desc

// aggregate.cu:231.  Flips the double buffers (slot 1 always holds the previous frame's
// structure), rebuilds the member projections on x from the new integer bounds, sorts them
// with radixSortSingleWarp -- eight passes whose temp half is the second half of the same
// arrays, so both halves end up written exactly as the CUDA writes them -- and records each
// member's start/end position (sapBox1D) and the start-projection bit masks.
[shader("compute")]
[numthreads(32, 1, 1)]
void sortAndUpdateAggregateProjections(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;
    const PxgBroadPhaseDescGpu bp = Ptr<PxgBroadPhaseDescGpu>(gTwo.p0)[0];
    const PxgAggregateDescGpu desc = Ptr<PxgAggregateDescGpu>(gTwo.p1)[0];
    for (uint itemLoop = gid.x; itemLoop < desc.numAgregates; itemLoop += WorkgroupCount().x)
    {
        // The P9.22 clamp (see the header): live iterations have globalWarpIndex == itemLoop.
        const uint globalWarpIndex = min(itemLoop, max(desc.numAgregates, 1u) - 1u);
        const uint64_t aggA = aggAddr(desc.aggregates, globalWarpIndex);
        const PxgAggregateGpu agg = Ptr<PxgAggregateGpu>(aggA)[0];
        if (!aggIsValid(agg))
            continue;

        const uint numBounds = agg.size;
        const uint64_t boundIndices = agg.boundIndices1;
        const uint64_t projections = agg.sortedProjections1;
        const uint64_t handles = agg.sortedHandles1;
        const uint64_t sapBox1D = agg.sapBox1D1;
        const uint64_t startMasks = agg.startMasks1;

        GroupMemoryBarrierWithGroupSync();
        if (lane == 0u)
        {
            Ptr<PxgAggregateGpu> a = Ptr<PxgAggregateGpu>(aggA);
            a[0].sortedProjections1 = agg.sortedProjections0;
            a[0].sortedProjections0 = projections;
            a[0].sortedHandles1 = agg.sortedHandles0;
            a[0].sortedHandles0 = handles;
            a[0].sapBox1D1 = agg.sapBox1D0;
            a[0].sapBox1D0 = sapBox1D;
            a[0].startMasks1 = agg.startMasks0;
            a[0].startMasks0 = startMasks;
            a[0].comparisons1 = agg.comparisons0;
            a[0].comparisons0 = agg.comparisons1;
            a[0].boundIndices1 = agg.boundIndices0;
            a[0].boundIndices0 = boundIndices;
        }
        AllMemoryBarrierWithGroupSync();

        const uint nbProjections = numBounds * 2u;
        const uint oddMask = lane & 1u;
        const uint offset = oddMask * 3u;
        for (uint i = lane; i < nbProjections; i += WARP)
        {
            const uint id = i / 2u;
            const uint handle = Ptr<uint>(agg.updateBoundIndices)[id];
            Ptr<uint>(projections)[i] = Ptr<uint>(bp.newIntegerBounds + uint64_t(handle) * 24UL)[offset];
            Ptr<uint>(handles)[i] = bpCreateHandle(id, oddMask == 0u, false);
            if ((i % 2u) == 0u)
                Ptr<uint>(boundIndices)[id] = Ptr<uint>(agg.updateBoundIndices)[id];
        }
        AllMemoryBarrierWithGroupSync();

        const uint nbUint4s = (nbProjections + 3u) / 4u;
        rsSortSingleWarp(lane, projections, handles, nbProjections, nbUint4s,
                         projections + uint64_t(nbUint4s) * 16UL, handles + uint64_t(nbUint4s) * 16UL, 8u);
        AllMemoryBarrierWithGroupSync();

        for (uint i = 0u; i < nbProjections; i += WARP)
        {
            const uint idx = i + lane;
            bool isStart = false;
            if (idx < nbProjections)
            {
                const uint sortedHandle = Ptr<uint>(handles)[idx];
                const uint handle = bpGetHandle(sortedHandle);
                isStart = bpIsStartProjection(sortedHandle);
                Ptr<uint>(sapBox1D)[handle * 2u + (isStart ? 0u : 1u)] = idx;
            }
            const uint startMask = warpBallot(lane, isStart);
            if (lane == 0u)
                Ptr<uint>(startMasks)[i / 32u] = startMask;
        }
        AllMemoryBarrierWithGroupSync();
    }
}

// aggregate.cu:467 generatePairs -- comparison `i` of the frame's structure: which member
// it belongs to, which other member's start projection it is, and whether that pair is new
// (overlapping in `currBounds`, not in `prevBounds`).
bool generatePair(uint i, uint nbBounds, uint64_t comparisons, uint64_t sapBox1D, uint64_t startMasks,
                  uint64_t sortedHandles, uint64_t currBounds, uint64_t prevBounds, uint64_t boundsId,
                  uint64_t groupId, out uint2 pair)
{
    pair = uint2(0u, 0u);
    const uint idx = binarySearchU(Ptr<uint>(comparisons), nbBounds, i);
    const uint startIdx = Ptr<uint>(sapBox1D)[idx * 2u];
    const uint boundsIdx = Ptr<uint>(boundsId)[idx];
    const uint index = getIndex(Ptr<uint>(startMasks), startIdx + 1u, i, Ptr<uint>(comparisons)[idx]);
    const uint compareHandle = Ptr<uint>(sortedHandles)[index];
    const uint handle = bpGetHandle(compareHandle);
    const uint otherBoundsIdx = Ptr<uint>(boundsId)[handle];
    if (Ptr<uint>(groupId)[otherBoundsIdx] != Ptr<uint>(groupId)[boundsIdx])
    {
        if (iaabbIntersects(loadIAABB(currBounds, otherBoundsIdx), loadIAABB(currBounds, boundsIdx)))
        {
            if (!iaabbIntersects(loadIAABB(prevBounds, otherBoundsIdx), loadIAABB(prevBounds, boundsIdx)))
            {
                pair = uint2(min(boundsIdx, otherBoundsIdx), max(boundsIdx, otherBoundsIdx));
                return true;
            }
        }
    }
    return false;
}

// Appends this warp's flagged pairs to a report: one atomicAdd on the shared counter by
// lane 0, each flagged lane at its popc offset, nothing past maxPairs.
void appendPairs(uint lane, bool generated, uint2 pair, uint64_t pairs, uint64_t sharedIndexAddr, uint maxPairs)
{
    const uint mask = warpBallot(lane, generated);
    const uint nbPairs = popc(mask);
    uint outIndex = 0u;
    if (lane == 0u && nbPairs != 0u)
        InterlockedAdd(Ptr<uint>(sharedIndexAddr)[0], nbPairs, outIndex);
    outIndex = warpShfl(lane, outIndex, 0u);
    if (generated)
    {
        const uint offset = popc(mask & ((1u << lane) - 1u)) + outIndex;
        if (offset < maxPairs)
            Ptr<uint2>(pairs)[offset] = pair;
    }
}

// aggregate.cu:516 generateAllPairs
void generateAllPairs(uint lane, PxgAggregateGpu agg, uint totalComparisons, uint nbBounds, uint64_t pairs,
                      uint64_t sharedIndexAddr, uint64_t bounds, uint64_t prevBounds, uint64_t groupIds,
                      uint isLostPair, uint maxPairs)
{
    const uint64_t comparisons = aggComparisons(agg, isLostPair);
    for (uint i = 0u; i < totalComparisons; i += WARP)
    {
        const uint idx = lane + i;
        bool generated = false;
        uint2 pair = uint2(0u, 0u);
        if (idx < totalComparisons)
            generated = generatePair(idx, nbBounds, comparisons, aggSapBox1D(agg, isLostPair),
                                     aggStartMasks(agg, isLostPair), aggSortedHandles(agg, isLostPair),
                                     bounds, prevBounds, aggBoundIndices(agg, isLostPair), groupIds, pair);
        appendPairs(lane, generated, pair, pairs, sharedIndexAddr, maxPairs);
    }
}

// aggregate.cu:557 doSelfCollision
[shader("compute")]
[numthreads(32, 1, 1)]
void doSelfCollision(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;
    const PxgBroadPhaseDescGpu bp = Ptr<PxgBroadPhaseDescGpu>(gTwo.p0)[0];
    const uint64_t descA = gTwo.p1;
    const PxgAggregateDescGpu desc = Ptr<PxgAggregateDescGpu>(descA)[0];
    for (uint itemLoop = gid.x; itemLoop < desc.numAgregates; itemLoop += WorkgroupCount().x)
    {
        // The P9.22 clamp (see the header): live iterations have globalWarpIndex == itemLoop.
        const uint globalWarpIndex = min(itemLoop, max(desc.numAgregates, 1u) - 1u);
        const uint64_t aggA = aggAddr(desc.aggregates, globalWarpIndex);
        const PxgAggregateGpu agg = Ptr<PxgAggregateGpu>(aggA)[0];
        const uint maxPairs = desc.max_found_lost_pairs;
        uint totalCount = 0u;

        if (aggIsValid(agg) && (agg.filterHint & 1u) != 0u)
        {
            const uint numBounds = agg.size;
            Ptr<uint> sapBox1D = Ptr<uint>(agg.sapBox1D0);
            Ptr<uint> startMasks = Ptr<uint>(agg.startMasks0);
            Ptr<uint> comparisons = Ptr<uint>(agg.comparisons0);

            for (uint i = lane; i < numBounds; i += WARP)
            {
                const uint startIdx = sapBox1D[i * 2u];
                const uint endIdx = sapBox1D[i * 2u + 1u];
                const uint startWord = startIdx / WARP;
                const uint endWord = endIdx / WARP;
                uint mask = ~maskUpTo(startIdx & 31u);
                uint count = 0u;
                for (uint w = startWord; w < endWord; ++w)
                {
                    count += popc(startMasks[w] & mask);
                    mask = 0xffffffffu;
                }
                mask &= maskUpTo(endIdx & 31u);
                count += popc(startMasks[endWord] & mask);
                comparisons[i] = count;
            }
            AllMemoryBarrierWithGroupSync();

            for (uint i = 0u; i < numBounds; i += WARP)
            {
                const uint idx = i + lane;
                uint count = 0u;
                if (idx < numBounds)
                    count = comparisons[idx];
                const uint inclusiveSum = warpScanAdd(lane, count);
                const uint offset = inclusiveSum - count + totalCount;
                if (idx < numBounds)
                    comparisons[idx] = offset;
                totalCount += warpShfl(lane, inclusiveSum, 31u);
            }
            AllMemoryBarrierWithGroupSync();

            if (!aggIsNew(aggA))
                generateAllPairs(lane, agg, agg.prevComparisons, agg.prevSize, desc.lostPairReport,
                                 descA + 36UL, bp.oldIntegerBounds, bp.newIntegerBounds, bp.updateData_groups, 1u, maxPairs);

            generateAllPairs(lane, agg, totalCount, agg.size, desc.foundPairReport,
                             descA + 32UL, bp.newIntegerBounds, bp.oldIntegerBounds, bp.updateData_groups, 0u, maxPairs);
        }

        if (lane == 0u)
            Ptr<PxgAggregateGpu>(aggA)[0].prevComparisons = totalCount;
        AllMemoryBarrierWithGroupSync();
    }
}

// ===========================================================================
// aggregate-vs-actor and aggregate-vs-aggregate pairs
// ===========================================================================

// A "side" of a pair is either an aggregate's frame structure in device memory or, for a
// single actor, the one-box structure the CUDA builds in shared memory
// (aggregate.cu:992-1005): two projections per frame, handles {start 0, end 0}, start
// mask 1, and the actor handle as both bound indices.  Slang pointers are device
// addresses only, so the shared variant is read through these accessors.
struct Side
{
    bool isShared;
    uint64_t proj;      // sortedProjections[frame]
    uint64_t handles;   // sortedHandles[frame]
    uint64_t masks;     // startMasks[frame]
    uint64_t bi;        // boundIndices[frame]
    uint frame;         // for the shared structure: which projections row
    uint nbObj;
}

groupshared uint gsSideProj[2][2];      // [frame][start/end]
groupshared uint gsSideBi[2];
groupshared uint gsCmp[WARP];           // shComparisons[threadIdx.y]
groupshared uint gsStart[WARP];         // shStartIdx[threadIdx.y]

uint sideProj(Side s, uint i)    { return s.isShared ? gsSideProj[s.frame][i] : Ptr<uint>(s.proj)[i]; }
uint sideHandle(Side s, uint i)  { return s.isShared ? bpCreateHandle(0u, i == 1u, false) : Ptr<uint>(s.handles)[i]; }
uint sideMask(Side s, uint w)    { return s.isShared ? 1u : Ptr<uint>(s.masks)[w]; }
uint sideBi(Side s, uint i)      { return s.isShared ? gsSideBi[i] : Ptr<uint>(s.bi)[i]; }

// bSearchFirstProjection (:663): the first index whose projection is >= proj
uint sideSearchFirst(Side s, uint numElements, uint proj)
{
    uint left = 0u;
    uint right = numElements;
    while (left < right)
    {
        const uint pos = (left + right) / 2u;
        if (sideProj(s, pos) < proj) left = pos + 1u; else right = pos;
    }
    return left;
}

// binarySearch (reduction.cuh:729) over a side's projections
uint sideSearchLast(Side s, uint numElements, uint value)
{
    uint left = 0u;
    uint right = numElements;
    while (left < right)
    {
        const uint pos = (left + right) / 2u;
        if (sideProj(s, pos) <= value) left = pos + 1u; else right = pos;
    }
    return left != 0u ? left - 1u : 0u;
}

// getIndex (:413) over a side's start masks
uint sideGetIndex(Side s, uint startIdx, uint targetIndex, uint baseIndex)
{
    const uint startWord = startIdx / WARP;
    const uint mask = ~((1u << (startIdx & 31u)) - 1u);
    uint count = 0u;
    const uint target = targetIndex - baseIndex;
    uint w = startWord;
    uint maskedResult = sideMask(s, w) & mask;
    for (;;)
    {
        const uint newCount = count + popc(maskedResult);
        if (newCount > target)
            break;
        count = newCount;
        maskedResult = sideMask(s, w + 1u);
        ++w;
    }
    return w * WARP + getIthBit(maskedResult, target - count);
}

// countComparisons (:737): the start projections of `other` in [startIdx, endIdx]
uint countComparisons(IAABB objBound, Side other, uint nbProjections, out uint outStartIdx)
{
    const uint startIdx = sideSearchFirst(other, nbProjections, objBound.v[0]);
    const uint endIdx = sideSearchLast(other, nbProjections, objBound.v[3]);
    const uint startWordId = startIdx / 32u;
    const uint endWordId = endIdx / 32u;
    uint w = sideMask(other, startWordId) & (~((1u << (startIdx & 31u)) - 1u));
    uint totalComparisons = 0u;
    for (uint i = startWordId; i < endWordId; )
    {
        totalComparisons += popc(w);
        ++i;
        w = sideMask(other, i);
    }
    w &= maskUpTo(endIdx & 31u);
    totalComparisons += popc(w);
    outStartIdx = startIdx;
    return totalComparisons;
}

// boxPruningProjection (:767): every object of `objs` against the sorted projections of
// `other`
void boxPruningProjection(uint lane, uint64_t bounds, uint64_t oldBounds, Side objs, Side other,
                          uint64_t groupIds, uint64_t pairs, uint64_t sharedIndexAddr, uint isFirst, uint maxPairs)
{
    const uint nbProjections = other.nbObj * 2u;
    for (uint i = 0u; i < objs.nbObj; i += WARP)
    {
        const uint objId = i + lane;
        uint startIdx = 0u;
        uint count = 0u;
        if (objId < objs.nbObj)
        {
            const uint handle = sideBi(objs, objId);
            count = countComparisons(loadIAABB(bounds, handle), other, nbProjections, startIdx);
        }

        uint inclusiveSum = warpScanAdd(lane, count);
        const uint totalCount = warpShfl(lane, inclusiveSum, 31u);
        inclusiveSum -= count;

        gsCmp[lane] = inclusiveSum;
        gsStart[lane] = startIdx;
        GroupMemoryBarrierWithGroupSync();

        for (uint b = 0u; b < totalCount; b += WARP)
        {
            const uint cmpId = b + lane;
            bool hasContact = false;
            uint2 pair = uint2(0u, 0u);
            if (cmpId < totalCount)
            {
                // binarySearch(shComparisons, 32, cmpId)
                uint left = 0u;
                uint right = WARP;
                while (left < right)
                {
                    const uint pos = (left + right) / 2u;
                    if (gsCmp[pos] <= cmpId) left = pos + 1u; else right = pos;
                }
                const uint idx = left != 0u ? left - 1u : 0u;

                const uint handle = sideBi(objs, i + idx);
                const uint index = sideGetIndex(other, gsStart[idx], cmpId, gsCmp[idx]);
                const uint otherId = bpGetHandle(sideHandle(other, index));
                const uint otherHandle = sideBi(other, otherId);

                if (Ptr<uint>(groupIds)[handle] != Ptr<uint>(groupIds)[otherHandle])
                {
                    const IAABB bh = loadIAABB(bounds, handle);
                    const IAABB bo = loadIAABB(bounds, otherHandle);
                    if (iaabbIntersects(bh, bo))
                    {
                        if (!iaabbIntersects(loadIAABB(oldBounds, handle), loadIAABB(oldBounds, otherHandle)))
                        {
                            if (isFirst != 0u || bh.v[0] != bo.v[0])
                            {
                                hasContact = true;
                                pair = uint2(min(handle, otherHandle), max(handle, otherHandle));
                            }
                        }
                    }
                }
            }

            const uint contactMask = warpBallot(lane, hasContact);
            const uint nbPairs = popc(contactMask);
            if (nbPairs != 0u)
            {
                uint outIndex = 0u;
                if (lane == 0u)
                    InterlockedAdd(Ptr<uint>(sharedIndexAddr)[0], nbPairs, outIndex);
                outIndex = warpShfl(lane, outIndex, 0u);
                if (hasContact)
                {
                    const uint offset = popc(contactMask & ((1u << lane) - 1u)) + outIndex;
                    if (offset < maxPairs)
                        Ptr<uint2>(pairs)[offset] = pair;
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

bool testBitmap(uint actorHandle, uint64_t bitmap)
{
    return (Ptr<uint>(bitmap)[actorHandle / 32u] & (1u << (actorHandle & 31u))) != 0u;
}

PxgAggregateGpu zeroAggregate()
{
    PxgAggregateGpu a;
    a.updateBoundIndices = 0UL; a.boundIndices0 = 0UL; a.boundIndices1 = 0UL;
    a.sortedProjections0 = 0UL; a.sortedProjections1 = 0UL; a.sortedHandles0 = 0UL; a.sortedHandles1 = 0UL;
    a.sapBox1D0 = 0UL; a.sapBox1D1 = 0UL; a.startMasks0 = 0UL; a.startMasks1 = 0UL;
    a.comparisons0 = 0UL; a.comparisons1 = 0UL;
    a.prevComparisons = 0u; a.prevSize = 0u; a.mIndex = 0u; a.size = 0u; a.filterHint = 0u; a.mEnvID = 0u;
    a.isNewWord = 0u; a.pad132 = 0u;
    return a;
}

Side makeSide(PxgAggregateGpu agg, bool isAggregate, uint frame)
{
    Side s;
    s.isShared = !isAggregate;
    s.frame = frame;
    s.proj = aggSortedProjections(agg, frame);
    s.handles = aggSortedHandles(agg, frame);
    s.masks = aggStartMasks(agg, frame);
    s.bi = aggBoundIndices(agg, frame);
    s.nbObj = isAggregate ? (frame == 0u ? agg.size : agg.prevSize) : 1u;
    return s;
}

// aggregate.cu:928 doAggPairCollisions.  At most ONE side of a pair is a single actor (an
// actor-actor pair never reaches the aggregate pair list), which is what lets the shared
// one-box structure be a single set of groupshared words.
[shader("compute")]
[numthreads(32, 1, 1)]
void doAggPairCollisions(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;
    const PxgBroadPhaseDescGpu bp = Ptr<PxgBroadPhaseDescGpu>(gTwo.p0)[0];
    const uint64_t descA = gTwo.p1;
    const PxgAggregateDescGpu desc = Ptr<PxgAggregateDescGpu>(descA)[0];
    Ptr<PxgAggregatePairGpu> pairs = Ptr<PxgAggregatePairGpu>(desc.aggPairs);
    const uint nbAggregatePairs = min(Ptr<uint>(desc.aggPairCount)[0], desc.max_agg_pairs);
    const uint maxFoundLostPairs = desc.max_found_lost_pairs;
    Ptr<uint> aggBitmap = Ptr<uint>(desc.removeBitmap);

    // The CUDA's own grid-stride loop, `i += blockDim.y * gridDim.x`: one group per CUDA
    // warp, so the warp index is gid.x and the stride the group count.
    {
        const uint globalWarpIndex = gid.x;
        for (uint i = globalWarpIndex; i < nbAggregatePairs; i += WorkgroupCount().x)
        {
            // The P9.22 clamp of the load index (see the header).  The host grid is a constant
            // 8192 * 2 groups, so every group past the live pair count starts out of range, and
            // llvmpipe's empty-mask pass would read pairs[gid.x] past the pool and follow its
            // handles.  Live iterations have iL == i; the (masked) stores stay at i.
            const uint iL = min(i, max(nbAggregatePairs, 1u) - 1u);
            const PxgAggregatePairGpu pair = pairs[iL];
            const bool pairIsNew = (pair.flagsWord & 0xffu) != 0u;
            const bool pairIsDead = ((pair.flagsWord >> 8) & 0xffu) != 0u;
            bool shouldRemove = false;
            if (!pairIsDead)
            {
                const bool isValid0 = !testBitmap(pair.actorHandle0, bp.aabbMngr_removedHandleMap);
                const bool isValid1 = !testBitmap(pair.actorHandle1, bp.aabbMngr_removedHandleMap);
                if (!isValid0 || !isValid1)
                {
                    if (lane == 0u)
                        aggBitmap[i] = 1u;
                    AllMemoryBarrierWithGroupSync();
                    continue;
                }

                shouldRemove = !iaabbIntersects(loadIAABB(bp.newIntegerBounds, pair.actorHandle0),
                                                loadIAABB(bp.newIntegerBounds, pair.actorHandle1));

                const VolumeDataGpu vd0 = Ptr<VolumeDataGpu>(bp.aabbMngr_volumeData)[pair.actorHandle0];
                const VolumeDataGpu vd1 = Ptr<VolumeDataGpu>(bp.aabbMngr_volumeData)[pair.actorHandle1];
                // VolumeData::isAggregate: not a single actor and the low bit set
                const bool isAggregate0 = vd0.aggregate != 0xffffffffu && (vd0.aggregate & 1u) != 0u;
                const bool isAggregate1 = vd1.aggregate != 0xffffffffu && (vd1.aggregate & 1u) != 0u;

                const PxgAggregateGpu agg0 = isAggregate0
                    ? Ptr<PxgAggregateGpu>(aggAddr(desc.aggregates, vd0.aggregate >> 1))[0] : zeroAggregate();
                const PxgAggregateGpu agg1 = isAggregate1
                    ? Ptr<PxgAggregateGpu>(aggAddr(desc.aggregates, vd1.aggregate >> 1))[0] : zeroAggregate();

                // the single actor's one-box structure (:997-1005, :1056-1061, :1092-1097).  The
                // CUDA writes side 0's actor first and side 1's second into the same shared
                // words, so if both were actors side 1's would stand -- mirrored, though an
                // actor-actor pair never reaches the aggregate pair list.
                GroupMemoryBarrierWithGroupSync();
                if (lane < 2u)
                {
                    const uint actor = !isAggregate1 ? pair.actorHandle1 : pair.actorHandle0;
                    if (!isAggregate0 || !isAggregate1)
                    {
                        gsSideBi[lane] = actor;
                        gsSideProj[0][lane] = Ptr<uint>(bp.newIntegerBounds + uint64_t(actor) * 24UL)[lane * 3u];
                        gsSideProj[1][lane] = Ptr<uint>(bp.oldIntegerBounds + uint64_t(actor) * 24UL)[lane * 3u];
                    }
                }
                GroupMemoryBarrierWithGroupSync();

                const Side s00 = makeSide(agg0, isAggregate0, 0u);
                const Side s10 = makeSide(agg1, isAggregate1, 0u);
                // bipartiteBoxPruning (:906), found pairs on the current structures
                boxPruningProjection(lane, bp.newIntegerBounds, bp.oldIntegerBounds, s00, s10, bp.updateData_groups,
                                     desc.foundPairReport, descA + 32UL, 1u, maxFoundLostPairs);
                boxPruningProjection(lane, bp.newIntegerBounds, bp.oldIntegerBounds, s10, s00, bp.updateData_groups,
                                     desc.foundPairReport, descA + 32UL, 0u, maxFoundLostPairs);

                if (!pairIsNew)
                {
                    const Side s01 = makeSide(agg0, isAggregate0, 1u);
                    const Side s11 = makeSide(agg1, isAggregate1, 1u);
                    boxPruningProjection(lane, bp.oldIntegerBounds, bp.newIntegerBounds, s01, s11, bp.updateData_groups,
                                         desc.lostPairReport, descA + 36UL, 1u, maxFoundLostPairs);
                    boxPruningProjection(lane, bp.oldIntegerBounds, bp.newIntegerBounds, s11, s01, bp.updateData_groups,
                                         desc.lostPairReport, descA + 36UL, 0u, maxFoundLostPairs);
                }
                AllMemoryBarrierWithGroupSync();

                if (lane == 0u)
                    storeByte(desc.aggPairs + uint64_t(i) * 12UL + 8UL, 0u);   // pairs[i].isNew = false
            }

            if (lane == 0u)
                aggBitmap[i] = shouldRemove ? 1u : 0u;
            AllMemoryBarrierWithGroupSync();
        }
    }
}

// ===========================================================================
// the report copy and the persistent aggregate-pair pool
// ===========================================================================

// (PxgAggregateDesc*) -> kOnePtr
struct OnePtrParams
{
    uint64_t p0;
}
[[vk::push_constant]] ConstantBuffer<OnePtrParams> gOne;

// aggregate.cu:1133 aggCopyReports (64 x 256)
[shader("compute")]
[numthreads(256, 1, 1)]
void aggCopyReports(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint64_t descA = gOne.p0;
    const PxgAggregateDescGpu desc = Ptr<PxgAggregateDescGpu>(descA)[0];
    const uint nbCreatedPairs = desc.sharedFoundPairIndex;
    const uint nbLostPairs = desc.sharedLostPairIndex;
    const uint maxPairs = desc.max_found_lost_pairs;
    const uint idx = gtid.x + gid.x * COPY_BLOCK;

    const bool foundOverflow = nbCreatedPairs > maxPairs;
    const bool lostOverflow = nbLostPairs > maxPairs;
    if (gtid.x == 0u && gid.x == 0u)
    {
        // `found_pairs_overflow_flags |= foundOverflow`: a bool |= stores (old | new) != 0
        const bool found = loadBool(descA + uint64_t(DESC_FLAGS_OFF)) || foundOverflow;
        storeByte(descA + uint64_t(DESC_FLAGS_OFF), found ? 1u : 0u);
        storeByte(descA + uint64_t(DESC_FLAGS_OFF) + 1UL, lostOverflow ? 1u : 0u);
    }

    const uint tCreated = foundOverflow ? maxPairs : nbCreatedPairs;
    const uint tLost = lostOverflow ? maxPairs : nbLostPairs;
    const uint nbCreateElements = tCreated * 2u;   // PxgBroadPhasePair = 2 PxU32
    const uint nbLostElements = tLost * 2u;
    for (uint i = idx; i < nbCreateElements; i += COPY_BLOCK * COPY_GRID)
        Ptr<uint>(desc.foundPairReportMap)[i] = Ptr<uint>(desc.foundPairReport)[i];
    for (uint i = idx; i < nbLostElements; i += COPY_BLOCK * COPY_GRID)
        Ptr<uint>(desc.lostPairReportMap)[i] = Ptr<uint>(desc.lostPairReport)[i];
}

// (PxgAggregateDesc* aggDesc, PxgBroadPhaseDesc* bpDesc) -> kTwoPtr, gTwo: p0 = aggDesc, p1 = bpDesc

// aggregate.cu:1179 addAggPairsStage1 (64 x 1024): the broadphase's new aggregate pairs
// into free slots first (taken from the END of the free list), then past maxIndex.
[shader("compute")]
[numthreads(1024, 1, 1)]
void addAggPairsStage1(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const PxgAggregateDescGpu desc = Ptr<PxgAggregateDescGpu>(gTwo.p0)[0];
    const PxgBroadPhaseDescGpu bp = Ptr<PxgBroadPhaseDescGpu>(gTwo.p1)[0];
    const uint nbFoundPairsSafe = min(bp.sharedFoundAggPairIndex, desc.max_found_lost_pairs);
    const uint numFreeIndices = Ptr<uint>(desc.freeBufferList)[0];
    const uint maxIndex = Ptr<uint>(desc.freeBufferList)[1];
    const uint maxAggPairs = desc.max_agg_pairs;

    for (uint i = gtid.x + gid.x * ADD_BLOCK; i < nbFoundPairsSafe; i += ADD_BLOCK * ADD_GRID)
    {
        uint index = 0u;
        if (i < numFreeIndices)
            index = Ptr<uint>(desc.freeIndices)[numFreeIndices - i - 1u];
        else
            index = maxIndex + (i - numFreeIndices);
        if (index < maxAggPairs)
        {
            const uint2 fp = Ptr<uint2>(bp.foundAggPairReport)[i];
            Ptr<PxgAggregatePairGpu> p = Ptr<PxgAggregatePairGpu>(desc.aggPairs);
            p[index].actorHandle0 = fp.x;
            p[index].actorHandle1 = fp.y;
            p[index].flagsWord = 1u;   // isNew = true, isDead = false, pad = 0: all four bytes, one writer
        }
    }
}

// aggregate.cu:1223 addAggPairsStage2 (thread 0 alone)
[shader("compute")]
[numthreads(1024, 1, 1)]
void addAggPairsStage2(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    if (gtid.x != 0u || gid.x != 0u)
        return;
    const uint64_t descA = gTwo.p0;
    const PxgAggregateDescGpu desc = Ptr<PxgAggregateDescGpu>(descA)[0];
    const PxgBroadPhaseDescGpu bp = Ptr<PxgBroadPhaseDescGpu>(gTwo.p1)[0];
    Ptr<uint> freeList = Ptr<uint>(desc.freeBufferList);
    Ptr<PxgAggregateDescGpu> d = Ptr<PxgAggregateDescGpu>(descA);

    const uint nbFoundAggPairs = bp.sharedFoundAggPairIndex;
    const uint nbFoundAggPairsSafe = min(nbFoundAggPairs, desc.max_found_lost_pairs);
    const uint nbFreeInds = freeList[0];
    uint newFreeInds = 0u;
    uint numToIncrement = 0u;
    if (nbFreeInds >= nbFoundAggPairsSafe)
        newFreeInds = nbFreeInds - nbFoundAggPairsSafe;
    else
        numToIncrement = nbFoundAggPairsSafe - nbFreeInds;
    freeList[0] = newFreeInds;

    uint maxIndex = freeList[1] + numToIncrement;
    d[0].aggPairOverflowCount = maxIndex;
    if (maxIndex > desc.max_agg_pairs)
    {
        storeByte(descA + uint64_t(DESC_FLAGS_OFF) + 2UL, 1u);
        maxIndex = desc.max_agg_pairs;
    }
    freeList[1] = maxIndex;
    Ptr<uint>(desc.aggPairCount)[0] = maxIndex;

    if (nbFoundAggPairs > desc.max_found_lost_pairs)
    {
        storeByte(descA + uint64_t(DESC_FLAGS_OFF), 1u);
        d[0].foundCandidatePairOverflowCount = nbFoundAggPairs;
    }
}

// ---- warp-32 tiles of a 1024-thread workgroup (removeAggPairsStage1/2) ----
groupshared uint gsTile[REMOVE_BLOCK];
groupshared uint gsWarpSum[REMOVE_BLOCK / WARP];

// __ballot_sync(FULL_MASK, pred) within the caller's 32-lane tile
uint tileBallot(uint tid, bool pred)
{
    gsTile[tid] = pred ? 1u : 0u;
    GroupMemoryBarrierWithGroupSync();
    const uint base = tid & ~(WARP - 1u);
    uint mask = 0u;
    for (uint l = 0u; l < WARP; ++l)
        mask |= gsTile[base + l] << l;
    GroupMemoryBarrierWithGroupSync();
    return mask;
}

// __shfl_sync(FULL_MASK, v, srcLane) within the caller's tile
uint tileShfl(uint tid, uint v, uint src)
{
    gsTile[tid] = v;
    GroupMemoryBarrierWithGroupSync();
    const uint r = gsTile[(tid & ~(WARP - 1u)) + (src & (WARP - 1u))];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

// aggregate.cu:1277 removeAggPairsStage1 (32 blocks of 32 x 32): how many pairs each
// block of each wave removes
[shader("compute")]
[numthreads(1024, 1, 1)]
void removeAggPairsStage1(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint lane = tid & (WARP - 1u);
    const uint warp = tid / WARP;
    const PxgAggregateDescGpu desc = Ptr<PxgAggregateDescGpu>(gOne.p0)[0];
    const uint nbAggPairMasks = Ptr<uint>(desc.aggPairCount)[0];
    Ptr<uint> aggBitmap = Ptr<uint>(desc.removeBitmap);
    Ptr<uint> aggHisto = Ptr<uint>(desc.removeHistogram);

    const uint nbWarpsPerWave = (REMOVE_BLOCK / WARP) * REMOVE_GRID;
    const uint nbWarpsRequired = (nbAggPairMasks + 31u) / 32u;
    const uint globalWarpIndex = warp + gid.x * (REMOVE_BLOCK / WARP);
    const uint nbIterations = (nbWarpsRequired + nbWarpsPerWave - 1u) / nbWarpsPerWave;

    for (uint i = 0u; i < nbIterations; ++i)
    {
        const uint idx = i * nbWarpsPerWave + globalWarpIndex;
        const uint thIdx = idx * WARP + lane;
        uint remove = 0u;
        if (thIdx < nbAggPairMasks)
            remove = aggBitmap[thIdx];
        const uint mask = tileBallot(tid, remove != 0u);
        if (lane == 0u)
            gsWarpSum[warp] = countbits(mask);
        GroupMemoryBarrierWithGroupSync();
        if (warp == 0u)
        {
            // warpReduction<AddOpPxU32> over the 32 warp counts
            uint blockCount = 0u;
            for (uint w = 0u; w < REMOVE_BLOCK / WARP; ++w)
                blockCount += gsWarpSum[w];
            if (lane == 0u)
                aggHisto[gid.x + i * REMOVE_GRID] = blockCount;
        }
        // (the CUDA rewrites shWarpSum in the next wave without a barrier; ordered here)
        GroupMemoryBarrierWithGroupSync();
    }
}

// aggregate.cu:1335 removeAggPairsStage2: each removed pair is marked dead and its slot
// appended to the free list, in bitmap order
[shader("compute")]
[numthreads(1024, 1, 1)]
void removeAggPairsStage2(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint lane = tid & (WARP - 1u);
    const uint warp = tid / WARP;
    const uint64_t descA = gOne.p0;
    const PxgAggregateDescGpu desc = Ptr<PxgAggregateDescGpu>(descA)[0];
    const uint nbAggPairMasks = Ptr<uint>(desc.aggPairCount)[0];
    Ptr<uint> aggBitmap = Ptr<uint>(desc.removeBitmap);
    Ptr<uint> aggHisto = Ptr<uint>(desc.removeHistogram);
    const uint numFreeInds = Ptr<uint>(desc.freeBufferList)[0];
    Ptr<uint> freeIndices = Ptr<uint>(desc.freeIndices);

    const uint nbWarpsPerWave = (REMOVE_BLOCK / WARP) * REMOVE_GRID;
    const uint nbWarpsRequired = (nbAggPairMasks + 31u) / 32u;
    const uint globalWarpIndex = warp + gid.x * (REMOVE_BLOCK / WARP);
    const uint nbIterations = (nbWarpsRequired + nbWarpsPerWave - 1u) / nbWarpsPerWave;

    uint accum = 0u;
    for (uint i = 0u; i < nbIterations; ++i)
    {
        const uint idx = i * nbWarpsPerWave + globalWarpIndex;
        const uint thIdx = idx * WARP + lane;

        // the block's offset within the wave: a scan over the 32 block counts
        const uint val = aggHisto[lane + i * WARP];
        const uint blockSum = warp32InclusiveScan(tid, val);
        const uint totalBlockSum = tileShfl(tid, blockSum, 31u);
        const uint blockOffset = tileShfl(tid, blockSum - val, gid.x) + accum;
        accum += totalBlockSum;

        uint remove = 0u;
        if (thIdx < nbAggPairMasks)
            remove = aggBitmap[thIdx];
        const uint removeMask = tileBallot(tid, remove != 0u);
        const uint offsetInWarp = countbits(removeMask & ((1u << lane) - 1u));
        if (lane == 31u)
            gsWarpSum[warp] = countbits(removeMask);
        GroupMemoryBarrierWithGroupSync();

        const uint ws = gsWarpSum[lane];
        const uint blockCount = warp32InclusiveScan(tid, ws);
        const uint warpOffset = tileShfl(tid, blockCount - ws, warp);
        const uint outputIndex = blockOffset + warpOffset + offsetInWarp;
        if (remove != 0u)
        {
            storeByte(desc.aggPairs + uint64_t(thIdx) * 12UL + 9UL, 1u);   // aggPairs[thIdx].isDead = true
            freeIndices[numFreeInds + outputIndex] = thIdx;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0u && gid.x == 0u)
        Ptr<PxgAggregateDescGpu>(descA)[0].nbRemoved = accum;
}

// aggregate.cu:1416 removeAggPairsStage3 (1 x 32): the removed slots join the free list
[shader("compute")]
[numthreads(32, 1, 1)]
void removeAggPairsStage3(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    if (gtid.x != 0u || gid.x != 0u)
        return;
    const PxgAggregateDescGpu desc = Ptr<PxgAggregateDescGpu>(gOne.p0)[0];
    Ptr<uint> freeList = Ptr<uint>(desc.freeBufferList);
    freeList[0] = freeList[0] + desc.nbRemoved;
}
