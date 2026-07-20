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

// broadphaseReports.slang
//
// P5 Step 4 (BDA, engine launch path): the broad-phase found/lost pair report
// accumulation pipeline (gpubroadphase/src/CUDA/broadphase.cu, L2284-2596),
// driven exactly as the engine launches it — ONE PxgBroadPhaseDesc device
// pointer, walked through the byte-exact PxgBroadPhaseDescGpu mirror:
//   - accumulateReportsStage_1 (L2284): per-block count of the aggregate (or
//     actor) pairs in each of the four report categories (found/lost x
//     agg/actor), written to the desc's aggReportBlock/actorReportBlock.
//   - accumulateReportsStage_2 (L2382): exclusive prefix over the per-block
//     counts (the agg totals land in the desc's sharedFoundAggPairIndex /
//     sharedLostAggPairIndex, written through the desc pointer), then a stable
//     compaction of each category's matching pairs into its output array
//     (foundAgg/foundActor/lostAgg/lostActor).
//   - copyReports (L2563): copy the compacted actor found/lost pairs to the
//     mapped output buffers and store the overflow flag into the desc.
//
// A pair is an "aggregate" report unless BOTH of its volumes are single actors
// (Bp::VolumeData::isSingleActor(), mAggregate == 0xffffffff at byte offset 8
// of the 16-byte VolumeData; the .cu's volumeData NULL-pointer branch is the
// address == 0 check).  The four categories are selected by the dispatch's Y
// grid index (blockIdx.y): 0 = found/agg, 1 = found/actor, 2 = lost/agg,
// 3 = lost/actor.
//
// Portability: the CUDA warp __ballot_sync+__popc reductions/scans become
// groupshared 32-lane scans; the .cu's numIterationPerBlock divides by the
// compile-time grid constant (=32), preserved as NUM_BLOCKS; copyReports'
// grid-stride uses gridDim.x = WorkgroupCount().x (engine grid 64).  The
// overflow bool is byte 0 of overflowFlagsWord — bytes 1-3 are struct tail
// padding and the .cu stores the bool unconditionally, so a whole-word store
// of 0/1 is exact.

import pxgSlangBpDesc;

static const uint BLOCK = 256;
static const uint NUM_BLOCKS = 32; // == BP_COMPUTE_INCREMENTAL_CMP_COUNTS1/2 grid
static const uint WARP = 32;
static const uint WARPS_PER_BLOCK = BLOCK / WARP; // 8
static const uint PX_INVALID_U32 = 0xffffffffu;

// engine ABI: all three entries take the PxgBroadPhaseDesc device pointer only
struct BpDescParams
{
    uint64_t bpDesc;
}
[[vk::push_constant]]
ConstantBuffer<BpDescParams> gBp;

// Bp::VolumeData is 16 bytes: { void* mUserData; PxU32 mAggregate; pad } —
// mAggregate is uint 2 of the 4-uint stride.  A null volumeData pointer means
// "everything is a single actor" (the .cu's ?: on the pointer).
bool isSingleActor(uint64_t volumeDataAddr, uint volIndex)
{
    if (volumeDataAddr == 0)
        return true;
    return (Ptr<uint>(volumeDataAddr))[volIndex * 4u + 2u] == PX_INVALID_U32;
}

// classify a pair: returns true if it is an aggregate report (not both single
// actors), false if it is an actor report
bool isAggregatePair(uint64_t volumeDataAddr, uint volA, uint volB)
{
    const bool singleA = isSingleActor(volumeDataAddr, volA);
    const bool singleB = isSingleActor(volumeDataAddr, volB);
    return !(singleA && singleB);
}

groupshared uint gsLane[BLOCK];         // per-lane scan scratch
groupshared uint gsWarp[WARPS_PER_BLOCK];
groupshared uint gsBlockAccum;
groupshared uint gsBlockHistogram[NUM_BLOCKS];

// inclusive scan of gsLane within each 32-lane warp (all warps in parallel)
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

// exclusive scan over the WARPS_PER_BLOCK warp totals in gsWarp; folds the
// grand total into gsBlockAccum.  Uniform control flow (all threads reach every
// barrier); only lanes < 8 touch gsWarp.
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
// stage 1: per-block count
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void accumulateReportsStage_1(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    Ptr<PxgBroadPhaseDescGpu> desc = Ptr<PxgBroadPhaseDescGpu>(gBp.bpDesc);
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const uint category = gid.y; // 0=found/agg,1=found/actor,2=lost/agg,3=lost/actor
    const bool found = (category / 2) == 0;
    const bool outputAggReport = (category % 2) == 0;

    const uint nbPairs = found ? desc[0].sharedFoundPairIndex : desc[0].sharedLostPairIndex;
    Ptr<uint> pairs = Ptr<uint>(found ? desc[0].foundPairReport : desc[0].lostPairReport); // 2 uints/pair
    const uint nbPairsSafe = min(nbPairs, desc[0].max_found_lost_pairs);
    const uint64_t volumeDataAddr = desc[0].aabbMngr_volumeData;

    const uint totalBlockRequired = (nbPairs + (BLOCK - 1)) / BLOCK;
    const uint numIterationPerBlock = (totalBlockRequired + (NUM_BLOCKS - 1)) / NUM_BLOCKS;

    // count matching pairs in this block's region
    uint localCount = 0;
    for (uint it = 0; it < numIterationPerBlock; ++it)
    {
        const uint workIndex = it * BLOCK + tid + numIterationPerBlock * blockIdx * BLOCK;
        if (workIndex < nbPairsSafe)
        {
            const uint volA = pairs[workIndex * 2u + 0u];
            const uint volB = pairs[workIndex * 2u + 1u];
            const bool isAgg = isAggregatePair(volumeDataAddr, volA, volB);
            const bool matches = outputAggReport ? isAgg : !isAgg;
            if (matches)
                ++localCount;
        }
    }

    // block reduction of localCount -> gsBlockAccum
    if (tid == 0)
        gsBlockAccum = 0;
    gsLane[tid] = localCount;
    GroupMemoryBarrierWithGroupSync();
    warpScanLane(tid);
    if ((tid & (WARP - 1)) == WARP - 1)
        gsWarp[tid / WARP] = gsLane[tid]; // warp total (inclusive last lane)
    GroupMemoryBarrierWithGroupSync();
    scanWarpTotals(tid);
    GroupMemoryBarrierWithGroupSync();

    if (tid == 0)
    {
        // .cu: bpDesc->aggReportBlock[blockIdx.y/2][blockIdx.x] (or actor)
        const uint64_t blockArr = outputAggReport
            ? (found ? desc[0].aggReportBlock[0] : desc[0].aggReportBlock[1])
            : (found ? desc[0].actorReportBlock[0] : desc[0].actorReportBlock[1]);
        (Ptr<uint>(blockArr))[blockIdx] = gsBlockAccum;
    }
}

// ---------------------------------------------------------------------------
// stage 2: prefix over block counts + stable compaction
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void accumulateReportsStage_2(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    Ptr<PxgBroadPhaseDescGpu> desc = Ptr<PxgBroadPhaseDescGpu>(gBp.bpDesc);
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const uint category = gid.y;
    const bool found = (category / 2) == 0;
    const bool outputAggReport = (category % 2) == 0;
    const uint warpIndex = tid / WARP;

    // .cu category selection (blockIdx.y 0..3)
    const uint nbPairs = found ? desc[0].sharedFoundPairIndex : desc[0].sharedLostPairIndex;
    Ptr<uint> pairs = Ptr<uint>(found ? desc[0].foundPairReport : desc[0].lostPairReport);
    const uint maxPairs = outputAggReport ? desc[0].max_found_lost_agg_pairs : desc[0].max_found_lost_pairs;
    const uint64_t outAddr = found
        ? (outputAggReport ? desc[0].foundAggPairReport : desc[0].foundActorPairReport)
        : (outputAggReport ? desc[0].lostAggPairReport : desc[0].lostActorPairReport);
    Ptr<uint> aggOrActorPairs = Ptr<uint>(outAddr);

    // AD: pairs buffer is sized according to this, aggregate count is not considered.
    const uint nbPairsSafe = min(nbPairs, desc[0].max_found_lost_pairs);
    const uint64_t volumeDataAddr = desc[0].aabbMngr_volumeData;

    if (tid == 0)
        gsBlockAccum = 0;

    // warp 0 in the .cu: exclusive prefix over the 32 per-block counts
    if (tid < NUM_BLOCKS)
    {
        const uint64_t blockArr = outputAggReport
            ? (found ? desc[0].aggReportBlock[0] : desc[0].aggReportBlock[1])
            : (found ? desc[0].actorReportBlock[0] : desc[0].actorReportBlock[1]);
        gsLane[tid] = (Ptr<uint>(blockArr))[tid];
    }
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0)
    {
        uint acc = 0;
        for (uint i = 0; i < NUM_BLOCKS; ++i)
        {
            const uint v = gsLane[i];
            gsBlockHistogram[i] = acc;
            acc += v;
        }
        // .cu: the agg totals go to the desc (only blockIdx.y == 0 / 2 write;
        // all 32 x-blocks store the identical value, as in CUDA)
        if (outputAggReport)
        {
            if (found)
                desc[0].sharedFoundAggPairIndex = acc;
            else
                desc[0].sharedLostAggPairIndex = acc;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    const uint totalBlockRequired = (nbPairs + (BLOCK - 1)) / BLOCK;
    const uint numIterationPerBlock = (totalBlockRequired + (NUM_BLOCKS - 1)) / NUM_BLOCKS;
    const uint blockStartIndex = gsBlockHistogram[blockIdx];

    for (uint it = 0; it < numIterationPerBlock; ++it)
    {
        const uint workIndex = it * BLOCK + tid + numIterationPerBlock * blockIdx * BLOCK;

        bool matches = false;
        uint volA = 0;
        uint volB = 0;
        if (workIndex < nbPairsSafe)
        {
            volA = pairs[workIndex * 2u + 0u];
            volB = pairs[workIndex * 2u + 1u];
            const bool isAgg = isAggregatePair(volumeDataAddr, volA, volB);
            matches = outputAggReport ? isAgg : !isAgg;
        }

        // intra-warp exclusive prefix (offset) + warp total (accum)
        gsLane[tid] = matches ? 1u : 0u;
        GroupMemoryBarrierWithGroupSync();
        warpScanLane(tid);
        const uint offset = gsLane[tid] - (matches ? 1u : 0u);
        if ((tid & (WARP - 1)) == WARP - 1)
            gsWarp[warpIndex] = gsLane[tid];

        const uint prevAccum = gsBlockAccum;
        GroupMemoryBarrierWithGroupSync();

        scanWarpTotals(tid);

        GroupMemoryBarrierWithGroupSync();

        if (workIndex < nbPairs && matches)
        {
            const uint index = offset + prevAccum + gsWarp[warpIndex] + blockStartIndex;
            if (index < maxPairs)
            {
                aggOrActorPairs[index * 2u + 0u] = volA;
                aggOrActorPairs[index * 2u + 1u] = volB;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

// ---------------------------------------------------------------------------
// copyReports: copy the compacted actor found/lost reports to the mapped output
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void copyReports(uint3 dtid : SV_DispatchThreadID)
{
    Ptr<PxgBroadPhaseDescGpu> desc = Ptr<PxgBroadPhaseDescGpu>(gBp.bpDesc);
    const uint maxPairs = desc[0].max_found_lost_pairs;

    // sharedFoundPairIndex may exceed the buffer capacity; sharedFound/Lost-
    // AggPairIndex are the corrected aggregate counts written by stage 2.
    const uint nbCreatedPairs = min(desc[0].sharedFoundPairIndex, maxPairs) - desc[0].sharedFoundAggPairIndex;
    const uint nbLostPairs = min(desc[0].sharedLostPairIndex, maxPairs) - desc[0].sharedLostAggPairIndex;

    Ptr<uint> foundReports = Ptr<uint>(desc[0].foundActorPairReport);
    Ptr<uint> lostReports = Ptr<uint>(desc[0].lostActorPairReport);
    Ptr<uint> foundReportMap = Ptr<uint>(desc[0].foundPairReportMap);
    Ptr<uint> lostReportMap = Ptr<uint>(desc[0].lostPairReportMap);

    const bool foundOverflow = desc[0].sharedFoundPairIndex > maxPairs;
    const bool lostOverflow = desc[0].sharedLostPairIndex > maxPairs;
    if (dtid.x == 0)
    {
        // .cu stores the bool unconditionally; bytes 1-3 of the word are the
        // struct's tail padding, so the whole-word store is exact.
        desc[0].overflowFlagsWord = (foundOverflow || lostOverflow) ? 1u : 0u;
    }

    // PxgBroadPhasePair = 2 PxU32 elements
    const uint nbCreateElements = nbCreatedPairs * 2;
    const uint nbLostElements = nbLostPairs * 2;
    const uint stride = WorkgroupCount().x * 256u; // .cu: blockDim.x * gridDim.x

    for (uint i = dtid.x; i < nbCreateElements; i += stride)
        foundReportMap[i] = foundReports[i];
    for (uint i = dtid.x; i < nbLostElements; i += stride)
        lostReportMap[i] = lostReports[i];
}
