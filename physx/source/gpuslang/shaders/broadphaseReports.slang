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
// Slang port of the broad-phase found/lost pair report accumulation pipeline
// (gpubroadphase/src/CUDA/broadphase.cu):
//   - accumulateReportsStage_1: per-block count of the aggregate (or actor)
//     pairs in each of the four report categories (found/lost x agg/actor).
//   - accumulateReportsStage_2: exclusive prefix over the per-block counts,
//     then a stable compaction of each category's matching pairs into its
//     output array (foundAgg/foundActor/lostAgg/lostActor).
//   - copyReports: copy the compacted actor found/lost pairs to the mapped
//     output buffers and raise the overflow flag.
//
// A pair is an "aggregate" report unless BOTH of its volumes are single actors
// (VolumeData::isSingleActor()); otherwise it is an "actor" report.  The four
// categories are selected by the dispatch's Y grid index (blockIdx.y):
//   0 = found/agg, 1 = found/actor, 2 = lost/agg, 3 = lost/actor.
//
// Portability: the CUDA warp __ballot_sync+__popc reductions/scans become
// groupshared 32-lane scans; the fixed 32-block device grid and its
// between-block accumulation are preserved (BLOCK = 256, 8 warps).

static const uint BLOCK = 256;
static const uint NUM_BLOCKS = 32;
static const uint WARP = 32;
static const uint WARPS_PER_BLOCK = BLOCK / WARP; // 8
static const uint PX_INVALID_U32 = 0xffffffffu;

// mirrors PxgBroadPhasePair (8 bytes)
struct PxgBroadPhasePair
{
    uint mVolA;
    uint mVolB;
}

// mirrors Bp::VolumeData (16 bytes): { void* mUserData; PxU32 mAggregate; pad }
struct VolumeData
{
    uint2 mUserData;
    uint mAggregate;
    uint pad;
}

struct ReportParams
{
    uint numFound;    // sharedFoundPairIndex
    uint numLost;     // sharedLostPairIndex
    uint maxPairs;    // max_found_lost_pairs
    uint maxAggPairs; // max_found_lost_agg_pairs
    uint hasVolumeData;
}

[[vk::push_constant]]
ConstantBuffer<ReportParams> gParams;

RWStructuredBuffer<PxgBroadPhasePair> gFoundPairs;          // binding 0
RWStructuredBuffer<PxgBroadPhasePair> gLostPairs;           // binding 1
RWStructuredBuffer<VolumeData> gVolumeData;                 // binding 2
RWStructuredBuffer<uint> gAggReportBlock;                   // binding 3 (flat [2][32])
RWStructuredBuffer<uint> gActorReportBlock;                 // binding 4 (flat [2][32])
RWStructuredBuffer<PxgBroadPhasePair> gFoundAggPairReport;  // binding 5
RWStructuredBuffer<PxgBroadPhasePair> gFoundActorPairReport;// binding 6
RWStructuredBuffer<PxgBroadPhasePair> gLostAggPairReport;   // binding 7
RWStructuredBuffer<PxgBroadPhasePair> gLostActorPairReport; // binding 8
RWStructuredBuffer<uint> gSharedAggIndex;                   // binding 9 ([0]=found,[1]=lost)
RWStructuredBuffer<uint> gFoundReportMap;                   // binding 10
RWStructuredBuffer<uint> gLostReportMap;                    // binding 11
RWStructuredBuffer<uint> gOverflowFlag;                     // binding 12

bool isSingleActor(uint volIndex)
{
    if (gParams.hasVolumeData == 0)
        return true;
    return gVolumeData[volIndex].mAggregate == PX_INVALID_U32;
}

// classify a pair: returns true if it is an aggregate report (not both single
// actors), false if it is an actor report
bool isAggregatePair(PxgBroadPhasePair pair)
{
    const bool singleA = isSingleActor(pair.mVolA);
    const bool singleB = isSingleActor(pair.mVolB);
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
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const uint category = gid.y; // 0=found/agg,1=found/actor,2=lost/agg,3=lost/actor
    const bool found = (category / 2) == 0;
    const bool outputAggReport = (category % 2) == 0;

    const uint nbPairs = found ? gParams.numFound : gParams.numLost;
    const uint nbPairsSafe = min(nbPairs, gParams.maxPairs);

    const uint totalBlockRequired = (nbPairs + (BLOCK - 1)) / BLOCK;
    const uint numIterationPerBlock = (totalBlockRequired + (NUM_BLOCKS - 1)) / NUM_BLOCKS;

    // count matching pairs in this block's region
    uint localCount = 0;
    for (uint it = 0; it < numIterationPerBlock; ++it)
    {
        const uint workIndex = it * BLOCK + tid + numIterationPerBlock * blockIdx * BLOCK;
        if (workIndex < nbPairsSafe)
        {
            const PxgBroadPhasePair pair = found ? gFoundPairs[workIndex] : gLostPairs[workIndex];
            const bool isAgg = isAggregatePair(pair);
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
        const uint slot = (category / 2) * NUM_BLOCKS + blockIdx;
        if (outputAggReport)
            gAggReportBlock[slot] = gsBlockAccum;
        else
            gActorReportBlock[slot] = gsBlockAccum;
    }
}

// ---------------------------------------------------------------------------
// stage 2: prefix over block counts + stable compaction
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void accumulateReportsStage_2(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const uint category = gid.y;
    const bool found = (category / 2) == 0;
    const bool outputAggReport = (category % 2) == 0;
    const uint warpIndex = tid / WARP;

    const uint nbPairs = found ? gParams.numFound : gParams.numLost;
    const uint nbPairsSafe = min(nbPairs, gParams.maxPairs);
    const uint maxPairs = outputAggReport ? gParams.maxAggPairs : gParams.maxPairs;

    if (tid == 0)
        gsBlockAccum = 0;

    // warp 0: exclusive prefix over the 32 per-block counts
    if (tid < NUM_BLOCKS)
    {
        const uint slot = (category / 2) * NUM_BLOCKS + tid;
        gsLane[tid] = outputAggReport ? gAggReportBlock[slot] : gActorReportBlock[slot];
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
        // total agg count is recorded for the agg categories (0 = found, 2 = lost)
        if (outputAggReport)
            gSharedAggIndex[found ? 0u : 1u] = acc;
    }
    GroupMemoryBarrierWithGroupSync();

    const uint totalBlockRequired = (nbPairs + (BLOCK - 1)) / BLOCK;
    const uint numIterationPerBlock = (totalBlockRequired + (NUM_BLOCKS - 1)) / NUM_BLOCKS;
    const uint blockStartIndex = gsBlockHistogram[blockIdx];

    for (uint it = 0; it < numIterationPerBlock; ++it)
    {
        const uint workIndex = it * BLOCK + tid + numIterationPerBlock * blockIdx * BLOCK;

        bool matches = false;
        PxgBroadPhasePair pair;
        pair.mVolA = 0;
        pair.mVolB = 0;
        if (workIndex < nbPairsSafe)
        {
            pair = found ? gFoundPairs[workIndex] : gLostPairs[workIndex];
            const bool isAgg = isAggregatePair(pair);
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
                if (found)
                {
                    if (outputAggReport) gFoundAggPairReport[index] = pair;
                    else gFoundActorPairReport[index] = pair;
                }
                else
                {
                    if (outputAggReport) gLostAggPairReport[index] = pair;
                    else gLostActorPairReport[index] = pair;
                }
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
    const uint maxPairs = gParams.maxPairs;
    const uint nbCreatedPairs = min(gParams.numFound, maxPairs) - gSharedAggIndex[0];
    const uint nbLostPairs = min(gParams.numLost, maxPairs) - gSharedAggIndex[1];

    const bool foundOverflow = gParams.numFound > maxPairs;
    const bool lostOverflow = gParams.numLost > maxPairs;
    if (dtid.x == 0)
        gOverflowFlag[0] = (foundOverflow || lostOverflow) ? 1u : 0u;

    // PxgBroadPhasePair = 2 uints
    const uint nbCreateElements = nbCreatedPairs * 2;
    const uint nbLostElements = nbLostPairs * 2;
    const uint stride = BLOCK * NUM_BLOCKS;

    for (uint i = dtid.x; i < nbCreateElements; i += stride)
    {
        const PxgBroadPhasePair p = gFoundActorPairReport[i / 2];
        gFoundReportMap[i] = (i & 1) == 0 ? p.mVolA : p.mVolB;
    }
    for (uint i = dtid.x; i < nbLostElements; i += stride)
    {
        const PxgBroadPhasePair p = gLostActorPairReport[i / 2];
        gLostReportMap[i] = (i & 1) == 0 ? p.mVolA : p.mVolB;
    }
}
