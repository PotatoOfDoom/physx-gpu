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

// npPairManagement.slang
//
// Slang port of the sparse-remove keep/drop compaction core used by the
// narrow-phase contact-manager removal (gpunarrowphase/src/CUDA/pairManagement.cu
// -> gpucommon/src/CUDA/SparseRemove.cuh), plus the manifold-replication
// initializer.  This tranche ports the first four removeContactManagers stages
// (which build the run-sum buffer that the later swap stage consumes) and
// initializeManifolds:
//
//   initializeKeepDropBuffer (Stage1): buffer[i] = i < (N-K) ? 0 : 1
//   markKeepDropBuff        (Stage2):  for each of the K remove indices,
//       buffer[removeIndex[j]] = (removeIndex[j] < N-K) ? 1 : 0
//   processKeepDropBuff     (Stage3):  per-block exclusive scan of the buffer,
//       writing each block's total to the cross-block accumulator
//   accumulateKeepDrop      (Stage4):  add the cross-block exclusive prefix, so
//       buffer[i] becomes the device-wide EXCLUSIVE prefix sum of the marks
//   initializeManifolds: broadcast one dataSize-byte source structure into
//       nbTimesToReplicate contiguous copies of the destination array
//
// The underlying scanKernel1of2/2of2 (reduction.cuh) build an exclusive prefix
// sum; that output is decomposition-independent, so the two passes are the same
// proven groupshared two-pass exclusive scan used elsewhere in this port
// (BLOCK = 256, NUM_BLOCKS = 32).  getNbSwapsRequired/getSwapIndices and the
// actual data swap (Stage5) are a follow-up tranche.

static const uint BLOCK = 256;
static const uint NUM_BLOCKS = 32;
static const uint WARP = 32;
static const uint WARPS_PER_BLOCK = BLOCK / WARP; // 8

struct PairMgmtParams
{
    uint totalCount;         // N (number of pairs)
    uint nbToRemove;         // K
    uint gridStride;         // blockDim*gridDim for the grid-stride stages
    uint dataSize;           // initializeManifolds: source size in bytes
    uint nbTimesToReplicate; // initializeManifolds
}

[[vk::push_constant]]
ConstantBuffer<PairMgmtParams> gParams;

RWStructuredBuffer<uint>   gRunSum;         // binding 0  keep/drop buffer -> run-sum
RWStructuredBuffer<uint>   gRemoveIndices;  // binding 1  indices to remove
RWStructuredBuffer<uint>   gBlockTotals;    // binding 2  cross-block accumulator [NUM_BLOCKS]
RWStructuredBuffer<float4> gManifoldDst;    // binding 3  initializeManifolds destination
RWStructuredBuffer<float4> gManifoldSrc;    // binding 4  initializeManifolds source

// ---------------------------------------------------------------------------
// Stage 1: initializeKeepDropBuffer
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void initializeKeepDropBuffer(uint3 dtid : SV_DispatchThreadID)
{
    const uint newArraySize = gParams.totalCount - gParams.nbToRemove;
    const uint stride = gParams.gridStride;
    for (uint i = dtid.x; i < gParams.totalCount; i += stride)
        gRunSum[i] = (i < newArraySize) ? 0u : 1u;
}

// ---------------------------------------------------------------------------
// Stage 2: markKeepDropBuff
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void markKeepDropBuff(uint3 dtid : SV_DispatchThreadID)
{
    const uint newArraySize = gParams.totalCount - gParams.nbToRemove;
    const uint stride = gParams.gridStride;
    for (uint i = dtid.x; i < gParams.nbToRemove; i += stride)
    {
        const uint index = gRemoveIndices[i];
        gRunSum[index] = (index < newArraySize) ? 1u : 0u;
    }
}

// ---------------------------------------------------------------------------
// exclusive-scan scaffolding (uint)
// ---------------------------------------------------------------------------
groupshared uint gsPairs[BLOCK];
groupshared uint gsWarpPairs[WARPS_PER_BLOCK];
groupshared uint gsBlockPairs;
groupshared uint gsBlockAccum[NUM_BLOCKS];

void warpScanPairs(uint tid)
{
    for (uint off = 1; off < WARP; off <<= 1)
    {
        const uint lane = tid & (WARP - 1);
        const uint add = (lane >= off) ? gsPairs[tid - off] : 0u;
        GroupMemoryBarrierWithGroupSync();
        gsPairs[tid] += add;
        GroupMemoryBarrierWithGroupSync();
    }
}

void scanWarpTotalsPairs(uint tid)
{
    const bool active = tid < WARPS_PER_BLOCK;
    const uint p0 = active ? gsWarpPairs[tid] : 0u;
    uint pInc = p0;
    for (uint off = 1; off < WARPS_PER_BLOCK; off <<= 1)
    {
        const bool take = active && tid >= off;
        const uint pa = take ? gsWarpPairs[tid - off] : 0u;
        GroupMemoryBarrierWithGroupSync();
        pInc += pa;
        if (active) gsWarpPairs[tid] = pInc;
        GroupMemoryBarrierWithGroupSync();
    }
    GroupMemoryBarrierWithGroupSync();
    if (active) gsWarpPairs[tid] = pInc - p0;
    if (tid == WARPS_PER_BLOCK - 1) gsBlockPairs += pInc;
}

// ---------------------------------------------------------------------------
// Stage 3: processKeepDropBuff — per-block exclusive scan, write block totals
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void processKeepDropBuff(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const uint n = gParams.totalCount;
    const uint warpIndex = tid / WARP;

    const uint totalBlockRequired = (n + (BLOCK - 1)) / BLOCK;
    const uint numIterationPerBlock = (totalBlockRequired + (NUM_BLOCKS - 1)) / NUM_BLOCKS;

    if (tid == 0)
        gsBlockPairs = 0u;
    GroupMemoryBarrierWithGroupSync();

    for (uint it = 0; it < numIterationPerBlock; ++it)
    {
        const uint workIndex = tid + it * BLOCK + BLOCK * blockIdx * numIterationPerBlock;

        uint val = 0u;
        if (workIndex < n)
            val = gRunSum[workIndex];

        gsPairs[tid] = val;
        GroupMemoryBarrierWithGroupSync();
        warpScanPairs(tid);
        const uint accum = gsPairs[tid] - val; // exclusive within warp

        if ((tid & (WARP - 1)) == WARP - 1)
            gsWarpPairs[warpIndex] = accum + val;

        const uint prevBlock = gsBlockPairs;
        GroupMemoryBarrierWithGroupSync();

        scanWarpTotalsPairs(tid);

        GroupMemoryBarrierWithGroupSync();

        if (workIndex < n)
            gRunSum[workIndex] = accum + prevBlock + gsWarpPairs[warpIndex];
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0)
        gBlockTotals[blockIdx] = gsBlockPairs;
}

// ---------------------------------------------------------------------------
// Stage 4: accumulateKeepDrop — add cross-block exclusive prefix
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void accumulateKeepDrop(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint blockIdx = gid.x;
    const uint n = gParams.totalCount;

    if (tid < NUM_BLOCKS)
        gsBlockAccum[tid] = gBlockTotals[tid];
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0)
    {
        uint acc = 0u;
        for (uint i = 0; i < NUM_BLOCKS; ++i)
        {
            const uint t = gsBlockAccum[i];
            gsBlockAccum[i] = acc;
            acc += t;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    const uint totalBlockRequired = (n + (BLOCK - 1)) / BLOCK;
    const uint numIterationPerBlock = (totalBlockRequired + (NUM_BLOCKS - 1)) / NUM_BLOCKS;
    const uint blockAccum = gsBlockAccum[blockIdx];

    for (uint it = 0; it < numIterationPerBlock; ++it)
    {
        const uint workIndex = it * BLOCK + tid + numIterationPerBlock * blockIdx * BLOCK;
        if (workIndex < n)
            gRunSum[workIndex] = gRunSum[workIndex] + blockAccum;
    }
}

// ---------------------------------------------------------------------------
// initializeManifolds: broadcast one source structure into N copies
// ---------------------------------------------------------------------------
groupshared float gsSourceData[1024]; // MaxStructureSize (4096 bytes) / sizeof(float)

[shader("compute")]
[numthreads(256, 1, 1)]
void initializeManifolds(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint nbThreadsPerElement = gParams.dataSize / 16u; // dataSize / sizeof(float4)
    const uint nbPerBlock = BLOCK / nbThreadsPerElement;
    const uint MaxThreadIdx = nbPerBlock * nbThreadsPerElement;
    const uint nbTimesToReplicate = gParams.nbTimesToReplicate;
    const uint nbBlocksRequired = (nbTimesToReplicate + nbPerBlock - 1) / nbPerBlock;

    if (tid < nbThreadsPerElement)
    {
        const float4 value = gManifoldSrc[tid];
        gsSourceData[tid] = value.x;
        gsSourceData[tid + nbThreadsPerElement] = value.y;
        gsSourceData[tid + 2u * nbThreadsPerElement] = value.z;
        gsSourceData[tid + 3u * nbThreadsPerElement] = value.w;
    }

    const uint threadReadIdx = tid % nbThreadsPerElement;

    GroupMemoryBarrierWithGroupSync();

    for (uint a = gid.x; a < nbBlocksRequired; a += NUM_BLOCKS)
    {
        const uint startBlockIdx = nbPerBlock * a;
        const uint startPtr = startBlockIdx * nbThreadsPerElement; // float4 offset
        const uint idx = startBlockIdx + tid / nbThreadsPerElement;

        if (tid < MaxThreadIdx && idx < nbTimesToReplicate)
        {
            gManifoldDst[startPtr + tid] = float4(gsSourceData[threadReadIdx],
                                                  gsSourceData[threadReadIdx + nbThreadsPerElement],
                                                  gsSourceData[threadReadIdx + 2u * nbThreadsPerElement],
                                                  gsSourceData[threadReadIdx + 3u * nbThreadsPerElement]);
        }
    }
}
