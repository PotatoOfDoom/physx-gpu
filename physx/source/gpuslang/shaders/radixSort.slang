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

// radixSort.slang
//
// Slang port of the PhysX GPU radix sort
// (gpucommon/src/CUDA/RadixSort.cuh + radixSortImpl.cu):
//   - radixSortMultiBlockLaunch          -> radixSortHistogram
//   - radixSortMultiCalculateRanksLaunch -> radixSortCalculateRanks
//
// The sort is an LSD radix sort over 4-bit digits: 8 passes sort 32-bit keys
// together with a 32-bit rank payload, ping-ponging between two buffer pairs
// exactly like PxgRadixSort::sort on the host side.  Grid/block geometry is
// identical to CUDA (32 blocks x 1024 threads, PxgRadixSortKernelGridDim /
// PxgRadixSortKernelBlockDim) and the intermediate radix-count buffer keeps
// the CUDA layout: radixCounts[digit * 32 + block].
//
// Intentional differences from the CUDA implementation:
//   - The PxgRadixSortDesc structure embeds device pointers, which do not
//     exist under Vulkan/D3D12.  Buffers are bound explicitly and the scalar
//     arguments (numKeys, startBit) arrive as push constants.  Multi-desc
//     launches (blockIdx.y) become one dispatch per descriptor.
//   - Warp-shuffle scans are replaced by portable groupshared scans (see
//     pxgSlangScan.slang); "virtual warps" of 32 threads keep the same packed
//     8-bit-counter ranking scheme as the CUDA kernels.
//   - Out-of-range keys are skipped instead of being padded with 0xFFFFFFFF;
//     both approaches leave output[0..numKeys) identical (the CUDA padding
//     only ever occupied slots beyond numKeys).
//   - Results are scattered directly to global memory instead of being staged
//     through shared memory for write coalescing.  This is a performance
//     detail only; ordering (= stability) is unchanged.

import pxgSlangScan;

static const uint BLOCK_SIZE = 1024;  // PxgRadixSortKernelBlockDim::RADIX_SORT
static const uint NUM_BLOCKS = 32;    // PxgRadixSortKernelGridDim::RADIX_SORT
static const uint RADIX_SIZE = 16;    // 4-bit digits
static const uint WARPS_PER_BLOCK = BLOCK_SIZE / 32;

struct RadixSortParams
{
    uint numKeys;
    uint startBit;
};

[[vk::push_constant]]
ConstantBuffer<RadixSortParams> gParams;

RWStructuredBuffer<uint> gInputKeys;   // binding 0
RWStructuredBuffer<uint> gInputRanks;  // binding 1
RWStructuredBuffer<uint> gOutputKeys;  // binding 2
RWStructuredBuffer<uint> gOutputRanks; // binding 3
RWStructuredBuffer<uint> gRadixCounts; // binding 4: RADIX_SIZE * NUM_BLOCKS entries

groupshared uint gsHistogram[RADIX_SIZE];
groupshared uint gsDigitStart[RADIX_SIZE];             // running global write cursor per digit
groupshared uint gsWarpCounts[RADIX_SIZE * WARPS_PER_BLOCK];
groupshared uint gsIterTotals[RADIX_SIZE];

uint getRadix(uint key, uint startBit)
{
    return (key >> startBit) & 0xFu;
}

// Work partition per block, identical to radixSortSingleBlock /
// radixSortCalculateRanks: work is measured in uint4 units (4 keys), each
// block processes numIterationPerBlock * BLOCK_SIZE consecutive uint4s.
struct BlockChunk
{
    uint startU4;
    uint endU4;
    uint iterations;
};

BlockChunk computeBlockChunk(uint blockId, uint numKeys)
{
    const uint numKeysU4 = (numKeys + 3) / 4;
    const uint totalBlocksRequired = (numKeysU4 + (BLOCK_SIZE - 1)) / BLOCK_SIZE;
    const uint numIterationPerBlock = (totalBlocksRequired + (NUM_BLOCKS - 1)) / NUM_BLOCKS;

    BlockChunk c;
    c.startU4 = min(numIterationPerBlock * blockId * BLOCK_SIZE, numKeysU4);
    c.endU4 = min(c.startU4 + numIterationPerBlock * BLOCK_SIZE, numKeysU4);
    c.iterations = numIterationPerBlock;
    return c;
}

// ---------------------------------------------------------------------------
// Pass kernel 1: per-block digit histogram
// (CUDA: radixSortMultiBlockLaunch -> radixSortSingleBlock)
// Writes gRadixCounts[digit * NUM_BLOCKS + block] = number of keys in this
// block's chunk whose current 4-bit digit equals `digit`.
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void radixSortHistogram(uint3 groupThreadId : SV_GroupThreadID,
                        uint3 groupId : SV_GroupID)
{
    const uint tid = groupThreadId.x;
    const uint blockId = groupId.x;
    const uint numKeys = gParams.numKeys;
    const uint startBit = gParams.startBit;

    if (tid < RADIX_SIZE)
        gsHistogram[tid] = 0u;

    GroupMemoryBarrierWithGroupSync();

    const BlockChunk chunk = computeBlockChunk(blockId, numKeys);

    for (uint u4 = chunk.startU4 + tid; u4 < chunk.endU4; u4 += BLOCK_SIZE)
    {
        [ForceUnroll]
        for (uint c = 0; c < 4; ++c)
        {
            const uint keyIndex = u4 * 4 + c;
            if (keyIndex < numKeys)
            {
                const uint digit = getRadix(gInputKeys[keyIndex], startBit);
                InterlockedAdd(gsHistogram[digit], 1u);
            }
        }
    }

    GroupMemoryBarrierWithGroupSync();

    if (tid < RADIX_SIZE)
        gRadixCounts[tid * NUM_BLOCKS + blockId] = gsHistogram[tid];
}

// ---------------------------------------------------------------------------
// Pass kernel 2: stable reorder
// (CUDA: radixSortMultiCalculateRanksLaunch -> radixSortCalculateRanks)
//
// For every key in the block's chunk the stable output position is
//     globalDigitStart(digit)
//   + countOfDigitInEarlierBlocks(digit)
//   + stableRankWithinBlockChunk(digit)
// computed per 4096-key iteration via packed per-virtual-warp counting scans
// (4 x 8-bit counters per scan word — max 128 counts per warp fits 8 bits,
// the same packing radixSortCalculateRanks uses).
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void radixSortCalculateRanks(uint3 groupThreadId : SV_GroupThreadID,
                             uint3 groupId : SV_GroupID)
{
    const uint tid = groupThreadId.x;
    const uint blockId = groupId.x;
    const uint warpIndex = tid >> 5;
    const uint numKeys = gParams.numKeys;
    const uint startBit = gParams.startBit;

    // ---- Step 1: global write cursor per digit for this block ----
    // gsDigitStart[d] = sum over all digits d' < d of total(d')   (digit base)
    //                 + sum over blocks b < blockId of count[d][b] (block offset)
    if (tid < RADIX_SIZE)
    {
        const uint d = tid;
        uint blockPrefix = 0;
        uint digitTotal = 0;
        for (uint b = 0; b < NUM_BLOCKS; ++b)
        {
            const uint cnt = gRadixCounts[d * NUM_BLOCKS + b];
            if (b < blockId)
                blockPrefix += cnt;
            digitTotal += cnt;
        }
        gsIterTotals[d] = digitTotal;
        gsDigitStart[d] = blockPrefix;
    }

    GroupMemoryBarrierWithGroupSync();

    if (tid == 0)
    {
        uint running = 0;
        for (uint d = 0; d < RADIX_SIZE; ++d)
        {
            gsDigitStart[d] += running;
            running += gsIterTotals[d];
        }
    }

    GroupMemoryBarrierWithGroupSync();

    // ---- Step 2: iterate the chunk, ranking 4096 keys per iteration ----
    const BlockChunk chunk = computeBlockChunk(blockId, numKeys);

    for (uint iter = 0; iter < chunk.iterations; ++iter)
    {
        const uint u4 = chunk.startU4 + iter * BLOCK_SIZE + tid;
        const bool rowValid = u4 < chunk.endU4;

        uint keys[4];
        uint ranks[4];
        uint digits[4];
        bool valid[4];
        uint withinWarpRank[4];

        [ForceUnroll]
        for (uint c = 0; c < 4; ++c)
        {
            const uint keyIndex = u4 * 4 + c;
            valid[c] = rowValid && keyIndex < numKeys;
            keys[c] = valid[c] ? gInputKeys[keyIndex] : 0u;
            ranks[c] = valid[c] ? gInputRanks[keyIndex] : 0u;
            digits[c] = getRadix(keys[c], startBit);
            withinWarpRank[c] = 0u;
        }

        // Rank within the 32-lane virtual warp, 4 digits at a time packed as
        // 8-bit counters (mirrors the CUDA "accum" packing).
        for (uint group = 0; group < RADIX_SIZE; group += 4)
        {
            uint accum = 0;
            [ForceUnroll]
            for (uint c = 0; c < 4; ++c)
            {
                const uint rel = digits[c] - group;
                if (valid[c] && rel < 4)
                    accum += 1u << (rel << 3);
            }

            const uint inclusive = warp32InclusiveScan(tid, accum);

            if ((tid & 31u) == 31u)
            {
                [ForceUnroll]
                for (uint dd = 0; dd < 4; ++dd)
                    gsWarpCounts[(group + dd) * WARPS_PER_BLOCK + warpIndex] = (inclusive >> (dd << 3)) & 0xFFu;
            }

            uint exclusive = inclusive - accum;

            [ForceUnroll]
            for (uint c = 0; c < 4; ++c)
            {
                const uint rel = digits[c] - group;
                if (valid[c] && rel < 4)
                {
                    withinWarpRank[c] = (exclusive >> (rel << 3)) & 0xFFu;
                    exclusive += 1u << (rel << 3);
                }
            }

            GroupMemoryBarrierWithGroupSync();
        }

        // Exclusive-scan the per-warp totals for each digit and derive the
        // iteration totals (16 digits handled by 16 threads).
        if (tid < RADIX_SIZE)
        {
            uint running = 0;
            for (uint w = 0; w < WARPS_PER_BLOCK; ++w)
            {
                const uint t = gsWarpCounts[tid * WARPS_PER_BLOCK + w];
                gsWarpCounts[tid * WARPS_PER_BLOCK + w] = running;
                running += t;
            }
            gsIterTotals[tid] = running;
        }

        GroupMemoryBarrierWithGroupSync();

        // Scatter to the stable output position.
        [ForceUnroll]
        for (uint c = 0; c < 4; ++c)
        {
            if (valid[c])
            {
                const uint d = digits[c];
                const uint outIndex = gsDigitStart[d]
                    + gsWarpCounts[d * WARPS_PER_BLOCK + warpIndex]
                    + withinWarpRank[c];
                gOutputKeys[outIndex] = keys[c];
                gOutputRanks[outIndex] = ranks[c];
            }
        }

        GroupMemoryBarrierWithGroupSync();

        // Advance the global cursor by what this iteration consumed.
        if (tid < RADIX_SIZE)
            gsDigitStart[tid] += gsIterTotals[tid];

        GroupMemoryBarrierWithGroupSync();
    }
}
