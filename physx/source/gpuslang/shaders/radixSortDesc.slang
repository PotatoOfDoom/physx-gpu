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

// radixSortDesc.slang — P5 Step 4 (BDA, engine launch path): the descriptor-
// based multi-launch radix sort the broad phase drives
// (gpucommon/src/CUDA/radixSortImpl.cu):
//   radixSortMultiBlockLaunchWithCount(PxgRadixSortDesc* desc, PxU32 numKeys,
//                                      PxU32 startBit)
//   radixSortMultiCalculateRanksLaunchWithCount(same ABI)
// launched with grid (32, 3, 1): blockIdx.y selects one of THREE contiguous
// PxgRadixSortDesc entries (one per broad-phase axis), 8 passes with
// startBit += 4 and a desc ping-pong of +3 per odd pass (runRadixSort,
// PxgCudaBroadPhaseSap.cpp:648-670).
//
// The ranking core is the PROVEN implementation from radixSort.slang
// (verified bit-exact against the CPU reference by testRadixSort), transcribed
// to buffer-device-address access: the buffers arrive as device pointers
// inside the descriptor, exactly as CUDA reads them.  Same intentional
// differences as radixSort.slang: portable groupshared scans instead of warp
// shuffles, direct global scatter instead of shared-staged writes (stability
// unchanged), out-of-range keys skipped instead of padded.
//
// P6.2 adds the WithoutCount twins (radixSortImpl.cu:69 / :81), IDENTICAL to
// the WithCount entrypoints except the CUDA signature drops the numKeys push
// param and sources it from desc[blockIdx.y].count (the descriptor's PxU32
// count field, mirrored at offset 40 in PxgRadixSortDescGpu) instead.  Their
// ranking cores are copied verbatim from the WithCount entrypoints (only
// numKeys is re-sourced), so the proven WithCount SPIR-V is byte-untouched.

import pxgSlangScan;

static const uint BLOCK_SIZE = 1024;  // PxgRadixSortKernelBlockDim::RADIX_SORT
static const uint NUM_BLOCKS = 32;    // PxgRadixSortKernelGridDim::RADIX_SORT
static const uint RADIX_SIZE = 16;    // 4-bit digits
static const uint WARPS_PER_BLOCK = BLOCK_SIZE / 32;

// mirrors PxgRadixSortDesc (gpucommon/include/PxgRadixSortDesc.h):
// 5 device pointers + PxU32 count; C sizeof = 48 (8-align padded).
struct PxgRadixSortDescGpu
{
    uint64_t inputKeys;
    uint64_t inputRanks;
    uint64_t outputKeys;
    uint64_t outputRanks;
    uint64_t radixBlockCounts;
    uint count;
    uint pad0;
}

// CUDA ABI: (PxgRadixSortDesc* desc, PxU32 numKeys, PxU32 startBit) -> 16 bytes.
struct RadixSortDescParams
{
    uint64_t desc;  // base of the 3-entry descriptor array (blockIdx.y indexes)
    uint numKeys;
    uint startBit;
}

[[vk::push_constant]]
ConstantBuffer<RadixSortDescParams> gRsd;

groupshared uint gsHistogram[RADIX_SIZE];
groupshared uint gsDigitStart[RADIX_SIZE];
groupshared uint gsWarpCounts[RADIX_SIZE * WARPS_PER_BLOCK];
groupshared uint gsIterTotals[RADIX_SIZE];

uint getRadix(uint key, uint startBit)
{
    return (key >> startBit) & 0xFu;
}

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

// mirrors radixSortMultiBlockLaunchWithCount -> radixSortSingleBlock:
// per-block digit histogram of desc[blockIdx.y]'s input keys.
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void radixSortMultiBlockLaunchWithCount(uint3 groupThreadId : SV_GroupThreadID,
                                        uint3 groupId : SV_GroupID)
{
    const uint tid = groupThreadId.x;
    const uint blockId = groupId.x;
    const uint numKeys = gRsd.numKeys;
    const uint startBit = gRsd.startBit;

    PxgRadixSortDescGpu desc = (Ptr<PxgRadixSortDescGpu>(gRsd.desc))[groupId.y];
    Ptr<uint> inputKeys = Ptr<uint>(desc.inputKeys);
    Ptr<uint> radixCounts = Ptr<uint>(desc.radixBlockCounts);

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
                const uint digit = getRadix(inputKeys[keyIndex], startBit);
                InterlockedAdd(gsHistogram[digit], 1u);
            }
        }
    }

    GroupMemoryBarrierWithGroupSync();

    if (tid < RADIX_SIZE)
        radixCounts[tid * NUM_BLOCKS + blockId] = gsHistogram[tid];
}

// mirrors radixSortMultiCalculateRanksLaunchWithCount -> radixSortCalculateRanks:
// stable reorder of desc[blockIdx.y]'s keys+ranks by the current 4-bit digit.
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void radixSortMultiCalculateRanksLaunchWithCount(uint3 groupThreadId : SV_GroupThreadID,
                                                 uint3 groupId : SV_GroupID)
{
    const uint tid = groupThreadId.x;
    const uint blockId = groupId.x;
    const uint warpIndex = tid >> 5;
    const uint numKeys = gRsd.numKeys;
    const uint startBit = gRsd.startBit;

    PxgRadixSortDescGpu desc = (Ptr<PxgRadixSortDescGpu>(gRsd.desc))[groupId.y];
    Ptr<uint> inputKeys = Ptr<uint>(desc.inputKeys);
    Ptr<uint> inputRanks = Ptr<uint>(desc.inputRanks);
    Ptr<uint> outputKeys = Ptr<uint>(desc.outputKeys);
    Ptr<uint> outputRanks = Ptr<uint>(desc.outputRanks);
    Ptr<uint> radixCounts = Ptr<uint>(desc.radixBlockCounts);

    // ---- Step 1: global write cursor per digit for this block ----
    if (tid < RADIX_SIZE)
    {
        const uint d = tid;
        uint blockPrefix = 0;
        uint digitTotal = 0;
        for (uint b = 0; b < NUM_BLOCKS; ++b)
        {
            const uint cnt = radixCounts[d * NUM_BLOCKS + b];
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
            keys[c] = valid[c] ? inputKeys[keyIndex] : 0u;
            ranks[c] = valid[c] ? inputRanks[keyIndex] : 0u;
            digits[c] = getRadix(keys[c], startBit);
            withinWarpRank[c] = 0u;
        }

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

        [ForceUnroll]
        for (uint c = 0; c < 4; ++c)
        {
            if (valid[c])
            {
                const uint d = digits[c];
                const uint outIndex = gsDigitStart[d]
                    + gsWarpCounts[d * WARPS_PER_BLOCK + warpIndex]
                    + withinWarpRank[c];
                outputKeys[outIndex] = keys[c];
                outputRanks[outIndex] = ranks[c];
            }
        }

        GroupMemoryBarrierWithGroupSync();

        if (tid < RADIX_SIZE)
            gsDigitStart[tid] += gsIterTotals[tid];

        GroupMemoryBarrierWithGroupSync();
    }
}

// ---------------------------------------------------------------------------
// WithoutCount twins (radixSortImpl.cu:69 / :81) — numKeys from desc.count.
// ---------------------------------------------------------------------------
// CUDA ABI: (PxgRadixSortDesc* desc, PxU32 startBit) -> 8 + 4 = 12, rounded to
// the 8-byte alignment -> 16-byte blob (argSizes {8,4}).
struct RadixSortDescNoCountParams
{
    uint64_t desc;  // base of the descriptor array (blockIdx.y indexes)
    uint startBit;
}

[[vk::push_constant]]
ConstantBuffer<RadixSortDescNoCountParams> gRsdNc;

// mirrors radixSortMultiBlockLaunchWithoutCount -> radixSortSingleBlock:
// per-block digit histogram of desc[blockIdx.y]'s input keys, numKeys sourced
// from desc.count (identical body to radixSortMultiBlockLaunchWithCount).
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void radixSortMultiBlockLaunchWithoutCount(uint3 groupThreadId : SV_GroupThreadID,
                                           uint3 groupId : SV_GroupID)
{
    const uint tid = groupThreadId.x;
    const uint blockId = groupId.x;
    const uint startBit = gRsdNc.startBit;

    PxgRadixSortDescGpu desc = (Ptr<PxgRadixSortDescGpu>(gRsdNc.desc))[groupId.y];
    const uint numKeys = desc.count;
    Ptr<uint> inputKeys = Ptr<uint>(desc.inputKeys);
    Ptr<uint> radixCounts = Ptr<uint>(desc.radixBlockCounts);

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
                const uint digit = getRadix(inputKeys[keyIndex], startBit);
                InterlockedAdd(gsHistogram[digit], 1u);
            }
        }
    }

    GroupMemoryBarrierWithGroupSync();

    if (tid < RADIX_SIZE)
        radixCounts[tid * NUM_BLOCKS + blockId] = gsHistogram[tid];
}

// mirrors radixSortMultiCalculateRanksLaunchWithoutCount ->
// radixSortCalculateRanks: stable reorder of desc[blockIdx.y]'s keys+ranks by
// the current 4-bit digit, numKeys sourced from desc.count (identical body to
// radixSortMultiCalculateRanksLaunchWithCount).
[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void radixSortMultiCalculateRanksLaunchWithoutCount(uint3 groupThreadId : SV_GroupThreadID,
                                                    uint3 groupId : SV_GroupID)
{
    const uint tid = groupThreadId.x;
    const uint blockId = groupId.x;
    const uint warpIndex = tid >> 5;
    const uint startBit = gRsdNc.startBit;

    PxgRadixSortDescGpu desc = (Ptr<PxgRadixSortDescGpu>(gRsdNc.desc))[groupId.y];
    const uint numKeys = desc.count;
    Ptr<uint> inputKeys = Ptr<uint>(desc.inputKeys);
    Ptr<uint> inputRanks = Ptr<uint>(desc.inputRanks);
    Ptr<uint> outputKeys = Ptr<uint>(desc.outputKeys);
    Ptr<uint> outputRanks = Ptr<uint>(desc.outputRanks);
    Ptr<uint> radixCounts = Ptr<uint>(desc.radixBlockCounts);

    // ---- Step 1: global write cursor per digit for this block ----
    if (tid < RADIX_SIZE)
    {
        const uint d = tid;
        uint blockPrefix = 0;
        uint digitTotal = 0;
        for (uint b = 0; b < NUM_BLOCKS; ++b)
        {
            const uint cnt = radixCounts[d * NUM_BLOCKS + b];
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
            keys[c] = valid[c] ? inputKeys[keyIndex] : 0u;
            ranks[c] = valid[c] ? inputRanks[keyIndex] : 0u;
            digits[c] = getRadix(keys[c], startBit);
            withinWarpRank[c] = 0u;
        }

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

        [ForceUnroll]
        for (uint c = 0; c < 4; ++c)
        {
            if (valid[c])
            {
                const uint d = digits[c];
                const uint outIndex = gsDigitStart[d]
                    + gsWarpCounts[d * WARPS_PER_BLOCK + warpIndex]
                    + withinWarpRank[c];
                outputKeys[outIndex] = keys[c];
                outputRanks[outIndex] = ranks[c];
            }
        }

        GroupMemoryBarrierWithGroupSync();

        if (tid < RADIX_SIZE)
            gsDigitStart[tid] += gsIterTotals[tid];

        GroupMemoryBarrierWithGroupSync();
    }
}
