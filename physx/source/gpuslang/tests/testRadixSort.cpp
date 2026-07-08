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

// Verification of radixSort.slang against the semantics of the CUDA radix
// sort (RadixSort.cuh / radixSortImpl.cu): a stable 8-pass LSD sort of
// 32-bit keys with 32-bit rank payloads, orchestrated on the host exactly
// like PxgRadixSort::sort.

#include "pxgTestFramework.h"

#include <algorithm>
#include <numeric>
#include <random>

namespace
{

using namespace pxgtest;

const uint32_t kNumBlocks = 32; // PxgRadixSortKernelGridDim::RADIX_SORT
const uint32_t kRadixSize = 16;

struct RadixSortParams
{
    uint32_t numKeys;
    uint32_t startBit;
};

// Runs `numPasses` 4-bit passes over (keys, ranks), ping-ponging buffers like
// PxgRadixSort::sort.  Returns the sorted keys and ranks.
void runRadixSort(std::vector<uint32_t>& keys, std::vector<uint32_t>& ranks, uint32_t numPasses = 8)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t n = uint32_t(keys.size());

    Kernel histogram("radixSort.spv", "radixSortHistogram", { 4, 4, 4, 4, 4 }, sizeof(RadixSortParams));
    Kernel calcRanks("radixSort.spv", "radixSortCalculateRanks", { 4, 4, 4, 4, 4 }, sizeof(RadixSortParams));

    PxgBufferHandle keysA = createBufferFrom(dev, keys);
    PxgBufferHandle ranksA = createBufferFrom(dev, ranks);
    PxgBufferHandle keysB = dev->createBuffer(n * sizeof(uint32_t));
    PxgBufferHandle ranksB = dev->createBuffer(n * sizeof(uint32_t));
    PxgBufferHandle radixCounts = dev->createBuffer(kRadixSize * kNumBlocks * sizeof(uint32_t));

    PxgBufferHandle inKeys = keysA, inRanks = ranksA, outKeys = keysB, outRanks = ranksB;

    for (uint32_t pass = 0; pass < numPasses; ++pass)
    {
        const RadixSortParams params = { n, pass * 4 };
        histogram.dispatch(kNumBlocks, { inKeys, inRanks, outKeys, outRanks, radixCounts }, params);
        dev->barrier();
        calcRanks.dispatch(kNumBlocks, { inKeys, inRanks, outKeys, outRanks, radixCounts }, params);
        dev->barrier();
        std::swap(inKeys, outKeys);
        std::swap(inRanks, outRanks);
    }
    dev->flush();

    keys = downloadBuffer<uint32_t>(dev, inKeys, n);
    ranks = downloadBuffer<uint32_t>(dev, inRanks, n);

    dev->destroyBuffer(keysA);
    dev->destroyBuffer(ranksA);
    dev->destroyBuffer(keysB);
    dev->destroyBuffer(ranksB);
    dev->destroyBuffer(radixCounts);
}

void checkRadixSort(uint32_t n, uint32_t seed, uint32_t keyMask = 0xFFFFFFFFu)
{
    std::mt19937 rng(seed);
    std::vector<uint32_t> keys(n);
    for (uint32_t& k : keys)
        k = rng() & keyMask;

    std::vector<uint32_t> ranks(n);
    std::iota(ranks.begin(), ranks.end(), 0u);

    // reference: stable sort of (key, original index)
    std::vector<uint32_t> order(n);
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(),
                     [&](uint32_t a, uint32_t b) { return keys[a] < keys[b]; });

    std::vector<uint32_t> gpuKeys = keys;
    std::vector<uint32_t> gpuRanks = ranks;
    runRadixSort(gpuKeys, gpuRanks);

    size_t mismatches = 0;
    for (uint32_t i = 0; i < n; ++i)
    {
        const uint32_t expectedKey = keys[order[i]];
        const uint32_t expectedRank = order[i]; // stability: exact original index order
        if (gpuKeys[i] != expectedKey || gpuRanks[i] != expectedRank)
        {
            if (mismatches < 5)
                PXG_CHECK_MSG(false, "n=%u i=%u: gpu=(%08x,%u) ref=(%08x,%u)", n, i, gpuKeys[i],
                              gpuRanks[i], expectedKey, expectedRank);
            ++mismatches;
        }
    }
    PXG_CHECK_MSG(mismatches == 0, "n=%u seed=%u: %zu mismatches", n, seed, mismatches);
}

PXG_TEST(radixSort_small)
{
    checkRadixSort(1, 100);
    checkRadixSort(3, 101);
    checkRadixSort(64, 102);
    checkRadixSort(1023, 103);
}

PXG_TEST(radixSort_unaligned)
{
    // sizes that are not multiples of 4 exercise the uint4 tail handling
    checkRadixSort(4095, 110);
    checkRadixSort(4097, 111);
    checkRadixSort(65537, 112);
    checkRadixSort(99991, 113);
}

PXG_TEST(radixSort_blockChunking)
{
    // sizes around the 32-block x 1024-thread x 4-key partition boundaries
    checkRadixSort(32 * 1024 * 4, 120);
    checkRadixSort(32 * 1024 * 4 + 1, 121);
    checkRadixSort(32 * 1024 * 4 - 1, 122);
}

PXG_TEST(radixSort_duplicateHeavy)
{
    // few distinct keys -> massive duplicate runs; the rank payload check
    // verifies per-pass stability
    checkRadixSort(50000, 130, 0x7u);
    checkRadixSort(50000, 131, 0xF000u);
    checkRadixSort(4096, 132, 0x0u); // all keys identical
}

PXG_TEST(radixSort_largeRandom)
{
    checkRadixSort(1u << 20, 140);
}

// 64-bit keys: sort the low words (8 passes), gather the high words in rank
// order via radixSortCopyHigh32Bits, then sort those stably (8 more passes).
// The final rank array is the 64-bit stable order — the same chaining
// PxgRadixSort uses with the CUDA kernels.
void checkRadixSort64(uint32_t n, uint32_t seed, uint64_t keyMask = ~0ull)
{
    PxgSlangComputeDevice* dev = testDevice();

    std::mt19937_64 rng(seed);
    std::vector<uint64_t> keys64(n);
    for (uint64_t& k : keys64)
        k = rng() & keyMask;

    // low words + identity ranks
    std::vector<uint32_t> lowKeys(n), ranks(n);
    for (uint32_t i = 0; i < n; ++i)
        lowKeys[i] = uint32_t(keys64[i] & 0xFFFFFFFFu);
    std::iota(ranks.begin(), ranks.end(), 0u);

    // pass 1: sort by low word
    runRadixSort(lowKeys, ranks);

    // gather high words in rank order on the GPU
    Kernel copyHigh("radixSort64.spv", "radixSortCopyHigh32Bits", { 8, 4, 4, 4 }, 0);

    PxgBufferHandle keys64Buf = createBufferFrom(dev, keys64);
    PxgBufferHandle highBuf = dev->createBuffer(n * sizeof(uint32_t));
    PxgBufferHandle rankBuf = createBufferFrom(dev, ranks);
    std::vector<uint32_t> numKeysHost = { n };
    PxgBufferHandle numKeysBuf = createBufferFrom(dev, numKeysHost);

    copyHigh.dispatchNoParams((n + 255) / 256, { keys64Buf, highBuf, rankBuf, numKeysBuf });
    dev->flush();

    std::vector<uint32_t> highKeys = downloadBuffer<uint32_t>(dev, highBuf, n);

    // pass 2: stable sort of the high words, rank payload threads through
    runRadixSort(highKeys, ranks);

    // reference: stable sort of the 64-bit keys
    std::vector<uint32_t> order(n);
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(),
                     [&](uint32_t a, uint32_t b) { return keys64[a] < keys64[b]; });

    size_t mismatches = 0;
    for (uint32_t i = 0; i < n; ++i)
    {
        const uint64_t gpuKey = keys64[ranks[i]];
        const uint64_t refKey = keys64[order[i]];
        if (gpuKey != refKey || ranks[i] != order[i])
        {
            if (mismatches < 5)
                PXG_CHECK_MSG(false, "n=%u i=%u: gpu=(%016llx,%u) ref=(%016llx,%u)", n, i,
                              (unsigned long long)gpuKey, ranks[i], (unsigned long long)refKey, order[i]);
            ++mismatches;
        }
    }
    PXG_CHECK_MSG(mismatches == 0, "64-bit n=%u seed=%u: %zu mismatches", n, seed, mismatches);

    dev->destroyBuffer(keys64Buf);
    dev->destroyBuffer(highBuf);
    dev->destroyBuffer(rankBuf);
    dev->destroyBuffer(numKeysBuf);
}

PXG_TEST(radixSort64_random)
{
    checkRadixSort64(1, 150);
    checkRadixSort64(4097, 151);
    checkRadixSort64(100003, 152);
}

PXG_TEST(radixSort64_duplicateHeavy)
{
    // duplicated high words force the stability of the second sort chain
    checkRadixSort64(50000, 153, 0x0000000FFFFFFFFFull);
    // duplicated low words
    checkRadixSort64(50000, 154, 0xFFFFFFFF0000000Full);
}

} // namespace
