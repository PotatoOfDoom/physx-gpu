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

// Verification of scan4x4.slang, radixFourBit.slang and reorderData.slang:
// the PxGpuRadixSort pipeline (PxgAlgorithms.cpp) reassembled on the host
// exactly like PxGpuRadixSort::sort / computeBlockSum, checked against
// std::stable_sort, plus a standalone int4x4 scan test.

#include "pxgTestFramework.h"
#include "pxgGpuStructs.h"

#include <algorithm>
#include <numeric>
#include <random>

namespace
{

using namespace pxgtest;

const uint32_t kBlockSize = 512;

struct Int4x4
{
    uint32_t v[16];
};
static_assert(sizeof(Int4x4) == 64, "int4x4 mirror must be 64 bytes");

struct Scan4x4Params
{
    uint32_t length;
    uint32_t exclusiveScan;
    uint32_t writeResult;
    uint32_t writeTotalSum;
    uint32_t numGroups;
};

struct RadixFourBitParams
{
    uint32_t length;
    uint32_t passIndex;
    uint32_t writeTotalSum;
    uint32_t hasDependent;
};

// ---------------------------------------------------------------------------
// standalone int4x4 per-block scan vs component-wise reference
// ---------------------------------------------------------------------------
PXG_TEST(scan4x4_perBlock)
{
    std::mt19937 rng(1000);

    for (uint32_t n : { 1u, 511u, 512u, 513u, 2048u, 5000u })
    {
        std::vector<Int4x4> input(n);
        for (Int4x4& e : input)
            for (uint32_t& c : e.v)
                c = rng() & 0xFFu;

        const uint32_t numBlocks = (n + kBlockSize - 1) / kBlockSize;

        PxgSlangComputeDevice* dev = testDevice();
        Kernel scan("scan4x4.spv", "scanPerBlockKernel4x4", { 64, 64, 64, 64 }, sizeof(Scan4x4Params));

        PxgBufferHandle dataBuf = createBufferFrom(dev, input);
        PxgBufferHandle resultBuf = dev->createBuffer(n * sizeof(Int4x4));
        PxgBufferHandle partialsBuf = dev->createBuffer(numBlocks * sizeof(Int4x4));
        PxgBufferHandle totalBuf = dev->createBuffer(sizeof(Int4x4));

        const Scan4x4Params params = { n, 0u /*inclusive*/, 1u, 1u, numBlocks };
        scan.dispatch(numBlocks, { dataBuf, resultBuf, partialsBuf, totalBuf }, params);
        dev->flush();

        const std::vector<Int4x4> gpuResult = downloadBuffer<Int4x4>(dev, resultBuf, n);
        const std::vector<Int4x4> gpuPartials = downloadBuffer<Int4x4>(dev, partialsBuf, numBlocks);

        // reference: per-block inclusive scan, component-wise
        size_t mismatches = 0;
        for (uint32_t b = 0; b < numBlocks; ++b)
        {
            Int4x4 running = {};
            const uint32_t begin = b * kBlockSize;
            const uint32_t end = std::min(n, begin + kBlockSize);
            for (uint32_t i = begin; i < end; ++i)
            {
                for (int c = 0; c < 16; ++c)
                    running.v[c] += input[i].v[c];
                for (int c = 0; c < 16; ++c)
                    if (gpuResult[i].v[c] != running.v[c])
                        ++mismatches;
            }
            // partialSums = whole-block total incl. zero padding
            for (int c = 0; c < 16; ++c)
                if (gpuPartials[b].v[c] != running.v[c])
                    ++mismatches;
        }
        PXG_CHECK_MSG(mismatches == 0, "n=%u: %zu component mismatches", n, mismatches);

        dev->destroyBuffer(dataBuf);
        dev->destroyBuffer(resultBuf);
        dev->destroyBuffer(partialsBuf);
        dev->destroyBuffer(totalBuf);
    }
}

// ---------------------------------------------------------------------------
// full PxGpuRadixSort pipeline (count -> computeBlockSum -> reorder)
// ---------------------------------------------------------------------------

// mirrors PxGpuRadixSort::sort for numBlocks <= 512 (no computeBlockSum
// recursion — matches the CUDA path for up to 512*512 elements)
void runGpuRadixSort(std::vector<uint32_t>& keys, std::vector<uint32_t>& reorderMap,
                     uint32_t numBitsToSort)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t n = uint32_t(keys.size());
    const uint32_t numBlocks = (n + kBlockSize - 1) / kBlockSize;
    PXG_CHECK(numBlocks <= kBlockSize);

    uint32_t numPasses = (numBitsToSort + 3) / 4;
    numPasses += numPasses % 2; // toNextHigherEvenNumber

    Kernel count("radixFourBit.spv", "radixFourBitCountPerBlockKernel", { 4, 4, 4, 64, 64, 4, 4 },
                 sizeof(RadixFourBitParams));
    Kernel reorder("radixFourBit.spv", "radixFourBitReorderKernel", { 4, 4, 4, 64, 64, 4, 4 },
                   sizeof(RadixFourBitParams));
    Kernel scan4x4("scan4x4.spv", "scanPerBlockKernel4x4", { 64, 64, 64, 64 }, sizeof(Scan4x4Params));
    Kernel addSums("scan4x4.spv", "addBlockSumsKernel4x4", { 64, 64, 64, 64 }, sizeof(Scan4x4Params));

    PxgBufferHandle keysA = createBufferFrom(dev, keys);
    PxgBufferHandle keysB = dev->createBuffer(n * sizeof(uint32_t));
    PxgBufferHandle offsets = dev->createBuffer(n * sizeof(uint32_t));
    PxgBufferHandle blockSums = dev->createBuffer(numBlocks * sizeof(Int4x4));
    PxgBufferHandle blockSumScan = dev->createBuffer(numBlocks * sizeof(Int4x4));
    PxgBufferHandle blockSumsL2 = dev->createBuffer(sizeof(Int4x4)); // recursion partials (unused at depth 1)
    PxgBufferHandle totalSum = dev->createBuffer(sizeof(Int4x4));
    PxgBufferHandle depA = dev->createBuffer(n * sizeof(uint32_t));
    PxgBufferHandle depB = dev->createBuffer(n * sizeof(uint32_t));

    PxgBufferHandle inKeys = keysA, outKeys = keysB, inDep = depA, outDep = depB;

    for (uint32_t pass = 0; pass < numPasses; ++pass)
    {
        // 1. radixFourBitCountPerBlockLaunch
        const RadixFourBitParams countParams = { n, pass, 1u, 0u };
        count.dispatch(numBlocks, { inKeys, offsets, outKeys, blockSums, totalSum, inDep, outDep },
                       countParams);
        dev->barrier();

        // 2. computeBlockSum: exclusive scan of the per-block histograms ...
        const Scan4x4Params scanParams = { numBlocks, 1u, 1u, 0u, 1u };
        scan4x4.dispatch(1, { blockSums, blockSumScan, blockSumsL2, totalSum }, scanParams);
        dev->barrier();

        // ... then fold the last block's scanned base into the totals and
        // exclusively scan the 16 slot counters (addBlockSums with data=NULL)
        const Scan4x4Params addParams = { n, 0u, 0u /*data NULL*/, 1u, numBlocks };
        addSums.dispatch(numBlocks, { blockSums /*dummy*/, blockSums /*unused*/, blockSumScan, totalSum },
                         addParams);
        dev->barrier();

        // 3. radixFourBitReorderLaunch
        const RadixFourBitParams reorderParams = { n, pass, 0u, 1u };
        reorder.dispatch(numBlocks, { inKeys, offsets, outKeys, blockSumScan, totalSum, inDep, outDep },
                         reorderParams);
        dev->barrier();

        std::swap(inKeys, outKeys);
        std::swap(inDep, outDep);
    }
    dev->flush();

    keys = downloadBuffer<uint32_t>(dev, inKeys, n);
    reorderMap = downloadBuffer<uint32_t>(dev, inDep, n);

    dev->destroyBuffer(keysA);
    dev->destroyBuffer(keysB);
    dev->destroyBuffer(offsets);
    dev->destroyBuffer(blockSums);
    dev->destroyBuffer(blockSumScan);
    dev->destroyBuffer(blockSumsL2);
    dev->destroyBuffer(totalSum);
    dev->destroyBuffer(depA);
    dev->destroyBuffer(depB);
}

void checkGpuRadixSort(uint32_t n, uint32_t seed, uint32_t numBits)
{
    std::mt19937 rng(seed);
    const uint32_t mask = numBits >= 32 ? 0xFFFFFFFFu : ((1u << numBits) - 1u);
    std::vector<uint32_t> keys(n);
    for (uint32_t& k : keys)
        k = rng() & mask;

    std::vector<uint32_t> order(n);
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(),
                     [&](uint32_t a, uint32_t b) { return keys[a] < keys[b]; });

    std::vector<uint32_t> gpuKeys = keys;
    std::vector<uint32_t> gpuMap;
    runGpuRadixSort(gpuKeys, gpuMap, numBits);

    size_t mismatches = 0;
    for (uint32_t i = 0; i < n; ++i)
    {
        if (gpuKeys[i] != keys[order[i]] || gpuMap[i] != order[i])
        {
            if (mismatches < 5)
                PXG_CHECK_MSG(false, "n=%u bits=%u i=%u: gpu=(%08x,%u) ref=(%08x,%u)", n, numBits, i,
                              gpuKeys[i], gpuMap[i], keys[order[i]], order[i]);
            ++mismatches;
        }
    }
    PXG_CHECK_MSG(mismatches == 0, "n=%u bits=%u: %zu mismatches", n, numBits, mismatches);
}

PXG_TEST(gpuRadixSort_fullWidth)
{
    checkGpuRadixSort(1, 1100, 32);
    checkGpuRadixSort(511, 1101, 32);
    checkGpuRadixSort(513, 1102, 32);
    checkGpuRadixSort(100000, 1103, 32);
}

PXG_TEST(gpuRadixSort_partialBits)
{
    // numBitsToSort < 32 rounds up to an even pass count (PxGpuRadixSort)
    checkGpuRadixSort(50000, 1110, 8);
    checkGpuRadixSort(50000, 1111, 10); // odd nibble count -> rounded to 12 bits' worth of passes
    checkGpuRadixSort(50000, 1112, 16);
}

PXG_TEST(gpuRadixSort_duplicateHeavy)
{
    checkGpuRadixSort(65536, 1120, 4);
    checkGpuRadixSort(4096, 1121, 32);
}

// ---------------------------------------------------------------------------
// reorderKernel: float4 payload gather through the reorder map
// ---------------------------------------------------------------------------
PXG_TEST(reorderKernel_gather)
{
    std::mt19937 rng(1200);
    const uint32_t n = 40000;

    std::vector<uint32_t> keys(n);
    for (uint32_t& k : keys)
        k = rng();

    std::vector<Float4> payload(n);
    for (uint32_t i = 0; i < n; ++i)
        payload[i] = Float4{ float(rng() & 0xFFFF), float(rng() & 0xFFFF), float(rng() & 0xFFFF), float(i) };

    std::vector<uint32_t> gpuKeys = keys;
    std::vector<uint32_t> gpuMap;
    runGpuRadixSort(gpuKeys, gpuMap, 32);

    PxgSlangComputeDevice* dev = testDevice();
    Kernel reorder("reorderData.spv", "reorderKernel", { 16, 16, 4 }, sizeof(uint32_t));

    PxgBufferHandle dataBuf = createBufferFrom(dev, payload);
    PxgBufferHandle outBuf = dev->createBuffer(n * sizeof(Float4));
    PxgBufferHandle mapBuf = createBufferFrom(dev, gpuMap);

    reorder.dispatch((n + kBlockSize - 1) / kBlockSize, { dataBuf, outBuf, mapBuf }, n);
    dev->flush();

    const std::vector<Float4> gpuOut = downloadBuffer<Float4>(dev, outBuf, n);

    size_t mismatches = 0;
    for (uint32_t i = 0; i < n; ++i)
    {
        const Float4& expected = payload[gpuMap[i]];
        if (std::memcmp(&gpuOut[i], &expected, sizeof(Float4)) != 0)
            ++mismatches;
    }
    PXG_CHECK_MSG(mismatches == 0, "%zu payload mismatches", mismatches);

    dev->destroyBuffer(dataBuf);
    dev->destroyBuffer(outBuf);
    dev->destroyBuffer(mapBuf);
}

} // namespace
