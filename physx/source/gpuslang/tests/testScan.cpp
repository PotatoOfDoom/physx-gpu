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

// Verification of scan.slang against the semantics of the CUDA kernels
// scanPerBlockKernel / addBlockSumsKernel (algorithms.cu).

#include "pxgTestFramework.h"

#include <random>

namespace
{

using namespace pxgtest;

const uint32_t kBlockSize = 1024;

struct ScanParams
{
    uint32_t length;
    uint32_t exclusiveScan;
    uint32_t writeResult;
    uint32_t writeTotalSum;
};

// Device-wide scan composed exactly like the CUDA pipeline (see the header
// comment in scan.slang).
void runDeviceScan(const std::vector<uint32_t>& input, bool exclusive,
                   std::vector<uint32_t>& result, uint32_t& totalSum)
{
    PxgSlangComputeDevice* dev = testDevice();

    const uint32_t n = uint32_t(input.size());
    const uint32_t numBlocks = (n + kBlockSize - 1) / kBlockSize;
    PXG_CHECK(numBlocks <= kBlockSize); // single-level composition limit

    Kernel scanKernel("scan.spv", "scanPerBlockKernel", { 4, 4, 4, 4 }, sizeof(ScanParams));
    Kernel addKernel("scan.spv", "addBlockSumsKernel", { 4, 4, 4, 4 }, sizeof(ScanParams));

    PxgBufferHandle data = createBufferFrom(dev, input);
    PxgBufferHandle resultBuf = dev->createBuffer(n * sizeof(uint32_t));
    PxgBufferHandle partialSums = dev->createBuffer(numBlocks * sizeof(uint32_t));
    PxgBufferHandle partialPartials = dev->createBuffer(sizeof(uint32_t));
    PxgBufferHandle totalBuf = dev->createBuffer(sizeof(uint32_t));

    const uint32_t zero = 0;
    dev->upload(totalBuf, &zero, sizeof(zero));

    // 1. per-block scan of the input
    ScanParams params = { n, exclusive ? 1u : 0u, 1u, 1u };
    scanKernel.dispatch(numBlocks, { data, resultBuf, partialSums, totalBuf }, params);
    dev->barrier();

    // 2. exclusive scan of the per-block sums (in place, single block)
    ScanParams params2 = { numBlocks, 1u, 1u, 0u };
    scanKernel.dispatch(1, { partialSums, partialSums, partialPartials, totalBuf }, params2);
    dev->barrier();

    // 3. uniform add of the scanned block sums
    ScanParams params3 = { n, 0u, 1u, 1u };
    addKernel.dispatch(numBlocks, { resultBuf, resultBuf /*unused*/, partialSums, totalBuf }, params3);
    dev->flush();

    result = downloadBuffer<uint32_t>(dev, resultBuf, n);
    totalSum = downloadBuffer<uint32_t>(dev, totalBuf, 1)[0];

    dev->destroyBuffer(data);
    dev->destroyBuffer(resultBuf);
    dev->destroyBuffer(partialSums);
    dev->destroyBuffer(partialPartials);
    dev->destroyBuffer(totalBuf);
}

void referenceScan(const std::vector<uint32_t>& input, bool exclusive,
                   std::vector<uint32_t>& result, uint32_t& totalSum)
{
    result.resize(input.size());
    uint32_t running = 0;
    for (size_t i = 0; i < input.size(); ++i)
    {
        if (exclusive)
        {
            result[i] = running;
            running += input[i];
        }
        else
        {
            running += input[i];
            result[i] = running;
        }
    }
    totalSum = running;
}

void checkScan(uint32_t n, bool exclusive, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::vector<uint32_t> input(n);
    for (uint32_t& v : input)
        v = rng() & 0xFFFFu; // keep single values small; the running sum still exercises the full 32-bit range for large n

    std::vector<uint32_t> gpuResult;
    uint32_t gpuTotal = 0;
    runDeviceScan(input, exclusive, gpuResult, gpuTotal);

    std::vector<uint32_t> refResult;
    uint32_t refTotal = 0;
    referenceScan(input, exclusive, refResult, refTotal);

    PXG_CHECK_MSG(gpuTotal == refTotal, "n=%u exclusive=%d total: gpu=%u ref=%u", n, int(exclusive), gpuTotal, refTotal);

    size_t mismatches = 0;
    for (uint32_t i = 0; i < n; ++i)
    {
        if (gpuResult[i] != refResult[i])
        {
            if (mismatches < 5)
                PXG_CHECK_MSG(false, "n=%u exclusive=%d i=%u: gpu=%u ref=%u", n, int(exclusive), i, gpuResult[i], refResult[i]);
            ++mismatches;
        }
    }
    PXG_CHECK_MSG(mismatches == 0, "n=%u exclusive=%d: %zu mismatches", n, int(exclusive), mismatches);
}

PXG_TEST(scan_exclusive_small)
{
    checkScan(1, true, 1);
    checkScan(7, true, 2);
    checkScan(333, true, 3);
}

PXG_TEST(scan_exclusive_blockBoundaries)
{
    checkScan(1023, true, 4);
    checkScan(1024, true, 5);
    checkScan(1025, true, 6);
    checkScan(4096, true, 7);
}

PXG_TEST(scan_exclusive_large)
{
    checkScan(100000, true, 8);
    checkScan((1u << 18) + 39, true, 9);
}

PXG_TEST(scan_inclusive)
{
    checkScan(1, false, 10);
    checkScan(1024, false, 11);
    checkScan(12345, false, 12);
    checkScan(100000, false, 13);
}

} // namespace
