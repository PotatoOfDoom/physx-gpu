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

// Verification of npPairManagement.slang: the sparse-remove keep/drop core
// (initializeKeepDropBuffer -> markKeepDropBuff -> processKeepDropBuff ->
// accumulateKeepDrop), which must leave the run-sum buffer holding the
// device-wide EXCLUSIVE prefix sum of the keep/drop marks, and initializeManifolds
// (broadcast one source structure into N copies).  Both are compared bit-exactly
// against C++ references.

#include "pxgTestFramework.h"

#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct PairMgmtParams { uint32_t totalCount, nbToRemove, gridStride, dataSize, nbTimesToReplicate; };
struct F4 { float x, y, z, w; };

PXG_TEST(npPairManagement_keepDropScan)
{
    std::mt19937 rng(39010);
    const uint32_t N = 9000;
    const uint32_t K = 1200;
    const uint32_t NUM_BLOCKS = 32;
    const uint32_t newArraySize = N - K;

    // K distinct remove indices across [0, N)
    std::vector<uint32_t> perm(N);
    for (uint32_t i = 0; i < N; ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<uint32_t> removeIndices(perm.begin(), perm.begin() + K);

    // ---- C++ reference: init + mark, then exclusive prefix sum ----
    std::vector<uint32_t> marks(N);
    for (uint32_t i = 0; i < N; ++i) marks[i] = (i < newArraySize) ? 0u : 1u;
    for (uint32_t j = 0; j < K; ++j)
    {
        const uint32_t idx = removeIndices[j];
        marks[idx] = (idx < newArraySize) ? 1u : 0u;
    }
    std::vector<uint32_t> refRunSum(N);
    {
        uint32_t run = 0;
        for (uint32_t i = 0; i < N; ++i) { refRunSum[i] = run; run += marks[i]; }
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t gridStride = NUM_BLOCKS * 256;

    std::vector<uint32_t> runInit(N, 0), blockInit(NUM_BLOCKS, 0), dummy(16, 0);
    std::vector<F4> dummyF4(16, F4{ 0, 0, 0, 0 });
    PxgBufferHandle bRun = createBufferFrom(dev, runInit);
    PxgBufferHandle bRemove = createBufferFrom(dev, removeIndices);
    PxgBufferHandle bBlock = createBufferFrom(dev, blockInit);
    PxgBufferHandle bMDst = createBufferFrom(dev, dummyF4);
    PxgBufferHandle bMSrc = createBufferFrom(dev, dummyF4);

    const std::vector<uint32_t> strides = { 4, 4, 4, 16, 16 };
    const std::vector<PxgBufferHandle> binds = { bRun, bRemove, bBlock, bMDst, bMSrc };
    PairMgmtParams p = { N, K, gridStride, 0, 0 };

    Kernel k1("npPairManagement.spv", "initializeKeepDropBuffer", strides, sizeof(PairMgmtParams));
    k1.dispatch(NUM_BLOCKS, binds, p); dev->flush();
    Kernel k2("npPairManagement.spv", "markKeepDropBuff", strides, sizeof(PairMgmtParams));
    k2.dispatch(NUM_BLOCKS, binds, p); dev->flush();
    Kernel k3("npPairManagement.spv", "processKeepDropBuff", strides, sizeof(PairMgmtParams));
    k3.dispatch(NUM_BLOCKS, binds, p); dev->flush();
    Kernel k4("npPairManagement.spv", "accumulateKeepDrop", strides, sizeof(PairMgmtParams));
    k4.dispatch(NUM_BLOCKS, binds, p); dev->flush();

    const auto g = downloadBuffer<uint32_t>(dev, bRun, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i) if (g[i] != refRunSum[i]) ++mm;
    PXG_CHECK_MSG(mm == 0, "keep/drop run-sum: %zu / %u mismatches", mm, N);

    // getNbSwapsRequired == run-sum[newArraySize] == number of removed elems in kept region
    uint32_t refNbSwaps = 0;
    for (uint32_t i = 0; i < newArraySize; ++i) refNbSwaps += marks[i];
    PXG_CHECK_MSG(g[newArraySize] == refNbSwaps, "nbSwaps: gpu %u vs ref %u", g[newArraySize], refNbSwaps);

    for (PxgBufferHandle h : binds) dev->destroyBuffer(h);
}

PXG_TEST(npPairManagement_initializeManifolds)
{
    std::mt19937 rng(39020);
    const uint32_t NUM_BLOCKS = 32;
    const uint32_t dataSize = 64;                 // 4 float4 per structure
    const uint32_t nbF4 = dataSize / 16;          // 4
    const uint32_t nbTimesToReplicate = 5000;
    const uint32_t dstElems = nbF4 * nbTimesToReplicate + 64; // + preserved tail

    std::vector<F4> src(nbF4);
    auto rf = [&]() { return float(int(rng() % 20000) - 10000) * 0.01f; };
    for (auto& v : src) v = { rf(), rf(), rf(), rf() };

    std::vector<F4> dstInit(dstElems, F4{ -7.f, -7.f, -7.f, -7.f });

    // ---- reference: dst[c*nbF4 + e] = src[e] for each copy c ----
    std::vector<F4> refDst = dstInit;
    for (uint32_t c = 0; c < nbTimesToReplicate; ++c)
        for (uint32_t e = 0; e < nbF4; ++e)
            refDst[c * nbF4 + e] = src[e];

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<uint32_t> dummy(16, 0);
    PxgBufferHandle bRun = createBufferFrom(dev, dummy);
    PxgBufferHandle bRemove = createBufferFrom(dev, dummy);
    PxgBufferHandle bBlock = createBufferFrom(dev, dummy);
    PxgBufferHandle bMDst = createBufferFrom(dev, dstInit);
    PxgBufferHandle bMSrc = createBufferFrom(dev, src);

    const std::vector<uint32_t> strides = { 4, 4, 4, 16, 16 };
    PairMgmtParams p = { 0, 0, 0, dataSize, nbTimesToReplicate };
    Kernel k("npPairManagement.spv", "initializeManifolds", strides, sizeof(PairMgmtParams));
    k.dispatch(NUM_BLOCKS, { bRun, bRemove, bBlock, bMDst, bMSrc }, p);
    dev->flush();

    const auto g = downloadBuffer<F4>(dev, bMDst, dstElems);
    size_t mm = 0;
    for (uint32_t e = 0; e < dstElems; ++e)
    {
        const F4& a = g[e];
        const F4& r = refDst[e];
        if (a.x != r.x || a.y != r.y || a.z != r.z || a.w != r.w) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "initializeManifolds: %zu / %u mismatches", mm, dstElems);

    for (PxgBufferHandle h : { bRun, bRemove, bBlock, bMDst, bMSrc }) dev->destroyBuffer(h);
}

} // namespace
