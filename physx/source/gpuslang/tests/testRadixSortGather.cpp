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

// Verification of radixSortGather.slang: the rank-indexed 64-bit gather kernels
// (radixSortCopy / radixSortCopy2 / radixSortCopyBits2 / radixSortDoubleCopy /
// radixSortDoubleCopyHigh32Bits), compared bit-exactly against C++ references,
// including the aggregate-sentinel (0xffffffff) masking of the "2"/"Bits2" forms.

#include "pxgTestFramework.h"

#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

const uint32_t AGG = 0xffffffffu;

PXG_TEST(radixSortGather_all)
{
    std::mt19937 rng(33010);
    const uint32_t n = 5000;
    const uint32_t GRID = 32;

    std::vector<uint64_t> inA(n), inB(n);
    for (uint32_t i = 0; i < n; ++i)
    {
        inA[i] = (uint64_t(rng()) << 32) | rng();
        inB[i] = (uint64_t(rng()) << 32) | rng();
    }
    // valid rank permutations
    std::vector<uint32_t> rankA(n), rankB(n);
    std::iota(rankA.begin(), rankA.end(), 0u); std::shuffle(rankA.begin(), rankA.end(), rng);
    std::iota(rankB.begin(), rankB.end(), 0u); std::shuffle(rankB.begin(), rankB.end(), rng);
    // rank with aggregate sentinels (~8%)
    std::vector<uint32_t> rankAgg = rankA;
    for (uint32_t i = 0; i < n; ++i) if (rng() % 100 < 8) rankAgg[i] = AGG;

    PxgSlangComputeDevice* dev = testDevice();
    const std::vector<uint32_t> strides = { 8, 4, 8, 4, 8, 4, 8, 4 };
    const uint32_t gridX = (n + 255) / 256 < GRID ? (n + 255) / 256 : GRID;

    struct GatherParams { uint32_t numKeys, lowBit, gridDimX; };

    // shared device buffers
    PxgBufferHandle bInA = createBufferFrom(dev, inA);
    PxgBufferHandle bInB = createBufferFrom(dev, inB);
    PxgBufferHandle bRankA = createBufferFrom(dev, rankA);
    PxgBufferHandle bRankB = createBufferFrom(dev, rankB);
    PxgBufferHandle bRankAgg = createBufferFrom(dev, rankAgg);
    std::vector<uint64_t> zero64(n, 0); std::vector<uint32_t> zero32(n, 0), dummy(16, 0);
    PxgBufferHandle bDummy = createBufferFrom(dev, dummy);

    auto make64 = [&]() { return createBufferFrom(dev, zero64); };
    auto make32 = [&]() { return createBufferFrom(dev, zero32); };

    // 1) radixSortCopy: out64[w] = inA[rankA[w]]
    {
        PxgBufferHandle out = make64();
        Kernel k("radixSortGather.spv", "radixSortCopy", strides, sizeof(GatherParams));
        GatherParams p = { n, 0, gridX };
        k.dispatch(gridX, { bInA, bRankA, out, bDummy, bDummy, bDummy, bDummy, bDummy }, p);
        dev->flush();
        const auto g = downloadBuffer<uint64_t>(dev, out, n);
        size_t mm = 0; for (uint32_t w = 0; w < n; ++w) if (g[w] != inA[rankA[w]]) ++mm;
        PXG_CHECK_MSG(mm == 0, "radixSortCopy: %zu mismatches", mm);
        dev->destroyBuffer(out);
    }
    // 2) radixSortCopy2: aggregate -> 0xffff..ffff, else gather
    {
        PxgBufferHandle out = make64();
        Kernel k("radixSortGather.spv", "radixSortCopy2", strides, sizeof(GatherParams));
        GatherParams p = { n, 0, gridX };
        k.dispatch(gridX, { bInA, bRankAgg, out, bDummy, bDummy, bDummy, bDummy, bDummy }, p);
        dev->flush();
        const auto g = downloadBuffer<uint64_t>(dev, out, n);
        size_t mm = 0;
        for (uint32_t w = 0; w < n; ++w)
        {
            const uint64_t exp = (rankAgg[w] == AGG) ? 0xffffffffffffffffull : inA[rankAgg[w]];
            if (g[w] != exp) ++mm;
        }
        PXG_CHECK_MSG(mm == 0, "radixSortCopy2: %zu mismatches", mm);
        dev->destroyBuffer(out);
    }
    // 3) radixSortCopyBits2: low + high word, aggregate -> 0xffffffff
    for (uint32_t lowBit : { 1u, 0u })
    {
        PxgBufferHandle out = make32();
        Kernel k("radixSortGather.spv", "radixSortCopyBits2", strides, sizeof(GatherParams));
        GatherParams p = { n, lowBit, gridX };
        k.dispatch(gridX, { bInA, bRankAgg, bDummy, out, bDummy, bDummy, bDummy, bDummy }, p);
        dev->flush();
        const auto g = downloadBuffer<uint32_t>(dev, out, n);
        size_t mm = 0;
        for (uint32_t w = 0; w < n; ++w)
        {
            uint32_t exp;
            if (rankAgg[w] == AGG) exp = AGG;
            else exp = lowBit ? uint32_t(inA[rankAgg[w]] & 0xffffffffull) : uint32_t(inA[rankAgg[w]] >> 32);
            if (g[w] != exp) ++mm;
        }
        PXG_CHECK_MSG(mm == 0, "radixSortCopyBits2 (lowBit=%u): %zu mismatches", lowBit, mm);
        dev->destroyBuffer(out);
    }
    // 4) radixSortDoubleCopy
    {
        PxgBufferHandle o0 = make64(), o1 = make64();
        Kernel k("radixSortGather.spv", "radixSortDoubleCopy", strides, sizeof(GatherParams));
        GatherParams p = { n, 0, gridX };
        k.dispatch(gridX, { bInA, bRankA, o0, bDummy, bInB, bRankB, o1, bDummy }, p);
        dev->flush();
        const auto g0 = downloadBuffer<uint64_t>(dev, o0, n);
        const auto g1 = downloadBuffer<uint64_t>(dev, o1, n);
        size_t mm = 0;
        for (uint32_t w = 0; w < n; ++w) { if (g0[w] != inA[rankA[w]]) ++mm; if (g1[w] != inB[rankB[w]]) ++mm; }
        PXG_CHECK_MSG(mm == 0, "radixSortDoubleCopy: %zu mismatches", mm);
        dev->destroyBuffer(o0); dev->destroyBuffer(o1);
    }
    // 5) radixSortDoubleCopyHigh32Bits
    {
        PxgBufferHandle o0 = make32(), o1 = make32();
        Kernel k("radixSortGather.spv", "radixSortDoubleCopyHigh32Bits", strides, sizeof(GatherParams));
        GatherParams p = { n, 0, gridX };
        k.dispatch(gridX, { bInA, bRankA, bDummy, o0, bInB, bRankB, bDummy, o1 }, p);
        dev->flush();
        const auto g0 = downloadBuffer<uint32_t>(dev, o0, n);
        const auto g1 = downloadBuffer<uint32_t>(dev, o1, n);
        size_t mm = 0;
        for (uint32_t w = 0; w < n; ++w)
        {
            if (g0[w] != uint32_t(inA[rankA[w]] >> 32)) ++mm;
            if (g1[w] != uint32_t(inB[rankB[w]] >> 32)) ++mm;
        }
        PXG_CHECK_MSG(mm == 0, "radixSortDoubleCopyHigh32Bits: %zu mismatches", mm);
        dev->destroyBuffer(o0); dev->destroyBuffer(o1);
    }

    for (PxgBufferHandle h : { bInA, bInB, bRankA, bRankB, bRankAgg, bDummy }) dev->destroyBuffer(h);
}

} // namespace
