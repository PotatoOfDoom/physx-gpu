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

// Verification of broadphaseRegionHistogram.slang against a C++ reference: the
// compute+output pair implements a device-wide exclusive prefix sum over a flat
// PxU32 array plus its grand total (std::exclusive_scan / accumulate).  Covers
// sizes that straddle the warp/block/grid-stride boundaries and a large-value
// case that would fail if any 16-bit packing crept in.  The start-region and
// region variants share this exact code path (only the bound array differs).

#include "pxgTestFramework.h"

#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct RegionHistParams
{
    uint32_t maxWorkIndex;
};

void runRegionHistCase(const char* tag, uint32_t seed, uint32_t nbProj, uint32_t maxVal)
{
    std::mt19937 rng(seed);
    std::vector<uint32_t> input(nbProj);
    for (uint32_t i = 0; i < nbProj; ++i)
        input[i] = rng() % (maxVal + 1);

    // reference: exclusive prefix sum + grand total (uint32 modular, matching GPU)
    std::vector<uint32_t> ref(nbProj);
    uint32_t acc = 0;
    for (uint32_t i = 0; i < nbProj; ++i)
    {
        ref[i] = acc;
        acc += input[i];
    }
    const uint32_t refTotal = acc;

    PxgSlangComputeDevice* dev = testDevice();

    std::vector<uint32_t> blockInit(32, 0xdeadbeefu);
    std::vector<uint32_t> totalInit(1, 0xdeadbeefu);

    PxgBufferHandle hBuf = createBufferFrom(dev, input);
    PxgBufferHandle bBuf = createBufferFrom(dev, blockInit);
    PxgBufferHandle tBuf = createBufferFrom(dev, totalInit);

    RegionHistParams p = {};
    p.maxWorkIndex = nbProj;

    const std::vector<uint32_t> strides(3, 4);
    const std::vector<PxgBufferHandle> buffers = { hBuf, bBuf, tBuf };

    Kernel k1("broadphaseRegionHistogram.spv", "computeRegionsHistogram", strides, sizeof(RegionHistParams));
    k1.dispatch(32, buffers, p);
    dev->barrier();

    Kernel k2("broadphaseRegionHistogram.spv", "outputRegionsHistogram", strides, sizeof(RegionHistParams));
    k2.dispatch(32, buffers, p);
    dev->flush();

    const auto gHist = downloadBuffer<uint32_t>(dev, hBuf, nbProj);
    const uint32_t gTotal = downloadBuffer<uint32_t>(dev, tBuf, 1)[0];

    size_t mism = 0;
    uint32_t firstBad = 0xffffffffu;
    for (uint32_t i = 0; i < nbProj; ++i)
    {
        if (gHist[i] != ref[i])
        {
            if (firstBad == 0xffffffffu) firstBad = i;
            ++mism;
        }
    }
    if (mism)
        PXG_CHECK_MSG(mism == 0, "[%s n=%u] %zu prefix mismatches (first at %u: gpu=%u ref=%u)", tag, nbProj, mism,
                      firstBad, gHist[firstBad], ref[firstBad]);
    else
        PXG_CHECK_MSG(mism == 0, "[%s n=%u] %zu prefix mismatches", tag, nbProj, mism);
    PXG_CHECK_MSG(gTotal == refTotal, "[%s n=%u] total: gpu=%u ref=%u", tag, nbProj, gTotal, refTotal);

    for (PxgBufferHandle h : { hBuf, bBuf, tBuf })
        dev->destroyBuffer(h);
}

PXG_TEST(broadphaseRegionHistogram_prefixSum)
{
    // small counts (0..7); sizes straddling warp (32), block (256) and
    // grid-stride tile (256*32 = 8192) boundaries
    runRegionHistCase("small-100", 17010, 100, 7);
    runRegionHistCase("small-260", 17011, 260, 7);
    runRegionHistCase("exact-8192", 17012, 8192, 7);
    runRegionHistCase("straddle-8193", 17013, 8193, 7);
    runRegionHistCase("multi-iter-20000", 17014, 20000, 7);
    // wide values (> 16 bits) with the total kept < 2^32 -> catches any packing
    runRegionHistCase("wide-values", 17015, 9001, (1u << 20));
}

} // namespace
