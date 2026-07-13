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

// Verification of npCvxLostFound.slang: prepareLostFoundPairs_Stage1/_Stage2
// (cudaGJKEPA.cu).  Exclusive scan over the 2*numPairs flag values, compaction
// of the flagged managers' {nbPatches, prevPatches, statusFlag} + opaque
// handle at the scan offsets, lost count (prefix at idx == numPairs) -> x and
// grand total -> y.  Everything integer: comparisons exact.

#include "pxgTestFramework.h"

#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct LostFoundParams { uint32_t numPairs, pad0, pad1, pad2; };

static void runOne(uint32_t numPairs, uint32_t seed, float density)
{
    std::mt19937 rng(seed);
    const uint32_t n = 2 * numPairs;
    const uint32_t SENTINEL = 0xdeadbeefu;

    std::vector<uint32_t> flags(n);
    for (uint32_t i = 0; i < n; ++i) flags[i] = (float(rng() % 1000) / 1000.f < density) ? 1u : 0u;
    std::vector<uint32_t> mgrWords(numPairs), mgrHandles(numPairs);
    for (uint32_t i = 0; i < numPairs; ++i) { mgrWords[i] = rng(); mgrHandles[i] = rng(); }

    // CPU reference: exclusive scan + compact
    std::vector<uint32_t> refCompact(n, SENTINEL), refPtrs(n, SENTINEL);
    uint32_t refX = 0xffffffffu, refY = 0, off = 0;
    for (uint32_t i = 0; i < n; ++i)
    {
        if (i == numPairs) refX = off;
        if (flags[i])
        {
            const uint32_t w = mgrWords[i % numPairs];
            const uint32_t nbPatches = w & 0xFF, statusFlag = (w >> 8) & 0xFF, prevPatches = (w >> 16) & 0xFF;
            refCompact[off] = nbPatches | (prevPatches << 8) | (statusFlag << 16);
            refPtrs[off] = mgrHandles[i % numPairs];
            ++off;
        }
    }
    refY = off;

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t NUM_BLOCKS = 32;

    std::vector<uint32_t> runInit(n, 0), blockInit(NUM_BLOCKS, 0);
    std::vector<uint32_t> compactInit(n, SENTINEL), ptrsInit(n, SENTINEL), ltInit(2, 0xffffffffu);

    PxgBufferHandle bFlags = createBufferFrom(dev, flags);
    PxgBufferHandle bRunSum = createBufferFrom(dev, runInit);
    PxgBufferHandle bBlocks = createBufferFrom(dev, blockInit);
    PxgBufferHandle bWords = createBufferFrom(dev, mgrWords);
    PxgBufferHandle bHandles = createBufferFrom(dev, mgrHandles);
    PxgBufferHandle bCompact = createBufferFrom(dev, compactInit);
    PxgBufferHandle bPtrs = createBufferFrom(dev, ptrsInit);
    PxgBufferHandle bLT = createBufferFrom(dev, ltInit);

    LostFoundParams p = { numPairs, 0, 0, 0 };
    std::vector<uint32_t> strides = { 4, 4, 4, 4, 4, 4, 4, 4 };
    std::vector<PxgBufferHandle> binds = { bFlags, bRunSum, bBlocks, bWords, bHandles, bCompact, bPtrs, bLT };

    Kernel k1("npCvxLostFound.spv", "lostFoundStage1", strides, sizeof(LostFoundParams));
    k1.dispatch(NUM_BLOCKS, binds, p); dev->flush();
    Kernel k2("npCvxLostFound.spv", "lostFoundStage2", strides, sizeof(LostFoundParams));
    k2.dispatch(NUM_BLOCKS, binds, p); dev->flush();

    const auto gCompact = downloadBuffer<uint32_t>(dev, bCompact, n);
    const auto gPtrs = downloadBuffer<uint32_t>(dev, bPtrs, n);
    const auto gLT = downloadBuffer<uint32_t>(dev, bLT, 2);

    size_t mmC = 0, mmP = 0;
    for (uint32_t i = 0; i < n; ++i)
    {
        if (gCompact[i] != refCompact[i]) ++mmC;
        if (gPtrs[i] != refPtrs[i]) ++mmP;
    }
    PXG_CHECK_MSG(mmC == 0, "lostFound(%u): %zu compacted mismatches", numPairs, mmC);
    PXG_CHECK_MSG(mmP == 0, "lostFound(%u): %zu handle mismatches", numPairs, mmP);
    PXG_CHECK_MSG(gLT[0] == refX, "lostFound(%u): x (lost) %u != %u", numPairs, gLT[0], refX);
    PXG_CHECK_MSG(gLT[1] == refY, "lostFound(%u): y (total) %u != %u", numPairs, gLT[1], refY);

    for (PxgBufferHandle h : { bFlags, bRunSum, bBlocks, bWords, bHandles, bCompact, bPtrs, bLT })
        dev->destroyBuffer(h);
}

PXG_TEST(npCvxLostFound_compact)
{
    runOne(1, 150001, 0.5f);       // minimal
    runOne(5, 150002, 0.8f);       // tiny, dense
    runOne(128, 150003, 0.4f);     // exact block fraction
    runOne(300, 150004, 0.3f);     // non-multiple
    runOne(4096, 150005, 0.5f);    // exactly one iteration per block
    runOne(6000, 150006, 0.35f);   // multi-iteration, non-multiple
    runOne(9000, 150007, 0.02f);   // sparse flags
    runOne(7777, 150008, 0.97f);   // nearly all flagged
}

} // namespace
