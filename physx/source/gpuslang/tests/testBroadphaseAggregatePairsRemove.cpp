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

// Verification of broadphaseAggregatePairsRemove.slang against a C++
// transcription of the removeAggPairsStage1/2 compaction: removed pairs (a
// non-zero remove-bitmap entry) are appended to the free-index list in
// increasing pair-index order (stable compaction), marked dead, and the total
// removed count is recorded.  Run end-to-end (both stages) and compared exactly.

#include "pxgTestFramework.h"

#include <algorithm>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

const uint32_t SENT = 0x77777777u;

struct RemoveParams
{
    uint32_t nbAggregatePairs;
    uint32_t numFreeIndices;
};

PXG_TEST(broadphaseAggregatePairsRemove_compact)
{
    std::mt19937 rng(28010);
    const uint32_t nbPairs = 20000, numFreeInds = 500;
    const uint32_t NUM_BLOCKS = 32;

    // remove bitmap: ~30% flagged, using assorted NON-ZERO values (ballot treats
    // any non-zero as remove) so the != 0 semantic is exercised.
    std::vector<uint32_t> bitmap(nbPairs, 0);
    std::vector<uint32_t> removed;
    for (uint32_t i = 0; i < nbPairs; ++i)
    {
        if (rng() % 100 < 30)
        {
            static const uint32_t nz[] = { 1u, 7u, 0xdeadu, 0x80000000u, 0xffffffffu };
            bitmap[i] = nz[rng() % 5];
            removed.push_back(i); // ascending by construction
        }
    }
    const uint32_t nbRemoved = uint32_t(removed.size());

    std::vector<uint32_t> aggPairs(nbPairs * 4);
    for (uint32_t i = 0; i < nbPairs * 4; ++i) aggPairs[i] = 1000 + i;
    std::vector<uint32_t> freeIndices(numFreeInds + nbRemoved, SENT);
    for (uint32_t i = 0; i < numFreeInds; ++i) freeIndices[i] = 300000 + i; // existing free list
    std::vector<uint32_t> histogram(NUM_BLOCKS, 0);
    std::vector<uint32_t> nbRemovedBuf = { SENT };

    // reference
    std::vector<uint32_t> rFree = freeIndices, rAgg = aggPairs;
    for (uint32_t k = 0; k < nbRemoved; ++k)
    {
        rFree[numFreeInds + k] = removed[k];
        rAgg[removed[k] * 4 + 3] = 1;
    }

    PxgSlangComputeDevice* dev = testDevice();
    PxgBufferHandle bBit = createBufferFrom(dev, bitmap);
    PxgBufferHandle bHist = createBufferFrom(dev, histogram);
    PxgBufferHandle bFree = createBufferFrom(dev, freeIndices);
    PxgBufferHandle bAgg = createBufferFrom(dev, aggPairs);
    PxgBufferHandle bNb = createBufferFrom(dev, nbRemovedBuf);

    RemoveParams p = { nbPairs, numFreeInds };
    const std::vector<uint32_t> strides(5, 4);
    Kernel k1("broadphaseAggregatePairsRemove.spv", "removeAggPairsStage1", strides, sizeof(RemoveParams));
    k1.dispatch(NUM_BLOCKS, { bBit, bHist, bFree, bAgg, bNb }, p);
    dev->flush();
    Kernel k2("broadphaseAggregatePairsRemove.spv", "removeAggPairsStage2", strides, sizeof(RemoveParams));
    k2.dispatch(NUM_BLOCKS, { bBit, bHist, bFree, bAgg, bNb }, p);
    dev->flush();

    const auto gFree = downloadBuffer<uint32_t>(dev, bFree, numFreeInds + nbRemoved);
    const auto gAgg = downloadBuffer<uint32_t>(dev, bAgg, nbPairs * 4);
    const auto gNb = downloadBuffer<uint32_t>(dev, bNb, 1);

    size_t fM = 0, aM = 0;
    for (uint32_t i = 0; i < numFreeInds + nbRemoved; ++i) if (gFree[i] != rFree[i]) ++fM;
    for (uint32_t i = 0; i < nbPairs * 4; ++i) if (gAgg[i] != rAgg[i]) ++aM;
    PXG_CHECK_MSG(gNb[0] == nbRemoved, "nbRemoved=%u exp %u", gNb[0], nbRemoved);
    PXG_CHECK_MSG(fM == 0, "%zu free-index mismatches (nbRemoved=%u)", fM, nbRemoved);
    PXG_CHECK_MSG(aM == 0, "%zu aggPair-isDead mismatches", aM);
    for (PxgBufferHandle h : { bBit, bHist, bFree, bAgg, bNb }) dev->destroyBuffer(h);
}

} // namespace
