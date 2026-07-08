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

// Verification of broadphaseAggregateDirty.slang against C++ transcriptions of
// updateDirtyAggregate (new-bound-index scatter + full/selective staging->
// persistent merge) and clearDirtyAggregates (clear new flag, fix prevSize).
// PxgAggregate is modelled as a flat 8-uint record; the destination bound pool
// is laid out at shuffled offsets so a source/destination index swap is caught.

#include "pxgTestFramework.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

const uint32_t K = 8; // AGG_STRIDE
enum { A_BOUNDBASE = 0, A_PREVSIZE, A_MINDEX, A_SIZE, A_FILTER, A_MENV, A_PREVCMP, A_ISNEW };

struct DirtyParams
{
    uint32_t numDirtyAggregates;
};

// distinct random subset of [0,nbAgg) of length numDirty
std::vector<uint32_t> distinctIndices(std::mt19937& rng, uint32_t nbAgg, uint32_t numDirty)
{
    std::vector<uint32_t> all(nbAgg);
    std::iota(all.begin(), all.end(), 0u);
    std::shuffle(all.begin(), all.end(), rng);
    all.resize(numDirty);
    return all;
}

PXG_TEST(broadphaseAggregateDirty_update)
{
    std::mt19937 rng(25010);
    const uint32_t nbAgg = 1000, numDirty = 400;
    std::vector<uint32_t> dirtyIdx = distinctIndices(rng, nbAgg, numDirty);

    // persistent aggregates: distinct recognizable fields, ~half new
    std::vector<uint32_t> agg(nbAgg * K);
    for (uint32_t i = 0; i < nbAgg; ++i)
    {
        agg[i * K + A_BOUNDBASE] = 800000 + i;
        agg[i * K + A_PREVSIZE]  = 810000 + i;
        agg[i * K + A_MINDEX]    = 820000 + i;
        agg[i * K + A_SIZE]      = 1 + rng() % 50;
        agg[i * K + A_FILTER]    = 830000 + i;
        agg[i * K + A_MENV]      = 840000 + i;
        agg[i * K + A_PREVCMP]   = 850000 + i;
        agg[i * K + A_ISNEW]     = rng() % 2;
    }

    // dirty aggregates + pools
    std::vector<uint32_t> size(numDirty), startIndex(numDirty), boundBase(numDirty);
    uint32_t total = 0;
    for (uint32_t w = 0; w < numDirty; ++w) { size[w] = 1 + rng() % 40; startIndex[w] = total; total += size[w]; }
    // destination pool laid out in a shuffled order so boundBase != startIndex
    std::vector<uint32_t> perm(numDirty);
    std::iota(perm.begin(), perm.end(), 0u);
    std::shuffle(perm.begin(), perm.end(), rng);
    uint32_t dest = 0;
    for (uint32_t k = 0; k < numDirty; ++k) { boundBase[perm[k]] = dest; dest += size[perm[k]]; }

    std::vector<uint32_t> dirty(numDirty * K);
    for (uint32_t w = 0; w < numDirty; ++w)
    {
        dirty[w * K + A_BOUNDBASE] = boundBase[w];
        dirty[w * K + A_PREVSIZE]  = 700000 + w;
        dirty[w * K + A_MINDEX]    = 100000 + w;
        dirty[w * K + A_SIZE]      = size[w];
        dirty[w * K + A_FILTER]    = 200000 + w;
        dirty[w * K + A_MENV]      = 300000 + w;
        dirty[w * K + A_PREVCMP]   = 400000 + w;
        dirty[w * K + A_ISNEW]     = 1;
    }

    std::vector<uint32_t> srcPool(total), dstPool(total, 0xffffffffu);
    for (uint32_t v = 0; v < total; ++v) srcPool[v] = 9000000 + v;

    // reference
    std::vector<uint32_t> rAgg = agg, rDst = dstPool;
    for (uint32_t w = 0; w < numDirty; ++w)
    {
        for (uint32_t i = 0; i < size[w]; ++i)
            rDst[boundBase[w] + i] = srcPool[startIndex[w] + i];

        const uint32_t idx = dirtyIdx[w];
        if (agg[idx * K + A_ISNEW] != 0)
        {
            for (uint32_t j = 0; j < K; ++j) rAgg[idx * K + j] = dirty[w * K + j];
        }
        else
        {
            rAgg[idx * K + A_PREVSIZE]  = agg[idx * K + A_SIZE];
            rAgg[idx * K + A_BOUNDBASE] = dirty[w * K + A_BOUNDBASE];
            rAgg[idx * K + A_MINDEX]    = dirty[w * K + A_MINDEX];
            rAgg[idx * K + A_SIZE]      = dirty[w * K + A_SIZE];
            rAgg[idx * K + A_FILTER]    = dirty[w * K + A_FILTER];
        }
    }

    PxgSlangComputeDevice* dev = testDevice();
    PxgBufferHandle bAgg = createBufferFrom(dev, agg);
    PxgBufferHandle bDirty = createBufferFrom(dev, dirty);
    PxgBufferHandle bIdx = createBufferFrom(dev, dirtyIdx);
    PxgBufferHandle bSrc = createBufferFrom(dev, srcPool);
    PxgBufferHandle bStart = createBufferFrom(dev, startIndex);
    PxgBufferHandle bDst = createBufferFrom(dev, dstPool);

    DirtyParams p = { numDirty };
    const std::vector<uint32_t> strides(6, 4);
    Kernel k("broadphaseAggregateDirty.spv", "updateDirtyAggregate", strides, sizeof(DirtyParams));
    k.dispatch(numDirty, { bAgg, bDirty, bIdx, bSrc, bStart, bDst }, p);
    dev->flush();

    const auto gAgg = downloadBuffer<uint32_t>(dev, bAgg, nbAgg * K);
    const auto gDst = downloadBuffer<uint32_t>(dev, bDst, total);
    size_t aM = 0, dM = 0;
    for (uint32_t i = 0; i < nbAgg * K; ++i) if (gAgg[i] != rAgg[i]) ++aM;
    for (uint32_t i = 0; i < total; ++i) if (gDst[i] != rDst[i]) ++dM;
    PXG_CHECK_MSG(aM == 0, "%zu aggregate-field mismatches", aM);
    PXG_CHECK_MSG(dM == 0, "%zu bound-scatter mismatches (total=%u)", dM, total);
    for (PxgBufferHandle h : { bAgg, bDirty, bIdx, bSrc, bStart, bDst }) dev->destroyBuffer(h);
}

PXG_TEST(broadphaseAggregateDirty_clear)
{
    std::mt19937 rng(25020);
    const uint32_t nbAgg = 1000, numDirty = 400;
    std::vector<uint32_t> dirtyIdx = distinctIndices(rng, nbAgg, numDirty);

    std::vector<uint32_t> agg(nbAgg * K);
    for (uint32_t i = 0; i < nbAgg; ++i)
    {
        agg[i * K + A_BOUNDBASE] = 800000 + i;
        agg[i * K + A_PREVSIZE]  = 810000 + i;
        agg[i * K + A_MINDEX]    = 820000 + i;
        agg[i * K + A_SIZE]      = 1 + rng() % 50;
        agg[i * K + A_FILTER]    = 830000 + i;
        agg[i * K + A_MENV]      = 840000 + i;
        agg[i * K + A_PREVCMP]   = 850000 + i;
        agg[i * K + A_ISNEW]     = rng() % 2;
    }

    std::vector<uint32_t> rAgg = agg;
    for (uint32_t w = 0; w < numDirty; ++w)
    {
        const uint32_t idx = dirtyIdx[w];
        rAgg[idx * K + A_ISNEW] = 0;
        rAgg[idx * K + A_PREVSIZE] = agg[idx * K + A_SIZE];
    }

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<uint32_t> dummy(16, 0);
    PxgBufferHandle bAgg = createBufferFrom(dev, agg);
    PxgBufferHandle bIdx = createBufferFrom(dev, dirtyIdx);
    PxgBufferHandle bDummy = createBufferFrom(dev, dummy);

    DirtyParams p = { numDirty };
    const std::vector<uint32_t> strides(6, 4);
    Kernel k("broadphaseAggregateDirty.spv", "clearDirtyAggregates", strides, sizeof(DirtyParams));
    k.dispatch(numDirty, { bAgg, bDummy, bIdx, bDummy, bDummy, bDummy }, p);
    dev->flush();

    const auto gAgg = downloadBuffer<uint32_t>(dev, bAgg, nbAgg * K);
    size_t aM = 0;
    for (uint32_t i = 0; i < nbAgg * K; ++i) if (gAgg[i] != rAgg[i]) ++aM;
    PXG_CHECK_MSG(aM == 0, "%zu aggregate-field mismatches after clear", aM);
    for (PxgBufferHandle h : { bAgg, bIdx, bDummy }) dev->destroyBuffer(h);
}

} // namespace
