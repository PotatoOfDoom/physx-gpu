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

// Verification of broadphaseAggregateBounds.slang against C++ transcriptions of
// the two aggregate bounds/bitmap refresh kernels from aggregate.cu:
// markAggregateBoundsUpdatedBitmap (atomic OR the aggregate's own handle bit +
// atomic AND ~bit for each member shape) and updateAggregateBounds (union of
// contact-distance-inflated member bounds written back at the aggregate slot).
// The per-aggregate pointer lists are flattened into a shared pool indexed by a
// base offset; results are compared bit-exactly.

#include "pxgTestFramework.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

const uint32_t INVALID = 0xffffffffu;

struct AggBoundsParams
{
    uint32_t numAggregates;
};

// PxBounds3::setEmpty() sentinel: PX_MAX_BOUNDS_EXTENTS = FLT_MAX * 0.25f.
const float MAX_BOUNDS_EXTENTS = std::numeric_limits<float>::max() * 0.25f;

// Build a synthetic aggregate set: nbShapes member-shape bound slots [0,nbShapes)
// and nbAgg aggregate slots [nbShapes, nbShapes+nbAgg) (disjoint).  Each aggregate
// gets a distinct mIndex in its own slot, a member count, and a contiguous run in
// the shared bound-index pool.  ~5% are made invalid (half size 0, half mIndex
// INVALID) to exercise the isValid() guard.
struct AggSet
{
    uint32_t nbShapes, nbAgg, nbSlots;
    std::vector<uint32_t> mIndex, size, boundBase, boundIndices;
};

AggSet genAggSet(std::mt19937& rng, uint32_t nbAgg, uint32_t nbShapes)
{
    AggSet s;
    s.nbShapes = nbShapes;
    s.nbAgg = nbAgg;
    s.nbSlots = nbShapes + nbAgg;
    s.mIndex.resize(nbAgg);
    s.size.resize(nbAgg);
    s.boundBase.resize(nbAgg);
    uint32_t base = 0;
    for (uint32_t a = 0; a < nbAgg; ++a)
    {
        uint32_t sz = 1 + rng() % 40; // up to 40 members -> exercises >32 strided loop + reduction
        uint32_t mi = nbShapes + a;   // distinct, disjoint from member shape slots
        const uint32_t roll = rng() % 100;
        if (roll < 3) sz = 0;             // invalid via size 0
        else if (roll < 5) mi = INVALID; // invalid via mIndex
        s.size[a] = sz;
        s.mIndex[a] = mi;
        s.boundBase[a] = base;
        const uint32_t emit = (sz == 0) ? (1 + rng() % 8) : sz; // still reserve a run for size-0 aggs
        for (uint32_t i = 0; i < emit; ++i)
            s.boundIndices.push_back(rng() % nbShapes);
        base += emit;
    }
    return s;
}

bool valid(uint32_t size, uint32_t mIndex) { return size > 0 && mIndex != INVALID; }

PXG_TEST(broadphaseAggregateBounds_bitmap)
{
    std::mt19937 rng(24010);
    const uint32_t nbAgg = 500, nbShapes = 20000;
    AggSet s = genAggSet(rng, nbAgg, nbShapes);
    const uint32_t words = (s.nbSlots + 31) / 32;

    std::vector<uint32_t> changed(words);
    for (auto& w : changed) w = rng();

    // reference (order-independent: OR-set and AND-clear sets are disjoint)
    std::vector<uint32_t> rChanged = changed;
    for (uint32_t a = 0; a < nbAgg; ++a)
    {
        if (!valid(s.size[a], s.mIndex[a])) continue;
        const uint32_t mi = s.mIndex[a];
        rChanged[mi / 32] |= (1u << (mi % 32));
        for (uint32_t i = 0; i < s.size[a]; ++i)
        {
            const uint32_t idx = s.boundIndices[s.boundBase[a] + i];
            rChanged[idx / 32] &= ~(1u << (idx % 32));
        }
    }

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<float> fdummy(16, 0.f);
    PxgBufferHandle bMI = createBufferFrom(dev, s.mIndex);
    PxgBufferHandle bSz = createBufferFrom(dev, s.size);
    PxgBufferHandle bBase = createBufferFrom(dev, s.boundBase);
    PxgBufferHandle bIdx = createBufferFrom(dev, s.boundIndices);
    PxgBufferHandle bChg = createBufferFrom(dev, changed);
    PxgBufferHandle bFDummy = createBufferFrom(dev, fdummy);

    AggBoundsParams p = { nbAgg };
    const std::vector<uint32_t> strides(7, 4);
    Kernel k("broadphaseAggregateBounds.spv", "markAggregateBoundsUpdatedBitmap", strides, sizeof(AggBoundsParams));
    k.dispatch(nbAgg, { bMI, bSz, bBase, bIdx, bChg, bFDummy, bFDummy }, p);
    dev->flush();

    const auto gChanged = downloadBuffer<uint32_t>(dev, bChg, words);
    size_t mism = 0;
    for (uint32_t w = 0; w < words; ++w) if (gChanged[w] != rChanged[w]) ++mism;
    PXG_CHECK_MSG(mism == 0, "%zu bitmap-word mismatches (words=%u)", mism, words);
    for (PxgBufferHandle h : { bMI, bSz, bBase, bIdx, bChg, bFDummy }) dev->destroyBuffer(h);
}

PXG_TEST(broadphaseAggregateBounds_bounds)
{
    std::mt19937 rng(24020);
    const uint32_t nbAgg = 500, nbShapes = 20000;
    AggSet s = genAggSet(rng, nbAgg, nbShapes);

    // flat PxBounds3 pool: shape slots [0,nbShapes) carry real bounds; aggregate
    // slots [nbShapes,nbSlots) start as a sentinel and are overwritten by valid aggs.
    std::vector<float> bounds(s.nbSlots * 6, 0.f);
    std::vector<float> contactDist(s.nbSlots, 0.f);
    std::uniform_real_distribution<float> pos(-500.f, 500.f), ext(0.01f, 50.f), cd(0.f, 2.f);
    for (uint32_t i = 0; i < nbShapes; ++i)
    {
        float mnx = pos(rng), mny = pos(rng), mnz = pos(rng);
        bounds[i * 6 + 0] = mnx; bounds[i * 6 + 1] = mny; bounds[i * 6 + 2] = mnz;
        bounds[i * 6 + 3] = mnx + ext(rng); bounds[i * 6 + 4] = mny + ext(rng); bounds[i * 6 + 5] = mnz + ext(rng);
        contactDist[i] = cd(rng);
    }
    for (uint32_t i = nbShapes; i < s.nbSlots; ++i) // sentinel garbage in the aggregate slots
    {
        for (uint32_t c = 0; c < 6; ++c) bounds[i * 6 + c] = 12345.678f + float(i) + float(c);
    }

    // reference
    std::vector<float> rBounds = bounds;
    for (uint32_t a = 0; a < nbAgg; ++a)
    {
        if (!valid(s.size[a], s.mIndex[a])) continue;
        float mn[3] = { MAX_BOUNDS_EXTENTS, MAX_BOUNDS_EXTENTS, MAX_BOUNDS_EXTENTS };
        float mx[3] = { -MAX_BOUNDS_EXTENTS, -MAX_BOUNDS_EXTENTS, -MAX_BOUNDS_EXTENTS };
        for (uint32_t i = 0; i < s.size[a]; ++i)
        {
            const uint32_t idx = s.boundIndices[s.boundBase[a] + i];
            const float infl = contactDist[idx];
            for (uint32_t c = 0; c < 3; ++c)
            {
                mn[c] = std::min(mn[c], bounds[idx * 6 + c] - infl);
                mx[c] = std::max(mx[c], bounds[idx * 6 + 3 + c] + infl);
            }
        }
        const uint32_t o = s.mIndex[a] * 6;
        rBounds[o + 0] = mn[0]; rBounds[o + 1] = mn[1]; rBounds[o + 2] = mn[2];
        rBounds[o + 3] = mx[0]; rBounds[o + 4] = mx[1]; rBounds[o + 5] = mx[2];
    }

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<uint32_t> udummy(16, 0);
    PxgBufferHandle bMI = createBufferFrom(dev, s.mIndex);
    PxgBufferHandle bSz = createBufferFrom(dev, s.size);
    PxgBufferHandle bBase = createBufferFrom(dev, s.boundBase);
    PxgBufferHandle bIdx = createBufferFrom(dev, s.boundIndices);
    PxgBufferHandle bUDummy = createBufferFrom(dev, udummy);
    PxgBufferHandle bBounds = createBufferFrom(dev, bounds);
    PxgBufferHandle bCD = createBufferFrom(dev, contactDist);

    AggBoundsParams p = { nbAgg };
    const std::vector<uint32_t> strides(7, 4);
    Kernel k("broadphaseAggregateBounds.spv", "updateAggregateBounds", strides, sizeof(AggBoundsParams));
    k.dispatch(nbAgg, { bMI, bSz, bBase, bIdx, bUDummy, bBounds, bCD }, p);
    dev->flush();

    const auto gBounds = downloadBuffer<float>(dev, bBounds, s.nbSlots * 6);
    size_t mism = 0; float maxDiff = 0.f;
    for (uint32_t i = 0; i < s.nbSlots * 6; ++i)
    {
        if (std::memcmp(&gBounds[i], &rBounds[i], sizeof(float)) != 0)
        {
            ++mism;
            maxDiff = std::max(maxDiff, std::fabs(gBounds[i] - rBounds[i]));
        }
    }
    PXG_CHECK_MSG(mism == 0, "%zu aggregate-bounds mismatches (maxAbsDiff=%g)", mism, maxDiff);
    for (PxgBufferHandle h : { bMI, bSz, bBase, bIdx, bUDummy, bBounds, bCD }) dev->destroyBuffer(h);
}

} // namespace
