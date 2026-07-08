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

// Verification of broadphaseAggregateBoxPruning.slang (the aggregate collision
// SAP sweep) against a brute-force pair reference.  Two lists of boxes are built
// with a valid sorted X-projection SAP structure (distinct even mins / odd
// maxes, so every intersecting pair is enumerated exactly once across the two
// projection directions); the emitted pair set is compared order-independently
// against { (min,max) : new bounds intersect, old bounds do not, groups differ }.

#include "pxgTestFramework.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <set>
#include <vector>

namespace
{
using namespace pxgtest;

struct BoxPruneParams
{
    uint32_t nbObj0, nbObj1, maxPairs;
};

struct SapList
{
    std::vector<uint32_t> sortedProj, sortedHandles, startMask, boundIndices;
};

// build the sorted X-projection SAP structure for the given bound handles
SapList buildList(const std::vector<uint32_t>& handles, const std::vector<uint32_t>& bounds)
{
    const uint32_t n = uint32_t(handles.size());
    struct Entry { uint32_t value, objId, isStart; };
    std::vector<Entry> entries;
    for (uint32_t objId = 0; objId < n; ++objId)
    {
        const uint32_t h = handles[objId];
        entries.push_back({ bounds[h * 6 + 0], objId, 1 }); // min = start
        entries.push_back({ bounds[h * 6 + 3], objId, 0 }); // max = end
    }
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) { return a.value < b.value; });

    SapList s;
    const uint32_t nProj = 2 * n;
    s.sortedProj.resize(nProj);
    s.sortedHandles.resize(nProj);
    s.startMask.assign(nProj / 32 + 2, 0); // zero-padded so out-of-range reads see no start bits
    s.boundIndices = handles;
    for (uint32_t k = 0; k < nProj; ++k)
    {
        s.sortedProj[k] = entries[k].value;
        s.sortedHandles[k] = (entries[k].objId << 3) | entries[k].isStart; // createHandle(objId,isStart,false)
        if (entries[k].isStart) s.startMask[k / 32] |= (1u << (k & 31));
    }
    return s;
}

static bool intersects3D(const std::vector<uint32_t>& B, uint32_t a, uint32_t b)
{
    const uint32_t ab = a * 6, bb = b * 6;
    return !(B[bb + 0] > B[ab + 3] || B[ab + 0] > B[bb + 3] ||
             B[bb + 1] > B[ab + 4] || B[ab + 1] > B[bb + 4] ||
             B[bb + 2] > B[ab + 5] || B[ab + 2] > B[bb + 5]);
}

PXG_TEST(broadphaseAggregateBoxPruning_bipartite)
{
    std::mt19937 rng(29010);
    const uint32_t nbObj0 = 40, nbObj1 = 48, nB = nbObj0 + nbObj1;

    // build bounds with distinct even X-mins and odd X-maxes across ALL boxes
    std::vector<uint32_t> newB(nB * 6), oldB(nB * 6), groups(nB);
    std::set<uint32_t> used;
    for (uint32_t h = 0; h < nB; ++h)
    {
        // tight X domain + wide widths so X-overlaps (SAP candidates) are common
        uint32_t mnx;
        do { mnx = 2u * (rng() % 3000u); } while (used.count(mnx));
        used.insert(mnx);
        uint32_t mxx;
        do { mxx = mnx + 1u + 2u * (rng() % 1500u); } while (used.count(mxx));
        used.insert(mxx);
        newB[h * 6 + 0] = mnx; newB[h * 6 + 3] = mxx;
        newB[h * 6 + 1] = rng() % 4000; newB[h * 6 + 4] = newB[h * 6 + 1] + 1 + rng() % 3000;
        newB[h * 6 + 2] = rng() % 4000; newB[h * 6 + 5] = newB[h * 6 + 2] + 1 + rng() % 3000;
        // old bounds: independent random (some intersect the partner, some not)
        for (uint32_t c = 0; c < 3; ++c)
        {
            oldB[h * 6 + c] = rng() % 6000;
            oldB[h * 6 + 3 + c] = oldB[h * 6 + c] + 1 + rng() % 2500;
        }
        groups[h] = rng() % 4;
    }

    std::vector<uint32_t> h0(nbObj0), h1(nbObj1);
    std::iota(h0.begin(), h0.end(), 0u);
    std::iota(h1.begin(), h1.end(), nbObj0);
    SapList L0 = buildList(h0, newB), L1 = buildList(h1, newB);

    // brute-force reference
    std::set<std::pair<uint32_t, uint32_t>> ref;
    for (uint32_t a : h0)
        for (uint32_t b : h1)
            if (groups[a] != groups[b] && intersects3D(newB, a, b) && !intersects3D(oldB, a, b))
                ref.insert({ std::min(a, b), std::max(a, b) });

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t maxPairs = 10000;
    std::vector<uint32_t> pairs(maxPairs * 2, 0xffffffffu), shared = { 0u };
    PxgBufferHandle bNew = createBufferFrom(dev, newB);
    PxgBufferHandle bOld = createBufferFrom(dev, oldB);
    PxgBufferHandle bGrp = createBufferFrom(dev, groups);
    PxgBufferHandle bSP0 = createBufferFrom(dev, L0.sortedProj);
    PxgBufferHandle bSH0 = createBufferFrom(dev, L0.sortedHandles);
    PxgBufferHandle bSM0 = createBufferFrom(dev, L0.startMask);
    PxgBufferHandle bBI0 = createBufferFrom(dev, L0.boundIndices);
    PxgBufferHandle bSP1 = createBufferFrom(dev, L1.sortedProj);
    PxgBufferHandle bSH1 = createBufferFrom(dev, L1.sortedHandles);
    PxgBufferHandle bSM1 = createBufferFrom(dev, L1.startMask);
    PxgBufferHandle bBI1 = createBufferFrom(dev, L1.boundIndices);
    PxgBufferHandle bP = createBufferFrom(dev, pairs);
    PxgBufferHandle bS = createBufferFrom(dev, shared);

    BoxPruneParams p = { nbObj0, nbObj1, maxPairs };
    const std::vector<uint32_t> strides(13, 4);
    Kernel k("broadphaseAggregateBoxPruning.spv", "aggregateBipartiteBoxPruning", strides, sizeof(BoxPruneParams));
    k.dispatch(1, { bNew, bOld, bGrp, bSP0, bSH0, bSM0, bBI0, bSP1, bSH1, bSM1, bBI1, bP, bS }, p);
    dev->flush();

    const uint32_t n = downloadBuffer<uint32_t>(dev, bS, 1)[0];
    const auto gp = downloadBuffer<uint32_t>(dev, bP, std::min(n, maxPairs) * 2);
    std::set<std::pair<uint32_t, uint32_t>> got;
    for (uint32_t i = 0; i < std::min(n, maxPairs); ++i)
        got.insert({ gp[i * 2 + 0], gp[i * 2 + 1] });

    PXG_CHECK_MSG(n == ref.size(), "emitted %u pairs, reference has %zu", n, ref.size());
    PXG_CHECK_MSG(got == ref, "emitted pair set differs from reference (got %zu unique, ref %zu)", got.size(), ref.size());
    PXG_CHECK_MSG(ref.size() >= 8, "reference too small (%zu) — weak test", ref.size());

    for (PxgBufferHandle h : { bNew, bOld, bGrp, bSP0, bSH0, bSM0, bBI0, bSP1, bSH1, bSM1, bBI1, bP, bS })
        dev->destroyBuffer(h);
}

} // namespace
