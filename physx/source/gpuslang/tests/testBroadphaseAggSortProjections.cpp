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

// Verification of broadphaseAggSortProjections.slang against a C++ reference that
// builds the same sorted X-projection SAP structure (sorted projections, sorted
// start/end handles, sapBox1D positions, start-projection bitmask, local bound
// indices) for each valid aggregate.  Projection values are globally distinct so
// the sorted order is unambiguous.

#include "pxgTestFramework.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <set>
#include <vector>

namespace
{
using namespace pxgtest;

const uint32_t SENT = 0xccccccccu;

struct SortParams { uint32_t nbAggregates; };

PXG_TEST(broadphaseAggSortProjections_build)
{
    std::mt19937 rng(32010);
    const uint32_t nAgg = 10;

    struct Agg { std::vector<uint32_t> members; uint32_t valid; };
    std::vector<Agg> aggs(nAgg);
    uint32_t next = 0;
    for (uint32_t a = 0; a < nAgg; ++a)
    {
        const uint32_t cnt = 1 + rng() % 40;
        for (uint32_t m = 0; m < cnt; ++m) aggs[a].members.push_back(next++);
        aggs[a].valid = (a == 3 || a == 7) ? 0u : 1u; // a=3,7 invalid -> skipped
    }
    const uint32_t nB = next;

    // bounds: globally distinct even X-mins / odd X-maxes
    std::vector<uint32_t> newB(nB * 6);
    std::set<uint32_t> used;
    for (uint32_t h = 0; h < nB; ++h)
    {
        uint32_t mnx; do { mnx = 2u * (rng() % 200000u); } while (used.count(mnx)); used.insert(mnx);
        uint32_t mxx; do { mxx = mnx + 1u + 2u * (rng() % 5000u); } while (used.count(mxx)); used.insert(mxx);
        newB[h * 6 + 0] = mnx; newB[h * 6 + 3] = mxx;
        for (uint32_t c = 1; c < 3; ++c) { newB[h * 6 + c] = rng(); newB[h * 6 + 3 + c] = rng(); }
    }

    // pools + meta
    std::vector<uint32_t> ubi, meta(nAgg * 8);
    uint32_t projTot = 0, sapTot = 0, maskTot = 0, biTot = 0;
    for (uint32_t a = 0; a < nAgg; ++a)
    {
        const uint32_t n = uint32_t(aggs[a].members.size()), nProj = n * 2, words = nProj / 32 + 2;
        uint32_t* M = &meta[a * 8];
        M[0] = aggs[a].valid; M[1] = n; M[2] = uint32_t(ubi.size());
        M[3] = projTot; M[4] = sapTot; M[5] = maskTot; M[6] = biTot;
        for (uint32_t m : aggs[a].members) ubi.push_back(m);
        projTot += nProj; sapTot += n * 2; maskTot += words; biTot += n;
    }

    std::vector<uint32_t> projOut(projTot, SENT), handleOut(projTot, SENT), sapOut(sapTot, SENT),
                          maskOut(maskTot, SENT), biOut(biTot, SENT);

    // reference
    std::vector<uint32_t> rProj = projOut, rHandle = handleOut, rSap = sapOut, rMask = maskOut, rBi = biOut;
    for (uint32_t a = 0; a < nAgg; ++a)
    {
        if (!aggs[a].valid) continue;
        const auto& mem = aggs[a].members;
        const uint32_t n = uint32_t(mem.size()), nProj = n * 2;
        const uint32_t* M = &meta[a * 8];
        struct E { uint32_t v, h, orig; };
        std::vector<E> e(nProj);
        for (uint32_t i = 0; i < nProj; ++i)
        {
            const uint32_t id = i / 2, handle = mem[id];
            e[i].v = (i & 1) ? newB[handle * 6 + 3] : newB[handle * 6 + 0];
            e[i].h = (id << 3) | ((i & 1) == 0 ? 1u : 0u);
            e[i].orig = i;
            if ((i & 1) == 0) rBi[M[6] + id] = handle;
        }
        std::stable_sort(e.begin(), e.end(), [](const E& x, const E& y) { return x.v < y.v; });
        // clear the mask words we will write
        for (uint32_t w = 0; w < nProj / 32 + 2; ++w) rMask[M[5] + w] = 0; // padding words end up 0
        // note: shader only writes ceil(nProj/32) words; leave the rest as SENT to match
        for (uint32_t w = (nProj + 31) / 32; w < nProj / 32 + 2; ++w) rMask[M[5] + w] = SENT;
        for (uint32_t pos = 0; pos < nProj; ++pos)
        {
            rProj[M[3] + pos] = e[pos].v;
            rHandle[M[3] + pos] = e[pos].h;
            const uint32_t box = e[pos].h >> 3, isStart = e[pos].h & 1;
            rSap[M[4] + box * 2 + (isStart ? 0 : 1)] = pos;
            if (isStart) rMask[M[5] + pos / 32] |= (1u << (pos & 31));
        }
    }

    PxgSlangComputeDevice* dev = testDevice();
    PxgBufferHandle bNew = createBufferFrom(dev, newB), bUbi = createBufferFrom(dev, ubi), bMeta = createBufferFrom(dev, meta);
    PxgBufferHandle bProj = createBufferFrom(dev, projOut), bHand = createBufferFrom(dev, handleOut);
    PxgBufferHandle bSap = createBufferFrom(dev, sapOut), bMask = createBufferFrom(dev, maskOut), bBi = createBufferFrom(dev, biOut);

    SortParams p = { nAgg };
    const std::vector<uint32_t> strides(8, 4);
    Kernel k("broadphaseAggSortProjections.spv", "sortAndUpdateAggregateProjections", strides, sizeof(SortParams));
    k.dispatch(nAgg, { bNew, bUbi, bMeta, bProj, bHand, bSap, bMask, bBi }, p);
    dev->flush();

    const auto gProj = downloadBuffer<uint32_t>(dev, bProj, projTot);
    const auto gHand = downloadBuffer<uint32_t>(dev, bHand, projTot);
    const auto gSap = downloadBuffer<uint32_t>(dev, bSap, sapTot);
    const auto gMask = downloadBuffer<uint32_t>(dev, bMask, maskTot);
    const auto gBi = downloadBuffer<uint32_t>(dev, bBi, biTot);
    size_t pm = 0, hm = 0, sm = 0, mm = 0, bm = 0;
    for (uint32_t i = 0; i < projTot; ++i) { if (gProj[i] != rProj[i]) ++pm; if (gHand[i] != rHandle[i]) ++hm; }
    for (uint32_t i = 0; i < sapTot; ++i) if (gSap[i] != rSap[i]) ++sm;
    for (uint32_t i = 0; i < maskTot; ++i) if (gMask[i] != rMask[i]) ++mm;
    for (uint32_t i = 0; i < biTot; ++i) if (gBi[i] != rBi[i]) ++bm;
    PXG_CHECK_MSG(pm == 0, "%zu projection mismatches", pm);
    PXG_CHECK_MSG(hm == 0, "%zu handle mismatches", hm);
    PXG_CHECK_MSG(sm == 0, "%zu sapBox1D mismatches", sm);
    PXG_CHECK_MSG(mm == 0, "%zu startMask mismatches", mm);
    PXG_CHECK_MSG(bm == 0, "%zu boundIndices mismatches", bm);

    for (PxgBufferHandle h : { bNew, bUbi, bMeta, bProj, bHand, bSap, bMask, bBi }) dev->destroyBuffer(h);
}

} // namespace
