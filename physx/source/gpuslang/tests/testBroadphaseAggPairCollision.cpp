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

// Verification of broadphaseAggPairCollision.slang (doAggPairCollisions) against
// a C++ reference over a mix of aggregate-aggregate, aggregate-actor and
// actor-actor pairs (plus dead / removed pairs).  Aggregates carry a precomputed
// sorted X-projection SAP structure (built into shared pools for both frames);
// single actors are synthesized 1-box structures.  The emitted found/lost pair
// sets, the total counts, the per-pair remove bitmap and the cleared new flags
// are all compared exactly.

#include "pxgTestFramework.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <set>
#include <vector>

namespace
{
using namespace pxgtest;

const uint32_t INVALID = 0xffffffffu;

struct AggPairParams
{
    uint32_t nbAggregatePairs, maxFoundLostPairs;
};

static bool intersects3D(const std::vector<uint32_t>& B, uint32_t a, uint32_t b)
{
    const uint32_t ab = a * 6, bb = b * 6;
    return !(B[bb + 0] > B[ab + 3] || B[ab + 0] > B[bb + 3] ||
             B[bb + 1] > B[ab + 4] || B[ab + 1] > B[bb + 4] ||
             B[bb + 2] > B[ab + 5] || B[ab + 2] > B[bb + 5]);
}

struct Pools
{
    std::vector<uint32_t> proj, handle, mask, bi;
    // append one list (members sorted by the given bounds' X); return {projBase, maskBase, biBase, n}
    void append(const std::vector<uint32_t>& members, const std::vector<uint32_t>& B,
                uint32_t& projBase, uint32_t& maskBase, uint32_t& biBase, uint32_t& n)
    {
        n = uint32_t(members.size());
        struct E { uint32_t v, objId, isStart; };
        std::vector<E> e;
        for (uint32_t o = 0; o < n; ++o) { e.push_back({ B[members[o] * 6 + 0], o, 1 }); e.push_back({ B[members[o] * 6 + 3], o, 0 }); }
        std::sort(e.begin(), e.end(), [](const E& x, const E& y) { return x.v < y.v; });
        projBase = uint32_t(proj.size()); maskBase = uint32_t(mask.size()); biBase = uint32_t(bi.size());
        const uint32_t nProj = 2 * n, words = nProj / 32 + 2;
        std::vector<uint32_t> mk(words, 0);
        for (uint32_t k = 0; k < nProj; ++k)
        {
            proj.push_back(e[k].v);
            handle.push_back((e[k].objId << 3) | e[k].isStart);
            if (e[k].isStart) mk[k / 32] |= (1u << (k & 31));
        }
        for (uint32_t w = 0; w < words; ++w) mask.push_back(mk[w]);
        for (uint32_t o = 0; o < n; ++o) bi.push_back(members[o]);
    }
};

// bipartite reference: emit (min,max) iff groups differ, primary-frame intersects, other-frame does not
static void sweepRef(const std::vector<uint32_t>& m0, const std::vector<uint32_t>& m1,
                     const std::vector<uint32_t>& primary, const std::vector<uint32_t>& other,
                     const std::vector<uint32_t>& groups, std::set<std::pair<uint32_t, uint32_t>>& out)
{
    for (uint32_t a : m0)
        for (uint32_t b : m1)
            if (groups[a] != groups[b] && intersects3D(primary, a, b) && !intersects3D(other, a, b))
                out.insert({ std::min(a, b), std::max(a, b) });
}

PXG_TEST(broadphaseAggPairCollision_mixed)
{
    std::mt19937 rng(30010);
    const uint32_t nAgg = 6, nActor = 8;

    // handle layout: [0,nAgg) aggregate reps; then members; then actors
    std::vector<std::vector<uint32_t>> aggMembers(nAgg);
    uint32_t next = nAgg;
    for (uint32_t a = 0; a < nAgg; ++a)
    {
        const uint32_t cnt = 4 + rng() % 7;
        for (uint32_t m = 0; m < cnt; ++m) aggMembers[a].push_back(next++);
    }
    std::vector<uint32_t> actorHandle(nActor);
    for (uint32_t a = 0; a < nActor; ++a) actorHandle[a] = next++;
    const uint32_t nB = next;

    // bounds: reps get independent random bounds (drive shouldRemove); members and
    // actors get distinct even mins / odd maxes so each intersecting pair is unique
    std::vector<uint32_t> newB(nB * 6), oldB(nB * 6), groups(nB), vol(nB, INVALID);
    std::set<uint32_t> used;
    auto isMemberOrActor = [&](uint32_t h) { return h >= nAgg; };
    for (uint32_t h = 0; h < nB; ++h)
    {
        groups[h] = rng() % 4;
        if (isMemberOrActor(h))
        {
            uint32_t mnx; do { mnx = 2u * (rng() % 4000u); } while (used.count(mnx)); used.insert(mnx);
            uint32_t mxx; do { mxx = mnx + 1u + 2u * (rng() % 1500u); } while (used.count(mxx)); used.insert(mxx);
            newB[h * 6 + 0] = mnx; newB[h * 6 + 3] = mxx;
            newB[h * 6 + 1] = rng() % 4000; newB[h * 6 + 4] = newB[h * 6 + 1] + 1 + rng() % 3000;
            newB[h * 6 + 2] = rng() % 4000; newB[h * 6 + 5] = newB[h * 6 + 2] + 1 + rng() % 3000;
            for (uint32_t c = 0; c < 3; ++c) { oldB[h * 6 + c] = rng() % 6000; oldB[h * 6 + 3 + c] = oldB[h * 6 + c] + 1 + rng() % 2500; }
        }
        else // aggregate rep bound (independent — drives shouldRemove)
        {
            for (uint32_t c = 0; c < 3; ++c)
            {
                newB[h * 6 + c] = rng() % 6000; newB[h * 6 + 3 + c] = newB[h * 6 + c] + 1 + rng() % 3000;
                oldB[h * 6 + c] = rng() % 6000; oldB[h * 6 + 3 + c] = oldB[h * 6 + c] + 1 + rng() % 3000;
            }
        }
    }
    for (uint32_t a = 0; a < nAgg; ++a) vol[a] = (a << 1) | 1u; // aggregate rep

    // build pools + meta (curr from newB, prev from oldB)
    Pools pools;
    std::vector<uint32_t> meta(nAgg * 8);
    for (uint32_t a = 0; a < nAgg; ++a)
    {
        uint32_t pb, mb, bb, n;
        pools.append(aggMembers[a], newB, pb, mb, bb, n);
        meta[a * 8 + 0] = pb; meta[a * 8 + 1] = mb; meta[a * 8 + 2] = bb; meta[a * 8 + 3] = n;
        pools.append(aggMembers[a], oldB, pb, mb, bb, n);
        meta[a * 8 + 4] = pb; meta[a * 8 + 5] = mb; meta[a * 8 + 6] = bb; meta[a * 8 + 7] = n;
    }

    // members of a pair side (aggregate members, or the actor itself)
    auto sideMembers = [&](uint32_t handle) -> std::vector<uint32_t> {
        if (handle < nAgg) return aggMembers[handle];   // rep handle a -> agg a's members (handle==a here)
        return { handle };
    };

    // pairs: mix of agg-agg / agg-actor / actor-actor, some dead, some isNew, plus removed
    struct Pair { uint32_t h0, h1, isNew, isDead; };
    std::vector<Pair> pairs;
    auto repOf = [&](uint32_t aggIdx) { return aggIdx; };            // rep handle == aggregate index
    pairs.push_back({ repOf(0), repOf(1), 0, 0 });   // agg-agg, not new
    pairs.push_back({ repOf(2), repOf(3), 1, 0 });   // agg-agg, new (no lost)
    pairs.push_back({ repOf(4), actorHandle[0], 0, 0 }); // agg-actor
    pairs.push_back({ actorHandle[1], actorHandle[2], 0, 0 }); // actor-actor
    pairs.push_back({ repOf(5), actorHandle[3], 1, 0 }); // agg-actor, new
    pairs.push_back({ actorHandle[4], actorHandle[5], 0, 1 }); // dead -> skipped
    pairs.push_back({ repOf(0), actorHandle[6], 0, 0 }); // agg-actor
    pairs.push_back({ repOf(1), repOf(2), 0, 0 });   // agg-agg
    const uint32_t nbPairs = uint32_t(pairs.size());

    // removed bitmap: mark actorHandle[7] removed, and put it in one pair to exercise the skip
    pairs.push_back({ actorHandle[7], repOf(3), 0, 0 });
    const uint32_t nbPairsAll = uint32_t(pairs.size());
    std::vector<uint32_t> removedMap((nB + 31) / 32, 0);
    removedMap[actorHandle[7] / 32] |= (1u << (actorHandle[7] & 31));

    std::vector<uint32_t> aggPairs(nbPairsAll * 4), aggBitmapInit(nbPairsAll, 9);
    for (uint32_t i = 0; i < nbPairsAll; ++i)
    {
        aggPairs[i * 4 + 0] = pairs[i].h0; aggPairs[i * 4 + 1] = pairs[i].h1;
        aggPairs[i * 4 + 2] = pairs[i].isNew; aggPairs[i * 4 + 3] = pairs[i].isDead;
    }

    // reference
    std::set<std::pair<uint32_t, uint32_t>> foundRef, lostRef;
    std::vector<uint32_t> refBitmap(nbPairsAll), refIsNew(nbPairsAll);
    for (uint32_t i = 0; i < nbPairsAll; ++i)
    {
        const Pair& p = pairs[i];
        refIsNew[i] = p.isNew;
        if (p.isDead) { refBitmap[i] = 0; continue; }
        const bool removed = (removedMap[p.h0 / 32] >> (p.h0 & 31) & 1) || (removedMap[p.h1 / 32] >> (p.h1 & 31) & 1);
        if (removed) { refBitmap[i] = 1; continue; }
        refBitmap[i] = intersects3D(newB, p.h0, p.h1) ? 0 : 1;
        std::vector<uint32_t> m0 = sideMembers(p.h0), m1 = sideMembers(p.h1);
        sweepRef(m0, m1, newB, oldB, groups, foundRef);           // found
        if (!p.isNew) sweepRef(m0, m1, oldB, newB, groups, lostRef); // lost
        refIsNew[i] = 0; // cleared
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t maxFL = 20000;
    std::vector<uint32_t> foundBuf(maxFL * 2, 0xffffffffu), lostBuf(maxFL * 2, 0xffffffffu), shared = { 0u, 0u };
    PxgBufferHandle bNew = createBufferFrom(dev, newB), bOld = createBufferFrom(dev, oldB), bGrp = createBufferFrom(dev, groups);
    PxgBufferHandle bVol = createBufferFrom(dev, vol), bRem = createBufferFrom(dev, removedMap);
    PxgBufferHandle bAP = createBufferFrom(dev, aggPairs), bBM = createBufferFrom(dev, aggBitmapInit), bMeta = createBufferFrom(dev, meta);
    PxgBufferHandle bPP = createBufferFrom(dev, pools.proj), bHP = createBufferFrom(dev, pools.handle);
    PxgBufferHandle bMP = createBufferFrom(dev, pools.mask), bBP = createBufferFrom(dev, pools.bi);
    PxgBufferHandle bF = createBufferFrom(dev, foundBuf), bL = createBufferFrom(dev, lostBuf), bS = createBufferFrom(dev, shared);

    AggPairParams pr = { nbPairsAll, maxFL };
    const std::vector<uint32_t> strides(15, 4);
    Kernel k("broadphaseAggPairCollision.spv", "doAggPairCollisions", strides, sizeof(AggPairParams));
    k.dispatch(nbPairsAll, { bNew, bOld, bGrp, bVol, bRem, bAP, bBM, bMeta, bPP, bHP, bMP, bBP, bF, bL, bS }, pr);
    dev->flush();

    const auto sh = downloadBuffer<uint32_t>(dev, bS, 2);
    const auto gF = downloadBuffer<uint32_t>(dev, bF, std::min(sh[0], maxFL) * 2);
    const auto gL = downloadBuffer<uint32_t>(dev, bL, std::min(sh[1], maxFL) * 2);
    const auto gBM = downloadBuffer<uint32_t>(dev, bBM, nbPairsAll);
    const auto gAP = downloadBuffer<uint32_t>(dev, bAP, nbPairsAll * 4);

    std::set<std::pair<uint32_t, uint32_t>> gotF, gotL;
    for (uint32_t i = 0; i < std::min(sh[0], maxFL); ++i) gotF.insert({ gF[i * 2], gF[i * 2 + 1] });
    for (uint32_t i = 0; i < std::min(sh[1], maxFL); ++i) gotL.insert({ gL[i * 2], gL[i * 2 + 1] });

    PXG_CHECK_MSG(sh[0] == foundRef.size(), "found count %u exp %zu", sh[0], foundRef.size());
    PXG_CHECK_MSG(sh[1] == lostRef.size(), "lost count %u exp %zu", sh[1], lostRef.size());
    PXG_CHECK_MSG(gotF == foundRef, "found pair set differs (got %zu, ref %zu)", gotF.size(), foundRef.size());
    PXG_CHECK_MSG(gotL == lostRef, "lost pair set differs (got %zu, ref %zu)", gotL.size(), lostRef.size());
    PXG_CHECK_MSG(foundRef.size() >= 8, "found reference too small (%zu)", foundRef.size());
    size_t bm = 0, nw = 0;
    for (uint32_t i = 0; i < nbPairsAll; ++i) { if (gBM[i] != refBitmap[i]) ++bm; if (gAP[i * 4 + 2] != refIsNew[i]) ++nw; }
    PXG_CHECK_MSG(bm == 0, "%zu aggBitmap mismatches", bm);
    PXG_CHECK_MSG(nw == 0, "%zu isNew mismatches", nw);

    for (PxgBufferHandle h : { bNew, bOld, bGrp, bVol, bRem, bAP, bBM, bMeta, bPP, bHP, bMP, bBP, bF, bL, bS })
        dev->destroyBuffer(h);
}

} // namespace
