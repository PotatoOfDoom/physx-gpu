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

// Verification of broadphaseAggSelfCollision.slang (doSelfCollision) against a
// brute-force self-collision reference over several aggregates (valid / invalid
// / self-collision-off / new).  Each aggregate carries a sorted X-projection SAP
// structure (sapBox1D + start masks + sorted handles + bound indices) for both
// frames; the previous frame's comparison prefix is precomputed.  The emitted
// found/lost pair sets, their counts and the per-aggregate previous-comparison
// output are compared exactly.

#include "pxgTestFramework.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <set>
#include <vector>

namespace
{
using namespace pxgtest;

struct SelfParams
{
    uint32_t nbAggregates, maxFoundLostPairs;
};

// popcount helper (matches HLSL countbits)
static uint32_t popc(uint32_t x) { uint32_t c = 0; while (x) { c += x & 1u; x >>= 1; } return c; }

struct Built
{
    std::vector<uint32_t> sap, mask, handle, bi; // sap: 2/box, handle: 2n, mask: words, bi: n
    uint32_t n;
};

// build the sorted X-projection SAP structure for the given members using bounds B
static Built buildSelf(const std::vector<uint32_t>& members, const std::vector<uint32_t>& B)
{
    Built r; r.n = uint32_t(members.size());
    struct E { uint32_t v, obj, isStart; };
    std::vector<E> e;
    for (uint32_t o = 0; o < r.n; ++o) { e.push_back({ B[members[o] * 6 + 0], o, 1 }); e.push_back({ B[members[o] * 6 + 3], o, 0 }); }
    std::sort(e.begin(), e.end(), [](const E& x, const E& y) { return x.v < y.v; });
    const uint32_t nProj = 2 * r.n, words = nProj / 32 + 2;
    r.mask.assign(words, 0);
    r.handle.resize(nProj);
    r.sap.assign(r.n * 2, 0);
    r.bi = members;
    for (uint32_t k = 0; k < nProj; ++k)
    {
        r.handle[k] = (e[k].obj << 3) | e[k].isStart;
        if (e[k].isStart) { r.mask[k / 32] |= (1u << (k & 31)); r.sap[e[k].obj * 2 + 0] = k; }
        else r.sap[e[k].obj * 2 + 1] = k;
    }
    return r;
}

// per-box comparison count (mirrors the shader's count loop)
static uint32_t countBox(const Built& b, uint32_t box)
{
    const uint32_t startIdx = b.sap[box * 2 + 0], endIdx = b.sap[box * 2 + 1];
    const uint32_t startWord = startIdx / 32, endWord = endIdx / 32;
    uint32_t mask = ~((2u << (startIdx & 31)) - 1u), count = 0;
    for (uint32_t w = startWord; w < endWord; ++w) { count += popc(b.mask[w] & mask); mask = 0xffffffffu; }
    mask &= ((2u << (endIdx & 31)) - 1u);
    count += popc(b.mask[endWord] & mask);
    return count;
}

static bool intersects3D(const std::vector<uint32_t>& B, uint32_t a, uint32_t b)
{
    const uint32_t ab = a * 6, bb = b * 6;
    return !(B[bb + 0] > B[ab + 3] || B[ab + 0] > B[bb + 3] ||
             B[bb + 1] > B[ab + 4] || B[ab + 1] > B[bb + 4] ||
             B[bb + 2] > B[ab + 5] || B[ab + 2] > B[bb + 5]);
}

PXG_TEST(broadphaseAggSelfCollision_selfpairs)
{
    std::mt19937 rng(31010);
    const uint32_t nAgg = 8;

    struct Agg { std::vector<uint32_t> members; uint32_t valid, selfBit, isNew; };
    std::vector<Agg> aggs(nAgg);
    uint32_t next = 0;
    for (uint32_t a = 0; a < nAgg; ++a)
    {
        const uint32_t cnt = 3 + rng() % 10;
        for (uint32_t m = 0; m < cnt; ++m) aggs[a].members.push_back(next++);
        aggs[a].valid = (a == 2) ? 0u : 1u;       // aggregate 2 invalid -> skipped
        aggs[a].selfBit = (a == 5) ? 0u : 1u;     // aggregate 5 self-collision off -> skipped
        aggs[a].isNew = (a == 1 || a == 6) ? 1u : 0u; // new -> no lost pairs
    }
    const uint32_t nB = next;

    // bounds: distinct even mins / odd maxes for BOTH new and old (so each
    // self-collision pair is enumerated exactly once per frame)
    std::vector<uint32_t> newB(nB * 6), oldB(nB * 6), groups(nB);
    std::set<uint32_t> un, uo;
    for (uint32_t h = 0; h < nB; ++h)
    {
        groups[h] = rng() % 3;
        uint32_t nmn; do { nmn = 2u * (rng() % 1500u); } while (un.count(nmn)); un.insert(nmn);
        uint32_t nmx; do { nmx = nmn + 1u + 2u * (rng() % 1200u); } while (un.count(nmx)); un.insert(nmx);
        newB[h * 6 + 0] = nmn; newB[h * 6 + 3] = nmx;
        newB[h * 6 + 1] = rng() % 2500; newB[h * 6 + 4] = newB[h * 6 + 1] + 1 + rng() % 2500;
        newB[h * 6 + 2] = rng() % 2500; newB[h * 6 + 5] = newB[h * 6 + 2] + 1 + rng() % 2500;
        uint32_t omn; do { omn = 2u * (rng() % 1500u); } while (uo.count(omn)); uo.insert(omn);
        uint32_t omx; do { omx = omn + 1u + 2u * (rng() % 1200u); } while (uo.count(omx)); uo.insert(omx);
        oldB[h * 6 + 0] = omn; oldB[h * 6 + 3] = omx;
        oldB[h * 6 + 1] = rng() % 2500; oldB[h * 6 + 4] = oldB[h * 6 + 1] + 1 + rng() % 2500;
        oldB[h * 6 + 2] = rng() % 2500; oldB[h * 6 + 5] = oldB[h * 6 + 2] + 1 + rng() % 2500;
    }

    // pools + meta
    std::vector<uint32_t> sapP, maskP, handP, biP, compP, meta(nAgg * 16);
    auto appendBuilt = [&](const Built& b, uint32_t& sapBase, uint32_t& maskBase, uint32_t& handBase, uint32_t& biBase, uint32_t& compBase) {
        sapBase = uint32_t(sapP.size()); maskBase = uint32_t(maskP.size()); handBase = uint32_t(handP.size());
        biBase = uint32_t(biP.size()); compBase = uint32_t(compP.size());
        for (uint32_t v : b.sap) sapP.push_back(v);
        for (uint32_t v : b.mask) maskP.push_back(v);
        for (uint32_t v : b.handle) handP.push_back(v);
        for (uint32_t v : b.bi) biP.push_back(v);
    };

    std::vector<uint32_t> refPrevComp(nAgg, 0);
    for (uint32_t a = 0; a < nAgg; ++a)
    {
        Built cur = buildSelf(aggs[a].members, newB), prev = buildSelf(aggs[a].members, oldB);
        uint32_t sb, mb, hb, bb, cb;
        // curr
        appendBuilt(cur, sb, mb, hb, bb, cb);
        const uint32_t compBaseC = uint32_t(compP.size());
        compP.insert(compP.end(), cur.n, 0u); // scratch
        // prev with precomputed comparison prefix
        uint32_t sbp, mbp, hbp, bbp, cbp;
        appendBuilt(prev, sbp, mbp, hbp, bbp, cbp);
        const uint32_t compBaseP = uint32_t(compP.size());
        uint32_t prevTotal = 0;
        for (uint32_t i = 0; i < prev.n; ++i) { uint32_t c = countBox(prev, i); compP.push_back(prevTotal); prevTotal += c; }

        // curr total (== refPrevComp) computed the same way
        uint32_t curTotal = 0; for (uint32_t i = 0; i < cur.n; ++i) curTotal += countBox(cur, i);
        const bool process = aggs[a].valid && aggs[a].selfBit;
        refPrevComp[a] = process ? curTotal : 0u;

        uint32_t* M = &meta[a * 16];
        M[0] = aggs[a].valid; M[1] = aggs[a].selfBit; M[2] = aggs[a].isNew;
        M[3] = cur.n; M[4] = prev.n; M[5] = prevTotal;
        M[6] = sb; M[7] = mb; M[8] = hb; M[9] = bb; M[10] = compBaseC;
        M[11] = sbp; M[12] = mbp; M[13] = hbp; M[14] = bbp; M[15] = compBaseP;
    }

    // brute-force reference: for each processed aggregate, all unordered member pairs
    std::set<std::pair<uint32_t, uint32_t>> foundRef, lostRef;
    for (uint32_t a = 0; a < nAgg; ++a)
    {
        if (!(aggs[a].valid && aggs[a].selfBit)) continue;
        const auto& mem = aggs[a].members;
        for (size_t x = 0; x < mem.size(); ++x)
            for (size_t y = x + 1; y < mem.size(); ++y)
            {
                uint32_t p = mem[x], q = mem[y];
                if (groups[p] == groups[q]) continue;
                if (intersects3D(newB, p, q) && !intersects3D(oldB, p, q)) foundRef.insert({ std::min(p, q), std::max(p, q) });
                if (!aggs[a].isNew && intersects3D(oldB, p, q) && !intersects3D(newB, p, q)) lostRef.insert({ std::min(p, q), std::max(p, q) });
            }
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t maxFL = 20000;
    std::vector<uint32_t> foundBuf(maxFL * 2, 0xffffffffu), lostBuf(maxFL * 2, 0xffffffffu), shared = { 0u, 0u }, prevOut(nAgg, 123);
    PxgBufferHandle bNew = createBufferFrom(dev, newB), bOld = createBufferFrom(dev, oldB), bGrp = createBufferFrom(dev, groups);
    PxgBufferHandle bMeta = createBufferFrom(dev, meta), bSap = createBufferFrom(dev, sapP), bMask = createBufferFrom(dev, maskP);
    PxgBufferHandle bHand = createBufferFrom(dev, handP), bBi = createBufferFrom(dev, biP), bComp = createBufferFrom(dev, compP);
    PxgBufferHandle bF = createBufferFrom(dev, foundBuf), bL = createBufferFrom(dev, lostBuf), bS = createBufferFrom(dev, shared), bP = createBufferFrom(dev, prevOut);

    SelfParams pr = { nAgg, maxFL };
    const std::vector<uint32_t> strides(13, 4);
    Kernel k("broadphaseAggSelfCollision.spv", "doSelfCollision", strides, sizeof(SelfParams));
    k.dispatch(nAgg, { bNew, bOld, bGrp, bMeta, bSap, bMask, bHand, bBi, bComp, bF, bL, bS, bP }, pr);
    dev->flush();

    const auto sh = downloadBuffer<uint32_t>(dev, bS, 2);
    const auto gF = downloadBuffer<uint32_t>(dev, bF, std::min(sh[0], maxFL) * 2);
    const auto gL = downloadBuffer<uint32_t>(dev, bL, std::min(sh[1], maxFL) * 2);
    const auto gP = downloadBuffer<uint32_t>(dev, bP, nAgg);
    std::set<std::pair<uint32_t, uint32_t>> gotF, gotL;
    for (uint32_t i = 0; i < std::min(sh[0], maxFL); ++i) gotF.insert({ gF[i * 2], gF[i * 2 + 1] });
    for (uint32_t i = 0; i < std::min(sh[1], maxFL); ++i) gotL.insert({ gL[i * 2], gL[i * 2 + 1] });

    PXG_CHECK_MSG(sh[0] == foundRef.size(), "found count %u exp %zu", sh[0], foundRef.size());
    PXG_CHECK_MSG(sh[1] == lostRef.size(), "lost count %u exp %zu", sh[1], lostRef.size());
    PXG_CHECK_MSG(gotF == foundRef, "found set differs (got %zu, ref %zu)", gotF.size(), foundRef.size());
    PXG_CHECK_MSG(gotL == lostRef, "lost set differs (got %zu, ref %zu)", gotL.size(), lostRef.size());
    PXG_CHECK_MSG(foundRef.size() >= 6, "found reference too small (%zu)", foundRef.size());
    size_t pm = 0;
    for (uint32_t a = 0; a < nAgg; ++a) if (gP[a] != refPrevComp[a]) ++pm;
    PXG_CHECK_MSG(pm == 0, "%zu prevComparisons mismatches", pm);

    for (PxgBufferHandle h : { bNew, bOld, bGrp, bMeta, bSap, bMask, bHand, bBi, bComp, bF, bL, bS, bP })
        dev->destroyBuffer(h);
}

} // namespace
