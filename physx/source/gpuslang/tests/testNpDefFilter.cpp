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

// Verification of npDefFilter.slang: the deformableElementFilter.cuh sorted
// binary searches (ctFindParticleRigid / ctFindRigid / ctFindNonRigid, incl.
// the ignoreIndex2 early-accept), ctFindRange (the runsum upper-bound lookup
// incl. its deliberate r-1 wrap to 0xffffffff below the first entry) and the
// four EdgeEncoding authorship predicates (ctEdgeAuth).  Everything is pure
// integer, so all results are compared bit-exactly.  Present / absent /
// index2-mismatch / wrap cases are exercised with coverage floors.

#include "pxgTestFramework.h"

#include <algorithm>
#include <cstdint>
#include <random>
#include <set>
#include <vector>

namespace
{
using namespace pxgtest;

struct U2 { uint32_t lo, hi; };
struct PairPR { U2 id0, id1; };
struct PairR { U2 index0, index1; uint32_t index2, pad0, pad1, pad2; };
struct PairNR { U2 index0; uint32_t index1, index2; };
struct DfQuery { U2 q0, q1; uint32_t idx2, mode, value, auth; };
struct U4 { uint32_t x, y, z, w; };
struct DfParams { uint32_t numTests, numPairs, numValues, pad0; };

static U2 u2(uint64_t v) { return { (uint32_t)(v & 0xffffffffu), (uint32_t)(v >> 32) }; }
static uint64_t u64(U2 v) { return ((uint64_t)v.hi << 32) | v.lo; }

// 64-bit key generator that exercises the uint2 representation: a fraction of
// keys get a nonzero hi word (so keys differing only in hi, and keys whose lo
// order contradicts the hi order, both occur) and top-bit-set lo/hi words (so
// a signed-compare bug cannot hide).
static uint64_t genKey(std::mt19937_64& rng, uint64_t range)
{
    uint64_t lo = rng() % range;
    if (rng() % 4 == 0) lo |= 0x80000000ull;
    uint64_t hi = 0;
    if (rng() % 4 == 0) hi = 1 + rng() % 3;
    if (rng() % 16 == 0) hi |= 0x80000000ull;
    return (hi << 32) | lo;
}

// One shared buffer set (all-buffers pattern); strides for every entry.
struct Bufs { PxgBufferHandle pr, r, nr, vals, q, out; };
static Kernel makeKernel(const char* entry)
{
    return Kernel("npDefFilter.spv", entry,
                  { (uint32_t)sizeof(PairPR), (uint32_t)sizeof(PairR), (uint32_t)sizeof(PairNR),
                    4, (uint32_t)sizeof(DfQuery), (uint32_t)sizeof(U4) },
                  sizeof(DfParams));
}

// ---- C++ references (faithful transcriptions) -----------------------------

static bool refFindPR(const std::vector<PairPR>& pairs, uint64_t id0, uint64_t id1)
{
    uint32_t l = 0, r = (uint32_t)pairs.size();
    while (l < r)
    {
        const uint32_t mid = (l + r) / 2;
        const uint64_t p0 = u64(pairs[mid].id0), p1 = u64(pairs[mid].id1);
        if (id0 == p0 && id1 == p1) return true;
        else if (id0 < p0 || (id0 == p0 && id1 < p1)) r = mid;
        else l = mid + 1;
    }
    return false;
}

static int cmp3(uint64_t a0, uint64_t b0, uint64_t a1, uint64_t b1, uint32_t a2, uint32_t b2)
{
    if (a0 < b0) return -1;
    if (a0 > b0) return 1;
    if (a1 < b1) return -1;
    if (a1 > b1) return 1;
    if (a2 < b2) return -1;
    if (a2 > b2) return 1;
    return 0;
}

static bool refFindR(const std::vector<PairR>& pairs, uint64_t q0, uint64_t q1)
{
    uint32_t l = 0, r = (uint32_t)pairs.size();
    while (l < r)
    {
        const uint32_t mid = (l + r) / 2;
        const PairR& p = pairs[mid];
        const int cmp = cmp3(q0, u64(p.index0), q1, u64(p.index1), 0u, p.index2);
        if (cmp == 0 || (q0 == u64(p.index0) && q1 == u64(p.index1))) return true;
        if (cmp < 0) r = mid; else l = mid + 1;
    }
    return false;
}

static bool refFindNR(const std::vector<PairNR>& pairs, uint64_t q0, uint32_t q1, uint32_t q2, bool ignoreIndex2)
{
    uint32_t l = 0, r = (uint32_t)pairs.size();
    while (l < r)
    {
        const uint32_t mid = (l + r) / 2;
        const PairNR& p = pairs[mid];
        const int cmp = cmp3(q0, u64(p.index0), (uint64_t)q1, (uint64_t)p.index1, q2, p.index2);
        if (cmp == 0 || (ignoreIndex2 && q0 == u64(p.index0) && q1 == p.index1)) return true;
        if (cmp < 0) r = mid; else l = mid + 1;
    }
    return false;
}

static uint32_t refFindRange(uint32_t value, const std::vector<uint32_t>& values)
{
    if (values.empty()) return 0;
    uint32_t l = 0, r = (uint32_t)values.size();
    while (l < r)
    {
        const uint32_t m = (l + r) / 2;
        if (values[m] > value) r = m; else l = m + 1;
    }
    return r - 1; // wraps to 0xffffffff when value < values[0]
}

// EdgeEncoding positions (PxgFEMCloth.h)
static const uint32_t T0_EDGE = 0, T0_VTX = 11, T1_EDGE = 16, T1_VTX = 27;
static uint32_t refEdgeAuth(uint32_t auth, uint32_t li)
{
    uint32_t bits = 0;
    if (auth & (1u << (T0_EDGE + li))) bits |= 1u;
    if (auth & (1u << (T1_EDGE + li))) bits |= 2u;
    if (auth & (1u << (T0_VTX + li))) bits |= 4u;
    if (auth & (1u << (T1_VTX + li))) bits |= 8u;
    return bits;
}

// ---- shared dispatch helper ------------------------------------------------

static std::vector<U4> runEntry(const char* entry, uint32_t N,
                                const std::vector<PairPR>& pr, const std::vector<PairR>& r,
                                const std::vector<PairNR>& nr, const std::vector<uint32_t>& vals,
                                const std::vector<DfQuery>& q, uint32_t numPairs, uint32_t numValues)
{
    PxgSlangComputeDevice* dev = testDevice();
    std::vector<U4> outInit(N, { 0xdeadbeefu, 0, 0, 0 });
    Bufs b = { createBufferFrom(dev, pr), createBufferFrom(dev, r), createBufferFrom(dev, nr),
               createBufferFrom(dev, vals), createBufferFrom(dev, q), createBufferFrom(dev, outInit) };
    DfParams prm = { N, numPairs, numValues, 0 };
    Kernel k = makeKernel(entry);
    k.dispatch((N + 255) / 256, { b.pr, b.r, b.nr, b.vals, b.q, b.out }, prm);
    dev->flush();
    std::vector<U4> res = downloadBuffer<U4>(dev, b.out, N);
    dev->destroyBuffer(b.pr); dev->destroyBuffer(b.r); dev->destroyBuffer(b.nr);
    dev->destroyBuffer(b.vals); dev->destroyBuffer(b.q); dev->destroyBuffer(b.out);
    return res;
}

// Small dummies reused by tests that don't touch a given array.
static std::vector<PairPR> dummyPR() { return std::vector<PairPR>(1); }
static std::vector<PairR> dummyR() { return std::vector<PairR>(1); }
static std::vector<PairNR> dummyNR() { return std::vector<PairNR>(1); }
static std::vector<uint32_t> dummyVals() { return std::vector<uint32_t>(1, 0); }

PXG_TEST(npDefFilter_findParticleRigid)
{
    std::mt19937_64 rng(75011);
    const uint32_t NP = 977; // odd size to vary the midpoints
    std::set<std::pair<uint64_t, uint64_t>> keys;
    while (keys.size() < NP)
        keys.insert({ genKey(rng, 5000), genKey(rng, 5000) }); // small ranges -> near-miss collisions
    std::vector<PairPR> pairs;
    size_t hiUsed = 0;
    for (const auto& kk : keys)
    {
        pairs.push_back({ u2(kk.first), u2(kk.second) });
        if ((kk.first >> 32) != 0 || (kk.second >> 32) != 0) ++hiUsed;
    }

    const uint32_t N = 8192;
    std::vector<DfQuery> q(N);
    std::vector<U4> ref(N);
    size_t found = 0, missed = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        uint64_t id0, id1;
        if (i & 1) { const auto& kk = pairs[rng() % NP]; id0 = u64(kk.id0); id1 = u64(kk.id1); }
        else { id0 = genKey(rng, 5000); id1 = genKey(rng, 5000); }
        q[i] = { u2(id0), u2(id1), 0, 0, 0, 0 };
        const bool f = refFindPR(pairs, id0, id1);
        ref[i] = { f ? 1u : 0u, 0, 0, 0 };
        if (f) ++found; else ++missed;
    }

    const auto g = runEntry("ctFindParticleRigid", N, pairs, dummyR(), dummyNR(), dummyVals(), q, NP, 1);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i) if (g[i].x != ref[i].x) ++mm;
    PXG_CHECK_MSG(mm == 0, "ctFindParticleRigid: %zu mismatches", mm);
    PXG_CHECK_MSG(found > N / 3 && missed > N / 8 && hiUsed > NP / 8,
                  "ctFindParticleRigid: coverage thin (found=%zu missed=%zu hiUsed=%zu)", found, missed, hiUsed);
}

PXG_TEST(npDefFilter_findRigid)
{
    std::mt19937_64 rng(75023);
    const uint32_t NP = 1021;
    std::set<std::pair<uint64_t, uint64_t>> keys;
    while (keys.size() < NP)
        keys.insert({ genKey(rng, 4000), genKey(rng, 4000) });
    std::vector<PairR> pairs;
    for (const auto& kk : keys)
        pairs.push_back({ u2(kk.first), u2(kk.second), (uint32_t)(rng() % 7), 0, 0, 0 }); // random nonzero-capable index2
    // sort by full 3-key compare
    std::sort(pairs.begin(), pairs.end(), [](const PairR& a, const PairR& b) {
        return cmp3(u64(a.index0), u64(b.index0), u64(a.index1), u64(b.index1), a.index2, b.index2) < 0; });

    const uint32_t N = 8192;
    std::vector<DfQuery> q(N);
    std::vector<U4> ref(N);
    size_t found = 0, missed = 0, viaIgnore = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        uint64_t id0, id1;
        if (i & 1) { const PairR& p = pairs[rng() % NP]; id0 = u64(p.index0); id1 = u64(p.index1); if (p.index2 != 0) ++viaIgnore; }
        else { id0 = genKey(rng, 4000); id1 = genKey(rng, 4000); }
        q[i] = { u2(id0), u2(id1), 0, 0, 0, 0 };
        const bool f = refFindR(pairs, id0, id1);
        ref[i] = { f ? 1u : 0u, 0, 0, 0 };
        if (f) ++found; else ++missed;
    }

    const auto g = runEntry("ctFindRigid", N, dummyPR(), pairs, dummyNR(), dummyVals(), q, NP, 1);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i) if (g[i].x != ref[i].x) ++mm;
    PXG_CHECK_MSG(mm == 0, "ctFindRigid: %zu mismatches", mm);
    // every present query with stored index2 != 0 exercises the ignoreIndex2 accept
    PXG_CHECK_MSG(found > N / 3 && missed > N / 8 && viaIgnore > 500,
                  "ctFindRigid: coverage thin (found=%zu missed=%zu viaIgnore=%zu)", found, missed, viaIgnore);
}

PXG_TEST(npDefFilter_findNonRigid)
{
    std::mt19937_64 rng(75041);
    const uint32_t NP = 953;
    std::set<std::tuple<uint64_t, uint32_t, uint32_t>> keys;
    while (keys.size() < NP)
        keys.insert({ genKey(rng, 3000),
                      (uint32_t)((rng() % 3000) | (rng() % 4 == 0 ? 0x80000000u : 0u)),
                      (uint32_t)(rng() % 5) });
    std::vector<PairNR> pairs;
    for (const auto& kk : keys)
        pairs.push_back({ u2(std::get<0>(kk)), std::get<1>(kk), std::get<2>(kk) });
    std::sort(pairs.begin(), pairs.end(), [](const PairNR& a, const PairNR& b) {
        return cmp3(u64(a.index0), u64(b.index0), a.index1, b.index1, a.index2, b.index2) < 0; });

    const uint32_t N = 8192;
    std::vector<DfQuery> q(N);
    std::vector<U4> ref(N);
    size_t m0found = 0, m1exact = 0, m1wrong = 0, missed = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const uint32_t mode = i % 3 == 2 ? 1u : 0u; // 1/3 exact-mode queries
        uint64_t id0; uint32_t id1, idx2;
        const uint32_t cat = i % 4;
        if (cat < 2) { const PairNR& p = pairs[rng() % NP]; id0 = u64(p.index0); id1 = p.index1; idx2 = p.index2; }
        else if (cat == 2) { const PairNR& p = pairs[rng() % NP]; id0 = u64(p.index0); id1 = p.index1; idx2 = p.index2 + 1; } // wrong index2
        else { id0 = genKey(rng, 3000); id1 = (uint32_t)((rng() % 3000) | (rng() % 4 == 0 ? 0x80000000u : 0u)); idx2 = (uint32_t)(rng() % 5); }
        q[i] = { u2(id0), { id1, 0 }, idx2, mode, 0, 0 };
        const bool ignore = (mode == 0);
        const bool f = refFindNR(pairs, id0, id1, ignore ? 0u : idx2, ignore);
        ref[i] = { f ? 1u : 0u, 0, 0, 0 };
        if (f && mode == 0) ++m0found;
        else if (f && mode == 1) ++m1exact;
        else if (!f && mode == 1 && cat == 2) ++m1wrong;
        else if (!f) ++missed;
    }

    const auto g = runEntry("ctFindNonRigid", N, dummyPR(), dummyR(), pairs, dummyVals(), q, NP, 1);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i) if (g[i].x != ref[i].x) ++mm;
    PXG_CHECK_MSG(mm == 0, "ctFindNonRigid: %zu mismatches", mm);
    PXG_CHECK_MSG(m0found > 1000 && m1exact > 300 && m1wrong > 100 && missed > 500,
                  "ctFindNonRigid: coverage thin (m0=%zu m1=%zu m1wrong=%zu missed=%zu)", m0found, m1exact, m1wrong, missed);
}

PXG_TEST(npDefFilter_rangeAuth)
{
    std::mt19937 rng(75063);
    // --- ctFindRange over a runsum-like sorted array ---
    const uint32_t NV = 37;
    std::vector<uint32_t> vals(NV);
    uint32_t acc = 20; // nonzero first entry so below-first queries wrap
    for (uint32_t i = 0; i < NV; ++i) { vals[i] = acc; acc += 1 + rng() % 40; }

    const uint32_t N = 8192;
    std::vector<DfQuery> q(N);
    std::vector<U4> refRange(N), refAuth(N);
    size_t wrapped = 0, interior = 0, above = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const uint32_t value = rng() % (acc + 60); // spans below-first .. above-last
        const uint32_t auth = rng();
        const uint32_t li = rng() % 3;
        q[i] = { { 0, 0 }, { 0, 0 }, 0, 0, 0, 0 };
        q[i].value = value; q[i].auth = auth; q[i].mode = li;
        const uint32_t rr = refFindRange(value, vals);
        refRange[i] = { rr, 0, 0, 0 };
        if (rr == 0xffffffffu) ++wrapped; else if (value >= vals[NV - 1]) ++above; else ++interior;
        refAuth[i] = { refEdgeAuth(auth, li), 0, 0, 0 };
    }
    // ctFindRange reads q.value; ctEdgeAuth reads q.auth and q.value as the local index —
    // so run the auth entry with its own query buffer where value = li.
    std::vector<DfQuery> qa = q;
    for (uint32_t i = 0; i < N; ++i) qa[i].value = q[i].mode;

    const auto gR = runEntry("ctFindRange", N, dummyPR(), dummyR(), dummyNR(), vals, q, 1, NV);
    const auto gA = runEntry("ctEdgeAuth", N, dummyPR(), dummyR(), dummyNR(), vals, qa, 1, NV);

    size_t mmR = 0, mmA = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (gR[i].x != refRange[i].x) ++mmR;
        if (gA[i].x != refAuth[i].x) ++mmA;
    }
    PXG_CHECK_MSG(mmR == 0, "ctFindRange: %zu mismatches", mmR);
    PXG_CHECK_MSG(mmA == 0, "ctEdgeAuth: %zu mismatches", mmA);
    PXG_CHECK_MSG(wrapped > 50 && interior > N / 2 && above > 50,
                  "ctFindRange: coverage thin (wrapped=%zu interior=%zu above=%zu)", wrapped, interior, above);

    // length == 0 early-out: with numValues = 0 every query must return 0.
    const uint32_t NZ = 256;
    std::vector<DfQuery> qz(q.begin(), q.begin() + NZ);
    const auto gz = runEntry("ctFindRange", NZ, dummyPR(), dummyR(), dummyNR(), vals, qz, 1, 0);
    size_t mmZ = 0;
    for (uint32_t i = 0; i < NZ; ++i) if (gz[i].x != 0) ++mmZ;
    PXG_CHECK_MSG(mmZ == 0, "ctFindRange(length=0): %zu mismatches", mmZ);
}

} // namespace
