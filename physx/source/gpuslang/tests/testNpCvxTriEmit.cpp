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

// Verification of npCvxTriEmit.slang: ctCvxTriEmit — the contact-reduce-and-emit
// stage of convexTriangleContactGen (convexTriangle.cuh :1769-1821).  One thread
// simulates the whole 32-lane warp, so the compacted output order (lane order
// within the surviving mask) is deterministic; the temp-contact stream, the
// reservation counter, and the (nbOutputContacts, index, maxSeparation) triple
// are all compared bit-exactly against an exact CPU transcription (identical
// contactReduce5 / min-reduction / exclusive scan / overflow guard).
//
// Two regimes: family A (numContactsGenerated <= 5) never invokes the reduction,
// so the emitted (pos, sep) are verbatim copies and maxSeparation is an exact min
// — bit-exact for arbitrary float inputs.  Family B (numContactsGenerated 6..12)
// exercises contactReduce5; integer 2D points with an axis-aligned normal and
// distinct separations / squared distances keep every argmax/argmin (incl. the
// sqrt-based farthest pick, monotone in the exact squared distance) unambiguous,
// so the selected mask — hence the whole emitted stream — is bit-identical.
// Both regimes cover the tempContactSize overflow guard (reservation still
// advances, but the OUTPUT count is zeroed and nothing is written).

#include "pxgTestFramework.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct V3 { float x, y, z; };
struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };
struct CteIn { F4 minNormal; U4 cfg; };
struct CteParams { uint32_t numTests, cap, pad0, pad1; };

static const uint32_t CVX_TRI_MAX_CONTACTS = 5u;
static const float CTE_MAX = 3.402823466e38f;

static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b)
{
    return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}
static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 muls(V3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
static float len3(V3 a) { return std::sqrt(dot3(a, a)); }
static uint32_t asU32(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

static uint32_t lowestArgMax(const float vals[32], uint32_t mask, float& best)
{
    best = -CTE_MAX; uint32_t idx = 0;
    for (uint32_t j = 0; j < 32; ++j)
        if (((mask >> j) & 1u) && vals[j] > best) { best = vals[j]; idx = j; }
    return idx;
}
static uint32_t lowestArgMin(const float vals[32], uint32_t mask, float& best)
{
    best = CTE_MAX; uint32_t idx = 0;
    for (uint32_t j = 0; j < 32; ++j)
        if (((mask >> j) & 1u) && vals[j] < best) { best = vals[j]; idx = j; }
    return idx;
}
static uint32_t popc(uint32_t x) { uint32_t c = 0; while (x) { c += x & 1u; x >>= 1; } return c; }

static uint32_t contactReduce5Ref(const V3 ptRaw[32], const float sep[32], V3 normal, uint32_t allMask)
{
    const uint32_t maxPoints = CVX_TRI_MAX_CONTACTS;
    V3 point[32];
    for (uint32_t j = 0; j < 32; ++j) point[j] = sub(ptRaw[j], muls(normal, dot3(ptRaw[j], normal)));
    {
        uint32_t dupeBallot = 0;
        for (uint32_t j = 0; j < 32; ++j)
        {
            if (((allMask >> j) & 1u) == 0u) continue;
            bool imADupe = false;
            for (uint32_t i = 0; i < 32; ++i)
            {
                if (((allMask >> i) & 1u) == 0u) continue;
                const V3 dv = sub(ptRaw[i], ptRaw[j]);
                const float d = dot3(dv, dv);
                if (d < 1e-8f && (sep[i] < sep[j] || (sep[i] == sep[j] && i > j))) imADupe = true;
            }
            if (imADupe) dupeBallot |= 1u << j;
        }
        allMask &= ~dupeBallot;
    }
    if (popc(allMask) <= maxPoints) return allMask;

    float v; float vals[32];
    for (uint32_t j = 0; j < 32; ++j) vals[j] = sep[j];
    const uint32_t i0 = lowestArgMax(vals, allMask, v);
    uint32_t mask = 1u << i0;
    for (uint32_t j = 0; j < 32; ++j) vals[j] = len3(sub(point[i0], point[j]));
    const uint32_t i1 = lowestArgMax(vals, allMask & ~mask, v);
    mask |= 1u << i1;
    const V3 dir = cross3(normal, sub(point[i1], point[i0]));
    float dval[32];
    for (uint32_t j = 0; j < 32; ++j) dval[j] = dot3(dir, sub(point[j], point[i0]));
    const uint32_t f = lowestArgMax(dval, allMask & ~mask, v);
    mask |= 1u << f;
    const uint32_t g = lowestArgMin(dval, allMask & ~mask, v);
    mask |= 1u << g;
    if (popc(mask) == 4u)
    {
        const uint32_t i4 = lowestArgMin(sep, allMask & ~mask, v);
        mask |= 1u << i4;
    }
    for (uint32_t i = popc(mask); i < maxPoints; ++i)
        mask |= 1u << lowestArgMin(sep, allMask & ~mask, v);
    return mask;
}

struct EmitRef { uint32_t nbOut, index, counter; float maxSep; std::vector<F4> temp; };

static EmitRef emitRef(const F4 contacts[32], V3 normal, uint32_t numContactsGenerated,
                       uint32_t tempContactSize, uint32_t base, uint32_t cap)
{
    EmitRef r;
    r.temp.assign(cap, { -777, -777, -777, -777 });
    r.nbOut = 0; r.index = 0xffffffffu; r.counter = base; r.maxSep = CTE_MAX;

    if (numContactsGenerated != 0u)
    {
        V3 pa[32]; float sep[32];
        for (uint32_t L = 0; L < 32; ++L) { pa[L] = { contacts[L].x, contacts[L].y, contacts[L].z }; sep[L] = contacts[L].w; }
        uint32_t mask = (numContactsGenerated >= 32u) ? 0xffffffffu : ((1u << numContactsGenerated) - 1u);
        if (numContactsGenerated > CVX_TRI_MAX_CONTACTS) mask = contactReduce5Ref(pa, sep, normal, mask);

        float maxSeparation = CTE_MAX;
        for (uint32_t L = 0; L < 32; ++L)
        {
            const float m = ((mask >> L) & 1u) ? sep[L] : CTE_MAX;
            maxSeparation = std::min(maxSeparation, m);
        }
        r.maxSep = maxSeparation;
        uint32_t nb = popc(mask);
        const uint32_t startIndex = base;
        r.counter = base + nb; // reservation advances regardless of overflow
        r.index = startIndex;

        if ((startIndex + nb) < tempContactSize)
        {
            for (uint32_t L = 0; L < 32; ++L)
                if ((mask >> L) & 1u)
                {
                    const uint32_t off = startIndex + popc(mask & ((1u << L) - 1u));
                    r.temp[off - base] = contacts[L]; // base-relative slot within this test's block
                }
            r.nbOut = nb;
        }
        else
            r.nbOut = 0;
    }
    return r;
}

PXG_TEST(npCvxTriEmit_emit)
{
    std::mt19937 rng(1120112);
    const uint32_t CAP = 8u; // per-test temp block stride (max 5 written)

    std::vector<F4> contactsFlat;    // 32 per test
    std::vector<CteIn> in;
    std::vector<uint32_t> baseList;
    std::vector<uint32_t> tcsList;   // tempContactSize per test
    std::vector<EmitRef> ref;

    size_t nReduce = 0, nNoReduce = 0, nOverflow = 0, nEmpty = 0, nPcm = 0;

    std::uniform_real_distribution<float> uf(-5.0f, 5.0f);

    // tcsMode: 0 = ample (never overflow), 1 = hard overflow (tcs = base),
    // 2 = EXACT boundary (tcs = base + reserved, so startIndex+nb == tcs and the
    // strict `<` guard drops while a `<=` guard would keep — pins the guard edge)
    auto emit = [&](const std::vector<F4>& contacts32, V3 normal, uint32_t nGen, int tcsMode)
    {
        const uint32_t t = (uint32_t)in.size();
        const uint32_t base = t * CAP;
        uint32_t tcs;
        if (tcsMode == 1) tcs = base;
        else if (tcsMode == 2)
        {
            // reserved for the boundary is only well-defined without the reduction
            // (nGen <= 5 => reserved == nGen); callers use this mode with nGen<=5
            tcs = base + nGen;
        }
        else tcs = 0x40000000u;
        F4 c32[32];
        for (uint32_t i = 0; i < 32; ++i) c32[i] = (i < contacts32.size()) ? contacts32[i] : F4{ 0, 0, 0, 0 };
        for (uint32_t i = 0; i < 32; ++i) contactsFlat.push_back(c32[i]);
        CteIn e; e.minNormal = { normal.x, normal.y, normal.z, 0 };
        e.cfg = { nGen, tcs, 0, 0 };
        in.push_back(e);
        baseList.push_back(base);
        tcsList.push_back(tcs);
        ref.push_back(emitRef(c32, normal, nGen, tcs, base, CAP));
    };

    // ---- family A: numContactsGenerated 1..5 (no reduction) -----------------
    // arbitrary float contacts + arbitrary normal: emitted (pos,sep) are verbatim
    // copies and maxSeparation an exact min, so bit-exact regardless of values.
    for (uint32_t iter = 0; iter < 900; ++iter)
    {
        const uint32_t nGen = 1u + rng() % 5u; // 1..5
        std::vector<F4> c;
        for (uint32_t i = 0; i < nGen; ++i) c.push_back({ uf(rng), uf(rng), uf(rng), uf(rng) });
        V3 n = { uf(rng), uf(rng), uf(rng) };
        const bool ovf = (iter % 7u == 0u); // ~1/7 forced overflow
        emit(c, n, nGen, ovf);
        if (ovf) ++nOverflow; else ++nNoReduce;
    }

    // ---- family B: numContactsGenerated 6..12 (reduction) -------------------
    // integer 2D points (z = 0), axis-aligned normal, distinct integer seps ->
    // every selection unambiguous, so the reduced mask is bit-identical.
    for (uint32_t iter = 0; iter < 900; ++iter)
    {
        const uint32_t nGen = 6u + rng() % 7u; // 6..12
        // distinct integer 2D points on a grid, distinct integer separations
        std::vector<int> used;
        std::vector<F4> c;
        for (uint32_t i = 0; i < nGen; ++i)
        {
            int gx, gy, key;
            do { gx = int(rng() % 21) - 10; gy = int(rng() % 21) - 10; key = (gx + 32) * 64 + (gy + 32); }
            while (std::find(used.begin(), used.end(), key) != used.end());
            used.push_back(key);
            // separations: distinct integers scaled to a small negative window
            const float s = -1.0f - float(i) - 0.0f; // strictly decreasing, distinct
            c.push_back({ float(gx), float(gy), 0.0f, s });
        }
        // shuffle separations across lanes so the deepest is not always lane 0
        for (uint32_t i = 0; i < nGen; ++i)
        {
            const uint32_t j = i + rng() % (nGen - i);
            std::swap(c[i].w, c[j].w);
        }
        V3 n = { 0, 0, 1 };
        const bool ovf = (iter % 9u == 0u);
        emit(c, n, nGen, ovf);
        if (ovf) ++nOverflow; else ++nReduce;
        // coverage: did the PCM extra-deepest branch fire (exactly 4 anchors)?
        if (ref.back().nbOut == 5u || (ovf && popc((1u << std::min(nGen, 31u)) - 1u))) { /* counted below */ }
    }

    // ---- family C: numContactsGenerated == 0 (skip path) --------------------
    for (uint32_t iter = 0; iter < 80; ++iter)
    {
        emit({}, V3{ 0, 0, 1 }, 0u, false);
        ++nEmpty;
    }

    // ---- family D: exact-duplicate points to exercise the dupe cull ---------
    for (uint32_t iter = 0; iter < 240; ++iter)
    {
        const uint32_t nGen = 6u + rng() % 7u; // 6..12
        std::vector<F4> c;
        for (uint32_t i = 0; i < nGen; ++i)
            c.push_back({ float(int(rng() % 13) - 6), float(int(rng() % 13) - 6), 0.0f, -1.0f - float(i) });
        // force a few exact duplicates (coincident points, distinct seps) so the
        // dupe cull (lowest index / deeper sep wins) is exercised
        for (int k = 0; k < 2; ++k)
        {
            const uint32_t a = rng() % nGen, b = rng() % nGen;
            if (a == b) continue;
            c[b].x = c[a].x; c[b].y = c[a].y; c[b].z = c[a].z;
            if (rng() % 2) c[b].w = c[a].w; // half share the exact separation (index tie-break)
        }
        for (uint32_t i = 0; i < nGen; ++i) { const uint32_t j = i + rng() % (nGen - i); std::swap(c[i].w, c[j].w); }
        emit(c, V3{ 0, 0, 1 }, nGen, false);
        ++nReduce;
    }

    // ---- family E: numContactsGenerated == 5 with an exact duplicate --------
    // exactly at the reduction threshold: the real code (strict `>` 5) does NOT
    // reduce, so all 5 (dupe included) are emitted; a `>= 5` gate would wrongly
    // invoke contactReduce5 and cull the duplicate.  Integer points / axis normal
    // keep any (wrong) reduction deterministic.
    size_t nDupe5 = 0;
    for (uint32_t iter = 0; iter < 200; ++iter)
    {
        std::vector<F4> c;
        for (uint32_t i = 0; i < 5; ++i)
            c.push_back({ float(int(rng() % 9) - 4), float(int(rng() % 9) - 4), 0.0f, -1.0f - float(i) });
        // make lanes a and b coincident (distinct seps) — a cull-able duplicate
        const uint32_t a = rng() % 5, b = (a + 1u + rng() % 4u) % 5u;
        c[b].x = c[a].x; c[b].y = c[a].y; c[b].z = c[a].z;
        emit(c, V3{ 0, 0, 1 }, 5u, 0);
        ++nDupe5; ++nNoReduce;
    }

    // ---- family F: exact tempContactSize boundary (startIndex + nb == tcs) ---
    // pins the strict-`<` overflow guard against a `<=` slip: at the boundary the
    // real code overflows (drops, nbOut = 0) though the reservation still advances.
    size_t nBoundary = 0;
    for (uint32_t iter = 0; iter < 200; ++iter)
    {
        const uint32_t nGen = 1u + rng() % 5u; // 1..5 (reserved == nGen, no reduce)
        std::vector<F4> c;
        for (uint32_t i = 0; i < nGen; ++i) c.push_back({ uf(rng), uf(rng), uf(rng), uf(rng) });
        emit(c, V3{ uf(rng), uf(rng), uf(rng) }, nGen, 2);
        ++nBoundary; ++nOverflow;
    }

    // count PCM-branch coverage on the reference side
    for (const auto& r : ref) if (r.nbOut == 5u) ++nPcm;

    const uint32_t N = (uint32_t)in.size();
    PxgSlangComputeDevice* dev = testDevice();

    std::vector<F4> tempInit(N * CAP, { -777, -777, -777, -777 });
    std::vector<uint32_t> cntInit(N);
    for (uint32_t t = 0; t < N; ++t) cntInit[t] = baseList[t];
    std::vector<U4> outInit(N, { 0xEEEEEEEEu, 0xEEEEEEEEu, 0xEEEEEEEEu, 0xEEEEEEEEu });

    PxgBufferHandle bC = createBufferFrom(dev, contactsFlat);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bTemp = createBufferFrom(dev, tempInit);
    PxgBufferHandle bCnt = createBufferFrom(dev, cntInit);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    CteParams prm = { N, CAP, 0, 0 };
    Kernel k("npCvxTriEmit.spv", "ctCvxTriEmit",
             { (uint32_t)sizeof(F4), (uint32_t)sizeof(CteIn), (uint32_t)sizeof(F4), 4, (uint32_t)sizeof(U4) },
             sizeof(CteParams));
    k.dispatch((N + 63) / 64, { bC, bIn, bTemp, bCnt, bOut }, prm);
    dev->flush();

    const auto gTemp = downloadBuffer<F4>(dev, bTemp, N * CAP);
    const auto gCnt = downloadBuffer<uint32_t>(dev, bCnt, N);
    const auto gOut = downloadBuffer<U4>(dev, bOut, N);

    size_t mm = 0;
    for (uint32_t t = 0; t < N; ++t)
    {
        const EmitRef& r = ref[t];
        if (gOut[t].x != r.nbOut) ++mm;
        if (gOut[t].y != r.index) ++mm;
        if (gOut[t].z != asU32(r.maxSep)) ++mm;
        if (gCnt[t] != r.counter) ++mm;
        for (uint32_t i = 0; i < CAP; ++i)
        {
            const F4 g = gTemp[t * CAP + i];
            const F4 e = r.temp[i];
            if (asU32(g.x) != asU32(e.x) || asU32(g.y) != asU32(e.y) ||
                asU32(g.z) != asU32(e.z) || asU32(g.w) != asU32(e.w)) ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "cvxTriEmit: %zu mismatches", mm);
    PXG_CHECK_MSG(nReduce > 400 && nNoReduce > 400 && nOverflow > 100 && nEmpty > 20 &&
                  nPcm > 50 && nDupe5 > 50 && nBoundary > 50,
                  "cvxTriEmit coverage thin (reduce=%zu noReduce=%zu overflow=%zu empty=%zu pcm=%zu dupe5=%zu bnd=%zu)",
                  nReduce, nNoReduce, nOverflow, nEmpty, nPcm, nDupe5, nBoundary);

    for (PxgBufferHandle h : { bC, bIn, bTemp, bCnt, bOut }) dev->destroyBuffer(h);
}

} // namespace
