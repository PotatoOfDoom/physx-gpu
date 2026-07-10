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

// Verification of npSbRemap.slang (softbodyPrimitives.cu remap/reorder
// tail).  The remap fixtures pin the batch semantics — a CONTAINING sim
// tet breaks at the FIRST batch with any hit and the LOWEST intersecting
// lane wins (a containing tet in a later batch is ignored; the fallback
// min-distance search by contrast is GLOBAL across batches) — plus the
// warpReduction lowest-lane float-equality tiebreak (duplicate tet
// coordinates on two lanes produce bit-identical distances), the
// count-clamp write-back, the prevContactCount window (contacts below
// it stay untouched), the empty-range release behaviour (winner lane 0
// with simTetInd 0xffffffff OR-flooding the encoded pair — faithful),
// rmFem's isValidTet candidate gate / all-invalid tet0+zero-bary
// fallback / side select / contact-stashed-in-barycentric convention /
// unclamped fallback bary, and the reorder gather with the rigid-id u64
// extraction.  Containment decisions use robust-margin fixtures (points
// well inside/outside the 1e-5 bary window); fallback distances are
// separated by >= 5% so the winner index is exact while the barycentric
// output compares with tolerance.

#include "pxgTestFramework.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };
struct SrParams { uint32_t maxContacts, mode, hasPrev, pad0; };

struct V3 { float x, y, z; };
static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 add(V3 a, V3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 mul(V3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 crossp(V3 a, V3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static V3 v3(F4 f) { return { f.x, f.y, f.z }; }

static V3 closestPtPointTriangle2Ref(V3 p, V3 a, V3 b, V3 c, V3 ab, V3 ac)
{
    const V3 ap = sub(p, a);
    const float d1 = dot3(ab, ap), d2 = dot3(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;
    const V3 bp = sub(p, b);
    const float d3 = dot3(ab, bp), d4 = dot3(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) return add(a, mul(ab, d1 / (d1 - d3)));
    const V3 cp = sub(p, c);
    const float d5 = dot3(ab, cp), d6 = dot3(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) return add(a, mul(ac, d2 / (d2 - d6)));
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
    {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return add(b, mul(sub(c, b), w));
    }
    const float denom = 1.0f / (va + vb + vc);
    return add(a, add(mul(ab, vb * denom), mul(ac, vc * denom)));
}

// Gu::closestPtPointTetrahedron: face order (a,b,c),(a,c,d),(a,d,b),(b,d,c)
static V3 guClosestRef(V3 p, V3 a, V3 b, V3 c, V3 d)
{
    const V3 ab = sub(b, a), ac = sub(c, a), ad = sub(d, a), bc = sub(c, b), bd = sub(d, b);
    V3 best = closestPtPointTriangle2Ref(p, a, b, c, ab, ac);
    V3 diff = sub(best, p);
    float bestSq = dot3(diff, diff);
    V3 pt = closestPtPointTriangle2Ref(p, a, c, d, ac, ad);
    diff = sub(pt, p);
    float sq = dot3(diff, diff);
    if (sq < bestSq) { best = pt; bestSq = sq; }
    pt = closestPtPointTriangle2Ref(p, a, d, b, ad, ab);
    diff = sub(pt, p);
    sq = dot3(diff, diff);
    if (sq < bestSq) { best = pt; bestSq = sq; }
    pt = closestPtPointTriangle2Ref(p, b, d, c, bd, bc);
    diff = sub(pt, p);
    sq = dot3(diff, diff);
    if (sq < bestSq) { best = pt; bestSq = sq; }
    return best;
}

static bool tetBaryRef(V3 a, V3 b, V3 c, V3 d, V3 point, F4& bary)
{
    const float eps = 1e-5f;
    const V3 ba = sub(b, a), ca = sub(c, a), da = sub(d, a), pa = sub(point, a);
    const float detBcd = dot3(ba, crossp(ca, da));
    const float v = dot3(pa, crossp(ca, da)) / detBcd;
    const float w = dot3(ba, crossp(pa, da)) / detBcd;
    const float x = dot3(ba, crossp(ca, pa)) / detBcd;
    const float u = 1.0f - v - w - x;
    auto cl = [](float t) { return std::min(std::max(t, 0.0f), 1.0f); };
    bary = { cl(u), cl(v), cl(w), cl(x) };
    return (u >= -eps && u <= 1 + eps) && (v >= -eps && v <= 1 + eps) &&
           (w >= -eps && w <= 1 + eps) && (x >= -eps && x <= 1 + eps);
}
static bool isValidTetRef(V3 a, V3 b, V3 c, V3 d)
{
    return dot3(sub(b, a), crossp(sub(c, a), sub(d, a))) > 1e-9f;
}

static float grid64(std::mt19937& rng, float lo, float hi)
{
    const int n = int((hi - lo) * 64.0f);
    return lo + float(int(rng() % uint32_t(n + 1))) / 64.0f;
}

static const uint32_t POISON = 0xCDCDCDCDu;
static const uint32_t SBID = 11;

// scene: sim tets + remap tables shared by both remap tests -------------------
struct RemapScene
{
    std::vector<F4> simVerts;
    std::vector<U4> simTets;    // some entries degenerate (coplanar)
    std::vector<uint32_t> remap;
    std::vector<uint32_t> accum; // per collision tet
    std::vector<V3> tetCenter;  // center of each sim tet (garbage for degenerate)
    std::vector<char> tetValid;
};

// build one well-conditioned sim tet at a grid offset; returns its index
static uint32_t addTet(RemapScene& sc, std::mt19937& rng, V3 base, bool degenerate = false)
{
    const uint32_t v0 = (uint32_t)sc.simVerts.size();
    const V3 a = base;
    const V3 b = add(base, V3{ 1.0f + grid64(rng, 0, 0.25f), grid64(rng, 0, 0.125f), grid64(rng, 0, 0.125f) });
    const V3 c = add(base, V3{ grid64(rng, 0, 0.125f), 1.0f + grid64(rng, 0, 0.25f), grid64(rng, 0, 0.125f) });
    V3 d = add(base, V3{ grid64(rng, 0, 0.125f), grid64(rng, 0, 0.125f), 1.0f + grid64(rng, 0, 0.25f) });
    if (degenerate) // INVERTED: strictly negative volume (isValidTet needs > 1e-9),
                    // never exactly zero so barycentrics stay finite
        d = add(base, V3{ 0.5f, 0.5f, -0.25f });
    sc.simVerts.push_back({ a.x, a.y, a.z, 1 });
    sc.simVerts.push_back({ b.x, b.y, b.z, 1 });
    sc.simVerts.push_back({ c.x, c.y, c.z, 1 });
    sc.simVerts.push_back({ d.x, d.y, d.z, 1 });
    sc.simTets.push_back({ v0, v0 + 1, v0 + 2, v0 + 3 });
    sc.tetCenter.push_back(mul(add(add(a, b), add(c, d)), 0.25f));
    sc.tetValid.push_back(degenerate ? 0 : 1);
    return (uint32_t)sc.simTets.size() - 1;
}

// GPU runner -------------------------------------------------------------------
struct RmRun
{
    std::vector<F4> bary0, bary1, sPt, sNp, sBary, dbg;
    std::vector<U4> info, sInfo;
    std::vector<uint32_t> rigidIds;
    uint32_t count;
};
static RmRun runRm(const char* entry, const SrParams& prm, uint32_t NC,
                   const std::vector<F4>& point, const std::vector<F4>& bary0,
                   const std::vector<F4>& bary1, const std::vector<U4>& info,
                   const RemapScene& sc, uint32_t count0, uint32_t prev0,
                   const std::vector<uint32_t>& remapByRigid,
                   const std::vector<F4>& normalPen)
{
    PxgSlangComputeDevice* dev = testDevice();
    std::vector<uint32_t> cnt = { count0 }, prv = { prev0 };
    std::vector<F4> f4Init(NC, { -9, -9, -9, -9 });
    std::vector<U4> u4Init(NC, { POISON, POISON, POISON, POISON });
    std::vector<uint32_t> u2Init(NC * 2, POISON);
    std::vector<F4> dbgInit(NC, { -7, -7, -7, -7 });
    std::vector<uint32_t> dum = { 0 };

    PxgBufferHandle bP = createBufferFrom(dev, point);
    PxgBufferHandle b0 = createBufferFrom(dev, bary0);
    PxgBufferHandle b1 = createBufferFrom(dev, bary1);
    PxgBufferHandle bI = createBufferFrom(dev, info);
    PxgBufferHandle bT = createBufferFrom(dev, sc.simTets.empty() ? std::vector<U4>{ { 0, 0, 0, 0 } } : sc.simTets);
    PxgBufferHandle bV = createBufferFrom(dev, sc.simVerts.empty() ? std::vector<F4>{ { 0, 0, 0, 0 } } : sc.simVerts);
    PxgBufferHandle bR = createBufferFrom(dev, sc.remap.empty() ? dum : sc.remap);
    PxgBufferHandle bA = createBufferFrom(dev, sc.accum.empty() ? dum : sc.accum);
    PxgBufferHandle bC = createBufferFrom(dev, cnt);
    PxgBufferHandle bPr = createBufferFrom(dev, prv);
    PxgBufferHandle bRR = createBufferFrom(dev, remapByRigid.empty() ? dum : remapByRigid);
    PxgBufferHandle bN = createBufferFrom(dev, normalPen.empty() ? std::vector<F4>{ { 0, 0, 0, 0 } } : normalPen);
    PxgBufferHandle bSP = createBufferFrom(dev, f4Init);
    PxgBufferHandle bSN = createBufferFrom(dev, f4Init);
    PxgBufferHandle bSB = createBufferFrom(dev, f4Init);
    PxgBufferHandle bSI = createBufferFrom(dev, u4Init);
    PxgBufferHandle bRI = createBufferFrom(dev, u2Init);
    PxgBufferHandle bD = createBufferFrom(dev, dbgInit);

    Kernel k("npSbRemap.spv", entry,
             { 16, 16, 16, 16, 16, 16, 4, 4, 4, 4, 4, 16, 16, 16, 16, 16, 8, 16 },
             sizeof(SrParams));
    k.dispatch((NC + 63) / 64, { bP, b0, b1, bI, bT, bV, bR, bA, bC, bPr, bRR, bN,
                                 bSP, bSN, bSB, bSI, bRI, bD }, prm);
    dev->flush();
    RmRun r;
    r.bary0 = downloadBuffer<F4>(dev, b0, NC);
    r.bary1 = downloadBuffer<F4>(dev, b1, NC);
    r.info = downloadBuffer<U4>(dev, bI, NC);
    r.sPt = downloadBuffer<F4>(dev, bSP, NC);
    r.sNp = downloadBuffer<F4>(dev, bSN, NC);
    r.sBary = downloadBuffer<F4>(dev, bSB, NC);
    r.sInfo = downloadBuffer<U4>(dev, bSI, NC);
    r.rigidIds = downloadBuffer<uint32_t>(dev, bRI, NC * 2);
    r.dbg = downloadBuffer<F4>(dev, bD, NC);
    r.count = downloadBuffer<uint32_t>(dev, bC, 1)[0];
    for (PxgBufferHandle h : { bP, b0, b1, bI, bT, bV, bR, bA, bC, bPr, bRR, bN, bSP, bSN, bSB, bSI, bRI, bD })
        dev->destroyBuffer(h);
    return r;
}

// shared CPU reference for the remap core (mode: 0 = other, 1/2 = fem side 0/1)
struct RemapRef
{
    uint32_t newPair;
    F4 bary;
    bool wroteBary;
    int kind; // 0 = intersect, 1 = fallback min, 2 = no-valid-tet fallback
};
static RemapRef refRemap(const RemapScene& sc, V3 contact, uint32_t colTet, uint32_t sbId,
                         bool femVariant, bool clampFallback)
{
    RemapRef ro;
    ro.wroteBary = true;
    const uint32_t endInd = sc.accum[colTet];
    const uint32_t startInd = colTet != 0 ? sc.accum[colTet - 1] : 0;
    const uint32_t range = endInd - startInd;
    const uint32_t nbIter = (range + 31) / 32;

    uint32_t laneSimTet[32];
    F4 laneBary[32];
    float laneSq[32];
    for (int l = 0; l < 32; ++l) { laneSimTet[l] = 0xffffffffu; laneBary[l] = { 0, 0, 0, 0 }; laneSq[l] = 3.402823466e38f; }

    bool anyTet = false;
    for (uint32_t j = 0; j < nbIter; ++j)
    {
        int hitLane = -1;
        uint32_t hitTet = 0;
        F4 hitBary = {};
        for (uint32_t l = 0; l < 32; ++l)
        {
            const uint32_t workIndex = l + j * 32;
            if (workIndex >= range) continue;
            const uint32_t tmp = sc.remap[startInd + workIndex];
            const U4 t = sc.simTets[tmp];
            const V3 a = v3(sc.simVerts[t.x]), b = v3(sc.simVerts[t.y]), c = v3(sc.simVerts[t.z]), d = v3(sc.simVerts[t.w]);
            if (femVariant && !isValidTetRef(a, b, c, d)) continue;
            if (femVariant) anyTet = true;
            F4 tb;
            const bool intersect = tetBaryRef(a, b, c, d, contact, tb);
            if (!intersect)
            {
                const V3 cp = guClosestRef(contact, a, b, c, d);
                const V3 v = sub(contact, cp);
                const float dd = dot3(v, v);
                if (dd < laneSq[l])
                {
                    laneSq[l] = dd;
                    laneSimTet[l] = tmp;
                    F4 cb;
                    tetBaryRef(a, b, c, d, cp, cb);
                    laneBary[l] = cb;
                }
            }
            else if (hitLane < 0)
            {
                hitLane = (int)l;
                hitTet = tmp;
                hitBary = tb;
            }
        }
        if (hitLane >= 0)
        {
            ro.newPair = (sbId << 20) | hitTet;
            ro.bary = hitBary;
            ro.kind = 0;
            return ro;
        }
    }
    if (femVariant && !anyTet)
    {
        ro.newPair = (sbId << 20) | 0;
        ro.bary = { 0, 0, 0, 0 };
        ro.kind = 2;
        return ro;
    }
    float best = 3.402823466e38f;
    for (int l = 0; l < 32; ++l) best = std::min(best, laneSq[l]);
    int winner = 0;
    for (int l = 0; l < 32; ++l)
        if (laneSq[l] == best) { winner = l; break; }
    F4 bb = laneBary[winner];
    if (clampFallback)
    {
        auto cl = [](float t) { return std::min(std::max(t, 0.0f), 1.0f); };
        bb = { cl(bb.x), cl(bb.y), cl(bb.z), cl(bb.w) };
    }
    ro.newPair = (sbId << 20) | laneSimTet[winner];
    ro.bary = bb;
    ro.kind = 1;
    return ro;
}

static bool eqF4(F4 a, F4 b, float tol)
{
    return almostEqual(a.x, b.x, tol, tol) && almostEqual(a.y, b.y, tol, tol) &&
           almostEqual(a.z, b.z, tol, tol) && almostEqual(a.w, b.w, tol, tol);
}

// build the shared scene + per-contact expectations ----------------------------
struct Built
{
    RemapScene sc;
    std::vector<F4> point;
    std::vector<U4> info;    // pairInd0 lo/hi, pairInd1, aux
    std::vector<int> family; // 0..5
    std::vector<uint32_t> colTet, sbId; // decoupled from the contact index
    std::vector<std::vector<uint32_t>> entriesPerContact;
};
static Built buildRemapFixtures(uint32_t NC, uint32_t seed)
{
    std::mt19937 rng(seed);
    Built bt;
    RemapScene& sc = bt.sc;
    bt.point.resize(NC);
    bt.info.resize(NC);
    bt.family.resize(NC);
    // one collision tet per contact, each with its own candidate range
    for (uint32_t i = 0; i < NC; ++i)
    {
        const V3 base = { grid64(rng, -3, 3), grid64(rng, -3, 3), grid64(rng, -3, 3) };
        const int fam = i % 8;
        bt.family[i] = fam;
        uint32_t rangeLen = 0;
        std::vector<uint32_t> entries;
        auto far_ = [&](uint32_t k, bool degen = false) { return addTet(sc, rng, add(base, V3{ 4.0f + 1.5f * k, 4, 4 }), degen); };
        V3 contact;
        switch (fam)
        {
        case 0: // containing tet in batch 1 (slot 3 of 8)
        {
            for (uint32_t k = 0; k < 3; ++k) entries.push_back(far_(k));
            const uint32_t tgt = addTet(sc, rng, base, false);
            entries.push_back(tgt);
            for (uint32_t k = 3; k < 7; ++k) entries.push_back(far_(k));
            contact = sc.tetCenter[tgt];
            break;
        }
        case 1: // containing tet only in batch 2 (slot 40 of 45)
        {
            for (uint32_t k = 0; k < 40; ++k) entries.push_back(far_(k));
            const uint32_t tgt = addTet(sc, rng, base, false);
            entries.push_back(tgt);
            for (uint32_t k = 40; k < 44; ++k) entries.push_back(far_(k));
            contact = sc.tetCenter[tgt];
            break;
        }
        case 2: // TWO containing tets (exact duplicates): the batch-1 lane must
                // beat the batch-2 one — pins break-after-first-hit-batch
        {
            const uint32_t tgt1 = addTet(sc, rng, base, false);
            // exact coordinate duplicate under a DIFFERENT tet index
            const U4 t = sc.simTets[tgt1];
            const uint32_t v0 = (uint32_t)sc.simVerts.size();
            for (int q = 0; q < 4; ++q) sc.simVerts.push_back(sc.simVerts[(&t.x)[q]]);
            sc.simTets.push_back({ v0, v0 + 1, v0 + 2, v0 + 3 });
            sc.tetCenter.push_back(sc.tetCenter[tgt1]);
            sc.tetValid.push_back(1);
            const uint32_t tgt2 = (uint32_t)sc.simTets.size() - 1;
            for (uint32_t k = 0; k < 5; ++k) entries.push_back(far_(k));
            entries.push_back(tgt1); // batch 1, lane 5
            for (uint32_t k = 5; k < 31; ++k) entries.push_back(far_(k)); // pad batch 1 to exactly 32
            entries.push_back(tgt2); // index 32 = batch 2, lane 0
            contact = sc.tetCenter[tgt1];
            break;
        }
        case 3: // no containment: fallback global min, TRUE min in batch 2
        {
            for (uint32_t k = 0; k < 34; ++k) entries.push_back(far_(k + 2));
            const uint32_t nearT = addTet(sc, rng, add(base, V3{ 2.0f, 0, 0 })); // closest
            entries.push_back(nearT);
            contact = base;
            break;
        }
        case 4: // duplicate-coordinate tets on lanes 4 and 9: lowest lane wins
        {
            for (uint32_t k = 0; k < 4; ++k) entries.push_back(far_(k + 20));
            const uint32_t dupA = addTet(sc, rng, add(base, V3{ 2.0f, 0, 0 }));
            entries.push_back(dupA);
            for (uint32_t k = 0; k < 4; ++k) entries.push_back(far_(k + 30));
            // EXACT duplicate: same vertex coordinates, new tet index
            const U4 t = sc.simTets[dupA];
            const uint32_t v0 = (uint32_t)sc.simVerts.size();
            for (int q = 0; q < 4; ++q) sc.simVerts.push_back(sc.simVerts[(&t.x)[q]]);
            sc.simTets.push_back({ v0, v0 + 1, v0 + 2, v0 + 3 });
            sc.tetCenter.push_back(sc.tetCenter[dupA]);
            sc.tetValid.push_back(1);
            entries.push_back((uint32_t)sc.simTets.size() - 1);
            contact = base;
            break;
        }
        case 5: // empty range (release: garbage OR quirk)
            contact = base;
            break;
        case 6: // fem: degenerate candidates first, valid later in the range
        {
            entries.push_back(addTet(sc, rng, base, true));  // degenerate
            entries.push_back(addTet(sc, rng, base, true));  // degenerate
            const uint32_t tgt = addTet(sc, rng, base, false);
            entries.push_back(tgt);
            contact = sc.tetCenter[tgt];
            break;
        }
        default: // fem: ALL candidates degenerate -> tet0 + zero bary fallback
        {
            entries.push_back(addTet(sc, rng, base, true));
            entries.push_back(addTet(sc, rng, base, true));
            contact = base;
            break;
        }
        }
        rangeLen = (uint32_t)entries.size();
        bt.entriesPerContact.push_back(entries);
        (void)rangeLen;
        bt.point[i] = { contact.x, contact.y, contact.z, 0 };
    }
    // decouple the encoded element index from the contact index (a curPair/field
    // transposition or a decode typo must not be able to hide behind identity)
    bt.colTet.resize(NC);
    bt.sbId.resize(NC);
    std::vector<std::vector<uint32_t>> byCol(NC);
    for (uint32_t i = 0; i < NC; ++i)
    {
        bt.colTet[i] = (i * 7 + 13) % NC;
        bt.sbId[i] = 8 + i % 5;
        byCol[bt.colTet[i]] = bt.entriesPerContact[i];
    }
    for (uint32_t t = 0; t < NC; ++t)
    {
        for (uint32_t e : byCol[t]) sc.remap.push_back(e);
        sc.accum.push_back((uint32_t)sc.remap.size());
    }
    for (uint32_t i = 0; i < NC; ++i)
        bt.info[i] = { 0x9000u + i, 3 + i % 5, (bt.sbId[i] << 20) | bt.colTet[i], 0x40000000u + i };
    return bt;
}

// ============================================================================
PXG_TEST(npSbRemap_other)
{
    const uint32_t NC = 512, PREV = 23;
    Built bt = buildRemapFixtures(NC, 1010701);
    // rmOther has no isValidTet gate: families 6/7 treat degenerate tets as
    // candidates (their barycentrics are inf/nan garbage — mark them skipped
    // from content checks below via the family)
    std::vector<F4> bary1Init(NC, { -5, -5, -5, -5 });
    std::vector<F4> bary0Init(NC, { -6, -6, -6, -6 });

    const SrParams prm = { NC, 0, 1, 0 };
    const RmRun r = runRm("rmOther", prm, NC, bt.point, bary0Init, bary1Init, bt.info,
                          bt.sc, NC + 77 /*count needs clamping*/, PREV, {}, {});

    PXG_CHECK_MSG(r.count == NC, "other: clamp write-back %u != %u", r.count, NC);
    size_t mm = 0, famN[8] = {};
    for (uint32_t i = 0; i < NC; ++i)
    {
        const int fam = bt.family[i];
        if (i < PREV)
        {
            // below prevContactCount: untouched
            if (r.bary1[i].x != -5.0f || r.info[i].z != bt.info[i].z) ++mm;
            continue;
        }
        ++famN[fam];
        // families 6/7 have INVERTED (finite-volume) candidates: rmOther has no
        // isValidTet gate, so they participate deterministically
        const RemapRef ro = refRemap(bt.sc, v3(bt.point[i]), bt.colTet[i], bt.sbId[i], false, true);
        if (fam == 5)
        {
            // empty range: winner lane 0, simTet 0xffffffff, OR-flooded pair;
            // the zero bary is PORT-DEFINED (the CUDA reads an uninitialized
            // register here; the port initializes lane state to zero)
            if (r.info[i].z != ((bt.sbId[i] << 20) | 0xffffffffu)) ++mm;
            if (r.bary1[i].x != 0 || r.bary1[i].w != 0) ++mm;
            continue;
        }
        if (r.info[i].z != ro.newPair) ++mm;
        if (!eqF4(r.bary1[i], ro.bary, 5e-3f)) ++mm;
        if ((r.dbg[i].x == 1.0f) != (ro.kind == 0)) ++mm;
        // untouched fields
        if (r.info[i].x != bt.info[i].x || r.info[i].y != bt.info[i].y || r.info[i].w != bt.info[i].w) ++mm;
        if (r.bary0[i].x != -6.0f) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "other: %zu mismatches", mm);
    // hasPrev == 0: gPrev must be ignored (contact 0 gets remapped)
    const SrParams prm0 = { NC, 0, 0, 0 };
    const RmRun r0 = runRm("rmOther", prm0, NC, bt.point, bary0Init, bary1Init, bt.info,
                           bt.sc, NC, 137 /*must be ignored*/, {}, {});
    PXG_CHECK_MSG(r0.bary1[0].x != -5.0f && r0.bary1[100].x != -5.0f,
                  "other: hasPrev==0 must ignore gPrev");
    PXG_CHECK_MSG(famN[0] > 40 && famN[1] > 40 && famN[2] > 40 && famN[3] > 40 && famN[4] > 40 && famN[5] > 40,
                  "other coverage thin (%zu %zu %zu %zu %zu %zu)",
                  famN[0], famN[1], famN[2], famN[3], famN[4], famN[5]);
}

// ============================================================================
PXG_TEST(npSbRemap_fem)
{
    const uint32_t NC = 512;
    Built bt = buildRemapFixtures(NC, 1010702);
    // fem variant: the CONTACT POINT is stashed in the remapped barycentric
    // array; put it in bary1 for side 0 and bary0 for side 1
    std::vector<F4> baryPt(NC);
    for (uint32_t i = 0; i < NC; ++i) baryPt[i] = bt.point[i];
    std::vector<F4> other(NC, { -6, -6, -6, -6 });

    for (int side = 0; side < 2; ++side)
    {
        // side 1 reads pairInd0 (low dword): encode the col tet there instead
        std::vector<U4> info = bt.info;
        if (side == 1)
            for (uint32_t i = 0; i < NC; ++i)
                info[i] = { (bt.sbId[i] << 20) | bt.colTet[i], 0xEEu /*high dword: truncated away*/, 0x7000u + i, 0x40000000u + i };
        // side 0 additionally exercises the fem count clamp + prev window
        const uint32_t PREVF = side == 0 ? 17 : 0;
        const uint32_t COUNT0 = side == 0 ? NC + 31 : NC;
        const SrParams prm = { NC, (uint32_t)side, 0, 0 };
        const RmRun r = runRm("rmFem", prm, NC,
                              std::vector<F4>(NC, { 0, 0, 0, 0 }),
                              side == 1 ? baryPt : other,
                              side == 0 ? baryPt : other,
                              info, bt.sc, COUNT0, PREVF, {}, {});
        PXG_CHECK_MSG(r.count == NC, "fem s%d: clamp write-back %u", side, r.count);
        size_t mm = 0, invN = 0, fallN = 0;
        for (uint32_t i = 0; i < NC; ++i)
        {
            if (i < PREVF)
            {
                // below prevContactCount: untouched
                if (r.bary1[i].x != baryPt[i].x || r.info[i].z != info[i].z) ++mm;
                continue;
            }
            const RemapRef ro = refRemap(bt.sc, v3(bt.point[i]), bt.colTet[i], bt.sbId[i], true, false);
            const int fam = bt.family[i];
            if (fam == 5)
            {
                // empty range: hasTet ballot 0 -> tet0 + zero bary
                // (range 0 => no lanes; anyTet stays false)
            }
            const F4 gBary = side == 1 ? r.bary0[i] : r.bary1[i];
            const uint32_t gPair = side == 1 ? r.info[i].x : r.info[i].z;
            if (gPair != ro.newPair) ++mm;
            if (!eqF4(gBary, ro.bary, 5e-3f)) ++mm;
            if (side == 1 && r.info[i].y != 0) ++mm;             // u32 -> u64 widen zeroes the high dword
            if (side == 1 && r.info[i].z != 0x7000u + i) ++mm;   // other side untouched
            if (side == 0 && (r.info[i].x != info[i].x || r.info[i].y != info[i].y)) ++mm;
            if (r.info[i].w != info[i].w) ++mm;                  // auxInd must stay untouched
            const F4 gOther = side == 1 ? r.bary1[i] : r.bary0[i];
            if (gOther.x != -6.0f) ++mm;                          // other bary array untouched
            if (ro.kind == 2) ++invN;
            if (ro.kind == 1) ++fallN;
        }
        PXG_CHECK_MSG(mm == 0, "fem s%d: %zu mismatches", side, mm);
        PXG_CHECK_MSG(invN > 80 && fallN > 60, "fem s%d coverage thin (inv=%zu fall=%zu)", side, invN, fallN);
    }
}

// ============================================================================
PXG_TEST(npSbRemap_reorder)
{
    const uint32_t NC = 777;
    std::mt19937 rng(1010703);
    std::vector<F4> pt(NC), np(NC), by(NC);
    std::vector<U4> info(NC);
    std::vector<uint32_t> remap(NC);
    for (uint32_t i = 0; i < NC; ++i)
    {
        pt[i] = { float(i), float(i) + 0.25f, float(i) + 0.5f, float(i) + 0.75f };
        np[i] = { -float(i), 1, 2, 3 };
        by[i] = { 0.25f, 0.25f, 0.25f, 0.25f + float(i) };
        info[i] = { 0xA000u + i, 7 + i % 9, 0xB000u + i, 0xC000u + i };
        remap[i] = i;
    }
    std::shuffle(remap.begin(), remap.end(), rng);

    for (int mode = 0; mode < 2; ++mode)
    {
        const SrParams prm = { NC, (uint32_t)mode, 0, 0 };
        RemapScene empty;
        std::vector<F4> byOther(NC);
        for (uint32_t i = 0; i < NC; ++i) byOther[i] = { -1, -2, -3, -float(i) }; // decoy in bary1
        const RmRun r = runRm("rmReorder", prm, NC, pt, by, byOther, info, empty, NC, 0, remap, np);
        size_t mm = 0;
        for (uint32_t i = 0; i < NC; ++i)
        {
            const uint32_t s = remap[i];
            if (r.sPt[i].x != pt[s].x || r.sPt[i].w != pt[s].w) ++mm;
            if (r.sNp[i].x != np[s].x) ++mm;
            if (r.sBary[i].w != by[s].w) ++mm;
            if (r.sInfo[i].x != info[s].x || r.sInfo[i].y != info[s].y ||
                r.sInfo[i].z != info[s].z || r.sInfo[i].w != info[s].w) ++mm;
            if (mode == 1)
            {
                if (r.rigidIds[i * 2] != info[s].x || r.rigidIds[i * 2 + 1] != info[s].y) ++mm;
            }
            else if (r.rigidIds[i * 2] != POISON) ++mm; // ps flavour must not write rigid ids
        }
        PXG_CHECK_MSG(mm == 0, "reorder m%d: %zu mismatches", mode, mm);
    }
}

} // namespace
