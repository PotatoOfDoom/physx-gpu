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

// Verification of npCvxHfMid.slang against exact CPU transcriptions
// (1/64-grid geometry, signed-permutation quaternions, power-of-two
// scales keep every comparison bit-deterministic; the non-pow2 margin
// products (0.02/0.2/0.25/tolerance*0.05) are single IEEE multiplies
// computed identically on both sides, and fixtures keep decision margins
// robust anyway).  chMidphase families: invalidate-by-position,
// invalidate-by-rotation (exact-quat dots are <= 0.5), CACHED (same
// transform, undrifted contacts -> count == CONVEX_TRIMESH_CACHED, pen
// write-backs verified, no stack writes), refresh-LOST (one drifted
// contact -> mNbManifolds zeroed, full gen), off-grid bound (offsets
// stay 0xFFffFFff in the ConvexMeshPair), zero-area span with cd == 0
// (noTriangles), plus per-shape margin selection (sphere/capsule/convex).
// A single-case run with a tiny maxPairs pins the capped stack writes
// and the PxMax(maxPairs, prevNbPairs) - prevNbPairs counter clamp
// (including the FAITHFUL repetition of payload z after the cap).
// sbShrink pins the (n-1)*cd double-sided shrink and the strict
// intersects comparisons with exact-touch fixtures.

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
struct ChCase
{
    F4 aToBq, aToBp, prevQ, prevP, hfQ, hfP, boundMin, boundMax, shapeData, hfScale, cvxScale;
    U4 ids, counts, nbContacts;
};
struct ChContact { F4 pointA, pointB, nPen; };
struct ChPairOut { F4 aToBq, aToBp; U4 meta, mats; };
struct ChParams { uint32_t numCases, maxPairs, nbRows, nbCols; };

struct V3 { float x, y, z; };
static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 add(V3 a, V3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 mul(V3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 v3(F4 f) { return { f.x, f.y, f.z }; }
static float dot4(F4 a, F4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

static V3 quatRotate(F4 q, V3 v)
{
    const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d = q.x * vx + q.y * vy + q.z * vz;
    return { vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d,
             vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d,
             vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d };
}
static V3 quatRotateInv(F4 q, V3 v)
{
    const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d = q.x * vx + q.y * vy + q.z * vz;
    return { vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * d,
             vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * d,
             vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * d };
}
static F4 exactQuat(std::mt19937& rng)
{
    if (rng() % 4 == 0) return { 0, 0, 0, 1 };
    const uint32_t s = rng() % 16;
    return { (s & 1) ? -0.5f : 0.5f, (s & 2) ? -0.5f : 0.5f, (s & 4) ? -0.5f : 0.5f, (s & 8) ? -0.5f : 0.5f };
}
static float grid64(std::mt19937& rng, float lo, float hi)
{
    const int n = int((hi - lo) * 64.0f);
    return lo + float(int(rng() % uint32_t(n + 1))) / 64.0f;
}

static const uint32_t HF_ROWS = 28, HF_COLS = 20;
static const uint32_t CACHED = 0xffffffffu;
static const uint32_t POISON = 0xCDCDCDCDu;

static float hfHeightRef(const std::vector<uint32_t>& smp, uint32_t v) { return float(int16_t(smp[v] & 0xffffu)); }
static uint32_t hfMat0(const std::vector<uint32_t>& smp, uint32_t v) { return (smp[v] >> 16) & 0x7fu; }
static uint32_t hfMat1(const std::vector<uint32_t>& smp, uint32_t v) { return (smp[v] >> 24) & 0x7fu; }
static uint32_t clampI(int v, int lo, int hi) { return uint32_t(std::min(std::max(v, lo), hi)); }

// full CPU reference for one case; appends expected stack payloads (uncapped run)
struct ChRef
{
    bool fullGen = false, invalidate = false, lost = false;
    uint32_t nbManOut = 0;
    uint32_t nbPairs = 0;
    std::vector<U4> pairs; // payload without the global slot (order preserved)
    std::vector<uint32_t> batchOf; // 32-work-item batch index per pair
    std::vector<float> penOut; // 24 slots (refresh write-backs; NaN = untouched)
};
static ChRef refCh(const ChCase& cs, const std::vector<ChContact>& manifold, uint32_t base,
                   const std::vector<uint32_t>& smp)
{
    ChRef ro;
    ro.penOut.assign(24, std::nanf(""));
    const uint32_t shapeType = uint32_t(cs.shapeData.w);
    float minMargin, ratio, breakingRatio;
    if (shapeType == 0) { minMargin = cs.shapeData.x; ratio = 0.02f; breakingRatio = 0.05f; }
    else if (shapeType == 1) { minMargin = cs.shapeData.y; ratio = 0.02f; breakingRatio = 0.05f; }
    else
    {
        const V3 e = { cs.shapeData.x * cs.cvxScale.x, cs.shapeData.y * cs.cvxScale.y, cs.shapeData.z * cs.cvxScale.z };
        const float mn = std::min(std::min(e.x, e.y), e.z);
        minMargin = std::min(mn * 0.25f, cs.aToBp.w * 0.05f);
        ratio = 0.2f;
        breakingRatio = 0.8f;
    }
    const float thresholdP = minMargin * ratio;
    const V3 d = sub(v3(cs.aToBp), v3(cs.prevP));
    const float deltaP = std::max(std::fabs(d.x), std::max(std::fabs(d.y), std::fabs(d.z)));
    const float deltaQ = dot4(cs.aToBq, cs.prevQ);
    ro.invalidate = (deltaP > thresholdP) || (deltaQ < 0.9998f);

    uint32_t nbMan = cs.counts.x;
    if (!ro.invalidate)
    {
        const float thr = minMargin * breakingRatio;
        const float sqThr = thr * thr;
        for (uint32_t s = 0; s < 24; ++s)
        {
            const uint32_t mi = s / 6;
            if (mi >= nbMan) continue;
            if (s % 6 >= (&cs.nbContacts.x)[mi]) continue;
            const ChContact& c = manifold[base + s];
            const V3 localAInB = add(quatRotate(cs.aToBq, v3(c.pointA)), v3(cs.aToBp));
            const V3 v = sub(localAInB, v3(c.pointB));
            const float penNew = dot3(v, v3(c.nPen));
            const V3 projected = sub(localAInB, mul(v3(c.nPen), c.nPen.w));
            const V3 pd = sub(v3(c.pointB), projected);
            ro.penOut[s] = penNew;
            if (dot3(pd, pd) > sqThr) ro.lost = true;
        }
        if (ro.lost) nbMan = 0;
    }
    ro.nbManOut = nbMan;
    ro.fullGen = ro.invalidate || ro.lost;
    if (!ro.fullGen) return ro;

    const F4 q = cs.hfQ;
    const F4 qinv = { -q.x, -q.y, -q.z, q.w };
    const V3 centerW = mul(add(v3(cs.boundMin), v3(cs.boundMax)), 0.5f);
    const V3 extentW = mul(sub(v3(cs.boundMax), v3(cs.boundMin)), 0.5f);
    const V3 invP = quatRotateInv(q, { -cs.hfP.x, -cs.hfP.y, -cs.hfP.z });
    const V3 center = add(quatRotateInv(q, centerW), invP);
    const float x = qinv.x, y = qinv.y, z = qinv.z, w = qinv.w;
    const float x2 = x + x, y2 = y + y, z2 = z + z;
    const float xx = x2 * x, yy = y2 * y, zz = z2 * z;
    const float xy = x2 * y, xz = x2 * z, xw = x2 * w;
    const float yz = y2 * z, yw = y2 * w, zw = z2 * w;
    const V3 c0 = { 1.0f - yy - zz, xy + zw, xz - yw };
    const V3 c1 = { xy - zw, 1.0f - xx - zz, yz + xw };
    const V3 c2 = { xz + yw, yz - xw, 1.0f - xx - yy };
    const V3 e0 = mul(c0, extentW.x), e1 = mul(c1, extentW.y), e2 = mul(c2, extentW.z);
    const V3 wext = { std::fabs(e0.x) + std::fabs(e1.x) + std::fabs(e2.x),
                      std::fabs(e0.y) + std::fabs(e1.y) + std::fabs(e2.y),
                      std::fabs(e0.z) + std::fabs(e1.z) + std::fabs(e2.z) };
    const float cd = cs.boundMin.w;
    V3 lmn = { center.x - wext.x - cd, center.y - wext.y - cd, center.z - wext.z - cd };
    V3 lmx = { center.x + wext.x + cd, center.y + wext.y + cd, center.z + wext.z + cd };
    lmn = { lmn.x / std::fabs(cs.hfScale.x), lmn.y / cs.hfScale.y, lmn.z / std::fabs(cs.hfScale.z) };
    lmx = { lmx.x / std::fabs(cs.hfScale.x), lmx.y / cs.hfScale.y, lmx.z / std::fabs(cs.hfScale.z) };

    if ((lmn.x > float(HF_ROWS - 1)) || (lmn.z > float(HF_COLS - 1)) || (lmx.x < 0.0f) || (lmx.z < 0.0f))
        return ro;
    const uint32_t minRow = clampI(int(std::floor(lmn.x)), 0, int(HF_ROWS) - 2);
    const uint32_t maxRow = clampI(int(std::ceil(lmx.x)), 0, int(HF_ROWS) - 1);
    const uint32_t minColumn = clampI(int(std::floor(lmn.z)), 0, int(HF_COLS) - 2);
    const uint32_t maxColumn = clampI(int(std::ceil(lmx.z)), 0, int(HF_COLS) - 1);
    if (2u * (maxColumn - minColumn) * (maxRow - minRow) == 0u)
        return ro;
    const float miny = lmn.y, maxy = lmx.y;
    const uint32_t columnSpan = maxColumn - minColumn;
    const uint32_t total = (maxRow - minRow) * columnSpan * 2u;
    for (uint32_t wi = 0; wi < total; ++wi)
    {
        const uint32_t index = wi / 2u;
        const uint32_t vertexIndex = (minRow + index / columnSpan) * HF_COLS + (minColumn + index % columnSpan);
        const float h0 = hfHeightRef(smp, vertexIndex);
        const float h1 = hfHeightRef(smp, vertexIndex + 1u);
        const float h2 = hfHeightRef(smp, vertexIndex + HF_COLS);
        const float h3 = hfHeightRef(smp, vertexIndex + HF_COLS + 1u);
        const bool con0 = maxy < h0 && maxy < h1 && maxy < h2 && maxy < h3;
        const bool con1 = miny > h0 && miny > h1 && miny > h2 && miny > h3;
        if (con0 || con1) continue;
        const bool isM1 = (wi & 1u) != 0u;
        const uint32_t material = isM1 ? hfMat1(smp, vertexIndex) : hfMat0(smp, vertexIndex);
        if (material == 127u) continue;
        const uint32_t tri = isM1 ? ((vertexIndex << 1) + 1u) : (vertexIndex << 1);
        // payload z (nbPairsPerCM + offset) reconstructed for the UNCAPPED run:
        // batches never cross the cap, so it is just the running pair ordinal
        ro.pairs.push_back({ cs.ids.x, tri, ro.nbPairs, cs.ids.y });
        ro.batchOf.push_back(wi / 32u);
        ++ro.nbPairs;
    }
    return ro;
}

struct ChRun
{
    std::vector<U4> stack, pairMeta, pairMats;
    std::vector<F4> pairQ, pairP, dbg;
    std::vector<ChContact> manifold;
    std::vector<uint32_t> nbMan;
    uint32_t nbPairsFound, pairsNum, pairsNumPad;
};
static ChRun runCh(const char* entry, const std::vector<ChCase>& cases,
                   const std::vector<ChContact>& manifold, const std::vector<uint32_t>& samples,
                   const std::vector<F4>& shrink, uint32_t maxPairs, uint32_t stackCap)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t n = (uint32_t)cases.size();
    std::vector<U4> stackInit(stackCap, { POISON, POISON, POISON, POISON });
    std::vector<uint32_t> z1 = { 0 }, z2 = { 0 }, z3 = { 0 };
    std::vector<uint32_t> nbManInit(std::max<uint32_t>(n, 1), POISON);
    std::vector<ChPairOut> pairInit(std::max<uint32_t>(n, 1));
    std::memset(pairInit.data(), 0xCD, pairInit.size() * sizeof(ChPairOut));
    std::vector<F4> dbgInit(std::max<uint32_t>(n, 1), { -7, -7, -7, -7 });
    std::vector<ChContact> manIn = manifold.empty() ? std::vector<ChContact>(1) : manifold;
    std::vector<F4> shrinkIn = shrink.empty() ? std::vector<F4>(4, { 0, 0, 0, 0 }) : shrink;
    std::vector<uint32_t> smpIn = samples.empty() ? std::vector<uint32_t>{ 0 } : samples;

    PxgBufferHandle bC = createBufferFrom(dev, cases);
    PxgBufferHandle bM = createBufferFrom(dev, manIn);
    PxgBufferHandle bS = createBufferFrom(dev, smpIn);
    PxgBufferHandle bSt = createBufferFrom(dev, stackInit);
    PxgBufferHandle bN = createBufferFrom(dev, z1);
    PxgBufferHandle bP = createBufferFrom(dev, z2);
    PxgBufferHandle bPp = createBufferFrom(dev, z3);
    PxgBufferHandle bPo = createBufferFrom(dev, pairInit);
    PxgBufferHandle bNm = createBufferFrom(dev, nbManInit);
    PxgBufferHandle bSh = createBufferFrom(dev, shrinkIn);
    PxgBufferHandle bD = createBufferFrom(dev, dbgInit);
    ChParams prm = { n, maxPairs, HF_ROWS, HF_COLS };
    Kernel k("npCvxHfMid.spv", entry,
             { (uint32_t)sizeof(ChCase), (uint32_t)sizeof(ChContact), 4, 16, 4, 4, 4,
               (uint32_t)sizeof(ChPairOut), 4, 16, 16 },
             sizeof(ChParams));
    k.dispatch((n + 63) / 64, { bC, bM, bS, bSt, bN, bP, bPp, bPo, bNm, bSh, bD }, prm);
    dev->flush();
    ChRun r;
    r.stack = downloadBuffer<U4>(dev, bSt, stackCap);
    const auto po = downloadBuffer<ChPairOut>(dev, bPo, n);
    r.pairQ.resize(n); r.pairP.resize(n); r.pairMeta.resize(n); r.pairMats.resize(n);
    for (uint32_t i = 0; i < n; ++i)
    {
        r.pairQ[i] = po[i].aToBq; r.pairP[i] = po[i].aToBp;
        r.pairMeta[i] = po[i].meta; r.pairMats[i] = po[i].mats;
    }
    r.manifold = downloadBuffer<ChContact>(dev, bM, (uint32_t)manIn.size());
    r.nbMan = downloadBuffer<uint32_t>(dev, bNm, n);
    r.dbg = downloadBuffer<F4>(dev, bD, n);
    r.nbPairsFound = downloadBuffer<uint32_t>(dev, bN, 1)[0];
    r.pairsNum = downloadBuffer<uint32_t>(dev, bP, 1)[0];
    r.pairsNumPad = downloadBuffer<uint32_t>(dev, bPp, 1)[0];
    for (PxgBufferHandle h : { bC, bM, bS, bSt, bN, bP, bPp, bPo, bNm, bSh, bD })
        dev->destroyBuffer(h);
    return r;
}

PXG_TEST(npCvxHfMid_midphase)
{
    const uint32_t NC = 512;
    std::mt19937 rng(1020701);
    std::vector<uint32_t> samples(HF_ROWS * HF_COLS);
    for (uint32_t s = 0; s < samples.size(); ++s)
    {
        const int16_t h = int16_t(int(rng() % 81) - 30);
        uint32_t m0 = (s % 11 == 4) ? 127u : (rng() % 5);
        uint32_t m1 = (s % 13 == 6) ? 127u : (rng() % 5);
        if (rng() % 3 == 0) m0 |= 0x80u;
        if (rng() % 4 == 0) m1 |= 0x80u;
        samples[s] = uint32_t(uint16_t(h)) | (m0 << 16) | (m1 << 24);
    }

    std::vector<ChCase> cases(NC);
    std::vector<ChContact> manifold(NC * 24);
    std::memset(manifold.data(), 0, manifold.size() * sizeof(ChContact));
    size_t invPosN = 0, invRotN = 0, cachedN = 0, lostN = 0, offGridN = 0, zeroSpanN = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        ChCase& cs = cases[i];
        std::memset(&cs, 0, sizeof(cs));
        const uint32_t shapeType = i % 3;
        const float tol = 0.5f * float(1 + i % 3);
        cs.aToBq = exactQuat(rng);
        cs.aToBp = { grid64(rng, -2, 2), grid64(rng, -2, 2), grid64(rng, -2, 2), tol };
        cs.hfQ = exactQuat(rng);
        cs.hfP = { float(int(rng() % 9)) - 4.0f, float(int(rng() % 9)) - 4.0f, float(int(rng() % 9)) - 4.0f, 0 };
        const float sc[3] = { 0.5f, 1.0f, 2.0f };
        cs.hfScale = { sc[rng() % 3], sc[rng() % 3], sc[rng() % 3], 0 };
        cs.cvxScale = { sc[rng() % 3], sc[rng() % 3], sc[rng() % 3], 0 };
        // sphere: radius in x; capsule: radius in y; convex: extents
        cs.shapeData = { 0.5f + 0.25f * float(i % 3), 0.5f + 0.125f * float(i % 4),
                         0.75f + 0.25f * float(i % 2), float(shapeType) };
        if (i % 24 == 7)
        {
            // margin-window probe: capsule with radius-x 1.0 / radius-y 0.5 and a
            // position delta of 0.015625 — ABOVE the correct capsule threshold
            // (0.5*0.02) but BELOW the sphere-slot one (1.0*0.02): a port taking
            // the margin from scale.x stays cached instead of invalidating
            cs.shapeData = { 1.0f, 0.5f, 0.75f, 1.0f };
        }
        cs.ids = { i, 0x600u + i, 0x11u + i % 7, 0x22u + i % 5 };
        cs.counts = { 2, 0, 0, 0 }; // two sub-manifolds
        cs.nbContacts = { 3, 2, 0, 0 };

        // cached manifold contacts consistent with aToB (undrifted by default):
        // choose pointA on the grid; pointB = aToB.transform(pointA) => drift 0
        const uint32_t base = i * 24;
        for (uint32_t s = 0; s < 24; ++s)
        {
            const V3 pA = { grid64(rng, -1, 1), grid64(rng, -1, 1), grid64(rng, -1, 1) };
            const V3 img = add(quatRotate(cs.aToBq, pA), v3(cs.aToBp));
            // pointB = localAInB - n*penOld + n*t: 2D drift is exactly 0 while
            // penNew = penOld - t != penOld, pinning the refresh write-back
            const float penOld = 0.25f, t = 0.015625f;
            V3 pB = { img.x, img.y, img.z - penOld + t };
            const bool active = (s / 6 < 2) && (s % 6 < (s / 6 == 0 ? 3u : 2u));
            // invalidate families get DRIFTED contacts: a mutant that runs the
            // refresh despite the invalidation loses the manifolds
            if (i % 8 == 0 || i % 8 == 1) pB.x += 3.0f;
            manifold[base + s].pointA = { pA.x, pA.y, pA.z, 0 };
            manifold[base + s].pointB = { pB.x, pB.y, pB.z, 0 };
            manifold[base + s].nPen = { 0, 0, 1, active ? penOld : 0.5f };
        }

        // heightfield-space target region -> world bound through the exact chain
        const V3 lo = { grid64(rng, 1, 20), grid64(rng, -10, 10), grid64(rng, 1, 13) };
        const V3 hi = add(lo, V3{ grid64(rng, 1, 4), grid64(rng, 1, 8), grid64(rng, 1, 4) });
        float cd = 0.25f;
        V3 loHf = lo, hiHf = hi;
        const uint32_t fam = i % 8;
        switch (fam)
        {
        case 0: // invalidate by position: prev far away
            cs.prevQ = cs.aToBq;
            cs.prevP = { cs.aToBp.x + 4.0f, cs.aToBp.y, cs.aToBp.z, 0 };
            ++invPosN;
            break;
        case 1: // invalidate by rotation: different exact quat (dot <= 0.5)
        {
            F4 pq = exactQuat(rng);
            while (std::fabs(dot4(pq, cs.aToBq)) > 0.9f) pq = exactQuat(rng);
            cs.prevQ = pq;
            cs.prevP = { cs.aToBp.x, cs.aToBp.y, cs.aToBp.z, 0 };
            ++invRotN;
            break;
        }
        case 2: // CACHED: identical transform, undrifted contacts
        case 5:
            cs.prevQ = cs.aToBq;
            cs.prevP = { cs.aToBp.x, cs.aToBp.y, cs.aToBp.z, 0 };
            ++cachedN;
            break;
        case 3: // refresh-LOST: identical transform but one contact drifted
            cs.prevQ = cs.aToBq;
            cs.prevP = { cs.aToBp.x, cs.aToBp.y, cs.aToBp.z, 0 };
            manifold[base + 1].pointB.x += 5.0f; // way past any breaking threshold
            ++lostN;
            break;
        case 4: // off-grid bound (full gen but boundsDontOverlap)
            cs.prevQ = cs.aToBq;
            cs.prevP = { cs.aToBp.x + 4.0f, cs.aToBp.y, cs.aToBp.z, 0 };
            loHf = { float(HF_ROWS) + 4.0f, lo.y, lo.z };
            hiHf = { float(HF_ROWS) + 6.0f, hi.y, hi.z };
            ++offGridN;
            break;
        case 6: // zero row span: cd == 0 and a FLAT x bound exactly on a grid line
            cs.prevQ = cs.aToBq;
            cs.prevP = { cs.aToBp.x + 4.0f, cs.aToBp.y, cs.aToBp.z, 0 };
            cd = 0.0f;
            loHf = { float(3 + int(i / 8) % 20), lo.y, lo.z };
            hiHf = { loHf.x, hi.y, hi.z };
            ++zeroSpanN;
            break;
        default: // full gen via position invalidate, generic region
            cs.prevQ = cs.aToBq;
            cs.prevP = { cs.aToBp.x, cs.aToBp.y + 3.0f, cs.aToBp.z, 0 };
            break;
        }
        if (i % 24 == 7)
        {
            // margin-window probe overrides the family transform: tiny delta
            cs.prevQ = cs.aToBq;
            cs.prevP = { cs.aToBp.x + 0.015625f, cs.aToBp.y, cs.aToBp.z, 0 };
        }
        // world bound = hfT.transform(scale * hfSpaceBound) for the exact chain:
        // build the world box whose transformFast image is EXACTLY [loHf, hiHf]
        // (signed-permutation rotation maps boxes to boxes)
        const V3 sLo = { loHf.x * cs.hfScale.x, loHf.y * cs.hfScale.y, loHf.z * cs.hfScale.z };
        const V3 sHi = { hiHf.x * cs.hfScale.x, hiHf.y * cs.hfScale.y, hiHf.z * cs.hfScale.z };
        const V3 wA = add(quatRotate(cs.hfQ, sLo), v3(cs.hfP));
        const V3 wB = add(quatRotate(cs.hfQ, sHi), v3(cs.hfP));
        const V3 wLo = { std::min(wA.x, wB.x), std::min(wA.y, wB.y), std::min(wA.z, wB.z) };
        const V3 wHi = { std::max(wA.x, wB.x), std::max(wA.y, wB.y), std::max(wA.z, wB.z) };
        cs.boundMin = { wLo.x, wLo.y, wLo.z, cd };
        cs.boundMax = { wHi.x, wHi.y, wHi.z, 0 };
    }

    // reference (uncapped)
    std::vector<ChRef> ref(NC);
    uint32_t totalPairs = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        ref[i] = refCh(cases[i], manifold, i * 24, samples);
        totalPairs += ref[i].nbPairs;
    }
    PXG_CHECK_MSG(totalPairs > 800, "cvxHfMid: scene too sparse (%u)", totalPairs);

    const uint32_t CAP = totalPairs + 64;
    const ChRun r = runCh("chMidphase", cases, manifold, samples, {}, CAP, CAP);
    PXG_CHECK_MSG(r.nbPairsFound == totalPairs && r.pairsNum == totalPairs,
                  "cvxHfMid: counters %u/%u != %u", r.nbPairsFound, r.pairsNum, totalPairs);

    // group stack rows by cmIdx (payload x)
    std::vector<std::vector<U4>> byCase(NC);
    std::vector<std::vector<uint32_t>> slotsOf(NC);
    size_t mm = 0;
    for (uint32_t s = 0; s < std::min(r.nbPairsFound, CAP); ++s)
    {
        if (r.stack[s].x >= NC) { ++mm; continue; }
        byCase[r.stack[s].x].push_back(r.stack[s]);
        slotsOf[r.stack[s].x].push_back(s);
    }
    uint32_t padSum = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        const ChRef& ro = ref[i];
        if ((r.dbg[i].x != 0.0f) != ro.fullGen) ++mm;
        if (r.nbMan[i] != ro.nbManOut) ++mm;
        // per-case pair payloads in order (batch bases are contiguous per case)
        if (byCase[i].size() != ro.pairs.size()) { ++mm; continue; }
        for (size_t j = 0; j < ro.pairs.size(); ++j)
        {
            const U4& g = byCase[i][j];
            const U4& e = ro.pairs[j];
            if (g.y != e.y || g.z != e.z || g.w != e.w) ++mm;
            // contiguity holds only WITHIN one 32-work-item batch (batches of
            // different cases interleave through the global atomic); batch k of
            // a case still lands before batch k+1 (sequential reservations)
            if (j > 0 && ro.batchOf[j] == ro.batchOf[j - 1] &&
                slotsOf[i][j] != slotsOf[i][j - 1] + 1) ++mm;
            if (j > 0 && slotsOf[i][j] <= slotsOf[i][j - 1]) ++mm; // order across batches
        }
        // ConvexMeshPair record
        const uint32_t expCount = ro.fullGen ? ro.nbPairs : CACHED;
        if (r.pairMeta[i].x != i || r.pairMeta[i].z != expCount) ++mm;
        if (ro.nbPairs == 0)
        {
            if (r.pairMeta[i].y != 0xffffffffu || r.pairMeta[i].w != 0xffffffffu) ++mm;
        }
        else
        {
            if (r.pairMeta[i].y >= totalPairs) ++mm;
            padSum += ((ro.nbPairs + 3u) & ~3u) * 2u;
        }
        if (r.pairMats[i].x != cases[i].ids.z || r.pairMats[i].y != cases[i].ids.w) ++mm;
        if (r.pairQ[i].x != cases[i].aToBq.x || r.pairQ[i].y != cases[i].aToBq.y ||
            r.pairQ[i].z != cases[i].aToBq.z || r.pairQ[i].w != cases[i].aToBq.w ||
            r.pairP[i].x != cases[i].aToBp.x || r.pairP[i].y != cases[i].aToBp.y ||
            r.pairP[i].z != cases[i].aToBp.z) ++mm;
        // refresh pen write-backs (exact)
        for (uint32_t s = 0; s < 24; ++s)
        {
            const float got = r.manifold[i * 24 + s].nPen.w;
            const bool active = (s / 6 < 2) && (s % 6 < (s / 6 == 0 ? 3u : 2u));
            if (std::isnan(ref[i].penOut[s]))
            {
                if (got != (active ? 0.25f : 0.5f)) ++mm; // untouched slots keep the seed
            }
            else if (got != ref[i].penOut[s]) ++mm;
        }
    }
    PXG_CHECK_MSG(r.pairsNumPad == padSum, "cvxHfMid: padded counter %u != %u", r.pairsNumPad, padSum);
    // startIndex / roundedStartIndex partition checks: the per-case ranges must
    // be disjoint and exactly cover [0, totalPairs) resp. [0, padSum)
    {
        std::vector<std::pair<uint32_t, uint32_t>> rangesY, rangesW;
        for (uint32_t i = 0; i < NC; ++i)
        {
            if (ref[i].nbPairs == 0) continue;
            rangesY.push_back({ r.pairMeta[i].y, ref[i].nbPairs });
            rangesW.push_back({ r.pairMeta[i].w, ((ref[i].nbPairs + 3u) & ~3u) * 2u });
        }
        auto checkPartition = [&](std::vector<std::pair<uint32_t, uint32_t>>& v, uint32_t total, const char* what)
        {
            std::sort(v.begin(), v.end());
            uint32_t cursor = 0;
            size_t bad = 0;
            for (const auto& pr : v)
            {
                if (pr.first != cursor) ++bad;
                cursor = pr.first + pr.second;
            }
            if (cursor != total) ++bad;
            PXG_CHECK_MSG(bad == 0, "cvxHfMid: %s ranges do not partition [0, %u)", what, total);
        };
        checkPartition(rangesY, totalPairs, "startIndex");
        checkPartition(rangesW, padSum, "roundedStartIndex");
    }
    // no writes beyond the reserved region in the uncapped run
    {
        size_t dirty = 0;
        for (uint32_t sIdx = totalPairs; sIdx < CAP; ++sIdx)
            if (r.stack[sIdx].x != POISON) ++dirty;
        PXG_CHECK_MSG(dirty == 0, "cvxHfMid: %zu dirty slots beyond totalPairs", dirty);
    }
    PXG_CHECK_MSG(mm == 0, "cvxHfMid: %zu mismatches", mm);
    PXG_CHECK_MSG(invPosN > 50 && invRotN > 50 && cachedN > 100 && lostN > 50 && offGridN > 50 && zeroSpanN > 50,
                  "cvxHfMid coverage thin (%zu %zu %zu %zu %zu %zu)",
                  invPosN, invRotN, cachedN, lostN, offGridN, zeroSpanN);

    // ---- capped single-case run: pins the capped writes + counter clamp ----
    uint32_t big = 0;
    for (uint32_t i = 0; i < NC; ++i)
        if (ref[i].nbPairs > ref[big].nbPairs) big = i;
    const uint32_t bigPairs = ref[big].nbPairs;
    PXG_CHECK_MSG(bigPairs > 40, "cvxHfMid: no big case (%u)", bigPairs);
    const uint32_t TINY = bigPairs / 2;
    std::vector<ChCase> one = { cases[big] };
    one[0].ids.x = 0;
    std::vector<ChContact> oneMan(manifold.begin() + big * 24, manifold.begin() + big * 24 + 24);
    const ChRun rc = runCh("chMidphase", one, oneMan, samples, {}, TINY, bigPairs + 8);
    // counter keeps counting (reservations ungated); stores stop at TINY
    PXG_CHECK_MSG(rc.nbPairsFound == bigPairs, "cvxHfMid cap: found %u != %u", rc.nbPairsFound, bigPairs);
    size_t bad = 0;
    for (uint32_t s = 0; s < TINY; ++s)
    {
        const ChRef& ro = ref[big];
        if (rc.stack[s].x != 0 || rc.stack[s].y != ro.pairs[s].y ||
            rc.stack[s].z != ro.pairs[s].z || rc.stack[s].w != ro.pairs[s].w) ++bad;
    }
    if (rc.pairMeta[0].y != 0 || rc.pairMeta[0].w != 0) ++bad; // single case: offsets are 0
    for (uint32_t s = TINY; s < bigPairs + 8; ++s)
        if (rc.stack[s].x != POISON) ++bad;
    // per-CM counter clamps at the cap: nbPairsPerCM == TINY (batch-crossing clamp)
    // and the padded reservation uses the clamped value
    PXG_CHECK_MSG(rc.pairMeta[0].z == TINY && rc.pairsNum == TINY &&
                  rc.pairsNumPad == ((TINY + 3u) & ~3u) * 2u,
                  "cvxHfMid cap: clamped count %u/%u/%u (want %u)",
                  rc.pairMeta[0].z, rc.pairsNum, rc.pairsNumPad, TINY);
    PXG_CHECK_MSG(bad == 0, "cvxHfMid cap: %zu bad slots", bad);
}

PXG_TEST(npCvxHfMid_sbShrink)
{
    const uint32_t NS = 256;
    std::mt19937 rng(1020702);
    std::vector<F4> shrink(NS * 4);
    std::vector<char> expectKeep(NS);
    size_t nOneN = 0, touchN = 0, dropN = 0;
    for (uint32_t i = 0; i < NS; ++i)
    {
        const float cd = 0.25f + 0.25f * float(i % 3);
        const float n = (i % 4 == 0) ? 1.0f : (i % 4 == 1) ? 2.0f : (i % 4 == 2) ? 3.5f : 2.5f;
        const V3 cLo = { grid64(rng, -4, 2), grid64(rng, -4, 2), grid64(rng, -4, 2) };
        const V3 cHi = add(cLo, V3{ 2.0f + grid64(rng, 0, 2), 2.0f + grid64(rng, 0, 2), 2.0f + grid64(rng, 0, 2) });
        V3 sLo, sHi;
        const float sh = (n - 1.0f) * cd;
        bool keep;
        switch (i % 5)
        {
        case 0: // clearly overlapping even after shrink
            sLo = add(cLo, V3{ 0.5f, 0.5f, 0.5f });
            sHi = add(sLo, V3{ 1, 1, 1 });
            keep = true;
            break;
        case 1: // exact touch AFTER shrink: shrunkMax.x == sbMin.x (strict > keeps it)
            sLo = { cHi.x - sh, cLo.y, cLo.z };
            sHi = add(sLo, V3{ 1, 1, 1 });
            keep = true;
            ++touchN;
            break;
        case 2: // separated only BECAUSE of the shrink (would overlap unshrunk)
            sLo = { cHi.x - sh + 0.125f, cLo.y, cLo.z };
            sHi = add(sLo, V3{ 1, 1, 1 });
            keep = false; // provisional; the recompute block below is the source of truth
            break;
        case 3: // far separated
            sLo = add(cHi, V3{ 3, 3, 3 });
            sHi = add(sLo, V3{ 1, 1, 1 });
            keep = false;
            ++dropN;
            break;
        default: // n == 1 cases keep everything that overlaps the raw bounds
            sLo = add(cLo, V3{ -0.5f, -0.5f, -0.5f });
            sHi = add(cLo, V3{ 0.5f, 0.5f, 0.5f });
            keep = true;
            break;
        }
        // recompute keep exactly per the ported predicate (source of truth)
        {
            V3 mn = cLo, mx = cHi;
            if (n > 1.0f)
            {
                mn = add(cLo, V3{ sh, sh, sh });
                mx = sub(cHi, V3{ sh, sh, sh });
            }
            keep = !(sLo.x > mx.x || mn.x > sHi.x || sLo.y > mx.y || mn.y > sHi.y ||
                     sLo.z > mx.z || mn.z > sHi.z);
            if (!(n > 1.0f)) keep = true; // upstream: no test at all when n <= 1
        }
        if (n == 1.0f) ++nOneN;
        expectKeep[i] = keep ? 1 : 0;
        shrink[i * 4 + 0] = { cLo.x, cLo.y, cLo.z, cd };
        shrink[i * 4 + 1] = { cHi.x, cHi.y, cHi.z, n };
        shrink[i * 4 + 2] = { sLo.x, sLo.y, sLo.z, 0 };
        shrink[i * 4 + 3] = { sHi.x, sHi.y, sHi.z, 0 };
    }

    std::vector<ChCase> dummyCases(NS);
    std::memset(dummyCases.data(), 0, dummyCases.size() * sizeof(ChCase));
    for (uint32_t i = 0; i < NS; ++i) dummyCases[i].ids.x = i;
    const ChRun r = runCh("sbShrink", dummyCases, {}, {}, shrink, 16, 16);
    size_t mm = 0;
    for (uint32_t i = 0; i < NS; ++i)
        if ((r.dbg[i].x != 0.0f) != (expectKeep[i] != 0)) ++mm;
    PXG_CHECK_MSG(mm == 0, "sbShrink: %zu mismatches", mm);
    PXG_CHECK_MSG(nOneN > 40 && touchN > 30 && dropN > 30,
                  "sbShrink coverage thin (%zu %zu %zu)", nOneN, touchN, dropN);
}

} // namespace
