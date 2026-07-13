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

// Verification of npCvxPlaneGen.slang: the convex-plane contact generation --
// face selection (155a semantics with -n), the vertex-emission loop with the
// LOOP-INVARIANT polyDesc read (cu L2966 uses bestPlaneBIdx in both
// iterations - the second face re-emits the first face's vertices), the
// sparse-mask >16 reduce, the shared compaction, the dense >4 reduce and the
// manifold/moreData write (nothing written when zero contacts).  Reference
// transcribes the CUDA; counts/masks exact, floats to tolerance, selector/
// ballot/reduce near-ties margin-skipped.

#include "pxgTestFramework.h"

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
struct V3 { float x, y, z; };
struct GenIn { F4 planes[64], verts[64]; uint32_t polyDesc[64]; U4 vtxData[32], fbe[12]; F4 scale, rot, n, bToAq, bToAp, planeTq, convexTq; };
struct GenOut { F4 mNp[4], mA[4], mB[4]; U4 md0, md1, md2; F4 p[32]; U4 result, aux; };

const float FLTMAX = 3.4028235e38f;
const float TIE = 2e-4f;

static inline V3 v3(const F4& f) { return V3{ f.x, f.y, f.z }; }
static inline V3 add(const V3& a, const V3& b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static inline V3 sub(const V3& a, const V3& b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static inline V3 mul(const V3& a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; }
static inline float dot3(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 crossp(const V3& a, const V3& b) { return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static inline V3 unit(const V3& a) { float l = std::sqrt(dot3(a, a)); return l > 0 ? mul(a, 1.f / l) : a; }
static V3 qRot(const F4& q, const V3& v) { const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z, w2 = q.w * q.w - 0.5f, d2 = q.x * vx + q.y * vy + q.z * vz; return V3{ vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d2, vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d2, vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d2 }; }
static V3 qRotInv(const F4& q, const V3& v) { const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z, w2 = q.w * q.w - 0.5f, d2 = q.x * vx + q.y * vy + q.z * vz; return V3{ vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * d2, vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * d2, vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * d2 }; }
static V3 getNormalized(const V3& v) { const float m = dot3(v, v); return m > 0 ? mul(v, 1.f / std::sqrt(m)) : V3{ 0, 0, 0 }; }
static V3 shape2Vertex(const V3& v, const V3& sc, const F4& rot)
{
    const V3 r = qRot(rot, v);
    return qRotInv(rot, V3{ r.x / sc.x, r.y / sc.y, r.z / sc.z });
}
static V3 vertex2Shape(const V3& v, const V3& sc, const F4& rot)
{
    const V3 r = qRot(rot, v);
    return qRotInv(rot, V3{ r.x * sc.x, r.y * sc.y, r.z * sc.z });
}
static V3 loadNormal(const F4& p, const V3& sc, const F4& rot) { return getNormalized(shape2Vertex(v3(p), sc, rot)); }
static uint32_t popc(uint32_t m) { uint32_t c = 0; while (m) { c += m & 1u; m >>= 1; } return c; }

static uint32_t byteAt(const U4* w16, uint32_t b)
{
    const uint32_t* w = &w16[b / 16].x;
    return (w[(b / 4) & 3] >> (8 * (b & 3))) & 0xFF;
}

static uint32_t minIdxM(const float v[32], uint32_t mask, bool& marg)
{
    float mn = FLTMAX;
    for (int L = 0; L < 32; ++L) if (mask & (1u << L)) mn = std::fmin(mn, v[L]);
    uint32_t win = 0; for (int L = 0; L < 32; ++L) if ((mask & (1u << L)) && v[L] == mn) { win = (uint32_t)L; break; }
    for (int L = 0; L < 32; ++L) { const float d = std::fabs(v[L] - mn); if ((mask & (1u << L)) && d != 0.f && d < TIE) marg = true; }
    return win;
}
static uint32_t maxIdxM(const float v[32], uint32_t mask, bool& marg)
{
    float mx = -FLTMAX;
    for (int L = 0; L < 32; ++L) if (mask & (1u << L)) mx = std::fmax(mx, v[L]);
    uint32_t win = 0; for (int L = 0; L < 32; ++L) if ((mask & (1u << L)) && v[L] == mx) { win = (uint32_t)L; break; }
    for (int L = 0; L < 32; ++L) { const float d = std::fabs(mx - v[L]); if ((mask & (1u << L)) && d != 0.f && d < TIE) marg = true; }
    return win;
}
static uint32_t reduce4(const V3 pt[32], const float sep[32], const V3& n, uint32_t allMask, bool& marg)
{
    const uint32_t i0 = minIdxM(sep, allMask, marg);
    uint32_t mask = (1u << i0);
    float d1[32];
    for (int L = 0; L < 32; ++L) { const V3 df = sub(pt[i0], pt[L]); d1[L] = dot3(df, df); }
    const uint32_t i1 = maxIdxM(d1, allMask & ~mask, marg);
    mask |= (1u << i1);
    const V3 dir = crossp(n, sub(pt[i1], pt[i0]));
    float d[32];
    for (int L = 0; L < 32; ++L) d[L] = dot3(dir, sub(pt[L], pt[i0]));
    const uint32_t f = maxIdxM(d, allMask & ~mask, marg);
    mask |= (1u << f);
    uint32_t g = minIdxM(d, allMask & ~mask, marg);
    if (std::fabs(d[f] * d[g]) < TIE) marg = true;
    if (d[f] * d[g] > 0.f) g = maxIdxM(d, allMask & ~mask, marg);
    mask |= (1u << g);
    return mask;
}

static void refPlaneGen(const GenIn& s, GenOut& o, bool& marg)
{
    const uint32_t nbP = (uint32_t)s.scale.w;
    const uint32_t nbE = (uint32_t)s.n.w;
    const V3 sc = v3(s.scale);
    const F4 rot = s.rot;
    const V3 n = v3(s.n);
    const F4 bToAq = s.bToAq;
    const V3 bToAp = v3(s.bToAp);
    const float contactDist = s.bToAp.w;
    const V3 localNormal{ 1, 0, 0 };

    // selector with -n (155a semantics; symmetric near-tie skips)
    const V3 selN = mul(n, -1.f);
    float laneBest[32]; uint32_t laneIdx[32];
    for (int L = 0; L < 32; ++L) { laneBest[L] = -FLTMAX; laneIdx[L] = 0; }
    for (uint32_t i = 0; i < nbP; ++i)
    {
        const int L = i & 31;
        const float proj = dot3(getNormalized(shape2Vertex(v3(s.planes[i]), sc, rot)), selN);
        const float dl = std::fabs(proj - laneBest[L]);
        if (laneBest[L] > -FLTMAX && dl != 0.f && dl < TIE) marg = true;
        if (laneBest[L] < proj) { laneIdx[L] = i; laneBest[L] = proj; }
    }
    float bestProj = -FLTMAX;
    for (int L = 0; L < 32; ++L) bestProj = std::fmax(bestProj, laneBest[L]);
    uint32_t bestLane = 0;
    for (int L = 0; L < 32; ++L) if (laneBest[L] == bestProj) { bestLane = (uint32_t)L; break; }
    for (int L = 0; L < 32; ++L)
        { const float dl = std::fabs(laneBest[L] - bestProj); if ((uint32_t)L != bestLane && laneBest[L] > -FLTMAX && dl != 0.f && dl < TIE) marg = true; }
    uint32_t bestPlaneBIdx = laneIdx[bestLane];

    float laneE[32]; uint32_t laneEIdx[32];
    for (int L = 0; L < 32; ++L) { laneE[L] = -FLTMAX; laneEIdx[L] = 0xFFFFFFFFu; }
    for (uint32_t i = 0; i < nbE; ++i)
    {
        const int L = i & 31;
        const uint32_t f0 = byteAt(s.fbe, 2 * i), f1 = byteAt(s.fbe, 2 * i + 1);
        const float dp = dot3(getNormalized(add(loadNormal(s.planes[f0], sc, rot), loadNormal(s.planes[f1], sc, rot))), selN);
        const float de = std::fabs(dp - laneE[L]);
        if (laneE[L] > -FLTMAX && de != 0.f && de < TIE) marg = true;
        if (dp > laneE[L]) { laneE[L] = dp; laneEIdx[L] = i; }
    }
    float bestEMax = -FLTMAX;
    for (int L = 0; L < 32; ++L) bestEMax = std::fmax(bestEMax, laneE[L]);
    uint32_t bestELane = 0;
    for (int L = 0; L < 32; ++L) if (laneE[L] == bestEMax) { bestELane = (uint32_t)L; break; }
    for (int L = 0; L < 32; ++L)
        { const float de = std::fabs(laneE[L] - bestEMax); if ((uint32_t)L != bestELane && laneE[L] > -FLTMAX && de != 0.f && de < TIE) marg = true; }
    const float bestEdge = bestEMax + 5e-4f;
    const uint32_t closestEdge = laneEIdx[bestELane];
    if (nbE > 0 && std::fabs(bestEdge - bestProj) < TIE) marg = true;

    uint32_t plane2 = 0xFFFFFFFFu;
    if (bestEdge > bestProj)
    {
        const uint32_t f0 = byteAt(s.fbe, closestEdge * 2), f1 = byteAt(s.fbe, closestEdge * 2 + 1);
        const float dp0 = dot3(loadNormal(s.planes[f0], sc, rot), selN);
        const float dp1 = dot3(loadNormal(s.planes[f1], sc, rot), selN);
        if (std::fabs(dp0 - dp1) < TIE) marg = true;
        if (dp0 > dp1) { bestPlaneBIdx = f0; plane2 = f1; }
        else           { bestPlaneBIdx = f1; plane2 = f0; }
    }
    const uint32_t selPlane2 = plane2;

    // emission loop (polyDesc read is LOOP-INVARIANT: bestPlaneBIdx)
    V3 shPos[32]; float shSep[32];
    uint32_t total = 0;
    for (uint32_t planeIndex = bestPlaneBIdx; planeIndex != 0xFFFFFFFFu;)
    {
        V3 pips[32]; float sd[32];
        const uint32_t polyDesc = s.polyDesc[bestPlaneBIdx];
        const uint32_t nbV = (polyDesc & 0xFFFF) >> 8, vRef8 = polyDesc >> 16;
        for (int L = 0; L < 32; ++L)
        {
            pips[L] = V3{ 0, 0, 0 }; sd[L] = FLTMAX;
            if ((uint32_t)L < nbV)
            {
                const uint32_t index = byteAt(s.vtxData, vRef8 + L);
                const V3 pss = vertex2Shape(v3(s.verts[index]), sc, rot);
                pips[L] = add(qRot(bToAq, pss), bToAp);
                sd[L] = pips[L].x;
            }
        }
        uint32_t flags = 0;
        for (int L = 0; L < 32; ++L)
        {
            if (sd[L] != FLTMAX && std::fabs(sd[L] - contactDist) < TIE) marg = true;
            if (sd[L] < contactDist) flags |= (1u << L);
        }
        uint32_t nbC = popc(flags);
        if (nbC)
        {
            if (nbC > 16)
            {
                flags = reduce4(pips, sd, localNormal, flags, marg);
                nbC = popc(flags);
            }
            for (int L = 0; L < 32; ++L)
                if (flags & (1u << L))
                {
                    const uint32_t idx = popc(flags & ((1u << L) - 1));
                    shPos[total + idx] = pips[L];
                    shSep[total + idx] = sd[L];
                }
            total += nbC;
        }
        planeIndex = plane2;
        plane2 = 0xFFFFFFFFu;
    }

    uint32_t nbContacts = total;
    for (int j = 0; j < 4; ++j) { o.mNp[j] = F4{ 0, 0, 0, 0 }; o.mA[j] = F4{ 0, 0, 0, 0 }; o.mB[j] = F4{ 0, 0, 0, 0 }; }
    o.md0 = U4{ 0, 0, 0, 0 }; o.md1 = U4{ 0, 0, 0, 0 }; o.md2 = U4{ 0, 0, 0, 0 };
    V3 pL[32]; float sepL[32];
    for (int L = 0; L < 32; ++L) { pL[L] = V3{ 0, 0, 0 }; sepL[L] = 0.f; }
    uint32_t mask = 0, wrote = 0;

    if (nbContacts)
    {
        mask = (1u << total) - 1;
        for (uint32_t L = 0; L < 32 && L < nbContacts; ++L) { pL[L] = shPos[L]; sepL[L] = shSep[L]; }
        if (nbContacts > 4)
        {
            mask = reduce4(pL, sepL, localNormal, mask, marg);
            nbContacts = popc(mask);
        }
        for (int L = 0; L < 32; ++L)
            if (mask & (1u << L))
            {
                const uint32_t off = popc(mask & ((1u << L) - 1));
                const V3 p = pL[L]; const float sep = sepL[L];
                const V3 pInB = qRotInv(bToAq, sub(p, bToAp));
                V3 pOnPlane = p; pOnPlane.x -= sep;
                o.mB[off] = F4{ pInB.x, pInB.y, pInB.z, 0 };
                o.mA[off] = F4{ pOnPlane.x, pOnPlane.y, pOnPlane.z, 0 };
                o.mNp[off] = F4{ n.x, n.y, n.z, sep };
            }
        uint32_t b[3];
        const V3 negP = mul(bToAp, -1.f);
        std::memcpy(&b[0], &negP.x, 4); std::memcpy(&b[1], &negP.y, 4); std::memcpy(&b[2], &negP.z, 4);
        o.md0 = U4{ b[0], b[1], b[2], 0 };
        uint32_t q[4];
        std::memcpy(q, &s.planeTq, 16); o.md1 = U4{ q[0], q[1], q[2], q[3] };
        std::memcpy(q, &s.convexTq, 16); o.md2 = U4{ q[0], q[1], q[2], q[3] };
        wrote = 1;
    }

    for (int L = 0; L < 32; ++L) o.p[L] = F4{ pL[L].x, pL[L].y, pL[L].z, sepL[L] };
    o.result = U4{ nbContacts, mask, total, wrote };
    o.aux = U4{ bestPlaneBIdx, selPlane2, wrote, 0 };
}

static bool eqF4(const F4& a, const F4& b) { return almostEqual(a.x, b.x, 2e-4f, 2e-5f) && almostEqual(a.y, b.y, 2e-4f, 2e-5f) && almostEqual(a.z, b.z, 2e-4f, 2e-5f) && almostEqual(a.w, b.w, 2e-4f, 2e-4f); }

PXG_TEST(npCvxPlaneGen_gen)
{
    std::mt19937 rng(155201);
    const uint32_t N = 8000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto randQuat = [&]() { V3 ax = unit(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) }); const float a = rf(-3.1f, 3.1f); return F4{ ax.x * std::sin(a * .5f), ax.y * std::sin(a * .5f), ax.z * std::sin(a * .5f), std::cos(a * .5f) }; };

    std::vector<GenIn> in(N);
    std::vector<uint8_t> skip(N, 0);
    std::vector<GenOut> ref(N);
    size_t nSkip = 0, cEdgePath = 0, cZero = 0, cRed = 0, cBig = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        GenIn s{};
        const uint32_t nbP = 4 + rng() % 12;      // 4..15 polygons
        const uint32_t nbE = 3 + rng() % 20;      // 3..22 edges
        // vertices
        const uint32_t nbVtx = 8 + rng() % 56;    // 8..63
        for (uint32_t v = 0; v < 64; ++v)
            s.verts[v] = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), 0 };
        // polygons: descriptor packs vRef8 (byte offset into vtxData) and vert count
        // start deep into the byte table every few tests so vRef8 exceeds 255
        // (it is a full PxU16 in the blob -- pins the shift/mask extraction)
        uint32_t cursor = (i % 5 == 0) ? 260 + rng() % 100 : 0;
        for (uint32_t p = 0; p < nbP && cursor < 500; ++p)
        {
            uint32_t nv = 3 + rng() % 18;         // up to 20 verts (drives the >16 reduce)
            if (i % 6 != 0) nv = 3 + rng() % 6;   // mostly small faces
            if (cursor + nv > 500) nv = 500 - cursor;
            s.polyDesc[p] = (cursor << 16) | (nv << 8) | 0;
            for (uint32_t k = 0; k < nv; ++k)
            {
                const uint32_t b = cursor + k;
                uint32_t* w = &(&s.vtxData[b / 16].x)[(b / 4) & 3];
                *w |= (rng() % nbVtx) << (8 * (b & 3));
            }
            cursor += nv;
        }
        const V3 pn0 = unit(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) });
        for (uint32_t p = 0; p < 64; ++p)
        {
            const V3 pn = unit(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) });
            s.planes[p] = F4{ pn.x, pn.y, pn.z, rf(-2, 2) };
        }
        (void)pn0;
        for (uint32_t e = 0; e < 96; ++e)
        {
            const uint32_t f0 = rng() % nbP;
            uint32_t f1 = rng() % nbP;
            if (f1 == f0) f1 = (f1 + 1) % nbP;
            const uint32_t b0 = 2 * e, b1 = 2 * e + 1;
            uint32_t* w0 = &(&s.fbe[b0 / 16].x)[(b0 / 4) & 3];
            *w0 |= (f0 & 0xFF) << (8 * (b0 & 3));
            uint32_t* w1 = &(&s.fbe[b1 / 16].x)[(b1 / 4) & 3];
            *w1 |= (f1 & 0xFF) << (8 * (b1 & 3));
        }
        s.scale = F4{ rf(0.5f, 2), rf(0.5f, 2), rf(0.5f, 2), float(nbP) };
        s.rot = randQuat();
        const V3 n = unit(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) });
        s.n = F4{ n.x, n.y, n.z, float(nbE) };
        s.bToAq = randQuat();
        // bToAp.x controls how many vertices fall below the plane: sweep it
        s.bToAp = F4{ rf(-3.f, 2.f), rf(-1, 1), rf(-1, 1), rf(0.05f, 0.5f) };
        s.planeTq = randQuat();
        s.convexTq = randQuat();
        in[i] = s;

        bool marg = false;
        refPlaneGen(s, ref[i], marg);
        skip[i] = marg ? 1u : 0u;
        if (marg) { ++nSkip; continue; }
        if (ref[i].aux.y != 0xFFFFFFFFu) ++cEdgePath;
        if (ref[i].result.x == 0) ++cZero;
        if (ref[i].result.z > 4) ++cRed;
        if (ref[i].result.z > 16) ++cBig;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<GenOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct GenParams { uint32_t numTests; } prm = { N };
    Kernel k("npCvxPlaneGen.spv", "cvxPlaneGen",
             { (uint32_t)sizeof(GenIn), (uint32_t)sizeof(GenOut) }, sizeof(GenParams));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<GenOut>(dev, bOut, N);
    size_t mmR = 0, mmM = 0, mmD = 0, mmP = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (skip[i]) continue;
        if (g[i].result.x != ref[i].result.x || g[i].result.y != ref[i].result.y ||
            g[i].result.z != ref[i].result.z || g[i].result.w != ref[i].result.w ||
            g[i].aux.x != ref[i].aux.x || g[i].aux.y != ref[i].aux.y) { ++mmR; continue; }
        for (int j = 0; j < 4; ++j)
            if (!eqF4(ref[i].mNp[j], g[i].mNp[j]) || !eqF4(ref[i].mA[j], g[i].mA[j]) || !eqF4(ref[i].mB[j], g[i].mB[j])) { ++mmM; break; }
        if (ref[i].result.w)
        {
            // moreData quat words are bit-copies; md0 floats compared as floats
            F4 rp, gp; std::memcpy(&rp, &ref[i].md0, 16); std::memcpy(&gp, &g[i].md0, 16);
            if (!eqF4(rp, gp) || g[i].md1.x != ref[i].md1.x || g[i].md1.w != ref[i].md1.w ||
                g[i].md2.x != ref[i].md2.x || g[i].md2.w != ref[i].md2.w) ++mmD;
        }
        else if (g[i].md0.x != 0 || g[i].md1.x != 0 || g[i].md2.x != 0) ++mmD;
        for (int L = 0; L < 32; ++L) if (!eqF4(ref[i].p[L], g[i].p[L])) { ++mmP; break; }
    }
    PXG_CHECK_MSG(mmR == 0, "planeGen: %zu result mismatches", mmR);
    PXG_CHECK_MSG(mmM == 0, "planeGen: %zu manifold mismatches", mmM);
    PXG_CHECK_MSG(mmD == 0, "planeGen: %zu moreData mismatches", mmD);
    PXG_CHECK_MSG(mmP == 0, "planeGen: %zu reload mismatches", mmP);
    PXG_CHECK_MSG(nSkip < N / 2, "planeGen: too many marginal (%zu/%u)", nSkip, N);
    PXG_CHECK_MSG(cEdgePath > N / 30 && cZero > N / 30 && cRed > N / 20 && cBig > N / 100,
                  "planeGen: coverage thin (edge %zu zero %zu red %zu big %zu)", cEdgePath, cZero, cRed, cBig);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
