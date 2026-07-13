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

// Verification of npCvxPlanePolyIndex.slang: getPolygonIndexFromPlaneNormal --
// the convex-vs-plane face selector (warp argmax over covariant-transformed
// polygon normals, per-lane strict-< bests with lowest-lane cross-ties, the
// edge pass with its +5e-4 preference bias, and the edge-path face pick).
// The reference transcribes the CUDA warp semantics exactly; argmax and gate
// near-ties are margin-skipped.

#include "pxgTestFramework.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };
struct V3 { float x, y, z; };
struct PpIn { F4 planes[64]; U4 fbe[12]; F4 scale, rot, n; };
struct PpOut { uint32_t index, plane2, usedEdge, pad0; };

const float FLTMAX = 3.4028235e38f;
const float TIE = 2e-4f;

static inline V3 v3(const F4& f) { return V3{ f.x, f.y, f.z }; }
static inline V3 add(const V3& a, const V3& b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static inline V3 mul(const V3& a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; }
static inline float dot3(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 unit(const V3& a) { float l = std::sqrt(dot3(a, a)); return l > 0 ? mul(a, 1.f / l) : a; }
static V3 qRot(const F4& q, const V3& v) { const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z, w2 = q.w * q.w - 0.5f, d2 = q.x * vx + q.y * vy + q.z * vz; return V3{ vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d2, vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d2, vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d2 }; }
static V3 qRotInv(const F4& q, const V3& v) { const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z, w2 = q.w * q.w - 0.5f, d2 = q.x * vx + q.y * vy + q.z * vz; return V3{ vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * d2, vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * d2, vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * d2 }; }
static V3 getNormalized(const V3& v) { const float m = dot3(v, v); return m > 0 ? mul(v, 1.f / std::sqrt(m)) : V3{ 0, 0, 0 }; }
static V3 shape2Vertex(const V3& v, const V3& sc, const F4& rot)
{
    const V3 r = qRot(rot, v);
    return qRotInv(rot, V3{ r.x / sc.x, r.y / sc.y, r.z / sc.z });
}
static V3 loadNormal(const F4& p, const V3& sc, const F4& rot) { return getNormalized(shape2Vertex(v3(p), sc, rot)); }

static uint32_t fbeByte(const PpIn& s, uint32_t b)
{
    const uint32_t* w = &s.fbe[b / 16].x;
    return (w[(b / 4) & 3] >> (8 * (b & 3))) & 0xFF;
}

static void refPlanePoly(const PpIn& s, PpOut& o, bool& marg)
{
    const uint32_t nbP = (uint32_t)s.scale.w;
    const uint32_t nbE = (uint32_t)s.n.w;
    const V3 sc = v3(s.scale);
    const F4 rot = s.rot;
    const V3 n = v3(s.n);

    float laneBest[32]; uint32_t laneIdx[32];
    for (int L = 0; L < 32; ++L) { laneBest[L] = -FLTMAX; laneIdx[L] = 0; }
    for (uint32_t i = 0; i < nbP; ++i)
    {
        const int L = i & 31;
        const float proj = dot3(getNormalized(shape2Vertex(v3(s.planes[i]), sc, rot)), n);
        const float dl = std::fabs(proj - laneBest[L]);
        if (laneBest[L] > -FLTMAX && dl != 0.f && dl < TIE) marg = true; // near-tie EITHER way (exact ties are deterministic)
        if (laneBest[L] < proj) { laneIdx[L] = i; laneBest[L] = proj; }
    }
    float bestProj = -FLTMAX;
    for (int L = 0; L < 32; ++L) bestProj = std::fmax(bestProj, laneBest[L]);
    uint32_t bestLane = 0;
    for (int L = 0; L < 32; ++L) if (laneBest[L] == bestProj) { bestLane = (uint32_t)L; break; }
    for (int L = 0; L < 32; ++L)
        { const float dl = std::fabs(laneBest[L] - bestProj); if ((uint32_t)L != bestLane && laneBest[L] > -FLTMAX && dl != 0.f && dl < TIE) marg = true; }
    uint32_t index = laneIdx[bestLane];

    float laneE[32]; uint32_t laneEIdx[32];
    for (int L = 0; L < 32; ++L) { laneE[L] = -FLTMAX; laneEIdx[L] = 0xFFFFFFFFu; }
    for (uint32_t i = 0; i < nbE; ++i)
    {
        const int L = i & 31;
        const uint32_t f0 = fbeByte(s, 2 * i), f1 = fbeByte(s, 2 * i + 1);
        const V3 e = getNormalized(add(loadNormal(s.planes[f0], sc, rot), loadNormal(s.planes[f1], sc, rot)));
        const float dp = dot3(e, n);
        const float de = std::fabs(dp - laneE[L]);
        if (laneE[L] > -FLTMAX && de != 0.f && de < TIE) marg = true; // near-tie either way
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

    if (nbE > 0 && std::fabs(bestEdge - bestProj) < TIE) marg = true; // bias-gate near-tie

    uint32_t plane2 = 0xFFFFFFFFu, usedEdge = 0;
    if (bestEdge > bestProj)
    {
        usedEdge = 1;
        const uint32_t f0 = fbeByte(s, closestEdge * 2), f1 = fbeByte(s, closestEdge * 2 + 1);
        const float dp0 = dot3(loadNormal(s.planes[f0], sc, rot), n);
        const float dp1 = dot3(loadNormal(s.planes[f1], sc, rot), n);
        if (std::fabs(dp0 - dp1) < TIE) marg = true;
        if (dp0 > dp1) { index = f0; plane2 = f1; }
        else           { index = f1; plane2 = f0; }
    }
    o = PpOut{ index, plane2, usedEdge, 0 };
}

PXG_TEST(npCvxPlanePolyIndex_select)
{
    std::mt19937 rng(155001);
    const uint32_t N = 12000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto randQuat = [&]() { V3 ax = unit(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) }); const float a = rf(-3.1f, 3.1f); return F4{ ax.x * std::sin(a * .5f), ax.y * std::sin(a * .5f), ax.z * std::sin(a * .5f), std::cos(a * .5f) }; };

    std::vector<PpIn> in(N);
    std::vector<uint8_t> skip(N, 0);
    std::vector<PpOut> ref(N);
    size_t nSkip = 0, cEdge = 0, cFace = 0, cZeroE = 0, cManyP = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        PpIn s{};
        const uint32_t nbP = 4 + rng() % 61; // 4..64
        if (nbP > 32) ++cManyP;
        uint32_t nbE = (i % 9 == 0) ? 0u : (1 + rng() % 96); // include the zero-edge face path
        if (nbE == 0) ++cZeroE;
        for (uint32_t j = 0; j < 64; ++j)
        {
            const V3 pn = unit(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) });
            s.planes[j] = F4{ pn.x, pn.y, pn.z, rf(-2, 2) };
        }
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
        F4 scaleQ{ rf(0.5f, 2), rf(0.5f, 2), rf(0.5f, 2), 0 };
        F4 rotQ = randQuat();
        V3 n = unit(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) });
        if (i % 13 == 0 && nbP > 34)
        {
            // identity transform so vertex-space normals ARE shape-space normals:
            // exact-tie fixtures for the per-lane (0 vs 32) and cross-lane
            // (lanes 2 vs 17 -- ALSO distinguishes a halved lane mask, 17&15=1)
            // argmax tie-breaks; edges off so the face path decides
            scaleQ = F4{ 1, 1, 1, 0 }; rotQ = F4{ 0, 0, 0, 1 };
            s.planes[2] = F4{ n.x, n.y, n.z, rf(-2, 2) }; // the global best
            s.planes[17] = s.planes[2]; // cross-lane tie (lanes 2 vs 17)
            s.planes[34] = s.planes[2]; // SAME-lane tie on the winner (34 & 31 == 2)
            nbE = 0; ++cZeroE;
        }
        else if (i % 17 == 3 && nbP > 33)
        {
            // duplicate-edge fixture: two edges in the SAME lane (0 and 32) with
            // bitwise-identical averaged normals but DIFFERENT face pairs -- the
            // per-lane strict-> first-wins tie-break picks different faces
            scaleQ = F4{ 1, 1, 1, 0 }; rotQ = F4{ 0, 0, 0, 1 };
            V3 arb{ 0.3f, -0.7f, 0.64f };
            V3 u = unit(V3{ n.y * arb.z - n.z * arb.y, n.z * arb.x - n.x * arb.z, n.x * arb.y - n.y * arb.x });
            const float c25 = std::cos(25.f * 3.14159265f / 180.f), s25 = std::sin(25.f * 3.14159265f / 180.f);
            const float c35 = std::cos(35.f * 3.14159265f / 180.f), s35 = std::sin(35.f * 3.14159265f / 180.f);
            const V3 A = add(mul(n, c25), mul(u, s25));
            const V3 B = add(mul(n, c35), mul(u, -s35));
            s.planes[4] = F4{ A.x, A.y, A.z, rf(-2, 2) }; s.planes[5] = s.planes[4];
            s.planes[6] = F4{ B.x, B.y, B.z, rf(-2, 2) }; s.planes[8] = s.planes[6];
            // other faces weakly aligned so the edge (average ~5 deg off n) wins
            for (uint32_t j = 0; j < 64; ++j)
            {
                if (j == 4 || j == 5 || j == 6 || j == 8) continue;
                V3 pn = unit(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) });
                if (dot3(pn, n) > 0.6f) pn = mul(pn, -1.f);
                s.planes[j] = F4{ pn.x, pn.y, pn.z, rf(-2, 2) };
            }
            nbE = 33 + rng() % 60;
            // edge 0 = (4,6), edge 32 = (5,8): same lane 0, identical dp
            for (uint32_t e = 0; e < 96; ++e)
            {
                uint32_t f0, f1;
                if (e == 0) { f0 = 4; f1 = 6; }
                else if (e == 32) { f0 = 5; f1 = 8; }
                else { f0 = 9 + rng() % (nbP - 9); f1 = 9 + rng() % (nbP - 9); if (f1 == f0) f1 = 9 + (f1 - 8) % (nbP - 9); }
                const uint32_t b0 = 2 * e, b1 = 2 * e + 1;
                uint32_t* w0 = &(&s.fbe[b0 / 16].x)[(b0 / 4) & 3];
                *w0 = (*w0 & ~(0xFFu << (8 * (b0 & 3)))) | ((f0 & 0xFF) << (8 * (b0 & 3)));
                uint32_t* w1 = &(&s.fbe[b1 / 16].x)[(b1 / 4) & 3];
                *w1 = (*w1 & ~(0xFFu << (8 * (b1 & 3)))) | ((f1 & 0xFF) << (8 * (b1 & 3)));
            }
        }
        s.scale = F4{ scaleQ.x, scaleQ.y, scaleQ.z, float(nbP) };
        s.rot = rotQ;
        s.n = F4{ n.x, n.y, n.z, float(nbE) };
        in[i] = s;

        bool marg = false;
        refPlanePoly(s, ref[i], marg);
        skip[i] = marg ? 1u : 0u;
        if (marg) { ++nSkip; continue; }
        if (ref[i].usedEdge) ++cEdge; else ++cFace;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<PpOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct PpParams { uint32_t numTests; } prm = { N };
    Kernel k("npCvxPlanePolyIndex.spv", "cvxPlanePolyIndex",
             { (uint32_t)sizeof(PpIn), (uint32_t)sizeof(PpOut) }, sizeof(PpParams));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<PpOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (skip[i]) continue;
        if (g[i].index != ref[i].index || g[i].plane2 != ref[i].plane2 || g[i].usedEdge != ref[i].usedEdge) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "planePolyIndex: %zu mismatches", mm);
    PXG_CHECK_MSG(nSkip < N / 3, "planePolyIndex: too many marginal (%zu/%u)", nSkip, N);
    PXG_CHECK_MSG(cEdge > N / 20 && cFace > N / 20 && cZeroE > N / 20 && cManyP > N / 10,
                  "planePolyIndex: coverage thin (edge %zu face %zu zeroE %zu manyP %zu)",
                  cEdge, cFace, cZeroE, cManyP);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
