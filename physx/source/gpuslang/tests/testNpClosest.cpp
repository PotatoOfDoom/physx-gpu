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

// Verification of npClosest.slang: ctClosestPtTriangle (closestPtPointTriangle,
// the Ericson 7-region closest point) and ctClosestPtEdge (closestPtPointEdge).
// The closest point and its (s,t) parameters (and the edge parameter w) are
// continuous functions of the query point that agree across region boundaries,
// so they are compared with almostEqual with no measure-zero skip; the C++
// reference also classifies which Voronoi region was taken so all seven are
// exercised.

#include "pxgTestFramework.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct V3 { float x, y, z; };
struct F4 { float x, y, z, w; };
struct TriIn { F4 p, a, b, c; };
struct TriOut { F4 closest, st; };
struct EdgeIn { F4 v0, ev0, ev1; };
struct ClParams { uint32_t numTests, pad0, pad1, pad2; };

static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 add(V3 a, V3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 mul(V3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
static V3 cross3(V3 a, V3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static float len(V3 a) { return std::sqrt(dot3(a, a)); }
static float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
static F4 f4(V3 v) { return { v.x, v.y, v.z, 0.0f }; }

static V3 refClosestTri(V3 p, V3 a, V3 b, V3 c, float& s, float& t, int& region)
{
    const V3 ab = sub(b, a), ac = sub(c, a), ap = sub(p, a);
    const float d1 = dot3(ab, ap), d2 = dot3(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) { s = 0; t = 0; region = 0; return a; }
    const V3 bp = sub(p, b);
    const float d3 = dot3(ab, bp), d4 = dot3(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) { s = 1; t = 0; region = 1; return b; }
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) { const float v = d1 / (d1 - d3); s = v; t = 0; region = 2; return add(a, mul(ab, v)); }
    const V3 cp = sub(p, c);
    const float d5 = dot3(ab, cp), d6 = dot3(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) { s = 0; t = 1; region = 3; return c; }
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) { const float w = d2 / (d2 - d6); s = 0; t = w; region = 4; return add(a, mul(ac, w)); }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) { const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6)); s = 1 - w; t = w; region = 5; return add(b, mul(sub(c, b), w)); }
    const float denom = 1.0f / (va + vb + vc);
    const float v = vb * denom, w = vc * denom;
    s = v; t = w; region = 6;
    return add(add(a, mul(ab, v)), mul(ac, w));
}

PXG_TEST(npClosest_pointTriangle)
{
    std::mt19937 rng(72011);
    const uint32_t N = 16000;
    std::vector<TriIn> in(N);
    std::vector<TriOut> ref(N);
    std::uniform_real_distribution<float> uc(-2, 2);
    std::uniform_real_distribution<float> upt(-4, 4); // wide, to hit all 7 regions
    size_t regc[7] = { 0, 0, 0, 0, 0, 0, 0 };
    for (uint32_t i = 0; i < N; ++i)
    {
        V3 a, b, c;
        do { a = { uc(rng), uc(rng), uc(rng) }; b = { uc(rng), uc(rng), uc(rng) }; c = { uc(rng), uc(rng), uc(rng) };
        } while (len(cross3(sub(b, a), sub(c, a))) < 0.5f);
        const V3 p = { upt(rng), upt(rng), upt(rng) };
        in[i] = { f4(p), f4(a), f4(b), f4(c) };
        float s, t; int region; const V3 cp = refClosestTri(p, a, b, c, s, t, region);
        ref[i] = { f4(cp), { s, t, 0, 0 } };
        ++regc[region];
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<TriOut> outInit(N);
    std::vector<EdgeIn> eD(N); std::vector<float> eoD(N);
    PxgBufferHandle bT = createBufferFrom(dev, in);
    PxgBufferHandle bTo = createBufferFrom(dev, outInit);
    PxgBufferHandle bE = createBufferFrom(dev, eD);
    PxgBufferHandle bEo = createBufferFrom(dev, eoD);

    ClParams prm = { N, 0, 0, 0 };
    Kernel k("npClosest.spv", "ctClosestPtTriangle",
             { (uint32_t)sizeof(TriIn), (uint32_t)sizeof(TriOut), (uint32_t)sizeof(EdgeIn), 4 }, sizeof(ClParams));
    k.dispatch(GROUPS, { bT, bTo, bE, bEo }, prm);
    dev->flush();

    const auto g = downloadBuffer<TriOut>(dev, bTo, N);
    size_t mmC = 0, mmST = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const TriOut& r = ref[i]; const TriOut& v = g[i];
        if (!almostEqual(v.closest.x, r.closest.x, 1e-4f, 1e-4f) || !almostEqual(v.closest.y, r.closest.y, 1e-4f, 1e-4f) || !almostEqual(v.closest.z, r.closest.z, 1e-4f, 1e-4f)) ++mmC;
        if (!almostEqual(v.st.x, r.st.x, 1e-4f, 1e-4f) || !almostEqual(v.st.y, r.st.y, 1e-4f, 1e-4f)) ++mmST;
    }
    PXG_CHECK_MSG(mmC == 0, "ctClosestPtTriangle: %zu closest-point mismatches", mmC);
    PXG_CHECK_MSG(mmST == 0, "ctClosestPtTriangle: %zu (s,t) mismatches", mmST);
    PXG_CHECK_MSG(regc[0] > 50 && regc[1] > 50 && regc[2] > 50 && regc[3] > 50 && regc[4] > 50 && regc[5] > 50 && regc[6] > 50,
                  "ctClosestPtTriangle: region coverage thin (%zu/%zu/%zu/%zu/%zu/%zu/%zu)", regc[0], regc[1], regc[2], regc[3], regc[4], regc[5], regc[6]);

    dev->destroyBuffer(bT); dev->destroyBuffer(bTo); dev->destroyBuffer(bE); dev->destroyBuffer(bEo);
}

PXG_TEST(npClosest_pointEdge)
{
    std::mt19937 rng(72029);
    const uint32_t N = 12000;
    std::vector<EdgeIn> in(N);
    std::vector<float> ref(N);
    std::uniform_real_distribution<float> uc(-3, 3);
    size_t lo = 0, hi = 0, mid = 0, deg = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        // Every 8th sample uses a zero-length (degenerate) edge to exercise the
        // d2==0 -> 0.5 fallback path in closestPtPointEdge.
        const bool degen = (i % 8 == 0);
        V3 ev0, ev1;
        if (degen) { ev0 = { uc(rng), uc(rng), uc(rng) }; ev1 = ev0; }
        else { do { ev0 = { uc(rng), uc(rng), uc(rng) }; ev1 = { uc(rng), uc(rng), uc(rng) }; } while (len(sub(ev1, ev0)) < 0.3f); }
        const V3 v0 = { uc(rng), uc(rng), uc(rng) };
        in[i] = { f4(v0), f4(ev0), f4(ev1) };
        const V3 r = sub(ev1, ev0);
        const float d2 = dot3(r, r);
        const float w = (d2 > 0.0f) ? clampf(dot3(r, sub(v0, ev0)) / d2, 0.0f, 1.0f) : 0.5f;
        ref[i] = w;
        if (degen) ++deg; else if (w <= 0.0f) ++lo; else if (w >= 1.0f) ++hi; else ++mid;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<TriIn> tD(N); std::vector<TriOut> toD(N);
    std::vector<float> outInit(N, -9.0f);
    PxgBufferHandle bT = createBufferFrom(dev, tD);
    PxgBufferHandle bTo = createBufferFrom(dev, toD);
    PxgBufferHandle bE = createBufferFrom(dev, in);
    PxgBufferHandle bEo = createBufferFrom(dev, outInit);

    ClParams prm = { N, 0, 0, 0 };
    Kernel k("npClosest.spv", "ctClosestPtEdge",
             { (uint32_t)sizeof(TriIn), (uint32_t)sizeof(TriOut), (uint32_t)sizeof(EdgeIn), 4 }, sizeof(ClParams));
    k.dispatch(GROUPS, { bT, bTo, bE, bEo }, prm);
    dev->flush();

    const auto g = downloadBuffer<float>(dev, bEo, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i) if (!almostEqual(g[i], ref[i], 1e-5f, 1e-5f)) ++mm;
    PXG_CHECK_MSG(mm == 0, "ctClosestPtEdge: %zu w mismatches", mm);
    PXG_CHECK_MSG(lo > 100 && hi > 100 && mid > N / 4 && deg > N / 16, "ctClosestPtEdge: coverage thin (lo=%zu hi=%zu mid=%zu deg=%zu)", lo, hi, mid, deg);

    dev->destroyBuffer(bT); dev->destroyBuffer(bTo); dev->destroyBuffer(bE); dev->destroyBuffer(bEo);
}

} // namespace
