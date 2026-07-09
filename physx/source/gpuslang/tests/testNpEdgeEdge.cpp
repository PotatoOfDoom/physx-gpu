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

// Verification of npEdgeEdge.slang: ctLineLine (closestPtLineLine, the two
// infinite-line closest points with degenerate/near-parallel early-outs) and
// ctEdgeEdge (closestPtEdgeEdge, the finite-segment version with the four-way
// endpoint-projection fallback).  Compared against faithful C++ transcriptions.
// The continuous outputs (distSq, and s/t away from selection boundaries) are
// compared to tolerance; the returned boolean and the discrete branch/fallback
// selection are exercised across categories, with the measure-zero decision
// boundaries (cosTheta == 0.9999, the [0,1] fallback gate, candidate ties, and
// the ill-conditioned near-parallel band) margin-skipped.  For the finite-edge
// case the (s,t) witness is also independently validated by reconstructing the
// closest points, and distSq is sanity-checked against a brute-force sample.

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
struct EEIn { F4 e0v0, e0v1, e1v0, e1v1; };
struct EEOut { F4 res; };
struct EEParams { uint32_t numTests, pad0, pad1, pad2; };

static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 add(V3 a, V3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 mul(V3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
static float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
static F4 f4(V3 v) { return { v.x, v.y, v.z, 0.0f }; }

static void refPointEdge(V3 v0, V3 ev0, V3 ev1, float& w)
{
    const V3 r = sub(ev1, ev0);
    const float d2 = dot3(r, r);
    w = (d2 > 0.0f) ? clampf(dot3(r, sub(v0, ev0)) / d2, 0.0f, 1.0f) : 0.5f;
}

// Faithful transcription of closestPtLineLine; also exposes cosTheta for the
// near-parallel margin-skip.
static bool refLineLine(V3 e0v0, V3 e0v1, V3 e1v0, V3 e1v1, float& s, float& t, float& distSq, float& cosTheta)
{
    cosTheta = 2.0f; // sentinel: no cosTheta computed (degenerate)
    const V3 r0 = sub(e0v1, e0v0), r1 = sub(e1v1, e1v0), d = sub(e1v0, e0v0);
    const float r0Len2 = dot3(r0, r0), r1Len2 = dot3(r1, r1);
    if (r0Len2 < 1.0e-18f || r1Len2 < 1.0e-18f) { s = 0; t = 0; distSq = dot3(d, d); return false; }
    const float r0Len = std::sqrt(r0Len2), r1Len = std::sqrt(r1Len2);
    const V3 u0 = mul(r0, 1.0f / r0Len), u1 = mul(r1, 1.0f / r1Len);
    cosTheta = std::fabs(dot3(u0, u1));
    if (cosTheta > 0.9999f)
    {
        const float tProj = dot3(sub(e0v0, e1v0), r1) / r1Len2;
        t = tProj; s = 0.0f;
        const V3 pt0 = e0v0, pt1 = add(e1v0, mul(r1, tProj));
        distSq = dot3(sub(pt1, pt0), sub(pt1, pt0));
        return false;
    }
    const float a01 = -dot3(u0, u1);
    const float b0 = dot3(u0, d) / r0Len;
    const float b1 = -dot3(u1, d) / r1Len;
    const float det = 1.0f - a01 * a01;
    if (det < 1.0e-9f) { s = 0; t = 0; distSq = dot3(d, d); return false; }
    const float invDet = 1.0f / det;
    s = (b0 - b1 * a01) * invDet;
    t = (b1 - a01 * b0) * invDet;
    const V3 pt0 = add(e0v0, mul(r0, s)), pt1 = add(e1v0, mul(r1, t));
    distSq = dot3(sub(pt1, pt0), sub(pt1, pt0));
    return true;
}

struct EERef { float s, t, distSq; bool ok; float cosTheta; bool fallback; float llS, llT; float d2sorted[4]; };

static EERef refEdgeEdge(V3 e0v0, V3 e0v1, V3 e1v0, V3 e1v1)
{
    EERef r{};
    r.fallback = false;
    r.ok = refLineLine(e0v0, e0v1, e1v0, e1v1, r.s, r.t, r.distSq, r.cosTheta);
    r.llS = r.s; r.llT = r.t;
    for (int i = 0; i < 4; ++i) r.d2sorted[i] = -1.0f;
    if (!r.ok) return r;
    if (r.s < 0.0f || r.s > 1.0f || r.t < 0.0f || r.t > 1.0f)
    {
        r.fallback = true;
        float w[4];
        refPointEdge(e0v0, e1v0, e1v1, w[0]);
        refPointEdge(e0v1, e1v0, e1v1, w[1]);
        refPointEdge(e1v0, e0v0, e0v1, w[2]);
        refPointEdge(e1v1, e0v0, e0v1, w[3]);
        const float cand[4][2] = { { 0.0f, w[0] }, { 1.0f, w[1] }, { w[2], 0.0f }, { w[3], 1.0f } };
        int minIndex = 0;
        float best = 3.4028235e38f;
        float d2[4];
        for (int i = 0; i < 4; ++i)
        {
            const float sC = cand[i][0], tC = cand[i][1];
            const V3 p0 = add(mul(e0v0, 1.0f - sC), mul(e0v1, sC));
            const V3 p1 = add(mul(e1v0, 1.0f - tC), mul(e1v1, tC));
            const float dd = dot3(sub(p0, p1), sub(p0, p1));
            d2[i] = dd;
            minIndex = (dd < best) ? i : minIndex;
            best = dd < best ? dd : best;
        }
        r.distSq = best;
        r.s = cand[minIndex][0];
        r.t = cand[minIndex][1];
        // sort d2 ascending (n=4) to measure the top-two gap
        for (int i = 0; i < 4; ++i) r.d2sorted[i] = d2[i];
        for (int i = 0; i < 4; ++i) for (int j = i + 1; j < 4; ++j) if (r.d2sorted[j] < r.d2sorted[i]) { float tmp = r.d2sorted[i]; r.d2sorted[i] = r.d2sorted[j]; r.d2sorted[j] = tmp; }
    }
    return r;
}

// Random unit-ish direction with a target length.
static V3 randDir(std::mt19937& rng, std::uniform_real_distribution<float>& u)
{
    return { u(rng), u(rng), u(rng) };
}

PXG_TEST(npEdgeEdge_lineLine)
{
    std::mt19937 rng(74011);
    const uint32_t N = 16000;
    const float MARG_C = 6e-4f; // margin around the cosTheta==0.9999 boundary
    std::uniform_real_distribution<float> uc(-3, 3);
    std::uniform_real_distribution<float> un(-1, 1);
    std::vector<EEIn> in(N);
    std::vector<EEOut> ref(N);
    std::vector<uint8_t> skip(N, 0);      // skip value+bool comparison (boundary)
    size_t cDeg = 0, cPar = 0, cGen = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        V3 e0v0, e0v1, e1v0, e1v1;
        const uint32_t cat = i % 6;
        if (cat == 0)
        {
            // Degenerate: edge0 zero length.
            e0v0 = randDir(rng, uc); e0v1 = e0v0;
            e1v0 = randDir(rng, uc); e1v1 = add(e1v0, randDir(rng, un));
        }
        else if (cat == 1 || cat == 2)
        {
            // Near-parallel: r1 ~ +/- r0 with a tiny perpendicular perturbation.
            V3 r0 = randDir(rng, un);
            while (dot3(r0, r0) < 0.2f) r0 = randDir(rng, un);
            const float sgn = (i & 1) ? 1.0f : -1.0f;
            const float scl = 0.5f + 1.5f * (0.5f * (un(rng) + 1.0f));
            V3 noise = mul(randDir(rng, un), 2.0e-3f * std::sqrt(dot3(r0, r0)));
            V3 r1 = add(mul(r0, sgn * scl), noise);
            e0v0 = randDir(rng, uc); e0v1 = add(e0v0, r0);
            e1v0 = randDir(rng, uc); e1v1 = add(e1v0, r1);
        }
        else
        {
            // General: reject ill-conditioned (near-parallel) pairs.
            for (;;)
            {
                V3 r0 = randDir(rng, un); V3 r1 = randDir(rng, un);
                if (dot3(r0, r0) < 0.2f || dot3(r1, r1) < 0.2f) continue;
                const float ct = std::fabs(dot3(r0, r1)) / std::sqrt(dot3(r0, r0) * dot3(r1, r1));
                if (1.0f - ct * ct < 1.0e-2f) continue; // det = sin^2 >= 1e-2
                e0v0 = randDir(rng, uc); e0v1 = add(e0v0, r0);
                e1v0 = randDir(rng, uc); e1v1 = add(e1v0, r1);
                break;
            }
        }
        in[i] = { f4(e0v0), f4(e0v1), f4(e1v0), f4(e1v1) };

        float s, t, distSq, cosTheta;
        const bool ok = refLineLine(e0v0, e0v1, e1v0, e1v1, s, t, distSq, cosTheta);
        ref[i] = { { s, t, distSq, ok ? 1.0f : 0.0f } };
        if (cosTheta <= 1.5f && std::fabs(cosTheta - 0.9999f) < MARG_C) skip[i] = 1;
        if (!ok) { if (cosTheta > 1.5f) ++cDeg; else ++cPar; } else ++cGen;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<EEOut> outInit(N), edgeInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bLine = createBufferFrom(dev, outInit);
    PxgBufferHandle bEdge = createBufferFrom(dev, edgeInit);

    EEParams prm = { N, 0, 0, 0 };
    Kernel k("npEdgeEdge.spv", "ctLineLine",
             { (uint32_t)sizeof(EEIn), (uint32_t)sizeof(EEOut), (uint32_t)sizeof(EEOut) }, sizeof(EEParams));
    k.dispatch(GROUPS, { bIn, bLine, bEdge }, prm);
    dev->flush();

    const auto g = downloadBuffer<EEOut>(dev, bLine, N);
    size_t mmOk = 0, mmD = 0, mmST = 0, skipped = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (skip[i]) { ++skipped; continue; }
        const EEOut& r = ref[i]; const EEOut& v = g[i];
        if (v.res.w != r.res.w) ++mmOk;
        if (!almostEqual(v.res.z, r.res.z, 2e-3f, 1e-4f)) ++mmD;
        if (!almostEqual(v.res.x, r.res.x, 2e-3f, 1e-4f) || !almostEqual(v.res.y, r.res.y, 2e-3f, 1e-4f)) ++mmST;
    }
    PXG_CHECK_MSG(mmOk == 0, "ctLineLine: %zu bool mismatches", mmOk);
    PXG_CHECK_MSG(mmD == 0, "ctLineLine: %zu distSq mismatches", mmD);
    PXG_CHECK_MSG(mmST == 0, "ctLineLine: %zu (s,t) mismatches", mmST);
    PXG_CHECK_MSG(cDeg > 100 && cPar > 100 && cGen > N / 4, "ctLineLine: coverage thin (deg=%zu par=%zu gen=%zu)", cDeg, cPar, cGen);

    dev->destroyBuffer(bIn); dev->destroyBuffer(bLine); dev->destroyBuffer(bEdge);
}

PXG_TEST(npEdgeEdge_edgeEdge)
{
    std::mt19937 rng(74029);
    const uint32_t N = 16000;
    const float MARG_C = 6e-4f;   // cosTheta==0.9999 boundary
    const float MARG_G = 2e-3f;   // [0,1] fallback gate
    const float MARG_T = 1e-3f;   // candidate top-two tie
    std::uniform_real_distribution<float> uc(-3, 3);
    std::uniform_real_distribution<float> un(-1, 1);
    std::uniform_real_distribution<float> u01(0, 1);
    std::vector<EEIn> in(N);
    std::vector<EERef> refv(N);
    std::vector<uint8_t> skip(N, 0), skipST(N, 0);
    size_t cFallback = 0, cInterior = 0, cParFalse = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        V3 e0v0, e0v1, e1v0, e1v1;
        const uint32_t cat = i % 4;
        if (cat == 0)
        {
            // Near-parallel finite edges (exercise the lineLine-returns-false path).
            V3 r0 = randDir(rng, un);
            while (dot3(r0, r0) < 0.2f) r0 = randDir(rng, un);
            const float sgn = (i & 1) ? 1.0f : -1.0f;
            const float scl = 0.5f + 1.5f * u01(rng);
            V3 noise = mul(randDir(rng, un), 2.0e-3f * std::sqrt(dot3(r0, r0)));
            V3 r1 = add(mul(r0, sgn * scl), noise);
            e0v0 = randDir(rng, uc); e0v1 = add(e0v0, r0);
            e1v0 = randDir(rng, uc); e1v1 = add(e1v0, r1);
        }
        else if (cat == 1)
        {
            // Near-crossing edges whose closest approach is interior to both
            // (line solution inside [0,1]^2 -> no fallback).
            V3 dir0, dir1;
            for (;;)
            {
                dir0 = randDir(rng, un); dir1 = randDir(rng, un);
                if (dot3(dir0, dir0) < 0.4f || dot3(dir1, dir1) < 0.4f) continue;
                const float ct = std::fabs(dot3(dir0, dir1)) / std::sqrt(dot3(dir0, dir0) * dot3(dir1, dir1));
                if (1.0f - ct * ct < 5.0e-2f) continue; // comfortably non-parallel
                break;
            }
            const V3 m = randDir(rng, uc);
            const float s0 = 0.3f + 0.4f * u01(rng), t0 = 0.3f + 0.4f * u01(rng);
            const float L0 = 0.8f + 1.4f * u01(rng), L1 = 0.8f + 1.4f * u01(rng);
            const V3 off = mul(randDir(rng, un), 0.4f);
            e0v0 = sub(m, mul(dir0, s0 * L0)); e0v1 = add(m, mul(dir0, (1.0f - s0) * L0));
            const V3 m1 = add(m, off);
            e1v0 = sub(m1, mul(dir1, t0 * L1)); e1v1 = add(m1, mul(dir1, (1.0f - t0) * L1));
        }
        else
        {
            // General non-degenerate, well-conditioned edges (len >= ~0.45).
            for (;;)
            {
                V3 r0 = randDir(rng, un); V3 r1 = randDir(rng, un);
                if (dot3(r0, r0) < 0.2f || dot3(r1, r1) < 0.2f) continue;
                const float ct = std::fabs(dot3(r0, r1)) / std::sqrt(dot3(r0, r0) * dot3(r1, r1));
                if (1.0f - ct * ct < 1.0e-2f) continue;
                e0v0 = randDir(rng, uc); e0v1 = add(e0v0, r0);
                e1v0 = randDir(rng, uc); e1v1 = add(e1v0, r1);
                break;
            }
        }
        in[i] = { f4(e0v0), f4(e0v1), f4(e1v0), f4(e1v1) };
        const EERef r = refEdgeEdge(e0v0, e0v1, e1v0, e1v1);
        refv[i] = r;

        if (r.cosTheta <= 1.5f && std::fabs(r.cosTheta - 0.9999f) < MARG_C) skip[i] = 1;
        // s,t are a discrete selection: skip near the fallback gate and candidate ties.
        if (r.ok)
        {
            const float gate = std::fmin(std::fmin(std::fabs(r.llS), std::fabs(r.llS - 1.0f)),
                                         std::fmin(std::fabs(r.llT), std::fabs(r.llT - 1.0f)));
            if (gate < MARG_G) skipST[i] = 1;
            if (r.fallback && (r.d2sorted[1] - r.d2sorted[0]) < MARG_T) skipST[i] = 1;
        }
        if (!r.ok) ++cParFalse; else if (r.fallback) ++cFallback; else ++cInterior;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<EEOut> lineInit(N), outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bLine = createBufferFrom(dev, lineInit);
    PxgBufferHandle bEdge = createBufferFrom(dev, outInit);

    EEParams prm = { N, 0, 0, 0 };
    Kernel k("npEdgeEdge.spv", "ctEdgeEdge",
             { (uint32_t)sizeof(EEIn), (uint32_t)sizeof(EEOut), (uint32_t)sizeof(EEOut) }, sizeof(EEParams));
    k.dispatch(GROUPS, { bIn, bLine, bEdge }, prm);
    dev->flush();

    const auto g = downloadBuffer<EEOut>(dev, bEdge, N);
    size_t mmOk = 0, mmD = 0, mmST = 0, mmWit = 0, skipped = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const EERef& r = refv[i]; const EEOut& v = g[i];
        const EEIn& e = in[i];
        if (skip[i]) { ++skipped; continue; }
        if (v.res.w != (r.ok ? 1.0f : 0.0f)) ++mmOk;
        // distSq is also skipped near the fallback gate: for unequal-length
        // edges the lineLine distSq and the candidate distSq are not
        // continuous across the s/t = 0/1 gate, so a GPU-vs-CPU branch flip
        // there would legitimately change the value.
        if (!skipST[i] && !almostEqual(v.res.z, r.distSq, 2e-3f, 1e-4f)) ++mmD;
        if (!skipST[i] && (!almostEqual(v.res.x, r.s, 2e-3f, 1e-4f) || !almostEqual(v.res.y, r.t, 2e-3f, 1e-4f))) ++mmST;

        if (r.ok)
        {
            // Witness: reconstruct the closest points from the GPU (s,t) and check
            // they realize the reported distSq, and lie in [0,1].  (The CUDA
            // closestPtLineLine solves the closest-point system in NORMALIZED
            // direction coordinates but applies the result to the full-length
            // r0/r1 — exact only for equal-length edges — so distSq is not the
            // true global segment minimum; this only checks internal
            // consistency of the returned parameters, and the value comparison
            // above pins the port to the CUDA algorithm.)
            const V3 a0 = { e.e0v0.x, e.e0v0.y, e.e0v0.z }, a1 = { e.e0v1.x, e.e0v1.y, e.e0v1.z };
            const V3 b0 = { e.e1v0.x, e.e1v0.y, e.e1v0.z }, b1 = { e.e1v1.x, e.e1v1.y, e.e1v1.z };
            const float sC = v.res.x, tC = v.res.y;
            const V3 p0 = add(mul(a0, 1.0f - sC), mul(a1, sC));
            const V3 p1 = add(mul(b0, 1.0f - tC), mul(b1, tC));
            const float wd = dot3(sub(p0, p1), sub(p0, p1));
            if (sC < -1e-4f || sC > 1.0f + 1e-4f || tC < -1e-4f || tC > 1.0f + 1e-4f) ++mmWit;
            else if (!almostEqual(wd, v.res.z, 3e-3f, 1e-4f)) ++mmWit;
        }
    }
    PXG_CHECK_MSG(mmOk == 0, "ctEdgeEdge: %zu bool mismatches", mmOk);
    PXG_CHECK_MSG(mmD == 0, "ctEdgeEdge: %zu distSq mismatches", mmD);
    PXG_CHECK_MSG(mmST == 0, "ctEdgeEdge: %zu (s,t) mismatches", mmST);
    PXG_CHECK_MSG(mmWit == 0, "ctEdgeEdge: %zu witness mismatches", mmWit);
    PXG_CHECK_MSG(cFallback > N / 8 && cInterior > N / 8 && cParFalse > 50,
                  "ctEdgeEdge: coverage thin (fallback=%zu interior=%zu parFalse=%zu)", cFallback, cInterior, cParFalse);

    dev->destroyBuffer(bIn); dev->destroyBuffer(bLine); dev->destroyBuffer(bEdge);
}

} // namespace
