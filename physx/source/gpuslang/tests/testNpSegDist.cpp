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

// Verification of npSegDist.slang: sdPointSegment (distancePointSegmentSquared)
// and sdSegmentSegment (distanceSegmentSegmentSquared) — the closest-point
// distance primitives.  All outputs (squared distance and the clamped
// parameters s/t) are continuous, so they are compared with almostEqual.  The
// segments are generated non-degenerate (direction magnitude bounded away from
// zero) so the eps-gated reciprocals and the parallel fallback never straddle
// their branch boundaries; parallel segment pairs are included to exercise the
// denom<eps fallback path.

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

struct V3 { float x, y, z; };
struct F4 { float x, y, z, w; };
struct PsIn { F4 a, b, p; };
struct PsOut { F4 r; };
struct SsIn { F4 p1, d1, p2, d2; };
struct SsOut { F4 r; };
struct SdParams { uint32_t numTests, pad0, pad1, pad2; };

static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 sub3(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
// P9.86f: clampf(v, 0, 1) is exactly PxClamp(v, 0, 1) = PxMin(1, PxMax(0, v)) with the float selects of
// PxUnixMathIntrinsics.h:116-125 (-0 and NaN pass through).  gSdVariant replays one pre-P9.86f port clamp
// as GLSL FClamp on llvmpipe (FMin(FMax(v, +0), 1): +-0 -> +0, NaN -> 0): 1 = the point-segment tValue
// (sphereCollision.cuh:98), 2 = s0, 3 = t2, 4 = s2 (distanceSegmentSegment.cuh:62/:76/:81).
static int gSdVariant = 0;
static float fclampLp01(float v) { const float m = (v > 0.f) ? v : 0.f; return (m < 1.f) ? m : 1.f; }
static float clamp01(float v, int site) { return (gSdVariant == site) ? fclampLp01(v) : clampf(v, 0.0f, 1.0f); }
static uint32_t f2uSd(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static F4 f4(V3 v) { return { v.x, v.y, v.z, 0.0f }; }

static float refPointSeg(V3 a, V3 b, V3 p, float& param)
{
    const V3 ap = sub3(p, a), ab = sub3(b, a);
    const float nom = dot3(ap, ab), denom = dot3(ab, ab);
    const float tValue = clamp01(nom / denom, 1);
    const float t = denom > 0.0f ? tValue : 0.0f;
    const V3 v = { ap.x - ab.x * t, ap.y - ab.y * t, ap.z - ab.z * t };
    param = t;
    return dot3(v, v);
}

static void refSegSeg(V3 p1, V3 d1, V3 p2, V3 d2, float& s, float& t)
{
    const float eps = 1e-6f;
    const V3 r = sub3(p1, p2);
    const float a = dot3(d1, d1), e = dot3(d2, d2), b = dot3(d1, d2), c = dot3(d1, r);
    const float aRecip = a > eps ? (1.0f / a) : 0.0f;
    const float eRecip = e > eps ? (1.0f / e) : 0.0f;
    const float f = dot3(d2, r);
    const float denom = a * e - b * b;
    const float temp = b * f - c * e;
    const float s0 = clamp01(temp / denom, 2);
    const float sTmp = eps > denom ? 0.5f : s0;
    const float tTmp = (b * sTmp + f) * eRecip;
    const float t2 = clamp01(tTmp, 3);
    const float comp = (b * t2 - c) * aRecip;
    const float s2 = clamp01(comp, 4);
    s = s2; t = t2;
}

PXG_TEST(npSegDist_pointSegment)
{
    std::mt19937 rng(67011);
    const uint32_t N = 12000;
    std::vector<PsIn> in(N);
    std::vector<float> refSq(N), refT(N);
    std::uniform_real_distribution<float> uc(-3.0f, 3.0f);
    std::uniform_real_distribution<float> ud(0.5f, 2.5f);
    std::uniform_real_distribution<float> u11(-1.0f, 1.0f);
    size_t clampLo = 0, clampHi = 0, interior = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const V3 a = { uc(rng), uc(rng), uc(rng) };
        // direction magnitude bounded away from zero
        V3 dir; float m; do { dir = { u11(rng), u11(rng), u11(rng) }; m = std::sqrt(dot3(dir, dir)); } while (m < 0.3f);
        const float dl = ud(rng);
        const V3 b = { a.x + dir.x / m * dl, a.y + dir.y / m * dl, a.z + dir.z / m * dl };
        const V3 p = { uc(rng), uc(rng), uc(rng) };
        in[i] = { f4(a), f4(b), f4(p) };
        refSq[i] = refPointSeg(a, b, p, refT[i]);
        if (refT[i] <= 0.0f) ++clampLo; else if (refT[i] >= 1.0f) ++clampHi; else ++interior;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<PsOut> outInit(N);
    std::vector<SsIn> ssDummy(N);
    std::vector<SsOut> ssoDummy(N);
    PxgBufferHandle bPs = createBufferFrom(dev, in);
    PxgBufferHandle bPsOut = createBufferFrom(dev, outInit);
    PxgBufferHandle bSs = createBufferFrom(dev, ssDummy);
    PxgBufferHandle bSsOut = createBufferFrom(dev, ssoDummy);

    SdParams prm = { N, 0, 0, 0 };
    Kernel k("npSegDist.spv", "sdPointSegment",
             { (uint32_t)sizeof(PsIn), (uint32_t)sizeof(PsOut), (uint32_t)sizeof(SsIn), (uint32_t)sizeof(SsOut) }, sizeof(SdParams));
    k.dispatch(GROUPS, { bPs, bPsOut, bSs, bSsOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<PsOut>(dev, bPsOut, N);
    size_t mmSq = 0, mmT = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (!almostEqual(g[i].r.x, refSq[i], 1e-4f, 1e-4f)) ++mmSq;
        if (!almostEqual(g[i].r.y, refT[i], 1e-5f, 1e-5f)) ++mmT;
    }
    PXG_CHECK_MSG(mmSq == 0, "sdPointSegment: %zu sqDist mismatches", mmSq);
    PXG_CHECK_MSG(mmT == 0, "sdPointSegment: %zu param mismatches", mmT);
    PXG_CHECK_MSG(clampLo > 100 && clampHi > 100 && interior > 1000,
                  "sdPointSegment: coverage thin (lo=%zu hi=%zu int=%zu)", clampLo, clampHi, interior);

    dev->destroyBuffer(bPs); dev->destroyBuffer(bPsOut); dev->destroyBuffer(bSs); dev->destroyBuffer(bSsOut);
}

PXG_TEST(npSegDist_segmentSegment)
{
    std::mt19937 rng(67029);
    const uint32_t N = 16000;
    std::vector<SsIn> in(N);
    std::vector<float> refS(N), refT(N);
    std::vector<uint8_t> skip(N, 0);
    std::uniform_real_distribution<float> uc(-3.0f, 3.0f);
    std::uniform_real_distribution<float> u11(-1.0f, 1.0f);
    std::uniform_real_distribution<float> uk(-2.0f, 2.0f);
    size_t parallelC = 0, generalC = 0, skipped = 0;
    auto randDir = [&](float lenLo, float lenHi) {
        V3 d; float m; do { d = { u11(rng), u11(rng), u11(rng) }; m = std::sqrt(dot3(d, d)); } while (m < 0.3f);
        std::uniform_real_distribution<float> ul(lenLo, lenHi); const float L = ul(rng);
        return V3{ d.x / m * L, d.y / m * L, d.z / m * L };
    };
    for (uint32_t i = 0; i < N; ++i)
    {
        const V3 p1 = { uc(rng), uc(rng), uc(rng) };
        const V3 p2 = { uc(rng), uc(rng), uc(rng) };
        V3 d1, d2;
        const bool parallel = (rng() % 4u == 0u);
        if (parallel)
        {
            // small-magnitude parallel pair: denom = a*e - b*b cancels to ~0 with
            // negligible error (a*e is tiny), robustly below eps on both GPU and CPU,
            // so the parallel-fallback branch is exercised without a boundary flip.
            d1 = randDir(0.05f, 0.2f);
            float k; do { k = uk(rng); } while (std::fabs(k) < 0.2f);
            d2 = { d1.x * k, d1.y * k, d1.z * k };
            ++parallelC;
        }
        else
        {
            d1 = randDir(0.5f, 2.5f);
            d2 = randDir(0.5f, 2.5f);
            // skip the ill-conditioned near-parallel-by-chance band where denom is
            // small enough that temp/denom rounds differently on GPU vs CPU
            // (conservative: 1e-1 covers the worst near-parallel/large-magnitude corner)
            const float a = dot3(d1, d1), e = dot3(d2, d2), b = dot3(d1, d2);
            const float denom = a * e - b * b;
            if (denom < 1e-1f) { skip[i] = 1; ++skipped; } else ++generalC;
        }
        in[i] = { f4(p1), f4(d1), f4(p2), f4(d2) };
        refSegSeg(p1, d1, p2, d2, refS[i], refT[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<PsIn> psDummy(N);
    std::vector<PsOut> psoDummy(N);
    std::vector<SsOut> outInit(N);
    PxgBufferHandle bPs = createBufferFrom(dev, psDummy);
    PxgBufferHandle bPsOut = createBufferFrom(dev, psoDummy);
    PxgBufferHandle bSs = createBufferFrom(dev, in);
    PxgBufferHandle bSsOut = createBufferFrom(dev, outInit);

    SdParams prm = { N, 0, 0, 0 };
    Kernel k("npSegDist.spv", "sdSegmentSegment",
             { (uint32_t)sizeof(PsIn), (uint32_t)sizeof(PsOut), (uint32_t)sizeof(SsIn), (uint32_t)sizeof(SsOut) }, sizeof(SdParams));
    k.dispatch(GROUPS, { bPs, bPsOut, bSs, bSsOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<SsOut>(dev, bSsOut, N);
    size_t mmS = 0, mmT = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (skip[i]) continue;
        if (!almostEqual(g[i].r.x, refS[i], 1e-4f, 1e-4f)) ++mmS;
        if (!almostEqual(g[i].r.y, refT[i], 1e-4f, 1e-4f)) ++mmT;
    }
    PXG_CHECK_MSG(mmS == 0, "sdSegmentSegment: %zu s mismatches", mmS);
    PXG_CHECK_MSG(mmT == 0, "sdSegmentSegment: %zu t mismatches", mmT);
    PXG_CHECK_MSG(parallelC > N / 8 && generalC > N / 4,
                  "sdSegmentSegment: coverage thin (parallel=%zu general=%zu skip=%zu)", parallelC, generalC, skipped);

    dev->destroyBuffer(bPs); dev->destroyBuffer(bPsOut); dev->destroyBuffer(bSs); dev->destroyBuffer(bSsOut);
}

// P9.86f DIRECTED SIGNED ZERO / NaN through the four PxClamp selects, compared bitwise.  Every value is
// exact binary (no rounding, so contraction cannot move a bit):
//   P0: a = 0, b = (1,0,0), p = (-0,-1,-1): ap.ab = (-0) + (-0) + (-0) = -0 -> param -0 (FClamp +0)
//   P1: p = (inf,-inf,0), b = (1,1,0): ap.ab = inf - inf = NaN -> param NaN (FClamp 0)
//   S0: d1 = (1,0,0), d2 = (-0,1,0), r = p1 - p2 = (+0,-0,-1): b = +0, c = +0, f = -0 -> temp = -0,
//       s0 = -0, tTmp = +0*-0 + -0 = -0, comp = +0*-0 - +0 = -0: (s,t) = (-0,-0); the s0 twin gives +0,+0
//   S1: d2 = (2^-13,0,0) (eRecip = 0, parallel): tTmp = (2^-14 - 2^-13) * 0 = -0 -> t = -0, s = 1
//   S2: d1 = (2^-13,0,0) (aRecip = 0, parallel), r.x = 2: t = 1, comp = (2^-13 - 2^-12) * 0 = -0 -> s = -0
PXG_TEST(npSegDist_zeroSign)
{
    const float NZ = -0.0f, H = 1.220703125e-4f;   // 2^-13
    const V3 pa[2] = { { 0, 0, 0 }, { 0, 0, 0 } }, pb[2] = { { 1, 0, 0 }, { 1, 1, 0 } };
    const V3 pp[2] = { { NZ, -1, -1 }, { INFINITY, -INFINITY, 0 } };
    const V3 s1[3] = { { 0, NZ, 0 }, { 0, 0, 0 }, { 0, 0, 0 } }, sd1[3] = { { 1, 0, 0 }, { 1, 0, 0 }, { H, 0, 0 } };
    const V3 s2[3] = { { 0, 0, 1 }, { 1, 0, 0 }, { -2, 0, 0 } }, sd2[3] = { { NZ, 1, 0 }, { H, 0, 0 }, { 1, 0, 0 } };
    // expected CUDA bits: point-segment param; segment-segment (s, t)
    const uint32_t wantP[2] = { 0x80000000u, 0x7FC00000u };
    const uint32_t wantS[3] = { 0x80000000u, 0x3F800000u, 0x80000000u }, wantT[3] = { 0x80000000u, 0x80000000u, 0x3F800000u };
    // which twin flips which case: tValue -> P0,P1; s0 -> S0; t2 -> S0,S1; s2 -> S0,S2
    size_t twinHits[5] = {};
    std::vector<PsIn> pin(2); std::vector<float> refP(2), refD(2);
    for (uint32_t i = 0; i < 2; ++i)
    {
        pin[i] = { f4(pa[i]), f4(pb[i]), f4(pp[i]) };
        refD[i] = refPointSeg(pa[i], pb[i], pp[i], refP[i]);
        PXG_CHECK_MSG(std::isnan(refP[i]) ? (wantP[i] == 0x7FC00000u) : (f2uSd(refP[i]) == wantP[i]),
                      "pointSeg directed %u: CUDA mirror param %08x, want %08x", i, f2uSd(refP[i]), wantP[i]);
        for (int v = 1; v <= 4; ++v)
        {
            gSdVariant = v; float tw; refPointSeg(pa[i], pb[i], pp[i], tw); gSdVariant = 0;
            if (f2uSd(tw) != f2uSd(refP[i])) ++twinHits[v];
        }
    }
    std::vector<SsIn> sin(3); std::vector<float> refS(3), refT(3);
    for (uint32_t i = 0; i < 3; ++i)
    {
        sin[i] = { f4(s1[i]), f4(sd1[i]), f4(s2[i]), f4(sd2[i]) };
        refSegSeg(s1[i], sd1[i], s2[i], sd2[i], refS[i], refT[i]);
        PXG_CHECK_MSG(f2uSd(refS[i]) == wantS[i] && f2uSd(refT[i]) == wantT[i],
                      "segSeg directed %u: CUDA mirror (s,t) %08x %08x, want %08x %08x", i, f2uSd(refS[i]), f2uSd(refT[i]), wantS[i], wantT[i]);
        for (int v = 1; v <= 4; ++v)
        {
            gSdVariant = v; float ts, tt; refSegSeg(s1[i], sd1[i], s2[i], sd2[i], ts, tt); gSdVariant = 0;
            if (f2uSd(ts) != f2uSd(refS[i]) || f2uSd(tt) != f2uSd(refT[i])) ++twinHits[v];
        }
    }
    PXG_CHECK_MSG(twinHits[1] == 2 && twinHits[2] == 1 && twinHits[3] == 2 && twinHits[4] == 2,
                  "npSegDist directed: twin coverage tValue %zu s0 %zu t2 %zu s2 %zu (want 2 1 2 2)",
                  twinHits[1], twinHits[2], twinHits[3], twinHits[4]);

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<PsOut> poInit(2);
    std::vector<SsOut> soInit(3);
    PxgBufferHandle bPs = createBufferFrom(dev, pin), bPsOut = createBufferFrom(dev, poInit);
    PxgBufferHandle bSs = createBufferFrom(dev, sin), bSsOut = createBufferFrom(dev, soInit);
    const std::vector<uint32_t> strides = { (uint32_t)sizeof(PsIn), (uint32_t)sizeof(PsOut), (uint32_t)sizeof(SsIn), (uint32_t)sizeof(SsOut) };
    SdParams prmP = { 2, 0, 0, 0 }, prmS = { 3, 0, 0, 0 };
    Kernel kp("npSegDist.spv", "sdPointSegment", strides, sizeof(SdParams));
    kp.dispatch(1, { bPs, bPsOut, bSs, bSsOut }, prmP);
    Kernel ks("npSegDist.spv", "sdSegmentSegment", strides, sizeof(SdParams));
    ks.dispatch(1, { bPs, bPsOut, bSs, bSsOut }, prmS);
    dev->flush();
    const auto gp = downloadBuffer<PsOut>(dev, bPsOut, 2);
    const auto gs = downloadBuffer<SsOut>(dev, bSsOut, 3);
    for (uint32_t i = 0; i < 2; ++i)
    {
        const bool ok = std::isnan(refP[i]) ? std::isnan(gp[i].r.y) : (f2uSd(gp[i].r.y) == f2uSd(refP[i]));
        PXG_CHECK_MSG(ok, "sdPointSegment directed %u: GPU param %08x, CUDA %08x", i, f2uSd(gp[i].r.y), f2uSd(refP[i]));
    }
    for (uint32_t i = 0; i < 3; ++i)
        PXG_CHECK_MSG(f2uSd(gs[i].r.x) == f2uSd(refS[i]) && f2uSd(gs[i].r.y) == f2uSd(refT[i]),
                      "sdSegmentSegment directed %u: GPU (s,t) %08x %08x, CUDA %08x %08x",
                      i, f2uSd(gs[i].r.x), f2uSd(gs[i].r.y), f2uSd(refS[i]), f2uSd(refT[i]));
    dev->destroyBuffer(bPs); dev->destroyBuffer(bPsOut); dev->destroyBuffer(bSs); dev->destroyBuffer(bSsOut);
}

} // namespace
