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
static F4 f4(V3 v) { return { v.x, v.y, v.z, 0.0f }; }

static float refPointSeg(V3 a, V3 b, V3 p, float& param)
{
    const V3 ap = sub3(p, a), ab = sub3(b, a);
    const float nom = dot3(ap, ab), denom = dot3(ab, ab);
    const float tValue = clampf(nom / denom, 0.0f, 1.0f);
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
    const float s0 = clampf(temp / denom, 0.0f, 1.0f);
    const float sTmp = eps > denom ? 0.5f : s0;
    const float tTmp = (b * sTmp + f) * eRecip;
    const float t2 = clampf(tTmp, 0.0f, 1.0f);
    const float comp = (b * t2 - c) * aRecip;
    const float s2 = clampf(comp, 0.0f, 1.0f);
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

} // namespace
