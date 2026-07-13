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

// Verification of npCvxCollideDecide.slang: the collide() post-GJK decision glue
// (eDISTANT early-out, anomaly = eCLOSE && closestPointDir.dot(direction) <
// 0.999, EPA gate !separated || anomaly, degenerate re-run) with injected
// GJK/EPA results, and the sphere-path witness store (sep = (x+y)+z of
// per-component products, pointA = 0, pointB = witnessB, normalPen =
// (-direction, sep)).  Exact CPU transcription; flags exact, floats to
// tolerance, anomaly 0.999 boundary margin-skipped.

#include "pxgTestFramework.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct V3 { float x, y, z; };
struct DecIn { F4 closestPointDir, direction, witnessA, witnessB; uint32_t gjkResult, epaResult1, degenerate, epaResult2; };
struct DecOut { F4 mA0, mB0, mNp0; uint32_t proceed, ranEpa, ranRetry, separated, numManifold, pad0, pad1, pad2; };

const float TIE = 2e-4f;

static inline V3 v3(const F4& f) { return V3{ f.x, f.y, f.z }; }
static inline V3 mul(const V3& a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; }
static inline float dot3(const V3& a, const V3& b) { return (a.x * b.x + a.y * b.y) + a.z * b.z; }
static inline V3 unit(const V3& a) { float l = std::sqrt(dot3(a, a)); return l > 0 ? mul(a, 1.f / l) : a; }

static void refDecide(const DecIn& s, DecOut& o, bool& marg)
{
    o = DecOut{};
    if (s.gjkResult == 0) return; // eDISTANT
    o.proceed = 1;

    const V3 cpd = v3(s.closestPointDir), dir = v3(s.direction);
    const float dp = dot3(cpd, dir);
    if (std::fabs(dp - 0.999f) < TIE) marg = true;
    const bool anomaly = (s.gjkResult == 1) && (dp < 0.999f);
    bool separated = (s.gjkResult == 1);

    if (!separated || anomaly)
    {
        o.ranEpa = 1;
        separated = (s.epaResult1 == 1);
        if (s.degenerate != 0)
        {
            o.ranRetry = 1;
            separated = (s.epaResult2 == 1);
        }
    }
    o.separated = separated ? 1u : 0u;

    const V3 wA = v3(s.witnessA), wB = v3(s.witnessB);
    const float sep = ((wB.x - wA.x) * dir.x + (wB.y - wA.y) * dir.y) + (wB.z - wA.z) * dir.z;
    o.mA0 = F4{ 0, 0, 0, 0 };
    o.mB0 = F4{ wB.x, wB.y, wB.z, 0 };
    o.mNp0 = F4{ -dir.x, -dir.y, -dir.z, sep };
    o.numManifold = 1;
}

static bool eqF4(const F4& a, const F4& b) { return almostEqual(a.x, b.x, 1e-5f, 1e-6f) && almostEqual(a.y, b.y, 1e-5f, 1e-6f) && almostEqual(a.z, b.z, 1e-5f, 1e-6f) && almostEqual(a.w, b.w, 1e-4f, 1e-5f); }

PXG_TEST(npCvxCollideDecide_decide)
{
    std::mt19937 rng(149001);
    const uint32_t N = 12000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<DecIn> in(N);
    std::vector<uint8_t> skip(N, 0);
    std::vector<DecOut> ref(N);
    size_t nSkip = 0, cDistant = 0, cAnomaly = 0, cEpa = 0, cRetry = 0, cNoEpa = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        DecIn s{};
        // gjkResult: 0 eDISTANT / 1 eCLOSE / 2 eOVERLAP
        s.gjkResult = i % 3;
        s.epaResult1 = rng() % 3;
        s.degenerate = (i % 4 == 0) ? 1u : 0u;
        s.epaResult2 = rng() % 3;

        const V3 dir = unit(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) });
        // closestPointDir: half aligned with dir (dp ~ 1 > 0.999), half random
        V3 cpd;
        if (i % 2 == 0)
            cpd = dir; // exact alignment -> dp == |dir|^2 ~ 1
        else
            cpd = unit(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) });
        s.direction = { dir.x, dir.y, dir.z, 0 };
        s.closestPointDir = { cpd.x, cpd.y, cpd.z, 0 };
        s.witnessA = { rf(-2, 2), rf(-2, 2), rf(-2, 2), 0 };
        s.witnessB = { rf(-2, 2), rf(-2, 2), rf(-2, 2), 0 };
        in[i] = s;

        bool marg = false;
        refDecide(s, ref[i], marg);
        skip[i] = marg ? 1u : 0u;
        if (marg) { ++nSkip; continue; }
        if (!ref[i].proceed) ++cDistant;
        else if (ref[i].ranRetry) ++cRetry;
        else if (ref[i].ranEpa) ++cEpa;
        else ++cNoEpa;
        if (ref[i].proceed && s.gjkResult == 1 && ref[i].ranEpa) ++cAnomaly;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<DecOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct DecParams { uint32_t numTests; } prm = { N };
    Kernel k("npCvxCollideDecide.spv", "cvxCollideDecide",
             { (uint32_t)sizeof(DecIn), (uint32_t)sizeof(DecOut) }, sizeof(DecParams));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<DecOut>(dev, bOut, N);
    size_t mmF = 0, mmS = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (skip[i]) continue;
        if (g[i].proceed != ref[i].proceed || g[i].ranEpa != ref[i].ranEpa ||
            g[i].ranRetry != ref[i].ranRetry || g[i].separated != ref[i].separated ||
            g[i].numManifold != ref[i].numManifold) { ++mmF; continue; }
        if (!eqF4(ref[i].mA0, g[i].mA0) || !eqF4(ref[i].mB0, g[i].mB0) || !eqF4(ref[i].mNp0, g[i].mNp0)) ++mmS;
    }
    PXG_CHECK_MSG(mmF == 0, "collideDecide: %zu flag mismatches", mmF);
    PXG_CHECK_MSG(mmS == 0, "collideDecide: %zu sphere-store mismatches", mmS);
    PXG_CHECK_MSG(nSkip < N / 4, "collideDecide: too many marginal (%zu/%u)", nSkip, N);
    PXG_CHECK_MSG(cDistant > N / 10 && cEpa > N / 20 && cRetry > N / 20 && cNoEpa > N / 20 && cAnomaly > N / 40,
                  "collideDecide: coverage thin (distant %zu / epa %zu / retry %zu / noEpa %zu / anomaly %zu)",
                  cDistant, cEpa, cRetry, cNoEpa, cAnomaly);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
