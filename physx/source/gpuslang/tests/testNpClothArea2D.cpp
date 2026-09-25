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

// Verification of npClothArea2D.slang (clothArea2D): the XPBD area-conservation
// constraint areaConstraint_F2X2 (FEMClothUtil.cuh L376-419, the active `#if 1`
// branch), against an exact CPU transcription.  Outputs bitwise mod +-0 (eqFZ).

#include "pxgTestFramework.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct AaIn  { F4 x01x02, dx0dx1, dx2LamAlpha, massArea; };
struct AaOut { F4 dx0dx1, dx2Lam; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed

static const float FEMCLOTH_THRESHOLD = 1.0e-14f;

struct V2 { float x, y; };
static V2 add2(V2 a, V2 b) { return V2{ a.x + b.x, a.y + b.y }; }
static V2 scl2(float s, V2 a) { return V2{ s * a.x, s * a.y }; }
static float magSq2(V2 v) { return v.x * v.x + v.y * v.y; }

static void refArea(const AaIn& s, AaOut& o)
{
    const V2 x01 = V2{ s.x01x02.x, s.x01x02.y };
    const V2 x02 = V2{ s.x01x02.z, s.x01x02.w };
    V2 dx0 = V2{ s.dx0dx1.x, s.dx0dx1.y };
    V2 dx1 = V2{ s.dx0dx1.z, s.dx0dx1.w };
    V2 dx2 = V2{ s.dx2LamAlpha.x, s.dx2LamAlpha.y };
    float lambda = s.dx2LamAlpha.z;
    const float alphaTilde = s.dx2LamAlpha.w;
    const float massInv0 = s.massArea.x, massInv1 = s.massArea.y, massInv2 = s.massArea.z;
    const float area = s.massArea.w;

    const float x01CrossX02 = x01.x * x02.y - x01.y * x02.x;
    const float undeformedAreaInv = 1.0f / area;

    const float C = 0.5f * x01CrossX02 * undeformedAreaInv - 1.0f;

    const V2 grad1 = V2{ 0.5f * undeformedAreaInv * x02.y, -0.5f * undeformedAreaInv * x02.x };
    const V2 grad2 = V2{ -0.5f * undeformedAreaInv * x01.y, 0.5f * undeformedAreaInv * x01.x };
    const V2 grad0 = V2{ -grad1.x - grad2.x, -grad1.y - grad2.y };

    const float denom = (massInv0 * magSq2(grad0) + massInv1 * magSq2(grad1) + massInv2 * magSq2(grad2)) + alphaTilde;
    float deltaLambda;
    if (denom < FEMCLOTH_THRESHOLD) deltaLambda = 0.0f;
    else deltaLambda = (-C - alphaTilde * lambda) / denom;

    lambda = lambda + deltaLambda;

    dx0 = add2(dx0, scl2(massInv0 * deltaLambda, grad0));
    dx1 = add2(dx1, scl2(massInv1 * deltaLambda, grad1));
    dx2 = add2(dx2, scl2(massInv2 * deltaLambda, grad2));

    o.dx0dx1 = F4{ dx0.x, dx0.y, dx1.x, dx1.y };
    o.dx2Lam = F4{ dx2.x, dx2.y, lambda, 0.0f };
}

PXG_TEST(npClothArea2D)
{
    std::mt19937 rng(220001);
    const uint32_t N = 5000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<AaIn> in(N);
    std::vector<AaOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        AaIn s{};
        s.x01x02 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f) };
        s.dx0dx1 = F4{ rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-1.f, 1.f) };
        float lambda = rf(-2.f, 2.f);
        float alphaTilde = rf(0.f, 2.f);
        float mi0 = rf(0.f, 3.f), mi1 = rf(0.f, 3.f), mi2 = rf(0.f, 3.f);
        // area: both signs (undeformedAreaInv = 1/area) but away from 0
        float area = ((i % 2u) == 0u) ? rf(0.2f, 5.f) : rf(-5.f, -0.2f);

        // denom-guard stripe: massInv 0 and tiny alphaTilde -> denom < 1e-14 -> dl 0
        if ((i % 11u) == 6u)
        {
            mi0 = mi1 = mi2 = 0.f;
            alphaTilde = 1.0e-15f;
        }
        // rest-configuration stripe: C ~ 0 does NOT gate anything here (no C guard),
        // still a valid arithmetic path; keep a plain deformed case otherwise.

        s.dx2LamAlpha = F4{ rf(-1.f, 1.f), rf(-1.f, 1.f), lambda, alphaTilde };
        s.massArea = F4{ mi0, mi1, mi2, area };
        in[i] = s;
        refArea(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<AaOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npClothArea2D.spv", "clothArea2D",
             { (uint32_t)sizeof(AaIn), (uint32_t)sizeof(AaOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<AaOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* rfp = &ref[i].dx0dx1.x;
        const float* qfp = &g[i].dx0dx1.x;
        const size_t nf = sizeof(AaOut) / 4;
        bool ok = true;
        for (size_t x = 0; x < nf && ok; ++x) ok = eqFZ(rfp[x], qfp[x]);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u lamR %g lamG %g dx0R (%g %g) dx0G (%g %g)\n", i,
                            ref[i].dx2Lam.z, g[i].dx2Lam.z,
                            ref[i].dx0dx1.x, ref[i].dx0dx1.y, g[i].dx0dx1.x, g[i].dx0dx1.y);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "clothArea2D outputs bitwise-identical (mod +-0)");
}

}  // namespace
