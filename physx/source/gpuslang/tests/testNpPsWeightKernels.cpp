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

// Verification of npPsWeightKernels.slang (psWeightKernels): the particle-system
// polynomial weight helpers cube/quart/oct (particlesystem.cu L110-117) and
// WCohesion (L3397-3401), against an exact CPU transcription.  Bitwise mod +-0.

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
struct WkIn  { F4 a, wc; };
struct WkOut { F4 r; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b) || (a == 0.f && b == 0.f); }

static float cube(float x) { return x * x * x; }
static float quart(float x) { return x * x * x * x; }
static float sqr(float x) { return x * x; }
static float oct(float x)
{
    float x2 = x * x;
    float x4 = x2 * x2;
    return x4 * x4;
}
static float WCohesion(float x, float k1, float k2, float kInvR)
{
    float w = x * kInvR;
    return k1 * cube(w) + k2 * sqr(w) - 1.0f;
}

static void refWk(const WkIn& s, WkOut& o)
{
    const float x = s.a.x;
    o.r = F4{ cube(x), quart(x), oct(x), WCohesion(s.wc.x, s.wc.y, s.wc.z, s.wc.w) };
}

PXG_TEST(npPsWeightKernels)
{
    std::mt19937 rng(242001);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<WkIn> in(N);
    std::vector<WkOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        WkIn s{};
        s.a = F4{ rf(-3.f, 3.f), 0.f, 0.f, 0.f };
        s.wc = F4{ rf(0.f, 2.f), rf(-4.f, 4.f), rf(-4.f, 4.f), rf(0.1f, 3.f) };  // x, kCohesion1, kCohesion2, kInvRadius
        // negative-x stripe (odd powers cube/quart keep sign structure)
        if ((i % 7u) == 2u) s.a.x = rf(-3.f, -0.5f);
        // zero-x stripe
        if ((i % 13u) == 5u) s.a.x = 0.f;
        in[i] = s;
        refWk(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<WkOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npPsWeightKernels.spv", "psWeightKernels",
             { (uint32_t)sizeof(WkIn), (uint32_t)sizeof(WkOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<WkOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        bool ok = eqFZ(ref[i].r.x, g[i].r.x) && eqFZ(ref[i].r.y, g[i].r.y)
               && eqFZ(ref[i].r.z, g[i].r.z) && eqFZ(ref[i].r.w, g[i].r.w);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u R (%g %g %g %g) G (%g %g %g %g)\n", i,
                            ref[i].r.x, ref[i].r.y, ref[i].r.z, ref[i].r.w,
                            g[i].r.x, g[i].r.y, g[i].r.z, g[i].r.w);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "psWeightKernels outputs bitwise-identical (mod +-0)");
}

}  // namespace
