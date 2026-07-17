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

// Verification of npSphSpikyKernel.slang (sphSpikyKernel): the SPH spiky-kernel
// scalar helpers sqr/W/dWdx (particleSystem.cuh L58-68), against an exact CPU
// transcription.  Outputs bitwise mod +-0 (eqFZ).

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
struct SkIn  { F4 in; };
struct SkOut { F4 out; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b) || (a == 0.f && b == 0.f); }

static float sqr(float x) { return x * x; }
static float W(float x, float kSpiky1, float kInvRadius) { return kSpiky1 * sqr(1.0f - x * kInvRadius); }
static float dWdx(float x, float kSpiky2, float kInvRadius) { return -kSpiky2 * (1.0f - x * kInvRadius); }

static void refSph(const SkIn& s, SkOut& o)
{
    const float x = s.in.x;
    const float kSpiky1 = s.in.y, kSpiky2 = s.in.z, kInvRadius = s.in.w;
    o.out = F4{ sqr(x), W(x, kSpiky1, kInvRadius), dWdx(x, kSpiky2, kInvRadius), 0.f };
}

PXG_TEST(npSphSpikyKernel)
{
    std::mt19937 rng(225001);
    const uint32_t N = 5000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<SkIn> in(N);
    std::vector<SkOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        SkIn s{};
        // x is a distance in [0, radius]; kInvRadius = 1/radius so x*kInvRadius in [0,1].
        float x = rf(0.f, 3.f);
        float kInvRadius = rf(0.2f, 2.f);
        float kSpiky1 = rf(0.5f, 20.f), kSpiky2 = rf(0.5f, 20.f);
        // x==0 edge (1 - 0 = 1), and x*kInvRadius==1 edge (1 - 1 = 0 -> W=0, dWdx=0)
        if ((i % 7u) == 2u) x = 0.f;
        if ((i % 7u) == 4u) x = 1.0f / kInvRadius;   // makes (1 - x*kInvRadius) ~ 0
        s.in = F4{ x, kSpiky1, kSpiky2, kInvRadius };
        in[i] = s;
        refSph(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<SkOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npSphSpikyKernel.spv", "sphSpikyKernel",
             { (uint32_t)sizeof(SkIn), (uint32_t)sizeof(SkOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<SkOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        bool ok = eqFZ(ref[i].out.x, g[i].out.x) && eqFZ(ref[i].out.y, g[i].out.y) && eqFZ(ref[i].out.z, g[i].out.z);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u R (%g %g %g) G (%g %g %g)\n", i,
                            ref[i].out.x, ref[i].out.y, ref[i].out.z, g[i].out.x, g[i].out.y, g[i].out.z);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "sphSpikyKernel outputs bitwise-identical (mod +-0)");
}

}  // namespace
