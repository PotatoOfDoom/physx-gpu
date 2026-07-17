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

// Verification of npClothAtan.slang (clothAtan): the FEM-cloth approximate
// arctangent atanApprox / atan2Approx (FEMClothUtil.cuh L153-173), against an
// exact CPU transcription.  Outputs bitwise modulo the documented +-0 (eqFZ).

#include "pxgTestFramework.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct AtIn  { F4 misc; };
struct AtOut { F4 res; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b) || (a == 0.f && b == 0.f); }

static const float FEMCLOTH_PI = 3.14159265358979323846f;
static const float FEMCLOTH_HALF_PI = 1.57079632679489661923f;

static float atanApprox(float x)
{
    float x2 = x * x;
    return x * (0.99997726f + x2 * (-0.33262347f + x2 * (0.19354346f + x2 * (-0.11643287f + x2 * (0.05265332f + x2 * (-0.01172120f))))));
}
static float atan2Approx(float y, float x)
{
    bool swap = std::fabs(x) < std::fabs(y);
    float input = swap ? (x / y) : (y / x);
    float output = atanApprox(input);
    output = swap ? ((input >= 0.f ? FEMCLOTH_HALF_PI : -FEMCLOTH_HALF_PI) - output) : output;
    if (x < 0.f)
        output += (y >= 0.f ? FEMCLOTH_PI : -FEMCLOTH_PI);
    return output;
}

static void refAtan(const AtIn& s, AtOut& o)
{
    o.res = F4{ atanApprox(s.misc.x), atan2Approx(s.misc.y, s.misc.z), 0.f, 0.f };
}

PXG_TEST(npClothAtan)
{
    std::mt19937 rng(213001);
    const uint32_t N = 5000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<AtIn> in(N);
    std::vector<AtOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        AtIn s{};
        const float x = rf(-4.f, 4.f);         // atanApprox arg
        float yy = rf(-4.f, 4.f);
        float xx = rf(-4.f, 4.f);
        // keep xx away from zero so the non-swap y/x has no 0/0; swap branch uses x/y with |y|>|x|>0
        if (std::fabs(xx) < 0.3f) xx += (xx < 0.f ? -0.5f : 0.5f);
        if (std::fabs(yy) < 0.3f) yy += (yy < 0.f ? -0.5f : 0.5f);
        s.misc = F4{ x, yy, xx, 0.f };
        in[i] = s;
        refAtan(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<AtOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npClothAtan.spv", "clothAtan",
             { (uint32_t)sizeof(AtIn), (uint32_t)sizeof(AtOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<AtOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (!eqFZ(ref[i].res.x, g[i].res.x) || !eqFZ(ref[i].res.y, g[i].res.y))
        {
            if (mm < 5)
                std::printf("MM %u atan R %g G %g | atan2 R %g G %g (x=%g y=%g)\n", i, ref[i].res.x, g[i].res.x,
                            ref[i].res.y, g[i].res.y, in[i].misc.x, in[i].misc.y);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "clothAtan outputs bitwise-identical (mod +-0)");
}

}  // namespace
