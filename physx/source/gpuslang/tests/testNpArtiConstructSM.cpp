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

// Verification of npArtiConstructSM.slang (constructSM): Dy::SpatialMatrix
// constructSpatialMatrix(Is, stI) (DyFeatherstoneArticulationUtils.h L364-386),
// the block outer-product builder, against an exact CPU transcription.
// Outputs bitwise modulo the documented +-0 (eqFZ).

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

struct CsIn { F4 isTop, isBot, stiTop, stiBot; };
struct CsOut { F4 tlC0, tlC1, tlC2, trC0, trC1, trC2, blC0, blC1, blC2; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed

static F4 scale(const F4& v, float s) { return F4{ v.x * s, v.y * s, v.z * s, 0.f }; }

// exact transcription of constructSpatialMatrix(Is, stI)
static void refConstructSM(const CsIn& s, CsOut& o)
{
    o.tlC0 = scale(s.isTop, s.stiTop.x);
    o.tlC1 = scale(s.isTop, s.stiTop.y);
    o.tlC2 = scale(s.isTop, s.stiTop.z);
    o.trC0 = scale(s.isTop, s.stiBot.x);
    o.trC1 = scale(s.isTop, s.stiBot.y);
    o.trC2 = scale(s.isTop, s.stiBot.z);
    o.blC0 = scale(s.isBot, s.stiTop.x);
    o.blC1 = scale(s.isBot, s.stiTop.y);
    o.blC2 = scale(s.isBot, s.stiTop.z);
}

PXG_TEST(npArtiConstructSM)
{
    std::mt19937 rng(197001);
    const uint32_t N = 3000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto rv = [&](float a, float b) { return F4{ rf(a, b), rf(a, b), rf(a, b), 0.f }; };

    std::vector<CsIn> in(N);
    std::vector<CsOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        CsIn s{};
        s.isTop = rv(-3, 3); s.isBot = rv(-3, 3);
        s.stiTop = rv(-3, 3); s.stiBot = rv(-3, 3);
        in[i] = s;
        refConstructSM(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<CsOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npArtiConstructSM.spv", "constructSM",
             { (uint32_t)sizeof(CsIn), (uint32_t)sizeof(CsOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<CsOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* rfp = &ref[i].tlC0.x;
        const float* qfp = &g[i].tlC0.x;
        const size_t nf = sizeof(CsOut) / 4;
        bool ok = true;
        for (size_t x = 0; x < nf && ok; ++x) ok = eqFZ(rfp[x], qfp[x]);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u tlC0R (%g %g %g) G (%g %g %g)\n", i, ref[i].tlC0.x,
                            ref[i].tlC0.y, ref[i].tlC0.z, g[i].tlC0.x, g[i].tlC0.y, g[i].tlC0.z);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "constructSM outputs bitwise-identical (mod +-0)");
}

}  // namespace
