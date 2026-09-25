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

// Verification of npArtiAccumLinkImp.slang (accumLinkImp): fold a scalar
// constraint impulse into parent/child velocity-delta, velocity, and impulse
// accumulators (accumulateLinkImpulsesAndLinkVelocities, internalConstraints2.cu
// L1554-1575), against an exact CPU transcription.  Outputs bitwise mod +-0.

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

struct AiIn
{
    F4 deltaVATop, deltaVABot, deltaVBTop, deltaVBBot, row0Top, row0Bot, row1Top, row1Bot,
       parentDeltaVTop, parentDeltaVBot, deltaVTop, deltaVBot, parentVTop, parentVBot,
       childVTop, childVBot, parentImpTop, parentImpBot, impulseTop, impulseBot, misc;
};
struct AiOut
{
    F4 parentDeltaVTop, parentDeltaVBot, deltaVTop, deltaVBot, parentVTop, parentVBot,
       childVTop, childVBot, parentImpTop, parentImpBot, impulseTop, impulseBot;
};

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed

static F4 addS(const F4& a, const F4& b, float s) { return F4{ a.x + b.x*s, a.y + b.y*s, a.z + b.z*s, 0.f }; }

// exact transcription of accumulateLinkImpulsesAndLinkVelocities
static void refAccum(const AiIn& s, AiOut& o)
{
    const float deltaF = s.misc.x;
    // pDelta = deltaVA * -deltaF; cDelta = deltaVB * -deltaF
    const F4 pTop = F4{ s.deltaVATop.x*-deltaF, s.deltaVATop.y*-deltaF, s.deltaVATop.z*-deltaF, 0.f };
    const F4 pBot = F4{ s.deltaVABot.x*-deltaF, s.deltaVABot.y*-deltaF, s.deltaVABot.z*-deltaF, 0.f };
    const F4 cTop = F4{ s.deltaVBTop.x*-deltaF, s.deltaVBTop.y*-deltaF, s.deltaVBTop.z*-deltaF, 0.f };
    const F4 cBot = F4{ s.deltaVBBot.x*-deltaF, s.deltaVBBot.y*-deltaF, s.deltaVBBot.z*-deltaF, 0.f };

    auto add = [](const F4& a, const F4& b) { return F4{ a.x+b.x, a.y+b.y, a.z+b.z, 0.f }; };
    o.parentDeltaVTop = add(s.parentDeltaVTop, pTop);
    o.parentDeltaVBot = add(s.parentDeltaVBot, pBot);
    o.deltaVTop = add(s.deltaVTop, cTop);
    o.deltaVBot = add(s.deltaVBot, cBot);
    o.parentVTop = add(s.parentVTop, pTop);
    o.parentVBot = add(s.parentVBot, pBot);
    o.childVTop = add(s.childVTop, cTop);
    o.childVBot = add(s.childVBot, cBot);

    // parentImp += row0 * deltaF;  impulse -= row1 * deltaF
    o.parentImpTop = addS(s.parentImpTop, s.row0Top, deltaF);
    o.parentImpBot = addS(s.parentImpBot, s.row0Bot, deltaF);
    o.impulseTop = addS(s.impulseTop, s.row1Top, -deltaF);
    o.impulseBot = addS(s.impulseBot, s.row1Bot, -deltaF);
}

PXG_TEST(npArtiAccumLinkImp)
{
    std::mt19937 rng(205001);
    const uint32_t N = 3000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto rv = [&](float a, float b) { return F4{ rf(a, b), rf(a, b), rf(a, b), 0.f }; };

    std::vector<AiIn> in(N);
    std::vector<AiOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        AiIn s{};
        F4* p = &s.deltaVATop;
        for (int b = 0; b < 20; ++b) p[b] = rv(-3, 3);   // all 20 spatial-vector halves
        s.misc = F4{ rf(-4.f, 4.f), 0.f, 0.f, 0.f };
        in[i] = s;
        refAccum(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<AiOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npArtiAccumLinkImp.spv", "accumLinkImp",
             { (uint32_t)sizeof(AiIn), (uint32_t)sizeof(AiOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<AiOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* rfp = &ref[i].parentDeltaVTop.x;
        const float* qfp = &g[i].parentDeltaVTop.x;
        const size_t nf = sizeof(AiOut) / 4;
        bool ok = true;
        for (size_t x = 0; x < nf && ok; ++x) ok = eqFZ(rfp[x], qfp[x]);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u impulseTopR (%g %g %g) G (%g %g %g)\n", i, ref[i].impulseTop.x,
                            ref[i].impulseTop.y, ref[i].impulseTop.z, g[i].impulseTop.x, g[i].impulseTop.y, g[i].impulseTop.z);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "accumLinkImp outputs bitwise-identical (mod +-0)");
}

}  // namespace
