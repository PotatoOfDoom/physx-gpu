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

// Verification of npArtiPropagateZA.slang (propZA): the prismatic/revolute
// (single-dof) branch of computePropagateSpatialInertia_ZA_ZIc
// (forwardDynamic2.cu L1420-1448), against an exact CPU transcription.
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

struct PpIn
{
    F4 saTop, saBot, isTop, isBot, zTop, zBot, zicTop, zicBot, zaTop, zaBot, zaiTop, zaiBot, misc;
};
struct PpOut
{
    F4 scalars, isInvDTop, isInvDBot, zaTop, zaBot, zaiTop, zaiBot,
       tlC0, tlC1, tlC2, trC0, trC1, trC2, blC0, blC1, blC2;
};

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed

static const uint32_t JF_NULL = 1u;

static float dot3(const F4& a, const F4& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
// spatial duality inner product a.innerProduct(b) = a.bottom.b.top + a.top.b.bottom
static float inner(const F4& aTop, const F4& aBot, const F4& bTop, const F4& bBot)
{
    return dot3(aBot, bTop) + dot3(aTop, bBot);
}
static F4 scaleF(const F4& v, float s) { return F4{ v.x * s, v.y * s, v.z * s, 0.f }; }
static F4 addF(const F4& a, const F4& b) { return F4{ a.x + b.x, a.y + b.y, a.z + b.z, 0.f }; }

// exact transcription of the single-dof branch of computePropagateSpatialInertia_ZA_ZIc
static void refPropZA(const PpIn& s, PpOut& o)
{
    const float armature = s.misc.x;
    const float jF0 = s.misc.y;
    const bool jFnull = f2u(s.misc.z) == JF_NULL;

    const float stIs = inner(s.saTop, s.saBot, s.isTop, s.isBot) + armature;
    const float iStIs = (stIs > 0.f) ? (1.f / stIs) : 0.f;

    const F4 isInvDTop = scaleF(s.isTop, iStIs);
    const F4 isInvDBot = scaleF(s.isBot, iStIs);

    // stI = (Is.bottom, Is.top)
    const F4 stiT = s.isBot;
    const F4 stiB = s.isTop;

    const float stZ = inner(s.saTop, s.saBot, s.zTop, s.zBot);
    const float stZInt = inner(s.saTop, s.saBot, s.zicTop, s.zicBot);

    const float qstZ = -stZ;
    const float qstZIcInternal = (jFnull ? 0.f : jF0) - stZInt;

    const F4 zaTop = addF(s.zaTop, scaleF(isInvDTop, qstZ));
    const F4 zaBot = addF(s.zaBot, scaleF(isInvDBot, qstZ));
    const F4 zaiTop = addF(s.zaiTop, scaleF(isInvDTop, qstZIcInternal));
    const F4 zaiBot = addF(s.zaiBot, scaleF(isInvDBot, qstZIcInternal));

    o.scalars = F4{ iStIs, qstZ, qstZIcInternal, 0.f };
    o.isInvDTop = F4{ isInvDTop.x, isInvDTop.y, isInvDTop.z, 0.f };
    o.isInvDBot = F4{ isInvDBot.x, isInvDBot.y, isInvDBot.z, 0.f };
    o.zaTop = zaTop; o.zaBot = zaBot; o.zaiTop = zaiTop; o.zaiBot = zaiBot;

    o.tlC0 = scaleF(isInvDTop, stiT.x); o.tlC1 = scaleF(isInvDTop, stiT.y); o.tlC2 = scaleF(isInvDTop, stiT.z);
    o.trC0 = scaleF(isInvDTop, stiB.x); o.trC1 = scaleF(isInvDTop, stiB.y); o.trC2 = scaleF(isInvDTop, stiB.z);
    o.blC0 = scaleF(isInvDBot, stiT.x); o.blC1 = scaleF(isInvDBot, stiT.y); o.blC2 = scaleF(isInvDBot, stiT.z);
}

PXG_TEST(npArtiPropagateZA)
{
    std::mt19937 rng(200001);
    const uint32_t N = 3000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto rv = [&](float a, float b) { return F4{ rf(a, b), rf(a, b), rf(a, b), 0.f }; };

    std::vector<PpIn> in(N);
    std::vector<PpOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        PpIn s{};
        s.saTop = rv(-2, 2); s.saBot = rv(-2, 2);
        s.isTop = rv(-3, 3); s.isBot = rv(-3, 3);
        s.zTop = rv(-3, 3); s.zBot = rv(-3, 3);
        s.zicTop = rv(-3, 3); s.zicBot = rv(-3, 3);
        s.zaTop = rv(-3, 3); s.zaBot = rv(-3, 3);
        s.zaiTop = rv(-3, 3); s.zaiBot = rv(-3, 3);
        const float armature = rf(0.f, 2.f);
        const float jF0 = rf(-4.f, 4.f);
        // jFnull flag stored as a bit pattern read back via asuint in the shader
        const bool jFnull = (i % 3) == 0;
        s.misc = F4{ armature, jF0, u2f(jFnull ? JF_NULL : 0u), 0.f };
        // Exercise the iStIs <= 0 guard branch: force stIs <= 0 for a stripe.
        if ((i % 17) == 5)
        {
            // zero the motion matrix so sa.innerProduct(Is) == 0 and armature == 0 -> stIs == 0
            s.saTop = F4{ 0, 0, 0, 0 }; s.saBot = F4{ 0, 0, 0, 0 };
            s.misc.x = 0.f;
        }
        in[i] = s;
        refPropZA(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<PpOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npArtiPropagateZA.spv", "propZA",
             { (uint32_t)sizeof(PpIn), (uint32_t)sizeof(PpOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<PpOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* rfp = &ref[i].scalars.x;
        const float* qfp = &g[i].scalars.x;
        const size_t nf = sizeof(PpOut) / 4;
        bool ok = true;
        for (size_t x = 0; x < nf && ok; ++x) ok = eqFZ(rfp[x], qfp[x]);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u scalarsR (%g %g %g) G (%g %g %g)\n", i, ref[i].scalars.x,
                            ref[i].scalars.y, ref[i].scalars.z, g[i].scalars.x, g[i].scalars.y, g[i].scalars.z);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "propZA outputs bitwise-identical (mod +-0)");
}

}  // namespace
