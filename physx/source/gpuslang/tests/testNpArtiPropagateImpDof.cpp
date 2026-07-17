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

// Verification of npArtiPropagateImpDof.slang (propImpDof): the single-dof
// zero-acceleration impulse propagation propagateImpulseDofAligned
// (internalConstraints2.cu L53-62), against an exact CPU transcription.
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

struct PiIn  { F4 isInvDTop, isInvDBot, saTop, saBot, zTop, zBot, c2p; };
struct PiOut { F4 resTop, resBot, scalars; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b) || (a == 0.f && b == 0.f); }

static float dot3(const F4& a, const F4& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static F4 cross3(const F4& a, const F4& b) { return F4{ a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x, 0.f }; }
static float inner(const F4& aTop, const F4& aBot, const F4& bTop, const F4& bBot) { return dot3(aBot, bTop) + dot3(aTop, bBot); }

// exact transcription of propagateImpulseDofAligned
static void refPropImpDof(const PiIn& s, PiOut& o)
{
    const float stZ = inner(s.saTop, s.saBot, s.zTop, s.zBot);
    const F4 tempTop = F4{ s.isInvDTop.x*stZ, s.isInvDTop.y*stZ, s.isInvDTop.z*stZ, 0.f };
    const F4 tempBot = F4{ s.isInvDBot.x*stZ, s.isInvDBot.y*stZ, s.isInvDBot.z*stZ, 0.f };
    const float qstZ = -stZ;

    const F4 xTop = F4{ s.zTop.x - tempTop.x, s.zTop.y - tempTop.y, s.zTop.z - tempTop.z, 0.f };
    const F4 xBot = F4{ s.zBot.x - tempBot.x, s.zBot.y - tempBot.y, s.zBot.z - tempBot.z, 0.f };

    const F4 cr = cross3(s.c2p, xTop);
    o.resTop = F4{ xTop.x, xTop.y, xTop.z, 0.f };
    o.resBot = F4{ xBot.x + cr.x, xBot.y + cr.y, xBot.z + cr.z, 0.f };
    o.scalars = F4{ qstZ, 0.f, 0.f, 0.f };
}

PXG_TEST(npArtiPropagateImpDof)
{
    std::mt19937 rng(204001);
    const uint32_t N = 3000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto rv = [&](float a, float b) { return F4{ rf(a, b), rf(a, b), rf(a, b), 0.f }; };

    std::vector<PiIn> in(N);
    std::vector<PiOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        PiIn s{};
        s.isInvDTop = rv(-3, 3); s.isInvDBot = rv(-3, 3);
        s.saTop = rv(-2, 2); s.saBot = rv(-2, 2);
        s.zTop = rv(-3, 3); s.zBot = rv(-3, 3);
        s.c2p = rv(-2, 2);
        in[i] = s;
        refPropImpDof(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<PiOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npArtiPropagateImpDof.spv", "propImpDof",
             { (uint32_t)sizeof(PiIn), (uint32_t)sizeof(PiOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<PiOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* rfp = &ref[i].resTop.x;
        const float* qfp = &g[i].resTop.x;
        const size_t nf = sizeof(PiOut) / 4;
        bool ok = true;
        for (size_t x = 0; x < nf && ok; ++x) ok = eqFZ(rfp[x], qfp[x]);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u resBotR (%g %g %g) G (%g %g %g)\n", i, ref[i].resBot.x, ref[i].resBot.y,
                            ref[i].resBot.z, g[i].resBot.x, g[i].resBot.y, g[i].resBot.z);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "propImpDof outputs bitwise-identical (mod +-0)");
}

}  // namespace
