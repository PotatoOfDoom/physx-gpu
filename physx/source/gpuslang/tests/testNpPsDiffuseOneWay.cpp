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

// Verification of npPsDiffuseOneWay.slang (psDiffuseOneWay): the diffuse-particle
// one-way static-collision position correction (diffuseParticles.cu L143-159)
// modeled as a fixed-order K=8 accumulation, against an exact CPU transcription.
// Outputs bitwise mod +-0 (eqFZ).

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
struct CcIn  { F4 pos, cnt; F4 contacts[8]; };
struct CcOut { F4 pos; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b) || (a == 0.f && b == 0.f); }

static void refCc(const CcIn& s, CcOut& o)
{
    const uint32_t rawCount = f2u(s.cnt.x);
    const uint32_t contactCount = (rawCount < 8u) ? rawCount : 8u;

    float px = s.pos.x, py = s.pos.y, pz = s.pos.z;
    for (uint32_t c = 0; c < contactCount; ++c)
    {
        const float nx = s.contacts[c].x, ny = s.contacts[c].y, nz = s.contacts[c].z;
        const float pen = s.contacts[c].w;
        px += -nx * pen; py += -ny * pen; pz += -nz * pen;
    }
    o.pos = F4{ px, py, pz, s.pos.w };
}

PXG_TEST(npPsDiffuseOneWay)
{
    std::mt19937 rng(246001);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<CcIn> in(N);
    std::vector<CcOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        CcIn s{};
        s.pos = F4{ rf(-10.f, 10.f), rf(-10.f, 10.f), rf(-10.f, 10.f), rf(-5.f, 5.f) };
        // contactCount spans 0..8 (0 -> unchanged passthrough)
        const uint32_t count = i % 9u;
        s.cnt = F4{ u2f(count), 0.f, 0.f, 0.f };
        for (uint32_t c = 0; c < 8u; ++c)
        {
            // roughly-unit normals with a penetration depth
            s.contacts[c] = F4{ rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-0.5f, 0.5f) };
        }
        // over-count stripe: count > 8 -> clamped to 8 (PxMin(12, count) with our K=8)
        if ((i % 13u) == 6u) s.cnt.x = u2f(10u + (i % 3u));
        in[i] = s;
        refCc(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<CcOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npPsDiffuseOneWay.spv", "psDiffuseOneWay",
             { (uint32_t)sizeof(CcIn), (uint32_t)sizeof(CcOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<CcOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        bool ok = eqFZ(ref[i].pos.x, g[i].pos.x) && eqFZ(ref[i].pos.y, g[i].pos.y)
               && eqFZ(ref[i].pos.z, g[i].pos.z) && eqFZ(ref[i].pos.w, g[i].pos.w);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u R (%g %g %g %g) G (%g %g %g %g)\n", i,
                            ref[i].pos.x, ref[i].pos.y, ref[i].pos.z, ref[i].pos.w,
                            g[i].pos.x, g[i].pos.y, g[i].pos.z, g[i].pos.w);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "psDiffuseOneWay outputs bitwise-identical (mod +-0)");
}

}  // namespace
