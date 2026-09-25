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

// Verification of npSmoothPos.slang (smoothPos): the smoothPositionsLaunch final
// blending stage (anisotropy.cu L200-209) -- PxMin-limited smoothing weight,
// reciprocal-multiply xs/=ws, and Lerp toward the neighborhood mean -- against
// an exact CPU transcription.  Outputs bitwise mod +-0 (eqFZ).

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
struct SpIn  { F4 xi, xs, params; };
struct SpOut { F4 pos; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed

static float cube(float x) { return x * x * x; }
static float Wa(float x, float invr) { return 1.0f - cube(x * invr); }

static void refSp(const SpIn& s, SpOut& o)
{
    const float xix = s.xi.x, xiy = s.xi.y, xiz = s.xi.z, w = s.xi.w;
    const float ws = s.params.x, smoothing = s.params.y;
    const float contactDist = s.params.z, contactDistInv = s.params.w;

    float ox = xix, oy = xiy, oz = xiz;
    if (ws > 0.0f)
    {
        const float f = 4.0f * Wa(contactDist * 0.5f, contactDistInv);
        const float wsf = ws / f;
        const float mn = (1.0f < wsf) ? 1.0f : wsf;   // PxMin(1, ws/f)
        const float smooth = mn * smoothing;

        const float invws = 1.0f / ws;                // xs /= ws (reciprocal-multiply)
        const float sx = s.xs.x * invws, sy = s.xs.y * invws, sz = s.xs.z * invws;

        ox = xix + (sx - xix) * smooth;               // Lerp(xi, xsN, smooth)
        oy = xiy + (sy - xiy) * smooth;
        oz = xiz + (sz - xiz) * smooth;
    }
    o.pos = F4{ ox, oy, oz, w };
}

PXG_TEST(npSmoothPos)
{
    std::mt19937 rng(238001);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<SpIn> in(N);
    std::vector<SpOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        SpIn s{};
        s.xi = F4{ rf(-10.f, 10.f), rf(-10.f, 10.f), rf(-10.f, 10.f), rf(-5.f, 5.f) };
        // xs is a weighted sum: magnitude scaled by ws so xs/ws lands near a real position
        const float ws = rf(0.01f, 30.f);
        s.xs = F4{ rf(-10.f, 10.f) * ws, rf(-10.f, 10.f) * ws, rf(-10.f, 10.f) * ws, 0.f };
        const float cd = rf(0.2f, 2.f);
        s.params = F4{ ws, rf(0.f, 1.f), cd, 1.0f / cd };  // ws, smoothing, contactDist, contactDistInv
        // passthrough stripe: ws <= 0 -> output == original xi (incl. w)
        if ((i % 9u) == 4u) s.params.x = (i % 18u == 4u) ? 0.0f : rf(-5.f, -0.01f);
        // full-smoothing stripe: huge ws -> ws/f > 1 -> PxMin picks 1
        if ((i % 9u) == 7u) s.params.x = rf(100.f, 500.f);
        // zero-smoothing stripe: smoothing == 0 -> xiOut == xi even with ws>0
        if ((i % 11u) == 3u) s.params.y = 0.0f;
        in[i] = s;
        refSp(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<SpOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npSmoothPos.spv", "smoothPos",
             { (uint32_t)sizeof(SpIn), (uint32_t)sizeof(SpOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<SpOut>(dev, bOut, N);
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
    PXG_CHECK_MSG(mm == 0, "smoothPos outputs bitwise-identical (mod +-0)");
}

}  // namespace
