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

// Verification of npAnisoHelpers.slang (anisoHelpers): the anisotropy small
// math helpers cube / Wa / Lerp / Clamp (anisotropy.cu L45-60), against an
// exact CPU transcription.  Outputs bitwise modulo the documented +-0 (eqFZ).

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
struct AnIn  { F4 misc, start, end, a, st; };
struct AnOut { F4 scal, lerp, clamp; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b) || (a == 0.f && b == 0.f); }

static float pmin(float a, float b) { return (a < b) ? a : b; }
static float pmax(float a, float b) { return (a > b) ? a : b; }
static float cube(float x) { return x * x * x; }
static float Wa(float x, float invr) { return 1.0f - cube(x * invr); }

static void refAniso(const AnIn& s, AnOut& o)
{
    const float cx = cube(s.misc.x);
    const float wa = Wa(s.misc.y, s.misc.z);
    const float t = s.misc.w;
    const F4 a = s.start, b = s.end;
    const F4 lp = F4{ a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t, 0.f };
    const float sc = s.st.x, tc = s.st.y;
    const F4 c = s.a;
    const F4 cl = F4{ pmin(tc, pmax(sc, c.x)), pmin(tc, pmax(sc, c.y)), pmin(tc, pmax(sc, c.z)), 0.f };
    o.scal = F4{ cx, wa, 0.f, 0.f };
    o.lerp = lp;
    o.clamp = cl;
}

PXG_TEST(npAnisoHelpers)
{
    std::mt19937 rng(214001);
    const uint32_t N = 4000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto rv = [&](float a, float b) { return F4{ rf(a, b), rf(a, b), rf(a, b), 0.f }; };

    std::vector<AnIn> in(N);
    std::vector<AnOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        AnIn s{};
        s.misc = F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), rf(-0.5f, 1.5f) };  // cubeX, waX, waInvr, lerpT
        s.start = rv(-3, 3);
        s.end = rv(-3, 3);
        // clamp: a in [-3,3] spans below/inside/above a band [s,t] with s<t
        s.a = rv(-3, 3);
        const float sc = rf(-1.5f, 0.f), tc = rf(0.f, 1.5f);   // s < t
        s.st = F4{ sc, tc, 0.f, 0.f };
        in[i] = s;
        refAniso(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<AnOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npAnisoHelpers.spv", "anisoHelpers",
             { (uint32_t)sizeof(AnIn), (uint32_t)sizeof(AnOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<AnOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* rfp = &ref[i].scal.x;
        const float* qfp = &g[i].scal.x;
        const size_t nf = sizeof(AnOut) / 4;
        bool ok = true;
        for (size_t x = 0; x < nf && ok; ++x) ok = eqFZ(rfp[x], qfp[x]);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u cube R %g G %g Wa R %g G %g\n", i, ref[i].scal.x, g[i].scal.x,
                            ref[i].scal.y, g[i].scal.y);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "anisoHelpers outputs bitwise-identical (mod +-0)");
}

}  // namespace
