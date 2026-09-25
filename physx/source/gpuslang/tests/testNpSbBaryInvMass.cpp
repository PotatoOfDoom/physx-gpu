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

// Verification of npSbBaryInvMass.slang (sbBaryInvMass): the tetrahedron
// barycentric computeBarycentric (softBody.cu L532-575) composed with the
// p-based computeInvMass (L577-606), against an exact CPU transcription.
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
struct BcIn  { F4 a, b, c, d, p; };
struct BcOut { F4 bary, invMass; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed

struct V3 { float x, y, z; };
static V3 sub3(V3 a, V3 b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b) { return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static float clamp01(float t) { const float hi = (1.0f < t) ? 1.0f : t; return (0.0f > hi) ? 0.0f : hi; }

static void refBary(const BcIn& s, BcOut& o)
{
    const V3 a = V3{ s.a.x, s.a.y, s.a.z };
    const V3 b = V3{ s.b.x, s.b.y, s.b.z };
    const V3 c = V3{ s.c.x, s.c.y, s.c.z };
    const V3 d = V3{ s.d.x, s.d.y, s.d.z };
    const V3 p = V3{ s.p.x, s.p.y, s.p.z };

    const V3 ba = sub3(b, a);
    const V3 ca = sub3(c, a);
    const V3 da = sub3(d, a);
    const V3 pa = sub3(p, a);

    const float detBcd = dot3(ba, cross3(ca, da));
    const float detPcd = dot3(pa, cross3(ca, da));
    const float v = detPcd / detBcd;
    const float detBpd = dot3(ba, cross3(pa, da));
    const float w = detBpd / detBcd;
    const float detBcp = dot3(ba, cross3(ca, pa));
    const float x = detBcp / detBcd;

    const float u = 1.0f - v - w - x;
    const float invMass = s.a.w * u + s.b.w * v + s.c.w * w + s.d.w * x;

    o.bary = F4{ clamp01(u), clamp01(v), clamp01(w), clamp01(x) };
    o.invMass = F4{ invMass, 0.f, 0.f, 0.f };
}

PXG_TEST(npSbBaryInvMass)
{
    std::mt19937 rng(222001);
    const uint32_t N = 5000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<BcIn> in(N);
    std::vector<BcOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        BcIn s{};
        // A reasonably non-degenerate tetrahedron: base near the axes plus jitter.
        s.a = F4{ rf(-0.5f, 0.5f),       rf(-0.5f, 0.5f),       rf(-0.5f, 0.5f),       rf(0.f, 3.f) };
        s.b = F4{ rf(1.5f, 2.5f),        rf(-0.5f, 0.5f),       rf(-0.5f, 0.5f),       rf(0.f, 3.f) };
        s.c = F4{ rf(-0.5f, 0.5f),       rf(1.5f, 2.5f),        rf(-0.5f, 0.5f),       rf(0.f, 3.f) };
        s.d = F4{ rf(-0.5f, 0.5f),       rf(-0.5f, 0.5f),       rf(1.5f, 2.5f),        rf(0.f, 3.f) };
        // p mostly inside the tet (so 0<=bary<=1, exercising the pass-through of
        // clamp01) but sometimes outside (exercising both clamp rails).
        if ((i % 3u) == 0u)
        {
            // outside: extrapolate along a random direction from the centroid
            const float cx = (s.a.x + s.b.x + s.c.x + s.d.x) * 0.25f;
            const float cy = (s.a.y + s.b.y + s.c.y + s.d.y) * 0.25f;
            const float cz = (s.a.z + s.b.z + s.c.z + s.d.z) * 0.25f;
            s.p = F4{ cx + rf(-4.f, 4.f), cy + rf(-4.f, 4.f), cz + rf(-4.f, 4.f), 0.f };
        }
        else
        {
            // inside-ish: convex combination with random weights
            float w0 = rf(0.05f, 1.f), w1 = rf(0.05f, 1.f), w2 = rf(0.05f, 1.f), w3 = rf(0.05f, 1.f);
            const float sum = w0 + w1 + w2 + w3;
            w0 /= sum; w1 /= sum; w2 /= sum; w3 /= sum;
            s.p = F4{ w0 * s.a.x + w1 * s.b.x + w2 * s.c.x + w3 * s.d.x,
                      w0 * s.a.y + w1 * s.b.y + w2 * s.c.y + w3 * s.d.y,
                      w0 * s.a.z + w1 * s.b.z + w2 * s.c.z + w3 * s.d.z, 0.f };
        }
        in[i] = s;
        refBary(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<BcOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npSbBaryInvMass.spv", "sbBaryInvMass",
             { (uint32_t)sizeof(BcIn), (uint32_t)sizeof(BcOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<BcOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* rfp = &ref[i].bary.x;
        const float* qfp = &g[i].bary.x;
        bool ok = true;
        for (size_t xi = 0; xi < 5 && ok; ++xi) ok = eqFZ(rfp[xi], qfp[xi]);   // bary.xyzw + invMass.x
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u baryR (%g %g %g %g) imR %g baryG (%g %g %g %g) imG %g\n", i,
                            ref[i].bary.x, ref[i].bary.y, ref[i].bary.z, ref[i].bary.w, ref[i].invMass.x,
                            g[i].bary.x, g[i].bary.y, g[i].bary.z, g[i].bary.w, g[i].invMass.x);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "sbBaryInvMass outputs bitwise-identical (mod +-0)");
}

}  // namespace
