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

// Verification of npClothTriContact.slang (clothTriContact): the final
// trivial-leaf batch - computeTriangleContact float2/float4-barycentric
// variants (FEMClothExternalSolve.cu L155-194; the float2 variant's
// u = 1.0 - bx - by is an f64 spot: the DOUBLE literal promotes the chain),
// bvhComputeTotalInvEdges and pow3/rangesOverlaps (SDFConstruction.cu
// L245-251, L441-448) - against an exact CPU transcription.  Outputs bitwise
// (floats mod +-0; uints/flags exact).

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
struct TcIn  { F4 misc, v0, v1, v2, rng, iw; };
struct TcOut { F4 vel, aux; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed

static void refTc(const TcIn& s, TcOut& o)
{
    std::memset(&o, 0, sizeof(o));
    const uint32_t mode = f2u(s.misc.x);

    if (mode == 0u)
    {
        // f64 spot: u = PxReal(1.0 - bx - by), double-literal chain
        const float u = float(1.0 - double(s.misc.y) - double(s.misc.z));
        o.vel = F4{ s.v0.x * u + s.v1.x * s.misc.y + s.v2.x * s.misc.z,
                    s.v0.y * u + s.v1.y * s.misc.y + s.v2.y * s.misc.z,
                    s.v0.z * u + s.v1.z * s.misc.y + s.v2.z * s.misc.z,
                    s.v0.w * u + s.v1.w * s.misc.y + s.v2.w * s.misc.z };
        o.aux = F4{ s.v0.w, s.v1.w, s.v2.w, 0.f };
    }
    else if (mode == 1u)
    {
        o.vel = F4{ s.v0.x * s.misc.y + s.v1.x * s.misc.z + s.v2.x * s.misc.w,
                    s.v0.y * s.misc.y + s.v1.y * s.misc.z + s.v2.y * s.misc.w,
                    s.v0.z * s.misc.y + s.v1.z * s.misc.z + s.v2.z * s.misc.w,
                    s.v0.w * s.misc.y + s.v1.w * s.misc.z + s.v2.w * s.misc.w };
        o.aux = F4{ s.v0.w, s.v1.w, s.v2.w, 0.f };
    }
    else if (mode == 2u)
    {
        const float ex = (s.v1.x - s.v0.x) + 0.0001f;
        const float ey = (s.v1.y - s.v0.y) + 0.0001f;
        const float ez = (s.v1.z - s.v0.z) + 0.0001f;
        o.aux = F4{ 1.0f / ex, 1.0f / ey, 1.0f / ez, 0.f };
    }
    else
    {
        const uint32_t i = f2u(s.iw.x);
        const uint32_t p3 = i * i * i;
        const bool overlap = !(s.rng.x > s.rng.w || s.rng.z > s.rng.y);
        o.aux = F4{ u2f(p3), overlap ? u2f(1u) : u2f(0u), 0.f, 0.f };
    }
}

PXG_TEST(npClothTriContact)
{
    std::mt19937 rng(263001);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<TcIn> in(N);
    std::vector<TcOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        TcIn s{};
        const uint32_t mode = i % 4u;
        s.misc.x = u2f(mode);
        if (mode <= 1u)
        {
            s.misc.y = rf(0.f, 1.f);
            s.misc.z = rf(0.f, 1.f);
            s.misc.w = rf(0.f, 1.f);
            s.v0 = F4{ rf(-3.f, 3.f), rf(-3.f, 3.f), rf(-3.f, 3.f), rf(0.f, 3.f) };
            s.v1 = F4{ rf(-3.f, 3.f), rf(-3.f, 3.f), rf(-3.f, 3.f), rf(0.f, 3.f) };
            s.v2 = F4{ rf(-3.f, 3.f), rf(-3.f, 3.f), rf(-3.f, 3.f), rf(0.f, 3.f) };
            // u < 0 stripe (bary sum > 1)
            if ((i % 7u) == 2u) { s.misc.y = 0.8f; s.misc.z = 0.7f; }
            // corner stripe: bary (0,0) -> u = 1 exact
            if ((i % 13u) == 6u) { s.misc.y = 0.f; s.misc.z = 0.f; }
        }
        else if (mode == 2u)
        {
            s.v0 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
            s.v1 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
            // degenerate bounds: upper == lower -> edges = 0.0001f exactly
            if ((i % 13u) == 6u) s.v1 = s.v0;
        }
        else
        {
            s.rng = F4{ rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-1.f, 1.f) };
            // touching-boundary stripe: maxB == minA exactly (overlap stays true)
            if ((i % 7u) == 2u) s.rng.w = s.rng.x;
            // pow3 incl. u32-wrap inputs (1626^3 > 2^32 wraps identically)
            const uint32_t pi = ((i % 11u) == 4u) ? 2000u + (rng() % 100u) : rng() % 1626u;
            s.iw.x = u2f(pi);
        }
        in[i] = s;
        refTc(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<TcOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npClothTriContact.spv", "clothTriContact",
             { (uint32_t)sizeof(TcIn), (uint32_t)sizeof(TcOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<TcOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const uint32_t mode = i % 4u;
        bool ok = eqFZ(ref[i].vel.x, g[i].vel.x) && eqFZ(ref[i].vel.y, g[i].vel.y)
               && eqFZ(ref[i].vel.z, g[i].vel.z) && eqFZ(ref[i].vel.w, g[i].vel.w);
        if (mode == 3u)
            ok = ok && f2u(ref[i].aux.x) == f2u(g[i].aux.x) && f2u(ref[i].aux.y) == f2u(g[i].aux.y);
        else
            ok = ok && eqFZ(ref[i].aux.x, g[i].aux.x) && eqFZ(ref[i].aux.y, g[i].aux.y)
                    && eqFZ(ref[i].aux.z, g[i].aux.z) && eqFZ(ref[i].aux.w, g[i].aux.w);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u mode %u velR (%g %g %g %g) velG (%g %g %g %g)\n", i, mode,
                            ref[i].vel.x, ref[i].vel.y, ref[i].vel.z, ref[i].vel.w,
                            g[i].vel.x, g[i].vel.y, g[i].vel.z, g[i].vel.w);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "clothTriContact outputs bitwise-identical");
}

}  // namespace
