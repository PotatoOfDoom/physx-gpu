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

// Verification of npClothWind.slang (clothWind): the "Frozen" aerodynamic wind
// force disneyWindModelEffect (particleSystem.cuh L71-96), against an exact CPU
// transcription.  Outputs bitwise mod +-0 (eqFZ).

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
struct WdIn  { F4 x0, x1, x2, vel0, vel1, vel2, wind, params, params2; };
struct WdOut { F4 force; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed

struct V3 { float x, y, z; };
static V3 sub3(V3 a, V3 b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 add3(V3 a, V3 b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 scl3(float s, V3 a) { return V3{ s * a.x, s * a.y, s * a.z }; }
static V3 addv(V3 a, V3 b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b) { return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }

static void refWind(const WdIn& s, WdOut& o)
{
    const V3 x0 = V3{ s.x0.x, s.x0.y, s.x0.z };
    const V3 x1 = V3{ s.x1.x, s.x1.y, s.x1.z };
    const V3 x2 = V3{ s.x2.x, s.x2.y, s.x2.z };
    const V3 vel0 = V3{ s.vel0.x, s.vel0.y, s.vel0.z };
    const V3 vel1 = V3{ s.vel1.x, s.vel1.y, s.vel1.z };
    const V3 vel2 = V3{ s.vel2.x, s.vel2.y, s.vel2.z };
    const V3 wind = V3{ s.wind.x, s.wind.y, s.wind.z };
    const float inverseMass = s.params.x, drag = s.params.y, lift = s.params.z, dt = s.params.w;
    const float airDensity = s.params2.x;

    const V3 x01 = sub3(x1, x0);
    const V3 x02 = sub3(x2, x0);
    V3 n = cross3(x01, x02);

    const V3 v = scl3(0.3333f, add3(add3(vel0, vel1), vel2));
    const V3 vrel = sub3(wind, v);

    if (dot3(vrel, n) < 0.0f)
        n = scl3(-1.0f, n);

    const float coef = 0.25f * airDensity * dt * inverseMass;

    // coef * ((drag-lift)*vrel.dot(n)*vrel + lift*|vrel|^2*n)
    const V3 A = scl3((drag - lift) * dot3(vrel, n), vrel);
    const V3 B = scl3(lift * dot3(vrel, vrel), n);
    const V3 force = scl3(coef, addv(A, B));

    o.force = F4{ force.x, force.y, force.z, 0.0f };
}

PXG_TEST(npClothWind)
{
    std::mt19937 rng(221001);
    const uint32_t N = 5000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<WdIn> in(N);
    std::vector<WdOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        WdIn s{};
        s.x0 = F4{ rf(-3.f, 3.f), rf(-3.f, 3.f), rf(-3.f, 3.f), 0.f };
        s.x1 = F4{ rf(-3.f, 3.f), rf(-3.f, 3.f), rf(-3.f, 3.f), 0.f };
        s.x2 = F4{ rf(-3.f, 3.f), rf(-3.f, 3.f), rf(-3.f, 3.f), 0.f };
        s.vel0 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
        s.vel1 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
        s.vel2 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
        s.wind = F4{ rf(-5.f, 5.f), rf(-5.f, 5.f), rf(-5.f, 5.f), 0.f };
        s.params = F4{ rf(0.f, 3.f), rf(0.f, 2.f), rf(0.f, 2.f), rf(0.005f, 0.05f) };  // invMass,drag,lift,dt
        s.params2 = F4{ rf(0.5f, 2.f), 0.f, 0.f, 0.f };                                 // airDensity
        // Both signs of vrel.dot(n) arise naturally from the random wind vs the
        // triangle normal orientation, covering the n-flip branch.
        // Coplanar stripe: put the triangle in the z=0 plane so n = (0,0,nz), and
        // zero all velocity/wind z so vrel.z = 0 -> vrel.dot(n) == 0 EXACTLY.  This
        // pins down the strict `< 0` flip boundary (no flip at exactly 0, so n keeps
        // the cross(x1-x0, x2-x0) orientation): a swapped edge/cross-order that would
        // otherwise be output-equivalent (the flip canonicalizes n) now diverges in
        // force.z, matching the exact CUDA behavior at the perpendicular boundary.
        if ((i % 9u) == 5u)
        {
            s.x0.z = s.x1.z = s.x2.z = 0.f;
            s.vel0.z = s.vel1.z = s.vel2.z = 0.f;
            s.wind.z = 0.f;
        }
        in[i] = s;
        refWind(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<WdOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npClothWind.spv", "clothWind",
             { (uint32_t)sizeof(WdIn), (uint32_t)sizeof(WdOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<WdOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        bool ok = eqFZ(ref[i].force.x, g[i].force.x) && eqFZ(ref[i].force.y, g[i].force.y) && eqFZ(ref[i].force.z, g[i].force.z);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u R (%g %g %g) G (%g %g %g)\n", i,
                            ref[i].force.x, ref[i].force.y, ref[i].force.z,
                            g[i].force.x, g[i].force.y, g[i].force.z);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "clothWind outputs bitwise-identical (mod +-0)");
}

}  // namespace
