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

// Verification of npPsDiffuseSpawn.slang (psDiffuseSpawn): the diffuse-particle
// spawn decision + spawn-state computation in ps_diffuseParticleCreate
// (diffuseParticles.cu L456-495), against an exact CPU transcription.  Outputs
// bitwise mod +-0 (eqFZ).

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
struct SpIn  { F4 vel, pos, ptn, wts, rnd, params; };
struct SpOut { F4 q, misc; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed

struct V3 { float x, y, z; };
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 sub(V3 a, V3 b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 add(V3 a, V3 b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 muls(V3 a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; }

static void refSp(const SpIn& s, SpOut& o)
{
    const V3 vi = V3{ s.vel.x, s.vel.y, s.vel.z };
    const V3 xi = V3{ s.pos.x, s.pos.y, s.pos.z };
    const float divergencePot = s.ptn.x, pressurePot = s.ptn.y;
    const float kineticEnergyWeight = s.wts.x, divergenceWeight = s.wts.y, pressureWeight = s.wts.z, threshold = s.wts.w;
    const float r0 = s.rnd.x, r1 = s.rnd.y, r2 = s.rnd.z, r3 = s.rnd.w;
    const float restOffset = s.params.x, dt = s.params.y, lifetimeMax = s.params.z;

    const float kineticEnergy = dot3(vi, vi) * kineticEnergyWeight;
    const float divergence = divergenceWeight * divergencePot;
    const float pressure = pressureWeight * pressurePot;
    const float intensity = pressure - divergence + kineticEnergy;

    const float spawn = (r0 * intensity > threshold) ? 1.0f : 0.0f;

    const float lifeMin = 1.0f;
    const float lifeMax = lifetimeMax;
    const float ratio = intensity / threshold;
    const float lifeScale = ((ratio < 1.0f) ? ratio : 1.0f) * r1;
    const float lifetime = lifeMin + lifeScale * (lifeMax - lifeMin);

    const V3 jitter = muls(muls(V3{ r1, r2, r3 }, restOffset), 0.25f);
    const V3 q = add(sub(xi, muls(muls(vi, r2), dt)), jitter);

    o.q = F4{ q.x, q.y, q.z, lifetime };
    o.misc = F4{ intensity, spawn, 0.f, 0.f };
}

PXG_TEST(npPsDiffuseSpawn)
{
    std::mt19937 rng(244001);
    const uint32_t N = 8000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<SpIn> in(N);
    std::vector<SpOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        SpIn s{};
        s.vel = F4{ rf(-5.f, 5.f), rf(-5.f, 5.f), rf(-5.f, 5.f), 0.f };
        s.pos = F4{ rf(-10.f, 10.f), rf(-10.f, 10.f), rf(-10.f, 10.f), 0.f };
        s.ptn = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f, 0.f };   // divergence pot, pressure pot
        s.wts = F4{ rf(0.f, 2.f), rf(0.f, 2.f), rf(0.f, 2.f), rf(0.1f, 5.f) };  // keW, divW, presW, threshold(>0)
        s.rnd = F4{ rf(0.f, 1.f), rf(0.f, 1.f), rf(0.f, 1.f), rf(0.f, 1.f) };   // r0, r1, r2, r3
        s.params = F4{ rf(0.05f, 1.f), rf(0.005f, 0.05f), rf(2.f, 20.f), 0.f };  // restOffset, dt, lifetimeMax
        // high-intensity stripe: ratio >= 1 (fminf picks 1) and spawn likely true
        if ((i % 5u) == 2u) { s.wts = F4{ rf(1.f, 3.f), 0.1f, rf(1.f, 3.f), rf(0.1f, 0.5f) }; s.rnd.x = rf(0.7f, 1.f); }
        // low-intensity stripe: intensity small -> ratio < 1, spawn likely false
        if ((i % 5u) == 3u) { s.wts.x = rf(0.f, 0.05f); s.wts.z = rf(0.f, 0.05f); s.rnd.x = rf(0.f, 0.2f); }
        in[i] = s;
        refSp(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<SpOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npPsDiffuseSpawn.spv", "psDiffuseSpawn",
             { (uint32_t)sizeof(SpIn), (uint32_t)sizeof(SpOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<SpOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        bool ok = eqFZ(ref[i].q.x, g[i].q.x) && eqFZ(ref[i].q.y, g[i].q.y) && eqFZ(ref[i].q.z, g[i].q.z)
               && eqFZ(ref[i].q.w, g[i].q.w) && eqFZ(ref[i].misc.x, g[i].misc.x) && eqFZ(ref[i].misc.y, g[i].misc.y);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u qR (%g %g %g %g) qG (%g %g %g %g) iR %g iG %g spR %g spG %g\n", i,
                            ref[i].q.x, ref[i].q.y, ref[i].q.z, ref[i].q.w,
                            g[i].q.x, g[i].q.y, g[i].q.z, g[i].q.w,
                            ref[i].misc.x, g[i].misc.x, ref[i].misc.y, g[i].misc.y);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "psDiffuseSpawn outputs bitwise-identical (mod +-0)");
}

}  // namespace
