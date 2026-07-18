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

// Verification of npPsDiffuseUpdate.slang (psDiffuseUpdate): the diffuse-particle
// velocity/position integration in ps_diffuseParticleCompact (diffuseParticles.cu
// L332-363), against an exact CPU transcription.  Outputs bitwise mod +-0 (eqFZ).

#include "pxgTestFramework.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct DpIn  { F4 pos, oldPos, velAvg, gravity, params, params2; };
struct DpOut { F4 newPos, newVel; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b) || (a == 0.f && b == 0.f); }

struct V3 { float x, y, z; };
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static float pmin(float a, float b) { return (a < b) ? a : b; }
static V3 sub(V3 a, V3 b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 add(V3 a, V3 b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 muls(V3 a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; }

static void refDp(const DpIn& s, DpOut& o)
{
    const V3 pos = V3{ s.pos.x, s.pos.y, s.pos.z };
    const float life = s.pos.w;
    const V3 oldPos = V3{ s.oldPos.x, s.oldPos.y, s.oldPos.z };
    const V3 velAvg = V3{ s.velAvg.x, s.velAvg.y, s.velAvg.z };
    const float classW = s.velAvg.w;
    const V3 gravity = V3{ s.gravity.x, s.gravity.y, s.gravity.z };
    const float airDrag = s.params.x, buoyancy = s.params.y, bubbleDrag = s.params.z, dt = s.params.w;
    const float maxVel = s.params2.x;

    const V3 vel = muls(sub(pos, oldPos), 1.0f / dt);

    V3 newVel;
    if (classW < 4.0f)
        newVel = muls(vel, 1.0f - airDrag * dt);
    else if (classW < 8.0f)
        newVel = velAvg;
    else
        newVel = add(sub(vel, muls(muls(gravity, 1.0f + buoyancy), dt)), muls(sub(velAvg, vel), bubbleDrag));

    const float magSq = dot3(newVel, newVel);
    if (magSq > 0.0f)
    {
        const float mag = std::sqrt(magSq);
        const V3 nrm = muls(newVel, 1.0f / mag);
        newVel = muls(nrm, pmin(mag, maxVel));
    }

    const V3 newPosCorr = add(pos, muls(sub(newVel, vel), dt));
    const float lifeTime = ((life - dt) > 0.0f) ? (life - dt) : 0.0f;

    o.newPos = F4{ newPosCorr.x, newPosCorr.y, newPosCorr.z, lifeTime };
    o.newVel = F4{ newVel.x, newVel.y, newVel.z, 0.f };
}

PXG_TEST(npPsDiffuseUpdate)
{
    std::mt19937 rng(243001);
    const uint32_t N = 8000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<DpIn> in(N);
    std::vector<DpOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        DpIn s{};
        s.pos = F4{ rf(-10.f, 10.f), rf(-10.f, 10.f), rf(-10.f, 10.f), rf(0.f, 2.f) };  // pos.xyz, life
        s.oldPos = F4{ rf(-10.f, 10.f), rf(-10.f, 10.f), rf(-10.f, 10.f), 0.f };
        // classW selects spray (<4), foam ([4,8)), bubble (>=8)
        const float classW = (i % 3u == 0u) ? rf(0.f, 3.99f) : (i % 3u == 1u) ? rf(4.f, 7.99f) : rf(8.f, 40.f);
        s.velAvg = F4{ rf(-5.f, 5.f), rf(-5.f, 5.f), rf(-5.f, 5.f), classW };
        s.gravity = F4{ rf(-1.f, 1.f), rf(-10.f, 0.f), rf(-1.f, 1.f), 0.f };
        s.params = F4{ rf(0.f, 2.f), rf(-0.5f, 2.f), rf(0.f, 2.f), rf(0.005f, 0.05f) };  // airDrag, buoyancy, bubbleDrag, dt
        s.params2 = F4{ rf(1.f, 30.f), 0.f, 0.f, 0.f };  // maxVel
        // speed-clamp-active stripe: small maxVel so magnitude exceeds it
        if ((i % 5u) == 2u) s.params2.x = rf(0.05f, 0.5f);
        // zero-velocity stripe (foam with velAvg == 0 -> newVel == 0 -> magSq == 0 skip)
        if ((i % 13u) == 4u) { s.velAvg = F4{ 0.f, 0.f, 0.f, 5.f }; }
        // expired-life stripe: life < dt -> lifeTime == 0
        if ((i % 17u) == 9u) s.pos.w = rf(0.f, 0.004f);
        in[i] = s;
        refDp(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<DpOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npPsDiffuseUpdate.spv", "psDiffuseUpdate",
             { (uint32_t)sizeof(DpIn), (uint32_t)sizeof(DpOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<DpOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        bool ok = eqFZ(ref[i].newPos.x, g[i].newPos.x) && eqFZ(ref[i].newPos.y, g[i].newPos.y)
               && eqFZ(ref[i].newPos.z, g[i].newPos.z) && eqFZ(ref[i].newPos.w, g[i].newPos.w)
               && eqFZ(ref[i].newVel.x, g[i].newVel.x) && eqFZ(ref[i].newVel.y, g[i].newVel.y)
               && eqFZ(ref[i].newVel.z, g[i].newVel.z);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u pR (%g %g %g %g) pG (%g %g %g %g) vR (%g %g %g) vG (%g %g %g)\n", i,
                            ref[i].newPos.x, ref[i].newPos.y, ref[i].newPos.z, ref[i].newPos.w,
                            g[i].newPos.x, g[i].newPos.y, g[i].newPos.z, g[i].newPos.w,
                            ref[i].newVel.x, ref[i].newVel.y, ref[i].newVel.z,
                            g[i].newVel.x, g[i].newVel.y, g[i].newVel.z);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "psDiffuseUpdate outputs bitwise-identical (mod +-0)");
}

}  // namespace
