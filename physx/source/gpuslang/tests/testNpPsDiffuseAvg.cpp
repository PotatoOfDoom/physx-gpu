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

// Verification of npPsDiffuseAvg.slang (psDiffuseAvg): the diffuse-particle PBF
// neighborhood velocity averaging (diffuseParticles.cu L215-271) modeled as a
// fixed-order K=8 reduction, against an exact CPU transcription.  Outputs
// bitwise mod +-0 (eqFZ).

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
struct AvIn  { F4 pos, prm; F4 xj[8]; F4 vj[8]; };
struct AvOut { F4 velAvg; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b) || (a == 0.f && b == 0.f); }

static float WDiffuse(float h, float invR) { return 1.0f - h * invR; }

static void refAv(const AvIn& s, AvOut& o)
{
    const float px = s.pos.x, py = s.pos.y, pz = s.pos.z;
    const float contactDistanceSq = s.pos.w;
    const float invContactDistance = s.prm.x;

    float ax = 0.f, ay = 0.f, az = 0.f;   // velAvg
    float weightSum = 0.f;
    float sx = 0.f, sy = 0.f, sz = 0.f;   // velocitySum
    uint32_t numNeighbors = 0u;

    for (uint32_t k = 0; k < 8u; ++k)
    {
        const float xijx = px - s.xj[k].x, xijy = py - s.xj[k].y, xijz = pz - s.xj[k].z;
        const float dSq = xijx * xijx + xijy * xijy + xijz * xijz;
        if (dSq < contactDistanceSq)
        {
            const float w = WDiffuse(std::sqrt(dSq), invContactDistance);
            weightSum += w;
            sx += s.vj[k].x * w; sy += s.vj[k].y * w; sz += s.vj[k].z * w;
            ++numNeighbors;
        }
    }

    if (weightSum > 0.f)
    {
        const float inv = 1.0f / weightSum;
        ax = sx * inv; ay = sy * inv; az = sz * inv;
    }

    o.velAvg = F4{ ax, ay, az, float(numNeighbors) };
}

PXG_TEST(npPsDiffuseAvg)
{
    std::mt19937 rng(245001);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<AvIn> in(N);
    std::vector<AvOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        AvIn s{};
        s.pos = F4{ rf(-5.f, 5.f), rf(-5.f, 5.f), rf(-5.f, 5.f), 0.f };
        const float contactDist = rf(0.5f, 3.f);
        s.pos.w = contactDist * contactDist;                 // contactDistanceSq
        s.prm = F4{ 1.0f / contactDist, 0.f, 0.f, 0.f };     // invContactDistance
        for (uint32_t k = 0; k < 8u; ++k)
        {
            // place neighbors within a couple contact-distances of pos so both
            // inside (dSq < cdSq) and outside cases occur
            s.xj[k] = F4{ s.pos.x + rf(-2.f * contactDist, 2.f * contactDist),
                          s.pos.y + rf(-2.f * contactDist, 2.f * contactDist),
                          s.pos.z + rf(-2.f * contactDist, 2.f * contactDist), 0.f };
            s.vj[k] = F4{ rf(-5.f, 5.f), rf(-5.f, 5.f), rf(-5.f, 5.f), 0.f };
        }
        // no-neighbor stripe: push all neighbors far away -> weightSum == 0 -> velAvg 0
        if ((i % 11u) == 4u)
            for (uint32_t k = 0; k < 8u; ++k)
                s.xj[k] = F4{ s.pos.x + 100.f, s.pos.y + 100.f, s.pos.z + 100.f, 0.f };
        // all-inside stripe: cluster neighbors tightly around pos -> all contribute
        if ((i % 11u) == 7u)
            for (uint32_t k = 0; k < 8u; ++k)
                s.xj[k] = F4{ s.pos.x + rf(-0.1f, 0.1f), s.pos.y + rf(-0.1f, 0.1f), s.pos.z + rf(-0.1f, 0.1f), 0.f };
        in[i] = s;
        refAv(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<AvOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npPsDiffuseAvg.spv", "psDiffuseAvg",
             { (uint32_t)sizeof(AvIn), (uint32_t)sizeof(AvOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<AvOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        bool ok = eqFZ(ref[i].velAvg.x, g[i].velAvg.x) && eqFZ(ref[i].velAvg.y, g[i].velAvg.y)
               && eqFZ(ref[i].velAvg.z, g[i].velAvg.z) && eqFZ(ref[i].velAvg.w, g[i].velAvg.w);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u R (%g %g %g n%g) G (%g %g %g n%g)\n", i,
                            ref[i].velAvg.x, ref[i].velAvg.y, ref[i].velAvg.z, ref[i].velAvg.w,
                            g[i].velAvg.x, g[i].velAvg.y, g[i].velAvg.z, g[i].velAvg.w);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "psDiffuseAvg outputs bitwise-identical (mod +-0)");
}

}  // namespace
