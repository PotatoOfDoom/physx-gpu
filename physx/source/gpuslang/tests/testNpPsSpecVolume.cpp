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

// Verification of npPsSpecVolume.slang (psSpecVolume): the particle speculative
// contact volume getParticleSpeculativeContactVolume (gridCal.cuh L122-140),
// against an exact CPU transcription.  Outputs bitwise mod +-0 (eqFZ).

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
struct SvIn  { F4 cur, pred, misc; };
struct SvOut { F4 volPos, scalars; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b) || (a == 0.f && b == 0.f); }

static const float SCALE_INV_PGS = 1.0f / 1.0f;
static const float SCALE_INV_TGS = 1.0f / 0.5f;
static const float SCALE_INV_DIFFUSE = 1.0f / 1.0f;

static void refSv(const SvIn& s, SvOut& o)
{
    const float cx = s.cur.x, cy = s.cur.y, cz = s.cur.z;
    const float px = s.pred.x, py = s.pred.y, pz = s.pred.z;
    const float contactDist = s.misc.x;
    const bool isDiffuse = f2u(s.misc.y) == 1u;
    const bool isTGS = f2u(s.misc.z) == 1u;

    float scaleInv;
    if (isDiffuse)
        scaleInv = SCALE_INV_DIFFUSE;
    else
        scaleInv = isTGS ? SCALE_INV_TGS : SCALE_INV_PGS;

    const float ox = (px - cx) * scaleInv * 0.5f;
    const float oy = (py - cy) * scaleInv * 0.5f;
    const float oz = (pz - cz) * scaleInv * 0.5f;
    const float radius = std::sqrt(ox * ox + oy * oy + oz * oz) + contactDist;

    o.volPos = F4{ cx + ox, cy + oy, cz + oz, 0.f };
    o.scalars = F4{ radius, 0.f, 0.f, 0.f };
}

PXG_TEST(npPsSpecVolume)
{
    std::mt19937 rng(234001);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<SvIn> in(N);
    std::vector<SvOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        SvIn s{};
        s.cur = F4{ rf(-10.f, 10.f), rf(-10.f, 10.f), rf(-10.f, 10.f), 0.f };
        s.pred = F4{ rf(-10.f, 10.f), rf(-10.f, 10.f), rf(-10.f, 10.f), 0.f };
        // cover all four (isDiffuse, isTGS) combinations: diffuse-only (scale 1),
        // TGS-only (scale 2), PGS/neither (scale 1), and BOTH set (scale 1 via the
        // isDiffuse-priority branch).  The both-set case is what makes the
        // isDiffuse-over-isTGS precedence observable: a precedence-swapping mutant
        // would pick TGS (2.0) there and diverge (SCALE_DIFFUSE==SCALE_PGS==1.0, so
        // the guard is otherwise invisible).
        const uint32_t sel = i % 4u;
        const uint32_t isDiffuse = (sel == 0u || sel == 3u) ? 1u : 0u;
        const uint32_t isTGS = (sel == 1u || sel == 3u) ? 1u : 0u;
        s.misc = F4{ rf(0.f, 2.f), u2f(isDiffuse), u2f(isTGS), 0.f };
        // zero-motion stripe: predicted == current -> offset 0 -> radius = contactDist
        if ((i % 11u) == 4u) s.pred = s.cur;
        in[i] = s;
        refSv(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<SvOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npPsSpecVolume.spv", "psSpecVolume",
             { (uint32_t)sizeof(SvIn), (uint32_t)sizeof(SvOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<SvOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        bool ok = eqFZ(ref[i].volPos.x, g[i].volPos.x) && eqFZ(ref[i].volPos.y, g[i].volPos.y)
               && eqFZ(ref[i].volPos.z, g[i].volPos.z) && eqFZ(ref[i].scalars.x, g[i].scalars.x);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u rR %g rG %g vpR (%g %g %g) vpG (%g %g %g)\n", i,
                            ref[i].scalars.x, g[i].scalars.x,
                            ref[i].volPos.x, ref[i].volPos.y, ref[i].volPos.z,
                            g[i].volPos.x, g[i].volPos.y, g[i].volPos.z);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "psSpecVolume outputs bitwise-identical (mod +-0)");
}

}  // namespace
