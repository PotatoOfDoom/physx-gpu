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

// Verification of npDefMaterial.slang (defMaterial): the deformable
// material/geometry helpers projectVectorOntoPlane and lameParameters
// (deformableUtils.cuh L118-130), against an exact CPU transcription.
// Outputs bitwise modulo the documented +-0 (eqFZ).

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
struct DmIn  { F4 v, n, misc; };
struct DmOut { F4 proj, lame; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b) || (a == 0.f && b == 0.f); }

static float dot3(const F4& a, const F4& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }

static void refDef(const DmIn& s, DmOut& o)
{
    const float scale = dot3(s.n, s.v) / dot3(s.n, s.n);
    o.proj = F4{ s.v.x - scale * s.n.x, s.v.y - scale * s.n.y, s.v.z - scale * s.n.z, 0.f };

    const float Young = s.misc.x, Poisson = s.misc.y;
    const float lambda = Young * Poisson / ((1.0f + Poisson) * (1.0f - 2.0f * Poisson));
    const float mu = Young / (2.0f * (1.0f + Poisson));
    o.lame = F4{ lambda, mu, 0.f, 0.f };
}

PXG_TEST(npDefMaterial)
{
    std::mt19937 rng(210001);
    const uint32_t N = 3000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<DmIn> in(N);
    std::vector<DmOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        DmIn s{};
        s.v = F4{ rf(-3, 3), rf(-3, 3), rf(-3, 3), 0.f };
        // planeNormal: keep away from zero so magnitudeSquared > 0
        s.n = F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), 0.f };
        if (dot3(s.n, s.n) < 0.25f) s.n.x += 1.5f;
        float young = rf(1.f, 1000.f);
        float poisson = rf(-0.4f, 0.49f);           // valid range; avoids the 1-2*Poisson->0 pole
        s.misc = F4{ young, poisson, 0.f, 0.f };
        in[i] = s;
        refDef(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<DmOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npDefMaterial.spv", "defMaterial",
             { (uint32_t)sizeof(DmIn), (uint32_t)sizeof(DmOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<DmOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* rfp = &ref[i].proj.x;
        const float* qfp = &g[i].proj.x;
        const size_t nf = sizeof(DmOut) / 4;
        bool ok = true;
        for (size_t x = 0; x < nf && ok; ++x) ok = eqFZ(rfp[x], qfp[x]);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u projR (%g %g %g) lame (%g %g) G proj (%g %g %g) lame (%g %g)\n", i,
                            ref[i].proj.x, ref[i].proj.y, ref[i].proj.z, ref[i].lame.x, ref[i].lame.y,
                            g[i].proj.x, g[i].proj.y, g[i].proj.z, g[i].lame.x, g[i].lame.y);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "defMaterial outputs bitwise-identical (mod +-0)");
}

}  // namespace
