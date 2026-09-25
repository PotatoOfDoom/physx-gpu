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

// Verification of npIsoVecNormalize.slang (isoVecNormalize): the isosurface
// vector helpers areaTimes2 and the two normalize overloads
// (isosurfaceExtraction.cu L895-898 / L956-962 / L997-1001), against an exact
// CPU transcription.  Outputs bitwise mod +-0 (eqFZ).

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
struct NvIn  { F4 v1, v2, v4; };
struct NvOut { F4 normVec3, normVec4, scalars; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed

static float pmax(float a, float b) { return (a > b) ? a : b; }

static void refNv(const NvIn& s, NvOut& o)
{
    const float v1x = s.v1.x, v1y = s.v1.y, v1z = s.v1.z;
    const float v2x = s.v2.x, v2y = s.v2.y, v2z = s.v2.z;

    // areaTimes2 = |v1 x v2|
    const float crx = v1y * v2z - v1z * v2y;
    const float cry = v1z * v2x - v1x * v2z;
    const float crz = v1x * v2y - v1y * v2x;
    const float area = std::sqrt(crx * crx + cry * cry + crz * crz);

    // normalize(PxVec3): m = |v|; PxVec3::operator/=(m) is reciprocal-multiply
    // (f = 1/m; x*=f; y*=f; z*=f), NOT per-component division.
    float n3x = v1x, n3y = v1y, n3z = v1z;
    const float m = std::sqrt(v1x * v1x + v1y * v1y + v1z * v1z);
    if (m > 0.0f) { const float invM = 1.0f / m; n3x = v1x * invM; n3y = v1y * invM; n3z = v1z * invM; }

    // normalize(PxVec4): reciprocal of eps-clamped xyz magnitude, keep w
    const float sc = 1.0f / pmax(1.0e-6f, std::sqrt(s.v4.x * s.v4.x + s.v4.y * s.v4.y + s.v4.z * s.v4.z));
    const F4 nv4 = F4{ sc * s.v4.x, sc * s.v4.y, sc * s.v4.z, s.v4.w };

    o.normVec3 = F4{ n3x, n3y, n3z, 0.f };
    o.normVec4 = nv4;
    o.scalars = F4{ area, 0.f, 0.f, 0.f };
}

PXG_TEST(npIsoVecNormalize)
{
    std::mt19937 rng(230001);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<NvIn> in(N);
    std::vector<NvOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        NvIn s{};
        s.v1 = F4{ rf(-3.f, 3.f), rf(-3.f, 3.f), rf(-3.f, 3.f), 0.f };
        s.v2 = F4{ rf(-3.f, 3.f), rf(-3.f, 3.f), rf(-3.f, 3.f), 0.f };
        s.v4 = F4{ rf(-3.f, 3.f), rf(-3.f, 3.f), rf(-3.f, 3.f), rf(-2.f, 2.f) };
        // zero-v1 stripe: exercises the vec3 normalize m==0 pass-through (returns
        // v1 unchanged = 0) and areaTimes2 with a zero cross.
        if ((i % 11u) == 4u) s.v1 = F4{ 0.f, 0.f, 0.f, 0.f };
        // tiny-v4 stripe: xyz magnitude below 1e-6 so the vec4 normalize eps-clamp
        // (max(1e-6, ...)) governs the reciprocal instead of the true magnitude.
        if ((i % 13u) == 6u) s.v4 = F4{ rf(-1e-8f, 1e-8f), rf(-1e-8f, 1e-8f), rf(-1e-8f, 1e-8f), rf(-2.f, 2.f) };
        in[i] = s;
        refNv(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<NvOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npIsoVecNormalize.spv", "isoVecNormalize",
             { (uint32_t)sizeof(NvIn), (uint32_t)sizeof(NvOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<NvOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* rfp = &ref[i].normVec3.x;
        const float* qfp = &g[i].normVec3.x;
        const size_t nf = sizeof(NvOut) / 4;
        bool ok = true;
        for (size_t xi = 0; xi < nf && ok; ++xi) ok = eqFZ(rfp[xi], qfp[xi]);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u areaR %g areaG %g n3R (%g %g %g) n3G (%g %g %g)\n", i,
                            ref[i].scalars.x, g[i].scalars.x,
                            ref[i].normVec3.x, ref[i].normVec3.y, ref[i].normVec3.z,
                            g[i].normVec3.x, g[i].normVec3.y, g[i].normVec3.z);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "isoVecNormalize outputs bitwise-identical (mod +-0)");
}

}  // namespace
