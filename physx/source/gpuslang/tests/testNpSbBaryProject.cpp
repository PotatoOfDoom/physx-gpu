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

// Verification of npSbBaryProject.slang (sbBaryProject): the soft-body
// barycentric projection/mass family barycentricProject / barycentricProjectTri
// / computeInvMass(bary) (softBody.cu L608-648) with getSoftBodyInvMass
// (softBody.cuh L38-42), against an exact CPU transcription.  Bitwise mod +-0.

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
struct BpIn  { F4 a, b, c, d, bary; };
struct BpOut { F4 projTet, projTri, scalars; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b) || (a == 0.f && b == 0.f); }

static void refBp(const BpIn& s, BpOut& o)
{
    const F4 a = s.a, b = s.b, c = s.c, d = s.d, bc = s.bary;

    F4 projTet, projTri;
    projTet.x = a.x * bc.x + b.x * bc.y + c.x * bc.z + d.x * bc.w;
    projTet.y = a.y * bc.x + b.y * bc.y + c.y * bc.z + d.y * bc.w;
    projTet.z = a.z * bc.x + b.z * bc.y + c.z * bc.z + d.z * bc.w;
    projTet.w = a.w * bc.x + b.w * bc.y + c.w * bc.z + d.w * bc.w;
    projTri.x = a.x * bc.x + b.x * bc.y + c.x * bc.z;
    projTri.y = a.y * bc.x + b.y * bc.y + c.y * bc.z;
    projTri.z = a.z * bc.x + b.z * bc.y + c.z * bc.z;
    projTri.w = a.w * bc.x + b.w * bc.y + c.w * bc.z;

    const float invMass = a.w * bc.x + b.w * bc.y + c.w * bc.z + d.w * bc.w;
    const float scale = std::sqrt(bc.x * bc.x + bc.y * bc.y + bc.z * bc.z + bc.w * bc.w);
    const float tetInvMass = invMass * scale;

    o.projTet = projTet;
    o.projTri = projTri;
    o.scalars = F4{ tetInvMass, 0.f, 0.f, 0.f };
}

PXG_TEST(npSbBaryProject)
{
    std::mt19937 rng(229001);
    const uint32_t N = 5000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<BpIn> in(N);
    std::vector<BpOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        BpIn s{};
        s.a = F4{ rf(-3.f, 3.f), rf(-3.f, 3.f), rf(-3.f, 3.f), rf(0.f, 3.f) };
        s.b = F4{ rf(-3.f, 3.f), rf(-3.f, 3.f), rf(-3.f, 3.f), rf(0.f, 3.f) };
        s.c = F4{ rf(-3.f, 3.f), rf(-3.f, 3.f), rf(-3.f, 3.f), rf(0.f, 3.f) };
        s.d = F4{ rf(-3.f, 3.f), rf(-3.f, 3.f), rf(-3.f, 3.f), rf(0.f, 3.f) };
        // barycentric weights: mostly a normalized convex combination, but
        // sometimes with a negative coordinate (valid for extrapolated points).
        float w0 = rf(-0.3f, 1.f), w1 = rf(-0.3f, 1.f), w2 = rf(-0.3f, 1.f), w3 = rf(-0.3f, 1.f);
        if ((i % 3u) == 0u)
        {
            const float sum = w0 + w1 + w2 + w3;
            if (std::fabs(sum) > 1e-3f) { w0 /= sum; w1 /= sum; w2 /= sum; w3 /= sum; }
        }
        s.bary = F4{ w0, w1, w2, w3 };
        in[i] = s;
        refBp(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<BpOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npSbBaryProject.spv", "sbBaryProject",
             { (uint32_t)sizeof(BpIn), (uint32_t)sizeof(BpOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<BpOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* rfp = &ref[i].projTet.x;
        const float* qfp = &g[i].projTet.x;
        const size_t nf = sizeof(BpOut) / 4;
        bool ok = true;
        for (size_t xi = 0; xi < nf && ok; ++xi) ok = eqFZ(rfp[xi], qfp[xi]);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u tetR (%g %g %g %g) imR %g tetG (%g %g %g %g) imG %g\n", i,
                            ref[i].projTet.x, ref[i].projTet.y, ref[i].projTet.z, ref[i].projTet.w, ref[i].scalars.x,
                            g[i].projTet.x, g[i].projTet.y, g[i].projTet.z, g[i].projTet.w, g[i].scalars.x);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "sbBaryProject outputs bitwise-identical (mod +-0)");
}

}  // namespace
