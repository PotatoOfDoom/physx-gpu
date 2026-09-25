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

// Verification of npClothDeltaLambda.slang (clothDeltaLambda): the FEM-cloth
// XPBD delta-lambda family queryDeltaLambda (FEMClothUtil.cuh L211-...), against
// an exact CPU transcription.  Outputs bitwise modulo the documented +-0 (eqFZ).

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
struct DlIn  { F4 d0, d1, d2, d3, mi, scal, damp; };
struct DlOut { F4 res; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed

static const float FEMCLOTH_THRESHOLD = 1.0e-14f;
static float magSq2(const F4& v) { return v.x * v.x + v.y * v.y; }
static float magSq3(const F4& v) { return v.x * v.x + v.y * v.y + v.z * v.z; }

static float qdl3_2d(float C, const F4& d0, const F4& d1, const F4& d2, float at, float lam, float mi0, float mi1, float mi2)
{
    const float denom = (mi0 * magSq2(d0) + mi1 * magSq2(d1) + mi2 * magSq2(d2)) + at;
    if (denom < FEMCLOTH_THRESHOLD) return 0.0f;
    return (-C - at * lam) / denom;
}
static float qdl3_3d_damp(float C, const F4& d0, const F4& d1, const F4& d2, float at, float lam,
                          float mi0, float mi1, float mi2, float dmp, float dtInv, float dCdT)
{
    const float denom = (1.0f + dmp * dtInv) * (mi0 * magSq3(d0) + mi1 * magSq3(d1) + mi2 * magSq3(d2)) + at;
    if (denom < FEMCLOTH_THRESHOLD) return 0.0f;
    return -(C + at * lam + dmp * dCdT) / denom;
}
static float qdl4(float C, const F4& d0, const F4& d1, const F4& d2, const F4& d3, float at, float lam,
                  float mi0, float mi1, float mi2, float mi3)
{
    const float denom = (mi0 * magSq3(d0) + mi1 * magSq3(d1) + mi2 * magSq3(d2) + mi3 * magSq3(d3)) + at;
    if (denom < FEMCLOTH_THRESHOLD) return 0.0f;
    return (-C - at * lam) / denom;
}
static float qdl4_damp(float C, const F4& d0, const F4& d1, const F4& d2, const F4& d3, float at, float lam,
                       float mi0, float mi1, float mi2, float mi3, float dmp, float dtInv, float dCdT)
{
    const float denom = (1.0f + dmp * dtInv) * (mi0 * magSq3(d0) + mi1 * magSq3(d1) + mi2 * magSq3(d2) + mi3 * magSq3(d3)) + at;
    if (denom < FEMCLOTH_THRESHOLD) return 0.0f;
    return -(C + at * lam + dmp * dCdT) / denom;
}

static void refDl(const DlIn& s, DlOut& o)
{
    const float C = s.scal.x, at = s.scal.y, lam = s.scal.z;
    const float dmp = s.damp.x, dtInv = s.damp.y, dCdT = s.damp.z;
    const float mi0 = s.mi.x, mi1 = s.mi.y, mi2 = s.mi.z, mi3 = s.mi.w;
    o.res = F4{
        qdl3_2d(C, s.d0, s.d1, s.d2, at, lam, mi0, mi1, mi2),
        qdl3_3d_damp(C, s.d0, s.d1, s.d2, at, lam, mi0, mi1, mi2, dmp, dtInv, dCdT),
        qdl4(C, s.d0, s.d1, s.d2, s.d3, at, lam, mi0, mi1, mi2, mi3),
        qdl4_damp(C, s.d0, s.d1, s.d2, s.d3, at, lam, mi0, mi1, mi2, mi3, dmp, dtInv, dCdT)
    };
}

PXG_TEST(npClothDeltaLambda)
{
    std::mt19937 rng(216001);
    const uint32_t N = 4000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto rv = [&](float a, float b) { return F4{ rf(a, b), rf(a, b), rf(a, b), 0.f }; };

    std::vector<DlIn> in(N);
    std::vector<DlOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        DlIn s{};
        s.d0 = rv(-2, 2); s.d1 = rv(-2, 2); s.d2 = rv(-2, 2); s.d3 = rv(-2, 2);
        s.mi = F4{ rf(0.f, 3.f), rf(0.f, 3.f), rf(0.f, 3.f), rf(0.f, 3.f) };
        s.scal = F4{ rf(-3.f, 3.f), rf(0.f, 2.f), rf(-3.f, 3.f), 0.f };   // C, alphaTilde, lambda
        s.damp = F4{ rf(0.f, 2.f), rf(0.f, 60.f), rf(-2.f, 2.f), 0.f };   // damping, dtInv, dCdT
        // threshold stripe: massInv all 0 and alphaTilde tiny -> denom < 1e-14 -> return 0
        if ((i % 15u) == 7u)
        {
            s.mi = F4{ 0, 0, 0, 0 };
            s.scal.y = 1.0e-15f;   // alphaTilde
        }
        in[i] = s;
        refDl(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<DlOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npClothDeltaLambda.spv", "clothDeltaLambda",
             { (uint32_t)sizeof(DlIn), (uint32_t)sizeof(DlOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<DlOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* rfp = &ref[i].res.x;
        const float* qfp = &g[i].res.x;
        bool ok = true;
        for (int x = 0; x < 4 && ok; ++x) ok = eqFZ(rfp[x], qfp[x]);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u R (%g %g %g %g) G (%g %g %g %g)\n", i, ref[i].res.x, ref[i].res.y,
                            ref[i].res.z, ref[i].res.w, g[i].res.x, g[i].res.y, g[i].res.z, g[i].res.w);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "clothDeltaLambda outputs bitwise-identical (mod +-0)");
}

}  // namespace
