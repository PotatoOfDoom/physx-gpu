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

// Verification of npClothExtractRot2D.slang (clothExtractRot2D): the 2x2 polar
// decomposition extractRotation2D (FEMClothUtil.cuh L88-144), computing the
// rotation R from a deformation gradient F via the closed-form square root of
// F^T F, against an exact CPU transcription.  Outputs bitwise mod +-0 (eqFZ).
//
// Only the first (det < THRESHOLD) early-return is reachable: if det passes,
// then S2.x + S2.y + 2*sqrt(det) >= 2*sqrt(det) >= 2e-7 (t-guard) and
// sDet == sqrt(det) >= 1e-7 (sDet-guard), so the latter two returns are
// mathematically unreachable defensive code and are exercised only along the
// main path.  The det branch is forced by a degenerate-F (rank-deficient) stripe.

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
struct ErIn  { F4 f; };   // f0.xy, f1.xy
struct ErOut { F4 r; };   // r0.xy, r1.xy

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b) || (a == 0.f && b == 0.f); }

static const float FEMCLOTH_THRESHOLD = 1.0e-14f;

// exact transcription of extractRotation2D
static void refExtract(const ErIn& s, ErOut& o)
{
    const float f0x = s.f.x, f0y = s.f.y;
    const float f1x = s.f.z, f1y = s.f.w;

    const float S2x = f0x * f0x + f0y * f0y;   // f0.dot(f0)
    const float S2y = f1x * f1x + f1y * f1y;   // f1.dot(f1)
    const float S2z = f0x * f1x + f0y * f1y;   // f0.dot(f1)
    const float det = S2x * S2y - S2z * S2z;

    if (det < FEMCLOTH_THRESHOLD)
    {
        o.r = F4{ 1.0f, 0.0f, 0.0f, 1.0f };
        return;
    }

    const float s0 = std::sqrt(det);
    const float t = std::sqrt(S2x + S2y + 2.0f * s0);

    if (t < FEMCLOTH_THRESHOLD)
    {
        o.r = F4{ 1.0f, 0.0f, 0.0f, 1.0f };
        return;
    }
    const float tInv = 1.0f / t;

    float Sx = S2x, Sy = S2y, Sz = S2z;
    Sx = Sx + s0;
    Sy = Sy + s0;
    Sx = Sx * tInv;
    Sy = Sy * tInv;
    Sz = Sz * tInv;

    const float sDet = Sx * Sy - Sz * Sz;

    if (sDet < FEMCLOTH_THRESHOLD)
    {
        o.r = F4{ 1.0f, 0.0f, 0.0f, 1.0f };
        return;
    }

    const float sDetInv = 1.0f / sDet;

    float SInvx = Sy, SInvy = Sx, SInvz = -Sz;
    SInvx = SInvx * sDetInv;
    SInvy = SInvy * sDetInv;
    SInvz = SInvz * sDetInv;

    // r0 = SInv.x*f0 + SInv.z*f1 ; r1 = SInv.z*f0 + SInv.y*f1
    const float r0x = SInvx * f0x + SInvz * f1x;
    const float r0y = SInvx * f0y + SInvz * f1y;
    const float r1x = SInvz * f0x + SInvy * f1x;
    const float r1y = SInvz * f0y + SInvy * f1y;

    o.r = F4{ r0x, r0y, r1x, r1y };
}

PXG_TEST(npClothExtractRot2D)
{
    std::mt19937 rng(217001);
    const uint32_t N = 4000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<ErIn> in(N);
    std::vector<ErOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        ErIn s{};
        float f0x = rf(-3.f, 3.f), f0y = rf(-3.f, 3.f);
        float f1x = rf(-3.f, 3.f), f1y = rf(-3.f, 3.f);

        // Degenerate-F stripe: force det < THRESHOLD to reach the identity return.
        // (a) exactly-zero F, and (b) rank-deficient F (f1 parallel to f0).
        if ((i % 11u) == 4u) { f0x = f0y = f1x = f1y = 0.f; }
        else if ((i % 11u) == 8u)
        {
            const float k = rf(-2.f, 2.f);
            f1x = k * f0x;
            f1y = k * f0y;   // f1 = k*f0 -> det = 0 (Cauchy-Schwarz equality)
        }
        // Near-identity / small-rotation stripe: F close to a pure rotation, so R
        // should recover it and the main path runs with well-conditioned S.
        else if ((i % 7u) == 2u)
        {
            const float c = rf(-1.f, 1.f), sn = rf(-1.f, 1.f);
            f0x = c;  f0y = sn;
            f1x = -sn; f1y = c;
        }

        s.f = F4{ f0x, f0y, f1x, f1y };
        in[i] = s;
        refExtract(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<ErOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npClothExtractRot2D.spv", "clothExtractRot2D",
             { (uint32_t)sizeof(ErIn), (uint32_t)sizeof(ErOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<ErOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* rfp = &ref[i].r.x;
        const float* qfp = &g[i].r.x;
        bool ok = true;
        for (size_t x = 0; x < 4 && ok; ++x) ok = eqFZ(rfp[x], qfp[x]);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u f(%g %g %g %g) R (%g %g %g %g) G (%g %g %g %g)\n", i,
                            in[i].f.x, in[i].f.y, in[i].f.z, in[i].f.w,
                            ref[i].r.x, ref[i].r.y, ref[i].r.z, ref[i].r.w,
                            g[i].r.x, g[i].r.y, g[i].r.z, g[i].r.w);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "clothExtractRot2D outputs bitwise-identical (mod +-0)");
}

}  // namespace
