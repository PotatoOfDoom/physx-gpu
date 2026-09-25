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

// Verification of npSbCompliance.slang (sbCompliance): the soft-body FEM
// compliance / constraint-gradient helpers evalStiffnessMatrix6x6,
// multiplyStrainByCompliance and compute_dCdx (softBodyGM.cu L507-555), against
// an exact CPU transcription.  Outputs bitwise mod +-0 (eqFZ).

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
struct SmIn  { F4 params, ltop, lbot, fc0, fc1, fc2, qc0, qc1, qc2; };
struct SmOut { F4 dCdx1, dCdx2, dCdx3, scalars; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed

struct V3 { float x, y, z; };
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

static float evalStiffness(uint32_t row, uint32_t col, float invE, float nu)
{
    float result = 0.0f;
    if (row == col)
    {
        result = invE;
        if (row >= 3u)
            result = result * (1.0f + nu);
    }
    else if (row < 3u && col < 3u)
    {
        result = -invE * nu;
    }
    return result;
}

static float multiplyStrainByCompliance(float invE, float nu, uint32_t si, V3 ltop, V3 lbot)
{
    return ltop.x * evalStiffness(si, 0u, invE, nu)
         + ltop.y * evalStiffness(si, 1u, invE, nu)
         + ltop.z * evalStiffness(si, 2u, invE, nu)
         + lbot.x * evalStiffness(si, 3u, invE, nu)
         + lbot.y * evalStiffness(si, 4u, invE, nu)
         + lbot.z * evalStiffness(si, 5u, invE, nu);
}

static void refSm(const SmIn& s, SmOut& o)
{
    const float invE = s.params.x, nu = s.params.y;
    const uint32_t si = f2u(s.params.z);
    const V3 ltop = V3{ s.ltop.x, s.ltop.y, s.ltop.z };
    const V3 lbot = V3{ s.lbot.x, s.lbot.y, s.lbot.z };

    const float mult = multiplyStrainByCompliance(invE, nu, si, ltop, lbot);
    const float evalDiag = evalStiffness(si, si, invE, nu);
    const float evalNext = evalStiffness(si, (si + 1u) % 6u, invE, nu);

    const V3 r0 = V3{ s.qc0.x, s.qc1.x, s.qc2.x };
    const V3 r1 = V3{ s.qc0.y, s.qc1.y, s.qc2.y };
    const V3 r2 = V3{ s.qc0.z, s.qc1.z, s.qc2.z };
    const V3 p0 = V3{ s.fc0.x, s.fc1.x, s.fc2.x };
    const V3 p1 = V3{ s.fc0.y, s.fc1.y, s.fc2.y };
    const V3 p2 = V3{ s.fc0.z, s.fc1.z, s.fc2.z };

    o.dCdx1 = F4{ dot3(r0, p0), dot3(r0, p1), dot3(r0, p2), 0.f };
    o.dCdx2 = F4{ dot3(r1, p0), dot3(r1, p1), dot3(r1, p2), 0.f };
    o.dCdx3 = F4{ dot3(r2, p0), dot3(r2, p1), dot3(r2, p2), 0.f };
    o.scalars = F4{ mult, evalDiag, evalNext, 0.f };
}

PXG_TEST(npSbCompliance)
{
    std::mt19937 rng(231001);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<SmIn> in(N);
    std::vector<SmOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        SmIn s{};
        const uint32_t si = i % 6u;   // cover all six strain rows (0..2 diag/off, 3..5 shear)
        s.params = F4{ rf(0.01f, 5.f), rf(0.f, 0.49f), u2f(si), 0.f };  // invE, nu, strainIndex
        s.ltop = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
        s.lbot = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
        s.fc0 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
        s.fc1 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
        s.fc2 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
        s.qc0 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
        s.qc1 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
        s.qc2 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
        in[i] = s;
        refSm(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<SmOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npSbCompliance.spv", "sbCompliance",
             { (uint32_t)sizeof(SmIn), (uint32_t)sizeof(SmOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<SmOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* rfp = &ref[i].dCdx1.x;
        const float* qfp = &g[i].dCdx1.x;
        const size_t nf = sizeof(SmOut) / 4;
        bool ok = true;
        for (size_t xi = 0; xi < nf && ok; ++xi) ok = eqFZ(rfp[xi], qfp[xi]);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u si %u multR %g multG %g dDiagR %g dDiagG %g\n", i, f2u(in[i].params.z),
                            ref[i].scalars.x, g[i].scalars.x, ref[i].scalars.y, g[i].scalars.y);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "sbCompliance outputs bitwise-identical (mod +-0)");
}

}  // namespace
