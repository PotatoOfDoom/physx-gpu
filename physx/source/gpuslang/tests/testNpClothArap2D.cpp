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

// Verification of npClothArap2D.slang (clothArap2D): the XPBD ARAP membrane
// constraint ARAPConstraint_F2X2 (FEMClothUtil.cuh L335-368), composing the
// 2x2 deformation gradient, extractRotation2D polar decomposition, constraint
// gradient and queryDeltaLambda, against an exact CPU transcription.  Outputs
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
struct ArIn  { F4 qinv, x01x02, dx0dx1, dx2LamAlpha, mass; };
struct ArOut { F4 dx0dx1, dx2Lam; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b) || (a == 0.f && b == 0.f); }

static const float FEMCLOTH_THRESHOLD = 1.0e-14f;

struct V2 { float x, y; };
static V2 sub2(V2 a, V2 b) { return V2{ a.x - b.x, a.y - b.y }; }
static V2 neg2(V2 a) { return V2{ -a.x, -a.y }; }
static V2 scl2(float s, V2 a) { return V2{ s * a.x, s * a.y }; }
static V2 add2(V2 a, V2 b) { return V2{ a.x + b.x, a.y + b.y }; }
static float dot2(V2 a, V2 b) { return a.x * b.x + a.y * b.y; }
static float magSq2(V2 v) { return v.x * v.x + v.y * v.y; }

// extractRotation2D (tranche 217 transcription)
static void extractRotation2D(V2& r0, V2& r1, V2 f0, V2 f1)
{
    const float S2x = dot2(f0, f0), S2y = dot2(f1, f1), S2z = dot2(f0, f1);
    const float det = S2x * S2y - S2z * S2z;
    if (det < FEMCLOTH_THRESHOLD) { r0 = V2{ 1.f, 0.f }; r1 = V2{ 0.f, 1.f }; return; }
    const float s0 = std::sqrt(det);
    const float t = std::sqrt(S2x + S2y + 2.0f * s0);
    if (t < FEMCLOTH_THRESHOLD) { r0 = V2{ 1.f, 0.f }; r1 = V2{ 0.f, 1.f }; return; }
    const float tInv = 1.0f / t;
    float Sx = S2x, Sy = S2y, Sz = S2z;
    Sx = Sx + s0; Sy = Sy + s0;
    Sx = Sx * tInv; Sy = Sy * tInv; Sz = Sz * tInv;
    const float sDet = Sx * Sy - Sz * Sz;
    if (sDet < FEMCLOTH_THRESHOLD) { r0 = V2{ 1.f, 0.f }; r1 = V2{ 0.f, 1.f }; return; }
    const float sDetInv = 1.0f / sDet;
    float SInvx = Sy, SInvy = Sx, SInvz = -Sz;
    SInvx = SInvx * sDetInv; SInvy = SInvy * sDetInv; SInvz = SInvz * sDetInv;
    r0 = add2(scl2(SInvx, f0), scl2(SInvz, f1));
    r1 = add2(scl2(SInvz, f0), scl2(SInvy, f1));
}

static void refArap(const ArIn& s, ArOut& o)
{
    const F4 QInv = s.qinv;
    const V2 x01 = V2{ s.x01x02.x, s.x01x02.y };
    const V2 x02 = V2{ s.x01x02.z, s.x01x02.w };
    V2 dx0 = V2{ s.dx0dx1.x, s.dx0dx1.y };
    V2 dx1 = V2{ s.dx0dx1.z, s.dx0dx1.w };
    V2 dx2 = V2{ s.dx2LamAlpha.x, s.dx2LamAlpha.y };
    float lambda = s.dx2LamAlpha.z;
    const float alphaTilde = s.dx2LamAlpha.w;
    const float massInv0 = s.mass.x, massInv1 = s.mass.y, massInv2 = s.mass.z;

    // queryDeformationGradient_F2x2
    const V2 f0 = add2(scl2(QInv.x, x01), scl2(QInv.y, x02));
    const V2 f1 = add2(scl2(QInv.z, x01), scl2(QInv.w, x02));

    V2 r0, r1;
    extractRotation2D(r0, r1, f0, f1);

    const V2 FMinusR0 = sub2(f0, r0);
    const V2 FMinusR1 = sub2(f1, r1);

    const float C = std::sqrt(dot2(FMinusR0, FMinusR0) + dot2(FMinusR1, FMinusR1));

    if (C > FEMCLOTH_THRESHOLD)
    {
        const float CInv = 1.0f / C;
        const V2 pC_pF0 = scl2(CInv, FMinusR0);
        const V2 pC_pF1 = scl2(CInv, FMinusR1);

        // queryConstraintGradient_F2x2
        const V2 grad1 = add2(scl2(QInv.x, pC_pF0), scl2(QInv.z, pC_pF1));
        const V2 grad2 = add2(scl2(QInv.y, pC_pF0), scl2(QInv.w, pC_pF1));
        const V2 grad0 = sub2(neg2(grad1), grad2);   // -grad1 - grad2

        const float denom = (massInv0 * magSq2(grad0) + massInv1 * magSq2(grad1) + massInv2 * magSq2(grad2)) + alphaTilde;
        float deltaLambda;
        if (denom < FEMCLOTH_THRESHOLD) deltaLambda = 0.0f;
        else deltaLambda = (-C - alphaTilde * lambda) / denom;

        lambda = lambda + deltaLambda;

        dx0 = add2(dx0, scl2(massInv0 * deltaLambda, grad0));
        dx1 = add2(dx1, scl2(massInv1 * deltaLambda, grad1));
        dx2 = add2(dx2, scl2(massInv2 * deltaLambda, grad2));
    }

    o.dx0dx1 = F4{ dx0.x, dx0.y, dx1.x, dx1.y };
    o.dx2Lam = F4{ dx2.x, dx2.y, lambda, 0.0f };
}

PXG_TEST(npClothArap2D)
{
    std::mt19937 rng(219001);
    const uint32_t N = 5000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<ArIn> in(N);
    std::vector<ArOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        ArIn s{};
        s.qinv   = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f) };
        s.x01x02 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f) };
        s.dx0dx1 = F4{ rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-1.f, 1.f) };
        float lambda = rf(-2.f, 2.f);
        float alphaTilde = rf(0.f, 2.f);
        float mi0 = rf(0.f, 3.f), mi1 = rf(0.f, 3.f), mi2 = rf(0.f, 3.f);

        // rigid stripe: F == identity exactly (QInv=I, x01=(1,0), x02=(0,1)) so
        // extractRotation2D returns R=F, C=0, and the C>THRESHOLD block is skipped.
        if ((i % 9u) == 4u)
        {
            s.qinv   = F4{ 1.f, 0.f, 0.f, 1.f };
            s.x01x02 = F4{ 1.f, 0.f, 0.f, 1.f };
        }
        // denom-guard stripe: all massInv 0 and a tiny alphaTilde so denom < 1e-14
        // -> deltaLambda 0 (lambda/dx unchanged even though C > THRESHOLD).
        if ((i % 11u) == 6u)
        {
            mi0 = mi1 = mi2 = 0.f;
            alphaTilde = 1.0e-15f;
        }

        s.dx2LamAlpha = F4{ rf(-1.f, 1.f), rf(-1.f, 1.f), lambda, alphaTilde };
        s.mass = F4{ mi0, mi1, mi2, 0.f };
        in[i] = s;
        refArap(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<ArOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npClothArap2D.spv", "clothArap2D",
             { (uint32_t)sizeof(ArIn), (uint32_t)sizeof(ArOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<ArOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* rfp = &ref[i].dx0dx1.x;
        const float* qfp = &g[i].dx0dx1.x;
        const size_t nf = sizeof(ArOut) / 4;
        bool ok = true;
        for (size_t x = 0; x < nf && ok; ++x) ok = eqFZ(rfp[x], qfp[x]);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u lamR %g lamG %g dx0R (%g %g) dx0G (%g %g)\n", i,
                            ref[i].dx2Lam.z, g[i].dx2Lam.z,
                            ref[i].dx0dx1.x, ref[i].dx0dx1.y, g[i].dx0dx1.x, g[i].dx0dx1.y);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "clothArap2D outputs bitwise-identical (mod +-0)");
}

}  // namespace
