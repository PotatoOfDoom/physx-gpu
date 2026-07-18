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

// Verification of npSbSolveNeoHookean.slang (sbSolveNeoHookean): the NeoHookean
// XPBD tet solve tetrahedronsSolveInnerNeoHookean (softBodyGM.cu L657-717),
// with the per-tet multiplier slot modeled as an in/out scalar, against an
// exact CPU transcription.  Outputs bitwise mod +-0 (eqFZ).

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
struct NhIn  { F4 pc0, pc1, pc2, qc0, qc1, qc2, im, prm; };
struct NhOut { F4 d0, d1, d2; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b) || (a == 0.f && b == 0.f); }

struct V3 { float x, y, z; };
static V3 add(V3 a, V3 b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 sub(V3 a, V3 b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 muls(V3 a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; }
static V3 neg(V3 a) { return V3{ -a.x, -a.y, -a.z }; }
static float magSq(V3 v) { return v.x * v.x + v.y * v.y + v.z * v.z; }

// PxMat33::transform: (c0*v.x + c1*v.y) + c2*v.z
static V3 mat33Transform(V3 c0, V3 c1, V3 c2, V3 v)
{
    return add(add(muls(c0, v.x), muls(c1, v.y)), muls(c2, v.z));
}

static void refNh(const NhIn& s, NhOut& o)
{
    const V3 P0 = V3{ s.pc0.x, s.pc0.y, s.pc0.z }, P1 = V3{ s.pc1.x, s.pc1.y, s.pc1.z }, P2 = V3{ s.pc2.x, s.pc2.y, s.pc2.z };
    const V3 Q0 = V3{ s.qc0.x, s.qc0.y, s.qc0.z }, Q1 = V3{ s.qc1.x, s.qc1.y, s.qc1.z }, Q2 = V3{ s.qc2.x, s.qc2.y, s.qc2.z };
    const float invDtSq = s.prm.x, invE = s.prm.y;
    const bool isTGS = f2u(s.prm.z) == 1u;
    float totalLambdaX = s.prm.w;

    const V3 F0 = mat33Transform(P0, P1, P2, Q0);
    const V3 F1 = mat33Transform(P0, P1, P2, Q1);
    const V3 F2 = mat33Transform(P0, P1, P2, Q2);

    // g_i = F.c0*2*q0 + F.c1*2*q1 + F.c2*2*q2, each product ((F.cj*2)*q)
    const V3 g1 = add(add(muls(muls(F0, 2.0f), Q0.x), muls(muls(F1, 2.0f), Q1.x)), muls(muls(F2, 2.0f), Q2.x));
    const V3 g2 = add(add(muls(muls(F0, 2.0f), Q0.y), muls(muls(F1, 2.0f), Q1.y)), muls(muls(F2, 2.0f), Q2.y));
    const V3 g3 = add(add(muls(muls(F0, 2.0f), Q0.z), muls(muls(F1, 2.0f), Q1.z)), muls(muls(F2, 2.0f), Q2.z));

    const float C = magSq(F0) + magSq(F1) + magSq(F2) - 3.0f;

    const V3 g0 = sub(sub(neg(g1), g2), g3);

    const float w = magSq(g0) * s.im.x + magSq(g1) * s.im.y + magSq(g2) * s.im.z + magSq(g3) * s.im.w;

    const float alpha = invDtSq * invE;
    float dlambda;

    if (!isTGS)
    {
        dlambda = -(C + alpha * totalLambdaX) / (w + alpha);
        totalLambdaX += dlambda;
    }
    else
    {
        dlambda = -C / (w + alpha);
    }

    const V3 d0 = muls(g1, dlambda), d1 = muls(g2, dlambda), d2 = muls(g3, dlambda);
    o.d0 = F4{ d0.x, d0.y, d0.z, dlambda };
    o.d1 = F4{ d1.x, d1.y, d1.z, totalLambdaX };
    o.d2 = F4{ d2.x, d2.y, d2.z, 0.f };
}

PXG_TEST(npSbSolveNeoHookean)
{
    std::mt19937 rng(252001);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<NhIn> in(N);
    std::vector<NhOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        NhIn s{};
        s.pc0 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
        s.pc1 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
        s.pc2 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
        s.qc0 = F4{ rf(-1.5f, 1.5f), rf(-1.5f, 1.5f), rf(-1.5f, 1.5f), 0.f };
        s.qc1 = F4{ rf(-1.5f, 1.5f), rf(-1.5f, 1.5f), rf(-1.5f, 1.5f), 0.f };
        s.qc2 = F4{ rf(-1.5f, 1.5f), rf(-1.5f, 1.5f), rf(-1.5f, 1.5f), 0.f };
        s.im  = F4{ rf(0.f, 3.f), rf(0.f, 3.f), rf(0.f, 3.f), rf(0.f, 3.f) };
        const uint32_t isTGS = i % 2u;
        float tl = rf(-2.f, 2.f);
        // zero-accumulator stripe (fresh PGS iteration)
        if ((i % 9u) == 4u) tl = 0.f;
        s.prm = F4{ rf(100.f, 40000.f), rf(1e-8f, 1e-3f), u2f(isTGS), tl };  // invDtSq, invE (1/youngs), isTGS, totalLambdaIn
        // rest-pose stripe: P = identity, Qinv = identity -> F = I, C = 0 -> dlambda 0
        if ((i % 11u) == 6u)
        {
            s.pc0 = F4{ 1.f, 0.f, 0.f, 0.f }; s.pc1 = F4{ 0.f, 1.f, 0.f, 0.f }; s.pc2 = F4{ 0.f, 0.f, 1.f, 0.f };
            s.qc0 = F4{ 1.f, 0.f, 0.f, 0.f }; s.qc1 = F4{ 0.f, 1.f, 0.f, 0.f }; s.qc2 = F4{ 0.f, 0.f, 1.f, 0.f };
        }
        in[i] = s;
        refNh(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<NhOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npSbSolveNeoHookean.spv", "sbSolveNeoHookean",
             { (uint32_t)sizeof(NhIn), (uint32_t)sizeof(NhOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<NhOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        bool ok = eqFZ(ref[i].d0.x, g[i].d0.x) && eqFZ(ref[i].d0.y, g[i].d0.y) && eqFZ(ref[i].d0.z, g[i].d0.z)
               && eqFZ(ref[i].d0.w, g[i].d0.w)
               && eqFZ(ref[i].d1.x, g[i].d1.x) && eqFZ(ref[i].d1.y, g[i].d1.y) && eqFZ(ref[i].d1.z, g[i].d1.z)
               && eqFZ(ref[i].d1.w, g[i].d1.w)
               && eqFZ(ref[i].d2.x, g[i].d2.x) && eqFZ(ref[i].d2.y, g[i].d2.y) && eqFZ(ref[i].d2.z, g[i].d2.z);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u dlR %g dlG %g tlR %g tlG %g d0R (%g %g %g) d0G (%g %g %g)\n", i,
                            ref[i].d0.w, g[i].d0.w, ref[i].d1.w, g[i].d1.w,
                            ref[i].d0.x, ref[i].d0.y, ref[i].d0.z, g[i].d0.x, g[i].d0.y, g[i].d0.z);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "sbSolveNeoHookean outputs bitwise-identical (mod +-0)");
}

}  // namespace
