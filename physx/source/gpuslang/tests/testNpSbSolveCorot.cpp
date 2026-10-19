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

// Verification of npSbSolveCorot.slang (sbSolveCorot): the co-rotational
// linear-elastic 6-Voigt XPBD tet solve tetrahedronsSolveInner (softBodyGM.cu
// L769-922), with the per-tet 6-float multiplier slot modeled as in/out
// top/bottom vectors, against an exact CPU transcription.  Bitwise mod +-0.

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
struct CrIn  { F4 pc0, pc1, pc2, qc0, qc1, qc2, rt0, rt1, rt2, v0, v1, v2, v3, mat, prm, tlTop, tlBot; };
struct CrOut { F4 d0, d1, d2, tlTopOut, tlBotOut; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed

struct V3 { float x, y, z; };
static V3 add(V3 a, V3 b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 sub(V3 a, V3 b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 muls(V3 a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b)
{
    return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}
static float magSq(V3 v) { return v.x * v.x + v.y * v.y + v.z * v.z; }
static V3 mat33Transform(V3 c0, V3 c1, V3 c2, V3 v)
{
    return add(add(muls(c0, v.x), muls(c1, v.y)), muls(c2, v.z));
}
static float comp(V3 v, uint32_t i) { return (i == 0u) ? v.x : (i == 1u) ? v.y : v.z; }
static float matElem(V3 c0, V3 c1, V3 c2, uint32_t row, uint32_t col)
{
    const V3 c = (col == 0u) ? c0 : (col == 1u) ? c1 : c2;
    return comp(c, row);
}

static float evalStiffness(uint32_t row, uint32_t col, float invE, float nu)
{
    float result = 0.0f;
    if (row == col)
    {
        result = invE;
        if (row >= 3u) result = result * (1.0f + nu);
    }
    else if (row < 3u && col < 3u)
    {
        result = -invE * nu;
    }
    return result;
}

static float multiplyStrainByCompliance(float invE, float nu, uint32_t si, V3 lt, V3 lb)
{
    return lt.x * evalStiffness(si, 0u, invE, nu)
         + lt.y * evalStiffness(si, 1u, invE, nu)
         + lt.z * evalStiffness(si, 2u, invE, nu)
         + lb.x * evalStiffness(si, 3u, invE, nu)
         + lb.y * evalStiffness(si, 4u, invE, nu)
         + lb.z * evalStiffness(si, 5u, invE, nu);
}

static void refCr(const CrIn& s, CrOut& o)
{
    const V3 P0 = V3{ s.pc0.x, s.pc0.y, s.pc0.z }, P1 = V3{ s.pc1.x, s.pc1.y, s.pc1.z }, P2 = V3{ s.pc2.x, s.pc2.y, s.pc2.z };
    const V3 Q0 = V3{ s.qc0.x, s.qc0.y, s.qc0.z }, Q1 = V3{ s.qc1.x, s.qc1.y, s.qc1.z }, Q2 = V3{ s.qc2.x, s.qc2.y, s.qc2.z };
    const V3 R0 = V3{ s.rt0.x, s.rt0.y, s.rt0.z }, R1 = V3{ s.rt1.x, s.rt1.y, s.rt1.z }, R2 = V3{ s.rt2.x, s.rt2.y, s.rt2.z };
    const float youngs = s.mat.x, poissons = s.mat.y, damping = s.mat.z;
    const float invDtSq = s.prm.x, invDt = s.prm.y;
    const bool isTGS = f2u(s.prm.z) == 1u;
    V3 tlTop = V3{ s.tlTop.x, s.tlTop.y, s.tlTop.z };
    V3 tlBot = V3{ s.tlBot.x, s.tlBot.y, s.tlBot.z };

    const V3 F0 = mat33Transform(P0, P1, P2, Q0);
    const V3 F1 = mat33Transform(P0, P1, P2, Q1);
    const V3 F2 = mat33Transform(P0, P1, P2, Q2);

    const float det = dot3(Q0, cross3(Q1, Q2));
    const float volume = 1.0f / (6.0f * det);
    const float volumeSqrt = std::sqrt(volume);

    V3 G0 = muls(add(F0, V3{ F0.x, F1.x, F2.x }), 0.5f);
    V3 G1 = muls(add(F1, V3{ F0.y, F1.y, F2.y }), 0.5f);
    V3 G2 = muls(add(F2, V3{ F0.z, F1.z, F2.z }), 0.5f);
    G0.x -= 1.0f;
    G1.y -= 1.0f;
    G2.z -= 1.0f;
    G0 = muls(G0, volumeSqrt); G1 = muls(G1, volumeSqrt); G2 = muls(G2, volumeSqrt);

    const float alpha = invDtSq * (1.0f / youngs);

    const V3 uv1 = sub(V3{ s.v1.x, s.v1.y, s.v1.z }, V3{ s.v0.x, s.v0.y, s.v0.z });
    const V3 uv2 = sub(V3{ s.v2.x, s.v2.y, s.v2.z }, V3{ s.v0.x, s.v0.y, s.v0.z });
    const V3 uv3 = sub(V3{ s.v3.x, s.v3.y, s.v3.z }, V3{ s.v0.x, s.v0.y, s.v0.z });

    const V3 M0 = mat33Transform(uv1, uv2, uv3, Q0);
    const V3 M1 = mat33Transform(uv1, uv2, uv3, Q1);
    const V3 M2 = mat33Transform(uv1, uv2, uv3, Q2);
    const V3 V0c = mat33Transform(R0, R1, R2, M0);
    const V3 V1c = mat33Transform(R0, R1, R2, M1);
    const V3 V2c = mat33Transform(R0, R1, R2, M2);

    V3 E0 = muls(add(V0c, V3{ V0c.x, V1c.x, V2c.x }), 0.5f);
    V3 E1 = muls(add(V1c, V3{ V0c.y, V1c.y, V2c.y }), 0.5f);
    V3 E2 = muls(add(V2c, V3{ V0c.z, V1c.z, V2c.z }), 0.5f);
    E0 = muls(E0, volumeSqrt); E1 = muls(E1, volumeSqrt); E2 = muls(E2, volumeSqrt);

    const float sqLength1 = magSq(Q0);
    const float sqLength2 = magSq(Q1);
    const float sqLength3 = magSq(Q2);
    const float sqLengthSum = magSq(add(add(Q0, Q1), Q2));

    const float common = (1.0f + damping * invDt) * (volumeSqrt * volumeSqrt * (sqLength1 * s.v1.w + sqLength2 * s.v2.w + sqLength3 * s.v3.w + sqLengthSum * s.v0.w));
    const float scale = 1.0f / (common + alpha);

    V3 lamTop, lamBot;
    if (!isTGS)
    {
        const float invE = 1.0f / youngs;
        const float nu = poissons;
        lamTop.x = -(matElem(G0, G1, G2, 0u, 0u) + invDtSq * multiplyStrainByCompliance(invE, nu, 0u, tlTop, tlBot) + damping * matElem(E0, E1, E2, 0u, 0u)) * scale;
        lamTop.y = -(matElem(G0, G1, G2, 1u, 1u) + invDtSq * multiplyStrainByCompliance(invE, nu, 1u, tlTop, tlBot) + damping * matElem(E0, E1, E2, 1u, 1u)) * scale;
        lamTop.z = -(matElem(G0, G1, G2, 2u, 2u) + invDtSq * multiplyStrainByCompliance(invE, nu, 2u, tlTop, tlBot) + damping * matElem(E0, E1, E2, 2u, 2u)) * scale;
        lamBot.x = -(matElem(G0, G1, G2, 1u, 2u) + invDtSq * multiplyStrainByCompliance(invE, nu, 3u, tlTop, tlBot) + damping * matElem(E0, E1, E2, 1u, 2u)) * scale;
        lamBot.y = -(matElem(G0, G1, G2, 0u, 2u) + invDtSq * multiplyStrainByCompliance(invE, nu, 4u, tlTop, tlBot) + damping * matElem(E0, E1, E2, 0u, 2u)) * scale;
        lamBot.z = -(matElem(G0, G1, G2, 0u, 1u) + invDtSq * multiplyStrainByCompliance(invE, nu, 5u, tlTop, tlBot) + damping * matElem(E0, E1, E2, 0u, 1u)) * scale;
        tlTop = add(tlTop, lamTop);
        tlBot = add(tlBot, lamBot);
    }
    else
    {
        lamTop.x = -(matElem(G0, G1, G2, 0u, 0u) + damping * matElem(E0, E1, E2, 0u, 0u)) * scale;
        lamTop.y = -(matElem(G0, G1, G2, 1u, 1u) + damping * matElem(E0, E1, E2, 1u, 1u)) * scale;
        lamTop.z = -(matElem(G0, G1, G2, 2u, 2u) + damping * matElem(E0, E1, E2, 2u, 2u)) * scale;
        lamBot.x = -(matElem(G0, G1, G2, 1u, 2u) + damping * matElem(E0, E1, E2, 1u, 2u)) * scale;
        lamBot.y = -(matElem(G0, G1, G2, 0u, 2u) + damping * matElem(E0, E1, E2, 0u, 2u)) * scale;
        lamBot.z = -(matElem(G0, G1, G2, 0u, 1u) + damping * matElem(E0, E1, E2, 0u, 1u)) * scale;
    }

    const float hs = (0.5f * volumeSqrt);
    const V3 a0 = V3{ volumeSqrt, 0.f, 0.f }, a1 = V3{ 0.f, volumeSqrt, 0.f }, a2 = V3{ 0.f, 0.f, volumeSqrt };
    const V3 b0 = V3{ hs, 0.f, 0.f }, b1 = V3{ 0.f, hs, 0.f }, b2 = V3{ 0.f, 0.f, hs };

    V3 D0 = muls(muls(a0, Q0.x), lamTop.x);
    V3 D1 = muls(muls(a0, Q0.y), lamTop.x);
    V3 D2 = muls(muls(a0, Q0.z), lamTop.x);

    D0 = add(D0, muls(muls(a1, Q1.x), lamTop.y));
    D1 = add(D1, muls(muls(a1, Q1.y), lamTop.y));
    D2 = add(D2, muls(muls(a1, Q1.z), lamTop.y));

    D0 = add(D0, muls(muls(a2, Q2.x), lamTop.z));
    D1 = add(D1, muls(muls(a2, Q2.y), lamTop.z));
    D2 = add(D2, muls(muls(a2, Q2.z), lamTop.z));

    D0 = add(D0, muls(add(muls(b1, Q2.x), muls(b2, Q1.x)), lamBot.x));
    D1 = add(D1, muls(add(muls(b1, Q2.y), muls(b2, Q1.y)), lamBot.x));
    D2 = add(D2, muls(add(muls(b1, Q2.z), muls(b2, Q1.z)), lamBot.x));

    D0 = add(D0, muls(add(muls(b0, Q2.x), muls(b2, Q0.x)), lamBot.y));
    D1 = add(D1, muls(add(muls(b0, Q2.y), muls(b2, Q0.y)), lamBot.y));
    D2 = add(D2, muls(add(muls(b0, Q2.z), muls(b2, Q0.z)), lamBot.y));

    D0 = add(D0, muls(add(muls(b0, Q1.x), muls(b1, Q0.x)), lamBot.z));
    D1 = add(D1, muls(add(muls(b0, Q1.y), muls(b1, Q0.y)), lamBot.z));
    D2 = add(D2, muls(add(muls(b0, Q1.z), muls(b1, Q0.z)), lamBot.z));

    o.d0 = F4{ D0.x, D0.y, D0.z, 0.f };
    o.d1 = F4{ D1.x, D1.y, D1.z, 0.f };
    o.d2 = F4{ D2.x, D2.y, D2.z, 0.f };
    o.tlTopOut = F4{ tlTop.x, tlTop.y, tlTop.z, 0.f };
    o.tlBotOut = F4{ tlBot.x, tlBot.y, tlBot.z, 0.f };
}

PXG_TEST(npSbSolveCorot)
{
    std::mt19937 rng(253001);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<CrIn> in(N);
    std::vector<CrOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        CrIn s{};
        s.pc0 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
        s.pc1 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
        s.pc2 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
        // Qinv with safely positive determinant (volumeSqrt must be real)
        for (;;)
        {
            s.qc0 = F4{ rf(-1.5f, 1.5f), rf(-1.5f, 1.5f), rf(-1.5f, 1.5f), 0.f };
            s.qc1 = F4{ rf(-1.5f, 1.5f), rf(-1.5f, 1.5f), rf(-1.5f, 1.5f), 0.f };
            s.qc2 = F4{ rf(-1.5f, 1.5f), rf(-1.5f, 1.5f), rf(-1.5f, 1.5f), 0.f };
            const V3 q0 = V3{ s.qc0.x, s.qc0.y, s.qc0.z }, q1 = V3{ s.qc1.x, s.qc1.y, s.qc1.z }, q2 = V3{ s.qc2.x, s.qc2.y, s.qc2.z };
            if (dot3(q0, cross3(q1, q2)) > 0.05f) break;
        }
        s.rt0 = F4{ rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-1.f, 1.f), 0.f };
        s.rt1 = F4{ rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-1.f, 1.f), 0.f };
        s.rt2 = F4{ rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-1.f, 1.f), 0.f };
        s.v0 = F4{ rf(-3.f, 3.f), rf(-3.f, 3.f), rf(-3.f, 3.f), rf(0.f, 3.f) };
        s.v1 = F4{ rf(-3.f, 3.f), rf(-3.f, 3.f), rf(-3.f, 3.f), rf(0.f, 3.f) };
        s.v2 = F4{ rf(-3.f, 3.f), rf(-3.f, 3.f), rf(-3.f, 3.f), rf(0.f, 3.f) };
        s.v3 = F4{ rf(-3.f, 3.f), rf(-3.f, 3.f), rf(-3.f, 3.f), rf(0.f, 3.f) };
        const uint32_t dampScale = rng() % 65536u;
        // mat.w carried the 5.6.1 dampingScale; 5.12 has none and the shader ignores the slot.
        s.mat = F4{ rf(1e4f, 1e8f), rf(0.f, 0.49f), rf(0.f, 2.f), u2f(dampScale) };  // youngs, poissons, damping, (unused)
        const float invDt = rf(10.f, 200.f);
        s.prm = F4{ invDt * invDt, invDt, u2f(i % 2u), 0.f };                        // invDtSq, invDt, isTGS
        s.tlTop = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
        s.tlBot = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
        // zero-accumulator stripe (fresh PGS pass)
        if ((i % 9u) == 4u) { s.tlTop = F4{ 0.f, 0.f, 0.f, 0.f }; s.tlBot = F4{ 0.f, 0.f, 0.f, 0.f }; }
        // zero-damping stripe (E path multiplied by 0)
        if ((i % 11u) == 6u) s.mat.z = 0.f;
        // alpha-dominant stripe: invDt = 200 exact, small integer youngs, tiny
        // masses and zero damping keep common << alpha, so the reciprocal-
        // multiply alpha = invDtSq*(1/youngs) stays distinguishable from a
        // single-rounded invDtSq/youngs (elsewhere the 1-ulp alpha difference
        // is absorbed by the common+alpha rounding)
        if ((i % 13u) == 8u)
        {
            s.mat.x = 10007.f + float(i % 97u);
            s.mat.z = 0.f;
            s.v0.w = 0.001f; s.v1.w = 0.002f; s.v2.w = 0.0015f; s.v3.w = 0.0005f;
            s.prm.x = 40000.f; s.prm.y = 200.f;
        }
        in[i] = s;
        refCr(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<CrOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npSbSolveCorot.spv", "sbSolveCorot",
             { (uint32_t)sizeof(CrIn), (uint32_t)sizeof(CrOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<CrOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        bool ok = true;
        const F4* rr = &ref[i].d0; const F4* gg = &g[i].d0;
        for (int f = 0; f < 5 && ok; ++f)
            ok = eqFZ(rr[f].x, gg[f].x) && eqFZ(rr[f].y, gg[f].y) && eqFZ(rr[f].z, gg[f].z);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u d0R (%g %g %g) d0G (%g %g %g) tlR (%g %g %g) tlG (%g %g %g)\n", i,
                            ref[i].d0.x, ref[i].d0.y, ref[i].d0.z, g[i].d0.x, g[i].d0.y, g[i].d0.z,
                            ref[i].tlTopOut.x, ref[i].tlTopOut.y, ref[i].tlTopOut.z,
                            g[i].tlTopOut.x, g[i].tlTopOut.y, g[i].tlTopOut.z);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "sbSolveCorot outputs bitwise-identical (mod +-0)");
}

}  // namespace
