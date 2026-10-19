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

// Verification of npSbSolveGM.slang (sbSolveGM): the per-tet material-model
// dispatcher tetrahedronsSolveGM (softBodyGM.cu L926-1025) composing the
// NeoHookean inner solve + volumeSolve (NEO_HOOKEAN) or ARAP_constraint +
// volume_constraint via compute_deltaLambdaXgradC (CO_ROTATIONAL), against an
// exact CPU transcription.  The float2 multiplier slot and the
// deltaLambdaXgradC accumulator are modeled as in/out.  Outputs bitwise mod
// +-0 (eqFZ).

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
struct GmIn  { F4 gu1, gu2, gu3, v0, v1, v2, v3, qc0, qc1, qc2, quat, mat, prm, mul, acc0, acc1, acc2; };
struct GmOut { F4 dc0, dc1, dc2, mulOut; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed

struct V3 { float x, y, z; };
static V3 add(V3 a, V3 b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 sub(V3 a, V3 b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 muls(V3 a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; }
static V3 neg(V3 a) { return V3{ -a.x, -a.y, -a.z }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b)
{
    return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}
static float magSq(V3 v) { return v.x * v.x + v.y * v.y + v.z * v.z; }
static float pabs(float a) { return a < 0.f ? -a : a; }

// PxMat33::transform: (c0*v.x + c1*v.y) + c2*v.z
static V3 mat33Transform(V3 c0, V3 c1, V3 c2, V3 v)
{
    return add(add(muls(c0, v.x), muls(c1, v.y)), muls(c2, v.z));
}

// PxMat33(PxQuat) rotation-matrix constructor.
static void quatToMat(F4 q, V3& c0, V3& c1, V3& c2)
{
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    const float x2 = x + x, y2 = y + y, z2 = z + z;
    const float xx = x2 * x, yy = y2 * y, zz = z2 * z;
    const float xy = x2 * y, xz = x2 * z, xw = x2 * w;
    const float yz = y2 * z, yw = y2 * w, zw = z2 * w;
    c0 = V3{ 1.0f - yy - zz, xy + zw, xz - yw };
    c1 = V3{ xy - zw, 1.0f - xx - zz, yz + xw };
    c2 = V3{ xz + yw, yz - xw, 1.0f - xx - yy };
}

// compute_deltaLambdaXgradC (softBodyGM.cu L558-604).
static void computeDLXG(V3& a0, V3& a1, V3& a2, float& lambda,
                        float C, V3 dFc0, V3 dFc1, V3 dFc2,
                        V3 q0, V3 q1, V3 q2,
                        const F4& v0, const F4& v1, const F4& v2, const F4& v3,
                        float alphaTilde, float dtInv, float damping)
{
    const V3 r0 = V3{ q0.x, q1.x, q2.x };
    const V3 r1 = V3{ q0.y, q1.y, q2.y };
    const V3 r2 = V3{ q0.z, q1.z, q2.z };
    const V3 p0 = V3{ dFc0.x, dFc1.x, dFc2.x };
    const V3 p1 = V3{ dFc0.y, dFc1.y, dFc2.y };
    const V3 p2 = V3{ dFc0.z, dFc1.z, dFc2.z };
    const V3 dCdx1 = V3{ dot3(r0, p0), dot3(r0, p1), dot3(r0, p2) };
    const V3 dCdx2 = V3{ dot3(r1, p0), dot3(r1, p1), dot3(r1, p2) };
    const V3 dCdx3 = V3{ dot3(r2, p0), dot3(r2, p1), dot3(r2, p2) };
    const V3 dCdx0 = sub(sub(neg(dCdx1), dCdx2), dCdx3);

    float deltaLambda;
    if (damping == 0.f)
    {
        const float denom = magSq(dCdx0) * v0.w + magSq(dCdx1) * v1.w + magSq(dCdx2) * v2.w + magSq(dCdx3) * v3.w + alphaTilde;
        if (pabs(denom) < 1.0e-15f)
            return;
        deltaLambda = (-C - alphaTilde * lambda) / denom;
    }
    else
    {
        const float denom = (1.0f + damping * dtInv) * (magSq(dCdx0) * v0.w + magSq(dCdx1) * v1.w + magSq(dCdx2) * v2.w + magSq(dCdx3) * v3.w) + alphaTilde;
        if (pabs(denom) < 1.0e-15f)
        {
            a0 = V3{ 0.f, 0.f, 0.f };
            a1 = V3{ 0.f, 0.f, 0.f };
            a2 = V3{ 0.f, 0.f, 0.f };
            return;
        }
        const float dampDot = dot3(dCdx0, V3{ v0.x, v0.y, v0.z }) + dot3(dCdx1, V3{ v1.x, v1.y, v1.z })
                            + dot3(dCdx2, V3{ v2.x, v2.y, v2.z }) + dot3(dCdx3, V3{ v3.x, v3.y, v3.z });
        deltaLambda = (-C - alphaTilde * lambda - damping * dampDot) / denom;
    }

    lambda = lambda + deltaLambda;
    a0 = add(a0, muls(dCdx1, deltaLambda));
    a1 = add(a1, muls(dCdx2, deltaLambda));
    a2 = add(a2, muls(dCdx3, deltaLambda));
}

static void refGm(const GmIn& s, GmOut& o)
{
    const V3 u1 = V3{ s.gu1.x, s.gu1.y, s.gu1.z }, u2 = V3{ s.gu2.x, s.gu2.y, s.gu2.z }, u3 = V3{ s.gu3.x, s.gu3.y, s.gu3.z };
    const V3 Q0 = V3{ s.qc0.x, s.qc0.y, s.qc0.z }, Q1 = V3{ s.qc1.x, s.qc1.y, s.qc1.z }, Q2 = V3{ s.qc2.x, s.qc2.y, s.qc2.z };
    const float youngs = s.mat.x, poissons = s.mat.y;
    const float invDtSq = s.prm.x, invDt = s.prm.y;
    const bool isTGS = f2u(s.prm.z) == 1u;
    const bool isNeoHookean = f2u(s.prm.w) == 1u;

    V3 dl0 = V3{ s.acc0.x, s.acc0.y, s.acc0.z };
    V3 dl1 = V3{ s.acc1.x, s.acc1.y, s.acc1.z };
    V3 dl2 = V3{ s.acc2.x, s.acc2.y, s.acc2.z };
    float mulX = s.mul.x, mulY = s.mul.y;

    if (youngs < 1.0e-18f)
    {
        o.dc0 = F4{ dl0.x, dl0.y, dl0.z, 0.f };
        o.dc1 = F4{ dl1.x, dl1.y, dl1.z, 0.f };
        o.dc2 = F4{ dl2.x, dl2.y, dl2.z, 0.f };
        o.mulOut = F4{ mulX, mulY, 0.f, 0.f };
        return;
    }

    if (isNeoHookean)
    {
        const float invE = 1.0f / youngs;
        V3 Rc0, Rc1, Rc2;
        quatToMat(s.quat, Rc0, Rc1, Rc2);
        const V3 T0 = V3{ Rc0.x, Rc1.x, Rc2.x };
        const V3 T1 = V3{ Rc0.y, Rc1.y, Rc2.y };
        const V3 T2 = V3{ Rc0.z, Rc1.z, Rc2.z };
        V3 P0 = mat33Transform(T0, T1, T2, u1);
        V3 P1 = mat33Transform(T0, T1, T2, u2);
        V3 P2 = mat33Transform(T0, T1, T2, u3);

        const V3 F0 = mat33Transform(P0, P1, P2, Q0);
        const V3 F1 = mat33Transform(P0, P1, P2, Q1);
        const V3 F2 = mat33Transform(P0, P1, P2, Q2);

        // g_i = F.c0*2*q0 + F.c1*2*q1 + F.c2*2*q2, each product ((F.cj*2)*q)
        const V3 g1 = add(add(muls(muls(F0, 2.0f), Q0.x), muls(muls(F1, 2.0f), Q1.x)), muls(muls(F2, 2.0f), Q2.x));
        const V3 g2 = add(add(muls(muls(F0, 2.0f), Q0.y), muls(muls(F1, 2.0f), Q1.y)), muls(muls(F2, 2.0f), Q2.y));
        const V3 g3 = add(add(muls(muls(F0, 2.0f), Q0.z), muls(muls(F1, 2.0f), Q1.z)), muls(muls(F2, 2.0f), Q2.z));

        const float C = magSq(F0) + magSq(F1) + magSq(F2) - 3.0f;
        const V3 g0 = sub(sub(neg(g1), g2), g3);
        const float w = magSq(g0) * s.v0.w + magSq(g1) * s.v1.w + magSq(g2) * s.v2.w + magSq(g3) * s.v3.w;
        const float alpha = invDtSq * invE;
        float dlambda;
        if (!isTGS)
        {
            dlambda = -(C + alpha * mulX) / (w + alpha);
            mulX += dlambda;
        }
        else
        {
            dlambda = -C / (w + alpha);
        }
        V3 d0 = muls(g1, dlambda), d1 = muls(g2, dlambda), d2 = muls(g3, dlambda);

        const V3 delta0 = muls(neg(add(add(d0, d1), d2)), s.v0.w);
        P0 = add(P0, sub(muls(d0, s.v1.w), delta0));
        P1 = add(P1, sub(muls(d1, s.v2.w), delta0));
        P2 = add(P2, sub(muls(d2, s.v3.w), delta0));

        if (poissons > 1.0e-18f)
        {
            const float nu = poissons;
            const float volumeLambda = (nu < 0.5f - 1.0e-18f) ? nu / ((1.0f - 2.0f * nu) * youngs) : 0.0f;

            const float det = dot3(Q0, cross3(Q1, Q2));
            const float volume = 1.0f / (6.0f * det);
            const float alphaTilde = (volumeLambda == 0.0f) ? 0.0f : 1.0f / (volumeLambda * volume * invDtSq);

            const V3 vF0 = mat33Transform(P0, P1, P2, Q0);
            const V3 vF1 = mat33Transform(P0, P1, P2, Q1);
            const V3 vF2 = mat33Transform(P0, P1, P2, Q2);

            const V3 vg0 = cross3(vF1, vF2);
            const V3 vg1 = cross3(vF2, vF0);
            const V3 vg2 = cross3(vF0, vF1);

            const V3 grad1 = add(add(muls(vg0, Q0.x), muls(vg1, Q1.x)), muls(vg2, Q2.x));
            const V3 grad2 = add(add(muls(vg0, Q0.y), muls(vg1, Q1.y)), muls(vg2, Q2.y));
            const V3 grad3 = add(add(muls(vg0, Q0.z), muls(vg1, Q1.z)), muls(vg2, Q2.z));
            const V3 grad0 = sub(sub(neg(grad1), grad2), grad3);

            const float detF = dot3(vg2, vF2);
            const float Cv = detF - 1.0f;
            const float wv = magSq(grad0) * s.v0.w + magSq(grad1) * s.v1.w + magSq(grad2) * s.v2.w + magSq(grad3) * s.v3.w;
            const float dlv = -Cv / (wv + alphaTilde);

            d0 = add(d0, muls(grad1, dlv));
            d1 = add(d1, muls(grad2, dlv));
            d2 = add(d2, muls(grad3, dlv));
        }

        dl0 = mat33Transform(Rc0, Rc1, Rc2, d0);
        dl1 = mat33Transform(Rc0, Rc1, Rc2, d1);
        dl2 = mat33Transform(Rc0, Rc1, Rc2, d2);
    }
    else // CO_ROTATIONAL
    {
        const float lamL = youngs * poissons / ((1.0f + poissons) * (1.0f - 2.0f * poissons));
        const float mu = youngs / (2.0f * (1.0f + poissons));

        const float det = dot3(Q0, cross3(Q1, Q2));
        const float volume = 1.0f / (6.0f * det);

        V3 F0 = mat33Transform(u1, u2, u3, Q0);
        V3 F1 = mat33Transform(u1, u2, u3, Q1);
        V3 F2 = mat33Transform(u1, u2, u3, Q2);

        const float alphaTilde0 = invDtSq / (2.0f * mu * volume);

        float damping = s.mat.z;

        float lambdasX = isTGS ? 0.f : mulX;
        float lambdasY = isTGS ? 0.f : mulY;

        // ARAP_constraint
        {
            V3 Rc0, Rc1, Rc2;
            quatToMat(s.quat, Rc0, Rc1, Rc2);
            const V3 t0 = sub(F0, Rc0), t1 = sub(F1, Rc1), t2 = sub(F2, Rc2);
            const float arapC = std::sqrt(dot3(t0, t0) + dot3(t1, t1) + dot3(t2, t2) + 1.0e-18f);
            const float inv = 1.0f / arapC;
            const V3 dF0 = muls(t0, inv), dF1 = muls(t1, inv), dF2 = muls(t2, inv);
            computeDLXG(dl0, dl1, dl2, lambdasX, arapC, dF0, dF1, dF2,
                        Q0, Q1, Q2, s.v0, s.v1, s.v2, s.v3, alphaTilde0, invDt, damping);
        }

        const V3 dx0 = muls(neg(add(add(dl0, dl1), dl2)), s.v0.w);
        const V3 dx1 = muls(dl0, s.v1.w);
        const V3 dx2 = muls(dl1, s.v2.w);
        const V3 dx3 = muls(dl2, s.v3.w);

        const V3 x01 = add(u1, sub(dx1, dx0));
        const V3 x02 = add(u2, sub(dx2, dx0));
        const V3 x03 = add(u3, sub(dx3, dx0));

        F0 = mat33Transform(x01, x02, x03, Q0);
        F1 = mat33Transform(x01, x02, x03, Q1);
        F2 = mat33Transform(x01, x02, x03, Q2);

        if (poissons > 1.0e-18f)
        {
            float alphaTilde1 = 0.0f;
            float ratio = 0.0f;

            if (poissons < 0.5f - 1.0e-18f)
            {
                alphaTilde1 = invDtSq / (lamL * volume);
                ratio = 2.0f * mu / lamL;
            }

            damping *= ratio;

            // volume_constraint
            const float detF = dot3(F0, cross3(F1, F2));
            const float Cv = detF - 1.0f;
            const V3 dF0 = cross3(F1, F2);
            const V3 dF1 = cross3(F2, F0);
            const V3 dF2 = cross3(F0, F1);
            computeDLXG(dl0, dl1, dl2, lambdasY, Cv, dF0, dF1, dF2,
                        Q0, Q1, Q2, s.v0, s.v1, s.v2, s.v3, alphaTilde1, invDt, damping);
        }

        if (!isTGS)
        {
            mulX = lambdasX;
            mulY = lambdasY;
        }
    }

    o.dc0 = F4{ dl0.x, dl0.y, dl0.z, 0.f };
    o.dc1 = F4{ dl1.x, dl1.y, dl1.z, 0.f };
    o.dc2 = F4{ dl2.x, dl2.y, dl2.z, 0.f };
    o.mulOut = F4{ mulX, mulY, 0.f, 0.f };
}

PXG_TEST(npSbSolveGM)
{
    std::mt19937 rng(254001);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<GmIn> in(N);
    std::vector<GmOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        GmIn s{};
        s.gu1 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
        s.gu2 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
        s.gu3 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
        s.v0 = F4{ rf(-3.f, 3.f), rf(-3.f, 3.f), rf(-3.f, 3.f), rf(0.f, 3.f) };
        s.v1 = F4{ rf(-3.f, 3.f), rf(-3.f, 3.f), rf(-3.f, 3.f), rf(0.f, 3.f) };
        s.v2 = F4{ rf(-3.f, 3.f), rf(-3.f, 3.f), rf(-3.f, 3.f), rf(0.f, 3.f) };
        s.v3 = F4{ rf(-3.f, 3.f), rf(-3.f, 3.f), rf(-3.f, 3.f), rf(0.f, 3.f) };
        // Qinv with safely positive determinant
        for (;;)
        {
            s.qc0 = F4{ rf(-1.5f, 1.5f), rf(-1.5f, 1.5f), rf(-1.5f, 1.5f), 0.f };
            s.qc1 = F4{ rf(-1.5f, 1.5f), rf(-1.5f, 1.5f), rf(-1.5f, 1.5f), 0.f };
            s.qc2 = F4{ rf(-1.5f, 1.5f), rf(-1.5f, 1.5f), rf(-1.5f, 1.5f), 0.f };
            const V3 q0 = V3{ s.qc0.x, s.qc0.y, s.qc0.z }, q1 = V3{ s.qc1.x, s.qc1.y, s.qc1.z }, q2 = V3{ s.qc2.x, s.qc2.y, s.qc2.z };
            if (dot3(q0, cross3(q1, q2)) > 0.05f) break;
        }
        // random unit-ish quaternion (input data only; both sides read the same bits)
        {
            float qx = rf(-1.f, 1.f), qy = rf(-1.f, 1.f), qz = rf(-1.f, 1.f), qw = rf(-1.f, 1.f);
            const float n = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
            if (n > 1e-3f) { qx /= n; qy /= n; qz /= n; qw /= n; }
            else { qx = 0.f; qy = 0.f; qz = 0.f; qw = 1.f; }
            s.quat = F4{ qx, qy, qz, qw };
        }
        const uint32_t dampScale = rng() % 65536u;
        // mat.w carried the 5.6.1 dampingScale; 5.12 has none and the shader ignores the slot.
        s.mat = F4{ rf(1e4f, 1e8f), rf(0.f, 0.49f), rf(0.f, 2.f), u2f(dampScale) };  // youngs, poissons, damping, (unused)
        const float invDt = rf(10.f, 200.f);
        const uint32_t isTGS = i % 2u;
        const uint32_t isNeo = (i / 2u) % 2u;
        s.prm = F4{ invDt * invDt, invDt, u2f(isTGS), u2f(isNeo) };
        s.mul = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f, 0.f };
        s.acc0 = F4{ rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-1.f, 1.f), 0.f };
        s.acc1 = F4{ rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-1.f, 1.f), 0.f };
        s.acc2 = F4{ rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-1.f, 1.f), 0.f };
        // poissons == 0 stripe: volume solve / volume_constraint skipped
        if ((i % 7u) == 2u) s.mat.y = 0.f;
        // zero-damping stripe: compute_deltaLambdaXgradC no-damping branch
        if ((i % 11u) == 6u) s.mat.z = 0.f;
        // stiffness-threshold early-out stripe: outputs pass through unchanged
        if ((i % 17u) == 3u) s.mat.x = 0.f;
        // threshold boundary stripe: youngs == 1e-18f exactly must NOT early-out
        if ((i % 17u) == 10u) s.mat.x = 1.0e-18f;
        // incompressible stripe: nu = 0.5 -> volumeLambda 0 / ratio 0 guards
        if ((i % 19u) == 5u) s.mat.y = 0.5f;
        // degenerate-denominator stripe (corot only): all masses 0 and a huge
        // youngs make |denom| < 1e-15, hitting the accumulator-passthrough
        // (damping 0) or accumulator-ZEROING (damping > 0) paths
        if ((i % 23u) == 7u)
        {
            s.prm.w = u2f(0u);
            s.mat.x = 1e38f;
            s.mat.y = 0.3f;
            s.mat.z = ((i / 23u) % 2u) ? 0.7f : 0.f;
            s.v0.w = 0.f; s.v1.w = 0.f; s.v2.w = 0.f; s.v3.w = 0.f;
            s.prm.x = 100.f; s.prm.y = 10.f;
        }
        in[i] = s;
        refGm(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<GmOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npSbSolveGM.spv", "sbSolveGM",
             { (uint32_t)sizeof(GmIn), (uint32_t)sizeof(GmOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<GmOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        bool ok = eqFZ(ref[i].dc0.x, g[i].dc0.x) && eqFZ(ref[i].dc0.y, g[i].dc0.y) && eqFZ(ref[i].dc0.z, g[i].dc0.z)
               && eqFZ(ref[i].dc1.x, g[i].dc1.x) && eqFZ(ref[i].dc1.y, g[i].dc1.y) && eqFZ(ref[i].dc1.z, g[i].dc1.z)
               && eqFZ(ref[i].dc2.x, g[i].dc2.x) && eqFZ(ref[i].dc2.y, g[i].dc2.y) && eqFZ(ref[i].dc2.z, g[i].dc2.z)
               && eqFZ(ref[i].mulOut.x, g[i].mulOut.x) && eqFZ(ref[i].mulOut.y, g[i].mulOut.y);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u dc0R (%g %g %g) dc0G (%g %g %g) mulR (%g %g) mulG (%g %g)\n", i,
                            ref[i].dc0.x, ref[i].dc0.y, ref[i].dc0.z, g[i].dc0.x, g[i].dc0.y, g[i].dc0.z,
                            ref[i].mulOut.x, ref[i].mulOut.y, g[i].mulOut.x, g[i].mulOut.y);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "sbSolveGM outputs bitwise-identical (mod +-0)");
}

}  // namespace
