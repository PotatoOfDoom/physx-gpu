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

// Verification of npClothSharedEnergy.slang (clothSharedEnergy): the FEM-cloth
// triangle-pair shell-energy driver clothSharedEnergySolvePerTrianglePair
// (FEMClothUtil.cuh L661-703) composing two membrane solves (tranche 255 core,
// on vertex triples (x2,x3,x0)/(x2,x3,x1) with QInv derived from the rest
// edges via cutil's RECIPROCAL-MULTIPLY float4/scalar division) and the
// bending solve (tranche 256 core) on the post-membrane positions, against an
// exact CPU transcription.  The bending atan2f seam is modeled as in tranche
// 256 (host-computed input + bitwise-pinned trig outputs).  Outputs bitwise
// mod +-0 (eqFZ).

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
struct GsIn  { F4 x0, x1, x2, x3, vrc, rest, prm, mat0c, mat1c, bendc, lamM, lamB; };
struct GsOut { F4 x0, x1, x2, x3, lamMOut, aux; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed

const float FEMCLOTH_THRESHOLD = 1.0e-14f;
const float FEMCLOTH_HALF_PI = 1.57079632679489661923f;
const float FEMCLOTH_2PI = 6.28318530717958647692f;

struct V2 { float x, y; };
struct V3 { float x, y, z; };
static V2 add2(V2 a, V2 b) { return V2{ a.x + b.x, a.y + b.y }; }
static V2 sub2(V2 a, V2 b) { return V2{ a.x - b.x, a.y - b.y }; }
static V2 muls2(V2 a, float s) { return V2{ a.x * s, a.y * s }; }
static V2 neg2(V2 a) { return V2{ -a.x, -a.y }; }
static float dot2(V2 a, V2 b) { return a.x * b.x + a.y * b.y; }
static float magSq2(V2 v) { return v.x * v.x + v.y * v.y; }
static V3 add3(V3 a, V3 b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 sub3(V3 a, V3 b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 muls3(V3 a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b)
{
    return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}
static float magSq3(V3 v) { return v.x * v.x + v.y * v.y + v.z * v.z; }
static float pabs(float a) { return a < 0.f ? -a : a; }
static float selMin(float a, float b) { return a < b ? a : b; }
static float selMax(float a, float b) { return a > b ? a : b; }

static V3 getNormalized3(V3 v)
{
    const float m = magSq3(v);
    return m > 0.f ? muls3(v, 1.0f / std::sqrt(m)) : V3{ 0.f, 0.f, 0.f };
}

static void extractRotation2D(V2& r0, V2& r1, V2 f0, V2 f1)
{
    const float s2x = dot2(f0, f0), s2y = dot2(f1, f1), s2z = dot2(f0, f1);
    const float det = s2x * s2y - s2z * s2z;
    if (det < FEMCLOTH_THRESHOLD) { r0 = V2{ 1.f, 0.f }; r1 = V2{ 0.f, 1.f }; return; }

    const float s0 = std::sqrt(det);
    const float t = std::sqrt(s2x + s2y + 2.0f * s0);
    if (t < FEMCLOTH_THRESHOLD) { r0 = V2{ 1.f, 0.f }; r1 = V2{ 0.f, 1.f }; return; }
    const float tInv = 1.0f / t;

    const float Sx = (s2x + s0) * tInv, Sy = (s2y + s0) * tInv, Sz = s2z * tInv;

    const float sDet = Sx * Sy - Sz * Sz;
    if (sDet < FEMCLOTH_THRESHOLD) { r0 = V2{ 1.f, 0.f }; r1 = V2{ 0.f, 1.f }; return; }
    const float sDetInv = 1.0f / sDet;

    const float ix = Sy * sDetInv, iy = Sx * sDetInv, iz = (-Sz) * sDetInv;

    r0 = add2(muls2(f0, ix), muls2(f1, iz));
    r1 = add2(muls2(f0, iz), muls2(f1, iy));
}

static float queryDeltaLambda3(float C, V2 g0, V2 g1, V2 g2, float alphaTilde, float lambda,
                               float m0, float m1, float m2)
{
    const float denom = (m0 * magSq2(g0) + m1 * magSq2(g1) + m2 * magSq2(g2)) + alphaTilde;
    if (denom < FEMCLOTH_THRESHOLD)
        return 0.0f;
    return (-C - alphaTilde * lambda) / denom;
}

static void arapConstraint2D(float& lambda, V2& dx0, V2& dx1, V2& dx2, float alphaTilde,
                             const F4& QInv, V2 x01, V2 x02, float m0, float m1, float m2)
{
    const V2 f0 = add2(muls2(x01, QInv.x), muls2(x02, QInv.y));
    const V2 f1 = add2(muls2(x01, QInv.z), muls2(x02, QInv.w));

    V2 r0, r1;
    extractRotation2D(r0, r1, f0, f1);

    const V2 FMinusR0 = sub2(f0, r0);
    const V2 FMinusR1 = sub2(f1, r1);

    const float C = std::sqrt(dot2(FMinusR0, FMinusR0) + dot2(FMinusR1, FMinusR1));

    if (C > FEMCLOTH_THRESHOLD)
    {
        const float CInv = 1.0f / C;

        const V2 pC_pF0 = muls2(FMinusR0, CInv);
        const V2 pC_pF1 = muls2(FMinusR1, CInv);

        const V2 grad1 = add2(muls2(pC_pF0, QInv.x), muls2(pC_pF1, QInv.z));
        const V2 grad2 = add2(muls2(pC_pF0, QInv.y), muls2(pC_pF1, QInv.w));
        const V2 grad0 = sub2(neg2(grad1), grad2);

        const float deltaLambda = queryDeltaLambda3(C, grad0, grad1, grad2, alphaTilde, lambda, m0, m1, m2);
        lambda = lambda + deltaLambda;

        dx0 = add2(dx0, muls2(grad0, m0 * deltaLambda));
        dx1 = add2(dx1, muls2(grad1, m1 * deltaLambda));
        dx2 = add2(dx2, muls2(grad2, m2 * deltaLambda));
    }
}

static void areaConstraint2D(float& lambda, V2& dx0, V2& dx1, V2& dx2, float alphaTilde,
                             V2 x01, V2 x02, float m0, float m1, float m2, float area)
{
    const float x01CrossX02 = x01.x * x02.y - x01.y * x02.x;
    const float undeformedAreaInv = 1.0f / area;

    const float C = 0.5f * x01CrossX02 * undeformedAreaInv - 1.0f;

    const V2 grad1 = V2{ 0.5f * undeformedAreaInv * x02.y, -(0.5f * undeformedAreaInv * x02.x) };
    const V2 grad2 = V2{ -(0.5f * undeformedAreaInv * x01.y), 0.5f * undeformedAreaInv * x01.x };
    const V2 grad0 = sub2(neg2(grad1), grad2);

    const float deltaLambda = queryDeltaLambda3(C, grad0, grad1, grad2, alphaTilde, lambda, m0, m1, m2);
    lambda = lambda + deltaLambda;

    dx0 = add2(dx0, muls2(grad0, m0 * deltaLambda));
    dx1 = add2(dx1, muls2(grad1, m1 * deltaLambda));
    dx2 = add2(dx2, muls2(grad2, m2 * deltaLambda));
}

// membraneEnergySolvePerTriangle (FEMClothUtil.cuh L436-527).
static void refMembrane(F4& xx0, F4& xx1, F4& xx2, float dt, float youngs, float poissons,
                        float thickness, const F4& QInv, float vs0, float vs1, float vs2,
                        float& slotX, float& slotY, bool isTGS)
{
    if (youngs < FEMCLOTH_THRESHOLD)
        return;

    V3 x0 = V3{ xx0.x, xx0.y, xx0.z };
    V3 x1 = V3{ xx1.x, xx1.y, xx1.z };
    V3 x2 = V3{ xx2.x, xx2.y, xx2.z };

    const V3 x01 = sub3(x1, x0);
    const V3 x02 = sub3(x2, x0);

    const V3 axis0 = getNormalized3(x01);
    const V3 normal = cross3(x01, x02);
    const V3 axis1 = getNormalized3(cross3(normal, axis0));

    const float dt2 = dt * dt;

    const float det = QInv.x * QInv.w - QInv.y * QInv.z;
    const float area = 1.0f / (2.0f * det);
    const float volume = area * thickness;

    V2 dx0 = V2{ 0.f, 0.f }, dx1 = V2{ 0.f, 0.f }, dx2 = V2{ 0.f, 0.f };
    float lambda0 = 0.f, lambda1 = 0.f;

    if (!isTGS)
    {
        lambda0 = slotX;
        lambda1 = slotY;
    }

    const float lamL = youngs * poissons / ((1.0f + poissons) * (1.0f - 2.0f * poissons));
    const float mu = youngs / (2.0f * (1.0f + poissons));

    V2 xp01 = V2{ dot3(axis0, x01), dot3(axis1, x01) };
    V2 xp02 = V2{ dot3(axis0, x02), dot3(axis1, x02) };

    const float alphaTilde0 = 1.0f / (2.0f * mu * volume * dt2);

    arapConstraint2D(lambda0, dx0, dx1, dx2, alphaTilde0, QInv, xp01, xp02,
                     vs0 * xx0.w, vs1 * xx1.w, vs2 * xx2.w);

    if (poissons > FEMCLOTH_THRESHOLD)
    {
        float alphaTilde1 = 0.0f;

        if (poissons < 0.5f - FEMCLOTH_THRESHOLD)
        {
            alphaTilde1 = 1.0f / (lamL * volume * dt2);
        }

        xp01 = add2(xp01, sub2(dx1, dx0));
        xp02 = add2(xp02, sub2(dx2, dx0));

        areaConstraint2D(lambda1, dx0, dx1, dx2, alphaTilde1, xp01, xp02,
                         vs0 * xx0.w, vs1 * xx1.w, vs2 * xx2.w, area);
    }

    x0 = add3(x0, add3(muls3(axis0, dx0.x), muls3(axis1, dx0.y)));
    x1 = add3(x1, add3(muls3(axis0, dx1.x), muls3(axis1, dx1.y)));
    x2 = add3(x2, add3(muls3(axis0, dx2.x), muls3(axis1, dx2.y)));

    if (!isTGS)
    {
        slotX = lambda0;
        slotY = lambda1;
    }

    xx0.x = x0.x; xx0.y = x0.y; xx0.z = x0.z;
    xx1.x = x1.x; xx1.y = x1.y; xx1.z = x1.z;
    xx2.x = x2.x; xx2.y = x2.y; xx2.z = x2.z;
}

// Runs the two membrane solves of the shared driver in place; returns flags.
static uint32_t refMembranes(const GsIn& s, F4& x0, F4& x1, F4& x2, F4& x3,
                             float& m0x, float& m0y, float& m1x, float& m1y)
{
    const float restSharedEdgeLength = s.prm.x;
    const float dt = s.prm.y;
    const bool isTGS = f2u(s.prm.z) == 1u;

    const float det0 = restSharedEdgeLength * s.rest.y;
    const float det1 = restSharedEdgeLength * s.rest.w;

    uint32_t flags = 0u;
    if (pabs(det0) > FEMCLOTH_THRESHOLD)
    {
        flags |= 1u;
        const float inv0 = 1.0f / det0;
        const F4 QInv0 = F4{ s.rest.y * inv0, 0.0f * inv0, -s.rest.x * inv0, restSharedEdgeLength * inv0 };
        refMembrane(x2, x3, x0, dt, s.mat0c.x, s.mat0c.y, s.mat0c.z, QInv0,
                    s.vrc.z, s.vrc.w, s.vrc.x, m0x, m0y, isTGS);
    }
    if (pabs(det1) > FEMCLOTH_THRESHOLD)
    {
        flags |= 2u;
        const float inv1 = 1.0f / det1;
        const F4 QInv1 = F4{ s.rest.w * inv1, 0.0f * inv1, -s.rest.z * inv1, restSharedEdgeLength * inv1 };
        refMembrane(x2, x3, x1, dt, s.mat1c.x, s.mat1c.y, s.mat1c.z, QInv1,
                    s.vrc.z, s.vrc.w, s.vrc.y, m1x, m1y, isTGS);
    }
    return flags;
}

// Post-membrane bending trig (mirrors the shader).  Returns bendPath.
static uint32_t bendTrig(const GsIn& s, const F4& x0, const F4& x1, const F4& x2, const F4& x3,
                         float& sinAngle, float& cosAngle)
{
    sinAngle = 0.f; cosAngle = 0.f;
    const V3 p0 = V3{ x0.x, x0.y, x0.z }, p1 = V3{ x1.x, x1.y, x1.z };
    const V3 p2 = V3{ x2.x, x2.y, x2.z }, p3 = V3{ x3.x, x3.y, x3.z };
    const V3 x23 = sub3(p3, p2);
    const float x23Len = std::sqrt(magSq3(x23));
    if (x23Len < FEMCLOTH_THRESHOLD) return 1u;
    if (s.bendc.y <= 0.f) return 2u;

    const V3 x02 = sub3(p2, p0), x03 = sub3(p3, p0), x13 = sub3(p3, p1), x12 = sub3(p2, p1);
    const float x23LenInv = 1.0f / x23Len;
    const V3 x23n = muls3(x23, x23LenInv);
    const V3 scaledN0 = cross3(x02, x03);
    const V3 scaledN1 = cross3(x13, x12);
    const float n0LenInv = 1.0f / std::sqrt(magSq3(scaledN0));
    const float n1LenInv = 1.0f / std::sqrt(magSq3(scaledN1));
    const V3 n0 = muls3(scaledN0, n0LenInv);
    const V3 n1 = muls3(scaledN1, n1LenInv);
    cosAngle = dot3(n0, n1);
    sinAngle = dot3(cross3(n0, n1), x23n);
    return 0u;
}

static void refGs(const GsIn& s, GsOut& o)
{
    F4 x0 = s.x0, x1 = s.x1, x2 = s.x2, x3 = s.x3;
    float m0x = s.lamM.x, m0y = s.lamM.y, m1x = s.lamM.z, m1y = s.lamM.w;

    uint32_t flags = refMembranes(s, x0, x1, x2, x3, m0x, m0y, m1x, m1y);

    float sinAngle, cosAngle;
    const uint32_t bendPath = bendTrig(s, x0, x1, x2, x3, sinAngle, cosAngle);
    float bendLambdaOut = s.lamB.x;

    if (bendPath == 0u)
    {
        const bool isTGS = f2u(s.prm.z) == 1u;
        const float dt = s.prm.y;
        const float restBendingAngle = s.bendc.x, kInv = s.bendc.y;
        const float angle = s.bendc.z;

        const V3 p0 = V3{ x0.x, x0.y, x0.z }, p1 = V3{ x1.x, x1.y, x1.z };
        const V3 p2 = V3{ x2.x, x2.y, x2.z }, p3 = V3{ x3.x, x3.y, x3.z };
        const V3 x02 = sub3(p2, p0), x03 = sub3(p3, p0), x13 = sub3(p3, p1), x12 = sub3(p2, p1), x23 = sub3(p3, p2);
        const float x23Len = std::sqrt(magSq3(x23));
        const float x23LenInv = 1.0f / x23Len;
        const V3 x23n = muls3(x23, x23LenInv);
        const V3 scaledN0 = cross3(x02, x03);
        const V3 scaledN1 = cross3(x13, x12);
        const float n0LenInv = 1.0f / std::sqrt(magSq3(scaledN0));
        const float n1LenInv = 1.0f / std::sqrt(magSq3(scaledN1));
        const V3 n0 = muls3(scaledN0, n0LenInv);
        const V3 n1 = muls3(scaledN1, n1LenInv);

        const float dtInv = 1.0f / dt;
        const float alphaTilde = kInv * dtInv * dtInv;
        float C = angle - restBendingAngle;

        if (pabs(C + FEMCLOTH_2PI) < pabs(C))
            C += FEMCLOTH_2PI;
        else if (pabs(C - FEMCLOTH_2PI) < pabs(C))
            C -= FEMCLOTH_2PI;

        C = selMin(FEMCLOTH_HALF_PI, selMax(-FEMCLOTH_HALF_PI, C));

        const V3 temp0 = muls3(n0, n0LenInv);
        const V3 temp1 = muls3(n1, n1LenInv);
        const V3 dCdx0 = muls3(temp0, -x23Len);
        const V3 dCdx1 = muls3(temp1, -x23Len);
        const V3 dCdx2 = V3{ dot3(x03, x23n) * temp0.x + dot3(x13, x23n) * temp1.x,
                             dot3(x03, x23n) * temp0.y + dot3(x13, x23n) * temp1.y,
                             dot3(x03, x23n) * temp0.z + dot3(x13, x23n) * temp1.z };
        const V3 s3 = V3{ dot3(x02, x23n) * temp0.x + dot3(x12, x23n) * temp1.x,
                          dot3(x02, x23n) * temp0.y + dot3(x12, x23n) * temp1.y,
                          dot3(x02, x23n) * temp0.z + dot3(x12, x23n) * temp1.z };
        const V3 dCdx3 = V3{ -s3.x, -s3.y, -s3.z };

        float lambda = 0.f;
        if (!isTGS)
            lambda = s.lamB.x;

        const float m0 = s.vrc.x * x0.w;
        const float m1 = s.vrc.y * x1.w;
        const float m2 = s.vrc.z * x2.w;
        const float m3 = s.vrc.w * x3.w;
        const float denom = (m0 * magSq3(dCdx0) + m1 * magSq3(dCdx1) + m2 * magSq3(dCdx2) + m3 * magSq3(dCdx3)) + alphaTilde;
        float deltaLambda;
        if (denom < FEMCLOTH_THRESHOLD)
            deltaLambda = 0.f;
        else
            deltaLambda = (-C - alphaTilde * lambda) / denom;

        if (!isTGS)
            bendLambdaOut = lambda + deltaLambda;

        const float scale0 = s.vrc.x * x0.w * deltaLambda;
        const float scale1 = s.vrc.y * x1.w * deltaLambda;
        const float scale2 = s.vrc.z * x2.w * deltaLambda;
        const float scale3 = s.vrc.w * x3.w * deltaLambda;

        x0 = F4{ x0.x + scale0 * dCdx0.x, x0.y + scale0 * dCdx0.y, x0.z + scale0 * dCdx0.z, x0.w };
        x1 = F4{ x1.x + scale1 * dCdx1.x, x1.y + scale1 * dCdx1.y, x1.z + scale1 * dCdx1.z, x1.w };
        x2 = F4{ x2.x + scale2 * dCdx2.x, x2.y + scale2 * dCdx2.y, x2.z + scale2 * dCdx2.z, x2.w };
        x3 = F4{ x3.x + scale3 * dCdx3.x, x3.y + scale3 * dCdx3.y, x3.z + scale3 * dCdx3.z, x3.w };
    }

    o.x0 = x0; o.x1 = x1; o.x2 = x2; o.x3 = x3;
    o.lamMOut = F4{ m0x, m0y, m1x, m1y };
    o.aux = F4{ bendLambdaOut, sinAngle, cosAngle, u2f(flags | (bendPath << 2u)) };
}

PXG_TEST(npClothSharedEnergy)
{
    std::mt19937 rng(257001);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<GsIn> in(N);
    std::vector<GsOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        GsIn s{};
        s.x0 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), rf(0.f, 3.f) };
        s.x1 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), rf(0.f, 3.f) };
        s.x2 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), rf(0.f, 3.f) };
        s.x3 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), rf(0.f, 3.f) };
        s.vrc = F4{ rf(0.f, 1.f), rf(0.f, 1.f), rf(0.f, 1.f), rf(0.f, 1.f) };
        const float restLen = rf(0.1f, 2.f);
        // rest edges with well-conditioned dets (det_i = restLen * edge_i.y)
        for (;;)
        {
            s.rest = F4{ rf(-1.5f, 1.5f), rf(-1.5f, 1.5f), rf(-1.5f, 1.5f), rf(-1.5f, 1.5f) };
            const float d0 = restLen * s.rest.y, d1 = restLen * s.rest.w;
            if ((d0 > 0.05f || d0 < -0.05f) && (d1 > 0.05f || d1 < -0.05f)) break;
        }
        const uint32_t isTGS = i % 2u;
        s.prm = F4{ restLen, rf(0.005f, 0.1f), u2f(isTGS), 0.f };
        s.mat0c = F4{ rf(1e4f, 1e8f), rf(0.f, 0.49f), rf(1e-3f, 0.1f), 0.f };
        s.mat1c = F4{ rf(1e4f, 1e8f), rf(0.f, 0.49f), rf(1e-3f, 0.1f), 0.f };
        s.bendc = F4{ rf(-3.14f, 3.14f), rf(1e-8f, 1e-2f), 0.f, 0.f };  // angleIn filled below
        s.lamM = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f) };
        s.lamB = F4{ rf(-2.f, 2.f), 0.f, 0.f, 0.f };
        // membrane skip stripes: rest edge y = 0 -> det = +-0, guard fails
        if ((i % 13u) == 4u) s.rest.y = 0.f;
        if ((i % 13u) == 9u) s.rest.w = 0.f;
        // membrane0 internal youngs early-out (guard passes, solve no-ops)
        if ((i % 17u) == 3u) s.mat0c.x = 0.f;
        // membrane poissons corners
        if ((i % 7u) == 2u) s.mat1c.y = 0.f;
        if ((i % 19u) == 5u) s.mat0c.y = 0.5f;
        // bending kInv early-out (zero / negative)
        if ((i % 11u) == 3u) s.bendc.y = ((i / 11u) % 2u) ? -0.5f : 0.f;
        // bending x23 early-out: both membranes skipped AND x3 == x2 exactly
        if ((i % 29u) == 11u)
        {
            s.rest.y = 0.f;
            s.rest.w = 0.f;
            s.x3 = F4{ s.x2.x, s.x2.y, s.x2.z, s.x3.w };
        }
        // host-side atan2f of the post-membrane trig values (the seam)
        {
            F4 x0 = s.x0, x1 = s.x1, x2 = s.x2, x3 = s.x3;
            float m0x = s.lamM.x, m0y = s.lamM.y, m1x = s.lamM.z, m1y = s.lamM.w;
            refMembranes(s, x0, x1, x2, x3, m0x, m0y, m1x, m1y);
            float sinAngle, cosAngle;
            if (bendTrig(s, x0, x1, x2, x3, sinAngle, cosAngle) == 0u)
                s.bendc.z = std::atan2(sinAngle, cosAngle);
        }
        in[i] = s;
        refGs(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<GsOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npClothSharedEnergy.spv", "clothSharedEnergy",
             { (uint32_t)sizeof(GsIn), (uint32_t)sizeof(GsOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<GsOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const F4* rr = &ref[i].x0; const F4* gg = &g[i].x0;
        bool ok = true;
        for (int f = 0; f < 5 && ok; ++f)
            ok = eqFZ(rr[f].x, gg[f].x) && eqFZ(rr[f].y, gg[f].y) && eqFZ(rr[f].z, gg[f].z) && eqFZ(rr[f].w, gg[f].w);
        ok = ok && eqFZ(ref[i].aux.x, g[i].aux.x) && eqFZ(ref[i].aux.y, g[i].aux.y)
               && eqFZ(ref[i].aux.z, g[i].aux.z) && f2u(ref[i].aux.w) == f2u(g[i].aux.w);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u x2R (%g %g %g) x2G (%g %g %g) lamR (%g %g %g %g) lamG (%g %g %g %g) fR %u fG %u\n", i,
                            ref[i].x2.x, ref[i].x2.y, ref[i].x2.z, g[i].x2.x, g[i].x2.y, g[i].x2.z,
                            ref[i].lamMOut.x, ref[i].lamMOut.y, ref[i].lamMOut.z, ref[i].lamMOut.w,
                            g[i].lamMOut.x, g[i].lamMOut.y, g[i].lamMOut.z, g[i].lamMOut.w,
                            f2u(ref[i].aux.w), f2u(g[i].aux.w));
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "clothSharedEnergy outputs bitwise-identical (mod +-0)");
}

}  // namespace
