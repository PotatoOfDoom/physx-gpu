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

// Verification of npClothMembrane.slang (clothMembrane): the FEM-cloth
// fixed-corotated membrane energy dispatcher membraneEnergySolvePerTriangle
// (FEMClothUtil.cuh L436-527) composing the 2x2 ARAP + area constraints,
// against an exact CPU transcription.  The per-triangle lambda slot is modeled
// as in/out float2.  Outputs bitwise mod +-0 (eqFZ).

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
struct CmIn  { F4 xx0, xx1, xx2, qinv, vscale, mat, mul; };
struct CmOut { F4 xx0, xx1, xx2, mulOut; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b) || (a == 0.f && b == 0.f); }

const float FEMCLOTH_THRESHOLD = 1.0e-14f;

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

// PxVec3::getNormalized: m > 0 ? v * PxRecipSqrt(m) : 0, PxRecipSqrt = 1/sqrtf.
static V3 getNormalized3(V3 v)
{
    const float m = magSq3(v);
    return m > 0.f ? muls3(v, 1.0f / std::sqrt(m)) : V3{ 0.f, 0.f, 0.f };
}

// extractRotation2D (FEMClothUtil.cuh L88-144).
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

// queryDeltaLambda (3-vertex, no damping, FEMClothUtil.cuh L211-223).
static float queryDeltaLambda(float C, V2 g0, V2 g1, V2 g2, float alphaTilde, float lambda,
                              float m0, float m1, float m2)
{
    const float denom = (m0 * magSq2(g0) + m1 * magSq2(g1) + m2 * magSq2(g2)) + alphaTilde;
    if (denom < FEMCLOTH_THRESHOLD)
        return 0.0f;
    return (-C - alphaTilde * lambda) / denom;
}

// ARAPConstraint_F2X2 (FEMClothUtil.cuh L335-368).
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

        const float deltaLambda = queryDeltaLambda(C, grad0, grad1, grad2, alphaTilde, lambda, m0, m1, m2);
        lambda = lambda + deltaLambda;

        dx0 = add2(dx0, muls2(grad0, m0 * deltaLambda));
        dx1 = add2(dx1, muls2(grad1, m1 * deltaLambda));
        dx2 = add2(dx2, muls2(grad2, m2 * deltaLambda));
    }
}

// areaConstraint_F2X2 (FEMClothUtil.cuh L376-419, active branch).
static void areaConstraint2D(float& lambda, V2& dx0, V2& dx1, V2& dx2, float alphaTilde,
                             V2 x01, V2 x02, float m0, float m1, float m2, float area)
{
    const float x01CrossX02 = x01.x * x02.y - x01.y * x02.x;
    const float undeformedAreaInv = 1.0f / area;

    const float C = 0.5f * x01CrossX02 * undeformedAreaInv - 1.0f;

    const V2 grad1 = V2{ 0.5f * undeformedAreaInv * x02.y, -(0.5f * undeformedAreaInv * x02.x) };
    const V2 grad2 = V2{ -(0.5f * undeformedAreaInv * x01.y), 0.5f * undeformedAreaInv * x01.x };
    const V2 grad0 = sub2(neg2(grad1), grad2);

    const float deltaLambda = queryDeltaLambda(C, grad0, grad1, grad2, alphaTilde, lambda, m0, m1, m2);
    lambda = lambda + deltaLambda;

    dx0 = add2(dx0, muls2(grad0, m0 * deltaLambda));
    dx1 = add2(dx1, muls2(grad1, m1 * deltaLambda));
    dx2 = add2(dx2, muls2(grad2, m2 * deltaLambda));
}

static void refCm(const CmIn& s, CmOut& o)
{
    const float youngs = s.mat.x, poissons = s.mat.y, thickness = s.mat.z;
    const bool isTGS = f2u(s.mat.w) == 1u;
    const float vs0 = s.vscale.x, vs1 = s.vscale.y, vs2 = s.vscale.z;
    const float dt = s.vscale.w;
    const F4 QInv = s.qinv;

    if (youngs < FEMCLOTH_THRESHOLD)
    {
        o.xx0 = s.xx0; o.xx1 = s.xx1; o.xx2 = s.xx2;
        o.mulOut = F4{ s.mul.x, s.mul.y, 0.f, 0.f };
        return;
    }

    V3 x0 = V3{ s.xx0.x, s.xx0.y, s.xx0.z };
    V3 x1 = V3{ s.xx1.x, s.xx1.y, s.xx1.z };
    V3 x2 = V3{ s.xx2.x, s.xx2.y, s.xx2.z };

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
        lambda0 = s.mul.x;
        lambda1 = s.mul.y;
    }

    const float lamL = youngs * poissons / ((1.0f + poissons) * (1.0f - 2.0f * poissons));
    const float mu = youngs / (2.0f * (1.0f + poissons));

    V2 xp01 = V2{ dot3(axis0, x01), dot3(axis1, x01) };
    V2 xp02 = V2{ dot3(axis0, x02), dot3(axis1, x02) };

    const float alphaTilde0 = 1.0f / (2.0f * mu * volume * dt2);

    arapConstraint2D(lambda0, dx0, dx1, dx2, alphaTilde0, QInv, xp01, xp02,
                     vs0 * s.xx0.w, vs1 * s.xx1.w, vs2 * s.xx2.w);

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
                         vs0 * s.xx0.w, vs1 * s.xx1.w, vs2 * s.xx2.w, area);
    }

    x0 = add3(x0, add3(muls3(axis0, dx0.x), muls3(axis1, dx0.y)));
    x1 = add3(x1, add3(muls3(axis0, dx1.x), muls3(axis1, dx1.y)));
    x2 = add3(x2, add3(muls3(axis0, dx2.x), muls3(axis1, dx2.y)));

    float mulOutX = s.mul.x, mulOutY = s.mul.y;
    if (!isTGS)
    {
        mulOutX = lambda0;
        mulOutY = lambda1;
    }

    o.xx0 = F4{ x0.x, x0.y, x0.z, s.xx0.w };
    o.xx1 = F4{ x1.x, x1.y, x1.z, s.xx1.w };
    o.xx2 = F4{ x2.x, x2.y, x2.z, s.xx2.w };
    o.mulOut = F4{ mulOutX, mulOutY, 0.f, 0.f };
}

PXG_TEST(npClothMembrane)
{
    std::mt19937 rng(255001);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<CmIn> in(N);
    std::vector<CmOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        CmIn s{};
        s.xx0 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), rf(0.f, 3.f) };
        s.xx1 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), rf(0.f, 3.f) };
        s.xx2 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), rf(0.f, 3.f) };
        // QInv with det away from zero (area = 1/(2*det) must stay sane)
        for (;;)
        {
            s.qinv = F4{ rf(-1.5f, 1.5f), rf(-1.5f, 1.5f), rf(-1.5f, 1.5f), rf(-1.5f, 1.5f) };
            const float det = s.qinv.x * s.qinv.w - s.qinv.y * s.qinv.z;
            if (det > 0.05f || det < -0.05f) break;
        }
        s.vscale = F4{ rf(0.f, 2.f), rf(0.f, 2.f), rf(0.f, 2.f), rf(0.005f, 0.1f) };  // vertexScale0..2, dt
        const uint32_t isTGS = i % 2u;
        s.mat = F4{ rf(1e4f, 1e8f), rf(0.f, 0.49f), rf(1e-3f, 0.1f), u2f(isTGS) };    // youngs, poissons, thickness
        s.mul = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f, 0.f };
        // poissons == 0 stripe: area constraint skipped
        if ((i % 7u) == 2u) s.mat.y = 0.f;
        // rest-pose stripe: F == R exactly -> ARAP C == 0 skips the C>threshold
        // branch (identity QInv, unit right triangle in the xy plane)
        if ((i % 13u) == 4u)
        {
            s.xx0.x = 0.f; s.xx0.y = 0.f; s.xx0.z = 0.f;
            s.xx1.x = 1.f; s.xx1.y = 0.f; s.xx1.z = 0.f;
            s.xx2.x = 0.f; s.xx2.y = 1.f; s.xx2.z = 0.f;
            s.qinv = F4{ 1.f, 0.f, 0.f, 1.f };
        }
        // early-out stripe: youngs = 0 -> all outputs pass through unchanged
        if ((i % 17u) == 3u) s.mat.x = 0.f;
        // threshold boundary stripe: youngs == 1e-14f exactly must NOT early-out
        if ((i % 17u) == 10u) s.mat.x = 1.0e-14f;
        // incompressible stripe: nu = 0.5 -> alphaTilde1 stays 0
        if ((i % 19u) == 5u) s.mat.y = 0.5f;
        // degenerate-denominator stripe: zero vertex scales + huge youngs make
        // denom < 1e-14 -> queryDeltaLambda returns 0 (lambda += 0, dx += 0)
        if ((i % 23u) == 7u)
        {
            s.vscale.x = 0.f; s.vscale.y = 0.f; s.vscale.z = 0.f;
            s.vscale.w = 0.1f;
            s.mat.x = 1e38f;
            s.mat.y = 0.3f;
            s.mat.z = 0.01f;
        }
        // degenerate-triangle stripe: x1 == x0 exactly -> x01 = 0 exercises the
        // getNormalized m > 0 guard (axis0 = axis1 = 0, F = 0, R = identity)
        if ((i % 29u) == 11u)
        {
            s.xx1.x = s.xx0.x; s.xx1.y = s.xx0.y; s.xx1.z = s.xx0.z;
        }
        in[i] = s;
        refCm(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<CmOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npClothMembrane.spv", "clothMembrane",
             { (uint32_t)sizeof(CmIn), (uint32_t)sizeof(CmOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<CmOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const F4* rr = &ref[i].xx0; const F4* gg = &g[i].xx0;
        bool ok = true;
        for (int f = 0; f < 3 && ok; ++f)
            ok = eqFZ(rr[f].x, gg[f].x) && eqFZ(rr[f].y, gg[f].y) && eqFZ(rr[f].z, gg[f].z) && eqFZ(rr[f].w, gg[f].w);
        ok = ok && eqFZ(ref[i].mulOut.x, g[i].mulOut.x) && eqFZ(ref[i].mulOut.y, g[i].mulOut.y);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u x0R (%g %g %g) x0G (%g %g %g) mulR (%g %g) mulG (%g %g)\n", i,
                            ref[i].xx0.x, ref[i].xx0.y, ref[i].xx0.z, g[i].xx0.x, g[i].xx0.y, g[i].xx0.z,
                            ref[i].mulOut.x, ref[i].mulOut.y, g[i].mulOut.x, g[i].mulOut.y);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "clothMembrane outputs bitwise-identical (mod +-0)");
}

}  // namespace
