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

// Verification of npClothBending.slang (clothBending): the FEM-cloth Discrete
// Shells dihedral bending constraint bendingEnergySolvePerTrianglePair
// (FEMClothUtil.cuh L543-655), against an exact CPU transcription.  The
// atan2f evaluation is the documented non-portable seam: the host computes
// angle = atan2f(sinAngle, cosAngle) from a CPU replica of the shader's exact
// sinAngle/cosAngle and passes it in; the shader outputs its own
// sinAngle/cosAngle, compared bitwise, pinning everything upstream of the
// seam.  Everything downstream is ported and compared verbatim.  The lambda
// slot is modeled as an in/out scalar.  Outputs bitwise mod +-0 (eqFZ).

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
struct CbIn  { F4 x0, x1, x2, x3, vrc, bend, misc; };
struct CbOut { F4 x0, x1, x2, x3, aux; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed

const float FEMCLOTH_THRESHOLD = 1.0e-14f;
const float FEMCLOTH_HALF_PI = 1.57079632679489661923f;
const float FEMCLOTH_2PI = 6.28318530717958647692f;

struct V3 { float x, y, z; };
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

// Replicates the shader up to (sinAngle, cosAngle).  Returns false on either
// early-out (pathTaken 1 or 2).
static bool trigOf(const CbIn& s, float& sinAngle, float& cosAngle, uint32_t& path)
{
    const V3 p0 = V3{ s.x0.x, s.x0.y, s.x0.z }, p1 = V3{ s.x1.x, s.x1.y, s.x1.z };
    const V3 p2 = V3{ s.x2.x, s.x2.y, s.x2.z }, p3 = V3{ s.x3.x, s.x3.y, s.x3.z };
    const V3 x02 = sub3(p2, p0), x03 = sub3(p3, p0), x13 = sub3(p3, p1), x12 = sub3(p2, p1), x23 = sub3(p3, p2);
    const float x23Len = std::sqrt(magSq3(x23));
    if (x23Len < FEMCLOTH_THRESHOLD) { path = 1u; return false; }
    if (s.bend.y <= 0.f) { path = 2u; return false; }

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
    path = 0u;
    return true;
}

static void refCb(const CbIn& s, CbOut& o)
{
    o.x0 = s.x0; o.x1 = s.x1; o.x2 = s.x2; o.x3 = s.x3;

    float sinAngle = 0.f, cosAngle = 0.f;
    uint32_t path = 0u;
    if (!trigOf(s, sinAngle, cosAngle, path))
    {
        o.aux = F4{ s.misc.x, 0.f, 0.f, u2f(path) };
        return;
    }

    const float restBendingAngle = s.bend.x, kInv = s.bend.y, dt = s.bend.z;
    const bool isTGS = f2u(s.bend.w) == 1u;
    const float angle = s.misc.y;   // host atan2f of (sinAngle, cosAngle)

    const V3 p0 = V3{ s.x0.x, s.x0.y, s.x0.z }, p1 = V3{ s.x1.x, s.x1.y, s.x1.z };
    const V3 p2 = V3{ s.x2.x, s.x2.y, s.x2.z }, p3 = V3{ s.x3.x, s.x3.y, s.x3.z };
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
        lambda = s.misc.x;

    const float m0 = s.vrc.x * s.x0.w;
    const float m1 = s.vrc.y * s.x1.w;
    const float m2 = s.vrc.z * s.x2.w;
    const float m3 = s.vrc.w * s.x3.w;
    const float denom = (m0 * magSq3(dCdx0) + m1 * magSq3(dCdx1) + m2 * magSq3(dCdx2) + m3 * magSq3(dCdx3)) + alphaTilde;
    float deltaLambda;
    if (denom < FEMCLOTH_THRESHOLD)
        deltaLambda = 0.f;
    else
        deltaLambda = (-C - alphaTilde * lambda) / denom;

    float lambdaOut = s.misc.x;
    if (!isTGS)
        lambdaOut = lambda + deltaLambda;

    const float scale0 = s.vrc.x * s.x0.w * deltaLambda;
    const float scale1 = s.vrc.y * s.x1.w * deltaLambda;
    const float scale2 = s.vrc.z * s.x2.w * deltaLambda;
    const float scale3 = s.vrc.w * s.x3.w * deltaLambda;

    o.x0 = F4{ s.x0.x + scale0 * dCdx0.x, s.x0.y + scale0 * dCdx0.y, s.x0.z + scale0 * dCdx0.z, s.x0.w };
    o.x1 = F4{ s.x1.x + scale1 * dCdx1.x, s.x1.y + scale1 * dCdx1.y, s.x1.z + scale1 * dCdx1.z, s.x1.w };
    o.x2 = F4{ s.x2.x + scale2 * dCdx2.x, s.x2.y + scale2 * dCdx2.y, s.x2.z + scale2 * dCdx2.z, s.x2.w };
    o.x3 = F4{ s.x3.x + scale3 * dCdx3.x, s.x3.y + scale3 * dCdx3.y, s.x3.z + scale3 * dCdx3.z, s.x3.w };
    o.aux = F4{ lambdaOut, sinAngle, cosAngle, u2f(0u) };
}

PXG_TEST(npClothBending)
{
    std::mt19937 rng(256001);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<CbIn> in(N);
    std::vector<CbOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        CbIn s{};
        s.x0 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), rf(0.f, 3.f) };
        s.x1 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), rf(0.f, 3.f) };
        s.x2 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), rf(0.f, 3.f) };
        s.x3 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), rf(0.f, 3.f) };
        s.vrc = F4{ rf(0.f, 1.f), rf(0.f, 1.f), rf(0.f, 1.f), rf(0.f, 1.f) };
        const uint32_t isTGS = i % 2u;
        s.bend = F4{ rf(-3.14f, 3.14f), rf(1e-8f, 1e-2f), rf(0.005f, 0.1f), u2f(isTGS) };  // rest angle, kInv, dt
        s.misc = F4{ rf(-2.f, 2.f), 0.f, 0.f, 0.f };                                       // lambda in; angle filled below
        // C-wrap stripes: rest angle near +-pi forces the 2PI wraps
        if ((i % 7u) == 2u) s.bend.x = 3.0f;
        if ((i % 7u) == 5u) s.bend.x = -3.0f;
        // kInv <= 0 early-out stripe (zero and negative variants)
        if ((i % 11u) == 3u) s.bend.y = ((i / 11u) % 2u) ? -0.5f : 0.f;
        // x3 == x2 exactly: x23Len = 0 early-out (before any normal math)
        if ((i % 13u) == 6u) s.x3 = F4{ s.x2.x, s.x2.y, s.x2.z, s.x3.w };
        // degenerate-denominator stripe: zero reference counts + tiny kInv
        if ((i % 23u) == 7u)
        {
            s.vrc = F4{ 0.f, 0.f, 0.f, 0.f };
            s.bend.y = 1e-38f;
            s.bend.z = 1.f;
        }
        // host-side atan2f of the exact shader trig values (the documented seam)
        float sinAngle, cosAngle;
        uint32_t path;
        if (trigOf(s, sinAngle, cosAngle, path))
            s.misc.y = std::atan2(sinAngle, cosAngle);
        in[i] = s;
        refCb(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<CbOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npClothBending.spv", "clothBending",
             { (uint32_t)sizeof(CbIn), (uint32_t)sizeof(CbOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<CbOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const F4* rr = &ref[i].x0; const F4* gg = &g[i].x0;
        bool ok = true;
        for (int f = 0; f < 4 && ok; ++f)
            ok = eqFZ(rr[f].x, gg[f].x) && eqFZ(rr[f].y, gg[f].y) && eqFZ(rr[f].z, gg[f].z) && eqFZ(rr[f].w, gg[f].w);
        ok = ok && eqFZ(ref[i].aux.x, g[i].aux.x) && eqFZ(ref[i].aux.y, g[i].aux.y)
               && eqFZ(ref[i].aux.z, g[i].aux.z) && f2u(ref[i].aux.w) == f2u(g[i].aux.w);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u x0R (%g %g %g) x0G (%g %g %g) auxR (%g %g %g %u) auxG (%g %g %g %u)\n", i,
                            ref[i].x0.x, ref[i].x0.y, ref[i].x0.z, g[i].x0.x, g[i].x0.y, g[i].x0.z,
                            ref[i].aux.x, ref[i].aux.y, ref[i].aux.z, f2u(ref[i].aux.w),
                            g[i].aux.x, g[i].aux.y, g[i].aux.z, f2u(g[i].aux.w));
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "clothBending outputs bitwise-identical (mod +-0)");
}

}  // namespace
