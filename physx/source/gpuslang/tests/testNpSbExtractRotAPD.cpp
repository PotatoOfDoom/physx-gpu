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

// Verification of npSbExtractRotAPD.slang (sbExtractRotAPD): the
// analytic-polar-decomposition rotation extractor sb_extractRotationAPD
// (deformableUtils.cuh L69-116), against an exact CPU transcription (including
// the two double-precision spots at L87 and L109).  Quaternion bitwise mod +-0.

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
struct U4 { uint32_t x, y, z, w; };
struct ArIn  { F4 fc0, fc1, fc2, q0; U4 it; };
struct ArOut { F4 q; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed

struct V3 { float x, y, z; };
struct Q4 { float x, y, z, w; };

static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// PxQuat operator* (PxQuat.h L388-392).
static Q4 quatMul(Q4 a, Q4 b)
{
    return Q4{ a.w * b.x + b.w * a.x + a.y * b.z - b.y * a.z,
               a.w * b.y + b.w * a.y + a.z * b.x - b.z * a.x,
               a.w * b.z + b.w * a.z + a.x * b.y - b.x * a.y,
               a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z };
}

// PxMat33(PxQuat) rotation ctor (PxMat33.h L136-163).
static void mat33FromQuat(Q4 q, V3& c0, V3& c1, V3& c2)
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

static V3 mat33Transform(V3 c0, V3 c1, V3 c2, V3 v)
{
    return V3{ c0.x * v.x + c1.x * v.y + c2.x * v.z,
               c0.y * v.x + c1.y * v.y + c2.y * v.z,
               c0.z * v.x + c1.z * v.y + c2.z * v.z };
}

// Exact CPU transcription of sb_extractRotationAPD (deformableUtils.cuh L69-116).
static Q4 refAPD(V3 F0, V3 F1, V3 F2, Q4 q, int maxIter)
{
    const float eps = 1.0e-6;          // double literal narrowed, as in the source
    const float threshold = 1 - eps;
    for (int i = 0; i < maxIter; ++i)
    {
        V3 r0, r1, r2;
        const Q4 conj = Q4{ -q.x, -q.y, -q.z, q.w };
        mat33FromQuat(conj, r0, r1, r2);
        const V3 B0 = mat33Transform(r0, r1, r2, F0);
        const V3 B1 = mat33Transform(r0, r1, r2, F1);
        const V3 B2 = mat33Transform(r0, r1, r2, F2);

        V3 gradient = V3{ B2.y - B1.z, B0.z - B2.x, B1.x - B0.y };
        if (i == 0 && dot3(gradient, gradient) < 1e-16)   // f64 compare (double literal)
            gradient = V3{ -2.f, 0.f, 0.f };
        const float h00 = B1.y + B2.z;
        const float h11 = B0.x + B2.z;
        const float h22 = B0.x + B1.y;
        const float h01 = -0.5f * (B1.x + B0.y);
        const float h02 = -0.5f * (B2.x + B0.z);
        const float h12 = -0.5f * (B2.y + B1.z);
        const float detH = -h02 * h02 * h11 + 2.0f * h01 * h02 * h12 - h00 * h12 * h12 - h01 * h01 * h22 + h00 * h11 * h22;
        V3 omega;
        const float factor = -0.25f / detH;
        omega.x = factor * ((h11 * h22 - h12 * h12) * gradient.x + (h02 * h12 - h01 * h22) * gradient.y + (h01 * h12 - h02 * h11) * gradient.z);
        omega.y = factor * ((h02 * h12 - h01 * h22) * gradient.x + (h00 * h22 - h02 * h02) * gradient.y + (h01 * h02 - h00 * h12) * gradient.z);
        omega.z = factor * ((h01 * h12 - h02 * h11) * gradient.x + (h01 * h02 - h00 * h12) * gradient.y + (h00 * h11 - h01 * h01) * gradient.z);
        if (std::fabs(detH) < 1e-9f)
            omega = V3{ -gradient.x, -gradient.y, -gradient.z };
        if (dot3(omega, gradient) > 0.0f)
            omega = V3{ gradient.x * -0.125f, gradient.y * -0.125f, gradient.z * -0.125f };
        const float l_omega2 = dot3(omega, omega);
        const float w = (1.0 - l_omega2) / (1.0f + l_omega2);   // f64 sub+div, narrowed
        const float s2 = 2.0f / (1.0f + l_omega2);
        const V3 vec = V3{ omega.x * s2, omega.y * s2, omega.z * s2 };
        q = quatMul(q, Q4{ vec.x, vec.y, vec.z, w });
        if (w > threshold)
            break;
    }
    return q;
}

PXG_TEST(npSbExtractRotAPD)
{
    std::mt19937 rng(251001);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<ArIn> in(N);
    std::vector<ArOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        ArIn s{};
        // random unit-ish initial quaternion
        Q4 q0;
        {
            float qx = rf(-1.f, 1.f), qy = rf(-1.f, 1.f), qz = rf(-1.f, 1.f), qw = rf(-1.f, 1.f);
            const float n = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
            const float inv = (n > 1e-3f) ? 1.0f / n : 1.0f;
            q0 = Q4{ qx * inv, qy * inv, qz * inv, (n > 1e-3f) ? qw * inv : 1.0f };
        }
        V3 F0 = V3{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f) };
        V3 F1 = V3{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f) };
        V3 F2 = V3{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f) };
        uint32_t maxIter = 1u + (i % 6u);

        const uint32_t sel = i % 7u;
        if (sel == 1u)
        {
            // near-rotation: F = R(q_true) + small noise -> convergence path
            V3 c0, c1, c2;
            mat33FromQuat(q0, c0, c1, c2);
            F0 = V3{ c0.x + rf(-0.05f, 0.05f), c0.y + rf(-0.05f, 0.05f), c0.z + rf(-0.05f, 0.05f) };
            F1 = V3{ c1.x + rf(-0.05f, 0.05f), c1.y + rf(-0.05f, 0.05f), c1.z + rf(-0.05f, 0.05f) };
            F2 = V3{ c2.x + rf(-0.05f, 0.05f), c2.y + rf(-0.05f, 0.05f), c2.z + rf(-0.05f, 0.05f) };
        }
        else if (sel == 3u)
        {
            // exact rotation of q0: B = R^T*R ~ identity so the iteration-0
            // gradient magnitude^2 falls below 1e-16 and the DISTORTION branch
            // fires (kicking omega ~ 0.25 about x), exercising recovery from a
            // converged start.  (The w > threshold early exit is covered by the
            // near-rotation rows, ~2300 hits.)
            V3 c0, c1, c2;
            mat33FromQuat(q0, c0, c1, c2);
            F0 = c0; F1 = c1; F2 = c2;
            maxIter = 4u;
        }
        else if (sel == 5u)
        {
            // 180-degree rotation about x with identity q0: zero gradient in
            // iteration 0 -> distorted-gradient branch (L87-91)
            F0 = V3{ 1.f, 0.f, 0.f }; F1 = V3{ 0.f, -1.f, 0.f }; F2 = V3{ 0.f, 0.f, -1.f };
            q0 = Q4{ 0.f, 0.f, 0.f, 1.f };
            maxIter = 3u;
        }
        else if (sel == 6u)
        {
            // zero F with identity q0: detH == 0 -> fabs(detH) < 1e-9 fallback
            // (omega = -gradient, after the NaN-producing solve is discarded)
            F0 = V3{ 0.f, 0.f, 0.f }; F1 = V3{ 0.f, 0.f, 0.f }; F2 = V3{ 0.f, 0.f, 0.f };
            q0 = Q4{ 0.f, 0.f, 0.f, 1.f };
            maxIter = 2u;
        }

        s.fc0 = F4{ F0.x, F0.y, F0.z, 0.f };
        s.fc1 = F4{ F1.x, F1.y, F1.z, 0.f };
        s.fc2 = F4{ F2.x, F2.y, F2.z, 0.f };
        s.q0 = F4{ q0.x, q0.y, q0.z, q0.w };
        s.it = U4{ maxIter, 0u, 0u, 0u };
        in[i] = s;
        const Q4 r = refAPD(F0, F1, F2, q0, int(maxIter));
        ref[i].q = F4{ r.x, r.y, r.z, r.w };
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<ArOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npSbExtractRotAPD.spv", "sbExtractRotAPD",
             { (uint32_t)sizeof(ArIn), (uint32_t)sizeof(ArOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<ArOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        bool ok = eqFZ(ref[i].q.x, g[i].q.x) && eqFZ(ref[i].q.y, g[i].q.y)
               && eqFZ(ref[i].q.z, g[i].q.z) && eqFZ(ref[i].q.w, g[i].q.w);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u R (%g %g %g %g) G (%g %g %g %g)\n", i,
                            ref[i].q.x, ref[i].q.y, ref[i].q.z, ref[i].q.w,
                            g[i].q.x, g[i].q.y, g[i].q.z, g[i].q.w);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "sbExtractRotAPD quaternions bitwise-identical (mod +-0)");
}

}  // namespace
