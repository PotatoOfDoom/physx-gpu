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

// Verification of npSbStress.slang (sbStress): the sb_calculateStressLaunch
// compute core (softBodyGM.cu L2756-2817) - corotated Cauchy stress with the
// Pixar Stable-NeoHookean Lame reparameterization and the von-Mises stress
// coefficient - against an exact CPU transcription.  The per-tet rotation
// quaternion is a data input (no trig).  Outputs bitwise mod +-0 (eqFZ).

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
struct StIn  { F4 p0, p1, p2, p3, qc0, qc1, qc2, quat, mat; };
struct StOut { F4 s0, s1, s2, aux; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed

struct V3 { float x, y, z; };
static V3 add3(V3 a, V3 b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 sub3(V3 a, V3 b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 muls3(V3 a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; }

static V3 mat33Transform(V3 c0, V3 c1, V3 c2, V3 v)
{
    return add3(add3(muls3(c0, v.x), muls3(c1, v.y)), muls3(c2, v.z));
}

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

static void refSt(const StIn& s, StOut& o)
{
    const V3 u1 = sub3(V3{ s.p1.x, s.p1.y, s.p1.z }, V3{ s.p0.x, s.p0.y, s.p0.z });
    const V3 u2 = sub3(V3{ s.p2.x, s.p2.y, s.p2.z }, V3{ s.p0.x, s.p0.y, s.p0.z });
    const V3 u3 = sub3(V3{ s.p3.x, s.p3.y, s.p3.z }, V3{ s.p0.x, s.p0.y, s.p0.z });

    const V3 Q0 = V3{ s.qc0.x, s.qc0.y, s.qc0.z };
    const V3 Q1 = V3{ s.qc1.x, s.qc1.y, s.qc1.z };
    const V3 Q2 = V3{ s.qc2.x, s.qc2.y, s.qc2.z };

    const V3 F0 = mat33Transform(u1, u2, u3, Q0);
    const V3 F1 = mat33Transform(u1, u2, u3, Q1);
    const V3 F2 = mat33Transform(u1, u2, u3, Q2);

    V3 Rc0, Rc1, Rc2;
    quatToMat(s.quat, Rc0, Rc1, Rc2);
    const V3 T0 = V3{ Rc0.x, Rc1.x, Rc2.x };
    const V3 T1 = V3{ Rc0.y, Rc1.y, Rc2.y };
    const V3 T2 = V3{ Rc0.z, Rc1.z, Rc2.z };

    const V3 G0 = mat33Transform(T0, T1, T2, F0);
    const V3 G1 = mat33Transform(T0, T1, T2, F1);
    const V3 G2 = mat33Transform(T0, T1, T2, F2);

    const float E = s.mat.x;
    const float nu = s.mat.y;
    float lambda = E * nu / ((1.0f + nu) * (1.0f - 2.0f * nu));
    float mu = 0.5f * E / (1.0f + nu);
    lambda += mu * (5.0f / 6.0f);
    mu = mu * (4.0f / 3.0f);

    V3 e0 = muls3(add3(V3{ G0.x, G1.x, G2.x }, G0), 0.5f);
    V3 e1 = muls3(add3(V3{ G0.y, G1.y, G2.y }, G1), 0.5f);
    V3 e2 = muls3(add3(V3{ G0.z, G1.z, G2.z }, G2), 0.5f);
    e0.x -= 1.0f;
    e1.y -= 1.0f;
    e2.z -= 1.0f;

    const float trace = e0.x + e1.y + e2.z;

    const float dg = (1.0f * lambda) * trace;
    const float od = (0.0f * lambda) * trace;
    const V3 s0 = add3(muls3(muls3(e0, 2.0f), mu), V3{ dg, od, od });
    const V3 s1 = add3(muls3(muls3(e1, 2.0f), mu), V3{ od, dg, od });
    const V3 s2 = add3(muls3(muls3(e2, 2.0f), mu), V3{ od, od, dg });

    const float q1 = e0.x, q2 = e1.y, q3 = e2.z;
    const float q12 = q1 - q2;
    const float q23 = q2 - q3;
    const float q31 = q3 - q1;
    const float constant = 0.70710678118654752440084436210485f;
    const float stress = std::sqrt(q12 * q12 + q23 * q23 + q31 * q31) * constant;

    o.s0 = F4{ s0.x, s0.y, s0.z, 0.f };
    o.s1 = F4{ s1.x, s1.y, s1.z, 0.f };
    o.s2 = F4{ s2.x, s2.y, s2.z, 0.f };
    o.aux = F4{ stress, 0.f, 0.f, 0.f };
}

PXG_TEST(npSbStress)
{
    std::mt19937 rng(261001);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<StIn> in(N);
    std::vector<StOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        StIn s{};
        s.p0 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
        s.p1 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
        s.p2 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
        s.p3 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
        s.qc0 = F4{ rf(-1.5f, 1.5f), rf(-1.5f, 1.5f), rf(-1.5f, 1.5f), 0.f };
        s.qc1 = F4{ rf(-1.5f, 1.5f), rf(-1.5f, 1.5f), rf(-1.5f, 1.5f), 0.f };
        s.qc2 = F4{ rf(-1.5f, 1.5f), rf(-1.5f, 1.5f), rf(-1.5f, 1.5f), 0.f };
        {
            float qx = rf(-1.f, 1.f), qy = rf(-1.f, 1.f), qz = rf(-1.f, 1.f), qw = rf(-1.f, 1.f);
            const float n = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
            if (n > 1e-3f) { qx /= n; qy /= n; qz /= n; qw /= n; }
            else { qx = 0.f; qy = 0.f; qz = 0.f; qw = 1.f; }
            s.quat = F4{ qx, qy, qz, qw };
        }
        s.mat = F4{ rf(1e4f, 1e8f), rf(0.f, 0.49f), 0.f, 0.f };
        // nu == 0: base lambda term vanishes, reparam terms remain
        if ((i % 7u) == 2u) s.mat.y = 0.f;
        // identity rotation stripe
        if ((i % 11u) == 4u) s.quat = F4{ 0.f, 0.f, 0.f, 1.f };
        // exact rest pose: P = I, Qinv = I, R = I -> eps = 0, stress = 0
        if ((i % 13u) == 6u)
        {
            s.p0 = F4{ 0.f, 0.f, 0.f, 0.f };
            s.p1 = F4{ 1.f, 0.f, 0.f, 0.f };
            s.p2 = F4{ 0.f, 1.f, 0.f, 0.f };
            s.p3 = F4{ 0.f, 0.f, 1.f, 0.f };
            s.qc0 = F4{ 1.f, 0.f, 0.f, 0.f };
            s.qc1 = F4{ 0.f, 1.f, 0.f, 0.f };
            s.qc2 = F4{ 0.f, 0.f, 1.f, 0.f };
            s.quat = F4{ 0.f, 0.f, 0.f, 1.f };
        }
        in[i] = s;
        refSt(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<StOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npSbStress.spv", "sbStress",
             { (uint32_t)sizeof(StIn), (uint32_t)sizeof(StOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<StOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const F4* rr = &ref[i].s0; const F4* gg = &g[i].s0;
        bool ok = true;
        for (int f = 0; f < 3 && ok; ++f)
            ok = eqFZ(rr[f].x, gg[f].x) && eqFZ(rr[f].y, gg[f].y) && eqFZ(rr[f].z, gg[f].z);
        ok = ok && eqFZ(ref[i].aux.x, g[i].aux.x);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u s0R (%g %g %g) s0G (%g %g %g) stR %g stG %g\n", i,
                            ref[i].s0.x, ref[i].s0.y, ref[i].s0.z, g[i].s0.x, g[i].s0.y, g[i].s0.z,
                            ref[i].aux.x, g[i].aux.x);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "sbStress outputs bitwise-identical (mod +-0)");
}

}  // namespace
