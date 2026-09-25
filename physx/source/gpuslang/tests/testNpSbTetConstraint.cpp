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

// Verification of npSbTetConstraint.slang (sbTetConstraint): the three soft-body
// XPBD tetrahedron constraints ARAP_constraint / traceSMinusI_constraint /
// volume_constraint (softBodyGM.cu L722-767), each deriving (C, dCdF) then
// running the shared compute_deltaLambdaXgradC core, against an exact CPU
// transcription.  Outputs bitwise mod +-0 (eqFZ).

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
struct TcIn  { F4 misc, misc2, quat, fc0, fc1, fc2, qc0, qc1, qc2, v0, v1, v2, v3, acc0, acc1, acc2; };
struct TcOut { F4 dc0, dc1, dc2, scalars; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed

struct V3 { float x, y, z; };
static V3 mk(float x, float y, float z) { return V3{ x, y, z }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b) { return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static float magSq3(V3 v) { return v.x * v.x + v.y * v.y + v.z * v.z; }
static float pabs(float a) { return a < 0.f ? -a : a; }
static V3 sclv(V3 v, float s) { return V3{ v.x * s, v.y * s, v.z * s }; }
static V3 subv(V3 a, V3 b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }

static void quatToMat(F4 q, V3& c0, V3& c1, V3& c2)
{
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    const float x2 = x + x, y2 = y + y, z2 = z + z;
    const float xx = x2 * x, yy = y2 * y, zz = z2 * z;
    const float xy = x2 * y, xz = x2 * z, xw = x2 * w;
    const float yz = y2 * z, yw = y2 * w, zw = z2 * w;
    c0 = V3{ 1.f - yy - zz, xy + zw, xz - yw };
    c1 = V3{ xy - zw, 1.f - xx - zz, yz + xw };
    c2 = V3{ xz + yw, yz - xw, 1.f - xx - yy };
}

static void refTc(const TcIn& s, TcOut& o)
{
    float lambda = s.misc.x;
    const float alphaTilde = s.misc.y, dtInv = s.misc.z;
    const uint32_t ctype = f2u(s.misc.w);
    const float damping = s.misc2.x, scaledDamping = s.misc2.y;

    const V3 Fc0 = mk(s.fc0.x, s.fc0.y, s.fc0.z);
    const V3 Fc1 = mk(s.fc1.x, s.fc1.y, s.fc1.z);
    const V3 Fc2 = mk(s.fc2.x, s.fc2.y, s.fc2.z);

    float C;
    V3 dFc0, dFc1, dFc2;
    if (ctype == 0u)
    {
        V3 Rc0, Rc1, Rc2;
        quatToMat(s.quat, Rc0, Rc1, Rc2);
        const V3 t0 = subv(Fc0, Rc0), t1 = subv(Fc1, Rc1), t2 = subv(Fc2, Rc2);
        C = std::sqrt(dot3(t0, t0) + dot3(t1, t1) + dot3(t2, t2) + 1.0e-18f);
        const float inv = 1.0f / C;
        dFc0 = sclv(t0, inv); dFc1 = sclv(t1, inv); dFc2 = sclv(t2, inv);
    }
    else if (ctype == 1u)
    {
        V3 Rc0, Rc1, Rc2;
        quatToMat(s.quat, Rc0, Rc1, Rc2);
        C = dot3(Rc0, Fc0) + dot3(Rc1, Fc1) + dot3(Rc2, Fc2) - 3.0f;
        dFc0 = Rc0; dFc1 = Rc1; dFc2 = Rc2;
    }
    else
    {
        const float detF = dot3(Fc0, cross3(Fc1, Fc2));
        C = detF - 1.0f;
        dFc0 = cross3(Fc1, Fc2);
        dFc1 = cross3(Fc2, Fc0);
        dFc2 = cross3(Fc0, Fc1);
    }

    const V3 r0 = mk(s.qc0.x, s.qc1.x, s.qc2.x);
    const V3 r1 = mk(s.qc0.y, s.qc1.y, s.qc2.y);
    const V3 r2 = mk(s.qc0.z, s.qc1.z, s.qc2.z);
    const V3 p0 = mk(dFc0.x, dFc1.x, dFc2.x);
    const V3 p1 = mk(dFc0.y, dFc1.y, dFc2.y);
    const V3 p2 = mk(dFc0.z, dFc1.z, dFc2.z);
    const V3 dCdx1 = mk(dot3(r0, p0), dot3(r0, p1), dot3(r0, p2));
    const V3 dCdx2 = mk(dot3(r1, p0), dot3(r1, p1), dot3(r1, p2));
    const V3 dCdx3 = mk(dot3(r2, p0), dot3(r2, p1), dot3(r2, p2));
    const V3 dCdx0 = mk(-dCdx1.x - dCdx2.x - dCdx3.x, -dCdx1.y - dCdx2.y - dCdx3.y, -dCdx1.z - dCdx2.z - dCdx3.z);

    const float w0 = s.v0.w, w1 = s.v1.w, w2 = s.v2.w, w3 = s.v3.w;
    const float sumWMag = magSq3(dCdx0) * w0 + magSq3(dCdx1) * w1 + magSq3(dCdx2) * w2 + magSq3(dCdx3) * w3;

    float deltaLambda;
    if (damping == 0.f)
    {
        const float denom = sumWMag + alphaTilde;
        if (pabs(denom) < 1.0e-15f)
        {
            o.dc0 = s.acc0; o.dc1 = s.acc1; o.dc2 = s.acc2;
            o.scalars = F4{ lambda, C, 0.f, 0.f };
            return;
        }
        deltaLambda = (-C - alphaTilde * lambda) / denom;
    }
    else
    {
        const float denom = (1.0f + damping * dtInv) * sumWMag + alphaTilde;
        if (pabs(denom) < 1.0e-15f)
        {
            o.dc0 = F4{ 0, 0, 0, 0 }; o.dc1 = F4{ 0, 0, 0, 0 }; o.dc2 = F4{ 0, 0, 0, 0 };
            o.scalars = F4{ lambda, C, 0.f, 0.f };
            return;
        }
        const float dampDot = dot3(dCdx0, mk(s.v0.x, s.v0.y, s.v0.z)) + dot3(dCdx1, mk(s.v1.x, s.v1.y, s.v1.z))
                            + dot3(dCdx2, mk(s.v2.x, s.v2.y, s.v2.z)) + dot3(dCdx3, mk(s.v3.x, s.v3.y, s.v3.z));
        deltaLambda = (-C - alphaTilde * lambda - scaledDamping * dampDot) / denom;
    }

    lambda = lambda + deltaLambda;
    o.dc0 = F4{ s.acc0.x + deltaLambda * dCdx1.x, s.acc0.y + deltaLambda * dCdx1.y, s.acc0.z + deltaLambda * dCdx1.z, 0.f };
    o.dc1 = F4{ s.acc1.x + deltaLambda * dCdx2.x, s.acc1.y + deltaLambda * dCdx2.y, s.acc1.z + deltaLambda * dCdx2.z, 0.f };
    o.dc2 = F4{ s.acc2.x + deltaLambda * dCdx3.x, s.acc2.y + deltaLambda * dCdx3.y, s.acc2.z + deltaLambda * dCdx3.z, 0.f };
    o.scalars = F4{ lambda, C, 0.f, 0.f };
}

PXG_TEST(npSbTetConstraint)
{
    std::mt19937 rng(233001);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<TcIn> in(N);
    std::vector<TcOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        TcIn s{};
        const uint32_t ctype = i % 3u;   // 0=ARAP, 1=trace, 2=volume
        float damping = ((i % 2u) == 0u) ? 0.f : rf(0.1f, 3.f);
        s.misc = F4{ rf(-2.f, 2.f), rf(0.f, 2.f), rf(10.f, 120.f), u2f(ctype) };  // lambda, alphaTilde, dtInv, ctype
        s.misc2 = F4{ damping, rf(0.f, 2.f), 0.f, 0.f };
        // unit quaternion for the rotation (only used by ARAP/trace)
        float qx = rf(-1.f, 1.f), qy = rf(-1.f, 1.f), qz = rf(-1.f, 1.f), qw = rf(-1.f, 1.f);
        float qn = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
        if (qn < 1e-4f) { qx = qy = qz = 0.f; qw = 1.f; qn = 1.f; }
        s.quat = F4{ qx / qn, qy / qn, qz / qn, qw / qn };
        // F near a rotation + jitter (keeps det(F) away from 0 for the volume path)
        s.fc0 = F4{ rf(0.6f, 1.4f), rf(-0.4f, 0.4f), rf(-0.4f, 0.4f), 0.f };
        s.fc1 = F4{ rf(-0.4f, 0.4f), rf(0.6f, 1.4f), rf(-0.4f, 0.4f), 0.f };
        s.fc2 = F4{ rf(-0.4f, 0.4f), rf(-0.4f, 0.4f), rf(0.6f, 1.4f), 0.f };
        s.qc0 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
        s.qc1 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
        s.qc2 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
        s.v0 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), rf(0.2f, 3.f) };
        s.v1 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), rf(0.2f, 3.f) };
        s.v2 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), rf(0.2f, 3.f) };
        s.v3 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), rf(0.2f, 3.f) };
        s.acc0 = F4{ rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-1.f, 1.f), 0.f };
        s.acc1 = F4{ rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-1.f, 1.f), 0.f };
        s.acc2 = F4{ rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-1.f, 1.f), 0.f };
        // ARAP-rest stripe: force ctype=ARAP and F = R exactly, so the deformation
        // tensor F-R is 0 and the ARAP sum vanishes -> C = sqrt(SB_ARAP_EPSILON).
        // This makes the 1e-18 epsilon observable (otherwise it is dwarfed by the
        // O(1) sum for a general F and a change to it is an equivalent mutant).
        if ((i % 17u) == 5u)
        {
            s.misc.w = u2f(0u);   // ARAP
            V3 R0, R1, R2;
            quatToMat(s.quat, R0, R1, R2);
            s.fc0 = F4{ R0.x, R0.y, R0.z, 0.f };
            s.fc1 = F4{ R1.x, R1.y, R1.z, 0.f };
            s.fc2 = F4{ R2.x, R2.y, R2.z, 0.f };
        }
        in[i] = s;
        refTc(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<TcOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npSbTetConstraint.spv", "sbTetConstraint",
             { (uint32_t)sizeof(TcIn), (uint32_t)sizeof(TcOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<TcOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* rfp = &ref[i].dc0.x;
        const float* qfp = &g[i].dc0.x;
        const size_t nf = sizeof(TcOut) / 4;
        bool ok = true;
        for (size_t xi = 0; xi < nf && ok; ++xi) ok = eqFZ(rfp[xi], qfp[xi]);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u ctype %u CR %g CG %g lamR %g lamG %g dc0R (%g %g %g) dc0G (%g %g %g)\n", i,
                            f2u(in[i].misc.w), ref[i].scalars.y, g[i].scalars.y, ref[i].scalars.x, g[i].scalars.x,
                            ref[i].dc0.x, ref[i].dc0.y, ref[i].dc0.z, g[i].dc0.x, g[i].dc0.y, g[i].dc0.z);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "sbTetConstraint outputs bitwise-identical (mod +-0)");
}

}  // namespace
