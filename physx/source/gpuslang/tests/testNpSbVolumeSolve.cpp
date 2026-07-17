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

// Verification of npSbVolumeSolve.slang (sbVolumeSolve): the XPBD soft-body
// volume-constraint solve volumeSolve (softBodyGM.cu L621-655), against an exact
// CPU transcription.  Outputs bitwise mod +-0 (eqFZ).

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
struct VsIn  { F4 pc0, pc1, pc2, qc0, qc1, qc2, mass, params; };
struct VsOut { F4 dc0, dc1, dc2; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b) || (a == 0.f && b == 0.f); }

struct V3 { float x, y, z; };
static V3 mk(float x, float y, float z) { return V3{ x, y, z }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b) { return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static float magSq3(V3 v) { return v.x * v.x + v.y * v.y + v.z * v.z; }
static V3 sclv(V3 v, float s) { return V3{ v.x * s, v.y * s, v.z * s }; }
static V3 addv(V3 a, V3 b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }

static void refVol(const VsIn& s, VsOut& o)
{
    const V3 Pc0 = mk(s.pc0.x, s.pc0.y, s.pc0.z), Pc1 = mk(s.pc1.x, s.pc1.y, s.pc1.z), Pc2 = mk(s.pc2.x, s.pc2.y, s.pc2.z);
    const V3 Qc0 = mk(s.qc0.x, s.qc0.y, s.qc0.z), Qc1 = mk(s.qc1.x, s.qc1.y, s.qc1.z), Qc2 = mk(s.qc2.x, s.qc2.y, s.qc2.z);
    const float m0 = s.mass.x, m1 = s.mass.y, m2 = s.mass.z, m3 = s.mass.w;
    const float invDtSq = s.params.x, volumeLambda = s.params.y;

    const float det = dot3(Qc0, cross3(Qc1, Qc2));
    const float volume = 1.0f / (6.0f * det);
    const float alphaTilde = (volumeLambda == 0.f) ? 0.f : 1.f / (volumeLambda * volume * invDtSq);

    // F = P * Qinv : F.col_i = P.transform(Qinv.col_i) = Pc0*x + Pc1*y + Pc2*z
    const V3 Fc0 = addv(addv(sclv(Pc0, Qc0.x), sclv(Pc1, Qc0.y)), sclv(Pc2, Qc0.z));
    const V3 Fc1 = addv(addv(sclv(Pc0, Qc1.x), sclv(Pc1, Qc1.y)), sclv(Pc2, Qc1.z));
    const V3 Fc2 = addv(addv(sclv(Pc0, Qc2.x), sclv(Pc1, Qc2.y)), sclv(Pc2, Qc2.z));

    const V3 g0 = cross3(Fc1, Fc2);
    const V3 g1 = cross3(Fc2, Fc0);
    const V3 g2 = cross3(Fc0, Fc1);

    const V3 grad1 = addv(addv(sclv(g0, Qc0.x), sclv(g1, Qc1.x)), sclv(g2, Qc2.x));
    const V3 grad2 = addv(addv(sclv(g0, Qc0.y), sclv(g1, Qc1.y)), sclv(g2, Qc2.y));
    const V3 grad3 = addv(addv(sclv(g0, Qc0.z), sclv(g1, Qc1.z)), sclv(g2, Qc2.z));
    const V3 grad0 = V3{ -grad1.x - grad2.x - grad3.x, -grad1.y - grad2.y - grad3.y, -grad1.z - grad2.z - grad3.z };

    const float detF = dot3(g2, Fc2);
    const float C = detF - 1.f;

    const float w = magSq3(grad0) * m0 + magSq3(grad1) * m1 + magSq3(grad2) * m2 + magSq3(grad3) * m3;

    const float dlambda = -C / (w + alphaTilde);

    const V3 d0 = sclv(grad1, dlambda), d1 = sclv(grad2, dlambda), d2 = sclv(grad3, dlambda);
    o.dc0 = F4{ d0.x, d0.y, d0.z, 0.f };
    o.dc1 = F4{ d1.x, d1.y, d1.z, 0.f };
    o.dc2 = F4{ d2.x, d2.y, d2.z, 0.f };
}

PXG_TEST(npSbVolumeSolve)
{
    std::mt19937 rng(223001);
    const uint32_t N = 5000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<VsIn> in(N);
    std::vector<VsOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        VsIn s{};
        // P near identity + jitter (deformation gradient-ish, well away from singular)
        s.pc0 = F4{ rf(0.7f, 1.3f), rf(-0.3f, 0.3f), rf(-0.3f, 0.3f), 0.f };
        s.pc1 = F4{ rf(-0.3f, 0.3f), rf(0.7f, 1.3f), rf(-0.3f, 0.3f), 0.f };
        s.pc2 = F4{ rf(-0.3f, 0.3f), rf(-0.3f, 0.3f), rf(0.7f, 1.3f), 0.f };
        // Qinv near identity + jitter so det stays safely nonzero
        s.qc0 = F4{ rf(0.7f, 1.3f), rf(-0.3f, 0.3f), rf(-0.3f, 0.3f), 0.f };
        s.qc1 = F4{ rf(-0.3f, 0.3f), rf(0.7f, 1.3f), rf(-0.3f, 0.3f), 0.f };
        s.qc2 = F4{ rf(-0.3f, 0.3f), rf(-0.3f, 0.3f), rf(0.7f, 1.3f), 0.f };
        s.mass = F4{ rf(0.2f, 3.f), rf(0.2f, 3.f), rf(0.2f, 3.f), rf(0.2f, 3.f) };
        float volumeLambda = rf(0.1f, 4.f);
        // alphaTilde==0 stripe: volumeLambda == 0 exactly
        if ((i % 5u) == 2u) volumeLambda = 0.f;
        s.params = F4{ rf(100.f, 4000.f), volumeLambda, 0.f, 0.f };  // invDtSq, volumeLambda
        in[i] = s;
        refVol(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<VsOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npSbVolumeSolve.spv", "sbVolumeSolve",
             { (uint32_t)sizeof(VsIn), (uint32_t)sizeof(VsOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<VsOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* rfp = &ref[i].dc0.x;
        const float* qfp = &g[i].dc0.x;
        const size_t nf = sizeof(VsOut) / 4;
        bool ok = true;
        for (size_t xi = 0; xi < nf && ok; ++xi) ok = eqFZ(rfp[xi], qfp[xi]);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u dc0R (%g %g %g) dc0G (%g %g %g)\n", i,
                            ref[i].dc0.x, ref[i].dc0.y, ref[i].dc0.z, g[i].dc0.x, g[i].dc0.y, g[i].dc0.z);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "sbVolumeSolve outputs bitwise-identical (mod +-0)");
}

}  // namespace
