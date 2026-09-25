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

// Verification of npSbDefGrad.slang (sbDefGrad): the soft-body
// deformation-gradient calculateDeformationGradient and getSoftBodyInvMass
// (softBody.cuh L38-59), against an exact CPU transcription.  Outputs bitwise
// modulo the documented +-0 (eqFZ).

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
struct SbIn  { F4 u1, u2, u3, qinvC0, qinvC1, qinvC2, rtC0, rtC1, rtC2, bary, misc; };
struct SbOut { F4 fC0, fC1, fC2, scalars; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed

struct M3 { float m[3][3]; };  // m[col][row]
static M3 fromF4(const F4& c0, const F4& c1, const F4& c2)
{
    M3 r; r.m[0][0]=c0.x; r.m[0][1]=c0.y; r.m[0][2]=c0.z;
    r.m[1][0]=c1.x; r.m[1][1]=c1.y; r.m[1][2]=c1.z;
    r.m[2][0]=c2.x; r.m[2][1]=c2.y; r.m[2][2]=c2.z; return r;
}
static void mtx(const M3& a, const float v[3], float o[3])
{ for (int r=0;r<3;++r) o[r]=a.m[0][r]*v[0]+a.m[1][r]*v[1]+a.m[2][r]*v[2]; }
static M3 matmul(const M3& a, const M3& b)
{ M3 r; for(int c=0;c<3;++c){ const float col[3]={b.m[c][0],b.m[c][1],b.m[c][2]}; float o[3]; mtx(a,col,o); r.m[c][0]=o[0]; r.m[c][1]=o[1]; r.m[c][2]=o[2]; } return r; }
static void storeF4(const M3& m, F4& c0, F4& c1, F4& c2)
{ c0=F4{m.m[0][0],m.m[0][1],m.m[0][2],0.f}; c1=F4{m.m[1][0],m.m[1][1],m.m[1][2],0.f}; c2=F4{m.m[2][0],m.m[2][1],m.m[2][2],0.f}; }

static void refSb(const SbIn& s, SbOut& o)
{
    M3 P; P.m[0][0]=s.u1.x; P.m[0][1]=s.u1.y; P.m[0][2]=s.u1.z;
    P.m[1][0]=s.u2.x; P.m[1][1]=s.u2.y; P.m[1][2]=s.u2.z;
    P.m[2][0]=s.u3.x; P.m[2][1]=s.u3.y; P.m[2][2]=s.u3.z;
    const M3 Qinv = fromF4(s.qinvC0, s.qinvC1, s.qinvC2);
    const M3 RT = fromF4(s.rtC0, s.rtC1, s.rtC2);
    M3 F = matmul(P, Qinv);
    F = matmul(RT, F);

    const F4 b = s.bary;
    const float sumSq = b.x*b.x + b.y*b.y + b.z*b.z + b.w*b.w;
    const float invMass = s.misc.x * std::sqrt(sumSq);

    storeF4(F, o.fC0, o.fC1, o.fC2);
    o.scalars = F4{ invMass, 0.f, 0.f, 0.f };
}

PXG_TEST(npSbDefGrad)
{
    std::mt19937 rng(209001);
    const uint32_t N = 3000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto rv = [&](float a, float b) { return F4{ rf(a, b), rf(a, b), rf(a, b), 0.f }; };

    std::vector<SbIn> in(N);
    std::vector<SbOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        SbIn s{};
        s.u1 = rv(-2, 2); s.u2 = rv(-2, 2); s.u3 = rv(-2, 2);
        s.qinvC0 = rv(-2, 2); s.qinvC1 = rv(-2, 2); s.qinvC2 = rv(-2, 2);
        s.rtC0 = rv(-2, 2); s.rtC1 = rv(-2, 2); s.rtC2 = rv(-2, 2);
        s.bary = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), rf(-1, 1) };
        s.misc = F4{ rf(0.f, 5.f), 0.f, 0.f, 0.f };
        in[i] = s;
        refSb(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<SbOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npSbDefGrad.spv", "sbDefGrad",
             { (uint32_t)sizeof(SbIn), (uint32_t)sizeof(SbOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<SbOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* rfp = &ref[i].fC0.x;
        const float* qfp = &g[i].fC0.x;
        const size_t nf = sizeof(SbOut) / 4;
        bool ok = true;
        for (size_t x = 0; x < nf && ok; ++x) ok = eqFZ(rfp[x], qfp[x]);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u fC0R (%g %g %g) invM %g G (%g %g %g) invM %g\n", i, ref[i].fC0.x,
                            ref[i].fC0.y, ref[i].fC0.z, ref[i].scalars.x, g[i].fC0.x, g[i].fC0.y, g[i].fC0.z, g[i].scalars.x);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "sbDefGrad outputs bitwise-identical (mod +-0)");
}

}  // namespace
