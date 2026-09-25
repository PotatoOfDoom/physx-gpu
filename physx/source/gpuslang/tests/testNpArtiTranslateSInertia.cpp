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

// Verification of npArtiTranslateSInertia.slang (transSInertia): the spatial
// inertia translate translateInertia(sTod, SpatialMatrix) (forwardDynamic2.cu
// L1546-1558), against an exact CPU transcription.  Outputs bitwise mod +-0.

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

struct TsIn { F4 sC0, sC1, sC2, tlC0, tlC1, tlC2, trC0, trC1, trC2, blC0, blC1, blC2; };
struct TsOut { F4 tlC0, tlC1, tlC2, trC0, trC1, trC2, blC0, blC1, blC2; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed

struct M3 { float m[3][3]; };  // m[col][row]

static M3 fromF4(const F4& c0, const F4& c1, const F4& c2)
{
    M3 r;
    r.m[0][0]=c0.x; r.m[0][1]=c0.y; r.m[0][2]=c0.z;
    r.m[1][0]=c1.x; r.m[1][1]=c1.y; r.m[1][2]=c1.z;
    r.m[2][0]=c2.x; r.m[2][1]=c2.y; r.m[2][2]=c2.z;
    return r;
}
static void mtx(const M3& a, const float v[3], float out[3])
{
    for (int r = 0; r < 3; ++r) out[r] = a.m[0][r]*v[0] + a.m[1][r]*v[1] + a.m[2][r]*v[2];
}
static M3 matmul(const M3& a, const M3& b)
{
    M3 r;
    for (int c = 0; c < 3; ++c) { const float col[3]={b.m[c][0],b.m[c][1],b.m[c][2]}; float o[3]; mtx(a,col,o); r.m[c][0]=o[0]; r.m[c][1]=o[1]; r.m[c][2]=o[2]; }
    return r;
}
static M3 transpose3(const M3& a) { M3 r; for(int c=0;c<3;++c) for(int rr=0;rr<3;++rr) r.m[c][rr]=a.m[rr][c]; return r; }
static M3 scale3(const M3& a, float s) { M3 r; for(int c=0;c<3;++c) for(int rr=0;rr<3;++rr) r.m[c][rr]=a.m[c][rr]*s; return r; }
static M3 add3(const M3& a, const M3& b) { M3 r; for(int c=0;c<3;++c) for(int rr=0;rr<3;++rr) r.m[c][rr]=a.m[c][rr]+b.m[c][rr]; return r; }
static void storeF4(const M3& m, F4& c0, F4& c1, F4& c2)
{
    c0 = F4{ m.m[0][0], m.m[0][1], m.m[0][2], 0.f };
    c1 = F4{ m.m[1][0], m.m[1][1], m.m[1][2], 0.f };
    c2 = F4{ m.m[2][0], m.m[2][1], m.m[2][2], 0.f };
}

// exact transcription of translateInertia(sTod, SpatialMatrix)
static void refTransSInertia(const TsIn& s, TsOut& o)
{
    const M3 sTod = fromF4(s.sC0, s.sC1, s.sC2);
    const M3 TL = fromF4(s.tlC0, s.tlC1, s.tlC2);
    const M3 TR = fromF4(s.trC0, s.trC1, s.trC2);
    const M3 BL = fromF4(s.blC0, s.blC1, s.blC2);

    const M3 dTos = transpose3(sTod);
    const M3 BR = transpose3(TL);

    const M3 bl = add3(matmul(sTod, TL), BL);
    const M3 br = add3(matmul(sTod, TR), BR);

    const M3 newTL = add3(TL, matmul(TR, dTos));
    const M3 newBL0 = add3(bl, matmul(br, dTos));
    const M3 newBL = scale3(add3(newBL0, transpose3(newBL0)), 0.5f);

    storeF4(newTL, o.tlC0, o.tlC1, o.tlC2);
    storeF4(TR, o.trC0, o.trC1, o.trC2);
    storeF4(newBL, o.blC0, o.blC1, o.blC2);
}

PXG_TEST(npArtiTranslateSInertia)
{
    std::mt19937 rng(199001);
    const uint32_t N = 3000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto rv = [&](float a, float b) { return F4{ rf(a, b), rf(a, b), rf(a, b), 0.f }; };

    std::vector<TsIn> in(N);
    std::vector<TsOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        TsIn s{};
        F4* p = &s.sC0;
        for (int b = 0; b < 12; ++b) p[b] = rv(-2, 2);
        in[i] = s;
        refTransSInertia(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<TsOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npArtiTranslateSInertia.spv", "transSInertia",
             { (uint32_t)sizeof(TsIn), (uint32_t)sizeof(TsOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<TsOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* rfp = &ref[i].tlC0.x;
        const float* qfp = &g[i].tlC0.x;
        const size_t nf = sizeof(TsOut) / 4;
        bool ok = true;
        for (size_t x = 0; x < nf && ok; ++x) ok = eqFZ(rfp[x], qfp[x]);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u blC0R (%g %g %g) G (%g %g %g)\n", i, ref[i].blC0.x, ref[i].blC0.y,
                            ref[i].blC0.z, g[i].blC0.x, g[i].blC0.y, g[i].blC0.z);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "transSInertia outputs bitwise-identical (mod +-0)");
}

}  // namespace
