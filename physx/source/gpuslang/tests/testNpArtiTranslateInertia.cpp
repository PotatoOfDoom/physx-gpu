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

// Verification of npArtiTranslateInertia.slang (transInertia): the parallel-axis
// inertia translate translateInertia(inertia, mass, t) (forwardDynamic2.cu
// L1569-1577), against an exact CPU transcription.  Outputs bitwise mod +-0.

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

struct TiIn { F4 iC0, iC1, iC2, tMass; };
struct TiOut { F4 c0, c1, c2; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b) || (a == 0.f && b == 0.f); }

struct M3 { float m[3][3]; };  // m[col][row], column-major

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

// exact transcription of translateInertia(inertia, mass, t)
static void refTransInertia(const TiIn& in, TiOut& o)
{
    const M3 inertia = fromF4(in.iC0, in.iC1, in.iC2);
    const float tx = in.tMass.x, ty = in.tMass.y, tz = in.tMass.z, mass = in.tMass.w;

    M3 s;
    s.m[0][0]=0.f;  s.m[0][1]=tz;  s.m[0][2]=-ty;
    s.m[1][0]=-tz;  s.m[1][1]=0.f; s.m[1][2]=tx;
    s.m[2][0]=ty;   s.m[2][1]=-tx; s.m[2][2]=0.f;

    const M3 sT = transpose3(s);
    const M3 sTs = matmul(sT, s);
    const M3 scaled = scale3(sTs, mass);
    const M3 res = add3(scaled, inertia);

    o.c0 = F4{ res.m[0][0], res.m[0][1], res.m[0][2], 0.f };
    o.c1 = F4{ res.m[1][0], res.m[1][1], res.m[1][2], 0.f };
    o.c2 = F4{ res.m[2][0], res.m[2][1], res.m[2][2], 0.f };
}

PXG_TEST(npArtiTranslateInertia)
{
    std::mt19937 rng(198001);
    const uint32_t N = 3000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto rv = [&](float a, float b) { return F4{ rf(a, b), rf(a, b), rf(a, b), 0.f }; };

    std::vector<TiIn> in(N);
    std::vector<TiOut> ref(N);
    size_t cZeroT = 0, cZeroMass = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        TiIn s{};
        s.iC0 = rv(-2, 2); s.iC1 = rv(-2, 2); s.iC2 = rv(-2, 2);
        const bool zeroT = (i % 11 == 3);
        const bool zeroMass = (i % 13 == 7);
        if (zeroT) ++cZeroT;
        if (zeroMass) ++cZeroMass;
        s.tMass = F4{ zeroT ? 0.f : rf(-3, 3), zeroT ? 0.f : rf(-3, 3), zeroT ? 0.f : rf(-3, 3),
                      zeroMass ? 0.f : rf(0.1f, 5.f) };
        in[i] = s;
        refTransInertia(s, ref[i]);
    }

    std::printf("[ info ] ti cov: zeroT %zu zeroMass %zu\n", cZeroT, cZeroMass);
    PXG_CHECK(cZeroT > 150);
    PXG_CHECK(cZeroMass > 150);

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<TiOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npArtiTranslateInertia.spv", "transInertia",
             { (uint32_t)sizeof(TiIn), (uint32_t)sizeof(TiOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<TiOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* rfp = &ref[i].c0.x;
        const float* qfp = &g[i].c0.x;
        const size_t nf = sizeof(TiOut) / 4;
        bool ok = true;
        for (size_t x = 0; x < nf && ok; ++x) ok = eqFZ(rfp[x], qfp[x]);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u c0R (%g %g %g) G (%g %g %g)\n", i, ref[i].c0.x, ref[i].c0.y,
                            ref[i].c0.z, g[i].c0.x, g[i].c0.y, g[i].c0.z);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "transInertia outputs bitwise-identical (mod +-0)");
}

}  // namespace
