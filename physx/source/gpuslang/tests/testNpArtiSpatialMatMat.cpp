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

// Verification of npArtiSpatialMatMat.slang (smMulMat): Dy::SpatialMatrix
// operator*(const SpatialMatrix&) (DyFeatherstoneArticulationUtils.h L328-338),
// the articulated-inertia composition, against an exact CPU transcription.
// Outputs bitwise modulo the documented +-0 (eqFZ).

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

struct MmIn
{
    F4 aTlC0, aTlC1, aTlC2, aTrC0, aTrC1, aTrC2, aBlC0, aBlC1, aBlC2;
    F4 bTlC0, bTlC1, bTlC2, bTrC0, bTrC1, bTrC2, bBlC0, bBlC1, bBlC2;
};

struct MmOut
{
    F4 tlC0, tlC1, tlC2, trC0, trC1, trC2, blC0, blC1, blC2;
};

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b) || (a == 0.f && b == 0.f); }

// column-major 3x3 as float m[3][3] (m[col][row])
struct M3 { float m[3][3]; };

static M3 fromF4(const F4& c0, const F4& c1, const F4& c2)
{
    M3 r;
    r.m[0][0]=c0.x; r.m[0][1]=c0.y; r.m[0][2]=c0.z;
    r.m[1][0]=c1.x; r.m[1][1]=c1.y; r.m[1][2]=c1.z;
    r.m[2][0]=c2.x; r.m[2][1]=c2.y; r.m[2][2]=c2.z;
    return r;
}
// transform: col0*v.x + col1*v.y + col2*v.z, per row
static void mtx(const M3& a, const float v[3], float out[3])
{
    for (int r = 0; r < 3; ++r) out[r] = a.m[0][r]*v[0] + a.m[1][r]*v[1] + a.m[2][r]*v[2];
}
static M3 matmul(const M3& a, const M3& b)  // a * b = (a.transform(b.col_i))
{
    M3 r;
    for (int c = 0; c < 3; ++c) { const float col[3]={b.m[c][0],b.m[c][1],b.m[c][2]}; float o[3]; mtx(a,col,o); r.m[c][0]=o[0]; r.m[c][1]=o[1]; r.m[c][2]=o[2]; }
    return r;
}
static M3 transpose3(const M3& a) { M3 r; for(int c=0;c<3;++c) for(int rr=0;rr<3;++rr) r.m[c][rr]=a.m[rr][c]; return r; }
static M3 add3(const M3& a, const M3& b) { M3 r; for(int c=0;c<3;++c) for(int rr=0;rr<3;++rr) r.m[c][rr]=a.m[c][rr]+b.m[c][rr]; return r; }

static void storeF4(const M3& m, F4& c0, F4& c1, F4& c2)
{
    c0 = F4{ m.m[0][0], m.m[0][1], m.m[0][2], 0.f };
    c1 = F4{ m.m[1][0], m.m[1][1], m.m[1][2], 0.f };
    c2 = F4{ m.m[2][0], m.m[2][1], m.m[2][2], 0.f };
}

// exact transcription of SpatialMatrix::operator*(SpatialMatrix)
static void refSmMulMat(const MmIn& s, MmOut& o)
{
    const M3 aTl = fromF4(s.aTlC0, s.aTlC1, s.aTlC2);
    const M3 aTr = fromF4(s.aTrC0, s.aTrC1, s.aTrC2);
    const M3 aBl = fromF4(s.aBlC0, s.aBlC1, s.aBlC2);
    const M3 bTl = fromF4(s.bTlC0, s.bTlC1, s.bTlC2);
    const M3 bTr = fromF4(s.bTrC0, s.bTrC1, s.bTrC2);
    const M3 bBl = fromF4(s.bBlC0, s.bBlC1, s.bBlC2);

    const M3 sBottomRight = transpose3(bTl);
    const M3 bottomRight = transpose3(aTl);

    const M3 ntl = add3(matmul(aTl, bTl), matmul(aTr, bBl));
    const M3 ntr = add3(matmul(aTl, bTr), matmul(aTr, sBottomRight));
    const M3 nbl = add3(matmul(aBl, bTl), matmul(bottomRight, bBl));

    storeF4(ntl, o.tlC0, o.tlC1, o.tlC2);
    storeF4(ntr, o.trC0, o.trC1, o.trC2);
    storeF4(nbl, o.blC0, o.blC1, o.blC2);
}

PXG_TEST(npArtiSpatialMatMat)
{
    std::mt19937 rng(196001);
    const uint32_t N = 3000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto rv = [&](float a, float b) { return F4{ rf(a, b), rf(a, b), rf(a, b), 0.f }; };

    std::vector<MmIn> in(N);
    std::vector<MmOut> ref(N);
    size_t cSymATl = 0, cSymBTl = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        MmIn s{};
        F4* p = &s.aTlC0;
        for (int b = 0; b < 18; ++b) p[b] = rv(-2, 2);

        // symmetric-topLeft stripes so transpose is not the only distinguisher,
        // and the asymmetric majority genuinely exercises the two transposes.
        const bool symA = (i % 13 == 4);
        const bool symB = (i % 13 == 9);
        if (symA) { ++cSymATl; s.aTlC1.x = s.aTlC0.y; s.aTlC2.x = s.aTlC0.z; s.aTlC2.y = s.aTlC1.z; }
        if (symB) { ++cSymBTl; s.bTlC1.x = s.bTlC0.y; s.bTlC2.x = s.bTlC0.z; s.bTlC2.y = s.bTlC1.z; }

        in[i] = s;
        refSmMulMat(s, ref[i]);
    }

    std::printf("[ info ] smm cov: symATl %zu symBTl %zu\n", cSymATl, cSymBTl);
    PXG_CHECK(cSymATl > 120);
    PXG_CHECK(cSymBTl > 120);

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<MmOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npArtiSpatialMatMat.spv", "smMulMat",
             { (uint32_t)sizeof(MmIn), (uint32_t)sizeof(MmOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<MmOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* rfp = &ref[i].tlC0.x;
        const float* qfp = &g[i].tlC0.x;
        const size_t nf = sizeof(MmOut) / 4;
        bool ok = true;
        for (size_t x = 0; x < nf && ok; ++x) ok = eqFZ(rfp[x], qfp[x]);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u tlC0R (%g %g %g) G (%g %g %g)\n", i, ref[i].tlC0.x,
                            ref[i].tlC0.y, ref[i].tlC0.z, g[i].tlC0.x, g[i].tlC0.y, g[i].tlC0.z);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "smMulMat outputs bitwise-identical (mod +-0)");
}

}  // namespace
