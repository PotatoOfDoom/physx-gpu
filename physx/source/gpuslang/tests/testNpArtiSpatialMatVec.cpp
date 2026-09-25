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

// Verification of npArtiSpatialMatVec.slang (smMulVec): Dy::SpatialMatrix
// operator*(UnAlignedSpatialVector) (DyFeatherstoneArticulationUtils.h
// L276-282), the spatial-inertia matrix times a spatial vector, against an
// exact CPU transcription.  Outputs bitwise modulo the documented +-0 (eqFZ).

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

struct SmIn
{
    F4 tlC0, tlC1, tlC2, trC0, trC1, trC2, blC0, blC1, blC2, sTop, sBot;
};

struct SmOut
{
    F4 outTop, outBot;
};

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed

static float dot3(const float a[3], const float b[3]) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
// PxMat33::transform: column0*v.x + column1*v.y + column2*v.z
static void mtx(const F4& c0, const F4& c1, const F4& c2, const float v[3], float out[3])
{
    out[0] = c0.x * v[0] + c1.x * v[1] + c2.x * v[2];
    out[1] = c0.y * v[0] + c1.y * v[1] + c2.y * v[2];
    out[2] = c0.z * v[0] + c1.z * v[1] + c2.z * v[2];
}

// exact transcription of SpatialMatrix::operator*(UnAlignedSpatialVector)
static void refSmMulVec(const SmIn& s, SmOut& o)
{
    const float st[3] = { s.sTop.x, s.sTop.y, s.sTop.z };
    const float sb[3] = { s.sBot.x, s.sBot.y, s.sBot.z };

    float tl_st[3], tr_sb[3], bl_st[3];
    mtx(s.tlC0, s.tlC1, s.tlC2, st, tl_st);   // topLeft * s.top
    mtx(s.trC0, s.trC1, s.trC2, sb, tr_sb);   // topRight * s.bottom
    mtx(s.blC0, s.blC1, s.blC2, st, bl_st);   // bottomLeft * s.top

    // topLeft.transformTranspose(s.bottom) = (col0.dot(sb), col1.dot(sb), col2.dot(sb))
    const float c0[3] = { s.tlC0.x, s.tlC0.y, s.tlC0.z };
    const float c1[3] = { s.tlC1.x, s.tlC1.y, s.tlC1.z };
    const float c2[3] = { s.tlC2.x, s.tlC2.y, s.tlC2.z };
    const float tlT_sb[3] = { dot3(c0, sb), dot3(c1, sb), dot3(c2, sb) };

    o.outTop = F4{ tl_st[0] + tr_sb[0], tl_st[1] + tr_sb[1], tl_st[2] + tr_sb[2], 0.f };
    o.outBot = F4{ bl_st[0] + tlT_sb[0], bl_st[1] + tlT_sb[1], bl_st[2] + tlT_sb[2], 0.f };
}

PXG_TEST(npArtiSpatialMatVec)
{
    std::mt19937 rng(194001);
    const uint32_t N = 3000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto rv = [&](float a, float b) { return F4{ rf(a, b), rf(a, b), rf(a, b), 0.f }; };

    std::vector<SmIn> in(N);
    std::vector<SmOut> ref(N);
    size_t cSymTL = 0, cZeroBot = 0, cZeroTop = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        SmIn s{};
        s.tlC0 = rv(-2, 2); s.tlC1 = rv(-2, 2); s.tlC2 = rv(-2, 2);
        s.trC0 = rv(-2, 2); s.trC1 = rv(-2, 2); s.trC2 = rv(-2, 2);
        s.blC0 = rv(-2, 2); s.blC1 = rv(-2, 2); s.blC2 = rv(-2, 2);
        // occasionally make topLeft symmetric (transformTranspose == transform)
        // so the transpose is *not* the only thing distinguishing lanes, and
        // the complementary asymmetric majority genuinely exercises it.
        const bool symTL = (i % 11 == 3);
        if (symTL)
        {
            ++cSymTL;
            s.tlC1.x = s.tlC0.y; s.tlC2.x = s.tlC0.z; s.tlC2.y = s.tlC1.z;
        }
        // zero-bottom stripe isolates the topLeft*top / bottomLeft*top terms;
        // zero-top isolates topRight*bottom / topLeft^T*bottom.
        const bool zeroBot = (i % 7 == 2);
        const bool zeroTop = (i % 7 == 5);
        if (zeroBot) ++cZeroBot;
        if (zeroTop) ++cZeroTop;
        s.sTop = zeroTop ? F4{ 0, 0, 0, 0 } : rv(-3, 3);
        s.sBot = zeroBot ? F4{ 0, 0, 0, 0 } : rv(-3, 3);

        in[i] = s;
        refSmMulVec(s, ref[i]);
    }

    std::printf("[ info ] smv cov: symTL %zu zeroBot %zu zeroTop %zu\n", cSymTL, cZeroBot, cZeroTop);
    PXG_CHECK(cSymTL > 150);
    PXG_CHECK(cZeroBot > 300);
    PXG_CHECK(cZeroTop > 300);

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<SmOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npArtiSpatialMatVec.spv", "smMulVec",
             { (uint32_t)sizeof(SmIn), (uint32_t)sizeof(SmOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<SmOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* rfp = &ref[i].outTop.x;
        const float* qfp = &g[i].outTop.x;
        const size_t nf = sizeof(SmOut) / 4;
        bool ok = true;
        for (size_t x = 0; x < nf && ok; ++x) ok = eqFZ(rfp[x], qfp[x]);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u topR (%g %g %g) topG (%g %g %g)\n", i, ref[i].outTop.x,
                            ref[i].outTop.y, ref[i].outTop.z, g[i].outTop.x, g[i].outTop.y,
                            g[i].outTop.z);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "smMulVec outputs bitwise-identical (mod +-0)");
}

}  // namespace
