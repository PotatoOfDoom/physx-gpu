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

// Verification of npArtiPropagateZASph.slang (propZASph): the spherical
// (multi-dof, dofs<=3) branch of computePropagateSpatialInertia_ZA_ZIc
// (forwardDynamic2.cu L1450-1537), against an exact CPU transcription.
// Outputs bitwise modulo the documented +-0 (eqFZ).

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

struct PsIn
{
    F4 sa0Top, sa0Bot, sa1Top, sa1Bot, sa2Top, sa2Bot;
    F4 is0Top, is0Bot, is1Top, is1Bot, is2Top, is2Bot;
    F4 zTop, zBot, zicTop, zicBot, zaTop, zaBot, zaiTop, zaiBot;
    F4 armature, jf, misc;   // misc: dofs, jFnull, 0, 0
};
struct PsOut
{
    F4 invStIsT0, invStIsT1, invStIsT2, qstZ, qstZIcInternal;
    F4 isID0Top, isID0Bot, isID1Top, isID1Bot, isID2Top, isID2Bot;
    F4 zaTop, zaBot, zaiTop, zaiBot;
    F4 tlC0, tlC1, tlC2, trC0, trC1, trC2, blC0, blC1, blC2;
};

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b) || (a == 0.f && b == 0.f); }

static float dot3(const F4& a, const F4& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static F4 cross3(const F4& a, const F4& b)
{
    return F4{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x, 0.f };
}
static float inner(const F4& aTop, const F4& aBot, const F4& bTop, const F4& bBot)
{
    return dot3(aBot, bTop) + dot3(aTop, bBot);
}

// exact transcription of the spherical branch of computePropagateSpatialInertia_ZA_ZIc
static void refPropZASph(const PsIn& s, PsOut& o)
{
    const uint32_t dofs = (uint32_t)s.misc.x;
    const bool jFnull = f2u(s.misc.y) == 1u;

    const F4 saT[3] = { s.sa0Top, s.sa1Top, s.sa2Top };
    const F4 saB[3] = { s.sa0Bot, s.sa1Bot, s.sa2Bot };
    const F4 isT[3] = { s.is0Top, s.is1Top, s.is2Top };
    const F4 isB[3] = { s.is0Bot, s.is1Bot, s.is2Bot };
    const float arm[3] = { s.armature.x, s.armature.y, s.armature.z };
    const float jF[3] = { s.jf.x, s.jf.y, s.jf.z };

    // M[col][row], identity init
    float M[3][3];
    for (int c = 0; c < 3; ++c) for (int r = 0; r < 3; ++r) M[c][r] = (c == r) ? 1.f : 0.f;

    float qstZG[3] = { 0, 0, 0 }, qstZIcIntG[3] = { 0, 0, 0 };
    for (uint32_t ind2 = 0; ind2 < 3; ++ind2)
    {
        if (ind2 < dofs)
        {
            const F4 mT = saT[ind2], mB = saB[ind2];
            for (uint32_t ind = 0; ind < 3; ++ind)
                if (ind < dofs) M[ind][ind2] = inner(mT, mB, isT[ind], isB[ind]);
            M[ind2][ind2] += arm[ind2];
            const float stZ = inner(mT, mB, s.zTop, s.zBot);
            const float stZInt = inner(mT, mB, s.zicTop, s.zicBot);
            qstZG[ind2] = -stZ;
            qstZIcIntG[ind2] = (jFnull ? 0.f : jF[ind2]) - stZInt;
        }
    }

    // getInverse
    const F4 col0 = F4{ M[0][0], M[0][1], M[0][2], 0.f };
    const F4 col1 = F4{ M[1][0], M[1][1], M[1][2], 0.f };
    const F4 col2 = F4{ M[2][0], M[2][1], M[2][2], 0.f };
    const float det = dot3(col0, cross3(col1, col2));

    float I[3][3];
    if (det != 0.f)
    {
        const float invDet = 1.f / det;
        I[0][0] = invDet *  (M[1][1] * M[2][2] - M[2][1] * M[1][2]);
        I[0][1] = invDet * -(M[0][1] * M[2][2] - M[2][1] * M[0][2]);
        I[0][2] = invDet *  (M[0][1] * M[1][2] - M[0][2] * M[1][1]);
        I[1][0] = invDet * -(M[1][0] * M[2][2] - M[1][2] * M[2][0]);
        I[1][1] = invDet *  (M[0][0] * M[2][2] - M[0][2] * M[2][0]);
        I[1][2] = invDet * -(M[0][0] * M[1][2] - M[0][2] * M[1][0]);
        I[2][0] = invDet *  (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
        I[2][1] = invDet * -(M[0][0] * M[2][1] - M[0][1] * M[2][0]);
        I[2][2] = invDet *  (M[0][0] * M[1][1] - M[1][0] * M[0][1]);
    }
    else
    {
        for (int c = 0; c < 3; ++c) for (int r = 0; r < 3; ++r) I[c][r] = (c == r) ? 1.f : 0.f;
    }

    F4 colT[6], colB[6];
    for (int k = 0; k < 6; ++k) { colT[k] = F4{ 0, 0, 0, 0 }; colB[k] = F4{ 0, 0, 0, 0 }; }

    F4 zaTop = s.zaTop, zaBot = s.zaBot, zaiTop = s.zaiTop, zaiBot = s.zaiBot;
    F4 isIDTops[3] = { F4{0,0,0,0}, F4{0,0,0,0}, F4{0,0,0,0} };
    F4 isIDBots[3] = { F4{0,0,0,0}, F4{0,0,0,0}, F4{0,0,0,0} };

    auto axpy = [](F4& acc, const F4& v, float w) { acc.x += v.x * w; acc.y += v.y * w; acc.z += v.z * w; };

    for (uint32_t ind = 0; ind < 3; ++ind)
    {
        if (ind < dofs)
        {
            F4 isIDTop = F4{ 0, 0, 0, 0 }, isIDBot = F4{ 0, 0, 0, 0 };
            for (uint32_t ind2 = 0; ind2 < 3; ++ind2)
            {
                if (ind2 < dofs)
                {
                    const float w = I[ind][ind2];
                    axpy(isIDTop, isT[ind2], w);
                    axpy(isIDBot, isB[ind2], w);
                }
            }
            isIDTops[ind] = isIDTop; isIDBots[ind] = isIDBot;

            const F4 ib = isB[ind], it = isT[ind];
            axpy(colT[0], isIDTop, ib.x); axpy(colB[0], isIDBot, ib.x);
            axpy(colT[1], isIDTop, ib.y); axpy(colB[1], isIDBot, ib.y);
            axpy(colT[2], isIDTop, ib.z); axpy(colB[2], isIDBot, ib.z);
            axpy(colT[3], isIDTop, it.x); axpy(colB[3], isIDBot, it.x);
            axpy(colT[4], isIDTop, it.y); axpy(colB[4], isIDBot, it.y);
            axpy(colT[5], isIDTop, it.z); axpy(colB[5], isIDBot, it.z);

            axpy(zaTop, isIDTop, qstZG[ind]); axpy(zaBot, isIDBot, qstZG[ind]);
            axpy(zaiTop, isIDTop, qstZIcIntG[ind]); axpy(zaiBot, isIDBot, qstZIcIntG[ind]);
        }
    }

    o.invStIsT0 = F4{ I[0][0], I[1][0], I[2][0], 0.f };
    o.invStIsT1 = F4{ I[0][1], I[1][1], I[2][1], 0.f };
    o.invStIsT2 = F4{ I[0][2], I[1][2], I[2][2], 0.f };
    o.qstZ = F4{ qstZG[0], qstZG[1], qstZG[2], 0.f };
    o.qstZIcInternal = F4{ qstZIcIntG[0], qstZIcIntG[1], qstZIcIntG[2], 0.f };
    o.isID0Top = isIDTops[0]; o.isID0Bot = isIDBots[0];
    o.isID1Top = isIDTops[1]; o.isID1Bot = isIDBots[1];
    o.isID2Top = isIDTops[2]; o.isID2Bot = isIDBots[2];
    o.zaTop = zaTop; o.zaBot = zaBot; o.zaiTop = zaiTop; o.zaiBot = zaiBot;
    o.tlC0 = colT[0]; o.tlC1 = colT[1]; o.tlC2 = colT[2];
    o.blC0 = colB[0]; o.blC1 = colB[1]; o.blC2 = colB[2];
    o.trC0 = colT[3]; o.trC1 = colT[4]; o.trC2 = colT[5];
    // clear w lanes for bitwise compare
    F4* p = &o.invStIsT0;
    const size_t nv = sizeof(PsOut) / sizeof(F4);
    for (size_t v = 0; v < nv; ++v) p[v].w = 0.f;
}

PXG_TEST(npArtiPropagateZASph)
{
    std::mt19937 rng(201001);
    const uint32_t N = 3000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto rv = [&](float a, float b) { return F4{ rf(a, b), rf(a, b), rf(a, b), 0.f }; };

    std::vector<PsIn> in(N);
    std::vector<PsOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        PsIn s{};
        s.sa0Top = rv(-2, 2); s.sa0Bot = rv(-2, 2);
        s.sa1Top = rv(-2, 2); s.sa1Bot = rv(-2, 2);
        s.sa2Top = rv(-2, 2); s.sa2Bot = rv(-2, 2);
        s.is0Top = rv(-3, 3); s.is0Bot = rv(-3, 3);
        s.is1Top = rv(-3, 3); s.is1Bot = rv(-3, 3);
        s.is2Top = rv(-3, 3); s.is2Bot = rv(-3, 3);
        s.zTop = rv(-3, 3); s.zBot = rv(-3, 3);
        s.zicTop = rv(-3, 3); s.zicBot = rv(-3, 3);
        s.zaTop = rv(-3, 3); s.zaBot = rv(-3, 3);
        s.zaiTop = rv(-3, 3); s.zaiBot = rv(-3, 3);
        s.armature = F4{ rf(0.f, 2.f), rf(0.f, 2.f), rf(0.f, 2.f), 0.f };
        s.jf = F4{ rf(-4.f, 4.f), rf(-4.f, 4.f), rf(-4.f, 4.f), 0.f };
        const uint32_t dofs = 1u + (i % 3u);          // 1, 2, 3
        const bool jFnull = (i % 4u) == 0u;
        s.misc = F4{ (float)dofs, u2f(jFnull ? 1u : 0u), 0.f, 0.f };
        in[i] = s;
        refPropZASph(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<PsOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npArtiPropagateZASph.spv", "propZASph",
             { (uint32_t)sizeof(PsIn), (uint32_t)sizeof(PsOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<PsOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* rfp = &ref[i].invStIsT0.x;
        const float* qfp = &g[i].invStIsT0.x;
        const size_t nf = sizeof(PsOut) / 4;
        bool ok = true;
        for (size_t x = 0; x < nf && ok; ++x) ok = eqFZ(rfp[x], qfp[x]);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u dofs=%g tlC0R (%g %g %g) G (%g %g %g)\n", i, in[i].misc.x,
                            ref[i].tlC0.x, ref[i].tlC0.y, ref[i].tlC0.z, g[i].tlC0.x, g[i].tlC0.y, g[i].tlC0.z);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "propZASph outputs bitwise-identical (mod +-0)");
}

}  // namespace
