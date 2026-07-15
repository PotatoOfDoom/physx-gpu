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

// Verification of npArtiPropagate.slang (apImpulse): the Featherstone impulse
// propagation propagateImpulseW_everythingPreLoaded (articulationDynamic.cuh
// L167-185), against an exact CPU transcription.  Exercises the spatial
// innerProduct (motion/force cross pairing) and translateSpatialVector.  All
// outputs bitwise modulo the documented +-0 class (eqFZ).

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

struct ApIn
{
    F4 Ztop, Zbot, c2p;
    U4 cfg;
    F4 mmTop[3], mmBot[3], idTop[3], idBot[3];
    F4 qstZin;
};

struct ApOut
{
    F4 propTop, propBot, qstZout;
};

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b) || (a == 0.f && b == 0.f); }

static float dot3(const float a[3], const float b[3]) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }

// exact transcription of propagateImpulseW_everythingPreLoaded
// (articulationDynamic.cuh L167-185)
static void refApImpulse(const ApIn& s, ApOut& o)
{
    const uint32_t dofCount = s.cfg.x < 3u ? s.cfg.x : 3u;

    const float Zt[3] = { s.Ztop.x, s.Ztop.y, s.Ztop.z };
    const float Zb[3] = { s.Zbot.x, s.Zbot.y, s.Zbot.z };

    float tempT[3] = { Zt[0], Zt[1], Zt[2] };  // temp = Z
    float tempB[3] = { Zb[0], Zb[1], Zb[2] };
    float qz[3] = { s.qstZin.x, s.qstZin.y, s.qstZin.z };

    for (uint32_t ind = 0; ind < 3; ++ind)
    {
        if (ind < dofCount)
        {
            const float mmB[3] = { s.mmBot[ind].x, s.mmBot[ind].y, s.mmBot[ind].z };
            const float mmT[3] = { s.mmTop[ind].x, s.mmTop[ind].y, s.mmTop[ind].z };
            // stZ = -motionMatrix[ind].innerProduct(Z)
            //     = -(motionMatrix.bottom . Z.top + motionMatrix.top . Z.bottom)
            const float stZ = -(dot3(mmB, Zt) + dot3(mmT, Zb));
            qz[ind] += stZ;
            for (int k = 0; k < 3; ++k)
            {
                tempT[k] += (&s.idTop[ind].x)[k] * stZ;  // temp += isInvD[ind] * stZ
                tempB[k] += (&s.idBot[ind].x)[k] * stZ;
            }
        }
    }

    // translateSpatialVector(childToParent, temp): (top, bottom + c2p x top)
    const float c[3] = { s.c2p.x, s.c2p.y, s.c2p.z };
    const float crx[3] = { c[1]*tempT[2] - c[2]*tempT[1],
                           c[2]*tempT[0] - c[0]*tempT[2],
                           c[0]*tempT[1] - c[1]*tempT[0] };
    o.propTop = F4{ tempT[0], tempT[1], tempT[2], 0.f };
    o.propBot = F4{ tempB[0] + crx[0], tempB[1] + crx[1], tempB[2] + crx[2], 0.f };
    o.qstZout = F4{ qz[0], qz[1], qz[2], 0.f };
}

PXG_TEST(npArtiPropagate)
{
    std::mt19937 rng(191001);
    const uint32_t N = 3000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto rv = [&](float a, float b) { return F4{ rf(a, b), rf(a, b), rf(a, b), 0.f }; };

    std::vector<ApIn> in(N);
    std::vector<ApOut> ref(N);
    size_t cDof[4] = { 0, 0, 0, 0 }, cZeroC2p = 0, cQnz = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        ApIn s{};
        const uint32_t dof = rng() % 4;  // 0..3 (0 => only translate applies)
        ++cDof[dof];
        s.cfg = U4{ dof, 0, 0, 0 };

        s.Ztop = rv(-3, 3);
        s.Zbot = rv(-3, 3);
        // childToParent zero sometimes (translate becomes identity)
        const bool zeroC2p = (i % 9 == 4);
        if (zeroC2p) ++cZeroC2p;
        s.c2p = zeroC2p ? F4{ 0, 0, 0, 0 } : rv(-2, 2);

        for (uint32_t d = 0; d < 3; ++d)
        {
            s.mmTop[d] = rv(-1.5f, 1.5f);
            s.mmBot[d] = rv(-1.5f, 1.5f);
            s.idTop[d] = rv(-1.5f, 1.5f);
            s.idBot[d] = rv(-1.5f, 1.5f);
        }
        // qstZ initial nonzero so the accumulation (+=) is observable
        const bool qnz = (i % 3 != 0);
        if (qnz && dof) ++cQnz;
        s.qstZin = qnz ? F4{ rf(-5, 5), rf(-5, 5), rf(-5, 5), 0.f } : F4{ 0, 0, 0, 0 };

        in[i] = s;
        refApImpulse(s, ref[i]);
    }

    std::printf("[ info ] prop cov: dof0 %zu dof1 %zu dof2 %zu dof3 %zu zeroC2p %zu qNonzero %zu\n",
                cDof[0], cDof[1], cDof[2], cDof[3], cZeroC2p, cQnz);
    PXG_CHECK(cDof[0] > 400);
    PXG_CHECK(cDof[1] > 400);
    PXG_CHECK(cDof[2] > 400);
    PXG_CHECK(cDof[3] > 400);
    PXG_CHECK(cZeroC2p > 200);
    PXG_CHECK(cQnz > 800);

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<ApOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npArtiPropagate.spv", "apImpulse",
             { (uint32_t)sizeof(ApIn), (uint32_t)sizeof(ApOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<ApOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* rfp = &ref[i].propTop.x;
        const float* qfp = &g[i].propTop.x;
        const size_t nf = sizeof(ApOut) / 4;
        bool ok = true;
        for (size_t x = 0; x < nf && ok; ++x) ok = eqFZ(rfp[x], qfp[x]);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u dof %u propTopR (%g %g %g) G (%g %g %g)\n", i, in[i].cfg.x,
                            ref[i].propTop.x, ref[i].propTop.y, ref[i].propTop.z, g[i].propTop.x,
                            g[i].propTop.y, g[i].propTop.z);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "apImpulse outputs bitwise-identical (mod +-0)");
}

}  // namespace
