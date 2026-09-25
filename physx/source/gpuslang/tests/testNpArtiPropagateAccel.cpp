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

// Verification of npArtiPropagateAccel.slang (apAccel): the Featherstone
// acceleration propagation propagateAccelerationW_everythingPreLoaded
// (articulationDynamic.cuh L206-242), against an exact CPU transcription.
// Exercises the negated-offset translate, the spatial innerProduct, and the
// two-pass tJointDelta / jDelta structure.  Outputs bitwise modulo +-0 (eqFZ).

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

struct AaIn
{
    F4 paTop, paBot, p2c;
    U4 cfg;
    F4 mmTop[3], mmBot[3], isTop[3], isBot[3], invS[3];
    F4 qmstz;
};

struct AaOut
{
    F4 pdTop, pdBot;
};

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed

static float dot3(const float a[3], const float b[3]) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }

// exact transcription of propagateAccelerationW_everythingPreLoaded
// (articulationDynamic.cuh L206-242)
static void refApAccel(const AaIn& s, AaOut& o)
{
    const uint32_t dofCount = s.cfg.x < 3u ? s.cfg.x : 3u;

    // pDeltaV = translateSpatialVector(-parentToChild, parentAccel)
    const float paT[3] = { s.paTop.x, s.paTop.y, s.paTop.z };
    const float negP2c[3] = { -s.p2c.x, -s.p2c.y, -s.p2c.z };
    const float crx[3] = { negP2c[1]*paT[2] - negP2c[2]*paT[1],
                           negP2c[2]*paT[0] - negP2c[0]*paT[2],
                           negP2c[0]*paT[1] - negP2c[1]*paT[0] };
    float pdT[3] = { paT[0], paT[1], paT[2] };
    float pdB[3] = { s.paBot.x + crx[0], s.paBot.y + crx[1], s.paBot.z + crx[2] };

    const float qm[3] = { s.qmstz.x, s.qmstz.y, s.qmstz.z };

    // pass 1: tJointDelta[i] = QMinusSTZ[i] - IsW[i].innerProduct(pDeltaV)
    float tjd[3] = { 0.f, 0.f, 0.f };
    for (uint32_t ind = 0; ind < 3; ++ind)
    {
        if (ind < dofCount)
        {
            const float isB[3] = { s.isBot[ind].x, s.isBot[ind].y, s.isBot[ind].z };
            const float isT[3] = { s.isTop[ind].x, s.isTop[ind].y, s.isTop[ind].z };
            const float temp = dot3(isB, pdT) + dot3(isT, pdB);  // innerProduct duality
            tjd[ind] = qm[ind] - temp;
        }
    }

    // pass 2: jDelta = invStISW[i] . tJointDelta; pDeltaV += motionMatrix[i]*jDelta
    for (uint32_t ind = 0; ind < 3; ++ind)
    {
        if (ind < dofCount)
        {
            const F4 iS = s.invS[ind];
            const float jDelta = iS.x * tjd[0] + iS.y * tjd[1] + iS.z * tjd[2];
            for (int k = 0; k < 3; ++k)
            {
                pdT[k] += (&s.mmTop[ind].x)[k] * jDelta;
                pdB[k] += (&s.mmBot[ind].x)[k] * jDelta;
            }
        }
    }

    o.pdTop = F4{ pdT[0], pdT[1], pdT[2], 0.f };
    o.pdBot = F4{ pdB[0], pdB[1], pdB[2], 0.f };
}

PXG_TEST(npArtiPropagateAccel)
{
    std::mt19937 rng(192001);
    const uint32_t N = 3000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto rv = [&](float a, float b) { return F4{ rf(a, b), rf(a, b), rf(a, b), 0.f }; };

    std::vector<AaIn> in(N);
    std::vector<AaOut> ref(N);
    size_t cDof[4] = { 0, 0, 0, 0 }, cZeroP2c = 0, cQnz = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        AaIn s{};
        const uint32_t dof = rng() % 4;  // 0..3 (0 => translate only)
        ++cDof[dof];
        s.cfg = U4{ dof, 0, 0, 0 };

        s.paTop = rv(-3, 3);
        s.paBot = rv(-3, 3);
        const bool zeroP2c = (i % 9 == 4);
        if (zeroP2c) ++cZeroP2c;
        s.p2c = zeroP2c ? F4{ 0, 0, 0, 0 } : rv(-2, 2);

        for (uint32_t d = 0; d < 3; ++d)
        {
            s.mmTop[d] = rv(-1.5f, 1.5f);
            s.mmBot[d] = rv(-1.5f, 1.5f);
            s.isTop[d] = rv(-1.5f, 1.5f);
            s.isBot[d] = rv(-1.5f, 1.5f);
            s.invS[d] = rv(-1.2f, 1.2f);
        }
        const bool qnz = (i % 3 != 0);
        if (qnz && dof) ++cQnz;
        s.qmstz = qnz ? F4{ rf(-5, 5), rf(-5, 5), rf(-5, 5), 0.f } : F4{ 0, 0, 0, 0 };

        in[i] = s;
        refApAccel(s, ref[i]);
    }

    std::printf("[ info ] paccel cov: dof0 %zu dof1 %zu dof2 %zu dof3 %zu zeroP2c %zu qNonzero %zu\n",
                cDof[0], cDof[1], cDof[2], cDof[3], cZeroP2c, cQnz);
    PXG_CHECK(cDof[0] > 400);
    PXG_CHECK(cDof[1] > 400);
    PXG_CHECK(cDof[2] > 400);
    PXG_CHECK(cDof[3] > 400);
    PXG_CHECK(cZeroP2c > 200);
    PXG_CHECK(cQnz > 800);

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<AaOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npArtiPropagateAccel.spv", "apAccel",
             { (uint32_t)sizeof(AaIn), (uint32_t)sizeof(AaOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<AaOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* rfp = &ref[i].pdTop.x;
        const float* qfp = &g[i].pdTop.x;
        const size_t nf = sizeof(AaOut) / 4;
        bool ok = true;
        for (size_t x = 0; x < nf && ok; ++x) ok = eqFZ(rfp[x], qfp[x]);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u dof %u pdTopR (%g %g %g) G (%g %g %g)\n", i, in[i].cfg.x,
                            ref[i].pdTop.x, ref[i].pdTop.y, ref[i].pdTop.z, g[i].pdTop.x,
                            g[i].pdTop.y, g[i].pdTop.z);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "apAccel outputs bitwise-identical (mod +-0)");
}

}  // namespace
