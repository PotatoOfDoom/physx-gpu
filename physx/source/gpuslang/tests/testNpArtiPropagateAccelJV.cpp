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

// Verification of npArtiPropagateAccelJV.slang (apAccelJV):
// propagateAccelerationW_nothingpreloaded_updateJointDofVels (articulationDynamic
// .cuh L329-386) against an exact CPU transcription.  Same accel propagation as
// apAccel (192) plus the joint-velocity writeback.  Outputs bitwise mod +-0.

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

struct JvIn
{
    F4 paTop, paBot, p2c;
    U4 cfg;
    F4 qstz, jvel;
    F4 mmTop[3], mmBot[3], isTop[3], isBot[3], invS[3];
};

struct JvOut
{
    F4 pdTop, pdBot, jvout;
};

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b) || (a == 0.f && b == 0.f); }

static float dot3(const float a[3], const float b[3]) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }

// exact transcription of propagateAccelerationW_nothingpreloaded_updateJointDofVels
static void refApAccelJV(const JvIn& s, JvOut& o)
{
    const uint32_t dofCount = s.cfg.x < 3u ? s.cfg.x : 3u;

    const float paT[3] = { s.paTop.x, s.paTop.y, s.paTop.z };
    const float negP2c[3] = { -s.p2c.x, -s.p2c.y, -s.p2c.z };
    const float crx[3] = { negP2c[1]*paT[2] - negP2c[2]*paT[1],
                           negP2c[2]*paT[0] - negP2c[0]*paT[2],
                           negP2c[0]*paT[1] - negP2c[1]*paT[0] };
    float pdT[3] = { paT[0], paT[1], paT[2] };
    float pdB[3] = { s.paBot.x + crx[0], s.paBot.y + crx[1], s.paBot.z + crx[2] };

    // tJointDelta zeroed, seeded with mDeferredQstZ only for active dofs (CUDA
    // L341/L353) -> inactive-lane qstz is a don't-care, correctly ignored.
    const float qm[3] = { s.qstz.x, s.qstz.y, s.qstz.z };
    float tjd[3] = { 0.f, 0.f, 0.f };
    const float jv[3] = { s.jvel.x, s.jvel.y, s.jvel.z };
    float jvo[3] = { jv[0], jv[1], jv[2] };

    for (uint32_t ind = 0; ind < 3; ++ind)
    {
        if (ind < dofCount)
        {
            const float isB[3] = { s.isBot[ind].x, s.isBot[ind].y, s.isBot[ind].z };
            const float isT[3] = { s.isTop[ind].x, s.isTop[ind].y, s.isTop[ind].z };
            tjd[ind] = qm[ind] - (dot3(isB, pdT) + dot3(isT, pdB));  // innerProduct duality
        }
    }

    for (uint32_t ind = 0; ind < 3; ++ind)
    {
        if (ind < dofCount)
        {
            const F4 iS = s.invS[ind];
            const float jDelta = iS.x * tjd[0] + iS.y * tjd[1] + iS.z * tjd[2];
            jvo[ind] = jv[ind] + jDelta;
            for (int k = 0; k < 3; ++k)
            {
                pdT[k] += (&s.mmTop[ind].x)[k] * jDelta;
                pdB[k] += (&s.mmBot[ind].x)[k] * jDelta;
            }
        }
    }

    o.pdTop = F4{ pdT[0], pdT[1], pdT[2], 0.f };
    o.pdBot = F4{ pdB[0], pdB[1], pdB[2], 0.f };
    o.jvout = F4{ jvo[0], jvo[1], jvo[2], 0.f };
}

PXG_TEST(npArtiPropagateAccelJV)
{
    std::mt19937 rng(193001);
    const uint32_t N = 3000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto rv = [&](float a, float b) { return F4{ rf(a, b), rf(a, b), rf(a, b), 0.f }; };

    std::vector<JvIn> in(N);
    std::vector<JvOut> ref(N);
    size_t cDof[4] = { 0, 0, 0, 0 }, cZeroP2c = 0, cQnz = 0, cJvnz = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        JvIn s{};
        const uint32_t dof = rng() % 4;
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
        s.qstz = qnz ? F4{ rf(-5, 5), rf(-5, 5), rf(-5, 5), 0.f } : F4{ 0, 0, 0, 0 };
        // nonzero joint-velocity seed so the writeback jVel + jDelta is observable
        const bool jvnz = (i % 4 != 0);
        if (jvnz && dof) ++cJvnz;
        s.jvel = jvnz ? F4{ rf(-4, 4), rf(-4, 4), rf(-4, 4), 0.f } : F4{ 0, 0, 0, 0 };

        in[i] = s;
        refApAccelJV(s, ref[i]);
    }

    std::printf("[ info ] jv cov: dof0 %zu dof1 %zu dof2 %zu dof3 %zu zeroP2c %zu qNz %zu jvNz %zu\n",
                cDof[0], cDof[1], cDof[2], cDof[3], cZeroP2c, cQnz, cJvnz);
    PXG_CHECK(cDof[0] > 400);
    PXG_CHECK(cDof[1] > 400);
    PXG_CHECK(cDof[2] > 400);
    PXG_CHECK(cDof[3] > 400);
    PXG_CHECK(cZeroP2c > 200);
    PXG_CHECK(cQnz > 800);
    PXG_CHECK(cJvnz > 800);

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<JvOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npArtiPropagateAccelJV.spv", "apAccelJV",
             { (uint32_t)sizeof(JvIn), (uint32_t)sizeof(JvOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<JvOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* rfp = &ref[i].pdTop.x;
        const float* qfp = &g[i].pdTop.x;
        const size_t nf = sizeof(JvOut) / 4;
        bool ok = true;
        for (size_t x = 0; x < nf && ok; ++x) ok = eqFZ(rfp[x], qfp[x]);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u dof %u jvR (%g %g %g) jvG (%g %g %g)\n", i, in[i].cfg.x,
                            ref[i].jvout.x, ref[i].jvout.y, ref[i].jvout.z, g[i].jvout.x,
                            g[i].jvout.y, g[i].jvout.z);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "apAccelJV outputs bitwise-identical (mod +-0)");
}

}  // namespace
