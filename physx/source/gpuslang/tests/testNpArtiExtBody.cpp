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

// Verification of npArtiExtBody.slang: createPxgSolverExtBody +
// createImpulseResponseVector + projectVelocity + projectAngular +
// getImpulseResponse (artiConstraintPrep2.cu L55-165) against an exact CPU
// transcription.  ALL outputs bitwise (modulo the documented +-0 class for
// the float products).

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
struct U4 { uint32_t x, y, z, w; };

struct EbIn
{
    U4 meta;
    F4 artQ, artP, artVelTop, artVelBot;
    F4 artResp[9];
    F4 rbLin, rbAng, rbFrameQ, rbFrameP;
    F4 rbInertia[3];
    F4 rbMisc;
    F4 impTop, impBot, vecTop, vecBot, pav;
};

struct EbOut
{
    F4 resp[9];
    F4 velTop, velBot, b2wQ, b2wP;
    U4 meta2;
    F4 misc;
    F4 respVecTop, respVecBot, impRespTop, impRespBot;
};

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b) || (a == 0.f && b == 0.f); }

// column-major [6][6] helper over the flattened F4[9] image
static float rAt(const F4* m, int j, int i)
{
    const int f = j * 6 + i;
    return (&m[f / 4].x)[f % 4];
}
static void rSet(F4* m, int j, int i, float v)
{
    const int f = j * 6 + i;
    (&m[f / 4].x)[f % 4] = v;
}

// exact transcription of artiConstraintPrep2.cu L55-165
static void refExtBody(const EbIn& s, EbOut& o)
{
    std::memset(&o, 0, sizeof(o));
    const bool isArticulation = s.meta.x != 0;

    float velTop[3], velBot[3];
    float penBiasClamp, maxImpulse, cfm, offsetSlop;
    uint32_t linkIndex;
    bool isKinematic;

    if (isArticulation)
    {
        linkIndex = s.meta.y;
        o.b2wQ = s.artQ;
        o.b2wP = F4{ s.artP.x, s.artP.y, s.artP.z, 0 };
        for (int f = 0; f < 9; ++f) o.resp[f] = s.artResp[f];
        isKinematic = false;
        penBiasClamp = s.artP.w;
        maxImpulse = 3.402823466e+38f;  // PX_MAX_F32
        velTop[0] = s.artVelTop.x; velTop[1] = s.artVelTop.y; velTop[2] = s.artVelTop.z;
        velBot[0] = s.artVelBot.x; velBot[1] = s.artVelBot.y; velBot[2] = s.artVelBot.z;
        cfm = s.artVelTop.w;
        offsetSlop = s.artVelBot.w;
    }
    else
    {
        const bool isStatic = s.meta.y != 0;
        isKinematic = (s.meta.z & 1u) != 0;

        velTop[0] = velTop[1] = velTop[2] = 0.f;
        velBot[0] = velBot[1] = velBot[2] = 0.f;
        penBiasClamp = -3.402823466e+38f;
        float invMass = 0.f;
        offsetSlop = 0.f;
        if (!isStatic)
        {
            penBiasClamp = s.rbAng.w;
            invMass = s.rbLin.w;
            velTop[0] = s.rbAng.x; velTop[1] = s.rbAng.y; velTop[2] = s.rbAng.z;
            velBot[0] = s.rbLin.x; velBot[1] = s.rbLin.y; velBot[2] = s.rbLin.z;
            offsetSlop = s.rbMisc.y;
        }

        if (!(isStatic || isKinematic))
        {
            rSet(o.resp, 0, 3, invMass);
            rSet(o.resp, 1, 4, invMass);
            rSet(o.resp, 2, 5, invMass);
            for (int c = 0; c < 3; ++c)
            {
                rSet(o.resp, 3 + c, 0, (&s.rbInertia[c].x)[0]);
                rSet(o.resp, 3 + c, 1, (&s.rbInertia[c].x)[1]);
                rSet(o.resp, 3 + c, 2, (&s.rbInertia[c].x)[2]);
            }
        }

        linkIndex = 0xFFFFu;
        o.b2wQ = s.rbFrameQ;
        o.b2wP = F4{ s.rbFrameP.x, s.rbFrameP.y, s.rbFrameP.z, 0 };
        maxImpulse = s.rbMisc.x;
        cfm = 0.f;
    }

    // createImpulseResponseVector
    if (isArticulation || isKinematic)
    {
        o.respVecTop = F4{ s.impTop.x, s.impTop.y, s.impTop.z, 0 };
        o.respVecBot = F4{ s.impBot.x, s.impBot.y, s.impBot.z, 0 };
    }
    else
    {
        o.respVecTop = F4{
            rAt(o.resp, 3, 0) * s.impTop.x + rAt(o.resp, 3, 1) * s.impTop.y + rAt(o.resp, 3, 2) * s.impTop.z,
            rAt(o.resp, 4, 0) * s.impTop.x + rAt(o.resp, 4, 1) * s.impTop.y + rAt(o.resp, 4, 2) * s.impTop.z,
            rAt(o.resp, 5, 0) * s.impTop.x + rAt(o.resp, 5, 1) * s.impTop.y + rAt(o.resp, 5, 2) * s.impTop.z,
            0 };
        o.respVecBot = F4{ s.impBot.x, s.impBot.y, s.impBot.z, 0 };
    }

    // projectVelocity: spatial dot (top.top + bottom.bottom)
    const float projVel =
        (velTop[0] * s.impTop.x + velTop[1] * s.impTop.y + velTop[2] * s.impTop.z) +
        (velBot[0] * s.impBot.x + velBot[1] * s.impBot.y + velBot[2] * s.impBot.z);
    // projectAngular
    const float projAng = velTop[0] * s.pav.x + velTop[1] * s.pav.y + velTop[2] * s.pav.z;

    // getImpulseResponse: 6x6 column product, j-order accumulation
    float st[6] = { s.vecTop.x, s.vecTop.y, s.vecTop.z, s.vecBot.x, s.vecBot.y, s.vecBot.z };
    float result[6];
    for (int i = 0; i < 6; ++i)
    {
        result[i] = 0.f;
        for (int j = 0; j < 6; ++j)
            result[i] += rAt(o.resp, j, i) * st[j];
    }

    o.velTop = F4{ velTop[0], velTop[1], velTop[2], penBiasClamp };
    o.velBot = F4{ velBot[0], velBot[1], velBot[2], maxImpulse };
    o.b2wP.w = cfm;
    o.meta2 = U4{ linkIndex, isKinematic ? 1u : 0u, s.meta.w, 0 };
    o.misc = F4{ offsetSlop, projVel, projAng, 0 };
    o.impRespTop = F4{ result[0], result[1], result[2], 0 };
    o.impRespBot = F4{ result[3], result[4], result[5], 0 };
}

PXG_TEST(npArtiExtBody_body)
{
    std::mt19937 rng(179001);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto rq = [&]() {
        float a[4]; float m2 = 0;
        for (int k = 0; k < 4; ++k) { a[k] = rf(-1, 1); m2 += a[k] * a[k]; }
        const float i = 1.f / std::sqrt(m2 > 0 ? m2 : 1.f);
        return F4{ a[0] * i, a[1] * i, a[2] * i, a[3] * i };
    };

    std::vector<EbIn> in(N);
    std::vector<EbOut> ref(N);
    size_t cArt = 0, cDyn = 0, cStat = 0, cKin = 0, cSalt = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        EbIn s{};
        const uint32_t kind = i % 10;  // 0-3 arti, 4-7 dynamic, 8 static, 9 kinematic
        const bool isArt = kind < 4;
        const bool isStat = kind == 8;
        const bool isKin = kind == 9;
        // salt unrelated flag bits: the kinematic gate tests bit 0 only
        uint32_t flags = (isKin ? 1u : 0u) | ((rng() % 2) ? 2u : 0u) | ((rng() % 2) ? 8u : 0u);
        if (flags & ~1u) ++cSalt;
        s.meta = U4{ isArt ? 1u : 0u, isArt ? (rng() % 64u) : (isStat ? 1u : 0u), flags,
                     rng() % 1024u };

        s.artQ = rq();
        s.artP = F4{ rf(-5, 5), rf(-5, 5), rf(-5, 5), rf(-10, 0) };
        s.artVelTop = F4{ rf(-3, 3), rf(-3, 3), rf(-3, 3), rf(0, 0.1f) };
        s.artVelBot = F4{ rf(-3, 3), rf(-3, 3), rf(-3, 3), rf(0, 0.02f) };
        for (int f = 0; f < 9; ++f)
            s.artResp[f] = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), rf(-1, 1) };

        s.rbLin = F4{ rf(-3, 3), rf(-3, 3), rf(-3, 3), rf(0.01f, 10.f) };
        s.rbAng = F4{ rf(-3, 3), rf(-3, 3), rf(-3, 3), rf(-10, 0) };
        s.rbFrameQ = rq();
        s.rbFrameP = F4{ rf(-5, 5), rf(-5, 5), rf(-5, 5), 0 };
        for (int c = 0; c < 3; ++c)
            s.rbInertia[c] = F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), 0 };
        s.rbMisc = F4{ rf(0, 1e6f), rf(0, 0.05f), 0, 0 };

        s.impTop = F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), 0 };
        s.impBot = F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), 0 };
        s.vecTop = F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), 0 };
        s.vecBot = F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), 0 };
        s.pav = F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), 0 };

        in[i] = s;
        refExtBody(s, ref[i]);
        if (isArt) ++cArt; else if (isStat) ++cStat; else if (isKin) ++cKin; else ++cDyn;
    }

    std::printf("[ info ] aeb cov: arti %zu dyn %zu static %zu kin %zu salted %zu\n", cArt, cDyn,
                cStat, cKin, cSalt);

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<EbOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npArtiExtBody.spv", "aebBody",
             { (uint32_t)sizeof(EbIn), (uint32_t)sizeof(EbOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<EbOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* r = &ref[i].resp[0].x;
        const float* q = &g[i].resp[0].x;
        bool ok = std::memcmp(&ref[i].meta2, &g[i].meta2, sizeof(U4)) == 0;
        const size_t nf = sizeof(EbOut) / 4;
        for (size_t x = 0; x < nf && ok; ++x)
        {
            if (x >= (offsetof(EbOut, meta2) / 4) && x < (offsetof(EbOut, meta2) / 4 + 4))
                continue;  // uint lanes compared above
            ok = eqFZ(r[x], q[x]);
        }
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u kind %u projR %g projG %g\n", i, in[i].meta.x, ref[i].misc.y,
                            g[i].misc.y);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "aebBody: %zu mismatches", mm);

    const bool cov = cArt > N / 3 && cDyn > N / 3 && cStat > 400 && cKin > 400 && cSalt > N / 2;
    PXG_CHECK_MSG(cov, "aebBody: coverage thin (arti %zu dyn %zu stat %zu kin %zu)", cArt, cDyn,
                  cStat, cKin);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
