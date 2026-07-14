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

// ---- setupFinalizeExtSolverConstraintsBlock, PGS (tranche 180) ----

struct FzIn
{
    U4 bMeta;
    F4 b0resp[9];
    F4 b0velTop, b0velBot, b0q, b0p;
    F4 b1resp[9];
    F4 b1velTop, b1velBot, b1q, b1p;
    U4 cfgU;
    F4 invMassScale, normalRest, mat, scal;
    F4 cPts[12];
    F4 anchors0[2], anchors1[2];
    U4 contactIDs;
};

struct FzOut
{
    U4 hdrU;
    U4 writeMask;
    F4 invMassOut, normalSF, hdrF;
    F4 fricN[2];
    F4 cOut[18];
    F4 artiC[18];
    F4 fOut[12];
    F4 artiF[12];
};

struct RBody
{
    const F4* resp;
    float velTop[3], velBot[3], p[3];
    F4 q;
    float penBiasClamp, maxImpulse, cfm;
    bool isArt, isKin;
};

static void rCross(const float* a, const float* b, float* o)
{
    o[0] = a[1] * b[2] - a[2] * b[1];
    o[1] = a[2] * b[0] - a[0] * b[2];
    o[2] = a[0] * b[1] - a[1] * b[0];
}
static float rDot(const float* a, const float* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
static void rQuatRot(const F4& q, const float* v, float* o)
{
    const float vx = 2.f * v[0], vy = 2.f * v[1], vz = 2.f * v[2];
    const float w2 = q.w * q.w - 0.5f;
    const float d2 = q.x * vx + q.y * vy + q.z * vz;
    o[0] = vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d2;
    o[1] = vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d2;
    o[2] = vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d2;
}
static float rmAt(const F4* m, int j, int i) { return rAt(m, j, i); }
static void rSpatialMul(const F4* resp, const float* vt, const float* vb, float* ot, float* ob)
{
    float st[6] = { vt[0], vt[1], vt[2], vb[0], vb[1], vb[2] };
    float res[6];
    for (int i = 0; i < 6; ++i)
    {
        res[i] = 0.f;
        for (int j = 0; j < 6; ++j)
            res[i] += rmAt(resp, j, i) * st[j];
    }
    ot[0] = res[0]; ot[1] = res[1]; ot[2] = res[2];
    ob[0] = res[3]; ob[1] = res[4]; ob[2] = res[5];
}
static void rCirv(const RBody& b, const float* top, const float* bot, float* ot, float* ob)
{
    if (b.isArt || b.isKin)
    {
        for (int k = 0; k < 3; ++k) { ot[k] = top[k]; ob[k] = bot[k]; }
    }
    else
    {
        ot[0] = rmAt(b.resp, 3, 0) * top[0] + rmAt(b.resp, 3, 1) * top[1] + rmAt(b.resp, 3, 2) * top[2];
        ot[1] = rmAt(b.resp, 4, 0) * top[0] + rmAt(b.resp, 4, 1) * top[1] + rmAt(b.resp, 4, 2) * top[2];
        ot[2] = rmAt(b.resp, 5, 0) * top[0] + rmAt(b.resp, 5, 1) * top[1] + rmAt(b.resp, 5, 2) * top[2];
        for (int k = 0; k < 3; ++k) ob[k] = bot[k];
    }
}
static void rSlopZero(float* v, float slop)
{
    for (int k = 0; k < 3; ++k) v[k] = std::fabs(v[k]) < slop ? 0.f : v[k];
}
static float pxMaxR(float a, float b) { return (a < b) ? b : a; }
static float pxMinR(float a, float b) { return (a < b) ? a : b; }

// exact transcription of setupFinalizeExtSolverConstraintsBlock PGS
// (artiConstraintPrep2.cu L169-613)
static void refFinalizePGS(const FzIn& s, FzOut& o)
{
    std::memset(&o, 0, sizeof(o));

    RBody b0{ s.b0resp, { s.b0velTop.x, s.b0velTop.y, s.b0velTop.z },
              { s.b0velBot.x, s.b0velBot.y, s.b0velBot.z }, { s.b0p.x, s.b0p.y, s.b0p.z }, s.b0q,
              s.b0velTop.w, s.b0velBot.w, s.b0p.w, s.bMeta.x != 0, s.bMeta.y != 0 };
    RBody b1{ s.b1resp, { s.b1velTop.x, s.b1velTop.y, s.b1velTop.z },
              { s.b1velBot.x, s.b1velBot.y, s.b1velBot.z }, { s.b1p.x, s.b1p.y, s.b1p.z }, s.b1q,
              s.b1velTop.w, s.b1velBot.w, s.b1p.w, s.bMeta.z != 0, s.bMeta.w != 0 };

    const uint32_t flags = s.cfgU.x;
    const uint32_t contactCount = s.cfgU.y;
    const uint32_t aCount = s.cfgU.z;
    const bool perPointFriction = s.cfgU.w != 0;
    const float restDistance = s.mat.x;
    const float invDt = s.scal.x, dt = s.scal.y, bounceThreshold = s.scal.z,
                solverOffsetSlop = s.scal.w;
    const float ccdMaxSeparation = 3.402823466e+38f;  // isCCD hard-false

    const float maxPenBias = pxMaxR(b0.penBiasClamp, b1.penBiasClamp);
    const float invDtp8 = invDt * 0.8f;
    const F4 ims = s.invMassScale;
    const float cfm = pxMaxR(b0.cfm, b1.cfm);
    const uint32_t anchorCount = aCount * 2;

    if (contactCount == 0)
    {
        o.hdrU = U4{ 0xFFFFFFFFu, 0, 0, 0 };
        o.writeMask.x = 1;
        return;
    }

    const float maxImpulse = pxMinR(b0.maxImpulse, b1.maxImpulse);
    o.invMassOut = F4{ ims.x, ims.z, ims.y, ims.w };

    const float normal[3] = { s.normalRest.x, s.normalRest.y, s.normalRest.z };
    const float restitution = s.normalRest.w;
    const float damping = s.mat.y;
    const float norVel0 = rDot(normal, b0.velBot);
    const float norVel1 = rDot(normal, b1.velBot);
    const float linNorVel = norVel0 - norVel1;

    float normalForce = 0.f;

    for (uint32_t j = 0; j < contactCount && j < 6; ++j)
    {
        const F4& p4 = s.cPts[j * 2 + 0];
        const F4& tv4 = s.cPts[j * 2 + 1];
        const float point[3] = { p4.x, p4.y, p4.z };
        const float separation = p4.w;
        const float targetVel3[3] = { tv4.x, tv4.y, tv4.z };
        const float cTargetVel = rDot(normal, targetVel3);

        float ra[3], rb[3], raXn[3], rbXn[3];
        for (int k = 0; k < 3; ++k) { ra[k] = point[k] - b0.p[k]; rb[k] = point[k] - b1.p[k]; }
        rCross(ra, normal, raXn);
        rCross(rb, normal, rbXn);

        const float angNorVel = rDot(raXn, b0.velTop) - rDot(rbXn, b1.velTop);
        const float slop =
            solverOffsetSlop * pxMaxR(linNorVel == 0.f ? 1.f : angNorVel / linNorVel, 1.f);
        rSlopZero(raXn, slop);
        rSlopZero(rbXn, slop);

        float nraXn[3] = { -rbXn[0], -rbXn[1], -rbXn[2] };
        float nNormal[3] = { -normal[0], -normal[1], -normal[2] };
        float r0t[3], r0b[3], r1t[3], r1b[3];
        rCirv(b0, raXn, normal, r0t, r0b);
        rCirv(b1, nraXn, nNormal, r1t, r1b);

        float in0t[3], in0b[3], in1t[3], in1b[3];
        for (int k = 0; k < 3; ++k)
        {
            in0t[k] = normal[k] * ims.x; in0b[k] = raXn[k] * ims.y;
            in1t[k] = -normal[k] * ims.z; in1b[k] = -rbXn[k] * ims.w;
        }
        float dV0t[3], dV0b[3], dV1t[3], dV1b[3];
        rSpatialMul(b0.resp, in0t, in0b, dV0t, dV0b);
        rSpatialMul(b1.resp, in1t, in1b, dV1t, dV1b);

        const float resp0 = rDot(dV0t, r0t) + rDot(dV0b, r0b);
        const float resp1 = rDot(dV1t, r1t) + rDot(dV1b, r1b);
        const float unitResponse = resp0 + resp1;

        const float vrel1 = norVel0 + rDot(b0.velTop, raXn);
        const float vrel2 = norVel1 + rDot(b1.velTop, rbXn);
        const float vrel = vrel1 - vrel2;

        const float penetration = separation - restDistance;
        const bool isSeparated = penetration >= 0.f;
        const float penetrationInvDt = penetration * invDt;
        const float penetrationInvDtPt8 = pxMaxR(maxPenBias, penetration * invDtp8);

        const bool collidingWithVrel = (-vrel) > penetrationInvDt;
        const bool isGreater2 = (restitution > 0.f) && (bounceThreshold > vrel) && collidingWithVrel;
        const bool ccdSeparationCondition = ccdMaxSeparation >= penetration;

        float targetVelocity = cTargetVel + (isGreater2 ? (-vrel) * restitution : 0.f);
        const float tvel = targetVelocity;
        if (!b0.isArt) targetVelocity -= vrel1;
        if (!b1.isArt) targetVelocity += vrel2;

        const float recipResponse = (unitResponse > 0.f) ? (1.f / (unitResponse + cfm)) : 0.f;
        float velMultiplier = recipResponse;

        float coeff0, coeff1;
        if (restitution < 0.f)
        {
            const bool accelSpring = (flags & 8u) != 0;
            const float massIfAccelElseOne = accelSpring ? recipResponse : 1.f;
            const float oneIfAccelElseR = accelSpring ? 1.f : unitResponse;
            const float nrdt = dt * restitution;
            const float dampingIfEnabled = (isSeparated && !collidingWithVrel) ? 0.f : damping;
            const float a = dt * (dampingIfEnabled - nrdt);
            const float x = 1.f / (a * oneIfAccelElseR + 1.f);
            velMultiplier = x * a * massIfAccelElseOne;
            coeff0 = a;
            coeff1 = -nrdt * penetration;
        }
        else
        {
            float scaledBias = isSeparated ? penetrationInvDt : penetrationInvDtPt8;
            if (ccdSeparationCondition && isGreater2)
                scaledBias = 0.f;
            float unbiasedErr = targetVelocity;
            const float biasedErr = unbiasedErr - scaledBias;
            if (!isGreater2)
                unbiasedErr -= pxMaxR(0.f, scaledBias);
            coeff0 = biasedErr;
            coeff1 = unbiasedErr;
        }

        o.cOut[j * 3 + 0] = F4{ r0t[0], r0t[1], r0t[2], targetVelocity };
        o.cOut[j * 3 + 1] = F4{ -r1t[0], -r1t[1], -r1t[2], pxMinR(maxImpulse, tv4.w) };
        o.cOut[j * 3 + 2] = F4{ resp0, resp1, coeff0, coeff1 };

        const float deltaF =
            pxMaxR((tvel - (vrel + pxMaxR(penetrationInvDt, 0.f))) * velMultiplier, 0.f);
        normalForce += deltaF;

        o.artiC[j * 3 + 0] = F4{ dV0b[0], dV0b[1], dV0b[2], dV0t[0] };
        o.artiC[j * 3 + 1] = F4{ dV0t[1], dV0t[2], dV1b[0], dV1b[1] };
        o.artiC[j * 3 + 2] = F4{ dV1b[2], dV1t[0], dV1t[1], dV1t[2] };
    }

    o.hdrF.z = (normalForce / float(contactCount)) * 0.25f;

    const float frictionScale = (aCount == 2) ? 0.5f : 1.f;
    const float staticFriction = s.mat.z * frictionScale;
    const float dynamicFriction = s.mat.w * frictionScale;
    const bool haveFriction = anchorCount != 0;

    o.hdrU = U4{ 0, flags, contactCount, anchorCount };
    o.hdrF.x = restitution;
    o.hdrF.y = cfm;
    o.hdrF.w = dynamicFriction;
    o.normalSF = F4{ normal[0], normal[1], normal[2], staticFriction };
    o.writeMask.x = 2;

    if (haveFriction)
    {
        o.writeMask.x |= 4;
        float vrelV[3];
        for (int k = 0; k < 3; ++k) vrelV[k] = b0.velBot[k] - b1.velBot[k];

        const float orthoThreshold = 0.70710678f;
        const float p1 = 0.0001f;
        float t0[3];
        {
            const float fb1[3] = { 0.f, -normal[2], normal[1] };
            const float fb2[3] = { -normal[1], normal[0], 0.f };
            const float* fb = orthoThreshold > std::fabs(normal[0]) ? fb1 : fb2;
            const float nv = rDot(normal, vrelV);
            for (int k = 0; k < 3; ++k) t0[k] = vrelV[k] - normal[k] * nv;
            if (!(rDot(t0, t0) > p1))
                for (int k = 0; k < 3; ++k) t0[k] = fb[k];
            const float m = std::sqrt(rDot(t0, t0));
            if (m > 0.f)
            {
                const float inv = 1.f / m;
                for (int k = 0; k < 3; ++k) t0[k] *= inv;
            }
        }
        float t1[3];
        rCross(normal, t0, t1);

        o.fricN[0] = F4{ t0[0], t0[1], t0[2], 0.f };
        o.fricN[1] = F4{ t1[0], t1[1], t1[2], 0.f };

        for (uint32_t j = 0; j < aCount && j < 2; ++j)
        {
            const float a0[3] = { s.anchors0[j].x, s.anchors0[j].y, s.anchors0[j].z };
            const float a1[3] = { s.anchors1[j].x, s.anchors1[j].y, s.anchors1[j].z };
            float ra[3], rb[3], error[3];
            rQuatRot(b0.q, a0, ra);
            rQuatRot(b1.q, a1, rb);
            for (int k = 0; k < 3; ++k) error[k] = (ra[k] + b0.p[k]) - (rb[k] + b1.p[k]);

            const uint32_t index = perPointFriction ? (j == 0 ? s.contactIDs.x : s.contactIDs.y) : 0;
            const float tvel3[3] = { s.cPts[index * 2 + 1].x, s.cPts[index * 2 + 1].y,
                                     s.cPts[index * 2 + 1].z };

            for (int rrow = 0; rrow < 2; ++rrow)
            {
                const float* t = rrow == 0 ? t0 : t1;
                float raXn[3], rbXn[3];
                rCross(ra, t, raXn);
                rCross(rb, t, rbXn);
                rSlopZero(raXn, solverOffsetSlop);
                rSlopZero(rbXn, solverOffsetSlop);

                float nrbXn[3] = { -rbXn[0], -rbXn[1], -rbXn[2] };
                float nt[3] = { -t[0], -t[1], -t[2] };
                float r0t[3], r0b[3], r1t[3], r1b[3];
                rCirv(b0, raXn, t, r0t, r0b);
                rCirv(b1, nrbXn, nt, r1t, r1b);

                float in0t[3], in0b[3], in1t[3], in1b[3];
                for (int k = 0; k < 3; ++k)
                {
                    in0t[k] = t[k] * s.invMassScale.x; in0b[k] = raXn[k] * s.invMassScale.y;
                    in1t[k] = -t[k] * s.invMassScale.z; in1b[k] = -rbXn[k] * s.invMassScale.w;
                }
                float dV0t[3], dV0b[3], dV1t[3], dV1b[3];
                rSpatialMul(b0.resp, in0t, in0b, dV0t, dV0b);
                rSpatialMul(b1.resp, in1t, in1b, dV1t, dV1b);

                const float resp0 = rDot(dV0t, r0t) + rDot(dV0b, r0b);
                const float resp1 = rDot(dV1t, r1t) + rDot(dV1b, r1b);

                float targetVel = rDot(tvel3, t);
                if (!b0.isArt)
                    targetVel -= rDot(b0.velTop, raXn) + rDot(b0.velBot, t);
                if (!b1.isArt)
                    targetVel += rDot(b1.velTop, rbXn) + rDot(b1.velBot, t);

                const float bias = rDot(t, error) * invDt;
                const uint32_t slot = j * 2 + rrow;
                o.fOut[slot * 3 + 0] = F4{ r0t[0], r0t[1], r0t[2], bias };
                o.fOut[slot * 3 + 1] = F4{ -r1t[0], -r1t[1], -r1t[2], targetVel };
                o.fOut[slot * 3 + 2] = F4{ resp0, resp1, 0.f, 0.f };

                o.artiF[slot * 3 + 0] = F4{ dV0b[0], dV0b[1], dV0b[2], dV0t[0] };
                o.artiF[slot * 3 + 1] = F4{ dV0t[1], dV0t[2], dV1b[0], dV1b[1] };
                o.artiF[slot * 3 + 2] = F4{ dV1b[2], dV1t[0], dV1t[1], dV1t[2] };
            }
        }
    }
}

PXG_TEST(npArtiExtBody_finalizePGS)
{
    std::mt19937 rng(180001);
    const uint32_t N = 3000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto rq = [&]() {
        float a[4]; float m2 = 0;
        for (int k = 0; k < 4; ++k) { a[k] = rf(-1, 1); m2 += a[k] * a[k]; }
        const float i = 1.f / std::sqrt(m2 > 0 ? m2 : 1.f);
        return F4{ a[0] * i, a[1] * i, a[2] * i, a[3] * i };
    };

    std::vector<FzIn> in(N);
    std::vector<FzOut> ref(N);
    size_t cEarly = 0, cCompliant = 0, cBounce = 0, cFric = 0, cNoFric = 0, cFall = 0, cZeroLNV = 0,
           cArtArt = 0, cMixed = 0, cKin = 0, cPPF = 0, cA2 = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        FzIn s{};
        // body kinds: 0 arti, 1 dynamic, 2 kinematic
        const uint32_t k0 = rng() % 3, k1 = rng() % 3;
        s.bMeta = U4{ k0 == 0 ? 1u : 0u, k0 == 2 ? 1u : 0u, k1 == 0 ? 1u : 0u, k1 == 2 ? 1u : 0u };
        if (k0 == 0 && k1 == 0) ++cArtArt; else if (k0 == 0 || k1 == 0) ++cMixed;
        if (k0 == 2 || k1 == 2) ++cKin;

        auto fillBody = [&](F4* resp, F4& vt, F4& vb, F4& q, F4& p) {
            for (int f = 0; f < 9; ++f)
                resp[f] = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), rf(-1, 1) };
            vt = F4{ rf(-3, 3), rf(-3, 3), rf(-3, 3), rf(-10, 0) };
            vb = F4{ rf(-3, 3), rf(-3, 3), rf(-3, 3), rf(0.5f, 1e5f) };
            q = rq();
            p = F4{ rf(-5, 5), rf(-5, 5), rf(-5, 5), rf(0, 0.1f) };
        };
        fillBody(s.b0resp, s.b0velTop, s.b0velBot, s.b0q, s.b0p);
        fillBody(s.b1resp, s.b1velTop, s.b1velBot, s.b1q, s.b1p);

        const uint32_t contactCount = (i % 13 == 7) ? 0u : (1u + rng() % 6u);
        const uint32_t aCount = rng() % 3u;
        const bool ppf = (rng() % 2) != 0;
        uint32_t solverFlags = (rng() % 2 ? 1u : 0u) | (rng() % 2 ? 2u : 0u) |
                               (rng() % 2 ? 4u : 0u) | (rng() % 2 ? 8u : 0u);
        s.cfgU = U4{ solverFlags, contactCount, aCount, ppf ? 1u : 0u };
        if (contactCount == 0) ++cEarly;
        if (aCount > 0 && contactCount) ++cFric; else if (contactCount) ++cNoFric;
        if (aCount == 2) ++cA2;
        if (ppf && aCount && contactCount) ++cPPF;

        s.invMassScale = F4{ rf(0, 1.5f), rf(0, 1.5f), rf(0, 1.5f), rf(0, 1.5f) };
        float n[3] = { rf(-1, 1), rf(-1, 1), rf(-1, 1) };
        if (i % 5 == 2) { n[0] = (rng() % 2) ? 0.9f : -0.9f; }  // |normal.x| > ortho
        const float nm = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        for (int k = 0; k < 3; ++k) n[k] /= (nm > 0 ? nm : 1.f);
        // restitution: <0 compliant, 0, >0 bounce
        const int rMode = rng() % 4;
        const float restitution = rMode == 0 ? rf(-500.f, -1.f) : (rMode == 1 ? 0.f : rf(0.1f, 1.f));
        if (rMode == 0) ++cCompliant;
        if (rMode >= 2) ++cBounce;
        s.normalRest = F4{ n[0], n[1], n[2], restitution };
        s.mat = F4{ rf(-0.05f, 0.1f), rf(0, 50.f), rf(0, 2.f), rf(0, 2.f) };
        s.scal = F4{ rf(30.f, 120.f), rf(0.008f, 0.033f), rf(-2.f, 0.5f),
                     (rng() % 3 == 0) ? 0.f : rf(0.f, 0.02f) };

        if (i % 17 == 5)
        {
            // linNorVel == 0 exactly: equal linear velocities
            s.b1velBot.x = s.b0velBot.x; s.b1velBot.y = s.b0velBot.y; s.b1velBot.z = s.b0velBot.z;
            ++cZeroLNV;
        }
        if (i % 19 == 9)
        {
            // t0 fallback: relative linear velocity parallel to the normal
            const float kk = rf(0.5f, 2.f);
            s.b1velBot.x = s.b0velBot.x - n[0] * kk;
            s.b1velBot.y = s.b0velBot.y - n[1] * kk;
            s.b1velBot.z = s.b0velBot.z - n[2] * kk;
            ++cFall;
        }

        for (int c = 0; c < 12; ++c)
            s.cPts[c] = F4{ rf(-3, 3), rf(-3, 3), rf(-3, 3),
                            (c & 1) ? rf(0.1f, 1e4f) : rf(-0.2f, 0.2f) };
        for (int a = 0; a < 2; ++a)
        {
            s.anchors0[a] = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), 0 };
            s.anchors1[a] = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), 0 };
        }
        s.contactIDs = U4{ contactCount ? rng() % contactCount : 0u,
                           contactCount ? rng() % contactCount : 0u, 0, 0 };

        in[i] = s;
        refFinalizePGS(s, ref[i]);
    }

    std::printf("[ info ] fz cov: early %zu compliant %zu bounce %zu fric %zu nofric %zu fall %zu "
                "zeroLNV %zu artArt %zu mixed %zu kin %zu ppf %zu a2 %zu\n",
                cEarly, cCompliant, cBounce, cFric, cNoFric, cFall, cZeroLNV, cArtArt, cMixed, cKin,
                cPPF, cA2);

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<EbIn> dummyIn(1);
    std::vector<EbOut> dummyOut(1);
    std::vector<FzOut> outInit(N);
    PxgBufferHandle bDIn = createBufferFrom(dev, dummyIn);
    PxgBufferHandle bDOut = createBufferFrom(dev, dummyOut);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npArtiExtBody.spv", "aebFinalizePGS",
             { (uint32_t)sizeof(EbIn), (uint32_t)sizeof(EbOut), (uint32_t)sizeof(FzIn),
               (uint32_t)sizeof(FzOut) },
             sizeof(P));
    k.dispatch(GROUPS, { bDIn, bDOut, bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<FzOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const FzOut& r = ref[i]; const FzOut& q = g[i];
        bool ok = std::memcmp(&r.hdrU, &q.hdrU, sizeof(U4) * 2) == 0;
        const float* rf2 = &r.invMassOut.x;
        const float* qf = &q.invMassOut.x;
        const size_t nf = (sizeof(FzOut) - 2 * sizeof(U4)) / 4;
        for (size_t x = 0; x < nf && ok; ++x)
            ok = eqFZ(rf2[x], qf[x]);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u wmR %u wmG %u c0R (%g %g %g %g) c0G (%g %g %g %g)\n", i,
                            r.writeMask.x, q.writeMask.x, r.cOut[0].x, r.cOut[0].y, r.cOut[0].z,
                            r.cOut[0].w, q.cOut[0].x, q.cOut[0].y, q.cOut[0].z, q.cOut[0].w);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "aebFinalizePGS: %zu mismatches", mm);

    const bool cov = cEarly > 150 && cCompliant > 500 && cBounce > 800 && cFric > 1000 &&
                     cNoFric > 500 && cFall > 100 && cZeroLNV > 100 && cArtArt > 200 &&
                     cMixed > 800 && cKin > 1000 && cPPF > 400 && cA2 > 700;
    PXG_CHECK_MSG(cov, "aebFinalizePGS: coverage thin (early %zu compliant %zu fric %zu fall %zu)",
                  cEarly, cCompliant, cFric, cFall);

    for (PxgBufferHandle h : { bDIn, bDOut, bIn, bOut }) dev->destroyBuffer(h);
}

// ---- setupFinalizeExtSolverConstraintsBlock, TGS (tranche 181) ----

struct TzIn
{
    U4 bMeta;
    F4 b0resp[9];
    F4 b0velTop, b0velBot, b0q, b0p;
    F4 b1resp[9];
    F4 b1velTop, b1velBot, b1q, b1p;
    U4 cfgU;
    F4 invMassScale, normalRest, mat, scal, scal2, scal3;
    F4 cPts[12];
    F4 anchors0[2], anchors1[2];
    U4 contactIDs;
};

struct TzOut
{
    U4 hdrU;
    U4 writeMask;
    F4 invMassOut, normalSF, hdrF, hdrF2;
    F4 fricN[2];
    F4 cOut[24];
    F4 artiC[18];
    F4 fOut[12];
    F4 artiF[12];
};

// exact transcription of setupFinalizeExtSolverConstraintsBlock TGS
// (artiConstraintPrep2.cu L617-1112)
static void refFinalizeTGS(const TzIn& s, TzOut& o)
{
    std::memset(&o, 0, sizeof(o));

    RBody b0{ s.b0resp, { s.b0velTop.x, s.b0velTop.y, s.b0velTop.z },
              { s.b0velBot.x, s.b0velBot.y, s.b0velBot.z }, { s.b0p.x, s.b0p.y, s.b0p.z }, s.b0q,
              s.b0velTop.w, s.b0velBot.w, s.b0p.w, s.bMeta.x != 0, s.bMeta.y != 0 };
    RBody b1{ s.b1resp, { s.b1velTop.x, s.b1velTop.y, s.b1velTop.z },
              { s.b1velBot.x, s.b1velBot.y, s.b1velBot.z }, { s.b1p.x, s.b1p.y, s.b1p.z }, s.b1q,
              s.b1velTop.w, s.b1velBot.w, s.b1p.w, s.bMeta.z != 0, s.bMeta.w != 0 };

    const uint32_t flags = s.cfgU.x;
    const uint32_t contactCount = s.cfgU.y;
    const uint32_t aCount = s.cfgU.z;
    const bool perPointFriction = s.cfgU.w != 0;
    const float restDistance = s.mat.x;
    const float invDt = s.scal.x, stepDt = s.scal.y, bounceThreshold = s.scal.z,
                solverOffsetSlop = s.scal.w;
    const float invTotalDt = s.scal2.x, biasCoefficient = s.scal2.y, totalDt = s.scal2.z;
    const float tfx = s.scal3.x, tfy = s.scal3.y;

    const float cfm = pxMaxR(b0.cfm, b1.cfm);
    const float maxPenBias = pxMaxR(b0.penBiasClamp, b1.penBiasClamp);
    const float p8 = pxMinR(0.8f, biasCoefficient);
    const float invDtp8 = invDt * p8;
    const F4 ims = s.invMassScale;

    uint32_t anchorCount = aCount * 2;
    const bool hasTorsional = (tfx != 0.f || tfy != 0.f) && aCount == 1;
    if (hasTorsional)
        ++anchorCount;

    if (contactCount == 0)
    {
        o.hdrU = U4{ 0xFFFFFFFFu, 0, 0, 0 };
        o.writeMask.x = 1;
        return;
    }

    const float maxImpulse = pxMinR(b0.maxImpulse, b1.maxImpulse);
    o.invMassOut = F4{ ims.x, ims.z, ims.y, ims.w };

    const float normal[3] = { s.normalRest.x, s.normalRest.y, s.normalRest.z };
    const float restitution = s.normalRest.w;
    const float damping = s.mat.y;
    const float norVel0 = rDot(normal, b0.velBot);
    const float norVel1 = rDot(normal, b1.velBot);
    const float linNorVel = norVel0 - norVel1;

    o.hdrF2 = F4{ maxPenBias, p8, invDt, 0.f };
    o.hdrF.x = restitution * stepDt;
    o.hdrF.y = cfm;

    float normalForce = 0.f;
    float minPen = 0.f;

    for (uint32_t j = 0; j < contactCount && j < 6; ++j)
    {
        const F4& p4 = s.cPts[j * 2 + 0];
        const F4& tv4 = s.cPts[j * 2 + 1];
        const float point[3] = { p4.x, p4.y, p4.z };
        const float separation = p4.w;
        const float targetVel3[3] = { tv4.x, tv4.y, tv4.z };
        const float cTargetVel = rDot(normal, targetVel3);

        float ra[3], rb[3], raXn[3], rbXn[3];
        for (int k = 0; k < 3; ++k) { ra[k] = point[k] - b0.p[k]; rb[k] = point[k] - b1.p[k]; }
        rCross(ra, normal, raXn);
        rCross(rb, normal, rbXn);

        const float angNorVel = rDot(raXn, b0.velTop) - rDot(rbXn, b1.velTop);
        const float slop =
            solverOffsetSlop * pxMaxR(linNorVel == 0.f ? 1.f : angNorVel / linNorVel, 1.f);
        rSlopZero(raXn, slop);
        rSlopZero(rbXn, slop);

        float nrbXn[3] = { -rbXn[0], -rbXn[1], -rbXn[2] };
        float nNormal[3] = { -normal[0], -normal[1], -normal[2] };
        float r0t[3], r0b[3], r1t[3], r1b[3];
        rCirv(b0, raXn, normal, r0t, r0b);
        rCirv(b1, nrbXn, nNormal, r1t, r1b);

        float in0t[3], in0b[3], in1t[3], in1b[3];
        for (int k = 0; k < 3; ++k)
        {
            in0t[k] = normal[k] * ims.x; in0b[k] = raXn[k] * ims.y;
            in1t[k] = -normal[k] * ims.z; in1b[k] = -rbXn[k] * ims.w;
        }
        float dV0t[3], dV0b[3], dV1t[3], dV1b[3];
        rSpatialMul(b0.resp, in0t, in0b, dV0t, dV0b);
        rSpatialMul(b1.resp, in1t, in1b, dV1t, dV1b);

        const float resp0 = rDot(dV0t, r0t) + rDot(dV0b, r0b);
        const float resp1 = rDot(dV1t, r1t) + rDot(dV1b, r1b);
        const float unitResponse = resp0 + resp1;

        const float vrel1 = norVel0 + rDot(b0.velTop, raXn);
        const float vrel2 = norVel1 + rDot(b1.velTop, rbXn);
        const float vrel = vrel1 - vrel2;

        const float penetration = separation - restDistance;
        const bool isSeparated = penetration > 0.f;  // STRICT in TGS
        const float penetrationInvDt = penetration * invTotalDt;

        float scaledBias, velMultiplier;
        const float recipResponse = (unitResponse > 0.f) ? (1.f / (unitResponse + cfm)) : 0.f;

        float compliantContactCoeff = 0.f;
        if (restitution < 0.f)
        {
            const bool collidingWithVrel = ((-vrel) * totalDt) > penetration;
            const bool accelSpring = (flags & 8u) != 0;
            const float massIfAccelElseOne = accelSpring ? recipResponse : 1.f;
            const float oneIfAccelElseR = accelSpring ? 1.f : unitResponse;
            const float dampingIfEnabled = (isSeparated && !collidingWithVrel) ? 0.f : damping;
            const float nrdt = stepDt * restitution;
            const float a = stepDt * (dampingIfEnabled - nrdt);
            const float x = 1.f / (a * oneIfAccelElseR + 1.f);
            velMultiplier = x * a * massIfAccelElseOne;
            scaledBias = nrdt * x * oneIfAccelElseR;
            compliantContactCoeff = a;
        }
        else
        {
            velMultiplier = recipResponse;
            scaledBias = isSeparated ? -invDt : -invDtp8;
        }

        minPen = pxMinR(penetration, minPen);

        const bool isGreater2 =
            ((restitution > 1e-5f) && (bounceThreshold > vrel)) && ((-vrel) > penetrationInvDt);

        float totalError = penetration;

        // CUDA precedence trap (cu L837): (cTargetVel + bool) is the condition
        float targetVelocity =
            ((cTargetVel + (isGreater2 ? 1.f : 0.f)) != 0.f) ? (-vrel) * restitution : 0.f;

        const float deltaF =
            pxMaxR((targetVelocity + (-penetrationInvDt - vrel)) * velMultiplier, 0.f);
        normalForce += deltaF;

        if (b0.isKin) targetVelocity -= vrel1;
        if (b1.isKin) targetVelocity += vrel2;

        if (isGreater2)
        {
            const float ratio = totalDt + penetration / vrel;
            totalError += ratio * targetVelocity;
        }

        o.cOut[j * 4 + 0] = F4{ r0t[0], r0t[1], r0t[2], compliantContactCoeff };
        o.cOut[j * 4 + 1] = F4{ -r1t[0], -r1t[1], -r1t[2], targetVelocity };
        o.cOut[j * 4 + 2] = F4{ resp0, resp1, pxMinR(maxImpulse, tv4.w), totalError };
        o.cOut[j * 4 + 3] = F4{ scaledBias, 0.f, 0.f, 0.f };

        o.artiC[j * 3 + 0] = F4{ dV0b[0], dV0b[1], dV0b[2], dV0t[0] };
        o.artiC[j * 3 + 1] = F4{ dV0t[1], dV0t[2], dV1b[0], dV1b[1] };
        o.artiC[j * 3 + 2] = F4{ dV1b[2], dV1t[0], dV1t[1], dV1t[2] };
    }

    o.hdrF.z = (normalForce / float(contactCount)) * 0.25f;

    const float frictionScale = (aCount == 2) ? 0.5f : 1.f;
    const float staticFriction = s.mat.z * frictionScale;
    const float dynamicFriction = s.mat.w * frictionScale;
    const bool haveFriction = anchorCount != 0;

    o.hdrU = U4{ 0, flags, contactCount, anchorCount };
    o.hdrF.w = dynamicFriction;
    o.normalSF = F4{ normal[0], normal[1], normal[2], staticFriction };
    o.writeMask.x = 2;

    if (haveFriction)
    {
        o.writeMask.x |= 4;
        float vrelV[3];
        for (int k = 0; k < 3; ++k) vrelV[k] = b0.velBot[k] - b1.velBot[k];

        const float orthoThreshold = 0.70710678f;
        const float p1 = 0.0001f;
        float t0[3];
        {
            const float fb1[3] = { 0.f, -normal[2], normal[1] };
            const float fb2[3] = { -normal[1], normal[0], 0.f };
            const float* fb = orthoThreshold > std::fabs(normal[0]) ? fb1 : fb2;
            const float nv = rDot(normal, vrelV);
            for (int k = 0; k < 3; ++k) t0[k] = vrelV[k] - normal[k] * nv;
            if (!(rDot(t0, t0) > p1))
                for (int k = 0; k < 3; ++k) t0[k] = fb[k];
            const float m = std::sqrt(rDot(t0, t0));
            if (m > 0.f)
            {
                const float inv = 1.f / m;
                for (int k = 0; k < 3; ++k) t0[k] *= inv;
            }
        }
        float t1[3];
        rCross(normal, t0, t1);

        o.fricN[0] = F4{ t0[0], t0[1], t0[2], 0.f };
        o.fricN[1] = F4{ t1[0], t1[1], t1[2], 0.f };

        float relTr[3];
        for (int k = 0; k < 3; ++k) relTr[k] = b0.p[k] - b1.p[k];

        for (uint32_t j = 0; j < aCount && j < 2; ++j)
        {
            const float a0[3] = { s.anchors0[j].x, s.anchors0[j].y, s.anchors0[j].z };
            const float a1[3] = { s.anchors1[j].x, s.anchors1[j].y, s.anchors1[j].z };
            float ra[3], rb[3], error[3];
            rQuatRot(b0.q, a0, ra);
            rQuatRot(b1.q, a1, rb);
            for (int k = 0; k < 3; ++k) error[k] = (ra[k] - rb[k]) + relTr[k];  // TGS association

            const uint32_t index = perPointFriction ? (j == 0 ? s.contactIDs.x : s.contactIDs.y) : 0;
            const float tvel3[3] = { s.cPts[index * 2 + 1].x, s.cPts[index * 2 + 1].y,
                                     s.cPts[index * 2 + 1].z };

            for (int rrow = 0; rrow < 2; ++rrow)
            {
                const float* t = rrow == 0 ? t0 : t1;
                float raXn[3], rbXn[3];
                rCross(ra, t, raXn);
                rCross(rb, t, rbXn);
                rSlopZero(raXn, solverOffsetSlop);
                rSlopZero(rbXn, solverOffsetSlop);

                float nrbXn[3] = { -rbXn[0], -rbXn[1], -rbXn[2] };
                float nt[3] = { -t[0], -t[1], -t[2] };
                float r0t[3], r0b[3], r1t[3], r1b[3];
                rCirv(b0, raXn, t, r0t, r0b);
                rCirv(b1, nrbXn, nt, r1t, r1b);

                float in0t[3], in0b[3], in1t[3], in1b[3];
                for (int k = 0; k < 3; ++k)
                {
                    in0t[k] = t[k] * ims.x; in0b[k] = raXn[k] * ims.y;
                    in1t[k] = -t[k] * ims.z; in1b[k] = -rbXn[k] * ims.w;
                }
                float dV0t[3], dV0b[3], dV1t[3], dV1b[3];
                rSpatialMul(b0.resp, in0t, in0b, dV0t, dV0b);
                rSpatialMul(b1.resp, in1t, in1b, dV1t, dV1b);

                const float resp0 = rDot(dV0t, r0t) + rDot(dV0b, r0b);
                const float resp1 = rDot(dV1t, r1t) + rDot(dV1b, r1b);

                float targetVel = rDot(tvel3, t);
                if (b0.isKin)
                    targetVel -= rDot(b0.velTop, raXn) + rDot(b0.velBot, t);
                if (b1.isKin)
                    targetVel += rDot(b1.velTop, rbXn) + rDot(b1.velBot, t);

                const float frictionError = rDot(error, t);  // no invDt in TGS
                const uint32_t slot = j * 2 + rrow;
                o.fOut[slot * 3 + 0] = F4{ r0t[0], r0t[1], r0t[2], frictionError };
                o.fOut[slot * 3 + 1] = F4{ -r1t[0], -r1t[1], -r1t[2], targetVel };
                o.fOut[slot * 3 + 2] = F4{ resp0, resp1, 0.f, 0.f };

                o.artiF[slot * 3 + 0] = F4{ dV0b[0], dV0b[1], dV0b[2], dV0t[0] };
                o.artiF[slot * 3 + 1] = F4{ dV0t[1], dV0t[2], dV1b[0], dV1b[1] };
                o.artiF[slot * 3 + 2] = F4{ dV1b[2], dV1t[0], dV1t[1], dV1t[2] };
            }
        }

        if (hasTorsional)
        {
            const float torsionalScale =
                pxMaxR(tfy, std::sqrt(pxMaxR(0.f, -minPen) * tfx));
            o.hdrF2.w = torsionalScale;

            float nNormal[3] = { -normal[0], -normal[1], -normal[2] };
            float zero3[3] = { 0.f, 0.f, 0.f };
            float r0t[3], r0b[3], r1t[3], r1b[3];
            rCirv(b0, normal, zero3, r0t, r0b);
            rCirv(b1, nNormal, zero3, r1t, r1b);

            float in0t[3], in0b[3], in1t[3], in1b[3];
            for (int k = 0; k < 3; ++k)
            {
                in0t[k] = 0.f * ims.x; in0b[k] = normal[k] * ims.y;
                in1t[k] = 0.f * ims.z; in1b[k] = -normal[k] * ims.w;
            }
            float dV0t[3], dV0b[3], dV1t[3], dV1b[3];
            rSpatialMul(b0.resp, in0t, in0b, dV0t, dV0b);
            rSpatialMul(b1.resp, in1t, in1b, dV1t, dV1b);

            const float resp0 = rDot(dV0t, r0t) + rDot(dV0b, r0b);
            const float resp1 = rDot(dV1t, r1t) + rDot(dV1b, r1b);

            const uint32_t slot = 2;
            o.fOut[slot * 3 + 0] = F4{ r0t[0], r0t[1], r0t[2], 0.f };
            o.fOut[slot * 3 + 1] = F4{ -r1t[0], -r1t[1], -r1t[2], 0.f };
            o.fOut[slot * 3 + 2] = F4{ resp0, resp1, 0.f, 0.f };

            o.artiF[slot * 3 + 0] = F4{ dV0b[0], dV0b[1], dV0b[2], dV0t[0] };
            o.artiF[slot * 3 + 1] = F4{ dV0t[1], dV0t[2], dV1b[0], dV1b[1] };
            o.artiF[slot * 3 + 2] = F4{ dV1b[2], dV1t[0], dV1t[1], dV1t[2] };
        }
    }
}

PXG_TEST(npArtiExtBody_finalizeTGS)
{
    std::mt19937 rng(181001);
    const uint32_t N = 3000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto rq = [&]() {
        float a[4]; float m2 = 0;
        for (int k = 0; k < 4; ++k) { a[k] = rf(-1, 1); m2 += a[k] * a[k]; }
        const float i = 1.f / std::sqrt(m2 > 0 ? m2 : 1.f);
        return F4{ a[0] * i, a[1] * i, a[2] * i, a[3] * i };
    };

    std::vector<TzIn> in(N);
    std::vector<TzOut> ref(N);
    size_t cEarly = 0, cCompliant = 0, cBounce = 0, cTors = 0, cFric = 0, cFall = 0, cZeroLNV = 0,
           cKin = 0, cPPF = 0, cA2 = 0, cBiasLo = 0, cZeroTV = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        TzIn s{};
        const uint32_t k0 = rng() % 3, k1 = rng() % 3;
        s.bMeta = U4{ k0 == 0 ? 1u : 0u, k0 == 2 ? 1u : 0u, k1 == 0 ? 1u : 0u, k1 == 2 ? 1u : 0u };
        if (k0 == 2 || k1 == 2) ++cKin;

        auto fillBody = [&](F4* resp, F4& vt, F4& vb, F4& q, F4& p) {
            for (int f = 0; f < 9; ++f)
                resp[f] = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), rf(-1, 1) };
            vt = F4{ rf(-3, 3), rf(-3, 3), rf(-3, 3), rf(-10, 0) };
            vb = F4{ rf(-3, 3), rf(-3, 3), rf(-3, 3), rf(0.5f, 1e5f) };
            q = rq();
            p = F4{ rf(-5, 5), rf(-5, 5), rf(-5, 5), rf(0, 0.1f) };
        };
        fillBody(s.b0resp, s.b0velTop, s.b0velBot, s.b0q, s.b0p);
        fillBody(s.b1resp, s.b1velTop, s.b1velBot, s.b1q, s.b1p);

        const uint32_t contactCount = (i % 13 == 7) ? 0u : (1u + rng() % 6u);
        const uint32_t aCount = rng() % 3u;
        const bool ppf = (rng() % 2) != 0;
        uint32_t solverFlags = (rng() % 2 ? 1u : 0u) | (rng() % 2 ? 2u : 0u) |
                               (rng() % 2 ? 4u : 0u) | (rng() % 2 ? 8u : 0u);
        s.cfgU = U4{ solverFlags, contactCount, aCount, ppf ? 1u : 0u };
        if (contactCount == 0) ++cEarly;
        if (aCount > 0 && contactCount) ++cFric;
        if (aCount == 2) ++cA2;
        if (ppf && aCount && contactCount) ++cPPF;

        s.invMassScale = F4{ rf(0, 1.5f), rf(0, 1.5f), rf(0, 1.5f), rf(0, 1.5f) };
        float n[3] = { rf(-1, 1), rf(-1, 1), rf(-1, 1) };
        if (i % 5 == 2) { n[0] = (rng() % 2) ? 0.9f : -0.9f; }
        const float nm = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        for (int k = 0; k < 3; ++k) n[k] /= (nm > 0 ? nm : 1.f);
        const int rMode = rng() % 4;
        const float restitution = rMode == 0 ? rf(-500.f, -1.f) : (rMode == 1 ? 0.f : rf(0.1f, 1.f));
        if (rMode == 0) ++cCompliant;
        if (rMode >= 2) ++cBounce;
        s.normalRest = F4{ n[0], n[1], n[2], restitution };
        s.mat = F4{ rf(-0.05f, 0.1f), rf(0, 50.f), rf(0, 2.f), rf(0, 2.f) };
        const float bias = (rng() % 3 == 0) ? rf(0.1f, 0.7f) : rf(0.9f, 2.f);
        if (bias < 0.8f) ++cBiasLo;
        s.scal = F4{ rf(200.f, 500.f), rf(0.002f, 0.008f), rf(-2.f, 0.5f),
                     (rng() % 3 == 0) ? 0.f : rf(0.f, 0.02f) };
        s.scal2 = F4{ rf(30.f, 120.f), bias, rf(0.008f, 0.033f), 0 };
        // torsional: only meaningful at aCount == 1
        const bool wantTors = (i % 4 == 1);
        s.scal3 = F4{ wantTors ? rf(0.f, 2.f) : 0.f, wantTors ? rf(0.f, 0.5f) : 0.f, 0, 0 };
        if (wantTors && (s.scal3.x != 0.f || s.scal3.y != 0.f) && aCount == 1 && contactCount)
            ++cTors;

        if (i % 17 == 5)
        {
            s.b1velBot.x = s.b0velBot.x; s.b1velBot.y = s.b0velBot.y; s.b1velBot.z = s.b0velBot.z;
            ++cZeroLNV;
        }
        if (i % 19 == 9)
        {
            const float kk = rf(0.5f, 2.f);
            s.b1velBot.x = s.b0velBot.x - n[0] * kk;
            s.b1velBot.y = s.b0velBot.y - n[1] * kk;
            s.b1velBot.z = s.b0velBot.z - n[2] * kk;
            ++cFall;
        }

        for (int c = 0; c < 12; ++c)
            s.cPts[c] = F4{ rf(-3, 3), rf(-3, 3), rf(-3, 3),
                            (c & 1) ? rf(0.1f, 1e4f) : rf(-0.2f, 0.2f) };
        // exact-zero contact target velocities: the dominant production path,
        // and the ONLY way to reach the precedence trap's 0.f arm
        // (cTargetVel + (isGreater2?1:0) == 0 requires cTargetVel == 0 exactly)
        if (i % 7 == 3)
        {
            for (int c = 1; c < 12; c += 2)
                s.cPts[c].x = s.cPts[c].y = s.cPts[c].z = 0.f;
            if (contactCount)
                ++cZeroTV;
        }
        for (int a = 0; a < 2; ++a)
        {
            s.anchors0[a] = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), 0 };
            s.anchors1[a] = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), 0 };
        }
        s.contactIDs = U4{ contactCount ? rng() % contactCount : 0u,
                           contactCount ? rng() % contactCount : 0u, 0, 0 };

        in[i] = s;
        refFinalizeTGS(s, ref[i]);
    }

    std::printf("[ info ] tz cov: early %zu compliant %zu bounce %zu tors %zu fric %zu fall %zu "
                "zeroLNV %zu kin %zu ppf %zu a2 %zu biasLo %zu zeroTV %zu\n",
                cEarly, cCompliant, cBounce, cTors, cFric, cFall, cZeroLNV, cKin, cPPF, cA2,
                cBiasLo, cZeroTV);

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<EbIn> dummyIn(1);
    std::vector<EbOut> dummyOut(1);
    std::vector<FzIn> dummyIn2(1);
    std::vector<FzOut> dummyOut2(1);
    std::vector<TzOut> outInit(N);
    PxgBufferHandle bDIn = createBufferFrom(dev, dummyIn);
    PxgBufferHandle bDOut = createBufferFrom(dev, dummyOut);
    PxgBufferHandle bDIn2 = createBufferFrom(dev, dummyIn2);
    PxgBufferHandle bDOut2 = createBufferFrom(dev, dummyOut2);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npArtiExtBody.spv", "aebFinalizeTGS",
             { (uint32_t)sizeof(EbIn), (uint32_t)sizeof(EbOut), (uint32_t)sizeof(FzIn),
               (uint32_t)sizeof(FzOut), (uint32_t)sizeof(TzIn), (uint32_t)sizeof(TzOut) },
             sizeof(P));
    k.dispatch(GROUPS, { bDIn, bDOut, bDIn2, bDOut2, bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<TzOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const TzOut& r = ref[i]; const TzOut& q = g[i];
        bool ok = std::memcmp(&r.hdrU, &q.hdrU, sizeof(U4) * 2) == 0;
        const float* rf2 = &r.invMassOut.x;
        const float* qf = &q.invMassOut.x;
        const size_t nf = (sizeof(TzOut) - 2 * sizeof(U4)) / 4;
        for (size_t x = 0; x < nf && ok; ++x)
            ok = eqFZ(rf2[x], qf[x]);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u wmR %u wmG %u c0R (%g %g %g %g) c0G (%g %g %g %g)\n", i,
                            r.writeMask.x, q.writeMask.x, r.cOut[0].x, r.cOut[0].y, r.cOut[0].z,
                            r.cOut[0].w, q.cOut[0].x, q.cOut[0].y, q.cOut[0].z, q.cOut[0].w);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "aebFinalizeTGS: %zu mismatches", mm);

    const bool cov = cEarly > 150 && cCompliant > 500 && cBounce > 800 && cTors > 100 &&
                     cFric > 1000 && cFall > 100 && cZeroLNV > 100 && cKin > 1000 && cPPF > 400 &&
                     cA2 > 700 && cBiasLo > 500;
    PXG_CHECK_MSG(cov, "aebFinalizeTGS: coverage thin (early %zu compliant %zu tors %zu fric %zu)",
                  cEarly, cCompliant, cTors, cFric);

    for (PxgBufferHandle h : { bDIn, bDOut, bDIn2, bDOut2, bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
