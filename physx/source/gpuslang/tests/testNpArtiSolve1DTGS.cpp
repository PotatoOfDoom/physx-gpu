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

// Verification of npArtiSolve1DTGS.slang (as1dTGS): solveExt1DBlockTGS
// (solverBlockTGS.cuh L932-1089), the TGS articulation ext-body 1D solve,
// against an exact CPU transcription.  All outputs bitwise modulo the
// documented +-0 class (eqFZ).

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

static const uint32_t SC_SPRING = 1u << 5;
static const uint32_t SC_INEQUALITY = 1u << 6;
static const uint32_t SC_ACCEL = 1u << 7;
static const float DY_MIN_RESPONSE = 1e-12f;
static const float PX_MAX_F32 = 3.402823466e+38f;

struct TRow
{
    F4 lin0_c0, lin1_c1, ang0_c2, ang1_c3, misc, misc2;
    U4 flagsU;
    F4 dRA, dRA2, dRB;
};

struct At1In
{
    U4 cfg;
    F4 hdr, refs, rAWorld, rBWorld, deltaQ0, deltaQ1;
    F4 motion0Top, motion0Bot, motion1Top, motion1Bot;
    F4 vel0Top, vel0Bot, vel1Top, vel1Bot;
    TRow rows[20];
};

struct At1Out
{
    F4 vel0Top, vel0Bot, vel1Top, vel1Bot;
    F4 imp0, imp0b, imp1, imp1b;
    F4 appliedForce[5];
    F4 residual[5];
};

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b) || (a == 0.f && b == 0.f); }
static float dot3(const float a[3], const float b[3]) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }

static void rQuatRot(const F4& q, const float v[3], float out[3])
{
    const float vx = 2.f*v[0], vy = 2.f*v[1], vz = 2.f*v[2];
    const float w2 = q.w*q.w - 0.5f;
    const float dot2 = q.x*vx + q.y*vy + q.z*vz;
    out[0] = vx*w2 + (q.y*vz - q.z*vy)*q.w + q.x*dot2;
    out[1] = vy*w2 + (q.z*vx - q.x*vz)*q.w + q.y*dot2;
    out[2] = vz*w2 + (q.x*vy - q.y*vx)*q.w + q.z*dot2;
}

static float rRecipUnitResponse(float u, float minR) { return u <= minR ? 0.f : 1.f / u; }

static float rMinBiasTGS(uint32_t flags, float maxBias)
{
    return -((flags & SC_INEQUALITY) != 0 ? PX_MAX_F32 : maxBias);
}

static float rResolvedGeomErrTGS(const float dLin0[3], const float dLin1[3], const float cLin0[3],
                                 const float cLin1[3], const float dAng0[3], const float dAng1[3],
                                 const float cAng0[3], const float cAng1[3], float angErrScale,
                                 bool isSpring, float targetVel, float elapsedTime)
{
    const float deltaAng = (dot3(cAng0, dAng0) - dot3(cAng1, dAng1)) * angErrScale;
    const float deltaLin = dot3(cLin0, dLin0) - dot3(cLin1, dLin1);
    const float motion = deltaLin + deltaAng;
    return isSpring ? motion : motion - targetVel * elapsedTime;
}

static void rConstantsTGS(bool isSpring, bool isAccel, float geomErr, float unitResponse,
                          float recipUnitResponse, float coeff0, float coeff1, float coeff2,
                          float coeff3, float& biasScale, float& velMultiplier, float& error,
                          float& targetVel)
{
    const float initJointSpeed = coeff0;
    biasScale = coeff1;
    if (isSpring)
    {
        const float a = coeff2, b = coeff3;
        if (isAccel)
        {
            biasScale *= recipUnitResponse;
            error = biasScale * geomErr;
            const float x = 1.f / (1.f + a);
            velMultiplier = -x * a * recipUnitResponse;
            targetVel = x * b * recipUnitResponse - velMultiplier * initJointSpeed;
        }
        else
        {
            const float x = 1.f / (1.f + a * unitResponse);
            biasScale *= x;
            error = biasScale * geomErr;
            velMultiplier = -x * a;
            targetVel = x * b - velMultiplier * initJointSpeed;
        }
    }
    else
    {
        const float bounceVel = coeff2, velocityTarget = coeff3;
        if (bounceVel != 0.f)
        {
            biasScale = 0.f;
            error = 0.f;
            velMultiplier = -1.f;
            targetVel = bounceVel - velMultiplier * initJointSpeed;
        }
        else
        {
            error = geomErr * biasScale;
            velMultiplier = -1.f;
            targetVel = velocityTarget - velMultiplier * initJointSpeed;
        }
    }
}

struct RefStats { bool springNA, springA, bounce, hard, ineq, clampImp, biasZero; };

// exact transcription of solveExt1DBlockTGS (solverBlockTGS.cuh L932-1089)
static void refAs1dTGS(const At1In& s, At1Out& o, RefStats& st)
{
    std::memset(&o, 0, sizeof(o));
    st = RefStats{ false, false, false, false, false, false, false };

    const uint32_t rowCount = s.cfg.x < 20u ? s.cfg.x : 20u;
    const bool residualReportingEnabled = s.cfg.y != 0u;
    const float ref0 = s.refs.x, ref1 = s.refs.y;
    const float cfm = s.hdr.z, elapsedTime = s.hdr.w;

    const float invMass0 = ref0 * s.rAWorld.w;
    const float invMass1 = ref1 * s.rBWorld.w;
    const float invInertiaScale0 = ref0 * s.hdr.x;
    const float invInertiaScale1 = ref1 * s.hdr.y;

    const float raPrev[3] = { s.rAWorld.x, s.rAWorld.y, s.rAWorld.z };
    const float rbPrev[3] = { s.rBWorld.x, s.rBWorld.y, s.rBWorld.z };
    float ra[3], rb[3];
    rQuatRot(s.deltaQ0, raPrev, ra);
    rQuatRot(s.deltaQ1, rbPrev, rb);
    const float m0b[3] = { s.motion0Bot.x, s.motion0Bot.y, s.motion0Bot.z };
    const float m1b[3] = { s.motion1Bot.x, s.motion1Bot.y, s.motion1Bot.z };
    const float m0t[3] = { s.motion0Top.x, s.motion0Top.y, s.motion0Top.z };
    const float m1t[3] = { s.motion1Top.x, s.motion1Top.y, s.motion1Top.z };
    float raMotion[3], rbMotion[3];
    for (int k = 0; k < 3; ++k) { raMotion[k] = (ra[k] + m0b[k]) - raPrev[k]; rbMotion[k] = (rb[k] + m1b[k]) - rbPrev[k]; }

    float linVel0[3] = { s.vel0Bot.x, s.vel0Bot.y, s.vel0Bot.z };
    float linVel1[3] = { s.vel1Bot.x, s.vel1Bot.y, s.vel1Bot.z };
    float angVel0[3] = { s.vel0Top.x, s.vel0Top.y, s.vel0Top.z };
    float angVel1[3] = { s.vel1Top.x, s.vel1Top.y, s.vel1Top.z };
    float li0[3] = {0,0,0}, li1[3] = {0,0,0}, ai0[3] = {0,0,0}, ai1[3] = {0,0,0};

    for (uint32_t i = 0; i < rowCount; ++i)
    {
        const TRow& r = s.rows[i];
        const float clinVel0[3] = { r.lin0_c0.x, r.lin0_c0.y, r.lin0_c0.z };
        const float clinVel1[3] = { r.lin1_c1.x, r.lin1_c1.y, r.lin1_c1.z };
        const float cangVel0[3] = { r.ang0_c2.x, r.ang0_c2.y, r.ang0_c2.z };
        const float cangVel1[3] = { r.ang1_c3.x, r.ang1_c3.y, r.ang1_c3.z };
        const uint32_t flags = r.flagsU.x;
        const bool isSpring = (flags & SC_SPRING) != 0;
        const bool isAccel = (flags & SC_ACCEL) != 0;

        const float resp0 = ref0 * r.misc2.x;
        const float resp1 = ref1 * r.misc2.y;
        float unitResponse = resp0 + resp1;
        unitResponse += cfm;
        const float recipResponse = rRecipUnitResponse(unitResponse, DY_MIN_RESPONSE);

        const float coeff0 = r.lin0_c0.w, coeff1 = r.lin1_c1.w, coeff2 = r.ang0_c2.w, coeff3 = r.ang1_c3.w;
        const float geometricError = r.misc.x;

        float biasScale, velMultiplier, initBias, targetVel;
        rConstantsTGS(isSpring, isAccel, geometricError, unitResponse, recipResponse, coeff0, coeff1,
                      coeff2, coeff3, biasScale, velMultiplier, initBias, targetVel);
        if (isSpring && isAccel) st.springA = true;
        else if (isSpring) st.springNA = true;
        else if (coeff2 != 0.f) st.bounce = true;
        else st.hard = true;
        if (biasScale == 0.f) st.biasZero = true;

        initBias = (biasScale == 0.f) ? 0.f : initBias;

        const float angularErrorScale = r.misc.z;
        const float errorDelta = rResolvedGeomErrTGS(raMotion, rbMotion, clinVel0, clinVel1, m0t,
                                                     m1t, cangVel0, cangVel1, angularErrorScale,
                                                     isSpring, targetVel, elapsedTime);
        const float maxBias = r.misc.y;
        const float vMul = isSpring ? velMultiplier : recipResponse * velMultiplier;
        const float appliedForce = r.misc.w;

        const float unclampedBias = initBias + errorDelta * biasScale;
        const float minBias = rMinBiasTGS(flags, maxBias);
        if (flags & SC_INEQUALITY) st.ineq = true;
        const float bias = std::fmin(std::fmax(unclampedBias, minBias), maxBias);

        const float constant = isSpring ? (bias + targetVel) : recipResponse * (bias + targetVel);
        const float maxImpulse = r.misc2.z, minImpulse = r.misc2.w;

        const float v0 = dot3(linVel0, clinVel0) + dot3(angVel0, cangVel0);
        const float v1 = dot3(linVel1, clinVel1) + dot3(angVel1, cangVel1);
        const float normalVel = v0 - v1;

        const float unclampedForce = appliedForce + (vMul * normalVel + constant);
        const float clampedForce = std::fmin(maxImpulse, std::fmax(minImpulse, unclampedForce));
        const float deltaF = clampedForce - appliedForce;
        if (clampedForce == maxImpulse || clampedForce == minImpulse) st.clampImp = true;

        (&o.appliedForce[i / 4].x)[i % 4] = clampedForce;
        if (residualReportingEnabled)
            (&o.residual[i / 4].x)[i % 4] = (vMul == 0.f) ? 0.f : deltaF / vMul;

        const float dRALin[3] = { r.dRA.x, r.dRA.y, r.dRA.z };
        const float dRAAng[3] = { r.dRA.w, r.dRA2.x, r.dRA2.y };
        const float dRBLin[3] = { r.dRA2.z, r.dRA2.w, r.dRB.x };
        const float dRBAng[3] = { r.dRB.y, r.dRB.z, r.dRB.w };
        for (int k = 0; k < 3; ++k)
        {
            li0[k] = clinVel0[k] * deltaF + li0[k];
            ai0[k] = cangVel0[k] * deltaF + ai0[k];
            li1[k] = clinVel1[k] * deltaF + li1[k];
            ai1[k] = cangVel1[k] * deltaF + ai1[k];
            linVel0[k] = (ref0 * dRALin[k]) * deltaF + linVel0[k];
            angVel0[k] = (ref0 * dRAAng[k]) * deltaF + angVel0[k];
            linVel1[k] = (ref1 * dRBLin[k]) * deltaF + linVel1[k];
            angVel1[k] = (ref1 * dRBAng[k]) * deltaF + angVel1[k];
        }
    }

    o.vel0Top = F4{ angVel0[0], angVel0[1], angVel0[2], 0.f };
    o.vel0Bot = F4{ linVel0[0], linVel0[1], linVel0[2], 0.f };
    o.vel1Top = F4{ angVel1[0], angVel1[1], angVel1[2], 0.f };
    o.vel1Bot = F4{ linVel1[0], linVel1[1], linVel1[2], 0.f };
    o.imp0 = F4{ li0[0]*invMass0, li0[1]*invMass0, li0[2]*invMass0, 0.f };
    o.imp0b = F4{ ai0[0]*invInertiaScale0, ai0[1]*invInertiaScale0, ai0[2]*invInertiaScale0, 0.f };
    o.imp1 = F4{ li1[0]*invMass1, li1[1]*invMass1, li1[2]*invMass1, 0.f };
    o.imp1b = F4{ ai1[0]*invInertiaScale1, ai1[1]*invInertiaScale1, ai1[2]*invInertiaScale1, 0.f };
}

PXG_TEST(npArtiSolve1DTGS_pgs)
{
    std::mt19937 rng(187001);
    const uint32_t N = 3000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto rq = [&]() {
        float a[4]; float m2 = 0;
        for (int k = 0; k < 4; ++k) { a[k] = rf(-1, 1); m2 += a[k]*a[k]; }
        const float inv = 1.f / std::sqrt(m2 > 0 ? m2 : 1.f);
        return F4{ a[0]*inv, a[1]*inv, a[2]*inv, a[3]*inv };
    };

    std::vector<At1In> in(N);
    std::vector<At1Out> ref(N);
    size_t cRows0 = 0, cSpringNA = 0, cSpringA = 0, cBounce = 0, cHard = 0, cIneq = 0, cClampImp = 0,
           cBiasZero = 0, cResid = 0, cRefScale = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        At1In s{};
        const uint32_t rowCount = (i % 11 == 4) ? 0u : (1u + rng() % 12u);
        if (rowCount == 0) ++cRows0;
        const bool resid = (rng() % 2) != 0;
        if (resid) ++cResid;
        s.cfg = U4{ rowCount, resid ? 1u : 0u, 0, 0 };
        const bool refScale = (i % 5 == 2);
        if (refScale) ++cRefScale;
        s.hdr = F4{ rf(0.f, 2.f), rf(0.f, 2.f), rf(0.f, 0.05f), rf(0.001f, 0.02f) };
        s.refs = F4{ refScale ? rf(0.3f, 1.5f) : 1.f, refScale ? rf(0.3f, 1.5f) : 1.f, 0.f, 0.f };
        s.rAWorld = F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), rf(0.f, 2.f) };
        s.rBWorld = F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), rf(0.f, 2.f) };
        s.deltaQ0 = rq(); s.deltaQ1 = rq();
        s.motion0Top = F4{ rf(-0.1f, 0.1f), rf(-0.1f, 0.1f), rf(-0.1f, 0.1f), 0 };
        s.motion0Bot = F4{ rf(-0.1f, 0.1f), rf(-0.1f, 0.1f), rf(-0.1f, 0.1f), 0 };
        s.motion1Top = F4{ rf(-0.1f, 0.1f), rf(-0.1f, 0.1f), rf(-0.1f, 0.1f), 0 };
        s.motion1Bot = F4{ rf(-0.1f, 0.1f), rf(-0.1f, 0.1f), rf(-0.1f, 0.1f), 0 };
        s.vel0Top = F4{ rf(-3, 3), rf(-3, 3), rf(-3, 3), 0 };
        s.vel0Bot = F4{ rf(-3, 3), rf(-3, 3), rf(-3, 3), 0 };
        s.vel1Top = F4{ rf(-3, 3), rf(-3, 3), rf(-3, 3), 0 };
        s.vel1Bot = F4{ rf(-3, 3), rf(-3, 3), rf(-3, 3), 0 };

        for (uint32_t r = 0; r < rowCount; ++r)
        {
            const int mode = static_cast<int>(rng() % 4);  // 0 springNA, 1 springA, 2 bounce, 3 hard
            uint32_t flags = 0;
            if (mode == 0) flags = SC_SPRING;
            else if (mode == 1) flags = SC_SPRING | SC_ACCEL;
            if (rng() % 3 == 0) flags |= SC_INEQUALITY;
            // coeff2: for spring = a; for non-spring = bounceVel (mode 2 nonzero, mode 3 zero)
            const float coeff2 = (mode == 3) ? 0.f : rf(0.1f, 2.f);
            const bool tight = (rng() % 3 == 0);
            const float minI = tight ? rf(-0.02f, 0.f) : rf(-1e3f, 0.f);
            const float maxI = tight ? rf(0.f, 0.02f) : rf(0.f, 1e3f);
            s.rows[r].lin0_c0 = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), rf(-2, 2) };
            s.rows[r].lin1_c1 = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), rf(-2, 2) };
            s.rows[r].ang0_c2 = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), coeff2 };
            s.rows[r].ang1_c3 = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), rf(-2, 2) };
            s.rows[r].misc = F4{ rf(-0.5f, 0.5f), rf(0.1f, 50.f), (rng() % 2) ? 1.f : 0.f, rf(-0.1f, 0.1f) };
            s.rows[r].misc2 = F4{ rf(0.05f, 2.f), rf(0.05f, 2.f), maxI, minI };
            s.rows[r].flagsU = U4{ flags, 0, 0, 0 };
            s.rows[r].dRA = F4{ rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), rf(-0.5f, 0.5f) };
            s.rows[r].dRA2 = F4{ rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), rf(-0.5f, 0.5f) };
            s.rows[r].dRB = F4{ rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), rf(-0.5f, 0.5f) };
        }

        in[i] = s;
        RefStats st;
        refAs1dTGS(s, ref[i], st);
        if (st.springNA) ++cSpringNA;
        if (st.springA) ++cSpringA;
        if (st.bounce) ++cBounce;
        if (st.hard) ++cHard;
        if (st.ineq) ++cIneq;
        if (st.clampImp) ++cClampImp;
        if (st.biasZero) ++cBiasZero;
    }

    std::printf("[ info ] at1 cov: rows0 %zu springNA %zu springA %zu bounce %zu hard %zu ineq %zu "
                "clampImp %zu biasZero %zu resid %zu refScale %zu\n",
                cRows0, cSpringNA, cSpringA, cBounce, cHard, cIneq, cClampImp, cBiasZero, cResid,
                cRefScale);
    PXG_CHECK(cRows0 > 100);
    PXG_CHECK(cSpringNA > 800);
    PXG_CHECK(cSpringA > 800);
    PXG_CHECK(cBounce > 800);
    PXG_CHECK(cHard > 800);
    PXG_CHECK(cIneq > 800);
    PXG_CHECK(cClampImp > 800);
    PXG_CHECK(cBiasZero > 800);
    PXG_CHECK(cResid > 1000);
    PXG_CHECK(cRefScale > 400);

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<At1Out> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npArtiSolve1DTGS.spv", "as1dTGS",
             { (uint32_t)sizeof(At1In), (uint32_t)sizeof(At1Out) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<At1Out>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* rfp = &ref[i].vel0Top.x;
        const float* qfp = &g[i].vel0Top.x;
        const size_t nf = sizeof(At1Out) / 4;
        bool ok = true;
        for (size_t x = 0; x < nf && ok; ++x)
            ok = eqFZ(rfp[x], qfp[x]);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u v0tR (%g %g %g) v0tG (%g %g %g)\n", i, ref[i].vel0Top.x,
                            ref[i].vel0Top.y, ref[i].vel0Top.z, g[i].vel0Top.x, g[i].vel0Top.y,
                            g[i].vel0Top.z);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "as1dTGS outputs bitwise-identical (mod +-0)");
}

}  // namespace
