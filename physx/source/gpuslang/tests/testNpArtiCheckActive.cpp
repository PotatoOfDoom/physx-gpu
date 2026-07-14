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

// Verification of npArtiCheckActive.slang: the two light articulation
// active-contact predicates ceacPGS (checkExtActiveContactBlock, solver.cuh
// L322-426) and ceacTGS (checkExtActiveContactBlockTGS, solver.cuh L737-831),
// against exact CPU transcriptions.  The shader exposes per-contact deltaF for
// strong bit-exact verification (velocity is not propagated, so each contact's
// deltaF is independent) plus the faithful early-return active bool (any
// |deltaF| > 1e-8).  Floats eqFZ; the active lane exact.

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

static const float ACTIVE_EPS = 1.0e-8f;
static const uint32_t CONTACT_FLAG_ACCEL_SPRING = 1u << 3;

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b) || (a == 0.f && b == 0.f); }
static float dot3(const float a[3], const float b[3]) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
static float pxMaxR(float a, float b) { return (a > b) ? a : b; }
static float pxMinR(float a, float b) { return (a < b) ? a : b; }

// mirrors computeContactCoefficients() (PGS path)
static void rContactCoeffs(uint32_t flags, float restitution, float unitResponse,
                           float recipResponse, float targetVelocity, float coeff0, float coeff1,
                           float& velMultiplier, float& impulseMultiplier, float& unbiasedErr,
                           float& biasedErr)
{
    if (restitution < 0.f)
    {
        const bool accelSpring = (flags & CONTACT_FLAG_ACCEL_SPRING) != 0;
        const float massIfAccelElseOne = accelSpring ? recipResponse : 1.f;
        const float oneIfAccelElseR = accelSpring ? 1.f : unitResponse;
        const float a = coeff0;
        const float b = coeff1 * massIfAccelElseOne;
        const float x = 1.f / (a * oneIfAccelElseR + 1.f);
        const float scaledBias = x * b;
        velMultiplier = x * a * massIfAccelElseOne;
        impulseMultiplier = 1.f - x;
        unbiasedErr = biasedErr = targetVelocity * velMultiplier - scaledBias;
    }
    else
    {
        velMultiplier = recipResponse;
        biasedErr = coeff0 * velMultiplier;
        unbiasedErr = coeff1 * velMultiplier;
        impulseMultiplier = 1.f;
    }
}

// runtime a-coefficient overload of computeCompliantContactCoefficientsTGS
static void rCompliantTGS(uint32_t flags, float nrdt, float unitResponse, float recipResponse,
                          float a, float& velMultiplier, float& scaledBias)
{
    const bool accelSpring = (flags & CONTACT_FLAG_ACCEL_SPRING) != 0;
    const float massIfAccelElseOne = accelSpring ? recipResponse : 1.f;
    const float oneIfAccelElseR = accelSpring ? 1.f : unitResponse;
    const float x = 1.f / (a * oneIfAccelElseR + 1.f);
    velMultiplier = x * a * massIfAccelElseOne;
    scaledBias = nrdt * x * oneIfAccelElseR;
}

struct CeacOut { F4 dF[2]; U4 active; };

// ============================ PGS ============================

struct CPRow { F4 raXn_tv, rbXn_maxI, crm, crm2; };
struct CeacPgsIn
{
    U4 cfg; F4 hdr, normalSF;
    F4 vel0Top, vel0Bot, vel1Top, vel1Bot;
    CPRow contacts[6];
};

// exact transcription of checkExtActiveContactBlock (solver.cuh L322-426)
static void refCeacPGS(const CeacPgsIn& s, CeacOut& o, bool& active)
{
    std::memset(&o, 0, sizeof(o));
    const uint32_t numNormalConstr = s.cfg.x < 6u ? s.cfg.x : 6u;
    const uint32_t flags = s.cfg.y;
    const float restitution = s.hdr.x, cfm = s.hdr.y;
    const float normal[3] = { s.normalSF.x, s.normalSF.y, s.normalSF.z };
    const float v0Top[3] = { s.vel0Top.x, s.vel0Top.y, s.vel0Top.z };
    const float v0Bot[3] = { s.vel0Bot.x, s.vel0Bot.y, s.vel0Bot.z };
    const float v1Top[3] = { s.vel1Top.x, s.vel1Top.y, s.vel1Top.z };
    const float v1Bot[3] = { s.vel1Bot.x, s.vel1Bot.y, s.vel1Bot.z };

    active = false;
    for (uint32_t i = 0; i < numNormalConstr; ++i)
    {
        const CPRow& c = s.contacts[i];
        const float raXn[3] = { c.raXn_tv.x, c.raXn_tv.y, c.raXn_tv.z };
        const float targetVelocity = c.raXn_tv.w;
        const float rbXn[3] = { c.rbXn_maxI.x, c.rbXn_maxI.y, c.rbXn_maxI.z };
        const float maxImpulse = c.rbXn_maxI.w;
        const float appliedForce = c.crm.x;
        const float resp0 = c.crm.y, resp1 = c.crm.z, coeff0 = c.crm.w, coeff1 = c.crm2.x;

        const float unitResponse = resp0 + resp1;
        const float recipResponse = (unitResponse > 0.f) ? (1.f / (unitResponse + cfm)) : 0.f;
        float velMultiplier, impulseMul, unbiasedError, biasedErr;
        rContactCoeffs(flags, restitution, unitResponse, recipResponse, targetVelocity, coeff0,
                       coeff1, velMultiplier, impulseMul, unbiasedError, biasedErr);

        const float v0_ = dot3(v0Bot, normal) + dot3(v0Top, raXn);
        const float v1_ = dot3(v1Bot, normal) + dot3(v1Top, rbXn);
        const float normalVel = v0_ - v1_;

        const float tempDeltaF = biasedErr - normalVel * velMultiplier;
        const float _deltaF = std::fmax(tempDeltaF, -appliedForce);
        const float _newForce = appliedForce * impulseMul + _deltaF;
        const float newForce = std::fmin(_newForce, maxImpulse);
        const float deltaF = newForce - appliedForce;

        (&o.dF[i / 4].x)[i % 4] = deltaF;
        if (std::fabs(deltaF) > ACTIVE_EPS)
            active = true;
    }
    o.active.x = active ? 1u : 0u;
}

PXG_TEST(npArtiCheckActive_pgs)
{
    std::mt19937 rng(189001);
    const uint32_t N = 3000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto rv3 = [&](float a, float b) { return F4{ rf(a, b), rf(a, b), rf(a, b), 0.f }; };

    std::vector<CeacPgsIn> in(N);
    std::vector<CeacOut> ref(N);
    size_t cActive = 0, cInactive = 0, cCompliant = 0, cAccel = 0, cZeroResp = 0, cNc0 = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        CeacPgsIn s{};
        const uint32_t numNormal = (i % 17 == 3) ? 0u : (1u + rng() % 6u);
        if (numNormal == 0) ++cNc0;
        // quiescent stripe: zero vel + appliedForce + coeff0, restitution>=0,
        // large maxImpulse -> deltaF == 0 -> inactive branch of the bool
        const bool quiescent = (i % 9 == 4);
        const bool zeroResp = (i % 13 == 5);
        const uint32_t accelBit = (rng() % 2) ? CONTACT_FLAG_ACCEL_SPRING : 0u;
        const bool compliant = !quiescent && (rng() % 3 == 0);
        const float restitution = compliant ? rf(-2.f, -0.1f) : rf(0.f, 1.f);
        if (compliant) ++cCompliant;
        if (compliant && accelBit) ++cAccel;
        if (zeroResp && numNormal) ++cZeroResp;
        s.cfg = U4{ numNormal, accelBit, 0u, 0u };
        s.hdr = F4{ restitution, zeroResp ? 0.f : rf(0.f, 0.05f), 0.f, 0.f };

        float n[3] = { rf(-1, 1), rf(-1, 1), rf(-1, 1) };
        const float nm = std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
        for (int k = 0; k < 3; ++k) n[k] /= (nm > 0 ? nm : 1.f);
        s.normalSF = F4{ n[0], n[1], n[2], 0.f };

        s.vel0Top = quiescent ? F4{0,0,0,0} : rv3(-3, 3);
        s.vel0Bot = quiescent ? F4{0,0,0,0} : rv3(-3, 3);
        s.vel1Top = quiescent ? F4{0,0,0,0} : rv3(-3, 3);
        s.vel1Bot = quiescent ? F4{0,0,0,0} : rv3(-3, 3);

        for (uint32_t c = 0; c < numNormal; ++c)
        {
            const bool tight = (rng() % 4 == 0);
            const float cr0 = zeroResp ? 0.f : rf(0.05f, 2.f);
            const float cr1 = zeroResp ? 0.f : rf(0.05f, 2.f);
            const float applied = quiescent ? 0.f : rf(-0.05f, 0.2f);
            const float coeff0 = quiescent ? 0.f : rf(-1.f, 1.f);
            s.contacts[c].raXn_tv = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), rf(-2, 2) };
            s.contacts[c].rbXn_maxI = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1),
                                          quiescent ? rf(1.f, 1e3f) : (tight ? rf(0.f, 0.05f) : rf(0.5f, 1e3f)) };
            s.contacts[c].crm = F4{ applied, cr0, cr1, coeff0 };
            s.contacts[c].crm2 = F4{ rf(-1, 1), 0, 0, 0 };
        }

        in[i] = s;
        bool act;
        refCeacPGS(s, ref[i], act);
        if (numNormal) { if (act) ++cActive; else ++cInactive; }
    }

    std::printf("[ info ] ceacPGS cov: active %zu inactive %zu compliant %zu accel %zu zeroResp %zu nc0 %zu\n",
                cActive, cInactive, cCompliant, cAccel, cZeroResp, cNc0);
    PXG_CHECK(cActive > 800);
    PXG_CHECK(cInactive > 100);
    PXG_CHECK(cCompliant > 700);
    PXG_CHECK(cAccel > 300);
    PXG_CHECK(cZeroResp > 80);
    PXG_CHECK(cNc0 > 80);

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<CeacOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);
    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npArtiCheckActive.spv", "ceacPGS",
             { (uint32_t)sizeof(CeacPgsIn), (uint32_t)sizeof(CeacOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<CeacOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        bool ok = std::memcmp(&ref[i].active, &g[i].active, sizeof(U4)) == 0;
        for (int x = 0; x < 8 && ok; ++x)
            ok = eqFZ((&ref[i].dF[0].x)[x], (&g[i].dF[0].x)[x]);
        if (!ok && mm++ < 5)
            std::printf("MM %u actR %u actG %u dF0R %g dF0G %g\n", i, ref[i].active.x, g[i].active.x,
                        ref[i].dF[0].x, g[i].dF[0].x);
    }
    PXG_CHECK_MSG(mm == 0, "ceacPGS outputs bitwise-identical (mod +-0)");
}

// ============================ TGS ============================

struct CTRow { F4 raXn_extra, rbXn_tv, misc, resp; };
struct CeacTgsIn
{
    U4 cfg; F4 hdr, hdr2, normalSF;
    F4 d0Top, d0Bot, d1Top, d1Bot;
    F4 vel0Top, vel0Bot, vel1Top, vel1Bot;
    CTRow contacts[6];
};

// exact transcription of checkExtActiveContactBlockTGS (solver.cuh L737-831)
static void refCeacTGS(const CeacTgsIn& s, CeacOut& o, bool& active)
{
    std::memset(&o, 0, sizeof(o));
    const uint32_t numNormalConstr = s.cfg.x < 6u ? s.cfg.x : 6u;
    const uint32_t flags = s.cfg.y;
    const float restitutionXdt = s.hdr.x, cfm = s.hdr.y, maxPenBias = s.hdr.z, minPen = s.hdr.w;
    const float elapsedTime = s.hdr2.x;
    const float normal[3] = { s.normalSF.x, s.normalSF.y, s.normalSF.z };
    const float v0Top[3] = { s.vel0Top.x, s.vel0Top.y, s.vel0Top.z };
    const float v0Bot[3] = { s.vel0Bot.x, s.vel0Bot.y, s.vel0Bot.z };
    const float v1Top[3] = { s.vel1Top.x, s.vel1Top.y, s.vel1Top.z };
    const float v1Bot[3] = { s.vel1Bot.x, s.vel1Bot.y, s.vel1Bot.z };
    const float d0Top[3] = { s.d0Top.x, s.d0Top.y, s.d0Top.z };
    const float d0Bot[3] = { s.d0Bot.x, s.d0Bot.y, s.d0Bot.z };
    const float d1Top[3] = { s.d1Top.x, s.d1Top.y, s.d1Top.z };
    const float d1Bot[3] = { s.d1Bot.x, s.d1Bot.y, s.d1Bot.z };

    const float relMotion[3] = { d0Bot[0]-d1Bot[0], d0Bot[1]-d1Bot[1], d0Bot[2]-d1Bot[2] };
    const float deltaV = dot3(normal, relMotion);

    active = false;
    for (uint32_t i = 0; i < numNormalConstr; ++i)
    {
        const CTRow& c = s.contacts[i];
        const float raXn[3] = { c.raXn_extra.x, c.raXn_extra.y, c.raXn_extra.z };
        const float compliantContactCoef = c.raXn_extra.w;
        const float rbXn[3] = { c.rbXn_tv.x, c.rbXn_tv.y, c.rbXn_tv.z };
        const float targetVel = c.rbXn_tv.w;
        const float appliedForce = c.misc.x, separation = c.misc.y, maxImpulse = c.misc.z;
        float biasCoefficient = c.misc.w;
        const float resp0 = c.resp.x, resp1 = c.resp.y;

        const float unitResponse = resp0 + resp1;
        const float recipResponse = (unitResponse > 0.f) ? (1.f / (unitResponse + cfm)) : 0.f;
        float velMultiplier = recipResponse;
        if (restitutionXdt < 0.f)
            rCompliantTGS(flags, restitutionXdt, unitResponse, recipResponse, compliantContactCoef,
                          velMultiplier, biasCoefficient);

        const float v0 = dot3(v0Top, raXn) + dot3(v0Bot, normal);
        const float v1 = dot3(v1Top, rbXn) + dot3(v1Bot, normal);
        const float normalVel = v0 - v1;

        const float deltaBias =
            deltaV + dot3(d0Top, raXn) - dot3(d1Top, rbXn) - targetVel * elapsedTime;
        const float sep = pxMaxR(minPen, separation + deltaBias);
        const float biased = pxMinR(-maxPenBias, biasCoefficient * sep);
        const float tVelBias = recipResponse * biased;

        const float tempDeltaF = tVelBias - (normalVel - targetVel) * velMultiplier;
        const float _deltaF = std::fmax(tempDeltaF, -appliedForce);
        const float _newForce = appliedForce + _deltaF;
        const float newForce = std::fmin(_newForce, maxImpulse);
        const float deltaF = newForce - appliedForce;

        (&o.dF[i / 4].x)[i % 4] = deltaF;
        if (std::fabs(deltaF) > ACTIVE_EPS)
            active = true;
    }
    o.active.x = active ? 1u : 0u;
}

PXG_TEST(npArtiCheckActive_tgs)
{
    std::mt19937 rng(189777);
    const uint32_t N = 3000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto rv3 = [&](float a, float b) { return F4{ rf(a, b), rf(a, b), rf(a, b), 0.f }; };

    std::vector<CeacTgsIn> in(N);
    std::vector<CeacOut> ref(N);
    size_t cActive = 0, cInactive = 0, cCompliant = 0, cAccel = 0, cZeroResp = 0, cNc0 = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        CeacTgsIn s{};
        const uint32_t numNormal = (i % 17 == 3) ? 0u : (1u + rng() % 6u);
        if (numNormal == 0) ++cNc0;
        // quiescent stripe: zero vel/delta + appliedForce + maxPenBias, targetVel 0,
        // separation>=0 so sep>=0 -> biased=min(0,+)=0 -> tVelBias 0 -> deltaF 0
        const bool quiescent = (i % 9 == 4);
        const bool zeroResp = (i % 13 == 5);
        const uint32_t accelBit = (rng() % 2) ? CONTACT_FLAG_ACCEL_SPRING : 0u;
        const bool compliant = !quiescent && (rng() % 3 == 0);
        const float restitutionXdt = compliant ? rf(-2.f, -0.1f) : rf(0.f, 1.f);
        if (compliant) ++cCompliant;
        if (compliant && accelBit) ++cAccel;
        if (zeroResp && numNormal) ++cZeroResp;
        s.cfg = U4{ numNormal, accelBit, 0u, 0u };
        s.hdr = F4{ restitutionXdt, zeroResp ? 0.f : rf(0.f, 0.05f),
                    quiescent ? 0.f : rf(0.02f, 2.f), rf(-2.f, -0.001f) };
        s.hdr2 = F4{ rf(0.f, 0.05f), 0.f, 0.f, 0.f };

        float n[3] = { rf(-1, 1), rf(-1, 1), rf(-1, 1) };
        const float nm = std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
        for (int k = 0; k < 3; ++k) n[k] /= (nm > 0 ? nm : 1.f);
        s.normalSF = F4{ n[0], n[1], n[2], 0.f };

        s.d0Top = quiescent ? F4{0,0,0,0} : rv3(-0.3f, 0.3f);
        s.d0Bot = quiescent ? F4{0,0,0,0} : rv3(-0.3f, 0.3f);
        s.d1Top = quiescent ? F4{0,0,0,0} : rv3(-0.3f, 0.3f);
        s.d1Bot = quiescent ? F4{0,0,0,0} : rv3(-0.3f, 0.3f);
        s.vel0Top = quiescent ? F4{0,0,0,0} : rv3(-3, 3);
        s.vel0Bot = quiescent ? F4{0,0,0,0} : rv3(-3, 3);
        s.vel1Top = quiescent ? F4{0,0,0,0} : rv3(-3, 3);
        s.vel1Bot = quiescent ? F4{0,0,0,0} : rv3(-3, 3);

        for (uint32_t c = 0; c < numNormal; ++c)
        {
            const bool tight = (rng() % 4 == 0);
            const float cr0 = zeroResp ? 0.f : rf(0.05f, 2.f);
            const float cr1 = zeroResp ? 0.f : rf(0.05f, 2.f);
            const float applied = quiescent ? 0.f : rf(-0.05f, 0.2f);
            const float targetVel = quiescent ? 0.f : rf(-2.f, 2.f);
            const float separation = quiescent ? rf(0.f, 1.f) : rf(-0.2f, 0.2f);
            const float biasCoef = quiescent ? rf(0.f, 4.f) : rf(0.f, 4.f);
            s.contacts[c].raXn_extra = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), rf(0.f, 2.f) };
            s.contacts[c].rbXn_tv = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), targetVel };
            s.contacts[c].misc = F4{ applied, separation,
                                     quiescent ? rf(1.f, 1e3f) : (tight ? rf(0.f, 0.05f) : rf(0.5f, 1e3f)),
                                     biasCoef };
            s.contacts[c].resp = F4{ cr0, cr1, 0.f, 0.f };
        }

        in[i] = s;
        bool act;
        refCeacTGS(s, ref[i], act);
        if (numNormal) { if (act) ++cActive; else ++cInactive; }
    }

    std::printf("[ info ] ceacTGS cov: active %zu inactive %zu compliant %zu accel %zu zeroResp %zu nc0 %zu\n",
                cActive, cInactive, cCompliant, cAccel, cZeroResp, cNc0);
    PXG_CHECK(cActive > 800);
    PXG_CHECK(cInactive > 100);
    PXG_CHECK(cCompliant > 700);
    PXG_CHECK(cAccel > 300);
    PXG_CHECK(cZeroResp > 80);
    PXG_CHECK(cNc0 > 80);

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    // ceacTGS uses bindings 2/3 (ceacPGS occupies 0/1 in the shared module);
    // fill 0/1 with dummies and pass the real buffers at 2/3.
    std::vector<CeacPgsIn> dummyIn(1);
    std::vector<CeacOut> dummyOut(1);
    std::vector<CeacOut> outInit(N);
    PxgBufferHandle bDIn = createBufferFrom(dev, dummyIn);
    PxgBufferHandle bDOut = createBufferFrom(dev, dummyOut);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);
    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npArtiCheckActive.spv", "ceacTGS",
             { (uint32_t)sizeof(CeacPgsIn), (uint32_t)sizeof(CeacOut),
               (uint32_t)sizeof(CeacTgsIn), (uint32_t)sizeof(CeacOut) }, sizeof(P));
    k.dispatch(GROUPS, { bDIn, bDOut, bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<CeacOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        bool ok = std::memcmp(&ref[i].active, &g[i].active, sizeof(U4)) == 0;
        for (int x = 0; x < 8 && ok; ++x)
            ok = eqFZ((&ref[i].dF[0].x)[x], (&g[i].dF[0].x)[x]);
        if (!ok && mm++ < 5)
            std::printf("MM %u actR %u actG %u dF0R %g dF0G %g\n", i, ref[i].active.x, g[i].active.x,
                        ref[i].dF[0].x, g[i].dF[0].x);
    }
    PXG_CHECK_MSG(mm == 0, "ceacTGS outputs bitwise-identical (mod +-0)");
}

}  // namespace
