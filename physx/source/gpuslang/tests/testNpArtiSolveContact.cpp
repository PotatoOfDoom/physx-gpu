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

// Verification of npArtiSolveContact.slang (asCtPGS): solveExtContactsBlock
// (solver.cuh L54-320), the PGS articulation ext-body contact solve, against
// an exact CPU transcription.  All outputs bitwise modulo the documented +-0
// class (eqFZ).

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

static const float P8 = 0.8f;
static const float PX_EPS_REAL = 1.192092896e-07f;
static const uint32_t CONTACT_FLAG_ACCEL_SPRING = 1u << 3;

struct CRow
{
    F4 raXn_tv, rbXn_maxI, crm, crm2, dRA, dRA2, dRB;
};
struct FRow
{
    F4 raXn_bias, rbXn_tvW, frm, dRA, dRA2, dRB;
};

struct AscIn
{
    U4 cfg;
    F4 invMassAngDom, normalSF, hdr;
    F4 fricN[2];
    F4 refs;
    F4 vel0Top, vel0Bot, vel1Top, vel1Bot;
    CRow contacts[6];
    FRow frictions[6];
};

struct AscOut
{
    F4 vel0Top, vel0Bot, vel1Top, vel1Bot;
    F4 imp0, imp0b, imp1, imp1b;
    F4 cApplied[2], fApplied[2];
    U4 broken;
};

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b) || (a == 0.f && b == 0.f); }

static float dot3(const float a[3], const float b[3]) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
static float pxMaxR(float a, float b) { return (a > b) ? a : b; }

// mirrors computeContactCoefficients() in constraintPrepShared.cuh
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

struct RefStats { bool clampN, clampF, brokenSet, minForce, epsGuard, compliant; };

// exact transcription of solveExtContactsBlock (solver.cuh L54-320)
static void refAscPGS(const AscIn& s, AscOut& o, RefStats& st)
{
    std::memset(&o, 0, sizeof(o));
    st = RefStats{ false, false, false, false, false, false };

    const uint32_t numNormalConstr = s.cfg.x < 6u ? s.cfg.x : 6u;
    const uint32_t numFrictionConstr = s.cfg.y < 6u ? s.cfg.y : 6u;
    const bool doFriction = s.cfg.z != 0u;
    const uint32_t flags = s.cfg.w;
    const float ref0 = s.refs.x, ref1 = s.refs.y;
    const float invMass0 = s.invMassAngDom.x, invMass1 = s.invMassAngDom.y;
    const float angDom0 = s.invMassAngDom.z, angDom1 = s.invMassAngDom.w;
    const float normal[3] = { s.normalSF.x, s.normalSF.y, s.normalSF.z };
    const float staticCof = s.normalSF.w;
    const float restitution = s.hdr.x, cfm = s.hdr.y;
    if (restitution < 0.f) st.compliant = true;

    float imp0Top[3] = {0,0,0}, imp0Bot[3] = {0,0,0}, imp1Top[3] = {0,0,0}, imp1Bot[3] = {0,0,0};
    float v0Top[3] = { s.vel0Top.x, s.vel0Top.y, s.vel0Top.z };
    float v0Bot[3] = { s.vel0Bot.x, s.vel0Bot.y, s.vel0Bot.z };
    float v1Top[3] = { s.vel1Top.x, s.vel1Top.y, s.vel1Top.z };
    float v1Bot[3] = { s.vel1Bot.x, s.vel1Bot.y, s.vel1Bot.z };
    float accum = 0.f;

    for (uint32_t i = 0; i < numNormalConstr; ++i)
    {
        const CRow& c = s.contacts[i];
        const float raXn[3] = { c.raXn_tv.x, c.raXn_tv.y, c.raXn_tv.z };
        const float targetVelocity = c.raXn_tv.w;
        const float rbXn[3] = { c.rbXn_maxI.x, c.rbXn_maxI.y, c.rbXn_maxI.z };
        const float maxImpulse = c.rbXn_maxI.w;
        const float appliedForce = c.crm.x;
        const float resp0 = c.crm.y, resp1 = c.crm.z, coeff0 = c.crm.w, coeff1 = c.crm2.x;

        const float unitResponse = ref0 * resp0 + ref1 * resp1;
        const float recipResponse = (unitResponse > 0.f) ? (1.f / (unitResponse + cfm)) : 0.f;
        float velMultiplier = recipResponse, impulseMul = 1.f, unbiasedError, biasedErr;
        rContactCoeffs(flags, restitution, unitResponse, recipResponse, targetVelocity, coeff0,
                       coeff1, velMultiplier, impulseMul, unbiasedError, biasedErr);

        const float dvATop[3] = { ref0 * c.dRA.w, ref0 * c.dRA2.x, ref0 * c.dRA2.y };
        const float dvABot[3] = { ref0 * c.dRA.x, ref0 * c.dRA.y, ref0 * c.dRA.z };
        const float dvBTop[3] = { ref1 * c.dRB.y, ref1 * c.dRB.z, ref1 * c.dRB.w };
        const float dvBBot[3] = { ref1 * c.dRA2.z, ref1 * c.dRA2.w, ref1 * c.dRB.x };

        const float v0_ = dot3(v0Bot, normal) + dot3(v0Top, raXn);
        const float v1_ = dot3(v1Bot, normal) + dot3(v1Top, rbXn);
        const float normalVel = v0_ - v1_;

        const float tempDeltaF = biasedErr - normalVel * velMultiplier;
        const float _deltaF = std::fmax(tempDeltaF, -appliedForce);
        const float _newForce = appliedForce * impulseMul + _deltaF;
        const float newForce = std::fmin(_newForce, maxImpulse);
        const float deltaF = newForce - appliedForce;
        if (newForce == maxImpulse && _newForce > maxImpulse) st.clampN = true;

        (&o.cApplied[i / 4].x)[i % 4] = newForce;

        for (int k = 0; k < 3; ++k)
        {
            imp0Bot[k] -= raXn[k] * deltaF;
            imp0Top[k] -= normal[k] * deltaF;
            imp1Bot[k] += rbXn[k] * deltaF;
            imp1Top[k] += normal[k] * deltaF;
            v0Top[k] += dvATop[k] * deltaF;
            v0Bot[k] += dvABot[k] * deltaF;
            v1Top[k] += dvBTop[k] * deltaF;
            v1Bot[k] += dvBBot[k] * deltaF;
        }
        accum = accum + newForce;
    }

    const float minNormalForce = s.hdr.z;
    if (minNormalForce > accum) st.minForce = true;
    accum = pxMaxR(accum, minNormalForce);

    if (numFrictionConstr != 0 && doFriction)
    {
        const float dynamicFrictionCof = s.hdr.w;
        const float maxFrictionImpulse = staticCof * accum;
        const float maxDynFrictionImpulse = dynamicFrictionCof * accum;
        uint32_t broken = 0;

        for (uint32_t i = 0; i < numFrictionConstr; ++i)
        {
            const FRow& f = s.frictions[i];
            const F4 frictionNormal = s.fricN[i & 1];
            const float raXn[3] = { f.raXn_bias.x, f.raXn_bias.y, f.raXn_bias.z };
            const float bias = f.raXn_bias.w;
            const float rbXn[3] = { f.rbXn_tvW.x, f.rbXn_tvW.y, f.rbXn_tvW.z };
            const float targetVel = f.rbXn_tvW.w;
            const float appliedForce = f.frm.x;
            const float resp0 = f.frm.y, resp1 = f.frm.z;

            const float respF = ref0 * resp0 + ref1 * resp1;
            const float velMultiplier = (respF > PX_EPS_REAL) ? (P8 / respF) : 0.f;
            if (!(respF > PX_EPS_REAL)) st.epsGuard = true;

            const float dvATop[3] = { ref0 * f.dRA.w, ref0 * f.dRA2.x, ref0 * f.dRA2.y };
            const float dvABot[3] = { ref0 * f.dRA.x, ref0 * f.dRA.y, ref0 * f.dRA.z };
            const float dvBTop[3] = { ref1 * f.dRB.y, ref1 * f.dRB.z, ref1 * f.dRB.w };
            const float dvBBot[3] = { ref1 * f.dRA2.z, ref1 * f.dRA2.w, ref1 * f.dRB.x };

            const float fnormal[3] = { frictionNormal.x, frictionNormal.y, frictionNormal.z };
            const float v0_ = dot3(v0Top, raXn) + dot3(v0Bot, fnormal);
            const float v1_ = dot3(v1Top, rbXn) + dot3(v1Bot, fnormal);
            const float normalVel = v0_ - v1_;

            const float tmp1 = appliedForce - (bias - targetVel) * velMultiplier;
            const float totalImpulse = tmp1 - normalVel * velMultiplier;
            const bool clamp = std::fabs(totalImpulse) > maxFrictionImpulse;
            const float totalClamped =
                std::fmin(maxDynFrictionImpulse, std::fmax(-maxDynFrictionImpulse, totalImpulse));
            const float newAppliedForce = clamp ? totalClamped : totalImpulse;
            const float deltaF = newAppliedForce - appliedForce;
            if (clamp) st.clampF = true;

            for (int k = 0; k < 3; ++k)
            {
                v0Top[k] += dvATop[k] * deltaF;
                v0Bot[k] += dvABot[k] * deltaF;
                v1Top[k] += dvBTop[k] * deltaF;
                v1Bot[k] += dvBBot[k] * deltaF;
                imp0Bot[k] -= raXn[k] * deltaF;
                imp0Top[k] -= fnormal[k] * deltaF;
                imp1Bot[k] += rbXn[k] * deltaF;
                imp1Top[k] += fnormal[k] * deltaF;
            }

            (&o.fApplied[i / 4].x)[i % 4] = newAppliedForce;
            broken = broken | (clamp ? 1u : 0u);
        }
        o.broken.x = broken;
        if (broken) st.brokenSet = true;
    }

    o.imp0 = F4{ imp0Top[0]*(ref0*angDom0), imp0Top[1]*(ref0*angDom0), imp0Top[2]*(ref0*angDom0), 0.f };
    o.imp0b = F4{ imp0Bot[0]*(ref0*invMass0), imp0Bot[1]*(ref0*invMass0), imp0Bot[2]*(ref0*invMass0), 0.f };
    o.imp1 = F4{ imp1Top[0]*(ref1*invMass1), imp1Top[1]*(ref1*invMass1), imp1Top[2]*(ref1*invMass1), 0.f };
    o.imp1b = F4{ imp1Bot[0]*(ref1*angDom1), imp1Bot[1]*(ref1*angDom1), imp1Bot[2]*(ref1*angDom1), 0.f };
    o.vel0Top = F4{ v0Top[0], v0Top[1], v0Top[2], 0.f };
    o.vel0Bot = F4{ v0Bot[0], v0Bot[1], v0Bot[2], 0.f };
    o.vel1Top = F4{ v1Top[0], v1Top[1], v1Top[2], 0.f };
    o.vel1Bot = F4{ v1Bot[0], v1Bot[1], v1Bot[2], 0.f };
}

PXG_TEST(npArtiSolveContact_pgs)
{
    std::mt19937 rng(186001);
    const uint32_t N = 3000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto rv3 = [&](float a, float b) { return F4{ rf(a, b), rf(a, b), rf(a, b), 0.f }; };

    std::vector<AscIn> in(N);
    std::vector<AscOut> ref(N);
    size_t cNorm0 = 0, cFric0 = 0, cCompliant = 0, cAccel = 0, cClampN = 0, cClampF = 0,
           cBroken = 0, cMinForce = 0, cEps = 0, cNoFric = 0, cRefScale = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        AscIn s{};
        const uint32_t numNormal = (i % 13 == 6) ? 0u : (1u + rng() % 6u);
        const uint32_t numFric = (i % 7 == 3) ? 0u : (rng() % 7u);
        const bool doFriction = (i % 5 != 0);
        if (numNormal == 0) ++cNorm0;
        if (numFric == 0) ++cFric0;
        if (!doFriction) ++cNoFric;

        const int rMode = static_cast<int>(rng() % 3);
        const uint32_t accelBit = (rng() % 2) ? CONTACT_FLAG_ACCEL_SPRING : 0u;
        const float restitution = (rMode == 0) ? rf(-2.f, -0.1f) : rf(0.f, 1.f);
        if (rMode == 0) ++cCompliant;
        if (rMode == 0 && accelBit) ++cAccel;
        s.cfg = U4{ numNormal, numFric, doFriction ? 1u : 0u, accelBit };

        const bool refScale = (i % 5 == 2);
        if (refScale) ++cRefScale;
        s.refs = F4{ refScale ? rf(0.3f, 1.5f) : 1.f, refScale ? rf(0.3f, 1.5f) : 1.f, 0.f, 0.f };
        s.invMassAngDom = F4{ rf(0.f, 1.5f), rf(0.f, 1.5f), rf(0.f, 1.5f), rf(0.f, 1.5f) };

        float n[3] = { rf(-1, 1), rf(-1, 1), rf(-1, 1) };
        const float nm = std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
        for (int k = 0; k < 3; ++k) n[k] /= (nm > 0 ? nm : 1.f);
        // minNormalForce sometimes exceeds the accumulated normal impulse
        const bool bigMinForce = (i % 11 == 4);
        const float minNF = bigMinForce ? rf(50.f, 200.f) : rf(0.f, 0.5f);
        s.normalSF = F4{ n[0], n[1], n[2], rf(0.f, 1.f) };
        s.hdr = F4{ restitution, rf(0.f, 0.05f), minNF, rf(0.f, 1.f) };
        s.fricN[0] = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), 0.f };
        s.fricN[1] = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), 0.f };
        s.vel0Top = rv3(-3, 3); s.vel0Bot = rv3(-3, 3);
        s.vel1Top = rv3(-3, 3); s.vel1Bot = rv3(-3, 3);

        for (uint32_t c = 0; c < numNormal; ++c)
        {
            // tight maxImpulse sometimes to force the normal clamp
            const bool tight = (rng() % 3 == 0);
            s.contacts[c].raXn_tv = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), rf(-2, 2) };
            s.contacts[c].rbXn_maxI = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), tight ? rf(0.f, 0.05f) : rf(0.5f, 1e3f) };
            s.contacts[c].crm = F4{ rf(-0.05f, 0.2f), rf(0.05f, 2.f), rf(0.05f, 2.f), rf(-1, 1) };
            s.contacts[c].crm2 = F4{ rf(-1, 1), 0, 0, 0 };
            s.contacts[c].dRA = F4{ rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), rf(-0.5f, 0.5f) };
            s.contacts[c].dRA2 = F4{ rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), rf(-0.5f, 0.5f) };
            s.contacts[c].dRB = F4{ rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), rf(-0.5f, 0.5f) };
        }
        for (uint32_t fr = 0; fr < numFric; ++fr)
        {
            const bool zeroResp = (i % 17 == 8);  // eps-guard: respF ~ 0
            const float r0 = zeroResp ? 0.f : rf(0.05f, 2.f);
            const float r1 = zeroResp ? 0.f : rf(0.05f, 2.f);
            s.frictions[fr].raXn_bias = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), rf(-1, 1) };
            s.frictions[fr].rbXn_tvW = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), rf(-1, 1) };
            s.frictions[fr].frm = F4{ rf(-0.2f, 0.2f), r0, r1, 0 };
            s.frictions[fr].dRA = F4{ rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), rf(-0.5f, 0.5f) };
            s.frictions[fr].dRA2 = F4{ rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), rf(-0.5f, 0.5f) };
            s.frictions[fr].dRB = F4{ rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), rf(-0.5f, 0.5f) };
        }

        in[i] = s;
        RefStats st;
        refAscPGS(s, ref[i], st);
        if (st.clampN) ++cClampN;
        if (st.clampF) ++cClampF;
        if (st.brokenSet) ++cBroken;
        if (st.minForce) ++cMinForce;
        if (st.epsGuard) ++cEps;
    }

    std::printf("[ info ] asc cov: norm0 %zu fric0 %zu compliant %zu accel %zu clampN %zu clampF %zu "
                "broken %zu minForce %zu eps %zu nofric %zu refScale %zu\n",
                cNorm0, cFric0, cCompliant, cAccel, cClampN, cClampF, cBroken, cMinForce, cEps,
                cNoFric, cRefScale);
    PXG_CHECK(cNorm0 > 100);
    PXG_CHECK(cFric0 > 300);
    PXG_CHECK(cCompliant > 700);
    PXG_CHECK(cAccel > 300);
    PXG_CHECK(cClampN > 300);
    PXG_CHECK(cClampF > 300);
    PXG_CHECK(cBroken > 300);
    PXG_CHECK(cMinForce > 100);
    PXG_CHECK(cEps > 100);
    PXG_CHECK(cNoFric > 300);
    PXG_CHECK(cRefScale > 400);

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<AscOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npArtiSolveContact.spv", "asCtPGS",
             { (uint32_t)sizeof(AscIn), (uint32_t)sizeof(AscOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<AscOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        bool ok = std::memcmp(&ref[i].broken, &g[i].broken, sizeof(U4)) == 0;
        const float* rfp = &ref[i].vel0Top.x;
        const float* qfp = &g[i].vel0Top.x;
        const size_t nf = (sizeof(AscOut) - sizeof(U4)) / 4;
        for (size_t x = 0; x < nf && ok; ++x)
            ok = eqFZ(rfp[x], qfp[x]);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u brkR %u brkG %u imp0R (%g %g %g) imp0G (%g %g %g)\n", i,
                            ref[i].broken.x, g[i].broken.x, ref[i].imp0.x, ref[i].imp0.y,
                            ref[i].imp0.z, g[i].imp0.x, g[i].imp0.y, g[i].imp0.z);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "asCtPGS outputs bitwise-identical (mod +-0)");
}

}  // namespace
