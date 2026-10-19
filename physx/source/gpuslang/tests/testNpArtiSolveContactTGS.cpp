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

// Verification of npArtiSolveContactTGS.slang (asCtTGS): solveExtContactBlockTGS
// (solver.cuh L430-735), the TGS articulation ext-body contact solve, against
// an exact CPU transcription.  A normal loop, a paired-friction loop (combined
// sqrt magnitude + ratio clamp), and a torsional-friction tail; the final
// impulse scale is SYMMETRIC (both bodies top*invMass, bottom*angDom).  All
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

static const float PX_EPS_REAL = 1.192092896e-07f;
static const uint32_t CONTACT_FLAG_ACCEL_SPRING = 1u << 4; // 5.12 PxgSolverContactFlags (5.6.1: 1 << 3)

struct CRow
{
    F4 raXn_extra, rbXn_tv, misc, resp, dRA, dRA2, dRB;
};
struct FRow
{
    F4 raXn_err, rbXn_tv, frm, dRA, dRA2, dRB;
};

struct AtcIn
{
    U4 cfg;
    F4 invMassAngDom, normalSF, hdr, hdr2, fricHdr;
    F4 fricN[2];
    F4 refs;
    F4 d0Top, d0Bot, d1Top, d1Bot;
    F4 vel0Top, vel0Bot, vel1Top, vel1Bot;
    CRow contacts[6];
    FRow frictions[6];
};

struct AtcOut
{
    F4 vel0Top, vel0Bot, vel1Top, vel1Bot;
    F4 imp0, imp0b, imp1, imp1b;
    F4 cApplied[2], fApplied[2];
    U4 broken;
};

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed

static float dot3(const float a[3], const float b[3]) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
static float pxMaxR(float a, float b) { return (a > b) ? a : b; }  // PxMax float = selectMax
static float pxMinR(float a, float b) { return (a < b) ? a : b; }  // PxMin float = selectMin

// mirrors the runtime `a`-coefficient overload of
// computeCompliantContactCoefficientsTGS (constraintPrepShared.cuh L365-376)
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

struct RefStats { bool clampN, clampF, brokenSet, minForce, epsGuard, compliant, torsional, torClamp; };

// exact transcription of solveExtContactBlockTGS (solver.cuh L430-735)
static void refAtcTGS(const AtcIn& s, AtcOut& o, RefStats& st)
{
    std::memset(&o, 0, sizeof(o));
    st = RefStats{ false, false, false, false, false, false, false, false };

    const uint32_t numNormalConstr = s.cfg.x < 6u ? s.cfg.x : 6u;
    const uint32_t totalFrictionConstr = s.cfg.y < 6u ? s.cfg.y : 6u;
    const uint32_t numFrictionConstr = totalFrictionConstr & (~1u);
    const uint32_t flags = s.cfg.w;
    const float ref0 = s.refs.x, ref1 = s.refs.y;
    const float invMass0 = s.invMassAngDom.x, invMass1 = s.invMassAngDom.y;
    const float angDom0 = s.invMassAngDom.z, angDom1 = s.invMassAngDom.w;
    const float normal[3] = { s.normalSF.x, s.normalSF.y, s.normalSF.z };
    const float staticFrictionCof = s.normalSF.w;
    const float restitutionXdt = s.hdr.x, p8 = s.hdr.y, cfm = s.hdr.z, maxPenBias = s.hdr.w;
    const float minNormalForce = s.hdr2.x, minPen = s.hdr2.y, elapsedTime = s.hdr2.z,
                dynamicFrictionCof = s.hdr2.w;
    if (restitutionXdt < 0.f) st.compliant = true;

    float imp0Top[3] = {0,0,0}, imp0Bot[3] = {0,0,0}, imp1Top[3] = {0,0,0}, imp1Bot[3] = {0,0,0};
    float v0Top[3] = { s.vel0Top.x, s.vel0Top.y, s.vel0Top.z };
    float v0Bot[3] = { s.vel0Bot.x, s.vel0Bot.y, s.vel0Bot.z };
    float v1Top[3] = { s.vel1Top.x, s.vel1Top.y, s.vel1Top.z };
    float v1Bot[3] = { s.vel1Bot.x, s.vel1Bot.y, s.vel1Bot.z };
    const float d0Top[3] = { s.d0Top.x, s.d0Top.y, s.d0Top.z };
    const float d0Bot[3] = { s.d0Bot.x, s.d0Bot.y, s.d0Bot.z };
    const float d1Top[3] = { s.d1Top.x, s.d1Top.y, s.d1Top.z };
    const float d1Bot[3] = { s.d1Bot.x, s.d1Bot.y, s.d1Bot.z };

    const float relMotion[3] = { d0Bot[0]-d1Bot[0], d0Bot[1]-d1Bot[1], d0Bot[2]-d1Bot[2] };
    const float deltaV = dot3(normal, relMotion);
    float accum = 0.f;

    for (uint32_t i = 0; i < numNormalConstr; ++i)
    {
        const CRow& c = s.contacts[i];
        const float raXn[3] = { c.raXn_extra.x, c.raXn_extra.y, c.raXn_extra.z };
        const float compliantContactCoef = c.raXn_extra.w;
        const float rbXn[3] = { c.rbXn_tv.x, c.rbXn_tv.y, c.rbXn_tv.z };
        const float targetVel = c.rbXn_tv.w;
        const float appliedForce = c.misc.x;
        const float separation = c.misc.y;
        const float maxImpulse = c.misc.z;
        float biasCoefficient = c.misc.w;
        const float resp0 = ref0 * c.resp.x;
        const float resp1 = ref1 * c.resp.y;

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
        if (newForce == maxImpulse && _newForce > maxImpulse) st.clampN = true;

        (&o.cApplied[i / 4].x)[i % 4] = newForce;

        const float dvALin[3] = { ref0 * c.dRA.x, ref0 * c.dRA.y, ref0 * c.dRA.z };
        const float dvAAng[3] = { ref0 * c.dRA.w, ref0 * c.dRA2.x, ref0 * c.dRA2.y };
        const float dvBLin[3] = { ref1 * c.dRA2.z, ref1 * c.dRA2.w, ref1 * c.dRB.x };
        const float dvBAng[3] = { ref1 * c.dRB.y, ref1 * c.dRB.z, ref1 * c.dRB.w };
        for (int k = 0; k < 3; ++k)
        {
            v0Bot[k] += dvALin[k] * deltaF;
            v1Bot[k] += dvBLin[k] * deltaF;
            v0Top[k] += dvAAng[k] * deltaF;
            v1Top[k] += dvBAng[k] * deltaF;
            imp0Top[k] -= normal[k] * deltaF;
            imp0Bot[k] -= raXn[k] * deltaF;
            imp1Top[k] += normal[k] * deltaF;
            imp1Bot[k] += rbXn[k] * deltaF;
        }
        accum = accum + newForce;
    }

    if (minNormalForce > accum) st.minForce = true;
    accum = pxMaxR(accum, minNormalForce);

    if (numFrictionConstr != 0u)
    {
        const float biasCoefficient = s.fricHdr.x;
        const float maxFrictionImpulse = staticFrictionCof * accum;
        const float maxDynFrictionImpulse = dynamicFrictionCof * accum;
        uint32_t broken = 0;

        const float normal0[3] = { s.fricN[0].x, s.fricN[0].y, s.fricN[0].z };
        const float normal1[3] = { s.fricN[1].x, s.fricN[1].y, s.fricN[1].z };
        const float deltaMotion0 = dot3(normal0, relMotion);
        const float deltaMotion1 = dot3(normal1, relMotion);

        for (uint32_t i = 0; i < numFrictionConstr; i += 2)
        {
            const FRow& f0 = s.frictions[i];
            const FRow& f1 = s.frictions[i + 1];

            const float raXn0[3] = { f0.raXn_err.x, f0.raXn_err.y, f0.raXn_err.z };
            const float initialError0 = f0.raXn_err.w;
            const float rbXn0[3] = { f0.rbXn_tv.x, f0.rbXn_tv.y, f0.rbXn_tv.z };
            const float targetVel0 = f0.rbXn_tv.w;
            const float appliedForce0 = f0.frm.x;

            const float raXn1[3] = { f1.raXn_err.x, f1.raXn_err.y, f1.raXn_err.z };
            const float initialError1 = f1.raXn_err.w;
            const float rbXn1[3] = { f1.rbXn_tv.x, f1.rbXn_tv.y, f1.rbXn_tv.z };
            const float targetVel1 = f1.rbXn_tv.w;
            const float appliedForce1 = f1.frm.x;

            const float resp0 = ref0 * f0.frm.y + ref1 * f0.frm.z;
            const float velMultiplier0 = (resp0 > PX_EPS_REAL) ? (p8 / (resp0 + cfm)) : 0.f;
            const float resp1 = ref0 * f1.frm.y + ref1 * f1.frm.z;
            const float velMultiplier1 = (resp1 > PX_EPS_REAL) ? (p8 / (resp1 + cfm)) : 0.f;
            if (!(resp0 > PX_EPS_REAL) || !(resp1 > PX_EPS_REAL)) st.epsGuard = true;

            const float v00 = dot3(v0Top, raXn0) + dot3(v0Bot, normal0);
            const float v10 = dot3(v1Top, rbXn0) + dot3(v1Bot, normal0);
            const float normalVel0 = v00 - v10;
            const float v01 = dot3(v0Top, raXn1) + dot3(v0Bot, normal1);
            const float v11 = dot3(v1Top, rbXn1) + dot3(v1Bot, normal1);
            const float normalVel1 = v01 - v11;

            const float error0 = initialError0 - targetVel0 * elapsedTime +
                                 (dot3(raXn0, d0Top) - dot3(rbXn0, d1Top) + deltaMotion0);
            const float bias0 = error0 * biasCoefficient;
            const float tmp10 = appliedForce0 - (bias0 - targetVel0) * velMultiplier0;
            const float totalImpulse0 = tmp10 - normalVel0 * velMultiplier0;

            const float error1 = initialError1 - targetVel1 * elapsedTime +
                                 (dot3(raXn1, d0Top) - dot3(rbXn1, d1Top) + deltaMotion1);
            const float bias1 = error1 * biasCoefficient;
            const float tmp11 = appliedForce1 - (bias1 - targetVel1) * velMultiplier1;
            const float totalImpulse1 = tmp11 - normalVel1 * velMultiplier1;

            const float totalImpulse =
                std::sqrt(totalImpulse0 * totalImpulse0 + totalImpulse1 * totalImpulse1);
            const bool clamp = totalImpulse > maxFrictionImpulse;
            const float ratio =
                clamp ? (std::fmin(maxDynFrictionImpulse, totalImpulse) / totalImpulse) : 1.f;
            if (clamp) st.clampF = true;

            const float newAppliedForce0 = totalImpulse0 * ratio;
            const float newAppliedForce1 = totalImpulse1 * ratio;
            const float deltaF0 = newAppliedForce0 - appliedForce0;
            const float deltaF1 = newAppliedForce1 - appliedForce1;

            const float r0Lin[3] = { f0.dRA.x, f0.dRA.y, f0.dRA.z };
            const float r0Ang[3] = { f0.dRA.w, f0.dRA2.x, f0.dRA2.y };
            const float r0BLin[3] = { f0.dRA2.z, f0.dRA2.w, f0.dRB.x };
            const float r0BAng[3] = { f0.dRB.y, f0.dRB.z, f0.dRB.w };
            const float r1Lin[3] = { f1.dRA.x, f1.dRA.y, f1.dRA.z };
            const float r1Ang[3] = { f1.dRA.w, f1.dRA2.x, f1.dRA2.y };
            const float r1BLin[3] = { f1.dRA2.z, f1.dRA2.w, f1.dRB.x };
            const float r1BAng[3] = { f1.dRB.y, f1.dRB.z, f1.dRB.w };
            for (int k = 0; k < 3; ++k)
            {
                v0Bot[k] += (ref0 * r0Lin[k]) * deltaF0;
                v1Bot[k] += (ref1 * r0BLin[k]) * deltaF0;
                v0Top[k] += (ref0 * r0Ang[k]) * deltaF0;
                v1Top[k] += (ref1 * r0BAng[k]) * deltaF0;
            }
            for (int k = 0; k < 3; ++k)
            {
                v0Bot[k] += (ref0 * r1Lin[k]) * deltaF1;
                v1Bot[k] += (ref1 * r1BLin[k]) * deltaF1;
                v0Top[k] += (ref0 * r1Ang[k]) * deltaF1;
                v1Top[k] += (ref1 * r1BAng[k]) * deltaF1;
            }

            (&o.fApplied[i / 4].x)[i % 4] = newAppliedForce0;
            (&o.fApplied[(i + 1) / 4].x)[(i + 1) % 4] = newAppliedForce1;
            broken = broken | (clamp ? 1u : 0u);

            for (int k = 0; k < 3; ++k)
            {
                imp0Top[k] -= normal0[k] * deltaF0;
                imp0Bot[k] -= raXn0[k] * deltaF0;
                imp1Top[k] += normal0[k] * deltaF0;
                imp1Bot[k] += rbXn0[k] * deltaF0;
                imp0Top[k] -= normal1[k] * deltaF1;
                imp0Bot[k] -= raXn1[k] * deltaF1;
                imp1Top[k] += normal1[k] * deltaF1;
                imp1Bot[k] += rbXn1[k] * deltaF1;
            }
        }

        if (numFrictionConstr < totalFrictionConstr)
        {
            st.torsional = true;
            const float frictionScale = s.fricHdr.y;
            const FRow& f0 = s.frictions[numFrictionConstr];
            const float raXn0[3] = { f0.raXn_err.x, f0.raXn_err.y, f0.raXn_err.z };
            const float rbXn0[3] = { f0.rbXn_tv.x, f0.rbXn_tv.y, f0.rbXn_tv.z };
            const float appliedForce0 = f0.frm.x;
            const float targetVel0 = f0.rbXn_tv.w;

            const float resp0 = ref0 * f0.frm.y + ref1 * f0.frm.z;
            const float velMultiplier0 = (resp0 > 0.f) ? (p8 / (resp0 + cfm)) : 0.f;

            const float v00 = dot3(v0Top, raXn0);
            const float v10 = dot3(v1Top, rbXn0);
            const float normalVel0 = v00 - v10;

            const float tmp10 = appliedForce0 - (-targetVel0) * velMultiplier0;
            const float totalImpulse = tmp10 - normalVel0 * velMultiplier0;
            const bool clamp = std::fabs(totalImpulse) > (maxFrictionImpulse * frictionScale);
            const float lo = -maxDynFrictionImpulse * frictionScale;
            const float hi = maxDynFrictionImpulse * frictionScale;
            const float totalClamped = pxMinR(hi, pxMaxR(lo, totalImpulse));
            const float newAppliedForce = clamp ? totalClamped : totalImpulse;
            const float deltaF0 = newAppliedForce - appliedForce0;
            if (clamp) st.torClamp = true;

            const float r0Lin[3] = { f0.dRA.x, f0.dRA.y, f0.dRA.z };
            const float r0Ang[3] = { f0.dRA.w, f0.dRA2.x, f0.dRA2.y };
            const float r0BLin[3] = { f0.dRA2.z, f0.dRA2.w, f0.dRB.x };
            const float r0BAng[3] = { f0.dRB.y, f0.dRB.z, f0.dRB.w };
            for (int k = 0; k < 3; ++k)
            {
                v0Bot[k] += (ref0 * r0Lin[k]) * deltaF0;
                v1Bot[k] += (ref1 * r0BLin[k]) * deltaF0;
                v0Top[k] += (ref0 * r0Ang[k]) * deltaF0;
                v1Top[k] += (ref1 * r0BAng[k]) * deltaF0;
            }

            (&o.fApplied[numFrictionConstr / 4].x)[numFrictionConstr % 4] = newAppliedForce;
            broken = broken | (clamp ? 1u : 0u);

            for (int k = 0; k < 3; ++k)
            {
                imp0Bot[k] -= raXn0[k] * deltaF0;
                imp1Bot[k] += rbXn0[k] * deltaF0;
            }
        }

        o.broken.x = broken;
        if (broken) st.brokenSet = true;
    }

    o.imp0 = F4{ imp0Top[0]*(invMass0*ref0), imp0Top[1]*(invMass0*ref0), imp0Top[2]*(invMass0*ref0), 0.f };
    o.imp0b = F4{ imp0Bot[0]*(angDom0*ref0), imp0Bot[1]*(angDom0*ref0), imp0Bot[2]*(angDom0*ref0), 0.f };
    o.imp1 = F4{ imp1Top[0]*(invMass1*ref1), imp1Top[1]*(invMass1*ref1), imp1Top[2]*(invMass1*ref1), 0.f };
    o.imp1b = F4{ imp1Bot[0]*(angDom1*ref1), imp1Bot[1]*(angDom1*ref1), imp1Bot[2]*(angDom1*ref1), 0.f };
    o.vel0Top = F4{ v0Top[0], v0Top[1], v0Top[2], 0.f };
    o.vel0Bot = F4{ v0Bot[0], v0Bot[1], v0Bot[2], 0.f };
    o.vel1Top = F4{ v1Top[0], v1Top[1], v1Top[2], 0.f };
    o.vel1Bot = F4{ v1Bot[0], v1Bot[1], v1Bot[2], 0.f };
}

PXG_TEST(npArtiSolveContactTGS_pgs)
{
    std::mt19937 rng(188001);
    const uint32_t N = 3200;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto rv3 = [&](float a, float b) { return F4{ rf(a, b), rf(a, b), rf(a, b), 0.f }; };

    std::vector<AtcIn> in(N);
    std::vector<AtcOut> ref(N);
    size_t cNorm0 = 0, cFric0 = 0, cCompliant = 0, cAccel = 0, cClampN = 0, cClampF = 0,
           cBroken = 0, cMinForce = 0, cEps = 0, cTorsional = 0, cTorClamp = 0, cRefScale = 0,
           cZeroRespN = 0, cTinyResp = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        AtcIn s{};
        const uint32_t numNormal = (i % 13 == 6) ? 0u : (1u + rng() % 6u);
        // total friction 0..6; odd>=3 exercises the torsional tail
        const uint32_t numFric = (i % 7 == 3) ? 0u : (rng() % 7u);
        if (numNormal == 0) ++cNorm0;
        if (numFric == 0) ++cFric0;
        // tiny-friction-response stripe: seeds friction resp in the OPEN
        // interval (0, PX_EPS_REAL] so the paired guard `resp > PX_EPS_REAL`
        // and the torsional guard `resp > 0` DIVERGE (they agree at resp==0
        // and resp>=0.015; only this interval separates them).  ref0*r0 stays
        // < PX_EPS since 1.5f * 7e-8f = 1.05e-7 < 1.192e-7.
        const bool tinyResp = (i % 23 == 7);
        if (tinyResp && numFric) ++cTinyResp;

        const bool zeroRespN = (i % 19 == 5);
        const int rMode = zeroRespN ? (1 + static_cast<int>(rng() % 2)) : static_cast<int>(rng() % 3);
        const uint32_t accelBit = (rng() % 2) ? CONTACT_FLAG_ACCEL_SPRING : 0u;
        const float restitutionXdt = (rMode == 0) ? rf(-2.f, -0.1f) : rf(0.f, 1.f);
        if (rMode == 0) ++cCompliant;
        if (rMode == 0 && accelBit) ++cAccel;
        if (zeroRespN && numNormal) ++cZeroRespN;
        s.cfg = U4{ numNormal, numFric, 0u, accelBit };

        const bool refScale = (i % 5 == 2);
        if (refScale) ++cRefScale;
        s.refs = F4{ refScale ? rf(0.3f, 1.5f) : 1.f, refScale ? rf(0.3f, 1.5f) : 1.f, 0.f, 0.f };
        s.invMassAngDom = F4{ rf(0.f, 1.5f), rf(0.f, 1.5f), rf(0.f, 1.5f), rf(0.f, 1.5f) };

        float n[3] = { rf(-1, 1), rf(-1, 1), rf(-1, 1) };
        const float nm = std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
        for (int k = 0; k < 3; ++k) n[k] /= (nm > 0 ? nm : 1.f);
        const bool bigMinForce = (i % 11 == 4);
        const float minNF = bigMinForce ? rf(50.f, 200.f) : rf(0.f, 0.5f);
        s.normalSF = F4{ n[0], n[1], n[2], rf(0.f, 1.f) };
        s.hdr = F4{ restitutionXdt, 0.8f, zeroRespN ? 0.f : rf(0.f, 0.05f), rf(0.02f, 2.f) };
        s.hdr2 = F4{ minNF, rf(-2.f, -0.001f), rf(0.f, 0.05f), rf(0.f, 1.f) };
        // friction header biasCoefficient + torsionalFrictionScale
        s.fricHdr = F4{ rf(-2.f, 2.f), rf(0.05f, 1.5f), 0.f, 0.f };
        s.fricN[0] = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), 0.f };
        s.fricN[1] = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), 0.f };
        s.d0Top = rv3(-0.3f, 0.3f); s.d0Bot = rv3(-0.3f, 0.3f);
        s.d1Top = rv3(-0.3f, 0.3f); s.d1Bot = rv3(-0.3f, 0.3f);
        s.vel0Top = rv3(-3, 3); s.vel0Bot = rv3(-3, 3);
        s.vel1Top = rv3(-3, 3); s.vel1Bot = rv3(-3, 3);

        for (uint32_t c = 0; c < numNormal; ++c)
        {
            const bool tight = (rng() % 3 == 0);
            const float cr0 = zeroRespN ? 0.f : rf(0.05f, 2.f);
            const float cr1 = zeroRespN ? 0.f : rf(0.05f, 2.f);
            s.contacts[c].raXn_extra = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), rf(0.f, 2.f) };
            s.contacts[c].rbXn_tv = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), rf(-2, 2) };
            s.contacts[c].misc = F4{ rf(-0.05f, 0.2f), rf(-0.2f, 0.2f),
                                     tight ? rf(0.f, 0.05f) : rf(0.5f, 1e3f), rf(0.f, 4.f) };
            s.contacts[c].resp = F4{ cr0, cr1, 0.f, 0.f };
            s.contacts[c].dRA = F4{ rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), rf(-0.5f, 0.5f) };
            s.contacts[c].dRA2 = F4{ rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), rf(-0.5f, 0.5f) };
            s.contacts[c].dRB = F4{ rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), rf(-0.5f, 0.5f) };
        }
        for (uint32_t fr = 0; fr < numFric; ++fr)
        {
            const bool zeroResp = (i % 17 == 8);  // eps-guard: respF == 0
            const float r0 = tinyResp ? rf(1e-9f, 7e-8f) : (zeroResp ? 0.f : rf(0.05f, 2.f));
            const float r1 = tinyResp ? 0.f : (zeroResp ? 0.f : rf(0.05f, 2.f));
            s.frictions[fr].raXn_err = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), rf(-1, 1) };
            s.frictions[fr].rbXn_tv = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), rf(-1, 1) };
            s.frictions[fr].frm = F4{ rf(-0.2f, 0.2f), r0, r1, 0 };
            s.frictions[fr].dRA = F4{ rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), rf(-0.5f, 0.5f) };
            s.frictions[fr].dRA2 = F4{ rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), rf(-0.5f, 0.5f) };
            s.frictions[fr].dRB = F4{ rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), rf(-0.5f, 0.5f) };
        }

        in[i] = s;
        RefStats st;
        refAtcTGS(s, ref[i], st);
        if (st.clampN) ++cClampN;
        if (st.clampF) ++cClampF;
        if (st.brokenSet) ++cBroken;
        if (st.minForce) ++cMinForce;
        if (st.epsGuard) ++cEps;
        if (st.torsional) ++cTorsional;
        if (st.torClamp) ++cTorClamp;
    }

    std::printf("[ info ] atc cov: norm0 %zu fric0 %zu compliant %zu accel %zu clampN %zu clampF %zu "
                "broken %zu minForce %zu eps %zu torsional %zu torClamp %zu refScale %zu zeroRespN %zu "
                "tinyResp %zu\n",
                cNorm0, cFric0, cCompliant, cAccel, cClampN, cClampF, cBroken, cMinForce, cEps,
                cTorsional, cTorClamp, cRefScale, cZeroRespN, cTinyResp);
    PXG_CHECK(cNorm0 > 100);
    PXG_CHECK(cFric0 > 300);
    PXG_CHECK(cCompliant > 700);
    PXG_CHECK(cAccel > 300);
    PXG_CHECK(cClampN > 200);
    PXG_CHECK(cClampF > 200);
    PXG_CHECK(cBroken > 200);
    PXG_CHECK(cMinForce > 100);
    PXG_CHECK(cEps > 80);
    PXG_CHECK(cTorsional > 200);
    PXG_CHECK(cTorClamp > 80);
    PXG_CHECK(cRefScale > 400);
    PXG_CHECK(cZeroRespN > 80);
    PXG_CHECK(cTinyResp > 60);

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<AtcOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npArtiSolveContactTGS.spv", "asCtTGS",
             { (uint32_t)sizeof(AtcIn), (uint32_t)sizeof(AtcOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<AtcOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        bool ok = std::memcmp(&ref[i].broken, &g[i].broken, sizeof(U4)) == 0;
        const float* rfp = &ref[i].vel0Top.x;
        const float* qfp = &g[i].vel0Top.x;
        const size_t nf = (sizeof(AtcOut) - sizeof(U4)) / 4;
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
    PXG_CHECK_MSG(mm == 0, "asCtTGS outputs bitwise-identical (mod +-0)");
}

}  // namespace
