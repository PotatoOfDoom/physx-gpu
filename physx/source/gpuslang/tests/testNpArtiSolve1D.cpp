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

// Verification of npArtiSolve1D.slang (as1dPGS): solveExt1DBlock
// (solverBlock.cuh L208-326), the PGS articulation ext-body 1D-constraint
// solve, against an exact CPU transcription.  All outputs bitwise modulo
// the documented +-0 class (eqFZ).

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

static const uint32_t SC_SPRING = 1u << 5;
static const uint32_t SC_ACCEL = 1u << 7;
static const float DY_MIN_RESPONSE = 1e-12f;

struct SRow
{
    F4 lin0XYZ_minImpulse;
    F4 lin1XYZ_maxImpulse;
    F4 ang0XYZ_resp0;
    F4 ang1XYZ_resp1;
    F4 misc;    // initJointSpeed, coeff0, coeff1, appliedForce
    U4 flagsU;
    F4 dRA, dRA2, dRB;
};

struct As1In
{
    U4 cfg;     // rowCounts, residualEnabled, 0, 0
    F4 hdr;     // invMass0D0, invMass1D1, invInertiaScale0, invInertiaScale1
    F4 hdr2;    // cfm, ref0, ref1, 0
    F4 vel0Top, vel0Bot, vel1Top, vel1Bot;
    F4 imp0Top, imp0Bot, imp1Top, imp1Bot;
    SRow rows[20];
};

struct As1Out
{
    F4 vel0Top, vel0Bot, vel1Top, vel1Bot;
    F4 imp0, imp0b, imp1, imp1b;
    F4 appliedForce[5];
    F4 residual[5];
};

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed

static float dot3(const float a[3], const float b[3]) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }

static float rRecipUnitResponse(float unitResponse, float minRowResponse)
{
    return unitResponse <= minRowResponse ? 0.f : 1.f / unitResponse;
}

// mirrors compute1dConstraintSolverConstantsPGS (coeff2 = initJointSpeed)
static void rConstantsPGS(bool isSpring, bool isAccel, float coeff0, float coeff1, float coeff2,
                          float unitResponse, float recipUnitResponse, float& constant,
                          float& unbiasedConstant, float& velMultiplier, float& impulseMultiplier)
{
    if (isSpring)
    {
        const float a = coeff0;
        const float b = coeff1;
        if (isAccel)
        {
            const float x = 1.f / (1.f + a);
            constant = x * recipUnitResponse * b;
            unbiasedConstant = constant;
            velMultiplier = -x * recipUnitResponse * a;
            impulseMultiplier = 1.f - x;
        }
        else
        {
            const float x = 1.f / (1.f + a * unitResponse);
            constant = x * b;
            unbiasedConstant = constant;
            velMultiplier = -x * a;
            impulseMultiplier = 1.f - x;
        }
    }
    else
    {
        velMultiplier = -recipUnitResponse;
        impulseMultiplier = 1.f;
        constant = coeff0 * recipUnitResponse;
        unbiasedConstant = coeff1 * recipUnitResponse;
    }
    const float velBias = coeff2 * velMultiplier;
    constant += velBias;
    unbiasedConstant += velBias;
}

static float rResidual(float deltaF, float vMul) { return vMul == 0.f ? 0.f : deltaF / vMul; }

// exact transcription of solveExt1DBlock (solverBlock.cuh L208-326)
static void refAs1dPGS(const As1In& s, As1Out& o)
{
    std::memset(&o, 0, sizeof(o));
    const uint32_t numRows = s.cfg.x < 20u ? s.cfg.x : 20u;
    const bool residualReportingEnabled = s.cfg.y != 0u;
    const float ref0 = s.hdr2.y, ref1 = s.hdr2.z;

    float linVel0[3] = { s.vel0Bot.x, s.vel0Bot.y, s.vel0Bot.z };
    float linVel1[3] = { s.vel1Bot.x, s.vel1Bot.y, s.vel1Bot.z };
    float angVel0[3] = { s.vel0Top.x, s.vel0Top.y, s.vel0Top.z };
    float angVel1[3] = { s.vel1Top.x, s.vel1Top.y, s.vel1Top.z };

    float li0[3] = { s.imp0Bot.x, s.imp0Bot.y, s.imp0Bot.z };
    float li1[3] = { s.imp1Bot.x, s.imp1Bot.y, s.imp1Bot.z };
    float ai0[3] = { s.imp0Top.x, s.imp0Top.y, s.imp0Top.z };
    float ai1[3] = { s.imp1Top.x, s.imp1Top.y, s.imp1Top.z };

    const float invMass0 = ref0 * s.hdr.x;
    const float invMass1 = ref1 * s.hdr.y;
    const float invInertiaScale0 = ref0 * s.hdr.z;
    const float invInertiaScale1 = ref1 * s.hdr.w;
    const float cfm = s.hdr2.x;

    for (uint32_t i = 0; i < numRows; ++i)
    {
        const SRow& r = s.rows[i];
        const float clinVel0[3] = { r.lin0XYZ_minImpulse.x, r.lin0XYZ_minImpulse.y, r.lin0XYZ_minImpulse.z };
        const float clinVel1[3] = { r.lin1XYZ_maxImpulse.x, r.lin1XYZ_maxImpulse.y, r.lin1XYZ_maxImpulse.z };
        const float cangVel0[3] = { r.ang0XYZ_resp0.x, r.ang0XYZ_resp0.y, r.ang0XYZ_resp0.z };
        const float cangVel1[3] = { r.ang1XYZ_resp1.x, r.ang1XYZ_resp1.y, r.ang1XYZ_resp1.z };

        const float resp0 = ref0 * r.ang0XYZ_resp0.w;
        const float resp1 = ref1 * r.ang1XYZ_resp1.w;
        const float initJointSpeed = r.misc.x;
        const float coeff0 = r.misc.y;
        const float coeff1 = r.misc.z;
        const uint32_t flags = r.flagsU.x;

        const float unitResponse = resp0 + resp1 + cfm;
        const float recipResponse = rRecipUnitResponse(unitResponse, DY_MIN_RESPONSE);

        float constant, unbiasedConstant, vMul, iMul;
        rConstantsPGS((flags & SC_SPRING) != 0, (flags & SC_ACCEL) != 0, coeff0, coeff1,
                      initJointSpeed, unitResponse, recipResponse, constant, unbiasedConstant, vMul,
                      iMul);

        const float appliedForce = r.misc.w;
        const float maxImpulse = r.lin1XYZ_maxImpulse.w;
        const float minImpulse = r.lin0XYZ_minImpulse.w;

        const float v0 = dot3(linVel0, clinVel0) + dot3(angVel0, cangVel0);
        const float v1 = dot3(linVel1, clinVel1) + dot3(angVel1, cangVel1);
        const float normalVel = v0 - v1;

        const float unclampedForce = iMul * appliedForce + (vMul * normalVel + constant);
        const float clampedForce = std::fmin(maxImpulse, std::fmax(minImpulse, unclampedForce));
        const float deltaF = clampedForce - appliedForce;

        (&o.appliedForce[i / 4].x)[i % 4] = clampedForce;
        if (residualReportingEnabled)
            (&o.residual[i / 4].x)[i % 4] = rResidual(deltaF, vMul);

        for (int k = 0; k < 3; ++k)
        {
            li0[k] = clinVel0[k] * deltaF + li0[k];
            ai0[k] = cangVel0[k] * deltaF + ai0[k];
            li1[k] = clinVel1[k] * deltaF + li1[k];
            ai1[k] = cangVel1[k] * deltaF + ai1[k];
        }

        const float dRALin[3] = { r.dRA.x, r.dRA.y, r.dRA.z };
        const float dRAAng[3] = { r.dRA.w, r.dRA2.x, r.dRA2.y };
        const float dRBLin[3] = { r.dRA2.z, r.dRA2.w, r.dRB.x };
        const float dRBAng[3] = { r.dRB.y, r.dRB.z, r.dRB.w };

        for (int k = 0; k < 3; ++k)
        {
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

PXG_TEST(npArtiSolve1D_pgs)
{
    std::mt19937 rng(185001);
    const uint32_t N = 3000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<As1In> in(N);
    std::vector<As1Out> ref(N);
    size_t cRows0 = 0, cSpring = 0, cAccel = 0, cHard = 0, cClampMax = 0, cClampMin = 0,
           cResid = 0, cRefScale = 0, cZeroResp = 0, cInvClamp = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        As1In s{};
        const uint32_t numRows = (i % 11 == 4) ? 0u : (1u + rng() % 15u);
        if (numRows == 0) ++cRows0;
        const bool resid = (rng() % 2) != 0;
        if (resid) ++cResid;
        s.cfg = U4{ numRows, resid ? 1u : 0u, 0, 0 };
        s.hdr = F4{ rf(0.f, 2.f), rf(0.f, 2.f), rf(0.f, 2.f), rf(0.f, 2.f) };
        const bool refScale = (i % 5 == 2);
        if (refScale) ++cRefScale;
        // degenerate-response stripe: unitResponse == 0 <= DY_MIN_RESPONSE so
        // the computeRecipUnitResponse zero-guard fires (recipResponse = 0);
        // these rows are prep-filtered in reality but the solve code guards.
        const bool zeroResp = (i % 23 == 5);
        // inverted-clamp stripe: minImpulse > maxImpulse, distinguishing the
        // exact clamp nesting min(max, max(min, x)) from max(min, min(max, x)).
        const bool invClamp = (i % 29 == 7);
        const float cfmV = zeroResp ? 0.f : rf(0.f, 0.05f);
        s.hdr2 = F4{ cfmV, refScale ? rf(0.3f, 1.5f) : 1.f, refScale ? rf(0.3f, 1.5f) : 1.f, 0.f };
        if (zeroResp && numRows) ++cZeroResp;
        if (invClamp && numRows) ++cInvClamp;
        s.vel0Top = F4{ rf(-3, 3), rf(-3, 3), rf(-3, 3), 0 };
        s.vel0Bot = F4{ rf(-3, 3), rf(-3, 3), rf(-3, 3), 0 };
        s.vel1Top = F4{ rf(-3, 3), rf(-3, 3), rf(-3, 3), 0 };
        s.vel1Bot = F4{ rf(-3, 3), rf(-3, 3), rf(-3, 3), 0 };
        s.imp0Top = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), 0 };
        s.imp0Bot = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), 0 };
        s.imp1Top = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), 0 };
        s.imp1Bot = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), 0 };

        for (uint32_t r = 0; r < numRows; ++r)
        {
            const int mode = static_cast<int>(rng() % 4);
            uint32_t flags = 0;
            if (mode == 0) { flags = SC_SPRING; ++cSpring; }
            else if (mode == 1) { flags = SC_SPRING | SC_ACCEL; ++cSpring; ++cAccel; }
            else ++cHard;
            // narrow clamp window some of the time to force min/max clamping
            const bool tightClamp = (rng() % 3 == 0);
            float minI = tightClamp ? rf(-0.02f, 0.f) : rf(-1e3f, 0.f);
            float maxI = tightClamp ? rf(0.f, 0.02f) : rf(0.f, 1e3f);
            if (invClamp) { minI = rf(0.5f, 1.f); maxI = rf(-1.f, -0.5f); }  // min > max
            if (tightClamp) { ++cClampMax; ++cClampMin; }
            const float resp0W = zeroResp ? 0.f : rf(0.05f, 2.f);
            const float resp1W = zeroResp ? 0.f : rf(0.05f, 2.f);
            s.rows[r].lin0XYZ_minImpulse = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), minI };
            s.rows[r].lin1XYZ_maxImpulse = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), maxI };
            s.rows[r].ang0XYZ_resp0 = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), resp0W };
            s.rows[r].ang1XYZ_resp1 = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), resp1W };
            s.rows[r].misc = F4{ rf(-2, 2), rf(-5, 5), rf(-5, 5), rf(-0.1f, 0.1f) };
            s.rows[r].flagsU = U4{ flags, 0, 0, 0 };
            s.rows[r].dRA = F4{ rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), rf(-0.5f, 0.5f) };
            s.rows[r].dRA2 = F4{ rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), rf(-0.5f, 0.5f) };
            s.rows[r].dRB = F4{ rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), rf(-0.5f, 0.5f) };
        }

        in[i] = s;
        refAs1dPGS(s, ref[i]);
    }

    std::printf("[ info ] as1 cov: rows0 %zu spring %zu accel %zu hard %zu clampMax %zu clampMin %zu "
                "resid %zu refScale %zu zeroResp %zu invClamp %zu\n",
                cRows0, cSpring, cAccel, cHard, cClampMax, cClampMin, cResid, cRefScale, cZeroResp,
                cInvClamp);
    PXG_CHECK(cRows0 > 100);
    PXG_CHECK(cSpring > 1000);
    PXG_CHECK(cAccel > 400);
    PXG_CHECK(cHard > 1000);
    PXG_CHECK(cClampMax > 1000);
    PXG_CHECK(cResid > 1000);
    PXG_CHECK(cRefScale > 400);
    PXG_CHECK(cZeroResp > 60);
    PXG_CHECK(cInvClamp > 50);

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<As1Out> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npArtiSolve1D.spv", "as1dPGS",
             { (uint32_t)sizeof(As1In), (uint32_t)sizeof(As1Out) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<As1Out>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* rfp = &ref[i].vel0Top.x;
        const float* qfp = &g[i].vel0Top.x;
        const size_t nf = sizeof(As1Out) / 4;
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
    PXG_CHECK_MSG(mm == 0, "as1dPGS outputs bitwise-identical (mod +-0)");
}

}  // namespace
