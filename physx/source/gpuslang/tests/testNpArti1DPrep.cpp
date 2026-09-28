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

// Verification of npArti1DPrep.slang entry a1dPGS: preprocessRowsBlock +
// setUpArti1DConstraintBlock PGS (artiConstraintPrep2.cu L1396-1540) with
// intializeBlock1D (jointConstraintBlockPrep.cuh L244-298),
// queryReduced1dConstraintSolverConstantsPGS + computeBounceVelocity
// (constraintPrepShared.cuh L186-220 / DyCpuGpu1dConstraint.h) and
// Dy::computeJointSpeedPGS (DyCpuGpu1dConstraint.h L90-130) against an
// exact CPU transcription.  Integer outputs (outCount, sortedRowIndices,
// cmod.flags) memcmp; float outputs bitwise modulo the documented +-0
// class (eqFZ).  Output lanes that CUDA leaves uninitialized (slots past
// outCount except the ang0Writeback of trailing skipped rows;
// appliedForce/residual are explicit zero stores) are zero-pinned on both
// sides by construction.

#include "pxgTestFramework.h"

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };

struct J1In
{
    U4 bMeta;
    F4 b0resp[9];
    F4 b0velTop, b0velBot, b0q, b0p;
    F4 b1resp[9];
    F4 b1velTop, b1velBot, b1q, b1p;
    U4 cfg;
    F4 invMassScale;
    F4 scal;
    F4 rvel[80];
    U4 rparU[20];
    F4 rparF[20];
};

struct J1Out
{
    U4 hdr;
    U4 sorted[5];
    U4 modFlags[5];
    F4 conOut[100];
    F4 modOut[40];
    F4 artiR[60];
};

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed

static const uint32_t F_SPRING = 1u << 0;
static const uint32_t F_ACCEL = 1u << 1;
static const uint32_t F_RESTITUTION = 1u << 2;
static const uint32_t F_KEEPBIAS = 1u << 3;
static const uint32_t F_OUTPUT_FORCE = 1u << 4;
static const uint32_t F_HAS_DRIVE_LIMIT = 1u << 5;
static const uint32_t CF_DRIVE_LIMITS_ARE_FORCES = 1u << 5;
static const uint32_t SC_SPRING = 1u << 5;
static const uint32_t SC_ACCEL = 1u << 7;
static const uint32_t SC_OUTPUT_FORCE = 1u << 1;

static float rmAt(const F4* m, int j, int i)
{
    const int f = j * 6 + i;
    return (&m[f / 4].x)[f % 4];
}

// PxSpatialMatrix::multiplyInertia
static void rMulInertia(const F4* resp, const float v[3], float out[3])
{
    for (int r = 0; r < 3; ++r)
        out[r] = rmAt(resp, 3 + r, 0) * v[0] + rmAt(resp, 3 + r, 1) * v[1] +
                 rmAt(resp, 3 + r, 2) * v[2];
}

// PxSpatialMatrix::operator* -- j ascending accumulation
static void rSpatialMul(const F4* resp, const float vTop[3], const float vBot[3], float oTop[3],
                        float oBot[3])
{
    float st[6] = { vTop[0], vTop[1], vTop[2], vBot[0], vBot[1], vBot[2] };
    float result[6];
    for (int i = 0; i < 6; ++i)
    {
        result[i] = 0.f;
        for (int j = 0; j < 6; ++j)
            result[i] += rmAt(resp, j, i) * st[j];
    }
    oTop[0] = result[0]; oTop[1] = result[1]; oTop[2] = result[2];
    oBot[0] = result[3]; oBot[1] = result[4]; oBot[2] = result[5];
}

struct RBody
{
    const F4* resp;
    float velTop[3], velBot[3];
    float cfm;
    bool isArt, isKin;
};

// createImpulseResponseVector
static void rCirv(const RBody& b, const float top[3], const float bot[3], float oTop[3],
                  float oBot[3])
{
    if (b.isArt || b.isKin)
    {
        for (int k = 0; k < 3; ++k) { oTop[k] = top[k]; oBot[k] = bot[k]; }
    }
    else
    {
        rMulInertia(b.resp, top, oTop);
        for (int k = 0; k < 3; ++k) oBot[k] = bot[k];
    }
}

static float rDot3(const float a[3], const float b[3])
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// projectVelocity = velocity.dot(v)
static float rProjVel(const RBody& b, const float vTop[3], const float vBot[3])
{
    return rDot3(b.velTop, vTop) + rDot3(b.velBot, vBot);
}

static float pxMaxR(float a, float b) { return (a > b) ? a : b; }   // PxMax<float> = selectMax

// Dy::computeBounceVelocity
static float rBounceVel(uint32_t flags, float jointSpeed, float bounceThreshold, float restitution,
                        float geometricError)
{
    const float bounceVel = jointSpeed * (-restitution);
    if ((flags & F_RESTITUTION) != 0 && (-jointSpeed > bounceThreshold) &&
        ((bounceVel * geometricError) <= 0.f))
        return bounceVel;
    return 0.f;
}

// queryReduced1dConstraintSolverConstantsPGS (constraintPrepShared.cuh L186-220)
static void rQueryReducedPGS(uint32_t flags, float stiffness, float damping, float restitution,
                             float bounceThreshold, float geometricError, float velocityTarget,
                             float jointSpeed, float erp, float simDt, float recipSimDt,
                             float& coeff0, float& coeff1)
{
    if (flags & F_SPRING)
    {
        coeff0 = simDt * simDt * stiffness + simDt * damping;
        coeff1 = simDt * (damping * velocityTarget - stiffness * geometricError);
    }
    else
    {
        const float bounceVel =
            rBounceVel(flags, jointSpeed, bounceThreshold, restitution, geometricError);
        if (bounceVel != 0.f)
        {
            coeff0 = bounceVel;
            coeff1 = bounceVel;
        }
        else
        {
            const float geomError = geometricError * erp;
            coeff0 = velocityTarget - geomError * recipSimDt;
            coeff1 = (!(flags & F_KEEPBIAS)) ? velocityTarget : coeff0;
        }
    }
}

struct RefStats
{
    bool anySkip, trailingSkip, anyBounce, anyDrive;
    // P9.86c (TGS only): live rows on which the pre-fix port spelling of
    // computeJointSpeedTGS (llvmpipe: 0 - v0 -> -v0) differs from CUDA's IEEE result,
    // and live rows reaching the kinematic-body-1-only 0 + (-0)
    uint32_t jsTwinK0, jsKin1NegZero;
};

// exact transcription of preprocessRowsBlock + setUpArti1DConstraintBlock PGS
static void refA1dPGS(const J1In& s, J1Out& o, RefStats& st)
{
    std::memset(&o, 0, sizeof(o));
    st = RefStats{ false, false, false, false };

    RBody b0{ s.b0resp, { s.b0velTop.x, s.b0velTop.y, s.b0velTop.z },
              { s.b0velBot.x, s.b0velBot.y, s.b0velBot.z }, s.b0p.w, s.bMeta.x != 0,
              s.bMeta.y != 0 };
    RBody b1{ s.b1resp, { s.b1velTop.x, s.b1velTop.y, s.b1velTop.z },
              { s.b1velBot.x, s.b1velBot.y, s.b1velBot.z }, s.b1p.w, s.bMeta.z != 0,
              s.bMeta.w != 0 };

    const uint32_t numRows = s.cfg.x < 20u ? s.cfg.x : 20u;
    const uint32_t cdFlags = s.cfg.y;
    const float dt = s.scal.x;
    const float recipdt = s.scal.y;

    // preprocessRowsBlock: stable insertion sort on solveHint (strict <)
    uint32_t sorted[20];
    for (uint32_t i = 0; i < numRows; ++i)
    {
        const uint32_t hintI = s.rparU[i].y;
        uint32_t j = i;
        for (; j > 0 && hintI < s.rparU[sorted[j - 1]].y; --j)
            sorted[j] = sorted[j - 1];
        sorted[j] = i;
    }
    for (uint32_t i = 0; i < numRows; ++i)
        (&o.sorted[i / 4].x)[i % 4] = sorted[i];

    const float erp = 1.0f;
    uint32_t outCount = 0;

    for (uint32_t i = 0; i < numRows; ++i)
    {
        const uint32_t index = sorted[i];
        const uint32_t rpFlags = s.rparU[index].x;

        const bool driveGate = (rpFlags & F_HAS_DRIVE_LIMIT) != 0 &&
                               (cdFlags & CF_DRIVE_LIMITS_ARE_FORCES) != 0;
        const float driveScale = driveGate ? std::fmin(dt, 1.0f) : 1.0f;
        if (driveGate)
            st.anyDrive = true;

        const F4 lin0 = s.rvel[index * 4 + 0];
        const F4 lin1 = s.rvel[index * 4 + 1];
        const F4 ang0 = s.rvel[index * 4 + 2];
        const F4 ang1 = s.rvel[index * 4 + 3];

        const float clin0[3] = { lin0.x, lin0.y, lin0.z };
        const float clin1[3] = { lin1.x, lin1.y, lin1.z };
        const float cang0[3] = { ang0.x, ang0.y, ang0.z };
        const float cang1[3] = { ang1.x, ang1.y, ang1.z };
        const float nclin1[3] = { -lin1.x, -lin1.y, -lin1.z };
        const float ncang1[3] = { -ang1.x, -ang1.y, -ang1.z };

        const float minImpulse = lin1.w * driveScale;
        const float maxImpulse = ang1.w * driveScale;

        float r0Top[3], r0Bot[3], r1Top[3], r1Bot[3];
        rCirv(b0, cang0, clin0, r0Top, r0Bot);
        rCirv(b1, ncang1, nclin1, r1Top, r1Bot);

        const float sc0Top[3] = { clin0[0] * s.invMassScale.x, clin0[1] * s.invMassScale.x,
                                  clin0[2] * s.invMassScale.x };
        const float sc0Bot[3] = { cang0[0] * s.invMassScale.y, cang0[1] * s.invMassScale.y,
                                  cang0[2] * s.invMassScale.y };
        const float sc1Top[3] = { nclin1[0] * s.invMassScale.z, nclin1[1] * s.invMassScale.z,
                                  nclin1[2] * s.invMassScale.z };
        const float sc1Bot[3] = { ncang1[0] * s.invMassScale.w, ncang1[1] * s.invMassScale.w,
                                  ncang1[2] * s.invMassScale.w };

        float dV0Top[3], dV0Bot[3], dV1Top[3], dV1Bot[3];
        rSpatialMul(b0.resp, sc0Top, sc0Bot, dV0Top, dV0Bot);
        rSpatialMul(b1.resp, sc1Top, sc1Bot, dV1Top, dV1Bot);

        const float resp0 = rDot3(dV0Top, r0Top) + rDot3(dV0Bot, r0Bot);
        const float resp1 = rDot3(dV1Top, r1Top) + rDot3(dV1Bot, r1Bot);
        const float unitResponse = resp0 + resp1;

        // ang0Writeback lands in slot outCount BEFORE the skip
        o.modOut[outCount * 2 + 0].x = cang0[0];
        o.modOut[outCount * 2 + 0].y = cang0[1];
        o.modOut[outCount * 2 + 0].z = cang0[2];

        if (unitResponse <= 1e-12f)  // DY_ARTICULATION_MIN_RESPONSE
        {
            st.anySkip = true;
            if (i == numRows - 1)
                st.trailingSkip = true;
            continue;
        }

        o.artiR[outCount * 3 + 0] = F4{ dV0Bot[0], dV0Bot[1], dV0Bot[2], dV0Top[0] };
        o.artiR[outCount * 3 + 1] = F4{ dV0Top[1], dV0Top[2], dV1Bot[0], dV1Bot[1] };
        o.artiR[outCount * 3 + 2] = F4{ dV1Bot[2], dV1Top[0], dV1Top[1], dV1Top[2] };

        const bool needNormalVel =
            (rpFlags & F_RESTITUTION) != 0 ||
            ((rpFlags & F_SPRING) != 0 && (rpFlags & F_ACCEL) != 0);
        const bool isRigid0 = !b0.isArt;
        const bool isRigid1 = !b1.isArt;

        float jointSpeed = 0.f;
        float initJointSpeed = 0.f;
        if (needNormalVel || isRigid0 || isRigid1)
        {
            const float vel0 = rProjVel(b0, cang0, clin0);
            const float vel1 = rProjVel(b1, cang1, clin1);
            // Dy::computeJointSpeedPGS
            initJointSpeed = 0.f;
            if (isRigid0)
                initJointSpeed = vel0;
            else if (isRigid1)
                initJointSpeed = -vel1;
            jointSpeed = vel0 - vel1;
        }

        // intializeBlock1D
        float coeff0, coeff1;
        rQueryReducedPGS(rpFlags, s.rparF[index].x, s.rparF[index].y, s.rparF[index].z,
                         s.rparF[index].w, lin0.w, ang0.w, jointSpeed, erp, dt, recipdt, coeff0,
                         coeff1);
        if ((rpFlags & F_SPRING) == 0 &&
            rBounceVel(rpFlags, jointSpeed, s.rparF[index].w, s.rparF[index].z, lin0.w) != 0.f)
            st.anyBounce = true;

        o.conOut[outCount * 5 + 0] = F4{ r0Bot[0], r0Bot[1], r0Bot[2], minImpulse };
        o.conOut[outCount * 5 + 1] = F4{ -r1Bot[0], -r1Bot[1], -r1Bot[2], maxImpulse };
        o.conOut[outCount * 5 + 2] = F4{ r0Top[0], r0Top[1], r0Top[2], resp0 };
        o.conOut[outCount * 5 + 3] = F4{ -r1Top[0], -r1Top[1], -r1Top[2], resp1 };
        o.conOut[outCount * 5 + 4] = F4{ initJointSpeed, 0.f, 0.f, 0.f };

        o.modOut[outCount * 2 + 0] = F4{ cang0[0], cang0[1], cang0[2], coeff0 };
        o.modOut[outCount * 2 + 1] = F4{ coeff1, 0.f, 0.f, 0.f };

        uint32_t rowFlags = 0;
        if (rpFlags & F_SPRING)
            rowFlags |= SC_SPRING;
        if (rpFlags & F_ACCEL)
            rowFlags |= SC_ACCEL;
        if (rpFlags & F_OUTPUT_FORCE)
            rowFlags |= SC_OUTPUT_FORCE;
        (&o.modFlags[outCount / 4].x)[outCount % 4] = rowFlags;

        ++outCount;
    }

    o.hdr.x = outCount;
}

PXG_TEST(npArti1DPrep_pgs)
{
    std::mt19937 rng(182001);
    const uint32_t N = 2000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<J1In> in(N);
    std::vector<J1Out> ref(N);
    size_t cRows0 = 0, cSkipSome = 0, cTrailSkip = 0, cSpring = 0, cAccel = 0, cBounce = 0,
           cKeep = 0, cDrive = 0, cOutF = 0, cSortDup = 0, cArtArt = 0, cKin = 0, cFullRows = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        J1In s{};
        const uint32_t k0 = rng() % 3, k1 = rng() % 3;
        s.bMeta = U4{ k0 == 0 ? 1u : 0u, k0 == 2 ? 1u : 0u, k1 == 0 ? 1u : 0u, k1 == 2 ? 1u : 0u };
        if (k0 == 0 && k1 == 0) ++cArtArt;
        if (k0 == 2 || k1 == 2) ++cKin;

        const bool zeroBodies = (i % 13 == 6);
        auto fillBody = [&](F4* resp, F4& vt, F4& vb, F4& q, F4& p) {
            for (int f = 0; f < 9; ++f)
                resp[f] = zeroBodies ? F4{ 0, 0, 0, 0 }
                                     : F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), rf(-1, 1) };
            vt = F4{ rf(-3, 3), rf(-3, 3), rf(-3, 3), rf(-10, 0) };
            vb = F4{ rf(-3, 3), rf(-3, 3), rf(-3, 3), rf(0.5f, 1e5f) };
            q = F4{ 0, 0, 0, 1 };
            p = F4{ rf(-5, 5), rf(-5, 5), rf(-5, 5), rf(0, 0.1f) };
        };
        fillBody(s.b0resp, s.b0velTop, s.b0velBot, s.b0q, s.b0p);
        fillBody(s.b1resp, s.b1velTop, s.b1velBot, s.b1q, s.b1p);

        const uint32_t numRows = (i % 11 == 4) ? 0u : (1u + rng() % 20u);
        if (numRows == 0)
            ++cRows0;
        if (numRows == 20)
            ++cFullRows;
        const uint32_t cdFlags = (rng() % 2) ? CF_DRIVE_LIMITS_ARE_FORCES : 0u;
        s.cfg = U4{ numRows, cdFlags, 0, 0 };
        s.invMassScale = F4{ rf(0, 1.5f), rf(0, 1.5f), rf(0, 1.5f), rf(0, 1.5f) };
        // dt > 1 stripe: unphysical but representable; the ONLY way the
        // fminf(dt, 1) clamp in driveScale is observable (review-182 find)
        s.scal = F4{ (i % 9 == 7) ? rf(1.2f, 2.5f) : rf(0.002f, 0.02f), rf(50.f, 500.f), 0, 0 };

        static const uint32_t hintPool[6] = { 0u, 1u, 2u, 5u, 6u, 256u };
        bool dupHint = false;
        uint32_t seenHints[20];
        for (uint32_t r = 0; r < numRows; ++r)
        {
            uint32_t flags;
            if (rng() % 17 == 0)
            {
                flags = rng() % 128;  // raw 7-bit combo, incl. API-invalid ones
            }
            else
            {
                const bool spring = rng() % 3 == 0;
                const bool accel = spring && (rng() % 2 != 0);
                const bool resti = !spring && (rng() % 2 != 0);
                flags = (spring ? F_SPRING : 0u) | (accel ? F_ACCEL : 0u) |
                        (resti ? F_RESTITUTION : 0u) | ((rng() % 4 == 0) ? F_KEEPBIAS : 0u) |
                        ((rng() % 2) ? F_OUTPUT_FORCE : 0u) |
                        ((rng() % 2) ? F_HAS_DRIVE_LIMIT : 0u) |
                        ((rng() % 2) ? (1u << 6) : 0u);
            }
            if (flags & F_SPRING) ++cSpring;
            if ((flags & F_SPRING) && (flags & F_ACCEL)) ++cAccel;
            if (flags & F_KEEPBIAS) ++cKeep;
            if (flags & F_OUTPUT_FORCE) ++cOutF;

            const uint32_t hint = hintPool[rng() % 6];
            for (uint32_t p = 0; p < r; ++p)
                if (seenHints[p] == hint)
                    dupHint = true;
            seenHints[r] = hint;
            s.rparU[r] = U4{ flags, hint, 0, 0 };
            s.rparF[r] = F4{ rf(0.f, 1e4f), rf(0.f, 1e3f), rf(0.f, 1.f), rf(0.f, 1.5f) };

            const bool zeroRow = !zeroBodies && (rng() % 5 == 0);
            auto jac = [&]() { return zeroRow ? 0.f : rf(-2.f, 2.f); };
            s.rvel[r * 4 + 0] = F4{ jac(), jac(), jac(), rf(-0.5f, 0.5f) };
            s.rvel[r * 4 + 1] = F4{ jac(), jac(), jac(), rf(-1e4f, 0.f) };
            s.rvel[r * 4 + 2] = F4{ jac(), jac(), jac(), rf(-3.f, 3.f) };
            s.rvel[r * 4 + 3] = F4{ jac(), jac(), jac(), rf(0.f, 1e4f) };
        }
        if (dupHint)
            ++cSortDup;

        in[i] = s;
        RefStats st;
        refA1dPGS(s, ref[i], st);
        if (st.anySkip) ++cSkipSome;
        if (st.trailingSkip) ++cTrailSkip;
        if (st.anyBounce) ++cBounce;
        if (st.anyDrive) ++cDrive;
    }

    std::printf("[ info ] j1 cov: rows0 %zu skip %zu trailSkip %zu spring %zu accel %zu bounce %zu "
                "keep %zu drive %zu outF %zu sortDup %zu artArt %zu kin %zu full %zu\n",
                cRows0, cSkipSome, cTrailSkip, cSpring, cAccel, cBounce, cKeep, cDrive, cOutF,
                cSortDup, cArtArt, cKin, cFullRows);
    PXG_CHECK(cRows0 > 100);
    PXG_CHECK(cSkipSome > 800);
    PXG_CHECK(cTrailSkip > 200);
    PXG_CHECK(cSpring > 2000);
    PXG_CHECK(cAccel > 900);
    PXG_CHECK(cBounce > 200);
    PXG_CHECK(cKeep > 1000);
    PXG_CHECK(cDrive > 800);
    PXG_CHECK(cSortDup > 1500);
    PXG_CHECK(cKin > 900);

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<J1Out> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npArti1DPrep.spv", "a1dPGS",
             { (uint32_t)sizeof(J1In), (uint32_t)sizeof(J1Out) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<J1Out>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const J1Out& r = ref[i];
        const J1Out& q = g[i];
        bool ok = std::memcmp(&r.hdr, &q.hdr, sizeof(U4) * 11) == 0;
        const float* rfp = &r.conOut[0].x;
        const float* qfp = &q.conOut[0].x;
        for (size_t x = 0; x < 200 * 4 && ok; ++x)
            ok = eqFZ(rfp[x], qfp[x]);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u cntR %u cntG %u con0R (%g %g %g %g) con0G (%g %g %g %g)\n", i,
                            r.hdr.x, q.hdr.x, r.conOut[0].x, r.conOut[0].y, r.conOut[0].z,
                            r.conOut[0].w, q.conOut[0].x, q.conOut[0].y, q.conOut[0].z,
                            q.conOut[0].w);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "a1dPGS outputs bitwise-identical (mod +-0)");
}


struct T1In
{
    U4 bMeta;
    F4 b0resp[9];
    F4 b0velTop, b0velBot, b0q, b0p;
    F4 b1resp[9];
    F4 b1velTop, b1velBot, b1q, b1p;
    U4 cfg;
    F4 invMassScale;
    F4 scal;
    F4 scal2;
    F4 rvel[80];
    U4 rparU[20];
    F4 rparF[20];
};

struct T1Out
{
    U4 hdr;
    U4 sorted[5];
    U4 conFlags[5];
    F4 conOut[120];
    F4 artiR[60];
};

static const uint32_t F_ANGULAR = 1u << 6;
static const uint32_t SC_KEEP_BIAS = 1u << 2;
static const uint32_t SC_INEQUALITY = 1u << 6;

// Dy::computeMaxBiasVelocityTGS (DyCpuGpu1dConstraint.h L332-361)
static float rMaxBiasVelTGS(uint32_t flags, float jointSpeed, float bounceThreshold,
                            float restitution, float geometricError, bool isExtended,
                            float lengthScale, float recipSimDt)
{
    float maxBiasSpeed = 3.402823466e+38f;
    if (flags & F_SPRING)
    {
        maxBiasSpeed = 3.402823466e+38f;
    }
    else
    {
        const float bounceVel =
            rBounceVel(flags, jointSpeed, bounceThreshold, restitution, geometricError);
        if (bounceVel != 0.f)
            maxBiasSpeed = 0.f;
        else if (flags & F_ANGULAR)
            maxBiasSpeed = recipSimDt * 0.75f;
        else
            maxBiasSpeed = isExtended ? recipSimDt * 1.5f * lengthScale
                                      : recipSimDt * 15.f * lengthScale;
    }
    return maxBiasSpeed;
}

// Dy::queryReduced1dConstraintSolverConstantsTGS (DyCpuGpu1dConstraint.h L489-539)
static void rQueryReducedTGS(uint32_t flags, float stiffness, float damping, float restitution,
                             float bounceThreshold, float geometricError, float velocityTarget,
                             float jointSpeed, float initJointSpeed, float erp, float stepDt,
                             float recipStepDt, float& coeff0, float& coeff1, float& coeff2,
                             float& coeff3)
{
    coeff0 = initJointSpeed;
    if (flags & F_SPRING)
    {
        const float a = stepDt * (stepDt * stiffness + damping);
        const float b = stepDt * (damping * velocityTarget);
        coeff2 = a;
        coeff3 = b;
        if (flags & F_ACCEL)
        {
            const float x = 1.0f / (1.0f + a);
            coeff1 = -x * stiffness * stepDt;
        }
        else
        {
            coeff1 = -stiffness * stepDt;
        }
    }
    else
    {
        const float bounceVel =
            rBounceVel(flags, jointSpeed, bounceThreshold, restitution, geometricError);
        coeff2 = bounceVel;
        coeff3 = velocityTarget;
        if (bounceVel != 0.f)
            coeff1 = 0.f;
        else
            coeff1 = -recipStepDt * erp;
    }
}

// exact transcription of preprocessRowsBlock + setUpArti1DConstraintBlock TGS
// (artiConstraintPrep2.cu L1594-1714) + intializeTGSBlock1D (L1545-1591)
static void refA1dTGS(const T1In& s, T1Out& o, RefStats& st)
{
    std::memset(&o, 0, sizeof(o));
    st = RefStats{ false, false, false, false };

    RBody b0{ s.b0resp, { s.b0velTop.x, s.b0velTop.y, s.b0velTop.z },
              { s.b0velBot.x, s.b0velBot.y, s.b0velBot.z }, s.b0p.w, s.bMeta.x != 0,
              s.bMeta.y != 0 };
    RBody b1{ s.b1resp, { s.b1velTop.x, s.b1velTop.y, s.b1velTop.z },
              { s.b1velBot.x, s.b1velBot.y, s.b1velBot.z }, s.b1p.w, s.bMeta.z != 0,
              s.bMeta.w != 0 };

    const uint32_t numRows = s.cfg.x < 20u ? s.cfg.x : 20u;
    const uint32_t cdFlags = s.cfg.y;
    const float stepDt = s.scal.x;
    const float simDt = s.scal.y;
    const float recipStepDt = s.scal.z;
    const float recipSimDt = s.scal.w;
    const float lengthScale = s.scal2.x;
    const float biasCoefficient = s.scal2.y;

    uint32_t sorted[20];
    for (uint32_t i = 0; i < numRows; ++i)
    {
        const uint32_t hintI = s.rparU[i].y;
        uint32_t j = i;
        for (; j > 0 && hintI < s.rparU[sorted[j - 1]].y; --j)
            sorted[j] = sorted[j - 1];
        sorted[j] = i;
    }
    for (uint32_t i = 0; i < numRows; ++i)
        (&o.sorted[i / 4].x)[i % 4] = sorted[i];

    const float erp = 0.5f * biasCoefficient;
    uint32_t outCount = 0;

    for (uint32_t i = 0; i < numRows; ++i)
    {
        const uint32_t index = sorted[i];
        const uint32_t rpFlags = s.rparU[index].x;
        const uint32_t rpSolveHint = s.rparU[index].y;

        const F4 lin0 = s.rvel[index * 4 + 0];
        const F4 lin1 = s.rvel[index * 4 + 1];
        const F4 ang0 = s.rvel[index * 4 + 2];
        const F4 ang1 = s.rvel[index * 4 + 3];

        const float clin0[3] = { lin0.x, lin0.y, lin0.z };
        const float clin1[3] = { lin1.x, lin1.y, lin1.z };
        const float cang0[3] = { ang0.x, ang0.y, ang0.z };
        const float cang1[3] = { ang1.x, ang1.y, ang1.z };
        const float nclin1[3] = { -lin1.x, -lin1.y, -lin1.z };
        const float ncang1[3] = { -ang1.x, -ang1.y, -ang1.z };

        float r0Top[3], r0Bot[3], r1Top[3], r1Bot[3];
        rCirv(b0, cang0, clin0, r0Top, r0Bot);
        rCirv(b1, ncang1, nclin1, r1Top, r1Bot);

        const float sc0Top[3] = { clin0[0] * s.invMassScale.x, clin0[1] * s.invMassScale.x,
                                  clin0[2] * s.invMassScale.x };
        const float sc0Bot[3] = { cang0[0] * s.invMassScale.y, cang0[1] * s.invMassScale.y,
                                  cang0[2] * s.invMassScale.y };
        const float sc1Top[3] = { nclin1[0] * s.invMassScale.z, nclin1[1] * s.invMassScale.z,
                                  nclin1[2] * s.invMassScale.z };
        const float sc1Bot[3] = { ncang1[0] * s.invMassScale.w, ncang1[1] * s.invMassScale.w,
                                  ncang1[2] * s.invMassScale.w };

        float dV0Top[3], dV0Bot[3], dV1Top[3], dV1Bot[3];
        rSpatialMul(b0.resp, sc0Top, sc0Bot, dV0Top, dV0Bot);
        rSpatialMul(b1.resp, sc1Top, sc1Bot, dV1Top, dV1Bot);

        const float resp0 = rDot3(dV0Top, r0Top) + rDot3(dV0Bot, r0Bot);
        const float resp1 = rDot3(dV1Top, r1Top) + rDot3(dV1Bot, r1Bot);
        const float unitResponse = resp0 + resp1;

        if (unitResponse <= 1e-12f)
        {
            st.anySkip = true;
            if (i == numRows - 1)
                st.trailingSkip = true;
            continue;
        }

        o.artiR[outCount * 3 + 0] = F4{ dV0Bot[0], dV0Bot[1], dV0Bot[2], dV0Top[0] };
        o.artiR[outCount * 3 + 1] = F4{ dV0Top[1], dV0Top[2], dV1Bot[0], dV1Bot[1] };
        o.artiR[outCount * 3 + 2] = F4{ dV1Bot[2], dV1Top[0], dV1Top[1], dV1Top[2] };

        // Dy::computeJointSpeedTGS: ungated, isKinematic-keyed
        float jointSpeed, initJointSpeed;
        {
            const float vel0 = rProjVel(b0, cang0, clin0);
            const float vel1 = rProjVel(b1, cang1, clin1);
            initJointSpeed = 0.f;
            if (b0.isKin)
                initJointSpeed -= vel0;
            if (b1.isKin)
                initJointSpeed += vel1;
            jointSpeed = vel0 - vel1;

            // P9.86c: host twin of the pre-fix port (F1/F2: the +0 seed forwarded into
            // OpFSub(0, v0), which llvmpipe emits as -v0); coverage for the r12 kill
            float jsTwin = 0.f;
            if (b0.isKin)
                jsTwin = -vel0;
            if (b1.isKin)
                jsTwin += vel1;
            if (b0.isKin && !b1.isKin && f2u(jsTwin) != f2u(initJointSpeed))
                ++st.jsTwinK0;
            if (!b0.isKin && b1.isKin && f2u(vel1) == 0x80000000u)
                ++st.jsKin1NegZero;
        }

        const float geometricError = lin0.w;
        const float velocityTarget = ang0.w;
        const float stiffness = s.rparF[index].x;
        const float damping = s.rparF[index].y;
        const float restitution = s.rparF[index].z;
        const float bounceThreshold = s.rparF[index].w;

        const float maxBiasVelocity =
            rMaxBiasVelTGS(rpFlags, jointSpeed, bounceThreshold, restitution, geometricError,
                           true, lengthScale, recipSimDt);

        float coeff0, coeff1, coeff2, coeff3;
        rQueryReducedTGS(rpFlags, stiffness, damping, restitution, bounceThreshold,
                         geometricError, velocityTarget, jointSpeed, initJointSpeed, erp, stepDt,
                         recipStepDt, coeff0, coeff1, coeff2, coeff3);
        if ((rpFlags & F_SPRING) == 0 &&
            rBounceVel(rpFlags, jointSpeed, bounceThreshold, restitution, geometricError) != 0.f)
            st.anyBounce = true;

        o.conOut[outCount * 6 + 0] = F4{ r0Bot[0], r0Bot[1], r0Bot[2], coeff0 };
        o.conOut[outCount * 6 + 1] = F4{ -r1Bot[0], -r1Bot[1], -r1Bot[2], coeff1 };
        o.conOut[outCount * 6 + 2] = F4{ r0Top[0], r0Top[1], r0Top[2], coeff2 };
        o.conOut[outCount * 6 + 3] = F4{ -r1Top[0], -r1Top[1], -r1Top[2], coeff3 };

        const float angularErrorScale = (rpFlags & F_ANGULAR) != 0 ? 1.f : 0.f;
        o.conOut[outCount * 6 + 4] = F4{ geometricError, maxBiasVelocity, 0.f, angularErrorScale };

        // Dy::computeMinMaxImpulseOrForceAsImpulse(simDt)
        const bool driveGate = (rpFlags & F_HAS_DRIVE_LIMIT) != 0 &&
                               (cdFlags & CF_DRIVE_LIMITS_ARE_FORCES) != 0;
        const float driveScale = driveGate ? simDt : 1.0f;
        if (driveGate)
            st.anyDrive = true;
        o.conOut[outCount * 6 + 5] =
            F4{ resp0, resp1, lin1.w * driveScale, ang1.w * driveScale };

        // Dy::raiseInternalFlagsTGS
        uint32_t rowFlags = 0;
        if (rpFlags & F_SPRING) rowFlags |= SC_SPRING;
        if (rpFlags & F_ACCEL) rowFlags |= SC_ACCEL;
        if (rpFlags & F_OUTPUT_FORCE) rowFlags |= SC_OUTPUT_FORCE;
        if (rpFlags & F_KEEPBIAS) rowFlags |= SC_KEEP_BIAS;
        if (rpSolveHint & 1u) rowFlags |= SC_INEQUALITY;
        (&o.conFlags[outCount / 4].x)[outCount % 4] = rowFlags;

        ++outCount;
    }

    o.hdr.x = outCount;
}

PXG_TEST(npArti1DPrep_tgs)
{
    std::mt19937 rng(183001);
    const uint32_t N = 2000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<T1In> in(N);
    std::vector<T1Out> ref(N);
    size_t cRows0 = 0, cSkipSome = 0, cTrailSkip = 0, cSpring = 0, cAccel = 0, cBounce = 0,
           cKeep = 0, cDrive = 0, cIneq = 0, cAngular = 0, cSortDup = 0, cKin = 0;
    size_t cJsK0 = 0, cJsK1Nz = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        T1In s{};
        const uint32_t k0 = rng() % 3, k1 = rng() % 3;
        s.bMeta = U4{ k0 == 0 ? 1u : 0u, k0 == 2 ? 1u : 0u, k1 == 0 ? 1u : 0u, k1 == 2 ? 1u : 0u };
        if (k0 == 2 || k1 == 2) ++cKin;

        const bool zeroBodies = (i % 13 == 6);
        auto fillBody = [&](F4* resp, F4& vt, F4& vb, F4& q, F4& p) {
            for (int f = 0; f < 9; ++f)
                resp[f] = zeroBodies ? F4{ 0, 0, 0, 0 }
                                     : F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), rf(-1, 1) };
            vt = F4{ rf(-3, 3), rf(-3, 3), rf(-3, 3), rf(-10, 0) };
            vb = F4{ rf(-3, 3), rf(-3, 3), rf(-3, 3), rf(0.5f, 1e5f) };
            q = F4{ 0, 0, 0, 1 };
            p = F4{ rf(-5, 5), rf(-5, 5), rf(-5, 5), rf(0, 0.1f) };
        };
        fillBody(s.b0resp, s.b0velTop, s.b0velBot, s.b0q, s.b0p);
        fillBody(s.b1resp, s.b1velTop, s.b1velBot, s.b1q, s.b1p);

        // P9.86c directed class: a kinematic rigid body AT REST jointed to an articulation
        // link (CUDA createPxgSolverExtBody: kinematic -> mSpatialResponse memset to 0).
        //   restCase 1: body 0 kinematic, v0 = +0  -> CUDA 0 - (+0) = +0 (pre-fix port: -0)
        //   restCase 2: body 1 kinematic, v1 = -0  -> CUDA 0 + (-0) = +0 (zero-seed fold probe)
        // No rng draws are added, so the random stream of the other tests is unchanged.
        const uint32_t restCase = (i % 17u == 9u) ? 1u + (i / 17u) % 2u : 0u;
        if (restCase != 0u)
        {
            const bool kin0 = restCase == 1u;
            s.bMeta = kin0 ? U4{ 0u, 1u, 1u, 0u } : U4{ 1u, 0u, 0u, 1u };
            F4* kr = kin0 ? s.b0resp : s.b1resp;
            F4& kt = kin0 ? s.b0velTop : s.b1velTop;
            F4& kb = kin0 ? s.b0velBot : s.b1velBot;
            for (int f = 0; f < 9; ++f)
                kr[f] = F4{ 0.f, 0.f, 0.f, 0.f };
            kt = F4{ 0.f, 0.f, 0.f, kt.w };
            kb = F4{ 0.f, 0.f, 0.f, kb.w };
        }

        const uint32_t numRows = (i % 11 == 4) ? 0u : (1u + rng() % 20u);
        if (numRows == 0)
            ++cRows0;
        const uint32_t cdFlags = (rng() % 2) ? CF_DRIVE_LIMITS_ARE_FORCES : 0u;
        s.cfg = U4{ numRows, cdFlags, 0, 0 };
        s.invMassScale = F4{ rf(0, 1.5f), rf(0, 1.5f), rf(0, 1.5f), rf(0, 1.5f) };
        s.scal = F4{ rf(0.002f, 0.008f), rf(0.008f, 0.033f), rf(120.f, 500.f), rf(30.f, 120.f) };
        s.scal2 = F4{ rf(0.5f, 2.f), rf(0.1f, 2.f), 0, 0 };

        static const uint32_t hintPool[6] = { 0u, 1u, 2u, 5u, 6u, 256u };
        bool dupHint = false;
        uint32_t seenHints[20];
        for (uint32_t r = 0; r < numRows; ++r)
        {
            uint32_t flags;
            if (rng() % 17 == 0)
            {
                flags = rng() % 128;
            }
            else
            {
                const bool spring = rng() % 3 == 0;
                const bool accel = spring && (rng() % 2 != 0);
                const bool resti = !spring && (rng() % 2 != 0);
                flags = (spring ? F_SPRING : 0u) | (accel ? F_ACCEL : 0u) |
                        (resti ? F_RESTITUTION : 0u) | ((rng() % 4 == 0) ? F_KEEPBIAS : 0u) |
                        ((rng() % 2) ? F_OUTPUT_FORCE : 0u) |
                        ((rng() % 2) ? F_HAS_DRIVE_LIMIT : 0u) |
                        ((rng() % 2) ? F_ANGULAR : 0u);
            }
            if (flags & F_SPRING) ++cSpring;
            if ((flags & F_SPRING) && (flags & F_ACCEL)) ++cAccel;
            if (flags & F_KEEPBIAS) ++cKeep;
            if (flags & F_ANGULAR) ++cAngular;

            const uint32_t hint = hintPool[rng() % 6];
            if (hint & 1u) ++cIneq;
            for (uint32_t p = 0; p < r; ++p)
                if (seenHints[p] == hint)
                    dupHint = true;
            seenHints[r] = hint;
            s.rparU[r] = U4{ flags, hint, 0, 0 };
            s.rparF[r] = F4{ rf(0.f, 1e4f), rf(0.f, 1e3f), rf(0.f, 1.f), rf(0.f, 1.5f) };

            const bool zeroRow = !zeroBodies && (rng() % 5 == 0);
            auto jac = [&]() { return zeroRow ? 0.f : rf(-2.f, 2.f); };
            s.rvel[r * 4 + 0] = F4{ jac(), jac(), jac(), rf(-0.5f, 0.5f) };
            s.rvel[r * 4 + 1] = F4{ jac(), jac(), jac(), rf(-1e4f, 0.f) };
            s.rvel[r * 4 + 2] = F4{ jac(), jac(), jac(), rf(-3.f, 3.f) };
            s.rvel[r * 4 + 3] = F4{ jac(), jac(), jac(), rf(0.f, 1e4f) };
            if (restCase == 1u)
            {
                F4& l0 = s.rvel[r * 4 + 0];
                l0.x = std::fabs(l0.x);   // (+0)*(+c) = +0 makes v0 = +0 deterministically
            }
            else if (restCase == 2u)
            {
                F4& l1 = s.rvel[r * 4 + 1];
                F4& a1 = s.rvel[r * 4 + 3];   // every (+0)*(-c) = -0 makes v1 = -0
                l1.x = -std::fabs(l1.x); l1.y = -std::fabs(l1.y); l1.z = -std::fabs(l1.z);
                a1.x = -std::fabs(a1.x); a1.y = -std::fabs(a1.y); a1.z = -std::fabs(a1.z);
            }
        }
        if (dupHint)
            ++cSortDup;

        in[i] = s;
        RefStats st;
        refA1dTGS(s, ref[i], st);
        cJsK0 += st.jsTwinK0;
        cJsK1Nz += st.jsKin1NegZero;
        if (st.anySkip) ++cSkipSome;
        if (st.trailingSkip) ++cTrailSkip;
        if (st.anyBounce) ++cBounce;
        if (st.anyDrive) ++cDrive;
    }

    std::printf("[ info ] t1 cov: rows0 %zu skip %zu trailSkip %zu spring %zu accel %zu bounce %zu "
                "keep %zu drive %zu ineq %zu angular %zu sortDup %zu kin %zu\n",
                cRows0, cSkipSome, cTrailSkip, cSpring, cAccel, cBounce, cKeep, cDrive, cIneq,
                cAngular, cSortDup, cKin);
    PXG_CHECK(cRows0 > 100);
    PXG_CHECK(cSkipSome > 800);
    PXG_CHECK(cTrailSkip > 200);
    PXG_CHECK(cSpring > 2000);
    PXG_CHECK(cAccel > 900);
    PXG_CHECK(cBounce > 200);
    PXG_CHECK(cKeep > 1000);
    PXG_CHECK(cDrive > 600);
    PXG_CHECK(cIneq > 2000);
    PXG_CHECK(cAngular > 3000);
    PXG_CHECK(cKin > 900);
    std::printf("[ info ] t1 jointSpeed at rest: kin0 0-(+0) twin-differs %zu, kin1-only 0+(-0) %zu\n",
                cJsK0, cJsK1Nz);
    PXG_CHECK_MSG(cJsK0 >= 100, "resting kinematic body 0 must make the pre-fix twin differ (%zu)",
                  cJsK0);
    PXG_CHECK_MSG(cJsK1Nz >= 100, "resting kinematic body 1 must reach 0 + (-0) (%zu)", cJsK1Nz);

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<J1In> dummyIn(1);
    std::vector<J1Out> dummyOut(1);
    std::vector<T1Out> outInit(N);
    PxgBufferHandle bDIn = createBufferFrom(dev, dummyIn);
    PxgBufferHandle bDOut = createBufferFrom(dev, dummyOut);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npArti1DPrep.spv", "a1dTGS",
             { (uint32_t)sizeof(J1In), (uint32_t)sizeof(J1Out), (uint32_t)sizeof(T1In),
               (uint32_t)sizeof(T1Out) }, sizeof(P));
    k.dispatch(GROUPS, { bDIn, bDOut, bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<T1Out>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const T1Out& r = ref[i];
        const T1Out& q = g[i];
        bool ok = std::memcmp(&r.hdr, &q.hdr, sizeof(U4) * 11) == 0;
        const float* rfp = &r.conOut[0].x;
        const float* qfp = &q.conOut[0].x;
        for (size_t x = 0; x < 180 * 4 && ok; ++x)
            ok = eqFZ(rfp[x], qfp[x]);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u cntR %u cntG %u con0R (%g %g %g %g) con0G (%g %g %g %g)\n", i,
                            r.hdr.x, q.hdr.x, r.conOut[0].x, r.conOut[0].y, r.conOut[0].z,
                            r.conOut[0].w, q.conOut[0].x, q.conOut[0].y, q.conOut[0].z,
                            q.conOut[0].w);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "a1dTGS outputs bitwise-identical (mod +-0)");
}

}  // namespace
