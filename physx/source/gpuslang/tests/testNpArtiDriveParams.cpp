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

// Verification of npArtiDriveParams.slang (driveParams): the articulation
// implicit-drive scalar family (computeImplicitDriveParamsForceDrive/
// AccelerationDrive, computeDriveImpulse, computeFrictionImpulse) from
// DyCpuGpuArticulation.h, against an exact CPU transcription.  Bitwise mod +-0.

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

struct DpIn  { F4 p0, p1, di, fr, misc; };
struct DpOut { F4 forceDesc0, forceDesc1, accelDesc0, accelDesc1, outs; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b) || (a == 0.f && b == 0.f); }

struct DriveDesc { float tvBias, biasCoef, velMul, impMul, posBias; };

static DriveDesc forceDrive(float stiffness, float damping, float dt, float simDt,
                            float unitResponse, float geomError, float targetVelocity, bool isTGS)
{
    const float a = dt * (dt * stiffness + damping);
    const float b = dt * (damping * targetVelocity);
    const float x = unitResponse > 0.f ? 1.0f / (1.0f + a * unitResponse) : 0.f;
    const float initialBias = geomError - (simDt - dt) * targetVelocity;
    const float biasCoefficient = stiffness * x * dt;
    DriveDesc d;
    d.tvBias = (x * b) + biasCoefficient * initialBias;
    d.velMul = -x * a;
    d.biasCoef = biasCoefficient;
    d.impMul = isTGS ? 1.f : (1.0f - x);
    d.posBias = isTGS ? (biasCoefficient * targetVelocity) : 0.f;
    return d;
}

static DriveDesc accelDrive(float stiffness, float damping, float dt, float simDt,
                            float recipUnitResponse, float geomError, float targetVelocity, bool isTGS)
{
    const float a = dt * (dt * stiffness + damping);
    const float b = dt * (damping * targetVelocity);
    const float x = 1.0f / (1.0f + a);
    const float initialBias = geomError - (simDt - dt) * targetVelocity;
    const float biasCoefficient = stiffness * x * recipUnitResponse * dt;
    DriveDesc d;
    d.tvBias = (x * b * recipUnitResponse) + biasCoefficient * initialBias;
    d.velMul = -x * a * recipUnitResponse;
    d.biasCoef = biasCoefficient;
    d.impMul = isTGS ? 1.f : (1.0f - x);
    d.posBias = isTGS ? (biasCoefficient * targetVelocity) : 0.f;
    return d;
}

static float driveImpulse(float accum, float jointVel, float jointDeltaPos, float elapsed, const DriveDesc& d)
{
    return accum * d.impMul + jointVel * d.velMul + d.tvBias - jointDeltaPos * d.biasCoef + elapsed * d.posBias;
}

static float frictionImpulse(float fImp, float staticF, float dynamicF, float viscous, float jointVel)
{
    if (std::fabs(fImp) > staticF)
    {
        const float lo = -dynamicF - viscous * std::fabs(jointVel);
        const float hi = dynamicF + viscous * std::fabs(jointVel);
        const float mx = (lo > fImp) ? lo : fImp;
        fImp = (hi < mx) ? hi : mx;
    }
    return fImp;
}

static void refDriveParams(const DpIn& s, DpOut& o)
{
    const float stiffness = s.p0.x, damping = s.p0.y, dt = s.p0.z, simDt = s.p0.w;
    const float unitResp = s.p1.x, recipUnitResp = s.p1.y, geomError = s.p1.z, targetVel = s.p1.w;
    const bool isTGS = f2u(s.misc.y) == 1u;
    const bool isAccel = f2u(s.misc.z) == 1u;

    const DriveDesc fd = forceDrive(stiffness, damping, dt, simDt, unitResp, geomError, targetVel, isTGS);
    const DriveDesc ad = accelDrive(stiffness, damping, dt, simDt, recipUnitResp, geomError, targetVel, isTGS);
    const DriveDesc sel = isAccel ? ad : fd;

    const float drimp = driveImpulse(s.di.x, s.di.y, s.di.z, s.di.w, sel);
    const float frimp = frictionImpulse(s.fr.x, s.fr.y, s.fr.z, s.fr.w, s.misc.x);

    o.forceDesc0 = F4{ fd.tvBias, fd.biasCoef, fd.velMul, fd.impMul };
    o.forceDesc1 = F4{ fd.posBias, 0.f, 0.f, 0.f };
    o.accelDesc0 = F4{ ad.tvBias, ad.biasCoef, ad.velMul, ad.impMul };
    o.accelDesc1 = F4{ ad.posBias, 0.f, 0.f, 0.f };
    o.outs = F4{ drimp, frimp, 0.f, 0.f };
}

PXG_TEST(npArtiDriveParams)
{
    std::mt19937 rng(206001);
    const uint32_t N = 4000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<DpIn> in(N);
    std::vector<DpOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        DpIn s{};
        s.p0 = F4{ rf(0.f, 50.f), rf(0.f, 10.f), rf(0.01f, 0.05f), rf(0.01f, 0.05f) };  // stiffness,damping,dt,simDt
        float unitResp = rf(0.1f, 5.f);
        if ((i % 11u) == 3u) unitResp = 0.f;          // force-drive unitResponse<=0 branch (x=0)
        if ((i % 11u) == 7u) unitResp = rf(-5.f, -0.1f);
        const float recipUnitResp = rf(0.1f, 5.f);
        s.p1 = F4{ unitResp, recipUnitResp, rf(-2.f, 2.f), rf(-3.f, 3.f) };  // unitResp,recip,geomError,targetVel
        s.di = F4{ rf(-3.f, 3.f), rf(-3.f, 3.f), rf(-2.f, 2.f), rf(0.f, 0.05f) };  // accum,jointVel,jointDeltaPos,elapsed
        // friction: alternate between abs>staticF (clamp active) and not
        float fImp = rf(-5.f, 5.f);
        float staticF = ((i % 2u) == 0u) ? rf(0.f, 0.5f) : rf(5.f, 8.f);  // small -> clamp fires; large -> passthrough
        float dynamicF = rf(0.f, 2.f), viscous = rf(0.f, 1.f);
        // Guard-boundary stripe: abs(fImp)==staticF exactly AND staticF beyond the
        // clamp band (staticF > dynamicF, viscous=0), so > vs >= diverges (the
        // >= branch would clamp fImp down, the > branch would not).
        if ((i % 13u) == 5u)
        {
            dynamicF = 0.5f; viscous = 0.f;
            staticF = 5.0f;
            fImp = (i & 1u) ? 5.0f : -5.0f;  // abs(fImp) == staticF exactly
        }
        s.fr = F4{ fImp, staticF, dynamicF, viscous };  // fImp,staticF,dynamicF,viscous
        const bool isTGS = (i % 3u) == 0u;
        const bool isAccel = (i % 2u) == 1u;
        s.misc = F4{ rf(-3.f, 3.f), u2f(isTGS ? 1u : 0u), u2f(isAccel ? 1u : 0u), 0.f };  // frictionJointVel, isTGS, isAccel
        in[i] = s;
        refDriveParams(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<DpOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npArtiDriveParams.spv", "driveParams",
             { (uint32_t)sizeof(DpIn), (uint32_t)sizeof(DpOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<DpOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* rfp = &ref[i].forceDesc0.x;
        const float* qfp = &g[i].forceDesc0.x;
        const size_t nf = sizeof(DpOut) / 4;
        bool ok = true;
        for (size_t x = 0; x < nf && ok; ++x) ok = eqFZ(rfp[x], qfp[x]);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u outsR (%g %g) G (%g %g)\n", i, ref[i].outs.x, ref[i].outs.y, g[i].outs.x, g[i].outs.y);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "driveParams outputs bitwise-identical (mod +-0)");
}

}  // namespace
