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

// Verification of npArtiClampDriveImp.slang (clampDriveImp): the articulation
// drive-impulse performance-envelope clamp (computeBoundsForNegativeImpulse,
// computeBoundsForPositiveImpulse, clampDriveImpulse; DyCpuGpuArticulation.h
// L197-331), against an exact CPU transcription.  Outputs bitwise mod +-0.

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
struct CdIn  { F4 a, b; };
struct CdOut { F4 outs; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b) || (a == 0.f && b == 0.f); }

static const float PX_MAX_F32 = u2f(0x7f7fffffu);
static const float PX_EPS_F32 = u2f(0x34000000u);

static float pmin(float a, float b) { return (a < b) ? a : b; }
static float pmax(float a, float b) { return (a > b) ? a : b; }
static float pclamp(float v, float lo, float hi) { return pmin(hi, pmax(lo, v)); }
static float pabs(float a) { return a < 0.f ? -a : a; }

struct F2 { float x, y; };

static F2 boundsNeg(float velDeviation, float maxv, float denom1, float denom2, float impulse)
{
    if (pabs(denom1) < PX_EPS_F32) return F2{ -PX_MAX_F32, PX_MAX_F32 };
    if (pabs(denom2) < PX_EPS_F32)
    {
        const float bound1 = (velDeviation - maxv) / denom1;
        float bound2;
        if (pabs(velDeviation + maxv) < PX_EPS_F32) bound2 = PX_MAX_F32;
        else bound2 = (velDeviation + maxv) > 0.f ? PX_MAX_F32 : -PX_MAX_F32;
        return F2{ bound1, pmin(0.f, bound2) };
    }
    const float bound1 = (velDeviation - maxv) / denom1;
    const float bound2 = (velDeviation + maxv) / denom2;
    if (denom2 >= 0.f) return F2{ pmax(bound1, impulse), pmin(0.f, bound2) };
    return F2{ pmax(pmax(bound1, bound2), impulse), 0.f };
}

static F2 boundsPos(float velDeviation, float maxv, float denom1, float denom2, float impulse)
{
    if (pabs(denom2) < PX_EPS_F32) return F2{ -PX_MAX_F32, PX_MAX_F32 };
    if (pabs(denom1) < PX_EPS_F32)
    {
        const float bound2 = (velDeviation + maxv) / denom2;
        float bound1;
        if (pabs(velDeviation - maxv) < PX_EPS_F32) bound1 = -PX_MAX_F32;
        else bound1 = (velDeviation - maxv) > 0.f ? PX_MAX_F32 : -PX_MAX_F32;
        return F2{ pmax(0.f, bound1), bound2 };
    }
    const float bound1 = (velDeviation - maxv) / denom1;
    const float bound2 = (velDeviation + maxv) / denom2;
    if (denom1 >= 0.f) return F2{ pmax(0.f, bound1), pmin(bound2, impulse) };
    return F2{ 0.f, pmin(pmin(bound1, bound2), impulse) };
}

static float clampDriveImpulse(float jointVel0, float driveImpulse0, float driveImpulse, float response,
                               float maxJointVel, float maxImpulse, float speedImpulseGradient, float vdr)
{
    const float velDeviation = driveImpulse0 * response - jointVel0;
    F2 bounds1, bounds2;
    if (driveImpulse > 0.f)
    {
        float denom1 = response - speedImpulseGradient;
        float denom2 = response + speedImpulseGradient;
        bounds1 = boundsPos(velDeviation, maxJointVel, denom1, denom2, driveImpulse);
        if (vdr == 0.f) bounds2 = F2{ -maxImpulse, maxImpulse };
        else { denom1 = response - 1.f / vdr; denom2 = response + 1.f / vdr; bounds2 = boundsPos(velDeviation, maxImpulse / vdr, denom1, denom2, driveImpulse); }
    }
    else
    {
        float denom1 = response + speedImpulseGradient;
        float denom2 = response - speedImpulseGradient;
        bounds1 = boundsNeg(velDeviation, maxJointVel, denom1, denom2, driveImpulse);
        if (vdr == 0.f) bounds2 = F2{ -maxImpulse, maxImpulse };
        else { denom1 = response + 1.f / vdr; denom2 = response - 1.f / vdr; bounds2 = boundsNeg(velDeviation, maxImpulse / vdr, denom1, denom2, driveImpulse); }
    }
    const float lowerBound = pmax(bounds1.x, bounds2.x);
    const float upperBound = pmin(bounds1.y, bounds2.y);
    if (lowerBound > upperBound) return 0.f;
    return pclamp(driveImpulse, lowerBound, upperBound);
}

static void refClamp(const CdIn& s, CdOut& o)
{
    const float r = clampDriveImpulse(s.a.x, s.a.y, s.a.z, s.a.w, s.b.x, s.b.y, s.b.z, s.b.w);
    o.outs = F4{ r, 0.f, 0.f, 0.f };
}

PXG_TEST(npArtiClampDriveImp)
{
    std::mt19937 rng(207001);
    const uint32_t N = 5000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<CdIn> in(N);
    std::vector<CdOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        CdIn s{};
        const float jointVel0 = rf(-3.f, 3.f);
        const float driveImpulse0 = rf(-3.f, 3.f);
        float driveImpulse = rf(-3.f, 3.f);
        if ((i % 2u) == 0u) driveImpulse = pabs(driveImpulse) + 0.01f;   // > 0 path
        else driveImpulse = -pabs(driveImpulse) - 0.01f;                 // <= 0 path
        float response = rf(0.5f, 4.f);
        float maxJointVel = rf(0.5f, 4.f);
        float maxImpulse = rf(0.5f, 4.f);
        float speedImpulseGradient = rf(0.1f, 3.f);
        float vdr = ((i % 3u) == 0u) ? 0.f : rf(0.2f, 3.f);              // vdr==0 branch

        // denom-epsilon stripe: make response == speedImpulseGradient so one denom is 0
        if ((i % 5u) == 1u) speedImpulseGradient = response;
        // velDeviation +/- max near zero stripe (combine with a denom-zero to reach it)
        if ((i % 7u) == 2u) { maxJointVel = 0.f; }

        s.a = F4{ jointVel0, driveImpulse0, driveImpulse, response };
        s.b = F4{ maxJointVel, maxImpulse, speedImpulseGradient, vdr };
        in[i] = s;
        refClamp(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<CdOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npArtiClampDriveImp.spv", "clampDriveImp",
             { (uint32_t)sizeof(CdIn), (uint32_t)sizeof(CdOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<CdOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (!eqFZ(ref[i].outs.x, g[i].outs.x))
        {
            if (mm < 5)
                std::printf("MM %u R %g G %g (di=%g resp=%g sig=%g vdr=%g)\n", i, ref[i].outs.x, g[i].outs.x,
                            in[i].a.z, in[i].a.w, in[i].b.z, in[i].b.w);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "clampDriveImp outputs bitwise-identical (mod +-0)");
}

}  // namespace
