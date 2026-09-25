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

// Verification of npPsSolvePCTGS.slang (psSolvePCTGS): the TGS PBD
// particle-contact solve output solvePCOutputDeltaVTGS (particlesystem.cu
// L1473-1527) against an exact CPU transcription.  Outputs bitwise mod +-0.

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
struct PcIn  { F4 normal, delta, fric0, fric1, tan, vm, adh, applied; };
struct PcOut { F4 f, fr; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed

static float dot3(F4 a, F4 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static float quart(float x) { return x * x * x * x; }
static float pmin(float a, float b) { return (a < b) ? a : b; }
static float pmax(float a, float b) { return (a > b) ? a : b; }

static void refPc(const PcIn& s, PcOut& o)
{
    const float error = s.normal.w, normalVel = s.delta.w;
    const float tanVel0 = s.fric0.w, tanVel1 = s.fric1.w;
    const float tanDelta0 = s.tan.x, tanDelta1 = s.tan.y, normalDelta = s.tan.z, velMultiplier = s.tan.w;
    const float vmF0 = s.vm.x, vmF1 = s.vm.y, frictionCoefficient = s.vm.z, restOffset = s.vm.w;
    const float adhesion = s.adh.x, adhesionRadiusScale = s.adh.y, dt = s.adh.z;
    const bool isVelocityIteration = f2u(s.adh.w) == 1u;
    float appliedForceX = s.applied.x, appliedForceY = s.applied.y;

    const float projectionDt = isVelocityIteration ? 0.0f : dt;
    const float separation = (error - dot3(s.normal, s.delta) + normalDelta) + normalVel * projectionDt;
    float clamp = -appliedForceX;

    if (adhesionRadiusScale > 0.0f && adhesion > 0.0f)
    {
        const float adhesionRadius = restOffset * adhesionRadiusScale - restOffset;
        if (separation < adhesionRadiusScale)
        {
            clamp = pmin(clamp, -pmax(0.0f, adhesion * quart(1.0f - separation / adhesionRadius) * pmax(0.0f, separation) * velMultiplier));
        }
    }

    const float deltaF = pmax(clamp, -separation * velMultiplier);
    appliedForceX = pmax(0.0f, appliedForceX + deltaF);

    const float maxFriction = std::fabs(appliedForceX) * frictionCoefficient;

    const float biasedF0 = -(dot3(s.fric0, s.delta) - tanDelta0);
    const float biasedF1 = -(dot3(s.fric1, s.delta) - tanDelta1);
    const float requiredF0Force = (biasedF0 + tanVel0 * projectionDt) * vmF0;
    const float requiredF1Force = (biasedF1 + tanVel1 * projectionDt) * vmF1;

    const float requiredForce = std::sqrt(requiredF0Force * requiredF0Force + requiredF1Force * requiredF1Force);
    const float recipRequiredForce = (requiredForce >= 1e-16f) ? (1.0f / requiredForce) : 0.0f;

    const float ratio0 = requiredF0Force * recipRequiredForce;
    const float ratio1 = requiredF1Force * recipRequiredForce;

    const float deltaFr = pmin(requiredForce + appliedForceY, maxFriction) - appliedForceY;
    appliedForceY = appliedForceY + deltaFr;

    o.f = F4{ deltaF, appliedForceX, appliedForceY, 0.f };
    o.fr = F4{ deltaFr * ratio0, deltaFr * ratio1, 0.f, 0.f };
}

PXG_TEST(npPsSolvePCTGS)
{
    std::mt19937 rng(241001);
    const uint32_t N = 8000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<PcIn> in(N);
    std::vector<PcOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        PcIn s{};
        s.normal = F4{ rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-2.f, 2.f) };
        s.delta  = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-3.f, 3.f) };
        s.fric0  = F4{ rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-2.f, 2.f) };
        s.fric1  = F4{ rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-2.f, 2.f) };
        s.tan    = F4{ rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-1.f, 1.f), rf(0.1f, 5.f) };  // tanDelta0, tanDelta1, normalDelta, velMult
        s.vm     = F4{ rf(0.1f, 5.f), rf(0.1f, 5.f), rf(0.f, 1.f), rf(0.1f, 1.f) };   // vmF0, vmF1, fricCoeff, restOffset
        const uint32_t sel = i % 4u;
        const float adhesion = (sel == 0u) ? 0.f : rf(0.1f, 3.f);
        const float adhScale = (sel == 1u) ? 0.f : rf(1.5f, 4.f);
        const uint32_t isVel = (i % 2u);                                              // both velocity-iter values
        s.adh    = F4{ adhesion, adhScale, rf(0.001f, 0.05f), u2f(isVel) };           // adhesion, adhScale, dt, isVelIter
        s.applied = F4{ rf(0.f, 3.f), rf(0.f, 3.f), 0.f, 0.f };
        // requiredForce ~ 0 stripe
        if ((i % 15u) == 7u) { s.fric0 = F4{ 0.f, 0.f, 0.f, 0.f }; s.fric1 = F4{ 0.f, 0.f, 0.f, 0.f }; s.tan.x = 0.f; s.tan.y = 0.f; }
        // zero-friction-coefficient stripe
        if ((i % 15u) == 11u) s.vm.z = 0.f;
        // adhesion-binding stripe (velocity iteration -> projectionDt=0, normal=0, normalDelta=0 -> separation=error)
        if ((i % 23u) == 9u)
        {
            const float sepT = rf(0.3f, 0.9f);
            s.normal = F4{ 0.f, 0.f, 0.f, sepT };
            s.tan = F4{ rf(-1.f, 1.f), rf(-1.f, 1.f), 0.f, 5.0f };   // normalDelta=0, velMult=5
            s.vm = F4{ rf(0.1f, 5.f), rf(0.1f, 5.f), rf(0.f, 1.f), 0.5f };  // restOffset=0.5
            s.adh = F4{ rf(0.5f, 2.f), 2.0f, rf(0.001f, 0.03f), u2f(1u) };  // adhesion, adhScale=2, dt, isVelIter=1
            s.applied = F4{ 0.f, rf(0.f, 3.f), 0.f, 0.f };
        }
        // degenerate-guard stripe: adhScale=1, restOffset=0.5 -> adhesionRadius=0, adhesion=0,
        // isVelIter=1 (projectionDt=0), normal=0, normalDelta=0 -> separation=error in (0,1)<adhScale.
        // The correct `&&` guard SKIPS (adhesion>0 false) so no division-by-zero; a `||` guard
        // mutant ENTERS -> separation/0 = inf -> 0*inf = NaN -> divergent output.  Makes the
        // &&->|| guard mutant observable (NOT a true equivalent, only unobservable off-degenerate).
        if ((i % 29u) == 13u)
        {
            s.normal = F4{ 0.f, 0.f, 0.f, rf(0.2f, 0.9f) };
            s.tan = F4{ rf(-1.f, 1.f), rf(-1.f, 1.f), 0.f, rf(0.1f, 5.f) };  // normalDelta=0
            s.vm = F4{ rf(0.1f, 5.f), rf(0.1f, 5.f), rf(0.f, 1.f), 0.5f };   // restOffset=0.5
            s.adh = F4{ 0.f, 1.0f, rf(0.001f, 0.03f), u2f(1u) };  // adhesion=0, adhScale=1, isVelIter=1
        }
        in[i] = s;
        refPc(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<PcOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npPsSolvePCTGS.spv", "psSolvePCTGS",
             { (uint32_t)sizeof(PcIn), (uint32_t)sizeof(PcOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<PcOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        bool ok = eqFZ(ref[i].f.x, g[i].f.x) && eqFZ(ref[i].f.y, g[i].f.y) && eqFZ(ref[i].f.z, g[i].f.z)
               && eqFZ(ref[i].fr.x, g[i].fr.x) && eqFZ(ref[i].fr.y, g[i].fr.y);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u fR (%g %g %g) fG (%g %g %g) frR (%g %g) frG (%g %g)\n", i,
                            ref[i].f.x, ref[i].f.y, ref[i].f.z, g[i].f.x, g[i].f.y, g[i].f.z,
                            ref[i].fr.x, ref[i].fr.y, g[i].fr.x, g[i].fr.y);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "psSolvePCTGS outputs bitwise-identical (mod +-0)");
}

}  // namespace
