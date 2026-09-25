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

// Verification of npPsSolvePC.slang (psSolvePC): the PBD particle-contact solve
// output solvePCOutputDeltaV (particlesystem.cu L1421-1471) -- adhesion-clamped
// normal impulse increment + Coulomb friction distribution -- against an exact
// CPU transcription.  Outputs bitwise mod +-0 (eqFZ).

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
struct PcIn  { F4 normal, delta, fric0, fric1, vm, adh, applied; };
struct PcOut { F4 f, fr; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed

static float dot3(F4 a, F4 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static float quart(float x) { return x * x * x * x; }
static float pmin(float a, float b) { return (a < b) ? a : b; }
static float pmax(float a, float b) { return (a > b) ? a : b; }

static void refPc(const PcIn& s, PcOut& o)
{
    const float error = s.normal.w, normalVel = s.delta.w;
    const float tanVel0 = s.fric0.w, tanVel1 = s.fric1.w;
    const float velMultiplier = s.vm.x, vmF0 = s.vm.y, vmF1 = s.vm.z, frictionCoefficient = s.vm.w;
    const float restOffset = s.adh.x, adhesion = s.adh.y, adhesionRadiusScale = s.adh.z, dt = s.adh.w;
    float appliedForceX = s.applied.x, appliedForceY = s.applied.y;

    const float separation = (error - dot3(s.normal, s.delta));
    float clamp = -appliedForceX;

    if (adhesionRadiusScale > 0.0f && adhesion > 0.0f)
    {
        const float adhesionRadius = restOffset * adhesionRadiusScale - restOffset;
        if (separation < adhesionRadiusScale)
        {
            clamp = pmin(clamp, -pmax(0.0f, adhesion * quart(1.0f - separation / adhesionRadius) * pmax(0.0f, separation) * velMultiplier));
        }
    }

    const float deltaF = pmax(clamp, (-separation - normalVel * dt) * velMultiplier);
    appliedForceX = pmax(0.0f, deltaF + appliedForceX);

    const float friction = appliedForceX * frictionCoefficient;

    const float biasedF0 = -(dot3(s.fric0, s.delta));
    const float biasedF1 = -(dot3(s.fric1, s.delta));
    const float requiredF0Force = (biasedF0 + tanVel0 * dt) * vmF0;
    const float requiredF1Force = (biasedF1 + tanVel1 * dt) * vmF1;

    const float requiredForce = std::sqrt(requiredF0Force * requiredF0Force + requiredF1Force * requiredF1Force);
    const float recipRequiredForce = (requiredForce >= 1e-16f) ? (1.0f / requiredForce) : 0.0f;

    const float ratio0 = requiredF0Force * recipRequiredForce;
    const float ratio1 = requiredF1Force * recipRequiredForce;

    const float deltaFr = pmin(requiredForce + appliedForceY, friction) - appliedForceY;
    appliedForceY = appliedForceY + deltaFr;

    o.f = F4{ deltaF, appliedForceX, appliedForceY, 0.f };
    o.fr = F4{ deltaFr * ratio0, deltaFr * ratio1, 0.f, 0.f };
}

PXG_TEST(npPsSolvePC)
{
    std::mt19937 rng(240001);
    const uint32_t N = 8000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<PcIn> in(N);
    std::vector<PcOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        PcIn s{};
        s.normal = F4{ rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-2.f, 2.f) };  // normal.xyz, error
        s.delta  = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-3.f, 3.f) };  // delta.xyz, normalVel
        s.fric0  = F4{ rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-2.f, 2.f) };  // fric0.xyz, tanVel0
        s.fric1  = F4{ rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-2.f, 2.f) };  // fric1.xyz, tanVel1
        s.vm     = F4{ rf(0.1f, 5.f), rf(0.1f, 5.f), rf(0.1f, 5.f), rf(0.f, 1.f) };   // velMult, vmF0, vmF1, fricCoeff
        // adhesion: half enabled (scale in [1.5,4], adhesion>0), half disabled
        const uint32_t sel = i % 4u;
        const float adhesion = (sel == 0u) ? 0.f : rf(0.1f, 3.f);              // disabled when 0
        const float adhScale = (sel == 1u) ? 0.f : rf(1.5f, 4.f);             // disabled when <=0
        s.adh    = F4{ rf(0.1f, 1.f), adhesion, adhScale, rf(0.001f, 0.05f) };  // restOffset, adhesion, adhScale, dt
        s.applied = F4{ rf(0.f, 3.f), rf(0.f, 3.f), 0.f, 0.f };                // appliedForce.x, .y
        // requiredForce ~ 0 stripe: make fric dots + tanVel*dt tiny so recip guard picks 0
        if ((i % 15u) == 7u) { s.fric0 = F4{ 0.f, 0.f, 0.f, 0.f }; s.fric1 = F4{ 0.f, 0.f, 0.f, 0.f }; }
        // zero-friction-coefficient stripe: friction == 0 -> deltaFr clamps against 0
        if ((i % 15u) == 11u) s.vm.w = 0.f;
        // adhesion-binding stripe: set up the regime where the adhesion lower-bound
        // clamp actually WINS the deltaF pmax (so the adhesion branch is observable at
        // the output).  normal=0 makes separation = error directly controllable; a small
        // separation in (0, adhScale) with a moderate adhesionRadius keeps the quart term
        // tiny (adhesionTerm ~ 1e-3), while a large velMultiplier drives `other` very
        // negative so the adhesion clamp binds.  This is what makes the separation<adhScale
        // guard (and the whole adhesion path) observable rather than a no-op.
        if ((i % 23u) == 9u)
        {
            const float sepT = rf(0.3f, 0.9f);
            s.normal = F4{ 0.f, 0.f, 0.f, sepT };          // separation = error = sepT
            s.adh = F4{ 0.5f, rf(0.5f, 2.f), 2.0f, rf(0.001f, 0.03f) };  // restOffset, adhesion, adhScale=2, dt
            s.vm = F4{ 5.0f, rf(0.1f, 5.f), rf(0.1f, 5.f), rf(0.f, 1.f) }; // velMult=5 large
            s.applied = F4{ 0.f, rf(0.f, 3.f), 0.f, 0.f }; // appliedForce.x = 0 so the tiny adhesionTerm binds
        }
        // degenerate-guard stripe: adhScale=1, restOffset=0.5 -> adhesionRadius=0, adhesion=0,
        // 0<separation<1.  The correct `&&` guard (adhesionRadiusScale>0 && adhesion>0) SKIPS
        // the block (adhesion>0 false), so no division-by-zero; a `||` guard mutant would ENTER,
        // hit separation/adhesionRadius = x/0 = inf, and 0*inf = NaN -> divergent output.  This
        // makes the &&->|| guard mutant observable (it is NOT a true equivalent mutant, only
        // unobservable in the non-degenerate domain).
        if ((i % 29u) == 13u)
        {
            s.normal = F4{ 0.f, 0.f, 0.f, rf(0.2f, 0.9f) };  // separation = error in (0,1) < adhScale
            s.adh = F4{ 0.5f, 0.f, 1.0f, rf(0.001f, 0.03f) };  // restOffset=0.5, adhesion=0, adhScale=1
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
    Kernel k("npPsSolvePC.spv", "psSolvePC",
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
    PXG_CHECK_MSG(mm == 0, "psSolvePC outputs bitwise-identical (mod +-0)");
}

}  // namespace
