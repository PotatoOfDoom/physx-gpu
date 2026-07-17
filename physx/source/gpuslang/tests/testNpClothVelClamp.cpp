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

// Verification of npClothVelClamp.slang (clothVelClamp): the cloth-vertex
// velocity clamp velocityClamping (FEMClothUtil.cuh L175-194), which caps the
// velocity magnitude and reconciles the position + accumulated delta, against
// an exact CPU transcription.  Outputs bitwise mod +-0 (eqFZ).

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
struct VcIn  { F4 pos, vel, accumDelta, prevPos, misc; };   // misc: maxVel, dt, _, _
struct VcOut { F4 pos, vel, accumDelta, flag; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b) || (a == 0.f && b == 0.f); }

// exact transcription of velocityClamping
static void refClamp(const VcIn& s, VcOut& o)
{
    F4 pos = s.pos;
    F4 vel = s.vel;
    F4 accumDelta = s.accumDelta;
    const F4 prevPos = s.prevPos;
    const float maxVel = s.misc.x;
    const float dt = s.misc.y;

    const float maxVelSq = maxVel * maxVel;
    const float velMagSq = vel.x * vel.x + vel.y * vel.y + vel.z * vel.z;

    if (velMagSq > maxVelSq)
    {
        const float scale = maxVel / std::sqrt(velMagSq);
        vel.x = vel.x * scale; vel.y = vel.y * scale; vel.z = vel.z * scale; vel.w = vel.w * scale;

        F4 newPos;
        newPos.x = prevPos.x + vel.x * dt;
        newPos.y = prevPos.y + vel.y * dt;
        newPos.z = prevPos.z + vel.z * dt;
        newPos.w = prevPos.w + vel.w * dt;
        newPos.w = pos.w;
        vel.w = pos.w;

        F4 delta;
        delta.x = newPos.x - pos.x;
        delta.y = newPos.y - pos.y;
        delta.z = newPos.z - pos.z;
        delta.w = newPos.w - pos.w;
        pos = newPos;
        accumDelta.x = accumDelta.x + delta.x;
        accumDelta.y = accumDelta.y + delta.y;
        accumDelta.z = accumDelta.z + delta.z;
        accumDelta.w = accumDelta.w + delta.w;

        o.pos = pos;
        o.vel = vel;
        o.accumDelta = accumDelta;
        o.flag = F4{ 1.0f, 0.0f, 0.0f, 0.0f };
    }
    else
    {
        o.pos = pos;
        o.vel = vel;
        o.accumDelta = accumDelta;
        o.flag = F4{ 0.0f, 0.0f, 0.0f, 0.0f };
    }
}

PXG_TEST(npClothVelClamp)
{
    std::mt19937 rng(218001);
    const uint32_t N = 5000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<VcIn> in(N);
    std::vector<VcOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        VcIn s{};
        s.pos      = F4{ rf(-5.f, 5.f), rf(-5.f, 5.f), rf(-5.f, 5.f), rf(-2.f, 2.f) };
        s.accumDelta = F4{ rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-1.f, 1.f) };
        s.prevPos  = F4{ rf(-5.f, 5.f), rf(-5.f, 5.f), rf(-5.f, 5.f), rf(-2.f, 2.f) };
        float maxVel = rf(0.5f, 4.f);
        // velocity: alternate below / above the clamp threshold
        float vscale = ((i % 2u) == 0u) ? rf(0.05f, 0.9f) : rf(1.1f, 4.f);  // fraction of maxVel target
        // build a velocity whose xyz magnitude is ~vscale*maxVel
        float vx = rf(-1.f, 1.f), vy = rf(-1.f, 1.f), vz = rf(-1.f, 1.f);
        float vmag = std::sqrt(vx * vx + vy * vy + vz * vz);
        if (vmag < 1e-4f) { vx = 1.f; vy = vz = 0.f; vmag = 1.f; }
        const float target = vscale * maxVel;
        s.vel = F4{ vx / vmag * target, vy / vmag * target, vz / vmag * target, rf(-2.f, 2.f) };
        // exact-boundary stripe: velMagSq == maxVelSq (strict > must NOT fire)
        if ((i % 11u) == 3u)
        {
            s.vel = F4{ maxVel, 0.f, 0.f, rf(-2.f, 2.f) };  // velMagSq == maxVelSq exactly
        }
        // zero-velocity stripe: no clamp
        if ((i % 13u) == 6u) { s.vel = F4{ 0.f, 0.f, 0.f, rf(-2.f, 2.f) }; }
        s.misc = F4{ maxVel, rf(0.005f, 0.05f), 0.f, 0.f };  // maxVel, dt
        in[i] = s;
        refClamp(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<VcOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npClothVelClamp.spv", "clothVelClamp",
             { (uint32_t)sizeof(VcIn), (uint32_t)sizeof(VcOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<VcOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* rfp = &ref[i].pos.x;
        const float* qfp = &g[i].pos.x;
        const size_t nf = sizeof(VcOut) / 4;
        bool ok = true;
        for (size_t x = 0; x < nf && ok; ++x) ok = eqFZ(rfp[x], qfp[x]);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u flagR %g flagG %g posR (%g %g %g) posG (%g %g %g)\n", i,
                            ref[i].flag.x, g[i].flag.x,
                            ref[i].pos.x, ref[i].pos.y, ref[i].pos.z,
                            g[i].pos.x, g[i].pos.y, g[i].pos.z);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "clothVelClamp outputs bitwise-identical (mod +-0)");
}

}  // namespace
