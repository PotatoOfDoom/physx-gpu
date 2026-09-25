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

// Verification of npSbDeltaLambdaGradC.slang (sbDeltaLambdaGradC): the XPBD
// soft-body per-constraint delta-lambda accumulation compute_deltaLambdaXgradC
// (softBodyGM.cu L558-604), against an exact CPU transcription.  Outputs
// bitwise mod +-0 (eqFZ).

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
struct DlIn  { F4 misc, misc2, fc0, fc1, fc2, qc0, qc1, qc2, v0, v1, v2, v3, acc0, acc1, acc2; };
struct DlOut { F4 dc0, dc1, dc2, scalars; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed

struct V3 { float x, y, z; };
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static float magSq3(V3 v) { return v.x * v.x + v.y * v.y + v.z * v.z; }
static float pabs(float a) { return a < 0.f ? -a : a; }
static V3 mk(float x, float y, float z) { return V3{ x, y, z }; }

static void refDl(const DlIn& s, DlOut& o)
{
    const float C = s.misc.x, alphaTilde = s.misc.z, dtInv = s.misc.w;
    float lambda = s.misc.y;
    const float damping = s.misc2.x, scaledDamping = s.misc2.y;

    const V3 r0 = mk(s.qc0.x, s.qc1.x, s.qc2.x);
    const V3 r1 = mk(s.qc0.y, s.qc1.y, s.qc2.y);
    const V3 r2 = mk(s.qc0.z, s.qc1.z, s.qc2.z);
    const V3 p0 = mk(s.fc0.x, s.fc1.x, s.fc2.x);
    const V3 p1 = mk(s.fc0.y, s.fc1.y, s.fc2.y);
    const V3 p2 = mk(s.fc0.z, s.fc1.z, s.fc2.z);
    const V3 dCdx1 = mk(dot3(r0, p0), dot3(r0, p1), dot3(r0, p2));
    const V3 dCdx2 = mk(dot3(r1, p0), dot3(r1, p1), dot3(r1, p2));
    const V3 dCdx3 = mk(dot3(r2, p0), dot3(r2, p1), dot3(r2, p2));
    const V3 dCdx0 = mk(-dCdx1.x - dCdx2.x - dCdx3.x, -dCdx1.y - dCdx2.y - dCdx3.y, -dCdx1.z - dCdx2.z - dCdx3.z);

    const float w0 = s.v0.w, w1 = s.v1.w, w2 = s.v2.w, w3 = s.v3.w;
    const float sumWMag = magSq3(dCdx0) * w0 + magSq3(dCdx1) * w1 + magSq3(dCdx2) * w2 + magSq3(dCdx3) * w3;

    float deltaLambda;
    if (damping == 0.f)
    {
        const float denom = sumWMag + alphaTilde;
        if (pabs(denom) < 1.0e-15f)
        {
            o.dc0 = s.acc0; o.dc1 = s.acc1; o.dc2 = s.acc2;
            o.scalars = F4{ lambda, 0.f, 0.f, 0.f };
            return;
        }
        deltaLambda = (-C - alphaTilde * lambda) / denom;
    }
    else
    {
        const float denom = (1.0f + damping * dtInv) * sumWMag + alphaTilde;
        if (pabs(denom) < 1.0e-15f)
        {
            o.dc0 = F4{ 0, 0, 0, 0 }; o.dc1 = F4{ 0, 0, 0, 0 }; o.dc2 = F4{ 0, 0, 0, 0 };
            o.scalars = F4{ lambda, 0.f, 0.f, 0.f };
            return;
        }
        const float dampDot = dot3(dCdx0, mk(s.v0.x, s.v0.y, s.v0.z)) + dot3(dCdx1, mk(s.v1.x, s.v1.y, s.v1.z))
                            + dot3(dCdx2, mk(s.v2.x, s.v2.y, s.v2.z)) + dot3(dCdx3, mk(s.v3.x, s.v3.y, s.v3.z));
        deltaLambda = (-C - alphaTilde * lambda - scaledDamping * dampDot) / denom;
    }

    lambda = lambda + deltaLambda;
    o.dc0 = F4{ s.acc0.x + deltaLambda * dCdx1.x, s.acc0.y + deltaLambda * dCdx1.y, s.acc0.z + deltaLambda * dCdx1.z, 0.f };
    o.dc1 = F4{ s.acc1.x + deltaLambda * dCdx2.x, s.acc1.y + deltaLambda * dCdx2.y, s.acc1.z + deltaLambda * dCdx2.z, 0.f };
    o.dc2 = F4{ s.acc2.x + deltaLambda * dCdx3.x, s.acc2.y + deltaLambda * dCdx3.y, s.acc2.z + deltaLambda * dCdx3.z, 0.f };
    o.scalars = F4{ lambda, 0.f, 0.f, 0.f };
}

PXG_TEST(npSbDeltaLambdaGradC)
{
    std::mt19937 rng(232001);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<DlIn> in(N);
    std::vector<DlOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        DlIn s{};
        float damping = ((i % 2u) == 0u) ? 0.f : rf(0.1f, 3.f);   // both damping branches
        s.misc = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(0.f, 2.f), rf(10.f, 120.f) };  // C, lambda, alphaTilde, dtInv
        s.misc2 = F4{ damping, rf(0.f, 2.f), 0.f, 0.f };                              // damping, scaledDamping
        s.fc0 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
        s.fc1 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
        s.fc2 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
        s.qc0 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
        s.qc1 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
        s.qc2 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
        float mi0 = rf(0.2f, 3.f), mi1 = rf(0.2f, 3.f), mi2 = rf(0.2f, 3.f), mi3 = rf(0.2f, 3.f);
        s.v0 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), mi0 };
        s.v1 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), mi1 };
        s.v2 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), mi2 };
        s.v3 = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), mi3 };
        s.acc0 = F4{ rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-1.f, 1.f), 0.f };
        s.acc1 = F4{ rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-1.f, 1.f), 0.f };
        s.acc2 = F4{ rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-1.f, 1.f), 0.f };

        // denom-guard stripe: all inverse masses 0 and alphaTilde ~ 1e-16 so
        // |denom| < 1e-15.  For the no-damping half (i%2==0) this returns the
        // accumulator UNCHANGED; for the damping half it ZEROES the accumulator.
        if ((i % 5u) == 1u)
        {
            s.v0.w = s.v1.w = s.v2.w = s.v3.w = 0.f;
            s.misc.z = 1.0e-16f;   // alphaTilde
        }
        in[i] = s;
        refDl(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<DlOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npSbDeltaLambdaGradC.spv", "sbDeltaLambdaGradC",
             { (uint32_t)sizeof(DlIn), (uint32_t)sizeof(DlOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<DlOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* rfp = &ref[i].dc0.x;
        const float* qfp = &g[i].dc0.x;
        const size_t nf = sizeof(DlOut) / 4;
        bool ok = true;
        for (size_t xi = 0; xi < nf && ok; ++xi) ok = eqFZ(rfp[xi], qfp[xi]);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u damp %g lamR %g lamG %g dc0R (%g %g %g) dc0G (%g %g %g)\n", i, in[i].misc2.x,
                            ref[i].scalars.x, g[i].scalars.x,
                            ref[i].dc0.x, ref[i].dc0.y, ref[i].dc0.z, g[i].dc0.x, g[i].dc0.y, g[i].dc0.z);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "sbDeltaLambdaGradC outputs bitwise-identical (mod +-0)");
}

}  // namespace
