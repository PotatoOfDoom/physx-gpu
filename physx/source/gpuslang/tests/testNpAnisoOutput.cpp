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

// Verification of npAnisoOutput.slang (anisoOutput): the anisotropy output
// assembly stage (anisotropy.cu L386-404) -- sanitize covariance diagonal,
// sqrt to eigenvalue magnitudes, scale by mAnisotropy, clamp to [kmin, kmax],
// and pack the rotation columns + lambda into q1/q2/q3 -- against an exact CPU
// transcription.  Outputs bitwise mod +-0 (eqFZ).

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
struct AoIn  { F4 covDiag, rc0, rc1, rc2, params; };
struct AoOut { F4 q1, q2, q3; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed

// PhysX PxMax(a,b)=selectMax=a>b?a:b, PxMin(a,b)=selectMin=a<b?a:b; Clamp per
// comp = PxMin(t, PxMax(s, a)).
static float clampComp(float a, float s, float t)
{
    const float mx = (s > a) ? s : a;
    return (t < mx) ? t : mx;
}
static float sanitize(float v) { return (v > 0.0f) ? v : 0.0f; }  // max(v, 0.f)

static void refAo(const AoIn& s, AoOut& o)
{
    const float c0 = sanitize(s.covDiag.x), c1 = sanitize(s.covDiag.y), c2 = sanitize(s.covDiag.z);
    float lx = std::sqrt(c0), ly = std::sqrt(c1), lz = std::sqrt(c2);

    const float ks = s.params.x;
    const float kmin = s.params.y * s.params.w;
    const float kmax = s.params.z * s.params.w;

    lx = lx * ks; ly = ly * ks; lz = lz * ks;
    lx = clampComp(lx, kmin, kmax);
    ly = clampComp(ly, kmin, kmax);
    lz = clampComp(lz, kmin, kmax);

    o.q1 = F4{ s.rc0.x, s.rc0.y, s.rc0.z, lx };
    o.q2 = F4{ s.rc1.x, s.rc1.y, s.rc1.z, ly };
    o.q3 = F4{ s.rc2.x, s.rc2.y, s.rc2.z, lz };
}

PXG_TEST(npAnisoOutput)
{
    std::mt19937 rng(237001);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<AoIn> in(N);
    std::vector<AoOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        AoIn s{};
        // covariance diagonal: mostly positive, some negative to exercise sanitize
        s.covDiag = F4{ rf(-0.5f, 4.f), rf(-0.5f, 4.f), rf(-0.5f, 4.f), 0.f };
        s.rc0 = F4{ rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-1.f, 1.f), 0.f };
        s.rc1 = F4{ rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-1.f, 1.f), 0.f };
        s.rc2 = F4{ rf(-1.f, 1.f), rf(-1.f, 1.f), rf(-1.f, 1.f), 0.f };
        // ks (mAnisotropy), anisoMin, anisoMax, contactDistance.  anisoMin < anisoMax
        // so kmin < kmax, and the ranges are chosen so lambda straddles both bounds.
        const float amin = rf(0.1f, 0.6f), amax = rf(0.7f, 1.5f);
        s.params = F4{ rf(0.2f, 3.f), amin, amax, rf(0.5f, 2.f) };
        // stripes: force lambda below kmin (tiny cov) and above kmax (large cov)
        if ((i % 7u) == 2u) s.covDiag = F4{ rf(0.f, 0.001f), rf(0.f, 0.001f), rf(0.f, 0.001f), 0.f };
        if ((i % 7u) == 5u) s.covDiag = F4{ rf(20.f, 50.f), rf(20.f, 50.f), rf(20.f, 50.f), 0.f };
        // all-negative stripe -> sanitize to 0 -> lambda = clamp(0, kmin, kmax) = kmin
        if ((i % 13u) == 6u) s.covDiag = F4{ rf(-3.f, -0.01f), rf(-3.f, -0.01f), rf(-3.f, -0.01f), 0.f };
        in[i] = s;
        refAo(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<AoOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npAnisoOutput.spv", "anisoOutput",
             { (uint32_t)sizeof(AoIn), (uint32_t)sizeof(AoOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<AoOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        bool ok = eqFZ(ref[i].q1.x, g[i].q1.x) && eqFZ(ref[i].q1.y, g[i].q1.y) && eqFZ(ref[i].q1.z, g[i].q1.z) && eqFZ(ref[i].q1.w, g[i].q1.w)
               && eqFZ(ref[i].q2.x, g[i].q2.x) && eqFZ(ref[i].q2.y, g[i].q2.y) && eqFZ(ref[i].q2.z, g[i].q2.z) && eqFZ(ref[i].q2.w, g[i].q2.w)
               && eqFZ(ref[i].q3.x, g[i].q3.x) && eqFZ(ref[i].q3.y, g[i].q3.y) && eqFZ(ref[i].q3.z, g[i].q3.z) && eqFZ(ref[i].q3.w, g[i].q3.w);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u lamR (%g %g %g) lamG (%g %g %g)\n", i,
                            ref[i].q1.w, ref[i].q2.w, ref[i].q3.w,
                            g[i].q1.w, g[i].q2.w, g[i].q3.w);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "anisoOutput outputs bitwise-identical (mod +-0)");
}

}  // namespace
