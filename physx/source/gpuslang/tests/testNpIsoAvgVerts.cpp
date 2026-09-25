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

// Verification of npIsoAvgVerts.slang (isoAvgVerts): the isosurface averageVerts
// blend (isosurfaceExtraction.cu L934-948) -- weight-normalize the accumulator
// via reciprocal-multiply v.xyz/v.w and blend it against the current output with
// blendWeight, clearing the accumulator -- against an exact CPU transcription.
// Outputs bitwise mod +-0 (eqFZ).

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
struct AvIn  { F4 vtx, outp, params; };
struct AvOut { F4 outNew, vtxNew; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed

static void refAv(const AvIn& s, AvOut& o)
{
    const float vw = s.vtx.w;
    const float bw = s.params.x;

    F4 outNew = s.outp;   // v.w <= 0 : unchanged
    if (vw > 0.0f)
    {
        const float inv = 1.0f / vw;                    // v.xyz / v.w (reciprocal-multiply)
        const float rx = s.vtx.x * inv, ry = s.vtx.y * inv, rz = s.vtx.z * inv;
        const float ombw = 1.0f - bw;
        outNew = F4{ ombw * s.outp.x + bw * rx,
                     ombw * s.outp.y + bw * ry,
                     ombw * s.outp.z + bw * rz, 0.f };
    }
    o.outNew = outNew;
    o.vtxNew = F4{ 0.f, 0.f, 0.f, 0.f };
}

PXG_TEST(npIsoAvgVerts)
{
    std::mt19937 rng(239001);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<AvIn> in(N);
    std::vector<AvOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        AvIn s{};
        const float w = rf(0.1f, 40.f);
        // vtx.xyz is a weighted sum: scale so v.xyz/v.w lands near a real position
        s.vtx = F4{ rf(-10.f, 10.f) * w, rf(-10.f, 10.f) * w, rf(-10.f, 10.f) * w, w };
        s.outp = F4{ rf(-10.f, 10.f), rf(-10.f, 10.f), rf(-10.f, 10.f), rf(-5.f, 5.f) };
        s.params = F4{ rf(0.f, 1.5f), 0.f, 0.f, 0.f };  // blendWeight (incl. extrapolation >1)
        // passthrough stripe: v.w <= 0 -> output unchanged (incl. w)
        if ((i % 9u) == 4u) s.vtx.w = (i % 18u == 4u) ? 0.0f : rf(-5.f, -0.01f);
        // full-new stripe: blendWeight == 1 -> output == v.xyz/v.w
        if ((i % 7u) == 2u) s.params.x = 1.0f;
        // full-old stripe: blendWeight == 0 -> output == current x (with w=0)
        if ((i % 7u) == 5u) s.params.x = 0.0f;
        in[i] = s;
        refAv(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<AvOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npIsoAvgVerts.spv", "isoAvgVerts",
             { (uint32_t)sizeof(AvIn), (uint32_t)sizeof(AvOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<AvOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        bool ok = eqFZ(ref[i].outNew.x, g[i].outNew.x) && eqFZ(ref[i].outNew.y, g[i].outNew.y)
               && eqFZ(ref[i].outNew.z, g[i].outNew.z) && eqFZ(ref[i].outNew.w, g[i].outNew.w)
               && eqFZ(ref[i].vtxNew.x, g[i].vtxNew.x) && eqFZ(ref[i].vtxNew.y, g[i].vtxNew.y)
               && eqFZ(ref[i].vtxNew.z, g[i].vtxNew.z) && eqFZ(ref[i].vtxNew.w, g[i].vtxNew.w);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u oR (%g %g %g %g) oG (%g %g %g %g)\n", i,
                            ref[i].outNew.x, ref[i].outNew.y, ref[i].outNew.z, ref[i].outNew.w,
                            g[i].outNew.x, g[i].outNew.y, g[i].outNew.z, g[i].outNew.w);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "isoAvgVerts outputs bitwise-identical (mod +-0)");
}

}  // namespace
