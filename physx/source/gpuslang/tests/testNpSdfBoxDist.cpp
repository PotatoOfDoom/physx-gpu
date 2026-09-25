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

// Verification of npSdfBoxDist.slang (sdfBoxDist): the AABB primitives getCenter
// and distancePointBoxSquared from the SDF/BVH build (SDFConstruction.cu L44-51,
// L279-283), against an exact CPU transcription.  Outputs bitwise mod +-0 (eqFZ).

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
struct BxIn  { F4 lower, upper, point; };
struct BxOut { F4 center, dist; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed

static float pmin(float a, float b) { return (a < b) ? a : b; }
static float pmax(float a, float b) { return (a > b) ? a : b; }

static void refBx(const BxIn& s, BxOut& o)
{
    const float cx = 0.5f * (s.lower.x + s.upper.x);
    const float cy = 0.5f * (s.lower.y + s.upper.y);
    const float cz = 0.5f * (s.lower.z + s.upper.z);

    const float ix = pmin(s.upper.x, s.point.x), iy = pmin(s.upper.y, s.point.y), iz = pmin(s.upper.z, s.point.z);
    const float px = pmax(s.lower.x, ix), py = pmax(s.lower.y, iy), pz = pmax(s.lower.z, iz);
    const float dx = px - s.point.x, dy = py - s.point.y, dz = pz - s.point.z;
    const float distSq = dx * dx + dy * dy + dz * dz;

    o.center = F4{ cx, cy, cz, 0.f };
    o.dist = F4{ distSq, 0.f, 0.f, 0.f };
}

PXG_TEST(npSdfBoxDist)
{
    std::mt19937 rng(247001);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<BxIn> in(N);
    std::vector<BxOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        BxIn s{};
        // build a valid box (lower <= upper per component)
        const float lx = rf(-8.f, 8.f), ly = rf(-8.f, 8.f), lz = rf(-8.f, 8.f);
        s.lower = F4{ lx, ly, lz, 0.f };
        s.upper = F4{ lx + rf(0.1f, 6.f), ly + rf(0.1f, 6.f), lz + rf(0.1f, 6.f), 0.f };
        // point: sometimes inside the box (dist 0), sometimes outside
        s.point = F4{ rf(-12.f, 12.f), rf(-12.f, 12.f), rf(-12.f, 12.f), 0.f };
        // inside stripe: point inside box -> closestPt == point -> distSq 0
        if ((i % 5u) == 2u)
            s.point = F4{ rf(lx, s.upper.x), rf(ly, s.upper.y), rf(lz, s.upper.z), 0.f };
        // on-face stripe: clamp active on exactly one axis
        if ((i % 7u) == 3u)
            s.point = F4{ s.upper.x + rf(0.1f, 3.f), rf(ly, s.upper.y), rf(lz, s.upper.z), 0.f };
        in[i] = s;
        refBx(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<BxOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npSdfBoxDist.spv", "sdfBoxDist",
             { (uint32_t)sizeof(BxIn), (uint32_t)sizeof(BxOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<BxOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        bool ok = eqFZ(ref[i].center.x, g[i].center.x) && eqFZ(ref[i].center.y, g[i].center.y)
               && eqFZ(ref[i].center.z, g[i].center.z) && eqFZ(ref[i].dist.x, g[i].dist.x);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u cR (%g %g %g) cG (%g %g %g) dR %g dG %g\n", i,
                            ref[i].center.x, ref[i].center.y, ref[i].center.z,
                            g[i].center.x, g[i].center.y, g[i].center.z,
                            ref[i].dist.x, g[i].dist.x);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "sdfBoxDist outputs bitwise-identical (mod +-0)");
}

}  // namespace
