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

// Verification of npSdfBvhMorton.slang (sdfBvhMorton): the SDF-BVH build
// front-end bvhComputeTriangleBounds (SDFConstruction.cu L150-172) and
// bvhCalculateMortonCodes (L53-82) with part1by2/morton3 (bvh.cuh L40-59) and
// getCenter (L44-51), against an exact CPU transcription.  Bounds bitwise mod
// +-0 (eqFZ); Morton keys exact-bits.

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
struct BmIn  { F4 misc, va, vb, vc, vd; };
struct BmOut { F4 lower, upper, aux; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed

static float selMin(float a, float b) { return a < b ? a : b; }
static float selMax(float a, float b) { return a > b ? a : b; }
static int32_t iselMin(int32_t a, int32_t b) { return a < b ? a : b; }
static int32_t iselMax(int32_t a, int32_t b) { return a < b ? b : a; }

static uint32_t part1by2(uint32_t n)
{
    n = (n ^ (n << 16)) & 0xff0000ffu;
    n = (n ^ (n << 8)) & 0x0300f00fu;
    n = (n ^ (n << 4)) & 0x030c30c3u;
    n = (n ^ (n << 2)) & 0x09249249u;
    return n;
}

static uint32_t morton3(float x, float y, float z, int32_t dim)
{
    const uint32_t ux = uint32_t(iselMin(dim - 1, iselMax(0, int32_t(x * float(dim)))));
    const uint32_t uy = uint32_t(iselMin(dim - 1, iselMax(0, int32_t(y * float(dim)))));
    const uint32_t uz = uint32_t(iselMin(dim - 1, iselMax(0, int32_t(z * float(dim)))));
    return (part1by2(uz) << 2) | (part1by2(uy) << 1) | part1by2(ux);
}

static void refBm(const BmIn& s, BmOut& o)
{
    std::memset(&o, 0, sizeof(o));
    const uint32_t mode = f2u(s.misc.y);

    if (mode == 0u)
    {
        const float margin = s.misc.x;
        float lox = s.va.x, loy = s.va.y, loz = s.va.z;
        float hix = s.va.x, hiy = s.va.y, hiz = s.va.z;
        lox = selMin(lox, s.vb.x); loy = selMin(loy, s.vb.y); loz = selMin(loz, s.vb.z);
        hix = selMax(hix, s.vb.x); hiy = selMax(hiy, s.vb.y); hiz = selMax(hiz, s.vb.z);
        lox = selMin(lox, s.vc.x); loy = selMin(loy, s.vc.y); loz = selMin(loz, s.vc.z);
        hix = selMax(hix, s.vc.x); hiy = selMax(hiy, s.vc.y); hiz = selMax(hiz, s.vc.z);
        lox -= margin; loy -= margin; loz -= margin;
        hix += margin; hiy += margin; hiz += margin;
        o.lower = F4{ lox, loy, loz, 0.f };
        o.upper = F4{ hix, hiy, hiz, 0.f };
    }
    else
    {
        const int32_t priority = (int32_t)f2u(s.misc.z);
        const bool hasPriority = f2u(s.misc.w) == 1u;

        const float cx = 0.5f * (s.va.x + s.vb.x);
        const float cy = 0.5f * (s.va.y + s.vb.y);
        const float cz = 0.5f * (s.va.z + s.vb.z);
        const float lx = (cx - s.vc.x) * s.vd.x;
        const float ly = (cy - s.vc.y) * s.vd.y;
        const float lz = (cz - s.vc.z) * s.vd.z;

        int32_t key;
        if (hasPriority)
        {
            key = (int32_t)morton3(lx, ly, lz, 512);
            key |= (~priority) << 27;
        }
        else
        {
            key = (int32_t)morton3(lx, ly, lz, 1024);
        }
        o.aux = F4{ u2f((uint32_t)key), 0.f, 0.f, 0.f };
    }
}

PXG_TEST(npSdfBvhMorton)
{
    std::mt19937 rng(260001);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<BmIn> in(N);
    std::vector<BmOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        BmIn s{};
        const uint32_t mode = i % 2u;
        if (mode == 0u)
        {
            s.misc = F4{ rf(0.f, 0.2f), u2f(0u), 0.f, 0.f };
            s.va = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
            s.vb = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
            s.vc = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
            // zero-margin stripe
            if ((i % 7u) == 2u) s.misc.x = 0.f;
            // degenerate triangle: all three vertices identical
            if ((i % 13u) == 6u) { s.vb = s.va; s.vc = s.va; }
        }
        else
        {
            const uint32_t hasPriority = (i / 2u) % 2u;
            const uint32_t priority = rng() % 32u;   // 5-bit priorities
            s.misc = F4{ 0.f, u2f(1u), u2f(priority), u2f(hasPriority) };
            // item bounds around a center within/near the grid
            const float cx = rf(-1.f, 3.f), cy = rf(-1.f, 3.f), cz = rf(-1.f, 3.f);
            const float ex = rf(0.f, 0.3f), ey = rf(0.f, 0.3f), ez = rf(0.f, 0.3f);
            s.va = F4{ cx - ex, cy - ey, cz - ez, 0.f };
            s.vb = F4{ cx + ex, cy + ey, cz + ez, 0.f };
            s.vc = F4{ rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), 0.f };  // gridLower
            s.vd = F4{ rf(0.2f, 1.f), rf(0.2f, 1.f), rf(0.2f, 1.f), 0.f };        // gridInvEdges
            // out-of-grid stripe: local coords negative / > 1 exercise BOTH
            // clamp arms of morton3
            if ((i % 11u) == 4u) s.vc = F4{ 5.f, 5.f, 5.f, 0.f };
            if ((i % 11u) == 8u) s.vc = F4{ -9.f, -9.f, -9.f, 0.f };
            // exact grid-corner stripe: local == 0 and local == 1 boundaries
            if ((i % 19u) == 5u)
            {
                s.va = F4{ 1.f, 1.f, 1.f, 0.f }; s.vb = F4{ 1.f, 1.f, 1.f, 0.f };
                s.vc = F4{ ((i / 38u) % 2u) ? 1.f : 0.f, 0.f, 1.f, 0.f };
                s.vd = F4{ 1.f, 1.f, 1.f, 0.f };
            }
            // priority extremes: 0 and 31
            if ((i % 23u) == 7u) s.misc.z = u2f(((i / 46u) % 2u) ? 31u : 0u);
        }
        in[i] = s;
        refBm(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<BmOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npSdfBvhMorton.spv", "sdfBvhMorton",
             { (uint32_t)sizeof(BmIn), (uint32_t)sizeof(BmOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<BmOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        bool ok = eqFZ(ref[i].lower.x, g[i].lower.x) && eqFZ(ref[i].lower.y, g[i].lower.y) && eqFZ(ref[i].lower.z, g[i].lower.z)
               && eqFZ(ref[i].upper.x, g[i].upper.x) && eqFZ(ref[i].upper.y, g[i].upper.y) && eqFZ(ref[i].upper.z, g[i].upper.z)
               && f2u(ref[i].aux.x) == f2u(g[i].aux.x);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u loR (%g %g %g) loG (%g %g %g) keyR %08x keyG %08x\n", i,
                            ref[i].lower.x, ref[i].lower.y, ref[i].lower.z,
                            g[i].lower.x, g[i].lower.y, g[i].lower.z,
                            f2u(ref[i].aux.x), f2u(g[i].aux.x));
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "sdfBvhMorton outputs bitwise-identical (keys exact)");
}

}  // namespace
