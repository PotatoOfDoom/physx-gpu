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

// Verification of npDenseGridCell.slang (denseGridCell): the dense-grid
// getCellIndexFromParticleAndTransformToLocalCoordinates
// (denseGridStandalone.cuh L123-131), against an exact CPU transcription.  The
// local coordinates are bitwise mod +-0; the integer cell index is exact.

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
struct I4 { int32_t x, y, z, w; };
struct DcIn  { F4 p, origin, params; };
struct DcOut { F4 pLocal; I4 cell; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed

static void refCell(const DcIn& s, DcOut& o)
{
    float px = s.p.x - s.origin.x;
    float py = s.p.y - s.origin.y;
    float pz = s.p.z - s.origin.z;
    const float invDx = 1.0f / s.params.x;
    const int32_t cxi = (int32_t)std::floor(px * invDx);
    const int32_t cyi = (int32_t)std::floor(py * invDx);
    const int32_t czi = (int32_t)std::floor(pz * invDx);
    o.pLocal = F4{ px, py, pz, 0.f };
    o.cell = I4{ cxi, cyi, czi, -1 };
}

PXG_TEST(npDenseGridCell)
{
    std::mt19937 rng(228001);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<DcIn> in(N);
    std::vector<DcOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        DcIn s{};
        s.p = F4{ rf(-50.f, 50.f), rf(-50.f, 50.f), rf(-50.f, 50.f), 0.f };
        s.origin = F4{ rf(-10.f, 10.f), rf(-10.f, 10.f), rf(-10.f, 10.f), 0.f };
        s.params = F4{ rf(0.05f, 2.f), 0.f, 0.f, 0.f };   // gridSpacing
        // p == origin stripe: local coords 0 -> cell 0
        if ((i % 9u) == 3u) s.origin = s.p;
        in[i] = s;
        refCell(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<DcOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npDenseGridCell.spv", "denseGridCell",
             { (uint32_t)sizeof(DcIn), (uint32_t)sizeof(DcOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<DcOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        bool ok = eqFZ(ref[i].pLocal.x, g[i].pLocal.x) && eqFZ(ref[i].pLocal.y, g[i].pLocal.y)
               && eqFZ(ref[i].pLocal.z, g[i].pLocal.z)
               && ref[i].cell.x == g[i].cell.x && ref[i].cell.y == g[i].cell.y
               && ref[i].cell.z == g[i].cell.z && ref[i].cell.w == g[i].cell.w;
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u plR (%g %g %g) cR (%d %d %d %d) plG (%g %g %g) cG (%d %d %d %d)\n", i,
                            ref[i].pLocal.x, ref[i].pLocal.y, ref[i].pLocal.z,
                            ref[i].cell.x, ref[i].cell.y, ref[i].cell.z, ref[i].cell.w,
                            g[i].pLocal.x, g[i].pLocal.y, g[i].pLocal.z,
                            g[i].cell.x, g[i].cell.y, g[i].cell.z, g[i].cell.w);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "denseGridCell outputs bitwise-identical (mod +-0)");
}

}  // namespace
