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

// Verification of npGridIndex.slang (gridIndex): the dense-grid cell-index
// arithmetic getCellNr / getCellCoords / getCellIndexSafe
// (denseGridStandalone.cuh L38-83), against an exact CPU transcription.
// Pure integer arithmetic -> exact int equality.

#include "pxgTestFramework.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct I4 { int32_t x, y, z, w; };
struct GiIn  { I4 grid, idx, off; };
struct GiOut { I4 cellNr, coords; uint32_t safe[4]; };

static const uint32_t EMPTY_SUBGRID = 0xffffffffu;

static int32_t getCellNr(int32_t nx, int32_t ny, int32_t xi, int32_t yi, int32_t zi)
{
    return (zi * ny + yi) * nx + xi;
}
static I4 getCellCoords(int32_t nx, int32_t ny, int32_t cellNr)
{
    I4 r;
    r.x = cellNr % nx;
    cellNr /= nx;
    r.y = cellNr % ny;
    r.z = cellNr / ny;
    r.w = -1;
    return r;
}
static uint32_t getCellIndexSafe(int32_t nx, int32_t ny, int32_t nz, int32_t xi, int32_t yi, int32_t zi,
                                 int32_t ox, int32_t oy, int32_t oz)
{
    if (xi + ox < 0 || yi + oy < 0 || zi + oz < 0 ||
        xi + ox >= nx || yi + oy >= ny || zi + oz >= nz)
        return EMPTY_SUBGRID;
    return (uint32_t)getCellNr(nx, ny, xi + ox, yi + oy, zi + oz);
}

static void refGi(const GiIn& s, GiOut& o)
{
    const int32_t nx = s.grid.x, ny = s.grid.y, nz = s.grid.z;
    const int32_t xi = s.idx.x, yi = s.idx.y, zi = s.idx.z;
    const int32_t cellNr = getCellNr(nx, ny, xi, yi, zi);
    o.cellNr = I4{ cellNr, 0, 0, 0 };
    o.coords = getCellCoords(nx, ny, cellNr);
    o.safe[0] = getCellIndexSafe(nx, ny, nz, xi, yi, zi, s.off.x, s.off.y, s.off.z);
    o.safe[1] = o.safe[2] = o.safe[3] = 0u;
}

PXG_TEST(npGridIndex)
{
    std::mt19937 rng(211001);
    const uint32_t N = 4000;
    auto ri = [&](int lo, int hi) { return lo + int(rng() % uint32_t(hi - lo + 1)); };

    std::vector<GiIn> in(N);
    std::vector<GiOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        GiIn s{};
        const int32_t nx = ri(1, 20), ny = ri(1, 20), nz = ri(1, 20);
        s.grid = I4{ nx, ny, nz, 0 };
        s.idx = I4{ ri(0, nx - 1), ri(0, ny - 1), ri(0, nz - 1), 0 };
        // offsets in [-2,2] -> exercises both in-range and out-of-range safe branches
        s.off = I4{ ri(-2, 2), ri(-2, 2), ri(-2, 2), 0 };
        in[i] = s;
        refGi(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<GiOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npGridIndex.spv", "gridIndex",
             { (uint32_t)sizeof(GiIn), (uint32_t)sizeof(GiOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<GiOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (std::memcmp(&ref[i], &g[i], sizeof(GiOut)) != 0)
        {
            if (mm < 5)
                std::printf("MM %u cellNr R %d G %d coords R (%d %d %d) G (%d %d %d) safe R %u G %u\n", i,
                            ref[i].cellNr.x, g[i].cellNr.x, ref[i].coords.x, ref[i].coords.y, ref[i].coords.z,
                            g[i].coords.x, g[i].coords.y, g[i].coords.z, ref[i].safe[0], g[i].safe[0]);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "gridIndex outputs bitwise-identical (integer exact)");
}

}  // namespace
