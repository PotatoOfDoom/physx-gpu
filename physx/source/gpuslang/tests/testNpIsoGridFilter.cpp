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

// Verification of npIsoGridFilter.slang (isoGridFilter): the isosurface
// dense-grid Gauss and dilate/erode filters (isosurfaceExtraction.cu
// L1097-1125 / L1138-1153) over a self-contained <=64-cell grid, against an
// exact CPU transcription.  Outputs bitwise mod +-0 (eqFZ).

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
struct U4 { uint32_t x, y, z, w; };
struct GfIn  { F4 vals[16]; U4 dims; F4 prm; };
struct GfOut { F4 r; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b) || (a == 0.f && b == 0.f); }

static float pmax(float a, float b) { return (a > b) ? a : b; }

static float valAt(const GfIn& s, int32_t i) { const float* v = &s.vals[uint32_t(i) >> 2].x; return v[uint32_t(i) & 3u]; }
static int32_t getCellNr(int32_t nx, int32_t ny, int32_t xi, int32_t yi, int32_t zi) { return (zi * ny + yi) * nx + xi; }

static float getGridValueSafe(const GfIn& s, int32_t nx, int32_t ny, int32_t nz,
                              int32_t x, int32_t y, int32_t z, int32_t ox, int32_t oy, int32_t oz)
{
    if (x + ox < 0 || y + oy < 0 || z + oz < 0 || x + ox >= nx || y + oy >= ny || z + oz >= nz)
        return 0.0f;
    return valAt(s, getCellNr(nx, ny, x + ox, y + oy, z + oz));
}

static void refGf(const GfIn& s, GfOut& o)
{
    const int32_t nx = int32_t(s.dims.x), ny = int32_t(s.dims.y), nz = int32_t(s.dims.z);
    const int32_t threadIndex = int32_t(s.dims.w);
    const float neighborWeight = s.prm.x;
    const float sign = s.prm.y;

    int32_t t = threadIndex;
    const int32_t cx = t % nx; t /= nx;
    const int32_t cy = t % ny;
    const int32_t cz = t / ny;

    float w[3];
    w[0] = neighborWeight;
    w[1] = 1.0f;
    w[2] = w[0];

    float weightSum = 0.0f, m = 0.0f;
    for (int32_t x = -1; x <= 1; ++x)
        for (int32_t y = -1; y <= 1; ++y)
            for (int32_t z = -1; z <= 1; ++z)
            {
                const float weight = w[x + 1] * w[y + 1] * w[z + 1];
                weightSum += weight;
                m += weight * getGridValueSafe(s, nx, ny, nz, cx, cy, cz, x, y, z);
            }
    const float gauss = m / weightSum;

    float md = sign * valAt(s, threadIndex);
    for (int32_t x = -1; x <= 1; ++x)
        for (int32_t y = -1; y <= 1; ++y)
            for (int32_t z = -1; z <= 1; ++z)
                md = pmax(md, sign * getGridValueSafe(s, nx, ny, nz, cx, cy, cz, x, y, z));
    const float dilateErode = sign * md;

    o.r = F4{ gauss, dilateErode, 0.f, 0.f };
}

PXG_TEST(npIsoGridFilter)
{
    std::mt19937 rng(250001);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto ru = [&](uint32_t lo, uint32_t hi) { return lo + rng() % (hi - lo + 1u); };

    std::vector<GfIn> in(N);
    std::vector<GfOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        GfIn s{};
        const uint32_t nx = ru(2u, 4u), ny = ru(2u, 4u), nz = ru(2u, 4u);
        const uint32_t cells = nx * ny * nz;
        float* v = &s.vals[0].x;
        for (uint32_t c = 0; c < cells; ++c) v[c] = rf(-5.f, 5.f);

        uint32_t threadIndex = ru(0u, cells - 1u);
        // corner stripe: many out-of-bounds taps (value 0, weight still counted)
        if ((i % 5u) == 1u) threadIndex = 0u;
        // interior stripe: 4x4x4 grid, center cell -> no OOB taps at all
        if ((i % 5u) == 3u)
        {
            s.dims = U4{ 4u, 4u, 4u, 0u };
            for (uint32_t c = 0; c < 64u; ++c) v[c] = rf(-5.f, 5.f);
            threadIndex = uint32_t(getCellNr(4, 4, 1 + int32_t(ru(0u, 1u)), 1 + int32_t(ru(0u, 1u)), 1 + int32_t(ru(0u, 1u))));
            s.dims.w = threadIndex;
            s.prm = F4{ rf(0.05f, 1.f), (i % 2u) ? 1.f : -1.f, 0.f, 0.f };
            in[i] = s;
            refGf(s, ref[i]);
            continue;
        }
        s.dims = U4{ nx, ny, nz, threadIndex };
        s.prm = F4{ rf(0.05f, 1.f), (i % 2u) ? 1.f : -1.f, 0.f, 0.f };   // neighborWeight, sign (dilate/erode both)
        in[i] = s;
        refGf(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<GfOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npIsoGridFilter.spv", "isoGridFilter",
             { (uint32_t)sizeof(GfIn), (uint32_t)sizeof(GfOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<GfOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        bool ok = eqFZ(ref[i].r.x, g[i].r.x) && eqFZ(ref[i].r.y, g[i].r.y);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u gR %g gG %g dR %g dG %g\n", i,
                            ref[i].r.x, g[i].r.x, ref[i].r.y, g[i].r.y);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "isoGridFilter outputs bitwise-identical (mod +-0)");
}

}  // namespace
