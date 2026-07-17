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

// Verification of npGridLocation.slang (gridLocation): the two grid cell->world
// location functions -- denseGridStandalone.cuh getLocation (L118-121) and
// sparseGridStandalone.cuh getLocationFromHashkey (L202-211, haloSize=0) --
// against an exact CPU transcription.  Outputs bitwise mod +-0 (eqFZ).

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
struct I4 { int32_t x, y, z, w; };
struct GlIn  { F4 origin; F4 params; U4 sizes; I4 index; };
struct GlOut { F4 denseLoc; F4 sparseLoc; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b) || (a == 0.f && b == 0.f); }

static const int MAX_SPARSEGRID_DIM = 1024;
static const int MIN_SPARSEGRID_ID = -512;

// sparseGridStandalone.cuh subgridHashToId (L114-121).
struct I3 { int x, y, z; };
static I3 subgridHashToId(uint32_t hashKey)
{
    const int ihashKey = int(hashKey);
    return I3{ ihashKey % MAX_SPARSEGRID_DIM + MIN_SPARSEGRID_ID,
               (ihashKey / MAX_SPARSEGRID_DIM) % MAX_SPARSEGRID_DIM + MIN_SPARSEGRID_ID,
               ihashKey / MAX_SPARSEGRID_DIM / MAX_SPARSEGRID_DIM + MIN_SPARSEGRID_ID };
}

// sparseGridStandalone.cuh calcSubgridHash (L62-66) -- used to build valid hashes.
static uint32_t calcSubgridHash(int idx, int idy, int idz)
{
    const int sx = idx - MIN_SPARSEGRID_ID, sy = idy - MIN_SPARSEGRID_ID, sz = idz - MIN_SPARSEGRID_ID;
    return uint32_t(MAX_SPARSEGRID_DIM * MAX_SPARSEGRID_DIM * sz + MAX_SPARSEGRID_DIM * sy + sx);
}

static void refGl(const GlIn& s, GlOut& o)
{
    const float ix = float(s.index.x), iy = float(s.index.y), iz = float(s.index.z);

    // dense getLocation: origin + PxVec3(index) * gridSpacing
    const float ddx = s.params.x;
    o.denseLoc = F4{ s.origin.x + ix * ddx, s.origin.y + iy * ddx, s.origin.z + iz * ddx, 0.f };

    // sparse getLocationFromHashkey (haloSize = 0)
    const float sdx = s.params.y;
    const I3 sid = subgridHashToId(f2u(s.params.z));
    const int szx = int(s.sizes.x), szy = int(s.sizes.y), szz = int(s.sizes.z);
    const float ox = float(sid.x) * sdx * float(szx);
    const float oy = float(sid.y) * sdx * float(szy);
    const float oz = float(sid.z) * sdx * float(szz);
    o.sparseLoc = F4{ ox + ix * sdx, oy + iy * sdx, oz + iz * sdx, 0.f };
}

PXG_TEST(npGridLocation)
{
    std::mt19937 rng(236001);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto ri = [&](int lo, int hi) { return lo + int(rng() % uint32_t(hi - lo + 1)); };

    std::vector<GlIn> in(N);
    std::vector<GlOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        GlIn s{};
        s.origin = F4{ rf(-20.f, 20.f), rf(-20.f, 20.f), rf(-20.f, 20.f), 0.f };
        // subgridId in a range that round-trips through the hash (sid + 512 in [0,1024))
        const int idx = ri(-200, 200), idy = ri(-200, 200), idz = ri(-200, 200);
        const uint32_t hash = calcSubgridHash(idx, idy, idz);
        s.params = F4{ rf(0.01f, 2.f), rf(0.01f, 2.f), u2f(hash), 0.f };  // denseSpacing, sparseSpacing, hash
        const uint32_t szx = uint32_t(ri(2, 64)), szy = uint32_t(ri(2, 64)), szz = uint32_t(ri(2, 64));
        s.sizes = U4{ szx, szy, szz, 0u };
        s.index = I4{ ri(-50, 50), ri(-50, 50), ri(-50, 50), 0 };
        // origin stripe: index == 0 -> dense location == origin, sparse == subgridOrigin
        if ((i % 13u) == 3u) s.index = I4{ 0, 0, 0, 0 };
        // zero-subgrid stripe: hash for sid == 0 -> subgridOrigin == 0
        if ((i % 17u) == 5u) s.params.z = u2f(calcSubgridHash(0, 0, 0));
        in[i] = s;
        refGl(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<GlOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npGridLocation.spv", "gridLocation",
             { (uint32_t)sizeof(GlIn), (uint32_t)sizeof(GlOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<GlOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        bool ok = eqFZ(ref[i].denseLoc.x, g[i].denseLoc.x) && eqFZ(ref[i].denseLoc.y, g[i].denseLoc.y)
               && eqFZ(ref[i].denseLoc.z, g[i].denseLoc.z)
               && eqFZ(ref[i].sparseLoc.x, g[i].sparseLoc.x) && eqFZ(ref[i].sparseLoc.y, g[i].sparseLoc.y)
               && eqFZ(ref[i].sparseLoc.z, g[i].sparseLoc.z);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u dR (%g %g %g) dG (%g %g %g) sR (%g %g %g) sG (%g %g %g)\n", i,
                            ref[i].denseLoc.x, ref[i].denseLoc.y, ref[i].denseLoc.z,
                            g[i].denseLoc.x, g[i].denseLoc.y, g[i].denseLoc.z,
                            ref[i].sparseLoc.x, ref[i].sparseLoc.y, ref[i].sparseLoc.z,
                            g[i].sparseLoc.x, g[i].sparseLoc.y, g[i].sparseLoc.z);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "gridLocation outputs bitwise-identical (mod +-0)");
}

}  // namespace
