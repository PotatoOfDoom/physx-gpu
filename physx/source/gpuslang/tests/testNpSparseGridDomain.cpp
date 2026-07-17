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

// Verification of npSparseGridDomain.slang (sparseGridDomain): the sparse-grid
// getSubgridDomainSize + calcSubgridId (sparseGridStandalone.cuh L51-59),
// against an exact CPU transcription.  domainSize is bitwise mod +-0; the
// integer subgrid id is exact.

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
struct GdIn  { F4 params, sub, pos; };
struct GdOut { F4 domainSize; I4 subgridId; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b) || (a == 0.f && b == 0.f); }

static void refDomain(const GdIn& s, GdOut& o)
{
    const float dx = s.params.x;
    const uint32_t haloSize = f2u(s.params.y);
    const uint32_t subX = f2u(s.sub.x), subY = f2u(s.sub.y), subZ = f2u(s.sub.z);
    const float px = s.pos.x, py = s.pos.y, pz = s.pos.z;

    const float dsx = dx * float(subX - 2u * haloSize);
    const float dsy = dx * float(subY - 2u * haloSize);
    const float dsz = dx * float(subZ - 2u * haloSize);

    const int32_t idx = (int32_t)std::floor(px / dsx);
    const int32_t idy = (int32_t)std::floor(py / dsy);
    const int32_t idz = (int32_t)std::floor(pz / dsz);

    o.domainSize = F4{ dsx, dsy, dsz, 0.f };
    o.subgridId = I4{ idx, idy, idz, 0 };
}

PXG_TEST(npSparseGridDomain)
{
    std::mt19937 rng(227001);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<GdIn> in(N);
    std::vector<GdOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        GdIn s{};
        const float dx = rf(0.01f, 0.5f);
        const uint32_t halo = rng() % 4u;                   // 0..3
        const uint32_t subX = 8u + rng() % 57u;             // 8..64, > 2*halo
        const uint32_t subY = 8u + rng() % 57u;
        const uint32_t subZ = 8u + rng() % 57u;
        s.params = F4{ dx, u2f(halo), 0.f, 0.f };
        s.sub = F4{ u2f(subX), u2f(subY), u2f(subZ), 0.f };
        // pos spans positive and negative so floor yields both signs of subgrid id
        s.pos = F4{ rf(-100.f, 100.f), rf(-100.f, 100.f), rf(-100.f, 100.f), 0.f };
        // origin stripe: pos exactly 0 -> id 0
        if ((i % 9u) == 3u) s.pos = F4{ 0.f, 0.f, 0.f, 0.f };
        in[i] = s;
        refDomain(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<GdOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npSparseGridDomain.spv", "sparseGridDomain",
             { (uint32_t)sizeof(GdIn), (uint32_t)sizeof(GdOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<GdOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        bool ok = eqFZ(ref[i].domainSize.x, g[i].domainSize.x) && eqFZ(ref[i].domainSize.y, g[i].domainSize.y)
               && eqFZ(ref[i].domainSize.z, g[i].domainSize.z)
               && ref[i].subgridId.x == g[i].subgridId.x && ref[i].subgridId.y == g[i].subgridId.y
               && ref[i].subgridId.z == g[i].subgridId.z;
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u dsR (%g %g %g) idR (%d %d %d) dsG (%g %g %g) idG (%d %d %d)\n", i,
                            ref[i].domainSize.x, ref[i].domainSize.y, ref[i].domainSize.z,
                            ref[i].subgridId.x, ref[i].subgridId.y, ref[i].subgridId.z,
                            g[i].domainSize.x, g[i].domainSize.y, g[i].domainSize.z,
                            g[i].subgridId.x, g[i].subgridId.y, g[i].subgridId.z);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "sparseGridDomain outputs bitwise-identical (mod +-0)");
}

}  // namespace
