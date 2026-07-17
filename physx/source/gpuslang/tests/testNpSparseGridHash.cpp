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

// Verification of npSparseGridHash.slang (sparseGridHash): the sparse-grid
// subgrid hash/index arithmetic (calcSubgridHash, subgridHashToId,
// subgridNeighborIndex, isSubgridInsideRange, mod, sparseGridAccess;
// sparseGridStandalone.cuh L62-139), against an exact CPU transcription.
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
struct SgIn  { I4 id, nb, modab, acc, sizes; };
struct SgOut { uint32_t hash[4]; I4 rtId, misc; uint32_t access[4]; };

static const int MAX_SPARSEGRID_DIM = 1024;
static const int MIN_SPARSEGRID_ID = -512;
static const int MAX_SPARSEGRID_ID = 511;

static uint32_t calcSubgridHash(int x, int y, int z)
{
    const int sx = x - MIN_SPARSEGRID_ID, sy = y - MIN_SPARSEGRID_ID, sz = z - MIN_SPARSEGRID_ID;
    return (uint32_t)(MAX_SPARSEGRID_DIM * MAX_SPARSEGRID_DIM * sz + MAX_SPARSEGRID_DIM * sy + sx);
}
static I4 subgridHashToId(uint32_t hashKey)
{
    const int ih = (int)hashKey;
    return I4{ ih % MAX_SPARSEGRID_DIM + MIN_SPARSEGRID_ID,
               (ih / MAX_SPARSEGRID_DIM) % MAX_SPARSEGRID_DIM + MIN_SPARSEGRID_ID,
               ih / MAX_SPARSEGRID_DIM / MAX_SPARSEGRID_DIM + MIN_SPARSEGRID_ID, 0 };
}
static int subgridNeighborIndex(int x, int y, int z) { return ((x + 1) + 3 * (y + 1) + 9 * (z + 1)); }
static bool isSubgridInsideRange(int x, int y, int z)
{
    return x >= MIN_SPARSEGRID_ID && x <= MAX_SPARSEGRID_ID &&
           y >= MIN_SPARSEGRID_ID && y <= MAX_SPARSEGRID_ID &&
           z >= MIN_SPARSEGRID_ID && z <= MAX_SPARSEGRID_ID;
}
static int imod(int a, int b) { return (a + b) % b; }
static uint32_t sparseGridAccess(int i, int j, int k, int si, int sX, int sY, int sZ)
{
    return (uint32_t)(i + sX * (j + sY * (k + sZ * si)));
}

static void refSg(const SgIn& s, SgOut& o)
{
    const uint32_t h = calcSubgridHash(s.id.x, s.id.y, s.id.z);
    o.hash[0] = h; o.hash[1] = o.hash[2] = o.hash[3] = 0u;
    o.rtId = subgridHashToId(h);
    const int ni = subgridNeighborIndex(s.nb.x, s.nb.y, s.nb.z);
    const int md = imod(s.modab.x, s.modab.y);
    const bool inside = isSubgridInsideRange(s.id.x, s.id.y, s.id.z);
    o.misc = I4{ ni, md, inside ? 1 : 0, 0 };
    o.access[0] = sparseGridAccess(s.acc.x, s.acc.y, s.acc.z, s.acc.w, s.sizes.x, s.sizes.y, s.sizes.z);
    o.access[1] = o.access[2] = o.access[3] = 0u;
}

PXG_TEST(npSparseGridHash)
{
    std::mt19937 rng(215001);
    const uint32_t N = 5000;
    auto ri = [&](int lo, int hi) { return lo + int(rng() % uint32_t(hi - lo + 1)); };

    std::vector<SgIn> in(N);
    std::vector<SgOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        SgIn s{};
        // id mostly in-range (round-trip valid); a stripe pushes out of range for isSubgridInsideRange=false
        if ((i % 9u) == 4u)
            s.id = I4{ ri(-800, 800), ri(-800, 800), ri(-800, 800), 0 };
        else
            s.id = I4{ ri(MIN_SPARSEGRID_ID, MAX_SPARSEGRID_ID), ri(MIN_SPARSEGRID_ID, MAX_SPARSEGRID_ID), ri(MIN_SPARSEGRID_ID, MAX_SPARSEGRID_ID), 0 };
        s.nb = I4{ ri(-1, 1), ri(-1, 1), ri(-1, 1), 0 };
        s.modab = I4{ ri(-100, 100), ri(1, 64), 0, 0 };   // b>0 so no div-by-zero
        s.acc = I4{ ri(0, 15), ri(0, 15), ri(0, 15), ri(0, 31) };
        s.sizes = I4{ ri(1, 16), ri(1, 16), ri(1, 16), 0 };
        in[i] = s;
        refSg(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<SgOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npSparseGridHash.spv", "sparseGridHash",
             { (uint32_t)sizeof(SgIn), (uint32_t)sizeof(SgOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<SgOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (std::memcmp(&ref[i], &g[i], sizeof(SgOut)) != 0)
        {
            if (mm < 5)
                std::printf("MM %u hash R %u G %u rtId R (%d %d %d) G (%d %d %d) misc R (%d %d %d) G (%d %d %d)\n", i,
                            ref[i].hash[0], g[i].hash[0], ref[i].rtId.x, ref[i].rtId.y, ref[i].rtId.z,
                            g[i].rtId.x, g[i].rtId.y, g[i].rtId.z, ref[i].misc.x, ref[i].misc.y, ref[i].misc.z,
                            g[i].misc.x, g[i].misc.y, g[i].misc.z);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "sparseGridHash outputs bitwise-identical (integer exact)");
}

}  // namespace
