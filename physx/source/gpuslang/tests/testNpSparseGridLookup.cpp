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

// Verification of npSparseGridLookup.slang (sparseGridLookup): the sparse-grid
// sorted-hash binary search searchSorted + tryFindHashkey
// (sparseGridStandalone.cuh L82-104) and the thread-to-cell decomposition
// getGridCoordinates (L213-237, haloSize=0), against an exact CPU
// transcription.  Pure-integer outputs compared exactly.

#include "pxgTestFramework.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct U4 { uint32_t x, y, z, w; };
struct I4 { int32_t x, y, z, w; };
struct LkIn  { U4 keys[4]; U4 misc; U4 sizes; };
struct LkOut { U4 res; I4 coords; };

static uint32_t keyAt(const LkIn& s, uint32_t i) { const uint32_t* k = &s.keys[i >> 2].x; return k[i & 3u]; }

// sparseGridStandalone.cuh searchSorted (L82-98).
static uint32_t searchSorted(const LkIn& s, uint32_t numElements, uint32_t value)
{
    uint32_t left = 0, right = numElements;
    while ((right - left) > 1u)
    {
        const uint32_t pos = (left + right) >> 1;
        const uint32_t element = keyAt(s, pos);
        if (element <= value)
            left = pos;
        else
            right = pos;
    }
    return left;
}

static void refLk(const LkIn& s, LkOut& o)
{
    const uint32_t numElements = s.misc.x, value = s.misc.y, threadIndex = s.misc.z;

    const uint32_t result = searchSorted(s, numElements, value);
    const uint32_t found = (keyAt(s, result) == value) ? 1u : 0u;

    const int32_t sX = int32_t(s.sizes.x), sY = int32_t(s.sizes.y), sZ = int32_t(s.sizes.z);
    const uint32_t numSubgridCells = uint32_t(sX * sY * sZ);
    const uint32_t si = threadIndex / numSubgridCells;
    const int32_t localThreadIndex = int32_t(threadIndex - si * numSubgridCells);
    const int32_t xi = localThreadIndex % sX;
    const int32_t yi = (localThreadIndex / sX) % sY;
    const int32_t zi = localThreadIndex / (sX * sY);

    o.res = U4{ result, found, 0u, 0u };
    o.coords = I4{ xi, yi, zi, int32_t(si) };
}

PXG_TEST(npSparseGridLookup)
{
    std::mt19937 rng(248001);
    const uint32_t N = 6000;
    auto ru = [&](uint32_t lo, uint32_t hi) { return lo + rng() % (hi - lo + 1u); };

    std::vector<LkIn> in(N);
    std::vector<LkOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        LkIn s{};
        // strictly increasing sorted key array (16 slots, like unique subgrid hashes)
        uint32_t k = ru(0u, 500u);
        uint32_t* keys = &s.keys[0].x;
        for (uint32_t j = 0; j < 16u; ++j) { keys[j] = k; k += ru(1u, 900u); }

        const uint32_t numElements = 1u + (i % 16u);   // 1..16 (incl. the loop-free 1 case)
        uint32_t value;
        const uint32_t sel = i % 5u;
        if (sel == 0u)      value = keys[ru(0u, numElements - 1u)];       // present
        else if (sel == 1u) value = (keys[0] > 0u) ? keys[0] - 1u : 0u;   // below first -> idx 0, found only if equal
        else if (sel == 2u) value = keys[numElements - 1u] + ru(1u, 50u); // above last in-range key
        else if (sel == 3u) value = keys[numElements - 1u];               // exactly last
        else                value = ru(0u, 20000u);                        // random (mostly absent)

        // decomposition: PxU16-like sizes, threadIndex within several subgrids
        const uint32_t sX = ru(2u, 32u), sY = ru(2u, 32u), sZ = ru(2u, 32u);
        const uint32_t cells = sX * sY * sZ;
        const uint32_t threadIndex = ru(0u, 20u) * cells + ru(0u, cells - 1u);

        s.misc = U4{ numElements, value, threadIndex, 0u };
        s.sizes = U4{ sX, sY, sZ, 0u };
        in[i] = s;
        refLk(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<LkOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel kx("npSparseGridLookup.spv", "sparseGridLookup",
              { (uint32_t)sizeof(LkIn), (uint32_t)sizeof(LkOut) }, sizeof(P));
    kx.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<LkOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        bool ok = ref[i].res.x == g[i].res.x && ref[i].res.y == g[i].res.y
               && ref[i].coords.x == g[i].coords.x && ref[i].coords.y == g[i].coords.y
               && ref[i].coords.z == g[i].coords.z && ref[i].coords.w == g[i].coords.w;
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u idxR %u idxG %u fR %u fG %u cR (%d %d %d %d) cG (%d %d %d %d)\n", i,
                            ref[i].res.x, g[i].res.x, ref[i].res.y, g[i].res.y,
                            ref[i].coords.x, ref[i].coords.y, ref[i].coords.z, ref[i].coords.w,
                            g[i].coords.x, g[i].coords.y, g[i].coords.z, g[i].coords.w);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "sparseGridLookup outputs identical");
}

}  // namespace
