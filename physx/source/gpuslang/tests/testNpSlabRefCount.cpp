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

// Verification of npSlabRefCount.slang (casCount): countActiveSlabs
// (solverBlock.cuh L84-97), the mass-splitting slab reference-count popcount,
// against an exact CPU transcription.  Pure unsigned-integer arithmetic —
// outputs compared exactly (no float tolerance).

#include "pxgTestFramework.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct U4 { uint32_t x, y, z, w; };

static uint32_t rpopc(uint32_t v)  // portable population count
{
    uint32_t c = 0;
    while (v) { v &= (v - 1u); ++c; }
    return c;
}

// exact transcription of countActiveSlabs (solverBlock.cuh L84-97)
static uint32_t refCountActiveSlabs(uint32_t index, uint32_t numSlabs, uint32_t numSolverBodies,
                                    const std::vector<uint32_t>& words)
{
    uint32_t referenceCount = 0;
    const uint32_t num32Slabs = (numSlabs + 31u) / 32u;
    for (uint32_t i = 0; i < num32Slabs; ++i)
    {
        const uint32_t id = words[index + i * numSolverBodies];
        referenceCount += rpopc(id);
    }
    return referenceCount > 1u ? referenceCount : 1u;  // PxMax(1u, referenceCount)
}

PXG_TEST(npSlabRefCount)
{
    std::mt19937 rng(190001);
    const uint32_t N = 4000;
    const uint32_t WORDS = 16384;

    // Shared encoded reference-count buffer; a mix of dense/sparse/zero/full
    // bit patterns so popcount spans 0..32 per word.
    std::vector<uint32_t> words(WORDS);
    for (uint32_t i = 0; i < WORDS; ++i)
    {
        const uint32_t mode = rng() % 5;
        if (mode == 0) words[i] = 0u;               // empty slab word
        else if (mode == 1) words[i] = 0xFFFFFFFFu; // fully-referenced word
        else if (mode == 2) words[i] = 1u << (rng() % 32); // single bit
        else words[i] = rng();                      // arbitrary
    }

    std::vector<U4> in(N);
    std::vector<U4> ref(N);
    size_t cMulti = 0, cSingle = 0, cFloor = 0, cAllZero = 0, cMax32 = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        // numSlabs 1..160 -> num32Slabs 1..5 (multi-word striding exercised)
        const uint32_t numSlabs = 1u + rng() % 160u;
        const uint32_t num32Slabs = (numSlabs + 31u) / 32u;
        // numSolverBodies bounded so the strided reads stay in [0, WORDS)
        const uint32_t numSolverBodies = 1u + rng() % 2000u;
        const uint32_t span = (num32Slabs - 1u) * numSolverBodies;
        const uint32_t index = (WORDS > span) ? (rng() % (WORDS - span)) : 0u;

        in[i] = U4{ index, numSlabs, numSolverBodies, 0u };
        const uint32_t c = refCountActiveSlabs(index, numSlabs, numSolverBodies, words);
        ref[i] = U4{ c, 0u, 0u, 0u };

        if (num32Slabs > 1) ++cMulti; else ++cSingle;
        // sum-before-floor == 0 means the PxMax(1,.) floor is load-bearing
        uint32_t raw = 0;
        for (uint32_t k = 0; k < num32Slabs; ++k) raw += rpopc(words[index + k * numSolverBodies]);
        if (raw == 0) ++cFloor;
        if (raw == 0) ++cAllZero;
        if (c >= 32) ++cMax32;
    }

    std::printf("[ info ] slabref cov: multi %zu single %zu floor %zu allZero %zu max32 %zu\n",
                cMulti, cSingle, cFloor, cAllZero, cMax32);
    PXG_CHECK(cMulti > 1000);
    PXG_CHECK(cSingle > 500);
    PXG_CHECK(cFloor > 10);       // the PxMax(1u,.) floor actually fires
    PXG_CHECK(cMax32 > 300);      // large reference counts across multiple words

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<U4> outInit(N);
    PxgBufferHandle bWords = createBufferFrom(dev, words);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npSlabRefCount.spv", "casCount",
             { (uint32_t)sizeof(uint32_t), (uint32_t)sizeof(U4), (uint32_t)sizeof(U4) }, sizeof(P));
    k.dispatch(GROUPS, { bWords, bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<U4>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (std::memcmp(&ref[i], &g[i], sizeof(U4)) != 0)
        {
            if (mm < 5)
                std::printf("MM %u refC %u gpuC %u (index %u numSlabs %u nsb %u)\n", i, ref[i].x,
                            g[i].x, in[i].x, in[i].y, in[i].z);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "casCount outputs exactly identical");
}

}  // namespace
