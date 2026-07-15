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

// Verification of npArtiBitfield.slang (bitScan): the 64-bit bitfield scan
// helpers articulationLowestSetBit / articulationHighestSetBit
// (articulationImpulseResponse.cuh L47-57), against an exact CPU transcription.
// Pure integer; outputs compared exactly.

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

// lowestSetBit(val) = val==0 ? 0 : __ffsll(val)-1  (= index of lowest set bit)
static uint32_t refLowest(uint64_t val)
{
    return val == 0 ? 0u : static_cast<uint32_t>(__builtin_ctzll(val));
}
// highestSetBit(val) = 63 - __clzll(val)  (val==0 -> firstbithigh(0)=0xFFFFFFFF)
static uint32_t refHighest(uint64_t val)
{
    return val == 0 ? 0xFFFFFFFFu : static_cast<uint32_t>(63 - __builtin_clzll(val));
}

PXG_TEST(npArtiBitfield)
{
    std::mt19937_64 rng(195001);
    const uint32_t N = 4000;

    std::vector<U4> in(N);
    std::vector<U4> ref(N);
    size_t cLoOnly = 0, cHiOnly = 0, cSingle = 0, cZero = 0, cBoundary = 0, cFull = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        uint64_t val;
        const uint32_t mode = static_cast<uint32_t>(rng() % 8);
        if (mode == 0) { val = 0; ++cZero; }                          // val==0 (lowest guard)
        else if (mode == 1) { val = uint64_t(1) << (rng() % 64); ++cSingle; }  // single bit
        else if (mode == 2) { val = 0xFFFFFFFFFFFFFFFFull; ++cFull; }  // all ones
        else if (mode == 3) { val = uint64_t(rng() & 0xFFFFFFFFu); if (val) ++cLoOnly; else val = 1; } // lo only
        else if (mode == 4) { val = uint64_t(rng()) << 32; if (val) ++cHiOnly; else val = uint64_t(1) << 32; } // hi only
        else if (mode == 5) { val = (uint64_t(1) << 31) | (uint64_t(1) << 32); ++cBoundary; } // 31/32 boundary
        else val = rng();                                             // arbitrary 64-bit
        if (val == 0 && mode != 0) val = rng() | 1u;

        in[i] = U4{ static_cast<uint32_t>(val & 0xFFFFFFFFu),
                    static_cast<uint32_t>(val >> 32), 0u, 0u };
        ref[i] = U4{ refLowest(val), refHighest(val), 0u, 0u };
    }

    std::printf("[ info ] bitfield cov: zero %zu single %zu full %zu loOnly %zu hiOnly %zu boundary %zu\n",
                cZero, cSingle, cFull, cLoOnly, cHiOnly, cBoundary);
    PXG_CHECK(cZero > 300);
    PXG_CHECK(cSingle > 300);
    PXG_CHECK(cLoOnly > 200);
    PXG_CHECK(cHiOnly > 200);
    PXG_CHECK(cBoundary > 300);

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<U4> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npArtiBitfield.spv", "bitScan",
             { (uint32_t)sizeof(U4), (uint32_t)sizeof(U4) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<U4>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        // For val==0 the CUDA highest is UB; both sides emit 0xFFFFFFFF, but
        // only the lowest lane is contractually meaningful there.  Compare both
        // (they agree) — the memcmp covers lowest and highest.
        if (std::memcmp(&ref[i], &g[i], sizeof(U4)) != 0)
        {
            if (mm < 5)
                std::printf("MM %u lo %08x hi %08x refL %u refH %u gL %u gH %u\n", i, in[i].x,
                            in[i].y, ref[i].x, ref[i].y, g[i].x, g[i].y);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "bitScan outputs exactly identical");
}

}  // namespace
