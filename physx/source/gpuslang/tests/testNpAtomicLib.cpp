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

// RUNTIME_PLAN.md P2.2 battery: CAS-loop float atomics (atomic.cuh parity)
// and 64-bit bit conventions.  64 threads contend per float cell and 128
// per 64-bit OR cell; all designs are order-invariant (min/max, exact
// integer-valued float sums, ORs), so every final cell value is bitwise
// deterministic and compared exactly.

#include "pxgTestFramework.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

static uint32_t popcll(uint32_t lo, uint32_t hi)
{
    return uint32_t(__builtin_popcount(lo) + __builtin_popcount(hi));
}
static uint32_t ffsll64(uint32_t lo, uint32_t hi)
{
    if (lo) return uint32_t(__builtin_ctz(lo)) + 1u;
    if (hi) return uint32_t(__builtin_ctz(hi)) + 33u;
    return 0u;
}
static uint32_t clzll64(uint32_t lo, uint32_t hi)
{
    if (hi) return uint32_t(__builtin_clz(hi));
    if (lo) return 32u + uint32_t(__builtin_clz(lo));
    return 64u;
}

PXG_TEST(npAtomicLib)
{
    const uint32_t NT = 4096, C = 64, C64 = 32;
    std::mt19937 rng(7777);

    std::vector<uint32_t> in(NT);
    for (uint32_t t = 0; t < NT; ++t)
    {
        // Finite, both signs; no NaN/inf, no -0.  Deliberate +0 bit patterns
        // (single zeros and zero PAIRS) exercise the lo==0 / hi==0 / both==0
        // branches of the 64-bit helpers, which real float data never hits.
        const float mag = 0.001f + float(rng() % 1000000) / 10000.f;
        in[t] = f2u((rng() % 2 ? -1.f : 1.f) * mag);
        if (t % 31 == 5)
            in[t] = 0;                       // isolated zero half
    }
    for (uint32_t k = 0; k < NT / 2; ++k)
        if (k % 37 == 11)
            in[2 * k] = in[2 * k + 1] = 0;   // zero-zero pair (ffsll=0, clzll=64)

    std::vector<uint32_t> cells(3 * C + 2 * C64);
    for (uint32_t c = 0; c < C; ++c)
    {
        cells[c] = f2u(3.402823466e+38f);            // min cells: FLT_MAX
        cells[C + c] = f2u(-3.402823466e+38f);       // max cells: -FLT_MAX
        cells[2 * C + c] = f2u(0.f);                 // add cells
    }

    PxgSlangComputeDevice* dev = testDevice();
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bCells = createBufferFrom(dev, cells);
    std::vector<uint32_t> outInit(NT, 0);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numThreads, numCells, numOrCells, pad; } prm = { NT, C, C64, 0 };
    Kernel k("npAtomicLib.spv", "alAtomics", { 4, 4, 4 }, sizeof(P));
    k.dispatch(NT / 64, { bIn, bCells, bOut }, prm);
    dev->flush();

    const auto gc = downloadBuffer<uint32_t>(dev, bCells, cells.size());
    const auto go = downloadBuffer<uint32_t>(dev, bOut, NT);

    // ---- reference ----
    size_t mm = 0;
    for (uint32_t c = 0; c < C; ++c)
    {
        float mn = 3.402823466e+38f, mx = -3.402823466e+38f;
        int64_t sum = 0;
        for (uint32_t t = c; t < NT; t += C)
        {
            const float v = u2f(in[t]);
            mn = v < mn ? v : mn;
            mx = v > mx ? v : mx;
            sum += int(t % 97u) - 48;
        }
        if (gc[c] != f2u(mn)) { if (mm < 5) std::printf("  MIN MM cell %u\n", c); ++mm; }
        if (gc[C + c] != f2u(mx)) { if (mm < 5) std::printf("  MAX MM cell %u\n", c); ++mm; }
        if (gc[2 * C + c] != f2u(float(sum))) { if (mm < 5) std::printf("  ADD MM cell %u: got %g want %g\n", c, u2f(gc[2 * C + c]), float(sum)); ++mm; }
    }
    for (uint32_t c = 0; c < C64; ++c)
    {
        uint32_t lo = 0, hi = 0;
        for (uint32_t t = c; t < NT; t += C64)
        {
            lo |= 1u << (t & 31u);
            hi |= 1u << ((t * 7u) & 31u);
        }
        if (gc[3 * C + 2 * c] != lo) { if (mm < 5) std::printf("  OR64 lo MM cell %u\n", c); ++mm; }
        if (gc[3 * C + 2 * c + 1] != hi) { if (mm < 5) std::printf("  OR64 hi MM cell %u\n", c); ++mm; }
    }
    for (uint32_t t = 0; t < NT; ++t)
    {
        const uint32_t lo = in[t], hi = in[t ^ 1u];
        const uint32_t e = popcll(lo, hi) | (ffsll64(lo, hi) << 8) | (clzll64(lo, hi) << 16);
        if (go[t] != e) { if (mm < 5) std::printf("  BIT64 MM t%u: got %08x want %08x\n", t, go[t], e); ++mm; }
    }
    PXG_CHECK_MSG(mm == 0, "atomic cells and 64-bit helpers bitwise-match the model (%zu bad)", mm);

    dev->destroyBuffer(bIn);
    dev->destroyBuffer(bCells);
    dev->destroyBuffer(bOut);
}

}  // namespace
