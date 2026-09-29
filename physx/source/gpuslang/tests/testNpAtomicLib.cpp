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

// ---------------------------------------------------------------------------
// P9.89: the SELECT rule of atomic.cuh:104-132.  The battery above feeds only
// finite, non-zero-tie values into FLT_MAX cells, where the CUDA's
// `while (val < old) CAS(val)` and an fminf/fmaxf combine agree -- so it could
// not see that the pre-P9.89 port CASed FMin/FMax(value, old).  These cells
// are exactly where the two differ, and each one's CUDA outcome is the same
// for EVERY arrival order (checked below by folding both ways):
//   g0  Min, resident +0, fed -0           -> +0 stays (-0 < +0 is false)
//   g1  Max, resident -0, fed +0           -> -0 stays
//   g2  Min, resident +0, fed -0/+0/pos    -> +0
//   g3  Max, resident -0, fed +0/-0/neg    -> -0
//   g4  Min, resident NaN, fed finite      -> the NaN, payload and all
//   g5  Max, resident NaN, fed finite      -> the NaN
//   g6  Min|Max, resident NaN, fed NaN+finite -> the NaN
//   g7  Min|Max, resident NaN, fed NaN only -> the resident NaN's bits
//   g8  Min|Max, resident +-FLT_MAX, fed finite + NaN -> select min/max (NaN val never written)
//   g9  Min resident -0 fed +0 / Max resident +0 fed -0 -> resident stays
//   g10 Min|Max, resident +-FLT_MAX, fed +-0 and nonzero -> the nonzero extreme
// g0..g6 are DISCRIMINATING: the twin of the old port (llvmpipe FMin/FMax =
// bcsel(feq, imin/imax(bits), fmin/fmax_nsz): -0/+0 on a zero tie in either
// order, the non-NaN operand of a one-NaN pair) must end elsewhere on every
// one of those cells.  g7's twin is unknowable (two-NaN payload choice), and
// g8..g10 guard the cases where both rules agree.
// ---------------------------------------------------------------------------

// atomicCAS(address, compare, val): returns the old word, writes val iff equal.
static uint32_t hostCas(uint32_t& word, uint32_t compare, uint32_t val)
{
    const uint32_t old = word;
    if (old == compare)
        word = val;
    return old;
}

// atomic.cuh:104-117 AtomicMin / :119-132 AtomicMax, one thread at a time (a
// legal serialization of the contended loop).
static void cudaAtomicMin(uint32_t& cell, float val)
{
    uint32_t old = cell, assumed;
    while (val < u2f(old))
    {
        assumed = old;
        old = hostCas(cell, assumed, f2u(val));
    }
}
static void cudaAtomicMax(uint32_t& cell, float val)
{
    uint32_t old = cell, assumed;
    while (val > u2f(old))
    {
        assumed = old;
        old = hostCas(cell, assumed, f2u(val));
    }
}

// Twin of the pre-P9.89 port: CAS FMin/FMax(value, old) with llvmpipe's
// lowering.  `nanPick` is set for a NaN/NaN pair, whose result payload
// gallivm picks (not modeled); any later pair with a non-NaN side is
// determined again, whatever NaN the cell held.
static uint32_t lpFminmax(uint32_t a, uint32_t b, bool isMax, bool& nanPick)
{
    nanPick = false;
    const float fa = u2f(a), fb = u2f(b);
    if (fa == fb)   // +-0 tie (or identical bits): imin/imax on the bit patterns
    {
        const int32_t ia = int32_t(a), ib = int32_t(b);
        return uint32_t(isMax ? (ia > ib ? ia : ib) : (ia < ib ? ia : ib));
    }
    const bool na = fa != fa, nb = fb != fb;
    if (na && nb) { nanPick = true; return a; }
    if (na) return b;
    if (nb) return a;
    return isMax ? (fa > fb ? a : b) : (fa < fb ? a : b);
}

PXG_TEST(npAtomicLibSelectRule)
{
    const uint32_t NT = 4096, G = 11, PER = 16, NC = G * PER;
    std::mt19937 rng(8989);

    // Non-round magnitude in [2^-7, 2^9): random exponent and full mantissa.
    auto randMag = [&]() { return u2f(((120u + rng() % 16u) << 23) | (rng() & 0x007fffffu)); };
    auto randNan = [&]() {
        uint32_t m = rng() & 0x007fffffu;
        if (rng() % 2) m |= 0x00400000u;        // quiet and signalling payloads
        if (m == 0u) m = 1u;
        return (rng() % 2 ? 0x80000000u : 0u) | 0x7f800000u | m;
    };
    const uint32_t PZ = 0x00000000u, NZ = 0x80000000u;
    const uint32_t FMAX = f2u(3.402823466e+38f), NFMAX = f2u(-3.402823466e+38f);

    std::vector<uint32_t> cells(NC), sel(NT), in(NT);
    std::vector<uint8_t> isMax(NC);
    for (uint32_t c = 0; c < NC; ++c)
    {
        const uint32_t g = c / PER;
        const bool mx = (g == 1 || g == 3 || g == 5) || ((g >= 6) && (c & 1u));
        isMax[c] = mx ? 1 : 0;
        switch (g)
        {
        case 0: case 2: cells[c] = PZ; break;
        case 1: case 3: cells[c] = NZ; break;
        case 4: case 5: case 6: case 7: cells[c] = randNan(); break;
        case 9: cells[c] = mx ? PZ : NZ; break;
        default: cells[c] = mx ? NFMAX : FMAX; break;   // g8, g10
        }
    }
    for (uint32_t t = 0; t < NT; ++t)
    {
        const uint32_t c = t % NC, g = c / PER, k = t / NC;
        const bool mx = isMax[c] != 0;
        sel[t] = c | (mx ? 0x80000000u : 0u);
        const float s = rng() % 2 ? -1.f : 1.f;
        uint32_t v;
        switch (g)
        {
        case 0: v = NZ; break;
        case 1: v = PZ; break;
        case 2: v = k % 3 == 0 ? NZ : k % 3 == 1 ? PZ : f2u(randMag()); break;
        case 3: v = k % 3 == 0 ? PZ : k % 3 == 1 ? NZ : f2u(-randMag()); break;
        case 4: case 5: v = f2u(s * randMag()); break;
        case 6: v = k % 2 == 0 ? randNan() : f2u(s * randMag()); break;
        case 7: v = randNan(); break;
        case 8: v = k % 3 == 2 ? randNan() : f2u(s * randMag()); break;
        case 9: v = mx ? NZ : PZ; break;
        default: v = k % 4 == 0 ? PZ : k % 4 == 1 ? NZ : f2u(s * randMag()); break;   // g10
        }
        in[t] = v;
    }

    // ---- CUDA oracle (both fold orders) and the old-port twin ----
    std::vector<uint32_t> want(cells), wantRev(cells), twin(cells);
    std::vector<uint8_t> twinKnown(NC, 1);
    for (uint32_t t = 0; t < NT; ++t)
    {
        const uint32_t c = t % NC;
        if (isMax[c]) cudaAtomicMax(want[c], u2f(in[t])); else cudaAtomicMin(want[c], u2f(in[t]));
        bool nanPick;
        twin[c] = lpFminmax(in[t], twin[c], isMax[c] != 0, nanPick);
        twinKnown[c] = nanPick ? 0 : 1;
    }
    for (uint32_t t = NT; t-- > 0;)
    {
        const uint32_t c = t % NC;
        if (isMax[c]) cudaAtomicMax(wantRev[c], u2f(in[t])); else cudaAtomicMin(wantRev[c], u2f(in[t]));
    }
    size_t orderDep = 0, twinDiff = 0, g10Zero = 0;
    for (uint32_t c = 0; c < NC; ++c)
    {
        orderDep += want[c] != wantRev[c] ? 1 : 0;
        if (c / PER <= 6)
            twinDiff += twinKnown[c] && twin[c] != want[c] ? 1 : 0;
        if (c / PER == 10)
            g10Zero += (want[c] & 0x7fffffffu) == 0u ? 1 : 0;
    }
    PXG_CHECK_MSG(orderDep == 0, "every scripted cell's CUDA outcome is arrival-order independent (%zu not)", orderDep);
    PXG_CHECK_MSG(g10Zero == 0, "g10 cells end on a nonzero extreme, not on a zero tie (%zu do)", g10Zero);
    PXG_CHECK_MSG(twinDiff == 7 * PER,
                  "the pre-P9.89 FMin/FMax-combine twin differs from atomic.cuh on every g0..g6 cell (%zu of %u)",
                  twinDiff, 7 * PER);

    // ---- device ----
    PxgSlangComputeDevice* dev = testDevice();
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bCells = createBufferFrom(dev, cells);
    PxgBufferHandle bSel = createBufferFrom(dev, sel);

    struct P { uint32_t numThreads, numCells, numOrCells, pad; } prm = { NT, NC, 0, 0 };
    Kernel k("npAtomicLib.spv", "alAtomicSelect", { 4, 4, 4 }, sizeof(P));
    k.dispatch(NT / 64, { bIn, bCells, bSel }, prm);
    dev->flush();
    const auto gc = downloadBuffer<uint32_t>(dev, bCells, NC);

    size_t mm = 0;
    size_t mmGroup[G] = {};
    for (uint32_t c = 0; c < NC; ++c)
        if (gc[c] != want[c])
        {
            if (mm < 6)
                std::printf("  SELECT MM cell %u (g%u, %s): got %08x want %08x (old-port twin %08x)\n", c, c / PER,
                            isMax[c] ? "max" : "min", gc[c], want[c], twin[c]);
            ++mm;
            ++mmGroup[c / PER];
        }
    if (mm)
    {
        std::printf("  mismatches per group:");
        for (uint32_t g = 0; g < G; ++g)
            std::printf(" g%u=%zu", g, mmGroup[g]);
        std::printf("\n");
    }
    PXG_CHECK_MSG(mm == 0, "AtomicMin/AtomicMax cells bitwise-match atomic.cuh's select rule (%zu bad)", mm);

    dev->destroyBuffer(bIn);
    dev->destroyBuffer(bCells);
    dev->destroyBuffer(bSel);
}

}  // namespace
