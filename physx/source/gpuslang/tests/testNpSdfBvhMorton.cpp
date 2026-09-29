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

// Verification of npSdfBvhMorton.slang (sdfBvhMorton): the SDF-BVH build
// front-end bvhComputeTriangleBounds (SDFConstruction.cu L150-172) and
// bvhCalculateMortonCodes (L53-82) with part1by2/morton3 (bvh.cuh L40-59) and
// getCenter (L44-51), against an exact CPU transcription.  Bounds and Morton keys
// compared bit for bit.  npSdfBvhMortonSelectTies (P9.89) drives the bounds folds with
// +-0 ties (margin +0 / -0) and NaN vertices, so the selectMin/selectMax winner of
// PxBounds3::include is observable, against an FMin/FMax and a swapped-select twin.

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
struct BmIn  { F4 misc, va, vb, vc, vd; };
struct BmOut { F4 lower, upper, aux; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed

static float selMin(float a, float b) { return a < b ? a : b; }
static float selMax(float a, float b) { return a > b ? a : b; }
static int32_t iselMin(int32_t a, int32_t b) { return a < b ? a : b; }
static int32_t iselMax(int32_t a, int32_t b) { return a < b ? b : a; }

static uint32_t part1by2(uint32_t n)
{
    n = (n ^ (n << 16)) & 0xff0000ffu;
    n = (n ^ (n << 8)) & 0x0300f00fu;
    n = (n ^ (n << 4)) & 0x030c30c3u;
    n = (n ^ (n << 2)) & 0x09249249u;
    return n;
}

static uint32_t morton3(float x, float y, float z, int32_t dim)
{
    const uint32_t ux = uint32_t(iselMin(dim - 1, iselMax(0, int32_t(x * float(dim)))));
    const uint32_t uy = uint32_t(iselMin(dim - 1, iselMax(0, int32_t(y * float(dim)))));
    const uint32_t uz = uint32_t(iselMin(dim - 1, iselMax(0, int32_t(z * float(dim)))));
    return (part1by2(uz) << 2) | (part1by2(uy) << 1) | part1by2(ux);
}

static void refBm(const BmIn& s, BmOut& o)
{
    std::memset(&o, 0, sizeof(o));
    const uint32_t mode = f2u(s.misc.y);

    if (mode == 0u)
    {
        const float margin = s.misc.x;
        float lox = s.va.x, loy = s.va.y, loz = s.va.z;
        float hix = s.va.x, hiy = s.va.y, hiz = s.va.z;
        lox = selMin(lox, s.vb.x); loy = selMin(loy, s.vb.y); loz = selMin(loz, s.vb.z);
        hix = selMax(hix, s.vb.x); hiy = selMax(hiy, s.vb.y); hiz = selMax(hiz, s.vb.z);
        lox = selMin(lox, s.vc.x); loy = selMin(loy, s.vc.y); loz = selMin(loz, s.vc.z);
        hix = selMax(hix, s.vc.x); hiy = selMax(hiy, s.vc.y); hiz = selMax(hiz, s.vc.z);
        lox -= margin; loy -= margin; loz -= margin;
        hix += margin; hiy += margin; hiz += margin;
        o.lower = F4{ lox, loy, loz, 0.f };
        o.upper = F4{ hix, hiy, hiz, 0.f };
    }
    else
    {
        const int32_t priority = (int32_t)f2u(s.misc.z);
        const bool hasPriority = f2u(s.misc.w) == 1u;

        const float cx = 0.5f * (s.va.x + s.vb.x);
        const float cy = 0.5f * (s.va.y + s.vb.y);
        const float cz = 0.5f * (s.va.z + s.vb.z);
        const float lx = (cx - s.vc.x) * s.vd.x;
        const float ly = (cy - s.vc.y) * s.vd.y;
        const float lz = (cz - s.vc.z) * s.vd.z;

        int32_t key;
        if (hasPriority)
        {
            key = (int32_t)morton3(lx, ly, lz, 512);
            key |= (~priority) << 27;
        }
        else
        {
            key = (int32_t)morton3(lx, ly, lz, 1024);
        }
        o.aux = F4{ u2f((uint32_t)key), 0.f, 0.f, 0.f };
    }
}

PXG_TEST(npSdfBvhMorton)
{
    std::mt19937 rng(260001);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<BmIn> in(N);
    std::vector<BmOut> ref(N);

    for (uint32_t i = 0; i < N; ++i)
    {
        BmIn s{};
        const uint32_t mode = i % 2u;
        if (mode == 0u)
        {
            s.misc = F4{ rf(0.f, 0.2f), u2f(0u), 0.f, 0.f };
            s.va = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
            s.vb = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
            s.vc = F4{ rf(-2.f, 2.f), rf(-2.f, 2.f), rf(-2.f, 2.f), 0.f };
            // zero-margin stripe
            if ((i % 7u) == 2u) s.misc.x = 0.f;
            // degenerate triangle: all three vertices identical
            if ((i % 13u) == 6u) { s.vb = s.va; s.vc = s.va; }
        }
        else
        {
            const uint32_t hasPriority = (i / 2u) % 2u;
            const uint32_t priority = rng() % 32u;   // 5-bit priorities
            s.misc = F4{ 0.f, u2f(1u), u2f(priority), u2f(hasPriority) };
            // item bounds around a center within/near the grid
            const float cx = rf(-1.f, 3.f), cy = rf(-1.f, 3.f), cz = rf(-1.f, 3.f);
            const float ex = rf(0.f, 0.3f), ey = rf(0.f, 0.3f), ez = rf(0.f, 0.3f);
            s.va = F4{ cx - ex, cy - ey, cz - ez, 0.f };
            s.vb = F4{ cx + ex, cy + ey, cz + ez, 0.f };
            s.vc = F4{ rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), 0.f };  // gridLower
            s.vd = F4{ rf(0.2f, 1.f), rf(0.2f, 1.f), rf(0.2f, 1.f), 0.f };        // gridInvEdges
            // out-of-grid stripe: local coords negative / > 1 exercise BOTH
            // clamp arms of morton3
            if ((i % 11u) == 4u) s.vc = F4{ 5.f, 5.f, 5.f, 0.f };
            if ((i % 11u) == 8u) s.vc = F4{ -9.f, -9.f, -9.f, 0.f };
            // exact grid-corner stripe: local == 0 and local == 1 boundaries
            if ((i % 19u) == 5u)
            {
                s.va = F4{ 1.f, 1.f, 1.f, 0.f }; s.vb = F4{ 1.f, 1.f, 1.f, 0.f };
                s.vc = F4{ ((i / 38u) % 2u) ? 1.f : 0.f, 0.f, 1.f, 0.f };
                s.vd = F4{ 1.f, 1.f, 1.f, 0.f };
            }
            // priority extremes: 0 and 31
            if ((i % 23u) == 7u) s.misc.z = u2f(((i / 46u) % 2u) ? 31u : 0u);
        }
        in[i] = s;
        refBm(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<BmOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npSdfBvhMorton.spv", "sdfBvhMorton",
             { (uint32_t)sizeof(BmIn), (uint32_t)sizeof(BmOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<BmOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        bool ok = eqFZ(ref[i].lower.x, g[i].lower.x) && eqFZ(ref[i].lower.y, g[i].lower.y) && eqFZ(ref[i].lower.z, g[i].lower.z)
               && eqFZ(ref[i].upper.x, g[i].upper.x) && eqFZ(ref[i].upper.y, g[i].upper.y) && eqFZ(ref[i].upper.z, g[i].upper.z)
               && f2u(ref[i].aux.x) == f2u(g[i].aux.x);
        if (!ok)
        {
            if (mm < 5)
                std::printf("MM %u loR (%g %g %g) loG (%g %g %g) keyR %08x keyG %08x\n", i,
                            ref[i].lower.x, ref[i].lower.y, ref[i].lower.z,
                            g[i].lower.x, g[i].lower.y, g[i].lower.z,
                            f2u(ref[i].aux.x), f2u(g[i].aux.x));
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "sdfBvhMorton outputs bitwise-identical (keys exact)");
}

// ---------------------------------------------------------------------------------------------
// P9.89 (npSdfBvhMortonSelectTies): the tie and NaN winner of the bounds folds.
// bvhComputeTriangleBounds (SDFConstruction.cu:161-164) folds PxBounds3(a, a); include(b);
// include(c), and include (PxBounds3.h:333-338) is PxVec3::minimum/maximum = componentwise
// PxMin/PxMax(float) = intrinsics::selectMin/selectMax (PxMath.h:82-101,
// PxUnixMathIntrinsics.h:116-125): acc < v ? acc : v / acc > v ? acc : v.  So on a +-0 tie the
// NEW vertex survives, a NaN vertex replaces the accumulator, and the next vertex replaces that
// NaN (a NaN in c therefore reaches the output, a NaN in a or b does not).  The old port spelled
// the folds as Slang min()/max(), i.e. GLSL.std.450 FMin/FMax, which the SZINP-stamped llvmpipe
// build runs zero-ordered (a +-0 tie gives -0 for min and +0 for max in either order) and
// NaN-ignoring (P9.86 F4 / P9.86f).  refBm above is the CUDA; refBmTwin replays, per fold
// position, either that FMin/FMax (llvmpipe) or the operand-swapped select v < acc ? v : acc
// (the SPIR-V spec's FMin = the "returns acc on a tie" spelling).
//
// The fattenFast margin decides which tie reaches the output: lower = lo - margin keeps the
// sign of a zero only for margin = +0, upper = hi + margin only for margin = -0.  So the tie
// rows come in a lower flavour (margin +0) and an upper flavour (margin -0); every component
// rotates through the four (fold, zero order) cases.  The NaN rows carry distinct payloads so
// the winning NaN is identified, with random finite partners on both sides of each other.
// ---------------------------------------------------------------------------------------------

// llvmpipe GLSL.std.450 FMin/FMax under SignedZeroInfNanPreserve (P9.86 F4): mutant twin only.
static float lpFMin(float a, float b)
{
    if (std::isnan(a)) return b;
    if (std::isnan(b)) return a;
    if (a == b) return std::signbit(a) ? a : b;
    return a < b ? a : b;
}
static float lpFMax(float a, float b)
{
    if (std::isnan(a)) return b;
    if (std::isnan(b)) return a;
    if (a == b) return std::signbit(a) ? b : a;
    return a > b ? a : b;
}

// Fold positions: p = fold * 3 + component, fold 0 = lo<-b, 1 = hi<-b, 2 = lo<-c, 3 = hi<-c.
// fminMask bit p: that position uses lpFMin/lpFMax; swapMask bit p: the operand-swapped select.
static void refBmTwin(const BmIn& s, BmOut& o, uint32_t fminMask, uint32_t swapMask)
{
    const float a[3] = { s.va.x, s.va.y, s.va.z };
    const float b[3] = { s.vb.x, s.vb.y, s.vb.z };
    const float c[3] = { s.vc.x, s.vc.y, s.vc.z };
    const float margin = s.misc.x;
    float lo[3], hi[3];
    auto mn = [&](uint32_t p, float acc, float v) {
        if (fminMask & (1u << p)) return lpFMin(acc, v);
        if (swapMask & (1u << p)) return selMin(v, acc);
        return selMin(acc, v);
    };
    auto mx = [&](uint32_t p, float acc, float v) {
        if (fminMask & (1u << p)) return lpFMax(acc, v);
        if (swapMask & (1u << p)) return selMax(v, acc);
        return selMax(acc, v);
    };
    for (uint32_t k = 0; k < 3; ++k)
    {
        lo[k] = a[k]; hi[k] = a[k];
        lo[k] = mn(0u * 3u + k, lo[k], b[k]);
        hi[k] = mx(1u * 3u + k, hi[k], b[k]);
        lo[k] = mn(2u * 3u + k, lo[k], c[k]);
        hi[k] = mx(3u * 3u + k, hi[k], c[k]);
        lo[k] -= margin;
        hi[k] += margin;
    }
    std::memset(&o, 0, sizeof(o));
    o.lower = F4{ lo[0], lo[1], lo[2], 0.f };
    o.upper = F4{ hi[0], hi[1], hi[2], 0.f };
}

static bool bmSame(const BmOut& x, const BmOut& y) { return std::memcmp(&x, &y, sizeof(BmOut)) == 0; }

PXG_TEST(npSdfBvhMortonSelectTies)
{
    std::mt19937 rng(890013);
    const uint32_t N = 4096;   // 64 workgroups of 64
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    const float PZ = 0.0f, NZ = u2f(0x80000000u);
    const float NAN_A = u2f(0x7fc0a0a1u), NAN_B = u2f(0x7fc0b0b3u), NAN_C = u2f(0xffc0c0c5u);

    enum { K_TIE_LO, K_TIE_HI, K_NAN_B, K_NAN_C, K_NAN_A, K_NAN_BC, K_MIXED, K_BASE, K_COUNT };
    uint32_t kindRows[K_COUNT] = {};

    std::vector<BmIn> in(N);
    std::vector<BmOut> ref(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        BmIn s{};
        const uint32_t kind = i % K_COUNT;
        const uint32_t rot = i / K_COUNT;
        ++kindRows[kind];
        s.misc = F4{ rf(0.01f, 0.2f), u2f(0u), 0.f, 0.f };
        float a[3], b[3], c[3];
        for (uint32_t k = 0; k < 3; ++k) { a[k] = rf(-2.f, 2.f); b[k] = rf(-2.f, 2.f); c[k] = rf(-2.f, 2.f); }
        for (uint32_t k = 0; k < 3; ++k)
        {
            const uint32_t j = (rot + k) % 4u;
            switch (kind)
            {
            case K_TIE_LO:   // margin +0: lower shows the zero sign; the tie is at lo<-b or lo<-c
                s.misc.x = PZ;
                if (j == 0) { a[k] = NZ; b[k] = PZ; c[k] = rf(0.1f, 2.f); }        // CUDA +0, FMin -0
                else if (j == 1) { a[k] = PZ; b[k] = NZ; c[k] = rf(0.1f, 2.f); }   // CUDA -0, swapped +0
                else if (j == 2) { a[k] = NZ; b[k] = rf(0.1f, 2.f); c[k] = PZ; }   // CUDA +0, FMin -0
                else { a[k] = PZ; b[k] = rf(0.1f, 2.f); c[k] = NZ; }               // CUDA -0, swapped +0
                break;
            case K_TIE_HI:   // margin -0: upper shows the zero sign; the tie is at hi<-b or hi<-c
                s.misc.x = NZ;
                if (j == 0) { a[k] = PZ; b[k] = NZ; c[k] = rf(-2.f, -0.1f); }       // CUDA -0, FMax +0
                else if (j == 1) { a[k] = NZ; b[k] = PZ; c[k] = rf(-2.f, -0.1f); }  // CUDA +0, swapped -0
                else if (j == 2) { a[k] = PZ; b[k] = rf(-2.f, -0.1f); c[k] = NZ; }  // CUDA -0, FMax +0
                else { a[k] = NZ; b[k] = rf(-2.f, -0.1f); c[k] = PZ; }              // CUDA +0, swapped -0
                break;
            case K_NAN_B:    // CUDA: lo = hi = c; FMin/FMax: min/max(a, c)
                b[k] = NAN_B;
                if (j == 3) s.misc.x = PZ;
                break;
            case K_NAN_C:    // CUDA: lo = hi = NaN_c (on alternating components); FMin: min/max(a, b)
                if (((rot + k) & 1u) == 0u) c[k] = NAN_C;
                break;
            case K_NAN_A:    // both: the NaN accumulator is replaced by b (agreement row)
                if (j != 1) a[k] = NAN_A;
                break;
            case K_NAN_BC:   // CUDA: NaN_c wins over NaN_b; FMin/FMax: a
                b[k] = NAN_B; c[k] = NAN_C;
                break;
            case K_MIXED:    // zeros of both signs mixed with finite values, all three margins
            {
                const float pool[4] = { PZ, NZ, rf(-2.f, 2.f), rf(-2.f, 2.f) };
                a[k] = pool[rng() % 4u]; b[k] = pool[rng() % 4u]; c[k] = pool[rng() % 4u];
                const uint32_t m = rng() % 3u;
                s.misc.x = (m == 0u) ? PZ : ((m == 1u) ? NZ : rf(0.01f, 0.2f));
                break;
            }
            default: break;   // K_BASE: distinct finite values, random margin
            }
        }
        s.va = F4{ a[0], a[1], a[2], 0.f };
        s.vb = F4{ b[0], b[1], b[2], 0.f };
        s.vc = F4{ c[0], c[1], c[2], 0.f };
        in[i] = s;
        refBm(s, ref[i]);
    }

    // The twin replay must agree with refBm when nothing is replaced (it is the same fold).
    size_t twinSelfBad = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        BmOut t;
        refBmTwin(in[i], t, 0u, 0u);
        if (!bmSame(t, ref[i])) ++twinSelfBad;
    }
    PXG_CHECK_MSG(twinSelfBad == 0, "P9.89: refBmTwin(0, 0) differs from refBm on %zu rows", twinSelfBad);

    // Coverage: every fold position (4 folds x 3 components) must be observable for both
    // replacements -- the llvmpipe FMin/FMax and the operand-swapped select -- on many rows,
    // and the whole old port (all 12 positions FMin/FMax) on a large share of the rows.
    const char* foldName[4] = { "lo<-b", "hi<-b", "lo<-c", "hi<-c" };
    uint32_t minFmin = ~0u, minSwap = ~0u;
    for (uint32_t p = 0; p < 12; ++p)
    {
        uint32_t dF = 0, dS = 0;
        for (uint32_t i = 0; i < N; ++i)
        {
            BmOut t;
            refBmTwin(in[i], t, 1u << p, 0u);
            if (!bmSame(t, ref[i])) ++dF;
            refBmTwin(in[i], t, 0u, 1u << p);
            if (!bmSame(t, ref[i])) ++dS;
        }
        std::printf("[ info ] selectTies %s.%c: FMin/FMax twin differs on %u rows, swapped select on %u\n",
                    foldName[p / 3], "xyz"[p % 3], dF, dS);
        if (dF < minFmin) minFmin = dF;
        if (dS < minSwap) minSwap = dS;
    }
    std::vector<BmOut> oldPort(N);
    uint32_t oldDiff = 0, oldDiffLower = 0, oldDiffUpper = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        refBmTwin(in[i], oldPort[i], 0xfffu, 0u);
        if (!bmSame(oldPort[i], ref[i])) ++oldDiff;
        if (std::memcmp(&oldPort[i].lower, &ref[i].lower, sizeof(F4)) != 0) ++oldDiffLower;
        if (std::memcmp(&oldPort[i].upper, &ref[i].upper, sizeof(F4)) != 0) ++oldDiffUpper;
    }
    std::printf("[ info ] selectTies: %u rows (%u per kind); old-port FMin/FMax twin differs on %u rows"
                " (%u lower, %u upper); per-position minimum %u (FMin/FMax) / %u (swapped)\n",
                N, kindRows[0], oldDiff, oldDiffLower, oldDiffUpper, minFmin, minSwap);
    PXG_CHECK_MSG(minFmin >= 32u && minSwap >= 32u && oldDiff >= N / 3u && oldDiffLower >= N / 8u &&
                      oldDiffUpper >= N / 8u,
                  "P9.89: the select-tie rows lost coverage (min FMin %u, min swapped %u, old port %u rows,"
                  " %u lower, %u upper)", minFmin, minSwap, oldDiff, oldDiffLower, oldDiffUpper);

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<BmOut> outInit(N);
    std::memset(outInit.data(), 0xcd, N * sizeof(BmOut));
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);
    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npSdfBvhMorton.spv", "sdfBvhMorton",
             { (uint32_t)sizeof(BmIn), (uint32_t)sizeof(BmOut) }, sizeof(P));
    k.dispatch((N + 63) / 64, { bIn, bOut }, prm);
    dev->flush();
    const auto g = downloadBuffer<BmOut>(dev, bOut, N);

    size_t mm = 0, asOldPort = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (bmSame(g[i], ref[i]))
            continue;
        if (!bmSame(oldPort[i], ref[i]) && bmSame(g[i], oldPort[i]))
            ++asOldPort;
        if (mm < 6)
            std::printf("MM %u kind %u lo R %08x %08x %08x G %08x %08x %08x | hi R %08x %08x %08x G %08x %08x %08x\n",
                        i, i % K_COUNT, f2u(ref[i].lower.x), f2u(ref[i].lower.y), f2u(ref[i].lower.z),
                        f2u(g[i].lower.x), f2u(g[i].lower.y), f2u(g[i].lower.z), f2u(ref[i].upper.x),
                        f2u(ref[i].upper.y), f2u(ref[i].upper.z), f2u(g[i].upper.x), f2u(g[i].upper.y),
                        f2u(g[i].upper.z));
        ++mm;
    }
    if (asOldPort)
        std::printf("MM selectTies: %zu rows reproduce the old port's FMin/FMax twin (Slang min()/max())\n",
                    asOldPort);
    PXG_CHECK_MSG(mm == 0, "P9.89: sdfBvhMorton bounds folds differ from the CUDA selectMin/selectMax on %zu rows"
                  " (%zu as the FMin/FMax twin)", mm, asOldPort);
}

}  // namespace
