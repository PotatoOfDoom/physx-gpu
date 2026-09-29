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

// RUNTIME_PLAN.md P2.1 battery: the warp-32 primitive library
// (pxgSlangWarpInc.slang, shared-memory emulation path) against a CPU model
// of the exact CUDA semantics.  Scripted per-workgroup tests sweep widths
// 1..32, partial masks (0x3F contact-patch, 4-lane nibbles, random), wrapped
// and clamped shuffle indices; every lane output must match bitwise.
// The model simulates each shuffle step in LOCKSTEP (snapshot arrays), which
// is exactly what the barrier-synchronized scratch round-trip guarantees.

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
static uint32_t popc(uint32_t v) { return uint32_t(__builtin_popcount(v)); }

static const uint32_t T = 256;   // tests (one 64-thread workgroup each)

// ---- CPU model of the library, per 32-lane tile (lockstep) ----

// P9.89: the shuffle model is the PTX ISA shfl.sync pseudo-code, fed the c
// operand the CUDA headers build (sm_30_intrinsics.hpp: __shfl_up_sync
// c = (32-width)<<8; __shfl_sync / __shfl_down_sync / __shfl_xor_sync
// c = ((32-width)<<8)|0x1f):
//   bval = b[4:0]; cval = c[4:0]; segmask = c[12:8];
//   maxLane = (lane & segmask) | (cval & ~segmask); minLane = lane & segmask;
//   up: j = lane - bval, pval = j >= minLane    down: j = lane + bval, pval = j <= maxLane
//   bfly: j = lane ^ bval, pval = j <= maxLane  idx: j = minLane | (bval & ~segmask), pval = j <= maxLane
//   if (!pval) j = lane;
// Before P9.89 this model copied the PORT's rule (clamp on ANY other segment,
// untruncated delta/xorMask), so npWarpLib passed on non-CUDA semantics.
enum ShflMode { kIdx, kUp, kDown, kBfly };
static float ptxShfl(const float* t, uint32_t lane, ShflMode mode, uint32_t b, uint32_t width)
{
    const uint32_t c = ((32u - width) << 8) | (mode == kUp ? 0u : 0x1fu);
    const uint32_t bval = b & 31u, cval = c & 31u, segmask = (c >> 8) & 31u;
    const int maxLane = int((lane & segmask) | (cval & ~segmask & 31u));
    const int minLane = int(lane & segmask);
    int j;
    bool pval;
    switch (mode)
    {
    case kUp:   j = int(lane) - int(bval); pval = j >= minLane; break;
    case kDown: j = int(lane) + int(bval); pval = j <= maxLane; break;
    case kBfly: j = int(lane ^ bval);      pval = j <= maxLane; break;
    default:    j = minLane | int(bval & ~segmask & 31u); pval = j <= maxLane; break;
    }
    if (!pval)
        j = int(lane);
    return t[j];
}
static float mShfl(const float* t, uint32_t lane, int srcLane, uint32_t width)
{
    return ptxShfl(t, lane, kIdx, uint32_t(srcLane), width);
}
static float mUp(const float* t, uint32_t lane, uint32_t delta, uint32_t width)
{
    return ptxShfl(t, lane, kUp, delta, width);
}
static float mDown(const float* t, uint32_t lane, uint32_t delta, uint32_t width)
{
    return ptxShfl(t, lane, kDown, delta, width);
}
static float mXor(const float* t, uint32_t lane, uint32_t xm, uint32_t width)
{
    return ptxShfl(t, lane, kBfly, xm, width);
}

// Twins of the pre-P9.89 port rule; the script must make them differ.
static float oldUp(const float* t, uint32_t lane, uint32_t delta, uint32_t width)
{
    return (lane & (width - 1)) < delta ? t[lane] : t[lane - delta];
}
static float oldDown(const float* t, uint32_t lane, uint32_t delta, uint32_t width)
{
    return (lane & (width - 1)) + delta >= width ? t[lane] : t[lane + delta];
}
static float oldXor(const float* t, uint32_t lane, uint32_t xm, uint32_t width)
{
    const uint32_t src = lane ^ xm;
    return (src & ~(width - 1)) != (lane & ~(width - 1)) ? t[lane] : t[src & 31];
}

template <typename V, typename OP>
static void mButterfly(V* v, uint32_t width, OP op)
{
    V nxt[32];
    for (uint32_t radius = width >> 1; radius > 0; radius >>= 1)
    {
        for (uint32_t lane = 0; lane < 32; ++lane)
            nxt[lane] = op(v[lane], v[lane ^ radius]);
        std::memcpy(v, nxt, sizeof(nxt));
    }
}

static void mScanAddInclusive(uint32_t* v, uint32_t width)
{
    uint32_t nxt[32];
    for (uint32_t radius = 1; radius < width; radius <<= 1)
    {
        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            const uint32_t laneInSeg = lane & (width - 1);
            nxt[lane] = laneInSeg >= radius ? v[lane] + v[lane - radius] : v[lane];
        }
        std::memcpy(v, nxt, sizeof(nxt));
    }
}

PXG_TEST(npWarpLib)
{
    std::mt19937 rng(4242);
    std::vector<uint32_t> script(4 * T);
    std::vector<float> vals(64 * T);
    std::vector<uint32_t> pred(64 * T);

    for (uint32_t t = 0; t < T; ++t)
    {
        uint32_t mask;
        switch (t % 5)
        {
        case 0: mask = 0xffffffffu; break;
        case 1: mask = 0x3fu; break;                          // 6-lane contact patch
        case 2: mask = 0xfu << ((t / 5) % 8 * 4); break;      // 4-lane nibble groups
        case 3: mask = rng() | 1u; break;
        default: mask = (rng() | 1u) & 0x0000ffffu | 0x10000u; break;
        }
        const uint32_t log2width = t % 6;                     // widths 1..32
        const int srcLane = int(rng() % 121) - 40;            // wraps + negatives
        // P9.89: delta up to 63 (was < 11) so the PTX 5-bit truncation of
        // up/down deltas >= 32 is exercised; xorMask reaches both earlier
        // segments (bits in [width, 32)) and >= 32 (truncated) partners.
        const uint32_t delta = rng() % 64;
        const uint32_t xorMask = rng() % 64;                  // includes out-of-warp XOR
        script[4 * t + 0] = mask;
        script[4 * t + 1] = log2width;
        script[4 * t + 2] = uint32_t(srcLane);
        script[4 * t + 3] = delta | (xorMask << 8);

        for (uint32_t i = 0; i < 64; ++i)
        {
            const uint32_t k = 64 * t + i;
            switch (i % 4)
            {
            case 0: vals[k] = -8.f + float(rng() % 160000) / 10000.f; break;
            case 1: vals[k] = float(int(rng() % 2001) - 1000); break;   // exact ints
            case 2: vals[k] = u2f((rng() & 0x007fffffu) | 0x3f800000u); break; // [1,2) mantissa soup
            default: vals[k] = (rng() % 2 ? -1.f : 1.f) * float(rng() % 97) * 0.5f; break;
            }
            pred[k] = (rng() % 3) == 0 ? 1u : 0u;
        }
    }

    PxgSlangComputeDevice* dev = testDevice();
    PxgBufferHandle bScript = createBufferFrom(dev, script);
    PxgBufferHandle bVals = createBufferFrom(dev, vals);
    PxgBufferHandle bPred = createBufferFrom(dev, pred);
    std::vector<uint32_t> outInit(4 * 3 * 64 * T, 0);

    struct P { uint32_t numTests, pad0, pad1, pad2; } prm = { T, 0, 0, 0 };

    // ---- wlShuffle ----
    {
        PxgBufferHandle bOut = createBufferFrom(dev, outInit);
        Kernel k("npWarpLib.spv", "wlShuffle", { 16, 4, 4, 16 }, sizeof(P));
        k.dispatch(T, { bScript, bVals, bPred, bOut }, prm);
        dev->flush();
        const auto g = downloadBuffer<uint32_t>(dev, bOut, outInit.size());

        size_t mm = 0;
        // Lanes where the pre-P9.89 twin gives other BITS, per class.
        size_t oldUpDiff = 0, oldDownDiff = 0, oldXorEarlier = 0, oldXorTrunc = 0;
        for (uint32_t t = 0; t < T; ++t)
        {
            const uint32_t width = 1u << script[4 * t + 1];
            const int srcLane = int(script[4 * t + 2]);
            const uint32_t delta = script[4 * t + 3] & 0xff;
            const uint32_t xm = (script[4 * t + 3] >> 8) & 0xff;
            for (uint32_t tile = 0; tile < 2; ++tile)
            {
                const float* tv = &vals[64 * t + 32 * tile];
                for (uint32_t lane = 0; lane < 32; ++lane)
                {
                    const uint32_t tid = 32 * tile + lane;
                    const uint32_t* o = &g[(64 * t + tid) * 12];
                    const uint32_t e[4] = { f2u(mShfl(tv, lane, srcLane, width)),
                                            f2u(mUp(tv, lane, delta, width)),
                                            f2u(mDown(tv, lane, delta, width)),
                                            f2u(mXor(tv, lane, xm, width)) };
                    oldUpDiff += f2u(oldUp(tv, lane, delta, width)) != e[1] ? 1 : 0;
                    oldDownDiff += f2u(oldDown(tv, lane, delta, width)) != e[2] ? 1 : 0;
                    if (f2u(oldXor(tv, lane, xm, width)) != e[3])
                        ++(xm >= 32u ? oldXorTrunc : oldXorEarlier);
                    for (int c = 0; c < 4; ++c)
                        if (o[c] != e[c])
                        {
                            if (mm < 5)
                                std::printf("  SH MM t%u tid%u c%d: got %08x want %08x\n", t, tid, c, o[c], e[c]);
                            ++mm;
                        }
                }
            }
        }
        std::printf("  pre-P9.89 twin differs on %zu up, %zu down, %zu xor-earlier-segment, %zu xor-mask>=32 lanes\n",
                    oldUpDiff, oldDownDiff, oldXorEarlier, oldXorTrunc);
        PXG_CHECK_MSG(oldUpDiff > 0 && oldDownDiff > 0 && oldXorEarlier > 0 && oldXorTrunc > 0,
                      "the script separates the PTX shfl.sync rule from the pre-P9.89 clamp in all four classes");
        PXG_CHECK_MSG(mm == 0, "all shuffle outputs bitwise-match the CUDA model (%zu bad)", mm);
        dev->destroyBuffer(bOut);
    }

    // ---- wlBallot ----
    {
        PxgBufferHandle bOut = createBufferFrom(dev, outInit);
        Kernel k("npWarpLib.spv", "wlBallot", { 16, 4, 4, 16 }, sizeof(P));
        k.dispatch(T, { bScript, bVals, bPred, bOut }, prm);
        dev->flush();
        const auto g = downloadBuffer<uint32_t>(dev, bOut, outInit.size());

        size_t mm = 0;
        for (uint32_t t = 0; t < T; ++t)
        {
            const uint32_t mask = script[4 * t + 0];
            const uint32_t width = 1u << script[4 * t + 1];
            for (uint32_t tile = 0; tile < 2; ++tile)
            {
                const float* tv = &vals[64 * t + 32 * tile];
                const uint32_t* tp = &pred[64 * t + 32 * tile];

                uint32_t ballot = 0;
                for (uint32_t i = 0; i < 32; ++i)
                    if (((mask >> i) & 1u) && tp[i])
                        ballot |= 1u << i;

                uint32_t scanEx[32], scanIn[32], redAddU[32], redOr[32];
                float redAddF[32], redMin[32], redMax[32];
                for (uint32_t i = 0; i < 32; ++i)
                {
                    scanEx[i] = scanIn[i] = redAddU[i] = f2u(tv[i]) & 0xffffu;
                    redOr[i] = f2u(tv[i]);
                    redAddF[i] = redMin[i] = redMax[i] = tv[i];
                }
                const std::vector<uint32_t> own(scanEx, scanEx + 32);
                mScanAddInclusive(scanEx, 32);
                mScanAddInclusive(scanIn, width);
                mButterfly(redAddU, width, [](uint32_t a, uint32_t b) { return a + b; });
                mButterfly(redOr, width, [](uint32_t a, uint32_t b) { return a | b; });
                mButterfly(redAddF, width, [](float a, float b) { return a + b; });
                mButterfly(redMin, width, [](float a, float b) { return a < b ? a : b; });
                mButterfly(redMax, width, [](float a, float b) { return a > b ? a : b; });

                uint32_t winnerBallot = 0;
                for (uint32_t i = 0; i < 32; ++i)
                    if (((mask >> i) & 1u) && redMax[i] == tv[i])
                        winnerBallot |= 1u << i;
                const uint32_t winner = winnerBallot == 0 ? 0xffffffffu
                                                          : uint32_t(__builtin_ctz(winnerBallot));

                for (uint32_t lane = 0; lane < 32; ++lane)
                {
                    const uint32_t tid = 32 * tile + lane;
                    const uint32_t* o = &g[(64 * t + tid) * 12];
                    const uint32_t e[12] = {
                        ballot, popc(mask & ((1u << lane) - 1u)), popc(ballot), scanEx[lane] - own[lane],
                        scanIn[lane], redAddU[lane], redOr[lane], winner,
                        f2u(redAddF[lane]), f2u(redMin[lane]), f2u(redMax[lane]), f2u(redMax[lane])
                    };
                    for (int c = 0; c < 12; ++c)
                        if (o[c] != e[c])
                        {
                            if (mm < 5)
                                std::printf("  BT MM t%u tid%u c%d: got %08x want %08x\n", t, tid, c, o[c], e[c]);
                            ++mm;
                        }
                }
            }
        }
        PXG_CHECK_MSG(mm == 0, "all ballot/scan/reduce outputs bitwise-match the CUDA model (%zu bad)", mm);
        dev->destroyBuffer(bOut);
    }

    dev->destroyBuffer(bScript);
    dev->destroyBuffer(bVals);
    dev->destroyBuffer(bPred);
}

}  // namespace
