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

// Verification of pxgSlangTriTriBlock against CPU mirrors of
// threadBlockCountAndBroadcast (reduction.cuh:576-589),
// threadBlockScanExclusive<32> (reduction.cuh:598-621), minIndex
// (nputils.cuh:151-164) and the cross-warp seed pick
// (convexMeshMidphase.cu:879-901).
//
// ONE WORKGROUP OF 1024 THREADS PER CASE.  Everything in P9.83b so far has been a
// serial body run on independent 64-thread lanes; these four are COLLECTIVES,
// where each thread's answer depends on the other 1023.  Running them on a
// smaller block would certify a shape the kernel never launches -- and would hide
// exactly the errors worth finding, because a per-warp bug is invisible when
// there is one warp.
//
// THE BOUNDARY INPUTS ARE CONSTRUCTED, per the rule P9.83b-5 (7-8/n) paid for
// four times.  For these bodies that means: an empty candidate mask, so
// lowestSetIndex(0) = 0xFFFFFFFF is a real input; a block where EVERY warp is
// empty, so the seed gate is what stops the loop; two warps sharing the block
// minimum, so the tie-break is exercised; and an UNMASKED lane whose separation
// equals its warp's masked minimum, which is the one input that distinguishes
// `ballot & mask` from `ballot`.

#include "pxgTestFramework.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

const float FLT_MAX_LOCAL = 3.402823466e+38f;
const uint32_t NO_INDEX = 0xFFFFFFFFu;

struct TtbCaseH
{
    uint32_t bitsA[32];   // accept2      -- the count predicate
    uint32_t bitsB[32];   // emits        -- the scan predicate
    uint32_t bitsC[32];   // candidate    -- the ballot the seed pick runs on
    float sep[1024];
};
static_assert(sizeof(TtbCaseH) == 4480, "case stride");

struct TtbPushH
{
    uint64_t cases, out;
    uint32_t numCases, pad;
};

const uint32_t OUT_STRIDE = 1152;

static uint32_t popc(uint32_t v) { uint32_t c = 0; while (v) { c += v & 1u; v >>= 1; } return c; }
static uint32_t bits32(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

// minIndex (nputils.cuh:151-164) over one warp's 32 lanes.  Written from the
// CUDA: the ballot compares against the lane's ORIGINAL value and the mask is
// applied AFTER it, so an unmasked lane equal to the masked minimum sets a bit
// that only the `& mask` removes.
static uint32_t minIndexRef(const float* v, uint32_t mask, float& minV)
{
    minV = FLT_MAX_LOCAL;
    for (uint32_t l = 0; l < 32; ++l)
        if (((mask >> l) & 1u) != 0u && v[l] < minV) minV = v[l];

    uint32_t ballot = 0;
    for (uint32_t l = 0; l < 32; ++l)
        if (v[l] == minV) ballot |= 1u << l;
    ballot &= mask;

    if (ballot == 0u) return NO_INDEX;
    uint32_t idx = 0;
    while (((ballot >> idx) & 1u) == 0u) ++idx;
    return idx;
}

} // namespace

PXG_TEST(npTriTriBlock_collectives)
{
    std::mt19937 rng(77001);
    const uint32_t N = 240;

    std::vector<TtbCaseH> in(N);
    std::memset(in.data(), 0, in.size() * sizeof(TtbCaseH));

    // A small discrete separation alphabet, so exact ties happen by construction
    // rather than by luck: a comparison decided by equality needs equal inputs.
    const float ALPHABET[8] = { -0.90f, -0.75f, -0.50f, -0.25f, -0.10f, 0.0f, 0.15f, 0.30f };

    struct RefH
    {
        uint32_t nbUsed, totalSum;
        uint32_t minIBlock, winnerCount, hasCandidates;
        float minSep;
        uint32_t warpMinI[32];
        float warpMinSep[32];
        uint32_t candMask[32];
        std::vector<uint32_t> offset;
    };
    std::vector<RefH> ref(N);

    size_t sawEmptyWarp = 0, sawAllEmpty = 0, sawBlockTie = 0, sawUnmaskedEqual = 0;
    size_t sawAllTrueA = 0, sawAllFalseA = 0, sawAllTrueB = 0, sawAllFalseB = 0;
    size_t sawSentinelMinI = 0, sawFltMaxCandidate = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        TtbCaseH& c = in[i];

        // ---- the predicates ----
        if (i % 30 == 0)      { for (int w = 0; w < 32; ++w) c.bitsA[w] = 0xFFFFFFFFu; ++sawAllTrueA; }
        else if (i % 30 == 1) { for (int w = 0; w < 32; ++w) c.bitsA[w] = 0u; ++sawAllFalseA; }
        else                  { for (int w = 0; w < 32; ++w) c.bitsA[w] = uint32_t(rng()); }

        if (i % 30 == 2)      { for (int w = 0; w < 32; ++w) c.bitsB[w] = 0xFFFFFFFFu; ++sawAllTrueB; }
        else if (i % 30 == 3) { for (int w = 0; w < 32; ++w) c.bitsB[w] = 0u; ++sawAllFalseB; }
        else
        {
            // Sparse, so the per-warp counts differ and the carry between warps is
            // a real number rather than a constant 32.
            for (int w = 0; w < 32; ++w)
                c.bitsB[w] = uint32_t(rng()) & uint32_t(rng());
        }

        // ---- the candidate mask, and the separations ----
        const bool allEmpty = (i % 30 == 4);
        if (allEmpty) ++sawAllEmpty;

        // PLANT 0: A WARP THAT HAS CANDIDATES AND STILL PUBLISHES FLT_MAX.
        // This is the input that tells `countersS[minIBlock] != 0` from "any warp
        // has candidates", and the first version of this fixture did not contain
        // it -- the separation alphabet is finite, so a warp with candidates
        // always published something below FLT_MAX and the two spellings agreed
        // on every case.  Construct it: warp 5 has candidates whose separations
        // are all exactly FLT_MAX, every other warp is empty.  Then all 32 warp
        // minima are FLT_MAX, the block minIndex breaks the tie on the LOWEST
        // index and picks warp 0, which has a zero count -- so the CUDA's gate is
        // FALSE while "any warp has candidates" is TRUE.
        const bool fltMaxCandidate = (i % 30 == 5);
        if (fltMaxCandidate)
        {
            for (int w = 0; w < 32; ++w)
            {
                c.bitsC[w] = (w == 5) ? 0x0000FF00u : 0u;
                for (int l = 0; l < 32; ++l)
                    c.sep[w * 32 + l] = (w == 5 && ((c.bitsC[w] >> l) & 1u))
                                        ? FLT_MAX_LOCAL : ALPHABET[rng() % 8];
            }
            ++sawFltMaxCandidate;
            for (int w = 0; w < 32; ++w) if (c.bitsC[w] == 0u) ++sawEmptyWarp;
        }
        else
        {
            for (int w = 0; w < 32; ++w)
            {
                uint32_t m;
                if (allEmpty) m = 0u;
                else if ((rng() % 8) == 0) m = 0u;      // one warp in eight has none
                else m = uint32_t(rng()) | (1u << (rng() % 32));
                c.bitsC[w] = m;
                if (m == 0u) ++sawEmptyWarp;

                for (int l = 0; l < 32; ++l)
                    c.sep[w * 32 + l] = ALPHABET[rng() % 8];
            }
        }

        // PLANT 1: an UNMASKED lane whose separation equals its warp's masked
        // minimum.  Warp 3, if it has both a masked and an unmasked lane.
        if (!fltMaxCandidate)
        {
            const int w = 3;
            uint32_t m = c.bitsC[w];
            if (m != 0u && m != 0xFFFFFFFFu)
            {
                int masked = -1, unmasked = -1;
                for (int l = 0; l < 32; ++l)
                {
                    if (((m >> l) & 1u) && masked < 0) masked = l;
                    if (!((m >> l) & 1u) && unmasked < 0) unmasked = l;
                }
                if (masked >= 0 && unmasked >= 0)
                {
                    // Make the masked lane the strict minimum of its warp, then
                    // give the unmasked lane the same value.
                    c.sep[w * 32 + masked] = -1.5f;
                    c.sep[w * 32 + unmasked] = -1.5f;
                    ++sawUnmaskedEqual;
                }
            }
        }

        // PLANT 2: TWO WARPS sharing the block minimum, so the second minIndex has
        // a tie and the lowest-index rule is the thing that decides.
        if (!allEmpty && !fltMaxCandidate && (i % 3 == 0))
        {
            int wa = -1, wb = -1;
            for (int w = 0; w < 32 && wb < 0; ++w)
                if (c.bitsC[w] != 0u) { if (wa < 0) wa = w; else wb = w; }
            if (wa >= 0 && wb >= 0)
            {
                for (int w : { wa, wb })
                {
                    const uint32_t m = c.bitsC[w];
                    for (int l = 0; l < 32; ++l)
                        if ((m >> l) & 1u) { c.sep[w * 32 + l] = -2.25f; break; }
                }
                ++sawBlockTie;
            }
        }

        // ---- the mirror ----
        RefH& r = ref[i];
        r.offset.assign(1024, 0);

        uint32_t nb = 0;
        for (int w = 0; w < 32; ++w) nb += popc(c.bitsA[w]);
        r.nbUsed = nb;

        uint32_t run = 0;
        for (int w = 0; w < 32; ++w)
            for (int l = 0; l < 32; ++l)
            {
                r.offset[w * 32 + l] = run;
                if ((c.bitsB[w] >> l) & 1u) ++run;
            }
        r.totalSum = run;

        for (int w = 0; w < 32; ++w)
        {
            r.candMask[w] = c.bitsC[w];
            float mv;
            r.warpMinI[w] = minIndexRef(&c.sep[w * 32], c.bitsC[w], mv);
            r.warpMinSep[w] = mv;
            if (r.warpMinI[w] == NO_INDEX) ++sawSentinelMinI;
        }

        float chosen;
        r.minIBlock = minIndexRef(r.warpMinSep, 0xFFFFFFFFu, chosen);
        r.minSep = chosen;
        r.winnerCount = popc(c.bitsC[r.minIBlock]);
        r.hasCandidates = r.winnerCount != 0u ? 1u : 0u;
    }

    PXG_CHECK_MSG(sawEmptyWarp > 200,
                  "npTriTriBlock: only %zu warps with an empty candidate mask -- that is "
                  "the one input where minIndex returns lowestSetIndex(0) = 0xFFFFFFFF",
                  sawEmptyWarp);
    PXG_CHECK_MSG(sawSentinelMinI > 200,
                  "npTriTriBlock: the 0xFFFFFFFF sentinel was produced only %zu times",
                  sawSentinelMinI);
    PXG_CHECK_MSG(sawAllEmpty >= 4,
                  "npTriTriBlock: %zu blocks where EVERY warp is empty -- without one, "
                  "the seed gate is never the thing that stops the loop", sawAllEmpty);
    PXG_CHECK_MSG(sawBlockTie > 40,
                  "npTriTriBlock: only %zu blocks with two warps on the block minimum; "
                  "the tie-break in the second minIndex needs a tie", sawBlockTie);
    PXG_CHECK_MSG(sawUnmaskedEqual > 100,
                  "npTriTriBlock: only %zu cases plant an unmasked lane equal to the "
                  "masked minimum -- that is the only input that tells `ballot & mask` "
                  "from `ballot`", sawUnmaskedEqual);
    PXG_CHECK_MSG(sawFltMaxCandidate >= 4,
                  "npTriTriBlock: %zu cases with a warp that HAS candidates and still "
                  "publishes FLT_MAX -- that is the only input that tells the CUDA's gate "
                  "(countersS[minIBlock] != 0) from \"any warp has candidates\"",
                  sawFltMaxCandidate);
    PXG_CHECK_MSG(sawAllTrueA >= 4 && sawAllFalseA >= 4 && sawAllTrueB >= 4 && sawAllFalseB >= 4,
                  "npTriTriBlock: predicate ends not covered (%zu/%zu all-true, "
                  "%zu/%zu all-false)", sawAllTrueA, sawAllTrueB, sawAllFalseA, sawAllFalseB);

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<uint32_t> outInit(size_t(N) * OUT_STRIDE, 0xDEADBEEFu);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    TtbPushH push{};
    push.cases = dev->getBufferDeviceAddress(bIn);
    push.out = dev->getBufferDeviceAddress(bOut);
    push.numCases = N;

    Kernel k("npTriTriBlock.spv", "ttbProbe", {}, sizeof(TtbPushH));
    k.dispatch(N, {}, push, 1u, 1u);
    dev->flush();

    const auto got = downloadBuffer<uint32_t>(dev, bOut, outInit.size());

    size_t mmCount = 0, mmTotal = 0, mmOffset = 0, mmSeed = 0, mmWarpMin = 0, mmMask = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const uint32_t* g = &got[size_t(i) * OUT_STRIDE];
        const RefH& r = ref[i];

        if (g[0] != r.nbUsed)
        {
            if (mmCount < 4) std::printf("  case %u: nbUsed got %u want %u\n", i, g[0], r.nbUsed);
            ++mmCount;
        }
        if (g[1] != r.totalSum)
        {
            if (mmTotal < 4) std::printf("  case %u: totalSum got %u want %u\n", i, g[1], r.totalSum);
            ++mmTotal;
        }
        if (g[2] != r.minIBlock || g[3] != bits32(r.minSep) ||
            g[4] != r.winnerCount || g[5] != r.hasCandidates)
        {
            if (mmSeed < 4)
                std::printf("  case %u: seed got (%u, %.9g, %u, %u) want (%u, %.9g, %u, %u)\n",
                            i, g[2], *(const float*)&g[3], g[4], g[5],
                            r.minIBlock, r.minSep, r.winnerCount, r.hasCandidates);
            ++mmSeed;
        }
        for (uint32_t w = 0; w < 32; ++w)
        {
            if (g[6 + w] != r.warpMinI[w] || g[38 + w] != bits32(r.warpMinSep[w]))
            {
                if (mmWarpMin < 6)
                    std::printf("  case %u warp %u: minI got 0x%08x want 0x%08x; "
                                "minSep got %.9g want %.9g\n", i, w, g[6 + w], r.warpMinI[w],
                                *(const float*)&g[38 + w], r.warpMinSep[w]);
                ++mmWarpMin;
            }
            if (g[70 + w] != r.candMask[w])
            {
                if (mmMask < 4)
                    std::printf("  case %u warp %u: candidateMask got 0x%08x want 0x%08x\n",
                                i, w, g[70 + w], r.candMask[w]);
                ++mmMask;
            }
        }
        for (uint32_t t = 0; t < 1024; ++t)
            if (g[102 + t] != r.offset[t])
            {
                if (mmOffset < 6)
                    std::printf("  case %u thread %u: offset got %u want %u\n",
                                i, t, g[102 + t], r.offset[t]);
                ++mmOffset;
            }
    }

    PXG_CHECK_MSG(mmCount == 0, "npTriTriBlock: %zu nbUsed mismatches", mmCount);
    PXG_CHECK_MSG(mmTotal == 0, "npTriTriBlock: %zu totalSum mismatches", mmTotal);
    PXG_CHECK_MSG(mmOffset == 0, "npTriTriBlock: %zu scan-offset mismatches", mmOffset);
    PXG_CHECK_MSG(mmWarpMin == 0, "npTriTriBlock: %zu per-warp minIndex mismatches", mmWarpMin);
    PXG_CHECK_MSG(mmMask == 0, "npTriTriBlock: %zu candidate-mask mismatches", mmMask);
    PXG_CHECK_MSG(mmSeed == 0, "npTriTriBlock: %zu seed-pick mismatches", mmSeed);

    std::printf("  npTriTriBlock: %u blocks of 1024 threads; %zu empty warps "
                "(%zu sentinel minI), %zu all-empty blocks, %zu block-minimum ties, "
                "%zu unmasked-equal plants, %zu FLT_MAX-candidate plants\n",
                N, sawEmptyWarp, sawSentinelMinI, sawAllEmpty, sawBlockTie, sawUnmaskedEqual,
                sawFltMaxCandidate);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}
