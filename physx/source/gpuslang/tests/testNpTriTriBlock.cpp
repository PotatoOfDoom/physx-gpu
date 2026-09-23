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
// (convexMeshMidphase.cu:879-902).
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
#include "pxgCrsRef.h"
#include "pxgTriTriRef.h"

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
using namespace pxgtest::crsref;   // FLT_MAX_LOCAL, V3/V4, refWrapper2 -- P9.83b-5 (14/n)
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

// minIndex (nputils.cuh:151-164) is tritriref::refMinIndex (pxgTriTriRef.h) since
// P9.83b-5 (15/n): this file's own copy and testNpTriTriReduce.cpp's were the same
// function twice, and they are one now.
static uint32_t minIndexRef(const float* v, uint32_t mask, float& minV)
{
    return pxgtest::tritriref::refMinIndex(v, mask, minV);
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

// ---------------------------------------------------------------------------
// P9.83b-5 (14/n): ttbWarpReduceSerial -- contactReduce2 at :810 and :931, one
// warp at a time through a 32-slot stage and a per-warp result slot.
// ---------------------------------------------------------------------------
// THE REDUCTION IS NOT WHAT THIS TESTS.  crsContactReduce2 is certified by
// gpuslang.npCrsWrapper2 against the same oracle this uses (pxgCrsRef.h).  What
// is new is the plumbing around it, and every way it can go wrong is a MIXUP:
// warp w's lanes staged as warp w', warp w's mask read from another slot, the
// answer reaching lane 0 but not lanes 1-31, the second call reading the first
// call's leftovers.  So the case set is built to make mixups VISIBLE:
//
//   * every warp gets its OWN point cloud and its own mask, so swapping two
//     warps' data changes both answers;
//   * most masks carry more than six bits, so the reduction actually runs and the
//     answer is not simply the input mask handed back;
//   * the second call's masks differ from the first call's, and the fixture
//     asserts that the two ANSWERS differ for most warps -- if they did not, a
//     second call that returned the first call's slot would pass;
//   * every one of the 1024 threads reports, so a broadcast that reached one lane
//     per warp fails.
namespace
{
struct TtbRedCaseH
{
    V4 pts[1024];
    uint32_t masks[32];
    uint32_t masks2[32];
    V4 normalBias;
};
static_assert(sizeof(TtbRedCaseH) == 16656, "reduce case stride");
}

PXG_TEST(npTriTriBlock_warpReduce)
{
    std::mt19937 rng(77002);
    const uint32_t N = 48;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 100000) / 100000.f); };

    std::vector<TtbRedCaseH> in(N);
    std::memset(in.data(), 0, in.size() * sizeof(TtbRedCaseH));
    std::vector<uint32_t> want1(size_t(N) * 32), want2(size_t(N) * 32);

    size_t reductionsRun = 0, earlyOut = 0, emptyMask = 0, callsDiffer = 0, answerNotInput = 0;
    size_t dupePlants = 0, dupeMatters = 0;

    // The same reduction with TDoDupeTest = true -- NOT what the engine runs; it
    // exists only to count the warps on which the plant above makes the two
    // configurations disagree.
    auto refWrapper2DupeOn = [](const V3* pt, const float* sep, V3 nn, uint32_t allMask, float bias) {
        float adj[32], initial[32];
        for (uint32_t j = 0; j < 32; ++j)
        {
            const V3 p = sub(pt[j], scale3(nn, dot3(pt[j], nn)));
            adj[j] = -sep[j];
            initial[j] = mag3(p);
        }
        return reduceShared(pt, sep, nn, allMask, bias, adj, initial, 6, true, true);
    };

    auto pickMask = [&](uint32_t k) -> uint32_t {
        switch (k % 8)
        {
        case 0: return 0u;                                          // an empty warp
        case 1: return 0xFFFFFFFFu;                                 // every lane
        case 2: return uint32_t(rng()) & 0x0000003Fu;               // <= 6 bits: early-out
        default: return uint32_t(rng()) | uint32_t(rng());          // dense: reduction runs
        }
    };

    for (uint32_t i = 0; i < N; ++i)
    {
        TtbRedCaseH& c = in[i];

        V3 n{ rf(-1, 1), rf(-1, 1), rf(-1, 1) };
        const float nl = mag3(n);
        n = scale3(n, nl > 1e-4f ? 1.0f / nl : 1.0f);
        const float bias = rf(1e-5f, 5e-3f);
        c.normalBias = V4{ n.x, n.y, n.z, bias };

        for (uint32_t w = 0; w < 32; ++w)
        {
            // Each warp its own cloud: a different centre per warp.
            const V3 centre{ rf(-6, 6), rf(-6, 6), rf(-6, 6) };
            for (uint32_t l = 0; l < 32; ++l)
                c.pts[w * 32 + l] = V4{ centre.x + rf(-2, 2), centre.y + rf(-2, 2),
                                        centre.z + rf(-2, 2), rf(-0.4f, 0.25f) };
            c.masks[w] = pickMask(rng());
            c.masks2[w] = pickMask(rng());

            // PLANT (added after the 14/n sweep, where config-dupe-on SURVIVED): two
            // masked lanes at the SAME position.  :810 and :931 instantiate
            // contactReduce2 with TDoDupeTest = false, so both copies stay
            // candidates.  A port with the test on drops one of them
            // (contactReduction.cuh:65-79: magnitudeSquared < 1e-8, the shallower
            // goes; on equal separation the LOWER lane, since i > threadIndexInWarp
            // marks the lane being tested).  Random clouds never put two points
            // 1e-4 apart, so without this the two configurations agree on every
            // case and the mutant's survival says nothing.
            if ((w & 1u) == 1u && popcount32(c.masks[w]) >= 2u)
            {
                uint32_t m = c.masks[w];
                uint32_t a = 0;
                while (((m >> a) & 1u) == 0u) ++a;
                m &= m - 1u;
                uint32_t b = 0;
                while (((m >> b) & 1u) == 0u) ++b;
                const V4 pa = c.pts[w * 32 + a];
                V4& pb = c.pts[w * 32 + b];
                pb.x = pa.x;
                pb.y = pa.y;
                pb.z = pa.z;
                if (w % 3u == 0u)
                    pb.w = pa.w;                                    // the tie-break arm
                ++dupePlants;
            }
        }

        for (uint32_t w = 0; w < 32; ++w)
        {
            V3 pt[32];
            float sp[32];
            for (uint32_t l = 0; l < 32; ++l)
            {
                const V4& q = c.pts[w * 32 + l];
                pt[l] = V3{ q.x, q.y, q.z };
                sp[l] = q.w;
            }
            const uint32_t a = refWrapper2(pt, sp, n, c.masks[w], bias);
            const uint32_t b = refWrapper2(pt, sp, n, c.masks2[w], bias);
            want1[size_t(i) * 32 + w] = a;
            want2[size_t(i) * 32 + w] = b;

            if (popcount32(c.masks[w]) > 6) ++reductionsRun; else if (c.masks[w]) ++earlyOut; else ++emptyMask;
            if (a != b) ++callsDiffer;
            if (a != c.masks[w]) ++answerNotInput;
            if (refWrapper2DupeOn(pt, sp, n, c.masks[w], bias) != a) ++dupeMatters;
        }
    }

    const size_t warps = size_t(N) * 32;
    PXG_CHECK_MSG(reductionsRun > warps / 2,
                  "npTriTriBlock_warpReduce: only %zu of %zu first-call masks exceed six "
                  "bits -- below that the reduction hands the mask back unchanged and "
                  "the staging is not exercised", reductionsRun, warps);
    PXG_CHECK_MSG(earlyOut > warps / 20 && emptyMask > warps / 20,
                  "npTriTriBlock_warpReduce: %zu early-out and %zu empty masks; both "
                  "ends of the mask are real inputs", earlyOut, emptyMask);
    PXG_CHECK_MSG(callsDiffer > warps / 2,
                  "npTriTriBlock_warpReduce: the two calls' answers differ for only %zu "
                  "of %zu warps -- a second call that returned the first call's slot "
                  "would pass", callsDiffer, warps);
    PXG_CHECK_MSG(answerNotInput > warps / 3,
                  "npTriTriBlock_warpReduce: only %zu answers differ from their input "
                  "mask -- a reduce that returned its input would pass", answerNotInput);
    PXG_CHECK_MSG(dupeMatters > warps / 20,
                  "npTriTriBlock_warpReduce: the dupe test changes the answer on only %zu "
                  "of %zu warps (%zu coincident-pair plants) -- a port with TDoDupeTest "
                  "on would pass", dupeMatters, warps, dupePlants);

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<uint32_t> outInit(size_t(N) * 2048, 0xDEADBEEFu);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    TtbPushH push{};
    push.cases = dev->getBufferDeviceAddress(bIn);
    push.out = dev->getBufferDeviceAddress(bOut);
    push.numCases = N;

    Kernel k("npTriTriBlock.spv", "ttbReduceProbe", {}, sizeof(TtbPushH));
    k.dispatch(N, {}, push, 1u, 1u);
    dev->flush();

    const auto got = downloadBuffer<uint32_t>(dev, bOut, outInit.size());

    size_t mm1 = 0, mm2 = 0, laneOnly = 0;
    for (uint32_t i = 0; i < N; ++i)
        for (uint32_t t = 0; t < 1024; ++t)
        {
            const uint32_t w = t >> 5;
            const uint32_t g1 = got[size_t(i) * 2048 + t];
            const uint32_t g2 = got[size_t(i) * 2048 + 1024 + t];
            const uint32_t e1 = want1[size_t(i) * 32 + w];
            const uint32_t e2 = want2[size_t(i) * 32 + w];
            if (g1 != e1)
            {
                if (mm1 < 6)
                    std::printf("  case %u thread %u (warp %u): call 1 got 0x%08x want 0x%08x\n",
                                i, t, w, g1, e1);
                ++mm1;
                if ((t & 31u) != 0u && got[size_t(i) * 2048 + (w << 5)] == e1) ++laneOnly;
            }
            if (g2 != e2)
            {
                if (mm2 < 6)
                    std::printf("  case %u thread %u (warp %u): call 2 got 0x%08x want 0x%08x\n",
                                i, t, w, g2, e2);
                ++mm2;
            }
        }

    PXG_CHECK_MSG(mm1 == 0, "npTriTriBlock_warpReduce: %zu call-1 mismatches (%zu of them in a "
                            "lane whose warp's lane 0 was right -- a broadcast gap)", mm1, laneOnly);
    PXG_CHECK_MSG(mm2 == 0, "npTriTriBlock_warpReduce: %zu call-2 mismatches", mm2);

    std::printf("  npTriTriBlock_warpReduce: %u blocks, %zu warps per call; %zu reductions run, "
                "%zu early-out, %zu empty; the two calls differ on %zu warps, %zu answers "
                "differ from their input; %zu coincident-pair plants, dupe test decisive "
                "on %zu warps\n",
                N, warps, reductionsRun, earlyOut, emptyMask, callsDiffer, answerNotInput,
                dupePlants, dupeMatters);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

// ---------------------------------------------------------------------------
// P9.83b-5 (15/n): the assignment -- tryToAssign over the existing patches, then
// createNewPatches -- as the kernel runs it each iteration (:1207-1220).
// ---------------------------------------------------------------------------
// The oracle is pxgTriTriRef.h's block mirror, written from the CUDA.  The case
// set is built from the 15/n review's fixture plan; every family names what it is
// there to kill, because a mutant that survives without its input is a coverage
// gap, not an equivalence.
//
// FMA PARITY, by construction rather than by luck.  llvmpipe fuses dot() and the
// unstamped reduce; the mirror is built with -ffp-contract=off.  So every NORMAL
// here has exactly one nonzero component: every projection is then one product,
// which rounds the same fused or not.  Positions sit on a 2^-4 grid within +-8
// and separations on a 2^-10 grid, so with an axis normal every projection,
// difference, cross and dot inside the reduce is exact.  Non-unit axis normals
// (0.9995f, 0.85f, 0.9375f ...) appear only where the reduce they feed takes the
// <= 6-point early-out, which makes no float decision.
//
// OFF-ENGINE INPUTS ARE MARKED.  In the engine remainingContacts equals the live
// candidate count and a non-candidate has a zero normal (:753-766, :1146-1152,
// :1202).  Two families break the first on purpose, because the gates at :1213 and
// :872 and the decrement at :815 are invisible otherwise.
namespace
{
using namespace pxgtest::tritriref;

struct AsgCaseH
{
    V4 posSep[1024];
    uint32_t normalCand[1024][4];
    uint32_t rec[129][32];          // 128-byte records
    uint32_t numPatchesToKeep;
    int32_t generatedThisIter;
    float clusterBias;
    uint32_t pad;
};
static_assert(sizeof(AsgCaseH) == 49296, "assign case stride");

static uint32_t fbits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float bitsf(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

static void packRecord(uint32_t* r, const RefPatch& p)
{
    for (uint32_t k = 0; k < 6; ++k)
    {
        r[k * 4 + 0] = fbits(p.keepPoints[k].x);
        r[k * 4 + 1] = fbits(p.keepPoints[k].y);
        r[k * 4 + 2] = fbits(p.keepPoints[k].z);
        r[k * 4 + 3] = fbits(p.keepPoints[k].w);
    }
    r[24] = fbits(p.normal.x);
    r[25] = fbits(p.normal.y);
    r[26] = fbits(p.normal.z);
    r[27] = fbits(p.deepest);
    r[28] = p.pointCount;
    r[29] = r[30] = r[31] = 0;
}

static RefPatch unpackRecord(const uint32_t* r)
{
    RefPatch p;
    for (uint32_t k = 0; k < 6; ++k)
        p.keepPoints[k] = { bitsf(r[k * 4]), bitsf(r[k * 4 + 1]), bitsf(r[k * 4 + 2]), bitsf(r[k * 4 + 3]) };
    p.normal = { bitsf(r[24]), bitsf(r[25]), bitsf(r[26]) };
    p.deepest = bitsf(r[27]);
    p.pointCount = r[28];
    return p;
}

// One case under construction: contacts, patches, header -- and the families
// the builder planted, for the coverage assertions.
struct AsgBuild
{
    RefContact c[1024];
    RefPatch patches[129];
    uint32_t numPatchesToKeep = 0;
    int32_t generatedThisIter = 0;
    bool consistent = true;         // generatedThisIter == live candidate count
    float clusterBias = 1.0f / 1024.0f;
};

} // namespace

PXG_TEST(npTriTriBlock_assign)
{
    std::mt19937 rng(77015);
    auto gridPos = [&]() { return float(int(rng() % 257u) - 128) / 16.0f; };       // 2^-4, +-8
    auto gridSep = [&]() { return float(int(rng() % 769u) - 512) / 1024.0f; };     // 2^-10
    const V3 AXES[6] = { { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 } };

    // Poison for everything the kernel must not have written.
    RefPatch poison;
    for (uint32_t k = 0; k < 6; ++k) poison.keepPoints[k] = { bitsf(0x7FC0DEADu), -777.0f, 555.0f, bitsf(0xFFBADBADu) };
    poison.normal = { 3.0f, -5.0f, 7.0f };
    poison.deepest = -12345.0f;
    // A VALID count (<= 6): a mutant that reads a record past numPatchesToKeep must
    // do something defined with it, or its kill is undefined behaviour and proves
    // nothing -- the first sweep's `drv-p-le` crashed on keepPoints[tid] with a count
    // of 0xDEAD, which is exactly that.
    poison.pointCount = 5u;

    auto freshCase = [&](AsgBuild& b) {
        for (uint32_t t = 0; t < 1024; ++t)
        {
            b.c[t].pos = { gridPos(), gridPos(), gridPos() };
            b.c[t].separation = FLT_MAX_LOCAL;
            b.c[t].normal = { 0, 0, 0 };
            b.c[t].candidate = false;
        }
        for (uint32_t r = 0; r < 129; ++r) b.patches[r] = poison;
        b.numPatchesToKeep = 0;
        b.generatedThisIter = 0;
        b.consistent = true;
        b.clusterBias = float(1 + rng() % 8u) / 1024.0f;
    };
    auto makeCandidate = [&](RefContact& c, V3 n) {
        c.candidate = true;
        c.separation = gridSep();
        c.normal = n;
    };
    auto existingPatch = [&](AsgBuild& b, uint32_t r, V3 n, uint32_t count) {
        RefPatch& p = b.patches[r];
        p.normal = n;
        p.pointCount = count;
        float deepest = FLT_MAX_LOCAL;
        for (uint32_t k = 0; k < 6; ++k)
        {
            p.keepPoints[k] = { gridPos(), gridPos(), gridPos(), gridSep() };
            if (k < count && p.keepPoints[k].w < deepest) deepest = p.keepPoints[k].w;
        }
        p.deepest = deepest;
    };
    auto liveCount = [](const AsgBuild& b) {
        int32_t n = 0;
        for (uint32_t t = 0; t < 1024; ++t) n += b.c[t].candidate ? 1 : 0;
        return n;
    };

    std::vector<AsgBuild> cases;
    std::vector<const char*> family;

    // ---- A: generic, engine-consistent ----------------------------------------
    for (uint32_t i = 0; i < 16; ++i)
    {
        AsgBuild b;
        freshCase(b);
        b.numPatchesToKeep = rng() % 7u;
        for (uint32_t r = 0; r < b.numPatchesToKeep; ++r)
            existingPatch(b, r, AXES[rng() % 6u], 1 + rng() % 6u);
        const uint32_t density = 20 + rng() % 70u;          // percent of threads that are candidates
        for (uint32_t t = 0; t < 1024; ++t)
            if (rng() % 100u < density)
                makeCandidate(b.c[t], AXES[rng() % 6u]);
        b.generatedThisIter = liveCount(b);
        cases.push_back(b);
        family.push_back("generic");
    }

    // ---- B: OFF-ENGINE remaining < live (kills the :815 decrement, the accept /
    // accept2 swap at :812, and the > 0 gates at :1213 / :872) -------------------
    for (uint32_t i = 0; i < 4; ++i)
    {
        AsgBuild b;
        freshCase(b);
        b.numPatchesToKeep = 2;
        existingPatch(b, 0, { 1, 0, 0 }, 3);
        existingPatch(b, 1, { 0, 1, 0 }, 2);
        int32_t acc2 = 0;
        for (uint32_t t = 0; t < 1024; ++t)
        {
            const uint32_t k = rng() % 10u;
            if (k == 0) { makeCandidate(b.c[t], { 1, 0, 0 }); ++acc2; }            // patch 0, strict accept
            else if (k == 1) { makeCandidate(b.c[t], { 0.9f, 0, 0 }); ++acc2; }    // patch 0, accept2 only
            else if (k == 2) makeCandidate(b.c[t], { 0, 1, 0 });                   // would join patch 1
            else if (k == 3) makeCandidate(b.c[t], { 0, 0, 1 });                   // would seed a new patch
        }
        b.generatedThisIter = (i & 1u) ? 0 : acc2;
        b.consistent = false;
        cases.push_back(b);
        family.push_back(i & 1u ? "offengine-zero" : "offengine-patch0-only");
    }

    // ---- C: the zero-normal seed (upstream candidate B), refilling to the cap -----
    for (uint32_t start : { 0u, 125u, 127u })
    {
        AsgBuild b;
        freshCase(b);
        b.numPatchesToKeep = start;
        for (uint32_t r = 0; r < start; ++r)
            existingPatch(b, r, AXES[r % 6u], 1 + r % 6u);
        for (uint32_t t = 0; t < 1024; t += 7)
            makeCandidate(b.c[t], AXES[rng() % 6u]);
        b.c[513].candidate = true;
        b.c[513].normal = { 0, 0, 0 };
        b.c[513].separation = -0.75f;                 // the deepest: it seeds
        b.generatedThisIter = liveCount(b);
        cases.push_back(b);
        family.push_back("zero-normal-seed");
    }

    // ---- D: a non-hanging kill for dropping `|| myNormal` (OFF-ENGINE normal) -------
    {
        AsgBuild b;
        freshCase(b);
        for (uint32_t t = 0; t < 1024; t += 5)
            makeCandidate(b.c[t], { 1, 0, 0 });
        b.c[300].normal = { 0.9375f, 0, 0 };          // |n|^2 = 0.87890625: accept2 for all, accept for none
        b.c[300].separation = -0.9f;
        b.generatedThisIter = liveCount(b);
        cases.push_back(b);
        family.push_back("seed-only-by-myNormal");
    }

    // ---- E: the thresholds, exactly on them ------------------------------------------
    {
        AsgBuild b;   // tryAssign: the ONLY near lane is exactly 0.9995f -> no anyKeep
        freshCase(b);
        b.numPatchesToKeep = 1;
        existingPatch(b, 0, { 1, 0, 0 }, 4);
        makeCandidate(b.c[77], { 0.9995f, 0, 0 });
        for (uint32_t t = 128; t < 1024; t += 9) makeCandidate(b.c[t], { 0, 0, 1 });
        b.generatedThisIter = liveCount(b);
        cases.push_back(b);
        family.push_back("exact-eps3-only-near");
    }
    {
        AsgBuild b;   // createNewPatches: <= 6 accepts, one exactly 0.9995f and one exactly 0.85f
        freshCase(b);
        makeCandidate(b.c[10], { 1, 0, 0 });  b.c[10].separation = -0.5f;       // the seed
        makeCandidate(b.c[40], { 0.9995f, 0, 0 });
        makeCandidate(b.c[41], { 0.9993f, 0, 0 });                               // inside (0.999, 0.9995]
        makeCandidate(b.c[700], { 0.85f, 0, 0 });
        makeCandidate(b.c[701], { 0.8501f, 0, 0 });
        makeCandidate(b.c[900], { 0, 1, 0 });
        b.generatedThisIter = liveCount(b);
        cases.push_back(b);
        family.push_back("exact-thresholds-new");
    }
    {
        AsgBuild b;   // accept2 somewhere, accept NOWHERE: no anyKeep, nothing retired
        freshCase(b);
        b.numPatchesToKeep = 1;
        existingPatch(b, 0, { 0, 0, -1 }, 5);
        for (uint32_t t = 0; t < 1024; t += 3) makeCandidate(b.c[t], { 0, 0, -0.95f });
        b.generatedThisIter = liveCount(b);
        cases.push_back(b);
        family.push_back("accept2-without-accept");
    }

    {
        AsgBuild b;   // tryAssign WITH anyKeep and a lane at exactly 0.85f: counted and
        freshCase(b); // retired only by a `>=` -- the first sweep's ta-accept2-ge survived
        b.numPatchesToKeep = 1;   // without this, since every 0.85 lane sat in createNewPatches
        existingPatch(b, 0, { 1, 0, 0 }, 3);
        for (uint32_t t = 0; t < 1024; t += 11) makeCandidate(b.c[t], { 1, 0, 0 });
        makeCandidate(b.c[333], { 0.85f, 0, 0 });
        makeCandidate(b.c[334], { 0.9f, 0, 0 });
        b.generatedThisIter = liveCount(b);
        cases.push_back(b);
        family.push_back("exact-085-in-tryassign");
    }
    {
        AsgBuild b;   // anyKeep raised for patch 0, NOT for patch 1, whose only lanes are
        freshCase(b); // in the accept2 band: a missing :853-854 reset would carry patch 0's
        b.numPatchesToKeep = 2;   // flag into patch 1, retire them and count them (ta-no-reset)
        existingPatch(b, 0, { 1, 0, 0 }, 2);
        existingPatch(b, 1, { 0, 0, 1 }, 4);
        for (uint32_t t = 0; t < 1024; t += 13) makeCandidate(b.c[t], { 1, 0, 0 });
        for (uint32_t t = 5; t < 1024; t += 17) makeCandidate(b.c[t], { 0, 0, 0.95f });
        b.generatedThisIter = liveCount(b);
        cases.push_back(b);
        family.push_back("anykeep-must-reset");
    }

    // ---- G: anyKeep raised by ONE warp only -- the first, then the last --------------
    for (uint32_t w : { 0u, 31u })
    {
        AsgBuild b;
        freshCase(b);
        b.numPatchesToKeep = 1;
        existingPatch(b, 0, { 0, 1, 0 }, 2);
        for (uint32_t t = 0; t < 1024; t += 4) makeCandidate(b.c[t], { 1, 0, 0 });
        makeCandidate(b.c[w * 32 + 17], { 0, 1, 0 });
        b.generatedThisIter = liveCount(b);
        cases.push_back(b);
        family.push_back(w == 0 ? "anykeep-warp0-only" : "anykeep-warp31-only");
    }

    // ---- H: capacity -- 6 kept + 6 emitters in each of 32 warps = 198 staged -----------
    {
        AsgBuild b;
        freshCase(b);
        b.numPatchesToKeep = 1;
        existingPatch(b, 0, { 0, 0, 1 }, 6);
        for (uint32_t w = 0; w < 32; ++w)
            for (uint32_t l = 0; l < 6; ++l)
            {
                RefContact& c = b.c[w * 32 + l * 5];
                makeCandidate(c, { 0, 0, 1 });
                c.separation = float(int(w * 6 + l)) / 1024.0f - 0.1f;
            }
        b.c[31 * 32 + 25].separation = -0.95f;       // the global deepest: the LAST emitter of warp 31
        b.generatedThisIter = liveCount(b);
        cases.push_back(b);
        family.push_back("capacity-198");
    }

    // ---- I: the cap -- entering at 128, 127, 126 -------------------------------------
    for (uint32_t start : { 128u, 127u, 126u })
    {
        AsgBuild b;
        freshCase(b);
        b.numPatchesToKeep = start;
        for (uint32_t r = 0; r < start; ++r)
            existingPatch(b, r, { 0, -1, 0 }, 1 + r % 6u);
        // The guard record, 128, is accepted by every live contact: a `p <= n` or
        // `n <= 128` mutant that reads or writes it changes something visible.
        b.patches[128].normal = { 1, 0, 0 };
        for (uint32_t t = 0; t < 1024; t += 3)
            makeCandidate(b.c[t], (t % 2u) ? V3{ 1, 0, 0 } : V3{ 0, 0, 1 });
        b.generatedThisIter = liveCount(b);
        cases.push_back(b);
        family.push_back(start == 128 ? "cap-128" : start == 127 ? "cap-127" : "cap-126");
    }

    // ---- J/K: seed ties, and a winning warp that is not the last one with candidates --
    {
        AsgBuild b;   // two lanes of one warp tied at the minimum, different normals
        freshCase(b);
        for (uint32_t t = 0; t < 1024; t += 6) makeCandidate(b.c[t], { 0, 1, 0 });
        makeCandidate(b.c[200], { 1, 0, 0 });  b.c[200].separation = -0.8f;
        makeCandidate(b.c[205], { 0, 0, 1 });  b.c[205].separation = -0.8f;
        b.generatedThisIter = liveCount(b);
        cases.push_back(b);
        family.push_back("tie-in-warp");
    }
    {
        AsgBuild b;   // two warps tied; the LOWER warp must seed; later warps also have candidates
        freshCase(b);
        for (uint32_t t = 0; t < 1024; t += 6) makeCandidate(b.c[t], { 0, -1, 0 });
        makeCandidate(b.c[3 * 32 + 4], { 1, 0, 0 });   b.c[3 * 32 + 4].separation = -0.6f;
        makeCandidate(b.c[20 * 32 + 9], { 0, 0, -1 }); b.c[20 * 32 + 9].separation = -0.6f;
        b.generatedThisIter = liveCount(b);
        cases.push_back(b);
        family.push_back("tie-across-warps");
    }

    const uint32_t N = uint32_t(cases.size());

    // ---- the oracle, and the upload ------------------------------------------------
    std::vector<AsgCaseH> in(N);
    std::memset(in.data(), 0, in.size() * sizeof(AsgCaseH));
    std::vector<AsgBuild> want = cases;
    std::vector<int> wantRemaining(N);
    size_t sawRetired = 0, sawNewPatches = 0, sawTryAssignHit = 0, sawCapReached = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const AsgBuild& b = cases[i];
        AsgCaseH& h = in[i];
        for (uint32_t t = 0; t < 1024; ++t)
        {
            h.posSep[t] = { b.c[t].pos.x, b.c[t].pos.y, b.c[t].pos.z, b.c[t].separation };
            h.normalCand[t][0] = fbits(b.c[t].normal.x);
            h.normalCand[t][1] = fbits(b.c[t].normal.y);
            h.normalCand[t][2] = fbits(b.c[t].normal.z);
            h.normalCand[t][3] = b.c[t].candidate ? 1u : 0u;
        }
        for (uint32_t r = 0; r < 129; ++r) packRecord(h.rec[r], b.patches[r]);
        h.numPatchesToKeep = b.numPatchesToKeep;
        h.generatedThisIter = b.generatedThisIter;
        h.clusterBias = b.clusterBias;

        AsgBuild& w = want[i];
        const uint32_t n0 = w.numPatchesToKeep;
        wantRemaining[i] = refAssignContacts(w.c, w.patches, w.clusterBias, w.generatedThisIter, w.numPatchesToKeep);
        size_t retired = 0;
        for (uint32_t t = 0; t < 1024; ++t) retired += (b.c[t].candidate && !w.c[t].candidate) ? 1u : 0u;
        if (retired) ++sawRetired;
        if (w.numPatchesToKeep > n0) ++sawNewPatches;
        if (w.numPatchesToKeep == MAX_PATCHES) ++sawCapReached;
        for (uint32_t r = 0; r < n0; ++r)
            if (std::memcmp(&w.patches[r], &b.patches[r], sizeof(RefPatch)) != 0) { ++sawTryAssignHit; break; }
    }
    PXG_CHECK_MSG(sawRetired > N / 2 && sawNewPatches > N / 3 && sawTryAssignHit > N / 4 && sawCapReached >= 3,
                  "npTriTriBlock_assign: the case set does not exercise the machinery (retired %zu, new "
                  "patches %zu, existing patch changed %zu, cap reached %zu of %u)",
                  sawRetired, sawNewPatches, sawTryAssignHit, sawCapReached, N);

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<uint32_t> outInit(size_t(N) * (65664 / 4), 0xDEADBEEFu);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);
    TtbPushH push{};
    push.cases = dev->getBufferDeviceAddress(bIn);
    push.out = dev->getBufferDeviceAddress(bOut);
    push.numCases = N;
    Kernel k("npTriTriBlock.spv", "ttbAssignProbe", {}, sizeof(TtbPushH));
    k.dispatch(N, {}, push, 1u, 1u);
    dev->flush();
    const auto got = downloadBuffer<uint32_t>(dev, bOut, outInit.size());

    size_t mmRec = 0, mmContact = 0, mmCount = 0, mmRemaining = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const uint32_t* o = &got[size_t(i) * (65664 / 4)];
        const AsgBuild& w = want[i];
        for (uint32_t r = 0; r < 129; ++r)
        {
            uint32_t expect[32];
            packRecord(expect, w.patches[r]);
            if (std::memcmp(expect, o + r * 32, 29 * 4) != 0)   // words 29-31 are not stored
            {
                if (mmRec < 6)
                    std::printf("  case %u (%s) record %u: count got %u want %u, normal.x got %g want %g\n",
                                i, family[i], r, o[r * 32 + 28], expect[28], bitsf(o[r * 32 + 24]),
                                w.patches[r].normal.x);
                ++mmRec;
            }
        }
        const uint32_t* th = o + 16512 / 4;
        for (uint32_t t = 0; t < 1024; ++t)
        {
            const uint32_t* q = th + t * 12;
            const RefContact& c = w.c[t];
            const bool same = q[0] == fbits(c.pos.x) && q[1] == fbits(c.pos.y) && q[2] == fbits(c.pos.z) &&
                              q[3] == fbits(c.separation) && q[4] == fbits(c.normal.x) &&
                              q[5] == fbits(c.normal.y) && q[6] == fbits(c.normal.z) &&
                              q[7] == (c.candidate ? 1u : 0u);
            if (!same)
            {
                if (mmContact < 6)
                    std::printf("  case %u (%s) thread %u: contact state differs (candidate got %u want %u)\n",
                                i, family[i], t, q[7], c.candidate ? 1u : 0u);
                ++mmContact;
            }
            if (q[8] != w.numPatchesToKeep)
            {
                if (mmCount < 6)
                    std::printf("  case %u (%s) thread %u: numPatchesToKeep got %u want %u\n",
                                i, family[i], t, q[8], w.numPatchesToKeep);
                ++mmCount;
            }
            if (int32_t(q[9]) != wantRemaining[i])
            {
                if (mmRemaining < 6)
                    std::printf("  case %u (%s) thread %u: remaining got %d want %d\n",
                                i, family[i], t, int32_t(q[9]), wantRemaining[i]);
                ++mmRemaining;
            }
        }
    }
    PXG_CHECK_MSG(mmRec == 0, "npTriTriBlock_assign: %zu patch-record mismatches", mmRec);
    PXG_CHECK_MSG(mmContact == 0, "npTriTriBlock_assign: %zu contact-state mismatches", mmContact);
    PXG_CHECK_MSG(mmCount == 0, "npTriTriBlock_assign: %zu per-thread numPatchesToKeep mismatches", mmCount);
    PXG_CHECK_MSG(mmRemaining == 0, "npTriTriBlock_assign: %zu remainingContacts mismatches", mmRemaining);

    std::printf("  npTriTriBlock_assign: %u cases; retired in %zu, new patches in %zu, an existing patch "
                "changed in %zu, the cap reached in %zu\n",
                N, sawRetired, sawNewPatches, sawTryAssignHit, sawCapReached);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}
