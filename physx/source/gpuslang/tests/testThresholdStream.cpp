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

// Verification of thresholdStream.slang against C++ references: the sort-key
// extraction + reorganize (a two-pass (nodeIndexA, nodeIndexB) sort) and the
// device-wide segmented force accumulation + exceeded-force detection.

#include "pxgTestFramework.h"
#include "pxgGpuStructs.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct ThresholdStreamElement
{
    uint64_t shapeInteraction;
    float normalForce;
    float threshold;
    uint64_t nodeIndexA;
    uint64_t nodeIndexB;
    float accumulatedForce;
    uint32_t pad;
};
static_assert(sizeof(ThresholdStreamElement) == 40, "threshold element");

struct ThresholdParams
{
    uint32_t nbThresholdElements;
    uint32_t nbPrevExceeded;
    float dt;
    float pad;
};

uint32_t idxOf(uint64_t node) { return uint32_t(node & 0xffffffffu); }
uint32_t f2uTs(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

// P9.63 -- THE FIXTURE MUST DISPATCH THE ENGINE'S GRID, AND THE ENGINE USES TWO.
// PxgSolverKernelIndices.h (PxgKernelGridDim) gives 32 groups to the four
// block-partitioned entries and 64 to the four plain grid-stride ones; this
// fixture used to dispatch 32 for all eight, which made the shader's stride
// constant self-consistent with the fixture and unobservable.  Keep these two
// names in step with the shader's NUM_BLOCKS / NUM_BLOCKS_WIDE.
static const uint32_t THRESH_GRID = 32;       // COMPUTE/OUTPUT_ACCUMULATED, COMPUTE/OUTPUT_EXCEEDEDFORCE
static const uint32_t THRESH_GRID_WIDE = 64;  // INITIALIZE_INPUT_AND_RANKS, REORGANIZE, WRITEOUT
static const uint32_t THRESH_BLOCK = 256;

// ---------------------------------------------------------------------------
// Test 1: sort-key extraction + reorganize
// ---------------------------------------------------------------------------
PXG_TEST(thresholdStream_sortKeysAndReorganize)
{
    std::mt19937 rng(5100);
    const uint32_t N = 733;

    std::vector<ThresholdStreamElement> stream(N);
    std::memset(stream.data(), 0, stream.size() * sizeof(ThresholdStreamElement));
    for (uint32_t i = 0; i < N; ++i)
    {
        // small node-id range so pairs repeat; y = 0 (non-articulation)
        stream[i].nodeIndexA = rng() % 40;
        stream[i].nodeIndexB = rng() % 40;
        stream[i].shapeInteraction = (uint64_t(rng()) << 32) | rng();
        stream[i].normalForce = float(rng() % 1000) * 0.01f;
        stream[i].threshold = float(rng() % 1000) * 0.01f;
        stream[i].accumulatedForce = -1.f; // must be preserved across reorganize (word 8 not copied)
        stream[i].pad = 0xdeadbeef;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t padCount = (4 - (N & 3)) & 3;

    std::vector<uint32_t> keysInit(N + padCount, 0), ranksInit(N + padCount, 0);
    std::vector<uint32_t> dummy(16, 0);

    PxgBufferHandle streamBuf = createBufferFrom(dev, stream);
    PxgBufferHandle keysBuf = createBufferFrom(dev, keysInit);
    PxgBufferHandle ranksBuf = createBufferFrom(dev, ranksInit);
    PxgBufferHandle dummyBuf = createBufferFrom(dev, dummy);

    ThresholdParams p = {};
    p.nbThresholdElements = N;
    p.dt = 1.f;

    // ---- initialRanksAndBodyIndexB ----
    Kernel kB("thresholdStream.spv", "initialRanksAndBodyIndexB", { 40, 4, 4 }, sizeof(ThresholdParams));
    kB.dispatch(THRESH_GRID_WIDE, { streamBuf, keysBuf, ranksBuf }, p);
    dev->flush();

    auto gKeysB = downloadBuffer<uint32_t>(dev, keysBuf, N + padCount);
    auto gRanksB = downloadBuffer<uint32_t>(dev, ranksBuf, N + padCount);
    size_t mism = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (gKeysB[i] != idxOf(stream[i].nodeIndexB)) ++mism;
        if (gRanksB[i] != i) ++mism;
    }
    for (uint32_t i = 0; i < padCount; ++i)
    {
        if (gKeysB[N + i] != 0xffffffffu) ++mism;
        if (gRanksB[N + i] != N + i) ++mism;
    }
    PXG_CHECK_MSG(mism == 0, "initialRanksAndBodyIndexB: %zu mismatches", mism);

    // ---- CPU B-sort (stable by nodeIndexB.index()), then initialRanksAndBodyIndexA ----
    std::vector<uint32_t> ranksAfterB(N);
    std::iota(ranksAfterB.begin(), ranksAfterB.end(), 0u);
    std::stable_sort(ranksAfterB.begin(), ranksAfterB.end(),
                     [&](uint32_t a, uint32_t b) { return idxOf(stream[a].nodeIndexB) < idxOf(stream[b].nodeIndexB); });

    std::vector<uint32_t> ranksBinput = ranksAfterB;
    ranksBinput.resize(N + padCount);
    for (uint32_t i = 0; i < padCount; ++i)
        ranksBinput[N + i] = N + i;
    PxgBufferHandle ranksB2 = createBufferFrom(dev, ranksBinput);

    Kernel kA("thresholdStream.spv", "initialRanksAndBodyIndexA", { 40, 4, 4 }, sizeof(ThresholdParams));
    kA.dispatch(THRESH_GRID_WIDE, { streamBuf, keysBuf, ranksB2 }, p);
    dev->flush();

    auto gKeysA = downloadBuffer<uint32_t>(dev, keysBuf, N + padCount);
    mism = 0;
    for (uint32_t i = 0; i < N; ++i)
        if (gKeysA[i] != idxOf(stream[ranksAfterB[i]].nodeIndexA)) ++mism;
    for (uint32_t i = 0; i < padCount; ++i)
        if (gKeysA[N + i] != 0xffffffffu) ++mism;
    PXG_CHECK_MSG(mism == 0, "initialRanksAndBodyIndexA: %zu mismatches", mism);

    // ---- CPU A-sort (stable) of the B-ordered ranks -> final (A,B) order ----
    std::vector<uint32_t> finalRanks = ranksAfterB;
    std::stable_sort(finalRanks.begin(), finalRanks.end(),
                     [&](uint32_t a, uint32_t b) { return idxOf(stream[a].nodeIndexA) < idxOf(stream[b].nodeIndexA); });

    std::vector<uint32_t> finalRanksInput = finalRanks;
    finalRanksInput.resize(N + padCount);
    PxgBufferHandle finalRanksBuf = createBufferFrom(dev, finalRanksInput);

    // reorganize: tmp holds the unsorted stream, dest (streamBuf) gets the sorted words 0..7
    PxgBufferHandle tmpBuf = createBufferFrom(dev, stream);
    std::vector<ThresholdStreamElement> destInit(N);
    std::memset(destInit.data(), 0, destInit.size() * sizeof(ThresholdStreamElement));
    for (auto& e : destInit) e.accumulatedForce = 12345.f; // sentinel: word 8 must NOT be overwritten
    PxgBufferHandle destBuf = createBufferFrom(dev, destInit);

    Kernel kR("thresholdStream.spv", "reorganizeThresholdElements", { 4, 4, 4, 40, 40 }, sizeof(ThresholdParams));
    // bindings 2,3,4 -> ranks, destWords, tmpWords (0,1 dummy)
    kR.dispatch(THRESH_GRID_WIDE, { dummyBuf, dummyBuf, finalRanksBuf, destBuf, tmpBuf }, p);
    dev->flush();

    auto gDest = downloadBuffer<ThresholdStreamElement>(dev, destBuf, N);
    mism = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const ThresholdStreamElement& r = stream[finalRanks[i]];
        const ThresholdStreamElement& g = gDest[i];
        if (g.shapeInteraction != r.shapeInteraction) ++mism;
        if (g.normalForce != r.normalForce) ++mism;
        if (g.threshold != r.threshold) ++mism;
        if (g.nodeIndexA != r.nodeIndexA) ++mism;
        if (g.nodeIndexB != r.nodeIndexB) ++mism;
        if (g.accumulatedForce != 12345.f) ++mism; // word 8 preserved (not part of the 8-word copy)
    }
    PXG_CHECK_MSG(mism == 0, "reorganize: %zu mismatches", mism);

    for (PxgBufferHandle h : { streamBuf, keysBuf, ranksBuf, dummyBuf, ranksB2, finalRanksBuf, tmpBuf, destBuf })
        dev->destroyBuffer(h);
}

// ---------------------------------------------------------------------------
// Test 2: segmented force accumulation + exceeded-force detection
// ---------------------------------------------------------------------------
PXG_TEST(thresholdStream_accumulateAndExceeded)
{
    for (uint32_t N : { 37u, 512u, 2001u })
    {
        std::mt19937 rng(5200 + N);
        const float dt = 0.5f;

        // build a stream already sorted by (nodeIndexA, nodeIndexB): walk pairs
        // in increasing order, each with a random run length; per-pair threshold
        // is set clearly above or below the segment force so the exceeded flag
        // is robust to parallel float-summation rounding
        std::vector<ThresholdStreamElement> stream;
        stream.reserve(N);
        uint32_t a = 0, b = 0;
        std::uniform_real_distribution<float> uf(0.5f, 1.5f);
        while (stream.size() < N)
        {
            // advance to a new pair
            b += 1 + rng() % 3;
            if (b > 100) { a += 1; b = rng() % 5; }
            const uint32_t runLen = std::min<uint32_t>(1 + rng() % 8, uint32_t(N - stream.size()));
            const bool exceed = (rng() & 1) != 0;
            for (uint32_t r = 0; r < runLen; ++r)
            {
                ThresholdStreamElement e = {};
                e.nodeIndexA = a;
                e.nodeIndexB = b;
                e.shapeInteraction = (uint64_t(a) << 20) | (uint64_t(b) << 4) | r;
                e.normalForce = uf(rng);
                // segment force ~ runLen * ~1.0; threshold*dt clearly below (exceed) or above
                e.threshold = exceed ? (0.001f) : (1.0e6f);
                e.accumulatedForce = 0.f;
                e.pad = 0;
                stream.push_back(e);
            }
        }
        stream.resize(N);

        PxgSlangComputeDevice* dev = testDevice();

        std::vector<float> accForce(N, 0.f), accForceBB(32, 0.f), accForceObjPairs(N + 1, 0.f);
        std::vector<uint32_t> writeIdx(N, 0), writeIdxBB(32, 0), writeable(N, 0);
        std::vector<ThresholdStreamElement> exceeded(N);
        std::memset(exceeded.data(), 0, exceeded.size() * sizeof(ThresholdStreamElement));
        std::vector<uint32_t> nbExceeded(1, 0), dummy(16, 0);

        PxgBufferHandle streamBuf = createBufferFrom(dev, stream);
        PxgBufferHandle accForceBuf = createBufferFrom(dev, accForce);
        PxgBufferHandle accForceBBBuf = createBufferFrom(dev, accForceBB);
        PxgBufferHandle writeIdxBuf = createBufferFrom(dev, writeIdx);
        PxgBufferHandle writeIdxBBBuf = createBufferFrom(dev, writeIdxBB);
        PxgBufferHandle writeableBuf = createBufferFrom(dev, writeable);
        PxgBufferHandle objPairsBuf = createBufferFrom(dev, accForceObjPairs);
        PxgBufferHandle exceededBuf = createBufferFrom(dev, exceeded);
        PxgBufferHandle nbExcBuf = createBufferFrom(dev, nbExceeded);
        PxgBufferHandle dummyBuf = createBufferFrom(dev, dummy);

        ThresholdParams p = {};
        p.nbThresholdElements = N;
        p.dt = dt;

        // computeAccumulate: bindings 0,5,6,7,8,9
        Kernel kAcc("thresholdStream.spv", "computeAccumulateThresholdStream",
                    { 40, 4, 4, 4, 4, 4, 4, 4, 4, 4 }, sizeof(ThresholdParams));
        kAcc.dispatch(THRESH_GRID, { streamBuf, dummyBuf, dummyBuf, dummyBuf, dummyBuf, accForceBuf, accForceBBBuf,
                            writeIdxBuf, writeIdxBBBuf, writeableBuf }, p);
        dev->barrier();
        // outputAccumulate: bindings 5,6,7,8
        Kernel kOut("thresholdStream.spv", "outputAccumulateThresholdStream",
                    { 4, 4, 4, 4, 4, 4, 4, 4, 4 }, sizeof(ThresholdParams));
        kOut.dispatch(THRESH_GRID, { dummyBuf, dummyBuf, dummyBuf, dummyBuf, dummyBuf, accForceBuf, accForceBBBuf,
                            writeIdxBuf, writeIdxBBBuf }, p);
        dev->barrier();
        // writeout: bindings 5,7,9,10
        Kernel kWrite("thresholdStream.spv", "writeoutAccumulatedForcePerObject",
                      { 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4 }, sizeof(ThresholdParams));
        kWrite.dispatch(THRESH_GRID_WIDE, { dummyBuf, dummyBuf, dummyBuf, dummyBuf, dummyBuf, accForceBuf, dummyBuf,
                              writeIdxBuf, dummyBuf, writeableBuf, objPairsBuf }, p);
        dev->flush();

        // ---- reference (stage 1) ----
        std::vector<float> refAcc(N, 0.f), refObjPairs(N + 1, 0.f);
        std::vector<uint32_t> refWriteIdx(N, 0), refWriteable(N, 0);
        float run = 0.f;
        uint32_t pairIdx = 0;
        for (uint32_t i = 0; i < N; ++i)
        {
            run += stream[i].normalForce;
            refAcc[i] = run;
            const bool isNew = (i + 1 >= N) ||
                               !(stream[i].nodeIndexA == stream[i + 1].nodeIndexA && stream[i].nodeIndexB == stream[i + 1].nodeIndexB);
            refWriteIdx[i] = pairIdx;
            refWriteable[i] = isNew ? 1u : 0u;
            if (isNew)
            {
                refObjPairs[pairIdx] = run; // inclusive prefix at the pair's last element
                ++pairIdx;
            }
        }

        auto gWriteIdx = downloadBuffer<uint32_t>(dev, writeIdxBuf, N);
        auto gWriteable = downloadBuffer<uint32_t>(dev, writeableBuf, N);
        auto gAcc = downloadBuffer<float>(dev, accForceBuf, N);
        auto gObj = downloadBuffer<float>(dev, objPairsBuf, N + 1);

        size_t mism = 0;
        for (uint32_t i = 0; i < N; ++i)
        {
            if (gWriteIdx[i] != refWriteIdx[i]) { if (mism < 4) PXG_CHECK_MSG(false, "N%u i%u writeIdx g%u r%u", N, i, gWriteIdx[i], refWriteIdx[i]); ++mism; }
            if (gWriteable[i] != refWriteable[i]) ++mism;
            if (!almostEqual(gAcc[i], refAcc[i], 1e-3f, 1e-2f)) ++mism;
        }
        for (uint32_t k = 0; k < pairIdx; ++k)
            if (!almostEqual(gObj[k], refObjPairs[k], 1e-3f, 1e-2f)) ++mism;
        PXG_CHECK_MSG(mism == 0, "N=%u: %zu accumulate mismatches", N, mism);

        // ---- exceeded stage ----
        // computeExceeded: bindings 0,7,8,9,10
        Kernel kExc("thresholdStream.spv", "computeExceededForceThresholdElementIndice",
                    { 40, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4 }, sizeof(ThresholdParams));
        kExc.dispatch(THRESH_GRID, { streamBuf, dummyBuf, dummyBuf, dummyBuf, dummyBuf, dummyBuf, dummyBuf, writeIdxBuf,
                            writeIdxBBBuf, writeableBuf, objPairsBuf }, p);
        dev->barrier();
        // outputExceeded: bindings 0,7,8,9,11,12
        Kernel kOutExc("thresholdStream.spv", "outputExceededForceThresholdElementIndice",
                       { 40, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 40, 4 }, sizeof(ThresholdParams));
        kOutExc.dispatch(THRESH_GRID, { streamBuf, dummyBuf, dummyBuf, dummyBuf, dummyBuf, dummyBuf, dummyBuf, writeIdxBuf,
                               writeIdxBBBuf, writeableBuf, dummyBuf, exceededBuf, nbExcBuf }, p);
        dev->flush();

        // ---- reference (stage 2) ----
        std::vector<ThresholdStreamElement> refExceeded;
        std::vector<float> refSegForce(N, 0.f);
        for (uint32_t i = 0; i < N; ++i)
        {
            const uint32_t wi = refWriteIdx[i];
            float seg = refObjPairs[wi];
            if (wi > 0)
                seg -= refObjPairs[wi - 1];
            refSegForce[i] = seg;
            if (seg > stream[i].threshold * dt)
                refExceeded.push_back(stream[i]);
        }

        auto gNb = downloadBuffer<uint32_t>(dev, nbExcBuf, 1)[0];
        auto gStream = downloadBuffer<ThresholdStreamElement>(dev, streamBuf, N);
        auto gExc = downloadBuffer<ThresholdStreamElement>(dev, exceededBuf, refExceeded.size() ? refExceeded.size() : 1);

        PXG_CHECK_MSG(gNb == refExceeded.size(), "N=%u: nbExceeded g%u r%zu", N, gNb, refExceeded.size());

        mism = 0;
        for (uint32_t i = 0; i < N; ++i) // per-element segment force written back onto the stream
            if (!almostEqual(gStream[i].accumulatedForce, refSegForce[i], 1e-3f, 1e-2f)) ++mism;
        PXG_CHECK_MSG(mism == 0, "N=%u: %zu segment-force mismatches", N, mism);

        mism = 0;
        for (size_t k = 0; k < refExceeded.size(); ++k)
        {
            if (gExc[k].nodeIndexA != refExceeded[k].nodeIndexA) ++mism;
            if (gExc[k].nodeIndexB != refExceeded[k].nodeIndexB) ++mism;
            if (gExc[k].shapeInteraction != refExceeded[k].shapeInteraction) ++mism;
            if (gExc[k].normalForce != refExceeded[k].normalForce) ++mism;
        }
        PXG_CHECK_MSG(mism == 0, "N=%u: %zu exceeded-element mismatches", N, mism);

        for (PxgBufferHandle h : { streamBuf, accForceBuf, accForceBBBuf, writeIdxBuf, writeIdxBBBuf, writeableBuf,
                                   objPairsBuf, exceededBuf, nbExcBuf, dummyBuf })
            dev->destroyBuffer(h);
    }
}

// ---------------------------------------------------------------------------
// P9.86f: the accumulated force BITWISE, against the CUDA's association
// ---------------------------------------------------------------------------
// computeAccumulateThresholdStream folds each iteration's block total into sBlockAccumulator as
// (accumulatedForce + forceVal) = ((incl - f7) + f7) (accumulateThresholdStream.cu:256-258), and
// outputAccumulateThresholdStream takes the between-block prefix as warpScan(val) - val, a 32-lane
// Hillis-Steele tree (:317, reduction.cuh:261-275).  The test above compares to tolerance against a
// running sum; this one mirrors the CUDA's order exactly and compares every word.  Twin 1 is the
// port's former `gsBlockForce += fInc`, twin 2 its former sequential between-block prefix; both
// must change at least one output of the directed stream (forces with arbitrary mantissas over 26
// blocks, three iterations each).
namespace
{
void tsCudaAccumulate(const std::vector<ThresholdStreamElement>& s, std::vector<float>& acc, std::vector<float>& bb, int twin)
{
    const uint32_t N = uint32_t(s.size());
    const uint32_t tbr = (N + THRESH_BLOCK - 1) / THRESH_BLOCK;
    const uint32_t nipb = (tbr + THRESH_GRID - 1) / THRESH_GRID;
    acc.assign(N, 0.f);
    bb.assign(THRESH_GRID, 0.f);
    for (uint32_t b = 0; b < THRESH_GRID; ++b)
    {
        float sBlock = 0.f;
        for (uint32_t it = 0; it < nipb; ++it)
        {
            float val[256], incl[256], accumVal[256], warpF[8], acc8[8];
            for (uint32_t t = 0; t < 256; ++t)
            {
                const uint32_t wi = t + it * 256 + 256 * b * nipb;
                val[t] = (wi < N) ? s[wi].normalForce : 0.f;
            }
            for (uint32_t w = 0; w < 8; ++w)
            {
                float x[32];
                for (uint32_t l = 0; l < 32; ++l) x[l] = val[w * 32 + l];
                for (uint32_t r = 1; r < 32; r <<= 1)
                {
                    float y[32];
                    for (uint32_t l = 0; l < 32; ++l) y[l] = (l >= r) ? x[l] + x[l - r] : x[l];
                    std::memcpy(x, y, sizeof(x));
                }
                for (uint32_t l = 0; l < 32; ++l) { incl[w * 32 + l] = x[l]; accumVal[w * 32 + l] = x[l] - val[w * 32 + l]; }
                warpF[w] = accumVal[w * 32 + 31] + val[w * 32 + 31];
            }
            const float prev = sBlock;
            float x8[8];
            std::memcpy(x8, warpF, sizeof(x8));
            for (uint32_t r = 1; r < 8; r <<= 1)
            {
                float y[8];
                for (uint32_t l = 0; l < 8; ++l) y[l] = (l >= r) ? x8[l] + x8[l - r] : x8[l];
                std::memcpy(x8, y, sizeof(x8));
            }
            for (uint32_t w = 0; w < 8; ++w) acc8[w] = x8[w] - warpF[w];
            sBlock += (twin == 1) ? x8[7] : (acc8[7] + warpF[7]);
            for (uint32_t t = 0; t < 256; ++t)
            {
                const uint32_t wi = t + it * 256 + 256 * b * nipb;
                if (wi < N) acc[wi] = val[t] + accumVal[t] + prev + acc8[t / 32];
            }
            (void)incl;
        }
        bb[b] = sBlock;
    }
    float res[32];
    if (twin == 2)
    {
        float run = 0.f;
        for (uint32_t i = 0; i < 32; ++i) { res[i] = run; run += bb[i]; }
    }
    else
    {
        float x[32];
        std::memcpy(x, bb.data(), sizeof(x));
        for (uint32_t r = 1; r < 32; r <<= 1)
        {
            float y[32];
            for (uint32_t l = 0; l < 32; ++l) y[l] = (l >= r) ? x[l] + x[l - r] : x[l];
            std::memcpy(x, y, sizeof(x));
        }
        for (uint32_t i = 0; i < 32; ++i) res[i] = x[i] - bb[i];
    }
    for (uint32_t b = 0; b < THRESH_GRID; ++b)
        for (uint32_t it = 0; it < nipb; ++it)
            for (uint32_t t = 0; t < 256; ++t)
            {
                const uint32_t wi = it * 256 + t + nipb * b * 256;
                if (wi < N) acc[wi] = acc[wi] + res[b];
            }
}
} // namespace

PXG_TEST(thresholdStream_accumulateBitExact)
{
    const uint32_t N = 20000; // 79 blocks of 256 -> 3 iterations per group, 27 groups non-empty
    std::mt19937 rng(8601);
    std::uniform_real_distribution<float> uf(0.1f, 100.f);
    std::vector<ThresholdStreamElement> stream(N);
    std::memset(stream.data(), 0, stream.size() * sizeof(ThresholdStreamElement));
    uint32_t a = 0, b = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (i == 0 || rng() % 4 == 0) { b += 1 + rng() % 3; if (b > 100) { a += 1; b = rng() % 5; } }
        stream[i].nodeIndexA = a;
        stream[i].nodeIndexB = b;
        stream[i].shapeInteraction = i;
        stream[i].normalForce = uf(rng);
        stream[i].threshold = 1.0e6f;
    }

    std::vector<float> refAcc, refBB, t1Acc, t1BB, t2Acc, t2BB;
    tsCudaAccumulate(stream, refAcc, refBB, 0);
    tsCudaAccumulate(stream, t1Acc, t1BB, 1);
    tsCudaAccumulate(stream, t2Acc, t2BB, 2);
    size_t k1 = 0, k2 = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (f2uTs(t1Acc[i]) != f2uTs(refAcc[i])) ++k1;
        if (f2uTs(t2Acc[i]) != f2uTs(refAcc[i])) ++k2;
    }
    std::printf("[ info ] thresholdStream bit-exact: twin1 (block total = incl) changes %zu, twin2 (sequential prefix) %zu of %u\n", k1, k2, N);
    PXG_CHECK_MSG(k1 > 0, "twin 1 (gsBlockForce += fInc) is not observable on this stream");
    PXG_CHECK_MSG(k2 > 0, "twin 2 (sequential between-block prefix) is not observable on this stream");

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<float> accForce(N, 0.f), accForceBB(32, 0.f);
    std::vector<uint32_t> writeIdx(N, 0), writeIdxBB(32, 0), writeable(N, 0), dummy(16, 0);
    PxgBufferHandle streamBuf = createBufferFrom(dev, stream);
    PxgBufferHandle accForceBuf = createBufferFrom(dev, accForce);
    PxgBufferHandle accForceBBBuf = createBufferFrom(dev, accForceBB);
    PxgBufferHandle writeIdxBuf = createBufferFrom(dev, writeIdx);
    PxgBufferHandle writeIdxBBBuf = createBufferFrom(dev, writeIdxBB);
    PxgBufferHandle writeableBuf = createBufferFrom(dev, writeable);
    PxgBufferHandle dummyBuf = createBufferFrom(dev, dummy);
    ThresholdParams p = {};
    p.nbThresholdElements = N;
    p.dt = 1.f;
    Kernel kAcc("thresholdStream.spv", "computeAccumulateThresholdStream",
                { 40, 4, 4, 4, 4, 4, 4, 4, 4, 4 }, sizeof(ThresholdParams));
    kAcc.dispatch(THRESH_GRID, { streamBuf, dummyBuf, dummyBuf, dummyBuf, dummyBuf, accForceBuf, accForceBBBuf,
                                 writeIdxBuf, writeIdxBBBuf, writeableBuf }, p);
    dev->barrier();
    const std::vector<float> gBB = downloadBuffer<float>(dev, accForceBBBuf, 32);
    Kernel kOut("thresholdStream.spv", "outputAccumulateThresholdStream",
                { 4, 4, 4, 4, 4, 4, 4, 4, 4 }, sizeof(ThresholdParams));
    kOut.dispatch(THRESH_GRID, { dummyBuf, dummyBuf, dummyBuf, dummyBuf, dummyBuf, accForceBuf, accForceBBBuf,
                                 writeIdxBuf, writeIdxBBBuf }, p);
    dev->flush();
    const std::vector<float> gAcc = downloadBuffer<float>(dev, accForceBuf, N);

    size_t bbBad = 0, accBad = 0;
    for (uint32_t i = 0; i < 32; ++i)
        if (f2uTs(gBB[i]) != f2uTs(refBB[i])) ++bbBad;
    for (uint32_t i = 0; i < N; ++i)
        if (f2uTs(gAcc[i]) != f2uTs(refAcc[i]))
        {
            if (accBad < 4) std::printf("  MM acc[%u] gpu %08x cuda %08x\n", i, f2uTs(gAcc[i]), f2uTs(refAcc[i]));
            ++accBad;
        }
    PXG_CHECK_MSG(bbBad == 0, "between-block force totals: %zu of 32 differ from the CUDA bitwise", bbBad);
    PXG_CHECK_MSG(accBad == 0, "accumulated force: %zu of %u differ from the CUDA bitwise", accBad, N);

    for (PxgBufferHandle h : { streamBuf, accForceBuf, accForceBBBuf, writeIdxBuf, writeIdxBBBuf, writeableBuf, dummyBuf })
        dev->destroyBuffer(h);
}

// ---------------------------------------------------------------------------
// Test 3: the grid-stride entries at the ENGINE's grid, above one pass
// ---------------------------------------------------------------------------
// P9.63.  Tests 1 and 2 run N <= 2001 against 32 x 256 = 8192 threads, so the
// four grid-stride loops NEVER ITERATE and their stride constant is unobserved.
// This runs the three that need no scan (both rank kernels and reorganize) with
// more elements than one engine pass covers.
//
// WHAT THIS PROVES, EXACTLY: that no index is SKIPPED at the engine's grid.  It
// does NOT prove the stride equals the engine's thread count -- a stride BELOW it
// makes threads revisit indices, and all three bodies are pure gathers whose
// destination buffer differs from their source, so a revisit rewrites the same
// bytes and passes.  (That is precisely the state the port was in before P9.63:
// stride 8192 against 16384 threads, benign but undocumented.)  What it does
// catch is the failure that is NOT benign -- a stride ABOVE the thread count,
// which leaves a hole -- and it catches it on the first element past the stride
// rather than at whatever scene size happens to reach it.
PXG_TEST(thresholdStream_engineGridCoverage)
{
    std::mt19937 rng(6311);
    // one engine pass is THRESH_GRID_WIDE * THRESH_BLOCK = 16384 elements for the
    // rank kernels, and an eighth of that in elements for reorganize (8 lanes each)
    const uint32_t N = THRESH_GRID_WIDE * THRESH_BLOCK + 1777; // 18161: two passes plus a tail

    std::vector<ThresholdStreamElement> stream(N);
    std::memset(stream.data(), 0, stream.size() * sizeof(ThresholdStreamElement));
    for (uint32_t i = 0; i < N; ++i)
    {
        stream[i].nodeIndexA = rng() % 4000;
        stream[i].nodeIndexB = rng() % 4000;
        stream[i].shapeInteraction = (uint64_t(rng()) << 32) | rng();
        stream[i].normalForce = float(rng() % 1000) * 0.01f;
        stream[i].threshold = float(rng() % 1000) * 0.01f;
        stream[i].accumulatedForce = -1.f;
        stream[i].pad = 0xdeadbeef;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t padCount = (4 - (N & 3)) & 3;

    // the sentinel is what makes a SKIP visible: an untouched slot keeps it
    std::vector<uint32_t> keysInit(N + padCount, 0xcdcdcdcdu), ranksInit(N + padCount, 0xcdcdcdcdu);
    std::vector<uint32_t> dummy(16, 0);

    PxgBufferHandle streamBuf = createBufferFrom(dev, stream);
    PxgBufferHandle keysBuf = createBufferFrom(dev, keysInit);
    PxgBufferHandle ranksBuf = createBufferFrom(dev, ranksInit);
    PxgBufferHandle dummyBuf = createBufferFrom(dev, dummy);

    ThresholdParams p = {};
    p.nbThresholdElements = N;
    p.dt = 1.f;

    Kernel kB("thresholdStream.spv", "initialRanksAndBodyIndexB", { 40, 4, 4 }, sizeof(ThresholdParams));
    kB.dispatch(THRESH_GRID_WIDE, { streamBuf, keysBuf, ranksBuf }, p);
    dev->flush();

    auto gKeysB = downloadBuffer<uint32_t>(dev, keysBuf, N + padCount);
    auto gRanksB = downloadBuffer<uint32_t>(dev, ranksBuf, N + padCount);
    size_t mism = 0, firstBad = N;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (gKeysB[i] != idxOf(stream[i].nodeIndexB) || gRanksB[i] != i)
        {
            if (mism == 0) firstBad = i;
            ++mism;
        }
    }
    PXG_CHECK_MSG(mism == 0, "initialRanksAndBodyIndexB N=%u: %zu mismatches, first at %zu (one pass = %u)",
                  N, mism, firstBad, THRESH_GRID_WIDE * THRESH_BLOCK);

    // initialRanksAndBodyIndexA over the identity permutation: keys[i] = A(ranks[i])
    std::vector<uint32_t> identity(N + padCount);
    std::iota(identity.begin(), identity.end(), 0u);
    PxgBufferHandle idRanksBuf = createBufferFrom(dev, identity);
    std::vector<uint32_t> keysReset(N + padCount, 0xcdcdcdcdu);
    PxgBufferHandle keysBuf2 = createBufferFrom(dev, keysReset);

    Kernel kA("thresholdStream.spv", "initialRanksAndBodyIndexA", { 40, 4, 4 }, sizeof(ThresholdParams));
    kA.dispatch(THRESH_GRID_WIDE, { streamBuf, keysBuf2, idRanksBuf }, p);
    dev->flush();

    auto gKeysA = downloadBuffer<uint32_t>(dev, keysBuf2, N + padCount);
    mism = 0; firstBad = N;
    for (uint32_t i = 0; i < N; ++i)
        if (gKeysA[i] != idxOf(stream[i].nodeIndexA))
        {
            if (mism == 0) firstBad = i;
            ++mism;
        }
    PXG_CHECK_MSG(mism == 0, "initialRanksAndBodyIndexA N=%u: %zu mismatches, first at %zu (one pass = %u)",
                  N, mism, firstBad, THRESH_GRID_WIDE * THRESH_BLOCK);

    // reorganize with a reversing permutation, so a skipped slot cannot coincide
    // with the value it would have held under the identity
    std::vector<uint32_t> revRanks(N + padCount, 0);
    for (uint32_t i = 0; i < N; ++i)
        revRanks[i] = N - 1 - i;
    PxgBufferHandle revRanksBuf = createBufferFrom(dev, revRanks);

    PxgBufferHandle tmpBuf = createBufferFrom(dev, stream);
    std::vector<ThresholdStreamElement> destInit(N);
    std::memset(destInit.data(), 0, destInit.size() * sizeof(ThresholdStreamElement));
    for (auto& e : destInit) e.accumulatedForce = 12345.f;
    PxgBufferHandle destBuf = createBufferFrom(dev, destInit);

    Kernel kR("thresholdStream.spv", "reorganizeThresholdElements", { 4, 4, 4, 40, 40 }, sizeof(ThresholdParams));
    kR.dispatch(THRESH_GRID_WIDE, { dummyBuf, dummyBuf, revRanksBuf, destBuf, tmpBuf }, p);
    dev->flush();

    auto gDest = downloadBuffer<ThresholdStreamElement>(dev, destBuf, N);
    mism = 0; firstBad = N;
    for (uint32_t i = 0; i < N; ++i)
    {
        const ThresholdStreamElement& r = stream[revRanks[i]];
        const ThresholdStreamElement& g = gDest[i];
        const bool bad = g.shapeInteraction != r.shapeInteraction || g.normalForce != r.normalForce ||
                         g.threshold != r.threshold || g.nodeIndexA != r.nodeIndexA ||
                         g.nodeIndexB != r.nodeIndexB || g.accumulatedForce != 12345.f;
        if (bad)
        {
            if (mism == 0) firstBad = i;
            ++mism;
        }
    }
    // reorganize spends 8 lanes per element, so one engine pass is 2048 elements
    PXG_CHECK_MSG(mism == 0, "reorganize N=%u: %zu mismatches, first at %zu (one pass = %u elements)",
                  N, mism, firstBad, (THRESH_GRID_WIDE * THRESH_BLOCK) / 8);

    for (PxgBufferHandle h : { streamBuf, keysBuf, ranksBuf, dummyBuf, idRanksBuf, keysBuf2,
                               revRanksBuf, tmpBuf, destBuf })
        dev->destroyBuffer(h);
}

} // namespace
