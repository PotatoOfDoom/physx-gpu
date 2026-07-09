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

// Verification of npHfMidphase.slang: hfmComputePairs + hfmOutputPairs — the
// heightfield midphase pair-generation core (heightfieldComputePairs /
// heightfieldOutputPairs).  All outputs are pure integer: the per-query pair
// count, and the emitted uint4(cmInd, elementInd, triangleIdx, 0) entries.  The
// y-range culling is an integer comparison (heights are exact i16->float, miny/
// maxy are small non-negative integers), so the whole computation is compared
// BIT-EXACTLY against a C++ transcription of the CUDA — no tolerance.

#include "pxgTestFramework.h"

#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct U4 { uint32_t x, y, z, w; };
struct HfQuery { uint32_t minColumn, maxColumn, minRow, maxRow, miny, maxy, cmInd, elementInd; };

static const uint32_t HF_HOLE = 127;

static float refHeight(uint32_t word) { return float((int16_t)(word & 0xffffu)); }
static uint32_t refMat0(uint32_t word) { return (word >> 16) & 0x7fu; }
static uint32_t refMat1(uint32_t word) { return (word >> 24) & 0x7fu; }

// C++ transcription of heightfieldComputePairs / heightfieldOutputPairs.  Emits
// into `out` (capped at pairsPerTest) and returns the number written; when
// out==nullptr it only counts (the computePairs behaviour, which has no cap).
static uint32_t refPairs(const HfQuery& q, uint32_t nbCols, const std::vector<uint32_t>& samples,
                         uint32_t pairsPerTest, U4* out)
{
    const float maxy = float(q.maxy), miny = float(q.miny);
    const uint32_t columnSpan = q.maxColumn - q.minColumn;
    const uint32_t totalNumProcessed = (q.maxRow - q.minRow) * columnSpan * 2u;
    uint32_t written = 0;
    for (uint32_t i = 0; i < totalNumProcessed; ++i)
    {
        if (out && written >= pairsPerTest) break; // output-slot cap (compute has none)
        const uint32_t index = i / 2u;
        const uint32_t vertexIndex = (q.minRow + index / columnSpan) * nbCols + (q.minColumn + index % columnSpan);
        const float h0 = refHeight(samples[vertexIndex]);
        const float h1 = refHeight(samples[vertexIndex + 1]);
        const float h2 = refHeight(samples[vertexIndex + nbCols]);
        const float h3 = refHeight(samples[vertexIndex + nbCols + 1]);
        const bool con0 = maxy < h0 && maxy < h1 && maxy < h2 && maxy < h3;
        const bool con1 = miny > h0 && miny > h1 && miny > h2 && miny > h3;
        if (!(con0 || con1))
        {
            const bool isMaterial1 = (i & 1u) != 0u;
            const uint32_t material = isMaterial1 ? refMat1(samples[vertexIndex]) : refMat0(samples[vertexIndex]);
            if (material != HF_HOLE)
            {
                if (out)
                {
                    const uint32_t triangleIdx = isMaterial1 ? ((vertexIndex << 1) + 1u) : (vertexIndex << 1);
                    out[written] = { q.cmInd, q.elementInd, triangleIdx, 0 };
                }
                ++written;
            }
        }
    }
    return written;
}

static void buildGrid(std::mt19937& rng, uint32_t NV, std::vector<uint32_t>& samples)
{
    samples.resize(NV);
    for (uint32_t i = 0; i < NV; ++i)
    {
        // heights in a modest band so the y-range culling actually fires
        const int16_t h = (int16_t)(rng() % 300);
        // materials 0..127 so ~1/128 are eHOLE; also make holes more common to cover the branch
        uint32_t m0 = rng() % 160; if (m0 >= 128) m0 = HF_HOLE; // ~20% holes
        uint32_t m1 = rng() % 160; if (m1 >= 128) m1 = HF_HOLE;
        // random top flag bits (tessFlag / reserved) that must be masked off
        m0 |= (rng() & 1u) << 7;
        m1 |= (rng() & 1u) << 7;
        samples[i] = (uint32_t(uint16_t(h))) | (m0 << 16) | (m1 << 24);
    }
}

static void makeQueries(std::mt19937& rng, uint32_t rows, uint32_t columns, uint32_t N,
                        std::vector<HfQuery>& q)
{
    q.resize(N);
    for (uint32_t t = 0; t < N; ++t)
    {
        const uint32_t minRow = rng() % (rows - 1);
        const uint32_t maxRow = minRow + 1 + rng() % (rows - 1 - minRow);       // (minRow, rows-1]
        const uint32_t minCol = rng() % (columns - 1);
        const uint32_t maxCol = minCol + 1 + rng() % (columns - 1 - minCol);    // (minCol, columns-1]
        const uint32_t miny = rng() % 300;
        const uint32_t maxy = miny + rng() % 100; // maxy >= miny, both in the height band
        q[t] = { minCol, maxCol, minRow, maxRow, miny, maxy, 1000u + t, 2000u + t };
    }
}

PXG_TEST(npHfMidphase_computePairs)
{
    std::mt19937 rng(61041);
    const uint32_t rows = 40, columns = 32;
    std::vector<uint32_t> samples; buildGrid(rng, rows * columns, samples);

    const uint32_t N = 8000;
    std::vector<HfQuery> queries; makeQueries(rng, rows, columns, N, queries);

    std::vector<uint32_t> ref(N);
    size_t zeroC = 0, nonzeroC = 0, bigC = 0;
    for (uint32_t t = 0; t < N; ++t)
    {
        ref[t] = refPairs(queries[t], columns, samples, 0, nullptr);
        if (ref[t] == 0) ++zeroC; else ++nonzeroC;
        if (ref[t] > 50) ++bigC;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<uint32_t> countInit(N, 0xdeadbeef);
    std::vector<U4> pairsDummy(N);
    PxgBufferHandle bSamples = createBufferFrom(dev, samples);
    PxgBufferHandle bQ = createBufferFrom(dev, queries);
    PxgBufferHandle bCounts = createBufferFrom(dev, countInit);
    PxgBufferHandle bPairs = createBufferFrom(dev, pairsDummy);

    struct HfmParams { uint32_t numTests, nbRows, nbCols, pairsPerTest; }
    prm = { N, rows, columns, 1 };
    Kernel k("npHfMidphase.spv", "hfmComputePairs",
             { 4, (uint32_t)sizeof(HfQuery), 4, (uint32_t)sizeof(U4) }, sizeof(HfmParams));
    k.dispatch(GROUPS, { bSamples, bQ, bCounts, bPairs }, prm);
    dev->flush();

    const auto g = downloadBuffer<uint32_t>(dev, bCounts, N);
    size_t mm = 0;
    for (uint32_t t = 0; t < N; ++t) if (g[t] != ref[t]) ++mm;
    PXG_CHECK_MSG(mm == 0, "hfmComputePairs: %zu count mismatches", mm);
    PXG_CHECK_MSG(zeroC > 20 && nonzeroC > N / 4 && bigC > 20,
                  "hfmComputePairs: coverage thin (zero=%zu nonzero=%zu big=%zu)", zeroC, nonzeroC, bigC);

    dev->destroyBuffer(bSamples); dev->destroyBuffer(bQ); dev->destroyBuffer(bCounts); dev->destroyBuffer(bPairs);
}

PXG_TEST(npHfMidphase_outputPairs)
{
    std::mt19937 rng(61047);
    const uint32_t rows = 40, columns = 32;
    std::vector<uint32_t> samples; buildGrid(rng, rows * columns, samples);

    const uint32_t N = 8000;
    std::vector<HfQuery> queries; makeQueries(rng, rows, columns, N, queries);

    // cap chosen so some queries fully emit and some are truncated by the cap
    const uint32_t PAIRS_PER_TEST = 40;
    std::vector<uint32_t> refCount(N);
    std::vector<U4> ref(N * PAIRS_PER_TEST);
    size_t cappedC = 0, fullEmitC = 0, zeroC = 0;
    for (uint32_t t = 0; t < N; ++t)
    {
        const uint32_t uncapped = refPairs(queries[t], columns, samples, 0, nullptr);
        refCount[t] = refPairs(queries[t], columns, samples, PAIRS_PER_TEST, &ref[t * PAIRS_PER_TEST]);
        if (uncapped > PAIRS_PER_TEST) ++cappedC; else if (uncapped > 0) ++fullEmitC; else ++zeroC;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<uint32_t> countInit(N, 0xdeadbeef);
    std::vector<U4> pairsInit(N * PAIRS_PER_TEST, U4{ 0, 0, 0, 0 });
    PxgBufferHandle bSamples = createBufferFrom(dev, samples);
    PxgBufferHandle bQ = createBufferFrom(dev, queries);
    PxgBufferHandle bCounts = createBufferFrom(dev, countInit);
    PxgBufferHandle bPairs = createBufferFrom(dev, pairsInit);

    struct HfmParams { uint32_t numTests, nbRows, nbCols, pairsPerTest; }
    prm = { N, rows, columns, PAIRS_PER_TEST };
    Kernel k("npHfMidphase.spv", "hfmOutputPairs",
             { 4, (uint32_t)sizeof(HfQuery), 4, (uint32_t)sizeof(U4) }, sizeof(HfmParams));
    k.dispatch(GROUPS, { bSamples, bQ, bCounts, bPairs }, prm);
    dev->flush();

    const auto gc = downloadBuffer<uint32_t>(dev, bCounts, N);
    const auto gp = downloadBuffer<U4>(dev, bPairs, N * PAIRS_PER_TEST);
    size_t mmCount = 0, mmPair = 0;
    for (uint32_t t = 0; t < N; ++t)
    {
        if (gc[t] != refCount[t]) { ++mmCount; continue; }
        for (uint32_t j = 0; j < refCount[t]; ++j)
        {
            const U4& r = ref[t * PAIRS_PER_TEST + j];
            const U4& v = gp[t * PAIRS_PER_TEST + j];
            if (r.x != v.x || r.y != v.y || r.z != v.z || r.w != v.w) { ++mmPair; break; }
        }
    }
    PXG_CHECK_MSG(mmCount == 0, "hfmOutputPairs: %zu count mismatches", mmCount);
    PXG_CHECK_MSG(mmPair == 0, "hfmOutputPairs: %zu pair-payload mismatches", mmPair);
    PXG_CHECK_MSG(cappedC > 20 && fullEmitC > N / 4 && zeroC > 20,
                  "hfmOutputPairs: coverage thin (capped=%zu full=%zu zero=%zu)", cappedC, fullEmitC, zeroC);

    dev->destroyBuffer(bSamples); dev->destroyBuffer(bQ); dev->destroyBuffer(bCounts); dev->destroyBuffer(bPairs);
}

} // namespace
