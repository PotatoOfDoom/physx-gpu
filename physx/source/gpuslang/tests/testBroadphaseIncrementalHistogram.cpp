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

// Verification of broadphaseIncrementalHistogram.slang against a C++
// transcription of the incremental-SAP comparison-count stage: for every
// projection on each axis, count the overlap comparisons as a difference of two
// prefix-sum histogram entries (selected by isStartHandle / isDownPass, deleted
// projections count 0), then check the device-wide two-pass exclusive prefix
// sum (per-projection running total, per-block totals and the grand total).

#include "pxgTestFramework.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct HistParams
{
    uint32_t numPreviousHandles;
};

PXG_TEST(broadphaseIncrementalHistogram_stage1and2)
{
    std::mt19937 rng(14000);

    const uint32_t numHandles = 6000;         // -> nbProj = 12000
    const uint32_t nbProj = numHandles * 2;
    const uint32_t NUM_BLOCKS = 32;
    const uint32_t BLOCK = 256;

    // ---- inputs ----
    // ranks: mostly in-range values (a random rank each), plus ~15% out-of-range
    // ranks (>= nbProj) which force the down pass and exercise the rank clamp.
    // handles: random low bits { isStart(0), isNew(1), isDeleted(2) } + handle.
    std::vector<uint32_t> ranks(3 * nbProj), handles(3 * nbProj);
    for (uint32_t axis = 0; axis < 3; ++axis)
    {
        for (uint32_t i = 0; i < nbProj; ++i)
        {
            uint32_t rank;
            if (rng() % 100 < 15)
                rank = nbProj + (rng() % 2000); // out of range -> down pass + clamp
            else
                rank = rng() % nbProj;
            ranks[axis * nbProj + i] = rank;

            const uint32_t isStart = rng() & 1u;
            const uint32_t isNew = rng() & 1u;
            const uint32_t isDeleted = (rng() % 100 < 10) ? 1u : 0u; // ~10% deleted
            const uint32_t handle = rng() % 100000u;
            handles[axis * nbProj + i] = (handle << 3) | isStart | (isNew << 1) | (isDeleted << 2);
        }
    }

    // histograms: non-decreasing prefix-sum arrays per (frame, axis); layout
    // [(frame*3 + axis) * nbProj + i].  Prefix sums of random 0/1 counts.
    std::vector<uint32_t> startHist(2 * 3 * nbProj), endHist(2 * 3 * nbProj);
    for (uint32_t frame = 0; frame < 2; ++frame)
    {
        for (uint32_t axis = 0; axis < 3; ++axis)
        {
            const uint32_t base = (frame * 3 + axis) * nbProj;
            uint32_t accS = 0, accE = 0;
            for (uint32_t i = 0; i < nbProj; ++i)
            {
                startHist[base + i] = accS;
                endHist[base + i] = accE;
                accS += (rng() & 1u);
                accE += (rng() & 1u);
            }
        }
    }

    // ---- reference ----
    const uint32_t nbBlocksRequired = (nbProj + BLOCK - 1) / BLOCK;
    const uint32_t numIterationPerBlock = (nbBlocksRequired + NUM_BLOCKS - 1) / NUM_BLOCKS;

    std::vector<uint32_t> refInc(3 * nbProj, 0);
    std::vector<uint32_t> refBlock(3 * NUM_BLOCKS, 0);
    std::vector<uint32_t> refTotal(3, 0);

    for (uint32_t axis = 0; axis < 3; ++axis)
    {
        std::vector<uint32_t> comp(nbProj, 0);
        for (uint32_t i = 0; i < nbProj; ++i)
        {
            const uint32_t rank = ranks[axis * nbProj + i];
            const uint32_t sh = handles[axis * nbProj + i];
            const bool isDownPass = i <= rank;
            const bool isStartHandle = (sh & 1u) != 0u;
            const bool isDeleted = (sh & (1u << 2)) != 0u;

            const uint32_t start = isDownPass ? i : (rank + 1);
            const uint32_t end = isDownPass ? std::min(rank, nbProj - 1) : i;

            uint32_t c = 0;
            if (!isDeleted)
            {
                const uint32_t frame = isDownPass ? 1u : 0u;
                const uint32_t base = (frame * 3 + axis) * nbProj;
                const std::vector<uint32_t>& H = isStartHandle ? endHist : startHist;
                c = H[base + end] - H[base + start];
            }
            comp[i] = c;
        }

        // device-wide exclusive prefix sum
        uint32_t acc = 0;
        for (uint32_t i = 0; i < nbProj; ++i)
        {
            refInc[axis * nbProj + i] = acc;
            acc += comp[i];
        }
        refTotal[axis] = acc;

        // per-block totals over each block's contiguous projection range
        for (uint32_t b = 0; b < NUM_BLOCKS; ++b)
        {
            const uint32_t s = b * numIterationPerBlock * BLOCK;
            const uint32_t e = std::min((b + 1) * numIterationPerBlock * BLOCK, nbProj);
            uint32_t bt = 0;
            for (uint32_t i = s; i < e; ++i)
                bt += comp[i];
            refBlock[axis * NUM_BLOCKS + b] = bt;
        }
    }

    // ---- device ----
    PxgSlangComputeDevice* dev = testDevice();

    std::vector<uint32_t> incInit(3 * nbProj, 0xdeadbeefu);
    std::vector<uint32_t> blockInit(3 * NUM_BLOCKS, 0xdeadbeefu);
    std::vector<uint32_t> totalInit(3, 0xdeadbeefu);

    PxgBufferHandle rBuf = createBufferFrom(dev, ranks);
    PxgBufferHandle hBuf = createBufferFrom(dev, handles);
    PxgBufferHandle shBuf = createBufferFrom(dev, startHist);
    PxgBufferHandle ehBuf = createBufferFrom(dev, endHist);
    PxgBufferHandle icBuf = createBufferFrom(dev, incInit);
    PxgBufferHandle bcBuf = createBufferFrom(dev, blockInit);
    PxgBufferHandle tBuf = createBufferFrom(dev, totalInit);

    HistParams p = {};
    p.numPreviousHandles = numHandles;

    const std::vector<uint32_t> strides(7, 4);
    const std::vector<PxgBufferHandle> buffers = { rBuf, hBuf, shBuf, ehBuf, icBuf, bcBuf, tBuf };

    Kernel k1("broadphaseIncrementalHistogram.spv", "computeIncrementalComparisonHistograms_Stage1", strides,
              sizeof(HistParams));
    k1.dispatch(32, buffers, p);
    dev->barrier();

    Kernel k2("broadphaseIncrementalHistogram.spv", "computeIncrementalComparisonHistograms_Stage2", strides,
              sizeof(HistParams));
    k2.dispatch(32, buffers, p);
    dev->flush();

    const auto gInc = downloadBuffer<uint32_t>(dev, icBuf, 3 * nbProj);
    const auto gBlock = downloadBuffer<uint32_t>(dev, bcBuf, 3 * NUM_BLOCKS);
    const auto gTotal = downloadBuffer<uint32_t>(dev, tBuf, 3);

    // per-block totals (written by stage 1)
    size_t mismB = 0;
    for (uint32_t i = 0; i < 3 * NUM_BLOCKS; ++i)
        if (gBlock[i] != refBlock[i]) ++mismB;
    PXG_CHECK_MSG(mismB == 0, "%zu per-block comparison mismatches", mismB);

    // device-wide exclusive prefix (stage 1 within-block + stage 2 block offset)
    size_t mismI = 0;
    uint32_t firstBad = 0xffffffffu;
    for (uint32_t i = 0; i < 3 * nbProj; ++i)
    {
        if (gInc[i] != refInc[i])
        {
            if (firstBad == 0xffffffffu) firstBad = i;
            ++mismI;
        }
    }
    if (mismI)
        PXG_CHECK_MSG(mismI == 0, "%zu incremental-comparison mismatches (first at %u: gpu=%u ref=%u)", mismI,
                      firstBad, gInc[firstBad], refInc[firstBad]);
    else
        PXG_CHECK_MSG(mismI == 0, "%zu incremental-comparison mismatches", mismI);

    for (uint32_t axis = 0; axis < 3; ++axis)
        PXG_CHECK_MSG(gTotal[axis] == refTotal[axis], "axis %u total: gpu=%u ref=%u", axis, gTotal[axis],
                      refTotal[axis]);

    for (PxgBufferHandle h : { rBuf, hBuf, shBuf, ehBuf, icBuf, bcBuf, tBuf })
        dev->destroyBuffer(h);
}

} // namespace
