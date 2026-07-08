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

// Verification of broadphaseEndPtHistogram.slang against a C++ transcription of
// the broad-phase start/end-point projection histogram builders: per-projection
// start/end/end2 classification, the device-wide (two-pass) exclusive prefix
// producing start/end/totalEnd point histograms and their per-block totals, and
// the compacting scatter of non-deleted handles into the ordered start/end
// point handle arrays.  Covers incremental (with and without removals) and
// full-rebuild (non-incremental) modes across the 32-block boundary.

#include "pxgTestFramework.h"

#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct EndPtHistParams
{
    uint32_t numPreviousHandles;
    uint32_t numCreatedHandles;
    uint32_t numRemovedHandles;
    uint32_t isIncremental;
};

const uint32_t NUM_BLOCKS = 32;
const uint32_t BLOCK = 256;
const uint32_t SENT = 0xdeadbeefu;

bool isStart(uint32_t sh) { return (sh & 1u) != 0u; }
bool isDeleted(uint32_t sh) { return (sh & (1u << 2)) != 0u; }
uint32_t getHandle(uint32_t sh) { return sh >> 3; }

// Runs one dispatch configuration end-to-end and asserts every produced array
// against a CPU reference.  Returns nothing; failures are recorded via
// PXG_CHECK_MSG.
void runEndPtCase(const char* tag, uint32_t seed, uint32_t numPrev, uint32_t numCreated,
                  uint32_t numRemoved, uint32_t isIncremental)
{
    std::mt19937 rng(seed);

    const uint32_t nH = isIncremental ? numPrev : (numPrev + numCreated - numRemoved);
    const uint32_t P = nH * 2;
    const bool doFrame1 = isIncremental && (numRemoved != 0);
    const uint32_t numFrames = doFrame1 ? 2u : 1u;

    const uint32_t nbBlocksRequired = (P + BLOCK - 1) / BLOCK;
    const uint32_t numIterationPerBlock = (nbBlocksRequired + NUM_BLOCKS - 1) / NUM_BLOCKS;

    // ---- inputs: boxHandles[2][3][P] flat (fill both frames/all axes) ----
    // Real broad-phase projections come in exact start/end pairs (one of each per
    // box), so every (frame,axis) has exactly nH start and nH end projections;
    // the ordered-handle scatter stride H = nH depends on that invariant. Build
    // nH pairs per (frame,axis) and shuffle them into a plausible sorted order,
    // with an independent "new" bit (~must not gate the scatter) and ~10% deleted.
    std::vector<uint32_t> boxHandles(2 * 3 * P);
    for (uint32_t f = 0; f < 2; ++f)
        for (uint32_t axis = 0; axis < 3; ++axis)
        {
            std::vector<uint32_t> proj;
            proj.reserve(P);
            for (uint32_t k = 0; k < nH; ++k)
            {
                const uint32_t h = rng() % 100000u;
                const uint32_t nS = rng() & 1u, nE = rng() & 1u; // "new" bit
                const uint32_t dS = (rng() % 100 < 10) ? 1u : 0u;
                const uint32_t dE = (rng() % 100 < 10) ? 1u : 0u;
                proj.push_back((h << 3) | 1u | (nS << 1) | (dS << 2)); // start projection (bit0 = 1)
                proj.push_back((h << 3) | 0u | (nE << 1) | (dE << 2)); // end projection   (bit0 = 0)
            }
            std::shuffle(proj.begin(), proj.end(), rng);
            for (uint32_t i = 0; i < P; ++i)
                boxHandles[(f * 3 + axis) * P + i] = proj[i];
        }

    // ---- device buffers ----
    PxgSlangComputeDevice* dev = testDevice();

    std::vector<uint32_t> endHistI(2 * 3 * P, SENT), startHistI(2 * 3 * P, SENT), totalEndI(3 * P, SENT);
    std::vector<uint32_t> blockEndI(2 * 3 * NUM_BLOCKS, SENT), blockStartI(2 * 3 * NUM_BLOCKS, SENT),
        blockTotalI(3 * NUM_BLOCKS, SENT);
    std::vector<uint32_t> endHandI(2 * 3 * nH, SENT), startHandI(2 * 3 * nH, SENT);

    PxgBufferHandle bh = createBufferFrom(dev, boxHandles);
    PxgBufferHandle eh = createBufferFrom(dev, endHistI);
    PxgBufferHandle sh = createBufferFrom(dev, startHistI);
    PxgBufferHandle th = createBufferFrom(dev, totalEndI);
    PxgBufferHandle beh = createBufferFrom(dev, blockEndI);
    PxgBufferHandle bsh = createBufferFrom(dev, blockStartI);
    PxgBufferHandle bth = createBufferFrom(dev, blockTotalI);
    PxgBufferHandle ehd = createBufferFrom(dev, endHandI);
    PxgBufferHandle shd = createBufferFrom(dev, startHandI);

    EndPtHistParams p = {};
    p.numPreviousHandles = numPrev;
    p.numCreatedHandles = numCreated;
    p.numRemovedHandles = numRemoved;
    p.isIncremental = isIncremental;

    const std::vector<uint32_t> strides(9, 4);
    const std::vector<PxgBufferHandle> buffers = { bh, eh, sh, th, beh, bsh, bth, ehd, shd };

    Kernel k1("broadphaseEndPtHistogram.spv", "computeEndPtsHistogram", strides, sizeof(EndPtHistParams));
    k1.dispatch(32, buffers, p);
    dev->barrier();

    Kernel k2("broadphaseEndPtHistogram.spv", "outputEndPtsHistogram", strides, sizeof(EndPtHistParams));
    k2.dispatch(32, buffers, p);
    dev->flush();

    const auto gEnd = downloadBuffer<uint32_t>(dev, eh, 2 * 3 * P);
    const auto gStart = downloadBuffer<uint32_t>(dev, sh, 2 * 3 * P);
    const auto gTotal = downloadBuffer<uint32_t>(dev, th, 3 * P);
    const auto gBlockEnd = downloadBuffer<uint32_t>(dev, beh, 2 * 3 * NUM_BLOCKS);
    const auto gBlockStart = downloadBuffer<uint32_t>(dev, bsh, 2 * 3 * NUM_BLOCKS);
    const auto gBlockTotal = downloadBuffer<uint32_t>(dev, bth, 3 * NUM_BLOCKS);
    const auto gEndHand = downloadBuffer<uint32_t>(dev, ehd, 2 * 3 * nH);
    const auto gStartHand = downloadBuffer<uint32_t>(dev, shd, 2 * 3 * nH);

    // ---- reference + assertions, per processed frame/axis ----
    size_t histMism = 0, blockMism = 0, handMism = 0;
    uint32_t firstBadHist = 0xffffffffu, badGpu = 0, badRef = 0;

    for (uint32_t frame = 0; frame < numFrames; ++frame)
    {
        const bool writeTotal = (frame == 0);
        for (uint32_t axis = 0; axis < 3; ++axis)
        {
            const uint32_t faBase = (frame * 3 + axis) * P;
            const uint32_t taBase = axis * P;
            const uint32_t blkBase = (frame * 3 + axis) * NUM_BLOCKS;
            const uint32_t tblkBase = axis * NUM_BLOCKS;
            const uint32_t hBase = (frame * 3 + axis) * nH;

            std::vector<uint32_t> refEnd(P), refStart(P), refTotal(P);
            std::vector<uint32_t> refEndHand, refStartHand;
            uint32_t accE = 0, accS = 0, accT = 0;
            for (uint32_t i = 0; i < P; ++i)
            {
                const uint32_t sortedHandle = boxHandles[faBase + i];
                const uint32_t end2 = 1u - (isStart(sortedHandle) ? 1u : 0u);
                const uint32_t st = isDeleted(sortedHandle) ? 0u : (1u - end2);
                const uint32_t en = isDeleted(sortedHandle) ? 0u : end2;
                refEnd[i] = accE;   // exclusive prefix (global — matches the two-pass result)
                refStart[i] = accS;
                refTotal[i] = accT;
                if (!isDeleted(sortedHandle))
                {
                    if (isStart(sortedHandle))
                        refStartHand.push_back(getHandle(sortedHandle));
                    else
                        refEndHand.push_back(getHandle(sortedHandle));
                }
                accE += en;
                accS += st;
                accT += end2;
            }

            for (uint32_t i = 0; i < P; ++i)
            {
                if (gEnd[faBase + i] != refEnd[i])
                {
                    if (firstBadHist == 0xffffffffu) { firstBadHist = faBase + i; badGpu = gEnd[faBase + i]; badRef = refEnd[i]; }
                    ++histMism;
                }
                if (gStart[faBase + i] != refStart[i]) ++histMism;
                if (writeTotal && gTotal[taBase + i] != refTotal[i]) ++histMism;
            }

            // block totals over each block's contiguous range
            for (uint32_t b = 0; b < NUM_BLOCKS; ++b)
            {
                const uint32_t s0 = b * numIterationPerBlock * BLOCK;
                const uint32_t e0 = std::min((b + 1) * numIterationPerBlock * BLOCK, P);
                uint32_t bE = 0, bS = 0, bT = 0;
                for (uint32_t i = s0; i < e0; ++i)
                {
                    const uint32_t sortedHandle = boxHandles[faBase + i];
                    const uint32_t end2 = 1u - (isStart(sortedHandle) ? 1u : 0u);
                    bT += end2;
                    if (!isDeleted(sortedHandle))
                    {
                        if (end2) ++bE; else ++bS;
                    }
                }
                if (gBlockEnd[blkBase + b] != bE) ++blockMism;
                if (gBlockStart[blkBase + b] != bS) ++blockMism;
                if (writeTotal && gBlockTotal[tblkBase + b] != bT) ++blockMism;
            }

            // ordered (compacted) handles
            for (size_t j = 0; j < refStartHand.size(); ++j)
                if (gStartHand[hBase + j] != refStartHand[j]) ++handMism;
            for (size_t j = 0; j < refEndHand.size(); ++j)
                if (gEndHand[hBase + j] != refEndHand[j]) ++handMism;
        }
    }

    if (histMism)
        PXG_CHECK_MSG(histMism == 0, "[%s] %zu histogram mismatches (first at %u: gpu=%u ref=%u)", tag, histMism,
                      firstBadHist, badGpu, badRef);
    else
        PXG_CHECK_MSG(histMism == 0, "[%s] %zu histogram mismatches", tag, histMism);
    PXG_CHECK_MSG(blockMism == 0, "[%s] %zu block-total mismatches", tag, blockMism);
    PXG_CHECK_MSG(handMism == 0, "[%s] %zu ordered-handle mismatches", tag, handMism);

    for (PxgBufferHandle h : { bh, eh, sh, th, beh, bsh, bth, ehd, shd })
        dev->destroyBuffer(h);
}

PXG_TEST(broadphaseEndPtHistogram_incrementalNoRemoval)
{
    runEndPtCase("incremental-no-removal", 15100, 6000, 0, 0, 1);
}

PXG_TEST(broadphaseEndPtHistogram_incrementalWithRemoval)
{
    runEndPtCase("incremental-with-removal", 15200, 6000, 0, 500, 1);
}

PXG_TEST(broadphaseEndPtHistogram_nonIncremental)
{
    runEndPtCase("non-incremental", 15300, 4000, 2500, 500, 0);
}

} // namespace
