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

// Verification of broadphaseAggregatePairsAdd.slang against C++ transcriptions
// of addAggPairsStage1 (free-list slot allocation + pair scatter) and
// addAggPairsStage2 (single-thread free-list finalize + overflow flags).

#include "pxgTestFramework.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

const uint32_t SENT = 0xeeeeeeeeu;

struct AddPairsParams
{
    uint32_t nbFoundAggPairs;
    uint32_t maxFoundLostPairs;
    uint32_t maxAggPairs;
    uint32_t gridDimX;
};

PXG_TEST(broadphaseAggregatePairsAdd_stage1)
{
    std::mt19937 rng(26010);
    const uint32_t maxAgg = 5400, maxFL = 100000;
    const uint32_t nbFound = 1000;
    const uint32_t numFreeIndices = 300, maxIndex = 5000;
    const uint32_t nbFoundSafe = std::min(nbFound, maxFL);

    // distinct free indices in [0,maxIndex) (holes below the high-water mark)
    std::vector<uint32_t> freeIndices(maxIndex);
    std::iota(freeIndices.begin(), freeIndices.end(), 0u);
    std::shuffle(freeIndices.begin(), freeIndices.end(), rng);
    freeIndices.resize(numFreeIndices);

    std::vector<uint32_t> foundPairs(nbFound * 2);
    for (uint32_t i = 0; i < nbFound; ++i) { foundPairs[i * 2] = rng(); foundPairs[i * 2 + 1] = rng(); }

    std::vector<uint32_t> freeList = { numFreeIndices, maxIndex };
    std::vector<uint32_t> aggPairs(maxAgg * 4, SENT);

    // reference
    std::vector<uint32_t> rAgg = aggPairs;
    for (uint32_t i = 0; i < nbFoundSafe; ++i)
    {
        uint32_t index;
        if (i < numFreeIndices) index = freeIndices[numFreeIndices - i - 1];
        else index = maxIndex + (i - numFreeIndices);
        if (index < maxAgg)
        {
            rAgg[index * 4 + 0] = foundPairs[i * 2 + 0];
            rAgg[index * 4 + 1] = foundPairs[i * 2 + 1];
            rAgg[index * 4 + 2] = 1;
            rAgg[index * 4 + 3] = 0;
        }
    }

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<uint32_t> dummy(16, 0);
    PxgBufferHandle bFound = createBufferFrom(dev, foundPairs);
    PxgBufferHandle bFree = createBufferFrom(dev, freeList);
    PxgBufferHandle bFreeIdx = createBufferFrom(dev, freeIndices);
    PxgBufferHandle bAgg = createBufferFrom(dev, aggPairs);
    PxgBufferHandle bDummy = createBufferFrom(dev, dummy);

    const uint32_t gridX = (nbFoundSafe + 255) / 256;
    AddPairsParams p = { nbFound, maxFL, maxAgg, gridX };
    const std::vector<uint32_t> strides(5, 4);
    Kernel k("broadphaseAggregatePairsAdd.spv", "addAggPairsStage1", strides, sizeof(AddPairsParams));
    k.dispatch(gridX, { bFound, bFree, bFreeIdx, bAgg, bDummy }, p);
    dev->flush();

    const auto gAgg = downloadBuffer<uint32_t>(dev, bAgg, maxAgg * 4);
    size_t mism = 0;
    for (uint32_t i = 0; i < maxAgg * 4; ++i) if (gAgg[i] != rAgg[i]) ++mism;
    PXG_CHECK_MSG(mism == 0, "%zu aggPair-slot mismatches", mism);
    for (PxgBufferHandle h : { bFound, bFree, bFreeIdx, bAgg, bDummy }) dev->destroyBuffer(h);
}

// run stage2 once with the given inputs and return {freeList(2), scalarOut(5)}
static void runStage2(PxgSlangComputeDevice* dev, uint32_t nbFound, uint32_t maxFL, uint32_t maxAgg,
                      uint32_t numFreeInds, uint32_t maxIndexIn, std::vector<uint32_t>& scalarInOut,
                      std::vector<uint32_t>& freeOut)
{
    std::vector<uint32_t> dummy(16, 0);
    std::vector<uint32_t> freeList = { numFreeInds, maxIndexIn };
    PxgBufferHandle bFree = createBufferFrom(dev, freeList);
    PxgBufferHandle bScalar = createBufferFrom(dev, scalarInOut);
    PxgBufferHandle bDummy = createBufferFrom(dev, dummy);

    AddPairsParams p = { nbFound, maxFL, maxAgg, 1 };
    const std::vector<uint32_t> strides(5, 4);
    Kernel k("broadphaseAggregatePairsAdd.spv", "addAggPairsStage2", strides, sizeof(AddPairsParams));
    k.dispatch(1, { bDummy, bFree, bDummy, bDummy, bScalar }, p);
    dev->flush();

    scalarInOut = downloadBuffer<uint32_t>(dev, bScalar, 5);
    freeOut = downloadBuffer<uint32_t>(dev, bFree, 2);
    for (PxgBufferHandle h : { bFree, bScalar, bDummy }) dev->destroyBuffer(h);
}

PXG_TEST(broadphaseAggregatePairsAdd_stage2)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t KEEP = 7; // sentinel for conditionally-written flags

    // Scenario A: found overflow (nbFound>maxFL), grow past free list, agg overflow past cap.
    // nbFound=1000, maxFL=800 -> safe=800; numFree=300 -> numToIncrement=500, newFree=0;
    // maxIndex=5000+500=5500 (overflow count); 5500>maxAgg(5400) -> clamp 5400, agg flag set;
    // nbFound(1000)>maxFL(800) -> found flag + candidate count 1000.
    {
        std::vector<uint32_t> scalar = { 0, KEEP, 0, KEEP, KEEP }, freeOut;
        runStage2(dev, 1000, 800, 5400, 300, 5000, scalar, freeOut);
        PXG_CHECK_MSG(scalar[0] == 5500, "A aggPairOverflowCount=%u exp 5500", scalar[0]);
        PXG_CHECK_MSG(scalar[1] == 1, "A agg_overflow_flag=%u exp 1", scalar[1]);
        PXG_CHECK_MSG(scalar[2] == 5400, "A aggPairCount=%u exp 5400", scalar[2]);
        PXG_CHECK_MSG(scalar[3] == 1, "A found_overflow_flag=%u exp 1", scalar[3]);
        PXG_CHECK_MSG(scalar[4] == 1000, "A foundCandidateCount=%u exp 1000", scalar[4]);
        PXG_CHECK_MSG(freeOut[0] == 0 && freeOut[1] == 5400, "A freeList={%u,%u} exp {0,5400}", freeOut[0], freeOut[1]);
    }

    // Scenario B: no overflow. numFree=1000 >= safe(500), newFree=500, numToIncrement=0;
    // maxIndex=200+0=200 <= maxAgg -> no agg flag; nbFound(500)<=maxFL(800) -> no found flag.
    // The conditionally-written flags/count must be PRESERVED (stay KEEP).
    {
        std::vector<uint32_t> scalar = { 0, KEEP, 0, KEEP, KEEP }, freeOut;
        runStage2(dev, 500, 800, 5400, 1000, 200, scalar, freeOut);
        PXG_CHECK_MSG(scalar[0] == 200, "B aggPairOverflowCount=%u exp 200", scalar[0]);
        PXG_CHECK_MSG(scalar[1] == KEEP, "B agg_overflow_flag=%u exp preserved", scalar[1]);
        PXG_CHECK_MSG(scalar[2] == 200, "B aggPairCount=%u exp 200", scalar[2]);
        PXG_CHECK_MSG(scalar[3] == KEEP, "B found_overflow_flag=%u exp preserved", scalar[3]);
        PXG_CHECK_MSG(scalar[4] == KEEP, "B foundCandidateCount=%u exp preserved", scalar[4]);
        PXG_CHECK_MSG(freeOut[0] == 500 && freeOut[1] == 200, "B freeList={%u,%u} exp {500,200}", freeOut[0], freeOut[1]);
    }
}

} // namespace
