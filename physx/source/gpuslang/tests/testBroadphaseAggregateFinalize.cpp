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

// Verification of broadphaseAggregateFinalize.slang against C++ transcriptions
// of aggCopyReports (clamped report-map copy + overflow flags), removeAggPairsStage3
// (free-index count update) and markAddedAndDeletedAggregatedBounds (setEmpty of
// added/deleted integer bounds).

#include "pxgTestFramework.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

const uint32_t SENT = 0xabcddcbau;
const uint32_t E_MIN = 0xff7fffffu;
const uint32_t E_MAX = 0x00800000u;

struct FinalizeParams
{
    uint32_t nbCreatedPairs, nbLostPairs, maxFoundLostPairs, nbRemoved, numAdded, numDeleted, gridDimX;
};

const std::vector<uint32_t> STRIDES10(10, 4);

// dispatch aggCopyReports once; fill outFoundMap/outLostMap/outFlags
static void runCopy(PxgSlangComputeDevice* dev, uint32_t nbCreated, uint32_t nbLost, uint32_t maxFL,
                    uint32_t initFound, uint32_t initLost,
                    std::vector<uint32_t>& outFoundMap, std::vector<uint32_t>& outLostMap,
                    std::vector<uint32_t>& outFlags)
{
    std::mt19937 rng(nbCreated * 131 + nbLost + maxFL);
    std::vector<uint32_t> foundSrc(nbCreated * 2), lostSrc(nbLost * 2);
    for (auto& v : foundSrc) v = rng();
    for (auto& v : lostSrc) v = rng();
    std::vector<uint32_t> foundMap(nbCreated * 2, SENT), lostMap(nbLost * 2, SENT);
    std::vector<uint32_t> flags = { initFound, initLost };
    std::vector<uint32_t> dummy(16, 0);

    PxgBufferHandle bFS = createBufferFrom(dev, foundSrc);
    PxgBufferHandle bLS = createBufferFrom(dev, lostSrc);
    PxgBufferHandle bFM = createBufferFrom(dev, foundMap);
    PxgBufferHandle bLM = createBufferFrom(dev, lostMap);
    PxgBufferHandle bFL = createBufferFrom(dev, flags);
    PxgBufferHandle bD = createBufferFrom(dev, dummy);

    const uint32_t gridX = (std::max(nbCreated, nbLost) * 2 + 255) / 256;
    FinalizeParams p = { nbCreated, nbLost, maxFL, 0, 0, 0, gridX };
    Kernel k("broadphaseAggregateFinalize.spv", "aggCopyReports", STRIDES10, sizeof(FinalizeParams));
    k.dispatch(gridX, { bFS, bLS, bFM, bLM, bFL, bD, bD, bD, bD, bD }, p);
    dev->flush();

    // reference
    const uint32_t tC = (nbCreated > maxFL ? maxFL : nbCreated) * 2;
    const uint32_t tL = (nbLost > maxFL ? maxFL : nbLost) * 2;
    std::vector<uint32_t> rF = foundMap, rL = lostMap;
    for (uint32_t i = 0; i < tC; ++i) rF[i] = foundSrc[i];
    for (uint32_t i = 0; i < tL; ++i) rL[i] = lostSrc[i];
    outFoundMap = downloadBuffer<uint32_t>(dev, bFM, nbCreated * 2);
    outLostMap = downloadBuffer<uint32_t>(dev, bLM, nbLost * 2);
    outFlags = downloadBuffer<uint32_t>(dev, bFL, 2);
    // stash references by overwriting the src vectors' meaning: compare here
    size_t fM = 0, lM = 0;
    for (uint32_t i = 0; i < nbCreated * 2; ++i) if (outFoundMap[i] != rF[i]) ++fM;
    for (uint32_t i = 0; i < nbLost * 2; ++i) if (outLostMap[i] != rL[i]) ++lM;
    PXG_CHECK_MSG(fM == 0, "%zu found-report-map mismatches", fM);
    PXG_CHECK_MSG(lM == 0, "%zu lost-report-map mismatches", lM);
    for (PxgBufferHandle h : { bFS, bLS, bFM, bLM, bFL, bD }) dev->destroyBuffer(h);
}

PXG_TEST(broadphaseAggregateFinalize_copyReports)
{
    PxgSlangComputeDevice* dev = testDevice();
    std::vector<uint32_t> fm, lm, flags;

    // Scenario 1: both overflow -> found flag OR'd to 1, lost flag assigned 1, copies clamped.
    runCopy(dev, 1500, 1500, 1000, 0, 0, fm, lm, flags);
    PXG_CHECK_MSG(flags[0] == 1 && flags[1] == 1, "S1 flags {%u,%u} exp {1,1}", flags[0], flags[1]);

    // Scenario 2: no overflow, init flags {1,1} -> found stays 1 (|=), lost becomes 0 (=), full copies.
    runCopy(dev, 500, 500, 1000, 1, 1, fm, lm, flags);
    PXG_CHECK_MSG(flags[0] == 1 && flags[1] == 0, "S2 flags {%u,%u} exp {1,0}", flags[0], flags[1]);
}

PXG_TEST(broadphaseAggregateFinalize_removeStage3)
{
    PxgSlangComputeDevice* dev = testDevice();
    std::vector<uint32_t> freeList = { 500, 3000 };
    std::vector<uint32_t> dummy(16, 0);
    PxgBufferHandle bFree = createBufferFrom(dev, freeList);
    PxgBufferHandle bD = createBufferFrom(dev, dummy);

    FinalizeParams p = { 0, 0, 0, 123, 0, 0, 1 };
    Kernel k("broadphaseAggregateFinalize.spv", "removeAggPairsStage3", STRIDES10, sizeof(FinalizeParams));
    k.dispatch(1, { bD, bD, bD, bD, bD, bFree, bD, bD, bD, bD }, p);
    dev->flush();

    const auto gFree = downloadBuffer<uint32_t>(dev, bFree, 2);
    PXG_CHECK_MSG(gFree[0] == 623, "numFreeIndices=%u exp 623", gFree[0]);
    PXG_CHECK_MSG(gFree[1] == 3000, "maxIndex=%u exp 3000 (unchanged)", gFree[1]);
    for (PxgBufferHandle h : { bFree, bD }) dev->destroyBuffer(h);
}

PXG_TEST(broadphaseAggregateFinalize_markBounds)
{
    std::mt19937 rng(27030);
    const uint32_t nSlots = 5000;
    std::vector<uint32_t> newB(nSlots * 6), oldB(nSlots * 6);
    for (uint32_t i = 0; i < nSlots * 6; ++i) { newB[i] = 1000 + i; oldB[i] = 5000000 + i; }

    // added handles in [0,2500), deleted in [2500,5000) — distinct and disjoint
    std::vector<uint32_t> lo(2500), hi(2500);
    std::iota(lo.begin(), lo.end(), 0u);
    std::iota(hi.begin(), hi.end(), 2500u);
    std::shuffle(lo.begin(), lo.end(), rng);
    std::shuffle(hi.begin(), hi.end(), rng);
    lo.resize(800); hi.resize(800);
    const uint32_t numAdded = 800, numDeleted = 800;

    // reference
    std::vector<uint32_t> rNew = newB, rOld = oldB;
    auto empty = [&](std::vector<uint32_t>& v, uint32_t h) {
        v[h * 6 + 0] = E_MIN; v[h * 6 + 1] = E_MIN; v[h * 6 + 2] = E_MIN;
        v[h * 6 + 3] = E_MAX; v[h * 6 + 4] = E_MAX; v[h * 6 + 5] = E_MAX;
    };
    for (uint32_t a = 0; a < numAdded; ++a) empty(rOld, lo[a]);           // added: old only
    for (uint32_t d = 0; d < numDeleted; ++d) { empty(rNew, hi[d]); empty(rOld, hi[d]); } // deleted: new+old

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<uint32_t> dummy(16, 0);
    PxgBufferHandle bNew = createBufferFrom(dev, newB);
    PxgBufferHandle bOld = createBufferFrom(dev, oldB);
    PxgBufferHandle bAdd = createBufferFrom(dev, lo);
    PxgBufferHandle bDel = createBufferFrom(dev, hi);
    PxgBufferHandle bD = createBufferFrom(dev, dummy);

    const uint32_t gridX = (std::max(numAdded, numDeleted) + 255) / 256;
    FinalizeParams p = { 0, 0, 0, 0, numAdded, numDeleted, gridX };
    Kernel k("broadphaseAggregateFinalize.spv", "markAddedAndDeletedAggregatedBounds", STRIDES10, sizeof(FinalizeParams));
    k.dispatch(gridX, { bD, bD, bD, bD, bD, bD, bNew, bOld, bAdd, bDel }, p);
    dev->flush();

    const auto gNew = downloadBuffer<uint32_t>(dev, bNew, nSlots * 6);
    const auto gOld = downloadBuffer<uint32_t>(dev, bOld, nSlots * 6);
    size_t nM = 0, oM = 0;
    for (uint32_t i = 0; i < nSlots * 6; ++i) if (gNew[i] != rNew[i]) ++nM;
    for (uint32_t i = 0; i < nSlots * 6; ++i) if (gOld[i] != rOld[i]) ++oM;
    PXG_CHECK_MSG(nM == 0, "%zu new-bounds mismatches", nM);
    PXG_CHECK_MSG(oM == 0, "%zu old-bounds mismatches", oM);
    for (PxgBufferHandle h : { bNew, bOld, bAdd, bDel, bD }) dev->destroyBuffer(h);
}

} // namespace
