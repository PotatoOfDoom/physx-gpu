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

// Verification of broadphaseReports.slang against a C++ transcription of the
// broad-phase found/lost pair report accumulation pipeline: classify each pair
// as aggregate vs actor, stably compact the four categories, and copy the actor
// reports to the mapped output.

#include "pxgTestFramework.h"
#include "pxgGpuStructs.h"

#include <algorithm>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

const uint32_t INVALID_U32 = 0xffffffffu;

struct BpPair
{
    uint32_t mVolA;
    uint32_t mVolB;
};
struct VolumeData
{
    uint32_t userLo, userHi;
    uint32_t mAggregate;
    uint32_t pad;
};

struct ReportParams
{
    uint32_t numFound;
    uint32_t numLost;
    uint32_t maxPairs;
    uint32_t maxAggPairs;
    uint32_t hasVolumeData;
};

PXG_TEST(broadphaseReports_accumulateAndCopy)
{
    std::mt19937 rng(7100);
    const uint32_t numVolumes = 4096;
    const uint32_t numFound = 5300;
    const uint32_t numLost = 2900;
    const uint32_t maxPairs = 1u << 20;
    const uint32_t maxAggPairs = 1u << 20;

    // volume data: ~55% single actors, rest aggregate/aggregated
    std::vector<VolumeData> volumes(numVolumes);
    std::memset(volumes.data(), 0, volumes.size() * sizeof(VolumeData));
    for (uint32_t v = 0; v < numVolumes; ++v)
    {
        if (rng() % 100 < 55)
            volumes[v].mAggregate = INVALID_U32; // single actor
        else
            volumes[v].mAggregate = (rng() % 500) << 1 | (rng() & 1); // aggregate or aggregated
    }

    auto makePairs = [&](uint32_t n) {
        std::vector<BpPair> pairs(n);
        for (uint32_t i = 0; i < n; ++i)
        {
            uint32_t a = rng() % numVolumes, b = rng() % numVolumes;
            while (b == a) b = rng() % numVolumes;
            pairs[i].mVolA = std::min(a, b);
            pairs[i].mVolB = std::max(a, b);
        }
        return pairs;
    };
    std::vector<BpPair> foundPairs = makePairs(numFound);
    std::vector<BpPair> lostPairs = makePairs(numLost);

    auto isSingle = [&](uint32_t v) { return volumes[v].mAggregate == INVALID_U32; };
    auto isAgg = [&](const BpPair& p) { return !(isSingle(p.mVolA) && isSingle(p.mVolB)); };

    PxgSlangComputeDevice* dev = testDevice();

    std::vector<uint32_t> aggBlock(2 * 32, 0), actorBlock(2 * 32, 0);
    std::vector<BpPair> foundAgg(numFound), foundActor(numFound), lostAgg(numLost), lostActor(numLost);
    std::memset(foundAgg.data(), 0xcd, foundAgg.size() * sizeof(BpPair));
    std::memset(foundActor.data(), 0xcd, foundActor.size() * sizeof(BpPair));
    std::memset(lostAgg.data(), 0xcd, lostAgg.size() * sizeof(BpPair));
    std::memset(lostActor.data(), 0xcd, lostActor.size() * sizeof(BpPair));
    std::vector<uint32_t> sharedAgg(2, 0xffffffffu);
    std::vector<uint32_t> foundMap(numFound * 2, 0xabababu), lostMap(numLost * 2, 0xcdcdcdu);
    std::vector<uint32_t> overflow(1, 0xffffffffu);
    std::vector<uint32_t> dummy(16, 0);

    PxgBufferHandle fpBuf = createBufferFrom(dev, foundPairs);
    PxgBufferHandle lpBuf = createBufferFrom(dev, lostPairs);
    PxgBufferHandle vdBuf = createBufferFrom(dev, volumes);
    PxgBufferHandle abBuf = createBufferFrom(dev, aggBlock);
    PxgBufferHandle acBuf = createBufferFrom(dev, actorBlock);
    PxgBufferHandle faBuf = createBufferFrom(dev, foundAgg);
    PxgBufferHandle fcBuf = createBufferFrom(dev, foundActor);
    PxgBufferHandle laBuf = createBufferFrom(dev, lostAgg);
    PxgBufferHandle lcBuf = createBufferFrom(dev, lostActor);
    PxgBufferHandle saBuf = createBufferFrom(dev, sharedAgg);
    PxgBufferHandle fmBuf = createBufferFrom(dev, foundMap);
    PxgBufferHandle lmBuf = createBufferFrom(dev, lostMap);
    PxgBufferHandle ofBuf = createBufferFrom(dev, overflow);
    PxgBufferHandle dummyBuf = createBufferFrom(dev, dummy);

    ReportParams p = {};
    p.numFound = numFound;
    p.numLost = numLost;
    p.maxPairs = maxPairs;
    p.maxAggPairs = maxAggPairs;
    p.hasVolumeData = 1;

    // stage 1: bindings 0..4
    Kernel k1("broadphaseReports.spv", "accumulateReportsStage_1", { 8, 8, 16, 4, 4 }, sizeof(ReportParams));
    k1.dispatch(32, { fpBuf, lpBuf, vdBuf, abBuf, acBuf }, p, 4 /*gridY = 4 categories*/);
    dev->barrier();

    // stage 2: bindings 0..9
    Kernel k2("broadphaseReports.spv", "accumulateReportsStage_2", { 8, 8, 16, 4, 4, 8, 8, 8, 8, 4 },
              sizeof(ReportParams));
    k2.dispatch(32, { fpBuf, lpBuf, vdBuf, abBuf, acBuf, faBuf, fcBuf, laBuf, lcBuf, saBuf }, p, 4);
    dev->barrier();

    // copyReports: bindings 6,8,9,10,11,12
    Kernel k3("broadphaseReports.spv", "copyReports", { 8, 8, 16, 4, 4, 8, 8, 8, 8, 4, 4, 4, 4 },
              sizeof(ReportParams));
    k3.dispatch(32, { dummyBuf, dummyBuf, dummyBuf, dummyBuf, dummyBuf, dummyBuf, fcBuf, dummyBuf, lcBuf, saBuf,
                      fmBuf, lmBuf, ofBuf }, p);
    dev->flush();

    // ---- reference ----
    std::vector<BpPair> rFoundAgg, rFoundActor, rLostAgg, rLostActor;
    for (const BpPair& p2 : foundPairs)
        (isAgg(p2) ? rFoundAgg : rFoundActor).push_back(p2);
    for (const BpPair& p2 : lostPairs)
        (isAgg(p2) ? rLostAgg : rLostActor).push_back(p2);

    auto eqPair = [](const BpPair& a, const BpPair& b) { return a.mVolA == b.mVolA && a.mVolB == b.mVolB; };

    const auto gFoundAgg = downloadBuffer<BpPair>(dev, faBuf, numFound);
    const auto gFoundActor = downloadBuffer<BpPair>(dev, fcBuf, numFound);
    const auto gLostAgg = downloadBuffer<BpPair>(dev, laBuf, numLost);
    const auto gLostActor = downloadBuffer<BpPair>(dev, lcBuf, numLost);
    const auto gSharedAgg = downloadBuffer<uint32_t>(dev, saBuf, 2);

    PXG_CHECK_MSG(gSharedAgg[0] == rFoundAgg.size(), "found agg count: gpu=%u ref=%zu", gSharedAgg[0], rFoundAgg.size());
    PXG_CHECK_MSG(gSharedAgg[1] == rLostAgg.size(), "lost agg count: gpu=%u ref=%zu", gSharedAgg[1], rLostAgg.size());

    size_t mism = 0;
    for (size_t i = 0; i < rFoundAgg.size(); ++i)
        if (!eqPair(gFoundAgg[i], rFoundAgg[i])) ++mism;
    for (size_t i = 0; i < rFoundActor.size(); ++i)
        if (!eqPair(gFoundActor[i], rFoundActor[i])) ++mism;
    for (size_t i = 0; i < rLostAgg.size(); ++i)
        if (!eqPair(gLostAgg[i], rLostAgg[i])) ++mism;
    for (size_t i = 0; i < rLostActor.size(); ++i)
        if (!eqPair(gLostActor[i], rLostActor[i])) ++mism;
    PXG_CHECK_MSG(mism == 0, "%zu compaction mismatches", mism);

    // ---- copyReports verification ----
    const uint32_t nbCreated = numFound - uint32_t(rFoundAgg.size()); // = actorFound count (no overflow)
    const uint32_t nbLostA = numLost - uint32_t(rLostAgg.size());
    const auto gFoundMap = downloadBuffer<uint32_t>(dev, fmBuf, numFound * 2);
    const auto gLostMap = downloadBuffer<uint32_t>(dev, lmBuf, numLost * 2);
    const auto gOverflow = downloadBuffer<uint32_t>(dev, ofBuf, 1)[0];

    PXG_CHECK_MSG(gOverflow == 0, "overflow flag should be 0, got %u", gOverflow);

    mism = 0;
    for (uint32_t i = 0; i < nbCreated; ++i)
    {
        if (gFoundMap[i * 2 + 0] != rFoundActor[i].mVolA) ++mism;
        if (gFoundMap[i * 2 + 1] != rFoundActor[i].mVolB) ++mism;
    }
    for (uint32_t i = 0; i < nbLostA; ++i)
    {
        if (gLostMap[i * 2 + 0] != rLostActor[i].mVolA) ++mism;
        if (gLostMap[i * 2 + 1] != rLostActor[i].mVolB) ++mism;
    }
    PXG_CHECK_MSG(mism == 0, "%zu copyReports mismatches", mism);

    for (PxgBufferHandle h : { fpBuf, lpBuf, vdBuf, abBuf, acBuf, faBuf, fcBuf, laBuf, lcBuf, saBuf, fmBuf, lmBuf,
                               ofBuf, dummyBuf })
        dev->destroyBuffer(h);
}

} // namespace
