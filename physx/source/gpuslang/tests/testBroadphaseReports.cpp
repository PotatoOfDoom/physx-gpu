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
// reports to the mapped output.  P5 Step 4: driven through PxgCudaShim on the
// ENGINE ABI — a REAL engine PxgBroadPhaseDesc is filled with live device
// addresses (pair counts in sharedFound/LostPairIndex, the per-category block
// arrays behind aggReportBlock[2]/actorReportBlock[2], the Bp::VolumeData
// array) and all three kernels launch with the one bpDesc pointer on the
// engine grids (32x4 for the two accumulate stages, 64 for copyReports).
// The desc scalar outputs (sharedFoundAggPairIndex / sharedLostAggPairIndex
// written by stage 2, found_lost_pairs_overflow_flags written by copyReports)
// are verified by reading the desc back.

#include "pxgTestFramework.h"
#include "pxgGpuStructs.h"
#include "PxgCudaShim.h"

#include "foundation/PxBounds3.h"
#include "PxgBroadPhaseDesc.h"

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
// mirrors Bp::VolumeData (16 bytes): { void* mUserData; PxU32 mAggregate; pad }
struct VolumeData
{
    uint32_t userLo, userHi;
    uint32_t mAggregate;
    uint32_t pad;
};

static std::vector<uint8_t> repShimLoader(const char* file) { return loadShaderBlob(file); }

// Fills a REAL engine PxgBroadPhaseDesc with live device addresses (same rig
// as testBroadphaseSetup.cpp).
struct BpDescRig
{
    physx::gpuslang::PxgCudaShim* shim;
    physx::PxgBroadPhaseDesc desc;

    explicit BpDescRig(physx::gpuslang::PxgCudaShim* s) : shim(s) { std::memset(&desc, 0, sizeof(desc)); }

    physx::gpuslang::PxgDevicePtr alloc(const void* data, size_t bytes)
    {
        physx::gpuslang::PxgDevicePtr p = 0;
        shim->memAlloc(&p, bytes);
        if (p && data)
            shim->memcpyHtoD(p, data, bytes);
        return p;
    }
    physx::gpuslang::PxgDevicePtr upload()
    {
        physx::gpuslang::PxgDevicePtr p = alloc(&desc, sizeof(desc));
        return p;
    }
};

PXG_TEST(broadphaseReports_accumulateAndCopy)
{
    using namespace physx;
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

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
    const PxgKernelDef manifest[] = {
        { "broadphaseReports.spv", "accumulateReportsStage_1", 8u, 0 },
        { "broadphaseReports.spv", "accumulateReportsStage_2", 8u, 0 },
        { "broadphaseReports.spv", "copyReports", 8u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 3, &repShimLoader);

    std::vector<uint32_t> blockZero(32, 0);
    std::vector<BpPair> foundAgg(numFound), foundActor(numFound), lostAgg(numLost), lostActor(numLost);
    std::memset(foundAgg.data(), 0xcd, foundAgg.size() * sizeof(BpPair));
    std::memset(foundActor.data(), 0xcd, foundActor.size() * sizeof(BpPair));
    std::memset(lostAgg.data(), 0xcd, lostAgg.size() * sizeof(BpPair));
    std::memset(lostActor.data(), 0xcd, lostActor.size() * sizeof(BpPair));
    std::vector<uint32_t> foundMap(numFound * 2, 0xabababu), lostMap(numLost * 2, 0xcdcdcdu);

    BpDescRig rig(&shim);
    const PxgDevicePtr dFp = rig.alloc(foundPairs.data(), foundPairs.size() * sizeof(BpPair));
    const PxgDevicePtr dLp = rig.alloc(lostPairs.data(), lostPairs.size() * sizeof(BpPair));
    const PxgDevicePtr dVd = rig.alloc(volumes.data(), volumes.size() * sizeof(VolumeData));
    const PxgDevicePtr dAbF = rig.alloc(blockZero.data(), blockZero.size() * 4); // aggReportBlock[0] (found)
    const PxgDevicePtr dAbL = rig.alloc(blockZero.data(), blockZero.size() * 4); // aggReportBlock[1] (lost)
    const PxgDevicePtr dCbF = rig.alloc(blockZero.data(), blockZero.size() * 4); // actorReportBlock[0]
    const PxgDevicePtr dCbL = rig.alloc(blockZero.data(), blockZero.size() * 4); // actorReportBlock[1]
    const PxgDevicePtr dFa = rig.alloc(foundAgg.data(), foundAgg.size() * sizeof(BpPair));
    const PxgDevicePtr dFc = rig.alloc(foundActor.data(), foundActor.size() * sizeof(BpPair));
    const PxgDevicePtr dLa = rig.alloc(lostAgg.data(), lostAgg.size() * sizeof(BpPair));
    const PxgDevicePtr dLc = rig.alloc(lostActor.data(), lostActor.size() * sizeof(BpPair));
    const PxgDevicePtr dFm = rig.alloc(foundMap.data(), foundMap.size() * 4);
    const PxgDevicePtr dLm = rig.alloc(lostMap.data(), lostMap.size() * 4);
    if (!dFp) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    rig.desc.sharedFoundPairIndex = numFound;
    rig.desc.sharedLostPairIndex = numLost;
    rig.desc.max_found_lost_pairs = maxPairs;
    rig.desc.max_found_lost_agg_pairs = maxAggPairs;
    rig.desc.foundPairReport = reinterpret_cast<PxgBroadPhasePair*>(dFp);
    rig.desc.lostPairReport = reinterpret_cast<PxgBroadPhasePair*>(dLp);
    rig.desc.aabbMngr_volumeData = reinterpret_cast<Bp::VolumeData*>(dVd);
    rig.desc.aggReportBlock[0] = reinterpret_cast<PxU32*>(dAbF);
    rig.desc.aggReportBlock[1] = reinterpret_cast<PxU32*>(dAbL);
    rig.desc.actorReportBlock[0] = reinterpret_cast<PxU32*>(dCbF);
    rig.desc.actorReportBlock[1] = reinterpret_cast<PxU32*>(dCbL);
    rig.desc.foundAggPairReport = reinterpret_cast<PxgBroadPhasePair*>(dFa);
    rig.desc.foundActorPairReport = reinterpret_cast<PxgBroadPhasePair*>(dFc);
    rig.desc.lostAggPairReport = reinterpret_cast<PxgBroadPhasePair*>(dLa);
    rig.desc.lostActorPairReport = reinterpret_cast<PxgBroadPhasePair*>(dLc);
    rig.desc.foundPairReportMap = reinterpret_cast<PxgBroadPhasePair*>(dFm);
    rig.desc.lostPairReportMap = reinterpret_cast<PxgBroadPhasePair*>(dLm);
    // stage 2 / copyReports must overwrite these desc outputs
    rig.desc.sharedFoundAggPairIndex = 0xffffffffu;
    rig.desc.sharedLostAggPairIndex = 0xffffffffu;
    rig.desc.found_lost_pairs_overflow_flags = true;
    const PxgDevicePtr dDesc = rig.upload();

    struct { uint64_t bpDesc; } prm = { dDesc };
    // engine grids: BP_COMPUTE_INCREMENTAL_CMP_COUNTS1/2 = 32 x 4 (blockIdx.y =
    // the 4 report categories), BP_COPY_REPORTS = 64 x 256
    PXG_CHECK_MSG(shim.launchKernel(0, 32, 4, 1, &prm, sizeof(prm)) == PxgResult::Success, "launch stage 1");
    dev->barrier();
    PXG_CHECK_MSG(shim.launchKernel(1, 32, 4, 1, &prm, sizeof(prm)) == PxgResult::Success, "launch stage 2");
    dev->barrier();
    PXG_CHECK_MSG(shim.launchKernel(2, 64, 1, 1, &prm, sizeof(prm)) == PxgResult::Success, "launch copyReports");
    shim.streamSynchronize();

    // ---- reference ----
    std::vector<BpPair> rFoundAgg, rFoundActor, rLostAgg, rLostActor;
    for (const BpPair& p2 : foundPairs)
        (isAgg(p2) ? rFoundAgg : rFoundActor).push_back(p2);
    for (const BpPair& p2 : lostPairs)
        (isAgg(p2) ? rLostAgg : rLostActor).push_back(p2);

    auto eqPair = [](const BpPair& a, const BpPair& b) { return a.mVolA == b.mVolA && a.mVolB == b.mVolB; };

    std::vector<BpPair> gFoundAgg(numFound), gFoundActor(numFound), gLostAgg(numLost), gLostActor(numLost);
    shim.memcpyDtoH(gFoundAgg.data(), dFa, numFound * sizeof(BpPair));
    shim.memcpyDtoH(gFoundActor.data(), dFc, numFound * sizeof(BpPair));
    shim.memcpyDtoH(gLostAgg.data(), dLa, numLost * sizeof(BpPair));
    shim.memcpyDtoH(gLostActor.data(), dLc, numLost * sizeof(BpPair));

    // desc scalar outputs: agg totals from stage 2, overflow flag from copyReports
    PxgBroadPhaseDesc back;
    std::memset(&back, 0, sizeof(back));
    PXG_CHECK_MSG(shim.memcpyDtoH(&back, dDesc, sizeof(back)) == PxgResult::Success, "DtoH desc");

    PXG_CHECK_MSG(back.sharedFoundAggPairIndex == rFoundAgg.size(), "found agg count: gpu=%u ref=%zu",
                  back.sharedFoundAggPairIndex, rFoundAgg.size());
    PXG_CHECK_MSG(back.sharedLostAggPairIndex == rLostAgg.size(), "lost agg count: gpu=%u ref=%zu",
                  back.sharedLostAggPairIndex, rLostAgg.size());

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
    std::vector<uint32_t> gFoundMap(numFound * 2), gLostMap(numLost * 2);
    shim.memcpyDtoH(gFoundMap.data(), dFm, numFound * 2 * 4);
    shim.memcpyDtoH(gLostMap.data(), dLm, numLost * 2 * 4);

    PXG_CHECK_MSG(back.found_lost_pairs_overflow_flags == false, "overflow flag should be false, got %u",
                  uint32_t(back.found_lost_pairs_overflow_flags));

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

    for (PxgDevicePtr pfree : { dFp, dLp, dVd, dAbF, dAbL, dCbF, dCbL, dFa, dFc, dLa, dLc, dFm, dLm, dDesc })
        shim.memFree(pfree);
}

} // namespace
