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

// P5 Step 4: empirical layout proof for the PxgBroadPhaseDescGpu mirror.
// Fills a REAL engine PxgBroadPhaseDesc (the exact struct the engine DMAs to
// the device) with distinct sentinels, uploads the raw bytes, and the
// bpDescProbe kernel reads 57 fields spanning the whole struct through the
// Slang mirror.  A single wrong offset — missing padding, wrong array stride,
// engine struct drift — mismatches bitwise.  Every bpDesc-walking kernel
// conversion rests on this proof.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "foundation/PxBounds3.h"
#include "PxgBroadPhaseDesc.h"

#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

static std::vector<uint8_t> bpDescLoader(const char* file) { return loadShaderBlob(file); }

// Distinct 64-bit sentinel per probed pointer field.
static uint64_t sent(uint32_t idx) { return 0xA000000000000100ull + uint64_t(idx) * 0x300007ull; }
// Distinct 32-bit sentinel per probed uint field.
static uint32_t sent32(uint32_t idx) { return 0x50000021u + idx * 0x10009u; }

PXG_TEST(bpDescLayout)
{
    using namespace physx;

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npBpDescProbe.spv", "bpDescProbe", 16u, 0 }, // { u64 desc; u64 out; }
    };
    PxgCudaShim shim(dev, manifest, 1, &bpDescLoader);

    // Fill the REAL engine struct with sentinels (0xEE background catches any
    // field the fill misses).
    PxgBroadPhaseDesc d;
    std::memset(&d, 0xEE, sizeof(d));
    uint32_t p = 0, u = 0;
    d.updateData_createdHandles = reinterpret_cast<PxU32*>(sent(p++));
    d.numCreatedHandles = sent32(u++);
    d.updateData_removedHandles = reinterpret_cast<PxU32*>(sent(p++));
    d.numRemovedHandles = sent32(u++);
    d.updateData_updatedHandles = reinterpret_cast<PxU32*>(sent(p++));
    d.numUpdatedHandles = sent32(u++);
    d.aabbMngr_changedHandleMap = reinterpret_cast<PxU32*>(sent(p++));
    d.aabbMngr_changedHandleBitMapWordCounts = sent32(u++);
    d.aabbMngr_addedHandleMap = reinterpret_cast<PxU32*>(sent(p++));
    d.aabbMngr_aggregatedBoundHandles = reinterpret_cast<PxU32*>(sent(p++));
    d.updateData_fpBounds = reinterpret_cast<PxBounds3*>(sent(p++));
    d.updateData_contactDistances = reinterpret_cast<PxReal*>(sent(p++));
    d.newIntegerBounds = reinterpret_cast<PxgIntegerAABB*>(sent(p++));
    d.updateData_envIDs = reinterpret_cast<PxU32*>(sent(p++));
    d.oldIntegerBounds = reinterpret_cast<PxgIntegerAABB*>(sent(p++));
    d.boxSapBox1D[0] = reinterpret_cast<PxgSapBox1D*>(sent(p++));
    d.boxSapBox1D[2] = reinterpret_cast<PxgSapBox1D*>(sent(p++));
    d.boxNewSapBox1D[1] = reinterpret_cast<PxgSapBox1D*>(sent(p++));
    d.boxProjections[2] = reinterpret_cast<PxU32*>(sent(p++));
    d.boxHandles[0][0] = reinterpret_cast<PxU32*>(sent(p++));
    d.boxHandles[1][2] = reinterpret_cast<PxU32*>(sent(p++));
    d.numPreviousHandles = sent32(u++);
    d.sharedFoundPairIndex = sent32(u++);
    d.sharedLostAggPairIndex = sent32(u++);
    d.foundPairReport = reinterpret_cast<PxgBroadPhasePair*>(sent(p++));
    d.lostActorPairReport = reinterpret_cast<PxgBroadPhasePair*>(sent(p++));
    d.lostPairReportMap = reinterpret_cast<PxgBroadPhasePair*>(sent(p++));
    d.aabbMngr_volumeData = reinterpret_cast<Bp::VolumeData*>(sent(p++));
    d.activeRegionsHistogram = reinterpret_cast<PxU32*>(sent(p++));
    d.orderedStartRegionHandles = reinterpret_cast<PxU32*>(sent(p++));
    d.blockOverlapChecksRegion = reinterpret_cast<regionOverlapType*>(sent(p++));
    d.overlapChecksTotalRegion = sent(p++); // u64 VALUE
    d.overlapChecksHandleRegiones = reinterpret_cast<PxgHandleRegion*>(sent(p++));
    d.regionRange = reinterpret_cast<PxgIntegerRegion*>(sent(p++));
    d.blockStartRegionAccum = reinterpret_cast<PxU32*>(sent(p++));
    d.startRegionAccumTotal = sent32(u++);
    d.regionAccum = reinterpret_cast<PxU32*>(sent(p++));
    d.regionAccumTotal = sent32(u++);
    d.endPtHistogram[0][0] = reinterpret_cast<PxU32*>(sent(p++));
    d.endPtHistogram[1][2] = reinterpret_cast<PxU32*>(sent(p++));
    d.blockEndPtHistogram[1][1] = reinterpret_cast<PxU32*>(sent(p++));
    d.endPointHandles[0][1] = reinterpret_cast<PxU32*>(sent(p++));
    d.totalEndPtHistogram[2] = reinterpret_cast<PxU32*>(sent(p++));
    d.blockTotalEndPtHistogram[0] = reinterpret_cast<PxU32*>(sent(p++));
    d.startPtHistogram[1][0] = reinterpret_cast<PxU32*>(sent(p++));
    d.blockStartPtHistogram[0][2] = reinterpret_cast<PxU32*>(sent(p++));
    d.startPointHandles[1][2] = reinterpret_cast<PxU32*>(sent(p++));
    d.boxProjectionRanks[1] = reinterpret_cast<PxU32*>(sent(p++));
    d.incrementalComparisons[0] = reinterpret_cast<PxU32*>(sent(p++));
    d.incrementalBlockComparisons[2] = reinterpret_cast<PxU32*>(sent(p++));
    d.aggReportBlock[1] = reinterpret_cast<PxU32*>(sent(p++));
    d.actorReportBlock[0] = reinterpret_cast<PxU32*>(sent(p++));
    d.totalIncrementalComparisons[0] = sent32(u++);
    d.totalIncrementalComparisons[2] = sent32(u++);
    d.max_found_lost_pairs = sent32(u++);
    d.max_found_lost_agg_pairs = sent32(u++);
    d.found_lost_pairs_overflow_flags = true;

    // Expected values in EXACTLY the probe kernel's write order.
    std::vector<uint64_t> expect;
    expect.push_back(uint64_t(d.updateData_createdHandles));
    expect.push_back(d.numCreatedHandles);
    expect.push_back(uint64_t(d.updateData_removedHandles));
    expect.push_back(d.numRemovedHandles);
    expect.push_back(uint64_t(d.updateData_updatedHandles));
    expect.push_back(d.numUpdatedHandles);
    expect.push_back(uint64_t(d.aabbMngr_changedHandleMap));
    expect.push_back(d.aabbMngr_changedHandleBitMapWordCounts);
    expect.push_back(uint64_t(d.aabbMngr_addedHandleMap));
    expect.push_back(uint64_t(d.aabbMngr_aggregatedBoundHandles));
    expect.push_back(uint64_t(d.updateData_fpBounds));
    expect.push_back(uint64_t(d.updateData_contactDistances));
    expect.push_back(uint64_t(d.newIntegerBounds));
    expect.push_back(uint64_t(d.updateData_envIDs));
    expect.push_back(uint64_t(d.oldIntegerBounds));
    expect.push_back(uint64_t(d.boxSapBox1D[0]));
    expect.push_back(uint64_t(d.boxSapBox1D[2]));
    expect.push_back(uint64_t(d.boxNewSapBox1D[1]));
    expect.push_back(uint64_t(d.boxProjections[2]));
    expect.push_back(uint64_t(d.boxHandles[0][0]));
    expect.push_back(uint64_t(d.boxHandles[1][2]));
    expect.push_back(d.numPreviousHandles);
    expect.push_back(d.sharedFoundPairIndex);
    expect.push_back(d.sharedLostAggPairIndex);
    expect.push_back(uint64_t(d.foundPairReport));
    expect.push_back(uint64_t(d.lostActorPairReport));
    expect.push_back(uint64_t(d.lostPairReportMap));
    expect.push_back(uint64_t(d.aabbMngr_volumeData));
    expect.push_back(uint64_t(d.activeRegionsHistogram));
    expect.push_back(uint64_t(d.orderedStartRegionHandles));
    expect.push_back(uint64_t(d.blockOverlapChecksRegion));
    expect.push_back(d.overlapChecksTotalRegion);
    expect.push_back(uint64_t(d.overlapChecksHandleRegiones));
    expect.push_back(uint64_t(d.regionRange));
    expect.push_back(uint64_t(d.blockStartRegionAccum));
    expect.push_back(d.startRegionAccumTotal);
    expect.push_back(uint64_t(d.regionAccum));
    expect.push_back(d.regionAccumTotal);
    expect.push_back(uint64_t(d.endPtHistogram[0][0]));
    expect.push_back(uint64_t(d.endPtHistogram[1][2]));
    expect.push_back(uint64_t(d.blockEndPtHistogram[1][1]));
    expect.push_back(uint64_t(d.endPointHandles[0][1]));
    expect.push_back(uint64_t(d.totalEndPtHistogram[2]));
    expect.push_back(uint64_t(d.blockTotalEndPtHistogram[0]));
    expect.push_back(uint64_t(d.startPtHistogram[1][0]));
    expect.push_back(uint64_t(d.blockStartPtHistogram[0][2]));
    expect.push_back(uint64_t(d.startPointHandles[1][2]));
    expect.push_back(uint64_t(d.boxProjectionRanks[1]));
    expect.push_back(uint64_t(d.incrementalComparisons[0]));
    expect.push_back(uint64_t(d.incrementalBlockComparisons[2]));
    expect.push_back(uint64_t(d.aggReportBlock[1]));
    expect.push_back(uint64_t(d.actorReportBlock[0]));
    expect.push_back(d.totalIncrementalComparisons[0]);
    expect.push_back(d.totalIncrementalComparisons[2]);
    expect.push_back(d.max_found_lost_pairs);
    expect.push_back(d.max_found_lost_agg_pairs);
    expect.push_back(1u); // found_lost_pairs_overflow_flags byte

    PxgDevicePtr dDesc = 0, dOut = 0;
    PXG_CHECK_MSG(shim.memAlloc(&dDesc, sizeof(d)) == PxgResult::Success, "memAlloc desc");
    PXG_CHECK_MSG(shim.memAlloc(&dOut, expect.size() * 8) == PxgResult::Success, "memAlloc out");
    if (!dDesc || !dOut)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    shim.memcpyHtoD(dDesc, &d, sizeof(d));
    shim.memsetD32(dOut, 0, expect.size() * 2);

    struct ProbeParams { uint64_t desc, out; } prm = { dDesc, dOut };
    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success, "launch probe");
    shim.streamSynchronize();

    std::vector<uint64_t> got(expect.size(), 0);
    shim.memcpyDtoH(got.data(), dOut, got.size() * 8);

    size_t bad = 0;
    for (size_t i = 0; i < expect.size(); ++i)
        if (got[i] != expect[i])
        {
            if (bad < 10)
                PXG_CHECK_MSG(false, "probe[%zu]: gpu=%016llx expected=%016llx", i,
                              (unsigned long long)got[i], (unsigned long long)expect[i]);
            ++bad;
        }
    PXG_CHECK_MSG(bad == 0, "PxgBroadPhaseDesc mirror layout is byte-exact (%zu bad of %zu)", bad, expect.size());

    shim.memFree(dDesc);
    shim.memFree(dOut);
}

} // namespace
