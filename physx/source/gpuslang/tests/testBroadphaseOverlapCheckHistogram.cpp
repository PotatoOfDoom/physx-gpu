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

// Verification of broadphaseOverlapCheckHistogram.slang — P5 Step 4: driven
// through PxgCudaShim on the ENGINE ABI.  Each test fills a REAL engine
// PxgBroadPhaseDesc with live device addresses and launches with the one
// bpDesc pointer + the engine's fixed 32-block grid, exactly as
// PxgCudaBroadPhaseSap::generateNewPairsKernel does.  Pass 1 (the
// per-comparison overlap-check count + handle/region record) is checked
// elementwise against a C++ transcription over a consistent synthetic region
// layout.  Passes 2+3 (the 64-bit device-wide exclusive prefix scan) are
// checked end-to-end against a uint64 exclusive scan whose total deliberately
// exceeds 2^32, proving the 64-bit accumulation path (the exact overflow PhysX
// types these buffers PxU64 for); the grand total is verified by reading the
// desc back (outputOverlapChecksForRegionHistogram writes the u64
// overlapChecksTotalRegion VALUE through the desc pointer).

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "foundation/PxBounds3.h"
#include "PxgBroadPhaseDesc.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

const uint32_t TOTAL_REGION_SIZE = 64;

uint32_t getHandle(uint32_t sh) { return sh >> 3; }
uint32_t isNewP(uint32_t sh) { return (sh >> 1) & 1u; }

uint32_t binarySearch(const std::vector<uint32_t>& data, uint32_t n, uint32_t value)
{
    uint32_t left = 0, right = n;
    while (left < right)
    {
        uint32_t pos = (left + right) / 2;
        if (data[pos] <= value) left = pos + 1;
        else right = pos;
    }
    return left ? left - 1 : 0;
}

static std::vector<uint8_t> ochShimLoader(const char* file) { return loadShaderBlob(file); }

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

// ---- pass 1: elementwise overlap-check counts + (handle, region) records ----
PXG_TEST(broadphaseOverlapCheckHistogram_writeout)
{
    using namespace physx;
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    std::mt19937 rng(20010);
    const uint32_t N = 1500;
    const uint32_t nbProj = N * 2;

    std::vector<uint32_t> regionRange(N * 8, 0), sapBoxX(N * 2), handlesX(nbProj), fullSpan(N);
    {
        std::vector<uint32_t> perm(nbProj);
        std::iota(perm.begin(), perm.end(), 0u);
        std::shuffle(perm.begin(), perm.end(), rng);
        for (uint32_t b = 0; b < N; ++b)
        {
            auto span = [&](uint32_t& s, uint32_t& e) { s = rng() % 4; e = s + rng() % (4 - s); };
            uint32_t sX, eX, sY, eY, sZ, eZ;
            span(sX, eX);
            span(sY, eY);
            span(sZ, eZ);
            regionRange[b * 8 + 0] = sX;
            regionRange[b * 8 + 1] = sY;
            regionRange[b * 8 + 2] = sZ;
            regionRange[b * 8 + 4] = eX;
            regionRange[b * 8 + 5] = eY;
            regionRange[b * 8 + 6] = eZ;
            fullSpan[b] = (eX - sX + 1) * (eY - sY + 1) * (eZ - sZ + 1);
            uint32_t mn = std::min(perm[2 * b], perm[2 * b + 1]);
            uint32_t mx = std::max(perm[2 * b], perm[2 * b + 1]);
            sapBoxX[b * 2 + 0] = mn;
            sapBoxX[b * 2 + 1] = mx;
            const uint32_t newBit = (rng() % 100 < 50) ? 1u : 0u;
            handlesX[mn] = (b << 3) | 1u | (newBit << 1);
            handlesX[mx] = (b << 3) | 0u;
        }
    }

    // regionAccum = exclusive prefix of full-3D-span counts (createRegions + scan)
    std::vector<uint32_t> regionAccum(nbProj);
    uint32_t acc = 0;
    for (uint32_t pos = 0; pos < nbProj; ++pos)
    {
        regionAccum[pos] = acc;
        const uint32_t sh = handlesX[pos];
        if ((sh & 1u) != 0u) acc += fullSpan[getHandle(sh)];
    }
    const uint32_t total = acc;

    // startHist / activeHist offset tables: non-decreasing per region
    std::vector<uint32_t> startHist(TOTAL_REGION_SIZE * nbProj), activeHist(TOTAL_REGION_SIZE * nbProj);
    for (uint32_t r = 0; r < TOTAL_REGION_SIZE; ++r)
    {
        uint32_t sa = 0, aa = 0;
        for (uint32_t i = 0; i < nbProj; ++i)
        {
            startHist[r * nbProj + i] = sa;
            activeHist[r * nbProj + i] = aa;
            sa += rng() & 1u;
            aa += rng() & 1u;
        }
    }

    // reference
    std::vector<uint64_t> refOverlap(total);
    std::vector<uint32_t> refHR(total * 2);
    for (uint32_t wi = 0; wi < total; ++wi)
    {
        const uint32_t bs = binarySearch(regionAccum, nbProj, wi);
        const uint32_t offset = wi - regionAccum[bs];
        const uint32_t sh = handlesX[bs];
        const uint32_t handle = getHandle(sh);
        const uint32_t isNew = isNewP(sh);
        const uint32_t be = sapBoxX[handle * 2 + 1];
        const uint32_t sX = regionRange[handle * 8 + 0], sY = regionRange[handle * 8 + 1],
                       sZ = regionRange[handle * 8 + 2];
        const uint32_t eY = regionRange[handle * 8 + 5], eZ = regionRange[handle * 8 + 6];
        const uint32_t nbZ = (eZ - sZ) + 1, nbY = (eY - sY) + 1;
        const uint32_t zR = (offset % nbZ) + sZ;
        const uint32_t yR = ((offset / nbZ) % nbY) + sY;
        const uint32_t xR = (offset / (nbZ * nbY)) + sX;
        const uint32_t regionIndex = (xR + yR * 4 + zR * 16) * nbProj;
        const std::vector<uint32_t>& tbl = isNew ? startHist : activeHist;
        const uint32_t rStart = tbl[regionIndex + bs + 1];
        const uint32_t rEnd = tbl[regionIndex + be];
        refHR[wi * 2 + 0] = handle;
        refHR[wi * 2 + 1] = regionIndex;
        refOverlap[wi] = uint64_t(rEnd - rStart);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "broadphaseOverlapCheckHistogram.spv", "writeOutOverlapChecksForInsertedBoundsRegionsHistogram", 8u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &ochShimLoader);

    std::vector<uint64_t> overlapInit(total, 0xdeadbeefdeadbeefull);
    std::vector<uint32_t> hrInit(total * 2, 0xcdcdcdcdu);

    BpDescRig rig(&shim);
    const PxgDevicePtr dAcc = rig.alloc(regionAccum.data(), regionAccum.size() * 4);
    const PxgDevicePtr dH = rig.alloc(handlesX.data(), handlesX.size() * 4);
    const PxgDevicePtr dSap = rig.alloc(sapBoxX.data(), sapBoxX.size() * 4);
    const PxgDevicePtr dRange = rig.alloc(regionRange.data(), regionRange.size() * 4);
    const PxgDevicePtr dSH = rig.alloc(startHist.data(), startHist.size() * 4);
    const PxgDevicePtr dAH = rig.alloc(activeHist.data(), activeHist.size() * 4);
    const PxgDevicePtr dOv = rig.alloc(overlapInit.data(), overlapInit.size() * 8);
    const PxgDevicePtr dHR = rig.alloc(hrInit.data(), hrInit.size() * 4);
    if (!dAcc) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    // numHandles = numPreviousHandles + numCreatedHandles - numRemovedHandles = N
    rig.desc.numPreviousHandles = N;
    rig.desc.regionAccum = reinterpret_cast<PxU32*>(dAcc);
    rig.desc.boxHandles[0][0] = reinterpret_cast<PxU32*>(dH);
    rig.desc.boxSapBox1D[0] = reinterpret_cast<PxgSapBox1D*>(dSap);
    rig.desc.regionRange = reinterpret_cast<PxgIntegerRegion*>(dRange);
    rig.desc.startRegionsHistogram = reinterpret_cast<PxU32*>(dSH);
    rig.desc.activeRegionsHistogram = reinterpret_cast<PxU32*>(dAH);
    rig.desc.overlapChecksRegion = reinterpret_cast<regionOverlapType*>(dOv);
    rig.desc.overlapChecksHandleRegiones = reinterpret_cast<PxgHandleRegion*>(dHR);
    rig.desc.regionAccumTotal = total;
    const PxgDevicePtr dDesc = rig.upload();

    struct { uint64_t bpDesc; } prm = { dDesc };
    // engine grid: BP_WRITEOUT_OVERLAPCHECKS_HISTOGRAM_NEWBOUNDS = 32 blocks x 256
    PXG_CHECK_MSG(shim.launchKernel(0, 32, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch writeOutOverlapChecks");
    shim.streamSynchronize();

    std::vector<uint64_t> gOverlap(total);
    std::vector<uint32_t> gHR(total * 2);
    shim.memcpyDtoH(gOverlap.data(), dOv, total * 8);
    shim.memcpyDtoH(gHR.data(), dHR, total * 2 * 4);

    size_t ovMism = 0, hrMism = 0;
    for (uint32_t i = 0; i < total; ++i)
    {
        if (gOverlap[i] != refOverlap[i]) ++ovMism;
        if (gHR[i * 2] != refHR[i * 2] || gHR[i * 2 + 1] != refHR[i * 2 + 1]) ++hrMism;
    }
    PXG_CHECK_MSG(ovMism == 0, "%zu overlap-count mismatches (total=%u)", ovMism, total);
    PXG_CHECK_MSG(hrMism == 0, "%zu handleRegion mismatches", hrMism);
    PXG_CHECK_MSG(total > 0, "expected nonzero comparisons");

    for (PxgDevicePtr pfree : { dAcc, dH, dSap, dRange, dSH, dAH, dOv, dHR, dDesc })
        shim.memFree(pfree);
}

// ---- passes 2+3: 64-bit device-wide exclusive prefix scan (sum > 2^32) ----
PXG_TEST(broadphaseOverlapCheckHistogram_scan64)
{
    using namespace physx;
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    std::mt19937 rng(20020);
    const uint32_t n = 100000; // regionAccumTotal
    std::vector<uint64_t> counts(n);
    for (uint32_t i = 0; i < n; ++i)
        counts[i] = 50000 + rng() % 10001; // in [50000,60000] -> sum in [5.0e9, 6.0e9] > 2^32

    // reference: 64-bit exclusive prefix + grand total
    std::vector<uint64_t> ref(n);
    uint64_t a = 0;
    for (uint32_t i = 0; i < n; ++i) { ref[i] = a; a += counts[i]; }
    const uint64_t refTotal = a;
    PXG_CHECK_MSG(refTotal > (uint64_t(1) << 32), "test setup: total %llu should exceed 2^32",
                  (unsigned long long) refTotal);

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "broadphaseOverlapCheckHistogram.spv", "computeOverlapChecksForRegionsHistogram", 8u, 0 },
        { "broadphaseOverlapCheckHistogram.spv", "outputOverlapChecksForRegionHistogram", 8u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 2, &ochShimLoader);

    std::vector<uint64_t> blkInit(TOTAL_REGION_SIZE, 0);

    BpDescRig rig(&shim);
    const PxgDevicePtr dOv = rig.alloc(counts.data(), counts.size() * 8);
    const PxgDevicePtr dBlk = rig.alloc(blkInit.data(), blkInit.size() * 8);
    if (!dOv) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    rig.desc.overlapChecksRegion = reinterpret_cast<regionOverlapType*>(dOv);
    rig.desc.blockOverlapChecksRegion = reinterpret_cast<regionOverlapType*>(dBlk);
    rig.desc.regionAccumTotal = n;
    rig.desc.overlapChecksTotalRegion = 0xeeeeeeeeeeeeeeeeull; // proves the desc write happened
    const PxgDevicePtr dDesc = rig.upload();

    struct { uint64_t bpDesc; } prm = { dDesc };
    // engine grids: BP_COMPUTE/OUTPUT_OVERLAPCHECKS_HISTOGRAM = 32 blocks x 256
    PXG_CHECK_MSG(shim.launchKernel(0, 32, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch computeOverlapChecks");
    dev->barrier();
    PXG_CHECK_MSG(shim.launchKernel(1, 32, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch outputOverlapChecks");
    shim.streamSynchronize();

    std::vector<uint64_t> gOverlap(n);
    shim.memcpyDtoH(gOverlap.data(), dOv, n * 8);

    // the grand total lives in the desc itself (u64 VALUE field) — read it back
    PxgBroadPhaseDesc back;
    std::memset(&back, 0, sizeof(back));
    PXG_CHECK_MSG(shim.memcpyDtoH(&back, dDesc, sizeof(back)) == PxgResult::Success, "DtoH desc");
    const uint64_t gTotal = back.overlapChecksTotalRegion;

    size_t mism = 0;
    uint32_t firstBad = 0xffffffffu;
    for (uint32_t i = 0; i < n; ++i)
        if (gOverlap[i] != ref[i]) { if (firstBad == 0xffffffffu) firstBad = i; ++mism; }
    if (mism)
        PXG_CHECK_MSG(mism == 0, "%zu prefix mismatches (first at %u: gpu=%llu ref=%llu)", mism, firstBad,
                      (unsigned long long) gOverlap[firstBad], (unsigned long long) ref[firstBad]);
    else
        PXG_CHECK_MSG(mism == 0, "%zu prefix mismatches", mism);
    PXG_CHECK_MSG(gTotal == refTotal, "total: gpu=%llu ref=%llu", (unsigned long long) gTotal,
                  (unsigned long long) refTotal);

    for (PxgDevicePtr pfree : { dOv, dBlk, dDesc })
        shim.memFree(pfree);
}

} // namespace
