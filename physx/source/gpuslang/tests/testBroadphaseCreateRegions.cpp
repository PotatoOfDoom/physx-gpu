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

// Verification of broadphaseCreateRegions.slang (createRegionsKernel) against
// a C++ transcription of the CUDA kernel — P5 Step 4: driven through
// PxgCudaShim on the ENGINE ABI.  A REAL PxgBroadPhaseDesc is filled with live
// device addresses (numHandles split across numPreviousHandles/
// numCreatedHandles/numRemovedHandles exactly as the .cu recombines them) and
// launched with the engine's fixed 32x256 grid.  A consistent synthetic SAP
// arrangement is built per axis (a permutation of the 2N endpoint positions,
// each box owning a min/max position pair, start projection at the min
// position) so the startRegionAccum writes partition the projection space with
// no collisions.  Every output array (regionRange, startRegionAccum,
// regionAccum) is compared bit-exactly.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "foundation/PxBounds3.h"
#include "PxgBroadPhaseDesc.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

const uint32_t REGION_SIZE_PER_AXIS = 4;
const uint32_t SENT = 0xdeadbeefu;

bool isStart(uint32_t sh) { return (sh & 1u) != 0u; }
uint32_t getHandle(uint32_t sh) { return sh >> 3; }

static std::vector<uint8_t> bpShimLoader(const char* file) { return loadShaderBlob(file); }

// Fills a REAL engine PxgBroadPhaseDesc with live device addresses and uploads
// it (same helper as testBroadphaseSetup.cpp).
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

void runCreateRegionsCase(const char* tag, uint32_t seed, uint32_t N)
{
    using namespace physx;
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    std::mt19937 rng(seed);
    const uint32_t numProj = N * 2;
    const uint32_t eltsPerReg = (N + (REGION_SIZE_PER_AXIS - 1)) / REGION_SIZE_PER_AXIS;

    // per-axis SAP arrangement: permute the 2N endpoint positions, box b owns
    // positions (min,max); start projection sits at the min position.
    std::vector<uint32_t> handlesX(numProj);
    std::vector<uint32_t> sapBox(3 * N * 2); // [axis][handle][2]
    for (uint32_t ax = 0; ax < 3; ++ax)
    {
        std::vector<uint32_t> perm(numProj);
        std::iota(perm.begin(), perm.end(), 0u);
        std::shuffle(perm.begin(), perm.end(), rng);
        for (uint32_t b = 0; b < N; ++b)
        {
            uint32_t a0 = perm[2 * b], a1 = perm[2 * b + 1];
            uint32_t mn = std::min(a0, a1), mx = std::max(a0, a1);
            sapBox[(ax * N + b) * 2 + 0] = mn;
            sapBox[(ax * N + b) * 2 + 1] = mx;
            if (ax == 0)
            {
                handlesX[mn] = (b << 3) | 1u; // start projection at min position
                handlesX[mx] = (b << 3) | 0u; // end projection at max position
            }
        }
    }

    // totalEndPtHistogram[axis]: non-decreasing prefix of random {0,1} counts
    std::vector<uint32_t> endHist(3 * numProj);
    for (uint32_t ax = 0; ax < 3; ++ax)
    {
        uint32_t acc = 0;
        for (uint32_t i = 0; i < numProj; ++i)
        {
            endHist[ax * numProj + i] = acc;
            acc += (rng() & 1u);
        }
    }

    // ---- reference ----
    std::vector<uint32_t> refRange(N * 8, SENT), refStart(numProj, SENT), refRegion(numProj, SENT);
    for (uint32_t workIndex = 0; workIndex < numProj; ++workIndex)
    {
        const uint32_t sortedHandle = handlesX[workIndex];
        uint32_t regionCount = 0;
        if (isStart(sortedHandle))
        {
            const uint32_t h = getHandle(sortedHandle);
            const uint32_t sX = sapBox[(0 * N + h) * 2 + 0], eX = sapBox[(0 * N + h) * 2 + 1];
            const uint32_t sRegX = endHist[0 * numProj + sX] / eltsPerReg;
            const uint32_t eRegX = endHist[0 * numProj + eX] / eltsPerReg;
            const uint32_t sRegY = endHist[1 * numProj + sapBox[(1 * N + h) * 2 + 0]] / eltsPerReg;
            const uint32_t eRegY = endHist[1 * numProj + sapBox[(1 * N + h) * 2 + 1]] / eltsPerReg;
            const uint32_t sRegZ = endHist[2 * numProj + sapBox[(2 * N + h) * 2 + 0]] / eltsPerReg;
            const uint32_t eRegZ = endHist[2 * numProj + sapBox[(2 * N + h) * 2 + 1]] / eltsPerReg;

            refRange[h * 8 + 0] = sRegX;
            refRange[h * 8 + 1] = sRegY;
            refRange[h * 8 + 2] = sRegZ;
            refRange[h * 8 + 3] = 0;
            refRange[h * 8 + 4] = eRegX;
            refRange[h * 8 + 5] = eRegY;
            refRange[h * 8 + 6] = eRegZ;
            refRange[h * 8 + 7] = 0;

            const uint32_t yz = (eRegY - sRegY + 1) * (eRegZ - sRegZ + 1);
            refStart[sX] = yz;
            refStart[eX] = 0;
            if (eRegX >= sRegX)
                regionCount = (eRegX - sRegX + 1) * yz;
        }
        refRegion[workIndex] = regionCount;
    }

    // ---- device: real PxgBroadPhaseDesc + engine launch ----
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = { { "broadphaseCreateRegions.spv", "createRegionsKernel", 8u, 0 } };
    PxgCudaShim shim(dev, manifest, 1, &bpShimLoader);

    std::vector<uint32_t> rangeInit(N * 8, SENT), startInit(numProj, SENT), regionInit(numProj, SENT);

    BpDescRig rig(&shim);
    const PxgDevicePtr dEnd = rig.alloc(endHist.data(), endHist.size() * 4);     // 3 axes, contiguous
    const PxgDevicePtr dHand = rig.alloc(handlesX.data(), handlesX.size() * 4);
    const PxgDevicePtr dSap = rig.alloc(sapBox.data(), sapBox.size() * 4);       // 3 axes, contiguous
    const PxgDevicePtr dRange = rig.alloc(rangeInit.data(), rangeInit.size() * 4);
    const PxgDevicePtr dStart = rig.alloc(startInit.data(), startInit.size() * 4);
    const PxgDevicePtr dRegion = rig.alloc(regionInit.data(), regionInit.size() * 4);
    if (!dEnd) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    // The kernel recombines numHandles = numPreviousHandles + numCreatedHandles
    // - numRemovedHandles; split N across the three fields to exercise that.
    const uint32_t numCreated = std::min(N, 25u);
    const uint32_t numRemoved = std::min(numCreated, 10u);
    rig.desc.numPreviousHandles = N - numCreated + numRemoved;
    rig.desc.numCreatedHandles = numCreated;
    rig.desc.numRemovedHandles = numRemoved;
    for (uint32_t ax = 0; ax < 3; ++ax)
    {
        rig.desc.totalEndPtHistogram[ax] = reinterpret_cast<PxU32*>(dEnd + ax * numProj * 4);
        rig.desc.boxSapBox1D[ax] = reinterpret_cast<PxgSapBox1D*>(dSap + ax * N * 2 * 4);
    }
    rig.desc.boxHandles[0][0] = reinterpret_cast<PxU32*>(dHand);
    rig.desc.regionRange = reinterpret_cast<PxgIntegerRegion*>(dRange);
    rig.desc.startRegionAccum = reinterpret_cast<PxU32*>(dStart);
    rig.desc.regionAccum = reinterpret_cast<PxU32*>(dRegion);
    const PxgDevicePtr dDesc = rig.upload();

    struct { uint64_t bpDesc; } prm = { dDesc };
    // engine grid: BP_CREATE_REGIONS = 32 blocks x 256
    PXG_CHECK_MSG(shim.launchKernel(0, 32, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "[%s N=%u] launch createRegionsKernel", tag, N);
    shim.streamSynchronize();

    std::vector<uint32_t> gRange(N * 8), gStart(numProj), gRegion(numProj);
    shim.memcpyDtoH(gRange.data(), dRange, gRange.size() * 4);
    shim.memcpyDtoH(gStart.data(), dStart, gStart.size() * 4);
    shim.memcpyDtoH(gRegion.data(), dRegion, gRegion.size() * 4);

    size_t rangeMism = 0, startMism = 0, regionMism = 0;
    uint32_t firstBad = 0xffffffffu, badG = 0, badR = 0;
    for (uint32_t i = 0; i < N * 8; ++i)
        if (gRange[i] != refRange[i]) ++rangeMism;
    for (uint32_t i = 0; i < numProj; ++i)
    {
        if (gStart[i] != refStart[i]) ++startMism;
        if (gRegion[i] != refRegion[i])
        {
            if (firstBad == 0xffffffffu) { firstBad = i; badG = gRegion[i]; badR = refRegion[i]; }
            ++regionMism;
        }
    }

    PXG_CHECK_MSG(rangeMism == 0, "[%s N=%u] %zu regionRange mismatches", tag, N, rangeMism);
    PXG_CHECK_MSG(startMism == 0, "[%s N=%u] %zu startRegionAccum mismatches", tag, N, startMism);
    if (regionMism)
        PXG_CHECK_MSG(regionMism == 0, "[%s N=%u] %zu regionAccum mismatches (first at %u: gpu=%u ref=%u)", tag, N,
                      regionMism, firstBad, badG, badR);
    else
        PXG_CHECK_MSG(regionMism == 0, "[%s N=%u] %zu regionAccum mismatches", tag, N, regionMism);

    for (PxgDevicePtr pfree : { dEnd, dHand, dSap, dRange, dStart, dRegion, dDesc })
        shim.memFree(pfree);
}

PXG_TEST(broadphaseCreateRegions_regionCounts)
{
    runCreateRegionsCase("small-300", 18010, 300);        // single partition tile
    runCreateRegionsCase("multi-iter-5000", 18011, 5000); // numProj=10000 -> 2 iterations/block
}

} // namespace
