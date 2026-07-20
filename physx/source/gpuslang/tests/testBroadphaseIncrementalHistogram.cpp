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

// P5 Step 4 (BDA, engine launch path): verification of
// broadphaseIncrementalHistogram.slang (computeIncrementalComparisonHistograms
// _Stage1/_Stage2) driven through PxgCudaShim EXACTLY as the engine launches
// them — a REAL PxgBroadPhaseDesc filled with live device addresses, ONE bpDesc
// pointer packed into the {8}-byte push blob, and the engine grid (32 x 256).
// Every per-projection running total, per-block total and the grand total
// (read back from the desc's inline totalIncrementalComparisons[3]) is compared
// bit-exactly against a C++ transcription of the CUDA kernels; a mutation check
// confirms the output prefix buffer was actually written.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "PxgBroadPhaseDesc.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

static std::vector<uint8_t> bpShimLoader(const char* file) { return loadShaderBlob(file); }

struct BpDescRig
{
    physx::gpuslang::PxgCudaShim* shim;
    physx::PxgBroadPhaseDesc desc;
    explicit BpDescRig(physx::gpuslang::PxgCudaShim* s) : shim(s) { std::memset(&desc, 0, sizeof(desc)); }
    physx::gpuslang::PxgDevicePtr alloc(const void* data, size_t bytes)
    {
        physx::gpuslang::PxgDevicePtr p = 0;
        shim->memAlloc(&p, bytes);
        if (p && data) shim->memcpyHtoD(p, data, bytes);
        return p;
    }
    physx::gpuslang::PxgDevicePtr upload() { return alloc(&desc, sizeof(desc)); }
};

PXG_TEST(broadphaseIncrementalHistogram_stage1and2)
{
    using namespace physx;
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    std::mt19937 rng(14000);

    const uint32_t numHandles = 6000;         // -> nbProj = 12000
    const uint32_t nbProj = numHandles * 2;
    const uint32_t NUM_BLOCKS = 32;
    const uint32_t BLOCK = 256;

    // ---- inputs (per axis / per (frame,axis)) ----
    std::vector<uint32_t> ranks[3], handles[3];
    for (uint32_t axis = 0; axis < 3; ++axis)
    {
        ranks[axis].resize(nbProj);
        handles[axis].resize(nbProj);
        for (uint32_t i = 0; i < nbProj; ++i)
        {
            uint32_t rank = (rng() % 100 < 15) ? (nbProj + (rng() % 2000)) : (rng() % nbProj);
            ranks[axis][i] = rank;
            const uint32_t isStart = rng() & 1u;
            const uint32_t isNew = rng() & 1u;
            const uint32_t isDeleted = (rng() % 100 < 10) ? 1u : 0u;
            const uint32_t handle = rng() % 100000u;
            handles[axis][i] = (handle << 3) | isStart | (isNew << 1) | (isDeleted << 2);
        }
    }

    // startPtHistogram[frame][axis] / endPtHistogram[frame][axis]: prefix sums
    std::vector<uint32_t> startHist[2][3], endHist[2][3];
    for (uint32_t frame = 0; frame < 2; ++frame)
        for (uint32_t axis = 0; axis < 3; ++axis)
        {
            startHist[frame][axis].resize(nbProj);
            endHist[frame][axis].resize(nbProj);
            uint32_t accS = 0, accE = 0;
            for (uint32_t i = 0; i < nbProj; ++i)
            {
                startHist[frame][axis][i] = accS;
                endHist[frame][axis][i] = accE;
                accS += (rng() & 1u);
                accE += (rng() & 1u);
            }
        }

    // ---- reference ----
    const uint32_t nbBlocksRequired = (nbProj + BLOCK - 1) / BLOCK;
    const uint32_t numIterationPerBlock = (nbBlocksRequired + NUM_BLOCKS - 1) / NUM_BLOCKS;

    std::vector<uint32_t> refInc[3], refBlock[3];
    std::vector<uint32_t> refTotal(3, 0);
    for (uint32_t axis = 0; axis < 3; ++axis)
    {
        refInc[axis].assign(nbProj, 0);
        refBlock[axis].assign(NUM_BLOCKS, 0);
        std::vector<uint32_t> comp(nbProj, 0);
        for (uint32_t i = 0; i < nbProj; ++i)
        {
            const uint32_t rank = ranks[axis][i];
            const uint32_t sh = handles[axis][i];
            const bool isDownPass = i <= rank;
            const bool isStartHandle = (sh & 1u) != 0u;
            const bool isDeleted = (sh & (1u << 2)) != 0u;
            const uint32_t start = isDownPass ? i : (rank + 1);
            const uint32_t end = isDownPass ? std::min(rank, nbProj - 1) : i;
            uint32_t c = 0;
            if (!isDeleted)
            {
                const uint32_t frame = isDownPass ? 1u : 0u;
                const std::vector<uint32_t>& H = isStartHandle ? endHist[frame][axis] : startHist[frame][axis];
                c = H[end] - H[start];
            }
            comp[i] = c;
        }
        uint32_t acc = 0;
        for (uint32_t i = 0; i < nbProj; ++i) { refInc[axis][i] = acc; acc += comp[i]; }
        refTotal[axis] = acc;
        for (uint32_t b = 0; b < NUM_BLOCKS; ++b)
        {
            const uint32_t s = b * numIterationPerBlock * BLOCK;
            const uint32_t e = std::min((b + 1) * numIterationPerBlock * BLOCK, nbProj);
            uint32_t bt = 0;
            for (uint32_t i = s; i < e; ++i) bt += comp[i];
            refBlock[axis][b] = bt;
        }
    }

    // ---- device ----
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "broadphaseIncrementalHistogram.spv", "computeIncrementalComparisonHistograms_Stage1", 8u, 0 },
        { "broadphaseIncrementalHistogram.spv", "computeIncrementalComparisonHistograms_Stage2", 8u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 2, &bpShimLoader);

    std::vector<uint32_t> incInit(nbProj, 0xdeadbeefu);
    std::vector<uint32_t> blockInit(NUM_BLOCKS, 0xdeadbeefu);

    BpDescRig rig(&shim);
    PxgDevicePtr dRanks[3], dHand[3], dInc[3], dBlock[3], dSH[2][3], dEH[2][3];
    for (uint32_t ax = 0; ax < 3; ++ax)
    {
        dRanks[ax] = rig.alloc(ranks[ax].data(), nbProj * 4);
        dHand[ax] = rig.alloc(handles[ax].data(), nbProj * 4);
        dInc[ax] = rig.alloc(incInit.data(), nbProj * 4);
        dBlock[ax] = rig.alloc(blockInit.data(), NUM_BLOCKS * 4);
        for (uint32_t fr = 0; fr < 2; ++fr)
        {
            dSH[fr][ax] = rig.alloc(startHist[fr][ax].data(), nbProj * 4);
            dEH[fr][ax] = rig.alloc(endHist[fr][ax].data(), nbProj * 4);
        }
    }
    if (!dRanks[0]) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    rig.desc.numPreviousHandles = numHandles;
    for (uint32_t ax = 0; ax < 3; ++ax)
    {
        rig.desc.boxProjectionRanks[ax] = reinterpret_cast<PxU32*>(dRanks[ax]);
        rig.desc.boxHandles[0][ax] = reinterpret_cast<PxU32*>(dHand[ax]);
        rig.desc.incrementalComparisons[ax] = reinterpret_cast<PxU32*>(dInc[ax]);
        rig.desc.incrementalBlockComparisons[ax] = reinterpret_cast<PxU32*>(dBlock[ax]);
        for (uint32_t fr = 0; fr < 2; ++fr)
        {
            rig.desc.startPtHistogram[fr][ax] = reinterpret_cast<PxU32*>(dSH[fr][ax]);
            rig.desc.endPtHistogram[fr][ax] = reinterpret_cast<PxU32*>(dEH[fr][ax]);
        }
    }
    const PxgDevicePtr dDesc = rig.upload();

    struct { uint64_t bpDesc; } prm = { dDesc };
    PXG_CHECK_MSG(shim.launchKernel(0, 32, 1, 1, &prm, sizeof(prm)) == PxgResult::Success, "launch stage1");
    shim.streamSynchronize();
    PXG_CHECK_MSG(shim.launchKernel(1, 32, 1, 1, &prm, sizeof(prm)) == PxgResult::Success, "launch stage2");
    shim.streamSynchronize();

    // read back per-block totals + prefix, and the grand total from the inline
    // desc field (written into device memory at dDesc).
    uint32_t gTotal[3] = { 0, 0, 0 };
    shim.memcpyDtoH(gTotal, dDesc + offsetof(PxgBroadPhaseDesc, totalIncrementalComparisons), 3 * 4);

    size_t mismB = 0, mismI = 0, mutated = 0;
    uint32_t firstBad = 0xffffffffu, gBad = 0, rBad = 0;
    for (uint32_t ax = 0; ax < 3; ++ax)
    {
        std::vector<uint32_t> gInc(nbProj), gBlock(NUM_BLOCKS);
        shim.memcpyDtoH(gInc.data(), dInc[ax], nbProj * 4);
        shim.memcpyDtoH(gBlock.data(), dBlock[ax], NUM_BLOCKS * 4);
        for (uint32_t i = 0; i < NUM_BLOCKS; ++i) if (gBlock[i] != refBlock[ax][i]) ++mismB;
        for (uint32_t i = 0; i < nbProj; ++i)
        {
            if (gInc[i] != 0xdeadbeefu) ++mutated;
            if (gInc[i] != refInc[ax][i])
            {
                if (firstBad == 0xffffffffu) { firstBad = ax * nbProj + i; gBad = gInc[i]; rBad = refInc[ax][i]; }
                ++mismI;
            }
        }
    }
    PXG_CHECK_MSG(mismB == 0, "%zu per-block comparison mismatches", mismB);
    PXG_CHECK_MSG(mismI == 0, "%zu incremental-comparison mismatches (first at %u: gpu=%u ref=%u)", mismI, firstBad,
                  gBad, rBad);
    for (uint32_t axis = 0; axis < 3; ++axis)
        PXG_CHECK_MSG(gTotal[axis] == refTotal[axis], "axis %u total: gpu=%u ref=%u", axis, gTotal[axis],
                      refTotal[axis]);
    PXG_CHECK_MSG(mutated > 0, "stage kernels did not write the prefix buffer (unbound launch?)");

    for (uint32_t ax = 0; ax < 3; ++ax)
    {
        for (PxgDevicePtr p : { dRanks[ax], dHand[ax], dInc[ax], dBlock[ax] }) shim.memFree(p);
        for (uint32_t fr = 0; fr < 2; ++fr) { shim.memFree(dSH[fr][ax]); shim.memFree(dEH[fr][ax]); }
    }
    shim.memFree(dDesc);
}

} // namespace
