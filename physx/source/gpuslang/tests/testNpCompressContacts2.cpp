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

// P5 Step 4: compressContactStage2 on the ENGINE ABI, driven through
// PxgCudaShim exactly as PxgNarrowphaseCore.cpp launches it — 16-argument
// blob {8,8,4,8,8,8,8,8,8,8,8,8,8,8,8,4} (128 bytes), grid 32 blocks x 256.
// REAL PxgContactManagerInput / PxsContactManagerOutput / PxGpuContactPair
// arrays + real CPU/GPU stream base pointers at live device addresses.  The
// cross-block compaction structure is the verified bound-buffer reference;
// this test additionally exercises the REAL pointer->offset conversions
// (cmOutput's stream pointers minus the cpu*Base args, PxReal* force diff /4)
// and the rebased device pointers written into PxGpuContactPair.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "PxNodeIndex.h"
#include "PxContact.h"
#include "PxgContactManager.h"
#include "PxsContactManagerState.h"

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

const uint32_t BLOCK = 256, NUM_BLOCKS = 32;
const uint32_t CONTACT_SIZE = 64, FRIC_SIZE = 52;
const uint32_t TABLE = 4096;

// host-side expected PxGpuContactPair (the fields the kernel writes)
struct ExpPair
{
    uint64_t contactPatches, contactPoints, contactForces, frictionPatches;
    uint32_t ref0, ref1;
    uint64_t node0, node1, actor0, actor1;
    uint32_t nbContactsPatches;
};

static std::vector<uint8_t> c2Loader(const char* file) { return loadShaderBlob(file); }

PXG_TEST(npCompressContacts2_stage2)
{
    using namespace physx;
    std::mt19937 rng(53010);
    const uint32_t N = 40000;

    // simulated compressed-stream base offsets (fake device bases the pointers
    // are relative to); the kernel rebases onto the GPU bases below
    const uint64_t cpuPatchBase = 0x10000000ull, cpuContactBase = 0x20000000ull, cpuForceBase = 0x30000000ull;

    std::vector<PxgContactManagerInput> cmIn(N);
    std::vector<PxsContactManagerOutput> outData(N);
    std::memset(cmIn.data(), 0, N * sizeof(PxgContactManagerInput));
    std::memset(outData.data(), 0, N * sizeof(PxsContactManagerOutput));
    std::vector<uint32_t> nb(N);
    std::vector<uint32_t> patchOffs(N), pointOffs(N), forceOffs(N), forcePres(N), nbPatchesArr(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        const uint32_t nbc = (rng() % 100 < 45) ? 0u : (1u + rng() % 8u);
        nb[i] = nbc;
        patchOffs[i] = (rng() % 20000) * CONTACT_SIZE + (rng() % CONTACT_SIZE);
        pointOffs[i] = rng() % 0x40000;
        forceOffs[i] = (rng() % 0x40000) & ~3u; // multiple of 4 (PxReal* diff)
        forcePres[i] = (rng() % 100 < 20) ? 0u : 1u;
        nbPatchesArr[i] = rng() % 0xff;
        const uint32_t ref0 = rng() % TABLE, ref1 = rng() % TABLE;
        cmIn[i].transformCacheRef0 = ref0;
        cmIn[i].transformCacheRef1 = ref1;
        outData[i].contactPatches = reinterpret_cast<PxU8*>(cpuPatchBase + patchOffs[i]);
        outData[i].contactPoints = reinterpret_cast<PxU8*>(cpuContactBase + pointOffs[i]);
        outData[i].contactForces = forcePres[i] ? reinterpret_cast<PxReal*>(cpuForceBase + forceOffs[i]) : nullptr;
        outData[i].nbPatches = PxU8(nbPatchesArr[i]);
        outData[i].nbContacts = PxU16(nbc);
        outData[i].flags = PxU16(rng() % 0xffff); // high 16 word9 = noise
    }
    std::vector<uint64_t> remap(TABLE), actorTab(TABLE);
    for (uint32_t i = 0; i < TABLE; ++i)
    {
        remap[i] = (uint64_t(rng()) << 32) | rng();
        actorTab[i] = (uint64_t(rng()) << 32) | rng();
    }

    // gBlockNumPairs
    const uint32_t totalBlockRequired = (N + BLOCK - 1) / BLOCK;
    const uint32_t numIter = (totalBlockRequired + NUM_BLOCKS - 1) / NUM_BLOCKS;
    std::vector<uint32_t> blockNum(NUM_BLOCKS, 0);
    for (uint32_t b = 0; b < NUM_BLOCKS; ++b)
    {
        uint32_t cnt = 0;
        for (uint32_t it = 0; it < numIter; ++it)
            for (uint32_t tid = 0; tid < BLOCK; ++tid)
            {
                const uint32_t w = it * BLOCK + tid + numIter * b * BLOCK;
                if (w < N && nb[w] != 0) ++cnt;
            }
        blockNum[b] = cnt;
    }
    std::vector<uint32_t> blockStart(NUM_BLOCKS, 0);
    uint32_t acc = 0;
    for (uint32_t b = 0; b < NUM_BLOCKS; ++b) { blockStart[b] = acc; acc += blockNum[b]; }
    const uint32_t numPairs = acc;

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npCompressContacts2.spv", "compressContactStage2", 128u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &c2Loader);

    // GPU rebase targets (arbitrary live device buffers big enough not to matter
    // for pointer arithmetic; we compare full u64 pointers)
    PxgDevicePtr dCmIn = 0, dOut = 0, dBlock = 0, dRemap = 0, dActor = 0, dPairs = 0, dNum = 0;
    PxgDevicePtr gPatchBase = 0, gContactBase = 0, gForceBase = 0, gFricBase = 0;
    if (shim.memAlloc(&dCmIn, N * sizeof(PxgContactManagerInput)) != PxgResult::Success || !dCmIn)
    { std::printf("  NOTE: no BDA -> skip\n"); return; }
    shim.memAlloc(&dOut, N * sizeof(PxsContactManagerOutput));
    shim.memAlloc(&dBlock, NUM_BLOCKS * 4);
    shim.memAlloc(&dRemap, TABLE * 8);
    shim.memAlloc(&dActor, TABLE * 8);
    shim.memAlloc(&dPairs, size_t(numPairs + 16) * sizeof(PxGpuContactPair));
    shim.memAlloc(&dNum, 4);
    // small placeholder base allocations (their device addresses are the bases)
    shim.memAlloc(&gPatchBase, 16);
    shim.memAlloc(&gContactBase, 16);
    shim.memAlloc(&gForceBase, 16);
    shim.memAlloc(&gFricBase, 16);

    shim.memcpyHtoD(dCmIn, cmIn.data(), N * sizeof(PxgContactManagerInput));
    shim.memcpyHtoD(dOut, outData.data(), N * sizeof(PxsContactManagerOutput));
    shim.memcpyHtoD(dBlock, blockNum.data(), NUM_BLOCKS * 4);
    shim.memcpyHtoD(dRemap, remap.data(), TABLE * 8);
    shim.memcpyHtoD(dActor, actorTab.data(), TABLE * 8);
    shim.memsetD32(dPairs, 0, size_t(numPairs + 16) * sizeof(PxGpuContactPair) / 4);
    shim.memsetD32(dNum, 0, 1);

    struct C2Blob
    {
        uint64_t inputData, outputData;
        uint32_t numTotalPairs;
        uint64_t shapeToRigidRemapTable, transformCacheIdToActorTable, gBlockNumPairs,
                 cpuCompressedPatchesBase, cpuCompressedContactsBase, cpuForceBufferBase,
                 gpuCompressedPatchesBase, gpuCompressedContactsBase, gpuForceBufferBase,
                 gpuFrictionPatchesBase, numPairs, outputPatchs;
        uint32_t maxOutputPatches;
    } blob = { dCmIn, dOut, N, dRemap, dActor, dBlock, cpuPatchBase, cpuContactBase, cpuForceBase,
               gPatchBase, gContactBase, gForceBase, gFricBase, dNum, dPairs, numPairs + 16 };
    static_assert(sizeof(C2Blob) == 128, "compressStage2 engine blob is 128 bytes");

    PXG_CHECK_MSG(shim.launchKernel(0, NUM_BLOCKS, 1, 1, &blob, sizeof(blob)) == PxgResult::Success,
                  "launch compressStage2 (engine ABI)");
    shim.streamSynchronize();

    // reference stable compaction + real pointer arithmetic
    std::vector<ExpPair> refOut(numPairs);
    for (uint32_t b = 0; b < NUM_BLOCKS; ++b)
    {
        uint32_t rank = 0;
        for (uint32_t it = 0; it < numIter; ++it)
            for (uint32_t tid = 0; tid < BLOCK; ++tid)
            {
                const uint32_t w = it * BLOCK + tid + numIter * b * BLOCK;
                if (w >= N || nb[w] == 0) continue;
                const uint32_t index = blockStart[b] + rank; ++rank;
                const uint32_t ref0 = cmIn[w].transformCacheRef0, ref1 = cmIn[w].transformCacheRef1;
                const uint32_t forceIndex = forcePres[w] ? (forceOffs[w] >> 2) : 0xFFFFFFFFu;
                ExpPair& o = refOut[index];
                o.ref0 = ref0; o.ref1 = ref1;
                o.node0 = remap[ref0]; o.node1 = remap[ref1];
                o.contactForces = uint64_t(gForceBase) + uint64_t(forceIndex) * 4u;
                o.frictionPatches = uint64_t(gFricBase) + (uint64_t(patchOffs[w]) / CONTACT_SIZE) * FRIC_SIZE;
                o.contactPatches = uint64_t(gPatchBase) + uint64_t(patchOffs[w]);
                o.contactPoints = uint64_t(gContactBase) + uint64_t(pointOffs[w]);
                o.nbContactsPatches = (nb[w] & 0xffffu) | ((nbPatchesArr[w] & 0xffu) << 16);
                o.actor0 = actorTab[ref0]; o.actor1 = actorTab[ref1];
            }
    }

    std::vector<PxGpuContactPair> gPairs(numPairs);
    std::vector<uint32_t> gNum(1);
    shim.memcpyDtoH(gPairs.data(), dPairs, numPairs * sizeof(PxGpuContactPair));
    shim.memcpyDtoH(gNum.data(), dNum, 4);

    auto u64of = [](const void* q) { return uint64_t(reinterpret_cast<uintptr_t>(q)); };
    size_t mm = 0;
    for (uint32_t i = 0; i < numPairs; ++i)
    {
        const ExpPair& r = refOut[i];
        const PxGpuContactPair& v = gPairs[i];
        const uint32_t vNbCp = uint32_t(v.nbContacts) | (uint32_t(v.nbPatches) << 16);
        if (v.transformCacheRef0 != r.ref0 || v.transformCacheRef1 != r.ref1 ||
            v.nodeIndex0.getInd() != r.node0 || v.nodeIndex1.getInd() != r.node1 ||
            u64of(v.actor0) != r.actor0 || u64of(v.actor1) != r.actor1 ||
            u64of(v.contactForces) != r.contactForces || u64of(v.frictionPatches) != r.frictionPatches ||
            u64of(v.contactPatches) != r.contactPatches || u64of(v.contactPoints) != r.contactPoints ||
            vNbCp != r.nbContactsPatches) ++mm;
    }
    PXG_CHECK_MSG(gNum[0] == numPairs, "stage2: numPairs %u != %u", gNum[0], numPairs);
    PXG_CHECK_MSG(mm == 0, "compressContactStage2: %zu compacted-pair mismatches", mm);
    PXG_CHECK_MSG(numPairs > N / 4 && numPairs < N, "stage2: survivor count implausible (%u/%u)", numPairs, N);

    for (PxgDevicePtr h : { dCmIn, dOut, dBlock, dRemap, dActor, dPairs, dNum,
                            gPatchBase, gContactBase, gForceBase, gFricBase })
        shim.memFree(h);
}

} // namespace
