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

// Verification of npCompressContacts.slang: compressContactStage1 on the
// ENGINE ABI (blob {8,4,8} = 24 bytes, real PxsContactManagerOutput array via
// Ptr<>, 32 blocks x 256 over the fixed partition) and updateFrictionPatches
// (bound-buffer harness form, unchanged).  Both against C++ references — the
// block count over the exact work partition, and the rescaled offset per pair.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "PxNodeIndex.h"
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
const uint32_t CONTACT_SIZE = 64, FRIC_SIZE = 52; // sizeof(PxContactPatch)/PxFrictionPatch

static std::vector<uint8_t> cLoader(const char* file) { return loadShaderBlob(file); }

PXG_TEST(npCompressContacts_stage1)
{
    using namespace physx;
    std::mt19937 rng(46010);
    const uint32_t N = 9000;

    // REAL PxsContactManagerOutput array; only nbContacts is read by stage1
    std::vector<PxsContactManagerOutput> outData(N);
    std::memset(outData.data(), 0, N * sizeof(PxsContactManagerOutput));
    std::vector<uint32_t> nbRef(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        const uint32_t nb = (rng() % 3 == 0) ? 0u : (1u + rng() % 8u);
        nbRef[i] = nb;
        outData[i].nbContacts = PxU16(nb);
        outData[i].flags = PxU16(rng() % 0xffff); // high 16 of word 9 = noise
        outData[i].contactPatches = reinterpret_cast<PxU8*>(uintptr_t(rng())); // unused by stage1
    }

    const uint32_t totalBlockRequired = (N + BLOCK - 1) / BLOCK;
    const uint32_t numIterationPerBlock = (totalBlockRequired + NUM_BLOCKS - 1) / NUM_BLOCKS;
    std::vector<uint32_t> refBlock(NUM_BLOCKS, 0);
    for (uint32_t b = 0; b < NUM_BLOCKS; ++b)
    {
        uint32_t cnt = 0;
        for (uint32_t it = 0; it < numIterationPerBlock; ++it)
            for (uint32_t tid = 0; tid < BLOCK; ++tid)
            {
                const uint32_t w = it * BLOCK + tid + numIterationPerBlock * b * BLOCK;
                if (w < N && nbRef[w] != 0) ++cnt;
            }
        refBlock[b] = cnt;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npCompressContacts.spv", "compressContactStage1", 24u, 0 }, // {8,4,8}
    };
    PxgCudaShim shim(dev, manifest, 1, &cLoader);

    PxgDevicePtr dOut = 0, dBlock = 0;
    if (shim.memAlloc(&dOut, N * sizeof(PxsContactManagerOutput)) != PxgResult::Success || !dOut)
    { std::printf("  NOTE: no BDA -> skip\n"); return; }
    shim.memAlloc(&dBlock, NUM_BLOCKS * 4);
    shim.memcpyHtoD(dOut, outData.data(), N * sizeof(PxsContactManagerOutput));
    shim.memsetD32(dBlock, 0, NUM_BLOCKS);

    struct { uint64_t outputData; uint32_t numTotalPairs; uint64_t gBlockNumPairs; } blob = { dOut, N, dBlock };
    static_assert(sizeof(blob) == 24, "compressStage1 engine blob is 24 bytes");
    PXG_CHECK_MSG(shim.launchKernel(0, NUM_BLOCKS, 1, 1, &blob, sizeof(blob)) == PxgResult::Success,
                  "launch compressStage1 (engine ABI)");
    shim.streamSynchronize();

    std::vector<uint32_t> g(NUM_BLOCKS);
    shim.memcpyDtoH(g.data(), dBlock, NUM_BLOCKS * 4);
    size_t mm = 0;
    uint32_t total = 0;
    for (uint32_t b = 0; b < NUM_BLOCKS; ++b) { if (g[b] != refBlock[b]) ++mm; total += refBlock[b]; }
    PXG_CHECK_MSG(mm == 0, "compressContactStage1: %zu block mismatches", mm);
    PXG_CHECK_MSG(total > N / 4, "stage1: too few pairs with contacts (%u)", total);

    shim.memFree(dOut); shim.memFree(dBlock);
}

PXG_TEST(npCompressContacts_updateFrictionPatches)
{
    using namespace physx;
    std::mt19937 rng(46020);
    const uint32_t N = 7000;
    // Real 64-bit device-pointer bases (the engine passes CUdeviceptr-scale
    // addresses; the arithmetic is (contactPatches - startContact) in bytes).
    const uint64_t startContact  = 0x0000000140001000ull;
    const uint64_t startFriction = 0x0000000180002000ull;

    // REAL PxsContactManagerOutput array; kernel reads contactPatches (byte 0)
    // and writes frictionPatches (byte 24) via the byte-exact mirror.
    std::vector<PxsContactManagerOutput> outData(N);
    std::memset(outData.data(), 0, N * sizeof(PxsContactManagerOutput));
    std::vector<uint64_t> contactPtr(N), refFriction(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        // contactPatches = startContact + patchIndex*64 (a multiple of the patch
        // size).  Every few entries use a non-multiple-of-64 diff too so the
        // multiply-before-divide ordering is observable (a divide-first
        // regression would differ here).
        const uint64_t off = (i % 5 == 0) ? (startContact + i * 32ull + 16ull)
                                          : (startContact + i * uint64_t(CONTACT_SIZE));
        contactPtr[i] = off;
        outData[i].contactPatches  = reinterpret_cast<PxU8*>(uintptr_t(off));
        outData[i].frictionPatches = reinterpret_cast<PxU8*>(uintptr_t(0xdeadbeefdeadbeefull)); // overwritten
        refFriction[i] = startFriction + (off - startContact) * FRIC_SIZE / CONTACT_SIZE;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npCompressContacts.spv", "updateFrictionPatches", 32u, 0 }, // {4,8,8,8}
    };
    PxgCudaShim shim(dev, manifest, 1, &cLoader);

    PxgDevicePtr dOut = 0;
    if (shim.memAlloc(&dOut, N * sizeof(PxsContactManagerOutput)) != PxgResult::Success || !dOut)
    { std::printf("  NOTE: no BDA -> skip\n"); return; }
    shim.memcpyHtoD(dOut, outData.data(), N * sizeof(PxsContactManagerOutput));

    struct { uint32_t pairCount; uint64_t startContact, startFriction, outputs; } blob =
        { N, startContact, startFriction, dOut };
    static_assert(sizeof(blob) == 32, "updateFrictionPatches engine blob is 32 bytes");
    const uint32_t GROUPS = (N + 255) / 256;
    PXG_CHECK_MSG(shim.launchKernel(0, GROUPS, 1, 1, &blob, sizeof(blob)) == PxgResult::Success,
                  "launch updateFrictionPatches (engine ABI)");
    shim.streamSynchronize();

    std::vector<PxsContactManagerOutput> g(N);
    shim.memcpyDtoH(g.data(), dOut, N * sizeof(PxsContactManagerOutput));
    size_t mm = 0, mutated = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const uint64_t got = uint64_t(uintptr_t(g[i].frictionPatches));
        if (got != refFriction[i]) ++mm;
        if (got != 0xdeadbeefdeadbeefull) ++mutated;
    }
    PXG_CHECK_MSG(mm == 0, "updateFrictionPatches: %zu mismatches", mm);
    PXG_CHECK_MSG(mutated == N, "updateFrictionPatches: kernel did not write (%zu/%u)", mutated, N);

    shim.memFree(dOut);
}

} // namespace
