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

// Verification of npCvxLostFound.slang: prepareLostFoundPairs_Stage1/_Stage2
// (cudaGJKEPA.cu).  Exclusive scan over the 2*numPairs flag values, compaction
// of the flagged managers' {nbPatches, prevPatches, statusFlag} + opaque
// handle at the scan offsets, lost count (prefix at idx == numPairs) -> x and
// grand total -> y.  Everything integer: comparisons exact.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"
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

static std::vector<uint8_t> cLoader(const char* file) { return loadShaderBlob(file); }

static void runOne(uint32_t numPairs, uint32_t seed, float density)
{
    using namespace physx;
    std::mt19937 rng(seed);
    const uint32_t n = 2 * numPairs;
    const uint32_t SENTINEL = 0xdeadbeefu;
    const uint64_t PTR_SENTINEL = 0xdeadbeefdeadbeefull;

    std::vector<uint32_t> flags(n);
    for (uint32_t i = 0; i < n; ++i) flags[i] = (float(rng() % 1000) / 1000.f < density) ? 1u : 0u;

    // REAL PxsContactManagerOutput array (allOutputManagers): the kernel reads the
    // packed flags word at byte 32 (allflagsStart | nbPatches<<8 | statusFlag<<16
    // | prevPatches<<24).  Contact-manager pointers are real 64-bit device pointers.
    std::vector<PxsContactManagerOutput> mgrs(numPairs);
    std::memset(mgrs.data(), 0, numPairs * sizeof(PxsContactManagerOutput));
    std::vector<uint64_t> mgrHandles(numPairs);
    for (uint32_t i = 0; i < numPairs; ++i)
    {
        mgrs[i].allflagsStart = PxU8(rng());
        mgrs[i].nbPatches     = PxU8(rng());
        mgrs[i].statusFlag    = PxU8(rng());
        mgrs[i].prevPatches   = PxU8(rng());
        mgrHandles[i] = (uint64_t(rng()) << 32) | uint64_t(rng()); // opaque PxsContactManager*
    }

    // CPU reference: exclusive scan + compact
    std::vector<uint32_t> refCompact(n, SENTINEL);
    std::vector<uint64_t> refPtrs(n, PTR_SENTINEL);
    uint32_t refX = 0xffffffffu, refY = 0, off = 0;
    for (uint32_t i = 0; i < n; ++i)
    {
        if (i == numPairs) refX = off;
        if (flags[i])
        {
            const uint32_t idx = i % numPairs;
            const uint32_t nbPatches = mgrs[idx].nbPatches, statusFlag = mgrs[idx].statusFlag,
                           prevPatches = mgrs[idx].prevPatches;
            refCompact[off] = nbPatches | (prevPatches << 8) | (statusFlag << 16);
            refPtrs[off] = mgrHandles[idx];
            ++off;
        }
    }
    refY = off;

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t NUM_BLOCKS = 32;
    const PxgKernelDef manifest[] = {
        { "npCvxLostFound.spv", "prepareLostFoundPairs_Stage1", 32u, 0 }, // {8,8,8,4}
        { "npCvxLostFound.spv", "prepareLostFoundPairs_Stage2", 72u, 0 }, // {8,8,8,8,8,8,8,4,8}
    };
    PxgCudaShim shim(dev, manifest, 2, &cLoader);

    PxgDevicePtr dFlags = 0, dRun = 0, dBlk = 0, dMgr = 0, dHandles = 0, dCompact = 0, dPtrs = 0, dLT = 0;
    if (shim.memAlloc(&dFlags, n * 4) != PxgResult::Success || !dFlags)
    { std::printf("  NOTE: no BDA -> skip\n"); return; }
    shim.memAlloc(&dRun, n * 4);
    shim.memAlloc(&dBlk, NUM_BLOCKS * 4);
    shim.memAlloc(&dMgr, numPairs * sizeof(PxsContactManagerOutput));
    shim.memAlloc(&dHandles, numPairs * 8);
    shim.memAlloc(&dCompact, n * 4);   // n PxsContactManagerOutputCounts (4 B each)
    shim.memAlloc(&dPtrs, n * 8);
    shim.memAlloc(&dLT, 2 * 4);

    shim.memcpyHtoD(dFlags, flags.data(), n * 4);
    shim.memsetD32(dRun, 0, n);
    shim.memsetD32(dBlk, 0, NUM_BLOCKS);
    shim.memcpyHtoD(dMgr, mgrs.data(), numPairs * sizeof(PxsContactManagerOutput));
    shim.memcpyHtoD(dHandles, mgrHandles.data(), numPairs * 8);
    shim.memsetD32(dCompact, SENTINEL, n);
    { std::vector<uint64_t> ps(n, PTR_SENTINEL); shim.memcpyHtoD(dPtrs, ps.data(), n * 8); }
    shim.memsetD32(dLT, 0xffffffffu, 2);

    struct { uint64_t inputFlags, tempRunsum, blockAccum; uint32_t numPairs; } b1 =
        { dFlags, dRun, dBlk, numPairs };
    static_assert(sizeof(b1) == 32, "stage1 blob 32 bytes");
    PXG_CHECK_MSG(shim.launchKernel(0, NUM_BLOCKS, 1, 1, &b1, sizeof(b1)) == PxgResult::Success,
                  "launch prepareLostFoundPairs_Stage1");
    shim.streamSynchronize();

    struct { uint64_t inputFlags, tempRunsum, allOutputManagers, compacted, ptrs, lostTotal, blockAccum;
             uint32_t numPairs; uint64_t cmArray; } b2 =
        { dFlags, dRun, dMgr, dCompact, dPtrs, dLT, dBlk, numPairs, dHandles };
    static_assert(sizeof(b2) == 72, "stage2 blob 72 bytes");
    PXG_CHECK_MSG(shim.launchKernel(1, NUM_BLOCKS, 1, 1, &b2, sizeof(b2)) == PxgResult::Success,
                  "launch prepareLostFoundPairs_Stage2");
    shim.streamSynchronize();

    std::vector<uint32_t> gCompact(n), gRun(n), gBlk(NUM_BLOCKS), gLT(2);
    std::vector<uint64_t> gPtrs(n);
    shim.memcpyDtoH(gCompact.data(), dCompact, n * 4);
    shim.memcpyDtoH(gPtrs.data(), dPtrs, n * 8);
    shim.memcpyDtoH(gLT.data(), dLT, 2 * 4);
    shim.memcpyDtoH(gRun.data(), dRun, n * 4);
    shim.memcpyDtoH(gBlk.data(), dBlk, NUM_BLOCKS * 4);

    // intermediate buffers must match CUDA's 512-thread-block chunking exactly:
    // tempRunsum[i] = exclusive prefix within the owning block's chunk,
    // blockAccumulation[b] = that chunk's total
    {
        const uint32_t BLOCK = 512;
        const uint32_t totalBlockRequired = (n + BLOCK - 1) / BLOCK;
        const uint32_t iters = (totalBlockRequired + NUM_BLOCKS - 1) / NUM_BLOCKS;
        const uint32_t chunk = BLOCK * iters;
        size_t mmR = 0, mmB = 0;
        for (uint32_t b = 0; b < NUM_BLOCKS; ++b)
        {
            uint32_t acc = 0;
            for (uint32_t idx = b * chunk; idx < (b + 1) * chunk && idx < n; ++idx)
            {
                if (gRun[idx] != acc) ++mmR;
                acc += flags[idx];
            }
            if (gBlk[b] != acc) ++mmB;
        }
        PXG_CHECK_MSG(mmR == 0, "lostFound(%u): %zu tempRunsum mismatches", numPairs, mmR);
        PXG_CHECK_MSG(mmB == 0, "lostFound(%u): %zu blockAccumulation mismatches", numPairs, mmB);
    }

    size_t mmC = 0, mmP = 0;
    for (uint32_t i = 0; i < n; ++i)
    {
        if (gCompact[i] != refCompact[i]) ++mmC;
        if (gPtrs[i] != refPtrs[i]) ++mmP;
    }
    PXG_CHECK_MSG(mmC == 0, "lostFound(%u): %zu compacted mismatches", numPairs, mmC);
    PXG_CHECK_MSG(mmP == 0, "lostFound(%u): %zu handle mismatches", numPairs, mmP);
    PXG_CHECK_MSG(gLT[0] == refX, "lostFound(%u): x (lost) %u != %u", numPairs, gLT[0], refX);
    PXG_CHECK_MSG(gLT[1] == refY, "lostFound(%u): y (total) %u != %u", numPairs, gLT[1], refY);

    for (PxgDevicePtr h : { dFlags, dRun, dBlk, dMgr, dHandles, dCompact, dPtrs, dLT })
        shim.memFree(h);
}

PXG_TEST(npCvxLostFound_compact)
{
    runOne(1, 150001, 0.5f);       // minimal
    runOne(5, 150002, 0.8f);       // tiny, dense
    runOne(256, 150003, 0.4f);     // n = 512: exactly one block chunk
    runOne(300, 150004, 0.3f);     // non-multiple
    runOne(8192, 150005, 0.5f);    // n = 16384 = 32*512: exactly one iteration per block
    runOne(6000, 150006, 0.35f);   // multi-iteration, non-multiple
    runOne(9000, 150007, 0.02f);   // sparse flags
    runOne(7777, 150008, 0.97f);   // nearly all flagged
}

} // namespace
