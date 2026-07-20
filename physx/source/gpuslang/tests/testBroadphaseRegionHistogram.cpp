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

// Verification of broadphaseRegionHistogram.slang on the ENGINE ABI (P5 Step
// 4): a REAL PxgBroadPhaseDesc is filled with live device addresses
// (regionAccum / blockRegionAccum) and both kernels are launched through
// PxgCudaShim exactly as the engine does — ONE bpDesc pointer, grid (32,1,1),
// nbProjections derived in-kernel as 2 * (numPreviousHandles +
// numCreatedHandles - numRemovedHandles), and the grand total written into the
// desc's own regionAccumTotal member (read back via memcpyDtoH of the desc).
// The compute+output pair implements a device-wide exclusive prefix sum over a
// flat PxU32 array plus its grand total.  Covers sizes that straddle the
// warp/block/grid-stride boundaries and a large-value case that would fail if
// any 16-bit packing crept in.  nbProjections is always even on the engine ABI
// (numHandles*2), so the odd harness sizes (8193/9001) become 8194/9002.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "foundation/PxBounds3.h"
#include "PxgBroadPhaseDesc.h"

#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

static std::vector<uint8_t> bpShimLoader(const char* file) { return loadShaderBlob(file); }

// Mirrors the bridge's marshalling: place each argument at the natural alignment
// inferred from its size, then round the blob to the largest alignment seen
// (a compiled struct's trailing padding).
struct EngineBlob
{
    std::vector<uint8_t> bytes;
    size_t maxAlign = 1;

    static size_t alignFor(size_t size)
    {
        if (size >= 16) return 16;
        if (size >= 8)  return 8;
        if (size >= 4)  return 4;
        if (size >= 2)  return 2;
        return 1;
    }
    template <typename T>
    void add(const T& v)
    {
        const size_t a = alignFor(sizeof(T));
        if (a > maxAlign) maxAlign = a;
        bytes.resize((bytes.size() + (a - 1)) & ~(a - 1));
        const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
        bytes.insert(bytes.end(), p, p + sizeof(T));
    }
    void finish() { bytes.resize((bytes.size() + (maxAlign - 1)) & ~(maxAlign - 1)); }
};

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

// numPrev/numCreated/numRemoved feed the in-kernel handle-count arithmetic;
// nbProjections = 2 * (numPrev + numCreated - numRemoved).
void runRegionHistCase(const char* tag, uint32_t seed, uint32_t numPrev, uint32_t numCreated,
                       uint32_t numRemoved, uint32_t maxVal)
{
    using namespace physx;
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    const uint32_t numHandles = numPrev + numCreated - numRemoved;
    const uint32_t nbProj = numHandles * 2;

    std::mt19937 rng(seed);
    std::vector<uint32_t> input(nbProj);
    for (uint32_t i = 0; i < nbProj; ++i)
        input[i] = rng() % (maxVal + 1);

    // reference: exclusive prefix sum + grand total (uint32 modular, matching GPU)
    std::vector<uint32_t> ref(nbProj);
    uint32_t acc = 0;
    for (uint32_t i = 0; i < nbProj; ++i)
    {
        ref[i] = acc;
        acc += input[i];
    }
    const uint32_t refTotal = acc;

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "broadphaseRegionHistogram.spv", "computeRegionsHistogram", 8u, 0 },
        { "broadphaseRegionHistogram.spv", "outputRegionsHistogram", 8u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 2, &bpShimLoader);

    const std::vector<uint32_t> blockInit(32, 0xdeadbeefu);

    BpDescRig rig(&shim);
    const PxgDevicePtr dHist = rig.alloc(input.data(), input.size() * 4);
    const PxgDevicePtr dBlock = rig.alloc(blockInit.data(), blockInit.size() * 4);
    if (!dHist)
    {
        std::printf("  NOTE: shim has no device-address allocations (no BDA) -> skip\n");
        return;
    }
    rig.desc.numPreviousHandles = numPrev;
    rig.desc.numCreatedHandles = numCreated;
    rig.desc.numRemovedHandles = numRemoved;
    rig.desc.regionAccum = reinterpret_cast<PxU32*>(dHist);
    rig.desc.blockRegionAccum = reinterpret_cast<PxU32*>(dBlock);
    rig.desc.regionAccumTotal = 0xdeadbeefu; // sentinel — the kernel must overwrite it
    const PxgDevicePtr dDesc = rig.upload();

    // Engine ABI blob: one device pointer — argSizes {8}.
    EngineBlob blob;
    blob.add<uint64_t>(dDesc);
    blob.finish();
    PXG_CHECK_MSG(blob.bytes.size() == 8, "[%s] region blob is 8 bytes (got %zu)", tag, blob.bytes.size());

    // Engine grid: (BP_COMPUTE/OUTPUT_REGION_HISTOGRAM = 32, 1, 1), 256 threads.
    PXG_CHECK_MSG(shim.launchKernel(0, 32, 1, 1, blob.bytes.data(), uint32_t(blob.bytes.size())) ==
                      PxgResult::Success,
                  "[%s] launch computeRegionsHistogram", tag);
    shim.streamSynchronize();
    PXG_CHECK_MSG(shim.launchKernel(1, 32, 1, 1, blob.bytes.data(), uint32_t(blob.bytes.size())) ==
                      PxgResult::Success,
                  "[%s] launch outputRegionsHistogram", tag);
    shim.streamSynchronize();

    std::vector<uint32_t> gHist(nbProj, 0);
    shim.memcpyDtoH(gHist.data(), dHist, nbProj * 4);

    // the grand total lives in the desc itself (bpDesc->regionAccumTotal)
    PxgBroadPhaseDesc descBack;
    std::memset(&descBack, 0, sizeof(descBack));
    shim.memcpyDtoH(&descBack, dDesc, sizeof(descBack));
    const uint32_t gTotal = descBack.regionAccumTotal;

    size_t mism = 0;
    uint32_t firstBad = 0xffffffffu;
    for (uint32_t i = 0; i < nbProj; ++i)
    {
        if (gHist[i] != ref[i])
        {
            if (firstBad == 0xffffffffu) firstBad = i;
            ++mism;
        }
    }
    if (mism)
        PXG_CHECK_MSG(mism == 0, "[%s n=%u] %zu prefix mismatches (first at %u: gpu=%u ref=%u)", tag, nbProj, mism,
                      firstBad, gHist[firstBad], ref[firstBad]);
    else
        PXG_CHECK_MSG(mism == 0, "[%s n=%u] %zu prefix mismatches", tag, nbProj, mism);
    PXG_CHECK_MSG(gTotal == refTotal, "[%s n=%u] total: gpu=%u ref=%u", tag, nbProj, gTotal, refTotal);

    for (PxgDevicePtr pfree : { dHist, dBlock, dDesc })
        shim.memFree(pfree);
}

PXG_TEST(broadphaseRegionHistogram_prefixSum)
{
    // small counts (0..7); sizes straddling warp (32), block (256) and
    // grid-stride tile (256*32 = 8192) boundaries.  The created/removed splits
    // exercise the in-kernel numHandles arithmetic.
    runRegionHistCase("small-100", 17010, 50, 0, 0, 7);
    runRegionHistCase("small-260", 17011, 100, 40, 10, 7);          // 130 handles
    runRegionHistCase("exact-8192", 17012, 4096, 0, 0, 7);
    runRegionHistCase("straddle-8194", 17013, 4000, 200, 103, 7);   // 4097 handles
    runRegionHistCase("multi-iter-20000", 17014, 9000, 1500, 500, 7); // 10000 handles
    // wide values (> 16 bits) with the total kept < 2^32 -> catches any packing
    runRegionHistCase("wide-values", 17015, 4501, 0, 0, (1u << 20));
}

} // namespace
