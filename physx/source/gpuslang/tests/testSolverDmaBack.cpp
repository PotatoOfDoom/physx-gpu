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

// Verification of solverDmaBack.slang on the ENGINE ABI (P5 Step 5):
// dmaBackChangedElems (word-wise copy of the 40-byte force-change threshold
// elements, source pointer + count read from a REAL PxgSolverCoreDesc) and
// dmaConstraintResidual (gather the .w of each writeback's
// angularImpulse_residual float4; 3-arg CUDA pack).  Both are compared
// bit-exactly against C++ references; a preserved tail catches over-copying
// and a short grid (fewer groups than words / 512) forces the copy loop to
// iterate, exercising the blockDim * gridDim stride.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "PxNodeIndex.h" // completes PxNodeIndex for the PxgSolverCoreDesc chain
#include "PxgSolverCoreDesc.h"

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

static std::vector<uint8_t> dmaBackLoader(const char* file) { return loadShaderBlob(file); }

PXG_TEST(solverDmaBack_changedElems)
{
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    std::mt19937 rng(38010);
    const uint32_t nbElems = 733;
    const uint32_t ELEMENT_WORDS = 10;      // sizeof(Dy::ThresholdStreamElement)/4
    const uint32_t nbWords = ELEMENT_WORDS * nbElems;
    const uint32_t dstWords = nbWords + 2000; // preserve a tail past the copy

    std::vector<uint32_t> src(nbWords);
    for (auto& w : src) w = rng();
    std::vector<uint32_t> dstInit(dstWords, 0xcdcdcdcdu);

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = { { "solverDmaBack.spv", "dmaBackChangedElems", 16u, 0 } };
    PxgCudaShim shim(dev, manifest, 1, &dmaBackLoader);

    // GROUPS * 512 = 1536 < nbWords = 7330 -> the grid-stride loop iterates
    const uint32_t GROUPS = 3;

    PxgDevicePtr dSrc = 0, dDst = 0, dDesc = 0;
    shim.memAlloc(&dSrc, src.size() * 4);
    shim.memAlloc(&dDst, dstInit.size() * 4);
    shim.memAlloc(&dDesc, sizeof(physx::PxgSolverCoreDesc));
    if (!dSrc || !dDst || !dDesc)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    shim.memcpyHtoD(dSrc, src.data(), src.size() * 4);
    shim.memcpyHtoD(dDst, dstInit.data(), dstInit.size() * 4);

    // the REAL engine desc: the kernel reads forceChangeThresholdElements +
    // nbForceChangeElements from it on-device
    physx::PxgSolverCoreDesc desc;
    std::memset(&desc, 0xCD, sizeof(desc)); // untouched fields must not matter
    desc.forceChangeThresholdElements = reinterpret_cast<physx::Dy::ThresholdStreamElement*>(dSrc);
    desc.nbForceChangeElements = nbElems;
    shim.memcpyHtoD(dDesc, &desc, sizeof(desc));

    // the engine's exact param pack: (PxgSolverCoreDesc*, hostChangedElems*)
    struct { uint64_t desc, dst; } prm = { dDesc, dDst };
    static_assert(sizeof(prm) == 16, "dmaBackChangedElems blob is 16 bytes");
    PXG_CHECK_MSG(shim.launchKernel(0, GROUPS, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch dmaBackChangedElems");
    shim.streamSynchronize();

    std::vector<uint32_t> g(dstWords);
    shim.memcpyDtoH(g.data(), dDst, g.size() * 4);
    size_t mm = 0;
    for (uint32_t i = 0; i < dstWords; ++i)
    {
        const uint32_t exp = (i < nbWords) ? src[i] : 0xcdcdcdcdu;
        if (g[i] != exp) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "dmaBackChangedElems: %zu mismatches", mm);

    for (PxgDevicePtr p : { dSrc, dDst, dDesc }) shim.memFree(p);
}

PXG_TEST(solverDmaBack_constraintResidual)
{
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    std::mt19937 rng(38020);
    const uint32_t count = 4000;

    // PxgConstraintWriteback = { float4 linearImpulse_broken; float4 angularImpulse_residual; }
    std::vector<float> writebacks(8 * count);
    auto rf = [&]() { return float(int(rng() % 20000) - 10000) * 0.01f; };
    for (uint32_t i = 0; i < count; ++i)
        for (uint32_t c = 0; c < 8; ++c) writebacks[8 * i + c] = rf();

    std::vector<float> residInit(count + 500, -123.f); // preserve a tail

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = { { "solverDmaBack.spv", "dmaConstraintResidual", 24u, 0 } };
    PxgCudaShim shim(dev, manifest, 1, &dmaBackLoader);

    PxgDevicePtr dWb = 0, dResid = 0;
    shim.memAlloc(&dWb, writebacks.size() * 4);
    shim.memAlloc(&dResid, residInit.size() * 4);
    if (!dWb || !dResid)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    shim.memcpyHtoD(dWb, writebacks.data(), writebacks.size() * 4);
    shim.memcpyHtoD(dResid, residInit.data(), residInit.size() * 4);

    // the engine's exact param pack: (writebacks*, residuals*, PxU32 count) ->
    // 24-byte blob (count @16, tail rounded to the 8-alignment); grid = the
    // engine's ceil(count / 256)
    struct { uint64_t wb, resid; uint32_t count, pad; } prm = { dWb, dResid, count, 0 };
    static_assert(sizeof(prm) == 24, "dmaConstraintResidual blob is 24 bytes");
    const uint32_t GROUPS = (count + 255) / 256; // one thread per constraint
    PXG_CHECK_MSG(shim.launchKernel(0, GROUPS, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch dmaConstraintResidual");
    shim.streamSynchronize();

    std::vector<float> g(count + 500);
    shim.memcpyDtoH(g.data(), dResid, g.size() * 4);
    size_t mm = 0;
    for (uint32_t i = 0; i < count + 500; ++i)
    {
        const float exp = (i < count) ? writebacks[8 * i + 7] : -123.f; // angularImpulse_residual.w
        if (g[i] != exp) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "dmaConstraintResidual: %zu mismatches", mm);

    for (PxgDevicePtr p : { dWb, dResid }) shim.memFree(p);
}

} // namespace
