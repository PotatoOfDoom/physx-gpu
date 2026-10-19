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

// RUNTIME_PLAN.md P3 test: the Vulkan kernel-wrangler (KernelWrangler
// replacement).  Proves the engine's launch path — getFunction(kernelId) ->
// launchKernel(cuFunction, ...) — works by ID: a kernel launched via the
// wrangler-resolved handle produces bitwise-correct results, unmapped/OOB IDs
// resolve to null, and the wrangler built over the generated 487-entry (PhysX 5.12)
// registry manifest reports exactly the audited coverage.

#include "pxgTestFramework.h"

#include "foundation/PxPreprocessor.h"

#if PX_SUPPORT_GPU_PHYSX

#include "PxgCudaShim.h"
#include "PxgVulkanCudaContext.h"
#include "PxgSlangKernelManifest.h"  // generated 487-entry registry manifest
#include "cudamanager/PxCudaContext.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::PxCudaContext;
using physx::PxCudaKernelParam;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgVulkanKernelWrangler;

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static std::vector<uint8_t> wrLoader(const char* file) { return loadShaderBlob(file); }

PXG_TEST(p3KernelWrangler)
{
    PxgSlangComputeDevice* dev = testDevice();

    struct AxpyBuffers { uint64_t x, y, out; };
    struct AxpyParams { AxpyBuffers bufs; float alpha; uint32_t count; };

    // A manifest indexed by kernel id: id 2 is the mapped AXPY kernel; the
    // others are unmapped placeholder slots (like the registry's not-yet-ported
    // rows).  The wrangler and shim share this id space.
    const PxgKernelDef manifest[] = {
        { nullptr, nullptr, 0, 0 },                                   // id 0 (unmapped)
        { nullptr, nullptr, 0, 0 },                                   // id 1 (unmapped)
        { "npShimAxpy.spv", "shimAxpy", uint32_t(sizeof(AxpyParams)), 0 },  // id 2 (mapped)
        { nullptr, nullptr, 0, 0 },                                   // id 3 (unmapped)
    };
    const uint32_t MAP_ID = 2, count = 4;
    PxgCudaShim shim(dev, manifest, count, &wrLoader);
    PxCudaContext* ctx = physx::gpuslang::PxgCreateVulkanCudaContext(&shim, manifest, count);

    PxgVulkanKernelWrangler wrangler(manifest, count);
    PXG_CHECK_MSG(wrangler.count() == count, "wrangler reports the manifest size");
    PXG_CHECK_MSG(wrangler.mappedCount() == 1, "wrangler counts the mapped kernels");
    PXG_CHECK_MSG(wrangler.isMapped(MAP_ID), "the AXPY id is mapped");
    PXG_CHECK_MSG(!wrangler.isMapped(0) && !wrangler.isMapped(999), "unmapped/OOB ids are not mapped");
    PXG_CHECK_MSG(wrangler.getFunction(MAP_ID) != nullptr, "getFunction resolves a mapped id");
    PXG_CHECK_MSG(wrangler.getFunction(0) == nullptr, "getFunction rejects an unmapped id");
    PXG_CHECK_MSG(wrangler.getFunction(999) == nullptr, "getFunction rejects an OOB id");

    // Launch AXPY BY ID through the wrangler-resolved handle.
    const uint32_t N = 2048;
    const float alpha = 2.25f;
    std::mt19937 rng(9182736);
    auto rf = [&]() { return -4.f + 8.f * float(rng() % 1000000) / 1000000.f; };
    std::vector<float> x(N), y(N);
    for (uint32_t i = 0; i < N; ++i) { x[i] = rf(); y[i] = rf(); }

    CUdeviceptr dx = 0, dy = 0, dOut = 0;
    ctx->memAlloc(&dx, N * 4);
    ctx->memAlloc(&dy, N * 4);
    ctx->memAlloc(&dOut, N * 4);
    ctx->memcpyHtoD(dx, x.data(), N * 4);
    ctx->memcpyHtoD(dy, y.data(), N * 4);
    ctx->memsetD32(dOut, 0, N);

    AxpyParams prm;
    prm.bufs.x = uint64_t(dx);
    prm.bufs.y = uint64_t(dy);
    prm.bufs.out = uint64_t(dOut);
    prm.alpha = alpha;
    prm.count = N;
    PxCudaKernelParam kp[1] = { { &prm, sizeof(prm) } };

    // getFunction(id) -> launchKernel(cuFunction, ...) : the engine's exact path.
    ctx->launchKernel(wrangler.getFunction(MAP_ID), (N + 63) / 64, 1, 1, 64, 1, 1, 0, nullptr,
                      kp, sizeof(kp), nullptr, __FILE__, __LINE__);
    ctx->streamSynchronize(nullptr);

    std::vector<float> out(N, 0.f);
    ctx->memcpyDtoH(out.data(), dOut, N * 4);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
        if (f2u(alpha * x[i] + y[i]) != f2u(out[i]))
            ++mm;
    PXG_CHECK_MSG(mm == 0, "launch-by-id through the wrangler is bitwise correct (%zu bad)", mm);

    ctx->memFree(dx); ctx->memFree(dy); ctx->memFree(dOut);
    ctx->release();

    // A wrangler over the GENERATED 487-entry registry manifest must report the
    // audited coverage (ties the wrangler to the real PxgKernelNames.h registry).
    {
        using physx::gpuslang::gPxgSlangKernelManifest;
        using physx::gpuslang::PXG_SLANG_KERNEL_COUNT;
        using physx::gpuslang::PXG_SLANG_KERNEL_MAPPED;
        std::vector<PxgKernelDef> reg(PXG_SLANG_KERNEL_COUNT);
        for (unsigned i = 0; i < PXG_SLANG_KERNEL_COUNT; ++i)
            reg[i] = PxgKernelDef{ gPxgSlangKernelManifest[i].spirvFile, gPxgSlangKernelManifest[i].entry, 0, 0 };
        PxgVulkanKernelWrangler regWrangler(reg.data(), PXG_SLANG_KERNEL_COUNT);
        PXG_CHECK_MSG(regWrangler.count() == PXG_SLANG_KERNEL_COUNT, "registry wrangler size == 487");
        PXG_CHECK_MSG(regWrangler.mappedCount() == PXG_SLANG_KERNEL_MAPPED,
                      "registry wrangler resolves exactly the mapped kernels (%u vs %u)",
                      regWrangler.mappedCount(), PXG_SLANG_KERNEL_MAPPED);
        std::printf("  wrangler over the registry manifest: %u/%u kernel ids resolve to a handle\n",
                    regWrangler.mappedCount(), PXG_SLANG_KERNEL_COUNT);
    }
}

}  // namespace

#else  // !PX_SUPPORT_GPU_PHYSX

#include "pxgTestFramework.h"
namespace { PXG_TEST(p3KernelWrangler) { std::printf("  NOTE: PX_SUPPORT_GPU_PHYSX=0 -> skipped\n"); } }

#endif
