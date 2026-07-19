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

// RUNTIME_PLAN.md P3 test: the PxCudaContextManager wrapper (the object the
// scene receives via PxSceneDesc::cudaContextManager).  Verifies the manager
// hands out a working PxCudaContext, reports device properties, and that the
// deprecated buffer helper templates (which drive the protected *Internal
// virtuals) alloc / copy / run a real kernel end-to-end.

#include "pxgTestFramework.h"

#include "foundation/PxPreprocessor.h"

#if PX_SUPPORT_GPU_PHYSX

#include "PxgCudaShim.h"
#include "PxgVulkanCudaContext.h"
#include "cudamanager/PxCudaContext.h"
#include "cudamanager/PxCudaContextManager.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::PxCudaContext;
using physx::PxCudaContextManager;
using physx::PxCudaKernelParam;
using physx::PxCUresult;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static std::vector<uint8_t> mgrLoader(const char* file) { return loadShaderBlob(file); }

PXG_TEST(p3CudaContextManager)
{
    PxgSlangComputeDevice* dev = testDevice();

    struct AxpyBuffers { uint64_t x, y, out; };
    struct AxpyParams { AxpyBuffers bufs; float alpha; uint32_t count; };
    const PxgKernelDef manifest[] = {
        { "npShimAxpy.spv", "shimAxpy", uint32_t(sizeof(AxpyParams)), 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &mgrLoader);

    PxCudaContextManager* mgr =
        physx::gpuslang::PxgCreateVulkanCudaContextManager(dev, &shim, manifest, 1);
    PXG_CHECK_MSG(mgr != nullptr, "PxCudaContextManager created");
    if (!mgr)
        return;

    // Manager surface the scene/simulation-controller reads.
    PXG_CHECK_MSG(mgr->contextIsValid(), "contextIsValid");
    PXG_CHECK_MSG(mgr->getCudaContext() != nullptr, "getCudaContext non-null");
    PXG_CHECK_MSG(mgr->supportsArchSM60(), "reports the modern SM level");
    PXG_CHECK_MSG(mgr->getMaxThreadsPerBlock() >= 64, "getMaxThreadsPerBlock sane");
    PXG_CHECK_MSG(mgr->getSharedMemPerBlock() > 0, "getSharedMemPerBlock sane");
    PXG_CHECK_MSG(mgr->getDeviceName() != nullptr, "getDeviceName non-null");
    // Recursive acquire/release must balance (the engine acquires around work).
    mgr->acquireContext();
    PXG_CHECK_MSG(mgr->tryAcquireContext(), "tryAcquireContext");
    mgr->releaseContext();
    mgr->releaseContext();
    size_t freeB = 0, totalB = 0;
    mgr->getDeviceMemoryInfo(freeB, totalB);
    PXG_CHECK_MSG(totalB > 0 && freeB <= totalB, "getDeviceMemoryInfo sane");

    // Run AXPY through the manager's context using CUDA-shaped memory ops.
    PxCudaContext* ctx = mgr->getCudaContext();
    const uint32_t N = 2048;
    const float alpha = 0.5f;
    std::mt19937 rng(864213);
    auto rf = [&]() { return -3.f + 6.f * float(rng() % 1000000) / 1000000.f; };
    std::vector<float> x(N), y(N);
    for (uint32_t i = 0; i < N; ++i) { x[i] = rf(); y[i] = rf(); }

    CUdeviceptr dx = 0, dy = 0, dOut = 0;
    ctx->memAlloc(&dx, N * 4);
    ctx->memAlloc(&dy, N * 4);
    ctx->memAlloc(&dOut, N * 4);
    ctx->memcpyHtoD(dx, x.data(), N * 4);
    ctx->memcpyHtoD(dy, y.data(), N * 4);
    ctx->memsetD32(dOut, 0, N);

    CUfunction fn = nullptr;
    ctx->moduleGetFunction(&fn, nullptr, "shimAxpy");
    AxpyParams prm;
    prm.bufs.x = uint64_t(dx);
    prm.bufs.y = uint64_t(dy);
    prm.bufs.out = uint64_t(dOut);
    prm.alpha = alpha;
    prm.count = N;
    PxCudaKernelParam kp[1] = { { &prm, sizeof(prm) } };
    ctx->launchKernel(fn, (N + 63) / 64, 1, 1, 64, 1, 1, 0, nullptr, kp, sizeof(kp), nullptr, __FILE__, __LINE__);
    ctx->streamSynchronize(nullptr);

    std::vector<float> out(N, 0.f);
    ctx->memcpyDtoH(out.data(), dOut, N * 4);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
        if (f2u(alpha * x[i] + y[i]) != f2u(out[i]))
            ++mm;
    PXG_CHECK_MSG(mm == 0, "kernel via the manager's context is bitwise correct (%zu bad)", mm);
    ctx->memFree(dx); ctx->memFree(dy); ctx->memFree(dOut);

    // Deprecated buffer-helper templates drive the protected *Internal virtuals:
    // allocDeviceBuffer -> allocDeviceBufferInternal, copyHToD/copyDToH -> the
    // copy*Internal pair.  A host->device->host round-trip must be exact.
    {
        float* db = nullptr;
        mgr->allocDeviceBuffer<float>(db, N);
        PXG_CHECK_MSG(db != nullptr, "allocDeviceBuffer (Internal) returns a device pointer");
        mgr->copyHToD<float>(db, x.data(), N);
        std::vector<float> back(N, 0.f);
        mgr->copyDToH<float>(back.data(), db, N);
        PXG_CHECK_MSG(std::memcmp(back.data(), x.data(), N * 4) == 0,
                      "Internal H2D/D2H round-trip is exact");
        mgr->freeDeviceBuffer(db);
    }

    mgr->release();
}

}  // namespace

#else  // !PX_SUPPORT_GPU_PHYSX

#include "pxgTestFramework.h"
namespace { PXG_TEST(p3CudaContextManager) { std::printf("  NOTE: PX_SUPPORT_GPU_PHYSX=0 -> skipped\n"); } }

#endif
