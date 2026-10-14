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

// RUNTIME_PLAN.md P3.1 test: drive a real Slang kernel end-to-end through the
// ACTUAL physx::PxCudaContext virtual interface (the real PhysX header), backed
// by the Vulkan runtime.  This is the integration seam the whole GPU engine
// calls: memAlloc / memcpyHtoD / moduleGetFunction / launchKernel (with a
// CUDA-shaped parameter block carrying device addresses) / streamSynchronize /
// memcpyDtoH.  Proves the CUDA-context replacement satisfies PxCudaContext
// without the CUDA toolkit.

#include "pxgTestFramework.h"

#include "foundation/PxPreprocessor.h"

#if PX_SUPPORT_GPU_PHYSX

#include "PxgCudaShim.h"
#include "PxgVulkanCudaContext.h"
#include "cudamanager/PxCudaContext.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <random>
#include <thread>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::PxCudaContext;
using physx::PxCudaKernelParam;
using physx::PxCUresult;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

static std::vector<uint8_t> ctxLoader(const char* file) { return loadShaderBlob(file); }

PXG_TEST(p3CudaContext)
{
    PxgSlangComputeDevice* dev = testDevice();

    struct AxpyBuffers { uint64_t x, y, out; };
    struct AxpyParams { AxpyBuffers bufs; float alpha; uint32_t count; };
    const PxgKernelDef manifest[] = {
        { "npShimAxpy.spv", "shimAxpy", uint32_t(sizeof(AxpyParams)), 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &ctxLoader);

    PxCudaContext* ctx = physx::gpuslang::PxgCreateVulkanCudaContext(&shim, manifest, 1);
    PXG_CHECK_MSG(ctx != nullptr, "PxCudaContext created");
    if (!ctx)
        return;

    const uint32_t N = 4096;
    const float alpha = 1.75f;
    std::mt19937 rng(20240501);
    auto rf = [&]() { return -5.f + 10.f * float(rng() % 1000000) / 1000000.f; };
    std::vector<float> x(N), y(N);
    for (uint32_t i = 0; i < N; ++i) { x[i] = rf(); y[i] = rf(); }

    // --- everything below goes through the abstract PxCudaContext* only ---
    CUdeviceptr dx = 0, dy = 0, dOut = 0;
    PXG_CHECK_MSG(ctx->memAlloc(&dx, N * 4) == 0, "memAlloc x");
    PXG_CHECK_MSG(ctx->memAlloc(&dy, N * 4) == 0, "memAlloc y");
    PXG_CHECK_MSG(ctx->memAlloc(&dOut, N * 4) == 0, "memAlloc out");
    PXG_CHECK_MSG(dx != 0 && dy != 0 && dOut != 0, "device addresses are non-null");

    PXG_CHECK_MSG(ctx->memcpyHtoD(dx, x.data(), N * 4) == 0, "memcpyHtoD x");
    PXG_CHECK_MSG(ctx->memcpyHtoD(dy, y.data(), N * 4) == 0, "memcpyHtoD y");
    PXG_CHECK_MSG(ctx->memsetD32(dOut, 0, N) == 0, "memsetD32 out");

    CUmodule mod = nullptr;
    PXG_CHECK_MSG(ctx->moduleLoadDataEx(&mod, nullptr, 0, nullptr, nullptr) == 0, "moduleLoadDataEx");
    CUfunction fn = nullptr;
    PXG_CHECK_MSG(ctx->moduleGetFunction(&fn, mod, "shimAxpy") == 0 && fn != nullptr,
                  "moduleGetFunction resolves the kernel");
    CUfunction missing = nullptr;
    PXG_CHECK_MSG(ctx->moduleGetFunction(&missing, mod, "doesNotExist") != 0,
                  "moduleGetFunction rejects an unknown name");

    // CUDA-shaped launch: one parameter struct carrying the three device
    // addresses + scalars, passed as a PxCudaKernelParam (the {data,size} form
    // PhysX uses).  The kernel dereferences the addresses as BDA pointers.
    AxpyParams prm;
    prm.bufs.x = uint64_t(dx);
    prm.bufs.y = uint64_t(dy);
    prm.bufs.out = uint64_t(dOut);
    prm.alpha = alpha;
    prm.count = N;
    PxCudaKernelParam kp[1] = { { &prm, sizeof(prm) } };

    const PxCUresult lr = ctx->launchKernel(fn, (N + 63) / 64, 1, 1, 64, 1, 1, 0, nullptr,
                                            kp, sizeof(kp), nullptr, __FILE__, __LINE__);
    PXG_CHECK_MSG(lr == 0, "launchKernel succeeds through PxCudaContext");
    PXG_CHECK_MSG(ctx->streamSynchronize(nullptr) == 0, "streamSynchronize");

    std::vector<float> out(N, 0.f);
    PXG_CHECK_MSG(ctx->memcpyDtoH(out.data(), dOut, N * 4) == 0, "memcpyDtoH out");

    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
        if (f2u(alpha * x[i] + y[i]) != f2u(out[i]))
            ++mm;
    PXG_CHECK_MSG(mm == 0, "kernel result through PxCudaContext is bitwise correct (%zu bad)", mm);

    // getLastError reflects the last (successful) call.
    PXG_CHECK_MSG(ctx->getLastError() == 0, "getLastError clear after a successful sequence");

    ctx->memFree(dx);
    ctx->memFree(dy);
    ctx->memFree(dOut);
    ctx->release();
}

// PhysX calls the CUDA layer from several worker threads at once: in one step the
// broadphase and narrowphase tasks launch kernels concurrently (an AMD RX 9070 XT
// trace showed convexPlaneNphase_Kernel and bpSignalComplete interleaved, then
// vkBeginCommandBuffer failing / DEVICE_LOST).  The CUDA driver is thread-safe; the
// Vulkan command buffer and the shim's bookkeeping are not, so every call holds
// pxgGpuApiMutex().  Here four threads each run alloc / upload / launch / sync /
// download / free cycles through ONE context, and every result must be bitwise
// right.  Without the lock this crashes or corrupts on llvmpipe as well.
PXG_TEST(p3CudaContextThreads)
{
    PxgSlangComputeDevice* dev = testDevice();

    struct AxpyBuffers { uint64_t x, y, out; };
    struct AxpyParams { AxpyBuffers bufs; float alpha; uint32_t count; };
    const PxgKernelDef manifest[] = {
        { "npShimAxpy.spv", "shimAxpy", uint32_t(sizeof(AxpyParams)), 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &ctxLoader);
    PxCudaContext* ctx = physx::gpuslang::PxgCreateVulkanCudaContext(&shim, manifest, 1);
    PXG_CHECK_MSG(ctx != nullptr, "PxCudaContext created");
    if (!ctx)
        return;
    CUmodule mod = nullptr;
    ctx->moduleLoadDataEx(&mod, nullptr, 0, nullptr, nullptr);
    CUfunction fn = nullptr;
    PXG_CHECK_MSG(ctx->moduleGetFunction(&fn, mod, "shimAxpy") == 0 && fn != nullptr, "kernel resolves");
    if (!fn)
    {
        ctx->release();
        return;
    }

    const uint32_t kThreads = 4, kIterations = 40, N = 2048;
    std::atomic<uint32_t> failedCalls(0), badValues(0), cycles(0);
    auto worker = [&](uint32_t tid) {
        std::mt19937 rng(1000u + tid);
        std::vector<float> x(N), y(N), out(N);
        for (uint32_t it = 0; it < kIterations; ++it)
        {
            for (uint32_t i = 0; i < N; ++i)
            {
                x[i] = float(int(rng() % 2001u) - 1000) * 0.01f;
                y[i] = float(int(rng() % 2001u) - 1000) * 0.01f;
            }
            const float alpha = 0.5f + float(tid) + float(it) * 0.25f;
            CUdeviceptr dx = 0, dy = 0, dOut = 0;
            uint32_t fails = 0;
            fails += ctx->memAlloc(&dx, N * 4) != 0;
            fails += ctx->memAlloc(&dy, N * 4) != 0;
            fails += ctx->memAlloc(&dOut, N * 4) != 0;
            if (fails)
            {
                failedCalls += fails;
                continue;
            }
            CUstream stream = nullptr;
            fails += ctx->streamCreate(&stream, 0) != 0;
            fails += ctx->memcpyHtoD(dx, x.data(), N * 4) != 0;
            fails += ctx->memcpyHtoD(dy, y.data(), N * 4) != 0;
            AxpyParams prm;
            prm.bufs.x = uint64_t(dx);
            prm.bufs.y = uint64_t(dy);
            prm.bufs.out = uint64_t(dOut);
            prm.alpha = alpha;
            prm.count = N;
            PxCudaKernelParam kp[1] = { { &prm, sizeof(prm) } };
            fails += ctx->launchKernel(fn, (N + 63) / 64, 1, 1, 64, 1, 1, 0, stream,
                                       kp, sizeof(kp), nullptr, __FILE__, __LINE__) != 0;
            fails += ctx->streamSynchronize(stream) != 0;
            fails += ctx->memcpyDtoH(out.data(), dOut, N * 4) != 0;
            uint32_t bad = 0;
            for (uint32_t i = 0; i < N; ++i)
                if (f2u(alpha * x[i] + y[i]) != f2u(out[i]))
                    ++bad;
            fails += ctx->streamDestroy(stream) != 0;
            fails += ctx->memFree(dx) != 0;
            fails += ctx->memFree(dy) != 0;
            fails += ctx->memFree(dOut) != 0;
            failedCalls += fails;
            badValues += bad;
            ++cycles;
        }
    };
    std::vector<std::thread> threads;
    for (uint32_t t = 0; t < kThreads; ++t)
        threads.emplace_back(worker, t);
    for (std::thread& th : threads)
        th.join();

    PXG_CHECK_MSG(cycles.load() == kThreads * kIterations, "every cycle ran (%u of %u)", cycles.load(), kThreads * kIterations);
    PXG_CHECK_MSG(failedCalls.load() == 0, "no call failed under concurrency (%u failed)", failedCalls.load());
    PXG_CHECK_MSG(badValues.load() == 0, "every concurrent result is bitwise correct (%u bad values)", badValues.load());
    ctx->release();
}

}  // namespace

#else  // !PX_SUPPORT_GPU_PHYSX

#include "pxgTestFramework.h"
namespace { PXG_TEST(p3CudaContext) { std::printf("  NOTE: PX_SUPPORT_GPU_PHYSX=0 -> skipped\n"); } }

#endif
