// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its

// P9.40 -- the UNMAPPED-launch guard in the bridge's PxCudaContext.
//
// Before this, a kernel that was in the registry but had no SPIR-V was simply absent
// from the CUDA function table; the engine wrangler left its slot null, and a host
// launch through that slot returned InvalidValue from the very first line of
// launchKernel -- before any trace, before any name.  On a Release build almost every
// engine call site checks that result only under a per-module debug macro, so the
// kernel was skipped and nothing anywhere said so.  A capture of all 33 scenes found
// zero launch-failure lines while static reading showed the FEM -> rigid impulse
// accumulation and two articulation body-update stages launched every step.
//
// Now the unmapped name IS in the table, moduleGetFunction hands back a non-null
// SENTINEL for it (still returning NotFound), and launchKernel names and counts the
// refusal.  This test pins the three properties that make that safe:
//
//  U1. behaviour is UNCHANGED: a launch through the sentinel is refused with the same
//      InvalidValue a null slot produced, and nothing is dispatched;
//  U2. it is COUNTED, by kernel id, and only for sentinel handles -- a genuinely null
//      handle stays anonymous and does not count (there is nothing to name);
//  U3. a MAPPED kernel is untouched: its handle is the plain encoding, fromFunc accepts
//      it, and its unmapped count stays 0 after a real launch.

#include "pxgTestFramework.h"

#include "foundation/PxPreprocessor.h"
#if PX_SUPPORT_GPU_PHYSX

#include "PxgCudaShim.h"
#include "PxgVulkanCudaContext.h"
#include "cudamanager/PxCudaContext.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::PxCudaContext;
using physx::PxCudaKernelParam;
using physx::PxCUresult;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;

static std::vector<uint8_t> ctxLoader(const char* file) { return loadShaderBlob(file); }

PXG_TEST(p3UnmappedLaunch)
{
    PxgSlangComputeDevice* dev = testDevice();

    struct AxpyBuffers { uint64_t x, y, out; };
    struct AxpyParams { AxpyBuffers bufs; float alpha; uint32_t count; };
    // Row 0 is mapped (the shim's own axpy); row 1 is a registry name WITHOUT SPIR-V,
    // shaped exactly like an unmapped manifest row: null spirvFile, null entry, a
    // cudaName.
    const PxgKernelDef manifest[] = {
        { "npShimAxpy.spv", "shimAxpy", uint32_t(sizeof(AxpyParams)), 0, "shimAxpy" },
        { nullptr, nullptr, 0u, 0u, "definitelyUnmappedKernel" },
    };
    PxgCudaShim shim(dev, manifest, 2, &ctxLoader);
    PxCudaContext* ctx = physx::gpuslang::PxgCreateVulkanCudaContext(&shim, manifest, 2);
    PXG_CHECK_MSG(ctx != nullptr, "PxCudaContext created");
    if (!ctx)
        return;

    CUmodule mod = nullptr;
    PXG_CHECK_MSG(ctx->moduleLoadDataEx(&mod, nullptr, 0, nullptr, nullptr) == 0, "moduleLoadDataEx");

    // ---- U3: the mapped kernel resolves as before ------------------------------------
    CUfunction fn = nullptr;
    PXG_CHECK_MSG(ctx->moduleGetFunction(&fn, mod, "shimAxpy") == 0 && fn != nullptr,
                  "a MAPPED name resolves with Success and a non-null handle");

    // ---- the unmapped name: NotFound, but a NON-NULL sentinel ---------------------------
    CUfunction sentinel = nullptr;
    const PxCUresult r = ctx->moduleGetFunction(&sentinel, mod, "definitelyUnmappedKernel");
    PXG_CHECK_MSG(r != 0, "an UNMAPPED registry name still reports NotFound (the wrangler's error "
                          "path is unchanged)");
    PXG_CHECK_MSG(sentinel != nullptr,
                  "...but hands back a non-null sentinel so a launch through it can be named");
    PXG_CHECK_MSG(sentinel != fn, "the sentinel is not a mapped handle");

    CUfunction unknown = reinterpret_cast<CUfunction>(uintptr_t(0xBEEF));
    PXG_CHECK_MSG(ctx->moduleGetFunction(&unknown, mod, "notInTheRegistryAtAll") != 0 &&
                  unknown == reinterpret_cast<CUfunction>(uintptr_t(0xBEEF)),
                  "a name that is not in the registry at all leaves the handle untouched");

    // ---- U1/U2: launching the sentinel is refused AND counted ---------------------------
    AxpyParams prm = {};
    PxCudaKernelParam kp[1] = { { &prm, sizeof(prm) } };
    for (int i = 0; i < 3; ++i)
    {
        const PxCUresult lr = ctx->launchKernel(sentinel, 1, 1, 1, 64, 1, 1, 0, nullptr,
                                                kp, sizeof(kp), nullptr, __FILE__, __LINE__);
        PXG_CHECK_MSG(lr != 0, "launch %d through the sentinel is REFUSED (InvalidValue), as a "
                               "null slot was", i);
    }
    PXG_CHECK_MSG(physx::gpuslang::pxgUnmappedLaunchCount(ctx, 1u) == 3u,
                  "the unmapped kernel's launch count is 3, got %u",
                  physx::gpuslang::pxgUnmappedLaunchCount(ctx, 1u));

    // the void** launch form takes the same refuse path
    void* vp[1] = { &prm };
    const PxCUresult lv = ctx->launchKernel(sentinel, 1, 1, 1, 64, 1, 1, 0, nullptr, vp, nullptr,
                                            __FILE__, __LINE__);
    PXG_CHECK_MSG(lv != 0, "the void** form refuses the sentinel too");
    PXG_CHECK_MSG(physx::gpuslang::pxgUnmappedLaunchCount(ctx, 1u) == 4u,
                  "...and counts it (4), got %u", physx::gpuslang::pxgUnmappedLaunchCount(ctx, 1u));

    // ---- U2: a genuinely NULL handle is still refused, and stays anonymous --------------
    const PxCUresult ln = ctx->launchKernel(nullptr, 1, 1, 1, 64, 1, 1, 0, nullptr,
                                            kp, sizeof(kp), nullptr, __FILE__, __LINE__);
    PXG_CHECK_MSG(ln != 0, "a null handle is refused");
    PXG_CHECK_MSG(physx::gpuslang::pxgUnmappedLaunchCount(ctx, 0u) == 0u &&
                  physx::gpuslang::pxgUnmappedLaunchCount(ctx, 1u) == 4u,
                  "a null handle names nothing, so no counter moves");

    // ---- U3: a real launch of the mapped kernel works and does not count ----------------
    const uint32_t N = 256;
    CUdeviceptr dx = 0, dy = 0, dOut = 0;
    PXG_CHECK_MSG(ctx->memAlloc(&dx, N * 4) == 0 && ctx->memAlloc(&dy, N * 4) == 0 &&
                  ctx->memAlloc(&dOut, N * 4) == 0, "memAlloc");
    std::vector<float> ones(N, 1.0f);
    ctx->memcpyHtoD(dx, ones.data(), N * 4);
    ctx->memcpyHtoD(dy, ones.data(), N * 4);
    prm.bufs.x = uint64_t(dx); prm.bufs.y = uint64_t(dy); prm.bufs.out = uint64_t(dOut);
    prm.alpha = 2.0f; prm.count = N;
    const PxCUresult lm = ctx->launchKernel(fn, (N + 63) / 64, 1, 1, 64, 1, 1, 0, nullptr,
                                            kp, sizeof(kp), nullptr, __FILE__, __LINE__);
    PXG_CHECK_MSG(lm == 0, "the mapped kernel launches");
    ctx->streamSynchronize(nullptr);
    std::vector<float> out(N, 0.f);
    ctx->memcpyDtoH(out.data(), dOut, N * 4);
    PXG_CHECK_MSG(out[0] == 3.0f && out[N - 1] == 3.0f, "and computed 2*1+1 = 3 (got %g, %g)",
                  double(out[0]), double(out[N - 1]));
    PXG_CHECK_MSG(physx::gpuslang::pxgUnmappedLaunchCount(ctx, 0u) == 0u,
                  "the mapped kernel's unmapped count stays 0");

    ctx->memFree(dx); ctx->memFree(dy); ctx->memFree(dOut);
    // release() reports the four skipped launches; with no foundation alive in this
    // harness it goes to stderr, which is fine -- the report is the point.
    ctx->release();
    std::printf("    [ note ] sentinel launches refused and counted: 4; null launches: anonymous\n");
}
} // namespace

#endif // PX_SUPPORT_GPU_PHYSX
