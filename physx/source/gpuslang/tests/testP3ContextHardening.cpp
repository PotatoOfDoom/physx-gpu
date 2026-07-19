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

// RUNTIME_PLAN.md P3.1 hardening test: the PxCudaContext contract details an
// adversarial review surfaced on the CUDA-context bridge.  Three behaviours the
// happy-path p3CudaContext test does not exercise:
//
//   1. MULTI-ARGUMENT launch marshalling.  CUDA lays each kernel argument out
//      at its natural alignment; the bridge must reproduce that layout when the
//      arguments arrive as separate PxCudaKernelParam entries (not a single
//      pre-packed struct).  A tight concatenation would misplace a value that
//      follows a smaller argument.
//   2. ABORT MODE.  After an allocation failure the engine flips the context to
//      abort mode; getLastError() must then report OUT_OF_MEMORY and every op
//      must no-op (return the sticky result without touching the GPU), and the
//      state must recover when abort clears.  (Matches CudaCtx.)
//   3. NAME-RESOLUTION / LAUNCHABILITY agreement.  moduleGetFunction must hand
//      out a launchable handle only for a kernel actually mapped to SPIR-V, so
//      it agrees with the wrangler's isMapped() gate.

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

static std::vector<uint8_t> hardenLoader(const char* file) { return loadShaderBlob(file); }

// CUDA result codes surfaced by the bridge (values from cuda.h's CUresult).
enum { CU_SUCCESS = 0, CU_INVALID_VALUE = 1, CU_OUT_OF_MEMORY = 2 };

PXG_TEST(p3ContextHardening)
{
    PxgSlangComputeDevice* dev = testDevice();

    // Manifest: id 0 is the misalignment-probe kernel (mapped); id 1 is a row
    // with an entry NAME but no SPIR-V (the bring-up state of an unported
    // kernel) — moduleGetFunction must refuse to resolve it to a handle.
    struct MisParams { uint64_t inPtr, outPtr; uint32_t n; uint64_t mul; };
    const PxgKernelDef manifest[] = {
        { "npShimMisalign.spv", "shimMisAdd", uint32_t(sizeof(MisParams)), 0 },  // id 0 (mapped)
        { nullptr, "reservedNoSpirv", 0, 0 },                                    // id 1 (name only)
    };
    PxgCudaShim shim(dev, manifest, 2, &hardenLoader);
    PxCudaContext* ctx = physx::gpuslang::PxgCreateVulkanCudaContext(&shim, manifest, 2);
    PXG_CHECK_MSG(ctx != nullptr, "PxCudaContext created");
    if (!ctx)
        return;

    CUmodule mod = nullptr;
    ctx->moduleLoadDataEx(&mod, nullptr, 0, nullptr, nullptr);

    // ---- (3) name-resolution / launchability agreement ----
    CUfunction fn = nullptr;
    PXG_CHECK_MSG(ctx->moduleGetFunction(&fn, mod, "shimMisAdd") == 0 && fn != nullptr,
                  "moduleGetFunction resolves a kernel mapped to SPIR-V");
    CUfunction noSpirv = reinterpret_cast<CUfunction>(uintptr_t(0xdeadbeef));
    const PxCUresult nsr = ctx->moduleGetFunction(&noSpirv, mod, "reservedNoSpirv");
    PXG_CHECK_MSG(nsr != 0,
                  "moduleGetFunction refuses a name with no compiled SPIR-V (%u)", nsr.value);

    // ---- (1) multi-argument natural-alignment marshalling ----
    // Pass the four fields as SEPARATE PxCudaKernelParam entries.  Under CUDA's
    // natural layout `mul` lands at offset 24; a tight concatenation would place
    // it at 20, so the kernel would read a garbage multiplier.  Both device
    // pointers precede the misalignment, so a wrong layout produces wrong DATA
    // (never a wild write) — a clean, deterministic failure.
    const uint32_t M = 1536;
    const uint32_t MUL = 3;
    std::vector<uint32_t> in(M);
    for (uint32_t i = 0; i < M; ++i)
        in[i] = 2654435761u * (i + 1u);   // arbitrary spread; uint32 arithmetic wraps deterministically

    CUdeviceptr dIn = 0, dOut = 0;
    PXG_CHECK_MSG(ctx->memAlloc(&dIn, M * 4) == 0, "memAlloc in");
    PXG_CHECK_MSG(ctx->memAlloc(&dOut, M * 4) == 0, "memAlloc out");
    ctx->memcpyHtoD(dIn, in.data(), M * 4);
    ctx->memsetD32(dOut, 0, M);

    MisParams mp;
    mp.inPtr = uint64_t(dIn);
    mp.outPtr = uint64_t(dOut);
    mp.n = M;
    mp.mul = MUL;
    PxCudaKernelParam kp[4] = {
        { &mp.inPtr, sizeof(mp.inPtr) },    // 8 bytes -> offset 0
        { &mp.outPtr, sizeof(mp.outPtr) },  // 8 bytes -> offset 8
        { &mp.n, sizeof(mp.n) },            // 4 bytes -> offset 16
        { &mp.mul, sizeof(mp.mul) },        // 8 bytes -> offset 24 (natural; tight would be 20)
    };
    const PxCUresult lr = ctx->launchKernel(fn, (M + 63) / 64, 1, 1, 64, 1, 1, 0, nullptr,
                                            kp, sizeof(kp), nullptr, __FILE__, __LINE__);
    PXG_CHECK_MSG(lr == 0, "multi-arg launchKernel succeeds");
    ctx->streamSynchronize(nullptr);

    std::vector<uint32_t> out(M, 0);
    ctx->memcpyDtoH(out.data(), dOut, M * 4);
    size_t bad = 0;
    for (uint32_t i = 0; i < M; ++i)
        if (out[i] != in[i] * MUL + i)
            ++bad;
    PXG_CHECK_MSG(bad == 0, "multi-arg natural-alignment marshalling is correct (%zu bad)", bad);

    // ---- (2) abort mode ----
    // Seed a buffer with a sentinel, then flip to abort and try to overwrite it.
    const uint32_t SENT = 0xA5A5A5A5u;
    ctx->memsetD32(dOut, SENT, M);
    ctx->streamSynchronize(nullptr);
    PXG_CHECK_MSG(ctx->getLastError() == 0, "getLastError clear before abort");

    ctx->setAbortMode(true);
    PXG_CHECK_MSG(ctx->isInAbortMode(), "context is in abort mode");
    PXG_CHECK_MSG(ctx->getLastError().value == uint32_t(CU_OUT_OF_MEMORY),
                  "getLastError reports OUT_OF_MEMORY in abort mode (%u)", ctx->getLastError().value);
    // This overwrite must be dropped: an op in abort mode does not touch the GPU.
    ctx->memsetD32(dOut, 0u, M);
    // An unsupported call made mid-abort must not record into mLastResult, or it
    // would leave a stale error that survives the abort->clear recovery (which
    // only resets the sticky OUT_OF_MEMORY).  It must still write a null result.
    CUdeviceptr mapped = 0xdead;
    ctx->memHostGetDevicePointer(&mapped, reinterpret_cast<void*>(uintptr_t(0x1000)), 0);
    PXG_CHECK_MSG(mapped == 0, "memHostGetDevicePointer writes a null device pointer");

    ctx->setAbortMode(false);
    PXG_CHECK_MSG(!ctx->isInAbortMode(), "abort mode cleared");
    PXG_CHECK_MSG(ctx->getLastError() == 0, "getLastError healthy again after abort clears");

    ctx->streamSynchronize(nullptr);
    std::vector<uint32_t> chk(M, 0);
    ctx->memcpyDtoH(chk.data(), dOut, M * 4);
    size_t clobbered = 0;
    for (uint32_t i = 0; i < M; ++i)
        if (chk[i] != SENT)
            ++clobbered;
    PXG_CHECK_MSG(clobbered == 0, "abort-mode op was a no-op (buffer intact, %zu clobbered)", clobbered);

    ctx->memFree(dIn);
    ctx->memFree(dOut);
    ctx->release();
}

}  // namespace

#else  // !PX_SUPPORT_GPU_PHYSX

#include "pxgTestFramework.h"
namespace { PXG_TEST(p3ContextHardening) { std::printf("  NOTE: PX_SUPPORT_GPU_PHYSX=0 -> skipped\n"); } }

#endif
