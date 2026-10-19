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

// RUNTIME_PLAN.md P4 test: the kernel-wrangler coupling seam.  It runs the REAL
// engine KernelWrangler (cudamanager) via the REAL PxgCudaKernelWranglerManager
// (gpucommon) — the exact concrete type the gpu subsystem ctors demand — over
// the Vulkan bridge.  The wrangler ctor name-matches every registry kernel
// against the live function table (PxgVulkanKernelTable.cpp, built from the
// manifest) and calls our PxCudaContext::moduleGetFunction, which returns
// toFunc(id) = id+1.  So getCuFunction(id) (5.12: the manager IS the
// KernelWrangler) yields our
// launch encoding for each of the 123 mapped kernels — the exact
// getCuFunction(id) -> launchKernel path every gpu* launch site uses, now driven
// by REAL engine code.  (The launch-by-id path itself is proven bit-exact in
// p3KernelWrangler; here we prove the ENGINE wrangler produces the same handle.)

#include "pxgTestFramework.h"

#include "foundation/PxPreprocessor.h"

#if PX_SUPPORT_GPU_PHYSX

#include "foundation/PxFoundation.h"
#include "foundation/PxAllocatorCallback.h"
#include "foundation/PxErrorCallback.h"
#include "foundation/PxPhysicsVersion.h"

#include "PxgCudaShim.h"
#include "PxgVulkanCudaContext.h"
#include "PxgSlangKernelManifest.h"           // generated registry manifest (cudaName/spirvFile/entry)
#include "PxgKernelWrangler.h"                 // gpucommon: the REAL PxgCudaKernelWranglerManager
#include "CudaKernelWrangler.h"                // cudamanager: the REAL KernelWrangler + getCuFunction
#include "cudamanager/PxCudaContext.h"
#include "cudamanager/PxCudaContextManager.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::PxCudaContextManager;
using physx::KernelWrangler;
using physx::PxgCudaKernelWranglerManager;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::gPxgSlangKernelManifest;
using physx::gpuslang::PXG_SLANG_KERNEL_COUNT;
using physx::gpuslang::PXG_SLANG_KERNEL_MAPPED;

struct WrAllocator : physx::PxAllocatorCallback
{
    void* allocate(size_t size, const char*, const char*, int) override
    {
        void* p = nullptr;
        if (posix_memalign(&p, 16, size ? size : 1))
            return nullptr;
        return p;
    }
    void deallocate(void* p) override { free(p); }
};
struct WrError : physx::PxErrorCallback  // silent: the 378 unmapped kernels report "not found"
{
    void reportError(physx::PxErrorCode::Enum, const char*, const char*, int) override {}
};

static std::vector<uint8_t> wrLoader(const char* file) { return loadShaderBlob(file); }

// The bridge's id -> CUfunction encoding (PxgVulkanCudaContext.cpp toFunc).
static CUfunction expectedFunc(uint32_t id) { return reinterpret_cast<CUfunction>(uintptr_t(id) + 1u); }

PXG_TEST(p4Wrangler)
{
    WrAllocator alloc;
    WrError err;
    physx::PxFoundation* foundation = PxCreateFoundation(PX_PHYSICS_VERSION, alloc, err);
    PXG_CHECK_MSG(foundation != nullptr, "PxCreateFoundation");
    if (!foundation)
        return;

    {
        PxgSlangComputeDevice* dev = testDevice();

        // The context's manifest IS the generated registry (enum-indexed), so
        // our moduleGetFunction resolves the registry kernel names by id.
        std::vector<PxgKernelDef> reg(PXG_SLANG_KERNEL_COUNT);
        for (unsigned i = 0; i < PXG_SLANG_KERNEL_COUNT; ++i)
            reg[i] = PxgKernelDef{ gPxgSlangKernelManifest[i].spirvFile, gPxgSlangKernelManifest[i].entry, 0u, 0u };

        PxgCudaShim shim(dev, reg.data(), PXG_SLANG_KERNEL_COUNT, &wrLoader);
        PxCudaContextManager* ccm =
            physx::gpuslang::PxgCreateVulkanCudaContextManager(dev, &shim, reg.data(), PXG_SLANG_KERNEL_COUNT);
        PXG_CHECK_MSG(ccm != nullptr, "Vulkan PxCudaContextManager");

        if (ccm)
        {
            // The REAL engine wrangler manager over the Vulkan context.  Its
            // KernelWrangler base ctor resolves all PXG_SLANG_KERNEL_COUNT registry
            // names through the live function table + our moduleGetFunction.
            // (5.12: PxgCudaKernelWranglerManager derives from KernelWrangler
            // instead of owning one behind getKernelWrangler().)
            PxgCudaKernelWranglerManager wranglerMgr(*ccm, err);
            KernelWrangler* kw = &wranglerMgr;
            PXG_CHECK_MSG(kw != nullptr, "real KernelWrangler constructed");

            if (kw)
            {
                // For every MAPPED registry kernel, the real wrangler must have
                // resolved getCuFunction(id) to our toFunc(id) encoding.  (We
                // only query mapped ids: getCuFunction PX_ASSERTs non-null, and
                // unmapped slots are intentionally null.)
                uint32_t mapped = 0, correct = 0;
                for (unsigned i = 0; i < PXG_SLANG_KERNEL_COUNT; ++i)
                {
                    if (!gPxgSlangKernelManifest[i].spirvFile)
                        continue;
                    ++mapped;
                    if (kw->getCuFunction(uint16_t(i)) == expectedFunc(i))
                        ++correct;
                }
                PXG_CHECK_MSG(mapped == PXG_SLANG_KERNEL_MAPPED,
                              "manifest has the audited mapped count (%u vs %u)", mapped, PXG_SLANG_KERNEL_MAPPED);
                PXG_CHECK_MSG(correct == mapped,
                              "real KernelWrangler resolves every mapped id to our launch encoding (%u/%u)",
                              correct, mapped);
                std::printf("  real engine KernelWrangler resolves %u/%u registry kernels over Vulkan (P4 seam)\n",
                            correct, PXG_SLANG_KERNEL_COUNT);
            }
        }  // wranglerMgr destroyed here (frees the KernelWrangler), ccm still alive

        if (ccm)
            ccm->release();
    }

    foundation->release();
}

}  // namespace

#else  // !PX_SUPPORT_GPU_PHYSX

#include "pxgTestFramework.h"
namespace { PXG_TEST(p4Wrangler) { std::printf("  NOTE: PX_SUPPORT_GPU_PHYSX=0 -> skipped\n"); } }

#endif
