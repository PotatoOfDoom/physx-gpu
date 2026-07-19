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

// RUNTIME_PLAN.md P4 test: the memory-manager coupling seam.  It runs the REAL
// gpucommon buddy allocator (PxgHeapMemoryAllocator / PxgHeapMemoryAllocatorManager)
// — the exact engine type the gpu subsystems demand — over the Vulkan bridge,
// via a ~30-line PxgVulkanMemoryManager whose device callback wraps the
// PxCudaContext's memAlloc/memFree.  It allocates several device buffers through
// the buddy allocator, proves each interior (buddy-split) address round-trips a
// host<->device memcpy (the shim's interval map resolving it), and frees them.
// This is the first real engine-code-over-Vulkan integration in P4.

#include "pxgTestFramework.h"

#include "foundation/PxPreprocessor.h"

#if PX_SUPPORT_GPU_PHYSX

#include "foundation/PxFoundation.h"
#include "foundation/PxAllocatorCallback.h"
#include "foundation/PxErrorCallback.h"
#include "foundation/PxPhysicsVersion.h"
#include "foundation/PxPreprocessor.h"        // PX_FL

#include "PxgCudaShim.h"
#include "PxgVulkanCudaContext.h"
#include "PxgVulkanMemoryManager.h"
#include "PxgHeapMemAllocator.h"               // gpucommon: the REAL buddy allocator
#include "cudamanager/PxCudaContext.h"
#include "cudamanager/PxCudaContextManager.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::PxCudaContext;
using physx::PxCudaContextManager;
using physx::PxgHeapMemoryAllocatorManager;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgVulkanMemoryManager;

struct HeapAllocator : physx::PxAllocatorCallback
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
struct HeapError : physx::PxErrorCallback
{
    void reportError(physx::PxErrorCode::Enum, const char*, const char*, int) override {}
};

static std::vector<uint8_t> heapLoader(const char* file) { return loadShaderBlob(file); }

PXG_TEST(p4Heap)
{
    HeapAllocator alloc;
    HeapError err;
    physx::PxFoundation* foundation = PxCreateFoundation(PX_PHYSICS_VERSION, alloc, err);
    PXG_CHECK_MSG(foundation != nullptr, "PxCreateFoundation");
    if (!foundation)
        return;

    {
        PxgSlangComputeDevice* dev = testDevice();
        const PxgKernelDef manifest[] = { { nullptr, nullptr, 0, 0 } };  // no kernels needed; just memory
        PxgCudaShim shim(dev, manifest, 1, &heapLoader);
        PxCudaContextManager* ccm =
            physx::gpuslang::PxgCreateVulkanCudaContextManager(dev, &shim, manifest, 1);
        PXG_CHECK_MSG(ccm != nullptr, "Vulkan PxCudaContextManager");
        PxCudaContext* ctx = ccm ? ccm->getCudaContext() : nullptr;
        PXG_CHECK_MSG(ctx != nullptr, "PxCudaContext");

        if (ctx)
        {
            // The Vulkan memory manager (device->shim, host->malloc) feeding the
            // REAL gpucommon buddy allocator manager, exactly as the engine wires
            // createGpuMemoryManager -> createGpuHeapMemoryAllocatorManager.
            PxgVulkanMemoryManager memMgr(ctx);
            {
                const physx::PxU32 heapCapacity = 2u * 1024u * 1024u;  // buddy block size
                PxgHeapMemoryAllocatorManager heapMgr(heapCapacity, &memMgr);
                PXG_CHECK_MSG(heapMgr.mDeviceMemoryAllocators != nullptr, "device buddy allocator built");

                // Allocate several device buffers through the buddy allocator and
                // round-trip data through each interior (split) device address.
                const uint32_t N = 512;                 // uint32 elements per buffer
                const uint32_t bytes = N * 4;
                const int NBUF = 6;
                void* dptr[NBUF] = {};
                size_t bad = 0, nulls = 0;
                for (int b = 0; b < NBUF; ++b)
                {
                    dptr[b] = heapMgr.mDeviceMemoryAllocators->allocate(bytes, physx::PxsHeapStats::eOTHER, PX_FL);
                    if (!dptr[b]) { ++nulls; continue; }

                    std::vector<uint32_t> in(N), out(N, 0);
                    for (uint32_t i = 0; i < N; ++i)
                        in[i] = (uint32_t(b) << 24) ^ (i * 2654435761u);

                    const CUdeviceptr dev64 = CUdeviceptr(uintptr_t(dptr[b]));
                    ctx->memcpyHtoD(dev64, in.data(), bytes);
                    ctx->memcpyDtoH(out.data(), dev64, bytes);
                    for (uint32_t i = 0; i < N; ++i)
                        if (out[i] != in[i])
                            ++bad;
                }
                PXG_CHECK_MSG(nulls == 0, "buddy allocator returned a device address for every request (%zu null)", nulls);
                PXG_CHECK_MSG(bad == 0, "every buddy-allocated device address round-trips host<->device (%zu bad)", bad);

                // Addresses must be distinct (the buddy split hands out unique interior offsets).
                size_t dupe = 0;
                for (int i = 0; i < NBUF; ++i)
                    for (int j = i + 1; j < NBUF; ++j)
                        if (dptr[i] && dptr[i] == dptr[j])
                            ++dupe;
                PXG_CHECK_MSG(dupe == 0, "buddy allocations are distinct (%zu dupes)", dupe);

                PXG_CHECK_MSG(heapMgr.getDeviceMemorySize() > 0, "heap reports non-zero device memory in use");

                for (int b = 0; b < NBUF; ++b)
                    if (dptr[b])
                        heapMgr.mDeviceMemoryAllocators->deallocate(dptr[b]);

                std::printf("  real gpucommon buddy allocator runs over the Vulkan memory manager (P4 seam)\n");
            }  // heapMgr destroyed here (frees its blocks via ctx->memFree), ctx still alive
        }

        if (ccm)
            ccm->release();
    }

    foundation->release();
}

}  // namespace

#else  // !PX_SUPPORT_GPU_PHYSX

#include "pxgTestFramework.h"
namespace { PXG_TEST(p4Heap) { std::printf("  NOTE: PX_SUPPORT_GPU_PHYSX=0 -> skipped\n"); } }

#endif
