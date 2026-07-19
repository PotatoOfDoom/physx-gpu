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

// RUNTIME_PLAN.md P4 test: the first CONCRETE gpu-subsystem coupling seam.  It
// constructs the REAL, unmodified engine broadphase — PxgCudaBroadPhaseSap, the
// exact concrete type PxgPhysXGpu::createGpuBroadPhase static_casts to — over the
// Vulkan context, driven by BOTH prior seams: the real PxgCudaKernelWranglerManager
// (P4.3) and the real PxgHeapMemoryAllocatorManager over PxgVulkanMemoryManager
// (P4.2).  The broadphase ctor takes the context lock, allocates its ~40 device
// buffers + host-mapped descriptors through the buddy allocator, and creates its
// stream + event through the bridge (including the first direct-driver seam,
// cuCtxGetStreamPriorityRange, shimmed by PxgVulkanDriverShim.cpp).  The test
// proves the whole ctor path runs over Vulkan with no engine edit: the object
// constructs; device bytes are charged to the eBROADPHASE heap group (the ctor
// really allocated through our buddy allocator, not a null factory); and — the
// load-bearing anti-stub check — a probe buffer carved from the SAME pool after
// the broadphase round-trips real host<->device data, so the device memory the
// broadphase now shares is genuinely usable (a broken device memAlloc fails here,
// whereas the byte count alone is only allocator bookkeeping and would not).
// Kernel LAUNCHES (update()) need full scene data and stay for later per-module
// bring-up; this pins the construction seam every scene using the GPU broadphase
// depends on.

#include "pxgTestFramework.h"

#include "foundation/PxPreprocessor.h"

#if PX_SUPPORT_GPU_PHYSX

#include "foundation/PxFoundation.h"
#include "foundation/PxAllocatorCallback.h"
#include "foundation/PxErrorCallback.h"
#include "foundation/PxPhysicsVersion.h"
#include "foundation/PxAllocator.h"           // PX_ALLOC / PX_FREE / PX_PLACEMENT_NEW
#include "foundation/PxPreprocessor.h"        // PX_FL

#include "PxBroadPhase.h"                      // PxGpuBroadPhaseDesc
#include "PxSceneDesc.h"                       // PxGpuDynamicsMemoryConfig

#include "PxgCudaShim.h"
#include "PxgVulkanCudaContext.h"
#include "PxgVulkanMemoryManager.h"
#include "PxgHeapMemAllocator.h"               // gpucommon: the REAL buddy allocator manager
#include "PxgSlangKernelManifest.h"            // generated registry manifest (for the wrangler)
#include "PxgKernelWrangler.h"                 // gpucommon: the REAL PxgCudaKernelWranglerManager
#include "PxgCudaBroadPhaseSap.h"              // gpubroadphase: the REAL concrete broadphase
#include "cudamanager/PxCudaContext.h"
#include "cudamanager/PxCudaContextManager.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::PxCudaContext;
using physx::PxCudaContextManager;
using physx::PxgHeapMemoryAllocatorManager;
using physx::PxgCudaKernelWranglerManager;
using physx::PxgCudaBroadPhaseSap;
using physx::PxGpuBroadPhaseDesc;
using physx::PxGpuDynamicsMemoryConfig;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgVulkanMemoryManager;
using physx::gpuslang::gPxgSlangKernelManifest;
using physx::gpuslang::PXG_SLANG_KERNEL_COUNT;

struct BpAllocator : physx::PxAllocatorCallback
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
struct BpError : physx::PxErrorCallback  // silent: unmapped kernel names report "not found"
{
    void reportError(physx::PxErrorCode::Enum, const char*, const char*, int) override {}
};

static std::vector<uint8_t> bpLoader(const char* file) { return loadShaderBlob(file); }

PXG_TEST(p4Broadphase)
{
    BpAllocator alloc;
    BpError err;
    physx::PxFoundation* foundation = PxCreateFoundation(PX_PHYSICS_VERSION, alloc, err);
    PXG_CHECK_MSG(foundation != nullptr, "PxCreateFoundation");
    if (!foundation)
        return;

    {
        PxgSlangComputeDevice* dev = testDevice();

        // The context's manifest IS the generated registry, so the real wrangler
        // manager's KernelWrangler ctor resolves every kernel by name (P4.3).
        std::vector<PxgKernelDef> reg(PXG_SLANG_KERNEL_COUNT);
        for (unsigned i = 0; i < PXG_SLANG_KERNEL_COUNT; ++i)
            reg[i] = PxgKernelDef{ gPxgSlangKernelManifest[i].spirvFile, gPxgSlangKernelManifest[i].entry, 0u, 0u };

        PxgCudaShim shim(dev, reg.data(), PXG_SLANG_KERNEL_COUNT, &bpLoader);
        PxCudaContextManager* ccm =
            physx::gpuslang::PxgCreateVulkanCudaContextManager(dev, &shim, reg.data(), PXG_SLANG_KERNEL_COUNT);
        PXG_CHECK_MSG(ccm != nullptr, "Vulkan PxCudaContextManager");
        PxCudaContext* ctx = ccm ? ccm->getCudaContext() : nullptr;

        if (ctx)
        {
            PxgVulkanMemoryManager memMgr(ctx);
            {
                // Both prior seams, exactly as the engine wires them.
                const physx::PxU32 heapCapacity = 2u * 1024u * 1024u;
                PxgHeapMemoryAllocatorManager heapMgr(heapCapacity, &memMgr);
                PxgCudaKernelWranglerManager wranglerMgr(*ccm, err);
                PXG_CHECK_MSG(heapMgr.mDeviceMemoryAllocators != nullptr, "device buddy allocator built");
                PXG_CHECK_MSG(wranglerMgr.getKernelWrangler() != nullptr, "real KernelWrangler built");

                // A modest found/lost capacity keeps the buffer footprint small but
                // still forces real device allocations through the buddy allocator.
                PxGpuBroadPhaseDesc bpDesc;                       // defaults (4-bit grid snap)
                PxGpuDynamicsMemoryConfig memCfg;
                memCfg.foundLostPairsCapacity = 4u * 1024u;
                memCfg.foundLostAggregatePairsCapacity = 1u * 1024u;

                // Device bytes charged to the broadphase heap group — 0 before the
                // broadphase exists (nothing else uses eBROADPHASE here).
                const physx::PxU64 bpBefore =
                    heapMgr.getDeviceHeapStats().stats[physx::PxsHeapStats::eBROADPHASE];

                // Construct the REAL concrete broadphase exactly as the factory does
                // (PxgPhysXGpu::createGpuBroadPhase): placement-new over PX_ALLOC,
                // static concrete wrangler + heap manager, on the Vulkan context.
                void* mem = PX_ALLOC(sizeof(PxgCudaBroadPhaseSap), "PxgCudaBroadPhaseSap");
                PxgCudaBroadPhaseSap* bp = PX_PLACEMENT_NEW(mem, PxgCudaBroadPhaseSap)(
                    bpDesc, &wranglerMgr, ccm, memCfg, &heapMgr, /*contextID*/ 0);
                PXG_CHECK_MSG(bp != nullptr, "real PxgCudaBroadPhaseSap constructed over Vulkan");

                const physx::PxU64 bpAfter =
                    heapMgr.getDeviceHeapStats().stats[physx::PxsHeapStats::eBROADPHASE];

                // Teeth: the ctor's ~40 device buffers (radix-sort scratch, found/lost
                // pair buffers, region histograms, descriptors, ...) were charged to
                // the broadphase group of our Vulkan-backed buddy allocator — so the
                // group's device byte count is now non-zero.  A stub memory manager or
                // a broadphase that never allocated leaves it at zero.
                PXG_CHECK_MSG(bpBefore == 0,
                              "broadphase heap group starts empty (%llu bytes)", (unsigned long long)bpBefore);
                PXG_CHECK_MSG(bpAfter > 0,
                              "broadphase ctor allocated device memory through the Vulkan buddy allocator (%llu bytes)",
                              (unsigned long long)bpAfter);

                // Teeth for the MEMORY seam itself: the heap byte-count above is only
                // the allocator's bookkeeping (it increments even for a NULL-backed
                // pool).  Prove the device memory the broadphase now shares is REAL by
                // round-tripping data through a probe buffer carved from the same pool
                // AFTER the broadphase took its buffers.  A broken device memAlloc
                // (NULL base) makes the flat address unresolvable -> the copy mismatches.
                const uint32_t N = 256, probeBytes = N * 4;
                void* probe = heapMgr.mDeviceMemoryAllocators->allocate(probeBytes, physx::PxsHeapStats::eOTHER, PX_FL);
                PXG_CHECK_MSG(probe != nullptr, "probe device buffer from the post-broadphase pool");
                size_t badProbe = 0;
                if (probe)
                {
                    std::vector<uint32_t> in(N), out(N, 0);
                    for (uint32_t i = 0; i < N; ++i) in[i] = 0xB47A0000u ^ (i * 2654435761u);
                    const CUdeviceptr d = CUdeviceptr(uintptr_t(probe));
                    ctx->memcpyHtoD(d, in.data(), probeBytes);
                    ctx->memcpyDtoH(out.data(), d, probeBytes);
                    for (uint32_t i = 0; i < N; ++i) if (out[i] != in[i]) ++badProbe;
                    heapMgr.mDeviceMemoryAllocators->deallocate(probe);
                }
                PXG_CHECK_MSG(badProbe == 0,
                              "the broadphase's Vulkan device pool round-trips real data (%zu mismatches)", badProbe);

                std::printf("  real PxgCudaBroadPhaseSap constructed over Vulkan; %llu bytes of device memory allocated via the buddy allocator, pool round-trips real data (P4 subsystem seam)\n",
                            (unsigned long long)bpAfter);

                bp->~PxgCudaBroadPhaseSap();  // frees its buffers via the buddy allocator + ctx
                PX_FREE(mem);
            }  // heapMgr / wranglerMgr destroyed here, ctx still alive
        }

        if (ccm)
            ccm->release();
    }

    foundation->release();
}

}  // namespace

#else  // !PX_SUPPORT_GPU_PHYSX

#include "pxgTestFramework.h"
namespace { PXG_TEST(p4Broadphase) { std::printf("  NOTE: PX_SUPPORT_GPU_PHYSX=0 -> skipped\n"); } }

#endif
