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

// RUNTIME_PLAN.md P4 test: the AABB-manager subsystem seam.  It constructs the
// REAL, unmodified PxgAABBManager (the type PxgPhysXGpu::createGpuAABBManager
// static_casts to) over the Vulkan context, wrapping the REAL PxgCudaBroadPhaseSap
// (P4.4) and driven by both prior seams (heap P4.2 + wrangler P4.3).  The AABB
// manager's ctor allocates its bookkeeping through the buddy allocator and, most
// tellingly, calls getGPUBroadPhase(bp).setGPUAABBManager(this) — coupling itself
// to the concrete broadphase.  The test proves that coupling ran over Vulkan with
// no engine edit: the manager constructs, and the broadphase's getAABBManager()
// now points back at it (the exact bp<->aabb link every ScScene wires).  Kernel
// launches (update()) need full scene data and stay for later per-module bring-up.

#include "pxgTestFramework.h"

#include "foundation/PxPreprocessor.h"

#if PX_SUPPORT_GPU_PHYSX

#include "foundation/PxFoundation.h"
#include "foundation/PxAllocatorCallback.h"
#include "foundation/PxErrorCallback.h"
#include "foundation/PxPhysicsVersion.h"
#include "foundation/PxAllocator.h"
#include "foundation/PxArray.h"
#include "CmPinnableArray.h"                  // Cm::PinnableArray (5.12: replaces PxPinnedArray)

#include "PxBroadPhase.h"                      // PxGpuBroadPhaseDesc
#include "PxSceneDesc.h"                       // PxGpuDynamicsMemoryConfig
#include "PxFiltering.h"                       // PxPairFilteringMode

#include "PxgCudaShim.h"
#include "PxgVulkanCudaContext.h"
#include "PxgMemoryManager.h"                // gpucommon: createPxgMemoryManager
#include "PxgHeapMemAllocator.h"
#include "PxgSlangKernelManifest.h"
#include "PxgKernelWrangler.h"
#include "PxgCudaBroadPhaseSap.h"
#include "PxgAABBManager.h"                    // gpubroadphase: the REAL AABB manager + PxgBoundsArray
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
using physx::PxgAABBManager;
using physx::PxgBoundsArray;
using physx::PxGpuBroadPhaseDesc;
using physx::PxGpuDynamicsMemoryConfig;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::PxgMemoryManager;
using physx::gpuslang::gPxgSlangKernelManifest;
using physx::gpuslang::PXG_SLANG_KERNEL_COUNT;

struct AabbAllocator : physx::PxAllocatorCallback
{
    void* allocate(size_t size, const char*, const char*, int) override
    {
        void* p = nullptr;
        if (posix_memalign(&p, 16, size ? size : 1)) return nullptr;
        return p;
    }
    void deallocate(void* p) override { free(p); }
};
struct AabbError : physx::PxErrorCallback
{
    void reportError(physx::PxErrorCode::Enum, const char*, const char*, int) override {}
};

static std::vector<uint8_t> aabbLoader(const char* file) { return loadShaderBlob(file); }

PXG_TEST(p4Aabb)
{
    AabbAllocator alloc;
    AabbError err;
    physx::PxFoundation* foundation = PxCreateFoundation(PX_PHYSICS_VERSION, alloc, err);
    PXG_CHECK_MSG(foundation != nullptr, "PxCreateFoundation");
    if (!foundation)
        return;

    {
        PxgSlangComputeDevice* dev = testDevice();
        std::vector<PxgKernelDef> reg(PXG_SLANG_KERNEL_COUNT);
        for (unsigned i = 0; i < PXG_SLANG_KERNEL_COUNT; ++i)
            reg[i] = PxgKernelDef{ gPxgSlangKernelManifest[i].spirvFile, gPxgSlangKernelManifest[i].entry, 0u, 0u };

        PxgCudaShim shim(dev, reg.data(), PXG_SLANG_KERNEL_COUNT, &aabbLoader);
        PxCudaContextManager* ccm =
            physx::gpuslang::PxgCreateVulkanCudaContextManager(dev, &shim, reg.data(), PXG_SLANG_KERNEL_COUNT);
        PXG_CHECK_MSG(ccm != nullptr, "Vulkan PxCudaContextManager");
        PxCudaContext* ctx = ccm ? ccm->getCudaContext() : nullptr;

        if (ctx)
        {
            physx::PxsMemoryManager* memMgr = physx::createPxgMemoryManager(ccm);
            PXG_CHECK_MSG(memMgr != nullptr, "createPxgMemoryManager");
            if (memMgr)
            {
                const physx::PxU32 heapCapacity = 2u * 1024u * 1024u;
                PxgHeapMemoryAllocatorManager heapMgr(heapCapacity, *static_cast<PxgMemoryManager*>(memMgr));
                physx::PxgAllocatorDesc allocDesc = physx::PxgCreateAllocatorDesc(heapMgr);
                PxgCudaKernelWranglerManager wranglerMgr(*ccm, err);

                PxGpuBroadPhaseDesc bpDesc;
                PxGpuDynamicsMemoryConfig memCfg;
                memCfg.foundLostPairsCapacity = 4u * 1024u;
                memCfg.foundLostAggregatePairsCapacity = 1u * 1024u;

                // The concrete broadphase (P4.4) the AABB manager wraps.
                void* bpMem = PX_ALLOC(sizeof(PxgCudaBroadPhaseSap), "PxgCudaBroadPhaseSap");
                PxgCudaBroadPhaseSap* bp = PX_PLACEMENT_NEW(bpMem, PxgCudaBroadPhaseSap)(
                    bpDesc, &wranglerMgr, ccm, memCfg, allocDesc, 0);
                PXG_CHECK_MSG(bp != nullptr, "broadphase constructed");
                PXG_CHECK_MSG(bp->getAABBManager() == nullptr, "broadphase starts with no AABB manager");

                // The pinned contact-distance array + bounds array the AABB manager
                // takes, over the heap manager's pinned host allocators -- exactly as
                // the 5.12 engine wires them (ScScene.cpp:834/843: mapped bounds for
                // the direct-API pipeline, contact distance on the pinned allocator).
                physx::Cm::PinnableArray<physx::PxReal> contactDistance(*heapMgr.mPinnedHostMemoryAllocator);

                void* baMem = PX_ALLOC(sizeof(PxgBoundsArray), "PxgBoundsArray");
                PxgBoundsArray* boundsArray = PX_PLACEMENT_NEW(baMem, PxgBoundsArray)(*heapMgr.mPinnedHostMappedMemoryAllocator);

                // Construct the REAL concrete AABB manager exactly as the factory does.
                void* amMem = PX_ALLOC(sizeof(PxgAABBManager), "PxgAABBManager");
                PxgAABBManager* aabb = PX_PLACEMENT_NEW(amMem, PxgAABBManager)(
                    &wranglerMgr, ccm, allocDesc, memCfg, *bp, *boundsArray, contactDistance,
                    /*maxNbAggregates*/ 8u, /*maxNbShapes*/ 64u, /*contextID*/ 0,
                    physx::PxPairFilteringMode::eKEEP, physx::PxPairFilteringMode::eKEEP);
                PXG_CHECK_MSG(aabb != nullptr, "real PxgAABBManager constructed over Vulkan");

                // Teeth: the AABB ctor coupled itself to the concrete broadphase via
                // setGPUAABBManager(this) — the exact bp<->aabb link every scene wires.
                PXG_CHECK_MSG(bp->getAABBManager() == aabb,
                              "AABB manager registered itself back on the broadphase (setGPUAABBManager)");

                std::printf("  real PxgAABBManager constructed over Vulkan and coupled to the broadphase (P4 subsystem seam)\n");

                aabb->~PxgAABBManager();
                PX_FREE(amMem);
                boundsArray->~PxgBoundsArray();
                PX_FREE(baMem);
                bp->~PxgCudaBroadPhaseSap();
                PX_FREE(bpMem);
            }
            PX_DELETE(memMgr);
        }

        if (ccm)
            ccm->release();
    }

    foundation->release();
}

}  // namespace

#else  // !PX_SUPPORT_GPU_PHYSX

#include "pxgTestFramework.h"
namespace { PXG_TEST(p4Aabb) { std::printf("  NOTE: PX_SUPPORT_GPU_PHYSX=0 -> skipped\n"); } }

#endif
