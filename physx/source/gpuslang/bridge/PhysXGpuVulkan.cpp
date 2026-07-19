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

// PhysXGpuVulkan.cpp — RUNTIME_PLAN.md P3.5: the C-ABI GPU-module export library.
//
// The real PhysX core reaches its GPU backend by dlopen'ing a shared object
// (libPhysXGpu_64.so) and dlsym'ing a fixed set of unmangled C symbols; on
// Linux it rejects the module unless PxCreatePhysXGpu, PxCreateCudaContextManager
// and PxGetSuggestedCudaDeviceOrdinal all resolve non-null
// (PxPhysXGpuModuleLoader.cpp:233).  This TU is the Vulkan drop-in that exports
// exactly those symbols (plus the rest the loader dlsym's), so the module loader
// and the ScScene wiring stay untouched — the scene receives our Vulkan-backed
// PxCudaContextManager via PxSceneDesc::cudaContextManager exactly as it would a
// CUDA one.
//
// Scope of THIS layer:
//   * PxCreateCudaContextManager  -> fully wired: stands up the process Vulkan
//     device + shim + enum-indexed kernel manifest and returns the bridge's
//     PxCudaContextManager (memory / stream / event / copy work today; kernel
//     launch works for kernels whose manifest rows are fully populated).
//   * PxGetSuggestedCudaDeviceOrdinal -> 0 when a Vulkan device comes up.
//   * PxCreatePhysXGpu -> returns a valid PxPhysXGpu whose ~13 subsystem
//     factories (broadphase, AABB, narrowphase, simulation controller, dynamics)
//     are the per-subsystem Vulkan bring-up still ahead (P4): each returns null
//     for now.  The symbol MUST exist and return non-null or the loader disables
//     GPU entirely, so the object is real even though its factories are stubs.
//   * the CUDART fatbin/function-table shims are no-ops: a SPIR-V world has no
//     fatbins; the bridge resolves kernels through the manifest, not that table.

#include "foundation/PxPreprocessor.h"

#if PX_SUPPORT_GPU_PHYSX

#include "PxPhysXGpu.h"               // physx/source/physxgpu/include: the interface + export decls
#include "PxgVulkanCudaContext.h"     // bridge: PxgCreateVulkanCudaContextManager
#include "PxgVulkanMemoryManager.h"   // bridge: the Vulkan PxsMemoryManager (P4 memory seam)
#include "PxgCudaShim.h"              // gpuslang/include: PxgCudaShim, PxgKernelDef, PxgSpirvLoader
#include "PxgSlangComputeDevice.h"    // gpuslang/include: PxgCreateVulkanComputeDevice
#include "PxgSlangKernelManifest.h"   // gpuslang/include: generated enum-indexed registry manifest
#include "PxgHeapMemAllocator.h"      // gpucommon: the REAL buddy heap allocator manager (reused)
#include "PxgKernelWrangler.h"        // gpucommon: the REAL PxgCudaKernelWranglerManager (reused)

// P5 GPU-scene wiring: the concrete Pxg* subsystem types the factories construct.
#include "PxgCudaBroadPhaseSap.h"           // gpubroadphase
#include "PxgAABBManager.h"                 // gpubroadphase (PxgAABBManager + PxgBoundsArray)
#include "PxgNphaseImplementationContext.h" // gpunarrowphase
#include "PxgSimulationController.h"        // gpusimulationcontroller
#include "PxgDynamicsContext.h"             // gpusolver (PGS)
#include "PxgTGSDynamicsContext.h"          // gpusolver (TGS)

#include "foundation/PxAllocator.h"   // PX_NEW
#include "foundation/PxErrorCallback.h"
#include "foundation/PxFoundation.h"  // PxGetErrorCallback

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

namespace physx
{
namespace gpuslang
{
namespace
{

// SPIR-V blob loader: resolves a manifest spirvFile name against $PXG_SHADER_DIR
// (the same convention the harness uses), falling back to the bare name.  Used
// lazily by the shim when a pipeline is first launched.
std::vector<uint8_t> envSpirvLoader(const char* file)
{
    std::vector<uint8_t> blob;
    if (!file)
        return blob;
    const char* dir = std::getenv("PXG_SHADER_DIR");
    const std::string path = dir ? (std::string(dir) + "/" + file) : std::string(file);
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f)
        return blob;
    const std::streamsize n = f.tellg();
    if (n <= 0)
        return blob;
    f.seekg(0);
    blob.resize(size_t(n));
    f.read(reinterpret_cast<char*>(blob.data()), n);
    if (!f)
        blob.clear();
    return blob;
}

// The process-global Vulkan runtime.  A Vulkan device is a process singleton, so
// PxCreateCudaContextManager hands out managers over ONE shared device/shim (one
// scene at a time — the P3 single-queue model); the objects live until process
// exit.  Lazily created on first PxCreateCudaContextManager /
// PxGetSuggestedCudaDeviceOrdinal, guarded by a mutex.
struct VulkanGpuRuntime
{
    PxgSlangComputeDevice* device = nullptr;
    PxgCudaShim* shim = nullptr;
    std::vector<PxgKernelDef> manifest;
    bool tried = false;
};

VulkanGpuRuntime& runtimeState()
{
    static VulkanGpuRuntime s;
    return s;
}

std::mutex& runtimeMutex()
{
    static std::mutex m;
    return m;
}

// Returns the runtime, creating it once; null if no Vulkan device is available.
VulkanGpuRuntime* ensureRuntime()
{
    std::lock_guard<std::mutex> lock(runtimeMutex());
    VulkanGpuRuntime& r = runtimeState();
    if (r.tried)
        return r.device ? &r : nullptr;
    r.tried = true;

    PxgSlangDeviceDesc desc;
    // A production drop-in must PREFER real hardware: default to false and let
    // PXG_PREFER_SOFTWARE_DEVICE=1 opt into a CPU device (llvmpipe/WARP) for CI.
    // The device selector still falls back to whatever Vulkan 1.2+ device exists
    // when no discrete GPU is present, so a software-only CI box still comes up.
    const char* preferSw = std::getenv("PXG_PREFER_SOFTWARE_DEVICE");
    desc.preferSoftwareDevice = (preferSw != nullptr) && (std::strcmp(preferSw, "0") != 0);
    desc.useDeviceLocalMemory = std::getenv("PXG_DEVICE_V2") != nullptr;
    r.device = PxgCreateVulkanComputeDevice(desc);
    if (!r.device)
        return nullptr;

    // Build the enum-indexed kernel manifest from the generated registry.  The
    // spirvFile/entry pointers reference static string data in the generated
    // header; pushConstantSize and numTextures stay 0 until the owning module is
    // brought up on Vulkan (P4).  Index stays aligned with PxgKernelIds.
    r.manifest.resize(PXG_SLANG_KERNEL_COUNT);
    for (unsigned i = 0; i < PXG_SLANG_KERNEL_COUNT; ++i)
        r.manifest[i] = PxgKernelDef{ gPxgSlangKernelManifest[i].spirvFile,
                                      gPxgSlangKernelManifest[i].entry, 0u, 0u };

    static PxgSpirvLoader loader = &envSpirvLoader;
    r.shim = new PxgCudaShim(r.device, r.manifest.data(), uint32_t(r.manifest.size()), loader);
    return &r;
}

// One-line diagnostic for a GPU subsystem factory not yet implemented on the
// Vulkan backend (P4).  Because the module advertises a working context manager
// (so the core enables GPU and accepts it), an engine that then requests GPU
// simulation would otherwise hit a bare null-deref here with no clue why; this
// makes the intermediate state debuggable.
void warnNotImplemented(const char* method)
{
    std::fprintf(stderr,
        "[PhysXGpuVulkan] %s is not implemented on the Vulkan backend yet "
        "(RUNTIME_PLAN.md P4); returning null. Keep GPU-simulation scene flags "
        "(eENABLE_GPU_DYNAMICS / GPU broadphase) disabled until the Vulkan "
        "subsystem suite lands.\n", method);
}

// The PxPhysXGpu the module returns.  Its subsystem factories are the remaining
// per-module Vulkan bring-up (P4); each warns and returns null for now.  A
// process singleton — release() is a no-op and the storage is reclaimed at exit.
class PhysXGpuVulkan : public physx::PxPhysXGpu
{
public:
    static PhysXGpuVulkan& getInstance()
    {
        static PhysXGpuVulkan s;
        return s;
    }

    void release() PX_OVERRIDE {}

    PxsParticleBuffer* createParticleBuffer(PxU32, PxU32, PxCudaContextManager&) PX_OVERRIDE { warnNotImplemented("createParticleBuffer"); return NULL; }
    PxsParticleAndDiffuseBuffer* createParticleAndDiffuseBuffer(PxU32, PxU32, PxU32, PxCudaContextManager&) PX_OVERRIDE { warnNotImplemented("createParticleAndDiffuseBuffer"); return NULL; }
    PxsParticleClothBuffer* createParticleClothBuffer(PxU32, PxU32, PxU32, PxU32, PxU32, PxCudaContextManager&) PX_OVERRIDE { warnNotImplemented("createParticleClothBuffer"); return NULL; }
    PxsParticleRigidBuffer* createParticleRigidBuffer(PxU32, PxU32, PxU32, PxCudaContextManager&) PX_OVERRIDE { warnNotImplemented("createParticleRigidBuffer"); return NULL; }

    // P4 memory seam (real): the Vulkan PxsMemoryManager (device->shim,
    // host->malloc) feeding the REAL gpucommon buddy allocator manager, verbatim
    // as the reference wires createGpuMemoryManager -> createGpuHeapMemoryAllocatorManager.
    PxsMemoryManager* createGpuMemoryManager(PxCudaContextManager* cudaContextManager) PX_OVERRIDE
    {
        return PX_NEW(PxgVulkanMemoryManager)(cudaContextManager->getCudaContext());
    }
    PxsHeapMemoryAllocatorManager* createGpuHeapMemoryAllocatorManager(const PxU32 heapCapacity, PxsMemoryManager* memoryManager, const PxU32 /*gpuComputeVersion*/) PX_OVERRIDE
    {
        return PX_NEW(PxgHeapMemoryAllocatorManager)(heapCapacity, memoryManager);
    }
    // P4 wrangler seam (real): the REAL gpucommon PxgCudaKernelWranglerManager,
    // whose KernelWrangler resolves each mapped kernel id to our toFunc(id)
    // encoding via the live function table + our moduleGetFunction.  Cached for
    // the process (single-context model) as the interface contract expects the
    // same instance back and the singleton PxPhysXGpu owns it until process exit.
    PxsKernelWranglerManager* getGpuKernelWranglerManager(PxCudaContextManager* cudaContextManager) PX_OVERRIDE
    {
        static PxsKernelWranglerManager* s =
            PX_NEW(PxgCudaKernelWranglerManager)(*cudaContextManager, PxGetFoundation().getErrorCallback());
        return s;
    }

    // P5 GPU-scene factories (real): each placement-news the concrete Pxg* type,
    // verbatim as physxgpu/src/PxgPhysXGpu.cpp wires them.  gpuComputeVersion==0 is
    // the only path the CUDA reference takes too (the else returns NULL there).
    Bp::BroadPhase* createGpuBroadPhase(const PxGpuBroadPhaseDesc& desc, PxsKernelWranglerManager* gpuKernelWrangler, PxCudaContextManager* cudaContextManager,
                                        PxU32 gpuComputeVersion, const PxGpuDynamicsMemoryConfig& config, PxsHeapMemoryAllocatorManager* heapMemoryManager, PxU64 contextID) PX_OVERRIDE
    {
        if (gpuComputeVersion == 0)
            return PX_PLACEMENT_NEW(PX_ALLOC(sizeof(PxgCudaBroadPhaseSap), "PxgCudaBroadPhaseSap"), PxgCudaBroadPhaseSap)(desc,
                static_cast<PxgCudaKernelWranglerManager*>(gpuKernelWrangler), cudaContextManager, config, static_cast<PxgHeapMemoryAllocatorManager*>(heapMemoryManager), contextID);
        return NULL;
    }

    Bp::AABBManagerBase* createGpuAABBManager(PxsKernelWranglerManager* gpuKernelWrangler, PxCudaContextManager* cudaContextManager, const PxU32 gpuComputeVersion,
                                              const PxGpuDynamicsMemoryConfig& config, PxsHeapMemoryAllocatorManager* heapMemoryManager,
                                              Bp::BroadPhase& bp, Bp::BoundsArray& boundsArray, PxFloatArrayPinnedSafe& contactDistance,
                                              PxU32 maxNbAggregates, PxU32 maxNbShapes, PxVirtualAllocator& allocator, PxU64 contextID,
                                              PxPairFilteringMode::Enum kineKineFilteringMode, PxPairFilteringMode::Enum staticKineFilteringMode) PX_OVERRIDE
    {
        if (gpuComputeVersion == 0)
            return PX_PLACEMENT_NEW(PX_ALLOC(sizeof(PxgAABBManager), "PxgAABBManager"), PxgAABBManager)(
                static_cast<PxgCudaKernelWranglerManager*>(gpuKernelWrangler), cudaContextManager, static_cast<PxgHeapMemoryAllocatorManager*>(heapMemoryManager),
                config, bp, boundsArray, contactDistance, maxNbAggregates, maxNbShapes, allocator, contextID, kineKineFilteringMode, staticKineFilteringMode);
        return NULL;
    }

    Bp::BoundsArray* createGpuBounds(PxVirtualAllocator& allocator) PX_OVERRIDE
    {
        return PX_PLACEMENT_NEW(PX_ALLOC(sizeof(PxgBoundsArray), "PxgBoundsArray"), PxgBoundsArray)(allocator);
    }

    PxvNphaseImplementationContext* createGpuNphaseImplementationContext(PxsContext& context, PxsKernelWranglerManager* gpuKernelWrangler,
                                                                         PxvNphaseImplementationFallback* fallbackForUnsupportedCMs,
                                                                         const PxGpuDynamicsMemoryConfig& gpuDynamicsConfig, void* contactStreamBase, void* patchStreamBase, void* forceAndIndiceStreamBase,
                                                                         PxBoundsArrayPinned& bounds, IG::IslandSim* islandSim,
                                                                         physx::Dy::Context* dynamicsContext, const PxU32 /*gpuComputeVersion*/,
                                                                         PxsHeapMemoryAllocatorManager* heapMemoryManager, bool useGPUBP) PX_OVERRIDE
    {
        return PX_PLACEMENT_NEW(PX_ALLOC(sizeof(PxgNphaseImplementationContext), "PxgNphaseImplementationContext"), PxgNphaseImplementationContext)(context, gpuKernelWrangler, fallbackForUnsupportedCMs, gpuDynamicsConfig, contactStreamBase, patchStreamBase, forceAndIndiceStreamBase,
            bounds, islandSim, dynamicsContext, static_cast<PxgHeapMemoryAllocatorManager*>(heapMemoryManager), useGPUBP);
    }

    PxsSimulationController* createGpuSimulationController(PxsKernelWranglerManager* gpuWranglerManagers, PxCudaContextManager* cudaContextManager,
                                                          Dy::Context* dynamicContext, PxvNphaseImplementationContext* npContext, Bp::BroadPhase* bp,
                                                          const bool useGpuBroadphase, PxsSimulationControllerCallback* callback, const PxU32 /*gpuComputeVersion*/,
                                                          PxsHeapMemoryAllocatorManager* heapMemoryManager, const PxU32 maxSoftBodyContacts, const PxU32 maxFemClothContacts,
                                                          const PxU32 maxParticleContacts, const PxU32 collisionStackSizeBytes, bool enableBodyAccelerations) PX_OVERRIDE
    {
        return PX_PLACEMENT_NEW(PX_ALLOC(sizeof(PxgSimulationController), "PxgSimulationController"),
            PxgSimulationController)(gpuWranglerManagers, cudaContextManager, static_cast<PxgDynamicsContext*>(dynamicContext), static_cast<PxgNphaseImplementationContext*>(npContext),
            bp, useGpuBroadphase, callback, static_cast<PxgHeapMemoryAllocatorManager*>(heapMemoryManager), maxSoftBodyContacts,
            maxFemClothContacts, maxParticleContacts, collisionStackSizeBytes, enableBodyAccelerations);
    }

    Dy::Context* createGpuDynamicsContext(Cm::FlushPool& taskPool, PxsKernelWranglerManager* gpuKernelWrangler, PxCudaContextManager* cudaContextManager,
                                          const PxGpuDynamicsMemoryConfig& config, IG::SimpleIslandManager& islandManager, PxU32 maxNumPartitions, PxU32 maxNumStaticPartitions,
                                          bool enableStabilization, bool useEnhancedDeterminism, bool solveArticulationContactLast, PxReal maxBiasCoefficient, PxU32 /*gpuComputeVersion*/,
                                          PxvSimStats& simStats, PxsHeapMemoryAllocatorManager* heapMemoryManager, bool frictionEveryIteration, bool externalForcesEveryTgsIterationEnabled,
                                          PxSolverType::Enum solverType, PxReal lengthScale, bool enableDirectGPUAPI, PxU64 contextID, bool isResidualReportingEnabled) PX_OVERRIDE
    {
        if (solverType == PxSolverType::eTGS)
            return PX_PLACEMENT_NEW(PX_ALLOC(sizeof(PxgTGSDynamicsContext), "PxgDynamicsContext"), PxgTGSDynamicsContext)(taskPool, gpuKernelWrangler, cudaContextManager, config, islandManager,
                maxNumPartitions, maxNumStaticPartitions, enableStabilization, useEnhancedDeterminism, solveArticulationContactLast, maxBiasCoefficient, simStats,
                static_cast<PxgHeapMemoryAllocatorManager*>(heapMemoryManager), externalForcesEveryTgsIterationEnabled, lengthScale, enableDirectGPUAPI, contextID, isResidualReportingEnabled);
        return PX_PLACEMENT_NEW(PX_ALLOC(sizeof(PxgDynamicsContext), "PxgDynamicsContext"), PxgDynamicsContext)(taskPool, gpuKernelWrangler, cudaContextManager, config, islandManager,
            maxNumPartitions, maxNumStaticPartitions, enableStabilization, useEnhancedDeterminism, solveArticulationContactLast, maxBiasCoefficient, simStats,
            static_cast<PxgHeapMemoryAllocatorManager*>(heapMemoryManager), frictionEveryIteration, lengthScale, enableDirectGPUAPI, contextID, isResidualReportingEnabled);
    }
};

} // anonymous namespace
} // namespace gpuslang
} // namespace physx

//----------------------------------------------------------------------------//
// C-ABI exports.  Signatures/linkage come from the extern "C" declarations in
// PxPhysXGpu.h; on Linux PX_PHYSX_GPU_API carries default visibility so these
// symbols are exported unmangled under exactly the names the loader dlsym's.
//----------------------------------------------------------------------------//

physx::PxPhysXGpu* PxCreatePhysXGpu()
{
    return &physx::gpuslang::PhysXGpuVulkan::getInstance();
}

physx::PxCudaContextManager* PxCreateCudaContextManager(physx::PxFoundation& /*foundation*/,
                                                        const physx::PxCudaContextManagerDesc& /*desc*/,
                                                        physx::PxProfilerCallback* /*profilerCallback*/,
                                                        bool /*launchSynchronous*/)
{
    // The bridge manager reports errors through its own result codes, so the
    // foundation/profiler globals the CUDA lib sets for its static-lib config
    // are not needed here; desc.deviceOrdinal maps onto the (single) Vulkan
    // device selected at runtime creation.
    physx::gpuslang::VulkanGpuRuntime* r = physx::gpuslang::ensureRuntime();
    if (!r)
        return NULL;
    return physx::gpuslang::PxgCreateVulkanCudaContextManager(r->device, r->shim, r->manifest.data(),
                                                              uint32_t(r->manifest.size()));
}

void PxSetPhysXGpuProfilerCallback(physx::PxProfilerCallback* /*profilerCallback*/)
{
}

void PxSetPhysXGpuFoundationInstance(physx::PxFoundation& /*foundation*/)
{
}

int PxGetSuggestedCudaDeviceOrdinal(physx::PxErrorCallback& /*errc*/)
{
    // 0 == the single Vulkan device; -1 when none is available (GPU disabled).
    return physx::gpuslang::ensureRuntime() ? 0 : -1;
}

// CUDART fatbin/function-table shims: no fatbins exist in a SPIR-V world, and
// the Vulkan bridge resolves kernels through the manifest rather than this
// table, so these are inert (still exported because the loader dlsym's them).
void PxGpuCudaRegisterFunction(int /*moduleIndex*/, const char* /*functionName*/)
{
}

void** PxGpuCudaRegisterFatBinary(void* /*fatBin*/)
{
    return NULL;
}

// PxGpuGetCudaFunctionTable / ...Size / PxGpuGetCudaModuleTable / ...Size live in
// PxgVulkanKernelTable.cpp — the function table is populated from the manifest so
// the real engine KernelWrangler resolves the mapped kernels through our context.

physx::PxPhysicsGpu* PxGpuCreatePhysicsGpu()
{
    return NULL;
}

#endif // PX_SUPPORT_GPU_PHYSX
