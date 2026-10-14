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
#include "PxgVulkanKernelAbi.h"       // bridge: per-kernel launch-ABI table (P5 Step 4)
#include "PxgSlangComputeDevice.h"    // gpuslang/include: PxgCreateVulkanComputeDevice
#include "PxgSlangEnv.h"              // gpuslang/include: pxgDeviceV2FromEnv (shared with gpuslang-tests)
#include "PxgSlangKernelManifest.h"   // gpuslang/include: generated enum-indexed registry manifest
#include "PxgModulePath.h"            // bridge: module directory + UTF-8 file/env helpers
#include "PxgHeapMemAllocator.h"      // gpucommon: the REAL buddy heap allocator manager (reused)
#include "PxgKernelWrangler.h"        // gpucommon: the REAL PxgCudaKernelWranglerManager (reused)

// P5 GPU-scene wiring: the concrete Pxg* subsystem types the factories construct.
#include "PxgCudaBroadPhaseSap.h"           // gpubroadphase
#include "PxgAABBManager.h"                 // gpubroadphase (PxgAABBManager + PxgBoundsArray)
#include "PxgNphaseImplementationContext.h" // gpunarrowphase
#include "PxgSimulationController.h"        // gpusimulationcontroller
#include "PxgParticleSystem.h"              // gpusimulationcontroller: PxgParticleBuffer (P6.0)
#include "PxgDynamicsContext.h"             // gpusolver (PGS)
#include "PxgTGSDynamicsContext.h"          // gpusolver (TGS)

#include "foundation/PxAllocator.h"   // PX_NEW
#include "foundation/PxErrorCallback.h"
#include "foundation/PxFoundation.h"  // PxGetErrorCallback

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace physx
{
namespace gpuslang
{
namespace
{

// SPIR-V blob loader, used lazily by the shim when a pipeline is first launched.
// A manifest spirvFile name is resolved against, in order:
//   1. $PXG_SHADER_DIR/<file>                 -- when set and non-empty, ONLY there (development
//                                                and test builds point it at their
//                                                own shader directory);
//   2. <dir of this module>/PhysXGpuShaders/<file> -- the layout the SDK build
//                                                produces and installs;
//   3. <file> relative to the working directory (legacy).
// Warn (loudly, once per blob) when a MAPPED kernel's SPIR-V blob cannot be loaded.
// An empty blob silently produces an empty VkShaderModule, whose
// vkCreateComputePipelines fails, and a launch on that kernel then has no pipeline
// -- a failure mode that reads like an environment/driver fault but is really just
// a missing shader directory.  This diagnostic makes the real cause unmissable.
// (See RUNTIME_STATUS.md P7.0 environment note.)
void warnMissingSpirv(const char* file, const std::vector<std::string>& tried)
{
    std::string list;
    for (const std::string& p : tried)
        list += "\n    " + p;
    std::fprintf(stderr,
                 "[gpuslang/bridge] FATAL-ish: SPIR-V blob '%s' is missing or empty. "
                 "The kernels in it cannot run. Looked in:%s\n"
                 "  Ship PhysXGpuShaders/ next to the PhysXGpu library, or set PXG_SHADER_DIR to the "
                 "compiled-shader directory (e.g. <build-gpuslang>/shaders).\n",
                 file, list.c_str());
}

std::vector<uint8_t> envSpirvLoader(const char* file)
{
    std::vector<uint8_t> blob;
    if (!file)
        return blob;
    std::vector<std::string> candidates;
    std::string dir;
    if (pxgGetEnv("PXG_SHADER_DIR", dir) && !dir.empty()) // set-but-empty counts as unset
        candidates.push_back(dir + "/" + file);
    else
    {
        const std::string moduleDir = pxgModuleDirectory();
        if (!moduleDir.empty())
            candidates.push_back(moduleDir + "PhysXGpuShaders/" + file);
        candidates.push_back(file);
    }
    for (const std::string& path : candidates)
    {
        if (pxgReadFile(path, blob))
            return blob;
    }
    warnMissingSpirv(file, candidates);
    return blob;
}

// The engine KernelWrangler (cudamanager/src/CudaKernelWrangler.cpp) resolves every
// kernel of the registry when it is built and reports each one it cannot resolve as
// an eINTERNAL_ERROR, "Could not find CUDA module containing function '<name>'.".
// The kernels the manifest deliberately leaves without SPIR-V (the PxGetPhysicsGpu()
// utilities -- SDF builder, isosurface, sparse grid, anisotropy, smoothing -- and
// kernels nothing launches) would print 73 such errors for every application that
// enables GPU dynamics, although nothing is wrong.  This forwards everything else
// unchanged and drops exactly those notices: a kernel that IS mapped and fails to
// resolve, a name missing from the function table, or any other message still
// reaches the application.  An actual launch of an unmapped kernel is refused,
// counted and reported when the context is released (P9.40), so nothing that
// matters is hidden.
class UnmappedKernelNoticeFilter : public PxErrorCallback
{
public:
    explicit UnmappedKernelNoticeFilter(PxErrorCallback& next) : mNext(next) {}

    void reportError(PxErrorCode::Enum code, const char* message, const char* file, int line) PX_OVERRIDE
    {
        if (code == PxErrorCode::eINTERNAL_ERROR && isUnmappedKernelNotice(message))
        {
            ++mSuppressed;
            return;
        }
        mNext.reportError(code, message, file, line);
    }

    uint32_t suppressed() const { return mSuppressed; }

    static bool isUnmappedKernelNotice(const char* message)
    {
        static const char prefix[] = "Could not find CUDA module containing function '";
        if (!message || std::strncmp(message, prefix, sizeof(prefix) - 1) != 0)
            return false;
        const char* name = message + sizeof(prefix) - 1;
        const char* end = std::strchr(name, '\'');
        if (!end || std::strcmp(end, "'.") != 0)
            return false;
        const size_t len = size_t(end - name);
        for (unsigned i = 0; i < PXG_SLANG_KERNEL_COUNT; ++i)
        {
            const PxgSlangKernelRow& row = gPxgSlangKernelManifest[i];
            if (!row.spirvFile && row.cudaName && std::strlen(row.cudaName) == len &&
                std::strncmp(row.cudaName, name, len) == 0)
                return true;
        }
        return false;
    }

private:
    PxErrorCallback& mNext;
    uint32_t mSuppressed = 0;
};

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

// Per-kernel push-constant blob size, derived from the launch-ABI table
// (bridge/PxgVulkanKernelAbi.cpp — the single source of truth for a kernel's
// CUDA argument sizes).  Filled as each kernel's engine-path launch is brought
// up (RUNTIME_PLAN.md P5 Step 4); the shim validates every launch blob against
// this, so an unlisted kernel (0) is refused rather than dispatched with a bad blob.
uint32_t pushConstantSizeFor(const char* cudaName)
{
    const PxgKernelAbiRow* row = pxgFindKernelAbi(cudaName);
    return row ? pxgKernelAbiPushSize(*row) : 0u;
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
    // PXG_DEVICE_V2: unset/""/"0" -> host-visible (default), "1" -> device-local + staged;
    // anything else warns and stays default.  Same parser as gpuslang-tests.
    desc.useDeviceLocalMemory = physx::gpuslang::pxgDeviceV2FromEnv();
    r.device = PxgCreateVulkanComputeDevice(desc);
    if (!r.device)
        return nullptr;
    std::fprintf(stderr, "[PhysXGpuVulkan] device: %s, memory mode: %s\n", r.device->getDeviceName(),
                 desc.useDeviceLocalMemory ? "device-local + staged (PXG_DEVICE_V2=1)"
                                           : "host-visible (default)");

    // Build the enum-indexed kernel manifest from the generated registry.  The
    // spirvFile/entry pointers reference static string data in the generated
    // header; numTextures stays 0 (only SDF/heightfield 3D-texture kernels set it).
    // pushConstantSize is the compiled push-constant struct size the shim validates
    // each launch blob against; it is filled per kernel as that kernel's launch is
    // brought up on the engine path (RUNTIME_PLAN.md P5 Step 4).  0 means "not yet
    // wired" — the shim rejects the launch with a clear InvalidValue rather than
    // dispatching a mismatched blob.  Index stays aligned with PxgKernelIds.
    r.manifest.resize(PXG_SLANG_KERNEL_COUNT);
    for (unsigned i = 0; i < PXG_SLANG_KERNEL_COUNT; ++i)
        r.manifest[i] = PxgKernelDef{ gPxgSlangKernelManifest[i].spirvFile,
                                      gPxgSlangKernelManifest[i].entry,
                                      pushConstantSizeFor(gPxgSlangKernelManifest[i].cudaName), 0u,
                                      gPxgSlangKernelManifest[i].cudaName };

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

    // P6.0 PBD particle bring-up: mirror physxgpu/src/PxgPhysXGpu.cpp:98-102 —
    // the concrete PxgParticleBuffer allocates its device buffers through the
    // Vulkan shim (PxgVulkanMemoryManager) and launches no kernels, so it
    // constructs the moment this stub is replaced.  A plain PxPBDParticleSystem +
    // one PxParticleBuffer only ever calls createParticleBuffer (NpParticleBuffer.cpp:610);
    // the diffuse/cloth/rigid variants stay stubbed (P6.1+).
    PxsParticleBuffer* createParticleBuffer(PxU32 maxNumParticles, PxU32 maxVolumes, PxCudaContextManager& cudaContextManager) PX_OVERRIDE
    {
        return static_cast<PxsParticleBuffer*>(PX_NEW(PxgParticleBuffer)(maxNumParticles, maxVolumes, cudaContextManager));
    }
    // P9.13a: the DIFFUSE buffer, third of the four and the same shape as the two
    // above -- PxgParticleAndDiffuseBuffer allocates its device arrays through the
    // Vulkan shim and launches no kernels (mirrors physxgpu/src/PxgPhysXGpu.cpp:104-108).
    //
    // This stub was the gate in front of an ELEVEN-kernel sub-chain: every
    // ps_diffuseParticle* kernel, ps_calculateHashForDiffuseParticlesLaunch,
    // ps_preDiffuseIntegrateLaunch, ps_updateUnsortedDiffuseArrayLaunch and the
    // two diffuse collision launches are unreachable while the engine cannot
    // allocate a buffer to put diffuse particles in -- exactly the situation P9.4
    // found for cloth.  Note the diffuse pass of ps_sdfMeshCollisonLaunch and
    // ps_meshCollisonLaunch (P9.10/P9.12h) has been mapped and swept all along and
    // has never once run in a live scene for the same reason.
    PxsParticleAndDiffuseBuffer* createParticleAndDiffuseBuffer(PxU32 maxNumParticles, PxU32 maxVolumes,
                                                               PxU32 maxNumDiffuseParticles,
                                                               PxCudaContextManager& cudaContextManager) PX_OVERRIDE
    {
        return static_cast<PxsParticleAndDiffuseBuffer*>(PX_NEW(PxgParticleAndDiffuseBuffer)(
            maxNumParticles, maxVolumes, maxNumDiffuseParticles, cudaContextManager));
    }
    // P9.4: the PBD CLOTH buffer, the same shape as createParticleBuffer above and
    // for the same reason -- PxgParticleClothBuffer allocates its device arrays
    // through the Vulkan shim and launches no kernels, so it constructs as soon as
    // the stub is replaced (mirrors physxgpu/src/PxgPhysXGpu.cpp:110-114).
    //
    // This stub, not any missing kernel, was what blocked PXG_SCENE=particlecloth:
    // the whole spring chain had been mapped in P9.3a-c and nothing could reach it,
    // because the engine could not even allocate a cloth buffer to hang springs on.
    // Note also that returning NULL here is not survivable upstream -- NpPhysics
    // hands the result straight to NpParticleClothBuffer's constructor, which
    // dereferences it (NpParticleBuffer.cpp), so the warn-and-null contract of the
    // other stubs SEGFAULTS rather than degrading.  That is worth knowing for the
    // two that remain.
    PxsParticleClothBuffer* createParticleClothBuffer(PxU32 maxNumParticles, PxU32 maxVolumes, PxU32 maxNumCloths, PxU32 maxNumTriangles, PxU32 maxNumSprings, PxCudaContextManager& cudaContextManager) PX_OVERRIDE
    {
        return static_cast<PxsParticleClothBuffer*>(PX_NEW(PxgParticleClothBuffer)(
            maxNumParticles, maxVolumes, maxNumCloths, maxNumTriangles, maxNumSprings, cudaContextManager));
    }
    // P9.13a: the RIGID buffer, the last of the four (physxgpu/src/PxgPhysXGpu.cpp:116-120).
    // Gate in front of the particle-vs-rigid ATTACHMENT sub-chain --
    // ps_rigidAttachmentPrepareLaunch, ps_solveRigidAttachments{,TGS}Launch and
    // ps_solvePCOutputRigidDeltaV{,TGS}Launch.
    //
    // With this, none of PxPhysXGpu's four particle-buffer factories returns NULL
    // any more, and the warn-and-null contract the other stubs use is gone from
    // this class entirely.  That contract was never survivable here: NpPhysics
    // hands the result straight to the Np wrapper's constructor, which
    // dereferences it, so a NULL SEGFAULTS rather than degrading.
    PxsParticleRigidBuffer* createParticleRigidBuffer(PxU32 maxNumParticles, PxU32 maxVolumes,
                                                     PxU32 maxNumRigids,
                                                     PxCudaContextManager& cudaContextManager) PX_OVERRIDE
    {
        return static_cast<PxsParticleRigidBuffer*>(PX_NEW(PxgParticleRigidBuffer)(
            maxNumParticles, maxVolumes, maxNumRigids, cudaContextManager));
    }

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
        // The wrangler keeps a reference to the callback but uses it only while it is
        // constructed; the filter is leaked with the manager it serves.
        static UnmappedKernelNoticeFilter* filter = new UnmappedKernelNoticeFilter(PxGetFoundation().getErrorCallback());
        static PxsKernelWranglerManager* s = [&]() {
            PxsKernelWranglerManager* m = PX_NEW(PxgCudaKernelWranglerManager)(*cudaContextManager, *filter);
            if (filter->suppressed())
                std::fprintf(stderr, "[PhysXGpuVulkan] %u registry kernels have no SPIR-V port (the PxGetPhysicsGpu() "
                                     "utilities and kernels nothing launches); a launch of one is refused and reported.\n",
                             filter->suppressed());
            return m;
        }();
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

physx::PxCudaContextManager* PxCreateCudaContextManager(physx::PxFoundation& foundation,
                                                        const physx::PxCudaContextManagerDesc& /*desc*/,
                                                        physx::PxProfilerCallback* profilerCallback,
                                                        bool /*launchSynchronous*/)
{
    // As the CUDA module does (physxgpu/src/PxgPhysXGpu.cpp): a PhysXGpu built as
    // its own shared library carries its OWN foundation globals, which stay null
    // until the application's foundation is handed over -- and the first PX_NEW /
    // PxGetFoundation() inside the module (the memory managers, the subsystem
    // factories) would then dereference null.  With a static SDK on Linux and on
    // every Windows build there is no symbol interposition to paper over that.
    PxSetFoundationInstance(foundation);
    PxSetProfilerCallback(profilerCallback);

    // desc.deviceOrdinal maps onto the (single) Vulkan device selected at runtime
    // creation.
    physx::gpuslang::VulkanGpuRuntime* r = physx::gpuslang::ensureRuntime();
    if (!r)
        return NULL;
    return physx::gpuslang::PxgCreateVulkanCudaContextManager(r->device, r->shim, r->manifest.data(),
                                                              uint32_t(r->manifest.size()));
}

void PxSetPhysXGpuProfilerCallback(physx::PxProfilerCallback* profilerCallback)
{
    PxSetProfilerCallback(profilerCallback);
}

void PxSetPhysXGpuFoundationInstance(physx::PxFoundation& foundation)
{
    PxSetFoundationInstance(foundation);
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
