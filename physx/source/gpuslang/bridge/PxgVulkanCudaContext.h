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

#ifndef PXG_VULKAN_CUDA_CONTEXT_H
#define PXG_VULKAN_CUDA_CONTEXT_H

// PxgVulkanCudaContext — RUNTIME_PLAN.md P3.1: the concrete physx::PxCudaContext
// backed by the Vulkan runtime.  This is the seam where the real PhysX GPU
// engine meets the Slang/Vulkan port: ScScene, the simulation controller and
// every gpu* module call the abstract PxCudaContext virtual interface, and this
// adapter forwards each call onto PxgCudaShim (memory / launch / stream / event)
// on top of the v2 PxgSlangComputeDevice.
//
// It builds against the REAL PhysX headers (foundation + cudamanager) with
// PX_SUPPORT_GPU_PHYSX = 1 but WITHOUT the CUDA toolkit: PxCudaTypes.h defines
// CUdeviceptr as a 64-bit integer (== our device address) and the other CU*
// handles as opaque pointers we reinterpret as shim handles.  Handle mapping:
//   CUdeviceptr <-> PxgDevicePtr           (identity; both uint64 addresses)
//   CUstream    <-> PxgStreamId            (id in the pointer bits; 0 = default)
//   CUevent     <-> shim event id          (id in the pointer bits)
//   CUfunction  <-> PxgKernelId + 1        (moduleGetFunction resolves name->id)
//   CUmodule    <-> a non-null sentinel    (kernels are precompiled SPIR-V)

#include "PxgCudaShim.h"

#include "foundation/PxPreprocessor.h"

#if PX_SUPPORT_GPU_PHYSX

#include "cudamanager/PxCudaContext.h"
#include "cudamanager/PxCudaContextManager.h"

namespace physx
{
namespace gpuslang
{

// Creates a PxCudaContext that forwards onto `shim`.  `manifest`/`manifestCount`
// let moduleGetFunction resolve an entry-point name to a kernel id (the same
// manifest the shim launches from).  The returned context is released via its
// PxCudaContext::release().  Neither the shim nor the manifest is owned.
physx::PxCudaContext* PxgCreateVulkanCudaContext(PxgCudaShim* shim, const PxgKernelDef* manifest,
                                                 uint32_t manifestCount);

// Creates the PxCudaContextManager the scene receives via
// PxSceneDesc::cudaContextManager.  It owns an internal PxCudaContext (the one
// above), reports device properties from `device`, and is a no-op for the
// per-thread CUDA-context acquire/release (a single Vulkan device needs none).
// Released via PxCudaContextManager::release(); does not own device/shim/manifest.
physx::PxCudaContextManager* PxgCreateVulkanCudaContextManager(PxgSlangComputeDevice* device,
                                                               PxgCudaShim* shim,
                                                               const PxgKernelDef* manifest,
                                                               uint32_t manifestCount);

// Vulkan replacement for the engine's KernelWrangler: turns a kernel id
// (PxgKernelIds::X, the manifest index) into the CUfunction the PxCudaContext
// launches.  The returned handle is the SAME encoding this bridge's
// moduleGetFunction produces, so a launch driven by
//   ctx->launchKernel(wrangler.getFunction(id), ...)
// resolves back to manifest[id] — the exact id -> launch path the engine's
// getCuFunction(id) -> launchKernel path uses.  Unmapped (spirvFile == null)
// or out-of-range ids return null.  Does not own the manifest.
// P9.40 -- how often the kernel with this manifest index was LAUNCHED WHILE
// UNMAPPED on `ctx`.  Such a launch used to be an anonymous InvalidValue through a
// null wrangler slot; it is now a named, counted refusal (see the sentinel notes in
// PxgVulkanCudaContext.cpp), reported through the foundation error callback as
// "[pxg-unmapped] <cudaName> ..." when the context is released.  Returns 0 for a
// context that is not this bridge's.
uint32_t pxgUnmappedLaunchCount(const physx::PxCudaContext* ctx, uint32_t kernelId);

class PxgVulkanKernelWrangler
{
public:
    PxgVulkanKernelWrangler(const PxgKernelDef* manifest, uint32_t count)
        : mManifest(manifest), mCount(count) {}

    CUfunction getFunction(uint32_t kernelId) const;          // null if unmapped
    const char* getFunctionName(uint32_t kernelId) const;     // Slang entry, or null
    bool isMapped(uint32_t kernelId) const;
    uint32_t count() const { return mCount; }
    uint32_t mappedCount() const;

private:
    const PxgKernelDef* mManifest;
    uint32_t mCount;
};

} // namespace gpuslang
} // namespace physx

#endif // PX_SUPPORT_GPU_PHYSX
#endif // PXG_VULKAN_CUDA_CONTEXT_H
