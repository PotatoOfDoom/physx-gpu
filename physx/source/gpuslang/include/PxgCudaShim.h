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

#ifndef PXG_CUDA_SHIM_H
#define PXG_CUDA_SHIM_H

// PxgCudaShim — RUNTIME_PLAN.md P3 core: the CUDA-context replacement's
// load-bearing middle, isolated from the PhysX build so it is testable in
// the gpuslang harness.  It reproduces the essentials of the PxCudaContext
// surface that ScScene actually exercises — device allocation, host<->device
// copies keyed on raw device addresses, and kernel launch — on top of the v2
// PxgSlangComputeDevice.  The full virtual PxCudaContext / PxCudaContextManager
// wrappers (P3.1/P3.4/P3.5) forward one-to-one onto this object once the PhysX
// headers are in the build; nothing here depends on them.
//
// Two design decisions transcribed from PxCudaContext:
//   - CUdeviceptr is a uint64_t device address.  memAlloc returns one;
//     memcpy* take them; kernel parameter structs embed them (often nested
//     inside PxgSoftBody-style structs).  The shim keeps an allocation
//     interval map so an arbitrary address resolves to (buffer, offset) —
//     exactly what CUDA's flat address space gives for free.
//   - launchKernel receives an opaque parameter blob the caller has already
//     populated with device addresses.  The shim binds ZERO descriptors and
//     hands the blob through as push constants; the Slang kernel dereferences
//     the embedded addresses as buffer-device-address pointers (proven by the
//     P0 BDA spike).  This preserves the CUDA pointer-passing model 1:1.

#include "PxgSlangComputeDevice.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <unordered_map>
#include <vector>

namespace physx
{
namespace gpuslang
{

typedef uint64_t PxgDevicePtr;    // CUdeviceptr parity; 0 = null
typedef uint32_t PxgKernelId;     // index into the manifest
typedef uint32_t PxgStreamId;     // CUstream parity; 0 = the null/default stream

// One manifest row: which compiled entry point a kernel id maps to, plus the
// launch metadata CUDA carries out-of-band (block/shared) that the Vulkan
// pipeline needs at creation time.  numTextures > 0 selects the sampled-image
// pipeline path (createPipelineTex).
struct PxgKernelDef
{
    const char* spirvFile;      // e.g. "npSpikeBda.spv" (loaded via loader cb)
    const char* entry;          // e.g. "spikeBda"
    uint32_t pushConstantSize;  // parameter-blob size in bytes
    uint32_t numTextures;       // sampled 3D images the kernel declares
    // CUDA __global__ name (registry key), for per-kernel launch-ABI lookups
    // (the void** launch form carries no argument sizes — P5 Step 4).  Optional:
    // harness manifests leave it null.
    const char* cudaName = nullptr;
};

enum class PxgResult
{
    Success = 0,
    InvalidValue,
    NotFound,
    OutOfMemory
};

// Loads a compiled SPIR-V blob by manifest file name.  In the engine this is
// the module manifest lookup; in the harness it is the test's shader loader.
typedef std::vector<uint8_t> (*PxgSpirvLoader)(const char* file);

class PxgCudaShim
{
public:
    PxgCudaShim(PxgSlangComputeDevice* device, const PxgKernelDef* manifest, uint32_t manifestCount,
                PxgSpirvLoader loader);
    ~PxgCudaShim();

    // ---- memory (PxCudaContext memAlloc/memFree/memcpy* parity) ----
    PxgResult memAlloc(PxgDevicePtr* out, size_t bytes);
    PxgResult memFree(PxgDevicePtr ptr);
    PxgResult memcpyHtoD(PxgDevicePtr dst, const void* src, size_t bytes);
    PxgResult memcpyDtoH(void* dst, PxgDevicePtr src, size_t bytes);
    PxgResult memcpyDtoD(PxgDevicePtr dst, PxgDevicePtr src, size_t bytes);
    PxgResult memsetD32(PxgDevicePtr dst, uint32_t value, size_t numWords);

    // ---- pinned host memory (cuMemHostAlloc / cuMemHostGetDevicePointer
    // parity, RUNTIME_PLAN.md P5 Step 4b) ----
    // memHostAlloc returns a CPU pointer into a persistently mapped host-
    // visible buffer that ALSO has a device address, registered in the flat
    // address space (so memcpys/resolve() see it and kernels dereference it
    // via BDA).  hostGetDevicePointer maps any interior CPU pointer of such an
    // allocation to the matching device address (0 when unknown — the CUDA
    // null result for unregistered host memory).
    PxgResult memHostAlloc(void** out, size_t bytes);
    PxgResult memFreeHost(void* p);
    PxgDevicePtr hostGetDevicePointer(const void* p) const;

    // ---- launch (PxCudaContext launchKernel parity) ----
    // gridDim in workgroups; block dims are informational parity (the Slang
    // numthreads is fixed at compile time).  paramBlob is the opaque
    // device-address-bearing parameter pack; textures are the sampled 3D
    // images bound in declaration order (may be null when numTextures == 0).
    PxgResult launchKernel(PxgKernelId id, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                           const void* paramBlob, uint32_t paramBlobSize,
                           const PxgTextureHandle* textures = nullptr, uint32_t numTextures = 0);

    // ---- streams & events (PxCudaContext stream*/event* parity) ----
    // P3 runs every stream on ONE VkQueue in submission order (the P1
    // correctness-first serialization), so a stream is a logical ordering
    // handle and an event is a timeline ticket.  Cross-stream ordering is
    // automatically satisfied by that serialization; true multi-queue async
    // compute is deferred to P6.  streamSynchronize() (no arg) syncs the
    // default stream.
    PxgResult streamCreate(PxgStreamId* out);
    PxgResult streamDestroy(PxgStreamId stream);
    PxgResult streamSynchronize(PxgStreamId stream = 0);

    // Event as a timeline ticket captured at record time.
    PxgResult eventCreate(uint32_t* out);
    PxgResult eventDestroy(uint32_t event);
    PxgResult eventRecord(uint32_t event, PxgStreamId stream = 0);
    PxgResult eventSynchronize(uint32_t event);
    PxgResult eventQuery(uint32_t event);              // Success = complete, InvalidValue = pending
    PxgResult streamWaitEvent(PxgStreamId stream, uint32_t event);

    // Address resolution helper (exposed for the geometry/SDF manager).
    bool resolve(PxgDevicePtr ptr, PxgBufferHandle* buffer, size_t* offset) const;

private:
    PxgPipelineHandle pipelineFor(PxgKernelId id);

    struct Alloc
    {
        PxgBufferHandle buffer;
        PxgDevicePtr base;
        size_t size;
    };

    PxgSlangComputeDevice* mDevice;
    const PxgKernelDef* mManifest;
    uint32_t mManifestCount;
    PxgSpirvLoader mLoader;

    // base address -> allocation, ordered so an interior address resolves via
    // upper_bound (the CUDA flat-address-space lookup).
    std::map<PxgDevicePtr, Alloc> mAllocs;
    // Pinned allocations, additionally indexed by CPU base address so a host
    // pointer (interior included) resolves to its device address.  The Alloc
    // itself lives in mAllocs (which owns buffer destruction); this is an index.
    struct HostAlloc
    {
        PxgDevicePtr devBase;
        size_t size;
    };
    std::map<uintptr_t, HostAlloc> mHostAllocs;
    std::unordered_map<PxgKernelId, PxgPipelineHandle> mPipelines;

    // Stream/event bookkeeping.  Streams are logical handles (all serialize
    // on the one queue); an event stores the submission ticket at record time.
    std::unordered_map<PxgStreamId, bool> mStreams;   // id -> alive
    std::unordered_map<uint32_t, uint64_t> mEvents;   // id -> recorded ticket
    PxgStreamId mNextStream = 1;
    uint32_t mNextEvent = 1;
    uint64_t mLastLaunchTicket = 0;                   // ticket of the last submitted work
};

} // namespace gpuslang
} // namespace physx

#endif // PXG_CUDA_SHIM_H
