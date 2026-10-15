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
#include <atomic>
#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <mutex>
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
    OutOfMemory,
    // The device cannot back this operation at all -- distinct from
    // InvalidValue, which means the ARGUMENTS were wrong.  Added in P9.12b for
    // the 3D-texture path: a device without descriptor indexing must report
    // "not available here" rather than "you asked wrongly", because the caller's
    // remedy is different (skip the feature, not fix the call).
    Unsupported
};

// Loads a compiled SPIR-V blob by manifest file name.  In the engine this is
// the module manifest lookup; in the harness it is the test's shader loader.
typedef std::vector<uint8_t> (*PxgSpirvLoader)(const char* file);

// The GPU API lock.  PhysX calls the CUDA layer from several worker threads at once
// (the broadphase and narrowphase tasks of one step launch kernels concurrently), which
// the CUDA driver allows: it is thread-safe.  Vulkan is not -- a command buffer, its
// pool and the queue must be externally synchronized -- and neither is the shim's own
// bookkeeping.  Every PxgCudaShim entry point and every PxgVulkanCudaContext call
// holds this process-wide lock, so concurrent callers are serialized exactly where
// the CUDA driver serializes them internally.  Recursive: the context calls into the
// shim, and the shim into itself.  (Found on an AMD RX 9070 XT: two threads recording
// into one command buffer -> vkBeginCommandBuffer VK_ERROR_UNKNOWN / DEVICE_LOST;
// llvmpipe tolerated it.)
std::recursive_mutex& pxgGpuApiMutex();

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
    // cuMemcpyHtoDAsync / cuMemcpyDtoHAsync.  When the host side lies in a pinned
    // allocation the copy is RECORDED into the stream as a device copy -- read or
    // written when it executes, exactly like CUDA's DMA from/to pinned memory -- and
    // nothing is drained; the engine synchronises before it reads the destination.
    // Pageable host memory takes the synchronous paths above (CUDA stages those too).
    PxgResult memcpyHtoDAsync(PxgDevicePtr dst, const void* src, size_t bytes);
    PxgResult memcpyDtoHAsync(void* dst, PxgDevicePtr src, size_t bytes);
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

    // ---- 3D textures: cuArray3DCreate / cuMemcpy3D / cuTexObjectCreate parity
    // (P9.12b, the SDF upload path) ----
    //
    // CUDA splits one upload across THREE calls -- allocate the array, copy into
    // it, then build a texture object over it -- while the backend creates the
    // image, uploads it and makes its view in one.  So a CUDA array is a HOST
    // staging record here: dimensions, format, and the bytes copied into it.
    // The Vulkan image is not created until textureObjectCreate, which is the
    // first moment all three are known.
    //
    // The engine's own sequence (PxgGeometryManager.cpp:733-769) does exactly
    // this and in exactly this order, so nothing is being anticipated.  A copy
    // that arrives AFTER the texture object was built would be silently lost,
    // and is refused rather than ignored.
    PxgResult arrayCreate3D(uint32_t* outArray, uint32_t width, uint32_t height, uint32_t depth,
                            PxgSlangComputeDevice::TextureFormat format);
    PxgResult arrayUpload(uint32_t arrayId, const void* src, size_t bytes);
    PxgResult arrayDestroy(uint32_t arrayId);
    PxgResult textureObjectCreate(PxgTextureHandle* out, uint32_t arrayId);
    PxgResult textureObjectDestroy(PxgTextureHandle handle);

    PxgSlangComputeDevice* device() const { return mDevice; }

    // ---- pipeline warm-up ----
    // A kernel's pipeline is otherwise built at its FIRST launch, under the API lock:
    // the driver compiles the SPIR-V then (up to hundreds of ms for the large
    // narrowphase/solver kernels on real hardware), and every thread that wants the
    // GPU waits.  The warm-up builds the pipelines of the rows in `order` (in that
    // order; rows that are not launchable -- no SPIR-V, entry or push size -- are
    // skipped) on `threadCount` background threads WITHOUT the API lock.  A launch
    // whose pipeline a warm-up thread is building waits for exactly that build; one
    // not reached yet is built by the launch itself.  Every pipeline is built once.
    // Only when the device supportsConcurrentPipelineCreation(); otherwise a no-op that
    // returns false.  `onComplete` runs on the last warm-up thread once all rows are
    // built (not when stopped early).
    bool startPipelineWarmup(const std::vector<PxgKernelId>& order, unsigned threadCount,
                             std::function<void()> onComplete = std::function<void()>());
    // The rows launched so far, in order of their first launch -- what a later run
    // should warm up, and in which order.
    std::vector<PxgKernelId> launchedKernels() const;
    // Lets the threads finish the pipeline each is building, then joins them.  Called
    // by the destructor; idempotent.
    void stopPipelineWarmup();
    // Pipelines built so far (warm-up and launches); for diagnostics and tests.
    uint32_t builtPipelineCount() const;

private:
    PxgPipelineHandle pipelineFor(PxgKernelId id);
    // Builds the pipeline of row `id`.  Takes no API lock: runs on warm-up threads.
    PxgPipelineHandle buildPipeline(PxgKernelId id, bool warmup);
    // The SPIR-V of `file`, loaded once and shared by every row that uses it.
    std::shared_ptr<const std::vector<uint8_t> > spirvBlob(const char* file);
    void warmupWorker();
    // P9.98 check round 1: resolve() plus "the `bytes` from ptr on lie inside the
    // allocation" -- the transfer entry points' check (InvalidValue otherwise).
    bool resolveRange(PxgDevicePtr ptr, size_t bytes, PxgBufferHandle* buffer, size_t* offset,
                      const char* caller) const;
    // The pinned allocation holding [host, host + bytes): its buffer and offset.  false
    // (silently) for pageable memory or a range that leaves the allocation.
    bool resolvePinnedRange(const void* host, size_t bytes, PxgBufferHandle* buffer, size_t* offset) const;

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

    // One slot per manifest row: not built, being built (by a launch or a warm-up
    // thread), or built (handle; 0 = refused, and the launch is refused).
    enum : uint8_t { kPipeNone = 0, kPipeBuilding = 1, kPipeReady = 2 };
    struct PipelineSlot
    {
        uint8_t state = kPipeNone;
        bool launched = false;
        PxgPipelineHandle handle = 0;
    };
    std::vector<PxgKernelId> mLaunchOrder;   // first launches, guarded by mSlotMutex
    std::vector<PxgKernelId> mWarmOrder;
    std::vector<PipelineSlot> mSlots;
    mutable std::mutex mSlotMutex;
    std::condition_variable mSlotCv;
    uint32_t mBuiltCount = 0;
    std::mutex mBlobMutex;
    std::unordered_map<std::string, std::shared_ptr<const std::vector<uint8_t> > > mBlobs;
    std::vector<std::thread> mWarmThreads;
    std::atomic<bool> mWarmStop;
    std::atomic<uint32_t> mWarmNext;
    std::atomic<uint32_t> mWarmActive;
    std::function<void()> mWarmDone;

    // CUDA arrays: host staging records, keyed by the id handed back as CUarray.
    struct CudaArray
    {
        uint32_t width = 0, height = 0, depth = 0;
        PxgSlangComputeDevice::TextureFormat format = PxgSlangComputeDevice::TextureFormat::R32Float;
        std::vector<uint8_t> host;
        bool uploaded = false;   // a copy has arrived
        bool consumed = false;   // a texture object has been built over it
    };
    std::unordered_map<uint32_t, CudaArray> mArrays;
    uint32_t mNextArrayId = 1;

    // Stream/event bookkeeping.  Streams are logical handles (all serialize
    // on the one queue); an event stores the submission ticket at record time.
    std::unordered_map<PxgStreamId, bool> mStreams;   // id -> alive
    std::unordered_map<uint32_t, uint64_t> mEvents;   // id -> recorded ticket
    PxgStreamId mNextStream = 1;
    uint32_t mNextEvent = 1;
    uint64_t mLastLaunchTicket = 0;                   // ticket of the last submitted work
};

// P9.84: true when `entry` in the SPIR-V `blob` uses a variable decorated
// DescriptorSet `set` (per entry point from SPIR-V 1.4 on; below that, whenever
// the module declares the set at all).  pipelineFor uses it to decide which
// engine kernels need the bindless SDF set; exposed so a fixture can check the
// classification on real blobs.
bool pxgSpirvEntryUsesDescriptorSet(const std::vector<uint8_t>& blob, const char* entry, uint32_t set);

} // namespace gpuslang
} // namespace physx

#endif // PXG_CUDA_SHIM_H
