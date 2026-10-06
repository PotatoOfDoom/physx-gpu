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

#ifndef PXG_SLANG_COMPUTE_DEVICE_H
#define PXG_SLANG_COMPUTE_DEVICE_H

// PxgSlangComputeDevice
//
// Minimal compute-device abstraction backing the Slang port of the PhysX GPU
// kernels.  It replaces the CUDA driver-API layer (PxgCudaKernelWranglerManager
// / PxCudaContextManager) with an API that maps directly onto Vulkan and
// Direct3D 12 compute queues.
//
// Model:
//   - Storage buffers are opaque handles; all kernel buffer parameters map to
//     a single descriptor set (Vulkan) / one UAV descriptor table (D3D12)
//     with binding i == the i-th buffer passed to dispatch().
//   - Small scalar kernel parameters travel as push constants (Vulkan) /
//     root constants (D3D12), mirroring the CUDA kernel scalar arguments.
//   - Kernels are precompiled offline by slangc: SPIR-V for the Vulkan
//     backend, DXIL (via HLSL) for the D3D12 backend.
//   - dispatch() records work; barrier() orders successive dispatches
//     (equivalent to work on a single CUDA stream); flush() submits and waits.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace physx
{
namespace gpuslang
{

typedef uint32_t PxgBufferHandle;   // 0 = invalid
typedef uint32_t PxgPipelineHandle; // 0 = invalid
typedef uint32_t PxgTextureHandle;  // 0 = invalid

struct PxgSlangDeviceDesc
{
    bool enableValidation = false;
    // When true, prefer a software/CPU device (llvmpipe, WARP) if present —
    // used by the verification test suite.
    bool preferSoftwareDevice = false;
    // When true, storage buffers live in device-local memory and all
    // host<->device transfers go through persistently mapped staging rings
    // (the production path for discrete GPUs).  When false, buffers are
    // host-visible and transfers are direct maps (fast on UMA/software
    // devices).  Kernel results are identical either way.
    bool useDeviceLocalMemory = false;
    // Optional path to a VkPipelineCache blob on disk.  When set, the cache
    // is loaded at device creation and written back at destruction, so
    // pipeline (shader) compilation is reused across runs.  NULL disables
    // the on-disk cache (an in-memory cache is still used within a run).
    const char* pipelineCachePath = nullptr;
};

class PxgSlangComputeDevice
{
public:
    virtual ~PxgSlangComputeDevice() {}

    virtual const char* getDeviceName() const = 0;
    virtual const char* getBackendName() const = 0;

    // ---- resources ----
    virtual PxgBufferHandle createBuffer(size_t sizeBytes) = 0;
    virtual void destroyBuffer(PxgBufferHandle buffer) = 0;

    // Pinned-host buffer (CUDA cuMemHostAlloc parity, RUNTIME_PLAN.md P5 Step
    // 4b): host-visible memory persistently mapped for the buffer's lifetime —
    // the CPU reads/writes it through `*mappedPtr` while shaders access it by
    // buffer device address.  Freed with destroyBuffer.  Returns 0 (and leaves
    // *mappedPtr untouched) on backends without pinned-mapped support.
    virtual PxgBufferHandle createHostVisibleBuffer(size_t /*sizeBytes*/, void** /*mappedPtr*/) { return 0; }

    virtual void upload(PxgBufferHandle buffer, const void* data, size_t sizeBytes, size_t dstOffset = 0) = 0;
    virtual void download(PxgBufferHandle buffer, void* data, size_t sizeBytes, size_t srcOffset = 0) = 0;

    // Device-to-device copy (CUDA cuMemcpyDtoD parity) and 32-bit fill
    // (cuMemsetD32 parity).  Recorded into the current stream in submission
    // order.  Offsets and sizes are byte counts; fillBuffer's must be
    // 4-byte aligned.  Base defaults stage through the host so backends
    // without a device path still work.
    virtual void copyBuffer(PxgBufferHandle dst, size_t dstOffset, PxgBufferHandle src, size_t srcOffset,
                            size_t sizeBytes)
    {
        std::vector<uint8_t> tmp(sizeBytes);
        download(src, tmp.data(), sizeBytes, srcOffset);
        upload(dst, tmp.data(), sizeBytes, dstOffset);
    }
    virtual void fillBuffer(PxgBufferHandle dst, size_t dstOffset, uint32_t value, size_t sizeBytes)
    {
        std::vector<uint32_t> fill(sizeBytes / 4, value);
        upload(dst, fill.data(), fill.size() * 4, dstOffset);
    }

    // ---- pipelines ----
    // `code` is backend-specific precompiled shader code (SPIR-V for Vulkan,
    // DXIL for D3D12) containing a single compute entry point `entryName`.
    // `bufferStrides[i]` is the structure byte stride of the i-th
    // RWStructuredBuffer declared by the kernel; D3D12 needs it to create
    // matching structured UAVs (Vulkan derives everything from SPIR-V and
    // ignores it).  May be NULL only when numBuffers == 0.
    virtual PxgPipelineHandle createPipeline(const void* code, size_t codeSize, const char* entryName,
                                             uint32_t numBuffers, const uint32_t* bufferStrides,
                                             uint32_t pushConstantSize) = 0;
    virtual void destroyPipeline(PxgPipelineHandle pipeline) = 0;

    // ---- execution ----
    // Records a dispatch of gridX x gridY x gridZ workgroups.  `buffers`
    // bind to the shader's storage buffers in declaration order.
    virtual void dispatch(PxgPipelineHandle pipeline, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                          const PxgBufferHandle* buffers, uint32_t numBuffers,
                          const void* pushConstants, uint32_t pushConstantSize) = 0;

    // Indirect dispatch (cuLaunchKernel with a device-computed grid): the
    // workgroup counts are read from `argBuffer` at `argOffset` as three
    // consecutive uint32 (x, y, z) — the VkDispatchIndirectCommand /
    // D3D12 dispatch-args layout.  Lets a kernel compute the next grid on
    // device without a count readback (RUNTIME_PLAN.md P6).  Base default
    // is unsupported (returns without recording); backends override.
    virtual void dispatchIndirect(PxgPipelineHandle /*pipeline*/, PxgBufferHandle /*argBuffer*/,
                                  size_t /*argOffset*/, const PxgBufferHandle* /*buffers*/,
                                  uint32_t /*numBuffers*/, const void* /*pushConstants*/,
                                  uint32_t /*pushConstantSize*/) {}
    virtual bool supportsIndirectDispatch() const { return false; }

    // ---- GPU timestamps (per-stage timing, RUNTIME_PLAN.md P1.6) ----
    // writeTimestamp() records a GPU timestamp at the current point in the
    // recorded stream and returns its query index (UINT32_MAX if
    // unsupported).  After the work is submitted and retired (flush / wait),
    // getTimestampNanos(a, b) returns the elapsed nanoseconds between two
    // recorded timestamps.  Query indices are valid only until the next
    // recording batch begins (the pool is reset per batch).
    virtual bool supportsTimestamps() const { return false; }
    virtual uint32_t writeTimestamp() { return 0xffffffffu; }
    virtual double getTimestampNanos(uint32_t /*begin*/, uint32_t /*end*/) { return -1.0; }

    // Compute->compute execution + memory barrier between recorded dispatches
    // (the ordering CUDA guarantees implicitly on a single stream).
    virtual void barrier() = 0;

    // P9.85: how many dispatches were recorded while the PREVIOUS dispatch of the
    // same batch had no barrier() after it -- i.e. pairs the device is free to
    // overlap.  A submit boundary counts as ordered (each batch waits on the one
    // before it).  Per-buffer transfer barriers do not count: they order one
    // VkBuffer, and a BDA kernel reaches others.  Diagnostic only; with
    // PXG_SYNC_STATS=1 the Vulkan device also prints it on destruction.
    virtual uint64_t unorderedDispatchCount() const { return 0; }

    // Submits all recorded work and waits for completion.  Equivalent to
    // wait(submit()).
    virtual void flush() = 0;

    // ---- v2 submission (RUNTIME_PLAN.md P1.1) ----
    // submit() hands all recorded work to the device WITHOUT waiting and
    // returns a ticket; tickets are monotonic per device and 0 means
    // "already complete".  wait() blocks until the ticketed submission has
    // retired; poll() is the non-blocking query.  Calling submit() with
    // nothing recorded returns the most recent ticket.  Backends without a
    // non-blocking path may submit synchronously and return 0.
    //
    // Ordering contract: consecutive submissions on one device execute and
    // become visible in submission order (single-CUDA-stream semantics), so
    // a dispatch recorded after submit() may safely consume the results of
    // the submitted batch without a host-side wait().
    virtual uint64_t submit()
    {
        flush();
        return 0;
    }
    virtual void wait(uint64_t /*ticket*/) {}
    virtual bool poll(uint64_t /*ticket*/) { return true; }

    // ---- Phase-2 runtime capabilities (RUNTIME_PLAN.md) ----
    // Additive with conservative defaults so backends opt in individually.

    struct SubgroupCaps
    {
        uint32_t defaultSize = 0;   // 0 = subgroup queries unsupported
        uint32_t minSize = 0;
        uint32_t maxSize = 0;
        bool canRequireSize = false; // subgroupSizeControl usable on compute
        bool fullSubgroups = false;  // computeFullSubgroups
    };
    virtual SubgroupCaps getSubgroupCaps() const { return SubgroupCaps(); }

    // Like createPipeline, but requests an exact subgroup size (with full
    // subgroups) for the entry point.  requiredSubgroupSize == 0 behaves
    // exactly like createPipeline.  Returns 0 when the size cannot be
    // honored on this device.
    virtual PxgPipelineHandle createPipelineEx(const void* code, size_t codeSize, const char* entryName,
                                               uint32_t numBuffers, const uint32_t* bufferStrides,
                                               uint32_t pushConstantSize, uint32_t requiredSubgroupSize)
    {
        if (requiredSubgroupSize != 0)
            return 0;
        return createPipeline(code, codeSize, entryName, numBuffers, bufferStrides, pushConstantSize);
    }

    // Buffer device address (CUDA device-pointer parity, used by the
    // buffer-device-address argument-marshalling model).  0 when the
    // backend/device has no address support.
    virtual uint64_t getBufferDeviceAddress(PxgBufferHandle) const { return 0; }

    // ---- 3D sampled textures (RUNTIME_PLAN.md P2: the tex3D replacement) ----
    // Single-channel float volumes with trilinear filtering and
    // clamp-to-edge addressing — the configuration the CUDA SDF path uses
    // (cudaFilterModeLinear / cudaAddressModeClamp / element read mode).
    // Coordinates are normalized in the shader; sampling at voxel centers
    // ((p + 0.5) / dim with power-of-two dims) is exact, interior points
    // carry the implementation's filtering precision (documented tolerance,
    // NOT bit-exact across vendors).
    virtual PxgTextureHandle createTexture3D(uint32_t /*width*/, uint32_t /*height*/, uint32_t /*depth*/,
                                             const float* /*texels*/) { return 0; }
    virtual void destroyTexture3D(PxgTextureHandle) {}

    // ---- bindless 3D textures (P9.12a: the CUtexObject replacement) ----
    //
    // The engine stores a CUtexObject INSIDE a device-side struct
    // (PxgTriangleMesh::mTexObject) and dereferences it in the kernel:
    // `tex3D<float>(mesh.mTexObject, ...)`.  SPIR-V has no such thing as a
    // texture handle in a buffer, so the equivalent is descriptor indexing --
    // one unbounded array of sampled 3D images, with the handle as the INDEX.
    // A texture handle is therefore its slot in that array, and handle 0 is
    // reserved as the null texture exactly as CUDA leaves mTexObject at 0 for
    // a mesh with no SDF (PxgGeometryManager.cpp:725-731).
    //
    // Formats are the three the engine's SDF upload actually asks for
    // (PxgGeometryManager.cpp:642/663/669/675): the dense grid is FLOAT, and
    // the sparse subgrid texture is UNSIGNED_INT8, UNSIGNED_INT16 or FLOAT.
    // CUDA promotes the integer formats to normalized [0,1] floats because
    // cuTexObjectCreate is called WITHOUT CU_TRSF_READ_AS_INTEGER, which is
    // what SparseSDFTexture::applySubgridSdfScale then rescales -- so the
    // Vulkan equivalents are the UNORM formats, not the UINT ones.
    enum class TextureFormat
    {
        R32Float = 0,
        R8Unorm = 1,
        R16Unorm = 2,
    };

    // True when the device can back the bindless path (descriptor indexing
    // with a runtime array of sampled images).  When false every call below
    // is a no-op returning 0, and the SDF kernels stay unmapped.
    virtual bool supportsBindlessTextures() const { return false; }

    // Uploads a single-channel volume and returns its bindless slot (never 0
    // on success).  `texels` is width*height*depth elements of the format's
    // element type (float / uint8_t / uint16_t).
    virtual PxgTextureHandle createTexture3DFmt(uint32_t /*width*/, uint32_t /*height*/, uint32_t /*depth*/,
                                                TextureFormat /*format*/, const void* /*texels*/)
    { return 0; }

    // Like createPipeline, but the kernel additionally declares the bindless
    // set at descriptor set 1, which has ONE binding since P9.97:
    // `Texture3D<float> gPxgTex3D[]` at binding 0 (shaders/pxgSlangTex3d.slang;
    // pxgTex3D fetches texels, so there is no sampler).  A module that declares
    // anything else in set 1, or any set above 1, or a sampler type, is REFUSED
    // (returns 0, reason on stderr; pxgSpirvFitsBindlessLayout / pxgSpirvDeclaresSampler)
    // -- llvmpipe would not refuse it.  Kernels
    // that do not sample must keep using createPipeline: a pipeline layout that
    // declares a set it never reads costs a bind per dispatch for nothing.
    virtual PxgPipelineHandle createPipelineBindless(const void* /*code*/, size_t /*codeSize*/,
                                                     const char* /*entryName*/, uint32_t /*numBuffers*/,
                                                     const uint32_t* /*bufferStrides*/,
                                                     uint32_t /*pushConstantSize*/)
    { return 0; }

    // Like createPipeline/dispatch, but the kernel additionally declares
    // numTextures Texture3D+SamplerState pairs AFTER its storage buffers
    // (descriptor bindings numBuffers+2t and numBuffers+2t+1).
    virtual PxgPipelineHandle createPipelineTex(const void* /*code*/, size_t /*codeSize*/,
                                                const char* /*entryName*/, uint32_t /*numBuffers*/,
                                                const uint32_t* /*bufferStrides*/,
                                                uint32_t /*pushConstantSize*/, uint32_t /*numTextures*/)
    { return 0; }
    virtual void dispatchTex(PxgPipelineHandle /*pipeline*/, uint32_t /*gridX*/, uint32_t /*gridY*/,
                             uint32_t /*gridZ*/, const PxgBufferHandle* /*buffers*/, uint32_t /*numBuffers*/,
                             const PxgTextureHandle* /*textures*/, uint32_t /*numTextures*/,
                             const void* /*pushConstants*/, uint32_t /*pushConstantSize*/) {}

    // ---- P9.97 diagnostics -- APPENDED, so every existing vtable slot keeps its index ----

    // P9.97: the bindless set AS HANDED TO THE DRIVER -- diagnostic only, for the
    // fixture that pins it (shimBindless_setLayout).  The Vulkan device records it
    // at its vkCreateDescriptorSetLayout / vkCreateDescriptorPool call sites from
    // the very create-infos it passes (the binding flags from the pNext chain), so a
    // binding added to those arrays shows up here.  Raw Vulkan enum values; all
    // zero when the device has no bindless set.
    struct BindlessSetInfo
    {
        static const uint32_t kMax = 4;
        uint32_t layoutFlags = 0;
        uint32_t bindingCount = 0;       // entries beyond kMax are counted, not copied
        uint32_t bindingFlagsCount = 0;  // VkDescriptorSetLayoutBindingFlagsCreateInfo::bindingCount
        uint32_t binding[kMax] = {}, descriptorType[kMax] = {}, descriptorCount[kMax] = {},
                 stageFlags[kMax] = {}, bindingFlags[kMax] = {};
        uint32_t poolFlags = 0, poolMaxSets = 0, poolSizeCount = 0;
        uint32_t poolType[kMax] = {}, poolCount[kMax] = {};
    };
    virtual BindlessSetInfo bindlessSetInfo() const { return BindlessSetInfo(); }

    // P9.97: VkSamplers this device owns right now (created minus destroyed through
    // its own two helpers).  Diagnostic only.  Since P9.97 the only owner is a
    // createTexture3D texture (set 0, the P2 probe); a bindless texture owns none.
    virtual uint32_t liveSamplerCount() const { return 0; }
};

// P9.97: does the module's use of descriptor sets >= 1 fit the bindless set -- set 1 with
// binding 0 only, nothing in set 2 or above?  MODULE-WIDE: every variable decorated
// DescriptorSet counts, used by the entry or not.  NOT checked: set 0 (the caller's
// numBuffers storage buffers) and descriptor TYPES -- pxgSpirvDeclaresSampler covers the
// sampler case.  OpDecorate only (slangc emits no decoration groups / OpDecorateId;
// P9.97 scan of all 471 .spv).  On false, *badSet / *badBinding (either may be null)
// name an offending variable (binding ~0u = no Binding decoration); a blob that is not
// a well-formed SPIR-V word stream does not fit either (both ~0u).  Defined by the
// Vulkan backend.
bool pxgSpirvFitsBindlessLayout(const void* code, size_t codeSize, uint32_t* badSet, uint32_t* badBinding);

// P9.97: true when the module declares a sampler type (OpTypeSampler / OpTypeSampledImage),
// i.e. could sample in hardware.  The bindless path never does since P9.96, and
// createPipelineBindless refuses such a module.  False for a malformed blob (the layout
// check refuses that one).
bool pxgSpirvDeclaresSampler(const void* code, size_t codeSize);

// Factory functions; each returns NULL when the backend is unavailable.
PxgSlangComputeDevice* PxgCreateVulkanComputeDevice(const PxgSlangDeviceDesc& desc);
#if defined(_WIN32)
PxgSlangComputeDevice* PxgCreateDx12ComputeDevice(const PxgSlangDeviceDesc& desc);
#endif

} // namespace gpuslang
} // namespace physx

#endif // PXG_SLANG_COMPUTE_DEVICE_H
