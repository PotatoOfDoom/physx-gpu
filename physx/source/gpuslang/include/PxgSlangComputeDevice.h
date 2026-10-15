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
#include <cstdio>
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
    // Alternatively the cache's initial bytes from memory (the engine bridge reads and
    // writes the file itself, with UTF-8 paths on Windows; see getPipelineCacheData).
    // Used only when pipelineCachePath is NULL.  A blob whose header names another
    // vendor, device or pipelineCacheUUID is ignored -- the driver checks that too,
    // this keeps a buggy driver from being handed a foreign cache.
    const void* pipelineCacheData = nullptr;
    size_t pipelineCacheDataSize = 0;
    // P9.98 test hooks: when nonzero, LOWER the corresponding device limit to this
    // value (never raise it) before any pipeline is checked against it, so a fixture
    // can show that a kernel over a smaller device's limit is REFUSED rather than
    // built.  The environment variables PXG_VK_LIMIT_PUSH_CONSTANTS,
    // PXG_VK_LIMIT_SHARED_MEMORY, PXG_VK_LIMIT_WORKGROUP_INVOCATIONS and
    // PXG_VK_LIMIT_WORKGROUP_SIZE do the same for a whole process (a scene), and the
    // smallest of the three sources (device, desc, environment) wins.
    // limitMaxComputeWorkGroupSize lowers each of the three dimensions.
    // P9.98 check round 1: limitMaxComputeWorkGroupCount / PXG_VK_LIMIT_WORKGROUP_COUNT
    // lower each dimension of maxComputeWorkGroupCount, which dispatch checks the grid
    // against.
    uint32_t limitMaxPushConstantsSize = 0;
    uint32_t limitMaxComputeSharedMemorySize = 0;
    uint32_t limitMaxComputeWorkGroupInvocations = 0;
    uint32_t limitMaxComputeWorkGroupSize = 0;
    uint32_t limitMaxComputeWorkGroupCount = 0;
    // P9.99 test hooks for createPipelineEx.  Like the limits above they only take a
    // capability AWAY, so a fixture can show each refusal on llvmpipe; the device keeps
    // the features enabled, only what createPipelineEx checks against changes.
    // limitMaxComputeWorkgroupSubgroups lowers SubgroupCaps::maxComputeWorkgroupSubgroups
    // (never raises it); noSubgroupSizeControl / noFullSubgroups clear canRequireSize /
    // fullSubgroups.  No environment variables: nothing outside the fixtures needs them.
    uint32_t limitMaxComputeWorkgroupSubgroups = 0;
    bool noSubgroupSizeControl = false;
    bool noFullSubgroups = false;
};

class PxgSlangComputeDevice
{
public:
    virtual ~PxgSlangComputeDevice() {}

    virtual const char* getDeviceName() const = 0;
    virtual const char* getBackendName() const = 0;

    // ---- resources ----
    // Returns 0 when the buffer cannot be created (an out-of-memory allocation on the
    // Vulkan backend since P9.98: the attempt is released and the failing call named on
    // stderr; the shim reports OutOfMemory).
    virtual PxgBufferHandle createBuffer(size_t sizeBytes) = 0;
    virtual void destroyBuffer(PxgBufferHandle buffer) = 0;

    // Pinned-host buffer (CUDA cuMemHostAlloc parity, RUNTIME_PLAN.md P5 Step
    // 4b): host-visible memory persistently mapped for the buffer's lifetime —
    // the CPU reads/writes it through `*mappedPtr` while shaders access it by
    // buffer device address.  Freed with destroyBuffer.  Returns 0 (and leaves
    // *mappedPtr untouched) on backends without pinned-mapped support; the Vulkan
    // backend returns 0 with *mappedPtr = NULL when the allocation fails.
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
    // P9.98: returns false -- and records NOTHING -- when the pipeline handle is 0
    // or unknown (e.g. a createPipeline* that was refused) or the push blob is
    // larger than the pipeline's push range; since P9.98 check round 0 also when
    // numBuffers / numTextures differ from the pipeline's, a buffer or texture handle
    // is 0 or unknown (dispatchIndirect: the arg buffer too, and an arg offset that is
    // not a 4-byte aligned command inside it), or no descriptor set can be allocated;
    // since check round 1 also when gridX / gridY / gridZ exceeds the device's
    // maxComputeWorkGroupCount (65535 on llvmpipe and the Vulkan minimum -- CUDA allows a
    // gridDim.x up to 2^31-1, Vulkan does not; dispatchIndirect's counts live on the device
    // and cannot be checked here).
    // The device prints why; the shim turns it into a refused launch (NotFound),
    // exactly like a pipeline it could not build.
    virtual bool dispatch(PxgPipelineHandle pipeline, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                          const PxgBufferHandle* buffers, uint32_t numBuffers,
                          const void* pushConstants, uint32_t pushConstantSize) = 0;

    // Indirect dispatch (cuLaunchKernel with a device-computed grid): the
    // workgroup counts are read from `argBuffer` at `argOffset` as three
    // consecutive uint32 (x, y, z) — the VkDispatchIndirectCommand /
    // D3D12 dispatch-args layout.  Lets a kernel compute the next grid on
    // device without a count readback (RUNTIME_PLAN.md P6).  Base default
    // is unsupported (returns without recording); backends override.
    virtual bool dispatchIndirect(PxgPipelineHandle /*pipeline*/, PxgBufferHandle /*argBuffer*/,
                                  size_t /*argOffset*/, const PxgBufferHandle* /*buffers*/,
                                  uint32_t /*numBuffers*/, const void* /*pushConstants*/,
                                  uint32_t /*pushConstantSize*/) { return false; }
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

    // The ticket the work recorded so far WILL carry, without submitting it (a CUDA
    // event record: a marker, not a flush); isSubmitted() says whether that ticket's
    // batch has been handed to the device yet -- if not, submit() before wait()/poll().
    // Defaults keep the submit-at-record behaviour for backends without lazy tickets.
    virtual uint64_t pendingTicket() { return submit(); }

    // Pipeline creation from several threads at once (createPipeline*, nothing else):
    // true when the backend's create calls may run concurrently with each other and
    // with recording/dispatch on the API thread.  The shim's background warm-up
    // (PxgCudaShim::startPipelineWarmup) is only used when this is true.
    virtual bool supportsConcurrentPipelineCreation() const { return false; }
    // The pipeline cache's current contents (to persist across runs); false when the
    // backend has none.  May be called while pipelines are being created.
    virtual bool getPipelineCacheData(std::vector<uint8_t>* /*out*/) { return false; }
    virtual bool isSubmitted(uint64_t /*ticket*/) const { return true; }

    // ---- Phase-2 runtime capabilities (RUNTIME_PLAN.md) ----
    // Additive with conservative defaults so backends opt in individually.

    struct SubgroupCaps
    {
        uint32_t defaultSize = 0;   // 0 = subgroup queries unsupported
        uint32_t minSize = 0;
        uint32_t maxSize = 0;
        bool canRequireSize = false; // subgroupSizeControl usable on compute
        bool fullSubgroups = false;  // computeFullSubgroups
        // P9.99: VkPhysicalDeviceSubgroupSizeControlProperties::maxComputeWorkgroupSubgroups.
        // A required size S admits at most S x this many invocations (VUID-02756); 0 = unknown.
        uint32_t maxComputeWorkgroupSubgroups = 0;
    };
    virtual SubgroupCaps getSubgroupCaps() const { return SubgroupCaps(); }

    // Like createPipeline, but requests an exact subgroup size WITH full subgroups
    // (REQUIRE_FULL_SUBGROUPS) for the entry point.  requiredSubgroupSize == 0 behaves
    // exactly like createPipeline.  Returns 0, with a "; refused" line naming the rule,
    // when the size cannot be honoured for THIS entry (P9.99: the device's window AND the
    // entry's workgroup shape, pxgRequiredSubgroupSizeFits).  This default -- the D3D12
    // backend's -- cannot require a size at all.
    virtual PxgPipelineHandle createPipelineEx(const void* code, size_t codeSize, const char* entryName,
                                               uint32_t numBuffers, const uint32_t* bufferStrides,
                                               uint32_t pushConstantSize, uint32_t requiredSubgroupSize)
    {
        if (requiredSubgroupSize != 0)
        {
            std::fprintf(stderr, "[PxgSlangComputeDevice] createPipelineEx(%s): required subgroup size %u: the %s "
                                 "backend cannot require a subgroup size; refused\n",
                         entryName ? entryName : "?", requiredSubgroupSize, getBackendName());
            return 0;
        }
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
    virtual bool dispatchTex(PxgPipelineHandle /*pipeline*/, uint32_t /*gridX*/, uint32_t /*gridY*/,
                             uint32_t /*gridZ*/, const PxgBufferHandle* /*buffers*/, uint32_t /*numBuffers*/,
                             const PxgTextureHandle* /*textures*/, uint32_t /*numTextures*/,
                             const void* /*pushConstants*/, uint32_t /*pushConstantSize*/) { return false; }

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

    // ---- P9.98 diagnostics -- APPENDED like the P9.97 ones ----

    // The limits createPipeline* checks every entry point against: the device's
    // VkPhysicalDeviceLimits, lowered by the PxgSlangDeviceDesc / environment test
    // hooks.  All zero on a backend that does not check.  maxComputeWorkGroupCount
    // (check round 1) is the one dispatch checks the grid against.
    struct ComputeLimits
    {
        uint32_t maxPushConstantsSize = 0;
        uint32_t maxComputeSharedMemorySize = 0;
        uint32_t maxComputeWorkGroupInvocations = 0;
        uint32_t maxComputeWorkGroupSize[3] = { 0, 0, 0 };
        uint32_t maxComputeWorkGroupCount[3] = { 0, 0, 0 };
    };
    virtual ComputeLimits computeLimits() const { return ComputeLimits(); }

    // The size of the push-constant range the pipeline's LAYOUT declares:
    // max(the caller's pushConstantSize, the entry point's SPIR-V push block), so the
    // declared block always lies inside the layout (VUID-07987, P9.98 F2).  ~0u for a
    // handle the device does not hold (0 is a held pipeline with no push range), and
    // on a backend that does not report it.
    virtual uint32_t pipelinePushRangeBytes(PxgPipelineHandle /*pipeline*/) const { return ~0u; }
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

// P9.98: what one entry point of a SPIR-V module asks of the device, read from the
// binary.  PER ENTRY POINT when the module is SPIR-V 1.4+ (OpEntryPoint's interface
// list then names every global the entry statically uses -- Slang emits 1.5), else
// module-wide (perEntry = false), which can only over-count.
//   pushBlockBytes  the entry's PushConstant block: max(member Offset + size), rounded
//                   up to 4 (0 = none).  The size VUID-07987 compares the layout with.
//   workgroupBytes  the summed size of its Workgroup variables (the same rules as
//                   tools/check_groupshared.py) -- what maxComputeSharedMemorySize caps.
//   localSize       OpExecutionMode LocalSize (or LocalSizeId of constants); 0 = unknown.
// A size the walker cannot decide clears *Decided (the caller then skips that check).
// Check round 1: sizes are computed in 64 bits and a size that does not fit 32 bits is
// reported as UINT32_MAX (decided, so over every limit) -- they used to wrap, and a 4 GiB
// groupshared array reflected as 0 B and was built.
struct PxgSpirvEntryInfo
{
    uint32_t pushBlockBytes = 0;
    uint32_t workgroupBytes = 0;
    uint32_t localSize[3] = { 0, 0, 0 };
    bool pushDecided = true;
    bool workgroupDecided = true;
    bool perEntry = true;
};

// False when the blob is not a well-formed SPIR-V word stream or declares no entry
// point named `entry`; *out is then left default.  Defined by the Vulkan backend.
bool pxgSpirvReflectEntry(const void* code, size_t codeSize, const char* entry, PxgSpirvEntryInfo* out);

// P9.99: can an entry of workgroup size `localSize` (pxgSpirvReflectEntry's; 0 = unknown) be
// built with `requiredSubgroupSize` and full subgroups -- createPipelineEx's contract -- on a
// device with `caps`?  0 = no request (true).  The rules, in order: the device can require a
// size in compute (VUID-02755) and has computeFullSubgroups (VUID-02785); the size is a power
// of two (VUID-02760) inside [minSize, maxSize] (VUID-02761/02762); the shape is known; x is a
// multiple of the size (VUID-02757); x*y*z <= size * maxComputeWorkgroupSubgroups
// (VUID-02756).  On false, `why` (if non-null and whyLen > 0) names the rule.  Pure host
// function, defined by the Vulkan backend.
bool pxgRequiredSubgroupSizeFits(const PxgSlangComputeDevice::SubgroupCaps& caps, uint32_t requiredSubgroupSize,
                                 const uint32_t localSize[3], char* why, size_t whyLen);

// Factory functions; each returns NULL when the backend is unavailable.
PxgSlangComputeDevice* PxgCreateVulkanComputeDevice(const PxgSlangDeviceDesc& desc);
#if defined(_WIN32)
PxgSlangComputeDevice* PxgCreateDx12ComputeDevice(const PxgSlangDeviceDesc& desc);
#endif

} // namespace gpuslang
} // namespace physx

#endif // PXG_SLANG_COMPUTE_DEVICE_H
