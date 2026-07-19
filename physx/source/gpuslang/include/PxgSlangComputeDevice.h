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

    // Compute->compute execution + memory barrier between recorded dispatches
    // (the ordering CUDA guarantees implicitly on a single stream).
    virtual void barrier() = 0;

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
};

// Factory functions; each returns NULL when the backend is unavailable.
PxgSlangComputeDevice* PxgCreateVulkanComputeDevice(const PxgSlangDeviceDesc& desc);
#if defined(_WIN32)
PxgSlangComputeDevice* PxgCreateDx12ComputeDevice(const PxgSlangDeviceDesc& desc);
#endif

} // namespace gpuslang
} // namespace physx

#endif // PXG_SLANG_COMPUTE_DEVICE_H
