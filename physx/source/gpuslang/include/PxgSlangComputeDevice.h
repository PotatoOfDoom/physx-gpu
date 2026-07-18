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

namespace physx
{
namespace gpuslang
{

typedef uint32_t PxgBufferHandle;   // 0 = invalid
typedef uint32_t PxgPipelineHandle; // 0 = invalid

struct PxgSlangDeviceDesc
{
    bool enableValidation = false;
    // When true, prefer a software/CPU device (llvmpipe, WARP) if present —
    // used by the verification test suite.
    bool preferSoftwareDevice = false;
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

    // Compute->compute execution + memory barrier between recorded dispatches
    // (the ordering CUDA guarantees implicitly on a single stream).
    virtual void barrier() = 0;

    // Submits all recorded work and waits for completion.
    virtual void flush() = 0;

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
};

// Factory functions; each returns NULL when the backend is unavailable.
PxgSlangComputeDevice* PxgCreateVulkanComputeDevice(const PxgSlangDeviceDesc& desc);
#if defined(_WIN32)
PxgSlangComputeDevice* PxgCreateDx12ComputeDevice(const PxgSlangDeviceDesc& desc);
#endif

} // namespace gpuslang
} // namespace physx

#endif // PXG_SLANG_COMPUTE_DEVICE_H
