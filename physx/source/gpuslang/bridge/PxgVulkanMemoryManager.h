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

#ifndef PXG_VULKAN_MEMORY_MANAGER_H
#define PXG_VULKAN_MEMORY_MANAGER_H

// PxgVulkanMemoryManager — RUNTIME_PLAN.md P4: the memory-manager coupling seam.
//
// The engine's concrete gpu-module buddy allocator (PxgHeapMemoryAllocator /
// PxgHeapMemoryAllocatorManager, gpucommon) is REUSED verbatim over Vulkan: it
// only ever calls PxVirtualAllocatorCallback::allocate/deallocate on the two
// callbacks a PxsMemoryManager hands it, treating returned addresses as opaque
// flat device pointers (pointer arithmetic on interior offsets, never a host
// deref).  Our PxCudaContext returns exactly such flat CUdeviceptr device
// addresses, and PxgCudaShim's interval map resolves any interior address the
// buddy split produces back to (buffer, offset) at launch/copy time.  So the
// only new code is this ~30-line PxsMemoryManager whose device callback wraps
// the context's memAlloc/memFree and whose host callback wraps its
// memHostAlloc/memFreeHost — the exact mirror of gpucommon's PxgMemoryManager
// (which wraps cuMemAlloc/cuMemHostAlloc).

#include "foundation/PxPreprocessor.h"

#if PX_SUPPORT_GPU_PHYSX

#include "foundation/PxAllocator.h"          // PxVirtualAllocatorCallback
#include "PxsMemoryManager.h"                 // PxsMemoryManager (lowlevel/software)
#include "cudamanager/PxCudaContext.h"        // PxCudaContext + CUdeviceptr

#include <cstdint>

namespace physx
{
namespace gpuslang
{

class PxgVulkanMemoryManager : public physx::PxsMemoryManager
{
public:
    explicit PxgVulkanMemoryManager(physx::PxCudaContext* context)
    {
        mDevice.mContext = context;
        mHost.mContext = context;
    }

    physx::PxVirtualAllocatorCallback* getHostMemoryAllocator() PX_OVERRIDE { return &mHost; }
    physx::PxVirtualAllocatorCallback* getDeviceMemoryAllocator() PX_OVERRIDE { return &mDevice; }

private:
    // Device memory: context memAlloc/memFree.  The returned CUdeviceptr is a
    // flat 64-bit device address the buddy allocator carves interior offsets
    // from; NULL on failure (the buddy allocator checks for it).
    struct DeviceCallback : physx::PxVirtualAllocatorCallback
    {
        physx::PxCudaContext* mContext;
        void* allocate(size_t size, int, const char*, int) PX_OVERRIDE
        {
            CUdeviceptr p = 0;
            if (mContext->memAlloc(&p, size) != 0 || p == 0)
                return NULL;
            return reinterpret_cast<void*>(uintptr_t(p));
        }
        void deallocate(void* ptr) PX_OVERRIDE
        {
            if (ptr)
                mContext->memFree(CUdeviceptr(uintptr_t(ptr)));
        }
    };

    // Host (pinned) memory: context memHostAlloc/memFreeHost (host malloc today).
    struct HostCallback : physx::PxVirtualAllocatorCallback
    {
        physx::PxCudaContext* mContext;
        void* allocate(size_t size, int, const char*, int) PX_OVERRIDE
        {
            void* p = NULL;
            mContext->memHostAlloc(&p, size, 0);
            return p;
        }
        void deallocate(void* ptr) PX_OVERRIDE
        {
            if (ptr)
                mContext->memFreeHost(ptr);
        }
    };

    DeviceCallback mDevice;
    HostCallback mHost;
};

} // namespace gpuslang
} // namespace physx

#endif // PX_SUPPORT_GPU_PHYSX
#endif // PXG_VULKAN_MEMORY_MANAGER_H
