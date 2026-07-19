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

// PxgVulkanDriverShim.cpp — RUNTIME_PLAN.md P4: definitions for the CUDA-driver
// entry points the gpu* host modules call OUTSIDE the PxCudaContext virtual
// surface.  In the CUDA build these resolve to libcuda; in the Vulkan / no-toolkit
// build there is no libcuda, so the bridge supplies backend-appropriate shims.
// Declarations live in gpucommon/include/PxgCudaToolkitShim.h.

#include "foundation/PxPreprocessor.h"

#if PX_SUPPORT_GPU_PHYSX

#include "PxgCudaToolkitShim.h"
#include "cudamanager/PxCudaContext.h"
#include <cstdio>

// The live PxCudaContext, registered by PxgVulkanCudaContext on construction, so the
// raw (context-less) driver entry points below can route through the Vulkan bridge.
namespace physx { namespace gpuslang {
static physx::PxCudaContext* gDriverShimContext = nullptr;
void pxgSetVulkanDriverContext(physx::PxCudaContext* ctx) { gDriverShimContext = ctx; }
} }

// cuCtxGetStreamPriorityRange: the engine calls this to pick a priority for
// streamCreateWithPriority.  The Vulkan backend exposes a single compute queue
// with no driver-level stream priority tiers, so report a degenerate [0,0] range
// (least == greatest == 0).  The PxCudaContext bridge's streamCreateWithPriority
// ignores the priority argument, so the chosen value is inert regardless.
extern "C" CUresult cuCtxGetStreamPriorityRange(int* leastPriority, int* greatestPriority)
{
    if (leastPriority)
        *leastPriority = 0;
    if (greatestPriority)
        *greatestPriority = 0;
    return CUDA_SUCCESS;
}

// cuStreamQuery: report the stream's work complete.  The Vulkan submission model
// drains a stream's work synchronously at the bridge's flush/synchronize points, so
// by the time the engine polls, the work is done -> "ready".
extern "C" CUresult cuStreamQuery(CUstream /*hStream*/)
{
    return CUDA_SUCCESS;
}

// Raw memcpys route through the live bridge context (interior flat-address resolve).
extern "C" CUresult cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount)
{
    if (!physx::gpuslang::gDriverShimContext) return CUDA_ERROR_NOT_READY;
    return CUresult(physx::gpuslang::gDriverShimContext->memcpyDtoH(dstHost, srcDevice, ByteCount));
}
extern "C" CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount)
{
    if (!physx::gpuslang::gDriverShimContext) return CUDA_ERROR_NOT_READY;
    return CUresult(physx::gpuslang::gDriverShimContext->memcpyHtoD(dstDevice, srcHost, ByteCount));
}
extern "C" CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount)
{
    if (!physx::gpuslang::gDriverShimContext) return CUDA_ERROR_NOT_READY;
    return CUresult(physx::gpuslang::gDriverShimContext->memcpyDtoD(dstDevice, srcDevice, ByteCount));
}

// --- Texture-object / CUDA-array API: SDF + heightfield 3D textures only.  These
// are NOT on the rigid-body pipeline.  Backing them with real Vulkan sampled images
// (via the gpuslang tex3D layer) is the SDF/heightfield bring-up; until then they
// error clearly if a scene actually registers such geometry, rather than silently
// returning bad handles. ---
static CUresult pxgTextureNotWired(const char* what)
{
    std::fprintf(stderr,
        "[PxgVulkan] %s is not yet wired (SDF/heightfield GPU textures); "
        "rigid/convex geometry does not use this path.\n", what);
    return CUDA_ERROR_NOT_READY;
}
extern "C" CUresult cuArray3DCreate(CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR*)
{
    if (pHandle) *pHandle = nullptr;
    return pxgTextureNotWired("cuArray3DCreate");
}
extern "C" CUresult cuArrayDestroy(CUarray) { return CUDA_SUCCESS; }
extern "C" CUresult cuTexObjectCreate(CUtexObject* pTexObject, const CUDA_RESOURCE_DESC*,
                                      const CUDA_TEXTURE_DESC*, const CUDA_RESOURCE_VIEW_DESC*)
{
    if (pTexObject) *pTexObject = 0;
    return pxgTextureNotWired("cuTexObjectCreate");
}
extern "C" CUresult cuTexObjectDestroy(CUtexObject) { return CUDA_SUCCESS; }
extern "C" CUresult cuMemcpy3D(const CUDA_MEMCPY3D*) { return pxgTextureNotWired("cuMemcpy3D"); }
extern "C" CUresult cuMemcpy3DAsync(const CUDA_MEMCPY3D*, CUstream) { return pxgTextureNotWired("cuMemcpy3DAsync"); }

#endif // PX_SUPPORT_GPU_PHYSX
