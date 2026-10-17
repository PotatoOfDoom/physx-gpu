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
#include "PxgCudaShim.h"
#include "cudamanager/PxCudaContext.h"
#include <cstdio>

// The live PxCudaContext, registered by PxgVulkanCudaContext on construction, so the
// raw (context-less) driver entry points below can route through the Vulkan bridge.
namespace physx { namespace gpuslang {
static physx::PxCudaContext* gDriverShimContext = nullptr;
void pxgSetVulkanDriverContext(physx::PxCudaContext* ctx) { gDriverShimContext = ctx; }
// The 3D-texture entry points have no PxCudaContext counterpart, so they need the
// shim itself (P9.12b).
static PxgCudaShim* gDriverShimShim = nullptr;
void pxgSetVulkanDriverShim(PxgCudaShim* shim) { gDriverShimShim = shim; }
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

// cuStreamSynchronize: 5.12's SDF upload (PxgGeometryManager) waits on its stream
// after cuMemcpy3DAsync.  Route to the live context's stream sync; with no context
// there is no recorded work to wait for.
extern "C" CUresult cuStreamSynchronize(CUstream hStream)
{
    if (!physx::gpuslang::gDriverShimContext) return CUDA_SUCCESS;
    return CUresult(physx::gpuslang::gDriverShimContext->streamSynchronize(hStream));
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

// --- Texture-object / CUDA-array API: the SDF 3D-texture upload path (P9.12b).
//
// The engine's sequence is cuArray3DCreate -> cuMemcpy3DAsync -> cuTexObjectCreate
// (PxgGeometryManager.cpp:732-770), and it is translated onto the shim's bindless
// texture layer rather than being reimplemented here: a CUarray is a host staging
// record and the Vulkan image is created at cuTexObjectCreate, the first moment
// dimensions, format AND contents are all known.
//
// A CUarray is an opaque pointer in the CUDA API but an integer id here, so the id
// is carried IN the pointer.  Ids start at 1 (PxgCudaShim::mNextArrayId), so a live
// array is never the null pointer -- which matters because the engine writes
// `cuArray = 0` to mean "no array" (PxgGeometryManager.cpp:727-731).
static CUarray pxgArrayToHandle(uint32_t id) { return reinterpret_cast<CUarray>(uintptr_t(id)); }
static uint32_t pxgHandleToArray(CUarray h) { return uint32_t(reinterpret_cast<uintptr_t>(h)); }

static CUresult pxgTextureNoShim(const char* what)
{
    std::fprintf(stderr, "[PxgVulkan] %s called with no live bridge shim\n", what);
    return CUDA_ERROR_NOT_READY;
}

// CUDA promotes an integer texture format to a normalized float UNLESS
// CU_TRSF_READ_AS_INTEGER is set, which is why the UNORM Vulkan formats are the
// right equivalents.  Formats the SDF path never asks for are refused rather than
// approximated: silently substituting a format would change sampled values.
static bool pxgMapArrayFormat(CUarray_format f, unsigned int numChannels,
                              physx::gpuslang::PxgSlangComputeDevice::TextureFormat* out)
{
    using TF = physx::gpuslang::PxgSlangComputeDevice::TextureFormat;
    if (numChannels != 1)
        return false;
    switch (f)
    {
    case CU_AD_FORMAT_FLOAT:          *out = TF::R32Float;  return true;
    case CU_AD_FORMAT_UNSIGNED_INT8:  *out = TF::R8Unorm;   return true;
    case CU_AD_FORMAT_UNSIGNED_INT16: *out = TF::R16Unorm;  return true;
    default:                          return false;
    }
}

extern "C" CUresult cuArray3DCreate(CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray)
{
    if (pHandle) *pHandle = nullptr;
    if (!pHandle || !pAllocateArray)
        return CUDA_ERROR_INVALID_VALUE;
    if (!physx::gpuslang::gDriverShimShim)
        return pxgTextureNoShim("cuArray3DCreate");

    physx::gpuslang::PxgSlangComputeDevice::TextureFormat fmt;
    if (!pxgMapArrayFormat(pAllocateArray->Format, pAllocateArray->NumChannels, &fmt))
    {
        std::fprintf(stderr, "[PxgVulkan] cuArray3DCreate: unsupported format 0x%x x %u channels\n",
                     unsigned(pAllocateArray->Format), pAllocateArray->NumChannels);
        return CUDA_ERROR_INVALID_VALUE;
    }

    uint32_t id = 0;
    const physx::gpuslang::PxgResult r = physx::gpuslang::gDriverShimShim->arrayCreate3D(
        &id, uint32_t(pAllocateArray->Width), uint32_t(pAllocateArray->Height),
        uint32_t(pAllocateArray->Depth), fmt);
    if (r != physx::gpuslang::PxgResult::Success)
        return r == physx::gpuslang::PxgResult::Unsupported ? CUDA_ERROR_NOT_READY
                                                            : CUDA_ERROR_INVALID_VALUE;
    *pHandle = pxgArrayToHandle(id);
    return CUDA_SUCCESS;
}

extern "C" CUresult cuArrayDestroy(CUarray hArray)
{
    if (!hArray || !physx::gpuslang::gDriverShimShim)
        return CUDA_SUCCESS;
    physx::gpuslang::gDriverShimShim->arrayDestroy(pxgHandleToArray(hArray));
    return CUDA_SUCCESS;
}

extern "C" CUresult cuTexObjectCreate(CUtexObject* pTexObject, const CUDA_RESOURCE_DESC* pResDesc,
                                      const CUDA_TEXTURE_DESC* pTexDesc, const CUDA_RESOURCE_VIEW_DESC* pResViewDesc)
{
    if (pTexObject) *pTexObject = 0;
    if (!pTexObject || !pResDesc || !pTexDesc)
        return CUDA_ERROR_INVALID_VALUE;
    if (!physx::gpuslang::gDriverShimShim)
        return pxgTextureNoShim("cuTexObjectCreate");
    if (pResDesc->resType != CU_RESOURCE_TYPE_ARRAY || !pResDesc->res.array.hArray)
        return CUDA_ERROR_INVALID_VALUE;

    // ONE TEXTURE DESCRIPTOR IS IMPLEMENTED, so a descriptor that asks for
    // anything else would be silently read with the wrong addressing, filter or
    // coordinate convention.  Since P9.96 there is no sampler at all: pxgTex3D
    // (shaders/pxgSlangTex3d.slang) fetches texels and evaluates exactly one
    // descriptor in shader code -- clamp on all three axes, linear, normalized
    // element reads, UNNORMALIZED coordinates -- because that is what the engine
    // builds every SDF texture with (PxgGeometryManager.cpp:762-770: PxMemZero'd,
    // flags 0, no resource view).  The remedy for anything else is stated: extend
    // pxgTex3D (and carry the mode with the handle).
    //
    // CHECKED, and refused unless equal to the engine's: the three address modes,
    // the filter mode, EVERY flags bit (READ_AS_INTEGER, NORMALIZED_COORDINATES --
    // refused since P9.96, the shader always read texel-space coordinates -- and
    // any other, e.g. CU_TRSF_SRGB 0x10, which would change the 8-bit conversion),
    // and the resource view (must be absent).  NOT checked: maxAnisotropy, the
    // mipmap fields and borderColor.  The engine zeroes them, and they are not
    // expected to change a sample of a single-level CUarray under clamp addressing
    // (borderColor is read only in BORDER mode, which is refused), but nothing here
    // verifies that.
    const bool clamped = pTexDesc->addressMode[0] == CU_TR_ADDRESS_MODE_CLAMP &&
                         pTexDesc->addressMode[1] == CU_TR_ADDRESS_MODE_CLAMP &&
                         pTexDesc->addressMode[2] == CU_TR_ADDRESS_MODE_CLAMP;
    const bool linear = pTexDesc->filterMode == CU_TR_FILTER_MODE_LINEAR;
    const bool elementRead = (pTexDesc->flags & CU_TRSF_READ_AS_INTEGER) == 0;
    const bool unnormalized = (pTexDesc->flags & CU_TRSF_NORMALIZED_COORDINATES) == 0;
    const bool noOtherFlag =
        (pTexDesc->flags & ~unsigned(CU_TRSF_READ_AS_INTEGER | CU_TRSF_NORMALIZED_COORDINATES)) == 0;
    const bool noView = pResViewDesc == nullptr;
    if (!clamped || !linear || !elementRead || !unnormalized || !noOtherFlag || !noView)
    {
        std::fprintf(stderr,
            "[PxgVulkan] cuTexObjectCreate: this bridge implements ONE texture descriptor in "
            "pxgTex3D (clamp/clamp/clamp, linear, normalized reads, unnormalized coordinates, "
            "flags 0, no resource view); asked for %d/%d/%d, filter %d, flags 0x%x, %s resource "
            "view.  Extend pxgTex3D (shaders/pxgSlangTex3d.slang) to support it.\n",
            int(pTexDesc->addressMode[0]), int(pTexDesc->addressMode[1]),
            int(pTexDesc->addressMode[2]), int(pTexDesc->filterMode), pTexDesc->flags,
            noView ? "no" : "a");
        return CUDA_ERROR_INVALID_VALUE;
    }

    physx::gpuslang::PxgTextureHandle h = 0;
    const physx::gpuslang::PxgResult r = physx::gpuslang::gDriverShimShim->textureObjectCreate(
        &h, pxgHandleToArray(pResDesc->res.array.hArray));
    if (r != physx::gpuslang::PxgResult::Success)
        return r == physx::gpuslang::PxgResult::Unsupported ? CUDA_ERROR_NOT_READY
                                                            : CUDA_ERROR_INVALID_VALUE;
    *pTexObject = CUtexObject(h);
    return CUDA_SUCCESS;
}

extern "C" CUresult cuTexObjectDestroy(CUtexObject texObject)
{
    if (!physx::gpuslang::gDriverShimShim)
        return CUDA_SUCCESS;
    physx::gpuslang::gDriverShimShim->textureObjectDestroy(
        physx::gpuslang::PxgTextureHandle(texObject));
    return CUDA_SUCCESS;
}

// Only the one shape the SDF upload uses is accepted: a tightly packed HOST source
// into a whole ARRAY destination, no sub-region.  An offset or a pitch that is not
// the row length would silently land the data somewhere else, so both are checked.
static CUresult pxgMemcpy3D(const CUDA_MEMCPY3D* p, const char* what)
{
    if (!p)
        return CUDA_ERROR_INVALID_VALUE;
    if (!physx::gpuslang::gDriverShimShim)
        return pxgTextureNoShim(what);
    if (p->srcMemoryType != CU_MEMORYTYPE_HOST || p->dstMemoryType != CU_MEMORYTYPE_ARRAY ||
        !p->srcHost || !p->dstArray)
        return CUDA_ERROR_INVALID_VALUE;
    if (p->srcXInBytes || p->srcY || p->srcZ || p->dstXInBytes || p->dstY || p->dstZ)
        return CUDA_ERROR_INVALID_VALUE;
    if (p->srcPitch != p->WidthInBytes || (p->srcHeight != 0 && p->srcHeight != p->Height))
        return CUDA_ERROR_INVALID_VALUE;

    const size_t bytes = p->WidthInBytes * p->Height * p->Depth;
    const physx::gpuslang::PxgResult r =
        physx::gpuslang::gDriverShimShim->arrayUpload(pxgHandleToArray(p->dstArray), p->srcHost, bytes);
    return r == physx::gpuslang::PxgResult::Success ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

extern "C" CUresult cuMemcpy3D(const CUDA_MEMCPY3D* pCopy) { return pxgMemcpy3D(pCopy, "cuMemcpy3D"); }
extern "C" CUresult cuMemcpy3DAsync(const CUDA_MEMCPY3D* pCopy, CUstream)
{
    // The upload is a host-side memcpy into a staging record, so there is nothing
    // to order against the stream; the bytes reach the GPU at cuTexObjectCreate.
    return pxgMemcpy3D(pCopy, "cuMemcpy3DAsync");
}

#endif // PX_SUPPORT_GPU_PHYSX
