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

// PxgCudaToolkitShim.h — RUNTIME_PLAN.md P4: the no-CUDA-toolkit driver-API shim.
//
// In the Vulkan / SPIR-V build the gpu* host modules compile WITHOUT the CUDA
// toolkit (no <cuda.h>).  PxCudaTypes.h already supplies the opaque handle types
// (CUdeviceptr / CUstream / CUevent / CUmodule / CUfunction / CUcontext), and the
// PxCudaContext virtual surface replaces the bulk of the driver API.  This header
// fills the small remaining gap the host modules reference directly:
//   * the driver result type + success code (CUresult / CUDA_SUCCESS),
//   * the handful of stream/event creation flag enumerators passed by name, and
//   * declarations for the driver entry points a few modules call OUTSIDE the
//     PxCudaContext surface (defined by bridge/PxgVulkanDriverShim.cpp).
//
// Each gpucommon host header that used to `#include "cuda.h"` now selects between
// the real toolkit header and this shim on PXG_NO_CUDA_TOOLKIT, so the toolkit
// build is byte-for-byte unchanged and only the no-toolkit build sees the shim.

#ifndef PXG_CUDA_TOOLKIT_SHIM_H
#define PXG_CUDA_TOOLKIT_SHIM_H

#include "foundation/PxPreprocessor.h"

#if PX_SUPPORT_GPU_PHYSX

// Opaque driver handle types (CUdeviceptr/CUstream/CUevent/CUmodule/CUfunction/...).
#include "cudamanager/PxCudaTypes.h"

// Texture / CUDA-array handles the geometry manager references for its texture-
// object path (not covered by PxCudaTypes.h).  Values/spellings mirror cuda.h:
// CUarray is an opaque pointer, CUtexObject an unsigned 64-bit handle.
#ifndef PXG_SHIM_TEXTURE_HANDLES
#define PXG_SHIM_TEXTURE_HANDLES
typedef struct CUarray_st* CUarray;
typedef struct CUmipmappedArray_st* CUmipmappedArray;
typedef struct CUtexref_st* CUtexref;
typedef unsigned long long CUtexObject;
typedef unsigned long long CUsurfObject;
#endif

// The driver result type and the success code the modules compare against.  cuda.h
// models CUresult as an enum with CUDA_SUCCESS==0; a plain int matches the ABI and
// the PxCudaContext bridge already returns these values through PxCUenum (which is
// convertible to/from the integer code).  Guarded so a translation unit that
// separately hard-defines CUDA_SUCCESS (e.g. CudaKernelWrangler.cpp) never clashes.
#ifndef CUDA_SUCCESS
typedef int CUresult;
#define CUDA_SUCCESS 0
#endif

// Stream / event creation flag enumerators passed by name.  Values mirror cuda.h.
// The Vulkan bridge has no driver-level stream priority or event-timing tiers, so
// only the names must resolve — streamCreateWithPriority / eventCreate ignore them.
#ifndef CU_STREAM_DEFAULT
#define CU_STREAM_DEFAULT       0x0
#define CU_STREAM_NON_BLOCKING  0x1
#endif
#ifndef CU_EVENT_DEFAULT
#define CU_EVENT_DEFAULT        0x0
#define CU_EVENT_DISABLE_TIMING 0x2
#endif
#ifndef CU_MEMHOSTALLOC_PORTABLE
#define CU_MEMHOSTALLOC_PORTABLE   0x1
#define CU_MEMHOSTALLOC_DEVICEMAP  0x2
#define CU_MEMHOSTALLOC_WRITECOMBINED 0x4
#endif

// A few driver result codes the modules compare against by name (beyond
// CUDA_SUCCESS).  Values mirror cuda.h's CUresult enum.
#ifndef CUDA_ERROR_INVALID_VALUE
#define CUDA_ERROR_INVALID_VALUE    1
#define CUDA_ERROR_OUT_OF_MEMORY    2
#define CUDA_ERROR_NOT_READY        600
#endif

// --- CUDA array / texture-object API (used by PxgGeometryManager for SDF and
// heightfield 3D textures).  Structs/enums mirror cuda.h; only host-referenced
// fields are reproduced.  The Vulkan backing lives in the gpuslang tex3D layer and
// is wired during SDF/heightfield bring-up — these entry points are not on the
// rigid-body pipeline, so the driver shim provides inert definitions for now. ---
#ifndef PXG_SHIM_TEXTURE_API
#define PXG_SHIM_TEXTURE_API
typedef enum CUarray_format_enum {
    CU_AD_FORMAT_UNSIGNED_INT8  = 0x01,
    CU_AD_FORMAT_UNSIGNED_INT16 = 0x02,
    CU_AD_FORMAT_UNSIGNED_INT32 = 0x03,
    CU_AD_FORMAT_SIGNED_INT8    = 0x08,
    CU_AD_FORMAT_SIGNED_INT16   = 0x09,
    CU_AD_FORMAT_SIGNED_INT32   = 0x0a,
    CU_AD_FORMAT_HALF           = 0x10,
    CU_AD_FORMAT_FLOAT          = 0x20
} CUarray_format;
typedef enum CUaddress_mode_enum {
    CU_TR_ADDRESS_MODE_WRAP = 0, CU_TR_ADDRESS_MODE_CLAMP = 1,
    CU_TR_ADDRESS_MODE_MIRROR = 2, CU_TR_ADDRESS_MODE_BORDER = 3
} CUaddress_mode;
typedef enum CUfilter_mode_enum { CU_TR_FILTER_MODE_POINT = 0, CU_TR_FILTER_MODE_LINEAR = 1 } CUfilter_mode;
typedef enum CUresourcetype_enum {
    CU_RESOURCE_TYPE_ARRAY = 0, CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = 1,
    CU_RESOURCE_TYPE_LINEAR = 2, CU_RESOURCE_TYPE_PITCH2D = 3
} CUresourcetype;
typedef enum CUmemorytype_enum {
    CU_MEMORYTYPE_HOST = 1, CU_MEMORYTYPE_DEVICE = 2, CU_MEMORYTYPE_ARRAY = 3, CU_MEMORYTYPE_UNIFIED = 4
} CUmemorytype;
#define CU_TRSF_READ_AS_INTEGER 1
typedef struct CUDA_ARRAY3D_DESCRIPTOR_st {
    size_t Width; size_t Height; size_t Depth;
    CUarray_format Format; unsigned int NumChannels; unsigned int Flags;
} CUDA_ARRAY3D_DESCRIPTOR;
typedef struct CUDA_RESOURCE_DESC_st {
    CUresourcetype resType;
    union { struct { CUarray hArray; } array; struct { CUmipmappedArray hMipmappedArray; } mipmap;
            struct { CUdeviceptr devPtr; CUarray_format format; unsigned int numChannels; size_t sizeInBytes; } linear;
            struct { CUdeviceptr devPtr; CUarray_format format; unsigned int numChannels; size_t width, height, pitchInBytes; } pitch2D;
            struct { int reserved[32]; } reserved; } res;
    unsigned int flags;
} CUDA_RESOURCE_DESC;
typedef struct CUDA_TEXTURE_DESC_st {
    CUaddress_mode addressMode[3]; CUfilter_mode filterMode; unsigned int flags;
    unsigned int maxAnisotropy; CUfilter_mode mipmapFilterMode;
    float mipmapLevelBias; float minMipmapLevelClamp; float maxMipmapLevelClamp;
    float borderColor[4]; int reserved[12];
} CUDA_TEXTURE_DESC;
typedef struct CUDA_RESOURCE_VIEW_DESC_st { int format; size_t width, height, depth; unsigned int firstMipmapLevel, lastMipmapLevel, firstLayer, lastLayer, reserved[16]; } CUDA_RESOURCE_VIEW_DESC;
typedef struct CUDA_MEMCPY3D_st {
    size_t srcXInBytes, srcY, srcZ, srcLOD; CUmemorytype srcMemoryType;
    const void* srcHost; CUdeviceptr srcDevice; CUarray srcArray; void* reserved0;
    size_t srcPitch, srcHeight;
    size_t dstXInBytes, dstY, dstZ, dstLOD; CUmemorytype dstMemoryType;
    void* dstHost; CUdeviceptr dstDevice; CUarray dstArray; void* reserved1;
    size_t dstPitch, dstHeight; size_t WidthInBytes, Height, Depth;
} CUDA_MEMCPY3D;
#endif // PXG_SHIM_TEXTURE_API

// Direct CUDA-driver entry points the gpu modules call outside the PxCudaContext
// virtual surface.  Defined by the Vulkan driver shim (bridge/PxgVulkanDriverShim.cpp).
#ifdef __cplusplus
extern "C" {
#endif

// The broadphase (and other modules) query the stream priority range to pick a
// priority for streamCreateWithPriority.  The Vulkan backend has a single compute
// queue with no priority tiers; the shim reports a degenerate [0,0] range.
CUresult cuCtxGetStreamPriorityRange(int* leastPriority, int* greatestPriority);

// Raw memcpy / stream-query entry points a few modules call outside the context
// surface.  The shim routes memcpys through the live PxCudaContext (registered by
// PxgVulkanCudaContext); cuStreamQuery reports ready (the Vulkan submission model
// drains synchronously).
CUresult cuStreamQuery(CUstream hStream);
CUresult cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount);
CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount);
CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount);

// Texture-object / array entry points (SDF + heightfield only; see note above).
CUresult cuArray3DCreate(CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray);
CUresult cuArrayDestroy(CUarray hArray);
CUresult cuTexObjectCreate(CUtexObject* pTexObject, const CUDA_RESOURCE_DESC* pResDesc,
                           const CUDA_TEXTURE_DESC* pTexDesc, const CUDA_RESOURCE_VIEW_DESC* pResViewDesc);
CUresult cuTexObjectDestroy(CUtexObject texObject);
CUresult cuMemcpy3D(const CUDA_MEMCPY3D* pCopy);
CUresult cuMemcpy3DAsync(const CUDA_MEMCPY3D* pCopy, CUstream hStream);

#ifdef __cplusplus
}
#endif

#endif // PX_SUPPORT_GPU_PHYSX
#endif // PXG_CUDA_TOOLKIT_SHIM_H
