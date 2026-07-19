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

// Direct CUDA-driver entry points the gpu modules call outside the PxCudaContext
// virtual surface.  Defined by the Vulkan driver shim (bridge/PxgVulkanDriverShim.cpp).
#ifdef __cplusplus
extern "C" {
#endif

// The broadphase (and other modules) query the stream priority range to pick a
// priority for streamCreateWithPriority.  The Vulkan backend has a single compute
// queue with no priority tiers; the shim reports a degenerate [0,0] range.
CUresult cuCtxGetStreamPriorityRange(int* leastPriority, int* greatestPriority);

#ifdef __cplusplus
}
#endif

#endif // PX_SUPPORT_GPU_PHYSX
#endif // PXG_CUDA_TOOLKIT_SHIM_H
