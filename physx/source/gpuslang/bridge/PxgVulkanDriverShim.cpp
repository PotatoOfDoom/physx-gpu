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

#endif // PX_SUPPORT_GPU_PHYSX
