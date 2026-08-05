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

// PxgVulkanKernelAbi — RUNTIME_PLAN.md P5 Step 4: the per-kernel CUDA launch
// ABI, keyed by the CUDA __global__ name.  One row per kernel brought up on
// the engine launch path: the ordered argument sizes of the .cu signature.
// This is the single source of truth both launch forms marshal from:
//   - the sized PxCudaKernelParam form validates against the derived
//     push-constant size (pxgKernelAbiPushSize), and
//   - the void** form (PX_CUDA_KERNEL_PARAM2; broadphase/AABB/articulation)
//     carries NO sizes — CUDA reads them from the compiled kernel's metadata,
//     and the bridge reads them from this table instead.
// A kernel with no row is refused cleanly at launch (InvalidValue), never
// dispatched with a guessed blob.

#ifndef PXG_VULKAN_KERNEL_ABI_H
#define PXG_VULKAN_KERNEL_ABI_H

#include <stdint.h>

namespace physx
{
namespace gpuslang
{

struct PxgKernelAbiRow
{
    const char* cudaName;     // CUDA __global__ name (registry key)
    const uint8_t* argSizes;  // per-argument byte sizes, in signature order
    uint32_t argCount;

    // Bit i set: argument i is a pointer into PINNED HOST memory and must be
    // TRANSLATED to a buffer device address before it reaches the shader.
    //
    // WHY THIS EXISTS.  Under CUDA's unified addressing a `cuMemHostAlloc`
    // pointer is simultaneously a valid device pointer, so PhysX hands such
    // pointers straight to kernels — `mMaxConvexMeshTempMemory`
    // (PxgNarrowphaseCore.cpp:223) is one, written on the GPU with an atomicMax
    // and read on the CPU with no copy at all (:1644).  This runtime backs pinned
    // memory with a HOST-VISIBLE VULKAN BUFFER (PxgCudaShim::memHostAlloc), whose
    // CPU mapping and whose device address are unrelated numbers.  The blob
    // builder copies argument bytes verbatim, so without this mask the shader
    // would receive a CPU virtual address and dereference nonsense.
    //
    // Zero for every row that has no such argument, which is all of them today:
    // the four kernels that take that pointer — the convex/sphere trimesh and
    // heightfield cores — are all still unmapped, so this is a prerequisite for
    // bringing them up rather than a fix for a live fault.
    uint32_t hostPtrMask;
};

// nullptr when the kernel has no ABI row yet (not brought up on the engine path).
const PxgKernelAbiRow* pxgFindKernelAbi(const char* cudaName);

// The compiled push-constant struct size for a row: each argument at the
// natural alignment inferred from its size, rounded up to the largest
// alignment seen (a compiled struct's trailing padding).  This mirrors
// CUDA's .param-bank layout and PxgVulkanCudaContext's blob assembly.
uint32_t pxgKernelAbiPushSize(const PxgKernelAbiRow& row);

} // namespace gpuslang
} // namespace physx

#endif // PXG_VULKAN_KERNEL_ABI_H
