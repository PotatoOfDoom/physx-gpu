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

// PxgVulkanKernelInit.cpp — RUNTIME_PLAN.md P4: the per-module kernel-registration
// anchors, no-op'd for the SPIR-V build.
//
// In the CUDA build each gpu* module's .cu translation units expose an
// extern "C" void initXxxKernels<N>() symbol that the module's host code calls at
// startup; the call exists purely to force the linker to keep the .cu object (and
// its embedded fatbin) so CUDART fatbin registration runs.  In the Vulkan / SPIR-V
// build there are no fatbins — every kernel is resolved by name through the live
// manifest-driven function table (bridge/PxgVulkanKernelTable.cpp) — so these
// anchors carry no work and are defined here as empty, matching the .cu bodies
// (`extern "C" __host__ void initBroadphaseKernels0() {}`).  As each concrete
// subsystem is brought up over Vulkan, its init anchors are added here.

#include "foundation/PxPreprocessor.h"

#if PX_SUPPORT_GPU_PHYSX

extern "C"
{
// --- gpubroadphase (PxgBroadPhase.cpp) ---
void initBroadphaseKernels0() {}
void initBroadphaseKernels1() {}
}

#endif // PX_SUPPORT_GPU_PHYSX
