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

// PxgVulkanKernelTable.cpp — RUNTIME_PLAN.md P4: the live CUDA function table.
//
// The engine's KernelWrangler ctor (cudamanager) resolves each kernel by name
// against PxGpuGetCudaFunctionTable(): for every funcName it linear-scans the
// table for a matching entry, then calls
//   mCudaContext->moduleGetFunction(&mCuFunctions[i], cuModules[moduleIndex], funcName).
// In the CUDA build the table is filled by CUDART fatbin registration; in the
// SPIR-V/Vulkan build there are no fatbins, so we build the table directly from
// the generated registry manifest: one entry {moduleIndex=0, functionName=cudaName}
// per MAPPED kernel (spirvFile != null).  Our PxCudaContext::moduleGetFunction
// then matches the name in the manifest and returns toFunc(id) = id+1, so
// mCuFunctions[id] holds our launch encoding — the exact getCuFunction(id) ->
// launchKernel path the gpu subsystems use.
//
// P9.40: UNMAPPED kernels are in the table too.  Until P9.40 they were absent, the
// wrangler left their slots null, and a host launch through a null slot returned
// InvalidValue from launchKernel BEFORE any trace line -- so an unmapped kernel on a
// live scene path was skipped in silence: most engine call sites check the result
// only under a per-module debug macro that a Release build compiles out.  A
// 33-scene capture found ZERO launch-failure lines while static analysis showed
// the FEM/particle -> rigid impulse accumulation and two articulation body-update
// stages being launched every step.  With the name present, moduleGetFunction is
// asked for it and hands back a distinguishable non-null SENTINEL handle (still
// reporting NotFound, so the wrangler's error path is unchanged); launchKernel
// still refuses it, but can now NAME and COUNT it.  See PxgVulkanCudaContext.cpp.
//
// This TU is shared by the PhysXGpuVulkan drop-in and the harness so the real
// KernelWrangler links the same table in both.

#include "foundation/PxPreprocessor.h"

#if PX_SUPPORT_GPU_PHYSX

#include "PxPhysXGpu.h"                 // the PX_C_EXPORT declarations of these symbols
#include "PxgSlangKernelManifest.h"     // gpuslang/include: generated enum-indexed registry manifest

#include <mutex>
#include <vector>

namespace
{
std::vector<physx::PxKernelIndex>& functionTable()
{
    static std::vector<physx::PxKernelIndex> t;
    return t;
}
std::mutex& tableMutex()
{
    static std::mutex m;
    return m;
}

void ensureFunctionTable()
{
    std::lock_guard<std::mutex> lock(tableMutex());
    std::vector<physx::PxKernelIndex>& t = functionTable();
    if (!t.empty())
        return;
    using namespace physx::gpuslang;
    for (unsigned i = 0; i < PXG_SLANG_KERNEL_COUNT; ++i)
    {
        const PxgSlangKernelRow& row = gPxgSlangKernelManifest[i];
        if (row.cudaName)   // every registry name, mapped or not (P9.40)
        {
            physx::PxKernelIndex entry;
            entry.moduleIndex = 0u;             // single sentinel module
            entry.functionName = row.cudaName;  // the registry key the wrangler matches
            t.push_back(entry);
        }
    }
}
} // anonymous namespace

physx::PxKernelIndex* PxGpuGetCudaFunctionTable()
{
    ensureFunctionTable();
    return functionTable().data();
}

physx::PxU32 PxGpuGetCudaFunctionTableSize()
{
    ensureFunctionTable();
    return physx::PxU32(functionTable().size());
}

// No CUmodule table in a SPIR-V world (the manager's getCuModules() returns a
// one-element sentinel array the wrangler indexes at moduleIndex 0).
void** PxGpuGetCudaModuleTable()
{
    return NULL;
}

physx::PxU32 PxGpuGetCudaModuleTableSize()
{
    return 0;
}

#endif // PX_SUPPORT_GPU_PHYSX
