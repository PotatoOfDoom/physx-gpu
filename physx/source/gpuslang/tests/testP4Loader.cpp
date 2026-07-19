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

// RUNTIME_PLAN.md P3.5 test: load the PhysXGpuVulkan C-ABI module the way the
// real PhysX core does — dlopen the shared object, dlsym the unmangled C
// symbols, and enforce the SAME mandatory-symbol compatibility gate the engine
// applies (PxPhysXGpuModuleLoader.cpp: reject unless PxCreatePhysXGpu,
// PxCreateCudaContextManager, PxGetSuggestedCudaDeviceOrdinal all resolve
// non-null).  Then it drives the two foundation-free exports end-to-end:
// PxGetSuggestedCudaDeviceOrdinal (which stands up the Vulkan device + shim +
// manifest inside the module) and PxCreatePhysXGpu (the subsystem-factory
// object).  PxCreateCudaContextManager needs a live PxFoundation, so it is only
// verified to resolve as a symbol here — its device/shim/manager construction is
// exercised in-process by p3CudaContextManager.

#include "pxgTestFramework.h"

#include "foundation/PxPreprocessor.h"

#if PX_SUPPORT_GPU_PHYSX && defined(PXG_PHYSXGPUVULKAN_SO) && !defined(_WIN32)

#include "foundation/PxErrorCallback.h"
#include "cudamanager/PxCudaContextManager.h"
#include "PxPhysXGpu.h"   // physx/source/physxgpu/include: the PxPhysXGpu interface

#include <cstdio>
#include <dlfcn.h>

namespace
{
using namespace pxgtest;

// The exact unmangled C symbols the PhysX module loader resolves.  The three
// marked MANDATORY are the ones its Linux compatibility check null-tests.
static const char* kMandatorySymbols[] = {
    "PxCreatePhysXGpu",
    "PxCreateCudaContextManager",
    "PxGetSuggestedCudaDeviceOrdinal",
};
static const char* kSecondarySymbols[] = {
    "PxSetPhysXGpuProfilerCallback",
    "PxSetPhysXGpuFoundationInstance",
    "PxGpuCudaRegisterFunction",
    "PxGpuCudaRegisterFatBinary",
    "PxGpuGetCudaFunctionTable",
    "PxGpuGetCudaFunctionTableSize",
    "PxGpuGetCudaModuleTable",
    "PxGpuGetCudaModuleTableSize",
    "PxGpuCreatePhysicsGpu",
};

struct SilentError : physx::PxErrorCallback
{
    void reportError(physx::PxErrorCode::Enum, const char*, const char*, int) override {}
};

typedef physx::PxPhysXGpu* (*CreatePhysXGpuFn)();
typedef int (*GetOrdinalFn)(physx::PxErrorCallback&);

PXG_TEST(p4Loader)
{
    // dlopen exactly as the engine does (RTLD_NOW resolves all symbols eagerly).
    void* lib = dlopen(PXG_PHYSXGPUVULKAN_SO, RTLD_NOW | RTLD_LOCAL);
    PXG_CHECK_MSG(lib != nullptr, "dlopen(%s) succeeds: %s", PXG_PHYSXGPUVULKAN_SO,
                  lib ? "" : dlerror());
    if (!lib)
        return;

    // The engine's compatibility gate: all three mandatory symbols must resolve.
    bool allMandatory = true;
    for (const char* sym : kMandatorySymbols)
    {
        void* p = dlsym(lib, sym);
        PXG_CHECK_MSG(p != nullptr, "mandatory symbol %s resolves", sym);
        allMandatory = allMandatory && (p != nullptr);
    }
    PXG_CHECK_MSG(allMandatory, "module passes the loader compatibility gate (all 3 mandatory symbols present)");

    // A full drop-in also provides the symbols the loader additionally dlsym's.
    for (const char* sym : kSecondarySymbols)
        PXG_CHECK_MSG(dlsym(lib, sym) != nullptr, "loader symbol %s resolves", sym);

    // Drive PxGetSuggestedCudaDeviceOrdinal: this stands up the Vulkan device +
    // shim + enum-indexed manifest inside the module, so a 0 return proves the
    // module's runtime bring-up (device creation + manifest conversion) works.
    GetOrdinalFn getOrdinal = reinterpret_cast<GetOrdinalFn>(dlsym(lib, "PxGetSuggestedCudaDeviceOrdinal"));
    PXG_CHECK_MSG(getOrdinal != nullptr, "PxGetSuggestedCudaDeviceOrdinal is callable");
    if (getOrdinal)
    {
        SilentError err;
        const int ordinal = getOrdinal(err);
        PXG_CHECK_MSG(ordinal == 0, "suggested device ordinal is 0 (Vulkan device came up), got %d", ordinal);
    }

    // PxCreatePhysXGpu must return a valid (non-null) PxPhysXGpu — the loader
    // caches whatever it returns; a null would crash the engine at GPU setup.
    CreatePhysXGpuFn createGpu = reinterpret_cast<CreatePhysXGpuFn>(dlsym(lib, "PxCreatePhysXGpu"));
    PXG_CHECK_MSG(createGpu != nullptr, "PxCreatePhysXGpu is callable");
    if (createGpu)
    {
        physx::PxPhysXGpu* gpu = createGpu();
        PXG_CHECK_MSG(gpu != nullptr, "PxCreatePhysXGpu returns a non-null PxPhysXGpu");
        if (gpu)
            gpu->release();  // singleton: no-op, must not crash
    }

    // PxCreateCudaContextManager needs a live PxFoundation to invoke; here we
    // only confirm the symbol is exported (its manager construction path is
    // covered in-process by p3CudaContextManager).
    PXG_CHECK_MSG(dlsym(lib, "PxCreateCudaContextManager") != nullptr,
                  "PxCreateCudaContextManager is exported");

    dlclose(lib);
    std::printf("  PhysXGpuVulkan module loaded and passed the C-ABI compatibility gate\n");
}

}  // namespace

#else  // requirements not met

#include "pxgTestFramework.h"
namespace { PXG_TEST(p4Loader) { std::printf("  NOTE: PhysXGpuVulkan dlopen test not applicable in this config -> skipped\n"); } }

#endif
