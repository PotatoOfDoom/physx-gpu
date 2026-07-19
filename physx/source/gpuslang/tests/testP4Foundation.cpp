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

// RUNTIME_PLAN.md P4.1 test: the engine-build keystone.  It proves the REAL
// PhysX foundation library builds and RUNS in the hand-rolled harness (no
// packman, no CUDA toolkit) — i.e. PxCreateFoundation stands up the global
// allocator + error callback, and PX_NEW and the engine containers work.  This
// is the prerequisite for linking the real gpu-module engine code (the buddy
// heap allocator, the kernel wrangler, and eventually the Pxg* subsystems)
// against the Vulkan bridge — the P4 shift from isolated harness kernels to
// full-engine integration.

#include "pxgTestFramework.h"

#include "foundation/PxFoundation.h"
#include "foundation/PxAllocatorCallback.h"
#include "foundation/PxErrorCallback.h"
#include "foundation/PxUserAllocated.h"
#include "foundation/PxArray.h"
#include "foundation/PxPhysicsVersion.h"

#include <cstdio>
#include <cstdlib>

namespace
{
using namespace pxgtest;
using namespace physx;

struct HarnessAllocator : PxAllocatorCallback
{
    void* allocate(size_t size, const char*, const char*, int) override
    {
        void* p = nullptr;
        if (posix_memalign(&p, 16, size ? size : 1))
            return nullptr;
        return p;
    }
    void deallocate(void* p) override { free(p); }
};

struct HarnessError : PxErrorCallback
{
    void reportError(PxErrorCode::Enum, const char*, const char*, int) override {}
};

struct Widget : PxUserAllocated { PxU32 v[16]; };

PXG_TEST(p4Foundation)
{
    // alloc/err are referenced by the foundation, so they must outlive its
    // release() — declared here, destroyed only after release() returns.
    HarnessAllocator alloc;
    HarnessError err;

    PxFoundation* foundation = PxCreateFoundation(PX_PHYSICS_VERSION, alloc, err);
    PXG_CHECK_MSG(foundation != nullptr, "PxCreateFoundation succeeds in the harness (no packman, no CUDA)");
    if (!foundation)
        return;

    // Everything allocated through the foundation's global allocator lives in
    // this inner scope, so it is destroyed BEFORE the foundation is released
    // (freeing a container through a torn-down allocator would crash).
    {
        Widget* w = PX_NEW(Widget)();
        PXG_CHECK_MSG(w != nullptr, "PX_NEW allocates through the foundation global allocator");
        if (w)
        {
            w->v[0] = 0xC0FFEEu;
            PXG_CHECK_MSG(w->v[0] == 0xC0FFEEu, "PX_NEW'd storage is writable");
            PX_DELETE(w);
        }

        PxArray<PxU32> arr;               // an engine container on the engine allocator
        for (PxU32 i = 0; i < 1000; ++i)
            arr.pushBack(i * 3u + 1u);
        size_t bad = 0;
        for (PxU32 i = 0; i < arr.size(); ++i)
            if (arr[i] != i * 3u + 1u)
                ++bad;
        PXG_CHECK_MSG(arr.size() == 1000 && bad == 0,
                      "PxArray grows/reads correctly on the engine allocator (%zu bad)", bad);
    }

    foundation->release();
    std::printf("  PhysX foundation runtime is live in the harness — engine code can now link (P4)\n");
}

}  // namespace
