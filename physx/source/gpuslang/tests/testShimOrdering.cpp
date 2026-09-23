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

// testShimOrdering.cpp -- P9.85: are two engine launches on one stream ORDERED?
//
// CUDA runs kernels on one stream in order, and every engine kernel that reads
// what the previous one wrote depends on it.  The device interface states the
// Vulkan equivalent (PxgSlangComputeDevice.h): "dispatch() records work;
// barrier() orders successive dispatches".  PxgCudaShim::launchKernel is the
// engine's only launch path, and before P9.85 it never called barrier() --
// nothing in the shim or the bridge did; only the fixtures, which drive the
// device directly, call it themselves.
//
// WHAT CAN AND CANNOT BE SEEN HERE.  llvmpipe executes the dispatches of a
// command buffer one after the other, so a missing barrier never produces a wrong
// VALUE on this device -- the second kernel below computes the right answer with
// or without it, and the value check is there only so the test is not vacuous.
// Sync validation cannot see it either: it tracks descriptor-bound resources,
// and engine kernels touch memory only through buffer device addresses.  What
// CAN be measured is the command stream itself: the device counts dispatches
// recorded while the previous dispatch of the same batch has no barrier after it
// (unorderedDispatchCount).  That count is the defect, stated as a number.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgResult;

static std::vector<uint8_t> orderingLoader(const char* file) { return loadShaderBlob(file); }

struct AxpyBuffers { uint64_t x, y, out; };
struct AxpyParams { AxpyBuffers bufs; float alpha; uint32_t count; };
} // namespace

PXG_TEST(shimOrdering_backToBack)
{
    PxgSlangComputeDevice* dev = testDevice();

    const PxgKernelDef manifest[] = {
        { "npShimAxpy.spv", "shimAxpy", uint32_t(sizeof(AxpyParams)), 0u },
    };
    PxgCudaShim shim(dev, manifest, 1, &orderingLoader);

    const uint32_t N = 4096;
    std::mt19937 rng(0x0D0E5u);
    std::vector<float> x(N), y(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        x[i] = float(int(rng() % 2001) - 1000) * 0.25f;   // exact quarter-integers
        y[i] = float(int(rng() % 2001) - 1000) * 0.5f;    // exact half-integers
    }

    PxgDevicePtr dx = 0, dy = 0, d1 = 0, d2 = 0;
    for (PxgDevicePtr* p : { &dx, &dy, &d1, &d2 })
        PXG_CHECK_MSG(shim.memAlloc(p, N * 4) == PxgResult::Success, "memAlloc");
    if (!dx || !dy || !d1 || !d2)
        return;
    shim.memcpyHtoD(dx, x.data(), N * 4);
    shim.memcpyHtoD(dy, y.data(), N * 4);
    shim.memsetD32(d1, 0xDEADBEEFu, N);
    shim.memsetD32(d2, 0xDEADBEEFu, N);
    dev->flush();   // start the measured batch empty

    // Launch B reads what launch A wrote: out1 = 2x + y, then out2 = 3*out1 + y.
    const uint64_t before = dev->unorderedDispatchCount();
    AxpyParams a{ { dx, dy, d1 }, 2.0f, N };
    AxpyParams b{ { d1, dy, d2 }, 3.0f, N };
    PXG_CHECK_MSG(shim.launchKernel(0, (N + 63) / 64, 1, 1, &a, sizeof(a)) == PxgResult::Success, "launch A");
    PXG_CHECK_MSG(shim.launchKernel(0, (N + 63) / 64, 1, 1, &b, sizeof(b)) == PxgResult::Success, "launch B");
    const uint64_t unordered = dev->unorderedDispatchCount() - before;
    shim.streamSynchronize();

    std::vector<float> got(N);
    shim.memcpyDtoH(got.data(), d2, N * 4);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float want = 3.0f * (2.0f * x[i] + y[i]) + y[i];   // exact on this grid
        uint32_t gu, wu;
        std::memcpy(&gu, &got[i], 4);
        std::memcpy(&wu, &want, 4);
        if (gu != wu)
            ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "shimOrdering: %zu of %u chained results wrong", mm, N);

    PXG_CHECK_MSG(unordered == 0,
                  "shimOrdering: launch B was recorded with NO compute->compute barrier after "
                  "launch A (%llu unordered dispatch(es)) -- on a device that overlaps dispatches, "
                  "B may read A's output before A has written it",
                  (unsigned long long)unordered);

    std::printf("  shimOrdering_backToBack: two dependent launches, %llu unordered, %zu wrong values\n",
                (unsigned long long)unordered, mm);

    for (PxgDevicePtr p : { dx, dy, d1, d2 })
        shim.memFree(p);
}

// ---------------------------------------------------------------------------
// The positive control: the counter itself, on the device directly.
// ---------------------------------------------------------------------------
// shimOrdering_backToBack asserts the count is ZERO, and a counter that never
// counts passes that too -- the sweep's `count-never-increments` mutant said so.
// So the counter is exercised where the answer is known and not zero: two
// dispatches with no barrier between them are ONE unordered pair; a barrier in
// between makes it none; and a submit boundary makes it none (each batch waits on
// the one before it).
PXG_TEST(shimOrdering_counter)
{
    PxgSlangComputeDevice* dev = testDevice();

    const uint32_t N = 64;
    std::vector<float> ones(N, 1.0f);
    const PxgBufferHandle bx = createBufferFrom(dev, ones);
    const PxgBufferHandle by = createBufferFrom(dev, ones);
    const PxgBufferHandle bo = createBufferFrom(dev, ones);
    AxpyParams p{ { dev->getBufferDeviceAddress(bx), dev->getBufferDeviceAddress(by),
                    dev->getBufferDeviceAddress(bo) }, 1.0f, N };

    Kernel k("npShimAxpy.spv", "shimAxpy", {}, sizeof(AxpyParams));
    PXG_CHECK_MSG(k.isValid(), "shimAxpy pipeline");
    if (!k.isValid())
        return;

    dev->flush();
    const uint64_t c0 = dev->unorderedDispatchCount();
    k.dispatch(1, {}, p);
    k.dispatch(1, {}, p);                  // no barrier between: one unordered pair
    const uint64_t c1 = dev->unorderedDispatchCount();
    dev->barrier();
    k.dispatch(1, {}, p);                  // ordered by the barrier
    const uint64_t c2 = dev->unorderedDispatchCount();
    dev->flush();
    k.dispatch(1, {}, p);                  // first of a new batch: ordered by the submit
    const uint64_t c3 = dev->unorderedDispatchCount();
    dev->flush();

    PXG_CHECK_MSG(c1 - c0 == 1, "shimOrdering_counter: two dispatches with no barrier counted %llu, want 1",
                  (unsigned long long)(c1 - c0));
    PXG_CHECK_MSG(c2 - c1 == 0, "shimOrdering_counter: a dispatch after barrier() counted %llu, want 0",
                  (unsigned long long)(c2 - c1));
    PXG_CHECK_MSG(c3 - c2 == 0, "shimOrdering_counter: the first dispatch of a new batch counted %llu, want 0",
                  (unsigned long long)(c3 - c2));
    std::printf("  shimOrdering_counter: +%llu without a barrier, +%llu after one, +%llu across a submit\n",
                (unsigned long long)(c1 - c0), (unsigned long long)(c2 - c1), (unsigned long long)(c3 - c2));

    for (PxgBufferHandle h : { bx, by, bo })
        dev->destroyBuffer(h);
}
