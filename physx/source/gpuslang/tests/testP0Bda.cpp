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

// RUNTIME_PLAN.md P0 spike test: buffer-device-address argument marshalling.
// A CUDA-shaped parameter block carrying three raw device addresses travels
// through push constants; the dispatch binds ZERO descriptors; the kernel
// dereferences the addresses like CUDA device pointers.  This is the model
// the P3 PxCudaContext shim uses to launch all 487 registry kernels (PhysX 5.12) without
// per-kernel binding-table rewrites.

#include "pxgTestFramework.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

PXG_TEST(p0Bda)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t N = 4096;
    const uint32_t GROUPS = (N + 63) / 64;

    std::mt19937 rng(9001);
    auto rf = [&]() { return -4.f + 8.f * (float(rng() % 1000000) / 1000000.f); };

    std::vector<float> a(N), b(N), zero(N, 0.f);
    for (uint32_t i = 0; i < N; ++i) { a[i] = rf(); b[i] = rf(); }

    PxgBufferHandle bA = createBufferFrom(dev, a);
    PxgBufferHandle bB = createBufferFrom(dev, b);
    PxgBufferHandle bD = createBufferFrom(dev, zero);

    const uint64_t addrA = dev->getBufferDeviceAddress(bA);
    const uint64_t addrB = dev->getBufferDeviceAddress(bB);
    const uint64_t addrD = dev->getBufferDeviceAddress(bD);
    std::printf("  device addresses: A=0x%llx B=0x%llx D=0x%llx\n",
                (unsigned long long)addrA, (unsigned long long)addrB, (unsigned long long)addrD);

    if (addrA == 0 || addrB == 0 || addrD == 0)
    {
        std::printf("  NOTE: bufferDeviceAddress unavailable on this device/backend -> "
                    "the P3 shim would use the binding-table fallback here\n");
        dev->destroyBuffer(bA); dev->destroyBuffer(bB); dev->destroyBuffer(bD);
        return;
    }

    struct P
    {
        uint64_t srcA, srcB, dst;
        uint32_t count, pad;
    } prm = { addrA, addrB, addrD, N, 0 };

    Kernel k("npSpikeBda.spv", "spikeBda", {}, sizeof(P));
    k.dispatch(GROUPS, {}, prm);
    dev->flush();

    const auto g = downloadBuffer<float>(dev, bD, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float ref = a[i] + b[i];
        if (f2u(ref) != f2u(g[i]))
        {
            if (mm < 5)
                std::printf("  MM %u: ref %g got %g\n", i, ref, g[i]);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "BDA pointer kernel results bitwise-identical to CPU");

    dev->destroyBuffer(bA); dev->destroyBuffer(bB); dev->destroyBuffer(bD);
}

}  // namespace
