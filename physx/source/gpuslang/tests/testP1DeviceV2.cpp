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

// RUNTIME_PLAN.md P1 test: the v2 device paths.
//   - device-local buffers with staged transfers (P1.2): bitwise round-trips
//     that cross the 8 MiB staging ring several times, offset semantics, and
//     a real kernel dispatch reading/writing device-local storage;
//   - timeline submission (P1.1): submit()/wait()/poll() ticket semantics.
// Runs a dedicated device with useDeviceLocalMemory = true regardless of the
// suite-wide PXG_DEVICE_V2 mode, so the staged path is covered on every run.

#include "pxgTestFramework.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

namespace
{
using namespace pxgtest;

static PxgSlangComputeDevice* createV2Device()
{
    PxgSlangDeviceDesc desc;
    const char* preferSw = std::getenv("PXG_PREFER_SOFTWARE_DEVICE");
    desc.preferSoftwareDevice = preferSw == nullptr || std::strcmp(preferSw, "0") != 0;
    desc.useDeviceLocalMemory = true;
    return physx::gpuslang::PxgCreateVulkanComputeDevice(desc);
}

PXG_TEST(p1DeviceV2)
{
    std::unique_ptr<PxgSlangComputeDevice> dev(createV2Device());
    PXG_CHECK_MSG(dev != nullptr, "v2 (device-local) device creation succeeds");
    if (!dev)
        return;

    // --- staged round-trip crossing the staging ring (20 MiB > 2x 8 MiB) ---
    {
        const size_t N = 20u * 1024u * 1024u / 4u;
        std::vector<uint32_t> src(N);
        for (size_t i = 0; i < N; ++i)
            src[i] = uint32_t(i * 2654435761u ^ 0x9e3779b9u);

        PxgBufferHandle b = dev->createBuffer(N * 4);
        dev->upload(b, src.data(), N * 4);
        std::vector<uint32_t> got(N, 0);
        dev->download(b, got.data(), N * 4);
        PXG_CHECK_MSG(std::memcmp(src.data(), got.data(), N * 4) == 0,
                      "20 MiB staged upload/download round-trip is bitwise exact");
        dev->destroyBuffer(b);
    }

    // --- offset semantics on the staged path ---
    {
        const size_t N = 4096;
        std::vector<uint32_t> zero(N, 0), patch(64);
        for (size_t i = 0; i < patch.size(); ++i)
            patch[i] = 0xa5000000u + uint32_t(i);

        PxgBufferHandle b = dev->createBuffer(N * 4);
        dev->upload(b, zero.data(), N * 4);
        dev->upload(b, patch.data(), patch.size() * 4, 1024 * 4);

        std::vector<uint32_t> got(patch.size() + 2, 0);
        dev->download(b, got.data(), got.size() * 4, (1024 - 1) * 4);
        PXG_CHECK_MSG(got.front() == 0 && got.back() == 0,
                      "staged offset upload leaves neighbors untouched");
        PXG_CHECK_MSG(std::memcmp(&got[1], patch.data(), patch.size() * 4) == 0,
                      "staged offset upload/download places bytes correctly");
        dev->destroyBuffer(b);
    }

    // --- device-side fill + copy (cuMemsetD32 / cuMemcpyDtoD parity) ---
    {
        const uint32_t N = 8192;
        std::vector<uint32_t> seed(N);
        for (uint32_t i = 0; i < N; ++i)
            seed[i] = i * 2246822519u + 1u;
        PxgBufferHandle a = createBufferFrom(dev.get(), seed);
        PxgBufferHandle b = dev->createBuffer(N * 4);

        // fillBuffer over a sub-range: [1024, 3072) set to a sentinel, ends
        // untouched.
        dev->fillBuffer(b, 0, 0u, N * 4);
        dev->fillBuffer(b, 1024 * 4, 0xdeadbeefu, 2048 * 4);
        std::vector<uint32_t> fb(N);
        dev->download(b, fb.data(), N * 4);
        size_t fbad = 0;
        for (uint32_t i = 0; i < N; ++i)
        {
            const uint32_t want = (i >= 1024 && i < 1024 + 2048) ? 0xdeadbeefu : 0u;
            if (fb[i] != want)
                ++fbad;
        }
        PXG_CHECK_MSG(fbad == 0, "fillBuffer sets exactly the requested sub-range (%zu bad)", fbad);

        // copyBuffer a sub-range of a into the middle of b, offsets differ.
        dev->fillBuffer(b, 0, 0u, N * 4);
        dev->copyBuffer(b, 512 * 4, a, 256 * 4, 4096 * 4);
        std::vector<uint32_t> cb(N);
        dev->download(b, cb.data(), N * 4);
        size_t cbad = 0;
        for (uint32_t i = 0; i < N; ++i)
        {
            const uint32_t want = (i >= 512 && i < 512 + 4096) ? seed[256 + (i - 512)] : 0u;
            if (cb[i] != want)
                ++cbad;
        }
        PXG_CHECK_MSG(cbad == 0, "copyBuffer moves exactly the requested range at both offsets (%zu bad)", cbad);

        dev->destroyBuffer(a);
        dev->destroyBuffer(b);
    }

    // --- dispatch through device-local storage + ticket semantics ---
    {
        const uint32_t N = 4096;
        const uint32_t GROUPS = N / 64;
        struct SgOut { uint32_t x, y, z, w; };
        struct P { uint32_t numThreads; } prm = { N };

        const std::vector<uint8_t> blob = loadShaderBlob("npSpikeSubgroup.spv");
        const uint32_t stride = uint32_t(sizeof(SgOut));
        const PxgPipelineHandle pipe = dev->createPipeline(blob.data(), blob.size(), "spikeSubgroup",
                                                           1, &stride, sizeof(P));
        PXG_CHECK_MSG(pipe != 0, "pipeline creation on the v2 device succeeds");

        std::vector<SgOut> init(N);
        PxgBufferHandle bOut = dev->createBuffer(N * sizeof(SgOut));
        dev->upload(bOut, init.data(), N * sizeof(SgOut));

        dev->dispatch(pipe, GROUPS, 1, 1, &bOut, 1, &prm, sizeof(P));
        const uint64_t t1 = dev->submit();
        if (t1 == 0)
        {
            std::printf("  NOTE: synchronous-submit backend (no timeline) -> ticket checks skipped\n");
        }
        else
        {
            dev->wait(t1);
            PXG_CHECK_MSG(dev->poll(t1), "poll() confirms a waited ticket has retired");
            dev->dispatch(pipe, GROUPS, 1, 1, &bOut, 1, &prm, sizeof(P));
            const uint64_t t2 = dev->submit();
            PXG_CHECK_MSG(t2 > t1, "tickets are monotonic");
            dev->wait(t2);
        }

        std::vector<SgOut> g(N);
        dev->download(bOut, g.data(), N * sizeof(SgOut));
        size_t bad = 0;
        for (size_t i = 0; i < N; ++i)
        {
            const uint32_t laneIdx = g[i].x & 0xffffu;
            const uint32_t laneCount = g[i].x >> 16;
            if (!(laneCount >= 4 && laneIdx < laneCount && g[i].y == laneIdx &&
                  g[i].z == laneCount && g[i].w == laneCount))
                ++bad;
        }
        PXG_CHECK_MSG(bad == 0, "kernel results through device-local buffers are correct");

        dev->destroyBuffer(bOut);
        dev->destroyPipeline(pipe);
    }

    // --- cross-submission ordering (single-stream contract) ---
    // Batch 1 computes d = a + b; batch 2 (submitted WITHOUT a host wait)
    // computes e = d + d.  The ordering contract alone must make batch 1's
    // results visible to batch 2, like back-to-back kernels on one CUDA
    // stream.
    {
        const uint32_t N = 4096;
        const uint32_t GROUPS = N / 64;
        std::vector<float> a(N), b(N), zero(N, 0.f);
        for (uint32_t i = 0; i < N; ++i)
        {
            a[i] = -3.f + 0.001f * float(i % 6007);
            b[i] = 2.5f - 0.0007f * float(i % 5003);
        }
        PxgBufferHandle bA = createBufferFrom(dev.get(), a);
        PxgBufferHandle bB = createBufferFrom(dev.get(), b);
        PxgBufferHandle bD = createBufferFrom(dev.get(), zero);
        PxgBufferHandle bE = createBufferFrom(dev.get(), zero);

        const uint64_t addrA = dev->getBufferDeviceAddress(bA);
        const uint64_t addrB = dev->getBufferDeviceAddress(bB);
        const uint64_t addrD = dev->getBufferDeviceAddress(bD);
        const uint64_t addrE = dev->getBufferDeviceAddress(bE);
        if (addrA && addrB && addrD && addrE)
        {
            struct BdaP { uint64_t srcA, srcB, dst; uint32_t count, pad; };
            const std::vector<uint8_t> blob = loadShaderBlob("npSpikeBda.spv");
            const PxgPipelineHandle pipe = dev->createPipeline(blob.data(), blob.size(), "spikeBda",
                                                               0, nullptr, sizeof(BdaP));
            const BdaP p1 = { addrA, addrB, addrD, N, 0 };
            dev->dispatch(pipe, GROUPS, 1, 1, nullptr, 0, &p1, sizeof(BdaP));
            dev->submit();
            const BdaP p2 = { addrD, addrD, addrE, N, 0 };
            dev->dispatch(pipe, GROUPS, 1, 1, nullptr, 0, &p2, sizeof(BdaP));
            dev->wait(dev->submit());

            std::vector<float> e(N, 0.f);
            dev->download(bE, e.data(), N * 4);
            size_t mm = 0;
            for (uint32_t i = 0; i < N; ++i)
            {
                const float d = a[i] + b[i];
                const float ref = d + d;
                uint32_t ur, ug;
                std::memcpy(&ur, &ref, 4);
                std::memcpy(&ug, &e[i], 4);
                if (ur != ug)
                    ++mm;
            }
            PXG_CHECK_MSG(mm == 0, "dependent dispatch across submit() sees prior batch results (%zu mismatches)", mm);
            dev->destroyPipeline(pipe);
        }
        else
        {
            std::printf("  NOTE: no buffer device addresses -> cross-submission chain check skipped\n");
        }

        dev->destroyBuffer(bA);
        dev->destroyBuffer(bB);
        dev->destroyBuffer(bD);
        dev->destroyBuffer(bE);
    }
}

}  // namespace
