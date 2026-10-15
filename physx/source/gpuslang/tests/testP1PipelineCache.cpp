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

// RUNTIME_PLAN.md P1.5 test: on-disk VkPipelineCache.  A first device with a
// cache path builds pipelines and, at destruction, writes a valid cache blob;
// a second device seeded from that blob builds the same pipelines and produces
// identical kernel results.  The blob's Vulkan pipeline-cache header is
// checked directly so a bogus write can't masquerade as a hit.

#include "pxgTestFramework.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace
{
using namespace pxgtest;

static PxgSlangComputeDevice* createDevice(const char* cachePath)
{
    PxgSlangDeviceDesc desc;
    const char* preferSw = std::getenv("PXG_PREFER_SOFTWARE_DEVICE");
    desc.preferSoftwareDevice = preferSw == nullptr || std::strcmp(preferSw, "0") != 0;
    desc.pipelineCachePath = cachePath;
    return physx::gpuslang::PxgCreateVulkanComputeDevice(desc);
}

// Builds a pipeline (the operation the cache accelerates) and runs it, so two
// devices sharing a cache can be compared bitwise.  Uses the self-contained
// subgroup spike: 1 storage buffer, deterministic per-lane output.
static std::vector<uint32_t> runKernel(PxgSlangComputeDevice* dev)
{
    const uint32_t N = 1024;
    const uint32_t GROUPS = N / 64;
    struct SgOut { uint32_t x, y, z, w; };
    struct P { uint32_t numThreads; } prm = { N };

    std::vector<SgOut> init(N);
    PxgBufferHandle bOut = createBufferFrom(dev, init);

    const std::vector<uint8_t> blob = loadShaderBlob("npSpikeSubgroup.spv");
    const uint32_t stride = uint32_t(sizeof(SgOut));
    PxgPipelineHandle pipe = dev->createPipeline(blob.data(), blob.size(), "spikeSubgroup",
                                                 1, &stride, sizeof(P));
    dev->dispatch(pipe, GROUPS, 1, 1, &bOut, 1, &prm, sizeof(P));
    dev->flush();
    const auto g = downloadBuffer<SgOut>(dev, bOut, N);
    dev->destroyPipeline(pipe);
    dev->destroyBuffer(bOut);

    std::vector<uint32_t> out(N);
    for (uint32_t i = 0; i < N; ++i)
        out[i] = g[i].x ^ g[i].y ^ g[i].z ^ g[i].w;
    return out;
}

PXG_TEST(p1PipelineCache)
{
    std::string path = "pxg_pipeline_cache.bin";
    if (const char* dir = std::getenv("PXG_SHADER_DIR"))
        path = std::string(dir) + "/pxg_pipeline_cache.bin";
    std::remove(path.c_str());

    // First device: no seed, builds pipelines, writes the cache on destroy.
    std::vector<uint32_t> firstOut;
    {
        std::unique_ptr<PxgSlangComputeDevice> dev(createDevice(path.c_str()));
        PXG_CHECK_MSG(dev != nullptr, "device with a cache path is created");
        if (!dev)
            return;
        firstOut = runKernel(dev.get());
    }  // destructor writes the cache

    // The written blob exists and carries a valid pipeline-cache header:
    //   uint32 length, uint32 headerVersion==1, uint32 vendorID, uint32 deviceID.
    std::vector<uint8_t> blob;
    if (FILE* f = std::fopen(path.c_str(), "rb"))
    {
        std::fseek(f, 0, SEEK_END);
        const long sz = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        if (sz > 0)
        {
            blob.resize(size_t(sz));
            if (std::fread(blob.data(), 1, size_t(sz), f) != size_t(sz))
                blob.clear();
        }
        std::fclose(f);
    }
    PXG_CHECK_MSG(blob.size() >= 32, "pipeline cache blob was written and is header-sized");
    if (blob.size() >= 8)
    {
        uint32_t headerLen = 0, headerVersion = 0;
        std::memcpy(&headerLen, &blob[0], 4);
        std::memcpy(&headerVersion, &blob[4], 4);
        PXG_CHECK_MSG(headerLen >= 16 && headerLen <= blob.size(), "cache header length is sane");
        PXG_CHECK_MSG(headerVersion == 1, "cache header version is VkPipelineCacheHeaderVersionOne");
    }

    // Second device: seeded from the blob; identical kernel results.
    {
        std::unique_ptr<PxgSlangComputeDevice> dev(createDevice(path.c_str()));
        PXG_CHECK_MSG(dev != nullptr, "device seeded from the cache blob is created");
        if (dev)
        {
            const auto secondOut = runKernel(dev.get());
            PXG_CHECK_MSG(secondOut.size() == firstOut.size() &&
                              std::memcmp(secondOut.data(), firstOut.data(), firstOut.size() * 4) == 0,
                          "cache-seeded device produces identical kernel results");
        }
    }

    // A garbage blob must NOT crash device creation (driver header check
    // rejects it and starts an empty cache).
    if (FILE* f = std::fopen(path.c_str(), "wb"))
    {
        const char junk[64] = { 'n', 'o', 't', 'a', 'c', 'a', 'c', 'h', 'e' };
        std::fwrite(junk, 1, sizeof(junk), f);
        std::fclose(f);
    }
    {
        std::unique_ptr<PxgSlangComputeDevice> dev(createDevice(path.c_str()));
        PXG_CHECK_MSG(dev != nullptr, "device tolerates a corrupt cache blob");
        if (dev)
        {
            const auto out = runKernel(dev.get());
            PXG_CHECK_MSG(out.size() == firstOut.size() &&
                              std::memcmp(out.data(), firstOut.data(), firstOut.size() * 4) == 0,
                          "results correct even after a rejected cache blob");
        }
    }

    std::remove(path.c_str());
}


// The in-memory form the engine bridge uses (it reads and writes the file itself,
// UTF-8 paths on Windows): getPipelineCacheData returns a blob with this device's
// header, a device seeded from it through desc.pipelineCacheData produces identical
// results, and a blob whose header names another device is ignored rather than
// handed to the driver.
PXG_TEST(p1PipelineCacheMemory)
{
    std::vector<uint8_t> blob;
    std::vector<uint32_t> firstOut;
    {
        std::unique_ptr<PxgSlangComputeDevice> dev(createDevice(nullptr));
        PXG_CHECK_MSG(dev != nullptr, "device without a cache path is created");
        if (!dev)
            return;
        firstOut = runKernel(dev.get());
        PXG_CHECK_MSG(dev->getPipelineCacheData(&blob), "getPipelineCacheData returns the cache");
    }
    PXG_CHECK_MSG(blob.size() >= 32, "the cache blob is at least header-sized (%zu B)", blob.size());
    if (blob.size() < 32)
        return;
    uint32_t hdr[4];
    std::memcpy(hdr, blob.data(), sizeof(hdr));
    PXG_CHECK_MSG(hdr[0] >= 32 && hdr[1] == 1, "VkPipelineCacheHeaderVersionOne (length %u, version %u)", hdr[0], hdr[1]);

    auto seeded = [](const std::vector<uint8_t>& seed) {
        PxgSlangDeviceDesc desc;
        const char* preferSw = std::getenv("PXG_PREFER_SOFTWARE_DEVICE");
        desc.preferSoftwareDevice = preferSw == nullptr || std::strcmp(preferSw, "0") != 0;
        desc.pipelineCacheData = seed.data();
        desc.pipelineCacheDataSize = seed.size();
        return std::unique_ptr<PxgSlangComputeDevice>(physx::gpuslang::PxgCreateVulkanComputeDevice(desc));
    };
    {
        std::unique_ptr<PxgSlangComputeDevice> dev = seeded(blob);
        PXG_CHECK_MSG(dev != nullptr, "device seeded from memory is created");
        if (dev)
        {
            const auto out = runKernel(dev.get());
            PXG_CHECK_MSG(out == firstOut, "memory-seeded device produces identical results");
            std::vector<uint8_t> again;
            PXG_CHECK_MSG(dev->getPipelineCacheData(&again) && again.size() >= 32 &&
                              std::memcmp(again.data(), blob.data(), 32) == 0,
                          "the seeded device's cache carries the same header");
        }
    }
    {
        // Another device's blob: vendor id changed.  Ignored, not fed to the driver.
        std::vector<uint8_t> foreign = blob;
        foreign[8] ^= 0x5a;
        std::unique_ptr<PxgSlangComputeDevice> dev = seeded(foreign);
        PXG_CHECK_MSG(dev != nullptr, "device tolerates a foreign cache blob");
        if (dev)
            PXG_CHECK_MSG(runKernel(dev.get()) == firstOut, "results correct after a foreign blob");
    }
}

// Pipelines built on several threads at once while the calling thread records and
// submits dispatches (supportsConcurrentPipelineCreation, the shim's warm-up): every
// build yields a distinct live pipeline, and they all compute the same result.
PXG_TEST(p1PipelineCacheConcurrent)
{
    PxgSlangComputeDevice* dev = testDevice();
    if (!dev->supportsConcurrentPipelineCreation())
    {
        std::printf("  NOTE: backend does not build pipelines concurrently -> skip\n");
        return;
    }
    const std::vector<uint8_t> blob = loadShaderBlob("npSpikeSubgroup.spv");
    struct SgOut { uint32_t x, y, z, w; };
    struct P { uint32_t numThreads; } prm = { 1024 };
    const uint32_t stride = uint32_t(sizeof(SgOut));
    const unsigned kThreads = 6, kPerThread = 4;
    std::vector<PxgPipelineHandle> handles(kThreads * kPerThread, 0);
    std::vector<std::thread> threads;
    for (unsigned t = 0; t < kThreads; ++t)
        threads.emplace_back([&, t] {
            for (unsigned k = 0; k < kPerThread; ++k)
                handles[t * kPerThread + k] =
                    dev->createPipeline(blob.data(), blob.size(), "spikeSubgroup", 1, &stride, sizeof(P));
        });
    // Meanwhile: dispatches on this thread through a pipeline of its own.
    std::vector<SgOut> init(1024);
    PxgBufferHandle bOut = createBufferFrom(dev, init);
    PxgPipelineHandle own = dev->createPipeline(blob.data(), blob.size(), "spikeSubgroup", 1, &stride, sizeof(P));
    for (int i = 0; i < 20; ++i)
        dev->dispatch(own, 16, 1, 1, &bOut, 1, &prm, sizeof(P));
    dev->flush();
    for (std::thread& th : threads)
        th.join();
    const auto ref = downloadBuffer<SgOut>(dev, bOut, 1024);

    bool allLive = true, distinct = true;
    for (size_t i = 0; i < handles.size(); ++i)
    {
        allLive = allLive && handles[i] != 0;
        for (size_t j = 0; j < i; ++j)
            distinct = distinct && handles[i] != handles[j];
    }
    PXG_CHECK_MSG(allLive, "every concurrently built pipeline is live");
    PXG_CHECK_MSG(distinct, "concurrently built pipelines got distinct handles");
    size_t bad = 0;
    for (PxgPipelineHandle h : handles)
    {
        if (!h)
            continue;
        std::vector<SgOut> zero(1024);
        PxgBufferHandle b = createBufferFrom(dev, zero);
        dev->dispatch(h, 16, 1, 1, &b, 1, &prm, sizeof(P));
        dev->flush();
        const auto got = downloadBuffer<SgOut>(dev, b, 1024);
        if (std::memcmp(got.data(), ref.data(), ref.size() * sizeof(SgOut)) != 0)
            ++bad;
        dev->destroyBuffer(b);
        dev->destroyPipeline(h);
    }
    PXG_CHECK_MSG(bad == 0, "%zu concurrently built pipeline(s) computed a different result", bad);
    dev->destroyPipeline(own);
    dev->destroyBuffer(bOut);
}

}  // namespace
