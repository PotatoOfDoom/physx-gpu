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

}  // namespace
