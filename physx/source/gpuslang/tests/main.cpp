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

#include "pxgTestFramework.h"

namespace pxgtest
{

int gFailureCount = 0;

std::vector<TestCase>& testRegistry()
{
    static std::vector<TestCase> registry;
    return registry;
}

PxgSlangComputeDevice* testDevice()
{
    static PxgSlangComputeDevice* device = nullptr;
    static bool attempted = false;
    if (!attempted)
    {
        attempted = true;
        PxgSlangDeviceDesc desc;
        // The verification suite runs on CPU Vulkan devices (llvmpipe) in CI
        // containers; on developer machines a real GPU is picked when
        // PXG_PREFER_SOFTWARE_DEVICE is not set.
        const char* preferSw = std::getenv("PXG_PREFER_SOFTWARE_DEVICE");
        desc.preferSoftwareDevice = preferSw == nullptr || std::strcmp(preferSw, "0") != 0;
        // v2 device mode (RUNTIME_PLAN.md P1): device-local buffers + staged
        // transfers; lets the whole bit-exact suite re-run through the
        // production memory path.
        const char* v2 = std::getenv("PXG_DEVICE_V2");
        desc.useDeviceLocalMemory = v2 != nullptr && std::strcmp(v2, "0") != 0;
#if defined(_WIN32)
        const char* backend = std::getenv("PXG_TEST_BACKEND");
        if (backend && std::strcmp(backend, "dx12") == 0)
            device = physx::gpuslang::PxgCreateDx12ComputeDevice(desc);
        else
            device = physx::gpuslang::PxgCreateVulkanComputeDevice(desc);
#else
        device = physx::gpuslang::PxgCreateVulkanComputeDevice(desc);
#endif
        if (device)
            std::printf("[gpuslang-tests] backend: %s, device: %s%s\n", device->getBackendName(),
                        device->getDeviceName(),
                        desc.useDeviceLocalMemory ? " [v2: device-local + staged]" : "");
    }
    return device;
}

std::vector<uint8_t> loadShaderBlob(const char* fileName)
{
    const char* dir = std::getenv("PXG_SHADER_DIR");
    std::string path = dir ? std::string(dir) + "/" + fileName : std::string(fileName);

    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f)
    {
        std::printf("FATAL: cannot open shader blob '%s'\n", path.c_str());
        std::exit(2);
    }
    std::fseek(f, 0, SEEK_END);
    const long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> blob(static_cast<size_t>(size), 0);
    const size_t read = std::fread(blob.data(), 1, size_t(size), f);
    std::fclose(f);
    if (read != size_t(size))
    {
        std::printf("FATAL: short read on shader blob '%s'\n", path.c_str());
        std::exit(2);
    }
    return blob;
}

} // namespace pxgtest

int main(int argc, char** argv)
{
    const char* filter = argc > 1 ? argv[1] : nullptr;

    if (!pxgtest::testDevice())
    {
        std::printf("FATAL: no compute device available (Vulkan loader/ICD missing?)\n");
        return 2;
    }

    int ran = 0;
    int failedTests = 0;
    for (const pxgtest::TestCase& test : pxgtest::testRegistry())
    {
        if (filter && std::strstr(test.name, filter) == nullptr)
            continue;

        std::printf("[ RUN  ] %s\n", test.name);
        const int failuresBefore = pxgtest::gFailureCount;
        test.fn();
        ++ran;
        if (pxgtest::gFailureCount != failuresBefore)
        {
            ++failedTests;
            std::printf("[ FAIL ] %s\n", test.name);
        }
        else
        {
            std::printf("[  OK  ] %s\n", test.name);
        }
    }

    std::printf("\n%d test(s) ran, %d failed, %d check failure(s)\n", ran, failedTests,
                pxgtest::gFailureCount);
    return pxgtest::gFailureCount == 0 && ran > 0 ? 0 : 1;
}
