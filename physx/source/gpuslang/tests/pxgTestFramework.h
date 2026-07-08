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

#ifndef PXG_TESTS_FRAMEWORK_H
#define PXG_TESTS_FRAMEWORK_H

// Minimal test framework + device/shader helpers for the gpuslang
// verification suite.  Deliberately dependency-free (no gtest) so the suite
// builds anywhere the Vulkan SDK headers are available.

#include "PxgSlangComputeDevice.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace pxgtest
{

using physx::gpuslang::PxgBufferHandle;
using physx::gpuslang::PxgPipelineHandle;
using physx::gpuslang::PxgSlangComputeDevice;
using physx::gpuslang::PxgSlangDeviceDesc;

// ---------------------------------------------------------------------------
// Test registry
// ---------------------------------------------------------------------------
struct TestCase
{
    const char* name;
    std::function<void()> fn;
};

std::vector<TestCase>& testRegistry();

struct TestRegistrar
{
    TestRegistrar(const char* name, std::function<void()> fn)
    {
        testRegistry().push_back({ name, std::move(fn) });
    }
};

#define PXG_TEST(name)                                                        \
    static void pxgTest_##name();                                             \
    static ::pxgtest::TestRegistrar pxgTestRegistrar_##name(#name, pxgTest_##name); \
    static void pxgTest_##name()

// Failure accounting (checked by main after each test).
extern int gFailureCount;

#define PXG_CHECK_MSG(cond, ...)                                              \
    do                                                                        \
    {                                                                         \
        if (!(cond))                                                          \
        {                                                                     \
            ++::pxgtest::gFailureCount;                                       \
            std::printf("    CHECK FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            std::printf("      ");                                            \
            std::printf(__VA_ARGS__);                                         \
            std::printf("\n");                                                \
        }                                                                     \
    } while (0)

#define PXG_CHECK(cond) PXG_CHECK_MSG(cond, "%s", "")

inline bool almostEqual(float a, float b, float relTol = 1e-5f, float absTol = 1e-6f)
{
    const float diff = std::fabs(a - b);
    if (diff <= absTol)
        return true;
    const float largest = std::fmax(std::fabs(a), std::fabs(b));
    return diff <= largest * relTol;
}

// ---------------------------------------------------------------------------
// Device + shader helpers
// ---------------------------------------------------------------------------

// Returns the process-wide test device (created on first use).
PxgSlangComputeDevice* testDevice();

// Loads a compiled shader (SPIR-V) from the directory given by the
// PXG_SHADER_DIR environment variable (set by CTest / CMake).
std::vector<uint8_t> loadShaderBlob(const char* fileName);

// Convenience: buffer upload/download with typed vectors.
template <typename T>
PxgBufferHandle createBufferFrom(PxgSlangComputeDevice* dev, const std::vector<T>& data)
{
    PxgBufferHandle h = dev->createBuffer(data.size() * sizeof(T));
    if (!data.empty())
        dev->upload(h, data.data(), data.size() * sizeof(T));
    return h;
}

template <typename T>
std::vector<T> downloadBuffer(PxgSlangComputeDevice* dev, PxgBufferHandle h, size_t count)
{
    std::vector<T> out(count);
    if (count)
        dev->download(h, out.data(), count * sizeof(T));
    return out;
}

// RAII pipeline wrapper.
class Kernel
{
public:
    Kernel(const char* shaderFile, const char* entryName, std::vector<uint32_t> bufferStrides,
           uint32_t pushConstantSize)
        : mStrides(std::move(bufferStrides))
    {
        std::vector<uint8_t> blob = loadShaderBlob(shaderFile);
        mPipeline = testDevice()->createPipeline(blob.data(), blob.size(), entryName,
                                                 uint32_t(mStrides.size()),
                                                 mStrides.empty() ? nullptr : mStrides.data(),
                                                 pushConstantSize);
    }

    ~Kernel()
    {
        testDevice()->destroyPipeline(mPipeline);
    }

    template <typename Params>
    void dispatch(uint32_t gridX, const std::vector<PxgBufferHandle>& buffers, const Params& params,
                  uint32_t gridY = 1, uint32_t gridZ = 1) const
    {
        testDevice()->dispatch(mPipeline, gridX, gridY, gridZ, buffers.data(), uint32_t(buffers.size()),
                               &params, sizeof(Params));
    }

    void dispatchNoParams(uint32_t gridX, const std::vector<PxgBufferHandle>& buffers,
                          uint32_t gridY = 1, uint32_t gridZ = 1) const
    {
        testDevice()->dispatch(mPipeline, gridX, gridY, gridZ, buffers.data(), uint32_t(buffers.size()),
                               nullptr, 0);
    }

private:
    std::vector<uint32_t> mStrides;
    PxgPipelineHandle mPipeline;
};

} // namespace pxgtest

#endif // PXG_TESTS_FRAMEWORK_H
