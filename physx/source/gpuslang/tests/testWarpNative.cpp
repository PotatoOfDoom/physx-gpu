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

// RUNTIME_PLAN.md P2.1: native subgroup-path validation.  The emulation
// battery (npWarpLib) proves the shared-memory path bit-exactly; this proves
// the NATIVE path's SPIR-V (WaveShuffle / WaveActiveBallot lowerings) is
// well-formed and accepted by the driver — the coverage that the shared build
// (slangc) plus this driver load together provide.  The 32-lane CUDA warp
// semantics of the native path are correct only under requiredSubgroupSize==32
// (hardware; CI's llvmpipe is fixed at 8), so this test loads + dispatches
// without asserting 32-lane results; when the device CAN require 32 it also
// exercises the required-size pipeline path.

#include "pxgTestFramework.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace
{
using namespace pxgtest;

PXG_TEST(warpNative)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t N = 32;

    const std::vector<uint8_t> blob = loadShaderBlob("npWarpNative.spv");
    const uint32_t stride = 4;

    // Driver-side validation: a compute pipeline built from the native
    // subgroup module must be accepted (GroupNonUniformShuffle / Ballot).
    const PxgPipelineHandle pipe = dev->createPipeline(blob.data(), blob.size(), "warpNativeProbe",
                                                       1, &stride, 16);
    PXG_CHECK_MSG(pipe != 0, "driver accepts the native subgroup module (default size)");
    if (pipe != 0)
    {
        std::vector<uint32_t> outInit(N, 0);
        PxgBufferHandle bOut = createBufferFrom(dev, outInit);
        struct P { uint32_t numThreads, pad0, pad1, pad2; } prm = { N, 0, 0, 0 };
        dev->dispatch(pipe, 1, 1, 1, &bOut, 1, &prm, sizeof(P));
        dev->flush();  // must execute without device loss
        dev->destroyBuffer(bOut);
        dev->destroyPipeline(pipe);
    }

    const PxgSlangComputeDevice::SubgroupCaps caps = dev->getSubgroupCaps();
    if (caps.canRequireSize && caps.minSize <= 32 && caps.maxSize >= 32)
    {
        const PxgPipelineHandle pipe32 = dev->createPipelineEx(blob.data(), blob.size(), "warpNativeProbe",
                                                               1, &stride, 16, 32);
        PXG_CHECK_MSG(pipe32 != 0, "native module builds at the REQUIRED subgroup size 32");
        if (pipe32 != 0)
        {
            std::vector<uint32_t> outInit(N, 0);
            PxgBufferHandle bOut = createBufferFrom(dev, outInit);
            struct P { uint32_t numThreads, pad0, pad1, pad2; } prm = { N, 0, 0, 0 };
            dev->dispatch(pipe32, 1, 1, 1, &bOut, 1, &prm, sizeof(P));
            dev->flush();
            const auto g = downloadBuffer<uint32_t>(dev, bOut, N);
            // At a genuine 32-lane subgroup the native path realizes the
            // CUDA warp semantics; here just require the dispatch produced
            // output (no device loss / all-zero stall).
            size_t nonzero = 0;
            for (uint32_t i = 0; i < N; ++i)
                nonzero += g[i] != 0 ? 1 : 0;
            PXG_CHECK_MSG(nonzero > 0, "native required-32 dispatch produced results");
            dev->destroyBuffer(bOut);
            dev->destroyPipeline(pipe32);
        }
    }
    else
    {
        std::printf("  NOTE: device cannot require subgroup 32 (window [%u,%u]) -> native 32-lane "
                    "semantics validated on hardware in P5; compile+load+dispatch validated here\n",
                    caps.minSize, caps.maxSize);
    }
}

}  // namespace
