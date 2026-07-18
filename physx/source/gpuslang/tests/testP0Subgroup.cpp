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

// RUNTIME_PLAN.md P0 spike test: wave/subgroup intrinsics + required
// subgroup size.  Validates the size-independent invariants of the wave ops
// the warp-32 primitive library builds on, reports the device's subgroup
// capabilities, and — when subgroup size control is available — proves the
// createPipelineEx required-size mechanism by requesting a size within the
// device's [min, max] window and verifying the observed lane count.

#include "pxgTestFramework.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace
{
using namespace pxgtest;

struct SgOut { uint32_t x, y, z, w; };

static size_t validateWaveInvariants(const std::vector<SgOut>& g, uint32_t expectLaneCount)
{
    size_t bad = 0;
    for (size_t i = 0; i < g.size(); ++i)
    {
        const uint32_t laneIdx = g[i].x & 0xffffu;
        const uint32_t laneCount = g[i].x >> 16;
        bool ok = laneCount >= 4 && laneCount <= 128 && (laneCount & (laneCount - 1)) == 0
               && laneIdx < laneCount
               && g[i].y == laneIdx        // WavePrefixSum(1) == laneIdx
               && g[i].z == laneCount      // popcount(ballot(true)) == laneCount
               && g[i].w == laneCount;     // WaveActiveSum(1) == laneCount
        if (expectLaneCount != 0)
            ok = ok && laneCount == expectLaneCount;
        if (!ok)
        {
            if (bad < 5)
                std::printf("  BAD thread %zu: laneIdx %u laneCount %u prefix %u ballot %u sum %u (expect %u)\n",
                            i, laneIdx, laneCount, g[i].y, g[i].z, g[i].w, expectLaneCount);
            ++bad;
        }
    }
    return bad;
}

PXG_TEST(p0Subgroup)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t N = 4096;                 // exact multiple of 64 => no divergence
    const uint32_t GROUPS = N / 64;

    const PxgSlangComputeDevice::SubgroupCaps caps = dev->getSubgroupCaps();
    std::printf("  subgroup caps: default=%u min=%u max=%u canRequire=%d fullSubgroups=%d\n",
                caps.defaultSize, caps.minSize, caps.maxSize, int(caps.canRequireSize), int(caps.fullSubgroups));

    struct P { uint32_t numThreads; } prm = { N };

    // --- default pipeline: invariants at the device's default subgroup size ---
    {
        std::vector<SgOut> outInit(N);
        PxgBufferHandle bOut = createBufferFrom(dev, outInit);
        Kernel k("npSpikeSubgroup.spv", "spikeSubgroup", { (uint32_t)sizeof(SgOut) }, sizeof(P));
        k.dispatch(GROUPS, { bOut }, prm);
        dev->flush();
        const auto g = downloadBuffer<SgOut>(dev, bOut, N);
        const uint32_t observed = g[0].x >> 16;
        std::printf("  default-pipeline observed subgroup size: %u\n", observed);
        PXG_CHECK_MSG(validateWaveInvariants(g, 0) == 0, "wave invariants hold at default subgroup size");
        dev->destroyBuffer(bOut);
    }

    // --- required-size pipeline: prove the createPipelineEx mechanism ---
    if (caps.canRequireSize)
    {
        // Prefer 32 (the warp-32 profile); otherwise exercise the mechanism at
        // the device's minimum size (e.g. llvmpipe is fixed at 8).
        const uint32_t want = (caps.minSize <= 32 && caps.maxSize >= 32) ? 32u : caps.minSize;
        const std::vector<unsigned char> blob = loadShaderBlob("npSpikeSubgroup.spv");
        const uint32_t stride = (uint32_t)sizeof(SgOut);
        const PxgPipelineHandle pipe = dev->createPipelineEx(blob.data(), blob.size(), "spikeSubgroup",
                                                             1, &stride, sizeof(P), want);
        PXG_CHECK_MSG(pipe != 0, "createPipelineEx honors an in-range required subgroup size");
        if (pipe != 0)
        {
            std::vector<SgOut> outInit(N);
            PxgBufferHandle bOut = createBufferFrom(dev, outInit);
            dev->dispatch(pipe, GROUPS, 1, 1, &bOut, 1, &prm, sizeof(P));
            dev->flush();
            const auto g = downloadBuffer<SgOut>(dev, bOut, N);
            std::printf("  required-size pipeline (%u) observed subgroup size: %u\n", want, g[0].x >> 16);
            PXG_CHECK_MSG(validateWaveInvariants(g, want) == 0, "wave invariants hold at the REQUIRED subgroup size");
            dev->destroyBuffer(bOut);
            dev->destroyPipeline(pipe);
        }
        if (!(caps.minSize <= 32 && caps.maxSize >= 32))
            std::printf("  NOTE: required-32 unavailable on this device (window [%u,%u]) -> "
                        "warp-32 primitives will use the EMULATION path here\n",
                        caps.minSize, caps.maxSize);
        // Out-of-range requests must be rejected, not misconfigured.
        const PxgPipelineHandle bad = dev->createPipelineEx(blob.data(), blob.size(), "spikeSubgroup",
                                                            1, &stride, sizeof(P), 256);
        PXG_CHECK_MSG(bad == 0, "createPipelineEx rejects an out-of-range subgroup size");
    }
    else
    {
        std::printf("  NOTE: subgroupSizeControl unavailable -> emulation path only on this device\n");
    }
}

}  // namespace
