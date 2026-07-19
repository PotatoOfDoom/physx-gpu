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

// RUNTIME_PLAN.md P6 test: indirect dispatch.  A producer kernel computes a
// workgroup count on device and writes it as a dispatch-indirect command; the
// consumer is then launched with dispatchIndirect reading that command.  The
// grid is therefore chosen by the device, not the host — if the indirect args
// were ignored (e.g. a fixed 1-group dispatch), only the first 64 outputs
// would be written and the check fails.

#include "pxgTestFramework.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace
{
using namespace pxgtest;

PXG_TEST(p6Indirect)
{
    PxgSlangComputeDevice* dev = testDevice();
    if (!dev->supportsIndirectDispatch())
    {
        std::printf("  NOTE: backend has no indirect dispatch -> skip\n");
        return;
    }

    const uint32_t N = 4096;   // exact multiple of 64 -> 64 workgroups
    struct P { uint32_t count, pad0, pad1, pad2; } prm = { N, 0, 0, 0 };

    // Buffer 0 doubles as the arg buffer (first 3 words) then the output; a
    // dedicated arg buffer keeps the roles clean.
    std::vector<uint32_t> zeros(N, 0);
    PxgBufferHandle bArgs = createBufferFrom(dev, std::vector<uint32_t>(4, 0));
    PxgBufferHandle bOut = createBufferFrom(dev, zeros);

    const std::vector<uint8_t> blob = loadShaderBlob("npIndirect.spv");
    const uint32_t stride = 4;
    PxgPipelineHandle pipeWrite = dev->createPipeline(blob.data(), blob.size(), "writeArgs",
                                                      1, &stride, sizeof(P));
    PxgPipelineHandle pipeFill = dev->createPipeline(blob.data(), blob.size(), "fillGrid",
                                                     1, &stride, sizeof(P));

    // Producer: writes (64, 1, 1) into bArgs on device.
    dev->dispatch(pipeWrite, 1, 1, 1, &bArgs, 1, &prm, sizeof(P));
    dev->barrier();
    // Consumer: grid comes from bArgs, not from the host.
    dev->dispatchIndirect(pipeFill, bArgs, 0, &bOut, 1, &prm, sizeof(P));
    dev->flush();

    // The device-written args are (ceil(N/64), 1, 1).
    const auto args = downloadBuffer<uint32_t>(dev, bArgs, 3);
    std::printf("  device-written dispatch args: (%u, %u, %u)\n", args[0], args[1], args[2]);
    PXG_CHECK_MSG(args[0] == N / 64 && args[1] == 1 && args[2] == 1,
                  "producer wrote the expected indirect workgroup counts");

    const auto g = downloadBuffer<uint32_t>(dev, bOut, N);
    size_t bad = 0;
    for (uint32_t i = 0; i < N; ++i)
        if (g[i] != i + 1)
        {
            if (bad < 5)
                std::printf("  MM %u: got %u want %u\n", i, g[i], i + 1);
            ++bad;
        }
    PXG_CHECK_MSG(bad == 0, "indirect-dispatched grid covered all %u elements (%zu bad)", N, bad);

    dev->destroyPipeline(pipeWrite);
    dev->destroyPipeline(pipeFill);
    dev->destroyBuffer(bArgs);
    dev->destroyBuffer(bOut);
}

}  // namespace
