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

// RUNTIME_PLAN.md P1.6 test: GPU timestamp queries.  A begin/end pair
// brackets a dispatch; after the batch retires, the elapsed time is a
// non-negative, finite nanosecond count.  Also checks index monotonicity
// and that out-of-range query indices are rejected.

#include "pxgTestFramework.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace
{
using namespace pxgtest;

PXG_TEST(p1Timestamps)
{
    PxgSlangComputeDevice* dev = testDevice();
    if (!dev->supportsTimestamps())
    {
        std::printf("  NOTE: device has no usable timestamps -> skip\n");
        return;
    }

    const uint32_t N = 4096;
    const uint32_t GROUPS = N / 64;
    struct SgOut { uint32_t x, y, z, w; };
    struct P { uint32_t numThreads; } prm = { N };

    std::vector<SgOut> init(N);
    PxgBufferHandle bOut = createBufferFrom(dev, init);
    Kernel k("npSpikeSubgroup.spv", "spikeSubgroup", { (uint32_t)sizeof(SgOut) }, sizeof(P));

    const uint32_t t0 = dev->writeTimestamp();
    k.dispatch(GROUPS, { bOut }, prm);
    const uint32_t t1 = dev->writeTimestamp();
    dev->flush();

    PXG_CHECK_MSG(t0 != 0xffffffffu && t1 != 0xffffffffu, "timestamps recorded (valid indices)");
    PXG_CHECK_MSG(t1 == t0 + 1, "timestamp query indices are monotonic within a batch");

    const double ns = dev->getTimestampNanos(t0, t1);
    std::printf("  dispatch bracketed by timestamps: %.1f ns\n", ns);
    PXG_CHECK_MSG(ns >= 0.0, "elapsed time is non-negative");
    PXG_CHECK_MSG(ns < 1.0e12, "elapsed time is a finite, plausible nanosecond count");

    // Order-insensitivity: swapping begin/end gives the same magnitude.
    const double nsRev = dev->getTimestampNanos(t1, t0);
    PXG_CHECK_MSG(nsRev >= 0.0 && nsRev == ns, "getTimestampNanos is symmetric in its two indices");

    // Out-of-range / sentinel indices are rejected.
    PXG_CHECK_MSG(dev->getTimestampNanos(0xffffffffu, t1) < 0.0, "invalid begin index rejected");
    PXG_CHECK_MSG(dev->getTimestampNanos(t0, 100000u) < 0.0, "out-of-range end index rejected");

    dev->destroyBuffer(bOut);
}

}  // namespace
