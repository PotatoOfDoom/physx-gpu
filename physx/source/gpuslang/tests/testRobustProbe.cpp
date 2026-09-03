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

// testRobustProbe.cpp -- P9.50.  Does an out-of-range DESCRIPTOR-BOUND read get
// bounds-checked on this platform?
//
// This decides whether check_speculated_loads.py's scope -- it tracks `Ptr<>`
// (buffer-device-address) reads and nothing else -- is a sound limit or an
// unexamined one.  The log asserts both answers in different places; see the
// header of shaders/npRobustProbe.slang for which, and why the one that says
// "bounds-checked" does not carry its own evidence.
//
// WHY IT IS OPT-IN, and why that is not a way of hiding it:
//
// The probe deliberately reads far outside an allocation.  If the platform does
// NOT bounds-check, the honest outcome of this test is a SIGSEGV -- which, in a
// ctest run, would look exactly like the flaky suite failures this port has
// spent three tranches chasing, and would poison every future bisection.  So it
// runs only when PXG_PROBE_ROBUST=1 is set, and prints its verdict rather than
// asserting one: the finding belongs in RUNTIME_STATUS, where a person reads it,
// not in a gate that fails for a reason unrelated to the change under test.
//
// Run it with:
//     PXG_PROBE_ROBUST=1 PXG_SHADER_DIR=$PWD/shaders ./gpuslang-tests robustProbe

#include "pxgTestFramework.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace
{
using namespace pxgtest;

struct RobustProbeParams
{
    uint32_t index;
    uint32_t pad0, pad1, pad2;
};

PXG_TEST(robustProbe)
{
    const char* on = std::getenv("PXG_PROBE_ROBUST");
    if (!on || on[0] != '1')
    {
        std::printf("  SKIPPED: set PXG_PROBE_ROBUST=1 to run it.  It reads far past an\n"
                    "  allocation on purpose; if this platform does not bounds-check, the\n"
                    "  correct outcome is a SIGSEGV, and that must never land in ctest.\n");
        return;
    }

    PxgSlangComputeDevice* dev = testDevice();

    // A small source buffer, filled with a recognisable pattern so a value that
    // comes back can be told apart from both zero and this buffer's contents.
    const uint32_t N = 64u;
    std::vector<uint32_t> src(N);
    for (uint32_t i = 0; i < N; ++i)
        src[i] = 0xAB00u + i;
    std::vector<uint32_t> outInit(4u, 0xEEEEEEEEu);

    PxgBufferHandle bSrc = createBufferFrom(dev, src);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);
    const std::vector<uint32_t> strides = { 4, 4 };
    const std::vector<PxgBufferHandle> binds = { bSrc, bOut };

    // In-range first.  A probe whose CONTROL has not been shown to work cannot
    // distinguish "bounds-checked" from "the dispatch never ran".
    auto probe = [&](const char* what, uint32_t index) -> uint32_t
    {
        std::vector<uint32_t> reset(4u, 0xEEEEEEEEu);
        dev->upload(bOut, reset.data(), reset.size() * sizeof(uint32_t));
        RobustProbeParams p = { index, 0u, 0u, 0u };
        Kernel k("npRobustProbe.spv", "robustProbeRead", strides, sizeof(RobustProbeParams));
        std::printf("  reading index %-12u (%s) ... ", index, what);
        std::fflush(stdout);
        k.dispatch(1u, binds, p);
        dev->flush();
        const auto g = downloadBuffer<uint32_t>(dev, bOut, 4u);
        std::printf("value 0x%08x, marker 0x%08x\n", g[0], g[1]);
        std::fflush(stdout);
        return g[0];
    };

    const uint32_t inRange = probe("in range, the control", N - 1u);
    PXG_CHECK_MSG(inRange == 0xAB00u + (N - 1u),
                  "the control read returns the element that is actually there");

    // Now outward.  Each step is a separate dispatch so the log names the last
    // one that survived.  4 bytes per element: 1 MiB, 16 MiB, 256 MiB, 1 GiB.
    probe("just past the end", N);
    probe("+1 MiB", 256u * 1024u);
    probe("+16 MiB", 4u * 1024u * 1024u);
    probe("+256 MiB", 64u * 1024u * 1024u);
    probe("+1 GiB", 256u * 1024u * 1024u);

    std::printf("\n  VERDICT: the process survived every read above.\n"
                "  A value of 0x00000000 far out means the access IS bounds-checked;\n"
                "  a non-zero value means it is not, and merely landed in mapped memory.\n"
                "  Reaching this line at all rules out the third outcome, a fault.\n");

    dev->destroyBuffer(bSrc);
    dev->destroyBuffer(bOut);
}

}  // namespace
