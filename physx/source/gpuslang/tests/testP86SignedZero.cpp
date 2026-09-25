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

// P9.86 -- the sign of zero on the device, with and without
// OpExecutionMode SignedZeroInfNanPreserve (see shaders/p86SignedZero.slang).

#include "pxgTestFramework.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace
{
using namespace pxgtest;

struct SzPushH { uint64_t inp, outp; };

const uint32_t P0 = 0x00000000u, N0 = 0x80000000u;
const uint32_t NOUT = 19;
// IEEE-754 results, which is what the CUDA computes.
const uint32_t kIeee[NOUT] = { P0, P0, P0, N0, P0, N0,  P0, P0, P0, N0, N0,  0, 0, N0, N0,  P0, P0, N0, P0 };
const char* const kName[NOUT] = { "0 - (+0)", "(-0) + 0", "0 + (-0)", "0 * (-0)", "(0,0,0) - v", "(-0) - 0",
                                  "rz - (+0)", "(-0) + rz", "rz + (-0)", "rz * (-0)", "-1 * (+0)",
                                  "min(+0,-0)", "max(-0,+0)", "PxMin(+0,-0)", "PxMax(+0,-0)",
                                  "(-0) + pz", "pz - (+0)", "(0,0,0) * -1", "e=0; e -= +0" };

std::vector<uint32_t> run(const char* spv)
{
    PxgSlangComputeDevice* dev = testDevice();
    const std::vector<float> in = { 0.0f, -0.0f, -1.0f, 0.0f };
    const std::vector<uint32_t> outInit(NOUT, 0xDEADBEEFu);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);
    SzPushH push{ dev->getBufferDeviceAddress(bIn), dev->getBufferDeviceAddress(bOut) };
    Kernel k(spv, "szProbe", {}, sizeof(SzPushH));
    k.dispatch(1u, {}, push, 1u, 1u);
    dev->flush();
    std::vector<uint32_t> got = downloadBuffer<uint32_t>(dev, bOut, NOUT);
    dev->destroyBuffer(bIn); dev->destroyBuffer(bOut);
    return got;
}
} // namespace

PXG_TEST(p86SignedZero)
{
    const std::vector<uint32_t> plain = run("p86SignedZero.spv");
    const std::vector<uint32_t> sz = run("p86SignedZeroSz.spv");
    std::printf("  %-14s %-10s %-10s %-10s\n", "expr", "IEEE", "plain", "SZINP");
    for (uint32_t k = 0; k < NOUT; ++k)
        std::printf("  %-14s %08x   %08x   %08x%s\n", kName[k], kIeee[k], plain[k], sz[k],
                    (k < 11 || k >= 13) && (plain[k] != kIeee[k] || sz[k] != kIeee[k]) ? "   <-- differs" : "");
    // Runtime-operand arithmetic and the PxMin/PxMax spellings are IEEE in both builds.
    for (uint32_t k = 6; k < 11; ++k)
        PXG_CHECK_MSG(plain[k] == kIeee[k] && sz[k] == kIeee[k], "%s: runtime-operand arithmetic lost the sign of zero", kName[k]);
    for (uint32_t k = 13; k < 15; ++k)
        PXG_CHECK_MSG(plain[k] == kIeee[k] && sz[k] == kIeee[k], "%s: select spelling lost the sign of zero", kName[k]);
    // What the engine relies on since P9.86 (every shader is stamped SignedZeroInfNanPreserve):
    // the mode stops the driver folding x * 0 and x +/- a PROVEN zero.  It does NOT stop
    // OpFSub(constant 0, x) -> -x (rows 0, 4, 18) -- the port spells those as selects -- and
    // rows 1-2 are slangc's own folds, never reaching the driver.
    for (uint32_t k : { 3u, 15u, 16u, 17u })
        PXG_CHECK_MSG(sz[k] == kIeee[k], "%s: SignedZeroInfNanPreserve no longer honoured", kName[k]);
}
