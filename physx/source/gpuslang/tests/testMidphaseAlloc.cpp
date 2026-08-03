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

// P8.29 — midphaseAllocate's layout arithmetic (midphaseAllocBda.spv /
// pxgSlangMidphaseAlloc).
//
// Suite (ctest filter "midphaseAlloc"):
//   midphaseAllocLayout — every one of the seven sub-buffer offsets, over an
//                         EXHAUSTIVE sweep of the padding boundary plus a set of
//                         realistic counts, against a CPU transcription.
//
// This is cheap to test exhaustively because it is pure integer arithmetic, and
// worth testing exhaustively because it is a CONTRACT: convexTrimeshNarrowphase
// carves the stack this way and the five kernels after it re-derive the same
// layout.  An error here does not fault -- every wrong address still lands
// inside the stack allocation -- it just silently reads someone else's data.

#include "pxgTestFramework.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace
{
using namespace pxgtest;

struct AllocPush { uint64_t cases, out; uint32_t count, pad20; };
static_assert(sizeof(AllocPush) == 24, "alloc blob must be 24 bytes");

// midphaseAllocate.cuh:129, in ALLOCATION order -- which is NOT the order the
// arguments are declared in.  See the module header.
struct Layout { uint64_t pairs, ni, contacts, ordered, maxDepth, interm, secondPass; };

static Layout cpuLayout(uint32_t numPairsUnpadded, uint32_t numPaddedPairs)
{
    const uint64_t n = uint64_t((numPairsUnpadded + 3u) & ~3u);
    const uint64_t nPadded = uint64_t(numPaddedPairs);
    Layout L{};
    uint64_t p = 0;
    L.pairs      = p; p += n * 16;   // uint4
    L.ni         = p; p += n * 16;   // ConvexTriNormalAndIndex
    L.contacts   = p; p += n * 4;    // ConvexTriContacts
    L.ordered    = p; p += nPadded * 4;
    L.maxDepth   = p; p += n * 4;
    L.interm     = p; p += n * 4;
    L.secondPass = p;
    return L;
}

PXG_TEST(midphaseAllocLayout)
{
    PxgSlangComputeDevice* dev = testDevice();

    // The padding boundary EXHAUSTIVELY over 0..64 -- (n+3)&~3 has a four-step
    // period, so this covers every residue many times over and pins the two
    // boundaries that matter (0 stays 0, and 1..4 all round to 4).  The padded
    // counter is varied independently, because it is accumulated by a different
    // atomic and is NOT a function of the unpadded one.
    std::vector<uint32_t> cases;
    for (uint32_t n = 0; n <= 64; ++n)
    {
        cases.push_back(n);
        cases.push_back(((n + 3u) & ~3u) * 2u);   // what the midphase would have accumulated
    }
    // Realistic and adversarial counts: a large pair set, a padded counter that is
    // NOT twice the rounded unpadded one (which is what a port that "simplified"
    // the second counter away would produce), and zero-on-one-side pairs.
    const uint32_t extra[][2] = {
        { 1u, 0u }, { 0u, 8u }, { 3u, 8u }, { 4u, 8u }, { 5u, 16u },
        { 1000u, 2000u }, { 1001u, 2008u }, { 65535u, 131072u },
        { 7u, 3u },            // padded SMALLER than the rounded unpadded count
        { 12u, 999u },         // and much larger
    };
    for (const auto& e : extra) { cases.push_back(e[0]); cases.push_back(e[1]); }

    const uint32_t N = uint32_t(cases.size() / 2);
    std::vector<uint64_t> outInit(size_t(N) * 7u, 0xDEADBEEFDEADBEEFull);

    PxgBufferHandle bCases = createBufferFrom(dev, cases);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    AllocPush push{};
    push.cases = dev->getBufferDeviceAddress(bCases);
    push.out = dev->getBufferDeviceAddress(bOut);
    push.count = N;

    Kernel k("midphaseAllocBda.spv", "midphaseAllocLaunch", {}, sizeof(AllocPush));
    k.dispatch((N + 63u) / 64u, {}, push, 1u, 1u);
    dev->flush();

    const auto got = downloadBuffer<uint64_t>(dev, bOut, N * 7u);

    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const Layout e = cpuLayout(cases[2*i], cases[2*i + 1]);
        const uint64_t exp[7] = { e.pairs, e.ni, e.contacts, e.ordered,
                                  e.maxDepth, e.interm, e.secondPass };
        const char* names[7] = { "pairs", "ni", "contacts", "ordered",
                                 "maxDepth", "interm", "secondPass" };
        for (int f = 0; f < 7; ++f)
        {
            if (got[i * 7 + f] != exp[f])
            {
                ++mm;
                if (mm <= 12)
                    std::printf("  case %u (n=%u, padded=%u) %s: got=%llu exp=%llu\n",
                                i, cases[2*i], cases[2*i+1], names[f],
                                (unsigned long long)got[i*7+f], (unsigned long long)exp[f]);
            }
        }
    }
    PXG_CHECK_MSG(mm == 0, "midphaseAllocate layout matches CPU over %u cases (%zu mismatches)",
                  N, mm);

    dev->destroyBuffer(bCases);
    dev->destroyBuffer(bOut);
}

} // namespace
