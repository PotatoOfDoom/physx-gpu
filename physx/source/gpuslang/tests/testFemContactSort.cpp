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

// P8.20 FEM contact-sort bring-up test: the three orchestration kernels
// PxgSoftBodyCore::sortContacts() and PxgFEMClothCore::sortContacts() launch
// (femContactSortBda.spv) on the engine ABI (BDA).  Each kernel is driven by the
// EXACT CUDA argument pack of device pointers pushed as a constant blob, with
// every buffer addressed by its buffer-device-address — the same EngineBlob
// pattern testParticleContactSort uses for the twin-gather siblings.  Each
// output is compared BIT-EXACT against a CPU transcription of the gpucommon .cu
// (radixSortImpl.cu / MemCopyBalanced.cu).
//
// Suites (substring-matched by the "femContactSort" ctest filter):
//   femContactSortHigh32Bits  — radixSortCopyHigh32Bits single high-word gather
//   femContactSortCopy        — radixSortCopy single full 64-bit gather
//   femContactSortClampValues — clampMaxValues three-counter in-place clamp

#include "pxgTestFramework.h"

#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

// A device buffer + its buffer-device-address (BDA).
struct DevBuf
{
    PxgBufferHandle h = 0;
    uint64_t addr = 0;
};

template <typename T>
DevBuf makeDev(PxgSlangComputeDevice* dev, const std::vector<T>& data)
{
    DevBuf b;
    b.h = createBufferFrom(dev, data);
    b.addr = dev->getBufferDeviceAddress(b.h);
    return b;
}

// The CUDA pack shared by radixSortCopyHigh32Bits / radixSortCopy:
// 4 device pointers -> 32-byte blob.
struct RadixCopyPush
{
    uint64_t inValue, outValue, rank, numKeys;
};

// The rank array the engine hands these kernels is the output of an 8-pass radix
// sort, so it is a PERMUTATION — but the kernels never assume that, they just
// gather.  Building a shuffled permutation is the faithful input; building it
// over a capacity LARGER than numKeys is what proves the *numKeys deref (a
// kernel that ignored the device-pointer count and ran to capacity would write
// past the active range and be caught by the sentinel tail).
struct GatherInputs
{
    std::vector<uint64_t> in;
    std::vector<uint32_t> rank;
    uint32_t numKeys;
    uint32_t cap;
};

GatherInputs makeGatherInputs(std::mt19937& rng, uint32_t cap, uint32_t numKeys)
{
    GatherInputs g;
    g.cap = cap;
    g.numKeys = numKeys;
    g.in.resize(cap);
    for (uint32_t i = 0; i < cap; ++i)
    {
        // Spread real bits through BOTH words: a key whose high word was always
        // zero would let a low/high mix-up in the High32Bits gather pass.
        g.in[i] = (uint64_t(rng()) << 32) | rng();
    }
    g.rank.resize(numKeys);
    std::iota(g.rank.begin(), g.rank.end(), 0u);
    std::shuffle(g.rank.begin(), g.rank.end(), rng);
    return g;
}

// ---------------------------------------------------------------------------
// radixSortCopyHigh32Bits — rank-indexed gather keeping the HIGH word.
//   out[w] = uint32_t(in[rank[w]] >> 32);   for w < *numKeys
// ---------------------------------------------------------------------------
PXG_TEST(femContactSortHigh32Bits)
{
    PxgSlangComputeDevice* dev = testDevice();
    std::mt19937 rng(0x5B0C1EAu);

    // The launch is grid (32,1,1) x block (1024,1,1) = 32768 threads, so numKeys
    // MUST exceed 32768 or the grid-stride loop never takes a second turn and the
    // stride is dead code the fixture cannot see.  It was 700 at first, and a
    // mutant that DOUBLED the stride survived for exactly that reason.
    const uint32_t CAP = 40000;       // buffer capacity
    const uint32_t numKeys = 35000;   // active count: > 32768 (loops twice) and < CAP
    const uint32_t SENT = 0xFEEDFACEu;

    GatherInputs gi = makeGatherInputs(rng, CAP, numKeys);
    std::vector<uint32_t> out(CAP, SENT);
    std::vector<uint32_t> numBuf = { numKeys };

    DevBuf bIn = makeDev(dev, gi.in);
    DevBuf bRank = makeDev(dev, gi.rank);
    DevBuf bOut = makeDev(dev, out);
    DevBuf bNum = makeDev(dev, numBuf);

    RadixCopyPush push = { bIn.addr, bOut.addr, bRank.addr, bNum.addr };

    // grid (32,1,1) block (1024,1,1) mirrors PxgSoftBodyCore.cpp:559 / :645 and
    // PxgFEMClothCore.cpp:935 / :1019.
    Kernel k("femContactSortBda.spv", "radixSortCopyHigh32Bits", {}, sizeof(RadixCopyPush));
    k.dispatch(32u, {}, push, /*gridY*/ 1u, /*gridZ*/ 1u);
    dev->flush();

    const auto g = downloadBuffer<uint32_t>(dev, bOut.h, CAP);

    size_t mm = 0;
    for (uint32_t w = 0; w < CAP; ++w)
    {
        const uint32_t e = (w < numKeys) ? uint32_t(gi.in[gi.rank[w]] >> 32) : SENT;
        if (g[w] != e) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "radixSortCopyHigh32Bits bit-exact vs CPU (%zu mismatches)", mm);

    for (PxgBufferHandle h : { bIn.h, bRank.h, bOut.h, bNum.h })
        dev->destroyBuffer(h);
}

// ---------------------------------------------------------------------------
// radixSortCopy — the same rank-indexed gather, full 64 bits.
//   out[w] = in[rank[w]];   for w < *numKeys
// ---------------------------------------------------------------------------
PXG_TEST(femContactSortCopy)
{
    PxgSlangComputeDevice* dev = testDevice();
    std::mt19937 rng(0x5B0C1EBu);

    const uint32_t CAP = 40000;       // see femContactSortHigh32Bits: numKeys must
    const uint32_t numKeys = 35000;   // exceed the 32768-thread grid to exercise the stride
    const uint64_t SENT = 0xDEADBEEFCAFEF00Dull;

    GatherInputs gi = makeGatherInputs(rng, CAP, numKeys);
    std::vector<uint64_t> out(CAP, SENT);
    std::vector<uint32_t> numBuf = { numKeys };

    DevBuf bIn = makeDev(dev, gi.in);
    DevBuf bRank = makeDev(dev, gi.rank);
    DevBuf bOut = makeDev(dev, out);
    DevBuf bNum = makeDev(dev, numBuf);

    RadixCopyPush push = { bIn.addr, bOut.addr, bRank.addr, bNum.addr };

    // grid (32,1,1) block (1024,1,1) mirrors PxgSoftBodyCore.cpp:614 and
    // PxgFEMClothCore.cpp:991 / :1078.
    Kernel k("femContactSortBda.spv", "radixSortCopy", {}, sizeof(RadixCopyPush));
    k.dispatch(32u, {}, push, /*gridY*/ 1u, /*gridZ*/ 1u);
    dev->flush();

    const auto g = downloadBuffer<uint64_t>(dev, bOut.h, CAP);

    size_t mm = 0;
    for (uint32_t w = 0; w < CAP; ++w)
    {
        const uint64_t e = (w < numKeys) ? gi.in[gi.rank[w]] : SENT;
        if (g[w] != e) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "radixSortCopy bit-exact vs CPU (%zu mismatches)", mm);

    for (PxgBufferHandle h : { bIn.h, bRank.h, bOut.h, bNum.h })
        dev->destroyBuffer(h);
}

// ---------------------------------------------------------------------------
// clampMaxValues — in-place clamp of THREE device counters.
//   if (*v0 > maxValue) *v0 = maxValue;   (and the same for v1, v2)
//
// The three counters get DIFFERENT values in every case, and the buffers are
// allocated separately, so a shader that clamped the same pointer three times —
// or that read value1/value2 from the wrong push-constant offset — fails here
// even though the single-counter clampMaxValue sibling would still pass.
// ---------------------------------------------------------------------------
PXG_TEST(femContactSortClampValues)
{
    PxgSlangComputeDevice* dev = testDevice();

    struct ClampPush { uint64_t value0, value1, value2; uint32_t maxValue; uint32_t pad; };
    const uint32_t maxValue = 4096u;

    // (initial v0, v1, v2) — chosen so that in every case the three counters land
    // in DIFFERENT branches of the clamp, and so that each slot is over the max in
    // at least one case and under it in at least one other.
    struct Case { uint32_t v[3]; };
    const Case cases[] = {
        { { 9000u, 4096u,    12u } },  // over / exactly-at / under
        { {   12u, 9000u,     0u } },  // under / over / zero
        { {    0u,    7u,  9000u } },  // zero / under / over
        { { 5000u, 6000u,  7000u } },  // all three over — all three must clamp
        { {    1u,    2u,     3u } },  // none over — nothing may move
    };

    size_t mm = 0;
    for (const Case& c : cases)
    {
        std::vector<uint32_t> b0 = { c.v[0] }, b1 = { c.v[1] }, b2 = { c.v[2] };
        DevBuf d0 = makeDev(dev, b0), d1 = makeDev(dev, b1), d2 = makeDev(dev, b2);

        ClampPush push = { d0.addr, d1.addr, d2.addr, maxValue, 0u };

        // grid (1,1,1) block (1,1,1) mirrors PxgSoftBodyCore.cpp:474 and
        // PxgFEMClothCore.cpp:826.
        Kernel k("femContactSortBda.spv", "clampMaxValues", {}, sizeof(ClampPush));
        k.dispatch(1u, {}, push, /*gridY*/ 1u, /*gridZ*/ 1u);
        dev->flush();

        const DevBuf* bufs[3] = { &d0, &d1, &d2 };
        for (int i = 0; i < 3; ++i)
        {
            const uint32_t exp = c.v[i] > maxValue ? maxValue : c.v[i];
            const auto g = downloadBuffer<uint32_t>(dev, bufs[i]->h, 1);
            if (g[0] != exp)
            {
                ++mm;
                std::printf("  clampMaxValues MM slot %d: in=%u max=%u got=%u exp=%u\n",
                            i, c.v[i], maxValue, g[0], exp);
            }
        }
        for (PxgBufferHandle h : { d0.h, d1.h, d2.h })
            dev->destroyBuffer(h);
    }
    PXG_CHECK_MSG(mm == 0, "clampMaxValues bit-exact vs CPU (%zu mismatches)", mm);
}

} // namespace
