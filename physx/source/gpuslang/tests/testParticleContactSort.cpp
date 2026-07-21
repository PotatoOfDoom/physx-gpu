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

// P6.1 particle contact-sort bring-up test: the three contact-sort orchestration
// kernels (particleContactSortBda.spv) on the engine ABI (BDA).  Each kernel is
// driven by the EXACT CUDA argument pack of device pointers pushed as a
// constant blob (the engine-ABI form), with the input/output buffers addressed
// by their buffer-device-address — the same EngineBlob pattern testParticleSim
// uses.  Each output is compared BIT-EXACT against a CPU transcription of the
// gpucommon .cu (radixSortImpl.cu / MemCopyBalanced.cu).
//
// Suites (substring-matched by the "particleContactSort" ctest filter):
//   particleContactSortDoubleCopy   — radixSortDoubleCopy twin 64-bit gather
//   particleContactSortHigh32Bits   — radixSortDoubleCopyHigh32Bits high-word gather
//   particleContactSortClampMaxValue — clampMaxValue in-place counter clamp

#include "pxgTestFramework.h"

#include <cstdint>
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

// The CUDA pack shared by radixSortDoubleCopy / radixSortDoubleCopyHigh32Bits:
// 7 device pointers -> 56-byte blob.
struct DoubleCopyPush
{
    uint64_t inValue0, outValue0, rank0, inValue1, outValue1, rank1, numKeys;
};

// ---------------------------------------------------------------------------
// radixSortDoubleCopy — two independent rank-indexed full 64-bit gathers.
//   out0[w] = in0[rank0[w]];  out1[w] = in1[rank1[w]];   for w < *numKeys.
// ---------------------------------------------------------------------------
PXG_TEST(particleContactSortDoubleCopy)
{
    PxgSlangComputeDevice* dev = testDevice();
    std::mt19937 rng(0xC0FFEE);

    const uint32_t CAP = 1024;      // buffer capacity
    const uint32_t numKeys = 700;   // active count (< CAP: proves *numKeys deref + tail untouched)
    const uint64_t SENT = 0xDEADBEEFCAFEF00Dull;

    std::vector<uint64_t> inA(CAP), inB(CAP);
    for (uint32_t i = 0; i < CAP; ++i)
    {
        inA[i] = (uint64_t(rng()) << 32) | rng();
        inB[i] = (uint64_t(rng()) << 32) | rng();
    }
    // valid rank permutations of [0, numKeys)
    std::vector<uint32_t> rankA(numKeys), rankB(numKeys);
    std::iota(rankA.begin(), rankA.end(), 0u); std::shuffle(rankA.begin(), rankA.end(), rng);
    std::iota(rankB.begin(), rankB.end(), 0u); std::shuffle(rankB.begin(), rankB.end(), rng);

    std::vector<uint64_t> out0(CAP, SENT), out1(CAP, SENT);
    std::vector<uint32_t> numBuf = { numKeys };

    DevBuf bInA = makeDev(dev, inA);
    DevBuf bInB = makeDev(dev, inB);
    DevBuf bRankA = makeDev(dev, rankA);
    DevBuf bRankB = makeDev(dev, rankB);
    DevBuf bOut0 = makeDev(dev, out0);
    DevBuf bOut1 = makeDev(dev, out1);
    DevBuf bNum = makeDev(dev, numBuf);

    DoubleCopyPush push = { bInA.addr, bOut0.addr, bRankA.addr,
                            bInB.addr, bOut1.addr, bRankB.addr, bNum.addr };

    // grid (32,1,1) block (1024,1,1) mirrors PxgParticleSystemCore.cpp:1268.
    Kernel k("particleContactSortBda.spv", "radixSortDoubleCopy", {}, sizeof(DoubleCopyPush));
    k.dispatch(32u, {}, push, /*gridY*/ 1u, /*gridZ*/ 1u);
    dev->flush();

    const auto g0 = downloadBuffer<uint64_t>(dev, bOut0.h, CAP);
    const auto g1 = downloadBuffer<uint64_t>(dev, bOut1.h, CAP);

    size_t mm = 0;
    for (uint32_t w = 0; w < CAP; ++w)
    {
        const uint64_t e0 = (w < numKeys) ? inA[rankA[w]] : SENT;
        const uint64_t e1 = (w < numKeys) ? inB[rankB[w]] : SENT;
        if (g0[w] != e0) ++mm;
        if (g1[w] != e1) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "radixSortDoubleCopy bit-exact vs CPU (%zu mismatches)", mm);

    for (PxgBufferHandle h : { bInA.h, bInB.h, bRankA.h, bRankB.h, bOut0.h, bOut1.h, bNum.h })
        dev->destroyBuffer(h);
}

// ---------------------------------------------------------------------------
// radixSortDoubleCopyHigh32Bits — two high-word (PxU32(key >> 32)) gathers.
//   out0[w] = uint32_t(in0[rank0[w]] >> 32);  out1[w] = uint32_t(in1[rank1[w]] >> 32);
// ---------------------------------------------------------------------------
PXG_TEST(particleContactSortHigh32Bits)
{
    PxgSlangComputeDevice* dev = testDevice();
    std::mt19937 rng(0x1234BEEF);

    const uint32_t CAP = 1024;
    const uint32_t numKeys = 555;
    const uint32_t SENT = 0xA5A5A5A5u;

    std::vector<uint64_t> inA(CAP), inB(CAP);
    for (uint32_t i = 0; i < CAP; ++i)
    {
        inA[i] = (uint64_t(rng()) << 32) | rng();
        inB[i] = (uint64_t(rng()) << 32) | rng();
    }
    std::vector<uint32_t> rankA(numKeys), rankB(numKeys);
    std::iota(rankA.begin(), rankA.end(), 0u); std::shuffle(rankA.begin(), rankA.end(), rng);
    std::iota(rankB.begin(), rankB.end(), 0u); std::shuffle(rankB.begin(), rankB.end(), rng);

    std::vector<uint32_t> out0(CAP, SENT), out1(CAP, SENT);
    std::vector<uint32_t> numBuf = { numKeys };

    DevBuf bInA = makeDev(dev, inA);
    DevBuf bInB = makeDev(dev, inB);
    DevBuf bRankA = makeDev(dev, rankA);
    DevBuf bRankB = makeDev(dev, rankB);
    DevBuf bOut0 = makeDev(dev, out0);
    DevBuf bOut1 = makeDev(dev, out1);
    DevBuf bNum = makeDev(dev, numBuf);

    DoubleCopyPush push = { bInA.addr, bOut0.addr, bRankA.addr,
                            bInB.addr, bOut1.addr, bRankB.addr, bNum.addr };

    // grid (32,1,1) block (1024,1,1) mirrors PxgParticleSystemCore.cpp:1207.
    Kernel k("particleContactSortBda.spv", "radixSortDoubleCopyHigh32Bits", {}, sizeof(DoubleCopyPush));
    k.dispatch(32u, {}, push, /*gridY*/ 1u, /*gridZ*/ 1u);
    dev->flush();

    const auto g0 = downloadBuffer<uint32_t>(dev, bOut0.h, CAP);
    const auto g1 = downloadBuffer<uint32_t>(dev, bOut1.h, CAP);

    size_t mm = 0;
    for (uint32_t w = 0; w < CAP; ++w)
    {
        const uint32_t e0 = (w < numKeys) ? uint32_t(inA[rankA[w]] >> 32) : SENT;
        const uint32_t e1 = (w < numKeys) ? uint32_t(inB[rankB[w]] >> 32) : SENT;
        if (g0[w] != e0) ++mm;
        if (g1[w] != e1) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "radixSortDoubleCopyHigh32Bits bit-exact vs CPU (%zu mismatches)", mm);

    for (PxgBufferHandle h : { bInA.h, bInB.h, bRankA.h, bRankB.h, bOut0.h, bOut1.h, bNum.h })
        dev->destroyBuffer(h);
}

// ---------------------------------------------------------------------------
// clampMaxValue — in-place clamp of one device counter.
//   if (*value > maxValue) *value = maxValue;
// ---------------------------------------------------------------------------
PXG_TEST(particleContactSortClampMaxValue)
{
    PxgSlangComputeDevice* dev = testDevice();

    struct ClampPush { uint64_t value; uint32_t maxValue; uint32_t pad; };
    const uint32_t maxValue = 4096u;

    // (initial value, expected after clamp)
    struct Case { uint32_t val, exp; };
    const Case cases[] = {
        { 9000u, maxValue },   // value > max  -> clamps to max
        { 4096u, 4096u },      // value == max -> unchanged
        { 12u,   12u },        // value <  max -> unchanged
        { 0u,    0u },         // zero (no contacts) -> unchanged
    };

    size_t mm = 0;
    for (const Case& c : cases)
    {
        std::vector<uint32_t> valBuf = { c.val };
        DevBuf bVal = makeDev(dev, valBuf);

        ClampPush push = { bVal.addr, maxValue, 0u };

        // grid (1,2,1) block (1,1,1) mirrors PxgParticleSystemCore.cpp:1105 (the
        // two groups clamp idempotently — same as the CUDA launch).
        Kernel k("particleContactSortBda.spv", "clampMaxValue", {}, sizeof(ClampPush));
        k.dispatch(1u, {}, push, /*gridY*/ 2u, /*gridZ*/ 1u);
        dev->flush();

        const auto g = downloadBuffer<uint32_t>(dev, bVal.h, 1);
        if (g[0] != c.exp)
        {
            ++mm;
            std::printf("  clampMaxValue MM: in=%u max=%u got=%u exp=%u\n", c.val, maxValue, g[0], c.exp);
        }
        dev->destroyBuffer(bVal.h);
    }
    PXG_CHECK_MSG(mm == 0, "clampMaxValue bit-exact vs CPU (%zu mismatches)", mm);
}

} // namespace
