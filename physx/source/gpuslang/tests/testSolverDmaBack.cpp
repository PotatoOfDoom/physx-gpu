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

// Verification of solverDmaBack.slang: dmaBackChangedElems (word-wise copy of the
// 40-byte force-change threshold elements) and dmaConstraintResidual (gather the
// .w of each writeback's angularImpulse_residual float4).  Both are compared
// bit-exactly against C++ references; a preserved tail catches over-copying and a
// short grid stride forces the copy loop to iterate.

#include "pxgTestFramework.h"

#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct DmaParams { uint32_t nbElems, count, gridStride; };

PXG_TEST(solverDmaBack_changedElems)
{
    std::mt19937 rng(38010);
    const uint32_t nbElems = 733;
    const uint32_t ELEMENT_WORDS = 10;      // sizeof(Dy::ThresholdStreamElement)/4
    const uint32_t nbWords = ELEMENT_WORDS * nbElems;
    const uint32_t dstElems = nbWords + 2000; // preserve a tail past the copy

    std::vector<uint32_t> src(nbWords);
    for (auto& w : src) w = rng();
    std::vector<uint32_t> dstInit(dstElems, 0xcdcdcdcdu);

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = 3;
    const uint32_t gridStride = GROUPS * 256; // 768 < nbWords -> loop iterates
    std::vector<uint32_t> dummy(16, 0);

    PxgBufferHandle bSrc = createBufferFrom(dev, src);
    PxgBufferHandle bDst = createBufferFrom(dev, dstInit);
    PxgBufferHandle bDummy = createBufferFrom(dev, dummy);

    Kernel k("solverDmaBack.spv", "dmaBackChangedElems", { 4, 4, 32, 4 }, sizeof(DmaParams));
    DmaParams p = { nbElems, 0, gridStride };
    k.dispatch(GROUPS, { bSrc, bDst, bDummy, bDummy }, p);
    dev->flush();

    const auto g = downloadBuffer<uint32_t>(dev, bDst, dstElems);
    size_t mm = 0;
    for (uint32_t i = 0; i < dstElems; ++i)
    {
        const uint32_t exp = (i < nbWords) ? src[i] : 0xcdcdcdcdu;
        if (g[i] != exp) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "dmaBackChangedElems: %zu mismatches", mm);

    for (PxgBufferHandle h : { bSrc, bDst, bDummy }) dev->destroyBuffer(h);
}

PXG_TEST(solverDmaBack_constraintResidual)
{
    std::mt19937 rng(38020);
    const uint32_t count = 4000;

    // PxgConstraintWriteback = { float4 linearImpulse_broken; float4 angularImpulse_residual; }
    std::vector<float> writebacks(8 * count);
    auto rf = [&]() { return float(int(rng() % 20000) - 10000) * 0.01f; };
    for (uint32_t i = 0; i < count; ++i)
        for (uint32_t c = 0; c < 8; ++c) writebacks[8 * i + c] = rf();

    std::vector<float> residInit(count + 500, -123.f); // preserve a tail

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (count + 255) / 256; // one thread per constraint
    std::vector<uint32_t> dummy(16, 0);

    PxgBufferHandle bWb = createBufferFrom(dev, writebacks);
    PxgBufferHandle bResid = createBufferFrom(dev, residInit);
    PxgBufferHandle bDummy = createBufferFrom(dev, dummy);

    Kernel k("solverDmaBack.spv", "dmaConstraintResidual", { 4, 4, 32, 4 }, sizeof(DmaParams));
    DmaParams p = { 0, count, GROUPS * 256 };
    k.dispatch(GROUPS, { bDummy, bDummy, bWb, bResid }, p);
    dev->flush();

    const auto g = downloadBuffer<float>(dev, bResid, count + 500);
    size_t mm = 0;
    for (uint32_t i = 0; i < count + 500; ++i)
    {
        const float exp = (i < count) ? writebacks[8 * i + 7] : -123.f; // angularImpulse_residual.w
        if (g[i] != exp) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "dmaConstraintResidual: %zu mismatches", mm);

    for (PxgBufferHandle h : { bWb, bResid, bDummy }) dev->destroyBuffer(h);
}

} // namespace
