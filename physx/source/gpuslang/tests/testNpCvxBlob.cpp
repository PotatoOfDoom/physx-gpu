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

// Verification of npCvxBlob.slang: the convex-hull blob bit-field accessors and
// sub-array byte-offset arithmetic.  All operations are pure integer, so the
// reference matches the shader EXACTLY (no tolerance, no marginal skips).

#include "pxgTestFramework.h"

#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct BlobIn { uint32_t word, nbV, nbP, nbE; };
struct BlobOut
{
    uint32_t vref8, nbVerts, minIndex, lo16, valCount, valOffset, valMerged, word3merged,
             planesOff, polyDescOff, vertexDataOff, pad0;
};

// exact transcription of the CUDA bit helpers
static uint32_t u32High(uint32_t v) { return (v >> 16) & 0xffffu; }
static uint32_t u32Low(uint32_t v) { return v & 0xffffu; }
static uint32_t u16High(uint32_t v) { return (v >> 8) & 0xffu; }
static uint32_t u16Low(uint32_t v) { return v & 0xffu; }
static uint32_t merge16(uint32_t hi, uint32_t lo) { return ((hi & 0xffffu) << 16) | (lo & 0xffffu); }
static uint32_t merge8(uint32_t hi, uint32_t lo) { return ((hi & 0xffu) << 8) | (lo & 0xffu); }

PXG_TEST(npCvxBlob_accessors)
{
    std::mt19937 rng(141001);
    const uint32_t N = 20000;

    std::vector<BlobIn> in(N);
    std::vector<BlobOut> ref(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        BlobIn s;
        s.word = rng();
        // realistic small hull dimensions, plus some large stress values
        if (i % 5 == 0) { s.nbV = rng() % 100000; s.nbP = rng() % 100000; s.nbE = rng() % 100000; }
        else { s.nbV = rng() % 256; s.nbP = rng() % 256; s.nbE = rng() % 512; }
        in[i] = s;

        BlobOut r{};
        const uint32_t lo16 = u32Low(s.word), hi16 = u32High(s.word);
        const uint32_t midByte = u16High(lo16), loByte = u16Low(lo16);
        r.vref8 = u32High(s.word);
        r.nbVerts = u16High(u32Low(s.word));
        r.minIndex = u16Low(u32Low(s.word));
        r.lo16 = lo16;
        r.valCount = u32High(s.word);
        r.valOffset = u32Low(s.word);
        r.valMerged = merge16(u32High(s.word), u32Low(s.word));
        r.word3merged = merge16(hi16, merge8(midByte, loByte));
        r.planesOff = 32u + 16u * s.nbV;
        r.polyDescOff = 32u + 16u * s.nbV + 16u * s.nbP;
        r.vertexDataOff = 32u + 16u * s.nbV + 20u * s.nbP + 6u * s.nbE + 3u * s.nbV;
        ref[i] = r;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<BlobOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct BlobParams { uint32_t numTests; } prm = { N };
    Kernel k("npCvxBlob.spv", "cvxBlob", { (uint32_t)sizeof(BlobIn), (uint32_t)sizeof(BlobOut) }, sizeof(BlobParams));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<BlobOut>(dev, bOut, N);
    size_t mm = 0;
    size_t rt = 0; // round-trip sanity: valMerged/word3merged must reconstruct word
    for (uint32_t i = 0; i < N; ++i)
    {
        const BlobOut& a = ref[i];
        const BlobOut& b = g[i];
        if (a.vref8 != b.vref8 || a.nbVerts != b.nbVerts || a.minIndex != b.minIndex || a.lo16 != b.lo16 ||
            a.valCount != b.valCount || a.valOffset != b.valOffset || a.valMerged != b.valMerged ||
            a.word3merged != b.word3merged || a.planesOff != b.planesOff || a.polyDescOff != b.polyDescOff ||
            a.vertexDataOff != b.vertexDataOff)
            ++mm;
        if (b.valMerged == in[i].word && b.word3merged == in[i].word) ++rt;
    }
    PXG_CHECK_MSG(mm == 0, "blob: %zu mismatches", mm);
    PXG_CHECK_MSG(rt == N, "blob: %zu/%u round-trips reconstructed the word", rt, N);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
