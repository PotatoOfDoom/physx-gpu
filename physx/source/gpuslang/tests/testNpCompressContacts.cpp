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

// Verification of npCompressContacts.slang: compressContactStage1 (per-block count
// of pairs with contacts) and updateFrictionPatches (friction-patch offset
// rescale).  Both are compared against C++ references — the block count over the
// exact fixed work partition, and the rescaled offset per pair.

#include "pxgTestFramework.h"

#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct CompressParams { uint32_t numTotalPairs, pairCount, startContact, startFriction, contactSize, fricSize; };

const uint32_t OUTPUT_WORDS = 12;
const uint32_t BLOCK = 256, NUM_BLOCKS = 32;
const uint32_t CONTACT_SIZE = 64, FRIC_SIZE = 52; // sizeof(PxContactPatch)/PxFrictionPatch

PXG_TEST(npCompressContacts_stage1)
{
    std::mt19937 rng(46010);
    const uint32_t N = 9000;

    // PxsContactManagerOutput word view: word9 low16 = nbContacts
    std::vector<uint32_t> outData(N * OUTPUT_WORDS, 0);
    std::vector<uint32_t> nbRef(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        const uint32_t nb = (rng() % 3 == 0) ? 0u : (1u + rng() % 8u);
        nbRef[i] = nb;
        outData[i * OUTPUT_WORDS + 9] = (nb & 0xffffu) | ((rng() % 0xffff) << 16); // flags in high 16 = noise
        outData[i * OUTPUT_WORDS + 0] = rng(); // contactPatches (unused by stage1)
    }

    // reference: per-block count over the same work partition
    const uint32_t totalBlockRequired = (N + BLOCK - 1) / BLOCK;
    const uint32_t numIterationPerBlock = (totalBlockRequired + NUM_BLOCKS - 1) / NUM_BLOCKS;
    std::vector<uint32_t> refBlock(NUM_BLOCKS, 0);
    for (uint32_t b = 0; b < NUM_BLOCKS; ++b)
    {
        uint32_t cnt = 0;
        for (uint32_t it = 0; it < numIterationPerBlock; ++it)
            for (uint32_t tid = 0; tid < BLOCK; ++tid)
            {
                const uint32_t w = it * BLOCK + tid + numIterationPerBlock * b * BLOCK;
                if (w < N && nbRef[w] != 0) ++cnt;
            }
        refBlock[b] = cnt;
    }

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<uint32_t> blockInit(NUM_BLOCKS, 0);
    PxgBufferHandle bOut = createBufferFrom(dev, outData);
    PxgBufferHandle bBlock = createBufferFrom(dev, blockInit);

    CompressParams p = { N, 0, 0, 0, CONTACT_SIZE, FRIC_SIZE };
    Kernel k("npCompressContacts.spv", "compressContactStage1", { 4, 4 }, sizeof(CompressParams));
    k.dispatch(NUM_BLOCKS, { bOut, bBlock }, p);
    dev->flush();

    const auto g = downloadBuffer<uint32_t>(dev, bBlock, NUM_BLOCKS);
    size_t mm = 0;
    uint32_t total = 0;
    for (uint32_t b = 0; b < NUM_BLOCKS; ++b) { if (g[b] != refBlock[b]) ++mm; total += refBlock[b]; }
    PXG_CHECK_MSG(mm == 0, "compressContactStage1: %zu block mismatches", mm);
    PXG_CHECK_MSG(total > N / 4, "stage1: too few pairs with contacts (%u)", total);

    dev->destroyBuffer(bOut); dev->destroyBuffer(bBlock);
}

PXG_TEST(npCompressContacts_updateFrictionPatches)
{
    std::mt19937 rng(46020);
    const uint32_t N = 7000;
    const uint32_t startContact = 4096, startFriction = 100000;

    std::vector<uint32_t> outData(N * OUTPUT_WORDS, 0);
    std::vector<uint32_t> contactOff(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        // contactPatches = startContact + patchIndex*64 (a multiple of the patch size).
        // Every few entries, use a non-multiple-of-64 diff too so the multiply-before-
        // divide ordering is observable (a divide-first regression would differ here).
        const uint32_t off = (i % 5 == 0) ? (startContact + i * 32u + 16u) : (startContact + i * CONTACT_SIZE);
        contactOff[i] = off;
        outData[i * OUTPUT_WORDS + 0] = off;              // contactPatches
        outData[i * OUTPUT_WORDS + 6] = 0xdeadbeefu;      // frictionPatches (to be overwritten)
    }

    // reference
    std::vector<uint32_t> refFriction(N);
    for (uint32_t i = 0; i < N; ++i)
        refFriction[i] = startFriction + (contactOff[i] - startContact) * FRIC_SIZE / CONTACT_SIZE;

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<uint32_t> dummy(NUM_BLOCKS, 0);
    PxgBufferHandle bOut = createBufferFrom(dev, outData);
    PxgBufferHandle bDummy = createBufferFrom(dev, dummy);

    CompressParams p = { 0, N, startContact, startFriction, CONTACT_SIZE, FRIC_SIZE };
    const uint32_t GROUPS = (N + 255) / 256;
    Kernel k("npCompressContacts.spv", "updateFrictionPatches", { 4, 4 }, sizeof(CompressParams));
    k.dispatch(GROUPS, { bOut, bDummy }, p);
    dev->flush();

    const auto g = downloadBuffer<uint32_t>(dev, bOut, N * OUTPUT_WORDS);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
        if (g[i * OUTPUT_WORDS + 6] != refFriction[i]) ++mm;
    PXG_CHECK_MSG(mm == 0, "updateFrictionPatches: %zu mismatches", mm);

    dev->destroyBuffer(bOut); dev->destroyBuffer(bDummy);
}

} // namespace
