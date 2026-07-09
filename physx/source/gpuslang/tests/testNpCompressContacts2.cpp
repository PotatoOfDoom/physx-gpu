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

// Verification of npCompressContacts2.slang: compressContactStage2 — the
// cross-block compaction of pairs-with-contacts into the dense PxGpuContactPair
// list.  The per-block survivor counts (gBlockNumPairs) are built from the exact
// Stage1 work partition, then the reference reproduces the stable compaction (in
// work order) and the pointer->offset / remap / actor-table field conversions.
// Everything is integer, so the comparison is bit-exact (no tolerance).

#include "pxgTestFramework.h"

#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

const uint32_t OUTPUT_WORDS = 12;
const uint32_t BLOCK = 256, NUM_BLOCKS = 32;
const uint32_t CONTACT_SIZE = 64, FRIC_SIZE = 52;
const uint32_t TABLE = 4096;

struct Compress2Params { uint32_t numTotalPairs, maxOutputPatches, contactSize, fricSize; };
struct GpuPairOut { uint32_t ref0, ref1, node0, node1, forceOff, fricOff, patchOff, pointOff, nbPatches, nbContacts, actor0, actor1; };

PXG_TEST(npCompressContacts2_stage2)
{
    std::mt19937 rng(53010);
    const uint32_t N = 40000;

    std::vector<uint32_t> outData(N * OUTPUT_WORDS, 0);
    std::vector<uint32_t> refs(N * 2);
    std::vector<uint32_t> nb(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        const uint32_t nbc = (rng() % 100 < 45) ? 0u : (1u + rng() % 8u);
        nb[i] = nbc;
        outData[i * OUTPUT_WORDS + 0] = (rng() % 20000) * 64u + (rng() % 64); // contactPatches offset
        outData[i * OUTPUT_WORDS + 2] = rng();                                // contactPoints offset
        outData[i * OUTPUT_WORDS + 4] = rng();                                // contactForces offset (low)
        outData[i * OUTPUT_WORDS + 5] = (rng() % 100 < 20) ? 0u : 1u;          // presence (0 == null)
        outData[i * OUTPUT_WORDS + 8] = (rng() % 0xff) << 8 | (rng() & 0xff) | ((rng() & 0xffff) << 16); // nbPatches in bits 8-15
        outData[i * OUTPUT_WORDS + 9] = (nbc & 0xffffu) | ((rng() % 0xffff) << 16); // flags noise in high 16
        refs[i * 2 + 0] = rng() % TABLE;
        refs[i * 2 + 1] = rng() % TABLE;
    }
    std::vector<uint32_t> remap(TABLE), actorTab(TABLE);
    for (uint32_t i = 0; i < TABLE; ++i) { remap[i] = rng(); actorTab[i] = rng(); }

    // gBlockNumPairs: per-block survivor count over the Stage1 partition
    const uint32_t totalBlockRequired = (N + BLOCK - 1) / BLOCK;
    const uint32_t numIter = (totalBlockRequired + NUM_BLOCKS - 1) / NUM_BLOCKS;
    std::vector<uint32_t> blockNum(NUM_BLOCKS, 0);
    for (uint32_t b = 0; b < NUM_BLOCKS; ++b)
    {
        uint32_t cnt = 0;
        for (uint32_t it = 0; it < numIter; ++it)
            for (uint32_t tid = 0; tid < BLOCK; ++tid)
            {
                const uint32_t w = it * BLOCK + tid + numIter * b * BLOCK;
                if (w < N && nb[w] != 0) ++cnt;
            }
        blockNum[b] = cnt;
    }

    // reference stable compaction (work order) + field conversions
    std::vector<uint32_t> blockStart(NUM_BLOCKS, 0);
    uint32_t acc = 0;
    for (uint32_t b = 0; b < NUM_BLOCKS; ++b) { blockStart[b] = acc; acc += blockNum[b]; }
    const uint32_t numPairs = acc;

    std::vector<GpuPairOut> refOut(numPairs);
    for (uint32_t b = 0; b < NUM_BLOCKS; ++b)
    {
        uint32_t rank = 0;
        for (uint32_t it = 0; it < numIter; ++it)
            for (uint32_t tid = 0; tid < BLOCK; ++tid)
            {
                const uint32_t w = it * BLOCK + tid + numIter * b * BLOCK;
                if (w >= N || nb[w] == 0) continue;
                const uint32_t index = blockStart[b] + rank; ++rank;
                const uint32_t w0 = w * OUTPUT_WORDS;
                const uint32_t patch = outData[w0 + 0], point = outData[w0 + 2];
                const uint32_t fOff = outData[w0 + 4], fPres = outData[w0 + 5];
                GpuPairOut o{};
                o.ref0 = refs[w * 2 + 0]; o.ref1 = refs[w * 2 + 1];
                o.node0 = remap[o.ref0]; o.node1 = remap[o.ref1];
                o.forceOff = fPres ? fOff : 0xFFFFFFFFu;
                o.fricOff = patch / CONTACT_SIZE * FRIC_SIZE;
                o.patchOff = patch; o.pointOff = point;
                o.nbPatches = (outData[w0 + 8] >> 8) & 0xff;
                o.nbContacts = outData[w0 + 9] & 0xffff;
                o.actor0 = actorTab[o.ref0]; o.actor1 = actorTab[o.ref1];
                refOut[index] = o;
            }
    }

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<GpuPairOut> outInit(numPairs + 16);
    std::vector<uint32_t> npInit(1, 0);
    // uint2 refs buffer
    std::vector<uint32_t> refBuf = refs; // already interleaved (ref0,ref1) per pair
    PxgBufferHandle bBlock = createBufferFrom(dev, blockNum);
    PxgBufferHandle bRefs = createBufferFrom(dev, refBuf);
    PxgBufferHandle bOut = createBufferFrom(dev, outData);
    PxgBufferHandle bRemap = createBufferFrom(dev, remap);
    PxgBufferHandle bActor = createBufferFrom(dev, actorTab);
    PxgBufferHandle bPairs = createBufferFrom(dev, outInit);
    PxgBufferHandle bNum = createBufferFrom(dev, npInit);

    Compress2Params p = { N, numPairs + 16, CONTACT_SIZE, FRIC_SIZE };
    Kernel k("npCompressContacts2.spv", "compressContactStage2",
             { 4, 8, 4, 4, 4, (uint32_t)sizeof(GpuPairOut), 4 }, sizeof(Compress2Params));
    k.dispatch(NUM_BLOCKS, { bBlock, bRefs, bOut, bRemap, bActor, bPairs, bNum }, p);
    dev->flush();

    const auto gPairs = downloadBuffer<GpuPairOut>(dev, bPairs, numPairs);
    const auto gNum = downloadBuffer<uint32_t>(dev, bNum, 1);

    size_t mm = 0;
    for (uint32_t i = 0; i < numPairs; ++i)
    {
        const GpuPairOut& r = refOut[i]; const GpuPairOut& v = gPairs[i];
        if (r.ref0 != v.ref0 || r.ref1 != v.ref1 || r.node0 != v.node0 || r.node1 != v.node1 ||
            r.forceOff != v.forceOff || r.fricOff != v.fricOff || r.patchOff != v.patchOff ||
            r.pointOff != v.pointOff || r.nbPatches != v.nbPatches || r.nbContacts != v.nbContacts ||
            r.actor0 != v.actor0 || r.actor1 != v.actor1) ++mm;
    }
    PXG_CHECK_MSG(gNum[0] == numPairs, "stage2: numPairs %u != %u", gNum[0], numPairs);
    PXG_CHECK_MSG(mm == 0, "compressContactStage2: %zu compacted-pair mismatches", mm);
    PXG_CHECK_MSG(numPairs > N / 4 && numPairs < N, "stage2: survivor count implausible (%u/%u)", numPairs, N);

    for (PxgBufferHandle h : { bBlock, bRefs, bOut, bRemap, bActor, bPairs, bNum }) dev->destroyBuffer(h);
}

} // namespace
