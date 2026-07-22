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

// P6.4b-1: the contact-ORGANIZATION half of the PBD particle/primitive consume
// path on the BDA engine ABI.  Three bit-exact unit tests:
//
//  (a) psContactSort_plainRadix — the PLAIN radix pair radixSortMultiBlockLaunch /
//      radixSortMultiCalculateRanksLaunch (radixSortImpl.cu:39 / :53).  Unlike the
//      WithCount (numKeys push param) / WithoutCount (desc.count field) twins, the
//      plain variant DEREFERENCES numKeys from a PxgRadixSortBlockDesc::numKeys
//      DEVICE POINTER (@48).  Mirrors PxgParticleSystemCore.cpp:1141: grid
//      (RADIX_SORT, 2, 1) with blockIdx.y selecting one of TWO block descs, 8
//      passes with startBit += 4 and a ping-pong of the desc buffer per odd pass.
//      Sorted result compared bitwise vs a CPU stable sort (both channels).
//  (b) psContactSort_reorder — ps_reorderPrimitiveContactsLaunch gathers the two
//      sorted contact arrays through two independent rank permutations; checked
//      bit-exact vs the CPU gather.
//  (c) psContactSort_findStartEnd — the 2-pass ps_findStartEndParticle{First,
//      Second} produces each particle's [start,end) contact range + the pair
//      count for a known sorted-by-particle contact set; checked bit-exact.

#include "pxgTestFramework.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
// PxgParticlePrimitiveContact (PxgParticleSystem.h, 32 bytes).
struct PsContact { F4 normal_pen; uint32_t rigidLo, rigidHi, particleLo, particleHi; };
static_assert(sizeof(PsContact) == 32, "PsContact stride");

static uint64_t addrOf(PxgSlangComputeDevice* dev, PxgBufferHandle h)
{
    return dev->getBufferDeviceAddress(h);
}

// ===========================================================================
// (a) PLAIN radix pair — numKeys from *desc.numKeys device pointer.
// ===========================================================================
struct RadixBlockDescMirror
{
    uint64_t inputKeys;         // @0
    uint64_t inputRanks;        // @8
    uint64_t outputKeys;        // @16
    uint64_t outputRanks;       // @24
    uint64_t radixBlockCounts;  // @32
    uint32_t count;             // @40 (unused by the plain variant)
    uint32_t pad;               // @44
    uint64_t numKeys;           // @48 PxU32* device pointer
};
static_assert(sizeof(RadixBlockDescMirror) == 56, "block desc mirror must be 56 bytes");

PXG_TEST(psContactSort_plainRadix)
{
    std::mt19937 rng(0xC0FFEE);
    const uint32_t N = 6000;                 // spans many blocks, odd-ish tail
    const uint32_t RADIX_BLOCKS = 32, RADIX_SIZE = 16, CH = 2; // 2 blockIdx.y channels

    PxgSlangComputeDevice* dev = testDevice();

    // per-channel keys (32-bit) + identity ranks
    std::vector<std::vector<uint32_t>> keys(CH), ranks(CH);
    for (uint32_t c = 0; c < CH; ++c)
    {
        keys[c].resize(N);
        ranks[c].resize(N);
        for (uint32_t i = 0; i < N; ++i) { keys[c][i] = rng(); ranks[c][i] = i; }
    }

    // ping-pong buffers per channel + per-channel histogram + per-channel numKeys.
    PxgBufferHandle dKeys[CH][2], dRanks[CH][2], dCounts[CH], dNumKeys[CH];
    for (uint32_t c = 0; c < CH; ++c)
    {
        for (uint32_t s = 0; s < 2; ++s)
        {
            dKeys[c][s] = createBufferFrom(dev, s == 0 ? keys[c] : std::vector<uint32_t>(N, 0));
            dRanks[c][s] = createBufferFrom(dev, s == 0 ? ranks[c] : std::vector<uint32_t>(N, 0));
        }
        dCounts[c] = createBufferFrom(dev, std::vector<uint32_t>(RADIX_SIZE * RADIX_BLOCKS, 0));
        dNumKeys[c] = createBufferFrom(dev, std::vector<uint32_t>(1, N));
    }
    if (!addrOf(dev, dKeys[0][0])) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    // TWO desc buffers (even/odd ping-pong), each holding CH block descs.
    RadixBlockDescMirror even[CH], odd[CH];
    std::memset(even, 0, sizeof(even));
    std::memset(odd, 0, sizeof(odd));
    for (uint32_t c = 0; c < CH; ++c)
    {
        even[c] = { addrOf(dev, dKeys[c][0]), addrOf(dev, dRanks[c][0]),
                    addrOf(dev, dKeys[c][1]), addrOf(dev, dRanks[c][1]),
                    addrOf(dev, dCounts[c]), 0u, 0u, addrOf(dev, dNumKeys[c]) };
        odd[c]  = { addrOf(dev, dKeys[c][1]), addrOf(dev, dRanks[c][1]),
                    addrOf(dev, dKeys[c][0]), addrOf(dev, dRanks[c][0]),
                    addrOf(dev, dCounts[c]), 0u, 0u, addrOf(dev, dNumKeys[c]) };
    }
    std::vector<RadixBlockDescMirror> evenV(even, even + CH), oddV(odd, odd + CH);
    PxgBufferHandle dDescEven = createBufferFrom(dev, evenV);
    PxgBufferHandle dDescOdd = createBufferFrom(dev, oddV);

    Kernel kHist("radixSortDesc.spv", "radixSortMultiBlockLaunch", {}, 16u);
    Kernel kRanks("radixSortDesc.spv", "radixSortMultiCalculateRanksLaunch", {}, 16u);

    struct { uint64_t desc; uint32_t startBit; uint32_t pad; } prm;
    for (uint32_t pass = 0; pass < 8; ++pass)
    {
        prm.desc = addrOf(dev, (pass & 1) ? dDescOdd : dDescEven);
        prm.startBit = pass * 4;
        prm.pad = 0;
        kHist.dispatch(RADIX_BLOCKS, {}, prm, CH, 1u);
        kRanks.dispatch(RADIX_BLOCKS, {}, prm, CH, 1u);
    }
    dev->flush();

    // after 8 passes the result is back in slot 0
    size_t bad = 0;
    for (uint32_t c = 0; c < CH; ++c)
    {
        std::vector<uint32_t> order(N);
        std::iota(order.begin(), order.end(), 0u);
        std::stable_sort(order.begin(), order.end(),
                         [&](uint32_t a, uint32_t b) { return keys[c][a] < keys[c][b]; });

        const auto gk = downloadBuffer<uint32_t>(dev, dKeys[c][0], N);
        const auto gr = downloadBuffer<uint32_t>(dev, dRanks[c][0], N);
        for (uint32_t i = 0; i < N; ++i)
        {
            if (gk[i] != keys[c][order[i]]) ++bad;
            if (gr[i] != order[i]) ++bad; // identity ranks -> rank == original index (stability)
        }
    }
    PXG_CHECK_MSG(bad == 0, "plain radix 8-pass 2-channel sort bitwise-correct (%zu bad)", bad);

    for (uint32_t c = 0; c < CH; ++c)
    {
        for (uint32_t s = 0; s < 2; ++s) { dev->destroyBuffer(dKeys[c][s]); dev->destroyBuffer(dRanks[c][s]); }
        dev->destroyBuffer(dCounts[c]); dev->destroyBuffer(dNumKeys[c]);
    }
    dev->destroyBuffer(dDescEven); dev->destroyBuffer(dDescOdd);
}

// ===========================================================================
// (b) ps_reorderPrimitiveContactsLaunch — dual rank-indexed gather.
// ===========================================================================
PXG_TEST(psContactSort_reorder)
{
    std::mt19937 rng(0xBEEF01);
    PxgSlangComputeDevice* dev = testDevice();

    const uint32_t NC = 5000; // multiple device blocks / iterations
    std::vector<PsContact> contacts(NC);
    for (uint32_t i = 0; i < NC; ++i)
    {
        contacts[i].normal_pen = { float(rng()), float(rng()), float(rng()), float(rng()) };
        contacts[i].rigidLo = rng(); contacts[i].rigidHi = rng();
        contacts[i].particleLo = rng(); contacts[i].particleHi = rng();
    }
    std::vector<uint32_t> remapByRigid(NC), remapByParticle(NC);
    std::iota(remapByRigid.begin(), remapByRigid.end(), 0u);
    std::iota(remapByParticle.begin(), remapByParticle.end(), 0u);
    std::shuffle(remapByRigid.begin(), remapByRigid.end(), rng);
    std::shuffle(remapByParticle.begin(), remapByParticle.end(), rng);

    PxgBufferHandle bContacts = createBufferFrom(dev, contacts);
    PxgBufferHandle bNum = createBufferFrom(dev, std::vector<uint32_t>(1, NC));
    PxgBufferHandle bRemapR = createBufferFrom(dev, remapByRigid);
    PxgBufferHandle bRemapP = createBufferFrom(dev, remapByParticle);
    PxgBufferHandle bOutR = createBufferFrom(dev, std::vector<PsContact>(NC));
    PxgBufferHandle bOutP = createBufferFrom(dev, std::vector<PsContact>(NC));
    if (!addrOf(dev, bContacts)) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    struct ReorderPush
    {
        uint64_t particleContacts, numParticleContacts, remapByRigid, remapByParticle,
            sortedContactsByRigid, sortedContactsByParticle;
    };
    ReorderPush push = { addrOf(dev, bContacts), addrOf(dev, bNum), addrOf(dev, bRemapR),
                         addrOf(dev, bRemapP), addrOf(dev, bOutR), addrOf(dev, bOutP) };
    Kernel k("psContactSortBda.spv", "ps_reorderPrimitiveContactsLaunch", {}, sizeof(ReorderPush));
    k.dispatch(32u, {}, push); // grid BOUNDCELLUPDATE
    dev->flush();

    const auto gR = downloadBuffer<PsContact>(dev, bOutR, NC);
    const auto gP = downloadBuffer<PsContact>(dev, bOutP, NC);
    size_t bad = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        if (std::memcmp(&gR[i], &contacts[remapByRigid[i]], sizeof(PsContact)) != 0) ++bad;
        if (std::memcmp(&gP[i], &contacts[remapByParticle[i]], sizeof(PsContact)) != 0) ++bad;
    }
    PXG_CHECK_MSG(bad == 0, "reorder dual gather bit-exact (%zu mismatches)", bad);

    dev->destroyBuffer(bContacts); dev->destroyBuffer(bNum); dev->destroyBuffer(bRemapR);
    dev->destroyBuffer(bRemapP); dev->destroyBuffer(bOutR); dev->destroyBuffer(bOutP);
}

// ===========================================================================
// (c) ps_findStartEndParticle{First,Second} — per-particle [start,end) ranges.
// ===========================================================================
PXG_TEST(psContactSort_findStartEnd)
{
    std::mt19937 rng(0xF1D5E7);
    PxgSlangComputeDevice* dev = testDevice();

    const uint32_t BLOCK = 512u, GRID = 32u;

    // Build a sorted-by-particle contact set: consecutive runs of equal 64-bit
    // particle ids (distinct per group, exercising both halves of the id).  ~40
    // device blocks -> 2 iterations per block, boundary handling exercised.
    std::vector<PsContact> contacts;
    std::vector<uint64_t> groupId;   // per group
    std::uniform_int_distribution<uint32_t> runLen(1u, 9u);
    uint32_t g = 0;
    while (contacts.size() < 20000)
    {
        const uint64_t pid = (uint64_t(g) << 20) | uint64_t(0x50000u + g * 2654435761u & 0xffffful);
        groupId.push_back(pid);
        const uint32_t len = runLen(rng);
        for (uint32_t r = 0; r < len; ++r)
        {
            PsContact ct;
            ct.normal_pen = { float(rng()), float(rng()), float(rng()), float(rng()) };
            ct.rigidLo = rng(); ct.rigidHi = rng();
            ct.particleLo = uint32_t(pid & 0xffffffffu);
            ct.particleHi = uint32_t(pid >> 32);
            contacts.push_back(ct);
        }
        ++g;
    }
    const uint32_t NC = uint32_t(contacts.size());

    // CPU reference: offsets (0-based group index of each contact), per-group
    // [start,end), total pair count.
    std::vector<uint32_t> refOffsets(NC), refStart, refEnd;
    {
        uint32_t gc = 0; // groups seen so far
        for (uint32_t i = 0; i < NC; ++i)
        {
            const bool isStart = (i == 0) ||
                contacts[i].particleLo != contacts[i - 1].particleLo ||
                contacts[i].particleHi != contacts[i - 1].particleHi;
            if (isStart)
            {
                if (gc > 0) refEnd.push_back(i);
                refStart.push_back(i);
                ++gc;
            }
            refOffsets[i] = gc - 1;
        }
        refEnd.push_back(NC);
    }
    const uint32_t totalPairs = uint32_t(refStart.size());

    PxgBufferHandle bContacts = createBufferFrom(dev, contacts);
    PxgBufferHandle bNum = createBufferFrom(dev, std::vector<uint32_t>(1, NC));
    PxgBufferHandle bBlockOff = createBufferFrom(dev, std::vector<uint32_t>(GRID, 0));
    PxgBufferHandle bOffsets = createBufferFrom(dev, std::vector<uint32_t>(NC, 0xffffffffu));
    PxgBufferHandle bPairCount = createBufferFrom(dev, std::vector<uint32_t>(1, 0));
    PxgBufferHandle bStart = createBufferFrom(dev, std::vector<uint32_t>(NC, 0xffffffffu));
    PxgBufferHandle bEnd = createBufferFrom(dev, std::vector<uint32_t>(NC, 0xffffffffu));
    if (!addrOf(dev, bContacts)) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    // pass 1: particleSystems ptr is ABI-parity only (never dereferenced) -> 0.
    struct Find1Push
    {
        uint64_t particleSystems, sortedContacts, numContacts, blockOffsets, offsets;
    };
    Find1Push p1 = { 0ull, addrOf(dev, bContacts), addrOf(dev, bNum),
                     addrOf(dev, bBlockOff), addrOf(dev, bOffsets) };
    Kernel k1("psContactSortBda.spv", "ps_findStartEndParticleFirstLaunch", {}, sizeof(Find1Push));
    k1.dispatch(GRID, {}, p1);
    dev->flush();

    const auto gOffsets = downloadBuffer<uint32_t>(dev, bOffsets, NC);
    size_t badOff = 0;
    for (uint32_t i = 0; i < NC; ++i)
        if (gOffsets[i] != refOffsets[i]) ++badOff;
    PXG_CHECK_MSG(badOff == 0, "findStartEnd pass1 offsets bit-exact (%zu mismatches)", badOff);

    // pass 2: per-particle [start,end) + total pair count.
    struct Find2Push
    {
        uint64_t sortedContacts, numContacts, blockOffsets, offsets, totalNumPairs,
            rangeStart, rangeEnd;
    };
    Find2Push p2 = { addrOf(dev, bContacts), addrOf(dev, bNum), addrOf(dev, bBlockOff),
                     addrOf(dev, bOffsets), addrOf(dev, bPairCount),
                     addrOf(dev, bStart), addrOf(dev, bEnd) };
    Kernel k2("psContactSortBda.spv", "ps_findStartEndParticleSecondLaunch", {}, sizeof(Find2Push));
    k2.dispatch(GRID, {}, p2);
    dev->flush();

    const uint32_t gPairs = downloadBuffer<uint32_t>(dev, bPairCount, 1)[0];
    const auto gStart = downloadBuffer<uint32_t>(dev, bStart, NC);
    const auto gEnd = downloadBuffer<uint32_t>(dev, bEnd, NC);
    PXG_CHECK_MSG(gPairs == totalPairs, "findStartEnd pair count %u != %u", gPairs, totalPairs);
    size_t badRange = 0;
    for (uint32_t p = 0; p < totalPairs; ++p)
    {
        if (gStart[p] != refStart[p]) ++badRange;
        if (gEnd[p] != refEnd[p]) ++badRange;
    }
    PXG_CHECK_MSG(badRange == 0, "findStartEnd [start,end) ranges bit-exact (%zu mismatches)", badRange);
    PXG_CHECK_MSG(totalPairs > 1500 && NC > 512 * 32 / BLOCK, // real multi-block coverage
                  "findStartEnd coverage thin (pairs=%u NC=%u)", totalPairs, NC);

    dev->destroyBuffer(bContacts); dev->destroyBuffer(bNum); dev->destroyBuffer(bBlockOff);
    dev->destroyBuffer(bOffsets); dev->destroyBuffer(bPairCount);
    dev->destroyBuffer(bStart); dev->destroyBuffer(bEnd);
}

} // namespace
