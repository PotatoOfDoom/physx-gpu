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

// Verification of npContactPatch.slang: setContactPointAndForcePointers and
// registerContactPatch.  The per-pair status-flag bit-packing and touch/patch
// change flags are deterministic and compared against a C++ reference; the atomic
// byte/patch reservations are verified via partition invariants (unique offsets
// that tile [0, total) exactly, counters at the correct totals).  Buffer limits
// are set large so no overflow occurs (the deterministic, verifiable regime).

#include "pxgTestFramework.h"

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct ContactPatchParams { uint32_t numTests, contactSize, forceSize, patchSize, contactBytesLimit, forceBytesLimit, patchBytesLimit; };

const uint32_t INVALID = 0xffffffffu;
const uint32_t FLAG_HAS_NO_TOUCH = 1, FLAG_HAS_TOUCH = 2, FLAG_TOUCH_KNOWN = 3;

uint32_t u32High(uint32_t v) { return (v >> 16) & 0xffffu; }
uint32_t u32Low(uint32_t v) { return v & 0xffffu; }
uint32_t u16High(uint32_t v) { return (v >> 8) & 0xffu; }
uint32_t u16Low(uint32_t v) { return v & 0xffu; }
uint32_t mergeU16(uint32_t hi, uint32_t lo) { return ((hi & 0xffffu) << 16) | (lo & 0xffffu); }
uint32_t mergeU8(uint32_t hi, uint32_t lo) { return ((hi & 0xffu) << 8) | (lo & 0xffu); }

// verify offsets[] over active indices form a valid atomic partition: each active
// offset unique, a multiple of 'unit'*nb, tiling [0, total) with no gap/overlap.
bool checkPartition(const std::vector<uint32_t>& offsets, const std::vector<uint32_t>& nb,
                    uint32_t unit, uint32_t total)
{
    std::vector<std::pair<uint32_t, uint32_t>> active; // (offset, bytes)
    for (uint32_t i = 0; i < offsets.size(); ++i)
        if (nb[i] != 0) active.push_back({ offsets[i], unit * nb[i] });
    std::sort(active.begin(), active.end());
    uint32_t run = 0;
    for (auto& a : active)
    {
        if (a.first != run) return false;
        run += a.second;
    }
    return run == total;
}

PXG_TEST(npContactPatch_setAndRegister)
{
    std::mt19937 rng(44010);
    const uint32_t N = 6000;
    const uint32_t contactSize = 16, forceSize = 4, patchSize = 64;

    std::vector<uint32_t> nbContacts(N), priorAllflags(N);
    uint32_t sumNb = 0, nbActive = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const uint32_t nb = (rng() % 3 == 0) ? 0u : (1u + rng() % 6u); // ~1/3 with no contact
        nbContacts[i] = nb;
        sumNb += nb;
        if (nb) ++nbActive;
        // random prior packed status word: byte1=nbPatches(0/1), byte2=statusFlag(random), byte3=prevPatches
        const uint32_t statusFlag = rng() % 128;      // various status bits incl. touch-known combos
        const uint32_t nbPatches = rng() % 2;
        const uint32_t prevP = rng() % 2;
        priorAllflags[i] = (prevP << 24) | (statusFlag << 16) | (nbPatches << 8) | (rng() % 256);
    }
    const uint32_t contactTotal = contactSize * sumNb;
    const uint32_t forceTotal = forceSize * sumNb;
    const uint32_t patchTotal = patchSize * nbActive;

    // ---- reference: deterministic flags + allflags for registerContactPatch ----
    std::vector<uint32_t> refAllflags(N), refTouch(N), refPatchChange(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        const uint32_t nb = nbContacts[i], allflags = priorAllflags[i];
        const uint32_t oldStatus = u16Low(u32High(allflags));
        uint32_t statusFlags = oldStatus & (~FLAG_TOUCH_KNOWN);
        statusFlags |= (nb != 0) ? FLAG_HAS_TOUCH : FLAG_HAS_NO_TOUCH;
        const uint32_t prevPatches = u16High(u32Low(allflags));
        const bool prevHadTouch = (oldStatus & FLAG_HAS_TOUCH) != 0;
        const bool prevTouchKnown = (oldStatus & FLAG_TOUCH_KNOWN) != 0;
        const uint32_t numPatches = (nb > 0) ? 1u : 0u;
        const bool currHasTouch = nb > 0;
        const bool change = (prevHadTouch != currHasTouch) || (!prevTouchKnown);
        refTouch[i] = change ? 1u : 0u;
        refPatchChange[i] = (prevPatches != numPatches) ? 1u : 0u;
        refAllflags[i] = mergeU16(mergeU8(prevPatches, statusFlags), mergeU8(numPatches, 0u));
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    const std::vector<uint32_t> strides(9, 4);
    ContactPatchParams p = { N, contactSize, forceSize, patchSize, 1u << 30, 1u << 30, 1u << 30 };

    std::vector<uint32_t> counters(4, 0), outAllflags(N, 0), touch(N, 9), patchCh(N, 9),
        cOff(N, 7), fOff(N, 7), pIdx(N, 7);

    PxgBufferHandle bNb = createBufferFrom(dev, nbContacts);
    PxgBufferHandle bCnt = createBufferFrom(dev, counters);
    PxgBufferHandle bPrior = createBufferFrom(dev, priorAllflags);
    PxgBufferHandle bOut = createBufferFrom(dev, outAllflags);
    PxgBufferHandle bTouch = createBufferFrom(dev, touch);
    PxgBufferHandle bPatchCh = createBufferFrom(dev, patchCh);
    PxgBufferHandle bCOff = createBufferFrom(dev, cOff);
    PxgBufferHandle bFOff = createBufferFrom(dev, fOff);
    PxgBufferHandle bPIdx = createBufferFrom(dev, pIdx);
    const std::vector<PxgBufferHandle> binds = { bNb, bCnt, bPrior, bOut, bTouch, bPatchCh, bCOff, bFOff, bPIdx };

    Kernel kSet("npContactPatch.spv", "setContactPointAndForcePointers", strides, sizeof(ContactPatchParams));
    kSet.dispatch(GROUPS, binds, p); dev->flush();
    Kernel kReg("npContactPatch.spv", "registerContactPatch", strides, sizeof(ContactPatchParams));
    kReg.dispatch(GROUPS, binds, p); dev->flush();

    const auto gCounters = downloadBuffer<uint32_t>(dev, bCnt, 4);
    const auto gAllflags = downloadBuffer<uint32_t>(dev, bOut, N);
    const auto gTouch = downloadBuffer<uint32_t>(dev, bTouch, N);
    const auto gPatchCh = downloadBuffer<uint32_t>(dev, bPatchCh, N);
    const auto gCOff = downloadBuffer<uint32_t>(dev, bCOff, N);
    const auto gFOff = downloadBuffer<uint32_t>(dev, bFOff, N);
    const auto gPIdx = downloadBuffer<uint32_t>(dev, bPIdx, N);

    // deterministic flag outputs
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
        if (gAllflags[i] != refAllflags[i] || gTouch[i] != refTouch[i] || gPatchCh[i] != refPatchChange[i]) ++mm;
    PXG_CHECK_MSG(mm == 0, "flag outputs: %zu mismatches", mm);

    // inactive (nb==0) offsets/patch must be INVALID
    size_t mmInv = 0;
    for (uint32_t i = 0; i < N; ++i)
        if (nbContacts[i] == 0 && (gCOff[i] != INVALID || gFOff[i] != INVALID || gPIdx[i] != INVALID)) ++mmInv;
    PXG_CHECK_MSG(mmInv == 0, "inactive offsets not INVALID: %zu", mmInv);

    // atomic-partition invariants + counter totals
    PXG_CHECK_MSG(gCounters[1] == contactTotal, "contactsBytes: gpu %u vs %u", gCounters[1], contactTotal);
    PXG_CHECK_MSG(gCounters[2] == forceTotal, "forceBytes: gpu %u vs %u", gCounters[2], forceTotal);
    PXG_CHECK_MSG(gCounters[0] == patchTotal, "patchesBytes: gpu %u vs %u", gCounters[0], patchTotal);
    PXG_CHECK_MSG(gCounters[3] == 0, "unexpected overflow flag %u", gCounters[3]);
    PXG_CHECK_MSG(checkPartition(gCOff, nbContacts, contactSize, contactTotal), "contact offsets not a valid partition");
    PXG_CHECK_MSG(checkPartition(gFOff, nbContacts, forceSize, forceTotal), "force offsets not a valid partition");
    {
        std::vector<uint32_t> ones(N, 0);
        for (uint32_t i = 0; i < N; ++i) ones[i] = nbContacts[i] ? 1u : 0u; // one patch per active
        PXG_CHECK_MSG(checkPartition(gPIdx, ones, patchSize, patchTotal), "patch indices not a valid partition");
    }
    PXG_CHECK_MSG(nbActive > N / 4 && (N - nbActive) > N / 8, "activity mix under-exercised (active=%u)", nbActive);

    for (PxgBufferHandle h : binds) dev->destroyBuffer(h);
}

} // namespace
