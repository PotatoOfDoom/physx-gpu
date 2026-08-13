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

// P8.91b: npArtiMassMatrix.slang — `writeMatrix`, `warpCopy` and the set-bit walk.
//
// ===========================================================================
// THERE IS NO ENGINE ORACLE HERE, AND THIS SAYS SO
// ===========================================================================
//
// `writeMatrix` and `warpCopy` are `__device__`-only and their whole content is
// lane arithmetic, so there is nothing PX_CUDA_CALLABLE to compare against — the
// P8.73-P8.80 arrangement, not P8.86's.  What carries them instead:
//
//   * the expectation is a PERMUTATION, not a computation.  Every active lane's
//     144 bytes must appear at that lane's destination and nowhere else, so the
//     test is "is this scatter a bijection onto the right slots", which is
//     exactly what a rank/lane confusion breaks;
//   * whole-region poison.  Every destination region is more than the lanes need
//     and starts at 0xAB..., so an EXTRA write fails as loudly as a wrong one —
//     including a write past the 144th byte of a record, which is what a
//     `warpCopy` with the wrong element count produces;
//   * four masks per scatter, chosen so the pass structure differs in every one.
//
// ===========================================================================
// THE FIXTURE'S OWN COINCIDENCES, REMOVED ON PURPOSE
// ===========================================================================
//
// The standing rule from eleven earlier tranches is that a sweep survivor is
// usually the fixture agreeing with the shader by accident.  The three that
// would happen here by default:
//
//  1. **A contiguous mask makes a lane's RANK equal its lane index.**  Every mask
//     below is scattered, and lane 0 is clear in three of the four, so
//     `warpScanExclusive(mask, lane)` and `lane` disagree for almost every lane.
//  2. **A destination table in lane order makes a permuted scatter look right.**
//     Slot assignment is a stride-13 permutation over 40 slots (13 and 40 are
//     coprime), so consecutive lanes land far apart and any reordering shows.
//  3. **A count that is a multiple of eight hides the `min` in `NbToProcess` and
//     the wrap in `offset - i`.**  The four counts are 21, 5, 0 and 32 — three
//     passes with a short tail, one short pass, no pass at all, and exactly four
//     full passes.
//
// The record stride for the zero walk is 192 bytes, not 144: the 48 bytes past
// each `PxSpatialMatrix` must come back poisoned, which is what makes
// `warpCopy`'s `totalSize / sizeof(T)` element count observable rather than
// merely plausible.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

static std::vector<uint8_t> artiMassMatrixLoader(const char* file) { return loadShaderBlob(file); }

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

const uint32_t LANES = 32;
const uint32_t MATRIX_FLOATS = 36;             // sizeof(PxSpatialMatrix) / 4
const uint32_t MATRIX_BYTES = MATRIX_FLOATS * 4;

// The push-constant blob, mirroring MmWriteParams in npArtiMassMatrix.slang.
struct MmParams
{
    uint64_t srcFloats;
    uint64_t dstTable;
    uint64_t idxTable;
    uint64_t ptrTable;
    uint32_t activeMask;
    uint32_t zeroMask;
    uint32_t zeroBytes;
    uint32_t pad0;
};

// ===========================================================================
// writeMatrix
// ===========================================================================

PXG_TEST(artiMassMatrixWrite)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiMassMatrix.spv", "artiMassMatrixWrite", 48u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &artiMassMatrixLoader);

    // Four masks.  Counts 21 / 5 / 0 / 32, and the first three are scattered so
    // no lane's rank equals its index.  0xFFFFFFFF is included precisely BECAUSE
    // rank == lane there: it is the one case where the exact-multiple-of-eight
    // pass structure is exercised, and the other three keep it honest.
    const uint32_t MASKS[4] = { 0x36EDDDB6u, 0x40810204u, 0x00000000u, 0xFFFFFFFFu };
    const uint32_t EXPECT_COUNT[4] = { 21u, 5u, 0u, 32u };

    const uint32_t SLOTS = 40;   // more slots than lanes: an overrun has room to show

    // 36 distinct, non-round values per lane, distinct between lanes.
    auto val = [](uint32_t lane, uint32_t k) {
        return (float(lane) * 61.0f + float(k) * 1.5f + 1.0f) * 0.05859375f - 7.0f;
    };

    std::vector<float> src(size_t(LANES) * MATRIX_FLOATS);
    for (uint32_t lane = 0; lane < LANES; ++lane)
        for (uint32_t k = 0; k < MATRIX_FLOATS; ++k)
            src[size_t(lane) * MATRIX_FLOATS + k] = val(lane, k);

    // stride-13 permutation over 40 slots: coprime, so lane -> slot is a bijection
    // and consecutive lanes land 13 slots apart
    uint32_t slotOf[LANES];
    for (uint32_t lane = 0; lane < LANES; ++lane)
        slotOf[lane] = (lane * 13u + 7u) % SLOTS;
    {
        bool seen[SLOTS] = {};
        for (uint32_t lane = 0; lane < LANES; ++lane)
        {
            PXG_CHECK_MSG(!seen[slotOf[lane]], "slot permutation collides at lane %u", lane);
            seen[slotOf[lane]] = true;
        }
    }

    PxgDevicePtr dSrc = 0, dTable = 0, dOut = 0;
    shim.memAlloc(&dSrc, src.size() * sizeof(float));
    shim.memAlloc(&dTable, size_t(LANES) * sizeof(uint64_t));
    shim.memAlloc(&dOut, size_t(SLOTS) * MATRIX_BYTES);
    if (!dSrc || !dTable || !dOut)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }

    std::vector<uint64_t> table(LANES);
    for (uint32_t lane = 0; lane < LANES; ++lane)
        table[lane] = uint64_t(dOut) + uint64_t(slotOf[lane]) * MATRIX_BYTES;

    shim.memcpyHtoD(dSrc, src.data(), src.size() * sizeof(float));
    shim.memcpyHtoD(dTable, table.data(), table.size() * sizeof(uint64_t));

    size_t bad = 0;
    for (uint32_t m = 0; m < 4; ++m)
    {
        const uint32_t mask = MASKS[m];
        PXG_CHECK_MSG(__builtin_popcount(mask) == int(EXPECT_COUNT[m]),
                      "mask %u has %d bits, expected %u", m, __builtin_popcount(mask),
                      EXPECT_COUNT[m]);

        // a fresh poison image per dispatch, so a leftover from the previous mask
        // cannot be mistaken for a correct write
        std::vector<uint32_t> poison(size_t(SLOTS) * MATRIX_FLOATS);
        for (size_t k = 0; k < poison.size(); ++k)
            poison[k] = 0xAB000000u | (m << 20) | uint32_t(k & 0xFFFFFu);
        shim.memcpyHtoD(dOut, poison.data(), poison.size() * 4);

        MmParams prm{};
        prm.srcFloats = uint64_t(dSrc);
        prm.dstTable = uint64_t(dTable);
        prm.idxTable = 0;
        prm.ptrTable = 0;
        prm.activeMask = mask;
        prm.zeroMask = 0;
        prm.zeroBytes = 0;
        PXG_CHECK_MSG(sizeof(prm) == 48, "push-constant blob is %zu bytes", sizeof(prm));

        PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                      "launch artiMassMatrixWrite (mask %u)", m);
        shim.streamSynchronize();

        std::vector<uint32_t> got(poison.size());
        shim.memcpyDtoH(got.data(), dOut, got.size() * 4);

        // the expectation: the poison image with each ACTIVE lane's 36 words moved
        // into that lane's slot, and nothing else touched
        std::vector<uint32_t> want = poison;
        for (uint32_t lane = 0; lane < LANES; ++lane)
            if (mask & (1u << lane))
                for (uint32_t k = 0; k < MATRIX_FLOATS; ++k)
                    want[size_t(slotOf[lane]) * MATRIX_FLOATS + k] =
                        f2u(src[size_t(lane) * MATRIX_FLOATS + k]);

        size_t badHere = 0;
        for (size_t k = 0; k < want.size(); ++k)
        {
            if (want[k] == got[k])
                continue;
            if (bad < 12)
                std::printf("MM mask %u slot %zu word %zu  ref %08x  gpu %08x\n", m,
                            k / MATRIX_FLOATS, k % MATRIX_FLOATS, want[k], got[k]);
            ++bad;
            ++badHere;
        }
        if (badHere)
            std::printf("[ info ] mask %08x (count %u): %zu words differ\n", mask, EXPECT_COUNT[m],
                        badHere);
    }

    PXG_CHECK_MSG(bad == 0, "writeMatrix scatters every active lane to its own slot and nowhere else");

    shim.memFree(dSrc);
    shim.memFree(dTable);
    shim.memFree(dOut);
}

// ===========================================================================
// The set-bit walk + warpCopy
// ===========================================================================
//
// `for (i = fixMask; i; i = clearLowestSetBit(i))` with a `__shfl_sync` off
// `lowestSetIndex(i)`, feeding `warpCopy<uint>(record, 0, sizeof(PxSpatialMatrix))`.
//
// The index table is what makes the shuffle observable: the record a given
// iteration clears is named by the lane the walk is currently on, so a walk that
// shuffles from the wrong lane — or that iterates over lanes rather than over the
// mask's set bits — clears a record that must have stayed poisoned.

PXG_TEST(artiMassMatrixZeroWalk)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiMassMatrix.spv", "artiMassMatrixZeroWalk", 48u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &artiMassMatrixLoader);

    // 192, not 144: the 48 bytes past each PxSpatialMatrix must survive, which is
    // what makes warpCopy's element count observable
    const uint32_t RECORD_STRIDE = 192;
    const uint32_t RECORDS = 40;
    const uint32_t RECORD_WORDS = RECORD_STRIDE / 4;

    // six scattered bits, none of them lane 0, and none adjacent
    const uint32_t ZERO_MASK = 0x20840884u;
    PXG_CHECK_MSG(__builtin_popcount(ZERO_MASK) == 6, "the walk mask must have six bits");
    PXG_CHECK_MSG((ZERO_MASK & 1u) == 0, "lane 0 must be clear so rank != lane");

    // Every lane names a DISTINCT record, and the records the mask does not select
    // are real, in-range records that must come back poisoned.  A stride-11
    // permutation over 40, offset so no lane names its own index.
    uint32_t idx[LANES];
    for (uint32_t lane = 0; lane < LANES; ++lane)
        idx[lane] = (lane * 11u + 5u) % RECORDS;
    for (uint32_t lane = 0; lane < LANES; ++lane)
        PXG_CHECK_MSG(idx[lane] != lane, "lane %u names its own index", lane);

    PxgDevicePtr dIdx = 0, dPtr = 0, dRec = 0;
    shim.memAlloc(&dIdx, size_t(LANES) * sizeof(uint32_t));
    shim.memAlloc(&dPtr, size_t(RECORDS) * sizeof(uint64_t));
    shim.memAlloc(&dRec, size_t(RECORDS) * RECORD_STRIDE);
    if (!dIdx || !dPtr || !dRec)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }

    std::vector<uint64_t> ptrs(RECORDS);
    for (uint32_t r = 0; r < RECORDS; ++r)
        ptrs[r] = uint64_t(dRec) + uint64_t(r) * RECORD_STRIDE;

    std::vector<uint32_t> poison(size_t(RECORDS) * RECORD_WORDS);
    for (size_t k = 0; k < poison.size(); ++k)
        poison[k] = 0xAB000000u | uint32_t(k & 0xFFFFFFu);

    shim.memcpyHtoD(dIdx, idx, sizeof(idx));
    shim.memcpyHtoD(dPtr, ptrs.data(), ptrs.size() * sizeof(uint64_t));
    shim.memcpyHtoD(dRec, poison.data(), poison.size() * 4);

    MmParams prm{};
    prm.srcFloats = 0;
    prm.dstTable = 0;
    prm.idxTable = uint64_t(dIdx);
    prm.ptrTable = uint64_t(dPtr);
    prm.activeMask = 0;
    prm.zeroMask = ZERO_MASK;
    prm.zeroBytes = MATRIX_BYTES;   // 144, the engine's sizeof(PxSpatialMatrix)

    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch artiMassMatrixZeroWalk");
    shim.streamSynchronize();

    std::vector<uint32_t> got(poison.size());
    shim.memcpyDtoH(got.data(), dRec, got.size() * 4);

    std::vector<uint32_t> want = poison;
    uint32_t cleared = 0;
    for (uint32_t lane = 0; lane < LANES; ++lane)
    {
        if ((ZERO_MASK & (1u << lane)) == 0)
            continue;
        ++cleared;
        for (uint32_t k = 0; k < MATRIX_FLOATS; ++k)
            want[size_t(idx[lane]) * RECORD_WORDS + k] = 0u;
    }
    PXG_CHECK_MSG(cleared == 6, "expected six records cleared, got %u", cleared);

    size_t bad = 0;
    for (size_t k = 0; k < want.size(); ++k)
    {
        if (want[k] == got[k])
            continue;
        if (bad < 12)
            std::printf("MM walk record %zu word %zu  ref %08x  gpu %08x\n", k / RECORD_WORDS,
                        k % RECORD_WORDS, want[k], got[k]);
        ++bad;
    }
    PXG_CHECK_MSG(bad == 0, "the set-bit walk clears exactly the six selected records' 144 bytes");

    shim.memFree(dIdx);
    shim.memFree(dPtr);
    shim.memFree(dRec);
}

}  // namespace
