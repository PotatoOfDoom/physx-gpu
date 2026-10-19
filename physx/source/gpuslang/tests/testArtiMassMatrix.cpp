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

#include "foundation/PxMat33.h"
#include "foundation/PxVec3.h"
#include "CmSpatialVector.h"
#include "DyFeatherstoneArticulationUtils.h"
#include "PxSpatialMatrix.h"
#include "PxgArticulation.h"
#include "PxgArticulationCoreDesc.h"
#include "PxgBodySim.h"
#include "PxNodeIndex.h"
#include "pxgArtiRefHelpers.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::PxMat33;
using physx::PxReal;
using physx::PxSpatialMatrix;
using physx::PxU32;
using physx::PxMax;
using physx::PxVec3;
using physx::Cm::UnAlignedSpatialVector;
using physx::Dy::SpatialMatrix;
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


// ===========================================================================
// P8.91c: computeMassMatrix1T
// ===========================================================================
//
// The mirror below is the CUDA's control flow over the ENGINE's arithmetic:
// `Dy::SpatialMatrix::operator*`, `PxSpatialMatrix::operator*`,
// `loadSpatialMatrix` / `storeSpatialMatrix` / `zeroSpatialMatrix` (both the
// 7-column and the 6x6 forms), `loadSpatialVector` / `storeSpatialVector`, and
// `refPropagateImpulseW` / `refPropagateAccelerationW` from `pxgArtiRefHelpers.h`
// — the pair P8.90's fixture already uses, hoisted rather than transcribed a
// third time.  Nothing in the expectation is hand-computed.
//
// What the fixture is shaped to see, beyond the standing poison discipline:
//
//   * **fixed-base and free-base articulations IN THE SAME WARP**, so both
//     ballots are non-empty and non-full and neither branch can be skipped;
//   * **link counts that differ within a warp**, so `maxLinksInWarp` exceeds
//     several lanes' own count and those lanes must keep taking part in the
//     ballot with `thisThreadActive` false;
//   * **a warp that runs past `nbArticulations`**, so the block guard and the
//     per-lane `active` test are distinguishable;
//   * **record and block link counts that DISAGREE** — the record's is a
//     nonsense 0xFEE0-series value — so reading the wrong one fails at once;
//   * **an articulation-index permutation**, so the record a lane writes is never
//     the one at its own lane index, and `mArticulationIndex` cannot be confused
//     with the lane;
//   * **a `cfmScale` that varies per link**, so the root's `[0]` and the link's
//     `[linkID]` are different numbers.

const uint32_t MMT_MAX_LINKS = 7;    // never equal to a lane index that matters
const uint32_t MMT_MAX_DOFS = 13;    // distinct from MAX_LINKS and from every count
const uint32_t MMT_CFM_SLOTS = 9;    // > MMT_MAX_LINKS, so an overrun has room

// P9.86c: two DIRECTED fixed-base articulations (both tests pick them up through
// these helpers; k % 3 == 1 for both, so they are fixed-base already).  Each is a
// root plus ONE child, so every value on the columns concerned is a pure signed
// zero whose sign comes only from the chain analysed at mmtFillBlocks' end:
//   k = 4  -> 1: an eFIX child (dof 0) -- the linear response diagonal is
//                (+0, +0, -0), a PxMax zero tie (forwardDynamic2.cu:1918);
//   k = 10 -> 2: a 2-dof spherical child with axis-aligned motion -- impulse
//                columns 0 and 3 give stZ = -0 for both dofs, so qstZ's `+0 + stZ`
//                seed add (articulationDynamic.cuh:178) decides the response sign.
static uint32_t mmtDirected(uint32_t k) { return k == 4u ? 1u : (k == 10u ? 2u : 0u); }
static uint32_t mmtNumLinks(uint32_t k) { return mmtDirected(k) ? 2u : 1u + ((k * 5u) % 6u); }
static bool mmtFixBase(uint32_t k) { return (k % 3u) == 1u; }
static uint8_t mmtDofs(uint32_t k, uint32_t i)
{
    if (mmtDirected(k) == 1u)
        return 0;   // eFIX: the GPU handles it (forwardDynamic2.cu:2648)
    if (mmtDirected(k) == 2u)
        return 2;
    return uint8_t(1u + ((k + i * 2u) % 3u));
}
static uint32_t mmtParent(uint32_t i) { return i == 0 ? 0u : (i - 1u); }

static uint32_t mmtTotalDofs(uint32_t k, uint32_t numLinks)
{
    uint32_t t = 0;
    for (uint32_t i = 1; i < numLinks; ++i)
        t += mmtDofs(k, i);
    return t;
}

// Seed one articulation's slice of the three blocks.  Every value is distinct per
// articulation, per link and per slot, and almost none is round (v() is exactly 0
// where (k*13 + i*7 + f) % 211 == 103) -- EXCEPT on the two directed lanes, whose
// link 1 is overwritten at the end with round, axis-aligned engine data (P9.86c).
static void mmtFillBlocks(physx::PxgArticulationBlockLinkData* lk,
                          physx::PxgArticulationBlockDofData* df,
                          physx::PxgArticulationBlockData& ab, uint32_t k, uint32_t numLinks,
                          bool fixBase, uint32_t articulationIndex, uint32_t lane)
{
    auto v = [&](uint32_t i, uint32_t f) {
        return (float((k * 13u + i * 7u + f) % 211u) + 1.0f) * 0.03125f - 3.25f;
    };

    ab.mNumLinks[lane] = numLinks;
    // Both flag words are NON-ZERO and differ in more than the fix bit, so a test
    // of `flags != 0` cannot stand in for the bit test -- the sweep caught that.
    ab.mFlags[lane] = fixBase ? (PxU32(physx::PxArticulationFlag::eFIX_BASE)
                                      | (1u << 2))   // the 5.6.1 eDRIVE_LIMITS_ARE_FORCES bit, unassigned in 5.12
                              : PxU32(physx::PxArticulationFlag::eDISABLE_SELF_COLLISION);
    ab.mArticulationIndex[lane] = articulationIndex;

    // a dense, deliberately NON-symmetric inverse articulated inertia: the root's
    // response uses `topLeft.transformTranspose` for its bottom half, which agrees
    // with `topLeft *` only for a symmetric block
    SpatialMatrix inv(PxMat33(PxVec3(v(0, 1), v(0, 2), v(0, 3)),
                              PxVec3(v(0, 4), v(0, 5), v(0, 6)),
                              PxVec3(v(0, 7), v(0, 8), v(0, 9))),
                      PxMat33(PxVec3(v(0, 10), v(0, 11), v(0, 12)),
                              PxVec3(v(0, 13), v(0, 14), v(0, 15)),
                              PxVec3(v(0, 16), v(0, 17), v(0, 18))),
                      PxMat33(PxVec3(v(0, 19), v(0, 20), v(0, 21)),
                              PxVec3(v(0, 22), v(0, 23), v(0, 24)),
                              PxVec3(v(0, 25), v(0, 26), v(0, 27))));
    storeSpatialMatrix(ab.mInvSpatialArticulatedInertia, lane, inv);

    uint32_t cursor = 0;
    for (uint32_t i = 0; i < numLinks; ++i)
    {
        physx::PxgArticulationBlockLinkData& b = lk[i];
        const uint8_t dofs = (i == 0) ? uint8_t(0) : mmtDofs(k, i);
        b.mParents[lane] = mmtParent(i);
        b.mDofs[lane] = dofs;
        b.mJointOffset[lane] = cursor;
        b.mRw_x[lane] = v(i, 30);
        b.mRw_y[lane] = v(i, 31);
        b.mRw_z[lane] = v(i, 32);

        for (uint32_t dd = 0; dd < dofs; ++dd)
        {
            physx::PxgArticulationBlockDofData& d = df[cursor + dd];
            const uint32_t o = 40 + dd * 24;
            storeSpatialVector(d.mWorldMotionMatrix,
                               UnAlignedSpatialVector(PxVec3(v(i, o + 0), v(i, o + 1), v(i, o + 2)),
                                                      PxVec3(v(i, o + 3) * 3.0f, v(i, o + 4) * 3.0f,
                                                             v(i, o + 5) * 3.0f)),
                               lane);
            storeSpatialVector(d.mIsInvDW,
                               UnAlignedSpatialVector(PxVec3(v(i, o + 6), v(i, o + 7), v(i, o + 8)),
                                                      PxVec3(v(i, o + 9) * 5.0f, v(i, o + 10) * 5.0f,
                                                             v(i, o + 11) * 5.0f)),
                               lane);
            storeSpatialVector(d.mIsW,
                               UnAlignedSpatialVector(PxVec3(v(i, o + 12), v(i, o + 13), v(i, o + 14)),
                                                      PxVec3(v(i, o + 15) * 7.0f, v(i, o + 16) * 7.0f,
                                                             v(i, o + 17) * 7.0f)),
                               lane);
            d.mInvStIsT_x[lane] = v(i, o + 18);
            d.mInvStIsT_y[lane] = v(i, o + 19);
            d.mInvStIsT_z[lane] = v(i, o + 20);
        }
        cursor += dofs;
    }

    // ---- P9.86c: the directed lanes (see mmtDirected) ----
    //
    // Both have a fixed base, so link 0's response is zeroSpatialMatrix (+0,
    // forwardDynamic2.cu:1793), `parentResponse * Zp` is +0 in every row
    // (PxSpatialMatrix.h: a +0 seed that only `+=`s) and Zr = -(that) is -0 x6.
    const uint32_t dir = mmtDirected(k);
    if (dir == 1u)
    {
        // eFIX child: pDeltaV = translateSpatialVector(-rw, Zr), so bottom =
        // -0 + (-rw) x (-0,-0,-0).  With -rw = (1, -2, -3) the cross product is
        // (+0, +0, -0), so every column's linear diagonal is (+0, +0, -0) and the
        // CUDA's PxMax<float> (selectMax `a > b ? a : b`, PxUnixMathIntrinsics.h:122)
        // nests to the LAST one: cfm = cfmScale * -0.
        lk[1].mRw_x[lane] = -1.0f;
        lk[1].mRw_y[lane] = 2.0f;
        lk[1].mRw_z[lane] = 3.0f;
    }
    else if (dir == 2u)
    {
        // A 2-dof spherical joint, swing axes +y and -z, the child COM at
        // r = (0.25, 0, 0) from the anchor (motion bottom = axis x r); mass 4 and a
        // COM inertia [[1,0,0],[0,1,0.5],[0,0.5,1]], so IsW = (m v, I w) exactly and
        // the stIs coupling is -0.5.  invStIsT and IsInvDW are derived as the
        // engine's eSPHERICAL path derives them (forwardDynamic2.cu:1455-1528,
        // armature 0).
        //
        // WHY EACH MOTION VECTOR HAS A NEGATIVE COMPONENT (measured, P9.86c): llvmpipe
        // rewrites mmAxis's `bcsel(i == k, -1.0, 0.0)` into `fneg(b2f32(i == k))`, so
        // the port's unit impulse Z carries -0, not +0, in its zero lanes.  At HEAD that
        // is unobservable (Z only reaches the canonicalised qstZ and Zp, which only
        // reaches asmM6MulVec's canonicalised rows).  But the reverted fold leaves
        // qstZ = -(m . Z) itself, and with -0 lanes that is -0 only if some product
        // m_j * (-0) is +0 -- i.e. only if m has a negative component.  The first
        // design (swing axes +y/+z, s0 all non-negative) kept qstZ[0] = +0 and the
        // revert survived; the r3 twins below model both lane conventions.
        physx::PxgArticulationBlockDofData* d = df + lk[1].mJointOffset[lane];
        lk[1].mRw_x[lane] = 0.5f;
        lk[1].mRw_y[lane] = 0.0f;
        lk[1].mRw_z[lane] = 0.0f;

        const UnAlignedSpatialVector s[2] = {
            UnAlignedSpatialVector(PxVec3(0.0f, 1.0f, 0.0f), PxVec3(0.0f, 0.0f, -0.25f)),
            UnAlignedSpatialVector(PxVec3(0.0f, 0.0f, -1.0f), PxVec3(0.0f, -0.25f, 0.0f)) };
        const UnAlignedSpatialVector isw[2] = {
            UnAlignedSpatialVector(PxVec3(0.0f, 0.0f, -1.0f), PxVec3(0.0f, 1.0f, 0.5f)),
            UnAlignedSpatialVector(PxVec3(0.0f, -1.0f, 0.0f), PxVec3(0.0f, -0.5f, -1.0f)) };

        PxMat33 invStIs(physx::PxIdentity);
        for (PxU32 ind2 = 0; ind2 < 2; ++ind2)
        {
            for (PxU32 ind = 0; ind < 2; ++ind)
                invStIs[ind][ind2] = s[ind2].innerProduct(isw[ind]);
            invStIs[ind2][ind2] += 0.0f;   // mArmature
        }
        invStIs = invStIs.getInverse();

        for (PxU32 ind = 0; ind < 2; ++ind)
        {
            UnAlignedSpatialVector isID(PxVec3(0.0f), PxVec3(0.0f));
            for (PxU32 ind2 = 0; ind2 < 2; ++ind2)
                isID += isw[ind2] * invStIs[ind][ind2];
            storeSpatialVector(d[ind].mWorldMotionMatrix, s[ind], lane);
            storeSpatialVector(d[ind].mIsW, isw[ind], lane);
            storeSpatialVector(d[ind].mIsInvDW, isID, lane);
            d[ind].mInvStIsT_x[lane] = invStIs[0][ind];
            d[ind].mInvStIsT_y[lane] = invStIs[1][ind];
            d[ind].mInvStIsT_z[lane] = invStIs[2][ind];
        }

        // The rows the r3 chain needs: (pos, pos, -0) for BOTH dofs.  With qstZ =
        // tJointDelta = +0 the CUDA's jDelta is +0 + +0 + (-0)(+0) = +0; a -0 qstZ
        // (the folded seed add) makes every product -0 and jDelta -0.
        PXG_CHECK_MSG(f2u(d[0].mInvStIsT_z[lane]) == 0x80000000u
                          && f2u(d[1].mInvStIsT_z[lane]) == 0x80000000u
                          && d[0].mInvStIsT_x[lane] > 0.0f && d[0].mInvStIsT_y[lane] > 0.0f
                          && d[1].mInvStIsT_x[lane] > 0.0f && d[1].mInvStIsT_y[lane] > 0.0f,
                      "directed spherical lane %u: invStIsT rows must be (pos, pos, -0), got "
                      "(%08x %08x %08x) (%08x %08x %08x)",
                      k, f2u(d[0].mInvStIsT_x[lane]), f2u(d[0].mInvStIsT_y[lane]),
                      f2u(d[0].mInvStIsT_z[lane]), f2u(d[1].mInvStIsT_x[lane]),
                      f2u(d[1].mInvStIsT_y[lane]), f2u(d[1].mInvStIsT_z[lane]));
    }
}

// P9.86c: mutant TWINS inside the mirror, so each test can prove its directed
// input still separates the engine from the port's pre-P9.86 behaviour:
//   0 = the engine (PxMax<float> = selectMax; qstZ = +0 + stZ, IEEE);
//   1 = the r2 twin: the generic PxMax template `a < b ? b : a` (PxMath.h:74-77),
//       which keeps the OTHER operand on a +-0 tie;
//   2 = the r3 twin AS LLVMPIPE COMPILES IT: qstZ[k] = stZ (the literal `{0,0,0}`
//       seed forwarded and `+0 + stZ` folded, F3 -- the reverted build's LLVM IR,
//       GALLIVM_DEBUG=ir, stores `fneg(ip)` with no add), where stZ is formed from
//       the impulse llvmpipe materialises: mmAxis's `bcsel(i == k, -1.0, 0.0)` is
//       `fneg(b2f32(i == k))` in its NIR (NIR_DEBUG=print_cs), so every zero lane
//       is -0;
//   3 = the plain r3 twin: the same fold against the IEEE impulse (+0 lanes), i.e.
//       what the revert would do on a driver without that bcsel rewrite.
static int gMmtVariant = 0;

static PxReal mmtPxMax(PxReal a, PxReal b)
{
    return gMmtVariant == 1 ? (a < b ? b : a) : PxMax(a, b);
}

// The mirror: one articulation, the CUDA's structure, the engine's arithmetic.
// `recordMatrices` and `cfms` are that articulation's record arrays; the fixed
// base path clears the record matrix (`warpCopy` of a zero) as well as the block.
static void mmtMirror(physx::PxgArticulationBlockLinkData* lk,
                      const physx::PxgArticulationBlockDofData* df,
                      const physx::PxgArticulationBlockData& ab, PxSpatialMatrix* recordMatrices,
                      PxReal* cfms, const PxReal* cfmScale, uint32_t numLinks, bool fixBase,
                      uint32_t lane)
{
    if (fixBase)
    {
        zeroSpatialMatrix(lk[0].mSpatialResponseMatrix, lane);
        std::memset(&recordMatrices[0], 0, sizeof(PxSpatialMatrix));
        cfms[0] = 0.f;
        lk[0].mCfm[lane] = 0.f;
    }
    else
    {
        SpatialMatrix inv;
        loadSpatialMatrix(ab.mInvSpatialArticulatedInertia, lane, inv);

        PxSpatialMatrix response;
        for (PxU32 i = 0; i < 6; ++i)
        {
            UnAlignedSpatialVector vec = UnAlignedSpatialVector::Zero();
            vec[i] = 1.f;
            const UnAlignedSpatialVector r = inv * vec;
            response.column[i][0] = r.top.x;
            response.column[i][1] = r.top.y;
            response.column[i][2] = r.top.z;
            response.column[i][3] = r.bottom.x;
            response.column[i][4] = r.bottom.y;
            response.column[i][5] = r.bottom.z;
        }
        storeSpatialMatrix(lk[0].mSpatialResponseMatrix, lane, response);

        const PxReal cfmVal =
            cfmScale[0] * mmtPxMax(inv.bottomLeft.column0.x,
                                   mmtPxMax(inv.bottomLeft.column1.y, inv.bottomLeft.column2.z));
        cfms[0] = cfmVal;
        lk[0].mCfm[lane] = cfmVal;

        recordMatrices[0] = response;
    }

    for (uint32_t linkID = 1; linkID < numLinks; ++linkID)
    {
        physx::PxgArticulationBlockLinkData& thisLink = lk[linkID];
        const physx::PxgArticulationBlockDofData* dofData = df + thisLink.mJointOffset[lane];
        const uint32_t parentId = thisLink.mParents[lane];
        const PxVec3 rw(thisLink.mRw_x[lane], thisLink.mRw_y[lane], thisLink.mRw_z[lane]);
        const PxU32 dof = thisLink.mDofs[lane];

        UnAlignedSpatialVector motionV[3], isInvD[3], isW[3];
        PxVec3 invStIsT[3];
        for (PxU32 i = 0; i < 3; ++i)
        {
            motionV[i] = UnAlignedSpatialVector(PxVec3(0.f), PxVec3(0.f));
            isInvD[i] = motionV[i];
            isW[i] = motionV[i];
            invStIsT[i] = PxVec3(0.f);
            if (i < dof)
            {
                motionV[i] = loadSpatialVector(dofData[i].mWorldMotionMatrix, lane);
                invStIsT[i] = PxVec3(dofData[i].mInvStIsT_x[lane], dofData[i].mInvStIsT_y[lane],
                                     dofData[i].mInvStIsT_z[lane]);
                isInvD[i] = loadSpatialVector(dofData[i].mIsInvDW, lane);
                isW[i] = loadSpatialVector(dofData[i].mIsW, lane);
            }
        }

        PxSpatialMatrix parentResponse;
        loadSpatialMatrix(lk[parentId].mSpatialResponseMatrix, lane, parentResponse);

        PxSpatialMatrix response;
        for (PxU32 i = 0; i < 6; ++i)
        {
            UnAlignedSpatialVector vec(PxVec3(0.f), PxVec3(0.f));
            vec[i] = -1.f;

            PxReal qstZ[3] = { 0.f, 0.f, 0.f };
            const UnAlignedSpatialVector Zp =
                refPropagateImpulseW(isInvD, rw, motionV, vec, dof, qstZ);
            if (gMmtVariant == 2 || gMmtVariant == 3)   // the r3 twins: the folded seed add
            {
                UnAlignedSpatialVector z = vec;
                if (gMmtVariant == 2)
                    for (PxU32 j = 0; j < 6; ++j)
                        if (j != i)
                            z[j] = -0.0f;
                for (PxU32 k = 0; k < dof; ++k)
                    qstZ[k] = -motionV[k].innerProduct(z);
            }
            const UnAlignedSpatialVector Zr = -(parentResponse * Zp);
            const UnAlignedSpatialVector deltaV =
                refPropagateAccelerationW(rw, invStIsT, motionV, Zr, dof, isW, qstZ);

            response.column[i][0] = deltaV.top.x;
            response.column[i][1] = deltaV.top.y;
            response.column[i][2] = deltaV.top.z;
            response.column[i][3] = deltaV.bottom.x;
            response.column[i][4] = deltaV.bottom.y;
            response.column[i][5] = deltaV.bottom.z;
        }

        const PxReal cfmVal =
            cfmScale[linkID]
            * mmtPxMax(response.column[0][3],
                       mmtPxMax(response.column[1][4], response.column[2][5]));
        cfms[linkID] = cfmVal;
        thisLink.mCfm[lane] = cfmVal;

        storeSpatialMatrix(thisLink.mSpatialResponseMatrix, lane, response);
        recordMatrices[linkID] = response;
    }
}

// P9.86c: re-run a test's expectation builder under a mutant twin and count the
// 32-bit words of the three output images (link blocks, record matrices, record
// cfms) that differ from the engine's.
template <typename Build>
static size_t mmtTwinDiff(const Build& build, int variant,
                          const std::vector<physx::PxgArticulationBlockLinkData>& wantLk,
                          const std::vector<uint32_t>& wantMat, const std::vector<uint32_t>& wantCfm)
{
    std::vector<physx::PxgArticulationBlockLinkData> lk;
    std::vector<uint32_t> mat, cfm;
    build(variant, lk, mat, cfm);
    size_t n = 0;
    auto count = [&n](const void* ap, const void* bp, size_t bytes) {
        const uint32_t* a = reinterpret_cast<const uint32_t*>(ap);
        const uint32_t* b = reinterpret_cast<const uint32_t*>(bp);
        for (size_t x = 0; x < bytes / 4; ++x)
            n += (a[x] != b[x]) ? 1u : 0u;
    };
    count(lk.data(), wantLk.data(), lk.size() * sizeof(physx::PxgArticulationBlockLinkData));
    count(mat.data(), wantMat.data(), mat.size() * 4);
    count(cfm.data(), wantCfm.data(), cfm.size() * 4);
    return n;
}

// P9.86c: pin the engine's values on the directed lanes, so the inputs cannot
// silently stop producing the signed zeros the two reverts are measured on.
//   eFIX lane:      cfm = cfmScale * selectMax-nest(+0, +0, -0) = -0, in the record's
//                   cfms[1] AND the link block's mCfm;
//   spherical lane: impulse columns 0 and 3, top x/y/z, are +0 (block and record).
static void mmtCheckDirectedWant(const physx::PxgArticulationBlockLinkData* wantLk,
                                 const uint32_t* wantMat, const uint32_t* wantCfm,
                                 uint32_t recFix, uint32_t laneFix, uint32_t recSph,
                                 uint32_t laneSph, const char* what)
{
    PXG_CHECK_MSG(wantCfm[size_t(recFix) * MMT_CFM_SLOTS + 1] == 0x80000000u
                      && f2u(wantLk[1].mCfm[laneFix]) == 0x80000000u,
                  "%s: the eFIX lane's link cfm must be -0 (record %08x, block %08x)", what,
                  wantCfm[size_t(recFix) * MMT_CFM_SLOTS + 1], f2u(wantLk[1].mCfm[laneFix]));

    PxSpatialMatrix blk;
    loadSpatialMatrix(wantLk[1].mSpatialResponseMatrix, laneSph, blk);
    const uint32_t* rec = wantMat + (size_t(recSph) * MMT_CFM_SLOTS + 1) * MATRIX_FLOATS;
    uint32_t zeros = 0;
    for (uint32_t c = 0; c < 6; c += 3)
        for (uint32_t row = 0; row < 3; ++row)
            zeros += (f2u(blk.column[c][row]) == 0u && rec[c * 6 + row] == 0u) ? 1u : 0u;
    PXG_CHECK_MSG(zeros == 6, "%s: the spherical lane's impulse columns 0/3 top must be +0 in "
                              "block and record (%u of 6)", what, zeros);
}

PXG_TEST(artiMassMatrixBody)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiMassMatrix.spv", "artiMassMatrixBody", 48u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &artiMassMatrixLoader);

    // 27 live lanes, five left past the end so the per-lane `active` test is
    // distinguishable from "the whole warp".  27 is odd, prime to 3, and not a
    // multiple of 8, so the scatter's pass structure has a short tail.
    const uint32_t LIVE = 27;
    const uint32_t RECORDS = 40;   // > LIVE, and the index map is a permutation

    std::vector<physx::PxgArticulationBlockLinkData> lkImg(MMT_MAX_LINKS);
    std::vector<physx::PxgArticulationBlockDofData> dfImg(MMT_MAX_DOFS);
    std::vector<physx::PxgArticulationBlockData> abImg(1);
    std::memset(lkImg.data(), 0xAB, lkImg.size() * sizeof(physx::PxgArticulationBlockLinkData));
    std::memset(dfImg.data(), 0xAB, dfImg.size() * sizeof(physx::PxgArticulationBlockDofData));
    std::memset(abImg.data(), 0xAB, abImg.size() * sizeof(physx::PxgArticulationBlockData));

    // a stride-11 permutation over 40: no lane names its own index
    uint32_t artiIdx[32];
    for (uint32_t lane = 0; lane < 32; ++lane)
        artiIdx[lane] = (lane * 11u + 3u) % RECORDS;
    for (uint32_t lane = 0; lane < 32; ++lane)
        PXG_CHECK_MSG(artiIdx[lane] != lane, "lane %u names its own record", lane);

    uint32_t fixSeen = 0, freeSeen = 0, maxLinks = 0;
    for (uint32_t lane = 0; lane < LIVE; ++lane)
    {
        const uint32_t n = mmtNumLinks(lane);
        PXG_CHECK_MSG(n <= MMT_MAX_LINKS, "articulation %u has %u links", lane, n);
        PXG_CHECK_MSG(mmtTotalDofs(lane, n) <= MMT_MAX_DOFS, "articulation %u needs %u dofs", lane,
                      mmtTotalDofs(lane, n));
        mmtFillBlocks(lkImg.data(), dfImg.data(), abImg[0], lane, n, mmtFixBase(lane),
                      artiIdx[lane], lane);
        mmtFixBase(lane) ? ++fixSeen : ++freeSeen;
        maxLinks = maxLinks > n ? maxLinks : n;
    }
    std::printf("[ info ] mm body: %u fixed-base, %u free-base, maxLinksInWarp %u\n", fixSeen,
                freeSeen, maxLinks);
    PXG_CHECK_MSG(fixSeen > 4 && freeSeen > 4, "both ballots must be non-empty and non-full");
    PXG_CHECK_MSG(maxLinks == 6, "maxLinksInWarp should be 6, got %u", maxLinks);

    PxgDevicePtr dLk = 0, dDf = 0, dAb = 0, dRecs = 0, dMat = 0, dCfm = 0, dScale = 0;
    shim.memAlloc(&dLk, lkImg.size() * sizeof(physx::PxgArticulationBlockLinkData));
    shim.memAlloc(&dDf, dfImg.size() * sizeof(physx::PxgArticulationBlockDofData));
    shim.memAlloc(&dAb, abImg.size() * sizeof(physx::PxgArticulationBlockData));
    shim.memAlloc(&dRecs, size_t(RECORDS) * sizeof(physx::PxgArticulation));
    shim.memAlloc(&dMat, size_t(RECORDS) * MMT_CFM_SLOTS * sizeof(PxSpatialMatrix));
    shim.memAlloc(&dCfm, size_t(RECORDS) * MMT_CFM_SLOTS * sizeof(PxReal));
    shim.memAlloc(&dScale, size_t(RECORDS) * MMT_CFM_SLOTS * sizeof(PxReal));
    if (!dLk || !dDf || !dAb || !dRecs || !dMat || !dCfm || !dScale)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }

    // per-link cfmScale, so the root's [0] and a link's [linkID] differ
    std::vector<PxReal> cfmScale(size_t(RECORDS) * MMT_CFM_SLOTS);
    for (size_t j = 0; j < cfmScale.size(); ++j)
        cfmScale[j] = 0.5f + 0.0625f * float(j % 23);

    std::vector<physx::PxgArticulation> recs(RECORDS);
    std::memset(recs.data(), 0, recs.size() * sizeof(physx::PxgArticulation));
    for (uint32_t r = 0; r < RECORDS; ++r)
    {
        recs[r].data.numLinks = physx::PxU16(0xFEE0u + r);   // never a plausible link count
        recs[r].spatialResponseMatrixW = reinterpret_cast<PxSpatialMatrix*>(
            dMat + PxgDevicePtr(size_t(r) * MMT_CFM_SLOTS * sizeof(PxSpatialMatrix)));
        recs[r].cfms =
            reinterpret_cast<PxReal*>(dCfm + PxgDevicePtr(size_t(r) * MMT_CFM_SLOTS * sizeof(PxReal)));
        recs[r].cfmScale = reinterpret_cast<PxReal*>(
            dScale + PxgDevicePtr(size_t(r) * MMT_CFM_SLOTS * sizeof(PxReal)));
    }

    // the record-side destinations, poisoned
    std::vector<uint32_t> matPoison(size_t(RECORDS) * MMT_CFM_SLOTS * sizeof(PxSpatialMatrix) / 4);
    for (size_t j = 0; j < matPoison.size(); ++j)
        matPoison[j] = 0xAB000000u | uint32_t(j & 0xFFFFFFu);
    std::vector<uint32_t> cfmPoison(size_t(RECORDS) * MMT_CFM_SLOTS);
    for (size_t j = 0; j < cfmPoison.size(); ++j)
        cfmPoison[j] = 0xAC000000u | uint32_t(j & 0xFFFFFFu);

    shim.memcpyHtoD(dLk, lkImg.data(), lkImg.size() * sizeof(physx::PxgArticulationBlockLinkData));
    shim.memcpyHtoD(dDf, dfImg.data(), dfImg.size() * sizeof(physx::PxgArticulationBlockDofData));
    shim.memcpyHtoD(dAb, abImg.data(), abImg.size() * sizeof(physx::PxgArticulationBlockData));
    shim.memcpyHtoD(dRecs, recs.data(), recs.size() * sizeof(physx::PxgArticulation));
    shim.memcpyHtoD(dMat, matPoison.data(), matPoison.size() * 4);
    shim.memcpyHtoD(dCfm, cfmPoison.data(), cfmPoison.size() * 4);
    shim.memcpyHtoD(dScale, cfmScale.data(), cfmScale.size() * sizeof(PxReal));

    struct BodyParams
    {
        uint64_t artiBlock, linkBlocks, dofBlocks, articulations;
        uint32_t nbLanes, pad0, pad1, pad2;
    } prm = { uint64_t(dAb), uint64_t(dLk), uint64_t(dDf), uint64_t(dRecs), LIVE, 0, 0, 0 };
    PXG_CHECK_MSG(sizeof(prm) == 48, "push-constant blob is %zu bytes", sizeof(prm));

    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch artiMassMatrixBody");
    shim.streamSynchronize();

    // ---- the expectation ----
    // Built by a lambda so the mutant twins (gMmtVariant) re-run the SAME loop.
    auto buildWant = [&](int variant, std::vector<physx::PxgArticulationBlockLinkData>& wLk,
                         std::vector<uint32_t>& wMat, std::vector<uint32_t>& wCfm) {
        gMmtVariant = variant;
        wLk = lkImg;
        wMat = matPoison;
        wCfm = cfmPoison;
        for (uint32_t lane = 0; lane < LIVE; ++lane)
        {
            const uint32_t r = artiIdx[lane];
            mmtMirror(wLk.data(), dfImg.data(), abImg[0],
                      reinterpret_cast<PxSpatialMatrix*>(wMat.data())
                          + size_t(r) * MMT_CFM_SLOTS,
                      reinterpret_cast<PxReal*>(wCfm.data()) + size_t(r) * MMT_CFM_SLOTS,
                      cfmScale.data() + size_t(r) * MMT_CFM_SLOTS, mmtNumLinks(lane),
                      mmtFixBase(lane), lane);
        }
        gMmtVariant = 0;
    };
    std::vector<physx::PxgArticulationBlockLinkData> wantLk;
    std::vector<uint32_t> wantMat, wantCfm;
    buildWant(0, wantLk, wantMat, wantCfm);

    // P9.86c coverage: each twin must change exactly the words its directed lane
    // was built for, or the input has stopped separating engine from revert.
    PXG_CHECK_MSG(LIVE > 10 && mmtFixBase(4) && mmtFixBase(10),
                  "the directed lanes 4 and 10 must be live and fixed-base");
    {
        const size_t d = mmtTwinDiff(buildWant, 1, wantLk, wantMat, wantCfm);
        std::printf("[ info ] mm body: r2 twin (generic PxMax) changes %zu words\n", d);
        PXG_CHECK_MSG(d == 2, "r2 twin: the eFIX lane's selectMax tie must flip record cfms[1] "
                              "and link1.mCfm (got %zu words)", d);
    }
    // Both r3 twins: the fold as llvmpipe compiles it (-0 impulse lanes) and the
    // plain fold (+0 lanes).  Either way, impulse columns 0 and 3 of the spherical
    // lane come out with top (-0, -0, +0) where the engine has (+0, +0, +0).
    for (int variant = 2; variant <= 3; ++variant)
    {
        const size_t d = mmtTwinDiff(buildWant, variant, wantLk, wantMat, wantCfm);
        std::printf("[ info ] mm body: r3 twin %d (folded qstZ seed) changes %zu words\n",
                    variant, d);
        PXG_CHECK_MSG(d == 8, "r3 twin %d: the spherical lane's impulse columns 0/3 must flip "
                              "top x/y in block and record (got %zu words)", variant, d);
    }
    mmtCheckDirectedWant(wantLk.data(), wantMat.data(), wantCfm.data(), artiIdx[4], 4u,
                         artiIdx[10], 10u, "body");

    std::vector<physx::PxgArticulationBlockLinkData> gotLk(lkImg.size());
    std::vector<physx::PxgArticulationBlockData> gotAb(abImg.size());
    std::vector<physx::PxgArticulationBlockDofData> gotDf(dfImg.size());
    std::vector<uint32_t> gotMat(matPoison.size()), gotCfm(cfmPoison.size());
    shim.memcpyDtoH(gotLk.data(), dLk, gotLk.size() * sizeof(physx::PxgArticulationBlockLinkData));
    shim.memcpyDtoH(gotAb.data(), dAb, gotAb.size() * sizeof(physx::PxgArticulationBlockData));
    shim.memcpyDtoH(gotDf.data(), dDf, gotDf.size() * sizeof(physx::PxgArticulationBlockDofData));
    shim.memcpyDtoH(gotMat.data(), dMat, gotMat.size() * 4);
    shim.memcpyDtoH(gotCfm.data(), dCfm, gotCfm.size() * 4);

    // Fully BITWISE (P9.86c): the +0/-0 forgiveness that stood here for the
    // caller-side qstZ seed is retired -- P9.86 canonicalises that seed, and the
    // directed lanes above are what would expose it (and the PxMax tie) if it came back.
    size_t bad = 0;
    auto cmp = [&](const char* what, const void* wp, const void* gp, size_t bytes) {
        const uint32_t* w = reinterpret_cast<const uint32_t*>(wp);
        const uint32_t* g = reinterpret_cast<const uint32_t*>(gp);
        for (size_t x = 0; x < bytes / 4; ++x)
        {
            if (w[x] == g[x])
                continue;
            if (bad < 12)
                std::printf("MM body %s word %zu  ref %08x  gpu %08x\n", what, x, w[x], g[x]);
            ++bad;
        }
    };
    cmp("link", wantLk.data(), gotLk.data(),
        wantLk.size() * sizeof(physx::PxgArticulationBlockLinkData));
    // the dof blocks and the articulation block are INPUTS: nothing may touch them
    cmp("dof", dfImg.data(), gotDf.data(), dfImg.size() * sizeof(physx::PxgArticulationBlockDofData));
    cmp("arti", abImg.data(), gotAb.data(), abImg.size() * sizeof(physx::PxgArticulationBlockData));
    cmp("record matrices", wantMat.data(), gotMat.data(), wantMat.size() * 4);
    cmp("record cfms", wantCfm.data(), gotCfm.data(), wantCfm.size() * 4);

    PXG_CHECK_MSG(bad == 0, "the mass-matrix body matches the engine-built mirror");

    shim.memFree(dLk);
    shim.memFree(dDf);
    shim.memFree(dAb);
    shim.memFree(dRecs);
    shim.memFree(dMat);
    shim.memFree(dCfm);
    shim.memFree(dScale);
}


// ---------------------------------------------------------------------------
// The kernel: computeMassMatrix1T
// ---------------------------------------------------------------------------
//
// 130 articulations over three 1T blocks' worth of warps (grid num1TBlocks * 2,
// block (32, 1, 1)), so the last warp runs 30 lanes past `nbArticulations` and a
// sixth warp slice is allocated and left entirely poisoned.  Three things this
// adds over the body test, each a mutant:
//
//   * the per-warp block strides (`maxLinks` / `maxDofs` for the two arrays,
//     ONE block for the articulation block);
//   * `nbArticulationBlocks = ceil(nbArticulations / 32)` as the grid guard,
//     which is NOT the per-lane `active` test;
//   * that the kernel reads `mArticulationIndex` out of the block and never
//     touches the island-node / bodySim chain the two kernels before it walk —
//     so that chain is filled with values that would produce different records
//     if it were used.

const uint32_t MMK_NB_ARTI = 130;
const uint32_t MMK_WARPS_ALLOC = 6;   // five live, one poisoned
const uint32_t MMK_RECORDS = 160;

PXG_TEST(artiMassMatrixLaunch)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        // engine ABI { 8 } -- the descriptor pointer alone
        { "npArtiMassMatrix.spv", "computeMassMatrix1T", 8u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &artiMassMatrixLoader);

    std::vector<physx::PxgArticulationBlockLinkData> lkImg(size_t(MMK_WARPS_ALLOC) * MMT_MAX_LINKS);
    std::vector<physx::PxgArticulationBlockDofData> dfImg(size_t(MMK_WARPS_ALLOC) * MMT_MAX_DOFS);
    std::vector<physx::PxgArticulationBlockData> abImg(MMK_WARPS_ALLOC);
    std::memset(lkImg.data(), 0xAB, lkImg.size() * sizeof(physx::PxgArticulationBlockLinkData));
    std::memset(dfImg.data(), 0xAB, dfImg.size() * sizeof(physx::PxgArticulationBlockDofData));
    std::memset(abImg.data(), 0xAB, abImg.size() * sizeof(physx::PxgArticulationBlockData));

    // stride-37 over 160: coprime, so the record a lane writes is never its own
    // index and never its neighbour's
    std::vector<uint32_t> artiIdx(MMK_NB_ARTI);
    for (uint32_t k = 0; k < MMK_NB_ARTI; ++k)
        artiIdx[k] = (k * 37u + 9u) % MMK_RECORDS;

    for (uint32_t k = 0; k < MMK_NB_ARTI; ++k)
    {
        const uint32_t warp = k / 32u, lane = k % 32u;
        const uint32_t n = mmtNumLinks(k);
        PXG_CHECK_MSG(mmtTotalDofs(k, n) <= MMT_MAX_DOFS, "articulation %u needs %u dofs", k,
                      mmtTotalDofs(k, n));
        mmtFillBlocks(lkImg.data() + size_t(warp) * MMT_MAX_LINKS,
                      dfImg.data() + size_t(warp) * MMT_MAX_DOFS, abImg[warp], k, n, mmtFixBase(k),
                      artiIdx[k], lane);
    }

    // The island / bodySim chain the two kernels before this one walk.  This
    // kernel must NOT use it, so it is filled with indices that resolve to
    // different records -- a port that copies P8.85's resolution writes the wrong
    // articulations and fails everywhere at once.
    const uint32_t ARTI_OFFSET = 5;
    std::vector<physx::PxNodeIndex> island(ARTI_OFFSET + MMK_NB_ARTI + 3);
    for (size_t i = 0; i < island.size(); ++i)
        island[i] = physx::PxNodeIndex(physx::PxU32((i * 3u + 17u) % 191u));
    std::vector<physx::PxgBodySim> bodySims(192);
    std::memset(bodySims.data(), 0xEE, bodySims.size() * sizeof(physx::PxgBodySim));
    for (size_t i = 0; i < bodySims.size(); ++i)
        bodySims[i].articulationRemapId = physx::PxU32((i * 5u + 1u) % MMK_RECORDS);

    std::vector<PxReal> cfmScale(size_t(MMK_RECORDS) * MMT_CFM_SLOTS);
    for (size_t j = 0; j < cfmScale.size(); ++j)
        cfmScale[j] = 0.5f + 0.0625f * float(j % 23);

    PxgDevicePtr dLk = 0, dDf = 0, dAb = 0, dRecs = 0, dMat = 0, dCfm = 0, dScale = 0;
    PxgDevicePtr dDesc = 0, dIsland = 0, dBody = 0;
    shim.memAlloc(&dLk, lkImg.size() * sizeof(physx::PxgArticulationBlockLinkData));
    shim.memAlloc(&dDf, dfImg.size() * sizeof(physx::PxgArticulationBlockDofData));
    shim.memAlloc(&dAb, abImg.size() * sizeof(physx::PxgArticulationBlockData));
    shim.memAlloc(&dRecs, size_t(MMK_RECORDS) * sizeof(physx::PxgArticulation));
    shim.memAlloc(&dMat, size_t(MMK_RECORDS) * MMT_CFM_SLOTS * sizeof(PxSpatialMatrix));
    shim.memAlloc(&dCfm, size_t(MMK_RECORDS) * MMT_CFM_SLOTS * sizeof(PxReal));
    shim.memAlloc(&dScale, size_t(MMK_RECORDS) * MMT_CFM_SLOTS * sizeof(PxReal));
    shim.memAlloc(&dDesc, sizeof(physx::PxgArticulationCoreDesc));
    shim.memAlloc(&dIsland, island.size() * sizeof(physx::PxNodeIndex));
    shim.memAlloc(&dBody, bodySims.size() * sizeof(physx::PxgBodySim));
    if (!dLk || !dDf || !dAb || !dRecs || !dMat || !dCfm || !dScale || !dDesc || !dIsland || !dBody)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }

    std::vector<physx::PxgArticulation> recs(MMK_RECORDS);
    std::memset(recs.data(), 0, recs.size() * sizeof(physx::PxgArticulation));
    for (uint32_t r = 0; r < MMK_RECORDS; ++r)
    {
        recs[r].data.numLinks = physx::PxU16(0xFEE0u + (r & 0xFu));
        recs[r].spatialResponseMatrixW = reinterpret_cast<PxSpatialMatrix*>(
            dMat + PxgDevicePtr(size_t(r) * MMT_CFM_SLOTS * sizeof(PxSpatialMatrix)));
        recs[r].cfms = reinterpret_cast<PxReal*>(
            dCfm + PxgDevicePtr(size_t(r) * MMT_CFM_SLOTS * sizeof(PxReal)));
        recs[r].cfmScale = reinterpret_cast<PxReal*>(
            dScale + PxgDevicePtr(size_t(r) * MMT_CFM_SLOTS * sizeof(PxReal)));
    }

    std::vector<uint32_t> matPoison(size_t(MMK_RECORDS) * MMT_CFM_SLOTS
                                    * sizeof(PxSpatialMatrix) / 4);
    for (size_t j = 0; j < matPoison.size(); ++j)
        matPoison[j] = 0xAB000000u | uint32_t(j & 0xFFFFFFu);
    std::vector<uint32_t> cfmPoison(size_t(MMK_RECORDS) * MMT_CFM_SLOTS);
    for (size_t j = 0; j < cfmPoison.size(); ++j)
        cfmPoison[j] = 0xAC000000u | uint32_t(j & 0xFFFFFFu);

    shim.memcpyHtoD(dLk, lkImg.data(), lkImg.size() * sizeof(physx::PxgArticulationBlockLinkData));
    shim.memcpyHtoD(dDf, dfImg.data(), dfImg.size() * sizeof(physx::PxgArticulationBlockDofData));
    shim.memcpyHtoD(dAb, abImg.data(), abImg.size() * sizeof(physx::PxgArticulationBlockData));
    shim.memcpyHtoD(dRecs, recs.data(), recs.size() * sizeof(physx::PxgArticulation));
    shim.memcpyHtoD(dMat, matPoison.data(), matPoison.size() * 4);
    shim.memcpyHtoD(dCfm, cfmPoison.data(), cfmPoison.size() * 4);
    shim.memcpyHtoD(dScale, cfmScale.data(), cfmScale.size() * sizeof(PxReal));
    shim.memcpyHtoD(dIsland, island.data(), island.size() * sizeof(physx::PxNodeIndex));
    shim.memcpyHtoD(dBody, bodySims.data(), bodySims.size() * sizeof(physx::PxgBodySim));

    physx::PxgArticulationCoreDesc desc;
    std::memset(&desc, 0x5A, sizeof(desc));
    desc.mBodySimBufferDeviceData = reinterpret_cast<physx::PxgBodySim*>(dBody);
    desc.islandNodeIndices = reinterpret_cast<physx::PxNodeIndex*>(dIsland);
    desc.articulations = reinterpret_cast<physx::PxgArticulation*>(dRecs);
    desc.articulationOffset = ARTI_OFFSET;
    desc.nbArticulations = MMK_NB_ARTI;
    desc.mArticulationBlocks = reinterpret_cast<physx::PxgArticulationBlockData*>(dAb);
    desc.mArticulationLinkBlocks = reinterpret_cast<physx::PxgArticulationBlockLinkData*>(dLk);
    desc.mArticulationDofBlocks = reinterpret_cast<physx::PxgArticulationBlockDofData*>(dDf);
    desc.mMaxLinksPerArticulation = MMT_MAX_LINKS;
    desc.mMaxDofsPerArticulation = MMT_MAX_DOFS;
    shim.memcpyHtoD(dDesc, &desc, sizeof(desc));

    struct Params { uint64_t desc; } prm = { uint64_t(dDesc) };
    const uint32_t num1TBlocks = (MMK_NB_ARTI + 63u) / 64u;
    PXG_CHECK_MSG(num1TBlocks == 3u, "fixture assumes three 1T blocks, got %u", num1TBlocks);
    PXG_CHECK_MSG(shim.launchKernel(0, num1TBlocks * 2u, 1, 1, &prm, sizeof(prm))
                      == PxgResult::Success,
                  "launch computeMassMatrix1T");
    shim.streamSynchronize();

    auto buildWant = [&](int variant, std::vector<physx::PxgArticulationBlockLinkData>& wLk,
                         std::vector<uint32_t>& wMat, std::vector<uint32_t>& wCfm) {
        gMmtVariant = variant;
        wLk = lkImg;
        wMat = matPoison;
        wCfm = cfmPoison;
        for (uint32_t k = 0; k < MMK_NB_ARTI; ++k)
        {
            const uint32_t warp = k / 32u, lane = k % 32u, r = artiIdx[k];
            mmtMirror(wLk.data() + size_t(warp) * MMT_MAX_LINKS,
                      dfImg.data() + size_t(warp) * MMT_MAX_DOFS, abImg[warp],
                      reinterpret_cast<PxSpatialMatrix*>(wMat.data()) + size_t(r) * MMT_CFM_SLOTS,
                      reinterpret_cast<PxReal*>(wCfm.data()) + size_t(r) * MMT_CFM_SLOTS,
                      cfmScale.data() + size_t(r) * MMT_CFM_SLOTS, mmtNumLinks(k),
                      mmtFixBase(k), lane);
        }
        gMmtVariant = 0;
    };
    std::vector<physx::PxgArticulationBlockLinkData> wantLk;
    std::vector<uint32_t> wantMat, wantCfm;
    buildWant(0, wantLk, wantMat, wantCfm);

    // P9.86c coverage: the directed articulations 4 and 10 sit in warp 0.
    {
        const size_t d = mmtTwinDiff(buildWant, 1, wantLk, wantMat, wantCfm);
        std::printf("[ info ] mm launch: r2 twin (generic PxMax) changes %zu words\n", d);
        PXG_CHECK_MSG(d == 2, "r2 twin: the eFIX lane's selectMax tie must flip record cfms[1] "
                              "and link1.mCfm (got %zu words)", d);
    }
    // Both r3 twins: the fold as llvmpipe compiles it (-0 impulse lanes) and the
    // plain fold (+0 lanes).  Either way, impulse columns 0 and 3 of the spherical
    // lane come out with top (-0, -0, +0) where the engine has (+0, +0, +0).
    for (int variant = 2; variant <= 3; ++variant)
    {
        const size_t d = mmtTwinDiff(buildWant, variant, wantLk, wantMat, wantCfm);
        std::printf("[ info ] mm launch: r3 twin %d (folded qstZ seed) changes %zu words\n",
                    variant, d);
        PXG_CHECK_MSG(d == 8, "r3 twin %d: the spherical lane's impulse columns 0/3 must flip "
                              "top x/y in block and record (got %zu words)", variant, d);
    }
    mmtCheckDirectedWant(wantLk.data(), wantMat.data(), wantCfm.data(), artiIdx[4], 4u,
                         artiIdx[10], 10u, "launch");

    std::vector<physx::PxgArticulationBlockLinkData> gotLk(lkImg.size());
    std::vector<physx::PxgArticulationBlockData> gotAb(abImg.size());
    std::vector<physx::PxgArticulationBlockDofData> gotDf(dfImg.size());
    std::vector<uint32_t> gotMat(matPoison.size()), gotCfm(cfmPoison.size());
    shim.memcpyDtoH(gotLk.data(), dLk, gotLk.size() * sizeof(physx::PxgArticulationBlockLinkData));
    shim.memcpyDtoH(gotAb.data(), dAb, gotAb.size() * sizeof(physx::PxgArticulationBlockData));
    shim.memcpyDtoH(gotDf.data(), dDf, gotDf.size() * sizeof(physx::PxgArticulationBlockDofData));
    shim.memcpyDtoH(gotMat.data(), dMat, gotMat.size() * 4);
    shim.memcpyDtoH(gotCfm.data(), dCfm, gotCfm.size() * 4);

    // fully BITWISE (P9.86c) -- see the body test
    size_t bad = 0;
    auto cmp = [&](const char* what, const void* wp, const void* gp, size_t bytes) {
        const uint32_t* w = reinterpret_cast<const uint32_t*>(wp);
        const uint32_t* g = reinterpret_cast<const uint32_t*>(gp);
        for (size_t x = 0; x < bytes / 4; ++x)
        {
            if (w[x] == g[x])
                continue;
            if (bad < 12)
                std::printf("MM launch %s word %zu  ref %08x  gpu %08x\n", what, x, w[x], g[x]);
            ++bad;
        }
    };
    cmp("link", wantLk.data(), gotLk.data(),
        wantLk.size() * sizeof(physx::PxgArticulationBlockLinkData));
    cmp("dof", dfImg.data(), gotDf.data(), dfImg.size() * sizeof(physx::PxgArticulationBlockDofData));
    cmp("arti", abImg.data(), gotAb.data(), abImg.size() * sizeof(physx::PxgArticulationBlockData));
    cmp("record matrices", wantMat.data(), gotMat.data(), wantMat.size() * 4);
    cmp("record cfms", wantCfm.data(), gotCfm.data(), wantCfm.size() * 4);

    PXG_CHECK_MSG(bad == 0, "the kernel's whole block and record image matches the mirror");

    shim.memFree(dLk);
    shim.memFree(dDf);
    shim.memFree(dAb);
    shim.memFree(dRecs);
    shim.memFree(dMat);
    shim.memFree(dCfm);
    shim.memFree(dScale);
    shim.memFree(dDesc);
    shim.memFree(dIsland);
    shim.memFree(dBody);
}

}  // namespace
