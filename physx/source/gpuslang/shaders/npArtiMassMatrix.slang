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

// npArtiMassMatrix.slang
//
// P8.91b: the warp-cooperative machinery `computeMassMatrix1T` is built out of,
// ported ahead of the kernel itself so it can carry its own fixture.
//
//   * `writeMatrix` (forwardDynamic2.cu:1699) -- a scatter of one
//     `PxSpatialMatrix` per ACTIVE lane to a per-lane destination pointer,
//     staged eight at a time through shared memory so the actual stores are
//     coalesced across the warp;
//   * `warpCopy<uint>(dst, 0, sizeof(PxSpatialMatrix))` (copy.cuh:54) -- the
//     lane-strided fill the fixed-base path zeroes a record matrix with;
//   * the `for (m = mask; m; m = clearLowestSetBit(m))` set-bit walk with a
//     `__shfl_sync` off `lowestSetIndex(m)`, which is how the fixed-base path
//     turns a ballot into a sequence of articulation indices.
//
// ===========================================================================
// THIS IS THE FIRST WARP-COOPERATIVE ARTICULATION CODE
// ===========================================================================
//
// P8.83-P8.90 were all one thread per articulation with no cross-lane traffic,
// which is why none of them needed the warp library.  Everything below does, and
// the failure mode changes with it: not a wrong offset but a lane-divergence or
// mask bug, and the platform it has to survive is llvmpipe at **subgroup 8**, so
// `pxgSyncWarp` lowers to a WORKGROUP barrier rather than to nothing.
//
// That makes barrier uniformity load-bearing rather than advisory.  Every loop
// below that contains a `pxgSyncWarp` is bounded by a warp-uniform value —
// `countbits(mask)` — and every call site in the kernel to come is inside a
// warp-uniform branch (`if (notFixMask)`, `linkID < maxLinksInWarp`, both
// derived from ballots and reductions).  A port that made any of those per-lane
// would deadlock the workgroup on llvmpipe while working on a real warp.
//
// ===========================================================================
// THE TRAPS IN `writeMatrix`, EACH OF WHICH GETS A MUTANT
// ===========================================================================
//
//  1. **`off = offset - i` is UNSIGNED, and that is what excludes the lanes of
//     earlier passes.**  A lane whose rank is below the current pass wraps to a
//     huge number and fails `off < NbPerPass`.  Written with a signed type it
//     would pass the test with a negative index and stage out of bounds.
//  2. **`offset` is the lane's RANK WITHIN THE MASK, not its lane index.**  They
//     coincide only for a mask whose set bits are the low lanes, so the fixture's
//     mask is deliberately scattered.
//  3. **The staging is guarded by `isActive`.**  An inactive lane's rank equals
//     the rank of the next active lane above it, so an unguarded stage
//     overwrites a live slot with garbage rather than landing somewhere harmless.
//  4. **Lanes 0..17 each move one `uint2`** — 18 * 8 == 144 == the size of a
//     `PxSpatialMatrix`.  Both the 18 and the pairing are transcribed rather than
//     recomputed.
//  5. **The trailing `pxgSyncWarp` is not decoration.**  `shMatrix` and `outBuff`
//     are read and written in the same loop, so without it a fast lane can begin
//     staging pass `i+1` while a slow lane is still copying pass `i` out.  The
//     engine's own comment says exactly this.
//
//  6. **`NbToProcess = min(count - i, NbPerPass)` is a real bound**, and it took
//     the sweep to establish that.  The tempting argument is that dropping the
//     `min` only re-runs the last pass's leftover slots, and that because a staged
//     matrix and its destination pointer travel together in the same slot, a stale
//     slot re-writes an earlier result with its own value — idempotent, and
//     therefore invisible.  That argument is half right, and the half it misses is
//     the one that matters: on the **first** pass of a call whose `count` is below
//     eight, the leftover slots have never been written at all and the destination
//     is whatever shared memory happened to hold.  The fixture's 5-bit mask is
//     exactly that case, and the mutant dies on it.
//
// What IS equivalent, measured rather than assumed: a pass stride below the
// staging window (`i += 7` rather than 8).  At pass `i`, slot `j` always holds
// rank `i + j`, so every copy is a correct pair whatever the stride — a shorter
// stride merely repeats work.  A stride of NINE is not equivalent; it skips rank
// 8 entirely.  Likewise, dropping `isActive` from the staging guard cannot change
// the copied-out image, because ranks step up only AFTER an active lane, so an
// inactive lane always shares its slot with a strictly higher active one (and the
// lanes above the top active one land exactly one past `nbToProcess`, which is
// never copied).  The guard is transcribed anyway: that leaves a same-address
// groupshared race whose resolution is this driver's, not the memory model's.

#include "pxgSlangWarpInc.slang"

import pxgSlangCommon;
import pxgSlangArticulationDesc;
import pxgSlangArticulationBlocks;
import pxgSlangSpatialMatrix;
import pxgSlangArticulationDynamic;

static const uint MM_WARP = 32u;
static const uint MM_FULL_MASK = 0xFFFFFFFFu;

// `const PxU32 NbPerPass = 8;`
static const uint MM_NB_PER_PASS = 8u;

// sizeof(PxSpatialMatrix) == 144 == 36 floats == 18 uint2.
static const uint MM_MATRIX_FLOATS = 36u;
static const uint MM_MATRIX_BYTES = 144u;
static const uint MM_UINT2_LANES = 18u;

// `__shared__ PxSpatialMatrix shMatrix[NbPerPass];`
// `__shared__ PxSpatialMatrix* outBuff[NbPerPass];`
//
// The CUDA declares these inside the function; `__shared__` is per BLOCK, and
// `computeMassMatrix1T` launches (32, 1, 1) — one warp per block — so per block
// and per warp are the same thing there.  The port keeps that: these arrays are
// correct only for a single-warp workgroup, which is why `MM_WARPS_PER_BLOCK`
// will be 1 and why this note exists.
groupshared float gsMmMatrix[MM_NB_PER_PASS * MM_MATRIX_FLOATS];
groupshared uint gsMmOutLo[MM_NB_PER_PASS];
groupshared uint gsMmOutHi[MM_NB_PER_PASS];

// `AbMatrix6x6::e` is flat in the same order as `PxSpatialMatrix::column`
// (`e[i * 6 + j] == column[i][j]`, P8.89), so the byte image the copy below
// produces is the engine's own — no repacking on the way through shared memory.
void mmWriteMatrix(AbMatrix6x6 matrix, uint64_t dstMatrix, uint mask, uint lane, bool isActive)
{
    const uint count = countbits(mask);
    const uint offset = pxgWarpRankInMask(mask, lane);   // warpScanExclusive(mask, lane)

    for (uint i = 0u; i < count; i += MM_NB_PER_PASS)
    {
        // UNSIGNED on purpose -- trap 1
        const uint off = offset - i;
        if (isActive && off < MM_NB_PER_PASS)
        {
            for (uint k = 0u; k < MM_MATRIX_FLOATS; ++k)
                gsMmMatrix[off * MM_MATRIX_FLOATS + k] = matrix.e[k];

            gsMmOutLo[off] = uint(dstMatrix & 0xFFFFFFFFull);
            gsMmOutHi[off] = uint(dstMatrix >> 32);
        }

        pxgSyncWarp(MM_FULL_MASK);

        const uint nbToProcess = min(count - i, MM_NB_PER_PASS);
        for (uint j = 0u; j < nbToProcess; ++j)
        {
            Ptr<uint> dst = Ptr<uint>(uint64_t(gsMmOutLo[j]) | (uint64_t(gsMmOutHi[j]) << 32));

            // one uint2 per lane, lanes 0..17 -- trap 4
            if (lane < MM_UINT2_LANES)
            {
                const uint w = lane * 2u;
                dst[w + 0u] = asuint(gsMmMatrix[j * MM_MATRIX_FLOATS + w + 0u]);
                dst[w + 1u] = asuint(gsMmMatrix[j * MM_MATRIX_FLOATS + w + 1u]);
            }
        }

        // shMatrix and outBuff are read and written in the same loop -- trap 5
        pxgSyncWarp(MM_FULL_MASK);
    }
}

// copy.cuh:54, `warpCopy<uint>(dst, value, totalSize)`.  `requiredThreads` is a
// COUNT OF ELEMENTS derived from the byte size, and the stride is the warp width
// rather than the loop bound -- a grid-stride loop over one warp.
void mmWarpCopyU(uint64_t dst, uint value, uint totalBytes, uint lane)
{
    Ptr<uint> d = Ptr<uint>(dst);
    const uint requiredThreads = totalBytes / 4u;
    for (uint i = lane; i < requiredThreads; i += MM_WARP)
        d[i] = value;
}

// ===========================================================================
// P8.91c: the body of `computeMassMatrix1T` (forwardDynamic2.cu:1736-1930)
// ===========================================================================
//
// The impulse-response matrix of every link: apply a unit impulse on each of the
// six spatial axes in turn, propagate it up to the parent, take the parent's
// already-computed response, and propagate the resulting acceleration back down.
// Six columns per link, one link at a time, the whole warp in lockstep.
//
// Zero new ABI for the FOURTH tranche running: every field it touches —
// `mInvSpatialArticulatedInertia` (0), `mSpatialResponseMatrix` (3584), `mCfm`
// (28160), `mJointOffset` (28544), `mParents` (28672), `mDofs` (28800),
// `mWorldMotionMatrix` (1152), `mIsInvDW` (2688), `mIsW` (3456),
// `mInvStIsT_{x,y,z}` (4992/5120/5248) — and the record's
// `spatialResponseMatrixW` (176), `cfms` (336) and `cfmScale` (344) were already
// mapped.
//
// ===========================================================================
// SEVEN THINGS TO TRANSCRIBE RATHER THAN TIDY, EACH A MUTANT
// ===========================================================================
//
//  1. **The grid is over ARTICULATION BLOCKS, not articulations**, and the guard
//     is `globalWarpIndex < nbArticulationBlocks` with a SECOND per-lane `active`
//     test.  P8.83-P8.88 all returned early on `globalThreadIndex >=
//     nbArticulations`; doing that here removes lanes from every ballot and every
//     `writeMatrix` they are required to take part in.
//  2. **`articulationIndex` comes from the BLOCK** (`mArticulationIndex[lane]`),
//     not from the island-node/bodySim chain P8.83 and P8.85 walk and not from
//     the record.  Third kernel, third source.
//  3. **The fixed-base path zeroes the record through `warpCopy` but the block
//     through `zeroSpatialMatrix`** — two mechanisms for the same logical zero,
//     because the record's matrix is one contiguous 144-byte object per
//     articulation while the block's is a 32-lane interleave.
//  4. **`cfm` is a max over three DIAGONAL entries, and the index pattern differs
//     between the root and the links.**  Root: `bottomLeft.{c0.x, c1.y, c2.z}` of
//     the INVERSE articulated inertia.  Link: `response.column[i][3 + i]` for
//     i = 0,1,2.  The two are not the same slots of the same object.
//  5. **The unit impulse is NEGATED for the links and NOT for the root.**  The
//     root builds its response from `+1` on each axis of the inverse inertia
//     directly; the link path drives `propagateImpulseW` with `-1`.
//  6. **`maxLinksInWarp` is a warp MAX, so lanes iterate past their own link
//     count** — with `thisThreadActive` false — precisely so they stay in the
//     ballot and in `writeMatrix`.
//  7. **`response` is left UNINITIALISED for an inactive lane** and handed to
//     `writeMatrix` anyway, which never reads it for such a lane.  The port zeroes
//     it instead of leaving it undefined, which is a strict improvement and is
//     unobservable for exactly that reason.
//
// And two the CUDA itself leaves lying around, dropped by the port with a note
// rather than transcribed: `articulationsInBlock` is computed and never used, and
// the mixture of `threadIdx.x` with `threadIndexInWarp` inside the link loop is
// an engine inconsistency that is only safe because `blockDim.x == 32` (the same
// unification P8.88 recorded).
//
// ---------------------------------------------------------------------------
// The signed-zero seed, and why the CALLER fixes it
// ---------------------------------------------------------------------------
//
// `PxReal qstZ[3] = { 0.f, 0.f, 0.f }` is the literal-zero accumulator P8.87
// identified: `adPropagateImpulseW` does `qstZ[ind] = qstZ[ind] + stZ`, so with a
// single accumulated term the driver's `x + 0.0` fold makes a `-0` term survive
// as `-0` where the engine's C++ returns `+0`.  `pxgSlangArticulationDynamic`'s
// header leaves this to the caller (its other callers seed differently); P9.86
// canonicalises qstZ with a select right after the call in the link loop below.

// Load the six per-dof quantities the link path needs.  `dof` is a PxU8 in the
// block and is read as one -- a word read here picks up `mDisableGravity` and
// friends, which is the P8.88 hazard that cost a sweep round.
struct MmDofData
{
    AbSpatialVector motionV[3];
    float3 invStIsT[3];
    AbSpatialVector isInvD[3];
    AbSpatialVector isW[3];
};

MmDofData mmLoadDofData(uint64_t dofData, uint dof, uint lane)
{
    MmDofData d;
    for (uint i = 0u; i < 3u; ++i)
    {
        d.motionV[i] = asmSVec(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f));
        d.invStIsT[i] = float3(0.0f, 0.0f, 0.0f);
        d.isInvD[i] = asmSVec(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f));
        d.isW[i] = asmSVec(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f));

        if (i < dof)
        {
            const uint64_t b = dofData + uint64_t(i) * PXG_AB_DOF_SIZE;
            float3 t, bo;
            abLoadSpatialVector(b, PXG_AB_DOF_mWorldMotionMatrix, lane, t, bo);
            d.motionV[i] = asmSVec(t, bo);
            d.invStIsT[i] = float3(abF32(b, PXG_AB_DOF_mInvStIsT_x, lane),
                                   abF32(b, PXG_AB_DOF_mInvStIsT_y, lane),
                                   abF32(b, PXG_AB_DOF_mInvStIsT_z, lane));
            abLoadSpatialVector(b, PXG_AB_DOF_mIsInvDW, lane, t, bo);
            d.isInvD[i] = asmSVec(t, bo);
            abLoadSpatialVector(b, PXG_AB_DOF_mIsW, lane, t, bo);
            d.isW[i] = asmSVec(t, bo);
        }
    }
    return d;
}

// `Cm::UnAlignedSpatialVector::operator[]`: 0..2 index the TOP half, 3..5 the
// bottom one.  Used to build `vec[i] = +/-1` without a six-way branch.
AbSpatialVector mmAxis(uint i, float s)
{
    return asmSVec(float3(i == 0u ? s : 0.0f, i == 1u ? s : 0.0f, i == 2u ? s : 0.0f),
                   float3(i == 3u ? s : 0.0f, i == 4u ? s : 0.0f, i == 5u ? s : 0.0f));
}

// `response.column[i][0..5] = (r.top, r.bottom)` -- one COLUMN per axis, and
// `AbMatrix6x6::e[i * 6 + j] == column[i][j]`.
void mmSetColumn(inout AbMatrix6x6 m, uint i, AbSpatialVector r)
{
    m.e[i * 6u + 0u] = r.top.x;
    m.e[i * 6u + 1u] = r.top.y;
    m.e[i * 6u + 2u] = r.top.z;
    m.e[i * 6u + 3u] = r.bottom.x;
    m.e[i * 6u + 4u] = r.bottom.y;
    m.e[i * 6u + 5u] = r.bottom.z;
}

// PxMax (PxMath.h:82): the FLOAT specialisation, intrinsics::selectMax
// (PxUnixMathIntrinsics.h:122) `a > b ? a : b` -- NOT the generic template's
// `a < b ? b : a`, which keeps the OTHER operand on a +-0 tie and on NaN.
// P9.86: a fixed-base link whose COM cannot translate has an all-zero response
// diagonal of mixed zero signs, and the CUDA's nesting keeps the LAST one.
// Spelled out because `max()` (FMax) would be a third answer on the tie.
float mmPxMax(float a, float b) { return a > b ? a : b; }

// The whole kernel body, with the block bases and the record pointers resolved.
// Split out from the entry point so the fixture can drive it with hand-fed bases
// exactly as P8.88's loop mirror does.
void mmComputeMassMatrix(uint64_t artiBlock, uint64_t linkBlocks, uint64_t dofBlocks,
                         uint64_t articulationsBase, bool active, uint lane)
{
    // zero for an inactive lane -- it still takes part in the reduction below
    const uint numLinks = active ? abU32(artiBlock, PXG_AB_mNumLinks, lane) : 0u;

    // the articulation index comes from the BLOCK -- trap 2.  An inactive lane
    // keeps the invalid id and null record pointers; nothing below dereferences
    // them, and the CUDA's own `assert(artiInd != invalidArticulationId)` says the
    // walk must never reach one.
    uint articulationIndex = 0xFFFFFFFFu;
    uint64_t responseMatrices = 0;
    uint64_t cfms = 0;
    uint64_t cfmScale = 0;
    if (active)
    {
        articulationIndex = abU32(artiBlock, PXG_AB_mArticulationIndex, lane);
        const uint64_t a = articulationsBase + uint64_t(articulationIndex) * PXG_ARTI_STRIDE;
        responseMatrices = artiPtrField(a, PXG_ARTI_SPATIAL_RESPONSE_MATRIX_W);
        cfms = artiPtrField(a, PXG_ARTI_CFMS);
        cfmScale = artiPtrField(a, PXG_ARTI_CFM_SCALE);
    }

    // The flags are read UNCONDITIONALLY, exactly as the CUDA reads them: the
    // block slice is allocated for all 32 lanes, and it is `active` in the line
    // below that keeps an inactive lane out of the ballot.  Guarding the read
    // instead would make that `&& active` dead — and the sweep says so, because
    // the fixture's poison flags have the fix bit set, so an inactive lane that
    // reached the fixed-base walk would resolve `articulationIndex` 0xFFFFFFFF.
    const uint flags = abU32(artiBlock, PXG_AB_mFlags, lane);
    const bool fixBase = active && (flags & PXG_ARTI_FLAG_FIX_BASE) != 0u;
    const bool notFixBase = !fixBase && active;

    const uint fixMask = pxgBallotSync(MM_FULL_MASK, lane, fixBase);
    const uint notFixMask = pxgBallotSync(MM_FULL_MASK, lane, notFixBase);

    if (fixMask != 0u)
    {
        // the BLOCK's matrix: a 32-lane interleave, so one lane's slots -- trap 3
        if (fixBase)
            abZeroSpatialMatrix6x6(linkBlocks, PXG_AB_LINK_mSpatialResponseMatrix, lane);

        // The RECORD's matrix: 144 contiguous bytes per articulation, so the whole
        // warp clears one at a time, walking the ballot's set bits.
        //
        // PLATFORM RULE P8.40 APPLIES HERE, and it is what this loop is shaped
        // around.  The CUDA broadcasts the ARTICULATION INDEX and then resolves
        // `articulations[artiInd].spatialResponseMatrixW` inside the loop — a
        // pointer LOADED through a BDA and immediately dereferenced, one level
        // down, inside a loop.  On llvmpipe that faults: measured, with the
        // identical pointer value, resolving it outside the loop works and
        // resolving it inside does not.  So the port broadcasts the ALREADY
        // RESOLVED pointer instead, in two halves.
        //
        // That is the same pointer, not an approximation: `responseMatrices` was
        // set to exactly `articulations[articulationIndex].spatialResponseMatrixW`
        // for every ACTIVE lane, and only active lanes can be in `fixMask` — the
        // CUDA's own `assert(artiInd != invalidArticulationId)` says so.
        for (uint m = fixMask; m != 0u; m = pxgClearLowestSetBit(m))
        {
            const int src = pxgLowestSetIndex(m);
            const uint lo = pxgShflSyncU(MM_FULL_MASK, lane,
                                         uint(responseMatrices & 0xFFFFFFFFull), src, 32u);
            const uint hi = pxgShflSyncU(MM_FULL_MASK, lane, uint(responseMatrices >> 32), src,
                                         32u);
            mmWarpCopyU(uint64_t(lo) | (uint64_t(hi) << 32), 0u, MM_MATRIX_BYTES, lane);
        }

        if (fixBase)
        {
            (Ptr<float>(cfms))[0] = 0.0f;
            abStoreF32(linkBlocks, PXG_AB_LINK_mCfm, lane, 0.0f);
        }
    }

    if (notFixMask != 0u)
    {
        AbMatrix6x6 response = abMatrix6x6Zero();   // trap 7

        if (notFixBase)
        {
            const AbSpatialMatrix inv =
                asmLoadSpatialMatrix(artiBlock, PXG_AB_mInvSpatialArticulatedInertia, lane);

            // the root's unit impulses are POSITIVE -- trap 5
            for (uint i = 0u; i < 6u; ++i)
                mmSetColumn(response, i, asmMulVec(inv, mmAxis(i, 1.0f)));

            abStoreSpatialMatrix6x6(linkBlocks, PXG_AB_LINK_mSpatialResponseMatrix, lane, response);

            // the root's cfm reads the INVERSE INERTIA's bottomLeft diagonal -- trap 4
            const float cfmVal =
                (Ptr<float>(cfmScale))[0]
                * mmPxMax(inv.bottomLeft.c0.x, mmPxMax(inv.bottomLeft.c1.y, inv.bottomLeft.c2.z));

            (Ptr<float>(cfms))[0] = cfmVal;
            abStoreF32(linkBlocks, PXG_AB_LINK_mCfm, lane, cfmVal);
        }

        mmWriteMatrix(response, responseMatrices, notFixMask, lane, notFixBase);
    }

    // a warp MAX, so short articulations keep iterating with thisThreadActive
    // false rather than dropping out of the ballots -- trap 6
    const uint maxLinksInWarp = pxgWarpReduceMaxU(MM_FULL_MASK, lane, numLinks, 5u);

    for (uint linkID = 1u; linkID < maxLinksInWarp; ++linkID)
    {
        AbMatrix6x6 response = abMatrix6x6Zero();   // trap 7

        const bool thisThreadActive = linkID < numLinks;
        const uint activeThreadMask = pxgBallotSync(MM_FULL_MASK, lane, thisThreadActive);

        if (thisThreadActive)
        {
            const uint64_t thisLink = linkBlocks + uint64_t(linkID) * PXG_AB_LINK_SIZE;
            const uint jointOffset = abU32(thisLink, PXG_AB_LINK_mJointOffset, lane);
            const uint64_t dofData = dofBlocks + uint64_t(jointOffset) * PXG_AB_DOF_SIZE;

            const uint parentId = abU32(thisLink, PXG_AB_LINK_mParents, lane);
            const float3 rw = float3(abF32(thisLink, PXG_AB_LINK_mRw_x, lane),
                                     abF32(thisLink, PXG_AB_LINK_mRw_y, lane),
                                     abF32(thisLink, PXG_AB_LINK_mRw_z, lane));
            const uint dof = abU8(thisLink, PXG_AB_LINK_mDofs, lane);

            const MmDofData d = mmLoadDofData(dofData, dof, lane);

            // the parent's response, written by an earlier iteration of THIS loop
            // (`parents[linkID] < linkID` is the engine's invariant)
            const AbMatrix6x6 parentResponse = abLoadSpatialMatrix6x6(
                linkBlocks + uint64_t(parentId) * PXG_AB_LINK_SIZE,
                PXG_AB_LINK_mSpatialResponseMatrix, lane);

            for (uint i = 0u; i < 6u; ++i)
            {
                // the impulse is NEGATED here -- trap 5
                const AbSpatialVector vec = mmAxis(i, -1.0f);

                // the literal-zero seed -- see the signed-zero note in the header
                float qstZ[3] = { 0.0f, 0.0f, 0.0f };

                // (1) propagate the impulse up to the parent
                const AbSpatialVector Zp =
                    adPropagateImpulseW(d.isInvD, rw, d.motionV, vec, dof, qstZ);

                // P9.86: each qstZ[k] is `+0 + stZ` (one add from the literal +0 seed,
                // articulationDynamic.cuh:178), which IEEE never leaves at -0; llvmpipe
                // forwards the seed and folds the add, so canonicalise -0 -> +0 as a
                // select (every non-zero, inf and NaN passes through unchanged).
                qstZ[0] = (qstZ[0] == 0.0f) ? 0.0f : qstZ[0];
                qstZ[1] = (qstZ[1] == 0.0f) ? 0.0f : qstZ[1];
                qstZ[2] = (qstZ[2] == 0.0f) ? 0.0f : qstZ[2];

                // (2) the parent's deltaV response, negated, propagated back down
                // (forwardDynamic2.cu:1910).  P9.86: the product's zero-seeded rows
                // (PxSpatialMatrix.h:64-70) are canonicalised inside asmM6MulVec itself.
                const AbSpatialVector Zr = asmSVecScale(asmM6MulVec(parentResponse, Zp), -1.0f);
                const AbSpatialVector deltaV =
                    adPropagateAccelerationW(rw, d.invStIsT, d.motionV, Zr, dof, d.isW, qstZ);

                mmSetColumn(response, i, deltaV);
            }

            // the link's cfm reads the RESPONSE's off-block diagonal -- trap 4
            const float cfmVal =
                (Ptr<float>(cfmScale))[linkID]
                * mmPxMax(abM6Get(response, 0u, 3u),
                          mmPxMax(abM6Get(response, 1u, 4u), abM6Get(response, 2u, 5u)));

            (Ptr<float>(cfms))[linkID] = cfmVal;
            abStoreF32(thisLink, PXG_AB_LINK_mCfm, lane, cfmVal);

            abStoreSpatialMatrix6x6(thisLink, PXG_AB_LINK_mSpatialResponseMatrix, lane, response);
        }

        mmWriteMatrix(response, responseMatrices + uint64_t(linkID) * MM_MATRIX_BYTES,
                      activeThreadMask, lane, thisThreadActive);
    }
}

// ---------------------------------------------------------------------------
// computeMassMatrix1T (forwardDynamic2.cu:1736)
// ---------------------------------------------------------------------------
//
// Block (32, 1, 1), grid `num1TBlocks * 2` — the same shape as P8.88's kernel,
// and the reason `MM_WARPS_PER_BLOCK` is its own constant rather than shared with
// P8.85's 2.  Here it is also load-bearing for a second reason: `writeMatrix`'s
// shared arrays are per BLOCK, so a second warp in the block would scribble on
// the first one's staging slots.

struct MmLaunchParams
{
    uint64_t scDesc;   // @0  PxgArticulationCoreDesc*
};

[[vk::push_constant]]
ConstantBuffer<MmLaunchParams> gMmLaunch;

static const uint MM_WARPS_PER_BLOCK = 1u;

[shader("compute")]
[numthreads(32, 1, 1)]
void computeMassMatrix1T(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint64_t d = gMmLaunch.scDesc;

    const uint lane = groupThreadId.x & 31u;
    const uint globalWarpIndex = groupId.x * MM_WARPS_PER_BLOCK + groupThreadId.y;
    const uint globalThreadIndex = globalWarpIndex * 32u + groupThreadId.x;

    const uint nbArticulations = artiU32Field(d, PXG_ARTI_CORE_NB_ARTICULATIONS);
    const uint nbArticulationBlocks = (nbArticulations + 31u) / 32u;

    // the guard is over BLOCKS, and the per-lane test is separate -- trap 1
    if (globalWarpIndex >= nbArticulationBlocks)
        return;

    const uint maxLinks = artiU32Field(d, PXG_ARTI_CORE_MAX_LINKS);
    const uint maxDofs = artiU32Field(d, PXG_ARTI_CORE_MAX_DOFS);

    const uint64_t w = uint64_t(globalWarpIndex);
    const uint64_t artiBlock = artiPtrField(d, PXG_ARTI_CORE_ARTICULATION_BLOCKS)
                             + w * PXG_AB_BLOCK_SIZE;
    const uint64_t linkBlocks = artiPtrField(d, PXG_ARTI_CORE_LINK_BLOCKS)
                              + w * uint64_t(maxLinks) * PXG_AB_LINK_SIZE;
    const uint64_t dofBlocks = artiPtrField(d, PXG_ARTI_CORE_DOF_BLOCKS)
                             + w * uint64_t(maxDofs) * PXG_AB_DOF_SIZE;

    const uint64_t articulations = artiPtrField(d, PXG_ARTI_CORE_ARTICULATIONS);

    const bool active = globalThreadIndex < nbArticulations;

    mmComputeMassMatrix(artiBlock, linkBlocks, dofBlocks, articulations, active, lane);
}

// The same body with the block bases handed in directly, one warp, so the fixture
// can drive it without building a descriptor -- the P8.88 loop-mirror arrangement.
struct MmBodyParams
{
    uint64_t artiBlock;      // @0
    uint64_t linkBlocks;     // @8
    uint64_t dofBlocks;      // @16
    uint64_t articulations;  // @24
    uint nbLanes;            // @32
    uint pad0;               // @36
    uint pad1;               // @40
    uint pad2;               // @44 -> 48 bytes
};

[[vk::push_constant]]
ConstantBuffer<MmBodyParams> gMmBody;

[shader("compute")]
[numthreads(32, 1, 1)]
void artiMassMatrixBody(uint3 groupThreadId : SV_GroupThreadID)
{
    const uint lane = groupThreadId.x;
    mmComputeMassMatrix(gMmBody.artiBlock, gMmBody.linkBlocks, gMmBody.dofBlocks,
                        gMmBody.articulations, lane < gMmBody.nbLanes, lane);
}

// ===========================================================================
// The fixture entry points
// ===========================================================================
//
// Neither of these is an engine kernel; they are the smallest shapes that drive
// the three helpers above with the same lane arithmetic `computeMassMatrix1T`
// does.

struct MmWriteParams
{
    uint64_t srcFloats;   // @0   float[32 * 36]        -- one matrix per lane
    uint64_t dstTable;    // @8   uint64_t[32]          -- one destination per lane
    uint64_t idxTable;    // @16  uint32_t[32]          -- the "articulation index" per lane
    uint64_t ptrTable;    // @24  uint64_t[32]          -- indexed BY the broadcast index
    uint activeMask;      // @32  the ballot the scatter is driven by
    uint zeroMask;        // @36  the ballot the set-bit walk is driven by
    uint zeroBytes;       // @40  how much of each record the walk clears
    uint pad0;            // @44 -> 48 bytes
};

[[vk::push_constant]]
ConstantBuffer<MmWriteParams> gMmWrite;

// `writeMatrix(response, responseMatrices + linkID, activeThreadMask, lane,
//  thisThreadActive)`.
//
// The destination table is supplied per lane rather than derived, so a port that
// confused the lane with its rank in the mask scatters to the wrong records
// instead of merely to the wrong shared slot.
[shader("compute")]
[numthreads(32, 1, 1)]
void artiMassMatrixWrite(uint3 groupThreadId : SV_GroupThreadID)
{
    const uint lane = groupThreadId.x;

    Ptr<float> src = Ptr<float>(gMmWrite.srcFloats + uint64_t(lane) * uint64_t(MM_MATRIX_FLOATS) * 4);

    AbMatrix6x6 m = abMatrix6x6Zero();
    for (uint k = 0u; k < MM_MATRIX_FLOATS; ++k)
        m.e[k] = src[k];

    Ptr<uint64_t> dstTable = Ptr<uint64_t>(gMmWrite.dstTable);
    const uint64_t dst = dstTable[lane];

    const bool isActive = (gMmWrite.activeMask & (1u << lane)) != 0u;

    mmWriteMatrix(m, dst, gMmWrite.activeMask, lane, isActive);
}

// The fixed-base block's record clearing, minus the record:
//
//   for (i = fixMask; i != 0; i = clearLowestSetBit(i))
//       artiInd = __shfl_sync(FULL_MASK, articulationIndex, lowestSetIndex(i));
//       warpCopy<uint>(articulations[artiInd].spatialResponseMatrixW, 0, 144);
//
// The whole warp participates in every iteration -- the loop is over the MASK,
// not over the lane -- which is what makes the shuffle and the copy legal, and
// what makes the trip count warp-uniform so the barriers inside `pxgShflSyncU`
// are reached by every thread.
[shader("compute")]
[numthreads(32, 1, 1)]
void artiMassMatrixZeroWalk(uint3 groupThreadId : SV_GroupThreadID)
{
    const uint lane = groupThreadId.x;

    Ptr<uint> idxTable = Ptr<uint>(gMmWrite.idxTable);
    const uint articulationIndex = idxTable[lane];

    Ptr<uint64_t> ptrTable = Ptr<uint64_t>(gMmWrite.ptrTable);

    // The record pointer is resolved ONCE, outside the loop, and the loop
    // broadcasts the resolved value rather than the index -- P8.40 again.  See
    // the mass-matrix walk above for the measurement: the pointer LOAD is what
    // must not happen inside the loop, and re-resolving `ptrTable[artiInd]` per
    // iteration is exactly that shape.
    const uint64_t myRecord = ptrTable[articulationIndex];

    for (uint m = gMmWrite.zeroMask; m != 0u; m = pxgClearLowestSetBit(m))
    {
        const int src = pxgLowestSetIndex(m);
        const uint lo = pxgShflSyncU(MM_FULL_MASK, lane, uint(myRecord & 0xFFFFFFFFull), src, 32u);
        const uint hi = pxgShflSyncU(MM_FULL_MASK, lane, uint(myRecord >> 32), src, 32u);

        mmWarpCopyU(uint64_t(lo) | (uint64_t(hi) << 32), 0u, gMmWrite.zeroBytes, lane);
    }
}
