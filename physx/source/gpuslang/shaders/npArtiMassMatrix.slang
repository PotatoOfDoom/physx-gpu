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
import pxgSlangArticulationBlocks;
import pxgSlangSpatialMatrix;

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
// The fixture entry points
// ===========================================================================
//
// Neither of these is an engine kernel; they are the smallest shapes that drive
// the three helpers above with the same lane arithmetic `computeMassMatrix1T`
// will.  The kernel itself is P8.91c.

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

    for (uint m = gMmWrite.zeroMask; m != 0u; m = pxgClearLowestSetBit(m))
    {
        const uint srcLane = uint(pxgLowestSetIndex(m));
        const uint artiInd = pxgShflSyncU(MM_FULL_MASK, lane, articulationIndex, int(srcLane), 32u);

        mmWarpCopyU(ptrTable[artiInd], 0u, gMmWrite.zeroBytes, lane);
    }
}
