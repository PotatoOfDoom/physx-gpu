// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

// sortTriIndicesBda.slang
//
// `sortTriangleIndices` (convexMeshPostProcess.cu:623) on the engine BDA ABI: a
// thin wrapper that hands each contact manager's triangle-index run to the warp
// radix sort verified in P8.47.
//
// THE DISPATCH REPRODUCES THE CUDA EXACTLY, with no ceiling — unlike the core
// kernel of P8.46c.  The host launches ceil(numTests / 2) blocks of two warps and
// passes numTests as `nbPairs` (PxgNarrowphaseCore.cpp:2649-2654), so the grid
// covers every contact manager and the CUDA has NO grid-stride loop to give up.
// One workgroup per block, a constant-trip loop over its two warps, and the
// coverage is identical.
//
// THE CHAINED LOAD IS FINE HERE.  `*orderedCvxTriIntermPtr` is an address read out
// of memory and then dereferenced, which is the P8.40 pattern — but the hazard is
// the GRID-STRIDE form, and this loop has a constant trip count, the same shape
// convexTrimeshMidphaseBda and convexTrimeshCoreBda already ship green with.
//
// SIX OF THE TWENTY-TWO SWEEP MUTANTS SURVIVE, and none of them is a fixture gap.
// They are recorded here so a later reader does not go hunting for a test that
// cannot exist:
//
//   * dropping EITHER early-out, or both.  `(0 + 3) / 4` and `(0xFFFFFFFF + 3) / 4`
//     are BOTH zero — the second because the round-up wraps — so a skipped run
//     reaches the sort with zero uint4s and every loop inside it has zero trips.
//     The guard is defensive, not load-bearing; no input can make it matter.  It
//     stays because the CUDA has it and because a future change to the rounding
//     would make it load-bearing again.
//   * putting the temp half at `input + count * 4` instead of `input + nbUint4 * 16`.
//     The two differ only when the count is not a multiple of four, and then the
//     wrong window starts exactly at the FIRST PADDING WORD — never on a live key —
//     and ends earlier than the right one, so it can neither corrupt a key nor
//     overrun the region.  What it does clobber is the sanitised tail, and the
//     scatter re-sanitises on every read (pxgSlangRadixWarp.slang:208), so the
//     clobber is erased before it is used.
//   * raising NP_TRIMESH_WARPS_PER_BLOCK to four.  The HOST fixes the grid at
//     ceil(nbPairs / 2); a wider block only stretches each block's range of
//     UNIQUE warp indices, so every pair is still covered exactly once and the
//     extra indices fall out on the bounds check.  Equivalent modulo wasted
//     workgroups — unkillable from inside the shader.
//   * dropping the barrier at the top of the loop.  Every groupshared access in
//     the sort is already bracketed by its own barriers (rsPass:192, the scatter's
//     tail at :256 and :266, rsReduceAdd at :94 and :96), so this one is provably
//     redundant TODAY.  It stays as a structural guard: it is the only thing that
//     would still be right if the sort's internal barriers moved.
//
// The one survivor that WAS a fixture gap — dropping `globalWarpIndex >= nbPairs`
// — is now killed by a decoy ConvexMeshPair one entry past nbPairs, which is what
// the engine's capacity-sized array really holds.

import pxgSlangRadixWarp;

static const uint NP_TRIMESH_WARPS_PER_BLOCK = 2u;

// (ConvexMeshPair*, PxU32**, PxU32) -> 8 @0, 8 @8, 4 @16, rounded to 24.
struct SortTriParams
{
    uint64_t cvxTrimeshPair;         // @0
    uint64_t orderedCvxTriIntermPtr; // @8
    uint     nbPairs;                // @16
    uint     _pad20;
}                                    // = 24

[[vk::push_constant]]
ConstantBuffer<SortTriParams> gSt;

[shader("compute")]
[numthreads(32, 1, 1)]
void sortTriangleIndices(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;

    // The ordered array's base, read once: a chained load, but outside any
    // grid-stride loop.
    const uint64_t ordered = Ptr<uint64_t>(gSt.orderedCvxTriIntermPtr)[0];

    for (uint sub = 0u; sub < NP_TRIMESH_WARPS_PER_BLOCK; ++sub)
    {
        GroupMemoryBarrierWithGroupSync();

        const uint globalWarpIndex = gid.x * NP_TRIMESH_WARPS_PER_BLOCK + sub;
        if (globalWarpIndex >= gSt.nbPairs)
            continue;

        // ConvexMeshPair, 56-byte stride: count @32, roundedStartIndex @40.  The
        // declaration's trailing numbers are running END offsets -- reading them
        // as field offsets puts count at 36 (`cmIndex`) and the start at 44
        // (`pad`).  P8.46b nearly took that route; both are pinned by fixture
        // there and by this comment here.
        const uint64_t pairAddr = gSt.cvxTrimeshPair + uint64_t(globalWarpIndex) * 56UL;
        const uint nbTriangles = Ptr<uint>(pairAddr + 32UL)[0];
        const uint startIndex = Ptr<uint>(pairAddr + 40UL)[0];

        // Nothing to sort, or the run was cached from last frame.  Warp-uniform,
        // so `continue` keeps the whole group together for the next barrier.
        if (nbTriangles == 0u || nbTriangles == 0xffffffffu)
            continue;

        const uint numUint4 = (nbTriangles + 3u) / 4u;

        // The temp half sits immediately after the rounded key run, which is what
        // the midphase's doubled counter reserved for it.
        const uint64_t input = ordered + uint64_t(startIndex) * 4UL;
        const uint64_t temp = input + uint64_t(numUint4) * 16UL;

        rsSortSingleWarpKeysOnly(lane, input, nbTriangles, numUint4, temp);
    }
}
