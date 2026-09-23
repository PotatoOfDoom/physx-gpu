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

// pxgSlangTriTriBlock -- P9.83b-5 (13/n).
//
//   The BLOCK collectives of doTriangleTriangleCollision: the two reduction.cuh
//   helpers it calls, the warp minIndex its patch seeding is built on, and the
//   cross-warp seed pick at convexMeshMidphase.cu:879-902.
//
// ---------------------------------------------------------------------------
// THIS FILE IS #included, NOT imported, AND THAT IS NOT A STYLE CHOICE
// ---------------------------------------------------------------------------
// It needs pxgSlangWarpInc, whose emulation path declares `groupshared uint
// gPxgWarpScratch[PXG_WARP_BLOCK]` -- and PXG_WARP_BLOCK is set by the INCLUDING
// file, because it is the block size of the kernel, not of the library.  A Slang
// `import` compiles the imported file as its own module with its own
// preprocessor state, so a module cannot take that define from its importer and
// two modules that both included the library would each carry a scratch array.
// The corpus already settled this: npSdfFind.slang sets PXG_WARP_BLOCK, includes
// the warp library, and then includes pxgSlangSdfFind.slang.  Same shape here.
//
// So an including file must do, in this order:
//
//     #define PXG_WARP_BLOCK 1024
//     #include "pxgSlangWarpInc.slang"
//     #include "pxgSlangTriTriBlock.slang"
//
// ---------------------------------------------------------------------------
// THE TWO SHARED ARRAYS ARE THE KERNEL'S, DECLARED HERE ONCE
// ---------------------------------------------------------------------------
// convexMeshMidphase.cu:1005-1010 declares `countersS[32]` and `shMinSepS[32]`,
// and every helper below is passed one or both of them in the CUDA.  Slang has no
// portable way to pass groupshared by reference, so they are declared here and
// used by name.  An including file must NOT declare its own: the sizes are the
// CUDA's and the measured budget counts them once (npTriTriSharedProbe's gCounters
// and gMinSep).
groupshared uint gTtbCounters[32];
groupshared float gTtbMinSep[32];

// FLT_MAX as the CUDA's <cfloat> spells it.  pxgSlangContactReduceSerial carries
// the same constant as CRS_FLT_MAX, and it is NOT reached from here: that file is
// an `import` module and this one is a textual include, so taking the name would
// force every includer to import a module it may not otherwise need.  One
// constant, two names, both written from the same literal.
static const float TTB_FLT_MAX = 3.402823466e+38f;

// ---------------------------------------------------------------------------
// THE SWEEP: 19 + 6 MUTANTS, 19 KILLED, 3 EQUIVALENT AND PROVED SO
// ---------------------------------------------------------------------------
// The three survivors are all the same shape -- `if (lane == 0u)` moved to lane 1,
// or removed so all 32 lanes write -- and they survive because the value being
// published is LANE-INVARIANT: pxgBallotSync computes the same ballot in every
// lane of the warp, so popc of it is the same number everywhere, and the guard
// reduces 32 identical writes to one rather than selecting a value.
//
// THAT ARGUMENT IS NOT WHY THEY ARE RECORDED AS EQUIVALENT.  The same argument
// would be made by a fixture that simply could not see gTtbCounters at all, so it
// was checked the other way: `count-lane1-publishes-plus-one` and
// `scan-lane1-publishes-plus-one` leave the lane-0 write in place and add a
// SECOND write of `perWarpCount + 1u` from lane 1.  Both are KILLED.  The fixture
// therefore does see the slot, and moving which lane writes it does not change
// what lands there.
//
// The guard is still worth keeping: it is the CUDA's, and it matters the moment
// the warp is not full -- with lane 0 inactive, `if (lane == 0)` writes nothing
// and the slot keeps a stale value, while the unguarded form writes from whoever
// is live.  Both helpers are documented "must be called by full warps", so the
// difference is unreachable here and is not a licence to drop the guard.
//
// The nineteenth mutant, `seed-gate-any-warp`, SURVIVED the first sweep and was
// killed by the second after the fixture was given the input it was missing.  See
// the note on the seed pick below: that survival was a fixture gap wearing the
// costume of an equivalence, and it had a comment above it asserting the
// equivalence as fact.
//
// ---------------------------------------------------------------------------
// threadBlockCountAndBroadcast (reduction.cuh:576-589)
// ---------------------------------------------------------------------------
// NOTE THE ASYMMETRY WITH THE SCAN BELOW: this one reads sharedMem[lane] with NO
// `lane < NbWarps` guard, because it is not a template on NbWarps at all -- it
// assumes the array has 32 entries and the block has 32 warps.  The scan DOES
// guard.  Transcribing the guard into both would be tidier and would change what
// a 16-warp caller reads; this kernel launches 32 warps, and the difference is
// recorded rather than removed.
//
// The __syncthreads() at :584 is INSIDE.  Its caller at convexMeshMidphase.cu:813
// then issues a SECOND one "for countersS reuse" -- that one is the caller's and
// is not folded in here.
uint ttbBlockCountAndBroadcast(uint tid, bool threadContributesElement)
{
    const uint lane = tid & 31u;
    const uint warp = tid >> 5;

    const uint perWarpCount = pxgWarpCountAndBroadcast(PXG_FULL_MASK, tid, threadContributesElement);

    if (lane == 0u)
        gTtbCounters[warp] = perWarpCount;

    GroupMemoryBarrierWithGroupSync();

    // warpReduction<AddOpPxU32>(FULL_MASK, sharedMem[threadIndexInWarp]) -- the
    // butterfly, broadcast to every lane.
    return pxgWarpReduceAddU(PXG_FULL_MASK, tid, gTtbCounters[lane], 5u);
}

// ---------------------------------------------------------------------------
// threadBlockScanExclusive<32>(bool, totalSum, perWarpCountS)
// (reduction.cuh:598-621), the THREE-argument overload -- the one
// convexMeshMidphase.cu:820 and :938 call with countersS.
// ---------------------------------------------------------------------------
// The two-argument overload allocates its own perWarpCountS and is what
// addIndexToBuffer reaches through; that one is a different array and is measured
// separately (npTriTriSharedProbe's gScanPerWarpCount).  Calling one where the
// CUDA calls the other would share a buffer the CUDA keeps apart.
uint ttbBlockScanExclusive(uint tid, bool threadContributesElement, out uint totalSum)
{
    const uint lane = tid & 31u;
    const uint warp = tid >> 5;

    const uint ballotMask = pxgBallotSync(PXG_FULL_MASK, tid, threadContributesElement);
    const uint nbInteresting = pxgPopc(ballotMask);

    if (lane == 0u)
        gTtbCounters[warp] = nbInteresting;

    GroupMemoryBarrierWithGroupSync();

    // The `< NbWarps` guard IS here, with NbWarps = 32, so it never fires on a
    // 32-warp block.  Kept because it is the CUDA's and because its absence in
    // the count helper above is a real difference between the two.
    const uint warpCount = lane < 32u ? gTtbCounters[lane] : 0u;

    const uint total = pxgWarpScanAddInclusiveU(PXG_FULL_MASK, tid, warpCount, 5u);

    const uint carryExclusiveScan = pxgShflSyncU(PXG_FULL_MASK, tid, total, int(warp), 32u)
                                    - gTtbCounters[warp];

    totalSum = pxgShflSyncU(PXG_FULL_MASK, tid, total, 31, 32u);

    return carryExclusiveScan + pxgWarpRankInMask(ballotMask, lane);
}

// ---------------------------------------------------------------------------
// minIndex (nputils.cuh:151-164), in its WARP-SHUFFLE form.
// ---------------------------------------------------------------------------
// pxgSlangTriTriPatch already carries ttpMinIndex, which is the same function
// written over a 32-element ARRAY for the serial reduce.  This is the same
// function over a warp's lanes, and both are needed: the reduce runs on one
// thread and takes an array, while :885 and :899 run on all 1024 threads and take
// a lane value.  They are kept as two spellings of one body rather than one
// spelling plus a staging buffer, because staging would add a barrier inside a
// function the CUDA calls with none.
//
// TWO THINGS THE TRANSCRIPTION MUST NOT TIDY:
//
//   1. The ballot compares `minV == v` against the lane's ORIGINAL v, not the
//      FLT_MAX-substituted one.  An UNMASKED lane whose value happens to equal
//      the masked minimum therefore sets its ballot bit -- and the `& mask` after
//      it is what removes it.  Swapping the two would change nothing; dropping
//      the mask would.
//   2. lowestSetIndex(0) is __ffs(0) - 1 = 0xFFFFFFFF, and an EMPTY mask reaches
//      it: :885 is called with candidateMask, which is 0 in a warp with no
//      candidates.  The caller at :902 then indexes countersS with the result of
//      a SECOND minIndex, never with this one, so the sentinel is never used as
//      an index -- but it is returned and it is load-bearing.
//
// fminf vs min: the CUDA uses fminf, which returns the second argument when the
// first is NaN; Slang's min lowers to GLSL FMin, which is undefined for NaN.  No
// separation reaching here can be NaN -- a lane is either a candidate, whose
// separation passed `sep < contactDistance`, or it is FLT_MAX -- so the two agree
// on every input this kernel produces.  Stated because it is an assumption, not a
// proof about float semantics.
uint ttbWarpMinIndex(uint tid, float v, uint mask, out float minV)
{
    const uint lane = tid & 31u;

    minV = ((mask >> lane) & 1u) != 0u ? v : TTB_FLT_MAX;

    minV = min(minV, pxgShflXorSyncF(PXG_FULL_MASK, tid, minV, 16u, 32u));
    minV = min(minV, pxgShflXorSyncF(PXG_FULL_MASK, tid, minV, 8u, 32u));
    minV = min(minV, pxgShflXorSyncF(PXG_FULL_MASK, tid, minV, 4u, 32u));
    minV = min(minV, pxgShflXorSyncF(PXG_FULL_MASK, tid, minV, 2u, 32u));
    minV = min(minV, pxgShflXorSyncF(PXG_FULL_MASK, tid, minV, 1u, 32u));

    const uint ballot = pxgBallotSync(PXG_FULL_MASK, tid, minV == v) & mask;
    return uint(pxgLowestSetIndex(ballot));
}

// ---------------------------------------------------------------------------
// The cross-warp seed pick, createNewPatches (convexMeshMidphase.cu:879-902).
// ---------------------------------------------------------------------------
// Each warp reduces its own candidates to (minSep, minI) and publishes minSep to
// shMinSepS[warpIndex] and its candidate COUNT to countersS[warpIndex].  After
// bar.sync 3 every warp reads ALL 32 warp minima into its 32 lanes -- note the
// index is `threadIndexInWarp`, so lane L of every warp reads warp L's minimum --
// and runs minIndex over them with FULL_MASK to find the deepest warp.
//
// THE GATE IS `countersS[minIBlock] != 0`, AND IT IS **NOT** THE SAME PREDICATE
// AS "any warp has candidates".  This comment said it was; the claim was written
// down, mutated, and KILLED, and the correction is the useful part.
//
// The argument that looked convincing: a warp with no candidates publishes
// FLT_MAX, because minIndex over an empty mask leaves every lane at FLT_MAX.  So
// the winner of the second minIndex is a warp with real candidates whenever one
// exists, and only an all-empty block reaches a zero count.
//
// WHAT IT MISSES: a warp can have candidates AND publish FLT_MAX, if its
// candidates' separations are FLT_MAX.  Then every warp minimum is FLT_MAX, the
// block minIndex breaks the tie on the LOWEST index, and it can land on an EMPTY
// warp -- zero count, gate false -- while "any warp has candidates" is true.
// Constructed in testNpTriTriBlock.cpp (warp 5 has candidates at FLT_MAX, every
// other warp is empty) and the `seed-gate-any-warp` mutant dies on it.  The first
// version of that fixture drew separations from a finite alphabet, so a warp with
// candidates always published below FLT_MAX and the two spellings agreed on every
// one of its 240 cases; the mutant survived, and surviving is what made the claim
// look proven.
//
// ON THE ENGINE'S INPUTS THE TWO DO COINCIDE, because a candidate's separation
// passed `sep < contactDistance` and is finite.  That is a precondition of the
// CALLER, not a property of this function -- so this function transcribes the
// CUDA's spelling and is correct without it.
//
// `minI` is the seeding LANE inside the winning warp and comes from the FIRST
// minIndex; it is NOT recomputed here.  The caller pairs it with minIBlock at
// :908 -- `warpIndex == minIBlock && threadIndexInWarp == minI` -- which is the
// one place 0xFFFFFFFF from an empty candidateMask can appear, and it cannot
// match a lane index, so a warp with no candidates seeds nothing.
struct TtbSeed
{
    uint minIBlock;      // winning WARP index
    float minSep;        // the block-wide minimum separation
    uint winnerCount;    // countersS[minIBlock]
    bool hasCandidates;  // the :902 gate
};

TtbSeed ttbBlockSeedSelect(uint tid, float laneSep, uint candidateMask, out uint minI)
{
    const uint lane = tid & 31u;
    const uint warp = tid >> 5;

    float warpMinSep;
    minI = ttbWarpMinIndex(tid, laneSep, candidateMask, warpMinSep);   // :885

    if (lane == 0u)                                                    // :887-891
    {
        gTtbMinSep[warp] = warpMinSep;
        gTtbCounters[warp] = pxgPopc(candidateMask);
    }

    GroupMemoryBarrierWithGroupSync();                                 // bar.sync 3 (:894)

    // :897 -- indexed by LANE, so lane L holds warp L's minimum.
    const float blockSep = gTtbMinSep[lane];

    TtbSeed s;
    float chosen;
    s.minIBlock = ttbWarpMinIndex(tid, blockSep, PXG_FULL_MASK, chosen);   // :899
    s.minSep = chosen;
    s.winnerCount = gTtbCounters[s.minIBlock];
    s.hasCandidates = s.winnerCount != 0u;
    return s;
}

// ---------------------------------------------------------------------------
// contactReduce2 at :810 and :931, SERIALISED ONE WARP AT A TIME (14/n)
// ---------------------------------------------------------------------------
// Both call sites are block-uniform and called by ALL 32 warps, each on its own
// 32 lanes, each with its own mask.  In CUDA every warp reduces in parallel and
// gets its answer back in its own lanes' registers.  Here that cannot be done:
//
//   * a 1:1 transcription lowers every shuffle inside contactReduce2 to a
//     WORKGROUP barrier, and 32 warps each running their own reduce is exactly
//     the warp-varying-barrier shape that hangs llvmpipe (P9.83b-2);
//   * staging all 32 warps at once is 32*32*16 B and was MEASURED over the limit
//     at 46684 bytes (P9.83b-4, 3/n).
//
// So warp w stages its 32 lanes into gTtbWarpStage, thread 0 runs the serial
// crsContactReduce2 on them, and the mask comes back to warp w through
// gTtbReduceMask[w].  32 rounds, two barriers each.  That gTtbReduceMask is a
// buffer the CUDA does not have is what made the Workgroup-only budget
// impossible (P9.83b-5, 12/n: 32732 -> 32860 with it).
//
// REQUIREMENT ON THE INCLUDER: crsContactReduce2 lives in the MODULE
// pxgSlangContactReduceSerial, and this file is a textual include, so the
// including file must `import pxgSlangContactReduceSerial;`.
//
// THE INPUT MASK TRAVELS THROUGH THE SAME SLOT AS THE RESULT.  Lane 0 of each warp
// publishes its warp's mask into gTtbReduceMask[warp] before the first round;
// thread 0 reads slot w and overwrites it in round w.  Each slot is written once
// by its warp before any round and once by thread 0 in its own round, so no round
// reads a slot another round has already rewritten.
//
// THE TRAILING BARRIER IS NOT IN THE CUDA AND IS NOT OPTIONAL HERE.  In CUDA the
// result is a register and the next call cannot disturb it.  Here it is shared
// memory: without a barrier after the read, a second call's lane-0 publish of a
// NEW input mask could land before a slower thread of the same warp has read the
// OLD result.  The kernel happens to put other barriers between the two call
// sites, but a function whose correctness depends on what its caller does next
// is a function that breaks when the caller changes.
//
// No dupe test, maxPoints 6, clustering on: <true, true, MESH_MESH_CONTACT_LIMIT,
// false>, as at both call sites.
groupshared float4 gTtbWarpStage[32];
groupshared uint gTtbReduceMask[32];

uint ttbWarpReduceSerial(uint tid, float3 pos, float sep, float3 patchNormal,
                         uint warpMask, float clusterBias)
{
    const uint lane = tid & 31u;
    const uint warp = tid >> 5;

    if (lane == 0u)
        gTtbReduceMask[warp] = warpMask;

    for (uint w = 0u; w < 32u; ++w)
    {
        GroupMemoryBarrierWithGroupSync();
        if (warp == w)
            gTtbWarpStage[lane] = float4(pos, sep);
        GroupMemoryBarrierWithGroupSync();

        if (tid == 0u)
        {
            float3 pts[32];
            float seps[32];
            for (uint l = 0u; l < 32u; ++l)
            {
                const float4 s = gTtbWarpStage[l];
                pts[l] = s.xyz;
                seps[l] = s.w;
            }
            gTtbReduceMask[w] = crsContactReduce2(pts, seps, patchNormal, gTtbReduceMask[w],
                                                  clusterBias, crsConfig(true, true, 6, false));
        }
    }

    GroupMemoryBarrierWithGroupSync();
    const uint result = gTtbReduceMask[warp];
    GroupMemoryBarrierWithGroupSync();   // see THE TRAILING BARRIER above
    return result;
}
