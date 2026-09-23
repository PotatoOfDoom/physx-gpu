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
//   cross-warp seed pick at convexMeshMidphase.cu:886-899.
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
// thread and takes an array, while :894 and :899 run on all 1024 threads and take
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
//      it: :894 is called with candidateMask, which is 0 in a warp with no
//      candidates.  The caller at :899 then indexes countersS with the result of
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
// The cross-warp seed pick, createNewPatches (convexMeshMidphase.cu:879-901).
// ---------------------------------------------------------------------------
// Each warp reduces its own candidates to (minSep, minI) and publishes minSep to
// shMinSepS[warpIndex] and its candidate COUNT to countersS[warpIndex].  After
// bar.sync 3 every warp reads ALL 32 warp minima into its 32 lanes -- note the
// index is `threadIndexInWarp`, so lane L of every warp reads warp L's minimum --
// and runs minIndex over them with FULL_MASK to find the deepest warp.
//
// THE GATE IS `countersS[minIBlock] != 0`, NOT "any warp has candidates".  They
// are the same predicate and the CUDA does not spell it the short way: a warp
// with no candidates published FLT_MAX (minIndex over an empty mask leaves every
// lane at FLT_MAX), so the winner of the second minIndex is a warp with real
// candidates whenever one exists.  Only when EVERY warp published FLT_MAX does
// the winner have a zero count -- and then the tie-break decides which index, and
// the count is zero whichever it picks.  Transcribed as written, because the
// equality is an argument and the spelling is the engine.
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
    bool hasCandidates;  // the :901 gate
};

TtbSeed ttbBlockSeedSelect(uint tid, float laneSep, uint candidateMask, out uint minI)
{
    const uint lane = tid & 31u;
    const uint warp = tid >> 5;

    float warpMinSep;
    minI = ttbWarpMinIndex(tid, laneSep, candidateMask, warpMinSep);   // :894

    if (lane == 0u)                                                    // :896-900
    {
        gTtbMinSep[warp] = warpMinSep;
        gTtbCounters[warp] = pxgPopc(candidateMask);
    }

    GroupMemoryBarrierWithGroupSync();                                 // bar.sync 3 (:903)

    // :906 -- indexed by LANE, so lane L holds warp L's minimum.
    const float blockSep = gTtbMinSep[lane];

    TtbSeed s;
    float chosen;
    s.minIBlock = ttbWarpMinIndex(tid, blockSep, PXG_FULL_MASK, chosen);   // :908
    s.minSep = chosen;
    s.winnerCount = gTtbCounters[s.minIBlock];
    s.hasCandidates = s.winnerCount != 0u;
    return s;
}
