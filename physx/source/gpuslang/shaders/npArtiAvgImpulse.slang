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


// npArtiAvgImpulse.slang
//
// P8.94c: `averageLinkImpulsesAndPropagate` (articulationImpulseResponse.cuh:114)
// — the slab-averaging impulse gather and the leaf→root propagate that
// `stepArticulation1TTGS` runs immediately before `PxcFsFlushVelocity`
// (forwardDynamic2.cu:3020), guarded by `eHAS_IMPULSES`.
//
// Zero new ABI for the THIRD tranche running: `mLinkWithDeferredImpulse`
// (5376), `mRootDeferredZ` (5760), `mScratchImpulse` (11776),
// `mSolverSpatialImpulse` (14080), `mJointOffset` (28544), `mParents` (28672),
// `mDofs` (28800), `mRw_{x,y,z}` and the dof block's `mIsInvDW` (2688),
// `mWorldMotionMatrix` (1152) and `mDeferredQstZ` (896) are all mapped.
//
// ===========================================================================
// THE ALIGNMENT DANCE, AND WHY THE PORT COLLAPSES IT
// ===========================================================================
//
// `impulses` is an array of 24-byte `Cm::UnAlignedSpatialVector`, so every
// other element is 8-byte- rather than 16-byte-aligned.  The CUDA branches on
// `ptr & 0xf` and issues either `float2` then `float4` or `float4` then
// `float2`, then zeroes through the same two pointers.
//
// Read carefully, **both branches touch the same 24 bytes and map them to the
// same six components**: `top.{x,y,z}, bottom.{x,y,z}` are floats 0..5 from
// `ptr` either way, and the two stores together zero exactly `[ptr, ptr+24)`
// either way.  The branch exists to pick a vectorised load WIDTH, not to change
// which bytes are read.  The port therefore reads six floats and writes six
// zeroes, with no branch, and this note is the argument rather than an
// assertion — the fixture places impulse entries on BOTH alignments and
// compares bit for bit, so if the reading above were wrong it would fail.
//
// ===========================================================================
// THE ORACLE
// ===========================================================================
//
// `static __device__`, so this is the P8.73-P8.80 arrangement: the mirror is
// control flow over engine primitives.  The algebra underneath —
// `propagateImpulseW`'s body — is P8.90's `adPropagateImpulseW` except for
// the null-joint-force `stZ = 0.0f - dot` (P9.86: `aiPropagateImpulseWNullJf`
// below spells that `0 - x` as a select); seeding its `inout qstZ` from
// `mDeferredQstZ` and writing the result back is exactly the `+=` the
// engine's `WriteToDeferredQstZ` performs.
//
// ===========================================================================
// THE TRAPS
// ===========================================================================
//
//  1. **`scale` is computed only when the caller passes 0** — which the step
//     kernel does — **and only when `nbSlabs > 1`.**  It counts slabs whose
//     dirty pair is not `(~0, ~0)`, with an **OR**: a slab counts if EITHER
//     half is set.  Then `scale = count > 1 ? 1/count : 1`, so a count of
//     exactly one still gives 1.
//  2. **`maxIndex` is SEEDED from `mLinkWithDeferredImpulse`**, not from zero,
//     and only grows.
//  3. **The dirty pair is reset to `(~0, ~0)` for EVERY slab**, outside the
//     `dirtyIndex != ~0` guard — so a slab with nothing dirty is still reset.
//  4. **`deltaIdx = slabOffset + dirtyIndex * nbArticulations`** where
//     `slabOffset` starts at `articulationId` and advances by
//     `maxLinks * nbArticulations`, while `slabDirtyIndex` advances by
//     `nbArticulations`.  Two different strides over two different arrays.
//  5. **The scale multiplies at the STORE, not at the load** — the impulse is
//     read, the source is zeroed, and only then is `impulse * scale` added to
//     the scratch impulse.
//  6. **The propagate loop runs only when `maxIndex != 0`**, and it runs
//     leaf→root, `linkID` from `maxIndex` down to 1.
//  7. **`dofData` is a BACKWARDS cursor**: it starts at
//     `artiLinks[maxIndex].mJointOffset` and is decremented at the end of each
//     iteration by **`nextDofs`** — the count prefetched from `linkID - 1`, not
//     this link's own.
//  8. **`Z` and the parent's scratch impulse cannot be preloaded** and the CUDA
//     says so in a comment: an earlier iteration may have written either.
//     `mSolverSpatialImpulse` can be and is.
//  9. **The root tail loads `mRootDeferredZ` BEFORE the loop and
//     `artiLinks[0].mScratchImpulse` AFTER it** — the second with its own
//     comment explaining that the loop modifies it.  Then
//     `mRootDeferredZ = rootScratch + rootDeferredZ` and the root's scratch is
//     zeroed.
// 10. **Every link the loop visits has its scratch impulse zeroed**, and the
//     parent's accumulates the propagated value — so the loop both consumes and
//     produces the same field.
//
// ===========================================================================
// THE SWEEP: 39/44, AND ALL FIVE SURVIVORS PROVEN
// ===========================================================================
//
// Three of them are the class P8.94b named — **a guard that protects only
// READS**, which a bit-comparison test cannot see because it measures writes:
//
//  * **A26 — `if (maxIndex != 0)` dropped.**  With `maxIndex == 0` the loop
//    `for (linkID = 0; linkID > 0; --linkID)` never runs, so all the guard
//    saves is the handful of `artiLinks[0]` reads above it.  Unlike P8.94b's
//    pair this one is not even a range guard: index 0 is always valid.  It is
//    kept because the CUDA has it.
//  * **A31 — the prefetch guard `linkID > 1` widened to `linkID > 0`.**  At
//    `linkID == 1` the prefetch reads link 0, in range, and the loop then ends
//    without using the result.
//  * **A18 — the slab reset moved inside the dirty guard.**  A slab whose pair
//    is already `(~0, ~0)` is reset to `(~0, ~0)`: the store is real but the
//    value is unchanged, so the mutation is invisible by construction rather
//    than by any property of the fixture.
//
// The other two are the same arithmetic fact twice, and both were predicted
// while the mutants were being written rather than explained afterwards:
//
//  * **A07 — `count > 1` relaxed to `count > 0`** and **A08 — `nbSlabs > 1`
//    relaxed to `nbSlabs > 0`.**  `1.0f / 1.0f` is exactly `1.0f`, and a count
//    of zero means no slab carried a dirty index, so `scale` is never read.
//    **The `> 1` test is therefore a divide-by-zero guard and nothing else** —
//    it does not change the averaging for any reachable input.  Worth stating
//    because the natural reading of "count > 1" is that a single contributing
//    slab is deliberately left unscaled, and that reading is right about the
//    intent but vacuous about the behaviour.

import pxgSlangCommon;
import pxgSlangArticulationBlocks;
import pxgSlangSpatialMatrix;
import pxgSlangArticulationDynamic;
import npArtiSolveTendons;   // svHighestBit64 (P8.101a)
import pxgSlangArticulationDesc;   // P8.106: the artiPushImpulse desc walk

static const AbSpatialVector AI_ZERO = { float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f) };
static const uint AI_NONE = 0xFFFFFFFFu;

AbSpatialVector aiLoadSV(uint64_t base, uint fieldOffset, uint lane)
{
    float3 top, bottom;
    abLoadSpatialVector(base, fieldOffset, lane, top, bottom);
    return asmSVec(top, bottom);
}

void aiStoreSV(uint64_t base, uint fieldOffset, uint lane, AbSpatialVector v)
{
    abStoreSpatialVector(base, fieldOffset, lane, v.top, v.bottom);
}

// P9.86: propagateImpulseW<WriteToDeferredQstZ> with jointForce == NULL
// (articulationDynamic.cuh:145-146) is stZ = 0.0f - innerProduct: +0 for EITHER
// zero, where adPropagateImpulseW's preloaded `-dot` gives -0 at a +0 dot.  IEEE
// `0 - x` spelled as a select (a literal-zero OpFSub is rewritten to -x by
// llvmpipe, including via adPropagateImpulseWJf's inlined hasJf = false).
// Otherwise adPropagateImpulseW's body, expression for expression.
float aiZeroMinus(float x) { return (x == 0.0f) ? 0.0f : -x; }

AbSpatialVector aiPropagateImpulseWNullJf(AbSpatialVector isInvD[3], float3 childToParent,
                                          AbSpatialVector motionMatrix[3], AbSpatialVector Z,
                                          uint dofCount, inout float qstZ[3])
{
    AbSpatialVector temp = Z;
    for (uint ind = 0u; ind < 3u; ++ind)
    {
        if (ind < dofCount)
        {
            // against Z, NOT against temp
            const float stZ = aiZeroMinus(asmInnerProduct(motionMatrix[ind], Z));
            qstZ[ind] = qstZ[ind] + stZ;
            temp = asmSVecAdd(temp, asmSVecScale(isInvD[ind], stZ));
        }
    }
    return adTranslateSpatialVector(childToParent, temp);
}

// averageLinkImpulsesAndPropagate (articulationImpulseResponse.cuh:114).  The
// CUDA has ONE function with a `scale = 0.0f` default: scale == 0 computes the
// (non-momentum-conserving) slab count internally; a caller-provided scale
// (P8.99's 1/referenceCount) skips that.  The Ex form carries the parameter;
// the old name forwards with 0, entering exactly the CUDA's default branch.
public void aiAverageAndPropagateEx(uint64_t isSlabDirty, uint64_t impulses, uint64_t artiBlock,
                                    uint64_t linkBlocks, uint64_t dofBlocks, uint articulationId,
                                    uint maxLinks, uint nbArticulations, uint nbSlabs,
                                    float scaleIn, uint lane)
{
    const uint slabStepSize = maxLinks * nbArticulations;

    // trap 1: the count is over slabs whose pair is not both-empty, and the
    // reciprocal is taken only above one
    float scale = scaleIn;
    if (scaleIn == 0.0f)
    {
        scale = 1.0f;
        if (nbSlabs > 1u)
        {
            float count = 0.0f;
            for (uint s = 0u, di = articulationId; s < nbSlabs; ++s, di += nbArticulations)
            {
                const uint2 dirty = (Ptr<uint2>(isSlabDirty))[di];
                if (dirty.x != AI_NONE || dirty.y != AI_NONE)
                    count += 1.0f;
            }
            scale = count > 1.0f ? 1.0f / count : 1.0f;
        }
    }

    // trap 2: SEEDED from the block, not from zero
    uint maxIndex = abU32(artiBlock, PXG_AB_mLinkWithDeferredImpulse, lane);

    for (uint s = 0u, slabOffset = articulationId, di = articulationId; s < nbSlabs;
         ++s, slabOffset += slabStepSize, di += nbArticulations)
    {
        const uint2 pair = (Ptr<uint2>(isSlabDirty))[di];
        for (uint i = 0u; i < 2u; ++i)
        {
            const uint dirtyIndex = (i == 0u) ? pair.x : pair.y;
            if (dirtyIndex != AI_NONE)
            {
                const uint64_t dLink = linkBlocks + uint64_t(dirtyIndex) * PXG_AB_LINK_SIZE;
                const AbSpatialVector preloadedScratchImpulse =
                    aiLoadSV(dLink, PXG_AB_LINK_mScratchImpulse, lane);

                // trap 4: the two strides differ
                const uint deltaIdx = slabOffset + dirtyIndex * nbArticulations;

                // See the header: the CUDA's float2/float4 alignment branch
                // touches the same 24 bytes either way, so this is six floats.
                Ptr<float> f = Ptr<float>(impulses + uint64_t(deltaIdx) * 24u);
                const AbSpatialVector impulse = asmSVec(float3(f[0], f[1], f[2]),
                                                        float3(f[3], f[4], f[5]));
                for (uint z = 0u; z < 6u; ++z)
                    f[z] = 0.0f;

                // trap 5: the scale is applied HERE, not at the load
                aiStoreSV(dLink, PXG_AB_LINK_mScratchImpulse, lane,
                          asmSVecAdd(preloadedScratchImpulse, asmSVecScale(impulse, scale)));
                maxIndex = max(dirtyIndex, maxIndex);
            }
        }
        // trap 3: reset for EVERY slab, outside the guard
        (Ptr<uint2>(isSlabDirty))[di] = uint2(AI_NONE, AI_NONE);
    }

    // trap 9: this one is loaded BEFORE the loop
    const AbSpatialVector preloadedRootDeferredZ = aiLoadSV(artiBlock, PXG_AB_mRootDeferredZ, lane);

    if (maxIndex != 0u)   // trap 6
    {
        const uint64_t maxLink = linkBlocks + uint64_t(maxIndex) * PXG_AB_LINK_SIZE;
        // trap 7: a BACKWARDS cursor
        uint64_t dofData = dofBlocks
                         + uint64_t(abU32(maxLink, PXG_AB_LINK_mJointOffset, lane))
                           * PXG_AB_DOF_SIZE;

        uint nextParent = abU32(maxLink, PXG_AB_LINK_mParents, lane);
        uint nextDofs = abU8(maxLink, PXG_AB_LINK_mDofs, lane);
        float nextC2Px = abF32(maxLink, PXG_AB_LINK_mRw_x, lane);
        float nextC2Py = abF32(maxLink, PXG_AB_LINK_mRw_y, lane);
        float nextC2Pz = abF32(maxLink, PXG_AB_LINK_mRw_z, lane);

        for (uint linkID = maxIndex; linkID > 0u; --linkID)
        {
            const uint64_t linkData = linkBlocks + uint64_t(linkID) * PXG_AB_LINK_SIZE;

            // trap 8: NOT preloadable -- an earlier iteration may have written it
            const AbSpatialVector Z = aiLoadSV(linkData, PXG_AB_LINK_mScratchImpulse, lane);
            const AbSpatialVector solverSpatialImpulse =
                aiLoadSV(linkData, PXG_AB_LINK_mSolverSpatialImpulse, lane);

            const uint parent = nextParent;
            const uint64_t pLink = linkBlocks + uint64_t(parent) * PXG_AB_LINK_SIZE;
            const AbSpatialVector parentScratchImpulse =
                aiLoadSV(pLink, PXG_AB_LINK_mScratchImpulse, lane);
            const float c2px = nextC2Px;
            const float c2py = nextC2Py;
            const float c2pz = nextC2Pz;
            const uint dofCount = nextDofs;

            if (linkID > 1u)
            {
                const uint64_t nLink = linkBlocks + uint64_t(linkID - 1u) * PXG_AB_LINK_SIZE;
                nextParent = abU32(nLink, PXG_AB_LINK_mParents, lane);
                nextDofs = abU8(nLink, PXG_AB_LINK_mDofs, lane);
                nextC2Px = abF32(nLink, PXG_AB_LINK_mRw_x, lane);
                nextC2Py = abF32(nLink, PXG_AB_LINK_mRw_y, lane);
                nextC2Pz = abF32(nLink, PXG_AB_LINK_mRw_z, lane);
            }

            // propagateImpulseW<WriteToDeferredQstZ> with a null joint force:
            // P8.90's body, seeded from `mDeferredQstZ` so the write-back is the
            // engine's `+=`.
            AbSpatialVector isInvD[3];
            AbSpatialVector motionMatrix[3];
            float qstZ[3];
            for (uint ind = 0u; ind < 3u; ++ind)
            {
                isInvD[ind] = AI_ZERO;
                motionMatrix[ind] = AI_ZERO;
                qstZ[ind] = 0.0f;
                if (ind < dofCount)
                {
                    const uint64_t d = dofData + uint64_t(ind) * PXG_AB_DOF_SIZE;
                    isInvD[ind] = aiLoadSV(d, PXG_AB_DOF_mIsInvDW, lane);
                    motionMatrix[ind] = aiLoadSV(d, PXG_AB_DOF_mWorldMotionMatrix, lane);
                    qstZ[ind] = abF32(d, PXG_AB_DOF_mDeferredQstZ, lane);
                }
            }

            const AbSpatialVector propagatedZ =   // P9.86: the CUDA's `0.0f - dot` (articulationImpulseResponse.cuh:236)
                aiPropagateImpulseWNullJf(isInvD, float3(c2px, c2py, c2pz), motionMatrix, Z,
                                          dofCount, qstZ);

            for (uint ind = 0u; ind < 3u; ++ind)
                if (ind < dofCount)
                    abStoreF32(dofData + uint64_t(ind) * PXG_AB_DOF_SIZE,
                               PXG_AB_DOF_mDeferredQstZ, lane, qstZ[ind]);

            aiStoreSV(linkData, PXG_AB_LINK_mSolverSpatialImpulse, lane,
                      asmSVecAdd(solverSpatialImpulse, Z));
            // trap 10: the parent accumulates, this link is zeroed
            aiStoreSV(pLink, PXG_AB_LINK_mScratchImpulse, lane,
                      asmSVecAdd(parentScratchImpulse, propagatedZ));
            aiStoreSV(linkData, PXG_AB_LINK_mScratchImpulse, lane, AI_ZERO);

            // trap 7 again: decremented by the PREFETCHED count
            dofData -= uint64_t(nextDofs) * PXG_AB_DOF_SIZE;
        }
    }

    // trap 9: and this one AFTER, because the loop writes it
    const AbSpatialVector preloadedRootScratchImpulse =
        aiLoadSV(linkBlocks, PXG_AB_LINK_mScratchImpulse, lane);
    aiStoreSV(linkBlocks, PXG_AB_LINK_mScratchImpulse, lane, AI_ZERO);
    aiStoreSV(artiBlock, PXG_AB_mRootDeferredZ, lane,
              asmSVecAdd(preloadedRootScratchImpulse, preloadedRootDeferredZ));
}

// ---------------------------------------------------------------------------
// The entry point the fixture drives.
// ---------------------------------------------------------------------------
struct AiParams
{
    uint64_t isSlabDirty;   // @0
    uint64_t impulses;      // @8
    uint64_t artiBlock;     // @16
    uint64_t linkBlocks;    // @24
    uint64_t dofBlocks;     // @32
    uint maxLinks;          // @40
    uint nbArticulations;   // @44
    uint nbSlabs;           // @48
    uint nbLanes;           // @52 -> 56
};

[[vk::push_constant]]
ConstantBuffer<AiParams> gAi;

[shader("compute")]
[numthreads(32, 1, 1)]
void artiAvgImpulse(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint lane = dispatchThreadId.x;
    if (lane >= gAi.nbLanes)
        return;

    // one articulation per lane, and the articulation id IS the lane, so the
    // slab and impulse strides are exercised with a non-zero base
    aiAverageAndPropagate(gAi.isSlabDirty, gAi.impulses, gAi.artiBlock, gAi.linkBlocks,
                          gAi.dofBlocks, lane, gAi.maxLinks, gAi.nbArticulations, gAi.nbSlabs,
                          lane);
}

public void aiAverageAndPropagate(uint64_t isSlabDirty, uint64_t impulses, uint64_t artiBlock,
                                  uint64_t linkBlocks, uint64_t dofBlocks, uint articulationId,
                                  uint maxLinks, uint nbArticulations, uint nbSlabs, uint lane)
{
    aiAverageAndPropagateEx(isSlabDirty, impulses, artiBlock, linkBlocks, dofBlocks,
                            articulationId, maxLinks, nbArticulations, nbSlabs, 0.0f, lane);
}

// ===========================================================================
// P8.101a: averageLinkImpulsesAndPropagate2
// (articulationImpulseResponse.cuh:263-385) — the PARTIAL propagate the
// per-partition impulse kernel artiPropagateImpulses2 runs: instead of
// walking all the way to the root, it propagates the scratch impulses down
// the precomputed per-partition dirty path until it reaches the COMMON NODE
// (mImpulseHoldingLink), caches the common node's deltaV in
// mCommonLinkDeltaVelocity, and leaves the common node's scratch impulse
// UNPROPAGATED for the eventual full walk.
//
// Traps, each a sweep mutant unless marked:
//  1. The slab-average phase runs ONLY under dirtyFlags & eHAS_IMPULSES; the
//     walk and the tail run REGARDLESS.
//  2. The slab phase is the Ex phase minus the maxIndex tracking and minus
//     the internal-count branch — scale is always the caller's.
//  3. The walk goes word HIGHEST-first (j = wordSize-1 down) and bit
//     HIGHEST-first within a word (deepest link first).
//  4. **`break` at the common node exits only the CURRENT word's loop** —
//     lower words keep processing (the engine's own spelling; with the
//     common node in word 1, path bits in word 0 — its ancestors — are
//     still propagated).  Transcribed as-is; the fixture pins it.
//  5. The bit clear is `word &= ~(uint64_t(1) << bitIndex)` — the 64-bit
//     LHS expression, never the 32-bit `1uL` literal (the P8.99b lesson).
//  6. mSolverSpatialImpulse books += Z with NO −ZInternalConstraint (unlike
//     the DFS backprop booking).
//  7. The common node's scratch impulse is NOT zeroed (a load-bearing
//     engine comment: it has not been propagated to the root yet) — the
//     tail READS it for the deltaV but leaves it in place.
//  8. The tail stores mCommonLinkDeltaVelocity = response6x6 * (−scratch)
//     and mLinkWithDeferredImpulse = commonNode, both UNCONDITIONALLY.
// ===========================================================================

public void aiAverageAndPropagate2(uint64_t isSlabDirty, uint64_t impulses, uint64_t artiBlock,
                                   uint64_t linkBlocks, uint64_t dofBlocks, uint articulationId,
                                   uint maxLinks, uint nbArticulations, uint nbSlabs,
                                   float scale, uint64_t pathToRootPerPartition, uint wordSize,
                                   uint commonNode, uint dirtyFlags, uint lane)
{
    // trap 1: only the averaging is gated
    if ((dirtyFlags & 2u) != 0u)   // PxgArtiStateDirtyFlag::eHAS_IMPULSES
    {
        const uint slabStepSize = maxLinks * nbArticulations;
        for (uint s = 0u, slabOffset = articulationId, di = articulationId; s < nbSlabs;
             ++s, slabOffset += slabStepSize, di += nbArticulations)
        {
            const uint2 pair = (Ptr<uint2>(isSlabDirty))[di];
            for (uint i = 0u; i < 2u; ++i)
            {
                const uint dirtyIndex = (i == 0u) ? pair.x : pair.y;
                if (dirtyIndex != AI_NONE)
                {
                    const uint64_t dLink = linkBlocks + uint64_t(dirtyIndex) * PXG_AB_LINK_SIZE;
                    const AbSpatialVector preloadedScratchImpulse =
                        aiLoadSV(dLink, PXG_AB_LINK_mScratchImpulse, lane);

                    const uint deltaIdx = slabOffset + dirtyIndex * nbArticulations;

                    // the alignment dance collapses to six floats (see header)
                    Ptr<float> f = Ptr<float>(impulses + uint64_t(deltaIdx) * 24u);
                    const AbSpatialVector impulse = asmSVec(float3(f[0], f[1], f[2]),
                                                            float3(f[3], f[4], f[5]));
                    for (uint z = 0u; z < 6u; ++z)
                        f[z] = 0.0f;

                    aiStoreSV(dLink, PXG_AB_LINK_mScratchImpulse, lane,
                              asmSVecAdd(preloadedScratchImpulse, asmSVecScale(impulse, scale)));
                }
            }
            // reset for EVERY slab, outside the guard
            (Ptr<uint2>(isSlabDirty))[di] = uint2(AI_NONE, AI_NONE);
        }
    }

    // trap 3: words from the top, bits highest-first
    for (int j = int(wordSize) - 1; j >= 0; --j)
    {
        const uint bitOffset = uint(j) * 64u;
        uint64_t word = (Ptr<uint64_t>(pathToRootPerPartition
                                       + uint64_t(j) * PXG_AB_BITFIELD_STACK_SIZE))[lane];
        if (word != 0uL)
        {
            while (word != 0uL)
            {
                const uint bitIndex = svHighestBit64(word);
                const uint index = bitIndex + bitOffset;

                if (index == commonNode)
                    break;   // trap 4: only THIS word's loop ends

                word &= ~(uint64_t(1) << bitIndex);   // trap 5

                const uint64_t linkData = linkBlocks + uint64_t(index) * PXG_AB_LINK_SIZE;
                const uint jointOffset = abU32(linkData, PXG_AB_LINK_mJointOffset, lane);
                const uint parent = abU32(linkData, PXG_AB_LINK_mParents, lane);
                const float c2px = abF32(linkData, PXG_AB_LINK_mRw_x, lane);
                const float c2py = abF32(linkData, PXG_AB_LINK_mRw_y, lane);
                const float c2pz = abF32(linkData, PXG_AB_LINK_mRw_z, lane);
                const uint dofCount = abU8(linkData, PXG_AB_LINK_mDofs, lane);

                const AbSpatialVector Z = aiLoadSV(linkData, PXG_AB_LINK_mScratchImpulse, lane);
                const uint64_t pLink = linkBlocks + uint64_t(parent) * PXG_AB_LINK_SIZE;
                const AbSpatialVector parentScratchImpulse =
                    aiLoadSV(pLink, PXG_AB_LINK_mScratchImpulse, lane);
                const AbSpatialVector solverSpatialImpulse =
                    aiLoadSV(linkData, PXG_AB_LINK_mSolverSpatialImpulse, lane);

                // propagateImpulseW<WriteToDeferredQstZ>: P8.90's body seeded
                // from mDeferredQstZ so the write-back is the engine's +=
                const uint64_t dofData = dofBlocks + uint64_t(jointOffset) * PXG_AB_DOF_SIZE;
                AbSpatialVector isInvD[3];
                AbSpatialVector motionMatrix[3];
                float qstZ[3];
                for (uint ind = 0u; ind < 3u; ++ind)
                {
                    isInvD[ind] = AI_ZERO;
                    motionMatrix[ind] = AI_ZERO;
                    qstZ[ind] = 0.0f;
                    if (ind < dofCount)
                    {
                        const uint64_t d = dofData + uint64_t(ind) * PXG_AB_DOF_SIZE;
                        isInvD[ind] = aiLoadSV(d, PXG_AB_DOF_mIsInvDW, lane);
                        motionMatrix[ind] = aiLoadSV(d, PXG_AB_DOF_mWorldMotionMatrix, lane);
                        qstZ[ind] = abF32(d, PXG_AB_DOF_mDeferredQstZ, lane);
                    }
                }

                const AbSpatialVector propagatedZ =   // P9.86: the CUDA's `0.0f - dot` (articulationImpulseResponse.cuh:359)
                    aiPropagateImpulseWNullJf(isInvD, float3(c2px, c2py, c2pz), motionMatrix, Z,
                                              dofCount, qstZ);

                for (uint ind = 0u; ind < 3u; ++ind)
                    if (ind < dofCount)
                        abStoreF32(dofData + uint64_t(ind) * PXG_AB_DOF_SIZE,
                                   PXG_AB_DOF_mDeferredQstZ, lane, qstZ[ind]);

                // trap 6: NO internal-constraint share subtracted here
                aiStoreSV(linkData, PXG_AB_LINK_mSolverSpatialImpulse, lane,
                          asmSVecAdd(solverSpatialImpulse, Z));

                aiStoreSV(pLink, PXG_AB_LINK_mScratchImpulse, lane,
                          asmSVecAdd(parentScratchImpulse, propagatedZ));
                aiStoreSV(linkData, PXG_AB_LINK_mScratchImpulse, lane, AI_ZERO);
            }
        }
    }

    // trap 7/8: the tail — read the common node's scratch, do NOT zero it
    const uint64_t cLink = linkBlocks + uint64_t(commonNode) * PXG_AB_LINK_SIZE;
    const AbMatrix6x6 mat = abLoadSpatialMatrix6x6(cLink, PXG_AB_LINK_mSpatialResponseMatrix, lane);
    const AbSpatialVector cScratch = aiLoadSV(cLink, PXG_AB_LINK_mScratchImpulse, lane);
    const AbSpatialVector deltaV = asmM6MulVec(mat, asmSVec(-cScratch.top, -cScratch.bottom));

    aiStoreSV(artiBlock, PXG_AB_mCommonLinkDeltaVelocity, lane, deltaV);
    abStoreU32(artiBlock, PXG_AB_mLinkWithDeferredImpulse, lane, commonNode);
}

// ---------------------------------------------------------------------------
// test entry for the partial propagate
// ---------------------------------------------------------------------------
struct Ai2Params
{
    uint64_t isSlabDirty;    // @0
    uint64_t impulses;       // @8
    uint64_t artiBlock;      // @16
    uint64_t linkBlocks;     // @24
    uint64_t dofBlocks;      // @32
    uint64_t pathToRoot;     // @40  per-lane words: base + j*256 + lane*8
    uint64_t io;             // @48  per-lane: scale @0, commonNode @4, dirtyFlags @8 (12 floats stride)
    uint maxLinks;           // @56
    uint nbArticulations;    // @60
    uint nbSlabs;            // @64
    uint wordSize;           // @68
    uint nbLanes;            // @72
    uint pad0;               // @76 -> 80
};

[[vk::push_constant]]
ConstantBuffer<Ai2Params> gAi2;

[shader("compute")]
[numthreads(32, 1, 1)]
void aiAvg2Test(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint lane = dispatchThreadId.x;
    if (lane >= gAi2.nbLanes)
        return;

    Ptr<uint> pio = Ptr<uint>(gAi2.io + uint64_t(lane) * 12u);
    const float scale = asfloat(pio[0]);
    const uint commonNode = pio[1];
    const uint dirtyFlags = pio[2];

    aiAverageAndPropagate2(gAi2.isSlabDirty, gAi2.impulses, gAi2.artiBlock, gAi2.linkBlocks,
                           gAi2.dofBlocks, lane, gAi2.maxLinks, gAi2.nbArticulations,
                           gAi2.nbSlabs, scale, gAi2.pathToRoot, gAi2.wordSize,
                           commonNode, dirtyFlags, lane);
}

// ===========================================================================
// P8.106: artiPushImpulse (forwardDynamic2.cu:4139-4209).  Grid ceil(nb/32),
// block (32,1,1) -- one warp per block, w = groupId.x (the P8.105 shape).
// NOTE: PxgArticulationCore::pushImpulse has NO caller in PhysX 5.6.1 -- the
// kernel is dead host code that the wrangler still resolves eagerly; mapping
// it removes it from the refused list, and the fixture + sweep are what
// verify it (no scene can dispatch it).
//
// TRAPS, each a sweep mutant:
//  1. Gated on mStateDirty != 0, and the flag is NOT written -- unlike
//     artiOutputVelocity, which clears it.
//  2. The per-link walk indexes dofs via the link's mJointOffset field --
//     NOT the running cursor aiAverageAndPropagateEx uses.
//  3. Leaf -> root (index = nbLinks-1 down to 1): a child adds into its
//     parent's mScratchImpulse BEFORE the parent is processed; ascending
//     order loses every grandchild contribution.
//  4. The propagate is the TEMPLATE propagateImpulseW<WriteToDeferredQstZ>
//     with jointForce == NULL, i.e. the `0.0f - dot` spelling
//     (adPropagateImpulseWJf with hasJf = false), not the preloaded `-dot`.
//  5. Each link's mScratchImpulse is zeroed AFTER propagating; the parent
//     target comes from mParents, not index - 1.
//  6. The root tail runs only for a FREE base (eFIX_BASE is bit 0 with
//     decoys above it): deltaV = response6x6 * (-scratch0), STORED to
//     mRootDeferredZ (an overwrite, NOT an add), scratch0 zeroed.
//     The engine stores a VELOCITY-dimension quantity into a field every
//     reader treats as an impulse (PxcFsFlushVelocity computes
//     invInertia * (-mRootDeferredZ)) -- transcribed verbatim; flagged in
//     RUNTIME_STATUS as the sixth upstream-bug candidate.
// ===========================================================================

struct PiParams
{
    uint64_t desc;   // @0  PxgArticulationCoreDesc*
}

[[vk::push_constant]]
ConstantBuffer<PiParams> gPi;

[shader("compute")]
[numthreads(32, 1, 1)]
void artiPushImpulse(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint64_t d = gPi.desc;
    const uint lane = groupThreadId.x;
    const uint globalWarpIndex = groupId.x;
    const uint globalThreadIndex = globalWarpIndex * 32u + lane;

    const uint nbArticulations = artiU32Field(d, PXG_ARTI_CORE_NB_ARTICULATIONS);
    if (globalThreadIndex >= nbArticulations)
        return;

    const uint64_t artiBlock = artiPtrField(d, PXG_ARTI_CORE_ARTICULATION_BLOCKS)
                             + uint64_t(globalWarpIndex) * PXG_AB_BLOCK_SIZE;

    const uint dirtyFlag = abU8(artiBlock, PXG_AB_mStateDirty, lane);   // trap 1
    if (dirtyFlag == 0u)
        return;

    const uint maxLinks = artiU32Field(d, PXG_ARTI_CORE_MAX_LINKS);
    const uint maxDofs = artiU32Field(d, PXG_ARTI_CORE_MAX_DOFS);
    const uint64_t linkBase = artiPtrField(d, PXG_ARTI_CORE_LINK_BLOCKS)
                            + uint64_t(globalWarpIndex) * uint64_t(maxLinks) * PXG_AB_LINK_SIZE;
    const uint64_t dofBase = artiPtrField(d, PXG_ARTI_CORE_DOF_BLOCKS)
                           + uint64_t(globalWarpIndex) * uint64_t(maxDofs) * PXG_AB_DOF_SIZE;

    const uint nbLinks = abU32(artiBlock, PXG_AB_mNumLinks, lane);

    for (uint index = nbLinks - 1u; index > 0u; --index)   // trap 3
    {
        const uint64_t tLink = linkBase + uint64_t(index) * PXG_AB_LINK_SIZE;
        // trap 2: the dof window comes from the link's own field
        const uint jointOffset = abU32(tLink, PXG_AB_LINK_mJointOffset, lane);
        const uint64_t dofData = dofBase + uint64_t(jointOffset) * PXG_AB_DOF_SIZE;

        const uint parent = abU32(tLink, PXG_AB_LINK_mParents, lane);
        const float3 rw = float3(abF32(tLink, PXG_AB_LINK_mRw_x, lane),
                                 abF32(tLink, PXG_AB_LINK_mRw_y, lane),
                                 abF32(tLink, PXG_AB_LINK_mRw_z, lane));
        const uint dofCount = abU8(tLink, PXG_AB_LINK_mDofs, lane);

        const AbSpatialVector Z = aiLoadSV(tLink, PXG_AB_LINK_mScratchImpulse, lane);

        AbSpatialVector isInvD[3];
        AbSpatialVector motionMatrix[3];
        float qstZ[3];
        float jfNone[3];
        for (uint ind = 0u; ind < 3u; ++ind)
        {
            isInvD[ind] = AI_ZERO;
            motionMatrix[ind] = AI_ZERO;
            qstZ[ind] = 0.0f;
            jfNone[ind] = 0.0f;
            if (ind < dofCount)
            {
                const uint64_t dd = dofData + uint64_t(ind) * PXG_AB_DOF_SIZE;
                isInvD[ind] = aiLoadSV(dd, PXG_AB_DOF_mIsInvDW, lane);
                motionMatrix[ind] = aiLoadSV(dd, PXG_AB_DOF_mWorldMotionMatrix, lane);
                qstZ[ind] = abF32(dd, PXG_AB_DOF_mDeferredQstZ, lane);
            }
        }

        // trap 4: the NULL-jointForce template spelling
        const AbSpatialVector propagatedZ =
            adPropagateImpulseWJf(isInvD, rw, motionMatrix, Z, dofCount, qstZ, jfNone, 1.0f,
                                  false);

        for (uint ind = 0u; ind < 3u; ++ind)
            if (ind < dofCount)
                abStoreF32(dofData + uint64_t(ind) * PXG_AB_DOF_SIZE, PXG_AB_DOF_mDeferredQstZ,
                           lane, qstZ[ind]);

        // trap 5
        const uint64_t pLink = linkBase + uint64_t(parent) * PXG_AB_LINK_SIZE;
        aiStoreSV(pLink, PXG_AB_LINK_mScratchImpulse, lane,
                  asmSVecAdd(aiLoadSV(pLink, PXG_AB_LINK_mScratchImpulse, lane), propagatedZ));
        aiStoreSV(tLink, PXG_AB_LINK_mScratchImpulse, lane, AI_ZERO);
    }

    const bool fixedBase = (abU32(artiBlock, PXG_AB_mFlags, lane) & 1u) != 0u;   // trap 6
    if (!fixedBase)
    {
        const AbMatrix6x6 mat =
            abLoadSpatialMatrix6x6(linkBase, PXG_AB_LINK_mSpatialResponseMatrix, lane);
        const AbSpatialVector scratch0 = aiLoadSV(linkBase, PXG_AB_LINK_mScratchImpulse, lane);
        const AbSpatialVector deltaV =
            asmM6MulVec(mat, asmSVec(-scratch0.top, -scratch0.bottom));

        aiStoreSV(artiBlock, PXG_AB_mRootDeferredZ, lane, deltaV);   // a STORE, not an add
        aiStoreSV(linkBase, PXG_AB_LINK_mScratchImpulse, lane, AI_ZERO);
    }
}

// ===========================================================================
// P8.108: artiApplyTgsSubstepForces (forwardDynamic2.cu:3482-3557).  Grid
// ceil(nb/64), block (64,1,1) -- ONE THREAD per articulation, TWO warps per
// block: lane = tid & 31, warp = globalThreadIndex / 32 (not groupId.x).
// Launched once per TGS position iteration (PxgTGSCudaSolverCore.cpp:1377)
// when the scene enables eENABLE_EXTERNAL_FORCES_EVERY_ITERATION_TGS.
//
// Turns the per-substep isolated external forces (the RECORD's zAForces,
// filled by computeSpatialInertiaW when that flag is set) plus the user
// joint forces into deferred impulses.
//
// TRAPS, each a sweep mutant:
//  1. articulatedYW = zExt[link] * stepDt + mScratchImpulse -- the scratch
//     may be NON-ZERO on entry (multi-child accumulation): add, never
//     overwrite; and the scratch is re-zeroed after propagating.
//  2. The propagate is propagateImpulseW<WriteToDeferredQstZ> WITH a joint
//     force: jf = jointForce[ind] * stepDt happens PER DOF inside the
//     propagate (adPropagateImpulseWJf, hasJf = true, multiplier = stepDt),
//     and the slice &jointForces[jointOffset] is indexed by the BLOCK's
//     mJointOffset while the array itself lives in the RECORD.
//  3. The parent add is GATED: (parent > 0 || !isFixedBase) -- a child of a
//     FIXED root drops its propagated impulse (the root cannot move), but
//     its own scratch is still zeroed.
//  4. The free-base root tail ADDS zExt[0]*stepDt + scratch0 into
//     mRootDeferredZ (an add, unlike artiPushImpulse's store) and zeroes
//     scratch0; a FIXED base skips the tail entirely, so the root's scratch
//     SURVIVES there.
//  5. mStateDirty is ASSIGNED eVEL_DIRTY (1) -- not OR-ed -- clobbering any
//     eHAS_IMPULSES bit, unconditionally for every live thread.
// ===========================================================================

struct AtsParams
{
    uint64_t desc;   // @0  PxgArticulationCoreDesc*
    float stepDt;    // @8
    uint pad0;       // @12 -> 16
};

[[vk::push_constant]]
ConstantBuffer<AtsParams> gAts;

[shader("compute")]
[numthreads(64, 1, 1)]
void artiApplyTgsSubstepForces(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint64_t d = gAts.desc;
    const float stepDt = gAts.stepDt;
    const uint globalThreadIndex = dispatchThreadId.x;
    const uint lane = globalThreadIndex & 31u;
    const uint globalWarpIndex = globalThreadIndex / 32u;

    const uint nbArticulations = artiU32Field(d, PXG_ARTI_CORE_NB_ARTICULATIONS);
    const uint maxLinks = artiU32Field(d, PXG_ARTI_CORE_MAX_LINKS);
    const uint maxDofs = artiU32Field(d, PXG_ARTI_CORE_MAX_DOFS);

    if (globalThreadIndex >= nbArticulations)
        return;

    const uint64_t record = artiPtrField(d, PXG_ARTI_CORE_ARTICULATIONS)
                          + uint64_t(globalThreadIndex) * PXG_ARTI_STRIDE;
    const uint64_t zExt = artiPtrField(record, PXG_ARTI_ZA_FORCES);
    const uint64_t jointForces = artiPtrField(record, PXG_ARTI_JOINT_FORCE);

    const uint64_t artiBlock = artiPtrField(d, PXG_ARTI_CORE_ARTICULATION_BLOCKS)
                             + uint64_t(globalWarpIndex) * PXG_AB_BLOCK_SIZE;
    const uint64_t linkBase = artiPtrField(d, PXG_ARTI_CORE_LINK_BLOCKS)
                            + uint64_t(globalWarpIndex) * uint64_t(maxLinks) * PXG_AB_LINK_SIZE;
    const uint64_t dofBase = artiPtrField(d, PXG_ARTI_CORE_DOF_BLOCKS)
                           + uint64_t(globalWarpIndex) * uint64_t(maxDofs) * PXG_AB_DOF_SIZE;

    const uint numLinks = abU32(artiBlock, PXG_AB_mNumLinks, lane);
    const bool isFixedBase = (abU32(artiBlock, PXG_AB_mFlags, lane) & 1u) != 0u;

    for (uint linkIdx = numLinks - 1u; linkIdx > 0u; --linkIdx)
    {
        const uint64_t tLink = linkBase + uint64_t(linkIdx) * PXG_AB_LINK_SIZE;

        const uint parent = abU32(tLink, PXG_AB_LINK_mParents, lane);
        const float3 rw = float3(abF32(tLink, PXG_AB_LINK_mRw_x, lane),
                                 abF32(tLink, PXG_AB_LINK_mRw_y, lane),
                                 abF32(tLink, PXG_AB_LINK_mRw_z, lane));
        const uint dofCount = abU8(tLink, PXG_AB_LINK_mDofs, lane);
        const uint jointOffset = abU32(tLink, PXG_AB_LINK_mJointOffset, lane);
        const uint64_t dofData = dofBase + uint64_t(jointOffset) * PXG_AB_DOF_SIZE;

        // trap 1: isolated * stepDt PLUS whatever the scratch accumulated
        Ptr<float> z = Ptr<float>(zExt + uint64_t(linkIdx) * PXG_ARTI_SPATIAL_VECTOR_SIZE);
        const AbSpatialVector isolatedYW =
            asmSVec(float3(z[0], z[1], z[2]), float3(z[3], z[4], z[5]));
        const AbSpatialVector articulatedYW =
            asmSVecAdd(asmSVecScale(isolatedYW, stepDt),
                       aiLoadSV(tLink, PXG_AB_LINK_mScratchImpulse, lane));

        AbSpatialVector isInvD[3];
        AbSpatialVector motionMatrix[3];
        float qstZ[3];
        float jf[3];
        Ptr<float> jfSlice = Ptr<float>(jointForces + uint64_t(jointOffset) * 4u);
        for (uint ind = 0u; ind < 3u; ++ind)
        {
            isInvD[ind] = AI_ZERO;
            motionMatrix[ind] = AI_ZERO;
            qstZ[ind] = 0.0f;
            jf[ind] = 0.0f;
            if (ind < dofCount)
            {
                const uint64_t dd = dofData + uint64_t(ind) * PXG_AB_DOF_SIZE;
                isInvD[ind] = aiLoadSV(dd, PXG_AB_DOF_mIsInvDW, lane);
                motionMatrix[ind] = aiLoadSV(dd, PXG_AB_DOF_mWorldMotionMatrix, lane);
                qstZ[ind] = abF32(dd, PXG_AB_DOF_mDeferredQstZ, lane);
                // trap 2: the joint force is read here, multiplied inside the
                // propagate by jfMult = stepDt
                jf[ind] = jfSlice[ind];
            }
        }

        const AbSpatialVector propagatedYWParent =
            adPropagateImpulseWJf(isInvD, rw, motionMatrix, articulatedYW, dofCount, qstZ, jf,
                                  stepDt, true);

        for (uint ind = 0u; ind < 3u; ++ind)
            if (ind < dofCount)
                abStoreF32(dofData + uint64_t(ind) * PXG_AB_DOF_SIZE, PXG_AB_DOF_mDeferredQstZ,
                           lane, qstZ[ind]);

        // trap 3
        if (parent > 0u || !isFixedBase)
        {
            const uint64_t pLink = linkBase + uint64_t(parent) * PXG_AB_LINK_SIZE;
            aiStoreSV(pLink, PXG_AB_LINK_mScratchImpulse, lane,
                      asmSVecAdd(aiLoadSV(pLink, PXG_AB_LINK_mScratchImpulse, lane),
                                 propagatedYWParent));
        }
        aiStoreSV(tLink, PXG_AB_LINK_mScratchImpulse, lane, AI_ZERO);
    }

    // trap 4
    if (!isFixedBase)
    {
        Ptr<float> z = Ptr<float>(zExt);
        const AbSpatialVector isolatedYW =
            asmSVec(float3(z[0], z[1], z[2]), float3(z[3], z[4], z[5]));
        const AbSpatialVector articulatedYW =
            asmSVecAdd(asmSVecScale(isolatedYW, stepDt),
                       aiLoadSV(linkBase, PXG_AB_LINK_mScratchImpulse, lane));

        aiStoreSV(artiBlock, PXG_AB_mRootDeferredZ, lane,
                  asmSVecAdd(aiLoadSV(artiBlock, PXG_AB_mRootDeferredZ, lane), articulatedYW));
        aiStoreSV(linkBase, PXG_AB_LINK_mScratchImpulse, lane, AI_ZERO);
    }

    abStoreU8(artiBlock, PXG_AB_mStateDirty, lane, 1u);   // trap 5: eVEL_DIRTY ASSIGNED
}
