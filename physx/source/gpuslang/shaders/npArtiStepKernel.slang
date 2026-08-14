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


// npArtiStepKernel.slang
//
// P8.94d: `stepArticulation1TTGS` (forwardDynamic2.cu:3873) — the TGS
// sub-step integrator, launched once per position iteration from
// PxgTGSCudaSolverCore::solveContactMultiBlockParallel with
// stepDt = dt / nbPositionIterations.  Manifest 276 -> 277.
//
// THIS MODULE IS COMPOSITION.  Every body it runs was landed, fixtured and
// swept in its own tranche: `aiAverageAndPropagate` (P8.94c),
// `fvFlushVelocity` (P8.94b — the SIX-argument overload; the eight-argument
// one at forwardDynamic2.cu:3990 writes the solver's outVelocity buffer and
// is NOT this call), `asUpdateRootBody` and `asPropagateLink<isSubstep=true>`
// (P8.94a).  This file adds the per-link deltaQ / mDeltaMotion /
// mPosMotionVelocity bookkeeping the kernel wraps around them — that part is
// new and gets its own mutants.
//
// It is also the first module to IMPORT other np* modules rather than
// pxgSlang* layers only.  Measured before relying on it: slangc prunes the
// imported modules' [[vk::push_constant]] globals per entry point, so the
// compiled SPIR-V carries exactly ONE push-constant block (this one's).
//
// ===========================================================================
// THE TRAPS
// ===========================================================================
//
//  1. **`stateDirty` is a MASK, not a bool.**  eVEL_DIRTY (1) alone flushes
//     without averaging; eHAS_IMPULSES (2) additionally averages first.  Both
//     bits are commonly set together (3).  The clear writes 0 through the
//     PxU8 store — four lanes per word, so it MUST be abStoreU8's atomic
//     and/or pair, never a read-modify-write.
//  2. **The averaging call passes `globalThreadIndex` as the articulation id**
//     — the THREAD index, not the warp index.  (Its `nbLinks` parameter is
//     dead in the engine — it appears in the signature and is never read —
//     and the P8.94c port dropped it; this call site is where that shows.)
//  3. **`mFlags` is read once and reused** for both the flush call's fixBase
//     and the root-branch guard.  The CUDA reads it twice; nothing between
//     the two reads writes it, so one read is equivalent — said here rather
//     than discovered.
//  4. **The root branch is guarded by !eFIX_BASE; the link loop is NOT.**  A
//     fixed-base articulation still integrates links 1..numLinks-1.
//  5. **`dq = body2World.q * prevPoseConjugate`** — the NEW pose on the left,
//     the conjugate of mPreTransform's quat on the right, then the
//     shortest-arc flip.  The engine's loadSpatialTransformQuatConjugate is a
//     fused load that negates exactly (x, y, z); here it is the composition
//     asLoadTransform(..).q -> asQuatConjugate.
//  6. **The two halves store their velocity sums in opposite spellings** —
//     root: `posMotionVelocity + delta`; links: `deltaMotionVel +
//     posMotionVelocity`.  Float addition is commutative bit-for-bit, so the
//     port keeps each spelling as written purely so a reader diffing against
//     the CUDA finds what it expects.
//  7. **The link loop is strictly sequential**: asPropagateLink(linkID) reads
//     the PARENT's mAccumulatedPose, which iteration parent<linkID already
//     overwrote.  storeSpatialTransform therefore happens INSIDE the loop
//     body before the next iteration, never batched after it.
//  8. **storeSpatialTransform writes p.w = 0**, not the loaded w — pinned by
//     the abStoreSpatialTransform port and visible in the fixture's poison.

import pxgSlangCommon;
import pxgSlangArticulationDesc;
import pxgSlangArticulationBlocks;
import pxgSlangSpatialMatrix;
import npArtiStep;
import npArtiFlushVel;
import npArtiAvgImpulse;

static const uint ASK_WARPS_PER_BLOCK = 2u;   // COMPUTE_UNCONSTRAINED_VELOCITES / 32

AbSpatialVector askLoadSV(uint64_t base, uint fieldOffset, uint lane)
{
    float3 top, bottom;
    abLoadSpatialVector(base, fieldOffset, lane, top, bottom);
    return asmSVec(top, bottom);
}

// The CUDA arg pack: descriptor pointer @0, stepDt by value @8 -> 16-byte blob.
struct AskParams
{
    uint64_t desc;   // @0
    float stepDt;    // @8
    uint pad0;       // @12 -> 16
};

[[vk::push_constant]]
ConstantBuffer<AskParams> gAsk;

[shader("compute")]
[numthreads(32, ASK_WARPS_PER_BLOCK, 1)]
void stepArticulation1TTGS(uint3 groupId : SV_GroupID, uint3 groupThreadId : SV_GroupThreadID)
{
    const uint64_t d = gAsk.desc;
    const float stepDt = gAsk.stepDt;

    const uint nbArticulations = artiU32Field(d, PXG_ARTI_CORE_NB_ARTICULATIONS);
    const uint warpIndex = groupThreadId.y;
    const uint globalWarpIndex = groupId.x * ASK_WARPS_PER_BLOCK + warpIndex;
    const uint lane = groupThreadId.x & 31u;
    const uint globalThreadIndex = globalWarpIndex * 32u + groupThreadId.x;

    if (globalThreadIndex >= nbArticulations)
        return;

    const uint maxLinks = artiU32Field(d, PXG_ARTI_CORE_MAX_LINKS);
    const uint maxDofs = artiU32Field(d, PXG_ARTI_CORE_MAX_DOFS);

    const uint64_t artiBlock = artiPtrField(d, PXG_ARTI_CORE_ARTICULATION_BLOCKS)
                             + uint64_t(globalWarpIndex) * PXG_AB_BLOCK_SIZE;
    const uint64_t linkBlocks = artiPtrField(d, PXG_ARTI_CORE_LINK_BLOCKS)
                              + uint64_t(globalWarpIndex) * uint64_t(maxLinks) * PXG_AB_LINK_SIZE;
    const uint64_t dofBlocks = artiPtrField(d, PXG_ARTI_CORE_DOF_BLOCKS)
                             + uint64_t(globalWarpIndex) * uint64_t(maxDofs) * PXG_AB_DOF_SIZE;

    const uint numLinks = abU32(artiBlock, PXG_AB_mNumLinks, lane);
    const uint stateDirty = abU8(artiBlock, PXG_AB_mStateDirty, lane);
    // trap 3: one read, two users
    const bool fixBase = (abU32(artiBlock, PXG_AB_mFlags, lane) & PXG_ARTI_FLAG_FIX_BASE) != 0u;

    if (stateDirty != 0u)
    {
        if ((stateDirty & PXG_ARTI_STATE_HAS_IMPULSES) != 0u)
        {
            // trap 2: the THREAD index is the articulation id
            aiAverageAndPropagate(artiPtrField(d, PXG_ARTI_CORE_SLAB_HAS_CHANGES),
                                  artiPtrField(d, PXG_ARTI_CORE_IMPULSES), artiBlock, linkBlocks,
                                  dofBlocks, globalThreadIndex, maxLinks, nbArticulations,
                                  artiU32Field(d, PXG_ARTI_CORE_NB_SLABS), lane);
        }
        fvFlushVelocity(artiBlock, linkBlocks, dofBlocks, numLinks, fixBase, lane);
        // trap 1: the atomic U8 store, four lanes per word
        abStoreU8(artiBlock, PXG_AB_mStateDirty, lane, 0u);
    }

    // ---- the root (trap 4: guarded; the link loop below is not) ----
    if (!fixBase)
    {
        const float4 prevPoseConjugate =
            asQuatConjugate(asLoadTransform(linkBlocks, PXG_AB_LINK_mPreTransform, lane).q);
        const AbSpatialVector motionVelocity =
            askLoadSV(linkBlocks, PXG_AB_LINK_mMotionVelocity, lane);
        const AbSpatialVector deltaMotion = askLoadSV(linkBlocks, PXG_AB_LINK_mDeltaMotion, lane);
        const AbSpatialVector posMotionVelocity =
            askLoadSV(linkBlocks, PXG_AB_LINK_mPosMotionVelocity, lane);

        const AsTransform body2World = asUpdateRootBody(
            motionVelocity, asLoadTransform(linkBlocks, PXG_AB_LINK_mAccumulatedPose, lane),
            stepDt);

        // trap 5
        float4 dq = quatMul(body2World.q, prevPoseConjugate);
        if (dq.w < 0.0f)
            dq = -dq;
        abStoreF4(linkBlocks, PXG_AB_LINK_mDeltaQ, lane, dq);

        const AbSpatialVector delta = asmSVecScale(motionVelocity, stepDt);
        const AbSpatialVector newDeltaMotion = asmSVecAdd(deltaMotion, delta);
        // trap 6: the root spells it posMotionVelocity + delta
        const AbSpatialVector newPosMotionVel = asmSVecAdd(posMotionVelocity, delta);
        abStoreSpatialVector(linkBlocks, PXG_AB_LINK_mDeltaMotion, lane, newDeltaMotion.top,
                             newDeltaMotion.bottom);
        abStoreSpatialVector(linkBlocks, PXG_AB_LINK_mPosMotionVelocity, lane,
                             newPosMotionVel.top, newPosMotionVel.bottom);
        abStoreSpatialTransform(linkBlocks, PXG_AB_LINK_mAccumulatedPose, lane, body2World.q,
                                body2World.p);
    }

    // ---- the links (trap 7: strictly sequential) ----
    for (uint linkID = 1u; linkID < numLinks; ++linkID)
    {
        const uint64_t link = linkBlocks + uint64_t(linkID) * PXG_AB_LINK_SIZE;

        const float4 prevPoseConjugate =
            asQuatConjugate(asLoadTransform(link, PXG_AB_LINK_mPreTransform, lane).q);
        AbSpatialVector deltaMotionVel = askLoadSV(link, PXG_AB_LINK_mMotionVelocity, lane);
        AbSpatialVector deltaMotion = askLoadSV(link, PXG_AB_LINK_mDeltaMotion, lane);
        const AbSpatialVector posMotionVelocity =
            askLoadSV(link, PXG_AB_LINK_mPosMotionVelocity, lane);

        const AsTransform body2World =
            asPropagateLink(linkBlocks, dofBlocks, linkID, stepDt, true, lane);

        float4 dq = quatMul(body2World.q, prevPoseConjugate);
        if (dq.w < 0.0f)
            dq = -dq;
        abStoreF4(link, PXG_AB_LINK_mDeltaQ, lane, dq);

        // the delta is integrated from the velocity, NOT recovered from the
        // pose difference -- the engine's own comment (PX-3638) says the pose
        // difference accumulates more numeric noise
        deltaMotionVel = asmSVecScale(deltaMotionVel, stepDt);
        deltaMotion = asmSVecAdd(deltaMotion, deltaMotionVel);

        abStoreSpatialVector(link, PXG_AB_LINK_mDeltaMotion, lane, deltaMotion.top,
                             deltaMotion.bottom);
        // trap 6: the links spell it deltaMotionVel + posMotionVelocity
        const AbSpatialVector newPosMotionVel = asmSVecAdd(deltaMotionVel, posMotionVelocity);
        abStoreSpatialVector(link, PXG_AB_LINK_mPosMotionVelocity, lane, newPosMotionVel.top,
                             newPosMotionVel.bottom);
        abStoreSpatialTransform(link, PXG_AB_LINK_mAccumulatedPose, lane, body2World.q,
                                body2World.p);
    }
}
