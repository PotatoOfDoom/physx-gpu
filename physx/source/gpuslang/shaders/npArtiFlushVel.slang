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


// npArtiFlushVel.slang
//
// P8.94b: `PxcFsFlushVelocity` (articulationDynamic.cuh:390) — the deferred-
// impulse flush `stepArticulation1TTGS` runs before it integrates
// (forwardDynamic2.cu:3023), and the same function four other launches call.
//
// NOTE there are TWO overloads of this name.  This is the SIX-argument one in
// `articulationDynamic.cuh`, which accumulates `mSolverSpatialDeltaVel`; the
// eight-argument one in forwardDynamic2.cu:3990 writes the solver's
// `outVelocity` buffer instead and does NOT touch `mSolverSpatialDeltaVel`.
// They are not interchangeable and the port takes the one the step kernel calls.
//
// Zero new ABI for the second tranche running: `mRootDeferredZ` (5760),
// `mInvSpatialArticulatedInertia` (0), `mCommonLinkDeltaVelocity` (4608),
// `mMotionVelocity` (9472), `mSolverSpatialDeltaVel` (13312), `mScratchDeltaV`
// (12544), `mConstraintForces` (17920), `mRw_{x,y,z}` (21120/21248/21376),
// `mParents` (28672), `mDofs` (28800) and the five dof fields are all mapped.
//
// ===========================================================================
// THE ORACLE
// ===========================================================================
//
// `PxcFsFlushVelocity` is `static __device__`, so this is the P8.73-P8.80
// arrangement rather than P8.81's: the mirror is control flow over engine
// primitives.  What it is built from is callable and IS called —
// `SpatialMatrix::operator*`, `UnAlignedSpatialVector::innerProduct`,
// `translateSpatialVector`, `loadSpatialVector` / `storeSpatialVector` /
// `loadSpatialMatrix` — and each already has its own fixture underneath it from
// P8.86 and P8.90.  What carries the rest is whole-region poison comparison and
// the sweep, and this note says so rather than presenting the sweep as an
// oracle.
//
// ===========================================================================
// THE TRAPS
// ===========================================================================
//
//  1. **`deferredZ` is NEGATED at the load** — `-loadSpatialVector(mRootDeferredZ)`
//     — and the SAME negated value is used twice: once as the right-hand side of
//     the inverse-inertia product and once as the increment to the ROOT's
//     `mConstraintForces`.
//  2. **A fixed base still writes three things.**  The `!fixBase` block is
//     skipped, but `deltaV` stays zero and `mScratchDeltaV`,
//     `mConstraintForces` and `mCommonLinkDeltaVelocity` are still written, and
//     the whole per-link loop still runs.
//  3. **`mRootDeferredZ` is zeroed ONLY when `!fixBase`.**  A fixed base leaves
//     it standing.  `mCommonLinkDeltaVelocity` is the opposite: zeroed
//     unconditionally, outside the guard.
//  4. **The root's constraint force takes `deferredZ`; every other link's takes
//     `deltaV`.**  That asymmetry is the single easiest thing to "tidy" wrong.
//  5. **Three different accumulations at the root** — `mMotionVelocity +=
//     deltaV`, `mSolverSpatialDeltaVel += deltaV`, `mConstraintForces +=
//     deferredZ — and only the first two are inside the `!fixBase` guard.
//  6. **`deltaV` CARRIES between iterations** and is re-loaded from
//     `artiLinks[parent].mScratchDeltaV` only when `parent != i - 1`.  A port
//     that never re-loads is wrong for a branching chain, so the fixture needs a
//     link whose parent is not the previous one.
//  7. **`dofs` is a running CURSOR** advanced by the LINK block's `mDofs`, not
//     `jointOffset + i` — the same shape P8.81 records.
//  8. **`mDeferredQstZ` is zeroed over `nbDofs` AFTER the propagate reads it**,
//     so the read and the zero are ordered.
//  9. **The loop is guarded by `linkCount > 1`**, and its prefetch reads link
//     `i + 1` before link `i` is written.
//
// ===========================================================================
// THE SWEEP: 32/37, AND ALL FIVE SURVIVORS PROVEN
// ===========================================================================
//
// Two were predicted before the sweep ran rather than explained after it, and
// three fall into two proof shapes:
//
//  * **F18 / F21 — the parent re-load made unconditional.**  `deltaV` carries
//    between iterations, and iteration `i - 1` stored exactly that value into
//    link `i - 1`'s `mScratchDeltaV` (the root stores it too, which is what
//    makes `i == 1` work).  So re-loading when `parent == i - 1` reads back the
//    value already in hand.  F21 (`parent != i` instead of `parent != i - 1`)
//    is the same mutation spelled differently: `parent < i` is the engine's
//    tree invariant, so `parent != i` is always true and the branch always
//    takes the load.
//  * **F33 — the constraint force re-loaded at the store.**  Every store in an
//    iteration targets link `i` alone, and the only cross-link read is
//    `artiLinks[parent].mScratchDeltaV`, so nothing an iteration writes can
//    change what its own preload would have read.  The same argument covers the
//    other three preloads.
//  * **F14 / F16 — the two range guards, and this one is worth stating
//    carefully.**  `linkCount > 1` and `(i + 1) < linkCount` guard nothing but
//    the PREFETCH, whose results a single-link lane (F14) or a final iteration
//    (F16) never uses.  They are therefore write-equivalent, and no
//    bit-comparison test can see them.  They are NOT redundant: without them
//    the prefetch reads `artiLinks[linkCount]`, which is past this warp's link
//    array in the engine's real allocation.  The guards are kept, and the
//    reason they survive the sweep is that the sweep measures writes.

import pxgSlangCommon;
import pxgSlangArticulationBlocks;
import pxgSlangSpatialMatrix;
import pxgSlangArticulationDynamic;
import pxgSlangArticulationDesc;   // P8.105: the artiOutputVelocity desc walk
import pxgSlangSolverDesc;         // P8.105: outArtiVelocity

static const AbSpatialVector FV_ZERO = { float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f) };

AbSpatialVector fvNeg(AbSpatialVector v) { return asmSVec(-v.top, -v.bottom); }

// addSpatialVector (PxgArticulation.h): a read-modify-write of the SoA field.
void fvAddSpatialVector(uint64_t base, uint fieldOffset, uint lane, AbSpatialVector d)
{
    float3 top, bottom;
    abLoadSpatialVector(base, fieldOffset, lane, top, bottom);
    abStoreSpatialVector(base, fieldOffset, lane, top + d.top, bottom + d.bottom);
}

AbSpatialVector fvLoadSV(uint64_t base, uint fieldOffset, uint lane)
{
    float3 top, bottom;
    abLoadSpatialVector(base, fieldOffset, lane, top, bottom);
    return asmSVec(top, bottom);
}

void fvStoreSV(uint64_t base, uint fieldOffset, uint lane, AbSpatialVector v)
{
    abStoreSpatialVector(base, fieldOffset, lane, v.top, v.bottom);
}

// PxcFsFlushVelocity (articulationDynamic.cuh:390).
public void fvFlushVelocity(uint64_t artiBlock, uint64_t linkBlocks, uint64_t dofBlocks,
                            uint linkCount, bool fixBase, uint lane)
{
    AbSpatialVector deltaV = FV_ZERO;
    // trap 1: NEGATED at the load, and used twice
    const AbSpatialVector deferredZ = fvNeg(fvLoadSV(artiBlock, PXG_AB_mRootDeferredZ, lane));

    if (!fixBase)
    {
        // preloaded before the stores -- see the header note on why that is
        // equivalent here and is kept anyway
        const AbSpatialVector motionVelocity0 =
            fvLoadSV(linkBlocks, PXG_AB_LINK_mMotionVelocity, lane);
        const AbSpatialVector solverSpatialDeltaVel0 =
            fvLoadSV(linkBlocks, PXG_AB_LINK_mSolverSpatialDeltaVel, lane);

        AbMat33 tl, tr, bl;
        abLoadSpatialMatrix(artiBlock, PXG_AB_mInvSpatialArticulatedInertia, lane, tl, tr, bl);
        const AbSpatialMatrix invInertia = asmMatrix(tl, tr, bl);

        deltaV = asmMulVec(invInertia, deferredZ);

        fvStoreSV(linkBlocks, PXG_AB_LINK_mMotionVelocity, lane,
                  asmSVecAdd(motionVelocity0, deltaV));
        // trap 3: only the FIXED-BASE path leaves this standing
        fvStoreSV(artiBlock, PXG_AB_mRootDeferredZ, lane, FV_ZERO);
        fvStoreSV(linkBlocks, PXG_AB_LINK_mSolverSpatialDeltaVel, lane,
                  asmSVecAdd(solverSpatialDeltaVel0, deltaV));
    }

    // trap 2: all three of these run for a fixed base too
    fvStoreSV(linkBlocks, PXG_AB_LINK_mScratchDeltaV, lane, deltaV);
    // trap 4: the ROOT's increment is deferredZ, not deltaV
    fvAddSpatialVector(linkBlocks, PXG_AB_LINK_mConstraintForces, lane, deferredZ);
    fvStoreSV(artiBlock, PXG_AB_mCommonLinkDeltaVelocity, lane, FV_ZERO);

    uint64_t dofs = dofBlocks;

    if (linkCount > 1u)
    {
        const uint64_t link1 = linkBlocks + uint64_t(PXG_AB_LINK_SIZE);
        AbSpatialVector nextMotionV = fvLoadSV(link1, PXG_AB_LINK_mMotionVelocity, lane);
        uint nextNbDofs = abU8(link1, PXG_AB_LINK_mDofs, lane);
        uint nextParent = abU32(link1, PXG_AB_LINK_mParents, lane);

        for (uint i = 1u; i < linkCount; ++i)
        {
            const uint64_t tLink = linkBlocks + uint64_t(i) * PXG_AB_LINK_SIZE;
            const uint nbDofs = nextNbDofs;
            const uint parent = nextParent;

            const AbSpatialVector preloadedConstraintForces =
                fvLoadSV(tLink, PXG_AB_LINK_mConstraintForces, lane);
            const AbSpatialVector preloadedSolverSpatialDeltaVel =
                fvLoadSV(tLink, PXG_AB_LINK_mSolverSpatialDeltaVel, lane);

            AbSpatialVector motionV = nextMotionV;

            // trap 9: the prefetch reads link i+1 before link i is written
            if ((i + 1u) < linkCount)
            {
                const uint64_t nLink = linkBlocks + uint64_t(i + 1u) * PXG_AB_LINK_SIZE;
                nextMotionV = fvLoadSV(nLink, PXG_AB_LINK_mMotionVelocity, lane);
                nextNbDofs = abU8(nLink, PXG_AB_LINK_mDofs, lane);
                nextParent = abU32(nLink, PXG_AB_LINK_mParents, lane);
            }

            // trap 6: the carry, broken only when the parent is not i-1
            if (parent != (i - 1u))
            {
                const uint64_t pLink = linkBlocks + uint64_t(parent) * PXG_AB_LINK_SIZE;
                deltaV = fvLoadSV(pLink, PXG_AB_LINK_mScratchDeltaV, lane);
            }

            // propagateAccelerationW_nothingpreloaded_updateJointDofVels
            // (articulationDynamic.cuh:329): the P8.90 body, with the block
            // loads done here and the joint velocities written back.
            AbSpatialVector isW[3];
            float qstZ[3];
            float jVel[3];
            float3 invStIsT[3];
            AbSpatialVector motionMatrix[3];
            for (uint ind = 0u; ind < 3u; ++ind)
            {
                isW[ind] = FV_ZERO;
                qstZ[ind] = 0.0f;
                jVel[ind] = 0.0f;
                invStIsT[ind] = float3(0.0f, 0.0f, 0.0f);
                motionMatrix[ind] = FV_ZERO;
                if (ind < nbDofs)
                {
                    const uint64_t d = dofs + uint64_t(ind) * PXG_AB_DOF_SIZE;
                    isW[ind] = fvLoadSV(d, PXG_AB_DOF_mIsW, lane);
                    qstZ[ind] = abF32(d, PXG_AB_DOF_mDeferredQstZ, lane);
                    jVel[ind] = abF32(d, PXG_AB_DOF_mJointVelocities, lane);
                    invStIsT[ind] = float3(abF32(d, PXG_AB_DOF_mInvStIsT_x, lane),
                                           abF32(d, PXG_AB_DOF_mInvStIsT_y, lane),
                                           abF32(d, PXG_AB_DOF_mInvStIsT_z, lane));
                    motionMatrix[ind] = fvLoadSV(d, PXG_AB_DOF_mWorldMotionMatrix, lane);
                }
            }

            const float3 parentToChild = float3(abF32(tLink, PXG_AB_LINK_mRw_x, lane),
                                                abF32(tLink, PXG_AB_LINK_mRw_y, lane),
                                                abF32(tLink, PXG_AB_LINK_mRw_z, lane));

            float jDelta[3];
            deltaV = adPropagateAccelerationWEx(parentToChild, invStIsT, motionMatrix, deltaV,
                                                nbDofs, isW, qstZ, jDelta);

            for (uint ind = 0u; ind < 3u; ++ind)
            {
                if (ind < nbDofs)
                {
                    const uint64_t d = dofs + uint64_t(ind) * PXG_AB_DOF_SIZE;
                    abStoreF32(d, PXG_AB_DOF_mJointVelocities, lane, jVel[ind] + jDelta[ind]);
                }
            }

            fvStoreSV(tLink, PXG_AB_LINK_mSolverSpatialDeltaVel, lane,
                      asmSVecAdd(preloadedSolverSpatialDeltaVel, deltaV));

            // trap 8: zeroed AFTER the propagate read it
            for (uint ind = 0u; ind < nbDofs; ++ind)
                abStoreF32(dofs + uint64_t(ind) * PXG_AB_DOF_SIZE, PXG_AB_DOF_mDeferredQstZ, lane,
                           0.0f);

            motionV = asmSVecAdd(motionV, deltaV);

            fvStoreSV(tLink, PXG_AB_LINK_mScratchDeltaV, lane, deltaV);
            fvStoreSV(tLink, PXG_AB_LINK_mMotionVelocity, lane, motionV);
            // trap 4 again: every NON-root link takes deltaV
            fvStoreSV(tLink, PXG_AB_LINK_mConstraintForces, lane,
                      asmSVecAdd(preloadedConstraintForces, deltaV));

            dofs += uint64_t(nbDofs) * PXG_AB_DOF_SIZE;   // trap 7: a running cursor
        }
    }
}

// ---------------------------------------------------------------------------
// The entry point the fixture drives.  One articulation per lane, block bases
// handed in directly -- the index chain belongs to the kernel (P8.94d).
// ---------------------------------------------------------------------------
struct FvParams
{
    uint64_t artiBlock;    // @0
    uint64_t linkBlocks;   // @8
    uint64_t dofBlocks;    // @16
    uint64_t counts;       // @24  uint[32] link counts
    uint64_t flags;        // @32  uint[32] non-zero == fixed base
    uint nbLanes;          // @40
    uint pad0;             // @44 -> 48
};

[[vk::push_constant]]
ConstantBuffer<FvParams> gFv;

[shader("compute")]
[numthreads(32, 1, 1)]
void artiFlushVelocity(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint lane = dispatchThreadId.x;
    if (lane >= gFv.nbLanes)
        return;

    const uint linkCount = (Ptr<uint>(gFv.counts))[lane];
    const bool fixBase = (Ptr<uint>(gFv.flags))[lane] != 0u;

    fvFlushVelocity(gFv.artiBlock, gFv.linkBlocks, gFv.dofBlocks, linkCount, fixBase, lane);
}

// ===========================================================================
// P8.105: the EIGHT-argument PxcFsFlushVelocity overload
// (forwardDynamic2.cu:3990-4073) and the artiOutputVelocity engine entry
// (:4076-4137).  The overload is the six-argument body above MINUS the two
// mSolverSpatialDeltaVel accumulations, PLUS the solver outVelocity writes
// (linear at [i], angular at [i + offset], w = 0).
//
// TRAPS (continuing the module's numbering; each carries a sweep mutant):
// 10. **No mSolverSpatialDeltaVel writes anywhere** -- the six-argument
//     overload accumulates it at the root and per link, this one must NOT.
// 11. **outVelocity[0] takes the UPDATED root velocity** (vel + deltaV), and
//     only on the free-base path -- a fixed base leaves out[0]/out[offset]
//     UNWRITTEN in the dirty branch.
// 12. **The kernel's else-branch** (mStateDirty == 0) copies mMotionVelocity
//     straight out WITHOUT flushing, for every link including the root.
// 13. **The isTGS tail writes mDeltaMotion at [i + 2*offset] / [i + 3*offset]**
//     regardless of the dirty flag.
// 14. mStateDirty is cleared (0) only on the DIRTY path.
// 15. outArtiVelocity is indexed per-ARTICULATION: base + globalThreadIndex *
//     maxLinks, with offset = nbArticulations * maxLinks.
// ===========================================================================

public void fvFlushVelocityOut(uint64_t artiBlock, uint64_t linkBlocks, uint64_t dofBlocks,
                               uint linkCount, bool fixBase, uint lane,
                               uint64_t outVelocity, uint offset)
{
    Ptr<float4> outVel = Ptr<float4>(outVelocity);

    AbSpatialVector deltaV = FV_ZERO;
    const AbSpatialVector deferredZ = fvNeg(fvLoadSV(artiBlock, PXG_AB_mRootDeferredZ, lane));

    if (!fixBase)
    {
        const AbSpatialVector motionVelocity0 =
            fvLoadSV(linkBlocks, PXG_AB_LINK_mMotionVelocity, lane);

        AbMat33 tl, tr, bl;
        abLoadSpatialMatrix(artiBlock, PXG_AB_mInvSpatialArticulatedInertia, lane, tl, tr, bl);
        const AbSpatialMatrix invInertia = asmMatrix(tl, tr, bl);

        deltaV = asmMulVec(invInertia, deferredZ);

        const AbSpatialVector vel = asmSVecAdd(motionVelocity0, deltaV);

        // trap 11
        outVel[0] = float4(vel.bottom, 0.0f);
        outVel[offset] = float4(vel.top, 0.0f);

        fvStoreSV(linkBlocks, PXG_AB_LINK_mMotionVelocity, lane, vel);
        fvStoreSV(artiBlock, PXG_AB_mRootDeferredZ, lane, FV_ZERO);
        // trap 10: NO mSolverSpatialDeltaVel accumulation
    }

    fvStoreSV(linkBlocks, PXG_AB_LINK_mScratchDeltaV, lane, deltaV);
    fvAddSpatialVector(linkBlocks, PXG_AB_LINK_mConstraintForces, lane, deferredZ);
    fvStoreSV(artiBlock, PXG_AB_mCommonLinkDeltaVelocity, lane, FV_ZERO);

    uint64_t dofs = dofBlocks;

    if (linkCount > 1u)
    {
        const uint64_t link1 = linkBlocks + uint64_t(PXG_AB_LINK_SIZE);
        AbSpatialVector nextMotionV = fvLoadSV(link1, PXG_AB_LINK_mMotionVelocity, lane);
        uint nextNbDofs = abU8(link1, PXG_AB_LINK_mDofs, lane);
        uint nextParent = abU32(link1, PXG_AB_LINK_mParents, lane);

        for (uint i = 1u; i < linkCount; ++i)
        {
            const uint64_t tLink = linkBlocks + uint64_t(i) * PXG_AB_LINK_SIZE;
            const uint nbDofs = nextNbDofs;
            const uint parent = nextParent;

            const AbSpatialVector preloadedConstraintForces =
                fvLoadSV(tLink, PXG_AB_LINK_mConstraintForces, lane);

            AbSpatialVector motionV = nextMotionV;

            if ((i + 1u) < linkCount)
            {
                const uint64_t nLink = linkBlocks + uint64_t(i + 1u) * PXG_AB_LINK_SIZE;
                nextMotionV = fvLoadSV(nLink, PXG_AB_LINK_mMotionVelocity, lane);
                nextNbDofs = abU8(nLink, PXG_AB_LINK_mDofs, lane);
                nextParent = abU32(nLink, PXG_AB_LINK_mParents, lane);
            }

            if (parent != (i - 1u))
            {
                const uint64_t pLink = linkBlocks + uint64_t(parent) * PXG_AB_LINK_SIZE;
                deltaV = fvLoadSV(pLink, PXG_AB_LINK_mScratchDeltaV, lane);
            }

            AbSpatialVector isW[3];
            float qstZ[3];
            float jVel[3];
            float3 invStIsT[3];
            AbSpatialVector motionMatrix[3];
            for (uint ind = 0u; ind < 3u; ++ind)
            {
                isW[ind] = FV_ZERO;
                qstZ[ind] = 0.0f;
                jVel[ind] = 0.0f;
                invStIsT[ind] = float3(0.0f, 0.0f, 0.0f);
                motionMatrix[ind] = FV_ZERO;
                if (ind < nbDofs)
                {
                    const uint64_t d = dofs + uint64_t(ind) * PXG_AB_DOF_SIZE;
                    isW[ind] = fvLoadSV(d, PXG_AB_DOF_mIsW, lane);
                    qstZ[ind] = abF32(d, PXG_AB_DOF_mDeferredQstZ, lane);
                    jVel[ind] = abF32(d, PXG_AB_DOF_mJointVelocities, lane);
                    invStIsT[ind] = float3(abF32(d, PXG_AB_DOF_mInvStIsT_x, lane),
                                           abF32(d, PXG_AB_DOF_mInvStIsT_y, lane),
                                           abF32(d, PXG_AB_DOF_mInvStIsT_z, lane));
                    motionMatrix[ind] = fvLoadSV(d, PXG_AB_DOF_mWorldMotionMatrix, lane);
                }
            }

            const float3 parentToChild = float3(abF32(tLink, PXG_AB_LINK_mRw_x, lane),
                                                abF32(tLink, PXG_AB_LINK_mRw_y, lane),
                                                abF32(tLink, PXG_AB_LINK_mRw_z, lane));

            float jDelta[3];
            deltaV = adPropagateAccelerationWEx(parentToChild, invStIsT, motionMatrix, deltaV,
                                                nbDofs, isW, qstZ, jDelta);

            for (uint ind = 0u; ind < 3u; ++ind)
            {
                if (ind < nbDofs)
                {
                    const uint64_t d = dofs + uint64_t(ind) * PXG_AB_DOF_SIZE;
                    abStoreF32(d, PXG_AB_DOF_mJointVelocities, lane, jVel[ind] + jDelta[ind]);
                }
            }

            for (uint ind = 0u; ind < nbDofs; ++ind)
                abStoreF32(dofs + uint64_t(ind) * PXG_AB_DOF_SIZE, PXG_AB_DOF_mDeferredQstZ, lane,
                           0.0f);

            motionV = asmSVecAdd(motionV, deltaV);

            fvStoreSV(tLink, PXG_AB_LINK_mScratchDeltaV, lane, deltaV);

            outVel[i] = float4(motionV.bottom, 0.0f);
            outVel[i + offset] = float4(motionV.top, 0.0f);

            fvStoreSV(tLink, PXG_AB_LINK_mMotionVelocity, lane, motionV);
            fvStoreSV(tLink, PXG_AB_LINK_mConstraintForces, lane,
                      asmSVecAdd(preloadedConstraintForces, deltaV));

            dofs += uint64_t(nbDofs) * PXG_AB_DOF_SIZE;
        }
    }
}

// ---------------------------------------------------------------------------
// artiOutputVelocity (forwardDynamic2.cu:4076): grid ceil(nb/32), block
// (32,1,1) -- one warp per block, w = groupId.x directly.
// ---------------------------------------------------------------------------

struct OvParams
{
    uint64_t desc;        // @0  PxgArticulationCoreDesc*
    uint64_t solverDesc;  // @8  PxgSolverCoreDesc*
    uint isTGSWord;       // @16 bool byte 0 (bridge zero-pads 17..23)
}

[[vk::push_constant]]
ConstantBuffer<OvParams> gOv;

[shader("compute")]
[numthreads(32, 1, 1)]
void artiOutputVelocity(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint64_t d = gOv.desc;
    const uint lane = groupThreadId.x;
    const uint globalWarpIndex = groupId.x;
    const uint globalThreadIndex = globalWarpIndex * 32u + lane;

    const uint nbArticulations = artiU32Field(d, PXG_ARTI_CORE_NB_ARTICULATIONS);
    if (globalThreadIndex >= nbArticulations)
        return;

    const uint maxLinks = artiU32Field(d, PXG_ARTI_CORE_MAX_LINKS);
    const uint maxDofs = artiU32Field(d, PXG_ARTI_CORE_MAX_DOFS);

    const uint64_t artiBlock = artiPtrField(d, PXG_ARTI_CORE_ARTICULATION_BLOCKS)
                             + uint64_t(globalWarpIndex) * PXG_AB_BLOCK_SIZE;
    const uint64_t linkBase = artiPtrField(d, PXG_ARTI_CORE_LINK_BLOCKS)
                            + uint64_t(globalWarpIndex) * uint64_t(maxLinks) * PXG_AB_LINK_SIZE;
    const uint64_t dofBase = artiPtrField(d, PXG_ARTI_CORE_DOF_BLOCKS)
                           + uint64_t(globalWarpIndex) * uint64_t(maxDofs) * PXG_AB_DOF_SIZE;

    const uint numLinks = abU32(artiBlock, PXG_AB_mNumLinks, lane);

    Ptr<PxgSolverCoreDescGpu> solverDesc = Ptr<PxgSolverCoreDescGpu>(gOv.solverDesc);
    // trap 15: per-articulation slice of the solver's output buffer
    const uint64_t outArtiVelocity = solverDesc[0].outArtiVelocity
                                   + uint64_t(globalThreadIndex) * uint64_t(maxLinks) * 16u;
    const uint offset = nbArticulations * maxLinks;

    const uint dirtyFlag = abU8(artiBlock, PXG_AB_mStateDirty, lane);
    if (dirtyFlag != 0u)
    {
        const bool fixBase = (abU32(artiBlock, PXG_AB_mFlags, lane) & 1u) != 0u;  // eFIX_BASE
        fvFlushVelocityOut(artiBlock, linkBase, dofBase, numLinks, fixBase, lane,
                           outArtiVelocity, offset);
        abStoreU8(artiBlock, PXG_AB_mStateDirty, lane, 0u);   // trap 14
    }
    else
    {
        // trap 12
        Ptr<float4> outVel = Ptr<float4>(outArtiVelocity);
        for (uint i = 0u; i < numLinks; ++i)
        {
            const AbSpatialVector vel =
                fvLoadSV(linkBase + uint64_t(i) * PXG_AB_LINK_SIZE, PXG_AB_LINK_mMotionVelocity,
                         lane);
            outVel[i] = float4(vel.bottom, 0.0f);
            outVel[i + offset] = float4(vel.top, 0.0f);
        }
    }

    // trap 13
    if ((gOv.isTGSWord & 0xFFu) != 0u)
    {
        Ptr<float4> outVel = Ptr<float4>(outArtiVelocity);
        for (uint i = 0u; i < numLinks; ++i)
        {
            const AbSpatialVector delta =
                fvLoadSV(linkBase + uint64_t(i) * PXG_AB_LINK_SIZE, PXG_AB_LINK_mDeltaMotion, lane);
            outVel[i + 2u * offset] = float4(delta.bottom, 0.0f);
            outVel[i + 3u * offset] = float4(delta.top, 0.0f);
        }
    }
}
