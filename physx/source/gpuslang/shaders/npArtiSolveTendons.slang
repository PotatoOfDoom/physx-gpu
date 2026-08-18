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


// npArtiSolveTendons.slang
//
// P8.98a: the two deferred-impulse workhorses of the per-iteration solve
// chain (internalConstraints2.cu), which every internal-constraint solve body
// composes:
//
//   * `svGetVelocity`   <- pxcFsGetVelocity (:2116) — the current velocity of
//     one link under the DEFERRED impulses: root deltaV from
//     mInvSpatialArticulatedInertia * -mRootDeferredZ (free base only), then
//     the path-to-root walk root→link propagating accelerations with
//     Q-stZ read from mDeferredQstZ, optionally reporting the link's updated
//     joint dof speeds.
//   * `svApplyImpulses` <- pxcFsApplyImpulses (:2190) — apply a spatial (and
//     optional joint) impulse at TWO links: find the common ancestor via the
//     bitfields (TOP word down, HIGHEST bit), propagate each Z to the common
//     link, sum, propagate the sum to the root, accumulating into
//     mDeferredQstZ and mSolverSpatialImpulse along the way and into
//     mRootDeferredZ at the end.
//
// ===========================================================================
// TRAPS (each carries a mutant in the sweep)
// ===========================================================================
//
//  1. **The velocity walk is WORD-0-UP, LOWEST-bit-first** (root-to-link),
//     and the ROOT (index 0) is SKIPPED — but only after the word offset is
//     added (`index != 0`), so bit 0 of word 1 (link 64) is NOT skipped.
//  2. **The common-ancestor search is TOP-word-DOWN with the HIGHEST set
//     bit** — the opposite scan direction from the velocity walk.
//  3. **Z is built NEGATED**: `Z = (-linear, -angular)` with linear in the
//     TOP half (an impulse, so top = linear, the P8.84 convention).
//  4. **mSolverSpatialImpulse accumulates the PRE-propagation Z** of each
//     hop — the add happens BEFORE propagateImpulseW, per walk iteration.
//  5. **The joint impulse applies only at the link's OWN inbound joint**
//     (`i == linkID`), and if linkID IS the common link it is applied on the
//     common→root walk instead — accumulated from BOTH ends when linkID0 ==
//     linkID1 == commonLink.
//  6. **The common→root loop does not run for commonLink == 0**, so a joint
//     impulse "at the root" is dropped (the root has no inbound joint).
//  7. **The joint-speed report is jointVelocities + delta**, with the delta
//     coming only from the hop where `index == linkID`, and the deltas of
//     OTHER links' hops discarded.
//  8. **The template instance spells `stZ = jf - dot` where the preloaded
//     variant spells `-dot`** — different signed zero at dot == +0; the
//     module calls adPropagateImpulseWJf (the template spelling), never
//     adPropagateImpulseW.
//  9. `qstZ` seeds from mDeferredQstZ and the result is written BACK — the
//     `+=` of WriteToDeferredQstZ is the load/accumulate/store composition.
// 10. The root deltaV negates the WHOLE rootDeferredZ vector BEFORE the
//     6x6 multiply (`M * -Z`), not the product.

import pxgSlangCommon;
import pxgSlangArticulationBlocks;
import pxgSlangSpatialMatrix;
import pxgSlangArticulationDynamic;

uint svHighestBit64(uint64_t v)
{
    const uint hi = uint(v >> 32);
    return hi != 0u ? 32u + firstbithigh(hi) : firstbithigh(uint(v));
}

uint svLowestBit64(uint64_t v)
{
    const uint lo = uint(v);
    return lo != 0u ? firstbitlow(lo) : 32u + firstbitlow(uint(v >> 32));
}

AbSpatialVector svLoadSV(uint64_t base, uint fieldOffset, uint lane)
{
    float3 top, bottom;
    abLoadSpatialVector(base, fieldOffset, lane, top, bottom);
    return asmSVec(top, bottom);
}

void svAddSV(uint64_t base, uint fieldOffset, uint lane, AbSpatialVector v)
{
    const AbSpatialVector cur = svLoadSV(base, fieldOffset, lane);
    const AbSpatialVector sum = asmSVecAdd(cur, v);
    abStoreSpatialVector(base, fieldOffset, lane, sum.top, sum.bottom);
}

// the per-link dof arrays every propagation hop needs
void svLoadDofArrays(uint64_t dofBlocks, uint jointOffset, uint dofCount, uint lane,
                     out AbSpatialVector motion[3], out AbSpatialVector isInvD[3],
                     out AbSpatialVector isW[3], out float3 invStIsT[3])
{
    for (uint i = 0u; i < 3u; ++i)
    {
        motion[i] = asmSVec(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f));
        isInvD[i] = motion[i];
        isW[i] = motion[i];
        invStIsT[i] = float3(0.0f, 0.0f, 0.0f);
        if (i < dofCount)
        {
            const uint64_t d = dofBlocks + uint64_t(jointOffset + i) * PXG_AB_DOF_SIZE;
            motion[i] = svLoadSV(d, PXG_AB_DOF_mWorldMotionMatrix, lane);
            isInvD[i] = svLoadSV(d, PXG_AB_DOF_mIsInvDW, lane);
            isW[i] = svLoadSV(d, PXG_AB_DOF_mIsW, lane);
            invStIsT[i] = float3(abF32(d, PXG_AB_DOF_mInvStIsT_x, lane),
                                 abF32(d, PXG_AB_DOF_mInvStIsT_y, lane),
                                 abF32(d, PXG_AB_DOF_mInvStIsT_z, lane));
        }
    }
}

// propagateAccelerationW_child2ParentPreLoaded<ReadFromDeferredQstZ>
// (articulationDynamic.cuh:257): loads everything afresh, Q-stZ from
// mDeferredQstZ, delta joint speeds reported when wanted.
//
// NOT adPropagateAccelerationWEx: this variant accumulates the per-dof
// contributions into a SEPARATE jointSpatialDeltaV and returns
// `pDeltaV + jointSpatialDeltaV` at the end, where the everythingPreLoaded
// variant accumulates directly onto pDeltaV per dof — for a two-dof joint
// that is `pd + (a + b)` versus `(pd + a) + b`, a different rounding, and
// the fixture measured 1-2 ULP divergence on exactly the free-base cases
// before this was spelled per the .cuh.
AbSpatialVector svPropagateAccelDeferred(float3 parentToChild, uint64_t dofBlocks,
                                         uint jointOffset, uint dofCount,
                                         AbSpatialVector parentDeltaV, uint lane,
                                         inout float deltaJointSpeeds[3], bool wantSpeeds)
{
    AbSpatialVector motion[3];
    AbSpatialVector isInvD[3];
    AbSpatialVector isW[3];
    float3 invStIsT[3];
    svLoadDofArrays(dofBlocks, jointOffset, dofCount, lane, motion, isInvD, isW, invStIsT);

    // NEGATED offset, as in every acceleration propagation
    const AbSpatialVector pDeltaV = adTranslateSpatialVector(-parentToChild, parentDeltaV);

    // pass one: every entry from the PRE-accumulation pDeltaV
    float tJointDelta[3] = { 0.0f, 0.0f, 0.0f };
    for (uint ind = 0u; ind < 3u; ++ind)
    {
        if (ind < dofCount)
        {
            const float temp = asmInnerProduct(isW[ind], pDeltaV);
            const float qstz = abF32(dofBlocks + uint64_t(jointOffset + ind) * PXG_AB_DOF_SIZE,
                                     PXG_AB_DOF_mDeferredQstZ, lane);
            tJointDelta[ind] = qstz - temp;
        }
    }

    // pass two: accumulate SEPARATELY, add to pDeltaV at the END
    AbSpatialVector jointSpatialDeltaV =
        asmSVec(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f));
    for (uint ind = 0u; ind < 3u; ++ind)
    {
        if (ind < dofCount)
        {
            const float3 iStIsTi = invStIsT[ind];
            const float jDelta = iStIsTi.x * tJointDelta[0]
                               + iStIsTi.y * tJointDelta[1]
                               + iStIsTi.z * tJointDelta[2];
            if (wantSpeeds)
                deltaJointSpeeds[ind] = jDelta;
            jointSpatialDeltaV = asmSVecAdd(jointSpatialDeltaV,
                                            asmSVecScale(motion[ind], jDelta));
        }
    }

    return asmSVecAdd(pDeltaV, jointSpatialDeltaV);
}

// propagateImpulseW<WriteToDeferredQstZ>: qstZ seeded from mDeferredQstZ,
// accumulated by adPropagateImpulseWJf, written back (trap 9).
AbSpatialVector svPropagateImpulseDeferred(float3 childToParent, uint64_t dofBlocks,
                                           uint jointOffset, uint dofCount,
                                           AbSpatialVector Z, uint lane,
                                           float jf[3], float jfMult, bool hasJf)
{
    AbSpatialVector motion[3];
    AbSpatialVector isInvD[3];
    AbSpatialVector isW[3];
    float3 invStIsT[3];
    svLoadDofArrays(dofBlocks, jointOffset, dofCount, lane, motion, isInvD, isW, invStIsT);

    float qstZ[3] = { 0.0f, 0.0f, 0.0f };
    for (uint i = 0u; i < 3u; ++i)
    {
        if (i < dofCount)
            qstZ[i] = abF32(dofBlocks + uint64_t(jointOffset + i) * PXG_AB_DOF_SIZE,
                            PXG_AB_DOF_mDeferredQstZ, lane);
    }

    const AbSpatialVector r = adPropagateImpulseWJf(isInvD, childToParent, motion, Z,
                                                    dofCount, qstZ, jf, jfMult, hasJf);

    for (uint i = 0u; i < 3u; ++i)
    {
        if (i < dofCount)
            abStoreF32(dofBlocks + uint64_t(jointOffset + i) * PXG_AB_DOF_SIZE,
                       PXG_AB_DOF_mDeferredQstZ, lane, qstZ[i]);
    }
    return r;
}

// ===========================================================================
// pxcFsGetVelocity (internalConstraints2.cu:2116)
// ===========================================================================

public AbSpatialVector svGetVelocity(uint64_t artiBlock, uint64_t linkBase,
                                     uint64_t dofBlocks, uint64_t bitFields,
                                     uint wordSize, uint linkID, bool fixBase,
                                     uint lane,
                                     inout float jointDofSpeeds[3], bool wantSpeeds)
{
    AbSpatialVector deltaV = asmSVec(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f));

    if (!fixBase)
    {
        const AbSpatialVector rootDeferredZ = svLoadSV(artiBlock, PXG_AB_mRootDeferredZ, lane);
        const AbSpatialMatrix m =
            asmLoadSpatialMatrix(artiBlock, PXG_AB_mInvSpatialArticulatedInertia, lane);
        // trap 10: M * (-Z), the vector negated before the multiply
        deltaV = asmMulVec(m, asmSVec(-rootDeferredZ.top, -rootDeferredZ.bottom));
    }

    float deltaJointDofSpeeds[3] = { 0.0f, 0.0f, 0.0f };

    const uint64_t link = linkBase + uint64_t(linkID) * PXG_AB_LINK_SIZE;

    for (uint j = 0u, wordOffset = 0u; j < wordSize; ++j, wordOffset += 64u)
    {
        uint64_t pathToRoot =
            abU64(bitFields + uint64_t(linkID * wordSize + j) * PXG_AB_BITFIELD_STACK_SIZE,
                  0u, lane);

        while (pathToRoot != 0uL)
        {
            const uint index = svLowestBit64(pathToRoot) + wordOffset;

            if (index != 0u)   // trap 1: the root is skipped AFTER the offset add
            {
                const uint64_t cLink = linkBase + uint64_t(index) * PXG_AB_LINK_SIZE;
                const uint jointOffset = abU32(cLink, PXG_AB_LINK_mJointOffset, lane);
                const uint dofCount = abU8(cLink, PXG_AB_LINK_mDofs, lane);
                const float3 childToParent = float3(abF32(cLink, PXG_AB_LINK_mRw_x, lane),
                                                    abF32(cLink, PXG_AB_LINK_mRw_y, lane),
                                                    abF32(cLink, PXG_AB_LINK_mRw_z, lane));

                // trap 7: only the link's OWN hop reports speeds
                const bool wantHere = (linkID == index) && wantSpeeds;

                deltaV = svPropagateAccelDeferred(childToParent, dofBlocks, jointOffset,
                                                  dofCount, deltaV, lane,
                                                  deltaJointDofSpeeds, wantHere);
            }

            pathToRoot &= (pathToRoot - 1uL);
        }
    }

    if (wantSpeeds)
    {
        const uint dofCount = abU8(link, PXG_AB_LINK_mDofs, lane);
        const uint jointOffset = abU32(link, PXG_AB_LINK_mJointOffset, lane);
        for (uint i = 0u; i < 3u; ++i)
        {
            if (i < dofCount)
                jointDofSpeeds[i] =
                    abF32(dofBlocks + uint64_t(jointOffset + i) * PXG_AB_DOF_SIZE,
                          PXG_AB_DOF_mJointVelocities, lane)
                    + deltaJointDofSpeeds[i];
        }
    }

    const AbSpatialVector motionVelocity = svLoadSV(link, PXG_AB_LINK_mMotionVelocity, lane);
    return asmSVecAdd(motionVelocity, deltaV);
}

// ===========================================================================
// pxcFsApplyImpulses (internalConstraints2.cu:2190)
// ===========================================================================

public void svApplyImpulses(uint64_t artiBlock, uint64_t linkBase, uint64_t dofBlocks,
                            uint64_t bitFields, uint wordSize,
                            uint linkID0, float3 linear0, float3 angular0,
                            float jointImpulse0[3], bool hasJi0,
                            uint linkID1, float3 linear1, float3 angular1,
                            float jointImpulse1[3], bool hasJi1,
                            uint lane)
{
    // trap 2: TOP word down, HIGHEST set bit
    uint commonLink = 0u;
    for (int i = int(wordSize) - 1; i >= 0; --i)
    {
        const uint64_t wordA =
            abU64(bitFields + uint64_t(linkID0 * wordSize + uint(i)) * PXG_AB_BITFIELD_STACK_SIZE,
                  0u, lane);
        const uint64_t wordB =
            abU64(bitFields + uint64_t(linkID1 * wordSize + uint(i)) * PXG_AB_BITFIELD_STACK_SIZE,
                  0u, lane);
        const uint64_t commonId = wordA & wordB;
        if (commonId != 0uL)
        {
            commonLink = svHighestBit64(commonId) + uint(i) * 64u;
            break;
        }
    }

    // trap 3: negated on construction, linear in the top half
    AbSpatialVector Z0 = asmSVec(-linear0, -angular0);
    AbSpatialVector Z1 = asmSVec(-linear1, -angular1);

    for (uint i = linkID1; i != commonLink;
         i = abU32(linkBase + uint64_t(i) * PXG_AB_LINK_SIZE, PXG_AB_LINK_mParents, lane))
    {
        const uint64_t tlink = linkBase + uint64_t(i) * PXG_AB_LINK_SIZE;
        const uint jointOffset = abU32(tlink, PXG_AB_LINK_mJointOffset, lane);
        const uint dofCount = abU8(tlink, PXG_AB_LINK_mDofs, lane);
        const float3 c2p = float3(abF32(tlink, PXG_AB_LINK_mRw_x, lane),
                                  abF32(tlink, PXG_AB_LINK_mRw_y, lane),
                                  abF32(tlink, PXG_AB_LINK_mRw_z, lane));

        // trap 5: the joint impulse lands only on linkID1's own inbound joint
        const bool applyHere = (linkID1 == i) && hasJi1;

        // trap 4: the PRE-propagation Z accumulates
        svAddSV(tlink, PXG_AB_LINK_mSolverSpatialImpulse, lane, Z1);

        Z1 = svPropagateImpulseDeferred(c2p, dofBlocks, jointOffset, dofCount, Z1, lane,
                                        jointImpulse1, 1.0f, applyHere);
    }

    for (uint i = linkID0; i != commonLink;
         i = abU32(linkBase + uint64_t(i) * PXG_AB_LINK_SIZE, PXG_AB_LINK_mParents, lane))
    {
        const uint64_t tlink = linkBase + uint64_t(i) * PXG_AB_LINK_SIZE;
        const uint jointOffset = abU32(tlink, PXG_AB_LINK_mJointOffset, lane);
        const uint dofCount = abU8(tlink, PXG_AB_LINK_mDofs, lane);
        const float3 c2p = float3(abF32(tlink, PXG_AB_LINK_mRw_x, lane),
                                  abF32(tlink, PXG_AB_LINK_mRw_y, lane),
                                  abF32(tlink, PXG_AB_LINK_mRw_z, lane));

        const bool applyHere = (linkID0 == i) && hasJi0;

        svAddSV(tlink, PXG_AB_LINK_mSolverSpatialImpulse, lane, Z0);

        Z0 = svPropagateImpulseDeferred(c2p, dofBlocks, jointOffset, dofCount, Z0, lane,
                                        jointImpulse0, 1.0f, applyHere);
    }

    AbSpatialVector ZCommon = asmSVecAdd(Z0, Z1);

    // trap 5 continued: if either link IS the common link, its joint impulse
    // has not been applied yet — accumulate from both ends
    float jiCommon[3] = { 0.0f, 0.0f, 0.0f };
    bool hasJiCommon = false;
    if (((linkID0 == commonLink) && hasJi0) || ((linkID1 == commonLink) && hasJi1))
    {
        hasJiCommon = true;
        const uint dofCountAtCommon =
            abU8(linkBase + uint64_t(commonLink) * PXG_AB_LINK_SIZE, PXG_AB_LINK_mDofs, lane);
        if ((linkID0 == commonLink) && hasJi0)
        {
            for (uint i = 0u; i < 3u; ++i)
                if (i < dofCountAtCommon) jiCommon[i] += jointImpulse0[i];
        }
        if ((linkID1 == commonLink) && hasJi1)
        {
            for (uint i = 0u; i < 3u; ++i)
                if (i < dofCountAtCommon) jiCommon[i] += jointImpulse1[i];
        }
    }

    // trap 6: does not run when commonLink == 0
    for (uint i = commonLink; i != 0u;
         i = abU32(linkBase + uint64_t(i) * PXG_AB_LINK_SIZE, PXG_AB_LINK_mParents, lane))
    {
        const uint64_t tlink = linkBase + uint64_t(i) * PXG_AB_LINK_SIZE;
        const uint jointOffset = abU32(tlink, PXG_AB_LINK_mJointOffset, lane);
        const uint dofCount = abU8(tlink, PXG_AB_LINK_mDofs, lane);
        const float3 c2p = float3(abF32(tlink, PXG_AB_LINK_mRw_x, lane),
                                  abF32(tlink, PXG_AB_LINK_mRw_y, lane),
                                  abF32(tlink, PXG_AB_LINK_mRw_z, lane));

        const bool applyHere = (commonLink == i) && hasJiCommon;

        svAddSV(tlink, PXG_AB_LINK_mSolverSpatialImpulse, lane, ZCommon);

        ZCommon = svPropagateImpulseDeferred(c2p, dofBlocks, jointOffset, dofCount, ZCommon,
                                             lane, jiCommon, 1.0f, applyHere);
    }

    svAddSV(artiBlock, PXG_AB_mRootDeferredZ, lane, ZCommon);
}

// ---------------------------------------------------------------------------
// fixture entry points: one articulation per lane, one warp per group; block
// bases handed in directly
// ---------------------------------------------------------------------------

struct SvParams
{
    uint64_t artiBlocks;   // @0
    uint64_t linkBlocks;   // @8
    uint64_t dofBlocks;    // @16
    uint64_t bitFields;    // @24
    uint64_t scratch;      // @32  svGetVelTest: out (vel 6 + speeds 3) per case
    uint nbArticulations;  // @40
    uint maxLinks;         // @44
    uint maxDofs;          // @48
    uint caseCount;        // @52  number of (linkID | flags) cases
    uint64_t caseTable;    // @56  u32 per case: linkID | (wantSpeeds<<8) | (fixBase<<9)
    // svApplyImpTest: caseTable u32 pairs are packed differently, see fixture
};

[[vk::push_constant]]
ConstantBuffer<SvParams> gSv;

[shader("compute")]
[numthreads(32, 1, 1)]
void svGetVelTest(uint3 groupId : SV_GroupID, uint3 groupThreadId : SV_GroupThreadID)
{
    const uint w = groupId.x;
    const uint lane = groupThreadId.x;
    const uint idx = w * 32u + lane;
    if (idx >= gSv.nbArticulations)
        return;

    const uint wordSize = (gSv.maxLinks + 63u) / 64u;
    const uint64_t artiBlock = gSv.artiBlocks + uint64_t(w) * PXG_AB_BLOCK_SIZE;
    const uint64_t linkBase = gSv.linkBlocks
                            + uint64_t(w) * uint64_t(gSv.maxLinks) * PXG_AB_LINK_SIZE;
    const uint64_t dofBase = gSv.dofBlocks
                           + uint64_t(w) * uint64_t(gSv.maxDofs) * PXG_AB_DOF_SIZE;
    const uint64_t bitFields = gSv.bitFields + uint64_t(w) * uint64_t(gSv.maxLinks)
                             * uint64_t(wordSize) * PXG_AB_BITFIELD_STACK_SIZE;

    for (uint c = 0u; c < gSv.caseCount; ++c)
    {
        const uint spec = (Ptr<uint>(gSv.caseTable))[c];
        const uint linkID = spec & 0xFFu;
        const bool wantSpeeds = (spec & 0x100u) != 0u;
        const bool fixBase = (spec & 0x200u) != 0u;

        float speeds[3] = { -777.0f, -777.0f, -777.0f };
        const AbSpatialVector v = svGetVelocity(artiBlock, linkBase, dofBase, bitFields,
                                                wordSize, linkID, fixBase, lane,
                                                speeds, wantSpeeds);

        // scratch layout: [case][articulation][9 floats]
        Ptr<float> outp = Ptr<float>(gSv.scratch
                                     + uint64_t(c * gSv.nbArticulations + idx) * 36u);
        outp[0] = v.top.x; outp[1] = v.top.y; outp[2] = v.top.z;
        outp[3] = v.bottom.x; outp[4] = v.bottom.y; outp[5] = v.bottom.z;
        outp[6] = speeds[0]; outp[7] = speeds[1]; outp[8] = speeds[2];
    }
}

[shader("compute")]
[numthreads(32, 1, 1)]
void svApplyImpTest(uint3 groupId : SV_GroupID, uint3 groupThreadId : SV_GroupThreadID)
{
    const uint w = groupId.x;
    const uint lane = groupThreadId.x;
    const uint idx = w * 32u + lane;
    if (idx >= gSv.nbArticulations)
        return;

    const uint wordSize = (gSv.maxLinks + 63u) / 64u;
    const uint64_t artiBlock = gSv.artiBlocks + uint64_t(w) * PXG_AB_BLOCK_SIZE;
    const uint64_t linkBase = gSv.linkBlocks
                            + uint64_t(w) * uint64_t(gSv.maxLinks) * PXG_AB_LINK_SIZE;
    const uint64_t dofBase = gSv.dofBlocks
                           + uint64_t(w) * uint64_t(gSv.maxDofs) * PXG_AB_DOF_SIZE;
    const uint64_t bitFields = gSv.bitFields + uint64_t(w) * uint64_t(gSv.maxLinks)
                             * uint64_t(wordSize) * PXG_AB_BITFIELD_STACK_SIZE;

    for (uint c = 0u; c < gSv.caseCount; ++c)
    {
        // per case: {linkID0 | hasJi0<<8, linkID1 | hasJi1<<8, then 18 floats
        // in the float table: linear0, angular0, ji0, linear1, angular1, ji1}
        const uint spec0 = (Ptr<uint>(gSv.caseTable))[c * 2u + 0u];
        const uint spec1 = (Ptr<uint>(gSv.caseTable))[c * 2u + 1u];
        const uint linkID0 = spec0 & 0xFFu;
        const bool hasJi0 = (spec0 & 0x100u) != 0u;
        const uint linkID1 = spec1 & 0xFFu;
        const bool hasJi1 = (spec1 & 0x100u) != 0u;

        Ptr<float> f = Ptr<float>(gSv.scratch + uint64_t(c) * 72u);
        const float3 linear0 = float3(f[0], f[1], f[2]);
        const float3 angular0 = float3(f[3], f[4], f[5]);
        float ji0[3] = { f[6], f[7], f[8] };
        const float3 linear1 = float3(f[9], f[10], f[11]);
        const float3 angular1 = float3(f[12], f[13], f[14]);
        float ji1[3] = { f[15], f[16], f[17] };

        svApplyImpulses(artiBlock, linkBase, dofBase, bitFields, wordSize,
                        linkID0, linear0 * (1.0f + float(idx) * 0.03125f),
                        angular0, ji0, hasJi0,
                        linkID1, linear1, angular1 * (1.0f - float(idx) * 0.015625f),
                        ji1, hasJi1, lane);
    }
}
