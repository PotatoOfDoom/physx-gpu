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

public uint svHighestBit64(uint64_t v)
{
    const uint hi = uint(v >> 32);
    return hi != 0u ? 32u + firstbithigh(hi) : firstbithigh(uint(v));
}

public uint svLowestBit64(uint64_t v)
{
    const uint lo = uint(v);
    return lo != 0u ? firstbitlow(lo) : 32u + firstbitlow(uint(v >> 32));
}

public AbSpatialVector svLoadSV(uint64_t base, uint fieldOffset, uint lane)
{
    float3 top, bottom;
    abLoadSpatialVector(base, fieldOffset, lane, top, bottom);
    return asmSVec(top, bottom);
}

public void svAddSV(uint64_t base, uint fieldOffset, uint lane, AbSpatialVector v)
{
    const AbSpatialVector cur = svLoadSV(base, fieldOffset, lane);
    const AbSpatialVector sum = asmSVecAdd(cur, v);
    abStoreSpatialVector(base, fieldOffset, lane, sum.top, sum.bottom);
}

// the per-link dof arrays every propagation hop needs
public void svLoadDofArrays(uint64_t dofBlocks, uint jointOffset, uint dofCount, uint lane,
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
public AbSpatialVector svPropagateAccelDeferred(float3 parentToChild, uint64_t dofBlocks,
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
public AbSpatialVector svPropagateImpulseDeferred(float3 childToParent, uint64_t dofBlocks,
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

// ===========================================================================
// P8.98b: the three per-iteration solve bodies (spatial :2317, fixed :2435,
// mimic :2693) plus the TGS length-refresh DFS (:2015).
//
// TRAPS (each carries a mutant):
//  11. the spatial parent velocity is computed ONCE (from the LAST
//      constraint's mLink0) and then updated as a SCALAR via
//      `parentV += mDeltaVA * -deltaF`; the constraint loop runs BACKWARDS.
//  12. `parentV` seeds from constraint[0]'s row0 but `parentID` from the
//      LAST constraint's mLink0 (the engine asserts they agree).
//  13. the fixed DFS keeps TWO stacks: `bitStack` (consumed) and
//      `totalStack` (the union, replayed highest-first in pass two).
//  14. `scale = count ? 1/count : 0` divides by the popcount of totalStack.
//  15. the fixed pass-two impulse is `row1 * -deltaF + cImpulse`, but
//      mSolverSpatialImpulse accumulates cImpulse ALONE (impulse - YInt).
//  16. the final fixed Z is `sLink.mScratchImpulse + row0(firstChild's
//      constraint) * rootImp`, walked from sLinkIndex to the root.
//  17. mScratchImpulse is re-zeroed over nbLinks AFTER the tendon loop,
//      only when numTendons > 0.
//  18. the fixed DFS reads the LINK parent (cLink.mParents), not the
//      tendon-joint parent; the CUDA's `parent = child` store is dead and
//      the port drops it.
//  19. the TGS refresh writes ONLY leaf constraints' mAccumulatedLength,
//      in the same DFS order the setup used.
//  20. mimic: the erp/cfm three-way branch — compliant only when
//      naturalFrequency > 0 AND dampingRatio > 0 AND !velocityIteration.
//  21. the dead isTGS parameters of the fixed and mimic bodies are dropped
//      (the P8.87 dead-parameter arrangement).
// ===========================================================================

static const uint SV_FLAG_FIX_BASE = 1u;   // PxArticulationFlag::eFIX_BASE

uint svPopc64(uint64_t v)
{
    return countbits(uint(v)) + countbits(uint(v >> 32));
}

PxgTransform28 svLoadPose(uint64_t link, uint fieldOffset, uint lane)
{
    const float4 q = abF4(link, fieldOffset + PXG_AB_STB_Q, lane);
    const float4 p = abF4(link, fieldOffset + PXG_AB_STB_P, lane);
    return makeTransform28(q, float3(p.x, p.y, p.z));
}

// updateSolveInternalTendonConstraintsTGS (internalConstraints2.cu:2015)
public void svUpdateTendonConstraintsTGS(uint64_t artiBlock, uint64_t linkBase,
                                         uint64_t tendonBase, uint64_t constraintBase,
                                         uint64_t attachmentBase, uint maxAttachments,
                                         uint lane)
{
    const uint numTendons = abU32(artiBlock, PXG_AB_mNumSpatialTendons, lane);

    float accumLength[PXG_DY_ARTICULATION_TENDON_MAX_SIZE];
    for (uint k = 0u; k < PXG_DY_ARTICULATION_TENDON_MAX_SIZE; ++k)
        accumLength[k] = 0.0f;

    for (uint i = 0u; i < numTendons; ++i)
    {
        const uint64_t tendon = tendonBase + uint64_t(i) * PXG_AB_ST_SIZE;
        const float offset = abF32(tendon, PXG_AB_ST_mOffset, lane);

        const uint64_t attachmentBlock = attachmentBase
                                       + uint64_t(i) * uint64_t(maxAttachments) * PXG_AB_AT_SIZE;
        const uint64_t constraintBlock = constraintBase
                                       + uint64_t(i) * uint64_t(maxAttachments) * PXG_AB_ITC_SIZE;

        uint64_t bitStack = abU64(attachmentBlock, PXG_AB_AT_mChildrens, lane);
        const float coefficient = abF32(attachmentBlock, PXG_AB_AT_mCoefficient, lane);

        uint stackCount = svPopc64(bitStack);
        uint parent = 0u;
        uint numConstraints = 0u;
        accumLength[parent] = offset * coefficient;

        uint child = svHighestBit64(bitStack);

        while (stackCount != 0u)
        {
            stackCount--;

            const uint64_t attachment = attachmentBlock + uint64_t(child) * PXG_AB_AT_SIZE;
            const uint64_t pAttachment = attachmentBlock + uint64_t(parent) * PXG_AB_AT_SIZE;

            const uint linkInd = abU16(attachment, PXG_AB_AT_mLinkIndex, lane);
            const uint64_t cLink = linkBase + uint64_t(linkInd) * PXG_AB_LINK_SIZE;
            const uint pLinkInd = abU16(pAttachment, PXG_AB_AT_mLinkIndex, lane);
            const uint64_t pLink = linkBase + uint64_t(pLinkInd) * PXG_AB_LINK_SIZE;

            const PxgTransform28 cB2w = svLoadPose(cLink, PXG_AB_LINK_mAccumulatedPose, lane);
            const PxgTransform28 pB2w = svLoadPose(pLink, PXG_AB_LINK_mAccumulatedPose, lane);

            const float3 rb = quatRotate(cB2w.q(), abF3(attachment, PXG_AB_AT_mRelativeOffset, lane));
            const float3 ra = quatRotate(pB2w.q(), abF3(pAttachment, PXG_AB_AT_mRelativeOffset, lane));

            const float3 cAttachPoint = cB2w.p() + rb;
            const float3 pAttachPoint = pB2w.p() + ra;

            const float3 dif = pAttachPoint - cAttachPoint;
            const float distanceSq = asmDot(dif, dif);
            const float distance = sqrt(distanceSq);

            const float u = distance * abF32(attachment, PXG_AB_AT_mCoefficient, lane)
                          + accumLength[parent];

            const uint64_t children = abU64(attachment, PXG_AB_AT_mChildrens, lane);

            if (children != 0uL)
            {
                stackCount += svPopc64(children);
                accumLength[child] = u;
            }
            else
            {
                // trap 19: only leaves store, in setup DFS order
                const uint64_t constraint = constraintBlock
                                          + uint64_t(numConstraints) * PXG_AB_ITC_SIZE;
                numConstraints++;
                abStoreF32(constraint, PXG_AB_ITC_mAccumulatedLength, lane, u);
            }

            if (stackCount > 0u)
            {
                bitStack &= (~(uint64_t(1) << child));
                bitStack |= children;
                child = svHighestBit64(bitStack);
                parent = abU32(attachmentBlock + uint64_t(child) * PXG_AB_AT_SIZE,
                               PXG_AB_AT_mParents, lane);
            }
        }
    }
}

// solveInternalSpatialConstraints (internalConstraints2.cu:2317)
public void svSolveSpatialConstraints(uint64_t artiBlock, uint64_t linkBase,
                                      uint64_t dofBlocks, uint64_t bitFields, uint wordSize,
                                      uint64_t tendonBase, uint64_t constraintBase,
                                      uint64_t attachmentBase, uint maxAttachments,
                                      bool isTGS, uint lane)
{
    const bool fixBase = (abU32(artiBlock, PXG_AB_mFlags, lane) & SV_FLAG_FIX_BASE) != 0u;

    if (isTGS)
        svUpdateTendonConstraintsTGS(artiBlock, linkBase, tendonBase, constraintBase,
                                     attachmentBase, maxAttachments, lane);

    const uint numTendons = abU32(artiBlock, PXG_AB_mNumSpatialTendons, lane);

    for (uint i = 0u; i < numTendons; ++i)
    {
        const uint64_t tendon = tendonBase + uint64_t(i) * PXG_AB_ST_SIZE;
        const uint64_t constraintBlock = constraintBase
                                       + uint64_t(i) * uint64_t(maxAttachments) * PXG_AB_ITC_SIZE;

        const uint numConstraints = abU32(tendon, PXG_AB_ST_mNumConstraints, lane);

        if (numConstraints > 0u)
        {
            // trap 12: parentID from the LAST constraint, parentV from
            // constraint 0's row0
            const uint64_t lastConstraint = constraintBlock
                                          + uint64_t(numConstraints - 1u) * PXG_AB_ITC_SIZE;
            const uint parentID = abU32(lastConstraint, PXG_AB_ITC_mLink0, lane);

            float noSpeeds[3] = { 0.0f, 0.0f, 0.0f };
            const AbSpatialVector parentVel = svGetVelocity(artiBlock, linkBase, dofBlocks,
                                                            bitFields, wordSize, parentID,
                                                            fixBase, lane, noSpeeds, false);

            float parentV = asmInnerProduct(svLoadSV(constraintBlock, PXG_AB_ITC_mRow0, lane),
                                            parentVel);

            // trap 11: BACKWARDS over the constraints
            for (int j = int(numConstraints) - 1; j >= 0; --j)
            {
                const uint64_t constraint = constraintBlock + uint64_t(j) * PXG_AB_ITC_SIZE;

                const uint childID = abU32(constraint, PXG_AB_ITC_mLink1, lane);

                const AbSpatialVector childVel = svGetVelocity(artiBlock, linkBase, dofBlocks,
                                                               bitFields, wordSize, childID,
                                                               fixBase, lane, noSpeeds, false);

                const float accumLength = abF32(constraint, PXG_AB_ITC_mAccumulatedLength, lane);
                const float error = abF32(constraint, PXG_AB_ITC_mRestDistance, lane)
                                  - accumLength;

                float error2 = 0.0f;
                const float lowLimit = abF32(constraint, PXG_AB_ITC_mLowLimit, lane);
                const float highLimit = abF32(constraint, PXG_AB_ITC_mHighLimit, lane);
                if (accumLength > highLimit)
                    error2 = highLimit - accumLength;
                else if (accumLength < lowLimit)
                    error2 = lowLimit - accumLength;

                const AbSpatialVector row1 = svLoadSV(constraint, PXG_AB_ITC_mRow1, lane);
                const AbSpatialVector row0 = svLoadSV(constraint, PXG_AB_ITC_mRow0, lane);

                const float jointV = asmInnerProduct(row1, childVel) - parentV;

                const float velMultiplier = abF32(constraint, PXG_AB_ITC_mVelMultiplier, lane);
                const float biasCoefficient = abF32(constraint, PXG_AB_ITC_mBiasCoefficient, lane);
                const float appliedForce = abF32(constraint, PXG_AB_ITC_mAppliedForce, lane);
                const float impulseMultiplier =
                    abF32(constraint, PXG_AB_ITC_mImpulseMultiplier, lane);

                const float limitBiasCoefficient =
                    abF32(constraint, PXG_AB_ITC_mLimitBiasCoefficient, lane);
                const float limitAppliedForce =
                    abF32(constraint, PXG_AB_ITC_mLimitAppliedForce, lane);
                const float limitImpulseMultiplier =
                    abF32(constraint, PXG_AB_ITC_mLimitImpulseMultiplier, lane);

                const float unclampedForce = jointV * velMultiplier + error * biasCoefficient
                                           + appliedForce * impulseMultiplier;
                const float unclampedForce2 = (error2 * limitBiasCoefficient)
                                            + limitAppliedForce * limitImpulseMultiplier;

                const float deltaF = (unclampedForce - appliedForce)
                                   + (unclampedForce2 - limitAppliedForce);

                abStoreF32(constraint, PXG_AB_ITC_mAppliedForce, lane, unclampedForce);
                abStoreF32(constraint, PXG_AB_ITC_mLimitAppliedForce, lane, unclampedForce2);

                parentV += abF32(constraint, PXG_AB_ITC_mDeltaVA, lane) * -deltaF;

                if (deltaF != 0.0f)
                {
                    const AbSpatialVector i0 = asmSVecScale(row0, -deltaF);
                    const AbSpatialVector i1 = asmSVecScale(row1, deltaF);

                    float noJi[3] = { 0.0f, 0.0f, 0.0f };
                    svApplyImpulses(artiBlock, linkBase, dofBlocks, bitFields, wordSize,
                                    parentID, i0.top, i0.bottom, noJi, false,
                                    childID, i1.top, i1.bottom, noJi, false, lane);
                }
            }
        }
    }
}

// solveInternalFixedConstraints (internalConstraints2.cu:2435).  The CUDA's
// isTGS parameter is dead and dropped (trap 21); its `parent = child` store
// is dead and dropped (trap 18).
public void svSolveFixedConstraints(uint64_t artiBlock, uint64_t linkBase,
                                    uint64_t dofBlocks, uint64_t bitFields, uint wordSize,
                                    uint64_t ftBase, uint64_t constraintBase,
                                    uint64_t tjBase, uint maxTendonJoints, uint lane)
{
    const bool fixBase = (abU32(artiBlock, PXG_AB_mFlags, lane) & SV_FLAG_FIX_BASE) != 0u;
    const uint numTendons = abU32(artiBlock, PXG_AB_mNumFixedTendons, lane);

    for (uint i = 0u; i < numTendons; ++i)
    {
        const uint64_t tendon = ftBase + uint64_t(i) * PXG_AB_FT_SIZE;

        const float lowLimit = abF32(tendon, PXG_AB_FT_mLowLimit, lane);
        const float highLimit = abF32(tendon, PXG_AB_FT_mHighLimit, lane);

        const uint64_t tendonJointBlock = tjBase
                                        + uint64_t(i) * uint64_t(maxTendonJoints) * PXG_AB_TJ_SIZE;
        const uint64_t constraintBlock = constraintBase
                                       + uint64_t(i) * uint64_t(maxTendonJoints) * PXG_AB_ITC_SIZE;

        uint64_t bitStack = abU64(tendonJointBlock, PXG_AB_TJ_mChildrens, lane);
        uint stackCount = svPopc64(bitStack);

        const uint sLinkIndex = abU16(tendonJointBlock, PXG_AB_TJ_mLinkIndex, lane);
        const uint64_t sLink = linkBase + uint64_t(sLinkIndex) * PXG_AB_LINK_SIZE;

        float noSpeeds[3] = { 0.0f, 0.0f, 0.0f };
        const AbSpatialVector parentVel = svGetVelocity(artiBlock, linkBase, dofBlocks,
                                                        bitFields, wordSize, sLinkIndex,
                                                        fixBase, lane, noSpeeds, false);
        const AbSpatialVector velA = svLoadSV(sLink, PXG_AB_LINK_mMotionVelocity, lane);

        const AbSpatialVector delta = asmSVecSub(parentVel, velA);
        abStoreSpatialVector(sLink, PXG_AB_LINK_mScratchImpulse, lane,
                             float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f));
        abStoreSpatialVector(sLink, PXG_AB_LINK_mScratchDeltaV, lane, delta.top, delta.bottom);

        float rootImp = 0.0f;
        uint64_t totalStack = 0uL;

        float error = 0.0f;
        float velocity = 0.0f;

        const uint firstChild = svHighestBit64(bitStack);

        totalStack |= bitStack;

        // pass one: the DFS, accumulating error and velocity (trap 13)
        while (stackCount != 0u)
        {
            const uint child = svHighestBit64(bitStack);
            bitStack &= (~(uint64_t(1) << child));
            stackCount--;

            const uint64_t tj = tendonJointBlock + uint64_t(child) * PXG_AB_TJ_SIZE;

            const uint cLinkInd = abU16(tj, PXG_AB_TJ_mLinkIndex, lane);
            const uint64_t cLink = linkBase + uint64_t(cLinkInd) * PXG_AB_LINK_SIZE;
            // trap 18: the LINK parent, not the tendon-joint parent
            const uint pLinkInd = abU32(cLink, PXG_AB_LINK_mParents, lane);
            const uint64_t pLink = linkBase + uint64_t(pLinkInd) * PXG_AB_LINK_SIZE;
            const uint jointOffset = abU32(cLink, PXG_AB_LINK_mJointOffset, lane);
            const uint dofCount = abU8(cLink, PXG_AB_LINK_mDofs, lane);

            const float3 c2p = float3(abF32(cLink, PXG_AB_LINK_mRw_x, lane),
                                      abF32(cLink, PXG_AB_LINK_mRw_y, lane),
                                      abF32(cLink, PXG_AB_LINK_mRw_z, lane));

            const uint constraintId = abU32(tj, PXG_AB_TJ_mConstraintId, lane);
            const uint64_t constraint = constraintBlock + uint64_t(constraintId) * PXG_AB_ITC_SIZE;

            const uint tjAxis = abU32(tj, PXG_AB_TJ_mAxis, lane);
            const float coefficient = abF32(tj, PXG_AB_TJ_mCoefficient, lane);

            const uint dofIndex = abU8(cLink, PXG_AB_LINK_mInvDofIds + tjAxis * 32u, lane);
            const uint64_t dof = dofBlocks + uint64_t(jointOffset + dofIndex) * PXG_AB_DOF_SIZE;

            const float jointPose = abF32(dof, PXG_AB_DOF_mJointPositions, lane);

            const AbSpatialVector parentDeltaV = svLoadSV(pLink, PXG_AB_LINK_mScratchDeltaV, lane);
            const AbSpatialVector parentV =
                asmSVecAdd(svLoadSV(pLink, PXG_AB_LINK_mMotionVelocity, lane), parentDeltaV);

            float noSp[3] = { 0.0f, 0.0f, 0.0f };
            const AbSpatialVector cDeltaV = svPropagateAccelDeferred(c2p, dofBlocks, jointOffset,
                                                                     dofCount, parentDeltaV,
                                                                     lane, noSp, false);

            abStoreSpatialVector(cLink, PXG_AB_LINK_mScratchDeltaV, lane,
                                 cDeltaV.top, cDeltaV.bottom);

            const AbSpatialVector velB = svLoadSV(cLink, PXG_AB_LINK_mMotionVelocity, lane);
            const AbSpatialVector childVel = asmSVecAdd(velB, cDeltaV);

            abStoreSpatialVector(cLink, PXG_AB_LINK_mScratchImpulse, lane,
                                 float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f));

            const uint64_t children = abU64(tj, PXG_AB_TJ_mChildrens, lane);

            const AbSpatialVector row0 = svLoadSV(constraint, PXG_AB_ITC_mRow0, lane);
            const AbSpatialVector row1 = svLoadSV(constraint, PXG_AB_ITC_mRow1, lane);
            const float jointV = asmInnerProduct(row1, childVel)
                               - asmInnerProduct(row0, parentV);

            error += jointPose * coefficient;
            velocity += jointV * coefficient;

            if (children != 0uL)
            {
                stackCount += svPopc64(children);
                bitStack |= children;
                totalStack |= children;
            }
        }

        const uint count = svPopc64(totalStack);
        const float scale = count != 0u ? 1.0f / float(count) : 0.0f;   // trap 14

        const float length = error + abF32(tendon, PXG_AB_FT_mOffset, lane);

        float limitError = 0.0f;
        if (length < lowLimit)
            limitError = length - lowLimit;
        else if (length > highLimit)
            limitError = length - highLimit;

        error = (length - abF32(tendon, PXG_AB_FT_mRestLength, lane)) * scale;
        limitError *= scale;
        velocity *= scale;

        // pass two: replay the union highest-first (trap 13)
        while (totalStack != 0uL)
        {
            const uint child = svHighestBit64(totalStack);
            totalStack &= (~(uint64_t(1) << child));

            const uint64_t tj = tendonJointBlock + uint64_t(child) * PXG_AB_TJ_SIZE;

            const uint cLinkInd = abU16(tj, PXG_AB_TJ_mLinkIndex, lane);
            const uint64_t cLink = linkBase + uint64_t(cLinkInd) * PXG_AB_LINK_SIZE;
            const uint pLinkInd = abU32(cLink, PXG_AB_LINK_mParents, lane);
            const uint64_t pLink = linkBase + uint64_t(pLinkInd) * PXG_AB_LINK_SIZE;
            const uint jointOffset = abU32(cLink, PXG_AB_LINK_mJointOffset, lane);
            const uint dofCount = abU8(cLink, PXG_AB_LINK_mDofs, lane);

            const float3 c2p = float3(abF32(cLink, PXG_AB_LINK_mRw_x, lane),
                                      abF32(cLink, PXG_AB_LINK_mRw_y, lane),
                                      abF32(cLink, PXG_AB_LINK_mRw_z, lane));

            const uint constraintId = abU32(tj, PXG_AB_TJ_mConstraintId, lane);
            const uint64_t constraint = constraintBlock + uint64_t(constraintId) * PXG_AB_ITC_SIZE;

            const float recipCoefficient = abF32(tj, PXG_AB_TJ_mRecipCoefficient, lane);
            const float velMultiplier = abF32(constraint, PXG_AB_ITC_mVelMultiplier, lane);
            const float biasCoefficient = abF32(constraint, PXG_AB_ITC_mBiasCoefficient, lane);
            const float limitBiasCoefficient =
                abF32(constraint, PXG_AB_ITC_mLimitBiasCoefficient, lane);
            const float impulseMultiplier = abF32(constraint, PXG_AB_ITC_mImpulseMultiplier, lane);
            const float appliedForce = abF32(constraint, PXG_AB_ITC_mAppliedForce, lane);
            const float limitImpulseMultiplier =
                abF32(constraint, PXG_AB_ITC_mLimitImpulseMultiplier, lane);
            const float limitAppliedForce = abF32(constraint, PXG_AB_ITC_mLimitAppliedForce, lane);

            const float unclampedForce =
                ((velocity * velMultiplier + error * biasCoefficient) * recipCoefficient)
                + appliedForce * impulseMultiplier;
            const float unclampedForce2 = (limitError * limitBiasCoefficient * recipCoefficient)
                                        + limitAppliedForce * limitImpulseMultiplier;

            const float deltaF = ((unclampedForce - appliedForce)
                                + (unclampedForce2 - limitAppliedForce));

            abStoreF32(constraint, PXG_AB_ITC_mAppliedForce, lane, unclampedForce);
            abStoreF32(constraint, PXG_AB_ITC_mLimitAppliedForce, lane, unclampedForce2);

            rootImp += deltaF;

            const AbSpatialVector cImpulse = svLoadSV(cLink, PXG_AB_LINK_mScratchImpulse, lane);

            const AbSpatialVector impulse =
                asmSVecAdd(asmSVecScale(svLoadSV(constraint, PXG_AB_ITC_mRow1, lane), -deltaF),
                           cImpulse);

            // trap 15: cImpulse ALONE (impulse - YInt == cImpulse)
            svAddSV(cLink, PXG_AB_LINK_mSolverSpatialImpulse, lane, cImpulse);

            float noJi[3] = { 0.0f, 0.0f, 0.0f };
            const AbSpatialVector propagatedImpulse =
                svPropagateImpulseDeferred(c2p, dofBlocks, jointOffset, dofCount, impulse,
                                           lane, noJi, 1.0f, false);

            svAddSV(pLink, PXG_AB_LINK_mScratchImpulse, lane, propagatedImpulse);
        }

        // trap 16: the root walk from sLinkIndex
        const uint firstConstraintId =
            abU32(tendonJointBlock + uint64_t(firstChild) * PXG_AB_TJ_SIZE,
                  PXG_AB_TJ_mConstraintId, lane);
        const uint64_t firstConstraint = constraintBlock
                                       + uint64_t(firstConstraintId) * PXG_AB_ITC_SIZE;

        const AbSpatialVector propagatedImpulse = svLoadSV(sLink, PXG_AB_LINK_mScratchImpulse, lane);

        AbSpatialVector Z = asmSVecAdd(propagatedImpulse,
                                       asmSVecScale(svLoadSV(firstConstraint, PXG_AB_ITC_mRow0, lane),
                                                    rootImp));

        for (uint linkID = sLinkIndex; linkID != 0u;
             linkID = abU32(linkBase + uint64_t(linkID) * PXG_AB_LINK_SIZE,
                            PXG_AB_LINK_mParents, lane))
        {
            const uint64_t link = linkBase + uint64_t(linkID) * PXG_AB_LINK_SIZE;
            const uint jointOffset = abU32(link, PXG_AB_LINK_mJointOffset, lane);
            const uint dofCount = abU8(link, PXG_AB_LINK_mDofs, lane);

            const float3 c2p = float3(abF32(link, PXG_AB_LINK_mRw_x, lane),
                                      abF32(link, PXG_AB_LINK_mRw_y, lane),
                                      abF32(link, PXG_AB_LINK_mRw_z, lane));

            float noJi[3] = { 0.0f, 0.0f, 0.0f };
            Z = svPropagateImpulseDeferred(c2p, dofBlocks, jointOffset, dofCount, Z, lane,
                                           noJi, 1.0f, false);
        }

        svAddSV(artiBlock, PXG_AB_mRootDeferredZ, lane, Z);
    }

    // trap 17: the scratch cleanup, only when tendons exist
    if (numTendons > 0u)
    {
        const uint nbLinks = abU32(artiBlock, PXG_AB_mNumLinks, lane);
        for (uint i = 0u; i < nbLinks; ++i)
        {
            abStoreSpatialVector(linkBase + uint64_t(i) * PXG_AB_LINK_SIZE,
                                 PXG_AB_LINK_mScratchImpulse, lane,
                                 float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f));
        }
    }
}

// computeMimicJointImpulses (DyCpuGpuArticulation.h:637), transcribed
void svMimicJointImpulses(float biasCoefficient, float dt, float invDt,
                          float qA, float qB, float qADot, float qBDot,
                          float gearRatio, float offset,
                          float naturalFrequency, float dampingRatio,
                          float r, bool isVelocityIteration,
                          out float jointImpA, out float jointImpB)
{
    float erp = 0.0f;
    float cfm = 0.0f;
    if (naturalFrequency <= 0.0f || dampingRatio <= 0.0f || isVelocityIteration)
    {
        erp = isVelocityIteration ? 0.0f : biasCoefficient;
        cfm = 0.0f;
    }
    else
    {
        const float mu = naturalFrequency;
        const float zeta = dampingRatio;
        const float kp = mu * mu / r;
        const float kd = 2.0f * mu * zeta / r;
        cfm = 1.0f / (dt * (dt * kp + kd));
        erp = (dt * kp) / (dt * kp + kd);
    }

    const float C = qA + gearRatio * qB + offset;
    const float b = erp * C * invDt;
    const float Jv = qADot + gearRatio * qBDot;
    const float effectiveInertia = 1.0f / (r + cfm);
    const float lambda = -(b + Jv) * effectiveInertia;
    jointImpA = lambda;
    jointImpB = gearRatio * lambda;
}

// solveInternalMimicJointConstraints (internalConstraints2.cu:2693).  The
// CUDA's isTGS parameter is dead and dropped (trap 21).
public void svSolveMimicConstraints(uint64_t artiBlock, uint64_t linkBase,
                                    uint64_t dofBlocks, uint64_t bitFields, uint wordSize,
                                    uint64_t mimicBase,
                                    float biasCoefficient, float dt, float recipDt,
                                    bool isVelocityIteration, uint lane)
{
    const bool fixBase = (abU32(artiBlock, PXG_AB_mFlags, lane) & SV_FLAG_FIX_BASE) != 0u;
    const uint numMimicJoints = abU32(artiBlock, PXG_AB_mNumMimicJoints, lane);

    for (uint i = 0u; i < numMimicJoints; ++i)
    {
        const uint64_t mj = mimicBase + uint64_t(i) * PXG_AB_MJ_SIZE;
        const uint linkA = abU32(mj, PXG_AB_MJ_mLinkA, lane);
        const uint linkB = abU32(mj, PXG_AB_MJ_mLinkB, lane);
        const uint dofA = abU32(mj + PXG_AB_MJ_mInternalData, PXG_AB_IMJ_mDofA, lane);
        const uint dofB = abU32(mj + PXG_AB_MJ_mInternalData, PXG_AB_IMJ_mDofB, lane);
        const float recipEffectiveInertia =
            abF32(mj + PXG_AB_MJ_mInternalData, PXG_AB_IMJ_recipEffectiveInertia, lane);
        const float gearRatio = abF32(mj, PXG_AB_MJ_mGearRatio, lane);
        const float offset = abF32(mj, PXG_AB_MJ_mOffset, lane);
        const float naturalFrequency = abF32(mj, PXG_AB_MJ_mNaturalFrequency, lane);
        const float dampingRatio = abF32(mj, PXG_AB_MJ_mDampingRatio, lane);

        const uint jointOffsetA = abU32(linkBase + uint64_t(linkA) * PXG_AB_LINK_SIZE,
                                        PXG_AB_LINK_mJointOffset, lane);
        const uint jointOffsetB = abU32(linkBase + uint64_t(linkB) * PXG_AB_LINK_SIZE,
                                        PXG_AB_LINK_mJointOffset, lane);

        const float qA = abF32(dofBlocks + uint64_t(jointOffsetA + dofA) * PXG_AB_DOF_SIZE,
                               PXG_AB_DOF_mJointPositions, lane);
        const float qB = abF32(dofBlocks + uint64_t(jointOffsetB + dofB) * PXG_AB_DOF_SIZE,
                               PXG_AB_DOF_mJointPositions, lane);

        float jointDofSpeedsA[3] = { 0.0f, 0.0f, 0.0f };
        float jointDofSpeedsB[3] = { 0.0f, 0.0f, 0.0f };
        svGetVelocity(artiBlock, linkBase, dofBlocks, bitFields, wordSize, linkA, fixBase,
                      lane, jointDofSpeedsA, true);
        svGetVelocity(artiBlock, linkBase, dofBlocks, bitFields, wordSize, linkB, fixBase,
                      lane, jointDofSpeedsB, true);
        const float qADot = jointDofSpeedsA[dofA];
        const float qBDot = jointDofSpeedsB[dofB];

        float jointImpDofA, jointImpDofB;
        svMimicJointImpulses(biasCoefficient, dt, recipDt, qA, qB, qADot, qBDot,
                             gearRatio, offset, naturalFrequency, dampingRatio,
                             recipEffectiveInertia, isVelocityIteration,
                             jointImpDofA, jointImpDofB);

        float jointImpulseA[3] = { 0.0f, 0.0f, 0.0f };
        float jointImpulseB[3] = { 0.0f, 0.0f, 0.0f };
        jointImpulseA[dofA] = jointImpDofA;
        jointImpulseB[dofB] = jointImpDofB;

        const float3 zero = float3(0.0f, 0.0f, 0.0f);
        svApplyImpulses(artiBlock, linkBase, dofBlocks, bitFields, wordSize,
                        linkA, zero, zero, jointImpulseA, true,
                        linkB, zero, zero, jointImpulseB, true, lane);
    }
}

// ---------------------------------------------------------------------------
// fixture entry points for the three bodies
// ---------------------------------------------------------------------------

struct SvbParams
{
    uint64_t artiBlocks;      // @0
    uint64_t linkBlocks;      // @8
    uint64_t dofBlocks;       // @16
    uint64_t bitFields;       // @24
    uint64_t tendonBlocks;    // @32
    uint64_t constraintBlocks;// @40
    uint64_t elementBlocks;   // @48 attachments / tendon joints / mimic joints
    float biasCoefficient;    // @56
    float dt;                 // @60
    float recipDt;            // @64
    uint isTGS;               // @68
    uint isVelIter;           // @72
    uint nbArticulations;     // @76
    uint maxLinks;            // @80
    uint maxDofs;             // @84
    uint maxTendons;          // @88
    uint maxElements;         // @92 -> 96
};

[[vk::push_constant]]
ConstantBuffer<SvbParams> gSvb;

[shader("compute")]
[numthreads(32, 1, 1)]
void svSolveSpatialTest(uint3 groupId : SV_GroupID, uint3 groupThreadId : SV_GroupThreadID)
{
    const uint w = groupId.x;
    const uint lane = groupThreadId.x;
    if (w * 32u + lane >= gSvb.nbArticulations)
        return;

    const uint wordSize = (gSvb.maxLinks + 63u) / 64u;
    svSolveSpatialConstraints(
        gSvb.artiBlocks + uint64_t(w) * PXG_AB_BLOCK_SIZE,
        gSvb.linkBlocks + uint64_t(w) * uint64_t(gSvb.maxLinks) * PXG_AB_LINK_SIZE,
        gSvb.dofBlocks + uint64_t(w) * uint64_t(gSvb.maxDofs) * PXG_AB_DOF_SIZE,
        gSvb.bitFields + uint64_t(w) * uint64_t(gSvb.maxLinks) * uint64_t(wordSize)
            * PXG_AB_BITFIELD_STACK_SIZE,
        wordSize,
        gSvb.tendonBlocks + uint64_t(w) * uint64_t(gSvb.maxTendons) * PXG_AB_ST_SIZE,
        gSvb.constraintBlocks + uint64_t(w) * uint64_t(gSvb.maxTendons)
            * uint64_t(gSvb.maxElements) * PXG_AB_ITC_SIZE,
        gSvb.elementBlocks + uint64_t(w) * uint64_t(gSvb.maxTendons)
            * uint64_t(gSvb.maxElements) * PXG_AB_AT_SIZE,
        gSvb.maxElements, gSvb.isTGS != 0u, lane);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void svSolveFixedTest(uint3 groupId : SV_GroupID, uint3 groupThreadId : SV_GroupThreadID)
{
    const uint w = groupId.x;
    const uint lane = groupThreadId.x;
    if (w * 32u + lane >= gSvb.nbArticulations)
        return;

    const uint wordSize = (gSvb.maxLinks + 63u) / 64u;
    svSolveFixedConstraints(
        gSvb.artiBlocks + uint64_t(w) * PXG_AB_BLOCK_SIZE,
        gSvb.linkBlocks + uint64_t(w) * uint64_t(gSvb.maxLinks) * PXG_AB_LINK_SIZE,
        gSvb.dofBlocks + uint64_t(w) * uint64_t(gSvb.maxDofs) * PXG_AB_DOF_SIZE,
        gSvb.bitFields + uint64_t(w) * uint64_t(gSvb.maxLinks) * uint64_t(wordSize)
            * PXG_AB_BITFIELD_STACK_SIZE,
        wordSize,
        gSvb.tendonBlocks + uint64_t(w) * uint64_t(gSvb.maxTendons) * PXG_AB_FT_SIZE,
        gSvb.constraintBlocks + uint64_t(w) * uint64_t(gSvb.maxTendons)
            * uint64_t(gSvb.maxElements) * PXG_AB_ITC_SIZE,
        gSvb.elementBlocks + uint64_t(w) * uint64_t(gSvb.maxTendons)
            * uint64_t(gSvb.maxElements) * PXG_AB_TJ_SIZE,
        gSvb.maxElements, lane);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void svSolveMimicTest(uint3 groupId : SV_GroupID, uint3 groupThreadId : SV_GroupThreadID)
{
    const uint w = groupId.x;
    const uint lane = groupThreadId.x;
    if (w * 32u + lane >= gSvb.nbArticulations)
        return;

    const uint wordSize = (gSvb.maxLinks + 63u) / 64u;
    svSolveMimicConstraints(
        gSvb.artiBlocks + uint64_t(w) * PXG_AB_BLOCK_SIZE,
        gSvb.linkBlocks + uint64_t(w) * uint64_t(gSvb.maxLinks) * PXG_AB_LINK_SIZE,
        gSvb.dofBlocks + uint64_t(w) * uint64_t(gSvb.maxDofs) * PXG_AB_DOF_SIZE,
        gSvb.bitFields + uint64_t(w) * uint64_t(gSvb.maxLinks) * uint64_t(wordSize)
            * PXG_AB_BITFIELD_STACK_SIZE,
        wordSize,
        gSvb.elementBlocks + uint64_t(w) * uint64_t(gSvb.maxTendons) * PXG_AB_MJ_SIZE,
        gSvb.biasCoefficient, gSvb.dt, gSvb.recipDt, gSvb.isVelIter != 0u, lane);
}
