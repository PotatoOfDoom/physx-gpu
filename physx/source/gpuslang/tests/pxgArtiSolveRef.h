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

// pxgArtiSolveRef.h
//
// The solve-chain deferred-impulse mirrors (P8.98a): pxcFsGetVelocity and
// pxcFsApplyImpulses (internalConstraints2.cu:2116 / :2190) plus the two
// articulationDynamic.cuh template-instance hops they compose.  Hoisted from
// testArtiSolveVelImp.cpp so the solve-body fixtures (P8.98b+) are USERS, not
// second transcriptions.
//
// Only the STRUCTURE is transcribed; every operation inside is an engine
// PX_CUDA_CALLABLE (loadSpatialVector / loadSpatialMatrix /
// Dy::SpatialMatrix::operator* / translateSpatialVector).  Two order facts are
// load-bearing and each cost a first-run red before they were spelled:
//   * the impulse hop seeds temp from Z and adds per-dof terms with
//     `stZ = jf - dot` (the template spelling, NOT the shared
//     DyCpuGpuArticulation.h sum order and NOT the preloaded `-dot`);
//   * the acceleration hop accumulates per-dof contributions into a SEPARATE
//     jointSpatialDeltaV and returns `pDeltaV + jointSpatialDeltaV` at the
//     END (the child2ParentPreLoaded order, NOT the everythingPreLoaded
//     accumulate-onto-pDeltaV order).

#ifndef PXG_ARTI_SOLVE_REF_H
#define PXG_ARTI_SOLVE_REF_H

#include "foundation/PxVec3.h"
#include "CmSpatialVector.h"
#include "DyFeatherstoneArticulationUtils.h"
#include "DyFeatherstoneArticulation.h"
#include "PxgArticulation.h"

#include <cstdint>
#include <cstring>

namespace pxgtest
{

// propagateImpulseW<WriteToDeferredQstZ> (articulationDynamic.cuh:129)
inline physx::Cm::UnAlignedSpatialVector refPropImpulseDeferred(
    physx::PxgArticulationBlockDofData* dofs, physx::PxU32 jointOffset,
    physx::PxU32 dofCount, const physx::PxVec3& c2p,
    const physx::Cm::UnAlignedSpatialVector& Z, const physx::PxReal* jf,
    physx::PxReal jfMult, physx::PxU32 lane)
{
    using namespace physx;
    Cm::UnAlignedSpatialVector temp = Z;
    for (PxU32 ind = 0; ind < 3; ++ind)
    {
        if (ind < dofCount)
        {
            PxgArticulationBlockDofData& dd = dofs[jointOffset + ind];
            const Cm::UnAlignedSpatialVector sas = loadSpatialVector(dd.mWorldMotionMatrix, lane);
            const Cm::UnAlignedSpatialVector isInvD = loadSpatialVector(dd.mIsInvDW, lane);
            const PxReal jfv = jf ? jf[ind] * jfMult : 0.0f;
            const PxReal stZ = jfv - sas.innerProduct(Z);
            dd.mDeferredQstZ[lane] += stZ;
            temp += isInvD * stZ;
        }
    }
    return Dy::FeatherstoneArticulation::translateSpatialVector(c2p, temp);
}

// P9.86c (npArtiSolveInternal c1/c2): witness counters and host twins of the
// port's literal-seed fold in this hop.  llvmpipe unrolls the port's constant
// 3-trip loop, forwards the literal +0 seed of jointSpatialDeltaV and folds the
// first `0 + sas*jDelta` (F3), so the port's raw accumulator is the SEEDLESS
// sum: equal in value to the CUDA's +0-seeded one (articulationDynamic.cuh:292),
// but -0 where the CUDA's is +0.  npArtiSolveTendons.slang canonicalises that
// accumulator (-0 -> +0) before `pDeltaV + J`; the two DFS call sites of
// npArtiSolveInternal.slang wrap the result once more (isPlusZeroSV).
// gPropAccelVariant selects what the hop returns:
//   0  the CUDA (+0 seed) -- the reference every fixture compares against;
//   1  the seedless accumulator, canonicalised: HEAD's helper with the DFS
//      call-site wrap reverted (npArtiSolveInternal-c1-tgs / -c2-pgs);
//   2  the seedless accumulator as is: that revert on top of the helper's own
//      revert (npArtiSolveTendons-r0).
// Counted under every variant: gPropAccelFoldNegZero -- accumulator lanes the
// seedless sum leaves -0 (the CUDA's is +0); gPropAccelFoldHits -- those of
// them whose pDeltaV lane is -0 as well, i.e. where variant 2 returns -0
// against the CUDA's +0.
inline int gPropAccelVariant = 0;
inline physx::PxU32 gPropAccelFoldNegZero = 0;
inline physx::PxU32 gPropAccelFoldHits = 0;

inline bool refIsNegZero(float x)
{
    uint32_t b;
    std::memcpy(&b, &x, 4);
    return b == 0x80000000u;
}

// propagateAccelerationW_child2ParentPreLoaded<ReadFromDeferredQstZ> (:257)
inline physx::Cm::UnAlignedSpatialVector refPropAccelDeferred(
    const physx::PxgArticulationBlockDofData* dofs, physx::PxU32 jointOffset,
    physx::PxU32 dofCount, const physx::PxVec3& c2p,
    const physx::Cm::UnAlignedSpatialVector& parentDeltaV, physx::PxReal* jDeltaOut,
    physx::PxU32 lane)
{
    using namespace physx;
    const Cm::UnAlignedSpatialVector pDeltaV =
        Dy::FeatherstoneArticulation::translateSpatialVector(-c2p, parentDeltaV);

    PxReal tJointDelta[3] = { 0.f, 0.f, 0.f };
    for (PxU32 ind = 0; ind < 3; ++ind)
    {
        if (ind < dofCount)
        {
            const PxgArticulationBlockDofData& dd = dofs[jointOffset + ind];
            const Cm::UnAlignedSpatialVector IsW = loadSpatialVector(dd.mIsW, lane);
            const PxReal temp = IsW.innerProduct(pDeltaV);
            tJointDelta[ind] = dd.mDeferredQstZ[lane] - temp;
        }
    }

    Cm::UnAlignedSpatialVector jointSpatialDeltaV(PxVec3(0.f), PxVec3(0.f));
    Cm::UnAlignedSpatialVector seedless(PxVec3(0.f), PxVec3(0.f));   // the port's folded sum (twins only)
    for (PxU32 ind = 0; ind < 3; ++ind)
    {
        if (ind < dofCount)
        {
            const PxgArticulationBlockDofData& dd = dofs[jointOffset + ind];
            const Cm::UnAlignedSpatialVector sas = loadSpatialVector(dd.mWorldMotionMatrix, lane);
            const PxReal jDelta = dd.mInvStIsT_x[lane] * tJointDelta[0]
                                + dd.mInvStIsT_y[lane] * tJointDelta[1]
                                + dd.mInvStIsT_z[lane] * tJointDelta[2];
            if (jDeltaOut)
                jDeltaOut[ind] = jDelta;
            jointSpatialDeltaV += sas * jDelta;
            seedless = (ind == 0) ? sas * jDelta : seedless + sas * jDelta;
        }
    }

    const float sl[6] = { seedless.top.x, seedless.top.y, seedless.top.z,
                          seedless.bottom.x, seedless.bottom.y, seedless.bottom.z };
    const float pd[6] = { pDeltaV.top.x, pDeltaV.top.y, pDeltaV.top.z,
                          pDeltaV.bottom.x, pDeltaV.bottom.y, pDeltaV.bottom.z };
    for (PxU32 k = 0; k < 6; ++k)
        if (refIsNegZero(sl[k]))
        {
            ++gPropAccelFoldNegZero;
            if (refIsNegZero(pd[k]))
                ++gPropAccelFoldHits;
        }

    if (gPropAccelVariant == 1)
    {
        const Cm::UnAlignedSpatialVector canon(
            PxVec3(sl[0] == 0.f ? 0.f : sl[0], sl[1] == 0.f ? 0.f : sl[1], sl[2] == 0.f ? 0.f : sl[2]),
            PxVec3(sl[3] == 0.f ? 0.f : sl[3], sl[4] == 0.f ? 0.f : sl[4], sl[5] == 0.f ? 0.f : sl[5]));
        return pDeltaV + canon;
    }
    if (gPropAccelVariant == 2)
        return pDeltaV + seedless;

    return pDeltaV + jointSpatialDeltaV;
}

// pxcFsGetVelocity (internalConstraints2.cu:2116).  bf rows are
// [(link * wordSize + word) * 32 + lane].
inline physx::Cm::UnAlignedSpatialVector refGetVelocity(
    const physx::PxgArticulationBlockData& ab, physx::PxgArticulationBlockLinkData* lk,
    const physx::PxgArticulationBlockDofData* df, const uint64_t* bf,
    physx::PxU32 wordSize, physx::PxU32 linkID, bool fixBase,
    physx::PxReal* jointDofSpeeds, physx::PxU32 lane)
{
    using namespace physx;
    Cm::UnAlignedSpatialVector deltaV(PxVec3(0.f), PxVec3(0.f));
    if (!fixBase)
    {
        const Cm::UnAlignedSpatialVector rootDeferredZ = loadSpatialVector(ab.mRootDeferredZ, lane);
        Dy::SpatialMatrix m;
        loadSpatialMatrix(ab.mInvSpatialArticulatedInertia, lane, m);
        deltaV = m * -rootDeferredZ;
    }

    PxReal deltaJointDofSpeeds[3] = { 0.f, 0.f, 0.f };
    const PxgArticulationBlockLinkData& link = lk[linkID];

    for (PxU32 j = 0, wordOffset = 0; j < wordSize; ++j, wordOffset += 64)
    {
        uint64_t pathToRoot = bf[(linkID * wordSize + j) * 32 + lane];
        while (pathToRoot != 0)
        {
            const PxU32 index = PxU32(__builtin_ctzll(pathToRoot)) + wordOffset;
            if (index != 0)
            {
                const PxgArticulationBlockLinkData& cLink = lk[index];
                const PxU32 jointOffset = cLink.mJointOffset[lane];
                const PxU32 dofCount = cLink.mDofs[lane];
                const PxVec3 c2p(cLink.mRw_x[lane], cLink.mRw_y[lane], cLink.mRw_z[lane]);
                PxReal* opt = ((linkID == index) && jointDofSpeeds) ? deltaJointDofSpeeds
                                                                    : NULL;
                deltaV = refPropAccelDeferred(df, jointOffset, dofCount, c2p, deltaV, opt,
                                              lane);
            }
            pathToRoot &= (pathToRoot - 1);
        }
    }

    if (jointDofSpeeds)
    {
        const PxU32 dofCount = link.mDofs[lane];
        const PxU32 jointOffset = link.mJointOffset[lane];
        for (PxU32 i = 0; i < dofCount; ++i)
            jointDofSpeeds[i] = df[jointOffset + i].mJointVelocities[lane]
                              + deltaJointDofSpeeds[i];
    }

    const Cm::UnAlignedSpatialVector motionVelocity = loadSpatialVector(link.mMotionVelocity, lane);
    return motionVelocity + deltaV;
}

// P9.86c (npArtiSolveTendons-r1): witness counter and host twin of the port's
// literal-seed fold of jointImpulseToApplyAtCommonLink.  The CUDA seeds it {0,0,0}
// and only ever += the joint impulses of the ends that ARE the common link
// (internalConstraints2.cu:2269/2283), so in IEEE it is never -0 (+0 + -0 = +0).
// The port's loops have a constant trip count of 3; llvmpipe unrolls them, forwards
// the literal seed and folds the first `0 + ji` (F3), so its raw array is the
// SEEDLESS sum -- equal in value, but -0 where the CUDA's is +0.
// npArtiSolveTendons.slang canonicalises it (-0 -> +0).
// gJiCommonVariant selects what the common-link hop is handed:
//   0  the CUDA (+0 seed) -- the reference every fixture compares against;
//   1  the seedless sum as is (npArtiSolveTendons-r1, the canonicalisation reverted).
// Counted under every variant: gJiCommonFoldNegZero -- entries the seedless sum
// leaves -0 (the CUDA's is +0).
inline int gJiCommonVariant = 0;
inline physx::PxU32 gJiCommonFoldNegZero = 0;

// pxcFsApplyImpulses (internalConstraints2.cu:2190)
inline void refApplyImpulses(
    physx::PxgArticulationBlockData& ab, physx::PxgArticulationBlockLinkData* lk,
    physx::PxgArticulationBlockDofData* df, const uint64_t* bf, physx::PxU32 wordSize,
    physx::PxU32 linkID0, const physx::PxVec3& linear0, const physx::PxVec3& angular0,
    const physx::PxReal* ji0,
    physx::PxU32 linkID1, const physx::PxVec3& linear1, const physx::PxVec3& angular1,
    const physx::PxReal* ji1, physx::PxU32 lane)
{
    using namespace physx;
    PxU32 commonLink = 0;
    for (PxI32 i = PxI32(wordSize) - 1; i >= 0; --i)
    {
        const uint64_t wordA = bf[(linkID0 * wordSize + i) * 32 + lane];
        const uint64_t wordB = bf[(linkID1 * wordSize + i) * 32 + lane];
        const uint64_t commonId = wordA & wordB;
        if (commonId != 0)
        {
            commonLink = PxU32(63 - __builtin_clzll(commonId)) + PxU32(i) * 64;
            break;
        }
    }

    Cm::UnAlignedSpatialVector Z0(-linear0, -angular0);
    Cm::UnAlignedSpatialVector Z1(-linear1, -angular1);

    for (PxU32 i = linkID1; i != commonLink; i = lk[i].mParents[lane])
    {
        PxgArticulationBlockLinkData& tlink = lk[i];
        const PxU32 jointOffset = tlink.mJointOffset[lane];
        const PxU32 dofCount = tlink.mDofs[lane];
        const PxVec3 c2p(tlink.mRw_x[lane], tlink.mRw_y[lane], tlink.mRw_z[lane]);
        const PxReal* apply = (linkID1 == i) ? ji1 : NULL;
        storeSpatialVector(tlink.mSolverSpatialImpulse,
                           loadSpatialVector(tlink.mSolverSpatialImpulse, lane) + Z1, lane);
        Z1 = refPropImpulseDeferred(df, jointOffset, dofCount, c2p, Z1, apply, 1.0f, lane);
    }

    for (PxU32 i = linkID0; i != commonLink; i = lk[i].mParents[lane])
    {
        PxgArticulationBlockLinkData& tlink = lk[i];
        const PxU32 jointOffset = tlink.mJointOffset[lane];
        const PxU32 dofCount = tlink.mDofs[lane];
        const PxVec3 c2p(tlink.mRw_x[lane], tlink.mRw_y[lane], tlink.mRw_z[lane]);
        const PxReal* apply = (linkID0 == i) ? ji0 : NULL;
        storeSpatialVector(tlink.mSolverSpatialImpulse,
                           loadSpatialVector(tlink.mSolverSpatialImpulse, lane) + Z0, lane);
        Z0 = refPropImpulseDeferred(df, jointOffset, dofCount, c2p, Z0, apply, 1.0f, lane);
    }

    Cm::UnAlignedSpatialVector ZCommon = Z0 + Z1;

    PxReal jiCommon[3] = { 0.f, 0.f, 0.f };
    bool hasJiCommon = false;
    if (((linkID0 == commonLink) && ji0) || ((linkID1 == commonLink) && ji1))
    {
        hasJiCommon = true;
        const PxU32 dofCountAtCommon = lk[commonLink].mDofs[lane];
        const PxU32 linkIndices[2] = { linkID0, linkID1 };
        const PxReal* jis[2] = { ji0, ji1 };
        PxReal seedless[3] = { 0.f, 0.f, 0.f };   // the port's folded sum (twin only)
        bool seeded = false;
        for (PxU32 k = 0; k < 2; ++k)
        {
            if ((linkIndices[k] == commonLink) && jis[k])
            {
                for (PxU32 i = 0; i < dofCountAtCommon; ++i)
                {
                    jiCommon[i] += jis[k][i];
                    seedless[i] = seeded ? seedless[i] + jis[k][i] : jis[k][i];
                }
                seeded = true;
            }
        }
        for (PxU32 i = 0; i < dofCountAtCommon; ++i)
        {
            if (refIsNegZero(seedless[i]))
                ++gJiCommonFoldNegZero;
            if (gJiCommonVariant == 1)
                jiCommon[i] = seedless[i];
        }
    }

    for (PxU32 i = commonLink; i; i = lk[i].mParents[lane])
    {
        PxgArticulationBlockLinkData& tlink = lk[i];
        const PxU32 jointOffset = tlink.mJointOffset[lane];
        const PxU32 dofCount = tlink.mDofs[lane];
        const PxVec3 c2p(tlink.mRw_x[lane], tlink.mRw_y[lane], tlink.mRw_z[lane]);
        const PxReal* apply = ((commonLink == i) && hasJiCommon) ? jiCommon : NULL;
        storeSpatialVector(tlink.mSolverSpatialImpulse,
                           loadSpatialVector(tlink.mSolverSpatialImpulse, lane) + ZCommon,
                           lane);
        ZCommon = refPropImpulseDeferred(df, jointOffset, dofCount, c2p, ZCommon, apply,
                                         1.0f, lane);
    }

    storeSpatialVector(ab.mRootDeferredZ,
                       loadSpatialVector(ab.mRootDeferredZ, lane) + ZCommon, lane);
}


// The four P8.98b solve-body mirrors.  These use unqualified physx names
// freely; the enclosing namespace block below provides them.
namespace sbref_detail { using namespace physx; using physx::Cm::UnAlignedSpatialVector; }
using namespace physx;
using physx::Cm::UnAlignedSpatialVector;

// ---- the transcribed solve-body mirrors (P8.98b), parameterised ----

// updateSolveInternalTendonConstraintsTGS (:2015)
inline void refUpdateTendonTGS(physx::PxgArticulationBlockData& ab,
                               PxgArticulationBlockLinkData* lk,
                               physx::PxgArticulationBlockSpatialTendonData* st,
                               physx::PxgArticulationInternalTendonConstraintData* itc,
                               physx::PxgArticulationBlockAttachmentData* at,
                               physx::PxU32 maxAttachments, physx::PxU32 lane)
{
    const PxU32 numTendons = ab.mNumSpatialTendons[lane];
    PxReal accumLength[64];
    for (PxU32 k = 0; k < 64; ++k) accumLength[k] = 0.f;

    for (PxU32 i = 0; i < numTendons; ++i)
    {
        const PxReal offset = st[i].mOffset[lane];
        PxgArticulationBlockAttachmentData* attachmentBlock = &at[i * maxAttachments];
        PxgArticulationInternalTendonConstraintData* constraintBlock = &itc[i * maxAttachments];

        uint64_t bitStack = attachmentBlock[0].mChildrens[lane];
        const PxReal coefficient = attachmentBlock[0].mCoefficient[lane];
        PxU32 stackCount = PxU32(__builtin_popcountll(bitStack));
        PxU32 parent = 0, numConstraints = 0;
        accumLength[parent] = offset * coefficient;
        PxU32 child = PxU32(63 - __builtin_clzll(bitStack));

        while (stackCount != 0)
        {
            stackCount--;
            PxgArticulationBlockAttachmentData& attachmentData = attachmentBlock[child];
            PxgArticulationBlockAttachmentData& pAttachmentData = attachmentBlock[parent];

            const PxU32 linkInd = attachmentData.mLinkIndex[lane];
            const PxU32 pLinkInd = pAttachmentData.mLinkIndex[lane];
            const PxTransform cB2w = loadSpatialTransform(lk[linkInd].mAccumulatedPose, lane);
            const PxTransform pB2w = loadSpatialTransform(lk[pLinkInd].mAccumulatedPose, lane);

            const PxVec3 rb = cB2w.q.rotate(attachmentData.mRelativeOffset[lane]);
            const PxVec3 ra = pB2w.q.rotate(pAttachmentData.mRelativeOffset[lane]);
            const PxVec3 cAttachPoint = cB2w.p + rb;
            const PxVec3 pAttachPoint = pB2w.p + ra;
            const PxVec3 dif = pAttachPoint - cAttachPoint;
            const PxReal distance = PxSqrt(dif.dot(dif));

            const PxReal u = distance * attachmentData.mCoefficient[lane] + accumLength[parent];
            const uint64_t children = attachmentData.mChildrens[lane];

            if (children)
            {
                stackCount += PxU32(__builtin_popcountll(children));
                accumLength[child] = u;
            }
            else
            {
                constraintBlock[numConstraints++].mAccumulatedLength[lane] = u;
            }

            if (stackCount > 0)
            {
                bitStack &= (~(1ull << child));
                bitStack |= children;
                child = PxU32(63 - __builtin_clzll(bitStack));
                parent = attachmentBlock[child].mParents[lane];
            }
        }
    }
}

// solveInternalSpatialConstraints (:2317)
inline void refSolveSpatial(physx::PxgArticulationBlockData& ab,
                            physx::PxgArticulationBlockLinkData* lk,
                            physx::PxgArticulationBlockDofData* df, const uint64_t* bf,
                            physx::PxU32 wordSize,
                            physx::PxgArticulationBlockSpatialTendonData* st,
                            physx::PxgArticulationInternalTendonConstraintData* itc,
                            physx::PxgArticulationBlockAttachmentData* at,
                            physx::PxU32 maxAttachments, bool isTGS, physx::PxU32 lane,
                            physx::PxU32& nOver, physx::PxU32& nUnder,
                            physx::PxU32& nInside, physx::PxU32& nZeroDf)
{
    const bool fixBase = (ab.mFlags[lane] & 1u) != 0;

    if (isTGS)
        refUpdateTendonTGS(ab, lk, st, itc, at, maxAttachments, lane);

    const PxU32 numTendons = ab.mNumSpatialTendons[lane];
    for (PxU32 i = 0; i < numTendons; ++i)
    {
        PxgArticulationInternalTendonConstraintData* constraintBlock = &itc[i * maxAttachments];
        const PxU32 numConstraints = st[i].mNumConstraints[lane];
        if (numConstraints == 0)
            continue;

        const PxU32 parentID = constraintBlock[numConstraints - 1].mLink0[lane];
        const UnAlignedSpatialVector parentVel =
            refGetVelocity(ab, lk, df, bf, wordSize, parentID, fixBase, NULL, lane);
        PxReal parentV =
            loadSpatialVector(constraintBlock[0].mRow0, lane).innerProduct(parentVel);

        for (PxI32 j = PxI32(numConstraints) - 1; j >= 0; --j)
        {
            PxgArticulationInternalTendonConstraintData& C = constraintBlock[j];
            const PxU32 childID = C.mLink1[lane];
            const UnAlignedSpatialVector childVel =
                refGetVelocity(ab, lk, df, bf, wordSize, childID, fixBase, NULL, lane);

            const PxReal accumLength = C.mAccumulatedLength[lane];
            const PxReal error = C.mRestDistance[lane] - accumLength;
            PxReal error2 = 0.f;
            if (accumLength > C.mHighLimit[lane]) { error2 = C.mHighLimit[lane] - accumLength; ++nOver; }
            else if (accumLength < C.mLowLimit[lane]) { error2 = C.mLowLimit[lane] - accumLength; ++nUnder; }
            else ++nInside;

            const UnAlignedSpatialVector row1 = loadSpatialVector(C.mRow1, lane);
            const UnAlignedSpatialVector row0 = loadSpatialVector(C.mRow0, lane);
            const PxReal jointV = row1.innerProduct(childVel) - parentV;

            const PxReal unclampedForce = jointV * C.mVelMultiplier[lane]
                                        + error * C.mBiasCoefficient[lane]
                                        + C.mAppliedForce[lane] * C.mImpulseMultiplier[lane];
            const PxReal unclampedForce2 = (error2 * C.mLimitBiasCoefficient[lane])
                                         + C.mLimitAppliedForce[lane] * C.mLimitImpulseMultiplier[lane];
            const PxReal deltaF = (unclampedForce - C.mAppliedForce[lane])
                                + (unclampedForce2 - C.mLimitAppliedForce[lane]);

            C.mAppliedForce[lane] = unclampedForce;
            C.mLimitAppliedForce[lane] = unclampedForce2;
            parentV += C.mDeltaVA[lane] * -deltaF;

            if (deltaF != 0.f)
            {
                const UnAlignedSpatialVector i0 = row0 * (-deltaF);
                const UnAlignedSpatialVector i1 = row1 * deltaF;
                refApplyImpulses(ab, lk, df, bf, wordSize,
                                 parentID, i0.top, i0.bottom, NULL,
                                 childID, i1.top, i1.bottom, NULL, lane);
            }
            else
                ++nZeroDf;
        }
    }
}

// solveInternalFixedConstraints (:2435)
inline void refSolveFixed(physx::PxgArticulationBlockData& ab,
                          physx::PxgArticulationBlockLinkData* lk,
                          physx::PxgArticulationBlockDofData* df, const uint64_t* bf,
                          physx::PxU32 wordSize,
                          physx::PxgArticulationBlockFixedTendonData* ft,
                          physx::PxgArticulationInternalTendonConstraintData* itc,
                          physx::PxgArticulationBlockTendonJointData* tj,
                          physx::PxU32 maxTendonJoints, physx::PxU32 lane,
                          physx::PxU32& nOver, physx::PxU32& nUnder, physx::PxU32& nInside)
{
    const bool fixBase = (ab.mFlags[lane] & 1u) != 0;
    const PxU32 numTendons = ab.mNumFixedTendons[lane];

    for (PxU32 i = 0; i < numTendons; ++i)
    {
        PxgArticulationBlockFixedTendonData& tendonData = ft[i];
        const PxReal lowLimit = tendonData.mLowLimit[lane];
        const PxReal highLimit = tendonData.mHighLimit[lane];

        PxgArticulationBlockTendonJointData* tendonJointBlock = &tj[i * maxTendonJoints];
        PxgArticulationInternalTendonConstraintData* constraintBlock = &itc[i * maxTendonJoints];

        uint64_t bitStack = tendonJointBlock[0].mChildrens[lane];
        PxU32 stackCount = PxU32(__builtin_popcountll(bitStack));

        const PxU32 sLinkIndex = tendonJointBlock[0].mLinkIndex[lane];
        PxgArticulationBlockLinkData& sLink = lk[sLinkIndex];

        const UnAlignedSpatialVector parentVel =
            refGetVelocity(ab, lk, df, bf, wordSize, sLinkIndex, fixBase, NULL, lane);
        const UnAlignedSpatialVector velA = loadSpatialVector(sLink.mMotionVelocity, lane);
        const UnAlignedSpatialVector delta = parentVel - velA;
        storeSpatialVector(sLink.mScratchImpulse, UnAlignedSpatialVector(PxVec3(0.f), PxVec3(0.f)), lane);
        storeSpatialVector(sLink.mScratchDeltaV, delta, lane);

        PxReal rootImp = 0.f;
        uint64_t totalStack = 0;
        PxReal error = 0.f, velocity = 0.f;
        const PxU32 firstChild = PxU32(63 - __builtin_clzll(bitStack));
        totalStack |= bitStack;

        while (stackCount != 0)
        {
            const PxU32 child = PxU32(63 - __builtin_clzll(bitStack));
            bitStack &= (~(1ull << child));
            stackCount--;

            PxgArticulationBlockTendonJointData& tjData = tendonJointBlock[child];
            const PxU32 cLinkInd = tjData.mLinkIndex[lane];
            PxgArticulationBlockLinkData& cLink = lk[cLinkInd];
            const PxU32 pLinkInd = cLink.mParents[lane];
            PxgArticulationBlockLinkData& pLink = lk[pLinkInd];
            const PxU32 jointOffset = cLink.mJointOffset[lane];
            const PxU32 dofCount = cLink.mDofs[lane];
            const PxVec3 c2p(cLink.mRw_x[lane], cLink.mRw_y[lane], cLink.mRw_z[lane]);

            const PxU32 constraintId = tjData.mConstraintId[lane];
            PxgArticulationInternalTendonConstraintData& C = constraintBlock[constraintId];
            const PxU32 tjAxis = tjData.mAxis[lane];
            const PxReal coefficient = tjData.mCoefficient[lane];
            const PxU32 dofIndex = cLink.mInvDofIds[tjAxis][lane];
            const PxReal jointPose = df[jointOffset + dofIndex].mJointPositions[lane];

            const UnAlignedSpatialVector parentDeltaV = loadSpatialVector(pLink.mScratchDeltaV, lane);
            const UnAlignedSpatialVector parentV =
                loadSpatialVector(pLink.mMotionVelocity, lane) + parentDeltaV;

            const UnAlignedSpatialVector cDeltaV =
                refPropAccelDeferred(df, jointOffset, dofCount, c2p, parentDeltaV, NULL, lane);
            storeSpatialVector(cLink.mScratchDeltaV, cDeltaV, lane);

            const UnAlignedSpatialVector velB = loadSpatialVector(cLink.mMotionVelocity, lane);
            const UnAlignedSpatialVector childVel = velB + cDeltaV;

            storeSpatialVector(cLink.mScratchImpulse, UnAlignedSpatialVector(PxVec3(0.f), PxVec3(0.f)), lane);

            const uint64_t children = tjData.mChildrens[lane];

            const UnAlignedSpatialVector row0 = loadSpatialVector(C.mRow0, lane);
            const UnAlignedSpatialVector row1 = loadSpatialVector(C.mRow1, lane);
            const PxReal jointV = row1.innerProduct(childVel) - row0.innerProduct(parentV);

            error += jointPose * coefficient;
            velocity += jointV * coefficient;

            if (children)
            {
                stackCount += PxU32(__builtin_popcountll(children));
                bitStack |= children;
                totalStack |= children;
            }
        }

        const PxU32 count = PxU32(__builtin_popcountll(totalStack));
        const PxReal scale = count ? 1.f / PxReal(count) : 0.f;

        const PxReal length = error + tendonData.mOffset[lane];
        PxReal limitError = 0.f;
        if (length < lowLimit) { limitError = length - lowLimit; ++nUnder; }
        else if (length > highLimit) { limitError = length - highLimit; ++nOver; }
        else ++nInside;

        error = (length - tendonData.mRestLength[lane]) * scale;
        limitError *= scale;
        velocity *= scale;

        while (totalStack != 0)
        {
            const PxU32 child = PxU32(63 - __builtin_clzll(totalStack));
            totalStack &= (~(1ull << child));

            PxgArticulationBlockTendonJointData& tjData = tendonJointBlock[child];
            const PxU32 cLinkInd = tjData.mLinkIndex[lane];
            PxgArticulationBlockLinkData& cLink = lk[cLinkInd];
            const PxU32 pLinkInd = cLink.mParents[lane];
            PxgArticulationBlockLinkData& pLink = lk[pLinkInd];
            const PxU32 jointOffset = cLink.mJointOffset[lane];
            const PxU32 dofCount = cLink.mDofs[lane];
            const PxVec3 c2p(cLink.mRw_x[lane], cLink.mRw_y[lane], cLink.mRw_z[lane]);

            const PxU32 constraintId = tjData.mConstraintId[lane];
            PxgArticulationInternalTendonConstraintData& C = constraintBlock[constraintId];

            const PxReal recipCoefficient = tjData.mRecipCoefficient[lane];
            const PxReal unclampedForce =
                ((velocity * C.mVelMultiplier[lane] + error * C.mBiasCoefficient[lane]) * recipCoefficient)
                + C.mAppliedForce[lane] * C.mImpulseMultiplier[lane];
            const PxReal unclampedForce2 =
                (limitError * C.mLimitBiasCoefficient[lane] * recipCoefficient)
                + C.mLimitAppliedForce[lane] * C.mLimitImpulseMultiplier[lane];
            const PxReal deltaF = ((unclampedForce - C.mAppliedForce[lane])
                                 + (unclampedForce2 - C.mLimitAppliedForce[lane]));

            C.mAppliedForce[lane] = unclampedForce;
            C.mLimitAppliedForce[lane] = unclampedForce2;
            rootImp += deltaF;

            const UnAlignedSpatialVector cImpulse = loadSpatialVector(cLink.mScratchImpulse, lane);
            const UnAlignedSpatialVector impulse =
                loadSpatialVector(C.mRow1, lane) * (-deltaF) + cImpulse;

            storeSpatialVector(cLink.mSolverSpatialImpulse,
                               loadSpatialVector(cLink.mSolverSpatialImpulse, lane) + cImpulse,
                               lane);

            const UnAlignedSpatialVector propagatedImpulse =
                refPropImpulseDeferred(df, jointOffset, dofCount, c2p, impulse, NULL, 1.0f, lane);

            storeSpatialVector(pLink.mScratchImpulse,
                               loadSpatialVector(pLink.mScratchImpulse, lane) + propagatedImpulse,
                               lane);
        }

        const PxU32 firstConstraint = tendonJointBlock[firstChild].mConstraintId[lane];
        const PxgArticulationInternalTendonConstraintData& C0 = constraintBlock[firstConstraint];

        const UnAlignedSpatialVector propagatedImpulse = loadSpatialVector(sLink.mScratchImpulse, lane);
        UnAlignedSpatialVector Z = propagatedImpulse + loadSpatialVector(C0.mRow0, lane) * rootImp;

        for (PxU32 linkID = sLinkIndex; linkID; linkID = lk[linkID].mParents[lane])
        {
            PxgArticulationBlockLinkData& link = lk[linkID];
            const PxU32 jointOffset = link.mJointOffset[lane];
            const PxU32 dofCount = link.mDofs[lane];
            const PxVec3 c2p(link.mRw_x[lane], link.mRw_y[lane], link.mRw_z[lane]);
            Z = refPropImpulseDeferred(df, jointOffset, dofCount, c2p, Z, NULL, 1.0f, lane);
        }

        storeSpatialVector(ab.mRootDeferredZ,
                           loadSpatialVector(ab.mRootDeferredZ, lane) + Z, lane);
    }

    if (numTendons > 0)
    {
        const PxU32 nbLinks = ab.mNumLinks[lane];
        for (PxU32 i = 0; i < nbLinks; ++i)
            storeSpatialVector(lk[i].mScratchImpulse,
                               UnAlignedSpatialVector(PxVec3(0.f), PxVec3(0.f)), lane);
    }
}

// solveInternalMimicJointConstraints (:2693) -- the impulse scalar itself by
// the ENGINE's Dy::computeMimicJointImpulses
inline void refSolveMimic(physx::PxgArticulationBlockData& ab,
                          physx::PxgArticulationBlockLinkData* lk,
                          physx::PxgArticulationBlockDofData* df, const uint64_t* bf,
                          physx::PxU32 wordSize,
                          physx::PxgArticulationBlockMimicJointData* mjb,
                          physx::PxReal biasCoefficient, physx::PxReal dt,
                          physx::PxReal recipDt, bool isVelocityIteration,
                          physx::PxU32 lane,
                          physx::PxU32& nCompliant, physx::PxU32& nHard)
{
    const bool fixBase = (ab.mFlags[lane] & 1u) != 0;
    const PxU32 numMimicJoints = ab.mNumMimicJoints[lane];

    for (PxU32 i = 0; i < numMimicJoints; ++i)
    {
        const PxgArticulationBlockMimicJointData& M = mjb[i];
        const PxU32 linkA = M.mLinkA[lane];
        const PxU32 linkB = M.mLinkB[lane];
        const PxU32 dofA = M.mInternalData.mDofA[lane];
        const PxU32 dofB = M.mInternalData.mDofB[lane];
        const PxReal r = M.mInternalData.recipEffectiveInertia[lane];
        const PxReal gearRatio = M.mGearRatio[lane];
        const PxReal offset = M.mOffset[lane];
        const PxReal naturalFrequency = M.mNaturalFrequency[lane];
        const PxReal dampingRatio = M.mDampingRatio[lane];

        if (naturalFrequency > 0.f && dampingRatio > 0.f && !isVelocityIteration) ++nCompliant;
        else ++nHard;

        const PxU32 jointOffsetA = lk[linkA].mJointOffset[lane];
        const PxU32 jointOffsetB = lk[linkB].mJointOffset[lane];
        const PxReal qA = df[jointOffsetA + dofA].mJointPositions[lane];
        const PxReal qB = df[jointOffsetB + dofB].mJointPositions[lane];

        PxReal qADot = 0.f, qBDot = 0.f;
        {
            PxReal jointDofSpeedsA[3] = { 0, 0, 0 };
            PxReal jointDofSpeedsB[3] = { 0, 0, 0 };
            refGetVelocity(ab, lk, df, bf, wordSize, linkA, fixBase, jointDofSpeedsA, lane);
            refGetVelocity(ab, lk, df, bf, wordSize, linkB, fixBase, jointDofSpeedsB, lane);
            qADot = jointDofSpeedsA[dofA];
            qBDot = jointDofSpeedsB[dofB];
        }

        PxReal jointImpulseA[3] = { 0, 0, 0 };
        PxReal jointImpulseB[3] = { 0, 0, 0 };
        {
            PxReal jointImpDofA = 0.f, jointImpDofB = 0.f;
            Dy::computeMimicJointImpulses(biasCoefficient, dt, recipDt, qA, qB, qADot, qBDot,
                                          gearRatio, offset, naturalFrequency, dampingRatio,
                                          r, isVelocityIteration, jointImpDofA, jointImpDofB);
            jointImpulseA[dofA] = jointImpDofA;
            jointImpulseB[dofB] = jointImpDofB;
        }

        const PxVec3 zero(0.f);
        refApplyImpulses(ab, lk, df, bf, wordSize,
                         linkA, zero, zero, jointImpulseA,
                         linkB, zero, zero, jointImpulseB, lane);
    }
}


} // namespace pxgtest

#endif // PXG_ARTI_SOLVE_REF_H
