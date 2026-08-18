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
        }
    }

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
        for (PxU32 k = 0; k < 2; ++k)
        {
            if ((linkIndices[k] == commonLink) && jis[k])
                for (PxU32 i = 0; i < dofCountAtCommon; ++i)
                    jiCommon[i] += jis[k][i];
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

} // namespace pxgtest

#endif // PXG_ARTI_SOLVE_REF_H
