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

// pxgArtiRefHelpers.h
//
// The three articulation helpers that are `static __device__` in
// forwardDynamic2.cu and therefore cannot be called from the host, mirrored ONCE
// for every fixture that needs them (P8.86's algebra table and P8.88's
// leaf-to-root sweep).
//
// Only the STRUCTURE is transcribed: every operation inside is the engine's own
// `PxMat33` / `PxVec3` operator, so the arithmetic is still the engine's.  That
// distinction matters because the structure is exactly what P8.86's trap 5 is
// about — `translateInertia` forms `bl` and `br` from the OLD `topLeft` /
// `topRight` before overwriting `topLeft`, and never touches `topRight`.
//
// A second copy of these five lines per fixture is precisely the duplication this
// port exists to avoid, which is why they live in a header rather than being
// pasted twice.

#ifndef PXG_ARTI_REF_HELPERS_H
#define PXG_ARTI_REF_HELPERS_H

#include "foundation/PxMat33.h"
#include "foundation/PxVec3.h"
#include "foundation/PxQuat.h"
#include "foundation/PxTransform.h"
#include "foundation/PxMathUtils.h"
#include "CmSpatialVector.h"
#include "DyFeatherstoneArticulationUtils.h"
#include "DyFeatherstoneArticulation.h"
#include "PxgArticulation.h"

namespace pxgtest
{
// constructSkewSymmetricMatrix (forwardDynamic2.cu:1561).  A permutation of r's
// components with three sign flips -- no arithmetic, so nothing to round.  `r` is
// a float4 in the engine because the caller has one; w is ignored.
inline physx::PxMat33 refSkewSymmetric(float x, float y, float z)
{
    using physx::PxVec3;
    return physx::PxMat33(PxVec3(0.f, z, -y), PxVec3(-z, 0.f, x), PxVec3(y, -x, 0.f));
}

// translateInertia (forwardDynamic2.cu:1546).  `bl` and `br` come from the OLD
// topLeft / topRight; topRight is never written; bottomLeft is symmetrised last.
inline void refTranslateInertia(const physx::PxMat33& sTod, physx::Dy::SpatialMatrix& inertia)
{
    const physx::PxMat33 dTos = sTod.getTranspose();

    physx::PxMat33 bl = sTod * inertia.topLeft + inertia.bottomLeft;
    physx::PxMat33 br = sTod * inertia.topRight + inertia.getBottomRight();

    inertia.topLeft = inertia.topLeft + inertia.topRight * dTos;
    inertia.bottomLeft = bl + br * dTos;

    inertia.bottomLeft = (inertia.bottomLeft + inertia.bottomLeft.getTranspose()) * 0.5f;
}

// translateSpatialVectorInPlace (forwardDynamic2.cu:57).  The bottom half picks up
// `offset x top`; the top half is untouched.
inline void refTranslateSpatialVectorInPlace(const physx::PxVec3& offset,
                                             physx::Cm::UnAlignedSpatialVector& vec)
{
    vec.bottom += offset.cross(vec.top);
}

// ---------------------------------------------------------------------------
// The impulse-response pair (articulationDynamic.cuh:167 / :206)
// ---------------------------------------------------------------------------
//
// `propagateImpulseW_everythingPreLoaded` and
// `propagateAccelerationW_everythingPreLoaded` are `__device__` too, so the same
// rule applies: the STRUCTURE is transcribed, every operation inside is the
// engine's own (`innerProduct`, `operator*`, `operator+=`, and
// `FeatherstoneArticulation::translateSpatialVector`, all PX_CUDA_CALLABLE).
//
// Hoisted here in P8.91c because `computeMassMatrix1T` calls them six times per
// link, and its fixture would otherwise be the THIRD transcription of the same
// twelve lines.  P8.90's `testArtiDynamic.cpp` uses these same two.
//
// The traps, from P8.90's module header:
//   * `innerProduct` is the DUAL product, not `dot`;
//   * `stZ` is taken against `Z`, never against the running `temp`;
//   * the acceleration direction NEGATES its offset, the impulse one does not;
//   * the acceleration direction is TWO passes — every `tJointDelta` comes from
//     the pre-accumulation `pDeltaV`;
//   * `jDelta` is a FULL three-lane dot, including the lanes at or beyond
//     `dofCount`.
inline physx::Cm::UnAlignedSpatialVector refPropagateImpulseW(
    const physx::Cm::UnAlignedSpatialVector* isInvD, const physx::PxVec3& childToParent,
    const physx::Cm::UnAlignedSpatialVector* motionMatrix,
    const physx::Cm::UnAlignedSpatialVector& Z, physx::PxU32 dofCount, physx::PxReal* qstZ)
{
    physx::Cm::UnAlignedSpatialVector temp = Z;
    for (physx::PxU32 ind = 0; ind < 3; ++ind)
    {
        if (ind < dofCount)
        {
            const physx::PxReal stZ = -motionMatrix[ind].innerProduct(Z);
            qstZ[ind] += stZ;
            temp += isInvD[ind] * stZ;
        }
    }
    return physx::Dy::FeatherstoneArticulation::translateSpatialVector(childToParent, temp);
}

inline physx::Cm::UnAlignedSpatialVector refPropagateAccelerationW(
    const physx::PxVec3& parentToChild, const physx::PxVec3* invStISW,
    const physx::Cm::UnAlignedSpatialVector* motionMatrix,
    const physx::Cm::UnAlignedSpatialVector& parentLinkAccelerationW, physx::PxU32 dofCount,
    const physx::Cm::UnAlignedSpatialVector* IsW, const physx::PxReal* QMinusSTZ)
{
    physx::Cm::UnAlignedSpatialVector pDeltaV =
        physx::Dy::FeatherstoneArticulation::translateSpatialVector(-parentToChild,
                                                                    parentLinkAccelerationW);

    physx::PxReal tJointDelta[3] = { 0.f, 0.f, 0.f };
    for (physx::PxU32 ind = 0; ind < 3; ++ind)
    {
        if (ind < dofCount)
        {
            const physx::PxReal temp = IsW[ind].innerProduct(pDeltaV);
            tJointDelta[ind] = (QMinusSTZ[ind] - temp);
        }
    }

    for (physx::PxU32 ind = 0; ind < 3; ++ind)
    {
        if (ind < dofCount)
        {
            const physx::PxVec3 iStIsTi = invStISW[ind];
            const physx::PxReal jDelta = iStIsTi.x * tJointDelta[0] + iStIsTi.y * tJointDelta[1]
                                       + iStIsTi.z * tJointDelta[2];
            pDeltaV += motionMatrix[ind] * jDelta;
        }
    }
    return pDeltaV;
}

// The OTHER translateInertia (forwardDynamic2.cu:1569) -- the parallel axis
// theorem on a plain 3x3, which RETURNS rather than mutating.  Structure only;
// `getTranspose`, both `operator*` forms and `operator+` are the engine's.
//
// The association is load-bearing: `((s^T * s) * mass) + inertia`, left to right.
inline physx::PxMat33 refTranslateInertiaMass(const physx::PxMat33& inertia, physx::PxReal mass,
                                              const physx::PxVec3& t)
{
    using physx::PxVec3;
    const physx::PxMat33 s(PxVec3(0.f, t.z, -t.y), PxVec3(-t.z, 0.f, t.x), PxVec3(t.y, -t.x, 0.f));
    return s.getTranspose() * s * mass + inertia;
}

}  // namespace pxgtest


// ===========================================================================
// P8.94d hoists.  These mirrors were file-static in testArtiStep.cpp,
// testArtiFlushVel.cpp and testArtiAvgImpulse.cpp; the stepArticulation1TTGS
// kernel fixture composes all three, so they moved here VERBATIM (static ->
// inline is the only edit) -- one mirror, N users, no second transcription.
// The AI_NONE constant and the JointType enum moved with them.
// ===========================================================================

namespace pxgref
{
using namespace physx;
using physx::Cm::UnAlignedSpatialVector;

enum JointType : uint8_t { J_FIX = 0, J_PRISMATIC, J_REVOLUTE, J_REVOLUTE_UNWRAPPED, J_SPHERICAL };

const uint32_t AI_NONE = 0xFFFFFFFFu;

inline PxVec3 refToRotationVector(const PxQuat& q)
{
    const float quatEpsilon = 1.0e-8f;
    const float s2 = q.x * q.x + q.y * q.y + q.z * q.z;
    PxReal angle;
    PxVec3 axis;
    if (s2 < quatEpsilon * quatEpsilon)
    {
        angle = 0.0f;
        axis = PxVec3(1.0f, 0.0f, 0.0f);
    }
    else
    {
        const float s = physx::PxRecipSqrt(s2);
        axis = PxVec3(q.x, q.y, q.z) * s;
        angle = physx::PxAbs(q.w) < quatEpsilon ? physx::PxPi
                                                : physx::PxAtan2(s2 * s, q.w) * 2.0f;
    }
    return axis * angle;
}

inline PxTransform refUpdateRootBody(const UnAlignedSpatialVector& motionVelocity,
                                     const PxTransform& preTransform, PxReal dt)
{
    const PxVec3 comLinVel = motionVelocity.bottom;
    const PxVec3 newP = (preTransform.p) + comLinVel * dt;
    const PxQuat deltaQ = physx::PxExp(motionVelocity.top * dt);
    return PxTransform(newP, (deltaQ * preTransform.q).getNormalized());
}

inline PxQuat refSphericalJointPositions(const PxQuat& relativeQuat, const PxQuat& newRot,
                                         const PxQuat& pBody2WorldRot,
                                         physx::PxgArticulationBlockDofData* dofData, PxU32 dofs,
                                         uint32_t lane)
{
    PxQuat newParentToChild = (newRot.getConjugate() * pBody2WorldRot).getNormalized();
    if (newParentToChild.w < 0.f)
        newParentToChild = -newParentToChild;

    PxQuat jointRotation = newParentToChild * relativeQuat.getConjugate();
    if (jointRotation.w < 0)
        jointRotation = -jointRotation;

    const PxVec3 axis = refToRotationVector(jointRotation);
    for (PxU32 dd = 0; dd < dofs; ++dd)
    {
        const float4 top = dofData[dd].mLocalMotionMatrix.mTopxyz_bx[lane];
        const PxReal ang = -PxVec3(top.x, top.y, top.z).dot(axis);
        dofData[dd].mJointPositions[lane] = ang;
    }
    return newParentToChild;
}

inline PxTransform refPropagateLink(physx::PxgArticulationBlockLinkData* lk,
                                    physx::PxgArticulationBlockDofData* df, uint32_t linkID,
                                    PxReal dt, bool isSubstep, uint32_t lane)
{
    physx::PxgArticulationBlockLinkData& link = lk[linkID];
    const PxU32 jointOffset = link.mJointOffset[lane];
    const PxU32 jointType = link.mJointType[lane];
    const PxU32 parent = link.mParents[lane];
    physx::PxgArticulationBlockDofData* dof = df + jointOffset;

    const PxTransform pBody2World = loadSpatialTransform(lk[parent].mAccumulatedPose, lane);

    const float4 co = link.mChildPose.p[lane];
    const float4 po = link.mParentPose.p[lane];
    const float4 rq = link.mRelativeQuat[lane];
    const PxVec3 childOffset(-co.x, -co.y, -co.z);
    const PxVec3 parentOffset(po.x, po.y, po.z);
    const PxQuat relativeQuat(rq.x, rq.y, rq.z, rq.w);

    PxQuat newParentToChild = relativeQuat;
    PxVec3 offset(0.f);

    if (jointType == J_PRISMATIC)
    {
        const PxReal delta = (isSubstep ? dof[0].mJointVelocities[lane]
                                        : dof[0].mPosJointVelocities[lane]) * dt;
        const PxReal pos = dof[0].mJointPositions[lane] + delta;
        dof[0].mJointPositions[lane] = pos;
        newParentToChild = relativeQuat;
        const UnAlignedSpatialVector mm = loadSpatialVector(dof[0].mLocalMotionMatrix, lane);
        offset = mm.bottom * pos;
    }
    else if (jointType == J_REVOLUTE)
    {
        const PxReal delta = (isSubstep ? dof[0].mJointVelocities[lane]
                                        : dof[0].mPosJointVelocities[lane]) * dt;
        PxReal jPos = dof[0].mJointPositions[lane] + delta;
        if (jPos > physx::PxTwoPi)
            jPos -= physx::PxTwoPi * 2.f;
        else if (jPos < -physx::PxTwoPi)
            jPos += physx::PxTwoPi * 2.f;
        jPos = physx::PxClamp(jPos, -physx::PxTwoPi * 2.f, physx::PxTwoPi * 2.f);
        dof[0].mJointPositions[lane] = jPos;
        const float4 u = dof[0].mLocalMotionMatrix.mTopxyz_bx[lane];
        PxQuat jointRotation = PxQuat(-jPos, PxVec3(u.x, u.y, u.z));
        if (jointRotation.w < 0)
            jointRotation = -jointRotation;
        newParentToChild = (jointRotation * relativeQuat).getNormalized();
    }
    else if (jointType == J_REVOLUTE_UNWRAPPED)
    {
        const PxReal delta = (isSubstep ? dof[0].mJointVelocities[lane]
                                        : dof[0].mPosJointVelocities[lane]) * dt;
        const PxReal jPos = dof[0].mJointPositions[lane] + delta;
        dof[0].mJointPositions[lane] = jPos;
        const float4 u = dof[0].mLocalMotionMatrix.mTopxyz_bx[lane];
        PxQuat jointRotation = PxQuat(-jPos, PxVec3(u.x, u.y, u.z));
        if (jointRotation.w < 0)
            jointRotation = -jointRotation;
        newParentToChild = (jointRotation * relativeQuat).getNormalized();
    }
    else if (jointType == J_SPHERICAL)
    {
        const PxU32 nbDofs = link.mDofs[lane];
        const PxTransform oldTransform = loadSpatialTransform(link.mAccumulatedPose, lane);
        UnAlignedSpatialVector worldVel = isSubstep
                                              ? loadSpatialVector(link.mMotionVelocity, lane)
                                              : loadSpatialVector(link.mPosMotionVelocity, lane);
        PxVec3 worldAngVel = worldVel.top;
        const PxReal dist = worldAngVel.normalize() * dt;
        PxQuat newWorldQ;
        if (dist > 1e-6f)
            newWorldQ = PxQuat(dist, worldAngVel) * oldTransform.q;
        else
            newWorldQ = oldTransform.q;
        newParentToChild =
            refSphericalJointPositions(relativeQuat, newWorldQ, pBody2World.q, dof, nbDofs, lane);
    }
    else
    {
        newParentToChild = relativeQuat;
    }

    const PxVec3 e = newParentToChild.rotate(parentOffset);
    const PxVec3 d = childOffset;
    const PxVec3 r = e + d + offset;

    PxTransform body2World;
    body2World.q = (pBody2World.q * newParentToChild.getConjugate()).getNormalized();
    body2World.p = pBody2World.p + body2World.q.rotate(r);
    return body2World;
}

inline UnAlignedSpatialVector fvPropagateAccel(physx::PxgArticulationBlockLinkData& link,
                                               physx::PxgArticulationBlockDofData* dofData,
                                               PxU32 dofCount,
                                               const UnAlignedSpatialVector& parentAccel,
                                               uint32_t lane)
{
    const float rx = link.mRw_x[lane];
    const float ry = link.mRw_y[lane];
    const float rz = link.mRw_z[lane];

    PxVec3 invStIsT[3];
    PxReal tJointDelta[3] = { 0.f, 0.f, 0.f };
    UnAlignedSpatialVector isW[3];
    UnAlignedSpatialVector motionMatrix[3];
    PxReal jVel[3] = { 0.f, 0.f, 0.f };

    for (PxU32 ind = 0; ind < 3; ++ind)
    {
        if (ind < dofCount)
        {
            isW[ind] = loadSpatialVector(dofData[ind].mIsW, lane);
            tJointDelta[ind] = dofData[ind].mDeferredQstZ[lane];
            jVel[ind] = dofData[ind].mJointVelocities[lane];
            invStIsT[ind] = PxVec3(dofData[ind].mInvStIsT_x[lane], dofData[ind].mInvStIsT_y[lane],
                                   dofData[ind].mInvStIsT_z[lane]);
            motionMatrix[ind] = loadSpatialVector(dofData[ind].mWorldMotionMatrix, lane);
        }
    }

    UnAlignedSpatialVector pDeltaV =
        physx::Dy::FeatherstoneArticulation::translateSpatialVector(PxVec3(-rx, -ry, -rz),
                                                                    parentAccel);

    for (PxU32 ind = 0; ind < 3; ++ind)
        if (ind < dofCount)
            tJointDelta[ind] -= isW[ind].innerProduct(pDeltaV);

    for (PxU32 ind = 0; ind < 3; ++ind)
    {
        if (ind < dofCount)
        {
            const PxReal jDelta = invStIsT[ind].x * tJointDelta[0]
                                + invStIsT[ind].y * tJointDelta[1]
                                + invStIsT[ind].z * tJointDelta[2];
            dofData[ind].mJointVelocities[lane] = jVel[ind] + jDelta;
            pDeltaV += motionMatrix[ind] * jDelta;
        }
    }
    return pDeltaV;
}

inline void fvMirror(physx::PxgArticulationBlockData* ab,
                     physx::PxgArticulationBlockLinkData* lk,
                     physx::PxgArticulationBlockDofData* df, uint32_t linkCount, bool fixBase,
                     uint32_t lane)
{
    UnAlignedSpatialVector deltaV = UnAlignedSpatialVector::Zero();
    const UnAlignedSpatialVector deferredZ = -loadSpatialVector(ab->mRootDeferredZ, lane);

    if (!fixBase)
    {
        const UnAlignedSpatialVector motionVelocity0 = loadSpatialVector(lk[0].mMotionVelocity,
                                                                        lane);
        const UnAlignedSpatialVector solverSpatialDeltaVel0 =
            loadSpatialVector(lk[0].mSolverSpatialDeltaVel, lane);

        physx::Dy::SpatialMatrix invInertia;
        loadSpatialMatrix(ab->mInvSpatialArticulatedInertia, lane, invInertia);

        deltaV = invInertia * deferredZ;

        storeSpatialVector(lk[0].mMotionVelocity, motionVelocity0 + deltaV, lane);
        storeSpatialVector(ab->mRootDeferredZ, UnAlignedSpatialVector::Zero(), lane);
        storeSpatialVector(lk[0].mSolverSpatialDeltaVel, solverSpatialDeltaVel0 + deltaV, lane);
    }

    storeSpatialVector(lk[0].mScratchDeltaV, deltaV, lane);
    storeSpatialVector(lk[0].mConstraintForces,
                       loadSpatialVector(lk[0].mConstraintForces, lane) + deferredZ, lane);
    storeSpatialVector(ab->mCommonLinkDeltaVelocity, UnAlignedSpatialVector::Zero(), lane);

    physx::PxgArticulationBlockDofData* dofs = df;

    if (linkCount > 1)
    {
        UnAlignedSpatialVector nextMotionV = loadSpatialVector(lk[1].mMotionVelocity, lane);
        PxU32 nextNbDofs = lk[1].mDofs[lane];
        PxU32 nextParent = lk[1].mParents[lane];

        for (PxU32 i = 1; i < linkCount; i++)
        {
            physx::PxgArticulationBlockLinkData& tLink = lk[i];
            const PxU32 nbDofs = nextNbDofs;
            const PxU32 parent = nextParent;

            const UnAlignedSpatialVector preloadedConstraintForces =
                loadSpatialVector(tLink.mConstraintForces, lane);
            const UnAlignedSpatialVector preloadedSolverSpatialDeltaVel =
                loadSpatialVector(tLink.mSolverSpatialDeltaVel, lane);

            UnAlignedSpatialVector motionV = nextMotionV;

            if ((i + 1) < linkCount)
            {
                nextMotionV = loadSpatialVector(lk[i + 1].mMotionVelocity, lane);
                nextNbDofs = lk[i + 1].mDofs[lane];
                nextParent = lk[i + 1].mParents[lane];
            }

            if (parent != (i - 1))
                deltaV = loadSpatialVector(lk[parent].mScratchDeltaV, lane);

            deltaV = fvPropagateAccel(tLink, dofs, nbDofs, deltaV, lane);

            storeSpatialVector(tLink.mSolverSpatialDeltaVel,
                               preloadedSolverSpatialDeltaVel + deltaV, lane);

            for (PxU32 ind = 0; ind < nbDofs; ++ind)
                dofs[ind].mDeferredQstZ[lane] = 0.f;

            motionV += deltaV;

            storeSpatialVector(tLink.mScratchDeltaV, deltaV, lane);
            storeSpatialVector(tLink.mMotionVelocity, motionV, lane);
            storeSpatialVector(tLink.mConstraintForces, preloadedConstraintForces + deltaV, lane);

            dofs += nbDofs;
        }
    }
}

inline UnAlignedSpatialVector aiPropagateImpulse(const PxVec3& parentToChild,
                                                 physx::PxgArticulationBlockDofData* dofData,
                                                 const UnAlignedSpatialVector& Z, PxU32 dofCount,
                                                 uint32_t lane)
{
    UnAlignedSpatialVector temp = Z;
    for (PxU32 ind = 0; ind < 3; ++ind)
    {
        if (ind < dofCount)
        {
            const UnAlignedSpatialVector sas = loadSpatialVector(dofData[ind].mWorldMotionMatrix,
                                                                 lane);
            const UnAlignedSpatialVector isInvD = loadSpatialVector(dofData[ind].mIsInvDW, lane);
            const PxReal stZ = 0.0f - sas.innerProduct(Z);
            dofData[ind].mDeferredQstZ[lane] += stZ;
            temp += isInvD * stZ;
        }
    }
    return physx::Dy::FeatherstoneArticulation::translateSpatialVector(parentToChild, temp);
}

inline void aiMirror(uint32_t* isSlabDirty, float* impulses,
                     physx::PxgArticulationBlockData* ab,
                     physx::PxgArticulationBlockLinkData* lk,
                     physx::PxgArticulationBlockDofData* df, uint32_t articulationId,
                     uint32_t maxLinks, uint32_t nbArticulations, uint32_t nbSlabs, uint32_t lane,
                     uint32_t& alignedHits, uint32_t& unalignedHits)
{
    const PxU32 slabStepSize = maxLinks * nbArticulations;

    PxReal scale = 1.0f;
    if (nbSlabs > 1)
    {
        PxReal count = 0.0f;
        for (PxU32 s = 0, di = articulationId; s < nbSlabs; ++s, di += nbArticulations)
        {
            if (isSlabDirty[di * 2 + 0] != AI_NONE || isSlabDirty[di * 2 + 1] != AI_NONE)
                count += 1.0f;
        }
        scale = count > 1.0f ? 1.0f / count : 1.0f;
    }

    PxU32 maxIndex = ab->mLinkWithDeferredImpulse[lane];

    for (PxU32 s = 0, slabOffset = articulationId, di = articulationId; s < nbSlabs;
         ++s, slabOffset += slabStepSize, di += nbArticulations)
    {
        const PxU32 dx = isSlabDirty[di * 2 + 0];
        const PxU32 dy = isSlabDirty[di * 2 + 1];
        for (PxU32 i = 0, dirtyIndex = dx; i < 2; ++i, dirtyIndex = dy)
        {
            if (dirtyIndex != AI_NONE)
            {
                const UnAlignedSpatialVector preloadedScratchImpulse =
                    loadSpatialVector(lk[dirtyIndex].mScratchImpulse, lane);

                const PxU32 deltaIdx = slabOffset + dirtyIndex * nbArticulations;

                // THE BRANCH, kept literally -- see the header
                UnAlignedSpatialVector impulse;
                float* base = impulses + size_t(deltaIdx) * 6;
                const size_t ptr = reinterpret_cast<size_t>(base);
                if (ptr & 0xf)
                {
                    ++unalignedHits;
                    float* f2 = base;
                    float* f4 = base + 2;
                    impulse.top.x = f2[0]; impulse.top.y = f2[1]; impulse.top.z = f4[0];
                    impulse.bottom.x = f4[1]; impulse.bottom.y = f4[2]; impulse.bottom.z = f4[3];
                    for (int z = 0; z < 4; ++z) f4[z] = 0.f;
                    for (int z = 0; z < 2; ++z) f2[z] = 0.f;
                }
                else
                {
                    ++alignedHits;
                    float* f4 = base;
                    float* f2 = base + 4;
                    impulse.top.x = f4[0]; impulse.top.y = f4[1]; impulse.top.z = f4[2];
                    impulse.bottom.x = f4[3]; impulse.bottom.y = f2[0]; impulse.bottom.z = f2[1];
                    for (int z = 0; z < 4; ++z) f4[z] = 0.f;
                    for (int z = 0; z < 2; ++z) f2[z] = 0.f;
                }

                storeSpatialVector(lk[dirtyIndex].mScratchImpulse,
                                   preloadedScratchImpulse + impulse * scale, lane);
                maxIndex = physx::PxMax(dirtyIndex, maxIndex);
            }
        }
        isSlabDirty[di * 2 + 0] = AI_NONE;
        isSlabDirty[di * 2 + 1] = AI_NONE;
    }

    const UnAlignedSpatialVector preloadedRootDeferredZ = loadSpatialVector(ab->mRootDeferredZ,
                                                                           lane);
    if (maxIndex)
    {
        physx::PxgArticulationBlockDofData* dofData = &df[lk[maxIndex].mJointOffset[lane]];

        PxU32 nextParent = lk[maxIndex].mParents[lane];
        PxU32 nextDofs = lk[maxIndex].mDofs[lane];
        float nx = lk[maxIndex].mRw_x[lane];
        float ny = lk[maxIndex].mRw_y[lane];
        float nz = lk[maxIndex].mRw_z[lane];

        for (PxU32 linkID = maxIndex; linkID > 0; linkID--)
        {
            physx::PxgArticulationBlockLinkData& linkData = lk[linkID];

            const UnAlignedSpatialVector Z = loadSpatialVector(linkData.mScratchImpulse, lane);
            const UnAlignedSpatialVector solverSpatialImpulse =
                loadSpatialVector(linkData.mSolverSpatialImpulse, lane);

            const PxU32 parent = nextParent;
            const UnAlignedSpatialVector parentScratchImpulse =
                loadSpatialVector(lk[parent].mScratchImpulse, lane);
            const float cx = nx, cy = ny, cz = nz;
            const PxU32 dofCount = nextDofs;

            if (linkID > 1)
            {
                const PxU32 nextIndex = linkID - 1;
                nextParent = lk[nextIndex].mParents[lane];
                nextDofs = lk[nextIndex].mDofs[lane];
                nx = lk[nextIndex].mRw_x[lane];
                ny = lk[nextIndex].mRw_y[lane];
                nz = lk[nextIndex].mRw_z[lane];
            }

            const UnAlignedSpatialVector propagatedZ =
                aiPropagateImpulse(PxVec3(cx, cy, cz), dofData, Z, dofCount, lane);

            storeSpatialVector(linkData.mSolverSpatialImpulse, solverSpatialImpulse + Z, lane);
            storeSpatialVector(lk[parent].mScratchImpulse, parentScratchImpulse + propagatedZ,
                               lane);
            storeSpatialVector(linkData.mScratchImpulse, UnAlignedSpatialVector::Zero(), lane);

            dofData -= nextDofs;
        }
    }
    const UnAlignedSpatialVector preloadedRootScratchImpulse =
        loadSpatialVector(lk[0].mScratchImpulse, lane);
    storeSpatialVector(lk[0].mScratchImpulse, UnAlignedSpatialVector::Zero(), lane);
    storeSpatialVector(ab->mRootDeferredZ, preloadedRootScratchImpulse + preloadedRootDeferredZ,
                       lane);
}

}  // namespace pxgref

#endif
