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

// pxgArtiInternalRef.h
//
// The internal-constraint solve mirrors (P8.100a/b), hoisted from
// testArtiSolveStatic.cpp / testArtiSolveDof.cpp (parameterization is the
// whole diff) so the DFS fixture (P8.100c) composes them:
//  * refSolveDofLane -- one lane of the per-dof internal solve (:1747-1890)
//    over the ENGINE oracles Dy::computeDriveImpulse / clampDriveImpulse /
//    computeFrictionImpulse / computeLimitImpulse / getImplicitDriveDesc;
//  * refSolveStaticLane -- one lane of the static solveStaticConstraints
//    (TGS overload, :1395-1552) over the pxgArtiExtSolveRef mirrors.

#ifndef PXG_ARTI_INTERNAL_REF_H
#define PXG_ARTI_INTERNAL_REF_H

#include "pxgArtiRefHelpers.h"
#include "pxgArtiSolveRef.h"
#include "pxgArtiExtSolveRef.h"
#include "DyCpuGpuArticulation.h"

#include <cstdint>

namespace pxgtest
{
using namespace physx;
using physx::Cm::UnAlignedSpatialVector;

struct DofCov
{
    PxU32 locked = 0, limited = 0, freeDof = 0, oldModel = 0, newModel = 0;
    PxU32 maxArm = 0, forceArm = 0, velHits = 0;
};

struct StaticCov
{
    PxU32 aSide = 0, bSide = 0;
};

inline void refSolveDofLane(PxgArticulationBlockDofData* df, PxU32 jointOffset, PxU32 dofCount,
                            const float* jf,
                            float dt, float sceneDt, float invDt, float elapsedTime, float erp,
                            bool velIter, bool extEvery, bool doFDPL, bool velBefore,
                            const UnAlignedSpatialVector& childDelta,
                            const UnAlignedSpatialVector& parentDelta,
                            UnAlignedSpatialVector& parentDeltaV, UnAlignedSpatialVector& deltaV,
                            UnAlignedSpatialVector& parentV, UnAlignedSpatialVector& childV,
                            UnAlignedSpatialVector& parentImp, UnAlignedSpatialVector& impulse,
                            bool residual, RefError& err, PxU32 lane, DofCov* cov)
{
    for (PxU32 dof = 0; dof < dofCount; ++dof)
    {
        PxgArticulationBlockDofData& D = df[jointOffset + dof];
        PxgArticulationInternalConstraintData& ic = D.mConstraintData;
        const PxU32 motion = D.mMotion[lane];
        if (motion == PxArticulationMotion::eLOCKED) { if (cov) ++cov->locked; continue; }
        if (cov) { if (motion == PxArticulationMotion::eLIMITED) ++cov->limited; else ++cov->freeDof; }

        const float maxJointVel = ic.mMaxJointVelocity[lane];
        const UnAlignedSpatialVector row0 = loadSpatialVector(ic.mRow0, lane);
        const UnAlignedSpatialVector row1 = loadSpatialVector(ic.mRow1, lane);
        const float maxDriveForce = ic.mConstraintMaxImpulse[lane];
        const float appliedDriveImpulseOrig = ic.mDriveImpulse[lane];
        const float recipResponse = ic.mRecipResponse[lane];
        const float response = ic.mResponse[lane];
        const float maxFrictionForce = ic.mMaxFrictionForce[lane];
        const UnAlignedSpatialVector dVA = loadSpatialVector(ic.mDeltaVA, lane);
        const UnAlignedSpatialVector dVB = loadSpatialVector(ic.mDeltaVB, lane);

        const bool perSubset = extEvery && !velIter;
        const float effectiveDt = perSubset ? dt : sceneDt;

        const float staticFrictionImpulse = ic.mStaticFrictionEffort[lane] * effectiveDt;
        const float dynamicFrictionImpulse = ic.mDynamicFrictionEffort[lane] * effectiveDt;
        const float viscousFrictionCoefficient = ic.mViscousFrictionCoefficient[lane] * effectiveDt;
        const float maxImpulse = ic.mMaxEffort[lane] * effectiveDt;
        const float speedImpulseGradient = ic.mSpeedEffortGradient[lane] / effectiveDt;
        const float velocityDependentResistance = ic.mVelocityDependentResistance[lane] * effectiveDt;
        const float externalJointImpulse = jf[jointOffset + dof] * effectiveDt;
        const float maxActuatorVelocity = ic.mMaxActuatorVelocity[lane];

        Dy::ArticulationImplicitDriveDesc driveDesc(PxZero);
        if (!velIter)
            driveDesc = ic.getImplicitDriveDesc(lane);

        float jointV = row1.innerProduct(childV) - row0.innerProduct(parentV);
        const float jointDeltaP = row1.innerProduct(childDelta) - row0.innerProduct(parentDelta);

        float frictionDeltaF = 0.0f, driveDeltaF = 0.0f, posLimitDeltaF = 0.0f;
        if (doFDPL)
        {
            const bool newFrictionModel = staticFrictionImpulse != 0.0f || viscousFrictionCoefficient != 0.0f;
            if (!newFrictionModel)
            {
                if (cov) ++cov->oldModel;
                const float appliedFriction = 0.0f;   // TGS
                const float frictionForce = PxClamp(-jointV * recipResponse + appliedFriction,
                                                    -maxFrictionForce, maxFrictionForce);
                ic.mAccumulatedFrictionImpulse[lane] = frictionForce;
                frictionDeltaF = frictionForce - appliedFriction;
                jointV += frictionDeltaF * response;
            }

            if (maxImpulse > 0.0f)
            {
                if (cov) ++cov->maxArm;
                const float appliedDriveImpulse = perSubset ? 0.0f : appliedDriveImpulseOrig;
                const float unclampedImpulse = velIter ? appliedDriveImpulse
                    : Dy::computeDriveImpulse(appliedDriveImpulse, jointV, jointDeltaP, elapsedTime, driveDesc);
                const float clampedImpulse = Dy::clampDriveImpulse(jointV,
                    appliedDriveImpulse + externalJointImpulse, unclampedImpulse + externalJointImpulse,
                    response, maxActuatorVelocity, maxImpulse, speedImpulseGradient,
                    velocityDependentResistance) - externalJointImpulse;
                driveDeltaF = clampedImpulse - appliedDriveImpulse;
                ic.mDriveImpulse[lane] = appliedDriveImpulseOrig + driveDeltaF;
            }
            else
            {
                if (cov) ++cov->forceArm;
                const float unclampedForce = velIter ? appliedDriveImpulseOrig
                    : Dy::computeDriveImpulse(appliedDriveImpulseOrig, jointV, jointDeltaP, elapsedTime, driveDesc);
                const float clampedForce = PxClamp(unclampedForce, -maxDriveForce, maxDriveForce);
                driveDeltaF = clampedForce - appliedDriveImpulseOrig;
                ic.mDriveImpulse[lane] = clampedForce;
            }
            jointV += driveDeltaF * response;

            if (newFrictionModel)
            {
                if (cov) ++cov->newModel;
                const float appliedFriction = perSubset ? 0.0f : ic.mAccumulatedFrictionImpulse[lane];
                float totalImpulse = appliedFriction - jointV * recipResponse;
                totalImpulse = Dy::computeFrictionImpulse(totalImpulse, staticFrictionImpulse,
                                                          dynamicFrictionImpulse,
                                                          viscousFrictionCoefficient, jointV);
                frictionDeltaF = totalImpulse - appliedFriction;
                ic.mAccumulatedFrictionImpulse[lane] += frictionDeltaF;
                jointV += frictionDeltaF * response;
            }

            if (motion == PxArticulationMotion::eLIMITED)
            {
                const float errorLow = ic.mLimitError_LowX_highY[lane].x;
                const float errorHigh = ic.mLimitError_LowX_highY[lane].y;
                posLimitDeltaF = Dy::computeLimitImpulse(dt, invDt, velIter,
                                                         response, recipResponse, erp,
                                                         errorLow, errorHigh, jointDeltaP,
                                                         ic.mLowImpulse[lane],
                                                         ic.mHighImpulse[lane], jointV);
            }
        }

        float velLimitDeltaF = 0.0f;
        if (velBefore && PxAbs(jointV) > maxJointVel)
        {
            if (cov) ++cov->velHits;
            const float newJointV = PxClamp(jointV, -maxJointVel, maxJointVel);
            velLimitDeltaF = (newJointV - jointV) * recipResponse;
            jointV = newJointV;
        }

        const float deltaF = frictionDeltaF + driveDeltaF + posLimitDeltaF + velLimitDeltaF;
        if (residual)
            err.add(deltaF, recipResponse);

        // accumulateLinkImpulsesAndLinkVelocities (:1554)
        const UnAlignedSpatialVector pDelta = dVA * -deltaF;
        const UnAlignedSpatialVector cDelta = dVB * -deltaF;
        parentDeltaV += pDelta;
        deltaV += cDelta;
        parentV += pDelta;
        childV += cDelta;
        parentImp += row0 * deltaF;
        impulse -= row1 * deltaF;
    }
}

inline void refSolveStaticLane(const PxgArticulationBlockLinkData& L, PxU32 lane,
                               PxgBlockConstraintBatch* bt,
                               PxgTGSBlockSolverContactHeader* ch,
                               PxgTGSBlockSolverFrictionHeader* fh,
                               PxgTGSBlockSolverContactPoint* cp,
                               PxgTGSBlockSolverContactFriction* fp,
                               PxgTGSBlockSolverConstraint1DHeader* jh,
                               PxgTGSBlockSolverConstraint1DCon* jr,
                               PxgArticulationBlockResponse* rs,
                               float elapsedTime, float minPen, bool accumErr,
                               UnAlignedSpatialVector& vel, UnAlignedSpatialVector& impulse,
                               UnAlignedSpatialVector& deltaV,
                               RefError& err, PxU32 constraintCounts, StaticCov* cov)
{
    const PxU32 jStart = L.mStaticJointStartIndex[lane];
    const PxU32 cStart = L.mStaticContactStartIndex[lane];
    const PxU32 contactCounts = L.mNbStaticContacts[lane];

    const UnAlignedSpatialVector oldVel = vel;
    UnAlignedSpatialVector delta = loadSpatialVector(L.mDeltaMotion, lane);
    PxQuat deltaQ(PxIdentity);
    if (constraintCounts > 0)
    {
        const float4 q = L.mDeltaQ[lane];
        deltaQ = PxQuat(q.x, q.y, q.z, q.w);
    }

    for (PxU32 i = 0; i < constraintCounts; ++i)
    {
        PxgBlockConstraintBatch& batch = bt[jStart + i];
        const PxU32 offset = PxU32(__builtin_popcount(batch.mask & ((1u << lane) - 1u)));
        const bool aIsArti = batch.bodyANodeIndex[offset].isArticulation() != 0;
        if (cov) { if (aIsArti) ++cov->aSide; else ++cov->bSide; }

        UnAlignedSpatialVector vel0, vel1, delta0, delta1;
        PxQuat q0, q1;
        if (aIsArti) { vel0 = vel; delta0 = delta; q0 = deltaQ;
                       vel1 = UnAlignedSpatialVector::Zero(); delta1 = UnAlignedSpatialVector::Zero(); q1 = PxQuat(PxIdentity); }
        else         { vel0 = UnAlignedSpatialVector::Zero(); delta0 = UnAlignedSpatialVector::Zero(); q0 = PxQuat(PxIdentity);
                       vel1 = vel; delta1 = delta; q1 = deltaQ; }
        UnAlignedSpatialVector imp0(PxVec3(0.f), PxVec3(0.f)), imp1(PxVec3(0.f), PxVec3(0.f));
        refSolveExt1DTGS(batch, vel0, vel1, delta0, delta1, offset,
                         jh, jr, rs, q0, q1, elapsedTime, imp0, imp1, accumErr, 1.f, 1.f);
        if (aIsArti) { impulse += imp0; vel = vel0; delta = delta0; deltaQ = q0; }
        else         { impulse += imp1; vel = vel1; delta = delta1; deltaQ = q1; }
    }
    for (PxU32 i = 0; i < contactCounts; ++i)
    {
        PxgBlockConstraintBatch& batch = bt[cStart + i];
        const PxU32 offset = PxU32(__builtin_popcount(batch.mask & ((1u << lane) - 1u)));
        const bool aIsArti = batch.bodyANodeIndex[offset].isArticulation() != 0;
        if (cov) { if (aIsArti) ++cov->aSide; else ++cov->bSide; }

        UnAlignedSpatialVector vel0, vel1, delta0, delta1;
        PxQuat q0, q1;
        if (aIsArti) { vel0 = vel; delta0 = delta; q0 = deltaQ;
                       vel1 = UnAlignedSpatialVector::Zero(); delta1 = UnAlignedSpatialVector::Zero(); q1 = PxQuat(PxIdentity); }
        else         { vel0 = UnAlignedSpatialVector::Zero(); delta0 = UnAlignedSpatialVector::Zero(); q0 = PxQuat(PxIdentity);
                       vel1 = vel; delta1 = delta; q1 = deltaQ; }
        UnAlignedSpatialVector imp0(PxVec3(0.f), PxVec3(0.f)), imp1(PxVec3(0.f), PxVec3(0.f));
        refSolveExtContactTGS(batch, vel0, vel1, delta0, delta1, offset,
                              ch, fh, cp, fp, rs, elapsedTime, minPen, imp0, imp1,
                              accumErr ? &err : NULL, 1.f, 1.f);
        if (aIsArti) { impulse += imp0; vel = vel0; delta = delta0; deltaQ = q0; }
        else         { impulse += imp1; vel = vel1; delta = delta1; deltaQ = q1; }
    }

    if (constraintCounts + contactCounts > 0)
        deltaV += vel - oldVel;
}

}  // namespace pxgtest

#endif  // PXG_ARTI_INTERNAL_REF_H
