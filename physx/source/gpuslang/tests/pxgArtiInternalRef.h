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
#include <cstring>

namespace pxgtest
{
using namespace physx;
using physx::Cm::UnAlignedSpatialVector;

// P9.86: TGS old-model friction evaluations whose `-jointV * recipResponse` is -0,
// where the CUDA's `+ appliedFriction` (a literal 0) turns it into the stored +0.
inline PxU32 gTgsOldFrictionNegZero = 0;

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
                { const float p = -jointV * recipResponse; PxU32 pb; std::memcpy(&pb, &p, 4);
                  if (pb == 0x80000000u) ++gTgsOldFrictionNegZero; }
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

// ===========================================================================
// P8.123b: the PGS per-dof solve mirror (the dof loop with isTGS == false).
// Divergences from refSolveDofLane, each transcribed from the template:
//  * perSubset = isTGS && extEvery && !velIter (:1765) is FALSE always, so
//    effectiveDt is ALWAYS the scene dt and extEvery is dead (param dropped);
//  * the implicit drive desc is loaded UNCONDITIONALLY (:1778);
//  * the deprecated friction model reads the ACCUMULATED impulse (:1796);
//  * BOTH drive arms call computeDriveImpulse even on velocity iterations
//    (:1811/:1832 — the isVelIter shortcut is TGS-only), so elapsedTime is
//    live on every iteration type;
//  * the new friction model reads the accumulated impulse always (:1845).
// ===========================================================================

inline void refSolveDofLanePgs(PxgArticulationBlockDofData* df, PxU32 jointOffset, PxU32 dofCount,
                               const float* jf,
                               float dt, float sceneDt, float invDt, float elapsedTime, float erp,
                               bool velIter, bool doFDPL, bool velBefore,
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

        const float effectiveDt = sceneDt;   // perSubset is false on PGS

        const float staticFrictionImpulse = ic.mStaticFrictionEffort[lane] * effectiveDt;
        const float dynamicFrictionImpulse = ic.mDynamicFrictionEffort[lane] * effectiveDt;
        const float viscousFrictionCoefficient = ic.mViscousFrictionCoefficient[lane] * effectiveDt;
        const float maxImpulse = ic.mMaxEffort[lane] * effectiveDt;
        const float speedImpulseGradient = ic.mSpeedEffortGradient[lane] / effectiveDt;
        const float velocityDependentResistance = ic.mVelocityDependentResistance[lane] * effectiveDt;
        const float externalJointImpulse = jf[jointOffset + dof] * effectiveDt;
        const float maxActuatorVelocity = ic.mMaxActuatorVelocity[lane];

        // loaded UNCONDITIONALLY on PGS (:1778)
        Dy::ArticulationImplicitDriveDesc driveDesc(PxZero);
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
                const float appliedFriction = ic.mAccumulatedFrictionImpulse[lane];   // PGS accumulates
                const float frictionForce = PxClamp(-jointV * recipResponse + appliedFriction,
                                                    -maxFrictionForce, maxFrictionForce);
                ic.mAccumulatedFrictionImpulse[lane] = frictionForce;
                frictionDeltaF = frictionForce - appliedFriction;
                jointV += frictionDeltaF * response;
            }

            if (maxImpulse > 0.0f)
            {
                if (cov) ++cov->maxArm;
                const float appliedDriveImpulse = appliedDriveImpulseOrig;   // no perSubset zero
                const float unclampedImpulse =
                    Dy::computeDriveImpulse(appliedDriveImpulse, jointV, jointDeltaP, elapsedTime, driveDesc);
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
                const float unclampedForce =
                    Dy::computeDriveImpulse(appliedDriveImpulseOrig, jointV, jointDeltaP, elapsedTime, driveDesc);
                const float clampedForce = PxClamp(unclampedForce, -maxDriveForce, maxDriveForce);
                driveDeltaF = clampedForce - appliedDriveImpulseOrig;
                ic.mDriveImpulse[lane] = clampedForce;
            }
            jointV += driveDeltaF * response;

            if (newFrictionModel)
            {
                if (cov) ++cov->newModel;
                const float appliedFriction = ic.mAccumulatedFrictionImpulse[lane];   // PGS accumulates
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

// ===========================================================================
// P8.123a: the PGS static-solve mirror (solveStaticConstraints PGS overload,
// internalConstraints2.cu:1268-1392) over the P8.122a PGS ext-solve mirrors.
// Divergences from refSolveStaticLane: no delta/deltaQ, doFriction live,
// 1D carries rowsMod with the response base OFFSET by
// batch.mArticulationResponseIndex, minPen/elapsedTime dead (dropped).
// ===========================================================================

inline void refSolveStaticLanePgs(const PxgArticulationBlockLinkData& L, PxU32 lane,
                                  PxgBlockConstraintBatch* bt,
                                  PxgBlockSolverContactHeader* ch,
                                  PxgBlockSolverFrictionHeader* fh,
                                  PxgBlockSolverContactPoint* cp,
                                  PxgBlockSolverContactFriction* fp,
                                  PxgBlockSolverConstraint1DHeader* jh,
                                  PxgBlockSolverConstraint1DCon* jr,
                                  PxgBlockSolverConstraint1DMod* jm,
                                  PxgArticulationBlockResponse* rs,
                                  bool doFriction, bool accumErr,
                                  UnAlignedSpatialVector& vel, UnAlignedSpatialVector& impulse,
                                  UnAlignedSpatialVector& deltaV,
                                  RefError& err, PxU32 constraintCounts, StaticCov* cov)
{
    const PxU32 jStart = L.mStaticJointStartIndex[lane];
    const PxU32 cStart = L.mStaticContactStartIndex[lane];
    const PxU32 contactCounts = L.mNbStaticContacts[lane];

    const UnAlignedSpatialVector oldVel = vel;

    for (PxU32 i = 0; i < constraintCounts; ++i)
    {
        PxgBlockConstraintBatch& batch = bt[jStart + i];
        const PxU32 offset = PxU32(__builtin_popcount(batch.mask & ((1u << lane) - 1u)));
        const bool aIsArti = batch.bodyANodeIndex[offset].isArticulation() != 0;
        if (cov) { if (aIsArti) ++cov->aSide; else ++cov->bSide; }

        UnAlignedSpatialVector vel0, vel1;
        if (aIsArti) { vel0 = vel; vel1 = UnAlignedSpatialVector::Zero(); }
        else         { vel0 = UnAlignedSpatialVector::Zero(); vel1 = vel; }
        UnAlignedSpatialVector imp0(PxVec3(0.f), PxVec3(0.f)), imp1(PxVec3(0.f), PxVec3(0.f));
        refSolveExt1DPGS(batch, vel0, vel1, offset,
                         jh, jr, jm, &rs[batch.mArticulationResponseIndex],
                         imp0, imp1, accumErr, 1.f, 1.f);
        if (aIsArti) { impulse += imp0; vel = vel0; }
        else         { impulse += imp1; vel = vel1; }
    }
    for (PxU32 i = 0; i < contactCounts; ++i)
    {
        PxgBlockConstraintBatch& batch = bt[cStart + i];
        const PxU32 offset = PxU32(__builtin_popcount(batch.mask & ((1u << lane) - 1u)));
        const bool aIsArti = batch.bodyANodeIndex[offset].isArticulation() != 0;
        if (cov) { if (aIsArti) ++cov->aSide; else ++cov->bSide; }

        UnAlignedSpatialVector vel0, vel1;
        if (aIsArti) { vel0 = vel; vel1 = UnAlignedSpatialVector::Zero(); }
        else         { vel0 = UnAlignedSpatialVector::Zero(); vel1 = vel; }
        UnAlignedSpatialVector imp0(PxVec3(0.f), PxVec3(0.f)), imp1(PxVec3(0.f), PxVec3(0.f));
        refSolveExtContactPGS(batch, vel0, vel1, doFriction, offset,
                              ch, fh, cp, fp, rs, imp0, imp1,
                              accumErr ? &err : NULL, 1.f, 1.f);
        if (aIsArti) { impulse += imp0; vel = vel0; }
        else         { impulse += imp1; vel = vel1; }
    }

    if (constraintCounts + contactCounts > 0)
        deltaV += vel - oldVel;
}

// ===========================================================================
// P8.100d hoist: the siSolveInternalDFS mirror, moved VERBATIM from
// testArtiSolveDFS.cpp (second user: testArtiSolveInternalLaunch.cpp) — the
// parameterization is the whole diff.  Needs refPropAccelDeferred /
// refPropImpulseDeferred (pxgArtiSolveRef.h) included first.
// ===========================================================================

// the AFTER-pass's no-write impulse hop (WriteToProvidedQstZ with NULL): the
// template spelling `0 - dot`, dof rows loaded and the qstZ sum DISCARDED
inline Cm::UnAlignedSpatialVector refPropImpulseNoWrite(
    const PxgArticulationBlockDofData* df, PxU32 dofBase, PxU32 dofCount,
    const PxVec3& c2p, const Cm::UnAlignedSpatialVector& Z, PxU32 lane)
{
    Cm::UnAlignedSpatialVector temp = Z;
    for (PxU32 ind = 0; ind < 3; ++ind)
    {
        if (ind < dofCount)
        {
            const PxgArticulationBlockDofData& dd = df[dofBase + ind];
            const Cm::UnAlignedSpatialVector sas = loadSpatialVector(dd.mWorldMotionMatrix, lane);
            const Cm::UnAlignedSpatialVector isInvD = loadSpatialVector(dd.mIsInvDW, lane);
            const PxReal dot = sas.innerProduct(Z);
            PxReal stZ = 0.0f - dot;
            {   // P9.86d twin + counter (pxgArtiSolveRef.h)
                PxU32 b;
                std::memcpy(&b, &dot, 4);
                if (b == 0u)
                    ++gPropImpNullJfPosZeroDot;
                if (gPropImpNullJfNegDot)
                    stZ = -dot;
            }
            temp += isInvD * stZ;
        }
    }
    return Dy::FeatherstoneArticulation::translateSpatialVector(c2p, temp);
}

// one lane of the DFS (internalConstraints2.cu:1662-2002); returns the final
// parentImp (what the epilogue adds onto mRootDeferredZ)
inline Cm::UnAlignedSpatialVector refSolveInternalDFSLane(
    PxgArticulationBlockLinkData* lk, PxgArticulationBlockDofData* df,
    PxgArticulationTraversalStackData* stack, const float* jf,
    PxgBlockConstraintBatch* bt,
    PxgTGSBlockSolverContactHeader* ch, PxgTGSBlockSolverFrictionHeader* fh,
    PxgTGSBlockSolverContactPoint* cp, PxgTGSBlockSolverContactFriction* fp,
    PxgTGSBlockSolverConstraint1DHeader* jh, PxgTGSBlockSolverConstraint1DCon* jr,
    PxgArticulationBlockResponse* rs,
    float dt, float sceneDt, float invDt, float elapsedTime, float erp, float minPen,
    bool velIter, bool extEvery, bool doFDPL, uint32_t doVelLimit, bool doStatic,
    bool residual,
    const Cm::UnAlignedSpatialVector& storeParentDeltaV,
    Cm::UnAlignedSpatialVector parentImp, Cm::UnAlignedSpatialVector parentDeltaV,
    RefError& err, RefError& contactErr, PxU32 lane)
{
    using Cm::UnAlignedSpatialVector;

    uint32_t parent = 0;
    storeSpatialVector(stack[parent].deltaVStack, storeParentDeltaV, lane);
    storeSpatialVector(stack[parent].impulseStack, parentImp, lane);

    uint32_t offset = lk[parent].mChildrenOffset[lane];
    uint32_t numChildren = lk[parent].mNumChildren[lane];
    for (uint32_t i = 0; i < numChildren; ++i)
        stack[i].indices[lane] = offset + i;
    uint32_t stackCount = numChildren;
    uint32_t linkID = stackCount != 0 ? stack[stackCount - 1].indices[lane] : 0xffffffffu;

    while (stackCount != 0)
    {
        const bool isBackProp = (parent == linkID);
        PxgArticulationBlockLinkData& L = lk[linkID];
        const PxVec3 c2p(L.mRw_x[lane], L.mRw_y[lane], L.mRw_z[lane]);
        const uint32_t jointOffset = L.mJointOffset[lane];
        const uint32_t dofCount = L.mDofs[lane];

        if (isBackProp)
        {
            parent = L.mParents[lane];
            const UnAlignedSpatialVector solverSpatialImpulse = loadSpatialVector(L.mSolverSpatialImpulse, lane);
            const UnAlignedSpatialVector ZInternalConstraint =
                loadSpatialVector(L.mSolverSpatialInternalConstraintImpulse, lane);
            PxSpatialMatrix spatialResponse;
            loadSpatialMatrix(lk[parent].mSpatialResponseMatrix, lane, spatialResponse);

            stackCount--;
            const UnAlignedSpatialVector impulse = parentImp;
            parentDeltaV = loadSpatialVector(stack[parent].deltaVStack, lane);
            parentImp = loadSpatialVector(stack[parent].impulseStack, lane);

            storeSpatialVector(L.mSolverSpatialImpulse,
                               solverSpatialImpulse + impulse - ZInternalConstraint, lane);

            linkID = stackCount > 0 ? stack[stackCount - 1].indices[lane] : 0xffffffffu;

            const UnAlignedSpatialVector propagateImp =
                refPropImpulseDeferred(df, jointOffset, dofCount, c2p, impulse, NULL, 1.f, lane);
            parentImp += propagateImp;

            const UnAlignedSpatialVector deltaV = spatialResponse * -parentImp;
            parentDeltaV += deltaV;

            storeSpatialVector(stack[parent].impulseStack, parentImp, lane);
        }
        else
        {
            UnAlignedSpatialVector parentV =
                parentDeltaV + loadSpatialVector(lk[parent].mMotionVelocity, lane);
            const UnAlignedSpatialVector childDelta = loadSpatialVector(L.mDeltaMotion, lane);
            const UnAlignedSpatialVector parentDelta = loadSpatialVector(lk[parent].mDeltaMotion, lane);

            UnAlignedSpatialVector deltaV =
                refPropAccelDeferred(df, jointOffset, dofCount, c2p, parentDeltaV, NULL, lane);
            UnAlignedSpatialVector childV = deltaV + loadSpatialVector(L.mMotionVelocity, lane);

            storeSpatialVector(stack[linkID].deltaVStack, deltaV, lane);

            UnAlignedSpatialVector impulse(PxVec3(0.f), PxVec3(0.f));

            refSolveDofLane(df, jointOffset, dofCount, jf,
                            dt, sceneDt, invDt, elapsedTime, erp,
                            velIter, extEvery, doFDPL, doVelLimit == 1u,
                            childDelta, parentDelta,
                            parentDeltaV, deltaV, parentV, childV, parentImp, impulse,
                            residual, err, lane, NULL);

            const uint32_t constraintCounts = L.mNbStaticJoints[lane];
            numChildren = L.mNumChildren[lane];

            UnAlignedSpatialVector childImpulseInternal = impulse;
            UnAlignedSpatialVector childImpulseFromStatic(PxVec3(0.f), PxVec3(0.f));
            if (doStatic)
            {
                const UnAlignedSpatialVector before = impulse;
                refSolveStaticLane(L, lane, bt, ch, fh, cp, fp, jh, jr, rs,
                                   elapsedTime, minPen, residual,
                                   childV, impulse, deltaV, contactErr, constraintCounts, NULL);
                childImpulseFromStatic = impulse - before;
            }

            if (doVelLimit == 2u)
            {
                const UnAlignedSpatialVector i1BeforeVelLimit = impulse;
                const UnAlignedSpatialVector propagatedImpulseAtParentW =
                    refPropImpulseNoWrite(df, 0, dofCount, c2p, childImpulseFromStatic, lane);
                PxSpatialMatrix mat;
                loadSpatialMatrix(lk[parent].mSpatialResponseMatrix, lane, mat);
                const UnAlignedSpatialVector deltaVParent = mat * (-propagatedImpulseAtParentW);
                const UnAlignedSpatialVector parentVCorrected = parentV + deltaVParent;

                for (uint32_t dof = 0; dof < dofCount; ++dof)
                {
                    PxgArticulationBlockDofData& D = df[jointOffset + dof];
                    PxgArticulationInternalConstraintData& ic = D.mConstraintData;
                    if (D.mMotion[lane] == PxArticulationMotion::eLOCKED)
                        continue;
                    const float maxJointVel = ic.mMaxJointVelocity[lane];
                    const UnAlignedSpatialVector row0 = loadSpatialVector(ic.mRow0, lane);
                    const UnAlignedSpatialVector row1 = loadSpatialVector(ic.mRow1, lane);
                    const float recipResponse = ic.mRecipResponse[lane];
                    const UnAlignedSpatialVector dVA = loadSpatialVector(ic.mDeltaVA, lane);
                    const UnAlignedSpatialVector dVB = loadSpatialVector(ic.mDeltaVB, lane);
                    float jointV = row1.innerProduct(childV) - row0.innerProduct(parentVCorrected);

                    float velLimitDeltaF = 0.0f;
                    if (PxAbs(jointV) > maxJointVel)
                    {
                        const float newJointV = PxClamp(jointV, -maxJointVel, maxJointVel);
                        velLimitDeltaF = (newJointV - jointV) * recipResponse;
                        jointV = newJointV;
                    }
                    if (residual)
                        err.add(velLimitDeltaF, recipResponse);

                    const UnAlignedSpatialVector pDelta = dVA * -velLimitDeltaF;
                    const UnAlignedSpatialVector cDelta = dVB * -velLimitDeltaF;
                    parentDeltaV += pDelta;
                    deltaV += cDelta;
                    parentV += pDelta;
                    childV += cDelta;
                    parentImp += row0 * velLimitDeltaF;
                    impulse -= row1 * velLimitDeltaF;
                }
                childImpulseInternal += (impulse - i1BeforeVelLimit);
            }

            storeSpatialVector(L.mSolverSpatialInternalConstraintImpulse, childImpulseInternal, lane);
            storeSpatialVector(stack[parent].impulseStack, parentImp, lane);
            storeSpatialVector(stack[linkID].impulseStack, impulse, lane);

            parent = linkID;
            if (numChildren > 0)
            {
                offset = L.mChildrenOffset[lane];
                for (uint32_t i = 0; i < numChildren; ++i)
                    stack[stackCount++].indices[lane] = offset + i;
                linkID = stack[stackCount - 1].indices[lane];
            }
            parentDeltaV = deltaV;
            parentImp = impulse;
        }
    }

    return parentImp;
}

// ===========================================================================
// P8.123c: the PGS DFS mirror — the walk with the PGS dispatch
// (refSolveDofLanePgs / refSolveStaticLanePgs over the PGS pools).  minPen
// and extEvery are dead on the PGS path and dropped; doFriction is live.
// ===========================================================================

inline Cm::UnAlignedSpatialVector refSolveInternalDFSLanePgs(
    PxgArticulationBlockLinkData* lk, PxgArticulationBlockDofData* df,
    PxgArticulationTraversalStackData* stack, const float* jf,
    PxgBlockConstraintBatch* bt,
    PxgBlockSolverContactHeader* ch, PxgBlockSolverFrictionHeader* fh,
    PxgBlockSolverContactPoint* cp, PxgBlockSolverContactFriction* fp,
    PxgBlockSolverConstraint1DHeader* jh, PxgBlockSolverConstraint1DCon* jr,
    PxgBlockSolverConstraint1DMod* jm,
    PxgArticulationBlockResponse* rs,
    float dt, float sceneDt, float invDt, float elapsedTime, float erp,
    bool velIter, bool doFriction, bool doFDPL, uint32_t doVelLimit, bool doStatic,
    bool residual,
    const Cm::UnAlignedSpatialVector& storeParentDeltaV,
    Cm::UnAlignedSpatialVector parentImp, Cm::UnAlignedSpatialVector parentDeltaV,
    RefError& err, RefError& contactErr, PxU32 lane)
{
    using Cm::UnAlignedSpatialVector;

    uint32_t parent = 0;
    storeSpatialVector(stack[parent].deltaVStack, storeParentDeltaV, lane);
    storeSpatialVector(stack[parent].impulseStack, parentImp, lane);

    uint32_t offset = lk[parent].mChildrenOffset[lane];
    uint32_t numChildren = lk[parent].mNumChildren[lane];
    for (uint32_t i = 0; i < numChildren; ++i)
        stack[i].indices[lane] = offset + i;
    uint32_t stackCount = numChildren;
    uint32_t linkID = stackCount != 0 ? stack[stackCount - 1].indices[lane] : 0xffffffffu;

    while (stackCount != 0)
    {
        const bool isBackProp = (parent == linkID);
        PxgArticulationBlockLinkData& L = lk[linkID];
        const PxVec3 c2p(L.mRw_x[lane], L.mRw_y[lane], L.mRw_z[lane]);
        const uint32_t jointOffset = L.mJointOffset[lane];
        const uint32_t dofCount = L.mDofs[lane];

        if (isBackProp)
        {
            parent = L.mParents[lane];
            const UnAlignedSpatialVector solverSpatialImpulse = loadSpatialVector(L.mSolverSpatialImpulse, lane);
            const UnAlignedSpatialVector ZInternalConstraint =
                loadSpatialVector(L.mSolverSpatialInternalConstraintImpulse, lane);
            PxSpatialMatrix spatialResponse;
            loadSpatialMatrix(lk[parent].mSpatialResponseMatrix, lane, spatialResponse);

            stackCount--;
            const UnAlignedSpatialVector impulse = parentImp;
            parentDeltaV = loadSpatialVector(stack[parent].deltaVStack, lane);
            parentImp = loadSpatialVector(stack[parent].impulseStack, lane);

            storeSpatialVector(L.mSolverSpatialImpulse,
                               solverSpatialImpulse + impulse - ZInternalConstraint, lane);

            linkID = stackCount > 0 ? stack[stackCount - 1].indices[lane] : 0xffffffffu;

            const UnAlignedSpatialVector propagateImp =
                refPropImpulseDeferred(df, jointOffset, dofCount, c2p, impulse, NULL, 1.f, lane);
            parentImp += propagateImp;

            const UnAlignedSpatialVector deltaV = spatialResponse * -parentImp;
            parentDeltaV += deltaV;

            storeSpatialVector(stack[parent].impulseStack, parentImp, lane);
        }
        else
        {
            UnAlignedSpatialVector parentV =
                parentDeltaV + loadSpatialVector(lk[parent].mMotionVelocity, lane);
            const UnAlignedSpatialVector childDelta = loadSpatialVector(L.mDeltaMotion, lane);
            const UnAlignedSpatialVector parentDelta = loadSpatialVector(lk[parent].mDeltaMotion, lane);

            UnAlignedSpatialVector deltaV =
                refPropAccelDeferred(df, jointOffset, dofCount, c2p, parentDeltaV, NULL, lane);
            UnAlignedSpatialVector childV = deltaV + loadSpatialVector(L.mMotionVelocity, lane);

            storeSpatialVector(stack[linkID].deltaVStack, deltaV, lane);

            UnAlignedSpatialVector impulse(PxVec3(0.f), PxVec3(0.f));

            refSolveDofLanePgs(df, jointOffset, dofCount, jf,
                               dt, sceneDt, invDt, elapsedTime, erp,
                               velIter, doFDPL, doVelLimit == 1u,
                               childDelta, parentDelta,
                               parentDeltaV, deltaV, parentV, childV, parentImp, impulse,
                               residual, err, lane, NULL);

            const uint32_t constraintCounts = L.mNbStaticJoints[lane];
            numChildren = L.mNumChildren[lane];

            UnAlignedSpatialVector childImpulseInternal = impulse;
            UnAlignedSpatialVector childImpulseFromStatic(PxVec3(0.f), PxVec3(0.f));
            if (doStatic)
            {
                const UnAlignedSpatialVector before = impulse;
                refSolveStaticLanePgs(L, lane, bt, ch, fh, cp, fp, jh, jr, jm, rs,
                                      doFriction, residual,
                                      childV, impulse, deltaV, contactErr, constraintCounts, NULL);
                childImpulseFromStatic = impulse - before;
            }

            if (doVelLimit == 2u)
            {
                const UnAlignedSpatialVector i1BeforeVelLimit = impulse;
                const UnAlignedSpatialVector propagatedImpulseAtParentW =
                    refPropImpulseNoWrite(df, 0, dofCount, c2p, childImpulseFromStatic, lane);
                PxSpatialMatrix mat;
                loadSpatialMatrix(lk[parent].mSpatialResponseMatrix, lane, mat);
                const UnAlignedSpatialVector deltaVParent = mat * (-propagatedImpulseAtParentW);
                const UnAlignedSpatialVector parentVCorrected = parentV + deltaVParent;

                for (uint32_t dof = 0; dof < dofCount; ++dof)
                {
                    PxgArticulationBlockDofData& D = df[jointOffset + dof];
                    PxgArticulationInternalConstraintData& ic = D.mConstraintData;
                    if (D.mMotion[lane] == PxArticulationMotion::eLOCKED)
                        continue;
                    const float maxJointVel = ic.mMaxJointVelocity[lane];
                    const UnAlignedSpatialVector row0 = loadSpatialVector(ic.mRow0, lane);
                    const UnAlignedSpatialVector row1 = loadSpatialVector(ic.mRow1, lane);
                    const float recipResponse = ic.mRecipResponse[lane];
                    const UnAlignedSpatialVector dVA = loadSpatialVector(ic.mDeltaVA, lane);
                    const UnAlignedSpatialVector dVB = loadSpatialVector(ic.mDeltaVB, lane);
                    float jointV = row1.innerProduct(childV) - row0.innerProduct(parentVCorrected);

                    float velLimitDeltaF = 0.0f;
                    if (PxAbs(jointV) > maxJointVel)
                    {
                        const float newJointV = PxClamp(jointV, -maxJointVel, maxJointVel);
                        velLimitDeltaF = (newJointV - jointV) * recipResponse;
                        jointV = newJointV;
                    }
                    if (residual)
                        err.add(velLimitDeltaF, recipResponse);

                    const UnAlignedSpatialVector pDelta = dVA * -velLimitDeltaF;
                    const UnAlignedSpatialVector cDelta = dVB * -velLimitDeltaF;
                    parentDeltaV += pDelta;
                    deltaV += cDelta;
                    parentV += pDelta;
                    childV += cDelta;
                    parentImp += row0 * velLimitDeltaF;
                    impulse -= row1 * velLimitDeltaF;
                }
                childImpulseInternal += (impulse - i1BeforeVelLimit);
            }

            storeSpatialVector(L.mSolverSpatialInternalConstraintImpulse, childImpulseInternal, lane);
            storeSpatialVector(stack[parent].impulseStack, parentImp, lane);
            storeSpatialVector(stack[linkID].impulseStack, impulse, lane);

            parent = linkID;
            if (numChildren > 0)
            {
                offset = L.mChildrenOffset[lane];
                for (uint32_t i = 0; i < numChildren; ++i)
                    stack[stackCount++].indices[lane] = offset + i;
                linkID = stack[stackCount - 1].indices[lane];
            }
            parentDeltaV = deltaV;
            parentImp = impulse;
        }
    }

    return parentImp;
}

}  // namespace pxgtest

#endif  // PXG_ARTI_INTERNAL_REF_H
