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


// npArtiSolveInternal.slang
//
// P8.100a: the first slice of artiSolveInternalConstraintsTGS1T
// (internalConstraints2.cu:2842, template body :1579-2013) — the STATIC
// constraint solve, solveStaticConstraints (TGS overload, :1395-1552):
// one articulated link against the static-world batches the P8.96b prep
// wrote into the link's per-link windows (mStaticJointStartIndex /
// mStaticContactStartIndex / mNbStaticContacts; the JOINT count comes from
// the CALLER, as in the CUDA).
//
// ===========================================================================
// TRAPS (each carries a mutant in the sweep unless marked equivalent)
// ===========================================================================
//
//  1. **The joint loop runs BEFORE the contact loop**, and `vel` travels
//     through both — a batch's solve sees every earlier batch's velocity
//     update.
//  2. **The articulated link can sit on EITHER side of a static batch**
//     (bodyA or bodyB, decided per batch by PxNodeIndex::isArticulation);
//     the static side gets ZERO velocity/delta and the IDENTITY deltaQ, and
//     its impulse output is DISCARDED.  `impulse` accumulates only the
//     articulation side's output.
//  3. **The per-lane batch slot is the exclusive-popcount offset** within
//     batch.mask, exactly as in the self-solve.
//  4. **deltaQ is loaded only when the JOINT count is nonzero** — the
//     contact path never reads it.  (Loading it unconditionally would be
//     numerically equivalent — the load has no side effect — documented,
//     not swept.)
//  5. **`deltaV += vel - oldVel` fires only when any batch ran** — with both
//     counts zero, deltaV must come back untouched.
//  6. The CUDA reassigns delta/deltaQ from the chosen side after each batch;
//     the solve bodies take them BY VALUE, so those reassignments restore
//     the same values — kept for fidelity, equivalent by construction.
//  7. doFriction is NOT passed to the TGS contact body (the PGS overload
//     takes it; TGS always solves friction) — the caller's flag dies here.

import pxgSlangCommon;
import pxgSlangArticulationBlocks;
import pxgSlangSpatialMatrix;
import pxgSlangArticulationDynamic;
import pxgSlangSolverBlocks;
import pxgSlangArticulationDesc;
import pxgSlangSolverDesc;
import npArtiSolveTendons;
import npArtiSolveExtTGS;
import npArtiSolveExtPGS;
import npArtiAvgImpulse;

static const uint SI_BATCH_STRIDE = 1536u;   // sizeof(PxgBlockConstraintBatch)

// solveStaticConstraints (TGS overload, internalConstraints2.cu:1395)
public void siSolveStaticConstraintsTGS(
    uint64_t link, uint64_t batches,
    uint64_t contactHeaders, uint64_t frictionHeaders,
    uint64_t contactPoints, uint64_t frictionPoints,
    uint64_t jointHeaders, uint64_t jointRowsCon, uint64_t responses,
    inout AbSpatialVector vel, inout AbSpatialVector impulse, inout AbSpatialVector deltaV,
    uint lane, float minPen, float elapsedTime, bool accumulateError,
    inout SvxError error, uint constraintCounts)
{
    const uint constraintBatchOffset = abU32(link, PXG_AB_LINK_mStaticJointStartIndex, lane);
    const uint contactBatchOffset = abU32(link, PXG_AB_LINK_mStaticContactStartIndex, lane);
    const uint contactCounts = abU32(link, PXG_AB_LINK_mNbStaticContacts, lane);

    const AbSpatialVector oldVel = vel;

    AbSpatialVector delta = svLoadSV(link, PXG_AB_LINK_mDeltaMotion, lane);

    // trap 4: only the joint path reads deltaQ
    float4 deltaQ = float4(0.0f, 0.0f, 0.0f, 1.0f);
    if (constraintCounts > 0u)
        deltaQ = abF4(link, PXG_AB_LINK_mDeltaQ, lane);

    const AbSpatialVector zeroV = asmSVec(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f));
    const float4 identityQ = float4(0.0f, 0.0f, 0.0f, 1.0f);

    // trap 1: joints first
    for (uint i = 0u; i < constraintCounts; ++i)
    {
        Ptr<PxgBlockConstraintBatch> batch =
            Ptr<PxgBlockConstraintBatch>(batches + uint64_t(constraintBatchOffset + i) * SI_BATCH_STRIDE);
        const uint mask = batch[0].mask;
        const uint offset = countbits(mask & ((1u << lane) - 1u));
        const bool aIsArti = nodeIndexIsArticulation(batch[0].bodyANodeIndex[offset]);

        AbSpatialVector vel0, vel1, delta0, delta1;
        float4 deltaQ0, deltaQ1;
        if (aIsArti)
        {
            vel0 = vel;   delta0 = delta;  deltaQ0 = deltaQ;
            vel1 = zeroV; delta1 = zeroV;  deltaQ1 = identityQ;
        }
        else
        {
            vel0 = zeroV; delta0 = zeroV;  deltaQ0 = identityQ;
            vel1 = vel;   delta1 = delta;  deltaQ1 = deltaQ;
        }

        AbSpatialVector impulse0 = zeroV;
        AbSpatialVector impulse1 = zeroV;

        svxSolveExt1DBlockTGS(batch, vel0, vel1, delta0, delta1, offset,
                              Ptr<PxgTGSBlockSolverConstraint1DHeader>(jointHeaders),
                              Ptr<PxgTGSBlockSolverConstraint1DCon>(jointRowsCon),
                              Ptr<PxgArticulationBlockResponse>(responses),
                              deltaQ0, deltaQ1, elapsedTime, impulse0, impulse1,
                              accumulateError, 1.0f, 1.0f);

        if (aIsArti) { impulse = asmSVecAdd(impulse, impulse0); vel = vel0; delta = delta0; deltaQ = deltaQ0; }
        else         { impulse = asmSVecAdd(impulse, impulse1); vel = vel1; delta = delta1; deltaQ = deltaQ1; }
    }

    for (uint i = 0u; i < contactCounts; ++i)
    {
        Ptr<PxgBlockConstraintBatch> batch =
            Ptr<PxgBlockConstraintBatch>(batches + uint64_t(contactBatchOffset + i) * SI_BATCH_STRIDE);
        const uint mask = batch[0].mask;
        const uint offset = countbits(mask & ((1u << lane) - 1u));
        const bool aIsArti = nodeIndexIsArticulation(batch[0].bodyANodeIndex[offset]);

        AbSpatialVector vel0, vel1, delta0, delta1;
        float4 deltaQ0, deltaQ1;
        if (aIsArti)
        {
            vel0 = vel;   delta0 = delta;  deltaQ0 = deltaQ;
            vel1 = zeroV; delta1 = zeroV;  deltaQ1 = identityQ;
        }
        else
        {
            vel0 = zeroV; delta0 = zeroV;  deltaQ0 = identityQ;
            vel1 = vel;   delta1 = delta;  deltaQ1 = deltaQ;
        }

        AbSpatialVector impulse0 = zeroV;
        AbSpatialVector impulse1 = zeroV;

        svxSolveExtContactBlockTGS(batch, vel0, vel1, delta0, delta1, offset,
                                   Ptr<PxgTGSBlockSolverContactHeader>(contactHeaders),
                                   Ptr<PxgTGSBlockSolverFrictionHeader>(frictionHeaders),
                                   Ptr<PxgTGSBlockSolverContactPoint>(contactPoints),
                                   Ptr<PxgTGSBlockSolverContactFriction>(frictionPoints),
                                   Ptr<PxgArticulationBlockResponse>(responses),
                                   elapsedTime, minPen, impulse0, impulse1,
                                   error, accumulateError, 1.0f, 1.0f);

        if (aIsArti) { impulse = asmSVecAdd(impulse, impulse0); vel = vel0; delta = delta0; deltaQ = deltaQ0; }
        else         { impulse = asmSVecAdd(impulse, impulse1); vel = vel1; delta = delta1; deltaQ = deltaQ1; }
    }

    // trap 5
    if ((constraintCounts + contactCounts) > 0u)
        deltaV = asmSVecAdd(deltaV, asmSVecSub(vel, oldVel));
}

// ===========================================================================
// test entry
// ===========================================================================

struct SiParams
{
    uint64_t link;            // @0
    uint64_t batches;         // @8
    uint64_t contactHeaders;  // @16
    uint64_t frictionHeaders; // @24
    uint64_t contactPoints;   // @32
    uint64_t frictionPoints;  // @40
    uint64_t jointHeaders;    // @48
    uint64_t jointRowsCon;    // @56
    uint64_t responses;       // @64
    uint64_t io;              // @72  in: vel/impulse/deltaV (18 floats @ lane*24);
                              //      out at 32*24 + lane*24 (+error triple)
    float minPen;             // @80
    float elapsedTime;        // @84
    uint accumulateError;     // @88
    uint nbLanes;             // @92
}

[[vk::push_constant]]
ConstantBuffer<SiParams> gSi;

static const uint SI_IO_STRIDE = 24u;
static const uint SI_IO_OUT_BASE = 32u * 24u;

// IEEE `x + 0` as a select (P9.86): slangc folds a literal-zero add away even with
// -fp-mode precise, which keeps a -0 the CUDA turns into +0.
float isPlusZero(float x) { return (x == 0.0f) ? 0.0f : x; }

[shader("compute")]
[numthreads(32, 1, 1)]
void siStaticTest(uint3 groupThreadId : SV_GroupThreadID)
{
    const uint lane = groupThreadId.x;
    if (lane >= gSi.nbLanes)
        return;

    Ptr<float> fin = Ptr<float>(gSi.io + uint64_t(lane * SI_IO_STRIDE) * 4u);
    AbSpatialVector vel = asmSVec(float3(fin[0], fin[1], fin[2]), float3(fin[3], fin[4], fin[5]));
    AbSpatialVector impulse = asmSVec(float3(fin[6], fin[7], fin[8]), float3(fin[9], fin[10], fin[11]));
    AbSpatialVector deltaV = asmSVec(float3(fin[12], fin[13], fin[14]), float3(fin[15], fin[16], fin[17]));

    // the CALLER reads the joint count from the link, as the kernel does
    const uint constraintCounts = abU32(gSi.link, PXG_AB_LINK_mNbStaticJoints, lane);

    SvxError error = { 0.0f, 0, 0.0f };
    siSolveStaticConstraintsTGS(gSi.link, gSi.batches,
                                gSi.contactHeaders, gSi.frictionHeaders,
                                gSi.contactPoints, gSi.frictionPoints,
                                gSi.jointHeaders, gSi.jointRowsCon, gSi.responses,
                                vel, impulse, deltaV,
                                lane, gSi.minPen, gSi.elapsedTime,
                                gSi.accumulateError != 0u, error, constraintCounts);

    Ptr<float> fout = Ptr<float>(gSi.io + uint64_t(SI_IO_OUT_BASE + lane * SI_IO_STRIDE) * 4u);
    fout[0] = vel.top.x; fout[1] = vel.top.y; fout[2] = vel.top.z;
    fout[3] = vel.bottom.x; fout[4] = vel.bottom.y; fout[5] = vel.bottom.z;
    fout[6] = impulse.top.x; fout[7] = impulse.top.y; fout[8] = impulse.top.z;
    fout[9] = impulse.bottom.x; fout[10] = impulse.bottom.y; fout[11] = impulse.bottom.z;
    fout[12] = deltaV.top.x; fout[13] = deltaV.top.y; fout[14] = deltaV.top.z;
    fout[15] = deltaV.bottom.x; fout[16] = deltaV.bottom.y; fout[17] = deltaV.bottom.z;
    fout[18] = error.sumOfSquares; fout[19] = float(error.counter); fout[20] = error.maxError;
}

// ===========================================================================
// P8.100b: the per-dof internal solve (the dof loop of
// artiSolveInternalConstraints1T, :1747-1890) and
// accumulateLinkImpulsesAndLinkVelocities (:1554-1575).  This module is the
// TGS instantiation: isTGS is TRUE by construction, so
//   * the deprecated friction model's appliedFriction is 0 (PGS accumulates);
//   * the drive's unclamped impulse degenerates to the applied impulse on
//     velocity iterations, and the implicit drive desc is loaded only on
//     position iterations (on velocity iterations it is never used);
//   * perSubset = isExternalForceEveryStep && !isVelIter picks dt over the
//     scene dt for every effort-to-impulse conversion.
//
// TRAPS (each a sweep mutant unless marked equivalent):
//  1. **mDriveImpulse's writeback in the maxImpulse>0 arm is a memory RMW
//     against the ORIGINAL stored value** (`+= driveDeltaF`), while
//     driveDeltaF itself is measured against the perSubset-ZEROED copy —
//     under perSubset the two bases differ.
//  2. **jointV is threaded**: friction (old model) -> drive -> friction (new
//     model) -> limit -> velocity limit, each `jointV += deltaF * response`.
//  3. **The joint speed is the DUAL inner product row1*childV - row0*parentV**,
//     and the position delta the same over the TGS delta motions.
//  4. **externalJointImpulse enters and leaves the clamp** (`clamp(applied +
//     ext, unclamped + ext, ...) - ext`).
//  5. **The limit writeback**: computeLimitImpulse updates low/high impulses
//     conditionally; the port loads both, passes inout, stores both — the
//     unchanged one stores its own loaded bytes (equivalent, documented).
//  6. eLOCKED dofs are skipped ENTIRELY (no error counter tick either).
//  7. The residual counter ticks for every non-locked dof EVEN IF deltaF is
//     zero — the RMS depends on the count.
//  8. accumulateLinkImpulsesAndLinkVelocities: the deltas are deltaV{A,B} *
//     MINUS deltaF; the impulse rows are parentImp += row0 * deltaF and
//     impulse -= row1 * deltaF — three sign asymmetries in one helper.
// ===========================================================================

static const uint SI_MOTION_LOCKED = 0u;    // PxArticulationMotion (PxSolverDefs.h:288)
static const uint SI_MOTION_LIMITED = 1u;

public void siAccumulate(float deltaF,
                         AbSpatialVector deltaVA, AbSpatialVector deltaVB,
                         AbSpatialVector row0, AbSpatialVector row1,
                         inout AbSpatialVector parentDeltaV, inout AbSpatialVector deltaV,
                         inout AbSpatialVector parentV, inout AbSpatialVector childV,
                         inout AbSpatialVector parentImp, inout AbSpatialVector impulse)
{
    const AbSpatialVector pDelta = asmSVecScale(deltaVA, -deltaF);
    const AbSpatialVector cDelta = asmSVecScale(deltaVB, -deltaF);

    parentDeltaV = asmSVecAdd(parentDeltaV, pDelta);
    deltaV = asmSVecAdd(deltaV, cDelta);

    parentV = asmSVecAdd(parentV, pDelta);
    childV = asmSVecAdd(childV, cDelta);

    parentImp = asmSVecAdd(parentImp, asmSVecScale(row0, deltaF));
    impulse = asmSVecSub(impulse, asmSVecScale(row1, deltaF));
}

public void siSolveDofConstraints(
    uint64_t dofBlocks, uint jointOffset, uint dofCount,
    uint64_t jointForces,               // the RECORD's jointForce array (float[dof])
    float dt, float sceneDt, float invDt, float elapsedTime, float erp,
    bool isVelIter, bool isExternalForceEveryStep,
    bool doFrictionDrivePosLimit, bool velLimitBefore,
    AbSpatialVector childDelta, AbSpatialVector parentDelta,
    inout AbSpatialVector parentDeltaV, inout AbSpatialVector deltaV,
    inout AbSpatialVector parentV, inout AbSpatialVector childV,
    inout AbSpatialVector parentImp, inout AbSpatialVector impulse,
    bool residualReporting, inout SvxError error,
    uint lane)
{
    for (uint dof = 0u; dof < dofCount; ++dof)
    {
        const uint64_t d = dofBlocks + uint64_t(jointOffset + dof) * PXG_AB_DOF_SIZE;
        const uint motion = abU8(d, PXG_AB_DOF_mMotion, lane);
        if (motion == SI_MOTION_LOCKED)
            continue;

        const uint64_t ic = d + PXG_AB_DOF_mConstraintData;
        const float maxJointVel = abF32(ic, PXG_AB_IC_mMaxJointVelocity, lane);
        const AbSpatialVector row0 = svLoadSV(ic, PXG_AB_IC_mRow0, lane);
        const AbSpatialVector row1 = svLoadSV(ic, PXG_AB_IC_mRow1, lane);
        const float maxDriveForce = abF32(ic, PXG_AB_IC_mConstraintMaxImpulse, lane);
        const float appliedDriveImpulseOrig = abF32(ic, PXG_AB_IC_mDriveImpulse, lane);
        const float recipResponse = abF32(ic, PXG_AB_IC_mRecipResponse, lane);
        const float response = abF32(ic, PXG_AB_IC_mResponse, lane);
        const float maxFrictionForce = abF32(ic, PXG_AB_IC_mMaxFrictionForce, lane);
        const AbSpatialVector deltaVA = svLoadSV(ic, PXG_AB_IC_mDeltaVA, lane);
        const AbSpatialVector deltaVB = svLoadSV(ic, PXG_AB_IC_mDeltaVB, lane);

        const bool perSubset = isExternalForceEveryStep && !isVelIter;   // isTGS == true
        const float effectiveDt = perSubset ? dt : sceneDt;

        const float staticFrictionImpulse = abF32(ic, PXG_AB_IC_mStaticFrictionEffort, lane) * effectiveDt;
        const float dynamicFrictionImpulse = abF32(ic, PXG_AB_IC_mDynamicFrictionEffort, lane) * effectiveDt;
        const float viscousFrictionCoefficient = abF32(ic, PXG_AB_IC_mViscousFrictionCoefficient, lane) * effectiveDt;

        const float maxImpulse = abF32(ic, PXG_AB_IC_mMaxEffort, lane) * effectiveDt;
        const float speedImpulseGradient = abF32(ic, PXG_AB_IC_mSpeedEffortGradient, lane) / effectiveDt;
        const float velocityDependentResistance = abF32(ic, PXG_AB_IC_mVelocityDependentResistance, lane) * effectiveDt;
        const float externalJointImpulse = (Ptr<float>(jointForces))[jointOffset + dof] * effectiveDt;
        const float maxActuatorVelocity = abF32(ic, PXG_AB_IC_mMaxActuatorVelocity, lane);

        AdDriveDesc driveDesc;
        driveDesc.tvBias = 0.0f; driveDesc.biasCoef = 0.0f; driveDesc.velMul = 0.0f;
        driveDesc.impMul = 0.0f; driveDesc.posBias = 0.0f;
        if (!isVelIter)   // !(isTGS && isVelIter), isTGS == true
        {
            const float4 packed = abF4(ic, PXG_AB_IC_mTargetVelPlusInitialBiasX_DriveBiasCoefficientY_VelMultiplierZ_ImpulseMultiplierW, lane);
            driveDesc.tvBias = packed.x;
            driveDesc.biasCoef = packed.y;
            driveDesc.velMul = packed.z;
            driveDesc.impMul = packed.w;
            driveDesc.posBias = abF32(ic, PXG_AB_IC_mTargetPosBias, lane);
        }

        float jointV = asmInnerProduct(row1, childV) - asmInnerProduct(row0, parentV);
        const float jointDeltaP = asmInnerProduct(row1, childDelta) - asmInnerProduct(row0, parentDelta);

        float frictionDeltaF = 0.0f;
        float driveDeltaF = 0.0f;
        float posLimitDeltaF = 0.0f;
        if (doFrictionDrivePosLimit)
        {
            const bool newFrictionModel = staticFrictionImpulse != 0.0f || viscousFrictionCoefficient != 0.0f;

            // deprecated friction model; appliedFriction is 0 on TGS
            if (!newFrictionModel)
            {
                const float appliedFriction = 0.0f;
                // `+ appliedFriction` with appliedFriction == 0 (internalConstraints2.cu:1796-1798,
                // isTGS): nvcc keeps the add, which turns a -0 into +0; slangc folds it away
                // (P9.86), so it is spelled as the select it amounts to.
                const float frictionForce = adPClamp(isPlusZero(-jointV * recipResponse), -maxFrictionForce, maxFrictionForce);
                abStoreF32(ic, PXG_AB_IC_mAccumulatedFrictionImpulse, lane, frictionForce);
                frictionDeltaF = frictionForce - appliedFriction;
                jointV += frictionDeltaF * response;
            }

            if (maxImpulse > 0.0f)
            {
                const float appliedDriveImpulse = perSubset ? 0.0f : appliedDriveImpulseOrig;
                const float unclampedImpulse = isVelIter ? appliedDriveImpulse : adDriveImpulse(appliedDriveImpulse, jointV, jointDeltaP, elapsedTime, driveDesc);
                const float clampedImpulse = adClampDriveImpulse(jointV, appliedDriveImpulse + externalJointImpulse, unclampedImpulse + externalJointImpulse, response, maxActuatorVelocity, maxImpulse, speedImpulseGradient, velocityDependentResistance) - externalJointImpulse;
                driveDeltaF = clampedImpulse - appliedDriveImpulse;
                // trap 1: the RMW base is the ORIGINAL stored value
                abStoreF32(ic, PXG_AB_IC_mDriveImpulse, lane, appliedDriveImpulseOrig + driveDeltaF);
            }
            else
            {
                const float unclampedForce = isVelIter ? appliedDriveImpulseOrig : adDriveImpulse(appliedDriveImpulseOrig, jointV, jointDeltaP, elapsedTime, driveDesc);
                const float clampedForce = adPClamp(unclampedForce, -maxDriveForce, maxDriveForce);
                driveDeltaF = clampedForce - appliedDriveImpulseOrig;
                abStoreF32(ic, PXG_AB_IC_mDriveImpulse, lane, clampedForce);
            }
            jointV += driveDeltaF * response;

            if (newFrictionModel)
            {
                const float appliedFriction = perSubset ? 0.0f : abF32(ic, PXG_AB_IC_mAccumulatedFrictionImpulse, lane);
                float totalImpulse = appliedFriction - jointV * recipResponse;
                totalImpulse = adFrictionImpulse(totalImpulse, staticFrictionImpulse, dynamicFrictionImpulse, viscousFrictionCoefficient, jointV);
                frictionDeltaF = totalImpulse - appliedFriction;
                abStoreF32(ic, PXG_AB_IC_mAccumulatedFrictionImpulse, lane, abF32(ic, PXG_AB_IC_mAccumulatedFrictionImpulse, lane) + frictionDeltaF);
                jointV += frictionDeltaF * response;
            }

            if (motion == SI_MOTION_LIMITED)
            {
                const float2 limitErr = abF2(ic, PXG_AB_IC_mLimitError_LowX_highY, lane);
                float lowImp = abF32(ic, PXG_AB_IC_mLowImpulse, lane);
                float highImp = abF32(ic, PXG_AB_IC_mHighImpulse, lane);
                posLimitDeltaF = adLimitImpulse(dt, invDt, isVelIter, response, recipResponse, erp, limitErr.x, limitErr.y, jointDeltaP, lowImp, highImp, jointV);
                // trap 5: the conditional writeback stores loaded bytes when unchanged
                abStoreF32(ic, PXG_AB_IC_mLowImpulse, lane, lowImp);
                abStoreF32(ic, PXG_AB_IC_mHighImpulse, lane, highImp);
            }
        }

        float velLimitDeltaF = 0.0f;
        if (velLimitBefore && (abs(jointV) > maxJointVel))
        {
            const float newJointV = adPClamp(jointV, -maxJointVel, maxJointVel);
            velLimitDeltaF = (newJointV - jointV) * recipResponse;
            jointV = newJointV;
        }

        const float deltaF = frictionDeltaF + driveDeltaF + posLimitDeltaF + velLimitDeltaF;

        // trap 7: the counter ticks even for a zero deltaF
        if (residualReporting)
            svxAccumulateError(error, deltaF, recipResponse);

        siAccumulate(deltaF, deltaVA, deltaVB, row0, row1,
                     parentDeltaV, deltaV, parentV, childV, parentImp, impulse);
    }
}

// ---------------------------------------------------------------------------
// test entry for the dof solve
// ---------------------------------------------------------------------------

struct SdParams
{
    uint64_t dofBlocks;    // @0
    uint64_t jointForces;  // @8   fixture scaffolding: per-lane slices
    uint64_t link;         // @16  jointOffset/dofCount come from here, as in the kernel
    uint64_t io;           // @24
    float dt;              // @32
    float sceneDt;         // @36
    float invDt;           // @40
    float elapsedTime;     // @44
    float erp;             // @48
    uint flagsWord;        // @52  bit0 velIter, bit1 extEvery, bit2 doFDPL, bit3 velBefore, bit4 residual
    uint nbLanes;          // @56
    uint maxDofs;          // @60  the per-lane jointForces slice stride (scaffolding)
}

[[vk::push_constant]]
ConstantBuffer<SdParams> gSd;

static const uint SD_IO_STRIDE = 48u;
static const uint SD_IO_OUT_BASE = 32u * 48u;

[shader("compute")]
[numthreads(32, 1, 1)]
void siDofTest(uint3 groupThreadId : SV_GroupThreadID)
{
    const uint lane = groupThreadId.x;
    if (lane >= gSd.nbLanes)
        return;

    Ptr<float> fin = Ptr<float>(gSd.io + uint64_t(lane * SD_IO_STRIDE) * 4u);
    AbSpatialVector parentV = asmSVec(float3(fin[0], fin[1], fin[2]), float3(fin[3], fin[4], fin[5]));
    AbSpatialVector childV = asmSVec(float3(fin[6], fin[7], fin[8]), float3(fin[9], fin[10], fin[11]));
    AbSpatialVector parentDeltaV = asmSVec(float3(fin[12], fin[13], fin[14]), float3(fin[15], fin[16], fin[17]));
    AbSpatialVector deltaV = asmSVec(float3(fin[18], fin[19], fin[20]), float3(fin[21], fin[22], fin[23]));
    AbSpatialVector parentImp = asmSVec(float3(fin[24], fin[25], fin[26]), float3(fin[27], fin[28], fin[29]));
    AbSpatialVector impulse = asmSVec(float3(fin[30], fin[31], fin[32]), float3(fin[33], fin[34], fin[35]));
    const AbSpatialVector childDelta = asmSVec(float3(fin[36], fin[37], fin[38]), float3(fin[39], fin[40], fin[41]));
    const AbSpatialVector parentDelta = asmSVec(float3(fin[42], fin[43], fin[44]), float3(fin[45], fin[46], fin[47]));

    const uint jointOffset = abU32(gSd.link, PXG_AB_LINK_mJointOffset, lane);
    const uint dofCount = abU8(gSd.link, PXG_AB_LINK_mDofs, lane);

    // fixture scaffolding: the kernel passes the articulation's own record
    const uint64_t jf = gSd.jointForces + uint64_t(lane) * uint64_t(gSd.maxDofs) * 4u;

    const bool velIter = (gSd.flagsWord & 1u) != 0u;
    const bool extEvery = (gSd.flagsWord & 2u) != 0u;
    const bool doFDPL = (gSd.flagsWord & 4u) != 0u;
    const bool velBefore = (gSd.flagsWord & 8u) != 0u;
    const bool residual = (gSd.flagsWord & 16u) != 0u;

    SvxError error = { 0.0f, 0, 0.0f };
    siSolveDofConstraints(gSd.dofBlocks, jointOffset, dofCount, jf,
                          gSd.dt, gSd.sceneDt, gSd.invDt, gSd.elapsedTime, gSd.erp,
                          velIter, extEvery, doFDPL, velBefore,
                          childDelta, parentDelta,
                          parentDeltaV, deltaV, parentV, childV, parentImp, impulse,
                          residual, error, lane);

    Ptr<float> fout = Ptr<float>(gSd.io + uint64_t(SD_IO_OUT_BASE + lane * SD_IO_STRIDE) * 4u);
    fout[0] = parentV.top.x; fout[1] = parentV.top.y; fout[2] = parentV.top.z;
    fout[3] = parentV.bottom.x; fout[4] = parentV.bottom.y; fout[5] = parentV.bottom.z;
    fout[6] = childV.top.x; fout[7] = childV.top.y; fout[8] = childV.top.z;
    fout[9] = childV.bottom.x; fout[10] = childV.bottom.y; fout[11] = childV.bottom.z;
    fout[12] = parentDeltaV.top.x; fout[13] = parentDeltaV.top.y; fout[14] = parentDeltaV.top.z;
    fout[15] = parentDeltaV.bottom.x; fout[16] = parentDeltaV.bottom.y; fout[17] = parentDeltaV.bottom.z;
    fout[18] = deltaV.top.x; fout[19] = deltaV.top.y; fout[20] = deltaV.top.z;
    fout[21] = deltaV.bottom.x; fout[22] = deltaV.bottom.y; fout[23] = deltaV.bottom.z;
    fout[24] = parentImp.top.x; fout[25] = parentImp.top.y; fout[26] = parentImp.top.z;
    fout[27] = parentImp.bottom.x; fout[28] = parentImp.bottom.y; fout[29] = parentImp.bottom.z;
    fout[30] = impulse.top.x; fout[31] = impulse.top.y; fout[32] = impulse.top.z;
    fout[33] = impulse.bottom.x; fout[34] = impulse.bottom.y; fout[35] = impulse.bottom.z;
    fout[36] = error.sumOfSquares; fout[37] = float(error.counter); fout[38] = error.maxError;
}

// ===========================================================================
// P8.100c: the DFS — the traversal-stack walk of artiSolveInternalConstraints
// (:1662-2002).  Forward pass: propagateAccelerationW_child2ParentPreLoaded
// <ReadFromDeferredQstZ> (svPropagateAccelDeferred — the SEPARATE-sum order,
// which THIS kernel uses, unlike the self-solve's direct order), the per-dof
// solve, the static solve, the optional AFTER-static velocity-limit pass.
// Backprop: propagateImpulseW<WriteToDeferredQstZ> (svPropagateImpulseDeferred)
// and the 6x6 spatial response deltaV = R * -parentImp.
//
// TRAPS (each a sweep mutant unless marked equivalent):
//  1. **The backprop's mSolverSpatialImpulse booking subtracts the link's
//     mSolverSpatialInternalConstraintImpulse**: solverSpatialImpulse +=
//     impulse - ZInternalConstraint — the internal-constraint share is
//     tracked separately and must not double-count.
//  2. **The backprop stores ONLY the parent's impulseStack**, never its
//     deltaVStack — parentDeltaV continues in the register.
//  3. **On a leaf (numChildren == 0) linkID stays put**, so the next
//     iteration sees parent == linkID and back-propagates.
//  4. **stack[parent].impulseStack is stored TWICE in the forward pass**
//     (once for the parent, once for the link) and parentImp/parentDeltaV
//     hand over to the child at the block end (parentDeltaV = deltaV).
//  5. **The AFTER-static velocity-limit pass propagates the STATIC impulse
//     inwards with the dof array at its BASE** — `dofData`, NOT
//     `dofData + jointOffset` (:1927/:1928).  That reads JOINT 0's dof rows
//     for every link; almost certainly an engine bug, transcribed as-is and
//     recorded here.  Its qstZ target is NULL (WriteToProvidedQstZ) — the
//     hop must NOT touch mDeferredQstZ.
//  6. **The AFTER pass recomputes jointV against parentV + deltaVParent**
//     (the parent velocity corrected by the static impulse response), with
//     the 6x6 response of the PARENT.
//  7. childImpulseInternal = the dof-solve impulse PLUS the AFTER pass's
//     velocity-limit contribution — the static share is excluded.
// ===========================================================================

public AbSpatialVector siSolveInternalDFS(
    uint64_t linkBase, uint64_t dofBlocks, uint64_t stackBase,
    uint64_t jointForces,
    uint64_t batches,
    uint64_t contactHeaders, uint64_t frictionHeaders,
    uint64_t contactPoints, uint64_t frictionPoints,
    uint64_t jointHeaders, uint64_t jointRowsCon, uint64_t responses,
    float dt, float sceneDt, float invDt, float elapsedTime, float erp, float minPen,
    bool isVelIter, bool isExternalForceEveryStep,
    bool doFrictionDrivePosLimit, uint doVelLimit,   // 0 none / 1 before / 2 after
    bool doStaticContactsAnd1dConstraints,
    AbSpatialVector storeParentDeltaV, AbSpatialVector parentImpIn,
    AbSpatialVector parentDeltaVIn,
    bool residualReporting, inout SvxError error, inout SvxError contactError,
    uint lane)
{
    AbSpatialVector parentDeltaV = parentDeltaVIn;
    AbSpatialVector parentImp = parentImpIn;

    uint parent = 0u;

    abStoreSpatialVector(stackBase + uint64_t(parent) * PXG_AB_TRAVERSAL_STACK_SIZE, PXG_AB_TRAVERSAL_deltaVStack, lane, storeParentDeltaV.top, storeParentDeltaV.bottom);
    abStoreSpatialVector(stackBase + uint64_t(parent) * PXG_AB_TRAVERSAL_STACK_SIZE, PXG_AB_TRAVERSAL_impulseStack, lane, parentImp.top, parentImp.bottom);

    uint offset = abU32(linkBase + uint64_t(parent) * PXG_AB_LINK_SIZE, PXG_AB_LINK_mChildrenOffset, lane);
    uint numChildren = abU32(linkBase + uint64_t(parent) * PXG_AB_LINK_SIZE, PXG_AB_LINK_mNumChildren, lane);

    for (uint i = 0u; i < numChildren; ++i)
        abStoreU32(stackBase + uint64_t(i) * PXG_AB_TRAVERSAL_STACK_SIZE, PXG_AB_TRAVERSAL_indices, lane, offset + i);

    uint stackCount = numChildren;
    uint linkID = (stackCount != 0u)
        ? abU32(stackBase + uint64_t(stackCount - 1u) * PXG_AB_TRAVERSAL_STACK_SIZE, PXG_AB_TRAVERSAL_indices, lane)
        : 0xffffffffu;

    while (stackCount != 0u)
    {
        const bool isBackProp = (parent == linkID);

        const uint64_t link = linkBase + uint64_t(linkID) * PXG_AB_LINK_SIZE;
        const float3 c2p = float3(abF32(link, PXG_AB_LINK_mRw_x, lane),
                                  abF32(link, PXG_AB_LINK_mRw_y, lane),
                                  abF32(link, PXG_AB_LINK_mRw_z, lane));
        const uint jointOffset = abU32(link, PXG_AB_LINK_mJointOffset, lane);
        const uint dofCount = abU8(link, PXG_AB_LINK_mDofs, lane);

        if (isBackProp)
        {
            parent = abU32(link, PXG_AB_LINK_mParents, lane);

            const AbSpatialVector solverSpatialImpulse = svLoadSV(link, PXG_AB_LINK_mSolverSpatialImpulse, lane);
            const AbSpatialVector ZInternalConstraint = svLoadSV(link, PXG_AB_LINK_mSolverSpatialInternalConstraintImpulse, lane);

            const AbMatrix6x6 spatialResponse = abLoadSpatialMatrix6x6(linkBase + uint64_t(parent) * PXG_AB_LINK_SIZE, PXG_AB_LINK_mSpatialResponseMatrix, lane);

            stackCount--;

            const AbSpatialVector impulse = parentImp;

            const uint64_t pStack = stackBase + uint64_t(parent) * PXG_AB_TRAVERSAL_STACK_SIZE;
            parentDeltaV = svLoadSV(pStack, PXG_AB_TRAVERSAL_deltaVStack, lane);
            parentImp = svLoadSV(pStack, PXG_AB_TRAVERSAL_impulseStack, lane);

            // trap 1: the internal-constraint share is subtracted
            const AbSpatialVector booked = asmSVecSub(asmSVecAdd(solverSpatialImpulse, impulse), ZInternalConstraint);
            abStoreSpatialVector(link, PXG_AB_LINK_mSolverSpatialImpulse, lane, booked.top, booked.bottom);

            linkID = (stackCount > 0u)
                ? abU32(stackBase + uint64_t(stackCount - 1u) * PXG_AB_TRAVERSAL_STACK_SIZE, PXG_AB_TRAVERSAL_indices, lane)
                : 0xffffffffu;

            float jfNone[3] = { 0.0f, 0.0f, 0.0f };
            const AbSpatialVector propagateImp = svPropagateImpulseDeferred(c2p, dofBlocks, jointOffset, dofCount, impulse, lane, jfNone, 1.0f, false);

            parentImp = asmSVecAdd(parentImp, propagateImp);

            const AbSpatialVector deltaV = asmM6MulVec(spatialResponse, asmSVec(-parentImp.top, -parentImp.bottom));

            parentDeltaV = asmSVecAdd(parentDeltaV, deltaV);

            // trap 2: only the impulse frame goes back to the stack
            abStoreSpatialVector(pStack, PXG_AB_TRAVERSAL_impulseStack, lane, parentImp.top, parentImp.bottom);
        }
        else
        {
            const uint64_t parentLink = linkBase + uint64_t(parent) * PXG_AB_LINK_SIZE;
            AbSpatialVector parentV = asmSVecAdd(parentDeltaV, svLoadSV(parentLink, PXG_AB_LINK_mMotionVelocity, lane));
            const AbSpatialVector childDelta = svLoadSV(link, PXG_AB_LINK_mDeltaMotion, lane);
            const AbSpatialVector parentDelta = svLoadSV(parentLink, PXG_AB_LINK_mDeltaMotion, lane);

            float dsp[3] = { 0.0f, 0.0f, 0.0f };
            AbSpatialVector deltaV = svPropagateAccelDeferred(c2p, dofBlocks, jointOffset, dofCount, parentDeltaV, lane, dsp, false);

            AbSpatialVector childV = asmSVecAdd(deltaV, svLoadSV(link, PXG_AB_LINK_mMotionVelocity, lane));

            abStoreSpatialVector(stackBase + uint64_t(linkID) * PXG_AB_TRAVERSAL_STACK_SIZE, PXG_AB_TRAVERSAL_deltaVStack, lane, deltaV.top, deltaV.bottom);

            AbSpatialVector impulse = asmSVec(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f));

            siSolveDofConstraints(dofBlocks, jointOffset, dofCount, jointForces,
                                  dt, sceneDt, invDt, elapsedTime, erp,
                                  isVelIter, isExternalForceEveryStep,
                                  doFrictionDrivePosLimit, doVelLimit == 1u,
                                  childDelta, parentDelta,
                                  parentDeltaV, deltaV, parentV, childV, parentImp, impulse,
                                  residualReporting, error, lane);

            const uint constraintCounts = abU32(link, PXG_AB_LINK_mNbStaticJoints, lane);
            numChildren = abU32(link, PXG_AB_LINK_mNumChildren, lane);

            AbSpatialVector childImpulseInternal = impulse;

            AbSpatialVector childImpulseFromStatic = asmSVec(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f));
            if (doStaticContactsAnd1dConstraints)
            {
                const AbSpatialVector before = impulse;
                siSolveStaticConstraintsTGS(link, batches,
                                            contactHeaders, frictionHeaders,
                                            contactPoints, frictionPoints,
                                            jointHeaders, jointRowsCon, responses,
                                            childV, impulse, deltaV,
                                            lane, minPen, elapsedTime,
                                            residualReporting, contactError, constraintCounts);
                childImpulseFromStatic = asmSVecSub(impulse, before);
            }

            if (doVelLimit == 2u)   // eAFTER_STATIC_CONSTRAINTS
            {
                const AbSpatialVector i1BeforeVelLimit = impulse;

                // trap 5: the dof array at its BASE (the engine's own spelling)
                AbSpatialVector motionB[3]; AbSpatialVector isInvDB[3]; AbSpatialVector isWB[3]; float3 invStIsTB[3];
                svLoadDofArrays(dofBlocks, 0u, dofCount, lane, motionB, isInvDB, isWB, invStIsTB);
                float qstzLocal[3] = { 0.0f, 0.0f, 0.0f };
                float jfNone2[3] = { 0.0f, 0.0f, 0.0f };
                const AbSpatialVector propagatedImpulseAtParentW = adPropagateImpulseWJf(isInvDB, c2p, motionB, childImpulseFromStatic, dofCount, qstzLocal, jfNone2, 0.0f, false);

                const AbMatrix6x6 mat = abLoadSpatialMatrix6x6(parentLink, PXG_AB_LINK_mSpatialResponseMatrix, lane);
                const AbSpatialVector deltaVParent = asmM6MulVec(mat, asmSVec(-propagatedImpulseAtParentW.top, -propagatedImpulseAtParentW.bottom));

                const AbSpatialVector parentVCorrected = asmSVecAdd(parentV, deltaVParent);

                for (uint dof = 0u; dof < dofCount; ++dof)
                {
                    const uint64_t d = dofBlocks + uint64_t(jointOffset + dof) * PXG_AB_DOF_SIZE;
                    const uint motion = abU8(d, PXG_AB_DOF_mMotion, lane);
                    if (motion == SI_MOTION_LOCKED)
                        continue;
                    const uint64_t ic = d + PXG_AB_DOF_mConstraintData;
                    const float maxJointVel = abF32(ic, PXG_AB_IC_mMaxJointVelocity, lane);
                    const AbSpatialVector row0 = svLoadSV(ic, PXG_AB_IC_mRow0, lane);
                    const AbSpatialVector row1 = svLoadSV(ic, PXG_AB_IC_mRow1, lane);
                    const float recipResponse = abF32(ic, PXG_AB_IC_mRecipResponse, lane);
                    const AbSpatialVector dVA = svLoadSV(ic, PXG_AB_IC_mDeltaVA, lane);
                    const AbSpatialVector dVB = svLoadSV(ic, PXG_AB_IC_mDeltaVB, lane);
                    float jointV = asmInnerProduct(row1, childV) - asmInnerProduct(row0, parentVCorrected);

                    float velLimitDeltaF = 0.0f;
                    if (abs(jointV) > maxJointVel)
                    {
                        const float newJointV = adPClamp(jointV, -maxJointVel, maxJointVel);
                        velLimitDeltaF = (newJointV - jointV) * recipResponse;
                        jointV = newJointV;
                    }

                    if (residualReporting)
                        svxAccumulateError(error, velLimitDeltaF, recipResponse);

                    siAccumulate(velLimitDeltaF, dVA, dVB, row0, row1,
                                 parentDeltaV, deltaV, parentV, childV, parentImp, impulse);
                }

                // trap 7: only the velocity-limit share joins the internal booking
                childImpulseInternal = asmSVecAdd(childImpulseInternal, asmSVecSub(impulse, i1BeforeVelLimit));
            }

            abStoreSpatialVector(link, PXG_AB_LINK_mSolverSpatialInternalConstraintImpulse, lane, childImpulseInternal.top, childImpulseInternal.bottom);

            abStoreSpatialVector(stackBase + uint64_t(parent) * PXG_AB_TRAVERSAL_STACK_SIZE, PXG_AB_TRAVERSAL_impulseStack, lane, parentImp.top, parentImp.bottom);
            abStoreSpatialVector(stackBase + uint64_t(linkID) * PXG_AB_TRAVERSAL_STACK_SIZE, PXG_AB_TRAVERSAL_impulseStack, lane, impulse.top, impulse.bottom);

            {
                parent = linkID;
                // trap 3: with no children the linkID stays -> next round backprops
                if (numChildren > 0u)
                {
                    offset = abU32(link, PXG_AB_LINK_mChildrenOffset, lane);
                    for (uint i = 0u; i < numChildren; ++i)
                    {
                        abStoreU32(stackBase + uint64_t(stackCount) * PXG_AB_TRAVERSAL_STACK_SIZE, PXG_AB_TRAVERSAL_indices, lane, offset + i);
                        stackCount++;
                    }
                    linkID = abU32(stackBase + uint64_t(stackCount - 1u) * PXG_AB_TRAVERSAL_STACK_SIZE, PXG_AB_TRAVERSAL_indices, lane);
                }

                parentDeltaV = deltaV;
                parentImp = impulse;
            }
        }
    }

    return parentImp;
}

// ---------------------------------------------------------------------------
// test entry for the DFS
// ---------------------------------------------------------------------------

struct SfParams
{
    uint64_t linkBase;        // @0
    uint64_t dofBlocks;       // @8
    uint64_t stackBase;       // @16
    uint64_t jointForces;     // @24  fixture scaffolding: per-lane slices
    uint64_t batches;         // @32
    uint64_t contactHeaders;  // @40
    uint64_t frictionHeaders; // @48
    uint64_t contactPoints;   // @56
    uint64_t frictionPoints;  // @64
    uint64_t jointHeaders;    // @72
    uint64_t jointRowsCon;    // @80
    uint64_t responses;       // @88
    uint64_t io;              // @96
    float dt;                 // @104
    float sceneDt;            // @108
    float invDt;              // @112
    float elapsedTime;        // @116
    float erp;                // @120
    float minPen;             // @124
    uint flagsWord;           // @128  bit0 velIter, bit1 extEvery, bit2 doFDPL,
                              //       bits 3-4 doVelLimit, bit5 doStatic, bit6 residual
    uint nbLanes;             // @132
    uint maxDofs;             // @136  jointForces slice stride (scaffolding)
    uint pad0;                // @140
}

[[vk::push_constant]]
ConstantBuffer<SfParams> gSf;

static const uint SF_IO_STRIDE = 24u;
static const uint SF_IO_OUT_BASE = 32u * 24u;

[shader("compute")]
[numthreads(32, 1, 1)]
void siDfsTest(uint3 groupThreadId : SV_GroupThreadID)
{
    const uint lane = groupThreadId.x;
    if (lane >= gSf.nbLanes)
        return;

    Ptr<float> fin = Ptr<float>(gSf.io + uint64_t(lane * SF_IO_STRIDE) * 4u);
    const AbSpatialVector storeParentDeltaV = asmSVec(float3(fin[0], fin[1], fin[2]), float3(fin[3], fin[4], fin[5]));
    const AbSpatialVector parentImpIn = asmSVec(float3(fin[6], fin[7], fin[8]), float3(fin[9], fin[10], fin[11]));
    const AbSpatialVector parentDeltaVIn = asmSVec(float3(fin[12], fin[13], fin[14]), float3(fin[15], fin[16], fin[17]));

    const uint64_t jf = gSf.jointForces + uint64_t(lane) * uint64_t(gSf.maxDofs) * 4u;

    const bool velIter = (gSf.flagsWord & 1u) != 0u;
    const bool extEvery = (gSf.flagsWord & 2u) != 0u;
    const bool doFDPL = (gSf.flagsWord & 4u) != 0u;
    const uint doVelLimit = (gSf.flagsWord >> 3u) & 3u;
    const bool doStatic = (gSf.flagsWord & 32u) != 0u;
    const bool residual = (gSf.flagsWord & 64u) != 0u;

    SvxError error = { 0.0f, 0, 0.0f };
    SvxError contactError = { 0.0f, 0, 0.0f };
    const AbSpatialVector finalImp = siSolveInternalDFS(
        gSf.linkBase, gSf.dofBlocks, gSf.stackBase, jf,
        gSf.batches, gSf.contactHeaders, gSf.frictionHeaders,
        gSf.contactPoints, gSf.frictionPoints,
        gSf.jointHeaders, gSf.jointRowsCon, gSf.responses,
        gSf.dt, gSf.sceneDt, gSf.invDt, gSf.elapsedTime, gSf.erp, gSf.minPen,
        velIter, extEvery, doFDPL, doVelLimit, doStatic,
        storeParentDeltaV, parentImpIn, parentDeltaVIn,
        residual, error, contactError, lane);

    Ptr<float> fout = Ptr<float>(gSf.io + uint64_t(SF_IO_OUT_BASE + lane * SF_IO_STRIDE) * 4u);
    fout[0] = finalImp.top.x; fout[1] = finalImp.top.y; fout[2] = finalImp.top.z;
    fout[3] = finalImp.bottom.x; fout[4] = finalImp.bottom.y; fout[5] = finalImp.bottom.z;
    fout[6] = error.sumOfSquares; fout[7] = float(error.counter); fout[8] = error.maxError;
    fout[9] = contactError.sumOfSquares; fout[10] = float(contactError.counter); fout[11] = contactError.maxError;
}

// ===========================================================================
// P8.100d: the kernel — artiSolveInternalConstraintsTGS1T
// (internalConstraints2.cu:2841, template body :1579-2012), composing:
//
//   1. record resolve via islandNodeIndices -> bodySim -> articulations
//   2. residual reset on BOTH record accumulators (Dy::ErrorAccumulator::
//      reset, three zeros each) when reporting
//   3. aiAverageAndPropagate WITHOUT scale under eHAS_IMPULSES — the CUDA
//      calls the .cuh function with its scale default (0 -> internal-count
//      branch), unlike the P8.99c kernel which passes 1/refCount.  No
//      countActiveSlabs and no resetSlabCount here — those belong to the
//      P8.99c kernel; the extern's comment claiming this kernel resets
//      reference counts is stale, the body does not.
//   4. mStateDirty = eVEL_DIRTY ALWAYS (for every live thread)
//   5. the root static solve under !fixBase && doStatic: parentDeltaV =
//      M * -rootDeferredZ (the 7-column inverse articulated inertia),
//      rootVel = link0.mMotionVelocity + parentDeltaV, storeParentDeltaV
//      captured BEFORE the static solve mutates parentDeltaV,
//      constraintCounts0 = link0.mNbStaticJoints
//   6. siSolveInternalDFS (P8.100c)
//   7. epilogue: mRootDeferredZ = rootDeferredZ + finalImp; both residual
//      accumulators pushed into the record with NO atomics (one thread owns
//      the articulation): += sum, += count, max on maxError.
//
// Wrapper-level facts, each a sweep mutant unless marked:
//  * erp = min(0.7, biasCoefficient), computed HERE (the extern's own line
//    :2852); the template calls the erp parameter "biasCoefficient".
//  * minPen = velocityIteration ? 0 : -PX_MAX_F32.
//  * sceneDt (the per-dof solve's non-perSubset effective dt) is the
//    DESCRIPTOR's dt at 48 — NOT the kernel's dt argument (:1766).
//  * staticContactUniqueIds / staticJointUniqueIds are DEAD parameters (the
//    body reads the static windows from the link blocks); doFriction is dead
//    on the TGS path (only the PGS static overload reads it).  The ABI row
//    still carries all three.
//  * launch: grid ceil(nbArticulations/32), block (32,1,1)
//    (SOLVE_INTERNAL_CONSTRAINTS = 32, numWarpsPerBlock = 1,
//    PxgArticulationCore.cpp:634-637) — the per-warp slice index is
//    groupId.x directly (blockIdx.x in the CUDA).
//  * the residual flags: the record reset + epilogue + the solve bodies run
//    under the residualReportingEnabled PARAM; the CUDA's static 1D rows
//    read scDesc->mContactErrorAccumulator.mCounter >= 0 instead (:1462) —
//    in the engine the two are always in sync (documented equivalence, the
//    port passes the param through, as P8.100a did).
// ===========================================================================

struct SipParams
{
    uint64_t scDesc;                 // @0  PxgArticulationCoreDesc*
    float dt;                        // @8
    float invDt;                     // @12
    uint velIterWord;                // @16 bool byte 0 (bridge zero-pads 17..19)
    float elapsedTime;               // @20
    float biasCoefficient;           // @24
    uint pad0;                       // @28 natural-alignment hole before the pointer
    uint64_t staticContactUniqueIds; // @32 DEAD — the body reads link-block windows
    uint64_t staticJointUniqueIds;   // @40 DEAD
    uint64_t sharedDesc;             // @48 PxgSolverSharedDesc<IterativeSolveDataTGS>*
    uint solveFlagsWord;             // @56: byte0 doFriction (dead on TGS),
                                     //      byte1 residualReportingEnabled,
                                     //      byte2 isExternalForceEveryStep,
                                     //      byte3 doFrictionDrivePosLimit
    uint doVelLimit;                 // @60 VelLimit enum: 0 none / 1 before / 2 after
    uint doStaticWord;               // @64 bool byte 0
    uint pad1;                       // @68 -> 72 B blob
}

[[vk::push_constant]]
ConstantBuffer<SipParams> gSip;

static const uint SIP_VEL_DIRTY = 1u;      // PxgArtiStateDirtyFlag::eVEL_DIRTY
static const uint SIP_HAS_IMPULSES = 2u;   // PxgArtiStateDirtyFlag::eHAS_IMPULSES

[shader("compute")]
[numthreads(32, 1, 1)]
void artiSolveInternalConstraintsTGS1T(uint3 groupThreadId : SV_GroupThreadID,
                                       uint3 groupId : SV_GroupID)
{
    const uint64_t d = gSip.scDesc;
    const uint lane = groupThreadId.x;
    const uint globalThreadIndex = groupId.x * 32u + lane;   // blockStride = blockDim.x

    const uint nbArticulations = artiU32Field(d, PXG_ARTI_CORE_NB_ARTICULATIONS);
    if (globalThreadIndex >= nbArticulations)
        return;

    const bool velocityIteration = (gSip.velIterWord & 0xFFu) != 0u;
    const bool residualReporting = ((gSip.solveFlagsWord >> 8u) & 0xFFu) != 0u;
    const bool isExternalForceEveryStep = ((gSip.solveFlagsWord >> 16u) & 0xFFu) != 0u;
    const bool doFrictionDrivePosLimit = ((gSip.solveFlagsWord >> 24u) & 0xFFu) != 0u;
    const bool doStatic = (gSip.doStaticWord & 0xFFu) != 0u;

    const float erp = min(0.7f, gSip.biasCoefficient);              // :2852
    const float minPen = velocityIteration ? 0.0f : -3.402823466e+38f;  // -PX_MAX_F32
    const float sceneDt = asfloat(artiU32Field(d, PXG_ARTI_CORE_DT));   // :1766

    SvxError error = { 0.0f, 0, 0.0f };
    SvxError contactError = { 0.0f, 0, 0.0f };

    // record resolve (:1605-1610)
    const uint articulationOffset = artiU32Field(d, PXG_ARTI_CORE_ARTICULATION_OFFSET);
    const uint64_t islandNodeIndices = artiPtrField(d, PXG_ARTI_CORE_ISLAND_NODE_INDICES);
    const uint2 node = (Ptr<uint2>(islandNodeIndices))[globalThreadIndex + articulationOffset];
    const uint nodeIndex = nodeIndexIndex(node);
    const uint64_t bodySim = artiPtrField(d, PXG_ARTI_CORE_BODY_SIM_POOL)
                           + uint64_t(nodeIndex) * PXG_BODY_SIM_SIZE;
    const uint articulationIndex = artiU32Field(bodySim, PXG_BODY_SIM_ARTICULATION_REMAP_ID);
    const uint64_t a = artiPtrField(d, PXG_ARTI_CORE_ARTICULATIONS)
                     + uint64_t(articulationIndex) * PXG_ARTI_STRIDE;

    if (residualReporting)
    {
        // Dy::ErrorAccumulator::reset() on both record accumulators (:1614-1615)
        Ptr<float> ir = Ptr<float>(a + PXG_ARTI_INTERNAL_RESIDUAL);
        ir[0] = 0.0f; (Ptr<int>(a + PXG_ARTI_INTERNAL_RESIDUAL))[1] = 0; ir[2] = 0.0f;
        Ptr<float> cr = Ptr<float>(a + PXG_ARTI_CONTACT_RESIDUAL);
        cr[0] = 0.0f; (Ptr<int>(a + PXG_ARTI_CONTACT_RESIDUAL))[1] = 0; cr[2] = 0.0f;
    }

    const uint64_t w = uint64_t(groupId.x);   // the per-warp slice index, DIRECTLY
    const uint maxLinks = artiU32Field(d, PXG_ARTI_CORE_MAX_LINKS);
    const uint maxDofs = artiU32Field(d, PXG_ARTI_CORE_MAX_DOFS);
    const uint64_t artiBlock = artiPtrField(d, PXG_ARTI_CORE_ARTICULATION_BLOCKS)
                             + w * PXG_AB_BLOCK_SIZE;
    const uint64_t linkBase = artiPtrField(d, PXG_ARTI_CORE_LINK_BLOCKS)
                            + w * uint64_t(maxLinks) * PXG_AB_LINK_SIZE;
    const uint64_t dofBase = artiPtrField(d, PXG_ARTI_CORE_DOF_BLOCKS)
                           + w * uint64_t(maxDofs) * PXG_AB_DOF_SIZE;

    // averaging preamble (:1628-1632): NO scale — the internal-count branch
    if ((abU8(artiBlock, PXG_AB_mStateDirty, lane) & SIP_HAS_IMPULSES) != 0u)
    {
        const uint nbSlabs = artiU32Field(d, PXG_ARTI_CORE_NB_SLABS);
        const uint64_t slabHasChanges = artiPtrField(d, PXG_ARTI_CORE_SLAB_HAS_CHANGES);
        const uint64_t impulses = artiPtrField(d, PXG_ARTI_CORE_IMPULSES);
        aiAverageAndPropagate(slabHasChanges, impulses, artiBlock, linkBase, dofBase,
                              globalThreadIndex, maxLinks, nbArticulations, nbSlabs, lane);
    }
    abStoreU8(artiBlock, PXG_AB_mStateDirty, lane, SIP_VEL_DIRTY);   // ALWAYS (:1633)

    const bool fixBase = (abU32(artiBlock, PXG_AB_mFlags, lane) & PXG_ARTI_FLAG_FIX_BASE) != 0u;
    const AbSpatialVector rootDeferredZ = svLoadSV(artiBlock, PXG_AB_mRootDeferredZ, lane);

    const AbSpatialVector zeroSV = asmSVec(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f));
    AbSpatialVector parentDeltaV = zeroSV;
    AbSpatialVector storeParentDeltaV = zeroSV;
    AbSpatialVector parentImp = zeroSV;

    Ptr<PxgSolverSharedDescTGSGpu> sharedDesc = Ptr<PxgSolverSharedDescTGSGpu>(gSip.sharedDesc);
    const uint64_t batches = sharedDesc[0].iter_blockConstraintBatch;
    const uint64_t contactHeaders = sharedDesc[0].iter_blockContactHeaders;
    const uint64_t frictionHeaders = sharedDesc[0].iter_blockFrictionHeaders;
    const uint64_t contactPoints = sharedDesc[0].iter_blockContactPoints;
    const uint64_t frictionPoints = sharedDesc[0].iter_blockFrictions;
    const uint64_t jointHeaders = sharedDesc[0].iter_blockJointConstraintHeaders;
    const uint64_t jointRowsCon = sharedDesc[0].iter_blockJointConstraintRowsCon;
    const uint64_t responses = sharedDesc[0].iter_artiResponse;

    // the root static solve (:1643-1660)
    if (!fixBase && doStatic)
    {
        const uint constraintCounts0 = abU32(linkBase, PXG_AB_LINK_mNbStaticJoints, lane);
        const AbSpatialMatrix m =
            asmLoadSpatialMatrix(artiBlock, PXG_AB_mInvSpatialArticulatedInertia, lane);
        const AbSpatialVector motionVelocity0 = svLoadSV(linkBase, PXG_AB_LINK_mMotionVelocity, lane);

        parentDeltaV = asmMulVec(m, asmSVec(-rootDeferredZ.top, -rootDeferredZ.bottom));
        AbSpatialVector rootVel = asmSVecAdd(motionVelocity0, parentDeltaV);
        storeParentDeltaV = parentDeltaV;   // captured BEFORE the solve mutates parentDeltaV

        siSolveStaticConstraintsTGS(linkBase, batches,
                                    contactHeaders, frictionHeaders,
                                    contactPoints, frictionPoints,
                                    jointHeaders, jointRowsCon, responses,
                                    rootVel, parentImp, parentDeltaV,
                                    lane, minPen, gSip.elapsedTime,
                                    residualReporting, contactError, constraintCounts0);
    }

    const uint64_t stackBase = artiPtrField(d, PXG_ARTI_CORE_TRAVERSAL_STACK)
                             + w * uint64_t(maxLinks) * PXG_AB_TRAVERSAL_STACK_SIZE;
    const uint64_t jointForces = artiPtrField(a, PXG_ARTI_JOINT_FORCE);

    const AbSpatialVector finalImp = siSolveInternalDFS(
        linkBase, dofBase, stackBase, jointForces,
        batches, contactHeaders, frictionHeaders, contactPoints, frictionPoints,
        jointHeaders, jointRowsCon, responses,
        gSip.dt, sceneDt, gSip.invDt, gSip.elapsedTime, erp, minPen,
        velocityIteration, isExternalForceEveryStep,
        doFrictionDrivePosLimit, gSip.doVelLimit, doStatic,
        storeParentDeltaV, parentImp, parentDeltaV,
        residualReporting, error, contactError, lane);

    // epilogue (:2004-2010)
    const AbSpatialVector newRootZ = asmSVecAdd(rootDeferredZ, finalImp);
    abStoreSpatialVector(artiBlock, PXG_AB_mRootDeferredZ, lane, newRootZ.top, newRootZ.bottom);

    if (residualReporting)
    {
        Ptr<float> ir = Ptr<float>(a + PXG_ARTI_INTERNAL_RESIDUAL);
        ir[0] = ir[0] + error.sumOfSquares;
        Ptr<int> irc = Ptr<int>(a + PXG_ARTI_INTERNAL_RESIDUAL);
        irc[1] = irc[1] + error.counter;
        if (error.maxError > ir[2])
            ir[2] = error.maxError;

        Ptr<float> cr = Ptr<float>(a + PXG_ARTI_CONTACT_RESIDUAL);
        cr[0] = cr[0] + contactError.sumOfSquares;
        Ptr<int> crc = Ptr<int>(a + PXG_ARTI_CONTACT_RESIDUAL);
        crc[1] = crc[1] + contactError.counter;
        if (contactError.maxError > cr[2])
            cr[2] = contactError.maxError;
    }
}

// ===========================================================================
// P8.123a: solveStaticConstraints, the PGS overload (internalConstraints2.cu
// :1268-1392) — the static solve of artiSolveInternalConstraints1T (the PGS
// extern at :2820).  The batch machinery (windows, exclusive-popcount offset,
// side select, the deltaV guard) is the swept TGS twin's; what diverges is
// PGS-vs-TGS (each a sweep mutant):
//
//  1. **minPen and elapsedTime are DEAD in the PGS overload** — the CUDA
//     comments the parameter names out (:1271).  The port drops them.
//  2. **NO delta/deltaQ loads** — the PGS solve pair reads velocities only
//     (mDeltaMotion/mDeltaQ stay unread; the fixture poisons them).
//  3. **The 1D dispatch carries rowsMod AND offsets the response base by
//     batch.mArticulationResponseIndex** (:1319-1321) — the TGS twin passes
//     the UNOFFSET artiResponse to its 1D body.  The contact dispatch passes
//     the unoffset base in BOTH models (the contact body offsets internally).
//  4. **doFriction is LIVE** — the PGS contact body branches on it (the TGS
//     contact body never sees it; module trap 7 above).
//  5. The 1D residual gate is scDesc->mContactErrorAccumulator.mCounter >= 0
//     (:1321) while the contact error pointer is the residual param — in the
//     engine the two are always in sync; the port passes one accumulateError
//     through, exactly as the TGS port does (documented equivalence).
// ===========================================================================

public void siSolveStaticConstraintsPgs(
    uint64_t link, uint64_t batches,
    uint64_t contactHeaders, uint64_t frictionHeaders,
    uint64_t contactPoints, uint64_t frictionPoints,
    uint64_t jointHeaders, uint64_t jointRowsCon, uint64_t jointRowsMod,
    uint64_t responses,
    inout AbSpatialVector vel, inout AbSpatialVector impulse, inout AbSpatialVector deltaV,
    uint lane, bool doFriction, bool accumulateError,
    inout SvpError error, uint constraintCounts)
{
    const uint constraintBatchOffset = abU32(link, PXG_AB_LINK_mStaticJointStartIndex, lane);
    const uint contactBatchOffset = abU32(link, PXG_AB_LINK_mStaticContactStartIndex, lane);
    const uint contactCounts = abU32(link, PXG_AB_LINK_mNbStaticContacts, lane);

    const AbSpatialVector oldVel = vel;

    const AbSpatialVector zeroV = asmSVec(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f));

    // joints first, exactly as in the TGS twin
    for (uint i = 0u; i < constraintCounts; ++i)
    {
        Ptr<PxgBlockConstraintBatch> batch =
            Ptr<PxgBlockConstraintBatch>(batches + uint64_t(constraintBatchOffset + i) * SI_BATCH_STRIDE);
        const uint mask = batch[0].mask;
        const uint offset = countbits(mask & ((1u << lane) - 1u));
        const bool aIsArti = nodeIndexIsArticulation(batch[0].bodyANodeIndex[offset]);

        AbSpatialVector vel0, vel1;
        if (aIsArti) { vel0 = vel;   vel1 = zeroV; }
        else         { vel0 = zeroV; vel1 = vel; }

        AbSpatialVector impulse0 = zeroV;
        AbSpatialVector impulse1 = zeroV;

        // trap 3: rowsMod carried, response base OFFSET (:1319)
        const uint responseIndex = batch[0].mArticulationResponseIndex;
        svpSolveExt1DBlock(batch, vel0, vel1, offset,
                           Ptr<SvpPgs1DHeader>(jointHeaders),
                           Ptr<SvpPgs1DCon>(jointRowsCon),
                           Ptr<SvpPgs1DMod>(jointRowsMod),
                           Ptr<PxgArticulationBlockResponse>(
                               responses + uint64_t(responseIndex) * 1536u),
                           impulse0, impulse1, accumulateError, 1.0f, 1.0f);

        if (aIsArti) { impulse = asmSVecAdd(impulse, impulse0); vel = vel0; }
        else         { impulse = asmSVecAdd(impulse, impulse1); vel = vel1; }
    }

    for (uint i = 0u; i < contactCounts; ++i)
    {
        Ptr<PxgBlockConstraintBatch> batch =
            Ptr<PxgBlockConstraintBatch>(batches + uint64_t(contactBatchOffset + i) * SI_BATCH_STRIDE);
        const uint mask = batch[0].mask;
        const uint offset = countbits(mask & ((1u << lane) - 1u));
        const bool aIsArti = nodeIndexIsArticulation(batch[0].bodyANodeIndex[offset]);

        AbSpatialVector vel0, vel1;
        if (aIsArti) { vel0 = vel;   vel1 = zeroV; }
        else         { vel0 = zeroV; vel1 = vel; }

        AbSpatialVector impulse0 = zeroV;
        AbSpatialVector impulse1 = zeroV;

        // trap 4: doFriction live; the contact base UNOFFSET (:1369)
        svpSolveExtContactsBlock(batch, vel0, vel1, doFriction, offset,
                                 Ptr<SvpPgsContactHeader>(contactHeaders),
                                 Ptr<SvpPgsFrictionHeader>(frictionHeaders),
                                 Ptr<SvpPgsContactPoint>(contactPoints),
                                 Ptr<SvpPgsContactFriction>(frictionPoints),
                                 Ptr<PxgArticulationBlockResponse>(responses),
                                 impulse0, impulse1, error, accumulateError, 1.0f, 1.0f);

        if (aIsArti) { impulse = asmSVecAdd(impulse, impulse0); vel = vel0; }
        else         { impulse = asmSVecAdd(impulse, impulse1); vel = vel1; }
    }

    if ((constraintCounts + contactCounts) > 0u)
        deltaV = asmSVecAdd(deltaV, asmSVecSub(vel, oldVel));
}

// ---------------------------------------------------------------------------
// test entry for the PGS static solve
// ---------------------------------------------------------------------------

struct SipgParams
{
    uint64_t link;            // @0
    uint64_t batches;         // @8
    uint64_t contactHeaders;  // @16
    uint64_t frictionHeaders; // @24
    uint64_t contactPoints;   // @32
    uint64_t frictionPoints;  // @40
    uint64_t jointHeaders;    // @48
    uint64_t jointRowsCon;    // @56
    uint64_t jointRowsMod;    // @64
    uint64_t responses;       // @72
    uint64_t io;              // @80  in: vel/impulse/deltaV (18 floats @ lane*24);
                              //      out at 32*24 + lane*24 (+error triple)
    uint doFrictionWord;      // @88
    uint accumulateError;     // @92
    uint nbLanes;             // @96
    uint pad0;                // @100 -> 104 B blob
}

[[vk::push_constant]]
ConstantBuffer<SipgParams> gSipg;

[shader("compute")]
[numthreads(32, 1, 1)]
void siStaticPgsTest(uint3 groupThreadId : SV_GroupThreadID)
{
    const uint lane = groupThreadId.x;
    if (lane >= gSipg.nbLanes)
        return;

    Ptr<float> fin = Ptr<float>(gSipg.io + uint64_t(lane * SI_IO_STRIDE) * 4u);
    AbSpatialVector vel = asmSVec(float3(fin[0], fin[1], fin[2]), float3(fin[3], fin[4], fin[5]));
    AbSpatialVector impulse = asmSVec(float3(fin[6], fin[7], fin[8]), float3(fin[9], fin[10], fin[11]));
    AbSpatialVector deltaV = asmSVec(float3(fin[12], fin[13], fin[14]), float3(fin[15], fin[16], fin[17]));

    // the CALLER reads the joint count from the link, as the kernel does
    const uint constraintCounts = abU32(gSipg.link, PXG_AB_LINK_mNbStaticJoints, lane);

    SvpError error = { 0.0f, 0, 0.0f };
    siSolveStaticConstraintsPgs(gSipg.link, gSipg.batches,
                                gSipg.contactHeaders, gSipg.frictionHeaders,
                                gSipg.contactPoints, gSipg.frictionPoints,
                                gSipg.jointHeaders, gSipg.jointRowsCon, gSipg.jointRowsMod,
                                gSipg.responses,
                                vel, impulse, deltaV,
                                lane, (gSipg.doFrictionWord & 0xFFu) != 0u,
                                gSipg.accumulateError != 0u, error, constraintCounts);

    Ptr<float> fout = Ptr<float>(gSipg.io + uint64_t(SI_IO_OUT_BASE + lane * SI_IO_STRIDE) * 4u);
    fout[0] = vel.top.x; fout[1] = vel.top.y; fout[2] = vel.top.z;
    fout[3] = vel.bottom.x; fout[4] = vel.bottom.y; fout[5] = vel.bottom.z;
    fout[6] = impulse.top.x; fout[7] = impulse.top.y; fout[8] = impulse.top.z;
    fout[9] = impulse.bottom.x; fout[10] = impulse.bottom.y; fout[11] = impulse.bottom.z;
    fout[12] = deltaV.top.x; fout[13] = deltaV.top.y; fout[14] = deltaV.top.z;
    fout[15] = deltaV.bottom.x; fout[16] = deltaV.bottom.y; fout[17] = deltaV.bottom.z;
    fout[18] = error.sumOfSquares; fout[19] = float(error.counter); fout[20] = error.maxError;
}

// ===========================================================================
// P8.123b: the per-dof internal solve, PGS instantiation (isTGS == FALSE in
// the template).  Divergences from siSolveDofConstraints, each a sweep
// mutant:
//  1. **perSubset is false always** (:1765: isTGS && ...), so effectiveDt is
//     ALWAYS the scene dt and isExternalForceEveryStep is dead — the
//     parameter is dropped here (the kernel ABI still carries the byte).
//  2. **The implicit drive desc is loaded UNCONDITIONALLY** (:1778) — on TGS
//     velocity iterations it is never read, on PGS it always is.
//  3. **The deprecated friction model reads the ACCUMULATED impulse**
//     (:1796: isTGS ? 0 : accumulated) — friction accumulates across
//     position iterations on PGS.
//  4. **BOTH drive arms call adDriveImpulse even on velocity iterations**
//     (:1811/:1832: the (isTGS && isVelIter) shortcut) — elapsedTime is live
//     on every iteration type on PGS.
//  5. **The new friction model reads the accumulated impulse always**
//     (:1845: the perSubset zero is TGS-only).
//  The threading (jointV += deltaF * response after every stage), the limit
//  writeback, the residual tick and siAccumulate are the swept TGS twin's.
// ===========================================================================

public void siSolveDofConstraintsPgs(
    uint64_t dofBlocks, uint jointOffset, uint dofCount,
    uint64_t jointForces,               // the RECORD's jointForce array (float[dof])
    float dt, float sceneDt, float invDt, float elapsedTime, float erp,
    bool isVelIter,
    bool doFrictionDrivePosLimit, bool velLimitBefore,
    AbSpatialVector childDelta, AbSpatialVector parentDelta,
    inout AbSpatialVector parentDeltaV, inout AbSpatialVector deltaV,
    inout AbSpatialVector parentV, inout AbSpatialVector childV,
    inout AbSpatialVector parentImp, inout AbSpatialVector impulse,
    bool residualReporting, inout SvxError error,
    uint lane)
{
    for (uint dof = 0u; dof < dofCount; ++dof)
    {
        const uint64_t d = dofBlocks + uint64_t(jointOffset + dof) * PXG_AB_DOF_SIZE;
        const uint motion = abU8(d, PXG_AB_DOF_mMotion, lane);
        if (motion == SI_MOTION_LOCKED)
            continue;

        const uint64_t ic = d + PXG_AB_DOF_mConstraintData;
        const float maxJointVel = abF32(ic, PXG_AB_IC_mMaxJointVelocity, lane);
        const AbSpatialVector row0 = svLoadSV(ic, PXG_AB_IC_mRow0, lane);
        const AbSpatialVector row1 = svLoadSV(ic, PXG_AB_IC_mRow1, lane);
        const float maxDriveForce = abF32(ic, PXG_AB_IC_mConstraintMaxImpulse, lane);
        const float appliedDriveImpulseOrig = abF32(ic, PXG_AB_IC_mDriveImpulse, lane);
        const float recipResponse = abF32(ic, PXG_AB_IC_mRecipResponse, lane);
        const float response = abF32(ic, PXG_AB_IC_mResponse, lane);
        const float maxFrictionForce = abF32(ic, PXG_AB_IC_mMaxFrictionForce, lane);
        const AbSpatialVector deltaVA = svLoadSV(ic, PXG_AB_IC_mDeltaVA, lane);
        const AbSpatialVector deltaVB = svLoadSV(ic, PXG_AB_IC_mDeltaVB, lane);

        // trap 1: perSubset is false — the scene dt scales every effort
        const float effectiveDt = sceneDt;

        const float staticFrictionImpulse = abF32(ic, PXG_AB_IC_mStaticFrictionEffort, lane) * effectiveDt;
        const float dynamicFrictionImpulse = abF32(ic, PXG_AB_IC_mDynamicFrictionEffort, lane) * effectiveDt;
        const float viscousFrictionCoefficient = abF32(ic, PXG_AB_IC_mViscousFrictionCoefficient, lane) * effectiveDt;

        const float maxImpulse = abF32(ic, PXG_AB_IC_mMaxEffort, lane) * effectiveDt;
        const float speedImpulseGradient = abF32(ic, PXG_AB_IC_mSpeedEffortGradient, lane) / effectiveDt;
        const float velocityDependentResistance = abF32(ic, PXG_AB_IC_mVelocityDependentResistance, lane) * effectiveDt;
        const float externalJointImpulse = (Ptr<float>(jointForces))[jointOffset + dof] * effectiveDt;
        const float maxActuatorVelocity = abF32(ic, PXG_AB_IC_mMaxActuatorVelocity, lane);

        // trap 2: loaded unconditionally on PGS
        AdDriveDesc driveDesc;
        {
            const float4 packed = abF4(ic, PXG_AB_IC_mTargetVelPlusInitialBiasX_DriveBiasCoefficientY_VelMultiplierZ_ImpulseMultiplierW, lane);
            driveDesc.tvBias = packed.x;
            driveDesc.biasCoef = packed.y;
            driveDesc.velMul = packed.z;
            driveDesc.impMul = packed.w;
            driveDesc.posBias = abF32(ic, PXG_AB_IC_mTargetPosBias, lane);
        }

        float jointV = asmInnerProduct(row1, childV) - asmInnerProduct(row0, parentV);
        const float jointDeltaP = asmInnerProduct(row1, childDelta) - asmInnerProduct(row0, parentDelta);

        float frictionDeltaF = 0.0f;
        float driveDeltaF = 0.0f;
        float posLimitDeltaF = 0.0f;
        if (doFrictionDrivePosLimit)
        {
            const bool newFrictionModel = staticFrictionImpulse != 0.0f || viscousFrictionCoefficient != 0.0f;

            // deprecated friction model; trap 3: PGS accumulates
            if (!newFrictionModel)
            {
                const float appliedFriction = abF32(ic, PXG_AB_IC_mAccumulatedFrictionImpulse, lane);
                const float frictionForce = adPClamp(-jointV * recipResponse + appliedFriction, -maxFrictionForce, maxFrictionForce);
                abStoreF32(ic, PXG_AB_IC_mAccumulatedFrictionImpulse, lane, frictionForce);
                frictionDeltaF = frictionForce - appliedFriction;
                jointV += frictionDeltaF * response;
            }

            if (maxImpulse > 0.0f)
            {
                const float appliedDriveImpulse = appliedDriveImpulseOrig;   // no perSubset zero
                // trap 4: no isVelIter shortcut on PGS
                const float unclampedImpulse = adDriveImpulse(appliedDriveImpulse, jointV, jointDeltaP, elapsedTime, driveDesc);
                const float clampedImpulse = adClampDriveImpulse(jointV, appliedDriveImpulse + externalJointImpulse, unclampedImpulse + externalJointImpulse, response, maxActuatorVelocity, maxImpulse, speedImpulseGradient, velocityDependentResistance) - externalJointImpulse;
                driveDeltaF = clampedImpulse - appliedDriveImpulse;
                abStoreF32(ic, PXG_AB_IC_mDriveImpulse, lane, appliedDriveImpulseOrig + driveDeltaF);
            }
            else
            {
                const float unclampedForce = adDriveImpulse(appliedDriveImpulseOrig, jointV, jointDeltaP, elapsedTime, driveDesc);
                const float clampedForce = adPClamp(unclampedForce, -maxDriveForce, maxDriveForce);
                driveDeltaF = clampedForce - appliedDriveImpulseOrig;
                abStoreF32(ic, PXG_AB_IC_mDriveImpulse, lane, clampedForce);
            }
            jointV += driveDeltaF * response;

            if (newFrictionModel)
            {
                // trap 5: the accumulated impulse, always
                const float appliedFriction = abF32(ic, PXG_AB_IC_mAccumulatedFrictionImpulse, lane);
                float totalImpulse = appliedFriction - jointV * recipResponse;
                totalImpulse = adFrictionImpulse(totalImpulse, staticFrictionImpulse, dynamicFrictionImpulse, viscousFrictionCoefficient, jointV);
                frictionDeltaF = totalImpulse - appliedFriction;
                abStoreF32(ic, PXG_AB_IC_mAccumulatedFrictionImpulse, lane, abF32(ic, PXG_AB_IC_mAccumulatedFrictionImpulse, lane) + frictionDeltaF);
                jointV += frictionDeltaF * response;
            }

            if (motion == SI_MOTION_LIMITED)
            {
                const float2 limitErr = abF2(ic, PXG_AB_IC_mLimitError_LowX_highY, lane);
                float lowImp = abF32(ic, PXG_AB_IC_mLowImpulse, lane);
                float highImp = abF32(ic, PXG_AB_IC_mHighImpulse, lane);
                posLimitDeltaF = adLimitImpulse(dt, invDt, isVelIter, response, recipResponse, erp, limitErr.x, limitErr.y, jointDeltaP, lowImp, highImp, jointV);
                abStoreF32(ic, PXG_AB_IC_mLowImpulse, lane, lowImp);
                abStoreF32(ic, PXG_AB_IC_mHighImpulse, lane, highImp);
            }
        }

        float velLimitDeltaF = 0.0f;
        if (velLimitBefore && (abs(jointV) > maxJointVel))
        {
            const float newJointV = adPClamp(jointV, -maxJointVel, maxJointVel);
            velLimitDeltaF = (newJointV - jointV) * recipResponse;
            jointV = newJointV;
        }

        const float deltaF = frictionDeltaF + driveDeltaF + posLimitDeltaF + velLimitDeltaF;

        // the counter ticks even for a zero deltaF
        if (residualReporting)
            svxAccumulateError(error, deltaF, recipResponse);

        siAccumulate(deltaF, deltaVA, deltaVB, row0, row1,
                     parentDeltaV, deltaV, parentV, childV, parentImp, impulse);
    }
}

// ---------------------------------------------------------------------------
// test entry for the PGS dof solve
// ---------------------------------------------------------------------------

struct SdpParams
{
    uint64_t dofBlocks;    // @0
    uint64_t jointForces;  // @8   fixture scaffolding: per-lane slices
    uint64_t link;         // @16  jointOffset/dofCount come from here, as in the kernel
    uint64_t io;           // @24
    float dt;              // @32
    float sceneDt;         // @36
    float invDt;           // @40
    float elapsedTime;     // @44
    float erp;             // @48
    uint flagsWord;        // @52  bit0 velIter, bit1 doFDPL, bit2 velBefore, bit3 residual
    uint nbLanes;          // @56
    uint maxDofs;          // @60  the per-lane jointForces slice stride (scaffolding)
}

[[vk::push_constant]]
ConstantBuffer<SdpParams> gSdp;

[shader("compute")]
[numthreads(32, 1, 1)]
void siDofPgsTest(uint3 groupThreadId : SV_GroupThreadID)
{
    const uint lane = groupThreadId.x;
    if (lane >= gSdp.nbLanes)
        return;

    Ptr<float> fin = Ptr<float>(gSdp.io + uint64_t(lane * SD_IO_STRIDE) * 4u);
    AbSpatialVector parentV = asmSVec(float3(fin[0], fin[1], fin[2]), float3(fin[3], fin[4], fin[5]));
    AbSpatialVector childV = asmSVec(float3(fin[6], fin[7], fin[8]), float3(fin[9], fin[10], fin[11]));
    AbSpatialVector parentDeltaV = asmSVec(float3(fin[12], fin[13], fin[14]), float3(fin[15], fin[16], fin[17]));
    AbSpatialVector deltaV = asmSVec(float3(fin[18], fin[19], fin[20]), float3(fin[21], fin[22], fin[23]));
    AbSpatialVector parentImp = asmSVec(float3(fin[24], fin[25], fin[26]), float3(fin[27], fin[28], fin[29]));
    AbSpatialVector impulse = asmSVec(float3(fin[30], fin[31], fin[32]), float3(fin[33], fin[34], fin[35]));
    const AbSpatialVector childDelta = asmSVec(float3(fin[36], fin[37], fin[38]), float3(fin[39], fin[40], fin[41]));
    const AbSpatialVector parentDelta = asmSVec(float3(fin[42], fin[43], fin[44]), float3(fin[45], fin[46], fin[47]));

    const uint jointOffset = abU32(gSdp.link, PXG_AB_LINK_mJointOffset, lane);
    const uint dofCount = abU8(gSdp.link, PXG_AB_LINK_mDofs, lane);
    const uint64_t jf = gSdp.jointForces + uint64_t(lane * gSdp.maxDofs) * 4u;

    SvxError error = { 0.0f, 0, 0.0f };
    siSolveDofConstraintsPgs(gSdp.dofBlocks, jointOffset, dofCount, jf,
                             gSdp.dt, gSdp.sceneDt, gSdp.invDt, gSdp.elapsedTime, gSdp.erp,
                             (gSdp.flagsWord & 1u) != 0u,
                             (gSdp.flagsWord & 2u) != 0u,
                             (gSdp.flagsWord & 4u) != 0u,
                             childDelta, parentDelta,
                             parentDeltaV, deltaV, parentV, childV, parentImp, impulse,
                             (gSdp.flagsWord & 8u) != 0u, error, lane);

    Ptr<float> fout = Ptr<float>(gSdp.io + uint64_t(SD_IO_OUT_BASE + lane * SD_IO_STRIDE) * 4u);
    fout[0] = parentV.top.x; fout[1] = parentV.top.y; fout[2] = parentV.top.z;
    fout[3] = parentV.bottom.x; fout[4] = parentV.bottom.y; fout[5] = parentV.bottom.z;
    fout[6] = childV.top.x; fout[7] = childV.top.y; fout[8] = childV.top.z;
    fout[9] = childV.bottom.x; fout[10] = childV.bottom.y; fout[11] = childV.bottom.z;
    fout[12] = parentDeltaV.top.x; fout[13] = parentDeltaV.top.y; fout[14] = parentDeltaV.top.z;
    fout[15] = parentDeltaV.bottom.x; fout[16] = parentDeltaV.bottom.y; fout[17] = parentDeltaV.bottom.z;
    fout[18] = deltaV.top.x; fout[19] = deltaV.top.y; fout[20] = deltaV.top.z;
    fout[21] = deltaV.bottom.x; fout[22] = deltaV.bottom.y; fout[23] = deltaV.bottom.z;
    fout[24] = parentImp.top.x; fout[25] = parentImp.top.y; fout[26] = parentImp.top.z;
    fout[27] = parentImp.bottom.x; fout[28] = parentImp.bottom.y; fout[29] = parentImp.bottom.z;
    fout[30] = impulse.top.x; fout[31] = impulse.top.y; fout[32] = impulse.top.z;
    fout[33] = impulse.bottom.x; fout[34] = impulse.bottom.y; fout[35] = impulse.bottom.z;
    fout[36] = error.sumOfSquares; fout[37] = float(error.counter); fout[38] = error.maxError;
}

// ===========================================================================
// P8.123c: the DFS, PGS instantiation — the traversal-stack walk of
// artiSolveInternalConstraints1T (:1662-2002 with isTGS == false).  The walk
// machinery (stack, backprop booking, the AFTER-pass with its dof-base-0
// engine quirk) is the swept TGS twin's; the dispatch diverges:
//   * siSolveDofConstraintsPgs (P8.123b — no isExternalForceEveryStep),
//   * siSolveStaticConstraintsPgs (P8.123a — doFriction LIVE, rowsMod
//     carried, no minPen/elapsedTime, no delta/deltaQ),
//   * the contact error rides an SvpError (the PGS ext-solve module's type).
// minPen and isExternalForceEveryStep are dead on the whole PGS path and are
// dropped from the signature; elapsedTime stays LIVE (adDriveImpulse runs on
// every iteration type on PGS).
// ===========================================================================

public AbSpatialVector siSolveInternalDFSPgs(
    uint64_t linkBase, uint64_t dofBlocks, uint64_t stackBase,
    uint64_t jointForces,
    uint64_t batches,
    uint64_t contactHeaders, uint64_t frictionHeaders,
    uint64_t contactPoints, uint64_t frictionPoints,
    uint64_t jointHeaders, uint64_t jointRowsCon, uint64_t jointRowsMod,
    uint64_t responses,
    float dt, float sceneDt, float invDt, float elapsedTime, float erp,
    bool isVelIter, bool doFriction,
    bool doFrictionDrivePosLimit, uint doVelLimit,   // 0 none / 1 before / 2 after
    bool doStaticContactsAnd1dConstraints,
    AbSpatialVector storeParentDeltaV, AbSpatialVector parentImpIn,
    AbSpatialVector parentDeltaVIn,
    bool residualReporting, inout SvxError error, inout SvpError contactError,
    uint lane)
{
    AbSpatialVector parentDeltaV = parentDeltaVIn;
    AbSpatialVector parentImp = parentImpIn;

    uint parent = 0u;

    abStoreSpatialVector(stackBase + uint64_t(parent) * PXG_AB_TRAVERSAL_STACK_SIZE, PXG_AB_TRAVERSAL_deltaVStack, lane, storeParentDeltaV.top, storeParentDeltaV.bottom);
    abStoreSpatialVector(stackBase + uint64_t(parent) * PXG_AB_TRAVERSAL_STACK_SIZE, PXG_AB_TRAVERSAL_impulseStack, lane, parentImp.top, parentImp.bottom);

    uint offset = abU32(linkBase + uint64_t(parent) * PXG_AB_LINK_SIZE, PXG_AB_LINK_mChildrenOffset, lane);
    uint numChildren = abU32(linkBase + uint64_t(parent) * PXG_AB_LINK_SIZE, PXG_AB_LINK_mNumChildren, lane);

    for (uint i = 0u; i < numChildren; ++i)
        abStoreU32(stackBase + uint64_t(i) * PXG_AB_TRAVERSAL_STACK_SIZE, PXG_AB_TRAVERSAL_indices, lane, offset + i);

    uint stackCount = numChildren;
    uint linkID = (stackCount != 0u)
        ? abU32(stackBase + uint64_t(stackCount - 1u) * PXG_AB_TRAVERSAL_STACK_SIZE, PXG_AB_TRAVERSAL_indices, lane)
        : 0xffffffffu;

    while (stackCount != 0u)
    {
        const bool isBackProp = (parent == linkID);

        const uint64_t link = linkBase + uint64_t(linkID) * PXG_AB_LINK_SIZE;
        const float3 c2p = float3(abF32(link, PXG_AB_LINK_mRw_x, lane),
                                  abF32(link, PXG_AB_LINK_mRw_y, lane),
                                  abF32(link, PXG_AB_LINK_mRw_z, lane));
        const uint jointOffset = abU32(link, PXG_AB_LINK_mJointOffset, lane);
        const uint dofCount = abU8(link, PXG_AB_LINK_mDofs, lane);

        if (isBackProp)
        {
            parent = abU32(link, PXG_AB_LINK_mParents, lane);

            const AbSpatialVector solverSpatialImpulse = svLoadSV(link, PXG_AB_LINK_mSolverSpatialImpulse, lane);
            const AbSpatialVector ZInternalConstraint = svLoadSV(link, PXG_AB_LINK_mSolverSpatialInternalConstraintImpulse, lane);

            const AbMatrix6x6 spatialResponse = abLoadSpatialMatrix6x6(linkBase + uint64_t(parent) * PXG_AB_LINK_SIZE, PXG_AB_LINK_mSpatialResponseMatrix, lane);

            stackCount--;

            const AbSpatialVector impulse = parentImp;

            const uint64_t pStack = stackBase + uint64_t(parent) * PXG_AB_TRAVERSAL_STACK_SIZE;
            parentDeltaV = svLoadSV(pStack, PXG_AB_TRAVERSAL_deltaVStack, lane);
            parentImp = svLoadSV(pStack, PXG_AB_TRAVERSAL_impulseStack, lane);

            // the internal-constraint share is subtracted (TGS trap 1)
            const AbSpatialVector booked = asmSVecSub(asmSVecAdd(solverSpatialImpulse, impulse), ZInternalConstraint);
            abStoreSpatialVector(link, PXG_AB_LINK_mSolverSpatialImpulse, lane, booked.top, booked.bottom);

            linkID = (stackCount > 0u)
                ? abU32(stackBase + uint64_t(stackCount - 1u) * PXG_AB_TRAVERSAL_STACK_SIZE, PXG_AB_TRAVERSAL_indices, lane)
                : 0xffffffffu;

            float jfNone[3] = { 0.0f, 0.0f, 0.0f };
            const AbSpatialVector propagateImp = svPropagateImpulseDeferred(c2p, dofBlocks, jointOffset, dofCount, impulse, lane, jfNone, 1.0f, false);

            parentImp = asmSVecAdd(parentImp, propagateImp);

            const AbSpatialVector deltaV = asmM6MulVec(spatialResponse, asmSVec(-parentImp.top, -parentImp.bottom));

            parentDeltaV = asmSVecAdd(parentDeltaV, deltaV);

            // only the impulse frame goes back to the stack (TGS trap 2)
            abStoreSpatialVector(pStack, PXG_AB_TRAVERSAL_impulseStack, lane, parentImp.top, parentImp.bottom);
        }
        else
        {
            const uint64_t parentLink = linkBase + uint64_t(parent) * PXG_AB_LINK_SIZE;
            AbSpatialVector parentV = asmSVecAdd(parentDeltaV, svLoadSV(parentLink, PXG_AB_LINK_mMotionVelocity, lane));
            const AbSpatialVector childDelta = svLoadSV(link, PXG_AB_LINK_mDeltaMotion, lane);
            const AbSpatialVector parentDelta = svLoadSV(parentLink, PXG_AB_LINK_mDeltaMotion, lane);

            float dsp[3] = { 0.0f, 0.0f, 0.0f };
            AbSpatialVector deltaV = svPropagateAccelDeferred(c2p, dofBlocks, jointOffset, dofCount, parentDeltaV, lane, dsp, false);

            AbSpatialVector childV = asmSVecAdd(deltaV, svLoadSV(link, PXG_AB_LINK_mMotionVelocity, lane));

            abStoreSpatialVector(stackBase + uint64_t(linkID) * PXG_AB_TRAVERSAL_STACK_SIZE, PXG_AB_TRAVERSAL_deltaVStack, lane, deltaV.top, deltaV.bottom);

            AbSpatialVector impulse = asmSVec(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f));

            siSolveDofConstraintsPgs(dofBlocks, jointOffset, dofCount, jointForces,
                                     dt, sceneDt, invDt, elapsedTime, erp,
                                     isVelIter,
                                     doFrictionDrivePosLimit, doVelLimit == 1u,
                                     childDelta, parentDelta,
                                     parentDeltaV, deltaV, parentV, childV, parentImp, impulse,
                                     residualReporting, error, lane);

            const uint constraintCounts = abU32(link, PXG_AB_LINK_mNbStaticJoints, lane);
            numChildren = abU32(link, PXG_AB_LINK_mNumChildren, lane);

            AbSpatialVector childImpulseInternal = impulse;

            AbSpatialVector childImpulseFromStatic = asmSVec(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f));
            if (doStaticContactsAnd1dConstraints)
            {
                const AbSpatialVector before = impulse;
                siSolveStaticConstraintsPgs(link, batches,
                                            contactHeaders, frictionHeaders,
                                            contactPoints, frictionPoints,
                                            jointHeaders, jointRowsCon, jointRowsMod,
                                            responses,
                                            childV, impulse, deltaV,
                                            lane, doFriction,
                                            residualReporting, contactError, constraintCounts);
                childImpulseFromStatic = asmSVecSub(impulse, before);
            }

            if (doVelLimit == 2u)   // eAFTER_STATIC_CONSTRAINTS
            {
                const AbSpatialVector i1BeforeVelLimit = impulse;

                // the dof array at its BASE (the engine's own spelling, TGS trap 5)
                AbSpatialVector motionB[3]; AbSpatialVector isInvDB[3]; AbSpatialVector isWB[3]; float3 invStIsTB[3];
                svLoadDofArrays(dofBlocks, 0u, dofCount, lane, motionB, isInvDB, isWB, invStIsTB);
                float qstzLocal[3] = { 0.0f, 0.0f, 0.0f };
                float jfNone2[3] = { 0.0f, 0.0f, 0.0f };
                const AbSpatialVector propagatedImpulseAtParentW = adPropagateImpulseWJf(isInvDB, c2p, motionB, childImpulseFromStatic, dofCount, qstzLocal, jfNone2, 0.0f, false);

                const AbMatrix6x6 mat = abLoadSpatialMatrix6x6(parentLink, PXG_AB_LINK_mSpatialResponseMatrix, lane);
                const AbSpatialVector deltaVParent = asmM6MulVec(mat, asmSVec(-propagatedImpulseAtParentW.top, -propagatedImpulseAtParentW.bottom));

                const AbSpatialVector parentVCorrected = asmSVecAdd(parentV, deltaVParent);

                for (uint dof = 0u; dof < dofCount; ++dof)
                {
                    const uint64_t d = dofBlocks + uint64_t(jointOffset + dof) * PXG_AB_DOF_SIZE;
                    const uint motion = abU8(d, PXG_AB_DOF_mMotion, lane);
                    if (motion == SI_MOTION_LOCKED)
                        continue;
                    const uint64_t ic = d + PXG_AB_DOF_mConstraintData;
                    const float maxJointVel = abF32(ic, PXG_AB_IC_mMaxJointVelocity, lane);
                    const AbSpatialVector row0 = svLoadSV(ic, PXG_AB_IC_mRow0, lane);
                    const AbSpatialVector row1 = svLoadSV(ic, PXG_AB_IC_mRow1, lane);
                    const float recipResponse = abF32(ic, PXG_AB_IC_mRecipResponse, lane);
                    const AbSpatialVector dVA = svLoadSV(ic, PXG_AB_IC_mDeltaVA, lane);
                    const AbSpatialVector dVB = svLoadSV(ic, PXG_AB_IC_mDeltaVB, lane);
                    float jointV = asmInnerProduct(row1, childV) - asmInnerProduct(row0, parentVCorrected);

                    float velLimitDeltaF = 0.0f;
                    if (abs(jointV) > maxJointVel)
                    {
                        const float newJointV = adPClamp(jointV, -maxJointVel, maxJointVel);
                        velLimitDeltaF = (newJointV - jointV) * recipResponse;
                        jointV = newJointV;
                    }

                    if (residualReporting)
                        svxAccumulateError(error, velLimitDeltaF, recipResponse);

                    siAccumulate(velLimitDeltaF, dVA, dVB, row0, row1,
                                 parentDeltaV, deltaV, parentV, childV, parentImp, impulse);
                }

                // only the velocity-limit share joins the internal booking (TGS trap 7)
                childImpulseInternal = asmSVecAdd(childImpulseInternal, asmSVecSub(impulse, i1BeforeVelLimit));
            }

            abStoreSpatialVector(link, PXG_AB_LINK_mSolverSpatialInternalConstraintImpulse, lane, childImpulseInternal.top, childImpulseInternal.bottom);

            abStoreSpatialVector(stackBase + uint64_t(parent) * PXG_AB_TRAVERSAL_STACK_SIZE, PXG_AB_TRAVERSAL_impulseStack, lane, parentImp.top, parentImp.bottom);
            abStoreSpatialVector(stackBase + uint64_t(linkID) * PXG_AB_TRAVERSAL_STACK_SIZE, PXG_AB_TRAVERSAL_impulseStack, lane, impulse.top, impulse.bottom);

            {
                parent = linkID;
                // with no children the linkID stays -> next round backprops (TGS trap 3)
                if (numChildren > 0u)
                {
                    offset = abU32(link, PXG_AB_LINK_mChildrenOffset, lane);
                    for (uint i = 0u; i < numChildren; ++i)
                    {
                        abStoreU32(stackBase + uint64_t(stackCount) * PXG_AB_TRAVERSAL_STACK_SIZE, PXG_AB_TRAVERSAL_indices, lane, offset + i);
                        stackCount++;
                    }
                    linkID = abU32(stackBase + uint64_t(stackCount - 1u) * PXG_AB_TRAVERSAL_STACK_SIZE, PXG_AB_TRAVERSAL_indices, lane);
                }

                parentDeltaV = deltaV;
                parentImp = impulse;
            }
        }
    }

    return parentImp;
}

// ---------------------------------------------------------------------------
// test entry for the PGS DFS
// ---------------------------------------------------------------------------

struct SfpParams
{
    uint64_t linkBase;        // @0
    uint64_t dofBlocks;       // @8
    uint64_t stackBase;       // @16
    uint64_t jointForces;     // @24  fixture scaffolding: per-lane slices
    uint64_t batches;         // @32
    uint64_t contactHeaders;  // @40
    uint64_t frictionHeaders; // @48
    uint64_t contactPoints;   // @56
    uint64_t frictionPoints;  // @64
    uint64_t jointHeaders;    // @72
    uint64_t jointRowsCon;    // @80
    uint64_t jointRowsMod;    // @88
    uint64_t responses;       // @96
    uint64_t io;              // @104
    float dt;                 // @112
    float sceneDt;            // @116
    float invDt;              // @120
    float elapsedTime;        // @124
    float erp;                // @128
    uint flagsWord;           // @132  bit0 velIter, bit1 doFriction, bit2 doFDPL,
                              //       bits 3-4 doVelLimit, bit5 doStatic, bit6 residual
    uint nbLanes;             // @136
    uint maxDofs;             // @140  jointForces slice stride (scaffolding) -> 144 B
}

[[vk::push_constant]]
ConstantBuffer<SfpParams> gSfp;

[shader("compute")]
[numthreads(32, 1, 1)]
void siDfsPgsTest(uint3 groupThreadId : SV_GroupThreadID)
{
    const uint lane = groupThreadId.x;
    if (lane >= gSfp.nbLanes)
        return;

    Ptr<float> fin = Ptr<float>(gSfp.io + uint64_t(lane * SF_IO_STRIDE) * 4u);
    const AbSpatialVector storeParentDeltaV = asmSVec(float3(fin[0], fin[1], fin[2]), float3(fin[3], fin[4], fin[5]));
    const AbSpatialVector parentImpIn = asmSVec(float3(fin[6], fin[7], fin[8]), float3(fin[9], fin[10], fin[11]));
    const AbSpatialVector parentDeltaVIn = asmSVec(float3(fin[12], fin[13], fin[14]), float3(fin[15], fin[16], fin[17]));

    const uint64_t jf = gSfp.jointForces + uint64_t(lane) * uint64_t(gSfp.maxDofs) * 4u;

    const bool velIter = (gSfp.flagsWord & 1u) != 0u;
    const bool doFriction = (gSfp.flagsWord & 2u) != 0u;
    const bool doFDPL = (gSfp.flagsWord & 4u) != 0u;
    const uint doVelLimit = (gSfp.flagsWord >> 3u) & 3u;
    const bool doStatic = (gSfp.flagsWord & 32u) != 0u;
    const bool residual = (gSfp.flagsWord & 64u) != 0u;

    SvxError error = { 0.0f, 0, 0.0f };
    SvpError contactError = { 0.0f, 0, 0.0f };
    const AbSpatialVector finalImp = siSolveInternalDFSPgs(
        gSfp.linkBase, gSfp.dofBlocks, gSfp.stackBase, jf,
        gSfp.batches, gSfp.contactHeaders, gSfp.frictionHeaders,
        gSfp.contactPoints, gSfp.frictionPoints,
        gSfp.jointHeaders, gSfp.jointRowsCon, gSfp.jointRowsMod, gSfp.responses,
        gSfp.dt, gSfp.sceneDt, gSfp.invDt, gSfp.elapsedTime, gSfp.erp,
        velIter, doFriction, doFDPL, doVelLimit, doStatic,
        storeParentDeltaV, parentImpIn, parentDeltaVIn,
        residual, error, contactError, lane);

    Ptr<float> fout = Ptr<float>(gSfp.io + uint64_t(SF_IO_OUT_BASE + lane * SF_IO_STRIDE) * 4u);
    fout[0] = finalImp.top.x; fout[1] = finalImp.top.y; fout[2] = finalImp.top.z;
    fout[3] = finalImp.bottom.x; fout[4] = finalImp.bottom.y; fout[5] = finalImp.bottom.z;
    fout[6] = error.sumOfSquares; fout[7] = float(error.counter); fout[8] = error.maxError;
    fout[9] = contactError.sumOfSquares; fout[10] = float(contactError.counter); fout[11] = contactError.maxError;
}

// ===========================================================================
// P8.123d: the kernel — artiSolveInternalConstraints1T (the PGS extern,
// internalConstraints2.cu:2820, template body :1579-2012 with
// IterativeSolveData and isTGS == false), composing the P8.123a/b/c pieces.
// The prologue/epilogue (record resolve, residual reset, aiAverageAndPropagate
// without scale, mStateDirty = eVEL_DIRTY, the root static solve shape, the
// mRootDeferredZ booking, the no-atomics residual push) is the swept TGS
// twin's.  Wrapper-level PGS facts, each a sweep mutant unless marked:
//  * **erp = biasCoefficient UNCLAMPED** — the TGS extern computes
//    min(0.7, biasCoefficient) (:2852); the PGS extern (:2820) passes the
//    argument straight through.
//  * **doFriction (flags byte 0) is LIVE** — the PGS static solve branches
//    on it (dead on the TGS path).
//  * **minPen does not exist** — the template computes it (:1597) but the
//    PGS static overload never reads it; not computed here.
//  * **isExternalForceEveryStep (flags byte 2) is DEAD** — the PGS dof solve
//    has no perSubset; the ABI byte is carried, the fixture varies it and
//    expects identity (deadness proof, not swept — no code to mutate).
//  * elapsedTime is LIVE on both iteration types (adDriveImpulse always runs).
//  * the shared desc is PxgSolverSharedDescPGSGpu — same 224-byte layout,
//    with iter_blockJointConstraintRowsMod (@120) additionally consumed.
//  * ABI row kArtiSolveInternalTGS reused (identical extern signature).
// ===========================================================================

[shader("compute")]
[numthreads(32, 1, 1)]
void artiSolveInternalConstraints1T(uint3 groupThreadId : SV_GroupThreadID,
                                    uint3 groupId : SV_GroupID)
{
    const uint64_t d = gSip.scDesc;
    const uint lane = groupThreadId.x;
    const uint globalThreadIndex = groupId.x * 32u + lane;   // blockStride = blockDim.x

    const uint nbArticulations = artiU32Field(d, PXG_ARTI_CORE_NB_ARTICULATIONS);
    if (globalThreadIndex >= nbArticulations)
        return;

    const bool velocityIteration = (gSip.velIterWord & 0xFFu) != 0u;
    const bool doFriction = (gSip.solveFlagsWord & 0xFFu) != 0u;   // LIVE on PGS
    const bool residualReporting = ((gSip.solveFlagsWord >> 8u) & 0xFFu) != 0u;
    const bool doFrictionDrivePosLimit = ((gSip.solveFlagsWord >> 24u) & 0xFFu) != 0u;
    const bool doStatic = (gSip.doStaticWord & 0xFFu) != 0u;

    const float erp = gSip.biasCoefficient;                          // UNCLAMPED (:2820)
    const float sceneDt = asfloat(artiU32Field(d, PXG_ARTI_CORE_DT));   // :1766

    SvxError error = { 0.0f, 0, 0.0f };
    SvpError contactError = { 0.0f, 0, 0.0f };

    // record resolve (:1605-1610)
    const uint articulationOffset = artiU32Field(d, PXG_ARTI_CORE_ARTICULATION_OFFSET);
    const uint64_t islandNodeIndices = artiPtrField(d, PXG_ARTI_CORE_ISLAND_NODE_INDICES);
    const uint2 node = (Ptr<uint2>(islandNodeIndices))[globalThreadIndex + articulationOffset];
    const uint nodeIndex = nodeIndexIndex(node);
    const uint64_t bodySim = artiPtrField(d, PXG_ARTI_CORE_BODY_SIM_POOL)
                           + uint64_t(nodeIndex) * PXG_BODY_SIM_SIZE;
    const uint articulationIndex = artiU32Field(bodySim, PXG_BODY_SIM_ARTICULATION_REMAP_ID);
    const uint64_t a = artiPtrField(d, PXG_ARTI_CORE_ARTICULATIONS)
                     + uint64_t(articulationIndex) * PXG_ARTI_STRIDE;

    if (residualReporting)
    {
        // Dy::ErrorAccumulator::reset() on both record accumulators (:1614-1615)
        Ptr<float> ir = Ptr<float>(a + PXG_ARTI_INTERNAL_RESIDUAL);
        ir[0] = 0.0f; (Ptr<int>(a + PXG_ARTI_INTERNAL_RESIDUAL))[1] = 0; ir[2] = 0.0f;
        Ptr<float> cr = Ptr<float>(a + PXG_ARTI_CONTACT_RESIDUAL);
        cr[0] = 0.0f; (Ptr<int>(a + PXG_ARTI_CONTACT_RESIDUAL))[1] = 0; cr[2] = 0.0f;
    }

    const uint64_t w = uint64_t(groupId.x);   // the per-warp slice index, DIRECTLY
    const uint maxLinks = artiU32Field(d, PXG_ARTI_CORE_MAX_LINKS);
    const uint maxDofs = artiU32Field(d, PXG_ARTI_CORE_MAX_DOFS);
    const uint64_t artiBlock = artiPtrField(d, PXG_ARTI_CORE_ARTICULATION_BLOCKS)
                             + w * PXG_AB_BLOCK_SIZE;
    const uint64_t linkBase = artiPtrField(d, PXG_ARTI_CORE_LINK_BLOCKS)
                            + w * uint64_t(maxLinks) * PXG_AB_LINK_SIZE;
    const uint64_t dofBase = artiPtrField(d, PXG_ARTI_CORE_DOF_BLOCKS)
                           + w * uint64_t(maxDofs) * PXG_AB_DOF_SIZE;

    // averaging preamble (:1628-1632): NO scale — the internal-count branch
    if ((abU8(artiBlock, PXG_AB_mStateDirty, lane) & SIP_HAS_IMPULSES) != 0u)
    {
        const uint nbSlabs = artiU32Field(d, PXG_ARTI_CORE_NB_SLABS);
        const uint64_t slabHasChanges = artiPtrField(d, PXG_ARTI_CORE_SLAB_HAS_CHANGES);
        const uint64_t impulses = artiPtrField(d, PXG_ARTI_CORE_IMPULSES);
        aiAverageAndPropagate(slabHasChanges, impulses, artiBlock, linkBase, dofBase,
                              globalThreadIndex, maxLinks, nbArticulations, nbSlabs, lane);
    }
    abStoreU8(artiBlock, PXG_AB_mStateDirty, lane, SIP_VEL_DIRTY);   // ALWAYS (:1633)

    const bool fixBase = (abU32(artiBlock, PXG_AB_mFlags, lane) & PXG_ARTI_FLAG_FIX_BASE) != 0u;
    const AbSpatialVector rootDeferredZ = svLoadSV(artiBlock, PXG_AB_mRootDeferredZ, lane);

    const AbSpatialVector zeroSV = asmSVec(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f));
    AbSpatialVector parentDeltaV = zeroSV;
    AbSpatialVector storeParentDeltaV = zeroSV;
    AbSpatialVector parentImp = zeroSV;

    Ptr<PxgSolverSharedDescPGSGpu> sharedDesc = Ptr<PxgSolverSharedDescPGSGpu>(gSip.sharedDesc);
    const uint64_t batches = sharedDesc[0].iter_blockConstraintBatch;
    const uint64_t contactHeaders = sharedDesc[0].iter_blockContactHeaders;
    const uint64_t frictionHeaders = sharedDesc[0].iter_blockFrictionHeaders;
    const uint64_t contactPoints = sharedDesc[0].iter_blockContactPoints;
    const uint64_t frictionPoints = sharedDesc[0].iter_blockFrictions;
    const uint64_t jointHeaders = sharedDesc[0].iter_blockJointConstraintHeaders;
    const uint64_t jointRowsCon = sharedDesc[0].iter_blockJointConstraintRowsCon;
    const uint64_t jointRowsMod = sharedDesc[0].iter_blockJointConstraintRowsMod;
    const uint64_t responses = sharedDesc[0].iter_artiResponse;

    // the root static solve (:1643-1660)
    if (!fixBase && doStatic)
    {
        const uint constraintCounts0 = abU32(linkBase, PXG_AB_LINK_mNbStaticJoints, lane);
        const AbSpatialMatrix m =
            asmLoadSpatialMatrix(artiBlock, PXG_AB_mInvSpatialArticulatedInertia, lane);
        const AbSpatialVector motionVelocity0 = svLoadSV(linkBase, PXG_AB_LINK_mMotionVelocity, lane);

        parentDeltaV = asmMulVec(m, asmSVec(-rootDeferredZ.top, -rootDeferredZ.bottom));
        AbSpatialVector rootVel = asmSVecAdd(motionVelocity0, parentDeltaV);
        storeParentDeltaV = parentDeltaV;   // captured BEFORE the solve mutates parentDeltaV

        siSolveStaticConstraintsPgs(linkBase, batches,
                                    contactHeaders, frictionHeaders,
                                    contactPoints, frictionPoints,
                                    jointHeaders, jointRowsCon, jointRowsMod,
                                    responses,
                                    rootVel, parentImp, parentDeltaV,
                                    lane, doFriction,
                                    residualReporting, contactError, constraintCounts0);
    }

    const uint64_t stackBase = artiPtrField(d, PXG_ARTI_CORE_TRAVERSAL_STACK)
                             + w * uint64_t(maxLinks) * PXG_AB_TRAVERSAL_STACK_SIZE;
    const uint64_t jointForces = artiPtrField(a, PXG_ARTI_JOINT_FORCE);

    const AbSpatialVector finalImp = siSolveInternalDFSPgs(
        linkBase, dofBase, stackBase, jointForces,
        batches, contactHeaders, frictionHeaders, contactPoints, frictionPoints,
        jointHeaders, jointRowsCon, jointRowsMod, responses,
        gSip.dt, sceneDt, gSip.invDt, gSip.elapsedTime, erp,
        velocityIteration, doFriction,
        doFrictionDrivePosLimit, gSip.doVelLimit, doStatic,
        storeParentDeltaV, parentImp, parentDeltaV,
        residualReporting, error, contactError, lane);

    // epilogue (:2004-2010)
    const AbSpatialVector newRootZ = asmSVecAdd(rootDeferredZ, finalImp);
    abStoreSpatialVector(artiBlock, PXG_AB_mRootDeferredZ, lane, newRootZ.top, newRootZ.bottom);

    if (residualReporting)
    {
        Ptr<float> ir = Ptr<float>(a + PXG_ARTI_INTERNAL_RESIDUAL);
        ir[0] = ir[0] + error.sumOfSquares;
        Ptr<int> irc = Ptr<int>(a + PXG_ARTI_INTERNAL_RESIDUAL);
        irc[1] = irc[1] + error.counter;
        if (error.maxError > ir[2])
            ir[2] = error.maxError;

        Ptr<float> cr = Ptr<float>(a + PXG_ARTI_CONTACT_RESIDUAL);
        cr[0] = cr[0] + contactError.sumOfSquares;
        Ptr<int> crc = Ptr<int>(a + PXG_ARTI_CONTACT_RESIDUAL);
        crc[1] = crc[1] + contactError.counter;
        if (contactError.maxError > cr[2])
            cr[2] = contactError.maxError;
    }
}
