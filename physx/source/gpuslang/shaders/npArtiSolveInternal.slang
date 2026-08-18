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
import npArtiSolveTendons;
import npArtiSolveExtTGS;

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
                const float frictionForce = adPClamp(-jointV * recipResponse + appliedFriction, -maxFrictionForce, maxFrictionForce);
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
