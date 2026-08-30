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


// npArtiSetupInternal.slang
//
// P8.97a: `setupInternalConstraints` (internalConstraints2.cu:122-352) plus its
// two private helpers `propagateImpulseDofAligned` (:53) and
// `getImpulseSelfResponseDofAligned` (:64) -- the per-dof drive/limit/friction
// setup that fills the internal-constraint block for the solve chain.  The
// wrapper kernel `setupInternalConstraintLaunch1T` and the three tendon/mimic
// setups are P8.97b/c.
//
// Almost everything here is composition: the propagate algebra and the
// implicit-drive scalar family live in pxgSlangArticulationDynamic (the drive
// family LIFTED there from npArtiDriveParams this tranche), the 6x6 response
// apply is P8.91a's asmM6MulVec, and every block field was mapped in
// P8.68/P8.75/P8.87.  ZERO new ABI.
//
// TRAPS, from the reading (each with a fixture case / mutant):
//  1. `driveError` is per-lane scratch that PERSISTS ACROSS LINKS (CUDA:
//     __shared__ indexed by lane, no cross-lane traffic -- ported as a plain
//     local, which is semantically identical).  The 1-dof branch writes ONE
//     component (index mDofIds[0]); the spherical branch writes all three but
//     only under hasAngularDrives.  Stale components are only ever READ for
//     dofs whose drive guard is false (the value is dead), so a port that
//     zeroes it per link is behaviour-equivalent -- the port keeps the CUDA's
//     lifetime anyway.
//  2. The spherical drive axis accumulates the LOCAL motion top
//     (mLocalMotionMatrix), not the world one, scaled by mDriveTargetPos.
//  3. PxVec3::normalize divides ONLY when magnitude > 0 (P8.84's trap 4);
//     angle < 1e-12 falls back to axis (1,0,0), angle 0.
//  4. targetQ flips sign when targetQ.dot(qB2qA) < 0 (4D dot).
//  5. driveError(spherical) = -2 * imaginary(conj(targetQ) * qB2qA).
//  6. The LINEAR path's error reads dofs[0].mDriveTargetPos -- dofs[0], not
//     dofs[i]; only prismatic (1 dof) reaches it, so they coincide.  As-is.
//  7. cA2w/cB2w in the linear path are the child/parent joint-pose POSITIONS
//     rotated by the link poses' quats -- rotate only, no translation add.
//  8. transmissionForce = |bias| * (frictionCoefficient * stepDt), where
//     |bias| is UnAlignedSpatialVector::magnitude = |top| + |bottom| -- the
//     SUM of two vector magnitudes, not sqrt of the total.
//  9. cfm = max(parent.mCfm, link.mCfm); recipResponse = 1/(unitResponse+cfm)
//     UNGUARDED (engine behaviour, transcribed).
// 10. mRow1 stores -axis1 (axis1 is already the negated axis; the store
//     negates again).
// 11. Drive guard: (maxForce>0 || maxEffort>0) && (stiffness>0 || damping>0);
//     the else branch writes a ZERO desc and zero mConstraintMaxImpulse.
// 12. maxForceScale = (flags & eDRIVE_LIMITS_ARE_FORCES) ? dt : 1.
// 13. mLimitError written only for eLIMITED motion.
// 14. computeImplicitDriveParams gets isTGSSolver ? stepDt : dt as its dt and
//     dt as simDt.
// 15. The dof cursor advances by nbDofs per link, even past driveless links.
// 16. isLinear = dofId >= eX(3); eLOCKED dofs still run the response path.
//
// The spherical drive-error path is the ONE transcendental site (sin/cos in
// the angle-axis quat) -- llvmpipe transcendentals are ~2 ULP off glibc
// (P8.94a), so the fixture compares drive-desc outputs of spherical-drive
// cases with a ULP bound and everything else bit-exact.
//
// SWEEP 16/18; the two survivors are EQUIVALENT BY THE CODE'S OWN STRUCTURE,
// not fixture gaps:
//  * norm-unguarded (drop normalize's m>0 guard): the only input where the
//    guard matters is angle == 0, and there the angle<1e-12 fallback
//    overwrites driveAxis with (1,0,0) anyway -- the NaN the unguarded
//    divide produces is discarded on every path that could see it.
//  * fallback-dropped (drop the angle<1e-12 branch): at angle == 0 exactly,
//    sin(0) = 0 zeroes the axis contribution, so targetQ is the identity
//    with or without the fallback.  The branch changes behaviour only for
//    |driveAxis| in (0, 1e-12), where the resulting driveError shift is
//    bounded by ~1e-12 -- eight orders below the spherical ULP tolerance
//    and any physically reachable target.  Kept for CUDA diffability.

import pxgSlangCommon;
import pxgSlangArticulationBlocks;
import pxgSlangSpatialMatrix;
import pxgSlangArticulationDynamic;

// PxArticulationJointType
static const uint SI_JOINT_FIX = 0u;
static const uint SI_JOINT_PRISMATIC = 1u;
static const uint SI_JOINT_REVOLUTE = 2u;
static const uint SI_JOINT_REVOLUTE_UNWRAPPED = 3u;
static const uint SI_JOINT_SPHERICAL = 4u;
// PxArticulationMotion
static const uint SI_MOTION_LOCKED = 0u;
static const uint SI_MOTION_LIMITED = 1u;
// PxArticulationDriveType (PxSolverDefs.h:323): eFORCE=0, eACCELERATION=1, eNONE=2
static const uint SI_DRIVE_NONE = 2u;
static const uint SI_DRIVE_FORCE = 0u;
static const uint SI_DRIVE_ACCELERATION = 1u;
// PxArticulationFlag::eDRIVE_LIMITS_ARE_FORCES (PxSolverDefs.h:315, 1<<1)
static const uint SI_FLAG_DRIVE_LIMITS_ARE_FORCES = 2u;
// PxArticulationAxis::eX
static const uint SI_AXIS_EX = 3u;

AbSpatialVector siLoadSV(uint64_t base, uint fieldOffset, uint lane)
{
    float3 top, bottom;
    abLoadSpatialVector(base, fieldOffset, lane, top, bottom);
    return asmSVec(top, bottom);
}

// propagateImpulseDofAligned (internalConstraints2.cu:53): single-dof impulse
// propagation whose qstZ is an ASSIGNMENT (-stZ), not an accumulation.
AbSpatialVector siPropagateImpulseDofAligned(AbSpatialVector isInvD, float3 childToParent,
                                             AbSpatialVector sa, AbSpatialVector Z,
                                             out float qstZ)
{
    const float stZ = asmInnerProduct(sa, Z);
    const AbSpatialVector temp = asmSVecScale(isInvD, stZ);
    qstZ = -stZ;
    return adTranslateSpatialVector(childToParent, asmSVecSub(Z, temp));
}

// getImpulseSelfResponseDofAligned (internalConstraints2.cu:64).  `dofBase` is
// the FIRST dof of the link (the CUDA's `dofs`), `thisDof` = dofBase + i.
void siGetImpulseSelfResponseDofAligned(AbSpatialVector impulse0, AbSpatialVector impulse1,
                                        out AbSpatialVector deltaV0, out AbSpatialVector deltaV1,
                                        uint64_t thisDof, uint64_t dofBase,
                                        float3 childToParent, AbMatrix6x6 parentResponse,
                                        AbSpatialVector thisMotionVector,
                                        uint dofCount, uint dofIndex, uint lane)
{
    const AbSpatialVector Z1W = asmSVec(-impulse1.top, -impulse1.bottom);

    const AbSpatialVector isInvD = siLoadSV(thisDof, PXG_AB_DOF_mIsInvDW, lane);

    float qstZ[3] = { 0.0f, 0.0f, 0.0f };
    float qstZDof;
    const AbSpatialVector Z0W =
        siPropagateImpulseDofAligned(isInvD, childToParent, thisMotionVector, Z1W, qstZDof);
    qstZ[dofIndex] = qstZDof;

    const AbSpatialVector impulseDifW = asmSVecSub(impulse0, Z0W);

    const AbSpatialVector deltaV0W = asmM6MulVec(parentResponse, impulseDifW);

    // preload the per-dof arrays for the acceleration propagation
    AbSpatialVector isW[3];
    AbSpatialVector motionMatrix[3];
    float3 invStIsT[3];
    for (uint ind = 0u; ind < 3u; ++ind)
    {
        isW[ind] = asmSVec(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f));
        motionMatrix[ind] = isW[ind];
        invStIsT[ind] = float3(0.0f, 0.0f, 0.0f);
        if (ind < dofCount)
        {
            const uint64_t d = dofBase + uint64_t(ind) * PXG_AB_DOF_SIZE;
            isW[ind] = siLoadSV(d, PXG_AB_DOF_mIsW, lane);
            motionMatrix[ind] = siLoadSV(d, PXG_AB_DOF_mWorldMotionMatrix, lane);
            invStIsT[ind] = float3(abF32(d, PXG_AB_DOF_mInvStIsT_x, lane),
                                   abF32(d, PXG_AB_DOF_mInvStIsT_y, lane),
                                   abF32(d, PXG_AB_DOF_mInvStIsT_z, lane));
        }
    }

    // the CUDA passes `childToParent` into a parameter NAMED parentToChild and
    // the body negates it -- adPropagateAccelerationW does the same, so the
    // value passes through unchanged (mRw is the parent-to-child offset; the
    // engine's local name `child2Parent` is the misleading one)
    deltaV1 = adPropagateAccelerationW(childToParent, invStIsT, motionMatrix, deltaV0W,
                                       dofCount, isW, qstZ);

    deltaV0 = deltaV0W;
}

// setupInternalConstraints (internalConstraints2.cu:122).  One thread per
// articulation; bases are this articulation's slices.
public void siSetupInternalConstraints(uint64_t artiBlock, uint64_t linkBase, uint64_t dofBase,
                                       float stepDt, float dt, float invDt, bool isTGSSolver,
                                       uint lane)
{
    const uint numLinks = abU32(artiBlock, PXG_AB_mNumLinks, lane);

    const float maxForceScale =
        (abU32(artiBlock, PXG_AB_mFlags, lane) & SI_FLAG_DRIVE_LIMITS_ARE_FORCES) != 0u
            ? dt : 1.0f;

    // trap 1: per-lane scratch, persists across links (CUDA: __shared__)
    float driveError[3] = { 0.0f, 0.0f, 0.0f };

    uint64_t dofs = dofBase;

    for (uint linkID = 1u; linkID < numLinks; ++linkID)
    {
        const uint64_t link = linkBase + uint64_t(linkID) * PXG_AB_LINK_SIZE;

        const uint nbDofs = abU8(link, PXG_AB_LINK_mDofs, lane);
        const uint parent = abU32(link, PXG_AB_LINK_mParents, lane);
        const uint64_t parentLink = linkBase + uint64_t(parent) * PXG_AB_LINK_SIZE;

        AbMatrix6x6 parentResponse;

        const float3 child2Parent = float3(abF32(link, PXG_AB_LINK_mRw_x, lane),
                                           abF32(link, PXG_AB_LINK_mRw_y, lane),
                                           abF32(link, PXG_AB_LINK_mRw_z, lane));
        float transmissionForce = 0.0f;
        float cfm = 0.0f;

        bool loaded = false;

        if (nbDofs != 0u)
        {
            const uint64_t ic0 = dofs + PXG_AB_DOF_mConstraintData;
            const float frictionCoefficient =
                abF32(ic0, PXG_AB_IC_mFrictionCoefficient, lane) * stepDt;

            const uint jointType = abU8(link, PXG_AB_LINK_mJointType, lane);
            const bool isAngularConstraint = jointType == SI_JOINT_REVOLUTE
                                          || jointType == SI_JOINT_REVOLUTE_UNWRAPPED
                                          || jointType == SI_JOINT_SPHERICAL;

            if (isAngularConstraint)
            {
                if (nbDofs > 1u)
                {
                    const float4 cRot = abF4(link, PXG_AB_LINK_mAccumulatedPose + PXG_AB_STB_Q, lane);
                    const float4 pRot = abF4(parentLink, PXG_AB_LINK_mAccumulatedPose + PXG_AB_STB_Q, lane);
                    const float4 parentQ = abF4(link, PXG_AB_LINK_mParentPose + PXG_AB_STB_Q, lane);
                    const float4 childQ = abF4(link, PXG_AB_LINK_mChildPose + PXG_AB_STB_Q, lane);

                    const float4 cA2w = quatMul(pRot, parentQ);
                    const float4 cB2w = quatMul(cRot, childQ);

                    const float4 qB2qA = quatMul(quatConjugate(cA2w), cB2w);

                    float3 driveAxis = float3(0.0f, 0.0f, 0.0f);

                    bool hasAngularDrives = false;

                    for (uint i = 0u; i < nbDofs; ++i)
                    {
                        const uint64_t d = dofs + uint64_t(i) * PXG_AB_DOF_SIZE;
                        const uint64_t ic = d + PXG_AB_DOF_mConstraintData;
                        const bool hasDrive =
                            abU8(d, PXG_AB_DOF_mMotion, lane) != SI_MOTION_LOCKED
                            && abU32(ic, PXG_AB_IC_mDriveType, lane) != SI_DRIVE_NONE;

                        if (hasDrive)
                        {
                            // trap 2: LOCAL motion matrix top
                            const float4 top = abF4(d, PXG_AB_DOF_mLocalMotionMatrix, lane);
                            const float targetPos = abF32(ic, PXG_AB_IC_mDriveTargetPos, lane);
                            driveAxis += float3(top.x, top.y, top.z) * targetPos;
                            hasAngularDrives = true;
                        }
                    }

                    if (hasAngularDrives)
                    {
                        // trap 3: PxVec3::normalize -- scale only when m > 0.  P9.36:
                        // internalConstraints2.cu:200 is `driveAxis.normalize()`, which
                        // reaches operator/= (PxVec3.h:260) -- ONE reciprocal, three
                        // multiplies, not three divides.
                        float angle = sqrt(driveAxis.x * driveAxis.x + driveAxis.y * driveAxis.y
                                         + driveAxis.z * driveAxis.z);
                        if (angle > 0.0f)
                            driveAxis = driveAxis * (1.0f / angle);

                        if (angle < 1e-12f)
                        {
                            driveAxis = float3(1.0f, 0.0f, 0.0f);
                            angle = 0.0f;
                        }

                        // PxQuat(angle, unitAxis): v*sin(a/2), w=cos(a/2)
                        const float half = angle * 0.5f;
                        const float s = sin(half);
                        float4 targetQ = float4(driveAxis * s, cos(half));

                        // trap 4: 4D dot sign flip
                        if (dot(targetQ, qB2qA) < 0.0f)
                            targetQ = -targetQ;

                        // trap 5
                        const float4 errQ = quatMul(quatConjugate(targetQ), qB2qA);
                        driveError[0] = -2.0f * errQ.x;
                        driveError[1] = -2.0f * errQ.y;
                        driveError[2] = -2.0f * errQ.z;
                    }
                }
                else
                {
                    const uint64_t ic = dofs + PXG_AB_DOF_mConstraintData;
                    if (abU8(dofs, PXG_AB_DOF_mMotion, lane) != SI_MOTION_LOCKED)
                    {
                        const uint i = abU8(dofs, PXG_AB_DOF_mDofIds, lane);
                        const float jointPos = abF32(dofs, PXG_AB_DOF_mJointPositions, lane);
                        driveError[i] =
                            abF32(ic, PXG_AB_IC_mDriveTargetPos, lane) - jointPos;
                    }
                }
            }

            for (uint i = 0u; i < nbDofs; ++i)
            {
                const uint64_t d = dofs + uint64_t(i) * PXG_AB_DOF_SIZE;
                const uint64_t ic = d + PXG_AB_DOF_mConstraintData;

                const uint motion = abU8(d, PXG_AB_DOF_mMotion, lane);
                const uint dofId = abU8(d, PXG_AB_DOF_mDofIds, lane);

                const float maxForce = abF32(ic, PXG_AB_IC_mMaxForce, lane);
                const float maxEffort = abF32(ic, PXG_AB_IC_mMaxEffort, lane);

                const float stiffness = abF32(ic, PXG_AB_IC_mDriveStiffness, lane);
                const float damping = abF32(ic, PXG_AB_IC_mDamping, lane);

                if (!loaded)
                {
                    parentResponse = abLoadSpatialMatrix6x6(parentLink,
                                                            PXG_AB_LINK_mSpatialResponseMatrix,
                                                            lane);
                    const AbSpatialVector bias = siLoadSV(link, PXG_AB_LINK_mBiasForce, lane);
                    // trap 8: |top| + |bottom|
                    const float mag = sqrt(dot(bias.top, bias.top))
                                    + sqrt(dot(bias.bottom, bias.bottom));
                    transmissionForce = mag * frictionCoefficient;
                    const float pCfm = abF32(parentLink, PXG_AB_LINK_mCfm, lane);
                    const float lCfm = abF32(link, PXG_AB_LINK_mCfm, lane);
                    cfm = pCfm > lCfm ? pCfm : lCfm;
                    loaded = true;
                }

                AbSpatialVector worldMotionVector = siLoadSV(d, PXG_AB_DOF_mWorldMotionMatrix, lane);

                AbSpatialVector axis0, axis1;
                const bool isLinear = dofId >= SI_AXIS_EX;
                const float position = abF32(d, PXG_AB_DOF_mJointPositions, lane);
                float error;
                if (!isLinear)
                {
                    axis0 = asmSVec(float3(0.0f, 0.0f, 0.0f), worldMotionVector.top);
                    axis1 = asmSVec(float3(0.0f, 0.0f, 0.0f), -worldMotionVector.top);
                    error = driveError[dofId];
                }
                else
                {
                    // trap 7: rotate only, no translation add
                    const float4 cRot = abF4(link, PXG_AB_LINK_mAccumulatedPose + PXG_AB_STB_Q, lane);
                    const float4 pRot = abF4(parentLink, PXG_AB_LINK_mAccumulatedPose + PXG_AB_STB_Q, lane);
                    const float4 parentP = abF4(link, PXG_AB_LINK_mParentPose + PXG_AB_STB_P, lane);
                    const float4 childP = abF4(link, PXG_AB_LINK_mChildPose + PXG_AB_STB_P, lane);
                    const float3 cA2w = quatRotate(pRot, float3(parentP.x, parentP.y, parentP.z));
                    const float3 cB2w = quatRotate(cRot, float3(childP.x, childP.y, childP.z));

                    const float3 axis = worldMotionVector.bottom;
                    const float3 ang0 = cross(cA2w, axis);
                    const float3 ang1 = cross(cB2w, axis);

                    axis0 = asmSVec(axis, ang0);
                    axis1 = asmSVec(-axis, -ang1);

                    // trap 6: dofs[0], as the engine spells it
                    const uint64_t ic0lin = dofs + PXG_AB_DOF_mConstraintData;
                    error = abF32(ic0lin, PXG_AB_IC_mDriveTargetPos, lane) - position;
                }

                AbSpatialVector deltaVA, deltaVB;
                siGetImpulseSelfResponseDofAligned(axis0, axis1, deltaVA, deltaVB,
                                                   d, dofs, child2Parent, parentResponse,
                                                   worldMotionVector, nbDofs, i, lane);

                const float r0 = asmInnerProduct(deltaVA, axis0);
                const float r1 = asmInnerProduct(deltaVB, axis1);

                const float unitResponse = r1 + r0;

                const float recipResponse = 1.0f / (unitResponse + cfm);

                abStoreF32(ic, PXG_AB_IC_mMaxFrictionForce, lane, transmissionForce);

                if (motion == SI_MOTION_LIMITED)
                {
                    const float2 limit = abF2(ic, PXG_AB_IC_mLimits_LowLimitX_highLimitY, lane);
                    abStoreF2(ic, PXG_AB_IC_mLimitError_LowX_highY, lane,
                              float2(position - limit.x, limit.y - position));
                }

                if ((maxForce > 0.0f || maxEffort > 0.0f) && (stiffness > 0.0f || damping > 0.0f))
                {
                    const float targetVelocity = abF32(ic, PXG_AB_IC_mDriveTargetVel, lane);
                    const uint driveType = abU32(ic, PXG_AB_IC_mDriveType, lane);
                    // computeImplicitDriveParams' switch: eFORCE / eACCELERATION,
                    // and a ZERO desc for anything else (the engine's default;
                    // eNONE reaching here would trip its PX_ASSERT)
                    AdDriveDesc desc;
                    desc.tvBias = 0.0f; desc.biasCoef = 0.0f; desc.velMul = 0.0f;
                    desc.impMul = 0.0f; desc.posBias = 0.0f;
                    if (driveType == SI_DRIVE_FORCE)
                        desc = adForceDrive(stiffness, damping, isTGSSolver ? stepDt : dt, dt,
                                            unitResponse, error, targetVelocity, isTGSSolver);
                    else if (driveType == SI_DRIVE_ACCELERATION)
                        desc = adAccelDrive(stiffness, damping, isTGSSolver ? stepDt : dt, dt,
                                            recipResponse, error, targetVelocity, isTGSSolver);
                    abStoreF4(ic, PXG_AB_IC_mTargetVelPlusInitialBiasX_DriveBiasCoefficientY_VelMultiplierZ_ImpulseMultiplierW,
                              lane, float4(desc.tvBias, desc.biasCoef, desc.velMul, desc.impMul));
                    abStoreF32(ic, PXG_AB_IC_mTargetPosBias, lane, desc.posBias);
                    abStoreF32(ic, PXG_AB_IC_mConstraintMaxImpulse, lane, maxForce * maxForceScale);
                }
                else
                {
                    abStoreF4(ic, PXG_AB_IC_mTargetVelPlusInitialBiasX_DriveBiasCoefficientY_VelMultiplierZ_ImpulseMultiplierW,
                              lane, float4(0.0f, 0.0f, 0.0f, 0.0f));
                    abStoreF32(ic, PXG_AB_IC_mTargetPosBias, lane, 0.0f);
                    abStoreF32(ic, PXG_AB_IC_mConstraintMaxImpulse, lane, 0.0f);
                }
                abStoreF32(ic, PXG_AB_IC_mDriveImpulse, lane, 0.0f);
                abStoreSpatialVector(ic, PXG_AB_IC_mDeltaVA, lane, deltaVA.top, deltaVA.bottom);
                abStoreSpatialVector(ic, PXG_AB_IC_mDeltaVB, lane, deltaVB.top, deltaVB.bottom);
                abStoreSpatialVector(ic, PXG_AB_IC_mRow0, lane, axis0.top, axis0.bottom);
                // trap 10: -axis1
                abStoreSpatialVector(ic, PXG_AB_IC_mRow1, lane, -axis1.top, -axis1.bottom);
                abStoreF32(ic, PXG_AB_IC_mRecipResponse, lane, recipResponse);
                abStoreF32(ic, PXG_AB_IC_mResponse, lane, unitResponse);
                abStoreF32(ic, PXG_AB_IC_mAccumulatedFrictionImpulse, lane, 0.0f);
                abStoreF32(ic, PXG_AB_IC_mLowImpulse, lane, 0.0f);
                abStoreF32(ic, PXG_AB_IC_mHighImpulse, lane, 0.0f);
            }
        }
        dofs += uint64_t(nbDofs) * PXG_AB_DOF_SIZE;
    }
}

// ---------------------------------------------------------------------------
// fixture entry point: one articulation per lane, one warp per group; block
// bases handed in directly (the wrapper's index chain is P8.97c's job)
// ---------------------------------------------------------------------------

struct SiParams
{
    uint64_t artiBlocks;   // @0
    uint64_t linkBlocks;   // @8
    uint64_t dofBlocks;    // @16
    float stepDt;          // @24
    float dt;              // @28
    float invDt;           // @32
    uint isTGS;            // @36
    uint nbArticulations;  // @40
    uint maxLinks;         // @44
    uint maxDofs;          // @48
    uint pad0;             // @52 -> 56
};

[[vk::push_constant]]
ConstantBuffer<SiParams> gSi;

[shader("compute")]
[numthreads(32, 1, 1)]
void siSetupInternalTest(uint3 groupId : SV_GroupID, uint3 groupThreadId : SV_GroupThreadID)
{
    const uint w = groupId.x;
    const uint lane = groupThreadId.x;
    const uint idx = w * 32u + lane;
    if (idx >= gSi.nbArticulations)
        return;

    const uint64_t artiBlock = gSi.artiBlocks + uint64_t(w) * PXG_AB_BLOCK_SIZE;
    const uint64_t linkBase = gSi.linkBlocks
                            + uint64_t(w) * uint64_t(gSi.maxLinks) * PXG_AB_LINK_SIZE;
    const uint64_t dofBase = gSi.dofBlocks
                           + uint64_t(w) * uint64_t(gSi.maxDofs) * PXG_AB_DOF_SIZE;

    siSetupInternalConstraints(artiBlock, linkBase, dofBase,
                               gSi.stepDt, gSi.dt, gSi.invDt, gSi.isTGS != 0u, lane);
}
