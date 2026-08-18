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

// pxgSlangPrepCommon.slang
//
// Shared pieces of the constraint-prep ports (constraintPrepPGS.slang /
// constraintPrepTGS.slang):
//   - byte-exact mirrors of the prep INPUT structures (PxgConstraintPrep.h,
//     PxgSolverConstraintDesc.h, PxgFrictionPatch.h)
//   - the Dy 1d-constraint solver-constant helpers shared between CPU and GPU
//     (lowleveldynamics/shared/DyCpuGpu1dConstraint.h)
//   - the compliant-contact coefficient helpers (constraintPrepShared.cuh)
//   - PxAlignedTransform math (gpucommon/include/AlignedTransform.h): float4
//     rotate()/transform() zero the w component exactly like the CUDA code

module pxgSlangPrepCommon;

import pxgSlangCommon;

// ---------------------------------------------------------------------------
// constants
// ---------------------------------------------------------------------------

public static const float PXC_SAME_NORMAL = 0.999f;
public static const float DY_MIN_RESPONSE = 1e-12f;
public static const uint MAX_CONSTRAINT_ROWS = 20;
public static const float PREP_PX_MAX_F32 = 3.4028234663852885981170418348452e+38f;

// Px1DConstraintFlag (PxConstraintDesc.h)
public static const uint PX1D_FLAG_SPRING = 1u << 0;
public static const uint PX1D_FLAG_ACCELERATION_SPRING = 1u << 1;
public static const uint PX1D_FLAG_RESTITUTION = 1u << 2;
public static const uint PX1D_FLAG_KEEPBIAS = 1u << 3;
public static const uint PX1D_FLAG_OUTPUT_FORCE = 1u << 4;
public static const uint PX1D_FLAG_HAS_DRIVE_LIMIT = 1u << 5;
public static const uint PX1D_FLAG_ANGULAR_CONSTRAINT = 1u << 6;

// PxConstraintSolveHint (PxConstraintDesc.h)
public static const uint PX_SOLVE_HINT_ROTATIONAL_EQUALITY = 1024;
public static const uint PX_SOLVE_HINT_ROTATIONAL_INEQUALITY = 1025;
public static const uint PX_SOLVE_HINT_EQUALITY = 2048;
public static const uint PX_SOLVE_HINT_INEQUALITY = 2049;

// PxConstraintFlag (PxConstraint.h)
public static const uint PX_CONSTRAINT_FLAG_DRIVE_LIMITS_ARE_FORCES = 1u << 5;
public static const uint PX_CONSTRAINT_FLAG_DISABLE_PREPROCESSING = 1u << 8;

// PxRigidBodyFlag (PxRigidBody.h)
public static const uint PX_RIGID_BODY_FLAG_KINEMATIC = 1u << 0;
public static const uint PX_RIGID_BODY_FLAG_ENABLE_SPECULATIVE_CCD = 1u << 4;

// PxgSolverContactFlags (PxgSolverFlags.h)
public static const uint PXG_SC_FLAG_HAS_FORCE_THRESHOLDS = 1u << 0;
public static const uint PXG_SC_FLAG_PER_POINT_FRICTION = 1u << 1;
public static const uint PXG_SC_FLAG_DISABLE_FRICTION = 1u << 2;
public static const uint PXG_SC_FLAG_COMPLIANT_ACCELERATION_SPRING = 1u << 3;

// DySolverConstraintTypes.h internal flags
public static const uint DY_SC_FLAG_OUTPUT_FORCE = 1u << 1;
public static const uint DY_SC_FLAG_KEEP_BIAS = 1u << 2;
public static const uint DY_SC_FLAG_ROT_EQ = 1u << 3;
public static const uint DY_SC_FLAG_ORTHO_TARGET = 1u << 4;
public static const uint DY_SC_FLAG_SPRING = 1u << 5;
public static const uint DY_SC_FLAG_INEQUALITY = 1u << 6;
public static const uint DY_SC_FLAG_ACCELERATION_SPRING = 1u << 7;

// ---------------------------------------------------------------------------
// input structure mirrors
// ---------------------------------------------------------------------------

// mirrors PxgBlockContactData (PxgConstraintPrep.h), 1664 bytes.
// contactData packs PxgMaterialContactData as one float4 per lane:
//   x = restDistance, y = staticFriction, z = dynamicFriction,
//   w = mNumContacts | mSolverFlags << 8 | prevFrictionPatchCount << 16 (PxU8s)
public struct PxgBlockContactData
{
    public float4 mInvMassScale[32];       // 0    (PxgConstraintInvMassScale: lin0X_ang0Y_lin1Z_ang1W)
    public float4 normal_restitutionW[32]; // 512
    public float4 contactData[32];         // 1024 (PxgMaterialContactData)
    public float damping[32];              // 1536
}                                          // = 1664

// mirrors PxgBlockContactPoint (PxgConstraintPrep.h), 1024 bytes
public struct PxgBlockContactPoint
{
    public float4 point_separationW[32];    // 0
    public float4 targetVel_maxImpulseW[32];// 512
}                                           // = 1024

// mirrors PxgBlockFrictionAnchorPatch (PxgFrictionPatch.h), 2048 bytes
public struct PxgBlockFrictionAnchorPatch
{
    public float4 body0Anchors0[32]; // 0    (body0Anchors[0])
    public float4 body0Anchors1[32]; // 512  (body0Anchors[1])
    public float4 body1Anchors0[32]; // 1024 (body1Anchors[0])
    public float4 body1Anchors1[32]; // 1536 (body1Anchors[1])
}                                    // = 2048

// mirrors PxgBlockConstraint1DData (PxgConstraintPrep.h), 1792 bytes
public struct PxgBlockConstraint1DData
{
    public float4 mInvMassScale[32];        // 0
    public float4 mRAWorld_linBreakForce[32]; // 512
    public float4 mRBWorld_AngBreakForce[32]; // 1024
    public uint mNumRows[32];               // 1536
    public uint mFlags[32];                 // 1664
}                                           // = 1792

// mirrors PxgBlockConstraint1DVelocities (PxgSolverConstraintDesc.h), 2048 bytes
public struct PxgBlockConstraint1DVelocities
{
    public float4 linear0XYZ_geometricErrorW[32];  // 0
    public float4 angular0XYZ_velocityTargetW[32]; // 512
    public float4 linear1XYZ_minImpulseW[32];      // 1024
    public float4 angular1XYZ_maxImpulseW[32];     // 1536
}                                                  // = 2048

// mirrors PxgBlockConstraint1DParameters (PxgSolverConstraintDesc.h), 512 bytes.
// The mods union: spring.{stiffness,damping} aliases bounce.{restitution,velocityThreshold}
public struct PxgBlockConstraint1DParameters
{
    public float springStiffness_restitution[32];    // 0   (mods.spring.stiffness / mods.bounce.restitution)
    public float springDamping_velocityThreshold[32];// 128 (mods.spring.damping / mods.bounce.velocityThreshold)
    public uint flags[32];                           // 256
    public uint solveHint[32];                       // 384
}                                                    // = 512

// mirrors PxgBlockWorkUnit (PxgSolverConstraintDesc.h), 1024 bytes
public struct PxgBlockWorkUnit
{
    public uint mWriteback[32];             // 0
    public float mRestDistance[32];         // 128
    public uint mEdgeIndex[32];             // 256
    public uint mFlags[32];                 // 384
    public uint mPatchIndex[32];            // 512
    public uint mFrictionPatchIndex[32];    // 640
    public float2 mTorsionalFrictionData[32]; // 768
}                                           // = 1024

// ---------------------------------------------------------------------------
// PxAlignedTransform math (float4 rotate/transform zero the w component)
// ---------------------------------------------------------------------------

public float4 alignedRotate(float4 q, float4 v)
{
    return float4(quatRotate(q, v.xyz), 0.0f);
}

public float4 alignedRotateInv(float4 q, float4 v)
{
    return float4(quatRotateInv(q, v.xyz), 0.0f);
}

// PxAlignedTransform::transform(float4)
public float4 alignedTransformPoint(PxgAlignedTransform t, float4 input)
{
    return alignedRotate(t.q, input) + t.p;
}

// PxAlignedTransform::transformInv(PxAlignedTransform):
// src = [srct, srcr] -> [r^-1*(srct-t), r^-1*srcr]
public PxgAlignedTransform alignedTransformInvTransform(PxgAlignedTransform a, PxgAlignedTransform b)
{
    const float4 qinv = quatConjugate(a.q);
    PxgAlignedTransform r;
    r.q = quatMul(qinv, b.q);
    r.p = float4(quatRotateInv(a.q, (b.p - a.p).xyz), 0.0f);
    return r;
}

// ---------------------------------------------------------------------------
// Dy / prep-shared helpers (verbatim transcriptions)
// ---------------------------------------------------------------------------

// mirrors pointsAreClose() in constraintPrepShared.cuh
public bool pointsAreClose(PxgAlignedTransform body1ToBody0, float4 localAnchor0, float4 localAnchor1,
                           float4 axis, float correlDist)
{
    const float4 body0PatchPoint1 = alignedTransformPoint(body1ToBody0, localAnchor1);
    return abs(dot((localAnchor0 - body0PatchPoint1).xyz, axis.xyz)) < correlDist;
}

// mirrors computeCompliantDamping() in constraintPrepShared.cuh
public float computeCompliantDamping(bool isSeparated, bool collidingWithVrel, float damping)
{
    return (isSeparated && !collidingWithVrel) ? 0.0f : damping;
}

// mirrors queryReducedCompliantContactCoefficients() in constraintPrepShared.cuh
public void queryReducedCompliantContactCoefficients(float dt, uint flags, float restitution, float damping,
                                                     float penetration, float targetVelocity, bool isSeparated,
                                                     bool collidingWithVrel, out float coeff0, out float coeff1)
{
    const float nrdt = dt * restitution;
    const float dampingIfEnabled = computeCompliantDamping(isSeparated, collidingWithVrel, damping);
    coeff0 = dt * (dampingIfEnabled - nrdt); // a
    coeff1 = -nrdt * penetration;            // b (to be scaled by massIfAccelElseOne)
}

// mirrors computeCompliantContactCoefficientsTGS() (full variant) in
// constraintPrepShared.cuh; returns the compliant coefficient "a"
public float computeCompliantContactCoefficientsTGSFull(float stepDt, uint flags, float restitution,
                                                        float damping, float unitResponse, float recipResponse,
                                                        bool isSeparated, bool collidingWithVrel,
                                                        out float velMultiplier, out float scaledBias)
{
    const bool accelSpring = (flags & PXG_SC_FLAG_COMPLIANT_ACCELERATION_SPRING) != 0;
    const float massIfAccelElseOne = accelSpring ? recipResponse : 1.0f;
    const float oneIfAccelElseR = accelSpring ? 1.0f : unitResponse;

    const float dampingIfEnabled = computeCompliantDamping(isSeparated, collidingWithVrel, damping);
    const float nrdt = stepDt * restitution;
    const float a = stepDt * (dampingIfEnabled - nrdt);
    const float x = 1.0f / (a * oneIfAccelElseR + 1.0f);

    velMultiplier = x * a * massIfAccelElseOne;
    scaledBias = nrdt * x * oneIfAccelElseR;

    return a;
}

// mirrors Dy::computeRecipUnitResponse()
public float computeRecipUnitResponse(float unitResponse, float minRowResponse)
{
    return unitResponse <= minRowResponse ? 0.0f : 1.0f / unitResponse;
}

// mirrors Dy::computeMinMaxImpulseOrForceAsImpulse()
public void computeMinMaxImpulseOrForceAsImpulse(float minSpecified, float maxSpecified, bool hasDriveLimit,
                                                 bool driveLimitsAreForces, float simDt,
                                                 out float minImpulse, out float maxImpulse)
{
    const float driveScale = (hasDriveLimit && driveLimitsAreForces) ? simDt : 1.0f;
    minImpulse = minSpecified * driveScale;
    maxImpulse = maxSpecified * driveScale;
}

// mirrors Dy::computeBounceVelocity()
public float computeBounceVelocity(uint constraintFlags, float jointSpeedForRestitutionBounce,
                                   float bounceThreshold, float restitution, float geometricError)
{
    const float bounceVel = jointSpeedForRestitutionBounce * (-restitution);

    if ((constraintFlags & PX1D_FLAG_RESTITUTION) != 0 &&
        (-jointSpeedForRestitutionBounce > bounceThreshold) &&
        ((bounceVel * geometricError) <= 0.0f))
    {
        return bounceVel;
    }
    return 0.0f;
}

// mirrors queryReduced1dConstraintSolverConstantsPGS() in constraintPrepShared.cuh
public void queryReduced1dConstraintSolverConstantsPGS(uint constraintFlags, float springStiffness,
                                                       float springDamping, float restitution,
                                                       float bounceThreshold, float geometricError,
                                                       float velocityTarget,
                                                       float jointSpeedForRestitutionBounce, float erp,
                                                       float simDt, float recipSimDt,
                                                       out float coeff0, out float coeff1)
{
    if ((constraintFlags & PX1D_FLAG_SPRING) != 0)
    {
        // coeff0: a, coeff1: b
        coeff0 = simDt * simDt * springStiffness + simDt * springDamping;
        coeff1 = simDt * (springDamping * velocityTarget - springStiffness * geometricError);
    }
    else
    {
        // coeff0: constant, coeff1: unbiasedConstant (both to be scaled by recipUnitResponse)
        const float bounceVel = computeBounceVelocity(constraintFlags, jointSpeedForRestitutionBounce,
                                                      bounceThreshold, restitution, geometricError);
        if (bounceVel != 0.0f)
        {
            coeff0 = bounceVel;
            coeff1 = bounceVel;
        }
        else
        {
            const float geomError = geometricError * erp;
            coeff0 = velocityTarget - geomError * recipSimDt;
            coeff1 = ((constraintFlags & PX1D_FLAG_KEEPBIAS) == 0) ? velocityTarget : coeff0;
        }
    }
}

// mirrors Dy::computeMaxBiasVelocityTGS()
public float computeMaxBiasVelocityTGS(uint constraintFlags, float jointSpeedForRestitutionBounce,
                                       float bounceThreshold, float restitution, float geometricError,
                                       bool isExtendedConstraint, float lengthScale, float recipSimDt)
{
    float maxBiasSpeed = PREP_PX_MAX_F32;
    if ((constraintFlags & PX1D_FLAG_SPRING) != 0)
    {
        maxBiasSpeed = PREP_PX_MAX_F32;
    }
    else
    {
        const float bounceVel = computeBounceVelocity(constraintFlags, jointSpeedForRestitutionBounce,
                                                      bounceThreshold, restitution, geometricError);
        if (bounceVel != 0.0f)
            maxBiasSpeed = 0.0f;
        else if ((constraintFlags & PX1D_FLAG_ANGULAR_CONSTRAINT) != 0)
            maxBiasSpeed = recipSimDt * 0.75f;
        else
            maxBiasSpeed = isExtendedConstraint ? recipSimDt * 1.5f * lengthScale
                                                : recipSimDt * 15.0f * lengthScale;
    }
    return maxBiasSpeed;
}

// mirrors Dy::queryReduced1dConstraintSolverConstantsTGS (DyCpuGpu1dConstraint.h L489-539)
public void queryReduced1dConstraintSolverConstantsTGS(uint constraintFlags, float springStiffness, float springDamping,
                        float restitution, float bounceThreshold, float geometricError,
                        float velocityTarget, float jointSpeedForRestitutionBounce,
                        float initJointSpeed, float erp, float stepDt, float recipStepDt,
                        out float coeff0, out float coeff1, out float coeff2, out float coeff3)
{
    // coeff0: initJointSpeed, coeff1: biasScale
    coeff0 = initJointSpeed;

    if ((constraintFlags & PX1D_FLAG_SPRING) != 0u)
    {
        // coeff2: a, coeff3: b
        const float a = stepDt * (stepDt * springStiffness + springDamping);
        const float b = stepDt * (springDamping * velocityTarget);
        coeff2 = a;
        coeff3 = b;

        if ((constraintFlags & PX1D_FLAG_ACCELERATION_SPRING) != 0u)
        {
            const float x = 1.0f / (1.0f + a);
            coeff1 = -x * springStiffness * stepDt;  // biasScale * unitResponse recovered later
        }
        else
        {
            coeff1 = -springStiffness * stepDt;  // to recover biasScale, multiply by x
        }
    }
    else
    {
        // coeff2: bounceVel, coeff3: velocityTarget
        const float bounceVel = computeBounceVelocity(constraintFlags, jointSpeedForRestitutionBounce,
                                                      bounceThreshold, restitution, geometricError);
        coeff2 = bounceVel;
        coeff3 = velocityTarget;

        if (bounceVel != 0.0f)
        {
            coeff1 = 0.0f;
        }
        else
        {
            coeff1 = -recipStepDt * erp;
        }
    }
}

// mirrors Dy::computeJointSpeedTGS()
public void computeJointSpeedTGS(float normalVel0, bool isKinematic0, float normalVel1, bool isKinematic1,
                                 out float jointSpeedForRestitutionBounce, out float initJointSpeed)
{
    initJointSpeed = 0.0f;
    if (isKinematic0)
        initJointSpeed -= normalVel0;
    if (isKinematic1)
        initJointSpeed += normalVel1;
    jointSpeedForRestitutionBounce = normalVel0 - normalVel1;
}

// mirrors Dy::compute1dConstraintSolverConstantsTGS() (full variant); the
// result members map to Constraint1dSolverConstantsTGS {biasScale, error,
// velMultiplier, targetVel}
public void compute1dConstraintSolverConstantsTGSFull(uint constraintFlags, float springStiffness,
                                                      float springDamping, float restitution,
                                                      float bounceThreshold, float geometricError,
                                                      float velocityTarget,
                                                      float jointSpeedForRestitutionBounce,
                                                      float initJointSpeed, float unitResponse,
                                                      float recipUnitResponse, float erp, float stepDt,
                                                      float recipStepDt, out float outBiasScale,
                                                      out float outError, out float outVelMultiplier,
                                                      out float outTargetVel)
{
    outBiasScale = 0.0f;
    outError = 0.0f;
    outVelMultiplier = 0.0f;
    outTargetVel = 0.0f;

    if ((constraintFlags & PX1D_FLAG_SPRING) != 0)
    {
        const float a = stepDt * (stepDt * springStiffness + springDamping);
        const float b = stepDt * (springDamping * velocityTarget);

        if ((constraintFlags & PX1D_FLAG_ACCELERATION_SPRING) != 0)
        {
            const float x = 1.0f / (1.0f + a);
            const float biasScale = -x * springStiffness * stepDt * recipUnitResponse;
            const float velMultiplier = -x * a * recipUnitResponse;

            outBiasScale = biasScale;
            outError = biasScale * geometricError;
            outVelMultiplier = velMultiplier;
            outTargetVel = x * b * recipUnitResponse - velMultiplier * initJointSpeed;
        }
        else
        {
            const float x = 1.0f / (1.0f + a * unitResponse);
            const float biasScale = -x * springStiffness * stepDt;
            const float velMultiplier = -x * a;

            outBiasScale = biasScale;
            outError = biasScale * geometricError;
            outVelMultiplier = velMultiplier;
            outTargetVel = x * b - velMultiplier * initJointSpeed;
        }
    }
    else
    {
        const float bounceVel = computeBounceVelocity(constraintFlags, jointSpeedForRestitutionBounce,
                                                      bounceThreshold, restitution, geometricError);
        if (bounceVel != 0.0f)
        {
            const float velMultiplier = -1.0f;
            outBiasScale = 0.0f;
            outError = 0.0f;
            outVelMultiplier = velMultiplier;
            outTargetVel = bounceVel - velMultiplier * initJointSpeed;
        }
        else
        {
            const float velMultiplier = -1.0f;
            const float biasScale = -recipStepDt * erp;

            outBiasScale = biasScale;
            outError = geometricError * biasScale;
            outVelMultiplier = velMultiplier;
            outTargetVel = velocityTarget - velMultiplier * initJointSpeed;
        }
    }
}

// mirrors Dy::raiseInternalFlagsTGS()
public void raiseInternalFlagsTGS(uint externalFlags, uint externalSolveHint, inout uint internalFlags)
{
    if ((externalFlags & PX1D_FLAG_SPRING) != 0)
        internalFlags |= DY_SC_FLAG_SPRING;
    if ((externalFlags & PX1D_FLAG_ACCELERATION_SPRING) != 0)
        internalFlags |= DY_SC_FLAG_ACCELERATION_SPRING;
    if ((externalFlags & PX1D_FLAG_OUTPUT_FORCE) != 0)
        internalFlags |= DY_SC_FLAG_OUTPUT_FORCE;
    if ((externalFlags & PX1D_FLAG_KEEPBIAS) != 0)
        internalFlags |= DY_SC_FLAG_KEEP_BIAS;
    if ((externalSolveHint & 1u) != 0)
        internalFlags |= DY_SC_FLAG_INEQUALITY;
}

// PxgSolverBodyData::projectVelocity()
public float projectVelocity(PxgSolverBodyData data, float3 lin, float3 ang)
{
    return dot(data.initialLinVelXYZ_invMassW.xyz, lin) + dot(data.initialAngVelXYZ_penBiasClamp.xyz, ang);
}

// PxAlignedTransform::transform(PxVec3) / transformInv(PxVec3)
public float3 alignedTransformVec(PxgAlignedTransform t, float3 v)
{
    return quatRotate(t.q, v) + t.p.xyz;
}

public float3 alignedTransformInvVec(PxgAlignedTransform t, float3 v)
{
    return quatRotateInv(t.q, v - t.p.xyz);
}

// ---------------------------------------------------------------------------
// friction correlation (contactConstraintBlockPrep.cuh); shared verbatim by
// the PGS and TGS contact prep.  The functions write only the [threadIndex]
// lane of the block structures, so they take the buffers + element indices
// instead of struct references (a copy-out of the whole 2816-byte block
// struct would clobber the other lanes).
// ---------------------------------------------------------------------------

// mirrors PxgBlockFrictionPatch (PxgFrictionPatch.h), 2816 bytes
public struct PxgBlockFrictionPatch
{
    public float4 body0Normal[32];   // 0
    public float4 body1Normal[32];   // 512
    public uint anchorCount[32];     // 1024
    public uint broken[32];          // 1152
    public uint contactID0[32];      // 1280 (contactID[0])
    public uint contactID1[32];      // 1408 (contactID[1])
    public float4 anchorPoints0[32]; // 1536 (anchorPoints[0])
    public float4 anchorPoints1[32]; // 2048 (anchorPoints[1])
    public uint patchIndex[32];      // 2560
    public uint pad0[32];            // 2688 (tail padding to the 256 alignment)
}                                    // = 2816

// mirrors initFrictionPatch() + correlatePatches()
public void correlatePatches(RWStructuredBuffer<PxgBlockFrictionPatch> patches, uint pi,
                             uint nbContacts, float3 normal, PxgAlignedTransform bodyFrame0,
                             PxgAlignedTransform bodyFrame1, uint threadIndex)
{
    if (nbContacts > 0)
    {
        const float3 body0Normal = quatRotateInv(bodyFrame0.q, normal);
        const float3 body1Normal = quatRotateInv(bodyFrame1.q, normal);

        patches[pi].contactID0[threadIndex] = 0xff;
        patches[pi].contactID1[threadIndex] = 0xff;

        // initFrictionPatch()
        patches[pi].body0Normal[threadIndex] = float4(body0Normal, 0.0f);
        patches[pi].body1Normal[threadIndex] = float4(body1Normal, 0.0f);
        patches[pi].anchorCount[threadIndex] = 0;
        patches[pi].broken[threadIndex] = 0;
    }
}

// mirrors getFrictionPatches(): tries to carry the friction anchors of a
// matching previous-frame patch over into the current patch
public void getFrictionPatches(RWStructuredBuffer<PxgBlockFrictionPatch> currPatches, uint pi,
                               RWStructuredBuffer<PxgBlockFrictionAnchorPatch> currAnchors, uint ai,
                               RWStructuredBuffer<uint2> prevFrictionIndices, bool hasPrevFrictionIndices,
                               uint prevFrictionStartIndex,
                               RWStructuredBuffer<PxgBlockFrictionPatch> prevPatches,
                               RWStructuredBuffer<PxgBlockFrictionAnchorPatch> prevAnchors,
                               uint frictionPatchCount, PxgAlignedTransform bodyFrame0,
                               PxgAlignedTransform bodyFrame1, float correlationDistance,
                               uint threadIndex, uint totalNbEdges, inout float patchExtents,
                               uint nbContacts)
{
    if (prevFrictionStartIndex == 0xFFFFFFFFu || frictionPatchCount == 0 || !hasPrevFrictionIndices)
        return;

    for (uint a = 0; a < frictionPatchCount; ++a)
    {
        const uint2 index = prevFrictionIndices[prevFrictionStartIndex + a * totalNbEdges];
        // PxgBlockFrictionIndex: PxU64 = patchIndex << 5 | threadIdx
        const uint oldThreadId = index.x & 31u;
        const uint patchIndex = (index.x >> 5) | (index.y << 27);

        if (prevPatches[patchIndex].broken[oldThreadId] == 0)
        {
            const float4 oldBody0Normal = prevPatches[patchIndex].body0Normal[oldThreadId];
            if (dot(oldBody0Normal.xyz, currPatches[pi].body0Normal[threadIndex].xyz) > PXC_SAME_NORMAL)
            {
                const uint anchorCount = prevPatches[patchIndex].anchorCount[oldThreadId];
                if (anchorCount != 0)
                {
                    // KS - if we have more anchors than contacts, we need to throw away the patch and rebuild
                    if (anchorCount <= nbContacts)
                    {
                        const PxgAlignedTransform body1ToBody0 = alignedTransformInvTransform(bodyFrame0, bodyFrame1);
                        const float4 oldBody1Normal = prevPatches[patchIndex].body1Normal[oldThreadId];
                        if (dot(oldBody0Normal.xyz, alignedRotate(body1ToBody0.q, oldBody1Normal).xyz) > PXC_SAME_NORMAL)
                        {
                            const float4 body0Anchor0 = prevAnchors[patchIndex].body0Anchors0[oldThreadId];
                            const float4 body1Anchor0 = prevAnchors[patchIndex].body1Anchors0[oldThreadId];
                            if (anchorCount == 0 ||
                                pointsAreClose(body1ToBody0, body0Anchor0, body1Anchor0, oldBody0Normal, correlationDistance))
                            {
                                const float4 body0Anchor1 = prevAnchors[patchIndex].body0Anchors1[oldThreadId];
                                const float4 body1Anchor1 = prevAnchors[patchIndex].body1Anchors1[oldThreadId];
                                if (anchorCount < 2 ||
                                    pointsAreClose(body1ToBody0, body0Anchor1, body1Anchor1, oldBody0Normal, correlationDistance))
                                {
                                    currPatches[pi].contactID0[threadIndex] = 0xff;
                                    currPatches[pi].contactID1[threadIndex] = 0xff;
                                    currPatches[pi].anchorCount[threadIndex] = anchorCount;
                                    currPatches[pi].body0Normal[threadIndex] = oldBody0Normal;
                                    currPatches[pi].body1Normal[threadIndex] = oldBody1Normal;
                                    currAnchors[ai].body0Anchors0[threadIndex] = body0Anchor0;
                                    currAnchors[ai].body0Anchors1[threadIndex] = body0Anchor1;
                                    currAnchors[ai].body1Anchors0[threadIndex] = body1Anchor0;
                                    currAnchors[ai].body1Anchors1[threadIndex] = body1Anchor1;
                                    const float4 ext = body0Anchor0 - body0Anchor1;
                                    patchExtents = ext.x * ext.x + ext.y * ext.y + ext.z * ext.z;
                                    return; // Found a match = terminate!
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// mirrors growPatches(): adds up to two friction anchors from the current
// contacts, keeping previous anchors where possible
public void growPatches(RWStructuredBuffer<PxgBlockFrictionPatch> patches, uint pi,
                        RWStructuredBuffer<PxgBlockFrictionAnchorPatch> anchors, uint ai,
                        RWStructuredBuffer<PxgBlockContactPoint> contacts, uint contactBase,
                        uint numContacts, PxgAlignedTransform bodyFrame0, PxgAlignedTransform bodyFrame1,
                        float frictionOffsetThreshold, uint threadIndex, float anchorSqDistance)
{
    float3 boundsMin = float3(PREP_PX_MAX_F32);
    float3 boundsMax = float3(-PREP_PX_MAX_F32);
    for (uint i = 0; i < numContacts; ++i)
    {
        const float4 cp = contacts[contactBase + i].point_separationW[threadIndex];
        boundsMin = min(boundsMin, cp.xyz);
        boundsMax = max(boundsMax, cp.xyz);
    }

    uint oldAnchorCount = patches[pi].anchorCount[threadIndex];

    if (oldAnchorCount == 2)
    {
        const float3 dims = boundsMax - boundsMin;
        const float frictionPatchDiagonalSq = dot(dims, dims);

        // If the squared distance between the anchors is more than a quarter
        // of the patch diagonal, we can keep, otherwise the anchors are
        // potentially clustered around a corner so force a rebuild of the patch
        if ((anchorSqDistance * 4.0f) >= frictionPatchDiagonalSq)
            return;

        oldAnchorCount = 0;
    }

    float3 worldAnchor0 = float3(0.0f);
    float3 worldAnchor1 = float3(0.0f);
    uint contactID0 = 0;
    uint contactID1 = 0;

    uint anchorCount = 0;
    float pointDistSq = 0.0f;

    // if we have an anchor already, keep it
    if (oldAnchorCount == 1)
    {
        const float4 a0 = anchors[ai].body0Anchors0[threadIndex];
        worldAnchor0 = alignedTransformVec(bodyFrame0, a0.xyz);
        contactID0 = 0xFF;
        anchorCount++;
    }

    const float eps = 1e-8f;

    for (uint j = 0; j < numContacts; j++)
    {
        const float4 worldPoint4_separationW = contacts[contactBase + j].point_separationW[threadIndex];

        if (worldPoint4_separationW.w < frictionOffsetThreshold)
        {
            const float3 worldPoint = worldPoint4_separationW.xyz;
            if (anchorCount == 0)
            {
                contactID0 = j & 0xFFFFu; // PxU16(j)
                worldAnchor0 = worldPoint;
                anchorCount++;
            }
            else if (anchorCount == 1)
            {
                const float3 d = worldPoint - worldAnchor0;
                pointDistSq = dot(d, d);
                if (pointDistSq > eps)
                {
                    contactID1 = j;
                    worldAnchor1 = worldPoint;
                    anchorCount++;
                }
            }
            else // case 2
            {
                const float3 d0v = worldPoint - worldAnchor0;
                const float3 d1v = worldPoint - worldAnchor1;
                const float dist0 = dot(d0v, d0v);
                const float dist1 = dot(d1v, d1v);
                if (dist0 > dist1)
                {
                    if (dist0 > pointDistSq)
                    {
                        contactID1 = j & 0xFFFFu;
                        worldAnchor1 = worldPoint;
                        pointDistSq = dist0;
                    }
                }
                else if (dist1 > pointDistSq)
                {
                    contactID0 = j & 0xFFFFu;
                    worldAnchor0 = worldPoint;
                    pointDistSq = dist1;
                }
            }
        }
    }

    // add the new anchor(s) to the patch (the CUDA switch falls through from
    // case 2 into case 1)
    if (anchorCount == 2)
    {
        const float3 localPoint0 = alignedTransformInvVec(bodyFrame0, worldAnchor1);
        const float3 localPoint1 = alignedTransformInvVec(bodyFrame1, worldAnchor1);
        anchors[ai].body0Anchors1[threadIndex] = float4(localPoint0, 0.0f);
        anchors[ai].body1Anchors1[threadIndex] = float4(localPoint1, 0.0f);
        patches[pi].contactID1[threadIndex] = contactID1;
    }
    if (anchorCount >= 1 && oldAnchorCount == 0)
    {
        const float3 localPoint0 = alignedTransformInvVec(bodyFrame0, worldAnchor0);
        const float3 localPoint1 = alignedTransformInvVec(bodyFrame1, worldAnchor0);
        anchors[ai].body0Anchors0[threadIndex] = float4(localPoint0, 0.0f);
        anchors[ai].body1Anchors0[threadIndex] = float4(localPoint1, 0.0f);
        patches[pi].contactID0[threadIndex] = contactID0;
    }

    patches[pi].anchorCount[threadIndex] = anchorCount;
}

// ===========================================================================
// P8.102b hoist: the Ptr-typed (BDA) twins of the correlation helpers,
// moved VERBATIM from constraintPrepTGS.slang (second user: the arti
// contact prep npArtiConstraintPrep).  The RWStructuredBuffer originals
// above remain for the buffer-based test entries.
// ===========================================================================


// mirrors initFrictionPatch() + correlatePatches()
public void correlatePatchesPtr(Ptr<PxgBlockFrictionPatch> patches, uint pi, uint nbContacts, float3 normal,
                         PxgAlignedTransform bodyFrame0, PxgAlignedTransform bodyFrame1, uint threadIndex)
{
    if (nbContacts > 0)
    {
        const float3 body0Normal = quatRotateInv(bodyFrame0.q, normal);
        const float3 body1Normal = quatRotateInv(bodyFrame1.q, normal);

        patches[pi].contactID0[threadIndex] = 0xff;
        patches[pi].contactID1[threadIndex] = 0xff;

        // initFrictionPatch()
        patches[pi].body0Normal[threadIndex] = float4(body0Normal, 0.0f);
        patches[pi].body1Normal[threadIndex] = float4(body1Normal, 0.0f);
        patches[pi].anchorCount[threadIndex] = 0;
        patches[pi].broken[threadIndex] = 0;
    }
}

// mirrors getFrictionPatches(); hasPrevFrictionIndices is the engine's
// blockPreviousFrictionIndices != NULL
public void getFrictionPatchesPtr(Ptr<PxgBlockFrictionPatch> currPatches, uint pi,
                           Ptr<PxgBlockFrictionAnchorPatch> currAnchors, uint ai,
                           Ptr<uint2> prevFrictionIndices, bool hasPrevFrictionIndices,
                           uint prevFrictionStartIndex, Ptr<PxgBlockFrictionPatch> prevPatches,
                           Ptr<PxgBlockFrictionAnchorPatch> prevAnchors, uint frictionPatchCount,
                           PxgAlignedTransform bodyFrame0, PxgAlignedTransform bodyFrame1,
                           float correlationDistance, uint threadIndex, uint totalNbEdges,
                           inout float patchExtents, uint nbContacts)
{
    if (prevFrictionStartIndex == 0xFFFFFFFFu || frictionPatchCount == 0 || !hasPrevFrictionIndices)
        return;

    for (uint a = 0; a < frictionPatchCount; ++a)
    {
        const uint2 index = prevFrictionIndices[prevFrictionStartIndex + a * totalNbEdges];
        // PxgBlockFrictionIndex: PxU64 = patchIndex << 5 | threadIdx
        const uint oldThreadId = index.x & 31u;
        const uint patchIndex = (index.x >> 5) | (index.y << 27);

        if (prevPatches[patchIndex].broken[oldThreadId] == 0)
        {
            const float4 oldBody0Normal = prevPatches[patchIndex].body0Normal[oldThreadId];
            if (dot(oldBody0Normal.xyz, currPatches[pi].body0Normal[threadIndex].xyz) > PXC_SAME_NORMAL)
            {
                const uint anchorCount = prevPatches[patchIndex].anchorCount[oldThreadId];
                if (anchorCount != 0)
                {
                    // KS - if we have more anchors than contacts, we need to throw away the patch and rebuild
                    if (anchorCount <= nbContacts)
                    {
                        const PxgAlignedTransform body1ToBody0 = alignedTransformInvTransform(bodyFrame0, bodyFrame1);
                        const float4 oldBody1Normal = prevPatches[patchIndex].body1Normal[oldThreadId];
                        if (dot(oldBody0Normal.xyz, alignedRotate(body1ToBody0.q, oldBody1Normal).xyz) > PXC_SAME_NORMAL)
                        {
                            const float4 body0Anchor0 = prevAnchors[patchIndex].body0Anchors0[oldThreadId];
                            const float4 body1Anchor0 = prevAnchors[patchIndex].body1Anchors0[oldThreadId];
                            if (anchorCount == 0 ||
                                pointsAreClose(body1ToBody0, body0Anchor0, body1Anchor0, oldBody0Normal, correlationDistance))
                            {
                                const float4 body0Anchor1 = prevAnchors[patchIndex].body0Anchors1[oldThreadId];
                                const float4 body1Anchor1 = prevAnchors[patchIndex].body1Anchors1[oldThreadId];
                                if (anchorCount < 2 ||
                                    pointsAreClose(body1ToBody0, body0Anchor1, body1Anchor1, oldBody0Normal, correlationDistance))
                                {
                                    currPatches[pi].contactID0[threadIndex] = 0xff;
                                    currPatches[pi].contactID1[threadIndex] = 0xff;
                                    currPatches[pi].anchorCount[threadIndex] = anchorCount;
                                    currPatches[pi].body0Normal[threadIndex] = oldBody0Normal;
                                    currPatches[pi].body1Normal[threadIndex] = oldBody1Normal;
                                    currAnchors[ai].body0Anchors0[threadIndex] = body0Anchor0;
                                    currAnchors[ai].body0Anchors1[threadIndex] = body0Anchor1;
                                    currAnchors[ai].body1Anchors0[threadIndex] = body1Anchor0;
                                    currAnchors[ai].body1Anchors1[threadIndex] = body1Anchor1;
                                    const float4 ext = body0Anchor0 - body0Anchor1;
                                    patchExtents = ext.x * ext.x + ext.y * ext.y + ext.z * ext.z;
                                    return; // Found a match = terminate!
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// mirrors growPatches()
public void growPatchesPtr(Ptr<PxgBlockFrictionPatch> patches, uint pi,
                    Ptr<PxgBlockFrictionAnchorPatch> anchors, uint ai,
                    Ptr<PxgBlockContactPoint> contacts, uint contactBase, uint numContacts,
                    PxgAlignedTransform bodyFrame0, PxgAlignedTransform bodyFrame1,
                    float frictionOffsetThreshold, uint threadIndex, float anchorSqDistance)
{
    float3 boundsMin = float3(PREP_PX_MAX_F32);
    float3 boundsMax = float3(-PREP_PX_MAX_F32);
    for (uint i = 0; i < numContacts; ++i)
    {
        const float4 cp = contacts[contactBase + i].point_separationW[threadIndex];
        boundsMin = min(boundsMin, cp.xyz);
        boundsMax = max(boundsMax, cp.xyz);
    }

    uint oldAnchorCount = patches[pi].anchorCount[threadIndex];

    if (oldAnchorCount == 2)
    {
        const float3 dims = boundsMax - boundsMin;
        const float frictionPatchDiagonalSq = dot(dims, dims);

        // If the squared distance between the anchors is more than a quarter
        // of the patch diagonal, we can keep, otherwise the anchors are
        // potentially clustered around a corner so force a rebuild of the patch
        if ((anchorSqDistance * 4.0f) >= frictionPatchDiagonalSq)
            return;

        oldAnchorCount = 0;
    }

    float3 worldAnchor0 = float3(0.0f);
    float3 worldAnchor1 = float3(0.0f);
    uint contactID0 = 0;
    uint contactID1 = 0;

    uint anchorCount = 0;
    float pointDistSq = 0.0f;

    // if we have an anchor already, keep it
    if (oldAnchorCount == 1)
    {
        const float4 a0 = anchors[ai].body0Anchors0[threadIndex];
        worldAnchor0 = alignedTransformVec(bodyFrame0, a0.xyz);
        contactID0 = 0xFF;
        anchorCount++;
    }

    const float eps = 1e-8f;

    for (uint j = 0; j < numContacts; j++)
    {
        const float4 worldPoint4_separationW = contacts[contactBase + j].point_separationW[threadIndex];

        if (worldPoint4_separationW.w < frictionOffsetThreshold)
        {
            const float3 worldPoint = worldPoint4_separationW.xyz;
            if (anchorCount == 0)
            {
                contactID0 = j & 0xFFFFu; // PxU16(j)
                worldAnchor0 = worldPoint;
                anchorCount++;
            }
            else if (anchorCount == 1)
            {
                const float3 d = worldPoint - worldAnchor0;
                pointDistSq = dot(d, d);
                if (pointDistSq > eps)
                {
                    contactID1 = j;
                    worldAnchor1 = worldPoint;
                    anchorCount++;
                }
            }
            else // case 2
            {
                const float3 d0v = worldPoint - worldAnchor0;
                const float3 d1v = worldPoint - worldAnchor1;
                const float dist0 = dot(d0v, d0v);
                const float dist1 = dot(d1v, d1v);
                if (dist0 > dist1)
                {
                    if (dist0 > pointDistSq)
                    {
                        contactID1 = j & 0xFFFFu;
                        worldAnchor1 = worldPoint;
                        pointDistSq = dist0;
                    }
                }
                else if (dist1 > pointDistSq)
                {
                    contactID0 = j & 0xFFFFu;
                    worldAnchor0 = worldPoint;
                    pointDistSq = dist1;
                }
            }
        }
    }

    // add the new anchor(s) to the patch (the CUDA switch falls through from
    // case 2 into case 1)
    if (anchorCount == 2)
    {
        const float3 localPoint0 = alignedTransformInvVec(bodyFrame0, worldAnchor1);
        const float3 localPoint1 = alignedTransformInvVec(bodyFrame1, worldAnchor1);
        anchors[ai].body0Anchors1[threadIndex] = float4(localPoint0, 0.0f);
        anchors[ai].body1Anchors1[threadIndex] = float4(localPoint1, 0.0f);
        patches[pi].contactID1[threadIndex] = contactID1;
    }
    if (anchorCount >= 1 && oldAnchorCount == 0)
    {
        const float3 localPoint0 = alignedTransformInvVec(bodyFrame0, worldAnchor0);
        const float3 localPoint1 = alignedTransformInvVec(bodyFrame1, worldAnchor0);
        anchors[ai].body0Anchors0[threadIndex] = float4(localPoint0, 0.0f);
        anchors[ai].body1Anchors0[threadIndex] = float4(localPoint1, 0.0f);
        patches[pi].contactID0[threadIndex] = contactID0;
    }

    patches[pi].anchorCount[threadIndex] = anchorCount;
}
