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

// constraintPrepTGS.slang
//
// Slang port of the TGS constraint prep kernels
// (gpusolver/src/CUDA/constraintBlockPrepTGS.cu + contactConstraintBlockPrep.cuh
//  + jointConstraintBlockPrepTGS.cuh):
//   - contactConstraintBlockPrepareParallelLaunchTGS
//     (createFinalizeSolverContactsBlockGPUTGS: friction correlation +
//      setupFinalizeSolverConstraintsBlockTGS incl. torsional friction anchor)
//   - jointConstraintBlockPrepareParallelLaunchTGS
//     (setupSolverConstraintBlockGPUTGS: solveHint promotion, preprocessRows/
//      orthogonalize + setUp1DConstraintBlockTGS incl. ortho-axis header setup)
//
// Same portability policy as constraintPrepPGS.slang: one 32-thread workgroup
// per batch, batch range/scalars in push constants, direct (non-warp-staged)
// loads, per-thread local scratch for the row preprocessing.

import pxgSlangCommon;
import pxgSlangPrepCommon;

// ---------------------------------------------------------------------------
// output block structures (byte-identical mirrors, same as solverTGS.slang)
// ---------------------------------------------------------------------------

// mirrors PxgBlockConstraintBatch (1536 bytes)
struct PxgBlockConstraintBatch
{
    uint descStride_constraintType;
    uint blockContactIndex;
    uint mConstraintBatchIndex;
    uint startConstraintIndex;
    uint startFrictionIndex;
    uint mStartPartitionIndex;
    uint mArticulationResponseIndex;
    uint mask;
    uint pad0[24];
    uint2 bodyANodeIndex[32];
    uint2 bodyBNodeIndex[32];
    uint bodyAIndex[32];
    uint bodyBIndex[32];
    uint remappedBodyAIndex[32];
    uint remappedBodyBIndex[32];
    uint slabId[32];
    uint2 shapeInteraction[32];
}

// mirrors PxgTGSBlockSolverContactHeader (2048 bytes)
struct PxgTGSBlockSolverContactHeader
{
    float4 invMass0_1_angDom0_1[32];  // 0
    float4 normal_staticFriction[32]; // 512
    float minNormalForce[32];         // 1024
    float maxPenBias[32];             // 1152
    uint flags[32];                   // 1280
    uint numNormalConstr[32];         // 1408
    uint forceWritebackOffset[32];    // 1536
    float restitutionXdt[32];         // 1664
    float cfm[32];                    // 1792
    float p8[32];                     // 1920
}                                     // = 2048

// mirrors PxgTGSBlockSolverFrictionHeader (1664 bytes)
struct PxgTGSBlockSolverFrictionHeader
{
    float4 frictionNormals0[32];      // 0
    float4 frictionNormals1[32];      // 512
    uint numFrictionConstr[32];       // 1024
    float dynamicFriction[32];        // 1152
    uint broken[32];                  // 1280
    float biasCoefficient[32];        // 1408
    float torsionalFrictionScale[32]; // 1536
}                                     // = 1664

// mirrors PxgTGSBlockSolverContactPoint (1792 bytes)
struct PxgTGSBlockSolverContactPoint
{
    float4 raXn_extraCoeff[32]; // 0
    float4 rbXn_targetVelW[32]; // 512
    float separation[32];       // 1024
    float maxImpulse[32];       // 1152
    float appliedForce[32];     // 1280
    float biasCoefficient[32];  // 1408
    float resp0[32];            // 1536
    float resp1[32];            // 1664
}                               // = 1792

// mirrors PxgTGSBlockSolverContactFriction (1408 bytes)
struct PxgTGSBlockSolverContactFriction
{
    float4 raXn_error[32];      // 0
    float4 rbXn_targetVelW[32]; // 512
    float appliedForce[32];     // 1024
    float resp0[32];            // 1152
    float resp1[32];            // 1280
}                               // = 1408

// mirrors PxgTGSBlockSolverConstraint1DHeader (4992 bytes)
struct PxgTGSBlockSolverConstraint1DHeader
{
    uint rowCounts_breakable_orthoAxisCount[32]; // 0 (uchar4 per lane)
    float4 rAWorld_invMass0D0[32];               // 128
    float4 rBWorld_invMass1D1[32];               // 640
    float invInertiaScale0[32];                  // 1152
    float invInertiaScale1[32];                  // 1280
    uint writeBackOffset[32];                    // 1408
    float4 angOrthoAxis0_recipResponseW0[32];    // 1536
    float4 angOrthoAxis0_recipResponseW1[32];    // 2048
    float4 angOrthoAxis0_recipResponseW2[32];    // 2560
    float4 angOrthoAxis1_ErrorW0[32];            // 3072
    float4 angOrthoAxis1_ErrorW1[32];            // 3584
    float4 angOrthoAxis1_ErrorW2[32];            // 4096
    float linBreakImpulse[32];                   // 4608
    float angBreakImpulse[32];                   // 4736
    float cfm[32];                               // 4864
}                                                // = 4992

// mirrors PxgTGSBlockSolverConstraint1DCon (3328 bytes)
struct PxgTGSBlockSolverConstraint1DCon
{
    float4 lin0XYZ_initBiasOrCoeff0[32];      // 0
    float4 lin1XYZ_biasScaleOrCoeff1[32];     // 512
    float4 ang0XYZ_velMultiplierOrCoeff2[32]; // 1024
    float4 ang1XYZ_velTargetOrCoeff3[32];     // 1536
    float resp0[32];                          // 2048
    float resp1[32];                          // 2176
    float geometricError[32];                 // 2304
    float minImpulse[32];                     // 2432
    float maxImpulse[32];                     // 2560
    float maxBias[32];                        // 2688
    float angularErrorScale[32];              // 2816
    uint flags[32];                           // 2944
    float appliedForce[32];                   // 3072
    float residual[32];                       // 3200
}                                             // = 3328

// ---------------------------------------------------------------------------
// push constants + bindings
// ---------------------------------------------------------------------------

struct PrepParams
{
    uint numBatches;
    uint totalPreviousEdges;
    uint totalCurrentEdges;
    uint hasPrevFrictionIndices;
    float stepDt;      // sharedDesc->stepDt
    float invStepDt;   // sharedDesc->stepInvDtF32
    float totalDt;     // sharedDesc->dt
    float invTotalDt;  // sharedDesc->invDtF32
    float bounceThreshold;
    float frictionOffsetThreshold;
    float correlationDistance;
    float ccdMaxSeparation;
    float lengthScale;
    float biasCoefficient;
};

[[vk::push_constant]]
ConstantBuffer<PrepParams> gParams;

RWStructuredBuffer<PxgBlockConstraintBatch> gBatches;                    // binding 0
RWStructuredBuffer<uint> gBatchIndices;                                  // binding 1
RWStructuredBuffer<PxgSolverBodyData> gSolverBodyData;                   // binding 2
RWStructuredBuffer<PxgSolverTxIData> gTxIData;                           // binding 3

// joint prep inputs/outputs
RWStructuredBuffer<PxgBlockConstraint1DData> gJointData;                 // binding 4
RWStructuredBuffer<PxgBlockConstraint1DVelocities> gJointVel;            // binding 5
RWStructuredBuffer<PxgBlockConstraint1DParameters> gJointPar;            // binding 6 (RW: solveHint promotion)
RWStructuredBuffer<uint> gConstraintUniqueIndices;                       // binding 7
RWStructuredBuffer<uint2> gSolverConstantData;                           // binding 8
RWStructuredBuffer<PxgTGSBlockSolverConstraint1DHeader> gJointHeaders;   // binding 9
RWStructuredBuffer<PxgTGSBlockSolverConstraint1DCon> gJointRowsCon;      // binding 10

// contact prep inputs/outputs
RWStructuredBuffer<PxgAlignedTransform> gBodyFrames;                     // binding 11
RWStructuredBuffer<PxgBlockWorkUnit> gWorkUnits;                         // binding 12
RWStructuredBuffer<PxgBlockContactData> gContactPrepData;               // binding 13
RWStructuredBuffer<PxgBlockContactPoint> gContactPrepPoints;            // binding 14
RWStructuredBuffer<PxgBlockFrictionPatch> gCurrFrictionPatches;         // binding 15
RWStructuredBuffer<PxgBlockFrictionPatch> gPrevFrictionPatches;         // binding 16
RWStructuredBuffer<PxgBlockFrictionAnchorPatch> gCurrAnchors;           // binding 17
RWStructuredBuffer<PxgBlockFrictionAnchorPatch> gPrevAnchors;           // binding 18
RWStructuredBuffer<uint2> gCurrFrictionIndices;                        // binding 19
RWStructuredBuffer<uint2> gPrevFrictionIndices;                        // binding 20
RWStructuredBuffer<PxgTGSBlockSolverContactHeader> gContactHeaders;     // binding 21
RWStructuredBuffer<PxgTGSBlockSolverFrictionHeader> gFrictionHeaders;   // binding 22
RWStructuredBuffer<PxgTGSBlockSolverContactPoint> gContactPoints;       // binding 23
RWStructuredBuffer<PxgTGSBlockSolverContactFriction> gFrictions;        // binding 24

// ---------------------------------------------------------------------------
// contact prep (setupFinalizeSolverConstraintsBlockTGS)
// ---------------------------------------------------------------------------

void setupFinalizeSolverConstraintsBlockTGS(uint prepIndex, uint contactBase, uint contactCount,
                                            PxgAlignedTransform bodyFrame0, PxgAlignedTransform bodyFrame1,
                                            PxgSolverBodyData data0, PxgSolverBodyData data1,
                                            float4 initialLinVel0_invMassW, float4 initialAngVel0_penBiasClamp,
                                            float4 initialLinVel1_invMassW, float4 initialAngVel1_penBiasClamp,
                                            PxgMat33 sqrtInvInertia0, PxgMat33 sqrtInvInertia1,
                                            float invStepDtF32, float stepDt, float totalDt,
                                            float invTotalDtF32, float bounceThresholdF32,
                                            float biasCoefficient, uint threadIndex,
                                            uint forceWritebackBufferOffset, bool perPointFriction,
                                            uint headerIndex, uint outContactBase, uint outFrictionBase,
                                            float restDistance, float staticFrictionCof,
                                            float dynamicFrictionCof, uint flags, float solverOffsetSlop,
                                            float2 torsionalFrictionData)
{
    const float3 bodyFrame0p = bodyFrame0.p.xyz;
    const float3 bodyFrame1p = bodyFrame1.p.xyz;

    const float maxPenBias = max(initialAngVel0_penBiasClamp.w, initialAngVel1_penBiasClamp.w);

    const float3 linVel0 = initialLinVel0_invMassW.xyz;
    const float3 linVel1 = initialLinVel1_invMassW.xyz;
    const float3 angVel0 = initialAngVel0_penBiasClamp.xyz;
    const float3 angVel1 = initialAngVel1_penBiasClamp.xyz;

    const float invStepDt = invStepDtF32;
    const float p8 = min(0.8f, biasCoefficient);
    const float bounceThreshold = bounceThresholdF32;

    const float invDtp8 = invStepDt * p8;

    const float4 lin0X_ang0Y_lin1Z_ang1W = gContactPrepData[prepIndex].mInvMassScale[threadIndex];

    const float invMass0D0 = initialLinVel0_invMassW.w * lin0X_ang0Y_lin1Z_ang1W.x;
    const float invMass1D1 = initialLinVel1_invMassW.w * lin0X_ang0Y_lin1Z_ang1W.z;

    const float angD0 = lin0X_ang0Y_lin1Z_ang1W.y;
    const float angD1 = lin0X_ang0Y_lin1Z_ang1W.w;

    const bool isKinematic0 = (data0.flags & PX_RIGID_BODY_FLAG_KINEMATIC) != 0;
    const bool isKinematic1 = (data1.flags & PX_RIGID_BODY_FLAG_KINEMATIC) != 0;

    const uint aCount = gCurrFrictionPatches[prepIndex].anchorCount[threadIndex];
    const uint anchorCount = aCount * 2;

    if (contactCount == 0)
    {
        gFrictionHeaders[headerIndex].numFrictionConstr[threadIndex] = 0;
        gContactHeaders[headerIndex].numNormalConstr[threadIndex] = 0;
        gContactHeaders[headerIndex].forceWritebackOffset[threadIndex] = 0xffffffffu;
        return;
    }

    const float maxImpulse = min(data0.maxImpulse, data1.maxImpulse);

    gContactHeaders[headerIndex].forceWritebackOffset[threadIndex] = forceWritebackBufferOffset;
    gContactHeaders[headerIndex].flags[threadIndex] = flags;
    gContactHeaders[headerIndex].maxPenBias[threadIndex] = maxPenBias;
    gContactHeaders[headerIndex].invMass0_1_angDom0_1[threadIndex] =
        float4(invMass0D0, invMass1D1, lin0X_ang0Y_lin1Z_ang1W.y, lin0X_ang0Y_lin1Z_ang1W.w);

    const float4 normal4_rW = gContactPrepData[prepIndex].normal_restitutionW[threadIndex];
    const float3 normal = normal4_rW.xyz;
    const float damping = gContactPrepData[prepIndex].damping[threadIndex];
    const float restitution = normal4_rW.w;
    const float normalLenSq = dot(normal, normal);
    const float norVel0 = dot(normal, linVel0);
    const float norVel1 = dot(normal, linVel1);
    const float linNorVel = norVel0 - norVel1;

    const float invMassNorLenSq0 = invMass0D0 * normalLenSq;
    const float invMassNorLenSq1 = invMass1D1 * normalLenSq;

    float maxPenetration = PREP_PX_MAX_F32;

    gContactHeaders[headerIndex].restitutionXdt[threadIndex] = restitution * stepDt;
    gContactHeaders[headerIndex].p8[threadIndex] = p8;

    for (uint j = 0; j < contactCount; j++)
    {
        const float4 point4_separationW = gContactPrepPoints[contactBase + j].point_separationW[threadIndex];
        const float3 point = point4_separationW.xyz;
        const float separation = point4_separationW.w;

        const float4 targetVel4_maxImpulseW = gContactPrepPoints[contactBase + j].targetVel_maxImpulseW[threadIndex];
        const float3 targetVel = targetVel4_maxImpulseW.xyz;

        const float cTargetVel = dot(normal, targetVel);

        const float3 ra = point - bodyFrame0p;
        const float3 rb = point - bodyFrame1p;

        float3 raXn = cross(ra, normal);
        float3 rbXn = cross(rb, normal);

        const float angNorVel = dot(raXn, angVel0) - dot(rbXn, angVel1);

        const float slop = solverOffsetSlop * max(linNorVel == 0.0f ? 1.0f : angNorVel / linNorVel, 1.0f);

        raXn.x = abs(raXn.x) < slop ? 0.0f : raXn.x;
        raXn.y = abs(raXn.y) < slop ? 0.0f : raXn.y;
        raXn.z = abs(raXn.z) < slop ? 0.0f : raXn.z;

        rbXn.x = abs(rbXn.x) < slop ? 0.0f : rbXn.x;
        rbXn.y = abs(rbXn.y) < slop ? 0.0f : rbXn.y;
        rbXn.z = abs(rbXn.z) < slop ? 0.0f : rbXn.z;

        const float3 raXnSqrtInertia = mat33Transform(sqrtInvInertia0, raXn);
        const float3 rbXnSqrtInertia = mat33Transform(sqrtInvInertia1, rbXn);

        const float resp0 = dot(raXnSqrtInertia, raXnSqrtInertia) * angD0 + invMassNorLenSq0;
        const float resp1 = dot(rbXnSqrtInertia, rbXnSqrtInertia) * angD1 + invMassNorLenSq1;

        const float unitResponse = resp0 + resp1;

        const float angNorVel0 = dot(raXn, angVel0);
        const float angNorVel1 = dot(rbXn, angVel1);

        const float vrel = linNorVel + (angNorVel0 - angNorVel1);

        float scaledBias = 0.0f;
        float velMultiplier = 0.0f;

        float penetration = separation - restDistance;
        const bool isSeparated = (penetration > 0.0f);
        const float penetrationInvDt = penetration * invTotalDtF32;

        const float recipResponse = (unitResponse > 0.0f) ? (1.0f / unitResponse) : 0.0f;

        float contactCoeff = 0.0f;

        if (restitution < 0.0f)
        {
            const bool collidingWithVrel = ((-vrel * totalDt) > penetration);
            contactCoeff = computeCompliantContactCoefficientsTGSFull(stepDt, flags, restitution, damping,
                                                                      unitResponse, recipResponse, isSeparated,
                                                                      collidingWithVrel, velMultiplier,
                                                                      scaledBias);
        }
        else
        {
            velMultiplier = recipResponse;
            scaledBias = isSeparated ? -invStepDt : -invDtp8;
        }

        gContactPoints[outContactBase + j].resp0[threadIndex] = resp0;
        gContactPoints[outContactBase + j].resp1[threadIndex] = resp1;

        maxPenetration = min(penetration, maxPenetration);

        const bool isGreater2 = ((restitution > 0.0f) && (bounceThreshold > vrel)) && ((-vrel) > penetrationInvDt);

        float targetVelocity = cTargetVel + (isGreater2 ? (-vrel) * restitution : 0.0f);

        if (isKinematic0)
            targetVelocity -= (norVel0 + angNorVel0);
        if (isKinematic1)
            targetVelocity += (norVel1 + angNorVel1);

        if (isGreater2)
        {
            const float ratio = totalDt + penetration / vrel;
            penetration += targetVelocity * ratio;
        }

        gContactPoints[outContactBase + j].raXn_extraCoeff[threadIndex] = float4(raXnSqrtInertia, contactCoeff);
        gContactPoints[outContactBase + j].rbXn_targetVelW[threadIndex] = float4(rbXnSqrtInertia, targetVelocity);
        gContactPoints[outContactBase + j].separation[threadIndex] = penetration;
        gContactPoints[outContactBase + j].maxImpulse[threadIndex] = min(maxImpulse, targetVel4_maxImpulseW.w);
        gContactPoints[outContactBase + j].appliedForce[threadIndex] = 0.0f;
        gContactPoints[outContactBase + j].biasCoefficient[threadIndex] = scaledBias;
    }

    const float frictionScale = (aCount == 2) ? 0.5f : 1.0f;

    const bool hasTorsionalFriction =
        (torsionalFrictionData.x > 0.0f || torsionalFrictionData.y > 0.0f) && aCount == 1;

    const float staticFriction = staticFrictionCof * frictionScale;
    const float dynamicFriction = dynamicFrictionCof * frictionScale;

    const bool haveFriction = (anchorCount != 0);
    gContactHeaders[headerIndex].numNormalConstr[threadIndex] = contactCount;
    gFrictionHeaders[headerIndex].numFrictionConstr[threadIndex] = anchorCount + (hasTorsionalFriction ? 1u : 0u);

    gContactHeaders[headerIndex].normal_staticFriction[threadIndex] = float4(normal, staticFriction);
    gFrictionHeaders[headerIndex].dynamicFriction[threadIndex] = dynamicFriction;

    if (haveFriction)
    {
        const float3 vrel = linVel0 - linVel1;

        const float orthoThreshold = 0.70710678f;
        const float p1 = 0.0001f;
        const float normalX = normal.x;
        const float normalY = normal.y;
        const float normalZ = normal.z;

        const float3 t0Fallback1 = float3(0.0f, -normalZ, normalY);
        const float3 t0Fallback2 = float3(-normalY, normalX, 0.0f);
        const float3 t0Fallback = orthoThreshold > abs(normalX) ? t0Fallback1 : t0Fallback2;

        float3 t0 = vrel - normal * dot(normal, vrel);
        t0 = dot(t0, t0) > p1 ? t0 : t0Fallback;
        t0 = normalize(t0);

        const float3 t1 = cross(normal, t0);

        gFrictionHeaders[headerIndex].broken[threadIndex] = 0;
        gFrictionHeaders[headerIndex].frictionNormals0[threadIndex] = float4(t0, 0.0f);
        gFrictionHeaders[headerIndex].frictionNormals1[threadIndex] = float4(t1, 0.0f);
        gFrictionHeaders[headerIndex].biasCoefficient[threadIndex] = invStepDt;

        for (uint j = 0; j < aCount; j++)
        {
            const uint f0 = outFrictionBase + 2 * j;
            const uint f1 = outFrictionBase + 2 * j + 1;

            const float4 body0Anchor4 = j == 0 ? gCurrAnchors[prepIndex].body0Anchors0[threadIndex]
                                               : gCurrAnchors[prepIndex].body0Anchors1[threadIndex];
            const float4 body1Anchor4 = j == 0 ? gCurrAnchors[prepIndex].body1Anchors0[threadIndex]
                                               : gCurrAnchors[prepIndex].body1Anchors1[threadIndex];

            const float3 ra = quatRotate(bodyFrame0.q, body0Anchor4.xyz);
            const float3 rb = quatRotate(bodyFrame1.q, body1Anchor4.xyz);

            const float3 error = (ra + bodyFrame0p) - (rb + bodyFrame1p);

            const uint index = perPointFriction
                ? (j == 0 ? gCurrFrictionPatches[prepIndex].contactID0[threadIndex]
                          : gCurrFrictionPatches[prepIndex].contactID1[threadIndex])
                : 0;

            const float4 targetVel4 = gContactPrepPoints[contactBase + index].targetVel_maxImpulseW[threadIndex];
            const float3 tvel = targetVel4.xyz;

            const float3 vel1 = linVel0 + cross(angVel0, ra);
            const float3 vel2 = linVel1 + cross(angVel1, rb);

            float3 tv = tvel;
            if (isKinematic0)
                tv -= vel1;
            if (isKinematic1)
                tv += vel2;

            {
                float3 raXn = cross(ra, t0);
                float3 rbXn = cross(rb, t0);

                raXn.x = abs(raXn.x) < solverOffsetSlop ? 0.0f : raXn.x;
                raXn.y = abs(raXn.y) < solverOffsetSlop ? 0.0f : raXn.y;
                raXn.z = abs(raXn.z) < solverOffsetSlop ? 0.0f : raXn.z;

                rbXn.x = abs(rbXn.x) < solverOffsetSlop ? 0.0f : rbXn.x;
                rbXn.y = abs(rbXn.y) < solverOffsetSlop ? 0.0f : rbXn.y;
                rbXn.z = abs(rbXn.z) < solverOffsetSlop ? 0.0f : rbXn.z;

                const float3 raXnSqrtInertia = mat33Transform(sqrtInvInertia0, raXn);
                const float3 rbXnSqrtInertia = mat33Transform(sqrtInvInertia1, rbXn);

                const float resp0 = angD0 * dot(raXnSqrtInertia, raXnSqrtInertia) + invMass0D0;
                const float resp1 = angD1 * dot(rbXnSqrtInertia, rbXnSqrtInertia) + invMass1D1;

                const float targetVelF = dot(tv, t0);
                const float frictionError = dot(t0, error);

                gFrictions[f0].resp0[threadIndex] = resp0;
                gFrictions[f0].resp1[threadIndex] = resp1;
                gFrictions[f0].raXn_error[threadIndex] = float4(raXnSqrtInertia, frictionError);
                gFrictions[f0].rbXn_targetVelW[threadIndex] = float4(rbXnSqrtInertia, targetVelF);
                gFrictions[f0].appliedForce[threadIndex] = 0.0f;
            }

            {
                float3 raXn = cross(ra, t1);
                float3 rbXn = cross(rb, t1);

                raXn.x = abs(raXn.x) < solverOffsetSlop ? 0.0f : raXn.x;
                raXn.y = abs(raXn.y) < solverOffsetSlop ? 0.0f : raXn.y;
                raXn.z = abs(raXn.z) < solverOffsetSlop ? 0.0f : raXn.z;

                rbXn.x = abs(rbXn.x) < solverOffsetSlop ? 0.0f : rbXn.x;
                rbXn.y = abs(rbXn.y) < solverOffsetSlop ? 0.0f : rbXn.y;
                rbXn.z = abs(rbXn.z) < solverOffsetSlop ? 0.0f : rbXn.z;

                const float3 raXnSqrtInertia = mat33Transform(sqrtInvInertia0, raXn);
                const float3 rbXnSqrtInertia = mat33Transform(sqrtInvInertia1, rbXn);

                const float resp0 = angD0 * dot(raXnSqrtInertia, raXnSqrtInertia) + invMass0D0;
                const float resp1 = angD1 * dot(rbXnSqrtInertia, rbXnSqrtInertia) + invMass1D1;

                const float targetVelF = dot(tv, t1);
                const float frictionError = dot(t1, error);

                gFrictions[f1].resp0[threadIndex] = resp0;
                gFrictions[f1].resp1[threadIndex] = resp1;
                gFrictions[f1].raXn_error[threadIndex] = float4(raXnSqrtInertia, frictionError);
                gFrictions[f1].rbXn_targetVelW[threadIndex] = float4(rbXnSqrtInertia, targetVelF);
                gFrictions[f1].appliedForce[threadIndex] = 0.0f;
            }
        }

        if (hasTorsionalFriction)
        {
            const uint f0 = outFrictionBase + 2; // frictionConstraints[2]

            const float torsionalPatchRadius = torsionalFrictionData.x;
            const float minTorsionalPatchRadius = torsionalFrictionData.y;

            const float torsionalFriction =
                max(minTorsionalPatchRadius, sqrt(max(0.0f, -maxPenetration) * torsionalPatchRadius));

            gFrictionHeaders[headerIndex].torsionalFrictionScale[threadIndex] = torsionalFriction;

            const float3 raXnI = mat33Transform(sqrtInvInertia0, normal);
            const float3 rbXnI = mat33Transform(sqrtInvInertia1, normal);

            const float resp0 = angD0 * dot(raXnI, raXnI);
            const float resp1 = angD1 * dot(rbXnI, rbXnI);

            float targetVelF = 0.0f;
            if (isKinematic0)
                targetVelF += dot(normal, angVel0);
            if (isKinematic1)
                targetVelF -= dot(normal, angVel1);

            const float frictionError = 0.0f;

            gFrictions[f0].resp0[threadIndex] = resp0;
            gFrictions[f0].resp1[threadIndex] = resp1;
            gFrictions[f0].raXn_error[threadIndex] = float4(raXnI, frictionError);
            gFrictions[f0].rbXn_targetVelW[threadIndex] = float4(rbXnI, targetVelF);
            gFrictions[f0].appliedForce[threadIndex] = 0.0f;
        }
    }
}

// mirrors contactConstraintBlockPrepareParallelLaunchTGS()
[shader("compute")]
[numthreads(32, 1, 1)]
void contactConstraintBlockPrepareParallelLaunchTGS(uint3 groupThreadId : SV_GroupThreadID,
                                                    uint3 groupId : SV_GroupID)
{
    const uint threadIndexInWarp = groupThreadId.x;
    const uint i = groupId.x;

    if (i >= gParams.numBatches)
        return;

    const uint batchIndex = gBatchIndices[i];

    const uint descStride = gBatches[batchIndex].descStride_constraintType & 0xFFFFu;
    if (threadIndexInWarp >= descStride)
        return;

    const uint bodyAIndex = gBatches[batchIndex].bodyAIndex[threadIndexInWarp];
    const uint bodyBIndex = gBatches[batchIndex].bodyBIndex[threadIndexInWarp];
    const uint descIndexBatch = gBatches[batchIndex].mConstraintBatchIndex;

    const PxgMat33 invInertia0 = mat33Load(gTxIData[bodyAIndex].sqrtInvInertia);
    const PxgMat33 invInertia1 = mat33Load(gTxIData[bodyBIndex].sqrtInvInertia);

    const uint edgeIndex = gWorkUnits[descIndexBatch].mEdgeIndex[threadIndexInWarp];
    const uint frictionIndex = edgeIndex + gParams.totalCurrentEdges * gWorkUnits[descIndexBatch].mPatchIndex[threadIndexInWarp];
    gCurrFrictionIndices[frictionIndex] =
        uint2((descIndexBatch << 5) | threadIndexInWarp, descIndexBatch >> 27);

    const PxgSolverBodyData bodyData0 = gSolverBodyData[bodyAIndex];
    const PxgSolverBodyData bodyData1 = gSolverBodyData[bodyBIndex];

    const PxgAlignedTransform bodyFrame0 = gBodyFrames[bodyAIndex];
    const PxgAlignedTransform bodyFrame1 = gBodyFrames[bodyBIndex];

    const float4 linVel_invMass0 = bodyData0.initialLinVelXYZ_invMassW;
    const float4 angVelXYZ_penBiasClamp0 = bodyData0.initialAngVelXYZ_penBiasClamp;
    const float4 linVel_invMass1 = bodyData1.initialLinVelXYZ_invMassW;
    const float4 angVelXYZ_penBiasClamp1 = bodyData1.initialAngVelXYZ_penBiasClamp;

    const float offsetSlop = max(bodyData0.offsetSlop, bodyData1.offsetSlop);

    const uint forceWritebackOffset = gWorkUnits[descIndexBatch].mWriteback[threadIndexInWarp];
    const float2 torsionalData = gWorkUnits[descIndexBatch].mTorsionalFrictionData[threadIndexInWarp];

    // ---- createFinalizeSolverContactsBlockGPUTGS ----

    const float4 data_ = gContactPrepData[descIndexBatch].contactData[threadIndexInWarp];
    const uint materialWord = asuint(data_.w);
    const uint nbContacts = materialWord & 0xFFu;
    const uint flags = (materialWord >> 8) & 0xFFu;
    const uint prevFrictionPatchCount = (materialWord >> 16) & 0xFFu;
    const float restDistance = data_.x;
    const float staticFrictionCof = data_.y;
    const float dynamicFrictionCof = data_.z;

    const float4 normal4 = gContactPrepData[descIndexBatch].normal_restitutionW[threadIndexInWarp];
    const float3 normal = normal4.xyz;

    const uint contactBase = gBatches[batchIndex].blockContactIndex;

    correlatePatches(gCurrFrictionPatches, descIndexBatch, nbContacts, normal, bodyFrame0, bodyFrame1,
                     threadIndexInWarp);

    const bool perPointFriction = (flags & PXG_SC_FLAG_PER_POINT_FRICTION) != 0;
    const bool disableFriction = (flags & PXG_SC_FLAG_DISABLE_FRICTION) != 0;

    float patchExtents = 0.0f;

    gCurrFrictionPatches[descIndexBatch].broken[threadIndexInWarp] = 0;

    if (!(perPointFriction || disableFriction))
    {
        getFrictionPatches(gCurrFrictionPatches, descIndexBatch, gCurrAnchors, descIndexBatch,
                           gPrevFrictionIndices, gParams.hasPrevFrictionIndices != 0, edgeIndex,
                           gPrevFrictionPatches, gPrevAnchors, prevFrictionPatchCount, bodyFrame0,
                           bodyFrame1, gParams.correlationDistance, threadIndexInWarp,
                           gParams.totalPreviousEdges, patchExtents, nbContacts);
    }

    if (!disableFriction)
        growPatches(gCurrFrictionPatches, descIndexBatch, gCurrAnchors, descIndexBatch, gContactPrepPoints,
                    contactBase, nbContacts, bodyFrame0, bodyFrame1,
                    gParams.frictionOffsetThreshold + restDistance, threadIndexInWarp, patchExtents);

    setupFinalizeSolverConstraintsBlockTGS(descIndexBatch, contactBase, nbContacts, bodyFrame0, bodyFrame1,
                                           bodyData0, bodyData1, linVel_invMass0, angVelXYZ_penBiasClamp0,
                                           linVel_invMass1, angVelXYZ_penBiasClamp1, invInertia0, invInertia1,
                                           gParams.invStepDt, gParams.stepDt, gParams.totalDt,
                                           gParams.invTotalDt, gParams.bounceThreshold, gParams.biasCoefficient,
                                           threadIndexInWarp, forceWritebackOffset, perPointFriction,
                                           descIndexBatch, gBatches[batchIndex].startConstraintIndex,
                                           gBatches[batchIndex].startFrictionIndex, restDistance,
                                           staticFrictionCof, dynamicFrictionCof, flags, offsetSlop,
                                           torsionalData);

    gCurrFrictionPatches[descIndexBatch].patchIndex[threadIndexInWarp] =
        gWorkUnits[descIndexBatch].mFrictionPatchIndex[threadIndexInWarp];

    // TGS wrapper uses the float4 transform path (w = 0)
    if (gCurrFrictionPatches[descIndexBatch].anchorCount[threadIndexInWarp] >= 1)
        gCurrFrictionPatches[descIndexBatch].anchorPoints0[threadIndexInWarp] =
            alignedTransformPoint(bodyFrame0, gCurrAnchors[descIndexBatch].body0Anchors0[threadIndexInWarp]);
    if (gCurrFrictionPatches[descIndexBatch].anchorCount[threadIndexInWarp] == 2)
        gCurrFrictionPatches[descIndexBatch].anchorPoints1[threadIndexInWarp] =
            alignedTransformPoint(bodyFrame0, gCurrAnchors[descIndexBatch].body0Anchors1[threadIndexInWarp]);
}

// ---------------------------------------------------------------------------
// joint prep (jointConstraintBlockPrepTGS.cuh)
// ---------------------------------------------------------------------------

// orthogonalize() is identical to the PGS variant (shared source); copied here
void orthogonalize(uint sortedRowIndices[MAX_CONSTRAINT_ROWS], uint startOffset, uint rowBase,
                   inout float3 angSqrtInvInertia0[MAX_CONSTRAINT_ROWS],
                   inout float3 angSqrtInvInertia1[MAX_CONSTRAINT_ROWS], uint rowCount, uint eqRowCount,
                   float invMass0, float invMass1, float invInertiaScale0, float invInertiaScale1,
                   uint threadIndex)
{
    float3 lin1m[6], ang1m[6], lin1[6], ang1[6];
    float3 lin0m[6], ang0m[6], lin0[6], ang0[6];
    float geomErr[6];
    float velTarget[6];

    for (uint i = 0; i < rowCount; i++)
    {
        const uint index = rowBase + sortedRowIndices[startOffset + i];
        const float4 linear0XYZ_geometricErrorW = gJointVel[index].linear0XYZ_geometricErrorW[threadIndex];
        const float4 linear1XYZ_minImpulseW = gJointVel[index].linear1XYZ_minImpulseW[threadIndex];
        const float4 angular0XYZ_velocityTargetW = gJointVel[index].angular0XYZ_velocityTargetW[threadIndex];
        const float4 angular1XYZ_maxImpulseW = gJointVel[index].angular1XYZ_maxImpulseW[threadIndex];

        float3 l0 = linear0XYZ_geometricErrorW.xyz;
        float3 a0 = angular0XYZ_velocityTargetW.xyz;
        float3 l1 = linear1XYZ_minImpulseW.xyz;
        float3 a1 = angular1XYZ_maxImpulseW.xyz;

        float3 angSqrtL0 = angSqrtInvInertia0[startOffset + i];
        float3 angSqrtL1 = angSqrtInvInertia1[startOffset + i];

        float g = linear0XYZ_geometricErrorW.w;
        float T = angular0XYZ_velocityTargetW.w;

        const uint eliminationRows = min(i, eqRowCount);
        for (uint j = 0; j < eliminationRows; j++)
        {
            const float3 s0 = l1 * lin1m[j] + l0 * lin0m[j];
            const float3 s1 = angSqrtL1 * ang1m[j] + angSqrtL0 * ang0m[j];
            const float3 s0s1 = s0 + s1;
            const float t = s0s1.x + s0s1.y + s0s1.z;

            l0 = l0 - lin0[j] * t;
            a0 = a0 - ang0[j] * t;
            l1 = l1 - lin1[j] * t;
            a1 = a1 - ang1[j] * t;
            g = g - geomErr[j] * t;
            T = T - velTarget[j] * t;
            angSqrtL0 = angSqrtL0 - angSqrtInvInertia0[startOffset + j] * t;
            angSqrtL1 = angSqrtL1 - angSqrtInvInertia1[startOffset + j] * t;
        }

        gJointVel[index].linear0XYZ_geometricErrorW[threadIndex] = float4(l0, g);
        gJointVel[index].angular0XYZ_velocityTargetW[threadIndex] = float4(a0, T);
        gJointVel[index].linear1XYZ_minImpulseW[threadIndex] = float4(l1, linear1XYZ_minImpulseW.w);
        gJointVel[index].angular1XYZ_maxImpulseW[threadIndex] = float4(a1, angular1XYZ_maxImpulseW.w);
        angSqrtInvInertia0[startOffset + i] = angSqrtL0;
        angSqrtInvInertia1[startOffset + i] = angSqrtL1;

        if (i < eqRowCount)
        {
            lin0[i] = l0;
            ang0[i] = a0;
            geomErr[i] = g;
            velTarget[i] = T;
            lin1[i] = l1;
            ang1[i] = a1;

            const float3 l0m = l0 * invMass0;
            const float3 l1m = l1 * invMass1;
            const float3 a0m = angSqrtL0 * invInertiaScale0;
            const float3 a1m = angSqrtL1 * invInertiaScale1;

            const float3 s0 = l0 * l0m + l1 * l1m;
            const float3 s1 = a0m * angSqrtL0 + a1m * angSqrtL1;
            const float3 s0s1 = s0 + s1;
            const float s = s0s1.x + s0s1.y + s0s1.z;
            const float a = s > 0.0f ? 1.0f / s : 0.0f;

            lin0m[i] = l0m * a;
            ang0m[i] = a0m * a;
            lin1m[i] = l1m * a;
            ang1m[i] = a1m * a;
        }
    }
}

// mirrors preprocessRows() (identical to PGS)
void preprocessRows(inout uint sortedRowIndices[MAX_CONSTRAINT_ROWS], uint jointDataIndex, uint rowBase,
                    out float3 angSqrtInvInertia0[MAX_CONSTRAINT_ROWS],
                    out float3 angSqrtInvInertia1[MAX_CONSTRAINT_ROWS], float4 invMassScale, float iMass0,
                    float iMass1, PxgMat33 i0, PxgMat33 i1, uint threadIndex, bool disablePreprocessing)
{
    const uint numRows = gJointData[jointDataIndex].mNumRows[threadIndex];

    for (uint i = 0; i < numRows; i++)
    {
        const uint hint = gJointPar[rowBase + i].solveHint[threadIndex];
        uint j = i;
        for (; j > 0 && hint < gJointPar[rowBase + sortedRowIndices[j - 1]].solveHint[threadIndex]; j--)
            sortedRowIndices[j] = sortedRowIndices[j - 1];
        sortedRowIndices[j] = i;
    }

    const float invMass0 = iMass0 * invMassScale.x;
    const float invMass1 = iMass1 * invMassScale.z;
    const float invInertiaScale0 = invMassScale.y;
    const float invInertiaScale1 = invMassScale.w;

    for (uint i = 0; i < numRows; ++i)
    {
        const uint index = rowBase + sortedRowIndices[i];
        const float3 angular0 = gJointVel[index].angular0XYZ_velocityTargetW[threadIndex].xyz;
        const float3 angular1 = gJointVel[index].angular1XYZ_maxImpulseW[threadIndex].xyz;
        angSqrtInvInertia0[i] = mat33Transform(i0, angular0);
        angSqrtInvInertia1[i] = mat33Transform(i1, angular1);
    }

    if (!disablePreprocessing)
    {
        for (uint i = 0; i < numRows;)
        {
            const uint groupMajorId = gJointPar[rowBase + sortedRowIndices[i]].solveHint[threadIndex] >> 8;
            const uint start = i++;
            while (i < numRows &&
                   (gJointPar[rowBase + sortedRowIndices[i]].solveHint[threadIndex] >> 8) == groupMajorId)
                i++;

            if (groupMajorId == 4 || groupMajorId == 8)
            {
                uint bCount = start;
                for (; bCount < i &&
                       (gJointPar[rowBase + sortedRowIndices[bCount]].solveHint[threadIndex] & 255u) == 0;
                     bCount++)
                    ;

                orthogonalize(sortedRowIndices, start, rowBase, angSqrtInvInertia0, angSqrtInvInertia1,
                              i - start, bCount - start, invMass0, invMass1, invInertiaScale0,
                              invInertiaScale1, threadIndex);
            }
        }
    }
}

// mirrors intializeBlock1DTGS(); returns 1 for a rotational-equality row (which
// contributes an ortho axis into slot eqCount), 0 otherwise
uint intializeBlock1DTGS(uint index, uint jointDataIndex, uint headerIndex, uint outRow,
                         float jointSpeedForRestitutionBounce, float initJointSpeed, float unitResponse,
                         float minRowResponse, float erp, float lengthScale, float stepDt, float simDt,
                         float recipStepDt, float recipSimDt, float3 angSqrtInvInertia0,
                         float3 angSqrtInvInertia1, float invInertiaScale0, float invInertiaScale1,
                         uint eqCount, uint threadIndex, bool disablePreprocessing)
{
    const float geometricError = gJointVel[index].linear0XYZ_geometricErrorW[threadIndex].w;

    const uint flags = gJointPar[index].flags[threadIndex];
    const float stiffness = gJointPar[index].springStiffness_restitution[threadIndex];
    const float damping = gJointPar[index].springDamping_velocityThreshold[threadIndex];
    const float restitution = gJointPar[index].springStiffness_restitution[threadIndex];
    const float bounceVelocityThreshold = gJointPar[index].springDamping_velocityThreshold[threadIndex];
    const float velocityTarget = gJointVel[index].angular0XYZ_velocityTargetW[threadIndex].w;

    const float maxBiasVelocity = computeMaxBiasVelocityTGS(flags, jointSpeedForRestitutionBounce,
                                                            bounceVelocityThreshold, restitution,
                                                            geometricError, false, lengthScale, recipSimDt);

    const float recipUnitResponse = computeRecipUnitResponse(unitResponse, minRowResponse);

    float biasScale, error, velMultiplier, targetVel;
    compute1dConstraintSolverConstantsTGSFull(flags, stiffness, damping, restitution, bounceVelocityThreshold,
                                              geometricError, velocityTarget, jointSpeedForRestitutionBounce,
                                              initJointSpeed, unitResponse, recipUnitResponse, erp, stepDt,
                                              recipStepDt, biasScale, error, velMultiplier, targetVel);

    float4 lin1XYZ_biasScale = gJointVel[index].linear1XYZ_minImpulseW[threadIndex];
    lin1XYZ_biasScale.w = biasScale;
    float4 lin0XYZ_initBiasW = gJointVel[index].linear0XYZ_geometricErrorW[threadIndex];
    lin0XYZ_initBiasW.w = error;
    float4 ang0XYZ_velMultiplierW = gJointVel[index].angular0XYZ_velocityTargetW[threadIndex];
    ang0XYZ_velMultiplierW.w = velMultiplier;
    float4 ang1XYZ_velTargetW = gJointVel[index].angular1XYZ_maxImpulseW[threadIndex];
    ang1XYZ_velTargetW.w = targetVel;

    float angularErrorScale = 1.0f;
    if ((flags & PX1D_FLAG_ANGULAR_CONSTRAINT) == 0)
    {
        ang0XYZ_velMultiplierW.x = ang0XYZ_velMultiplierW.y = ang0XYZ_velMultiplierW.z = 0.0f;
        ang1XYZ_velTargetW.x = ang1XYZ_velTargetW.y = ang1XYZ_velTargetW.z = 0.0f;
        angularErrorScale = 0.0f;
    }

    gJointRowsCon[outRow].lin0XYZ_initBiasOrCoeff0[threadIndex] = lin0XYZ_initBiasW;
    gJointRowsCon[outRow].lin1XYZ_biasScaleOrCoeff1[threadIndex] = lin1XYZ_biasScale;
    gJointRowsCon[outRow].ang0XYZ_velMultiplierOrCoeff2[threadIndex] = ang0XYZ_velMultiplierW;
    gJointRowsCon[outRow].ang1XYZ_velTargetOrCoeff3[threadIndex] = ang1XYZ_velTargetW;

    gJointRowsCon[outRow].maxBias[threadIndex] = maxBiasVelocity;
    gJointRowsCon[outRow].angularErrorScale[threadIndex] = angularErrorScale;
    gJointRowsCon[outRow].appliedForce[threadIndex] = 0.0f;
    gJointRowsCon[outRow].residual[threadIndex] = 0.0f;

    {
        const bool hasDriveLimit = (flags & PX1D_FLAG_HAS_DRIVE_LIMIT) != 0;
        const bool driveLimitsAreForces =
            (gJointData[jointDataIndex].mFlags[threadIndex] & PX_CONSTRAINT_FLAG_DRIVE_LIMITS_ARE_FORCES) != 0;
        float minImpulse, maxImpulse;
        computeMinMaxImpulseOrForceAsImpulse(gJointVel[index].linear1XYZ_minImpulseW[threadIndex].w,
                                             gJointVel[index].angular1XYZ_maxImpulseW[threadIndex].w,
                                             hasDriveLimit, driveLimitsAreForces, simDt, minImpulse,
                                             maxImpulse);
        gJointRowsCon[outRow].minImpulse[threadIndex] = minImpulse;
        gJointRowsCon[outRow].maxImpulse[threadIndex] = maxImpulse;
    }

    uint outFlags = 0;
    const uint solveHint = gJointPar[index].solveHint[threadIndex];
    raiseInternalFlagsTGS(flags, solveHint, outFlags);

    uint ret = 0;
    if (!disablePreprocessing)
    {
        if (solveHint == PX_SOLVE_HINT_ROTATIONAL_EQUALITY)
        {
            ret = 1;
            outFlags |= DY_SC_FLAG_ROT_EQ;

            const float4 axis0 = float4(angSqrtInvInertia0 * invInertiaScale0, recipUnitResponse);
            const float4 axis1 = float4(angSqrtInvInertia1 * invInertiaScale1, geometricError);
            if (eqCount == 0)
            {
                gJointHeaders[headerIndex].angOrthoAxis0_recipResponseW0[threadIndex] = axis0;
                gJointHeaders[headerIndex].angOrthoAxis1_ErrorW0[threadIndex] = axis1;
            }
            else if (eqCount == 1)
            {
                gJointHeaders[headerIndex].angOrthoAxis0_recipResponseW1[threadIndex] = axis0;
                gJointHeaders[headerIndex].angOrthoAxis1_ErrorW1[threadIndex] = axis1;
            }
            else
            {
                gJointHeaders[headerIndex].angOrthoAxis0_recipResponseW2[threadIndex] = axis0;
                gJointHeaders[headerIndex].angOrthoAxis1_ErrorW2[threadIndex] = axis1;
            }
        }
        else if ((solveHint & PX_SOLVE_HINT_EQUALITY) != 0)
        {
            outFlags |= DY_SC_FLAG_ORTHO_TARGET;
        }
    }

    gJointRowsCon[outRow].flags[threadIndex] = outFlags;

    return ret;
}

void setOrthoAxisZero(uint headerIndex, uint slot, uint threadIndex)
{
    if (slot == 0)
    {
        gJointHeaders[headerIndex].angOrthoAxis0_recipResponseW0[threadIndex] = float4(0.0f);
        gJointHeaders[headerIndex].angOrthoAxis1_ErrorW0[threadIndex] = float4(0.0f);
    }
    else if (slot == 1)
    {
        gJointHeaders[headerIndex].angOrthoAxis0_recipResponseW1[threadIndex] = float4(0.0f);
        gJointHeaders[headerIndex].angOrthoAxis1_ErrorW1[threadIndex] = float4(0.0f);
    }
    else
    {
        gJointHeaders[headerIndex].angOrthoAxis0_recipResponseW2[threadIndex] = float4(0.0f);
        gJointHeaders[headerIndex].angOrthoAxis1_ErrorW2[threadIndex] = float4(0.0f);
    }
}

// mirrors jointConstraintBlockPrepareParallelLaunchTGS() +
// setupSolverConstraintBlockGPUTGS() + setUp1DConstraintBlockTGS()
[shader("compute")]
[numthreads(32, 1, 1)]
void jointConstraintBlockPrepareParallelLaunchTGS(uint3 groupThreadId : SV_GroupThreadID,
                                                  uint3 groupId : SV_GroupID)
{
    const uint threadIndexInWarp = groupThreadId.x;
    const uint i = groupId.x;

    if (i >= gParams.numBatches)
        return;

    const uint batchIndex = gBatchIndices[i];

    const uint descStride = gBatches[batchIndex].descStride_constraintType & 0xFFFFu;
    if (threadIndexInWarp >= descStride)
        return;

    const uint bodyAIndex = gBatches[batchIndex].bodyAIndex[threadIndexInWarp];
    const uint bodyBIndex = gBatches[batchIndex].bodyBIndex[threadIndexInWarp];
    const uint descIndexBatch = gBatches[batchIndex].mConstraintBatchIndex;
    const uint rowBase = descIndexBatch * MAX_CONSTRAINT_ROWS;
    const uint outRowBase = gBatches[batchIndex].startConstraintIndex;

    const uint uniqueIndex =
        gConstraintUniqueIndices[gBatches[batchIndex].mStartPartitionIndex + threadIndexInWarp];

    const PxgSolverBodyData bodyData0 = gSolverBodyData[bodyAIndex];
    const PxgSolverBodyData bodyData1 = gSolverBodyData[bodyBIndex];

    const float stepDt = gParams.stepDt;
    const float recipStepDt = gParams.invStepDt;
    const float simDt = gParams.totalDt;
    const float recipSimDt = gParams.invTotalDt;
    const float lengthScale = gParams.lengthScale;
    const float biasCoefficient = gParams.biasCoefficient;
    const float erp = 0.5f * biasCoefficient;

    // ---- setupSolverConstraintBlockGPUTGS ----

    const uint numRows = gJointData[descIndexBatch].mNumRows[threadIndexInWarp];

    // rowCounts_breakable_orthoAxisCount is a uchar4 per lane; assemble it in a
    // register and store once at the end.
    uint rc_x = numRows & 0xFFu;

    gJointHeaders[descIndexBatch].writeBackOffset[threadIndexInWarp] = gSolverConstantData[uniqueIndex].y;

    const float4 lin0_ang0_lin1_ang1 = gJointData[descIndexBatch].mInvMassScale[threadIndexInWarp];

    const float4 initialLinVelXYZ_invMassW0 = bodyData0.initialLinVelXYZ_invMassW;
    const float4 initialLinVelXYZ_invMassW1 = bodyData1.initialLinVelXYZ_invMassW;

    const float4 rAWorld_linBreakForce = gJointData[descIndexBatch].mRAWorld_linBreakForce[threadIndexInWarp];
    const float4 rBWorld_AngBreakForce = gJointData[descIndexBatch].mRBWorld_AngBreakForce[threadIndexInWarp];

    const float linBreakImpulse = rAWorld_linBreakForce.w * simDt;
    const float angBreakImpulse = rBWorld_AngBreakForce.w * simDt;
    gJointHeaders[descIndexBatch].linBreakImpulse[threadIndexInWarp] = linBreakImpulse;
    gJointHeaders[descIndexBatch].angBreakImpulse[threadIndexInWarp] = angBreakImpulse;

    const float invMass0 = initialLinVelXYZ_invMassW0.w * lin0_ang0_lin1_ang1.x;
    const float invMass1 = initialLinVelXYZ_invMassW1.w * lin0_ang0_lin1_ang1.z;

    gJointHeaders[descIndexBatch].rAWorld_invMass0D0[threadIndexInWarp] = float4(rAWorld_linBreakForce.xyz, invMass0);
    gJointHeaders[descIndexBatch].rBWorld_invMass1D1[threadIndexInWarp] = float4(rBWorld_AngBreakForce.xyz, invMass1);
    gJointHeaders[descIndexBatch].invInertiaScale0[threadIndexInWarp] = lin0_ang0_lin1_ang1.y;
    gJointHeaders[descIndexBatch].invInertiaScale1[threadIndexInWarp] = lin0_ang0_lin1_ang1.w;

    uint rc_y = ((rAWorld_linBreakForce.w != PREP_PX_MAX_F32) || (rBWorld_AngBreakForce.w != PREP_PX_MAX_F32)) ? 1u : 0u;

    // Promote angular constraints' solve hints (modifies the input parameters
    // buffer in place before sorting)
    for (uint r = 0; r < numRows; ++r)
    {
        if ((gJointPar[rowBase + r].flags[threadIndexInWarp] & PX1D_FLAG_ANGULAR_CONSTRAINT) != 0)
        {
            uint hint = gJointPar[rowBase + r].solveHint[threadIndexInWarp];
            if (hint == PX_SOLVE_HINT_EQUALITY)
                hint = PX_SOLVE_HINT_ROTATIONAL_EQUALITY;
            else if (hint == PX_SOLVE_HINT_INEQUALITY)
                hint = PX_SOLVE_HINT_ROTATIONAL_INEQUALITY;
            gJointPar[rowBase + r].solveHint[threadIndexInWarp] = hint;
        }
    }

    const bool disablePreprocessing =
        (gJointData[descIndexBatch].mFlags[threadIndexInWarp] & PX_CONSTRAINT_FLAG_DISABLE_PREPROCESSING) != 0;

    uint sortedRowIndices[MAX_CONSTRAINT_ROWS];
    float3 angSqrtInvInertia0[MAX_CONSTRAINT_ROWS];
    float3 angSqrtInvInertia1[MAX_CONSTRAINT_ROWS];

    const PxgMat33 i0 = mat33Load(gTxIData[bodyAIndex].sqrtInvInertia);
    const PxgMat33 i1 = mat33Load(gTxIData[bodyBIndex].sqrtInvInertia);

    preprocessRows(sortedRowIndices, descIndexBatch, rowBase, angSqrtInvInertia0, angSqrtInvInertia1,
                   lin0_ang0_lin1_ang1, initialLinVelXYZ_invMassW0.w, initialLinVelXYZ_invMassW1.w, i0, i1,
                   threadIndexInWarp, disablePreprocessing);

    // ---- setUp1DConstraintBlockTGS ----

    const bool isKinematic0 = (bodyData0.flags & PX_RIGID_BODY_FLAG_KINEMATIC) != 0;
    const bool isKinematic1 = (bodyData1.flags & PX_RIGID_BODY_FLAG_KINEMATIC) != 0;

    uint eqCount = 0;
    for (uint r = 0; r < numRows; r++)
    {
        const uint index = rowBase + sortedRowIndices[r];
        const uint outRow = outRowBase + r;

        const float4 c_linear0XYZ_geometricErrorW = gJointVel[index].linear0XYZ_geometricErrorW[threadIndexInWarp];
        const float4 c_linear1XYZ_minImpulseW = gJointVel[index].linear1XYZ_minImpulseW[threadIndexInWarp];
        const float4 c_angular0XYZ_velocityTargetW = gJointVel[index].angular0XYZ_velocityTargetW[threadIndexInWarp];
        const float4 c_angular1XYZ_maxImpulseW = gJointVel[index].angular1XYZ_maxImpulseW[threadIndexInWarp];

        const float3 clin0 = c_linear0XYZ_geometricErrorW.xyz;
        const float3 clin1 = c_linear1XYZ_minImpulseW.xyz;
        const float3 cang0 = c_angular0XYZ_velocityTargetW.xyz;
        const float3 cang1 = c_angular1XYZ_maxImpulseW.xyz;
        const float3 angSqrt0 = angSqrtInvInertia0[r];
        const float3 angSqrt1 = angSqrtInvInertia1[r];

        float unitResponse;
        {
            const float resp0 = dot(clin0, clin0) * initialLinVelXYZ_invMassW0.w * lin0_ang0_lin1_ang1.x +
                                dot(angSqrt0, angSqrt0) * lin0_ang0_lin1_ang1.y;
            const float resp1 = dot(clin1, clin1) * initialLinVelXYZ_invMassW1.w * lin0_ang0_lin1_ang1.z +
                                dot(angSqrt1, angSqrt1) * lin0_ang0_lin1_ang1.w;
            unitResponse = resp0 + resp1;
        }

        float jointSpeedForRestitutionBounce, initJointSpeed;
        {
            const float vel0 = projectVelocity(bodyData0, clin0, cang0);
            const float vel1 = projectVelocity(bodyData1, clin1, cang1);
            computeJointSpeedTGS(vel0, isKinematic0, vel1, isKinematic1, jointSpeedForRestitutionBounce,
                                 initJointSpeed);
        }

        const float minRowResponse = DY_MIN_RESPONSE;

        eqCount += intializeBlock1DTGS(index, descIndexBatch, descIndexBatch, outRow,
                                       jointSpeedForRestitutionBounce, initJointSpeed, unitResponse,
                                       minRowResponse, erp, lengthScale, stepDt, simDt, recipStepDt, recipSimDt,
                                       angSqrt0, angSqrt1, lin0_ang0_lin1_ang1.y, lin0_ang0_lin1_ang1.w, eqCount,
                                       threadIndexInWarp, disablePreprocessing);
    }

    for (uint s = eqCount; s < 3; ++s)
        setOrthoAxisZero(descIndexBatch, s, threadIndexInWarp);

    // store the assembled uchar4 rowCounts_breakable_orthoAxisCount
    const uint packed = (rc_x & 0xFFu) | ((rc_y & 0xFFu) << 8) | ((eqCount & 0xFFu) << 16);
    gJointHeaders[descIndexBatch].rowCounts_breakable_orthoAxisCount[threadIndexInWarp] = packed;
}
