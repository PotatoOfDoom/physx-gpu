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

// solverTGS.slang
//
// Slang port of the rigid-body TGS constraint solver
// (gpusolver/src/CUDA/solverMultiBlockTGS.cu + solverBlockTGS.cuh):
//   - solveBlockPartitionTGS (rigid path of solveBlockUnified /
//     solveBlockPartitionTGSInternal: solveContactBlockTGS + solve1DBlockTGS)
//   - concludeBlocksTGS (conclude1DBlockTGS; the contact conclude body is
//     compiled out in the CUDA original and stays a no-op here)
//   - writebackBlocksTGS (writeBackContactBlockTGS + writeBack1DBlockTGS)
//   - writeBackBodiesTGS (accumulated deltas -> motion velocities)
//   - computeAverageSolverBodyVelocityTGS / propagateAverageSolverBodyVelocityTGS
//     (multi-slab mass-splitting velocity averaging + per-substep delta-pose
//     integration)
//
// TGS solves in sub-timesteps: each lane carries velocities AND accumulated
// linear/angular deltas (6 float4 per body pair in the velocity pool,
// readIndex = batch * 192 + lane).  Contacts re-derive the separation from
// the accumulated deltas; joints re-anchor their Jacobians through the
// per-body delta transforms (PxgSolverTxIData) and re-orthogonalize against
// the header's ortho axes.
//
// Differences from CUDA (same policy as solverPGS.slang):
//   - island bookkeeping arrives as a resolved batch range via push constants;
//   - the warp-shuffle coalesced PxgSolverTxIData loader (loadTxInertia) and
//     the explicit "next*" prefetch chains become direct loads — semantically
//     identical (the whole solve is gated on numNormalConstr > 0, so the
//     prefetch always resolves to each row's own data);
//   - the contact-error accumulator statistics are not ported yet; per-row
//     joint residual writeback is.

import pxgSlangCommon;
import pxgSlangSolverDesc;

static const uint DY_SC_FLAG_OUTPUT_FORCE = 1u << 1;
static const uint DY_SC_FLAG_KEEP_BIAS = 1u << 2;
static const uint DY_SC_FLAG_ORTHO_TARGET = 1u << 4;
static const uint DY_SC_FLAG_SPRING = 1u << 5;
static const uint DY_SC_FLAG_INEQUALITY = 1u << 6;
static const uint PXG_CONTACT_FLAG_HAS_FORCE_THRESHOLDS = 1u << 0;
static const uint PXG_CONTACT_FLAG_COMPLIANT_ACCELERATION_SPRING = 1u << 3;
static const uint PXG_CONSTRAINT_CONTACT = 0;
static const float PX_MAX_F32 = 3.4028234663852885981170418348452e+38f;

// ---------------------------------------------------------------------------
// TGS block SoA structures (byte-identical mirrors)
// ---------------------------------------------------------------------------

// mirrors PxgBlockConstraintBatch (1536 bytes) — same as solverPGS.slang
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
    float4 frictionNormals0[32];   // 0
    float4 frictionNormals1[32];   // 512
    uint numFrictionConstr[32];    // 1024
    float dynamicFriction[32];     // 1152
    uint broken[32];               // 1280
    float biasCoefficient[32];     // 1408
    float torsionalFrictionScale[32]; // 1536
}                                  // = 1664

// mirrors PxgTGSBlockSolverContactPoint (1792 bytes)
struct PxgTGSBlockSolverContactPoint
{
    float4 raXn_extraCoeff[32]; // 0 (w = compliant coefficient "a")
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
    float4 raXn_error[32];      // 0 (w = initial error)
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
    float4 angOrthoAxis0_recipResponseW0[32];    // 1536 (angOrthoAxis0_recipResponseW[0])
    float4 angOrthoAxis0_recipResponseW1[32];    // 2048
    float4 angOrthoAxis0_recipResponseW2[32];    // 2560
    float4 angOrthoAxis1_ErrorW0[32];            // 3072 (angOrthoAxis1_ErrorW[0])
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

// mirrors PxgBlockFrictionPatch (2816 bytes, PX_ALIGN(256)) — same as solverPGS.slang
struct PxgBlockFrictionPatch
{
    float4 body0Normal[32];   // 0
    float4 body1Normal[32];   // 512
    uint anchorCount[32];     // 1024
    uint broken[32];          // 1152
    uint contactID0[32];      // 1280 (contactID[0])
    uint contactID1[32];      // 1408 (contactID[1])
    float4 anchorPoints0[32]; // 1536 (anchorPoints[0]; offset is already 256-aligned)
    float4 anchorPoints1[32]; // 2048 (anchorPoints[1])
    uint patchIndex[32];      // 2560
    uint pad0[32];            // 2688 (tail padding to the 256 alignment)
}                             // = 2816

// mirrors Dy::ThresholdStreamElement (40 bytes)
struct ThresholdStreamElement
{
    uint2 shapeInteraction; // host pointer
    float normalForce;
    float threshold;
    uint2 nodeIndexA;
    uint2 nodeIndexB;
    float accumulatedForce;
    uint pad;
}                           // = 40

// mirrors PxgConstraintWriteback (32 bytes)
struct PxgConstraintWriteback
{
    float4 linearImpulse_broken;
    float4 angularImpulse_residual;
}

// mirrors PxgFrictionPatchGPU (52 bytes, tightly packed PxVec3s)
struct PxgFrictionPatchGPU
{
    float p0x, p0y, p0z; // points[0]
    float p1x, p1y, p1z; // points[1]
    float i0x, i0y, i0z; // impulses[0]
    float i1x, i1y, i1z; // impulses[1]
    uint anchors;
}                        // = 52

struct SolverTGSParams
{
    uint startIndex;
    uint endIndex;
    uint residualReportingEnabled;
    uint numSlabs;
    uint numTotalBodies;
    uint bodyOffset;
    float elapsedTime;
    float minPen;
    // conclude/writeback/averaging kernels:
    uint totalBodyCount;        // islandContextPool->mBodyCount (dynamic bodies)
    uint deltaVOffset;          // accumulatedBodyDeltaVOffset
    uint averageOutputOffsets;  // (numBatches + numArticBatches) * 32 * 3 * 2
    uint isVelocityIteration;   // propagate: velocity iterations skip the pose integration
    float invDt;
    float stepDt;
};

[[vk::push_constant]]
ConstantBuffer<SolverTGSParams> gParams;

RWStructuredBuffer<PxgBlockConstraintBatch> gBatches;                   // binding 0
RWStructuredBuffer<PxgTGSBlockSolverContactHeader> gContactHeaders;     // binding 1
RWStructuredBuffer<PxgTGSBlockSolverFrictionHeader> gFrictionHeaders;   // binding 2
RWStructuredBuffer<PxgTGSBlockSolverContactPoint> gContactPoints;       // binding 3
RWStructuredBuffer<PxgTGSBlockSolverContactFriction> gFrictions;        // binding 4
RWStructuredBuffer<PxgTGSBlockSolverConstraint1DHeader> gJointHeaders;  // binding 5
RWStructuredBuffer<PxgTGSBlockSolverConstraint1DCon> gJointRowsCon;     // binding 6
RWStructuredBuffer<PxgSolverTxIData> gTxIData;                          // binding 7
RWStructuredBuffer<float4> gSolverBodyVelPool;                          // binding 8
RWStructuredBuffer<uint> gEncodedReferenceCount;                        // binding 9

// writeback / averaging bindings
RWStructuredBuffer<PxgSolverBodyData> gSolverBodyData;                  // binding 10
RWStructuredBuffer<PxgConstraintWriteback> gConstraintWriteBack;        // binding 11
RWStructuredBuffer<float> gForceBuffer;                                 // binding 12
RWStructuredBuffer<ThresholdStreamElement> gThresholdStream;            // binding 13
RWStructuredBuffer<uint> gThresholdStreamIndex;                         // binding 14 (single atomic counter)
RWStructuredBuffer<PxgBlockFrictionPatch> gBlockFrictionPatches;        // binding 15
RWStructuredBuffer<PxgFrictionPatchGPU> gFrictionPatches;               // binding 16
RWStructuredBuffer<float4> gMotionVelocities;                           // binding 17
RWStructuredBuffer<uint> gSolverReferences;                             // binding 18 (PxgSolverReferences)
RWStructuredBuffer<PxgBodySim> gBodySims;                               // binding 19

uint countActiveSlabs(uint index, uint numSlabs, uint numSolverBodies)
{
    uint referenceCount = 0;
    const uint num32Slabs = (numSlabs + 31) / 32;
    for (uint i = 0; i < num32Slabs; ++i)
        referenceCount += countbits(gEncodedReferenceCount[index + i * numSolverBodies]);
    return max(1u, referenceCount);
}

// mirrors computeCompliantContactCoefficientsTGS() (reduced overload in
// constraintPrepShared.cuh); a = the precomputed compliant coefficient
void computeCompliantContactCoefficientsTGS(uint flags, float nrdt, float unitResponse,
                                            float recipResponse, float a,
                                            inout float velMultiplier, inout float scaledBias)
{
    const bool accelSpring = (flags & PXG_CONTACT_FLAG_COMPLIANT_ACCELERATION_SPRING) != 0;
    const float massIfAccelElseOne = accelSpring ? recipResponse : 1.0f;
    const float oneIfAccelElseR = accelSpring ? 1.0f : unitResponse;
    const float x = 1.0f / (a * oneIfAccelElseR + 1.0f);

    velMultiplier = x * a * massIfAccelElseOne;
    scaledBias = nrdt * x * oneIfAccelElseR;
}

// mirrors Dy::computeMinBiasTGS()
float computeMinBiasTGS(uint flags, float maxBias)
{
    return -((flags & DY_SC_FLAG_INEQUALITY) != 0 ? PX_MAX_F32 : maxBias);
}

// mirrors Dy::computeResolvedGeometricErrorTGS()
float computeResolvedGeometricErrorTGS(float3 deltaLin0, float3 deltaLin1, float3 cLinVel0,
                                       float3 cLinVel1, float3 deltaAngInertia0, float3 deltaAngInertia1,
                                       float3 cAngVelInertia0, float3 cAngVelInertia1,
                                       float angularErrorScale, bool isSpringConstraint,
                                       float targetVel, float elapsedTime)
{
    const float deltaAng = (dot(cAngVelInertia0, deltaAngInertia0) - dot(cAngVelInertia1, deltaAngInertia1)) * angularErrorScale;
    const float deltaLin = dot(cLinVel0, deltaLin0) - dot(cLinVel1, deltaLin1);
    const float motion = deltaLin + deltaAng;
    return isSpringConstraint ? motion : motion - targetVel * elapsedTime;
}

// mirrors solveContactBlockTGS() in solverBlockTGS.cuh
void solveContactBlockTGS(uint batchIndex, inout float3 linVel0, inout float3 angVel0,
                          inout float3 linVel1, inout float3 angVel1,
                          float3 b0LinDelta, float3 b0AngDelta, float3 b1LinDelta, float3 b1AngDelta,
                          uint threadIndex, float elapsedTime, float minPen, float ref0, float ref1)
{
    const uint constraintBatchIndex = gBatches[batchIndex].mConstraintBatchIndex;
    const uint startConstraintIndex = gBatches[batchIndex].startConstraintIndex;
    const uint startFrictionIndex = gBatches[batchIndex].startFrictionIndex;

    float accumulatedNormalImpulse = 0.0f;

    const uint numNormalConstr = gContactHeaders[constraintBatchIndex].numNormalConstr[threadIndex];
    const uint numFrictionConstrTotal = gFrictionHeaders[constraintBatchIndex].numFrictionConstr[threadIndex];
    const uint numFrictionConstr = numFrictionConstrTotal & ~1u;

    float accumDeltaF = 0.0f;

    if (numNormalConstr != 0)
    {
        const float maxPenBias = gContactHeaders[constraintBatchIndex].maxPenBias[threadIndex];
        const float4 invMass0_1_angDom0_1 = gContactHeaders[constraintBatchIndex].invMass0_1_angDom0_1[threadIndex];

        const float invMassA = ref0 * invMass0_1_angDom0_1.x;
        const float invMassB = ref1 * invMass0_1_angDom0_1.y;
        const float angDom0 = ref0 * invMass0_1_angDom0_1.z;
        const float angDom1 = ref1 * invMass0_1_angDom0_1.w;
        const float sumInvMass = invMassA + invMassB;

        const float4 normal_staticFriction = gContactHeaders[constraintBatchIndex].normal_staticFriction[threadIndex];
        const float3 normal = normal_staticFriction.xyz;
        const float staticFrictionCof = normal_staticFriction.w;

        const float restitutionXdt = gContactHeaders[constraintBatchIndex].restitutionXdt[threadIndex];
        const float p8 = gContactHeaders[constraintBatchIndex].p8[threadIndex];
        const uint flags = gContactHeaders[constraintBatchIndex].flags[threadIndex];

        const float3 delLinVel0 = normal * invMassA;
        const float3 delLinVel1 = normal * invMassB;

        const float3 relMotion = b0LinDelta - b1LinDelta;
        const float deltaV = dot(normal, relMotion);

        float relVel1 = dot(linVel0 - linVel1, normal);

        for (uint i = 0; i < numNormalConstr; i++)
        {
            const uint c = startConstraintIndex + i;

            const float4 raXn_contactCoeff = gContactPoints[c].raXn_extraCoeff[threadIndex];
            const float4 rbXn_targetVelW = gContactPoints[c].rbXn_targetVelW[threadIndex];
            const float appliedForce = gContactPoints[c].appliedForce[threadIndex];
            const float separation = gContactPoints[c].separation[threadIndex];
            const float maxImpulse = gContactPoints[c].maxImpulse[threadIndex];

            const float resp0 = ref0 * gContactPoints[c].resp0[threadIndex];
            const float resp1 = ref1 * gContactPoints[c].resp1[threadIndex];
            const float unitResponse = resp0 + resp1;
            const float recipResponse = (unitResponse > 0.0f) ? (1.0f / unitResponse) : 0.0f;

            float velMultiplier = recipResponse;
            float biasCoefficient = gContactPoints[c].biasCoefficient[threadIndex];

            if (restitutionXdt < 0.0f)
                computeCompliantContactCoefficientsTGS(flags, restitutionXdt, unitResponse, recipResponse,
                                                       raXn_contactCoeff.w, velMultiplier, biasCoefficient);

            const float3 raXn = raXn_contactCoeff.xyz;
            const float3 rbXn = rbXn_targetVelW.xyz;
            const float targetVel = rbXn_targetVelW.w;

            // Compute the normal velocity of the constraint.
            const float v0 = dot(angVel0, raXn);
            const float v1 = dot(angVel1, rbXn);
            const float normalVel = relVel1 + (v0 - v1);

            const float sep = max(minPen, separation + deltaV + dot(b0AngDelta, raXn) - dot(b1AngDelta, rbXn));

            // How much the target vel should have changed the position of this constraint...
            const float tVelErr = targetVel * elapsedTime;

            const float biasedErr = recipResponse * min(-maxPenBias, biasCoefficient * (sep - tVelErr));

            // KS - clamp the maximum force
            const float tempDeltaF = biasedErr - (normalVel - targetVel) * velMultiplier;
            const float _deltaF = max(tempDeltaF, -appliedForce);
            const float _newForce = appliedForce + _deltaF;
            const float newForce = min(_newForce, maxImpulse);
            const float deltaF = newForce - appliedForce;

            accumDeltaF += deltaF;
            relVel1 += sumInvMass * deltaF;

            angVel0 += raXn * (deltaF * angDom0);
            angVel1 -= rbXn * (deltaF * angDom1);

            gContactPoints[c].appliedForce[threadIndex] = newForce;
            accumulatedNormalImpulse += newForce;
        }

        linVel0 += delLinVel0 * accumDeltaF;
        linVel1 -= delLinVel1 * accumDeltaF;

        if (numFrictionConstrTotal != 0)
        {
            const float biasCoefficient = gFrictionHeaders[constraintBatchIndex].biasCoefficient[threadIndex];
            const float dynamicFrictionCof = gFrictionHeaders[constraintBatchIndex].dynamicFriction[threadIndex];
            const float maxFrictionImpulse = staticFrictionCof * accumulatedNormalImpulse;
            const float maxDynFrictionImpulse = dynamicFrictionCof * accumulatedNormalImpulse;

            uint broken = 0;

            const float4 frictionNormal0 = gFrictionHeaders[constraintBatchIndex].frictionNormals0[threadIndex];
            const float4 frictionNormal1 = gFrictionHeaders[constraintBatchIndex].frictionNormals1[threadIndex];
            const float3 normal0 = frictionNormal0.xyz;
            const float3 normal1 = frictionNormal1.xyz;

            const float3 delLinVel00 = normal0 * invMassA;
            const float3 delLinVel10 = normal0 * invMassB;
            const float3 delLinVel01 = normal1 * invMassA;
            const float3 delLinVel11 = normal1 * invMassB;

            const float relDelta0 = dot(relMotion, normal0);
            const float relDelta1 = dot(relMotion, normal1);

            for (uint i = 0; i < numFrictionConstr; i += 2)
            {
                const uint f0 = startFrictionIndex + i;
                const uint f1 = startFrictionIndex + i + 1;

                const float4 raXn_extraCoeff0 = gFrictions[f0].raXn_error[threadIndex];
                const float4 rbXn_targetVelW0 = gFrictions[f0].rbXn_targetVelW[threadIndex];
                const float initialError0 = raXn_extraCoeff0.w;
                const float appliedForce0 = gFrictions[f0].appliedForce[threadIndex];
                const float targetVel0 = rbXn_targetVelW0.w;
                const float f0_resp = ref0 * gFrictions[f0].resp0[threadIndex] + ref1 * gFrictions[f0].resp1[threadIndex];
                const float velMultiplier0 = (f0_resp > 0.0f) ? (p8 / f0_resp) : 0.0f;

                const float4 raXn_extraCoeff1 = gFrictions[f1].raXn_error[threadIndex];
                const float4 rbXn_targetVelW1 = gFrictions[f1].rbXn_targetVelW[threadIndex];
                const float initialError1 = raXn_extraCoeff1.w;
                const float appliedForce1 = gFrictions[f1].appliedForce[threadIndex];
                const float targetVel1 = rbXn_targetVelW1.w;
                const float f1_resp = ref0 * gFrictions[f1].resp0[threadIndex] + ref1 * gFrictions[f1].resp1[threadIndex];
                const float velMultiplier1 = (f1_resp > 0.0f) ? (p8 / f1_resp) : 0.0f;

                const float3 raXn0 = raXn_extraCoeff0.xyz;
                const float3 rbXn0 = rbXn_targetVelW0.xyz;
                const float v00 = dot(angVel0, raXn0) + dot(linVel0, normal0);
                const float v10 = dot(angVel1, rbXn0) + dot(linVel1, normal0);
                const float normalVel0 = v00 - v10;

                const float error0 = initialError0 - targetVel0 * elapsedTime +
                                     dot(raXn0, b0AngDelta) - dot(rbXn0, b1AngDelta) + relDelta0;

                const float bias0 = error0 * biasCoefficient;
                const float tmp10 = appliedForce0 - (bias0 - targetVel0) * velMultiplier0;
                const float totalImpulse0 = tmp10 - normalVel0 * velMultiplier0;

                const float3 raXn1 = raXn_extraCoeff1.xyz;
                const float3 rbXn1 = rbXn_targetVelW1.xyz;
                const float v01 = dot(angVel0, raXn1) + dot(linVel0, normal1);
                const float v11 = dot(angVel1, rbXn1) + dot(linVel1, normal1);
                const float normalVel1 = v01 - v11;

                const float error1 = initialError1 - targetVel1 * elapsedTime +
                                     dot(raXn1, b0AngDelta) - dot(rbXn1, b1AngDelta) + relDelta1;

                const float bias1 = error1 * biasCoefficient;
                const float tmp11 = appliedForce1 - (bias1 - targetVel1) * velMultiplier1;
                const float totalImpulse1 = tmp11 - normalVel1 * velMultiplier1;

                const float totalImpulse = sqrt(totalImpulse0 * totalImpulse0 + totalImpulse1 * totalImpulse1);

                const bool clampImpulse = totalImpulse > maxFrictionImpulse;
                const float ratio = clampImpulse ? min(maxDynFrictionImpulse, totalImpulse) / totalImpulse : 1.0f;

                const float newAppliedForce0 = totalImpulse0 * ratio;
                const float newAppliedForce1 = totalImpulse1 * ratio;

                const float deltaF0 = newAppliedForce0 - appliedForce0;
                const float deltaF1 = newAppliedForce1 - appliedForce1;

                linVel0 += delLinVel00 * deltaF0;
                linVel1 -= delLinVel10 * deltaF0;
                angVel0 += raXn0 * (deltaF0 * angDom0);
                angVel1 -= rbXn0 * (deltaF0 * angDom1);

                linVel0 += delLinVel01 * deltaF1;
                linVel1 -= delLinVel11 * deltaF1;
                angVel0 += raXn1 * (deltaF1 * angDom0);
                angVel1 -= rbXn1 * (deltaF1 * angDom1);

                gFrictions[f0].appliedForce[threadIndex] = newAppliedForce0;
                gFrictions[f1].appliedForce[threadIndex] = newAppliedForce1;
                broken = broken | (clampImpulse ? 1u : 0u);
            }

            if (numFrictionConstr < numFrictionConstrTotal)
            {
                // We have a torsional friction anchor, solve this...
                const float frictionScale = gFrictionHeaders[constraintBatchIndex].torsionalFrictionScale[threadIndex];

                const uint f0 = startFrictionIndex + numFrictionConstr;
                const float4 raXn_extraCoeff0 = gFrictions[f0].raXn_error[threadIndex];
                const float4 rbXn_targetVelW0 = gFrictions[f0].rbXn_targetVelW[threadIndex];
                const float appliedForce0 = gFrictions[f0].appliedForce[threadIndex];
                const float targetVel0 = rbXn_targetVelW0.w;
                const float f0_resp = ref0 * gFrictions[f0].resp0[threadIndex] + ref1 * gFrictions[f0].resp1[threadIndex];
                const float velMultiplier0 = (f0_resp > 0.0f) ? (p8 / f0_resp) : 0.0f;

                const float3 raXn0 = raXn_extraCoeff0.xyz;
                const float3 rbXn0 = rbXn_targetVelW0.xyz;

                const float v00 = dot(angVel0, raXn0);
                const float v10 = dot(angVel1, rbXn0);
                const float normalVel0 = v00 - v10;

                const float tmp10 = appliedForce0 - (-targetVel0) * velMultiplier0;
                const float totalImpulse = tmp10 - normalVel0 * velMultiplier0;

                const bool clampImpulse = abs(totalImpulse) > (maxFrictionImpulse * frictionScale);
                const float totalClamped = clamp(totalImpulse, -maxDynFrictionImpulse * frictionScale,
                                                 maxDynFrictionImpulse * frictionScale);
                const float newAppliedForce = clampImpulse ? totalClamped : totalImpulse;

                const float deltaF = newAppliedForce - appliedForce0;

                angVel0 += raXn0 * (deltaF * angDom0);
                angVel1 -= rbXn0 * (deltaF * angDom1);

                gFrictions[f0].appliedForce[threadIndex] = newAppliedForce;
                broken = broken | (clampImpulse ? 1u : 0u);
            }

            gFrictionHeaders[constraintBatchIndex].broken[threadIndex] = broken;
        }
    }
}

// mirrors solve1DBlockTGS() in solverBlockTGS.cuh
void solve1DBlockTGS(uint batchIndex, inout float3 linVel0, inout float3 angVel0,
                     inout float3 linVel1, inout float3 angVel1,
                     float3 linDelta0, float3 angDelta0, float3 linDelta1, float3 angDelta1,
                     uint threadIndex, PxgSolverTxIData iData0, PxgSolverTxIData iData1,
                     float elapsedTime, bool residualReportingEnabled, float ref0, float ref1)
{
    const uint constraintBatchIndex = gBatches[batchIndex].mConstraintBatchIndex;
    const uint startConstraintIndex = gBatches[batchIndex].startConstraintIndex;

    const float4 raInvMass0 = gJointHeaders[constraintBatchIndex].rAWorld_invMass0D0[threadIndex];
    const float4 rbInvMass1 = gJointHeaders[constraintBatchIndex].rBWorld_invMass1D1[threadIndex];

    const float invMass0 = ref0 * raInvMass0.w;
    const float invMass1 = ref1 * rbInvMass1.w;

    const float3 raPrev = raInvMass0.xyz;
    const float3 rbPrev = rbInvMass1.xyz;

    const float3 ra = quatRotate(iData0.deltaBody2World.q(), raPrev);
    const float3 rb = quatRotate(iData1.deltaBody2World.q(), rbPrev);

    const float3 raMotion = (ra + linDelta0) - raPrev;
    const float3 rbMotion = (rb + linDelta1) - rbPrev;

    const float invInertiaScale0 = ref0 * gJointHeaders[constraintBatchIndex].invInertiaScale0[threadIndex];
    const float invInertiaScale1 = ref1 * gJointHeaders[constraintBatchIndex].invInertiaScale1[threadIndex];

    const uint rowCounts_breakable_orthoAxisCount = gJointHeaders[constraintBatchIndex].rowCounts_breakable_orthoAxisCount[threadIndex];
    const uint rowCount = rowCounts_breakable_orthoAxisCount & 0xFFu; // uchar4.x

    const float4 ang0Ortho0_recipResponseW = gJointHeaders[constraintBatchIndex].angOrthoAxis0_recipResponseW0[threadIndex];
    const float4 ang0Ortho1_recipResponseW = gJointHeaders[constraintBatchIndex].angOrthoAxis0_recipResponseW1[threadIndex];
    const float4 ang0Ortho2_recipResponseW = gJointHeaders[constraintBatchIndex].angOrthoAxis0_recipResponseW2[threadIndex];

    const float4 ang1OrthoAxis0_ErrorW = gJointHeaders[constraintBatchIndex].angOrthoAxis1_ErrorW0[threadIndex];
    const float4 ang1OrthoAxis1_ErrorW = gJointHeaders[constraintBatchIndex].angOrthoAxis1_ErrorW1[threadIndex];
    const float4 ang1OrthoAxis2_ErrorW = gJointHeaders[constraintBatchIndex].angOrthoAxis1_ErrorW2[threadIndex];

    const float recipResponse0 = ang0Ortho0_recipResponseW.w;
    const float recipResponse1 = ang0Ortho1_recipResponseW.w;
    const float recipResponse2 = ang0Ortho2_recipResponseW.w;

    const float3 ang0Ortho0 = ang0Ortho0_recipResponseW.xyz;
    const float3 ang0Ortho1 = ang0Ortho1_recipResponseW.xyz;
    const float3 ang0Ortho2 = ang0Ortho2_recipResponseW.xyz;

    const float3 ang1Ortho0 = ang1OrthoAxis0_ErrorW.xyz;
    const float3 ang1Ortho1 = ang1OrthoAxis1_ErrorW.xyz;
    const float3 ang1Ortho2 = ang1OrthoAxis2_ErrorW.xyz;

    const float error0 = ang1OrthoAxis0_ErrorW.w + (dot(ang0Ortho0, angDelta0) - dot(ang1Ortho0, angDelta1));
    const float error1 = ang1OrthoAxis1_ErrorW.w + (dot(ang0Ortho1, angDelta0) - dot(ang1Ortho1, angDelta1));
    const float error2 = ang1OrthoAxis2_ErrorW.w + (dot(ang0Ortho2, angDelta0) - dot(ang1Ortho2, angDelta1));

    const PxgMat33 sqrtInvInertia0 = mat33Load(iData0.sqrtInvInertia);
    const PxgMat33 sqrtInvInertia1 = mat33Load(iData1.sqrtInvInertia);

    for (uint i = 0; i < rowCount; ++i)
    {
        const uint r = startConstraintIndex + i;

        const float4 _clinVel0_coeff0W = gJointRowsCon[r].lin0XYZ_initBiasOrCoeff0[threadIndex];
        const float4 _clinVel1_coeff1W = gJointRowsCon[r].lin1XYZ_biasScaleOrCoeff1[threadIndex];
        const float4 _cangVel0_coeff2W = gJointRowsCon[r].ang0XYZ_velMultiplierOrCoeff2[threadIndex];
        const float4 _cangVel1_coeff3W = gJointRowsCon[r].ang1XYZ_velTargetOrCoeff3[threadIndex];

        const float3 clinVel0 = _clinVel0_coeff0W.xyz;
        const float3 clinVel1 = _clinVel1_coeff1W.xyz;
        const float3 cangVel0_ = _cangVel0_coeff2W.xyz;
        const float3 cangVel1_ = _cangVel1_coeff3W.xyz;

        float initBias = _clinVel0_coeff0W.w;
        const float biasScale = _clinVel1_coeff1W.w;
        const float velMultiplier = _cangVel0_coeff2W.w;
        const float targetVel = _cangVel1_coeff3W.w;

        const float3 cangVel0 = cangVel0_ + cross(ra, clinVel0);
        const float3 cangVel1 = cangVel1_ + cross(rb, clinVel1);

        const uint flags = gJointRowsCon[r].flags[threadIndex];
        const float maxBias = gJointRowsCon[r].maxBias[threadIndex];
        const float minBias = computeMinBiasTGS(flags, maxBias);

        float3 raXnI = mat33Transform(sqrtInvInertia0, cangVel0);
        float3 rbXnI = mat33Transform(sqrtInvInertia1, cangVel1);

        if ((flags & DY_SC_FLAG_ORTHO_TARGET) != 0)
        {
            const float proj0 = (dot(raXnI, ang0Ortho0) + dot(rbXnI, ang1Ortho0)) * recipResponse0;
            const float proj1 = (dot(raXnI, ang0Ortho1) + dot(rbXnI, ang1Ortho1)) * recipResponse1;
            const float proj2 = (dot(raXnI, ang0Ortho2) + dot(rbXnI, ang1Ortho2)) * recipResponse2;

            const float3 delta0 = ang0Ortho0 * proj0 + ang0Ortho1 * proj1 + ang0Ortho2 * proj2;
            const float3 delta1 = ang1Ortho0 * proj0 + ang1Ortho1 * proj1 + ang1Ortho2 * proj2;

            raXnI -= delta0;
            rbXnI -= delta1;

            const float orthoBasisError = biasScale * (error0 * proj0 + error1 * proj1 + error2 * proj2);
            initBias -= orthoBasisError;
        }

        // If biasScale is zero, initBias must also be zero (velocity iterations).
        initBias = (biasScale == 0.0f) ? 0.0f : initBias;

        const bool isSpringConstraint = (flags & DY_SC_FLAG_SPRING) != 0;

        const float errorDelta = computeResolvedGeometricErrorTGS(raMotion, rbMotion, clinVel0, clinVel1,
                                                                  angDelta0, angDelta1, raXnI, rbXnI,
                                                                  gJointRowsCon[r].angularErrorScale[threadIndex],
                                                                  isSpringConstraint, targetVel, elapsedTime);

        const float resp0 = invMass0 * dot(clinVel0, clinVel0) + dot(raXnI, raXnI) * invInertiaScale0;
        const float resp1 = invMass1 * dot(clinVel1, clinVel1) + dot(rbXnI, rbXnI) * invInertiaScale1;

        const float response = resp0 + resp1;
        const float recipResponse = response > 0.0f ? 1.0f / response : 0.0f;

        const float vMul = isSpringConstraint ? velMultiplier : recipResponse * velMultiplier;

        const float unclampedBias = initBias + errorDelta * biasScale;
        const float bias = clamp(unclampedBias, minBias, maxBias);

        const float constant = isSpringConstraint ? bias + targetVel : recipResponse * (bias + targetVel);

        const float appliedForce = gJointRowsCon[r].appliedForce[threadIndex];
        const float maxImpulse = gJointRowsCon[r].maxImpulse[threadIndex];
        const float minImpulse = gJointRowsCon[r].minImpulse[threadIndex];

        const float v0 = dot(linVel0, clinVel0) + dot(angVel0, raXnI);
        const float v1 = dot(linVel1, clinVel1) + dot(angVel1, rbXnI);
        const float normalVel = v0 - v1;

        const float unclampedForce = appliedForce + (vMul * normalVel + constant);
        const float clampedForce = min(maxImpulse, max(minImpulse, unclampedForce));
        const float deltaF = clampedForce - appliedForce;

        gJointRowsCon[r].appliedForce[threadIndex] = clampedForce;
        if (residualReportingEnabled)
            gJointRowsCon[r].residual[threadIndex] = vMul == 0.0f ? 0.0f : deltaF / vMul;

        linVel0 = linVel0 + clinVel0 * (deltaF * invMass0);
        linVel1 = linVel1 - clinVel1 * (deltaF * invMass1);
        angVel0 = angVel0 + raXnI * deltaF * invInertiaScale0;
        angVel1 = angVel1 - rbXnI * deltaF * invInertiaScale1;
    }
}

// mirrors solveBlockPartitionTGSInternal() in solverMultiBlockTGS.cu
// (rigid path of solveBlockUnified); one 32-thread workgroup per batch
[shader("compute")]
[numthreads(32, 1, 1)]
void solveBlockPartitionTGS(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint threadIndexInWarp = groupThreadId.x;
    const uint k = gParams.startIndex + groupId.x;

    if (k >= gParams.endIndex)
        return;

    const uint descStride = gBatches[k].descStride_constraintType & 0xFFFFu;
    const uint constraintType = gBatches[k].descStride_constraintType >> 16;

    const uint readIndex = k * 192 + threadIndexInWarp;

    if (threadIndexInWarp >= descStride)
        return;

    // zero-init: only read on the joint path, where they are always loaded
    PxgSolverTxIData iData0 = {};
    PxgSolverTxIData iData1 = {};
    if (constraintType != PXG_CONSTRAINT_CONTACT)
    {
        // CUDA stages these through warp shuffles (loadTxInertia) purely for
        // memory coalescing; direct loads are semantically identical.
        iData0 = gTxIData[gBatches[k].bodyAIndex[threadIndexInWarp]];
        iData1 = gTxIData[gBatches[k].bodyBIndex[threadIndexInWarp]];
    }

    float4 vec0 = gSolverBodyVelPool[readIndex];
    float4 vec1 = gSolverBodyVelPool[readIndex + 32];
    float4 vec2 = gSolverBodyVelPool[readIndex + 64];
    float4 vec3 = gSolverBodyVelPool[readIndex + 96];
    float4 vec4 = gSolverBodyVelPool[readIndex + 128];
    float4 vec5 = gSolverBodyVelPool[readIndex + 160];

    float3 linVel0 = vec0.xyz;
    float3 angVel0 = float3(vec0.w, vec1.x, vec1.y);
    float3 linDelta0 = float3(vec1.z, vec1.w, vec2.x);
    float3 angDelta0 = float3(vec2.y, vec2.z, vec2.w);

    float3 linVel1 = vec3.xyz;
    float3 angVel1 = float3(vec3.w, vec4.x, vec4.y);
    float3 linDelta1 = float3(vec4.z, vec4.w, vec5.x);
    float3 angDelta1 = float3(vec5.y, vec5.z, vec5.w);

    // Reference counts for mass splitting.
    float curRef0 = 1.0f;
    float curRef1 = 1.0f;

    if (gBatches[k].bodyAIndex[threadIndexInWarp] >= gParams.bodyOffset)
        curRef0 = float(countActiveSlabs(gBatches[k].bodyAIndex[threadIndexInWarp], gParams.numSlabs,
                                         gParams.numTotalBodies));
    if (gBatches[k].bodyBIndex[threadIndexInWarp] >= gParams.bodyOffset)
        curRef1 = float(countActiveSlabs(gBatches[k].bodyBIndex[threadIndexInWarp], gParams.numSlabs,
                                         gParams.numTotalBodies));

    if (constraintType == PXG_CONSTRAINT_CONTACT)
        solveContactBlockTGS(k, linVel0, angVel0, linVel1, angVel1, linDelta0, angDelta0, linDelta1,
                             angDelta1, threadIndexInWarp, gParams.elapsedTime, gParams.minPen, curRef0,
                             curRef1);
    else
        solve1DBlockTGS(k, linVel0, angVel0, linVel1, angVel1, linDelta0, angDelta0, linDelta1, angDelta1,
                        threadIndexInWarp, iData0, iData1, gParams.elapsedTime,
                        gParams.residualReportingEnabled != 0, curRef0, curRef1);

    vec0 = float4(linVel0, angVel0.x);
    vec1 = float4(angVel0.y, angVel0.z, linDelta0.x, linDelta0.y);
    vec2 = float4(linDelta0.z, angDelta0.x, angDelta0.y, angDelta0.z);

    vec3 = float4(linVel1, angVel1.x);
    vec4 = float4(angVel1.y, angVel1.z, linDelta1.x, linDelta1.y);
    vec5 = float4(linDelta1.z, angDelta1.x, angDelta1.y, angDelta1.z);

    const uint outputA = gBatches[k].remappedBodyAIndex[threadIndexInWarp];
    const uint outputB = gBatches[k].remappedBodyBIndex[threadIndexInWarp];

    gSolverBodyVelPool[outputA] = vec0;
    gSolverBodyVelPool[outputA + 32] = vec1;
    gSolverBodyVelPool[outputA + 64] = vec2;

    gSolverBodyVelPool[outputB] = vec3;
    gSolverBodyVelPool[outputB + 32] = vec4;
    gSolverBodyVelPool[outputB + 64] = vec5;
}

// ---------------------------------------------------------------------------
// conclude / writeback
// ---------------------------------------------------------------------------

// mirrors concludeBlocksTGS() in solverMultiBlockTGS.cu.  The contact branch
// (concludeContactBlockTGS) is compiled out ("#if 0") in the CUDA original —
// TGS contacts keep their bias in the velocity iterations — so only the joint
// branch does any work.
[shader("compute")]
[numthreads(32, 1, 1)]
void concludeBlocksTGS(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint threadIndex = groupThreadId.x;
    const uint k = gParams.startIndex + groupId.x;

    if (k >= gParams.endIndex)
        return;

    const uint descStride = gBatches[k].descStride_constraintType & 0xFFFFu;
    const uint constraintType = gBatches[k].descStride_constraintType >> 16;
    if (threadIndex >= descStride || constraintType == PXG_CONSTRAINT_CONTACT)
        return;

    // conclude1DBlockTGS()
    const uint batchIndex = gBatches[k].mConstraintBatchIndex;
    const uint startConstraintIndex = gBatches[k].startConstraintIndex;

    const uint rowCount = gJointHeaders[batchIndex].rowCounts_breakable_orthoAxisCount[threadIndex] & 0xFFu;

    for (uint i = 0; i < rowCount; i++)
    {
        const uint r = startConstraintIndex + i;
        const uint flags = gJointRowsCon[r].flags[threadIndex];

        // biasScale, initBias, velTarget and velMultiplier are repacked with
        // other coefficients; zeroing biasScale requires initBias == 0 too,
        // which the solve enforces (initBias = biasScale == 0 ? 0 : initBias).
        if ((flags & DY_SC_FLAG_KEEP_BIAS) == 0)
        {
            float4 v = gJointRowsCon[r].lin1XYZ_biasScaleOrCoeff1[threadIndex];
            v.w = 0.0f;
            gJointRowsCon[r].lin1XYZ_biasScaleOrCoeff1[threadIndex] = v;
        }
        if ((flags & DY_SC_FLAG_SPRING) != 0)
        {
            float4 v0 = gJointRowsCon[r].lin0XYZ_initBiasOrCoeff0[threadIndex];
            float4 v1 = gJointRowsCon[r].lin1XYZ_biasScaleOrCoeff1[threadIndex];
            float4 v2 = gJointRowsCon[r].ang0XYZ_velMultiplierOrCoeff2[threadIndex];
            float4 v3 = gJointRowsCon[r].ang1XYZ_velTargetOrCoeff3[threadIndex];
            v0.w = 0.0f;
            v1.w = 0.0f;
            v2.w = 0.0f;
            v3.w = 0.0f;
            gJointRowsCon[r].lin0XYZ_initBiasOrCoeff0[threadIndex] = v0;
            gJointRowsCon[r].lin1XYZ_biasScaleOrCoeff1[threadIndex] = v1;
            gJointRowsCon[r].ang0XYZ_velMultiplierOrCoeff2[threadIndex] = v2;
            gJointRowsCon[r].ang1XYZ_velTargetOrCoeff3[threadIndex] = v3;
        }
    }
}

// mirrors writeBackContactBlockFriction() in solverBlockCommon.cuh (TGS
// header/friction types)
void writeBackContactBlockFrictionTGS(uint threadIndex, uint numFrictionConstr, uint batchIndex,
                                      uint startFrictionIndex)
{
    const uint patchIndex = gBlockFrictionPatches[batchIndex].patchIndex[threadIndex];
    if (patchIndex != 0xFFFFFFFFu)
    {
        const float4 axis0 = gFrictionHeaders[batchIndex].frictionNormals0[threadIndex];
        const float4 axis1 = gFrictionHeaders[batchIndex].frictionNormals1[threadIndex];

        gFrictionPatches[patchIndex].anchors = numFrictionConstr / 2;

        if (numFrictionConstr >= 2)
        {
            const float4 anchor = gBlockFrictionPatches[batchIndex].anchorPoints0[threadIndex];
            const float impulse0 = gFrictions[startFrictionIndex + 0].appliedForce[threadIndex];
            const float impulse1 = gFrictions[startFrictionIndex + 1].appliedForce[threadIndex];
            const float3 imp = axis0.xyz * impulse0 + axis1.xyz * impulse1;
            gFrictionPatches[patchIndex].p0x = anchor.x;
            gFrictionPatches[patchIndex].p0y = anchor.y;
            gFrictionPatches[patchIndex].p0z = anchor.z;
            gFrictionPatches[patchIndex].i0x = imp.x;
            gFrictionPatches[patchIndex].i0y = imp.y;
            gFrictionPatches[patchIndex].i0z = imp.z;
        }
        if (numFrictionConstr >= 4)
        {
            const float4 anchor = gBlockFrictionPatches[batchIndex].anchorPoints1[threadIndex];
            const float impulse0 = gFrictions[startFrictionIndex + 2].appliedForce[threadIndex];
            const float impulse1 = gFrictions[startFrictionIndex + 3].appliedForce[threadIndex];
            const float3 imp = axis0.xyz * impulse0 + axis1.xyz * impulse1;
            gFrictionPatches[patchIndex].p1x = anchor.x;
            gFrictionPatches[patchIndex].p1y = anchor.y;
            gFrictionPatches[patchIndex].p1z = anchor.z;
            gFrictionPatches[patchIndex].i1x = imp.x;
            gFrictionPatches[patchIndex].i1y = imp.y;
            gFrictionPatches[patchIndex].i1z = imp.z;
        }
    }
}

// 64-bit compare of PxNodeIndex values held as uint2
bool nodeIndexLess(uint2 a, uint2 b)
{
    return a.y < b.y || (a.y == b.y && a.x < b.x);
}

// mirrors writebackBlocksTGS() in solverMultiBlockTGS.cu (rigid path).  Like
// the PGS port, threshold elements are appended per-thread instead of through
// the CUDA warp-aggregated shared-memory staging — the stream contents are
// identical, ordering within the stream is nondeterministic in both versions.
[shader("compute")]
[numthreads(32, 1, 1)]
void writebackBlocksTGS(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint threadIndex = groupThreadId.x;
    const uint k = gParams.startIndex + groupId.x;

    if (k >= gParams.endIndex)
        return;

    const uint descStride = gBatches[k].descStride_constraintType & 0xFFFFu;
    const uint constraintType = gBatches[k].descStride_constraintType >> 16;
    if (threadIndex >= descStride)
        return;

    const uint batchIndex = gBatches[k].mConstraintBatchIndex;
    const uint startConstraintIndex = gBatches[k].startConstraintIndex;
    const uint startFrictionIndex = gBatches[k].startFrictionIndex;

    if (constraintType == PXG_CONSTRAINT_CONTACT)
    {
        // writeBackContactBlockTGS()
        const uint bodyAIndex = gBatches[k].bodyAIndex[threadIndex];
        const uint bodyBIndex = gBatches[k].bodyBIndex[threadIndex];

        float normalForce = 0.0f;

        const uint forceWritebackOffset = gContactHeaders[batchIndex].forceWritebackOffset[threadIndex];
        const bool forceThreshold =
            (gContactHeaders[batchIndex].flags[threadIndex] & PXG_CONTACT_FLAG_HAS_FORCE_THRESHOLDS) != 0;

        const uint numFrictionConstr = gFrictionHeaders[batchIndex].numFrictionConstr[threadIndex];
        const uint numNormalConstr = gContactHeaders[batchIndex].numNormalConstr[threadIndex];

        if (forceWritebackOffset != 0xFFFFFFFFu)
        {
            for (uint i = 0; i < numNormalConstr; i++)
            {
                const float appliedForce = gContactPoints[startConstraintIndex + i].appliedForce[threadIndex];
                gForceBuffer[forceWritebackOffset + i] = appliedForce;
                normalForce += appliedForce;
            }
        }

        writeBackContactBlockFrictionTGS(threadIndex, numFrictionConstr, batchIndex, startFrictionIndex);

        if (numFrictionConstr != 0 && gFrictionHeaders[batchIndex].broken[threadIndex] != 0)
            gBlockFrictionPatches[batchIndex].broken[threadIndex] = 1;

        const float reportThreshold0 = gSolverBodyData[bodyAIndex].reportThreshold;
        const float reportThreshold1 = gSolverBodyData[bodyBIndex].reportThreshold;

        if (forceThreshold && normalForce != 0.0f &&
            (reportThreshold0 < PX_MAX_F32 || reportThreshold1 < PX_MAX_F32))
        {
            ThresholdStreamElement elt;
            elt.normalForce = normalForce;
            elt.threshold = min(reportThreshold0, reportThreshold1);
            uint2 nodeA = gSolverBodyData[bodyAIndex].islandNodeIndex;
            uint2 nodeB = gSolverBodyData[bodyBIndex].islandNodeIndex;
            // PxOrder(): swap so nodeIndexA < nodeIndexB
            if (nodeIndexLess(nodeB, nodeA))
            {
                const uint2 t = nodeA;
                nodeA = nodeB;
                nodeB = t;
            }
            elt.nodeIndexA = nodeA;
            elt.nodeIndexB = nodeB;
            elt.shapeInteraction = gBatches[k].shapeInteraction[threadIndex];
            elt.accumulatedForce = 0.0f;
            elt.pad = 0;

            uint index;
            InterlockedAdd(gThresholdStreamIndex[0], 1, index);
            gThresholdStream[index] = elt;
        }
    }
    else
    {
        // writeBack1DBlockTGS()
        const uint forceWritebackOffset = gJointHeaders[batchIndex].writeBackOffset[threadIndex];
        const uint rowCounts_breakable = gJointHeaders[batchIndex].rowCounts_breakable_orthoAxisCount[threadIndex];
        const uint numRows = rowCounts_breakable & 0xFFu;        // uchar4.x
        const uint breakable = (rowCounts_breakable >> 8) & 0xFFu; // uchar4.y

        if (forceWritebackOffset != 0xFFFFFFFFu)
        {
            float3 linVel = float3(0.0f);
            float3 angVel = float3(0.0f);
            float constraintErrorSq = 0.0f;

            for (uint i = 0; i < numRows; ++i)
            {
                const uint r = startConstraintIndex + i;
                if ((gJointRowsCon[r].flags[threadIndex] & DY_SC_FLAG_OUTPUT_FORCE) != 0)
                {
                    const float4 lin0XYZ = gJointRowsCon[r].lin0XYZ_initBiasOrCoeff0[threadIndex];
                    const float4 ang0WriteBack = gJointRowsCon[r].ang0XYZ_velMultiplierOrCoeff2[threadIndex];
                    const float appliedForce = gJointRowsCon[r].appliedForce[threadIndex];
                    linVel += lin0XYZ.xyz * appliedForce;
                    angVel += ang0WriteBack.xyz * appliedForce;
                }

                const float err = gJointRowsCon[r].residual[threadIndex];
                constraintErrorSq += err * err;
            }

            const float4 body0WorldOffset_invMass0 = gJointHeaders[batchIndex].rAWorld_invMass0D0[threadIndex];
            angVel -= cross(body0WorldOffset_invMass0.xyz, linVel);

            const uint broken = breakable != 0
                ? ((length(linVel) > gJointHeaders[batchIndex].linBreakImpulse[threadIndex] ||
                    length(angVel) > gJointHeaders[batchIndex].angBreakImpulse[threadIndex]) ? 1u : 0u)
                : 0u;
            gConstraintWriteBack[forceWritebackOffset].angularImpulse_residual =
                float4(angVel, constraintErrorSq);
            gConstraintWriteBack[forceWritebackOffset].linearImpulse_broken =
                float4(linVel, broken != 0 ? -0.0f : 0.0f);
        }
    }
}

// ---------------------------------------------------------------------------
// P5 Step 4 BDA engine-launch ABI (writeBackBodiesTGS, computeAverage- and
// propagateAverageSolverBodyVelocityTGS): these entries take the ENGINE's
// CUDA parameter packs and read every count/offset/pool pointer on-device
// through the byte-exact pxgSlangSolverDesc mirrors (testSolverDescLayout).
// The solve/conclude/writeback-block entries above still use the harness
// SolverTGSParams push constants + bindings.
// ---------------------------------------------------------------------------

// (const PxgSolverCoreDesc*, const PxgSolverSharedDesc<IterativeSolveDataTGS>*)
struct SolverCoreSharedAbiParams
{
    uint64_t solverCoreDesc; // @0
    uint64_t sharedDesc;     // @8
}                            // = 16
[[vk::push_constant]]
ConstantBuffer<SolverCoreSharedAbiParams> gAvgAbi;

// (descs..., const PxU32 islandIndex): {8,8,4} -> blob rounded to 24
struct WriteBackBodiesAbiParams
{
    uint64_t solverCoreDesc; // @0
    uint64_t sharedDesc;     // @8
    uint islandIndex;        // @16
    uint abiPad0;            // @20 (blob tail padding, never read)
}                            // = 24
[[vk::push_constant]]
ConstantBuffer<WriteBackBodiesAbiParams> gWriteBackAbi;

// (descs..., bool isVelocityIteration, bool isLastPosIteration, PxReal
// biasCoefficient): the two C++ bools land in bytes 16 and 17 of the blob
// (bytes 18..19 are padding), the float at 20 -> 24 bytes total.
struct PropagateAbiParams
{
    uint64_t solverCoreDesc; // @0
    uint64_t sharedDesc;     // @8
    uint boolsWord;          // @16: byte0 = isVelocityIteration, byte1 = isLastPosIteration
    float biasCoefficient;   // @20 (currently unused by the kernel, like the .cu)
}                            // = 24
[[vk::push_constant]]
ConstantBuffer<PropagateAbiParams> gPropagateAbi;

// countActiveSlabs() over a BDA pointer instead of the bound buffer
uint countActiveSlabsBda(uint index, uint numSlabs, uint numSolverBodies, Ptr<uint> encodedReferenceCount)
{
    uint referenceCount = 0;
    const uint num32Slabs = (numSlabs + 31) / 32;
    for (uint i = 0; i < num32Slabs; ++i)
        referenceCount += countbits(encodedReferenceCount[index + i * numSolverBodies]);
    return max(1u, referenceCount);
}

// mirrors writeBackBodiesTGS() in solverMultiBlockTGS.cu: turns the
// accumulated linear/angular deltas into motion velocities (delta * invDt).
// numthreads matches the CUDA blockDim (PxgKernelBlockDim::WRITE_BACK_BODIES
// = 256) so the engine's grid arithmetic maps 1:1.
[shader("compute")]
[numthreads(256, 1, 1)]
void writeBackBodiesTGS(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    PxgSolverCoreDescGpu desc = (Ptr<PxgSolverCoreDescGpu>(gWriteBackAbi.solverCoreDesc))[0];
    PxgSolverSharedDescTGSGpu shared = (Ptr<PxgSolverSharedDescTGSGpu>(gWriteBackAbi.sharedDesc))[0];
    PxgIslandContextGpu island =
        (Ptr<PxgIslandContextGpu>(desc.islandContextPool))[gWriteBackAbi.islandIndex];

    Ptr<float4> bodyVelocities = Ptr<float4>(shared.iter_solverBodyVelPool);
    Ptr<float4> motionVelocities = Ptr<float4>(desc.motionVelocityArray);
    const uint bodyOffset = island.mBodyStartIndex;
    const uint numDynamicBodies = island.mBodyCount;

    const uint a = dispatchThreadId.x;
    if (a >= numDynamicBodies)
        return;

    const uint totalBodiesIncKinematics = numDynamicBodies + bodyOffset;
    const uint outputOffset = desc.accumulatedBodyDeltaVOffset;
    const float invDt = shared.invDtF32;
    const uint base = bodyOffset + outputOffset + a + totalBodiesIncKinematics;

    const float4 vel1 = bodyVelocities[base];
    const float4 vel2 = bodyVelocities[base + totalBodiesIncKinematics];

    motionVelocities[a + bodyOffset] = float4(vel1.z, vel1.w, vel2.x, 0.0f) * invDt;
    motionVelocities[a + bodyOffset + totalBodiesIncKinematics] =
        float4(vel2.y, vel2.z, vel2.w, 0.0f) * invDt;
}

// ---------------------------------------------------------------------------
// multi-slab velocity averaging
// ---------------------------------------------------------------------------

uint nextPowerOfTwo(uint x)
{
    x -= 1;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

// mirrors ComputeAverageBodyOutputIndex() in solverMultiBlockTGS.cu (three
// float4 per body slot instead of the two of the PGS variant)
uint computeAverageBodyOutputIndex(uint bodyIndex)
{
    return 3 * (bodyIndex & ~31u) + (bodyIndex & 31u);
}

groupshared float4 gsVel0[32];
groupshared float2 gsVel1[32];
groupshared uint gsHasRef[32];

// mirrors computeAverageSolverBodyVelocityTGS() in solverMultiBlockTGS.cu:
// averages the per-slab deltas of the 6-component velocity (linVel.xyz +
// angVel.x in vel0, angVel.yz in vel1.xy) and resets the slab reference
// counts; the accumulated position/angle deltas in vel1.zw are untouched.
// The CUDA sub-warp __shfl_xor reductions become groupshared reductions over
// the same power-of-two lane groups; one workgroup covers one warp's bodies
// (dispatch = engine block count * 8 warps/block).  Engine ABI: everything is
// read from the two descs on-device.
[shader("compute")]
[numthreads(32, 1, 1)]
void computeAverageSolverBodyVelocityTGS(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    PxgSolverCoreDescGpu desc = (Ptr<PxgSolverCoreDescGpu>(gAvgAbi.solverCoreDesc))[0];
    PxgSolverSharedDescTGSGpu shared = (Ptr<PxgSolverSharedDescTGSGpu>(gAvgAbi.sharedDesc))[0];
    PxgIslandContextGpu island = (Ptr<PxgIslandContextGpu>(desc.islandContextPool))[0];

    Ptr<uint> solverReferences = Ptr<uint>(desc.solverBodyReferences); // PxgSolverReferences
    Ptr<float4> bodyVelocities = Ptr<float4>(shared.iter_solverBodyVelPool);
    Ptr<uint> encodedReferenceCount = Ptr<uint>(shared.iter_solverEncodedReferenceCount);

    const uint numSlabs = desc.numSlabs;
    const uint totalBodyCount = island.mBodyCount; // nbBodies minus offset!
    const uint bodyOffset = island.mBodyStartIndex;
    const uint deltaVOffset = desc.accumulatedBodyDeltaVOffset;
    const uint totalBodiesIncKinematics = totalBodyCount + bodyOffset;
    const uint averageOutputOffsets = (desc.numArticBatches + desc.numBatches) * 32u * 3u * 2u;

    const uint pow2Slabs = (numSlabs & (numSlabs - 1)) == 0 ? numSlabs : nextPowerOfTwo(numSlabs);
    const uint numThreadsPerBody = min(pow2Slabs, 32u);
    const uint numBodiesPerWarp = 32 / numThreadsPerBody;

    const uint warpBodyStartIndex = groupId.x * numBodiesPerWarp;
    const uint threadIdInWorkUnit = groupThreadId.x & (numThreadsPerBody - 1);
    const uint bodyInWarp = groupThreadId.x / numThreadsPerBody;
    const uint bodyId = warpBodyStartIndex + bodyInWarp;

    float4 curVel0 = float4(0.0f);
    float2 curVel1 = float2(0.0f);
    bool hasRef = false;
    float4 vel0 = float4(0.0f);
    float4 vel1 = float4(0.0f);

    if (bodyId < totalBodyCount)
    {
        vel0 = bodyVelocities[bodyOffset + deltaVOffset + bodyId];
        vel1 = bodyVelocities[bodyOffset + deltaVOffset + bodyId + totalBodiesIncKinematics];

        for (uint b = threadIdInWorkUnit; b < numSlabs; b += numThreadsPerBody)
        {
            const uint bodyIndex = bodyId * numSlabs + b;
            const uint remappedBodyIndex = solverReferences[bodyIndex];

            if (remappedBodyIndex != 0xFFFFFFFFu)
            {
                const uint velIndex = averageOutputOffsets + computeAverageBodyOutputIndex(bodyIndex);

                const float4 delta0 = bodyVelocities[velIndex] - vel0;
                const float2 delta1 = bodyVelocities[velIndex + 32].xy - vel1.xy;

                curVel0 += delta0;
                curVel1 += delta1;
                hasRef = true;
            }
        }
    }

    // sub-group reduction over the numThreadsPerBody lanes of each body
    // (replaces the CUDA __shfl_xor_sync / __ballot_sync pattern)
    gsVel0[groupThreadId.x] = curVel0;
    gsVel1[groupThreadId.x] = curVel1;
    gsHasRef[groupThreadId.x] = hasRef ? 1u : 0u;
    GroupMemoryBarrierWithGroupSync();

    for (uint radius = numThreadsPerBody >> 1; radius > 0; radius >>= 1)
    {
        const uint partner = groupThreadId.x ^ radius;
        const float4 v0Other = gsVel0[partner];
        const float2 v1Other = gsVel1[partner];
        const uint rOther = gsHasRef[partner];
        GroupMemoryBarrierWithGroupSync();
        gsVel0[groupThreadId.x] += v0Other;
        gsVel1[groupThreadId.x] += v1Other;
        gsHasRef[groupThreadId.x] |= rOther;
        GroupMemoryBarrierWithGroupSync();
    }

    if (bodyId < totalBodyCount && gsHasRef[groupThreadId.x] != 0 && threadIdInWorkUnit == 0)
    {
        curVel0 = gsVel0[groupThreadId.x];
        curVel1 = gsVel1[groupThreadId.x];

        const uint numTotalBodies = bodyOffset + totalBodyCount + island.mArticulationCount;
        const uint referenceCount =
            countActiveSlabsBda(bodyOffset + bodyId, numSlabs, numTotalBodies, encodedReferenceCount);
        const float recipRefs = 1.0f / float(referenceCount);

        curVel0 *= recipRefs;
        curVel1 *= recipRefs;

        // Resetting rigid body reference count (resetSlabCount)
        const uint num32Slabs = (numSlabs + 31) / 32;
        for (uint i = 0; i < num32Slabs; ++i)
            encodedReferenceCount[bodyOffset + bodyId + i * numTotalBodies] = 0u;

        bodyVelocities[bodyOffset + deltaVOffset + bodyId] = vel0 + curVel0;
        bodyVelocities[bodyOffset + deltaVOffset + bodyId + totalBodiesIncKinematics] =
            float4(vel1.xy + curVel1, vel1.z, vel1.w);
    }
}

// mirrors propagateAverageSolverBodyVelocityTGS() in solverMultiBlockTGS.cu:
// copies each body's averaged 3-float4 state back into every slab slot that
// references it and — on position iterations — advances the accumulated
// position/angle deltas by stepDt and integrates the per-body delta pose
// (PxgSolverTxIData.deltaBody2World) with the closed-form quaternion
// integrator.  The CUDA shared-memory staging of the txInertia data is a
// bandwidth optimization only (each body is read by its own work unit and
// written by its lane 0) and becomes direct buffer access here.
// Engine ABI: (descs..., bool isVelocityIteration, bool isLastPosIteration,
// PxReal biasCoefficient) — the latter two ride in the pack but are unused by
// the kernel body (their uses are commented out in the .cu too).
[shader("compute")]
[numthreads(32, 1, 1)]
void propagateAverageSolverBodyVelocityTGS(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    PxgSolverCoreDescGpu desc = (Ptr<PxgSolverCoreDescGpu>(gPropagateAbi.solverCoreDesc))[0];
    PxgSolverSharedDescTGSGpu shared = (Ptr<PxgSolverSharedDescTGSGpu>(gPropagateAbi.sharedDesc))[0];
    PxgIslandContextGpu island = (Ptr<PxgIslandContextGpu>(desc.islandContextPool))[0];

    Ptr<uint> solverReferences = Ptr<uint>(desc.solverBodyReferences); // PxgSolverReferences
    Ptr<float4> bodyVelocities = Ptr<float4>(shared.iter_solverBodyVelPool);
    Ptr<PxgSolverTxIData> gIData = Ptr<PxgSolverTxIData>(desc.solverBodyTxIDataPool);
    Ptr<PxgSolverBodyData> solverBodyDataPool = Ptr<PxgSolverBodyData>(desc.solverBodyDataPool);
    Ptr<PxgBodySim> bodySimPool = Ptr<PxgBodySim>(desc.mBodySimBufferDeviceData);

    const uint numSlabs = desc.numSlabs;
    const uint bodyOffset = island.mBodyStartIndex;
    const uint deltaVOffset = desc.accumulatedBodyDeltaVOffset;
    const uint totalBodiesIncKinematics = island.mBodyCount + bodyOffset;
    const bool isVelocityIteration = (gPropagateAbi.boolsWord & 0xffu) != 0;
    const float stepDt = isVelocityIteration ? 0.0f : shared.stepDt;

    const uint pow2Slabs = (numSlabs & (numSlabs - 1)) == 0 ? numSlabs : nextPowerOfTwo(numSlabs);
    const uint numThreadsPerBody = min(pow2Slabs, 32u);
    const uint numBodiesPerWarp = 32 / numThreadsPerBody;

    const uint warpBodyStartIndex = groupId.x * numBodiesPerWarp;
    const uint threadIdInWorkUnit = groupThreadId.x & (numThreadsPerBody - 1);
    const uint bodyInWarp = groupThreadId.x / numThreadsPerBody;
    const uint bodyId = warpBodyStartIndex + bodyInWarp;

    const bool active = bodyId < totalBodiesIncKinematics;

    float4 curVel0 = float4(0.0f);
    float4 curVel1 = float4(0.0f);
    float4 curVel2 = float4(0.0f);
    if (active)
    {
        curVel0 = bodyVelocities[deltaVOffset + bodyId];
        curVel1 = bodyVelocities[deltaVOffset + bodyId + totalBodiesIncKinematics];
        curVel2 = bodyVelocities[deltaVOffset + bodyId + 2 * totalBodiesIncKinematics];
    }

    // Every lane of a work unit reads its body's accumulation slots above and
    // lane 0 overwrites them below.  CUDA warps execute this warp-synchronously;
    // portable workgroups need an explicit barrier between the two phases.
    GroupMemoryBarrierWithGroupSync();

    if (!active)
        return;

    if (!isVelocityIteration && bodyId > 0)
    {
        const uint nodeIndex = nodeIndexIndex(solverBodyDataPool[bodyId].islandNodeIndex);
        const uint lockFlags = bodySimPool[nodeIndex].lockFlags_disableGravity & 0xFFFFu;

        float3 linVel = curVel0.xyz;
        const float3 angVel = float3(curVel0.w, curVel1.x, curVel1.y);

        float3 angularMotionVel = angVel;
        if (bodyId >= bodyOffset)
            angularMotionVel = mat33Transform(mat33Load(gIData[bodyId].sqrtInvInertia), angVel);

        if ((lockFlags & PXG_LOCK_ANGULAR_X) != 0)
            angularMotionVel.x = 0.0f;
        if ((lockFlags & PXG_LOCK_ANGULAR_Y) != 0)
            angularMotionVel.y = 0.0f;
        if ((lockFlags & PXG_LOCK_ANGULAR_Z) != 0)
            angularMotionVel.z = 0.0f;
        if ((lockFlags & PXG_LOCK_LINEAR_X) != 0)
            linVel.x = 0.0f;
        if ((lockFlags & PXG_LOCK_LINEAR_Y) != 0)
            linVel.y = 0.0f;
        if ((lockFlags & PXG_LOCK_LINEAR_Z) != 0)
            linVel.z = 0.0f;

        const float3 deltaPThisStep = linVel * stepDt;

        curVel1.z += deltaPThisStep.x;
        curVel1.w += deltaPThisStep.y;
        curVel2.x += deltaPThisStep.z;
        curVel2.y += angVel.x * stepDt;
        curVel2.z += angVel.y * stepDt;
        curVel2.w += angVel.z * stepDt;

        if (threadIdInWorkUnit == 0)
        {
            float4 q = gIData[bodyId].deltaBody2World.q();
            float3 p = gIData[bodyId].deltaBody2World.p();

            const float w2 = dot(angularMotionVel, angularMotionVel);

            // Integrate the rotation using closed form quaternion integrator
            if (w2 != 0.0f)
            {
                const float w = sqrt(w2);
                const float v = w * 0.5f * stepDt;
                float s, c;
                sincos(v, s, c);
                s /= w;

                const float3 pqr = angularMotionVel * s;
                const float4 quatVel = float4(pqr, 0.0f);
                const float4 result = quatMul(quatVel, q) + q * c;

                q = normalize(result);
            }

            p += deltaPThisStep;
            gIData[bodyId].deltaBody2World = makeTransform28(q, p);
        }
    }

    if (bodyId >= bodyOffset)
    {
        // copy the final velocity to the corresponding body velocity slots
        for (uint b = threadIdInWorkUnit; b < numSlabs; b += numThreadsPerBody)
        {
            const uint bodyIndex = (bodyId - bodyOffset) * numSlabs + b;
            const uint remappedBodyIndex = solverReferences[bodyIndex];
            if (remappedBodyIndex != 0xFFFFFFFFu)
            {
                bodyVelocities[remappedBodyIndex] = curVel0;
                bodyVelocities[remappedBodyIndex + 32] = curVel1;
                bodyVelocities[remappedBodyIndex + 64] = curVel2;
            }
        }
    }

    if (!isVelocityIteration && threadIdInWorkUnit == 0)
    {
        // update the accumulated deltaP terms (needed for soft bodies /
        // particles and the final writeBackBodiesTGS)
        bodyVelocities[deltaVOffset + bodyId + totalBodiesIncKinematics] = curVel1;
        bodyVelocities[deltaVOffset + bodyId + 2 * totalBodiesIncKinematics] = curVel2;
    }
}
