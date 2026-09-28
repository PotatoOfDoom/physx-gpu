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
//   - the contact-error accumulator statistics are not ported on the
//     bound-buffer path (the per-partition CUDA launch passes error = NULL);
//     per-row joint residual writeback is.  The BDA solveWholeIslandTGS entry
//     at the end of this file DOES port them (its CUDA original accumulates
//     into solverDesc->contactErrorAccumulator itself).

import pxgSlangCommon;
import pxgSlangSolverDesc;
import pxgSlangSolverBlocks;
import npArtiSolvePartition;   // P8.104: the articulation branch of solveBlockUnified

static const uint DY_SC_FLAG_OUTPUT_FORCE = 1u << 1;
static const uint DY_SC_FLAG_KEEP_BIAS = 1u << 2;
static const uint DY_SC_FLAG_ORTHO_TARGET = 1u << 4;
static const uint DY_SC_FLAG_SPRING = 1u << 5;
static const uint DY_SC_FLAG_INEQUALITY = 1u << 6;
static const uint PXG_CONTACT_FLAG_HAS_FORCE_THRESHOLDS = 1u << 0;
static const uint PXG_CONTACT_FLAG_COMPLIANT_ACCELERATION_SPRING = 1u << 3;
static const uint PXG_CONSTRAINT_CONTACT = 0;
// PxgSolverConstraintDesc::PxgConstraintType — the encoding the engine's block
// prep actually writes into descStride_constraintType>>16 (Slang path:
// npRigidSumBatches.slang:356 writes eCONTACT=1; CUDA path:
// constraintBlockPrePrep.cu:1898).  eCONSTRAINT_1D=0, eCONTACT=1,
// eARTICULATION_CONSTRAINT_1D=2, eARTICULATION_CONTACT=3.  concludeBlocksTGS
// (engine ABI) discriminates contact vs joint batches with these, matching
// concludeBlocksTGS() in solverMultiBlockTGS.cu (contact = no-op).
static const uint PXG_SOLVER_CONSTRAINT_TYPE_CONTACT = 1u;
static const uint PXG_SOLVER_CONSTRAINT_TYPE_ARTI_CONTACT = 3u;
// A batch is a contact batch iff its type is eCONTACT (1) or eARTICULATION_CONTACT
// (3); eCONSTRAINT_1D (0) / eARTICULATION_CONSTRAINT_1D (2) are 1D joints.  This
// is the encoding the engine's block prep writes — the engine-ABI kernels MUST
// use it (the legacy PXG_CONSTRAINT_CONTACT=0 convention above is inverted vs the
// engine and only survives in self-consistent harness fixtures).
bool pxgIsContactType(uint t)
{
    return t == PXG_SOLVER_CONSTRAINT_TYPE_CONTACT || t == PXG_SOLVER_CONSTRAINT_TYPE_ARTI_CONTACT;
}
static const float PX_MAX_F32 = 3.4028234663852885981170418348452e+38f;

// ---------------------------------------------------------------------------
// TGS block SoA structures (byte-identical mirrors)
// ---------------------------------------------------------------------------

// The TGS block SoA mirrors (batch, contact/friction headers and rows, the
// 1D header/rows, and the articulation response block) moved to the
// importable pxgSlangSolverBlocks module in P8.99a.

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
                // P9.86: PxClamp(v, lo, hi) = PxMin(hi, PxMax(lo, v)) = selectMin(hi, selectMax(lo, v))
                // (solverBlockTGS.cuh:473). FClamp orders signed zeros under SZINP: (lo, hi) = (+0, -0)
                // with v < 0 gives -0 where the CUDA gives +0.
                const float torsLo = -maxDynFrictionImpulse * frictionScale;
                const float torsHi = maxDynFrictionImpulse * frictionScale;
                const float torsMax = (torsLo > totalImpulse) ? torsLo : totalImpulse;
                const float totalClamped = (torsHi < torsMax) ? torsHi : torsMax;
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
        // P9.86: CUDA fminf(maxImpulse, fmaxf(minImpulse, u)) (solverBlockTGS.cuh:907) orders the zeros
        // (fmaxf(+-0) = +0 unless both -0, fminf(+-0) = -0 unless both +0); llvmpipe FMax/FMin return the
        // second operand on a zero tie. Zero ties are spelled as selects; all else keeps max/min.
        const float maxedForce = (minImpulse == 0.0f && unclampedForce == 0.0f)
                                     ? ((asuint(minImpulse) == 0u) ? minImpulse : unclampedForce)
                                     : max(minImpulse, unclampedForce);
        const float clampedForce = (maxImpulse == 0.0f && maxedForce == 0.0f)
                                       ? ((asuint(maxImpulse) != 0u) ? maxImpulse : maxedForce)
                                       : min(maxImpulse, maxedForce);
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
// (rigid path of solveBlockUnified); one 32-thread workgroup per batch.
// Bound-buffer HARNESS form driven by the solverTGS_contactAndJointBatches unit
// test; the engine-ABI (BDA) form is solveBlockUnified at the end of this file.
// The contact-vs-joint discriminator uses the engine's PxgConstraintType
// encoding (pxgIsContactType: contact = eCONTACT(1)/eARTI_CONTACT(3), joint =
// 0/2), matching the block prep the engine actually writes and its unit-test
// fixture — NOT the legacy PXG_CONSTRAINT_CONTACT=0 convention.
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
    if (!pxgIsContactType(constraintType))
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

    if (pxgIsContactType(constraintType))
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

// concludeBlocksTGS (P5 Step 4 BDA engine ABI)
//
// Port of concludeBlocksTGS() in solverMultiBlockTGS.cu.  The engine launches
// it with (const PxgSolverCoreDesc*, const PxgSolverSharedDesc<...>*,
// PxU32 islandIndex): argSizes {8,8,4} -> 24-byte blob.  Everything the harness
// form read from gParams (the batch start/end range) is derived from the island
// context, exactly like the .cu; one warp per batch, block = CONCLUDE_BLOCKS
// (256) threads, grid = nbBlocksRequired (WorkgroupCount().x), so the GLOBAL
// warp index selects the batch.
//
// The contact branch (concludeContactBlockTGS) is compiled out ("#if 0") in the
// CUDA original — TGS contacts keep their bias in the velocity iterations — so
// contact / articulation-contact batches are a no-op and only the 1D-joint
// branch (conclude1DBlockTGS) does any work.  The contact discriminator uses
// the engine's PxgConstraintType encoding (see the constants above), NOT the
// harness kernels' PXG_CONSTRAINT_CONTACT=0 convention, so real contact batches
// (constraintType == eCONTACT == 1) are correctly skipped.
struct ConcludeBlocksTGSAbiParams
{
    uint64_t solverCoreDesc; // @0
    uint64_t sharedDesc;     // @8
    uint islandIndex;        // @16
}                            // = 24 (round to maxAlign 8)
[[vk::push_constant]]
ConstantBuffer<ConcludeBlocksTGSAbiParams> gConcludeAbi;

[shader("compute")]
[numthreads(256, 1, 1)]
void concludeBlocksTGS(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    PxgSolverCoreDescGpu desc = (Ptr<PxgSolverCoreDescGpu>(gConcludeAbi.solverCoreDesc))[0];
    PxgSolverSharedDescTGSGpu sharedDesc = (Ptr<PxgSolverSharedDescTGSGpu>(gConcludeAbi.sharedDesc))[0];
    PxgIslandContextGpu island =
        (Ptr<PxgIslandContextGpu>(desc.islandContextPool))[gConcludeAbi.islandIndex];

    const uint startIndex = island.mBatchStartIndex;
    // endIndex mirrors the .cu: mBatchCount + mArtiBatchCount +
    // mStaticArtiBatchCount + mSelfArtiBatchCount + startIndex +
    // mStaticRigidBatchCount.
    const uint endIndex = island.mBatchCount + island.mArtiBatchCount +
        island.mStaticArtiBatchCount + island.mSelfArtiBatchCount + startIndex +
        island.mStaticRigidBatchCount;

    const uint warpSize = 32u;
    const uint warpIndex = (groupThreadId.x + groupId.x * 256u) / warpSize;
    const uint threadIndexInWarp = groupThreadId.x & (warpSize - 1u);

    const uint k = startIndex + warpIndex;
    if (k >= endIndex)
        return;

    // Null-guard (llvmpipe): a constraint-free island leaves the block-batch
    // pointer null/unallocated; the LLVM optimizer would speculatively hoist
    // the batch load above the range check and fault.  No batch buffer ==
    // nothing to conclude.
    if (sharedDesc.iter_blockConstraintBatch == 0)
        return;

    Ptr<PxgBlockConstraintBatch> batchHeaders =
        Ptr<PxgBlockConstraintBatch>(sharedDesc.iter_blockConstraintBatch);

    const uint descStride = batchHeaders[k].descStride_constraintType & 0xFFFFu;
    const uint constraintType = batchHeaders[k].descStride_constraintType >> 16;

    const bool isContact = (constraintType == PXG_SOLVER_CONSTRAINT_TYPE_CONTACT) ||
                           (constraintType == PXG_SOLVER_CONSTRAINT_TYPE_ARTI_CONTACT);
    if (threadIndexInWarp >= descStride || isContact)
        return;

    // conclude1DBlockTGS()
    Ptr<PxgTGSBlockSolverConstraint1DHeader> jointHeaders =
        Ptr<PxgTGSBlockSolverConstraint1DHeader>(sharedDesc.iter_blockJointConstraintHeaders);
    Ptr<PxgTGSBlockSolverConstraint1DCon> jointRowsCon =
        Ptr<PxgTGSBlockSolverConstraint1DCon>(sharedDesc.iter_blockJointConstraintRowsCon);

    const uint batchIndex = batchHeaders[k].mConstraintBatchIndex;
    const uint startConstraintIndex = batchHeaders[k].startConstraintIndex;

    const uint rowCount =
        jointHeaders[batchIndex].rowCounts_breakable_orthoAxisCount[threadIndexInWarp] & 0xFFu;

    for (uint i = 0; i < rowCount; i++)
    {
        const uint r = startConstraintIndex + i;
        const uint flags = jointRowsCon[r].flags[threadIndexInWarp];

        // biasScale, initBias, velTarget and velMultiplier are repacked with
        // other coefficients; zeroing biasScale requires initBias == 0 too,
        // which the solve enforces (initBias = biasScale == 0 ? 0 : initBias).
        if ((flags & DY_SC_FLAG_KEEP_BIAS) == 0)
        {
            float4 v = jointRowsCon[r].lin1XYZ_biasScaleOrCoeff1[threadIndexInWarp];
            v.w = 0.0f;
            jointRowsCon[r].lin1XYZ_biasScaleOrCoeff1[threadIndexInWarp] = v;
        }
        if ((flags & DY_SC_FLAG_SPRING) != 0)
        {
            float4 v0 = jointRowsCon[r].lin0XYZ_initBiasOrCoeff0[threadIndexInWarp];
            float4 v1 = jointRowsCon[r].lin1XYZ_biasScaleOrCoeff1[threadIndexInWarp];
            float4 v2 = jointRowsCon[r].ang0XYZ_velMultiplierOrCoeff2[threadIndexInWarp];
            float4 v3 = jointRowsCon[r].ang1XYZ_velTargetOrCoeff3[threadIndexInWarp];
            v0.w = 0.0f;
            v1.w = 0.0f;
            v2.w = 0.0f;
            v3.w = 0.0f;
            jointRowsCon[r].lin0XYZ_initBiasOrCoeff0[threadIndexInWarp] = v0;
            jointRowsCon[r].lin1XYZ_biasScaleOrCoeff1[threadIndexInWarp] = v1;
            jointRowsCon[r].ang0XYZ_velMultiplierOrCoeff2[threadIndexInWarp] = v2;
            jointRowsCon[r].ang1XYZ_velTargetOrCoeff3[threadIndexInWarp] = v3;
        }
    }
}

// mirrors writeBackContactBlockFriction() in solverBlockCommon.cuh (TGS
// header/friction types).  BDA form: the friction/patch pools are reached
// through Ptr<> off the descriptors instead of the bound-buffer globals.
void writeBackContactBlockFrictionTGSBda(
    uint threadIndex, uint numFrictionConstr, uint batchIndex, uint startFrictionIndex,
    Ptr<PxgTGSBlockSolverFrictionHeader> frictionHeaders,
    Ptr<PxgTGSBlockSolverContactFriction> frictions,
    Ptr<PxgBlockFrictionPatch> blockFrictionPatches,
    Ptr<PxgFrictionPatchGPU> frictionPatches)
{
    const uint patchIndex = blockFrictionPatches[batchIndex].patchIndex[threadIndex];
    if (patchIndex != 0xFFFFFFFFu)
    {
        const float4 axis0 = frictionHeaders[batchIndex].frictionNormals0[threadIndex];
        const float4 axis1 = frictionHeaders[batchIndex].frictionNormals1[threadIndex];

        frictionPatches[patchIndex].anchors = numFrictionConstr / 2;

        if (numFrictionConstr >= 2)
        {
            const float4 anchor = blockFrictionPatches[batchIndex].anchorPoints0[threadIndex];
            const float impulse0 = frictions[startFrictionIndex + 0].appliedForce[threadIndex];
            const float impulse1 = frictions[startFrictionIndex + 1].appliedForce[threadIndex];
            const float3 imp = axis0.xyz * impulse0 + axis1.xyz * impulse1;
            frictionPatches[patchIndex].p0x = anchor.x;
            frictionPatches[patchIndex].p0y = anchor.y;
            frictionPatches[patchIndex].p0z = anchor.z;
            frictionPatches[patchIndex].i0x = imp.x;
            frictionPatches[patchIndex].i0y = imp.y;
            frictionPatches[patchIndex].i0z = imp.z;
        }
        if (numFrictionConstr >= 4)
        {
            const float4 anchor = blockFrictionPatches[batchIndex].anchorPoints1[threadIndex];
            const float impulse0 = frictions[startFrictionIndex + 2].appliedForce[threadIndex];
            const float impulse1 = frictions[startFrictionIndex + 3].appliedForce[threadIndex];
            const float3 imp = axis0.xyz * impulse0 + axis1.xyz * impulse1;
            frictionPatches[patchIndex].p1x = anchor.x;
            frictionPatches[patchIndex].p1y = anchor.y;
            frictionPatches[patchIndex].p1z = anchor.z;
            frictionPatches[patchIndex].i1x = imp.x;
            frictionPatches[patchIndex].i1y = imp.y;
            frictionPatches[patchIndex].i1z = imp.z;
        }
    }
}

// 64-bit compare of PxNodeIndex values held as uint2
bool nodeIndexLess(uint2 a, uint2 b)
{
    return a.y < b.y || (a.y == b.y && a.x < b.x);
}

// writebackBlocksTGS (P5 Step 4 BDA engine ABI)
//
// Port of writebackBlocksTGS() in solverMultiBlockTGS.cu (rigid path).  The
// engine launches it with (PxgSolverCoreDesc*, const PxgSolverSharedDesc<...>*,
// PxU32 islandIndex): argSizes {8,8,4} -> 24-byte blob.  Everything the harness
// form read from gParams / bound buffers is now derived from the descriptors
// via Ptr<>; the batch start/end range is derived from the island context
// exactly like the .cu.  One warp per batch, block = WRITEBACK_BLOCKS (256)
// threads = 8 warps, grid = nbBlocksRequired (WorkgroupCount().x), so the GLOBAL
// warp index selects the batch, mirroring concludeBlocksTGS.
//
// Deviations from the .cu:
//  - The contact discriminator uses the engine's PxgConstraintType encoding
//    (contact == eCONTACT(1) / eARTICULATION_CONTACT(3) via pxgIsContactType),
//    NOT the harness kernels' inverted PXG_CONSTRAINT_CONTACT=0 convention.
//  - Threshold elements are appended per-thread through an atomic on the desc's
//    sharedThresholdStreamIndex counter instead of the CUDA warp-aggregated
//    shared-memory staging (the __shared__ ThresholdStreamElement blob + warp
//    prefix): the stream CONTENTS are identical, and the intra-stream ordering
//    is nondeterministic in both versions, so no warp/ballot reduction and no
//    large groupshared blob are needed (keeps well under the 32 KiB budget).
struct WritebackBlocksTGSAbiParams
{
    uint64_t solverCoreDesc; // @0
    uint64_t sharedDesc;     // @8
    uint islandIndex;        // @16
}                            // = 24 (round to maxAlign 8)
[[vk::push_constant]]
ConstantBuffer<WritebackBlocksTGSAbiParams> gWritebackAbi;

[shader("compute")]
[numthreads(256, 1, 1)]
void writebackBlocksTGS(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgSolverCoreDescGpu> descPtr = Ptr<PxgSolverCoreDescGpu>(gWritebackAbi.solverCoreDesc);
    PxgSolverCoreDescGpu desc = descPtr[0];
    PxgSolverSharedDescTGSGpu sharedDesc =
        (Ptr<PxgSolverSharedDescTGSGpu>(gWritebackAbi.sharedDesc))[0];
    PxgIslandContextGpu island =
        (Ptr<PxgIslandContextGpu>(desc.islandContextPool))[gWritebackAbi.islandIndex];

    const uint startIndex = island.mBatchStartIndex;
    // endIndex mirrors the .cu: mBatchCount + mArtiBatchCount +
    // mStaticArtiBatchCount + mSelfArtiBatchCount + mStaticRigidBatchCount +
    // startIndex.
    const uint endIndex = island.mBatchCount + island.mArtiBatchCount +
        island.mStaticArtiBatchCount + island.mSelfArtiBatchCount +
        island.mStaticRigidBatchCount + startIndex;

    const uint warpSize = 32u;
    const uint warpIndex = (groupThreadId.x + groupId.x * 256u) / warpSize;
    const uint threadIndex = groupThreadId.x & (warpSize - 1u);

    const uint k = startIndex + warpIndex;
    if (k >= endIndex)
        return;

    // Null-guard (llvmpipe): a constraint-free island leaves the block-batch
    // pointer null/unallocated; the LLVM optimizer would speculatively hoist
    // the batch load above the range check and fault.  No batch buffer ==
    // nothing to write back.
    if (sharedDesc.iter_blockConstraintBatch == 0)
        return;

    Ptr<PxgBlockConstraintBatch> batchHeaders =
        Ptr<PxgBlockConstraintBatch>(sharedDesc.iter_blockConstraintBatch);

    const uint descStride = batchHeaders[k].descStride_constraintType & 0xFFFFu;
    const uint constraintType = batchHeaders[k].descStride_constraintType >> 16;
    if (threadIndex >= descStride)
        return;

    const uint batchIndex = batchHeaders[k].mConstraintBatchIndex;
    const uint startConstraintIndex = batchHeaders[k].startConstraintIndex;
    const uint startFrictionIndex = batchHeaders[k].startFrictionIndex;

    if (pxgIsContactType(constraintType))
    {
        // writeBackContactBlockTGS()
        Ptr<PxgTGSBlockSolverContactHeader> contactHeaders =
            Ptr<PxgTGSBlockSolverContactHeader>(sharedDesc.iter_blockContactHeaders);
        Ptr<PxgTGSBlockSolverFrictionHeader> frictionHeaders =
            Ptr<PxgTGSBlockSolverFrictionHeader>(sharedDesc.iter_blockFrictionHeaders);
        Ptr<PxgTGSBlockSolverContactPoint> contactPoints =
            Ptr<PxgTGSBlockSolverContactPoint>(sharedDesc.iter_blockContactPoints);
        Ptr<PxgTGSBlockSolverContactFriction> frictions =
            Ptr<PxgTGSBlockSolverContactFriction>(sharedDesc.iter_blockFrictions);
        Ptr<PxgBlockFrictionPatch> blockFrictionPatches =
            Ptr<PxgBlockFrictionPatch>(sharedDesc.blockCurrentFrictionPatches);
        Ptr<float> forceBuffer = Ptr<float>(desc.forceBuffer);
        Ptr<PxgFrictionPatchGPU> frictionPatches = Ptr<PxgFrictionPatchGPU>(desc.frictionPatches);
        Ptr<PxgSolverBodyData> solverBodyData = Ptr<PxgSolverBodyData>(desc.solverBodyDataPool);

        const uint bodyAIndex = batchHeaders[k].bodyAIndex[threadIndex];
        const uint bodyBIndex = batchHeaders[k].bodyBIndex[threadIndex];

        float normalForce = 0.0f;

        const uint forceWritebackOffset = contactHeaders[batchIndex].forceWritebackOffset[threadIndex];
        const bool forceThreshold =
            (contactHeaders[batchIndex].flags[threadIndex] & PXG_CONTACT_FLAG_HAS_FORCE_THRESHOLDS) != 0;

        const uint numFrictionConstr = frictionHeaders[batchIndex].numFrictionConstr[threadIndex];
        const uint numNormalConstr = contactHeaders[batchIndex].numNormalConstr[threadIndex];

        if (forceWritebackOffset != 0xFFFFFFFFu)
        {
            for (uint i = 0; i < numNormalConstr; i++)
            {
                const float appliedForce = contactPoints[startConstraintIndex + i].appliedForce[threadIndex];
                forceBuffer[forceWritebackOffset + i] = appliedForce;
                normalForce += appliedForce;
            }
        }

        writeBackContactBlockFrictionTGSBda(threadIndex, numFrictionConstr, batchIndex, startFrictionIndex,
                                            frictionHeaders, frictions, blockFrictionPatches, frictionPatches);

        if (numFrictionConstr != 0 && frictionHeaders[batchIndex].broken[threadIndex] != 0)
            blockFrictionPatches[batchIndex].broken[threadIndex] = 1;

        const float reportThreshold0 = solverBodyData[bodyAIndex].reportThreshold;
        const float reportThreshold1 = solverBodyData[bodyBIndex].reportThreshold;

        if (forceThreshold && normalForce != 0.0f &&
            (reportThreshold0 < PX_MAX_F32 || reportThreshold1 < PX_MAX_F32))
        {
            ThresholdStreamElement elt;
            elt.normalForce = normalForce;
            // P9.86: CUDA PxMin<float> = selectMin (a < b ? a : b; tie/NaN -> b) (solverBlockTGS.cuh:722).
            // GLSL.std.450 FMin's tie rule for +-0 (and NaN result) differs: spell the select.
            elt.threshold = (reportThreshold0 < reportThreshold1) ? reportThreshold0 : reportThreshold1;
            uint2 nodeA = solverBodyData[bodyAIndex].islandNodeIndex;
            uint2 nodeB = solverBodyData[bodyBIndex].islandNodeIndex;
            // PxOrder(): swap so nodeIndexA < nodeIndexB
            if (nodeIndexLess(nodeB, nodeA))
            {
                const uint2 t = nodeA;
                nodeA = nodeB;
                nodeB = t;
            }
            elt.nodeIndexA = nodeA;
            elt.nodeIndexB = nodeB;
            elt.shapeInteraction = batchHeaders[k].shapeInteraction[threadIndex];
            elt.accumulatedForce = 0.0f;
            elt.pad = 0;

            // atomicAdd(&sharedThresholdStreamIndex, 1): the counter lives in the
            // desc struct itself (PxI32 field), not a separate buffer.
            Ptr<ThresholdStreamElement> thresholdStream =
                Ptr<ThresholdStreamElement>(desc.thresholdStream);
            int index;
            InterlockedAdd(descPtr[0].sharedThresholdStreamIndex, 1, index);
            thresholdStream[uint(index)] = elt;
        }
    }
    else
    {
        // writeBack1DBlockTGS()
        Ptr<PxgTGSBlockSolverConstraint1DHeader> jointHeaders =
            Ptr<PxgTGSBlockSolverConstraint1DHeader>(sharedDesc.iter_blockJointConstraintHeaders);
        Ptr<PxgTGSBlockSolverConstraint1DCon> jointRowsCon =
            Ptr<PxgTGSBlockSolverConstraint1DCon>(sharedDesc.iter_blockJointConstraintRowsCon);
        Ptr<PxgConstraintWriteback> constraintWriteBack =
            Ptr<PxgConstraintWriteback>(desc.constraintWriteBack);

        const uint forceWritebackOffset = jointHeaders[batchIndex].writeBackOffset[threadIndex];
        const uint rowCounts_breakable = jointHeaders[batchIndex].rowCounts_breakable_orthoAxisCount[threadIndex];
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
                if ((jointRowsCon[r].flags[threadIndex] & DY_SC_FLAG_OUTPUT_FORCE) != 0)
                {
                    const float4 lin0XYZ = jointRowsCon[r].lin0XYZ_initBiasOrCoeff0[threadIndex];
                    const float4 ang0WriteBack = jointRowsCon[r].ang0XYZ_velMultiplierOrCoeff2[threadIndex];
                    const float appliedForce = jointRowsCon[r].appliedForce[threadIndex];
                    linVel += lin0XYZ.xyz * appliedForce;
                    angVel += ang0WriteBack.xyz * appliedForce;
                }

                const float err = jointRowsCon[r].residual[threadIndex];
                constraintErrorSq += err * err;
            }

            const float4 body0WorldOffset_invMass0 = jointHeaders[batchIndex].rAWorld_invMass0D0[threadIndex];
            angVel -= cross(body0WorldOffset_invMass0.xyz, linVel);

            // *** P9.79b -- AN UPSTREAM CPU-vs-CUDA DIVERGENCE, MEASURED AND KEPT. ***
            //
            // PhysX's own CPU TGS writeback (writeBack1DStep,
            // lowleveldynamics/src/DyTGSContactPrep.cpp:2856) accumulates
            //
            //     ang += (c->ang0 + c->lin0.cross(header->rAWorld)) * c->appliedForce;
            //
            // The CUDA this file mirrors (writeBack1DBlockTGS, solverBlockTGS.cuh:1157)
            // accumulates only
            //
            //     angVel += ang0WriteBack * appliedForce;
            //
            // -- the `+ lin0.cross(rAWorld)` term is NOT there.  Both then subtract
            // body0WorldOffset.cross(lin), and on the GPU side body0WorldOffset and
            // rAWorld are the SAME field (rAWorld_invMass0D0), so the CPU counts that
            // moment arm twice where the CUDA counts it once.
            //
            // This is NOT a port defect and NOT a transcription slip: it is a
            // difference between two of the engine's own solvers, and the PGS pair does
            // not have it (DySolverConstraints.cpp:646 and solverBlock.cuh:826 both
            // accumulate a pre-baked `ang0Writeback` with no cross term, which is why
            // CPU and GPU PGS agree exactly).
            //
            // MEASURED, not inferred.  PXG_SCENE=directapi holds a 1000 kg box on a
            // fully-locked D6 joint anchored at (4,3,2); |F| = 9810 N in all four
            // combinations, and the torque is
            //
            //     GPU TGS  (-19620, 0, +39240)      CPU TGS  (-39240, 0, +78480)
            //     GPU PGS  (     0, 0,  -4905)      CPU PGS  (     0, 0,  -4905)
            //
            // Adding the CPU's term here as an experiment made the GPU print
            // (-39240.00, +0.00, +78479.99) -- the CPU answer to the last digit, not
            // approximately.  That is what closes the diagnosis; the experiment was
            // then reverted.
            //
            // THE PORT KEEPS THE CUDA BEHAVIOUR.  This project's whole method is
            // bit-exact fidelity to the .cu, and "fix it to match the CPU" would make
            // every future GPU-vs-CUDA comparison meaningless while silently changing
            // what PxDirectGPUAPI::getD6JointData and PxConstraint::getForce report on
            // the GPU path.  Recorded here so the next reader does not rediscover it as
            // a bug in this file.

            const uint broken = breakable != 0
                // P9.47: the TGS twin of solverBlock.cuh:840 -- magnitude(), not Length.
                ? ((sqrt(dot(linVel, linVel)) > jointHeaders[batchIndex].linBreakImpulse[threadIndex] ||
                    sqrt(dot(angVel, angVel)) > jointHeaders[batchIndex].angBreakImpulse[threadIndex]) ? 1u : 0u)
                : 0u;
            constraintWriteBack[forceWritebackOffset].angularImpulse_residual =
                float4(angVel, constraintErrorSq);
            // THE BROKEN FLAG IS THE SIGN BIT OF A ZERO (solverBlockTGS.cuh:1172), and it
            // must NOT travel through a float select.  Vulkan does not require
            // SignedZeroInfNanPreserve, and llvmpipe folds `cond ? -0.0f : 0.0f` to a
            // plain +0.0f: the two arms compare equal, so the select is dead under
            // no-signed-zeros semantics.  The result is a joint that never reports as
            // broken — silently, because every other component of the writeback is
            // bit-exact.  Selecting the BIT PATTERN and reinterpreting keeps the sign.
            constraintWriteBack[forceWritebackOffset].linearImpulse_broken =
                float4(linVel, asfloat(broken != 0 ? 0x80000000u : 0u));
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
        // P9.86f: solverMultiBlockTGS.cu:348/410 accumulates into a float4 curVel1 = make_float4(0.f)
        // whose .z/.w are never touched, so `curVel1 += vel1` computes (+0) + vel1.z and (+0) + vel1.w:
        // a -0 delta-position lane comes back +0.  Spelled as the plusZero select (a literal 0 + x would
        // be folded, F1/F3); NaN and every nonzero value pass through as in the add.
        bodyVelocities[bodyOffset + deltaVOffset + bodyId + totalBodiesIncKinematics] =
            float4(vel1.xy + curVel1, (vel1.z == 0.0f) ? 0.0f : vel1.z, (vel1.w == 0.0f) ? 0.0f : vel1.w);
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

                // P9.47: solverMultiBlockTGS.cu:581 is `result.getNormalized()`, and
                // PxQuat::getNormalized (PxQuat.h:211-215) is s = 1/magnitude() then
                // FOUR multiplies -- magnitude() being PxSqrt of the 4-wide square sum.
                q = result * (1.0f / sqrt(dot(result, result)));
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

// ---------------------------------------------------------------------------
// solveWholeIslandTGS (P5 Step 4 BDA engine ABI)
//
// Port of solveWholeIslandTGS() in solverMultiBlockTGS.cu: one 256-thread
// block solves ALL partitions of a small island in a single launch, staging
// every body-slab velocity/delta in shared memory and writing the results to
// the averaging slots consumed by computeAverageSolverBodyVelocityTGS.
// The solve math itself is the *Bda transcription of the bit-verified
// bound-buffer helpers above (solveContactBlockTGS / solve1DBlockTGS), with
// the contact-residual statistics (PxgErrorAccumulator) added because this
// kernel — unlike the per-partition path — accumulates them into
// solverDesc->contactErrorAccumulator itself.
//
// Differences from the .cu:
//   - maxBodies is 640 instead of 944 (hardware limit): CUDA's four
//     PxVec3[944] staging arrays are 45312 B of shared memory, over the
//     portable Vulkan/llvmpipe maxComputeSharedMemorySize of 32768.  With 640
//     (4 x 640 x 12 B = 30720 B, + 1 KiB of residual-counter scratch = 31744 B)
//     the kernel fits.  The HOST gate in PxgTGSCudaSolverCore.cpp::
//     solvePartitions uses the same 640 under PXG_NO_CUDA_TOOLKIT, so islands
//     with more than 640 body-slabs fall back to the (CUDA-faithful)
//     per-partition solveBlockPartitionTGS path.  The arrays are flat
//     float[3n] rather than float3[n] because workgroup float3 arrays may be
//     given a 16-byte stride, which would overflow the budget again.
//   - the warp-shuffle loadTxInertia() staging becomes direct loads, gated on
//     the live lanes (lane < mDescStride) because the tail lanes'
//     bodyA/BIndex is uninitialized batch memory (same policy as
//     solveBlockPartitionTGS above);
//   - accumulateErrorGlobalFullWarp(): the __shfl_xor butterfly reduction
//     becomes the same butterfly through the (then-dead) staging arrays; the
//     reduction itself runs unconditionally (uniform control flow around the
//     barriers) and only the atomic tail is gated on
//     residualAccumulationEnabled — identical results, since a disabled
//     launch accumulates nothing;
//   - CUDA's float atomicAdd / AtomicMax on the accumulator become CAS loops
//     on the raw float bits (InterlockedCompareExchange has no float
//     overload), and __fdividef in the residual becomes exact division.
// ---------------------------------------------------------------------------

// (const PxgSolverCoreDesc*, const PxgSolverSharedDesc<IterativeSolveDataTGS>*,
//  const PxU32 islandIndex, bool doFriction, const PxReal elapsedTime,
//  const PxReal minPen): argSizes {8,8,4,1,4,4} -> the bool byte lands at
// blob offset 20 (bytes 21..23 are zero-filled padding), the floats at 24/28,
// 32 bytes total.
struct SolveWholeIslandTGSAbiParams
{
    uint64_t solverCoreDesc; // @0
    uint64_t sharedDesc;     // @8
    uint islandIndex;        // @16
    uint doFrictionWord;     // @20: byte 0 = doFriction (unused by the kernel body, exactly like the .cu)
    float elapsedTime;       // @24
    float minPen;            // @28
}                            // = 32
[[vk::push_constant]]
ConstantBuffer<SolveWholeIslandTGSAbiParams> gWholeIslandAbi;

// mirrors PxgErrorAccumulator (gpucommon/src/CUDA/solverResidual.cuh): the
// per-thread contact residual statistics.
struct PxgErrorAccumulatorSlang
{
    float sumOfSquares;
    int counter;
    float maxError;
}

// PxgErrorAccumulator::accumulateErrorLocal().  CUDA computes the residual
// with __fdividef; exact division here (the CPU references divide exactly
// too).
void pxgAccumulateErrorLocal(inout PxgErrorAccumulatorSlang e, float deltaF, float velocityMultiplier)
{
    const float r = velocityMultiplier == 0.0f ? 0.0f : deltaF / velocityMultiplier;
    e.sumOfSquares += r * r;
    ++e.counter;
    e.maxError = max(e.maxError, abs(r));
}

// mirrors checkActiveContactBlockTGS() in solverBlockTGS.cuh — the read-only
// "would this contact still push?" predicate that markActiveSlabTGS uses to
// decide whether a body's slab is active.  It is a stripped variant of the
// contact solve above: same per-normal-constraint deltaF computation, but it
// (a) never applies reference-count scaling (this runs BEFORE reference counts
// are known — it is what computes them), (b) never mutates the velocities /
// relVel1 between contact points (each point is tested against the SAME input
// velocities), (c) writes nothing back, and (d) short-circuits to true on the
// first contact whose |deltaF| exceeds 1e-8.  No friction is considered, exactly
// like the .cu.
bool checkActiveContactBlockTGSBda(Ptr<PxgBlockConstraintBatch> batches, uint batchIndex,
                                   float3 linVel0, float3 angVel0, float3 linVel1, float3 angVel1,
                                   float3 b0LinDelta, float3 b0AngDelta, float3 b1LinDelta, float3 b1AngDelta,
                                   uint threadIndex,
                                   Ptr<PxgTGSBlockSolverContactHeader> contactHeaders,
                                   Ptr<PxgTGSBlockSolverContactPoint> contactPoints,
                                   float elapsedTime, float minPen)
{
    const uint constraintBatchIndex = batches[batchIndex].mConstraintBatchIndex;
    const uint startConstraintIndex = batches[batchIndex].startConstraintIndex;

    const uint numNormalConstr = contactHeaders[constraintBatchIndex].numNormalConstr[threadIndex];

    if (numNormalConstr == 0)
        return false;

    const float maxPenBias = contactHeaders[constraintBatchIndex].maxPenBias[threadIndex];
    const float4 invMass0_1_angDom0_1 = contactHeaders[constraintBatchIndex].invMass0_1_angDom0_1[threadIndex];

    // No reference-count scaling here (see the (a) note above).
    const float invMassA = invMass0_1_angDom0_1.x;
    const float invMassB = invMass0_1_angDom0_1.y;

    const float4 normal_staticFriction = contactHeaders[constraintBatchIndex].normal_staticFriction[threadIndex];
    const float3 normal = normal_staticFriction.xyz;

    const float restitutionXdt = contactHeaders[constraintBatchIndex].restitutionXdt[threadIndex];
    const uint flags = contactHeaders[constraintBatchIndex].flags[threadIndex];

    // Input velocities stay constant across the loop (no impulse applied).
    const float3 relMotion = b0LinDelta - b1LinDelta;
    const float deltaV = dot(normal, relMotion);
    const float relVel1 = dot(linVel0 - linVel1, normal);

    for (uint i = 0; i < numNormalConstr; i++)
    {
        const uint c = startConstraintIndex + i;

        const float4 raXn_contactCoeff = contactPoints[c].raXn_extraCoeff[threadIndex];
        const float4 rbXn_targetVelW = contactPoints[c].rbXn_targetVelW[threadIndex];
        const float appliedForce = contactPoints[c].appliedForce[threadIndex];
        const float separation = contactPoints[c].separation[threadIndex];
        const float maxImpulse = contactPoints[c].maxImpulse[threadIndex];

        const float resp0 = contactPoints[c].resp0[threadIndex];
        const float resp1 = contactPoints[c].resp1[threadIndex];
        const float unitResponse = resp0 + resp1;
        const float recipResponse = (unitResponse > 0.0f) ? (1.0f / unitResponse) : 0.0f;

        float velMultiplier = recipResponse;
        float biasCoefficient = contactPoints[c].biasCoefficient[threadIndex];

        if (restitutionXdt < 0.0f)
            computeCompliantContactCoefficientsTGS(flags, restitutionXdt, unitResponse, recipResponse,
                                                   raXn_contactCoeff.w, velMultiplier, biasCoefficient);

        const float3 raXn = raXn_contactCoeff.xyz;
        const float3 rbXn = rbXn_targetVelW.xyz;
        const float targetVel = rbXn_targetVelW.w;

        const float v0 = dot(angVel0, raXn);
        const float v1 = dot(angVel1, rbXn);
        const float normalVel = relVel1 + (v0 - v1);

        const float sep = max(minPen, separation + deltaV + dot(b0AngDelta, raXn) - dot(b1AngDelta, rbXn));

        const float tVelErr = targetVel * elapsedTime;
        const float biasedErr = recipResponse * min(-maxPenBias, biasCoefficient * (sep - tVelErr));

        const float tempDeltaF = biasedErr - (normalVel - targetVel) * velMultiplier;
        const float _deltaF = max(tempDeltaF, -appliedForce);
        const float _newForce = appliedForce + _deltaF;
        const float newForce = min(_newForce, maxImpulse);
        const float deltaF = newForce - appliedForce;

        // active contact
        if (abs(deltaF) > 1.0e-8f)
            return true;
    }

    return false;
}

// mirrors solveContactBlockTGS() in solverBlockTGS.cuh — BDA variant of the
// bound-buffer helper above (identical body, Ptr<> data path), plus the
// PxgErrorAccumulator statistics the whole-island kernel needs (the CUDA
// helper takes a PxgErrorAccumulator* that the per-partition launch passes as
// NULL; errorEnabled mirrors that pointer's nullness).
void solveContactBlockTGSBda(Ptr<PxgBlockConstraintBatch> batches, uint batchIndex,
                             inout float3 linVel0, inout float3 angVel0,
                             inout float3 linVel1, inout float3 angVel1,
                             float3 b0LinDelta, float3 b0AngDelta, float3 b1LinDelta, float3 b1AngDelta,
                             uint threadIndex,
                             Ptr<PxgTGSBlockSolverContactHeader> contactHeaders,
                             Ptr<PxgTGSBlockSolverFrictionHeader> frictionHeaders,
                             Ptr<PxgTGSBlockSolverContactPoint> contactPoints,
                             Ptr<PxgTGSBlockSolverContactFriction> frictionsBuf,
                             float elapsedTime, float minPen,
                             bool errorEnabled, inout PxgErrorAccumulatorSlang error,
                             float ref0, float ref1)
{
    const uint constraintBatchIndex = batches[batchIndex].mConstraintBatchIndex;
    const uint startConstraintIndex = batches[batchIndex].startConstraintIndex;
    const uint startFrictionIndex = batches[batchIndex].startFrictionIndex;

    float accumulatedNormalImpulse = 0.0f;

    const uint numNormalConstr = contactHeaders[constraintBatchIndex].numNormalConstr[threadIndex];
    const uint numFrictionConstrTotal = frictionHeaders[constraintBatchIndex].numFrictionConstr[threadIndex];
    const uint numFrictionConstr = numFrictionConstrTotal & ~1u;

    float accumDeltaF = 0.0f;

    if (numNormalConstr != 0)
    {
        const float maxPenBias = contactHeaders[constraintBatchIndex].maxPenBias[threadIndex];
        const float4 invMass0_1_angDom0_1 = contactHeaders[constraintBatchIndex].invMass0_1_angDom0_1[threadIndex];

        const float invMassA = ref0 * invMass0_1_angDom0_1.x;
        const float invMassB = ref1 * invMass0_1_angDom0_1.y;
        const float angDom0 = ref0 * invMass0_1_angDom0_1.z;
        const float angDom1 = ref1 * invMass0_1_angDom0_1.w;
        const float sumInvMass = invMassA + invMassB;

        const float4 normal_staticFriction = contactHeaders[constraintBatchIndex].normal_staticFriction[threadIndex];
        const float3 normal = normal_staticFriction.xyz;
        const float staticFrictionCof = normal_staticFriction.w;

        const float restitutionXdt = contactHeaders[constraintBatchIndex].restitutionXdt[threadIndex];
        const float p8 = contactHeaders[constraintBatchIndex].p8[threadIndex];
        const uint flags = contactHeaders[constraintBatchIndex].flags[threadIndex];

        const float3 delLinVel0 = normal * invMassA;
        const float3 delLinVel1 = normal * invMassB;

        const float3 relMotion = b0LinDelta - b1LinDelta;
        const float deltaV = dot(normal, relMotion);

        float relVel1 = dot(linVel0 - linVel1, normal);

        for (uint i = 0; i < numNormalConstr; i++)
        {
            const uint c = startConstraintIndex + i;

            const float4 raXn_contactCoeff = contactPoints[c].raXn_extraCoeff[threadIndex];
            const float4 rbXn_targetVelW = contactPoints[c].rbXn_targetVelW[threadIndex];
            const float appliedForce = contactPoints[c].appliedForce[threadIndex];
            const float separation = contactPoints[c].separation[threadIndex];
            const float maxImpulse = contactPoints[c].maxImpulse[threadIndex];

            const float resp0 = ref0 * contactPoints[c].resp0[threadIndex];
            const float resp1 = ref1 * contactPoints[c].resp1[threadIndex];
            const float unitResponse = resp0 + resp1;
            const float recipResponse = (unitResponse > 0.0f) ? (1.0f / unitResponse) : 0.0f;

            float velMultiplier = recipResponse;
            float biasCoefficient = contactPoints[c].biasCoefficient[threadIndex];

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

            if (errorEnabled)
                pxgAccumulateErrorLocal(error, deltaF, velMultiplier);

            contactPoints[c].appliedForce[threadIndex] = newForce;
            accumulatedNormalImpulse += newForce;
        }

        linVel0 += delLinVel0 * accumDeltaF;
        linVel1 -= delLinVel1 * accumDeltaF;

        if (numFrictionConstrTotal != 0)
        {
            const float biasCoefficient = frictionHeaders[constraintBatchIndex].biasCoefficient[threadIndex];
            const float dynamicFrictionCof = frictionHeaders[constraintBatchIndex].dynamicFriction[threadIndex];
            const float maxFrictionImpulse = staticFrictionCof * accumulatedNormalImpulse;
            const float maxDynFrictionImpulse = dynamicFrictionCof * accumulatedNormalImpulse;

            uint broken = 0;

            const float4 frictionNormal0 = frictionHeaders[constraintBatchIndex].frictionNormals0[threadIndex];
            const float4 frictionNormal1 = frictionHeaders[constraintBatchIndex].frictionNormals1[threadIndex];
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

                const float4 raXn_extraCoeff0 = frictionsBuf[f0].raXn_error[threadIndex];
                const float4 rbXn_targetVelW0 = frictionsBuf[f0].rbXn_targetVelW[threadIndex];
                const float initialError0 = raXn_extraCoeff0.w;
                const float appliedForce0 = frictionsBuf[f0].appliedForce[threadIndex];
                const float targetVel0 = rbXn_targetVelW0.w;
                const float f0_resp = ref0 * frictionsBuf[f0].resp0[threadIndex] + ref1 * frictionsBuf[f0].resp1[threadIndex];
                const float velMultiplier0 = (f0_resp > 0.0f) ? (p8 / f0_resp) : 0.0f;

                const float4 raXn_extraCoeff1 = frictionsBuf[f1].raXn_error[threadIndex];
                const float4 rbXn_targetVelW1 = frictionsBuf[f1].rbXn_targetVelW[threadIndex];
                const float initialError1 = raXn_extraCoeff1.w;
                const float appliedForce1 = frictionsBuf[f1].appliedForce[threadIndex];
                const float targetVel1 = rbXn_targetVelW1.w;
                const float f1_resp = ref0 * frictionsBuf[f1].resp0[threadIndex] + ref1 * frictionsBuf[f1].resp1[threadIndex];
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

                if (errorEnabled)
                {
                    pxgAccumulateErrorLocal(error, deltaF0, velMultiplier0);
                    pxgAccumulateErrorLocal(error, deltaF1, velMultiplier1);
                }

                linVel0 += delLinVel00 * deltaF0;
                linVel1 -= delLinVel10 * deltaF0;
                angVel0 += raXn0 * (deltaF0 * angDom0);
                angVel1 -= rbXn0 * (deltaF0 * angDom1);

                linVel0 += delLinVel01 * deltaF1;
                linVel1 -= delLinVel11 * deltaF1;
                angVel0 += raXn1 * (deltaF1 * angDom0);
                angVel1 -= rbXn1 * (deltaF1 * angDom1);

                frictionsBuf[f0].appliedForce[threadIndex] = newAppliedForce0;
                frictionsBuf[f1].appliedForce[threadIndex] = newAppliedForce1;
                broken = broken | (clampImpulse ? 1u : 0u);
            }

            if (numFrictionConstr < numFrictionConstrTotal)
            {
                // We have a torsional friction anchor, solve this...
                const float frictionScale = frictionHeaders[constraintBatchIndex].torsionalFrictionScale[threadIndex];

                const uint f0 = startFrictionIndex + numFrictionConstr;
                const float4 raXn_extraCoeff0 = frictionsBuf[f0].raXn_error[threadIndex];
                const float4 rbXn_targetVelW0 = frictionsBuf[f0].rbXn_targetVelW[threadIndex];
                const float appliedForce0 = frictionsBuf[f0].appliedForce[threadIndex];
                const float targetVel0 = rbXn_targetVelW0.w;
                const float f0_resp = ref0 * frictionsBuf[f0].resp0[threadIndex] + ref1 * frictionsBuf[f0].resp1[threadIndex];
                const float velMultiplier0 = (f0_resp > 0.0f) ? (p8 / f0_resp) : 0.0f;

                const float3 raXn0 = raXn_extraCoeff0.xyz;
                const float3 rbXn0 = rbXn_targetVelW0.xyz;

                const float v00 = dot(angVel0, raXn0);
                const float v10 = dot(angVel1, rbXn0);
                const float normalVel0 = v00 - v10;

                const float tmp10 = appliedForce0 - (-targetVel0) * velMultiplier0;
                const float totalImpulse = tmp10 - normalVel0 * velMultiplier0;

                const bool clampImpulse = abs(totalImpulse) > (maxFrictionImpulse * frictionScale);
                // P9.86: PxClamp(v, lo, hi) = PxMin(hi, PxMax(lo, v)) = selectMin(hi, selectMax(lo, v))
                // (solverBlockTGS.cuh:473). FClamp orders signed zeros under SZINP: (lo, hi) = (+0, -0)
                // with v < 0 gives -0 where the CUDA gives +0.
                const float torsLo = -maxDynFrictionImpulse * frictionScale;
                const float torsHi = maxDynFrictionImpulse * frictionScale;
                const float torsMax = (torsLo > totalImpulse) ? torsLo : totalImpulse;
                const float totalClamped = (torsHi < torsMax) ? torsHi : torsMax;
                const float newAppliedForce = clampImpulse ? totalClamped : totalImpulse;

                const float deltaF = newAppliedForce - appliedForce0;
                if (errorEnabled)
                    pxgAccumulateErrorLocal(error, deltaF, velMultiplier0);

                angVel0 += raXn0 * (deltaF * angDom0);
                angVel1 -= rbXn0 * (deltaF * angDom1);

                frictionsBuf[f0].appliedForce[threadIndex] = newAppliedForce;
                broken = broken | (clampImpulse ? 1u : 0u);
            }

            frictionHeaders[constraintBatchIndex].broken[threadIndex] = broken;
        }
    }
}

// mirrors solve1DBlockTGS() in solverBlockTGS.cuh — BDA variant of the
// bound-buffer helper above (identical body, Ptr<> data path)
void solve1DBlockTGSBda(Ptr<PxgBlockConstraintBatch> batches, uint batchIndex,
                        inout float3 linVel0, inout float3 angVel0,
                        inout float3 linVel1, inout float3 angVel1,
                        float3 linDelta0, float3 angDelta0, float3 linDelta1, float3 angDelta1,
                        uint threadIndex,
                        Ptr<PxgTGSBlockSolverConstraint1DHeader> jointHeaders,
                        Ptr<PxgTGSBlockSolverConstraint1DCon> jointRowsCon,
                        PxgSolverTxIData iData0, PxgSolverTxIData iData1,
                        float elapsedTime, bool residualReportingEnabled, float ref0, float ref1)
{
    const uint constraintBatchIndex = batches[batchIndex].mConstraintBatchIndex;
    const uint startConstraintIndex = batches[batchIndex].startConstraintIndex;

    const float4 raInvMass0 = jointHeaders[constraintBatchIndex].rAWorld_invMass0D0[threadIndex];
    const float4 rbInvMass1 = jointHeaders[constraintBatchIndex].rBWorld_invMass1D1[threadIndex];

    const float invMass0 = ref0 * raInvMass0.w;
    const float invMass1 = ref1 * rbInvMass1.w;

    const float3 raPrev = raInvMass0.xyz;
    const float3 rbPrev = rbInvMass1.xyz;

    const float3 ra = quatRotate(iData0.deltaBody2World.q(), raPrev);
    const float3 rb = quatRotate(iData1.deltaBody2World.q(), rbPrev);

    const float3 raMotion = (ra + linDelta0) - raPrev;
    const float3 rbMotion = (rb + linDelta1) - rbPrev;

    const float invInertiaScale0 = ref0 * jointHeaders[constraintBatchIndex].invInertiaScale0[threadIndex];
    const float invInertiaScale1 = ref1 * jointHeaders[constraintBatchIndex].invInertiaScale1[threadIndex];

    const uint rowCounts_breakable_orthoAxisCount = jointHeaders[constraintBatchIndex].rowCounts_breakable_orthoAxisCount[threadIndex];
    const uint rowCount = rowCounts_breakable_orthoAxisCount & 0xFFu; // uchar4.x

    const float4 ang0Ortho0_recipResponseW = jointHeaders[constraintBatchIndex].angOrthoAxis0_recipResponseW0[threadIndex];
    const float4 ang0Ortho1_recipResponseW = jointHeaders[constraintBatchIndex].angOrthoAxis0_recipResponseW1[threadIndex];
    const float4 ang0Ortho2_recipResponseW = jointHeaders[constraintBatchIndex].angOrthoAxis0_recipResponseW2[threadIndex];

    const float4 ang1OrthoAxis0_ErrorW = jointHeaders[constraintBatchIndex].angOrthoAxis1_ErrorW0[threadIndex];
    const float4 ang1OrthoAxis1_ErrorW = jointHeaders[constraintBatchIndex].angOrthoAxis1_ErrorW1[threadIndex];
    const float4 ang1OrthoAxis2_ErrorW = jointHeaders[constraintBatchIndex].angOrthoAxis1_ErrorW2[threadIndex];

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

        const float4 _clinVel0_coeff0W = jointRowsCon[r].lin0XYZ_initBiasOrCoeff0[threadIndex];
        const float4 _clinVel1_coeff1W = jointRowsCon[r].lin1XYZ_biasScaleOrCoeff1[threadIndex];
        const float4 _cangVel0_coeff2W = jointRowsCon[r].ang0XYZ_velMultiplierOrCoeff2[threadIndex];
        const float4 _cangVel1_coeff3W = jointRowsCon[r].ang1XYZ_velTargetOrCoeff3[threadIndex];

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

        const uint flags = jointRowsCon[r].flags[threadIndex];
        const float maxBias = jointRowsCon[r].maxBias[threadIndex];
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
                                                                  jointRowsCon[r].angularErrorScale[threadIndex],
                                                                  isSpringConstraint, targetVel, elapsedTime);

        const float resp0 = invMass0 * dot(clinVel0, clinVel0) + dot(raXnI, raXnI) * invInertiaScale0;
        const float resp1 = invMass1 * dot(clinVel1, clinVel1) + dot(rbXnI, rbXnI) * invInertiaScale1;

        const float response = resp0 + resp1;
        const float recipResponse = response > 0.0f ? 1.0f / response : 0.0f;

        const float vMul = isSpringConstraint ? velMultiplier : recipResponse * velMultiplier;

        const float unclampedBias = initBias + errorDelta * biasScale;
        const float bias = clamp(unclampedBias, minBias, maxBias);

        const float constant = isSpringConstraint ? bias + targetVel : recipResponse * (bias + targetVel);

        const float appliedForce = jointRowsCon[r].appliedForce[threadIndex];
        const float maxImpulse = jointRowsCon[r].maxImpulse[threadIndex];
        const float minImpulse = jointRowsCon[r].minImpulse[threadIndex];

        const float v0 = dot(linVel0, clinVel0) + dot(angVel0, raXnI);
        const float v1 = dot(linVel1, clinVel1) + dot(angVel1, rbXnI);
        const float normalVel = v0 - v1;

        const float unclampedForce = appliedForce + (vMul * normalVel + constant);
        // P9.86: CUDA fminf(maxImpulse, fmaxf(minImpulse, u)) (solverBlockTGS.cuh:907) orders the zeros
        // (fmaxf(+-0) = +0 unless both -0, fminf(+-0) = -0 unless both +0); llvmpipe FMax/FMin return the
        // second operand on a zero tie. Zero ties are spelled as selects; all else keeps max/min.
        const float maxedForce = (minImpulse == 0.0f && unclampedForce == 0.0f)
                                     ? ((asuint(minImpulse) == 0u) ? minImpulse : unclampedForce)
                                     : max(minImpulse, unclampedForce);
        const float clampedForce = (maxImpulse == 0.0f && maxedForce == 0.0f)
                                       ? ((asuint(maxImpulse) != 0u) ? maxImpulse : maxedForce)
                                       : min(maxImpulse, maxedForce);
        const float deltaF = clampedForce - appliedForce;

        jointRowsCon[r].appliedForce[threadIndex] = clampedForce;
        if (residualReportingEnabled)
            jointRowsCon[r].residual[threadIndex] = vMul == 0.0f ? 0.0f : deltaF / vMul;

        linVel0 = linVel0 + clinVel0 * (deltaF * invMass0);
        linVel1 = linVel1 - clinVel1 * (deltaF * invMass1);
        angVel0 = angVel0 + raXnI * deltaF * invInertiaScale0;
        angVel1 = angVel1 - rbXnI * deltaF * invInertiaScale1;
    }
}

// See the maxBodies deviation note above: 640 body-slabs, flat float[3n]
// staging arrays (30720 B) + the 1 KiB residual counter scratch = 31744 B.
static const uint PXG_WHOLE_ISLAND_MAX_BODIES = 640;
groupshared float shWholeIslandLinVel[PXG_WHOLE_ISLAND_MAX_BODIES * 3];
groupshared float shWholeIslandAngVel[PXG_WHOLE_ISLAND_MAX_BODIES * 3];
groupshared float shWholeIslandLinDelta[PXG_WHOLE_ISLAND_MAX_BODIES * 3];
groupshared float shWholeIslandAngDelta[PXG_WHOLE_ISLAND_MAX_BODIES * 3];
// residual-reduction integer scratch; the float lanes of the reduction reuse
// the then-dead LinVel/AngVel staging arrays (shared budget is tight)
groupshared uint shWholeIslandErrCounter[256];

float3 wiLoadLinVel(uint i)   { return float3(shWholeIslandLinVel[3 * i], shWholeIslandLinVel[3 * i + 1], shWholeIslandLinVel[3 * i + 2]); }
float3 wiLoadAngVel(uint i)   { return float3(shWholeIslandAngVel[3 * i], shWholeIslandAngVel[3 * i + 1], shWholeIslandAngVel[3 * i + 2]); }
float3 wiLoadLinDelta(uint i) { return float3(shWholeIslandLinDelta[3 * i], shWholeIslandLinDelta[3 * i + 1], shWholeIslandLinDelta[3 * i + 2]); }
float3 wiLoadAngDelta(uint i) { return float3(shWholeIslandAngDelta[3 * i], shWholeIslandAngDelta[3 * i + 1], shWholeIslandAngDelta[3 * i + 2]); }
void wiStoreLinVel(uint i, float3 v)   { shWholeIslandLinVel[3 * i] = v.x; shWholeIslandLinVel[3 * i + 1] = v.y; shWholeIslandLinVel[3 * i + 2] = v.z; }
void wiStoreAngVel(uint i, float3 v)   { shWholeIslandAngVel[3 * i] = v.x; shWholeIslandAngVel[3 * i + 1] = v.y; shWholeIslandAngVel[3 * i + 2] = v.z; }
void wiStoreLinDelta(uint i, float3 v) { shWholeIslandLinDelta[3 * i] = v.x; shWholeIslandLinDelta[3 * i + 1] = v.y; shWholeIslandLinDelta[3 * i + 2] = v.z; }
void wiStoreAngDelta(uint i, float3 v) { shWholeIslandAngDelta[3 * i] = v.x; shWholeIslandAngDelta[3 * i + 1] = v.y; shWholeIslandAngDelta[3 * i + 2] = v.z; }

// mirrors solveWholeIslandTGS() in solverMultiBlockTGS.cu; fixed launch shape
// grid (1,1,1) x block (256,1,1).
[shader("compute")]
[numthreads(256, 1, 1)]
void solveWholeIslandTGS(uint3 groupThreadId : SV_GroupThreadID)
{
    Ptr<PxgSolverCoreDescGpu> descPtr = Ptr<PxgSolverCoreDescGpu>(gWholeIslandAbi.solverCoreDesc);
    PxgSolverCoreDescGpu desc = descPtr[0];
    PxgSolverSharedDescTGSGpu sharedDesc = (Ptr<PxgSolverSharedDescTGSGpu>(gWholeIslandAbi.sharedDesc))[0];
    PxgIslandContextGpu island =
        (Ptr<PxgIslandContextGpu>(desc.islandContextPool))[gWholeIslandAbi.islandIndex];


    const uint startPartitionIndex = island.mStartPartitionIndex;

    const uint numDynamicBodies = island.mBodyCount; // nbBodies minus offset!

    const uint bodyOffset = island.mBodyStartIndex;

    Ptr<float4> bodyVelocities = Ptr<float4>(sharedDesc.iter_solverBodyVelPool);
    Ptr<PxgBlockConstraintBatch> batches = Ptr<PxgBlockConstraintBatch>(sharedDesc.iter_blockConstraintBatch);
    Ptr<PxgTGSBlockSolverContactHeader> contactHeaders =
        Ptr<PxgTGSBlockSolverContactHeader>(sharedDesc.iter_blockContactHeaders);
    Ptr<PxgTGSBlockSolverFrictionHeader> frictionHeaders =
        Ptr<PxgTGSBlockSolverFrictionHeader>(sharedDesc.iter_blockFrictionHeaders);
    Ptr<PxgTGSBlockSolverContactPoint> contactPoints =
        Ptr<PxgTGSBlockSolverContactPoint>(sharedDesc.iter_blockContactPoints);
    Ptr<PxgTGSBlockSolverContactFriction> frictionsBuf =
        Ptr<PxgTGSBlockSolverContactFriction>(sharedDesc.iter_blockFrictions);
    Ptr<PxgTGSBlockSolverConstraint1DHeader> jointHeaders =
        Ptr<PxgTGSBlockSolverConstraint1DHeader>(sharedDesc.iter_blockJointConstraintHeaders);
    Ptr<PxgTGSBlockSolverConstraint1DCon> jointRowsCon =
        Ptr<PxgTGSBlockSolverConstraint1DCon>(sharedDesc.iter_blockJointConstraintRowsCon);
    Ptr<uint> encodedReferenceCount = Ptr<uint>(sharedDesc.iter_solverEncodedReferenceCount);
    Ptr<PxgSolverTxIData> iData = Ptr<PxgSolverTxIData>(desc.solverBodyTxIDataPool);
    Ptr<uint> constraintsPerPartition = Ptr<uint>(desc.constraintsPerPartition);
    Ptr<uint> solverBodyReferences = Ptr<uint>(desc.solverBodyReferences); // PxgSolverReferences

    uint startIndex = island.mBatchStartIndex;

    const uint threadIdxX = groupThreadId.x;
    const uint warpIndex = threadIdxX / 32;
    const uint threadIndexInWarp = threadIdxX & 31;
    const uint numWarps = 256 / 32;

    const uint deltaVOffset = desc.accumulatedBodyDeltaVOffset;
    const uint totalBodiesIncKinematics = numDynamicBodies + bodyOffset;

    const uint numSlabs = desc.numSlabs;
    const uint numBatches = desc.numBatches;
    const uint numArticBatches = desc.numArticBatches;

    const uint averageOutputOffsets = (numArticBatches + numBatches) * 32u * 3u * 2u; // PXG_BATCH_SIZE = 32

    for (uint index = threadIdxX; index < (totalBodiesIncKinematics * numSlabs); index += 256)
    {
        const uint inputBody = deltaVOffset + (index % totalBodiesIncKinematics);
        const float4 vel0 = bodyVelocities[inputBody];
        const float4 vel1 = bodyVelocities[inputBody + totalBodiesIncKinematics];
        const float4 vel2 = bodyVelocities[inputBody + totalBodiesIncKinematics + totalBodiesIncKinematics];

        const uint writeIndex = index;

        wiStoreLinVel(writeIndex, vel0.xyz);
        wiStoreAngVel(writeIndex, float3(vel0.w, vel1.x, vel1.y));
        wiStoreLinDelta(writeIndex, float3(vel1.z, vel1.w, vel2.x));
        wiStoreAngDelta(writeIndex, float3(vel2.y, vel2.z, vel2.w));
    }

    GroupMemoryBarrierWithGroupSync();

    const bool residualAccumulationEnabled = desc.contactErrorAccumulator_mCounter >= 0;
    PxgErrorAccumulatorSlang error = { 0.0f, 0, 0.0f };

    // The constraint-solve loops dereference the block-constraint buffers
    // (batches[k], contact/joint headers, …).  For a constraint-free island
    // (mNumPartitions == 0, e.g. a body in free fall) PhysX leaves those
    // device pointers null/unallocated.  The loops are then zero-trip and the
    // CUDA kernel never dereferences them — but the SPIR-V/llvmpipe optimizer
    // speculatively hoists the first batches[] load out of the zero-trip loop,
    // dereferencing the null pointer and faulting.  Gating the whole section on
    // a non-null batch pointer makes every such load control-dependent on the
    // guard (un-hoistable) and is semantically exact: no batch buffer == no
    // constraint batches to solve.
    // The dynamic-partition constraint loops dereference the block-constraint
    // buffers (batches[k], contact/joint headers, …).  An island with no dynamic
    // partitions (mNumPartitions == 0) — e.g. a lone body whose only contact is
    // against a static actor, resolved by solveStaticBlockTGS — must skip them
    // entirely: the CUDA loops are then zero-trip, but the SPIR-V/llvmpipe
    // optimizer speculatively hoists the first batches[]/constraintsPerPartition
    // load out of the zero-trip loop and faults on the empty dynamic range.  The
    // outer scalar guard (uniform, a real control-flow barrier) blocks the hoist.
    // iter_blockConstraintBatch alone is insufficient: a static contact allocates
    // that pool while leaving this island's dynamic partition count at zero.
    if (sharedDesc.iter_blockConstraintBatch != 0 && island.mNumPartitions != 0)
    {
    for (uint partitionIndex = 0; partitionIndex < island.mNumPartitions; ++partitionIndex)
    {
        const uint endIndex = constraintsPerPartition[partitionIndex + startPartitionIndex];

        for (uint k = startIndex + warpIndex; k < endIndex; k += numWarps)
        {
            const uint descStride = batches[k].descStride_constraintType & 0xFFFFu;
            const uint constraintType = batches[k].descStride_constraintType >> 16;

            const uint bodyAIndex = batches[k].bodyAIndex[threadIndexInWarp];
            const uint bodyBIndex = batches[k].bodyBIndex[threadIndexInWarp];

            const uint slabId = batches[k].slabId[threadIndexInWarp];

            // zero-init: only read on the joint path, where they are loaded below
            PxgSolverTxIData iData0 = {};
            PxgSolverTxIData iData1 = {};
            if (!pxgIsContactType(constraintType) && threadIndexInWarp < descStride)
            {
                // CUDA stages these through warp shuffles (loadTxInertia)
                // purely for memory coalescing; direct loads are semantically
                // identical.  Gated on the live lanes: the tail lanes'
                // bodyA/BIndex is uninitialized batch memory.  Only the 1D-joint
                // path reads txInertia (engine encoding: contact = eCONTACT(1) /
                // eARTICULATION_CONTACT(3), joint = 0/2).
                iData0 = iData[bodyAIndex];
                iData1 = iData[bodyBIndex];
            }

            // Pull out shared memory into registers to solve constraints
            if (threadIndexInWarp < descStride)
            {
                const uint readIndexA = bodyAIndex + slabId * totalBodiesIncKinematics;
                const uint readIndexB = bodyBIndex + slabId * totalBodiesIncKinematics;

                float3 linVel0 = wiLoadLinVel(readIndexA);
                float3 angVel0 = wiLoadAngVel(readIndexA);
                const float3 linDelta0 = wiLoadLinDelta(readIndexA);
                const float3 angDelta0 = wiLoadAngDelta(readIndexA);

                float3 linVel1 = wiLoadLinVel(readIndexB);
                float3 angVel1 = wiLoadAngVel(readIndexB);
                const float3 linDelta1 = wiLoadLinDelta(readIndexB);
                const float3 angDelta1 = wiLoadAngDelta(readIndexB);

                float curRef0 = 1.0f;
                float curRef1 = 1.0f;

                const uint numTotalBodies = bodyOffset + numDynamicBodies + island.mArticulationCount;

                if (bodyAIndex >= bodyOffset)
                {
                    // Counting the number of active slabs
                    curRef0 = float(countActiveSlabsBda(bodyAIndex, desc.numSlabs, numTotalBodies,
                                                        encodedReferenceCount));
                }

                if (bodyBIndex >= bodyOffset)
                {
                    // Counting the number of active slabs
                    curRef1 = float(countActiveSlabsBda(bodyBIndex, desc.numSlabs, numTotalBodies,
                                                        encodedReferenceCount));
                }

                if (pxgIsContactType(constraintType))
                    solveContactBlockTGSBda(batches, k, linVel0, angVel0, linVel1, angVel1,
                                            linDelta0, angDelta0, linDelta1, angDelta1,
                                            threadIndexInWarp, contactHeaders, frictionHeaders,
                                            contactPoints, frictionsBuf, gWholeIslandAbi.elapsedTime,
                                            gWholeIslandAbi.minPen, residualAccumulationEnabled, error,
                                            curRef0, curRef1);
                else
                    solve1DBlockTGSBda(batches, k, linVel0, angVel0, linVel1, angVel1,
                                       linDelta0, angDelta0, linDelta1, angDelta1,
                                       threadIndexInWarp, jointHeaders, jointRowsCon, iData0, iData1,
                                       gWholeIslandAbi.elapsedTime,
                                       desc.contactErrorAccumulator_mCounter >= 0, curRef0, curRef1);

                wiStoreLinVel(readIndexA, linVel0);
                wiStoreAngVel(readIndexA, angVel0);

                wiStoreLinVel(readIndexB, linVel1);
                wiStoreAngVel(readIndexB, angVel1);
            }
        }

        GroupMemoryBarrierWithGroupSync();

        startIndex = endIndex;
    }
    } // if (iter_blockConstraintBatch != 0)

    // Get here, write out to accumulation buffers...

    for (uint index = threadIdxX; index < (numDynamicBodies * numSlabs); index += 256)
    {
        const uint bodyId = index / numSlabs;
        const uint slabId = index % numSlabs;
        const uint bodyIndex = index;

        const uint remappedBodyIndex = solverBodyReferences[bodyIndex]; // PxgSolverReferences

        if (remappedBodyIndex != 0xFFFFFFFFu)
        {
            const uint velIndex = averageOutputOffsets + computeAverageBodyOutputIndex(bodyIndex);

            const uint readIndex = bodyId + bodyOffset + slabId * totalBodiesIncKinematics;

            const float3 linVel0 = wiLoadLinVel(readIndex);
            const float3 angVel0 = wiLoadAngVel(readIndex);
            const float3 linDelta0 = wiLoadLinDelta(readIndex);
            const float3 angDelta0 = wiLoadAngDelta(readIndex);

            bodyVelocities[velIndex] = float4(linVel0, angVel0.x);
            bodyVelocities[velIndex + 32] = float4(angVel0.y, angVel0.z, linDelta0.x, linDelta0.y);
            bodyVelocities[velIndex + 64] = float4(linDelta0.z, angDelta0.x, angDelta0.y, angDelta0.z);
        }
    }

    // PxgErrorAccumulator::accumulateErrorGlobalFullWarp(): a per-warp
    // sum/sum/max butterfly reduction, then lane 0 of each warp accumulates
    // into solverDesc->contactErrorAccumulator.  The reduction runs
    // unconditionally so every barrier sits in uniform control flow (a
    // disabled launch reduces zeros and skips the atomics — identical
    // observable behavior to the CUDA `if (residualAccumulationEnabled)`
    // around the whole call).  Scratch: the staging arrays are dead now that
    // the writeout loop above has completed.
    GroupMemoryBarrierWithGroupSync();
    shWholeIslandLinVel[threadIdxX] = error.sumOfSquares;
    shWholeIslandAngVel[threadIdxX] = error.maxError;
    shWholeIslandErrCounter[threadIdxX] = uint(error.counter);
    GroupMemoryBarrierWithGroupSync();

    // the __shfl_xor butterfly of warpReduction<OP, T, 5> (radius 16..1)
    for (uint radius = 16; radius > 0; radius >>= 1)
    {
        const uint partner = threadIdxX ^ radius; // stays inside the warp's 32 lanes
        const float s = shWholeIslandLinVel[partner];
        const float m = shWholeIslandAngVel[partner];
        const uint c = shWholeIslandErrCounter[partner];
        GroupMemoryBarrierWithGroupSync();
        shWholeIslandLinVel[threadIdxX] += s;
        shWholeIslandAngVel[threadIdxX] = max(shWholeIslandAngVel[threadIdxX], m);
        shWholeIslandErrCounter[threadIdxX] += c;
        GroupMemoryBarrierWithGroupSync();
    }

    if (residualAccumulationEnabled && threadIndexInWarp == 0)
    {
        const float sumSq = shWholeIslandLinVel[threadIdxX];
        const uint count = shWholeIslandErrCounter[threadIdxX];
        const float maxErr = shWholeIslandAngVel[threadIdxX];

        // atomicAdd(&mErrorSumOfSquares, s): float atomic add emulated with a
        // CAS loop on the raw bits (the mirror holds the field as uint)
        uint assumed = descPtr[0].contactErrorAccumulator_mErrorSumOfSquares;
        for (;;)
        {
            const uint newBits = asuint(asfloat(assumed) + sumSq);
            uint old;
            InterlockedCompareExchange(descPtr[0].contactErrorAccumulator_mErrorSumOfSquares, assumed,
                                       newBits, old);
            if (old == assumed)
                break;
            assumed = old;
        }

        // atomicAdd(&mCounter, count)
        int oldCount;
        InterlockedAdd(descPtr[0].contactErrorAccumulator_mCounter, int(count), oldCount);

        // if (maxErr > mMaxError) AtomicMax(&mMaxError, maxErr) — the CUDA
        // AtomicMax(float*) is itself a CAS loop; the pre-test read is
        // non-atomic exactly like the .cu.
        if (maxErr > asfloat(descPtr[0].contactErrorAccumulator_mMaxError))
        {
            assumed = descPtr[0].contactErrorAccumulator_mMaxError;
            for (;;)
            {
                const uint newBits = asuint(max(maxErr, asfloat(assumed)));
                uint old;
                InterlockedCompareExchange(descPtr[0].contactErrorAccumulator_mMaxError, assumed,
                                           newBits, old);
                if (old == assumed)
                    break;
                assumed = old;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// solveBlockUnified (P5 Step 5 tranche A, BDA engine ABI) — the per-partition
// dynamic-vs-dynamic rigid contact/joint velocity solve
// ---------------------------------------------------------------------------
//
// Port of solveBlockUnified() (rigid path -> solveBlockPartitionTGSInternal())
// in solverMultiBlockTGS.cu.  The engine launches it ONCE PER PARTITION inside
// solvePartitions() (PxgTGSCudaSolverCore.cpp), passing the current partition
// index as the partitionIndex argument; each launch solves exactly that one
// partition's rigid batches.  Signature:
//   (PxgSolverCoreDesc*, PxgSolverSharedDesc<IterativeSolveDataTGS>*,
//    PxU32 islandIndex, PxU32 partitionIndex, bool doFriction, PxReal elapsedTime,
//    PxReal minPen, PxgArticulationCoreDesc*) -> argSizes {8,8,4,4,1,4,4,8} =
//   48-byte blob.
//
// Thread layout mirrors the CUDA launch (numThreadsPerWarp = WARP_SIZE = 32,
// numWarpsPerBlock = COMPUTE_UNCONSTRAINED_VELOCITES(64)/32 = 2, block =
// SOLVE_BLOCK_PARTITION = 64 threads):
//   numthreads = (32, 2, 1); globalWarpIndex = groupId.x * 2 + groupThreadId.y;
//   lane = groupThreadId.x.  The CUDA grid (maxBlocks, blockY, 1) sizes maxBlocks
//   so gridDim.x * blockDim.y >= batches-in-partition, and the .cu body is a
//   single `if (k < endIndex)` (its strided loop is commented out).  We use the
//   equivalent strided loop, stride = WorkgroupCount().x * 2 — a
//   behaviour-preserving superset: within one partition no two batches share a
//   body, so processing several batches per warp is safe, and it makes the batch
//   loads control-dependent on a real loop guard (llvmpipe zero-trip safety,
//   same technique as solveWholeIslandTGS).
//
// RIGID PATH ONLY (this tranche): blockIdx.y == 0.  The engine only launches
// blockY == 2 when nbArtiBlocks > 0, which a rigid-only scene never triggers, so
// the articulation branch (groupId.y != 0 -> artiSolveBlockPartitionTGSInternal)
// is a DEFERRED no-op early-return.
//
// Body-velocity staging is the per-partition solverBodyVelPool layout
// (readIndex = k * 192 + lane; writeback to remappedBodyA/BIndex), identical to
// the bound-buffer solveBlockPartitionTGS above and to the .cu — NOT the deltaV/
// slab staging of solveWholeIslandTGS.  Descriptors are reached through Ptr<>
// (BDA); the contact-vs-joint discriminator is pxgIsContactType (engine
// PxgConstraintType encoding).  Like the .cu, the contact-residual statistics
// are accumulated into solverDesc->contactErrorAccumulator when residual
// reporting is enabled (accumulateErrorGlobalFullWarp).
// P8.104: PxgErrorAccumulator::accumulateErrorGlobalFullWarp(), extracted so
// BOTH branches of solveBlockUnified (rigid and articulation) share it: a
// per-warp sum/sum/max butterfly reduction through the groupshared scratch,
// then lane 0 of each warp accumulates into solverDesc->contactErrorAccumulator.
// The reduction runs unconditionally (uniform control flow around the
// barriers); a disabled launch skips only the atomics.
void solveUnifiedErrorTail(uint localIdx, uint threadIndexInWarp, float errSumSq, float errMax,
                           uint errCount, bool residualAccumulationEnabled,
                           Ptr<PxgSolverCoreDescGpu> descPtr)
{
    GroupMemoryBarrierWithGroupSync();
    shUnifiedErrSum[localIdx] = errSumSq;
    shUnifiedErrMax[localIdx] = errMax;
    shUnifiedErrCount[localIdx] = errCount;
    GroupMemoryBarrierWithGroupSync();

    for (uint radius = 16u; radius > 0u; radius >>= 1)
    {
        const uint partner = localIdx ^ radius; // stays inside the warp's 32 lanes
        const float s = shUnifiedErrSum[partner];
        const float m = shUnifiedErrMax[partner];
        const uint c = shUnifiedErrCount[partner];
        GroupMemoryBarrierWithGroupSync();
        shUnifiedErrSum[localIdx] += s;
        shUnifiedErrMax[localIdx] = max(shUnifiedErrMax[localIdx], m);
        shUnifiedErrCount[localIdx] += c;
        GroupMemoryBarrierWithGroupSync();
    }

    if (residualAccumulationEnabled && threadIndexInWarp == 0u)
    {
        const float sumSq = shUnifiedErrSum[localIdx];
        const uint count = shUnifiedErrCount[localIdx];
        const float maxErr = shUnifiedErrMax[localIdx];

        // atomicAdd(&mErrorSumOfSquares, s): float atomic add via CAS on raw bits
        uint assumed = descPtr[0].contactErrorAccumulator_mErrorSumOfSquares;
        for (;;)
        {
            const uint newBits = asuint(asfloat(assumed) + sumSq);
            uint old;
            InterlockedCompareExchange(descPtr[0].contactErrorAccumulator_mErrorSumOfSquares, assumed,
                                       newBits, old);
            if (old == assumed)
                break;
            assumed = old;
        }

        int oldCount;
        InterlockedAdd(descPtr[0].contactErrorAccumulator_mCounter, int(count), oldCount);

        if (maxErr > asfloat(descPtr[0].contactErrorAccumulator_mMaxError))
        {
            assumed = descPtr[0].contactErrorAccumulator_mMaxError;
            for (;;)
            {
                const uint newBits = asuint(max(maxErr, asfloat(assumed)));
                uint old;
                InterlockedCompareExchange(descPtr[0].contactErrorAccumulator_mMaxError, assumed,
                                           newBits, old);
                if (old == assumed)
                    break;
                assumed = old;
            }
        }
    }
}

struct SolveBlockUnifiedAbiParams
{
    uint64_t solverCoreDesc; // @0
    uint64_t sharedDesc;     // @8
    uint islandIndex;        // @16
    uint partitionIndex;     // @20
    uint doFrictionWord;     // @24: byte 0 = doFriction (unused by the rigid body, exactly like the .cu; bytes 25..27 padding)
    float elapsedTime;       // @28
    float minPen;            // @32
    // bytes 36..39: padding to 8-align artiDesc
    uint64_t artiDesc;       // @40: PxgArticulationCoreDesc* (articulation path deferred; unused here)
}                            // = 48 (maxAlign 8)
[[vk::push_constant]]
ConstantBuffer<SolveBlockUnifiedAbiParams> gSolveUnifiedAbi;

// per-warp residual-reduction scratch: block = 64 -> 2 warps of 32 lanes
// (mirrors solveStaticBlockTGS's accumulateErrorGlobalFullWarp scratch).
groupshared float shUnifiedErrSum[64];
groupshared float shUnifiedErrMax[64];
groupshared uint  shUnifiedErrCount[64];

[shader("compute")]
[numthreads(32, 2, 1)]
void solveBlockUnified(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    // P8.104: the articulation branch (blockIdx.y != 0 ->
    // artiSolveBlockPartitionTGSInternal, solverMultiBlockTGS.cu:756) lives in
    // npArtiSolvePartition.slang; the shared error tail keeps every barrier in
    // uniform control flow (the CUDA's per-lane early returns become guards
    // inside the module function, contributing zeros to the reduction).
    if (groupId.y != 0)
    {
        float aErrSumSq;
        float aErrMax;
        uint aErrCount;
        bool aResidual;
        aspSolveBlockPartitionTGS(gSolveUnifiedAbi.solverCoreDesc, gSolveUnifiedAbi.sharedDesc,
                                  gSolveUnifiedAbi.islandIndex, gSolveUnifiedAbi.partitionIndex,
                                  gSolveUnifiedAbi.elapsedTime, gSolveUnifiedAbi.minPen,
                                  gSolveUnifiedAbi.artiDesc,
                                  groupId.x * 2u + groupThreadId.y, groupThreadId.x,
                                  aErrSumSq, aErrMax, aErrCount, aResidual);
        solveUnifiedErrorTail(groupThreadId.y * 32u + groupThreadId.x, groupThreadId.x,
                              aErrSumSq, aErrMax, aErrCount, aResidual,
                              Ptr<PxgSolverCoreDescGpu>(gSolveUnifiedAbi.solverCoreDesc));
        return;
    }

    Ptr<PxgSolverCoreDescGpu> descPtr = Ptr<PxgSolverCoreDescGpu>(gSolveUnifiedAbi.solverCoreDesc);
    PxgSolverCoreDescGpu desc = descPtr[0];
    PxgSolverSharedDescTGSGpu sharedDesc =
        (Ptr<PxgSolverSharedDescTGSGpu>(gSolveUnifiedAbi.sharedDesc))[0];
    PxgIslandContextGpu island =
        (Ptr<PxgIslandContextGpu>(desc.islandContextPool))[gSolveUnifiedAbi.islandIndex];

    const uint partitionIndex = gSolveUnifiedAbi.partitionIndex;
    const uint startPartitionIndex = island.mStartPartitionIndex;

    const uint threadIndexInWarp = groupThreadId.x;               // lane
    const uint numWarpsPerBlock = 2u;                            // numthreads.y
    const uint globalWarpIndex = groupId.x * numWarpsPerBlock + groupThreadId.y;
    const uint numWarpsInGrid = WorkgroupCount().x * numWarpsPerBlock;
    const uint localIdx = groupThreadId.y * 32u + groupThreadId.x; // 0..63 (reduction index)

    Ptr<float4> bodyVelocities = Ptr<float4>(sharedDesc.iter_solverBodyVelPool);
    Ptr<PxgBlockConstraintBatch> batches = Ptr<PxgBlockConstraintBatch>(sharedDesc.iter_blockConstraintBatch);
    Ptr<PxgTGSBlockSolverContactHeader> contactHeaders =
        Ptr<PxgTGSBlockSolverContactHeader>(sharedDesc.iter_blockContactHeaders);
    Ptr<PxgTGSBlockSolverFrictionHeader> frictionHeaders =
        Ptr<PxgTGSBlockSolverFrictionHeader>(sharedDesc.iter_blockFrictionHeaders);
    Ptr<PxgTGSBlockSolverContactPoint> contactPoints =
        Ptr<PxgTGSBlockSolverContactPoint>(sharedDesc.iter_blockContactPoints);
    Ptr<PxgTGSBlockSolverContactFriction> frictionsBuf =
        Ptr<PxgTGSBlockSolverContactFriction>(sharedDesc.iter_blockFrictions);
    Ptr<PxgTGSBlockSolverConstraint1DHeader> jointHeaders =
        Ptr<PxgTGSBlockSolverConstraint1DHeader>(sharedDesc.iter_blockJointConstraintHeaders);
    Ptr<PxgTGSBlockSolverConstraint1DCon> jointRowsCon =
        Ptr<PxgTGSBlockSolverConstraint1DCon>(sharedDesc.iter_blockJointConstraintRowsCon);
    Ptr<PxgSolverTxIData> iData = Ptr<PxgSolverTxIData>(desc.solverBodyTxIDataPool);
    Ptr<uint> constraintsPerPartition = Ptr<uint>(desc.constraintsPerPartition);
    Ptr<uint> encodedReferenceCount = Ptr<uint>(sharedDesc.iter_solverEncodedReferenceCount);

    const uint bodyOffset = island.mBodyStartIndex;
    const uint numDynamicBodies = island.mBodyCount; // nbBodies minus offset!
    const uint numTotalBodies = bodyOffset + numDynamicBodies + island.mArticulationCount;

    const bool residualAccumulationEnabled = desc.contactErrorAccumulator_mCounter >= 0;
    PxgErrorAccumulatorSlang error = { 0.0f, 0, 0.0f };

    // Null / empty-partition guard (llvmpipe): a constraint-free island leaves
    // the block-constraint pointer null/unallocated and mNumPartitions == 0; the
    // SPIR-V optimizer would speculatively hoist the first batches[] load out of
    // the (then zero-trip) batch loop and fault.  The uniform scalar guard makes
    // every dependent load control-dependent on it — semantically exact: no batch
    // buffer / no partitions == no batches to solve.  Same technique as
    // solveWholeIslandTGS's mNumPartitions / iter_blockConstraintBatch guard.  The
    // residual reduction below runs unconditionally (uniform control flow around
    // its barriers) regardless of this guard.
    if (sharedDesc.iter_blockConstraintBatch != 0 && island.mNumPartitions != 0 &&
        partitionIndex < island.mNumPartitions)
    {
        // Single-partition batch range (solveBlockPartitionTGSInternal):
        //   startIndex = partitionIndex == 0 ? island.mBatchStartIndex
        //              : constraintsPerPartition[partitionIndex + startPartitionIndex - 1];
        //   endIndex   = constraintsPerPartition[partitionIndex + startPartitionIndex];
        // CLAMP THE INDEX, NOT JUST THE VALUE (P8.96c): the load sits on the
        // partitionIndex != 0 arm but is speculated on the == 0 dispatch too,
        // where startPartitionIndex - 1u can underflow to 0xFFFFFFFF.  The
        // clamp target [partitionIndex + startPartitionIndex] is in bounds
        // because the endIndex line reads it unconditionally.
        const uint prevIdx = (partitionIndex == 0)
            ? partitionIndex + startPartitionIndex
            : partitionIndex + startPartitionIndex - 1u;
        const uint prevCpp = constraintsPerPartition[prevIdx];
        const uint startIndex = (partitionIndex == 0) ? island.mBatchStartIndex : prevCpp;
        const uint endIndex = constraintsPerPartition[partitionIndex + startPartitionIndex];

        for (uint k = startIndex + globalWarpIndex; k < endIndex; k += numWarpsInGrid)
        {
            const uint descStride = batches[k].descStride_constraintType & 0xFFFFu;
            const uint constraintType = batches[k].descStride_constraintType >> 16;

            if (threadIndexInWarp >= descStride)
                continue; // tail lanes carry uninitialized batch memory

            const uint bodyAIndex = batches[k].bodyAIndex[threadIndexInWarp];
            const uint bodyBIndex = batches[k].bodyBIndex[threadIndexInWarp];

            // zero-init: only read on the 1D-joint path (engine encoding: contact
            // = eCONTACT(1)/eARTI_CONTACT(3), joint = 0/2).  Direct loads replace
            // the CUDA warp-shuffle loadTxInertia (memory-coalescing only, gated
            // on the live lanes since tail lanes' bodyA/BIndex is uninitialized).
            PxgSolverTxIData iData0 = {};
            PxgSolverTxIData iData1 = {};
            if (!pxgIsContactType(constraintType))
            {
                iData0 = iData[bodyAIndex];
                iData1 = iData[bodyBIndex];
            }

            const uint readIndex = k * 192u + threadIndexInWarp;

            float4 vec0 = bodyVelocities[readIndex];
            float4 vec1 = bodyVelocities[readIndex + 32u];
            float4 vec2 = bodyVelocities[readIndex + 64u];
            float4 vec3 = bodyVelocities[readIndex + 96u];
            float4 vec4 = bodyVelocities[readIndex + 128u];
            float4 vec5 = bodyVelocities[readIndex + 160u];

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
            if (bodyAIndex >= bodyOffset)
                curRef0 = float(countActiveSlabsBda(bodyAIndex, desc.numSlabs, numTotalBodies,
                                                    encodedReferenceCount));
            if (bodyBIndex >= bodyOffset)
                curRef1 = float(countActiveSlabsBda(bodyBIndex, desc.numSlabs, numTotalBodies,
                                                    encodedReferenceCount));

            if (pxgIsContactType(constraintType))
                solveContactBlockTGSBda(batches, k, linVel0, angVel0, linVel1, angVel1,
                                        linDelta0, angDelta0, linDelta1, angDelta1,
                                        threadIndexInWarp, contactHeaders, frictionHeaders,
                                        contactPoints, frictionsBuf, gSolveUnifiedAbi.elapsedTime,
                                        gSolveUnifiedAbi.minPen, residualAccumulationEnabled, error,
                                        curRef0, curRef1);
            else
                solve1DBlockTGSBda(batches, k, linVel0, angVel0, linVel1, angVel1,
                                   linDelta0, angDelta0, linDelta1, angDelta1,
                                   threadIndexInWarp, jointHeaders, jointRowsCon, iData0, iData1,
                                   gSolveUnifiedAbi.elapsedTime, residualAccumulationEnabled,
                                   curRef0, curRef1);

            vec0 = float4(linVel0, angVel0.x);
            vec1 = float4(angVel0.y, angVel0.z, linDelta0.x, linDelta0.y);
            vec2 = float4(linDelta0.z, angDelta0.x, angDelta0.y, angDelta0.z);
            vec3 = float4(linVel1, angVel1.x);
            vec4 = float4(angVel1.y, angVel1.z, linDelta1.x, linDelta1.y);
            vec5 = float4(linDelta1.z, angDelta1.x, angDelta1.y, angDelta1.z);

            const uint outputA = batches[k].remappedBodyAIndex[threadIndexInWarp];
            const uint outputB = batches[k].remappedBodyBIndex[threadIndexInWarp];

            bodyVelocities[outputA] = vec0;
            bodyVelocities[outputA + 32u] = vec1;
            bodyVelocities[outputA + 64u] = vec2;
            bodyVelocities[outputB] = vec3;
            bodyVelocities[outputB + 32u] = vec4;
            bodyVelocities[outputB + 64u] = vec5;
        }
    }

    solveUnifiedErrorTail(localIdx, threadIndexInWarp, error.sumOfSquares, error.maxError,
                          uint(error.counter), residualAccumulationEnabled, descPtr);
}

// ---------------------------------------------------------------------------
// markActiveSlabTGS (P5 Step 5, BDA engine ABI) — per-partition active-slab
// marking for the mass-splitting reference counts
// ---------------------------------------------------------------------------
//
// Port of markActiveSlabTGS() (rigid path -> markActiveSlab_rigidBodyTGS()) in
// solverMultiBlockTGS.cu.  precomputeReferenceCount() (PxgSolverCore.cpp)
// launches it ONCE per island BEFORE the TGS velocity iterations, passing the
// LAST partition index so a single launch marks every partition's batches (the
// batch range is [mBatchStartIndex, constraintsPerPartition[lastPartition])).
// For each batch lane it decides whether that constraint's slab is "active"
// (a joint is always active; a contact is active iff checkActiveContactBlockTGS
// says it would still push) and atomicOr's the slab bit into
// solverEncodedReferenceCount[bodyId], which countActiveSlabsBda() later reads
// to scale invMass/response for mass splitting.  Without it, countActiveSlabsBda
// falls back to max(1, 0) == 1 (no splitting) — correct for single-contact
// bodies, which is why the box/stack scenes rest without this launch, but wrong
// once a body carries several active contacts across slabs.
//
// Signature (note the arg ORDER differs from solveBlockUnified: minPen precedes
// elapsedTime here):
//   (PxgSolverCoreDesc*, PxgSolverSharedDesc<IterativeSolveDataTGS>*,
//    PxU32 islandIndex, PxU32 lastPartition, PxReal minPen, PxReal elapsedTime,
//    PxgArticulationCoreDesc*) -> argSizes {8,8,4,4,4,4,8} = 40-byte blob.
//
// Thread layout mirrors the CUDA launch (numThreadsPerWarp = WARP_SIZE = 32,
// numWarpsPerBlock = COMPUTE_UNCONSTRAINED_VELOCITES(64)/32 = 2):
//   numthreads = (32, 2, 1); globalWarpIndex = groupId.x * 2 + groupThreadId.y;
//   lane = groupThreadId.x; batchIndex = mBatchStartIndex + globalWarpIndex.
// The CUDA body is a single `if (batchIndex < endIndex)` (no strided loop), so
// the grid.x is sized to cover the whole batch range in one wave — mirrored 1:1.
//
// RIGID PATH ONLY (this milestone): groupId.y == 0.  The engine only launches
// blockY == 2 when the last partition has articulation batches, which a
// rigid-only scene never produces, so markActiveSlab_articulationTGS is a
// DEFERRED no-op early-return.
//
// Body-velocity staging is the accumulated deltaV layout (3 float4 per body at
// accumulatedBodyDeltaVOffset + bodyId + k*totalBodiesIncKinematics), identical
// to computeAverageSolverBodyVelocityTGS / the .cu — NOT the per-partition
// k*192+lane layout of solveBlockUnified.
struct MarkActiveSlabTGSAbiParams
{
    uint64_t solverCoreDesc; // @0
    uint64_t sharedDesc;     // @8
    uint islandIndex;        // @16
    uint lastPartition;      // @20
    float minPen;            // @24
    float elapsedTime;       // @28
    uint64_t artiDesc;       // @32: PxgArticulationCoreDesc* (articulation path deferred; unused here)
}                            // = 40 (maxAlign 8; @32 already 8-aligned, no padding)
[[vk::push_constant]]
ConstantBuffer<MarkActiveSlabTGSAbiParams> gMarkActiveSlabAbi;

[shader("compute")]
[numthreads(32, 2, 1)]
void markActiveSlabTGS(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    // Deferred articulation path: the engine only dispatches groupId.y != 0
    // (blockY == 2) when the last partition carries articulation batches, which
    // a rigid-only scene never produces.
    if (groupId.y != 0)
        return;

    PxgSolverCoreDescGpu desc = (Ptr<PxgSolverCoreDescGpu>(gMarkActiveSlabAbi.solverCoreDesc))[0];
    PxgSolverSharedDescTGSGpu sharedDesc =
        (Ptr<PxgSolverSharedDescTGSGpu>(gMarkActiveSlabAbi.sharedDesc))[0];
    PxgIslandContextGpu island =
        (Ptr<PxgIslandContextGpu>(desc.islandContextPool))[gMarkActiveSlabAbi.islandIndex];
    // The .cu reads mBodyCount / mArticulationCount from islandContextPool[0]
    // (islandContextPool->…), NOT islandContextPool[islandIndex]; mirror that
    // exactly.  For the single-island scenes islandIndex == 0 so it is moot.
    PxgIslandContextGpu island0 = (Ptr<PxgIslandContextGpu>(desc.islandContextPool))[0];

    const uint startPartitionIndex = island.mStartPartitionIndex;
    const uint threadIndexInWarp = groupThreadId.x;               // lane
    const uint numWarpsPerBlock = 2u;                            // numthreads.y
    const uint globalWarpIndex = groupId.x * numWarpsPerBlock + groupThreadId.y;

    Ptr<float4> bodyVelocities = Ptr<float4>(sharedDesc.iter_solverBodyVelPool);
    Ptr<PxgBlockConstraintBatch> batches = Ptr<PxgBlockConstraintBatch>(sharedDesc.iter_blockConstraintBatch);
    Ptr<PxgTGSBlockSolverContactHeader> contactHeaders =
        Ptr<PxgTGSBlockSolverContactHeader>(sharedDesc.iter_blockContactHeaders);
    Ptr<PxgTGSBlockSolverContactPoint> contactPoints =
        Ptr<PxgTGSBlockSolverContactPoint>(sharedDesc.iter_blockContactPoints);
    Ptr<uint> encodedReferenceCount = Ptr<uint>(sharedDesc.iter_solverEncodedReferenceCount);
    Ptr<uint> constraintsPerPartition = Ptr<uint>(desc.constraintsPerPartition);

    const uint bodyOffset = island.mBodyStartIndex;
    const uint outputOffset = desc.accumulatedBodyDeltaVOffset;
    const uint numDynamicBodies = island0.mBodyCount; // nbBodies minus offset!
    const uint totalBodiesIncKinematics = numDynamicBodies + bodyOffset;
    const uint numArticulations = island0.mArticulationCount;
    const uint numTotalBodies = bodyOffset + numDynamicBodies + numArticulations;

    // Null / empty-island guard (llvmpipe): precomputeReferenceCount only
    // launches this when mNumPartitions != 0, but the constraint pointers are
    // still null for a constraint-free island.  The uniform scalar guard makes
    // every dependent batch load control-dependent on it, blocking the SPIR-V
    // optimizer's speculative hoist of the first batches[] load past the
    // (then dead) `batchIndex < endIndex` test — same technique as
    // solveBlockUnified / solveWholeIslandTGS.
    if (sharedDesc.iter_blockConstraintBatch != 0 && island.mNumPartitions != 0)
    {
        const uint startIndex = island.mBatchStartIndex;
        const uint endIndex = constraintsPerPartition[gMarkActiveSlabAbi.lastPartition + startPartitionIndex];
        const uint batchIndex = startIndex + globalWarpIndex;

        if (batchIndex < endIndex)
        {
            const uint descStride = batches[batchIndex].descStride_constraintType & 0xFFFFu;
            const uint constraintType = batches[batchIndex].descStride_constraintType >> 16;

            if (threadIndexInWarp < descStride)
            {
                const uint slabId = batches[batchIndex].slabId[threadIndexInWarp];
                const uint bodyIdA = batches[batchIndex].bodyAIndex[threadIndexInWarp];
                const uint bodyIdB = batches[batchIndex].bodyBIndex[threadIndexInWarp];

                const uint slabIndexOffset = (slabId / 32u) * numTotalBodies;
                const uint encodedSlabIndex = (1u << (slabId % 32u));

                bool isActiveSlab = false;

                if (!pxgIsContactType(constraintType))
                {
                    // For joint constraints, simply mark slabs as active for
                    // efficiency (does not significantly change the count).
                    isActiveSlab = true;
                }
                else
                {
                    const uint finalIdA = outputOffset + bodyIdA;
                    const uint finalIdB = outputOffset + bodyIdB;

                    const float4 vec0 = bodyVelocities[finalIdA];
                    const float4 vec1 = bodyVelocities[finalIdA + totalBodiesIncKinematics];
                    const float4 vec2 = bodyVelocities[finalIdA + 2u * totalBodiesIncKinematics];

                    const float4 vec3 = bodyVelocities[finalIdB];
                    const float4 vec4 = bodyVelocities[finalIdB + totalBodiesIncKinematics];
                    const float4 vec5 = bodyVelocities[finalIdB + 2u * totalBodiesIncKinematics];

                    const float3 linVel0 = vec0.xyz;
                    const float3 angVel0 = float3(vec0.w, vec1.x, vec1.y);
                    const float3 linDelta0 = float3(vec1.z, vec1.w, vec2.x);
                    const float3 angDelta0 = float3(vec2.y, vec2.z, vec2.w);

                    const float3 linVel1 = vec3.xyz;
                    const float3 angVel1 = float3(vec3.w, vec4.x, vec4.y);
                    const float3 linDelta1 = float3(vec4.z, vec4.w, vec5.x);
                    const float3 angDelta1 = float3(vec5.y, vec5.z, vec5.w);

                    isActiveSlab = checkActiveContactBlockTGSBda(
                        batches, batchIndex, linVel0, angVel0, linVel1, angVel1,
                        linDelta0, angDelta0, linDelta1, angDelta1, threadIndexInWarp,
                        contactHeaders, contactPoints,
                        gMarkActiveSlabAbi.elapsedTime, gMarkActiveSlabAbi.minPen);
                }

                // Encode which slab is active in a 32-bit index; countActiveSlabsBda
                // popcounts these words to recover the reference count.
                if (isActiveSlab && bodyIdA >= bodyOffset)
                {
                    uint prev;
                    InterlockedOr(encodedReferenceCount[slabIndexOffset + bodyIdA], encodedSlabIndex, prev);
                }
                if (isActiveSlab && bodyIdB >= bodyOffset)
                {
                    uint prev;
                    InterlockedOr(encodedReferenceCount[slabIndexOffset + bodyIdB], encodedSlabIndex, prev);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// solveStaticBlockTGS + propagateStaticSolverBodyVelocitiesTGS
// (P5 Step 4 BDA engine ABI) — the dynamic-vs-static rigid contact/joint solve
// ---------------------------------------------------------------------------

// solveStaticBlockTGS: port of solveStaticBlockTGS() in solverMultiBlockTGS.cu.
// One thread per (dynamic body, static slab); each thread walks that body's
// static contact and joint batches (packed into the same block-constraint
// buffers, addressed through the per-body mRigidStatic*StartIndices) and writes
// the resulting body velocity to the per-slab tempStaticBodyOutputs scratch
// that propagateStaticSolverBodyVelocitiesTGS later averages back into the
// deltaV pool.  block = SOLVE_BLOCK_PARTITION (64) threads, grid =
// nbBlocksRequired (WorkgroupCount().x).
//
// argSizes {8,8,4,4,4,4,4} -> 40-byte blob (8@0, 8@8, 4@16, 4@20, 4@24, 4@28,
// 4@32, round to maxAlign 8 = 40).
//
// Deviations from the .cu (behaviour-preserving):
//   - the block-packed lane index is warpScanExclusive(batch.mask,
//     threadIndexInWarp) = popcount(mask & ((1<<lane)-1)); it is a pure
//     function of the STORED mask and the lane (no cross-lane ballot), so it is
//     llvmpipe-safe;
//   - ref0/ref1 are 1 (mass-splitting is not used for static interactions);
//   - a static body B has no txInertia entry, so iData1 is the identity
//     transform with zero inertia, exactly as the .cu builds it
//     (PxMat33(PxZero) / PxTransform(PxIdentity));
//   - accumulateErrorGlobalFullWarp(): the CUDA __shfl_xor per-warp butterfly
//     becomes the same butterfly through a 64-entry groupshared scratch
//     (block = 2 warps), reduced under uniform GroupMemoryBarrierWithGroupSync
//     barriers; only the atomic tail is gated on residualAccumulationEnabled,
//     and CUDA's float atomicAdd / AtomicMax become CAS loops on the raw bits.
struct SolveStaticBlockTGSAbiParams
{
    uint64_t solverCoreDesc;   // @0
    uint64_t sharedDesc;       // @8
    uint islandIndex;          // @16
    uint nbStaticSlabs;        // @20
    uint maxStaticPartitions;  // @24
    float elapsedTime;         // @28
    float minPen;              // @32
}                              // = 40 (round to maxAlign 8)
[[vk::push_constant]]
ConstantBuffer<SolveStaticBlockTGSAbiParams> gStaticSolveAbi;

// per-warp residual-reduction scratch: block = 64 -> 2 warps of 32 lanes.
groupshared float shStaticErrSum[64];
groupshared float shStaticErrMax[64];
groupshared uint  shStaticErrCount[64];

[shader("compute")]
[numthreads(64, 1, 1)]
void solveStaticBlockTGS(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgSolverCoreDescGpu> descPtr = Ptr<PxgSolverCoreDescGpu>(gStaticSolveAbi.solverCoreDesc);
    PxgSolverCoreDescGpu desc = descPtr[0];
    PxgSolverSharedDescTGSGpu sharedDesc =
        (Ptr<PxgSolverSharedDescTGSGpu>(gStaticSolveAbi.sharedDesc))[0];
    PxgIslandContextGpu island =
        (Ptr<PxgIslandContextGpu>(desc.islandContextPool))[gStaticSolveAbi.islandIndex];

    Ptr<float4> bodyVelocities = Ptr<float4>(sharedDesc.iter_solverBodyVelPool);
    Ptr<float4> bodyOutVelocities = Ptr<float4>(sharedDesc.iter_tempStaticBodyOutputs);

    const uint numDynamicBodies = island.mBodyCount;
    const uint numDynamicBodiesRounded = (island.mBodyCount + 31u) & ~31u; // multiple of 32
    const uint bodyOffset = island.mBodyStartIndex;
    const uint deltaVOffset = desc.accumulatedBodyDeltaVOffset;
    const uint totalBodiesIncKinematics = numDynamicBodies + bodyOffset;

    const uint nbStaticSlabs = gStaticSolveAbi.nbStaticSlabs;
    const uint maxStaticPartitions = gStaticSolveAbi.maxStaticPartitions;
    const float elapsedTime = gStaticSolveAbi.elapsedTime;
    const float minPen = gStaticSolveAbi.minPen;

    const uint nbDynamicBodiesToSolve = numDynamicBodiesRounded * nbStaticSlabs;

    const uint globalThreadIdx = groupThreadId.x + groupId.x * 64u;
    const uint warpSize = 32u;
    const uint threadIndexInWarp = groupThreadId.x & (warpSize - 1u);

    const bool residualAccumulationEnabled = desc.contactErrorAccumulator_mCounter >= 0;
    PxgErrorAccumulatorSlang error = { 0.0f, 0, 0.0f };

    Ptr<PxgSolverTxIData> iData = Ptr<PxgSolverTxIData>(desc.solverBodyTxIDataPool);
    Ptr<uint> rigidStaticContactCounts = Ptr<uint>(desc.mRigidStaticContactCounts);
    Ptr<uint> rigidStaticJointCounts = Ptr<uint>(desc.mRigidStaticJointCounts);
    Ptr<uint> rigidStaticContactStartIndices = Ptr<uint>(desc.mRigidStaticContactStartIndices);
    Ptr<uint> rigidStaticJointStartIndices = Ptr<uint>(desc.mRigidStaticJointStartIndices);

    Ptr<PxgBlockConstraintBatch> batches = Ptr<PxgBlockConstraintBatch>(sharedDesc.iter_blockConstraintBatch);
    Ptr<PxgTGSBlockSolverContactHeader> contactHeaders =
        Ptr<PxgTGSBlockSolverContactHeader>(sharedDesc.iter_blockContactHeaders);
    Ptr<PxgTGSBlockSolverFrictionHeader> frictionHeaders =
        Ptr<PxgTGSBlockSolverFrictionHeader>(sharedDesc.iter_blockFrictionHeaders);
    Ptr<PxgTGSBlockSolverContactPoint> contactPoints =
        Ptr<PxgTGSBlockSolverContactPoint>(sharedDesc.iter_blockContactPoints);
    Ptr<PxgTGSBlockSolverContactFriction> frictionsBuf =
        Ptr<PxgTGSBlockSolverContactFriction>(sharedDesc.iter_blockFrictions);
    Ptr<PxgTGSBlockSolverConstraint1DHeader> jointHeaders =
        Ptr<PxgTGSBlockSolverConstraint1DHeader>(sharedDesc.iter_blockJointConstraintHeaders);
    Ptr<PxgTGSBlockSolverConstraint1DCon> jointRowsCon =
        Ptr<PxgTGSBlockSolverConstraint1DCon>(sharedDesc.iter_blockJointConstraintRowsCon);

    // Null-guard (llvmpipe): static-only islands with no static rigid batches
    // leave the block-batch / count pointers null; gate every dependent load
    // behind the non-null check so the optimizer cannot hoist a BDA load out of
    // a zero-trip loop.  A body with zero static contacts/joints does no work.
    if (globalThreadIdx < nbDynamicBodiesToSolve && sharedDesc.iter_blockConstraintBatch != 0 &&
        desc.mRigidStaticContactCounts != 0 && desc.mRigidStaticJointCounts != 0)
    {
        const uint bodyIndex = globalThreadIdx % numDynamicBodiesRounded;
        const uint slabIdx = globalThreadIdx / numDynamicBodiesRounded;
        const uint startIndex = slabIdx * maxStaticPartitions;

        if (bodyIndex < numDynamicBodies)
        {
            const uint rawContact = rigidStaticContactCounts[bodyIndex];
            const uint rawJoint = rigidStaticJointCounts[bodyIndex];
            const uint contactCount = min(maxStaticPartitions, max(startIndex, rawContact) - startIndex);
            const uint jointCount = min(maxStaticPartitions, max(startIndex, rawJoint) - startIndex);

            const uint outputBody = slabIdx * totalBodiesIncKinematics * 2u;

            if (contactCount != 0u || jointCount != 0u)
            {
                PxgSolverTxIData iData0 = {};
                PxgSolverTxIData iData1 = {}; // sqrtInvInertia = 0
                iData1.deltaBody2World.qw = 1.0f; // identity quaternion (PxTransform(PxIdentity))

                if (jointCount != 0u)
                    iData0 = iData[bodyIndex + bodyOffset];

                const uint startContactIndex = rigidStaticContactStartIndices[bodyIndex] + startIndex;
                const uint startJointIndex = rigidStaticJointStartIndices[bodyIndex] + startIndex;

                const uint inputBody = deltaVOffset + bodyIndex + bodyOffset;

                float4 vel0 = bodyVelocities[inputBody];
                float4 vel1 = bodyVelocities[inputBody + totalBodiesIncKinematics];
                float4 vel2 = bodyVelocities[inputBody + totalBodiesIncKinematics + totalBodiesIncKinematics];

                float3 lv0 = float3(vel0.x, vel0.y, vel0.z);
                float3 av0 = float3(vel0.w, vel1.x, vel1.y);
                float3 ld0 = float3(vel1.z, vel1.w, vel2.x);
                float3 ad0 = float3(vel2.y, vel2.z, vel2.w);
                float3 lv1 = float3(0.0f, 0.0f, 0.0f);
                float3 av1 = float3(0.0f, 0.0f, 0.0f);
                float3 ld1 = float3(0.0f, 0.0f, 0.0f);
                float3 ad1 = float3(0.0f, 0.0f, 0.0f);

                for (uint i = 0; i < jointCount; ++i)
                {
                    const uint bk = startJointIndex + i;
                    const uint idx = countbits(batches[bk].mask & ((1u << threadIndexInWarp) - 1u));
                    // For interaction with static objects reference counts are 1.
                    solve1DBlockTGSBda(batches, bk, lv0, av0, lv1, av1, ld0, ad0, ld1, ad1, idx,
                                       jointHeaders, jointRowsCon, iData0, iData1, elapsedTime,
                                       residualAccumulationEnabled, 1.0f, 1.0f);
                }

                for (uint i = 0; i < contactCount; ++i)
                {
                    const uint bk = startContactIndex + i;
                    const uint idx = countbits(batches[bk].mask & ((1u << threadIndexInWarp) - 1u));
                    solveContactBlockTGSBda(batches, bk, lv0, av0, lv1, av1, ld0, ad0, ld1, ad1, idx,
                                            contactHeaders, frictionHeaders, contactPoints, frictionsBuf,
                                            elapsedTime, minPen, residualAccumulationEnabled, error,
                                            1.0f, 1.0f);
                }

                vel0.x = lv0.x; vel0.y = lv0.y; vel0.z = lv0.z;
                vel0.w = av0.x; vel1.x = av0.y; vel1.y = av0.z;

                bodyOutVelocities[bodyIndex + outputBody] = vel0;
                bodyOutVelocities[bodyIndex + outputBody + totalBodiesIncKinematics] = vel1;
            }
        }
    }

    // PxgErrorAccumulator::accumulateErrorGlobalFullWarp(): per-warp sum/sum/max
    // butterfly reduction, then lane 0 of each warp accumulates into
    // solverDesc->contactErrorAccumulator.  The reduction runs unconditionally
    // (uniform control flow around the barriers); idle threads reduce their
    // zero-init accumulators, and a disabled launch skips only the atomics.
    GroupMemoryBarrierWithGroupSync();
    shStaticErrSum[groupThreadId.x] = error.sumOfSquares;
    shStaticErrMax[groupThreadId.x] = error.maxError;
    shStaticErrCount[groupThreadId.x] = uint(error.counter);
    GroupMemoryBarrierWithGroupSync();

    for (uint radius = 16u; radius > 0u; radius >>= 1)
    {
        const uint partner = groupThreadId.x ^ radius; // stays inside the warp's 32 lanes
        const float s = shStaticErrSum[partner];
        const float m = shStaticErrMax[partner];
        const uint c = shStaticErrCount[partner];
        GroupMemoryBarrierWithGroupSync();
        shStaticErrSum[groupThreadId.x] += s;
        shStaticErrMax[groupThreadId.x] = max(shStaticErrMax[groupThreadId.x], m);
        shStaticErrCount[groupThreadId.x] += c;
        GroupMemoryBarrierWithGroupSync();
    }

    if (residualAccumulationEnabled && threadIndexInWarp == 0u)
    {
        const float sumSq = shStaticErrSum[groupThreadId.x];
        const uint count = shStaticErrCount[groupThreadId.x];
        const float maxErr = shStaticErrMax[groupThreadId.x];

        // atomicAdd(&mErrorSumOfSquares, s): float atomic add via CAS on raw bits
        uint assumed = descPtr[0].contactErrorAccumulator_mErrorSumOfSquares;
        for (;;)
        {
            const uint newBits = asuint(asfloat(assumed) + sumSq);
            uint old;
            InterlockedCompareExchange(descPtr[0].contactErrorAccumulator_mErrorSumOfSquares, assumed,
                                       newBits, old);
            if (old == assumed)
                break;
            assumed = old;
        }

        int oldCount;
        InterlockedAdd(descPtr[0].contactErrorAccumulator_mCounter, int(count), oldCount);

        if (maxErr > asfloat(descPtr[0].contactErrorAccumulator_mMaxError))
        {
            assumed = descPtr[0].contactErrorAccumulator_mMaxError;
            for (;;)
            {
                const uint newBits = asuint(max(maxErr, asfloat(assumed)));
                uint old;
                InterlockedCompareExchange(descPtr[0].contactErrorAccumulator_mMaxError, assumed,
                                           newBits, old);
                if (old == assumed)
                    break;
                assumed = old;
            }
        }
    }
}

// propagateStaticSolverBodyVelocitiesTGS: port of
// propagateStaticSolverBodyVelocitiesTGS() in solverMultiBlockTGS.cu.  Averages
// the per-slab static-solve outputs (tempStaticBodyOutputs, written by
// solveStaticBlockTGS) back into the body's deltaV velocity slot.  One thread
// per dynamic body; block = SOLVE_BLOCK_PARTITION (64), grid = nbBlocksRequired
// (WorkgroupCount().x).  Launched with the SAME 7-arg pack as
// solveStaticBlockTGS (argSizes {8,8,4,4,4,4,4} -> 40 bytes); nbStaticSlabs /
// elapsedTime / minPen are unused by this kernel body, exactly as the .cu
// ignores them.
struct PropagateStaticSolverBodyVelocitiesTGSAbiParams
{
    uint64_t solverCoreDesc;   // @0
    uint64_t sharedDesc;       // @8
    uint islandIndex;          // @16
    uint nbStaticSlabs;        // @20 (unused)
    uint maxStaticPartitions;  // @24
    float elapsedTime;         // @28 (unused)
    float minPen;              // @32 (unused)
}                              // = 40 (round to maxAlign 8)
[[vk::push_constant]]
ConstantBuffer<PropagateStaticSolverBodyVelocitiesTGSAbiParams> gStaticPropagateAbi;

[shader("compute")]
[numthreads(64, 1, 1)]
void propagateStaticSolverBodyVelocitiesTGS(uint3 groupThreadId : SV_GroupThreadID,
                                            uint3 groupId : SV_GroupID)
{
    PxgSolverCoreDescGpu desc = (Ptr<PxgSolverCoreDescGpu>(gStaticPropagateAbi.solverCoreDesc))[0];
    PxgSolverSharedDescTGSGpu sharedDesc =
        (Ptr<PxgSolverSharedDescTGSGpu>(gStaticPropagateAbi.sharedDesc))[0];
    PxgIslandContextGpu island =
        (Ptr<PxgIslandContextGpu>(desc.islandContextPool))[gStaticPropagateAbi.islandIndex];

    Ptr<float4> bodyVelocities = Ptr<float4>(sharedDesc.iter_solverBodyVelPool);
    Ptr<float4> bodyOutVelocities = Ptr<float4>(sharedDesc.iter_tempStaticBodyOutputs);
    const uint numDynamicBodies = island.mBodyCount;
    const uint bodyOffset = island.mBodyStartIndex;
    const uint deltaVOffset = desc.accumulatedBodyDeltaVOffset;
    const uint totalBodiesIncKinematics = numDynamicBodies + bodyOffset;
    const uint maxStaticPartitions = gStaticPropagateAbi.maxStaticPartitions;

    const uint globalThreadIdx = groupThreadId.x + groupId.x * 64u;

    Ptr<uint> rigidStaticContactCounts = Ptr<uint>(desc.mRigidStaticContactCounts);
    Ptr<uint> rigidStaticJointCounts = Ptr<uint>(desc.mRigidStaticJointCounts);

    // Null-guard (llvmpipe): skip when the per-body static count arrays are
    // absent (no static rigid work); the tempStaticBodyOutputs reads below are
    // then unreachable and cannot be hoisted.
    if (globalThreadIdx < numDynamicBodies && desc.mRigidStaticContactCounts != 0 &&
        desc.mRigidStaticJointCounts != 0)
    {
        const uint contactCount = rigidStaticContactCounts[globalThreadIdx];
        const uint jointCount = rigidStaticJointCounts[globalThreadIdx];

        if (contactCount != 0u || jointCount != 0u)
        {
            const uint maxOutputs = max(contactCount, jointCount);

            float scale = 0.0f;
            float4 vel0 = float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 vel1 = float4(0.0f, 0.0f, 0.0f, 0.0f);
            uint index = globalThreadIdx;
            for (uint i = 0; i < maxOutputs; i += maxStaticPartitions)
            {
                vel0 += bodyOutVelocities[index];
                vel1 += bodyOutVelocities[index + totalBodiesIncKinematics];
                index += totalBodiesIncKinematics * 2u;
                scale += 1.0f;
            }

            scale = 1.0f / scale;

            const uint outputBody = deltaVOffset + globalThreadIdx + bodyOffset;
            bodyVelocities[outputBody] = vel0 * scale;
            bodyVelocities[outputBody + totalBodiesIncKinematics] = vel1 * scale;
        }
    }
}
