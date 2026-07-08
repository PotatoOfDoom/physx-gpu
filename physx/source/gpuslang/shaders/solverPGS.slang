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

// solverPGS.slang
//
// Slang port of the rigid-body PGS constraint-solver iteration
// (gpusolver/src/CUDA/solverMultiBlock.cu + solverBlock.cuh +
//  constraintPrepShared.cuh + lowleveldynamics/shared/DyCpuGpu1dConstraint.h):
//   - solveBlockPartition (rigid path: solveContactBlock + solve1DBlock,
//     incl. mass-splitting reference counts via countActiveSlabs)
//
// One 32-thread workgroup processes one PxgBlockConstraintBatch; lane i
// solves constraint lane i of the 32-wide SoA block data — the same mapping
// as the CUDA one-warp-per-batch scheme.  The block data structures
// (PxgBlockConstraintBatch, PxgBlockSolverContactHeader/Point/Friction,
// PxgBlockSolverConstraint1DHeader/Con/Mod) are mirrored byte-for-byte
// including their PX_ALIGN(128/256) padding.
//
// Differences from CUDA:
//   - Island bookkeeping (islandContextPool, constraintsPerPartition) is host
//     work; the dispatch receives the resolved [startIndex, endIndex) batch
//     range via push constants.
//   - The staging of IterativeSolveData through shared memory and the
//     Pxldcg/Pxstcg cache hints are memory optimizations with no semantic
//     effect; buffers are bound directly.
//   - The contact-error accumulator (global convergence statistics) is not
//     ported yet; the per-row 1D residual writeback is.
//   - The explicit prefetch ("next*") load pattern in solveContactBlock is
//     flattened to direct loads: each contact/friction row uses exactly its
//     own data, which is what the prefetch chain resolves to whenever
//     numNormalConstr > 0 (friction with zero contacts does not occur).

import pxgSlangCommon;

static const float DY_MIN_RESPONSE = 1e-12f;

// DySolverConstraintTypes.h
static const uint DY_SC_FLAG_SPRING = 1u << 5;
static const uint DY_SC_FLAG_ACCELERATION_SPRING = 1u << 7;

// PxgSolverFlags.h
static const uint PXG_CONTACT_FLAG_COMPLIANT_ACCELERATION_SPRING = 1u << 3;

// PxgSolverConstraintDesc.h
static const uint PXG_CONSTRAINT_CONTACT = 0; // PxgSolverConstraintDesc::eCONTACT

// ---------------------------------------------------------------------------
// Block SoA data structures (byte-identical mirrors incl. alignment padding)
// ---------------------------------------------------------------------------

// mirrors PxgBlockConstraintBatch (1536 bytes)
struct PxgBlockConstraintBatch
{
    uint descStride_constraintType; // PxU16 mDescStride | PxU16 constraintType << 16
    uint blockContactIndex;
    uint mConstraintBatchIndex;
    uint startConstraintIndex;
    uint startFrictionIndex;
    uint mStartPartitionIndex;
    uint mArticulationResponseIndex;
    uint mask;
    uint pad0[24];             // PX_ALIGN(128) of the first array
    uint2 bodyANodeIndex[32];  // 128
    uint2 bodyBNodeIndex[32];  // 384
    uint bodyAIndex[32];       // 640
    uint bodyBIndex[32];       // 768
    uint remappedBodyAIndex[32]; // 896
    uint remappedBodyBIndex[32]; // 1024
    uint slabId[32];           // 1152
    uint2 shapeInteraction[32];// 1280 (host pointers, unused on device)
}                              // = 1536

// mirrors PxgBlockSolverContactHeader (1920 bytes)
struct PxgBlockSolverContactHeader
{
    float4 invMass0_1_angDom0_1[32];  // 0
    float4 normal_staticFriction[32]; // 512
    float accumNormalForce[32];       // 1024
    float minNormalForce[32];         // 1152
    uint flags[32];                   // 1280
    uint numNormalConstr[32];         // 1408
    uint forceWritebackOffset[32];    // 1536
    float restitution[32];            // 1664
    float cfm[32];                    // 1792
}                                     // = 1920

// mirrors PxgBlockSolverFrictionHeader (1408 bytes)
struct PxgBlockSolverFrictionHeader
{
    float4 frictionNormals0[32];  // 0   (frictionNormals[0])
    float4 frictionNormals1[32];  // 512 (frictionNormals[1])
    uint numFrictionConstr[32];   // 1024
    float dynamicFriction[32];    // 1152
    uint broken[32];              // 1280
}                                 // = 1408

// mirrors PxgBlockSolverContactPoint (1664 bytes)
struct PxgBlockSolverContactPoint
{
    float4 raXn_targetVelocity[32]; // 0
    float4 rbXn_maxImpulse[32];     // 512
    float appliedForce[32];         // 1024
    float resp0[32];                // 1152
    float resp1[32];                // 1280
    float coeff0[32];               // 1408
    float coeff1[32];               // 1536
}                                   // = 1664

// mirrors PxgBlockSolverContactFriction (1408 bytes)
struct PxgBlockSolverContactFriction
{
    float4 raXn_bias[32];       // 0
    float4 rbXn_targetVelW[32]; // 512
    float appliedForce[32];     // 1024
    float resp0[32];            // 1152
    float resp1[32];            // 1280
}                               // = 1408

// mirrors PxgBlockSolverConstraint1DHeader (1664 bytes; the PxU16 arrays are
// 64 bytes of payload padded to the 128-byte alignment of the next member)
struct PxgBlockSolverConstraint1DHeader
{
    uint rowCounts[16];    // 0:   PxU16[32], two lanes per word
    uint pad0[16];
    uint breakable[16];    // 128: PxU16[32]
    uint pad1[16];
    float angBreakImpulse[32];                 // 256
    float4 body0WorldOffset_linBreakImpulse[32]; // 384
    float invMass0D0[32];                      // 896
    float invMass1D1[32];                      // 1024
    float invInertiaScale0[32];                // 1152
    float invInertiaScale1[32];                // 1280
    uint writeBackOffset[32];                  // 1408
    float cfm[32];                             // 1536
}                                              // = 1664

// mirrors PxgBlockSolverConstraint1DCon (PX_ALIGN(256): 2304 bytes)
struct PxgBlockSolverConstraint1DCon
{
    float4 lin0XYZ_minImpulse[32]; // 0
    float4 lin1XYZ_maxImpulse[32]; // 512
    float4 ang0XYZ_resp0[32];      // 1024
    float4 ang1XYZ_resp1[32];      // 1536
    float initJointSpeed[32];      // 2048
    uint pad0[32];                 // 2176 (tail padding to the 256 alignment)
}                                  // = 2304

// mirrors PxgBlockSolverConstraint1DMod (1024 bytes)
struct PxgBlockSolverConstraint1DMod
{
    float ang0Writeback[96]; // 0: PxVec3[32], tightly packed
    float appliedForce[32];  // 384
    uint flags[32];          // 512
    float residual[32];      // 640
    float coeff0[32];        // 768
    float coeff1[32];        // 896
}                            // = 1024

// ---------------------------------------------------------------------------
// bindings
// ---------------------------------------------------------------------------

struct SolverParams
{
    uint startIndex;  // first batch of this partition
    uint endIndex;    // one past the last batch
    uint doFriction;
    uint residualReportingEnabled;
    uint numSlabs;
    uint numTotalBodies;
    uint bodyOffset;  // islandContextPool->mBodyStartIndex
};

[[vk::push_constant]]
ConstantBuffer<SolverParams> gParams;

RWStructuredBuffer<PxgBlockConstraintBatch> gBatches;                // binding 0
RWStructuredBuffer<PxgBlockSolverContactHeader> gContactHeaders;     // binding 1
RWStructuredBuffer<PxgBlockSolverFrictionHeader> gFrictionHeaders;   // binding 2
RWStructuredBuffer<PxgBlockSolverContactPoint> gContactPoints;       // binding 3
RWStructuredBuffer<PxgBlockSolverContactFriction> gFrictions;        // binding 4
RWStructuredBuffer<PxgBlockSolverConstraint1DHeader> gJointHeaders;  // binding 5
RWStructuredBuffer<PxgBlockSolverConstraint1DCon> gJointRowsCon;     // binding 6
RWStructuredBuffer<PxgBlockSolverConstraint1DMod> gJointRowsMod;     // binding 7
RWStructuredBuffer<float4> gSolverBodyVelPool;                       // binding 8
RWStructuredBuffer<uint> gEncodedReferenceCount;                     // binding 9

// ---------------------------------------------------------------------------
// helpers (transcriptions of the CUDA device functions)
// ---------------------------------------------------------------------------

// mirrors countActiveSlabs() in solverBlock.cuh
uint countActiveSlabs(uint index, uint numSlabs, uint numSolverBodies)
{
    uint referenceCount = 0;
    const uint num32Slabs = (numSlabs + 31) / 32;
    for (uint i = 0; i < num32Slabs; ++i)
        referenceCount += countbits(gEncodedReferenceCount[index + i * numSolverBodies]);
    return max(1u, referenceCount);
}

// mirrors Dy::computeRecipUnitResponse()
float computeRecipUnitResponse(float unitResponse, float minRowResponse)
{
    return unitResponse <= minRowResponse ? 0.0f : 1.0f / unitResponse;
}

// mirrors computeContactCoefficients() in constraintPrepShared.cuh
void computeContactCoefficients(uint flags, float restitution, float unitResponse, float recipResponse,
                                float targetVelocity, float coeff0, float coeff1,
                                out float velMultiplier, out float impulseMultiplier,
                                out float unbiasedErr, out float biasedErr)
{
    if (restitution < 0.0f)
    {
        // compliant contact (spring)
        const bool accelSpring = (flags & PXG_CONTACT_FLAG_COMPLIANT_ACCELERATION_SPRING) != 0;
        const float massIfAccelElseOne = accelSpring ? recipResponse : 1.0f;
        const float oneIfAccelElseR = accelSpring ? 1.0f : unitResponse;

        const float a = coeff0;
        const float b = coeff1 * massIfAccelElseOne;

        const float x = 1.0f / (a * oneIfAccelElseR + 1.0f);
        const float scaledBias = x * b;

        velMultiplier = x * a * massIfAccelElseOne;
        impulseMultiplier = 1.0f - x;
        unbiasedErr = biasedErr = targetVelocity * velMultiplier - scaledBias;
    }
    else
    {
        velMultiplier = recipResponse;
        biasedErr = coeff0 * velMultiplier;
        unbiasedErr = coeff1 * velMultiplier;
        impulseMultiplier = 1.0f;
    }
}

// mirrors Dy::compute1dConstraintSolverConstantsPGS() (reduced-coefficient
// variant in DyCpuGpu1dConstraint.h); coeff2 carries initJointSpeed
void compute1dConstraintSolverConstantsPGS(bool isSpring, bool isAccelerationSpring, float coeff0,
                                           float coeff1, float coeff2, float unitResponse,
                                           float recipUnitResponse, out float constant,
                                           out float unbiasedConstant, out float velMultiplier,
                                           out float impulseMultiplier)
{
    if (isSpring)
    {
        const float a = coeff0;
        const float b = coeff1;

        if (isAccelerationSpring)
        {
            const float x = 1.0f / (1.0f + a);
            constant = x * recipUnitResponse * b;
            unbiasedConstant = constant;
            velMultiplier = -x * recipUnitResponse * a;
            impulseMultiplier = 1.0f - x;
        }
        else
        {
            const float x = 1.0f / (1.0f + a * unitResponse);
            constant = x * b;
            unbiasedConstant = constant;
            velMultiplier = -x * a;
            impulseMultiplier = 1.0f - x;
        }
    }
    else
    {
        velMultiplier = -recipUnitResponse;
        impulseMultiplier = 1.0f;
        constant = coeff0 * recipUnitResponse;
        unbiasedConstant = coeff1 * recipUnitResponse;
    }

    const float velBias = coeff2 * velMultiplier;
    constant += velBias;
    unbiasedConstant += velBias;
}

// mirrors PxgErrorAccumulator::calculateResidual()
float calculateResidual(float deltaF, float velocityMultiplier)
{
    return velocityMultiplier == 0.0f ? 0.0f : deltaF / velocityMultiplier;
}

// ---------------------------------------------------------------------------
// solve functions
// ---------------------------------------------------------------------------

// mirrors solveContactBlock() in solverBlock.cuh
void solveContactBlock(uint batchIndex, inout float3 b0LinVel, inout float3 b0AngVel,
                       inout float3 b1LinVel, inout float3 b1AngVel, bool doFriction,
                       uint threadIndex, float ref0, float ref1)
{
    const uint constraintBatchIndex = gBatches[batchIndex].mConstraintBatchIndex;
    const uint startConstraintIndex = gBatches[batchIndex].startConstraintIndex;
    const uint startFrictionIndex = gBatches[batchIndex].startFrictionIndex;

    float3 linVel0 = b0LinVel;
    float3 linVel1 = b1LinVel;
    float3 angVel0 = b0AngVel;
    float3 angVel1 = b1AngVel;

    const uint numNormalConstr = gContactHeaders[constraintBatchIndex].numNormalConstr[threadIndex];
    const uint numFrictionConstr = gFrictionHeaders[constraintBatchIndex].numFrictionConstr[threadIndex];

    float accumulatedNormalImpulse = 0.0f;

    const float4 invMass0_1_angDom0_1 = gContactHeaders[constraintBatchIndex].invMass0_1_angDom0_1[threadIndex];

    const float invMassA = ref0 * invMass0_1_angDom0_1.x;
    const float invMassB = ref1 * invMass0_1_angDom0_1.y;
    const float angDom0 = ref0 * invMass0_1_angDom0_1.z;
    const float angDom1 = ref1 * invMass0_1_angDom0_1.w;

    const float4 normal_staticFriction = gContactHeaders[constraintBatchIndex].normal_staticFriction[threadIndex];
    const float3 normal = normal_staticFriction.xyz;
    const float staticFrictionCof = normal_staticFriction.w;

    const float restitution = gContactHeaders[constraintBatchIndex].restitution[threadIndex];
    const float p8 = 0.8f;
    const uint flags = gContactHeaders[constraintBatchIndex].flags[threadIndex];

    const float3 delLinVel0 = normal * invMassA;
    const float3 delLinVel1 = normal * invMassB;

    for (uint i = 0; i < numNormalConstr; i++)
    {
        const uint c = startConstraintIndex + i;

        const float4 raXn_targetVelocity = gContactPoints[c].raXn_targetVelocity[threadIndex];
        const float4 rbXn_maxImpulse = gContactPoints[c].rbXn_maxImpulse[threadIndex];
        const float appliedForce = gContactPoints[c].appliedForce[threadIndex];
        const float resp0 = gContactPoints[c].resp0[threadIndex];
        const float resp1 = gContactPoints[c].resp1[threadIndex];
        const float coeff0 = gContactPoints[c].coeff0[threadIndex];
        const float coeff1 = gContactPoints[c].coeff1[threadIndex];

        const float3 raXn = raXn_targetVelocity.xyz;
        const float3 rbXn = rbXn_maxImpulse.xyz;
        const float targetVelocity = raXn_targetVelocity.w;
        const float maxImpulse = rbXn_maxImpulse.w;

        const float unitResponse = ref0 * resp0 + ref1 * resp1;
        const float recipResponse = (unitResponse > 0.0f) ? (1.0f / unitResponse) : 0.0f;

        float velMultiplier = recipResponse;
        float impulseMul = 1.0f;
        float unbiasedError = 0.0f;
        float biasedErr = 0.0f;

        computeContactCoefficients(flags, restitution, unitResponse, recipResponse, targetVelocity,
                                   coeff0, coeff1, velMultiplier, impulseMul, unbiasedError, biasedErr);

        // Compute the normal velocity of the constraint.
        const float v0 = dot(linVel0, normal) + dot(angVel0, raXn);
        const float v1 = dot(linVel1, normal) + dot(angVel1, rbXn);
        const float normalVel = v0 - v1;

        // KS - clamp the maximum force
        const float tempDeltaF = biasedErr - normalVel * velMultiplier;
        const float _deltaF = max(tempDeltaF, -appliedForce);
        const float _newForce = appliedForce * impulseMul + _deltaF;
        const float newForce = min(_newForce, maxImpulse);
        const float deltaF = newForce - appliedForce;

        linVel0 += delLinVel0 * deltaF;
        linVel1 -= delLinVel1 * deltaF;
        angVel0 += raXn * (deltaF * angDom0);
        angVel1 -= rbXn * (deltaF * angDom1);

        gContactPoints[c].appliedForce[threadIndex] = newForce;

        accumulatedNormalImpulse = accumulatedNormalImpulse + newForce;
    }

    if (numFrictionConstr != 0 && doFriction)
    {
        const float dynamicFrictionCof = gFrictionHeaders[constraintBatchIndex].dynamicFriction[threadIndex];
        const float maxFrictionImpulse = staticFrictionCof * accumulatedNormalImpulse;
        const float maxDynFrictionImpulse = dynamicFrictionCof * accumulatedNormalImpulse;

        uint broken = 0;

        for (uint i = 0; i < numFrictionConstr; i++)
        {
            const uint f = startFrictionIndex + i;

            const float4 frictionNormal = (i & 1) != 0
                ? gFrictionHeaders[constraintBatchIndex].frictionNormals1[threadIndex]
                : gFrictionHeaders[constraintBatchIndex].frictionNormals0[threadIndex];

            const float4 raXn_bias = gFrictions[f].raXn_bias[threadIndex];
            const float4 rbXn_targetVelW = gFrictions[f].rbXn_targetVelW[threadIndex];
            const float appliedForce = gFrictions[f].appliedForce[threadIndex];
            const float resp0 = gFrictions[f].resp0[threadIndex];
            const float resp1 = gFrictions[f].resp1[threadIndex];

            const float3 fnormal = frictionNormal.xyz;
            const float3 raXn = raXn_bias.xyz;
            const float3 rbXn = rbXn_targetVelW.xyz;

            const float resp = ref0 * resp0 + ref1 * resp1;
            const float velMultiplier = (resp > 0.0f) ? (p8 / resp) : 0.0f;

            const float bias = raXn_bias.w;

            const float3 fdelLinVel0 = fnormal * invMassA;
            const float3 fdelLinVel1 = fnormal * invMassB;

            const float targetVel = rbXn_targetVelW.w;

            const float v0 = dot(angVel0, raXn) + dot(linVel0, fnormal);
            const float v1 = dot(angVel1, rbXn) + dot(linVel1, fnormal);
            const float normalVel = v0 - v1;

            const float tmp1 = appliedForce - (bias - targetVel) * velMultiplier;
            const float totalImpulse = tmp1 - normalVel * velMultiplier;

            const bool clampImpulse = abs(totalImpulse) > maxFrictionImpulse;
            const float totalClamped = min(maxDynFrictionImpulse, max(-maxDynFrictionImpulse, totalImpulse));

            const float newAppliedForce = clampImpulse ? totalClamped : totalImpulse;

            const float deltaF = newAppliedForce - appliedForce;

            linVel0 += fdelLinVel0 * deltaF;
            linVel1 -= fdelLinVel1 * deltaF;
            angVel0 += raXn * (deltaF * angDom0);
            angVel1 -= rbXn * (deltaF * angDom1);

            gFrictions[f].appliedForce[threadIndex] = newAppliedForce;
            broken = broken | (clampImpulse ? 1u : 0u);
        }
        gFrictionHeaders[constraintBatchIndex].broken[threadIndex] = broken;
    }

    b0LinVel = linVel0;
    b0AngVel = angVel0;
    b1LinVel = linVel1;
    b1AngVel = angVel1;
}

// mirrors solve1DBlock() in solverBlock.cuh
void solve1DBlock(uint batchIndex, inout float3 b0LinVel, inout float3 b0AngVel,
                  inout float3 b1LinVel, inout float3 b1AngVel, uint threadIndex,
                  bool residualReportingEnabled, float ref0, float ref1)
{
    const uint constraintBatchIndex = gBatches[batchIndex].mConstraintBatchIndex;
    const uint startConstraintIndex = gBatches[batchIndex].startConstraintIndex;

    float3 linVel0 = b0LinVel;
    float3 linVel1 = b1LinVel;
    float3 angVel0 = b0AngVel;
    float3 angVel1 = b1AngVel;

    const float invMass0 = ref0 * gJointHeaders[constraintBatchIndex].invMass0D0[threadIndex];
    const float invMass1 = ref1 * gJointHeaders[constraintBatchIndex].invMass1D1[threadIndex];
    const float invInertiaScale0 = ref0 * gJointHeaders[constraintBatchIndex].invInertiaScale0[threadIndex];
    const float invInertiaScale1 = ref1 * gJointHeaders[constraintBatchIndex].invInertiaScale1[threadIndex];

    // PxU16 rowCounts[32]: two lanes per word
    const uint rowCountWord = gJointHeaders[constraintBatchIndex].rowCounts[threadIndex >> 1];
    const uint rowCount = (rowCountWord >> ((threadIndex & 1) * 16)) & 0xFFFFu;

    for (uint i = 0; i < rowCount; ++i)
    {
        const uint r = startConstraintIndex + i;

        const float4 _clinVel0_minImpulse = gJointRowsCon[r].lin0XYZ_minImpulse[threadIndex];
        const float4 _clinVel1_maxImpulse = gJointRowsCon[r].lin1XYZ_maxImpulse[threadIndex];
        const float4 _cangVel0_resp0 = gJointRowsCon[r].ang0XYZ_resp0[threadIndex];
        const float4 _cangVel1_resp1 = gJointRowsCon[r].ang1XYZ_resp1[threadIndex];

        const float3 clinVel0 = _clinVel0_minImpulse.xyz;
        const float3 clinVel1 = _clinVel1_maxImpulse.xyz;
        const float3 cangVel0 = _cangVel0_resp0.xyz;
        const float3 cangVel1 = _cangVel1_resp1.xyz;

        const float resp0 = ref0 * _cangVel0_resp0.w;
        const float resp1 = ref1 * _cangVel1_resp1.w;
        const float initJointSpeed = gJointRowsCon[r].initJointSpeed[threadIndex];

        const float coeff0 = gJointRowsMod[r].coeff0[threadIndex];
        const float coeff1 = gJointRowsMod[r].coeff1[threadIndex];
        const uint flags = gJointRowsMod[r].flags[threadIndex];

        const float unitResponse = resp0 + resp1;
        const float recipResponse = computeRecipUnitResponse(unitResponse, DY_MIN_RESPONSE);

        float constant, unbiasedConstant, vMul, iMul;

        const bool isSpring = (flags & DY_SC_FLAG_SPRING) != 0;
        const bool isAccelerationSpring = (flags & DY_SC_FLAG_ACCELERATION_SPRING) != 0;

        compute1dConstraintSolverConstantsPGS(isSpring, isAccelerationSpring, coeff0, coeff1,
                                              initJointSpeed, unitResponse, recipResponse, constant,
                                              unbiasedConstant, vMul, iMul);

        const float appliedForce = gJointRowsMod[r].appliedForce[threadIndex];

        const float maxImpulse = _clinVel1_maxImpulse.w;
        const float minImpulse = _clinVel0_minImpulse.w;

        const float v0 = dot(linVel0, clinVel0) + dot(angVel0, cangVel0);
        const float v1 = dot(linVel1, clinVel1) + dot(angVel1, cangVel1);
        const float normalVel = v0 - v1;

        const float unclampedForce = iMul * appliedForce + (vMul * normalVel + constant);
        const float clampedForce = min(maxImpulse, max(minImpulse, unclampedForce));
        const float deltaF = clampedForce - appliedForce;

        gJointRowsMod[r].appliedForce[threadIndex] = clampedForce;
        if (residualReportingEnabled)
            gJointRowsMod[r].residual[threadIndex] = calculateResidual(deltaF, vMul);

        linVel0 = linVel0 + clinVel0 * (deltaF * invMass0);
        linVel1 = linVel1 - clinVel1 * (deltaF * invMass1);
        angVel0 = angVel0 + cangVel0 * deltaF * invInertiaScale0;
        angVel1 = angVel1 - cangVel1 * deltaF * invInertiaScale1;
    }

    b0LinVel = linVel0;
    b0AngVel = angVel0;
    b1LinVel = linVel1;
    b1AngVel = angVel1;
}

// ---------------------------------------------------------------------------
// kernel
// ---------------------------------------------------------------------------

// mirrors solveBlockPartition() in solverMultiBlock.cu (rigid-body path);
// one workgroup = one warp = one constraint batch
[shader("compute")]
[numthreads(32, 1, 1)]
void solveBlockPartition(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint threadIndexInWarp = groupThreadId.x;
    const uint k = gParams.startIndex + groupId.x;

    if (k >= gParams.endIndex)
        return;

    const uint descStride = gBatches[k].descStride_constraintType & 0xFFFFu;
    const uint constraintType = gBatches[k].descStride_constraintType >> 16;

    const uint readIndex = k * 128 + threadIndexInWarp;

    if (threadIndexInWarp < descStride)
    {
        const float4 linVel0 = gSolverBodyVelPool[readIndex];
        const float4 angVel0 = gSolverBodyVelPool[readIndex + 32];
        const float4 linVel1 = gSolverBodyVelPool[readIndex + 64];
        const float4 angVel1 = gSolverBodyVelPool[readIndex + 96];

        float3 lv0 = linVel0.xyz;
        float3 lv1 = linVel1.xyz;
        float3 av0 = angVel0.xyz;
        float3 av1 = angVel1.xyz;

        // Reference counts to be used for the current sub-timestep or iteration
        // (mass splitting across slabs).
        float curRef0 = 1.0f;
        float curRef1 = 1.0f;

        if (gBatches[k].bodyAIndex[threadIndexInWarp] >= gParams.bodyOffset)
            curRef0 = float(countActiveSlabs(gBatches[k].bodyAIndex[threadIndexInWarp], gParams.numSlabs,
                                             gParams.numTotalBodies));
        if (gBatches[k].bodyBIndex[threadIndexInWarp] >= gParams.bodyOffset)
            curRef1 = float(countActiveSlabs(gBatches[k].bodyBIndex[threadIndexInWarp], gParams.numSlabs,
                                             gParams.numTotalBodies));

        if (constraintType == PXG_CONSTRAINT_CONTACT)
            solveContactBlock(k, lv0, av0, lv1, av1, gParams.doFriction != 0, threadIndexInWarp, curRef0, curRef1);
        else
            solve1DBlock(k, lv0, av0, lv1, av1, threadIndexInWarp, gParams.residualReportingEnabled != 0,
                         curRef0, curRef1);

        const uint indexA = gBatches[k].remappedBodyAIndex[threadIndexInWarp]; // ComputeBodyBatchStartIndex = identity
        const uint indexB = gBatches[k].remappedBodyBIndex[threadIndexInWarp];

        gSolverBodyVelPool[indexA] = float4(lv0, 0.0f);
        gSolverBodyVelPool[indexA + 32] = float4(av0, 0.0f);
        gSolverBodyVelPool[indexB] = float4(lv1, 0.0f);
        gSolverBodyVelPool[indexB + 32] = float4(av1, 0.0f);
    }
}
