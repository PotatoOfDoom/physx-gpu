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

// solverPGS.slang — the rigid-body PGS constraint-solver kernels on the BDA
// ENGINE ABI (P5 Step 5).
//
// Slang port of gpusolver/src/CUDA/solverMultiBlock.cu + solverBlock.cuh +
// solverBlockCommon.cuh (+ constraintPrepShared.cuh /
// lowleveldynamics/shared/DyCpuGpu1dConstraint.h for the shared math):
//   - solveBlockPartition            (contact + 1D joint velocity solve,
//                                     mass-splitting reference counts)
//   - concludeBlocks                 (switch to the unbiased constants)
//   - writebackBlocks                (forces / friction patches / threshold
//                                     stream / 1D writeback)
//   - writeBackBodies                (solved velocities -> motion velocities)
//   - computeAverageSolverBodyVelocity / propagateSolverBodyVelocity
//                                    (multi-slab mass-splitting averaging)
//   - markActiveSlabPGS              (active-slab marking that FEEDS the
//                                     reference counts)
//
// Every entry takes the ENGINE's CUDA parameter pack as a push-constant blob
// (see the per-kernel AbiParams structs; argSizes are transcribed in
// bridge/PxgVulkanKernelAbi.cpp) and reads every count/offset/pool pointer
// on-device through the byte-exact pxgSlangSolverDesc mirrors — the PGS shared
// desc is PxgSolverSharedDesc<IterativeSolveData> (PxgSolverSharedDescPGSGpu).
// One 32-thread warp processes one PxgBlockConstraintBatch; lane i solves
// constraint lane i of the 32-wide SoA block data, the same mapping as the
// CUDA one-warp-per-batch scheme.  The block data structures are mirrored
// byte-for-byte including their PX_ALIGN(128/256) padding.
//
// PGS body-velocity layouts (BOTH differ from TGS's 3-float4 stripes):
//   - per-partition solve pool: readIndex = k * 128 + lane, 4 float4 stripes
//     (lin0 @+0, ang0 @+32, lin1 @+64, ang1 @+96) — TGS uses k * 192 + lane;
//   - accumulated deltaV pool: TWO float4 stripes per body — linVel @
//     outputOffset + bodyId, angVel @ +totalBodiesIncKinematics — and the
//     averaging slots are 2 float4 per slab slot
//     (ComputeAverageBodyBatchStartIndex float4sPerBody = 2).
//
// Differences from CUDA (same policy as solverTGS.slang):
//   - the staging of IterativeSolveData through shared memory and the
//     Pxldcg/Pxstcg cache hints are memory optimizations with no semantic
//     effect; data is read through BDA pointers directly;
//   - the explicit prefetch ("next*") load pattern in solveContactBlock is
//     flattened to direct loads: each contact/friction row uses exactly its
//     own data, which is what the prefetch chain resolves to whenever
//     numNormalConstr > 0 (friction with zero contacts does not occur).
//     checkActiveContactBlock keeps its unconditional contacts[0] prefetch
//     (see markActiveSlabPGS below);
//   - accumulateErrorGlobalFullWarp(): the __shfl_xor butterfly becomes the
//     same butterfly through groupshared scratch under uniform barriers, and
//     CUDA's float atomicAdd / AtomicMax on the accumulator become CAS loops
//     on the raw float bits (same as solveBlockUnified in solverTGS.slang);
//   - CUDA writebackBlocks stages threshold elements in shared memory and
//     appends them with one warp-aggregated atomic; the port appends
//     per-thread — the stream contents are identical, ordering within the
//     stream is nondeterministic in both versions.

import pxgSlangCommon;
import pxgSlangSolverDesc;

static const float DY_MIN_RESPONSE = 1e-12f;

// DySolverConstraintTypes.h
static const uint DY_SC_FLAG_OUTPUT_FORCE = 1u << 1;
static const uint DY_SC_FLAG_SPRING = 1u << 5;
static const uint DY_SC_FLAG_ACCELERATION_SPRING = 1u << 7;

// PxgSolverFlags.h
static const uint PXG_CONTACT_FLAG_HAS_FORCE_THRESHOLDS = 1u << 0;
static const uint PXG_CONTACT_FLAG_COMPLIANT_ACCELERATION_SPRING = 1u << 3;

static const float PX_MAX_REAL = 3.4028234663852885981170418348452e+38f;

// PxgSolverConstraintDesc.h — PxgConstraintType, the encoding the engine's block
// prep writes into descStride_constraintType>>16: eCONSTRAINT_1D=0, eCONTACT=1,
// eARTICULATION_CONSTRAINT_1D=2, eARTICULATION_CONTACT=3.  A batch is a CONTACT
// iff its type is eCONTACT(1) or eARTICULATION_CONTACT(3); 0/2 are 1D joints.
static const uint PXG_SOLVER_CONSTRAINT_TYPE_CONTACT = 1u;
static const uint PXG_SOLVER_CONSTRAINT_TYPE_ARTI_CONTACT = 3u;
bool pxgIsContactType(uint t)
{
    return t == PXG_SOLVER_CONSTRAINT_TYPE_CONTACT || t == PXG_SOLVER_CONSTRAINT_TYPE_ARTI_CONTACT;
}

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

// mirrors PxgBlockFrictionPatch (2816 bytes, PX_ALIGN(256))
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

// ---------------------------------------------------------------------------
// helpers (transcriptions of the CUDA device functions)
// ---------------------------------------------------------------------------

// mirrors countActiveSlabs() in solverBlock.cuh (BDA pointer form)
uint countActiveSlabsBda(uint index, uint numSlabs, uint numSolverBodies, Ptr<uint> encodedReferenceCount)
{
    uint referenceCount = 0;
    const uint num32Slabs = (numSlabs + 31) / 32;
    for (uint i = 0; i < num32Slabs; ++i)
        referenceCount += countbits(encodedReferenceCount[index + i * numSolverBodies]);
    return max(1u, referenceCount);
}

// mirrors resetSlabCount() in solverBlock.cuh
void resetSlabCountBda(uint index, uint numSlabs, uint numSolverBodies, Ptr<uint> encodedReferenceCount)
{
    const uint num32Slabs = (numSlabs + 31) / 32;
    for (uint i = 0; i < num32Slabs; ++i)
        encodedReferenceCount[index + i * numSolverBodies] = 0u;
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

// mirrors PxgErrorAccumulator (gpucommon/src/CUDA/solverResidual.cuh): the
// per-thread contact residual statistics.
struct PxgErrorAccumulatorSlang
{
    float sumOfSquares;
    int counter;
    float maxError;
}

// PxgErrorAccumulator::accumulateErrorLocal().  CUDA computes the residual
// with __fdividef; exact division here (the CPU references divide exactly too).
void pxgAccumulateErrorLocal(inout PxgErrorAccumulatorSlang e, float deltaF, float velocityMultiplier)
{
    const float r = velocityMultiplier == 0.0f ? 0.0f : deltaF / velocityMultiplier;
    e.sumOfSquares += r * r;
    ++e.counter;
    e.maxError = max(e.maxError, abs(r));
}

// ---------------------------------------------------------------------------
// solve functions (BDA)
// ---------------------------------------------------------------------------

// mirrors solveContactBlock() in solverBlock.cuh, incl. the PxgErrorAccumulator
// statistics (the CUDA helper takes a PxgErrorAccumulator* that is null when
// residual accumulation is off; errorEnabled mirrors that pointer's nullness)
void solveContactBlockBda(Ptr<PxgBlockConstraintBatch> batches, uint batchIndex,
                          inout float3 b0LinVel, inout float3 b0AngVel,
                          inout float3 b1LinVel, inout float3 b1AngVel, bool doFriction,
                          uint threadIndex,
                          Ptr<PxgBlockSolverContactHeader> contactHeaders,
                          Ptr<PxgBlockSolverFrictionHeader> frictionHeaders,
                          Ptr<PxgBlockSolverContactPoint> contactPointsBuf,
                          Ptr<PxgBlockSolverContactFriction> frictionsBuf,
                          bool errorEnabled, inout PxgErrorAccumulatorSlang error,
                          float ref0, float ref1)
{
    const uint constraintBatchIndex = batches[batchIndex].mConstraintBatchIndex;
    const uint startConstraintIndex = batches[batchIndex].startConstraintIndex;
    const uint startFrictionIndex = batches[batchIndex].startFrictionIndex;

    float3 linVel0 = b0LinVel;
    float3 linVel1 = b1LinVel;
    float3 angVel0 = b0AngVel;
    float3 angVel1 = b1AngVel;

    const uint numNormalConstr = contactHeaders[constraintBatchIndex].numNormalConstr[threadIndex];
    const uint numFrictionConstr = frictionHeaders[constraintBatchIndex].numFrictionConstr[threadIndex];

    float accumulatedNormalImpulse = 0.0f;

    const float4 invMass0_1_angDom0_1 = contactHeaders[constraintBatchIndex].invMass0_1_angDom0_1[threadIndex];

    const float invMassA = ref0 * invMass0_1_angDom0_1.x;
    const float invMassB = ref1 * invMass0_1_angDom0_1.y;
    const float angDom0 = ref0 * invMass0_1_angDom0_1.z;
    const float angDom1 = ref1 * invMass0_1_angDom0_1.w;

    const float4 normal_staticFriction = contactHeaders[constraintBatchIndex].normal_staticFriction[threadIndex];
    const float3 normal = normal_staticFriction.xyz;
    const float staticFrictionCof = normal_staticFriction.w;

    const float restitution = contactHeaders[constraintBatchIndex].restitution[threadIndex];
    const float p8 = 0.8f;
    const uint flags = contactHeaders[constraintBatchIndex].flags[threadIndex];

    const float3 delLinVel0 = normal * invMassA;
    const float3 delLinVel1 = normal * invMassB;

    for (uint i = 0; i < numNormalConstr; i++)
    {
        const uint c = startConstraintIndex + i;

        const float4 raXn_targetVelocity = contactPointsBuf[c].raXn_targetVelocity[threadIndex];
        const float4 rbXn_maxImpulse = contactPointsBuf[c].rbXn_maxImpulse[threadIndex];
        const float appliedForce = contactPointsBuf[c].appliedForce[threadIndex];
        const float resp0 = contactPointsBuf[c].resp0[threadIndex];
        const float resp1 = contactPointsBuf[c].resp1[threadIndex];
        const float coeff0 = contactPointsBuf[c].coeff0[threadIndex];
        const float coeff1 = contactPointsBuf[c].coeff1[threadIndex];

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

        if (errorEnabled)
            pxgAccumulateErrorLocal(error, deltaF, velMultiplier);

        contactPointsBuf[c].appliedForce[threadIndex] = newForce;

        accumulatedNormalImpulse = accumulatedNormalImpulse + newForce;
    }

    if (numFrictionConstr != 0 && doFriction)
    {
        const float dynamicFrictionCof = frictionHeaders[constraintBatchIndex].dynamicFriction[threadIndex];
        const float maxFrictionImpulse = staticFrictionCof * accumulatedNormalImpulse;
        const float maxDynFrictionImpulse = dynamicFrictionCof * accumulatedNormalImpulse;

        uint broken = 0;

        for (uint i = 0; i < numFrictionConstr; i++)
        {
            const uint f = startFrictionIndex + i;

            const float4 frictionNormal = (i & 1) != 0
                ? frictionHeaders[constraintBatchIndex].frictionNormals1[threadIndex]
                : frictionHeaders[constraintBatchIndex].frictionNormals0[threadIndex];

            const float4 raXn_bias = frictionsBuf[f].raXn_bias[threadIndex];
            const float4 rbXn_targetVelW = frictionsBuf[f].rbXn_targetVelW[threadIndex];
            const float appliedForce = frictionsBuf[f].appliedForce[threadIndex];
            const float resp0 = frictionsBuf[f].resp0[threadIndex];
            const float resp1 = frictionsBuf[f].resp1[threadIndex];

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

            if (errorEnabled)
                pxgAccumulateErrorLocal(error, deltaF, velMultiplier);

            linVel0 += fdelLinVel0 * deltaF;
            linVel1 -= fdelLinVel1 * deltaF;
            angVel0 += raXn * (deltaF * angDom0);
            angVel1 -= rbXn * (deltaF * angDom1);

            frictionsBuf[f].appliedForce[threadIndex] = newAppliedForce;
            broken = broken | (clampImpulse ? 1u : 0u);
        }
        frictionHeaders[constraintBatchIndex].broken[threadIndex] = broken;
    }

    b0LinVel = linVel0;
    b0AngVel = angVel0;
    b1LinVel = linVel1;
    b1AngVel = angVel1;
}

// mirrors solve1DBlock() in solverBlock.cuh
void solve1DBlockBda(Ptr<PxgBlockConstraintBatch> batches, uint batchIndex,
                     inout float3 b0LinVel, inout float3 b0AngVel,
                     inout float3 b1LinVel, inout float3 b1AngVel, uint threadIndex,
                     Ptr<PxgBlockSolverConstraint1DHeader> jointHeaders,
                     Ptr<PxgBlockSolverConstraint1DCon> jointRowsCon,
                     Ptr<PxgBlockSolverConstraint1DMod> jointRowsMod,
                     bool residualReportingEnabled, float ref0, float ref1)
{
    const uint constraintBatchIndex = batches[batchIndex].mConstraintBatchIndex;
    const uint startConstraintIndex = batches[batchIndex].startConstraintIndex;

    float3 linVel0 = b0LinVel;
    float3 linVel1 = b1LinVel;
    float3 angVel0 = b0AngVel;
    float3 angVel1 = b1AngVel;

    const float invMass0 = ref0 * jointHeaders[constraintBatchIndex].invMass0D0[threadIndex];
    const float invMass1 = ref1 * jointHeaders[constraintBatchIndex].invMass1D1[threadIndex];
    const float invInertiaScale0 = ref0 * jointHeaders[constraintBatchIndex].invInertiaScale0[threadIndex];
    const float invInertiaScale1 = ref1 * jointHeaders[constraintBatchIndex].invInertiaScale1[threadIndex];

    // PxU16 rowCounts[32]: two lanes per word
    const uint rowCountWord = jointHeaders[constraintBatchIndex].rowCounts[threadIndex >> 1];
    const uint rowCount = (rowCountWord >> ((threadIndex & 1) * 16)) & 0xFFFFu;

    for (uint i = 0; i < rowCount; ++i)
    {
        const uint r = startConstraintIndex + i;

        const float4 _clinVel0_minImpulse = jointRowsCon[r].lin0XYZ_minImpulse[threadIndex];
        const float4 _clinVel1_maxImpulse = jointRowsCon[r].lin1XYZ_maxImpulse[threadIndex];
        const float4 _cangVel0_resp0 = jointRowsCon[r].ang0XYZ_resp0[threadIndex];
        const float4 _cangVel1_resp1 = jointRowsCon[r].ang1XYZ_resp1[threadIndex];

        const float3 clinVel0 = _clinVel0_minImpulse.xyz;
        const float3 clinVel1 = _clinVel1_maxImpulse.xyz;
        const float3 cangVel0 = _cangVel0_resp0.xyz;
        const float3 cangVel1 = _cangVel1_resp1.xyz;

        const float resp0 = ref0 * _cangVel0_resp0.w;
        const float resp1 = ref1 * _cangVel1_resp1.w;
        const float initJointSpeed = jointRowsCon[r].initJointSpeed[threadIndex];

        const float coeff0 = jointRowsMod[r].coeff0[threadIndex];
        const float coeff1 = jointRowsMod[r].coeff1[threadIndex];
        const uint flags = jointRowsMod[r].flags[threadIndex];

        const float unitResponse = resp0 + resp1;
        const float recipResponse = computeRecipUnitResponse(unitResponse, DY_MIN_RESPONSE);

        float constant, unbiasedConstant, vMul, iMul;

        const bool isSpring = (flags & DY_SC_FLAG_SPRING) != 0;
        const bool isAccelerationSpring = (flags & DY_SC_FLAG_ACCELERATION_SPRING) != 0;

        compute1dConstraintSolverConstantsPGS(isSpring, isAccelerationSpring, coeff0, coeff1,
                                              initJointSpeed, unitResponse, recipResponse, constant,
                                              unbiasedConstant, vMul, iMul);

        const float appliedForce = jointRowsMod[r].appliedForce[threadIndex];

        const float maxImpulse = _clinVel1_maxImpulse.w;
        const float minImpulse = _clinVel0_minImpulse.w;

        const float v0 = dot(linVel0, clinVel0) + dot(angVel0, cangVel0);
        const float v1 = dot(linVel1, clinVel1) + dot(angVel1, cangVel1);
        const float normalVel = v0 - v1;

        const float unclampedForce = iMul * appliedForce + (vMul * normalVel + constant);
        const float clampedForce = min(maxImpulse, max(minImpulse, unclampedForce));
        const float deltaF = clampedForce - appliedForce;

        jointRowsMod[r].appliedForce[threadIndex] = clampedForce;
        if (residualReportingEnabled)
            jointRowsMod[r].residual[threadIndex] = calculateResidual(deltaF, vMul);

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
// solveBlockPartition (BDA engine ABI) — the per-partition dynamic-vs-dynamic
// rigid contact/joint velocity solve
// ---------------------------------------------------------------------------
//
// Port of solveBlockPartition() in solverMultiBlock.cu.  The engine launches it
// ONCE PER PARTITION inside solvePartitions() (PxgCudaSolverCore.cpp), passing
// the current partition index; each launch solves exactly that one partition's
// rigid batches.  Signature:
//   (PxgSolverCoreDesc*, const PxgSolverSharedDesc<IterativeSolveData>*,
//    PxU32 islandIndex, PxU32 partitionIndex, bool doFriction)
//   -> argSizes {8,8,4,4,1}: the bool byte lands at blob offset 24 (bytes
//   25..31 are zero-filled padding), 32 bytes total.
//
// Thread layout mirrors the CUDA launch: block = SOLVE_BLOCK_PARTITION = 64
// threads FLAT (unlike the TGS solveBlockUnified's (32,2,1)); the .cu computes
// the GLOBAL warp index (threadIdx.x + blockIdx.x * blockDim.x) / 32, so
// numthreads = (64,1,1) and grid = nbBlocks (WorkgroupCount().x).  The .cu body
// is a single `if (k < endIndex)` (its strided loop is commented out); we use
// the equivalent strided loop, stride = WorkgroupCount().x * 2 — a
// behaviour-preserving superset: within one partition no two batches share a
// body, so processing several batches per warp is safe, and it makes the batch
// loads control-dependent on a real loop guard (llvmpipe zero-trip safety,
// same technique as solveBlockUnified in solverTGS.slang).
//
// PGS velocity staging: readIndex = k * 128 + lane, FOUR float4 stripes
// (lin0/ang0/lin1/ang1 at +0/+32/+64/+96) — VERIFIED against the .cu (TGS uses
// k * 192).  Writeback goes to remappedBodyA/BIndex (ComputeBodyBatchStartIndex
// is the identity) at +0/+32.
//
// The .cu reads bodyOffset / numDynamicBodies / numArticulations from
// islandContextPool[0] (islandContextPool->…), NOT islandContextPool[islandIndex]
// — mirrored exactly; the batch range comes from island[islandIndex].
struct SolveBlockPartitionAbiParams
{
    uint64_t solverCoreDesc; // @0
    uint64_t sharedDesc;     // @8
    uint islandIndex;        // @16
    uint partitionIndex;     // @20
    uint doFrictionWord;     // @24: byte 0 = doFriction (bytes 25..27 padding)
    uint abiPad0;            // @28 (blob tail padding to the 8-alignment, never read)
}                            // = 32
[[vk::push_constant]]
ConstantBuffer<SolveBlockPartitionAbiParams> gSolveAbi;

// per-warp residual-reduction scratch: block = 64 -> 2 warps of 32 lanes
// (mirrors solveBlockUnified's accumulateErrorGlobalFullWarp scratch).
groupshared float shSolveErrSum[64];
groupshared float shSolveErrMax[64];
groupshared uint  shSolveErrCount[64];

[shader("compute")]
[numthreads(64, 1, 1)]
void solveBlockPartition(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgSolverCoreDescGpu> descPtr = Ptr<PxgSolverCoreDescGpu>(gSolveAbi.solverCoreDesc);
    PxgSolverCoreDescGpu desc = descPtr[0];
    PxgSolverSharedDescPGSGpu sharedDesc = (Ptr<PxgSolverSharedDescPGSGpu>(gSolveAbi.sharedDesc))[0];
    PxgIslandContextGpu island =
        (Ptr<PxgIslandContextGpu>(desc.islandContextPool))[gSolveAbi.islandIndex];
    // The .cu reads mBodyStartIndex / mBodyCount / mArticulationCount from
    // islandContextPool[0] (islandContextPool->…); mirror that exactly.
    PxgIslandContextGpu island0 = (Ptr<PxgIslandContextGpu>(desc.islandContextPool))[0];

    const uint partitionIndex = gSolveAbi.partitionIndex;
    const uint startPartitionIndex = island.mStartPartitionIndex;

    const uint threadIndexInWarp = groupThreadId.x & 31u;
    const uint warpIndex = (groupThreadId.x + groupId.x * 64u) / 32u; // global warp index
    const uint numWarpsInGrid = WorkgroupCount().x * 2u;

    Ptr<float4> bodyVelocities = Ptr<float4>(sharedDesc.iter_solverBodyVelPool);
    Ptr<PxgBlockConstraintBatch> batches = Ptr<PxgBlockConstraintBatch>(sharedDesc.iter_blockConstraintBatch);
    Ptr<PxgBlockSolverContactHeader> contactHeaders =
        Ptr<PxgBlockSolverContactHeader>(sharedDesc.iter_blockContactHeaders);
    Ptr<PxgBlockSolverFrictionHeader> frictionHeaders =
        Ptr<PxgBlockSolverFrictionHeader>(sharedDesc.iter_blockFrictionHeaders);
    Ptr<PxgBlockSolverContactPoint> contactPoints =
        Ptr<PxgBlockSolverContactPoint>(sharedDesc.iter_blockContactPoints);
    Ptr<PxgBlockSolverContactFriction> frictionsBuf =
        Ptr<PxgBlockSolverContactFriction>(sharedDesc.iter_blockFrictions);
    Ptr<PxgBlockSolverConstraint1DHeader> jointHeaders =
        Ptr<PxgBlockSolverConstraint1DHeader>(sharedDesc.iter_blockJointConstraintHeaders);
    Ptr<PxgBlockSolverConstraint1DCon> jointRowsCon =
        Ptr<PxgBlockSolverConstraint1DCon>(sharedDesc.iter_blockJointConstraintRowsCon);
    Ptr<PxgBlockSolverConstraint1DMod> jointRowsMod =
        Ptr<PxgBlockSolverConstraint1DMod>(sharedDesc.iter_blockJointConstraintRowsMod);
    Ptr<uint> constraintsPerPartition = Ptr<uint>(desc.constraintsPerPartition);
    Ptr<uint> encodedReferenceCount = Ptr<uint>(sharedDesc.iter_solverEncodedReferenceCount);

    const uint bodyOffset = island0.mBodyStartIndex;
    const uint numDynamicBodies = island0.mBodyCount; // nbBodies minus offset!
    const uint numTotalBodies = bodyOffset + numDynamicBodies + island0.mArticulationCount;

    const bool residualAccumulationEnabled = desc.contactErrorAccumulator_mCounter >= 0;
    PxgErrorAccumulatorSlang error = { 0.0f, 0, 0.0f };

    // Null / empty-partition guard (llvmpipe): a constraint-free island leaves
    // the block-constraint pointer null/unallocated and mNumPartitions == 0; the
    // SPIR-V optimizer would speculatively hoist the first batches[] load out of
    // the (then zero-trip) batch loop and fault.  The uniform scalar guard makes
    // every dependent load control-dependent on it — semantically exact: no
    // batch buffer / no partitions == no batches to solve.  The residual
    // reduction below runs unconditionally (uniform control flow around its
    // barriers) regardless of this guard.
    if (sharedDesc.iter_blockConstraintBatch != 0 && island.mNumPartitions != 0 &&
        partitionIndex < island.mNumPartitions)
    {
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

        for (uint k = startIndex + warpIndex; k < endIndex; k += numWarpsInGrid)
        {
            const uint descStride = batches[k].descStride_constraintType & 0xFFFFu;
            const uint constraintType = batches[k].descStride_constraintType >> 16;

            const uint readIndex = k * 128u + threadIndexInWarp;

            if (threadIndexInWarp < descStride)
            {
                const float4 linVel0 = bodyVelocities[readIndex];
                const float4 angVel0 = bodyVelocities[readIndex + 32u];
                const float4 linVel1 = bodyVelocities[readIndex + 64u];
                const float4 angVel1 = bodyVelocities[readIndex + 96u];

                float3 lv0 = linVel0.xyz;
                float3 lv1 = linVel1.xyz;
                float3 av0 = angVel0.xyz;
                float3 av1 = angVel1.xyz;

                // Reference counts to be used for the current sub-timestep or
                // iteration (mass splitting across slabs).
                float curRef0 = 1.0f;
                float curRef1 = 1.0f;

                const uint bodyAIndex = batches[k].bodyAIndex[threadIndexInWarp];
                const uint bodyBIndex = batches[k].bodyBIndex[threadIndexInWarp];

                if (bodyAIndex >= bodyOffset)
                    curRef0 = float(countActiveSlabsBda(bodyAIndex, desc.numSlabs, numTotalBodies,
                                                        encodedReferenceCount));
                if (bodyBIndex >= bodyOffset)
                    curRef1 = float(countActiveSlabsBda(bodyBIndex, desc.numSlabs, numTotalBodies,
                                                        encodedReferenceCount));

                if (pxgIsContactType(constraintType))
                    solveContactBlockBda(batches, k, lv0, av0, lv1, av1,
                                         (gSolveAbi.doFrictionWord & 0xffu) != 0, threadIndexInWarp,
                                         contactHeaders, frictionHeaders, contactPoints, frictionsBuf,
                                         residualAccumulationEnabled, error, curRef0, curRef1);
                else
                    solve1DBlockBda(batches, k, lv0, av0, lv1, av1, threadIndexInWarp,
                                    jointHeaders, jointRowsCon, jointRowsMod,
                                    residualAccumulationEnabled, curRef0, curRef1);

                const uint indexA = batches[k].remappedBodyAIndex[threadIndexInWarp]; // ComputeBodyBatchStartIndex = identity
                const uint indexB = batches[k].remappedBodyBIndex[threadIndexInWarp];

                bodyVelocities[indexA] = float4(lv0, 0.0f);
                bodyVelocities[indexA + 32u] = float4(av0, 0.0f);
                bodyVelocities[indexB] = float4(lv1, 0.0f);
                bodyVelocities[indexB + 32u] = float4(av1, 0.0f);
            }
        }
    }

    // PxgErrorAccumulator::accumulateErrorGlobalFullWarp(): per-warp sum/sum/max
    // butterfly reduction, then lane 0 of each warp accumulates into
    // solverDesc->contactErrorAccumulator.  Mirrors solveBlockUnified in
    // solverTGS.slang (same 64-thread / 2-warp block).  The reduction runs
    // unconditionally (uniform control flow around the barriers); warps that
    // solved joints only reduce their zero-init accumulators, and a disabled
    // launch skips only the atomics.
    GroupMemoryBarrierWithGroupSync();
    shSolveErrSum[groupThreadId.x] = error.sumOfSquares;
    shSolveErrMax[groupThreadId.x] = error.maxError;
    shSolveErrCount[groupThreadId.x] = uint(error.counter);
    GroupMemoryBarrierWithGroupSync();

    for (uint radius = 16u; radius > 0u; radius >>= 1)
    {
        const uint partner = groupThreadId.x ^ radius; // stays inside the warp's 32 lanes
        const float s = shSolveErrSum[partner];
        const float m = shSolveErrMax[partner];
        const uint c = shSolveErrCount[partner];
        GroupMemoryBarrierWithGroupSync();
        shSolveErrSum[groupThreadId.x] += s;
        shSolveErrMax[groupThreadId.x] = max(shSolveErrMax[groupThreadId.x], m);
        shSolveErrCount[groupThreadId.x] += c;
        GroupMemoryBarrierWithGroupSync();
    }

    if (residualAccumulationEnabled && threadIndexInWarp == 0u)
    {
        const float sumSq = shSolveErrSum[groupThreadId.x];
        const uint count = shSolveErrCount[groupThreadId.x];
        const float maxErr = shSolveErrMax[groupThreadId.x];

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

// ---------------------------------------------------------------------------
// concludeBlocks (BDA engine ABI)
// ---------------------------------------------------------------------------
//
// Port of concludeBlocks() in solverMultiBlock.cu (concludeContactBlock +
// conclude1DBlock in solverBlock.cuh): switches the constraint constants to
// their unbiased variants for the velocity-iteration phase.  Unlike the TGS
// twin, the contact branch is REAL work here (coeff0 <- coeff1 for
// non-compliant contacts, friction bias zeroing).  The engine launches it with
// (PxgSolverCoreDesc*, PxgSolverSharedDesc<IterativeSolveData>*, PxU32
// islandIndex): argSizes {8,8,4} -> 24-byte blob.  One warp per batch, block =
// CONCLUDE_BLOCKS (256) threads = 8 warps, grid = nbBlocksRequired
// (WorkgroupCount().x), so the GLOBAL warp index selects the batch.
struct ConcludeBlocksAbiParams
{
    uint64_t solverCoreDesc; // @0
    uint64_t sharedDesc;     // @8
    uint islandIndex;        // @16
    uint abiPad0;            // @20 (blob tail padding, never read)
}                            // = 24
[[vk::push_constant]]
ConstantBuffer<ConcludeBlocksAbiParams> gConcludeAbi;

[shader("compute")]
[numthreads(256, 1, 1)]
void concludeBlocks(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    PxgSolverCoreDescGpu desc = (Ptr<PxgSolverCoreDescGpu>(gConcludeAbi.solverCoreDesc))[0];
    PxgSolverSharedDescPGSGpu sharedDesc =
        (Ptr<PxgSolverSharedDescPGSGpu>(gConcludeAbi.sharedDesc))[0];
    PxgIslandContextGpu island =
        (Ptr<PxgIslandContextGpu>(desc.islandContextPool))[gConcludeAbi.islandIndex];

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
    // nothing to conclude.
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
        // concludeContactBlock()
        Ptr<PxgBlockSolverContactHeader> contactHeaders =
            Ptr<PxgBlockSolverContactHeader>(sharedDesc.iter_blockContactHeaders);
        Ptr<PxgBlockSolverFrictionHeader> frictionHeaders =
            Ptr<PxgBlockSolverFrictionHeader>(sharedDesc.iter_blockFrictionHeaders);
        Ptr<PxgBlockSolverContactPoint> contactPoints =
            Ptr<PxgBlockSolverContactPoint>(sharedDesc.iter_blockContactPoints);
        Ptr<PxgBlockSolverContactFriction> frictions =
            Ptr<PxgBlockSolverContactFriction>(sharedDesc.iter_blockFrictions);

        const uint numNormalConstr = contactHeaders[batchIndex].numNormalConstr[threadIndex];
        const uint numFrictionConstr = frictionHeaders[batchIndex].numFrictionConstr[threadIndex];

        if (numNormalConstr != 0)
        {
            const float restitution = contactHeaders[batchIndex].restitution[threadIndex];
            // Assigning unbiased error to biased error (compliant contacts
            // enforce this automatically).
            if (restitution >= 0.0f)
            {
                for (uint i = 0; i < numNormalConstr; i++)
                    contactPoints[startConstraintIndex + i].coeff0[threadIndex] =
                        contactPoints[startConstraintIndex + i].coeff1[threadIndex];
            }
        }

        for (uint i = 0; i < numFrictionConstr; i++)
        {
            float4 raXn_bias = frictions[startFrictionIndex + i].raXn_bias[threadIndex];
            raXn_bias.w = 0.0f;
            frictions[startFrictionIndex + i].raXn_bias[threadIndex] = raXn_bias;
        }
    }
    else
    {
        // conclude1DBlock()
        Ptr<PxgBlockSolverConstraint1DHeader> jointHeaders =
            Ptr<PxgBlockSolverConstraint1DHeader>(sharedDesc.iter_blockJointConstraintHeaders);
        Ptr<PxgBlockSolverConstraint1DMod> jointRowsMod =
            Ptr<PxgBlockSolverConstraint1DMod>(sharedDesc.iter_blockJointConstraintRowsMod);

        const uint rowCountWord = jointHeaders[batchIndex].rowCounts[threadIndex >> 1];
        const uint rowCount = (rowCountWord >> ((threadIndex & 1) * 16)) & 0xFFFFu;

        for (uint i = 0; i < rowCount; i++)
        {
            const uint r = startConstraintIndex + i;
            // For spring constraints, it is automatically satisfied.
            if ((jointRowsMod[r].flags[threadIndex] & DY_SC_FLAG_SPRING) == 0)
                jointRowsMod[r].coeff0[threadIndex] = jointRowsMod[r].coeff1[threadIndex];
        }
    }
}

// ---------------------------------------------------------------------------
// writebackBlocks (BDA engine ABI)
// ---------------------------------------------------------------------------

// mirrors writeBackContactBlockFriction() in solverBlockCommon.cuh
void writeBackContactBlockFrictionBda(uint threadIndex, uint numFrictionConstr, uint batchIndex,
                                      uint startFrictionIndex,
                                      Ptr<PxgBlockSolverFrictionHeader> frictionHeaders,
                                      Ptr<PxgBlockSolverContactFriction> frictions,
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

// Port of writebackBlocks() in solverMultiBlock.cu (rigid path:
// writeBackContactBlock + writeBack1DBlock in solverBlock.cuh).  Engine launch:
// (PxgSolverCoreDesc*, PxgSolverSharedDesc<IterativeSolveData>*, PxU32
// islandIndex): argSizes {8,8,4} -> 24-byte blob.  One warp per batch, block =
// WRITEBACK_BLOCKS (256) = 8 warps, grid = nbBlocksRequired.  Threshold
// elements are appended per-thread through an atomic on the desc's
// sharedThresholdStreamIndex counter instead of the CUDA warp-aggregated
// shared-memory staging: the stream CONTENTS are identical, and the
// intra-stream ordering is nondeterministic in both versions.
struct WritebackBlocksAbiParams
{
    uint64_t solverCoreDesc; // @0
    uint64_t sharedDesc;     // @8
    uint islandIndex;        // @16
    uint abiPad0;            // @20 (blob tail padding, never read)
}                            // = 24
[[vk::push_constant]]
ConstantBuffer<WritebackBlocksAbiParams> gWritebackAbi;

[shader("compute")]
[numthreads(256, 1, 1)]
void writebackBlocks(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgSolverCoreDescGpu> descPtr = Ptr<PxgSolverCoreDescGpu>(gWritebackAbi.solverCoreDesc);
    PxgSolverCoreDescGpu desc = descPtr[0];
    PxgSolverSharedDescPGSGpu sharedDesc =
        (Ptr<PxgSolverSharedDescPGSGpu>(gWritebackAbi.sharedDesc))[0];
    PxgIslandContextGpu island =
        (Ptr<PxgIslandContextGpu>(desc.islandContextPool))[gWritebackAbi.islandIndex];

    const uint startIndex = island.mBatchStartIndex;
    const uint endIndex = island.mBatchCount + island.mArtiBatchCount +
        island.mStaticArtiBatchCount + island.mSelfArtiBatchCount +
        island.mStaticRigidBatchCount + startIndex;

    const uint warpSize = 32u;
    const uint warpIndex = (groupThreadId.x + groupId.x * 256u) / warpSize;
    const uint threadIndex = groupThreadId.x & (warpSize - 1u);

    const uint k = startIndex + warpIndex;
    if (k >= endIndex)
        return;

    // Null-guard (llvmpipe): see concludeBlocks.
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
        // writeBackContactBlock()
        Ptr<PxgBlockSolverContactHeader> contactHeaders =
            Ptr<PxgBlockSolverContactHeader>(sharedDesc.iter_blockContactHeaders);
        Ptr<PxgBlockSolverFrictionHeader> frictionHeaders =
            Ptr<PxgBlockSolverFrictionHeader>(sharedDesc.iter_blockFrictionHeaders);
        Ptr<PxgBlockSolverContactPoint> contactPoints =
            Ptr<PxgBlockSolverContactPoint>(sharedDesc.iter_blockContactPoints);
        Ptr<PxgBlockSolverContactFriction> frictions =
            Ptr<PxgBlockSolverContactFriction>(sharedDesc.iter_blockFrictions);
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

        writeBackContactBlockFrictionBda(threadIndex, numFrictionConstr, batchIndex, startFrictionIndex,
                                         frictionHeaders, frictions, blockFrictionPatches, frictionPatches);

        if (numFrictionConstr != 0 && frictionHeaders[batchIndex].broken[threadIndex] != 0)
            blockFrictionPatches[batchIndex].broken[threadIndex] = 1;

        const float reportThreshold0 = solverBodyData[bodyAIndex].reportThreshold;
        const float reportThreshold1 = solverBodyData[bodyBIndex].reportThreshold;

        if (forceThreshold && normalForce != 0.0f &&
            (reportThreshold0 < PX_MAX_REAL || reportThreshold1 < PX_MAX_REAL))
        {
            ThresholdStreamElement elt;
            elt.normalForce = normalForce;
            // P9.86: solverBlock.cuh:781 PxMin<float> = `a < b ? a : b` (b on a tie, NaN in b propagates); the
            // builtin min (GLSL FMin) has an implementation-defined zero sign and NaN result; CUDA gives +0 for (-0,+0).
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
        // writeBack1DBlock()
        Ptr<PxgBlockSolverConstraint1DHeader> jointHeaders =
            Ptr<PxgBlockSolverConstraint1DHeader>(sharedDesc.iter_blockJointConstraintHeaders);
        Ptr<PxgBlockSolverConstraint1DCon> jointRowsCon =
            Ptr<PxgBlockSolverConstraint1DCon>(sharedDesc.iter_blockJointConstraintRowsCon);
        Ptr<PxgBlockSolverConstraint1DMod> jointRowsMod =
            Ptr<PxgBlockSolverConstraint1DMod>(sharedDesc.iter_blockJointConstraintRowsMod);
        Ptr<PxgConstraintWriteback> constraintWriteBack =
            Ptr<PxgConstraintWriteback>(desc.constraintWriteBack);

        const uint forceWritebackOffset = jointHeaders[batchIndex].writeBackOffset[threadIndex];
        const uint breakableWord = jointHeaders[batchIndex].breakable[threadIndex >> 1];
        const uint breakable = (breakableWord >> ((threadIndex & 1) * 16)) & 0xFFFFu;
        const uint rowCountWord = jointHeaders[batchIndex].rowCounts[threadIndex >> 1];
        const uint numRows = (rowCountWord >> ((threadIndex & 1) * 16)) & 0xFFFFu;

        if (forceWritebackOffset != 0xFFFFFFFFu)
        {
            float3 linVel = float3(0.0f);
            float3 angVel = float3(0.0f);
            float constraintErrorSq = 0.0f;

            for (uint i = 0; i < numRows; ++i)
            {
                const uint r = startConstraintIndex + i;
                if ((jointRowsMod[r].flags[threadIndex] & DY_SC_FLAG_OUTPUT_FORCE) != 0)
                {
                    const float4 lin0XYZ_minImpulse = jointRowsCon[r].lin0XYZ_minImpulse[threadIndex];
                    const float3 ang0WriteBack = float3(
                        jointRowsMod[r].ang0Writeback[threadIndex * 3 + 0],
                        jointRowsMod[r].ang0Writeback[threadIndex * 3 + 1],
                        jointRowsMod[r].ang0Writeback[threadIndex * 3 + 2]);
                    const float appliedForce = jointRowsMod[r].appliedForce[threadIndex];
                    linVel += lin0XYZ_minImpulse.xyz * appliedForce;
                    angVel += ang0WriteBack * appliedForce;
                }

                const float err = jointRowsMod[r].residual[threadIndex];
                constraintErrorSq += err * err;
            }

            const float4 body0WorldOffset_linBreakImpulse =
                jointHeaders[batchIndex].body0WorldOffset_linBreakImpulse[threadIndex];
            angVel -= cross(body0WorldOffset_linBreakImpulse.xyz, linVel);

            const uint broken = breakable != 0
                // P9.47: solverBlock.cuh:840 compares linVel.magnitude() and
                // angVel.magnitude(); a last-bit difference flips the break at the limit.
                ? ((sqrt(dot(linVel, linVel)) > body0WorldOffset_linBreakImpulse.w ||
                    sqrt(dot(angVel, angVel)) > jointHeaders[batchIndex].angBreakImpulse[threadIndex]) ? 1u : 0u)
                : 0u;
            constraintWriteBack[forceWritebackOffset].angularImpulse_residual =
                float4(angVel, constraintErrorSq);
            // The broken flag is the SIGN BIT of a zero, and llvmpipe folds
            // `cond ? -0.0f : 0.0f` away — see the twin in solverTGS.slang for why.
            // Select the bit pattern, not the float.
            constraintWriteBack[forceWritebackOffset].linearImpulse_broken =
                float4(linVel, asfloat(broken != 0 ? 0x80000000u : 0u));
        }
    }
}

// ---------------------------------------------------------------------------
// writeBackBodies (BDA engine ABI)
// ---------------------------------------------------------------------------
//
// Port of writeBackBodies() in solverMultiBlock.cu: copies each body's solved
// velocities (2 float4 stripes at sharedDesc->deltaOutOffset) to the motion
// velocity array.  Engine launch: (const PxgSolverCoreDesc*, const
// PxgSolverSharedDesc<IterativeSolveData>*, PxU32 islandIndex): argSizes
// {8,8,4} -> 24-byte blob; block = WRITE_BACK_BODIES (256), one thread per
// body, grid = nbBlocksRequired.  NOTE: unlike the TGS twin (which turns
// accumulated deltas into velocities via invDt), PGS copies the velocities
// directly, spanning [bodyStartIndex, numSolverBodies).
struct WriteBackBodiesAbiParams
{
    uint64_t solverCoreDesc; // @0
    uint64_t sharedDesc;     // @8
    uint islandIndex;        // @16
    uint abiPad0;            // @20 (blob tail padding, never read)
}                            // = 24
[[vk::push_constant]]
ConstantBuffer<WriteBackBodiesAbiParams> gWriteBackBodiesAbi;

[shader("compute")]
[numthreads(256, 1, 1)]
void writeBackBodies(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    PxgSolverCoreDescGpu desc = (Ptr<PxgSolverCoreDescGpu>(gWriteBackBodiesAbi.solverCoreDesc))[0];
    PxgSolverSharedDescPGSGpu sharedDesc =
        (Ptr<PxgSolverSharedDescPGSGpu>(gWriteBackBodiesAbi.sharedDesc))[0];
    PxgIslandContextGpu island =
        (Ptr<PxgIslandContextGpu>(desc.islandContextPool))[gWriteBackBodiesAbi.islandIndex];

    Ptr<float4> bodyVelocities = Ptr<float4>(sharedDesc.iter_solverBodyVelPool);
    Ptr<float4> motionVelocities = Ptr<float4>(desc.motionVelocityArray);

    const uint bodyStartIndex = island.mBodyStartIndex;
    const uint totalBodyCount = desc.numSolverBodies;
    const uint outputOffsetIndex = sharedDesc.deltaOutOffset;

    const uint a = dispatchThreadId.x + bodyStartIndex;
    if (a >= totalBodyCount)
        return;

    motionVelocities[a] = bodyVelocities[a + outputOffsetIndex];                    // linear
    motionVelocities[a + totalBodyCount] =
        bodyVelocities[a + outputOffsetIndex + totalBodyCount];                     // angular
}

// ---------------------------------------------------------------------------
// computeAverageSolverBodyVelocity / propagateSolverBodyVelocity
// (BDA engine ABI) — multi-slab mass-splitting velocity averaging
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

// mirrors ComputeAverageBodyBatchStartIndex() in solverBlock.cuh with the PGS
// default float4sPerBody = 2 (TGS uses 3)
uint computeAverageBodyBatchStartIndex(uint bodyIndex)
{
    return 2 * (bodyIndex & ~31u) + (bodyIndex & 31u);
}

// (const PxgSolverCoreDesc*, const PxgSolverSharedDesc<IterativeSolveData>*)
struct SolverCoreSharedAbiParams
{
    uint64_t solverCoreDesc; // @0
    uint64_t sharedDesc;     // @8
}                            // = 16
[[vk::push_constant]]
ConstantBuffer<SolverCoreSharedAbiParams> gAvgAbi;

// block = (32, COMPUTE_BODIES_AVERAGE_VELOCITY/32, 1) = (32, 8, 1) -> 256
// threads; the sub-warp reduction scratch is indexed by the flat thread id.
// The reduced components are the .xyz deltas ONLY — the CUDA __shfl_xor
// butterfly shuffles just x/y/z, each lane's local .w accumulation is NOT
// reduced and lane 0's own .w survives into the final store (mirrored by
// keeping the local float4 in registers and reducing float3s through shared).
groupshared float3 gsAvgLin[256];
groupshared float3 gsAvgAng[256];
groupshared uint gsAvgHasRef[256];

// mirrors computeAverageSolverBodyVelocity() in solverMultiBlock.cu: averages
// the per-slab velocity deltas of each body (mass splitting, PGS 2-float4
// layout) and resets the slab reference counts.  Engine launch: grid =
// nbBlocksRequired, block (32, 8, 1); the CUDA warp index is
// blockIdx.x * blockDim.y + threadIdx.y -> groupId.x * 8 + groupThreadId.y.
// The CUDA sub-warp __shfl_xor reductions become groupshared reductions over
// the same power-of-two lane groups (partner = flatId ^ radius stays inside
// each 32-lane warp because radius < numThreadsPerBody <= 32).
[shader("compute")]
[numthreads(32, 8, 1)]
void computeAverageSolverBodyVelocity(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    PxgSolverCoreDescGpu desc = (Ptr<PxgSolverCoreDescGpu>(gAvgAbi.solverCoreDesc))[0];
    PxgSolverSharedDescPGSGpu sharedDesc = (Ptr<PxgSolverSharedDescPGSGpu>(gAvgAbi.sharedDesc))[0];
    PxgIslandContextGpu island0 = (Ptr<PxgIslandContextGpu>(desc.islandContextPool))[0];

    Ptr<uint> solverReferences = Ptr<uint>(desc.solverBodyReferences); // PxgSolverReferences
    Ptr<float4> bodyVelocities = Ptr<float4>(sharedDesc.iter_solverBodyVelPool);
    Ptr<uint> encodedReferenceCount = Ptr<uint>(sharedDesc.iter_solverEncodedReferenceCount);

    const uint totalBodyCount = island0.mBodyCount; // nbBodies minus offset!
    const uint bodyOffset = island0.mBodyStartIndex;
    const uint numSlabs = desc.numSlabs;
    const uint deltaVOffset = desc.accumulatedBodyDeltaVOffset;

    const uint averageOutputOffsets = (desc.numBatches + desc.numArticBatches) * 32u * 2u * 2u;

    const uint pow2Slabs = (numSlabs & (numSlabs - 1)) == 0 ? numSlabs : nextPowerOfTwo(numSlabs);
    const uint numThreadsPerBody = min(pow2Slabs, 32u);
    const uint numBodiesPerWarp = 32 / numThreadsPerBody;

    const uint warpIndex = groupId.x * 8u + groupThreadId.y;
    const uint warpBodyStartIndex = warpIndex * numBodiesPerWarp;
    const uint threadIdInWorkUnit = groupThreadId.x & (numThreadsPerBody - 1);
    const uint bodyInWarp = groupThreadId.x / numThreadsPerBody;
    const uint bodyId = warpBodyStartIndex + bodyInWarp;
    const uint flatId = groupThreadId.y * 32u + groupThreadId.x;

    float4 linDelta = float4(0.0f);
    float4 angDelta = float4(0.0f);
    bool hasRef = false;
    float4 lastLinearVel = float4(0.0f);
    float4 lastAngularVel = float4(0.0f);
    uint outputBody = 0;

    if (bodyId < totalBodyCount)
    {
        outputBody = deltaVOffset + bodyId + bodyOffset;

        lastLinearVel = bodyVelocities[outputBody];
        lastAngularVel = bodyVelocities[outputBody + totalBodyCount + bodyOffset];

        for (uint b = threadIdInWorkUnit; b < numSlabs; b += numThreadsPerBody)
        {
            const uint bodyIndex = bodyId * numSlabs + b;
            const uint remappedBodyIndex = solverReferences[bodyIndex];

            if (remappedBodyIndex != 0xFFFFFFFFu)
            {
                // Outputs are grouped into little clusters of 32...
                const uint velIndex = averageOutputOffsets + computeAverageBodyBatchStartIndex(bodyIndex);

                const float4 curLinearDeltaV = bodyVelocities[velIndex];
                const float4 curAngularDeltaV = bodyVelocities[velIndex + 32];

                linDelta += curLinearDeltaV - lastLinearVel;
                angDelta += curAngularDeltaV - lastAngularVel;
                hasRef = true;
            }
        }
    }

    // sub-group reduction over the numThreadsPerBody lanes of each body
    // (replaces the CUDA __shfl_xor_sync / __ballot_sync pattern)
    gsAvgLin[flatId] = linDelta.xyz;
    gsAvgAng[flatId] = angDelta.xyz;
    gsAvgHasRef[flatId] = hasRef ? 1u : 0u;
    GroupMemoryBarrierWithGroupSync();

    for (uint radius = numThreadsPerBody >> 1; radius > 0; radius >>= 1)
    {
        const uint partner = flatId ^ radius;
        const float3 lOther = gsAvgLin[partner];
        const float3 aOther = gsAvgAng[partner];
        const uint rOther = gsAvgHasRef[partner];
        GroupMemoryBarrierWithGroupSync();
        gsAvgLin[flatId] += lOther;
        gsAvgAng[flatId] += aOther;
        gsAvgHasRef[flatId] |= rOther;
        GroupMemoryBarrierWithGroupSync();
    }

    if (bodyId < totalBodyCount && gsAvgHasRef[flatId] != 0 && threadIdInWorkUnit == 0)
    {
        linDelta = float4(gsAvgLin[flatId], linDelta.w);
        angDelta = float4(gsAvgAng[flatId], angDelta.w);

        // Counting the number of active slabs
        const uint numTotalBodies = bodyOffset + island0.mBodyCount + island0.mArticulationCount;
        const uint referenceCount =
            countActiveSlabsBda(bodyOffset + bodyId, numSlabs, numTotalBodies, encodedReferenceCount);
        const float recipRefs = 1.0f / float(referenceCount);

        linDelta *= recipRefs;
        angDelta *= recipRefs;

        // Resetting rigid body reference count
        resetSlabCountBda(bodyOffset + bodyId, numSlabs, numTotalBodies, encodedReferenceCount);

        bodyVelocities[outputBody] = lastLinearVel + linDelta;
        bodyVelocities[outputBody + totalBodyCount + bodyOffset] = lastAngularVel + angDelta;
    }
}

// mirrors propagateSolverBodyVelocity() in solverMultiBlock.cu: copies each
// body's averaged velocity back into every slab slot that references it.
// Engine launch (solveContactMultiBlockParallel): grid = nbBlocksRequired,
// block = COMPUTE_BODIES_AVERAGE_VELOCITY (256) FLAT; the CUDA warp index is
// the global flat thread index / 32.  Same 2-ptr pack as the averaging kernel.
[shader("compute")]
[numthreads(256, 1, 1)]
void propagateSolverBodyVelocity(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    PxgSolverCoreDescGpu desc = (Ptr<PxgSolverCoreDescGpu>(gAvgAbi.solverCoreDesc))[0];
    PxgSolverSharedDescPGSGpu sharedDesc = (Ptr<PxgSolverSharedDescPGSGpu>(gAvgAbi.sharedDesc))[0];
    PxgIslandContextGpu island0 = (Ptr<PxgIslandContextGpu>(desc.islandContextPool))[0];

    Ptr<uint> solverReferences = Ptr<uint>(desc.solverBodyReferences); // PxgSolverReferences
    Ptr<float4> bodyVelocities = Ptr<float4>(sharedDesc.iter_solverBodyVelPool);

    const uint totalBodyCount = island0.mBodyCount; // nbBodies minus offset!
    const uint bodyOffset = island0.mBodyStartIndex;
    const uint numSlabs = desc.numSlabs;
    const uint deltaVOffset = desc.accumulatedBodyDeltaVOffset;

    const uint threadIndex = groupThreadId.x + groupId.x * 256u;

    const uint pow2Slabs = (numSlabs & (numSlabs - 1)) == 0 ? numSlabs : nextPowerOfTwo(numSlabs);
    const uint numThreadsPerBody = min(pow2Slabs, 32u);
    const uint numBodiesPerWarp = 32 / numThreadsPerBody;

    const uint warpIndex = threadIndex / 32u;
    const uint warpBodyStartIndex = warpIndex * numBodiesPerWarp;
    const uint threadIdInWorkUnit = groupThreadId.x & (numThreadsPerBody - 1);
    const uint bodyInWarp = (groupThreadId.x & 31u) / numThreadsPerBody;
    const uint bodyId = warpBodyStartIndex + bodyInWarp;

    if (bodyId >= totalBodyCount)
        return;

    const uint outputBody = deltaVOffset + bodyId + bodyOffset;

    float4 linVel = bodyVelocities[outputBody];
    const float4 angVel = bodyVelocities[outputBody + totalBodyCount + bodyOffset];
    linVel.w = 0.0f;

    for (uint b = threadIdInWorkUnit; b < numSlabs; b += numThreadsPerBody)
    {
        const uint bodyIndex = bodyId * numSlabs + b;
        const uint remappedBodyIndex = solverReferences[bodyIndex];
        if (remappedBodyIndex != 0xFFFFFFFFu)
        {
            const uint velIndex = remappedBodyIndex; // ComputeBodyBatchStartIndex = identity
            bodyVelocities[velIndex] = linVel;
            bodyVelocities[velIndex + 32] = angVel;
        }
    }
}

// ---------------------------------------------------------------------------
// markActiveSlabPGS (BDA engine ABI) — per-island active-slab marking for the
// mass-splitting reference counts
// ---------------------------------------------------------------------------

// mirrors checkActiveContactBlock() in solverBlock.cuh — the read-only "would
// this contact still push?" predicate markActiveSlabPGS uses to decide whether
// a body's slab is active.  A stripped variant of solveContactBlockBda: same
// per-normal-constraint deltaF computation, but it (a) never applies
// reference-count scaling (this runs BEFORE reference counts are known — it is
// what computes them), (b) never mutates the velocities between contact points
// (each point is tested against the SAME input velocities), (c) writes nothing
// back, and (d) short-circuits to true on the first contact whose |deltaF|
// exceeds 1e-8.  No friction, exactly like the .cu.
//
// The .cu prefetches contacts[0] (raXn/rbXn/appliedForce/resp/coeff) BEFORE the
// numNormalConstr loop, UNCONDITIONALLY — even for numNormalConstr == 0 lanes.
// That read is always safe: batch.startConstraintIndex points into the
// allocated contact-point pool for every batch the block prep emitted (the
// pool is sized for all batches; an "empty" contact lane still has its rows
// reserved).  Ported verbatim so the port touches exactly the memory the .cu
// touches.
bool checkActiveContactBlockBda(Ptr<PxgBlockConstraintBatch> batches, uint batchIndex,
                                float3 linVel0, float3 angVel0, float3 linVel1, float3 angVel1,
                                uint threadIndex,
                                Ptr<PxgBlockSolverContactHeader> contactHeaders,
                                Ptr<PxgBlockSolverContactPoint> contactPoints)
{
    const uint constraintBatchIndex = batches[batchIndex].mConstraintBatchIndex;
    const uint startConstraintIndex = batches[batchIndex].startConstraintIndex;

    const uint numNormalConstr = contactHeaders[constraintBatchIndex].numNormalConstr[threadIndex];

    const float4 normal_staticFriction = contactHeaders[constraintBatchIndex].normal_staticFriction[threadIndex];
    const float3 normal = normal_staticFriction.xyz;

    const float restitution = contactHeaders[constraintBatchIndex].restitution[threadIndex];
    const uint flags = contactHeaders[constraintBatchIndex].flags[threadIndex];

    // Unconditional contacts[0] prefetch (see the note above).
    float4 nextRaxn_targetVelocity = contactPoints[startConstraintIndex].raXn_targetVelocity[threadIndex];
    float4 nextRbxn_maxImpulse = contactPoints[startConstraintIndex].rbXn_maxImpulse[threadIndex];
    float nextAppliedForce = contactPoints[startConstraintIndex].appliedForce[threadIndex];
    float nextResp0 = contactPoints[startConstraintIndex].resp0[threadIndex];
    float nextResp1 = contactPoints[startConstraintIndex].resp1[threadIndex];
    float nextCoeff0 = contactPoints[startConstraintIndex].coeff0[threadIndex];
    float nextCoeff1 = contactPoints[startConstraintIndex].coeff1[threadIndex];

    for (uint i = 0; i < numNormalConstr; i++)
    {
        const float4 raXn_targetVelocity = nextRaxn_targetVelocity;
        const float4 rbXn_maxImpulse = nextRbxn_maxImpulse;
        const float appliedForce = nextAppliedForce;
        const float resp0 = nextResp0;
        const float resp1 = nextResp1;
        const float coeff0 = nextCoeff0;
        const float coeff1 = nextCoeff1;

        if ((i + 1) < numNormalConstr)
        {
            const uint c = startConstraintIndex + i + 1;
            nextRaxn_targetVelocity = contactPoints[c].raXn_targetVelocity[threadIndex];
            nextRbxn_maxImpulse = contactPoints[c].rbXn_maxImpulse[threadIndex];
            nextAppliedForce = contactPoints[c].appliedForce[threadIndex];
            nextResp0 = contactPoints[c].resp0[threadIndex];
            nextResp1 = contactPoints[c].resp1[threadIndex];
            nextCoeff0 = contactPoints[c].coeff0[threadIndex];
            nextCoeff1 = contactPoints[c].coeff1[threadIndex];
        }

        const float3 raXn = raXn_targetVelocity.xyz;
        const float3 rbXn = rbXn_maxImpulse.xyz;
        const float targetVelocity = raXn_targetVelocity.w;
        const float maxImpulse = rbXn_maxImpulse.w;

        // NO reference-count scaling (see the (a) note above).
        const float unitResponse = resp0 + resp1;
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

        // Check for active contact.
        if (abs(deltaF) > 1.0e-8f)
            return true;
    }

    return false;
}

// Port of markActiveSlabPGS() (rigid path -> markActiveSlab_rigidBodyPGS()) in
// solverMultiBlock.cu.  precomputeReferenceCount() (PxgSolverCore.cpp) launches
// it ONCE per island at partition 0 of every PGS iteration, passing the LAST
// partition index so a single launch marks every partition's batches (the batch
// range is [mBatchStartIndex, constraintsPerPartition[lastPartition +
// mStartPartitionIndex])).  For each batch lane it decides whether that
// constraint's slab is "active" (a joint is always active; a contact is active
// iff checkActiveContactBlock says it would still push) and atomicOr's the slab
// bit into solverEncodedReferenceCount[bodyId], which countActiveSlabsBda()
// later popcounts to scale invMass/response for mass splitting.
//
// Signature (PGS has NO minPen/elapsedTime — velocities alone decide):
//   (const PxgSolverCoreDesc*, const PxgSolverSharedDesc<IterativeSolveData>*,
//    PxU32 islandIndex, PxU32 lastPartition, PxgArticulationCoreDesc*)
//   -> argSizes {8,8,4,4,8} = 32-byte blob (artiDesc @24).
//
// Thread layout mirrors the CUDA launch (numThreadsPerWarp = WARP_SIZE = 32,
// numWarpsPerBlock = COMPUTE_UNCONSTRAINED_VELOCITES(64)/32 = 2):
//   numthreads = (32, 2, 1); globalWarpIndex = groupId.x * 2 + groupThreadId.y;
//   lane = groupThreadId.x; batchIndex = mBatchStartIndex + globalWarpIndex.
// The CUDA body is a single `if (batchIndex < endIndex)` (no strided loop), so
// the grid.x is sized to cover the whole batch range in one wave — mirrored 1:1.
//
// RIGID PATH ONLY (this tranche): groupId.y == 0.  The engine only launches
// blockY == 2 when the last partition has articulation batches, which a
// rigid-only scene never produces, so markActiveSlab_articulationPGS is a
// DEFERRED no-op early-return.
//
// PGS body velocities are the TWO-float4-stripe accumulated layout: linVel @
// outputOffset + bodyId, angVel @ + totalBodiesIncKinematics — NOT the TGS
// 3-stripe layout, and NOT the per-partition k*128 pool.  numDynamicBodies /
// numArticulations are read from islandContextPool[0] (the .cu uses
// islandContextPool->…) while bodyOffset and the batch range come from
// island[islandIndex] — mirrored exactly.
struct MarkActiveSlabPGSAbiParams
{
    uint64_t solverCoreDesc; // @0
    uint64_t sharedDesc;     // @8
    uint islandIndex;        // @16
    uint lastPartition;      // @20
    uint64_t artiDesc;       // @24: PxgArticulationCoreDesc* (articulation path deferred; unused here)
}                            // = 32 (maxAlign 8; @24 already 8-aligned, no padding)
[[vk::push_constant]]
ConstantBuffer<MarkActiveSlabPGSAbiParams> gMarkActiveSlabAbi;

[shader("compute")]
[numthreads(32, 2, 1)]
void markActiveSlabPGS(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    // Deferred articulation path: the engine only dispatches groupId.y != 0
    // (blockY == 2) when the last partition carries articulation batches, which
    // a rigid-only scene never produces.
    if (groupId.y != 0)
        return;

    PxgSolverCoreDescGpu desc = (Ptr<PxgSolverCoreDescGpu>(gMarkActiveSlabAbi.solverCoreDesc))[0];
    PxgSolverSharedDescPGSGpu sharedDesc =
        (Ptr<PxgSolverSharedDescPGSGpu>(gMarkActiveSlabAbi.sharedDesc))[0];
    PxgIslandContextGpu island =
        (Ptr<PxgIslandContextGpu>(desc.islandContextPool))[gMarkActiveSlabAbi.islandIndex];
    // The .cu reads mBodyCount / mArticulationCount from islandContextPool[0]
    // (islandContextPool->…), NOT islandContextPool[islandIndex]; mirror that
    // exactly.  For the single-island scenes islandIndex == 0 so it is moot.
    PxgIslandContextGpu island0 = (Ptr<PxgIslandContextGpu>(desc.islandContextPool))[0];

    const uint startPartitionIndex = island.mStartPartitionIndex;
    const uint threadIndexInWarp = groupThreadId.x;              // lane
    const uint numWarpsPerBlock = 2u;                            // numthreads.y
    const uint globalWarpIndex = groupId.x * numWarpsPerBlock + groupThreadId.y;

    Ptr<float4> bodyVelocities = Ptr<float4>(sharedDesc.iter_solverBodyVelPool);
    Ptr<PxgBlockConstraintBatch> batches = Ptr<PxgBlockConstraintBatch>(sharedDesc.iter_blockConstraintBatch);
    Ptr<PxgBlockSolverContactHeader> contactHeaders =
        Ptr<PxgBlockSolverContactHeader>(sharedDesc.iter_blockContactHeaders);
    Ptr<PxgBlockSolverContactPoint> contactPoints =
        Ptr<PxgBlockSolverContactPoint>(sharedDesc.iter_blockContactPoints);
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
    // markActiveSlabTGS / solveBlockUnified in solverTGS.slang.
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

                    const float4 linVel0 = bodyVelocities[finalIdA];
                    const float4 angVel0 = bodyVelocities[finalIdA + totalBodiesIncKinematics];
                    const float4 linVel1 = bodyVelocities[finalIdB];
                    const float4 angVel1 = bodyVelocities[finalIdB + totalBodiesIncKinematics];

                    // Check if the contact/normal constraint is active.
                    isActiveSlab = checkActiveContactBlockBda(
                        batches, batchIndex, linVel0.xyz, angVel0.xyz, linVel1.xyz, angVel1.xyz,
                        threadIndexInWarp, contactHeaders, contactPoints);
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
// solveStaticBlock + propagateStaticSolverBodyVelocities
// (P5 Step 5 BDA engine ABI) — the dynamic-vs-static rigid contact/joint solve
// ---------------------------------------------------------------------------

// solveStaticBlock: port of solveStaticBlock() in solverMultiBlock.cu,
// mirroring the engine-proven TGS twin solveStaticBlockTGS (solverTGS.slang).
// One thread per (dynamic body, static slab); each thread walks that body's
// static contact and joint batches (packed into the same block-constraint
// buffers, addressed through the per-body mRigidStatic*StartIndices) and
// writes the resulting body velocity to the per-slab tempStaticBodyOutputPool
// scratch that propagateStaticSolverBodyVelocities later averages back into
// the deltaV pool.  block = SOLVE_BLOCK_PARTITION (64) threads, grid =
// nbBlocksRequired (WorkgroupCount().x is unused — the guard is the
// per-thread nbDynamicBodiesToSolve bound).
//
// Engine launch pack (PxgCudaSolverCore.cpp solveContactMultiBlockParallel):
//   (PxgSolverCoreDesc*, const PxgSolverSharedDesc<IterativeSolveData>*,
//    PxU32 islandIndex, PxU32 nbStaticSlabs, PxU32 maxStaticPartitions,
//    bool doFriction)
//   -> argSizes {8,8,4,4,4,1}: the bool byte lands at blob offset 28 (bytes
//   29..31 zero-filled padding), 32 bytes total.
//
// PGS differences from the TGS twin (all from the .cu):
//   - the velocity pool holds TWO full float4 stripes per body (linVel @
//     inputBody, angVel @ + totalBodiesIncKinematics) — no delta stripes and
//     no packed lin/ang split;
//   - solve1DBlock takes no per-body TxI inertia data (the PGS 1D rows carry
//     everything) and no elapsedTime/minPen anywhere;
//   - the block-packed lane index is warpScanExclusive(batch.mask,
//     threadIndexInWarp) = popcount(mask & ((1<<lane)-1)); it is a pure
//     function of the STORED mask and the lane (no cross-lane ballot), so it
//     is llvmpipe-safe — same as the TGS twin;
//   - ref0/ref1 are 1 (mass-splitting is not used for static interactions);
//   - accumulateErrorGlobalFullWarp(): same 64-entry groupshared butterfly +
//     CAS-loop float atomics as solveBlockPartition above.
struct SolveStaticBlockAbiParams
{
    uint64_t solverCoreDesc;   // @0
    uint64_t sharedDesc;       // @8
    uint islandIndex;          // @16
    uint nbStaticSlabs;        // @20
    uint maxStaticPartitions;  // @24
    uint doFrictionWord;       // @28: byte 0 = doFriction (bytes 29..31 padding)
}                              // = 32
[[vk::push_constant]]
ConstantBuffer<SolveStaticBlockAbiParams> gStaticSolveAbi;

// per-warp residual-reduction scratch: block = 64 -> 2 warps of 32 lanes.
groupshared float shStaticErrSum[64];
groupshared float shStaticErrMax[64];
groupshared uint  shStaticErrCount[64];

[shader("compute")]
[numthreads(64, 1, 1)]
void solveStaticBlock(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgSolverCoreDescGpu> descPtr = Ptr<PxgSolverCoreDescGpu>(gStaticSolveAbi.solverCoreDesc);
    PxgSolverCoreDescGpu desc = descPtr[0];
    PxgSolverSharedDescPGSGpu sharedDesc =
        (Ptr<PxgSolverSharedDescPGSGpu>(gStaticSolveAbi.sharedDesc))[0];
    PxgIslandContextGpu island =
        (Ptr<PxgIslandContextGpu>(desc.islandContextPool))[gStaticSolveAbi.islandIndex];

    Ptr<float4> bodyVelocities = Ptr<float4>(sharedDesc.iter_solverBodyVelPool);
    Ptr<float4> bodyOutVelocities = Ptr<float4>(sharedDesc.iter_tempStaticBodyOutputPool);

    const uint numDynamicBodies = island.mBodyCount; // nbBodies minus offset!
    const uint numDynamicBodiesRounded = (island.mBodyCount + 31u) & ~31u; // multiple of 32
    const uint bodyOffset = island.mBodyStartIndex;
    const uint deltaVOffset = desc.accumulatedBodyDeltaVOffset;
    const uint totalBodiesIncKinematics = numDynamicBodies + bodyOffset;

    const uint nbStaticSlabs = gStaticSolveAbi.nbStaticSlabs;
    const uint maxStaticPartitions = gStaticSolveAbi.maxStaticPartitions;
    const bool doFriction = (gStaticSolveAbi.doFrictionWord & 0xffu) != 0;

    const uint nbDynamicBodiesToSolve = numDynamicBodiesRounded * nbStaticSlabs;

    const uint globalThreadIdx = groupThreadId.x + groupId.x * 64u;
    const uint warpSize = 32u;
    const uint threadIndexInWarp = groupThreadId.x & (warpSize - 1u);

    const bool residualAccumulationEnabled = desc.contactErrorAccumulator_mCounter >= 0;
    PxgErrorAccumulatorSlang error = { 0.0f, 0, 0.0f };

    Ptr<uint> rigidStaticContactCounts = Ptr<uint>(desc.mRigidStaticContactCounts);
    Ptr<uint> rigidStaticJointCounts = Ptr<uint>(desc.mRigidStaticJointCounts);
    Ptr<uint> rigidStaticContactStartIndices = Ptr<uint>(desc.mRigidStaticContactStartIndices);
    Ptr<uint> rigidStaticJointStartIndices = Ptr<uint>(desc.mRigidStaticJointStartIndices);

    Ptr<PxgBlockConstraintBatch> batches = Ptr<PxgBlockConstraintBatch>(sharedDesc.iter_blockConstraintBatch);
    Ptr<PxgBlockSolverContactHeader> contactHeaders =
        Ptr<PxgBlockSolverContactHeader>(sharedDesc.iter_blockContactHeaders);
    Ptr<PxgBlockSolverFrictionHeader> frictionHeaders =
        Ptr<PxgBlockSolverFrictionHeader>(sharedDesc.iter_blockFrictionHeaders);
    Ptr<PxgBlockSolverContactPoint> contactPoints =
        Ptr<PxgBlockSolverContactPoint>(sharedDesc.iter_blockContactPoints);
    Ptr<PxgBlockSolverContactFriction> frictionsBuf =
        Ptr<PxgBlockSolverContactFriction>(sharedDesc.iter_blockFrictions);
    Ptr<PxgBlockSolverConstraint1DHeader> jointHeaders =
        Ptr<PxgBlockSolverConstraint1DHeader>(sharedDesc.iter_blockJointConstraintHeaders);
    Ptr<PxgBlockSolverConstraint1DCon> jointRowsCon =
        Ptr<PxgBlockSolverConstraint1DCon>(sharedDesc.iter_blockJointConstraintRowsCon);
    Ptr<PxgBlockSolverConstraint1DMod> jointRowsMod =
        Ptr<PxgBlockSolverConstraint1DMod>(sharedDesc.iter_blockJointConstraintRowsMod);

    // Null-guard (llvmpipe): islands with no static rigid batches leave the
    // block-batch / count pointers null; gate every dependent load behind the
    // non-null check so the optimizer cannot hoist a BDA load out of a
    // zero-trip loop.  A body with zero static contacts/joints does no work.
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
                const uint startContactIndex = rigidStaticContactStartIndices[bodyIndex] + startIndex;
                const uint startJointIndex = rigidStaticJointStartIndices[bodyIndex] + startIndex;

                const uint inputBody = deltaVOffset + bodyIndex + bodyOffset;

                // PGS 2-stripe accumulated layout: linVel @ inputBody, angVel
                // @ + totalBodiesIncKinematics.
                float4 linVel = bodyVelocities[inputBody];
                float4 angVel = bodyVelocities[inputBody + totalBodiesIncKinematics];

                float3 lv0 = linVel.xyz;
                float3 lv1 = float3(0.0f);
                float3 av0 = angVel.xyz;
                float3 av1 = float3(0.0f);

                for (uint i = 0; i < jointCount; ++i)
                {
                    const uint bk = startJointIndex + i;
                    // warpScanExclusive(batch.mask, threadIndexInWarp)
                    const uint idx = countbits(batches[bk].mask & ((1u << threadIndexInWarp) - 1u));
                    // For interaction with static objects, mass-splitting is
                    // not used; thus, reference counts are 1.
                    solve1DBlockBda(batches, bk, lv0, av0, lv1, av1, idx, jointHeaders, jointRowsCon,
                                    jointRowsMod, residualAccumulationEnabled, 1.0f, 1.0f);
                }

                for (uint i = 0; i < contactCount; ++i)
                {
                    const uint bk = startContactIndex + i;
                    const uint idx = countbits(batches[bk].mask & ((1u << threadIndexInWarp) - 1u));
                    solveContactBlockBda(batches, bk, lv0, av0, lv1, av1, doFriction, idx,
                                         contactHeaders, frictionHeaders, contactPoints, frictionsBuf,
                                         residualAccumulationEnabled, error, 1.0f, 1.0f);
                }

                linVel.x = lv0.x; linVel.y = lv0.y; linVel.z = lv0.z;
                angVel.x = av0.x; angVel.y = av0.y; angVel.z = av0.z;

                bodyOutVelocities[bodyIndex + outputBody] = linVel;
                bodyOutVelocities[bodyIndex + outputBody + totalBodiesIncKinematics] = angVel;
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

// propagateStaticSolverBodyVelocities: port of
// propagateStaticSolverBodyVelocities() in solverMultiBlock.cu.  Averages the
// per-slab static-solve outputs (tempStaticBodyOutputPool, written by
// solveStaticBlock) back into the body's deltaV velocity slot.  One thread per
// dynamic body; block = SOLVE_BLOCK_PARTITION (64), grid = nbBlocksRequired.
//
// The .cu SIGNATURE has FIVE parameters (desc, sharedDesc, islandIndex,
// nbStaticSlabs, maxStaticPartitions — no doFriction), but the engine launches
// it with the SAME six-entry staticKernelParams array as solveStaticBlock; the
// sized launch path then appends the stray doFriction byte at blob offset 28,
// which is exactly this struct's tail padding, so both packs produce the same
// 32-byte blob.  The ABI row (PxgVulkanKernelAbi.cpp) declares the 5-arg .cu
// signature {8,8,4,4,4}; nbStaticSlabs is unused by the kernel body, exactly
// as the .cu ignores it.
struct PropagateStaticSolverBodyVelocitiesAbiParams
{
    uint64_t solverCoreDesc;   // @0
    uint64_t sharedDesc;       // @8
    uint islandIndex;          // @16
    uint nbStaticSlabs;        // @20 (unused)
    uint maxStaticPartitions;  // @24
    uint abiPad0;              // @28 (blob tail padding / the engine's stray doFriction byte, never read)
}                              // = 32
[[vk::push_constant]]
ConstantBuffer<PropagateStaticSolverBodyVelocitiesAbiParams> gStaticPropagateAbi;

[shader("compute")]
[numthreads(64, 1, 1)]
void propagateStaticSolverBodyVelocities(uint3 groupThreadId : SV_GroupThreadID,
                                         uint3 groupId : SV_GroupID)
{
    PxgSolverCoreDescGpu desc = (Ptr<PxgSolverCoreDescGpu>(gStaticPropagateAbi.solverCoreDesc))[0];
    PxgSolverSharedDescPGSGpu sharedDesc =
        (Ptr<PxgSolverSharedDescPGSGpu>(gStaticPropagateAbi.sharedDesc))[0];
    PxgIslandContextGpu island =
        (Ptr<PxgIslandContextGpu>(desc.islandContextPool))[gStaticPropagateAbi.islandIndex];

    Ptr<float4> bodyVelocities = Ptr<float4>(sharedDesc.iter_solverBodyVelPool);
    Ptr<float4> bodyOutVelocities = Ptr<float4>(sharedDesc.iter_tempStaticBodyOutputPool);
    const uint numDynamicBodies = island.mBodyCount;
    const uint bodyOffset = island.mBodyStartIndex;
    const uint deltaVOffset = desc.accumulatedBodyDeltaVOffset;
    const uint totalBodiesIncKinematics = numDynamicBodies + bodyOffset;
    const uint maxStaticPartitions = gStaticPropagateAbi.maxStaticPartitions;

    const uint globalThreadIdx = groupThreadId.x + groupId.x * 64u;

    Ptr<uint> rigidStaticContactCounts = Ptr<uint>(desc.mRigidStaticContactCounts);
    Ptr<uint> rigidStaticJointCounts = Ptr<uint>(desc.mRigidStaticJointCounts);

    // Null-guard (llvmpipe): skip when the per-body static count arrays are
    // absent (no static rigid work); the tempStaticBodyOutputPool reads below
    // are then unreachable and cannot be hoisted.
    if (globalThreadIdx < numDynamicBodies && desc.mRigidStaticContactCounts != 0 &&
        desc.mRigidStaticJointCounts != 0)
    {
        const uint contactCount = rigidStaticContactCounts[globalThreadIdx];
        const uint jointCount = rigidStaticJointCounts[globalThreadIdx];

        if (contactCount != 0u || jointCount != 0u)
        {
            const uint maxOutputs = max(contactCount, jointCount);

            // We have velocity changes we need to propagate!
            const uint outputBody = deltaVOffset + globalThreadIdx + bodyOffset;

            float scale = 0.0f;
            float4 vel0 = float4(0.0f);
            float4 vel1 = float4(0.0f);
            uint index = globalThreadIdx;
            for (uint i = 0; i < maxOutputs; i += maxStaticPartitions)
            {
                vel0 += bodyOutVelocities[index];
                vel1 += bodyOutVelocities[index + totalBodiesIncKinematics];
                index += totalBodiesIncKinematics * 2u;
                scale += 1.0f;
            }

            scale = 1.0f / scale;

            bodyVelocities[outputBody] = vel0 * scale;
            bodyVelocities[outputBody + totalBodiesIncKinematics] = vel1 * scale;
        }
    }
}
