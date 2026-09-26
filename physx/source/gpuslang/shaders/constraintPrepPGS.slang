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

// constraintPrepPGS.slang — the PGS constraint prep kernels on the BDA ENGINE
// ABI (P5 Step 5), mirroring the constraintPrepTGS.slang conversion.
//
// Slang port of gpusolver/src/CUDA/constraintBlockPrep.cu
// (+ contactConstraintBlockPrep.cuh + jointConstraintBlockPrep.cuh):
//   - contactConstraintBlockPrepareParallelLaunch
//     (createFinalizeSolverContactsBlockGPU: friction correlation +
//      setupFinalizeSolverConstraintsBlock)
//   - jointConstraintBlockPrepareParallelLaunch
//     (setupSolverConstraintBlockGPU: preprocessRows/orthogonalize +
//      setUp1DConstraintBlock)
//
// Both entries take the ENGINE's CUDA parameter pack
//   (PxgConstraintPrepareDesc*, PxgSolverSharedDesc<IterativeSolveData>*)
// as a 16-byte push-constant blob (argSizes {8,8} in
// bridge/PxgVulkanKernelAbi.cpp) and read every count / scalar / pool pointer
// on-device through the byte-exact pxgSlangPrepDesc / pxgSlangSolverDesc
// mirrors.  The batch count is derived on-device exactly like the .cu:
//   contacts: numContactBatches + numStaticContactBatches
//   joints:   num1dConstraintBatches + numStatic1dConstraintBatches
// dt / invDt come from sharedDesc->dt / sharedDesc->invDtF32; the prep
// thresholds come from the prepare desc.
//
// Thread layout mirrors the CUDA launch: block =
// CONSTRAINT_PREPARE_BLOCK_PARALLEL (64) = 2 warps; warpIndex = blockIdx.x *
// (blockDim/32) + threadIdx.x/32 selects the batch.
//
// Differences from CUDA (same policy as the solver ports):
//   - the warp-staged sqrtInvInertia loads and the __syncwarp()s around the
//     per-lane friction-patch writes become direct loads (all accesses are
//     same-lane);
//   - the PxU16 rowCounts/breakable header fields are written with
//     InterlockedAnd/Or word updates (CUDA issues native 16-bit stores);
//   - the per-warp __shared__ scratch (sortedRowIndices, angSqrtInvInertia)
//     becomes per-thread local arrays;
//   - the CUDA null blockPreviousFrictionIndices pointer check becomes a
//     device-address != 0 test (hasPrevFrictionIndices).

import pxgSlangCommon;
import pxgSlangPrepCommon;
import pxgSlangPrepDesc;
import pxgSlangSolverDesc;

// ---------------------------------------------------------------------------
// output block structures (byte-identical mirrors, same as solverPGS.slang)
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
    float4 frictionNormals0[32];  // 0
    float4 frictionNormals1[32];  // 512
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

// mirrors PxgBlockSolverConstraint1DHeader (1664 bytes; PxU16 rowCounts /
// breakable are packed two lanes per word)
struct PxgBlockSolverConstraint1DHeader
{
    uint rowCounts[16];    // 0
    uint pad0[16];
    uint breakable[16];    // 128
    uint pad1[16];
    float angBreakImpulse[32];                   // 256
    float4 body0WorldOffset_linBreakImpulse[32]; // 384
    float invMass0D0[32];                        // 896
    float invMass1D1[32];                        // 1024
    float invInertiaScale0[32];                  // 1152
    float invInertiaScale1[32];                  // 1280
    uint writeBackOffset[32];                    // 1408
    float cfm[32];                               // 1536
}                                                // = 1664

// mirrors PxgBlockSolverConstraint1DCon (2304 bytes)
struct PxgBlockSolverConstraint1DCon
{
    float4 lin0XYZ_minImpulse[32]; // 0
    float4 lin1XYZ_maxImpulse[32]; // 512
    float4 ang0XYZ_resp0[32];      // 1024
    float4 ang1XYZ_resp1[32];      // 1536
    float initJointSpeed[32];      // 2048
    uint pad0[32];                 // 2176
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
// engine-ABI push constants (shared by both entries: the CUDA parameter pack
// is identical — PxgConstraintPrepareDesc*, PxgSolverSharedDesc<...>*)
// ---------------------------------------------------------------------------

struct PgsPrepLaunch
{
    uint64_t prepDesc;   // @0 PxgConstraintPrepareDesc*
    uint64_t sharedDesc; // @8 PxgSolverSharedDesc<IterativeSolveData>*
}

[[vk::push_constant]]
ConstantBuffer<PgsPrepLaunch> gPgsPrepLaunch;
// The Ptr-typed correlation helpers live in pxgSlangPrepCommon.slang
// (P8.102b hoist).


// ---------------------------------------------------------------------------
// contact prep (setupFinalizeSolverConstraintsBlock in contactConstraintBlockPrep.cuh)
// ---------------------------------------------------------------------------

void setupFinalizeSolverConstraintsBlock(Ptr<PxgBlockContactData> gContactPrepData,
                                         Ptr<PxgBlockContactPoint> gContactPrepPoints,
                                         Ptr<PxgBlockFrictionPatch> gCurrFrictionPatches,
                                         Ptr<PxgBlockFrictionAnchorPatch> gCurrAnchors,
                                         Ptr<PxgBlockSolverContactHeader> gContactHeaders,
                                         Ptr<PxgBlockSolverFrictionHeader> gFrictionHeaders,
                                         Ptr<PxgBlockSolverContactPoint> gContactPoints,
                                         Ptr<PxgBlockSolverContactFriction> gFrictions,
                                         uint prepIndex, uint contactBase, uint contactCount,
                                         PxgAlignedTransform bodyFrame0, PxgAlignedTransform bodyFrame1,
                                         PxgSolverBodyData data0, PxgSolverBodyData data1,
                                         float4 initialLinVel0_invMassW, float4 initialAngVel0_penBiasClamp,
                                         float4 initialLinVel1_invMassW, float4 initialAngVel1_penBiasClamp,
                                         PxgMat33 sqrtInvInertia0, PxgMat33 sqrtInvInertia1, float invDtF32,
                                         float dtF32, float bounceThresholdF32, uint threadIndex,
                                         uint forceWritebackBufferOffset, bool perPointFriction,
                                         uint headerIndex, uint outContactBase, uint outFrictionBase,
                                         float restDistance, float staticFrictionCof, float dynamicFrictionCof,
                                         uint flags, float ccdMaxSeparationThreshold, float solverOffsetSlop)
{
    const float3 bodyFrame0p = bodyFrame0.p.xyz;
    const float3 bodyFrame1p = bodyFrame1.p.xyz;

    const bool isCCD = (((data0.flags | data1.flags) & PX_RIGID_BODY_FLAG_ENABLE_SPECULATIVE_CCD) != 0);
    const float ccdMaxSeparation = isCCD ? ccdMaxSeparationThreshold : PREP_PX_MAX_F32;

    const float maxPenBias = max(initialAngVel0_penBiasClamp.w, initialAngVel1_penBiasClamp.w);

    const float3 linVel0 = initialLinVel0_invMassW.xyz;
    const float3 linVel1 = initialLinVel1_invMassW.xyz;
    const float3 angVel0 = initialAngVel0_penBiasClamp.xyz;
    const float3 angVel1 = initialAngVel1_penBiasClamp.xyz;

    const float invDt = invDtF32;
    const float p8 = 0.8f;
    const float bounceThreshold = bounceThresholdF32;

    const float invDtp8 = invDt * p8;

    const float4 lin0X_ang0Y_lin1Z_ang1W = gContactPrepData[prepIndex].mInvMassScale[threadIndex];

    const float invMass0D0 = initialLinVel0_invMassW.w * lin0X_ang0Y_lin1Z_ang1W.x;
    const float invMass1D1 = initialLinVel1_invMassW.w * lin0X_ang0Y_lin1Z_ang1W.z;

    const uint anchorCount = gCurrFrictionPatches[prepIndex].anchorCount[threadIndex] * 2;

    if (contactCount == 0)
    {
        gFrictionHeaders[headerIndex].numFrictionConstr[threadIndex] = 0;
        gContactHeaders[headerIndex].numNormalConstr[threadIndex] = 0;
        gContactHeaders[headerIndex].forceWritebackOffset[threadIndex] = 0xffffffffu;
        return;
    }

    // P9.86: PxMin is a select (a < b ? a : b), not FMin (contactConstraintBlockPrep.cuh:402)
    const float maxImpulse = (data0.maxImpulse < data1.maxImpulse) ? data0.maxImpulse : data1.maxImpulse;

    gContactHeaders[headerIndex].forceWritebackOffset[threadIndex] = forceWritebackBufferOffset;
    gContactHeaders[headerIndex].flags[threadIndex] = flags;
    gContactHeaders[headerIndex].invMass0_1_angDom0_1[threadIndex] =
        float4(invMass0D0, invMass1D1, lin0X_ang0Y_lin1Z_ang1W.y, lin0X_ang0Y_lin1Z_ang1W.w);

    const float4 normal4_rW = gContactPrepData[prepIndex].normal_restitutionW[threadIndex];
    const float3 normal = normal4_rW.xyz;
    const float restitution = normal4_rW.w;
    const float damping = gContactPrepData[prepIndex].damping[threadIndex];
    const float normalLenSq = dot(normal, normal);
    const float norVel0 = dot(normal, linVel0);
    const float norVel1 = dot(normal, linVel1);
    const float linNorVel = norVel0 - norVel1;

    const float invMassNorLenSq0 = invMass0D0 * normalLenSq;
    const float invMassNorLenSq1 = invMass1D1 * normalLenSq;

    gContactHeaders[headerIndex].restitution[threadIndex] = restitution;

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

        const float resp0 = lin0X_ang0Y_lin1Z_ang1W.y * dot(raXnSqrtInertia, raXnSqrtInertia) + invMassNorLenSq0;
        const float resp1 = lin0X_ang0Y_lin1Z_ang1W.w * dot(rbXnSqrtInertia, rbXnSqrtInertia) + invMassNorLenSq1;

        const float angNorVel0 = dot(raXn, angVel0);
        const float angNorVel1 = dot(rbXn, angVel1);

        const float vrel = linNorVel + (angNorVel0 - angNorVel1);

        const float penetration = separation - restDistance;
        const bool isSeparated = (penetration >= 0.0f);
        const float penetrationInvDt = penetration * invDt;
        const float penetrationInvDtPt8 = max(maxPenBias, penetration * invDtp8);

        const bool collidingWithVrel = ((-vrel) > penetrationInvDt);
        const bool isGreater2 = ((restitution > 0.0f) && (bounceThreshold > vrel)) && collidingWithVrel;

        const bool ccdSeparationCondition = ccdMaxSeparation >= penetration;

        float targetVelocity = cTargetVel + (isGreater2 ? (-vrel) * restitution : 0.0f);
        targetVelocity = targetVelocity - vrel;

        float coeff0, coeff1;
        if (restitution < 0.0f)
        {
            queryReducedCompliantContactCoefficients(dtF32, flags, restitution, damping, penetration,
                                                     targetVelocity, isSeparated, collidingWithVrel, coeff0,
                                                     coeff1);
        }
        else
        {
            float scaledBias = isSeparated ? penetrationInvDt : penetrationInvDtPt8;
            if (ccdSeparationCondition && isGreater2)
                scaledBias = 0.0f;

            float unbiasedErr = targetVelocity;
            const float biasedErr = unbiasedErr - scaledBias;

            if (!isGreater2)
                unbiasedErr -= max(0.0f, scaledBias);

            coeff0 = biasedErr;
            coeff1 = unbiasedErr;
        }

        gContactPoints[outContactBase + j].raXn_targetVelocity[threadIndex] = float4(raXnSqrtInertia, targetVelocity);
        gContactPoints[outContactBase + j].rbXn_maxImpulse[threadIndex] =
            // P9.86: PxMin is a select (a < b ? a : b), not FMin (contactConstraintBlockPrep.cuh:513)
            float4(rbXnSqrtInertia, (maxImpulse < targetVel4_maxImpulseW.w) ? maxImpulse : targetVel4_maxImpulseW.w);
        gContactPoints[outContactBase + j].resp0[threadIndex] = resp0;
        gContactPoints[outContactBase + j].resp1[threadIndex] = resp1;
        gContactPoints[outContactBase + j].coeff0[threadIndex] = coeff0;
        gContactPoints[outContactBase + j].coeff1[threadIndex] = coeff1;
        gContactPoints[outContactBase + j].appliedForce[threadIndex] = 0.0f;
    }

    const uint aCount = gCurrFrictionPatches[prepIndex].anchorCount[threadIndex];

    const float frictionScale = (aCount == 2) ? 0.5f : 1.0f;

    const float staticFriction = staticFrictionCof * frictionScale;
    const float dynamicFriction = dynamicFrictionCof * frictionScale;

    const bool haveFriction = (anchorCount != 0);
    gContactHeaders[headerIndex].numNormalConstr[threadIndex] = contactCount;
    gFrictionHeaders[headerIndex].numFrictionConstr[threadIndex] = anchorCount;

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
        // P9.47: artiConstraintPrep2.cu:468 is `t0.normalize()` -- PxVec3::normalize
        // (PxVec3.h:295) takes magnitude() = PxSqrt(msq), then scales through
        // operator/= (PxVec3.h:260): ONE reciprocal and three multiplies, under an
        // m > 0 guard.  GLSL.std.450 Normalize is neither of those things.
        {
            const float m = sqrt(dot(t0, t0));
            if (m > 0.0f)
                t0 = t0 * (1.0f / m);
        }

        const float3 t1 = cross(normal, t0);

        gFrictionHeaders[headerIndex].broken[threadIndex] = 0;
        gFrictionHeaders[headerIndex].frictionNormals0[threadIndex] = float4(t0, 0.0f);
        gFrictionHeaders[headerIndex].frictionNormals1[threadIndex] = float4(t1, 0.0f);

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

                const float resp0 = lin0X_ang0Y_lin1Z_ang1W.y * dot(raXnSqrtInertia, raXnSqrtInertia) + invMass0D0;
                const float resp1 = lin0X_ang0Y_lin1Z_ang1W.w * dot(rbXnSqrtInertia, rbXnSqrtInertia) + invMass1D1;

                float targetVel = dot(tvel, t0);

                const float vrel1 = dot(t0, linVel0) + dot(raXn, angVel0);
                const float vrel2 = dot(t0, linVel1) + dot(rbXn, angVel1);
                const float vrelF = vrel1 - vrel2;

                targetVel = targetVel - vrelF;

                const float bias = dot(t0, error) * invDt;

                gFrictions[f0].resp0[threadIndex] = resp0;
                gFrictions[f0].resp1[threadIndex] = resp1;
                gFrictions[f0].raXn_bias[threadIndex] = float4(raXnSqrtInertia, bias);
                gFrictions[f0].rbXn_targetVelW[threadIndex] = float4(rbXnSqrtInertia, targetVel);
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

                const float resp0 = lin0X_ang0Y_lin1Z_ang1W.y * dot(raXnSqrtInertia, raXnSqrtInertia) + invMass0D0;
                const float resp1 = lin0X_ang0Y_lin1Z_ang1W.w * dot(rbXnSqrtInertia, rbXnSqrtInertia) + invMass1D1;

                float targetVel = dot(tvel, t1);

                const float vrel1 = dot(t1, linVel0) + dot(raXn, angVel0);
                const float vrel2 = dot(t1, linVel1) + dot(rbXn, angVel1);
                const float vrelF = vrel1 - vrel2;

                targetVel = targetVel - vrelF;

                const float bias = dot(t1, error) * invDt;

                gFrictions[f1].resp0[threadIndex] = resp0;
                gFrictions[f1].resp1[threadIndex] = resp1;
                gFrictions[f1].raXn_bias[threadIndex] = float4(raXnSqrtInertia, bias);
                gFrictions[f1].rbXn_targetVelW[threadIndex] = float4(rbXnSqrtInertia, targetVel);
                gFrictions[f1].appliedForce[threadIndex] = 0.0f;
            }
        }
    }
}

// mirrors contactConstraintBlockPrepareParallelLaunch() +
// createFinalizeSolverContactsBlockGPU() on the ENGINE ABI: warpIndex =
// blockIdx.x * (blockDim/32) + threadIdx.x/32 selects the batch,
// nbContactBatches = numContactBatches + numStaticContactBatches from the desc.
[shader("compute")]
[numthreads(64, 1, 1)]
void contactConstraintBlockPrepareParallelLaunch(uint3 groupThreadId : SV_GroupThreadID,
                                                 uint3 groupId : SV_GroupID)
{
    const uint blockStride = 64u / 32u;
    const uint warpIndex = groupId.x * blockStride + groupThreadId.x / 32u;
    const uint threadIndexInWarp = groupThreadId.x & 31u;

    Ptr<PxgConstraintPrepareDescGpu> prepDesc =
        Ptr<PxgConstraintPrepareDescGpu>(gPgsPrepLaunch.prepDesc);
    Ptr<PxgSolverSharedDescPGSGpu> sharedDesc =
        Ptr<PxgSolverSharedDescPGSGpu>(gPgsPrepLaunch.sharedDesc);

    const uint totalPreviousEdges = prepDesc[0].totalPreviousEdges;
    const uint totalCurrentEdges = prepDesc[0].totalCurrentEdges;
    const uint nbContactBatches = prepDesc[0].numContactBatches + prepDesc[0].numStaticContactBatches;

    const uint i = warpIndex;
    if (i >= nbContactBatches)
        return;

    Ptr<PxgBlockWorkUnit> gWorkUnitsP = Ptr<PxgBlockWorkUnit>(prepDesc[0].blockWorkUnit);
    Ptr<uint> batchIndices = Ptr<uint>(prepDesc[0].contactConstraintBatchIndices);
    Ptr<PxgBlockConstraintBatch> batches =
        Ptr<PxgBlockConstraintBatch>(sharedDesc[0].iter_blockConstraintBatch);
    Ptr<PxgSolverBodyData> solverBodyDatas = Ptr<PxgSolverBodyData>(prepDesc[0].solverBodyDataPool);
    Ptr<PxgSolverTxIData> solverTxIDatas = Ptr<PxgSolverTxIData>(prepDesc[0].solverBodyTxIDataPool);
    Ptr<PxgAlignedTransform> bodyFrames = Ptr<PxgAlignedTransform>(prepDesc[0].body2WorldPool);
    Ptr<PxgBlockContactData> contactPrepPool =
        Ptr<PxgBlockContactData>(prepDesc[0].blockContactCurrentPrepPool);
    Ptr<PxgBlockContactPoint> contactPrepPoints = Ptr<PxgBlockContactPoint>(prepDesc[0].blockContactPoints);
    Ptr<PxgBlockFrictionPatch> currFrictionPatches =
        Ptr<PxgBlockFrictionPatch>(sharedDesc[0].blockCurrentFrictionPatches);
    Ptr<PxgBlockFrictionPatch> prevFrictionPatches =
        Ptr<PxgBlockFrictionPatch>(sharedDesc[0].blockPreviousFrictionPatches);
    Ptr<PxgBlockFrictionAnchorPatch> currAnchors =
        Ptr<PxgBlockFrictionAnchorPatch>(prepDesc[0].blockCurrentAnchorPatches);
    Ptr<PxgBlockFrictionAnchorPatch> prevAnchors =
        Ptr<PxgBlockFrictionAnchorPatch>(prepDesc[0].blockPreviousAnchorPatches);
    Ptr<uint2> currFrictionIndices = Ptr<uint2>(prepDesc[0].blockCurrentFrictionIndices);
    Ptr<uint2> prevFrictionIndices = Ptr<uint2>(prepDesc[0].blockPreviousFrictionIndices);
    Ptr<PxgBlockSolverContactHeader> contactHeaders =
        Ptr<PxgBlockSolverContactHeader>(sharedDesc[0].iter_blockContactHeaders);
    Ptr<PxgBlockSolverFrictionHeader> frictionHeaders =
        Ptr<PxgBlockSolverFrictionHeader>(sharedDesc[0].iter_blockFrictionHeaders);
    Ptr<PxgBlockSolverContactPoint> contactPoints =
        Ptr<PxgBlockSolverContactPoint>(sharedDesc[0].iter_blockContactPoints);
    Ptr<PxgBlockSolverContactFriction> frictions =
        Ptr<PxgBlockSolverContactFriction>(sharedDesc[0].iter_blockFrictions);

    const uint batchIndex = batchIndices[i];

    const uint descStride = batches[batchIndex].descStride_constraintType & 0xFFFFu;
    if (threadIndexInWarp >= descStride)
        return;

    const uint bodyAIndex = batches[batchIndex].bodyAIndex[threadIndexInWarp];
    const uint bodyBIndex = batches[batchIndex].bodyBIndex[threadIndexInWarp];
    const uint descIndexBatch = batches[batchIndex].mConstraintBatchIndex;

    // CUDA stages the sqrtInvInertia tensors through shared memory for
    // coalescing; direct loads are semantically identical.
    const PxgMat33 invInertia0 = mat33Load(solverTxIDatas[bodyAIndex].sqrtInvInertia);
    const PxgMat33 invInertia1 = mat33Load(solverTxIDatas[bodyBIndex].sqrtInvInertia);

    // Fill in correlation information for next frame...
    const uint edgeIndex = gWorkUnitsP[descIndexBatch].mEdgeIndex[threadIndexInWarp];
    const uint frictionIndex =
        edgeIndex + totalCurrentEdges * gWorkUnitsP[descIndexBatch].mPatchIndex[threadIndexInWarp];
    // PxgBlockFrictionIndex::createPatchIndex(): PxU64 = patchIndex << 5 | lane
    currFrictionIndices[frictionIndex] =
        uint2((descIndexBatch << 5) | threadIndexInWarp, descIndexBatch >> 27);

    const PxgSolverBodyData bodyData0 = solverBodyDatas[bodyAIndex];
    const PxgSolverBodyData bodyData1 = solverBodyDatas[bodyBIndex];

    const PxgAlignedTransform bodyFrame0 = bodyFrames[bodyAIndex];
    const PxgAlignedTransform bodyFrame1 = bodyFrames[bodyBIndex];

    const float4 linVel_invMass0 = bodyData0.initialLinVelXYZ_invMassW;
    const float4 angVelXYZ_penBiasClamp0 = bodyData0.initialAngVelXYZ_penBiasClamp;
    const float4 linVel_invMass1 = bodyData1.initialLinVelXYZ_invMassW;
    const float4 angVelXYZ_penBiasClamp1 = bodyData1.initialAngVelXYZ_penBiasClamp;

    const float solverOffsetSlop = max(bodyData0.offsetSlop, bodyData1.offsetSlop);

    const uint forceWritebackOffset = gWorkUnitsP[descIndexBatch].mWriteback[threadIndexInWarp];

    // ---- createFinalizeSolverContactsBlockGPU ----

    const float4 data_ = contactPrepPool[descIndexBatch].contactData[threadIndexInWarp];
    // PxgMaterialContactData: x=restDistance, y=staticFriction, z=dynamicFriction,
    // w packs mNumContacts | mSolverFlags << 8 | prevFrictionPatchCount << 16
    const uint materialWord = asuint(data_.w);
    const uint nbContacts = materialWord & 0xFFu;
    const uint flags = (materialWord >> 8) & 0xFFu;
    const uint prevFrictionPatchCount = (materialWord >> 16) & 0xFFu;
    const float restDistance = data_.x;
    const float staticFrictionCof = data_.y;
    const float dynamicFrictionCof = data_.z;

    const float4 normal4 = contactPrepPool[descIndexBatch].normal_restitutionW[threadIndexInWarp];
    const float3 normal = normal4.xyz;

    const uint contactBase = batches[batchIndex].blockContactIndex;

    correlatePatchesPtr(currFrictionPatches, descIndexBatch, nbContacts, normal, bodyFrame0, bodyFrame1,
                        threadIndexInWarp);

    const bool perPointFriction = (flags & PXG_SC_FLAG_PER_POINT_FRICTION) != 0;
    const bool disableFriction = (flags & PXG_SC_FLAG_DISABLE_FRICTION) != 0;

    // KS - ensure that the friction patch broken bit is set to 0
    currFrictionPatches[descIndexBatch].broken[threadIndexInWarp] = 0;

    float patchExtents = 0.0f;

    // (the CUDA __syncwarp() here guards only same-lane data; no cross-lane
    // accesses exist in the correlation functions)

    if (!(perPointFriction || disableFriction))
    {
        getFrictionPatchesPtr(currFrictionPatches, descIndexBatch, currAnchors, descIndexBatch,
                              prevFrictionIndices, prepDesc[0].blockPreviousFrictionIndices != 0, edgeIndex,
                              prevFrictionPatches, prevAnchors, prevFrictionPatchCount, bodyFrame0,
                              bodyFrame1, prepDesc[0].correlationDistance, threadIndexInWarp,
                              totalPreviousEdges, patchExtents, nbContacts);
    }

    if (!disableFriction)
        growPatchesPtr(currFrictionPatches, descIndexBatch, currAnchors, descIndexBatch, contactPrepPoints,
                       contactBase, nbContacts, bodyFrame0, bodyFrame1,
                       prepDesc[0].frictionOffsetThreshold + restDistance, threadIndexInWarp, patchExtents);

    setupFinalizeSolverConstraintsBlock(contactPrepPool, contactPrepPoints, currFrictionPatches, currAnchors,
                                        contactHeaders, frictionHeaders, contactPoints, frictions,
                                        descIndexBatch, contactBase, nbContacts, bodyFrame0, bodyFrame1,
                                        bodyData0, bodyData1, linVel_invMass0, angVelXYZ_penBiasClamp0,
                                        linVel_invMass1, angVelXYZ_penBiasClamp1, invInertia0, invInertia1,
                                        sharedDesc[0].invDtF32, sharedDesc[0].dt,
                                        prepDesc[0].bounceThresholdF32, threadIndexInWarp,
                                        forceWritebackOffset, perPointFriction, descIndexBatch,
                                        batches[batchIndex].startConstraintIndex,
                                        batches[batchIndex].startFrictionIndex, restDistance,
                                        staticFrictionCof, dynamicFrictionCof, flags,
                                        prepDesc[0].ccdMaxSeparation, solverOffsetSlop);

    // ---- back in the kernel wrapper ----

    currFrictionPatches[descIndexBatch].patchIndex[threadIndexInWarp] =
        gWorkUnitsP[descIndexBatch].mFrictionPatchIndex[threadIndexInWarp];

    // PxSave3(bodyFrame0.transform(PxLoad3(anchor))): PxVec3 path, w = 0
    if (currFrictionPatches[descIndexBatch].anchorCount[threadIndexInWarp] >= 1)
        currFrictionPatches[descIndexBatch].anchorPoints0[threadIndexInWarp] = float4(
            alignedTransformVec(bodyFrame0, currAnchors[descIndexBatch].body0Anchors0[threadIndexInWarp].xyz), 0.0f);
    if (currFrictionPatches[descIndexBatch].anchorCount[threadIndexInWarp] == 2)
        currFrictionPatches[descIndexBatch].anchorPoints1[threadIndexInWarp] = float4(
            alignedTransformVec(bodyFrame0, currAnchors[descIndexBatch].body0Anchors1[threadIndexInWarp].xyz), 0.0f);
}

// ---------------------------------------------------------------------------
// joint prep (jointConstraintBlockPrep.cuh)
// ---------------------------------------------------------------------------

// mirrors orthogonalize() in jointConstraintBlockPrep.cuh; operates on the
// rows gJointVel[rowBase + sortedRowIndices[startOffset + i]] and the local
// angSqrtInvInertia arrays (sub-range [startOffset, startOffset + rowCount))
void orthogonalize(Ptr<PxgBlockConstraint1DVelocities> gJointVel,
                   uint sortedRowIndices[MAX_CONSTRAINT_ROWS], uint startOffset, uint rowBase,
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
            const float a = s > 0.0f ? 1.0f / s : 0.0f; // with mass scaling, the inner product can be zero

            lin0m[i] = l0m * a;
            ang0m[i] = a0m * a;
            lin1m[i] = l1m * a;
            ang1m[i] = a1m * a;
        }
    }
}

// mirrors preprocessRows() in jointConstraintBlockPrep.cuh
void preprocessRows(Ptr<PxgBlockConstraint1DData> gJointData,
                    Ptr<PxgBlockConstraint1DVelocities> gJointVel,
                    Ptr<PxgBlockConstraint1DParameters> gJointPar,
                    inout uint sortedRowIndices[MAX_CONSTRAINT_ROWS], uint jointDataIndex, uint rowBase,
                    out float3 angSqrtInvInertia0[MAX_CONSTRAINT_ROWS],
                    out float3 angSqrtInvInertia1[MAX_CONSTRAINT_ROWS], float4 invMassScale, float iMass0,
                    float iMass1, PxgMat33 i0, PxgMat33 i1, uint threadIndex, bool disablePreprocessing)
{
    const uint numRows = gJointData[jointDataIndex].mNumRows[threadIndex];

    // insertion sort by solveHint
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
                uint bCount = start; // count of bilateral constraints
                for (; bCount < i &&
                       (gJointPar[rowBase + sortedRowIndices[bCount]].solveHint[threadIndex] & 255u) == 0;
                     bCount++)
                    ;

                orthogonalize(gJointVel, sortedRowIndices, start, rowBase, angSqrtInvInertia0,
                              angSqrtInvInertia1, i - start, bCount - start, invMass0, invMass1,
                              invInertiaScale0, invInertiaScale1, threadIndex);
            }
        }
    }
}

// packed-PxU16 store into a uint[16] header field (CUDA issues native 16-bit
// stores; adjacent lanes share a word, so the halves are updated atomically)
void storeHeaderU16(Ptr<PxgBlockSolverConstraint1DHeader> gJointHeaders, uint headerIndex,
                    bool isBreakable, uint threadIndex, uint value)
{
    const uint word = threadIndex >> 1;
    const uint shift = (threadIndex & 1) * 16;
    if (isBreakable)
    {
        uint prev;
        InterlockedAnd(gJointHeaders[headerIndex].breakable[word], ~(0xFFFFu << shift), prev);
        InterlockedOr(gJointHeaders[headerIndex].breakable[word], (value & 0xFFFFu) << shift, prev);
    }
    else
    {
        uint prev;
        InterlockedAnd(gJointHeaders[headerIndex].rowCounts[word], ~(0xFFFFu << shift), prev);
        InterlockedOr(gJointHeaders[headerIndex].rowCounts[word], (value & 0xFFFFu) << shift, prev);
    }
}

// mirrors jointConstraintBlockPrepareParallelLaunch() +
// setupSolverConstraintBlockGPU() + setUp1DConstraintBlock() on the ENGINE
// ABI: warpIndex = blockIdx.x * (blockDim/32) + threadIdx.x/32 selects the
// batch, num1dConstraintBatches = num1dConstraintBatches +
// numStatic1dConstraintBatches from the desc.
[shader("compute")]
[numthreads(64, 1, 1)]
void jointConstraintBlockPrepareParallelLaunch(uint3 groupThreadId : SV_GroupThreadID,
                                               uint3 groupId : SV_GroupID)
{
    const uint blockStride = 64u / 32u;
    const uint warpIndex = groupId.x * blockStride + groupThreadId.x / 32u;
    const uint threadIndexInWarp = groupThreadId.x & 31u;

    Ptr<PxgConstraintPrepareDescGpu> prepDesc =
        Ptr<PxgConstraintPrepareDescGpu>(gPgsPrepLaunch.prepDesc);
    Ptr<PxgSolverSharedDescPGSGpu> sharedDesc =
        Ptr<PxgSolverSharedDescPGSGpu>(gPgsPrepLaunch.sharedDesc);

    const uint num1dConstraintBatches =
        prepDesc[0].num1dConstraintBatches + prepDesc[0].numStatic1dConstraintBatches;

    const uint i = warpIndex;
    if (i >= num1dConstraintBatches)
        return;

    Ptr<uint> batchIndices = Ptr<uint>(prepDesc[0].jointConstraintBatchIndices);
    Ptr<PxgBlockConstraintBatch> batches =
        Ptr<PxgBlockConstraintBatch>(sharedDesc[0].iter_blockConstraintBatch);
    Ptr<PxgSolverBodyData> solverBodyDatas = Ptr<PxgSolverBodyData>(prepDesc[0].solverBodyDataPool);
    Ptr<PxgSolverTxIData> solverTxIDatas = Ptr<PxgSolverTxIData>(prepDesc[0].solverBodyTxIDataPool);
    Ptr<PxgBlockConstraint1DData> gJointData = Ptr<PxgBlockConstraint1DData>(prepDesc[0].blockJointPrepPool);
    Ptr<PxgBlockConstraint1DVelocities> gJointVel =
        Ptr<PxgBlockConstraint1DVelocities>(prepDesc[0].blockJointPrepPool0);
    Ptr<PxgBlockConstraint1DParameters> gJointPar =
        Ptr<PxgBlockConstraint1DParameters>(prepDesc[0].blockJointPrepPool1);
    Ptr<uint> constraintUniqueIndices = Ptr<uint>(prepDesc[0].constraintUniqueIndices);
    Ptr<uint2> solverConstantData = Ptr<uint2>(prepDesc[0].solverConstantData); // PxgSolverConstraintManagerConstants
    Ptr<PxgBlockSolverConstraint1DHeader> gJointHeaders =
        Ptr<PxgBlockSolverConstraint1DHeader>(sharedDesc[0].iter_blockJointConstraintHeaders);
    Ptr<PxgBlockSolverConstraint1DCon> gJointRowsCon =
        Ptr<PxgBlockSolverConstraint1DCon>(sharedDesc[0].iter_blockJointConstraintRowsCon);
    Ptr<PxgBlockSolverConstraint1DMod> gJointRowsMod =
        Ptr<PxgBlockSolverConstraint1DMod>(sharedDesc[0].iter_blockJointConstraintRowsMod);

    const uint batchIndex = batchIndices[i];

    const uint descStride = batches[batchIndex].descStride_constraintType & 0xFFFFu;
    if (threadIndexInWarp >= descStride)
        return;

    const uint bodyAIndex = batches[batchIndex].bodyAIndex[threadIndexInWarp];
    const uint bodyBIndex = batches[batchIndex].bodyBIndex[threadIndexInWarp];
    const uint descIndexBatch = batches[batchIndex].mConstraintBatchIndex;
    const uint rowBase = descIndexBatch * MAX_CONSTRAINT_ROWS;
    const uint outRowBase = batches[batchIndex].startConstraintIndex;

    const uint uniqueIndex =
        constraintUniqueIndices[batches[batchIndex].mStartPartitionIndex + threadIndexInWarp];

    const PxgSolverBodyData bodyData0 = solverBodyDatas[bodyAIndex];
    const PxgSolverBodyData bodyData1 = solverBodyDatas[bodyBIndex];

    const float dt = sharedDesc[0].dt;
    const float recipdt = sharedDesc[0].invDtF32;

    // ---- setupSolverConstraintBlockGPU ----

    const uint numRows = gJointData[descIndexBatch].mNumRows[threadIndexInWarp];

    storeHeaderU16(gJointHeaders, descIndexBatch, false, threadIndexInWarp, numRows & 0xFFu); // PxU8(mNumRows)

    gJointHeaders[descIndexBatch].writeBackOffset[threadIndexInWarp] = solverConstantData[uniqueIndex].y;

    const float4 lin0_ang0_lin1_ang1 = gJointData[descIndexBatch].mInvMassScale[threadIndexInWarp];

    const float4 initialLinVelXYZ_invMassW0 = bodyData0.initialLinVelXYZ_invMassW;
    const float4 initialLinVelXYZ_invMassW1 = bodyData1.initialLinVelXYZ_invMassW;

    const float4 raWorld_linBreakForce = gJointData[descIndexBatch].mRAWorld_linBreakForce[threadIndexInWarp];
    const float4 rbWorld_angBreakForce = gJointData[descIndexBatch].mRBWorld_AngBreakForce[threadIndexInWarp];
    const float linBreakImpulse = raWorld_linBreakForce.w * dt;
    const float angBreakForce = rbWorld_angBreakForce.w;
    const float angBreakImpulse = angBreakForce * dt;
    gJointHeaders[descIndexBatch].body0WorldOffset_linBreakImpulse[threadIndexInWarp] =
        float4(raWorld_linBreakForce.xyz, linBreakImpulse);
    gJointHeaders[descIndexBatch].angBreakImpulse[threadIndexInWarp] = angBreakImpulse;

    gJointHeaders[descIndexBatch].invMass0D0[threadIndexInWarp] =
        initialLinVelXYZ_invMassW0.w * lin0_ang0_lin1_ang1.x;
    gJointHeaders[descIndexBatch].invMass1D1[threadIndexInWarp] =
        initialLinVelXYZ_invMassW1.w * lin0_ang0_lin1_ang1.z;
    gJointHeaders[descIndexBatch].invInertiaScale0[threadIndexInWarp] = lin0_ang0_lin1_ang1.y;
    gJointHeaders[descIndexBatch].invInertiaScale1[threadIndexInWarp] = lin0_ang0_lin1_ang1.w;

    storeHeaderU16(gJointHeaders, descIndexBatch, true, threadIndexInWarp,
                   ((raWorld_linBreakForce.w != PREP_PX_MAX_F32) || (angBreakForce != PREP_PX_MAX_F32)) ? 1u : 0u);

    const bool disablePreprocessing =
        (gJointData[descIndexBatch].mFlags[threadIndexInWarp] & PX_CONSTRAINT_FLAG_DISABLE_PREPROCESSING) != 0;

    uint sortedRowIndices[MAX_CONSTRAINT_ROWS];
    float3 angSqrtInvInertia0[MAX_CONSTRAINT_ROWS];
    float3 angSqrtInvInertia1[MAX_CONSTRAINT_ROWS];

    const PxgMat33 i0 = mat33Load(solverTxIDatas[bodyAIndex].sqrtInvInertia);
    const PxgMat33 i1 = mat33Load(solverTxIDatas[bodyBIndex].sqrtInvInertia);

    preprocessRows(gJointData, gJointVel, gJointPar, sortedRowIndices, descIndexBatch, rowBase,
                   angSqrtInvInertia0, angSqrtInvInertia1, lin0_ang0_lin1_ang1,
                   initialLinVelXYZ_invMassW0.w, initialLinVelXYZ_invMassW1.w, i0, i1,
                   threadIndexInWarp, disablePreprocessing);

    // ---- setUp1DConstraintBlock ----

    const float erp = 1.0f;

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
        const float3 ang0 = angSqrtInvInertia0[r];
        const float3 ang1 = angSqrtInvInertia1[r];

        const uint rpFlags = gJointPar[index].flags[threadIndexInWarp];

        float minImpulse, maxImpulse;
        {
            const bool hasDriveLimit = (rpFlags & PX1D_FLAG_HAS_DRIVE_LIMIT) != 0;
            const bool driveLimitsAreForces =
                (gJointData[descIndexBatch].mFlags[threadIndexInWarp] & PX_CONSTRAINT_FLAG_DRIVE_LIMITS_ARE_FORCES) != 0;
            computeMinMaxImpulseOrForceAsImpulse(c_linear1XYZ_minImpulseW.w, c_angular1XYZ_maxImpulseW.w,
                                                 hasDriveLimit, driveLimitsAreForces, dt, minImpulse,
                                                 maxImpulse);
        }

        gJointRowsMod[outRow].ang0Writeback[threadIndexInWarp * 3 + 0] = cang0.x;
        gJointRowsMod[outRow].ang0Writeback[threadIndexInWarp * 3 + 1] = cang0.y;
        gJointRowsMod[outRow].ang0Writeback[threadIndexInWarp * 3 + 2] = cang0.z;

        const float resp0 = dot(clin0, clin0) * initialLinVelXYZ_invMassW0.w * lin0_ang0_lin1_ang1.x +
                            dot(ang0, ang0) * lin0_ang0_lin1_ang1.y;
        const float resp1 = dot(clin1, clin1) * initialLinVelXYZ_invMassW1.w * lin0_ang0_lin1_ang1.z +
                            dot(ang1, ang1) * lin0_ang0_lin1_ang1.w;

        const float initJointSpeed =
            projectVelocity(bodyData0, clin0, cang0) - projectVelocity(bodyData1, clin1, cang1);

        // ---- intializeBlock1D ----
        {
            const float springStiffness = gJointPar[index].springStiffness_restitution[threadIndexInWarp];
            const float springDamping = gJointPar[index].springDamping_velocityThreshold[threadIndexInWarp];
            const float restitution = gJointPar[index].springStiffness_restitution[threadIndexInWarp];
            const float bounceThreshold = gJointPar[index].springDamping_velocityThreshold[threadIndexInWarp];
            const float geomError = c_linear0XYZ_geometricErrorW.w;
            const float velocityTarget = c_angular0XYZ_velocityTargetW.w;

            float coeff0, coeff1;
            queryReduced1dConstraintSolverConstantsPGS(rpFlags, springStiffness, springDamping, restitution,
                                                       bounceThreshold, geomError, velocityTarget,
                                                       initJointSpeed, erp, dt, recipdt, coeff0, coeff1);

            gJointRowsCon[outRow].lin0XYZ_minImpulse[threadIndexInWarp] = float4(clin0, minImpulse);
            gJointRowsCon[outRow].lin1XYZ_maxImpulse[threadIndexInWarp] = float4(clin1, maxImpulse);
            gJointRowsCon[outRow].ang0XYZ_resp0[threadIndexInWarp] = float4(ang0, resp0);
            gJointRowsCon[outRow].ang1XYZ_resp1[threadIndexInWarp] = float4(ang1, resp1);
            gJointRowsCon[outRow].initJointSpeed[threadIndexInWarp] = initJointSpeed;

            gJointRowsMod[outRow].coeff0[threadIndexInWarp] = coeff0;
            gJointRowsMod[outRow].coeff1[threadIndexInWarp] = coeff1;
            gJointRowsMod[outRow].appliedForce[threadIndexInWarp] = 0.0f;
            gJointRowsMod[outRow].residual[threadIndexInWarp] = 0.0f;

            uint modFlags = 0;
            if ((rpFlags & PX1D_FLAG_SPRING) != 0)
                modFlags |= DY_SC_FLAG_SPRING;
            if ((rpFlags & PX1D_FLAG_ACCELERATION_SPRING) != 0)
                modFlags |= DY_SC_FLAG_ACCELERATION_SPRING;
            if ((rpFlags & PX1D_FLAG_OUTPUT_FORCE) != 0)
                modFlags |= DY_SC_FLAG_OUTPUT_FORCE;
            gJointRowsMod[outRow].flags[threadIndexInWarp] = modFlags;
        }
    }
}
