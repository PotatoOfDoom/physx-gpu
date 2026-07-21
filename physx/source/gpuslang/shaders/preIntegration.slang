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

// preIntegration.slang
//
// Slang port of the PGS solver pre-integration kernel
// (gpusolver/src/CUDA/preIntegration.cu / preIntegration.cuh):
//   - preIntegrationLaunch
//
// Per solver body this kernel:
//   - applies gravity, external accelerations and damping and clamps the
//     velocities (bodyCoreComputeUnconstrainedVelocity),
//   - applies the implicit gyroscopic torque model when enabled,
//   - computes the world-space sqrt(inv inertia) tensor,
//   - emits the solver body data / TxI data / transform / velocity pools.
//
// The CUDA kernel stages PxgBodySim loads and PxgSolverBodyData stores
// through bank-conflict-free swizzled shared memory (readSwizzledFloat4 &
// friends).  That is a memory-coalescing optimization with no semantic
// effect; the port uses direct structured-buffer accesses instead.
// Angular velocity is written to the velocity pool in "momocity" format
// (premultiplied by the world-space sqrt inertia), exactly like CUDA.

import pxgSlangCommon;

static const uint BLOCK_SIZE = 128; // PxgKernelBlockDim::PRE_INTEGRATION

// mirrors computeSafeSqrtInertia() in preIntegration.cuh
float3 computeSafeSqrtInertia(float3 v)
{
    return float3(v.x == 0.0f ? 0.0f : sqrt(v.x),
                  v.y == 0.0f ? 0.0f : sqrt(v.y),
                  v.z == 0.0f ? 0.0f : sqrt(v.z));
}

// mirrors getGravityAcceleration() in preIntegration.cuh
float3 getGravityAcceleration(uint disableGravity, float accelScale, float3 gravity, float dt)
{
    if (disableGravity == 0)
        return gravity * accelScale * dt;
    return float3(0.0f);
}

// mirrors bodyCoreComputeUnconstrainedVelocity() in preIntegration.cuh
void bodyCoreComputeUnconstrainedVelocity(
    float4 maxLinearVelocitySqX_maxAngularVelocitySqY_linearDampingZ_angularDampingW,
    inout float4 linearVelocityXYZ_inverseMassW,
    inout float4 angularVelocityXYZ_maxPenBiasW,
    uint lockFlags, uint disableGravity, float accelScale, float3 gravity,
    float4 linearAccel, float4 angularAccel, float dt)
{
    float3 linearVelocity = linearVelocityXYZ_inverseMassW.xyz;
    float3 angularVelocity = angularVelocityXYZ_maxPenBiasW.xyz;
    const float4 temp = maxLinearVelocitySqX_maxAngularVelocitySqY_linearDampingZ_angularDampingW;

    const float3 linearAccelTimesDT = getGravityAcceleration(disableGravity, accelScale, gravity, dt) + linearAccel.xyz * dt;
    const float3 angularAccelTimesDT = angularAccel.xyz * dt;
    const float linearDampingTimesDT = temp.z * dt;
    const float angularDampingTimesDT = temp.w * dt;
    const float oneMinusLinearDampingTimesDT = 1.0f - linearDampingTimesDT;
    const float oneMinusAngularDampingTimesDT = 1.0f - angularDampingTimesDT;

    linearVelocity += linearAccelTimesDT;
    angularVelocity += angularAccelTimesDT;

    // Apply damping.
    const float linVelMultiplier = fsel(oneMinusLinearDampingTimesDT, oneMinusLinearDampingTimesDT, 0.0f);
    const float angVelMultiplier = fsel(oneMinusAngularDampingTimesDT, oneMinusAngularDampingTimesDT, 0.0f);
    linearVelocity *= linVelMultiplier;
    angularVelocity *= angVelMultiplier;

    // Clamp velocity
    const float angVelSq = dot(angularVelocity, angularVelocity);
    if (angVelSq > temp.y)
        angularVelocity *= sqrt(temp.y / angVelSq);

    const float linVelSq = dot(linearVelocity, linearVelocity);
    if (linVelSq > temp.x)
        linearVelocity *= sqrt(temp.x / linVelSq);

    linearVelocityXYZ_inverseMassW.x = (lockFlags & PXG_LOCK_LINEAR_X) != 0 ? 0.0f : linearVelocity.x;
    linearVelocityXYZ_inverseMassW.y = (lockFlags & PXG_LOCK_LINEAR_Y) != 0 ? 0.0f : linearVelocity.y;
    linearVelocityXYZ_inverseMassW.z = (lockFlags & PXG_LOCK_LINEAR_Z) != 0 ? 0.0f : linearVelocity.z;
    angularVelocityXYZ_maxPenBiasW.x = (lockFlags & PXG_LOCK_ANGULAR_X) != 0 ? 0.0f : angularVelocity.x;
    angularVelocityXYZ_maxPenBiasW.y = (lockFlags & PXG_LOCK_ANGULAR_Y) != 0 ? 0.0f : angularVelocity.y;
    angularVelocityXYZ_maxPenBiasW.z = (lockFlags & PXG_LOCK_ANGULAR_Z) != 0 ? 0.0f : angularVelocity.z;
}

// mirrors preIntegration() in preIntegration.cuh (per-body math; the shared
// memory staging of the CUDA original is a coalescing optimization only).
// Pure compute shared by the bound-buffer PGS entry (preIntegrationLaunch)
// and the BDA engine-ABI TGS entry (preIntegrationLaunchTGS): given a body's
// PxgBodySim + its solver index, produce the four solver outputs.  `a` is the
// solver body index, `nbSolverBodies` the pool stride for the angular slot.
void preIntegrateBodyCompute(PxgBodySim bodySim, uint a, uint nbSolverBodies, float dt,
                             float3 gravity, bool skipGravityApplication, uint2 islandNodeIndex,
                             out float4 outLinVel, out float4 outAngVel,
                             out PxgAlignedTransform outBody2World, out PxgSolverBodyData outData,
                             out PxgSolverTxIData outTxIData)
{
    float4 linearVelocityXYZ_inverseMassW = bodySim.linearVelocityXYZ_inverseMassW;
    float4 angularVelocityXYZ_maxPenBiasW = bodySim.angularVelocityXYZ_maxPenBiasW;
    const float4 sleepAngVelAccXYZ_accelScaleW = bodySim.sleepAngVelAccXYZ_accelScaleW;
    const float4 inverseInertiaXYZ_contactReportThresholdW = bodySim.inverseInertiaXYZ_contactReportThresholdW;
    const float4 maxLinearVelocitySqX_maxAngularVelocitySqY_linearDampingZ_angularDampingW =
        bodySim.maxLinearVelocitySqX_maxAngularVelocitySqY_linearDampingZ_angularDampingW;
    const float maxImpulse = bodySim.body2Actor_maxImpulseW.p.w;
    const uint internalFlags = bodySim.internalFlags;
    const uint lockFlags = bodySim.lockFlags_disableGravity & 0xFFFFu;
    const uint disableGravity = bodySim.lockFlags_disableGravity >> 16;
    const float offsetSlop = bodySim.offsetSlop;

    outBody2World.q = bodySim.body2World.q;
    outBody2World.p = float4(bodySim.body2World.p.xyz, 0.0f);

    const float4 linearAccel = skipGravityApplication ? float4(0.0f) : bodySim.externalLinearAcceleration;
    const float4 angularAccel = skipGravityApplication ? float4(0.0f) : bodySim.externalAngularAcceleration;

    bodyCoreComputeUnconstrainedVelocity(
        maxLinearVelocitySqX_maxAngularVelocitySqY_linearDampingZ_angularDampingW,
        linearVelocityXYZ_inverseMassW, angularVelocityXYZ_maxPenBiasW,
        lockFlags, (disableGravity != 0 || skipGravityApplication) ? 1u : 0u,
        sleepAngVelAccXYZ_accelScaleW.w, gravity, linearAccel, angularAccel, dt);

    // initialize solver body data
    const float3 inverseInertia = inverseInertiaXYZ_contactReportThresholdW.xyz;
    const float3 safeSqrtInvInertia = computeSafeSqrtInertia(inverseInertia);
    const PxgMat33 rotation = mat33FromQuat(outBody2World.q);
    const PxgMat33 sqrtInvInertia = transformInertiaTensor(safeSqrtInvInertia, rotation);

    outLinVel = linearVelocityXYZ_inverseMassW;

    // KS - to make this compatible with the rigid body particle system, we
    // store the angular velocity in the gOutVelocityPool in momocity format!
    const float3 sqrtInertiaV = float3(
        safeSqrtInvInertia.x == 0.0f ? 0.0f : 1.0f / safeSqrtInvInertia.x,
        safeSqrtInvInertia.y == 0.0f ? 0.0f : 1.0f / safeSqrtInvInertia.y,
        safeSqrtInvInertia.z == 0.0f ? 0.0f : 1.0f / safeSqrtInvInertia.z);

    const PxgMat33 sqrtInertia = transformInertiaTensor(sqrtInertiaV, rotation);

    float3 angVel = angularVelocityXYZ_maxPenBiasW.xyz;

    if ((internalFlags & PXG_BODY_ENABLE_GYROSCOPIC) != 0)
    {
        const float3 localInertia = float3(
            inverseInertia.x == 0.0f ? 0.0f : 1.0f / inverseInertia.x,
            inverseInertia.y == 0.0f ? 0.0f : 1.0f / inverseInertia.y,
            inverseInertia.z == 0.0f ? 0.0f : 1.0f / inverseInertia.z);

        const float3 localAngVel = quatRotateInv(outBody2World.q, angVel);
        const float3 origMom = localInertia * localAngVel;
        const float3 torque = -cross(localAngVel, origMom);
        float3 newMom = origMom + torque * dt;
        const float denom = length(newMom);
        const float ratio = denom > 0.0f ? length(origMom) / denom : 0.0f;
        newMom *= ratio;
        const float3 newDeltaAngVel = quatRotate(outBody2World.q, inverseInertia * newMom - localAngVel);

        angVel += newDeltaAngVel;
    }

    angularVelocityXYZ_maxPenBiasW.xyz = angVel;

    angVel = mat33Transform(sqrtInertia, angVel);

    outAngVel = float4(angVel, angularVelocityXYZ_maxPenBiasW.w);

    PxgSolverBodyData data;
    data.body2World = outBody2World;
    data.initialLinVelXYZ_invMassW = linearVelocityXYZ_inverseMassW;
    data.initialAngVelXYZ_penBiasClamp = angularVelocityXYZ_maxPenBiasW;
    data.offsetSlop = offsetSlop;
    data.reportThreshold = inverseInertiaXYZ_contactReportThresholdW.w;
    data.islandNodeIndex = islandNodeIndex;
    data.maxImpulse = maxImpulse;
    uint flags = 0;
    if ((internalFlags & PXG_BODY_SPECULATIVE_CCD) != 0)
        flags |= PXG_RB_FLAG_ENABLE_SPECULATIVE_CCD;
    if ((internalFlags & PXG_BODY_ENABLE_GYROSCOPIC) != 0)
        flags |= PXG_RB_FLAG_ENABLE_GYROSCOPIC;
    data.flags = flags;
    data.pad = uint2(0, 0);
    outData = data;

    PxgSolverTxIData txIData;
    txIData.deltaBody2World = makeTransform28(float4(0, 0, 0, 1), float3(0.0f)); // PxTransform(PxIdentity)
    mat33Store(sqrtInvInertia, txIData.sqrtInvInertia);
    outTxIData = txIData;
}

// mirrors preIntegrationLaunch() in preIntegration.cu — the PGS entry point,
// P5 Step 5 BDA engine-ABI form.  The CUDA entry just forwards to the same
// preIntegration() as the TGS twin WITHOUT the trailing bool (preIntegration's
// skipGravityApplication parameter defaults to false).  The push-constant
// struct mirrors the CUDA parameter list EXACTLY (offset, nbSolverBodies, dt,
// PxVec3 gravity by value, then eight device pointers), natural alignment,
// 88-byte blob.  solverBodySleepDataPool is passed to match the ABI but unused
// (the CUDA preIntegration's sleep-data write is #if 0'd out).
struct PreIntegrationPgsParams
{
    uint offset;                       // @0
    uint nbSolverBodies;               // @4
    float dt;                          // @8
    float gravityX, gravityY, gravityZ;// @12,16,20
    uint64_t solverBodyDataPool;       // @24 PxgSolverBodyData*
    uint64_t solverBodySleepDataPool;  // @32 (unused)
    uint64_t solverTxIDataPool;        // @40 PxgSolverTxIData*
    uint64_t bodySimPool;              // @48 PxgBodySim*
    uint64_t islandNodeIndices;        // @56 PxNodeIndex* (uint2)
    uint64_t transforms;               // @64 PxAlignedTransform*
    uint64_t outVelocityPool;          // @72 float4*
    uint64_t solverBodyIndices;        // @80 PxU32*
}                                      // = 88 (maxAlign 8)
[[vk::push_constant]]
ConstantBuffer<PreIntegrationPgsParams> gPrePgs;

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void preIntegrationLaunch(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint idx = dispatchThreadId.x;
    const uint a = idx + gPrePgs.offset;

    if (a >= gPrePgs.nbSolverBodies)
        return;

    const float3 gravity = float3(gPrePgs.gravityX, gPrePgs.gravityY, gPrePgs.gravityZ);

    Ptr<PxgBodySim> bodySimPool = Ptr<PxgBodySim>(gPrePgs.bodySimPool);
    Ptr<uint2> islandNodeIndices = Ptr<uint2>(gPrePgs.islandNodeIndices);
    Ptr<uint> solverBodyIndices = Ptr<uint>(gPrePgs.solverBodyIndices);
    Ptr<float4> outVelocityPool = Ptr<float4>(gPrePgs.outVelocityPool);
    Ptr<PxgAlignedTransform> transforms = Ptr<PxgAlignedTransform>(gPrePgs.transforms);
    Ptr<PxgSolverBodyData> solverBodyDataPool = Ptr<PxgSolverBodyData>(gPrePgs.solverBodyDataPool);
    Ptr<PxgSolverTxIData> solverTxIDataPool = Ptr<PxgSolverTxIData>(gPrePgs.solverTxIDataPool);

    const uint2 islandNodeIndex = islandNodeIndices[a];
    const uint nodeIndex = nodeIndexIndex(islandNodeIndex);
    solverBodyIndices[nodeIndex] = a;

    float4 outLinVel, outAngVel;
    PxgAlignedTransform outBody2World;
    PxgSolverBodyData outData;
    PxgSolverTxIData outTxIData;
    preIntegrateBodyCompute(bodySimPool[nodeIndex], a, gPrePgs.nbSolverBodies, gPrePgs.dt, gravity,
                            /*skipGravityApplication=*/false, islandNodeIndex, outLinVel, outAngVel,
                            outBody2World, outData, outTxIData);

    outVelocityPool[a] = outLinVel;
    outVelocityPool[a + gPrePgs.nbSolverBodies] = outAngVel;
    transforms[a] = outBody2World;
    solverBodyDataPool[a] = outData;
    solverTxIDataPool[a] = outTxIData;
}

// mirrors initStaticKinematics() in preIntegration.cu — P5 Step 4 BDA
// engine-launch ABI: the push-constant struct mirrors the CUDA parameter list
// EXACTLY (two PxU32 counts then six device pointers, natural alignment).
// The pool structs are the bit-verified pxgSlangCommon mirrors, walked
// through Ptr<T>.
struct InitStaticKinematicsParams
{
    uint nbStaticKinematics;      // @0
    uint nbSolverBodies;          // @4
    uint64_t solverBodyDataPool;  // @8  PxgSolverBodyData*
    uint64_t solverTxIDataPool;   // @16 PxgSolverTxIData*
    uint64_t transforms;          // @24 PxAlignedTransform*
    uint64_t outVelocityPool;     // @32 float4*
    uint64_t activeNodeIndices;   // @40 PxNodeIndex*
    uint64_t solverBodyIndices;   // @48 PxU32*
}                                 // = 56
[[vk::push_constant]]
ConstantBuffer<InitStaticKinematicsParams> gStaticKine;

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void initStaticKinematics(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint idx = dispatchThreadId.x;

    if (idx >= gStaticKine.nbStaticKinematics)
        return;

    Ptr<PxgSolverBodyData> solverBodyDataPool = Ptr<PxgSolverBodyData>(gStaticKine.solverBodyDataPool);
    Ptr<PxgSolverTxIData> solverTxIDataPool = Ptr<PxgSolverTxIData>(gStaticKine.solverTxIDataPool);
    Ptr<PxgAlignedTransform> transforms = Ptr<PxgAlignedTransform>(gStaticKine.transforms);
    Ptr<float4> outVelocityPool = Ptr<float4>(gStaticKine.outVelocityPool);
    Ptr<uint2> activeNodeIndices = Ptr<uint2>(gStaticKine.activeNodeIndices);
    Ptr<uint> solverBodyIndices = Ptr<uint>(gStaticKine.solverBodyIndices);

    const uint2 index = activeNodeIndices[idx];
    if (!nodeIndexIsStaticBody(index))
        solverBodyIndices[nodeIndexIndex(index)] = idx;

    const PxgSolverBodyData data = solverBodyDataPool[idx];
    transforms[idx] = data.body2World;
    outVelocityPool[idx] = data.initialLinVelXYZ_invMassW;
    outVelocityPool[idx + gStaticKine.nbSolverBodies] = data.initialAngVelXYZ_penBiasClamp;

    PxgSolverTxIData txIData;
    txIData.deltaBody2World = makeTransform28(float4(0, 0, 0, 1), float3(0.0f));
    float zero9[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    txIData.sqrtInvInertia = zero9;
    solverTxIDataPool[idx] = txIData;
}

// mirrors preIntegrationLaunchTGS() in preIntegrationTGS.cu — the CUDA TGS
// entry point just forwards to the same preIntegration() as the PGS entry
// (see the "merge PRE_INTEGRATION and TGS_PRE_INTEGRATION" TODO in the engine)
// plus a trailing bool skipGravityApplication.  P5 Step 4 BDA engine-ABI form:
// the push-constant struct mirrors the CUDA parameter list EXACTLY (offset,
// nbSolverBodies, dt, PxVec3 gravity, then eight device pointers, then the
// bool), natural alignment; pools are the bit-verified pxgSlangCommon mirrors
// walked through Ptr<T>.  solverBodySleepDataPool is passed to match the ABI
// but unused (the CUDA preIntegration's sleep-data write is #if 0'd out).
struct PreIntegrationTgsParams
{
    uint offset;                       // @0
    uint nbSolverBodies;               // @4
    float dt;                          // @8
    float gravityX, gravityY, gravityZ;// @12,16,20
    uint64_t solverBodyDataPool;       // @24 PxgSolverBodyData*
    uint64_t solverBodySleepDataPool;  // @32 (unused)
    uint64_t solverTxIDataPool;        // @40 PxgSolverTxIData*
    uint64_t bodySimPool;              // @48 PxgBodySim*
    uint64_t islandNodeIndices;        // @56 PxNodeIndex* (uint2)
    uint64_t transforms;               // @64 PxAlignedTransform*
    uint64_t outVelocityPool;          // @72 float4*
    uint64_t solverBodyIndices;        // @80 PxU32*
    uint skipGravityApplication;       // @88 bool (masked word)
}                                      // = 96 (rounded to 8)
[[vk::push_constant]]
ConstantBuffer<PreIntegrationTgsParams> gPreTgs;

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void preIntegrationLaunchTGS(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint idx = dispatchThreadId.x;
    const uint a = idx + gPreTgs.offset;

    if (a >= gPreTgs.nbSolverBodies)
        return;

    const float3 gravity = float3(gPreTgs.gravityX, gPreTgs.gravityY, gPreTgs.gravityZ);
    const bool skipGravityApplication = (gPreTgs.skipGravityApplication & 0xFFu) != 0;

    Ptr<PxgBodySim> bodySimPool = Ptr<PxgBodySim>(gPreTgs.bodySimPool);
    Ptr<uint2> islandNodeIndices = Ptr<uint2>(gPreTgs.islandNodeIndices);
    Ptr<uint> solverBodyIndices = Ptr<uint>(gPreTgs.solverBodyIndices);
    Ptr<float4> outVelocityPool = Ptr<float4>(gPreTgs.outVelocityPool);
    Ptr<PxgAlignedTransform> transforms = Ptr<PxgAlignedTransform>(gPreTgs.transforms);
    Ptr<PxgSolverBodyData> solverBodyDataPool = Ptr<PxgSolverBodyData>(gPreTgs.solverBodyDataPool);
    Ptr<PxgSolverTxIData> solverTxIDataPool = Ptr<PxgSolverTxIData>(gPreTgs.solverTxIDataPool);

    const uint2 islandNodeIndex = islandNodeIndices[a];
    const uint nodeIndex = nodeIndexIndex(islandNodeIndex);
    solverBodyIndices[nodeIndex] = a;

    float4 outLinVel, outAngVel;
    PxgAlignedTransform outBody2World;
    PxgSolverBodyData outData;
    PxgSolverTxIData outTxIData;
    preIntegrateBodyCompute(bodySimPool[nodeIndex], a, gPreTgs.nbSolverBodies, gPreTgs.dt, gravity,
                            skipGravityApplication, islandNodeIndex, outLinVel, outAngVel,
                            outBody2World, outData, outTxIData);

    outVelocityPool[a] = outLinVel;
    outVelocityPool[a + gPreTgs.nbSolverBodies] = outAngVel;
    transforms[a] = outBody2World;
    solverBodyDataPool[a] = outData;
    solverTxIDataPool[a] = outTxIData;
}
