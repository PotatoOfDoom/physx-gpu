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

// integration.slang
//
// Slang port of the solver integration kernels
// (gpusolver/src/CUDA/integration.cu, integrationTGS.cu, integration.cuh):
//   - integrateCoreParallelLaunch    (PGS: integrateCore + sleepCheck +
//     updateWakeCounter)
//   - integrateCoreParallelLaunchTGS (TGS: integrateCoreTGS — absolute
//     velocities, delta-transform pose accumulation)
//
// Per solver body this kernel:
//   - finalizes the post-solver velocities (adds initial velocities, converts
//     the angular "momocity" back through sqrtInvInertia, applies lock flags),
//   - runs the sleep/freeze/wake-counter state machine (updateWakeCounter),
//   - integrates the pose (linear position + closed-form quaternion step),
//   - writes back velocities/pose to the body sim and the output pools.
//
// The PxgSolverCoreDesc / PxgSolverSharedDesc descriptor structs embed device
// pointers under CUDA; their buffer members are bound explicitly here and the
// scalar members arrive as push constants.

import pxgSlangCommon;
import pxgSlangSolverDesc;

static const uint BLOCK_SIZE = 128;

// mirrors updateWakeCounter() in integration.cuh
void updateWakeCounter(inout bool freeze, inout float4 solverBodyLinVel, inout float4 solverBodyAngVel,
                       PxgAlignedTransform body2World, inout PxgBodySim bodySim, out PxgSolverBodySleepData sleepData,
                       float4 inverseInertia, float3 linearMotionVel, float3 angularMotionVel,
                       float invertedMass, float dt, float invDt, bool enableStabilization,
                       bool hasStaticTouch, uint numCountedInteractions)
{
    const float wakeCounterResetTime = 20.0f * 0.02f;

    float4 freezeThresholdX_wakeCounterY_sleepThresholdZ_bodySimIndex = bodySim.freezeThresholdX_wakeCounterY_sleepThresholdZ_bodySimIndex;
    const float4 sleepLinVelAccXYZ_freezeCountW = bodySim.sleepLinVelAccXYZ_freezeCountW;
    const float4 sleepAngVelAccXYZ_accelScaleW = bodySim.sleepAngVelAccXYZ_accelScaleW;

    float wc = freezeThresholdX_wakeCounterY_sleepThresholdZ_bodySimIndex.y;

    uint flags = bodySim.internalFlags &
        (PXG_BODY_DISABLE_GRAVITY_GPU | PXG_BODY_FROZEN | PXG_BODY_ENABLE_GYROSCOPIC | PXG_BODY_RETAIN_ACCELERATION);
    float freezeCount = sleepLinVelAccXYZ_freezeCountW.w;
    float accelScale = sleepAngVelAccXYZ_accelScaleW.w;

    bool alreadyUpdateWC = false;
    float3 sleepLinVelAcc = float3(0.0f);
    float3 sleepAngVelAcc = float3(0.0f);

    if (enableStabilization)
    {
        const uint maxCountedInteractions = 10u;

        // calculate normalized energy: kinetic energy divided by mass
        const float3 inertia = float3(
            inverseInertia.x > 0.0f ? 1.0f / inverseInertia.x : 1.0f,
            inverseInertia.y > 0.0f ? 1.0f / inverseInertia.y : 1.0f,
            inverseInertia.z > 0.0f ? 1.0f / inverseInertia.z : 1.0f);
        sleepLinVelAcc = linearMotionVel;
        sleepAngVelAcc = angularMotionVel;

        // scale threshold by cluster factor (more contacts => higher sleep threshold)
        const uint clusterFactor = min(numCountedInteractions, maxCountedInteractions);

        float invMass = invertedMass;
        if (invMass == 0.0f)
            invMass = 1.0f;

        const float angular = dot(sleepAngVelAcc * sleepAngVelAcc, inertia) * invMass;
        const float linear = dot(sleepLinVelAcc, sleepLinVelAcc);
        const float frameNormalizedEnergy = 0.5f * (angular + linear);

        const float cf = hasStaticTouch ? float(clusterFactor) : 0.0f;
        const float freezeThresh = cf * freezeThresholdX_wakeCounterY_sleepThresholdZ_bodySimIndex.x;
        freezeCount = max(freezeCount - dt, 0.0f);

        bool settled = true;

        accelScale = min(1.0f, accelScale + dt);

        if (frameNormalizedEnergy >= freezeThresh)
        {
            settled = false;
            freezeCount = PXD_FREEZE_INTERVAL;
        }

        if (!hasStaticTouch)
        {
            accelScale = 1.0f;
            settled = false;
        }

        if (settled)
        {
            // Dampen bodies that are just about to go to sleep
            if (cf > 1.0f)
            {
                const float d = 1.0f - (PXD_SLEEP_DAMPING * dt);
                solverBodyLinVel = solverBodyLinVel * d;
                solverBodyAngVel = solverBodyAngVel * d;
                accelScale = accelScale * 0.75f + 0.25f * PXD_FREEZE_SCALE;
            }

            freeze = freezeCount == 0.0f &&
                frameNormalizedEnergy < (freezeThresholdX_wakeCounterY_sleepThresholdZ_bodySimIndex.x * PXD_FREEZE_TOLERANCE);
        }

        if (freeze)
        {
            const bool wasNotFrozen = (flags & PXG_BODY_FROZEN) == 0;
            flags |= PXG_BODY_FROZEN;
            if (wasNotFrozen)
                flags |= PXG_BODY_FREEZE_THIS_FRAME;
        }
        else
        {
            const bool wasFrozen = (flags & PXG_BODY_FROZEN) != 0;
            flags &= (PXG_BODY_DISABLE_GRAVITY_GPU | PXG_BODY_ENABLE_GYROSCOPIC | PXG_BODY_RETAIN_ACCELERATION);
            if (wasFrozen)
                flags |= PXG_BODY_UNFREEZE_THIS_FRAME;
        }

        // KS: New algorithm for sleeping when using stabilization:
        // Energy *this frame* must be higher than sleep threshold and
        // accumulated energy over previous frames must be higher than
        // clusterFactor*energyThreshold.
        if (wc < wakeCounterResetTime * 0.5f || wc < dt)
        {
            // Accumulate energy
            sleepLinVelAcc += sleepLinVelAccXYZ_freezeCountW.xyz;
            sleepAngVelAcc += sleepAngVelAccXYZ_accelScaleW.xyz;

            // If energy this frame is high
            const float sleepThreshold = freezeThresholdX_wakeCounterY_sleepThresholdZ_bodySimIndex.z;

            if (frameNormalizedEnergy >= sleepThreshold)
            {
                // Compute energy over sleep preparation time
                const float sleepAngular = dot(sleepAngVelAcc * sleepAngVelAcc, inertia) * invMass;
                const float sleepLinear = dot(sleepLinVelAcc, sleepLinVelAcc);

                const float normalizedEnergy = 0.5f * (sleepAngular + sleepLinear);
                const float sleepClusterFactor = float(clusterFactor) + 1.0f;

                // scale threshold by cluster factor (more contacts => higher sleep threshold)
                const float threshold = sleepClusterFactor * sleepThreshold;

                // If energy over sleep preparation time is high
                if (normalizedEnergy >= threshold)
                {
                    // Wake up
                    sleepAngVelAcc = float3(0.0f);
                    sleepLinVelAcc = float3(0.0f);

                    const float factor = sleepThreshold == 0.0f ? 2.0f : min(normalizedEnergy / threshold, 2.0f);
                    const float oldWc = wc;
                    wc = factor * 0.5f * wakeCounterResetTime + dt * (sleepClusterFactor - 1.0f);

                    if (oldWc == 0.0f)
                        flags |= PXG_BODY_ACTIVATE_THIS_FRAME;

                    alreadyUpdateWC = true;
                }
            }
        }
    }
    else
    {
        if (wc < wakeCounterResetTime * 0.5f || wc < dt)
        {
            // calculate normalized energy: kinetic energy divided by mass
            const float3 inertia = float3(
                inverseInertia.x > 0.0f ? 1.0f / inverseInertia.x : 1.0f,
                inverseInertia.y > 0.0f ? 1.0f / inverseInertia.y : 1.0f,
                inverseInertia.z > 0.0f ? 1.0f / inverseInertia.z : 1.0f);

            sleepLinVelAcc = linearMotionVel;
            sleepAngVelAcc = quatRotateInv(body2World.q, angularMotionVel);

            sleepLinVelAcc += sleepLinVelAccXYZ_freezeCountW.xyz;
            sleepAngVelAcc += sleepAngVelAccXYZ_accelScaleW.xyz;

            float invMass = invertedMass;
            if (invMass == 0.0f)
                invMass = 1.0f;

            const float angular = dot(sleepAngVelAcc * sleepAngVelAcc, inertia) * invMass;
            const float linear = dot(sleepLinVelAcc, sleepLinVelAcc);
            const float normalizedEnergy = 0.5f * (angular + linear);

            // scale threshold by cluster factor (more contacts => higher sleep threshold)
            const float clusterFactor = float(1u + numCountedInteractions);

            const float sleepThreshold = freezeThresholdX_wakeCounterY_sleepThresholdZ_bodySimIndex.z;

            const float threshold = clusterFactor * sleepThreshold;

            if (normalizedEnergy >= threshold)
            {
                sleepLinVelAcc = float3(0.0f);
                sleepAngVelAcc = float3(0.0f);
                const float factor = threshold == 0.0f ? 2.0f : min(normalizedEnergy / threshold, 2.0f);
                const float oldWc = wc;
                wc = factor * 0.5f * wakeCounterResetTime + dt * (clusterFactor - 1.0f);

                if (oldWc == 0.0f) // sleeping body activated by the system (not the user) AND processed by the solver
                    flags |= PXG_BODY_ACTIVATE_THIS_FRAME;

                alreadyUpdateWC = true;
            }
        }
    }

    if (!alreadyUpdateWC)
        wc = max(wc - dt, 0.0f);

    const bool wakeCounterZero = (wc == 0.0f);

    if (wakeCounterZero)
    {
        flags |= PXG_BODY_DEACTIVATE_THIS_FRAME;
        sleepLinVelAcc = float3(0.0f);
        sleepAngVelAcc = float3(0.0f);
    }

    bodySim.internalFlags = flags;
    bodySim.freezeThresholdX_wakeCounterY_sleepThresholdZ_bodySimIndex.y = wc;
    bodySim.sleepLinVelAccXYZ_freezeCountW = float4(sleepLinVelAcc, freezeCount);
    bodySim.sleepAngVelAccXYZ_accelScaleW = float4(sleepAngVelAcc, accelScale);

    if ((flags & PXG_BODY_RETAIN_ACCELERATION) == 0)
    {
        bodySim.externalLinearAcceleration = float4(0.0f);
        bodySim.externalAngularAcceleration = float4(0.0f);
    }

    sleepData.internalFlags = flags;
    sleepData.wakeCounter = wc;
}

// mirrors sleepCheck() in integration.cuh
bool sleepCheck(inout float4 solverBodyLinVel, inout float4 solverBodyAngVel, PxgAlignedTransform body2World,
                inout PxgBodySim bodySim, out PxgSolverBodySleepData sleepData, float4 inverseInertia,
                float3 linearMotionVel, float3 angularMotionVel, float invertedMass, float dt, float invDt,
                bool enableStabilization, bool hasStaticTouch, uint numCountedInteractions)
{
    bool freeze = false;
    updateWakeCounter(freeze, solverBodyLinVel, solverBodyAngVel, body2World, bodySim, sleepData, inverseInertia,
                      linearMotionVel, angularMotionVel, invertedMass, dt, invDt, enableStabilization,
                      hasStaticTouch, numCountedInteractions);
    return freeze;
}

// mirrors integrateCore() in integration.cuh (the PGS variant, i.e. without
// IS_TGS_SOLVER defined)
void integrateCore(float4 motionLinVelXYZW, float4 motionAngVelXYZW, float4 inverseInertia,
                   inout float4 solverBodyLinVel, inout float4 solverBodyAngVel,
                   inout PxgAlignedTransform body2World, PxgSolverBodyData solverBodyData,
                   inout PxgBodySim bodySim, out PxgSolverBodySleepData sleepData, PxgMat33 sqrtInvInertia,
                   float dt, float invDt, bool enableStabilization, bool hasStaticTouch,
                   uint numCountedInteractions)
{
    // Integrate linear part
    const float4 initialLinVelXYZ_invMassW = solverBodyData.initialLinVelXYZ_invMassW;
    const float4 initialAngVelXYZ_penBiasClamp = solverBodyData.initialAngVelXYZ_penBiasClamp;
    const float3 initialLinVel = initialLinVelXYZ_invMassW.xyz;
    const float3 initialAngVel = initialAngVelXYZ_penBiasClamp.xyz;

    const uint lockFlags = bodySim.lockFlags_disableGravity & 0xFFFFu;

    // update body lin and ang velocity
    float3 bodyLinearVelocity = solverBodyLinVel.xyz;
    float3 bodyAngVelocity = solverBodyAngVel.xyz;

    bodyLinearVelocity = initialLinVel + bodyLinearVelocity;
    bodyAngVelocity = initialAngVel + mat33Transform(sqrtInvInertia, bodyAngVelocity);

    solverBodyLinVel = float4(
        (lockFlags & PXG_LOCK_LINEAR_X) != 0 ? 0.0f : bodyLinearVelocity.x,
        (lockFlags & PXG_LOCK_LINEAR_Y) != 0 ? 0.0f : bodyLinearVelocity.y,
        (lockFlags & PXG_LOCK_LINEAR_Z) != 0 ? 0.0f : bodyLinearVelocity.z, 0.0f);
    solverBodyAngVel = float4(
        (lockFlags & PXG_LOCK_ANGULAR_X) != 0 ? 0.0f : bodyAngVelocity.x,
        (lockFlags & PXG_LOCK_ANGULAR_Y) != 0 ? 0.0f : bodyAngVelocity.y,
        (lockFlags & PXG_LOCK_ANGULAR_Z) != 0 ? 0.0f : bodyAngVelocity.z, 0.0f);

    // we need to perform sleep check here to decide whether we want to update
    // the body2World transform for the body
    const float3 motionLinVel = float3(
        (lockFlags & PXG_LOCK_LINEAR_X) != 0 ? 0.0f : motionLinVelXYZW.x,
        (lockFlags & PXG_LOCK_LINEAR_Y) != 0 ? 0.0f : motionLinVelXYZW.y,
        (lockFlags & PXG_LOCK_LINEAR_Z) != 0 ? 0.0f : motionLinVelXYZW.z);
    const float3 motionAngVel = float3(
        (lockFlags & PXG_LOCK_ANGULAR_X) != 0 ? 0.0f : motionAngVelXYZW.x,
        (lockFlags & PXG_LOCK_ANGULAR_Y) != 0 ? 0.0f : motionAngVelXYZW.y,
        (lockFlags & PXG_LOCK_ANGULAR_Z) != 0 ? 0.0f : motionAngVelXYZW.z);

    const float3 linearMotionVel = initialLinVel + motionLinVel;
    float3 angularMotionVel = initialAngVel + mat33Transform(sqrtInvInertia, motionAngVel);

    // Integrate the rotation using closed form quaternion integrator
    float w = dot(angularMotionVel, angularMotionVel);
    w = sqrt(w);
    const float maxW = 1e+7f; // Should be about sqrt(PX_MAX_REAL/2) or smaller
    if (w > maxW)
    {
        angularMotionVel = normalize(angularMotionVel) * maxW;
        w = maxW;
    }

    const bool freeze = sleepCheck(solverBodyLinVel, solverBodyAngVel, body2World, bodySim, sleepData,
                                   inverseInertia, linearMotionVel, angularMotionVel,
                                   initialLinVelXYZ_invMassW.w, dt, invDt, enableStabilization, hasStaticTouch,
                                   numCountedInteractions);

    if (!freeze)
    {
        const float3 delta = linearMotionVel * dt;
        body2World.p.x += delta.x;
        body2World.p.y += delta.y;
        body2World.p.z += delta.z;

        if (w != 0.0f)
        {
            const float v = dt * w * 0.5f;
            float s = sin(v);
            const float q = cos(v);
            s /= w;

            const float3 pqr = angularMotionVel * s;
            const float4 quatVel = float4(pqr, 0.0f);
            float4 result = quatMul(quatVel, body2World.q);

            result += body2World.q * q;

            body2World.q = quatNormalize(result);
        }
    }
}

// mirrors integrateCoreTGS() in integration.cuh (compiled with IS_TGS_SOLVER):
// velocities are absolute (no initial-velocity addition), and the pose is
// advanced by the solver-accumulated deltaBody2World instead of the
// closed-form quaternion integrator.
void integrateCoreTGS(float4 motionLinVelXYZW, float4 motionAngVelXYZW, float4 inverseInertia,
                      inout float4 solverBodyLinVel, inout float4 solverBodyAngVel,
                      inout PxgAlignedTransform body2World, PxgTransform28 deltaBody2World,
                      PxgSolverBodyData solverBodyData, inout PxgBodySim bodySim,
                      out PxgSolverBodySleepData sleepData, PxgMat33 sqrtInvInertia,
                      float dt, float invDt, bool enableStabilization, bool hasStaticTouch,
                      uint numCountedInteractions)
{
    const uint lockFlags = bodySim.lockFlags_disableGravity & 0xFFFFu;

    // KS - TODO - optimize this away
    const float4 initialLinVelXYZ_invMassW = solverBodyData.initialLinVelXYZ_invMassW;

    // update body lin and ang velocity
    const float3 bodyLinearVelocity = solverBodyLinVel.xyz;
    const float3 bodyAngVelocity = mat33Transform(sqrtInvInertia, solverBodyAngVel.xyz);

    solverBodyLinVel = float4(
        (lockFlags & PXG_LOCK_LINEAR_X) != 0 ? 0.0f : bodyLinearVelocity.x,
        (lockFlags & PXG_LOCK_LINEAR_Y) != 0 ? 0.0f : bodyLinearVelocity.y,
        (lockFlags & PXG_LOCK_LINEAR_Z) != 0 ? 0.0f : bodyLinearVelocity.z, 0.0f);
    solverBodyAngVel = float4(
        (lockFlags & PXG_LOCK_ANGULAR_X) != 0 ? 0.0f : bodyAngVelocity.x,
        (lockFlags & PXG_LOCK_ANGULAR_Y) != 0 ? 0.0f : bodyAngVelocity.y,
        (lockFlags & PXG_LOCK_ANGULAR_Z) != 0 ? 0.0f : bodyAngVelocity.z, 0.0f);

    // we need to perform sleep check here to decide whether we want to update
    // the body2World transform for the body
    const float3 motionLinVel = float3(
        (lockFlags & PXG_LOCK_LINEAR_X) != 0 ? 0.0f : motionLinVelXYZW.x,
        (lockFlags & PXG_LOCK_LINEAR_Y) != 0 ? 0.0f : motionLinVelXYZW.y,
        (lockFlags & PXG_LOCK_LINEAR_Z) != 0 ? 0.0f : motionLinVelXYZW.z);
    const float3 motionAngVel = float3(
        (lockFlags & PXG_LOCK_ANGULAR_X) != 0 ? 0.0f : motionAngVelXYZW.x,
        (lockFlags & PXG_LOCK_ANGULAR_Y) != 0 ? 0.0f : motionAngVelXYZW.y,
        (lockFlags & PXG_LOCK_ANGULAR_Z) != 0 ? 0.0f : motionAngVelXYZW.z);

    const float3 linearMotionVel = motionLinVel;
    float3 angularMotionVel = mat33Transform(sqrtInvInertia, motionAngVel);

    // Integrate the rotation using closed form quaternion integrator
    float w = dot(angularMotionVel, angularMotionVel);
    w = sqrt(w);
    const float maxW = 1e+7f;
    if (w > maxW)
    {
        angularMotionVel = normalize(angularMotionVel) * maxW;
        w = maxW;
    }

    const bool freeze = sleepCheck(solverBodyLinVel, solverBodyAngVel, body2World, bodySim, sleepData,
                                   inverseInertia, linearMotionVel, angularMotionVel,
                                   initialLinVelXYZ_invMassW.w, dt, invDt, enableStabilization,
                                   hasStaticTouch, numCountedInteractions);

    if (!freeze)
    {
        body2World.p.x += deltaBody2World.px;
        body2World.p.y += deltaBody2World.py;
        body2World.p.z += deltaBody2World.pz;

        const float4 q = quatNormalize(quatMul(deltaBody2World.q(), body2World.q));
        body2World.q = q;
    }
}

// mirrors integrateCoreParallelLaunch() in integration.cu — P5 Step 5 BDA
// engine-launch ABI, the same 48-byte pack shape as the TGS twin below:
// (PxU32 offset, const PxgSolverCoreDesc*, const
// PxgSolverSharedDesc<IterativeSolveData>*, const PxU32* islandIds, const
// PxU32* islandStaticTouchCounts, const PxU32* numCountedInteractions).
// All counts/offsets/pool pointers are read from the descs on-device through
// the byte-exact pxgSlangSolverDesc mirrors.  PGS differences from the TGS
// twin, mirrored from the .cu: the pose is read from bodySim.body2World (not
// outBody2World[a]); the velocity pool holds TWO full float4 stripes (linVel @
// index, angVel @ index + numSolverBodies) instead of the packed TGS layout;
// and integrateCore() adds the initial velocities (PGS solves deltas).
struct IntegrationPgsAbiParams
{
    uint offset;                      // @0  (4 bytes pad follow)
    uint64_t solverCoreDesc;          // @8  const PxgSolverCoreDesc*
    uint64_t sharedDesc;              // @16 const PxgSolverSharedDesc<IterativeSolveData>*
    uint64_t islandIds;               // @24 const PxU32*
    uint64_t islandStaticTouchCounts; // @32 const PxU32*
    uint64_t numCountedInteractions;  // @40 const PxU32*
}                                     // = 48
[[vk::push_constant]]
ConstantBuffer<IntegrationPgsAbiParams> gPgsAbi;

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void integrateCoreParallelLaunch(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    PxgSolverCoreDescGpu desc = (Ptr<PxgSolverCoreDescGpu>(gPgsAbi.solverCoreDesc))[0];
    PxgSolverSharedDescPGSGpu shared = (Ptr<PxgSolverSharedDescPGSGpu>(gPgsAbi.sharedDesc))[0];

    Ptr<float4> motionVelocityArray = Ptr<float4>(desc.motionVelocityArray);
    const uint numSolverBodies = desc.numSolverBodies;
    Ptr<float4> solverBodyVelocity = Ptr<float4>(shared.iter_solverBodyVelPool);
    Ptr<PxgSolverTxIData> txIData = Ptr<PxgSolverTxIData>(desc.solverBodyTxIDataPool);
    Ptr<float4> outSolverVelocity = Ptr<float4>(desc.outSolverVelocity);
    Ptr<PxgAlignedTransform> outBody2World = Ptr<PxgAlignedTransform>(desc.outBody2World);
    Ptr<PxgSolverBodyData> solverBodyDataPool = Ptr<PxgSolverBodyData>(desc.solverBodyDataPool);
    Ptr<PxgBodySim> bodySimPool = Ptr<PxgBodySim>(desc.mBodySimBufferDeviceData);
    Ptr<PxgSolverBodySleepData> sleepDataPool = Ptr<PxgSolverBodySleepData>(desc.solverBodySleepDataPool);
    Ptr<uint> islandIds = Ptr<uint>(gPgsAbi.islandIds);
    Ptr<uint> islandStaticTouchCounts = Ptr<uint>(gPgsAbi.islandStaticTouchCounts);
    Ptr<uint> numCountedInteractions = Ptr<uint>(gPgsAbi.numCountedInteractions);

    const uint idx = dispatchThreadId.x;
    const uint a = idx + gPgsAbi.offset;

    if (a >= numSolverBodies)
        return;

    const PxgSolverBodyData data = solverBodyDataPool[a];

    const uint nodeIndex = nodeIndexIndex(data.islandNodeIndex);

    PxgBodySim bodySim = bodySimPool[nodeIndex];

    const PxgMat33 sqrtInvInertia = mat33Load(txIData[a].sqrtInvInertia);

    // PT: TODO: TGS version uses outBody2World[a] here, why? (mirrored: PGS
    // reads the pose from the body sim)
    PxgAlignedTransform body2World = bodySim.body2World;
    const float4 inverseInertia = bodySim.inverseInertiaXYZ_contactReportThresholdW;

    const uint index = desc.accumulatedBodyDeltaVOffset + a;

    float4 linVel = solverBodyVelocity[index];
    float4 angVel = solverBodyVelocity[index + numSolverBodies];

    const uint staticTouchCount = islandStaticTouchCounts[islandIds[nodeIndex]];

    PxgSolverBodySleepData sleepData;

    integrateCore(motionVelocityArray[a], motionVelocityArray[a + numSolverBodies], inverseInertia,
                  linVel, angVel, body2World, data, bodySim, sleepData, sqrtInvInertia,
                  shared.dt, shared.invDtF32, (desc.enableStabilizationWord & 0xffu) != 0,
                  staticTouchCount != 0, numCountedInteractions[nodeIndex]);

    sleepDataPool[a] = sleepData;

    outSolverVelocity[a] = linVel;
    outSolverVelocity[a + numSolverBodies] = angVel;
    outBody2World[a] = body2World;

    // PT: for acceleration getters (eENABLE_BODY_ACCELERATIONS) — the CUDA
    // null-pointer branch on mBodySimPrevVelocitiesBufferDeviceData
    if (desc.mBodySimPrevVelocitiesBufferDeviceData != 0)
    {
        // PxgBodySimVelocities: 2 float4 per node
        Ptr<float4> prevVelocities = Ptr<float4>(desc.mBodySimPrevVelocitiesBufferDeviceData);
        prevVelocities[nodeIndex * 2 + 0] = bodySimPool[nodeIndex].linearVelocityXYZ_inverseMassW;
        prevVelocities[nodeIndex * 2 + 1] = bodySimPool[nodeIndex].angularVelocityXYZ_maxPenBiasW;
    }

    // write back linear velocity, angular velocity to pxgbodysim
    bodySim.linearVelocityXYZ_inverseMassW.x = linVel.x;
    bodySim.linearVelocityXYZ_inverseMassW.y = linVel.y;
    bodySim.linearVelocityXYZ_inverseMassW.z = linVel.z;
    bodySim.angularVelocityXYZ_maxPenBiasW.x = angVel.x;
    bodySim.angularVelocityXYZ_maxPenBiasW.y = angVel.y;
    bodySim.angularVelocityXYZ_maxPenBiasW.z = angVel.z;
    bodySim.body2World = body2World;

    bodySimPool[nodeIndex] = bodySim;
}

// mirrors integrateCoreParallelLaunchTGS() in integrationTGS.cu — P5 Step 4
// BDA engine-launch ABI: the push-constant struct mirrors the CUDA parameter
// list EXACTLY (PxU32 offset, then five device pointers at natural
// alignment).  All counts/offsets/pool pointers are read from the descs
// on-device through the byte-exact pxgSlangSolverDesc mirrors; the CUDA
// `if (prevVelocities)` null-pointer branch is a device-address check here.
struct IntegrationTGSAbiParams
{
    uint offset;                      // @0  (4 bytes pad follow)
    uint64_t solverCoreDesc;          // @8  const PxgSolverCoreDesc*
    uint64_t sharedDesc;              // @16 const PxgSolverSharedDesc<IterativeSolveDataTGS>*
    uint64_t islandIds;               // @24 const PxU32*
    uint64_t islandStaticTouchCounts; // @32 const PxU32*
    uint64_t numCountedInteractions;  // @40 const PxU32*
}                                     // = 48
[[vk::push_constant]]
ConstantBuffer<IntegrationTGSAbiParams> gTgsAbi;

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void integrateCoreParallelLaunchTGS(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    PxgSolverCoreDescGpu desc = (Ptr<PxgSolverCoreDescGpu>(gTgsAbi.solverCoreDesc))[0];
    PxgSolverSharedDescTGSGpu shared = (Ptr<PxgSolverSharedDescTGSGpu>(gTgsAbi.sharedDesc))[0];

    Ptr<float4> motionVelocityArray = Ptr<float4>(desc.motionVelocityArray);
    const uint numSolverBodies = desc.numSolverBodies;
    Ptr<float4> solverBodyVelocity = Ptr<float4>(shared.iter_solverBodyVelPool);
    Ptr<PxgSolverTxIData> txIDatas = Ptr<PxgSolverTxIData>(desc.solverBodyTxIDataPool);
    Ptr<float4> outSolverVelocity = Ptr<float4>(desc.outSolverVelocity);
    Ptr<PxgAlignedTransform> outBody2World = Ptr<PxgAlignedTransform>(desc.outBody2World);
    Ptr<PxgSolverBodyData> solverBodyDataPool = Ptr<PxgSolverBodyData>(desc.solverBodyDataPool);
    Ptr<PxgBodySim> bodySimPool = Ptr<PxgBodySim>(desc.mBodySimBufferDeviceData);
    Ptr<PxgSolverBodySleepData> sleepDataPool = Ptr<PxgSolverBodySleepData>(desc.solverBodySleepDataPool);
    Ptr<uint> islandIds = Ptr<uint>(gTgsAbi.islandIds);
    Ptr<uint> islandStaticTouchCounts = Ptr<uint>(gTgsAbi.islandStaticTouchCounts);
    Ptr<uint> numCountedInteractions = Ptr<uint>(gTgsAbi.numCountedInteractions);

    const uint idx = dispatchThreadId.x;
    const uint a = idx + gTgsAbi.offset;

    if (a >= numSolverBodies)
        return;

    const PxgSolverBodyData data = solverBodyDataPool[a];

    const uint nodeIndex = nodeIndexIndex(data.islandNodeIndex);

    PxgBodySim bodySim = bodySimPool[nodeIndex];

    const PxgMat33 sqrtInvInertia = mat33Load(txIDatas[a].sqrtInvInertia);
    const PxgTransform28 deltaTransform = txIDatas[a].deltaBody2World;

    // PT: TODO: PGS version uses bodySim.body2World here, why?
    PxgAlignedTransform body2World = outBody2World[a];

    const float4 inverseInertia = bodySim.inverseInertiaXYZ_contactReportThresholdW;

    const float4 motionLinVel = motionVelocityArray[a];
    const float4 motionAngVel = motionVelocityArray[a + numSolverBodies];

    const uint readIndex = desc.accumulatedBodyDeltaVOffset + a;

    // TGS velocity pool layout: v0 = (lin.xyz, ang.x), v1 = (ang.yz, ...)
    const float4 v0 = solverBodyVelocity[readIndex];
    const float4 v1 = solverBodyVelocity[readIndex + numSolverBodies];

    float4 linVel = float4(v0.x, v0.y, v0.z, 0.0f);
    float4 angVel = float4(v0.w, v1.x, v1.y, 0.0f);

    const uint staticTouchCount = islandStaticTouchCounts[islandIds[nodeIndex]];

    PxgSolverBodySleepData sleepData;

    integrateCoreTGS(motionLinVel, motionAngVel, inverseInertia, linVel, angVel, body2World,
                     deltaTransform, data, bodySim, sleepData, sqrtInvInertia,
                     shared.dt, shared.invDtF32, (desc.enableStabilizationWord & 0xffu) != 0,
                     staticTouchCount != 0, numCountedInteractions[nodeIndex]);

    sleepDataPool[a] = sleepData;

    outSolverVelocity[a] = linVel;
    outSolverVelocity[a + numSolverBodies] = angVel;
    outBody2World[a] = body2World;

    // PT: for acceleration getters (eENABLE_BODY_ACCELERATIONS) — the CUDA
    // null-pointer branch on mBodySimPrevVelocitiesBufferDeviceData
    if (desc.mBodySimPrevVelocitiesBufferDeviceData != 0)
    {
        // PxgBodySimVelocities: 2 float4 per node
        Ptr<float4> prevVelocities = Ptr<float4>(desc.mBodySimPrevVelocitiesBufferDeviceData);
        prevVelocities[nodeIndex * 2 + 0] = bodySimPool[nodeIndex].linearVelocityXYZ_inverseMassW;
        prevVelocities[nodeIndex * 2 + 1] = bodySimPool[nodeIndex].angularVelocityXYZ_maxPenBiasW;
    }

    // write back linear velocity, angular velocity to pxgbodysim
    bodySim.linearVelocityXYZ_inverseMassW.x = linVel.x;
    bodySim.linearVelocityXYZ_inverseMassW.y = linVel.y;
    bodySim.linearVelocityXYZ_inverseMassW.z = linVel.z;
    bodySim.angularVelocityXYZ_maxPenBiasW.x = angVel.x;
    bodySim.angularVelocityXYZ_maxPenBiasW.y = angVel.y;
    bodySim.angularVelocityXYZ_maxPenBiasW.z = angVel.z;
    bodySim.body2World = body2World;

    bodySimPool[nodeIndex] = bodySim;
}
