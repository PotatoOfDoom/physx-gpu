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

// Verification of integration.slang (integrateCoreParallelLaunch) against a
// C++ transcription of the CUDA kernel (integration.cu / integration.cuh):
// velocity finalization, the sleep/freeze/wake-counter state machine and the
// closed-form quaternion pose integration.

#include "pxgTestFramework.h"
#include "pxgGpuStructs.h"
#include "pxgRefMath.h"

#include <numeric>
#include <random>

namespace
{

using namespace pxgtest;

struct IntegrationParams
{
    uint32_t offset;
    uint32_t numSolverBodies;
    uint32_t accumulatedBodyDeltaVOffset;
    float dt;
    float invDtF32;
    uint32_t enableStabilization;
    uint32_t hasPrevVelocities;
};

// ---------------------------------------------------------------------------
// CPU reference: direct transcription of integration.cuh (PGS variant)
// ---------------------------------------------------------------------------

void refUpdateWakeCounter(bool& freeze, Float4& solverBodyLinVel, Float4& solverBodyAngVel,
                          const AlignedTransform& body2World, BodySim& bodySim, SolverBodySleepData& sleepData,
                          const Float4& inverseInertia, const Vec3& linearMotionVel, const Vec3& angularMotionVel,
                          float invertedMass, float dt, bool enableStabilization, bool hasStaticTouch,
                          uint32_t numCountedInteractions)
{
    const float wakeCounterResetTime = 20.0f * 0.02f;

    Float4 freezeThreshold_wc_sleepThreshold = bodySim.freezeThresholdX_wakeCounterY_sleepThresholdZ_bodySimIndex;
    const Float4 sleepLinVelAcc_freezeCount = bodySim.sleepLinVelAccXYZ_freezeCountW;
    const Float4 sleepAngVelAcc_accelScale = bodySim.sleepAngVelAccXYZ_accelScaleW;

    float wc = freezeThreshold_wc_sleepThreshold.y;

    uint32_t flags = bodySim.internalFlags &
        (eDISABLE_GRAVITY_GPU | eFROZEN | eENABLE_GYROSCOPIC | eRETAIN_ACCELERATION);
    float freezeCount = sleepLinVelAcc_freezeCount.w;
    float accelScale = sleepAngVelAcc_accelScale.w;

    bool alreadyUpdateWC = false;
    Vec3 sleepLinVelAcc(0.f), sleepAngVelAcc(0.f);

    if (enableStabilization)
    {
        const uint32_t maxCountedInteractions = 10u;

        const Vec3 inertia(inverseInertia.x > 0.f ? 1.0f / inverseInertia.x : 1.f,
                           inverseInertia.y > 0.f ? 1.0f / inverseInertia.y : 1.f,
                           inverseInertia.z > 0.f ? 1.0f / inverseInertia.z : 1.f);
        sleepLinVelAcc = linearMotionVel;
        sleepAngVelAcc = angularMotionVel;

        const uint32_t clusterFactor =
            numCountedInteractions < maxCountedInteractions ? numCountedInteractions : maxCountedInteractions;

        float invMass = invertedMass;
        if (invMass == 0.f)
            invMass = 1.f;

        const float angular = sleepAngVelAcc.multiply(sleepAngVelAcc).dot(inertia) * invMass;
        const float linear = sleepLinVelAcc.magnitudeSquared();
        const float frameNormalizedEnergy = 0.5f * (angular + linear);

        const float cf = hasStaticTouch ? float(clusterFactor) : 0.f;
        const float freezeThresh = cf * freezeThreshold_wc_sleepThreshold.x;
        freezeCount = std::fmax(freezeCount - dt, 0.0f);

        bool settled = true;

        accelScale = std::fmin(1.f, accelScale + dt);

        if (frameNormalizedEnergy >= freezeThresh)
        {
            settled = false;
            freezeCount = PXD_FREEZE_INTERVAL;
        }

        if (!hasStaticTouch)
        {
            accelScale = 1.f;
            settled = false;
        }

        if (settled)
        {
            if (cf > 1)
            {
                const float d = 1.0f - (PXD_SLEEP_DAMPING * dt);
                solverBodyLinVel = Float4{ solverBodyLinVel.x * d, solverBodyLinVel.y * d,
                                           solverBodyLinVel.z * d, solverBodyLinVel.w * d };
                solverBodyAngVel = Float4{ solverBodyAngVel.x * d, solverBodyAngVel.y * d,
                                           solverBodyAngVel.z * d, solverBodyAngVel.w * d };
                accelScale = accelScale * 0.75f + 0.25f * PXD_FREEZE_SCALE;
            }

            freeze = freezeCount == 0.f &&
                     frameNormalizedEnergy < (freezeThreshold_wc_sleepThreshold.x * PXD_FREEZE_TOLERANCE);
        }

        if (freeze)
        {
            const bool wasNotFrozen = (flags & eFROZEN) == 0;
            flags |= eFROZEN;
            if (wasNotFrozen)
                flags |= eFREEZE_THIS_FRAME;
        }
        else
        {
            const bool wasFrozen = (flags & eFROZEN) != 0;
            flags &= (eDISABLE_GRAVITY_GPU | eENABLE_GYROSCOPIC | eRETAIN_ACCELERATION);
            if (wasFrozen)
                flags |= eUNFREEZE_THIS_FRAME;
        }

        if (wc < wakeCounterResetTime * 0.5f || wc < dt)
        {
            sleepLinVelAcc.x += sleepLinVelAcc_freezeCount.x;
            sleepLinVelAcc.y += sleepLinVelAcc_freezeCount.y;
            sleepLinVelAcc.z += sleepLinVelAcc_freezeCount.z;

            sleepAngVelAcc.x += sleepAngVelAcc_accelScale.x;
            sleepAngVelAcc.y += sleepAngVelAcc_accelScale.y;
            sleepAngVelAcc.z += sleepAngVelAcc_accelScale.z;

            const float sleepThreshold = freezeThreshold_wc_sleepThreshold.z;

            if (frameNormalizedEnergy >= sleepThreshold)
            {
                const float sleepAngular = sleepAngVelAcc.multiply(sleepAngVelAcc).dot(inertia) * invMass;
                const float sleepLinear = sleepLinVelAcc.magnitudeSquared();

                const float normalizedEnergy = 0.5f * (sleepAngular + sleepLinear);
                const float sleepClusterFactor = clusterFactor + 1.f;

                const float threshold = sleepClusterFactor * sleepThreshold;

                if (normalizedEnergy >= threshold)
                {
                    sleepAngVelAcc = Vec3(0.f);
                    sleepLinVelAcc = Vec3(0.f);

                    const float factor =
                        sleepThreshold == 0.f ? 2.0f : std::fmin(normalizedEnergy / threshold, 2.0f);
                    const float oldWc = wc;
                    wc = factor * 0.5f * wakeCounterResetTime + dt * (sleepClusterFactor - 1.0f);

                    if (oldWc == 0.0f)
                        flags |= eACTIVATE_THIS_FRAME;

                    alreadyUpdateWC = true;
                }
            }
        }
    }
    else
    {
        if (wc < wakeCounterResetTime * 0.5f || wc < dt)
        {
            const Vec3 inertia(inverseInertia.x > 0.f ? 1.0f / inverseInertia.x : 1.f,
                               inverseInertia.y > 0.f ? 1.0f / inverseInertia.y : 1.f,
                               inverseInertia.z > 0.f ? 1.0f / inverseInertia.z : 1.f);

            sleepLinVelAcc = linearMotionVel;
            sleepAngVelAcc = QuatV(body2World.q).rotateInv(angularMotionVel);

            sleepLinVelAcc.x += sleepLinVelAcc_freezeCount.x;
            sleepLinVelAcc.y += sleepLinVelAcc_freezeCount.y;
            sleepLinVelAcc.z += sleepLinVelAcc_freezeCount.z;

            sleepAngVelAcc.x += sleepAngVelAcc_accelScale.x;
            sleepAngVelAcc.y += sleepAngVelAcc_accelScale.y;
            sleepAngVelAcc.z += sleepAngVelAcc_accelScale.z;

            float invMass = invertedMass;
            if (invMass == 0.f)
                invMass = 1.f;

            const float angular = sleepAngVelAcc.multiply(sleepAngVelAcc).dot(inertia) * invMass;
            const float linear = sleepLinVelAcc.magnitudeSquared();
            const float normalizedEnergy = 0.5f * (angular + linear);

            const float clusterFactor = float(1 + numCountedInteractions);

            const float sleepThreshold = freezeThreshold_wc_sleepThreshold.z;

            const float threshold = clusterFactor * sleepThreshold;

            if (normalizedEnergy >= threshold)
            {
                sleepLinVelAcc = Vec3(0.f);
                sleepAngVelAcc = Vec3(0.f);
                const float factor = threshold == 0.f ? 2.0f : std::fmin(normalizedEnergy / threshold, 2.0f);
                const float oldWc = wc;
                wc = factor * 0.5f * wakeCounterResetTime + dt * (clusterFactor - 1.0f);

                if (oldWc == 0.0f)
                    flags |= eACTIVATE_THIS_FRAME;

                alreadyUpdateWC = true;
            }
        }
    }

    if (!alreadyUpdateWC)
        wc = std::fmax(wc - dt, 0.0f);

    const bool wakeCounterZero = (wc == 0.0f);

    if (wakeCounterZero)
    {
        flags |= eDEACTIVATE_THIS_FRAME;
        sleepLinVelAcc = Vec3(0.f);
        sleepAngVelAcc = Vec3(0.f);
    }

    bodySim.internalFlags = flags;
    bodySim.freezeThresholdX_wakeCounterY_sleepThresholdZ_bodySimIndex.y = wc;
    bodySim.sleepLinVelAccXYZ_freezeCountW = Float4{ sleepLinVelAcc.x, sleepLinVelAcc.y, sleepLinVelAcc.z, freezeCount };
    bodySim.sleepAngVelAccXYZ_accelScaleW = Float4{ sleepAngVelAcc.x, sleepAngVelAcc.y, sleepAngVelAcc.z, accelScale };

    if (!(flags & eRETAIN_ACCELERATION))
    {
        bodySim.externalLinearAcceleration = Float4{ 0, 0, 0, 0 };
        bodySim.externalAngularAcceleration = Float4{ 0, 0, 0, 0 };
    }

    sleepData.internalFlags = flags;
    sleepData.wakeCounter = wc;
}

void refIntegrateCore(const Float4& motionLinVelXYZW, const Float4& motionAngVelXYZW,
                      const Float4& inverseInertia, Float4& solverBodyLinVel, Float4& solverBodyAngVel,
                      AlignedTransform& body2World, const SolverBodyData& solverBodyData, BodySim& bodySim,
                      SolverBodySleepData& sleepData, const Mat33& sqrtInvInertia, float dt,
                      bool enableStabilization, bool hasStaticTouch, uint32_t numCountedInteractions)
{
    const Float4 initialLinVelXYZ_invMassW = solverBodyData.initialLinVelXYZ_invMassW;
    const Float4 initialAngVelXYZ_penBiasClamp = solverBodyData.initialAngVelXYZ_penBiasClamp;
    const Vec3 initialLinVel(initialLinVelXYZ_invMassW.x, initialLinVelXYZ_invMassW.y, initialLinVelXYZ_invMassW.z);
    const Vec3 initialAngVel(initialAngVelXYZ_penBiasClamp.x, initialAngVelXYZ_penBiasClamp.y, initialAngVelXYZ_penBiasClamp.z);

    const uint32_t lockFlags = bodySim.lockFlags;

    Vec3 bodyLinearVelocity(solverBodyLinVel.x, solverBodyLinVel.y, solverBodyLinVel.z);
    Vec3 bodyAngVelocity(solverBodyAngVel.x, solverBodyAngVel.y, solverBodyAngVel.z);

    bodyLinearVelocity = initialLinVel + bodyLinearVelocity;
    bodyAngVelocity = initialAngVel + sqrtInvInertia.transform(bodyAngVelocity);

    solverBodyLinVel = Float4{ (lockFlags & eLOCK_LINEAR_X) ? 0.f : bodyLinearVelocity.x,
                               (lockFlags & eLOCK_LINEAR_Y) ? 0.f : bodyLinearVelocity.y,
                               (lockFlags & eLOCK_LINEAR_Z) ? 0.f : bodyLinearVelocity.z, 0.f };
    solverBodyAngVel = Float4{ (lockFlags & eLOCK_ANGULAR_X) ? 0.f : bodyAngVelocity.x,
                               (lockFlags & eLOCK_ANGULAR_Y) ? 0.f : bodyAngVelocity.y,
                               (lockFlags & eLOCK_ANGULAR_Z) ? 0.f : bodyAngVelocity.z, 0.f };

    const Vec3 motionLinVel((lockFlags & eLOCK_LINEAR_X) ? 0.f : motionLinVelXYZW.x,
                            (lockFlags & eLOCK_LINEAR_Y) ? 0.f : motionLinVelXYZW.y,
                            (lockFlags & eLOCK_LINEAR_Z) ? 0.f : motionLinVelXYZW.z);
    const Vec3 motionAngVel((lockFlags & eLOCK_ANGULAR_X) ? 0.f : motionAngVelXYZW.x,
                            (lockFlags & eLOCK_ANGULAR_Y) ? 0.f : motionAngVelXYZW.y,
                            (lockFlags & eLOCK_ANGULAR_Z) ? 0.f : motionAngVelXYZW.z);

    const Vec3 linearMotionVel = initialLinVel + motionLinVel;
    Vec3 angularMotionVel = initialAngVel + sqrtInvInertia.transform(motionAngVel);

    float w = angularMotionVel.magnitudeSquared();
    w = std::sqrt(w);
    const float maxW = 1e+7f;
    if (w > maxW)
    {
        angularMotionVel = angularMotionVel.getNormalized() * maxW;
        w = maxW;
    }

    bool freeze = false;
    refUpdateWakeCounter(freeze, solverBodyLinVel, solverBodyAngVel, body2World, bodySim, sleepData,
                         inverseInertia, linearMotionVel, angularMotionVel, initialLinVelXYZ_invMassW.w,
                         dt, enableStabilization, hasStaticTouch, numCountedInteractions);

    if (!freeze)
    {
        const Vec3 delta = linearMotionVel * dt;
        body2World.p.x += delta.x;
        body2World.p.y += delta.y;
        body2World.p.z += delta.z;

        if (w != 0.0f)
        {
            const float v = dt * w * 0.5f;
            float s = std::sin(v);
            const float q = std::cos(v);
            s /= w;

            const Vec3 pqr = angularMotionVel * s;
            const QuatV quatVel(pqr.x, pqr.y, pqr.z, 0);
            QuatV result = quatVel * QuatV(body2World.q);

            result = result + QuatV(body2World.q) * q;

            body2World.q = result.getNormalized().toFloat4();
        }
    }
}

// ---------------------------------------------------------------------------
// test
// ---------------------------------------------------------------------------

QuatV randomQuat(std::mt19937& rng)
{
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    QuatV q(dist(rng), dist(rng), dist(rng), dist(rng));
    return q.getNormalized();
}

void checkIntegration(uint32_t nbBodies, uint32_t seed, bool enableStabilization, bool sleepy)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> unit(0.f, 1.f);
    std::uniform_real_distribution<float> sym(-5.f, 5.f);

    const float dt = 1.f / 60.f;
    const float invDt = 60.f;
    const uint32_t maxNodes = nbBodies + 16;
    const uint32_t numIslands = 8;
    const uint32_t deltaVOffset = 64; // exercise a non-zero accumulatedBodyDeltaVOffset

    // velocity scale: "sleepy" populations have tiny velocities so the
    // sleep/freeze branches fire; active populations integrate normally.
    const float velScale = sleepy ? 0.001f : 1.f;

    std::vector<uint32_t> nodePool(maxNodes);
    std::iota(nodePool.begin(), nodePool.end(), 0u);
    std::shuffle(nodePool.begin(), nodePool.end(), rng);

    std::vector<BodySim> bodySims(maxNodes);
    std::memset(bodySims.data(), 0, bodySims.size() * sizeof(BodySim));
    std::vector<SolverBodyData> solverData(nbBodies);
    std::memset(solverData.data(), 0, solverData.size() * sizeof(SolverBodyData));
    std::vector<SolverTxIData> txIData(nbBodies);
    std::memset(txIData.data(), 0, txIData.size() * sizeof(SolverTxIData));
    std::vector<Float4> motionVel(2 * nbBodies);
    std::vector<Float4> velPool(deltaVOffset + 2 * nbBodies);
    std::vector<uint32_t> islandIds(maxNodes);
    std::vector<uint32_t> staticTouchCounts(numIslands);
    std::vector<uint32_t> numCountedInteractions(maxNodes);

    for (uint32_t n = 0; n < maxNodes; ++n)
    {
        islandIds[n] = rng() % numIslands;
        numCountedInteractions[n] = rng() % 12;
    }
    for (uint32_t i = 0; i < numIslands; ++i)
        staticTouchCounts[i] = (unit(rng) < 0.5f) ? 0u : (rng() % 5 + 1);

    for (uint32_t a = 0; a < nbBodies; ++a)
    {
        const uint32_t nodeIndex = nodePool[a];
        BodySim& b = bodySims[nodeIndex];

        const QuatV q = randomQuat(rng);
        b.body2World.q = q.toFloat4();
        b.body2World.p = Float4{ sym(rng), sym(rng), sym(rng), 0.f };
        b.inverseInertiaXYZ_contactReportThresholdW =
            Float4{ unit(rng) < 0.1f ? 0.f : unit(rng) * 4.f, unit(rng) < 0.1f ? 0.f : unit(rng) * 4.f,
                    unit(rng) < 0.1f ? 0.f : unit(rng) * 4.f, unit(rng) };
        b.linearVelocityXYZ_inverseMassW = Float4{ sym(rng), sym(rng), sym(rng), unit(rng) };
        b.angularVelocityXYZ_maxPenBiasW = Float4{ sym(rng), sym(rng), sym(rng), -unit(rng) };
        // wake counters: mix of zero, tiny and large; thresholds mid-range
        const float r = unit(rng);
        const float wakeCounter = r < 0.25f ? 0.f : (r < 0.6f ? unit(rng) * 0.05f : unit(rng) * 4.f);
        b.freezeThresholdX_wakeCounterY_sleepThresholdZ_bodySimIndex =
            Float4{ unit(rng) * 0.05f, wakeCounter, unit(rng) * 0.05f, 0.f };
        b.sleepLinVelAccXYZ_freezeCountW = Float4{ sym(rng) * 0.01f, sym(rng) * 0.01f, sym(rng) * 0.01f,
                                                   unit(rng) < 0.3f ? 0.f : unit(rng) };
        b.sleepAngVelAccXYZ_accelScaleW = Float4{ sym(rng) * 0.01f, sym(rng) * 0.01f, sym(rng) * 0.01f, unit(rng) };

        uint32_t flags = 0;
        if (unit(rng) < 0.3f)
            flags |= eFROZEN;
        if (unit(rng) < 0.2f)
            flags |= eRETAIN_ACCELERATION;
        if (unit(rng) < 0.2f)
            flags |= eDISABLE_GRAVITY_GPU;
        b.internalFlags = flags;
        b.lockFlags = uint16_t(rng() & 0x3Fu & (unit(rng) < 0.25f ? 0x3Fu : 0u));
        b.externalLinearAcceleration = Float4{ sym(rng), sym(rng), sym(rng), 0 };
        b.externalAngularAcceleration = Float4{ sym(rng), sym(rng), sym(rng), 0 };

        SolverBodyData& d = solverData[a];
        d.initialLinVelXYZ_invMassW = Float4{ sym(rng) * velScale, sym(rng) * velScale, sym(rng) * velScale,
                                              b.linearVelocityXYZ_inverseMassW.w };
        d.initialAngVelXYZ_penBiasClamp = Float4{ sym(rng) * velScale, sym(rng) * velScale,
                                                  sym(rng) * velScale, -unit(rng) };
        d.body2World = b.body2World;
        d.islandNodeIndex = makeNodeIndex(nodeIndex);

        // world-space sqrt inverse inertia for this pose
        const Vec3 invI(b.inverseInertiaXYZ_contactReportThresholdW.x,
                        b.inverseInertiaXYZ_contactReportThresholdW.y,
                        b.inverseInertiaXYZ_contactReportThresholdW.z);
        const Vec3 safeSqrt(invI.x == 0.f ? 0.f : std::sqrt(invI.x), invI.y == 0.f ? 0.f : std::sqrt(invI.y),
                            invI.z == 0.f ? 0.f : std::sqrt(invI.z));
        const Mat33 rot(q);
        // same formula as transformInertiaTensor
        const Mat33 sqrtInvInertia = Mat33(rot * Mat33(Vec3(safeSqrt.x, 0, 0), Vec3(0, safeSqrt.y, 0), Vec3(0, 0, safeSqrt.z)) * rot.getTranspose());
        sqrtInvInertia.store(txIData[a].sqrtInvInertia);

        motionVel[a] = Float4{ sym(rng) * velScale, sym(rng) * velScale, sym(rng) * velScale, 0 };
        motionVel[a + nbBodies] = Float4{ sym(rng) * velScale, sym(rng) * velScale, sym(rng) * velScale, 0 };
        velPool[deltaVOffset + a] = Float4{ sym(rng) * velScale, sym(rng) * velScale, sym(rng) * velScale, 0 };
        velPool[deltaVOffset + a + nbBodies] =
            Float4{ sym(rng) * velScale, sym(rng) * velScale, sym(rng) * velScale, 0 };
    }
    for (uint32_t i = 0; i < deltaVOffset; ++i)
        velPool[i] = Float4{ 999.f, 999.f, 999.f, 999.f }; // poison: must never be read

    // ---- GPU ----
    PxgSlangComputeDevice* dev = testDevice();

    Kernel kernel("integration.spv", "integrateCoreParallelLaunch",
                  { 16, 16, sizeof(SolverTxIData), sizeof(SolverBodyData), sizeof(BodySim), 4, 4, 4,
                    sizeof(SolverBodySleepData), 16, sizeof(AlignedTransform), 16 },
                  sizeof(IntegrationParams));

    PxgBufferHandle motionVelBuf = createBufferFrom(dev, motionVel);
    PxgBufferHandle velPoolBuf = createBufferFrom(dev, velPool);
    PxgBufferHandle txIBuf = createBufferFrom(dev, txIData);
    PxgBufferHandle solverDataBuf = createBufferFrom(dev, solverData);
    PxgBufferHandle bodySimBuf = createBufferFrom(dev, bodySims);
    PxgBufferHandle islandIdsBuf = createBufferFrom(dev, islandIds);
    PxgBufferHandle staticTouchBuf = createBufferFrom(dev, staticTouchCounts);
    PxgBufferHandle countedBuf = createBufferFrom(dev, numCountedInteractions);
    PxgBufferHandle sleepDataBuf = dev->createBuffer(nbBodies * sizeof(SolverBodySleepData));
    PxgBufferHandle outVelBuf = dev->createBuffer(2 * nbBodies * sizeof(Float4));
    PxgBufferHandle outPoseBuf = dev->createBuffer(nbBodies * sizeof(AlignedTransform));
    PxgBufferHandle prevVelBuf = dev->createBuffer(maxNodes * 2 * sizeof(Float4));

    const IntegrationParams params = { 0, nbBodies, deltaVOffset, dt, invDt,
                                       enableStabilization ? 1u : 0u, 1u };
    const uint32_t blockSize = 128;
    kernel.dispatch((nbBodies + blockSize - 1) / blockSize,
                    { motionVelBuf, velPoolBuf, txIBuf, solverDataBuf, bodySimBuf, islandIdsBuf,
                      staticTouchBuf, countedBuf, sleepDataBuf, outVelBuf, outPoseBuf, prevVelBuf },
                    params);
    dev->flush();

    // ---- reference ----
    std::vector<BodySim> refBodySims = bodySims;
    std::vector<SolverBodySleepData> refSleep(nbBodies);
    std::vector<Float4> refOutVel(2 * nbBodies);
    std::vector<AlignedTransform> refOutPose(nbBodies);
    std::vector<Float4> refPrevVel(maxNodes * 2);

    for (uint32_t a = 0; a < nbBodies; ++a)
    {
        const SolverBodyData& data = solverData[a];
        const uint32_t nodeIndex = uint32_t(data.islandNodeIndex & 0xFFFFFFFFu);
        BodySim& bodySim = refBodySims[nodeIndex];

        const Mat33 sqrtInvInertia = Mat33::load(txIData[a].sqrtInvInertia);
        AlignedTransform body2World = bodySim.body2World;
        const Float4 inverseInertia = bodySim.inverseInertiaXYZ_contactReportThresholdW;

        Float4 linVel = velPool[deltaVOffset + a];
        Float4 angVel = velPool[deltaVOffset + a + nbBodies];

        const bool hasStaticTouch = staticTouchCounts[islandIds[nodeIndex]] != 0;

        SolverBodySleepData sleepData = {};
        refIntegrateCore(motionVel[a], motionVel[a + nbBodies], inverseInertia, linVel, angVel, body2World,
                         data, bodySim, sleepData, sqrtInvInertia, dt, enableStabilization, hasStaticTouch,
                         numCountedInteractions[nodeIndex]);

        refSleep[a] = sleepData;
        refOutVel[a] = linVel;
        refOutVel[a + nbBodies] = angVel;
        refOutPose[a] = body2World;

        refPrevVel[nodeIndex * 2 + 0] = bodySim.linearVelocityXYZ_inverseMassW;
        refPrevVel[nodeIndex * 2 + 1] = bodySim.angularVelocityXYZ_maxPenBiasW;

        bodySim.linearVelocityXYZ_inverseMassW.x = linVel.x;
        bodySim.linearVelocityXYZ_inverseMassW.y = linVel.y;
        bodySim.linearVelocityXYZ_inverseMassW.z = linVel.z;
        bodySim.angularVelocityXYZ_maxPenBiasW.x = angVel.x;
        bodySim.angularVelocityXYZ_maxPenBiasW.y = angVel.y;
        bodySim.angularVelocityXYZ_maxPenBiasW.z = angVel.z;
        bodySim.body2World = body2World;
    }

    // ---- compare ----
    const std::vector<SolverBodySleepData> gpuSleep = downloadBuffer<SolverBodySleepData>(dev, sleepDataBuf, nbBodies);
    const std::vector<Float4> gpuOutVel = downloadBuffer<Float4>(dev, outVelBuf, 2 * nbBodies);
    const std::vector<AlignedTransform> gpuOutPose = downloadBuffer<AlignedTransform>(dev, outPoseBuf, nbBodies);
    const std::vector<BodySim> gpuBodySims = downloadBuffer<BodySim>(dev, bodySimBuf, maxNodes);

    auto eq4 = [&](const Float4& g, const Float4& r) {
        return almostEqual(g.x, r.x, 1e-4f, 2e-5f) && almostEqual(g.y, r.y, 1e-4f, 2e-5f) &&
               almostEqual(g.z, r.z, 1e-4f, 2e-5f) && almostEqual(g.w, r.w, 1e-4f, 2e-5f);
    };

    for (uint32_t a = 0; a < nbBodies; ++a)
    {
        PXG_CHECK_MSG(gpuSleep[a].internalFlags == refSleep[a].internalFlags,
                      "sleep.flags[%u]: gpu=0x%x ref=0x%x", a, gpuSleep[a].internalFlags,
                      refSleep[a].internalFlags);
        PXG_CHECK_MSG(almostEqual(gpuSleep[a].wakeCounter, refSleep[a].wakeCounter, 1e-4f, 1e-6f),
                      "sleep.wakeCounter[%u]: gpu=%g ref=%g", a, gpuSleep[a].wakeCounter,
                      refSleep[a].wakeCounter);
        PXG_CHECK_MSG(eq4(gpuOutVel[a], refOutVel[a]), "outLinVel[%u]: gpu=(%g,%g,%g) ref=(%g,%g,%g)", a,
                      gpuOutVel[a].x, gpuOutVel[a].y, gpuOutVel[a].z, refOutVel[a].x, refOutVel[a].y,
                      refOutVel[a].z);
        PXG_CHECK_MSG(eq4(gpuOutVel[a + nbBodies], refOutVel[a + nbBodies]), "outAngVel[%u]", a);
        PXG_CHECK_MSG(eq4(gpuOutPose[a].p, refOutPose[a].p), "outPose.p[%u]: gpu=(%g,%g,%g) ref=(%g,%g,%g)",
                      a, gpuOutPose[a].p.x, gpuOutPose[a].p.y, gpuOutPose[a].p.z, refOutPose[a].p.x,
                      refOutPose[a].p.y, refOutPose[a].p.z);
        PXG_CHECK_MSG(eq4(gpuOutPose[a].q, refOutPose[a].q), "outPose.q[%u]: gpu=(%g,%g,%g,%g) ref=(%g,%g,%g,%g)",
                      a, gpuOutPose[a].q.x, gpuOutPose[a].q.y, gpuOutPose[a].q.z, gpuOutPose[a].q.w,
                      refOutPose[a].q.x, refOutPose[a].q.y, refOutPose[a].q.z, refOutPose[a].q.w);

        const uint32_t nodeIndex = uint32_t(solverData[a].islandNodeIndex & 0xFFFFFFFFu);
        const BodySim& gb = gpuBodySims[nodeIndex];
        const BodySim& rb = refBodySims[nodeIndex];
        PXG_CHECK_MSG(gb.internalFlags == rb.internalFlags, "bodySim.flags[node %u]: gpu=0x%x ref=0x%x",
                      nodeIndex, gb.internalFlags, rb.internalFlags);
        PXG_CHECK_MSG(eq4(gb.linearVelocityXYZ_inverseMassW, rb.linearVelocityXYZ_inverseMassW),
                      "bodySim.linVel[node %u]", nodeIndex);
        PXG_CHECK_MSG(eq4(gb.angularVelocityXYZ_maxPenBiasW, rb.angularVelocityXYZ_maxPenBiasW),
                      "bodySim.angVel[node %u]", nodeIndex);
        PXG_CHECK_MSG(eq4(gb.sleepLinVelAccXYZ_freezeCountW, rb.sleepLinVelAccXYZ_freezeCountW),
                      "bodySim.sleepLinAcc[node %u]", nodeIndex);
        PXG_CHECK_MSG(eq4(gb.sleepAngVelAccXYZ_accelScaleW, rb.sleepAngVelAccXYZ_accelScaleW),
                      "bodySim.sleepAngAcc[node %u]", nodeIndex);
        PXG_CHECK_MSG(almostEqual(gb.freezeThresholdX_wakeCounterY_sleepThresholdZ_bodySimIndex.y,
                                  rb.freezeThresholdX_wakeCounterY_sleepThresholdZ_bodySimIndex.y, 1e-4f, 1e-6f),
                      "bodySim.wakeCounter[node %u]", nodeIndex);
        PXG_CHECK_MSG(eq4(gb.body2World.p, rb.body2World.p), "bodySim.pose.p[node %u]", nodeIndex);
        PXG_CHECK_MSG(eq4(gb.body2World.q, rb.body2World.q), "bodySim.pose.q[node %u]", nodeIndex);
        PXG_CHECK_MSG(eq4(gb.externalLinearAcceleration, rb.externalLinearAcceleration),
                      "bodySim.extLinAcc[node %u]", nodeIndex);
    }

    dev->destroyBuffer(motionVelBuf);
    dev->destroyBuffer(velPoolBuf);
    dev->destroyBuffer(txIBuf);
    dev->destroyBuffer(solverDataBuf);
    dev->destroyBuffer(bodySimBuf);
    dev->destroyBuffer(islandIdsBuf);
    dev->destroyBuffer(staticTouchBuf);
    dev->destroyBuffer(countedBuf);
    dev->destroyBuffer(sleepDataBuf);
    dev->destroyBuffer(outVelBuf);
    dev->destroyBuffer(outPoseBuf);
    dev->destroyBuffer(prevVelBuf);
}

PXG_TEST(integration_activeBodies)
{
    checkIntegration(1000, 500, false, false);
}

PXG_TEST(integration_stabilization)
{
    checkIntegration(1000, 501, true, false);
}

PXG_TEST(integration_sleepyBodies)
{
    // small velocities drive the wake-counter / freeze state machine
    checkIntegration(1000, 502, false, true);
    checkIntegration(1000, 503, true, true);
}

PXG_TEST(integration_partialBlock)
{
    checkIntegration(1, 504, true, false);
    checkIntegration(129, 505, false, true);
}

} // namespace
