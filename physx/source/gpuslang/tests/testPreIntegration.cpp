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

// Verification of preIntegration.slang against a C++ transcription of the
// CUDA kernel (preIntegration.cu / preIntegration.cuh).

#include "pxgTestFramework.h"
#include "pxgGpuStructs.h"
#include "pxgRefMath.h"

#include <numeric>
#include <random>

namespace
{

using namespace pxgtest;

struct PreIntegrationParams
{
    uint32_t offset;
    uint32_t nbSolverBodies;
    float dt;
    float gravityX, gravityY, gravityZ;
    uint32_t skipGravityApplication;
};

// ---------------------------------------------------------------------------
// CPU reference: direct transcription of preIntegration.cuh
// ---------------------------------------------------------------------------

Vec3 computeSafeSqrtInertia(const Vec3& v)
{
    return Vec3(v.x == 0.f ? 0.f : std::sqrt(v.x), v.y == 0.f ? 0.f : std::sqrt(v.y),
                v.z == 0.f ? 0.f : std::sqrt(v.z));
}

// mirrors transformInertiaTensor(): M * diag(invD) * M^T (symmetric result)
Mat33 transformInertiaTensor(const Vec3& invD, const Mat33& M)
{
    const float axx = invD.x * M.at(0, 0), axy = invD.x * M.at(1, 0), axz = invD.x * M.at(2, 0);
    const float byx = invD.y * M.at(0, 1), byy = invD.y * M.at(1, 1), byz = invD.y * M.at(2, 1);
    const float czx = invD.z * M.at(0, 2), czy = invD.z * M.at(1, 2), czz = invD.z * M.at(2, 2);

    const float m00 = axx * M.at(0, 0) + byx * M.at(0, 1) + czx * M.at(0, 2);
    const float m11 = axy * M.at(1, 0) + byy * M.at(1, 1) + czy * M.at(1, 2);
    const float m22 = axz * M.at(2, 0) + byz * M.at(2, 1) + czz * M.at(2, 2);

    const float m01 = axx * M.at(1, 0) + byx * M.at(1, 1) + czx * M.at(1, 2);
    const float m02 = axx * M.at(2, 0) + byx * M.at(2, 1) + czx * M.at(2, 2);
    const float m12 = axy * M.at(2, 0) + byy * M.at(2, 1) + czy * M.at(2, 2);

    return Mat33(Vec3(m00, m01, m02), Vec3(m01, m11, m12), Vec3(m02, m12, m22));
}

Vec3 getGravityAcceleration(uint32_t disableGravity, float accelScale, const Vec3& gravity, float dt)
{
    if (!disableGravity)
        return gravity * accelScale * dt;
    return Vec3(0.f);
}

void bodyCoreComputeUnconstrainedVelocity(const Float4& maxVelSq_damping, Float4& linVel_invMass,
                                          Float4& angVel_maxPenBias, uint32_t lockFlags,
                                          uint32_t disableGravity, float accelScale, const Vec3& gravity,
                                          const Float4& linearAccel, const Float4& angularAccel, float dt)
{
    Vec3 linearVelocity(linVel_invMass.x, linVel_invMass.y, linVel_invMass.z);
    Vec3 angularVelocity(angVel_maxPenBias.x, angVel_maxPenBias.y, angVel_maxPenBias.z);
    const Float4 temp = maxVelSq_damping;

    const Vec3 linearAccelTimesDT =
        getGravityAcceleration(disableGravity, accelScale, gravity, dt) + Vec3(linearAccel.x, linearAccel.y, linearAccel.z) * dt;
    const Vec3 angularAccelTimesDT = Vec3(angularAccel.x, angularAccel.y, angularAccel.z) * dt;
    const float linearDampingTimesDT = temp.z * dt;
    const float angularDampingTimesDT = temp.w * dt;
    const float oneMinusLinearDampingTimesDT = 1.0f - linearDampingTimesDT;
    const float oneMinusAngularDampingTimesDT = 1.0f - angularDampingTimesDT;

    linearVelocity += linearAccelTimesDT;
    angularVelocity += angularAccelTimesDT;

    const float linVelMultiplier = fsel(oneMinusLinearDampingTimesDT, oneMinusLinearDampingTimesDT, 0.0f);
    const float angVelMultiplier = fsel(oneMinusAngularDampingTimesDT, oneMinusAngularDampingTimesDT, 0.0f);
    linearVelocity *= linVelMultiplier;
    angularVelocity *= angVelMultiplier;

    const float angVelSq = angularVelocity.magnitudeSquared();
    if (angVelSq > temp.y)
        angularVelocity *= std::sqrt(temp.y / angVelSq);

    const float linVelSq = linearVelocity.magnitudeSquared();
    if (linVelSq > temp.x)
        linearVelocity *= std::sqrt(temp.x / linVelSq);

    linVel_invMass.x = (lockFlags & eLOCK_LINEAR_X) ? 0.f : linearVelocity.x;
    linVel_invMass.y = (lockFlags & eLOCK_LINEAR_Y) ? 0.f : linearVelocity.y;
    linVel_invMass.z = (lockFlags & eLOCK_LINEAR_Z) ? 0.f : linearVelocity.z;
    angVel_maxPenBias.x = (lockFlags & eLOCK_ANGULAR_X) ? 0.f : angularVelocity.x;
    angVel_maxPenBias.y = (lockFlags & eLOCK_ANGULAR_Y) ? 0.f : angularVelocity.y;
    angVel_maxPenBias.z = (lockFlags & eLOCK_ANGULAR_Z) ? 0.f : angularVelocity.z;
}

struct RefOutputs
{
    std::vector<SolverBodyData> solverBodyData;
    std::vector<SolverTxIData> txIData;
    std::vector<AlignedTransform> transforms;
    std::vector<Float4> velocityPool; // 2 * nbSolverBodies
    std::vector<uint32_t> solverBodyIndices;
};

void referencePreIntegration(uint32_t nbSolverBodies, float dt, const Vec3& gravity, bool skipGravity,
                             const std::vector<BodySim>& bodySims, const std::vector<uint64_t>& islandNodeIndices,
                             uint32_t maxNodes, RefOutputs& out)
{
    out.solverBodyData.resize(nbSolverBodies);
    out.txIData.resize(nbSolverBodies);
    out.transforms.resize(nbSolverBodies);
    out.velocityPool.resize(2 * nbSolverBodies);
    out.solverBodyIndices.assign(maxNodes, 0xFFFFFFFFu);

    for (uint32_t a = 0; a < nbSolverBodies; ++a)
    {
        const uint32_t nodeIndex = uint32_t(islandNodeIndices[a] & 0xFFFFFFFFu);
        out.solverBodyIndices[nodeIndex] = a;

        const BodySim& bodySim = bodySims[nodeIndex];

        Float4 linVel_invMass = bodySim.linearVelocityXYZ_inverseMassW;
        Float4 angVel_maxPenBias = bodySim.angularVelocityXYZ_maxPenBiasW;
        const Float4 sleepAng_accelScale = bodySim.sleepAngVelAccXYZ_accelScaleW;
        const Float4 invInertia_reportThreshold = bodySim.inverseInertiaXYZ_contactReportThresholdW;
        const float maxImpulse = bodySim.body2Actor_maxImpulseW.p.w;
        const uint32_t internalFlags = bodySim.internalFlags;
        const uint32_t lockFlags = bodySim.lockFlags;
        const uint32_t disableGravity = bodySim.disableGravity;

        AlignedTransform body2World;
        body2World.q = bodySim.body2World.q;
        body2World.p = Float4{ bodySim.body2World.p.x, bodySim.body2World.p.y, bodySim.body2World.p.z, 0.f };

        const Float4 zero4 = { 0, 0, 0, 0 };
        const Float4 linearAccel = skipGravity ? zero4 : bodySim.externalLinearAcceleration;
        const Float4 angularAccel = skipGravity ? zero4 : bodySim.externalAngularAcceleration;

        bodyCoreComputeUnconstrainedVelocity(
            bodySim.maxLinearVelocitySqX_maxAngularVelocitySqY_linearDampingZ_angularDampingW,
            linVel_invMass, angVel_maxPenBias, lockFlags,
            (disableGravity || skipGravity) ? 1u : 0u, sleepAng_accelScale.w, gravity, linearAccel,
            angularAccel, dt);

        const Vec3 inverseInertia(invInertia_reportThreshold.x, invInertia_reportThreshold.y, invInertia_reportThreshold.z);
        const Vec3 safeSqrtInvInertia = computeSafeSqrtInertia(inverseInertia);
        const Mat33 rotation(QuatV(body2World.q));
        const Mat33 sqrtInvInertia = transformInertiaTensor(safeSqrtInvInertia, rotation);

        out.velocityPool[a] = linVel_invMass;

        const Vec3 sqrtInertiaV(safeSqrtInvInertia.x == 0.f ? 0.f : 1.f / safeSqrtInvInertia.x,
                                safeSqrtInvInertia.y == 0.f ? 0.f : 1.f / safeSqrtInvInertia.y,
                                safeSqrtInvInertia.z == 0.f ? 0.f : 1.f / safeSqrtInvInertia.z);

        const Mat33 sqrtInertia = transformInertiaTensor(sqrtInertiaV, rotation);

        Vec3 angVel(angVel_maxPenBias.x, angVel_maxPenBias.y, angVel_maxPenBias.z);

        if (internalFlags & eENABLE_GYROSCOPIC)
        {
            const Vec3 localInertia(inverseInertia.x == 0.f ? 0.f : 1.f / inverseInertia.x,
                                    inverseInertia.y == 0.f ? 0.f : 1.f / inverseInertia.y,
                                    inverseInertia.z == 0.f ? 0.f : 1.f / inverseInertia.z);

            const QuatV q(body2World.q);
            const Vec3 localAngVel = q.rotateInv(angVel);
            const Vec3 origMom = localInertia.multiply(localAngVel);
            const Vec3 torque = -localAngVel.cross(origMom);
            Vec3 newMom = origMom + torque * dt;
            const float denom = newMom.magnitude();
            const float ratio = denom > 0.f ? origMom.magnitude() / denom : 0.f;
            newMom *= ratio;
            const Vec3 newDeltaAngVel = q.rotate(inverseInertia.multiply(newMom) - localAngVel);

            angVel += newDeltaAngVel;
        }

        angVel_maxPenBias.x = angVel.x;
        angVel_maxPenBias.y = angVel.y;
        angVel_maxPenBias.z = angVel.z;

        angVel = sqrtInertia.transform(angVel);

        out.velocityPool[a + nbSolverBodies] = Float4{ angVel.x, angVel.y, angVel.z, angVel_maxPenBias.w };

        out.transforms[a] = body2World;

        SolverBodyData& data = out.solverBodyData[a];
        std::memset(&data, 0, sizeof(data));
        data.body2World = body2World;
        data.initialLinVelXYZ_invMassW = linVel_invMass;
        data.initialAngVelXYZ_penBiasClamp = angVel_maxPenBias;
        data.offsetSlop = bodySim.offsetSlop;
        data.reportThreshold = invInertia_reportThreshold.w;
        data.islandNodeIndex = islandNodeIndices[a];
        data.maxImpulse = maxImpulse;
        uint32_t flags = 0;
        if (internalFlags & eSPECULATIVE_CCD)
            flags |= eRB_ENABLE_SPECULATIVE_CCD;
        if (internalFlags & eENABLE_GYROSCOPIC)
            flags |= eRB_ENABLE_GYROSCOPIC_FORCES;
        data.flags = flags;

        SolverTxIData& txI = out.txIData[a];
        txI.deltaBody2World.q = Quat{ 0, 0, 0, 1 };
        txI.deltaBody2World.p = Float3{ 0, 0, 0 };
        sqrtInvInertia.store(txI.sqrtInvInertia);
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

void checkPreIntegration(uint32_t nbBodies, uint32_t seed, bool skipGravity)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> unit(0.f, 1.f);
    std::uniform_real_distribution<float> sym(-10.f, 10.f);

    const uint32_t maxNodes = nbBodies + 32;

    // node indices: shuffled subset of [0, maxNodes)
    std::vector<uint32_t> nodePool(maxNodes);
    std::iota(nodePool.begin(), nodePool.end(), 0u);
    std::shuffle(nodePool.begin(), nodePool.end(), rng);

    std::vector<uint64_t> islandNodeIndices(nbBodies);
    for (uint32_t i = 0; i < nbBodies; ++i)
        islandNodeIndices[i] = makeNodeIndex(nodePool[i]);

    std::vector<BodySim> bodySims(maxNodes);
    std::memset(bodySims.data(), 0, bodySims.size() * sizeof(BodySim));
    for (BodySim& b : bodySims)
    {
        b.linearVelocityXYZ_inverseMassW = Float4{ sym(rng), sym(rng), sym(rng), unit(rng) };
        b.angularVelocityXYZ_maxPenBiasW = Float4{ sym(rng), sym(rng), sym(rng), -unit(rng) };
        // roughly half the bodies exceed their velocity clamps
        const float maxLin = unit(rng) < 0.5f ? 1.0f : 1e6f;
        const float maxAng = unit(rng) < 0.5f ? 1.0f : 1e6f;
        b.maxLinearVelocitySqX_maxAngularVelocitySqY_linearDampingZ_angularDampingW =
            Float4{ maxLin, maxAng, unit(rng) * 0.5f, unit(rng) * 0.5f };
        // some zero inverse-inertia components (kinematic-like locked axes)
        b.inverseInertiaXYZ_contactReportThresholdW =
            Float4{ unit(rng) < 0.15f ? 0.f : unit(rng) * 4.f, unit(rng) < 0.15f ? 0.f : unit(rng) * 4.f,
                    unit(rng) < 0.15f ? 0.f : unit(rng) * 4.f, unit(rng) };
        b.sleepLinVelAccXYZ_freezeCountW = Float4{ 0, 0, 0, unit(rng) };
        b.sleepAngVelAccXYZ_accelScaleW = Float4{ 0, 0, 0, unit(rng) };
        b.freezeThresholdX_wakeCounterY_sleepThresholdZ_bodySimIndex = Float4{ unit(rng), unit(rng), unit(rng), 0 };

        const QuatV q = randomQuat(rng);
        b.body2World.q = q.toFloat4();
        b.body2World.p = Float4{ sym(rng), sym(rng), sym(rng), 0.f };
        b.body2Actor_maxImpulseW.q = Float4{ 0, 0, 0, 1 };
        b.body2Actor_maxImpulseW.p = Float4{ 0, 0, 0, unit(rng) * 100.f };

        uint32_t flags = 0;
        if (unit(rng) < 0.3f)
            flags |= eENABLE_GYROSCOPIC;
        if (unit(rng) < 0.2f)
            flags |= eSPECULATIVE_CCD;
        b.internalFlags = flags;
        b.lockFlags = uint16_t(rng() & 0x3Fu & (unit(rng) < 0.3f ? 0x3Fu : 0u));
        b.disableGravity = uint16_t(unit(rng) < 0.25f ? 1 : 0);
        b.offsetSlop = unit(rng) * 0.01f;
        b.externalLinearAcceleration = Float4{ sym(rng), sym(rng), sym(rng), 0 };
        b.externalAngularAcceleration = Float4{ sym(rng), sym(rng), sym(rng), 0 };
    }

    const float dt = 1.f / 60.f;
    const Vec3 gravity(0.f, -9.81f, 0.f);

    // ---- GPU ----
    PxgSlangComputeDevice* dev = testDevice();

    Kernel kernel("preIntegration.spv", "preIntegrationLaunch",
                  { sizeof(SolverBodyData), sizeof(SolverTxIData), sizeof(BodySim), 8,
                    sizeof(AlignedTransform), 16, 4 },
                  sizeof(PreIntegrationParams));

    PxgBufferHandle solverBodyData = dev->createBuffer(nbBodies * sizeof(SolverBodyData));
    PxgBufferHandle txIData = dev->createBuffer(nbBodies * sizeof(SolverTxIData));
    PxgBufferHandle bodySimBuf = createBufferFrom(dev, bodySims);
    PxgBufferHandle nodeIndices = createBufferFrom(dev, islandNodeIndices);
    PxgBufferHandle transforms = dev->createBuffer(nbBodies * sizeof(AlignedTransform));
    PxgBufferHandle velocityPool = dev->createBuffer(2 * nbBodies * sizeof(Float4));
    std::vector<uint32_t> bodyIndicesInit(maxNodes, 0xFFFFFFFFu);
    PxgBufferHandle solverBodyIndices = createBufferFrom(dev, bodyIndicesInit);

    const PreIntegrationParams params = { 0, nbBodies, dt, gravity.x, gravity.y, gravity.z,
                                          skipGravity ? 1u : 0u };
    const uint32_t blockSize = 128;
    kernel.dispatch((nbBodies + blockSize - 1) / blockSize,
                    { solverBodyData, txIData, bodySimBuf, nodeIndices, transforms, velocityPool,
                      solverBodyIndices },
                    params);
    dev->flush();

    // ---- reference ----
    RefOutputs ref;
    referencePreIntegration(nbBodies, dt, gravity, skipGravity, bodySims, islandNodeIndices, maxNodes, ref);

    // ---- compare ----
    const std::vector<SolverBodyData> gpuData = downloadBuffer<SolverBodyData>(dev, solverBodyData, nbBodies);
    const std::vector<SolverTxIData> gpuTxI = downloadBuffer<SolverTxIData>(dev, txIData, nbBodies);
    const std::vector<AlignedTransform> gpuTransforms = downloadBuffer<AlignedTransform>(dev, transforms, nbBodies);
    const std::vector<Float4> gpuVel = downloadBuffer<Float4>(dev, velocityPool, 2 * nbBodies);
    const std::vector<uint32_t> gpuIndices = downloadBuffer<uint32_t>(dev, solverBodyIndices, maxNodes);

    auto checkFloat4 = [&](const char* what, uint32_t i, const Float4& g, const Float4& r) {
        PXG_CHECK_MSG(almostEqual(g.x, r.x) && almostEqual(g.y, r.y) && almostEqual(g.z, r.z) &&
                          almostEqual(g.w, r.w),
                      "%s[%u]: gpu=(%g,%g,%g,%g) ref=(%g,%g,%g,%g)", what, i, g.x, g.y, g.z, g.w, r.x,
                      r.y, r.z, r.w);
    };

    for (uint32_t i = 0; i < nbBodies; ++i)
    {
        checkFloat4("velLin", i, gpuVel[i], ref.velocityPool[i]);
        checkFloat4("velAng", i, gpuVel[i + nbBodies], ref.velocityPool[i + nbBodies]);
        checkFloat4("transform.q", i, gpuTransforms[i].q, ref.transforms[i].q);
        checkFloat4("transform.p", i, gpuTransforms[i].p, ref.transforms[i].p);

        const SolverBodyData& g = gpuData[i];
        const SolverBodyData& r = ref.solverBodyData[i];
        checkFloat4("data.initialAngVel", i, g.initialAngVelXYZ_penBiasClamp, r.initialAngVelXYZ_penBiasClamp);
        checkFloat4("data.initialLinVel", i, g.initialLinVelXYZ_invMassW, r.initialLinVelXYZ_invMassW);
        checkFloat4("data.body2World.q", i, g.body2World.q, r.body2World.q);
        PXG_CHECK_MSG(g.islandNodeIndex == r.islandNodeIndex, "data.islandNodeIndex[%u]", i);
        PXG_CHECK_MSG(g.flags == r.flags, "data.flags[%u]: gpu=%u ref=%u", i, g.flags, r.flags);
        PXG_CHECK_MSG(almostEqual(g.reportThreshold, r.reportThreshold), "data.reportThreshold[%u]", i);
        PXG_CHECK_MSG(almostEqual(g.maxImpulse, r.maxImpulse), "data.maxImpulse[%u]", i);
        PXG_CHECK_MSG(almostEqual(g.offsetSlop, r.offsetSlop), "data.offsetSlop[%u]", i);

        for (int m = 0; m < 9; ++m)
        {
            PXG_CHECK_MSG(almostEqual(gpuTxI[i].sqrtInvInertia[m], ref.txIData[i].sqrtInvInertia[m], 1e-4f, 1e-5f),
                          "txI[%u].m[%d]: gpu=%g ref=%g", i, m, gpuTxI[i].sqrtInvInertia[m],
                          ref.txIData[i].sqrtInvInertia[m]);
        }
    }

    for (uint32_t n = 0; n < maxNodes; ++n)
        PXG_CHECK_MSG(gpuIndices[n] == ref.solverBodyIndices[n], "solverBodyIndices[%u]: gpu=%u ref=%u",
                      n, gpuIndices[n], ref.solverBodyIndices[n]);

    dev->destroyBuffer(solverBodyData);
    dev->destroyBuffer(txIData);
    dev->destroyBuffer(bodySimBuf);
    dev->destroyBuffer(nodeIndices);
    dev->destroyBuffer(transforms);
    dev->destroyBuffer(velocityPool);
    dev->destroyBuffer(solverBodyIndices);
}

PXG_TEST(preIntegration_randomBodies)
{
    checkPreIntegration(1000, 42, false);
}

PXG_TEST(preIntegration_skipGravity)
{
    checkPreIntegration(300, 43, true);
}

PXG_TEST(preIntegration_partialBlock)
{
    checkPreIntegration(1, 44, false);
    checkPreIntegration(127, 45, false);
    checkPreIntegration(129, 46, false);
}

} // namespace
