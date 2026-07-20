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
#include "PxgCudaShim.h"

#include <cstring>
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

// initStaticKinematics on the ENGINE ABI (P5 Step 4): the kernel takes the
// CUDA param pack (PxU32 nbStaticKinematics, PxU32 nbSolverBodies, six device
// pointers) as a 56-byte blob through PxgCudaShim, with the engine's grid
// arithmetic (nbStaticBlocks = ceil(nbStaticKinematics / PRE_INTEGRATION =
// 128)).  Pure copies -> bit-exact comparison.
static std::vector<uint8_t> preIntShimLoader(const char* file) { return loadShaderBlob(file); }

template <typename T>
static std::vector<T> downloadShim(physx::gpuslang::PxgCudaShim& shim, physx::gpuslang::PxgDevicePtr p,
                                   uint32_t count)
{
    std::vector<T> out(count);
    shim.memcpyDtoH(out.data(), p, size_t(count) * sizeof(T));
    return out;
}

void checkInitStaticKinematics(uint32_t nbStaticKinematics, uint32_t seed)
{
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> unit(0.f, 1.f);
    std::uniform_real_distribution<float> sym(-10.f, 10.f);

    // static/kinematic bodies live in the first nbStaticKinematics solver
    // slots; the velocity pool is strided by the FULL solver body count
    const uint32_t nbSolverBodies = nbStaticKinematics + 64;
    const uint32_t maxNodes = nbStaticKinematics + 16;

    std::vector<uint32_t> nodePool(maxNodes);
    std::iota(nodePool.begin(), nodePool.end(), 0u);
    std::shuffle(nodePool.begin(), nodePool.end(), rng);

    // ~25% static bodies (PxNodeIndex with index PXG_INVALID_NODE), the rest
    // kinematic (valid node index -> solverBodyIndices write)
    std::vector<uint64_t> activeNodeIndices(nbStaticKinematics);
    for (uint32_t i = 0; i < nbStaticKinematics; ++i)
        activeNodeIndices[i] = (unit(rng) < 0.25f) ? makeNodeIndex(0xFFFFFFFFu) : makeNodeIndex(nodePool[i]);

    std::vector<SolverBodyData> bodyData(nbStaticKinematics);
    std::memset(bodyData.data(), 0, bodyData.size() * sizeof(SolverBodyData));
    for (SolverBodyData& d : bodyData)
    {
        const QuatV q = randomQuat(rng);
        d.body2World.q = q.toFloat4();
        d.body2World.p = Float4{ sym(rng), sym(rng), sym(rng), 0.f };
        d.initialLinVelXYZ_invMassW = Float4{ sym(rng), sym(rng), sym(rng), unit(rng) };
        d.initialAngVelXYZ_penBiasClamp = Float4{ sym(rng), sym(rng), sym(rng), -unit(rng) };
    }

    const float GARBAGE = 77.5f;
    std::vector<SolverTxIData> txiInit(nbStaticKinematics);
    std::memset(txiInit.data(), 0x3B, txiInit.size() * sizeof(SolverTxIData));
    std::vector<AlignedTransform> xformInit(nbStaticKinematics);
    std::memset(xformInit.data(), 0x3C, xformInit.size() * sizeof(AlignedTransform));
    std::vector<Float4> velInit(2 * nbSolverBodies, Float4{ GARBAGE, GARBAGE, GARBAGE, GARBAGE });
    std::vector<uint32_t> bodyIndicesInit(maxNodes, 0xFFFFFFFFu);

    // ---- GPU (engine ABI blob: {4,4,8,8,8,8,8,8} = 56 bytes) ----
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = { { "preIntegration.spv", "initStaticKinematics", 56u, 0 } };
    PxgCudaShim shim(dev, manifest, 1, &preIntShimLoader);

    PxgDevicePtr dData = 0, dTxi = 0, dXform = 0, dVel = 0, dNodes = 0, dIndices = 0;
    shim.memAlloc(&dData, bodyData.size() * sizeof(SolverBodyData));
    shim.memAlloc(&dTxi, txiInit.size() * sizeof(SolverTxIData));
    shim.memAlloc(&dXform, xformInit.size() * sizeof(AlignedTransform));
    shim.memAlloc(&dVel, velInit.size() * sizeof(Float4));
    shim.memAlloc(&dNodes, activeNodeIndices.size() * 8);
    shim.memAlloc(&dIndices, bodyIndicesInit.size() * 4);
    if (!dData || !dTxi || !dXform || !dVel || !dNodes || !dIndices)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    shim.memcpyHtoD(dData, bodyData.data(), bodyData.size() * sizeof(SolverBodyData));
    shim.memcpyHtoD(dTxi, txiInit.data(), txiInit.size() * sizeof(SolverTxIData));
    shim.memcpyHtoD(dXform, xformInit.data(), xformInit.size() * sizeof(AlignedTransform));
    shim.memcpyHtoD(dVel, velInit.data(), velInit.size() * sizeof(Float4));
    shim.memcpyHtoD(dNodes, activeNodeIndices.data(), activeNodeIndices.size() * 8);
    shim.memcpyHtoD(dIndices, bodyIndicesInit.data(), bodyIndicesInit.size() * 4);

    struct
    {
        uint32_t nbStaticKinematics;
        uint32_t nbSolverBodies;
        uint64_t solverBodyDataPool, solverTxIDataPool, transforms, outVelocityPool;
        uint64_t activeNodeIndices, solverBodyIndices;
    } prm = { nbStaticKinematics, nbSolverBodies, dData, dTxi, dXform, dVel, dNodes, dIndices };
    static_assert(sizeof(prm) == 56, "initStaticKinematics blob is 56 bytes");

    // engine grid: nbStaticBlocks = ceil(offset / PRE_INTEGRATION = 128)
    const uint32_t nbStaticBlocks = (nbStaticKinematics + 127) / 128;
    PXG_CHECK_MSG(shim.launchKernel(0, nbStaticBlocks, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch initStaticKinematics");
    shim.streamSynchronize();

    // ---- reference (bit-exact: the kernel only copies/zeroes) ----
    std::vector<uint32_t> refIndices = bodyIndicesInit;
    std::vector<AlignedTransform> refXform = xformInit;
    std::vector<Float4> refVel = velInit;
    std::vector<SolverTxIData> refTxi = txiInit;
    for (uint32_t i = 0; i < nbStaticKinematics; ++i)
    {
        const uint64_t node = activeNodeIndices[i];
        if (uint32_t(node & 0xFFFFFFFFu) != 0xFFFFFFFFu)
            refIndices[uint32_t(node & 0xFFFFFFFFu)] = i;
        refXform[i] = bodyData[i].body2World;
        refVel[i] = bodyData[i].initialLinVelXYZ_invMassW;
        refVel[i + nbSolverBodies] = bodyData[i].initialAngVelXYZ_penBiasClamp;
        refTxi[i].deltaBody2World.q = Quat{ 0, 0, 0, 1 };
        refTxi[i].deltaBody2World.p = Float3{ 0, 0, 0 };
        for (int m = 0; m < 9; ++m)
            refTxi[i].sqrtInvInertia[m] = 0.f;
    }

    std::vector<uint32_t> gIndices(maxNodes);
    std::vector<AlignedTransform> gXform(nbStaticKinematics);
    std::vector<Float4> gVel(2 * nbSolverBodies);
    std::vector<SolverTxIData> gTxi(nbStaticKinematics);
    shim.memcpyDtoH(gIndices.data(), dIndices, gIndices.size() * 4);
    shim.memcpyDtoH(gXform.data(), dXform, gXform.size() * sizeof(AlignedTransform));
    shim.memcpyDtoH(gVel.data(), dVel, gVel.size() * sizeof(Float4));
    shim.memcpyDtoH(gTxi.data(), dTxi, gTxi.size() * sizeof(SolverTxIData));

    size_t mm = 0;
    if (std::memcmp(gIndices.data(), refIndices.data(), gIndices.size() * 4) != 0) ++mm;
    if (std::memcmp(gXform.data(), refXform.data(), gXform.size() * sizeof(AlignedTransform)) != 0) ++mm;
    if (std::memcmp(gVel.data(), refVel.data(), gVel.size() * sizeof(Float4)) != 0) ++mm;
    if (std::memcmp(gTxi.data(), refTxi.data(), gTxi.size() * sizeof(SolverTxIData)) != 0) ++mm;
    PXG_CHECK_MSG(mm == 0, "initStaticKinematics(n=%u): %zu buffer mismatches (bitwise)",
                  nbStaticKinematics, mm);

    for (PxgDevicePtr pfree : { dData, dTxi, dXform, dVel, dNodes, dIndices })
        shim.memFree(pfree);
}

PXG_TEST(preIntegration_initStaticKinematics)
{
    checkInitStaticKinematics(200, 47);
    checkInitStaticKinematics(1, 48);
    checkInitStaticKinematics(129, 49);
}

// preIntegrationLaunchTGS on the ENGINE ABI (P5 Step 4): the TGS entry forwards
// to the SAME preIntegration() as the PGS entry plus a trailing bool
// skipGravityApplication.  The 13-arg CUDA pack (PxU32 offset, PxU32
// nbSolverBodies, PxReal dt, PxVec3 gravity, eight device pointers, bool) packs
// to a 96-byte blob (gravity is a 12-byte, 4-aligned by-value PxVec3, so the
// pointer bank starts at @24).  Same per-body math as preIntegration_randomBodies
// -> same CPU reference (referencePreIntegration).
void checkPreIntegrationTGS(uint32_t nbBodies, uint32_t seed, bool skipGravity, bool mutate)
{
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> unit(0.f, 1.f);
    std::uniform_real_distribution<float> sym(-10.f, 10.f);

    const uint32_t maxNodes = nbBodies + 32;
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
        const float maxLin = unit(rng) < 0.5f ? 1.0f : 1e6f;
        const float maxAng = unit(rng) < 0.5f ? 1.0f : 1e6f;
        b.maxLinearVelocitySqX_maxAngularVelocitySqY_linearDampingZ_angularDampingW =
            Float4{ maxLin, maxAng, unit(rng) * 0.5f, unit(rng) * 0.5f };
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
        if (unit(rng) < 0.3f) flags |= eENABLE_GYROSCOPIC;
        if (unit(rng) < 0.2f) flags |= eSPECULATIVE_CCD;
        b.internalFlags = flags;
        b.lockFlags = uint16_t(rng() & 0x3Fu & (unit(rng) < 0.3f ? 0x3Fu : 0u));
        b.disableGravity = uint16_t(unit(rng) < 0.25f ? 1 : 0);
        b.offsetSlop = unit(rng) * 0.01f;
        b.externalLinearAcceleration = Float4{ sym(rng), sym(rng), sym(rng), 0 };
        b.externalAngularAcceleration = Float4{ sym(rng), sym(rng), sym(rng), 0 };
    }

    const float dt = 1.f / 60.f;
    const Vec3 gravity(0.f, -9.81f, 0.f);

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = { { "preIntegration.spv", "preIntegrationLaunchTGS", 96u, 0 } };
    PxgCudaShim shim(dev, manifest, 1, &preIntShimLoader);

    const float GARBAGE = 77.5f;
    std::vector<SolverBodyData> dataInit(nbBodies);
    std::memset(dataInit.data(), 0x3A, dataInit.size() * sizeof(SolverBodyData));
    std::vector<SolverTxIData> txiInit(nbBodies);
    std::memset(txiInit.data(), 0x3B, txiInit.size() * sizeof(SolverTxIData));
    std::vector<AlignedTransform> xformInit(nbBodies);
    std::memset(xformInit.data(), 0x3C, xformInit.size() * sizeof(AlignedTransform));
    std::vector<Float4> velInit(2 * nbBodies, Float4{ GARBAGE, GARBAGE, GARBAGE, GARBAGE });
    std::vector<uint32_t> bodyIndicesInit(maxNodes, 0xFFFFFFFFu);

    PxgDevicePtr dData = 0, dSleep = 0, dTxi = 0, dBody = 0, dNodes = 0, dXform = 0, dVel = 0, dIndices = 0;
    shim.memAlloc(&dData, dataInit.size() * sizeof(SolverBodyData));
    shim.memAlloc(&dSleep, nbBodies * 32);        // solverBodySleepDataPool, unused by the kernel
    shim.memAlloc(&dTxi, txiInit.size() * sizeof(SolverTxIData));
    shim.memAlloc(&dBody, bodySims.size() * sizeof(BodySim));
    shim.memAlloc(&dNodes, islandNodeIndices.size() * 8);
    shim.memAlloc(&dXform, xformInit.size() * sizeof(AlignedTransform));
    shim.memAlloc(&dVel, velInit.size() * sizeof(Float4));
    shim.memAlloc(&dIndices, bodyIndicesInit.size() * 4);
    if (!dData || !dTxi || !dBody || !dNodes || !dXform || !dVel || !dIndices)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    shim.memcpyHtoD(dData, dataInit.data(), dataInit.size() * sizeof(SolverBodyData));
    shim.memcpyHtoD(dTxi, txiInit.data(), txiInit.size() * sizeof(SolverTxIData));
    shim.memcpyHtoD(dBody, bodySims.data(), bodySims.size() * sizeof(BodySim));
    shim.memcpyHtoD(dNodes, islandNodeIndices.data(), islandNodeIndices.size() * 8);
    shim.memcpyHtoD(dXform, xformInit.data(), xformInit.size() * sizeof(AlignedTransform));
    shim.memcpyHtoD(dVel, velInit.data(), velInit.size() * sizeof(Float4));
    shim.memcpyHtoD(dIndices, bodyIndicesInit.data(), bodyIndicesInit.size() * 4);

    struct
    {
        uint32_t offset;
        uint32_t nbSolverBodies;
        float dt;
        float gx, gy, gz;
        uint64_t solverBodyDataPool, solverBodySleepDataPool, solverTxIDataPool, bodySimPool;
        uint64_t islandNodeIndices, transforms, outVelocityPool, solverBodyIndices;
        uint32_t skipGravityApplication;
    } prm = { 0, nbBodies, dt, gravity.x, gravity.y, gravity.z,
              dData, dSleep, dTxi, dBody, dNodes, dXform, dVel, dIndices, skipGravity ? 1u : 0u };
    static_assert(sizeof(prm) == 96, "preIntegrationLaunchTGS blob is 96 bytes");

    // mutation: swap the transforms and outVelocityPool device pointers so the
    // kernel writes the pose where velocities are expected -> the compare fails.
    if (mutate) { const uint64_t t = prm.transforms; prm.transforms = prm.outVelocityPool; prm.outVelocityPool = t; }

    const uint32_t nbBlocks = (nbBodies + 127) / 128;
    PXG_CHECK_MSG(shim.launchKernel(0, nbBlocks, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch preIntegrationLaunchTGS");
    shim.streamSynchronize();

    RefOutputs ref;
    referencePreIntegration(nbBodies, dt, gravity, skipGravity, bodySims, islandNodeIndices, maxNodes, ref);

    std::vector<SolverBodyData> gData = downloadShim<SolverBodyData>(shim, dData, nbBodies);
    std::vector<SolverTxIData> gTxi = downloadShim<SolverTxIData>(shim, dTxi, nbBodies);
    std::vector<AlignedTransform> gXform = downloadShim<AlignedTransform>(shim, dXform, nbBodies);
    std::vector<Float4> gVel = downloadShim<Float4>(shim, dVel, 2 * nbBodies);
    std::vector<uint32_t> gIndices = downloadShim<uint32_t>(shim, dIndices, maxNodes);

    size_t bad = 0;
    auto f4bad = [&](const Float4& g, const Float4& r) {
        return !(almostEqual(g.x, r.x) && almostEqual(g.y, r.y) && almostEqual(g.z, r.z) && almostEqual(g.w, r.w));
    };
    for (uint32_t i = 0; i < nbBodies; ++i)
    {
        if (f4bad(gVel[i], ref.velocityPool[i])) ++bad;
        if (f4bad(gVel[i + nbBodies], ref.velocityPool[i + nbBodies])) ++bad;
        if (f4bad(gXform[i].q, ref.transforms[i].q)) ++bad;
        if (f4bad(gXform[i].p, ref.transforms[i].p)) ++bad;
        if (f4bad(gData[i].initialLinVelXYZ_invMassW, ref.solverBodyData[i].initialLinVelXYZ_invMassW)) ++bad;
        if (gData[i].islandNodeIndex != ref.solverBodyData[i].islandNodeIndex) ++bad;
        if (gData[i].flags != ref.solverBodyData[i].flags) ++bad;
        for (int m = 0; m < 9; ++m)
            if (!almostEqual(gTxi[i].sqrtInvInertia[m], ref.txIData[i].sqrtInvInertia[m], 1e-4f, 1e-5f)) ++bad;
    }
    for (uint32_t n = 0; n < maxNodes; ++n)
        if (gIndices[n] != ref.solverBodyIndices[n]) ++bad;

    if (mutate)
        PXG_CHECK_MSG(bad != 0, "preIntegrationLaunchTGS mutation must FAIL (swapped transforms/velocity ptrs)");
    else
        PXG_CHECK_MSG(bad == 0, "preIntegrationLaunchTGS(n=%u,skip=%d): %zu mismatches", nbBodies, skipGravity ? 1 : 0, bad);

    for (PxgDevicePtr p : { dData, dSleep, dTxi, dBody, dNodes, dXform, dVel, dIndices })
        shim.memFree(p);
}

PXG_TEST(preIntegration_launchTGS)
{
    checkPreIntegrationTGS(1000, 42, false, false);
    checkPreIntegrationTGS(1, 43, false, false);
    checkPreIntegrationTGS(129, 44, false, false);
    checkPreIntegrationTGS(500, 45, true, false);   // skipGravityApplication
    checkPreIntegrationTGS(200, 46, false, true);   // mutation: must FAIL then be caught
}

} // namespace
