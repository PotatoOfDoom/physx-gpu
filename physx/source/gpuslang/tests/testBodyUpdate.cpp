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

// Verification of bodyUpdate.slang and directApi.slang (+ the
// setRigidDynamicGlobalPose entry of updateBounds.slang) against C++
// transcriptions of updateBodiesAndShapes.cu.

#include "pxgTestFramework.h"
#include "pxgGpuStructs.h"
#include "pxgRefMath.h"

#include <numeric>
#include <random>

namespace
{

using namespace pxgtest;

enum DmaFlags : uint32_t
{
    eFIRST_BODY_COPY_GPU = 1u << 9,
    eVELOCITY_COPY_GPU = 1u << 10
};

struct BodyUpdateParams
{
    uint32_t count;
};

struct DirectApiParams
{
    uint32_t nbElements;
    float oneOverDt;
    uint32_t hasPrevVelocities;
};

QuatV randomQuat(std::mt19937& rng)
{
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    QuatV q(dist(rng), dist(rng), dist(rng), dist(rng));
    return q.getNormalized();
}

BodySim randomBodySim(std::mt19937& rng, uint32_t bodySimIndex)
{
    std::uniform_real_distribution<float> unit(0.f, 1.f);
    std::uniform_real_distribution<float> sym(-5.f, 5.f);

    BodySim b;
    std::memset(&b, 0, sizeof(b));
    b.linearVelocityXYZ_inverseMassW = Float4{ sym(rng), sym(rng), sym(rng), unit(rng) };
    b.angularVelocityXYZ_maxPenBiasW = Float4{ sym(rng), sym(rng), sym(rng), -unit(rng) };
    b.maxLinearVelocitySqX_maxAngularVelocitySqY_linearDampingZ_angularDampingW =
        Float4{ unit(rng), unit(rng), unit(rng), unit(rng) };
    b.inverseInertiaXYZ_contactReportThresholdW = Float4{ unit(rng), unit(rng), unit(rng), unit(rng) };
    b.sleepLinVelAccXYZ_freezeCountW = Float4{ sym(rng), sym(rng), sym(rng), unit(rng) };
    b.sleepAngVelAccXYZ_accelScaleW = Float4{ sym(rng), sym(rng), sym(rng), unit(rng) };
    Float4 freeze = { unit(rng), unit(rng), unit(rng), 0.f };
    std::memcpy(&freeze.w, &bodySimIndex, 4); // .w carries the body index bit pattern
    b.freezeThresholdX_wakeCounterY_sleepThresholdZ_bodySimIndex = freeze;
    b.body2World.q = randomQuat(rng).toFloat4();
    b.body2World.p = Float4{ sym(rng), sym(rng), sym(rng), 0.f };
    b.body2Actor_maxImpulseW.q = randomQuat(rng).toFloat4();
    b.body2Actor_maxImpulseW.p = Float4{ sym(rng) * 0.2f, sym(rng) * 0.2f, sym(rng) * 0.2f, unit(rng) * 10.f };
    b.articulationRemapId = rng();
    b.internalFlags = 0;
    b.lockFlags = uint16_t(rng() & 0x3F);
    b.disableGravity = uint16_t(rng() & 1);
    b.offsetSlop = unit(rng);
    b.externalLinearAcceleration = Float4{ sym(rng), sym(rng), sym(rng), 0.f };
    b.externalAngularAcceleration = Float4{ sym(rng), sym(rng), sym(rng), 0.f };
    return b;
}

// ---------------------------------------------------------------------------
// updateBodiesLaunch / updateShapesLaunch / updateBodyExternalVelocitiesLaunch
// ---------------------------------------------------------------------------

PXG_TEST(bodyUpdate_scatterKernels)
{
    std::mt19937 rng(2000);
    std::uniform_real_distribution<float> sym(-5.f, 5.f);

    const uint32_t poolSize = 300;
    const uint32_t nbNew = 120;

    // shuffled target indices
    std::vector<uint32_t> targets(poolSize);
    std::iota(targets.begin(), targets.end(), 0u);
    std::shuffle(targets.begin(), targets.end(), rng);

    std::vector<BodySim> pool(poolSize);
    std::memset(pool.data(), 0, pool.size() * sizeof(BodySim));
    std::vector<BodySim> staged(nbNew);
    for (uint32_t i = 0; i < nbNew; ++i)
        staged[i] = randomBodySim(rng, targets[i]);

    PxgSlangComputeDevice* dev = testDevice();

    Kernel updateBodies("bodyUpdate.spv", "updateBodiesLaunch", { 4, 4, 4 }, sizeof(BodyUpdateParams));

    PxgBufferHandle stagedBuf = createBufferFrom(dev, staged);
    PxgBufferHandle poolBuf = createBufferFrom(dev, pool);
    PxgBufferHandle dummyShapes = dev->createBuffer(16);

    updateBodies.dispatch((nbNew + 255) / 256, { stagedBuf, poolBuf, dummyShapes },
                          BodyUpdateParams{ nbNew });
    dev->flush();

    const std::vector<BodySim> gpuPool = downloadBuffer<BodySim>(dev, poolBuf, poolSize);

    std::vector<BodySim> refPool = pool;
    for (uint32_t i = 0; i < nbNew; ++i)
        refPool[targets[i]] = staged[i];

    PXG_CHECK_MSG(std::memcmp(gpuPool.data(), refPool.data(), poolSize * sizeof(BodySim)) == 0,
                  "updateBodiesLaunch pool mismatch");

    dev->destroyBuffer(stagedBuf);
    dev->destroyBuffer(poolBuf);

    // ---- updateShapesLaunch ----
    struct NewShapeSim
    {
        ShapeSim sim;
        uint32_t elementIndex;
        uint32_t pad;
    };
    static_assert(sizeof(NewShapeSim) == 80, "PxgNewShapeSim mirror must be 80 bytes");

    const uint32_t shapePool = 200, nbNewShapes = 80;
    std::vector<uint32_t> shapeTargets(shapePool);
    std::iota(shapeTargets.begin(), shapeTargets.end(), 0u);
    std::shuffle(shapeTargets.begin(), shapeTargets.end(), rng);

    std::vector<NewShapeSim> newShapes(nbNewShapes);
    std::memset(newShapes.data(), 0, newShapes.size() * sizeof(NewShapeSim));
    for (uint32_t i = 0; i < nbNewShapes; ++i)
    {
        NewShapeSim& s = newShapes[i];
        s.sim.transform.q = randomQuat(rng).toQuat();
        s.sim.transform.p = Float3{ sym(rng), sym(rng), sym(rng) };
        s.sim.localBoundsMin = Float3{ -1, -1, -1 };
        s.sim.localBoundsMax = Float3{ 1, 1, 1 };
        s.sim.bodySimIndex = makeNodeIndex(rng() % 64);
        s.sim.hullDataIndex = rng();
        s.sim.shapeFlags = uint16_t(rng() & 7);
        s.sim.shapeType = uint16_t(rng() % 8);
        s.elementIndex = shapeTargets[i];
    }

    std::vector<ShapeSim> shapePoolData(shapePool);
    std::memset(shapePoolData.data(), 0, shapePoolData.size() * sizeof(ShapeSim));

    Kernel updateShapes("bodyUpdate.spv", "updateShapesLaunch", { 4, 4, 4 }, sizeof(BodyUpdateParams));

    PxgBufferHandle newShapesBuf = createBufferFrom(dev, newShapes);
    PxgBufferHandle shapePoolBuf = createBufferFrom(dev, shapePoolData);
    PxgBufferHandle dummyBody = dev->createBuffer(16);

    updateShapes.dispatch((nbNewShapes + 255) / 256, { newShapesBuf, dummyBody, shapePoolBuf },
                          BodyUpdateParams{ nbNewShapes });
    dev->flush();

    const std::vector<ShapeSim> gpuShapes = downloadBuffer<ShapeSim>(dev, shapePoolBuf, shapePool);
    std::vector<ShapeSim> refShapes = shapePoolData;
    for (uint32_t i = 0; i < nbNewShapes; ++i)
        refShapes[newShapes[i].elementIndex] = newShapes[i].sim;

    // PxgShapeSim has 4 bytes of padding at offset 52 that updateShapesLaunch
    // copies verbatim (both are zero here) — memcmp is safe
    PXG_CHECK_MSG(std::memcmp(gpuShapes.data(), refShapes.data(), shapePool * sizeof(ShapeSim)) == 0,
                  "updateShapesLaunch pool mismatch");

    dev->destroyBuffer(newShapesBuf);
    dev->destroyBuffer(shapePoolBuf);
    dev->destroyBuffer(dummyBody);
    dev->destroyBuffer(dummyShapes);
}

PXG_TEST(bodyUpdate_externalVelocities)
{
    std::mt19937 rng(2001);
    std::uniform_real_distribution<float> sym(-5.f, 5.f);

    const uint32_t poolSize = 150;
    const uint32_t nbUpdates = 60;

    std::vector<BodySim> pool(poolSize);
    for (uint32_t i = 0; i < poolSize; ++i)
        pool[i] = randomBodySim(rng, i);

    std::vector<uint32_t> targets(poolSize);
    std::iota(targets.begin(), targets.end(), 0u);
    std::shuffle(targets.begin(), targets.end(), rng);

    // PxgBodySimVelocityUpdate: 4 float4
    std::vector<Float4> updates(nbUpdates * 4);
    for (uint32_t i = 0; i < nbUpdates; ++i)
    {
        Float4 lin = { sym(rng), sym(rng), sym(rng), 0.f };
        std::memcpy(&lin.w, &targets[i], 4);
        updates[i * 4 + 0] = lin;
        updates[i * 4 + 1] = Float4{ sym(rng), sym(rng), sym(rng), sym(rng) };
        updates[i * 4 + 2] = Float4{ sym(rng), sym(rng), sym(rng), 0.f };
        updates[i * 4 + 3] = Float4{ sym(rng), sym(rng), sym(rng), 0.f };
    }

    PxgSlangComputeDevice* dev = testDevice();
    Kernel kernel("bodyUpdate.spv", "updateBodyExternalVelocitiesLaunch", { 4, 4, 4 },
                  sizeof(BodyUpdateParams));

    PxgBufferHandle updatesBuf = createBufferFrom(dev, updates);
    PxgBufferHandle poolBuf = createBufferFrom(dev, pool);
    PxgBufferHandle dummyShapes = dev->createBuffer(16);

    kernel.dispatch((nbUpdates + 255) / 256, { updatesBuf, poolBuf, dummyShapes },
                    BodyUpdateParams{ nbUpdates });
    dev->flush();

    const std::vector<BodySim> gpuPool = downloadBuffer<BodySim>(dev, poolBuf, poolSize);

    std::vector<BodySim> refPool = pool;
    for (uint32_t i = 0; i < nbUpdates; ++i)
    {
        BodySim& b = refPool[targets[i]];
        Float4 lin = updates[i * 4 + 0];
        lin.w = b.linearVelocityXYZ_inverseMassW.w;
        b.linearVelocityXYZ_inverseMassW = lin;
        b.angularVelocityXYZ_maxPenBiasW = updates[i * 4 + 1];
        b.externalLinearAcceleration = updates[i * 4 + 2];
        b.externalAngularAcceleration = updates[i * 4 + 3];
    }

    PXG_CHECK_MSG(std::memcmp(gpuPool.data(), refPool.data(), poolSize * sizeof(BodySim)) == 0,
                  "external velocity update mismatch");

    dev->destroyBuffer(updatesBuf);
    dev->destroyBuffer(poolBuf);
    dev->destroyBuffer(dummyShapes);
}

// ---------------------------------------------------------------------------
// updateBodiesLaunchDirectAPI
// ---------------------------------------------------------------------------

PXG_TEST(bodyUpdate_directAPI)
{
    std::mt19937 rng(2002);
    std::uniform_real_distribution<float> unit(0.f, 1.f);

    const uint32_t poolSize = 200;
    const uint32_t nbNew = 100;

    std::vector<BodySim> pool(poolSize);
    for (uint32_t i = 0; i < poolSize; ++i)
        pool[i] = randomBodySim(rng, i);

    std::vector<uint32_t> targets(poolSize);
    std::iota(targets.begin(), targets.end(), 0u);
    std::shuffle(targets.begin(), targets.end(), rng);

    std::vector<BodySim> staged(nbNew);
    for (uint32_t i = 0; i < nbNew; ++i)
    {
        staged[i] = randomBodySim(rng, targets[i]);
        uint32_t flags = 0;
        const float r = unit(rng);
        if (r < 0.3f)
            flags |= eFIRST_BODY_COPY_GPU;
        if (unit(rng) < 0.5f)
            flags |= eVELOCITY_COPY_GPU;
        staged[i].internalFlags = flags;
        // half the updates keep the old body2Actor (no pose recompute)
        if (unit(rng) < 0.5f)
            staged[i].body2Actor_maxImpulseW = pool[targets[i]].body2Actor_maxImpulseW;
    }

    PxgSlangComputeDevice* dev = testDevice();
    Kernel kernel("bodyUpdate.spv", "updateBodiesLaunchDirectAPI", { 4, 4, 4 }, sizeof(BodyUpdateParams));

    PxgBufferHandle stagedBuf = createBufferFrom(dev, staged);
    PxgBufferHandle poolBuf = createBufferFrom(dev, pool);
    PxgBufferHandle dummyShapes = dev->createBuffer(16);

    kernel.dispatch((nbNew + 255) / 256, { stagedBuf, poolBuf, dummyShapes }, BodyUpdateParams{ nbNew });
    dev->flush();

    const std::vector<BodySim> gpuPool = downloadBuffer<BodySim>(dev, poolBuf, poolSize);

    // ---- reference (transcription of updateBodiesLaunchDirectAPI) ----
    std::vector<BodySim> refPool = pool;
    for (uint32_t i = 0; i < nbNew; ++i)
    {
        const BodySim& src = staged[i];
        BodySim& dst = refPool[targets[i]];

        const bool firstTransfer = (src.internalFlags & eFIRST_BODY_COPY_GPU) != 0;
        const bool copyVel = (src.internalFlags & eVELOCITY_COPY_GPU) != 0;

        if (firstTransfer)
        {
            dst = src;
            continue;
        }

        const AlignedTransform body2WorldOld = dst.body2World;
        const AlignedTransform oldBody2Actor = dst.body2Actor_maxImpulseW;

        // copy uint4s [0, 12) with the velocity .w-only rule
        // (memcpy, not reinterpret_cast word stores: the struct is re-read
        // through its members below, and type-punned stores would be a
        // strict-aliasing violation the optimizer is entitled to exploit)
        const unsigned char* s = reinterpret_cast<const unsigned char*>(&src);
        unsigned char* d = reinterpret_cast<unsigned char*>(&dst);
        for (uint32_t u4 = 0; u4 < 12; ++u4)
        {
            if (u4 < 2 && !copyVel)
                std::memcpy(d + u4 * 16 + 12, s + u4 * 16 + 12, 4);
            else
                std::memcpy(d + u4 * 16, s + u4 * 16, 16);
        }

        const AlignedTransform newBody2Actor = dst.body2Actor_maxImpulseW;

        const bool changed = !(oldBody2Actor.p.x == newBody2Actor.p.x &&
                               oldBody2Actor.p.y == newBody2Actor.p.y &&
                               oldBody2Actor.p.z == newBody2Actor.p.z &&
                               oldBody2Actor.q.x == newBody2Actor.q.x &&
                               oldBody2Actor.q.y == newBody2Actor.q.y &&
                               oldBody2Actor.q.z == newBody2Actor.q.z &&
                               oldBody2Actor.q.w == newBody2Actor.q.w);
        if (changed)
        {
            // PxAlignedTransform math (rotate() zeroes the w lane)
            const QuatV qW(body2WorldOld.q), qA(oldBody2Actor.q), qN(newBody2Actor.q);
            const Vec3 pW(body2WorldOld.p.x, body2WorldOld.p.y, body2WorldOld.p.z);
            const Vec3 pA(oldBody2Actor.p.x, oldBody2Actor.p.y, oldBody2Actor.p.z);
            const Vec3 pN(newBody2Actor.p.x, newBody2Actor.p.y, newBody2Actor.p.z);

            const QuatV qInv = qA.getConjugate();
            const Vec3 pInv = qA.rotateInv(-pA);

            const QuatV qA2W = qW * qInv;
            const Vec3 pA2W = qW.rotate(pInv) + pW;

            const QuatV qNew = qA2W * qN;
            const Vec3 pNew = qA2W.rotate(pN) + pA2W;

            dst.body2World.q = qNew.toFloat4();
            dst.body2World.p = Float4{ pNew.x, pNew.y, pNew.z, body2WorldOld.p.w };
        }
    }

    size_t mismatches = 0;
    for (uint32_t n = 0; n < poolSize; ++n)
    {
        const float* g = reinterpret_cast<const float*>(&gpuPool[n]);
        const float* r = reinterpret_cast<const float*>(&refPool[n]);
        for (uint32_t w = 0; w < sizeof(BodySim) / 4; ++w)
        {
            if (!almostEqual(g[w], r[w], 1e-5f, 1e-6f) &&
                reinterpret_cast<const uint32_t*>(g)[w] != reinterpret_cast<const uint32_t*>(r)[w])
            {
                if (mismatches < 5)
                    PXG_CHECK_MSG(false, "directAPI pool[%u] word %u: gpu=%g ref=%g", n, w, g[w], r[w]);
                ++mismatches;
            }
        }
    }
    PXG_CHECK_MSG(mismatches == 0, "directAPI: %zu word mismatches", mismatches);

    dev->destroyBuffer(stagedBuf);
    dev->destroyBuffer(poolBuf);
    dev->destroyBuffer(dummyShapes);
}

// ---------------------------------------------------------------------------
// direct API getters / setters
// ---------------------------------------------------------------------------

PXG_TEST(directApi_gettersAndSetters)
{
    std::mt19937 rng(2003);
    std::uniform_real_distribution<float> sym(-5.f, 5.f);

    const uint32_t poolSize = 120;
    const uint32_t nbElements = 50;
    const float oneOverDt = 60.f;

    std::vector<BodySim> pool(poolSize);
    for (uint32_t i = 0; i < poolSize; ++i)
        pool[i] = randomBodySim(rng, i);

    std::vector<Float4> prevVel(poolSize * 2);
    for (Float4& v : prevVel)
        v = Float4{ sym(rng), sym(rng), sym(rng), 0.f };

    std::vector<uint32_t> indices(poolSize);
    std::iota(indices.begin(), indices.end(), 0u);
    std::shuffle(indices.begin(), indices.end(), rng);
    indices.resize(nbElements);

    PxgSlangComputeDevice* dev = testDevice();

    const std::vector<uint32_t> strides = { 4, 4, 4, 16 };
    PxgBufferHandle poolBuf = createBufferFrom(dev, pool);
    PxgBufferHandle idxBuf = createBufferFrom(dev, indices);
    PxgBufferHandle prevBuf = createBufferFrom(dev, prevVel);
    PxgBufferHandle vecBuf = dev->createBuffer(nbElements * 3 * sizeof(float));
    PxgBufferHandle poseBuf = dev->createBuffer(nbElements * 7 * sizeof(float));

    const DirectApiParams params = { nbElements, oneOverDt, 1u };
    const uint32_t grid = (nbElements + 255) / 256;

    // ---- getters ----
    {
        Kernel getPose("directApi.spv", "getRigidDynamicGlobalPose", strides, sizeof(DirectApiParams));
        getPose.dispatch(grid, { poseBuf, idxBuf, poolBuf, prevBuf }, params);
        dev->flush();
        const std::vector<float> out = downloadBuffer<float>(dev, poseBuf, nbElements * 7);
        for (uint32_t i = 0; i < nbElements; ++i)
        {
            const BodySim& b = pool[indices[i]];
            const QuatV qW(b.body2World.q), qA(b.body2Actor_maxImpulseW.q);
            const Vec3 pW(b.body2World.p.x, b.body2World.p.y, b.body2World.p.z);
            const Vec3 pA(b.body2Actor_maxImpulseW.p.x, b.body2Actor_maxImpulseW.p.y, b.body2Actor_maxImpulseW.p.z);
            const QuatV q = qW * qA.getConjugate();
            const Vec3 p = qW.rotate(qA.rotateInv(-pA)) + pW;
            PXG_CHECK_MSG(almostEqual(out[i * 7 + 0], q.x) && almostEqual(out[i * 7 + 1], q.y) &&
                              almostEqual(out[i * 7 + 2], q.z) && almostEqual(out[i * 7 + 3], q.w) &&
                              almostEqual(out[i * 7 + 4], p.x, 1e-4f, 1e-4f) &&
                              almostEqual(out[i * 7 + 5], p.y, 1e-4f, 1e-4f) &&
                              almostEqual(out[i * 7 + 6], p.z, 1e-4f, 1e-4f),
                          "getGlobalPose[%u]", i);
        }
    }

    {
        Kernel getLin("directApi.spv", "getRigidDynamicLinearVelocity", strides, sizeof(DirectApiParams));
        Kernel getAngAcc("directApi.spv", "getRigidDynamicAngularAcceleration", strides, sizeof(DirectApiParams));

        getLin.dispatch(grid, { vecBuf, idxBuf, poolBuf, prevBuf }, params);
        dev->flush();
        std::vector<float> out = downloadBuffer<float>(dev, vecBuf, nbElements * 3);
        for (uint32_t i = 0; i < nbElements; ++i)
        {
            const Float4& v = pool[indices[i]].linearVelocityXYZ_inverseMassW;
            PXG_CHECK_MSG(out[i * 3] == v.x && out[i * 3 + 1] == v.y && out[i * 3 + 2] == v.z,
                          "getLinearVelocity[%u]", i);
        }

        getAngAcc.dispatch(grid, { vecBuf, idxBuf, poolBuf, prevBuf }, params);
        dev->flush();
        out = downloadBuffer<float>(dev, vecBuf, nbElements * 3);
        for (uint32_t i = 0; i < nbElements; ++i)
        {
            const uint32_t idx = indices[i];
            const Float4& v = pool[idx].angularVelocityXYZ_maxPenBiasW;
            const Float4& pv = prevVel[idx * 2 + 1];
            PXG_CHECK_MSG(almostEqual(out[i * 3], (v.x - pv.x) * oneOverDt) &&
                              almostEqual(out[i * 3 + 1], (v.y - pv.y) * oneOverDt) &&
                              almostEqual(out[i * 3 + 2], (v.z - pv.z) * oneOverDt),
                          "getAngularAcceleration[%u]", i);
        }
    }

    // ---- setters ----
    {
        std::vector<float> newVel(nbElements * 3);
        for (float& f : newVel)
            f = sym(rng);
        dev->upload(vecBuf, newVel.data(), newVel.size() * sizeof(float));

        Kernel setLin("directApi.spv", "setRigidDynamicLinearVelocity", strides, sizeof(DirectApiParams));
        Kernel setTorque("directApi.spv", "setRigidDynamicTorque", strides, sizeof(DirectApiParams));

        setLin.dispatch(grid, { vecBuf, idxBuf, poolBuf, prevBuf }, params);
        dev->barrier();
        setTorque.dispatch(grid, { vecBuf, idxBuf, poolBuf, prevBuf }, params);
        dev->flush();

        const std::vector<BodySim> gpuPool = downloadBuffer<BodySim>(dev, poolBuf, poolSize);
        const std::vector<Float4> gpuPrev = downloadBuffer<Float4>(dev, prevBuf, poolSize * 2);

        for (uint32_t i = 0; i < nbElements; ++i)
        {
            const uint32_t idx = indices[i];
            const BodySim& g = gpuPool[idx];
            const Vec3 nv(newVel[i * 3], newVel[i * 3 + 1], newVel[i * 3 + 2]);

            PXG_CHECK_MSG(g.linearVelocityXYZ_inverseMassW.x == nv.x &&
                              g.linearVelocityXYZ_inverseMassW.y == nv.y &&
                              g.linearVelocityXYZ_inverseMassW.z == nv.z &&
                              g.linearVelocityXYZ_inverseMassW.w == pool[idx].linearVelocityXYZ_inverseMassW.w,
                          "setLinearVelocity[%u]", i);
            PXG_CHECK_MSG(gpuPrev[idx * 2].x == nv.x && gpuPrev[idx * 2].y == nv.y && gpuPrev[idx * 2].z == nv.z,
                          "setLinearVelocity prev[%u]", i);

            // torque -> world-space inv inertia * torque
            const Float4& invI = pool[idx].inverseInertiaXYZ_contactReportThresholdW;
            const Mat33 rot(QuatV(pool[idx].body2World.q));
            const Mat33 invIWorld = rot * Mat33(Vec3(invI.x, 0, 0), Vec3(0, invI.y, 0), Vec3(0, 0, invI.z)) * rot.getTranspose();
            const Vec3 expected = invIWorld.transform(nv);
            PXG_CHECK_MSG(almostEqual(g.externalAngularAcceleration.x, expected.x, 1e-4f, 1e-5f) &&
                              almostEqual(g.externalAngularAcceleration.y, expected.y, 1e-4f, 1e-5f) &&
                              almostEqual(g.externalAngularAcceleration.z, expected.z, 1e-4f, 1e-5f),
                          "setTorque[%u]: gpu=(%g,%g,%g) ref=(%g,%g,%g)", i, g.externalAngularAcceleration.x,
                          g.externalAngularAcceleration.y, g.externalAngularAcceleration.z, expected.x,
                          expected.y, expected.z);
        }
    }

    dev->destroyBuffer(poolBuf);
    dev->destroyBuffer(idxBuf);
    dev->destroyBuffer(prevBuf);
    dev->destroyBuffer(vecBuf);
    dev->destroyBuffer(poseBuf);
}

} // namespace
