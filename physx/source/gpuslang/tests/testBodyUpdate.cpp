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
//
// updateBodiesLaunch and updateShapesLaunch are verified on the ENGINE
// launch ABI (P5 Step 4): the tests fill the REAL engine PxgNewBodiesDesc /
// raw CUDA parameter lists with live device addresses, marshal the push blob
// with the bridge's natural-alignment rule and launch through PxgCudaShim
// with the engine's grid geometry
// (PxgSimulationCoreKernelGridDim/BlockDim::UPDATE_BODIES_AND_SHAPES =
// 64 x 256).  The remaining entries stay on the bound-buffer harness.

#include "pxgTestFramework.h"
#include "pxgGpuStructs.h"
#include "pxgRefMath.h"
#include "PxgCudaShim.h"

#include "PxNodeIndex.h" // completes PxNodeIndex for the engine desc header
#include "PxgSimulationCoreDesc.h"
#include "PxgBodySim.h"          // P9.93: PxgBodySimVelocityUpdate / external-acceleration offsets

#include <algorithm>
#include <cstddef>

#include <numeric>
#include <random>

namespace
{

using namespace pxgtest;

// Mirrors the bridge's marshalling: place each argument at the natural
// alignment inferred from its size, then round the blob to the largest
// alignment seen (a compiled struct's trailing padding).
struct EngineBlob
{
    std::vector<uint8_t> bytes;
    size_t maxAlign = 1;

    static size_t alignFor(size_t size)
    {
        if (size >= 16) return 16;
        if (size >= 8)  return 8;
        if (size >= 4)  return 4;
        if (size >= 2)  return 2;
        return 1;
    }
    template <typename T>
    void add(const T& v)
    {
        const size_t a = alignFor(sizeof(T));
        if (a > maxAlign) maxAlign = a;
        bytes.resize((bytes.size() + (a - 1)) & ~(a - 1));
        const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
        bytes.insert(bytes.end(), p, p + sizeof(T));
    }
    void finish() { bytes.resize((bytes.size() + (maxAlign - 1)) & ~(maxAlign - 1)); }
};

static std::vector<uint8_t> bodyUpdateAbiLoader(const char* file) { return loadShaderBlob(file); }

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
// updateBodiesLaunch — ENGINE launch ABI ({8}: one PxgNewBodiesDesc pointer)
// ---------------------------------------------------------------------------
// The REAL engine PxgNewBodiesDesc is filled with live device addresses and
// the kernel is launched with the engine's 64 x 256 grid.  nbNew = 2501 is
// odd, spans 157 partially-filled 16-thread groups, and exceeds the
// 64*256/16 = 1024 bodies one grid pass covers, so the CUDA grid-stride
// `i += (blockDim.x * gridDim.x) / 16` wrap executes.  Scattered pool is
// compared bit-exactly (0xCC background pins untouched entries).

PXG_TEST(bodyUpdate_updateBodiesEngineAbi)
{
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    std::mt19937 rng(2000);

    const uint32_t poolSize = 3000;
    const uint32_t nbNew = 2501;

    // shuffled target indices (unique, so the scatter is race-free)
    std::vector<uint32_t> targets(poolSize);
    std::iota(targets.begin(), targets.end(), 0u);
    std::shuffle(targets.begin(), targets.end(), rng);

    std::vector<BodySim> pool(poolSize);
    std::memset(pool.data(), 0xCC, pool.size() * sizeof(BodySim));
    std::vector<BodySim> staged(nbNew);
    for (uint32_t i = 0; i < nbNew; ++i)
        staged[i] = randomBodySim(rng, targets[i]);

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "bodyUpdate.spv", "updateBodiesLaunch", 8u, 0 }, // { u64 }
    };
    PxgCudaShim shim(dev, manifest, 1, &bodyUpdateAbiLoader);

    PxgDevicePtr dStaged = 0, dPool = 0, dDesc = 0;
    shim.memAlloc(&dStaged, staged.size() * sizeof(BodySim));
    shim.memAlloc(&dPool, pool.size() * sizeof(BodySim));
    shim.memAlloc(&dDesc, sizeof(physx::PxgNewBodiesDesc));
    if (!dStaged || !dPool || !dDesc)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    shim.memcpyHtoD(dStaged, staged.data(), staged.size() * sizeof(BodySim));
    shim.memcpyHtoD(dPool, pool.data(), pool.size() * sizeof(BodySim));

    // the REAL engine desc with live device addresses
    physx::PxgNewBodiesDesc desc;
    std::memset(&desc, 0, sizeof(desc));
    desc.mNewBodySim = reinterpret_cast<const physx::PxgBodySim*>(dStaged);
    desc.mBodySimBufferDeviceData = reinterpret_cast<physx::PxgBodySim*>(dPool);
    desc.mNbNewBodies = nbNew;
    shim.memcpyHtoD(dDesc, &desc, sizeof(desc));

    // the CUDA argument list, packed by the bridge's rule ({8})
    EngineBlob blob;
    blob.add<uint64_t>(dDesc);
    blob.finish();
    PXG_CHECK_MSG(blob.bytes.size() == 8, "engine blob is 8 bytes (got %zu)", blob.bytes.size());

    // engine grid: UPDATE_BODIES_AND_SHAPES = 64 blocks x 256 threads
    PXG_CHECK_MSG(shim.launchKernel(0, 64, 1, 1, blob.bytes.data(), uint32_t(blob.bytes.size())) ==
                      PxgResult::Success,
                  "launch updateBodiesLaunch through engine-ABI blob");
    shim.streamSynchronize();

    std::vector<BodySim> gpuPool(poolSize);
    shim.memcpyDtoH(gpuPool.data(), dPool, poolSize * sizeof(BodySim));

    std::vector<BodySim> refPool = pool;
    for (uint32_t i = 0; i < nbNew; ++i)
        refPool[targets[i]] = staged[i];

    PXG_CHECK_MSG(std::memcmp(gpuPool.data(), refPool.data(), poolSize * sizeof(BodySim)) == 0,
                  "updateBodiesLaunch pool mismatch");

    shim.memFree(dStaged);
    shim.memFree(dPool);
    shim.memFree(dDesc);
}

// ---------------------------------------------------------------------------
// updateShapesLaunch — ENGINE launch ABI ({8,8,4}: two element pointers + count)
// ---------------------------------------------------------------------------
// Field-by-field scatter semantics from the .cu: everything except the 4
// padding bytes at offset 52 is copied.  The staged records carry a poison
// pad word and the destination pool a 0xCD background, so the bit-exact
// memcmp catches both a pad copy and a missed field.  Launched with the
// engine's 64 x 256 grid, then again with a 2-block grid on a fresh pool to
// execute the grid-stride wrap.

PXG_TEST(bodyUpdate_updateShapesEngineAbi)
{
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    std::mt19937 rng(2010);
    std::uniform_real_distribution<float> sym(-5.f, 5.f);

    struct NewShapeSim
    {
        ShapeSim sim;
        uint32_t elementIndex;
        uint32_t pad;
    };
    static_assert(sizeof(NewShapeSim) == 80, "PxgNewShapeSim mirror must be 80 bytes");

    const uint32_t shapePool = 900, nbNewShapes = 701;
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
        s.sim.pad0 = 0xAAAAAAAAu; // poison: the kernel must NOT copy this
        s.sim.bodySimIndex = makeNodeIndex(rng() % 64);
        s.sim.hullDataIndex = rng();
        s.sim.shapeFlags = uint16_t(rng() & 7);
        s.sim.shapeType = uint16_t(rng() % 8);
        s.elementIndex = shapeTargets[i];
        s.pad = 0xBBBBBBBBu;
    }

    std::vector<ShapeSim> shapePoolData(shapePool);
    std::memset(shapePoolData.data(), 0xCD, shapePoolData.size() * sizeof(ShapeSim));

    // reference: the .cu copies mTransform, mLocalBounds, mBodySimIndex,
    // mHullDataIndex, mShapeFlags, mShapeType — pad0 keeps the pool background
    std::vector<ShapeSim> refShapes = shapePoolData;
    for (uint32_t i = 0; i < nbNewShapes; ++i)
    {
        ShapeSim& dst = refShapes[newShapes[i].elementIndex];
        const ShapeSim& src = newShapes[i].sim;
        dst.transform = src.transform;
        dst.localBoundsMin = src.localBoundsMin;
        dst.localBoundsMax = src.localBoundsMax;
        dst.bodySimIndex = src.bodySimIndex;
        dst.hullDataIndex = src.hullDataIndex;
        dst.shapeFlags = src.shapeFlags;
        dst.shapeType = src.shapeType;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "bodyUpdate.spv", "updateShapesLaunch", 24u, 0 }, // { u64, u64, u32 } -> 24
    };
    PxgCudaShim shim(dev, manifest, 1, &bodyUpdateAbiLoader);

    PxgDevicePtr dNew = 0, dPool = 0;
    shim.memAlloc(&dNew, newShapes.size() * sizeof(NewShapeSim));
    shim.memAlloc(&dPool, shapePoolData.size() * sizeof(ShapeSim));
    if (!dNew || !dPool)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    shim.memcpyHtoD(dNew, newShapes.data(), newShapes.size() * sizeof(NewShapeSim));

    // the CUDA argument list, packed by the bridge's rule ({8,8,4} -> 24)
    EngineBlob blob;
    blob.add<uint64_t>(dNew);
    blob.add<uint64_t>(dPool);
    blob.add<uint32_t>(nbNewShapes);
    blob.finish();
    PXG_CHECK_MSG(blob.bytes.size() == 24, "engine blob is 24 bytes (got %zu)", blob.bytes.size());

    // engine grid (64 x 256, single pass) and a 2-block grid (512 threads ->
    // the grid-stride wrap covers the remaining shapes)
    for (uint32_t gridDim : { 64u, 2u })
    {
        shim.memcpyHtoD(dPool, shapePoolData.data(), shapePoolData.size() * sizeof(ShapeSim));

        PXG_CHECK_MSG(shim.launchKernel(0, gridDim, 1, 1, blob.bytes.data(), uint32_t(blob.bytes.size())) ==
                          PxgResult::Success,
                      "launch updateShapesLaunch through engine-ABI blob (grid %u)", gridDim);
        shim.streamSynchronize();

        std::vector<ShapeSim> gpuShapes(shapePool);
        shim.memcpyDtoH(gpuShapes.data(), dPool, shapePool * sizeof(ShapeSim));

        PXG_CHECK_MSG(std::memcmp(gpuShapes.data(), refShapes.data(), shapePool * sizeof(ShapeSim)) == 0,
                      "updateShapesLaunch pool mismatch (grid %u)", gridDim);
    }

    shim.memFree(dNew);
    shim.memFree(dPool);
}

// P9.63 -- THE FIXTURE MUST DISPATCH THE ENGINE'S GRID, NOT ONE SIZED TO ITS DATA.
// PxgSimulationCore launches updateBodiesLaunchDirectAPI (UPDATE_BODIES_AND_SHAPES)
// and updateBodyExternalVelocitiesLaunch (UPDATE_BODY_EXTERNAL_VELOCITIES) with a
// FIXED grid of 64 groups of 256, and the bridge drops blockDim, so a shader that
// does not loop reaches at most 64*256 records however many the engine hands it.
// These fixtures used to dispatch ceil(count/256) -- their own divisor, sized so
// that one thread always existed per record -- which made the grid-stride loop
// unnecessary and its absence invisible.  Both entries were in fact one-pass ports
// (fixed in P9.63); each test now runs the engine's grid at two sizes, the second
// past a full pass, so a record beyond the ceiling is checked like any other.
static const uint32_t BODY_UPDATE_GRID = 64;
static const uint32_t BODY_UPDATE_PASS = BODY_UPDATE_GRID * 256; // 16384 records

struct SizePair { uint32_t poolSize; uint32_t count; };
static const SizePair SIZE_CASES[2] = { { 150u, 60u },
                                        { BODY_UPDATE_PASS + 3616u, BODY_UPDATE_PASS + 1393u } };

// P9.93 -- updateBodyExternalVelocitiesLaunch on the ENGINE launch ABI ({8}: one
// PxgUpdatedBodiesDesc pointer), launched through PxgCudaShim with the engine's
// fixed 64-group grid, at the two SIZE_CASES (the second past one grid pass).
//
// What the records carry, and why:
//   * linVel.w is the TARGET BODY INDEX as raw bits (PX_FR in PxgBodySimManager).
//     Targets are 1..poolSize-1 shuffled, so small indices are DENORMAL floats: a
//     port that ever moved the word through a float with flush-to-zero, or through
//     a float-to-int conversion, would scatter them onto body 0 -- which is never a
//     target and must stay untouched.
//   * angVel.w and BOTH external accelerations' .w are non-zero: the kernel copies
//     all four lanes of each (updateBodiesAndShapes.cu:923-926), and the engine's
//     own producer writing w = 0 there must not let a three-lane copy pass.
//   * linVel.w of the POOL survives (the inverse mass).
// The whole pool, the records and the descriptor are compared bit for bit.
static_assert(sizeof(physx::PxgUpdatedBodiesDesc) == 24, "PxgUpdatedBodiesDesc size");
static_assert(offsetof(physx::PxgUpdatedBodiesDesc, mBodySimBufferDeviceData) == 0, "desc @0");
static_assert(offsetof(physx::PxgUpdatedBodiesDesc, mUpdatedBodySim) == 8, "desc @8");
static_assert(offsetof(physx::PxgUpdatedBodiesDesc, mNbUpdatedBodies) == 16, "desc @16");
static_assert(sizeof(physx::PxgBodySimVelocityUpdate) == 64, "PxgBodySimVelocityUpdate size");
static_assert(offsetof(physx::PxgBodySimVelocityUpdate, linearVelocityXYZ_bodySimIndexW) == 0, "upd @0");
static_assert(offsetof(physx::PxgBodySimVelocityUpdate, angularVelocityXYZ_maxPenBiasW) == 16, "upd @16");
static_assert(offsetof(physx::PxgBodySimVelocityUpdate, externalLinearAccelerationXYZ) == 32, "upd @32");
static_assert(offsetof(physx::PxgBodySimVelocityUpdate, externalAngularAccelerationXYZ) == 48, "upd @48");
static_assert(offsetof(physx::PxgBodySim, externalLinearAcceleration) == 192, "PxgBodySim @192");
static_assert(offsetof(physx::PxgBodySim, externalAngularAcceleration) == 208, "PxgBodySim @208");

PXG_TEST(bodyUpdate_externalVelocities)
{
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "bodyUpdate.spv", "updateBodyExternalVelocitiesLaunch", 8u, 0 }, // { u64 }
    };
    PxgCudaShim shim(dev, manifest, 1, &bodyUpdateAbiLoader);

    for (const SizePair sizes : SIZE_CASES)
    {
        std::mt19937 rng(2001);
        std::uniform_real_distribution<float> sym(-5.f, 5.f);
        auto nonZero = [&]() { float v = sym(rng); return v == 0.f ? 1.5f : v; };

        const uint32_t poolSize = sizes.poolSize;
        const uint32_t nbUpdates = sizes.count;

        std::vector<BodySim> pool(poolSize);
        for (uint32_t i = 0; i < poolSize; ++i)
            pool[i] = randomBodySim(rng, i);

        // body 0 is never a target (see the header)
        std::vector<uint32_t> targets(poolSize - 1);
        std::iota(targets.begin(), targets.end(), 1u);
        std::shuffle(targets.begin(), targets.end(), rng);
        // ...but the first records DO carry the smallest (denormal) indices
        for (uint32_t k = 0; k < 4; ++k)
            std::swap(targets[k], *std::find(targets.begin(), targets.end(), k + 1u));

        // PxgBodySimVelocityUpdate: 4 float4
        std::vector<Float4> updates(nbUpdates * 4);
        for (uint32_t i = 0; i < nbUpdates; ++i)
        {
            Float4 lin = { sym(rng), sym(rng), sym(rng), 0.f };
            std::memcpy(&lin.w, &targets[i], 4);
            updates[i * 4 + 0] = lin;
            updates[i * 4 + 1] = Float4{ sym(rng), sym(rng), sym(rng), nonZero() };
            updates[i * 4 + 2] = Float4{ sym(rng), sym(rng), sym(rng), nonZero() };
            updates[i * 4 + 3] = Float4{ sym(rng), sym(rng), sym(rng), nonZero() };
        }

        PxgDevicePtr dUpdates = 0, dPool = 0, dDesc = 0;
        shim.memAlloc(&dUpdates, updates.size() * sizeof(Float4));
        shim.memAlloc(&dPool, pool.size() * sizeof(BodySim));
        shim.memAlloc(&dDesc, sizeof(physx::PxgUpdatedBodiesDesc));
        if (!dUpdates || !dPool || !dDesc)
        {
            std::printf("  NOTE: no BDA -> skip\n");
            return;
        }
        shim.memcpyHtoD(dUpdates, updates.data(), updates.size() * sizeof(Float4));
        shim.memcpyHtoD(dPool, pool.data(), pool.size() * sizeof(BodySim));

        // the REAL engine desc -- pool FIRST, the reverse of PxgNewBodiesDesc
        physx::PxgUpdatedBodiesDesc desc;
        std::memset(&desc, 0xEE, sizeof(desc));   // the 4 pad bytes must survive
        desc.mBodySimBufferDeviceData = reinterpret_cast<physx::PxgBodySim*>(dPool);
        desc.mUpdatedBodySim = reinterpret_cast<physx::PxgBodySimVelocityUpdate*>(dUpdates);
        desc.mNbUpdatedBodies = nbUpdates;
        shim.memcpyHtoD(dDesc, &desc, sizeof(desc));

        EngineBlob blob;
        blob.add<uint64_t>(dDesc);
        blob.finish();
        PXG_CHECK_MSG(blob.bytes.size() == 8, "engine blob is 8 bytes (got %zu)", blob.bytes.size());

        // engine grid: UPDATE_BODY_EXTERNAL_VELOCITIES = 64 blocks x 256 threads
        PXG_CHECK_MSG(shim.launchKernel(0, BODY_UPDATE_GRID, 1, 1, blob.bytes.data(),
                                        uint32_t(blob.bytes.size())) == PxgResult::Success,
                      "launch updateBodyExternalVelocitiesLaunch through the engine-ABI blob");
        shim.streamSynchronize();

        std::vector<BodySim> gpuPool(poolSize);
        shim.memcpyDtoH(gpuPool.data(), dPool, poolSize * sizeof(BodySim));
        std::vector<Float4> gpuUpdates(updates.size());
        shim.memcpyDtoH(gpuUpdates.data(), dUpdates, updates.size() * sizeof(Float4));
        physx::PxgUpdatedBodiesDesc gpuDesc;
        shim.memcpyDtoH(&gpuDesc, dDesc, sizeof(gpuDesc));

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
                      "external velocity update mismatch (count %u)", nbUpdates);
        PXG_CHECK_MSG(std::memcmp(&gpuPool[0], &pool[0], sizeof(BodySim)) == 0,
                      "body 0 is never a target and must be untouched");
        PXG_CHECK_MSG(std::memcmp(gpuUpdates.data(), updates.data(), updates.size() * sizeof(Float4)) == 0,
                      "the update records are read-only");
        PXG_CHECK_MSG(std::memcmp(&gpuDesc, &desc, sizeof(desc)) == 0, "the descriptor is read-only");

        shim.memFree(dUpdates);
        shim.memFree(dPool);
        shim.memFree(dDesc);
    }
}

// ---------------------------------------------------------------------------
// updateBodiesLaunchDirectAPI
// ---------------------------------------------------------------------------

PXG_TEST(bodyUpdate_directAPI)
{
    for (const SizePair sizes : SIZE_CASES)
    {
        std::mt19937 rng(2002);
        std::uniform_real_distribution<float> unit(0.f, 1.f);

        const uint32_t poolSize = sizes.poolSize;
        const uint32_t nbNew = sizes.count;

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

        kernel.dispatch(BODY_UPDATE_GRID, { stagedBuf, poolBuf, dummyShapes }, BodyUpdateParams{ nbNew });
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
}

// ---------------------------------------------------------------------------
// P9.90: updateBodiesLaunchDirectAPI's body2World recompute, BITWISE
// ---------------------------------------------------------------------------
// The CUDA re-derives body2World when the CPU changed body2Actor
// (updateBodiesAndShapes.cu, updateBodiesLaunchDirectAPI):
//   actor2World = body2WorldOld * oldBody2Actor.getInverse();
//   body2World  = actor2World * newBody2Actor;
// i.e. PxAlignedQuat::rotateInv / rotate (AlignedQuat.h) and the Hamilton
// product -- mul+add chains the CUDA evaluates left to right, and the mirror
// here (built with -ffp-contract=off) does too.  bodyUpdate is stamped
// NoContraction (tests/CMakeLists.txt, P9.90) so the Vulkan side keeps that
// order; bodyUpdate_directAPI above only compares with a tolerance.  Every
// record here takes the recompute path (no first transfer, body2Actor
// changed), with general orientations and non-round positions, and the
// resulting body2World (q and p.xyzw) is compared bit for bit.
// The shader has no OpDot and llvmpipe does not contract its scalar chains, so
// an unstamped build passes too (measured): the stamp is unobservable on
// llvmpipe.  The case still pins the order -- a shader mutant with
// dot2 = x*vx + (y*vy + z*vz) in quatRotate fails it (1555/4096 poses).
namespace
{
// PxAlignedQuat::rotate / rotateInv in the CUDA order.  oldDot selects the
// re-associated dot2 = x*vx + (y*vy + z*vz) -- the twin that must differ.
Vec3 p90Rotate(const QuatV& q, const Vec3& v, bool inv, bool oldDot)
{
    const float vx = 2.0f * v.x;
    const float vy = 2.0f * v.y;
    const float vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = oldDot ? (q.x * vx + (q.y * vy + q.z * vz)) : (q.x * vx + q.y * vy + q.z * vz);
    const float s = inv ? -1.0f : 1.0f;
    // s * t is exact (sign flip), so vx*w2 + s*t == vx*w2 -/+ t bitwise
    return Vec3((vx * w2 + s * ((q.y * vz - q.z * vy) * q.w) + q.x * dot2),
                (vy * w2 + s * ((q.z * vx - q.x * vz) * q.w) + q.y * dot2),
                (vz * w2 + s * ((q.x * vy - q.y * vx) * q.w) + q.z * dot2));
}

AlignedTransform p90Recompute(const AlignedTransform& b2wOld, const AlignedTransform& oldB2A,
                              const AlignedTransform& newB2A, bool oldDot)
{
    const QuatV qW(b2wOld.q), qA(oldB2A.q), qN(newB2A.q);
    // getInverse(): q.rotateInv(make_float4(-p)) (w lane 0), q.getConjugate()
    const Vec3 pInv = p90Rotate(qA, Vec3(-oldB2A.p.x, -oldB2A.p.y, -oldB2A.p.z), true, oldDot);
    const QuatV qInv = qA.getConjugate();
    // transform(src): q.rotate(src.p) + p (float4 add: w = 0 + p.w), q * src.q
    const QuatV qA2W = qW * qInv;
    const Vec3 rA = p90Rotate(qW, pInv, false, oldDot);
    const Float4 pA2W{ rA.x + b2wOld.p.x, rA.y + b2wOld.p.y, rA.z + b2wOld.p.z, 0.0f + b2wOld.p.w };
    const QuatV qNew = qA2W * qN;
    const Vec3 rN = p90Rotate(qA2W, Vec3(newB2A.p.x, newB2A.p.y, newB2A.p.z), false, oldDot);
    AlignedTransform r;
    r.q = qNew.toFloat4();
    r.p = Float4{ rN.x + pA2W.x, rN.y + pA2W.y, rN.z + pA2W.z, 0.0f + pA2W.w };
    return r;
}
} // namespace

PXG_TEST(bodyUpdate_p90PoseRecomputeOrder)
{
    const uint32_t N = 4096;
    std::mt19937 rng(9090);
    std::uniform_real_distribution<float> sym(-7.3f, 7.3f);

    std::vector<BodySim> pool(N), staged(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        pool[i] = randomBodySim(rng, i);
        pool[i].body2World.p.w = 0.37f + 0.001f * float(i); // non-zero w lane rides through 0 + w
        pool[i].body2Actor_maxImpulseW.p = Float4{ sym(rng), sym(rng), sym(rng), 1.3f };
        staged[i] = randomBodySim(rng, i);
        staged[i].internalFlags = (i & 1) ? eVELOCITY_COPY_GPU : 0u; // never eFIRST_BODY_COPY_GPU
        staged[i].body2Actor_maxImpulseW.p = Float4{ sym(rng), sym(rng), sym(rng), 2.9f };
    }

    std::vector<AlignedTransform> ref(N), twin(N);
    uint32_t twinDiffers = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        ref[i] = p90Recompute(pool[i].body2World, pool[i].body2Actor_maxImpulseW, staged[i].body2Actor_maxImpulseW, false);
        twin[i] = p90Recompute(pool[i].body2World, pool[i].body2Actor_maxImpulseW, staged[i].body2Actor_maxImpulseW, true);
        if (std::memcmp(&ref[i], &twin[i], sizeof(AlignedTransform)) != 0)
            ++twinDiffers;
    }
    // coverage guard: the data must separate the CUDA dot order from x+(y+z)
    PXG_CHECK_MSG(twinDiffers > N / 4, "p90 twin: old-order dot2 changes only %u of %u poses", twinDiffers, N);

    PxgSlangComputeDevice* dev = testDevice();
    Kernel kernel("bodyUpdate.spv", "updateBodiesLaunchDirectAPI", { 4, 4, 4 }, sizeof(BodyUpdateParams));
    PxgBufferHandle stagedBuf = createBufferFrom(dev, staged);
    PxgBufferHandle poolBuf = createBufferFrom(dev, pool);
    PxgBufferHandle dummyShapes = dev->createBuffer(16);
    kernel.dispatch(BODY_UPDATE_GRID, { stagedBuf, poolBuf, dummyShapes }, BodyUpdateParams{ N });
    dev->flush();
    const std::vector<BodySim> gpuPool = downloadBuffer<BodySim>(dev, poolBuf, N);

    uint32_t mismatches = 0, equalsTwin = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const AlignedTransform& g = gpuPool[i].body2World;
        if (std::memcmp(&g, &ref[i], sizeof(AlignedTransform)) != 0)
        {
            if (std::memcmp(&g, &twin[i], sizeof(AlignedTransform)) == 0)
                ++equalsTwin;
            if (mismatches < 5)
                PXG_CHECK_MSG(false, "p90 body2World[%u]: gpu q=(%a %a %a %a) p=(%a %a %a %a) ref q=(%a %a %a %a) p=(%a %a %a %a)",
                              i, g.q.x, g.q.y, g.q.z, g.q.w, g.p.x, g.p.y, g.p.z, g.p.w,
                              ref[i].q.x, ref[i].q.y, ref[i].q.z, ref[i].q.w,
                              ref[i].p.x, ref[i].p.y, ref[i].p.z, ref[i].p.w);
            ++mismatches;
        }
    }
    PXG_CHECK_MSG(mismatches == 0, "p90 body2World: %u of %u poses differ bitwise (%u equal the old-order twin)",
                  mismatches, N, equalsTwin);

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

// ---------------------------------------------------------------------------
// P9.90: directApi's float chains, BITWISE
// ---------------------------------------------------------------------------
// directApi_gettersAndSetters above compares the pose and torque with a
// tolerance.  Here the two entries that carry mul+add chains are compared bit
// for bit against a -ffp-contract=off transcription in the CUDA order
// (updateBodiesAndShapes.cu):
//   getRigidDynamicGlobalPose:  body2World * body2Actor.getInverse()
//     -- PxAlignedQuat rotateInv/rotate and the Hamilton product
//        (AlignedQuat.h:257-366, AlignedTransform.h:124-186);
//   setRigidDynamicTorque:  Cm::transformInertiaTensor(invI, PxMat33(q))
//     (CmUtils.h:57-70, PxMat33.h:136-162) then PxMat33::transform
//     column0*t.x + column1*t.y + column2*t.z (PxMat33.h:344-347).
// Every chain is (a + b) + c, left to right.  directApi is stamped NoContraction
// (tests/CMakeLists.txt, P9.90) so the Vulkan side keeps that order.  The shader
// has no OpDot, so an unstamped build passes too on llvmpipe (measured): the
// stamp is unobservable there.  The old-order twin (every three-term chain
// summed a + (b + c)) must differ, so the case does pin the association: a
// shader mutant with dot2 = x*vx + (y*vy + z*vz) in quatRotate/quatRotateInv and
// c0*x + (c1*y + c2*z) in mat33Transform fails it (1691/4096 poses, 2509/4096
// torques).
namespace
{
inline float p90Sum3(float a, float b, float c, bool oldOrder) { return oldOrder ? a + (b + c) : a + b + c; }

void p90DirectPose(const BodySim& b, bool oldOrder, float out[7])
{
    const QuatV qW(b.body2World.q), qA(b.body2Actor_maxImpulseW.q);
    const Float4& pA = b.body2Actor_maxImpulseW.p;
    // getInverse(): q.rotateInv(-p), q.getConjugate()
    const Vec3 pInv = p90Rotate(qA, Vec3(-pA.x, -pA.y, -pA.z), true, oldOrder);
    const QuatV qInv = qA.getConjugate();
    // transform(src): q.rotate(src.p) + p, q * src.q
    const QuatV q = qW * qInv;
    const Vec3 r = p90Rotate(qW, pInv, false, oldOrder);
    out[0] = q.x; out[1] = q.y; out[2] = q.z; out[3] = q.w;
    out[4] = r.x + b.body2World.p.x;
    out[5] = r.y + b.body2World.p.y;
    out[6] = r.z + b.body2World.p.z;
}

Vec3 p90DirectTorque(const BodySim& b, const Vec3& t, bool oldOrder)
{
    const Float4& invD = b.inverseInertiaXYZ_contactReportThresholdW;
    const Mat33 M(QuatV(b.body2World.q)); // PxMat33(PxQuat)
    // M(r, c): column c, row r
    auto at = [&](int r, int c) {
        const Vec3& col = c == 0 ? M.col0 : (c == 1 ? M.col1 : M.col2);
        return r == 0 ? col.x : (r == 1 ? col.y : col.z);
    };
    const float axx = invD.x * at(0, 0), axy = invD.x * at(1, 0), axz = invD.x * at(2, 0);
    const float byx = invD.y * at(0, 1), byy = invD.y * at(1, 1), byz = invD.y * at(2, 1);
    const float czx = invD.z * at(0, 2), czy = invD.z * at(1, 2), czz = invD.z * at(2, 2);
    const float m00 = p90Sum3(axx * at(0, 0), byx * at(0, 1), czx * at(0, 2), oldOrder);
    const float m11 = p90Sum3(axy * at(1, 0), byy * at(1, 1), czy * at(1, 2), oldOrder);
    const float m22 = p90Sum3(axz * at(2, 0), byz * at(2, 1), czz * at(2, 2), oldOrder);
    const float m01 = p90Sum3(axx * at(1, 0), byx * at(1, 1), czx * at(1, 2), oldOrder);
    const float m02 = p90Sum3(axx * at(2, 0), byx * at(2, 1), czx * at(2, 2), oldOrder);
    const float m12 = p90Sum3(axy * at(2, 0), byy * at(2, 1), czy * at(2, 2), oldOrder);
    // columns (m00 m01 m02), (m01 m11 m12), (m02 m12 m22); transform = c0*t.x + c1*t.y + c2*t.z
    return Vec3(p90Sum3(m00 * t.x, m01 * t.y, m02 * t.z, oldOrder),
                p90Sum3(m01 * t.x, m11 * t.y, m12 * t.z, oldOrder),
                p90Sum3(m02 * t.x, m12 * t.y, m22 * t.z, oldOrder));
}
} // namespace

PXG_TEST(directApi_p90PoseTorqueOrder)
{
    const uint32_t N = 4096;
    std::mt19937 rng(9091);
    std::uniform_real_distribution<float> sym(-7.3f, 7.3f);
    std::uniform_real_distribution<float> inertia(0.013f, 3.7f);

    std::vector<BodySim> pool(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        pool[i] = randomBodySim(rng, i);
        pool[i].body2Actor_maxImpulseW.p = Float4{ sym(rng), sym(rng), sym(rng), 1.3f };
        pool[i].inverseInertiaXYZ_contactReportThresholdW = Float4{ inertia(rng), inertia(rng), inertia(rng), 0.5f };
    }
    std::vector<uint32_t> indices(N);
    std::iota(indices.begin(), indices.end(), 0u);
    std::shuffle(indices.begin(), indices.end(), rng);
    std::vector<float> torque(N * 3);
    for (float& f : torque)
        f = sym(rng);
    std::vector<Float4> prevVel(N * 2, Float4{ 0.f, 0.f, 0.f, 0.f });

    // CPU references and the old-order twins
    std::vector<float> refPose(N * 7), twinPose(N * 7);
    std::vector<Vec3> refAng(N), twinAng(N);
    uint32_t poseTwinDiffers = 0, torqueTwinDiffers = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const BodySim& b = pool[indices[i]];
        p90DirectPose(b, false, &refPose[i * 7]);
        p90DirectPose(b, true, &twinPose[i * 7]);
        if (std::memcmp(&refPose[i * 7], &twinPose[i * 7], 7 * sizeof(float)) != 0)
            ++poseTwinDiffers;
        const Vec3 t(torque[i * 3], torque[i * 3 + 1], torque[i * 3 + 2]);
        refAng[i] = p90DirectTorque(b, t, false);
        twinAng[i] = p90DirectTorque(b, t, true);
        if (std::memcmp(&refAng[i], &twinAng[i], sizeof(Vec3)) != 0)
            ++torqueTwinDiffers;
    }
    // coverage guards: the data must separate the CUDA order from a + (b + c)
    PXG_CHECK_MSG(poseTwinDiffers > N / 4, "p90 pose twin: old order changes only %u of %u poses", poseTwinDiffers, N);
    PXG_CHECK_MSG(torqueTwinDiffers > N / 4, "p90 torque twin: old order changes only %u of %u rows",
                  torqueTwinDiffers, N);

    PxgSlangComputeDevice* dev = testDevice();
    const std::vector<uint32_t> strides = { 4, 4, 4, 16 };
    PxgBufferHandle poolBuf = createBufferFrom(dev, pool);
    PxgBufferHandle idxBuf = createBufferFrom(dev, indices);
    PxgBufferHandle prevBuf = createBufferFrom(dev, prevVel);
    PxgBufferHandle poseBuf = dev->createBuffer(N * 7 * sizeof(float));
    PxgBufferHandle vecBuf = createBufferFrom(dev, torque);
    const DirectApiParams params = { N, 60.f, 0u };
    const uint32_t grid = (N + 255) / 256;

    Kernel getPose("directApi.spv", "getRigidDynamicGlobalPose", strides, sizeof(DirectApiParams));
    getPose.dispatch(grid, { poseBuf, idxBuf, poolBuf, prevBuf }, params);
    dev->barrier();
    Kernel setTorque("directApi.spv", "setRigidDynamicTorque", strides, sizeof(DirectApiParams));
    setTorque.dispatch(grid, { vecBuf, idxBuf, poolBuf, prevBuf }, params);
    dev->flush();
    const std::vector<float> gpuPose = downloadBuffer<float>(dev, poseBuf, N * 7);
    const std::vector<BodySim> gpuPool = downloadBuffer<BodySim>(dev, poolBuf, N);

    uint32_t poseMismatches = 0, poseEqualsTwin = 0;
    uint32_t torqueMismatches = 0, torqueEqualsTwin = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* g = &gpuPose[i * 7];
        if (std::memcmp(g, &refPose[i * 7], 7 * sizeof(float)) != 0)
        {
            if (std::memcmp(g, &twinPose[i * 7], 7 * sizeof(float)) == 0)
                ++poseEqualsTwin;
            if (poseMismatches < 5)
                PXG_CHECK_MSG(false, "p90 pose[%u]: gpu q=(%a %a %a %a) p=(%a %a %a) ref q=(%a %a %a %a) p=(%a %a %a)", i,
                              g[0], g[1], g[2], g[3], g[4], g[5], g[6], refPose[i * 7], refPose[i * 7 + 1],
                              refPose[i * 7 + 2], refPose[i * 7 + 3], refPose[i * 7 + 4], refPose[i * 7 + 5],
                              refPose[i * 7 + 6]);
            ++poseMismatches;
        }
        const Float4& a = gpuPool[indices[i]].externalAngularAcceleration;
        const Vec3 ga(a.x, a.y, a.z);
        if (std::memcmp(&ga, &refAng[i], sizeof(Vec3)) != 0 || a.w != 0.0f)
        {
            if (std::memcmp(&ga, &twinAng[i], sizeof(Vec3)) == 0)
                ++torqueEqualsTwin;
            if (torqueMismatches < 5)
                PXG_CHECK_MSG(false, "p90 torque[%u]: gpu=(%a %a %a %a) ref=(%a %a %a)", i, a.x, a.y, a.z, a.w,
                              refAng[i].x, refAng[i].y, refAng[i].z);
            ++torqueMismatches;
        }
    }
    PXG_CHECK_MSG(poseMismatches == 0, "p90 pose: %u of %u differ bitwise (%u equal the old-order twin)",
                  poseMismatches, N, poseEqualsTwin);
    PXG_CHECK_MSG(torqueMismatches == 0, "p90 torque: %u of %u differ bitwise (%u equal the old-order twin)",
                  torqueMismatches, N, torqueEqualsTwin);

    dev->destroyBuffer(poolBuf);
    dev->destroyBuffer(idxBuf);
    dev->destroyBuffer(prevBuf);
    dev->destroyBuffer(poseBuf);
    dev->destroyBuffer(vecBuf);
}

} // namespace
