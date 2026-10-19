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

// P9.93: the WRITE half of the rigid direct-GPU API and the two acceleration getters
// (directApiBda.slang) on the ENGINE launch ABI.
//
// Every kernel is launched through PxgCudaShim with the push blob
// PxgSimulationCore packs (the structs below are pinned byte for byte) and the
// host's (n + 255) / 256 grid, over device memory laid out as the REAL engine
// structs -- PxgBodySim, PxgBodySimVelocities, PxgUpdateActorDataDesc, PxgShapeSim,
// PxgShape, PxsCachedTransform, PxBounds3 -- each pinned below against its header.
// The oracle is a literal C++ transcription of each CUDA kernel built from the REAL
// PhysX types (PxTransform::transformInv / transform, PxMat33, PxMeshScale::toMat33,
// PxBounds3::transformFast, Cm::transformInertiaTensor), run on host images of the
// same buffers, and EVERY buffer -- inputs, descriptors and decoys included -- is
// compared bit for bit after every launch (poisoned identically on both sides first).
//
//   directApiRigidAccelGet       5.12: computeRigidBodyAccelerations over a pool
//                                with a tail past nbElements, then both gather
//                                getters: a query list with duplicates, a
//                                non-power-of-two 1/dt, vel == prev (+0) and
//                                -0 - +0 (-0), a grid tail with trap indices past it;
//                                then copyRigidBodyVelocitiesToPrevious
//   directApiRigidGetForceTorque 5.12 getRigidDynamicForce / Torque: zero and
//                                non-zero inverse mass / inertia components,
//                                non-axis rotations, duplicate queries
//   directApiRigidSetVel         both velocity setters, with and without the
//                                previous-velocity buffer, -0 / NaN payload / inf
//                                moves, .w of the pool and of prev
//   directApiRigidSetForceTorque force (inverse mass 0 against a negative force,
//                                non-zero prior .w) and torque (anisotropic inverse
//                                inertia under non-axis rotations, zero components)
//   directApiRigidSetPose        the pose setter's full shape walk: every geometry
//                                branch, the backward walk from pos 0 (wrap),
//                                0xFFFFFFFF shape slots first / middle / last in a
//                                run, statics and articulation links in the node
//                                table, a trap entry at [totalNumShapes], the
//                                totalNumShapes == 0 early return AFTER the pose
//                                write, and the two P9.93 spellings: the conjugate-
//                                rotate transformInv (sign of zero) and the
//                                scale-only PxMeshScale::isIdentity

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "foundation/PxVec3.h"
#include "foundation/PxQuat.h"
#include "foundation/PxTransform.h"
#include "foundation/PxMat33.h"
#include "foundation/PxBounds3.h"
#include "geometry/PxMeshScale.h"
#include "PxNodeIndex.h"
#include "PxgSimulationCoreDesc.h"
#include "PxgBodySim.h"
#include "PxgShapeSim.h"
#include "PxgConvexConvexShape.h"
#include "PxsTransformCache.h"
#include "CmUtils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::PxMat33;
using physx::PxQuat;
using physx::PxTransform;
using physx::PxVec3;
using physx::PxBounds3;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

// ---------------------------------------------------------------------------
// layouts, pinned against the engine headers
// ---------------------------------------------------------------------------
static const uint32_t SZ_BODYSIM = 224u;
static const uint32_t BS_LINVEL = 0u, BS_ANGVEL = 16u, BS_INVINERTIA = 48u, BS_B2W = 112u, BS_B2A = 144u,
                      BS_EXTLIN = 192u, BS_EXTANG = 208u;
static_assert(sizeof(physx::PxgBodySim) == SZ_BODYSIM, "PxgBodySim");
static_assert(offsetof(physx::PxgBodySim, linearVelocityXYZ_inverseMassW) == BS_LINVEL, "linVel");
static_assert(offsetof(physx::PxgBodySim, angularVelocityXYZ_maxPenBiasW) == BS_ANGVEL, "angVel");
static_assert(offsetof(physx::PxgBodySim, inverseInertiaXYZ_contactReportThresholdW) == BS_INVINERTIA, "invI");
static_assert(offsetof(physx::PxgBodySim, body2World) == BS_B2W, "body2World");
static_assert(offsetof(physx::PxgBodySim, body2Actor_maxImpulseW) == BS_B2A, "body2Actor");
static_assert(offsetof(physx::PxgBodySim, externalLinearAcceleration) == BS_EXTLIN, "extLin");
static_assert(offsetof(physx::PxgBodySim, externalAngularAcceleration) == BS_EXTANG, "extAng");

static const uint32_t SZ_PREV = 32u;
static_assert(sizeof(physx::PxgBodySimVelocities) == SZ_PREV, "PxgBodySimVelocities");
static_assert(offsetof(physx::PxgBodySimVelocities, linearVelocity) == 0, "prev lin");
static_assert(offsetof(physx::PxgBodySimVelocities, angularVelocity) == 16, "prev ang");

static const uint32_t SZ_DESC = 80u;
static const uint32_t D_BODYSIM = 0, D_SHAPES = 8, D_CACHE = 16, D_BOUNDS = 24, D_NODES = 32, D_SHAPEIDX = 40,
                      D_UPDATED = 48, D_CHANGED = 56, D_WORDS = 64, D_SHAPESIMS = 72;
static_assert(sizeof(physx::PxgUpdateActorDataDesc) == SZ_DESC, "PxgUpdateActorDataDesc");
static_assert(offsetof(physx::PxgUpdateActorDataDesc, mBodySimBufferDeviceData) == D_BODYSIM, "@0");
static_assert(offsetof(physx::PxgUpdateActorDataDesc, mShapes) == D_SHAPES, "@8");
static_assert(offsetof(physx::PxgUpdateActorDataDesc, mTransformCache) == D_CACHE, "@16");
static_assert(offsetof(physx::PxgUpdateActorDataDesc, mBounds) == D_BOUNDS, "@24");
static_assert(offsetof(physx::PxgUpdateActorDataDesc, mRigidNodeIndices) == D_NODES, "@32");
static_assert(offsetof(physx::PxgUpdateActorDataDesc, mShapeIndices) == D_SHAPEIDX, "@40");
static_assert(offsetof(physx::PxgUpdateActorDataDesc, mUpdated) == D_UPDATED, "@48");
static_assert(offsetof(physx::PxgUpdateActorDataDesc, mChangedAABBMgrHandles) == D_CHANGED, "@56");
static_assert(offsetof(physx::PxgUpdateActorDataDesc, mBitMapWordCounts) == D_WORDS, "@64");
static_assert(offsetof(physx::PxgUpdateActorDataDesc, mShapeSimsBufferDeviceData) == D_SHAPESIMS, "@72");

static_assert(sizeof(physx::PxgShapeSim) == 72, "PxgShapeSim");
static_assert(offsetof(physx::PxgShapeSim, mLocalBounds) == 28, "mLocalBounds");
static_assert(offsetof(physx::PxgShapeSim, mHullDataIndex) == 64, "mHullDataIndex");
static_assert(offsetof(physx::PxgShapeSim, mShapeType) == 70, "mShapeType");
static_assert(sizeof(physx::PxgShape) == 48, "PxgShape");
static_assert(offsetof(physx::PxgShape, hullOrMeshPtr) == 32, "hullOrMeshPtr");
static_assert(sizeof(physx::PxsCachedTransform) == 32, "PxsCachedTransform");
static_assert(offsetof(physx::PxsCachedTransform, flags) == 28, "cache flags");
static_assert(sizeof(PxBounds3) == 24, "PxBounds3");
static_assert(sizeof(physx::PxNodeIndex) == 8, "PxNodeIndex");

// PxgRigidBodyAcceleration (PxgSimulationCore.h, 5.12): { PxVec3 linear; pad; PxVec3 angular; pad },
// 16-aligned, 32 bytes.  Not pinned by sizeof: that header drags in the CUDA host side.
static const uint32_t SZ_ACCEL = 32u, RA_LINEAR = 0u, RA_ANGULAR = 16u;

// ---- push blobs, byte for byte the host packs (PxgSimulationCore.cpp) ----
// 5.12 getRigidDynamic{Linear,Angular}Acceleration: a gather from PxgRigidBodyAcceleration
struct AccelPush { uint64_t data, gpuIndices, accelerations; uint32_t nb, pad; };
static_assert(sizeof(AccelPush) == 32 && offsetof(AccelPush, nb) == 24, "accel");
// computeRigidBodyAccelerations (accelerations, bodySims, prevVelocities, nb, oneOverDt)
struct ComputeAccelPush { uint64_t accelerations, bodySims, prev; uint32_t nb; float oneOverDt; };
static_assert(sizeof(ComputeAccelPush) == 32 && offsetof(ComputeAccelPush, nb) == 24 &&
              offsetof(ComputeAccelPush, oneOverDt) == 28, "computeAccel");
// copyRigidBodyVelocitiesToPrevious (prevVelocities, bodySims, nb)
struct CopyPrevPush { uint64_t prev, bodySims; uint32_t nb, pad; };
static_assert(sizeof(CopyPrevPush) == 24 && offsetof(CopyPrevPush, nb) == 16, "copyPrev");
// getRigidDynamic{Force,Torque} (data, gpuIndices, bodySims, nb)
struct GetPush { uint64_t data, gpuIndices, bodySims; uint32_t nb, pad; };
static_assert(sizeof(GetPush) == 32 && offsetof(GetPush, nb) == 24, "get");
struct PosePush { uint64_t data, gpuIndices, desc; uint32_t nb, totalNumShapes; };
static_assert(sizeof(PosePush) == 32 && offsetof(PosePush, nb) == 24 && offsetof(PosePush, totalNumShapes) == 28, "pose");
struct VelPush { uint64_t data, gpuIndices, desc, prev; uint32_t nb, pad; };
static_assert(sizeof(VelPush) == 40 && offsetof(VelPush, prev) == 24 && offsetof(VelPush, nb) == 32, "vel");
struct ForcePush { uint64_t data, gpuIndices, desc; uint32_t nb, pad; };
static_assert(sizeof(ForcePush) == 32 && offsetof(ForcePush, nb) == 24, "force");

enum : uint32_t { K_LIN_ACC, K_ANG_ACC, K_SET_POSE, K_SET_LIN, K_SET_ANG, K_SET_FORCE, K_SET_TORQUE,
                  K_COMPUTE_ACC, K_COPY_PREV, K_GET_FORCE, K_GET_TORQUE, K_COUNT };

static std::vector<uint8_t> dawLoader(const char* file) { return loadShaderBlob(file); }

PxgCudaShim* makeShim(PxgSlangComputeDevice* dev)
{
    static const PxgKernelDef manifest[K_COUNT] = {
        { "directApiBda.spv", "getRigidDynamicLinearAcceleration", 32u, 0 },
        { "directApiBda.spv", "getRigidDynamicAngularAcceleration", 32u, 0 },
        { "directApiBda.spv", "setRigidDynamicGlobalPose", 32u, 0 },
        { "directApiBda.spv", "setRigidDynamicLinearVelocity", 40u, 0 },
        { "directApiBda.spv", "setRigidDynamicAngularVelocity", 40u, 0 },
        { "directApiBda.spv", "setRigidDynamicForce", 32u, 0 },
        { "directApiBda.spv", "setRigidDynamicTorque", 32u, 0 },
        { "directApiBda.spv", "computeRigidBodyAccelerations", 32u, 0 },
        { "directApiBda.spv", "copyRigidBodyVelocitiesToPrevious", 24u, 0 },
        { "directApiBda.spv", "getRigidDynamicForce", 32u, 0 },
        { "directApiBda.spv", "getRigidDynamicTorque", 32u, 0 },
    };
    return new PxgCudaShim(dev, manifest, K_COUNT, &dawLoader);
}

bool L(PxgCudaShim* shim, uint32_t id, uint32_t n, const void* blob, uint32_t size)
{
    const uint32_t grid = (n + 255u) / 256u;   // the host's grid for every one of these
    return shim->launchKernel(id, grid, 1, 1, blob, size) == PxgResult::Success;
}

// ---------------------------------------------------------------------------
// the flat device-memory model: every buffer is a Region with a host image
// ---------------------------------------------------------------------------
struct Region
{
    PxgDevicePtr dev = 0;
    std::vector<uint8_t> bytes;
};

struct Mem
{
    PxgCudaShim* shim = nullptr;
    std::vector<Region> regions;

    uint64_t alloc(size_t size, uint8_t fill = 0xA5)
    {
        Region r;
        r.bytes.assign((size + 15) & ~size_t(15), fill);
        shim->memAlloc(&r.dev, r.bytes.size());
        regions.push_back(std::move(r));
        return regions.back().dev;
    }
    uint8_t* host(uint64_t addr)
    {
        for (Region& r : regions)
            if (addr >= r.dev && addr < r.dev + r.bytes.size()) return r.bytes.data() + (addr - r.dev);
        std::printf("  oracle: stray address %llx\n", (unsigned long long)addr);
        static uint8_t junk[256];
        return junk;
    }
    template <typename T> T& at(uint64_t addr) { return *reinterpret_cast<T*>(host(addr)); }
    void upload()
    {
        for (Region& r : regions) shim->memcpyHtoD(r.dev, r.bytes.data(), r.bytes.size());
    }
    size_t compare(const char* stage)
    {
        size_t bad = 0;
        for (Region& r : regions)
        {
            std::vector<uint8_t> g(r.bytes.size());
            shim->memcpyDtoH(g.data(), r.dev, g.size());
            for (size_t i = 0; i < g.size(); i += 4)
            {
                uint32_t a, b;
                std::memcpy(&a, g.data() + i, 4);
                std::memcpy(&b, r.bytes.data() + i, 4);
                if (a != b)
                {
                    if (bad < 8)
                        std::printf("  MM %s region %llx word %zu: gpu %08x ref %08x\n", stage,
                                    (unsigned long long)r.dev, i / 4, a, b);
                    ++bad;
                }
            }
        }
        return bad;
    }
    ~Mem()   // declared AFTER the shim's owner in every test, so it runs first
    {
        for (Region& r : regions) shim->memFree(r.dev);
    }
};

static float bitsF(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static uint32_t fBits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

static PxQuat randomUnitQuat(std::mt19937& rng)
{
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    PxQuat q(d(rng), d(rng), d(rng), d(rng));
    return q.getNormalized();
}

// random PxgBodySim records: every word a float in [-5, 5] (or [0.1, 2] where the
// kernels multiply), unit quaternions where the engine keeps them unit
static void fillBodies(Mem& m, uint64_t pool, uint32_t n, std::mt19937& rng)
{
    std::uniform_real_distribution<float> sym(-5.f, 5.f), pos(0.1f, 2.f);
    for (uint32_t b = 0; b < n; ++b)
    {
        float* w = &m.at<float>(pool + uint64_t(b) * SZ_BODYSIM);
        for (uint32_t k = 0; k < 56; ++k) w[k] = sym(rng);
        w[3] = pos(rng);                                             // inverse mass
        for (uint32_t k = 12; k < 15; ++k) w[k] = pos(rng);          // inverse inertia
        const PxQuat q = randomUnitQuat(rng), qa = randomUnitQuat(rng);
        w[28] = q.x; w[29] = q.y; w[30] = q.z; w[31] = q.w;          // body2World.q
        w[36] = qa.x; w[37] = qa.y; w[38] = qa.z; w[39] = qa.w;      // body2Actor.q
        w[40] *= 0.2f; w[41] *= 0.2f; w[42] *= 0.2f;                 // body2Actor.p (a COM offset)
    }
}

// a gpuIndices list: n unique bodies out of `pool` in a non-monotone order,
// followed by `traps` more VALID indices past the launch
static std::vector<uint32_t> uniqueIndices(uint32_t n, uint32_t traps, uint32_t pool, std::mt19937& rng)
{
    std::vector<uint32_t> all(pool);
    std::iota(all.begin(), all.end(), 0u);
    std::shuffle(all.begin(), all.end(), rng);
    all.resize(n + traps);
    return all;
}

// ===========================================================================
// the literal CUDA transcriptions (updateBodiesAndShapes.cu)
// ===========================================================================

// computeRigidBodyAccelerations: (vel - prev) * oneOverDt, 4 lanes, PxLoad3 keeps
// xyz; the record's pad words are not written
static void oComputeAccel(Mem& m, const ComputeAccelPush& p)
{
    for (uint32_t index = 0; index < p.nb; ++index)
    {
        for (uint32_t k = 0; k < 2; ++k)
        {
            const uint32_t velOff = k ? BS_ANGVEL : BS_LINVEL;
            const float* v = &m.at<float>(p.bodySims + uint64_t(index) * SZ_BODYSIM + velOff);
            const float* pv = &m.at<float>(p.prev + uint64_t(index) * SZ_PREV + velOff);
            float* out = &m.at<float>(p.accelerations + uint64_t(index) * SZ_ACCEL + (k ? RA_ANGULAR : RA_LINEAR));
            for (uint32_t c = 0; c < 3; ++c)
            {
                const float delta = v[c] - pv[c];
                out[c] = delta * p.oneOverDt;
            }
        }
    }
}

// getRigidDynamic{Linear,Angular}Acceleration: data[i] = accelerations[index].linear / .angular
static void oAccel(Mem& m, const AccelPush& p, uint32_t accelOff)
{
    for (uint32_t i = 0; i < p.nb; ++i)
    {
        const uint32_t index = m.at<uint32_t>(p.gpuIndices + 4ull * i);
        const uint32_t* a = &m.at<uint32_t>(p.accelerations + uint64_t(index) * SZ_ACCEL + accelOff);
        uint32_t* out = &m.at<uint32_t>(p.data + 12ull * i);
        for (uint32_t c = 0; c < 3; ++c) out[c] = a[c];
    }
}

// copyRigidBodyVelocitiesToPrevious: both whole float4s
static void oCopyPrev(Mem& m, const CopyPrevPush& p)
{
    for (uint32_t index = 0; index < p.nb; ++index)
        for (uint32_t k = 0; k < 2; ++k)
        {
            const uint32_t velOff = k ? BS_ANGVEL : BS_LINVEL;
            const uint32_t* v = &m.at<uint32_t>(p.bodySims + uint64_t(index) * SZ_BODYSIM + velOff);
            uint32_t* pv = &m.at<uint32_t>(p.prev + uint64_t(index) * SZ_PREV + velOff);
            for (uint32_t c = 0; c < 4; ++c) pv[c] = v[c];
        }
}

// getRigidDynamicForce: invMass > 0 ? externalLinearAcceleration * (1 / invMass) : 0
static void oGetForce(Mem& m, const GetPush& p)
{
    for (uint32_t i = 0; i < p.nb; ++i)
    {
        const uint32_t index = m.at<uint32_t>(p.gpuIndices + 4ull * i);
        const float* bs = &m.at<float>(p.bodySims + uint64_t(index) * SZ_BODYSIM);
        const float invMass = bs[3];
        const float* e = bs + BS_EXTLIN / 4;
        const PxVec3 accel(e[0], e[1], e[2]);
        const PxVec3 f = (invMass > 0.f) ? (accel * (1.f / invMass)) : PxVec3(0.f);
        float* out = &m.at<float>(p.data + 12ull * i);
        out[0] = f.x; out[1] = f.y; out[2] = f.z;
    }
}

// getRigidDynamicTorque: Cm::transformInertiaTensor(1 / invInertia (0 where not > 0),
// PxMat33(body2World.q)) * externalAngularAcceleration
static void oGetTorque(Mem& m, const GetPush& p)
{
    for (uint32_t i = 0; i < p.nb; ++i)
    {
        const uint32_t index = m.at<uint32_t>(p.gpuIndices + 4ull * i);
        const float* bs = &m.at<float>(p.bodySims + uint64_t(index) * SZ_BODYSIM);
        const float* ii = bs + BS_INVINERTIA / 4;
        const PxVec3 inertia(ii[0] > 0.f ? 1.f / ii[0] : 0.f, ii[1] > 0.f ? 1.f / ii[1] : 0.f,
                             ii[2] > 0.f ? 1.f / ii[2] : 0.f);
        const float* q = bs + BS_B2W / 4;
        PxMat33 inertiaWorldSpace;
        physx::Cm::transformInertiaTensor(inertia, PxMat33(PxQuat(q[0], q[1], q[2], q[3])), inertiaWorldSpace);
        const float* e = bs + BS_EXTANG / 4;
        const PxVec3 t = inertiaWorldSpace * PxVec3(e[0], e[1], e[2]);
        float* out = &m.at<float>(p.data + 12ull * i);
        out[0] = t.x; out[1] = t.y; out[2] = t.z;
    }
}

// setRigidDynamic{Linear,Angular}Velocity
static void oSetVel(Mem& m, const VelPush& p, uint32_t velOff)
{
    const uint64_t pool = m.at<uint64_t>(p.desc + D_BODYSIM);
    for (uint32_t i = 0; i < p.nb; ++i)
    {
        const uint32_t index = m.at<uint32_t>(p.gpuIndices + 4ull * i);
        uint32_t* bs = &m.at<uint32_t>(pool + uint64_t(index) * SZ_BODYSIM + velOff);
        const uint32_t* d = &m.at<uint32_t>(p.data + 12ull * i);
        const uint32_t v[4] = { d[0], d[1], d[2], bs[3] };
        for (uint32_t c = 0; c < 4; ++c) bs[c] = v[c];
        if (p.prev)
        {
            uint32_t* pv = &m.at<uint32_t>(p.prev + uint64_t(index) * SZ_PREV + velOff);
            for (uint32_t c = 0; c < 4; ++c) pv[c] = v[c];
        }
    }
}

// setRigidDynamicForce: externalLinearAcceleration = (force * invMass, 0)
static void oSetForce(Mem& m, const ForcePush& p)
{
    const uint64_t pool = m.at<uint64_t>(p.desc + D_BODYSIM);
    for (uint32_t i = 0; i < p.nb; ++i)
    {
        const uint32_t index = m.at<uint32_t>(p.gpuIndices + 4ull * i);
        float* bs = &m.at<float>(pool + uint64_t(index) * SZ_BODYSIM);
        const float* d = &m.at<float>(p.data + 12ull * i);
        const PxVec3 force(d[0], d[1], d[2]);
        const PxVec3 deltaLin = force * bs[3];
        float* e = bs + BS_EXTLIN / 4;
        e[0] = deltaLin.x; e[1] = deltaLin.y; e[2] = deltaLin.z; e[3] = 0.f;
    }
}

// setRigidDynamicTorque: externalAngularAcceleration =
//   (Cm::transformInertiaTensor(invI, PxMat33(body2World.q)) * torque, 0)
static void oSetTorque(Mem& m, const ForcePush& p)
{
    const uint64_t pool = m.at<uint64_t>(p.desc + D_BODYSIM);
    for (uint32_t i = 0; i < p.nb; ++i)
    {
        const uint32_t index = m.at<uint32_t>(p.gpuIndices + 4ull * i);
        float* bs = &m.at<float>(pool + uint64_t(index) * SZ_BODYSIM);
        const float* d = &m.at<float>(p.data + 12ull * i);
        const PxVec3 torque(d[0], d[1], d[2]);
        const float* ii = bs + BS_INVINERTIA / 4;
        const float* q = bs + BS_B2W / 4;
        PxMat33 inverseInertiaWorldSpace;
        physx::Cm::transformInertiaTensor(PxVec3(ii[0], ii[1], ii[2]), PxMat33(PxQuat(q[0], q[1], q[2], q[3])),
                                          inverseInertiaWorldSpace);
        const PxVec3 deltaAng = inverseInertiaWorldSpace * torque;
        float* e = bs + BS_EXTANG / 4;
        e[0] = deltaAng.x; e[1] = deltaAng.y; e[2] = deltaAng.z; e[3] = 0.f;
    }
}

// updateCacheAndBound.cuh basisExtent / updateBounds (the real types)
static PxVec3 oBasisExtent(const PxMat33& basis, const PxVec3& extent)
{
    const PxVec3 c0 = basis.column0 * extent.x;
    const PxVec3 c1 = basis.column1 * extent.y;
    const PxVec3 c2 = basis.column2 * extent.z;
    PxVec3 w;
    w.x = physx::PxAbs(c0.x) + physx::PxAbs(c1.x) + physx::PxAbs(c2.x);
    w.y = physx::PxAbs(c0.y) + physx::PxAbs(c1.y) + physx::PxAbs(c2.y);
    w.z = physx::PxAbs(c0.z) + physx::PxAbs(c1.z) + physx::PxAbs(c2.z);
    return w;
}

// the old port's two spellings, kept as TWINS so the fixture can prove it
// discriminates them (P9.93): rotateInv for transformInv, and an isIdentity that
// also demanded rotation xyz == 0
static bool gTwinOldInv = false;     // transformInv as rotateInv
static bool gTwinOldIdent = false;   // isIdentity also demanding rotation xyz == 0

static PxTransform oAbsPose(const PxTransform& body2World, const PxTransform& shape2Actor, const PxTransform& body2Actor)
{
    if (!gTwinOldInv)
        return body2World.transform(body2Actor.transformInv(shape2Actor));
    const PxTransform t0(body2Actor.q.rotateInv(shape2Actor.p - body2Actor.p),
                         body2Actor.q.getConjugate() * shape2Actor.q);
    return body2World.transform(t0);
}

static void oUpdateBounds(Mem& m, const physx::PxgShapeSim& shapeSim, uint64_t convexShapes, uint64_t boundsArray,
                          const PxTransform& pose, uint32_t index)
{
    const PxBounds3& localBound = shapeSim.mLocalBounds;
    PxBounds3& updatedBound = m.at<PxBounds3>(boundsArray + 24ull * index);
    switch (shapeSim.mShapeType)
    {
    case 0u:   // eSPHERE
        updatedBound.minimum = pose.p + localBound.minimum;
        updatedBound.maximum = pose.p + localBound.maximum;
        break;
    case 2u:   // eCAPSULE
    {
        const float radius = localBound.maximum.y;
        const float halfHeight = localBound.maximum.x - radius;
        const PxVec3 d = pose.q.getBasisVector0();
        PxVec3 extents;
        for (uint32_t ax = 0; ax < 3; ax++)
            extents[ax] = physx::PxAbs(d[ax]) * halfHeight + radius;
        updatedBound.minimum = pose.p - extents;
        updatedBound.maximum = pose.p + extents;
        break;
    }
    case 3u:   // eBOX
    {
        const PxVec3 halfExtents = localBound.maximum;
        const PxVec3 extents = oBasisExtent(PxMat33(pose.q), halfExtents);
        updatedBound.minimum = pose.p - extents;
        updatedBound.maximum = pose.p + extents;
        break;
    }
    case 5u:   // eCONVEXMESH
    {
        const uint32_t hullIndex = shapeSim.mHullDataIndex;
        if (hullIndex != 0xFFFFFFFFu)
        {
            const physx::PxgShape& shape = m.at<physx::PxgShape>(convexShapes + 48ull * hullIndex);
            PxMat33 rot(pose.q);
            const bool identity = gTwinOldIdent
                ? (shape.scale.isIdentity() && shape.scale.rotation.x == 0.f && shape.scale.rotation.y == 0.f &&
                   shape.scale.rotation.z == 0.f)
                : shape.scale.isIdentity();
            if (!identity)
                rot = rot * shape.scale.toMat33();
            const uint64_t convexPtr = uint64_t(shape.hullOrMeshPtr);
            const uint32_t tmpX = m.at<uint32_t>(convexPtr + 16);
            const uint32_t nbHullVertices = (tmpX >> 8) & 0xFFu;   // u16High(u32Low(tmp.x))
            PxVec3 minV = PxVec3(PX_MAX_F32);
            PxVec3 maxV = PxVec3(-PX_MAX_F32);
            for (uint32_t i = 0; i < nbHullVertices; ++i)
            {
                const float* vf = &m.at<float>(convexPtr + 48 + 16ull * i);
                const PxVec3 v = PxVec3(vf[0], vf[1], vf[2]);
                const PxVec3 vertexV = rot.transform(v);
                minV = minV.minimum(vertexV);
                maxV = maxV.maximum(vertexV);
            }
            maxV += pose.p;
            minV += pose.p;
            updatedBound.minimum = minV;
            updatedBound.maximum = maxV;
        }
        else
        {
            updatedBound = PxBounds3::transformFast(pose, localBound);
        }
        break;
    }
    default:
        updatedBound = PxBounds3::transformFast(pose, localBound);
        break;
    }
}

struct PoseCoverage { uint32_t walked = 0, wrapped = 0, skippedFfff = 0, geom[11] = {}, noShapes = 0; };

// binarySearch<PxNodeIndex> (reduction.cuh): the last index with data <= value, or 0
static uint32_t oBinarySearch(const uint64_t* data, uint32_t n, uint64_t value)
{
    uint32_t left = 0, right = n;
    while (left < right)
    {
        const uint32_t pos = (left + right) / 2;
        if (data[pos] <= value) left = pos + 1;
        else right = pos;
    }
    return left ? left - 1 : 0;
}

// setRigidDynamicGlobalPose
static void oSetPose(Mem& m, const PosePush& p, PoseCoverage* cov)
{
    const physx::PxgUpdateActorDataDesc& desc = m.at<physx::PxgUpdateActorDataDesc>(p.desc);
    const uint64_t pool = uint64_t(reinterpret_cast<uintptr_t>(desc.mBodySimBufferDeviceData));
    for (uint32_t i = 0; i < p.nb; ++i)
    {
        const uint32_t index = m.at<uint32_t>(p.gpuIndices + 4ull * i);
        const float* d = &m.at<float>(p.data + 28ull * i);
        const PxQuat inQ(d[0], d[1], d[2], d[3]);
        const PxVec3 inP(d[4], d[5], d[6]);
        float* bs = &m.at<float>(pool + uint64_t(index) * SZ_BODYSIM);
        const PxQuat b2aQ(bs[36], bs[37], bs[38], bs[39]);
        const PxVec3 b2aP(bs[40], bs[41], bs[42]);

        // trans.transform(body2Actor_maxImpulseW): (q.rotate(src.p) + p, q * src.q)
        const PxTransform body2World(inQ.rotate(b2aP) + inP, inQ * b2aQ);
        bs[32] = body2World.p.x; bs[33] = body2World.p.y; bs[34] = body2World.p.z; bs[35] = 0.f;
        bs[28] = body2World.q.x; bs[29] = body2World.q.y; bs[30] = body2World.q.z; bs[31] = body2World.q.w;
        const PxTransform body2Actor(b2aP, b2aQ);

        if (p.totalNumShapes == 0)
            continue;

        const uint64_t* nodes = &m.at<uint64_t>(uint64_t(reinterpret_cast<uintptr_t>(desc.mRigidNodeIndices)));
        const uint32_t* shapeIdx = &m.at<uint32_t>(uint64_t(reinterpret_cast<uintptr_t>(desc.mShapeIndices)));
        const uint64_t nodeIndex = physx::PxNodeIndex(index).getInd();
        uint32_t pos = oBinarySearch(nodes, p.totalNumShapes, nodeIndex);
        if (cov && nodes[pos] != nodeIndex) ++cov->noShapes;
        while (pos != 0xFFFFFFFFu && nodes[pos] == nodeIndex)
        {
            const uint32_t shapeIndex = shapeIdx[pos];
            if (shapeIndex != 0xFFFFFFFFu)
            {
                const physx::PxgShapeSim& shapeSim = m.at<physx::PxgShapeSim>(
                    uint64_t(reinterpret_cast<uintptr_t>(desc.mShapeSimsBufferDeviceData)) + 72ull * shapeIndex);
                const PxTransform absPos = oAbsPose(body2World, shapeSim.mTransform, body2Actor);
                physx::PxsCachedTransform& ct = m.at<physx::PxsCachedTransform>(
                    uint64_t(reinterpret_cast<uintptr_t>(desc.mTransformCache)) + 32ull * shapeIndex);
                ct.transform = absPos;
                ct.flags = 0;
                oUpdateBounds(m, shapeSim, uint64_t(reinterpret_cast<uintptr_t>(desc.mShapes)),
                              uint64_t(reinterpret_cast<uintptr_t>(desc.mBounds)), absPos, shapeIndex);
                m.at<uint32_t>(uint64_t(reinterpret_cast<uintptr_t>(desc.mUpdated)) + 4ull * shapeIndex) = 1u;
                if (cov) { ++cov->walked; ++cov->geom[std::min<uint32_t>(shapeSim.mShapeType, 10u)]; }
            }
            else if (cov)
                ++cov->skippedFfff;
            if (cov && pos == 0) ++cov->wrapped;
            pos--;
        }
    }
}

// the 80-byte update-actor descriptor, 0xEE-filled so the pad word must survive
static uint64_t makeDesc(Mem& m, uint64_t pool, uint64_t decoy)
{
    const uint64_t desc = m.alloc(SZ_DESC, 0xEE);
    physx::PxgUpdateActorDataDesc& d = m.at<physx::PxgUpdateActorDataDesc>(desc);
    d.mBodySimBufferDeviceData = reinterpret_cast<physx::PxgBodySim*>(pool);
    d.mShapes = reinterpret_cast<physx::PxgShape*>(decoy);
    d.mTransformCache = reinterpret_cast<physx::PxsCachedTransform*>(decoy);
    d.mBounds = reinterpret_cast<PxBounds3*>(decoy);
    d.mRigidNodeIndices = reinterpret_cast<physx::PxNodeIndex*>(decoy);
    d.mShapeIndices = reinterpret_cast<physx::PxU32*>(decoy);
    d.mUpdated = reinterpret_cast<physx::PxU32*>(decoy);
    d.mChangedAABBMgrHandles = reinterpret_cast<physx::PxU32*>(decoy);
    d.mBitMapWordCounts = 0x13572468u;
    d.mShapeSimsBufferDeviceData = reinterpret_cast<const physx::PxgShapeSim*>(decoy);
    return desc;
}

static const uint32_t POOL = 700;   // bodies in the pool
static const uint32_t NQ = 300;     // elements per launch: two groups, a 44-lane tail
static const uint32_t TRAPS = 20;   // valid indices / data past the launch

} // namespace

// ===========================================================================
PXG_TEST(directApiRigidAccelGet)
{
    PxgSlangComputeDevice* dev = testDevice();
    std::unique_ptr<PxgCudaShim> shimOwner(makeShim(dev));
    PxgCudaShim* shim = shimOwner.get();
    Mem m;
    m.shim = shim;
    std::mt19937 rng(9301);
    std::uniform_real_distribution<float> sym(-5.f, 5.f);

    const uint64_t pool = m.alloc(size_t(POOL) * SZ_BODYSIM);
    const uint64_t prev = m.alloc(size_t(POOL) * SZ_PREV);
    const uint64_t accel = m.alloc(size_t(POOL) * SZ_ACCEL);   // 0xA5: pads and the tail must survive
    fillBodies(m, pool, POOL, rng);
    for (uint32_t k = 0; k < POOL * 8; ++k) m.at<float>(prev + 4ull * k) = sym(rng);

    // the query list: a non-monotone walk with DUPLICATES (the getters only read),
    // then valid trap indices past the launch; all below NB_ACC, the bodies the
    // compute pass covers
    const uint32_t NB_ACC = POOL - 37u;   // a compute grid tail, and bodies past it left alone
    const uint64_t idx = m.alloc(4ull * (NQ + TRAPS));
    for (uint32_t i = 0; i < NQ + TRAPS; ++i) m.at<uint32_t>(idx + 4ull * i) = (i * 7919u + 13u) % NB_ACC;
    m.at<uint32_t>(idx + 4ull * 17) = m.at<uint32_t>(idx + 4ull * 5);
    m.at<uint32_t>(idx + 4ull * 299) = m.at<uint32_t>(idx + 4ull * 0);

    // directed values on queried bodies: vel == prev -> +0; -0 - +0 -> -0; +0 - -0 -> +0
    for (uint32_t off : { BS_LINVEL, BS_ANGVEL })
    {
        const uint32_t b0 = m.at<uint32_t>(idx + 4ull * 3), b1 = m.at<uint32_t>(idx + 4ull * 4);
        float* v0 = &m.at<float>(pool + uint64_t(b0) * SZ_BODYSIM + off);
        float* p0 = &m.at<float>(prev + uint64_t(b0) * SZ_PREV + off);
        for (uint32_t c = 0; c < 4; ++c) p0[c] = v0[c];
        float* v1 = &m.at<float>(pool + uint64_t(b1) * SZ_BODYSIM + off);
        float* p1 = &m.at<float>(prev + uint64_t(b1) * SZ_PREV + off);
        v1[0] = -0.0f; p1[0] = 0.0f;
        v1[1] = 0.0f;  p1[1] = -0.0f;
    }

    const uint64_t data = m.alloc(12ull * (NQ + TRAPS));   // 0xA5 past the launch: must survive
    m.upload();

    size_t bad = 0;
    // 1. the per-step compute pass over the body pool (index == body index)
    const ComputeAccelPush pc = { accel, pool, prev, NB_ACC, 59.7f };
    PXG_CHECK_MSG(L(shim, K_COMPUTE_ACC, NB_ACC, &pc, sizeof(pc)), "launch computeRigidBodyAccelerations");
    shim->streamSynchronize();
    oComputeAccel(m, pc);
    bad += m.compare("compute accelerations");

    // 2. the two gathers
    const AccelPush p = { data, idx, accel, NQ, 0u };
    PXG_CHECK_MSG(L(shim, K_LIN_ACC, NQ, &p, sizeof(p)), "launch getRigidDynamicLinearAcceleration");
    shim->streamSynchronize();
    oAccel(m, p, RA_LINEAR);
    bad += m.compare("linear acceleration");

    PXG_CHECK_MSG(L(shim, K_ANG_ACC, NQ, &p, sizeof(p)), "launch getRigidDynamicAngularAcceleration");
    shim->streamSynchronize();
    oAccel(m, p, RA_ANGULAR);
    bad += m.compare("angular acceleration");

    // the directed lanes really are the zeros they were set up to be
    const float* out = &m.at<float>(data + 12ull * 4);
    PXG_CHECK_MSG(fBits(out[0]) == 0x80000000u && fBits(out[1]) == 0u, "the -0 / +0 lanes (%08x %08x)",
                  fBits(out[0]), fBits(out[1]));

    // 3. the end-of-step snapshot into prevVelocities (raw float4 moves; a NaN
    // payload and -0 on a covered body, so a float copy would show)
    m.at<uint32_t>(pool + 0ull * SZ_BODYSIM + BS_LINVEL + 4) = 0x7FC0ABCDu;
    m.at<uint32_t>(pool + 1ull * SZ_BODYSIM + BS_ANGVEL + 12) = 0x80000000u;
    m.upload();
    const CopyPrevPush pp = { prev, pool, NB_ACC, 0u };
    PXG_CHECK_MSG(L(shim, K_COPY_PREV, NB_ACC, &pp, sizeof(pp)), "launch copyRigidBodyVelocitiesToPrevious");
    shim->streamSynchronize();
    oCopyPrev(m, pp);
    bad += m.compare("copy velocities to previous");
    PXG_CHECK_MSG(bad == 0, "acceleration kernels: %zu word(s) differ", bad);
}

// ===========================================================================
PXG_TEST(directApiRigidGetForceTorque)
{
    PxgSlangComputeDevice* dev = testDevice();
    std::unique_ptr<PxgCudaShim> shimOwner(makeShim(dev));
    PxgCudaShim* shim = shimOwner.get();
    Mem m;
    m.shim = shim;
    std::mt19937 rng(9305);

    const uint64_t pool = m.alloc(size_t(POOL) * SZ_BODYSIM);
    fillBodies(m, pool, POOL, rng);
    // inverse mass / inverse inertia: positive, +0, -0 and negative, so both
    // sides of every `> 0` guard are taken
    for (uint32_t b = 0; b < POOL; ++b)
    {
        float* w = &m.at<float>(pool + uint64_t(b) * SZ_BODYSIM);
        if (b % 5u == 1u) w[3] = 0.0f;
        if (b % 7u == 2u) w[3] = -0.0f;
        if (b % 11u == 3u) w[12] = 0.0f;
        if (b % 13u == 4u) { w[13] = 0.0f; w[14] = -0.5f; }
    }

    const uint64_t idx = m.alloc(4ull * (NQ + TRAPS));
    for (uint32_t i = 0; i < NQ + TRAPS; ++i) m.at<uint32_t>(idx + 4ull * i) = (i * 6151u + 29u) % POOL;
    m.at<uint32_t>(idx + 4ull * 40) = m.at<uint32_t>(idx + 4ull * 2);   // a duplicate
    const uint64_t data = m.alloc(12ull * (NQ + TRAPS));
    m.upload();

    size_t bad = 0;
    const GetPush p = { data, idx, pool, NQ, 0u };
    PXG_CHECK_MSG(L(shim, K_GET_FORCE, NQ, &p, sizeof(p)), "launch getRigidDynamicForce");
    shim->streamSynchronize();
    oGetForce(m, p);
    bad += m.compare("get force");

    PXG_CHECK_MSG(L(shim, K_GET_TORQUE, NQ, &p, sizeof(p)), "launch getRigidDynamicTorque");
    shim->streamSynchronize();
    oGetTorque(m, p);
    bad += m.compare("get torque");
    PXG_CHECK_MSG(bad == 0, "force / torque getters: %zu word(s) differ", bad);
}

// ===========================================================================
PXG_TEST(directApiRigidSetVel)
{
    PxgSlangComputeDevice* dev = testDevice();
    std::unique_ptr<PxgCudaShim> shimOwner(makeShim(dev));
    PxgCudaShim* shim = shimOwner.get();
    Mem m;
    m.shim = shim;
    std::mt19937 rng(9302);
    std::uniform_real_distribution<float> sym(-5.f, 5.f);

    const uint64_t pool = m.alloc(size_t(POOL) * SZ_BODYSIM);
    const uint64_t prev = m.alloc(size_t(POOL) * SZ_PREV);
    const uint64_t decoy = m.alloc(4096, 0x3C);
    fillBodies(m, pool, POOL, rng);
    for (uint32_t k = 0; k < POOL * 8; ++k) m.at<float>(prev + 4ull * k) = sym(rng);
    const uint64_t desc = makeDesc(m, pool, decoy);

    // four launches, each on a fresh unique index list and fresh data
    struct Case { uint32_t kernel, off; bool withPrev; };
    const Case cases[4] = { { K_SET_LIN, BS_LINVEL, false }, { K_SET_ANG, BS_ANGVEL, true },
                            { K_SET_LIN, BS_LINVEL, true },  { K_SET_ANG, BS_ANGVEL, false } };
    size_t bad = 0;
    for (const Case& c : cases)
    {
        const std::vector<uint32_t> ids = uniqueIndices(NQ, TRAPS, POOL, rng);
        const uint64_t idx = m.alloc(4ull * ids.size());
        for (size_t i = 0; i < ids.size(); ++i) m.at<uint32_t>(idx + 4ull * i) = ids[i];
        const uint64_t data = m.alloc(12ull * (NQ + TRAPS));
        for (uint32_t k = 0; k < (NQ + TRAPS) * 3; ++k) m.at<float>(data + 4ull * k) = sym(rng);
        // pure moves: -0, a quiet NaN with a payload, and infinities go through bit-exact
        m.at<uint32_t>(data + 0) = 0x80000000u;
        m.at<uint32_t>(data + 4) = 0x7FC01234u;
        m.at<uint32_t>(data + 8) = 0xFF800000u;
        m.at<uint32_t>(data + 12 * 150 + 4) = 0xFFC0BEEFu;
        m.upload();

        VelPush p = { data, idx, desc, c.withPrev ? prev : 0ull, NQ, 0u };
        PXG_CHECK_MSG(L(shim, c.kernel, NQ, &p, sizeof(p)), "launch velocity setter");
        shim->streamSynchronize();
        oSetVel(m, p, c.off);
        bad += m.compare(c.off == BS_LINVEL ? "set linear velocity" : "set angular velocity");
    }
    PXG_CHECK_MSG(bad == 0, "velocity setters: %zu word(s) differ", bad);
}

// ===========================================================================
PXG_TEST(directApiRigidSetForceTorque)
{
    PxgSlangComputeDevice* dev = testDevice();
    std::unique_ptr<PxgCudaShim> shimOwner(makeShim(dev));
    PxgCudaShim* shim = shimOwner.get();
    Mem m;
    m.shim = shim;
    std::mt19937 rng(9303);
    std::uniform_real_distribution<float> sym(-5.f, 5.f), big(-3000.f, 3000.f);

    const uint64_t pool = m.alloc(size_t(POOL) * SZ_BODYSIM);
    const uint64_t decoy = m.alloc(4096, 0x3C);
    fillBodies(m, pool, POOL, rng);
    const uint64_t desc = makeDesc(m, pool, decoy);

    size_t bad = 0;
    for (uint32_t kernel : { K_SET_FORCE, K_SET_TORQUE })
    {
        const std::vector<uint32_t> ids = uniqueIndices(NQ, TRAPS, POOL, rng);
        const uint64_t idx = m.alloc(4ull * ids.size());
        for (size_t i = 0; i < ids.size(); ++i) m.at<uint32_t>(idx + 4ull * i) = ids[i];
        const uint64_t data = m.alloc(12ull * (NQ + TRAPS));
        for (uint32_t k = 0; k < (NQ + TRAPS) * 3; ++k) m.at<float>(data + 4ull * k) = big(rng);
        for (uint32_t i = 0; i < 12; ++i)
        {
            float* bs = &m.at<float>(pool + uint64_t(ids[i]) * SZ_BODYSIM);
            float* d = &m.at<float>(data + 12ull * i);
            if (kernel == K_SET_FORCE)
            {
                // inverse mass 0 against a negative force component: -0 must come out
                bs[3] = (i % 3 == 0) ? 0.f : bs[3];
                d[i % 3] = -std::fabs(d[i % 3]);
            }
            else
            {
                // anisotropic inverse inertia with one zero component, a negative torque
                bs[12] = 0.5f + 0.25f * float(i % 4);
                bs[13] = (i % 2) ? 0.f : 2.5f;
                bs[14] = 1.0f / 3.0f;
                d[1] = -std::fabs(d[1]);
            }
            // the prior acceleration is live (w non-zero) and must be overwritten
            bs[BS_EXTLIN / 4 + 3] = 7.f;
            bs[BS_EXTANG / 4 + 3] = -7.f;
        }
        m.upload();

        ForcePush p = { data, idx, desc, NQ, 0u };
        PXG_CHECK_MSG(L(shim, kernel, NQ, &p, sizeof(p)), "launch force / torque setter");
        shim->streamSynchronize();
        if (kernel == K_SET_FORCE) oSetForce(m, p); else oSetTorque(m, p);
        bad += m.compare(kernel == K_SET_FORCE ? "set force" : "set torque");

        if (kernel == K_SET_FORCE)
        {
            const float* e = &m.at<float>(pool + uint64_t(ids[0]) * SZ_BODYSIM + BS_EXTLIN);
            PXG_CHECK_MSG(fBits(e[0]) == 0x80000000u, "invMass 0 x a negative force is -0 (%08x)", fBits(e[0]));
        }
    }
    PXG_CHECK_MSG(bad == 0, "force / torque setters: %zu word(s) differ", bad);
}

// ===========================================================================
// setRigidDynamicGlobalPose -- the shape walk
// ===========================================================================
namespace
{
struct PoseWorld
{
    uint64_t pool = 0, desc = 0, nodes = 0, shapeIdx = 0, shapeSims = 0, shapes = 0, cache = 0, bounds = 0,
             updated = 0, changed = 0, blobA = 0, blobB = 0;
    uint32_t nbEntries = 0, trapTotal = 0, nbShapeSims = 0;
    std::vector<uint32_t> queried;
    uint32_t zeroSignBody = 0, zeroSignShape = 0, unitScaleShape = 0;
};

static const uint32_t PW_BODIES = 400;

// hull blob: 16 header bytes, a uint4 whose .x packs the vertex count in bits
// [8, 16), a float4, then float4 vertices from byte 48 (updateCacheAndBound.cuh)
static uint64_t makeHull(Mem& m, uint32_t nbVerts, std::mt19937& rng)
{
    std::uniform_real_distribution<float> sym(-1.5f, 1.5f);
    const uint64_t blob = m.alloc(48 + 16ull * nbVerts, 0x00);
    m.at<uint32_t>(blob + 16) = (nbVerts << 8) | 0x11u;   // other bits live: the mask must hold
    for (uint32_t v = 0; v < nbVerts; ++v)
        for (uint32_t c = 0; c < 4; ++c) m.at<float>(blob + 48 + 16ull * v + 4ull * c) = sym(rng);
    return blob;
}
} // namespace

PXG_TEST(directApiRigidSetPose)
{
    PxgSlangComputeDevice* dev = testDevice();
    std::unique_ptr<PxgCudaShim> shimOwner(makeShim(dev));
    PxgCudaShim* shim = shimOwner.get();
    Mem m;
    m.shim = shim;
    std::mt19937 rng(9304);
    std::uniform_real_distribution<float> sym(-5.f, 5.f), ext(0.1f, 1.5f);
    PoseWorld W;

    W.pool = m.alloc(size_t(PW_BODIES) * SZ_BODYSIM);
    fillBodies(m, W.pool, PW_BODIES, rng);

    // ---- the node table: rigid runs, then statics, then articulation links ----
    struct Entry { uint64_t node; uint32_t shape; };
    std::vector<Entry> table;
    uint32_t nextShape = 0;
    auto shapesOf = [](uint32_t b) -> uint32_t {
        if (b == 0) return 4;            // the run at pos 0: the walk wraps
        if (b == 20u || b == PW_BODIES - 2) return 3;   // the zero-sign body; the trap body
        if (b % 11 == 7) return 0;       // bodies with no shapes (queried ones too)
        return 1 + (b * 2654435761u >> 28) % 4;
    };
    for (uint32_t b = 0; b + 1 < PW_BODIES; ++b)       // the LAST body has no shapes: above every rigid entry
    {
        const uint32_t k = shapesOf(b);
        for (uint32_t s = 0; s < k; ++s)
        {
            uint32_t shape = nextShape++;
            // 0xFFFFFFFF slots first / middle / last in a run
            if ((b % 13 == 1 && s == 0) || (b % 13 == 2 && s == k / 2 && k > 2) || (b % 13 == 3 && s == k - 1))
                shape = 0xFFFFFFFFu;
            table.push_back({ physx::PxNodeIndex(b).getInd(), shape });
        }
    }
    // the trap: body TRAP's run continues past totalNumShapes in launch 3
    const uint32_t trapBody = PW_BODIES - 2;
    W.trapTotal = uint32_t(table.size()) - 1;      // cuts trapBody's last entry off
    // statics (mID 0xFFFFFFFF, link 0) and articulation links whose mID equals a
    // queried body -- a 32-bit or low-word compare would take them
    for (uint32_t s = 0; s < 6; ++s)
        table.push_back({ physx::PxNodeIndex(0xFFFFFFFFu).getInd(), nextShape++ });
    for (uint32_t l = 0; l < 6; ++l)
        table.push_back({ physx::PxNodeIndex(5u + l, l + 1u).getInd(), nextShape++ });
    std::stable_sort(table.begin(), table.end(), [](const Entry& a, const Entry& b) { return a.node < b.node; });
    // shape ELEMENT ids are not in table order in the engine (mShapeIndices maps a
    // sorted node entry to the shape's element id); a permutation makes a port that
    // indexed the shape pools by table position visibly wrong
    {
        std::vector<uint32_t> perm(nextShape);
        std::iota(perm.begin(), perm.end(), 0u);
        std::shuffle(perm.begin(), perm.end(), rng);
        for (Entry& e : table)
            if (e.shape != 0xFFFFFFFFu) e.shape = perm[e.shape];
    }
    W.nbEntries = uint32_t(table.size());
    W.nbShapeSims = nextShape;
    PXG_CHECK_MSG(table[W.trapTotal].node == physx::PxNodeIndex(trapBody).getInd() &&
                  table[W.trapTotal - 1].node == physx::PxNodeIndex(trapBody).getInd(),
                  "the trap entry sits at [trapTotal] inside trapBody's run");

    W.nodes = m.alloc(8ull * (W.nbEntries + 4));
    W.shapeIdx = m.alloc(4ull * (W.nbEntries + 4));
    for (uint32_t k = 0; k < W.nbEntries; ++k)
    {
        m.at<uint64_t>(W.nodes + 8ull * k) = table[k].node;
        m.at<uint32_t>(W.shapeIdx + 4ull * k) = table[k].shape;
    }
    // past the table: more entries matching queried bodies (never read)
    for (uint32_t k = W.nbEntries; k < W.nbEntries + 4; ++k)
    {
        m.at<uint64_t>(W.nodes + 8ull * k) = physx::PxNodeIndex(PW_BODIES - 1).getInd();
        m.at<uint32_t>(W.shapeIdx + 4ull * k) = k - W.nbEntries;
    }

    // ---- hulls and PxgShapes ----
    W.blobA = makeHull(m, 7, rng);
    W.blobB = makeHull(m, 12, rng);
    W.shapes = m.alloc(48ull * 4);
    {
        const PxQuat sr = randomUnitQuat(rng), sr2 = randomUnitQuat(rng);
        const physx::PxMeshScale scales[4] = {
            physx::PxMeshScale(PxVec3(1.f), PxQuat(physx::PxIdentity)),       // identity
            physx::PxMeshScale(PxVec3(1.25f, 0.75f, 1.5f), sr),                 // scaled + rotated
            physx::PxMeshScale(PxVec3(1.f), sr2),                               // P9.93: unit scale, rotated
            physx::PxMeshScale(PxVec3(0.6180339f, 1.7320508f, 1.4142135f), PxQuat(physx::PxIdentity)),
        };
        for (uint32_t h = 0; h < 4; ++h)
        {
            physx::PxgShape& s = m.at<physx::PxgShape>(W.shapes + 48ull * h);
            s.scale = scales[h];
            s.materialIndex = 0xABCD0000u + h;
            s.hullOrMeshPtr = size_t((h & 1) ? W.blobB : W.blobA);
            s.type = 5u;
            s.particleOrSoftbodyId = 0x5A5A0000u + h;
        }
    }

    // ---- shape sims: every geometry type 0-10, every hull variant, fast path ----
    W.shapeSims = m.alloc(72ull * W.nbShapeSims);
    for (uint32_t s = 0; s < W.nbShapeSims; ++s)
    {
        physx::PxgShapeSim& ss = m.at<physx::PxgShapeSim>(W.shapeSims + 72ull * s);
        ss.mTransform = PxTransform(PxVec3(sym(rng) * 0.3f, sym(rng) * 0.3f, sym(rng) * 0.3f), randomUnitQuat(rng));
        const PxVec3 c(sym(rng) * 0.2f, sym(rng) * 0.2f, sym(rng) * 0.2f), e(ext(rng), ext(rng), ext(rng));
        ss.mLocalBounds = PxBounds3(c - e, c + e);
        const uint32_t type = s % 11u;
        ss.mShapeType = uint16_t(type);
        ss.mShapeFlags = uint16_t(s % 3u == 0 ? 0u : (s % 3u == 1 ? 2u : 4u));   // 0, SQ-only, trigger: all ignored
        ss.mHullDataIndex = (type == 5u) ? ((s / 11u) % 5u == 4u ? 0xFFFFFFFFu : (s / 11u) % 4u) : 0xFFFFFFFFu;
        if (type == 2u)   // capsule: max.x = halfHeight + radius, max.y = radius
            ss.mLocalBounds = PxBounds3(PxVec3(-e.x - e.y, -e.y, -e.y), PxVec3(e.x + e.y, e.y, e.y));
        if (type == 3u)   // box: half extents in maximum
            ss.mLocalBounds = PxBounds3(-e, e);
    }

    // ---- the P9.93 directed cases, CHOSEN by measuring both spellings on the host ----
    // (1) conjugate-rotate vs rotateInv: search small exact-valued setups for a body /
    //     shape whose cache or bound bits differ between the two spellings
    // (2) a unit-scale, rotated hull (PxgShape 2) that differs between the two
    //     isIdentity spellings
    // Both are applied to QUERIED bodies below and the discrimination is ASSERTED.

    // queried bodies: 300 unique, including body 0 (the wrap), a no-shape body, the
    // last body (above every rigid entry), and trapBody
    std::vector<uint32_t> ids = uniqueIndices(NQ, TRAPS, PW_BODIES, rng);
    for (uint32_t must : { 0u, 7u, PW_BODIES - 1, trapBody, 20u })
        if (std::find(ids.begin(), ids.begin() + NQ, must) == ids.begin() + NQ)
        {
            auto it = std::find(ids.begin(), ids.end(), must);
            if (it != ids.end()) ids.erase(it);
            ids.insert(ids.begin() + 10, must);
        }
    ids.resize(NQ + TRAPS);
    const uint64_t idx = m.alloc(4ull * ids.size());
    for (size_t i = 0; i < ids.size(); ++i) m.at<uint32_t>(idx + 4ull * i) = ids[i];

    // the pose data: unit quaternions, and a -0 translation component on a few
    const uint64_t data = m.alloc(28ull * (NQ + TRAPS));
    for (uint32_t i = 0; i < NQ + TRAPS; ++i)
    {
        const PxQuat q = randomUnitQuat(rng);
        float* d = &m.at<float>(data + 28ull * i);
        d[0] = q.x; d[1] = q.y; d[2] = q.z; d[3] = q.w;
        d[4] = sym(rng); d[5] = sym(rng); d[6] = sym(rng);
    }

    // (1): the zero-sign case.  Body 20 (queried, three shapes): body2Actor.q from a
    // small set of exact quaternions, body2Actor.p == one shape's local p, identity /
    // exact input rotations and -0 input translations -- the first combination whose
    // two spellings give different BITS for that shape is kept.
    {
        const uint32_t zb = 20u;
        uint32_t zi = 0;
        while (ids[zi] != zb) ++zi;
        uint32_t zs = 0xFFFFFFFFu;
        for (uint32_t k = 0; k < W.nbEntries && zs == 0xFFFFFFFFu; ++k)
            if (table[k].node == physx::PxNodeIndex(zb).getInd() && table[k].shape != 0xFFFFFFFFu) zs = table[k].shape;
        PXG_CHECK_MSG(zs != 0xFFFFFFFFu, "body 20 has a live shape");
        // candidates: every unit quaternion whose components come from
        // {+0, -0, +-1/2, +-sqrt(1/2), +-1} (368 of them), zero body2Actor offsets of
        // either sign, and zero-ish input translations.  Measured offline over the
        // full product: about one combination in ten tells the spellings apart.
        std::vector<PxQuat> qs;
        {
            const float cs[8] = { 0.f, -0.f, 0.5f, -0.5f, 0.70710677f, -0.70710677f, 1.f, -1.f };
            for (float x : cs) for (float y : cs) for (float z : cs) for (float w : cs)
                if (std::fabs(x * x + y * y + z * z + w * w - 1.f) < 1e-6f) qs.push_back(PxQuat(x, y, z, w));
        }
        const PxVec3 ps[2] = { PxVec3(0.f), PxVec3(-0.f, -0.f, -0.f) };
        const PxVec3 inps[3] = { PxVec3(0.f, 0.f, -0.f), PxVec3(-0.f, -0.f, -0.f), PxVec3(0.f, 0.25f, -0.f) };
        bool found = false;
        physx::PxgShapeSim& ss = m.at<physx::PxgShapeSim>(W.shapeSims + 72ull * zs);
        ss.mShapeType = 0u;   // a sphere: the cache carries absPos straight through
        for (const PxQuat& qa : qs) for (const PxVec3& pa : ps) for (const PxQuat& qi : qs) for (const PxVec3& pi : inps)
        {
            if (found) break;
            const PxTransform b2a(pa, qa);
            const PxTransform b2w(qi.rotate(pa) + pi, qi * qa);
            const PxTransform s2a(pa, ss.mTransform.q);
            gTwinOldInv = false; const PxTransform aN = oAbsPose(b2w, s2a, b2a);
            gTwinOldInv = true;  const PxTransform aO = oAbsPose(b2w, s2a, b2a);
            gTwinOldInv = false;
            if (std::memcmp(&aN, &aO, sizeof(PxTransform)) != 0)
            {
                float* bs = &m.at<float>(W.pool + uint64_t(zb) * SZ_BODYSIM);
                bs[36] = qa.x; bs[37] = qa.y; bs[38] = qa.z; bs[39] = qa.w;
                bs[40] = pa.x; bs[41] = pa.y; bs[42] = pa.z;
                ss.mTransform.p = pa;
                float* d = &m.at<float>(data + 28ull * zi);
                d[0] = qi.x; d[1] = qi.y; d[2] = qi.z; d[3] = qi.w;
                d[4] = pi.x; d[5] = pi.y; d[6] = pi.z;
                std::printf("  [ info ] zero-sign case: body2Actor q=(%g,%g,%g,%g) p=(%g,%g,%g), pose q=(%g,%g,%g,%g) "
                            "p=(%g,%g,%g)\n", qa.x, qa.y, qa.z, qa.w, pa.x, pa.y, pa.z, qi.x, qi.y, qi.z, qi.w,
                            pi.x, pi.y, pi.z);
                found = true;
            }
        }
        PXG_CHECK_MSG(found, "a setup exists whose conjugate-rotate and rotateInv spellings differ in bits");
        W.zeroSignBody = zb;
        W.zeroSignShape = zs;
    }

    // ---- outputs, poisoned; decoys ----
    W.cache = m.alloc(32ull * W.nbShapeSims, 0xA5);
    W.bounds = m.alloc(24ull * W.nbShapeSims + 16, 0xA5);
    W.updated = m.alloc(4ull * W.nbShapeSims, 0x5A);
    W.changed = m.alloc(256, 0x3C);
    W.desc = m.alloc(SZ_DESC, 0xEE);
    {
        physx::PxgUpdateActorDataDesc& d = m.at<physx::PxgUpdateActorDataDesc>(W.desc);
        d.mBodySimBufferDeviceData = reinterpret_cast<physx::PxgBodySim*>(W.pool);
        d.mShapes = reinterpret_cast<physx::PxgShape*>(W.shapes);
        d.mTransformCache = reinterpret_cast<physx::PxsCachedTransform*>(W.cache);
        d.mBounds = reinterpret_cast<PxBounds3*>(W.bounds);
        d.mRigidNodeIndices = reinterpret_cast<physx::PxNodeIndex*>(W.nodes);
        d.mShapeIndices = reinterpret_cast<physx::PxU32*>(W.shapeIdx);
        d.mUpdated = reinterpret_cast<physx::PxU32*>(W.updated);
        d.mChangedAABBMgrHandles = reinterpret_cast<physx::PxU32*>(W.changed);
        d.mBitMapWordCounts = 0x2468ACE0u;
        d.mShapeSimsBufferDeviceData = reinterpret_cast<const physx::PxgShapeSim*>(W.shapeSims);
    }
    m.upload();

    size_t bad = 0;
    PoseCoverage cov;

    // launch 1: the full table.  Before it, the launch is replayed on the host with
    // each OLD spelling switched on alone: each must change some bit of this very
    // launch's image, or the fixture could not kill the revert of that fix.
    {
        PosePush p = { data, idx, W.desc, NQ, W.nbEntries };
        const std::vector<Region> before = m.regions;
        auto twinDiff = [&](bool& flag) -> size_t {
            std::vector<Region> saved = m.regions;
            m.regions = before;
            oSetPose(m, p, nullptr);
            const std::vector<Region> fresh = m.regions;
            m.regions = before;
            flag = true;
            oSetPose(m, p, nullptr);
            flag = false;
            size_t diff = 0;
            for (size_t r = 0; r < fresh.size(); ++r)
                for (size_t i = 0; i < fresh[r].bytes.size(); i += 4)
                    diff += std::memcmp(fresh[r].bytes.data() + i, m.regions[r].bytes.data() + i, 4) != 0;
            m.regions = saved;
            return diff;
        };
        const size_t dInv = twinDiff(gTwinOldInv), dIdent = twinDiff(gTwinOldIdent);
        std::printf("  [ info ] words the old spellings change: transformInv-as-rotateInv %zu, "
                    "isIdentity-with-rotation %zu\n", dInv, dIdent);
        PXG_CHECK_MSG(dInv > 0, "the launch tells the conjugate-rotate transformInv from rotateInv");
        PXG_CHECK_MSG(dIdent > 0, "the launch tells the scale-only isIdentity from the old one");

        PXG_CHECK_MSG(L(shim, K_SET_POSE, NQ, &p, sizeof(p)), "launch setRigidDynamicGlobalPose (full table)");
        shim->streamSynchronize();
        oSetPose(m, p, &cov);
        bad += m.compare("set pose, full table");
    }
    // launch 2: totalNumShapes == 0 -- the pose is still written, no shape is touched
    {
        for (uint32_t k = 0; k < (NQ + TRAPS) * 7; ++k)
            if (k % 7 >= 4) m.at<float>(data + 4ull * k) += 1.0f;
        m.upload();
        PosePush p = { data, idx, W.desc, NQ, 0u };
        PXG_CHECK_MSG(L(shim, K_SET_POSE, NQ, &p, sizeof(p)), "launch setRigidDynamicGlobalPose (no shapes)");
        shim->streamSynchronize();
        oSetPose(m, p, nullptr);
        bad += m.compare("set pose, totalNumShapes 0");
    }
    // launch 3: the table cut inside trapBody's run -- the entry at [total] is live
    // and matching, and must not be walked
    {
        std::memset(m.host(W.cache), 0xA5, 32ull * W.nbShapeSims);
        std::memset(m.host(W.updated), 0x5A, 4ull * W.nbShapeSims);
        for (uint32_t k = 0; k < (NQ + TRAPS) * 7; ++k)
            if (k % 7 >= 4) m.at<float>(data + 4ull * k) -= 0.5f;
        m.upload();
        PosePush p = { data, idx, W.desc, NQ, W.trapTotal };
        PXG_CHECK_MSG(L(shim, K_SET_POSE, NQ, &p, sizeof(p)), "launch setRigidDynamicGlobalPose (trap)");
        shim->streamSynchronize();
        oSetPose(m, p, nullptr);
        bad += m.compare("set pose, trap total");
    }
    // launch 4: the table WITHOUT its statics.  Now the articulation links -- whose
    // mID (5..10) is below trapBody's -- directly follow the last rigid run, and only
    // the full 64-bit compare of binarySearch<PxNodeIndex> keeps the search for
    // trapBody inside that run.  (With the statics in between, a low-word compare
    // never reaches the links: every static's low word is 0xFFFFFFFF.)
    {
        std::vector<Entry> t2;
        for (const Entry& e : table)
            if (e.node != physx::PxNodeIndex(0xFFFFFFFFu).getInd()) t2.push_back(e);
        const uint64_t nodes2 = m.alloc(8ull * t2.size());
        const uint64_t idx2 = m.alloc(4ull * t2.size());
        for (size_t k = 0; k < t2.size(); ++k)
        {
            m.at<uint64_t>(nodes2 + 8ull * k) = t2[k].node;
            m.at<uint32_t>(idx2 + 4ull * k) = t2[k].shape;
        }
        physx::PxgUpdateActorDataDesc& d = m.at<physx::PxgUpdateActorDataDesc>(W.desc);
        d.mRigidNodeIndices = reinterpret_cast<physx::PxNodeIndex*>(nodes2);
        d.mShapeIndices = reinterpret_cast<physx::PxU32*>(idx2);
        std::memset(m.host(W.cache), 0xA5, 32ull * W.nbShapeSims);
        std::memset(m.host(W.updated), 0x5A, 4ull * W.nbShapeSims);
        for (uint32_t k = 0; k < (NQ + TRAPS) * 7; ++k)
            if (k % 7 >= 4) m.at<float>(data + 4ull * k) += 0.25f;
        m.upload();
        PosePush p = { data, idx, W.desc, NQ, uint32_t(t2.size()) };
        PXG_CHECK_MSG(L(shim, K_SET_POSE, NQ, &p, sizeof(p)), "launch setRigidDynamicGlobalPose (no statics)");
        shim->streamSynchronize();
        oSetPose(m, p, nullptr);
        bad += m.compare("set pose, no statics");
        PXG_CHECK_MSG(m.at<uint32_t>(W.updated + 4ull * table[W.trapTotal - 1].shape) == 1u,
                      "trapBody's run was walked in the table without statics");
    }
    PXG_CHECK_MSG(bad == 0, "setRigidDynamicGlobalPose: %zu word(s) differ", bad);

    // coverage: every branch the header names was taken
    std::printf("  [ info ] pose walk: %u shapes, %u 0xFFFFFFFF slots, %u wraps, %u no-shape queries; geometry",
                cov.walked, cov.skippedFfff, cov.wrapped, cov.noShapes);
    for (uint32_t g = 0; g < 11; ++g) std::printf(" %u", cov.geom[g]);
    std::printf("\n");
    PXG_CHECK_MSG(cov.wrapped > 0, "the walk ran off pos 0");
    PXG_CHECK_MSG(cov.skippedFfff >= 3, "0xFFFFFFFF slots were met (%u)", cov.skippedFfff);
    PXG_CHECK_MSG(cov.noShapes >= 2, "queried bodies without shapes (%u)", cov.noShapes);
    for (uint32_t g = 0; g < 11; ++g) PXG_CHECK_MSG(cov.geom[g] > 0, "geometry type %u walked", g);
}
