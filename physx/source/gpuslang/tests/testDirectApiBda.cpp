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

// P9.79 -- directApiBda.slang: the DIRECT-GPU API on the engine launch ABI.
//
//   getRigidDynamicGlobalPose / LinearVelocity / AngularVelocity
//                                       (updateBodiesAndShapes.cu:999 / :1019 / :1037)
//   getD6JointForces / getD6JointTorques (updateBodiesAndShapes.cu:1344 / :1367)
//   radixSortCopy2 / radixSortCopyBits2  (radixSortImpl.cu:237 / :218)
//
// WHAT THIS FIXTURE CHOOSES THAT PXG_SCENE=directapi CANNOT.  The scene proves
// these kernels run and that their answers agree with PxConstraint::getForce and
// with a rest height this harness has measured fifty times.  It cannot prove much
// more than that, because every indirection in it collapses:
//
//   - it queries ONE body, at gpuIndex 0, so `index = gpuIndices[i]` is the
//     identity and a port that ignored gpuIndices entirely would pass;
//   - that body's CENTRE OF MASS is at its origin, so body2Actor is the IDENTITY
//     and the whole `body2World * body2Actor^-1` compose is invisible -- a port
//     that returned body2World unchanged would give the same 0.5000;
//   - it has ONE joint, whose id map entry is valid, so neither the
//     `index < constraintIdMapSize` guard nor the `isJointDataIdValid()` guard is
//     ever taken, and the PxVec3(0) they write is never observed;
//   - its oneOverDt is 60 for both getters at once, so a swapped force/torque row
//     would still be scaled the same way;
//   - and the two radix gathers see a rank array with no AGGREGATE entries at all
//     in a two-shape scene, so the 0xffffffff sentinel branch never runs.
//
// So this fixture uses: gpuIndices as a real PERMUTATION into a larger pool, a
// NON-IDENTITY body2Actor with both a rotation and a translation, an id map
// carrying eINVALID_ID, a queried index PAST constraintIdMapSize, a queried index
// whose map entry is valid but whose writeback row is distinct per component, a
// non-unit oneOverDt, and rank arrays with AGGREGATE entries interleaved among a
// permutation.  Every comparison is BIT-EXACT against a CPU transcription written
// from the .cu, not from the .slang.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

// The byte offsets below are not asserted by inspection: every one is pinned
// against the REAL engine struct at compile time (see the static_asserts under
// each layout block).  A header that moves a field fails the build here rather
// than handing the kernels a shifted pointer at runtime.
#include "PxgSimulationCoreDesc.h"
#include "PxgBodySim.h"
#include "PxgConstraintWriteBack.h"
#include "PxgConstraintIdMap.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstddef>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct V3 { float x, y, z; };

static uint64_t addrOf(PxgSlangComputeDevice* dev, PxgBufferHandle h) { return dev->getBufferDeviceAddress(h); }
static void putU64(std::vector<uint8_t>& b, size_t off, uint64_t v) { std::memcpy(b.data() + off, &v, 8); }
static void putU32(std::vector<uint8_t>& b, size_t off, uint32_t v) { std::memcpy(b.data() + off, &v, 4); }
static void putF32(std::vector<uint8_t>& b, size_t off, float v) { std::memcpy(b.data() + off, &v, 4); }
static void putF4(std::vector<uint8_t>& b, size_t off, F4 v)
{
    putF32(b, off + 0, v.x); putF32(b, off + 4, v.y); putF32(b, off + 8, v.z); putF32(b, off + 12, v.w);
}

// ---------------------------------------------------------------------------
// PxgBodySim byte layout (gpusimulationcontroller/include/PxgBodySim.h, 224 B)
// ---------------------------------------------------------------------------
static const uint32_t SZ_BODYSIM = 224u;
static const uint32_t BS_LINVEL = 0u;        // linearVelocityXYZ_inverseMassW
static const uint32_t BS_ANGVEL = 16u;       // angularVelocityXYZ_maxPenBiasW
static const uint32_t BS_BODY2WORLD = 112u;  // PxAlignedTransform (q @0, p @16)
static const uint32_t BS_BODY2ACTOR = 144u;  // PxAlignedTransform
static_assert(sizeof(physx::PxgBodySim) == SZ_BODYSIM, "PxgBodySim stride");
static_assert(offsetof(physx::PxgBodySim, linearVelocityXYZ_inverseMassW) == BS_LINVEL, "BS_LINVEL");
static_assert(offsetof(physx::PxgBodySim, angularVelocityXYZ_maxPenBiasW) == BS_ANGVEL, "BS_ANGVEL");
static_assert(offsetof(physx::PxgBodySim, body2World) == BS_BODY2WORLD, "BS_BODY2WORLD");
static_assert(offsetof(physx::PxgBodySim, body2Actor_maxImpulseW) == BS_BODY2ACTOR, "BS_BODY2ACTOR");

// PxgConstraintWriteback (gpusolver/include/PxgConstraintWriteBack.h, 32 B)
static const uint32_t SZ_WRITEBACK = 32u;
static const uint32_t WB_LINEAR = 0u;   // linearImpulse_broken
static const uint32_t WB_ANGULAR = 16u; // angularImpulse_residual

static_assert(sizeof(physx::PxgConstraintWriteback) == SZ_WRITEBACK, "PxgConstraintWriteback stride");
static_assert(offsetof(physx::PxgConstraintWriteback, linearImpulse_broken) == WB_LINEAR, "WB_LINEAR");
static_assert(offsetof(physx::PxgConstraintWriteback, angularImpulse_residual) == WB_ANGULAR, "WB_ANGULAR");

// PxgConstraintIdMapEntry is one PxU32; eINVALID_ID is 0xffffFFFF.
static const uint32_t CONSTRAINT_ID_INVALID = 0xffffffffu;
static_assert(sizeof(physx::PxgConstraintIdMapEntry) == 4, "PxgConstraintIdMapEntry stride");
static_assert(physx::PxgConstraintIdMapEntry::eINVALID_ID == CONSTRAINT_ID_INVALID, "eINVALID_ID");

// ---- push blobs, byte for byte the HOST packs ---------------------------
// PxgSimulationCore::getRigidDynamicData (PxgSimulationCore.cpp:2755): {8,8,8,4}
struct RigidGetPush
{
    uint64_t data;
    uint64_t gpuIndices;
    uint64_t bodySims;
    uint32_t nbElements;
    uint32_t _pad;
};
static_assert(sizeof(RigidGetPush) == 32, "getRigidDynamic* host pack is 32 bytes");

// PxgSimulationCore::getD6JointData (PxgSimulationCore.cpp:3098): {8,8,4,8,4,8,4}
struct D6GetPush
{
    uint64_t data;
    uint64_t gpuIndices;
    uint32_t nbElements;
    uint32_t _pad0;
    uint64_t constraintWriteBackBuffer;
    float    oneOverDt;
    uint32_t _pad1;
    uint64_t constraintIdMap;
    uint32_t constraintIdMapSize;
    uint32_t _pad2;
};
static_assert(sizeof(D6GetPush) == 56, "getD6Joint* host pack is 56 bytes");

// PxgGpuNarrowphaseCore::computeRigidsToShapes (PxgNarrowphaseCore.cpp:8855): {8,8,8,4}
struct RadixCopy2Push
{
    uint64_t inValue;
    uint64_t outValue;
    uint64_t rank;
    uint32_t numKeys;
    uint32_t _pad;
};
static_assert(sizeof(RadixCopy2Push) == 32, "radixSortCopy2 host pack is 32 bytes");

// PxgNarrowphaseCore.cpp:8681: {8,8,8,4,1} -- the bool is ONE byte at offset 28,
// and the three bytes after it are the blob's zero padding.
struct RadixCopyBits2Push
{
    uint64_t inValue;
    uint64_t outValue;
    uint64_t rank;
    uint32_t numKeys;
    uint32_t lowBit;   // one meaningful byte; the shim zero-fills 29..31
};
static_assert(sizeof(RadixCopyBits2Push) == 32, "radixSortCopyBits2 host pack is 32 bytes");

// ---------------------------------------------------------------------------
// CPU transcriptions, written from the .cu
// ---------------------------------------------------------------------------

// PxQuat::rotate (PxQuat.h) -- the DOUBLED-component spelling PxAlignedQuat uses.
static V3 quatRotate(F4 q, V3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = (q.x * vx + q.y * vy + q.z * vz);
    return V3{ vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * dot2,
               vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * dot2,
               vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * dot2 };
}

// PxQuat::rotateInv
static V3 quatRotateInv(F4 q, V3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = (q.x * vx + q.y * vy + q.z * vz);
    return V3{ vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * dot2,
               vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * dot2,
               vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * dot2 };
}

static F4 quatMul(F4 a, F4 b)
{
    return F4{ a.w * b.x + b.w * a.x + a.y * b.z - a.z * b.y,
               a.w * b.y + b.w * a.y + a.z * b.x - a.x * b.z,
               a.w * b.z + b.w * a.z + a.x * b.y - a.y * b.x,
               a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z };
}

static F4 quatConj(F4 q) { return F4{ -q.x, -q.y, -q.z, q.w }; }

// getRigidDynamicGlobalPose (updateBodiesAndShapes.cu:999):
//   pose = bodySim.body2World * bodySim.body2Actor_maxImpulseW.getInverse();
//   data[i] = pose.getTransform();          // 7 tightly packed floats: q, p
static void refGlobalPose(F4 qW, V3 pW, F4 qA, V3 pA, float* out7)
{
    const F4 qInv = quatConj(qA);
    const V3 negPA{ -pA.x, -pA.y, -pA.z };
    const V3 pInv = quatRotateInv(qA, negPA);

    const F4 q = quatMul(qW, qInv);
    const V3 rot = quatRotate(qW, pInv);
    out7[0] = q.x; out7[1] = q.y; out7[2] = q.z; out7[3] = q.w;
    out7[4] = rot.x + pW.x; out7[5] = rot.y + pW.y; out7[6] = rot.z + pW.z;
}

// getD6JointForceOrTorque<tExtractOperation> (updateBodiesAndShapes.cu:1298)
static void refD6(uint32_t index, uint32_t constraintIdMapSize,
                  const std::vector<uint32_t>& idMap, const std::vector<uint8_t>& wb,
                  float oneOverDt, bool wantTorque, float* out3)
{
    if ((index < constraintIdMapSize) && (idMap[index] != CONSTRAINT_ID_INVALID))
    {
        const uint32_t base = index * SZ_WRITEBACK + (wantTorque ? WB_ANGULAR : WB_LINEAR);
        float row[3];
        std::memcpy(row, wb.data() + base, 12);
        out3[0] = row[0] * oneOverDt;
        out3[1] = row[1] * oneOverDt;
        out3[2] = row[2] * oneOverDt;
    }
    else
    {
        out3[0] = out3[1] = out3[2] = 0.0f;
    }
}

// ---------------------------------------------------------------------------
// The shared scene.  NB_BODIES is larger than NB_QUERIES and the query list is a
// non-monotone permutation into it, so `gpuIndices` is an INDIRECTION.
// ---------------------------------------------------------------------------
static const uint32_t NB_BODIES = 9u;
static const uint32_t NB_QUERIES = 5u;
static const uint32_t QUERY[NB_QUERIES] = { 6u, 0u, 8u, 3u, 6u };  // note 6 appears TWICE

struct BodyState
{
    F4 qW, qA;
    V3 pW, pA;
    V3 linVel, angVel;
};

static BodyState makeBody(uint32_t i)
{
    // A real rotation (not near-identity) and a real translation in BOTH transforms.
    // body2Actor MUST NOT be the identity: with the centre of mass at the body
    // origin -- which is every scene in this harness -- the inverse-compose in
    // getRigidDynamicGlobalPose is unobservable.
    const float a = 0.21f + 0.37f * float(i);
    const float b = -0.63f + 0.17f * float(i);
    const float sa = std::sin(a * 0.5f), ca = std::cos(a * 0.5f);
    const float sb = std::sin(b * 0.5f), cb = std::cos(b * 0.5f);

    BodyState s;
    // a rotation about a non-axis-aligned direction, normalised by construction
    const float nx = 0.6f, ny = 0.48f, nz = 0.64f;   // |n| == 1
    s.qW = F4{ nx * sa, ny * sa, nz * sa, ca };
    s.qA = F4{ nz * sb, nx * sb, ny * sb, cb };
    s.pW = V3{ 1.5f + float(i), -2.25f + 0.5f * float(i), 0.75f * float(i) - 1.0f };
    s.pA = V3{ 0.13f * float(i) - 0.4f, 0.27f, -0.19f * float(i) };
    s.linVel = V3{ 0.5f * float(i) - 1.0f, 2.25f, -0.125f * float(i) };
    s.angVel = V3{ -0.75f, 0.375f * float(i), 1.125f };
    return s;
}

} // namespace

// ===========================================================================
// The three rigid getters.
// ===========================================================================
PXG_TEST(directApiRigidGet)
{
    PxgSlangComputeDevice* dev = testDevice();
    std::vector<PxgBufferHandle> all;
    auto mk = [&](auto const& vec) { PxgBufferHandle h = createBufferFrom(dev, vec); all.push_back(h); return h; };

    std::vector<BodyState> bodies;
    std::vector<uint8_t> pool(size_t(NB_BODIES) * SZ_BODYSIM, 0u);
    for (uint32_t i = 0; i < NB_BODIES; ++i)
    {
        const BodyState s = makeBody(i);
        bodies.push_back(s);
        const size_t o = size_t(i) * SZ_BODYSIM;
        // the .w lanes carry inverseMass / maxPenBias and must NOT reach the output
        putF4(pool, o + BS_LINVEL, F4{ s.linVel.x, s.linVel.y, s.linVel.z, 0.125f + float(i) });
        putF4(pool, o + BS_ANGVEL, F4{ s.angVel.x, s.angVel.y, s.angVel.z, -7.5f - float(i) });
        putF4(pool, o + BS_BODY2WORLD, s.qW);
        putF4(pool, o + BS_BODY2WORLD + 16u, F4{ s.pW.x, s.pW.y, s.pW.z, 0.0f });
        putF4(pool, o + BS_BODY2ACTOR, s.qA);
        putF4(pool, o + BS_BODY2ACTOR + 16u, F4{ s.pA.x, s.pA.y, s.pA.z, 0.0f });
    }

    PxgBufferHandle bPool = mk(pool);
    if (!addrOf(dev, bPool)) { std::printf("  NOTE: no BDA -> skip\n"); for (PxgBufferHandle h : all) dev->destroyBuffer(h); return; }
    PxgBufferHandle bIdx = mk(std::vector<uint32_t>(QUERY, QUERY + NB_QUERIES));

    // One element PAST the queried range is left poisoned in every output buffer:
    // a kernel that ignored nbElements and ran its whole block would clobber it.
    const float POISON = -1234.5f;
    std::vector<float> poseInit(size_t(NB_QUERIES) * 7u + 1u, POISON);
    std::vector<float> linInit(size_t(NB_QUERIES) * 3u + 1u, POISON);
    std::vector<float> angInit(size_t(NB_QUERIES) * 3u + 1u, POISON);
    PxgBufferHandle bPose = mk(poseInit), bLin = mk(linInit), bAng = mk(angInit);

    RigidGetPush p{};
    p.gpuIndices = addrOf(dev, bIdx);
    p.bodySims = addrOf(dev, bPool);
    p.nbElements = NB_QUERIES;

    // the engine's grid: (nbElements + 255) / 256 with block 256
    const uint32_t gridX = (NB_QUERIES + 255u) / 256u;
    {
        p.data = addrOf(dev, bPose);
        Kernel k("directApiBda.spv", "getRigidDynamicGlobalPose", {}, sizeof(RigidGetPush));
        k.dispatch(gridX, {}, p);
    }
    {
        p.data = addrOf(dev, bLin);
        Kernel k("directApiBda.spv", "getRigidDynamicLinearVelocity", {}, sizeof(RigidGetPush));
        k.dispatch(gridX, {}, p);
    }
    {
        p.data = addrOf(dev, bAng);
        Kernel k("directApiBda.spv", "getRigidDynamicAngularVelocity", {}, sizeof(RigidGetPush));
        k.dispatch(gridX, {}, p);
    }
    dev->flush();

    const std::vector<float> gPose = downloadBuffer<float>(dev, bPose, poseInit.size());
    const std::vector<float> gLin = downloadBuffer<float>(dev, bLin, linInit.size());
    const std::vector<float> gAng = downloadBuffer<float>(dev, bAng, angInit.size());
    for (PxgBufferHandle h : all) dev->destroyBuffer(h);

    std::vector<float> rPose = poseInit, rLin = linInit, rAng = angInit;
    for (uint32_t i = 0; i < NB_QUERIES; ++i)
    {
        const BodyState& s = bodies[QUERY[i]];
        refGlobalPose(s.qW, s.pW, s.qA, s.pA, rPose.data() + size_t(i) * 7u);
        rLin[i * 3 + 0] = s.linVel.x; rLin[i * 3 + 1] = s.linVel.y; rLin[i * 3 + 2] = s.linVel.z;
        rAng[i * 3 + 0] = s.angVel.x; rAng[i * 3 + 1] = s.angVel.y; rAng[i * 3 + 2] = s.angVel.z;
    }

    // Bit-exact, poison element included.
    auto cmp = [&](const char* what, const std::vector<float>& got, const std::vector<float>& ref)
    {
        for (size_t i = 0; i < ref.size(); ++i)
            PXG_CHECK_MSG(std::memcmp(&got[i], &ref[i], 4) == 0,
                          "%s[%zu]: got %.9g, expected %.9g", what, i, double(got[i]), double(ref[i]));
    };
    cmp("pose", gPose, rPose);
    cmp("linVel", gLin, rLin);
    cmp("angVel", gAng, rAng);

    // The pose must NOT be body2World: that is what a port which skipped the
    // inverse-compose would return, and every scene in this harness would agree
    // with it.  This is the fixture's whole reason for a non-identity body2Actor.
    const BodyState& s0 = bodies[QUERY[0]];
    PXG_CHECK_MSG(gPose[4] != s0.pW.x || gPose[5] != s0.pW.y || gPose[6] != s0.pW.z,
                  "the returned pose equals body2World -- body2Actor^-1 was not applied");
}

// ===========================================================================
// The two D6 joint getters.
// ===========================================================================
PXG_TEST(directApiD6JointGet)
{
    PxgSlangComputeDevice* dev = testDevice();
    std::vector<PxgBufferHandle> all;
    auto mk = [&](auto const& vec) { PxgBufferHandle h = createBufferFrom(dev, vec); all.push_back(h); return h; };

    // *** THE DEVICE MAP IS LARGER THAN constraintIdMapSize, ON PURPOSE. ***
    //
    // PxgSimulationCore::getD6JointData passes the HOST map's size, not the
    // device buffer's, and says why: "the size is not taken from the device side
    // map because it might be larger than the host side one".  The
    // `index < constraintIdMapSize` guard is what defends that difference.
    //
    // A first version of this fixture allocated the device map at exactly
    // MAP_SIZE, and the forced-fresh sweep caught it: the mutant that deletes the
    // guard SURVIVED.  With the buffer ending at MAP_SIZE, an out-of-range read
    // landed in zero-filled padding, took the eINVALID_ID branch by accident and
    // produced the same PxVec3(0) the guard produces on purpose -- Class D,
    // observationally identical.  So the device map now carries LIVE, VALID
    // entries past MAP_SIZE, with live writeback rows behind them: delete the
    // guard and the kernel returns a real, non-zero answer, which the comparison
    // sees.
    const uint32_t MAP_SIZE = 6u;          // what the HOST knows about
    const uint32_t DEVICE_MAP_SIZE = 12u;  // what the DEVICE buffer actually holds
    std::vector<uint32_t> idMap(DEVICE_MAP_SIZE);
    for (uint32_t i = 0; i < DEVICE_MAP_SIZE; ++i) idMap[i] = i * 3u + 1u;
    idMap[4] = CONSTRAINT_ID_INVALID;   // an inactive / sleeping joint

    // One writeback row per DEVICE map slot, every component distinct so a swapped
    // row, a swapped component or a shifted index all show up -- and so that a
    // read past constraintIdMapSize finds something loud rather than zeros.
    std::vector<uint8_t> wb(size_t(DEVICE_MAP_SIZE) * SZ_WRITEBACK, 0u);
    for (uint32_t i = 0; i < DEVICE_MAP_SIZE; ++i)
    {
        const size_t o = size_t(i) * SZ_WRITEBACK;
        putF4(wb, o + WB_LINEAR, F4{ 1.0f + float(i), -2.5f - float(i), 0.75f * float(i), 1.0f /*broken*/ });
        putF4(wb, o + WB_ANGULAR, F4{ -0.5f * float(i), 3.25f + float(i), -1.125f - float(i), 0.25f /*residual*/ });
    }

    //                                  valid, valid, INVALID entry, PAST the map, valid
    const uint32_t JQ[5] = { 2u, 0u, 4u, 9u, 5u };
    const uint32_t NB_J = 5u;
    const float ONE_OVER_DT = 60.0f;   // NOT 1: a dropped multiply must show

    PxgBufferHandle bWb = mk(wb);
    if (!addrOf(dev, bWb)) { std::printf("  NOTE: no BDA -> skip\n"); for (PxgBufferHandle h : all) dev->destroyBuffer(h); return; }
    PxgBufferHandle bMap = mk(idMap);
    PxgBufferHandle bIdx = mk(std::vector<uint32_t>(JQ, JQ + NB_J));

    const float POISON = -1234.5f;
    std::vector<float> fInit(size_t(NB_J) * 3u + 1u, POISON);
    std::vector<float> tInit(size_t(NB_J) * 3u + 1u, POISON);
    PxgBufferHandle bF = mk(fInit), bT = mk(tInit);

    D6GetPush p{};
    p.gpuIndices = addrOf(dev, bIdx);
    p.nbElements = NB_J;
    p.constraintWriteBackBuffer = addrOf(dev, bWb);
    p.oneOverDt = ONE_OVER_DT;
    p.constraintIdMap = addrOf(dev, bMap);
    p.constraintIdMapSize = MAP_SIZE;

    const uint32_t gridX = (NB_J + 255u) / 256u;
    {
        p.data = addrOf(dev, bF);
        Kernel k("directApiBda.spv", "getD6JointForces", {}, sizeof(D6GetPush));
        k.dispatch(gridX, {}, p);
    }
    {
        p.data = addrOf(dev, bT);
        Kernel k("directApiBda.spv", "getD6JointTorques", {}, sizeof(D6GetPush));
        k.dispatch(gridX, {}, p);
    }
    dev->flush();

    const std::vector<float> gF = downloadBuffer<float>(dev, bF, fInit.size());
    const std::vector<float> gT = downloadBuffer<float>(dev, bT, tInit.size());
    for (PxgBufferHandle h : all) dev->destroyBuffer(h);

    std::vector<float> rF = fInit, rT = tInit;
    for (uint32_t i = 0; i < NB_J; ++i)
    {
        refD6(JQ[i], MAP_SIZE, idMap, wb, ONE_OVER_DT, false, rF.data() + size_t(i) * 3u);
        refD6(JQ[i], MAP_SIZE, idMap, wb, ONE_OVER_DT, true, rT.data() + size_t(i) * 3u);
    }

    for (size_t i = 0; i < rF.size(); ++i)
        PXG_CHECK_MSG(std::memcmp(&gF[i], &rF[i], 4) == 0,
                      "force[%zu]: got %.9g, expected %.9g", i, double(gF[i]), double(rF[i]));
    for (size_t i = 0; i < rT.size(); ++i)
        PXG_CHECK_MSG(std::memcmp(&gT[i], &rT[i], 4) == 0,
                      "torque[%zu]: got %.9g, expected %.9g", i, double(gT[i]), double(rT[i]));

    // The two else-branch lanes must be EXACTLY zero, and they must be the ones
    // the CUDA says: entry 2 (a valid index whose map entry is eINVALID_ID) and
    // entry 3 (an index past constraintIdMapSize).  A port that dropped either
    // guard would read a live writeback row -- or, for entry 3, off the end.
    for (uint32_t lane : { 2u, 3u })
        for (uint32_t c = 0; c < 3u; ++c)
        {
            PXG_CHECK_MSG(gF[lane * 3 + c] == 0.0f, "force lane %u component %u is not zero", lane, c);
            PXG_CHECK_MSG(gT[lane * 3 + c] == 0.0f, "torque lane %u component %u is not zero", lane, c);
        }
    // ...and the answered lanes must NOT be zero, or the check above proves nothing.
    PXG_CHECK_MSG(gF[0] != 0.0f || gF[1] != 0.0f || gF[2] != 0.0f, "lane 0 force is all zero");
    PXG_CHECK_MSG(gT[0] != 0.0f || gT[1] != 0.0f || gT[2] != 0.0f, "lane 0 torque is all zero");
    // The force and the torque must differ: they read DIFFERENT writeback rows,
    // and a port that read linearImpulse_broken twice would pass everything above
    // that did not compare the two.
    PXG_CHECK_MSG(gF[0] != gT[0] || gF[1] != gT[1] || gF[2] != gT[2],
                  "force and torque are identical -- both read the same writeback row");
}

// ===========================================================================
// The two aggregate-aware radix gathers.
// ===========================================================================
PXG_TEST(directApiRadixCopy2)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t N = 37u;              // not a multiple of the 256 block
    const uint32_t AGG = 0xffffffffu;

    // A real permutation of [0, N) with AGGREGATE entries punched into it.  The
    // aggregate lanes are the branch the non-"2" gathers do not have.
    std::vector<uint32_t> rank(N);
    for (uint32_t i = 0; i < N; ++i) rank[i] = (i * 11u + 7u) % N;
    rank[0] = AGG; rank[13] = AGG; rank[N - 1u] = AGG;

    // 64-bit keys as uint2 (x = low, y = high).  low and high are DIFFERENT and
    // neither is derivable from the other, so a lowBit that selects the wrong word
    // cannot pass.
    std::vector<uint32_t> keys(size_t(N) * 2u);
    for (uint32_t i = 0; i < N; ++i)
    {
        keys[i * 2 + 0] = 0x1000u + i * 7u;
        keys[i * 2 + 1] = 0xAB00u + i * 3u;
    }

    std::vector<PxgBufferHandle> all;
    auto mk = [&](auto const& vec) { PxgBufferHandle h = createBufferFrom(dev, vec); all.push_back(h); return h; };

    PxgBufferHandle bKeys = mk(keys);
    if (!addrOf(dev, bKeys)) { std::printf("  NOTE: no BDA -> skip\n"); for (PxgBufferHandle h : all) dev->destroyBuffer(h); return; }
    PxgBufferHandle bRank = mk(rank);

    const uint32_t POISON = 0xCDCDCDCDu;
    std::vector<uint32_t> out64Init(size_t(N) * 2u + 2u, POISON);   // one extra key past the end
    std::vector<uint32_t> outLoInit(size_t(N) + 1u, POISON);
    std::vector<uint32_t> outHiInit(size_t(N) + 1u, POISON);
    PxgBufferHandle b64 = mk(out64Init), bLo = mk(outLoInit), bHi = mk(outHiInit);

    // the engine's grid (PxgNarrowphaseCore.cpp:8677): (totalNumShapes + 255) / 256
    const uint32_t gridX = (N + 255u) / 256u;
    {
        RadixCopy2Push p{};
        p.inValue = addrOf(dev, bKeys);
        p.outValue = addrOf(dev, b64);
        p.rank = addrOf(dev, bRank);
        p.numKeys = N;
        Kernel k("directApiBda.spv", "radixSortCopy2", {}, sizeof(RadixCopy2Push));
        k.dispatch(gridX, {}, p);
    }
    for (uint32_t low = 0; low < 2u; ++low)
    {
        RadixCopyBits2Push p{};
        p.inValue = addrOf(dev, bKeys);
        p.outValue = addrOf(dev, low ? bLo : bHi);
        p.rank = addrOf(dev, bRank);
        p.numKeys = N;
        p.lowBit = low;   // the CUDA passes a `const bool`; 1 and 0 are its two values
        Kernel k("directApiBda.spv", "radixSortCopyBits2", {}, sizeof(RadixCopyBits2Push));
        k.dispatch(gridX, {}, p);
    }
    dev->flush();

    const std::vector<uint32_t> g64 = downloadBuffer<uint32_t>(dev, b64, out64Init.size());
    const std::vector<uint32_t> gLo = downloadBuffer<uint32_t>(dev, bLo, outLoInit.size());
    const std::vector<uint32_t> gHi = downloadBuffer<uint32_t>(dev, bHi, outHiInit.size());
    for (PxgBufferHandle h : all) dev->destroyBuffer(h);

    // references, transcribed from radixSortImpl.cu:218 / :237
    std::vector<uint32_t> r64 = out64Init, rLo = outLoInit, rHi = outHiInit;
    for (uint32_t i = 0; i < N; ++i)
    {
        const uint32_t index = rank[i];
        if (index == AGG)
        {
            r64[i * 2 + 0] = 0xffffffffu; r64[i * 2 + 1] = 0xffffffffu;
            rLo[i] = 0xffffffffu; rHi[i] = 0xffffffffu;
        }
        else
        {
            r64[i * 2 + 0] = keys[index * 2 + 0];
            r64[i * 2 + 1] = keys[index * 2 + 1];
            rLo[i] = keys[index * 2 + 0];   // lowBit  -> PxU32(key & 0xffffffff)
            rHi[i] = keys[index * 2 + 1];   // !lowBit -> PxU32(key >> 32)
        }
    }

    for (size_t i = 0; i < r64.size(); ++i)
        PXG_CHECK_MSG(g64[i] == r64[i], "copy2[%zu]: got 0x%08x, expected 0x%08x", i, g64[i], r64[i]);
    for (size_t i = 0; i < rLo.size(); ++i)
        PXG_CHECK_MSG(gLo[i] == rLo[i], "copyBits2 low[%zu]: got 0x%08x, expected 0x%08x", i, gLo[i], rLo[i]);
    for (size_t i = 0; i < rHi.size(); ++i)
        PXG_CHECK_MSG(gHi[i] == rHi[i], "copyBits2 high[%zu]: got 0x%08x, expected 0x%08x", i, gHi[i], rHi[i]);

    // the two lowBit arms must differ somewhere, or "selects the word" is untested
    bool differs = false;
    for (uint32_t i = 0; i < N && !differs; ++i) differs = (gLo[i] != gHi[i]);
    PXG_CHECK_MSG(differs, "lowBit=1 and lowBit=0 produced identical output");
}

// ===========================================================================
// mergeChangedAABBMgrHandlesLaunch -- the `|=` twin.
// ===========================================================================
PXG_TEST(directApiMergeAabb)
{
    PxgSlangComputeDevice* dev = testDevice();
    std::vector<PxgBufferHandle> all;
    auto mk = [&](auto const& vec) { PxgBufferHandle h = createBufferFrom(dev, vec); all.push_back(h); return h; };

    // PxgUpdateActorDataDesc byte layout (PxgSimulationCoreDesc.h:375, 80 B).
    // These offsets are the ones pxgSlangSimCoreDesc.slang mirrors; the
    // simCoreDescLayout suite pins them against the real struct.
    const uint32_t SZ_UPDATE_ACTOR_DESC = 80u;
    const uint32_t UAD_UPDATED = 48u, UAD_CHANGED = 56u, UAD_WORDCOUNTS = 64u;
    static_assert(sizeof(physx::PxgUpdateActorDataDesc) == 80u, "PxgUpdateActorDataDesc size");
    static_assert(offsetof(physx::PxgUpdateActorDataDesc, mUpdated) == 48u, "UAD_UPDATED");
    static_assert(offsetof(physx::PxgUpdateActorDataDesc, mChangedAABBMgrHandles) == 56u, "UAD_CHANGED");
    static_assert(offsetof(physx::PxgUpdateActorDataDesc, mBitMapWordCounts) == 64u, "UAD_WORDCOUNTS");
    static_assert(offsetof(physx::PxgUpdateActorDataDesc, mShapeSimsBufferDeviceData) == 72u, "UAD_SHAPESIMS");

    // Three words, i.e. 96 shapes -- more than one workgroup's word so the
    // grid-stride `w += WorkgroupCount().x` has to advance, and the engine's grid
    // is 64 so the loop exits on the first pass for all three.
    const uint32_t NB_WORDS = 3u;
    std::vector<uint32_t> updated(size_t(NB_WORDS) * 32u, 0u);
    for (uint32_t i = 0; i < updated.size(); ++i)
        updated[i] = ((i * 7u + 3u) % 5u == 0u) ? (1u + i) : 0u;   // NON-BOOLEAN truthy values

    // The CPU-side words already carry bits the direct-API side does not know
    // about.  THIS is what `|=` protects and `=` would destroy, and the check
    // below fails loudly for a port that wrote `=`.
    std::vector<uint32_t> changedInit(NB_WORDS);
    changedInit[0] = 0x0000000Fu;
    changedInit[1] = 0x80000000u;
    changedInit[2] = 0xA5A5A5A5u;

    std::vector<uint8_t> desc(SZ_UPDATE_ACTOR_DESC, 0u);
    PxgBufferHandle bUpdated = mk(updated);
    if (!addrOf(dev, bUpdated)) { std::printf("  NOTE: no BDA -> skip\n"); for (PxgBufferHandle h : all) dev->destroyBuffer(h); return; }
    PxgBufferHandle bChanged = mk(changedInit);
    putU64(desc, UAD_UPDATED, addrOf(dev, bUpdated));
    putU64(desc, UAD_CHANGED, addrOf(dev, bChanged));
    putU32(desc, UAD_WORDCOUNTS, NB_WORDS);
    PxgBufferHandle bDesc = mk(desc);

    struct MergePush { uint64_t updateActorDesc; };
    MergePush p{ addrOf(dev, bDesc) };
    {
        // the engine's grid (PxgSimulationCore.cpp:2124): UPDATE_AABBMGR_HANDLES = 64
        Kernel k("updateBounds.spv", "mergeChangedAABBMgrHandlesLaunch", {}, sizeof(MergePush));
        k.dispatch(64u, {}, p);
    }
    dev->flush();

    const std::vector<uint32_t> got = downloadBuffer<uint32_t>(dev, bChanged, NB_WORDS);
    for (PxgBufferHandle h : all) dev->destroyBuffer(h);

    // reference, from updateTransformAndBoundArray.cu:217
    std::vector<uint32_t> ref = changedInit;
    for (uint32_t w = 0; w < NB_WORDS; ++w)
    {
        uint32_t word = 0u;
        for (uint32_t lane = 0; lane < 32u; ++lane)
            if (updated[w * 32u + lane] != 0u) word |= (1u << lane);   // __ballot_sync
        ref[w] |= word;                                                 // |=, not =
    }

    for (uint32_t w = 0; w < NB_WORDS; ++w)
        PXG_CHECK_MSG(got[w] == ref[w], "changed word %u: got 0x%08x, expected 0x%08x", w, got[w], ref[w]);

    // The pre-existing CPU-side bits must all have survived.  A port that wrote
    // `=` passes the ballot comparison above only if this scene happened to set
    // every one of them too -- so assert it directly.
    for (uint32_t w = 0; w < NB_WORDS; ++w)
        PXG_CHECK_MSG((got[w] & changedInit[w]) == changedInit[w],
                      "word %u dropped CPU-side bits: 0x%08x & 0x%08x", w, got[w], changedInit[w]);
    // ...and the merge must have ADDED something, or "survived" is vacuous.
    bool added = false;
    for (uint32_t w = 0; w < NB_WORDS; ++w) added = added || (got[w] != changedInit[w]);
    PXG_CHECK_MSG(added, "the merge changed nothing -- the ballot never contributed a bit");
}

// ===========================================================================
// updateBodiesLaunchDirectAPI -- the body-sim DMA the getters read after.
// ===========================================================================
PXG_TEST(directApiUpdateBodies)
{
    PxgSlangComputeDevice* dev = testDevice();
    std::vector<PxgBufferHandle> all;
    auto mk = [&](auto const& vec) { PxgBufferHandle h = createBufferFrom(dev, vec); all.push_back(h); return h; };

    // PxgNewBodiesDesc (24 B incl. tail padding) -- pxgSlangSimCoreDesc's mirror.
    const uint32_t SZ_NEW_BODIES_DESC = 24u;
    const uint32_t NBD_NEW_BODY_SIM = 0u, NBD_POOL = 8u, NBD_COUNT = 16u;
    static_assert(sizeof(physx::PxgNewBodiesDesc) == 24u, "PxgNewBodiesDesc size");
    static_assert(offsetof(physx::PxgNewBodiesDesc, mNewBodySim) == 0u, "NBD_NEW_BODY_SIM");
    static_assert(offsetof(physx::PxgNewBodiesDesc, mBodySimBufferDeviceData) == 8u, "NBD_POOL");
    static_assert(offsetof(physx::PxgNewBodiesDesc, mNbNewBodies) == 16u, "NBD_COUNT");

    const uint32_t BS_INTERNAL_FLAGS_BYTE = 180u;   // word 45
    const uint32_t BS_INDEX_BYTE = 108u;            // word 27: ..._bodySimIndex.w
    const uint32_t FIRST_BODY_COPY_GPU = 1u << 9;
    const uint32_t VELOCITY_COPY_GPU = 1u << 10;

    // Three staged records scattering into a LARGER pool at NON-MONOTONE indices,
    // so `bodyIndex` is read from the record and not assumed to be the loop index.
    const uint32_t NB_NEW = 3u;
    const uint32_t POOL_BODIES = 7u;
    const uint32_t DST[NB_NEW] = { 5u, 1u, 4u };
    // record 0: first transfer (copies everything)
    // record 1: subsequent, velocity NOT copied, body2Actor CHANGED (re-derive)
    // record 2: subsequent, velocity copied, body2Actor UNCHANGED (no re-derive)
    const uint32_t FLAGS[NB_NEW] = { FIRST_BODY_COPY_GPU, 0u, VELOCITY_COPY_GPU };

    std::vector<uint8_t> staged(size_t(NB_NEW) * SZ_BODYSIM, 0u);
    std::vector<uint8_t> pool(size_t(POOL_BODIES) * SZ_BODYSIM, 0u);

    // Fill the pool with a distinct pattern so anything NOT written shows up, and
    // give each destination a live body2World / body2Actor to be re-derived from.
    for (uint32_t i = 0; i < POOL_BODIES; ++i)
        for (uint32_t w = 0; w < SZ_BODYSIM / 4u; ++w)
            putU32(pool, size_t(i) * SZ_BODYSIM + size_t(w) * 4u, 0xB0000000u + i * 256u + w);
    for (uint32_t n = 0; n < NB_NEW; ++n)
    {
        const BodyState s = makeBody(n + 2u);
        const size_t o = size_t(DST[n]) * SZ_BODYSIM;
        putF4(pool, o + BS_BODY2WORLD, s.qW);
        putF4(pool, o + BS_BODY2WORLD + 16u, F4{ s.pW.x, s.pW.y, s.pW.z, 0.0f });
        putF4(pool, o + BS_BODY2ACTOR, s.qA);
        putF4(pool, o + BS_BODY2ACTOR + 16u, F4{ s.pA.x, s.pA.y, s.pA.z, 0.0f });
        putF4(pool, o + BS_LINVEL, F4{ -9.0f - float(n), 8.0f, -7.0f, 0.5f });
        putF4(pool, o + BS_ANGVEL, F4{ 6.0f, -5.0f + float(n), 4.0f, -0.25f });
    }
    for (uint32_t n = 0; n < NB_NEW; ++n)
    {
        const size_t o = size_t(n) * SZ_BODYSIM;
        for (uint32_t w = 0; w < SZ_BODYSIM / 4u; ++w)
            putU32(staged, o + size_t(w) * 4u, 0x5000000u + n * 256u + w);
        putU32(staged, o + BS_INDEX_BYTE, DST[n]);
        putU32(staged, o + BS_INTERNAL_FLAGS_BYTE, FLAGS[n]);
        const BodyState s = makeBody(n + 2u);
        // record 1 brings a DIFFERENT body2Actor (the setCMassLocalPose case);
        // record 2 brings the same one the pool already has.
        const BodyState alt = makeBody(n + 5u);
        const BodyState src = (n == 1u) ? alt : s;
        putF4(staged, o + BS_BODY2ACTOR, src.qA);
        putF4(staged, o + BS_BODY2ACTOR + 16u, F4{ src.pA.x, src.pA.y, src.pA.z, 0.0f });
        putF4(staged, o + BS_BODY2WORLD, src.qW);
        putF4(staged, o + BS_BODY2WORLD + 16u, F4{ src.pW.x, src.pW.y, src.pW.z, 0.0f });
        putF4(staged, o + BS_LINVEL, F4{ 1.5f + float(n), 2.5f, 3.5f, 0.125f * float(n) });
        putF4(staged, o + BS_ANGVEL, F4{ -1.5f, -2.5f - float(n), -3.5f, -0.75f });
    }

    PxgBufferHandle bStaged = mk(staged);
    if (!addrOf(dev, bStaged)) { std::printf("  NOTE: no BDA -> skip\n"); for (PxgBufferHandle h : all) dev->destroyBuffer(h); return; }
    PxgBufferHandle bPool = mk(pool);
    std::vector<uint8_t> desc(SZ_NEW_BODIES_DESC, 0u);
    putU64(desc, NBD_NEW_BODY_SIM, addrOf(dev, bStaged));
    putU64(desc, NBD_POOL, addrOf(dev, bPool));
    putU32(desc, NBD_COUNT, NB_NEW);
    PxgBufferHandle bDesc = mk(desc);

    struct NewBodiesPush { uint64_t scDesc; };
    NewBodiesPush p{ addrOf(dev, bDesc) };
    {
        // the engine's grid (PxgSimulationCore.cpp:2352): UPDATE_BODIES_AND_SHAPES = 64
        Kernel k("directApiBda.spv", "updateBodiesLaunchDirectAPI", {}, sizeof(NewBodiesPush));
        k.dispatch(64u, {}, p);
    }
    dev->flush();

    const std::vector<uint8_t> got = downloadBuffer<uint8_t>(dev, bPool, pool.size());
    for (PxgBufferHandle h : all) dev->destroyBuffer(h);

    // reference, transcribed from updateBodiesAndShapes.cu:105
    std::vector<uint8_t> ref = pool;
    auto loadF4At = [](const std::vector<uint8_t>& b, size_t off) {
        F4 v; std::memcpy(&v, b.data() + off, 16); return v;
    };
    for (uint32_t n = 0; n < NB_NEW; ++n)
    {
        const size_t src = size_t(n) * SZ_BODYSIM;
        const size_t dst = size_t(DST[n]) * SZ_BODYSIM;
        const bool firstTransfer = (FLAGS[n] & FIRST_BODY_COPY_GPU) != 0u;
        const bool copyVel = (FLAGS[n] & VELOCITY_COPY_GPU) != 0u;
        if (firstTransfer)
        {
            std::memcpy(ref.data() + dst, staged.data() + src, SZ_BODYSIM);
            continue;
        }
        const F4 oldW_q = loadF4At(ref, dst + BS_BODY2WORLD);
        const F4 oldW_p = loadF4At(ref, dst + BS_BODY2WORLD + 16u);
        const F4 oldA_q = loadF4At(ref, dst + BS_BODY2ACTOR);
        const F4 oldA_p = loadF4At(ref, dst + BS_BODY2ACTOR + 16u);

        for (uint32_t u4 = 0; u4 < 12u; ++u4)   // PXG_BODY_SIM_SIZE_WITHOUT_ACCELERATION
        {
            const size_t w0 = size_t(u4) * 16u;
            if (u4 < 2u && !copyVel)            // PXG_BODY_SIM_MAX_LIN_VEL_IND
                std::memcpy(ref.data() + dst + w0 + 12u, staged.data() + src + w0 + 12u, 4);
            else
                std::memcpy(ref.data() + dst + w0, staged.data() + src + w0, 16);
        }

        const F4 newA_q = loadF4At(ref, dst + BS_BODY2ACTOR);
        const F4 newA_p = loadF4At(ref, dst + BS_BODY2ACTOR + 16u);
        const bool notEqual = !(oldA_p.x == newA_p.x && oldA_p.y == newA_p.y && oldA_p.z == newA_p.z &&
                                oldA_q.x == newA_q.x && oldA_q.y == newA_q.y &&
                                oldA_q.z == newA_q.z && oldA_q.w == newA_q.w);
        if (notEqual)
        {
            // actor2World = body2WorldOld * oldBody2Actor^-1 ; body2World = actor2World * newBody2Actor
            const F4 invQ = quatConj(oldA_q);
            const V3 invP = quatRotateInv(oldA_q, V3{ -oldA_p.x, -oldA_p.y, -oldA_p.z });
            const F4 a2wQ = quatMul(oldW_q, invQ);
            const V3 rot0 = quatRotate(oldW_q, invP);
            const V3 a2wP{ rot0.x + oldW_p.x, rot0.y + oldW_p.y, rot0.z + oldW_p.z };

            const F4 b2wQ = quatMul(a2wQ, newA_q);
            const V3 rot1 = quatRotate(a2wQ, V3{ newA_p.x, newA_p.y, newA_p.z });
            const F4 b2wP{ rot1.x + a2wP.x, rot1.y + a2wP.y, rot1.z + a2wP.z, 0.0f };
            putF4(ref, dst + BS_BODY2WORLD, b2wQ);
            putF4(ref, dst + BS_BODY2WORLD + 16u, b2wP);
        }
    }

    PXG_CHECK_MSG(got.size() == ref.size(), "pool size changed");
    for (size_t i = 0; i < ref.size(); ++i)
        PXG_CHECK_MSG(got[i] == ref[i],
                      "pool byte %zu (body %zu, offset %zu): got 0x%02x, expected 0x%02x",
                      i, i / SZ_BODYSIM, i % SZ_BODYSIM, got[i], ref[i]);

    // Untouched pool slots must still carry their 0xB0.. pattern: the scatter goes
    // to the index the RECORD names, not to the loop index.
    for (uint32_t i = 0; i < POOL_BODIES; ++i)
    {
        bool isDst = false;
        for (uint32_t n = 0; n < NB_NEW; ++n) isDst = isDst || (DST[n] == i);
        if (isDst) continue;
        uint32_t w0 = 0; std::memcpy(&w0, got.data() + size_t(i) * SZ_BODYSIM, 4);
        PXG_CHECK_MSG(w0 == 0xB0000000u + i * 256u, "pool body %u was written but is not a destination", i);
    }
    // ...and record 1's body2World must actually have been re-derived, or the
    // `alignedTransformNotEqual` branch is untested.
    {
        const size_t dst = size_t(DST[1]) * SZ_BODYSIM + BS_BODY2WORLD;
        const F4 staged1 = loadF4At(staged, size_t(1) * SZ_BODYSIM + BS_BODY2WORLD);
        const F4 out1 = loadF4At(got, dst);
        PXG_CHECK_MSG(std::memcmp(&staged1, &out1, 16) != 0,
                      "record 1's body2World is the staged one -- it was not re-derived");
    }
}
