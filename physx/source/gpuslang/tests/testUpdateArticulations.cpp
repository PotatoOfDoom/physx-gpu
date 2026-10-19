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

// P8.124: engine-ABI fixture for updateArticulationsLaunch
// (updateArticulationsBda.slang) -- the per-frame dirty scatter.
//
// The kernel reads one PxgArticulationSimUpdate per warp, resolves the POOL
// record at update.articulationIndex, and copies exactly the dirty slices of
// the flattened new-* pools through the record's pointers.  The fixture
// drives SIXTEEN updates over two blocks (warps-per-block = 8 is
// load-bearing: the interesting warps 13..15 exist only in block 1's upper
// half); thirteen are all-clean fillers whose ONLY observable write is
// data.updateDirty = 0, and three exercise disjoint surfaces:
//
//   * A (pool slot 3, warp 14): JOINTS | MIMIC_JOINT | SPATIAL_TENDON |
//     SPATIAL_TENDON_ATTACHMENT | LINKS | POSITIONS | FORCES | TARGET_POS |
//     EXT_ACCEL | USER_FLAGS | IN_DIRTY_LIST -- with LINKS set, the
//     POSITIONS pose refresh must NOT fire (the else-if), while the dof
//     cursor advances POSITIONS -> FORCES -> TARGET_POS with VELOCITIES
//     skipped, so FORCES reads dofData[+5] and TARGET_POS dofData[+10];
//     two spatial tendons with 3+2 attachments pin the running mod cursor;
//     USER_FLAGS writes the pool byte pair; updateDirty must MASK OUT
//     FORCES and IN_DIRTY_LIST.
//   * B (pool slot 1, warp 15): ROOT_VELOCITIES | POSITIONS | VELOCITIES | TARGET_VEL |
//     WAKECOUNTER | FIXED_TENDON | FIXED_TENDON_JOINT -- links are copied
//     under ROOT_VELOCITIES but props/parents/children/body2World/
//     body2Actor must STAY POISON (the eDIRTY_LINKS-only block); two fixed
//     tendons with 2+1 tendon joints pin the coefficient cursor.
//   * C (pool slot 18, warp 13): POSITIONS only -- with neither LINKS nor
//     ROOT_VELOCITIES, the else-if pose refresh DOES fire (round 1).
//
// Round 2 repeats the same table with directAPI = true:
//   * the four tendon branches, the whole dof block and EXT_ACCEL are
//     skipped even where their flags are set (regions stay as seeded);
//   * A's body2Worlds go through the RECONCILIATION: link 0's new
//     body2Actor is bit-identical (no recompute), link 1 differs ONLY as
//     -0.0 vs +0.0 (PxTransform::operator== is float equality, so ALSO no
//     recompute -- a bitwise compare would recompute), links 2..5 differ
//     and are recomputed; the expectation is the ENGINE's own PxTransform
//     operator* / getInverse on the host, compared bit for bit;
//   * C degenerates to the lone updateDirty write (the else-if is
//     !directAPI-guarded);
//   * JOINTS, MIMIC_JOINT, WAKECOUNTER and USER_FLAGS still run.
//
// The CUDA's writes into its block-shared record copy (confiDirty under
// JOINTS, the five gpuDirtyFlag ORs) are DISCARDED shared memory in the
// engine too -- the pool-record comparison below proves the port writes
// exactly updateDirty (+ the USER_FLAGS pair) and nothing else.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "foundation/PxTransform.h"
#include "PxArticulationTendonData.h"
#include "PxgArticulation.h"
#include "PxgSimulationCoreDesc.h"
#include "PxgSimulationCoreKernelIndices.h"
#include "PxgSolverBody.h"
#include "DyArticulationCore.h"
#include "DyFeatherstoneArticulation.h"

#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

static std::vector<uint8_t> updArtiLoader(const char* file) { return loadShaderBlob(file); }

namespace pin
{
using namespace physx;
// the layout pins shared with newArticulationsLaunch live in
// testNewArticulations.cpp; here only what THIS kernel adds
static_assert(offsetof(PxgArticulationSimUpdate, spatialTendonAttachmentStartIndex) == 20,
              "attachmentStartIndex");
static_assert(offsetof(PxgArticulationSimUpdate, fixedTendonJointStartIndex) == 28,
              "tendonJointStartIndex");
static_assert(offsetof(PxgArticulationSimUpdate, userFlags) == 40, "PXG_SIM_UPDATE_USER_FLAGS");
static_assert(sizeof(PxgArticulationSimUpdate::userFlags) == 1, "userFlags byte");
static_assert(offsetof(PxgArticulationData, gpuDirtyFlag) == 8, "PXG_ARTI_DATA_gpuDirtyFlag");
static_assert(offsetof(PxgArticulationData, updateDirty) == 12, "PXG_ARTI_DATA_UPDATE_DIRTY");
static_assert(offsetof(PxgArticulationData, flags) == 26, "PXG_ARTI_DATA_FLAGS");
static_assert(offsetof(PxgArticulationData, confiDirty) == 27, "PXG_ARTI_DATA_CONFI_DIRTY");
static_assert(sizeof(PxgArticulationData::flags) == 1, "flags byte");
static_assert(sizeof(PxgArticulationData::confiDirty) == 1, "confiDirty byte");
static_assert(uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_JOINTS) == 1, "DIRTY_JOINTS");
static_assert(uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_LINKS) == 64, "DIRTY_LINKS");
static_assert(uint32_t(Dy::ArticulationDirtyFlag::eIN_DIRTY_LIST) == 128, "IN_DIRTY_LIST");
static_assert(uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_WAKECOUNTER) == 256, "DIRTY_WAKECOUNTER");
static_assert(uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_ROOT_VELOCITIES) == 32, "DIRTY_ROOT_VELOCITIES");
static_assert(uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_SPATIAL_TENDON) == 16384, "DIRTY_SPATIAL_TENDON");
static_assert(uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_SPATIAL_TENDON_ATTACHMENT) == 32768,
              "DIRTY_SPATIAL_TENDON_ATTACHMENT");
static_assert(uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_FIXED_TENDON) == 65536, "DIRTY_FIXED_TENDON");
static_assert(uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_FIXED_TENDON_JOINT) == 131072,
              "DIRTY_FIXED_TENDON_JOINT");
static_assert(uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_MIMIC_JOINT) == 262144, "DIRTY_MIMIC_JOINT");
static_assert(uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_USER_FLAGS) == 524288, "DIRTY_USER_FLAGS");
} // namespace pin

struct Arena
{
    std::vector<uint8_t> host;
    size_t cursor = 0;
    explicit Arena(size_t bytes) : host(bytes, 0xAB) {}
    size_t take(size_t bytes)
    {
        const size_t off = (cursor + 15u) & ~size_t(15);
        cursor = off + bytes;
        return off;
    }
};

static uint8_t pat(uint32_t poolId, size_t i) { return uint8_t(poolId * 37u + uint32_t(i) * 11u + 3u); }
static void patternFill(std::vector<uint8_t>& v, uint32_t poolId)
{
    for (size_t i = 0; i < v.size(); ++i)
        v[i] = pat(poolId, i);
}

struct ArtiSpec
{
    uint32_t poolIndex;
    uint32_t nbLinks, numDofs, nbSpatial, nbFixed, numMimic;
    uint32_t linkStart, dofStart, spatialStart, attachStart, fixedStart, tjointStart, mimicStart;
    uint32_t dirtyFlags;
    uint8_t userFlags;

    size_t oLinks, oWake, oProps, oParents, oChildren, oB2W, oB2A, oJoints, oJointData;
    size_t oSpatialTendons, oSpatialParams, oFixedTendons, oFixedParams;
    size_t oMimic;
    size_t oJointPos, oJointVel, oJointForce, oJointTgtPos, oJointTgtVel, oExtAccel;
};

// one full round: seed, dispatch, compare
static size_t runRound(PxgCudaShim& shim, bool directAPI)
{
    using namespace physx;

    // SIXTEEN updates: the three interesting ones sit at warp indices 13..15,
    // which only a correct warps-per-block of 8 reaches from block 1 (a port
    // with 4 computes globalWarpIndex 4..11 there and never processes them)
    const uint32_t NB_UPD = 16;
    std::vector<ArtiSpec> specs(NB_UPD);
    for (uint32_t i = 0; i < 13; ++i)
    {
        ArtiSpec& f = specs[i];
        std::memset(&f, 0, sizeof(f));
        f.poolIndex = 4 + i;   // slots 4..16
        f.nbLinks = 1; f.numDofs = 1;
        f.linkStart = i % 8u; f.dofStart = 0;   // flags 0: never read
        f.dirtyFlags = 0;
    }
    ArtiSpec& C = specs[13];
    std::memset(&C, 0, sizeof(C));
    C.poolIndex = 18; C.nbLinks = 3; C.numDofs = 3;
    C.linkStart = 0; C.dofStart = 21;
    C.dirtyFlags = uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_POSITIONS);

    ArtiSpec& A = specs[14];
    std::memset(&A, 0, sizeof(A));
    A.poolIndex = 3; A.nbLinks = 6; A.numDofs = 5;
    A.nbSpatial = 2; A.numMimic = 2;
    A.linkStart = 1; A.dofStart = 0;
    A.spatialStart = 1; A.attachStart = 5; A.mimicStart = 2;
    A.dirtyFlags = uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_JOINTS) |
                   uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_MIMIC_JOINT) |
                   uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_SPATIAL_TENDON) |
                   uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_SPATIAL_TENDON_ATTACHMENT) |
                   uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_LINKS) |
                   uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_POSITIONS) |
                   uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_FORCES) |
                   uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_JOINT_TARGET_POS) |
                   uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_EXT_ACCEL) |
                   uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_USER_FLAGS) |
                   uint32_t(Dy::ArticulationDirtyFlag::eIN_DIRTY_LIST);
    A.userFlags = 0x2A;

    ArtiSpec& B = specs[15];
    std::memset(&B, 0, sizeof(B));
    B.poolIndex = 1; B.nbLinks = 2; B.numDofs = 2;
    B.nbFixed = 2;
    B.linkStart = 5; B.dofStart = 15;
    B.fixedStart = 1; B.tjointStart = 2;
    // POSITIONS is deliberately SET on B while ROOT_VELOCITIES fires: the
    // else-if pose refresh must NOT run (B's body2Worlds stay poison) -- a
    // port that turns the else-if into a plain if copies them and dies here
    // (on A the else-if source equals the eDIRTY_LINKS copy, so only B can
    // see that mutant)
    B.dirtyFlags = uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_ROOT_VELOCITIES) |
                   uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_POSITIONS) |
                   uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_VELOCITIES) |
                   uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_JOINT_TARGET_VEL) |
                   uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_WAKECOUNTER) |
                   uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_FIXED_TENDON) |
                   uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_FIXED_TENDON_JOINT);

    // A's spatial tendon attachment counts (running cursor over 2 tendons);
    // B's fixed tendon joint counts
    const uint32_t A_ATT[2] = { 3, 2 };
    const uint32_t B_TJ[2] = { 2, 1 };

    // ---- destination arena ----
    Arena ar(256 * 1024);
    auto carve = [&ar](ArtiSpec& s) {
        const size_t guard = 64;
        s.oLinks = ar.take(80u * s.nbLinks + guard);
        s.oWake = ar.take(4u * s.nbLinks + guard);
        s.oProps = ar.take(16u * s.nbLinks + guard);
        s.oParents = ar.take(4u * s.nbLinks + guard);
        s.oChildren = ar.take(8u * s.nbLinks + guard);
        s.oB2W = ar.take(28u * s.nbLinks + guard);
        s.oB2A = ar.take(28u * s.nbLinks + guard);
        s.oJoints = ar.take(512u * s.nbLinks + guard);
        s.oJointData = ar.take(8u * s.nbLinks + guard);
        s.oSpatialTendons = ar.take(24u * s.nbSpatial + guard);
        s.oSpatialParams = ar.take(16u * s.nbSpatial + guard);
        s.oFixedTendons = ar.take(24u * s.nbFixed + guard);
        s.oFixedParams = ar.take(32u * s.nbFixed + guard);
        s.oMimic = ar.take(32u * s.numMimic + guard);
        s.oJointPos = ar.take(4u * s.numDofs + guard);
        s.oJointVel = ar.take(4u * s.numDofs + guard);
        s.oJointForce = ar.take(4u * s.numDofs + guard);
        s.oJointTgtPos = ar.take(4u * s.numDofs + guard);
        s.oJointTgtVel = ar.take(4u * s.numDofs + guard);
        s.oExtAccel = ar.take(24u * s.nbLinks + guard);
    };
    for (uint32_t i = 0; i < NB_UPD; ++i)
        carve(specs[i]);
    const size_t oAttachModDst0 = ar.take(32u * A_ATT[0] + 64);
    const size_t oAttachModDst1 = ar.take(32u * A_ATT[1] + 64);
    const size_t oTJCoeffDst0 = ar.take(16u * B_TJ[0] + 64);
    const size_t oTJCoeffDst1 = ar.take(16u * B_TJ[1] + 64);

    // A's OLD poses for the reconciliation (round 2 reads them; round 1
    // overwrites them with the pool copy).  Deterministic non-round floats.
    auto mkT = [](float s) {
        return PxTransform(PxVec3(0.25f + s, -0.5f + 2.0f * s, 1.0f - s),
                           PxQuat(0.1f + s, -0.2f + 0.5f * s, 0.3f - 0.25f * s, 0.9f + s));
    };
    std::vector<PxTransform> aOldB2W(A.nbLinks), aOldB2A(A.nbLinks), aNewB2A(A.nbLinks);
    for (uint32_t i = 0; i < A.nbLinks; ++i)
    {
        aOldB2W[i] = mkT(0.01f * float(i + 1));
        aOldB2A[i] = mkT(0.03f * float(i + 2));
        aNewB2A[i] = mkT(0.03f * float(i + 2));   // start identical
    }
    // link 0: bit-identical.  link 1: -0/+0 only (equal by float ==).
    aOldB2A[1].p.x = -0.0f;
    aNewB2A[1].p.x = 0.0f;
    // links 2..5: really different
    for (uint32_t i = 2; i < A.nbLinks; ++i)
        aNewB2A[i].p.y = aOldB2A[i].p.y + 0.25f * float(i);

    std::memcpy(&ar.host[A.oB2W], aOldB2W.data(), 28u * A.nbLinks);
    std::memcpy(&ar.host[A.oB2A], aOldB2A.data(), 28u * A.nbLinks);

    // ---- the flattened source pools ----
    const uint32_t LINK_POOL = 8;
    std::vector<uint8_t> pLinks(80u * LINK_POOL), pWake(4u * LINK_POOL), pProps(16u * LINK_POOL);
    std::vector<uint8_t> pParents(4u * LINK_POOL), pChildren(8u * LINK_POOL);
    std::vector<uint8_t> pJoints(512u * LINK_POOL), pJointData(8u * LINK_POOL);
    std::vector<uint8_t> pExtAccel(24u * LINK_POOL);
    patternFill(pLinks, 1); patternFill(pWake, 2); patternFill(pProps, 3);
    patternFill(pParents, 4); patternFill(pChildren, 5);
    patternFill(pJoints, 8); patternFill(pJointData, 9); patternFill(pExtAccel, 10);

    // body2World/body2Actor pools carry REAL transforms so both the plain
    // copy (round 1, byte-compared) and the reconciliation source (round 2)
    // are well-formed floats
    std::vector<PxTransform> pB2W(LINK_POOL), pB2A(LINK_POOL);
    for (uint32_t i = 0; i < LINK_POOL; ++i)
    {
        pB2W[i] = mkT(0.07f * float(i + 1));
        pB2A[i] = mkT(0.11f * float(i + 1));
    }
    // A's new body2Actors live at linkStart 1..6 of the pool
    for (uint32_t i = 0; i < A.nbLinks; ++i)
        pB2A[A.linkStart + i] = aNewB2A[i];

    std::vector<uint8_t> pSpatialParams(16u * 6), pFixedParams(32u * 6);
    std::vector<uint8_t> pAttachMod(32u * 16), pTJointCoeff(16u * 16);
    std::vector<uint8_t> pMimic(32u * 8);
    patternFill(pSpatialParams, 11); patternFill(pFixedParams, 12);
    patternFill(pAttachMod, 14); patternFill(pTJointCoeff, 16);
    patternFill(pMimic, 17);

    std::vector<float> dofData(24);
    for (size_t i = 0; i < dofData.size(); ++i)
        dofData[i] = 100.0f + float(i) * 0.5f;

    // ---- device allocations ----
    PxgDevicePtr dAr = 0, dPool = 0, dUpdates = 0, dDesc = 0, dDof = 0;
    PxgDevicePtr dLinks = 0, dWake = 0, dProps = 0, dParents = 0, dChildren = 0, dB2W = 0, dB2A = 0;
    PxgDevicePtr dJoints = 0, dJointData = 0, dExtAccel = 0;
    PxgDevicePtr dSpatialParams = 0, dFixedParams = 0, dAttachMod = 0, dTJointCoeff = 0, dMimic = 0;

    const uint32_t POOL_SLOTS = 19;   // slots 0, 2 and 17 stay poison
    shim.memAlloc(&dAr, ar.host.size());
    shim.memAlloc(&dPool, POOL_SLOTS * sizeof(PxgArticulation));
    shim.memAlloc(&dUpdates, NB_UPD * sizeof(PxgArticulationSimUpdate));
    shim.memAlloc(&dDesc, sizeof(PxgUpdateArticulationDesc));
    shim.memAlloc(&dDof, dofData.size() * 4);
    shim.memAlloc(&dLinks, pLinks.size());
    shim.memAlloc(&dWake, pWake.size());
    shim.memAlloc(&dProps, pProps.size());
    shim.memAlloc(&dParents, pParents.size());
    shim.memAlloc(&dChildren, pChildren.size());
    shim.memAlloc(&dB2W, pB2W.size() * sizeof(PxTransform));
    shim.memAlloc(&dB2A, pB2A.size() * sizeof(PxTransform));
    shim.memAlloc(&dJoints, pJoints.size());
    shim.memAlloc(&dJointData, pJointData.size());
    shim.memAlloc(&dExtAccel, pExtAccel.size());
    shim.memAlloc(&dSpatialParams, pSpatialParams.size());
    shim.memAlloc(&dFixedParams, pFixedParams.size());
    shim.memAlloc(&dAttachMod, pAttachMod.size());
    shim.memAlloc(&dTJointCoeff, pTJointCoeff.size());
    shim.memAlloc(&dMimic, pMimic.size());
    if (!dAr || !dPool || !dUpdates || !dDesc || !dDof || !dLinks || !dB2W || !dMimic)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return 0;
    }

    // ---- the tendon records inside the arena (destination state the kernel
    //      walks through): A's spatial tendons, B's fixed tendons ----
    {
        PxgArticulationTendon t[2];
        std::memset(t, 0xEE, sizeof(t));
        t[0].mModElements = reinterpret_cast<void*>(dAr + oAttachModDst0);
        t[0].mNbElements = A_ATT[0];
        t[1].mModElements = reinterpret_cast<void*>(dAr + oAttachModDst1);
        t[1].mNbElements = A_ATT[1];
        std::memcpy(&ar.host[A.oSpatialTendons], t, sizeof(t));

        PxgArticulationTendon u[2];
        std::memset(u, 0xEE, sizeof(u));
        u[0].mModElements = reinterpret_cast<void*>(dAr + oTJCoeffDst0);
        u[0].mNbElements = B_TJ[0];
        u[1].mModElements = reinterpret_cast<void*>(dAr + oTJCoeffDst1);
        u[1].mNbElements = B_TJ[1];
        std::memcpy(&ar.host[B.oFixedTendons], u, sizeof(u));
    }

    // ---- the POOL records: pointers into the arena, counts, byte poison in
    //      every field the kernel must not change ----
    auto stage = [&](const ArtiSpec& s, PxgArticulation& rec) {
        std::memset(&rec, 0xEE, sizeof(rec));
        rec.data.index = s.poolIndex;
        rec.data.numLinks = PxU16(s.nbLinks);
        rec.data.numJointDofs = PxU16(s.numDofs);
        rec.data.numSpatialTendons = PxU16(s.nbSpatial);
        rec.data.numFixedTendons = PxU16(s.nbFixed);
        rec.data.numMimicJoints = PxU16(s.numMimic);
        rec.data.numPathToRoots = 1;
        rec.data.updateDirty = 0xDDCCBBAAu;   // must be overwritten
        rec.data.flags = 0x11;
        rec.data.confiDirty = false;
        rec.data.gpuDirtyFlag = 0x77665544u;  // must SURVIVE (the ORs are dead)
        rec.links = reinterpret_cast<PxgArticulationLink*>(dAr + s.oLinks);
        rec.linkWakeCounters = reinterpret_cast<PxReal*>(dAr + s.oWake);
        rec.linkProps = reinterpret_cast<PxgArticulationLinkProp*>(dAr + s.oProps);
        rec.parents = reinterpret_cast<PxU32*>(dAr + s.oParents);
        rec.children = reinterpret_cast<Dy::ArticulationBitField*>(dAr + s.oChildren);
        rec.linkBody2Worlds = reinterpret_cast<PxTransform*>(dAr + s.oB2W);
        rec.linkBody2Actors = reinterpret_cast<PxTransform*>(dAr + s.oB2A);
        rec.joints = reinterpret_cast<Dy::ArticulationJointCore*>(dAr + s.oJoints);
        rec.jointData = reinterpret_cast<Dy::ArticulationJointCoreData*>(dAr + s.oJointData);
        rec.spatialTendons = reinterpret_cast<PxgArticulationTendon*>(dAr + s.oSpatialTendons);
        rec.spatialTendonParams = reinterpret_cast<PxGpuSpatialTendonData*>(dAr + s.oSpatialParams);
        rec.fixedTendons = reinterpret_cast<PxgArticulationTendon*>(dAr + s.oFixedTendons);
        rec.fixedTendonParams = reinterpret_cast<PxGpuFixedTendonData*>(dAr + s.oFixedParams);
        rec.mimicJointCores = reinterpret_cast<Dy::ArticulationMimicJointCore*>(dAr + s.oMimic);
        rec.jointPositions = reinterpret_cast<PxReal*>(dAr + s.oJointPos);
        rec.jointVelocities = reinterpret_cast<PxReal*>(dAr + s.oJointVel);
        rec.jointForce = reinterpret_cast<PxReal*>(dAr + s.oJointForce);
        rec.jointTargetPositions = reinterpret_cast<PxReal*>(dAr + s.oJointTgtPos);
        rec.jointTargetVelocities = reinterpret_cast<PxReal*>(dAr + s.oJointTgtVel);
        rec.externalAccelerations = reinterpret_cast<Cm::UnAlignedSpatialVector*>(dAr + s.oExtAccel);
    };
    std::vector<uint8_t> poolBytes(POOL_SLOTS * sizeof(PxgArticulation), 0xAB);
    std::vector<int> ownerOfSlot(POOL_SLOTS, -1);
    for (uint32_t i = 0; i < NB_UPD; ++i)
    {
        PXG_CHECK_MSG(ownerOfSlot[specs[i].poolIndex] < 0, "pool slot claimed twice");
        ownerOfSlot[specs[i].poolIndex] = int(i);
        PxgArticulation rec;
        stage(specs[i], rec);
        std::memcpy(&poolBytes[specs[i].poolIndex * sizeof(PxgArticulation)], &rec, sizeof(rec));
    }

    // ---- the sim-update records ----
    std::vector<PxgArticulationSimUpdate> updates(NB_UPD);
    std::memset(updates.data(), 0xEE, updates.size() * sizeof(PxgArticulationSimUpdate));
    for (uint32_t i = 0; i < NB_UPD; ++i)
    {
        const ArtiSpec& s = specs[i];
        PxgArticulationSimUpdate& u = updates[i];
        u.dirtyFlags = s.dirtyFlags;
        u.articulationIndex = s.poolIndex;
        u.linkStartIndex = s.linkStart;
        u.dofDataStartIndex = s.dofStart;
        u.spatialTendonStartIndex = s.spatialStart;
        u.spatialTendonAttachmentStartIndex = s.attachStart;
        u.fixedTendonStartIndex = s.fixedStart;
        u.fixedTendonJointStartIndex = s.tjointStart;
        u.mimicJointStartIndex = s.mimicStart;
        u.pathToRootIndex = 0;
        u.userFlags = s.userFlags;
    }

    // ---- the descriptor (fields this kernel never reads stay junk-poisoned
    //      so a stray read is loud) ----
    PxgUpdateArticulationDesc desc;
    std::memset(&desc, 0xEE, sizeof(desc));
    desc.mNewLinkWakeCounters = reinterpret_cast<PxReal*>(dWake);
    desc.mNewLinks = reinterpret_cast<PxgArticulationLink*>(dLinks);
    desc.mNewLinkProps = reinterpret_cast<PxgArticulationLinkProp*>(dProps);
    desc.mNewLinkParents = reinterpret_cast<PxU32*>(dParents);
    desc.mNewLinkChildren = reinterpret_cast<Dy::ArticulationBitField*>(dChildren);
    desc.mNewLinkBody2Worlds = reinterpret_cast<PxTransform*>(dB2W);
    desc.mNewLinkBody2Actors = reinterpret_cast<PxTransform*>(dB2A);
    desc.mNewLinkExtAccels = reinterpret_cast<Cm::UnAlignedSpatialVector*>(dExtAccel);
    desc.mNewJointCores = reinterpret_cast<Dy::ArticulationJointCore*>(dJoints);
    desc.mNewJointData = reinterpret_cast<Dy::ArticulationJointCoreData*>(dJointData);
    desc.mArticulationPool = reinterpret_cast<PxgArticulation*>(dPool);
    desc.mNewSpatialTendonParamsPool = reinterpret_cast<PxGpuSpatialTendonData*>(dSpatialParams);
    desc.mNewAttachmentModPool = reinterpret_cast<PxGpuTendonAttachmentData*>(dAttachMod);
    desc.mNewFixedTendonParamsPool = reinterpret_cast<PxGpuFixedTendonData*>(dFixedParams);
    desc.mNewTendonJointCoefficientPool =
        reinterpret_cast<PxGpuTendonJointCoefficientData*>(dTJointCoeff);
    desc.mNewArticulationMimicJointPool = reinterpret_cast<Dy::ArticulationMimicJointCore*>(dMimic);

    // ---- upload ----
    shim.memcpyHtoD(dAr, ar.host.data(), ar.host.size());
    shim.memcpyHtoD(dPool, poolBytes.data(), poolBytes.size());
    shim.memcpyHtoD(dUpdates, updates.data(), updates.size() * sizeof(PxgArticulationSimUpdate));
    shim.memcpyHtoD(dDesc, &desc, sizeof(desc));
    shim.memcpyHtoD(dDof, dofData.data(), dofData.size() * 4);
    shim.memcpyHtoD(dLinks, pLinks.data(), pLinks.size());
    shim.memcpyHtoD(dWake, pWake.data(), pWake.size());
    shim.memcpyHtoD(dProps, pProps.data(), pProps.size());
    shim.memcpyHtoD(dParents, pParents.data(), pParents.size());
    shim.memcpyHtoD(dChildren, pChildren.data(), pChildren.size());
    shim.memcpyHtoD(dB2W, pB2W.data(), pB2W.size() * sizeof(PxTransform));
    shim.memcpyHtoD(dB2A, pB2A.data(), pB2A.size() * sizeof(PxTransform));
    shim.memcpyHtoD(dJoints, pJoints.data(), pJoints.size());
    shim.memcpyHtoD(dJointData, pJointData.data(), pJointData.size());
    shim.memcpyHtoD(dExtAccel, pExtAccel.data(), pExtAccel.size());
    shim.memcpyHtoD(dSpatialParams, pSpatialParams.data(), pSpatialParams.size());
    shim.memcpyHtoD(dFixedParams, pFixedParams.data(), pFixedParams.size());
    shim.memcpyHtoD(dAttachMod, pAttachMod.data(), pAttachMod.size());
    shim.memcpyHtoD(dTJointCoeff, pTJointCoeff.data(), pTJointCoeff.size());
    shim.memcpyHtoD(dMimic, pMimic.data(), pMimic.size());

    // ---- dispatch: 2 blocks of (32, 8, 1), the engine grid ----
    const uint32_t warpsPerBlock = physx::PxgSimulationCoreKernelBlockDim::NEW_ARTICULATION / 32u;
    const uint32_t numBlocks = (NB_UPD + warpsPerBlock - 1u) / warpsPerBlock;
    PXG_CHECK_MSG(numBlocks == 2, "the fixture must span 2 blocks (got %u)", numBlocks);
    struct Params
    {
        uint64_t desc, simUpdates;
        uint32_t nb, pad0;
        uint64_t dofData;
        uint32_t directApi, pad1;
    } prm = { uint64_t(dDesc), uint64_t(dUpdates), NB_UPD, 0u,
              uint64_t(dDof), directAPI ? 1u : 0u, 0u };
    static_assert(sizeof(Params) == 40, "abi blob");
    PXG_CHECK_MSG(shim.launchKernel(0, numBlocks, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch updateArticulationsLaunch");
    shim.streamSynchronize();

    // ---- expected images ----
    std::vector<uint8_t> wantAr = ar.host;
    std::vector<uint8_t> wantPool = poolBytes;

    auto copyIn = [&](size_t dst, const void* src, size_t n) {
        std::memcpy(&wantAr[dst], src, n);
    };
    for (uint32_t i = 0; i < NB_UPD; ++i)
    {
        const ArtiSpec& s = specs[i];
        const uint32_t f = s.dirtyFlags;

        if (f & uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_JOINTS))
        {
            copyIn(s.oJoints, pJoints.data() + 512u * s.linkStart, 512u * s.nbLinks);
            copyIn(s.oJointData, pJointData.data() + 8u * s.linkStart, 8u * s.nbLinks);
        }
        if (f & uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_MIMIC_JOINT))
            copyIn(s.oMimic, pMimic.data() + 32u * s.mimicStart, 32u * s.numMimic);

        if (!directAPI)
        {
            if (f & uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_SPATIAL_TENDON))
                copyIn(s.oSpatialParams, pSpatialParams.data() + 16u * s.spatialStart,
                       16u * s.nbSpatial);
            if ((f & uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_SPATIAL_TENDON_ATTACHMENT)) &&
                s.nbSpatial == 2)
            {
                copyIn(oAttachModDst0, pAttachMod.data() + 32u * s.attachStart, 32u * A_ATT[0]);
                copyIn(oAttachModDst1, pAttachMod.data() + 32u * (s.attachStart + A_ATT[0]),
                       32u * A_ATT[1]);
            }
            if (f & uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_FIXED_TENDON))
                copyIn(s.oFixedParams, pFixedParams.data() + 32u * s.fixedStart, 32u * s.nbFixed);
            if ((f & uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_FIXED_TENDON_JOINT)) &&
                s.nbFixed == 2)
            {
                copyIn(oTJCoeffDst0, pTJointCoeff.data() + 16u * s.tjointStart, 16u * B_TJ[0]);
                copyIn(oTJCoeffDst1, pTJointCoeff.data() + 16u * (s.tjointStart + B_TJ[0]),
                       16u * B_TJ[1]);
            }
        }

        const bool linksOrRoot =
            (f & (uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_LINKS) |
                  uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_ROOT_VELOCITIES))) != 0;
        if (linksOrRoot)
        {
            copyIn(s.oLinks, pLinks.data() + 80u * s.linkStart, 80u * s.nbLinks);
            if (f & uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_LINKS))
            {
                copyIn(s.oProps, pProps.data() + 16u * s.linkStart, 16u * s.nbLinks);
                copyIn(s.oParents, pParents.data() + 4u * s.linkStart, 4u * s.nbLinks);
                copyIn(s.oChildren, pChildren.data() + 8u * s.linkStart, 8u * s.nbLinks);
                if (directAPI)
                {
                    // the reconciliation, replayed with the ENGINE's own math
                    for (uint32_t l = 0; l < s.nbLinks; ++l)
                    {
                        PxTransform oldB2A, newB2A, b2w;
                        std::memcpy(&oldB2A, &ar.host[s.oB2A + 28u * l], 28);
                        std::memcpy(&newB2A, &pB2A[s.linkStart + l], 28);
                        std::memcpy(&b2w, &ar.host[s.oB2W + 28u * l], 28);
                        if (!(oldB2A == newB2A))
                        {
                            const PxTransform actor2World = b2w * oldB2A.getInverse();
                            const PxTransform out = actor2World * newB2A;
                            copyIn(s.oB2W + 28u * l, &out, 28);
                        }
                    }
                }
                else
                {
                    copyIn(s.oB2W, pB2W.data() + s.linkStart, 28u * s.nbLinks);
                }
                copyIn(s.oB2A, pB2A.data() + s.linkStart, 28u * s.nbLinks);
            }
        }
        else if ((f & uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_POSITIONS)) && !directAPI)
        {
            copyIn(s.oB2W, pB2W.data() + s.linkStart, 28u * s.nbLinks);
        }

        if (!directAPI)
        {
            uint32_t cursor = s.dofStart;
            const uint32_t bits[5] = {
                uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_POSITIONS),
                uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_VELOCITIES),
                uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_FORCES),
                uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_JOINT_TARGET_POS),
                uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_JOINT_TARGET_VEL),
            };
            const size_t offs[5] = { s.oJointPos, s.oJointVel, s.oJointForce,
                                     s.oJointTgtPos, s.oJointTgtVel };
            for (int k = 0; k < 5; ++k)
                if (f & bits[k])
                {
                    copyIn(offs[k], dofData.data() + cursor, 4u * s.numDofs);
                    cursor += s.numDofs;
                }
        }

        if (f & uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_WAKECOUNTER))
            copyIn(s.oWake, pWake.data() + 4u * s.linkStart, 4u * s.nbLinks);

        if ((f & uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_EXT_ACCEL)) && !directAPI)
            copyIn(s.oExtAccel, pExtAccel.data() + 24u * s.linkStart, 24u * s.nbLinks);

        // the pool record: updateDirty always; the USER_FLAGS byte pair
        uint8_t* rec = &wantPool[s.poolIndex * sizeof(PxgArticulation)];
        const uint32_t updateDirty =
            f & ~(uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_FORCES) |
                  uint32_t(Dy::ArticulationDirtyFlag::eIN_DIRTY_LIST));
        std::memcpy(rec + offsetof(PxgArticulationData, updateDirty), &updateDirty, 4);
        if (f & uint32_t(Dy::ArticulationDirtyFlag::eDIRTY_USER_FLAGS))
        {
            rec[offsetof(PxgArticulationData, flags)] = s.userFlags;
            rec[offsetof(PxgArticulationData, confiDirty)] = 1;
        }
    }

    // ---- read back + byte compare ----
    std::vector<uint8_t> gotAr(ar.host.size()), gotPool(poolBytes.size());
    shim.memcpyDtoH(gotAr.data(), dAr, gotAr.size());
    shim.memcpyDtoH(gotPool.data(), dPool, gotPool.size());

    size_t bad = 0;
    for (size_t i = 0; i < gotAr.size(); ++i)
        if (gotAr[i] != wantAr[i])
        {
            if (bad < 12)
                std::printf("  arena[%zu]: got 0x%02X want 0x%02X (directAPI=%d)\n",
                            i, gotAr[i], wantAr[i], int(directAPI));
            ++bad;
        }
    for (size_t i = 0; i < gotPool.size(); ++i)
        if (gotPool[i] != wantPool[i])
        {
            if (bad < 12)
                std::printf("  pool[%zu] (slot %zu byte %zu): got 0x%02X want 0x%02X (directAPI=%d)\n",
                            i, i / sizeof(PxgArticulation), i % sizeof(PxgArticulation),
                            gotPool[i], wantPool[i], int(directAPI));
            ++bad;
        }

    shim.memFree(dAr); shim.memFree(dPool); shim.memFree(dUpdates); shim.memFree(dDesc);
    shim.memFree(dDof); shim.memFree(dLinks); shim.memFree(dWake); shim.memFree(dProps);
    shim.memFree(dParents); shim.memFree(dChildren); shim.memFree(dB2W); shim.memFree(dB2A);
    shim.memFree(dJoints); shim.memFree(dJointData); shim.memFree(dExtAccel);
    shim.memFree(dSpatialParams); shim.memFree(dFixedParams); shim.memFree(dAttachMod);
    shim.memFree(dTJointCoeff); shim.memFree(dMimic);

    std::printf("  updateArticulations(directAPI=%d): 16 updates over 2 blocks -- %s\n",
                int(directAPI), bad == 0 ? "all regions exact" : "MISMATCH");
    return bad;
}

} // namespace

PXG_TEST(updateArticulations)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "updateArticulationsBda.spv", "updateArticulationsLaunch", 40u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &updArtiLoader);

    size_t bad = 0;
    bad += runRound(shim, false);
    bad += runRound(shim, true);
    PXG_CHECK_MSG(bad == 0, "%zu byte(s) wrong", bad);
}
