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

// P8.76: jcalc phase C -- the `else if (dataDirty)` refresh.
//
// Phase C runs when the block LAYOUT is already valid and only values changed.  It
// is not a cheaper phase B: it writes a strict SUBSET of the same fields, and the
// subset is the content of this test.  Phase C must leave
//
//     mRetainsAcceleration, mDofs, mJointType, mJointOffset, mParents,
//     mChildrenOffset, mNumChildren, mInvDofIds, mDofIds, mMotion (5.12: no mJointAxis)
//
// untouched -- all of them fields phase B writes.  Since the destinations are
// poisoned and compared byte for byte, an extra write is a failure here, which is
// the only way this asymmetry can be checked at all.
//
// Four sub-cases, and each one exists to kill a specific way of getting it wrong:
//
//   1. eDIRTY_POSITIONS, directAPI = false   -- the full refresh
//   2. eDIRTY_POSITIONS, directAPI = true    -- identical EXCEPT the root motion
//      velocity, which must not be written; under the direct GPU state API the
//      block holds the authoritative root velocity and the record is stale
//   3. eDIRTY_SPATIAL_TENDON alone           -- the tendon bits are masked out
//      before asking whether anything is left to do, so NOTHING is written; a port
//      that forgot the mask does a full refresh here and fails
//   4. no dirty bits at all                  -- nothing written
//
// In every case updateDirty must come back ZERO: phase A deliberately leaves it
// set and phase C is the consumer that clears it.
//
// Oracle caveat, unchanged from P8.73-P8.75: jcalc is __device__-only, so the
// expectation is the fixture's.  Generated offsets, distinct values everywhere,
// whole-region poison comparison and a mutation sweep are what carry it.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "PxArticulationTendonData.h"
#include "PxgArticulation.h"
#include "PxgArticulationLink.h"
#include "DyArticulationJointCore.h"
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

static std::vector<uint8_t> jcalcCLoader(const char* file) { return loadShaderBlob(file); }

static const uint32_t LANES = 5;
static const uint32_t NB_LINKS = 3;
static const uint32_t MAX_DOFS = 4;
static const uint32_t GUARD = 128;

static_assert(uint32_t(physx::Dy::ArticulationDirtyFlag::eDIRTY_SPATIAL_TENDON) == 16384,
              "PXG_ARTI_DIRTY_SPATIAL_TENDON");
static_assert(uint32_t(physx::Dy::ArticulationDirtyFlag::eDIRTY_SPATIAL_TENDON_ATTACHMENT) == 32768,
              "PXG_ARTI_DIRTY_SPATIAL_TENDON_ATTACHMENT");
static_assert(uint32_t(physx::Dy::ArticulationDirtyFlag::eDIRTY_FIXED_TENDON) == 65536,
              "PXG_ARTI_DIRTY_FIXED_TENDON");
static_assert(uint32_t(physx::Dy::ArticulationDirtyFlag::eDIRTY_FIXED_TENDON_JOINT) == 131072,
              "PXG_ARTI_DIRTY_FIXED_TENDON_JOINT");
static_assert(uint32_t(physx::Dy::ArticulationDirtyFlag::eDIRTY_MIMIC_JOINT) == 262144,
              "PXG_ARTI_DIRTY_MIMIC_JOINT");

PXG_TEST(artiJcalcPhaseC)
{
    using namespace physx;
    using namespace physx::Dy;

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiJointFrame.spv", "artiJcalcPhaseC", 32u, 0 }, // { u64, u64, u64, u32, u32 }
    };
    PxgCudaShim shim(dev, manifest, 1, &jcalcCLoader);

    const uint32_t nbDof[NB_LINKS] = { 0u, 2u, 1u };
    const uint32_t jointOffset[NB_LINKS] = { 0u, 0u, 2u };
    const uint8_t dofIds[NB_LINKS][3] = { { 0, 0, 0 }, { 1, 2, 0 }, { 4, 0, 0 } };
    const uint32_t TOTAL_DOFS = 3;

    std::vector<PxgArticulationLink> links(size_t(LANES) * NB_LINKS);
    std::memset(links.data(), 0xEE, links.size() * sizeof(PxgArticulationLink));
    std::vector<PxTransform> body2Worlds(size_t(LANES) * NB_LINKS);
    std::vector<PxQuat> relativeQuats(size_t(LANES) * NB_LINKS);
    std::vector<ArticulationJointCoreData> jdata(size_t(LANES) * NB_LINKS);
    std::memset(jdata.data(), 0xEE, jdata.size() * sizeof(ArticulationJointCoreData));
    std::vector<ArticulationJointCore> joints;
    joints.reserve(size_t(LANES) * NB_LINKS);
    std::vector<SpatialSubspaceMatrix> motionMatrix(size_t(LANES) * NB_LINKS);
    std::vector<float> jPos(size_t(LANES) * MAX_DOFS), jVel(size_t(LANES) * MAX_DOFS);
    std::vector<float> jTPos(size_t(LANES) * MAX_DOFS), jTVel(size_t(LANES) * MAX_DOFS);

    auto val = [](uint32_t L, uint32_t link, uint32_t k, uint32_t field) {
        return float(1000 * (L + 1) + 100 * (link + 1) + 10 * (k + 1)) + 0.125f * field;
    };

    for (uint32_t L = 0; L < LANES; ++L)
    {
        for (uint32_t link = 0; link < NB_LINKS; ++link)
        {
            const size_t k = size_t(L) * NB_LINKS + link;
            jdata[k].nbDof = PxU8(nbDof[link]);
            jdata[k].jointOffset = jointOffset[link];

            const PxTransform idT(PxIdentity);   // named: `j(PxTransform(...), ...)` parses
            ArticulationJointCore j(idT, idT);    // as a function declaration

            for (uint32_t ax = 0; ax < PxArticulationAxis::eCOUNT; ++ax)
            {
                j.drives[ax].stiffness = val(L, link, ax, 1);
                j.drives[ax].damping = val(L, link, ax, 2);
                j.drives[ax].envelope.maxEffort = val(L, link, ax, 4);
                j.drives[ax].envelope.maxActuatorVelocity = val(L, link, ax, 5);
                j.drives[ax].envelope.velocityDependentResistance = val(L, link, ax, 6);
                j.drives[ax].envelope.speedEffortGradient = val(L, link, ax, 7);
                j.drives[ax].driveType = PxArticulationDriveType::Enum(ax % 3);
                j.limits[ax].low = val(L, link, ax, 8);
                j.limits[ax].high = val(L, link, ax, 9);
                j.armature[ax] = val(L, link, ax, 10);
                j.frictionParams[ax].staticFrictionEffort = val(L, link, ax, 11);
                j.frictionParams[ax].dynamicFrictionEffort = val(L, link, ax, 12);
                j.frictionParams[ax].viscousFrictionCoefficient = val(L, link, ax, 13);
                j.maxJointVelocity[ax] = val(L, link, ax, 14);
                j.motion[ax] = PxU8(1u + ((ax + L) % 3u));
                j.dofIds[ax] = 0xFF;
            }
            for (uint32_t i = 0; i < nbDof[link]; ++i)
                j.dofIds[i] = dofIds[link][i];
            joints.push_back(j);

            for (uint32_t c = 0; c < 3; ++c)
            {
                Cm::UnAlignedSpatialVector& mm = motionMatrix[k].columns[c];
                mm.top = PxVec3(val(L, link, c, 20), val(L, link, c, 21), val(L, link, c, 22));
                mm.bottom = PxVec3(val(L, link, c, 23), val(L, link, c, 24), val(L, link, c, 25));
            }
        }
        for (uint32_t d = 0; d < MAX_DOFS; ++d)
        {
            const size_t k = size_t(L) * MAX_DOFS + d;
            jPos[k] = val(L, 9, d, 40);
            jVel[k] = val(L, 9, d, 41);
            jTPos[k] = val(L, 9, d, 42);
            jTVel[k] = val(L, 9, d, 43);
        }
    }


    for (uint32_t L = 0; L < LANES; ++L)
    {
        for (uint32_t link = 0; link < NB_LINKS; ++link)
        {
            const size_t k = size_t(L) * NB_LINKS + link;
            PxgArticulationLink& lk = links[k];
            lk.initialAngVel = PxVec3(val(L, link, 0, 50), val(L, link, 0, 51), val(L, link, 0, 52));
            lk.penBiasClamp = val(L, link, 0, 53);
            lk.initialLinVel = PxVec3(val(L, link, 0, 54), val(L, link, 0, 55), val(L, link, 0, 56));
            lk.invMass = val(L, link, 0, 57);
            lk.linearDamping = val(L, link, 0, 60);
            lk.angularDamping = val(L, link, 0, 61);
            lk.maxLinearVelocitySq = val(L, link, 0, 62);
            lk.maxAngularVelocitySq = val(L, link, 0, 63);
            lk.disableGravity = ((L + link) & 1) != 0;
            lk.retainsAccelerations = ((L + link) & 2) != 0;
            relativeQuats[k] = PxQuat(val(L, link, 0, 70), val(L, link, 0, 71),
                                      val(L, link, 0, 72), val(L, link, 0, 73));
            body2Worlds[k] = PxTransform(
                PxVec3(val(L, link, 0, 74), val(L, link, 0, 75), val(L, link, 0, 76)),
                PxQuat(0.2f, -0.4f, 0.5f, 0.7f).getNormalized());
        }
    }

    const size_t LINK_BLOCK = sizeof(PxgArticulationBlockLinkData);
    const size_t DOF_BLOCK = sizeof(PxgArticulationBlockDofData);
    const size_t LINK_REGION = size_t(NB_LINKS) * LINK_BLOCK + GUARD;
    const size_t DOF_REGION = size_t(MAX_DOFS) * DOF_BLOCK + GUARD;

    PxgDevicePtr dLinks = 0, dJData = 0, dJoints = 0, dMM = 0, dRQ = 0, dB2W = 0;
    PxgDevicePtr dPos = 0, dVel = 0, dTPos = 0, dTVel = 0;
    PxgDevicePtr dArtis = 0, dLinkBlocks = 0, dDofBlocks = 0;
    shim.memAlloc(&dLinks, links.size() * sizeof(PxgArticulationLink));
    shim.memAlloc(&dJData, jdata.size() * sizeof(ArticulationJointCoreData));
    shim.memAlloc(&dJoints, joints.size() * sizeof(ArticulationJointCore));
    shim.memAlloc(&dMM, motionMatrix.size() * sizeof(SpatialSubspaceMatrix));
    shim.memAlloc(&dRQ, relativeQuats.size() * sizeof(PxQuat));
    shim.memAlloc(&dB2W, body2Worlds.size() * sizeof(PxTransform));
    shim.memAlloc(&dPos, jPos.size() * 4);
    shim.memAlloc(&dVel, jVel.size() * 4);
    shim.memAlloc(&dTPos, jTPos.size() * 4);
    shim.memAlloc(&dTVel, jTVel.size() * 4);
    shim.memAlloc(&dArtis, size_t(LANES) * sizeof(PxgArticulation));
    shim.memAlloc(&dLinkBlocks, LINK_REGION);
    shim.memAlloc(&dDofBlocks, DOF_REGION);
    if (!dLinks || !dJData || !dJoints || !dMM || !dRQ || !dB2W || !dPos || !dVel || !dTPos ||
        !dTVel || !dArtis || !dLinkBlocks || !dDofBlocks)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }

    std::vector<PxgArticulation> recs(LANES);
    std::memset(recs.data(), 0, recs.size() * sizeof(PxgArticulation));
    for (uint32_t L = 0; L < LANES; ++L)
    {
        const size_t l = size_t(L) * NB_LINKS;
        const size_t d = size_t(L) * MAX_DOFS;
        recs[L].data.numLinks = PxU16(NB_LINKS);
        recs[L].data.numJointDofs = PxU16(TOTAL_DOFS);
        recs[L].links = reinterpret_cast<PxgArticulationLink*>(
            dLinks + PxgDevicePtr(l * sizeof(PxgArticulationLink)));
        recs[L].jointData = reinterpret_cast<ArticulationJointCoreData*>(
            dJData + PxgDevicePtr(l * sizeof(ArticulationJointCoreData)));
        recs[L].joints = reinterpret_cast<ArticulationJointCore*>(
            dJoints + PxgDevicePtr(l * sizeof(ArticulationJointCore)));
        recs[L].motionMatrix = reinterpret_cast<SpatialSubspaceMatrix*>(
            dMM + PxgDevicePtr(l * sizeof(SpatialSubspaceMatrix)));
        recs[L].relativeQuat = reinterpret_cast<PxQuat*>(dRQ + PxgDevicePtr(l * sizeof(PxQuat)));
        recs[L].linkBody2Worlds = reinterpret_cast<PxTransform*>(
            dB2W + PxgDevicePtr(l * sizeof(PxTransform)));
        recs[L].jointPositions = reinterpret_cast<PxReal*>(dPos + PxgDevicePtr(d * 4));
        recs[L].jointVelocities = reinterpret_cast<PxReal*>(dVel + PxgDevicePtr(d * 4));
        recs[L].jointTargetPositions = reinterpret_cast<PxReal*>(dTPos + PxgDevicePtr(d * 4));
        recs[L].jointTargetVelocities = reinterpret_cast<PxReal*>(dTVel + PxgDevicePtr(d * 4));
    }

    shim.memcpyHtoD(dLinks, links.data(), links.size() * sizeof(PxgArticulationLink));
    shim.memcpyHtoD(dJData, jdata.data(), jdata.size() * sizeof(ArticulationJointCoreData));
    shim.memcpyHtoD(dJoints, joints.data(), joints.size() * sizeof(ArticulationJointCore));
    shim.memcpyHtoD(dMM, motionMatrix.data(), motionMatrix.size() * sizeof(SpatialSubspaceMatrix));
    shim.memcpyHtoD(dRQ, relativeQuats.data(), relativeQuats.size() * sizeof(PxQuat));
    shim.memcpyHtoD(dB2W, body2Worlds.data(), body2Worlds.size() * sizeof(PxTransform));
    shim.memcpyHtoD(dPos, jPos.data(), jPos.size() * 4);
    shim.memcpyHtoD(dVel, jVel.data(), jVel.size() * 4);
    shim.memcpyHtoD(dTPos, jTPos.data(), jTPos.size() * 4);
    shim.memcpyHtoD(dTVel, jTVel.data(), jTVel.size() * 4);

    struct Sub { const char* name; uint32_t dirty; bool directAPI; };
    const Sub subs[] = {
        { "positions dirty, not direct-API", uint32_t(ArticulationDirtyFlag::eDIRTY_POSITIONS), false },
        { "positions dirty, direct-API",     uint32_t(ArticulationDirtyFlag::eDIRTY_POSITIONS), true  },
        { "spatial tendon dirty only",       uint32_t(ArticulationDirtyFlag::eDIRTY_SPATIAL_TENDON), false },
        { "nothing dirty",                   0u, false },
    };

    size_t bad = 0;
    const size_t CD = offsetof(PxgArticulationBlockDofData, mConstraintData);

    for (const Sub& sub : subs)
    {
        for (uint32_t L = 0; L < LANES; ++L)
            recs[L].data.updateDirty = sub.dirty;
        shim.memcpyHtoD(dArtis, recs.data(), recs.size() * sizeof(PxgArticulation));

        std::vector<uint8_t> poisonLink(LINK_REGION, 0xAB), poisonDof(DOF_REGION, 0xAB);
        shim.memcpyHtoD(dLinkBlocks, poisonLink.data(), poisonLink.size());
        shim.memcpyHtoD(dDofBlocks, poisonDof.data(), poisonDof.size());

        struct Params
        {
            uint64_t artis;
            uint64_t linkBlocks;
            uint64_t dofBlocks;
            uint32_t nbArticulations;
            uint32_t directAPI;
        } prm = { uint64_t(dArtis), uint64_t(dLinkBlocks), uint64_t(dDofBlocks), LANES,
                  sub.directAPI ? 1u : 0u };

        PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                      "launch artiJcalcPhaseC");
        shim.streamSynchronize();

        std::vector<uint8_t> wantLink = poisonLink, wantDof = poisonDof;
        auto putU8 = [](std::vector<uint8_t>& img, size_t off, uint8_t v) { img[off] = v; };
        auto putF = [](std::vector<uint8_t>& img, size_t off, float v) {
            std::memcpy(img.data() + off, &v, 4);
        };
        auto putU32 = [](std::vector<uint8_t>& img, size_t off, uint32_t v) {
            std::memcpy(img.data() + off, &v, 4);
        };
        auto putSV = [&](std::vector<uint8_t>& img, size_t base, size_t field, uint32_t L,
                         const PxVec3& top, const PxVec3& bottom) {
            putF(img, base + field + 0 + size_t(L) * 16 + 0, top.x);
            putF(img, base + field + 0 + size_t(L) * 16 + 4, top.y);
            putF(img, base + field + 0 + size_t(L) * 16 + 8, top.z);
            putF(img, base + field + 0 + size_t(L) * 16 + 12, bottom.x);
            putF(img, base + field + 512 + size_t(L) * 8 + 0, bottom.y);
            putF(img, base + field + 512 + size_t(L) * 8 + 4, bottom.z);
        };
        auto putPose = [&](std::vector<uint8_t>& img, size_t base, size_t field, uint32_t L,
                           const PxTransform& t) {
            putF(img, base + field + 0 + size_t(L) * 16 + 0, t.q.x);
            putF(img, base + field + 0 + size_t(L) * 16 + 4, t.q.y);
            putF(img, base + field + 0 + size_t(L) * 16 + 8, t.q.z);
            putF(img, base + field + 0 + size_t(L) * 16 + 12, t.q.w);
            putF(img, base + field + 512 + size_t(L) * 16 + 0, t.p.x);
            putF(img, base + field + 512 + size_t(L) * 16 + 4, t.p.y);
            putF(img, base + field + 512 + size_t(L) * 16 + 8, t.p.z);
            putF(img, base + field + 512 + size_t(L) * 16 + 12, 0.0f);
        };
        auto putDamp = [&](std::vector<uint8_t>& img, size_t blk, uint32_t L,
                           const PxgArticulationLink& lk) {
            const size_t d = blk + offsetof(PxgArticulationBlockLinkData,
                                            mLinDampingX_AngDampingY_maxLinVelSqZ_maxAngVelSqW) +
                             size_t(L) * 16;
            putF(img, d + 0, lk.linearDamping);
            putF(img, d + 4, lk.angularDamping);
            putF(img, d + 8, lk.maxLinearVelocitySq);
            putF(img, d + 12, lk.maxAngularVelocitySq);
        };

        const uint32_t TENDON_BITS =
            uint32_t(ArticulationDirtyFlag::eDIRTY_SPATIAL_TENDON) |
            uint32_t(ArticulationDirtyFlag::eDIRTY_SPATIAL_TENDON_ATTACHMENT) |
            uint32_t(ArticulationDirtyFlag::eDIRTY_FIXED_TENDON) |
            uint32_t(ArticulationDirtyFlag::eDIRTY_FIXED_TENDON_JOINT);
        const bool refreshes = (sub.dirty & ~TENDON_BITS) != 0u;

        if (refreshes)
        {
            for (uint32_t L = 0; L < LANES; ++L)
            {
                const PxgArticulationLink& rootLk = links[size_t(L) * NB_LINKS];
                putU8(wantLink, offsetof(PxgArticulationBlockLinkData, mDisableGravity) + L,
                      rootLk.disableGravity ? 1u : 0u);
                putDamp(wantLink, 0, L, rootLk);
                if (!sub.directAPI)
                    putSV(wantLink, 0, offsetof(PxgArticulationBlockLinkData, mMotionVelocity), L,
                          rootLk.initialAngVel, rootLk.initialLinVel);
                putPose(wantLink, 0, offsetof(PxgArticulationBlockLinkData, mAccumulatedPose), L,
                        body2Worlds[size_t(L) * NB_LINKS]);

                for (uint32_t link = 1; link < NB_LINKS; ++link)
                {
                    const size_t k = size_t(L) * NB_LINKS + link;
                    const ArticulationJointCore& j = joints[k];
                    const size_t lblk = size_t(link) * LINK_BLOCK;

                    for (uint32_t i = 0; i < nbDof[link]; ++i)
                    {
                        const uint32_t slot = jointOffset[link] + i;
                        const uint32_t dofId = dofIds[link][i];
                        const size_t db = size_t(slot) * DOF_BLOCK;
                        const size_t ic = db + CD;
                        const size_t ak = size_t(L) * MAX_DOFS + slot;
                        typedef PxgArticulationInternalConstraintData IC;

                        putF(wantDof, ic + offsetof(IC, mDriveStiffness) + size_t(L) * 4,
                             j.drives[dofId].stiffness);
                        putF(wantDof, ic + offsetof(IC, mDamping) + size_t(L) * 4,
                             j.drives[dofId].damping);
                        putF(wantDof, ic + offsetof(IC, mDriveTargetVel) + size_t(L) * 4, jTVel[ak]);
                        putF(wantDof, ic + offsetof(IC, mTargetPosBias) + size_t(L) * 4, 0.0f);
                        putF(wantDof, ic + offsetof(IC, mArmature) + size_t(L) * 4,
                             j.armature[dofId]);
                        const size_t lim =
                            ic + offsetof(IC, mLimits_LowLimitX_highLimitY) + size_t(L) * 8;
                        putF(wantDof, lim + 0, j.limits[dofId].low);
                        putF(wantDof, lim + 4, j.limits[dofId].high);
                        putF(wantDof, ic + offsetof(IC, mStaticFrictionEffort) + size_t(L) * 4,
                             j.frictionParams[dofId].staticFrictionEffort);
                        putF(wantDof, ic + offsetof(IC, mDynamicFrictionEffort) + size_t(L) * 4,
                             j.frictionParams[dofId].dynamicFrictionEffort);
                        putF(wantDof, ic + offsetof(IC, mViscousFrictionCoefficient) + size_t(L) * 4,
                             j.frictionParams[dofId].viscousFrictionCoefficient);
                        putF(wantDof, ic + offsetof(IC, mMaxEffort) + size_t(L) * 4,
                             j.drives[dofId].envelope.maxEffort);
                        putF(wantDof, ic + offsetof(IC, mMaxActuatorVelocity) + size_t(L) * 4,
                             j.drives[dofId].envelope.maxActuatorVelocity);
                        putF(wantDof, ic + offsetof(IC, mVelocityDependentResistance) + size_t(L) * 4,
                             j.drives[dofId].envelope.velocityDependentResistance);
                        putF(wantDof, ic + offsetof(IC, mSpeedEffortGradient) + size_t(L) * 4,
                             j.drives[dofId].envelope.speedEffortGradient);
                        putU32(wantDof, ic + offsetof(IC, mDriveType) + size_t(L) * 4,
                               uint32_t(j.drives[dofId].driveType));
                        putF(wantDof, ic + offsetof(IC, mMaxJointVelocity) + size_t(L) * 4,
                             j.maxJointVelocity[dofId]);

                        putF(wantDof, db + offsetof(PxgArticulationBlockDofData, mJointPositions) +
                                          size_t(L) * 4, jPos[ak]);
                        putF(wantDof, db + offsetof(PxgArticulationBlockDofData, mJointVelocities) +
                                          size_t(L) * 4, jVel[ak]);
                        putF(wantDof, ic + offsetof(IC, mDriveTargetPos) + size_t(L) * 4, jTPos[ak]);
                        putSV(wantDof, db,
                              offsetof(PxgArticulationBlockDofData, mLocalMotionMatrix), L,
                              motionMatrix[k].columns[i].top, motionMatrix[k].columns[i].bottom);
                    }

                    const size_t rq = lblk +
                        offsetof(PxgArticulationBlockLinkData, mRelativeQuat) + size_t(L) * 16;
                    putF(wantLink, rq + 0, relativeQuats[k].x);
                    putF(wantLink, rq + 4, relativeQuats[k].y);
                    putF(wantLink, rq + 8, relativeQuats[k].z);
                    putF(wantLink, rq + 12, relativeQuats[k].w);
                    putDamp(wantLink, lblk, L, links[k]);
                    putU8(wantLink, lblk + offsetof(PxgArticulationBlockLinkData, mDisableGravity) + L,
                          links[k].disableGravity ? 1u : 0u);
                    putPose(wantLink, lblk, offsetof(PxgArticulationBlockLinkData, mChildPose), L,
                            j.childPose);
                    putPose(wantLink, lblk, offsetof(PxgArticulationBlockLinkData, mParentPose), L,
                            j.parentPose);
                    putPose(wantLink, lblk, offsetof(PxgArticulationBlockLinkData, mAccumulatedPose),
                            L, body2Worlds[k]);
                }
            }
        }

        auto region = [&](const char* what, PxgDevicePtr p, const std::vector<uint8_t>& want,
                          size_t stride) {
            std::vector<uint8_t> got(want.size(), 0);
            shim.memcpyDtoH(got.data(), p, got.size());
            size_t local = 0;
            for (size_t i = 0; i < want.size(); ++i)
            {
                if (got[i] != want[i])
                {
                    if (bad < 12)
                        std::printf("  [%s] %s +%zu (block %zu, +%zu): got 0x%02X want 0x%02X\n",
                                    sub.name, what, i, i / stride, i % stride, got[i], want[i]);
                    ++bad;
                    if (++local >= 6)
                        return;
                }
            }
        };
        region("dofBlocks", dDofBlocks, wantDof, DOF_BLOCK);
        region("linkBlocks", dLinkBlocks, wantLink, LINK_BLOCK);

        // updateDirty must be cleared in every case, including the ones that
        // wrote nothing
        std::vector<PxgArticulation> gotRecs(LANES);
        shim.memcpyDtoH(gotRecs.data(), dArtis, gotRecs.size() * sizeof(PxgArticulation));
        for (uint32_t L = 0; L < LANES; ++L)
        {
            if (gotRecs[L].data.updateDirty != 0u)
            {
                if (bad < 12)
                    std::printf("  [%s] updateDirty[%u]: got 0x%08X want 0\n",
                                sub.name, L, gotRecs[L].data.updateDirty);
                ++bad;
            }
        }
    }

    shim.memFree(dLinks);
    shim.memFree(dJData);
    shim.memFree(dJoints);
    shim.memFree(dMM);
    shim.memFree(dRQ);
    shim.memFree(dB2W);
    shim.memFree(dPos);
    shim.memFree(dVel);
    shim.memFree(dTPos);
    shim.memFree(dTVel);
    shim.memFree(dArtis);
    shim.memFree(dLinkBlocks);
    shim.memFree(dDofBlocks);

    PXG_CHECK_MSG(bad == 0, "%zu byte(s)/flag(s) wrong", bad);
    std::printf("  jcalc phase C: 4 dirty/direct-API combinations x 5 articulations -- the "
                "refresh subset byte-exact, the structural fields still poison, tendon-only "
                "dirty writes nothing, updateDirty always cleared\n");
}

} // namespace
