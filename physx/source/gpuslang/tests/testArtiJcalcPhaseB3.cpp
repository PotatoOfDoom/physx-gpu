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

// P8.75: jcalc phase B3 -- the per-dof fill and the internal constraint block.
//
// THE THING THIS TEST EXISTS TO CATCH is the two indices:
//
//   i      -- the dof's position within its joint, 0..nbDofs-1
//   dofId  -- joint.dofIds[i], the PxArticulationAxis it drives, 0..5
//
// The dof block is addressed by (jointOffset + i); every per-axis parameter --
// drives, limits, armature, friction, maxJointVelocity, motion -- by dofId.  For
// the common revolute joint with dofIds[0] = eTWIST = 0 the two coincide, so a
// fixture built from a revolute chain would pass with the indices swapped.  Here
// EVERY joint uses axes where they differ: a spherical on {eSWING1, eSWING2} and a
// prismatic on {eY}, so i is 0,1,0 while dofId is 1,2,4 and no pair is equal.
//
// mInvDofIds is the inverse map (mInvDofIds[dofId] = i) and it is the one write in
// this loop that lands in the LINK block, not the dof block -- so the link blocks
// are allocated and poison-checked too: every byte of them except those six-by-32
// bytes must come back 0xAB.
//
// Oracle, stated as in P8.73/P8.74: jcalc is __device__-only, so the expectation is
// built by the fixture.  The offsets it uses are the generated, static_assert-pinned
// ones; the values are all distinct per (lane, link, dof, field) so a write landing
// at the wrong field, stride or lane differs from the expectation rather than
// coinciding with it; and the whole destination is compared byte for byte.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "PxArticulationTendonData.h"
#include "PxgArticulation.h"
#include "PxgArticulationLink.h"
#include "DyArticulationJointCore.h"

#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

static std::vector<uint8_t> jcalcB3Loader(const char* file) { return loadShaderBlob(file); }

static const uint32_t LANES = 5;
static const uint32_t NB_LINKS = 3;   // root + spherical(2 dof) + prismatic(1 dof)
static const uint32_t MAX_DOFS = 4;   // one spare block, which must stay poison
static const uint32_t GUARD = 128;

PXG_TEST(artiJcalcPhaseB3)
{
    using namespace physx;
    using namespace physx::Dy;

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiJointFrame.spv", "artiJcalcPhaseB3", 32u, 0 }, // { u64, u64, u64, u32, u32 }
    };
    PxgCudaShim shim(dev, manifest, 1, &jcalcB3Loader);

    // link 0 is the root and has no joint; link 1 is a 2-dof spherical on
    // {eSWING1, eSWING2}; link 2 is a 1-dof prismatic on {eY}.
    const uint32_t nbDof[NB_LINKS] = { 0u, 2u, 1u };
    const uint32_t jointOffset[NB_LINKS] = { 0u, 0u, 2u };
    const uint8_t dofIds[NB_LINKS][3] = { { 0, 0, 0 }, { 1, 2, 0 }, { 4, 0, 0 } };
    const uint32_t TOTAL_DOFS = 3;

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

    const size_t LINK_BLOCK = sizeof(PxgArticulationBlockLinkData);
    const size_t DOF_BLOCK = sizeof(PxgArticulationBlockDofData);
    const size_t LINK_REGION = size_t(NB_LINKS) * LINK_BLOCK + GUARD;
    const size_t DOF_REGION = size_t(MAX_DOFS) * DOF_BLOCK + GUARD;

    PxgDevicePtr dJData = 0, dJoints = 0, dMM = 0;
    PxgDevicePtr dPos = 0, dVel = 0, dTPos = 0, dTVel = 0;
    PxgDevicePtr dArtis = 0, dLinkBlocks = 0, dDofBlocks = 0;
    shim.memAlloc(&dJData, jdata.size() * sizeof(ArticulationJointCoreData));
    shim.memAlloc(&dJoints, joints.size() * sizeof(ArticulationJointCore));
    shim.memAlloc(&dMM, motionMatrix.size() * sizeof(SpatialSubspaceMatrix));
    shim.memAlloc(&dPos, jPos.size() * 4);
    shim.memAlloc(&dVel, jVel.size() * 4);
    shim.memAlloc(&dTPos, jTPos.size() * 4);
    shim.memAlloc(&dTVel, jTVel.size() * 4);
    shim.memAlloc(&dArtis, size_t(LANES) * sizeof(PxgArticulation));
    shim.memAlloc(&dLinkBlocks, LINK_REGION);
    shim.memAlloc(&dDofBlocks, DOF_REGION);
    if (!dJData || !dJoints || !dMM || !dPos || !dVel || !dTPos || !dTVel ||
        !dArtis || !dLinkBlocks || !dDofBlocks)
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
        recs[L].jointData = reinterpret_cast<ArticulationJointCoreData*>(
            dJData + PxgDevicePtr(l * sizeof(ArticulationJointCoreData)));
        recs[L].joints = reinterpret_cast<ArticulationJointCore*>(
            dJoints + PxgDevicePtr(l * sizeof(ArticulationJointCore)));
        recs[L].motionMatrix = reinterpret_cast<SpatialSubspaceMatrix*>(
            dMM + PxgDevicePtr(l * sizeof(SpatialSubspaceMatrix)));
        recs[L].jointPositions = reinterpret_cast<PxReal*>(dPos + PxgDevicePtr(d * 4));
        recs[L].jointVelocities = reinterpret_cast<PxReal*>(dVel + PxgDevicePtr(d * 4));
        recs[L].jointTargetPositions = reinterpret_cast<PxReal*>(dTPos + PxgDevicePtr(d * 4));
        recs[L].jointTargetVelocities = reinterpret_cast<PxReal*>(dTVel + PxgDevicePtr(d * 4));
    }

    shim.memcpyHtoD(dJData, jdata.data(), jdata.size() * sizeof(ArticulationJointCoreData));
    shim.memcpyHtoD(dJoints, joints.data(), joints.size() * sizeof(ArticulationJointCore));
    shim.memcpyHtoD(dMM, motionMatrix.data(), motionMatrix.size() * sizeof(SpatialSubspaceMatrix));
    shim.memcpyHtoD(dPos, jPos.data(), jPos.size() * 4);
    shim.memcpyHtoD(dVel, jVel.data(), jVel.size() * 4);
    shim.memcpyHtoD(dTPos, jTPos.data(), jTPos.size() * 4);
    shim.memcpyHtoD(dTVel, jTVel.data(), jTVel.size() * 4);
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
        uint32_t pad0;
    } prm = { uint64_t(dArtis), uint64_t(dLinkBlocks), uint64_t(dDofBlocks), LANES, 0u };

    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch artiJcalcPhaseB3");
    shim.streamSynchronize();

    // ---- expectation ----
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

    const size_t CD = offsetof(PxgArticulationBlockDofData, mConstraintData);
    for (uint32_t L = 0; L < LANES; ++L)
    {
        for (uint32_t link = 1; link < NB_LINKS; ++link)
        {
            const size_t jk = size_t(L) * NB_LINKS + link;
            const ArticulationJointCore& j = joints[jk];
            const size_t lblk = size_t(link) * LINK_BLOCK;

            for (uint32_t i = 0; i < nbDof[link]; ++i)
            {
                const uint32_t slot = jointOffset[link] + i;
                const uint32_t dofId = dofIds[link][i];
                const size_t db = size_t(slot) * DOF_BLOCK;
                const size_t ic = db + CD;
                const size_t ak = size_t(L) * MAX_DOFS + slot;

                // (5.12: no mJointAxis in the dof block)
                putU8(wantDof, db + offsetof(PxgArticulationBlockDofData, mDofIds) + L,
                      uint8_t(dofId));
                putU8(wantLink, lblk + offsetof(PxgArticulationBlockLinkData, mInvDofIds) +
                                    size_t(dofId) * 32 + L,
                      uint8_t(i));

                putF(wantDof, ic + offsetof(PxgArticulationInternalConstraintData, mDriveStiffness) +
                                  size_t(L) * 4, j.drives[dofId].stiffness);
                putF(wantDof, ic + offsetof(PxgArticulationInternalConstraintData, mDamping) +
                                  size_t(L) * 4, j.drives[dofId].damping);
                putF(wantDof, ic + offsetof(PxgArticulationInternalConstraintData, mDriveTargetVel) +
                                  size_t(L) * 4, jTVel[ak]);
                putF(wantDof, ic + offsetof(PxgArticulationInternalConstraintData, mTargetPosBias) +
                                  size_t(L) * 4, 0.0f);
                putF(wantDof, ic + offsetof(PxgArticulationInternalConstraintData, mArmature) +
                                  size_t(L) * 4, j.armature[dofId]);
                putF(wantDof, ic + offsetof(PxgArticulationInternalConstraintData, mDriveTargetPos) +
                                  size_t(L) * 4, jTPos[ak]);
                const size_t lim = ic + offsetof(PxgArticulationInternalConstraintData,
                                                 mLimits_LowLimitX_highLimitY) + size_t(L) * 8;
                putF(wantDof, lim + 0, j.limits[dofId].low);
                putF(wantDof, lim + 4, j.limits[dofId].high);
                putF(wantDof, ic + offsetof(PxgArticulationInternalConstraintData,
                                            mStaticFrictionEffort) + size_t(L) * 4,
                     j.frictionParams[dofId].staticFrictionEffort);
                putF(wantDof, ic + offsetof(PxgArticulationInternalConstraintData,
                                            mDynamicFrictionEffort) + size_t(L) * 4,
                     j.frictionParams[dofId].dynamicFrictionEffort);
                putF(wantDof, ic + offsetof(PxgArticulationInternalConstraintData,
                                            mViscousFrictionCoefficient) + size_t(L) * 4,
                     j.frictionParams[dofId].viscousFrictionCoefficient);
                putF(wantDof, ic + offsetof(PxgArticulationInternalConstraintData, mMaxEffort) +
                                  size_t(L) * 4, j.drives[dofId].envelope.maxEffort);
                putF(wantDof, ic + offsetof(PxgArticulationInternalConstraintData,
                                            mMaxActuatorVelocity) + size_t(L) * 4,
                     j.drives[dofId].envelope.maxActuatorVelocity);
                putF(wantDof, ic + offsetof(PxgArticulationInternalConstraintData,
                                            mVelocityDependentResistance) + size_t(L) * 4,
                     j.drives[dofId].envelope.velocityDependentResistance);
                putF(wantDof, ic + offsetof(PxgArticulationInternalConstraintData,
                                            mSpeedEffortGradient) + size_t(L) * 4,
                     j.drives[dofId].envelope.speedEffortGradient);
                putU32(wantDof, ic + offsetof(PxgArticulationInternalConstraintData, mDriveType) +
                                    size_t(L) * 4, uint32_t(j.drives[dofId].driveType));
                putF(wantDof, ic + offsetof(PxgArticulationInternalConstraintData,
                                            mMaxJointVelocity) + size_t(L) * 4,
                     j.maxJointVelocity[dofId]);

                putU8(wantDof, db + offsetof(PxgArticulationBlockDofData, mMotion) + L,
                      uint8_t(j.motion[dofId]));
                putF(wantDof, db + offsetof(PxgArticulationBlockDofData, mJointPositions) +
                                  size_t(L) * 4, jPos[ak]);
                putF(wantDof, db + offsetof(PxgArticulationBlockDofData, mJointVelocities) +
                                  size_t(L) * 4, jVel[ak]);

                putSV(wantDof, db, offsetof(PxgArticulationBlockDofData, mLocalMotionMatrix), L,
                      motionMatrix[jk].columns[i].top, motionMatrix[jk].columns[i].bottom);
            }
        }
    }

    // ---- compare ----
    size_t bad = 0;
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
                    std::printf("  %s +%zu (block %zu, +%zu): got 0x%02X want 0x%02X\n",
                                what, i, i / stride, i % stride, got[i], want[i]);
                ++bad;
                if (++local >= 6)
                    return;
            }
        }
    };
    region("dofBlocks", dDofBlocks, wantDof, DOF_BLOCK);
    region("linkBlocks", dLinkBlocks, wantLink, LINK_BLOCK);

    shim.memFree(dJData);
    shim.memFree(dJoints);
    shim.memFree(dMM);
    shim.memFree(dPos);
    shim.memFree(dVel);
    shim.memFree(dTPos);
    shim.memFree(dTVel);
    shim.memFree(dArtis);
    shim.memFree(dLinkBlocks);
    shim.memFree(dDofBlocks);

    PXG_CHECK_MSG(bad == 0, "%zu byte(s) wrong", bad);
    std::printf("  jcalc phase B3: 5 articulations x (spherical 2-dof on SWING1/SWING2 + "
                "prismatic on Y) -- 24 per-dof writes plus mInvDofIds byte-exact, i != dofId "
                "everywhere, spare dof block still poison\n");
}

} // namespace
