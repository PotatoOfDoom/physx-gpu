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

// P8.79: the gpuDirtyFlag block -- the direct-GPU-API refresh.
//
// Ten branches over nine flags, because eDIRTY_POSITIONS drives TWO:
//
//     POSITIONS                        -> every dof's mJointPositions
//     ROOT_TRANSFORM | POSITIONS       -> every LINK's mAccumulatedPose
//
// The sub-case matrix is built to separate them, which the obvious one-flag-at-a-
// time matrix would not:
//
//   * POSITIONS alone       -- must write joint positions AND all poses
//   * ROOT_TRANSFORM alone  -- must write the poses and NOTHING else
//   * VELOCITIES alone      -- must NOT touch the poses
//
// A port that hung the pose loop off ROOT_TRANSFORM only passes cases 2 and 3 and
// fails case 1; one that hung it off POSITIONS only fails case 2.  The pose loop
// also runs from linkID 0 -- unlike the phase B2 and phase C link loops, which
// start at 1 because the root has no joint.  Here the root's pose is exactly what a
// root-transform poke changes, so a loop starting at 1 fails on lane-0 link 0.
//
// The four tendon branches call the halves P8.78's copy functions were split into,
// with the structural guard OFF.  So the counts, the topology and the artiBlock
// counters must stay at poison in every sub-case: a direct-API poke changes values,
// never structure.
//
// Oracle caveat as in P8.73-P8.78: __device__-only, so the expectation is the
// fixture's; generated offsets, distinct values, whole-region poison comparison and
// the sweep carry it.  Every sub-case also checks gpuDirtyFlag comes back ZERO --
// including the one where no bit was set at all, where the early-out means the
// clear must not have run either... except it must, because the CUDA clears inside
// `if (artiGpuDirty)`.  That asymmetry is checked explicitly.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "PxArticulationTendonData.h"
#include "PxgArticulation.h"
#include "PxgArticulationLink.h"
#include "PxgArticulationTendon.h"
#include "DyArticulationMimicJointCore.h"
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

static std::vector<uint8_t> jcalcGpuDirtyLoader(const char* file) { return loadShaderBlob(file); }

static const uint32_t LANES = 5;
static const uint32_t NB_LINKS = 3;
// DELIBERATELY != NB_LINKS.  The first sweep had two mutants survive -- the dof
// loop bounded by numLinks, and totalDofs read out of the numLinks field -- purely
// because the fixture had both equal to 3, which makes those mutants literal
// no-ops.  Two loop bounds that happen to coincide are two loop bounds nothing
// tests.
static const uint32_t TOTAL_DOFS = 4;
static const uint32_t NB_SPATIAL = 2;
static const uint32_t NB_FIXED = 2;
static const uint32_t NB_ELEMS = 2;
static const uint32_t MAX_ATTACH = 3;
static const uint32_t MAX_TJOINTS = 3;
static const uint32_t GUARD = 128;

static_assert(uint32_t(physx::Dy::ArticulationDirtyFlag::eDIRTY_ROOT_TRANSFORM) == 16,
              "PXG_ARTI_DIRTY_ROOT_TRANSFORM");
static_assert(uint32_t(physx::Dy::ArticulationDirtyFlag::eDIRTY_ROOT_VELOCITIES) == 32,
              "PXG_ARTI_DIRTY_ROOT_VELOCITIES");

PXG_TEST(artiJcalcGpuDirty)
{
    using namespace physx;
    using namespace physx::Dy;

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        // { u64 x7, u32 x4 }
        { "npArtiJointFrame.spv", "artiJcalcGpuDirty", 72u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &jcalcGpuDirtyLoader);

    auto fv = [](uint32_t L, uint32_t i, uint32_t f) {
        return float(20000 * (L + 1) + 100 * (i + 1)) + 0.5f * float(f);
    };
    auto uv = [](uint32_t L, uint32_t i, uint32_t f) {
        return uint32_t(0x60000000u + L * 0x01000000u + i * 0x00010000u + f);
    };

    // ---- records ----
    std::vector<float> jPos(size_t(LANES) * TOTAL_DOFS), jVel(size_t(LANES) * TOTAL_DOFS);
    std::vector<float> jTPos(size_t(LANES) * TOTAL_DOFS), jTVel(size_t(LANES) * TOTAL_DOFS);
    std::vector<PxTransform> body2Worlds(size_t(LANES) * NB_LINKS);
    std::vector<Cm::UnAlignedSpatialVector> motionVel(size_t(LANES) * NB_LINKS);
    std::vector<PxGpuSpatialTendonData> sParams(size_t(LANES) * NB_SPATIAL);
    std::vector<PxGpuFixedTendonData> fParams(size_t(LANES) * NB_FIXED);
    std::vector<PxgArticulationTendonElementFixedData> sFixed(size_t(LANES) * NB_SPATIAL * NB_ELEMS);
    std::vector<PxGpuTendonAttachmentData> sMod(size_t(LANES) * NB_SPATIAL * NB_ELEMS);
    std::vector<PxgArticulationTendonElementFixedData> fFixed(size_t(LANES) * NB_FIXED * NB_ELEMS);
    std::vector<PxGpuTendonJointCoefficientData> fMod(size_t(LANES) * NB_FIXED * NB_ELEMS);
    std::memset(sParams.data(), 0xEE, sParams.size() * sizeof(PxGpuSpatialTendonData));
    std::memset(fParams.data(), 0xEE, fParams.size() * sizeof(PxGpuFixedTendonData));
    std::memset(sFixed.data(), 0xEE, sFixed.size() * sizeof(PxgArticulationTendonElementFixedData));
    std::memset(sMod.data(), 0xEE, sMod.size() * sizeof(PxGpuTendonAttachmentData));
    std::memset(fFixed.data(), 0xEE, fFixed.size() * sizeof(PxgArticulationTendonElementFixedData));
    std::memset(fMod.data(), 0xEE, fMod.size() * sizeof(PxGpuTendonJointCoefficientData));

    for (uint32_t L = 0; L < LANES; ++L)
    {
        for (uint32_t d = 0; d < TOTAL_DOFS; ++d)
        {
            const size_t k = size_t(L) * TOTAL_DOFS + d;
            jPos[k] = fv(L, d, 1);
            jVel[k] = fv(L, d, 2);
            jTPos[k] = fv(L, d, 3);
            jTVel[k] = fv(L, d, 4);
        }
        for (uint32_t l = 0; l < NB_LINKS; ++l)
        {
            const size_t k = size_t(L) * NB_LINKS + l;
            body2Worlds[k] = PxTransform(PxVec3(fv(L, l, 5), fv(L, l, 6), fv(L, l, 7)),
                                         PxQuat(0.2f, -0.4f, 0.5f, 0.7f).getNormalized());
            motionVel[k].top = PxVec3(fv(L, l, 8), fv(L, l, 9), fv(L, l, 10));
            motionVel[k].bottom = PxVec3(fv(L, l, 11), fv(L, l, 12), fv(L, l, 13));
        }
        for (uint32_t i = 0; i < NB_SPATIAL; ++i)
        {
            PxGpuSpatialTendonData& p = sParams[size_t(L) * NB_SPATIAL + i];
            p.stiffness = fv(L, i, 20);
            p.damping = fv(L, i, 21);
            p.limitStiffness = fv(L, i, 22);
            p.offset = fv(L, i, 23);
            for (uint32_t j = 0; j < NB_ELEMS; ++j)
            {
                const size_t k = (size_t(L) * NB_SPATIAL + i) * NB_ELEMS + j;
                sFixed[k].children = (uint64_t(uv(L, i, 24)) << 32) | uint64_t(uv(L, i, 25) + j);
                sFixed[k].parent = uv(L, i, 26) + j;
                sFixed[k].linkInd = 0x7700u + L * 16 + i * 4 + j;
                sMod[k].relativeOffset = PxVec3(fv(L, i, 27 + j), fv(L, i, 28 + j),
                                                fv(L, i, 29 + j));
                sMod[k].restLength = fv(L, i, 30 + j);
                sMod[k].coefficient = fv(L, i, 32 + j);
                sMod[k].lowLimit = fv(L, i, 34 + j);
                sMod[k].highLimit = fv(L, i, 36 + j);
            }
        }
        for (uint32_t i = 0; i < NB_FIXED; ++i)
        {
            PxGpuFixedTendonData& p = fParams[size_t(L) * NB_FIXED + i];
            p.stiffness = fv(L, i, 40);
            p.damping = fv(L, i, 41);
            p.limitStiffness = fv(L, i, 42);
            p.offset = fv(L, i, 43);
            p.lowLimit = fv(L, i, 44);
            p.highLimit = fv(L, i, 45);
            p.restLength = fv(L, i, 46);
            for (uint32_t j = 0; j < NB_ELEMS; ++j)
            {
                const size_t k = (size_t(L) * NB_FIXED + i) * NB_ELEMS + j;
                fFixed[k].children = (uint64_t(uv(L, i, 47)) << 32) | uint64_t(uv(L, i, 48) + j);
                fFixed[k].parent = uv(L, i, 49) + j;
                fFixed[k].linkInd = 0x8800u + L * 16 + i * 4 + j;
                fMod[k].coefficient = fv(L, i, 50 + j);
                fMod[k].recipCoefficient = fv(L, i, 52 + j);
                fMod[k].axis = (uv(L, i, 54) + j) & 0xFFFFu;
            }
        }
    }

    const size_t LINK_BLOCK = sizeof(PxgArticulationBlockLinkData);
    const size_t DOF_BLOCK = sizeof(PxgArticulationBlockDofData);
    const size_t ST_SZ = sizeof(PxgArticulationBlockSpatialTendonData);
    const size_t AT_SZ = sizeof(PxgArticulationBlockAttachmentData);
    const size_t FT_SZ = sizeof(PxgArticulationBlockFixedTendonData);
    const size_t TJ_SZ = sizeof(PxgArticulationBlockTendonJointData);
    const size_t LINK_REGION = size_t(NB_LINKS) * LINK_BLOCK + GUARD;
    const size_t DOF_REGION = size_t(TOTAL_DOFS) * DOF_BLOCK + GUARD;
    const size_t ST_REGION = size_t(NB_SPATIAL) * ST_SZ + GUARD;
    const size_t AT_REGION = size_t(NB_SPATIAL) * MAX_ATTACH * AT_SZ + GUARD;
    const size_t FT_REGION = size_t(NB_FIXED) * FT_SZ + GUARD;
    const size_t TJ_REGION = size_t(NB_FIXED) * MAX_TJOINTS * TJ_SZ + GUARD;

    PxgDevicePtr dPos = 0, dVel = 0, dTPos = 0, dTVel = 0, dB2W = 0, dMV = 0;
    PxgDevicePtr dSParams = 0, dFParams = 0, dSFixed = 0, dSMod = 0, dFFixed = 0, dFMod = 0;
    PxgDevicePtr dSTend = 0, dFTend = 0, dArtis = 0;
    PxgDevicePtr dLink = 0, dDof = 0, dST = 0, dAT = 0, dFT = 0, dTJ = 0;
    shim.memAlloc(&dPos, jPos.size() * 4);
    shim.memAlloc(&dVel, jVel.size() * 4);
    shim.memAlloc(&dTPos, jTPos.size() * 4);
    shim.memAlloc(&dTVel, jTVel.size() * 4);
    shim.memAlloc(&dB2W, body2Worlds.size() * sizeof(PxTransform));
    shim.memAlloc(&dMV, motionVel.size() * sizeof(Cm::UnAlignedSpatialVector));
    shim.memAlloc(&dSParams, sParams.size() * sizeof(PxGpuSpatialTendonData));
    shim.memAlloc(&dFParams, fParams.size() * sizeof(PxGpuFixedTendonData));
    shim.memAlloc(&dSFixed, sFixed.size() * sizeof(PxgArticulationTendonElementFixedData));
    shim.memAlloc(&dSMod, sMod.size() * sizeof(PxGpuTendonAttachmentData));
    shim.memAlloc(&dFFixed, fFixed.size() * sizeof(PxgArticulationTendonElementFixedData));
    shim.memAlloc(&dFMod, fMod.size() * sizeof(PxGpuTendonJointCoefficientData));
    shim.memAlloc(&dSTend, size_t(LANES) * NB_SPATIAL * sizeof(PxgArticulationTendon));
    shim.memAlloc(&dFTend, size_t(LANES) * NB_FIXED * sizeof(PxgArticulationTendon));
    shim.memAlloc(&dArtis, size_t(LANES) * sizeof(PxgArticulation));
    shim.memAlloc(&dLink, LINK_REGION);
    shim.memAlloc(&dDof, DOF_REGION);
    shim.memAlloc(&dST, ST_REGION);
    shim.memAlloc(&dAT, AT_REGION);
    shim.memAlloc(&dFT, FT_REGION);
    shim.memAlloc(&dTJ, TJ_REGION);
    if (!dPos || !dVel || !dTPos || !dTVel || !dB2W || !dMV || !dSParams || !dFParams ||
        !dSFixed || !dSMod || !dFFixed || !dFMod || !dSTend || !dFTend || !dArtis ||
        !dLink || !dDof || !dST || !dAT || !dFT || !dTJ)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }

    std::vector<PxgArticulationTendon> sTend(size_t(LANES) * NB_SPATIAL);
    std::vector<PxgArticulationTendon> fTend(size_t(LANES) * NB_FIXED);
    for (uint32_t L = 0; L < LANES; ++L)
    {
        for (uint32_t i = 0; i < NB_SPATIAL; ++i)
        {
            const size_t base = (size_t(L) * NB_SPATIAL + i) * NB_ELEMS;
            PxgArticulationTendon& t = sTend[size_t(L) * NB_SPATIAL + i];
            t.mFixedElements = reinterpret_cast<void*>(
                dSFixed + PxgDevicePtr(base * sizeof(PxgArticulationTendonElementFixedData)));
            t.mModElements = reinterpret_cast<void*>(
                dSMod + PxgDevicePtr(base * sizeof(PxGpuTendonAttachmentData)));
            t.mNbElements = NB_ELEMS;
        }
        for (uint32_t i = 0; i < NB_FIXED; ++i)
        {
            const size_t base = (size_t(L) * NB_FIXED + i) * NB_ELEMS;
            PxgArticulationTendon& t = fTend[size_t(L) * NB_FIXED + i];
            t.mFixedElements = reinterpret_cast<void*>(
                dFFixed + PxgDevicePtr(base * sizeof(PxgArticulationTendonElementFixedData)));
            t.mModElements = reinterpret_cast<void*>(
                dFMod + PxgDevicePtr(base * sizeof(PxGpuTendonJointCoefficientData)));
            t.mNbElements = NB_ELEMS;
        }
    }

    std::vector<PxgArticulation> recs(LANES);
    std::memset(recs.data(), 0, recs.size() * sizeof(PxgArticulation));
    for (uint32_t L = 0; L < LANES; ++L)
    {
        recs[L].data.numLinks = PxU16(NB_LINKS);
        recs[L].data.numJointDofs = PxU16(TOTAL_DOFS);
        recs[L].data.numSpatialTendons = PxU16(NB_SPATIAL);
        recs[L].data.numFixedTendons = PxU16(NB_FIXED);
        recs[L].jointPositions = reinterpret_cast<PxReal*>(dPos + PxgDevicePtr(size_t(L) * TOTAL_DOFS * 4));
        recs[L].jointVelocities = reinterpret_cast<PxReal*>(dVel + PxgDevicePtr(size_t(L) * TOTAL_DOFS * 4));
        recs[L].jointTargetPositions = reinterpret_cast<PxReal*>(dTPos + PxgDevicePtr(size_t(L) * TOTAL_DOFS * 4));
        recs[L].jointTargetVelocities = reinterpret_cast<PxReal*>(dTVel + PxgDevicePtr(size_t(L) * TOTAL_DOFS * 4));
        recs[L].linkBody2Worlds = reinterpret_cast<PxTransform*>(
            dB2W + PxgDevicePtr(size_t(L) * NB_LINKS * sizeof(PxTransform)));
        recs[L].motionVelocities = reinterpret_cast<Cm::UnAlignedSpatialVector*>(
            dMV + PxgDevicePtr(size_t(L) * NB_LINKS * sizeof(Cm::UnAlignedSpatialVector)));
        recs[L].spatialTendonParams = reinterpret_cast<PxGpuSpatialTendonData*>(
            dSParams + PxgDevicePtr(size_t(L) * NB_SPATIAL * sizeof(PxGpuSpatialTendonData)));
        recs[L].spatialTendons = reinterpret_cast<PxgArticulationTendon*>(
            dSTend + PxgDevicePtr(size_t(L) * NB_SPATIAL * sizeof(PxgArticulationTendon)));
        recs[L].fixedTendonParams = reinterpret_cast<PxGpuFixedTendonData*>(
            dFParams + PxgDevicePtr(size_t(L) * NB_FIXED * sizeof(PxGpuFixedTendonData)));
        recs[L].fixedTendons = reinterpret_cast<PxgArticulationTendon*>(
            dFTend + PxgDevicePtr(size_t(L) * NB_FIXED * sizeof(PxgArticulationTendon)));
    }

    shim.memcpyHtoD(dPos, jPos.data(), jPos.size() * 4);
    shim.memcpyHtoD(dVel, jVel.data(), jVel.size() * 4);
    shim.memcpyHtoD(dTPos, jTPos.data(), jTPos.size() * 4);
    shim.memcpyHtoD(dTVel, jTVel.data(), jTVel.size() * 4);
    shim.memcpyHtoD(dB2W, body2Worlds.data(), body2Worlds.size() * sizeof(PxTransform));
    shim.memcpyHtoD(dMV, motionVel.data(), motionVel.size() * sizeof(Cm::UnAlignedSpatialVector));
    shim.memcpyHtoD(dSParams, sParams.data(), sParams.size() * sizeof(PxGpuSpatialTendonData));
    shim.memcpyHtoD(dFParams, fParams.data(), fParams.size() * sizeof(PxGpuFixedTendonData));
    shim.memcpyHtoD(dSFixed, sFixed.data(),
                    sFixed.size() * sizeof(PxgArticulationTendonElementFixedData));
    shim.memcpyHtoD(dSMod, sMod.data(), sMod.size() * sizeof(PxGpuTendonAttachmentData));
    shim.memcpyHtoD(dFFixed, fFixed.data(),
                    fFixed.size() * sizeof(PxgArticulationTendonElementFixedData));
    shim.memcpyHtoD(dFMod, fMod.data(), fMod.size() * sizeof(PxGpuTendonJointCoefficientData));
    shim.memcpyHtoD(dSTend, sTend.data(), sTend.size() * sizeof(PxgArticulationTendon));
    shim.memcpyHtoD(dFTend, fTend.data(), fTend.size() * sizeof(PxgArticulationTendon));

    typedef ArticulationDirtyFlag D;
    struct Sub { const char* name; uint32_t dirty; };
    const Sub subs[] = {
        { "positions (drives the pose loop too)", uint32_t(D::eDIRTY_POSITIONS) },
        { "root transform alone",                 uint32_t(D::eDIRTY_ROOT_TRANSFORM) },
        { "velocities alone (poses untouched)",   uint32_t(D::eDIRTY_VELOCITIES) },
        { "root velocities alone",                uint32_t(D::eDIRTY_ROOT_VELOCITIES) },
        { "joint target vel",                     uint32_t(D::eDIRTY_JOINT_TARGET_VEL) },
        { "joint target pos",                     uint32_t(D::eDIRTY_JOINT_TARGET_POS) },
        { "spatial tendon params",                uint32_t(D::eDIRTY_SPATIAL_TENDON) },
        { "spatial tendon attachments",           uint32_t(D::eDIRTY_SPATIAL_TENDON_ATTACHMENT) },
        { "fixed tendon params",                  uint32_t(D::eDIRTY_FIXED_TENDON) },
        { "fixed tendon joints",                  uint32_t(D::eDIRTY_FIXED_TENDON_JOINT) },
        { "everything at once",
          uint32_t(D::eDIRTY_POSITIONS) | uint32_t(D::eDIRTY_VELOCITIES) |
          uint32_t(D::eDIRTY_ROOT_TRANSFORM) | uint32_t(D::eDIRTY_ROOT_VELOCITIES) |
          uint32_t(D::eDIRTY_JOINT_TARGET_VEL) | uint32_t(D::eDIRTY_JOINT_TARGET_POS) |
          uint32_t(D::eDIRTY_SPATIAL_TENDON) | uint32_t(D::eDIRTY_SPATIAL_TENDON_ATTACHMENT) |
          uint32_t(D::eDIRTY_FIXED_TENDON) | uint32_t(D::eDIRTY_FIXED_TENDON_JOINT) },
        { "nothing dirty (early out)",            0u },
    };

    size_t bad = 0;
    const size_t CD = offsetof(PxgArticulationBlockDofData, mConstraintData);

    for (const Sub& sub : subs)
    {
        for (uint32_t L = 0; L < LANES; ++L)
            recs[L].data.gpuDirtyFlag = sub.dirty;
        shim.memcpyHtoD(dArtis, recs.data(), recs.size() * sizeof(PxgArticulation));

        std::vector<uint8_t> pLink(LINK_REGION, 0xAB), pDof(DOF_REGION, 0xAB);
        std::vector<uint8_t> pST(ST_REGION, 0xAB), pAT(AT_REGION, 0xAB);
        std::vector<uint8_t> pFT(FT_REGION, 0xAB), pTJ(TJ_REGION, 0xAB);
        shim.memcpyHtoD(dLink, pLink.data(), pLink.size());
        shim.memcpyHtoD(dDof, pDof.data(), pDof.size());
        shim.memcpyHtoD(dST, pST.data(), pST.size());
        shim.memcpyHtoD(dAT, pAT.data(), pAT.size());
        shim.memcpyHtoD(dFT, pFT.data(), pFT.size());
        shim.memcpyHtoD(dTJ, pTJ.data(), pTJ.size());

        struct Params
        {
            uint64_t artis, linkBlocks, dofBlocks, spatialTendonBlocks, attachmentBlocks;
            uint64_t fixedTendonBlocks, tendonJointBlocks;
            uint32_t maxNumAttachments, maxNumTendonJoints, nbArticulations, pad0;
        } prm = { uint64_t(dArtis), uint64_t(dLink), uint64_t(dDof), uint64_t(dST),
                  uint64_t(dAT), uint64_t(dFT), uint64_t(dTJ),
                  MAX_ATTACH, MAX_TJOINTS, LANES, 0u };

        PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                      "launch artiJcalcGpuDirty");
        shim.streamSynchronize();

        std::vector<uint8_t> wLink = pLink, wDof = pDof, wST = pST, wAT = pAT, wFT = pFT,
                             wTJ = pTJ;
        auto putF = [](std::vector<uint8_t>& img, size_t off, float v) {
            std::memcpy(img.data() + off, &v, 4);
        };
        auto putU32 = [](std::vector<uint8_t>& img, size_t off, uint32_t v) {
            std::memcpy(img.data() + off, &v, 4);
        };

        const uint32_t d = sub.dirty;
        for (uint32_t L = 0; L < LANES; ++L)
        {
            if (d & uint32_t(D::eDIRTY_POSITIONS))
                for (uint32_t i = 0; i < TOTAL_DOFS; ++i)
                    putF(wDof, size_t(i) * DOF_BLOCK +
                                   offsetof(PxgArticulationBlockDofData, mJointPositions) +
                                   size_t(L) * 4,
                         jPos[size_t(L) * TOTAL_DOFS + i]);

            if (d & uint32_t(D::eDIRTY_VELOCITIES))
                for (uint32_t i = 0; i < TOTAL_DOFS; ++i)
                    putF(wDof, size_t(i) * DOF_BLOCK +
                                   offsetof(PxgArticulationBlockDofData, mJointVelocities) +
                                   size_t(L) * 4,
                         jVel[size_t(L) * TOTAL_DOFS + i]);

            if (d & (uint32_t(D::eDIRTY_ROOT_TRANSFORM) | uint32_t(D::eDIRTY_POSITIONS)))
                for (uint32_t l = 0; l < NB_LINKS; ++l)   // from 0
                {
                    const PxTransform& t = body2Worlds[size_t(L) * NB_LINKS + l];
                    const size_t ap = size_t(l) * LINK_BLOCK +
                                      offsetof(PxgArticulationBlockLinkData, mAccumulatedPose);
                    putF(wLink, ap + 0 + size_t(L) * 16 + 0, t.q.x);
                    putF(wLink, ap + 0 + size_t(L) * 16 + 4, t.q.y);
                    putF(wLink, ap + 0 + size_t(L) * 16 + 8, t.q.z);
                    putF(wLink, ap + 0 + size_t(L) * 16 + 12, t.q.w);
                    putF(wLink, ap + 512 + size_t(L) * 16 + 0, t.p.x);
                    putF(wLink, ap + 512 + size_t(L) * 16 + 4, t.p.y);
                    putF(wLink, ap + 512 + size_t(L) * 16 + 8, t.p.z);
                    putF(wLink, ap + 512 + size_t(L) * 16 + 12, 0.0f);
                }

            if (d & uint32_t(D::eDIRTY_ROOT_VELOCITIES))
            {
                const Cm::UnAlignedSpatialVector& v = motionVel[size_t(L) * NB_LINKS];
                const size_t mv = offsetof(PxgArticulationBlockLinkData, mMotionVelocity);
                putF(wLink, mv + 0 + size_t(L) * 16 + 0, v.top.x);
                putF(wLink, mv + 0 + size_t(L) * 16 + 4, v.top.y);
                putF(wLink, mv + 0 + size_t(L) * 16 + 8, v.top.z);
                putF(wLink, mv + 0 + size_t(L) * 16 + 12, v.bottom.x);
                putF(wLink, mv + 512 + size_t(L) * 8 + 0, v.bottom.y);
                putF(wLink, mv + 512 + size_t(L) * 8 + 4, v.bottom.z);
            }

            if (d & uint32_t(D::eDIRTY_JOINT_TARGET_VEL))
                for (uint32_t i = 0; i < TOTAL_DOFS; ++i)
                    putF(wDof, size_t(i) * DOF_BLOCK + CD +
                                   offsetof(PxgArticulationInternalConstraintData, mDriveTargetVel) +
                                   size_t(L) * 4,
                         jTVel[size_t(L) * TOTAL_DOFS + i]);

            if (d & uint32_t(D::eDIRTY_JOINT_TARGET_POS))
                for (uint32_t i = 0; i < TOTAL_DOFS; ++i)
                    putF(wDof, size_t(i) * DOF_BLOCK + CD +
                                   offsetof(PxgArticulationInternalConstraintData, mDriveTargetPos) +
                                   size_t(L) * 4,
                         jTPos[size_t(L) * TOTAL_DOFS + i]);

            if (d & uint32_t(D::eDIRTY_SPATIAL_TENDON))
                for (uint32_t i = 0; i < NB_SPATIAL; ++i)
                {
                    typedef PxgArticulationBlockSpatialTendonData ST;
                    const PxGpuSpatialTendonData& p = sParams[size_t(L) * NB_SPATIAL + i];
                    const size_t tb = size_t(i) * ST_SZ;
                    putF(wST, tb + offsetof(ST, mStiffness) + size_t(L) * 4, p.stiffness);
                    putF(wST, tb + offsetof(ST, mLimitStiffness) + size_t(L) * 4, p.limitStiffness);
                    putF(wST, tb + offsetof(ST, mDamping) + size_t(L) * 4, p.damping);
                    putF(wST, tb + offsetof(ST, mOffset) + size_t(L) * 4, p.offset);
                }

            if (d & uint32_t(D::eDIRTY_SPATIAL_TENDON_ATTACHMENT))
                for (uint32_t i = 0; i < NB_SPATIAL; ++i)
                    for (uint32_t j = 0; j < NB_ELEMS; ++j)
                    {
                        typedef PxgArticulationBlockAttachmentData AT;
                        const size_t k = (size_t(L) * NB_SPATIAL + i) * NB_ELEMS + j;
                        const size_t slot = (size_t(i) * MAX_ATTACH + j) * AT_SZ;
                        const size_t ro = slot + offsetof(AT, mRelativeOffset) + size_t(L) * 12;
                        putF(wAT, ro + 0, sMod[k].relativeOffset.x);
                        putF(wAT, ro + 4, sMod[k].relativeOffset.y);
                        putF(wAT, ro + 8, sMod[k].relativeOffset.z);
                        putF(wAT, slot + offsetof(AT, mRestDistance) + size_t(L) * 4,
                             sMod[k].restLength);
                        putF(wAT, slot + offsetof(AT, mCoefficient) + size_t(L) * 4,
                             sMod[k].coefficient);
                        putF(wAT, slot + offsetof(AT, mLowLimit) + size_t(L) * 4,
                             sMod[k].lowLimit);
                        putF(wAT, slot + offsetof(AT, mHighLimit) + size_t(L) * 4,
                             sMod[k].highLimit);
                    }

            if (d & uint32_t(D::eDIRTY_FIXED_TENDON))
                for (uint32_t i = 0; i < NB_FIXED; ++i)
                {
                    typedef PxgArticulationBlockFixedTendonData FT;
                    const PxGpuFixedTendonData& p = fParams[size_t(L) * NB_FIXED + i];
                    const size_t tb = size_t(i) * FT_SZ;
                    putF(wFT, tb + offsetof(FT, mStiffness) + size_t(L) * 4, p.stiffness);
                    putF(wFT, tb + offsetof(FT, mDamping) + size_t(L) * 4, p.damping);
                    putF(wFT, tb + offsetof(FT, mLimitStiffness) + size_t(L) * 4, p.limitStiffness);
                    putF(wFT, tb + offsetof(FT, mOffset) + size_t(L) * 4, p.offset);
                    putF(wFT, tb + offsetof(FT, mLowLimit) + size_t(L) * 4, p.lowLimit);
                    putF(wFT, tb + offsetof(FT, mHighLimit) + size_t(L) * 4, p.highLimit);
                    putF(wFT, tb + offsetof(FT, mRestLength) + size_t(L) * 4, p.restLength);
                }

            if (d & uint32_t(D::eDIRTY_FIXED_TENDON_JOINT))
                for (uint32_t i = 0; i < NB_FIXED; ++i)
                    for (uint32_t j = 0; j < NB_ELEMS; ++j)
                    {
                        typedef PxgArticulationBlockTendonJointData TJ;
                        const size_t k = (size_t(L) * NB_FIXED + i) * NB_ELEMS + j;
                        const size_t slot = (size_t(i) * MAX_TJOINTS + j) * TJ_SZ;
                        putF(wTJ, slot + offsetof(TJ, mCoefficient) + size_t(L) * 4,
                             fMod[k].coefficient);
                        putF(wTJ, slot + offsetof(TJ, mRecipCoefficient) + size_t(L) * 4,
                             fMod[k].recipCoefficient);
                        putU32(wTJ, slot + offsetof(TJ, mAxis) + size_t(L) * 4, fMod[k].axis);
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
        region("linkBlocks", dLink, wLink, LINK_BLOCK);
        region("dofBlocks", dDof, wDof, DOF_BLOCK);
        region("spatialTendonBlocks", dST, wST, ST_SZ);
        region("attachmentBlocks", dAT, wAT, AT_SZ);
        region("fixedTendonBlocks", dFT, wFT, FT_SZ);
        region("tendonJointBlocks", dTJ, wTJ, TJ_SZ);

        // the flag is cleared inside `if (artiGpuDirty)`, so the no-bits case
        // leaves it alone -- which is the same observable value, zero, but for a
        // different reason.  Both must come back zero.
        std::vector<PxgArticulation> gotRecs(LANES);
        shim.memcpyDtoH(gotRecs.data(), dArtis, gotRecs.size() * sizeof(PxgArticulation));
        for (uint32_t L = 0; L < LANES; ++L)
            if (gotRecs[L].data.gpuDirtyFlag != 0u)
            {
                if (bad < 12)
                    std::printf("  [%s] gpuDirtyFlag[%u]: got 0x%08X want 0\n",
                                sub.name, L, gotRecs[L].data.gpuDirtyFlag);
                ++bad;
            }
    }

    shim.memFree(dPos); shim.memFree(dVel); shim.memFree(dTPos); shim.memFree(dTVel);
    shim.memFree(dB2W); shim.memFree(dMV);
    shim.memFree(dSParams); shim.memFree(dFParams);
    shim.memFree(dSFixed); shim.memFree(dSMod); shim.memFree(dFFixed); shim.memFree(dFMod);
    shim.memFree(dSTend); shim.memFree(dFTend); shim.memFree(dArtis);
    shim.memFree(dLink); shim.memFree(dDof);
    shim.memFree(dST); shim.memFree(dAT); shim.memFree(dFT); shim.memFree(dTJ);

    PXG_CHECK_MSG(bad == 0, "%zu byte(s)/flag(s) wrong", bad);
    std::printf("  jcalc gpuDirtyFlag: 12 sub-cases x 5 articulations -- positions drives the "
                "pose loop, root-transform-alone does too, velocities-alone does not; the "
                "structural fields never written; flag always cleared\n");
}

} // namespace
