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

// P8.78: the tendon and mimic-joint block helpers.
//
// The CUDA has six functions; the port has three plus an `initialize` guard,
// because initialize*Block and update*Block differ by exactly the structural
// fields.  This test is built around that difference:
//
//   * sub-case "initialize" writes everything;
//   * sub-case "update" must leave mNumAttachments / mNumTendonJoints, every
//     attachment's and tendon joint's mParents / mChildrens / mLinkIndex, and all
//     three artiBlock.mNum* counters AT POISON.
//
// Those topology fields come from `mFixedElements` -- fixed at initialisation by
// its own name -- while the update path re-reads only `mModElements`, the
// run-time-modifiable half.  An update that also wrote them would be correct
// today and wrong the moment a tendon is re-pointed without a re-init.
//
// TWO SHAPES THE FIXTURE FORCES:
//
//  * mNbElements (2) < maxNumAttachments / maxNumTendonJoints (3), with TWO
//    tendons.  If they were equal the two-level index
//    `attachmentBlocks[i * maxNumAttachments + j]` would collapse and a wrong
//    pitch would be invisible.
//  * FIVE articulations at lanes 0-4, because mLinkIndex is a PxU16[32]: lanes 0
//    and 1 share word 0, lanes 2 and 3 share word 1, and lane 4 starts word 2.  A
//    non-atomic halfword store loses one lane of every pair, and lanes 5-31 must
//    stay at poison.
//
// Oracle caveat as in P8.73-P8.76: these helpers are __device__-only, so the
// expectation is the fixture's.  Generated offsets, distinct values everywhere,
// whole-region poison comparison and the sweep are what carry it.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "PxArticulationTendonData.h"
#include "PxgArticulation.h"
#include "PxgArticulationLink.h"
#include "PxgArticulationTendon.h"
#include "DyArticulationMimicJointCore.h"

#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

static std::vector<uint8_t> jcalcTendonLoader(const char* file) { return loadShaderBlob(file); }

static const uint32_t LANES = 5;
static const uint32_t NB_SPATIAL = 2;     // spatial tendons per articulation
static const uint32_t NB_FIXED = 2;       // fixed tendons per articulation
static const uint32_t NB_MIMIC = 2;
static const uint32_t NB_ELEMS = 2;       // attachments / tendon joints per tendon
static const uint32_t MAX_ATTACH = 3;     // deliberately > NB_ELEMS
static const uint32_t MAX_TJOINTS = 3;
static const uint32_t GUARD = 128;

PXG_TEST(artiJcalcTendons)
{
    using namespace physx;
    using namespace physx::Dy;

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        // { u64 x7, u32 x4 }
        { "npArtiJointFrame.spv", "artiJcalcTendons", 72u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &jcalcTendonLoader);

    auto fval = [](uint32_t L, uint32_t i, uint32_t j, uint32_t f) {
        return float(10000 * (L + 1) + 1000 * (i + 1) + 100 * (j + 1)) + 0.25f * float(f);
    };
    auto uval = [](uint32_t L, uint32_t i, uint32_t j, uint32_t f) {
        return uint32_t(0x40000000u + L * 0x01000000u + i * 0x00100000u + j * 0x00010000u + f);
    };

    // ---- host records ----
    std::vector<PxGpuSpatialTendonData> sParams(size_t(LANES) * NB_SPATIAL);
    std::vector<PxGpuFixedTendonData> fParams(size_t(LANES) * NB_FIXED);
    std::vector<PxgArticulationTendonElementFixedData> sFixed(size_t(LANES) * NB_SPATIAL * NB_ELEMS);
    std::vector<PxGpuTendonAttachmentData> sMod(size_t(LANES) * NB_SPATIAL * NB_ELEMS);
    std::vector<PxgArticulationTendonElementFixedData> fFixed(size_t(LANES) * NB_FIXED * NB_ELEMS);
    std::vector<PxGpuTendonJointCoefficientData> fMod(size_t(LANES) * NB_FIXED * NB_ELEMS);
    std::vector<ArticulationMimicJointCore> mimic(size_t(LANES) * NB_MIMIC);
    std::memset(sParams.data(), 0xEE, sParams.size() * sizeof(PxGpuSpatialTendonData));
    std::memset(fParams.data(), 0xEE, fParams.size() * sizeof(PxGpuFixedTendonData));
    std::memset(sFixed.data(), 0xEE, sFixed.size() * sizeof(PxgArticulationTendonElementFixedData));
    std::memset(sMod.data(), 0xEE, sMod.size() * sizeof(PxGpuTendonAttachmentData));
    std::memset(fFixed.data(), 0xEE, fFixed.size() * sizeof(PxgArticulationTendonElementFixedData));
    std::memset(fMod.data(), 0xEE, fMod.size() * sizeof(PxGpuTendonJointCoefficientData));
    std::memset(mimic.data(), 0xEE, mimic.size() * sizeof(ArticulationMimicJointCore));

    for (uint32_t L = 0; L < LANES; ++L)
    {
        for (uint32_t i = 0; i < NB_SPATIAL; ++i)
        {
            PxGpuSpatialTendonData& p = sParams[size_t(L) * NB_SPATIAL + i];
            p.stiffness = fval(L, i, 0, 1);
            p.damping = fval(L, i, 0, 2);
            p.limitStiffness = fval(L, i, 0, 3);
            p.offset = fval(L, i, 0, 4);
            for (uint32_t j = 0; j < NB_ELEMS; ++j)
            {
                const size_t k = (size_t(L) * NB_SPATIAL + i) * NB_ELEMS + j;
                sFixed[k].children = (uint64_t(uval(L, i, j, 5)) << 32) | uint64_t(uval(L, i, j, 6));
                sFixed[k].parent = uval(L, i, j, 7);
                sFixed[k].linkInd = 0x5500u + L * 16 + i * 4 + j;   // fits a PxU16
                sMod[k].relativeOffset = PxVec3(fval(L, i, j, 8), fval(L, i, j, 9),
                                                fval(L, i, j, 10));
                sMod[k].restLength = fval(L, i, j, 11);
                sMod[k].coefficient = fval(L, i, j, 12);
                sMod[k].lowLimit = fval(L, i, j, 13);
                sMod[k].highLimit = fval(L, i, j, 14);
            }
        }
        for (uint32_t i = 0; i < NB_FIXED; ++i)
        {
            PxGpuFixedTendonData& p = fParams[size_t(L) * NB_FIXED + i];
            p.stiffness = fval(L, i, 0, 20);
            p.damping = fval(L, i, 0, 21);
            p.limitStiffness = fval(L, i, 0, 22);
            p.offset = fval(L, i, 0, 23);
            p.lowLimit = fval(L, i, 0, 24);
            p.highLimit = fval(L, i, 0, 25);
            p.restLength = fval(L, i, 0, 26);
            for (uint32_t j = 0; j < NB_ELEMS; ++j)
            {
                const size_t k = (size_t(L) * NB_FIXED + i) * NB_ELEMS + j;
                fFixed[k].children = (uint64_t(uval(L, i, j, 27)) << 32) | uint64_t(uval(L, i, j, 28));
                fFixed[k].parent = uval(L, i, j, 29);
                fFixed[k].linkInd = 0x6600u + L * 16 + i * 4 + j;
                fMod[k].coefficient = fval(L, i, j, 30);
                fMod[k].recipCoefficient = fval(L, i, j, 31);
                fMod[k].axis = uval(L, i, j, 32) & 0xFFFFu;
            }
        }
        for (uint32_t i = 0; i < NB_MIMIC; ++i)
        {
            ArticulationMimicJointCore& m = mimic[size_t(L) * NB_MIMIC + i];
            m.linkA = uval(L, i, 0, 40);
            m.linkB = uval(L, i, 0, 41);
            m.axisA = PxArticulationAxis::Enum((L + i) % 6);
            m.axisB = PxArticulationAxis::Enum((L + i + 3) % 6);
            m.gearRatio = fval(L, i, 0, 42);
            m.offset = fval(L, i, 0, 43);
            m.naturalFrequency = fval(L, i, 0, 44);
            m.dampingRatio = fval(L, i, 0, 45);
        }
    }

    // ---- device buffers ----
    const size_t ST_REGION = size_t(NB_SPATIAL) * sizeof(PxgArticulationBlockSpatialTendonData) + GUARD;
    const size_t AT_REGION =
        size_t(NB_SPATIAL) * MAX_ATTACH * sizeof(PxgArticulationBlockAttachmentData) + GUARD;
    const size_t FT_REGION = size_t(NB_FIXED) * sizeof(PxgArticulationBlockFixedTendonData) + GUARD;
    const size_t TJ_REGION =
        size_t(NB_FIXED) * MAX_TJOINTS * sizeof(PxgArticulationBlockTendonJointData) + GUARD;
    const size_t MJ_REGION = size_t(NB_MIMIC) * sizeof(PxgArticulationBlockMimicJointData) + GUARD;
    const size_t AB_REGION = sizeof(PxgArticulationBlockData) + GUARD;

    PxgDevicePtr dSParams = 0, dFParams = 0, dSFixed = 0, dSMod = 0, dFFixed = 0, dFMod = 0;
    PxgDevicePtr dMimic = 0, dSTend = 0, dFTend = 0, dArtis = 0;
    PxgDevicePtr dST = 0, dAT = 0, dFT = 0, dTJ = 0, dMJ = 0, dAB = 0;
    shim.memAlloc(&dSParams, sParams.size() * sizeof(PxGpuSpatialTendonData));
    shim.memAlloc(&dFParams, fParams.size() * sizeof(PxGpuFixedTendonData));
    shim.memAlloc(&dSFixed, sFixed.size() * sizeof(PxgArticulationTendonElementFixedData));
    shim.memAlloc(&dSMod, sMod.size() * sizeof(PxGpuTendonAttachmentData));
    shim.memAlloc(&dFFixed, fFixed.size() * sizeof(PxgArticulationTendonElementFixedData));
    shim.memAlloc(&dFMod, fMod.size() * sizeof(PxGpuTendonJointCoefficientData));
    shim.memAlloc(&dMimic, mimic.size() * sizeof(ArticulationMimicJointCore));
    shim.memAlloc(&dSTend, size_t(LANES) * NB_SPATIAL * sizeof(PxgArticulationTendon));
    shim.memAlloc(&dFTend, size_t(LANES) * NB_FIXED * sizeof(PxgArticulationTendon));
    shim.memAlloc(&dArtis, size_t(LANES) * sizeof(PxgArticulation));
    shim.memAlloc(&dST, ST_REGION);
    shim.memAlloc(&dAT, AT_REGION);
    shim.memAlloc(&dFT, FT_REGION);
    shim.memAlloc(&dTJ, TJ_REGION);
    shim.memAlloc(&dMJ, MJ_REGION);
    shim.memAlloc(&dAB, AB_REGION);
    if (!dSParams || !dFParams || !dSFixed || !dSMod || !dFFixed || !dFMod || !dMimic ||
        !dSTend || !dFTend || !dArtis || !dST || !dAT || !dFT || !dTJ || !dMJ || !dAB)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }

    // the tendon records carry DEVICE pointers into the element arrays
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
        recs[L].data.numSpatialTendons = PxU16(NB_SPATIAL);
        recs[L].data.numFixedTendons = PxU16(NB_FIXED);
        recs[L].data.numMimicJoints = PxU16(NB_MIMIC);
        recs[L].spatialTendonParams = reinterpret_cast<PxGpuSpatialTendonData*>(
            dSParams + PxgDevicePtr(size_t(L) * NB_SPATIAL * sizeof(PxGpuSpatialTendonData)));
        recs[L].spatialTendons = reinterpret_cast<PxgArticulationTendon*>(
            dSTend + PxgDevicePtr(size_t(L) * NB_SPATIAL * sizeof(PxgArticulationTendon)));
        recs[L].fixedTendonParams = reinterpret_cast<PxGpuFixedTendonData*>(
            dFParams + PxgDevicePtr(size_t(L) * NB_FIXED * sizeof(PxGpuFixedTendonData)));
        recs[L].fixedTendons = reinterpret_cast<PxgArticulationTendon*>(
            dFTend + PxgDevicePtr(size_t(L) * NB_FIXED * sizeof(PxgArticulationTendon)));
        recs[L].mimicJointCores = reinterpret_cast<ArticulationMimicJointCore*>(
            dMimic + PxgDevicePtr(size_t(L) * NB_MIMIC * sizeof(ArticulationMimicJointCore)));
    }

    shim.memcpyHtoD(dSParams, sParams.data(), sParams.size() * sizeof(PxGpuSpatialTendonData));
    shim.memcpyHtoD(dFParams, fParams.data(), fParams.size() * sizeof(PxGpuFixedTendonData));
    shim.memcpyHtoD(dSFixed, sFixed.data(),
                    sFixed.size() * sizeof(PxgArticulationTendonElementFixedData));
    shim.memcpyHtoD(dSMod, sMod.data(), sMod.size() * sizeof(PxGpuTendonAttachmentData));
    shim.memcpyHtoD(dFFixed, fFixed.data(),
                    fFixed.size() * sizeof(PxgArticulationTendonElementFixedData));
    shim.memcpyHtoD(dFMod, fMod.data(), fMod.size() * sizeof(PxGpuTendonJointCoefficientData));
    shim.memcpyHtoD(dMimic, mimic.data(), mimic.size() * sizeof(ArticulationMimicJointCore));
    shim.memcpyHtoD(dSTend, sTend.data(), sTend.size() * sizeof(PxgArticulationTendon));
    shim.memcpyHtoD(dFTend, fTend.data(), fTend.size() * sizeof(PxgArticulationTendon));
    shim.memcpyHtoD(dArtis, recs.data(), recs.size() * sizeof(PxgArticulation));

    size_t bad = 0;
    const size_t ST_SZ = sizeof(PxgArticulationBlockSpatialTendonData);
    const size_t AT_SZ = sizeof(PxgArticulationBlockAttachmentData);
    const size_t FT_SZ = sizeof(PxgArticulationBlockFixedTendonData);
    const size_t TJ_SZ = sizeof(PxgArticulationBlockTendonJointData);
    const size_t MJ_SZ = sizeof(PxgArticulationBlockMimicJointData);

    for (uint32_t pass = 0; pass < 2; ++pass)
    {
        const bool initialize = (pass == 0);
        const char* name = initialize ? "initialize" : "update";

        std::vector<uint8_t> pST(ST_REGION, 0xAB), pAT(AT_REGION, 0xAB), pFT(FT_REGION, 0xAB);
        std::vector<uint8_t> pTJ(TJ_REGION, 0xAB), pMJ(MJ_REGION, 0xAB), pAB(AB_REGION, 0xAB);
        shim.memcpyHtoD(dST, pST.data(), pST.size());
        shim.memcpyHtoD(dAT, pAT.data(), pAT.size());
        shim.memcpyHtoD(dFT, pFT.data(), pFT.size());
        shim.memcpyHtoD(dTJ, pTJ.data(), pTJ.size());
        shim.memcpyHtoD(dMJ, pMJ.data(), pMJ.size());
        shim.memcpyHtoD(dAB, pAB.data(), pAB.size());

        struct Params
        {
            uint64_t artis, artiBlocks, spatialTendonBlocks, attachmentBlocks;
            uint64_t fixedTendonBlocks, tendonJointBlocks, mimicJointBlocks;
            uint32_t maxNumAttachments, maxNumTendonJoints, nbArticulations, initialize;
        } prm = { uint64_t(dArtis), uint64_t(dAB), uint64_t(dST), uint64_t(dAT),
                  uint64_t(dFT), uint64_t(dTJ), uint64_t(dMJ),
                  MAX_ATTACH, MAX_TJOINTS, LANES, initialize ? 1u : 0u };

        PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                      "launch artiJcalcTendons");
        shim.streamSynchronize();

        std::vector<uint8_t> wST = pST, wAT = pAT, wFT = pFT, wTJ = pTJ, wMJ = pMJ, wAB = pAB;
        auto putF = [](std::vector<uint8_t>& img, size_t off, float v) {
            std::memcpy(img.data() + off, &v, 4);
        };
        auto putU32 = [](std::vector<uint8_t>& img, size_t off, uint32_t v) {
            std::memcpy(img.data() + off, &v, 4);
        };
        auto putU64 = [](std::vector<uint8_t>& img, size_t off, uint64_t v) {
            std::memcpy(img.data() + off, &v, 8);
        };
        auto putU16 = [](std::vector<uint8_t>& img, size_t off, uint16_t v) {
            std::memcpy(img.data() + off, &v, 2);
        };

        for (uint32_t L = 0; L < LANES; ++L)
        {
            for (uint32_t i = 0; i < NB_SPATIAL; ++i)
            {
                const size_t tb = size_t(i) * ST_SZ;
                const PxGpuSpatialTendonData& p = sParams[size_t(L) * NB_SPATIAL + i];
                typedef PxgArticulationBlockSpatialTendonData ST;
                if (initialize)
                    putU32(wST, tb + offsetof(ST, mNumAttachments) + size_t(L) * 4, NB_ELEMS);
                putF(wST, tb + offsetof(ST, mStiffness) + size_t(L) * 4, p.stiffness);
                putF(wST, tb + offsetof(ST, mLimitStiffness) + size_t(L) * 4, p.limitStiffness);
                putF(wST, tb + offsetof(ST, mDamping) + size_t(L) * 4, p.damping);
                putF(wST, tb + offsetof(ST, mOffset) + size_t(L) * 4, p.offset);

                for (uint32_t j = 0; j < NB_ELEMS; ++j)
                {
                    const size_t k = (size_t(L) * NB_SPATIAL + i) * NB_ELEMS + j;
                    const size_t slot = (size_t(i) * MAX_ATTACH + j) * AT_SZ;
                    typedef PxgArticulationBlockAttachmentData AT;
                    if (initialize)
                    {
                        putU32(wAT, slot + offsetof(AT, mParents) + size_t(L) * 4,
                               sFixed[k].parent);
                        putU64(wAT, slot + offsetof(AT, mChildrens) + size_t(L) * 8,
                               sFixed[k].children);
                        putU16(wAT, slot + offsetof(AT, mLinkIndex) + size_t(L) * 2,
                               uint16_t(sFixed[k].linkInd));
                    }
                    const size_t ro = slot + offsetof(AT, mRelativeOffset) + size_t(L) * 12;
                    putF(wAT, ro + 0, sMod[k].relativeOffset.x);
                    putF(wAT, ro + 4, sMod[k].relativeOffset.y);
                    putF(wAT, ro + 8, sMod[k].relativeOffset.z);
                    putF(wAT, slot + offsetof(AT, mRestDistance) + size_t(L) * 4,
                         sMod[k].restLength);
                    putF(wAT, slot + offsetof(AT, mCoefficient) + size_t(L) * 4,
                         sMod[k].coefficient);
                    putF(wAT, slot + offsetof(AT, mLowLimit) + size_t(L) * 4, sMod[k].lowLimit);
                    putF(wAT, slot + offsetof(AT, mHighLimit) + size_t(L) * 4, sMod[k].highLimit);
                }
            }

            for (uint32_t i = 0; i < NB_FIXED; ++i)
            {
                const size_t tb = size_t(i) * FT_SZ;
                const PxGpuFixedTendonData& p = fParams[size_t(L) * NB_FIXED + i];
                typedef PxgArticulationBlockFixedTendonData FT;
                if (initialize)
                    putU32(wFT, tb + offsetof(FT, mNumTendonJoints) + size_t(L) * 4, NB_ELEMS);
                putF(wFT, tb + offsetof(FT, mStiffness) + size_t(L) * 4, p.stiffness);
                putF(wFT, tb + offsetof(FT, mDamping) + size_t(L) * 4, p.damping);
                putF(wFT, tb + offsetof(FT, mLimitStiffness) + size_t(L) * 4, p.limitStiffness);
                putF(wFT, tb + offsetof(FT, mOffset) + size_t(L) * 4, p.offset);
                putF(wFT, tb + offsetof(FT, mLowLimit) + size_t(L) * 4, p.lowLimit);
                putF(wFT, tb + offsetof(FT, mHighLimit) + size_t(L) * 4, p.highLimit);
                putF(wFT, tb + offsetof(FT, mRestLength) + size_t(L) * 4, p.restLength);

                for (uint32_t j = 0; j < NB_ELEMS; ++j)
                {
                    const size_t k = (size_t(L) * NB_FIXED + i) * NB_ELEMS + j;
                    const size_t slot = (size_t(i) * MAX_TJOINTS + j) * TJ_SZ;
                    typedef PxgArticulationBlockTendonJointData TJ;
                    if (initialize)
                    {
                        putU32(wTJ, slot + offsetof(TJ, mParents) + size_t(L) * 4,
                               fFixed[k].parent);
                        putU64(wTJ, slot + offsetof(TJ, mChildrens) + size_t(L) * 8,
                               fFixed[k].children);
                        putU16(wTJ, slot + offsetof(TJ, mLinkIndex) + size_t(L) * 2,
                               uint16_t(fFixed[k].linkInd));
                    }
                    putF(wTJ, slot + offsetof(TJ, mCoefficient) + size_t(L) * 4,
                         fMod[k].coefficient);
                    putF(wTJ, slot + offsetof(TJ, mRecipCoefficient) + size_t(L) * 4,
                         fMod[k].recipCoefficient);
                    putU32(wTJ, slot + offsetof(TJ, mAxis) + size_t(L) * 4, fMod[k].axis);
                }
            }

            for (uint32_t i = 0; i < NB_MIMIC; ++i)
            {
                const ArticulationMimicJointCore& m = mimic[size_t(L) * NB_MIMIC + i];
                const size_t mb = size_t(i) * MJ_SZ;
                typedef PxgArticulationBlockMimicJointData MJ;
                putU32(wMJ, mb + offsetof(MJ, mLinkA) + size_t(L) * 4, m.linkA);
                putU32(wMJ, mb + offsetof(MJ, mLinkB) + size_t(L) * 4, m.linkB);
                putU32(wMJ, mb + offsetof(MJ, mAxisA) + size_t(L) * 4, uint32_t(m.axisA));
                putU32(wMJ, mb + offsetof(MJ, mAxisB) + size_t(L) * 4, uint32_t(m.axisB));
                putF(wMJ, mb + offsetof(MJ, mGearRatio) + size_t(L) * 4, m.gearRatio);
                putF(wMJ, mb + offsetof(MJ, mOffset) + size_t(L) * 4, m.offset);
                putF(wMJ, mb + offsetof(MJ, mNaturalFrequency) + size_t(L) * 4,
                     m.naturalFrequency);
                putF(wMJ, mb + offsetof(MJ, mDampingRatio) + size_t(L) * 4, m.dampingRatio);
            }

            if (initialize)
            {
                typedef PxgArticulationBlockData AB;
                putU32(wAB, offsetof(AB, mNumSpatialTendons) + size_t(L) * 4, NB_SPATIAL);
                putU32(wAB, offsetof(AB, mNumFixedTendons) + size_t(L) * 4, NB_FIXED);
                putU32(wAB, offsetof(AB, mNumMimicJoints) + size_t(L) * 4, NB_MIMIC);
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
                                    name, what, i, i / stride, i % stride, got[i], want[i]);
                    ++bad;
                    if (++local >= 6)
                        return;
                }
            }
        };
        region("spatialTendonBlocks", dST, wST, ST_SZ);
        region("attachmentBlocks", dAT, wAT, AT_SZ);
        region("fixedTendonBlocks", dFT, wFT, FT_SZ);
        region("tendonJointBlocks", dTJ, wTJ, TJ_SZ);
        region("mimicJointBlocks", dMJ, wMJ, MJ_SZ);
        region("artiBlock", dAB, wAB, sizeof(PxgArticulationBlockData));
    }

    shim.memFree(dSParams); shim.memFree(dFParams);
    shim.memFree(dSFixed); shim.memFree(dSMod);
    shim.memFree(dFFixed); shim.memFree(dFMod);
    shim.memFree(dMimic); shim.memFree(dSTend); shim.memFree(dFTend); shim.memFree(dArtis);
    shim.memFree(dST); shim.memFree(dAT); shim.memFree(dFT);
    shim.memFree(dTJ); shim.memFree(dMJ); shim.memFree(dAB);

    PXG_CHECK_MSG(bad == 0, "%zu byte(s) wrong", bad);
    std::printf("  jcalc tendon/mimic blocks: 5 articulations x (2 spatial + 2 fixed tendons "
                "of 2 elements at pitch 3 + 2 mimic joints), initialize and update -- the "
                "structural fields written once and left alone on update\n");
}

} // namespace
