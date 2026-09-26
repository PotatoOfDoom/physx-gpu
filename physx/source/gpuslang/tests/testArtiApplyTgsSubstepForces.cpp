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


// P8.108: the `artiApplyTgsSubstepForces` engine entry
// (forwardDynamic2.cu:3482) -- per-substep external forces (the RECORD's
// zAForces) plus user joint forces turned into deferred impulses, leaf to
// root, once per TGS position iteration when the scene enables
// eENABLE_EXTERNAL_FORCES_EVERY_ITERATION_TGS.
//
// The mirror is pxgref::aiPropagateImpulseJf -- the template
// propagateImpulseW<WriteToDeferredQstZ> WITH a joint force, `jf =
// jointForce[ind] * stepDt` per dof -- over engine primitives, plus the
// kernel's control flow transcribed here.
//
// WHAT THE FIXTURE IS SHAPED TO SEE
//
//   * mScratchImpulse seeded NON-ZERO on every link: the kernel must
//     accumulate into it (articulatedYW = zExt*stepDt + scratch), never
//     overwrite -- and re-zero it afterwards, EXCEPT the root of a
//     fixed-base articulation, whose scratch must SURVIVE;
//   * the parent add is gated (parent > 0 || !fixedBase): link 1 of a
//     fixed-base lane propagates into NOTHING, but its mDeferredQstZ writes
//     and its own scratch reset still happen;
//   * mRootDeferredZ seeded LARGE: the free-base tail ADDS (unlike
//     artiPushImpulse's store), so a store-mutant loses the seed;
//   * mStateDirty seeded with 0/2/3: the kernel ASSIGNS eVEL_DIRTY (1) to
//     every live lane -- an OR-mutant leaves 3 as 3, a skip-mutant leaves
//     0 and 2 standing;
//   * mJointOffset windows in REVERSED link order (disjoint, off the
//     forward cursor), and the record's jointForce slice is indexed by
//     exactly that offset -- a cursor mutant reads the wrong forces;
//   * the record pool is real: 544-byte PxgArticulation images with only
//     the zAForces (@88) and jointForce (@136) pointers live, everything
//     else 0xAB;
//   * one THREAD per articulation, block (64,1,1): 40 live threads over
//     two warp slices, threads 40..63 dead (their block lanes stay poison).

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"
#include "pxgArtiRefHelpers.h"

#include "foundation/PxVec3.h"
#include "CmSpatialVector.h"
#include "DyFeatherstoneArticulation.h"
#include "PxgArticulation.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::PxReal;
using physx::PxU32;
using physx::PxVec3;
using physx::Cm::UnAlignedSpatialVector;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;
using physx::loadSpatialVector;
using physx::storeSpatialVector;

static std::vector<uint8_t> artiAtsLoader(const char* file) { return loadShaderBlob(file); }

// ---------------------------------------------------------------------------

const uint32_t ATS_NB_ARTI = 40;
const uint32_t ATS_NB_WARPS = 2;
const uint32_t ATS_MAX_LINKS = 7;
const uint32_t ATS_MAX_DOFS = 15;
const uint32_t ATS_RECORD_SIZE = 544;       // sizeof(PxgArticulation), pinned in the desc map
const float ATS_STEP_DT = 0.0234375f;       // exactly representable, not 1

// PXG_ARTI_CORE_* offsets, pinned by testDmaArtiResidual
const uint32_t ATS_D_ARTICULATIONS = 32;
const uint32_t ATS_D_NB_ARTICULATIONS = 44;
const uint32_t ATS_D_ARTICULATION_BLOCKS = 128;
const uint32_t ATS_D_LINK_BLOCKS = 136;
const uint32_t ATS_D_DOF_BLOCKS = 184;
const uint32_t ATS_D_MAX_LINKS = 248;
const uint32_t ATS_D_MAX_DOFS = 252;
const uint32_t ATS_DESC_SIZE = 360;
// PxgArticulation record offsets, pinned by the desc map's static_asserts
const uint32_t ATS_R_ZA_FORCES = 88;
const uint32_t ATS_R_JOINT_FORCE = 136;

static uint32_t atsNumLinks(uint32_t k) { return 1u + (k % 7u); }
static bool atsFixBase(uint32_t k) { return (k % 3u) == 1u; }
// seeded mStateDirty: NEVER 1, so the kernel's unconditional assign of 1 is
// visible on every live lane
static uint32_t atsDirtySeed(uint32_t k) { const uint32_t v[3] = { 0u, 2u, 3u }; return v[k % 3u]; }
static uint32_t atsFlags(uint32_t k)
{
    return (atsFixBase(k) ? 1u : 0u) | ((k & 1u) << 1) | 8u | ((k % 3u) << 5);
}
static uint32_t atsDofs(uint32_t k, uint32_t i) { return (k + i) % 4u; }
static uint32_t atsParent(uint32_t i)
{
    if (i <= 1u) return 0u;
    return (i >= 3u && (i % 3u) == 0u) ? 1u : (i - 1u);
}
static uint32_t atsTotalDofs(uint32_t k, uint32_t numLinks)
{
    uint32_t t = 0;
    for (uint32_t i = 1; i < numLinks; ++i)
        t += atsDofs(k, i);
    return t;
}
// reversed windows: disjoint, off the forward cursor
static uint32_t atsJointOffset(uint32_t k, uint32_t i, uint32_t numLinks)
{
    uint32_t off = atsTotalDofs(k, numLinks);
    for (uint32_t j = 1; j <= i; ++j)
        off -= atsDofs(k, j);
    return off;
}

static float ats(uint32_t i, uint32_t k)
{
    return (float((i * 73u + k * 31u) % 503u) + 1.0f) * 0.015625f - 3.5f;
}
static UnAlignedSpatialVector atsSV(uint32_t seed, uint32_t base)
{
    return UnAlignedSpatialVector(
        PxVec3(ats(seed, base), ats(seed, base + 1), ats(seed, base + 2)),
        PxVec3(ats(seed, base + 3), ats(seed, base + 4), ats(seed, base + 5)));
}

// ---------------------------------------------------------------------------

static void atsFill(physx::PxgArticulationBlockData* ab,
                    physx::PxgArticulationBlockLinkData* lk,
                    physx::PxgArticulationBlockDofData* df, uint32_t lane, uint32_t k)
{
    const uint32_t numLinks = atsNumLinks(k);

    ab->mNumLinks[lane] = numLinks;
    ab->mFlags[lane] = atsFlags(k);
    ab->mStateDirty[lane] = uint8_t(atsDirtySeed(k));

    storeSpatialVector(ab->mRootDeferredZ,
                       UnAlignedSpatialVector(PxVec3(150.0f + ats(k, 0), 250.0f + ats(k, 1),
                                                     350.0f + ats(k, 2)),
                                              PxVec3(450.0f + ats(k, 3), 550.0f + ats(k, 4),
                                                     650.0f + ats(k, 5))),
                       lane);

    for (uint32_t i = 0; i < numLinks; ++i)
    {
        physx::PxgArticulationBlockLinkData& L = lk[i];
        const uint32_t dofs = (i == 0) ? 0u : atsDofs(k, i);

        L.mDofs[lane] = uint8_t(dofs);
        L.mParents[lane] = (i == 0) ? 0u : atsParent(i);
        L.mJointOffset[lane] = (i == 0) ? 0u : atsJointOffset(k, i, numLinks);

        L.mRw_x[lane] = ats(k, 50u + i * 3u);
        L.mRw_y[lane] = ats(k, 51u + i * 3u);
        L.mRw_z[lane] = ats(k, 52u + i * 3u);

        // trap 1: NON-zero scratch everywhere, including the root
        storeSpatialVector(L.mScratchImpulse, atsSV(k, 80u + i * 6u), lane);

        for (uint32_t dd = 0; dd < dofs; ++dd)
        {
            const uint32_t slot = atsJointOffset(k, i, numLinks) + dd;
            physx::PxgArticulationBlockDofData& d = df[slot];
            storeSpatialVector(d.mIsInvDW, atsSV(k, 300u + slot * 6u), lane);
            storeSpatialVector(d.mWorldMotionMatrix, atsSV(k, 400u + slot * 6u), lane);
            d.mDeferredQstZ[lane] = ats(k, 500u + slot);
        }
    }
}

// the mirror: the kernel's control flow over engine primitives
static void atsMirror(physx::PxgArticulationBlockData* ab,
                      physx::PxgArticulationBlockLinkData* lk,
                      physx::PxgArticulationBlockDofData* df, const float* zExt,
                      const float* jointForces, uint32_t lane, float stepDt)
{
    const uint32_t numLinks = ab->mNumLinks[lane];
    const bool isFixedBase = (ab->mFlags[lane] & 1u) != 0u;

    for (uint32_t linkIdx = numLinks - 1u; linkIdx > 0u; --linkIdx)
    {
        physx::PxgArticulationBlockLinkData& L = lk[linkIdx];
        const uint32_t parent = L.mParents[lane];
        const PxVec3 rw(L.mRw_x[lane], L.mRw_y[lane], L.mRw_z[lane]);
        const uint32_t dofCount = L.mDofs[lane];
        const uint32_t jointOffset = L.mJointOffset[lane];

        const float* z = zExt + size_t(linkIdx) * 6u;
        const UnAlignedSpatialVector isolatedYW(PxVec3(z[0], z[1], z[2]),
                                                PxVec3(z[3], z[4], z[5]));
        const UnAlignedSpatialVector articulatedYW =
            isolatedYW * stepDt + loadSpatialVector(L.mScratchImpulse, lane);

        const UnAlignedSpatialVector propagatedYWParent = pxgref::aiPropagateImpulseJf(
            rw, &df[jointOffset], articulatedYW, dofCount, lane, jointForces + jointOffset,
            stepDt);

        if (parent > 0u || !isFixedBase)
            storeSpatialVector(lk[parent].mScratchImpulse,
                               loadSpatialVector(lk[parent].mScratchImpulse, lane)
                                   + propagatedYWParent,
                               lane);
        storeSpatialVector(L.mScratchImpulse, UnAlignedSpatialVector::Zero(), lane);
    }

    if (!isFixedBase)
    {
        const UnAlignedSpatialVector isolatedYW(PxVec3(zExt[0], zExt[1], zExt[2]),
                                                PxVec3(zExt[3], zExt[4], zExt[5]));
        const UnAlignedSpatialVector articulatedYW =
            isolatedYW * stepDt + loadSpatialVector(lk[0].mScratchImpulse, lane);
        storeSpatialVector(ab->mRootDeferredZ,
                           loadSpatialVector(ab->mRootDeferredZ, lane) + articulatedYW, lane);
        storeSpatialVector(lk[0].mScratchImpulse, UnAlignedSpatialVector::Zero(), lane);
    }

    ab->mStateDirty[lane] = 1;   // eVEL_DIRTY, ASSIGNED
}

// ---------------------------------------------------------------------------

PXG_TEST(artiApplyTgsSubstepForces)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiAvgImpulse.spv", "artiApplyTgsSubstepForces", 16u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &artiAtsLoader);

    // coverage
    {
        uint32_t nFixed = 0, nFree = 0, nFixedMulti = 0, nBranch = 0, nZeroDof = 0;
        uint32_t nOffsetDiffers = 0, dirtySeen[4] = { 0, 0, 0, 0 };
        for (uint32_t k = 0; k < ATS_NB_ARTI; ++k)
        {
            const uint32_t nl = atsNumLinks(k);
            PXG_CHECK_MSG(atsTotalDofs(k, nl) <= ATS_MAX_DOFS, "arti %u needs %u dofs", k,
                          atsTotalDofs(k, nl));
            if (atsFixBase(k)) { ++nFixed; if (nl > 1) ++nFixedMulti; }
            else ++nFree;
            dirtySeen[atsDirtySeed(k)] += 1u;
            uint32_t cursor = 0;
            for (uint32_t i = 1; i < nl; ++i)
            {
                if (atsDofs(k, i) == 0) ++nZeroDof;
                if (atsParent(i) != i - 1) ++nBranch;
                if (atsJointOffset(k, i, nl) != cursor) ++nOffsetDiffers;
                cursor += atsDofs(k, i);
            }
        }
        std::printf("[ info ] ats: %u fixed (%u multi-link) / %u free, %u re-parented, "
                    "%u zero-dof, %u offsets off the cursor, dirty seeds 0/2/3 = %u/%u/%u\n",
                    nFixed, nFixedMulti, nFree, nBranch, nZeroDof, nOffsetDiffers, dirtySeen[0],
                    dirtySeen[2], dirtySeen[3]);
        PXG_CHECK_MSG(nFixed > 3 && nFree > 3, "the fixed-base gate is not straddled");
        PXG_CHECK_MSG(nFixedMulti > 2, "no fixed-base lane with a child (the drop gate)");
        PXG_CHECK_MSG(nBranch > 2, "no re-parented link");
        PXG_CHECK_MSG(nOffsetDiffers > 8, "the reversed windows collapse onto the cursor");
        PXG_CHECK_MSG(dirtySeen[0] > 0 && dirtySeen[2] > 0 && dirtySeen[3] > 0,
                      "mStateDirty seeds do not cover 0/2/3");
    }

    const size_t abBytes = size_t(ATS_NB_WARPS) * sizeof(physx::PxgArticulationBlockData);
    const size_t lkBytes =
        size_t(ATS_NB_WARPS) * ATS_MAX_LINKS * sizeof(physx::PxgArticulationBlockLinkData);
    const size_t dfBytes =
        size_t(ATS_NB_WARPS) * ATS_MAX_DOFS * sizeof(physx::PxgArticulationBlockDofData);
    const size_t zBytes = size_t(ATS_NB_ARTI) * ATS_MAX_LINKS * 6u * sizeof(float);
    const size_t jfBytes = size_t(ATS_NB_ARTI) * ATS_MAX_DOFS * sizeof(float);
    const size_t recBytes = size_t(ATS_NB_ARTI) * ATS_RECORD_SIZE;

    std::vector<physx::PxgArticulationBlockData> abImg(ATS_NB_WARPS);
    std::vector<physx::PxgArticulationBlockLinkData> lkImg(size_t(ATS_NB_WARPS) * ATS_MAX_LINKS);
    std::vector<physx::PxgArticulationBlockDofData> dfImg(size_t(ATS_NB_WARPS) * ATS_MAX_DOFS);
    std::vector<float> zImg(zBytes / 4), jfImg(jfBytes / 4);
    std::memset(abImg.data(), 0xAB, abBytes);
    std::memset(lkImg.data(), 0xAB, lkBytes);
    std::memset(dfImg.data(), 0xAB, dfBytes);

    for (uint32_t k = 0; k < ATS_NB_ARTI; ++k)
    {
        const uint32_t w = k / 32u, lane = k % 32u;
        atsFill(&abImg[w], &lkImg[size_t(w) * ATS_MAX_LINKS], &dfImg[size_t(w) * ATS_MAX_DOFS],
                lane, k);
        for (uint32_t i = 0; i < ATS_MAX_LINKS * 6u; ++i)
            zImg[size_t(k) * ATS_MAX_LINKS * 6u + i] = ats(k, 700u + i);
        for (uint32_t i = 0; i < ATS_MAX_DOFS; ++i)
            jfImg[size_t(k) * ATS_MAX_DOFS + i] = ats(k, 800u + i);
    }

    // the expectation
    std::vector<physx::PxgArticulationBlockData> wantAb = abImg;
    std::vector<physx::PxgArticulationBlockLinkData> wantLk = lkImg;
    std::vector<physx::PxgArticulationBlockDofData> wantDf = dfImg;
    for (uint32_t k = 0; k < ATS_NB_ARTI; ++k)
    {
        const uint32_t w = k / 32u, lane = k % 32u;
        atsMirror(&wantAb[w], &wantLk[size_t(w) * ATS_MAX_LINKS],
                  &wantDf[size_t(w) * ATS_MAX_DOFS],
                  zImg.data() + size_t(k) * ATS_MAX_LINKS * 6u,
                  jfImg.data() + size_t(k) * ATS_MAX_DOFS, lane, ATS_STEP_DT);
    }

    PxgDevicePtr dAb = 0, dLk = 0, dDf = 0, dZ = 0, dJf = 0, dRec = 0, dDesc = 0;
    shim.memAlloc(&dAb, abBytes);
    shim.memAlloc(&dLk, lkBytes);
    shim.memAlloc(&dDf, dfBytes);
    shim.memAlloc(&dZ, zBytes);
    shim.memAlloc(&dJf, jfBytes);
    shim.memAlloc(&dRec, recBytes);
    shim.memAlloc(&dDesc, ATS_DESC_SIZE);
    if (!dAb || !dLk || !dDf || !dZ || !dJf || !dRec || !dDesc)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }

    // the 544-byte record images: only the two pointers live
    std::vector<uint8_t> recImg(recBytes, 0xAB);
    for (uint32_t k = 0; k < ATS_NB_ARTI; ++k)
    {
        const uint64_t zPtr = uint64_t(dZ) + uint64_t(k) * ATS_MAX_LINKS * 6u * sizeof(float);
        const uint64_t jfPtr = uint64_t(dJf) + uint64_t(k) * ATS_MAX_DOFS * sizeof(float);
        std::memcpy(recImg.data() + size_t(k) * ATS_RECORD_SIZE + ATS_R_ZA_FORCES, &zPtr, 8);
        std::memcpy(recImg.data() + size_t(k) * ATS_RECORD_SIZE + ATS_R_JOINT_FORCE, &jfPtr, 8);
    }

    std::vector<uint8_t> descImg(ATS_DESC_SIZE, 0xAB);
    auto putU32 = [&](uint32_t off, uint32_t v) { std::memcpy(descImg.data() + off, &v, 4); };
    auto putU64 = [&](uint32_t off, uint64_t v) { std::memcpy(descImg.data() + off, &v, 8); };
    putU32(ATS_D_NB_ARTICULATIONS, ATS_NB_ARTI);
    putU32(ATS_D_MAX_LINKS, ATS_MAX_LINKS);
    putU32(ATS_D_MAX_DOFS, ATS_MAX_DOFS);
    putU64(ATS_D_ARTICULATIONS, uint64_t(dRec));
    putU64(ATS_D_ARTICULATION_BLOCKS, uint64_t(dAb));
    putU64(ATS_D_LINK_BLOCKS, uint64_t(dLk));
    putU64(ATS_D_DOF_BLOCKS, uint64_t(dDf));

    shim.memcpyHtoD(dAb, abImg.data(), abBytes);
    shim.memcpyHtoD(dLk, lkImg.data(), lkBytes);
    shim.memcpyHtoD(dDf, dfImg.data(), dfBytes);
    shim.memcpyHtoD(dZ, zImg.data(), zBytes);
    shim.memcpyHtoD(dJf, jfImg.data(), jfBytes);
    shim.memcpyHtoD(dRec, recImg.data(), recBytes);
    shim.memcpyHtoD(dDesc, descImg.data(), descImg.size());

    struct Params
    {
        uint64_t desc;
        float stepDt;
        uint32_t pad;
    } prm = { uint64_t(dDesc), ATS_STEP_DT, 0u };
    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch artiApplyTgsSubstepForces");
    shim.streamSynchronize();

    std::vector<physx::PxgArticulationBlockData> gotAb(abImg.size());
    std::vector<physx::PxgArticulationBlockLinkData> gotLk(lkImg.size());
    std::vector<physx::PxgArticulationBlockDofData> gotDf(dfImg.size());
    std::vector<uint8_t> gotRec(recBytes);
    shim.memcpyDtoH(gotAb.data(), dAb, abBytes);
    shim.memcpyDtoH(gotLk.data(), dLk, lkBytes);
    shim.memcpyDtoH(gotDf.data(), dDf, dfBytes);
    shim.memcpyDtoH(gotRec.data(), dRec, recBytes);

    size_t bad = 0, signedZeros = 0;
    auto cmp = [&](const char* what, const void* wp, const void* gp, size_t bytes) {
        const uint32_t* w = reinterpret_cast<const uint32_t*>(wp);
        const uint32_t* g = reinterpret_cast<const uint32_t*>(gp);
        for (size_t x = 0; x < bytes / 4; ++x)
        {
            if (w[x] == g[x])
                continue;
            // STRICT (P9.86c): counted apart for the info line, still a mismatch
            if ((w[x] == 0u && g[x] == 0x80000000u) || (w[x] == 0x80000000u && g[x] == 0u))
                ++signedZeros;
            if (bad < 12)
                std::printf("MM ats %s word %zu  ref %08x  gpu %08x\n", what, x, w[x], g[x]);
            ++bad;
        }
    };
    cmp("arti", wantAb.data(), gotAb.data(), abBytes);
    cmp("link", wantLk.data(), gotLk.data(), lkBytes);
    cmp("dof", wantDf.data(), gotDf.data(), dfBytes);
    // the records are READ-only for this kernel: byte-identical round trip
    cmp("record", recImg.data(), gotRec.data(), recBytes);

    std::printf("[ info ] ats: signed-zero mismatches %zu\n", signedZeros);
    PXG_CHECK_MSG(bad == 0, "the substep forces do not match the mirror bit for bit");

    shim.memFree(dAb);
    shim.memFree(dLk);
    shim.memFree(dDf);
    shim.memFree(dZ);
    shim.memFree(dJf);
    shim.memFree(dRec);
    shim.memFree(dDesc);
}

}  // namespace
