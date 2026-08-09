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

// P8.80: initialize() and the block-header fill.
//
// Between them these two functions touch FOUR element strides, and that is the
// whole risk surface -- there is no arithmetic here at all:
//
//   PxU32[32] / PxReal[32]   4 bytes   counters, header words
//   PxU8[32]                 1 byte    mStateDirty -- four lanes per word
//   float4[32]              16 bytes   mDeltaQ
//   pointer[32]              8 bytes   mMotionVelocitiesPtr
//
// THE ONE NON-ZERO WRITE in a function whose entire job is zeroing is `mDeltaQ`,
// set to the IDENTITY QUATERNION (0,0,0,1).  A memset-shaped port yields a zero
// quaternion -- not a rotation at all -- and nothing complains until something
// rotates by it.  The fixture pins all four components.
//
// LOOP BOUNDS ARE ALL DISTINCT: numLinks 3, totalDofs 5, lanes 5... no: lanes 5
// would coincide with totalDofs, so lanes is 5 and totalDofs is 6.  P8.79's sweep
// lost two mutants to numLinks == totalDofs and P8.78's would have lost two to
// mNbElements == maxNumAttachments; three bounds that can be confused need three
// different values.
//
// Two sub-cases: with and without the header fill, because the header is guarded
// by `refillBlockData || confiDirty` in the engine while initialize() is
// unconditional.  Without it, every header field must stay at poison.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "PxArticulationTendonData.h"
#include "PxgArticulation.h"
#include "PxgArticulationLink.h"

#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

static std::vector<uint8_t> jcalcInitLoader(const char* file) { return loadShaderBlob(file); }

static const uint32_t LANES = 5;
static const uint32_t NB_LINKS = 3;
static const uint32_t TOTAL_DOFS = 6;   // != NB_LINKS and != LANES, deliberately
// The link block ARRAY is sized past numLinks on purpose.  The first sweep had "the
// per-link loop runs over totalDofs" survive: with only NB_LINKS blocks allocated,
// the extra iterations wrote past the end of the checked region entirely and the
// comparison never saw them.  An overrun has to land somewhere the test looks.
static const uint32_t LINK_BLOCKS = 6;
static const uint32_t GUARD = 128;

PXG_TEST(artiJcalcInit)
{
    using namespace physx;

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        // { u64 x4, u32, u32, u32, f32, u32, u32 }
        { "npArtiJointFrame.spv", "artiJcalcInit", 56u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &jcalcInitLoader);

    const uint32_t BASE_INDEX = 0x00ABCD00u;
    const float BASE_SLEEP = 12.5f;

    // the record: only the header fill reads it
    // POISONED, not zeroed.  `flags` is a PxU8 at offset 26 with confiDirty at 27
    // and numPathToRoots at 28; with those zero, reading flags as a word gives the
    // right answer and the wrong-width mutant survives.  Fourth time in this chain.
    std::vector<PxgArticulation> recs(LANES);
    std::memset(recs.data(), 0xEE, recs.size() * sizeof(PxgArticulation));

    const size_t LINK_BLOCK = sizeof(PxgArticulationBlockLinkData);
    const size_t DOF_BLOCK = sizeof(PxgArticulationBlockDofData);
    const size_t AB_SIZE = sizeof(PxgArticulationBlockData);
    const size_t LINK_REGION = size_t(LINK_BLOCKS) * LINK_BLOCK + GUARD;
    const size_t DOF_REGION = size_t(TOTAL_DOFS) * DOF_BLOCK + GUARD;
    const size_t AB_REGION = AB_SIZE + GUARD;

    PxgDevicePtr dArtis = 0, dAB = 0, dLink = 0, dDof = 0, dMotionVel = 0;
    shim.memAlloc(&dArtis, size_t(LANES) * sizeof(PxgArticulation));
    shim.memAlloc(&dAB, AB_REGION);
    shim.memAlloc(&dLink, LINK_REGION);
    shim.memAlloc(&dDof, DOF_REGION);
    shim.memAlloc(&dMotionVel, size_t(LANES) * NB_LINKS * sizeof(Cm::UnAlignedSpatialVector));
    if (!dArtis || !dAB || !dLink || !dDof || !dMotionVel)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }

    for (uint32_t L = 0; L < LANES; ++L)
    {
        recs[L].data.numLinks = PxU16(NB_LINKS);
        recs[L].data.numJointDofs = PxU16(TOTAL_DOFS);
        recs[L].data.flags = PxU8(0x11u + L);
        recs[L].motionVelocities = reinterpret_cast<Cm::UnAlignedSpatialVector*>(
            dMotionVel + PxgDevicePtr(size_t(L) * NB_LINKS * sizeof(Cm::UnAlignedSpatialVector)));
    }
    shim.memcpyHtoD(dArtis, recs.data(), recs.size() * sizeof(PxgArticulation));

    size_t bad = 0;

    for (uint32_t pass = 0; pass < 2; ++pass)
    {
        const bool doHeader = (pass == 0);
        const char* name = doHeader ? "header + initialize" : "initialize only";

        std::vector<uint8_t> pAB(AB_REGION, 0xAB), pLink(LINK_REGION, 0xAB),
                             pDof(DOF_REGION, 0xAB);
        shim.memcpyHtoD(dAB, pAB.data(), pAB.size());
        shim.memcpyHtoD(dLink, pLink.data(), pLink.size());
        shim.memcpyHtoD(dDof, pDof.data(), pDof.size());

        struct Params
        {
            uint64_t artis, artiBlocks, linkBlocks, dofBlocks;
            uint32_t numLinks, totalDofs, articulationIndex;
            float sleepThreshold;
            uint32_t nbArticulations, doHeader;
        } prm = { uint64_t(dArtis), uint64_t(dAB), uint64_t(dLink), uint64_t(dDof),
                  NB_LINKS, TOTAL_DOFS, BASE_INDEX, BASE_SLEEP, LANES, doHeader ? 1u : 0u };

        PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                      "launch artiJcalcInit");
        shim.streamSynchronize();

        std::vector<uint8_t> wAB = pAB, wLink = pLink, wDof = pDof;
        auto putU8 = [](std::vector<uint8_t>& img, size_t off, uint8_t v) { img[off] = v; };
        auto putU32 = [](std::vector<uint8_t>& img, size_t off, uint32_t v) {
            std::memcpy(img.data() + off, &v, 4);
        };
        auto putU64 = [](std::vector<uint8_t>& img, size_t off, uint64_t v) {
            std::memcpy(img.data() + off, &v, 8);
        };
        auto putF = [](std::vector<uint8_t>& img, size_t off, float v) {
            std::memcpy(img.data() + off, &v, 4);
        };
        auto zeroSV = [&](std::vector<uint8_t>& img, size_t base, size_t field, uint32_t L) {
            for (uint32_t c = 0; c < 4; ++c)
                putF(img, base + field + 0 + size_t(L) * 16 + size_t(c) * 4, 0.0f);
            putF(img, base + field + 512 + size_t(L) * 8 + 0, 0.0f);
            putF(img, base + field + 512 + size_t(L) * 8 + 4, 0.0f);
        };

        typedef PxgArticulationBlockLinkData LK;
        typedef PxgArticulationBlockDofData DF;
        typedef PxgArticulationBlockData AB;

        for (uint32_t L = 0; L < LANES; ++L)
        {
            for (uint32_t i = 0; i < NB_LINKS; ++i)
            {
                const size_t blk = size_t(i) * LINK_BLOCK;
                zeroSV(wLink, blk, offsetof(LK, mDeltaMotion), L);
                zeroSV(wLink, blk, offsetof(LK, mPosMotionVelocity), L);
                // the identity quaternion -- the one non-zero write
                putF(wLink, blk + offsetof(LK, mDeltaQ) + size_t(L) * 16 + 0, 0.0f);
                putF(wLink, blk + offsetof(LK, mDeltaQ) + size_t(L) * 16 + 4, 0.0f);
                putF(wLink, blk + offsetof(LK, mDeltaQ) + size_t(L) * 16 + 8, 0.0f);
                putF(wLink, blk + offsetof(LK, mDeltaQ) + size_t(L) * 16 + 12, 1.0f);
                zeroSV(wLink, blk, offsetof(LK, mScratchImpulse), L);
                zeroSV(wLink, blk, offsetof(LK, mSolverSpatialDeltaVel), L);
                zeroSV(wLink, blk, offsetof(LK, mSolverSpatialImpulse), L);
                zeroSV(wLink, blk, offsetof(LK, mSolverSpatialInternalConstraintImpulse), L);
                zeroSV(wLink, blk, offsetof(LK, mConstraintForces), L);
                putU32(wLink, blk + offsetof(LK, mStaticContactStartIndex) + size_t(L) * 4, 0u);
                putU32(wLink, blk + offsetof(LK, mNbStaticContacts) + size_t(L) * 4, 0u);
                putU32(wLink, blk + offsetof(LK, mStaticJointStartIndex) + size_t(L) * 4, 0u);
                putU32(wLink, blk + offsetof(LK, mNbStaticJoints) + size_t(L) * 4, 0u);
                putF(wLink, blk + offsetof(LK, mDeltaScale) + size_t(L) * 4, 0.0f);
            }

            zeroSV(wAB, 0, offsetof(AB, mRootDeferredZ), L);
            zeroSV(wAB, 0, offsetof(AB, mCommonLinkDeltaVelocity), L);
            putU32(wAB, offsetof(AB, mLinkWithDeferredImpulse) + size_t(L) * 4, 0u);

            for (uint32_t i = 0; i < TOTAL_DOFS; ++i)
            {
                const size_t dof = size_t(i) * DOF_BLOCK;
                putF(wDof, dof + offsetof(DF, mDeferredQstZ) + size_t(L) * 4, 0.0f);
                putF(wDof, dof + offsetof(DF, mInvStIsT_x) + size_t(L) * 4, 0.0f);
                putF(wDof, dof + offsetof(DF, mInvStIsT_y) + size_t(L) * 4, 0.0f);
                putF(wDof, dof + offsetof(DF, mInvStIsT_z) + size_t(L) * 4, 0.0f);
            }

            if (doHeader)
            {
                putU32(wAB, offsetof(AB, mNumLinks) + size_t(L) * 4, NB_LINKS);
                putU32(wAB, offsetof(AB, mTotalDofs) + size_t(L) * 4, TOTAL_DOFS);
                putU32(wAB, offsetof(AB, mFlags) + size_t(L) * 4, uint32_t(recs[L].data.flags));
                putU32(wAB, offsetof(AB, mArticulationIndex) + size_t(L) * 4, BASE_INDEX + L);
                putF(wAB, offsetof(AB, mSleepThreshold) + size_t(L) * 4, BASE_SLEEP + float(L));
                putU8(wAB, offsetof(AB, mStateDirty) + L, 0u);
                putU64(wAB, offsetof(AB, mMotionVelocitiesPtr) + size_t(L) * 8,
                       uint64_t(dMotionVel) +
                           uint64_t(size_t(L) * NB_LINKS * sizeof(Cm::UnAlignedSpatialVector)));
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
        region("artiBlock", dAB, wAB, AB_SIZE);
        region("linkBlocks", dLink, wLink, LINK_BLOCK);
        region("dofBlocks", dDof, wDof, DOF_BLOCK);
    }

    shim.memFree(dArtis);
    shim.memFree(dAB);
    shim.memFree(dLink);
    shim.memFree(dDof);
    shim.memFree(dMotionVel);

    PXG_CHECK_MSG(bad == 0, "%zu byte(s) wrong", bad);
    std::printf("  jcalc initialize + header: 5 articulations, 3 links, 6 dofs (all bounds "
                "distinct) -- mDeltaQ is the identity quaternion, mStateDirty a byte lane, "
                "mMotionVelocitiesPtr an 8-byte lane; header absent when not requested\n");
}

} // namespace
