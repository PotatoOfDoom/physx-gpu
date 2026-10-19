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


// P8.106: the `artiPushImpulse` engine entry (forwardDynamic2.cu:4139) --
// leaf-to-root scratch-impulse propagation into mDeferredQstZ, then (5.12,
// OMPE-42519) the root scratch impulse flushed into mRootDeferredZ for BOTH
// base kinds -- impulse space, mRootDeferredZ += mScratchImpulse[0], scratch
// zeroed (5.6.1 stored mat * -scratch, a velocity, on free bases only).
//
// PxgArticulationCore::pushImpulse has NO caller in PhysX 5.6.1, so no scene
// can ever dispatch this kernel -- THIS fixture and its sweep are the whole
// verification.  The mirror is control flow over engine primitives
// (pxgref::aiPropagateImpulse == propagateImpulseW<WriteToDeferredQstZ> with
// a NULL joint force, loadSpatialMatrix(PxgSpatialMatrix6x6Block) and
// PxSpatialMatrix::operator* for the root).
//
// WHAT THE FIXTURE IS SHAPED TO SEE
//
//   * mStateDirty (period 4) against fixBase (period 3): dirty lanes of both
//     base kinds run the walk and the root flush; clean lanes run NOTHING (all their
//     seeded state must come back byte-identical, including mStateDirty --
//     unlike artiOutputVelocity this kernel does not clear it);
//   * mJointOffset windows assigned in REVERSED link order, so they are
//     disjoint but never equal the running forward cursor -- the field read
//     is distinguishable from a cursor;
//   * re-parented links (parent != i-1), so the parent accumulation target
//     comes from mParents and grandchild contributions stack leaf-first;
//   * mRootDeferredZ seeded with LARGE distinct values (the P8.97c poison
//     lesson): it must survive clean lanes and be ACCUMULATED (5.12) on
//     dirty lanes;
//   * two warp slices, 41 live articulations, guard lanes past the end;
//   * P9.86c, articulation PI_REST_ARTI: a RESTING fixed-base two-link
//     articulation.  Root and link 1 hold six -0 scratch words (rigidDeltaAccum
//     .cu:384's `-impulse` for a zero-impulse contact) and link 1 is a 1-dof
//     revolute joint about world -Z (motion (0,0,-1 | 0,0,0)), so its dual
//     product against the all -0 Z is exactly +0.  propagateImpulseW with a
//     NULL jointForce (forwardDynamic2.cu:4184-4186 -> articulationDynamic.cuh
//     :143-144) gives stZ = 0.0f - (+0) = +0, the preloaded `-dot` gives -0;
//     isInvD of mixed sign carries that into the root's scratch, which 5.12
//     flushes into an mRootDeferredZ seeded -0 (a resting root): -0 + x keeps
//     x, so the root deferred Z carries (+0,-0,+0 | +0,+0,+0) against
//     (-0,+0,-0 | -0,+0,+0).
//     The host twin pxgref::gAiNullJfNegDot must reproduce that split.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"
#include "pxgArtiRefHelpers.h"

#include "foundation/PxVec3.h"
#include "CmSpatialVector.h"
#include "DyFeatherstoneArticulation.h"
#include "PxgArticulation.h"
#include "PxSpatialMatrix.h"

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

static std::vector<uint8_t> artiPushLoader(const char* file) { return loadShaderBlob(file); }

// ---------------------------------------------------------------------------
// The fixture's shape (the P8.105 geometry, new seeds)
// ---------------------------------------------------------------------------

const uint32_t PI_NB_ARTI = 41;
const uint32_t PI_REST_ARTI = 40;   // P9.86c: the resting fixed-base pair (see the header)
const uint32_t PI_NB_WARPS = 2;
const uint32_t PI_MAX_LINKS = 7;
const uint32_t PI_MAX_DOFS = 15;

// PXG_ARTI_CORE_* offsets, pinned by testArtiCoreDescLayout
const uint32_t PI_D_NB_ARTICULATIONS = 44;
const uint32_t PI_D_ARTICULATION_BLOCKS = 128;
const uint32_t PI_D_LINK_BLOCKS = 136;
const uint32_t PI_D_DOF_BLOCKS = 184;
const uint32_t PI_D_MAX_LINKS = 248;
const uint32_t PI_D_MAX_DOFS = 252;
const uint32_t PI_DESC_SIZE = 344;

static uint32_t piNumLinks(uint32_t k) { return (k == PI_REST_ARTI) ? 2u : 1u + (k % 7u); }
static bool piFixBase(uint32_t k) { return (k % 3u) == 1u; }
static uint32_t piDirty(uint32_t k) { return (k + 2u) % 4u; }   // shifted vs P8.105
static uint32_t piFlags(uint32_t k)
{
    return (piFixBase(k) ? 1u : 0u) | ((k & 1u) << 1) | 4u | ((k % 3u) << 4);
}
static uint32_t piDofs(uint32_t k, uint32_t i) { return (k + i) % 4u; }
static uint32_t piParent(uint32_t i)
{
    if (i <= 1u) return 0u;
    return (i >= 3u && (i % 3u) == 0u) ? 1u : (i - 1u);
}
static uint32_t piTotalDofs(uint32_t k, uint32_t numLinks)
{
    uint32_t t = 0;
    for (uint32_t i = 1; i < numLinks; ++i)
        t += piDofs(k, i);
    return t;
}

// dof windows in REVERSED link order: disjoint, but jointOffset differs from
// the forward running cursor whenever a lane has two links with different
// dof counts (asserted below).
static uint32_t piJointOffset(uint32_t k, uint32_t i, uint32_t numLinks)
{
    uint32_t off = piTotalDofs(k, numLinks);
    for (uint32_t j = 1; j <= i; ++j)
        off -= piDofs(k, j);
    return off;
}

static float pi(uint32_t i, uint32_t k)
{
    return (float((i * 71u + k * 29u) % 499u) + 1.0f) * 0.015625f - 3.5f;
}

static UnAlignedSpatialVector piSV(uint32_t seed, uint32_t base)
{
    return UnAlignedSpatialVector(PxVec3(pi(seed, base), pi(seed, base + 1), pi(seed, base + 2)),
                                  PxVec3(pi(seed, base + 3), pi(seed, base + 4), pi(seed, base + 5)));
}

// ---------------------------------------------------------------------------

static void piFill(physx::PxgArticulationBlockData* ab,
                   physx::PxgArticulationBlockLinkData* lk,
                   physx::PxgArticulationBlockDofData* df, uint32_t lane, uint32_t k)
{
    const uint32_t numLinks = piNumLinks(k);

    ab->mNumLinks[lane] = numLinks;
    ab->mFlags[lane] = piFlags(k);
    ab->mStateDirty[lane] = uint8_t(piDirty(k));

    // LARGE seeded values (not poison): must survive fixed/clean lanes and
    // be overwritten -- never accumulated -- on free dirty lanes
    storeSpatialVector(ab->mRootDeferredZ,
                       UnAlignedSpatialVector(PxVec3(100.0f + pi(k, 0), 200.0f + pi(k, 1),
                                                     300.0f + pi(k, 2)),
                                              PxVec3(400.0f + pi(k, 3), 500.0f + pi(k, 4),
                                                     600.0f + pi(k, 5))),
                       lane);

    for (uint32_t i = 0; i < numLinks; ++i)
    {
        physx::PxgArticulationBlockLinkData& L = lk[i];
        const uint32_t dofs = (i == 0) ? 0u : piDofs(k, i);

        L.mDofs[lane] = uint8_t(dofs);
        L.mParents[lane] = (i == 0) ? 0u : piParent(i);
        L.mJointOffset[lane] = (i == 0) ? 0u : piJointOffset(k, i, numLinks);

        L.mRw_x[lane] = pi(k, 50u + i * 3u);
        L.mRw_y[lane] = pi(k, 51u + i * 3u);
        L.mRw_z[lane] = pi(k, 52u + i * 3u);

        storeSpatialVector(L.mScratchImpulse, piSV(k, 80u + i * 6u), lane);

        for (uint32_t dd = 0; dd < dofs; ++dd)
        {
            const uint32_t slot = piJointOffset(k, i, numLinks) + dd;
            physx::PxgArticulationBlockDofData& d = df[slot];
            storeSpatialVector(d.mIsInvDW, piSV(k, 300u + slot * 6u), lane);
            storeSpatialVector(d.mWorldMotionMatrix, piSV(k, 400u + slot * 6u), lane);
            d.mDeferredQstZ[lane] = pi(k, 500u + slot);
        }
    }

    if (k == PI_REST_ARTI)
    {
        const UnAlignedSpatialVector negZero(PxVec3(-0.0f, -0.0f, -0.0f),
                                             PxVec3(-0.0f, -0.0f, -0.0f));
        storeSpatialVector(lk[0].mScratchImpulse, negZero, lane);
        storeSpatialVector(lk[1].mScratchImpulse, negZero, lane);
        storeSpatialVector(ab->mRootDeferredZ, negZero, lane);   // a resting root
        physx::PxgArticulationBlockDofData& d = df[lk[1].mJointOffset[lane]];
        storeSpatialVector(d.mWorldMotionMatrix,
                           UnAlignedSpatialVector(PxVec3(0.0f, 0.0f, -1.0f), PxVec3(0.0f, 0.0f, 0.0f)),
                           lane);
        storeSpatialVector(d.mIsInvDW,
                           UnAlignedSpatialVector(PxVec3(0.75f, -0.5f, 0.25f),
                                                  PxVec3(0.125f, -0.375f, 0.625f)),
                           lane);
        d.mDeferredQstZ[lane] = 0.0f;   // freshly zeroed (forwardDynamic2.cu:97)
    }

    // the root's 6x6 response matrix, dense and non-symmetric
    {
        physx::PxSpatialMatrix m;
        for (uint32_t c = 0; c < 6; ++c)
            for (uint32_t r = 0; r < 6; ++r)
                m.column[c][r] = pi(k, 600u + c * 6u + r);
        physx::storeSpatialMatrix(lk[0].mSpatialResponseMatrix, lane, m);
    }
}

// the mirror: the CUDA's structure over engine primitives
static void piMirror(physx::PxgArticulationBlockData* ab,
                     physx::PxgArticulationBlockLinkData* lk,
                     physx::PxgArticulationBlockDofData* df, uint32_t lane)
{
    if (ab->mStateDirty[lane] == 0)
        return;

    const uint32_t nbLinks = ab->mNumLinks[lane];
    for (uint32_t index = nbLinks - 1u; index > 0u; --index)
    {
        physx::PxgArticulationBlockLinkData& L = lk[index];
        physx::PxgArticulationBlockDofData* dofData = &df[L.mJointOffset[lane]];
        const uint32_t parent = L.mParents[lane];
        const PxVec3 rw(L.mRw_x[lane], L.mRw_y[lane], L.mRw_z[lane]);
        const uint32_t dofCount = L.mDofs[lane];

        const UnAlignedSpatialVector Z = loadSpatialVector(L.mScratchImpulse, lane);
        const UnAlignedSpatialVector propagatedZ =
            pxgref::aiPropagateImpulse(rw, dofData, Z, dofCount, lane);

        storeSpatialVector(lk[parent].mScratchImpulse,
                           loadSpatialVector(lk[parent].mScratchImpulse, lane) + propagatedZ,
                           lane);
        storeSpatialVector(L.mScratchImpulse, UnAlignedSpatialVector::Zero(), lane);
    }

    // 5.12 (OMPE-42519): flush the root scratch impulse into mRootDeferredZ,
    // for free and fixed bases alike (addSpatialVector, then zero the scratch)
    {
        const UnAlignedSpatialVector rootZ = loadSpatialVector(lk[0].mScratchImpulse, lane);
        storeSpatialVector(ab->mRootDeferredZ,
                           loadSpatialVector(ab->mRootDeferredZ, lane) + rootZ, lane);
        storeSpatialVector(lk[0].mScratchImpulse, UnAlignedSpatialVector::Zero(), lane);
    }
}

// ---------------------------------------------------------------------------

PXG_TEST(artiPushImpulse)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiAvgImpulse.spv", "artiPushImpulse", 8u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &artiPushLoader);

    // coverage
    {
        uint32_t nDirtyFixed = 0, nDirtyFree = 0, nClean = 0, nSingleDirty = 0;
        uint32_t nBranch = 0, nZeroDof = 0, nOffsetDiffers = 0;
        for (uint32_t k = 0; k < PI_NB_ARTI; ++k)
        {
            const uint32_t nl = piNumLinks(k);
            PXG_CHECK_MSG(piTotalDofs(k, nl) <= PI_MAX_DOFS, "arti %u needs %u dofs", k,
                          piTotalDofs(k, nl));
            const bool dirty = piDirty(k) != 0u;
            if (dirty) { if (piFixBase(k)) ++nDirtyFixed; else ++nDirtyFree; }
            else ++nClean;
            if (nl == 1 && dirty) ++nSingleDirty;
            uint32_t cursor = 0;
            for (uint32_t i = 1; i < nl; ++i)
            {
                if (piDofs(k, i) == 0) ++nZeroDof;
                if (piParent(i) != i - 1) ++nBranch;
                if (piJointOffset(k, i, nl) != cursor) ++nOffsetDiffers;
                cursor += piDofs(k, i);
            }
        }
        std::printf("[ info ] pi: %u dirty-fixed / %u dirty-free / %u clean, %u single-dirty, "
                    "%u re-parented, %u zero-dof, %u joint offsets off the forward cursor\n",
                    nDirtyFixed, nDirtyFree, nClean, nSingleDirty, nBranch, nZeroDof,
                    nOffsetDiffers);
        PXG_CHECK_MSG(nDirtyFixed > 3 && nDirtyFree > 3, "the fixed-base tail is not straddled");
        PXG_CHECK_MSG(nClean > 3, "too few clean lanes");
        PXG_CHECK_MSG(nSingleDirty > 0, "no dirty single-link lane");
        PXG_CHECK_MSG(nBranch > 2, "no re-parented link");
        PXG_CHECK_MSG(nOffsetDiffers > 8, "the reversed windows collapse onto the cursor");
        PXG_CHECK_MSG(piFixBase(PI_REST_ARTI) && piDirty(PI_REST_ARTI) != 0u &&
                          piNumLinks(PI_REST_ARTI) == 2u && piDofs(PI_REST_ARTI, 1) == 1u,
                      "the resting articulation is not a dirty fixed-base one-dof pair");
    }

    const size_t abBytes = size_t(PI_NB_WARPS) * sizeof(physx::PxgArticulationBlockData);
    const size_t lkBytes =
        size_t(PI_NB_WARPS) * PI_MAX_LINKS * sizeof(physx::PxgArticulationBlockLinkData);
    const size_t dfBytes =
        size_t(PI_NB_WARPS) * PI_MAX_DOFS * sizeof(physx::PxgArticulationBlockDofData);

    std::vector<physx::PxgArticulationBlockData> abImg(PI_NB_WARPS);
    std::vector<physx::PxgArticulationBlockLinkData> lkImg(size_t(PI_NB_WARPS) * PI_MAX_LINKS);
    std::vector<physx::PxgArticulationBlockDofData> dfImg(size_t(PI_NB_WARPS) * PI_MAX_DOFS);
    std::memset(abImg.data(), 0xAB, abBytes);
    std::memset(lkImg.data(), 0xAB, lkBytes);
    std::memset(dfImg.data(), 0xAB, dfBytes);

    for (uint32_t k = 0; k < PI_NB_ARTI; ++k)
    {
        const uint32_t w = k / 32u, lane = k % 32u;
        piFill(&abImg[w], &lkImg[size_t(w) * PI_MAX_LINKS], &dfImg[size_t(w) * PI_MAX_DOFS],
               lane, k);
    }

    std::vector<physx::PxgArticulationBlockData> wantAb = abImg;
    std::vector<physx::PxgArticulationBlockLinkData> wantLk = lkImg;
    std::vector<physx::PxgArticulationBlockDofData> wantDf = dfImg;
    pxgref::gAiPosZeroDotHits = 0;
    for (uint32_t k = 0; k < PI_NB_ARTI; ++k)
    {
        const uint32_t w = k / 32u, lane = k % 32u;
        piMirror(&wantAb[w], &wantLk[size_t(w) * PI_MAX_LINKS], &wantDf[size_t(w) * PI_MAX_DOFS],
                 lane);
    }
    const uint32_t posZeroDotHits = pxgref::gAiPosZeroDotHits;

    // P9.86c: the `-dot` twin over the same INPUTS must split the resting
    // articulation's root scratch exactly as derived in the header.
    {
        std::vector<physx::PxgArticulationBlockData> twAb = abImg;
        std::vector<physx::PxgArticulationBlockLinkData> twLk = lkImg;
        std::vector<physx::PxgArticulationBlockDofData> twDf = dfImg;
        pxgref::gAiNullJfNegDot = true;
        for (uint32_t k = 0; k < PI_NB_ARTI; ++k)
        {
            const uint32_t w = k / 32u, lane = k % 32u;
            piMirror(&twAb[w], &twLk[size_t(w) * PI_MAX_LINKS], &twDf[size_t(w) * PI_MAX_DOFS],
                     lane);
        }
        pxgref::gAiNullJfNegDot = false;
        size_t twinWords = 0;
        auto diffWords = [&](const void* a, const void* b, size_t bytes) {
            const uint32_t* x = reinterpret_cast<const uint32_t*>(a);
            const uint32_t* y = reinterpret_cast<const uint32_t*>(b);
            for (size_t i = 0; i < bytes / 4; ++i) twinWords += (x[i] != y[i]) ? 1u : 0u;
        };
        diffWords(wantAb.data(), twAb.data(), abBytes);
        diffWords(wantLk.data(), twLk.data(), lkBytes);
        diffWords(wantDf.data(), twDf.data(), dfBytes);
        const uint32_t rw = PI_REST_ARTI / 32u, rl = PI_REST_ARTI % 32u;
        (void)PI_MAX_LINKS;
        const UnAlignedSpatialVector a = loadSpatialVector(wantAb[rw].mRootDeferredZ, rl);
        const UnAlignedSpatialVector b = loadSpatialVector(twAb[rw].mRootDeferredZ, rl);
        auto bits = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; };
        std::printf("[ info ] pi: +0 dual products %u; -dot twin changes %zu words; resting root "
                    "deferredZ top %08x %08x %08x bottom.x %08x (twin %08x %08x %08x / %08x)\n",
                    posZeroDotHits, twinWords, bits(a.top.x), bits(a.top.y), bits(a.top.z),
                    bits(a.bottom.x), bits(b.top.x), bits(b.top.y), bits(b.top.z), bits(b.bottom.x));
        PXG_CHECK_MSG(posZeroDotHits > 0, "no +0 dual product reached the push hop");
        PXG_CHECK_MSG(twinWords == 4 && bits(a.top.x) == 0x00000000u && bits(a.top.y) == 0x80000000u &&
                          bits(a.top.z) == 0x00000000u && bits(a.bottom.x) == 0x00000000u &&
                          bits(b.top.x) == 0x80000000u && bits(b.top.y) == 0x00000000u &&
                          bits(b.top.z) == 0x80000000u && bits(b.bottom.x) == 0x80000000u,
                      "resting articulation no longer separates `0 - dot` from `-dot` (%zu words)",
                      twinWords);
    }

    PxgDevicePtr dAb = 0, dLk = 0, dDf = 0, dDesc = 0;
    shim.memAlloc(&dAb, abBytes);
    shim.memAlloc(&dLk, lkBytes);
    shim.memAlloc(&dDf, dfBytes);
    shim.memAlloc(&dDesc, PI_DESC_SIZE);
    if (!dAb || !dLk || !dDf || !dDesc)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }

    std::vector<uint8_t> descImg(PI_DESC_SIZE, 0xAB);
    auto putU32 = [&](uint32_t off, uint32_t v) { std::memcpy(descImg.data() + off, &v, 4); };
    auto putU64 = [&](uint32_t off, uint64_t v) { std::memcpy(descImg.data() + off, &v, 8); };
    putU32(PI_D_NB_ARTICULATIONS, PI_NB_ARTI);
    putU32(PI_D_MAX_LINKS, PI_MAX_LINKS);
    putU32(PI_D_MAX_DOFS, PI_MAX_DOFS);
    putU64(PI_D_ARTICULATION_BLOCKS, uint64_t(dAb));
    putU64(PI_D_LINK_BLOCKS, uint64_t(dLk));
    putU64(PI_D_DOF_BLOCKS, uint64_t(dDf));
    shim.memcpyHtoD(dDesc, descImg.data(), descImg.size());

    shim.memcpyHtoD(dAb, abImg.data(), abBytes);
    shim.memcpyHtoD(dLk, lkImg.data(), lkBytes);
    shim.memcpyHtoD(dDf, dfImg.data(), dfBytes);

    struct Params
    {
        uint64_t desc;
    } prm = { uint64_t(dDesc) };
    PXG_CHECK_MSG(shim.launchKernel(0, PI_NB_WARPS, 1, 1, &prm, sizeof(prm)) ==
                      PxgResult::Success,
                  "launch artiPushImpulse");
    shim.streamSynchronize();

    std::vector<physx::PxgArticulationBlockData> gotAb(abImg.size());
    std::vector<physx::PxgArticulationBlockLinkData> gotLk(lkImg.size());
    std::vector<physx::PxgArticulationBlockDofData> gotDf(dfImg.size());
    shim.memcpyDtoH(gotAb.data(), dAb, abBytes);
    shim.memcpyDtoH(gotLk.data(), dLk, lkBytes);
    shim.memcpyDtoH(gotDf.data(), dDf, dfBytes);

    // bit for bit, the sign of zero included (P9.86c: the P8.84 signed-zero
    // tolerance is gone -- the 6x6 product's literal-zero seed is spelled as
    // the IEEE select in asmM6MulVec since P9.86)
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
                std::printf("MM pi %s word %zu  ref %08x  gpu %08x\n", what, x, w[x], g[x]);
            ++bad;
        }
    };
    cmp("arti", wantAb.data(), gotAb.data(), abBytes);
    cmp("link", wantLk.data(), gotLk.data(), lkBytes);
    cmp("dof", wantDf.data(), gotDf.data(), dfBytes);

    std::printf("[ info ] pi: signed-zero mismatches %zu\n", signedZeros);
    PXG_CHECK_MSG(bad == 0, "the pushed impulses do not match the mirror bit for bit");

    shim.memFree(dAb);
    shim.memFree(dLk);
    shim.memFree(dDf);
    shim.memFree(dDesc);
}

}  // namespace
