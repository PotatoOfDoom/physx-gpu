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


// P8.94c: npArtiAvgImpulse.slang — `averageLinkImpulsesAndPropagate`.
//
// ===========================================================================
// THE ORACLE
// ===========================================================================
//
// The function is `static __device__`, so the mirror below is CONTROL FLOW over
// engine primitives: `UnAlignedSpatialVector::innerProduct`,
// `Dy::FeatherstoneArticulation::translateSpatialVector`, `loadSpatialVector`
// and `storeSpatialVector` are all `PX_CUDA_CALLABLE` and are called rather
// than transcribed.
//
// ONE THING THE MIRROR DELIBERATELY DOES NOT SIMPLIFY.  The shader collapses
// the CUDA's `ptr & 0xf` float2/float4 alignment branch into six plain float
// loads and six zero stores, on the argument that both branches touch the same
// 24 bytes and map them to the same six components.  The mirror keeps the
// BRANCH, literally, so that argument is CHECKED rather than asserted — and the
// impulse array is laid out so both alignments occur (24-byte elements from a
// 16-byte-aligned base alternate 16- and 8-aligned), which the test counts and
// requires.
//
// ===========================================================================
// WHAT THE FIXTURE IS SHAPED TO SEE
// ===========================================================================
//
//   * **two dispatches, `nbSlabs` 4 and 1**, because the whole scale
//     computation sits behind `nbSlabs > 1`;
//   * per-lane dirty counts of 0, 1, 2, 3 and 4, so `scale` takes the values
//     1 (count 0), 1 (count 1 — the `> 1` guard, not `>= 1`), 1/2, 1/3 and 1/4
//     and a mutant that drops the guard is visible on the count-1 lanes alone;
//   * dirty pairs with **only `.x` set, only `.y` set, both set and neither**,
//     so the two halves are separable and the unconditional reset is visible on
//     the neither case;
//   * `mLinkWithDeferredImpulse` seeded **above** every dirty index on some
//     lanes and at zero on others, so the seed is distinguishable from a
//     zero-initialised max;
//   * lanes where `maxIndex` ends at **zero**, so the propagate loop is skipped
//     entirely while the root tail must still run;
//   * a **branching parent chain** and dof counts of 0..3, so the backwards
//     `dofData` cursor and its decrement by the PREFETCHED count are exercised;
//   * everything poisoned at 0xAB and compared bit for bit -- STRICTLY, the
//     sign of zero included (P9.86c);
//   * P9.86c, lane AI_REST_LANE: the EQUIVALENCE witness for the Ex walk's
//     NULL-jointForce hop (articulationImpulseResponse.cuh:236-238,
//     `stZ = 0.0f - dot`, articulationDynamic.cuh:143-144).  A resting
//     two-link articulation: both scratch impulses six -0 (rigidDeltaAccum.cu
//     :384's `-impulse` for a zero-impulse contact), link 1 a revolute joint
//     about world -Z (motion (0,0,-1 | 0,0,0)) so its dual product is exactly
//     +0, isInvD of mixed sign, and every accumulator the walk adds into at
//     its most sensitive VALID value, +0 (mSolverSpatialImpulse, mDeferredQstZ
//     and mRootDeferredZ are only ever zeroed or grown by adds in the CUDA, so
//     none of them can hold -0).  The preloaded `-dot` twin
//     (pxgref::gAiNullJfNegDot) flips the root scratch's zero signs, yet every
//     output word must come out identical -- the walk zeroes each scratch it
//     reads and the +0 accumulators absorb the sign.  A CONTROL run with an
//     INVALID -0 mRootDeferredZ seed must split, proving the lane does carry
//     the sign as far as the last absorbing add.

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

static std::vector<uint8_t> artiAvgLoader(const char* file) { return loadShaderBlob(file); }

const uint32_t AI_LANES = 20;
const uint32_t AI_REST_LANE = 19;   // P9.86c: the resting two-link lane (see the header)
const uint32_t AI_MAX_LINKS = 6;
const uint32_t AI_MAX_DOFS = 13;
const uint32_t AI_NONE = 0xFFFFFFFFu;

// 2..6, never equal to the lane index: `2 + (k % 5)` differs from `k` by
// `2 - 5*floor(k/5)`, never zero.
// (The resting lane is a two-link articulation.)
static uint32_t aiNumLinks(uint32_t k) { return (k == AI_REST_LANE) ? 2u : 2u + (k % 5u); }

// 0..3, keyed on `(k + i)` so a lane sees several counts and a zero-dof link
// (which does not move the cursor) sits between two that do.
// (The resting lane's link 1 is a 1-dof revolute joint.)
static uint32_t aiDofs(uint32_t k, uint32_t i)
{
    return (k == AI_REST_LANE && i == 1u) ? 1u : (k + i) % 4u;
}

// Every third link from 3 on re-parents to link 1 rather than to `i - 1`.
static uint32_t aiParent(uint32_t i)
{
    if (i <= 1u) return 0u;
    return (i >= 3u && (i % 3u) == 0u) ? 1u : (i - 1u);
}

static uint32_t aiTotalDofs(uint32_t k, uint32_t numLinks)
{
    uint32_t t = 0;
    for (uint32_t i = 1; i < numLinks; ++i) t += aiDofs(k, i);
    return t;
}

// How many of this lane's slabs carry a dirty index: 0..4, cycling so every
// count occurs and `scale` therefore takes every one of its distinct values.
// (The resting lane carries no slab impulse: the -0 scratch must reach the walk.)
static uint32_t aiDirtySlabs(uint32_t k) { return (k == AI_REST_LANE) ? 0u : k % 5u; }

static float ai(uint32_t i, uint32_t k)
{
    return (float((i * 53u + k * 23u) % 463u) + 1.0f) * 0.0078125f - 1.75f;
}

// ---------------------------------------------------------------------------
// The mirror: the CUDA's structure, the engine's arithmetic
// ---------------------------------------------------------------------------

// propagateImpulseW<WriteToDeferredQstZ>, jointForce NULL.
using pxgref::aiPropagateImpulse;
using pxgref::aiMirror;


// ---------------------------------------------------------------------------

static void aiRun(PxgCudaShim& shim, uint32_t nbSlabs, size_t& bad, size_t& signedZeros,
                  uint32_t& alignedHits, uint32_t& unalignedHits)
{
    std::vector<physx::PxgArticulationBlockData> abImg(1);
    std::vector<physx::PxgArticulationBlockLinkData> lkImg(AI_MAX_LINKS);
    std::vector<physx::PxgArticulationBlockDofData> dfImg(AI_MAX_DOFS);
    std::memset(abImg.data(), 0xAB, abImg.size() * sizeof(physx::PxgArticulationBlockData));
    std::memset(lkImg.data(), 0xAB, lkImg.size() * sizeof(physx::PxgArticulationBlockLinkData));
    std::memset(dfImg.data(), 0xAB, dfImg.size() * sizeof(physx::PxgArticulationBlockDofData));

    // isSlabDirty is uint2[nbSlabs * nbArticulations]; impulses is a 24-byte
    // spatial vector array of nbSlabs * maxLinks * nbArticulations entries.
    std::vector<uint32_t> dirty(size_t(nbSlabs) * AI_LANES * 2, AI_NONE);
    std::vector<float> imp(size_t(nbSlabs) * AI_MAX_LINKS * AI_LANES * 6, 0.f);
    for (size_t k = 0; k < imp.size(); ++k)
        imp[k] = ai(uint32_t(k % 977u), uint32_t(k / 977u)) * 1.5f;

    std::vector<uint32_t> counts(32, 0u);
    for (uint32_t lane = 0; lane < AI_LANES; ++lane)
    {
        const uint32_t nl = aiNumLinks(lane);
        counts[lane] = nl;

        // the dirty slabs for this lane
        const uint32_t nd = aiDirtySlabs(lane) > nbSlabs ? nbSlabs : aiDirtySlabs(lane);
        for (uint32_t s = 0; s < nd; ++s)
        {
            const size_t di = size_t(s) * AI_LANES + lane;
            // rotate through: only .x, only .y, both -- so the two halves are
            // separable and the pair is not a single flag
            const uint32_t mode = (lane + s) % 3u;
            const uint32_t a = 1u + ((lane + s) % (nl - 1u));           // 1 .. nl-1
            const uint32_t b = 1u + ((lane + s + 1u) % (nl - 1u));
            if (mode != 1u) dirty[di * 2 + 0] = a;
            if (mode != 0u) dirty[di * 2 + 1] = b;
        }

        // `mLinkWithDeferredImpulse`: above every dirty index on some lanes,
        // zero on others, so the seed is distinguishable from a zeroed max
        // (the resting lane seeds its only child so the walk runs)
        abImg[0].mLinkWithDeferredImpulse[lane] =
            (lane == AI_REST_LANE || (lane % 4u) == 2u) ? (nl - 1u) : 0u;

        storeSpatialVector(abImg[0].mRootDeferredZ,
                           UnAlignedSpatialVector(PxVec3(ai(lane, 0), ai(lane, 1), ai(lane, 2)),
                                                  PxVec3(ai(lane, 3), ai(lane, 4), ai(lane, 5))),
                           lane);

        uint32_t cursor = 0;
        for (uint32_t i = 0; i < nl; ++i)
        {
            physx::PxgArticulationBlockLinkData& L = lkImg[i];
            const uint32_t dofs = (i == 0) ? 0u : aiDofs(lane, i);
            L.mDofs[lane] = uint8_t(dofs);
            L.mParents[lane] = (i == 0) ? 0u : aiParent(i);
            L.mJointOffset[lane] = cursor;
            L.mRw_x[lane] = ai(lane, 10u + i * 3u);
            L.mRw_y[lane] = ai(lane, 11u + i * 3u);
            L.mRw_z[lane] = ai(lane, 12u + i * 3u);
            storeSpatialVector(L.mScratchImpulse,
                               UnAlignedSpatialVector(PxVec3(ai(lane, 40u + i * 6u),
                                                             ai(lane, 41u + i * 6u),
                                                             ai(lane, 42u + i * 6u)),
                                                      PxVec3(ai(lane, 43u + i * 6u),
                                                             ai(lane, 44u + i * 6u),
                                                             ai(lane, 45u + i * 6u))),
                               lane);
            storeSpatialVector(L.mSolverSpatialImpulse,
                               UnAlignedSpatialVector(PxVec3(ai(lane, 90u + i * 6u),
                                                             ai(lane, 91u + i * 6u),
                                                             ai(lane, 92u + i * 6u)),
                                                      PxVec3(ai(lane, 93u + i * 6u),
                                                             ai(lane, 94u + i * 6u),
                                                             ai(lane, 95u + i * 6u))),
                               lane);
            for (uint32_t dd = 0; dd < dofs; ++dd)
            {
                physx::PxgArticulationBlockDofData& d = dfImg[cursor + dd];
                storeSpatialVector(d.mIsInvDW,
                                   UnAlignedSpatialVector(
                                       PxVec3(ai(lane, 150u + (cursor + dd) * 6u),
                                              ai(lane, 151u + (cursor + dd) * 6u),
                                              ai(lane, 152u + (cursor + dd) * 6u)),
                                       PxVec3(ai(lane, 153u + (cursor + dd) * 6u),
                                              ai(lane, 154u + (cursor + dd) * 6u),
                                              ai(lane, 155u + (cursor + dd) * 6u))),
                                   lane);
                storeSpatialVector(d.mWorldMotionMatrix,
                                   UnAlignedSpatialVector(
                                       PxVec3(ai(lane, 250u + (cursor + dd) * 6u),
                                              ai(lane, 251u + (cursor + dd) * 6u),
                                              ai(lane, 252u + (cursor + dd) * 6u)),
                                       PxVec3(ai(lane, 253u + (cursor + dd) * 6u),
                                              ai(lane, 254u + (cursor + dd) * 6u),
                                              ai(lane, 255u + (cursor + dd) * 6u))),
                                   lane);
                d.mDeferredQstZ[lane] = ai(lane, 350u + cursor + dd);
            }
            cursor += dofs;
        }
        PXG_CHECK_MSG(cursor <= AI_MAX_DOFS, "lane %u needs %u dofs", lane, cursor);
        PXG_CHECK_MSG(nl != lane, "lane %u's link count equals its index", lane);
        (void)aiTotalDofs;

        if (lane == AI_REST_LANE)
        {
            PXG_CHECK_MSG(nl == 2u && lkImg[1].mDofs[lane] == 1u && lkImg[1].mParents[lane] == 0u &&
                              abImg[0].mLinkWithDeferredImpulse[lane] == 1u,
                          "resting lane: not a walked two-link, one-dof articulation");
            for (uint32_t s = 0; s < nbSlabs; ++s)
                PXG_CHECK_MSG(dirty[(size_t(s) * AI_LANES + lane) * 2 + 0] == AI_NONE &&
                                  dirty[(size_t(s) * AI_LANES + lane) * 2 + 1] == AI_NONE,
                              "resting lane: slab %u carries an impulse", s);
            const UnAlignedSpatialVector negZero(PxVec3(-0.0f, -0.0f, -0.0f),
                                                 PxVec3(-0.0f, -0.0f, -0.0f));
            storeSpatialVector(lkImg[0].mScratchImpulse, negZero, lane);
            storeSpatialVector(lkImg[1].mScratchImpulse, negZero, lane);
            // the accumulators, freshly zeroed (forwardDynamic2.cu:81/91/97)
            storeSpatialVector(lkImg[1].mSolverSpatialImpulse, UnAlignedSpatialVector::Zero(), lane);
            storeSpatialVector(abImg[0].mRootDeferredZ, UnAlignedSpatialVector::Zero(), lane);
            physx::PxgArticulationBlockDofData& d = dfImg[lkImg[1].mJointOffset[lane]];
            d.mDeferredQstZ[lane] = 0.0f;
            storeSpatialVector(d.mWorldMotionMatrix,
                               UnAlignedSpatialVector(PxVec3(0.0f, 0.0f, -1.0f), PxVec3(0.0f, 0.0f, 0.0f)),
                               lane);
            storeSpatialVector(d.mIsInvDW,
                               UnAlignedSpatialVector(PxVec3(0.75f, -0.5f, 0.25f),
                                                      PxVec3(0.125f, -0.375f, 0.625f)),
                               lane);
        }
    }

    std::vector<uint32_t> wantDirty = dirty;
    std::vector<float> wantImp = imp;
    std::vector<physx::PxgArticulationBlockData> wantAb = abImg;
    std::vector<physx::PxgArticulationBlockLinkData> wantLk = lkImg;
    std::vector<physx::PxgArticulationBlockDofData> wantDf = dfImg;
    pxgref::gAiPosZeroDotHits = 0;
    for (uint32_t lane = 0; lane < AI_LANES; ++lane)
        aiMirror(wantDirty.data(), wantImp.data(), wantAb.data(), wantLk.data(), wantDf.data(),
                 lane, AI_MAX_LINKS, AI_LANES, nbSlabs, lane, alignedHits, unalignedHits);
    const uint32_t posZeroDotHits = pxgref::gAiPosZeroDotHits;

    // P9.86c: the equivalence witness (see the header).  twin(abSeed) runs the
    // mirror over the INPUTS with the `-dot` twin on or off and returns the
    // images; the resting lane's mRootDeferredZ seed can be replaced first.
    {
        struct Img
        {
            std::vector<physx::PxgArticulationBlockData> ab;
            std::vector<physx::PxgArticulationBlockLinkData> lk;
            std::vector<physx::PxgArticulationBlockDofData> df;
        };
        auto run = [&](bool negDot, bool invalidNegZeroRootSeed) {
            std::vector<uint32_t> tDirty = dirty;
            std::vector<float> tImp = imp;
            Img r{ abImg, lkImg, dfImg };
            if (invalidNegZeroRootSeed)
                storeSpatialVector(r.ab[0].mRootDeferredZ,
                                   UnAlignedSpatialVector(PxVec3(-0.0f, -0.0f, -0.0f),
                                                          PxVec3(-0.0f, -0.0f, -0.0f)),
                                   AI_REST_LANE);
            uint32_t h0 = 0, h1 = 0;
            pxgref::gAiNullJfNegDot = negDot;
            for (uint32_t lane = 0; lane < AI_LANES; ++lane)
                aiMirror(tDirty.data(), tImp.data(), r.ab.data(), r.lk.data(), r.df.data(), lane,
                         AI_MAX_LINKS, AI_LANES, nbSlabs, lane, h0, h1);
            pxgref::gAiNullJfNegDot = false;
            return r;
        };
        auto diff = [](const Img& a, const Img& b) {
            size_t n = 0;
            auto words = [&](const void* x, const void* y, size_t bytes) {
                const uint32_t* p = reinterpret_cast<const uint32_t*>(x);
                const uint32_t* q = reinterpret_cast<const uint32_t*>(y);
                for (size_t k = 0; k < bytes / 4; ++k) n += (p[k] != q[k]) ? 1u : 0u;
            };
            words(a.ab.data(), b.ab.data(), a.ab.size() * sizeof(a.ab[0]));
            words(a.lk.data(), b.lk.data(), a.lk.size() * sizeof(a.lk[0]));
            words(a.df.data(), b.df.data(), a.df.size() * sizeof(a.df[0]));
            return n;
        };
        const Img want{ wantAb, wantLk, wantDf };
        const size_t validSplit = diff(want, run(true, false));
        const Img ctlRef = run(false, true);
        const Img ctlTwin = run(true, true);
        const size_t invalidSplit = diff(ctlRef, ctlTwin);
        const UnAlignedSpatialVector a = loadSpatialVector(ctlRef.ab[0].mRootDeferredZ, AI_REST_LANE);
        const UnAlignedSpatialVector b = loadSpatialVector(ctlTwin.ab[0].mRootDeferredZ, AI_REST_LANE);
        auto bits = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; };
        std::printf("[ info ] ai slabs=%u: +0 dual products %u; -dot twin splits %zu words on valid "
                    "data, %zu with an INVALID -0 mRootDeferredZ seed (top %08x %08x %08x vs "
                    "%08x %08x %08x)\n",
                    nbSlabs, posZeroDotHits, validSplit, invalidSplit, bits(a.top.x), bits(a.top.y),
                    bits(a.top.z), bits(b.top.x), bits(b.top.y), bits(b.top.z));
        PXG_CHECK_MSG(posZeroDotHits > 0, "no +0 dual product reached the Ex walk's hop");
        PXG_CHECK_MSG(validSplit == 0, "the `-dot` twin changes %zu words on VALID data -- the "
                      "Ex-walk equivalence (P9.86c r2) no longer holds", validSplit);
        PXG_CHECK_MSG(invalidSplit == 4 && bits(a.top.x) == 0x00000000u &&
                          bits(a.top.y) == 0x80000000u && bits(a.top.z) == 0x00000000u &&
                          bits(b.top.x) == 0x80000000u && bits(b.top.y) == 0x00000000u &&
                          bits(b.top.z) == 0x80000000u,
                      "resting lane no longer carries the zero sign to the last absorbing add "
                      "(%zu control words)", invalidSplit);
    }

    PxgDevicePtr dDirty = 0, dImp = 0, dAb = 0, dLk = 0, dDf = 0;
    shim.memAlloc(&dDirty, dirty.size() * sizeof(uint32_t));
    shim.memAlloc(&dImp, imp.size() * sizeof(float));
    shim.memAlloc(&dAb, abImg.size() * sizeof(physx::PxgArticulationBlockData));
    shim.memAlloc(&dLk, lkImg.size() * sizeof(physx::PxgArticulationBlockLinkData));
    shim.memAlloc(&dDf, dfImg.size() * sizeof(physx::PxgArticulationBlockDofData));
    if (!dDirty || !dImp || !dAb || !dLk || !dDf)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    shim.memcpyHtoD(dDirty, dirty.data(), dirty.size() * sizeof(uint32_t));
    shim.memcpyHtoD(dImp, imp.data(), imp.size() * sizeof(float));
    shim.memcpyHtoD(dAb, abImg.data(), abImg.size() * sizeof(physx::PxgArticulationBlockData));
    shim.memcpyHtoD(dLk, lkImg.data(), lkImg.size() * sizeof(physx::PxgArticulationBlockLinkData));
    shim.memcpyHtoD(dDf, dfImg.data(), dfImg.size() * sizeof(physx::PxgArticulationBlockDofData));

    struct Params
    {
        uint64_t dirty, imp, ab, lk, df;
        uint32_t maxLinks, nbArticulations, nbSlabs, nbLanes;
    } prm = { uint64_t(dDirty), uint64_t(dImp), uint64_t(dAb), uint64_t(dLk), uint64_t(dDf),
              AI_MAX_LINKS, AI_LANES, nbSlabs, AI_LANES };
    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch artiAvgImpulse");
    shim.streamSynchronize();

    std::vector<uint32_t> gotDirty(dirty.size());
    std::vector<float> gotImp(imp.size());
    std::vector<physx::PxgArticulationBlockData> gotAb(abImg.size());
    std::vector<physx::PxgArticulationBlockLinkData> gotLk(lkImg.size());
    std::vector<physx::PxgArticulationBlockDofData> gotDf(dfImg.size());
    shim.memcpyDtoH(gotDirty.data(), dDirty, gotDirty.size() * sizeof(uint32_t));
    shim.memcpyDtoH(gotImp.data(), dImp, gotImp.size() * sizeof(float));
    shim.memcpyDtoH(gotAb.data(), dAb, gotAb.size() * sizeof(physx::PxgArticulationBlockData));
    shim.memcpyDtoH(gotLk.data(), dLk, gotLk.size() * sizeof(physx::PxgArticulationBlockLinkData));
    shim.memcpyDtoH(gotDf.data(), dDf, gotDf.size() * sizeof(physx::PxgArticulationBlockDofData));

    // STRICT (P9.86c): a signed-zero swap is a mismatch, counted apart only
    // for the info line
    auto cmp = [&](const char* what, const void* wp, const void* gp, size_t bytes) {
        const uint32_t* w = reinterpret_cast<const uint32_t*>(wp);
        const uint32_t* g = reinterpret_cast<const uint32_t*>(gp);
        for (size_t k = 0; k < bytes / 4; ++k)
        {
            if (w[k] == g[k]) continue;
            if ((w[k] == 0u && g[k] == 0x80000000u) || (w[k] == 0x80000000u && g[k] == 0u))
                ++signedZeros;
            if (bad < 12)
                std::printf("MM ai slabs=%u %s word %zu  ref %08x  gpu %08x\n", nbSlabs, what, k,
                            w[k], g[k]);
            ++bad;
        }
    };
    cmp("dirty", wantDirty.data(), gotDirty.data(), wantDirty.size() * sizeof(uint32_t));
    cmp("impulses", wantImp.data(), gotImp.data(), wantImp.size() * sizeof(float));
    cmp("arti", wantAb.data(), gotAb.data(),
        wantAb.size() * sizeof(physx::PxgArticulationBlockData));
    cmp("link", wantLk.data(), gotLk.data(),
        wantLk.size() * sizeof(physx::PxgArticulationBlockLinkData));
    cmp("dof", wantDf.data(), gotDf.data(),
        wantDf.size() * sizeof(physx::PxgArticulationBlockDofData));

    shim.memFree(dDirty);
    shim.memFree(dImp);
    shim.memFree(dAb);
    shim.memFree(dLk);
    shim.memFree(dDf);
}

PXG_TEST(artiAvgImpulse)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiAvgImpulse.spv", "artiAvgImpulse", 56u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &artiAvgLoader);

    // coverage of the scale ladder, computed from the same predicate the fill
    // uses rather than from a second model of it
    uint32_t seenCount[5] = { 0, 0, 0, 0, 0 };
    for (uint32_t lane = 0; lane < AI_LANES; ++lane)
        seenCount[aiDirtySlabs(lane)] += 1u;
    std::printf("[ info ] ai: lanes with 0/1/2/3/4 dirty slabs: %u/%u/%u/%u/%u\n", seenCount[0],
                seenCount[1], seenCount[2], seenCount[3], seenCount[4]);
    for (uint32_t c = 0; c < 5; ++c)
        PXG_CHECK_MSG(seenCount[c] > 2, "only %u lanes have %u dirty slabs", seenCount[c], c);

    size_t bad = 0, signedZeros = 0;
    uint32_t alignedHits = 0, unalignedHits = 0;
    // nbSlabs > 1: the scale ladder is live
    aiRun(shim, 4u, bad, signedZeros, alignedHits, unalignedHits);
    // nbSlabs == 1: the whole scale computation is skipped and scale stays 1
    aiRun(shim, 1u, bad, signedZeros, alignedHits, unalignedHits);

    std::printf("[ info ] ai: impulse reads 16-aligned %u, 8-aligned %u, signed-zero mismatches %zu\n",
                alignedHits, unalignedHits, signedZeros);
    // The shader collapses the CUDA's alignment branch; the mirror keeps it.
    // The claim is only checked if BOTH sides of that branch are actually taken.
    PXG_CHECK_MSG(alignedHits > 5 && unalignedHits > 5,
                  "the impulse array does not exercise both alignments (%u / %u)", alignedHits,
                  unalignedHits);
    PXG_CHECK_MSG(bad == 0, "the averaged impulses do not match the engine bit for bit");
}

}  // namespace
