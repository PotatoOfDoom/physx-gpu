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

// testArtiAvgImpulse2.cpp -- P8.101a: averageLinkImpulsesAndPropagate2
// (articulationImpulseResponse.cuh:263-385), the PARTIAL per-partition
// propagate of artiPropagateImpulses2, against the pxgref::aiMirror2 host
// mirror (engine loads + the same aiPropagateImpulse hop + the ENGINE's
// loadSpatialMatrix / PxSpatialMatrix::operator* for the 6x6 tail).
//
// What the fixture is shaped to see:
//  * maxLinks 70 -> wordSize 2, chains of 60..70 links (every 7th re-parents
//    to i-3), so word-1 path bits are REAL on the longer lanes;
//  * commonNode rotation per lane: the root (0 -- the walk runs dry and the
//    tail reads the ROOT's scratch), a word-0 ancestor, a word-1 ancestor
//    (>= 64: the break leaves word 0's bits to be processed -- the engine's
//    break-exits-one-word spelling, counted and REQUIRED), and the deepest
//    link's parent (a one-hop walk);
//  * path bits exclude the root (the engine's own data never carries bit 0);
//  * dirtyFlags without eHAS_IMPULSES on every third lane: the slab phase
//    must be SKIPPED (pairs stay live, impulses stay unconsumed) while the
//    walk and the 6x6 tail still run;
//  * per-lane scale (never 0 -- the '2' variant has no internal-count
//    branch);
//  * the common node's scratch impulse must come back NOT zeroed (the
//    load-bearing engine comment), and mSolverSpatialImpulse books += Z
//    with NO internal-constraint share;
//  * everything poisoned 0xAB and compared bit for bit -- STRICTLY, the sign
//    of zero included (P9.86c: a 0 <-> 0x80000000 swap is a mismatch);
//  * P9.86c, lane A2_REST_LANE: a RESTING pair on the dirty path.  The
//    deepest link and its parent (the common node, a one-hop walk) both hold
//    six -0 scratch words -- what rigidDeltaAccum.cu:384's `-impulse` store
//    leaves for a zero-impulse contact (with eHAS_IMPULSES, :365) -- and the
//    deepest link has ONE dof, a revolute joint about world -Z in an
//    axis-aligned frame (motion (0,0,-1 | 0,0,0)).  Its dual product against
//    the all -0 Z is exactly +0, where propagateImpulseW's NULL-jointForce
//    `0.0f - dot` (articulationDynamic.cuh:143-144) gives stZ = +0 and the
//    preloaded `-dot` gives -0.  isInvD of mixed sign carries that into
//    temp = Z + isInvD * stZ, and the common node's scratch -- which the
//    kernel leaves UN-zeroed -- stores (+0,-0,+0 | +0,+0,+0) against the
//    `-dot` spelling's (-0,+0,-0 | -0,+0,+0).  The host twin
//    pxgref::gAiNullJfNegDot must reproduce exactly that split.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"
#include "pxgArtiRefHelpers.h"

#include "foundation/PxVec3.h"
#include "CmSpatialVector.h"
#include "DyFeatherstoneArticulation.h"
#include "PxSpatialMatrix.h"
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
using physx::loadSpatialMatrix;
using physx::storeSpatialMatrix;

static std::vector<uint8_t> ai2Loader(const char* file) { return loadShaderBlob(file); }

const uint32_t A2_LANES = 22;
const uint32_t A2_REST_LANE = 21;   // P9.86c: the resting pair (see the header)
const uint32_t A2_MAX_LINKS = 70;    // -> wordSize 2
const uint32_t A2_WORDSIZE = 2;
const uint32_t A2_MAX_DOFS = 100;
const uint32_t A2_NB_SLABS = 3;
const uint32_t A2_NONE = 0xFFFFFFFFu;

static uint32_t a2NumLinks(uint32_t k) { return 60u + (k % 11u); }   // 60..70
static uint32_t a2Parent(uint32_t i)
{
    if (i == 0) return 0u;
    return (i >= 3u && (i % 7u) == 0u) ? (i - 3u) : (i - 1u);
}
static uint32_t a2Dofs(uint32_t k, uint32_t i)
{
    // the resting lane's deepest link is a 1-dof revolute joint
    if (k == A2_REST_LANE && i == a2NumLinks(k) - 1u) return 1u;
    return (k + i) % 3u;
}

static float a2(uint32_t i, uint32_t k)
{
    return (float((i * 53u + k * 29u) % 467u) + 1.0f) * 0.0078125f - 1.8f;
}

static void a2Path(uint32_t lane, uint64_t words[2], std::vector<uint32_t>& path)
{
    words[0] = 0; words[1] = 0;
    path.clear();
    uint32_t c = a2NumLinks(lane) - 1u;   // the deepest link
    while (c != 0u)
    {
        words[c / 64u] |= (uint64_t(1) << (c % 64u));
        path.push_back(c);
        c = a2Parent(c);
    }
    // the root's bit stays clear -- the engine's per-partition paths never
    // carry it (cf. the assert in artiPropagateVelocityInternal)
}

static uint32_t a2CommonNode(uint32_t lane, const std::vector<uint32_t>& path)
{
    if (lane == A2_REST_LANE)                      // the resting pair: one hop
        return path.size() > 1 ? path[1] : 0u;
    const uint32_t mode = lane % 4u;
    if (mode == 0u) return 0u;                     // root: the walk runs dry
    if (mode == 3u) return path.size() > 1 ? path[1] : 0u;   // one-hop walk
    if (mode == 2u)
    {
        for (size_t i = path.size(); i-- > 0;)     // smallest >= 64 if any
            if (path[i] >= 64u) return path[i];
    }
    uint32_t best = path.back();                   // mode 1 (and 2 fallback):
    for (uint32_t p : path)                        // the ancestor nearest 20
        if ((p > 20u ? p - 20u : 20u - p) < (best > 20u ? best - 20u : 20u - best)) best = p;
    return best;
}

} // namespace

PXG_TEST(artiAvgImpulse2)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiAvgImpulse.spv", "aiAvg2Test", 80u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &ai2Loader);

    std::vector<physx::PxgArticulationBlockData> abImg(1);
    std::vector<physx::PxgArticulationBlockLinkData> lkImg(A2_MAX_LINKS);
    std::vector<physx::PxgArticulationBlockDofData> dfImg(A2_MAX_DOFS);
    std::memset(abImg.data(), 0xAB, abImg.size() * sizeof(abImg[0]));
    std::memset(lkImg.data(), 0xAB, lkImg.size() * sizeof(lkImg[0]));
    std::memset(dfImg.data(), 0xAB, dfImg.size() * sizeof(dfImg[0]));

    std::vector<uint32_t> dirty(size_t(A2_NB_SLABS) * A2_LANES * 2, A2_NONE);
    std::vector<float> imp(size_t(A2_NB_SLABS) * A2_MAX_LINKS * A2_LANES * 6, 0.f);
    for (size_t k = 0; k < imp.size(); ++k)
        imp[k] = a2(uint32_t(k % 977u), uint32_t(k / 977u)) * 1.5f;

    std::vector<uint64_t> pathWords(size_t(A2_WORDSIZE) * 32, 0xABABABABABABABABull);
    std::vector<uint32_t> io(size_t(A2_LANES) * 3, 0xABABABABu);

    for (uint32_t lane = 0; lane < A2_LANES; ++lane)
    {
        const uint32_t nl = a2NumLinks(lane);

        uint64_t words[2];
        std::vector<uint32_t> path;
        a2Path(lane, words, path);
        pathWords[0 * 32 + lane] = words[0];
        pathWords[1 * 32 + lane] = words[1];

        const uint32_t commonNode = a2CommonNode(lane, path);
        const float scale = 0.25f + 0.05f * float(lane);
        // the resting lane carries eHAS_IMPULSES, as rigidDeltaAccum.cu:365 sets it
        // alongside the `-impulse` scratch store
        const uint32_t dirtyFlags = (lane == A2_REST_LANE) ? 2u
                                  : (lane % 3u == 0u) ? 1u : ((lane % 3u == 1u) ? 3u : 2u);
        io[lane * 3 + 0] = physx::PxU32(*reinterpret_cast<const uint32_t*>(&scale));
        io[lane * 3 + 1] = commonNode;
        io[lane * 3 + 2] = dirtyFlags;

        // slab pairs: only-x / only-y / both / neither, indices 1..nl-1
        for (uint32_t s = 0; s < A2_NB_SLABS; ++s)
        {
            const size_t di = size_t(s) * A2_LANES + lane;
            const uint32_t mode = (lane + s) % 4u;
            const uint32_t a = 1u + ((lane * 5u + s * 11u) % (nl - 1u));
            const uint32_t b = 1u + ((lane * 7u + s * 13u + 1u) % (nl - 1u));
            if (mode == 0u || mode == 2u) dirty[di * 2 + 0] = a;
            if (mode == 1u || mode == 2u) dirty[di * 2 + 1] = b;
        }

        abImg[0].mLinkWithDeferredImpulse[lane] = 0xABABABABu;   // overwritten by the tail

        uint32_t cursor = 0;
        for (uint32_t i = 0; i < nl; ++i)
        {
            physx::PxgArticulationBlockLinkData& L = lkImg[i];
            const uint32_t dofs = (i == 0) ? 0u : a2Dofs(lane, i);
            L.mDofs[lane] = uint8_t(dofs);
            L.mParents[lane] = a2Parent(i);
            L.mJointOffset[lane] = cursor;
            L.mRw_x[lane] = a2(lane, 10u + i * 3u);
            L.mRw_y[lane] = a2(lane, 11u + i * 3u);
            L.mRw_z[lane] = a2(lane, 12u + i * 3u);
            storeSpatialVector(L.mScratchImpulse,
                               UnAlignedSpatialVector(
                                   PxVec3(a2(lane, 300u + i * 6u), a2(lane, 301u + i * 6u), a2(lane, 302u + i * 6u)),
                                   PxVec3(a2(lane, 303u + i * 6u), a2(lane, 304u + i * 6u), a2(lane, 305u + i * 6u))),
                               lane);
            storeSpatialVector(L.mSolverSpatialImpulse,
                               UnAlignedSpatialVector(
                                   PxVec3(a2(lane, 800u + i * 6u), a2(lane, 801u + i * 6u), a2(lane, 802u + i * 6u)),
                                   PxVec3(a2(lane, 803u + i * 6u), a2(lane, 804u + i * 6u), a2(lane, 805u + i * 6u))),
                               lane);
            physx::PxSpatialMatrix R;
            for (uint32_t c = 0; c < 6; ++c)
                for (uint32_t r = 0; r < 6; ++r)
                    R.column[c][r] = 0.2f * a2(lane, 1300u + i * 40u + c * 6u + r);
            storeSpatialMatrix(L.mSpatialResponseMatrix, lane, R);

            for (uint32_t dd = 0; dd < dofs; ++dd)
            {
                physx::PxgArticulationBlockDofData& d = dfImg[cursor + dd];
                storeSpatialVector(d.mIsInvDW,
                                   UnAlignedSpatialVector(
                                       PxVec3(a2(lane, 4000u + (cursor + dd) * 6u), a2(lane, 4001u + (cursor + dd) * 6u), a2(lane, 4002u + (cursor + dd) * 6u)),
                                       PxVec3(a2(lane, 4003u + (cursor + dd) * 6u), a2(lane, 4004u + (cursor + dd) * 6u), a2(lane, 4005u + (cursor + dd) * 6u))),
                                   lane);
                storeSpatialVector(d.mWorldMotionMatrix,
                                   UnAlignedSpatialVector(
                                       PxVec3(a2(lane, 5000u + (cursor + dd) * 6u), a2(lane, 5001u + (cursor + dd) * 6u), a2(lane, 5002u + (cursor + dd) * 6u)),
                                       PxVec3(a2(lane, 5003u + (cursor + dd) * 6u), a2(lane, 5004u + (cursor + dd) * 6u), a2(lane, 5005u + (cursor + dd) * 6u))),
                                   lane);
                d.mDeferredQstZ[lane] = a2(lane, 6000u + cursor + dd);
            }
            cursor += dofs;
        }
        PXG_CHECK_MSG(cursor <= A2_MAX_DOFS, "lane %u needs %u dofs", lane, cursor);

        if (lane == A2_REST_LANE)
        {
            const uint32_t deep = nl - 1u;
            const uint32_t cn = a2Parent(deep);
            PXG_CHECK_MSG(cn == commonNode && path.size() > 1 && path[0] == deep,
                          "resting lane: the common node %u is not the deepest link's parent %u",
                          commonNode, cn);
            PXG_CHECK_MSG(lkImg[deep].mDofs[lane] == 1u, "resting lane: the deepest link lost its dof");
            // the averaging must not touch the -0 words
            for (uint32_t s = 0; s < A2_NB_SLABS; ++s)
                for (uint32_t h = 0; h < 2; ++h)
                {
                    const uint32_t d = dirty[(size_t(s) * A2_LANES + lane) * 2 + h];
                    PXG_CHECK_MSG(d != deep && d != cn,
                                  "resting lane: slab %u names link %u of the resting pair", s, d);
                }
            // zero-impulse contacts on both links, as rigidDeltaAccum.cu:384 stores them
            const UnAlignedSpatialVector negZero(PxVec3(-0.0f, -0.0f, -0.0f),
                                                 PxVec3(-0.0f, -0.0f, -0.0f));
            storeSpatialVector(lkImg[deep].mScratchImpulse, negZero, lane);
            storeSpatialVector(lkImg[cn].mScratchImpulse, negZero, lane);
            // a revolute joint about world -Z; isInvD of mixed sign
            physx::PxgArticulationBlockDofData& d = dfImg[lkImg[deep].mJointOffset[lane]];
            storeSpatialVector(d.mWorldMotionMatrix,
                               UnAlignedSpatialVector(PxVec3(0.0f, 0.0f, -1.0f), PxVec3(0.0f, 0.0f, 0.0f)),
                               lane);
            storeSpatialVector(d.mIsInvDW,
                               UnAlignedSpatialVector(PxVec3(0.75f, -0.5f, 0.25f),
                                                      PxVec3(0.125f, -0.375f, 0.625f)),
                               lane);
            PXG_CHECK_MSG(d.mDeferredQstZ[lane] != 0.0f, "resting lane: mDeferredQstZ seed is zero");
        }
    }

    // mirror
    std::vector<uint32_t> wantDirty = dirty;
    std::vector<float> wantImp = imp;
    std::vector<physx::PxgArticulationBlockData> wantAb = abImg;
    std::vector<physx::PxgArticulationBlockLinkData> wantLk = lkImg;
    std::vector<physx::PxgArticulationBlockDofData> wantDf = dfImg;
    uint32_t slabHits = 0, walkHits = 0, lowerWordHits = 0;
    pxgref::gAiPosZeroDotHits = 0;
    for (uint32_t lane = 0; lane < A2_LANES; ++lane)
    {
        const float scale = *reinterpret_cast<const float*>(&io[lane * 3 + 0]);
        pxgref::aiMirror2(wantDirty.data(), wantImp.data(), wantAb.data(), wantLk.data(),
                          wantDf.data(), lane, A2_MAX_LINKS, A2_LANES, A2_NB_SLABS,
                          scale, pathWords.data(), A2_WORDSIZE,
                          io[lane * 3 + 1], io[lane * 3 + 2], lane,
                          slabHits, walkHits, lowerWordHits);
    }
    const uint32_t posZeroDotHits = pxgref::gAiPosZeroDotHits;
    std::printf("[ info ] avg2: slabHits %u, walkHits %u, lowerWordHits %u, +0 dual products %u\n",
                slabHits, walkHits, lowerWordHits, posZeroDotHits);
    PXG_CHECK_MSG(slabHits > 0 && walkHits > 0, "coverage hole");
    PXG_CHECK_MSG(lowerWordHits > 0, "the break-exits-one-word case never ran");
    PXG_CHECK_MSG(posZeroDotHits > 0, "no +0 dual product reached the null-jointForce hop");

    // P9.86c: the `-dot` twin (the port's pre-P9.86 adPropagateImpulseW call,
    // articulationImpulseResponse.cuh:359's `0.0f - dot` reverted) run over the
    // same INPUTS must split the resting lane's common-node scratch exactly as
    // derived in the header -- or the lane no longer separates the spellings.
    {
        std::vector<uint32_t> twDirty = dirty;
        std::vector<float> twImp = imp;
        std::vector<physx::PxgArticulationBlockData> twAb = abImg;
        std::vector<physx::PxgArticulationBlockLinkData> twLk = lkImg;
        std::vector<physx::PxgArticulationBlockDofData> twDf = dfImg;
        uint32_t t0 = 0, t1 = 0, t2 = 0;
        pxgref::gAiNullJfNegDot = true;
        for (uint32_t lane = 0; lane < A2_LANES; ++lane)
        {
            const float scale = *reinterpret_cast<const float*>(&io[lane * 3 + 0]);
            pxgref::aiMirror2(twDirty.data(), twImp.data(), twAb.data(), twLk.data(), twDf.data(),
                              lane, A2_MAX_LINKS, A2_LANES, A2_NB_SLABS, scale, pathWords.data(),
                              A2_WORDSIZE, io[lane * 3 + 1], io[lane * 3 + 2], lane, t0, t1, t2);
        }
        pxgref::gAiNullJfNegDot = false;

        size_t twinWords = 0;
        auto diffWords = [&](const void* a, const void* b, size_t bytes) {
            const uint32_t* x = reinterpret_cast<const uint32_t*>(a);
            const uint32_t* y = reinterpret_cast<const uint32_t*>(b);
            for (size_t k = 0; k < bytes / 4; ++k)
                twinWords += (x[k] != y[k]) ? 1u : 0u;
        };
        diffWords(wantAb.data(), twAb.data(), wantAb.size() * sizeof(wantAb[0]));
        diffWords(wantLk.data(), twLk.data(), wantLk.size() * sizeof(wantLk[0]));
        diffWords(wantDf.data(), twDf.data(), wantDf.size() * sizeof(wantDf[0]));

        const uint32_t cn = io[A2_REST_LANE * 3 + 1];
        const UnAlignedSpatialVector w = loadSpatialVector(wantLk[cn].mScratchImpulse, A2_REST_LANE);
        const UnAlignedSpatialVector t = loadSpatialVector(twLk[cn].mScratchImpulse, A2_REST_LANE);
        auto bits = [](float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; };
        std::printf("[ info ] avg2: -dot twin changes %zu words; resting common node %u scratch "
                    "top %08x %08x %08x bottom.x %08x (twin %08x %08x %08x / %08x)\n",
                    twinWords, cn, bits(w.top.x), bits(w.top.y), bits(w.top.z), bits(w.bottom.x),
                    bits(t.top.x), bits(t.top.y), bits(t.top.z), bits(t.bottom.x));
        PXG_CHECK_MSG(twinWords == 4, "the -dot twin changes %zu words, expected the 4 of the "
                      "resting common node's scratch", twinWords);
        PXG_CHECK_MSG(bits(w.top.x) == 0x00000000u && bits(w.top.y) == 0x80000000u &&
                      bits(w.top.z) == 0x00000000u && bits(w.bottom.x) == 0x00000000u &&
                      bits(t.top.x) == 0x80000000u && bits(t.top.y) == 0x00000000u &&
                      bits(t.top.z) == 0x80000000u && bits(t.bottom.x) == 0x80000000u,
                      "resting lane no longer separates `0 - dot` from `-dot`");
    }

    PxgDevicePtr dDirty = 0, dImp = 0, dAb = 0, dLk = 0, dDf = 0, dPw = 0, dIo = 0;
    shim.memAlloc(&dDirty, dirty.size() * 4);
    shim.memAlloc(&dImp, imp.size() * 4);
    shim.memAlloc(&dAb, abImg.size() * sizeof(abImg[0]));
    shim.memAlloc(&dLk, lkImg.size() * sizeof(lkImg[0]));
    shim.memAlloc(&dDf, dfImg.size() * sizeof(dfImg[0]));
    shim.memAlloc(&dPw, pathWords.size() * 8);
    shim.memAlloc(&dIo, io.size() * 4);
    if (!dDirty || !dImp || !dAb || !dLk || !dDf || !dPw || !dIo)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    shim.memcpyHtoD(dDirty, dirty.data(), dirty.size() * 4);
    shim.memcpyHtoD(dImp, imp.data(), imp.size() * 4);
    shim.memcpyHtoD(dAb, abImg.data(), abImg.size() * sizeof(abImg[0]));
    shim.memcpyHtoD(dLk, lkImg.data(), lkImg.size() * sizeof(lkImg[0]));
    shim.memcpyHtoD(dDf, dfImg.data(), dfImg.size() * sizeof(dfImg[0]));
    shim.memcpyHtoD(dPw, pathWords.data(), pathWords.size() * 8);
    shim.memcpyHtoD(dIo, io.data(), io.size() * 4);

    struct Params
    {
        uint64_t dirty, imp, ab, lk, df, pw, io;
        uint32_t maxLinks, nbArticulations, nbSlabs, wordSize, nbLanes, pad0;
    } prm = { uint64_t(dDirty), uint64_t(dImp), uint64_t(dAb), uint64_t(dLk), uint64_t(dDf),
              uint64_t(dPw), uint64_t(dIo),
              A2_MAX_LINKS, A2_LANES, A2_NB_SLABS, A2_WORDSIZE, A2_LANES, 0u };
    static_assert(sizeof(Params) == 80, "params blob");
    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch aiAvg2Test");
    shim.streamSynchronize();

    // STRICT (P9.86c): a signed-zero swap is a mismatch, counted apart only
    // for the info line
    size_t bad = 0, signedZeros = 0;
    auto cmp = [&](const char* what, const void* wp, PxgDevicePtr dp, size_t bytes) {
        std::vector<uint8_t> g(bytes);
        shim.memcpyDtoH(g.data(), dp, bytes);
        const uint32_t* w = reinterpret_cast<const uint32_t*>(wp);
        const uint32_t* gg = reinterpret_cast<const uint32_t*>(g.data());
        for (size_t k = 0; k < bytes / 4; ++k)
        {
            if (w[k] == gg[k]) continue;
            if ((w[k] == 0u && gg[k] == 0x80000000u) || (w[k] == 0x80000000u && gg[k] == 0u))
                ++signedZeros;
            if (bad < 12)
                std::printf("MM avg2 %s word %zu  ref %08x  gpu %08x\n", what, k, w[k], gg[k]);
            ++bad;
        }
    };
    cmp("dirty", wantDirty.data(), dDirty, wantDirty.size() * 4);
    cmp("impulses", wantImp.data(), dImp, wantImp.size() * 4);
    cmp("arti", wantAb.data(), dAb, wantAb.size() * sizeof(wantAb[0]));
    cmp("link", wantLk.data(), dLk, wantLk.size() * sizeof(wantLk[0]));
    cmp("dof", wantDf.data(), dDf, wantDf.size() * sizeof(wantDf[0]));
    cmp("pathWords", pathWords.data(), dPw, pathWords.size() * 8);
    cmp("io", io.data(), dIo, io.size() * 4);

    std::printf("[ info ] avg2: %zu signed-zero mismatches\n", signedZeros);
    PXG_CHECK_MSG(bad == 0, "%zu mismatches", bad);

    shim.memFree(dDirty);
    shim.memFree(dImp);
    shim.memFree(dAb);
    shim.memFree(dLk);
    shim.memFree(dDf);
    shim.memFree(dPw);
    shim.memFree(dIo);
}
