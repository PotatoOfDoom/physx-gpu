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

// testArtiSolveSelfLaunch.cpp -- P8.99c: the assembled kernel
// artiPropagateRigidImpulsesAndSolveSelfConstraintsTGS1T (manifest 284)
// against a FULL host mirror over device-resident descriptors.
//
// Mirror pieces, every one already swept on its own: countActiveSlabs /
// resetSlabCount (solverBlock.cuh:84/:100, transcribed below -- eight lines),
// pxgref::aiMirrorEx (the averageLinkImpulsesAndPropagate mirror with the
// P8.99c scale parameter), and pxgtest::refSolveSelfLane (pxgArtiSelfRef.h).
//
// Wrapper surface under test (the P8.99b module carries the solve itself):
//  * three warps of 32 with the third PARTIAL (68 articulations) -- the
//    thread guard, and per-warp slices advancing by blockIdx.x directly;
//  * articulationReferenceCountOffset = islandContextPool[0].mBodyCount +
//    mBodyStartIndex (5+9=14 here, deliberately nonzero), row stride
//    numTotalBodies = 14 + 68;
//  * nbSlabs = 34 so the reference-count rows span TWO words;
//  * mStateDirty modes a%4: countActiveSlabs+scale+aiAvgEx only under
//    eHAS_IMPULSES; resetSlabCount ALWAYS; eVEL_DIRTY stored always;
//  * per-warp scalar batch windows (3@1 / 2@4 / 1@6) in ONE global batch
//    array; poison entries 0 and 7 must stay 0xAB;
//  * minPen from the velocityIteration bool; accumulateError from the
//    SIGNED mCounter at CONTACT_ERROR_ACC+4 (round B: -1, 1D residuals must
//    stay poisoned);
//  * doFriction passed both ways and IGNORED (TGS path).

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "foundation/PxQuat.h"
#include "foundation/PxVec3.h"
#include "CmSpatialVector.h"
#include "DyFeatherstoneArticulationUtils.h"
#include "DyFeatherstoneArticulation.h"
#include "PxgArticulation.h"
#include "PxgIslandContext.h"
#include "pxgArtiRefHelpers.h"
#include "pxgArtiSolveRef.h"
#include "pxgArtiExtSolveRef.h"
#include "pxgArtiSelfRef.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

namespace
{
using namespace pxgtest;
using namespace physx;
using Cm::UnAlignedSpatialVector;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

static std::vector<uint8_t> slLoader(const char* file) { return loadShaderBlob(file); }

const uint32_t SL_NB = 68, SL_WARPS = 3;
const uint32_t SL_MAX_LINKS = 65;   // -> wordSize 2
const uint32_t SL_WORDSIZE = 2;
const uint32_t SL_MAX_DOFS = 137;
const uint32_t SL_ROOT_DOF_SLICE = 133;
const uint32_t SL_NB_SLABS = 34;    // -> TWO reference-count words
const uint32_t SL_BODY_START = 5, SL_BODY_COUNT = 9;   // refCountOffset = 14
const uint32_t SL_REF_OFFSET = SL_BODY_START + SL_BODY_COUNT;
const uint32_t SL_TOTAL_BODIES = SL_REF_OFFSET + SL_NB;
const uint32_t SL_NUM32 = (SL_NB_SLABS + 31) / 32;
const uint32_t SL_NB_BATCHES = 8;   // 0 and 7 poison
const uint32_t SL_NB_HDRS = 8;
const uint32_t SL_NB_CPTS = 28, SL_NB_FPTS = 40, SL_NB_ROWS = 34, SL_NB_RESP = 60;
const float SL_ELAPSED = 0.017f;

// offsets into the 360-byte PxgArticulationCoreDesc, each pinned by
// static_assert in testDmaArtiResidual.cpp against the engine header
const uint32_t D_NB_ARTICULATIONS = 44, D_NB_SLABS = 80, D_IMPULSES = 72;
const uint32_t D_SLAB_HAS_CHANGES = 88, D_ARTI_BLOCKS = 128, D_LINK_BLOCKS = 136;
const uint32_t D_DOF_BLOCKS = 184, D_PATH_TO_ROOT = 176;
const uint32_t D_TEMP_PATH_TO_ROOT = 152, D_TEMP_SHARED = 160, D_TEMP_ROOT = 168;
const uint32_t D_MAX_LINKS = 248, D_MAX_DOFS = 252, D_ERROR_COUNTER = 348;
// PxgSolverCoreDesc::islandContextPool -- pinned by testSolverDescLayout
const uint32_t SD_ISLAND_CONTEXT_POOL = 48;
// PxgSolverSharedDesc<IterativeSolveDataTGS>::iterativeData.* -- ditto
const uint32_t SH_BATCH = 96, SH_JHDR = 104, SH_JROW = 112, SH_CHDR = 128;
const uint32_t SH_FHDR = 136, SH_CPTS = 144, SH_FPTS = 152, SH_ENC = 176, SH_RESP = 216;
const uint32_t SH_SIZE = 224, SD_SIZE = 320, DESC_SIZE = 360;

static float fv(uint32_t a, uint32_t l, uint32_t d, uint32_t salt)
{
    uint32_t h = a * 2654435761u + l * 40503u + d * 9973u + salt * 733u + 19u;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    return -1.0f + 2.0f * float(h & 0xFFFFFFu) / float(0x1000000);
}

static uint32_t slParent(uint32_t l, uint32_t lane)
{
    if (l == 0 || l == 1 || l == 2) return 0;
    if (l < 8) return l - 1;
    const uint32_t back = 1u + (((l + lane) % 4u == 0u) ? 2u : 0u);
    return l - back;
}
static uint32_t slDofs(uint32_t l, uint32_t lane) { return 1u + ((l + lane) % 3u); }

static void slPair(uint32_t k, uint32_t& a, uint32_t& b)
{
    switch (k % 5u)
    {
    case 0: a = 60; b = 64; break;   // one word-1 link
    case 1: a = 64; b = 33; break;
    case 2: a = 44; b = 44; break;   // equal
    case 3: a = 7;  b = 3;  break;   // ancestor: B walk empty
    default: a = 1; b = 2;  break;   // root-only common
    }
}

static bool slIsContact(uint32_t b) { return (b % 2u) == 1u; }

static uint32_t slMask(uint32_t b)
{
    switch (b)
    {
    case 1: return 0x00F0700Fu;
    case 2: return 0x1F0F37F1u;
    case 3: return 0x2AAAAAAAu;
    case 4: return 0x00000E75u;
    case 5: return 0xC300F00Fu;
    default: return 0x0000008Fu;     // batch 6 (warp 2): lanes 0-3 live, bit 7 guarded off
    }
}

struct SlArena
{
    std::vector<uint8_t> desc, solverDesc, sharedDesc;
    std::vector<PxgIslandContext> islands;                 // engine struct, 2 entries
    std::vector<PxgArticulationBlockData> ab;
    std::vector<PxgArticulationBlockLinkData> lk;
    std::vector<PxgArticulationBlockDofData> df;
    std::vector<uint64_t> bf, tmpA, tmpB, tmpC;
    std::vector<uint32_t> enc;                             // encodedReferenceCount rows
    std::vector<uint32_t> slab;                            // uint2 per (slab, arti)
    std::vector<float> imp;                                // 6 floats per delta slot
    std::vector<PxgBlockConstraintBatch> bt;
    std::vector<PxgTGSBlockSolverContactHeader> ch;
    std::vector<PxgTGSBlockSolverFrictionHeader> fh;
    std::vector<PxgTGSBlockSolverContactPoint> cp;
    std::vector<PxgTGSBlockSolverContactFriction> fp;
    std::vector<PxgTGSBlockSolverConstraint1DHeader> jh;
    std::vector<PxgTGSBlockSolverConstraint1DCon> jr;
    std::vector<PxgArticulationBlockResponse> rs;
};

static void slWarpWindow(uint32_t w, uint32_t& count, uint32_t& start)
{
    if (w == 0) { count = 3; start = 1; }
    else if (w == 1) { count = 2; start = 4; }
    else { count = 1; start = 6; }
}

static void slSeed(SlArena& B, uint32_t salt, int32_t errCounter)
{
    B.desc.assign(DESC_SIZE, 0xAB);
    B.solverDesc.assign(SD_SIZE, 0xAB);
    B.sharedDesc.assign(SH_SIZE, 0xAB);
    B.islands.resize(2);
    B.ab.resize(SL_WARPS);
    B.lk.resize(size_t(SL_WARPS) * SL_MAX_LINKS);
    B.df.resize(size_t(SL_WARPS) * SL_MAX_DOFS);
    B.bf.resize(size_t(SL_WARPS) * SL_MAX_LINKS * SL_WORDSIZE * 32);
    B.tmpA.resize(size_t(SL_WARPS) * SL_WORDSIZE * 32);
    B.tmpB.resize(size_t(SL_WARPS) * SL_WORDSIZE * 32);
    B.tmpC.resize(size_t(SL_WARPS) * SL_WORDSIZE * 32);
    B.enc.resize(size_t(SL_NUM32) * SL_TOTAL_BODIES);
    B.slab.resize(size_t(SL_NB_SLABS) * SL_NB * 2);
    B.imp.resize(size_t(SL_NB_SLABS) * SL_MAX_LINKS * SL_NB * 6);
    B.bt.resize(SL_NB_BATCHES);
    B.ch.resize(SL_NB_HDRS);
    B.fh.resize(SL_NB_HDRS);
    B.cp.resize(SL_NB_CPTS);
    B.fp.resize(SL_NB_FPTS);
    B.jh.resize(SL_NB_HDRS);
    B.jr.resize(SL_NB_ROWS);
    B.rs.resize(SL_NB_RESP);
    std::memset(B.islands.data(), 0xAB, B.islands.size() * sizeof(B.islands[0]));
    std::memset(B.ab.data(), 0xAB, B.ab.size() * sizeof(B.ab[0]));
    std::memset(B.lk.data(), 0xAB, B.lk.size() * sizeof(B.lk[0]));
    std::memset(B.df.data(), 0xAB, B.df.size() * sizeof(B.df[0]));
    std::memset(B.bf.data(), 0xAB, B.bf.size() * 8);
    std::memset(B.tmpA.data(), 0xAB, B.tmpA.size() * 8);
    std::memset(B.tmpB.data(), 0xAB, B.tmpB.size() * 8);
    std::memset(B.tmpC.data(), 0xAB, B.tmpC.size() * 8);
    std::memset(B.bt.data(), 0xAB, B.bt.size() * sizeof(B.bt[0]));
    std::memset(B.ch.data(), 0xAB, B.ch.size() * sizeof(B.ch[0]));
    std::memset(B.fh.data(), 0xAB, B.fh.size() * sizeof(B.fh[0]));
    std::memset(B.cp.data(), 0xAB, B.cp.size() * sizeof(B.cp[0]));
    std::memset(B.fp.data(), 0xAB, B.fp.size() * sizeof(B.fp[0]));
    std::memset(B.jh.data(), 0xAB, B.jh.size() * sizeof(B.jh[0]));
    std::memset(B.jr.data(), 0xAB, B.jr.size() * sizeof(B.jr[0]));
    std::memset(B.rs.data(), 0xAB, B.rs.size() * sizeof(B.rs[0]));

    // desc scalars (pointers are patched in at upload time)
    auto w32 = [&](std::vector<uint8_t>& v, uint32_t off, uint32_t x) {
        std::memcpy(&v[off], &x, 4);
    };
    w32(B.desc, D_NB_ARTICULATIONS, SL_NB);
    w32(B.desc, D_NB_SLABS, SL_NB_SLABS);
    w32(B.desc, D_MAX_LINKS, SL_MAX_LINKS);
    w32(B.desc, D_MAX_DOFS, SL_MAX_DOFS);
    w32(B.desc, D_ERROR_COUNTER, uint32_t(errCounter));

    B.islands[0].mBodyStartIndex = SL_BODY_START;
    B.islands[0].mBodyCount = SL_BODY_COUNT;

    // reference-count rows: varied popcounts, one articulation with ALL-ZERO
    // rows (max(1,.)), rows outside the articulation window poisoned-but-live
    for (uint32_t r = 0; r < SL_NUM32; ++r)
        for (uint32_t bdy = 0; bdy < SL_TOTAL_BODIES; ++bdy)
        {
            uint32_t h = (bdy * 2654435761u) ^ (r * 40503u) ^ (salt * 733u);
            h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
            B.enc[size_t(r) * SL_TOTAL_BODIES + bdy] = h & 0x1FFFFu;
        }
    for (uint32_t r = 0; r < SL_NUM32; ++r)   // articulation 10 (dirty&2 set): zero rows
        B.enc[size_t(r) * SL_TOTAL_BODIES + SL_REF_OFFSET + 10] = 0u;

    // slab dirty pairs: mostly NONE, a few live per articulation
    for (size_t i = 0; i < B.slab.size(); ++i) B.slab[i] = pxgref::AI_NONE;
    for (uint32_t a = 0; a < SL_NB; ++a)
        for (uint32_t s = 0; s < SL_NB_SLABS; ++s)
        {
            if ((a + s) % 9u == 3u)
                B.slab[(size_t(s) * SL_NB + a) * 2 + 0] = 1u + ((a + s) % (SL_MAX_LINKS - 1u));
            if ((a + s) % 11u == 5u)
                B.slab[(size_t(s) * SL_NB + a) * 2 + 1] = 1u + ((a * 3u + s) % (SL_MAX_LINKS - 1u));
        }

    for (size_t i = 0; i < B.imp.size(); ++i)
        B.imp[i] = 0.4f * fv(uint32_t(i & 0xFFFFu), uint32_t(i >> 16), 7, salt);

    for (uint32_t a = 0; a < SL_NB; ++a)
    {
        const uint32_t w = a / 32, lane = a % 32;
        PxgArticulationBlockData& AB = B.ab[w];
        PxgArticulationBlockLinkData* L = &B.lk[size_t(w) * SL_MAX_LINKS];
        PxgArticulationBlockDofData* D = &B.df[size_t(w) * SL_MAX_DOFS];

        uint32_t count, start;
        slWarpWindow(w, count, start);
        AB.mTotalSelfConstraintCount = count;
        AB.mSelfConstraintOffset = start;

        AB.mStateDirty[lane] = PxU8(a % 4u);
        AB.mLinkWithDeferredImpulse[lane] = (a % 5u == 0u) ? 0u : 1u + (a % (SL_MAX_LINKS - 1u));

        storeSpatialVector(AB.mRootDeferredZ,
                           UnAlignedSpatialVector(
                               PxVec3(fv(a, 90, 0, salt), fv(a, 90, 1, salt), fv(a, 90, 2, salt)),
                               PxVec3(fv(a, 90, 3, salt), fv(a, 90, 4, salt), fv(a, 90, 5, salt))),
                           lane);
        Dy::SpatialMatrix M;
        for (uint32_t c = 0; c < 3; ++c)
            for (uint32_t r = 0; r < 3; ++r)
            {
                M.topLeft[c][r] = 0.35f * fv(a, 91, c * 3 + r, salt);
                M.topRight[c][r] = 0.35f * fv(a, 92, c * 3 + r, salt);
                M.bottomLeft[c][r] = 0.35f * fv(a, 93, c * 3 + r, salt);
            }
        storeSpatialMatrix(AB.mInvSpatialArticulatedInertia, lane, M);

        uint32_t cursor = 0;
        for (uint32_t l = 0; l < SL_MAX_LINKS; ++l)
        {
            PxgArticulationBlockLinkData& LK = L[l];
            LK.mParents[lane] = slParent(l, lane);
            if (l == 0)
            {
                LK.mRw_x[lane] = 7.5f; LK.mRw_y[lane] = -3.25f; LK.mRw_z[lane] = 9.0f;
                LK.mJointOffset[lane] = SL_ROOT_DOF_SLICE;
                LK.mDofs[lane] = 3;
            }
            else
            {
                LK.mRw_x[lane] = 0.4f * fv(a, l, 10, salt);
                LK.mRw_y[lane] = 0.4f * fv(a, l, 11, salt);
                LK.mRw_z[lane] = 0.4f * fv(a, l, 12, salt);
                LK.mJointOffset[lane] = cursor;
                LK.mDofs[lane] = PxU8(slDofs(l, lane));
                cursor += slDofs(l, lane);
            }
            storeSpatialVector(LK.mMotionVelocity,
                               UnAlignedSpatialVector(
                                   PxVec3(fv(a, l, 13, salt), fv(a, l, 14, salt), fv(a, l, 15, salt)),
                                   PxVec3(fv(a, l, 16, salt), fv(a, l, 17, salt), fv(a, l, 18, salt))),
                               lane);
            storeSpatialVector(LK.mDeltaMotion,
                               UnAlignedSpatialVector(
                                   PxVec3(0.1f * fv(a, l, 19, salt), 0.1f * fv(a, l, 20, salt), 0.1f * fv(a, l, 21, salt)),
                                   PxVec3(0.1f * fv(a, l, 22, salt), 0.1f * fv(a, l, 23, salt), 0.1f * fv(a, l, 24, salt))),
                               lane);
            PxQuat dq(fv(a, l, 25, salt), fv(a, l, 26, salt), fv(a, l, 27, salt),
                      1.0f + 0.4f * fv(a, l, 28, salt));
            dq.normalize();
            LK.mDeltaQ[lane] = make_float4(dq.x, dq.y, dq.z, dq.w);
            storeSpatialVector(LK.mScratchImpulse,
                               UnAlignedSpatialVector(
                                   PxVec3(0.2f * fv(a, l, 52, salt), 0.2f * fv(a, l, 53, salt), 0.2f * fv(a, l, 54, salt)),
                                   PxVec3(0.2f * fv(a, l, 55, salt), 0.2f * fv(a, l, 56, salt), 0.2f * fv(a, l, 57, salt))),
                               lane);
            storeSpatialVector(LK.mSolverSpatialImpulse,
                               UnAlignedSpatialVector(
                                   PxVec3(0.2f * fv(a, l, 58, salt), 0.2f * fv(a, l, 59, salt), 0.2f * fv(a, l, 60, salt)),
                                   PxVec3(0.2f * fv(a, l, 61, salt), 0.2f * fv(a, l, 62, salt), 0.2f * fv(a, l, 63, salt))),
                               lane);
        }
        PX_ASSERT(cursor <= SL_ROOT_DOF_SLICE);

        for (uint32_t d = 0; d < SL_MAX_DOFS; ++d)
        {
            PxgArticulationBlockDofData& DD = D[d];
            const bool rootSlice = (d >= SL_ROOT_DOF_SLICE && d < SL_ROOT_DOF_SLICE + 3);
            const float big = rootSlice ? 11.0f : 1.0f;
            storeSpatialVector(DD.mWorldMotionMatrix,
                               UnAlignedSpatialVector(
                                   PxVec3(big * fv(a, d, 30, salt), big * fv(a, d, 31, salt), big * fv(a, d, 32, salt)),
                                   PxVec3(big * fv(a, d, 33, salt), big * fv(a, d, 34, salt), big * fv(a, d, 35, salt))),
                               lane);
            storeSpatialVector(DD.mIsInvDW,
                               UnAlignedSpatialVector(
                                   PxVec3(big * fv(a, d, 36, salt), big * fv(a, d, 37, salt), big * fv(a, d, 38, salt)),
                                   PxVec3(big * fv(a, d, 39, salt), big * fv(a, d, 40, salt), big * fv(a, d, 41, salt))),
                               lane);
            storeSpatialVector(DD.mIsW,
                               UnAlignedSpatialVector(
                                   PxVec3(big * fv(a, d, 42, salt), big * fv(a, d, 43, salt), big * fv(a, d, 44, salt)),
                                   PxVec3(big * fv(a, d, 45, salt), big * fv(a, d, 46, salt), big * fv(a, d, 47, salt))),
                               lane);
            DD.mInvStIsT_x[lane] = 0.5f * fv(a, d, 48, salt);
            DD.mInvStIsT_y[lane] = 0.5f * fv(a, d, 49, salt);
            DD.mInvStIsT_z[lane] = 0.5f * fv(a, d, 50, salt);
            DD.mDeferredQstZ[lane] = 0.3f * fv(a, d, 51, salt);
        }

        for (uint32_t l = 0; l < SL_MAX_LINKS; ++l)
        {
            uint64_t words[SL_WORDSIZE] = { 0, 0 };
            uint32_t c = l;
            for (;;)
            {
                words[c / 64u] |= (uint64_t(1) << (c % 64u));
                if (c == 0) break;
                c = slParent(c, lane);
            }
            for (uint32_t j = 0; j < SL_WORDSIZE; ++j)
                B.bf[(size_t(w) * SL_MAX_LINKS + l) * SL_WORDSIZE * 32 + size_t(j) * 32 + lane] =
                    words[j];
        }
    }

    // batches 1..6 live
    for (uint32_t b = 1; b <= 6; ++b)
    {
        PxgBlockConstraintBatch& BT = B.bt[b];
        BT.mDescStride = PxU16(23 + b);
        BT.constraintType = PxU16(slIsContact(b) ? PxgSolverConstraintDesc::eARTICULATION_CONTACT
                                                 : PxgSolverConstraintDesc::eARTICULATION_CONSTRAINT_1D);
        BT.mask = slMask(b);
        BT.mConstraintBatchIndex = b;
        BT.startConstraintIndex = slIsContact(b) ? (b * 4 + 1) : (b * 4 + 2);
        BT.startFrictionIndex = b * 5 + 2;
        BT.mArticulationResponseIndex = b * 8 + 1;
        for (uint32_t k = 0; k < 32; ++k)
        {
            uint32_t la, lb;
            slPair(b * 7 + k, la, lb);
            BT.bodyANodeIndex[k] = PxNodeIndex(50 + b, la);
            BT.bodyBNodeIndex[k] = PxNodeIndex(50 + b, lb);
        }

        if (slIsContact(b))
        {
            PxgTGSBlockSolverContactHeader& H = B.ch[b];
            PxgTGSBlockSolverFrictionHeader& F = B.fh[b];
            for (uint32_t lane = 0; lane < 32; ++lane)
            {
                H.invMass0_1_angDom0_1[lane] = make_float4(0.6f + 0.2f * fv(lane, b, 60, salt),
                                                           0.5f + 0.2f * fv(lane, b, 61, salt),
                                                           0.7f + 0.2f * fv(lane, b, 62, salt),
                                                           0.8f + 0.2f * fv(lane, b, 63, salt));
                H.normal_staticFriction[lane] = make_float4(fv(lane, b, 64, salt), fv(lane, b, 65, salt),
                                                            fv(lane, b, 66, salt), 0.6f + 0.3f * fv(lane, b, 67, salt));
                H.minNormalForce[lane] = 0.05f + 0.05f * fv(lane, b, 68, salt);
                H.maxPenBias[lane] = -0.4f + 0.1f * fv(lane, b, 69, salt);
                H.flags[lane] = ((lane + b) % 3u == 1u) ? COMPLIANT_ACCEL : 0u;
                H.numNormalConstr[lane] = 1u + ((lane + b) % 3u);
                H.forceWritebackOffset[lane] = 0;
                H.restitutionXdt[lane] = ((lane + b) % 4u == 1u) ? (-0.3f + 0.1f * fv(lane, b, 70, salt))
                                                                 : (0.2f + 0.1f * fv(lane, b, 71, salt));
                H.cfm[lane] = 0.01f + 0.005f * fv(lane, b, 72, salt);
                H.p8[lane] = 0.9f;
                F.numFrictionConstr[lane] = ((lane + b) % 3u) * 2u + (((lane + b) % 5u == 2u) ? 1u : 0u);
                F.dynamicFriction[lane] = 0.4f + 0.2f * fv(lane, b, 73, salt);
                F.biasCoefficient[lane] = 0.5f + 0.2f * fv(lane, b, 74, salt);
                F.torsionalFrictionScale[lane] = 0.8f + 0.1f * fv(lane, b, 75, salt);
                F.frictionNormals[0][lane] = make_float4(fv(lane, b, 76, salt), fv(lane, b, 77, salt), fv(lane, b, 78, salt), 0.f);
                F.frictionNormals[1][lane] = make_float4(fv(lane, b, 79, salt), fv(lane, b, 80, salt), fv(lane, b, 81, salt), 0.f);
                F.broken[lane] = 0;
                for (uint32_t i = 0; i < 3; ++i)
                {
                    PxgTGSBlockSolverContactPoint& C = B.cp[BT.startConstraintIndex + i];
                    C.raXn_extraCoeff[lane] = make_float4(fv(lane, b * 4 + i, 82, salt), fv(lane, b * 4 + i, 83, salt),
                                                          fv(lane, b * 4 + i, 84, salt), 0.5f + 0.3f * fv(lane, b * 4 + i, 85, salt));
                    C.rbXn_targetVelW[lane] = make_float4(fv(lane, b * 4 + i, 86, salt), fv(lane, b * 4 + i, 87, salt),
                                                          fv(lane, b * 4 + i, 88, salt), 0.1f * fv(lane, b * 4 + i, 89, salt));
                    C.separation[lane] = 0.05f * fv(lane, b * 4 + i, 90, salt);
                    C.maxImpulse[lane] = 3.0f + fv(lane, b * 4 + i, 91, salt);
                    C.appliedForce[lane] = 0.3f + 0.2f * fv(lane, b * 4 + i, 92, salt);
                    C.biasCoefficient[lane] = 0.4f + 0.2f * fv(lane, b * 4 + i, 93, salt);
                    C.resp0[lane] = 0.5f + 0.3f * fv(lane, b * 4 + i, 94, salt);
                    C.resp1[lane] = 0.5f + 0.3f * fv(lane, b * 4 + i, 95, salt);
                }
                for (uint32_t i = 0; i < 5; ++i)
                {
                    PxgTGSBlockSolverContactFriction& FR = B.fp[BT.startFrictionIndex + i];
                    FR.raXn_error[lane] = make_float4(fv(lane, b * 6 + i, 96, salt), fv(lane, b * 6 + i, 97, salt),
                                                      fv(lane, b * 6 + i, 98, salt), 0.05f * fv(lane, b * 6 + i, 99, salt));
                    FR.rbXn_targetVelW[lane] = make_float4(fv(lane, b * 6 + i, 100, salt), fv(lane, b * 6 + i, 101, salt),
                                                           fv(lane, b * 6 + i, 102, salt), 0.1f * fv(lane, b * 6 + i, 103, salt));
                    FR.appliedForce[lane] = 0.2f + 0.1f * fv(lane, b * 6 + i, 104, salt);
                    FR.resp0[lane] = 0.4f + 0.3f * fv(lane, b * 6 + i, 105, salt);
                    FR.resp1[lane] = 0.4f + 0.3f * fv(lane, b * 6 + i, 106, salt);
                }
            }
        }
        else
        {
            PxgTGSBlockSolverConstraint1DHeader& H = B.jh[b];
            for (uint32_t lane = 0; lane < 32; ++lane)
            {
                const uint32_t rc = 1u + ((lane + b) % 4u);
                uchar4 packed;
                packed.x = PxU8(rc); packed.y = PxU8(rc + 2); packed.z = 1; packed.w = 2;
                H.rowCounts_breakable_orthoAxisCount[lane] = packed;
                H.rAWorld_invMass0D0[lane] = make_float4(fv(lane, b, 110, salt), fv(lane, b, 111, salt),
                                                         fv(lane, b, 112, salt), 0.5f + 0.2f * fv(lane, b, 113, salt));
                H.rBWorld_invMass1D1[lane] = make_float4(fv(lane, b, 114, salt), fv(lane, b, 115, salt),
                                                         fv(lane, b, 116, salt), 0.5f + 0.2f * fv(lane, b, 117, salt));
                H.invInertiaScale0[lane] = 0.6f + 0.2f * fv(lane, b, 118, salt);
                H.invInertiaScale1[lane] = 0.6f + 0.2f * fv(lane, b, 119, salt);
                H.cfm[lane] = 0.01f + 0.004f * fv(lane, b, 120, salt);
                for (uint32_t i = 0; i < 4; ++i)
                {
                    PxgTGSBlockSolverConstraint1DCon& R = B.jr[BT.startConstraintIndex + i];
                    R.lin0XYZ_initBiasOrCoeff0[lane] = make_float4(fv(lane, b * 5 + i, 121, salt), fv(lane, b * 5 + i, 122, salt),
                                                                   fv(lane, b * 5 + i, 123, salt), 0.2f * fv(lane, b * 5 + i, 124, salt));
                    R.lin1XYZ_biasScaleOrCoeff1[lane] = make_float4(fv(lane, b * 5 + i, 125, salt), fv(lane, b * 5 + i, 126, salt),
                                                                    fv(lane, b * 5 + i, 127, salt), 0.5f + 0.2f * fv(lane, b * 5 + i, 128, salt));
                    R.ang0XYZ_velMultiplierOrCoeff2[lane] = make_float4(fv(lane, b * 5 + i, 129, salt), fv(lane, b * 5 + i, 130, salt),
                                                                        fv(lane, b * 5 + i, 131, salt), 0.6f + 0.2f * fv(lane, b * 5 + i, 132, salt));
                    R.ang1XYZ_velTargetOrCoeff3[lane] = make_float4(fv(lane, b * 5 + i, 133, salt), fv(lane, b * 5 + i, 134, salt),
                                                                    fv(lane, b * 5 + i, 135, salt), 0.1f * fv(lane, b * 5 + i, 136, salt));
                    R.resp0[lane] = 0.5f + 0.3f * fv(lane, b * 5 + i, 137, salt);
                    R.resp1[lane] = 0.5f + 0.3f * fv(lane, b * 5 + i, 138, salt);
                    R.geometricError[lane] = 0.1f * fv(lane, b * 5 + i, 139, salt);
                    R.minImpulse[lane] = -4.0f + fv(lane, b * 5 + i, 140, salt);
                    R.maxImpulse[lane] = 3.0f + fv(lane, b * 5 + i, 141, salt);
                    R.maxBias[lane] = 0.8f + 0.2f * fv(lane, b * 5 + i, 142, salt);
                    R.angularErrorScale[lane] = ((lane + i) % 2u) ? 1.0f : 0.7f;
                    static const uint32_t modes[5] = { 0u, DY_FLAG_SPRING, DY_FLAG_SPRING | DY_FLAG_ACCEL,
                                                       DY_FLAG_INEQ, DY_FLAG_SPRING | DY_FLAG_INEQ };
                    R.flags[lane] = modes[(lane + b + i) % 5u];
                    R.appliedForce[lane] = 0.2f + 0.2f * fv(lane, b * 5 + i, 143, salt);
                }
            }
        }
    }

    for (uint32_t r = 0; r < SL_NB_RESP; ++r)
        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            PxgArticulationBlockResponse& R = B.rs[r];
            R.deltaRALin_x[lane] = 0.3f * fv(lane, r, 150, salt);
            R.deltaRALin_y[lane] = 0.3f * fv(lane, r, 151, salt);
            R.deltaRALin_z[lane] = 0.3f * fv(lane, r, 152, salt);
            R.deltaRAAng_x[lane] = 0.3f * fv(lane, r, 153, salt);
            R.deltaRAAng_y[lane] = 0.3f * fv(lane, r, 154, salt);
            R.deltaRAAng_z[lane] = 0.3f * fv(lane, r, 155, salt);
            R.deltaRBLin_x[lane] = 0.3f * fv(lane, r, 156, salt);
            R.deltaRBLin_y[lane] = 0.3f * fv(lane, r, 157, salt);
            R.deltaRBLin_z[lane] = 0.3f * fv(lane, r, 158, salt);
            R.deltaRBAng_x[lane] = 0.3f * fv(lane, r, 159, salt);
            R.deltaRBAng_y[lane] = 0.3f * fv(lane, r, 160, salt);
            R.deltaRBAng_z[lane] = 0.3f * fv(lane, r, 161, salt);
        }
}

// ---- mirror ----

static void slMirrorCountSlabs(const SlArena& W, uint32_t bodyIndex, uint32_t& refCount)
{
    refCount = 0;
    for (uint32_t i = 0; i < SL_NUM32; ++i)
        refCount += PxU32(__builtin_popcount(W.enc[size_t(i) * SL_TOTAL_BODIES + bodyIndex]));
    refCount = PxMax(1u, refCount);
}

static void slMirror(SlArena& W, bool velIter, bool accumErr, SelfCoverage& cov,
                     uint32_t& hasImpulses, uint32_t& refGt1, uint32_t& refClamped,
                     uint32_t& alignedHits, uint32_t& unalignedHits)
{
    const float minPen = velIter ? 0.0f : -PX_MAX_F32;

    for (uint32_t a = 0; a < SL_NB; ++a)
    {
        const uint32_t w = a / 32, lane = a % 32;
        PxgArticulationBlockData& ab = W.ab[w];
        PxgArticulationBlockLinkData* lk = &W.lk[size_t(w) * SL_MAX_LINKS];
        PxgArticulationBlockDofData* df = &W.df[size_t(w) * SL_MAX_DOFS];
        const uint64_t* bf = &W.bf[size_t(w) * SL_MAX_LINKS * SL_WORDSIZE * 32];

        const uint32_t dirty = ab.mStateDirty[lane];
        if (dirty != 0)
        {
            if (dirty & 2u)   // eHAS_IMPULSES
            {
                ++hasImpulses;
                uint32_t refCount;
                slMirrorCountSlabs(W, SL_REF_OFFSET + a, refCount);
                if (refCount > 1) ++refGt1;
                bool allZero = true;
                for (uint32_t i = 0; i < SL_NUM32; ++i)
                    if (W.enc[size_t(i) * SL_TOTAL_BODIES + SL_REF_OFFSET + a]) allZero = false;
                if (allZero) ++refClamped;
                const float scale = 1.0f / float(refCount);
                pxgref::aiMirrorEx(W.slab.data(), W.imp.data(), &ab, lk, df, a,
                                   SL_MAX_LINKS, SL_NB, SL_NB_SLABS, scale, lane,
                                   alignedHits, unalignedHits);
            }
        }

        // resetSlabCount ALWAYS
        for (uint32_t i = 0; i < SL_NUM32; ++i)
            W.enc[size_t(i) * SL_TOTAL_BODIES + SL_REF_OFFSET + a] = 0u;

        ab.mStateDirty[lane] = 1u;   // eVEL_DIRTY

        RefError err;
        refSolveSelfLane(ab, lk, df, bf, SL_WORDSIZE,
                         &W.tmpA[size_t(w) * SL_WORDSIZE * 32],
                         &W.tmpB[size_t(w) * SL_WORDSIZE * 32],
                         &W.tmpC[size_t(w) * SL_WORDSIZE * 32],
                         W.bt.data(), W.ch.data(), W.fh.data(), W.cp.data(), W.fp.data(),
                         W.jh.data(), W.jr.data(), W.rs.data(),
                         SL_ELAPSED, minPen, accumErr, err, cov, lane);
    }
}

// ---- run ----

struct SlParamsHost
{
    uint64_t desc;        // @0
    uint64_t solverDesc;  // @8
    uint32_t velIterWord; // @16
    float elapsedTime;    // @20
    uint64_t sharedDesc;  // @24
    uint32_t doFrictionWord; // @32
    uint32_t pad0;        // @36
};
static_assert(sizeof(SlParamsHost) == 40, "params blob");

static size_t slRun(PxgCudaShim& shim, bool velIter, bool doFriction, int32_t errCounter,
                    uint32_t salt)
{
    SlArena A;
    slSeed(A, salt, errCounter);

    SlArena W = A;
    SelfCoverage cov;
    uint32_t hasImpulses = 0, refGt1 = 0, refClamped = 0, alignedHits = 0, unalignedHits = 0;
    const bool accumErr = errCounter >= 0;
    slMirror(W, velIter, accumErr, cov, hasImpulses, refGt1, refClamped, alignedHits,
             unalignedHits);

    std::printf("[ info ] launch(%s): contact %u / 1D %u; hasImpulses %u (ref>1 %u, clamped %u); "
                "word1 %u, equal %u, emptyB %u, off!=lane %u, inactive %u\n",
                velIter ? "vel" : "pos", cov.contacts, cov.oneD, hasImpulses, refGt1, refClamped,
                cov.word1Hops, cov.equalPairs, cov.emptyBWalks, cov.offsetNeLane, cov.inactive);
    PXG_CHECK_MSG(cov.contacts > 0 && cov.oneD > 0, "both constraint types must run");
    PXG_CHECK_MSG(hasImpulses > 0 && refGt1 > 0 && refClamped > 0, "impulse-average coverage hole");
    PXG_CHECK_MSG(cov.word1Hops > 0 && cov.equalPairs > 0 && cov.emptyBWalks > 0
                      && cov.offsetNeLane > 0 && cov.inactive > 0,
                  "walk coverage hole");
    PXG_CHECK_MSG(std::memcmp(W.ab.data(), A.ab.data(), W.ab.size() * sizeof(W.ab[0])) != 0,
                  "vacuous");

    std::vector<PxgDevicePtr> bufs;
    auto up = [&](const void* p, size_t n) {
        PxgDevicePtr dp = 0;
        shim.memAlloc(&dp, n);
        if (dp && p) shim.memcpyHtoD(dp, p, n);
        bufs.push_back(dp);
        return dp;
    };
    PxgDevicePtr dAb = up(A.ab.data(), A.ab.size() * sizeof(A.ab[0]));
    PxgDevicePtr dLk = up(A.lk.data(), A.lk.size() * sizeof(A.lk[0]));
    PxgDevicePtr dDf = up(A.df.data(), A.df.size() * sizeof(A.df[0]));
    PxgDevicePtr dBf = up(A.bf.data(), A.bf.size() * 8);
    PxgDevicePtr dTa = up(A.tmpA.data(), A.tmpA.size() * 8);
    PxgDevicePtr dTb = up(A.tmpB.data(), A.tmpB.size() * 8);
    PxgDevicePtr dTc = up(A.tmpC.data(), A.tmpC.size() * 8);
    PxgDevicePtr dEnc = up(A.enc.data(), A.enc.size() * 4);
    PxgDevicePtr dSlab = up(A.slab.data(), A.slab.size() * 4);
    PxgDevicePtr dImp = up(A.imp.data(), A.imp.size() * 4);
    PxgDevicePtr dBt = up(A.bt.data(), A.bt.size() * sizeof(A.bt[0]));
    PxgDevicePtr dCh = up(A.ch.data(), A.ch.size() * sizeof(A.ch[0]));
    PxgDevicePtr dFh = up(A.fh.data(), A.fh.size() * sizeof(A.fh[0]));
    PxgDevicePtr dCp = up(A.cp.data(), A.cp.size() * sizeof(A.cp[0]));
    PxgDevicePtr dFp = up(A.fp.data(), A.fp.size() * sizeof(A.fp[0]));
    PxgDevicePtr dJh = up(A.jh.data(), A.jh.size() * sizeof(A.jh[0]));
    PxgDevicePtr dJr = up(A.jr.data(), A.jr.size() * sizeof(A.jr[0]));
    PxgDevicePtr dRs = up(A.rs.data(), A.rs.size() * sizeof(A.rs[0]));
    PxgDevicePtr dIsl = up(A.islands.data(), A.islands.size() * sizeof(A.islands[0]));

    // patch the three descriptor blobs with the device pointers, then upload
    auto w64 = [&](std::vector<uint8_t>& v, uint32_t off, uint64_t x) {
        std::memcpy(&v[off], &x, 8);
    };
    w64(A.desc, D_ARTI_BLOCKS, uint64_t(dAb));
    w64(A.desc, D_LINK_BLOCKS, uint64_t(dLk));
    w64(A.desc, D_DOF_BLOCKS, uint64_t(dDf));
    w64(A.desc, D_PATH_TO_ROOT, uint64_t(dBf));
    w64(A.desc, D_TEMP_SHARED, uint64_t(dTa));
    w64(A.desc, D_TEMP_ROOT, uint64_t(dTb));
    w64(A.desc, D_TEMP_PATH_TO_ROOT, uint64_t(dTc));
    w64(A.desc, D_SLAB_HAS_CHANGES, uint64_t(dSlab));
    w64(A.desc, D_IMPULSES, uint64_t(dImp));
    w64(A.solverDesc, SD_ISLAND_CONTEXT_POOL, uint64_t(dIsl));
    w64(A.sharedDesc, SH_BATCH, uint64_t(dBt));
    w64(A.sharedDesc, SH_JHDR, uint64_t(dJh));
    w64(A.sharedDesc, SH_JROW, uint64_t(dJr));
    w64(A.sharedDesc, SH_CHDR, uint64_t(dCh));
    w64(A.sharedDesc, SH_FHDR, uint64_t(dFh));
    w64(A.sharedDesc, SH_CPTS, uint64_t(dCp));
    w64(A.sharedDesc, SH_FPTS, uint64_t(dFp));
    w64(A.sharedDesc, SH_ENC, uint64_t(dEnc));
    w64(A.sharedDesc, SH_RESP, uint64_t(dRs));
    PxgDevicePtr dDesc = up(A.desc.data(), A.desc.size());
    PxgDevicePtr dSd = up(A.solverDesc.data(), A.solverDesc.size());
    PxgDevicePtr dSh = up(A.sharedDesc.data(), A.sharedDesc.size());

    bool ok = true;
    for (PxgDevicePtr p : bufs) ok = ok && (p != 0);
    if (!ok)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
        return 0;
    }

    SlParamsHost prm = { uint64_t(dDesc), uint64_t(dSd), velIter ? 1u : 0u, SL_ELAPSED,
                         uint64_t(dSh), doFriction ? 1u : 0u, 0u };
    PXG_CHECK_MSG(shim.launchKernel(0, SL_WARPS, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch self-solve kernel");
    shim.streamSynchronize();

    size_t bad = 0;
    auto cmp = [&](PxgDevicePtr dp, const void* host, size_t n, const char* name) {
        std::vector<uint8_t> g(n);
        shim.memcpyDtoH(g.data(), dp, n);
        if (std::memcmp(g.data(), host, n) != 0)
        {
            size_t mm = 0, first = size_t(-1);
            const uint8_t* h = reinterpret_cast<const uint8_t*>(host);
            for (size_t i = 0; i < n; ++i)
                if (g[i] != h[i]) { ++mm; if (first == size_t(-1)) first = i; }
            std::printf("MM launch/%s: %zu/%zu bytes differ (first at %zu)\n", name, mm, n, first);
            bad += mm;
        }
    };
    cmp(dAb, W.ab.data(), W.ab.size() * sizeof(W.ab[0]), "artiBlocks");
    cmp(dLk, W.lk.data(), W.lk.size() * sizeof(W.lk[0]), "linkBlocks");
    cmp(dDf, W.df.data(), W.df.size() * sizeof(W.df[0]), "dofBlocks");
    cmp(dBf, W.bf.data(), W.bf.size() * 8, "linkBitFields");
    cmp(dTa, W.tmpA.data(), W.tmpA.size() * 8, "tmpA");
    cmp(dTb, W.tmpB.data(), W.tmpB.size() * 8, "tmpB");
    cmp(dTc, W.tmpC.data(), W.tmpC.size() * 8, "tmpCommon");
    cmp(dEnc, W.enc.data(), W.enc.size() * 4, "encodedRefCount");
    cmp(dSlab, W.slab.data(), W.slab.size() * 4, "slabHasChanges");
    cmp(dImp, W.imp.data(), W.imp.size() * 4, "impulses");
    cmp(dBt, W.bt.data(), W.bt.size() * sizeof(W.bt[0]), "batches");
    cmp(dCh, W.ch.data(), W.ch.size() * sizeof(W.ch[0]), "contactHeaders");
    cmp(dFh, W.fh.data(), W.fh.size() * sizeof(W.fh[0]), "frictionHeaders");
    cmp(dCp, W.cp.data(), W.cp.size() * sizeof(W.cp[0]), "contactPoints");
    cmp(dFp, W.fp.data(), W.fp.size() * sizeof(W.fp[0]), "frictions");
    cmp(dJh, W.jh.data(), W.jh.size() * sizeof(W.jh[0]), "jointHeaders");
    cmp(dJr, W.jr.data(), W.jr.size() * sizeof(W.jr[0]), "jointRows");
    cmp(dRs, W.rs.data(), W.rs.size() * sizeof(W.rs[0]), "responses");
    cmp(dIsl, W.islands.data(), W.islands.size() * sizeof(W.islands[0]), "islands");

    for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
    return bad;
}

} // namespace

PXG_TEST(artiSolveSelfLaunch)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiSolveSelf.spv", "artiPropagateRigidImpulsesAndSolveSelfConstraintsTGS1T", 40u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &slLoader);

    size_t bad = 0;
    bad += slRun(shim, false, true, 3, 21);    // position iteration, residuals on
    bad += slRun(shim, true, false, -1, 29);   // velocity iteration, residuals stay poisoned
    PXG_CHECK_MSG(bad == 0, "%zu mismatches", bad);
}
