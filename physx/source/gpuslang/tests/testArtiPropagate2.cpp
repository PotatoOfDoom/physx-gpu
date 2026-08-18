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

// testArtiPropagate2.cpp -- P8.101b/c: the averageDeltaV kernel pair
// artiPropagateImpulses2TGS + artiPropagateVelocityTGS (manifest 286/287)
// against a FULL host mirror over device-resident descriptors, launched in
// the engine's own order (impulses2, then velocity over a 2D grid).
//
// Mirror pieces, every one already swept on its own: pxgref::fvMirror
// (PxcFsFlushVelocity, the partition-0 arm), pxgref::aiMirror2 (P8.101a,
// the partial propagate), the countActiveSlabs transcription (eight lines,
// as in testArtiSolveSelfLaunch), and pxgtest::refPropAccelNoJVel (the
// DIRECT-order NoJVel hop) for the velocity walk.
//
// Wrapper surface under test:
//  * block (32,2,1) -- TWO warps per block: 68 articulations = 2 blocks,
//    warps 0..2 live, warp 3 fully guarded; a one-warp-shape mutant
//    (w = blockIdx.x) misindexes warps 1+;
//  * the velocity kernel's 2D grid: gridDim.y = nbSlabs, blockIdx.y IS the
//    slab id;
//  * **the per-partition path OVERLAP, transcribed as-is**: offset2 =
//    globalWarpIndex + partitionId * ceil(nb/32) * wordSize -- within one
//    partition, warp w starts at entry w (NOT w * wordSize), so with
//    wordSize 2 adjacent warps SHARE an entry (warp 0 reads e0,e1; warp 1
//    reads e1,e2).  Almost certainly an engine bug that only manifests
//    above 64 links (fourth upstream-bug candidate); the fixture seeds the
//    overlapping layout consistently on both sides;
//  * dirty gating: impulses2 skips a clean articulation entirely; partition
//    0 flushes + zeroes mStateDirty (ZERO, not eVEL_DIRTY); partition != 0
//    runs only under eHAS_IMPULSES and only when mPartitionAverageScale > 0,
//    and does NOT clear the flags;
//  * the velocity early-out on scale == 0, the two (dirtyIndex, writeIndex)
//    pairs, the wordA & ~wordB walk seeded from mCommonLinkDeltaVelocity
//    (only when partitionId != 0 AND dirtyLink != ~0), and the TGS packing
//    into three float4 slots at writeIndex/+32/+64.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "foundation/PxVec3.h"
#include "CmSpatialVector.h"
#include "DyFeatherstoneArticulation.h"
#include "PxSpatialMatrix.h"
#include "PxgArticulation.h"
#include "PxgIslandContext.h"
#include "pxgArtiRefHelpers.h"
#include "pxgArtiSolveRef.h"
#include "pxgArtiExtSolveRef.h"
#include "pxgArtiSelfRef.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
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

static std::vector<uint8_t> p2Loader(const char* file) { return loadShaderBlob(file); }

const uint32_t P2_NB = 68;              // 2 blocks of (32,2,1); warp 3 guarded
const uint32_t P2_BLOCKS = 2, P2_WARPS = 3;
const uint32_t P2_MAX_LINKS = 70;       // -> wordSize 2
const uint32_t P2_WORDSIZE = 2;
const uint32_t P2_MAX_DOFS = 100;
const uint32_t P2_NB_SLABS = 3;
const uint32_t P2_NB_BATCH = (P2_NB + 31) / 32;   // 3
const uint32_t P2_NB_PARTITIONS = 3;
const uint32_t P2_NONE = 0xFFFFFFFFu;
const uint32_t P2_BODY_START = 4, P2_BODY_COUNT = 7;   // refCountOffset = 11
const uint32_t P2_REF_OFFSET = P2_BODY_START + P2_BODY_COUNT;
const uint32_t P2_TOTAL_BODIES = P2_REF_OFFSET + P2_NB;
const uint32_t P2_NUM32 = (P2_NB_SLABS + 31) / 32;     // 1

// desc offsets, each pinned by static_assert in testDmaArtiResidual.cpp
const uint32_t D_NB_ARTICULATIONS = 44, D_NB_SLABS = 80, D_IMPULSES = 72;
const uint32_t D_SLAB_HAS_CHANGES = 88, D_SLAB_DIRTY_MASKS = 96;
const uint32_t D_PATHS_PER_PARTITION = 104, D_IMPULSE_HOLDING_LINK = 112;
const uint32_t D_PARTITION_AVG_SCALE = 120;
const uint32_t D_ARTI_BLOCKS = 128, D_LINK_BLOCKS = 136, D_DOF_BLOCKS = 184;
const uint32_t D_PATH_TO_ROOT = 176, D_MAX_LINKS = 248, D_MAX_DOFS = 252;
const uint32_t SD_ISLAND_CONTEXT_POOL = 48;
const uint32_t SH_ENC = 176;
const uint32_t SH_SIZE = 224, SD_SIZE = 320, DESC_SIZE = 360;

static float fv(uint32_t a, uint32_t l, uint32_t d, uint32_t salt)
{
    uint32_t h = a * 2654435761u + l * 40503u + d * 9973u + salt * 733u + 19u;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    return -1.0f + 2.0f * float(h & 0xFFFFFFu) / float(0x1000000);
}

static uint32_t p2NumLinks(uint32_t a) { return 60u + (a % 11u); }   // 60..70
static uint32_t p2Parent(uint32_t i)
{
    if (i == 0) return 0u;
    return (i >= 3u && (i % 7u) == 0u) ? (i - 3u) : (i - 1u);
}
static uint32_t p2Dofs(uint32_t a, uint32_t i) { return (a + i) % 3u; }

struct P2Arena
{
    std::vector<uint8_t> desc, solverDesc, sharedDesc;
    std::vector<PxgIslandContext> islands;
    std::vector<PxgArticulationBlockData> ab;
    std::vector<PxgArticulationBlockLinkData> lk;
    std::vector<PxgArticulationBlockDofData> df;
    std::vector<uint32_t> slab;       // uint2 pairs (slabHasChanges)
    std::vector<float> imp;
    std::vector<uint64_t> bf;         // per-link path bitfields (velocity walk)
    std::vector<uint64_t> ppp;        // pathToRootsPerPartition entries (u64[32] each)
    std::vector<uint32_t> masks;      // slabDirtyMasks (uint4 per arti per slab per partition)
    std::vector<uint32_t> holding;    // mImpulseHoldingLink, per arti per partition
    std::vector<float> pscale;        // mPartitionAverageScale, per arti per partition
    std::vector<uint32_t> enc;        // encodedReferenceCount rows
    std::vector<float> velOut;        // float4 slots
};

// unique write bases: slot idx * 96 keeps {wi, wi+32, wi+64} collision-free
static uint32_t p2WriteIndex(uint32_t gti, uint32_t slab, uint32_t i)
{
    return ((slab * P2_NB + gti) * 2u + i) * 96u;
}

static void p2Seed(P2Arena& B, uint32_t salt)
{
    B.desc.assign(DESC_SIZE, 0xAB);
    B.solverDesc.assign(SD_SIZE, 0xAB);
    B.sharedDesc.assign(SH_SIZE, 0xAB);
    B.islands.resize(2);
    B.ab.resize(P2_WARPS + 1);        // warp 3 allocated, fully poisoned
    B.lk.resize(size_t(P2_WARPS + 1) * P2_MAX_LINKS);
    B.df.resize(size_t(P2_WARPS + 1) * P2_MAX_DOFS);
    B.slab.resize(size_t(P2_NB_SLABS) * P2_NB * 2);
    B.imp.resize(size_t(P2_NB_SLABS) * P2_MAX_LINKS * P2_NB * 6);
    B.bf.resize(size_t(P2_WARPS + 1) * P2_MAX_LINKS * P2_WORDSIZE * 32);
    B.ppp.resize(size_t(P2_WARPS + 1 + P2_NB_PARTITIONS * P2_NB_BATCH * P2_WORDSIZE) * 32);
    B.masks.resize(size_t(P2_NB_PARTITIONS) * P2_NB_SLABS * P2_NB * 4);
    B.holding.resize(size_t(P2_NB_PARTITIONS) * P2_NB);
    B.pscale.resize(size_t(P2_NB_PARTITIONS) * P2_NB);
    B.enc.resize(size_t(P2_NUM32) * P2_TOTAL_BODIES);
    B.velOut.resize(size_t(p2WriteIndex(P2_NB - 1, P2_NB_SLABS - 1, 1) + 96) * 4);
    std::memset(B.islands.data(), 0xAB, B.islands.size() * sizeof(B.islands[0]));
    std::memset(B.ab.data(), 0xAB, B.ab.size() * sizeof(B.ab[0]));
    std::memset(B.lk.data(), 0xAB, B.lk.size() * sizeof(B.lk[0]));
    std::memset(B.df.data(), 0xAB, B.df.size() * sizeof(B.df[0]));
    std::memset(B.bf.data(), 0xAB, B.bf.size() * 8);
    std::memset(B.ppp.data(), 0xAB, B.ppp.size() * 8);
    std::memset(B.masks.data(), 0xAB, B.masks.size() * 4);
    std::memset(B.velOut.data(), 0xAB, B.velOut.size() * 4);

    auto w32 = [&](std::vector<uint8_t>& v, uint32_t off, uint32_t x) {
        std::memcpy(&v[off], &x, 4);
    };
    w32(B.desc, D_NB_ARTICULATIONS, P2_NB);
    w32(B.desc, D_NB_SLABS, P2_NB_SLABS);
    w32(B.desc, D_MAX_LINKS, P2_MAX_LINKS);
    w32(B.desc, D_MAX_DOFS, P2_MAX_DOFS);

    B.islands[0].mBodyStartIndex = P2_BODY_START;
    B.islands[0].mBodyCount = P2_BODY_COUNT;

    for (uint32_t r = 0; r < P2_NUM32; ++r)
        for (uint32_t bdy = 0; bdy < P2_TOTAL_BODIES; ++bdy)
        {
            uint32_t h = (bdy * 2654435761u) ^ (r * 40503u) ^ (salt * 733u);
            h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
            B.enc[size_t(r) * P2_TOTAL_BODIES + bdy] = h & 0x7u;   // 3 slabs -> low bits
        }
    // one articulation with all-zero rows: countActiveSlabs 0 -> the CUDA
    // divides 1/0 -> inf scale; keep those rows NONZERO instead (the engine
    // reaches this kernel only for articulations with references)
    for (uint32_t r = 0; r < P2_NUM32; ++r)
        for (uint32_t a = 0; a < P2_NB; ++a)
            if (B.enc[size_t(r) * P2_TOTAL_BODIES + P2_REF_OFFSET + a] == 0)
                B.enc[size_t(r) * P2_TOTAL_BODIES + P2_REF_OFFSET + a] = 1u + (a % 5u);

    for (size_t i = 0; i < B.slab.size(); ++i) B.slab[i] = P2_NONE;
    for (uint32_t a = 0; a < P2_NB; ++a)
        for (uint32_t s = 0; s < P2_NB_SLABS; ++s)
        {
            const uint32_t nl = p2NumLinks(a);
            if ((a + s) % 3u == 1u)
                B.slab[(size_t(s) * P2_NB + a) * 2 + 0] = 1u + ((a * 5u + s * 11u) % (nl - 1u));
            if ((a + s) % 4u == 2u)
                B.slab[(size_t(s) * P2_NB + a) * 2 + 1] = 1u + ((a * 7u + s * 13u) % (nl - 1u));
        }
    for (size_t i = 0; i < B.imp.size(); ++i)
        B.imp[i] = 0.4f * fv(uint32_t(i & 0xFFFFu), uint32_t(i >> 16), 7, salt);

    for (uint32_t a = 0; a < P2_NB; ++a)
    {
        const uint32_t w = a / 32, lane = a % 32;
        PxgArticulationBlockData& AB = B.ab[w];
        PxgArticulationBlockLinkData* L = &B.lk[size_t(w) * P2_MAX_LINKS];
        PxgArticulationBlockDofData* D = &B.df[size_t(w) * P2_MAX_DOFS];
        const uint32_t nl = p2NumLinks(a);

        AB.mStateDirty[lane] = PxU8(a % 4u);
        AB.mNumLinks[lane] = nl;
        AB.mFlags[lane] = (a % 3u == 0u) ? PxU32(PxArticulationFlag::eFIX_BASE) : 0u;
        AB.mLinkWithDeferredImpulse[lane] = 0u;   // rewritten by the partial propagate
        storeSpatialVector(AB.mRootDeferredZ,
                           UnAlignedSpatialVector(
                               PxVec3(fv(a, 90, 0, salt), fv(a, 90, 1, salt), fv(a, 90, 2, salt)),
                               PxVec3(fv(a, 90, 3, salt), fv(a, 90, 4, salt), fv(a, 90, 5, salt))),
                           lane);
        storeSpatialVector(AB.mCommonLinkDeltaVelocity,
                           UnAlignedSpatialVector(
                               PxVec3(0.2f * fv(a, 94, 0, salt), 0.2f * fv(a, 94, 1, salt), 0.2f * fv(a, 94, 2, salt)),
                               PxVec3(0.2f * fv(a, 94, 3, salt), 0.2f * fv(a, 94, 4, salt), 0.2f * fv(a, 94, 5, salt))),
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
        for (uint32_t i = 0; i < P2_MAX_LINKS; ++i)
        {
            PxgArticulationBlockLinkData& LK = L[i];
            const bool live = i < nl;
            const uint32_t dofs = (!live || i == 0) ? 0u : p2Dofs(a, i);
            LK.mDofs[lane] = uint8_t(dofs);
            LK.mParents[lane] = live ? p2Parent(i) : 0u;
            LK.mJointOffset[lane] = cursor;
            LK.mRw_x[lane] = fv(a, 10u + i * 3u, 0, salt);
            LK.mRw_y[lane] = fv(a, 11u + i * 3u, 0, salt);
            LK.mRw_z[lane] = fv(a, 12u + i * 3u, 0, salt);
            storeSpatialVector(LK.mMotionVelocity,
                               UnAlignedSpatialVector(
                                   PxVec3(fv(a, i, 13, salt), fv(a, i, 14, salt), fv(a, i, 15, salt)),
                                   PxVec3(fv(a, i, 16, salt), fv(a, i, 17, salt), fv(a, i, 18, salt))),
                               lane);
            storeSpatialVector(LK.mDeltaMotion,
                               UnAlignedSpatialVector(
                                   PxVec3(0.1f * fv(a, i, 19, salt), 0.1f * fv(a, i, 20, salt), 0.1f * fv(a, i, 21, salt)),
                                   PxVec3(0.1f * fv(a, i, 22, salt), 0.1f * fv(a, i, 23, salt), 0.1f * fv(a, i, 24, salt))),
                               lane);
            storeSpatialVector(LK.mScratchImpulse,
                               UnAlignedSpatialVector(
                                   PxVec3(0.2f * fv(a, i, 45, salt), 0.2f * fv(a, i, 46, salt), 0.2f * fv(a, i, 47, salt)),
                                   PxVec3(0.2f * fv(a, i, 48, salt), 0.2f * fv(a, i, 49, salt), 0.2f * fv(a, i, 50, salt))),
                               lane);
            storeSpatialVector(LK.mSolverSpatialImpulse,
                               UnAlignedSpatialVector(
                                   PxVec3(0.2f * fv(a, i, 52, salt), 0.2f * fv(a, i, 53, salt), 0.2f * fv(a, i, 54, salt)),
                                   PxVec3(0.2f * fv(a, i, 55, salt), 0.2f * fv(a, i, 56, salt), 0.2f * fv(a, i, 57, salt))),
                               lane);
            storeSpatialVector(LK.mSolverSpatialDeltaVel,
                               UnAlignedSpatialVector(PxVec3(0.f), PxVec3(0.f)), lane);
            storeSpatialVector(LK.mConstraintForces,
                               UnAlignedSpatialVector(PxVec3(0.f), PxVec3(0.f)), lane);
            PxSpatialMatrix R;
            for (uint32_t c = 0; c < 6; ++c)
                for (uint32_t r = 0; r < 6; ++r)
                    R.column[c][r] = 0.2f * fv(a, 1300u + i * 40u + c * 6u + r, 0, salt);
            storeSpatialMatrix(LK.mSpatialResponseMatrix, lane, R);

            if (live)
            {
                uint64_t words[2] = { 0, 0 };
                uint32_t c = i;
                for (;;)
                {
                    words[c / 64u] |= (uint64_t(1) << (c % 64u));
                    if (c == 0) break;
                    c = p2Parent(c);
                }
                for (uint32_t j = 0; j < P2_WORDSIZE; ++j)
                    B.bf[(size_t(w) * P2_MAX_LINKS + i) * P2_WORDSIZE * 32 + size_t(j) * 32 + lane] =
                        words[j];
            }
            cursor += dofs;
        }
        PX_ASSERT(cursor <= P2_MAX_DOFS);

        for (uint32_t dnum = 0; dnum < P2_MAX_DOFS; ++dnum)
        {
            PxgArticulationBlockDofData& DD = D[dnum];
            storeSpatialVector(DD.mIsInvDW,
                               UnAlignedSpatialVector(
                                   PxVec3(fv(a, dnum, 30, salt), fv(a, dnum, 31, salt), fv(a, dnum, 32, salt)),
                                   PxVec3(fv(a, dnum, 33, salt), fv(a, dnum, 34, salt), fv(a, dnum, 35, salt))),
                               lane);
            storeSpatialVector(DD.mWorldMotionMatrix,
                               UnAlignedSpatialVector(
                                   PxVec3(fv(a, dnum, 36, salt), fv(a, dnum, 37, salt), fv(a, dnum, 38, salt)),
                                   PxVec3(fv(a, dnum, 39, salt), fv(a, dnum, 40, salt), fv(a, dnum, 41, salt))),
                               lane);
            storeSpatialVector(DD.mIsW,
                               UnAlignedSpatialVector(
                                   PxVec3(fv(a, dnum, 42, salt), fv(a, dnum, 43, salt), fv(a, dnum, 44, salt)),
                                   PxVec3(fv(a, dnum, 45, salt), fv(a, dnum, 46, salt), fv(a, dnum, 47, salt))),
                               lane);
            DD.mInvStIsT_x[lane] = 0.5f * fv(a, dnum, 48, salt);
            DD.mInvStIsT_y[lane] = 0.5f * fv(a, dnum, 49, salt);
            DD.mInvStIsT_z[lane] = 0.5f * fv(a, dnum, 50, salt);
            DD.mDeferredQstZ[lane] = 0.3f * fv(a, dnum, 51, salt);
            DD.mJointVelocities[lane] = 0.4f * fv(a, dnum, 52, salt);
            DD.mJointUnconstrainedVelocities[lane] = 0.4f * fv(a, dnum, 53, salt);
            DD.mJointAccel[lane] = 0.2f * fv(a, dnum, 54, salt);
        }

        // per-partition tables: scale 0 on some (arti, partition) pairs, the
        // holding link an ancestor of most dirty indices
        for (uint32_t p = 0; p < P2_NB_PARTITIONS; ++p)
        {
            const size_t off = size_t(p) * P2_NB + a;
            const bool involved = ((a + p) % 5u) != 3u;
            B.pscale[off] = involved ? (0.25f + 0.05f * float((a + p) % 8u)) : 0.0f;
            uint32_t hold;
            const uint32_t hm = (a + p) % 4u;
            if (hm == 0u) hold = 0u;
            else if (hm == 1u) hold = p2Parent(p2Parent(nl - 1u));
            else if (hm == 2u) { hold = nl - 1u; while (hold >= 64u) hold = p2Parent(hold); }
            else hold = P2_NONE;
            B.holding[off] = involved ? hold : P2_NONE;
        }
    }

    // pathToRootsPerPartition: the OVERLAPPING layout, seeded consistently --
    // entry e, lane l carries the path-union bits for whatever reads it
    for (size_t e = 0; e < B.ppp.size() / 32; ++e)
        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            uint32_t h = uint32_t(e) * 2654435761u + lane * 40503u + salt * 733u;
            h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
            // sparse-ish bits over the live link range, none at bit 0
            uint64_t word = 0;
            for (uint32_t b = 1; b < 64; ++b)
                if (((h >> (b % 29u)) ^ (b * 2654435761u)) % 23u == 3u)
                    word |= (uint64_t(1) << b);
            B.ppp[e * 32 + lane] = word;
        }
    // clamp the per-partition path bits to each lane's live links, so the
    // walk never reads a poisoned link block: word0 bits < numLinks, word1
    // bits < numLinks-64 (empty for lanes with <= 64 links)
    for (uint32_t p = 0; p < P2_NB_PARTITIONS; ++p)
        for (uint32_t w = 0; w < P2_WARPS + 1; ++w)
            for (uint32_t j = 0; j < P2_WORDSIZE; ++j)
            {
                const size_t e = size_t(w) + size_t(p) * P2_NB_BATCH * P2_WORDSIZE + j;
                if (e >= B.ppp.size() / 32) continue;
                for (uint32_t lane = 0; lane < 32; ++lane)
                {
                    const uint32_t a = w * 32 + lane;
                    if (a >= P2_NB) { B.ppp[e * 32 + lane] = 0; continue; }
                    const uint32_t nl = p2NumLinks(a);
                    // this entry serves as word j for warp w AND word j-1 for
                    // warp w+1 (the overlap) -- clamp to the TIGHTER range
                    uint64_t m0 = (j == 0)
                        ? ((nl >= 64u) ? ~0ull : ((uint64_t(1) << nl) - 1u)) & ~1ull
                        : ((nl > 64u) ? ((uint64_t(1) << (nl - 64u)) - 1u) : 0ull);
                    const uint32_t a2 = (w + 1) * 32 + lane;   // the overlapping reader
                    if (j == 1 && a2 < P2_NB)
                    {
                        const uint32_t nl2 = p2NumLinks(a2);
                        const uint64_t m1 =
                            ((nl2 >= 64u) ? ~0ull : ((uint64_t(1) << nl2) - 1u)) & ~1ull;
                        m0 &= m1;   // entry w+1 is warp w's word 1 AND warp w+1's word 0
                    }
                    B.ppp[e * 32 + lane] &= m0;
                }
            }

    // slabDirtyMasks: per partition, per slab, per articulation
    for (uint32_t p = 0; p < P2_NB_PARTITIONS; ++p)
        for (uint32_t s = 0; s < P2_NB_SLABS; ++s)
            for (uint32_t a = 0; a < P2_NB; ++a)
            {
                const size_t mi = (size_t(p) * P2_NB_SLABS * P2_NB + size_t(s) * P2_NB + a) * 4;
                const uint32_t nl = p2NumLinks(a);
                const uint32_t mode = (a + s + p) % 4u;
                B.masks[mi + 0] = (mode == 0u || mode == 2u)
                                      ? 1u + ((a * 3u + s * 5u + p) % (nl - 1u)) : P2_NONE;
                B.masks[mi + 1] = p2WriteIndex(a, s, 0);
                B.masks[mi + 2] = (mode == 1u || mode == 2u)
                                      ? 1u + ((a * 7u + s * 11u + p) % (nl - 1u)) : P2_NONE;
                B.masks[mi + 3] = p2WriteIndex(a, s, 1);
            }
}

// ---- mirrors ----

static uint32_t p2CountSlabs(const P2Arena& W, uint32_t bodyIndex)
{
    uint32_t rc = 0;
    for (uint32_t i = 0; i < P2_NUM32; ++i)
        rc += PxU32(__builtin_popcount(W.enc[size_t(i) * P2_TOTAL_BODIES + bodyIndex]));
    return PxMax(1u, rc);
}

static void p2MirrorImpulses(P2Arena& W, uint32_t partitionId,
                             uint32_t& flushes, uint32_t& partials, uint32_t& scaleSkips)
{
    for (uint32_t a = 0; a < P2_NB; ++a)
    {
        const uint32_t w = a / 32, lane = a % 32;
        PxgArticulationBlockData& ab = W.ab[w];
        PxgArticulationBlockLinkData* lk = &W.lk[size_t(w) * P2_MAX_LINKS];
        PxgArticulationBlockDofData* df = &W.df[size_t(w) * P2_MAX_DOFS];

        const uint32_t dirtyFlag = ab.mStateDirty[lane];
        if (dirtyFlag == 0)
            continue;

        if (partitionId == 0)
        {
            ++flushes;
            pxgref::fvMirror(&ab, lk, df, ab.mNumLinks[lane],
                             (ab.mFlags[lane] & PxArticulationFlag::eFIX_BASE) != 0, lane);
            ab.mStateDirty[lane] = 0;
        }
        else if (dirtyFlag & 2u)
        {
            const size_t off = size_t(partitionId) * P2_NB + a;
            if (W.pscale[off] > 0.f)
            {
                ++partials;
                const uint32_t dirtyLink = W.holding[off];
                const uint32_t offset2 = w + partitionId * P2_NB_BATCH * P2_WORDSIZE;
                const uint32_t refCount = p2CountSlabs(W, P2_REF_OFFSET + a);
                const float scale = 1.0f / float(refCount);
                uint32_t slabHits = 0, walkHits = 0, lowerWordHits = 0;
                pxgref::aiMirror2(W.slab.data(), W.imp.data(), &ab, lk, df, a,
                                  P2_MAX_LINKS, P2_NB, P2_NB_SLABS, scale,
                                  &W.ppp[size_t(offset2) * 32], P2_WORDSIZE,
                                  dirtyLink, dirtyFlag, lane,
                                  slabHits, walkHits, lowerWordHits);
            }
            else
                ++scaleSkips;
        }
    }
}

static void p2MirrorVelocity(P2Arena& W, uint32_t partitionId, bool isTGS,
                             uint32_t& writes, uint32_t& corrected, uint32_t& earlyOuts)
{
    for (uint32_t slabId = 0; slabId < P2_NB_SLABS; ++slabId)
        for (uint32_t a = 0; a < P2_NB; ++a)
        {
            const uint32_t w = a / 32, lane = a % 32;
            const size_t off = size_t(partitionId) * P2_NB + a;
            if (W.pscale[off] == 0.f) { ++earlyOuts; continue; }

            PxgArticulationBlockData& ab = W.ab[w];
            PxgArticulationBlockLinkData* lk = &W.lk[size_t(w) * P2_MAX_LINKS];
            PxgArticulationBlockDofData* df = &W.df[size_t(w) * P2_MAX_DOFS];
            const uint64_t* bf = &W.bf[size_t(w) * P2_MAX_LINKS * P2_WORDSIZE * 32];
            const uint32_t dirtyLink = W.holding[off];

            const size_t mi = (size_t(partitionId) * P2_NB_SLABS * P2_NB
                               + size_t(slabId) * P2_NB + a) * 4;
            for (uint32_t i = 0; i < 2; ++i)
            {
                const uint32_t dirtyIndex = W.masks[mi + (i ? 2 : 0)];
                const uint32_t writeIndex = W.masks[mi + (i ? 3 : 1)];
                if (dirtyIndex == P2_NONE)
                    continue;

                UnAlignedSpatialVector velocity =
                    loadSpatialVector(lk[dirtyIndex].mMotionVelocity, lane);

                if (partitionId != 0 && dirtyLink != P2_NONE)
                {
                    UnAlignedSpatialVector deltaV =
                        loadSpatialVector(ab.mCommonLinkDeltaVelocity, lane);
                    for (uint32_t j = 0; j < P2_WORDSIZE; ++j)
                    {
                        const uint32_t wordOffset = j * 64u;
                        const uint64_t wordA =
                            bf[(size_t(dirtyIndex) * P2_WORDSIZE + j) * 32 + lane];
                        const uint64_t wordB =
                            bf[(size_t(dirtyLink) * P2_WORDSIZE + j) * 32 + lane];
                        uint64_t pathToRoot = wordA & (~wordB);
                        while (pathToRoot)
                        {
                            const uint32_t index =
                                uint32_t(__builtin_ctzll(pathToRoot)) + wordOffset;
                            deltaV = pxgtest::refPropAccelNoJVel(lk, index, df, deltaV, lane);
                            pathToRoot &= (pathToRoot - 1);
                        }
                    }
                    velocity += deltaV;
                    ++corrected;
                }

                float* o0 = &W.velOut[size_t(writeIndex) * 4];
                float* o1 = &W.velOut[size_t(writeIndex + 32) * 4];
                if (isTGS)
                {
                    const UnAlignedSpatialVector deltaMotion =
                        loadSpatialVector(lk[dirtyIndex].mDeltaMotion, lane);
                    float* o2 = &W.velOut[size_t(writeIndex + 64) * 4];
                    o0[0] = velocity.bottom.x; o0[1] = velocity.bottom.y;
                    o0[2] = velocity.bottom.z; o0[3] = velocity.top.x;
                    o1[0] = velocity.top.y; o1[1] = velocity.top.z;
                    o1[2] = deltaMotion.bottom.x; o1[3] = deltaMotion.bottom.y;
                    o2[0] = deltaMotion.bottom.z; o2[1] = deltaMotion.top.x;
                    o2[2] = deltaMotion.top.y; o2[3] = deltaMotion.top.z;
                }
                else
                {
                    // the PGS packing: two slots, w = 0, NO deltaMotion -- the
                    // third slot region stays poisoned and the byte compare
                    // proves the PGS kernel never touches it
                    o0[0] = velocity.bottom.x; o0[1] = velocity.bottom.y;
                    o0[2] = velocity.bottom.z; o0[3] = 0.f;
                    o1[0] = velocity.top.x; o1[1] = velocity.top.y;
                    o1[2] = velocity.top.z; o1[3] = 0.f;
                }
                ++writes;
            }
        }
}

// ---- run ----

struct P2iParamsHost
{
    uint64_t scDesc, solverDesc, sharedDesc;
    uint32_t partitionId, pad0;
};
static_assert(sizeof(P2iParamsHost) == 32, "impulses2 blob");

struct P2vParamsHost
{
    uint64_t scDesc;
    uint32_t partitionId, pad0;
    uint64_t velocityOutput;
};
static_assert(sizeof(P2vParamsHost) == 24, "velocity blob");

static size_t p2Run(PxgCudaShim& shim, uint32_t partitionId, uint32_t salt, bool isTGS)
{
    P2Arena A;
    p2Seed(A, salt);

    P2Arena W = A;
    uint32_t flushes = 0, partials = 0, scaleSkips = 0;
    p2MirrorImpulses(W, partitionId, flushes, partials, scaleSkips);
    uint32_t writes = 0, corrected = 0, earlyOuts = 0;
    p2MirrorVelocity(W, partitionId, isTGS, writes, corrected, earlyOuts);

    std::printf("[ info ] prop2(p=%u,%s): flushes %u, partials %u, scaleSkips %u; "
                "writes %u, corrected %u, earlyOuts %u\n",
                partitionId, isTGS ? "tgs" : "pgs", flushes, partials, scaleSkips, writes,
                corrected, earlyOuts);
    if (partitionId == 0)
        PXG_CHECK_MSG(flushes > 0 && corrected == 0, "partition-0 coverage");
    else
        PXG_CHECK_MSG(partials > 0 && scaleSkips > 0 && corrected > 0, "partition-N coverage");
    PXG_CHECK_MSG(writes > 0 && earlyOuts > 0, "velocity coverage");

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
    PxgDevicePtr dSlab = up(A.slab.data(), A.slab.size() * 4);
    PxgDevicePtr dImp = up(A.imp.data(), A.imp.size() * 4);
    PxgDevicePtr dBf = up(A.bf.data(), A.bf.size() * 8);
    PxgDevicePtr dPpp = up(A.ppp.data(), A.ppp.size() * 8);
    PxgDevicePtr dMasks = up(A.masks.data(), A.masks.size() * 4);
    PxgDevicePtr dHold = up(A.holding.data(), A.holding.size() * 4);
    PxgDevicePtr dPs = up(A.pscale.data(), A.pscale.size() * 4);
    PxgDevicePtr dEnc = up(A.enc.data(), A.enc.size() * 4);
    PxgDevicePtr dIsl = up(A.islands.data(), A.islands.size() * sizeof(A.islands[0]));
    PxgDevicePtr dVo = up(A.velOut.data(), A.velOut.size() * 4);

    auto w64 = [&](std::vector<uint8_t>& v, uint32_t off, uint64_t x) {
        std::memcpy(&v[off], &x, 8);
    };
    w64(A.desc, D_ARTI_BLOCKS, uint64_t(dAb));
    w64(A.desc, D_LINK_BLOCKS, uint64_t(dLk));
    w64(A.desc, D_DOF_BLOCKS, uint64_t(dDf));
    w64(A.desc, D_SLAB_HAS_CHANGES, uint64_t(dSlab));
    w64(A.desc, D_IMPULSES, uint64_t(dImp));
    w64(A.desc, D_PATH_TO_ROOT, uint64_t(dBf));
    w64(A.desc, D_PATHS_PER_PARTITION, uint64_t(dPpp));
    w64(A.desc, D_SLAB_DIRTY_MASKS, uint64_t(dMasks));
    w64(A.desc, D_IMPULSE_HOLDING_LINK, uint64_t(dHold));
    w64(A.desc, D_PARTITION_AVG_SCALE, uint64_t(dPs));
    w64(A.solverDesc, SD_ISLAND_CONTEXT_POOL, uint64_t(dIsl));
    w64(A.sharedDesc, SH_ENC, uint64_t(dEnc));
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

    P2iParamsHost pi = { uint64_t(dDesc), uint64_t(dSd), uint64_t(dSh), partitionId, 0u };
    PXG_CHECK_MSG(shim.launchKernel(isTGS ? 0 : 2, P2_BLOCKS, 1, 1, &pi, sizeof(pi)) ==
                      PxgResult::Success,
                  "launch artiPropagateImpulses2");
    P2vParamsHost pv = { uint64_t(dDesc), partitionId, 0u, uint64_t(dVo) };
    PXG_CHECK_MSG(shim.launchKernel(isTGS ? 1 : 3, P2_BLOCKS, P2_NB_SLABS, 1, &pv, sizeof(pv)) ==
                      PxgResult::Success,
                  "launch artiPropagateVelocity");
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
            std::printf("MM prop2/p%u/%s: %zu/%zu bytes differ (first at %zu)\n",
                        partitionId, name, mm, n, first);
            bad += mm;
        }
    };
    cmp(dAb, W.ab.data(), W.ab.size() * sizeof(W.ab[0]), "artiBlocks");
    cmp(dLk, W.lk.data(), W.lk.size() * sizeof(W.lk[0]), "linkBlocks");
    cmp(dDf, W.df.data(), W.df.size() * sizeof(W.df[0]), "dofBlocks");
    cmp(dSlab, W.slab.data(), W.slab.size() * 4, "slabHasChanges");
    cmp(dImp, W.imp.data(), W.imp.size() * 4, "impulses");
    cmp(dBf, W.bf.data(), W.bf.size() * 8, "linkBitFields");
    cmp(dPpp, W.ppp.data(), W.ppp.size() * 8, "pathsPerPartition");
    cmp(dMasks, W.masks.data(), W.masks.size() * 4, "slabDirtyMasks");
    cmp(dHold, W.holding.data(), W.holding.size() * 4, "impulseHoldingLink");
    cmp(dPs, W.pscale.data(), W.pscale.size() * 4, "partitionAverageScale");
    cmp(dEnc, W.enc.data(), W.enc.size() * 4, "encodedRefCount");
    cmp(dVo, W.velOut.data(), W.velOut.size() * 4, "velocityOutput");

    for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
    return bad;
}

} // namespace

PXG_TEST(artiPropagate2)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiPropagate2.spv", "artiPropagateImpulses2TGS", 32u, 0 },
        { "npArtiPropagate2.spv", "artiPropagateVelocityTGS", 24u, 0 },
        { "npArtiPropagate2.spv", "artiPropagateImpulses2PGS", 32u, 0 },
        { "npArtiPropagate2.spv", "artiPropagateVelocity", 24u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 4, &p2Loader);

    size_t bad = 0;
    bad += p2Run(shim, 2u, 43, true);    // the partial-propagate + corrected-velocity path
    bad += p2Run(shim, 0u, 47, true);    // the flush path + uncorrected velocity
    // P8.109: the PGS twins over fresh arenas -- same bodies, the velocity
    // tail packs two slots instead of three
    bad += p2Run(shim, 2u, 53, false);
    bad += p2Run(shim, 0u, 59, false);
    PXG_CHECK_MSG(bad == 0, "%zu mismatches", bad);
}
