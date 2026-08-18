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

// testArtiSolvePartitionWrap.cpp -- P8.104: the ARTICULATION branch of
// solveBlockUnified (artiSolveBlockPartitionTGSInternal,
// solverMultiBlockTGS.cu:756-996), driven through the REAL engine entry with
// grid (2,2,1) -- blockIdx.y==0 runs the rigid path against an EMPTY rigid
// partition (start==end), blockIdx.y==1 runs the new articulation branch.
// Mirror: refSolveExtContactTGS / refSolveExt1DTGS (P8.99a, over the engine
// structs) + the countActiveSlabs popcount formula (P8.99c) + the staging
// unpack / writeback transcription.
//
// Geometry: 2 live arti batches at k=8 (contact, stride 26) and k=9 (1D,
// stride 13) behind articulationBatchOffset=3 (islandContextPool[0]
// .mBatchCount -- the POINTER-deref trap: island fields island[islandIndex=1]
// carry decoy values for the fields the kernel must read from pool[0]);
// warps 2..3 pin the k >= endIndex guard.  Lanes mix arti/rigid pairs on
// both sides: batch 8 lane%3 = {A arti / B rigid-dyn, A rigid / B arti,
// arti-arti}; batch 9 lane%4 = {rigid-rigid, A arti, rigid-rigid, B arti}
// so the 1D deltaQ loads exercise BOTH the block mDeltaQ and the
// txIData.deltaBody2World.q arm.  (artiId, slab) pairs are globally unique
// per side so the deferredZ / slabDirty scatters never collide.
// Two rounds: A partition 2 (the artiConstraintsPerPartition[idx-1] arm,
// residual disabled -> the accumulator must stay untouched), B partition 0
// (the mArtiBatchStartIndex arm, residual enabled -- only the contact warp
// contributes, so the atomic tail stays deterministic).

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "foundation/PxVec3.h"
#include "foundation/PxQuat.h"
#include "foundation/PxTransform.h"
#include "CmSpatialVector.h"
#include "PxNodeIndex.h"
#include "PxgSolverBody.h"
#include "PxgConstraintBlock.h"
#include "PxgSolverConstraintDesc.h"
#include "PxgSolverConstraintBlock1D.h"
#include "PxgSolverCoreDesc.h"
#include "PxgArticulation.h"
#include "PxgIslandContext.h"
#include "pxgArtiExtSolveRef.h"

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

static std::vector<uint8_t> aswLoader(const char* file) { return loadShaderBlob(file); }

const uint32_t ASW_NB_ARTI = 48;
const uint32_t ASW_MAX_LINKS = 9;
const uint32_t ASW_NB_SLABS = 3;
const uint32_t ASW_NUM32 = 1;              // ceil(numSlabs/32)
const uint32_t ASW_BODY_OFFSET = 4;
const uint32_t ASW_NB_DYN = 25;
const uint32_t ASW_TOTAL_BODIES = ASW_BODY_OFFSET + ASW_NB_DYN + ASW_NB_ARTI;   // 77
const uint32_t ASW_NB_RIGID = ASW_BODY_OFFSET + ASW_NB_DYN;                     // txIData pool
const uint32_t ASW_BATCH_OFFSET = 3;       // islandContextPool[0].mBatchCount
const uint32_t ASW_K0 = 8;                 // first live batch slot
const uint32_t ASW_NB_BATCH_ALLOC = 12;
const uint32_t ASW_NB_AB_BLOCKS = 2;       // ceil(48/32)
const uint32_t ASW_VEL_POOL = 16000;       // float4 slots (staging + rigid out region)
const uint32_t ASW_OUT_BASE = 3000;
const uint32_t ASW_NB_HDRS = 12;
const uint32_t ASW_NB_CPTS = 60;
const uint32_t ASW_NB_FPTS = 70;
const uint32_t ASW_NB_ROWS = 60;
const uint32_t ASW_NB_RESP = 120;

static_assert(sizeof(PxgBlockConstraintBatch) == 1536, "batch stride");
static_assert(sizeof(PxgArticulationBlockData) == 7168, "arti block stride");

static float fv(uint32_t a, uint32_t b, uint32_t c, uint32_t salt)
{
    uint32_t h = a * 2654435761u ^ (b * 40503u + c * 9176u) ^ (salt * 733u);
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    return (float(h % 479u) + 1.0f) * 0.0078125f - 1.85f;
}

// lane roles per batch
static bool aswIsArtiA(uint32_t b, uint32_t lane)
{
    return (b == 0) ? (lane % 3u != 1u) : (lane % 4u == 1u);
}
static bool aswIsArtiB(uint32_t b, uint32_t lane)
{
    return (b == 0) ? (lane % 3u != 0u) : (lane % 4u == 3u);
}
static uint32_t aswArtiIdA(uint32_t b, uint32_t lane)
{
    return (b == 0) ? lane : 40u + lane / 4u;            // batch8: 0..25; batch9: 40..42
}
static uint32_t aswArtiIdB(uint32_t b, uint32_t lane)
{
    return (b == 0) ? 26u + lane / 3u : 44u + lane / 4u; // batch8: 26..34; batch9: 44..46
}

struct AswArena
{
    std::vector<PxgBlockConstraintBatch> bt;
    std::vector<PxgTGSBlockSolverContactHeader> ch;
    std::vector<PxgTGSBlockSolverFrictionHeader> fh;
    std::vector<PxgTGSBlockSolverContactPoint> cp;
    std::vector<PxgTGSBlockSolverContactFriction> fp;
    std::vector<PxgTGSBlockSolverConstraint1DHeader> jh;
    std::vector<PxgTGSBlockSolverConstraint1DCon> jr;
    std::vector<PxgArticulationBlockResponse> rs;
    std::vector<float4> vel;
    std::vector<float> dz;                 // deferredZ, 6 floats per slot
    std::vector<uint32_t> slab;            // uint2 per (slab, arti)
    std::vector<PxgArticulationBlockData> ab;
    std::vector<PxgArticulationBlockLinkData> lk;
    std::vector<PxgSolverTxIData> txi;
    std::vector<uint32_t> enc;
    std::vector<PxgIslandContext> islands;
    std::vector<uint32_t> acp, cpp;        // arti / rigid constraintsPerPartition
    PxgSolverCoreDesc solverDesc;
    std::vector<uint8_t> artiDesc;         // 360-byte PxgArticulationCoreDesc image
};

// PXG_ARTI_CORE_* offsets, pinned by testDmaArtiResidual
const uint32_t ASW_D_SLAB_HAS_CHANGES = 88;
const uint32_t ASW_D_ARTICULATION_BLOCKS = 128;
const uint32_t ASW_D_LINK_BLOCKS = 136;
const uint32_t ASW_D_MAX_LINKS = 248;
const uint32_t ASW_DESC_SIZE = 360;

static void aswSeed(AswArena& W, uint32_t salt, uint32_t partitionIndex, int32_t errCounter)
{
    W.bt.assign(ASW_NB_BATCH_ALLOC, PxgBlockConstraintBatch());
    W.ch.resize(ASW_NB_HDRS);
    W.fh.resize(ASW_NB_HDRS);
    W.cp.resize(ASW_NB_CPTS);
    W.fp.resize(ASW_NB_FPTS);
    W.jh.resize(ASW_NB_HDRS);
    W.jr.resize(ASW_NB_ROWS);
    W.rs.resize(ASW_NB_RESP);
    W.vel.resize(ASW_VEL_POOL);
    W.dz.resize(size_t(ASW_NB_ARTI) * ASW_MAX_LINKS * ASW_NB_SLABS * 6);
    W.slab.resize(size_t(ASW_NB_SLABS) * ASW_NB_ARTI * 2);
    W.ab.resize(ASW_NB_AB_BLOCKS);
    W.lk.resize(size_t(ASW_NB_AB_BLOCKS) * ASW_MAX_LINKS);
    W.txi.resize(ASW_NB_RIGID);
    W.enc.resize(size_t(ASW_NUM32) * ASW_TOTAL_BODIES);
    W.islands.resize(2);
    W.acp.assign(8, 0);
    W.cpp.assign(8, 0);
    W.artiDesc.assign(ASW_DESC_SIZE, 0xAB);

    std::memset(W.bt.data(), 0xAB, W.bt.size() * sizeof(W.bt[0]));
    std::memset(W.ch.data(), 0xAB, W.ch.size() * sizeof(W.ch[0]));
    std::memset(W.fh.data(), 0xAB, W.fh.size() * sizeof(W.fh[0]));
    std::memset(W.cp.data(), 0xAB, W.cp.size() * sizeof(W.cp[0]));
    std::memset(W.fp.data(), 0xAB, W.fp.size() * sizeof(W.fp[0]));
    std::memset(W.jh.data(), 0xAB, W.jh.size() * sizeof(W.jh[0]));
    std::memset(W.jr.data(), 0xAB, W.jr.size() * sizeof(W.jr[0]));
    std::memset(W.rs.data(), 0xAB, W.rs.size() * sizeof(W.rs[0]));
    std::memset(W.vel.data(), 0xAB, W.vel.size() * sizeof(W.vel[0]));
    std::memset(W.dz.data(), 0xAB, W.dz.size() * 4);
    std::memset(W.slab.data(), 0xAB, W.slab.size() * 4);
    std::memset(W.ab.data(), 0xAB, W.ab.size() * sizeof(W.ab[0]));
    std::memset(W.lk.data(), 0xAB, W.lk.size() * sizeof(W.lk[0]));
    std::memset(W.txi.data(), 0xAB, W.txi.size() * sizeof(W.txi[0]));
    std::memset(W.islands.data(), 0xAB, W.islands.size() * sizeof(W.islands[0]));
    std::memset(&W.solverDesc, 0xAB, sizeof(W.solverDesc));

    // ---- islands: pool[0] carries the pointer-deref fields, pool[1] the island's ----
    W.islands[0].mBatchCount = ASW_BATCH_OFFSET;
    W.islands[0].mArticulationCount = ASW_NB_ARTI;
    W.islands[0].mBodyStartIndex = ASW_BODY_OFFSET;
    W.islands[0].mBodyCount = ASW_NB_DYN;
    // decoys on island[1] for those fields (a mutant reading pool[islandIndex] dies)
    W.islands[1].mBatchCount = ASW_BATCH_OFFSET + 5;
    W.islands[1].mArticulationCount = ASW_NB_ARTI + 7;
    W.islands[1].mBodyStartIndex = ASW_BODY_OFFSET + 2;
    W.islands[1].mBodyCount = ASW_NB_DYN + 3;
    W.islands[1].mStartPartitionIndex = 4;
    W.islands[1].mBatchStartIndex = 0;                       // rigid partition-0 start
    W.islands[1].mArtiBatchStartIndex = (partitionIndex == 0) ? 5 : 6;  // decoy when unused

    // arti partition table: live warps at k = 8, 9 (start 5, end 7 + offset 3)
    for (uint32_t i = 0; i < 8; ++i) W.acp[i] = 90u + i;     // decoys
    if (partitionIndex == 0)
        W.acp[4] = 7;                                        // endIndex arm
    else
    {
        W.acp[partitionIndex + 4 - 1] = 5;                   // startIndex arm
        W.acp[partitionIndex + 4] = 7;
    }
    for (uint32_t i = 0; i < 8; ++i) W.cpp[i] = 0;           // rigid partition EMPTY

    // ---- solver desc ----
    W.solverDesc.numSlabs = ASW_NB_SLABS;
    W.solverDesc.contactErrorAccumulator.mErrorSumOfSquares = 0.5f;
    W.solverDesc.contactErrorAccumulator.mCounter = errCounter;
    W.solverDesc.contactErrorAccumulator.mMaxError = 0.25f;

    // ---- arti desc image (scalars; pointers patched at upload) ----
    uint32_t maxLinks = ASW_MAX_LINKS;
    std::memcpy(&W.artiDesc[ASW_D_MAX_LINKS], &maxLinks, 4);

    // ---- reference counts ----
    for (uint32_t r = 0; r < ASW_NUM32; ++r)
        for (uint32_t bdy = 0; bdy < ASW_TOTAL_BODIES; ++bdy)
        {
            uint32_t h = (bdy * 2654435761u) ^ (r * 40503u) ^ (salt * 733u);
            h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
            W.enc[size_t(r) * ASW_TOTAL_BODIES + bdy] = h & 0x1FFFFu;
        }
    // a zero row inside the arti window pins the max(1, .) clamp
    W.enc[ASW_BODY_OFFSET + ASW_NB_DYN + 7] = 0u;

    // ---- txIData: deltaBody2World.q at offset 0 ----
    for (uint32_t i = 0; i < ASW_NB_RIGID; ++i)
    {
        PxQuat q(fv(i, 1, 0, salt), fv(i, 1, 1, salt), fv(i, 1, 2, salt),
                 1.0f + 0.4f * fv(i, 1, 3, salt));
        q.normalize();
        W.txi[i].deltaBody2World.q = q;
        W.txi[i].deltaBody2World.p = PxVec3(fv(i, 1, 4, salt), fv(i, 1, 5, salt), fv(i, 1, 6, salt));
    }

    // ---- link blocks: mDeltaQ for every (link, lane) ----
    for (uint32_t l = 0; l < W.lk.size(); ++l)
        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            PxQuat q(fv(l, 2, lane, salt), fv(l, 3, lane, salt), fv(l, 4, lane, salt),
                     1.0f + 0.3f * fv(l, 5, lane, salt));
            q.normalize();
            W.lk[l].mDeltaQ[lane] = make_float4(q.x, q.y, q.z, q.w);
        }

    // ---- the two live batches ----
    for (uint32_t b = 0; b < 2; ++b)
    {
        PxgBlockConstraintBatch& BT = W.bt[ASW_K0 + b];
        BT.mDescStride = PxU16(b == 0 ? 26 : 13);
        BT.constraintType = PxU16(b == 0 ? PxgSolverConstraintDesc::eARTICULATION_CONTACT
                                         : PxgSolverConstraintDesc::eARTICULATION_CONSTRAINT_1D);
        BT.mConstraintBatchIndex = b == 0 ? 4 : 7;
        BT.startConstraintIndex = b == 0 ? 5 : 20;
        BT.startFrictionIndex = b == 0 ? 8 : 40;
        BT.mArticulationResponseIndex = b == 0 ? 10 : 70;
        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            BT.slabId[lane] = (b == 0) ? lane % 3u : (lane + 1u) % 3u;
            const bool aA = aswIsArtiA(b, lane), aB = aswIsArtiB(b, lane);
            if (aA)
            {
                const uint32_t id = aswArtiIdA(b, lane);
                BT.bodyANodeIndex[lane] = PxNodeIndex(600 + id, 1u + (lane % (ASW_MAX_LINKS - 1u)));
                BT.bodyAIndex[lane] = id;
                BT.remappedBodyAIndex[lane] = (id + 5u) % ASW_NB_ARTI;   // refcount row, != id
            }
            else
            {
                // rigids: a mix of static (< bodyOffset) and dynamic ids
                const uint32_t rid = (lane % 5u == 0u) ? (lane % ASW_BODY_OFFSET)
                                                       : ASW_BODY_OFFSET + (lane * 3u + b) % ASW_NB_DYN;
                BT.bodyANodeIndex[lane] = PxNodeIndex(300 + b * 40 + lane);
                BT.bodyAIndex[lane] = rid;
                BT.remappedBodyAIndex[lane] = ASW_OUT_BASE + (b * 64u + lane * 2u + 0u) * 96u;
            }
            if (aB)
            {
                const uint32_t id = aswArtiIdB(b, lane);
                BT.bodyBNodeIndex[lane] = PxNodeIndex(700 + id, 1u + ((lane + 3u) % (ASW_MAX_LINKS - 1u)));
                BT.bodyBIndex[lane] = id;
                BT.remappedBodyBIndex[lane] = (id + 11u) % ASW_NB_ARTI;
            }
            else
            {
                const uint32_t rid = (lane % 7u == 2u) ? ((lane + 1u) % ASW_BODY_OFFSET)
                                                       : ASW_BODY_OFFSET + (lane * 5u + b + 1u) % ASW_NB_DYN;
                BT.bodyBNodeIndex[lane] = PxNodeIndex(400 + b * 40 + lane);
                BT.bodyBIndex[lane] = rid;
                BT.remappedBodyBIndex[lane] = ASW_OUT_BASE + (b * 64u + lane * 2u + 1u) * 96u;
            }
        }

        // staging slots for k = 8+b
        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            const uint32_t ri = (ASW_K0 + b) * 192u + lane;
            for (uint32_t s = 0; s < 6; ++s)
                W.vel[ri + s * 32u] = make_float4(0.5f * fv(lane, b * 8 + s, 20, salt),
                                                  0.5f * fv(lane, b * 8 + s, 21, salt),
                                                  0.5f * fv(lane, b * 8 + s, 22, salt),
                                                  0.5f * fv(lane, b * 8 + s, 23, salt));
        }
    }

    // ---- contact solve rows (batch 8, desc 4) ----
    {
        const uint32_t bi = 4;
        PxgTGSBlockSolverContactHeader& H = W.ch[bi];
        PxgTGSBlockSolverFrictionHeader& F = W.fh[bi];
        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            H.invMass0_1_angDom0_1[lane] = make_float4(0.6f + 0.2f * fv(lane, bi, 60, salt),
                                                       0.5f + 0.2f * fv(lane, bi, 61, salt),
                                                       0.7f + 0.2f * fv(lane, bi, 62, salt),
                                                       0.8f + 0.2f * fv(lane, bi, 63, salt));
            H.normal_staticFriction[lane] = make_float4(fv(lane, bi, 64, salt), fv(lane, bi, 65, salt),
                                                        fv(lane, bi, 66, salt),
                                                        0.6f + 0.3f * fv(lane, bi, 67, salt));
            H.minNormalForce[lane] = 0.05f + 0.05f * std::fabs(fv(lane, bi, 68, salt));
            H.maxPenBias[lane] = -0.4f + 0.1f * fv(lane, bi, 69, salt);
            H.flags[lane] = ((lane) % 3u == 1u) ? 8u : 0u;   // compliant-accel bit
            H.numNormalConstr[lane] = 1u + (lane % 3u);
            H.forceWritebackOffset[lane] = 0;
            H.restitutionXdt[lane] = (lane % 4u == 1u) ? (-0.3f + 0.1f * fv(lane, bi, 70, salt))
                                                       : (0.2f + 0.1f * std::fabs(fv(lane, bi, 71, salt)));
            H.cfm[lane] = 0.01f + 0.005f * std::fabs(fv(lane, bi, 72, salt));
            H.p8[lane] = 0.9f;
            F.numFrictionConstr[lane] = (lane % 3u) * 2u + ((lane % 5u == 2u) ? 1u : 0u);
            F.dynamicFriction[lane] = 0.4f + 0.2f * std::fabs(fv(lane, bi, 73, salt));
            F.biasCoefficient[lane] = 0.5f + 0.2f * std::fabs(fv(lane, bi, 74, salt));
            F.torsionalFrictionScale[lane] = 0.8f + 0.1f * fv(lane, bi, 75, salt);
            F.frictionNormals[0][lane] = make_float4(fv(lane, bi, 76, salt), fv(lane, bi, 77, salt),
                                                     fv(lane, bi, 78, salt), 0.f);
            F.frictionNormals[1][lane] = make_float4(fv(lane, bi, 79, salt), fv(lane, bi, 80, salt),
                                                     fv(lane, bi, 81, salt), 0.f);
            F.broken[lane] = 0;
            for (uint32_t i = 0; i < 3; ++i)
            {
                PxgTGSBlockSolverContactPoint& C = W.cp[5 + i];
                C.raXn_extraCoeff[lane] = make_float4(fv(lane, i, 82, salt), fv(lane, i, 83, salt),
                                                      fv(lane, i, 84, salt),
                                                      0.5f + 0.3f * std::fabs(fv(lane, i, 85, salt)));
                C.rbXn_targetVelW[lane] = make_float4(fv(lane, i, 86, salt), fv(lane, i, 87, salt),
                                                      fv(lane, i, 88, salt), 0.1f * fv(lane, i, 89, salt));
                C.separation[lane] = 0.05f * fv(lane, i, 90, salt);
                C.maxImpulse[lane] = 3.0f + std::fabs(fv(lane, i, 91, salt));
                C.appliedForce[lane] = 0.3f + 0.2f * std::fabs(fv(lane, i, 92, salt));
                C.biasCoefficient[lane] = 0.4f + 0.2f * fv(lane, i, 93, salt);
                C.resp0[lane] = 0.5f + 0.3f * std::fabs(fv(lane, i, 94, salt));
                C.resp1[lane] = 0.5f + 0.3f * std::fabs(fv(lane, i, 95, salt));
            }
            for (uint32_t i = 0; i < 7; ++i)
            {
                PxgTGSBlockSolverContactFriction& FR = W.fp[8 + i];
                FR.raXn_error[lane] = make_float4(fv(lane, i, 96, salt), fv(lane, i, 97, salt),
                                                  fv(lane, i, 98, salt), 0.05f * fv(lane, i, 99, salt));
                FR.rbXn_targetVelW[lane] = make_float4(fv(lane, i, 100, salt), fv(lane, i, 101, salt),
                                                       fv(lane, i, 102, salt), 0.1f * fv(lane, i, 103, salt));
                FR.appliedForce[lane] = 0.2f + 0.1f * std::fabs(fv(lane, i, 104, salt));
                FR.resp0[lane] = 0.4f + 0.3f * std::fabs(fv(lane, i, 105, salt));
                FR.resp1[lane] = 0.4f + 0.3f * std::fabs(fv(lane, i, 106, salt));
            }
        }
    }

    // ---- 1D solve rows (batch 9, desc 7) ----
    {
        const uint32_t bi = 7;
        PxgTGSBlockSolverConstraint1DHeader& H = W.jh[bi];
        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            const uint32_t rc = 1u + (lane % 4u);
            uchar4 packed;
            packed.x = PxU8(rc); packed.y = PxU8((lane & 1u) != 0u); packed.z = 0; packed.w = 0;
            H.rowCounts_breakable_orthoAxisCount[lane] = packed;
            H.rAWorld_invMass0D0[lane] = make_float4(fv(lane, bi, 110, salt), fv(lane, bi, 111, salt),
                                                     fv(lane, bi, 112, salt),
                                                     0.5f + 0.2f * fv(lane, bi, 113, salt));
            H.rBWorld_invMass1D1[lane] = make_float4(fv(lane, bi, 114, salt), fv(lane, bi, 115, salt),
                                                     fv(lane, bi, 116, salt),
                                                     0.5f + 0.2f * fv(lane, bi, 117, salt));
            H.invInertiaScale0[lane] = -0.8f - 0.2f * std::fabs(fv(lane, bi, 118, salt));
            H.invInertiaScale1[lane] = 0.8f + 0.2f * std::fabs(fv(lane, bi, 119, salt));
            H.writeBackOffset[lane] = 0;
            H.linBreakImpulse[lane] = 50.0f;
            H.angBreakImpulse[lane] = 60.0f;
            H.cfm[lane] = 0.02f;
            for (uint32_t r = 0; r < 4; ++r)
            {
                PxgTGSBlockSolverConstraint1DCon& C = W.jr[20 + r];
                C.lin0XYZ_initBiasOrCoeff0[lane] =
                    make_float4(fv(lane, r, 120, salt), fv(lane, r, 121, salt), fv(lane, r, 122, salt),
                                0.1f * fv(lane, r, 123, salt));
                C.lin1XYZ_biasScaleOrCoeff1[lane] =
                    make_float4(fv(lane, r, 124, salt), fv(lane, r, 125, salt), fv(lane, r, 126, salt),
                                -0.5f * std::fabs(fv(lane, r, 127, salt)));
                // coeff2: for NON-spring rows this is the bounce velocity, and a
                // nonzero bounce forces biasScale = 0 (killing the deltaQ error
                // term) -- row 0 carries NO bounce so the geometric error stays live
                C.ang0XYZ_velMultiplierOrCoeff2[lane] =
                    make_float4(fv(lane, r, 128, salt), fv(lane, r, 129, salt), fv(lane, r, 130, salt),
                                (r == 0u) ? 0.0f : 0.3f * std::fabs(fv(lane, r, 131, salt)));
                C.ang1XYZ_velTargetOrCoeff3[lane] =
                    make_float4(fv(lane, r, 132, salt), fv(lane, r, 133, salt), fv(lane, r, 134, salt),
                                0.2f * fv(lane, r, 135, salt));
                C.geometricError[lane] = 0.04f * fv(lane, r, 136, salt);
                C.maxBias[lane] = 5.0f + std::fabs(fv(lane, r, 137, salt));
                C.appliedForce[lane] = 0.1f + 0.1f * std::fabs(fv(lane, r, 138, salt));
                C.angularErrorScale[lane] = (r % 2u) ? 1.0f : 0.0f;
                C.minImpulse[lane] = -30.0f - std::fabs(fv(lane, r, 139, salt));
                C.maxImpulse[lane] = 30.0f + std::fabs(fv(lane, r, 140, salt));
                C.resp0[lane] = 0.5f + 0.3f * std::fabs(fv(lane, r, 141, salt));
                C.resp1[lane] = 0.5f + 0.3f * std::fabs(fv(lane, r, 142, salt));
                C.flags[lane] = (r == 1u) ? 32u : ((r == 3u) ? (32u | 128u) : 0u);   // SPRING / ACCEL_SPRING
                C.residual[lane] = 0.0f;
            }
        }
    }

    // responses read by both solves
    for (uint32_t r = 0; r < ASW_NB_RESP; ++r)
        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            float* f = &W.rs[r].deltaRALin_x[lane];
            for (uint32_t s = 0; s < 12; ++s)
                (&W.rs[r].deltaRALin_x[0])[s * 32 + lane] = 0.25f * fv(lane, r, 150 + s, salt);
            (void)f;
        }
}

static uint32_t aswCountSlabs(const AswArena& W, uint32_t bodyIndex)
{
    uint32_t c = 0;
    for (uint32_t i = 0; i < ASW_NUM32; ++i)
        c += PxU32(__builtin_popcount(W.enc[size_t(i) * ASW_TOTAL_BODIES + bodyIndex]));
    return PxMax(1u, c);
}

// the full host mirror of the articulation branch
static void aswMirror(AswArena& W, uint32_t partitionIndex, float elapsedTime, float minPen,
                      bool residual, uint32_t& artiWrites, uint32_t& rigidWrites,
                      uint32_t& staticRefs)
{
    RefError warpErrLanes[2][32];
    for (uint32_t w = 0; w < 4; ++w)   // 2 blocks x 2 warps
    {
        const uint32_t startIndex = (partitionIndex == 0)
            ? W.islands[1].mArtiBatchStartIndex
            : W.acp[partitionIndex + 4 - 1];
        const uint32_t endIndex = W.acp[partitionIndex + 4] + ASW_BATCH_OFFSET;
        const uint32_t k = startIndex + w + ASW_BATCH_OFFSET;
        if (k >= endIndex)
            continue;
        const uint32_t b = k - ASW_K0;
        PxgBlockConstraintBatch& BT = W.bt[k];

        for (uint32_t lane = 0; lane < BT.mDescStride; ++lane)
        {
            const uint32_t readIndex = k * 192u + lane;
            UnAlignedSpatialVector vel0, vel1, delta0, delta1;
            {
                const float4 v0 = W.vel[readIndex], v1 = W.vel[readIndex + 32],
                             v2 = W.vel[readIndex + 64];
                vel0 = UnAlignedSpatialVector(PxVec3(v0.w, v1.x, v1.y), PxVec3(v0.x, v0.y, v0.z));
                delta0 = UnAlignedSpatialVector(PxVec3(v2.y, v2.z, v2.w), PxVec3(v1.z, v1.w, v2.x));
            }
            {
                const float4 v0 = W.vel[readIndex + 96], v1 = W.vel[readIndex + 128],
                             v2 = W.vel[readIndex + 160];
                vel1 = UnAlignedSpatialVector(PxVec3(v0.w, v1.x, v1.y), PxVec3(v0.x, v0.y, v0.z));
                delta1 = UnAlignedSpatialVector(PxVec3(v2.y, v2.z, v2.w), PxVec3(v1.z, v1.w, v2.x));
            }

            const PxNodeIndex nA = BT.bodyANodeIndex[lane];
            const PxNodeIndex nB = BT.bodyBNodeIndex[lane];
            const uint32_t id0 = BT.bodyAIndex[lane];
            const uint32_t id1 = BT.bodyBIndex[lane];
            const uint32_t slabId = BT.slabId[lane];

            float ref0 = 1.f, ref1 = 1.f;
            if (nA.isArticulation())
                ref0 = float(aswCountSlabs(W, BT.remappedBodyAIndex[lane] + ASW_NB_DYN + ASW_BODY_OFFSET));
            else if (id0 >= ASW_BODY_OFFSET)
                ref0 = float(aswCountSlabs(W, id0));
            else
                staticRefs++;
            if (nB.isArticulation())
                ref1 = float(aswCountSlabs(W, BT.remappedBodyBIndex[lane] + ASW_NB_DYN + ASW_BODY_OFFSET));
            else if (id1 >= ASW_BODY_OFFSET)
                ref1 = float(aswCountSlabs(W, id1));
            else
                staticRefs++;

            PxQuat dq0(PxIdentity), dq1(PxIdentity);
            if (BT.constraintType == PxgSolverConstraintDesc::eARTICULATION_CONSTRAINT_1D)
            {
                if (nA.isArticulation())
                {
                    const uint32_t linkID = (id0 / 32u) * ASW_MAX_LINKS + nA.articulationLinkId();
                    const float4 q = W.lk[linkID].mDeltaQ[id0 & 31u];
                    dq0 = PxQuat(q.x, q.y, q.z, q.w);
                }
                else
                    dq0 = W.txi[id0].deltaBody2World.q;
                if (nB.isArticulation())
                {
                    const uint32_t linkID = (id1 / 32u) * ASW_MAX_LINKS + nB.articulationLinkId();
                    const float4 q = W.lk[linkID].mDeltaQ[id1 & 31u];
                    dq1 = PxQuat(q.x, q.y, q.z, q.w);
                }
                else
                    dq1 = W.txi[id1].deltaBody2World.q;
            }

            UnAlignedSpatialVector imp0(PxVec3(0.f), PxVec3(0.f)), imp1(PxVec3(0.f), PxVec3(0.f));
            if (BT.constraintType == PxgSolverConstraintDesc::eARTICULATION_CONTACT)
                refSolveExtContactTGS(BT, vel0, vel1, delta0, delta1, lane, W.ch.data(),
                                      W.fh.data(), W.cp.data(), W.fp.data(), W.rs.data(),
                                      elapsedTime, minPen, imp0, imp1,
                                      residual ? &warpErrLanes[b][lane] : nullptr, ref0, ref1);
            else
                refSolveExt1DTGS(BT, vel0, vel1, delta0, delta1, lane, W.jh.data(), W.jr.data(),
                                 W.rs.data(), dq0, dq1, elapsedTime, imp0, imp1, residual,
                                 ref0, ref1);

            if (!nA.isArticulation())
            {
                const uint32_t o = BT.remappedBodyAIndex[lane];
                W.vel[o] = make_float4(vel0.bottom.x, vel0.bottom.y, vel0.bottom.z, vel0.top.x);
                W.vel[o + 32] = make_float4(vel0.top.y, vel0.top.z, delta0.bottom.x, delta0.bottom.y);
                W.vel[o + 64] = make_float4(delta0.bottom.z, delta0.top.x, delta0.top.y, delta0.top.z);
                rigidWrites++;
            }
            else
            {
                const uint32_t idx = id0 + nA.articulationLinkId() * ASW_NB_ARTI
                                   + slabId * ASW_NB_ARTI * ASW_MAX_LINKS;
                float* z = &W.dz[size_t(idx) * 6];
                z[0] = imp0.top.x; z[1] = imp0.top.y; z[2] = imp0.top.z;
                z[3] = imp0.bottom.x; z[4] = imp0.bottom.y; z[5] = imp0.bottom.z;
                W.slab[(size_t(id0) + size_t(slabId) * ASW_NB_ARTI) * 2 + 0] = nA.articulationLinkId();
                W.ab[id0 / 32u].mStateDirty[id0 & 31u] = 3;
                artiWrites++;
            }
            if (!nB.isArticulation())
            {
                const uint32_t o = BT.remappedBodyBIndex[lane];
                W.vel[o] = make_float4(vel1.bottom.x, vel1.bottom.y, vel1.bottom.z, vel1.top.x);
                W.vel[o + 32] = make_float4(vel1.top.y, vel1.top.z, delta1.bottom.x, delta1.bottom.y);
                W.vel[o + 64] = make_float4(delta1.bottom.z, delta1.top.x, delta1.top.y, delta1.top.z);
                rigidWrites++;
            }
            else
            {
                const uint32_t idx = id1 + nB.articulationLinkId() * ASW_NB_ARTI
                                   + slabId * ASW_NB_ARTI * ASW_MAX_LINKS;
                float* z = &W.dz[size_t(idx) * 6];
                z[0] = imp1.top.x; z[1] = imp1.top.y; z[2] = imp1.top.z;
                z[3] = imp1.bottom.x; z[4] = imp1.bottom.y; z[5] = imp1.bottom.z;
                W.slab[(size_t(id1) + size_t(slabId) * ASW_NB_ARTI) * 2 + 1] = nB.articulationLinkId();
                W.ab[id1 / 32u].mStateDirty[id1 & 31u] = 3;
                artiWrites++;
            }
        }
    }

    // the global accumulator: the device reduces each warp's 32 lanes with a
    // BUTTERFLY (xor-partner tree), then lane 0 accumulates atomically -- the
    // mirror must reduce in the same tree order, a linear lane sum is 1 ULP off
    if (residual)
        for (uint32_t b = 0; b < 2; ++b)
        {
            float sums[32], maxs[32];
            uint32_t counts[32];
            for (uint32_t l = 0; l < 32; ++l)
            {
                sums[l] = warpErrLanes[b][l].sumOfSquares;
                maxs[l] = warpErrLanes[b][l].maxError;
                counts[l] = uint32_t(warpErrLanes[b][l].counter);
            }
            for (uint32_t radius = 16; radius > 0; radius >>= 1)
            {
                float ns[32], nm[32];
                uint32_t nc[32];
                for (uint32_t l = 0; l < 32; ++l)
                {
                    ns[l] = sums[l] + sums[l ^ radius];
                    nm[l] = PxMax(maxs[l], maxs[l ^ radius]);
                    nc[l] = counts[l] + counts[l ^ radius];
                }
                std::memcpy(sums, ns, sizeof(ns));
                std::memcpy(maxs, nm, sizeof(nm));
                std::memcpy(counts, nc, sizeof(nc));
            }
            W.solverDesc.contactErrorAccumulator.mErrorSumOfSquares += sums[0];
            W.solverDesc.contactErrorAccumulator.mCounter += int(counts[0]);
            W.solverDesc.contactErrorAccumulator.mMaxError =
                PxMax(W.solverDesc.contactErrorAccumulator.mMaxError, maxs[0]);
        }
}

} // namespace

PXG_TEST(artiSolvePartitionWrap)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "solverTGS.spv", "solveBlockUnified", 48u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &aswLoader);

    struct Round { uint32_t partition; int32_t errCounter; float elapsed, minPen; };
    const Round rounds[2] = {
        { 2u, -5, 0.13f, -PX_MAX_F32 },   // disabled accumulator, [idx-1] start arm
        { 0u, 0, 0.29f, 0.0f },           // enabled accumulator, mArtiBatchStartIndex arm
    };

    for (uint32_t round = 0; round < 2; ++round)
    {
        const Round& rd = rounds[round];
        AswArena A;
        aswSeed(A, 17u + round * 5u, rd.partition, rd.errCounter);

        // ---- upload ----
        std::vector<PxgDevicePtr> bufs;
        auto up = [&](const void* p, size_t nsz) {
            PxgDevicePtr dp = 0;
            shim.memAlloc(&dp, nsz);
            if (dp && p) shim.memcpyHtoD(dp, p, nsz);
            bufs.push_back(dp);
            return dp;
        };
        PxgDevicePtr dBt = up(A.bt.data(), A.bt.size() * sizeof(A.bt[0]));
        PxgDevicePtr dCh = up(A.ch.data(), A.ch.size() * sizeof(A.ch[0]));
        PxgDevicePtr dFh = up(A.fh.data(), A.fh.size() * sizeof(A.fh[0]));
        PxgDevicePtr dCp = up(A.cp.data(), A.cp.size() * sizeof(A.cp[0]));
        PxgDevicePtr dFp = up(A.fp.data(), A.fp.size() * sizeof(A.fp[0]));
        PxgDevicePtr dJh = up(A.jh.data(), A.jh.size() * sizeof(A.jh[0]));
        PxgDevicePtr dJr = up(A.jr.data(), A.jr.size() * sizeof(A.jr[0]));
        PxgDevicePtr dRs = up(A.rs.data(), A.rs.size() * sizeof(A.rs[0]));
        PxgDevicePtr dVel = up(A.vel.data(), A.vel.size() * sizeof(A.vel[0]));
        PxgDevicePtr dDz = up(A.dz.data(), A.dz.size() * 4);
        PxgDevicePtr dSlab = up(A.slab.data(), A.slab.size() * 4);
        PxgDevicePtr dAb = up(A.ab.data(), A.ab.size() * sizeof(A.ab[0]));
        PxgDevicePtr dLk = up(A.lk.data(), A.lk.size() * sizeof(A.lk[0]));
        PxgDevicePtr dTxi = up(A.txi.data(), A.txi.size() * sizeof(A.txi[0]));
        PxgDevicePtr dEnc = up(A.enc.data(), A.enc.size() * 4);
        PxgDevicePtr dIsl = up(A.islands.data(), A.islands.size() * sizeof(A.islands[0]));
        PxgDevicePtr dAcp = up(A.acp.data(), A.acp.size() * 4);
        PxgDevicePtr dCpp = up(A.cpp.data(), A.cpp.size() * 4);

        A.solverDesc.islandContextPool = reinterpret_cast<PxgIslandContext*>(dIsl);
        A.solverDesc.artiConstraintsPerPartition = reinterpret_cast<PxU32*>(dAcp);
        A.solverDesc.constraintsPerPartition = reinterpret_cast<PxU32*>(dCpp);
        A.solverDesc.solverBodyTxIDataPool = reinterpret_cast<PxgSolverTxIData*>(dTxi);
        PxgDevicePtr dSd = up(&A.solverDesc, sizeof(A.solverDesc));

        // shared desc image (memset poison; only the used fields live)
        PxgSolverSharedDesc<IterativeSolveDataTGS> sh;
        std::memset(&sh, 0xAB, sizeof(sh));
        sh.articulationDeferredZ = reinterpret_cast<Cm::UnAlignedSpatialVector*>(dDz);
        sh.iterativeData.blockConstraintBatch = reinterpret_cast<PxgBlockConstraintBatch*>(dBt);
        sh.iterativeData.blockContactHeaders = reinterpret_cast<PxgTGSBlockSolverContactHeader*>(dCh);
        sh.iterativeData.blockFrictionHeaders =
            reinterpret_cast<PxgTGSBlockSolverFrictionHeader*>(dFh);
        sh.iterativeData.blockContactPoints = reinterpret_cast<PxgTGSBlockSolverContactPoint*>(dCp);
        sh.iterativeData.blockFrictions = reinterpret_cast<PxgTGSBlockSolverContactFriction*>(dFp);
        sh.iterativeData.blockJointConstraintHeaders =
            reinterpret_cast<PxgTGSBlockSolverConstraint1DHeader*>(dJh);
        sh.iterativeData.blockJointConstraintRowsCon =
            reinterpret_cast<PxgTGSBlockSolverConstraint1DCon*>(dJr);
        sh.iterativeData.artiResponse = reinterpret_cast<PxgArticulationBlockResponse*>(dRs);
        sh.iterativeData.solverBodyVelPool = reinterpret_cast<float4*>(dVel);
        sh.iterativeData.solverEncodedReferenceCount = reinterpret_cast<PxU32*>(dEnc);
        PxgDevicePtr dSh = up(&sh, sizeof(sh));

        // arti desc image with device pointers patched
        {
            uint64_t v;
            v = uint64_t(dSlab); std::memcpy(&A.artiDesc[ASW_D_SLAB_HAS_CHANGES], &v, 8);
            v = uint64_t(dAb); std::memcpy(&A.artiDesc[ASW_D_ARTICULATION_BLOCKS], &v, 8);
            v = uint64_t(dLk); std::memcpy(&A.artiDesc[ASW_D_LINK_BLOCKS], &v, 8);
        }
        PxgDevicePtr dAd = up(A.artiDesc.data(), A.artiDesc.size());

        bool okAlloc = true;
        for (PxgDevicePtr p : bufs) okAlloc = okAlloc && (p != 0);
        if (!okAlloc)
        {
            std::printf("  NOTE: no BDA -> skip\n");
            for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
            return;
        }

        // ---- mirror (mutates A in place) ----
        uint32_t artiWrites = 0, rigidWrites = 0, staticRefs = 0;
        aswMirror(A, rd.partition, rd.elapsed, rd.minPen, rd.errCounter >= 0,
                  artiWrites, rigidWrites, staticRefs);
        PXG_CHECK_MSG(artiWrites >= 20 && rigidWrites >= 15 && staticRefs >= 2,
                      "coverage: arti %u rigid %u static %u", artiWrites, rigidWrites, staticRefs);

        // ---- launch: grid (2, 2, 1) ----
        struct Params
        {
            uint64_t solverCoreDesc, sharedDesc;
            uint32_t islandIndex, partitionIndex, doFrictionWord;
            float elapsedTime, minPen;
            uint32_t pad0;
            uint64_t artiDesc;
        } prm = { uint64_t(dSd), uint64_t(dSh), 1u, rd.partition, 1u,
                  rd.elapsed, rd.minPen, 0u, uint64_t(dAd) };
        static_assert(sizeof(Params) == 48, "params blob");
        PXG_CHECK_MSG(shim.launchKernel(0, 2, 2, 1, &prm, sizeof(prm)) == PxgResult::Success,
                      "launch solveBlockUnified");
        shim.streamSynchronize();

        // ---- compare ----
        size_t bad = 0, signedZeros = 0;
        auto cmp = [&](PxgDevicePtr dp, const void* wantp, size_t nsz, const char* tag) {
            std::vector<uint32_t> got(nsz / 4);
            shim.memcpyDtoH(got.data(), dp, nsz);
            const uint32_t* wv = reinterpret_cast<const uint32_t*>(wantp);
            for (size_t k = 0; k < got.size(); ++k)
                if (wv[k] != got[k])
                {
                    if ((wv[k] & 0x7FFFFFFFu) == 0u && (got[k] & 0x7FFFFFFFu) == 0u)
                    {
                        ++signedZeros;
                        continue;
                    }
                    if (bad < 12)
                        std::printf("MM asw r%u %s word %zu (lane %zu)  ref %08x  gpu %08x\n",
                                    round, tag, k, k % 32, wv[k], got[k]);
                    ++bad;
                }
        };
        cmp(dVel, A.vel.data(), A.vel.size() * sizeof(A.vel[0]), "velPool");
        cmp(dDz, A.dz.data(), A.dz.size() * 4, "deferredZ");
        cmp(dSlab, A.slab.data(), A.slab.size() * 4, "slabDirty");
        cmp(dAb, A.ab.data(), A.ab.size() * sizeof(A.ab[0]), "artiBlocks");
        cmp(dCh, A.ch.data(), A.ch.size() * sizeof(A.ch[0]), "contactHeaders");
        cmp(dFh, A.fh.data(), A.fh.size() * sizeof(A.fh[0]), "frictionHeaders");
        cmp(dCp, A.cp.data(), A.cp.size() * sizeof(A.cp[0]), "contactPoints");
        cmp(dFp, A.fp.data(), A.fp.size() * sizeof(A.fp[0]), "frictions");
        cmp(dJh, A.jh.data(), A.jh.size() * sizeof(A.jh[0]), "jointHeaders");
        cmp(dJr, A.jr.data(), A.jr.size() * sizeof(A.jr[0]), "jointRows");
        cmp(dRs, A.rs.data(), A.rs.size() * sizeof(A.rs[0]), "responses");
        cmp(dEnc, A.enc.data(), A.enc.size() * 4, "refCounts");
        // the accumulator words in the downloaded solver desc
        {
            std::vector<uint8_t> gotSd(sizeof(A.solverDesc));
            shim.memcpyDtoH(gotSd.data(), dSd, gotSd.size());
            const size_t accOff = offsetof(PxgSolverCoreDesc, contactErrorAccumulator);
            if (std::memcmp(&gotSd[accOff], &A.solverDesc.contactErrorAccumulator,
                            sizeof(Dy::ErrorAccumulator)) != 0)
            {
                float gs, gm; int gc;
                std::memcpy(&gs, &gotSd[accOff], 4);
                std::memcpy(&gc, &gotSd[accOff + 4], 4);
                std::memcpy(&gm, &gotSd[accOff + 8], 4);
                std::printf("MM asw r%u errorAccumulator ref (%a,%d,%a) gpu (%a,%d,%a)\n", round,
                            double(A.solverDesc.contactErrorAccumulator.mErrorSumOfSquares),
                            A.solverDesc.contactErrorAccumulator.mCounter,
                            double(A.solverDesc.contactErrorAccumulator.mMaxError), gs, gc, gm);
                ++bad;
            }
        }
        std::printf("  asw r%u +-0 tolerated %zu\n", round, signedZeros);
        PXG_CHECK_MSG(bad == 0, "round %u: %zu mismatches", round, bad);

        for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
    }
}
