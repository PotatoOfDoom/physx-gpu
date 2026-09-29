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

// testArtiSolvePartitionPgs.cpp -- P8.122b: the ENGINE entry
// artiSolveBlockPartition (solver.cu:47-266), the PGS twin of P8.104's
// solveBlockUnified arti branch, driven over REAL PxgSolverCoreDesc /
// PxgSolverSharedDesc<IterativeSolveData> / PxgArticulationCoreDesc images.
// Mirror: refSolveExtContactPGS / refSolveExt1DPGS (P8.122a, over the
// engine structs) + the countActiveSlabs popcount + the 2-slot staging /
// writeback transcription + the butterfly error reduction.
//
// Geometry (the P8.104 fixture's, PGS-adjusted): 2 live arti batches at
// k=8 (contact, stride 26) and k=9 (1D, stride 13) behind
// articulationBatchOffset=3 (islandContextPool[0].mBatchCount -- the
// POINTER-deref trap: island[islandIndex=1] carries DECOYS for the fields
// the kernel must read from pool[0]); grid 2 x block (32,2) = 4 warps, so
// warps 2..3 pin the k >= endIndex guard.  Lanes mix arti/rigid pairs on
// both sides; (artiId, slab) pairs are globally unique per side so the
// deferredZ / slabDirty scatters never collide.  The PGS staging is the
// 2-slot 128-stride packing (lin/ang whole halves), the rigid writeback the
// 2-slot (bottom,0)/(top,0) pair -- both DIFFERENT from the TGS 3-slot
// packing, each a trap.  PGS reads NO deltaQ / txIData / elapsed / minPen.
// Rounds: A partition 2 (the [idx-1] start arm, residual DISABLED ->
// the accumulator words must come back untouched), B partition 0 (the
// mArtiBatchStartIndex arm, residual enabled -> butterfly + atomic tail),
// C partition 2 with doFriction=0 (the friction pools of the contact batch
// must come back with only appliedForce untouched -- no broken write), and
// (P9.89) six more residual-enabled rounds on fresh salts, both start arms and
// both friction arms, so the warp butterfly is proven against the reversed
// radius order and the linear lane sum (host twins, checked at the end).

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "foundation/PxVec3.h"
#include "foundation/PxQuat.h"
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

static std::vector<uint8_t> appLoader(const char* file) { return loadShaderBlob(file); }

const uint32_t APP_NB_ARTI = 48;
const uint32_t APP_MAX_LINKS = 9;
const uint32_t APP_NB_SLABS = 3;
const uint32_t APP_NUM32 = 1;              // ceil(numSlabs/32)
const uint32_t APP_BODY_OFFSET = 4;
const uint32_t APP_NB_DYN = 25;
const uint32_t APP_TOTAL_BODIES = APP_BODY_OFFSET + APP_NB_DYN + APP_NB_ARTI;   // 77
const uint32_t APP_BATCH_OFFSET = 3;       // islandContextPool[0].mBatchCount
const uint32_t APP_K0 = 8;                 // first live batch slot
const uint32_t APP_NB_BATCH_ALLOC = 12;
const uint32_t APP_NB_AB_BLOCKS = 2;       // ceil(48/32)
const uint32_t APP_VEL_POOL = 12000;       // float4 slots (staging + rigid out region)
const uint32_t APP_OUT_BASE = 3000;
const uint32_t APP_NB_HDRS = 12;
const uint32_t APP_NB_CPTS = 60;
const uint32_t APP_NB_FPTS = 70;
const uint32_t APP_NB_ROWS = 60;
const uint32_t APP_NB_RESP = 120;

static_assert(sizeof(PxgBlockConstraintBatch) == 1536, "batch stride");
static_assert(sizeof(PxgArticulationBlockData) == 7168, "arti block stride");

static float fv(uint32_t a, uint32_t b, uint32_t c, uint32_t salt)
{
    uint32_t h = a * 2654435761u ^ (b * 40503u + c * 9176u) ^ (salt * 733u);
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    return (float(h % 479u) + 1.0f) * 0.0078125f - 1.85f;
}

// lane roles per batch
static bool appIsArtiA(uint32_t b, uint32_t lane)
{
    return (b == 0) ? (lane % 3u != 1u) : (lane % 4u == 1u);
}
static bool appIsArtiB(uint32_t b, uint32_t lane)
{
    return (b == 0) ? (lane % 3u != 0u) : (lane % 4u == 3u);
}
static uint32_t appArtiIdA(uint32_t b, uint32_t lane)
{
    return (b == 0) ? lane : 40u + lane / 4u;
}
static uint32_t appArtiIdB(uint32_t b, uint32_t lane)
{
    return (b == 0) ? 26u + lane / 3u : 44u + lane / 4u;
}

struct AppArena
{
    std::vector<PxgBlockConstraintBatch> bt;
    std::vector<PxgBlockSolverContactHeader> ch;
    std::vector<PxgBlockSolverFrictionHeader> fh;
    std::vector<PxgBlockSolverContactPoint> cp;
    std::vector<PxgBlockSolverContactFriction> fp;
    std::vector<PxgBlockSolverConstraint1DHeader> jh;
    std::vector<PxgBlockSolverConstraint1DCon> jr;
    std::vector<PxgBlockSolverConstraint1DMod> jm;
    std::vector<PxgArticulationBlockResponse> rs;
    std::vector<float4> vel;
    std::vector<float> dz;                 // deferredZ, 6 floats per slot
    std::vector<uint32_t> slab;            // uint2 per (slab, arti)
    std::vector<PxgArticulationBlockData> ab;
    std::vector<uint32_t> enc;
    std::vector<PxgIslandContext> islands;
    std::vector<uint32_t> acp;             // arti constraintsPerPartition
    PxgSolverCoreDesc solverDesc;
    std::vector<uint8_t> artiDesc;         // 360-byte PxgArticulationCoreDesc image
};

// PXG_ARTI_CORE_* offsets, pinned by testDmaArtiResidual
const uint32_t APP_D_SLAB_HAS_CHANGES = 88;
const uint32_t APP_D_ARTICULATION_BLOCKS = 128;
const uint32_t APP_D_MAX_LINKS = 248;
const uint32_t APP_DESC_SIZE = 360;

static void appSeed(AppArena& W, uint32_t salt, uint32_t partitionIndex, int32_t errCounter)
{
    W.bt.assign(APP_NB_BATCH_ALLOC, PxgBlockConstraintBatch());
    W.ch.resize(APP_NB_HDRS);
    W.fh.resize(APP_NB_HDRS);
    W.cp.resize(APP_NB_CPTS);
    W.fp.resize(APP_NB_FPTS);
    W.jh.resize(APP_NB_HDRS);
    W.jr.resize(APP_NB_ROWS);
    W.jm.resize(APP_NB_ROWS);
    W.rs.resize(APP_NB_RESP);
    W.vel.resize(APP_VEL_POOL);
    W.dz.resize(size_t(APP_NB_ARTI) * APP_MAX_LINKS * APP_NB_SLABS * 6);
    W.slab.resize(size_t(APP_NB_SLABS) * APP_NB_ARTI * 2);
    W.ab.resize(APP_NB_AB_BLOCKS);
    W.enc.resize(size_t(APP_NUM32) * APP_TOTAL_BODIES);
    W.islands.resize(2);
    W.acp.assign(8, 0);
    W.artiDesc.assign(APP_DESC_SIZE, 0xAB);

    std::memset(W.bt.data(), 0xAB, W.bt.size() * sizeof(W.bt[0]));
    std::memset(W.ch.data(), 0xAB, W.ch.size() * sizeof(W.ch[0]));
    std::memset(W.fh.data(), 0xAB, W.fh.size() * sizeof(W.fh[0]));
    std::memset(W.cp.data(), 0xAB, W.cp.size() * sizeof(W.cp[0]));
    std::memset(W.fp.data(), 0xAB, W.fp.size() * sizeof(W.fp[0]));
    std::memset(W.jh.data(), 0xAB, W.jh.size() * sizeof(W.jh[0]));
    std::memset(W.jr.data(), 0xAB, W.jr.size() * sizeof(W.jr[0]));
    std::memset(W.jm.data(), 0xAB, W.jm.size() * sizeof(W.jm[0]));
    std::memset(W.rs.data(), 0xAB, W.rs.size() * sizeof(W.rs[0]));
    std::memset(W.vel.data(), 0xAB, W.vel.size() * sizeof(W.vel[0]));
    std::memset(W.dz.data(), 0xAB, W.dz.size() * 4);
    std::memset(W.slab.data(), 0xAB, W.slab.size() * 4);
    std::memset(W.ab.data(), 0xAB, W.ab.size() * sizeof(W.ab[0]));
    std::memset(W.islands.data(), 0xAB, W.islands.size() * sizeof(W.islands[0]));
    std::memset(&W.solverDesc, 0xAB, sizeof(W.solverDesc));

    // ---- islands: pool[0] carries the pointer-deref fields, pool[1] the island's ----
    W.islands[0].mBatchCount = APP_BATCH_OFFSET;
    W.islands[0].mArticulationCount = APP_NB_ARTI;
    W.islands[0].mBodyStartIndex = APP_BODY_OFFSET;
    W.islands[0].mBodyCount = APP_NB_DYN;
    // decoys on island[1] for those fields (a pool[islandIndex] mutant dies)
    W.islands[1].mBatchCount = APP_BATCH_OFFSET + 5;
    W.islands[1].mArticulationCount = APP_NB_ARTI + 7;
    W.islands[1].mBodyStartIndex = APP_BODY_OFFSET + 2;
    W.islands[1].mBodyCount = APP_NB_DYN + 3;
    W.islands[1].mStartPartitionIndex = 4;
    W.islands[1].mBatchStartIndex = 0;
    W.islands[1].mArtiBatchStartIndex = (partitionIndex == 0) ? 5 : 6;   // decoy when unused

    // arti partition table: live warps at k = 8, 9 (start 5, end 7 + offset 3)
    for (uint32_t i = 0; i < 8; ++i) W.acp[i] = 90u + i;     // decoys
    if (partitionIndex == 0)
        W.acp[4] = 7;                                        // endIndex arm
    else
    {
        W.acp[partitionIndex + 4 - 1] = 5;                   // startIndex arm
        W.acp[partitionIndex + 4] = 7;
    }

    // ---- solver desc ----
    W.solverDesc.numSlabs = APP_NB_SLABS;
    W.solverDesc.contactErrorAccumulator.mErrorSumOfSquares = 0.5f;
    W.solverDesc.contactErrorAccumulator.mCounter = errCounter;
    W.solverDesc.contactErrorAccumulator.mMaxError = 0.25f;

    // ---- arti desc image (scalars; pointers patched at upload) ----
    uint32_t maxLinks = APP_MAX_LINKS;
    std::memcpy(&W.artiDesc[APP_D_MAX_LINKS], &maxLinks, 4);

    // ---- reference counts ----
    for (uint32_t r = 0; r < APP_NUM32; ++r)
        for (uint32_t bdy = 0; bdy < APP_TOTAL_BODIES; ++bdy)
        {
            uint32_t h = (bdy * 2654435761u) ^ (r * 40503u) ^ (salt * 733u);
            h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
            W.enc[size_t(r) * APP_TOTAL_BODIES + bdy] = h & 0x1FFFFu;
        }
    // a zero row inside the arti window pins the max(1, .) clamp
    W.enc[APP_BODY_OFFSET + APP_NB_DYN + 7] = 0u;

    // ---- the two live batches ----
    for (uint32_t b = 0; b < 2; ++b)
    {
        PxgBlockConstraintBatch& BT = W.bt[APP_K0 + b];
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
            const bool aA = appIsArtiA(b, lane), aB = appIsArtiB(b, lane);
            if (aA)
            {
                const uint32_t id = appArtiIdA(b, lane);
                BT.bodyANodeIndex[lane] = PxNodeIndex(600 + id, 1u + (lane % (APP_MAX_LINKS - 1u)));
                BT.bodyAIndex[lane] = id;
                BT.remappedBodyAIndex[lane] = (id + 5u) % APP_NB_ARTI;   // refcount row, != id
            }
            else
            {
                const uint32_t rid = (lane % 5u == 0u) ? (lane % APP_BODY_OFFSET)
                                                       : APP_BODY_OFFSET + (lane * 3u + b) % APP_NB_DYN;
                BT.bodyANodeIndex[lane] = PxNodeIndex(300 + b * 40 + lane);
                BT.bodyAIndex[lane] = rid;
                BT.remappedBodyAIndex[lane] = APP_OUT_BASE + (b * 64u + lane * 2u + 0u) * 64u;
            }
            if (aB)
            {
                const uint32_t id = appArtiIdB(b, lane);
                BT.bodyBNodeIndex[lane] = PxNodeIndex(700 + id, 1u + ((lane + 3u) % (APP_MAX_LINKS - 1u)));
                BT.bodyBIndex[lane] = id;
                BT.remappedBodyBIndex[lane] = (id + 11u) % APP_NB_ARTI;
            }
            else
            {
                const uint32_t rid = (lane % 7u == 2u) ? ((lane + 1u) % APP_BODY_OFFSET)
                                                       : APP_BODY_OFFSET + (lane * 5u + b + 1u) % APP_NB_DYN;
                BT.bodyBNodeIndex[lane] = PxNodeIndex(400 + b * 40 + lane);
                BT.bodyBIndex[lane] = rid;
                BT.remappedBodyBIndex[lane] = APP_OUT_BASE + (b * 64u + lane * 2u + 1u) * 64u;
            }
        }

        // staging slots for k = 8+b: the PGS 2-slot packing at k*128
        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            const uint32_t ri = (APP_K0 + b) * 128u + lane;
            for (uint32_t s = 0; s < 4; ++s)
                W.vel[ri + s * 32u] = make_float4(0.5f * fv(lane, b * 8 + s, 20, salt),
                                                  0.5f * fv(lane, b * 8 + s, 21, salt),
                                                  0.5f * fv(lane, b * 8 + s, 22, salt),
                                                  0.5f * fv(lane, b * 8 + s, 23, salt));
        }
    }

    // ---- contact solve rows (batch 8, desc 4) ----
    {
        const uint32_t bi = 4;
        PxgBlockSolverContactHeader& H = W.ch[bi];
        PxgBlockSolverFrictionHeader& F = W.fh[bi];
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
            H.flags[lane] = ((lane) % 3u == 1u) ? 8u : 0u;   // compliant-accel bit
            H.numNormalConstr[lane] = 1u + (lane % 3u);
            H.forceWritebackOffset[lane] = 0;
            H.restitution[lane] = (lane % 4u == 1u) ? (-600.0f - 5.0f * float(lane))
                                                    : (0.2f + 0.1f * std::fabs(fv(lane, bi, 71, salt)));
            H.cfm[lane] = 0.01f + 0.005f * std::fabs(fv(lane, bi, 72, salt));
            H.accumNormalForce[lane] = fv(lane, bi, 69, salt);   // unread by the solve
            F.numFrictionConstr[lane] = (lane % 3u) * 2u;        // 0/2/4, EVEN (PGS prep)
            F.dynamicFriction[lane] = 0.4f + 0.2f * std::fabs(fv(lane, bi, 73, salt));
            F.frictionNormals[0][lane] = make_float4(fv(lane, bi, 76, salt), fv(lane, bi, 77, salt),
                                                     fv(lane, bi, 78, salt), 0.f);
            F.frictionNormals[1][lane] = make_float4(fv(lane, bi, 79, salt), fv(lane, bi, 80, salt),
                                                     fv(lane, bi, 81, salt), 0.f);
            // broken stays POISONED: the doFriction=0 round must not write it
            for (uint32_t i = 0; i < 3; ++i)
            {
                PxgBlockSolverContactPoint& C = W.cp[5 + i];
                C.raXn_targetVelocity[lane] = make_float4(fv(lane, i, 82, salt), fv(lane, i, 83, salt),
                                                          fv(lane, i, 84, salt), 0.1f * fv(lane, i, 85, salt));
                C.rbXn_maxImpulse[lane] = make_float4(fv(lane, i, 86, salt), fv(lane, i, 87, salt),
                                                      fv(lane, i, 88, salt),
                                                      3.0f + std::fabs(fv(lane, i, 91, salt)));
                C.appliedForce[lane] = 0.3f + 0.2f * std::fabs(fv(lane, i, 92, salt));
                C.coeff0[lane] = (lane % 4u == 1u) ? 0.5f + 0.3f * std::fabs(fv(lane, i, 93, salt))
                                                   : 0.4f * fv(lane, i, 93, salt);
                C.coeff1[lane] = 0.3f * fv(lane, i, 90, salt);
                C.resp0[lane] = 0.5f + 0.3f * std::fabs(fv(lane, i, 94, salt));
                C.resp1[lane] = 0.5f + 0.3f * std::fabs(fv(lane, i, 95, salt));
            }
            for (uint32_t i = 0; i < 4; ++i)
            {
                PxgBlockSolverContactFriction& FR = W.fp[8 + i];
                FR.raXn_bias[lane] = make_float4(fv(lane, i, 96, salt), fv(lane, i, 97, salt),
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
        PxgBlockSolverConstraint1DHeader& H = W.jh[bi];
        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            H.rowCounts[lane] = PxU16(1u + (lane % 4u));
            H.breakable[lane] = PxU16(lane & 1u);   // unread by the solve
            H.invMass0D0[lane] = 0.5f + 0.2f * fv(lane, bi, 113, salt);
            H.invMass1D1[lane] = 0.5f + 0.2f * fv(lane, bi, 117, salt);
            H.invInertiaScale0[lane] = -0.8f - 0.2f * std::fabs(fv(lane, bi, 118, salt));
            H.invInertiaScale1[lane] = 0.8f + 0.2f * std::fabs(fv(lane, bi, 119, salt));
            H.writeBackOffset[lane] = 0;
            H.cfm[lane] = 0.02f;
            for (uint32_t r = 0; r < 4; ++r)
            {
                PxgBlockSolverConstraint1DCon& C = W.jr[20 + r];
                PxgBlockSolverConstraint1DMod& M = W.jm[20 + r];
                C.lin0XYZ_minImpulse[lane] =
                    make_float4(fv(lane, r, 120, salt), fv(lane, r, 121, salt), fv(lane, r, 122, salt),
                                -30.0f - std::fabs(fv(lane, r, 139, salt)));
                C.lin1XYZ_maxImpulse[lane] =
                    make_float4(fv(lane, r, 124, salt), fv(lane, r, 125, salt), fv(lane, r, 126, salt),
                                30.0f + std::fabs(fv(lane, r, 140, salt)));
                C.ang0XYZ_resp0[lane] =
                    make_float4(fv(lane, r, 128, salt), fv(lane, r, 129, salt), fv(lane, r, 130, salt),
                                0.5f + 0.3f * std::fabs(fv(lane, r, 141, salt)));
                C.ang1XYZ_resp1[lane] =
                    make_float4(fv(lane, r, 132, salt), fv(lane, r, 133, salt), fv(lane, r, 134, salt),
                                0.5f + 0.3f * std::fabs(fv(lane, r, 142, salt)));
                C.initJointSpeed[lane] = 0.2f * fv(lane, r, 135, salt);
                M.flags[lane] = (r == 1u) ? 32u : ((r == 3u) ? (32u | 128u) : 0u);   // SPRING / +ACCEL
                M.coeff0[lane] = (r == 1u || r == 3u) ? 0.5f + 0.3f * std::fabs(fv(lane, r, 136, salt))
                                                      : 0.1f * fv(lane, r, 136, salt);
                M.coeff1[lane] = 0.2f * fv(lane, r, 137, salt);
                M.appliedForce[lane] = 0.1f + 0.1f * std::fabs(fv(lane, r, 138, salt));
                // residual stays POISONED on the residual-off rounds
            }
        }
    }

    // responses read by both solves
    for (uint32_t r = 0; r < APP_NB_RESP; ++r)
        for (uint32_t lane = 0; lane < 32; ++lane)
            for (uint32_t s = 0; s < 12; ++s)
                (&W.rs[r].deltaRALin_x[0])[s * 32 + lane] = 0.25f * fv(lane, r, 150 + s, salt);
}

static uint32_t appCountSlabs(const AppArena& W, uint32_t bodyIndex)
{
    uint32_t c = 0;
    for (uint32_t i = 0; i < APP_NUM32; ++i)
        c += PxU32(__builtin_popcount(W.enc[size_t(i) * APP_TOTAL_BODIES + bodyIndex]));
    return PxMax(1u, c);
}

// P9.89: the per-warp sum of PxgErrorAccumulator::accumulateErrorGlobalFullWarp
// (gpucommon/src/CUDA/solverResidual.cuh:85 -> warpReduction<AddOpPxReal>,
// gpucommon/src/CUDA/reduction.cuh:188-199): a __shfl_xor_sync butterfly at
// radius 16, 8, 4, 2, 1 in which every lane ends with op(own, partner).  The
// lanes >= mDescStride `return`ed before it (solver.cu:100-101) and never
// reach the FULL_MASK shuffle; PTX leaves a read of an exited lane's register
// undefined.  They enter as +0, the AddOpPxReal identity, exactly as the port
// feeds them -- and at radius r lane 0 only ever reads a lane m in [r, 2r)
// whose accumulated partners m ^ S are all >= m, so it reads an exited lane
// only when that lane's whole subtree has exited: with +0 there, lane 0 of the
// CUDA and of the port carry the same tree bit for bit.
//   order 0: the CUDA's butterfly (what the mirror uses);
//   order 1: the WRONG radius order 1, 2, 4, 8, 16;
//   order 2: the WRONG linear lane sum ((l0 + l1) + l2) + ...
// The two wrong orders exist only as twins: the test proves its data makes
// them round differently from the butterfly (PXG_CHECK_MSG at the end), so a
// port that reassociates the reduction cannot pass the bitwise compare.
static float appWarpSum(const RefError lanes[32], int order)
{
    if (order == 2)
    {
        float s = 0.0f;
        for (uint32_t l = 0; l < 32; ++l)
            s += lanes[l].sumOfSquares;
        return s;
    }
    float sums[32];
    for (uint32_t l = 0; l < 32; ++l)
        sums[l] = lanes[l].sumOfSquares;
    for (uint32_t step = 0; step < 5; ++step)
    {
        const uint32_t radius = (order == 0) ? (16u >> step) : (1u << step);
        float ns[32];
        for (uint32_t l = 0; l < 32; ++l)
            ns[l] = sums[l] + sums[l ^ radius];
        std::memcpy(sums, ns, sizeof(ns));
    }
    return sums[0];
}

// the global accumulator after the residual round under each reduction order
struct AppTwins
{
    float acc[3];   // mErrorSumOfSquares with order 0 / 1 / 2
};

// the full host mirror
static void appMirror(AppArena& W, uint32_t partitionIndex, bool doFriction, bool residual,
                      uint32_t& artiWrites, uint32_t& rigidWrites, uint32_t& staticRefs,
                      AppTwins& twins)
{
    RefError warpErrLanes[2][32];
    for (uint32_t w = 0; w < 4; ++w)   // 2 blocks x 2 warps
    {
        const uint32_t startIndex = (partitionIndex == 0)
            ? W.islands[1].mArtiBatchStartIndex
            : W.acp[partitionIndex + 4 - 1];
        const uint32_t endIndex = W.acp[partitionIndex + 4] + APP_BATCH_OFFSET;
        const uint32_t k = startIndex + w + APP_BATCH_OFFSET;
        if (k >= endIndex)
            continue;
        const uint32_t b = k - APP_K0;
        PxgBlockConstraintBatch& BT = W.bt[k];

        for (uint32_t lane = 0; lane < BT.mDescStride; ++lane)
        {
            const uint32_t readIndex = k * 128u + lane;
            UnAlignedSpatialVector vel0, vel1;
            {
                const float4 lin = W.vel[readIndex], ang = W.vel[readIndex + 32];
                vel0 = UnAlignedSpatialVector(PxVec3(ang.x, ang.y, ang.z), PxVec3(lin.x, lin.y, lin.z));
            }
            {
                const float4 lin = W.vel[readIndex + 64], ang = W.vel[readIndex + 96];
                vel1 = UnAlignedSpatialVector(PxVec3(ang.x, ang.y, ang.z), PxVec3(lin.x, lin.y, lin.z));
            }

            const PxNodeIndex nA = BT.bodyANodeIndex[lane];
            const PxNodeIndex nB = BT.bodyBNodeIndex[lane];
            const uint32_t id0 = BT.bodyAIndex[lane];
            const uint32_t id1 = BT.bodyBIndex[lane];
            const uint32_t slabId = BT.slabId[lane];

            float ref0 = 1.f, ref1 = 1.f;
            if (nA.isArticulation())
                ref0 = float(appCountSlabs(W, BT.remappedBodyAIndex[lane] + APP_NB_DYN + APP_BODY_OFFSET));
            else if (id0 >= APP_BODY_OFFSET)
                ref0 = float(appCountSlabs(W, id0));
            else
                staticRefs++;
            if (nB.isArticulation())
                ref1 = float(appCountSlabs(W, BT.remappedBodyBIndex[lane] + APP_NB_DYN + APP_BODY_OFFSET));
            else if (id1 >= APP_BODY_OFFSET)
                ref1 = float(appCountSlabs(W, id1));
            else
                staticRefs++;

            UnAlignedSpatialVector imp0(PxVec3(0.f), PxVec3(0.f)), imp1(PxVec3(0.f), PxVec3(0.f));
            if (BT.constraintType == PxgSolverConstraintDesc::eARTICULATION_CONTACT)
                refSolveExtContactPGS(BT, vel0, vel1, doFriction, lane, W.ch.data(),
                                      W.fh.data(), W.cp.data(), W.fp.data(), W.rs.data(),
                                      imp0, imp1,
                                      residual ? &warpErrLanes[b][lane] : nullptr, ref0, ref1);
            else
                refSolveExt1DPGS(BT, vel0, vel1, lane, W.jh.data(), W.jr.data(), W.jm.data(),
                                 &W.rs[BT.mArticulationResponseIndex], imp0, imp1, residual,
                                 ref0, ref1);

            if (!nA.isArticulation())
            {
                const uint32_t o = BT.remappedBodyAIndex[lane];
                W.vel[o] = make_float4(vel0.bottom.x, vel0.bottom.y, vel0.bottom.z, 0.f);
                W.vel[o + 32] = make_float4(vel0.top.x, vel0.top.y, vel0.top.z, 0.f);
                rigidWrites++;
            }
            else
            {
                const uint32_t idx = id0 + nA.articulationLinkId() * APP_NB_ARTI
                                   + slabId * APP_NB_ARTI * APP_MAX_LINKS;
                float* z = &W.dz[size_t(idx) * 6];
                z[0] = imp0.top.x; z[1] = imp0.top.y; z[2] = imp0.top.z;
                z[3] = imp0.bottom.x; z[4] = imp0.bottom.y; z[5] = imp0.bottom.z;
                W.slab[(size_t(id0) + size_t(slabId) * APP_NB_ARTI) * 2 + 0] = nA.articulationLinkId();
                W.ab[id0 / 32u].mStateDirty[id0 & 31u] = 3;
                artiWrites++;
            }
            if (!nB.isArticulation())
            {
                const uint32_t o = BT.remappedBodyBIndex[lane];
                W.vel[o] = make_float4(vel1.bottom.x, vel1.bottom.y, vel1.bottom.z, 0.f);
                W.vel[o + 32] = make_float4(vel1.top.x, vel1.top.y, vel1.top.z, 0.f);
                rigidWrites++;
            }
            else
            {
                const uint32_t idx = id1 + nB.articulationLinkId() * APP_NB_ARTI
                                   + slabId * APP_NB_ARTI * APP_MAX_LINKS;
                float* z = &W.dz[size_t(idx) * 6];
                z[0] = imp1.top.x; z[1] = imp1.top.y; z[2] = imp1.top.z;
                z[3] = imp1.bottom.x; z[4] = imp1.bottom.y; z[5] = imp1.bottom.z;
                W.slab[(size_t(id1) + size_t(slabId) * APP_NB_ARTI) * 2 + 1] = nB.articulationLinkId();
                W.ab[id1 / 32u].mStateDirty[id1 & 31u] = 3;
                artiWrites++;
            }
        }
    }

    // the butterfly reduction of appWarpSum (order 0), then lane 0's atomic.
    // Only the contact warp (b = 0) carries a non-zero sum -- the 1D batch
    // writes its residual per row -- so the atomic order stays deterministic.
    for (int o = 0; o < 3; ++o)
        twins.acc[o] = W.solverDesc.contactErrorAccumulator.mErrorSumOfSquares;
    if (residual)
        for (uint32_t b = 0; b < 2; ++b)
        {
            float maxs[32];
            uint32_t counts[32];
            for (uint32_t l = 0; l < 32; ++l)
            {
                maxs[l] = warpErrLanes[b][l].maxError;
                counts[l] = uint32_t(warpErrLanes[b][l].counter);
            }
            for (uint32_t radius = 16; radius > 0; radius >>= 1)
            {
                float nm[32];
                uint32_t nc[32];
                for (uint32_t l = 0; l < 32; ++l)
                {
                    nm[l] = PxMax(maxs[l], maxs[l ^ radius]);
                    nc[l] = counts[l] + counts[l ^ radius];
                }
                std::memcpy(maxs, nm, sizeof(nm));
                std::memcpy(counts, nc, sizeof(nc));
            }
            for (int o = 0; o < 3; ++o)
                twins.acc[o] += appWarpSum(warpErrLanes[b], o);
            W.solverDesc.contactErrorAccumulator.mErrorSumOfSquares = twins.acc[0];
            W.solverDesc.contactErrorAccumulator.mCounter += int(counts[0]);
            W.solverDesc.contactErrorAccumulator.mMaxError =
                PxMax(W.solverDesc.contactErrorAccumulator.mMaxError, maxs[0]);
        }
}

} // namespace

PXG_TEST(artiSolvePartitionPgs)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiSolveExtPGS.spv", "artiSolveBlockPartition", 40u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &appLoader);

    // P9.89: rounds 3..8 add residual-enabled instances (fresh salts, both
    // start arms, friction on and off) so the warp butterfly is compared on
    // enough independent 26-lane sums that a reassociated reduction changes
    // the accumulator bits in more than one round (the twin check after the
    // loop); round 1 alone was the only reduction instance before.
    struct Round { uint32_t partition; int32_t errCounter; uint32_t doFriction; };
    const Round rounds[] = {
        { 2u, -5, 1u },   // disabled accumulator, [idx-1] start arm
        { 0u, 0, 1u },    // enabled accumulator, mArtiBatchStartIndex arm
        { 2u, -5, 0u },   // doFriction OFF: friction pools untouched, no broken write
        { 0u, 0, 1u },
        { 2u, 3, 1u },    // enabled on the [idx-1] arm, counter carried
        { 0u, 1, 0u },    // enabled with doFriction OFF: normal rows only
        { 2u, 0, 1u },
        { 0u, 0, 1u },
        { 2u, 2, 0u },
    };
    const uint32_t nbRounds = uint32_t(sizeof(rounds) / sizeof(rounds[0]));
    uint32_t residualRounds = 0, reversedDiffers = 0, linearDiffers = 0;

    for (uint32_t round = 0; round < nbRounds; ++round)
    {
        const Round& rd = rounds[round];
        AppArena A;
        appSeed(A, 17u + round * 5u, rd.partition, rd.errCounter);

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
        PxgDevicePtr dJm = up(A.jm.data(), A.jm.size() * sizeof(A.jm[0]));
        PxgDevicePtr dRs = up(A.rs.data(), A.rs.size() * sizeof(A.rs[0]));
        PxgDevicePtr dVel = up(A.vel.data(), A.vel.size() * sizeof(A.vel[0]));
        PxgDevicePtr dDz = up(A.dz.data(), A.dz.size() * 4);
        PxgDevicePtr dSlab = up(A.slab.data(), A.slab.size() * 4);
        PxgDevicePtr dAb = up(A.ab.data(), A.ab.size() * sizeof(A.ab[0]));
        PxgDevicePtr dEnc = up(A.enc.data(), A.enc.size() * 4);
        PxgDevicePtr dIsl = up(A.islands.data(), A.islands.size() * sizeof(A.islands[0]));
        PxgDevicePtr dAcp = up(A.acp.data(), A.acp.size() * 4);

        A.solverDesc.islandContextPool = reinterpret_cast<PxgIslandContext*>(dIsl);
        A.solverDesc.artiConstraintsPerPartition = reinterpret_cast<PxU32*>(dAcp);
        PxgDevicePtr dSd = up(&A.solverDesc, sizeof(A.solverDesc));

        // shared desc image (memset poison; only the used fields live)
        PxgSolverSharedDesc<IterativeSolveData> sh;
        std::memset(&sh, 0xAB, sizeof(sh));
        sh.articulationDeferredZ = reinterpret_cast<Cm::UnAlignedSpatialVector*>(dDz);
        sh.iterativeData.blockConstraintBatch = reinterpret_cast<PxgBlockConstraintBatch*>(dBt);
        sh.iterativeData.blockContactHeaders = reinterpret_cast<PxgBlockSolverContactHeader*>(dCh);
        sh.iterativeData.blockFrictionHeaders =
            reinterpret_cast<PxgBlockSolverFrictionHeader*>(dFh);
        sh.iterativeData.blockContactPoints = reinterpret_cast<PxgBlockSolverContactPoint*>(dCp);
        sh.iterativeData.blockFrictions = reinterpret_cast<PxgBlockSolverContactFriction*>(dFp);
        sh.iterativeData.blockJointConstraintHeaders =
            reinterpret_cast<PxgBlockSolverConstraint1DHeader*>(dJh);
        sh.iterativeData.blockJointConstraintRowsCon =
            reinterpret_cast<PxgBlockSolverConstraint1DCon*>(dJr);
        sh.iterativeData.blockJointConstraintRowsMod =
            reinterpret_cast<PxgBlockSolverConstraint1DMod*>(dJm);
        sh.iterativeData.artiResponse = reinterpret_cast<PxgArticulationBlockResponse*>(dRs);
        sh.iterativeData.solverBodyVelPool = reinterpret_cast<float4*>(dVel);
        sh.iterativeData.solverEncodedReferenceCount = reinterpret_cast<PxU32*>(dEnc);
        PxgDevicePtr dSh = up(&sh, sizeof(sh));

        // arti desc image with device pointers patched
        {
            uint64_t v;
            v = uint64_t(dSlab); std::memcpy(&A.artiDesc[APP_D_SLAB_HAS_CHANGES], &v, 8);
            v = uint64_t(dAb); std::memcpy(&A.artiDesc[APP_D_ARTICULATION_BLOCKS], &v, 8);
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
        AppTwins twins;
        appMirror(A, rd.partition, rd.doFriction != 0, rd.errCounter >= 0,
                  artiWrites, rigidWrites, staticRefs, twins);
        PXG_CHECK_MSG(artiWrites >= 20 && rigidWrites >= 15 && staticRefs >= 2,
                      "coverage: arti %u rigid %u static %u", artiWrites, rigidWrites, staticRefs);
        if (rd.errCounter >= 0)
        {
            uint32_t bits[3];
            std::memcpy(bits, twins.acc, sizeof(bits));
            ++residualRounds;
            reversedDiffers += (bits[1] != bits[0]) ? 1u : 0u;
            linearDiffers += (bits[2] != bits[0]) ? 1u : 0u;
            std::printf("  app r%u residual sum: butterfly %a  radius-1..16 %a  linear %a\n", round,
                        double(twins.acc[0]), double(twins.acc[1]), double(twins.acc[2]));
        }

        // ---- launch: grid (2, 1, 1) x block (32, 2, 1) ----
        struct Params
        {
            uint64_t solverCoreDesc, sharedDesc;
            uint32_t islandIndex, partitionIndex, doFrictionWord, pad0;
            uint64_t artiDesc;
        } prm = { uint64_t(dSd), uint64_t(dSh), 1u, rd.partition, rd.doFriction,
                  0u, uint64_t(dAd) };
        static_assert(sizeof(Params) == 40, "params blob");
        PXG_CHECK_MSG(shim.launchKernel(0, 2, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                      "launch artiSolveBlockPartition");
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
                        std::printf("MM app r%u %s word %zu (lane %zu)  ref %08x  gpu %08x\n",
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
        cmp(dJm, A.jm.data(), A.jm.size() * sizeof(A.jm[0]), "jointMods");
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
                std::printf("MM app r%u errorAccumulator ref (%a,%d,%a) gpu (%a,%d,%a)\n", round,
                            double(A.solverDesc.contactErrorAccumulator.mErrorSumOfSquares),
                            A.solverDesc.contactErrorAccumulator.mCounter,
                            double(A.solverDesc.contactErrorAccumulator.mMaxError), gs, gc, gm);
                ++bad;
            }
        }
        std::printf("  app r%u +-0 tolerated %zu\n", round, signedZeros);
        PXG_CHECK_MSG(bad == 0, "round %u: %zu mismatches", round, bad);

        for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
    }

    // P9.89: the data must make each wrong reduction order visible -- a fixture
    // whose sums round the same under every association verifies no tree
    std::printf("  app residual rounds %u: radius-1..16 differs in %u, linear differs in %u\n",
                residualRounds, reversedDiffers, linearDiffers);
    PXG_CHECK_MSG(reversedDiffers >= 2 && linearDiffers >= 2,
                  "reduction-order twins: radius-1..16 differs in %u, linear in %u of %u rounds",
                  reversedDiffers, linearDiffers, residualRounds);
}
