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

// testArtiSolveSelf.cpp -- P8.99b: ssSolveSelfConstraints, the self-constraint
// block of artiPropagateRigidImpulsesAndSolveSelfConstraints1T
// (internalConstraints2.cu:3861-4093, TGS instantiation).
//
// Mirror: control flow transcribed over engine loads (loadSpatialVector /
// loadSpatialMatrix / SpatialMatrix::operator*), the DIRECT-order acceleration
// hop (pxgtest::refPropagateAccelerationW = everythingPreLoaded, the order
// propagateAccelerationWNoJVelUpdate uses -- NOT the tendon-solve
// child2Parent order), the deferred impulse hop refPropImpulseDeferred with a
// NULL joint force, and the P8.99a ext-solve mirrors from pxgArtiExtSolveRef.h
// (whose 1D constants come from the ENGINE's host-callable
// compute1dConstraintSolverConstantsTGS).
//
// Fixture shape:
//  * ONE articulation block, 32 lanes, MAX_LINKS = 70 so wordSize = 2 -- the
//    word-crossing bookkeeping (wordOffset adds, top-word-down impulse walks)
//    is live, unlike the WORDSIZE=1 tendon fixtures where a dropped wordOffset
//    would be an identity.
//  * per-lane topologies differ (the branch step depends on the lane), while
//    links 0..7 form a lane-independent stem so two pairs are guaranteed:
//    (7,3) an ancestor pair (B-walk empty) and (1,2) a root-only common.
//  * batch node pairs cycle five scenarios including both-links-in-word-1 and
//    linkA == linkB; batch masks have holes (offset != lane), one mask
//    reaches the guarded-off lanes 30/31.
//  * five batches at a nonzero mSelfConstraintOffset, alternating contact/1D;
//    every start index and batch header index distinct; the poison tail of
//    every array must come back 0xAB.
//  * root link 0 carries LARGE sentinels (the P8.97c poison-size lesson): a
//    mutant that propagates at the root diverges loudly.
//  * two rounds: (minPen = -PX_MAX_F32, accumulateError on) and (minPen = 0,
//    accumulateError off -- 1D residuals must STAY poisoned).

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "foundation/PxQuat.h"
#include "foundation/PxVec3.h"
#include "CmSpatialVector.h"
#include "DyFeatherstoneArticulationUtils.h"
#include "DyFeatherstoneArticulation.h"
#include "PxgArticulation.h"
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

static std::vector<uint8_t> ssLoader(const char* file) { return loadShaderBlob(file); }

const uint32_t SS_LANES = 30;         // lanes 30/31 guarded off
const uint32_t SS_MAX_LINKS = 70;     // -> wordSize 2
const uint32_t SS_WORDSIZE = 2;
const uint32_t SS_MAX_DOFS = 211;     // per-lane jointOffset cursor stays below
const uint32_t SS_ROOT_DOF_SLICE = 205;  // root's poisoned dof slice (3 dofs)
const uint32_t SS_NB_BATCHES = 5;
const uint32_t SS_START_INDEX = 2;
const uint32_t SS_BATCH_ARRAY = SS_START_INDEX + SS_NB_BATCHES + 1;  // poison tail
const uint32_t SS_NB_HDRS = 8;
const uint32_t SS_NB_CPTS = 43;
const uint32_t SS_NB_FPTS = 40;
const uint32_t SS_NB_ROWS = 44;
const uint32_t SS_NB_RESP = 61;
const float SS_ELAPSED = 0.013f;

static_assert(sizeof(PxgBlockConstraintBatch) == 1536, "batch stride pin");
static_assert(offsetof(PxgBlockConstraintBatch, constraintType) == 2, "high half of the packed word");

static float fv(uint32_t a, uint32_t l, uint32_t d, uint32_t salt)
{
    uint32_t h = a * 2654435761u + l * 40503u + d * 9973u + salt * 733u + 19u;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    return -1.0f + 2.0f * float(h & 0xFFFFFFu) / float(0x1000000);
}

// links 0..7: lane-independent stem (1 and 2 both root children, 3..7 a chain
// under 2); above that the branch step varies with the lane.
static uint32_t ssParent(uint32_t l, uint32_t lane)
{
    if (l == 0 || l == 1 || l == 2) return 0;
    if (l < 8) return l - 1;
    const uint32_t back = 1u + (((l + lane) % 4u == 0u) ? 2u : 0u);
    return l - back;
}

static uint32_t ssDofs(uint32_t l, uint32_t lane) { return 1u + ((l + lane) % 3u); }

// the five node-pair scenarios (see header)
static void ssPair(uint32_t k, uint32_t& a, uint32_t& b)
{
    switch (k % 5u)
    {
    case 0: a = 67; b = 69; break;   // both word 1
    case 1: a = 64; b = 33; break;   // one per word
    case 2: a = 66; b = 66; break;   // equal links
    case 3: a = 7;  b = 3;  break;   // ancestor pair: B walk empty
    default: a = 1; b = 2;  break;   // root-only common
    }
}

static uint32_t ssMask(uint32_t b)
{
    switch (b)
    {
    case 0: return 0x00F0700Fu;              // holes -> offset != lane
    case 1: return 0x1F0F37F1u;
    case 2: return 0x00000E70u;              // sparse
    case 3: return 0xC300F00Fu;              // bits 30/31: guarded-off lanes
    default: return 0x2AAAAAAAu;
    }
}

struct SsArena
{
    std::vector<PxgArticulationBlockData> ab;      // 1
    std::vector<PxgArticulationBlockLinkData> lk;  // SS_MAX_LINKS
    std::vector<PxgArticulationBlockDofData> df;   // SS_MAX_DOFS
    std::vector<uint64_t> bf;                      // links * wordSize * 32
    std::vector<uint64_t> tmpA, tmpB, tmpC;        // wordSize * 32 each
    std::vector<PxgBlockConstraintBatch> bt;
    std::vector<PxgTGSBlockSolverContactHeader> ch;
    std::vector<PxgTGSBlockSolverFrictionHeader> fh;
    std::vector<PxgTGSBlockSolverContactPoint> cp;
    std::vector<PxgTGSBlockSolverContactFriction> fp;
    std::vector<PxgTGSBlockSolverConstraint1DHeader> jh;
    std::vector<PxgTGSBlockSolverConstraint1DCon> jr;
    std::vector<PxgArticulationBlockResponse> rs;
};

static bool ssIsContact(uint32_t b) { return (b % 2u) == 0u; }

static void ssSeed(SsArena& B, uint32_t salt)
{
    B.ab.resize(1);
    B.lk.resize(SS_MAX_LINKS);
    B.df.resize(SS_MAX_DOFS);
    B.bf.resize(size_t(SS_MAX_LINKS) * SS_WORDSIZE * 32);
    B.tmpA.resize(size_t(SS_WORDSIZE) * 32);
    B.tmpB.resize(size_t(SS_WORDSIZE) * 32);
    B.tmpC.resize(size_t(SS_WORDSIZE) * 32);
    B.bt.resize(SS_BATCH_ARRAY);
    B.ch.resize(SS_NB_HDRS);
    B.fh.resize(SS_NB_HDRS);
    B.cp.resize(SS_NB_CPTS);
    B.fp.resize(SS_NB_FPTS);
    B.jh.resize(SS_NB_HDRS);
    B.jr.resize(SS_NB_ROWS);
    B.rs.resize(SS_NB_RESP);
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

    PxgArticulationBlockData& AB = B.ab[0];

    // trap 7: per-block scalars
    AB.mTotalSelfConstraintCount = SS_NB_BATCHES;
    AB.mSelfConstraintOffset = SS_START_INDEX;

    for (uint32_t lane = 0; lane < 32; ++lane)
    {
        storeSpatialVector(AB.mRootDeferredZ,
                           UnAlignedSpatialVector(
                               PxVec3(fv(lane, 90, 0, salt), fv(lane, 90, 1, salt), fv(lane, 90, 2, salt)),
                               PxVec3(fv(lane, 90, 3, salt), fv(lane, 90, 4, salt), fv(lane, 90, 5, salt))),
                           lane);
        Dy::SpatialMatrix M;
        for (uint32_t c = 0; c < 3; ++c)
            for (uint32_t r = 0; r < 3; ++r)
            {
                M.topLeft[c][r] = 0.35f * fv(lane, 91, c * 3 + r, salt);
                M.topRight[c][r] = 0.35f * fv(lane, 92, c * 3 + r, salt);
                M.bottomLeft[c][r] = 0.35f * fv(lane, 93, c * 3 + r, salt);
            }
        storeSpatialMatrix(AB.mInvSpatialArticulatedInertia, lane, M);

        // links: rw, joint layout, velocities, TGS deltas; the dof cursor is
        // per lane (jointOffset[l] = sum of this lane's dofs below l)
        uint32_t cursor = 0;
        for (uint32_t l = 0; l < SS_MAX_LINKS; ++l)
        {
            PxgArticulationBlockLinkData& L = B.lk[l];
            if (l == 0)
            {
                // LARGE sentinels: nothing may read the root's hop data
                L.mRw_x[lane] = 7.5f; L.mRw_y[lane] = -3.25f; L.mRw_z[lane] = 9.0f;
                L.mJointOffset[lane] = SS_ROOT_DOF_SLICE;
                L.mDofs[lane] = 3;
            }
            else
            {
                L.mRw_x[lane] = 0.4f * fv(lane, l, 10, salt);
                L.mRw_y[lane] = 0.4f * fv(lane, l, 11, salt);
                L.mRw_z[lane] = 0.4f * fv(lane, l, 12, salt);
                L.mJointOffset[lane] = cursor;
                L.mDofs[lane] = PxU8(ssDofs(l, lane));
                cursor += ssDofs(l, lane);
            }
            storeSpatialVector(L.mMotionVelocity,
                               UnAlignedSpatialVector(
                                   PxVec3(fv(lane, l, 13, salt), fv(lane, l, 14, salt), fv(lane, l, 15, salt)),
                                   PxVec3(fv(lane, l, 16, salt), fv(lane, l, 17, salt), fv(lane, l, 18, salt))),
                               lane);
            storeSpatialVector(L.mDeltaMotion,
                               UnAlignedSpatialVector(
                                   PxVec3(0.1f * fv(lane, l, 19, salt), 0.1f * fv(lane, l, 20, salt), 0.1f * fv(lane, l, 21, salt)),
                                   PxVec3(0.1f * fv(lane, l, 22, salt), 0.1f * fv(lane, l, 23, salt), 0.1f * fv(lane, l, 24, salt))),
                               lane);
            PxQuat dq(fv(lane, l, 25, salt), fv(lane, l, 26, salt), fv(lane, l, 27, salt),
                      1.0f + 0.4f * fv(lane, l, 28, salt));
            dq.normalize();
            L.mDeltaQ[lane] = make_float4(dq.x, dq.y, dq.z, dq.w);
        }
        PX_ASSERT(cursor <= SS_ROOT_DOF_SLICE);

        // dof data everywhere (any link may sit on a path)
        for (uint32_t d = 0; d < SS_MAX_DOFS; ++d)
        {
            PxgArticulationBlockDofData& D = B.df[d];
            const bool rootSlice = (d >= SS_ROOT_DOF_SLICE && d < SS_ROOT_DOF_SLICE + 3);
            const float big = rootSlice ? 11.0f : 1.0f;
            storeSpatialVector(D.mWorldMotionMatrix,
                               UnAlignedSpatialVector(
                                   PxVec3(big * fv(lane, d, 30, salt), big * fv(lane, d, 31, salt), big * fv(lane, d, 32, salt)),
                                   PxVec3(big * fv(lane, d, 33, salt), big * fv(lane, d, 34, salt), big * fv(lane, d, 35, salt))),
                               lane);
            storeSpatialVector(D.mIsInvDW,
                               UnAlignedSpatialVector(
                                   PxVec3(big * fv(lane, d, 36, salt), big * fv(lane, d, 37, salt), big * fv(lane, d, 38, salt)),
                                   PxVec3(big * fv(lane, d, 39, salt), big * fv(lane, d, 40, salt), big * fv(lane, d, 41, salt))),
                               lane);
            storeSpatialVector(D.mIsW,
                               UnAlignedSpatialVector(
                                   PxVec3(big * fv(lane, d, 42, salt), big * fv(lane, d, 43, salt), big * fv(lane, d, 44, salt)),
                                   PxVec3(big * fv(lane, d, 45, salt), big * fv(lane, d, 46, salt), big * fv(lane, d, 47, salt))),
                               lane);
            D.mInvStIsT_x[lane] = 0.5f * fv(lane, d, 48, salt);
            D.mInvStIsT_y[lane] = 0.5f * fv(lane, d, 49, salt);
            D.mInvStIsT_z[lane] = 0.5f * fv(lane, d, 50, salt);
            D.mDeferredQstZ[lane] = 0.3f * fv(lane, d, 51, salt);
        }

        // path-to-root bitfields from this lane's parent chain
        for (uint32_t l = 0; l < SS_MAX_LINKS; ++l)
        {
            uint64_t words[SS_WORDSIZE] = { 0, 0 };
            uint32_t c = l;
            for (;;)
            {
                words[c / 64u] |= (uint64_t(1) << (c % 64u));
                if (c == 0) break;
                c = ssParent(c, lane);
            }
            for (uint32_t j = 0; j < SS_WORDSIZE; ++j)
                B.bf[size_t(l * SS_WORDSIZE + j) * 32 + lane] = words[j];
        }
    }

    // batches
    for (uint32_t b = 0; b < SS_NB_BATCHES; ++b)
    {
        PxgBlockConstraintBatch& BT = B.bt[SS_START_INDEX + b];
        BT.mDescStride = PxU16(17 + b);   // nonzero low half: a >>16 miss is visible
        BT.constraintType = PxU16(ssIsContact(b) ? PxgSolverConstraintDesc::eARTICULATION_CONTACT
                                                 : PxgSolverConstraintDesc::eARTICULATION_CONSTRAINT_1D);
        BT.mask = ssMask(b);
        BT.mConstraintBatchIndex = b + 2;
        BT.startConstraintIndex = ssIsContact(b) ? (b * 7 + 2) : (b * 8 + 1);
        BT.startFrictionIndex = b * 6 + 1;
        BT.mArticulationResponseIndex = b * 9 + 3;
        for (uint32_t k = 0; k < 32; ++k)
        {
            uint32_t la, lb;
            ssPair(b * 7 + k, la, lb);
            BT.bodyANodeIndex[k] = PxNodeIndex(40 + b, la);
            BT.bodyBNodeIndex[k] = PxNodeIndex(40 + b, lb);
        }
    }

    // contact blocks
    for (uint32_t b = 0; b < SS_NB_BATCHES; ++b)
    {
        if (!ssIsContact(b)) continue;
        const uint32_t hi = b + 2;
        PxgTGSBlockSolverContactHeader& H = B.ch[hi];
        PxgTGSBlockSolverFrictionHeader& F = B.fh[hi];
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
            H.biasCoefficient[lane] = 0.9f;   // 5.12 renamed p8
            F.numFrictionConstr[lane] = ((lane + b) % 3u) * 2u + (((lane + b) % 5u == 2u) ? 1u : 0u);
            F.dynamicFriction[lane] = 0.4f + 0.2f * fv(lane, b, 73, salt);
            F.biasCoefficient[lane] = 0.5f + 0.2f * fv(lane, b, 74, salt);
            F.torsionalFrictionScale[lane] = 0.8f + 0.1f * fv(lane, b, 75, salt);
            F.frictionNormals[0][lane] = make_float4(fv(lane, b, 76, salt), fv(lane, b, 77, salt), fv(lane, b, 78, salt), 0.f);
            F.frictionNormals[1][lane] = make_float4(fv(lane, b, 79, salt), fv(lane, b, 80, salt), fv(lane, b, 81, salt), 0.f);
            F.broken[lane] = 0;
            for (uint32_t i = 0; i < 3; ++i)
            {
                PxgTGSBlockSolverContactPoint& C = B.cp[B.bt[SS_START_INDEX + b].startConstraintIndex + i];
                C.raXn_extraCoeff[lane] = make_float4(fv(lane, b * 4 + i, 82, salt), fv(lane, b * 4 + i, 83, salt),
                                                      fv(lane, b * 4 + i, 84, salt), 0.5f + 0.3f * fv(lane, b * 4 + i, 85, salt));
                C.rbXn_targetVelW[lane] = make_float4(fv(lane, b * 4 + i, 86, salt), fv(lane, b * 4 + i, 87, salt),
                                                      fv(lane, b * 4 + i, 88, salt), 0.1f * fv(lane, b * 4 + i, 89, salt));
                C.separation[lane] = 0.05f * fv(lane, b * 4 + i, 90, salt);
                C.maxImpulse[lane] = 3.0f + fv(lane, b * 4 + i, 91, salt);
                C.appliedForce[lane] = 0.3f + 0.2f * fv(lane, b * 4 + i, 92, salt);
                C.biasCoefficient[lane] = 0.4f + 0.2f * fv(lane, b * 4 + i, 93, salt);
                // one lane per batch drives the unitResponse <= 0 branch
                const bool zeroResp = (lane == (b * 5 + 3) % 30) && (i == 0);
                C.resp0[lane] = zeroResp ? 0.f : 0.5f + 0.3f * fv(lane, b * 4 + i, 94, salt);
                C.resp1[lane] = zeroResp ? 0.f : 0.5f + 0.3f * fv(lane, b * 4 + i, 95, salt);
            }
            for (uint32_t i = 0; i < 5; ++i)
            {
                PxgTGSBlockSolverContactFriction& FR = B.fp[B.bt[SS_START_INDEX + b].startFrictionIndex + i];
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

    // 1D blocks
    for (uint32_t b = 0; b < SS_NB_BATCHES; ++b)
    {
        if (ssIsContact(b)) continue;
        const uint32_t hi = b + 2;
        PxgTGSBlockSolverConstraint1DHeader& H = B.jh[hi];
        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            // low byte = rowCount; the other packed bytes NONZERO so a wrong-byte
            // read lands on a different count
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
                PxgTGSBlockSolverConstraint1DCon& R = B.jr[B.bt[SS_START_INDEX + b].startConstraintIndex + i];
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
                // residual left poisoned: only the accumulateError round writes it
            }
        }
    }

    // responses
    for (uint32_t r = 0; r < SS_NB_RESP; ++r)
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

// ---- mirror (hoisted to pxgArtiSelfRef.h in P8.99c; this fixture is a user) ----

static void refSolveSelf(SsArena& W, float minPen, bool accumErr, SelfCoverage& cov)
{
    for (PxU32 lane = 0; lane < SS_LANES; ++lane)
    {
        RefError err;
        refSolveSelfLane(W.ab[0], W.lk.data(), W.df.data(), W.bf.data(), SS_WORDSIZE,
                         W.tmpA.data(), W.tmpB.data(), W.tmpC.data(),
                         W.bt.data(), W.ch.data(), W.fh.data(), W.cp.data(), W.fp.data(),
                         W.jh.data(), W.jr.data(), W.rs.data(),
                         SS_ELAPSED, minPen, accumErr, err, cov, lane);
    }
}

// ---- run ----

struct SsParamsHost
{
    uint64_t artiBlock, artiLinks, artiDofs, linkBitFields, tmpA, tmpB, tmpC;
    uint64_t batches, contactHeaders, frictionHeaders, contactPoints, frictionPoints;
    uint64_t jointHeaders, jointRowsCon, responses;
    float elapsedTime, minPen;
    uint32_t wordSize, accumulateError, nbLanes, pad0;
};
static_assert(sizeof(SsParamsHost) == 144, "params blob");

static size_t ssRun(PxgCudaShim& shim, float minPen, bool accumErr, uint32_t salt)
{
    SsArena A;
    ssSeed(A, salt);

    SsArena W = A;
    SelfCoverage cov;
    refSolveSelf(W, minPen, accumErr, cov);

    std::printf("[ info ] self(%s): contact %u / 1D %u batches; compliant %u, torsional %u, "
                "spring rows %u, zero-resp %u\n",
                accumErr ? "accum" : "plain", cov.contacts, cov.oneD, cov.compliant,
                cov.torsional, cov.springRows, cov.zeroResp);
    std::printf("[ info ] self walks: word1 hops %u, equal pairs %u, empty B %u, "
                "offset!=lane %u, inactive %u, root-only-common %u\n",
                cov.word1Hops, cov.equalPairs, cov.emptyBWalks, cov.offsetNeLane,
                cov.inactive, cov.rootOnlyCommon);
    PXG_CHECK_MSG(cov.contacts > 0 && cov.oneD > 0, "both constraint types must run");
    PXG_CHECK_MSG(cov.compliant > 0 && cov.torsional > 0 && cov.springRows > 0,
                  "solver branch coverage hole");
    PXG_CHECK_MSG(cov.word1Hops > 0 && cov.equalPairs > 0 && cov.emptyBWalks > 0,
                  "walk coverage hole");
    PXG_CHECK_MSG(cov.offsetNeLane > 0 && cov.inactive > 0 && cov.rootOnlyCommon > 0,
                  "mask coverage hole");
    PXG_CHECK_MSG(cov.zeroResp > 0, "zero-response contact case missing");

    // non-vacuity: the reference run must have changed the seed image
    PXG_CHECK_MSG(std::memcmp(W.ab.data(), A.ab.data(), sizeof(PxgArticulationBlockData)) != 0,
                  "vacuous: rootDeferredZ unchanged");
    PXG_CHECK_MSG(std::memcmp(W.df.data(), A.df.data(), W.df.size() * sizeof(W.df[0])) != 0,
                  "vacuous: no mDeferredQstZ accumulation");

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
    PxgDevicePtr dBt = up(A.bt.data(), A.bt.size() * sizeof(A.bt[0]));
    PxgDevicePtr dCh = up(A.ch.data(), A.ch.size() * sizeof(A.ch[0]));
    PxgDevicePtr dFh = up(A.fh.data(), A.fh.size() * sizeof(A.fh[0]));
    PxgDevicePtr dCp = up(A.cp.data(), A.cp.size() * sizeof(A.cp[0]));
    PxgDevicePtr dFp = up(A.fp.data(), A.fp.size() * sizeof(A.fp[0]));
    PxgDevicePtr dJh = up(A.jh.data(), A.jh.size() * sizeof(A.jh[0]));
    PxgDevicePtr dJr = up(A.jr.data(), A.jr.size() * sizeof(A.jr[0]));
    PxgDevicePtr dRs = up(A.rs.data(), A.rs.size() * sizeof(A.rs[0]));
    bool ok = true;
    for (PxgDevicePtr p : bufs) ok = ok && (p != 0);
    if (!ok)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
        return 0;
    }

    SsParamsHost prm = { uint64_t(dAb), uint64_t(dLk), uint64_t(dDf), uint64_t(dBf),
                         uint64_t(dTa), uint64_t(dTb), uint64_t(dTc),
                         uint64_t(dBt), uint64_t(dCh), uint64_t(dFh), uint64_t(dCp),
                         uint64_t(dFp), uint64_t(dJh), uint64_t(dJr), uint64_t(dRs),
                         SS_ELAPSED, minPen,
                         SS_WORDSIZE, accumErr ? 1u : 0u, SS_LANES, 0u };
    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch self solve");
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
            std::printf("MM self/%s: %zu/%zu bytes differ (first at %zu)\n", name, mm, n, first);
            bad += mm;
        }
    };
    cmp(dAb, W.ab.data(), W.ab.size() * sizeof(W.ab[0]), "artiBlock");
    cmp(dLk, W.lk.data(), W.lk.size() * sizeof(W.lk[0]), "linkBlocks");
    cmp(dDf, W.df.data(), W.df.size() * sizeof(W.df[0]), "dofBlocks");
    cmp(dBf, W.bf.data(), W.bf.size() * 8, "linkBitFields");
    cmp(dTa, W.tmpA.data(), W.tmpA.size() * 8, "tmpA");
    cmp(dTb, W.tmpB.data(), W.tmpB.size() * 8, "tmpB");
    cmp(dTc, W.tmpC.data(), W.tmpC.size() * 8, "tmpCommon");
    cmp(dBt, W.bt.data(), W.bt.size() * sizeof(W.bt[0]), "batches");
    cmp(dCh, W.ch.data(), W.ch.size() * sizeof(W.ch[0]), "contactHeaders");
    cmp(dFh, W.fh.data(), W.fh.size() * sizeof(W.fh[0]), "frictionHeaders");
    cmp(dCp, W.cp.data(), W.cp.size() * sizeof(W.cp[0]), "contactPoints");
    cmp(dFp, W.fp.data(), W.fp.size() * sizeof(W.fp[0]), "frictions");
    cmp(dJh, W.jh.data(), W.jh.size() * sizeof(W.jh[0]), "jointHeaders");
    cmp(dJr, W.jr.data(), W.jr.size() * sizeof(W.jr[0]), "jointRows");
    cmp(dRs, W.rs.data(), W.rs.size() * sizeof(W.rs[0]), "responses");

    for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
    return bad;
}

} // namespace

PXG_TEST(artiSolveSelf)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiSolveSelf.spv", "ssSolveSelfTest", 144u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &ssLoader);

    size_t bad = 0;
    bad += ssRun(shim, -PX_MAX_F32, true, 5);   // position-iteration shape, residuals on
    bad += ssRun(shim, 0.0f, false, 11);        // velocity-iteration shape, residuals stay poisoned
    PXG_CHECK_MSG(bad == 0, "%zu mismatches", bad);
}
