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

// testArtiSolveDFSPgs.cpp -- P8.123c: siSolveInternalDFSPgs, the DFS of
// artiSolveInternalConstraints1T (the PGS extern :2820, walk :1662-2002 with
// isTGS == false).  Mirror: refSolveInternalDFSLanePgs (pxgArtiInternalRef.h)
// over the PGS pools and the P8.123a/b lane mirrors.  The tree/window/pool
// geometry is the TGS fixture's; PGS facts: doFriction is LIVE through the
// static solve, the 1D static dispatch carries rowsMod with the OFFSET
// response base, mDeltaQ stays UNREAD (poisoned -- the PGS static solve has
// no deltaQ), the vel-iteration bias floor and isExternalForceEveryStep do not exist on this
// path, and elapsedTime is live on velocity iterations too (the drive
// shortcut is TGS-only).  mDeltaMotion stays seeded: the DOF solve reads it
// (jointDeltaP) even on PGS -- only the STATIC solve's delta reads are gone.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "foundation/PxQuat.h"
#include "foundation/PxVec3.h"
#include "CmSpatialVector.h"
#include "DyFeatherstoneArticulationUtils.h"
#include "DyFeatherstoneArticulation.h"
#include "DyCpuGpuArticulation.h"
#include "PxSpatialMatrix.h"
#include "PxgArticulation.h"
#include "pxgArtiRefHelpers.h"
#include "pxgArtiSolveRef.h"
#include "pxgArtiExtSolveRef.h"
#include "pxgArtiInternalRef.h"

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

static std::vector<uint8_t> sfpLoader(const char* file) { return loadShaderBlob(file); }

const uint32_t SFP_LANES = 30;
const uint32_t SFP_MAX_LINKS = 6, SFP_MAX_DOFS = 11;
const uint32_t SFP_NB_BATCHES = 14, SFP_J0 = 2, SFP_C0 = 8;
const uint32_t SFP_NB_HDRS = 14, SFP_NB_CPTS = 60, SFP_NB_FPTS = 80, SFP_NB_ROWS = 40, SFP_NB_RESP = 130;
const float SFP_DT = 0.01f, SFP_SCENE_DT = 0.033f, SFP_INV_DT = 100.0f;
const float SFP_ELAPSED = 0.02f, SFP_ERP = 0.6f;
const uint32_t SFP_IO_STRIDE = 24, SFP_IO_OUT = 32 * 24;

static_assert(sizeof(PxgArticulationTraversalStackData) == 1664, "stack stride pin");
static_assert(offsetof(PxgArticulationTraversalStackData, impulseStack) == 128, "stack impulse pin");
static_assert(offsetof(PxgArticulationTraversalStackData, deltaVStack) == 896, "stack deltaV pin");

static float fv(uint32_t a, uint32_t l, uint32_t d, uint32_t salt)
{
    uint32_t h = a * 2654435761u + l * 40503u + d * 9973u + salt * 733u + 19u;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    return -1.0f + 2.0f * float(h & 0xFFFFFFu) / float(0x1000000);
}

static uint32_t sfpNumLinks(uint32_t lane) { return 4u + (lane % 3u); }
static uint32_t sfpParent(uint32_t l)
{
    static const uint32_t p[6] = { 0, 0, 1, 1, 2, 3 };
    return p[l];
}
static void sfpChildren(uint32_t l, uint32_t numLinks, uint32_t& off, uint32_t& num)
{
    off = 0; num = 0;
    for (uint32_t c = 1; c < numLinks; ++c)
        if (sfpParent(c) == l) { if (num == 0) off = c; ++num; }
}
static uint32_t sfpDofs(uint32_t l, uint32_t lane) { return 1u + ((lane + l) % 2u); }

// static windows (per link, per lane) into the shared pools
static uint32_t sfpJStart(uint32_t lane, uint32_t l) { return SFP_J0 + ((lane + l) % 3u); }
static uint32_t sfpJCount(uint32_t lane, uint32_t l) { return ((lane + l) % 4u == 1u) ? 1u : 0u; }
static uint32_t sfpCStart(uint32_t lane, uint32_t l) { return SFP_C0 + ((lane + 2u * l) % 4u); }
static uint32_t sfpCCount(uint32_t lane, uint32_t l) { return ((lane + l) % 5u == 2u) ? 1u : 0u; }

static bool sfpIs1D(uint32_t b) { return b >= SFP_J0 && b < SFP_J0 + 5; }
static bool sfpIsContact(uint32_t b) { return b >= SFP_C0 && b < SFP_C0 + 5; }
static bool sfpAIsArti(uint32_t b, uint32_t k) { return ((b + k) % 3u) != 0u; }

struct SfpArena
{
    std::vector<PxgArticulationBlockLinkData> lk;
    std::vector<PxgArticulationBlockDofData> df;
    std::vector<PxgArticulationTraversalStackData> stack;
    std::vector<float> jf;
    std::vector<PxgBlockConstraintBatch> bt;
    std::vector<PxgBlockSolverContactHeader> ch;
    std::vector<PxgBlockSolverFrictionHeader> fh;
    std::vector<PxgBlockSolverContactPoint> cp;
    std::vector<PxgBlockSolverContactFriction> fp;
    std::vector<PxgBlockSolverConstraint1DHeader> jh;
    std::vector<PxgBlockSolverConstraint1DCon> jr;
    std::vector<PxgBlockSolverConstraint1DMod> jm;
    std::vector<PxgArticulationBlockResponse> rs;
    std::vector<float> io;
};

static void sfpSeed(SfpArena& B, uint32_t salt)
{
    B.lk.resize(SFP_MAX_LINKS);
    B.df.resize(SFP_MAX_DOFS);
    B.stack.resize(SFP_MAX_LINKS);
    B.jf.resize(32 * SFP_MAX_DOFS);
    B.bt.resize(SFP_NB_BATCHES);
    B.ch.resize(SFP_NB_HDRS);
    B.fh.resize(SFP_NB_HDRS);
    B.cp.resize(SFP_NB_CPTS);
    B.fp.resize(SFP_NB_FPTS);
    B.jh.resize(SFP_NB_HDRS);
    B.jr.resize(SFP_NB_ROWS);
    B.jm.resize(SFP_NB_ROWS);
    B.rs.resize(SFP_NB_RESP);
    B.io.resize(64 * SFP_IO_STRIDE);
    std::memset(B.lk.data(), 0xAB, B.lk.size() * sizeof(B.lk[0]));
    std::memset(B.df.data(), 0xAB, B.df.size() * sizeof(B.df[0]));
    std::memset(B.stack.data(), 0xAB, B.stack.size() * sizeof(B.stack[0]));
    std::memset(B.bt.data(), 0xAB, B.bt.size() * sizeof(B.bt[0]));
    std::memset(B.ch.data(), 0xAB, B.ch.size() * sizeof(B.ch[0]));
    std::memset(B.fh.data(), 0xAB, B.fh.size() * sizeof(B.fh[0]));
    std::memset(B.cp.data(), 0xAB, B.cp.size() * sizeof(B.cp[0]));
    std::memset(B.fp.data(), 0xAB, B.fp.size() * sizeof(B.fp[0]));
    std::memset(B.jh.data(), 0xAB, B.jh.size() * sizeof(B.jh[0]));
    std::memset(B.jr.data(), 0xAB, B.jr.size() * sizeof(B.jr[0]));
    std::memset(B.jm.data(), 0xAB, B.jm.size() * sizeof(B.jm[0]));
    std::memset(B.rs.data(), 0xAB, B.rs.size() * sizeof(B.rs[0]));
    std::memset(B.io.data(), 0xAB, B.io.size() * 4);

    for (size_t i = 0; i < B.jf.size(); ++i)
        B.jf[i] = 0.5f * fv(uint32_t(i), 210, 0, salt);

    for (uint32_t lane = 0; lane < 32; ++lane)
    {
        const uint32_t numLinks = sfpNumLinks(lane);
        uint32_t cursor = 0;
        for (uint32_t l = 0; l < SFP_MAX_LINKS; ++l)
        {
            PxgArticulationBlockLinkData& L = B.lk[l];
            const bool live = l < numLinks;
            L.mParents[lane] = live ? sfpParent(l) : 0;
            uint32_t off, num;
            sfpChildren(l, numLinks, off, num);
            L.mChildrenOffset[lane] = live ? off : 0;
            L.mNumChildren[lane] = live ? num : 0;
            if (l == 0)
            {
                L.mRw_x[lane] = 7.5f; L.mRw_y[lane] = -3.25f; L.mRw_z[lane] = 9.0f;
                L.mJointOffset[lane] = SFP_MAX_DOFS - 2;
                L.mDofs[lane] = 2;
            }
            else
            {
                L.mRw_x[lane] = 0.4f * fv(lane, l, 10, salt);
                L.mRw_y[lane] = 0.4f * fv(lane, l, 11, salt);
                L.mRw_z[lane] = 0.4f * fv(lane, l, 12, salt);
                L.mJointOffset[lane] = cursor;
                L.mDofs[lane] = PxU8(sfpDofs(l, lane));
                cursor += sfpDofs(l, lane);
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
            // mDeltaQ stays POISONED: the PGS static solve never reads it
            storeSpatialVector(L.mSolverSpatialImpulse,
                               UnAlignedSpatialVector(
                                   PxVec3(0.2f * fv(lane, l, 52, salt), 0.2f * fv(lane, l, 53, salt), 0.2f * fv(lane, l, 54, salt)),
                                   PxVec3(0.2f * fv(lane, l, 55, salt), 0.2f * fv(lane, l, 56, salt), 0.2f * fv(lane, l, 57, salt))),
                               lane);
            storeSpatialVector(L.mSolverSpatialInternalConstraintImpulse,
                               UnAlignedSpatialVector(
                                   PxVec3(0.15f * fv(lane, l, 58, salt), 0.15f * fv(lane, l, 59, salt), 0.15f * fv(lane, l, 60, salt)),
                                   PxVec3(0.15f * fv(lane, l, 61, salt), 0.15f * fv(lane, l, 62, salt), 0.15f * fv(lane, l, 63, salt))),
                               lane);
            PxSpatialMatrix R;
            for (uint32_t i = 0; i < 6; ++i)
                for (uint32_t j = 0; j < 6; ++j)
                    R.column[i][j] = 0.15f * fv(lane, l * 40 + i * 6 + j, 64, salt);
            storeSpatialMatrix(L.mSpatialResponseMatrix, lane, R);

            L.mStaticJointStartIndex[lane] = sfpJStart(lane, l);
            L.mNbStaticJoints[lane] = sfpJCount(lane, l);
            L.mStaticContactStartIndex[lane] = sfpCStart(lane, l);
            L.mNbStaticContacts[lane] = sfpCCount(lane, l);
        }
        PX_ASSERT(cursor <= SFP_MAX_DOFS - 2);

        for (uint32_t d = 0; d < SFP_MAX_DOFS; ++d)
        {
            PxgArticulationBlockDofData& D = B.df[d];
            PxgArticulationInternalConstraintData& ic = D.mConstraintData;
            const bool rootSlice = d >= SFP_MAX_DOFS - 2;
            const float big = rootSlice ? 11.0f : 1.0f;

            D.mMotion[lane] = PxU8((lane + d) % 3u);
            D.mDeferredQstZ[lane] = 0.3f * fv(lane, d, 51, salt);
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

            storeSpatialVector(ic.mRow0,
                               UnAlignedSpatialVector(
                                   PxVec3(fv(lane, d, 110, salt), fv(lane, d, 111, salt), fv(lane, d, 112, salt)),
                                   PxVec3(fv(lane, d, 113, salt), fv(lane, d, 114, salt), fv(lane, d, 115, salt))),
                               lane);
            storeSpatialVector(ic.mRow1,
                               UnAlignedSpatialVector(
                                   PxVec3(fv(lane, d, 116, salt), fv(lane, d, 117, salt), fv(lane, d, 118, salt)),
                                   PxVec3(fv(lane, d, 119, salt), fv(lane, d, 120, salt), fv(lane, d, 121, salt))),
                               lane);
            storeSpatialVector(ic.mDeltaVA,
                               UnAlignedSpatialVector(
                                   PxVec3(0.3f * fv(lane, d, 122, salt), 0.3f * fv(lane, d, 123, salt), 0.3f * fv(lane, d, 124, salt)),
                                   PxVec3(0.3f * fv(lane, d, 125, salt), 0.3f * fv(lane, d, 126, salt), 0.3f * fv(lane, d, 127, salt))),
                               lane);
            storeSpatialVector(ic.mDeltaVB,
                               UnAlignedSpatialVector(
                                   PxVec3(0.3f * fv(lane, d, 128, salt), 0.3f * fv(lane, d, 129, salt), 0.3f * fv(lane, d, 130, salt)),
                                   PxVec3(0.3f * fv(lane, d, 131, salt), 0.3f * fv(lane, d, 132, salt), 0.3f * fv(lane, d, 133, salt))),
                               lane);
            const float response = 0.6f + 0.3f * std::fabs(fv(lane, d, 134, salt));
            ic.mResponse[lane] = response;
            ic.mRecipResponse[lane] = 1.0f / response;
            ic.mMaxJointVelocity[lane] = 0.4f + 0.2f * std::fabs(fv(lane, d, 135, salt));
            ic.mConstraintMaxImpulse[lane] = 0.6f + 0.3f * std::fabs(fv(lane, d, 136, salt));
            ic.mDriveImpulse[lane] = 0.3f * fv(lane, d, 137, salt);
            ic.mMaxFrictionForce[lane] = 0.4f + 0.2f * std::fabs(fv(lane, d, 138, salt));
            ic.mAccumulatedFrictionImpulse[lane] = 0.2f * fv(lane, d, 139, salt);
            ic.mMaxEffort[lane] = ((lane + d) % 2u) ? (30.0f + 10.0f * std::fabs(fv(lane, d, 140, salt))) : 0.0f;
            ic.mMaxActuatorVelocity[lane] = 2.0f + std::fabs(fv(lane, d, 141, salt));
            ic.mSpeedEffortGradient[lane] = ((lane + d) % 5u == 2u) ? 0.0f : 0.02f * std::fabs(fv(lane, d, 142, salt));
            ic.mVelocityDependentResistance[lane] = ((lane + d) % 3u == 1u) ? (4.0f + std::fabs(fv(lane, d, 143, salt))) : 0.0f;
            const bool newModel = (lane % 3u) != 0u;
            ic.mStaticFrictionEffort[lane] = newModel ? (8.0f + 4.0f * std::fabs(fv(lane, d, 144, salt))) : 0.0f;
            ic.mDynamicFrictionEffort[lane] = newModel ? (5.0f + 2.0f * std::fabs(fv(lane, d, 145, salt))) : 0.0f;
            ic.mViscousFrictionCoefficient[lane] = (newModel && (lane % 4u == 1u)) ? (0.5f + std::fabs(fv(lane, d, 146, salt))) : 0.0f;
            const uint32_t lm = (lane + d) % 4u;
            float lo, hi;
            if (lm == 0) { lo = -0.05f - 0.02f * std::fabs(fv(lane, d, 147, salt)); hi = 0.4f; }
            else if (lm == 1) { lo = 0.004f; hi = 0.5f; }
            else if (lm == 2) { lo = 0.4f; hi = -0.03f - 0.02f * std::fabs(fv(lane, d, 148, salt)); }
            else { lo = 0.5f; hi = 0.5f; }
            ic.mLimitError_LowX_highY[lane] = make_float2(lo, hi);
            ic.mLowImpulse[lane] = 0.1f * std::fabs(fv(lane, d, 149, salt));
            ic.mHighImpulse[lane] = -0.1f * std::fabs(fv(lane, d, 150, salt));
            ic.mTargetVelPlusInitialBiasX_DriveBiasCoefficientY_VelMultiplierZ_ImpulseMultiplierW[lane] =
                make_float4(0.3f * fv(lane, d, 151, salt), 0.4f * fv(lane, d, 152, salt),
                            -0.5f * std::fabs(fv(lane, d, 153, salt)), 0.9f + 0.1f * fv(lane, d, 154, salt));
            ic.mTargetPosBias[lane] = 0.2f * fv(lane, d, 155, salt);
        }

        // io: storeParentDeltaV, parentImpIn, parentDeltaVIn
        for (uint32_t j = 0; j < 18; ++j)
            B.io[lane * SFP_IO_STRIDE + j] = 0.3f * fv(lane, 3, j, salt);
    }

    for (uint32_t b = 0; b < SFP_NB_BATCHES; ++b)
    {
        if (!sfpIs1D(b) && !sfpIsContact(b)) continue;
        PxgBlockConstraintBatch& BT = B.bt[b];
        BT.mDescStride = PxU16(31 + b);
        BT.constraintType = PxU16(sfpIsContact(b) ? PxgSolverConstraintDesc::eARTICULATION_CONTACT
                                                 : PxgSolverConstraintDesc::eARTICULATION_CONSTRAINT_1D);
        BT.mask = 0xFFFFFFFFu;
        BT.mConstraintBatchIndex = b;
        BT.startConstraintIndex = sfpIsContact(b) ? ((b - SFP_C0) * 5 + 3) : ((b - SFP_J0) * 6 + 2);
        BT.startFrictionIndex = (b - SFP_C0) * 7 + 4;
        BT.mArticulationResponseIndex = b * 9 + 2;
        for (uint32_t k = 0; k < 32; ++k)
        {
            if (sfpAIsArti(b, k))
            {
                BT.bodyANodeIndex[k] = PxNodeIndex(60 + b, 3 + (k % 5));
                BT.bodyBNodeIndex[k] = PxNodeIndex(PxU32(200 + b + k));
            }
            else
            {
                BT.bodyANodeIndex[k] = PxNodeIndex(PxU32(300 + b + k));
                BT.bodyBNodeIndex[k] = PxNodeIndex(60 + b, 3 + (k % 5));
            }
        }
        if (sfpIsContact(b))
        {
            PxgBlockSolverContactHeader& H = B.ch[b];
            PxgBlockSolverFrictionHeader& F = B.fh[b];
            for (uint32_t lane = 0; lane < 32; ++lane)
            {
                const uint32_t nc = 1u + ((lane + b) % 3u);
                H.numNormalConstr[lane] = nc;
                H.invMass0_1_angDom0_1[lane] = make_float4(0.6f + 0.2f * fv(lane, b, 60, salt), 0.5f + 0.2f * fv(lane, b, 61, salt),
                                                           0.7f + 0.2f * fv(lane, b, 62, salt), 0.8f + 0.2f * fv(lane, b, 63, salt));
                PxVec3 n(fv(lane, b, 64, salt), fv(lane, b, 65, salt), fv(lane, b, 66, salt));
                n = n.getNormalized();
                H.normal_staticFriction[lane] = make_float4(n.x, n.y, n.z, 0.6f + 0.3f * std::fabs(fv(lane, b, 67, salt)));
                H.minNormalForce[lane] = ((lane + b) % 5u == 2u) ? 2.0f + std::fabs(fv(lane, b, 68, salt))
                                                                 : 0.02f * std::fabs(fv(lane, b, 68, salt));
                H.flags[lane] = ((lane + b) % 3u == 1u) ? COMPLIANT_ACCEL : 0u;
                H.restitution[lane] = ((lane + b) % 4u == 1u) ? (-500.0f - 10.0f * float(lane))
                                                              : (0.2f + 0.1f * std::fabs(fv(lane, b, 71, salt)));
                H.cfm[lane] = 0.01f + 0.005f * std::fabs(fv(lane, b, 72, salt));
                H.accumNormalForce[lane] = fv(lane, b, 59, salt);   // NOT touched by the solve
                H.forceWritebackOffset[lane] = 0xABABABABu;         // NOT touched
                F.numFrictionConstr[lane] = ((lane + b) % 3u) * 2u; // prep invariant: EVEN
                F.dynamicFriction[lane] = 0.4f + 0.2f * std::fabs(fv(lane, b, 73, salt));
                PxVec3 f0(fv(lane, b, 76, salt), fv(lane, b, 77, salt), fv(lane, b, 78, salt));
                PxVec3 f1(fv(lane, b, 79, salt), fv(lane, b, 80, salt), fv(lane, b, 81, salt));
                f0 = f0.getNormalized(); f1 = f1.getNormalized();
                F.frictionNormals[0][lane] = make_float4(f0.x, f0.y, f0.z, 0.f);
                F.frictionNormals[1][lane] = make_float4(f1.x, f1.y, f1.z, 0.f);
                // broken stays POISONED: the doFriction=0 round must not write it
                for (uint32_t i = 0; i < 3; ++i)
                {
                    PxgBlockSolverContactPoint& C = B.cp[BT.startConstraintIndex + i];
                    C.raXn_targetVelocity[lane] = make_float4(fv(lane, b * 4 + i, 82, salt), fv(lane, b * 4 + i, 83, salt),
                                                              fv(lane, b * 4 + i, 84, salt), 0.2f * fv(lane, b * 4 + i, 85, salt));
                    C.rbXn_maxImpulse[lane] = make_float4(fv(lane, b * 4 + i, 86, salt), fv(lane, b * 4 + i, 87, salt),
                                                          fv(lane, b * 4 + i, 88, salt),
                                                          0.3f + 0.3f * std::fabs(fv(lane, b * 4 + i, 89, salt)));
                    C.appliedForce[lane] = 0.2f * std::fabs(fv(lane, b * 4 + i, 92, salt));
                    C.resp0[lane] = 0.5f + 0.3f * std::fabs(fv(lane, b * 4 + i, 94, salt));
                    C.resp1[lane] = 0.5f + 0.3f * std::fabs(fv(lane, b * 4 + i, 95, salt));
                    C.coeff0[lane] = ((lane + b) % 4u) == 1u ? 0.5f + 0.4f * std::fabs(fv(lane, b * 4 + i, 90, salt))
                                                             : 0.4f * fv(lane, b * 4 + i, 90, salt);
                    C.coeff1[lane] = 0.4f * fv(lane, b * 4 + i, 91, salt);
                }
                for (uint32_t i = 0; i < 5; ++i)
                {
                    PxgBlockSolverContactFriction& FR = B.fp[BT.startFrictionIndex + i];
                    FR.raXn_bias[lane] = make_float4(fv(lane, b * 6 + i, 96, salt), fv(lane, b * 6 + i, 97, salt),
                                                     fv(lane, b * 6 + i, 98, salt), 0.3f * fv(lane, b * 6 + i, 99, salt));
                    FR.rbXn_targetVelW[lane] = make_float4(fv(lane, b * 6 + i, 100, salt), fv(lane, b * 6 + i, 101, salt),
                                                           fv(lane, b * 6 + i, 102, salt), 0.1f * fv(lane, b * 6 + i, 103, salt));
                    FR.appliedForce[lane] = 0.2f + 0.1f * fv(lane, b * 6 + i, 104, salt);
                    FR.resp0[lane] = 0.4f + 0.3f * std::fabs(fv(lane, b * 6 + i, 105, salt));
                    FR.resp1[lane] = 0.4f + 0.3f * std::fabs(fv(lane, b * 6 + i, 106, salt));
                }
            }
        }
        else
        {
            PxgBlockSolverConstraint1DHeader& H = B.jh[b];
            for (uint32_t lane = 0; lane < 32; ++lane)
            {
                const uint32_t rc = 1u + ((lane + b) % 4u);
                H.rowCounts[lane] = PxU16(rc);
                H.breakable[lane] = PxU16(lane & 1);   // NOT read by the solve
                H.invMass0D0[lane] = 0.5f + 0.2f * fv(lane, b, 113, salt);
                H.invMass1D1[lane] = 0.5f + 0.2f * fv(lane, b, 117, salt);
                H.invInertiaScale0[lane] = 0.6f + 0.2f * fv(lane, b, 118, salt);
                H.invInertiaScale1[lane] = 0.6f + 0.2f * fv(lane, b, 119, salt);
                H.cfm[lane] = 0.01f + 0.004f * fv(lane, b, 120, salt);
                for (uint32_t i = 0; i < 5; ++i)
                {
                    PxgBlockSolverConstraint1DCon& R = B.jr[BT.startConstraintIndex + i];
                    PxgBlockSolverConstraint1DMod& M = B.jm[BT.startConstraintIndex + i];
                    const bool narrow = ((lane + i) % 4u) == 3u;
                    const float minImp = narrow ? -0.02f : -0.6f - 0.3f * std::fabs(fv(lane, b * 5 + i, 140, salt));
                    const float maxImp = narrow ? 0.02f : 0.6f + 0.3f * std::fabs(fv(lane, b * 5 + i, 141, salt));
                    R.lin0XYZ_minImpulse[lane] = make_float4(fv(lane, b * 5 + i, 121, salt), fv(lane, b * 5 + i, 122, salt),
                                                             fv(lane, b * 5 + i, 123, salt), minImp);
                    R.lin1XYZ_maxImpulse[lane] = make_float4(fv(lane, b * 5 + i, 125, salt), fv(lane, b * 5 + i, 126, salt),
                                                             fv(lane, b * 5 + i, 127, salt), maxImp);
                    R.ang0XYZ_resp0[lane] = make_float4(fv(lane, b * 5 + i, 129, salt), fv(lane, b * 5 + i, 130, salt),
                                                        fv(lane, b * 5 + i, 131, salt),
                                                        0.5f + 0.3f * std::fabs(fv(lane, b * 5 + i, 137, salt)));
                    R.ang1XYZ_resp1[lane] = make_float4(fv(lane, b * 5 + i, 133, salt), fv(lane, b * 5 + i, 134, salt),
                                                        fv(lane, b * 5 + i, 135, salt),
                                                        0.5f + 0.3f * std::fabs(fv(lane, b * 5 + i, 138, salt)));
                    R.initJointSpeed[lane] = 0.3f * fv(lane, b * 5 + i, 136, salt);
                    const uint32_t mode = (lane + b + i) % 3u;   // hard / spring / accel-spring
                    M.flags[lane] = (mode == 1u) ? DY_FLAG_SPRING
                                                 : (mode == 2u) ? (DY_FLAG_SPRING | DY_FLAG_ACCEL) : 0u;
                    M.coeff0[lane] = (mode != 0u) ? 0.5f + 0.4f * std::fabs(fv(lane, b * 5 + i, 124, salt))
                                                  : 0.1f * fv(lane, b * 5 + i, 124, salt);
                    M.coeff1[lane] = 0.3f * fv(lane, b * 5 + i, 128, salt);
                    M.appliedForce[lane] = 0.2f + 0.2f * fv(lane, b * 5 + i, 143, salt);
                    // residual stays POISONED: only reporting rounds may write it
                }
            }
        }
    }

    for (uint32_t r = 0; r < SFP_NB_RESP; ++r)
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

// ---- mirror: the DFS, one lane at a time over the hoisted
// pxgtest::refSolveInternalDFSLane (pxgArtiInternalRef.h) ----

static void sfpMirror(SfpArena& W, uint32_t flags)
{
    const bool velIter = (flags & 1u) != 0;
    const bool doFriction = (flags & 2u) != 0;
    const bool doFDPL = (flags & 4u) != 0;
    const uint32_t doVelLimit = (flags >> 3u) & 3u;
    const bool doStatic = (flags & 32u) != 0;
    const bool residual = (flags & 64u) != 0;

    for (uint32_t lane = 0; lane < SFP_LANES; ++lane)
    {
        float* fin = &W.io[lane * SFP_IO_STRIDE];
        const UnAlignedSpatialVector storeParentDeltaV(PxVec3(fin[0], fin[1], fin[2]), PxVec3(fin[3], fin[4], fin[5]));
        const UnAlignedSpatialVector parentImpIn(PxVec3(fin[6], fin[7], fin[8]), PxVec3(fin[9], fin[10], fin[11]));
        const UnAlignedSpatialVector parentDeltaVIn(PxVec3(fin[12], fin[13], fin[14]), PxVec3(fin[15], fin[16], fin[17]));

        const float* jf = &W.jf[lane * SFP_MAX_DOFS];
        RefError err, contactErr;

        const UnAlignedSpatialVector parentImp = refSolveInternalDFSLanePgs(
            W.lk.data(), W.df.data(), W.stack.data(), jf,
            W.bt.data(), W.ch.data(), W.fh.data(), W.cp.data(), W.fp.data(),
            W.jh.data(), W.jr.data(), W.jm.data(), W.rs.data(),
            SFP_DT, SFP_SCENE_DT, SFP_INV_DT, SFP_ELAPSED, SFP_ERP,
            velIter, doFriction, doFDPL, doVelLimit, doStatic, residual,
            storeParentDeltaV, parentImpIn, parentDeltaVIn,
            err, contactErr, lane);

        float* out = &W.io[SFP_IO_OUT + lane * SFP_IO_STRIDE];
        out[0] = parentImp.top.x; out[1] = parentImp.top.y; out[2] = parentImp.top.z;
        out[3] = parentImp.bottom.x; out[4] = parentImp.bottom.y; out[5] = parentImp.bottom.z;
        out[6] = err.sumOfSquares; out[7] = float(err.counter); out[8] = err.maxError;
        out[9] = contactErr.sumOfSquares; out[10] = float(contactErr.counter); out[11] = contactErr.maxError;
    }
}

// ---- run ----

struct SfpParamsHost
{
    uint64_t linkBase, dofBlocks, stackBase, jointForces;
    uint64_t batches, ch, fh, cp, fp, jh, jr, jm, rs, io;
    float dt, sceneDt, invDt, elapsedTime, erp;
    uint32_t flagsWord, nbLanes, maxDofs;
};
static_assert(sizeof(SfpParamsHost) == 144, "params blob");

static size_t sfpRun(PxgCudaShim& shim, uint32_t flags, uint32_t salt)
{
    SfpArena A;
    sfpSeed(A, salt);

    SfpArena W = A;
    sfpMirror(W, flags);
    PXG_CHECK_MSG(std::memcmp(W.stack.data(), A.stack.data(),
                              W.stack.size() * sizeof(W.stack[0])) != 0, "vacuous: stack untouched");

    std::vector<PxgDevicePtr> bufs;
    auto up = [&](const void* p, size_t n) {
        PxgDevicePtr dp = 0;
        shim.memAlloc(&dp, n);
        if (dp && p) shim.memcpyHtoD(dp, p, n);
        bufs.push_back(dp);
        return dp;
    };
    PxgDevicePtr dLk = up(A.lk.data(), A.lk.size() * sizeof(A.lk[0]));
    PxgDevicePtr dDf = up(A.df.data(), A.df.size() * sizeof(A.df[0]));
    PxgDevicePtr dSt = up(A.stack.data(), A.stack.size() * sizeof(A.stack[0]));
    PxgDevicePtr dJf = up(A.jf.data(), A.jf.size() * 4);
    PxgDevicePtr dBt = up(A.bt.data(), A.bt.size() * sizeof(A.bt[0]));
    PxgDevicePtr dCh = up(A.ch.data(), A.ch.size() * sizeof(A.ch[0]));
    PxgDevicePtr dFh = up(A.fh.data(), A.fh.size() * sizeof(A.fh[0]));
    PxgDevicePtr dCp = up(A.cp.data(), A.cp.size() * sizeof(A.cp[0]));
    PxgDevicePtr dFp = up(A.fp.data(), A.fp.size() * sizeof(A.fp[0]));
    PxgDevicePtr dJh = up(A.jh.data(), A.jh.size() * sizeof(A.jh[0]));
    PxgDevicePtr dJr = up(A.jr.data(), A.jr.size() * sizeof(A.jr[0]));
    PxgDevicePtr dJm = up(A.jm.data(), A.jm.size() * sizeof(A.jm[0]));
    PxgDevicePtr dRs = up(A.rs.data(), A.rs.size() * sizeof(A.rs[0]));
    PxgDevicePtr dIo = up(A.io.data(), A.io.size() * 4);
    bool ok = true;
    for (PxgDevicePtr p : bufs) ok = ok && (p != 0);
    if (!ok)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
        return 0;
    }

    SfpParamsHost prm = { uint64_t(dLk), uint64_t(dDf), uint64_t(dSt), uint64_t(dJf),
                         uint64_t(dBt), uint64_t(dCh), uint64_t(dFh), uint64_t(dCp),
                         uint64_t(dFp), uint64_t(dJh), uint64_t(dJr), uint64_t(dJm),
                         uint64_t(dRs), uint64_t(dIo),
                         SFP_DT, SFP_SCENE_DT, SFP_INV_DT, SFP_ELAPSED, SFP_ERP,
                         flags, SFP_LANES, SFP_MAX_DOFS };
    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch DFS pgs");
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
            std::printf("MM dfsPgs/%s: %zu/%zu bytes differ (first at %zu)\n", name, mm, n, first);
            bad += mm;
        }
    };
    cmp(dLk, W.lk.data(), W.lk.size() * sizeof(W.lk[0]), "linkBlocks");
    cmp(dDf, W.df.data(), W.df.size() * sizeof(W.df[0]), "dofBlocks");
    cmp(dSt, W.stack.data(), W.stack.size() * sizeof(W.stack[0]), "stack");
    cmp(dJf, W.jf.data(), W.jf.size() * 4, "jointForces");
    cmp(dBt, W.bt.data(), W.bt.size() * sizeof(W.bt[0]), "batches");
    cmp(dCh, W.ch.data(), W.ch.size() * sizeof(W.ch[0]), "contactHeaders");
    cmp(dFh, W.fh.data(), W.fh.size() * sizeof(W.fh[0]), "frictionHeaders");
    cmp(dCp, W.cp.data(), W.cp.size() * sizeof(W.cp[0]), "contactPoints");
    cmp(dFp, W.fp.data(), W.fp.size() * sizeof(W.fp[0]), "frictions");
    cmp(dJh, W.jh.data(), W.jh.size() * sizeof(W.jh[0]), "jointHeaders");
    cmp(dJr, W.jr.data(), W.jr.size() * sizeof(W.jr[0]), "jointRows");
    cmp(dJm, W.jm.data(), W.jm.size() * sizeof(W.jm[0]), "jointMods");
    cmp(dRs, W.rs.data(), W.rs.size() * sizeof(W.rs[0]), "responses");
    cmp(dIo, W.io.data(), W.io.size() * 4, "io");

    for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
    return bad;
}

} // namespace

PXG_TEST(artiSolveDFSPgs)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiSolveInternal.spv", "siDfsPgsTest", 144u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &sfpLoader);

    size_t bad = 0;
    bad += sfpRun(shim, 2u | 4u | 8u | 32u | 64u, 61);       // posIter, doF, BEFORE, static, residual
    bad += sfpRun(shim, 1u | 2u | 4u | 16u | 32u | 64u, 67); // velIter, doF, AFTER, static, residual
    bad += sfpRun(shim, 2u | 4u | 8u | 64u, 71);             // doF, BEFORE, no static
    bad += sfpRun(shim, 4u | 32u, 73);                       // doFriction OFF, static, no residual
    PXG_CHECK_MSG(bad == 0, "%zu mismatches", bad);
}
