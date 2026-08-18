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

// P8.98c: `artiSolveInternalTendonAndMimicJointConstraints1T`
// (internalConstraints2.cu:2775-2815), driven over the REAL engine ABI — a
// device-resident PxgArticulationCoreDesc, the (32, 1, 1) x
// ceil(nbArticulations / 32) launch shape, and the
// {ptr, float, float, float, bool, bool} argument pack with the two bools as
// adjacent bytes at 20/21.
//
// THE ORACLE: the FULL wrapper logic mirrored on the host — `aiMirror`
// (P8.94c, pxgArtiRefHelpers.h) for the eHAS_IMPULSES averaging preamble and
// the three hoisted refSolve* bodies (P8.98b, pxgArtiSolveRef.h), each
// already pinned by its own fixture.  This fixture is a USER of both helper
// headers, not a transcription.
//
// WHAT THE FIXTURE IS SHAPED TO SEE
//  * mStateDirty takes 0, 1, 2 and 3 across lanes — 2/3 run the averaging
//    preamble, 0/1 skip it, and ALL guarded lanes end at eVEL_DIRTY (1);
//  * every seventh lane has NO tendons and NO mimic joints — the outer guard
//    skips it entirely and its mStateDirty byte must come back UNTOUCHED even
//    though its word-mates were rewritten (the sub-word store hazard);
//  * dirty slab pairs with x-only / y-only / both, impulse entries on both
//    16- and 8-byte alignments (counted and required — the aiMirror branch);
//  * two rounds crossing the bool bytes: (velIter=0, isTGS=1) and
//    (velIter=1, isTGS=0) — a byte swap in the pack breaks both;
//  * everything else is the P8.98b arena (branch topology, both tendon
//    families, mimic compliance cases), compared byte for byte.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "foundation/PxQuat.h"
#include "foundation/PxVec3.h"
#include "foundation/PxTransform.h"
#include "CmSpatialVector.h"
#include "DyFeatherstoneArticulationUtils.h"
#include "DyFeatherstoneArticulation.h"
#include "DyCpuGpuArticulation.h"
#include "PxSpatialMatrix.h"
#include "PxgArticulation.h"
#include "PxgArticulationCoreDesc.h"
#include "pxgArtiRefHelpers.h"
#include "pxgArtiSolveRef.h"

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

static std::vector<uint8_t> tmLoader(const char* file) { return loadShaderBlob(file); }

const uint32_t TM_NB = 68, TM_WARPS = 3;
const uint32_t TM_MAX_LINKS = 6, TM_MAX_DOFS = 8, TM_WORDSIZE = 1;
const uint32_t TM_MAX_ST = 2, TM_MAX_AT = 5;
const uint32_t TM_MAX_FT = 2, TM_MAX_TJ = 4;
const uint32_t TM_MAX_MJ = 2;
const uint32_t TM_NB_SLABS = 3;
const float TM_BIAS = 0.7f, TM_DT = 0.0166f, TM_RECIP_DT = 60.24f;

static float fv(uint32_t a, uint32_t l, uint32_t d, uint32_t salt)
{
    uint32_t h = a * 2654435761u + l * 40503u + d * 9973u + salt * 733u + 19u;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    return -1.0f + 2.0f * float(h & 0xFFFFFFu) / float(0x1000000);
}
static PxQuat qv(uint32_t a, uint32_t l, uint32_t salt)
{
    PxQuat q(fv(a, l, 0, salt), fv(a, l, 1, salt), fv(a, l, 2, salt),
             1.0f + 0.4f * fv(a, l, 3, salt));
    return q.getNormalized();
}

static uint32_t tmNumLinks(uint32_t a) { return 4u + (a % 3u); }
static uint32_t tmParent(uint32_t l)
{
    static const uint32_t p[6] = { 0, 0, 1, 1, 2, 3 };
    return p[l];
}
static uint32_t tmDofs(uint32_t a, uint32_t l) { return 1u + ((a + l) % 2u); }
static bool tmGuardOff(uint32_t a) { return (a % 7u) == 3u; }   // no tendons, no mimic

struct TmArena
{
    std::vector<PxgArticulationBlockData> ab;
    std::vector<PxgArticulationBlockLinkData> lk;
    std::vector<PxgArticulationBlockDofData> df;
    std::vector<uint64_t> bf;
    std::vector<PxgArticulationBlockSpatialTendonData> st;
    std::vector<PxgArticulationInternalTendonConstraintData> itcS;
    std::vector<PxgArticulationBlockAttachmentData> at;
    std::vector<PxgArticulationBlockFixedTendonData> ft;
    std::vector<PxgArticulationInternalTendonConstraintData> itcF;
    std::vector<PxgArticulationBlockTendonJointData> tj;
    std::vector<PxgArticulationBlockMimicJointData> mjb;
    std::vector<uint32_t> dirty;    // uint2 per (slab, articulation)
    std::vector<float> imp;         // 6 floats per (slab, link, articulation)
};

static void tmSeed(TmArena& B)
{
    B.ab.resize(TM_WARPS);
    B.lk.resize(size_t(TM_WARPS) * TM_MAX_LINKS);
    B.df.resize(size_t(TM_WARPS) * TM_MAX_DOFS);
    B.bf.resize(size_t(TM_WARPS) * TM_MAX_LINKS * TM_WORDSIZE * 32);
    B.st.resize(size_t(TM_WARPS) * TM_MAX_ST);
    B.itcS.resize(size_t(TM_WARPS) * TM_MAX_ST * TM_MAX_AT);
    B.at.resize(size_t(TM_WARPS) * TM_MAX_ST * TM_MAX_AT);
    B.ft.resize(size_t(TM_WARPS) * TM_MAX_FT);
    B.itcF.resize(size_t(TM_WARPS) * TM_MAX_FT * TM_MAX_TJ);
    B.tj.resize(size_t(TM_WARPS) * TM_MAX_FT * TM_MAX_TJ);
    B.mjb.resize(size_t(TM_WARPS) * TM_MAX_MJ);
    std::memset(B.ab.data(), 0xAB, B.ab.size() * sizeof(B.ab[0]));
    std::memset(B.lk.data(), 0xAB, B.lk.size() * sizeof(B.lk[0]));
    std::memset(B.df.data(), 0xAB, B.df.size() * sizeof(B.df[0]));
    std::memset(B.bf.data(), 0xAB, B.bf.size() * 8);
    std::memset(B.st.data(), 0xAB, B.st.size() * sizeof(B.st[0]));
    std::memset(B.itcS.data(), 0xAB, B.itcS.size() * sizeof(B.itcS[0]));
    std::memset(B.at.data(), 0xAB, B.at.size() * sizeof(B.at[0]));
    std::memset(B.ft.data(), 0xAB, B.ft.size() * sizeof(B.ft[0]));
    std::memset(B.itcF.data(), 0xAB, B.itcF.size() * sizeof(B.itcF[0]));
    std::memset(B.tj.data(), 0xAB, B.tj.size() * sizeof(B.tj[0]));
    std::memset(B.mjb.data(), 0xAB, B.mjb.size() * sizeof(B.mjb[0]));

    // the averaging inputs: dirty pairs default to none, impulses hash-seeded
    B.dirty.assign(size_t(TM_NB_SLABS) * TM_NB * 2, pxgref::AI_NONE);
    B.imp.resize(size_t(TM_NB_SLABS) * TM_MAX_LINKS * TM_NB * 6);
    for (size_t k = 0; k < B.imp.size(); ++k)
        B.imp[k] = fv(uint32_t(k % 997u), uint32_t(k / 997u), 0, 99) * 0.015625f;

    for (uint32_t a = 0; a < TM_NB; ++a)
    {
        const uint32_t w = a / 32, lane = a % 32;
        PxgArticulationBlockData& AB = B.ab[w];
        PxgArticulationBlockLinkData* L = &B.lk[size_t(w) * TM_MAX_LINKS];
        PxgArticulationBlockDofData* D = &B.df[size_t(w) * TM_MAX_DOFS];
        uint64_t* BF = &B.bf[size_t(w) * TM_MAX_LINKS * TM_WORDSIZE * 32];

        const uint32_t numLinks = tmNumLinks(a);
        AB.mNumLinks[lane] = numLinks;
        AB.mFlags[lane] = (a & 1u);
        const bool guardOff = tmGuardOff(a);
        AB.mNumSpatialTendons[lane] = guardOff ? 0 : (numLinks >= 5 ? 2 : 1);
        AB.mNumFixedTendons[lane] = guardOff ? 0 : 1;
        AB.mNumMimicJoints[lane] = guardOff ? 0 : (numLinks >= 5 ? 2 : 1);
        AB.mStateDirty[lane] = PxU8(a % 4u);
        AB.mLinkWithDeferredImpulse[lane] = 1u + (a % (numLinks - 1u));
        storeSpatialVector(AB.mRootDeferredZ,
                           UnAlignedSpatialVector(
                               PxVec3(fv(a, 30, 0, 80), fv(a, 30, 1, 80), fv(a, 30, 2, 80)),
                               PxVec3(fv(a, 30, 3, 80), fv(a, 30, 4, 80), fv(a, 30, 5, 80))),
                           lane);
        {
            Dy::SpatialMatrix m;
            m.topLeft = PxMat33(PxZero);
            m.topRight = PxMat33(PxZero);
            m.bottomLeft = PxMat33(PxZero);
            for (uint32_t c = 0; c < 3; ++c)
                for (uint32_t r = 0; r < 3; ++r)
                {
                    m.topLeft[c][r] = 0.15f * fv(a, 31, c * 3 + r, 81) + (c == r ? 0.8f : 0.f);
                    m.topRight[c][r] = 0.15f * fv(a, 32, c * 3 + r, 82) + (c == r ? 0.9f : 0.f);
                    m.bottomLeft[c][r] = 0.15f * fv(a, 33, c * 3 + r, 83) + (c == r ? 0.7f : 0.f);
                }
            storeSpatialMatrix(AB.mInvSpatialArticulatedInertia, lane, m);
        }

        // dirty slabs only where the preamble will read them (state 2/3)
        if ((a % 4u) >= 2u)
        {
            for (uint32_t s = 0; s < TM_NB_SLABS; ++s)
            {
                const size_t di = size_t(s) * TM_NB + a;
                const uint32_t mode = (a + s) % 3u;
                const uint32_t x = 1u + ((a + s) % (numLinks - 1u));
                const uint32_t y = 1u + ((a + s + 1u) % (numLinks - 1u));
                if (mode != 1u) B.dirty[di * 2 + 0] = x;
                if (mode != 0u) B.dirty[di * 2 + 1] = y;
            }
        }

        uint32_t dofCursor = 0;
        for (uint32_t l = 0; l < numLinks; ++l)
        {
            PxgArticulationBlockLinkData& lnk = L[l];

            uint64_t path = 1ull << 0;
            for (uint32_t p = l; p != 0; p = tmParent(p))
                path |= 1ull << p;
            BF[(l * TM_WORDSIZE + 0) * 32 + lane] = path;

            const PxQuat accQ = qv(a, l, 31);
            lnk.mAccumulatedPose.q[lane] = make_float4(accQ.x, accQ.y, accQ.z, accQ.w);
            lnk.mAccumulatedPose.p[lane] = make_float4(2.0f * fv(a, l, 0, 32),
                                                       2.0f * fv(a, l, 1, 32),
                                                       2.0f * fv(a, l, 2, 32), 0.f);
            storeSpatialVector(lnk.mMotionVelocity,
                UnAlignedSpatialVector(
                    PxVec3(fv(a, l, 0, 84), fv(a, l, 1, 84), fv(a, l, 2, 84)),
                    PxVec3(fv(a, l, 3, 84), fv(a, l, 4, 84), fv(a, l, 5, 84))), lane);
            storeSpatialVector(lnk.mSolverSpatialImpulse,
                UnAlignedSpatialVector(
                    PxVec3(fv(a, l, 0, 85), fv(a, l, 1, 85), fv(a, l, 2, 85)),
                    PxVec3(fv(a, l, 3, 85), fv(a, l, 4, 85), fv(a, l, 5, 85))), lane);
            storeSpatialVector(lnk.mScratchImpulse,
                UnAlignedSpatialVector(
                    PxVec3(fv(a, l, 0, 79), fv(a, l, 1, 79), fv(a, l, 2, 79)),
                    PxVec3(fv(a, l, 3, 79), fv(a, l, 4, 79), fv(a, l, 5, 79))), lane);

            if (l == 0)
            {
                lnk.mJointOffset[lane] = 0;
                lnk.mDofs[lane] = 0;
                lnk.mRw_x[lane] = 7.5f;
                lnk.mRw_y[lane] = -3.25f;
                lnk.mRw_z[lane] = 9.0f;
                continue;
            }
            lnk.mParents[lane] = tmParent(l);
            lnk.mRw_x[lane] = 0.5f * fv(a, l, 0, 86);
            lnk.mRw_y[lane] = 0.5f * fv(a, l, 1, 86);
            lnk.mRw_z[lane] = 0.5f * fv(a, l, 2, 86);
            const uint32_t nbDofs = tmDofs(a, l);
            lnk.mJointOffset[lane] = dofCursor;
            lnk.mDofs[lane] = PxU8(nbDofs);
            for (uint32_t ax = 0; ax < 6; ++ax)
                lnk.mInvDofIds[ax][lane] = 0;
            for (uint32_t d = 0; d < nbDofs; ++d)
            {
                lnk.mInvDofIds[(l + d) % 6u][lane] = PxU8(d);
                PxgArticulationBlockDofData& dd = D[dofCursor + d];
                storeSpatialVector(dd.mWorldMotionMatrix,
                    UnAlignedSpatialVector(
                        PxVec3(fv(a, l, d, 87), fv(a, l, d + 3, 87), fv(a, l, d + 6, 87)),
                        PxVec3(fv(a, l, d, 88), fv(a, l, d + 3, 88), fv(a, l, d + 6, 88))), lane);
                storeSpatialVector(dd.mIsInvDW,
                    UnAlignedSpatialVector(
                        PxVec3(fv(a, l, d, 89), fv(a, l, d + 3, 89), fv(a, l, d + 6, 89)),
                        PxVec3(fv(a, l, d, 90), fv(a, l, d + 3, 90), fv(a, l, d + 6, 90))), lane);
                storeSpatialVector(dd.mIsW,
                    UnAlignedSpatialVector(
                        PxVec3(fv(a, l, d, 91), fv(a, l, d + 3, 91), fv(a, l, d + 6, 91)),
                        PxVec3(fv(a, l, d, 92), fv(a, l, d + 3, 92), fv(a, l, d + 6, 92))), lane);
                dd.mInvStIsT_x[lane] = fv(a, l, d, 93);
                dd.mInvStIsT_y[lane] = fv(a, l, d, 94);
                dd.mInvStIsT_z[lane] = fv(a, l, d, 95);
                dd.mDeferredQstZ[lane] = 0.6f * fv(a, l, d, 96);
                dd.mJointVelocities[lane] = fv(a, l, d, 97);
                dd.mJointPositions[lane] = 0.8f * fv(a, l, d, 98);
            }
            dofCursor += nbDofs;
        }
        AB.mTotalDofs[lane] = dofCursor;

        if (guardOff)
            continue;   // tendon/mimic blocks stay poisoned for guarded-off lanes

        // ---- spatial tendons (the P8.98b shapes) ----
        PxgArticulationBlockSpatialTendonData* ST = &B.st[size_t(w) * TM_MAX_ST];
        PxgArticulationInternalTendonConstraintData* IS =
            &B.itcS[size_t(w) * TM_MAX_ST * TM_MAX_AT];
        PxgArticulationBlockAttachmentData* AT = &B.at[size_t(w) * TM_MAX_ST * TM_MAX_AT];
        for (uint32_t t = 0; t < AB.mNumSpatialTendons[lane]; ++t)
        {
            ST[t].mOffset[lane] = 0.4f * fv(a, t, 3, 50);
            ST[t].mNumConstraints[lane] = 2;
            PxgArticulationBlockAttachmentData* T = &AT[t * TM_MAX_AT];
            auto att = [&](uint32_t idx, uint32_t linkIndex, uint32_t parent, uint64_t children) {
                T[idx].mLinkIndex[lane] = PxU16(linkIndex);
                T[idx].mParents[lane] = parent;
                T[idx].mChildrens[lane] = children;
                T[idx].mRelativeOffset[lane] = PxVec3(0.3f * fv(a, t * 8 + idx, 0, 51),
                                                      0.3f * fv(a, t * 8 + idx, 1, 51),
                                                      0.3f * fv(a, t * 8 + idx, 2, 51));
                T[idx].mCoefficient[lane] = 0.6f + 0.3f * fv(a, t * 8 + idx, 3, 51);
            };
            att(0, 0, 0, (1ull << 1) | (1ull << 2));
            att(1, 1, 0, 1ull << 3);
            att(2, 2, 0, 0);
            att(3, 3, 1, 0);
            const uint32_t leafLink[2] = { 2, 3 };
            for (uint32_t cix = 0; cix < 2; ++cix)
            {
                PxgArticulationInternalTendonConstraintData& C = IS[t * TM_MAX_AT + cix];
                C.mLink0[lane] = 0;
                C.mLink1[lane] = leafLink[cix];
                storeSpatialVector(C.mRow0,
                    UnAlignedSpatialVector(
                        PxVec3(fv(a, t * 4 + cix, 0, 52), fv(a, t * 4 + cix, 1, 52), fv(a, t * 4 + cix, 2, 52)),
                        PxVec3(fv(a, t * 4 + cix, 3, 52), fv(a, t * 4 + cix, 4, 52), fv(a, t * 4 + cix, 5, 52))), lane);
                storeSpatialVector(C.mRow1,
                    UnAlignedSpatialVector(
                        PxVec3(fv(a, t * 4 + cix, 0, 53), fv(a, t * 4 + cix, 1, 53), fv(a, t * 4 + cix, 2, 53)),
                        PxVec3(fv(a, t * 4 + cix, 3, 53), fv(a, t * 4 + cix, 4, 53), fv(a, t * 4 + cix, 5, 53))), lane);
                C.mDeltaVA[lane] = 0.2f * fv(a, t * 4 + cix, 6, 54);
                C.mAccumulatedLength[lane] = 0.5f + 0.6f * fv(a, t * 4 + cix, 7, 54);
                C.mRestDistance[lane] = fv(a, t * 4 + cix, 8, 54);
                C.mLowLimit[lane] = ((a % 5u) == 0u)
                    ? 3.5f + fv(a, t * 4 + cix, 9, 54)
                    : 0.1f + 0.3f * fv(a, t * 4 + cix, 9, 54);
                C.mHighLimit[lane] = C.mLowLimit[lane] + 0.35f
                                   + 0.3f * std::fabs(fv(a, t * 4 + cix, 10, 54));
                C.mVelMultiplier[lane] = 0.5f + 0.4f * fv(a, t * 4 + cix, 11, 54);
                C.mBiasCoefficient[lane] = 0.4f * fv(a, t * 4 + cix, 12, 54);
                C.mImpulseMultiplier[lane] = 0.85f + 0.1f * fv(a, t * 4 + cix, 13, 54);
                C.mAppliedForce[lane] = 0.3f * fv(a, t * 4 + cix, 14, 54);
                C.mLimitBiasCoefficient[lane] = 0.4f * fv(a, t * 4 + cix, 15, 54);
                C.mLimitImpulseMultiplier[lane] = 0.8f + 0.15f * fv(a, t * 4 + cix, 16, 54);
                C.mLimitAppliedForce[lane] = 0.25f * fv(a, t * 4 + cix, 17, 54);
            }
        }

        // ---- fixed tendon ----
        PxgArticulationBlockFixedTendonData* FT = &B.ft[size_t(w) * TM_MAX_FT];
        PxgArticulationInternalTendonConstraintData* IF =
            &B.itcF[size_t(w) * TM_MAX_FT * TM_MAX_TJ];
        PxgArticulationBlockTendonJointData* TJ = &B.tj[size_t(w) * TM_MAX_FT * TM_MAX_TJ];
        {
            FT[0].mOffset[lane] = 0.3f * fv(a, 9, 0, 55);
            FT[0].mLowLimit[lane] = -0.4f + 0.3f * fv(a, 9, 1, 55);
            FT[0].mHighLimit[lane] = FT[0].mLowLimit[lane] + 0.5f
                                   + 0.4f * std::fabs(fv(a, 9, 2, 55));
            FT[0].mRestLength[lane] = 0.2f * fv(a, 9, 3, 55);
            const bool branch = numLinks >= 5;
            auto tjf = [&](uint32_t idx, uint32_t linkIndex, uint32_t axis, uint64_t children,
                           uint32_t cid) {
                TJ[idx].mLinkIndex[lane] = PxU16(linkIndex);
                TJ[idx].mAxis[lane] = axis;
                TJ[idx].mChildrens[lane] = children;
                TJ[idx].mConstraintId[lane] = cid;
                TJ[idx].mCoefficient[lane] = 0.5f + 0.4f * fv(a, 40 + idx, 0, 56);
                TJ[idx].mRecipCoefficient[lane] = 0.6f + 0.3f * fv(a, 40 + idx, 1, 56);
            };
            tjf(0, 1, (1u + 0) % 6u, (1ull << 1) | (1ull << 2), 0);
            tjf(1, 2, (2u + 0) % 6u, branch ? (1ull << 3) : 0ull, 1);
            tjf(2, 3, (3u + 0) % 6u, 0, 2);
            if (branch)
                tjf(3, 4, (4u + 0) % 6u, 0, 3);
            const uint32_t nbCons = branch ? 4u : 3u;
            for (uint32_t cix = 0; cix < nbCons; ++cix)
            {
                PxgArticulationInternalTendonConstraintData& C = IF[cix];
                storeSpatialVector(C.mRow0,
                    UnAlignedSpatialVector(
                        PxVec3(fv(a, 50 + cix, 0, 57), fv(a, 50 + cix, 1, 57), fv(a, 50 + cix, 2, 57)),
                        PxVec3(fv(a, 50 + cix, 3, 57), fv(a, 50 + cix, 4, 57), fv(a, 50 + cix, 5, 57))), lane);
                storeSpatialVector(C.mRow1,
                    UnAlignedSpatialVector(
                        PxVec3(fv(a, 50 + cix, 0, 58), fv(a, 50 + cix, 1, 58), fv(a, 50 + cix, 2, 58)),
                        PxVec3(fv(a, 50 + cix, 3, 58), fv(a, 50 + cix, 4, 58), fv(a, 50 + cix, 5, 58))), lane);
                C.mVelMultiplier[lane] = 0.5f + 0.4f * fv(a, 50 + cix, 6, 59);
                C.mBiasCoefficient[lane] = 0.4f * fv(a, 50 + cix, 7, 59);
                C.mImpulseMultiplier[lane] = 0.85f + 0.1f * fv(a, 50 + cix, 8, 59);
                C.mAppliedForce[lane] = 0.3f * fv(a, 50 + cix, 9, 59);
                C.mLimitBiasCoefficient[lane] = 0.4f * fv(a, 50 + cix, 10, 59);
                C.mLimitImpulseMultiplier[lane] = 0.8f + 0.15f * fv(a, 50 + cix, 11, 59);
                C.mLimitAppliedForce[lane] = 0.25f * fv(a, 50 + cix, 12, 59);
            }
        }

        // ---- mimic joints ----
        PxgArticulationBlockMimicJointData* M = &B.mjb[size_t(w) * TM_MAX_MJ];
        auto mj = [&](uint32_t idx, uint32_t la, uint32_t lb) {
            M[idx].mLinkA[lane] = la;
            M[idx].mLinkB[lane] = lb;
            M[idx].mInternalData.mDofA[lane] = tmDofs(a, la) - 1u;
            M[idx].mInternalData.mDofB[lane] = tmDofs(a, lb) - 1u;
            M[idx].mInternalData.recipEffectiveInertia[lane] =
                0.5f + 0.4f * std::fabs(fv(a, 60 + idx, 0, 60));
            M[idx].mGearRatio[lane] = 0.8f + 0.5f * fv(a, 60 + idx, 1, 60);
            M[idx].mOffset[lane] = 0.3f * fv(a, 60 + idx, 2, 60);
            const uint32_t mode = (a + idx) % 4u;
            M[idx].mNaturalFrequency[lane] =
                (mode == 1u) ? -2.0f : 4.0f + 2.0f * std::fabs(fv(a, 60 + idx, 3, 60));
            M[idx].mDampingRatio[lane] =
                (mode == 2u) ? 0.0f : 0.5f + 0.4f * std::fabs(fv(a, 60 + idx, 4, 60));
        };
        mj(0, 2, 3);
        if (numLinks >= 5)
            mj(1, 3, 4);
    }
}

static size_t tmRun(PxgCudaShim& shim, bool velIter, bool isTGS)
{
    TmArena A;
    tmSeed(A);

    // ---- the FULL wrapper mirror ----
    TmArena W = A;
    uint32_t alignedHits = 0, unalignedHits = 0;
    uint32_t nPreamble = 0, nNoPreamble = 0, nGuardOff = 0;
    PxU32 c0 = 0, c1 = 0, c2 = 0, c3 = 0, c4 = 0, c5 = 0;
    for (uint32_t a = 0; a < TM_NB; ++a)
    {
        const uint32_t w = a / 32, lane = a % 32;
        PxgArticulationBlockData& ab = W.ab[w];
        PxgArticulationBlockLinkData* lk = &W.lk[size_t(w) * TM_MAX_LINKS];
        PxgArticulationBlockDofData* df = &W.df[size_t(w) * TM_MAX_DOFS];
        const uint64_t* bf = &W.bf[size_t(w) * TM_MAX_LINKS * TM_WORDSIZE * 32];

        if (ab.mNumSpatialTendons[lane] == 0 && ab.mNumFixedTendons[lane] == 0
            && ab.mNumMimicJoints[lane] == 0)
        {
            ++nGuardOff;
            continue;   // the outer guard: NOTHING happens, mStateDirty included
        }

        if (ab.mStateDirty[lane] & 2u)
        {
            ++nPreamble;
            pxgref::aiMirror(W.dirty.data(), W.imp.data(), &ab, lk, df, a, TM_MAX_LINKS,
                             TM_NB, TM_NB_SLABS, lane, alignedHits, unalignedHits);
        }
        else
            ++nNoPreamble;

        ab.mStateDirty[lane] = 1u;   // eVEL_DIRTY

        refSolveSpatial(ab, lk, df, bf, TM_WORDSIZE, &W.st[size_t(w) * TM_MAX_ST],
                        &W.itcS[size_t(w) * TM_MAX_ST * TM_MAX_AT],
                        &W.at[size_t(w) * TM_MAX_ST * TM_MAX_AT], TM_MAX_AT, isTGS, lane,
                        c0, c1, c2, c3);
        refSolveFixed(ab, lk, df, bf, TM_WORDSIZE, &W.ft[size_t(w) * TM_MAX_FT],
                      &W.itcF[size_t(w) * TM_MAX_FT * TM_MAX_TJ],
                      &W.tj[size_t(w) * TM_MAX_FT * TM_MAX_TJ], TM_MAX_TJ, lane,
                      c0, c1, c2);
        refSolveMimic(ab, lk, df, bf, TM_WORDSIZE, &W.mjb[size_t(w) * TM_MAX_MJ],
                      TM_BIAS, TM_DT, TM_RECIP_DT, velIter, lane, c4, c5);
    }
    std::printf("[ info ] tm(%s,%s): preamble %u, no-preamble %u, guard-off %u, "
                "impulse aligns %u/%u\n",
                velIter ? "vel" : "pos", isTGS ? "TGS" : "PGS", nPreamble, nNoPreamble,
                nGuardOff, alignedHits, unalignedHits);
    PXG_CHECK_MSG(nPreamble > 10 && nNoPreamble > 10 && nGuardOff > 5,
                  "wrapper gate coverage hole");
    PXG_CHECK_MSG(alignedHits > 5 && unalignedHits > 5,
                  "both impulse alignments must occur");

    // ---- device ----
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
    PxgDevicePtr dSt = up(A.st.data(), A.st.size() * sizeof(A.st[0]));
    PxgDevicePtr dIs = up(A.itcS.data(), A.itcS.size() * sizeof(A.itcS[0]));
    PxgDevicePtr dAt = up(A.at.data(), A.at.size() * sizeof(A.at[0]));
    PxgDevicePtr dFt = up(A.ft.data(), A.ft.size() * sizeof(A.ft[0]));
    PxgDevicePtr dIf = up(A.itcF.data(), A.itcF.size() * sizeof(A.itcF[0]));
    PxgDevicePtr dTj = up(A.tj.data(), A.tj.size() * sizeof(A.tj[0]));
    PxgDevicePtr dMj = up(A.mjb.data(), A.mjb.size() * sizeof(A.mjb[0]));
    PxgDevicePtr dDirty = up(A.dirty.data(), A.dirty.size() * 4);
    PxgDevicePtr dImp = up(A.imp.data(), A.imp.size() * 4);
    PxgDevicePtr dDesc = 0;
    shim.memAlloc(&dDesc, sizeof(PxgArticulationCoreDesc));
    bufs.push_back(dDesc);
    if (!dAb || !dLk || !dDf || !dBf || !dSt || !dIs || !dAt || !dFt || !dIf || !dTj
        || !dMj || !dDirty || !dImp || !dDesc)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
        return 0;
    }

    PxgArticulationCoreDesc desc;
    std::memset(&desc, 0xAB, sizeof(desc));
    desc.nbArticulations = TM_NB;
    desc.nbSlabs = TM_NB_SLABS;
    desc.mMaxLinksPerArticulation = TM_MAX_LINKS;
    desc.mMaxDofsPerArticulation = TM_MAX_DOFS;
    desc.mMaxSpatialTendonsPerArticulation = TM_MAX_ST;
    desc.mMaxAttachmentPerArticulation = TM_MAX_AT;
    desc.mMaxFixedTendonsPerArticulation = TM_MAX_FT;
    desc.mMaxTendonJointPerArticulation = TM_MAX_TJ;
    desc.mMaxMimicJointsPerArticulation = TM_MAX_MJ;
    desc.mArticulationBlocks = reinterpret_cast<PxgArticulationBlockData*>(dAb);
    desc.mArticulationLinkBlocks = reinterpret_cast<PxgArticulationBlockLinkData*>(dLk);
    desc.mArticulationDofBlocks = reinterpret_cast<PxgArticulationBlockDofData*>(dDf);
    desc.mPathToRootBitFieldBlocks = reinterpret_cast<PxgArticulationBitFieldData*>(dBf);
    desc.mArticulationSpatialTendonBlocks =
        reinterpret_cast<PxgArticulationBlockSpatialTendonData*>(dSt);
    desc.mArticulationSpatialTendonConstraintBlocks =
        reinterpret_cast<PxgArticulationInternalTendonConstraintData*>(dIs);
    desc.mArticulationAttachmentBlocks =
        reinterpret_cast<PxgArticulationBlockAttachmentData*>(dAt);
    desc.mArticulationFixedTendonBlocks =
        reinterpret_cast<PxgArticulationBlockFixedTendonData*>(dFt);
    desc.mArticulationFixedTendonConstraintBlocks =
        reinterpret_cast<PxgArticulationInternalTendonConstraintData*>(dIf);
    desc.mArticulationTendonJointBlocks =
        reinterpret_cast<PxgArticulationBlockTendonJointData*>(dTj);
    desc.mArticulationMimicJointBlocks =
        reinterpret_cast<PxgArticulationBlockMimicJointData*>(dMj);
    desc.slabHasChanges = reinterpret_cast<uint2*>(dDirty);
    desc.impulses = reinterpret_cast<UnAlignedSpatialVector*>(dImp);
    shim.memcpyHtoD(dDesc, &desc, sizeof(desc));

    struct Params
    {
        uint64_t desc; float bias, dt, recipDt; uint32_t boolsWord;
    } prm = { uint64_t(dDesc), TM_BIAS, TM_DT, TM_RECIP_DT,
              (velIter ? 1u : 0u) | ((isTGS ? 1u : 0u) << 8) };
    PXG_CHECK_MSG(shim.launchKernel(0, TM_WARPS, 1, 1, &prm, sizeof(prm))
                      == PxgResult::Success,
                  "launch artiSolveInternalTendonAndMimicJointConstraints1T");
    shim.streamSynchronize();

    // ---- compare everything byte for byte ----
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
            std::printf("MM %s: %zu/%zu bytes differ (first at %zu)\n", name, mm, n, first);
            bad += mm;
        }
    };
    cmp(dAb, W.ab.data(), W.ab.size() * sizeof(W.ab[0]), "artiBlocks");
    cmp(dLk, W.lk.data(), W.lk.size() * sizeof(W.lk[0]), "linkBlocks");
    cmp(dDf, W.df.data(), W.df.size() * sizeof(W.df[0]), "dofBlocks");
    cmp(dBf, W.bf.data(), W.bf.size() * 8, "bitFields");
    cmp(dSt, W.st.data(), W.st.size() * sizeof(W.st[0]), "spatialTendons");
    cmp(dIs, W.itcS.data(), W.itcS.size() * sizeof(W.itcS[0]), "itcSpatial");
    cmp(dAt, W.at.data(), W.at.size() * sizeof(W.at[0]), "attachments");
    cmp(dFt, W.ft.data(), W.ft.size() * sizeof(W.ft[0]), "fixedTendons");
    cmp(dIf, W.itcF.data(), W.itcF.size() * sizeof(W.itcF[0]), "itcFixed");
    cmp(dTj, W.tj.data(), W.tj.size() * sizeof(W.tj[0]), "tendonJoints");
    cmp(dMj, W.mjb.data(), W.mjb.size() * sizeof(W.mjb[0]), "mimicJoints");
    cmp(dDirty, W.dirty.data(), W.dirty.size() * 4, "slabDirty");
    cmp(dImp, W.imp.data(), W.imp.size() * 4, "impulses");
    {
        std::vector<uint8_t> g(sizeof(PxgArticulationCoreDesc));
        shim.memcpyDtoH(g.data(), dDesc, g.size());
        if (std::memcmp(g.data(), &desc, g.size()) != 0)
        {
            ++bad;
            std::printf("MM descriptor touched by the kernel\n");
        }
    }

    for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
    return bad;
}

} // namespace

PXG_TEST(artiSolveTMLaunch)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiSolveTMKernel.spv", "artiSolveInternalTendonAndMimicJointConstraints1T",
          24u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &tmLoader);

    size_t bad = 0;
    bad += tmRun(shim, false, true);    // position iteration, TGS
    bad += tmRun(shim, true, false);    // velocity iteration, PGS
    PXG_CHECK_MSG(bad == 0, "%zu mismatches", bad);
}
