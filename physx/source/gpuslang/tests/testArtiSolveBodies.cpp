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

// P8.98b: the three per-iteration internal-constraint solve bodies —
// `solveInternalSpatialConstraints` (internalConstraints2.cu:2317, with the
// TGS length-refresh DFS :2015), `solveInternalFixedConstraints` (:2435) and
// `solveInternalMimicJointConstraints` (:2693).
//
// THE ORACLE: control flow over engine primitives plus the P8.98a hoisted
// mirrors (pxgArtiSolveRef.h) — this fixture is their second USER, not a
// second transcription.  The mimic impulse scalar itself is computed by the
// ENGINE's own Dy::computeMimicJointImpulses (PX_CUDA_CALLABLE,
// DyCpuGpuArticulation.h:637), so that path has a true host oracle.
//
// WHAT THE FIXTURE IS SHAPED TO SEE
//  * the P8.98a branch topology (parents 0,1,1,2,3) reused, all buffers
//    seeded per-lane by hash;
//  * spatial: two leaf constraints per tendon 0 (attachment tree att0 root on
//    link 0 -> {att1 NON-leaf on link 1 -> att3 leaf on link 3, att2 leaf on
//    link 2}), TGS and PGS rounds — TGS refreshes mAccumulatedLength through
//    the pose DFS first, including the non-leaf accumLength chain;
//    a second tendon (lanes with 5+ links) whose constraint has EVERY
//    multiplier and applied force zero, driving deltaF to EXACTLY zero so
//    the `deltaF != 0` guard's skip arm actually runs;
//  * fixed: tendon-joint tree tj0(link 1) -> {tj2(link 3), tj1(link 2 ->
//    tj3(link 4) on branch lanes)}, the two-stack DFS, the popcount scale,
//    the scratch contract (mScratchDeltaV survives, mScratchImpulse re-zeroed
//    over nbLinks), and the limit branches;
//  * mimic: the P8.97c joint shapes with the compliance three-way branch
//    (naturalFrequency/dampingRatio both >0 / nf<=0 / dr<=0 across lanes)
//    in BOTH iteration modes (position and velocity);
//  * branch coverage counted on the host and PXG_CHECKed: spatial accum
//    over/under/inside the limits, fixed length over/under/inside, mimic
//    compliant vs hard, and the exact-zero deltaF case.

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

static std::vector<uint8_t> sbLoader(const char* file) { return loadShaderBlob(file); }

const uint32_t SB_NB = 68, SB_WARPS = 3;
const uint32_t SB_MAX_LINKS = 6, SB_MAX_DOFS = 8, SB_WORDSIZE = 1;
const uint32_t SB_MAX_ST = 2, SB_MAX_AT = 5;
const uint32_t SB_MAX_FT = 2, SB_MAX_TJ = 4;
const uint32_t SB_MAX_MJ = 2;
const float SB_BIAS = 0.7f, SB_DT = 0.0166f, SB_RECIP_DT = 60.24f;

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

static uint32_t sbNumLinks(uint32_t a) { return 4u + (a % 3u); }
static uint32_t sbParent(uint32_t l)
{
    static const uint32_t p[6] = { 0, 0, 1, 1, 2, 3 };
    return p[l];
}
static uint32_t sbDofs(uint32_t a, uint32_t l) { return 1u + ((a + l) % 2u); }

struct SbArena
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
};

static void sbSeed(SbArena& B)
{
    B.ab.resize(SB_WARPS);
    B.lk.resize(size_t(SB_WARPS) * SB_MAX_LINKS);
    B.df.resize(size_t(SB_WARPS) * SB_MAX_DOFS);
    B.bf.resize(size_t(SB_WARPS) * SB_MAX_LINKS * SB_WORDSIZE * 32);
    B.st.resize(size_t(SB_WARPS) * SB_MAX_ST);
    B.itcS.resize(size_t(SB_WARPS) * SB_MAX_ST * SB_MAX_AT);
    B.at.resize(size_t(SB_WARPS) * SB_MAX_ST * SB_MAX_AT);
    B.ft.resize(size_t(SB_WARPS) * SB_MAX_FT);
    B.itcF.resize(size_t(SB_WARPS) * SB_MAX_FT * SB_MAX_TJ);
    B.tj.resize(size_t(SB_WARPS) * SB_MAX_FT * SB_MAX_TJ);
    B.mjb.resize(size_t(SB_WARPS) * SB_MAX_MJ);
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

    for (uint32_t a = 0; a < SB_NB; ++a)
    {
        const uint32_t w = a / 32, lane = a % 32;
        PxgArticulationBlockData& AB = B.ab[w];
        PxgArticulationBlockLinkData* L = &B.lk[size_t(w) * SB_MAX_LINKS];
        PxgArticulationBlockDofData* D = &B.df[size_t(w) * SB_MAX_DOFS];
        uint64_t* BF = &B.bf[size_t(w) * SB_MAX_LINKS * SB_WORDSIZE * 32];

        const uint32_t numLinks = sbNumLinks(a);
        AB.mNumLinks[lane] = numLinks;
        // free base for even a, fixed for odd (the fixBase flag bit)
        AB.mFlags[lane] = (a & 1u);
        AB.mNumSpatialTendons[lane] = numLinks >= 5 ? 2 : 1;
        AB.mNumFixedTendons[lane] = 1;
        AB.mNumMimicJoints[lane] = numLinks >= 5 ? 2 : 1;
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

        uint32_t dofCursor = 0;
        for (uint32_t l = 0; l < numLinks; ++l)
        {
            PxgArticulationBlockLinkData& lnk = L[l];

            uint64_t path = 1ull << 0;
            for (uint32_t p = l; p != 0; p = sbParent(p))
                path |= 1ull << p;
            BF[(l * SB_WORDSIZE + 0) * 32 + lane] = path;

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

            if (l == 0)
            {
                lnk.mJointOffset[lane] = 0;
                lnk.mDofs[lane] = 0;
                lnk.mRw_x[lane] = 7.5f;
                lnk.mRw_y[lane] = -3.25f;
                lnk.mRw_z[lane] = 9.0f;
                continue;
            }
            lnk.mParents[lane] = sbParent(l);
            lnk.mRw_x[lane] = 0.5f * fv(a, l, 0, 86);
            lnk.mRw_y[lane] = 0.5f * fv(a, l, 1, 86);
            lnk.mRw_z[lane] = 0.5f * fv(a, l, 2, 86);
            const uint32_t nbDofs = sbDofs(a, l);
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

        // ---- spatial tendons: att0 root(link 0) -> {att1(link 2), att2(link 3)} ----
        PxgArticulationBlockSpatialTendonData* ST = &B.st[size_t(w) * SB_MAX_ST];
        PxgArticulationInternalTendonConstraintData* IS =
            &B.itcS[size_t(w) * SB_MAX_ST * SB_MAX_AT];
        PxgArticulationBlockAttachmentData* AT = &B.at[size_t(w) * SB_MAX_ST * SB_MAX_AT];
        for (uint32_t t = 0; t < AB.mNumSpatialTendons[lane]; ++t)
        {
            ST[t].mOffset[lane] = 0.4f * fv(a, t, 3, 50);
            ST[t].mNumConstraints[lane] = 2;
            PxgArticulationBlockAttachmentData* T = &AT[t * SB_MAX_AT];
            auto att = [&](uint32_t idx, uint32_t linkIndex, uint32_t parent, uint64_t children) {
                T[idx].mLinkIndex[lane] = PxU16(linkIndex);
                T[idx].mParents[lane] = parent;
                T[idx].mChildrens[lane] = children;
                T[idx].mRelativeOffset[lane] = PxVec3(0.3f * fv(a, t * 8 + idx, 0, 51),
                                                      0.3f * fv(a, t * 8 + idx, 1, 51),
                                                      0.3f * fv(a, t * 8 + idx, 2, 51));
                T[idx].mCoefficient[lane] = 0.6f + 0.3f * fv(a, t * 8 + idx, 3, 51);
            };
            // att1 is a NON-leaf (children {att3}): without one, the TGS
            // refresh's `children != 0` arm never runs and a mutant that
            // stores a constraint at every node is an identity at fixture
            // scale (the P8.96a lesson, caught by this sweep's one survivor)
            att(0, 0, 0, (1ull << 1) | (1ull << 2));
            att(1, 1, 0, 1ull << 3);
            att(2, 2, 0, 0);
            att(3, 3, 1, 0);
            // DFS leaf order: att2 (link 2) first, then att3 (link 3) under att1
            const uint32_t leafLink[2] = { 2, 3 };
            for (uint32_t cix = 0; cix < 2; ++cix)
            {
                PxgArticulationInternalTendonConstraintData& C = IS[t * SB_MAX_AT + cix];
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
                // every fifth lane gets a HIGH low limit so the TGS-refreshed
                // lengths (O(1-4) segment sums) also land BELOW it -- without
                // this the under branch is empty in the TGS round
                C.mLowLimit[lane] = ((a % 5u) == 0u)
                    ? 3.5f + fv(a, t * 4 + cix, 9, 54)
                    : 0.1f + 0.3f * fv(a, t * 4 + cix, 9, 54);
                C.mHighLimit[lane] = C.mLowLimit[lane] + 0.35f
                                   + 0.3f * std::fabs(fv(a, t * 4 + cix, 10, 54));
                if (t == 1)
                {
                    // the exact-zero deltaF case: every multiplier and both
                    // applied forces zero
                    C.mVelMultiplier[lane] = 0.f;
                    C.mBiasCoefficient[lane] = 0.f;
                    C.mImpulseMultiplier[lane] = 0.f;
                    C.mAppliedForce[lane] = 0.f;
                    C.mLimitBiasCoefficient[lane] = 0.f;
                    C.mLimitImpulseMultiplier[lane] = 0.f;
                    C.mLimitAppliedForce[lane] = 0.f;
                }
                else
                {
                    C.mVelMultiplier[lane] = 0.5f + 0.4f * fv(a, t * 4 + cix, 11, 54);
                    C.mBiasCoefficient[lane] = 0.4f * fv(a, t * 4 + cix, 12, 54);
                    C.mImpulseMultiplier[lane] = 0.85f + 0.1f * fv(a, t * 4 + cix, 13, 54);
                    C.mAppliedForce[lane] = 0.3f * fv(a, t * 4 + cix, 14, 54);
                    C.mLimitBiasCoefficient[lane] = 0.4f * fv(a, t * 4 + cix, 15, 54);
                    C.mLimitImpulseMultiplier[lane] = 0.8f + 0.15f * fv(a, t * 4 + cix, 16, 54);
                    C.mLimitAppliedForce[lane] = 0.25f * fv(a, t * 4 + cix, 17, 54);
                }
            }
        }

        // ---- fixed tendon: tj0(link 1) -> {tj2(link 3), tj1(link 2 -> tj3(link 4))} ----
        PxgArticulationBlockFixedTendonData* FT = &B.ft[size_t(w) * SB_MAX_FT];
        PxgArticulationInternalTendonConstraintData* IF =
            &B.itcF[size_t(w) * SB_MAX_FT * SB_MAX_TJ];
        PxgArticulationBlockTendonJointData* TJ = &B.tj[size_t(w) * SB_MAX_FT * SB_MAX_TJ];
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
            // tj0 root on link 1; children tj1, tj2.  tj1 gains tj3 on branch lanes.
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

        // ---- mimic joints (the P8.97c shapes + compliance cases) ----
        PxgArticulationBlockMimicJointData* M = &B.mjb[size_t(w) * SB_MAX_MJ];
        auto mj = [&](uint32_t idx, uint32_t la, uint32_t lb) {
            M[idx].mLinkA[lane] = la;
            M[idx].mLinkB[lane] = lb;
            M[idx].mInternalData.mDofA[lane] = sbDofs(a, la) - 1u;
            M[idx].mInternalData.mDofB[lane] = sbDofs(a, lb) - 1u;
            M[idx].mInternalData.recipEffectiveInertia[lane] =
                0.5f + 0.4f * std::fabs(fv(a, 60 + idx, 0, 60));
            M[idx].mGearRatio[lane] = 0.8f + 0.5f * fv(a, 60 + idx, 1, 60);
            M[idx].mOffset[lane] = 0.3f * fv(a, 60 + idx, 2, 60);
            // the compliance three-way branch across lanes
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

// ---- run one body family: seed, mirror, dispatch, byte-compare ----

struct SbParamsHost
{
    uint64_t ab, lk, df, bf, tendon, constraint, elem;
    float biasCoefficient, dt, recipDt;
    uint32_t isTGS, isVelIter, nb, maxLinks, maxDofs, maxTendons, maxElements;
};

enum { K_SPATIAL = 0, K_FIXED = 1, K_MIMIC = 2 };

static size_t sbCompareAll(PxgCudaShim& shim, const SbArena& W,
                           PxgDevicePtr dAb, PxgDevicePtr dLk, PxgDevicePtr dDf,
                           PxgDevicePtr dBf, PxgDevicePtr dT, PxgDevicePtr dC,
                           PxgDevicePtr dE, const SbArena& seedForSizes,
                           const void* tHost, size_t tBytes,
                           const void* cHost, size_t cBytes,
                           const void* eHost, size_t eBytes, const char* what)
{
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
            std::printf("MM %s/%s: %zu/%zu bytes differ (first at %zu)\n", what, name, mm, n,
                        first);
            bad += mm;
        }
    };
    cmp(dAb, W.ab.data(), W.ab.size() * sizeof(W.ab[0]), "artiBlocks");
    cmp(dLk, W.lk.data(), W.lk.size() * sizeof(W.lk[0]), "linkBlocks");
    cmp(dDf, W.df.data(), W.df.size() * sizeof(W.df[0]), "dofBlocks");
    cmp(dBf, W.bf.data(), W.bf.size() * 8, "bitFields");
    if (dT) cmp(dT, tHost, tBytes, "tendons");
    if (dC) cmp(dC, cHost, cBytes, "constraints");
    if (dE) cmp(dE, eHost, eBytes, "elements");
    PX_UNUSED(seedForSizes);
    return bad;
}

static size_t sbRun(PxgCudaShim& shim, int family, bool flagA)
{
    SbArena A;
    sbSeed(A);

    // mirror
    SbArena W = A;
    PxU32 nOver = 0, nUnder = 0, nInside = 0, nZeroDf = 0, nCompliant = 0, nHard = 0;
    for (uint32_t a = 0; a < SB_NB; ++a)
    {
        const uint32_t w = a / 32, lane = a % 32;
        PxgArticulationBlockData& ab = W.ab[w];
        PxgArticulationBlockLinkData* lk = &W.lk[size_t(w) * SB_MAX_LINKS];
        PxgArticulationBlockDofData* df = &W.df[size_t(w) * SB_MAX_DOFS];
        const uint64_t* bf = &W.bf[size_t(w) * SB_MAX_LINKS * SB_WORDSIZE * 32];
        if (family == K_SPATIAL)
            refSolveSpatial(ab, lk, df, bf, SB_WORDSIZE, &W.st[size_t(w) * SB_MAX_ST],
                            &W.itcS[size_t(w) * SB_MAX_ST * SB_MAX_AT],
                            &W.at[size_t(w) * SB_MAX_ST * SB_MAX_AT], SB_MAX_AT, flagA,
                            lane, nOver, nUnder, nInside, nZeroDf);
        else if (family == K_FIXED)
            refSolveFixed(ab, lk, df, bf, SB_WORDSIZE, &W.ft[size_t(w) * SB_MAX_FT],
                          &W.itcF[size_t(w) * SB_MAX_FT * SB_MAX_TJ],
                          &W.tj[size_t(w) * SB_MAX_FT * SB_MAX_TJ], SB_MAX_TJ, lane,
                          nOver, nUnder, nInside);
        else
            refSolveMimic(ab, lk, df, bf, SB_WORDSIZE, &W.mjb[size_t(w) * SB_MAX_MJ],
                          SB_BIAS, SB_DT, SB_RECIP_DT, flagA, lane, nCompliant, nHard);
    }
    if (family == K_SPATIAL)
    {
        std::printf("[ info ] sb spatial(%s): limits over/under/inside %u/%u/%u, zero-deltaF %u\n",
                    flagA ? "TGS" : "PGS", nOver, nUnder, nInside, nZeroDf);
        PXG_CHECK_MSG(nOver > 0 && nUnder > 0 && nInside > 0 && nZeroDf > 0,
                      "spatial branch coverage hole");
    }
    else if (family == K_FIXED)
    {
        std::printf("[ info ] sb fixed: limits over/under/inside %u/%u/%u\n", nOver, nUnder,
                    nInside);
        PXG_CHECK_MSG(nOver > 0 && nUnder > 0 && nInside > 0, "fixed branch coverage hole");
    }
    else
    {
        std::printf("[ info ] sb mimic(%s): compliant %u, hard %u\n",
                    flagA ? "velIter" : "posIter", nCompliant, nHard);
        PXG_CHECK_MSG(nHard > 0, "mimic branch coverage hole");
        if (!flagA)
            PXG_CHECK_MSG(nCompliant > 0, "mimic compliant case missing");
    }

    // device
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
    PxgDevicePtr dT = 0, dC = 0, dE = 0;
    const void* tHost = NULL; const void* cHost = NULL; const void* eHost = NULL;
    size_t tBytes = 0, cBytes = 0, eBytes = 0;
    uint32_t maxTendons = 0, maxElements = 0;
    if (family == K_SPATIAL)
    {
        dT = up(A.st.data(), tBytes = A.st.size() * sizeof(A.st[0]));
        dC = up(A.itcS.data(), cBytes = A.itcS.size() * sizeof(A.itcS[0]));
        dE = up(A.at.data(), eBytes = A.at.size() * sizeof(A.at[0]));
        tHost = W.st.data(); cHost = W.itcS.data(); eHost = W.at.data();
        maxTendons = SB_MAX_ST; maxElements = SB_MAX_AT;
    }
    else if (family == K_FIXED)
    {
        dT = up(A.ft.data(), tBytes = A.ft.size() * sizeof(A.ft[0]));
        dC = up(A.itcF.data(), cBytes = A.itcF.size() * sizeof(A.itcF[0]));
        dE = up(A.tj.data(), eBytes = A.tj.size() * sizeof(A.tj[0]));
        tHost = W.ft.data(); cHost = W.itcF.data(); eHost = W.tj.data();
        maxTendons = SB_MAX_FT; maxElements = SB_MAX_TJ;
    }
    else
    {
        dE = up(A.mjb.data(), eBytes = A.mjb.size() * sizeof(A.mjb[0]));
        eHost = W.mjb.data();
        maxTendons = SB_MAX_MJ; maxElements = 1;
    }
    if (!dAb || !dLk || !dDf || !dBf || (family != K_MIMIC && (!dT || !dC)) || !dE)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
        return 0;
    }

    SbParamsHost prm = { uint64_t(dAb), uint64_t(dLk), uint64_t(dDf), uint64_t(dBf),
                         uint64_t(dT), uint64_t(dC), uint64_t(dE),
                         SB_BIAS, SB_DT, SB_RECIP_DT,
                         (family == K_SPATIAL && flagA) ? 1u : 0u,
                         (family == K_MIMIC && flagA) ? 1u : 0u,
                         SB_NB, SB_MAX_LINKS, SB_MAX_DOFS, maxTendons, maxElements };
    PXG_CHECK_MSG(shim.launchKernel(family, SB_WARPS, 1, 1, &prm, sizeof(prm))
                      == PxgResult::Success,
                  "launch solve body");
    shim.streamSynchronize();

    const char* what = family == K_SPATIAL ? (flagA ? "spatialTGS" : "spatialPGS")
                     : family == K_FIXED ? "fixed"
                                          : (flagA ? "mimicVel" : "mimicPos");
    const size_t bad = sbCompareAll(shim, W, dAb, dLk, dDf, dBf, dT, dC, dE, A,
                                    tHost, tBytes, cHost, cBytes, eHost, eBytes, what);

    for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
    return bad;
}

} // namespace

PXG_TEST(artiSolveBodies)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiSolveTendons.spv", "svSolveSpatialTest", 96u, 0 },
        { "npArtiSolveTendons.spv", "svSolveFixedTest", 96u, 0 },
        { "npArtiSolveTendons.spv", "svSolveMimicTest", 96u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 3, &sbLoader);

    size_t bad = 0;
    bad += sbRun(shim, K_SPATIAL, true);    // TGS: length refresh + solve
    bad += sbRun(shim, K_SPATIAL, false);   // PGS: seeded lengths
    bad += sbRun(shim, K_FIXED, false);
    bad += sbRun(shim, K_MIMIC, false);     // position iteration
    bad += sbRun(shim, K_MIMIC, true);      // velocity iteration
    PXG_CHECK_MSG(bad == 0, "%zu mismatches", bad);
}
