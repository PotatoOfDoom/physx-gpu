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

// P8.97d: `setupInternalConstraintLaunch1T`, the assembled setup kernel, driven
// over the REAL engine ABI — a device-resident PxgArticulationCoreDesc, the
// (32, 1, 1) x ceil(nbArticulations / 32) launch shape (numWarpsPerBlock = 1,
// PxgArticulationCore.cpp:492-517), and the {ptr, float, float, float, bool}
// argument pack with the bool as ONE byte at offset 20.
//
// ===========================================================================
// THE ORACLE: a GPU DIFFERENTIAL against the three swept per-family entries
// ===========================================================================
//
// The wrapper is pure composition — its own surface is the per-warp index
// chain and the argument pack; the four bodies it calls are each pinned
// bit-exactly by their own fixtures (P8.97a host-oracle + coverage histogram,
// P8.97b engine-primitive mirror, P8.97c the full host-callable engine pair).
// So the expectation here is not a fourth transcription: the SAME seeded
// arena is uploaded twice, the real kernel runs on copy A, and the four
// already-verified test entries (siSetupInternalTest, stSetupSpatialTest,
// stSetupFixedTest, mjSetupMimicTest — each doing exactly the reference index
// chain over push-constant bases) run on copy B in the engine's order.  Every
// buffer is then compared byte for byte, A against B — poison included, the
// allocated-but-dead fourth warp slice included, the descriptor image
// included.  Both sides execute the identical imported module code (-O0 +
// nocontract), so the comparison is exact: no ULP masks, no tolerance.
//
// A wrapper index-chain error (wrong stride, wrong descriptor pointer, wrong
// warps-per-block, swapped scalars, mis-read bool byte) lands the work in a
// different slice or computes with different inputs on side A only, and the
// differential fires.  Vacuity is closed explicitly: after the reference run,
// side B must DIFFER from the seed in the spatial-ITC, dof and mimic buffers
// (a refused or no-op dispatch pair cannot pass).
//
// ===========================================================================
// WHAT THE FIXTURE IS SHAPED TO SEE
// ===========================================================================
//
//  * 68 articulations -> grid 3 with warp slice 2 only 4/32 occupied, plus a
//    FOURTH allocated slice that must stay poisoned on both sides;
//  * every per-warp pitch distinct (the standing rule): maxLinks 6, maxDofs 8,
//    maxSpatialTendons 2, maxAttachments 5, maxFixedTendons 3,
//    maxTendonJoints 4, maxMimicJoints 7 — so any stride swap in the wrapper's
//    slice arithmetic moves warp >= 1 onto different bytes;
//  * per-articulation content varies by hash in every seeded field, so
//    slice-index errors land on different numbers, not on a coincidence;
//  * topology is the P8.97b/c tree (chain 0-1-2-3, link 4 branching from 1,
//    link 5 chaining from 4; numLinks 4 + a%3) with per-link joint cases for
//    the drive setup, both tendon families, and both mimic-joint path shapes;
//  * both solver modes dispatched (TGS and PGS) over identically seeded
//    arenas — stepDt, dt, invDt all distinct so a swapped scalar is visible.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "foundation/PxQuat.h"
#include "foundation/PxVec3.h"
#include "foundation/PxTransform.h"
#include "CmSpatialVector.h"
#include "DyFeatherstoneArticulationUtils.h"
#include "DyFeatherstoneArticulation.h"
#include "PxSpatialMatrix.h"
#include "PxgArticulation.h"
#include "PxgArticulationCoreDesc.h"
#include "PxArticulationReducedCoordinate.h"

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

static std::vector<uint8_t> sicLoader(const char* file) { return loadShaderBlob(file); }

const uint32_t SIC_NB = 68, SIC_WARPS = 3, SIC_WARPS_ALLOC = 4;
const uint32_t SIC_MAX_LINKS = 6, SIC_MAX_DOFS = 8;
const uint32_t SIC_MAX_ST = 2, SIC_MAX_AT = 5;      // spatial tendons / attachments
const uint32_t SIC_MAX_FT = 3, SIC_MAX_TJ = 4;      // fixed tendons / tendon joints
const uint32_t SIC_MAX_MJ = 7;                      // mimic joints (live: 1-2)
const uint32_t SIC_WORDSIZE = (SIC_MAX_LINKS + 63u) / 64u;   // = 1
const float SIC_STEP_DT = 0.004f, SIC_DT = 0.0333f, SIC_INV_DT = 250.0f;

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
static uint32_t sicHash(uint32_t a, uint32_t l)
{
    uint32_t h = a * 2654435761u + l * 40503u + 23u;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    return h;
}

static uint32_t sicNumLinks(uint32_t a) { return 4u + (a % 3u); }
static uint32_t sicParent(uint32_t l) { return (l == 4) ? 1 : (l == 5 ? 4 : l - 1); }
static uint32_t sicDofs(uint32_t a, uint32_t l) { return 1u + ((a + l) % 2u); }

struct SicArena
{
    std::vector<PxgArticulationBlockData> ab;
    std::vector<PxgArticulationBlockLinkData> lk;
    std::vector<PxgArticulationBlockDofData> df;
    std::vector<PxgArticulationBlockSpatialTendonData> st;
    std::vector<PxgArticulationBlockFixedTendonData> ft;
    std::vector<PxgArticulationBlockAttachmentData> at;
    std::vector<PxgArticulationBlockTendonJointData> tj;
    std::vector<PxgArticulationInternalTendonConstraintData> itcS;
    std::vector<PxgArticulationInternalTendonConstraintData> itcF;
    std::vector<PxgArticulationBlockMimicJointData> mjb;
    std::vector<uint64_t> bf;
};

// the union of the P8.97a/b/c seeds on one consistent topology
static void sicSeed(SicArena& B)
{
    B.ab.resize(SIC_WARPS_ALLOC);
    B.lk.resize(size_t(SIC_WARPS_ALLOC) * SIC_MAX_LINKS);
    B.df.resize(size_t(SIC_WARPS_ALLOC) * SIC_MAX_DOFS);
    B.st.resize(size_t(SIC_WARPS_ALLOC) * SIC_MAX_ST);
    B.ft.resize(size_t(SIC_WARPS_ALLOC) * SIC_MAX_FT);
    B.at.resize(size_t(SIC_WARPS_ALLOC) * SIC_MAX_ST * SIC_MAX_AT);
    B.tj.resize(size_t(SIC_WARPS_ALLOC) * SIC_MAX_FT * SIC_MAX_TJ);
    B.itcS.resize(size_t(SIC_WARPS_ALLOC) * SIC_MAX_ST * SIC_MAX_AT);
    B.itcF.resize(size_t(SIC_WARPS_ALLOC) * SIC_MAX_FT * SIC_MAX_TJ);
    B.mjb.resize(size_t(SIC_WARPS_ALLOC) * SIC_MAX_MJ);
    B.bf.resize(size_t(SIC_WARPS_ALLOC) * SIC_MAX_LINKS * SIC_WORDSIZE * 32);
    std::memset(B.ab.data(), 0xAB, B.ab.size() * sizeof(B.ab[0]));
    std::memset(B.lk.data(), 0xAB, B.lk.size() * sizeof(B.lk[0]));
    std::memset(B.df.data(), 0xAB, B.df.size() * sizeof(B.df[0]));
    std::memset(B.st.data(), 0xAB, B.st.size() * sizeof(B.st[0]));
    std::memset(B.ft.data(), 0xAB, B.ft.size() * sizeof(B.ft[0]));
    std::memset(B.at.data(), 0xAB, B.at.size() * sizeof(B.at[0]));
    std::memset(B.tj.data(), 0xAB, B.tj.size() * sizeof(B.tj[0]));
    std::memset(B.itcS.data(), 0xAB, B.itcS.size() * sizeof(B.itcS[0]));
    std::memset(B.itcF.data(), 0xAB, B.itcF.size() * sizeof(B.itcF[0]));
    std::memset(B.mjb.data(), 0xAB, B.mjb.size() * sizeof(B.mjb[0]));
    std::memset(B.bf.data(), 0xAB, B.bf.size() * 8);

    for (uint32_t a = 0; a < SIC_NB; ++a)
    {
        const uint32_t w = a / 32, lane = a % 32;
        PxgArticulationBlockData& AB = B.ab[w];
        PxgArticulationBlockLinkData* L = &B.lk[size_t(w) * SIC_MAX_LINKS];
        PxgArticulationBlockDofData* D = &B.df[size_t(w) * SIC_MAX_DOFS];
        uint64_t* BF = &B.bf[size_t(w) * SIC_MAX_LINKS * SIC_WORDSIZE * 32];

        const uint32_t numLinks = sicNumLinks(a);
        AB.mNumLinks[lane] = numLinks;
        AB.mFlags[lane] = (a & 1u) ? PxArticulationFlag::eDRIVE_LIMITS_ARE_FORCES : 0u;
        AB.mNumSpatialTendons[lane] = numLinks >= 5 ? 2 : 1;
        AB.mNumFixedTendons[lane] = 1;
        AB.mNumMimicJoints[lane] = numLinks >= 5 ? 2 : 1;

        uint32_t dofCursor = 0;
        for (uint32_t l = 0; l < numLinks; ++l)
        {
            PxgArticulationBlockLinkData& lnk = L[l];
            const PxQuat accQ = qv(a, l, 31);
            lnk.mAccumulatedPose.q[lane] = make_float4(accQ.x, accQ.y, accQ.z, accQ.w);
            lnk.mAccumulatedPose.p[lane] = make_float4(2.0f * fv(a, l, 0, 32),
                                                       2.0f * fv(a, l, 1, 32),
                                                       2.0f * fv(a, l, 2, 32), 0.f);
            lnk.mCfm[lane] = 0.01f + 0.004f * float((a + l) % 6);
            const PxQuat paQ = qv(a, l, 33), chQ = qv(a, l, 34);
            lnk.mParentPose.q[lane] = make_float4(paQ.x, paQ.y, paQ.z, paQ.w);
            lnk.mParentPose.p[lane] = make_float4(fv(a, l, 3, 35), fv(a, l, 4, 35),
                                                  fv(a, l, 5, 35), 0.f);
            lnk.mChildPose.q[lane] = make_float4(chQ.x, chQ.y, chQ.z, chQ.w);
            lnk.mChildPose.p[lane] = make_float4(fv(a, l, 6, 36), fv(a, l, 7, 36),
                                                 fv(a, l, 8, 36), 0.f);
            {
                PxSpatialMatrix m;
                for (uint32_t cI = 0; cI < 6; ++cI)
                    for (uint32_t r = 0; r < 6; ++r)
                        m.column[cI][r] = 0.2f * fv(a, l, cI * 6 + r, 37)
                                        + (cI == r ? 1.0f : 0.0f);
                storeSpatialMatrix(lnk.mSpatialResponseMatrix, lane, m);
            }
            // the path-to-root bitfield row (word 0; root bit included)
            uint64_t path = 1ull << 0;
            for (uint32_t p = l; p != 0; p = sicParent(p))
                path |= 1ull << p;
            BF[(l * SIC_WORDSIZE + 0) * 32 + lane] = path;

            if (l == 0)
            {
                lnk.mJointOffset[lane] = 0;
                lnk.mDofs[lane] = 0;
                // the walks skip the root; LARGE sentinels keep a mutant that
                // reads it visible (0xAB floats are ~1e-12 — the P8.97c lesson)
                lnk.mRw_x[lane] = 7.5f;
                lnk.mRw_y[lane] = -3.25f;
                lnk.mRw_z[lane] = 9.0f;
                continue;
            }
            lnk.mParents[lane] = sicParent(l);
            lnk.mRw_x[lane] = 0.5f * fv(a, l, 0, 38);
            lnk.mRw_y[lane] = 0.5f * fv(a, l, 1, 38);
            lnk.mRw_z[lane] = 0.5f * fv(a, l, 2, 38);
            storeSpatialVector(lnk.mBiasForce,
                UnAlignedSpatialVector(
                    PxVec3(fv(a, l, 0, 9), fv(a, l, 1, 9), fv(a, l, 2, 9)),
                    PxVec3(fv(a, l, 3, 9), fv(a, l, 4, 9), fv(a, l, 5, 9))), lane);

            const uint32_t nbDofs = sicDofs(a, l);
            lnk.mJointOffset[lane] = dofCursor;
            lnk.mDofs[lane] = PxU8(nbDofs);

            // per-link joint case, consistent with nbDofs
            const uint32_t h = sicHash(a, l);
            uint8_t jointType, dofIds[2], motion[2], driveType[2];
            bool zeroDrive = false, dampingOnly = false;
            if (nbDofs == 1)
            {
                switch (h % 5u)
                {
                default:
                case 0: jointType = PxArticulationJointType::eREVOLUTE;
                        dofIds[0] = 0; motion[0] = 2; driveType[0] = 0; break;
                case 1: jointType = PxArticulationJointType::ePRISMATIC;
                        dofIds[0] = 3; motion[0] = 1; driveType[0] = 0; break;
                case 2: jointType = PxArticulationJointType::eREVOLUTE;
                        dofIds[0] = 1; motion[0] = 2; driveType[0] = 0;
                        dampingOnly = true; break;
                case 3: jointType = PxArticulationJointType::eREVOLUTE;
                        dofIds[0] = 0; motion[0] = 2; driveType[0] = 0;
                        zeroDrive = true; break;
                case 4: jointType = PxArticulationJointType::eREVOLUTE_UNWRAPPED;
                        dofIds[0] = 2; motion[0] = 2; driveType[0] = 1; break;
                }
            }
            else
            {
                jointType = PxArticulationJointType::eSPHERICAL;
                if (h & 1u) { dofIds[0] = 0; dofIds[1] = 2; }
                else        { dofIds[0] = 1; dofIds[1] = 2; }
                motion[0] = 2; motion[1] = (h & 2u) ? 1 : 2;
                driveType[0] = h % 3u; driveType[1] = (h / 3u) % 3u;
            }
            lnk.mJointType[lane] = jointType;

            for (uint32_t ax = 0; ax < 6; ++ax)
                lnk.mInvDofIds[ax][lane] = 0;
            for (uint32_t d = 0; d < nbDofs; ++d)
            {
                lnk.mInvDofIds[(l + d) % 6u][lane] = PxU8(d);
                PxgArticulationBlockDofData& dd = D[dofCursor + d];
                dd.mMotion[lane] = motion[d];
                dd.mDofIds[lane] = dofIds[d];
                dd.mJointPositions[lane] = 0.8f * fv(a, l, d, 11);
                storeSpatialVector(dd.mLocalMotionMatrix,
                    UnAlignedSpatialVector(
                        PxVec3(fv(a, l, d, 12), fv(a, l, d + 3, 12), fv(a, l, d + 6, 12)),
                        PxVec3(fv(a, l, d, 13), fv(a, l, d + 3, 13), fv(a, l, d + 6, 13))), lane);
                storeSpatialVector(dd.mWorldMotionMatrix,
                    UnAlignedSpatialVector(
                        PxVec3(fv(a, l, d, 40), fv(a, l, d + 3, 40), fv(a, l, d + 6, 40)),
                        PxVec3(fv(a, l, d, 41), fv(a, l, d + 3, 41), fv(a, l, d + 6, 41))), lane);
                storeSpatialVector(dd.mIsInvDW,
                    UnAlignedSpatialVector(
                        PxVec3(fv(a, l, d, 42), fv(a, l, d + 3, 42), fv(a, l, d + 6, 42)),
                        PxVec3(fv(a, l, d, 43), fv(a, l, d + 3, 43), fv(a, l, d + 6, 43))), lane);
                storeSpatialVector(dd.mIsW,
                    UnAlignedSpatialVector(
                        PxVec3(fv(a, l, d, 44), fv(a, l, d + 3, 44), fv(a, l, d + 6, 44)),
                        PxVec3(fv(a, l, d, 45), fv(a, l, d + 3, 45), fv(a, l, d + 6, 45))), lane);
                dd.mInvStIsT_x[lane] = fv(a, l, d, 46);
                dd.mInvStIsT_y[lane] = fv(a, l, d, 47);
                dd.mInvStIsT_z[lane] = fv(a, l, d, 48);

                PxgArticulationInternalConstraintData& ic = dd.mConstraintData;
                ic.mFrictionCoefficient[lane] = 0.5f + 0.5f * fv(a, l, d, 23);
                ic.mDriveType[lane] = driveType[d];
                ic.mDriveTargetPos[lane] =
                    (jointType == PxArticulationJointType::eSPHERICAL && (a % 8u) == 7u)
                        ? 0.f : 0.9f * fv(a, l, d, 24);
                ic.mDriveTargetVel[lane] = fv(a, l, d, 25);
                ic.mDriveStiffness[lane] =
                    (zeroDrive || dampingOnly) ? 0.f : 40.f + 10.f * fv(a, l, d, 26);
                ic.mDamping[lane] = 3.f + 1.f * fv(a, l, d, 27);
                ic.mMaxForce[lane] = zeroDrive ? 0.f : 100.f + 20.f * fv(a, l, d, 28);
                ic.mMaxEffort[lane] = 0.f;
                ic.mLimits_LowLimitX_highLimitY[lane] =
                    make_float2(-1.2f + 0.1f * fv(a, l, d, 29), 1.3f + 0.1f * fv(a, l, d, 30));
            }
            dofCursor += nbDofs;
        }
        AB.mTotalDofs[lane] = dofCursor;

        // ---- spatial tendons (the P8.97b shapes) ----
        PxgArticulationBlockSpatialTendonData* ST = &B.st[size_t(w) * SIC_MAX_ST];
        PxgArticulationBlockAttachmentData* AT = &B.at[size_t(w) * SIC_MAX_ST * SIC_MAX_AT];
        for (uint32_t t = 0; t < AB.mNumSpatialTendons[lane]; ++t)
        {
            ST[t].mStiffness[lane] = 30.f + 8.f * fv(a, t, 0, 50);
            ST[t].mDamping[lane] = 2.f + 1.f * fv(a, t, 1, 50);
            ST[t].mLimitStiffness[lane] = 20.f + 6.f * fv(a, t, 2, 50);
            ST[t].mOffset[lane] = 0.4f * fv(a, t, 3, 50);
            PxgArticulationBlockAttachmentData* T = &AT[t * SIC_MAX_AT];
            auto att = [&](uint32_t idx, uint32_t linkIndex, uint32_t parent, uint64_t children) {
                T[idx].mLinkIndex[lane] = PxU16(linkIndex);
                T[idx].mParents[lane] = parent;
                T[idx].mChildrens[lane] = children;
                T[idx].mRelativeOffset[lane] = PxVec3(0.3f * fv(a, t * 8 + idx, 0, 51),
                                                      0.3f * fv(a, t * 8 + idx, 1, 51),
                                                      0.3f * fv(a, t * 8 + idx, 2, 51));
                T[idx].mCoefficient[lane] = 0.6f + 0.3f * fv(a, t * 8 + idx, 3, 51);
                T[idx].mRestDistance[lane] = fv(a, t * 8 + idx, 4, 51);
                T[idx].mLowLimit[lane] = -1.f + 0.2f * fv(a, t * 8 + idx, 5, 51);
                T[idx].mHighLimit[lane] = 1.f + 0.2f * fv(a, t * 8 + idx, 6, 51);
            };
            if (t == 0)
            {
                const bool degen = (a & 1u) != 0u;
                att(0, 0, 0, degen ? ((1ull << 1) | (1ull << 2) | (1ull << 4))
                                   : ((1ull << 1) | (1ull << 2)));
                att(1, 1, 0, 1ull << 3);
                att(2, numLinks - 1, 0, 0);   // slow leaf, deep link
                att(3, 1, 1, 0);              // fast leaf, processed last
                if (degen)
                {
                    att(4, 0, 0, 0);
                    T[4].mRelativeOffset[lane] = T[0].mRelativeOffset[lane];
                    if ((a % 4u) == 3u)
                        T[4].mRelativeOffset[lane].x += 4e-4f;
                }
            }
            else
            {
                att(0, 4, 0, 1ull << 1);
                att(1, 3, 0, 0);
            }
        }

        // ---- fixed tendons ----
        PxgArticulationBlockFixedTendonData* FT = &B.ft[size_t(w) * SIC_MAX_FT];
        PxgArticulationBlockTendonJointData* TJ = &B.tj[size_t(w) * SIC_MAX_FT * SIC_MAX_TJ];
        {
            FT[0].mStiffness[lane] = 25.f + 7.f * fv(a, 9, 0, 52);
            FT[0].mDamping[lane] = 1.5f + 0.8f * fv(a, 9, 1, 52);
            FT[0].mLimitStiffness[lane] = 15.f + 5.f * fv(a, 9, 2, 52);
            auto tj = [&](uint32_t idx, uint32_t linkIndex, uint32_t axis, uint32_t parent,
                          uint64_t children) {
                TJ[idx].mLinkIndex[lane] = PxU16(linkIndex);
                TJ[idx].mAxis[lane] = axis;
                TJ[idx].mParents[lane] = parent;
                TJ[idx].mChildrens[lane] = children;
            };
            const bool branch = numLinks >= 5 && (a & 1u) != 0u;
            tj(0, 1, 0, 0, branch ? ((1ull << 1) | (1ull << 3)) : (1ull << 1));
            tj(1, 2, 2, 0, 1ull << 2);    // link 2 dof 0 axis (2+0)%6 = 2, angular
            tj(2, 3, 3, 1, 0);            // link 3 dof 0 axis (3+0)%6 = 3, linear
            if (branch)
                tj(3, 4, 4 % 6, 0, 0);
        }

        // ---- mimic joints (the P8.97c shapes) ----
        PxgArticulationBlockMimicJointData* M = &B.mjb[size_t(w) * SIC_MAX_MJ];
        M[0].mLinkA[lane] = 2;
        M[0].mLinkB[lane] = 3;
        M[0].mAxisA[lane] = (2u + (sicDofs(a, 2) - 1u)) % 6u;
        M[0].mAxisB[lane] = (3u + (sicDofs(a, 3) - 1u)) % 6u;
        M[0].mGearRatio[lane] = 0.8f + 0.5f * fv(a, 20, 0, 72);
        if (numLinks >= 5)
        {
            M[1].mLinkA[lane] = 3;
            M[1].mLinkB[lane] = 4;
            M[1].mAxisA[lane] = (3u + (sicDofs(a, 3) - 1u)) % 6u;
            M[1].mAxisB[lane] = (4u + (sicDofs(a, 4) - 1u)) % 6u;
            M[1].mGearRatio[lane] = -1.2f + 0.4f * fv(a, 21, 0, 73);
        }
    }
}

// upload one arena; returns the device pointers through refs
struct SicDev
{
    PxgDevicePtr ab, lk, df, st, ft, at, tj, itcS, itcF, mjb, bf;
    bool ok;
};

static SicDev sicUpload(PxgCudaShim& shim, const SicArena& A, std::vector<PxgDevicePtr>& bufs)
{
    SicDev d = {};
    auto up = [&](const void* p, size_t n) {
        PxgDevicePtr dp = 0;
        shim.memAlloc(&dp, n);
        if (dp && p) shim.memcpyHtoD(dp, p, n);
        bufs.push_back(dp);
        return dp;
    };
    d.ab = up(A.ab.data(), A.ab.size() * sizeof(A.ab[0]));
    d.lk = up(A.lk.data(), A.lk.size() * sizeof(A.lk[0]));
    d.df = up(A.df.data(), A.df.size() * sizeof(A.df[0]));
    d.st = up(A.st.data(), A.st.size() * sizeof(A.st[0]));
    d.ft = up(A.ft.data(), A.ft.size() * sizeof(A.ft[0]));
    d.at = up(A.at.data(), A.at.size() * sizeof(A.at[0]));
    d.tj = up(A.tj.data(), A.tj.size() * sizeof(A.tj[0]));
    d.itcS = up(A.itcS.data(), A.itcS.size() * sizeof(A.itcS[0]));
    d.itcF = up(A.itcF.data(), A.itcF.size() * sizeof(A.itcF[0]));
    d.mjb = up(A.mjb.data(), A.mjb.size() * sizeof(A.mjb[0]));
    d.bf = up(A.bf.data(), A.bf.size() * 8);
    d.ok = d.ab && d.lk && d.df && d.st && d.ft && d.at && d.tj && d.itcS && d.itcF
        && d.mjb && d.bf;
    return d;
}

// download a device buffer and byte-compare it against another download
static size_t sicCmp(PxgCudaShim& shim, PxgDevicePtr a, PxgDevicePtr b, size_t n,
                     const char* what)
{
    std::vector<uint8_t> ga(n), gb(n);
    shim.memcpyDtoH(ga.data(), a, n);
    shim.memcpyDtoH(gb.data(), b, n);
    if (std::memcmp(ga.data(), gb.data(), n) == 0)
        return 0;
    size_t bad = 0, firstOff = size_t(-1);
    for (size_t i = 0; i < n; ++i)
        if (ga[i] != gb[i]) { ++bad; if (firstOff == size_t(-1)) firstOff = i; }
    std::printf("MM %s: %zu/%zu bytes differ (first at %zu)\n", what, bad, n, firstOff);
    return bad;
}

// download and compare against a host image; returns differing byte count
static size_t sicDelta(PxgCudaShim& shim, PxgDevicePtr dp, const void* host, size_t n)
{
    std::vector<uint8_t> g(n);
    shim.memcpyDtoH(g.data(), dp, n);
    size_t diff = 0;
    const uint8_t* h = reinterpret_cast<const uint8_t*>(host);
    for (size_t i = 0; i < n; ++i)
        if (g[i] != h[i]) ++diff;
    return diff;
}

// kernel indices in the manifest below
enum { K_REAL = 0, K_SI = 1, K_ST_SPATIAL = 2, K_ST_FIXED = 3, K_MJ = 4 };

static size_t sicRun(PxgCudaShim& shim, bool isTGS)
{
    SicArena A;
    sicSeed(A);

    std::vector<PxgDevicePtr> bufs;
    SicDev dA = sicUpload(shim, A, bufs);   // the real kernel's arena
    SicDev dB = sicUpload(shim, A, bufs);   // the reference arena
    PxgDevicePtr dDesc = 0;
    shim.memAlloc(&dDesc, sizeof(PxgArticulationCoreDesc));
    bufs.push_back(dDesc);
    if (!dA.ok || !dB.ok || !dDesc)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
        return 0;
    }

    // the REAL descriptor, device-resident, poison outside the fields the
    // wrapper reads
    PxgArticulationCoreDesc desc;
    std::memset(&desc, 0xAB, sizeof(desc));
    desc.nbArticulations = SIC_NB;
    desc.mMaxLinksPerArticulation = SIC_MAX_LINKS;
    desc.mMaxDofsPerArticulation = SIC_MAX_DOFS;
    desc.mMaxSpatialTendonsPerArticulation = SIC_MAX_ST;
    desc.mMaxAttachmentPerArticulation = SIC_MAX_AT;
    desc.mMaxFixedTendonsPerArticulation = SIC_MAX_FT;
    desc.mMaxTendonJointPerArticulation = SIC_MAX_TJ;
    desc.mMaxMimicJointsPerArticulation = SIC_MAX_MJ;
    desc.mArticulationBlocks = reinterpret_cast<PxgArticulationBlockData*>(dA.ab);
    desc.mArticulationLinkBlocks = reinterpret_cast<PxgArticulationBlockLinkData*>(dA.lk);
    desc.mArticulationDofBlocks = reinterpret_cast<PxgArticulationBlockDofData*>(dA.df);
    desc.mArticulationSpatialTendonBlocks =
        reinterpret_cast<PxgArticulationBlockSpatialTendonData*>(dA.st);
    desc.mArticulationSpatialTendonConstraintBlocks =
        reinterpret_cast<PxgArticulationInternalTendonConstraintData*>(dA.itcS);
    desc.mArticulationAttachmentBlocks =
        reinterpret_cast<PxgArticulationBlockAttachmentData*>(dA.at);
    desc.mArticulationFixedTendonBlocks =
        reinterpret_cast<PxgArticulationBlockFixedTendonData*>(dA.ft);
    desc.mArticulationFixedTendonConstraintBlocks =
        reinterpret_cast<PxgArticulationInternalTendonConstraintData*>(dA.itcF);
    desc.mArticulationTendonJointBlocks =
        reinterpret_cast<PxgArticulationBlockTendonJointData*>(dA.tj);
    desc.mArticulationMimicJointBlocks =
        reinterpret_cast<PxgArticulationBlockMimicJointData*>(dA.mjb);
    desc.mPathToRootBitFieldBlocks =
        reinterpret_cast<PxgArticulationBitFieldData*>(dA.bf);
    shim.memcpyHtoD(dDesc, &desc, sizeof(desc));

    // ---- side A: the real kernel, engine launch shape ----
    struct RealParams
    {
        uint64_t desc; float stepDt, dt, invDt; uint32_t isTGSWord;
    } rp = { uint64_t(dDesc), SIC_STEP_DT, SIC_DT, SIC_INV_DT, isTGS ? 1u : 0u };
    PXG_CHECK_MSG(shim.launchKernel(K_REAL, SIC_WARPS, 1, 1, &rp, sizeof(rp))
                      == PxgResult::Success,
                  "launch setupInternalConstraintLaunch1T");

    // ---- side B: the four verified per-family entries, engine order ----
    struct SiParams
    {
        uint64_t ab, lk, df;
        float stepDt, dt, invDt;
        uint32_t isTGS, nb, maxLinks, maxDofs, pad0;
    } sp = { uint64_t(dB.ab), uint64_t(dB.lk), uint64_t(dB.df),
             SIC_STEP_DT, SIC_DT, SIC_INV_DT, isTGS ? 1u : 0u,
             SIC_NB, SIC_MAX_LINKS, SIC_MAX_DOFS, 0 };
    PXG_CHECK_MSG(shim.launchKernel(K_SI, SIC_WARPS, 1, 1, &sp, sizeof(sp))
                      == PxgResult::Success,
                  "launch siSetupInternalTest");

    struct StParams
    {
        uint64_t ab, lk, df, tendon, constraint, elem;
        float stepDt, dt, invDt;
        uint32_t isTGS, nb, maxLinks, maxDofs, maxTendons, maxElements, pad0;
    } tp = { uint64_t(dB.ab), uint64_t(dB.lk), uint64_t(dB.df),
             uint64_t(dB.st), uint64_t(dB.itcS), uint64_t(dB.at),
             SIC_STEP_DT, SIC_DT, SIC_INV_DT, isTGS ? 1u : 0u,
             SIC_NB, SIC_MAX_LINKS, SIC_MAX_DOFS, SIC_MAX_ST, SIC_MAX_AT, 0 };
    PXG_CHECK_MSG(shim.launchKernel(K_ST_SPATIAL, SIC_WARPS, 1, 1, &tp, sizeof(tp))
                      == PxgResult::Success,
                  "launch stSetupSpatialTest");

    tp.tendon = uint64_t(dB.ft);
    tp.constraint = uint64_t(dB.itcF);
    tp.elem = uint64_t(dB.tj);
    tp.maxTendons = SIC_MAX_FT;
    tp.maxElements = SIC_MAX_TJ;
    PXG_CHECK_MSG(shim.launchKernel(K_ST_FIXED, SIC_WARPS, 1, 1, &tp, sizeof(tp))
                      == PxgResult::Success,
                  "launch stSetupFixedTest");

    struct MjParams
    {
        uint64_t ab, lk, df, bf, mj;
        uint32_t nb, maxLinks, maxDofs, maxMimic, pad0, pad1;
    } mp = { uint64_t(dB.ab), uint64_t(dB.lk), uint64_t(dB.df),
             uint64_t(dB.bf), uint64_t(dB.mjb),
             SIC_NB, SIC_MAX_LINKS, SIC_MAX_DOFS, SIC_MAX_MJ, 0, 0 };
    PXG_CHECK_MSG(shim.launchKernel(K_MJ, SIC_WARPS, 1, 1, &mp, sizeof(mp))
                      == PxgResult::Success,
                  "launch mjSetupMimicTest");

    shim.streamSynchronize();

    // ---- non-vacuity: the reference run must have CHANGED side B ----
    {
        const size_t dItc = sicDelta(shim, dB.itcS, A.itcS.data(),
                                     A.itcS.size() * sizeof(A.itcS[0]));
        const size_t dDf = sicDelta(shim, dB.df, A.df.data(),
                                    A.df.size() * sizeof(A.df[0]));
        const size_t dMj = sicDelta(shim, dB.mjb, A.mjb.data(),
                                    A.mjb.size() * sizeof(A.mjb[0]));
        std::printf("[ info ] sic ref deltas vs seed: itcS %zu, dof %zu, mimic %zu bytes\n",
                    dItc, dDf, dMj);
        PXG_CHECK_MSG(dItc > 0 && dDf > 0 && dMj > 0,
                      "reference dispatches were no-ops -- the differential is vacuous");
    }

    // ---- the differential: every buffer, byte for byte ----
    size_t bad = 0;
    bad += sicCmp(shim, dA.ab, dB.ab, A.ab.size() * sizeof(A.ab[0]), "artiBlocks");
    bad += sicCmp(shim, dA.lk, dB.lk, A.lk.size() * sizeof(A.lk[0]), "linkBlocks");
    bad += sicCmp(shim, dA.df, dB.df, A.df.size() * sizeof(A.df[0]), "dofBlocks");
    bad += sicCmp(shim, dA.st, dB.st, A.st.size() * sizeof(A.st[0]), "spatialTendons");
    bad += sicCmp(shim, dA.ft, dB.ft, A.ft.size() * sizeof(A.ft[0]), "fixedTendons");
    bad += sicCmp(shim, dA.at, dB.at, A.at.size() * sizeof(A.at[0]), "attachments");
    bad += sicCmp(shim, dA.tj, dB.tj, A.tj.size() * sizeof(A.tj[0]), "tendonJoints");
    bad += sicCmp(shim, dA.itcS, dB.itcS, A.itcS.size() * sizeof(A.itcS[0]), "itcSpatial");
    bad += sicCmp(shim, dA.itcF, dB.itcF, A.itcF.size() * sizeof(A.itcF[0]), "itcFixed");
    bad += sicCmp(shim, dA.mjb, dB.mjb, A.mjb.size() * sizeof(A.mjb[0]), "mimicJoints");
    bad += sicCmp(shim, dA.bf, dB.bf, A.bf.size() * 8, "bitFields");

    // the descriptor itself must come back exactly as uploaded
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

PXG_TEST(artiSetupICLaunch)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiSetupICKernel.spv", "setupInternalConstraintLaunch1T", 24u, 0 },
        { "npArtiSetupInternal.spv", "siSetupInternalTest", 56u, 0 },
        { "npArtiSetupTendons.spv", "stSetupSpatialTest", 88u, 0 },
        { "npArtiSetupTendons.spv", "stSetupFixedTest", 88u, 0 },
        { "npArtiSetupMimic.spv", "mjSetupMimicTest", 64u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 5, &sicLoader);

    size_t bad = 0;
    std::printf("[ info ] sic: TGS round\n");
    bad += sicRun(shim, true);
    std::printf("[ info ] sic: PGS round\n");
    bad += sicRun(shim, false);
    PXG_CHECK_MSG(bad == 0, "%zu differential mismatches", bad);
}
