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


// P8.97c: npArtiSetupMimic.slang -- setupInternalMimicJointConstraints and
// the mimic self/cross response pair.
//
// THE ORACLE IS FULLY HOST-CALLABLE: the mimic path is built on the SHARED
// DyCpuGpuArticulation.h propagate pair (propagateImpulseW :494,
// propagateAccelerationW :549) and computeRecipMimicJointEffectiveInertia
// (:603), all PX_CUDA_CALLABLE -- the mirror calls the engine's own
// arithmetic and only the bitfield walks are transcribed control flow.
// Everything compares bit-exactly.
//
// WHAT THE FIXTURE IS SHAPED TO SEE
//   * the tendon fixture's topology (chain 0-1-2-3, branch 4 from 1, chain 5
//     from 4; numLinks 4..6; 1-or-2 dofs per link);
//   * mj0 on every articulation: linkA=2, linkB=3 -- A sits ON B's
//     root path, so the downward walk reads scratch the upward walk wrote;
//   * mj1 (numLinks >= 5): linkA=3, linkB=4 -- DISJOINT branches (common
//     ancestor 1): linkB's path never saw the upward walk, its dofs read
//     the freshly zeroed scratch;
//   * the axis of each end picks the link's HIGHEST dof (0 or 1 by parity),
//     so mInvDofIds resolves to both dof indices across the population;
//   * outputs: the mimic blocks (mDofA/mDofB/recipEffectiveInertia; the
//     rest must come back as seeded), the dof blocks (mDeferredQstZ ends
//     ZERO over mTotalDofs -- the exit-zeroing contract -- and poisoned
//     beyond), and the arti/link/bitfield buffers untouched.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "foundation/PxVec3.h"
#include "CmSpatialVector.h"
#include "DyFeatherstoneArticulationUtils.h"
#include "DyFeatherstoneArticulation.h"
#include "DyCpuGpuArticulation.h"
#include "PxSpatialMatrix.h"
#include "PxgArticulation.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using namespace physx;
using Cm::UnAlignedSpatialVector;
using Cm::SpatialVectorF;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

static std::vector<uint8_t> mjLoader(const char* file) { return loadShaderBlob(file); }

const uint32_t MJ_NB = 68, MJ_WARPS = 3;
const uint32_t MJ_MAX_LINKS = 6, MJ_MAX_DOFS = 8, MJ_MAX_MIMIC = 2;
const uint32_t MJ_WORDSIZE = 1;   // (6 + 63) / 64

static float fv(uint32_t a, uint32_t l, uint32_t d, uint32_t salt)
{
    uint32_t h = a * 2654435761u + l * 40503u + d * 9973u + salt * 733u + 23u;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    return -1.0f + 2.0f * float(h & 0xFFFFFFu) / float(0x1000000);
}

static uint32_t mjNumLinks(uint32_t a) { return 4u + (a % 3u); }
static uint32_t mjParent(uint32_t l) { return (l == 4) ? 1 : (l == 5 ? 4 : l - 1); }
static uint32_t mjDofs(uint32_t a, uint32_t l) { return 1u + ((a + l) % 2u); }

// ---- host mirror, control flow over the SHARED engine functions ----

static void mjLoadArrays(const PxgArticulationBlockDofData* df, PxU32 jointOffset,
                         PxU32 dofCount, PxU32 lane,
                         UnAlignedSpatialVector* motion, SpatialVectorF* isInvD,
                         SpatialVectorF* isW, Dy::InvStIs& invStIs)
{
    for (PxU32 i = 0; i < dofCount; ++i)
    {
        motion[i] = loadSpatialVector(df[jointOffset + i].mWorldMotionMatrix, lane);
        isInvD[i] = loadSpatialVectorF(df[jointOffset + i].mIsInvDW, lane);
        invStIs.invStIs[0][i] = df[jointOffset + i].mInvStIsT_x[lane];
        invStIs.invStIs[1][i] = df[jointOffset + i].mInvStIsT_y[lane];
        invStIs.invStIs[2][i] = df[jointOffset + i].mInvStIsT_z[lane];
        isW[i] = loadSpatialVectorF(df[jointOffset + i].mIsW, lane);
    }
}

static PxReal refSelfResponse(PxU32 linkIndex, PxU32 dof,
                              const PxgArticulationBlockLinkData* lk,
                              const PxgArticulationBlockDofData* df, PxU32 lane)
{
    const PxU32 parent = lk[linkIndex].mParents[lane];
    const PxVec3 rw(lk[linkIndex].mRw_x[lane], lk[linkIndex].mRw_y[lane],
                    lk[linkIndex].mRw_z[lane]);
    const PxU32 jointOffset = lk[linkIndex].mJointOffset[lane];
    const PxU8 dofCount = lk[linkIndex].mDofs[lane];

    const PxReal testJointImpulses[3][3] = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
    const PxReal* testJointImpulse = testJointImpulses[dof];

    UnAlignedSpatialVector motion[3];
    SpatialVectorF isInvD[3];
    SpatialVectorF isW[3];
    Dy::InvStIs invStIs;
    mjLoadArrays(df, jointOffset, dofCount, lane, motion, isInvD, isW, invStIs);

    PxReal QMinusStZ[3] = { 0.f, 0.f, 0.f };
    const SpatialVectorF Zp = Dy::propagateImpulseW(
        rw, SpatialVectorF(PxVec3(0, 0, 0), PxVec3(0, 0, 0)),
        testJointImpulse, isInvD, motion, dofCount, QMinusStZ);

    SpatialVectorF deltaVParent;
    {
        PxSpatialMatrix mat;
        loadSpatialMatrix(lk[parent].mSpatialResponseMatrix, lane, mat);
        const UnAlignedSpatialVector zu(Zp.top, Zp.bottom);
        const UnAlignedSpatialVector dv = mat * (-zu);
        deltaVParent = SpatialVectorF(dv.top, dv.bottom);
    }

    PxReal jointDeltaQDot[3] = { 0, 0, 0 };
    Dy::propagateAccelerationW(rw, deltaVParent, invStIs, motion, isW, QMinusStZ, dofCount,
                               jointDeltaQDot);
    return jointDeltaQDot[dof];
}

static PxReal refCrossResponse(PxU32 linkA, PxU32 dofA, PxU32 linkB, PxU32 dofB,
                               const PxgArticulationBlockData& ab,
                               const PxgArticulationBlockLinkData* lk,
                               PxgArticulationBlockDofData* df,
                               const uint64_t* bitRows /* [link * MJ_WORDSIZE + j][lane] */,
                               PxU32 lane)
{
    const PxReal testJointImpulses[3][3] = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };

    for (PxU32 i = 0; i < ab.mTotalDofs[lane]; ++i)
        df[i].mDeferredQstZ[lane] = 0.0f;

    SpatialVectorF Zp(PxVec3(0, 0, 0), PxVec3(0, 0, 0));
    for (PxI32 j = MJ_WORDSIZE - 1, bitOffset = (MJ_WORDSIZE - 1) * 64; j >= 0;
         j--, bitOffset -= 64)
    {
        PxU64 word = bitRows[(linkA * MJ_WORDSIZE + PxU32(j)) * 32 + lane];
        while (word)
        {
            const PxU32 bitIndex = 63 - PxU32(__builtin_clzll(word));
            const PxU32 linkIndex = bitIndex + PxU32(bitOffset);
            word &= (~(1ull << bitIndex));
            if (0 != linkIndex)
            {
                const PxReal* jointImpulse = NULL;
                SpatialVectorF linkImpulse(PxVec3(0, 0, 0), PxVec3(0, 0, 0));
                if (linkA == linkIndex)
                    jointImpulse = testJointImpulses[dofA];
                else
                    linkImpulse = Zp;

                const PxVec3 rw(lk[linkIndex].mRw_x[lane], lk[linkIndex].mRw_y[lane],
                                lk[linkIndex].mRw_z[lane]);
                const PxU32 jointOffset = lk[linkIndex].mJointOffset[lane];
                const PxU8 dofCount = lk[linkIndex].mDofs[lane];

                UnAlignedSpatialVector motion[3];
                SpatialVectorF isInvD[3];
                for (PxU32 i = 0; i < dofCount; ++i)
                {
                    motion[i] = loadSpatialVector(df[jointOffset + i].mWorldMotionMatrix, lane);
                    isInvD[i] = loadSpatialVectorF(df[jointOffset + i].mIsInvDW, lane);
                }

                PxReal QMinusStZ[3] = { 0, 0, 0 };
                Zp = Dy::propagateImpulseW(rw, linkImpulse, jointImpulse, isInvD, motion,
                                           dofCount, QMinusStZ);
                for (PxU32 i = 0; i < dofCount; ++i)
                    df[jointOffset + i].mDeferredQstZ[lane] = QMinusStZ[i];
            }
        }
    }

    SpatialVectorF deltaVRoot;
    {
        PxSpatialMatrix mat;
        loadSpatialMatrix(lk[0].mSpatialResponseMatrix, lane, mat);
        const UnAlignedSpatialVector zu(Zp.top, Zp.bottom);
        const UnAlignedSpatialVector dv = mat * (-zu);
        deltaVRoot = SpatialVectorF(dv.top, dv.bottom);
    }

    PxReal jointVelocity[3] = { 0, 0, 0 };
    SpatialVectorF deltaVParent = deltaVRoot;
    for (PxU32 j = 0, bitOffset = 0; j < MJ_WORDSIZE; j++, bitOffset += 64)
    {
        PxU64 word = bitRows[(linkB * MJ_WORDSIZE + j) * 32 + lane];
        while (word)
        {
            const PxU32 bitIndex = PxU32(__builtin_ctzll(word));
            const PxU32 linkIndex = bitIndex + bitOffset;
            word &= (~(1ull << bitIndex));
            if (linkIndex != 0)
            {
                const PxVec3 rw(lk[linkIndex].mRw_x[lane], lk[linkIndex].mRw_y[lane],
                                lk[linkIndex].mRw_z[lane]);
                const PxU32 jointOffset = lk[linkIndex].mJointOffset[lane];
                const PxU8 dofCount = lk[linkIndex].mDofs[lane];

                UnAlignedSpatialVector motion[3];
                SpatialVectorF isInvD[3];
                SpatialVectorF isW[3];
                Dy::InvStIs invStIs;
                mjLoadArrays(df, jointOffset, dofCount, lane, motion, isInvD, isW, invStIs);

                PxReal QMinusStZ[3] = { 0, 0, 0 };
                for (PxU32 i = 0; i < dofCount; ++i)
                    QMinusStZ[i] = df[jointOffset + i].mDeferredQstZ[lane];

                PxReal childJointSpeed[3] = { 0, 0, 0 };
                deltaVParent = Dy::propagateAccelerationW(rw, deltaVParent, invStIs, motion,
                                                          isW, QMinusStZ, dofCount,
                                                          childJointSpeed);
                jointVelocity[0] = childJointSpeed[0];
                jointVelocity[1] = childJointSpeed[1];
                jointVelocity[2] = childJointSpeed[2];
            }
        }
    }

    for (PxU32 i = 0; i < ab.mTotalDofs[lane]; ++i)
        df[i].mDeferredQstZ[lane] = 0.0f;

    return jointVelocity[dofB];
}

static void refSetupMimic(PxgArticulationBlockData& ab, PxgArticulationBlockLinkData* lk,
                          PxgArticulationBlockDofData* df, const uint64_t* bitRows,
                          PxgArticulationBlockMimicJointData* mj, PxU32 lane)
{
    const PxU32 nb = ab.mNumMimicJoints[lane];
    for (PxU32 i = 0; i < nb; ++i)
    {
        const PxU32 linkA = mj[i].mLinkA[lane];
        const PxU32 linkB = mj[i].mLinkB[lane];
        const PxU32 axisA = mj[i].mAxisA[lane];
        const PxU32 axisB = mj[i].mAxisB[lane];
        const PxU32 dofA = lk[linkA].mInvDofIds[axisA][lane];
        const PxU32 dofB = lk[linkB].mInvDofIds[axisB][lane];
        mj[i].mInternalData.mDofA[lane] = dofA;
        mj[i].mInternalData.mDofB[lane] = dofB;

        const PxReal rAA = refSelfResponse(linkA, dofA, lk, df, lane);
        const PxReal rBB = refSelfResponse(linkB, dofB, lk, df, lane);
        const PxReal rBA = refCrossResponse(linkA, dofA, linkB, dofB, ab, lk, df, bitRows, lane);
        const PxReal rAB = refCrossResponse(linkB, dofB, linkA, dofA, ab, lk, df, bitRows, lane);

        const PxReal gearRatio = mj[i].mGearRatio[lane];
        mj[i].mInternalData.recipEffectiveInertia[lane] =
            Dy::computeRecipMimicJointEffectiveInertia(rAA, rAB, rBB, rBA, gearRatio);
    }
}

// ---- the fixture ----

static size_t mjRun(PxgCudaShim& shim)
{
    std::vector<PxgArticulationBlockData> ab(MJ_WARPS);
    std::vector<PxgArticulationBlockLinkData> lk(size_t(MJ_WARPS) * MJ_MAX_LINKS);
    std::vector<PxgArticulationBlockDofData> df(size_t(MJ_WARPS) * MJ_MAX_DOFS);
    std::vector<PxgArticulationBlockMimicJointData> mj(size_t(MJ_WARPS) * MJ_MAX_MIMIC);
    // bitfield rows: [warp][link * wordSize + j][lane], PxU64[32] per row
    std::vector<uint64_t> bf(size_t(MJ_WARPS) * MJ_MAX_LINKS * MJ_WORDSIZE * 32);
    std::memset(ab.data(), 0xAB, ab.size() * sizeof(ab[0]));
    std::memset(lk.data(), 0xAB, lk.size() * sizeof(lk[0]));
    std::memset(df.data(), 0xAB, df.size() * sizeof(df[0]));
    std::memset(mj.data(), 0xAB, mj.size() * sizeof(mj[0]));
    std::memset(bf.data(), 0xAB, bf.size() * 8);

    for (uint32_t a = 0; a < MJ_NB; ++a)
    {
        const uint32_t w = a / 32, lane = a % 32;
        PxgArticulationBlockData& AB = ab[w];
        PxgArticulationBlockLinkData* L = &lk[size_t(w) * MJ_MAX_LINKS];
        PxgArticulationBlockDofData* D = &df[size_t(w) * MJ_MAX_DOFS];
        uint64_t* BF = &bf[size_t(w) * MJ_MAX_LINKS * MJ_WORDSIZE * 32];

        const uint32_t numLinks = mjNumLinks(a);
        AB.mNumLinks[lane] = numLinks;
        AB.mNumMimicJoints[lane] = numLinks >= 5 ? 2 : 1;

        uint32_t dofCursor = 0;
        for (uint32_t l = 0; l < numLinks; ++l)
        {
            PxgArticulationBlockLinkData& lnk = L[l];
            {
                PxSpatialMatrix m;
                for (uint32_t cI = 0; cI < 6; ++cI)
                    for (uint32_t r = 0; r < 6; ++r)
                        m.column[cI][r] = 0.2f * fv(a, l, cI * 6 + r, 61)
                                        + (cI == r ? 1.0f : 0.0f);
                storeSpatialMatrix(lnk.mSpatialResponseMatrix, lane, m);
            }
            // the path-to-root bitfield row (word 0; root bit included)
            uint64_t path = 1ull << 0;
            for (uint32_t p = l; p != 0; p = mjParent(p))
                path |= 1ull << p;
            BF[(l * MJ_WORDSIZE + 0) * 32 + lane] = path;

            if (l == 0)
            {
                lnk.mJointOffset[lane] = 0;
                lnk.mDofs[lane] = 0;
                // the root's rw is UNREAD by correct code (the walks skip
                // linkIndex 0) -- but 0xAB poison is ~1e-12 as a float, too
                // small to survive rounding when a mutant translates by it.
                // LARGE sentinels keep the poison discipline visible.
                lnk.mRw_x[lane] = 7.5f;
                lnk.mRw_y[lane] = -3.25f;
                lnk.mRw_z[lane] = 9.0f;
                continue;
            }
            lnk.mParents[lane] = mjParent(l);
            lnk.mRw_x[lane] = 0.5f * fv(a, l, 0, 62);
            lnk.mRw_y[lane] = 0.5f * fv(a, l, 1, 62);
            lnk.mRw_z[lane] = 0.5f * fv(a, l, 2, 62);
            const uint32_t nbDofs = mjDofs(a, l);
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
                        PxVec3(fv(a, l, d, 63), fv(a, l, d + 3, 63), fv(a, l, d + 6, 63)),
                        PxVec3(fv(a, l, d, 64), fv(a, l, d + 3, 64), fv(a, l, d + 6, 64))), lane);
                storeSpatialVector(dd.mIsInvDW,
                    UnAlignedSpatialVector(
                        PxVec3(fv(a, l, d, 65), fv(a, l, d + 3, 65), fv(a, l, d + 6, 65)),
                        PxVec3(fv(a, l, d, 66), fv(a, l, d + 3, 66), fv(a, l, d + 6, 66))), lane);
                storeSpatialVector(dd.mIsW,
                    UnAlignedSpatialVector(
                        PxVec3(fv(a, l, d, 67), fv(a, l, d + 3, 67), fv(a, l, d + 6, 67)),
                        PxVec3(fv(a, l, d, 68), fv(a, l, d + 3, 68), fv(a, l, d + 6, 68))), lane);
                dd.mInvStIsT_x[lane] = fv(a, l, d, 69);
                dd.mInvStIsT_y[lane] = fv(a, l, d, 70);
                dd.mInvStIsT_z[lane] = fv(a, l, d, 71);
            }
            dofCursor += nbDofs;
        }
        AB.mTotalDofs[lane] = dofCursor;

        PxgArticulationBlockMimicJointData* M = &mj[size_t(w) * MJ_MAX_MIMIC];
        // mj0: linkA=2 on linkB=3's root path
        M[0].mLinkA[lane] = 2;
        M[0].mLinkB[lane] = 3;
        M[0].mAxisA[lane] = (2u + (mjDofs(a, 2) - 1u)) % 6u;   // the link's HIGHEST dof
        M[0].mAxisB[lane] = (3u + (mjDofs(a, 3) - 1u)) % 6u;
        M[0].mGearRatio[lane] = 0.8f + 0.5f * fv(a, 20, 0, 72);
        if (numLinks >= 5)
        {
            // mj1: disjoint branches, common ancestor 1
            M[1].mLinkA[lane] = 3;
            M[1].mLinkB[lane] = 4;
            M[1].mAxisA[lane] = (3u + (mjDofs(a, 3) - 1u)) % 6u;
            M[1].mAxisB[lane] = (4u + (mjDofs(a, 4) - 1u)) % 6u;
            M[1].mGearRatio[lane] = -1.2f + 0.4f * fv(a, 21, 0, 73);
        }
    }

    // ---- mirror ----
    std::vector<PxgArticulationBlockDofData> wDf = df;
    std::vector<PxgArticulationBlockMimicJointData> wMj = mj;
    for (uint32_t a = 0; a < MJ_NB; ++a)
    {
        const uint32_t w = a / 32, lane = a % 32;
        refSetupMimic(ab[w], &lk[size_t(w) * MJ_MAX_LINKS], &wDf[size_t(w) * MJ_MAX_DOFS],
                      &bf[size_t(w) * MJ_MAX_LINKS * MJ_WORDSIZE * 32],
                      &wMj[size_t(w) * MJ_MAX_MIMIC], lane);
    }

    // ---- device ----
    std::vector<PxgDevicePtr> bufs;
    auto up = [&](const void* p, size_t n) {
        PxgDevicePtr dp = 0;
        shim.memAlloc(&dp, n);
        if (dp && p) shim.memcpyHtoD(dp, p, n);
        bufs.push_back(dp);
        return dp;
    };
    PxgDevicePtr dAb = up(ab.data(), ab.size() * sizeof(ab[0]));
    PxgDevicePtr dLk = up(lk.data(), lk.size() * sizeof(lk[0]));
    PxgDevicePtr dDf = up(df.data(), df.size() * sizeof(df[0]));
    PxgDevicePtr dBf = up(bf.data(), bf.size() * 8);
    PxgDevicePtr dMj = up(mj.data(), mj.size() * sizeof(mj[0]));
    if (!dAb || !dLk || !dDf || !dBf || !dMj)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return 0;
    }

    struct Params
    {
        uint64_t ab, lk, df, bf, mj;
        uint32_t nb, maxLinks, maxDofs, maxMimic, pad0, pad1;
    } prm = { uint64_t(dAb), uint64_t(dLk), uint64_t(dDf), uint64_t(dBf), uint64_t(dMj),
              MJ_NB, MJ_MAX_LINKS, MJ_MAX_DOFS, MJ_MAX_MIMIC, 0, 0 };
    PXG_CHECK_MSG(shim.launchKernel(0, MJ_WARPS, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch mjSetupMimicTest");
    shim.streamSynchronize();

    // ---- compare ----
    size_t bad = 0;
    auto cmp = [&](const char* what, PxgDevicePtr dp, const void* wp, size_t bytes) {
        std::vector<uint8_t> g(bytes);
        shim.memcpyDtoH(g.data(), dp, bytes);
        const uint32_t* gw = reinterpret_cast<const uint32_t*>(g.data());
        const uint32_t* ww = reinterpret_cast<const uint32_t*>(wp);
        for (size_t k = 0; k < bytes / 4; ++k)
            if (gw[k] != ww[k])
            {
                if (bad < 10)
                    std::printf("MM mj%s word %zu (byte %zu)  ref %08x  gpu %08x\n",
                                what, k, k * 4, ww[k], gw[k]);
                ++bad;
            }
    };
    cmp(" ab", dAb, ab.data(), ab.size() * sizeof(ab[0]));
    cmp(" lk", dLk, lk.data(), lk.size() * sizeof(lk[0]));
    cmp(" df", dDf, wDf.data(), wDf.size() * sizeof(wDf[0]));
    cmp(" bf", dBf, bf.data(), bf.size() * 8);
    cmp(" mj", dMj, wMj.data(), wMj.size() * sizeof(wMj[0]));

    for (PxgDevicePtr dp : bufs) if (dp) shim.memFree(dp);
    return bad;
}

PXG_TEST(artiSetupMimic)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiSetupMimic.spv", "mjSetupMimicTest", 64u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &mjLoader);

    const size_t bad = mjRun(shim);
    PXG_CHECK_MSG(bad == 0, "the mimic setup does not match the engine");
}

}  // namespace
