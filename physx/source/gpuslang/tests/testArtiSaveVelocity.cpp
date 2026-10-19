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


// P8.110: the `artiSaveVelocity1TPGS` engine entry (forwardDynamic2.cu:3255)
// -- the PGS save-velocities step: average pending impulses (eHAS_IMPULSES
// only), flush (any dirty), clear the flag, then snapshot mMotionVelocity ->
// mPosMotionVelocity and mJointVelocities -> mPosJointVelocities
// UNCONDITIONALLY.
//
// The mirror composes pxgref::aiMirror (P8.94c, engine-primitive control
// flow) and pxgref::fvMirror (P8.94b) plus the raw snapshot loops -- the
// same building blocks the shader entry composes on the device
// (aiAverageAndPropagate + fvFlushVelocity), each already swept on its own.
//
// WHAT THE FIXTURE IS SHAPED TO SEE
//
//   * mStateDirty seeds cycle {0,1,2,3}: clean lanes must SKIP average and
//     flush but still snapshot; eVEL_DIRTY-only lanes flush WITHOUT
//     averaging; eHAS_IMPULSES lanes average first;
//   * mTotalDofs is seeded ONE LESS than the true per-link sum on some
//     lanes: the dof snapshot is bounded by the header FIELD, and the last
//     dof's seeded mPosJointVelocities must survive there;
//   * mPos* targets seeded with distinct values (not poison): a lane that
//     skips its snapshot leaves them standing, which the byte compare sees;
//   * single-link lanes carry no eHAS_IMPULSES bit (the average path needs
//     a child link), covering flush-only and clean on one link;
//   * two warp slices, 40 live articulations, guard lanes past the end.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"
#include "pxgArtiRefHelpers.h"

#include "foundation/PxVec3.h"
#include "CmSpatialVector.h"
#include "DyFeatherstoneArticulation.h"
#include "PxgArticulation.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::PxReal;
using physx::PxU32;
using physx::PxVec3;
using physx::Cm::UnAlignedSpatialVector;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;
using physx::loadSpatialVector;
using physx::storeSpatialVector;

static std::vector<uint8_t> artiSvpLoader(const char* file) { return loadShaderBlob(file); }

// ---------------------------------------------------------------------------

const uint32_t SVP_NB_ARTI = 40;
const uint32_t SVP_NB_WARPS = 2;
const uint32_t SVP_MAX_LINKS = 7;
const uint32_t SVP_MAX_DOFS = 15;
const uint32_t SVP_NB_SLABS = 3;

const uint32_t SVP_D_NB_ARTICULATIONS = 44;
const uint32_t SVP_D_IMPULSES = 72;
const uint32_t SVP_D_NB_SLABS = 80;
const uint32_t SVP_D_SLAB_HAS_CHANGES = 88;
const uint32_t SVP_D_ARTICULATION_BLOCKS = 128;
const uint32_t SVP_D_LINK_BLOCKS = 136;
const uint32_t SVP_D_DOF_BLOCKS = 184;
const uint32_t SVP_D_MAX_LINKS = 248;
const uint32_t SVP_D_MAX_DOFS = 252;
const uint32_t SVP_DESC_SIZE = 344;
const uint32_t SVP_NONE = 0xFFFFFFFFu;

static uint32_t svpNumLinks(uint32_t k) { return 1u + (k % 7u); }
static bool svpFixBase(uint32_t k) { return (k % 3u) == 1u; }
// {0,1,2,3} on multi-link lanes; single-link lanes never carry bit 2
static uint32_t svpDirty(uint32_t k)
{
    const uint32_t v = k % 4u;
    return (svpNumLinks(k) == 1u) ? (v & 1u) : v;
}
static uint32_t svpFlags(uint32_t k)
{
    return (svpFixBase(k) ? 1u : 0u) | ((k & 1u) << 1) | 4u;
}
static uint32_t svpDofs(uint32_t k, uint32_t i) { return (k + i) % 4u; }
static uint32_t svpParent(uint32_t i)
{
    if (i <= 1u) return 0u;
    return (i >= 3u && (i % 3u) == 0u) ? 1u : (i - 1u);
}
static uint32_t svpSumDofs(uint32_t k)
{
    uint32_t t = 0;
    for (uint32_t i = 1; i < svpNumLinks(k); ++i)
        t += svpDofs(k, i);
    return t;
}
// trap 4: on every fifth lane with dofs, the header's total is ONE LESS than
// the true sum -- the snapshot loop must honour the field
static uint32_t svpTotalDofs(uint32_t k)
{
    const uint32_t t = svpSumDofs(k);
    return (t > 0u && (k % 5u) == 2u) ? (t - 1u) : t;
}

static float svp(uint32_t i, uint32_t k)
{
    return (float((i * 79u + k * 37u) % 509u) + 1.0f) * 0.015625f - 3.5f;
}
static UnAlignedSpatialVector svpSV(uint32_t seed, uint32_t base)
{
    return UnAlignedSpatialVector(
        PxVec3(svp(seed, base), svp(seed, base + 1), svp(seed, base + 2)),
        PxVec3(svp(seed, base + 3), svp(seed, base + 4), svp(seed, base + 5)));
}

// ---------------------------------------------------------------------------

static void svpFill(physx::PxgArticulationBlockData* ab,
                    physx::PxgArticulationBlockLinkData* lk,
                    physx::PxgArticulationBlockDofData* df, uint32_t* dirtySlabs, float* impulses,
                    uint32_t lane, uint32_t k)
{
    const uint32_t numLinks = svpNumLinks(k);

    ab->mNumLinks[lane] = numLinks;
    ab->mTotalDofs[lane] = svpTotalDofs(k);
    ab->mFlags[lane] = svpFlags(k);
    ab->mStateDirty[lane] = uint8_t(svpDirty(k));
    ab->mLinkWithDeferredImpulse[lane] = ((k % 4u) == 2u) ? (numLinks - 1u) : 0u;

    storeSpatialVector(ab->mRootDeferredZ, svpSV(k, 0), lane);
    {
        physx::Dy::SpatialMatrix m;
        for (uint32_t c = 0; c < 3; ++c)
            for (uint32_t r = 0; r < 3; ++r)
            {
                m.topLeft[c][r] = svp(k, 10u + c * 3u + r);
                m.topRight[c][r] = svp(k, 20u + c * 3u + r);
                m.bottomLeft[c][r] = svp(k, 30u + c * 3u + r);
            }
        physx::storeSpatialMatrix(ab->mInvSpatialArticulatedInertia, lane, m);
    }
    storeSpatialVector(ab->mCommonLinkDeltaVelocity, svpSV(k, 40), lane);

    // dirty slabs, only for lanes whose average path can run (numLinks > 1)
    if (numLinks > 1u && (svpDirty(k) & 2u) != 0u)
    {
        const uint32_t nd = 1u + (k % SVP_NB_SLABS);
        for (uint32_t s = 0; s < nd; ++s)
        {
            const size_t di = size_t(s) * SVP_NB_ARTI + k;
            const uint32_t mode = (k + s) % 3u;
            const uint32_t a = 1u + ((k + s) % (numLinks - 1u));
            const uint32_t b = 1u + ((k + s + 1u) % (numLinks - 1u));
            if (mode != 1u) dirtySlabs[di * 2 + 0] = a;
            if (mode != 0u) dirtySlabs[di * 2 + 1] = b;
        }
    }
    for (uint32_t s = 0; s < SVP_NB_SLABS; ++s)
        for (uint32_t l = 0; l < SVP_MAX_LINKS; ++l)
            for (uint32_t c = 0; c < 6; ++c)
                impulses[(size_t(s) * SVP_MAX_LINKS * SVP_NB_ARTI + size_t(l) * SVP_NB_ARTI + k)
                             * 6 + c] = svp(k, 900u + s * 50u + l * 7u + c) * 1.5f;

    for (uint32_t i = 0; i < numLinks; ++i)
    {
        physx::PxgArticulationBlockLinkData& L = lk[i];
        const uint32_t dofs = (i == 0) ? 0u : svpDofs(k, i);
        uint32_t cursor = 0;
        for (uint32_t j = 1; j < i; ++j)
            cursor += svpDofs(k, j);

        L.mDofs[lane] = uint8_t(dofs);
        L.mParents[lane] = (i == 0) ? 0u : svpParent(i);
        L.mJointOffset[lane] = (i == 0) ? 0u : cursor;

        L.mRw_x[lane] = svp(k, 50u + i * 3u);
        L.mRw_y[lane] = svp(k, 51u + i * 3u);
        L.mRw_z[lane] = svp(k, 52u + i * 3u);

        storeSpatialVector(L.mMotionVelocity, svpSV(k, 80u + i * 6u), lane);
        storeSpatialVector(L.mScratchImpulse, svpSV(k, 130u + i * 6u), lane);
        storeSpatialVector(L.mSolverSpatialImpulse, svpSV(k, 180u + i * 6u), lane);
        storeSpatialVector(L.mConstraintForces, svpSV(k, 230u + i * 6u), lane);
        storeSpatialVector(L.mScratchDeltaV, svpSV(k, 280u + i * 6u), lane);
        // the snapshot TARGET, seeded (not poison): a skipped copy leaves it
        storeSpatialVector(L.mPosMotionVelocity, svpSV(k, 730u + i * 6u), lane);
    }

    uint32_t cursor = 0;
    for (uint32_t i = 1; i < numLinks; ++i)
    {
        const uint32_t dofs = svpDofs(k, i);
        for (uint32_t dd = 0; dd < dofs; ++dd)
        {
            physx::PxgArticulationBlockDofData& d = df[cursor + dd];
            storeSpatialVector(d.mIsW, svpSV(k, 300u + (cursor + dd) * 6u), lane);
            storeSpatialVector(d.mIsInvDW, svpSV(k, 360u + (cursor + dd) * 6u), lane);
            storeSpatialVector(d.mWorldMotionMatrix, svpSV(k, 420u + (cursor + dd) * 6u), lane);
            d.mDeferredQstZ[lane] = svp(k, 500u + cursor + dd);
            d.mJointVelocities[lane] = svp(k, 540u + cursor + dd);
            d.mInvStIsT_x[lane] = svp(k, 580u + cursor + dd);
            d.mInvStIsT_y[lane] = svp(k, 620u + cursor + dd);
            d.mInvStIsT_z[lane] = svp(k, 660u + cursor + dd);
            d.mPosJointVelocities[lane] = svp(k, 700u + cursor + dd);   // seeded target
        }
        cursor += dofs;
    }
}

// ---------------------------------------------------------------------------

PXG_TEST(artiSaveVelocityPGS)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiPropagate2.spv", "artiSaveVelocity1TPGS", 8u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &artiSvpLoader);

    // coverage
    {
        uint32_t nClean = 0, nVelOnly = 0, nAvg = 0, nTotalShort = 0, nSingle = 0;
        for (uint32_t k = 0; k < SVP_NB_ARTI; ++k)
        {
            const uint32_t dv = svpDirty(k);
            if (dv == 0) ++nClean;
            else if ((dv & 2u) == 0u) ++nVelOnly;
            else ++nAvg;
            if (svpTotalDofs(k) != svpSumDofs(k)) ++nTotalShort;
            if (svpNumLinks(k) == 1u) ++nSingle;
        }
        std::printf("[ info ] svp: %u clean / %u vel-only / %u averaging, %u short totals, "
                    "%u single-link\n", nClean, nVelOnly, nAvg, nTotalShort, nSingle);
        PXG_CHECK_MSG(nClean > 3 && nVelOnly > 3 && nAvg > 3, "the dirty ladder is not covered");
        PXG_CHECK_MSG(nTotalShort > 2, "no lane pins the mTotalDofs field read");
        PXG_CHECK_MSG(nSingle > 0, "no single-link lane");
    }

    const size_t abBytes = size_t(SVP_NB_WARPS) * sizeof(physx::PxgArticulationBlockData);
    const size_t lkBytes =
        size_t(SVP_NB_WARPS) * SVP_MAX_LINKS * sizeof(physx::PxgArticulationBlockLinkData);
    const size_t dfBytes =
        size_t(SVP_NB_WARPS) * SVP_MAX_DOFS * sizeof(physx::PxgArticulationBlockDofData);

    std::vector<physx::PxgArticulationBlockData> abImg(SVP_NB_WARPS);
    std::vector<physx::PxgArticulationBlockLinkData> lkImg(size_t(SVP_NB_WARPS) * SVP_MAX_LINKS);
    std::vector<physx::PxgArticulationBlockDofData> dfImg(size_t(SVP_NB_WARPS) * SVP_MAX_DOFS);
    std::vector<uint32_t> dirtySlabs(size_t(SVP_NB_SLABS) * SVP_NB_ARTI * 2, SVP_NONE);
    std::vector<float> impulses(size_t(SVP_NB_SLABS) * SVP_MAX_LINKS * SVP_NB_ARTI * 6, 0.f);
    std::memset(abImg.data(), 0xAB, abBytes);
    std::memset(lkImg.data(), 0xAB, lkBytes);
    std::memset(dfImg.data(), 0xAB, dfBytes);

    for (uint32_t k = 0; k < SVP_NB_ARTI; ++k)
    {
        const uint32_t w = k / 32u, lane = k % 32u;
        svpFill(&abImg[w], &lkImg[size_t(w) * SVP_MAX_LINKS], &dfImg[size_t(w) * SVP_MAX_DOFS],
                dirtySlabs.data(), impulses.data(), lane, k);
    }

    // the expectation: the same composition the kernel runs, over engine
    // primitives
    std::vector<physx::PxgArticulationBlockData> wantAb = abImg;
    std::vector<physx::PxgArticulationBlockLinkData> wantLk = lkImg;
    std::vector<physx::PxgArticulationBlockDofData> wantDf = dfImg;
    std::vector<uint32_t> wantDirtySlabs = dirtySlabs;
    std::vector<float> wantImp = impulses;
    uint32_t ah = 0, uh = 0;
    for (uint32_t k = 0; k < SVP_NB_ARTI; ++k)
    {
        const uint32_t w = k / 32u, lane = k % 32u;
        physx::PxgArticulationBlockData* ab = &wantAb[w];
        physx::PxgArticulationBlockLinkData* lk = &wantLk[size_t(w) * SVP_MAX_LINKS];
        physx::PxgArticulationBlockDofData* df = &wantDf[size_t(w) * SVP_MAX_DOFS];
        const uint32_t nl = svpNumLinks(k);
        const uint32_t dv = ab->mStateDirty[lane];
        if (dv != 0)
        {
            if (dv & 2u)
                pxgref::aiMirror(wantDirtySlabs.data(), wantImp.data(), ab, lk, df, k,
                                 SVP_MAX_LINKS, SVP_NB_ARTI, SVP_NB_SLABS, lane, ah, uh);
            pxgref::fvMirror(ab, lk, df, nl, svpFixBase(k), lane);
            ab->mStateDirty[lane] = 0;
        }
        for (uint32_t i = 0; i < nl; ++i)
            storeSpatialVector(lk[i].mPosMotionVelocity,
                               loadSpatialVector(lk[i].mMotionVelocity, lane), lane);
        const uint32_t td = ab->mTotalDofs[lane];
        for (uint32_t dofId = 0; dofId < td; ++dofId)
            df[dofId].mPosJointVelocities[lane] = df[dofId].mJointVelocities[lane];
    }

    PxgDevicePtr dAb = 0, dLk = 0, dDf = 0, dSlab = 0, dImp = 0, dDesc = 0;
    shim.memAlloc(&dAb, abBytes);
    shim.memAlloc(&dLk, lkBytes);
    shim.memAlloc(&dDf, dfBytes);
    shim.memAlloc(&dSlab, dirtySlabs.size() * 4);
    shim.memAlloc(&dImp, impulses.size() * 4);
    shim.memAlloc(&dDesc, SVP_DESC_SIZE);
    if (!dAb || !dLk || !dDf || !dSlab || !dImp || !dDesc)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }

    std::vector<uint8_t> descImg(SVP_DESC_SIZE, 0xAB);
    auto putU32 = [&](uint32_t off, uint32_t v) { std::memcpy(descImg.data() + off, &v, 4); };
    auto putU64 = [&](uint32_t off, uint64_t v) { std::memcpy(descImg.data() + off, &v, 8); };
    putU32(SVP_D_NB_ARTICULATIONS, SVP_NB_ARTI);
    putU32(SVP_D_NB_SLABS, SVP_NB_SLABS);
    putU32(SVP_D_MAX_LINKS, SVP_MAX_LINKS);
    putU32(SVP_D_MAX_DOFS, SVP_MAX_DOFS);
    putU64(SVP_D_IMPULSES, uint64_t(dImp));
    putU64(SVP_D_SLAB_HAS_CHANGES, uint64_t(dSlab));
    putU64(SVP_D_ARTICULATION_BLOCKS, uint64_t(dAb));
    putU64(SVP_D_LINK_BLOCKS, uint64_t(dLk));
    putU64(SVP_D_DOF_BLOCKS, uint64_t(dDf));

    shim.memcpyHtoD(dAb, abImg.data(), abBytes);
    shim.memcpyHtoD(dLk, lkImg.data(), lkBytes);
    shim.memcpyHtoD(dDf, dfImg.data(), dfBytes);
    shim.memcpyHtoD(dSlab, dirtySlabs.data(), dirtySlabs.size() * 4);
    shim.memcpyHtoD(dImp, impulses.data(), impulses.size() * 4);
    shim.memcpyHtoD(dDesc, descImg.data(), descImg.size());

    struct Params
    {
        uint64_t desc;
    } prm = { uint64_t(dDesc) };
    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch artiSaveVelocity1TPGS");
    shim.streamSynchronize();

    std::vector<physx::PxgArticulationBlockData> gotAb(abImg.size());
    std::vector<physx::PxgArticulationBlockLinkData> gotLk(lkImg.size());
    std::vector<physx::PxgArticulationBlockDofData> gotDf(dfImg.size());
    std::vector<uint32_t> gotSlab(dirtySlabs.size());
    std::vector<float> gotImp(impulses.size());
    shim.memcpyDtoH(gotAb.data(), dAb, abBytes);
    shim.memcpyDtoH(gotLk.data(), dLk, lkBytes);
    shim.memcpyDtoH(gotDf.data(), dDf, dfBytes);
    shim.memcpyDtoH(gotSlab.data(), dSlab, gotSlab.size() * 4);
    shim.memcpyDtoH(gotImp.data(), dImp, gotImp.size() * 4);

    size_t bad = 0, signedZeros = 0;
    auto cmp = [&](const char* what, const void* wp, const void* gp, size_t bytes) {
        const uint32_t* w = reinterpret_cast<const uint32_t*>(wp);
        const uint32_t* g = reinterpret_cast<const uint32_t*>(gp);
        for (size_t x = 0; x < bytes / 4; ++x)
        {
            if (w[x] == g[x])
                continue;
            if ((w[x] == 0u && g[x] == 0x80000000u) || (w[x] == 0x80000000u && g[x] == 0u))
            {
                ++signedZeros;
                continue;
            }
            if (bad < 12)
                std::printf("MM svp %s word %zu  ref %08x  gpu %08x\n", what, x, w[x], g[x]);
            ++bad;
        }
    };
    cmp("arti", wantAb.data(), gotAb.data(), abBytes);
    cmp("link", wantLk.data(), gotLk.data(), lkBytes);
    cmp("dof", wantDf.data(), gotDf.data(), dfBytes);
    cmp("slabs", wantDirtySlabs.data(), gotSlab.data(), wantDirtySlabs.size() * 4);
    cmp("impulses", wantImp.data(), gotImp.data(), wantImp.size() * 4);

    std::printf("[ info ] svp: signed zeros %zu (aligned %u / unaligned %u impulse hits)\n",
                signedZeros, ah, uh);
    PXG_CHECK_MSG(bad == 0, "the saved velocities do not match the mirror bit for bit");

    shim.memFree(dAb);
    shim.memFree(dLk);
    shim.memFree(dDf);
    shim.memFree(dSlab);
    shim.memFree(dImp);
    shim.memFree(dDesc);
}

}  // namespace
