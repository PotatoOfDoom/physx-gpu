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


// P8.105: the `artiOutputVelocity` engine entry (forwardDynamic2.cu:4076) --
// the launch fixture over the EIGHT-argument PxcFsFlushVelocity overload plus
// the clean-copy branch and the isTGS mDeltaMotion tail.
//
// ===========================================================================
// THE ORACLE
// ===========================================================================
//
// The eight-argument overload's mirror is `pxgref::fvMirrorOut` -- P8.94b's
// fvMirror minus the two mSolverSpatialDeltaVel accumulations, plus the
// solver outVelocity writes.  Everything numeric in it is a PX_CUDA_CALLABLE
// engine primitive (`SpatialMatrix::operator*`, `loadSpatialVector`, ...).
// The kernel around it -- the guard, the dirty dispatch, the clean copy, the
// TGS tail, the per-articulation output slice -- is mirrored as control flow
// here, carried by whole-region poison comparison and the sweep.
//
// ===========================================================================
// WHAT THE FIXTURE IS SHAPED TO SEE
// ===========================================================================
//
//   * TWO warp slices (grid (2,1,1)), 40 live articulations: warp 1's lanes
//     8..31 are past `nbArticulations`, so the guard has real work -- their
//     block lanes stay poisoned and their output slice stays poisoned;
//   * mStateDirty in {0, 1, 2, 3} with period 4, fixBase with period 3, so
//     dirty/clean and fixed/free straddle each other in the SAME warp (the
//     P8.94a lesson: never key two predicates on the same residue);
//   * mFlags carries DECOY bits above eFIX_BASE (bit 0), so a port that
//     tests the whole word instead of `& 1` dies;
//   * link counts 1..7 (a single-link lane skips the whole per-link loop),
//     re-parented links, zero-dof links between live ones -- the FV shape;
//   * mDeltaMotion seeded distinctly from mMotionVelocity, so the four
//     output regions [i], [i+offset], [i+2*offset], [i+3*offset] can never
//     alias each other's values;
//   * two rounds, isTGS = true and false, freshly seeded (the dirty path
//     mutates the blocks), so the tail's gate is pinned from both sides;
//   * the output buffer is 4*offset float4s of 0xAB poison: an out-of-slice
//     write (wrong base, wrong offset multiple, guard breach) lands in
//     another articulation's region and fails the byte compare there.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"
#include "pxgArtiRefHelpers.h"

#include "foundation/PxVec3.h"
#include "CmSpatialVector.h"
#include "DyFeatherstoneArticulation.h"
#include "PxgArticulation.h"
#include "PxgSolverCoreDesc.h"

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

static std::vector<uint8_t> artiOutVelLoader(const char* file) { return loadShaderBlob(file); }

// ---------------------------------------------------------------------------
// The fixture's shape
// ---------------------------------------------------------------------------

const uint32_t OV_NB_ARTI = 40;     // two warp slices; warp 1 is 8/32 live
const uint32_t OV_NB_WARPS = 2;
const uint32_t OV_MAX_LINKS = 7;
const uint32_t OV_MAX_DOFS = 15;
const uint32_t OV_OFFSET = OV_NB_ARTI * OV_MAX_LINKS;   // 280 float4s per region

// PXG_ARTI_CORE_* offsets, pinned by testArtiCoreDescLayout
const uint32_t OV_D_NB_ARTICULATIONS = 44;
const uint32_t OV_D_ARTICULATION_BLOCKS = 128;
const uint32_t OV_D_LINK_BLOCKS = 136;
const uint32_t OV_D_DOF_BLOCKS = 184;
const uint32_t OV_D_MAX_LINKS = 248;
const uint32_t OV_D_MAX_DOFS = 252;
const uint32_t OV_DESC_SIZE = 360;

// 1..7, never equal to the articulation index (see testArtiFlushVel.cpp).
static uint32_t ovNumLinks(uint32_t k) { return 1u + (k % 7u); }

// Period 3: fixed-base at k % 3 == 1.
static bool ovFixBase(uint32_t k) { return (k % 3u) == 1u; }

// Period 4, COPRIME with the fixBase period: every (dirty, fixBase)
// combination occurs, including dirty fixed-base and clean free-base.
static uint32_t ovDirty(uint32_t k) { return k % 4u; }

// eFIX_BASE is bit 0; everything above it is a decoy the kernel must mask off.
static uint32_t ovFlags(uint32_t k)
{
    return (ovFixBase(k) ? 1u : 0u) | ((k & 1u) << 1) | 8u | ((k % 5u) << 4);
}

// 0..3 dofs per link, cycling on (k + i); link 0 has none.
static uint32_t ovDofs(uint32_t k, uint32_t i) { return (k + i) % 4u; }

// Link 1 hangs off the root; from link 3 on every third link re-parents to 1.
static uint32_t ovParent(uint32_t i)
{
    if (i <= 1u) return 0u;
    return (i >= 3u && (i % 3u) == 0u) ? 1u : (i - 1u);
}

static uint32_t ovTotalDofs(uint32_t k, uint32_t numLinks)
{
    uint32_t t = 0;
    for (uint32_t i = 1; i < numLinks; ++i)
        t += ovDofs(k, i);
    return t;
}

static float ov(uint32_t i, uint32_t k)
{
    return (float((i * 67u + k * 23u) % 491u) + 1.0f) * 0.015625f - 3.5f;
}

static UnAlignedSpatialVector ovSV(uint32_t seed, uint32_t base)
{
    return UnAlignedSpatialVector(PxVec3(ov(seed, base), ov(seed, base + 1), ov(seed, base + 2)),
                                  PxVec3(ov(seed, base + 3), ov(seed, base + 4), ov(seed, base + 5)));
}

// ---------------------------------------------------------------------------
// Seeding: the FV shape plus the kernel's header fields and mDeltaMotion
// ---------------------------------------------------------------------------

static void ovFill(physx::PxgArticulationBlockData* ab,
                   physx::PxgArticulationBlockLinkData* lk,
                   physx::PxgArticulationBlockDofData* df, uint32_t lane, uint32_t k,
                   uint32_t seed)
{
    const uint32_t numLinks = ovNumLinks(k);

    ab->mNumLinks[lane] = numLinks;
    ab->mFlags[lane] = ovFlags(k);
    ab->mStateDirty[lane] = uint8_t(ovDirty(k));

    storeSpatialVector(ab->mRootDeferredZ, ovSV(seed, 0), lane);
    {
        // non-symmetric, so operator*'s transformTranspose is distinguishable
        physx::Dy::SpatialMatrix m;
        for (uint32_t c = 0; c < 3; ++c)
            for (uint32_t r = 0; r < 3; ++r)
            {
                m.topLeft[c][r] = ov(seed, 10u + c * 3u + r);
                m.topRight[c][r] = ov(seed, 20u + c * 3u + r);
                m.bottomLeft[c][r] = ov(seed, 30u + c * 3u + r);
            }
        physx::storeSpatialMatrix(ab->mInvSpatialArticulatedInertia, lane, m);
    }
    storeSpatialVector(ab->mCommonLinkDeltaVelocity, ovSV(seed, 40), lane);

    uint32_t cursor = 0;
    for (uint32_t i = 0; i < numLinks; ++i)
    {
        physx::PxgArticulationBlockLinkData& L = lk[i];
        const uint32_t dofs = (i == 0) ? 0u : ovDofs(seed, i);

        L.mDofs[lane] = uint8_t(dofs);
        L.mParents[lane] = (i == 0) ? 0u : ovParent(i);

        L.mRw_x[lane] = ov(seed, 50u + i * 3u);
        L.mRw_y[lane] = ov(seed, 51u + i * 3u);
        L.mRw_z[lane] = ov(seed, 52u + i * 3u);

        storeSpatialVector(L.mMotionVelocity, ovSV(seed, 80u + i * 6u), lane);
        storeSpatialVector(L.mConstraintForces, ovSV(seed, 180u + i * 6u), lane);
        storeSpatialVector(L.mScratchDeltaV, ovSV(seed, 230u + i * 6u), lane);
        // distinct from mMotionVelocity, so the four output regions can
        // never carry each other's values by coincidence
        storeSpatialVector(L.mDeltaMotion, ovSV(seed, 700u + i * 6u), lane);

        for (uint32_t dd = 0; dd < dofs; ++dd)
        {
            physx::PxgArticulationBlockDofData& d = df[cursor + dd];
            storeSpatialVector(d.mIsW, ovSV(seed, 300u + (cursor + dd) * 6u), lane);
            storeSpatialVector(d.mWorldMotionMatrix, ovSV(seed, 400u + (cursor + dd) * 6u), lane);
            d.mDeferredQstZ[lane] = ov(seed, 500u + cursor + dd);
            d.mJointVelocities[lane] = ov(seed, 540u + cursor + dd);
            d.mInvStIsT_x[lane] = ov(seed, 580u + cursor + dd);
            d.mInvStIsT_y[lane] = ov(seed, 620u + cursor + dd);
            d.mInvStIsT_z[lane] = ov(seed, 660u + cursor + dd);
        }
        cursor += dofs;
    }
}

// ---------------------------------------------------------------------------

PXG_TEST(artiOutputVelocity)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiFlushVel.spv", "artiOutputVelocity", 24u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &artiOutVelLoader);

    // coverage: the fixture must actually contain the cases it claims to
    {
        uint32_t nDirtyFixed = 0, nDirtyFree = 0, nClean = 0, nSingleDirty = 0, nSingleClean = 0;
        uint32_t nWarp1Dirty = 0, nWarp1Clean = 0, nBranch = 0, nZeroDof = 0;
        uint32_t seenDofs[4] = { 0, 0, 0, 0 };
        for (uint32_t k = 0; k < OV_NB_ARTI; ++k)
        {
            const uint32_t nl = ovNumLinks(k);
            PXG_CHECK_MSG(nl <= OV_MAX_LINKS, "arti %u has %u links", k, nl);
            PXG_CHECK_MSG(ovTotalDofs(k, nl) <= OV_MAX_DOFS, "arti %u needs %u dofs", k,
                          ovTotalDofs(k, nl));
            const bool dirty = ovDirty(k) != 0u;
            if (dirty) { if (ovFixBase(k)) ++nDirtyFixed; else ++nDirtyFree; }
            else ++nClean;
            if (nl == 1) { if (dirty) ++nSingleDirty; else ++nSingleClean; }
            if (k >= 32) { if (dirty) ++nWarp1Dirty; else ++nWarp1Clean; }
            for (uint32_t i = 1; i < nl; ++i)
            {
                seenDofs[ovDofs(k, i)] += 1u;
                if (ovDofs(k, i) == 0) ++nZeroDof;
                if (ovParent(i) != i - 1) ++nBranch;
            }
        }
        std::printf("[ info ] ov: %u dirty-fixed / %u dirty-free / %u clean, "
                    "single-link %u dirty + %u clean, warp1 %u dirty + %u clean, "
                    "%u re-parented, %u zero-dof, dofs %u/%u/%u/%u\n",
                    nDirtyFixed, nDirtyFree, nClean, nSingleDirty, nSingleClean, nWarp1Dirty,
                    nWarp1Clean, nBranch, nZeroDof, seenDofs[0], seenDofs[1], seenDofs[2],
                    seenDofs[3]);
        PXG_CHECK_MSG(nDirtyFixed > 3 && nDirtyFree > 3, "the fixed-base guard is not straddled");
        PXG_CHECK_MSG(nClean > 3, "too few clean lanes");
        PXG_CHECK_MSG(nSingleDirty > 0 && nSingleClean > 0, "no single-link lane on both paths");
        PXG_CHECK_MSG(nWarp1Dirty > 0 && nWarp1Clean > 0, "warp 1 misses a dirty state");
        PXG_CHECK_MSG(nBranch > 2, "no re-parented link");
        for (uint32_t d = 0; d < 4; ++d)
            PXG_CHECK_MSG(seenDofs[d] > 2, "dof count %u occurs only %u times", d, seenDofs[d]);
    }

    const size_t abBytes = size_t(OV_NB_WARPS) * sizeof(physx::PxgArticulationBlockData);
    const size_t lkBytes =
        size_t(OV_NB_WARPS) * OV_MAX_LINKS * sizeof(physx::PxgArticulationBlockLinkData);
    const size_t dfBytes =
        size_t(OV_NB_WARPS) * OV_MAX_DOFS * sizeof(physx::PxgArticulationBlockDofData);
    const size_t outBytes = size_t(4) * OV_OFFSET * sizeof(float4);

    PxgDevicePtr dAb = 0, dLk = 0, dDf = 0, dOut = 0, dDesc = 0, dSolver = 0;
    shim.memAlloc(&dAb, abBytes);
    shim.memAlloc(&dLk, lkBytes);
    shim.memAlloc(&dDf, dfBytes);
    shim.memAlloc(&dOut, outBytes);
    shim.memAlloc(&dDesc, OV_DESC_SIZE);
    shim.memAlloc(&dSolver, sizeof(physx::PxgSolverCoreDesc));
    if (!dAb || !dLk || !dDf || !dOut || !dDesc || !dSolver)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }

    // the 360-byte PxgArticulationCoreDesc image: scalars + the three pools,
    // everything else 0xAB
    std::vector<uint8_t> descImg(OV_DESC_SIZE, 0xAB);
    auto putU32 = [&](uint32_t off, uint32_t v) { std::memcpy(descImg.data() + off, &v, 4); };
    auto putU64 = [&](uint32_t off, uint64_t v) { std::memcpy(descImg.data() + off, &v, 8); };
    putU32(OV_D_NB_ARTICULATIONS, OV_NB_ARTI);
    putU32(OV_D_MAX_LINKS, OV_MAX_LINKS);
    putU32(OV_D_MAX_DOFS, OV_MAX_DOFS);
    putU64(OV_D_ARTICULATION_BLOCKS, uint64_t(dAb));
    putU64(OV_D_LINK_BLOCKS, uint64_t(dLk));
    putU64(OV_D_DOF_BLOCKS, uint64_t(dDf));
    shim.memcpyHtoD(dDesc, descImg.data(), descImg.size());

    physx::PxgSolverCoreDesc solverDesc;
    std::memset(&solverDesc, 0xAB, sizeof(solverDesc));
    solverDesc.outArtiVelocity = reinterpret_cast<float4*>(uint64_t(dOut));
    shim.memcpyHtoD(dSolver, &solverDesc, sizeof(solverDesc));

    // two rounds: the isTGS tail gated from both sides, freshly seeded each
    // time because the dirty path mutates the blocks
    for (uint32_t round = 0; round < 2; ++round)
    {
        const bool isTGS = (round == 0);
        const uint32_t salt = round * 97u;

        std::vector<physx::PxgArticulationBlockData> abImg(OV_NB_WARPS);
        std::vector<physx::PxgArticulationBlockLinkData> lkImg(size_t(OV_NB_WARPS) * OV_MAX_LINKS);
        std::vector<physx::PxgArticulationBlockDofData> dfImg(size_t(OV_NB_WARPS) * OV_MAX_DOFS);
        std::vector<float4> outImg(size_t(4) * OV_OFFSET);
        std::memset(abImg.data(), 0xAB, abBytes);
        std::memset(lkImg.data(), 0xAB, lkBytes);
        std::memset(dfImg.data(), 0xAB, dfBytes);
        std::memset(outImg.data(), 0xAB, outBytes);

        for (uint32_t k = 0; k < OV_NB_ARTI; ++k)
        {
            const uint32_t w = k / 32u, lane = k % 32u;
            ovFill(&abImg[w], &lkImg[size_t(w) * OV_MAX_LINKS], &dfImg[size_t(w) * OV_MAX_DOFS],
                   lane, k, k + salt);
        }

        // the expectation: the mirror over a COPY of the seeded images
        std::vector<physx::PxgArticulationBlockData> wantAb = abImg;
        std::vector<physx::PxgArticulationBlockLinkData> wantLk = lkImg;
        std::vector<physx::PxgArticulationBlockDofData> wantDf = dfImg;
        std::vector<float4> wantOut = outImg;
        for (uint32_t k = 0; k < OV_NB_ARTI; ++k)
        {
            const uint32_t w = k / 32u, lane = k % 32u;
            physx::PxgArticulationBlockData* ab = &wantAb[w];
            physx::PxgArticulationBlockLinkData* lk = &wantLk[size_t(w) * OV_MAX_LINKS];
            physx::PxgArticulationBlockDofData* df = &wantDf[size_t(w) * OV_MAX_DOFS];
            float4* outBase = wantOut.data() + size_t(k) * OV_MAX_LINKS;
            const uint32_t nl = ovNumLinks(k);
            if (ovDirty(k) != 0u)
            {
                pxgref::fvMirrorOut(ab, lk, df, nl, ovFixBase(k), lane, outBase, OV_OFFSET);
                ab->mStateDirty[lane] = 0;
            }
            else
            {
                for (uint32_t i = 0; i < nl; ++i)
                {
                    const UnAlignedSpatialVector vel =
                        loadSpatialVector(lk[i].mMotionVelocity, lane);
                    outBase[i] = make_float4(vel.bottom.x, vel.bottom.y, vel.bottom.z, 0.f);
                    outBase[i + OV_OFFSET] = make_float4(vel.top.x, vel.top.y, vel.top.z, 0.f);
                }
            }
            if (isTGS)
            {
                for (uint32_t i = 0; i < nl; ++i)
                {
                    const UnAlignedSpatialVector delta =
                        loadSpatialVector(lk[i].mDeltaMotion, lane);
                    outBase[i + 2 * OV_OFFSET] =
                        make_float4(delta.bottom.x, delta.bottom.y, delta.bottom.z, 0.f);
                    outBase[i + 3 * OV_OFFSET] =
                        make_float4(delta.top.x, delta.top.y, delta.top.z, 0.f);
                }
            }
        }

        shim.memcpyHtoD(dAb, abImg.data(), abBytes);
        shim.memcpyHtoD(dLk, lkImg.data(), lkBytes);
        shim.memcpyHtoD(dDf, dfImg.data(), dfBytes);
        shim.memcpyHtoD(dOut, outImg.data(), outBytes);

        struct Params
        {
            uint64_t desc, solverDesc;
            uint32_t isTGSWord, pad;
        } prm = { uint64_t(dDesc), uint64_t(dSolver), isTGS ? 1u : 0u, 0u };
        PXG_CHECK_MSG(shim.launchKernel(0, OV_NB_WARPS, 1, 1, &prm, sizeof(prm)) ==
                          PxgResult::Success,
                      "launch artiOutputVelocity");
        shim.streamSynchronize();

        std::vector<physx::PxgArticulationBlockData> gotAb(abImg.size());
        std::vector<physx::PxgArticulationBlockLinkData> gotLk(lkImg.size());
        std::vector<physx::PxgArticulationBlockDofData> gotDf(dfImg.size());
        std::vector<float4> gotOut(outImg.size());
        shim.memcpyDtoH(gotAb.data(), dAb, abBytes);
        shim.memcpyDtoH(gotLk.data(), dLk, lkBytes);
        shim.memcpyDtoH(gotDf.data(), dDf, dfBytes);
        shim.memcpyDtoH(gotOut.data(), dOut, outBytes);

        // bit for bit, with the counted signed-zero pair (P8.84) tolerated
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
                    std::printf("MM ov r%u %s word %zu  ref %08x  gpu %08x\n", round, what, x,
                                w[x], g[x]);
                ++bad;
            }
        };
        cmp("arti", wantAb.data(), gotAb.data(), abBytes);
        cmp("link", wantLk.data(), gotLk.data(), lkBytes);
        cmp("dof", wantDf.data(), gotDf.data(), dfBytes);
        cmp("out", wantOut.data(), gotOut.data(), outBytes);

        std::printf("[ info ] ov round %u (isTGS=%u): signed zeros %zu\n", round,
                    isTGS ? 1u : 0u, signedZeros);
        PXG_CHECK_MSG(bad == 0, "round %u does not match the mirror bit for bit", round);
    }

    shim.memFree(dAb);
    shim.memFree(dLk);
    shim.memFree(dDf);
    shim.memFree(dOut);
    shim.memFree(dDesc);
    shim.memFree(dSolver);
}

}  // namespace
