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


// P8.94b: npArtiFlushVel.slang — `PxcFsFlushVelocity`, the six-argument overload
// in articulationDynamic.cuh:390 that `stepArticulation1TTGS` calls.
//
// ===========================================================================
// THE ORACLE
// ===========================================================================
//
// The function is `static __device__`, so this is the P8.73-P8.80 arrangement:
// the mirror below is CONTROL FLOW over engine primitives.  Everything it is
// built from is `PX_CUDA_CALLABLE` and is called rather than transcribed —
// `Dy::SpatialMatrix::operator*`, `UnAlignedSpatialVector::innerProduct`,
// `Dy::FeatherstoneArticulation::translateSpatialVector`, `loadSpatialVector`,
// `storeSpatialVector`, `loadSpatialMatrix` — and each already has its own
// fixture underneath it from P8.86 and P8.90.  The rest is carried by
// whole-region poison comparison, which makes an EXTRA write fail as loudly as
// a wrong one, plus the sweep.
//
// ===========================================================================
// WHAT THE FIXTURE IS SHAPED TO SEE
// ===========================================================================
//
//   * **fixed-base and free-base lanes side by side**, because almost every
//     trap in this function is about which side of that guard a write is on;
//   * a lane with **exactly one link**, so the whole per-link loop is skipped
//     and the root writes must still land;
//   * a **branching chain** — at least one link whose parent is NOT `i - 1` —
//     which is the only way the `deltaV` re-load is distinguishable from the
//     carry;
//   * **dof counts of 0, 1, 2 and 3 in the same lane**, so the running cursor
//     is not a multiple of anything and a zero-dof link (which advances the
//     cursor by nothing) sits between two that do;
//   * every count distinct from the lane index and from each other, per the
//     standing rule, and every predicate keyed on the same quantity as the
//     branch it is meant to reach — the lesson P8.94a's first sweep paid for;
//   * every destination poisoned at 0xAB, including the fields the function
//     must NOT touch: a fixed base must leave `mRootDeferredZ` standing, and
//     nothing may write `mDeltaMotion`, `mPosMotionVelocity` or the dof block's
//     `mJointPositions`.

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
using physx::loadSpatialMatrix;

static std::vector<uint8_t> artiFlushLoader(const char* file) { return loadShaderBlob(file); }

// ---------------------------------------------------------------------------
// The fixture's shape
// ---------------------------------------------------------------------------

const uint32_t FV_LANES = 23;
const uint32_t FV_MAX_LINKS = 7;
const uint32_t FV_MAX_DOFS = 15;

// 1..7, never equal to the lane index: `f(k) - k == 1 - 7*floor(k/7)` is zero
// only at k == 0, and f(0) == 1 != 0.  Lane 6 gets exactly ONE link, which is
// the case that skips the entire per-link loop.
static uint32_t fvNumLinks(uint32_t k) { return 1u + (k % 7u); }

// A lane is fixed-base when `(k % 3) == 1`, so both sides of the guard occur
// several times and neither is a property of the lane's link count.
static bool fvFixBase(uint32_t k) { return (k % 3u) == 1u; }

// 0..3 dofs, cycling on `(k + i)` so a lane sees several distinct counts and a
// ZERO-dof link (which advances the cursor by nothing) sits between two that do.
static uint32_t fvDofs(uint32_t k, uint32_t i) { return (k + i) % 4u; }

// The parent chain.  Link 1 is always the root's child; from link 3 on, every
// third link re-parents to link 1 instead of to `i - 1`, which is the ONLY way
// the `parent != i - 1` re-load is distinguishable from the carry.  The
// predicate is keyed on `i`, i.e. on the same index the comparison uses.
static uint32_t fvParent(uint32_t i)
{
    if (i <= 1u) return 0u;
    return (i >= 3u && (i % 3u) == 0u) ? 1u : (i - 1u);
}

static uint32_t fvTotalDofs(uint32_t k, uint32_t numLinks)
{
    uint32_t t = 0;
    for (uint32_t i = 1; i < numLinks; ++i)
        t += fvDofs(k, i);
    return t;
}

static float fv(uint32_t i, uint32_t k)
{
    return (float((i * 61u + k * 17u) % 487u) + 1.0f) * 0.015625f - 3.5f;
}

// ---------------------------------------------------------------------------
// Seeding
// ---------------------------------------------------------------------------

static void fvFill(physx::PxgArticulationBlockData* ab,
                   physx::PxgArticulationBlockLinkData* lk,
                   physx::PxgArticulationBlockDofData* df, uint32_t lane, uint32_t numLinks,
                   uint32_t seed)
{
    // the articulation block: the root deferred impulse and the inverse
    // articulated inertia.  Both are read; only the first is written.
    storeSpatialVector(ab->mRootDeferredZ,
                       UnAlignedSpatialVector(PxVec3(fv(seed, 0), fv(seed, 1), fv(seed, 2)),
                                              PxVec3(fv(seed, 3), fv(seed, 4), fv(seed, 5))),
                       lane);
    {
        // a non-symmetric inverse inertia, so `operator*`'s transformTranspose
        // on the bottom half is distinguishable from a plain multiply
        physx::Dy::SpatialMatrix m;
        for (uint32_t c = 0; c < 3; ++c)
            for (uint32_t r = 0; r < 3; ++r)
            {
                m.topLeft[c][r] = fv(seed, 10u + c * 3u + r);
                m.topRight[c][r] = fv(seed, 20u + c * 3u + r);
                m.bottomLeft[c][r] = fv(seed, 30u + c * 3u + r);
            }
        physx::storeSpatialMatrix(ab->mInvSpatialArticulatedInertia, lane, m);
    }
    storeSpatialVector(ab->mCommonLinkDeltaVelocity,
                       UnAlignedSpatialVector(PxVec3(fv(seed, 40), fv(seed, 41), fv(seed, 42)),
                                              PxVec3(fv(seed, 43), fv(seed, 44), fv(seed, 45))),
                       lane);

    uint32_t cursor = 0;
    for (uint32_t i = 0; i < numLinks; ++i)
    {
        physx::PxgArticulationBlockLinkData& L = lk[i];
        const uint32_t dofs = (i == 0) ? 0u : fvDofs(seed, i);

        L.mDofs[lane] = uint8_t(dofs);
        L.mParents[lane] = (i == 0) ? 0u : fvParent(i);

        L.mRw_x[lane] = fv(seed, 50u + i * 3u);
        L.mRw_y[lane] = fv(seed, 51u + i * 3u);
        L.mRw_z[lane] = fv(seed, 52u + i * 3u);

        storeSpatialVector(L.mMotionVelocity,
                           UnAlignedSpatialVector(PxVec3(fv(seed, 80u + i * 6u),
                                                         fv(seed, 81u + i * 6u),
                                                         fv(seed, 82u + i * 6u)),
                                                  PxVec3(fv(seed, 83u + i * 6u),
                                                         fv(seed, 84u + i * 6u),
                                                         fv(seed, 85u + i * 6u))),
                           lane);
        storeSpatialVector(L.mSolverSpatialDeltaVel,
                           UnAlignedSpatialVector(PxVec3(fv(seed, 130u + i * 6u),
                                                         fv(seed, 131u + i * 6u),
                                                         fv(seed, 132u + i * 6u)),
                                                  PxVec3(fv(seed, 133u + i * 6u),
                                                         fv(seed, 134u + i * 6u),
                                                         fv(seed, 135u + i * 6u))),
                           lane);
        storeSpatialVector(L.mConstraintForces,
                           UnAlignedSpatialVector(PxVec3(fv(seed, 180u + i * 6u),
                                                         fv(seed, 181u + i * 6u),
                                                         fv(seed, 182u + i * 6u)),
                                                  PxVec3(fv(seed, 183u + i * 6u),
                                                         fv(seed, 184u + i * 6u),
                                                         fv(seed, 185u + i * 6u))),
                           lane);
        // mScratchDeltaV is written for every link and read for a re-parented
        // one, so it is seeded with something the function must OVERWRITE
        storeSpatialVector(L.mScratchDeltaV,
                           UnAlignedSpatialVector(PxVec3(fv(seed, 230u + i * 6u),
                                                         fv(seed, 231u + i * 6u),
                                                         fv(seed, 232u + i * 6u)),
                                                  PxVec3(fv(seed, 233u + i * 6u),
                                                         fv(seed, 234u + i * 6u),
                                                         fv(seed, 235u + i * 6u))),
                           lane);

        for (uint32_t dd = 0; dd < dofs; ++dd)
        {
            physx::PxgArticulationBlockDofData& d = df[cursor + dd];
            storeSpatialVector(d.mIsW,
                               UnAlignedSpatialVector(PxVec3(fv(seed, 300u + (cursor + dd) * 6u),
                                                             fv(seed, 301u + (cursor + dd) * 6u),
                                                             fv(seed, 302u + (cursor + dd) * 6u)),
                                                      PxVec3(fv(seed, 303u + (cursor + dd) * 6u),
                                                             fv(seed, 304u + (cursor + dd) * 6u),
                                                             fv(seed, 305u + (cursor + dd) * 6u))),
                               lane);
            storeSpatialVector(d.mWorldMotionMatrix,
                               UnAlignedSpatialVector(PxVec3(fv(seed, 400u + (cursor + dd) * 6u),
                                                             fv(seed, 401u + (cursor + dd) * 6u),
                                                             fv(seed, 402u + (cursor + dd) * 6u)),
                                                      PxVec3(fv(seed, 403u + (cursor + dd) * 6u),
                                                             fv(seed, 404u + (cursor + dd) * 6u),
                                                             fv(seed, 405u + (cursor + dd) * 6u))),
                               lane);
            d.mDeferredQstZ[lane] = fv(seed, 500u + cursor + dd);
            d.mJointVelocities[lane] = fv(seed, 540u + cursor + dd);
            d.mInvStIsT_x[lane] = fv(seed, 580u + cursor + dd);
            d.mInvStIsT_y[lane] = fv(seed, 620u + cursor + dd);
            d.mInvStIsT_z[lane] = fv(seed, 660u + cursor + dd);
        }
        cursor += dofs;
    }
}

// ---------------------------------------------------------------------------
// The mirror: the CUDA's structure, the engine's arithmetic
// ---------------------------------------------------------------------------

using pxgref::fvPropagateAccel;


using pxgref::fvMirror;


// ---------------------------------------------------------------------------

PXG_TEST(artiFlushVelocity)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiFlushVel.spv", "artiFlushVelocity", 48u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &artiFlushLoader);

    std::vector<physx::PxgArticulationBlockData> abImg(1);
    std::vector<physx::PxgArticulationBlockLinkData> lkImg(FV_MAX_LINKS);
    std::vector<physx::PxgArticulationBlockDofData> dfImg(FV_MAX_DOFS);
    std::memset(abImg.data(), 0xAB, abImg.size() * sizeof(physx::PxgArticulationBlockData));
    std::memset(lkImg.data(), 0xAB, lkImg.size() * sizeof(physx::PxgArticulationBlockLinkData));
    std::memset(dfImg.data(), 0xAB, dfImg.size() * sizeof(physx::PxgArticulationBlockDofData));

    std::vector<uint32_t> counts(32, 0u), flags(32, 0u);
    uint32_t nFixed = 0, nFree = 0, nSingle = 0, nBranch = 0, nZeroDof = 0;
    uint32_t seenDofs[4] = { 0, 0, 0, 0 };
    for (uint32_t lane = 0; lane < FV_LANES; ++lane)
    {
        const uint32_t nl = fvNumLinks(lane);
        counts[lane] = nl;
        flags[lane] = fvFixBase(lane) ? 1u : 0u;
        PXG_CHECK_MSG(nl <= FV_MAX_LINKS, "lane %u has %u links", lane, nl);
        PXG_CHECK_MSG(nl != lane, "lane %u's link count equals its index", lane);
        PXG_CHECK_MSG(fvTotalDofs(lane, nl) <= FV_MAX_DOFS, "lane %u needs %u dofs", lane,
                      fvTotalDofs(lane, nl));

        if (flags[lane]) ++nFixed; else ++nFree;
        if (nl == 1) ++nSingle;
        for (uint32_t i = 1; i < nl; ++i)
        {
            seenDofs[fvDofs(lane, i)] += 1u;
            if (fvDofs(lane, i) == 0) ++nZeroDof;
            if (fvParent(i) != i - 1) ++nBranch;
        }
        fvFill(abImg.data(), lkImg.data(), dfImg.data(), lane, nl, lane);
    }
    std::printf("[ info ] fv: %u fixed / %u free, %u single-link, %u re-parented links, "
                "%u zero-dof links, dof counts %u/%u/%u/%u\n",
                nFixed, nFree, nSingle, nBranch, nZeroDof, seenDofs[0], seenDofs[1], seenDofs[2],
                seenDofs[3]);
    PXG_CHECK_MSG(nFixed > 3 && nFree > 3, "the fixed-base guard is not straddled");
    PXG_CHECK_MSG(nSingle > 0, "no lane has exactly one link");
    PXG_CHECK_MSG(nBranch > 2, "no lane has a re-parented link");
    for (uint32_t d = 0; d < 4; ++d)
        PXG_CHECK_MSG(seenDofs[d] > 2, "dof count %u occurs only %u times", d, seenDofs[d]);

    // the expectation: the mirror run over a COPY of the seeded image
    std::vector<physx::PxgArticulationBlockData> wantAb = abImg;
    std::vector<physx::PxgArticulationBlockLinkData> wantLk = lkImg;
    std::vector<physx::PxgArticulationBlockDofData> wantDf = dfImg;
    for (uint32_t lane = 0; lane < FV_LANES; ++lane)
        fvMirror(wantAb.data(), wantLk.data(), wantDf.data(), counts[lane], flags[lane] != 0u,
                 lane);

    PxgDevicePtr dAb = 0, dLk = 0, dDf = 0, dCounts = 0, dFlags = 0;
    shim.memAlloc(&dAb, abImg.size() * sizeof(physx::PxgArticulationBlockData));
    shim.memAlloc(&dLk, lkImg.size() * sizeof(physx::PxgArticulationBlockLinkData));
    shim.memAlloc(&dDf, dfImg.size() * sizeof(physx::PxgArticulationBlockDofData));
    shim.memAlloc(&dCounts, counts.size() * sizeof(uint32_t));
    shim.memAlloc(&dFlags, flags.size() * sizeof(uint32_t));
    if (!dAb || !dLk || !dDf || !dCounts || !dFlags)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    shim.memcpyHtoD(dAb, abImg.data(), abImg.size() * sizeof(physx::PxgArticulationBlockData));
    shim.memcpyHtoD(dLk, lkImg.data(), lkImg.size() * sizeof(physx::PxgArticulationBlockLinkData));
    shim.memcpyHtoD(dDf, dfImg.data(), dfImg.size() * sizeof(physx::PxgArticulationBlockDofData));
    shim.memcpyHtoD(dCounts, counts.data(), counts.size() * sizeof(uint32_t));
    shim.memcpyHtoD(dFlags, flags.data(), flags.size() * sizeof(uint32_t));

    struct Params
    {
        uint64_t ab, lk, df, counts, flags;
        uint32_t n, pad;
    } prm = { uint64_t(dAb), uint64_t(dLk), uint64_t(dDf), uint64_t(dCounts), uint64_t(dFlags),
              FV_LANES, 0u };
    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch artiFlushVelocity");
    shim.streamSynchronize();

    std::vector<physx::PxgArticulationBlockData> gotAb(abImg.size());
    std::vector<physx::PxgArticulationBlockLinkData> gotLk(lkImg.size());
    std::vector<physx::PxgArticulationBlockDofData> gotDf(dfImg.size());
    shim.memcpyDtoH(gotAb.data(), dAb, gotAb.size() * sizeof(physx::PxgArticulationBlockData));
    shim.memcpyDtoH(gotLk.data(), dLk, gotLk.size() * sizeof(physx::PxgArticulationBlockLinkData));
    shim.memcpyDtoH(gotDf.data(), dDf, gotDf.size() * sizeof(physx::PxgArticulationBlockDofData));

    // Bit for bit, everywhere.  There is no transcendental anywhere in this
    // function, so unlike P8.94a there is nothing to tolerate -- only the
    // signed-zero pair P8.84 documented, counted and reported.
    size_t bad = 0, signedZeros = 0;
    auto cmp = [&](const char* what, const void* wp, const void* gp, size_t bytes) {
        const uint32_t* w = reinterpret_cast<const uint32_t*>(wp);
        const uint32_t* g = reinterpret_cast<const uint32_t*>(gp);
        for (size_t k = 0; k < bytes / 4; ++k)
        {
            if (w[k] == g[k])
                continue;
            if ((w[k] == 0u && g[k] == 0x80000000u) || (w[k] == 0x80000000u && g[k] == 0u))
            {
                ++signedZeros;
                continue;
            }
            if (bad < 12)
                std::printf("MM fv %s word %zu  ref %08x  gpu %08x\n", what, k, w[k], g[k]);
            ++bad;
        }
    };
    cmp("arti", wantAb.data(), gotAb.data(),
        wantAb.size() * sizeof(physx::PxgArticulationBlockData));
    cmp("link", wantLk.data(), gotLk.data(),
        wantLk.size() * sizeof(physx::PxgArticulationBlockLinkData));
    cmp("dof", wantDf.data(), gotDf.data(),
        wantDf.size() * sizeof(physx::PxgArticulationBlockDofData));

    std::printf("[ info ] fv: signed zeros %zu\n", signedZeros);
    PXG_CHECK_MSG(bad == 0, "the flushed block does not match the engine bit for bit");

    shim.memFree(dAb);
    shim.memFree(dLk);
    shim.memFree(dDf);
    shim.memFree(dCounts);
    shim.memFree(dFlags);
}

}  // namespace
