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

// P8.98a: `pxcFsGetVelocity` + `pxcFsApplyImpulses`
// (internalConstraints2.cu:2116 / :2190) — the two deferred-impulse
// workhorses every internal-constraint solve body composes.
//
// THE ORACLE: control flow over engine primitives (the P8.73-P8.80
// arrangement).  `loadSpatialMatrix` / `loadSpatialVector` /
// `Dy::SpatialMatrix::operator*` are PX_CUDA_CALLABLE and are called; the
// propagate hops are transcriptions of the articulationDynamic.cuh template
// instances (both are __device__-only), spelled in the .cuh order: the
// impulse hop seeds temp from Z and adds per-dof terms (NOT the shared
// DyCpuGpuArticulation.h order — the P8.97c lesson), with `stZ = jf - dot`
// (the template spelling, which differs from the preloaded variant's `-dot`
// in signed zero at dot == +0).
//
// WHAT THE FIXTURE IS SHAPED TO SEE
//  * topology with a BRANCH at link 1 (parents 0,1,1,2,3) so a disjoint pair
//    (2,3) with common ancestor 1 exists in EVERY lane — the three-walk shape
//    of applyImpulses with both walks non-empty;
//  * getVelocity cases covering the root link (walk skips everything),
//    fixBase both ways (the root deltaV guard), wantSpeeds both ways (the
//    -777 sentinels must survive the noSpeeds cases);
//  * applyImpulses cases: disjoint pair, linkID0 == common (joint impulse
//    moved to the common walk), identical links (both impulses accumulate at
//    the common link), common == root (the common walk must NOT run and the
//    root-side joint impulse is dropped), linkID1 == common;
//  * mDeferredQstZ seeded NON-zero (the hops read AND accumulate it),
//    mSolverSpatialImpulse seeded (accumulated, not overwritten), root rw
//    large sentinels (the P8.97c poison-size lesson);
//  * after the getVelocity dispatch every input buffer must be UNTOUCHED
//    (it is a pure read), checked byte for byte; after applyImpulses the
//    whole arena is compared against the mirror byte for byte.

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

static std::vector<uint8_t> svLoader(const char* file) { return loadShaderBlob(file); }

const uint32_t SV_NB = 68, SV_WARPS = 3;
const uint32_t SV_MAX_LINKS = 6, SV_MAX_DOFS = 8, SV_WORDSIZE = 1;

static float fv(uint32_t a, uint32_t l, uint32_t d, uint32_t salt)
{
    uint32_t h = a * 2654435761u + l * 40503u + d * 9973u + salt * 733u + 19u;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    return -1.0f + 2.0f * float(h & 0xFFFFFFu) / float(0x1000000);
}

static uint32_t svNumLinks(uint32_t a) { return 4u + (a % 3u); }
// branch at link 1: parents 0,1,1,2,3
static uint32_t svParent(uint32_t l)
{
    static const uint32_t p[6] = { 0, 0, 1, 1, 2, 3 };
    return p[l];
}
static uint32_t svDofs(uint32_t a, uint32_t l) { return 1u + ((a + l) % 2u); }

// ---- the seed ----

struct SvArena
{
    std::vector<PxgArticulationBlockData> ab;
    std::vector<PxgArticulationBlockLinkData> lk;
    std::vector<PxgArticulationBlockDofData> df;
    std::vector<uint64_t> bf;
};

static void svSeed(SvArena& B)
{
    B.ab.resize(SV_WARPS);
    B.lk.resize(size_t(SV_WARPS) * SV_MAX_LINKS);
    B.df.resize(size_t(SV_WARPS) * SV_MAX_DOFS);
    B.bf.resize(size_t(SV_WARPS) * SV_MAX_LINKS * SV_WORDSIZE * 32);
    std::memset(B.ab.data(), 0xAB, B.ab.size() * sizeof(B.ab[0]));
    std::memset(B.lk.data(), 0xAB, B.lk.size() * sizeof(B.lk[0]));
    std::memset(B.df.data(), 0xAB, B.df.size() * sizeof(B.df[0]));
    std::memset(B.bf.data(), 0xAB, B.bf.size() * 8);

    for (uint32_t a = 0; a < SV_NB; ++a)
    {
        const uint32_t w = a / 32, lane = a % 32;
        PxgArticulationBlockData& AB = B.ab[w];
        PxgArticulationBlockLinkData* L = &B.lk[size_t(w) * SV_MAX_LINKS];
        PxgArticulationBlockDofData* D = &B.df[size_t(w) * SV_MAX_DOFS];
        uint64_t* BF = &B.bf[size_t(w) * SV_MAX_LINKS * SV_WORDSIZE * 32];

        const uint32_t numLinks = svNumLinks(a);
        AB.mNumLinks[lane] = numLinks;
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
            for (uint32_t p = l; p != 0; p = svParent(p))
                path |= 1ull << p;
            BF[(l * SV_WORDSIZE + 0) * 32 + lane] = path;

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
            lnk.mParents[lane] = svParent(l);
            lnk.mRw_x[lane] = 0.5f * fv(a, l, 0, 86);
            lnk.mRw_y[lane] = 0.5f * fv(a, l, 1, 86);
            lnk.mRw_z[lane] = 0.5f * fv(a, l, 2, 86);
            const uint32_t nbDofs = svDofs(a, l);
            lnk.mJointOffset[lane] = dofCursor;
            lnk.mDofs[lane] = PxU8(nbDofs);
            for (uint32_t d = 0; d < nbDofs; ++d)
            {
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
            }
            dofCursor += nbDofs;
        }
    }
}

// ---- cases ----

// getVelocity: linkID | wantSpeeds<<8 | fixBase<<9
static const uint32_t SV_VEL_CASES[] = {
    0u,                 // root, free base, no speeds
    1u | 0x100u | 0x200u,
    2u | 0x100u,
    3u | 0x200u,
    3u | 0x100u,
    2u | 0x200u,
};
const uint32_t SV_NUM_VEL_CASES = 6;

// applyImpulses: pairs {linkID0|hasJi0<<8, linkID1|hasJi1<<8}
static const uint32_t SV_IMP_CASES[][2] = {
    { 2u,          3u | 0x100u }, // disjoint pair, common 1
    { 1u | 0x100u, 3u | 0x100u }, // linkID0 == common
    { 3u | 0x100u, 3u | 0x100u }, // identical links, both accumulate
    { 0u | 0x100u, 2u | 0x100u }, // common == root: walk skipped, ji0 dropped
    { 2u | 0x100u, 1u },          // linkID1 == common
};
const uint32_t SV_NUM_IMP_CASES = 5;

struct SvParamsHost
{
    uint64_t ab, lk, df, bf, scratch;
    uint32_t nb, maxLinks, maxDofs, caseCount;
    uint64_t caseTable;
};

static size_t svRunGetVel(PxgCudaShim& shim)
{
    SvArena A;
    svSeed(A);

    // mirror
    std::vector<float> want(size_t(SV_NUM_VEL_CASES) * SV_NB * 9, -555.f);
    for (uint32_t c = 0; c < SV_NUM_VEL_CASES; ++c)
        for (uint32_t a = 0; a < SV_NB; ++a)
        {
            const uint32_t w = a / 32, lane = a % 32;
            const uint32_t spec = SV_VEL_CASES[c];
            const uint32_t linkID = spec & 0xFF;
            const bool wantSpeeds = (spec & 0x100u) != 0;
            const bool fixBase = (spec & 0x200u) != 0;
            PxReal speeds[3] = { -777.f, -777.f, -777.f };
            const UnAlignedSpatialVector v = refGetVelocity(
                A.ab[w], &A.lk[size_t(w) * SV_MAX_LINKS], &A.df[size_t(w) * SV_MAX_DOFS],
                &A.bf[size_t(w) * SV_MAX_LINKS * SV_WORDSIZE * 32],
                SV_WORDSIZE, linkID, fixBase, wantSpeeds ? speeds : NULL, lane);
            float* o = &want[(size_t(c) * SV_NB + a) * 9];
            o[0] = v.top.x; o[1] = v.top.y; o[2] = v.top.z;
            o[3] = v.bottom.x; o[4] = v.bottom.y; o[5] = v.bottom.z;
            o[6] = speeds[0]; o[7] = speeds[1]; o[8] = speeds[2];
        }

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
    PxgDevicePtr dScr = up(NULL, size_t(SV_NUM_VEL_CASES) * SV_NB * 9 * 4);
    PxgDevicePtr dCases = up(SV_VEL_CASES, sizeof(SV_VEL_CASES));
    if (!dAb || !dLk || !dDf || !dBf || !dScr || !dCases)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
        return 0;
    }
    {
        std::vector<float> poison(size_t(SV_NUM_VEL_CASES) * SV_NB * 9, -555.f);
        shim.memcpyHtoD(dScr, poison.data(), poison.size() * 4);
    }

    SvParamsHost prm = { uint64_t(dAb), uint64_t(dLk), uint64_t(dDf), uint64_t(dBf),
                         uint64_t(dScr), SV_NB, SV_MAX_LINKS, SV_MAX_DOFS,
                         SV_NUM_VEL_CASES, uint64_t(dCases) };
    PXG_CHECK_MSG(shim.launchKernel(0, SV_WARPS, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch svGetVelTest");
    shim.streamSynchronize();

    size_t bad = 0;
    // outputs
    {
        std::vector<float> got(want.size());
        shim.memcpyDtoH(got.data(), dScr, got.size() * 4);
        size_t mm = 0;
        for (size_t i = 0; i < got.size(); ++i)
        {
            uint32_t g, x;
            std::memcpy(&g, &got[i], 4);
            std::memcpy(&x, &want[i], 4);
            if (g != x)
            {
                if (mm < 4)
                    std::printf("MM vel[%zu] case %zu art %zu slot %zu: got %.9g want %.9g\n",
                                i, i / (size_t(SV_NB) * 9), (i / 9) % SV_NB, i % 9,
                                double(got[i]), double(want[i]));
                ++mm;
            }
        }
        if (mm) { std::printf("MM getVelocity outputs: %zu words\n", mm); bad += mm; }
    }
    // inputs must be untouched (pure read)
    auto checkUntouched = [&](PxgDevicePtr dp, const void* host, size_t n, const char* what) {
        std::vector<uint8_t> g(n);
        shim.memcpyDtoH(g.data(), dp, n);
        if (std::memcmp(g.data(), host, n) != 0)
        {
            ++bad;
            std::printf("MM %s touched by getVelocity\n", what);
        }
    };
    checkUntouched(dAb, A.ab.data(), A.ab.size() * sizeof(A.ab[0]), "artiBlocks");
    checkUntouched(dLk, A.lk.data(), A.lk.size() * sizeof(A.lk[0]), "linkBlocks");
    checkUntouched(dDf, A.df.data(), A.df.size() * sizeof(A.df[0]), "dofBlocks");
    checkUntouched(dBf, A.bf.data(), A.bf.size() * 8, "bitFields");

    for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
    return bad;
}

static size_t svRunApplyImp(PxgCudaShim& shim)
{
    SvArena A;
    svSeed(A);

    // the per-case impulse table (also the shader's input)
    std::vector<float> impTable(size_t(SV_NUM_IMP_CASES) * 18);
    for (uint32_t c = 0; c < SV_NUM_IMP_CASES; ++c)
        for (uint32_t k = 0; k < 18; ++k)
            impTable[c * 18 + k] = 0.8f * fv(900 + c, k, 0, 98);

    // mirror: same case order, same per-lane scaling as the shader entry
    SvArena W = A;
    for (uint32_t a = 0; a < SV_NB; ++a)
    {
        const uint32_t w = a / 32, lane = a % 32;
        for (uint32_t c = 0; c < SV_NUM_IMP_CASES; ++c)
        {
            const uint32_t spec0 = SV_IMP_CASES[c][0], spec1 = SV_IMP_CASES[c][1];
            const float* f = &impTable[c * 18];
            const PxVec3 linear0 = PxVec3(f[0], f[1], f[2]) * (1.0f + float(a) * 0.03125f);
            const PxVec3 angular0(f[3], f[4], f[5]);
            const PxReal ji0[3] = { f[6], f[7], f[8] };
            const PxVec3 linear1(f[9], f[10], f[11]);
            const PxVec3 angular1 = PxVec3(f[12], f[13], f[14])
                                  * (1.0f - float(a) * 0.015625f);
            const PxReal ji1[3] = { f[15], f[16], f[17] };
            refApplyImpulses(W.ab[w], &W.lk[size_t(w) * SV_MAX_LINKS],
                             &W.df[size_t(w) * SV_MAX_DOFS],
                             &W.bf[size_t(w) * SV_MAX_LINKS * SV_WORDSIZE * 32],
                             SV_WORDSIZE, spec0 & 0xFF, linear0, angular0,
                             (spec0 & 0x100u) ? ji0 : NULL,
                             spec1 & 0xFF, linear1, angular1,
                             (spec1 & 0x100u) ? ji1 : NULL, lane);
        }
    }

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
    PxgDevicePtr dImp = up(impTable.data(), impTable.size() * 4);
    PxgDevicePtr dCases = up(SV_IMP_CASES, sizeof(SV_IMP_CASES));
    if (!dAb || !dLk || !dDf || !dBf || !dImp || !dCases)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
        return 0;
    }

    SvParamsHost prm = { uint64_t(dAb), uint64_t(dLk), uint64_t(dDf), uint64_t(dBf),
                         uint64_t(dImp), SV_NB, SV_MAX_LINKS, SV_MAX_DOFS,
                         SV_NUM_IMP_CASES, uint64_t(dCases) };
    PXG_CHECK_MSG(shim.launchKernel(1, SV_WARPS, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch svApplyImpTest");
    shim.streamSynchronize();

    size_t bad = 0;
    auto cmp = [&](PxgDevicePtr dp, const void* host, size_t n, const char* what) {
        std::vector<uint8_t> g(n);
        shim.memcpyDtoH(g.data(), dp, n);
        if (std::memcmp(g.data(), host, n) != 0)
        {
            size_t mm = 0, first = size_t(-1);
            const uint8_t* h = reinterpret_cast<const uint8_t*>(host);
            for (size_t i = 0; i < n; ++i)
                if (g[i] != h[i]) { ++mm; if (first == size_t(-1)) first = i; }
            std::printf("MM %s: %zu/%zu bytes differ (first at %zu)\n", what, mm, n, first);
            bad += mm;
        }
    };
    cmp(dAb, W.ab.data(), W.ab.size() * sizeof(W.ab[0]), "artiBlocks");
    cmp(dLk, W.lk.data(), W.lk.size() * sizeof(W.lk[0]), "linkBlocks");
    cmp(dDf, W.df.data(), W.df.size() * sizeof(W.df[0]), "dofBlocks");
    cmp(dBf, W.bf.data(), W.bf.size() * 8, "bitFields");

    for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
    return bad;
}

} // namespace

PXG_TEST(artiSolveVelImp)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiSolveTendons.spv", "svGetVelTest", 64u, 0 },
        { "npArtiSolveTendons.spv", "svApplyImpTest", 64u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 2, &svLoader);

    size_t bad = 0;
    bad += svRunGetVel(shim);
    bad += svRunApplyImp(shim);
    PXG_CHECK_MSG(bad == 0, "%zu mismatches", bad);
}
