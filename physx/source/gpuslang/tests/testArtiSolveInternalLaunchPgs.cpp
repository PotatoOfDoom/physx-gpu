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

// testArtiSolveInternalLaunchPgs.cpp -- P8.123d: the assembled kernel
// artiSolveInternalConstraints1T (manifest 323) against a FULL host mirror
// over device-resident descriptors.
//
// Mirror pieces, every one already swept on its own: pxgref::aiMirror (the
// scale-0 internal-count branch), pxgtest::refSolveStaticLanePgs for the
// root static solve, and pxgtest::refSolveInternalDFSLanePgs (P8.123c).
//
// Wrapper surface under test -- the geometry is the TGS launch fixture's;
// PGS-specific facts:
//  * **erp = biasCoefficient UNCLAMPED** (the TGS extern clamps to 0.7;
//    round 1 passes bias 0.9 and the mirror uses 0.9 verbatim);
//  * **doFriction (flags byte 0) is LIVE** through the static solves;
//  * **isExternalForceEveryStep (flags byte 2) is DEAD** -- rounds 2/3 set
//    the byte while the mirror ignores it (deadness proof);
//  * the shared desc is the PGS variant: rowsMod at 120 additionally
//    consumed by the 1D static dispatch (with the OFFSET response base);
//  * mDeltaQ stays POISONED (the PGS static solve never reads it);
//  * no minPen anywhere on the PGS path.

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
#include "PxgBodySim.h"
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

static std::vector<uint8_t> ilpLoader(const char* file) { return loadShaderBlob(file); }

const uint32_t ILP_NB = 68, ILP_WARPS = 3;
const uint32_t ILP_MAX_LINKS = 6, ILP_MAX_DOFS = 11;
const uint32_t ILP_NB_SLABS = 5;
const uint32_t ILP_ARTI_OFFSET = 7;
const uint32_t ILP_NB_NODES = 80;          // bodySim pool entries
const uint32_t ILP_NB_BATCHES = 34;        // batch 0 + per-warp 11-blocks
const uint32_t ILP_NB_HDRS = 34;
const uint32_t ILP_NB_CPTS = 150, ILP_NB_FPTS = 220, ILP_NB_ROWS = 220, ILP_NB_RESP = 320;
const uint32_t ILP_REC_SIZE = 544;         // sizeof(PxgArticulation), = PXG_ARTI_STRIDE
const uint32_t ILP_BODYSIM_SIZE = 224;

static_assert(sizeof(PxgArticulation) == ILP_REC_SIZE, "record stride pin");
static_assert(offsetof(PxgArticulation, jointForce) == 136, "jointForce pin");
static_assert(offsetof(PxgArticulation, internalResidualAccumulator) == 512, "internal residual pin");
static_assert(offsetof(PxgArticulation, contactResidualAccumulator) == 524, "contact residual pin");
static_assert(sizeof(PxgBodySim) == ILP_BODYSIM_SIZE, "bodySim stride pin");

// offsets into the 360-byte PxgArticulationCoreDesc, each pinned by
// static_assert in testDmaArtiResidual.cpp against the engine header
const uint32_t D_BODY_SIM = 0, D_ISLAND_NODES = 16, D_ARTICULATIONS = 32;
const uint32_t D_ARTI_OFFSET = 40, D_NB_ARTICULATIONS = 44, D_DT = 48;
const uint32_t D_IMPULSES = 72, D_NB_SLABS = 80, D_SLAB_HAS_CHANGES = 88;
const uint32_t D_ARTI_BLOCKS = 128, D_LINK_BLOCKS = 136, D_TRAVERSAL_STACK = 144;
const uint32_t D_DOF_BLOCKS = 184, D_MAX_LINKS = 248, D_MAX_DOFS = 252;
// PxgSolverSharedDesc<IterativeSolveDataTGS>::iterativeData.*
const uint32_t SH_BATCH = 96, SH_JHDR = 104, SH_JROW = 112, SH_JMOD = 120, SH_CHDR = 128;
const uint32_t SH_FHDR = 136, SH_CPTS = 144, SH_FPTS = 152, SH_RESP = 216;
const uint32_t SH_SIZE = 224, DESC_SIZE = 360;

const float ILP_DT = 0.01f, ILP_SCENE_DT = 0.033f, ILP_INV_DT = 100.0f, ILP_ELAPSED = 0.017f;

static float fv(uint32_t a, uint32_t l, uint32_t d, uint32_t salt)
{
    uint32_t h = a * 2654435761u + l * 40503u + d * 9973u + salt * 733u + 19u;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    return -1.0f + 2.0f * float(h & 0xFFFFFFu) / float(0x1000000);
}

// trees: the P8.100c family, keyed by the articulation index
static uint32_t ilpNumLinks(uint32_t a) { return 4u + (a % 3u); }
static uint32_t ilpParent(uint32_t l)
{
    static const uint32_t p[6] = { 0, 0, 1, 1, 2, 3 };
    return p[l];
}
static void ilpChildren(uint32_t l, uint32_t numLinks, uint32_t& off, uint32_t& num)
{
    off = 0; num = 0;
    for (uint32_t c = 1; c < numLinks; ++c)
        if (ilpParent(c) == l) { if (num == 0) off = c; ++num; }
}
static uint32_t ilpDofs(uint32_t l, uint32_t a) { return 1u + ((a + l) % 2u); }

// the two permutations of the record resolve
static uint32_t ilpRemap(uint32_t gti) { return (gti * 7u + 3u) % ILP_NB; }
static uint32_t ilpNodeId(uint32_t gti) { return (gti * 3u + 11u) % ILP_NB_NODES; }

// per-warp DISJOINT static windows: warp w owns batches [1+11w .. 1+11w+9]
static uint32_t ilpJ0(uint32_t w) { return 1u + w * 11u; }
static uint32_t ilpC0(uint32_t w) { return 1u + w * 11u + 5u; }
static uint32_t ilpJStart(uint32_t a, uint32_t l) { return ilpJ0(a / 32u) + ((a + l) % 3u); }
static uint32_t ilpJCount(uint32_t a, uint32_t l) { return ((a + l) % 4u == 1u) ? 1u : 0u; }
static uint32_t ilpCStart(uint32_t a, uint32_t l) { return ilpC0(a / 32u) + ((a + 2u * l) % 4u); }
static uint32_t ilpCCount(uint32_t a, uint32_t l) { return ((a + l) % 5u == 2u) ? 1u : 0u; }

static bool ilpIs1D(uint32_t b)
{
    if (b == 0u || ((b - 1u) % 11u) == 10u) return false;   // poison rows
    return ((b - 1u) % 11u) < 5u;
}
static bool ilpIsContact(uint32_t b)
{
    if (b == 0u || ((b - 1u) % 11u) == 10u) return false;
    const uint32_t r = (b - 1u) % 11u;
    return r >= 5u && r < 10u;
}
static bool ilpAIsArti(uint32_t b, uint32_t k) { return ((b + k) % 3u) != 0u; }

struct IlpArena
{
    std::vector<uint8_t> desc, sharedDesc;
    std::vector<uint8_t> rec;                 // ILP_NB * 544
    std::vector<uint8_t> bodySims;            // ILP_NB_NODES * 224
    std::vector<PxNodeIndex> nodes;           // ILP_ARTI_OFFSET + ILP_NB + 3
    std::vector<float> jf;                    // ILP_NB * ILP_MAX_DOFS
    std::vector<PxgArticulationBlockData> ab;
    std::vector<PxgArticulationBlockLinkData> lk;
    std::vector<PxgArticulationBlockDofData> df;
    std::vector<PxgArticulationTraversalStackData> stack;
    std::vector<uint32_t> slab;
    std::vector<float> imp;
    std::vector<PxgBlockConstraintBatch> bt;
    std::vector<PxgBlockSolverContactHeader> ch;
    std::vector<PxgBlockSolverFrictionHeader> fh;
    std::vector<PxgBlockSolverContactPoint> cp;
    std::vector<PxgBlockSolverContactFriction> fp;
    std::vector<PxgBlockSolverConstraint1DHeader> jh;
    std::vector<PxgBlockSolverConstraint1DCon> jr;
    std::vector<PxgBlockSolverConstraint1DMod> jm;
    std::vector<PxgArticulationBlockResponse> rs;
};

static PxgArticulation* ilpRec(IlpArena& B, uint32_t r)
{
    return reinterpret_cast<PxgArticulation*>(&B.rec[size_t(r) * ILP_REC_SIZE]);
}

static void ilpSeed(IlpArena& B, uint32_t salt)
{
    B.desc.assign(DESC_SIZE, 0xAB);
    B.sharedDesc.assign(SH_SIZE, 0xAB);
    B.rec.assign(size_t(ILP_NB) * ILP_REC_SIZE, 0xAB);
    B.bodySims.assign(size_t(ILP_NB_NODES) * ILP_BODYSIM_SIZE, 0xAB);
    B.nodes.resize(ILP_ARTI_OFFSET + ILP_NB + 3);
    B.jf.resize(size_t(ILP_NB) * ILP_MAX_DOFS);
    B.ab.resize(ILP_WARPS);
    B.lk.resize(size_t(ILP_WARPS) * ILP_MAX_LINKS);
    B.df.resize(size_t(ILP_WARPS) * ILP_MAX_DOFS);
    B.stack.resize(size_t(ILP_WARPS) * ILP_MAX_LINKS);
    B.slab.resize(size_t(ILP_NB_SLABS) * ILP_NB * 2);
    B.imp.resize(size_t(ILP_NB_SLABS) * ILP_MAX_LINKS * ILP_NB * 6);
    B.bt.resize(ILP_NB_BATCHES);
    B.ch.resize(ILP_NB_HDRS);
    B.fh.resize(ILP_NB_HDRS);
    B.cp.resize(ILP_NB_CPTS);
    B.fp.resize(ILP_NB_FPTS);
    B.jh.resize(ILP_NB_HDRS);
    B.jr.resize(ILP_NB_ROWS);
    B.jm.resize(ILP_NB_ROWS);
    B.rs.resize(ILP_NB_RESP);
    std::memset(B.nodes.data(), 0xAB, B.nodes.size() * sizeof(B.nodes[0]));
    std::memset(B.ab.data(), 0xAB, B.ab.size() * sizeof(B.ab[0]));
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

    auto w32 = [&](std::vector<uint8_t>& v, uint32_t off, uint32_t x) {
        std::memcpy(&v[off], &x, 4);
    };
    w32(B.desc, D_NB_ARTICULATIONS, ILP_NB);
    w32(B.desc, D_NB_SLABS, ILP_NB_SLABS);
    w32(B.desc, D_MAX_LINKS, ILP_MAX_LINKS);
    w32(B.desc, D_MAX_DOFS, ILP_MAX_DOFS);
    w32(B.desc, D_ARTI_OFFSET, ILP_ARTI_OFFSET);
    float sceneDt = ILP_SCENE_DT;
    std::memcpy(&B.desc[D_DT], &sceneDt, 4);

    // the record resolve chain
    for (uint32_t gti = 0; gti < ILP_NB; ++gti)
    {
        const uint32_t nodeId = ilpNodeId(gti);
        B.nodes[ILP_ARTI_OFFSET + gti] = PxNodeIndex(nodeId);
        uint32_t remap = ilpRemap(gti);
        std::memcpy(&B.bodySims[size_t(nodeId) * ILP_BODYSIM_SIZE + 176], &remap, 4);
    }

    for (size_t i = 0; i < B.jf.size(); ++i)
        B.jf[i] = 0.5f * fv(uint32_t(i), 210, 0, salt);

    // slab dirty pairs: mostly NONE, a few live per articulation
    for (size_t i = 0; i < B.slab.size(); ++i) B.slab[i] = pxgref::AI_NONE;
    for (uint32_t a = 0; a < ILP_NB; ++a)
        for (uint32_t s = 0; s < ILP_NB_SLABS; ++s)
        {
            if ((a + s) % 4u == 3u)
                B.slab[(size_t(s) * ILP_NB + a) * 2 + 0] = 1u + ((a + s) % (ILP_MAX_LINKS - 1u));
            if ((a + s) % 5u == 2u)
                B.slab[(size_t(s) * ILP_NB + a) * 2 + 1] = 1u + ((a * 3u + s) % (ILP_MAX_LINKS - 1u));
        }
    for (size_t i = 0; i < B.imp.size(); ++i)
        B.imp[i] = 0.4f * fv(uint32_t(i & 0xFFFFu), uint32_t(i >> 16), 7, salt);

    for (uint32_t a = 0; a < ILP_NB; ++a)
    {
        const uint32_t w = a / 32, lane = a % 32;
        PxgArticulationBlockData& AB = B.ab[w];
        PxgArticulationBlockLinkData* L = &B.lk[size_t(w) * ILP_MAX_LINKS];
        PxgArticulationBlockDofData* D = &B.df[size_t(w) * ILP_MAX_DOFS];
        const uint32_t numLinks = ilpNumLinks(a);

        AB.mStateDirty[lane] = PxU8(a % 4u);
        AB.mNumLinks[lane] = numLinks;
        AB.mFlags[lane] = (a % 3u == 0u) ? PxU32(PxArticulationFlag::eFIX_BASE) : 0u;
        AB.mLinkWithDeferredImpulse[lane] = (a % 5u == 0u) ? 0u : 1u + (a % (numLinks - 1u));

        storeSpatialVector(AB.mRootDeferredZ,
                           UnAlignedSpatialVector(
                               PxVec3(fv(a, 90, 0, salt), fv(a, 90, 1, salt), fv(a, 90, 2, salt)),
                               PxVec3(fv(a, 90, 3, salt), fv(a, 90, 4, salt), fv(a, 90, 5, salt))),
                           lane);
        Dy::SpatialMatrix M;
        for (uint32_t c = 0; c < 3; ++c)
            for (uint32_t r = 0; r < 3; ++r)
            {
                M.topLeft[c][r] = 0.35f * fv(a, 91, c * 3 + r, salt);
                M.topRight[c][r] = 0.35f * fv(a, 92, c * 3 + r, salt);
                M.bottomLeft[c][r] = 0.35f * fv(a, 93, c * 3 + r, salt);
            }
        storeSpatialMatrix(AB.mInvSpatialArticulatedInertia, lane, M);

        uint32_t cursor = 0;
        for (uint32_t l = 0; l < ILP_MAX_LINKS; ++l)
        {
            PxgArticulationBlockLinkData& LK = L[l];
            const bool live = l < numLinks;
            LK.mParents[lane] = live ? ilpParent(l) : 0;
            uint32_t off, num;
            ilpChildren(l, numLinks, off, num);
            LK.mChildrenOffset[lane] = live ? off : 0;
            LK.mNumChildren[lane] = live ? num : 0;
            if (l == 0)
            {
                LK.mRw_x[lane] = 7.5f; LK.mRw_y[lane] = -3.25f; LK.mRw_z[lane] = 9.0f;
                LK.mJointOffset[lane] = ILP_MAX_DOFS - 2;
                LK.mDofs[lane] = 2;
            }
            else
            {
                LK.mRw_x[lane] = 0.4f * fv(a, l, 10, salt);
                LK.mRw_y[lane] = 0.4f * fv(a, l, 11, salt);
                LK.mRw_z[lane] = 0.4f * fv(a, l, 12, salt);
                LK.mJointOffset[lane] = cursor;
                LK.mDofs[lane] = PxU8(ilpDofs(l, a));
                cursor += ilpDofs(l, a);
            }
            storeSpatialVector(LK.mMotionVelocity,
                               UnAlignedSpatialVector(
                                   PxVec3(fv(a, l, 13, salt), fv(a, l, 14, salt), fv(a, l, 15, salt)),
                                   PxVec3(fv(a, l, 16, salt), fv(a, l, 17, salt), fv(a, l, 18, salt))),
                               lane);
            storeSpatialVector(LK.mDeltaMotion,
                               UnAlignedSpatialVector(
                                   PxVec3(0.1f * fv(a, l, 19, salt), 0.1f * fv(a, l, 20, salt), 0.1f * fv(a, l, 21, salt)),
                                   PxVec3(0.1f * fv(a, l, 22, salt), 0.1f * fv(a, l, 23, salt), 0.1f * fv(a, l, 24, salt))),
                               lane);
            // mDeltaQ stays POISONED: the PGS static solve never reads it
            storeSpatialVector(LK.mScratchImpulse,
                               UnAlignedSpatialVector(
                                   PxVec3(0.2f * fv(a, l, 45, salt), 0.2f * fv(a, l, 46, salt), 0.2f * fv(a, l, 47, salt)),
                                   PxVec3(0.2f * fv(a, l, 48, salt), 0.2f * fv(a, l, 49, salt), 0.2f * fv(a, l, 50, salt))),
                               lane);
            storeSpatialVector(LK.mSolverSpatialImpulse,
                               UnAlignedSpatialVector(
                                   PxVec3(0.2f * fv(a, l, 52, salt), 0.2f * fv(a, l, 53, salt), 0.2f * fv(a, l, 54, salt)),
                                   PxVec3(0.2f * fv(a, l, 55, salt), 0.2f * fv(a, l, 56, salt), 0.2f * fv(a, l, 57, salt))),
                               lane);
            storeSpatialVector(LK.mSolverSpatialInternalConstraintImpulse,
                               UnAlignedSpatialVector(
                                   PxVec3(0.15f * fv(a, l, 58, salt), 0.15f * fv(a, l, 59, salt), 0.15f * fv(a, l, 60, salt)),
                                   PxVec3(0.15f * fv(a, l, 61, salt), 0.15f * fv(a, l, 62, salt), 0.15f * fv(a, l, 63, salt))),
                               lane);
            PxSpatialMatrix R;
            for (uint32_t i = 0; i < 6; ++i)
                for (uint32_t j = 0; j < 6; ++j)
                    R.column[i][j] = 0.15f * fv(a, l * 40 + i * 6 + j, 64, salt);
            storeSpatialMatrix(LK.mSpatialResponseMatrix, lane, R);

            LK.mStaticJointStartIndex[lane] = ilpJStart(a, l);
            LK.mNbStaticJoints[lane] = ilpJCount(a, l);
            LK.mStaticContactStartIndex[lane] = ilpCStart(a, l);
            LK.mNbStaticContacts[lane] = ilpCCount(a, l);
        }
        PX_ASSERT(cursor <= ILP_MAX_DOFS - 2);

        for (uint32_t d = 0; d < ILP_MAX_DOFS; ++d)
        {
            PxgArticulationBlockDofData& DD = D[d];
            PxgArticulationInternalConstraintData& ic = DD.mConstraintData;
            const bool rootSlice = d >= ILP_MAX_DOFS - 2;
            const float big = rootSlice ? 11.0f : 1.0f;

            DD.mMotion[lane] = PxU8((a + d) % 3u);
            DD.mDeferredQstZ[lane] = 0.3f * fv(a, d, 51, salt);
            storeSpatialVector(DD.mWorldMotionMatrix,
                               UnAlignedSpatialVector(
                                   PxVec3(big * fv(a, d, 30, salt), big * fv(a, d, 31, salt), big * fv(a, d, 32, salt)),
                                   PxVec3(big * fv(a, d, 33, salt), big * fv(a, d, 34, salt), big * fv(a, d, 35, salt))),
                               lane);
            storeSpatialVector(DD.mIsInvDW,
                               UnAlignedSpatialVector(
                                   PxVec3(big * fv(a, d, 36, salt), big * fv(a, d, 37, salt), big * fv(a, d, 38, salt)),
                                   PxVec3(big * fv(a, d, 39, salt), big * fv(a, d, 40, salt), big * fv(a, d, 41, salt))),
                               lane);
            storeSpatialVector(DD.mIsW,
                               UnAlignedSpatialVector(
                                   PxVec3(big * fv(a, d, 42, salt), big * fv(a, d, 43, salt), big * fv(a, d, 44, salt)),
                                   PxVec3(big * fv(a, d, 45, salt), big * fv(a, d, 46, salt), big * fv(a, d, 47, salt))),
                               lane);
            DD.mInvStIsT_x[lane] = 0.5f * fv(a, d, 48, salt);
            DD.mInvStIsT_y[lane] = 0.5f * fv(a, d, 49, salt);
            DD.mInvStIsT_z[lane] = 0.5f * fv(a, d, 50, salt);

            storeSpatialVector(ic.mRow0,
                               UnAlignedSpatialVector(
                                   PxVec3(fv(a, d, 110, salt), fv(a, d, 111, salt), fv(a, d, 112, salt)),
                                   PxVec3(fv(a, d, 113, salt), fv(a, d, 114, salt), fv(a, d, 115, salt))),
                               lane);
            storeSpatialVector(ic.mRow1,
                               UnAlignedSpatialVector(
                                   PxVec3(fv(a, d, 116, salt), fv(a, d, 117, salt), fv(a, d, 118, salt)),
                                   PxVec3(fv(a, d, 119, salt), fv(a, d, 120, salt), fv(a, d, 121, salt))),
                               lane);
            storeSpatialVector(ic.mDeltaVA,
                               UnAlignedSpatialVector(
                                   PxVec3(0.3f * fv(a, d, 122, salt), 0.3f * fv(a, d, 123, salt), 0.3f * fv(a, d, 124, salt)),
                                   PxVec3(0.3f * fv(a, d, 125, salt), 0.3f * fv(a, d, 126, salt), 0.3f * fv(a, d, 127, salt))),
                               lane);
            storeSpatialVector(ic.mDeltaVB,
                               UnAlignedSpatialVector(
                                   PxVec3(0.3f * fv(a, d, 128, salt), 0.3f * fv(a, d, 129, salt), 0.3f * fv(a, d, 130, salt)),
                                   PxVec3(0.3f * fv(a, d, 131, salt), 0.3f * fv(a, d, 132, salt), 0.3f * fv(a, d, 133, salt))),
                               lane);
            const float response = 0.6f + 0.3f * std::fabs(fv(a, d, 134, salt));
            ic.mResponse[lane] = response;
            ic.mRecipResponse[lane] = 1.0f / response;
            ic.mMaxJointVelocity[lane] = 0.4f + 0.2f * std::fabs(fv(a, d, 135, salt));
            ic.mConstraintMaxImpulse[lane] = 0.6f + 0.3f * std::fabs(fv(a, d, 136, salt));
            ic.mDriveImpulse[lane] = 0.3f * fv(a, d, 137, salt);
            ic.mMaxFrictionForce[lane] = 0.4f + 0.2f * std::fabs(fv(a, d, 138, salt));
            ic.mAccumulatedFrictionImpulse[lane] = 0.2f * fv(a, d, 139, salt);
            ic.mMaxEffort[lane] = ((a + d) % 2u) ? (30.0f + 10.0f * std::fabs(fv(a, d, 140, salt))) : 0.0f;
            ic.mMaxActuatorVelocity[lane] = 2.0f + std::fabs(fv(a, d, 141, salt));
            ic.mSpeedEffortGradient[lane] = ((a + d) % 5u == 2u) ? 0.0f : 0.02f * std::fabs(fv(a, d, 142, salt));
            ic.mVelocityDependentResistance[lane] = ((a + d) % 3u == 1u) ? (4.0f + std::fabs(fv(a, d, 143, salt))) : 0.0f;
            const bool newModel = (a % 3u) != 0u;
            ic.mStaticFrictionEffort[lane] = newModel ? (8.0f + 4.0f * std::fabs(fv(a, d, 144, salt))) : 0.0f;
            ic.mDynamicFrictionEffort[lane] = newModel ? (5.0f + 2.0f * std::fabs(fv(a, d, 145, salt))) : 0.0f;
            ic.mViscousFrictionCoefficient[lane] = (newModel && (a % 4u == 1u)) ? (0.5f + std::fabs(fv(a, d, 146, salt))) : 0.0f;
            const uint32_t lm = (a + d) % 4u;
            float lo, hi;
            if (lm == 0) { lo = -0.05f - 0.02f * std::fabs(fv(a, d, 147, salt)); hi = 0.4f; }
            else if (lm == 1) { lo = 0.004f; hi = 0.5f; }
            else if (lm == 2) { lo = 0.4f; hi = -0.03f - 0.02f * std::fabs(fv(a, d, 148, salt)); }
            else { lo = 0.5f; hi = 0.5f; }
            ic.mLimitError_LowX_highY[lane] = make_float2(lo, hi);
            ic.mLowImpulse[lane] = 0.1f * std::fabs(fv(a, d, 149, salt));
            ic.mHighImpulse[lane] = -0.1f * std::fabs(fv(a, d, 150, salt));
            ic.mTargetVelPlusInitialBiasX_DriveBiasCoefficientY_VelMultiplierZ_ImpulseMultiplierW[lane] =
                make_float4(0.3f * fv(a, d, 151, salt), 0.4f * fv(a, d, 152, salt),
                            -0.5f * std::fabs(fv(a, d, 153, salt)), 0.9f + 0.1f * fv(a, d, 154, salt));
            ic.mTargetPosBias[lane] = 0.2f * fv(a, d, 155, salt);
        }
    }

    for (uint32_t b = 0; b < ILP_NB_BATCHES; ++b)
    {
        if (!ilpIs1D(b) && !ilpIsContact(b)) continue;
        PxgBlockConstraintBatch& BT = B.bt[b];
        BT.mDescStride = PxU16(31 + b);
        BT.constraintType = PxU16(ilpIsContact(b) ? PxgSolverConstraintDesc::eARTICULATION_CONTACT
                                                 : PxgSolverConstraintDesc::eARTICULATION_CONSTRAINT_1D);
        BT.mask = 0xFFFFFFFFu;
        BT.mConstraintBatchIndex = b;
        BT.startConstraintIndex = ilpIsContact(b) ? (b * 4 + 3) : (b * 6 + 2);
        BT.startFrictionIndex = b * 6 + 4;
        BT.mArticulationResponseIndex = b * 9 + 2;
        for (uint32_t k = 0; k < 32; ++k)
        {
            if (ilpAIsArti(b, k))
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
        if (ilpIsContact(b))
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

    for (uint32_t r = 0; r < ILP_NB_RESP; ++r)
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

// ---- mirror ----

static void ilpMirror(IlpArena& W, bool velIter, bool residual, bool doFriction, bool doFDPL,
                     uint32_t doVelLimit, bool doStatic, float biasCoefficient,
                     uint32_t& hasImpulses, uint32_t& fixSkips, uint32_t& rootStatics,
                     uint32_t& rootJointsLive)
{
    const float erp = biasCoefficient;   // UNCLAMPED on PGS (:2820)

    for (uint32_t gti = 0; gti < ILP_NB; ++gti)
    {
        const uint32_t w = gti / 32, lane = gti % 32;
        PxgArticulationBlockData& ab = W.ab[w];
        PxgArticulationBlockLinkData* lk = &W.lk[size_t(w) * ILP_MAX_LINKS];
        PxgArticulationBlockDofData* df = &W.df[size_t(w) * ILP_MAX_DOFS];
        PxgArticulationTraversalStackData* stack = &W.stack[size_t(w) * ILP_MAX_LINKS];

        // the record resolve
        const uint32_t nodeIndex = W.nodes[ILP_ARTI_OFFSET + gti].index();
        uint32_t remap;
        std::memcpy(&remap, &W.bodySims[size_t(nodeIndex) * ILP_BODYSIM_SIZE + 176], 4);
        PxgArticulation* rec = ilpRec(W, remap);

        if (residual)
        {
            rec->internalResidualAccumulator.reset();
            rec->contactResidualAccumulator.reset();
        }

        if (ab.mStateDirty[lane] & 2u)   // eHAS_IMPULSES
        {
            ++hasImpulses;
            uint32_t alignedHits = 0, unalignedHits = 0;
            pxgref::aiMirror(W.slab.data(), W.imp.data(), &ab, lk, df, gti,
                             ILP_MAX_LINKS, ILP_NB, ILP_NB_SLABS, lane, alignedHits, unalignedHits);
        }
        ab.mStateDirty[lane] = 1u;   // eVEL_DIRTY

        const bool fixBase = (ab.mFlags[lane] & PxArticulationFlag::eFIX_BASE) != 0;
        const UnAlignedSpatialVector rootDeferredZ = loadSpatialVector(ab.mRootDeferredZ, lane);

        UnAlignedSpatialVector parentDeltaV(PxVec3(0.f), PxVec3(0.f));
        UnAlignedSpatialVector storeParentDeltaV(PxVec3(0.f), PxVec3(0.f));
        UnAlignedSpatialVector parentImp(PxVec3(0.f), PxVec3(0.f));
        RefError err, contactErr;

        if (!fixBase && doStatic)
        {
            ++rootStatics;
            const uint32_t constraintCounts0 = lk[0].mNbStaticJoints[lane];
            if (constraintCounts0) ++rootJointsLive;
            Dy::SpatialMatrix M;
            loadSpatialMatrix(ab.mInvSpatialArticulatedInertia, lane, M);
            const UnAlignedSpatialVector mv0 = loadSpatialVector(lk[0].mMotionVelocity, lane);
            parentDeltaV = M * -rootDeferredZ;
            UnAlignedSpatialVector rootVel = mv0 + parentDeltaV;
            storeParentDeltaV = parentDeltaV;
            refSolveStaticLanePgs(lk[0], lane, W.bt.data(), W.ch.data(), W.fh.data(), W.cp.data(),
                                  W.fp.data(), W.jh.data(), W.jr.data(), W.jm.data(), W.rs.data(),
                                  doFriction, residual,
                                  rootVel, parentImp, parentDeltaV, contactErr, constraintCounts0, NULL);
        }
        else if (fixBase && doStatic)
        {
            ++fixSkips;
        }

        const UnAlignedSpatialVector finalImp = refSolveInternalDFSLanePgs(
            lk, df, stack, &W.jf[size_t(remap) * ILP_MAX_DOFS],
            W.bt.data(), W.ch.data(), W.fh.data(), W.cp.data(), W.fp.data(),
            W.jh.data(), W.jr.data(), W.jm.data(), W.rs.data(),
            ILP_DT, ILP_SCENE_DT, ILP_INV_DT, ILP_ELAPSED, erp,
            velIter, doFriction, doFDPL, doVelLimit, doStatic, residual,
            storeParentDeltaV, parentImp, parentDeltaV,
            err, contactErr, lane);

        storeSpatialVector(ab.mRootDeferredZ, rootDeferredZ + finalImp, lane);

        if (residual)
        {
            Dy::ErrorAccumulator& ir = rec->internalResidualAccumulator;
            ir.mErrorSumOfSquares += err.sumOfSquares;
            ir.mCounter += err.counter;
            if (err.maxError > ir.mMaxError) ir.mMaxError = err.maxError;
            Dy::ErrorAccumulator& cr = rec->contactResidualAccumulator;
            cr.mErrorSumOfSquares += contactErr.sumOfSquares;
            cr.mCounter += contactErr.counter;
            if (contactErr.maxError > cr.mMaxError) cr.mMaxError = contactErr.maxError;
        }
    }
}

// ---- run ----

struct IlpParamsHost
{
    uint64_t scDesc;                  // @0
    float dt;                         // @8
    float invDt;                      // @12
    uint32_t velIterWord;             // @16
    float elapsedTime;                // @20
    float biasCoefficient;            // @24
    uint32_t pad0;                    // @28
    uint64_t staticContactUniqueIds;  // @32 dead
    uint64_t staticJointUniqueIds;    // @40 dead
    uint64_t sharedDesc;              // @48
    uint32_t solveFlagsWord;          // @56
    uint32_t doVelLimit;              // @60
    uint32_t doStaticWord;            // @64
    uint32_t pad1;                    // @68
};
static_assert(sizeof(IlpParamsHost) == 72, "params blob");

static size_t ilpRun(PxgCudaShim& shim, bool velIter, bool residual, bool extEvery,
                    bool doFDPL, uint32_t doVelLimit, bool doStatic, bool doFriction,
                    float biasCoefficient, uint32_t salt)
{
    IlpArena A;
    ilpSeed(A, salt);

    std::vector<PxgDevicePtr> bufs;
    auto up = [&](const void* p, size_t n) {
        PxgDevicePtr dp = 0;
        shim.memAlloc(&dp, n);
        if (dp && p) shim.memcpyHtoD(dp, p, n);
        bufs.push_back(dp);
        return dp;
    };
    // jf first: the record images need its address
    PxgDevicePtr dJf = up(NULL, A.jf.size() * 4);
    if (dJf)
    {
        // patch each record's jointForce with its device slice, in the arena
        // BEFORE the mirror copy so host and device record images agree
        for (uint32_t r = 0; r < ILP_NB; ++r)
        {
            uint64_t p = uint64_t(dJf) + uint64_t(r) * ILP_MAX_DOFS * 4u;
            std::memcpy(&A.rec[size_t(r) * ILP_REC_SIZE + 136], &p, 8);
        }
        shim.memcpyHtoD(dJf, A.jf.data(), A.jf.size() * 4);
    }

    IlpArena W = A;
    uint32_t hasImpulses = 0, fixSkips = 0, rootStatics = 0, rootJointsLive = 0;
    // extEvery reaches ONLY the kernel's flags word: dead on PGS, the mirror
    // never sees it (rounds 2/3 set the byte -- the deadness proof)
    ilpMirror(W, velIter, residual, doFriction, doFDPL, doVelLimit, doStatic, biasCoefficient,
             hasImpulses, fixSkips, rootStatics, rootJointsLive);

    std::printf("[ info ] launchPgs(vel=%d res=%d ext=%d fdpl=%d vlim=%u stat=%d): "
                "hasImpulses %u, fixSkips %u, rootStatics %u (joints live %u)\n",
                velIter ? 1 : 0, residual ? 1 : 0, extEvery ? 1 : 0, doFDPL ? 1 : 0,
                doVelLimit, doStatic ? 1 : 0, hasImpulses, fixSkips, rootStatics, rootJointsLive);
    PXG_CHECK_MSG(hasImpulses > 0, "impulse-average coverage hole");
    if (doStatic)
        PXG_CHECK_MSG(fixSkips > 0 && rootStatics > 0 && rootJointsLive > 0,
                      "root-static coverage hole");
    PXG_CHECK_MSG(std::memcmp(W.ab.data(), A.ab.data(), W.ab.size() * sizeof(W.ab[0])) != 0,
                  "vacuous");

    PxgDevicePtr dRec = up(A.rec.data(), A.rec.size());
    PxgDevicePtr dBs = up(A.bodySims.data(), A.bodySims.size());
    PxgDevicePtr dNd = up(A.nodes.data(), A.nodes.size() * sizeof(A.nodes[0]));
    PxgDevicePtr dAb = up(A.ab.data(), A.ab.size() * sizeof(A.ab[0]));
    PxgDevicePtr dLk = up(A.lk.data(), A.lk.size() * sizeof(A.lk[0]));
    PxgDevicePtr dDf = up(A.df.data(), A.df.size() * sizeof(A.df[0]));
    PxgDevicePtr dSt = up(A.stack.data(), A.stack.size() * sizeof(A.stack[0]));
    PxgDevicePtr dSlab = up(A.slab.data(), A.slab.size() * 4);
    PxgDevicePtr dImp = up(A.imp.data(), A.imp.size() * 4);
    PxgDevicePtr dBt = up(A.bt.data(), A.bt.size() * sizeof(A.bt[0]));
    PxgDevicePtr dCh = up(A.ch.data(), A.ch.size() * sizeof(A.ch[0]));
    PxgDevicePtr dFh = up(A.fh.data(), A.fh.size() * sizeof(A.fh[0]));
    PxgDevicePtr dCp = up(A.cp.data(), A.cp.size() * sizeof(A.cp[0]));
    PxgDevicePtr dFp = up(A.fp.data(), A.fp.size() * sizeof(A.fp[0]));
    PxgDevicePtr dJh = up(A.jh.data(), A.jh.size() * sizeof(A.jh[0]));
    PxgDevicePtr dJr = up(A.jr.data(), A.jr.size() * sizeof(A.jr[0]));
    PxgDevicePtr dJm = up(A.jm.data(), A.jm.size() * sizeof(A.jm[0]));
    PxgDevicePtr dRs = up(A.rs.data(), A.rs.size() * sizeof(A.rs[0]));

    auto w64 = [&](std::vector<uint8_t>& v, uint32_t off, uint64_t x) {
        std::memcpy(&v[off], &x, 8);
    };
    w64(A.desc, D_BODY_SIM, uint64_t(dBs));
    w64(A.desc, D_ISLAND_NODES, uint64_t(dNd));
    w64(A.desc, D_ARTICULATIONS, uint64_t(dRec));
    w64(A.desc, D_ARTI_BLOCKS, uint64_t(dAb));
    w64(A.desc, D_LINK_BLOCKS, uint64_t(dLk));
    w64(A.desc, D_DOF_BLOCKS, uint64_t(dDf));
    w64(A.desc, D_TRAVERSAL_STACK, uint64_t(dSt));
    w64(A.desc, D_SLAB_HAS_CHANGES, uint64_t(dSlab));
    w64(A.desc, D_IMPULSES, uint64_t(dImp));
    w64(A.sharedDesc, SH_BATCH, uint64_t(dBt));
    w64(A.sharedDesc, SH_JHDR, uint64_t(dJh));
    w64(A.sharedDesc, SH_JROW, uint64_t(dJr));
    w64(A.sharedDesc, SH_JMOD, uint64_t(dJm));
    w64(A.sharedDesc, SH_CHDR, uint64_t(dCh));
    w64(A.sharedDesc, SH_FHDR, uint64_t(dFh));
    w64(A.sharedDesc, SH_CPTS, uint64_t(dCp));
    w64(A.sharedDesc, SH_FPTS, uint64_t(dFp));
    w64(A.sharedDesc, SH_RESP, uint64_t(dRs));
    PxgDevicePtr dDesc = up(A.desc.data(), A.desc.size());
    PxgDevicePtr dSh = up(A.sharedDesc.data(), A.sharedDesc.size());

    bool ok = true;
    for (PxgDevicePtr p : bufs) ok = ok && (p != 0);
    if (!ok)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
        return 0;
    }

    uint32_t flagsWord = (doFriction ? 1u : 0u)
                       | (residual ? (1u << 8) : 0u)
                       | (extEvery ? (1u << 16) : 0u)
                       | (doFDPL ? (1u << 24) : 0u);
    IlpParamsHost prm = { uint64_t(dDesc), ILP_DT, ILP_INV_DT, velIter ? 1u : 0u,
                         ILP_ELAPSED, biasCoefficient, 0u,
                         0xDEAD0000ull, 0xDEAD4000ull,   // dead UID pointers: junk
                         uint64_t(dSh), flagsWord, doVelLimit, doStatic ? 1u : 0u, 0u };
    PXG_CHECK_MSG(shim.launchKernel(0, ILP_WARPS, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch internal-solve kernel pgs");
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
            std::printf("MM launchPgs/%s: %zu/%zu bytes differ (first at %zu)\n", name, mm, n, first);
            bad += mm;
        }
    };
    cmp(dRec, W.rec.data(), W.rec.size(), "records");
    cmp(dBs, W.bodySims.data(), W.bodySims.size(), "bodySims");
    cmp(dNd, W.nodes.data(), W.nodes.size() * sizeof(W.nodes[0]), "islandNodes");
    cmp(dJf, W.jf.data(), W.jf.size() * 4, "jointForces");
    cmp(dAb, W.ab.data(), W.ab.size() * sizeof(W.ab[0]), "artiBlocks");
    cmp(dLk, W.lk.data(), W.lk.size() * sizeof(W.lk[0]), "linkBlocks");
    cmp(dDf, W.df.data(), W.df.size() * sizeof(W.df[0]), "dofBlocks");
    cmp(dSt, W.stack.data(), W.stack.size() * sizeof(W.stack[0]), "stack");
    cmp(dSlab, W.slab.data(), W.slab.size() * 4, "slabHasChanges");
    cmp(dImp, W.imp.data(), W.imp.size() * 4, "impulses");
    cmp(dBt, W.bt.data(), W.bt.size() * sizeof(W.bt[0]), "batches");
    cmp(dCh, W.ch.data(), W.ch.size() * sizeof(W.ch[0]), "contactHeaders");
    cmp(dFh, W.fh.data(), W.fh.size() * sizeof(W.fh[0]), "frictionHeaders");
    cmp(dCp, W.cp.data(), W.cp.size() * sizeof(W.cp[0]), "contactPoints");
    cmp(dFp, W.fp.data(), W.fp.size() * sizeof(W.fp[0]), "frictions");
    cmp(dJh, W.jh.data(), W.jh.size() * sizeof(W.jh[0]), "jointHeaders");
    cmp(dJr, W.jr.data(), W.jr.size() * sizeof(W.jr[0]), "jointRows");
    cmp(dJm, W.jm.data(), W.jm.size() * sizeof(W.jm[0]), "jointMods");
    cmp(dRs, W.rs.data(), W.rs.size() * sizeof(W.rs[0]), "responses");

    for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
    return bad;
}

} // namespace

PXG_TEST(artiSolveInternalLaunchPgs)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiSolveInternal.spv", "artiSolveInternalConstraints1T", 72u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &ilpLoader);

    size_t bad = 0;
    // posIter, residuals on, BEFORE vel-limit, static on, bias 0.9 -- erp stays
    // 0.9 UNCLAMPED on PGS (a min(0.7,.) mutant dies here).  doFriction !=
    // residualReporting so a crossed byte read cannot pass as a coincidence.
    bad += ilpRun(shim, false, true, false, true, 1u, true, false, 0.9f, 31);
    // velIter, residuals OFF (accumulators must stay 0xAB), AFTER vel-limit,
    // bias 0.5; the extEvery byte SET while the mirror ignores it (dead)
    bad += ilpRun(shim, true, false, true, true, 2u, true, true, 0.5f, 37);
    // posIter, extEvery byte set again, no static, no vel-limit, no FDPL
    bad += ilpRun(shim, false, true, true, false, 0u, false, true, 0.7f, 41);
    // velIter=1 with doFriction=0 and every consumer LIVE (static, FDPL,
    // BEFORE vel-limit): the two flags differ while both are consumed, so a
    // swapped pair in the DFS dispatch cannot pass (rounds 1/2 have them
    // equal; round 3 has neither consumed)
    bad += ilpRun(shim, true, true, false, true, 1u, true, false, 0.8f, 43);
    PXG_CHECK_MSG(bad == 0, "%zu mismatches", bad);
}
