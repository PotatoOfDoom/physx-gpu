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

// testArtiJointPrepWrap.cpp -- P8.103: the ENGINE entry
// artiTGSJointConstraintBlockPrepareParallelLaunch over real desc images,
// against a FULL host mirror over the P8.102a body layer plus the ENGINE's
// own Dy:: scalar family (DyCpuGpu1dConstraint.h -- PX_CUDA_CALLABLE, so
// computeJointSpeedTGS / computeMaxBiasVelocityTGS /
// queryReduced1dConstraintSolverConstantsTGS /
// computeMinMaxImpulseOrForceAsImpulse / raiseInternalFlagsTGS are CALLED,
// not transcribed).  The row-loop control flow was Phase-1-swept as
// npArti1DPrep's a1dTGS battery; this fixture pins the engine-layout lift
// and the wrapper's index chain.
//
// Geometry mirrors testArtiConstraintPrepWrap: 4 live batches (2 arti-1d +
// 1 static + 1 self) behind a rigid-batch offset of 3, batchIndices
// permuted, descStride 26/13/32/7, three blocks launched (6 warps, 2 pin
// the guard).  Per-lane row counts 0..6 (zero-row distance-constraint
// lanes included), solveHint ties for the stable sort, engineered
// DEGENERATE rows (tiny Jacobians -> unitResponse <= 1e-12) so outCount
// diverges from mNumRows, break forces finite/PX_MAX_F32 both ways, and
// the writeBackOffset two-hop indirection with per-lane unique indices.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "foundation/PxVec3.h"
#include "foundation/PxQuat.h"
#include "foundation/PxTransform.h"
#include "CmSpatialVector.h"
#include "PxSpatialMatrix.h"
#include "PxgArticulation.h"
#include "PxgArticulationLink.h"
#include "PxgSolverBody.h"
#include "PxNodeIndex.h"
#include "PxRigidBody.h"
#include "PxConstraintDesc.h"
#include "PxConstraint.h"
#include "PxgConstraintPrep.h"
#include "PxgFrictionPatch.h"
#include "PxgSolverConstraintBlock1D.h"
#include "PxgConstraintBlock.h"
#include "PxgSolverCoreDesc.h"
#include "PxgSolverConstraintDesc.h"
#include "DyCpuGpu1dConstraint.h"
#include "pxgArtiPrepRef.h"

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

static std::vector<uint8_t> ajpLoader(const char* file) { return loadShaderBlob(file); }

const uint32_t AJP_NB_ARTI = 6;
const uint32_t AJP_NB_RIGID = 16;
const uint32_t AJP_REC_SIZE = 544;

const uint32_t AJP_RIGID_OFFSET = 3;
const uint32_t AJP_LIVE_WARPS = 4;
const uint32_t AJP_NB_BATCH_ALLOC = 8;
const uint32_t AJP_NB_PREP = 6;
const uint32_t AJP_MAX_ROWS = 20;          // Dy::MAX_CONSTRAINT_ROWS
const uint32_t AJP_NB_CON = 33;
const uint32_t AJP_NB_RESP = 33;
const uint32_t AJP_NB_CONSTANTS = 50;
const uint32_t AJP_NB_UNIQUE = 160;

static_assert(sizeof(PxgBlockConstraintBatch) == 1536, "batch stride");
static_assert(sizeof(PxgBlockConstraint1DData) == 1792, "1D data stride");
static_assert(sizeof(PxgBlockConstraint1DVelocities) == 2048, "1D velocities stride");
static_assert(sizeof(PxgBlockConstraint1DParameters) == 512, "1D parameters stride");
static_assert(sizeof(PxgTGSBlockSolverConstraint1DHeader) == 4992, "TGS 1D header");
static_assert(sizeof(PxgTGSBlockSolverConstraint1DCon) == 3328, "TGS 1D con");
static_assert(sizeof(PxgSolverConstraintManagerConstants) == 8, "manager constants");
// AJP_MAX_ROWS mirrors Dy::MAX_CONSTRAINT_ROWS (DyConstraintPrep.h:55)

const uint32_t AJP_BATCH_IDX[AJP_LIVE_WARPS] = { 2, 0, 3, 1 };
const uint32_t AJP_DESC[AJP_LIVE_WARPS] = { 1, 4, 0, 2 };
const uint32_t AJP_STRIDE[AJP_LIVE_WARPS] = { 26, 13, 32, 7 };
const uint32_t AJP_START_CON[AJP_LIVE_WARPS] = { 0, 8, 16, 24 };
const uint32_t AJP_RESP_IDX[AJP_LIVE_WARPS] = { 0, 8, 16, 24 };
const uint32_t AJP_PART_IDX[AJP_LIVE_WARPS] = { 3, 40, 74, 110 };   // mStartPartitionIndex

static float av(uint32_t i, uint32_t k)
{
    return (float((i * 53u + k * 31u) % 479u) + 1.0f) * 0.0078125f - 1.85f;
}

static uint32_t ajpNumLinks(uint32_t r) { return 4u + (r % 6u); }

static uint32_t ajpNumRows(uint32_t w, uint32_t lane)
{
    return (lane + AJP_DESC[w]) % 7u;   // 0..6, zero-row lanes included
}

static bool ajpRowDegenerate(uint32_t w, uint32_t lane, uint32_t row)
{
    return ((row * 7u + lane + w) % 9u) == 4u;
}

} // namespace

PXG_TEST(artiJointPrepWrap)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiJointPrep.spv", "artiTGSJointConstraintBlockPrepareParallelLaunch", 16u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &ajpLoader);

    // ---- the P8.102a body pools ----
    uint32_t totalLinks = 0;
    for (uint32_t r = 0; r < AJP_NB_ARTI; ++r) totalLinks += ajpNumLinks(r);

    std::vector<uint8_t> rec(size_t(AJP_NB_ARTI) * AJP_REC_SIZE, 0xAB);
    std::vector<PxTransform> b2w(totalLinks);
    std::vector<PxSpatialMatrix> resp(totalLinks);
    std::vector<UnAlignedSpatialVector> mv(totalLinks);
    std::vector<float> cfms(totalLinks);
    std::vector<PxgArticulationLink> links(totalLinks);
    std::memset(links.data(), 0xAB, links.size() * sizeof(links[0]));

    uint32_t base = 0;
    std::vector<uint32_t> linkBases(AJP_NB_ARTI);
    for (uint32_t r = 0; r < AJP_NB_ARTI; ++r)
    {
        linkBases[r] = base;
        for (uint32_t l = 0; l < ajpNumLinks(r); ++l)
        {
            const uint32_t i = base + l;
            PxQuat q(av(i, 0), av(i, 1), av(i, 2), 1.0f + 0.4f * av(i, 3));
            q.normalize();
            b2w[i] = PxTransform(PxVec3(av(i, 4), av(i, 5), av(i, 6)), q);
            for (uint32_t c = 0; c < 6; ++c)
                for (uint32_t rr = 0; rr < 6; ++rr)
                    resp[i].column[c][rr] = 0.3f * av(i, 10u + c * 6u + rr);
            mv[i] = UnAlignedSpatialVector(PxVec3(av(i, 50), av(i, 51), av(i, 52)),
                                           PxVec3(av(i, 53), av(i, 54), av(i, 55)));
            cfms[i] = 0.02f + 0.01f * std::fabs(av(i, 56));
            links[i].initialAngVel = PxVec3(av(i, 60), av(i, 61), av(i, 62));
            links[i].penBiasClamp = -0.5f - std::fabs(av(i, 63));
            links[i].offsetSlop = 0.01f * std::fabs(av(i, 64));
        }
        base += ajpNumLinks(r);
    }

    std::vector<PxgSolverBodyData> sbd(AJP_NB_RIGID);
    std::vector<PxgSolverTxIData> txi(AJP_NB_RIGID);
    std::vector<PxAlignedTransform> frames(AJP_NB_RIGID);
    std::memset(sbd.data(), 0xAB, sbd.size() * sizeof(sbd[0]));
    std::memset(txi.data(), 0xAB, txi.size() * sizeof(txi[0]));
    for (uint32_t i = 0; i < AJP_NB_RIGID; ++i)
    {
        const uint32_t kind = i % 4u;
        PxQuat q(av(i, 70), av(i, 71), av(i, 72), 1.0f + 0.3f * av(i, 73));
        q.normalize();
        frames[i].q = PxAlignedQuat(q.x, q.y, q.z, q.w);
        frames[i].p = make_float4(av(i, 74), av(i, 75), av(i, 76), av(i, 77));

        sbd[i].flags = (kind == 3u) ? PxU32(PxRigidBodyFlag::eKINEMATIC) : 0u;
        sbd[i].maxImpulse = 2.0f + std::fabs(av(i, 78));
        if (kind != 2u)
        {
            sbd[i].initialAngVel = PxVec3(av(i, 80), av(i, 81), av(i, 82));
            sbd[i].penBiasClamp = -0.4f - std::fabs(av(i, 83));
            sbd[i].initialLinVel = PxVec3(av(i, 84), av(i, 85), av(i, 86));
            sbd[i].invMass = 0.5f + std::fabs(av(i, 87));
            sbd[i].offsetSlop = 0.02f * std::fabs(av(i, 88));
        }
        if (kind == 0u || kind == 1u)
        {
            txi[i].sqrtInvInertia = PxMat33(PxVec3(av(i, 90), av(i, 91), av(i, 92)),
                                            PxVec3(av(i, 93), av(i, 94), av(i, 95)),
                                            PxVec3(av(i, 96), av(i, 97), av(i, 98)));
        }
    }

    // ---- batches ----
    std::vector<PxgBlockConstraintBatch> batches(AJP_NB_BATCH_ALLOC);
    std::memset(batches.data(), 0xAB, batches.size() * sizeof(batches[0]));
    std::vector<uint32_t> batchIndices(AJP_LIVE_WARPS);
    for (uint32_t w = 0; w < AJP_LIVE_WARPS; ++w)
    {
        batchIndices[w] = AJP_BATCH_IDX[w];
        PxgBlockConstraintBatch& b = batches[AJP_BATCH_IDX[w] + AJP_RIGID_OFFSET];
        b.mDescStride = PxU16(AJP_STRIDE[w]);
        b.constraintType = PxU16(0xABABu);
        b.mConstraintBatchIndex = AJP_DESC[w];
        b.mArticulationResponseIndex = AJP_RESP_IDX[w];
        b.startConstraintIndex = AJP_START_CON[w];
        b.startFrictionIndex = 0xABABABABu;   // unread by the joint prep
        b.blockContactIndex = 0xABABABABu;    // unread
        b.mStartPartitionIndex = AJP_PART_IDX[w];
        for (uint32_t l = 0; l < 32; ++l)
        {
            uint32_t aKind = (w == 2u) ? 0u : (w + l) % 4u;
            uint32_t bKind = (w == 2u) ? 0u : (w / 2u + l * 3u + 1u) % 4u;
            if (w == 1u) bKind = 3u;

            auto fill = [&](uint32_t kind, uint32_t s, PxNodeIndex* nodeArr, PxU32* idxArr,
                            PxU32* remapArr) {
                if (kind == 0u)
                {
                    const uint32_t r = (w * 3u + l + s) % AJP_NB_ARTI;
                    const uint32_t linkId = (l * 3u + 1u + s) % ajpNumLinks(r);
                    nodeArr[l] = PxNodeIndex(700u + w * 40u + l * 2u + s, linkId);
                    remapArr[l] = r;
                    idxArr[l] = (l * 5u + 2u + s) % AJP_NB_ARTI;
                }
                else if (kind == 3u)
                {
                    nodeArr[l] = PxNodeIndex();
                    idxArr[l] = (((l * 5u + s * 7u) % AJP_NB_RIGID) & ~3u) | 2u;
                    remapArr[l] = (l + s) % AJP_NB_ARTI;
                }
                else
                {
                    nodeArr[l] = PxNodeIndex(300u + w * 40u + l * 2u + s);
                    uint32_t bi = (w * 7u + l * 3u + s) % AJP_NB_RIGID;
                    bi = (bi & ~3u) | ((kind == 2u) ? 3u : (l % 2u));
                    idxArr[l] = bi;
                    remapArr[l] = (l + s + 1u) % AJP_NB_ARTI;
                }
            };
            fill(aKind, 0, b.bodyANodeIndex, b.bodyAIndex, b.remappedBodyAIndex);
            fill(bKind, 1, b.bodyBNodeIndex, b.bodyBIndex, b.remappedBodyBIndex);
        }
    }

    // ---- P9.86c: -0 link cfms, the zero tie of cfm = PxMax(b0.cfm, b1.cfm) ----
    // PxMax<float> (artiConstraintPrep2.cu:1828) is selectMax a>b?a:b: a {+0,-0}
    // tie keeps b1.  A link with cfmScale -0.0f (setCfmScale checks cfmScale >= 0,
    // NpArticulationLink.cpp:329, which -0 passes) has cfms[] = invMass * (-0 *
    // invLengthScale) = -0 (forwardDynamic2.cu:391/394/421/424); a rigid, kinematic
    // or static b0 has cfm = 0.f (cu:131).  The CUDA stores -0 into
    // jointHeader->cfm, the pre-P9.86 port's stamped FMax stored +0.  The B link of
    // every lane < 8 with a rigid A gets -0: links 14, 1 (warp 0) and 0, 26
    // (warp 3) -> 7 (+0,-0) lanes (warp 0 lanes 1,5,13,17,25; warp 3 lanes 2,6),
    // plus (-0,+0) and (-0|pos) lanes on which both spellings agree.
    uint32_t negCfmLinks = 0;
    for (uint32_t w = 0; w < AJP_LIVE_WARPS; ++w)
    {
        const PxgBlockConstraintBatch& b = batches[AJP_BATCH_IDX[w] + AJP_RIGID_OFFSET];
        for (uint32_t l = 0; l < 8u && l < b.mDescStride; ++l)
        {
            const PxNodeIndex nA = b.bodyANodeIndex[l];
            const PxNodeIndex nB = b.bodyBNodeIndex[l];
            if (!nA.isArticulation() && nB.isArticulation())
            {
                float& c = cfms[linkBases[b.remappedBodyBIndex[l]] + nB.articulationLinkId()];
                if (!std::signbit(c)) negCfmLinks++;
                c = -0.0f;   // cfmScale = -0.0f
            }
        }
    }
    PXG_CHECK_MSG(negCfmLinks == 4u, "-0 cfm links (%u)", negCfmLinks);

    // ---- the joint prep pools ----
    std::vector<PxgBlockConstraint1DData> cdata(AJP_NB_PREP);
    std::vector<PxgBlockConstraint1DVelocities> rvel(AJP_NB_PREP * AJP_MAX_ROWS);
    std::vector<PxgBlockConstraint1DParameters> rpar(AJP_NB_PREP * AJP_MAX_ROWS);
    std::memset(cdata.data(), 0xAB, cdata.size() * sizeof(cdata[0]));
    std::memset(rvel.data(), 0xAB, rvel.size() * sizeof(rvel[0]));
    std::memset(rpar.data(), 0xAB, rpar.size() * sizeof(rpar[0]));

    for (uint32_t w = 0; w < AJP_LIVE_WARPS; ++w)
    {
        const uint32_t d = AJP_DESC[w];
        PxgBlockConstraint1DData& cd = cdata[d];
        for (uint32_t l = 0; l < 32; ++l)
        {
            cd.mNumRows[l] = ajpNumRows(w, l);
            cd.mFlags[l] = (l % 3u == 1u) ? PxU32(PxConstraintFlag::eDRIVE_LIMITS_ARE_FORCES) : 0u;
            cd.mInvMassScale[l].linear0 = 1.0f + 0.1f * av(l, 400 + d);
            cd.mInvMassScale[l].angular0 = 1.0f + 0.1f * av(l, 401 + d);
            cd.mInvMassScale[l].linear1 = 1.0f + 0.1f * av(l, 402 + d);
            cd.mInvMassScale[l].angular1 = 1.0f + 0.1f * av(l, 403 + d);
            cd.mRAWorld_linBreakForce[l] =
                make_float4(av(l, 410 + d), av(l, 411 + d), av(l, 412 + d),
                            (l % 4u == 2u) ? 55.0f : PX_MAX_F32);
            cd.mRBWorld_AngBreakForce[l] =
                make_float4(av(l, 413 + d), av(l, 414 + d), av(l, 415 + d),
                            (l % 5u == 3u) ? 70.0f : PX_MAX_F32);

            for (uint32_t r = 0; r < 7u; ++r)   // rows 0..6 cover every numRows
            {
                PxgBlockConstraint1DVelocities& rv = rvel[size_t(d) * AJP_MAX_ROWS + r];
                PxgBlockConstraint1DParameters& rp = rpar[size_t(d) * AJP_MAX_ROWS + r];
                const bool degen = ajpRowDegenerate(w, l, r);
                const float s = degen ? 1e-8f : 1.0f;
                rv.linear0XYZ_geometricErrorW[l] =
                    make_float4(s * av(l, 420 + r * 8), s * av(l, 421 + r * 8),
                                s * av(l, 422 + r * 8), 0.05f * av(l, 423 + r * 8));
                rv.linear1XYZ_minImpulseW[l] =
                    make_float4(s * av(l, 424 + r * 8), s * av(l, 425 + r * 8),
                                s * av(l, 426 + r * 8), -40.0f - std::fabs(av(l, 427 + r * 8)));
                rv.angular0XYZ_velocityTargetW[l] =
                    make_float4(s * av(l, 428 + r * 8), s * av(l, 429 + r * 8),
                                s * av(l, 430 + r * 8), 0.4f * av(l, 431 + r * 8));
                rv.angular1XYZ_maxImpulseW[l] =
                    make_float4(s * av(l, 432 + r * 8), s * av(l, 433 + r * 8),
                                s * av(l, 434 + r * 8), 40.0f + std::fabs(av(l, 435 + r * 8)));

                // ties on purpose: only four distinct hint values
                rp.solveHint[l] = ((r * 13u + l * 5u) % 4u) * 0x100u;
                uint32_t fl = 0;
                if (r % 2u == 0u) fl |= PxU32(Px1DConstraintFlag::eSPRING);
                if (r % 4u == 0u) fl |= PxU32(Px1DConstraintFlag::eACCELERATION_SPRING);
                if (r % 2u == 1u && l % 3u == 0u) fl |= PxU32(Px1DConstraintFlag::eRESTITUTION);
                if (l % 2u == 0u) fl |= PxU32(Px1DConstraintFlag::eOUTPUT_FORCE);
                if (r % 3u == 1u) fl |= PxU32(Px1DConstraintFlag::eHAS_DRIVE_LIMIT);
                if ((r + l) % 2u == 0u) fl |= PxU32(Px1DConstraintFlag::eANGULAR_CONSTRAINT);
                rp.flags[l] = fl;
                rp.mods.spring.stiffness[l] = 4.0f + std::fabs(av(l, 440 + r));
                rp.mods.spring.damping[l] = 0.6f + 0.3f * std::fabs(av(l, 441 + r));
            }
        }
    }

    // ---- the writeback indirection ----
    std::vector<uint32_t> uniqueIdx(AJP_NB_UNIQUE);
    for (uint32_t k = 0; k < AJP_NB_UNIQUE; ++k)
        uniqueIdx[k] = (k * 7u + 3u) % AJP_NB_CONSTANTS;
    std::vector<PxgSolverConstraintManagerConstants> constants(AJP_NB_CONSTANTS);
    for (uint32_t k = 0; k < AJP_NB_CONSTANTS; ++k)
    {
        constants[k].mEdgeIndex = 0xABAB0000u + k;
        constants[k].mConstraintWriteBackIndex = 4000u + k * 13u;
    }

    // ---- float params ----
    const float dtF = 0.35f, stepDtF = 0.07f;
    const float invDtF = 1.0f / dtF, stepInvDtF = 1.0f / stepDtF;
    const float lengthScaleF = 1.0f, biasF = 0.7f;

    // ---- device uploads ----
    std::vector<PxgDevicePtr> bufs;
    auto up = [&](const void* p, size_t nsz) {
        PxgDevicePtr dp = 0;
        shim.memAlloc(&dp, nsz);
        if (dp && p) shim.memcpyHtoD(dp, p, nsz);
        bufs.push_back(dp);
        return dp;
    };
    PxgDevicePtr dB2w = up(b2w.data(), b2w.size() * sizeof(b2w[0]));
    PxgDevicePtr dResp = up(resp.data(), resp.size() * sizeof(resp[0]));
    PxgDevicePtr dMv = up(mv.data(), mv.size() * sizeof(mv[0]));
    PxgDevicePtr dCfm = up(cfms.data(), cfms.size() * 4);
    PxgDevicePtr dLinks = up(links.data(), links.size() * sizeof(links[0]));
    for (uint32_t r = 0; r < AJP_NB_ARTI; ++r)
    {
        uint8_t* R = &rec[size_t(r) * AJP_REC_SIZE];
        auto w64at = [&](size_t off, uint64_t v) { std::memcpy(R + off, &v, 8); };
        w64at(offsetof(PxgArticulation, links), uint64_t(dLinks) + uint64_t(linkBases[r]) * 80u);
        w64at(offsetof(PxgArticulation, motionVelocities),
              uint64_t(dMv) + uint64_t(linkBases[r]) * 24u);
        w64at(offsetof(PxgArticulation, spatialResponseMatrixW),
              uint64_t(dResp) + uint64_t(linkBases[r]) * 144u);
        w64at(offsetof(PxgArticulation, linkBody2Worlds),
              uint64_t(dB2w) + uint64_t(linkBases[r]) * 28u);
        w64at(offsetof(PxgArticulation, cfms), uint64_t(dCfm) + uint64_t(linkBases[r]) * 4u);
    }
    PxgDevicePtr dRec = up(rec.data(), rec.size());
    PxgDevicePtr dSbd = up(sbd.data(), sbd.size() * sizeof(sbd[0]));
    PxgDevicePtr dTxi = up(txi.data(), txi.size() * sizeof(txi[0]));
    PxgDevicePtr dFrames = up(frames.data(), frames.size() * sizeof(frames[0]));
    PxgDevicePtr dBatches = up(batches.data(), batches.size() * sizeof(batches[0]));
    PxgDevicePtr dBatchIdx = up(batchIndices.data(), batchIndices.size() * 4);
    PxgDevicePtr dCdata = up(cdata.data(), cdata.size() * sizeof(cdata[0]));
    PxgDevicePtr dRvel = up(rvel.data(), rvel.size() * sizeof(rvel[0]));
    PxgDevicePtr dRpar = up(rpar.data(), rpar.size() * sizeof(rpar[0]));
    PxgDevicePtr dUnique = up(uniqueIdx.data(), uniqueIdx.size() * 4);
    PxgDevicePtr dConst = up(constants.data(), constants.size() * sizeof(constants[0]));

    std::vector<uint8_t> poisH(AJP_NB_PREP * sizeof(PxgTGSBlockSolverConstraint1DHeader), 0xAB);
    std::vector<uint8_t> poisC(AJP_NB_CON * sizeof(PxgTGSBlockSolverConstraint1DCon), 0xAB);
    std::vector<uint8_t> poisR(AJP_NB_RESP * sizeof(PxgArticulationBlockResponse), 0xAB);
    PxgDevicePtr dH = up(poisH.data(), poisH.size());
    PxgDevicePtr dC = up(poisC.data(), poisC.size());
    PxgDevicePtr dR = up(poisR.data(), poisR.size());

    // ---- the engine descs ----
    PxgConstraintPrepareDesc prep;
    std::memset(&prep, 0xAB, sizeof(prep));
    prep.artiJointConstraintBatchIndices = reinterpret_cast<PxU32*>(dBatchIdx);
    prep.solverBodyDataPool = reinterpret_cast<PxgSolverBodyData*>(dSbd);
    prep.solverBodyTxIDataPool = reinterpret_cast<PxgSolverTxIData*>(dTxi);
    prep.body2WorldPool = reinterpret_cast<PxAlignedTransform*>(dFrames);
    prep.blockJointPrepPool = reinterpret_cast<PxgBlockConstraint1DData*>(dCdata);
    prep.blockJointPrepPool0 = reinterpret_cast<PxgBlockConstraint1DVelocities*>(dRvel);
    prep.blockJointPrepPool1 = reinterpret_cast<PxgBlockConstraint1DParameters*>(dRpar);
    prep.artiConstraintUniqueIndices = reinterpret_cast<PxU32*>(dUnique);
    prep.solverConstantData = reinterpret_cast<PxgSolverConstraintManagerConstants*>(dConst);
    prep.numArti1dConstraintBatches = 2;
    prep.numArtiStatic1dConstraintBatches = 1;
    prep.numArtiSelf1dConstraintBatches = 1;
    prep.numBatches = AJP_RIGID_OFFSET;
    prep.biasCoefficient = biasF;

    PxgSolverSharedDesc<IterativeSolveDataTGS> shared;
    std::memset(&shared, 0xAB, sizeof(shared));
    shared.articulations = reinterpret_cast<PxgArticulation*>(dRec);
    shared.dt = dtF;
    shared.stepDt = stepDtF;
    shared.invDtF32 = invDtF;
    shared.stepInvDtF32 = stepInvDtF;
    shared.lengthScale = lengthScaleF;
    shared.iterativeData.blockConstraintBatch =
        reinterpret_cast<PxgBlockConstraintBatch*>(dBatches);
    shared.iterativeData.blockJointConstraintHeaders =
        reinterpret_cast<PxgTGSBlockSolverConstraint1DHeader*>(dH);
    shared.iterativeData.blockJointConstraintRowsCon =
        reinterpret_cast<PxgTGSBlockSolverConstraint1DCon*>(dC);
    shared.iterativeData.artiResponse = reinterpret_cast<PxgArticulationBlockResponse*>(dR);

    PxgDevicePtr dPrep = up(&prep, sizeof(prep));
    PxgDevicePtr dShared = up(&shared, sizeof(shared));

    bool okAlloc = true;
    for (PxgDevicePtr p : bufs) okAlloc = okAlloc && (p != 0);
    if (!okAlloc)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
        return;
    }

    // ---- expected via the host mirror ----
    std::vector<PxgTGSBlockSolverConstraint1DHeader> wantH(AJP_NB_PREP);
    std::vector<PxgTGSBlockSolverConstraint1DCon> wantC(AJP_NB_CON);
    std::vector<PxgArticulationBlockResponse> wantR(AJP_NB_RESP);
    std::memset(wantH.data(), 0xAB, wantH.size() * sizeof(wantH[0]));
    std::memset(wantC.data(), 0xAB, wantC.size() * sizeof(wantC[0]));
    std::memset(wantR.data(), 0xAB, wantR.size() * sizeof(wantR[0]));

    uint32_t skippedRows = 0, tieLanes = 0, kinLanes = 0, zeroRowLanes = 0;
    uint32_t cfmNeg0Lanes = 0, cfmTwinDiffLanes = 0;
    auto fbits = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; };
    for (uint32_t w = 0; w < AJP_LIVE_WARPS; ++w)
    {
        const PxgBlockConstraintBatch& b = batches[AJP_BATCH_IDX[w] + AJP_RIGID_OFFSET];
        const uint32_t d = b.mConstraintBatchIndex;
        for (uint32_t lane = 0; lane < b.mDescStride; ++lane)
        {
            const PxNodeIndex nodeA = b.bodyANodeIndex[lane];
            const PxNodeIndex nodeB = b.bodyBNodeIndex[lane];
            const uint32_t biA = nodeA.isArticulation() ? b.remappedBodyAIndex[lane]
                                                        : b.bodyAIndex[lane];
            const uint32_t biB = nodeB.isArticulation() ? b.remappedBodyBIndex[lane]
                                                        : b.bodyBIndex[lane];
            PxgSolverExtBody2 b0, b1;
            refCreateExtBody(nodeA, biA, linkBases.data(), b2w.data(), resp.data(), mv.data(),
                             cfms.data(), links.data(), sbd.data(), txi.data(), frames.data(), b0);
            refCreateExtBody(nodeB, biB, linkBases.data(), b2w.data(), resp.data(), mv.data(),
                             cfms.data(), links.data(), sbd.data(), txi.data(), frames.data(), b1);
            if (b0.isKinematic || b1.isKinematic) kinLanes++;

            PxgTGSBlockSolverConstraint1DHeader& H = wantH[d];
            const PxgBlockConstraint1DData& cd = cdata[d];

            const uint32_t uniqueIndex = uniqueIdx[b.mStartPartitionIndex + lane];
            H.writeBackOffset[lane] = constants[uniqueIndex].mConstraintWriteBackIndex;

            const PxgConstraintInvMassScale& imsH = cd.mInvMassScale[lane];
            const float4 ims =
                make_float4(imsH.linear0, imsH.angular0, imsH.linear1, imsH.angular1);
            const float4 ra4 = cd.mRAWorld_linBreakForce[lane];
            const float4 rb4 = cd.mRBWorld_AngBreakForce[lane];
            const float linBreakImpulse = ra4.w * stepDtF;
            const float angBreakImpulse = rb4.w * stepDtF;
            H.rAWorld_invMass0D0[lane] = make_float4(ra4.x, ra4.y, ra4.z, -ims.x);
            H.rBWorld_invMass1D1[lane] = make_float4(rb4.x, rb4.y, rb4.z, ims.z);
            H.angBreakImpulse[lane] = angBreakImpulse;
            H.invInertiaScale0[lane] = -ims.y;
            H.invInertiaScale1[lane] = ims.w;
            H.linBreakImpulse[lane] = linBreakImpulse;

            const uint32_t numRows = cd.mNumRows[lane];
            if (numRows == 0) zeroRowLanes++;
            const uint32_t breakable = ((ra4.w != PX_MAX_F32) || (rb4.w != PX_MAX_F32)) ? 1u : 0u;

            // preprocessRowsBlock
            uint32_t sorted[AJP_MAX_ROWS];
            bool tie = false;
            for (uint32_t r = 0; r < numRows; ++r)
            {
                const uint32_t hintR = rpar[size_t(d) * AJP_MAX_ROWS + r].solveHint[lane];
                uint32_t j = r;
                for (; j > 0 && hintR
                         < rpar[size_t(d) * AJP_MAX_ROWS + sorted[j - 1]].solveHint[lane]; --j)
                    sorted[j] = sorted[j - 1];
                if (j > 0
                    && hintR == rpar[size_t(d) * AJP_MAX_ROWS + sorted[j - 1]].solveHint[lane])
                    tie = true;
                sorted[j] = r;
            }
            if (tie) tieLanes++;

            // setUpArti1DConstraintBlock TGS
            const float erp = 0.5f * biasF;
            const float cfm = PxMax(b0.cfm, b1.cfm);   // selectMax (cu:1828)
            // P9.86c: the mutant twin -- the pre-P9.86 port's FMax (llvmpipe: +0 for
            // any {+0,-0} tie) -- must differ on the directed lanes
            if (fbits(cfm) == 0x80000000u) cfmNeg0Lanes++;
            if (fbits(cfm) != fbits(acpLpFMax(b0.cfm, b1.cfm))) cfmTwinDiffLanes++;
            uint32_t outCount = 0;
            for (uint32_t r = 0; r < numRows; ++r)
            {
                const uint32_t index = sorted[r];
                const PxgBlockConstraint1DParameters& rp =
                    rpar[size_t(d) * AJP_MAX_ROWS + index];
                const PxgBlockConstraint1DVelocities& rv =
                    rvel[size_t(d) * AJP_MAX_ROWS + index];
                const uint32_t rpFlags = rp.flags[lane];
                const uint32_t rpSolveHint = rp.solveHint[lane];

                const float4 l0 = rv.linear0XYZ_geometricErrorW[lane];
                const float4 l1 = rv.linear1XYZ_minImpulseW[lane];
                const float4 a0 = rv.angular0XYZ_velocityTargetW[lane];
                const float4 a1 = rv.angular1XYZ_maxImpulseW[lane];
                const PxVec3 clin0(l0.x, l0.y, l0.z), clin1(l1.x, l1.y, l1.z);
                const PxVec3 cang0(a0.x, a0.y, a0.z), cang1(a1.x, a1.y, a1.z);

                const UnAlignedSpatialVector response0 =
                    refCreateImpulseResponseVector(b0, UnAlignedSpatialVector(cang0, clin0));
                const UnAlignedSpatialVector response1 =
                    refCreateImpulseResponseVector(b1, UnAlignedSpatialVector(-cang1, -clin1));
                const UnAlignedSpatialVector deltaV0 = refGetImpulseResponse(
                    b0, UnAlignedSpatialVector(clin0, cang0).scale(ims.x, ims.y));
                const UnAlignedSpatialVector deltaV1 = refGetImpulseResponse(
                    b1, UnAlignedSpatialVector(-clin1, -cang1).scale(ims.z, ims.w));
                const float resp0 = deltaV0.dot(response0);
                const float resp1 = deltaV1.dot(response1);
                const float unitResponse = resp0 + resp1;

                if (unitResponse <= 1e-12f)   // DY_ARTICULATION_MIN_RESPONSE
                {
                    skippedRows++;
                    continue;
                }

                PxgTGSBlockSolverConstraint1DCon& C = wantC[b.startConstraintIndex + outCount];
                C.resp0[lane] = resp0;
                C.resp1[lane] = resp1;

                PxgArticulationBlockResponse& R = wantR[b.mArticulationResponseIndex + outCount];
                R.deltaRALin_x[lane] = deltaV0.bottom.x;
                R.deltaRALin_y[lane] = deltaV0.bottom.y;
                R.deltaRALin_z[lane] = deltaV0.bottom.z;
                R.deltaRAAng_x[lane] = deltaV0.top.x;
                R.deltaRAAng_y[lane] = deltaV0.top.y;
                R.deltaRAAng_z[lane] = deltaV0.top.z;
                R.deltaRBLin_x[lane] = deltaV1.bottom.x;
                R.deltaRBLin_y[lane] = deltaV1.bottom.y;
                R.deltaRBLin_z[lane] = deltaV1.bottom.z;
                R.deltaRBAng_x[lane] = deltaV1.top.x;
                R.deltaRBAng_y[lane] = deltaV1.top.y;
                R.deltaRBAng_z[lane] = deltaV1.top.z;

                PxReal jointSpeedForRestitutionBounce, initJointSpeed;
                {
                    const float vel0 =
                        refProjectVelocity(b0, UnAlignedSpatialVector(cang0, clin0));
                    const float vel1 =
                        refProjectVelocity(b1, UnAlignedSpatialVector(cang1, clin1));
                    Dy::computeJointSpeedTGS(vel0, b0.isKinematic != 0, vel1,
                                             b1.isKinematic != 0,
                                             jointSpeedForRestitutionBounce, initJointSpeed);
                }

                const float geometricError = l0.w;
                const float velocityTarget = a0.w;
                const float stiffness = rp.mods.spring.stiffness[lane];
                const float damping = rp.mods.spring.damping[lane];
                const float restitution = rp.mods.bounce.restitution[lane];
                const float bounceVelocityThreshold = rp.mods.bounce.velocityThreshold[lane];

                const float maxBiasVelocity = Dy::computeMaxBiasVelocityTGS(
                    PxU16(rpFlags), jointSpeedForRestitutionBounce, bounceVelocityThreshold,
                    restitution, geometricError, true, lengthScaleF, invDtF);

                PxReal coeff0, coeff1, coeff2, coeff3;
                Dy::queryReduced1dConstraintSolverConstantsTGS(
                    PxU16(rpFlags), stiffness, damping, restitution, bounceVelocityThreshold,
                    geometricError, velocityTarget, jointSpeedForRestitutionBounce,
                    initJointSpeed, erp, stepDtF, stepInvDtF, coeff0, coeff1, coeff2, coeff3);

                C.lin0XYZ_initBiasOrCoeff0[lane] =
                    make_float4(response0.bottom.x, response0.bottom.y, response0.bottom.z, coeff0);
                C.lin1XYZ_biasScaleOrCoeff1[lane] = make_float4(
                    -response1.bottom.x, -response1.bottom.y, -response1.bottom.z, coeff1);
                C.ang0XYZ_velMultiplierOrCoeff2[lane] =
                    make_float4(response0.top.x, response0.top.y, response0.top.z, coeff2);
                C.ang1XYZ_velTargetOrCoeff3[lane] =
                    make_float4(-response1.top.x, -response1.top.y, -response1.top.z, coeff3);
                C.geometricError[lane] = geometricError;
                C.maxBias[lane] = maxBiasVelocity;
                C.appliedForce[lane] = 0.f;
                C.angularErrorScale[lane] =
                    (rpFlags & Px1DConstraintFlag::eANGULAR_CONSTRAINT) ? 1.f : 0.f;

                const bool hasDriveLimit = (rpFlags & Px1DConstraintFlag::eHAS_DRIVE_LIMIT) != 0;
                const bool driveLimitsAreForces =
                    (cd.mFlags[lane] & PxConstraintFlag::eDRIVE_LIMITS_ARE_FORCES) != 0;
                Dy::computeMinMaxImpulseOrForceAsImpulse(l1.w, a1.w, hasDriveLimit,
                                                         driveLimitsAreForces, dtF,
                                                         C.minImpulse[lane], C.maxImpulse[lane]);

                PxU32 flags = 0;
                Dy::raiseInternalFlagsTGS(PxU16(rpFlags), PxU16(rpSolveHint), flags);
                C.flags[lane] = flags;

                outCount++;
            }

            H.rowCounts_breakable_orthoAxisCount[lane] =
                make_uchar4(PxU8(outCount), PxU8(breakable), 0, 0);
            H.cfm[lane] = cfm;
        }
    }

    PXG_CHECK_MSG(skippedRows >= 2, "degenerate-row coverage");
    PXG_CHECK_MSG(tieLanes >= 1, "solveHint-tie coverage");
    PXG_CHECK_MSG(kinLanes >= 1, "kinematic coverage");
    PXG_CHECK_MSG(zeroRowLanes >= 1, "zero-row coverage");
    std::printf("  ajp cfm zero tie: -0 lanes %u, FMax-twin lanes %u\n", cfmNeg0Lanes,
                cfmTwinDiffLanes);
    PXG_CHECK_MSG(cfmNeg0Lanes == 7u, "cfm PxMax(+0,-0) = -0 coverage (%u lanes)", cfmNeg0Lanes);
    PXG_CHECK_MSG(cfmTwinDiffLanes == 7u, "cfm FMax-twin coverage (%u lanes)", cfmTwinDiffLanes);

    // ---- launch: 3 blocks x 64 = 6 warps, 4 live ----
    struct Params { uint64_t prepDesc, sharedDesc; } prm = { uint64_t(dPrep), uint64_t(dShared) };
    static_assert(sizeof(Params) == 16, "params blob");
    PXG_CHECK_MSG(shim.launchKernel(0, 3, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch artiTGSJointConstraintBlockPrepareParallelLaunch");
    shim.streamSynchronize();

    // ---- compare ----
    size_t bad = 0;
    auto cmp = [&](PxgDevicePtr dp, const void* wantp, size_t nsz, const char* tag) {
        std::vector<uint32_t> got(nsz / 4);
        shim.memcpyDtoH(got.data(), dp, nsz);
        const uint32_t* wv = reinterpret_cast<const uint32_t*>(wantp);
        for (size_t k = 0; k < got.size(); ++k)
            if (wv[k] != got[k])   // BITWISE: the sign of zero included (P9.86c)
            {
                if (bad < 12)
                    std::printf("MM ajp %s word %zu (lane %zu)  ref %08x  gpu %08x\n",
                                tag, k, k % 32, wv[k], got[k]);
                ++bad;
            }
    };
    cmp(dH, wantH.data(), wantH.size() * sizeof(wantH[0]), "headers");
    cmp(dC, wantC.data(), wantC.size() * sizeof(wantC[0]), "cons");
    cmp(dR, wantR.data(), wantR.size() * sizeof(wantR[0]), "responses");
    PXG_CHECK_MSG(bad == 0, "%zu mismatches", bad);

    for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
}
