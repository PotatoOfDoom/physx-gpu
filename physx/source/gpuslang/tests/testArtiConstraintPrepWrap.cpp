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

// testArtiConstraintPrepWrap.cpp -- P8.102c: the ENGINE entry
// artiTGSContactConstraintBlockPrepareLaunch driven over REAL
// PxgConstraintPrepareDesc / PxgSolverSharedDesc<IterativeSolveDataTGS>
// images, against a FULL host mirror: refCreateExtBody (P8.102a) +
// refCorrelatePatches / refGetFrictionPatches / refGrowPatches (the
// dispatcher, over the engine's own PX_CUDA_CALLABLE PxAlignedTransform
// arithmetic) + refSetupExtContactsTGS (P8.102b) + the post-setup
// patchIndex/anchorPoints writes via PxTransform::transform.
//
// Geometry: 4 live batches (2 contact + 1 static + 1 self) behind a
// rigid-batch offset of 3, batchIndices permuted, descStride 26/13/32/7,
// THREE blocks launched (6 warps) so warps 4..5 pin the guard.  The desc-1
// batch carries engineered friction-cache lanes:
//   lane  2  inherit 2 anchors, extents LARGE + clustered contacts -> the
//            growPatches keep-two early return
//   lane  4  inherit 2 anchors via prev slot a=1 (a=0 rejects), tiny
//            extents + spread contacts -> rebuild
//   lane  6  broken=1 reject; lane 8 normal-mismatch reject; lane 10
//            anchorCount>nbContacts reject; lane 12 pointsAreClose reject
//   lane 14  inherit ONE anchor -> growPatches' keep-one-add-one path
// All engineered coverage is asserted, never hoped for.

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
#include "PxgConstraintPrep.h"
#include "PxgFrictionPatch.h"
#include "PxgSolverConstraintBlock1D.h"
#include "PxgConstraintBlock.h"
#include "PxgSolverCoreDesc.h"
#include "PxgSolverConstraintDesc.h"
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

static std::vector<uint8_t> acwLoader(const char* file) { return loadShaderBlob(file); }

const uint32_t ACW_NB_ARTI = 6;
const uint32_t ACW_NB_RIGID = 16;
// P9.86c: two EXTRA dynamic pool entries past the % ACW_NB_RIGID mapping (so no
// existing lane moves): 16 has maxContactImpulse -0 (setMaxContactImpulse(-0.f)
// passes the API's >= 0 check), 17 has +0 -- the zero-tie bodies of the
// directed lanes below
const uint32_t ACW_NB_RIGID_POOL = ACW_NB_RIGID + 2;
const uint32_t ACW_REC_SIZE = 496;

const uint32_t ACW_RIGID_OFFSET = 3;      // prepDesc.numBatches
const uint32_t ACW_LIVE_WARPS = 4;        // 2 contact + 1 static + 1 self
const uint32_t ACW_NB_BATCH_ALLOC = 8;
const uint32_t ACW_NB_PREP = 6;           // desc-indexed pools
const uint32_t ACW_NB_PREV = 5;
const uint32_t ACW_TOTAL_PREV_EDGES = 89;
const uint32_t ACW_TOTAL_CURR_EDGES = 97;
const uint32_t ACW_NB_CONTACT_ROWS = 13;  // input PxgBlockContactPoint pool
const uint32_t ACW_NB_OUT_CP = 16;
const uint32_t ACW_NB_OUT_FR = 20;
const uint32_t ACW_NB_RESP = 33;
const uint32_t ACW_NB_FIDX = ACW_TOTAL_CURR_EDGES * 2;   // currFrictionIndices
const uint32_t ACW_NB_PREV_FIDX = ACW_TOTAL_PREV_EDGES * 2;

static_assert(sizeof(PxgBlockConstraintBatch) == 1536, "batch stride");
static_assert(sizeof(PxgBlockWorkUnit) == 1024, "work unit stride");
static_assert(sizeof(PxgConstraintPrepareDesc) == 320, "prep desc");
static_assert(sizeof(PxgSolverSharedDesc<IterativeSolveDataTGS>) == 224, "TGS shared desc");
static_assert(sizeof(PxgBlockFrictionIndex) == 8, "friction index");

// per-warp batch table
const uint32_t ACW_BATCH_IDX[ACW_LIVE_WARPS] = { 2, 0, 3, 1 };            // batchIndices[i]
const uint32_t ACW_DESC[ACW_LIVE_WARPS] = { 1, 4, 0, 2 };                 // mConstraintBatchIndex
const uint32_t ACW_STRIDE[ACW_LIVE_WARPS] = { 26, 13, 32, 7 };
const uint32_t ACW_RESP_IDX[ACW_LIVE_WARPS] = { 0, 8, 16, 24 };  // DISJOINT 8-row windows
const uint32_t ACW_START_CON[ACW_LIVE_WARPS] = { 0, 4, 8, 12 };
const uint32_t ACW_START_FRIC[ACW_LIVE_WARPS] = { 0, 5, 10, 15 };
const uint32_t ACW_CONTACT_BASE[ACW_LIVE_WARPS] = { 0, 3, 6, 9 };
const uint32_t ACW_EDGE_BASE[ACW_LIVE_WARPS] = { 5, 31, 44, 76 };  // NONZERO base 0: edge != lane

static float av(uint32_t i, uint32_t k)
{
    return (float((i * 53u + k * 31u) % 479u) + 1.0f) * 0.0078125f - 1.85f;
}

static uint32_t acwNumLinks(uint32_t r) { return 4u + (r % 6u); }

static uint32_t acwNbContacts(uint32_t w, uint32_t lane)
{
    if (w == 0u)
    {
        if (lane == 2u) return 3u;
        if (lane == 4u) return 2u;
        if (lane == 10u) return 1u;   // anchorCount > nbContacts reject
        if (lane == 14u) return 2u;
        if (lane == 16u) return 3u;
    }
    return (lane + ACW_DESC[w]) % 4u;
}

static bool acwPerPoint(uint32_t w, uint32_t lane)
{
    if (w == 0u && lane == 16u) return true;   // gate-pin lane
    if (w == 0u && (lane == 2u || lane == 4u || lane == 6u || lane == 8u
                    || lane == 10u || lane == 12u || lane == 14u))
        return false;
    return lane % 7u == 6u;
}

static bool acwDisable(uint32_t w, uint32_t lane)
{
    if (w == 0u && (lane == 2u || lane == 4u || lane == 6u || lane == 8u
                    || lane == 10u || lane == 12u || lane == 14u))
        return false;
    return lane % 11u == 9u;
}

static uint32_t acwPrevCount(uint32_t w, uint32_t lane)
{
    if (w == 0u && (lane == 2u || lane == 6u || lane == 8u || lane == 10u
                    || lane == 12u || lane == 14u || lane == 16u))
        return 1u;
    if (w == 0u && lane == 4u) return 2u;   // match sits at a = 1
    return lane % 3u;
}

} // namespace

PXG_TEST(artiConstraintPrepWrap)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiConstraintPrep.spv", "artiTGSContactConstraintBlockPrepareLaunch", 16u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &acwLoader);

    // ---- the P8.102a body pools ----
    uint32_t totalLinks = 0;
    for (uint32_t r = 0; r < ACW_NB_ARTI; ++r) totalLinks += acwNumLinks(r);

    std::vector<uint8_t> rec(size_t(ACW_NB_ARTI) * ACW_REC_SIZE, 0xAB);
    std::vector<PxTransform> b2w(totalLinks);
    std::vector<PxSpatialMatrix> resp(totalLinks);
    std::vector<UnAlignedSpatialVector> mv(totalLinks);
    std::vector<float> cfms(totalLinks);
    std::vector<PxgArticulationLink> links(totalLinks);
    std::memset(links.data(), 0xAB, links.size() * sizeof(links[0]));

    uint32_t base = 0;
    std::vector<uint32_t> linkBases(ACW_NB_ARTI);
    for (uint32_t r = 0; r < ACW_NB_ARTI; ++r)
    {
        linkBases[r] = base;
        for (uint32_t l = 0; l < acwNumLinks(r); ++l)
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
        base += acwNumLinks(r);
    }

    std::vector<PxgSolverBodyData> sbd(ACW_NB_RIGID_POOL);
    std::vector<PxgSolverTxIData> txi(ACW_NB_RIGID_POOL);
    std::vector<PxAlignedTransform> frames(ACW_NB_RIGID_POOL);
    std::memset(sbd.data(), 0xAB, sbd.size() * sizeof(sbd[0]));
    std::memset(txi.data(), 0xAB, txi.size() * sizeof(txi[0]));
    for (uint32_t i = 0; i < ACW_NB_RIGID_POOL; ++i)
    {
        const uint32_t kind = i % 4u;   // 0/1 dynamic, 2 static, 3 kinematic
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

    // ---- the batches ----
    std::vector<PxgBlockConstraintBatch> batches(ACW_NB_BATCH_ALLOC);
    std::memset(batches.data(), 0xAB, batches.size() * sizeof(batches[0]));
    std::vector<uint32_t> batchIndices(ACW_LIVE_WARPS);
    for (uint32_t w = 0; w < ACW_LIVE_WARPS; ++w)
    {
        batchIndices[w] = ACW_BATCH_IDX[w];
        PxgBlockConstraintBatch& b = batches[ACW_BATCH_IDX[w] + ACW_RIGID_OFFSET];
        b.mDescStride = PxU16(ACW_STRIDE[w]);
        b.constraintType = PxU16(0xABABu);   // unread by this kernel
        b.mConstraintBatchIndex = ACW_DESC[w];
        b.mArticulationResponseIndex = ACW_RESP_IDX[w];
        b.startConstraintIndex = ACW_START_CON[w];
        b.startFrictionIndex = ACW_START_FRIC[w];
        b.blockContactIndex = ACW_CONTACT_BASE[w];
        for (uint32_t l = 0; l < 32; ++l)
        {
            // kinds: self batch (w==2) is arti-arti; otherwise mixed
            uint32_t aKind = (w == 2u) ? 0u : (w + l) % 4u;
            uint32_t bKind = (w == 2u) ? 0u : (w / 2u + l * 3u + 1u) % 4u;
            if (w == 1u) bKind = 3u;   // the static batch: B always static

            auto fill = [&](uint32_t kind, uint32_t s, PxNodeIndex* nodeArr, PxU32* idxArr,
                            PxU32* remapArr) {
                if (kind == 0u)
                {
                    const uint32_t r = (w * 3u + l + s) % ACW_NB_ARTI;
                    const uint32_t linkId = (l * 3u + 1u + s) % acwNumLinks(r);
                    nodeArr[l] = PxNodeIndex(700u + w * 40u + l * 2u + s, linkId);
                    remapArr[l] = r;
                    idxArr[l] = (l * 5u + 2u + s) % ACW_NB_ARTI;    // valid rigid AND record index, WRONG if used
                }
                else if (kind == 3u)
                {
                    nodeArr[l] = PxNodeIndex();
                    idxArr[l] = (((l * 5u + s * 7u) % ACW_NB_RIGID) & ~3u) | 2u;
                    remapArr[l] = (l + s) % ACW_NB_ARTI;            // valid but WRONG if used
                }
                else
                {
                    nodeArr[l] = PxNodeIndex(300u + w * 40u + l * 2u + s);
                    uint32_t bi = (w * 7u + l * 3u + s) % ACW_NB_RIGID;
                    bi = (bi & ~3u) | ((kind == 2u) ? 3u : (l % 2u));
                    idxArr[l] = bi;
                    remapArr[l] = (l + s + 1u) % ACW_NB_ARTI;       // valid but WRONG if used
                }
            };
            fill(aKind, 0, b.bodyANodeIndex, b.bodyAIndex, b.remappedBodyAIndex);
            fill(bKind, 1, b.bodyBNodeIndex, b.bodyBIndex, b.remappedBodyBIndex);
        }
    }

    // ---- P9.86c: the signed-zero tie lanes (warp 0 = desc 1, none of them an
    // engineered friction-cache lane, a perPoint lane or a disabled lane) ----
    //   lane  0  link (arti 0, link 1) vs dynamic pool 0, one contact at
    //            separation +0.02, torsional (0.7, +0): minPen stays the +0 seed
    //            -> torsionalFrictionScale = PxMax(+0, sqrt(PxMax(0,-0)*x)) = -0
    //   lane  1  dynamic pool 16 (maxImpulse -0) vs link (arti 2, link 5) whose
    //            cfm is -0 (cfmScale -0): contactHeader.cfm = PxMax(+0,-0) = -0;
    //            contact 0 has w = +0 (PxContactSet::ignore()): PxMin(-0,+0) = +0
    //   lane 21  dynamic pool 16 (-0) vs dynamic pool 17 (+0), B switched from
    //            its link: maxImpulse = PxMin(-0,+0) = +0
    //   lane 24  link vs dynamic pool 8, one contact at separation -0 against
    //            restDistance +0: penetration -0, PxMin(-0,+0) keeps the +0 seed
    const uint32_t ACW_ZS_LANES[4] = { 0u, 1u, 21u, 24u };
    sbd[16].maxImpulse = -0.0f;
    sbd[17].maxImpulse = 0.0f;
    {
        PxgBlockConstraintBatch& b0w = batches[ACW_BATCH_IDX[0] + ACW_RIGID_OFFSET];
        b0w.bodyAIndex[1] = 16u;                      // node 302 stays rigid
        b0w.bodyAIndex[21] = 16u;                     // node 342 stays rigid
        b0w.bodyBNodeIndex[21] = PxNodeIndex(343u);   // link -> rigid
        b0w.bodyBIndex[21] = 17u;
        const PxNodeIndex nB1 = b0w.bodyBNodeIndex[1];
        PXG_CHECK_MSG(nB1.isArticulation() && b0w.remappedBodyBIndex[1] == 2u
                          && nB1.articulationLinkId() == 5u,
                      "lane 1: B must be (arti 2, link 5)");
        cfms[linkBases[2] + 5u] = -0.0f;
    }

    // ---- desc-indexed prep pools ----
    std::vector<PxgBlockWorkUnit> workUnits(ACW_NB_PREP);
    std::vector<PxgBlockContactData> cdPool(ACW_NB_PREP);
    std::memset(workUnits.data(), 0xAB, workUnits.size() * sizeof(workUnits[0]));
    std::memset(cdPool.data(), 0xAB, cdPool.size() * sizeof(cdPool[0]));
    std::vector<PxgBlockContactPoint> contacts(ACW_NB_CONTACT_ROWS);
    std::memset(contacts.data(), 0xAB, contacts.size() * sizeof(contacts[0]));

    for (uint32_t w = 0; w < ACW_LIVE_WARPS; ++w)
    {
        const uint32_t d = ACW_DESC[w];
        PxgBlockWorkUnit& wu = workUnits[d];
        PxgBlockContactData& cd = cdPool[d];
        for (uint32_t l = 0; l < 32; ++l)
        {
            wu.mWriteback[l] = 3000u + w * 100u + l;
            wu.mEdgeIndex[l] = ACW_EDGE_BASE[w] + l;
            wu.mPatchIndex[l] = l % 2u;
            wu.mFrictionPatchIndex[l] = 7000u + w * 50u + l;
            wu.mTorsionalFrictionData[l] =
                (l % 2u) ? make_float2(0.7f, 0.3f) : make_float2(0.0f, 0.0f);

            const uint32_t nb = acwNbContacts(w, l);
            const uint32_t flags = (acwPerPoint(w, l) ? (2u | 4u) : 0u)
                                 | (acwDisable(w, l) ? 8u : 0u) | ((l & 1u) ? 16u : 0u);
            cd.contactData[l].restDistance = 0.005f * av(l, 300 + d);
            cd.contactData[l].staticFriction = 0.6f + 0.2f * std::fabs(av(l, 301));
            cd.contactData[l].dynamicFriction = 0.4f + 0.2f * std::fabs(av(l, 302));
            cd.contactData[l].mNumContacts = PxU8(nb);
            cd.contactData[l].mSolverFlags = PxU8(flags);
            cd.contactData[l].prevFrictionPatchCount = PxU8(acwPrevCount(w, l));
            cd.damping[l] = 15.0f + float(l);
            PxVec3 n(av(l, 310 + d) + 0.3f, av(l, 311 + d), av(l, 312 + d) - 0.2f);
            n.normalize();
            const float rest = (l % 5u == 0u) ? 0.5f : ((l % 5u == 3u) ? -600.0f : 0.0f);
            cd.normal_restitutionW[l] = make_float4(n.x, n.y, n.z, rest);
            cd.mInvMassScale[l].linear0 = 1.0f + 0.1f * av(l, 320);
            cd.mInvMassScale[l].angular0 = 1.0f + 0.1f * av(l, 321);
            cd.mInvMassScale[l].linear1 = 1.0f + 0.1f * av(l, 322);
            cd.mInvMassScale[l].angular1 = 1.0f + 0.1f * av(l, 323);
        }
        for (uint32_t j = 0; j < 3; ++j)
            for (uint32_t l = 0; l < 32; ++l)
            {
                contacts[ACW_CONTACT_BASE[w] + j].point_separationW[l] =
                    make_float4(av(l, 330 + j * 8 + d), av(l, 331 + j * 8 + d),
                                av(l, 332 + j * 8 + d), 0.05f * av(l, 333 + j * 8 + d));
                contacts[ACW_CONTACT_BASE[w] + j].targetVel_maxImpulseW[l] =
                    make_float4(0.3f * av(l, 334 + j * 8), 0.3f * av(l, 335 + j * 8),
                                0.3f * av(l, 336 + j * 8), 1.5f + std::fabs(av(l, 337 + j * 8)));
            }
    }

    // P9.86c: the torsional / ignore()-contact inputs of the zero-tie lanes
    {
        const uint32_t cb = ACW_CONTACT_BASE[0];
        workUnits[1].mTorsionalFrictionData[0] = make_float2(0.7f, 0.0f);
        cdPool[1].contactData[0].restDistance = 0.0f;
        contacts[cb + 0].point_separationW[0].w = 0.02f;
        workUnits[1].mTorsionalFrictionData[24] = make_float2(0.7f, 0.0f);
        cdPool[1].contactData[24].restDistance = 0.0f;
        contacts[cb + 0].point_separationW[24].w = -0.0f;
        contacts[cb + 0].targetVel_maxImpulseW[1].w = 0.0f;
    }

    // engineered contact clusters for the friction-cache lanes (warp 0, desc 1)
    {
        const uint32_t cb = ACW_CONTACT_BASE[0];
        auto setPt = [&](uint32_t j, uint32_t lane, PxVec3 p, float sep) {
            contacts[cb + j].point_separationW[lane] = make_float4(p.x, p.y, p.z, sep);
        };
        const PxVec3 p0(0.31f, 0.42f, 0.53f);
        setPt(0, 2, p0, -0.01f);                                // clustered (keep-two)
        setPt(1, 2, p0 + PxVec3(0.03f, 0.0f, 0.01f), -0.02f);
        setPt(2, 2, p0 + PxVec3(0.0f, 0.04f, -0.02f), -0.015f);
        setPt(0, 4, p0, -0.01f);                                // spread (rebuild)
        setPt(1, 4, p0 + PxVec3(1.0f, 0.2f, -0.3f), -0.02f);
        setPt(0, 14, p0 + PxVec3(0.1f, 0.0f, 0.0f), -0.01f);    // keep-one-add-one
        setPt(1, 14, p0 + PxVec3(1.1f, 0.1f, 0.2f), -0.02f);
        setPt(0, 16, p0, -0.01f);                               // clustered: a gate-widened
        setPt(1, 16, p0 + PxVec3(0.03f, 0.01f, 0.0f), -0.02f);  // mutant KEEPS its illegally
        setPt(2, 16, p0 + PxVec3(0.0f, 0.03f, -0.02f), -0.01f); // inherited far anchors
    }

    // ---- previous friction data ----
    std::vector<PxgBlockFrictionPatch> prevPatches(ACW_NB_PREV);
    std::vector<PxgBlockFrictionAnchorPatch> prevAnchors(ACW_NB_PREV);
    std::memset(prevPatches.data(), 0xAB, prevPatches.size() * sizeof(prevPatches[0]));
    std::memset(prevAnchors.data(), 0xAB, prevAnchors.size() * sizeof(prevAnchors[0]));
    std::vector<PxU64> prevIdx(ACW_NB_PREV_FIDX);
    for (uint32_t e = 0; e < ACW_NB_PREV_FIDX; ++e)
        prevIdx[e] = (PxU64(e % ACW_NB_PREV) << 5) | ((e * 11u) % 32u);
    // make the poison prev columns deterministic rejects: broken != 0 already
    // holds for 0xAB poison; nothing else needed.

    // engineer the desc-1 lanes.  The frames of the two bodies are needed to
    // build matching prev data, so resolve them the same way the mirror will.
    std::vector<PxNodeIndex> nodesTmp(2);
    auto bodyFrameOf = [&](uint32_t w, uint32_t lane, uint32_t side) {
        const PxgBlockConstraintBatch& b = batches[ACW_BATCH_IDX[w] + ACW_RIGID_OFFSET];
        const PxNodeIndex node = side ? b.bodyBNodeIndex[lane] : b.bodyANodeIndex[lane];
        const uint32_t bi = node.isArticulation()
                          ? (side ? b.remappedBodyBIndex[lane] : b.remappedBodyAIndex[lane])
                          : (side ? b.bodyBIndex[lane] : b.bodyAIndex[lane]);
        PxgSolverExtBody2 eb;
        refCreateExtBody(node, bi, linkBases.data(), b2w.data(), resp.data(), mv.data(),
                         cfms.data(), links.data(), sbd.data(), txi.data(), frames.data(), eb);
        return PxAlignedTransform(eb.body2World);
    };

    uint32_t engineeredInherit1 = 0;
    {
        const uint32_t d = 1;   // ACW_DESC[0]
        struct Eng { uint32_t lane; uint32_t nAnchors; bool broken, badNormal, farAnchor;
                     uint32_t matchSlot; float spread; };
        const Eng engs[] = {
            { 2u, 2u, false, false, false, 0u, 1.2f },   // keep-two
            { 4u, 2u, false, false, false, 1u, 0.004f }, // rebuild (slot a=1!)
            { 6u, 2u, true,  false, false, 0u, 0.8f },   // broken reject
            { 8u, 2u, false, true,  false, 0u, 0.8f },   // normal reject
            { 10u, 2u, false, false, false, 0u, 0.8f },  // anchorCount>nb reject
            { 12u, 2u, false, false, true, 0u, 0.8f },   // pointsAreClose reject
            { 14u, 1u, false, false, false, 0u, 0.0f },  // one-anchor inherit
            { 16u, 2u, false, false, false, 0u, 0.8f },  // WOULD match, but perPoint gates it
        };
        for (const Eng& e : engs)
        {
            const PxAlignedTransform bf0 = bodyFrameOf(0, e.lane, 0);
            const PxAlignedTransform bf1 = bodyFrameOf(0, e.lane, 1);
            const float4 n4 = cdPool[d].normal_restitutionW[e.lane];
            PxVec3 n(n4.x, n4.y, n4.z);
            if (e.badNormal) n = -n;
            const PxVec3 b0n = bf0.rotateInv(n);
            const PxVec3 b1n = bf1.rotateInv(n);

            const uint32_t edge = ACW_EDGE_BASE[0] + e.lane;
            const uint32_t prevPatch = (e.lane / 2u) % ACW_NB_PREV;
            const uint32_t oldTid = (e.lane * 5u + 1u) % 32u;
            prevIdx[edge + e.matchSlot * ACW_TOTAL_PREV_EDGES] =
                (PxU64(prevPatch) << 5) | oldTid;
            if (e.matchSlot == 1u)   // slot 0 must REJECT: point it at a broken column
                prevIdx[edge] = (PxU64((prevPatch + 1u) % ACW_NB_PREV) << 5) | ((oldTid + 3u) % 32u);

            PxgBlockFrictionPatch& pp = prevPatches[prevPatch];
            PxgBlockFrictionAnchorPatch& pa = prevAnchors[prevPatch];
            pp.broken[oldTid] = e.broken ? 1u : 0u;
            pp.anchorCount[oldTid] = e.nAnchors;
            pp.body0Normal[oldTid] = make_float4(b0n.x, b0n.y, b0n.z, 0.f);
            pp.body1Normal[oldTid] = make_float4(b1n.x, b1n.y, b1n.z, 0.f);
            const PxVec3 w0(0.31f, 0.42f, 0.53f);
            const PxVec3 w1 = w0 + PxVec3(e.spread, e.spread * 0.4f, -e.spread * 0.2f);
            const PxVec3 a00 = bf0.transformInv(w0), a10 = bf1.transformInv(w0);
            const PxVec3 a01 = bf0.transformInv(w1), a11 = bf1.transformInv(w1);
            const PxVec3 off = e.farAnchor ? bf0.rotateInv(n) * 0.5f : PxVec3(0.f);
            pa.body0Anchors[0][oldTid] = make_float4(a00.x + off.x, a00.y + off.y,
                                                     a00.z + off.z, 0.f);
            pa.body1Anchors[0][oldTid] = make_float4(a10.x, a10.y, a10.z, 0.f);
            pa.body0Anchors[1][oldTid] = make_float4(a01.x, a01.y, a01.z, 0.f);
            pa.body1Anchors[1][oldTid] = make_float4(a11.x, a11.y, a11.z, 0.f);
            if (e.nAnchors == 1u) engineeredInherit1++;
        }
    }

    // ---- float params ----
    const float dtF = 0.35f, stepDtF = 0.07f;
    const float invDtF = 1.0f / dtF, stepInvDtF = 1.0f / stepDtF;
    const float bounceF = -0.2f, biasF = 0.7f;
    const float fricOffThresh = 0.04f, correlDist = 0.025f;

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
    for (uint32_t r = 0; r < ACW_NB_ARTI; ++r)
    {
        uint8_t* R = &rec[size_t(r) * ACW_REC_SIZE];
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
    PxgDevicePtr dWu = up(workUnits.data(), workUnits.size() * sizeof(workUnits[0]));
    PxgDevicePtr dCd = up(cdPool.data(), cdPool.size() * sizeof(cdPool[0]));
    PxgDevicePtr dContacts = up(contacts.data(), contacts.size() * sizeof(contacts[0]));
    PxgDevicePtr dPrevP = up(prevPatches.data(), prevPatches.size() * sizeof(prevPatches[0]));
    PxgDevicePtr dPrevA = up(prevAnchors.data(), prevAnchors.size() * sizeof(prevAnchors[0]));
    PxgDevicePtr dPrevIdx = up(prevIdx.data(), prevIdx.size() * 8);

    // outputs, poisoned
    std::vector<uint8_t> poisP(ACW_NB_PREP * sizeof(PxgBlockFrictionPatch), 0xAB);
    std::vector<uint8_t> poisA(ACW_NB_PREP * sizeof(PxgBlockFrictionAnchorPatch), 0xAB);
    std::vector<uint8_t> poisFI(ACW_NB_FIDX * 8, 0xAB);
    std::vector<uint8_t> poisCH(ACW_NB_PREP * sizeof(PxgTGSBlockSolverContactHeader), 0xAB);
    std::vector<uint8_t> poisFH(ACW_NB_PREP * sizeof(PxgTGSBlockSolverFrictionHeader), 0xAB);
    std::vector<uint8_t> poisCP(ACW_NB_OUT_CP * sizeof(PxgTGSBlockSolverContactPoint), 0xAB);
    std::vector<uint8_t> poisFR(ACW_NB_OUT_FR * sizeof(PxgTGSBlockSolverContactFriction), 0xAB);
    std::vector<uint8_t> poisRS(ACW_NB_RESP * sizeof(PxgArticulationBlockResponse), 0xAB);
    PxgDevicePtr dCurrP = up(poisP.data(), poisP.size());
    PxgDevicePtr dCurrA = up(poisA.data(), poisA.size());
    PxgDevicePtr dCurrFI = up(poisFI.data(), poisFI.size());
    PxgDevicePtr dCH = up(poisCH.data(), poisCH.size());
    PxgDevicePtr dFH = up(poisFH.data(), poisFH.size());
    PxgDevicePtr dCP = up(poisCP.data(), poisCP.size());
    PxgDevicePtr dFR = up(poisFR.data(), poisFR.size());
    PxgDevicePtr dRS = up(poisRS.data(), poisRS.size());

    // ---- the REAL engine descs ----
    PxgConstraintPrepareDesc prep;
    std::memset(&prep, 0xAB, sizeof(prep));
    prep.artiContactConstraintBatchIndices = reinterpret_cast<PxU32*>(dBatchIdx);
    prep.solverBodyDataPool = reinterpret_cast<PxgSolverBodyData*>(dSbd);
    prep.solverBodyTxIDataPool = reinterpret_cast<PxgSolverTxIData*>(dTxi);
    prep.blockWorkUnit = reinterpret_cast<PxgBlockWorkUnit*>(dWu);
    prep.blockCurrentFrictionIndices = reinterpret_cast<PxgBlockFrictionIndex*>(dCurrFI);
    prep.blockPreviousFrictionIndices = reinterpret_cast<PxgBlockFrictionIndex*>(dPrevIdx);
    prep.blockContactCurrentPrepPool = reinterpret_cast<PxgBlockContactData*>(dCd);
    prep.blockContactPoints = reinterpret_cast<PxgBlockContactPoint*>(dContacts);
    prep.blockCurrentAnchorPatches = reinterpret_cast<PxgBlockFrictionAnchorPatch*>(dCurrA);
    prep.blockPreviousAnchorPatches = reinterpret_cast<PxgBlockFrictionAnchorPatch*>(dPrevA);
    prep.body2WorldPool = reinterpret_cast<PxAlignedTransform*>(dFrames);
    prep.numArtiContactBatches = 2;
    prep.numArtiStaticContactBatches = 1;
    prep.numArtiSelfContactBatches = 1;
    prep.numBatches = ACW_RIGID_OFFSET;
    prep.bounceThresholdF32 = bounceF;
    prep.frictionOffsetThreshold = fricOffThresh;
    prep.correlationDistance = correlDist;
    prep.biasCoefficients.rigidContact = biasF;   // 5.12: the per-kind collection
    prep.biasCoefficients.joint = 0.55f;
    prep.biasCoefficients.articulation = 0.45f;
    prep.totalPreviousEdges = ACW_TOTAL_PREV_EDGES;
    prep.totalCurrentEdges = ACW_TOTAL_CURR_EDGES;

    PxgSolverSharedDesc<IterativeSolveDataTGS> shared;
    std::memset(&shared, 0xAB, sizeof(shared));
    shared.blockCurrentFrictionPatches = reinterpret_cast<PxgBlockFrictionPatch*>(dCurrP);
    shared.blockPreviousFrictionPatches = reinterpret_cast<PxgBlockFrictionPatch*>(dPrevP);
    shared.articulations = reinterpret_cast<PxgArticulation*>(dRec);
    shared.dt = dtF;
    shared.stepDt = stepDtF;
    shared.invDtF32 = invDtF;
    shared.stepInvDtF32 = stepInvDtF;
    shared.iterativeData.blockConstraintBatch =
        reinterpret_cast<PxgBlockConstraintBatch*>(dBatches);
    shared.iterativeData.blockContactHeaders =
        reinterpret_cast<PxgTGSBlockSolverContactHeader*>(dCH);
    shared.iterativeData.blockFrictionHeaders =
        reinterpret_cast<PxgTGSBlockSolverFrictionHeader*>(dFH);
    shared.iterativeData.blockContactPoints =
        reinterpret_cast<PxgTGSBlockSolverContactPoint*>(dCP);
    shared.iterativeData.blockFrictions =
        reinterpret_cast<PxgTGSBlockSolverContactFriction*>(dFR);
    shared.iterativeData.artiResponse = reinterpret_cast<PxgArticulationBlockResponse*>(dRS);

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
    std::vector<PxgBlockFrictionPatch> wantP(ACW_NB_PREP);
    std::vector<PxgBlockFrictionAnchorPatch> wantA(ACW_NB_PREP);
    std::vector<PxU64> wantFI(ACW_NB_FIDX);
    std::vector<PxgTGSBlockSolverContactHeader> wantCH(ACW_NB_PREP);
    std::vector<PxgTGSBlockSolverFrictionHeader> wantFH(ACW_NB_PREP);
    std::vector<PxgTGSBlockSolverContactPoint> wantCP(ACW_NB_OUT_CP);
    std::vector<PxgTGSBlockSolverContactFriction> wantFR(ACW_NB_OUT_FR);
    std::vector<PxgArticulationBlockResponse> wantRS(ACW_NB_RESP);
    std::memset(wantP.data(), 0xAB, wantP.size() * sizeof(wantP[0]));
    std::memset(wantA.data(), 0xAB, wantA.size() * sizeof(wantA[0]));
    std::memset(wantFI.data(), 0xAB, wantFI.size() * 8);
    std::memset(wantCH.data(), 0xAB, wantCH.size() * sizeof(wantCH[0]));
    std::memset(wantFH.data(), 0xAB, wantFH.size() * sizeof(wantFH[0]));
    std::memset(wantCP.data(), 0xAB, wantCP.size() * sizeof(wantCP[0]));
    std::memset(wantFR.data(), 0xAB, wantFR.size() * sizeof(wantFR[0]));
    std::memset(wantRS.data(), 0xAB, wantRS.size() * sizeof(wantRS[0]));

    uint32_t inheritedCount = 0, keptTwoCount = 0, inherited1Count = 0;
    bool laneInherited[32] = {};
    for (uint32_t w = 0; w < ACW_LIVE_WARPS; ++w)
    {
        const PxgBlockConstraintBatch& b = batches[ACW_BATCH_IDX[w] + ACW_RIGID_OFFSET];
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

            const PxgBlockWorkUnit& wu = workUnits[d];
            const uint32_t edge = wu.mEdgeIndex[lane];
            const uint32_t fi = edge + ACW_TOTAL_CURR_EDGES * wu.mPatchIndex[lane];
            wantFI[fi] = (PxU64(d) << 5) | lane;

            const PxgMaterialContactData& data = cdPool[d].contactData[lane];
            const uint32_t nb = data.mNumContacts;
            const uint32_t flags = data.mSolverFlags;
            // 5.12 flags: a target-velocity lane raises eHAS_TARGET_VELOCITY (2) and
            // eDISABLE_STRONG_FRICTION (4), as the engine does; eDISABLE_FRICTION is 8
            const bool perPoint = (flags & 2u) != 0u;
            const bool disableStrong = (flags & 4u) != 0u;
            const bool disable = (flags & 8u) != 0u;

            const float4 n4 = cdPool[d].normal_restitutionW[lane];
            const PxVec3 normal(n4.x, n4.y, n4.z);

            const PxAlignedTransform bf0(b0.body2World);
            const PxAlignedTransform bf1(b1.body2World);

            refCorrelatePatches(wantP[d], lane, nb, normal, bf0, bf1);
            wantP[d].broken[lane] = 0;
            float patchExtents = 0.f;

            bool inh = false, kept = false;
            if (!(disableStrong || disable))
                refGetFrictionPatches(wantP[d], wantA[d], lane, prevIdx.data(), true, edge,
                                      prevPatches.data(), prevAnchors.data(),
                                      data.prevFrictionPatchCount, bf0, bf1, correlDist,
                                      ACW_TOTAL_PREV_EDGES, patchExtents, nb, &inh);
            if (!disable)
                refGrowPatches(wantP[d], wantA[d], lane, contacts.data(), ACW_CONTACT_BASE[w],
                               nb, bf0, bf1, fricOffThresh + data.restDistance, patchExtents,
                               &kept);
            if (inh)
            {
                inheritedCount++;
                if (w == 0) laneInherited[lane] = true;
            }
            if (kept) keptTwoCount++;
            if (inh && w == 0 && lane == 14u) inherited1Count++;

            const float offsetSlop = PxMax(b0.offsetSlop, b1.offsetSlop);
            refSetupExtContactsTGS(cdPool[d], &contacts[ACW_CONTACT_BASE[w]], nb, wantP[d],
                                   wantA[d], b0, b1, stepInvDtF, stepDtF, invDtF, bounceF, biasF,
                                   lane, wu.mWriteback[lane], wantCH[d], wantFH[d],
                                   &wantCP[b.startConstraintIndex], &wantFR[b.startFrictionIndex],
                                   &wantRS[b.mArticulationResponseIndex], data, offsetSlop,
                                   wu.mTorsionalFrictionData[lane], dtF);

            wantP[d].patchIndex[lane] = wu.mFrictionPatchIndex[lane];
            if (wantP[d].anchorCount[lane] >= 1)
            {
                const float4 a0 = wantA[d].body0Anchors[0][lane];
                const PxVec3 wp = b0.body2World.transform(PxVec3(a0.x, a0.y, a0.z));
                wantP[d].anchorPoints[0][lane] = make_float4(wp.x, wp.y, wp.z, 0.f);
            }
            if (wantP[d].anchorCount[lane] == 2)
            {
                const float4 a1 = wantA[d].body0Anchors[1][lane];
                const PxVec3 wp = b0.body2World.transform(PxVec3(a1.x, a1.y, a1.z));
                wantP[d].anchorPoints[1][lane] = make_float4(wp.x, wp.y, wp.z, 0.f);
            }
        }
    }

    // engineered coverage, asserted
    PXG_CHECK_MSG(laneInherited[2] && laneInherited[4] && laneInherited[14], "inherit lanes");
    PXG_CHECK_MSG(!laneInherited[6] && !laneInherited[8] && !laneInherited[10]
                      && !laneInherited[12] && !laneInherited[16], "reject lanes");
    PXG_CHECK_MSG(keptTwoCount >= 1, "keep-two coverage");
    PXG_CHECK_MSG(inherited1Count == engineeredInherit1, "one-anchor inherit coverage");
    PXG_CHECK_MSG(inheritedCount >= 3, "inherit coverage");

    // ---- P9.86c: zero-tie coverage, asserted through the mirror's own image ----
    auto bitsOf = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; };
    {
        const uint32_t zsMaxImpTie = (bitsOf(wantCP[0].maxImpulse[21]) == 0x00000000u ? 1u : 0u)
                                   + (bitsOf(wantCP[1].maxImpulse[21]) == 0x00000000u ? 1u : 0u);
        const bool zsMaxImpIgnore = bitsOf(wantCP[0].maxImpulse[1]) == 0x00000000u
                                 && bitsOf(wantCP[1].maxImpulse[1]) == 0x80000000u;
        const bool zsPenNeg0 = wantP[1].anchorCount[24] == 1u
                            && bitsOf(wantCP[0].separation[24]) == 0x80000000u;
        const uint32_t zsTorsNeg0 =
            (wantP[1].anchorCount[0] == 1u
             && bitsOf(wantFH[1].torsionalFrictionScale[0]) == 0x80000000u ? 1u : 0u)
            + (wantP[1].anchorCount[24] == 1u
               && bitsOf(wantFH[1].torsionalFrictionScale[24]) == 0x80000000u ? 1u : 0u);
        const bool zsCfmNeg0 = bitsOf(wantCH[1].cfm[1]) == 0x80000000u;
        std::printf("  acw zero ties: maxImpTie %u  ignore %d  penNeg0 %d  torsNeg0 %u  cfmNeg0 %d\n",
                    zsMaxImpTie, int(zsMaxImpIgnore), int(zsPenNeg0), zsTorsNeg0, int(zsCfmNeg0));
        PXG_CHECK_MSG(bitsOf(sbd[16].maxImpulse) == 0x80000000u
                          && bitsOf(sbd[17].maxImpulse) == 0x00000000u, "zero-tie bodies");
        PXG_CHECK_MSG(zsMaxImpTie == 2u, "lane 21: PxMin(-0,+0) body maxImpulse tie");
        PXG_CHECK_MSG(zsMaxImpIgnore, "lane 1: PxMin(-0 body, +0 ignore()) contact tie");
        PXG_CHECK_MSG(zsPenNeg0, "lane 24: -0 penetration single anchor");
        PXG_CHECK_MSG(zsTorsNeg0 == 2u, "lanes 0/24: -0 torsionalFrictionScale");
        PXG_CHECK_MSG(zsCfmNeg0, "lane 1: PxMax(+0,-0) cfm tie");
    }

    // P9.86c: the mutant twins -- each pre-P9.86 FMin/FMax spelling, re-run
    // over the directed lanes only, must change the expected image
    auto twinDiffers = [&](int variant) {
        std::vector<PxgTGSBlockSolverContactHeader> tCH(wantCH);
        std::vector<PxgTGSBlockSolverFrictionHeader> tFH(wantFH);
        std::vector<PxgTGSBlockSolverContactPoint> tCP(wantCP);
        std::vector<PxgTGSBlockSolverContactFriction> tFR(wantFR);
        std::vector<PxgArticulationBlockResponse> tRS(wantRS);
        const PxgBlockConstraintBatch& b = batches[ACW_BATCH_IDX[0] + ACW_RIGID_OFFSET];
        const uint32_t d = b.mConstraintBatchIndex;
        gAcpZeroVariant = variant;
        for (uint32_t lane : ACW_ZS_LANES)
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
            const PxgBlockWorkUnit& wu = workUnits[d];
            const PxgMaterialContactData& data = cdPool[d].contactData[lane];
            const float offsetSlop = PxMax(b0.offsetSlop, b1.offsetSlop);
            refSetupExtContactsTGS(cdPool[d], &contacts[ACW_CONTACT_BASE[0]], data.mNumContacts,
                                   wantP[d], wantA[d], b0, b1, stepInvDtF, stepDtF, invDtF,
                                   bounceF, biasF, lane, wu.mWriteback[lane], tCH[d], tFH[d],
                                   &tCP[b.startConstraintIndex], &tFR[b.startFrictionIndex],
                                   &tRS[b.mArticulationResponseIndex], data, offsetSlop,
                                   wu.mTorsionalFrictionData[lane], dtF);
        }
        gAcpZeroVariant = 0;
        return std::memcmp(tCH.data(), wantCH.data(), tCH.size() * sizeof(tCH[0])) != 0
            || std::memcmp(tFH.data(), wantFH.data(), tFH.size() * sizeof(tFH[0])) != 0
            || std::memcmp(tCP.data(), wantCP.data(), tCP.size() * sizeof(tCP[0])) != 0
            || std::memcmp(tFR.data(), wantFR.data(), tFR.size() * sizeof(tFR[0])) != 0
            || std::memcmp(tRS.data(), wantRS.data(), tRS.size() * sizeof(tRS[0])) != 0;
    };
    PXG_CHECK_MSG(!twinDiffers(0), "the CUDA twin must reproduce the image");
    for (int v : { 1, 2, 4, 8, 16 })
        PXG_CHECK_MSG(twinDiffers(v),
                      "mutant twin %d gives the same image -- the directed lanes do not observe it", v);

    // ---- launch: 3 blocks x (32,2,1) = 6 warps, 4 live ----
    struct Params { uint64_t prepDesc, sharedDesc; } prm = { uint64_t(dPrep), uint64_t(dShared) };
    static_assert(sizeof(Params) == 16, "params blob");
    PXG_CHECK_MSG(shim.launchKernel(0, 3, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch artiTGSContactConstraintBlockPrepareLaunch");
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
                    std::printf("MM acw %s word %zu (lane %zu)  ref %08x  gpu %08x\n",
                                tag, k, k % 32, wv[k], got[k]);
                ++bad;
            }
    };
    cmp(dCurrP, wantP.data(), wantP.size() * sizeof(wantP[0]), "frictionPatches");
    cmp(dCurrA, wantA.data(), wantA.size() * sizeof(wantA[0]), "anchors");
    cmp(dCurrFI, wantFI.data(), wantFI.size() * 8, "frictionIndices");
    cmp(dCH, wantCH.data(), wantCH.size() * sizeof(wantCH[0]), "contactHeaders");
    cmp(dFH, wantFH.data(), wantFH.size() * sizeof(wantFH[0]), "frictionHeaders");
    cmp(dCP, wantCP.data(), wantCP.size() * sizeof(wantCP[0]), "contactPoints");
    cmp(dFR, wantFR.data(), wantFR.size() * sizeof(wantFR[0]), "frictions");
    cmp(dRS, wantRS.data(), wantRS.size() * sizeof(wantRS[0]), "responses");
    PXG_CHECK_MSG(bad == 0, "%zu mismatches", bad);

    for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
}
