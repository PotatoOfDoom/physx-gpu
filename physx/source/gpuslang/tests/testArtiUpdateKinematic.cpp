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


// P8.117: the `artiUpdateKinematic` engine entry
// (articulationDirectGpuApi.cu:295, body updateKinematicInternal<> :63-292).
// The mirror is the CUDA's structure sequentially over engine primitives
// (PxQuat(angle, axis) / getNormalized / getConjugate / operator* / rotate,
// PxVec3::normalize, FeatherstoneArticulation::translateSpatialVector,
// PxTransform::transform/transformInv, PxMat33(q), the updateCacheAndBound
// geometry switch, the PxNodeIndex binary search + backward walk).
//
// Pins: the NULL-gpuIndices round (walk ALL articulations by index), the
// eNEEDS_KINEMATIC_UPDATE gate that is CLEARED even for articulations the
// !zeroSimOutput && !positionsDirty early-out skips, the 16-vs-12-byte
// temp strides, the NEGATED revolute angle + shortest-angle flip, the
// spherical exp map with the < 1e-10 identity path, the conjugate in the
// phase-B chain, the zeroSimOutput velocity update + sim-output zeroing,
// and the phase-D transform-cache/bounds refresh across all five geometry
// paths of the lifted pxgSlangCacheBound module (sphere / capsule / box /
// real convex hull incl. a non-identity mesh scale / fast path).
//
// TRANSCENDENTALS: revolute/spherical joint rotations go through sin/cos,
// which are ULP-close but not bit-equal between llvmpipe and host libm
// (the P8.94 measurement).  Records are classified: trig-free records
// (fix / prismatic joints / all-zero spherical positions) compare BIT
// EXACT everywhere; records with a live transcendental compare their
// pose-derived float outputs with the testArtiStepKernel tolerance
// (ULP-close OR absolutely-close, loudly counted).  Integer state (dirty
// flags, cache flags) and the zeroed sim outputs are bit-exact always.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"
#include "pxgArtiRefHelpers.h"

#include "foundation/PxVec3.h"
#include "foundation/PxQuat.h"
#include "foundation/PxTransform.h"
#include "foundation/PxMat33.h"
#include "foundation/PxBounds3.h"
#include "PxgShapeSim.h"
#include "PxNodeIndex.h"
#include "DyFeatherstoneArticulation.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::PxMat33;
using physx::PxQuat;
using physx::PxTransform;
using physx::PxVec3;
using physx::Cm::UnAlignedSpatialVector;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

static_assert(sizeof(physx::PxgShapeSim) == 72, "PxgShapeSim mirror stride");
static_assert(offsetof(physx::PxgShapeSim, mBodySimIndex) == 56, "node index @56");
static_assert(offsetof(physx::PxgShapeSim, mHullDataIndex) == 64, "hull index @64");
static_assert(offsetof(physx::PxgArticulation, jointAccelerations) == 128, "..JOINT_ACCELERATIONS");
static_assert(offsetof(physx::PxgArticulation, linkBody2Actors) == 200, "..LINK_BODY2ACTORS");
static_assert(offsetof(physx::PxgArticulation, tempParentToChilds) == 288, "..TEMP_PARENT_TO_CHILDS");
static_assert(offsetof(physx::PxgArticulation, tempRs) == 296, "..TEMP_RS");
static_assert(offsetof(physx::PxgArticulationData, bodySimIndex) == 4, "..DATA_BODY_SIM_INDEX");
static_assert(physx::Dy::ArticulationDirtyFlag::eNEEDS_KINEMATIC_UPDATE == (1u << 20), "dirty bit");

static std::vector<uint8_t> artiUkLoader(const char* file) { return loadShaderBlob(file); }

const uint32_t UK_NB_LIVE = 12;      // articulations the launch names
const uint32_t UK_NB_RECORDS = 16;   // + 4 guard records past the launch: eNEEDS
                                     // set, valid data, NEVER in a job -- a dropped
                                     // i<nb guard clears their dirty words visibly
const uint32_t UK_MAX_LINKS = 6;
const uint32_t UK_MAX_DOFS = 12;
const uint32_t UK_RECORD_SIZE = 544;
const uint32_t UK_JC_SIZE = 544;
const uint32_t UK_NB_SHAPES = 18;   // sorted node/shape table entries
const uint32_t UK_GRID = 4;         // 8 warps: ceil(12/8)=2 iters, slots 12..15 DEAD

// joint classes per record: 0 all-fix, 1 all-prismatic, 2 revolute mix,
// 3 spherical (record 7: all joint positions zero -> the identity path)
static uint32_t ukCls(uint32_t r) { return r % 4u; }
static bool ukTrig(uint32_t r) { return (ukCls(r) == 2u || ukCls(r) == 3u) && r != 7u; }
static uint32_t ukNumLinks(uint32_t r) { return r == 5u ? 1u : 2u + (r % 5u); }
static uint32_t ukJointType(uint32_t r, uint32_t link)
{
    switch (ukCls(r))
    {
    case 0: return 0u;                              // eFIX
    case 1: return 1u;                              // ePRISMATIC
    case 2: return (link & 1u) ? 3u : 2u;           // eREVOLUTE / eREVOLUTE_UNWRAPPED
    default: return 4u;                             // eSPHERICAL
    }
}
static uint32_t ukDofs(uint32_t r, uint32_t link)
{
    const uint32_t t = ukJointType(r, link);
    if (t == 0u) return 0u;
    if (t == 4u) return 1u + ((r + link) % 3u);
    return 1u;
}
static uint32_t ukParent(uint32_t link)
{
    if (link <= 1u) return 0u;
    return (link >= 3u && (link % 3u) == 0u) ? 1u : (link - 1u);
}
static uint32_t ukJointOffset(uint32_t r, uint32_t link)
{
    uint32_t off = 0;
    for (uint32_t j = 1; j < link; ++j)
        off += ukDofs(r, j);
    return off;
}
static uint32_t ukDofCount(uint32_t r) { return ukJointOffset(r, ukNumLinks(r)); }
static uint32_t ukBodySim(uint32_t r) { return 40u + r * 5u; }
// dirty state: r%6==3 -> no eNEEDS (untouched); r%3==1 -> velocities-only
// (skipped in the !zeroSim round, but the eNEEDS bit still clears)
static bool ukNeeds(uint32_t r) { return (r % 6u) != 3u; }
static bool ukPosDirty(uint32_t r) { return (r % 3u) != 1u; }
static uint32_t ukDirtyWord(uint32_t r)
{
    uint32_t w = 0x40000u | (r << 24);   // unrelated bits that must SURVIVE
    if (ukNeeds(r)) w |= (1u << 20);
    w |= ukPosDirty(r) ? 2u : 4u;        // eDIRTY_POSITIONS : eDIRTY_VELOCITIES
    return w;
}
static float ukf(uint32_t i, uint32_t k)
{
    return (float((i * 83u + k * 59u) % 523u) + 1.0f) * 0.015625f - 3.75f;
}
static PxQuat ukQuat(uint32_t i, uint32_t k)
{
    PxQuat q(0.13f + 0.03f * float((i + k) % 7u), -0.27f + 0.05f * float((i * 3u + k) % 5u),
             0.31f - 0.04f * float((i + 2u * k) % 6u), 0.9f);
    return q.getNormalized();
}
static float ukJPos(uint32_t r, uint32_t gdof)
{
    if (r == 7u) return 0.0f;                     // the spherical identity path
    if (ukCls(r) == 2u && (r + gdof) % 3u == 0u)
        return 3.6f + 0.13f * float(gdof);        // > pi: the w<0 flip fires
    if (r == 11u) return 2.4f + 0.17f * float(gdof);  // spherical |ang| > pi
    return 0.35f + 0.11f * float((r + gdof) % 7u);
}

// shape table: entry s -> (record, link, geometry). 0xFF record = a plain
// rigid entry (arti bit clear) the walk must never touch; type 9 = an entry
// whose shapeIndex is 0xFFFFFFFF (skipped but still decremented past).
struct UkShapeDef { uint32_t rec, link, geom; };
static UkShapeDef ukShape(uint32_t s)
{
    // geometries: 0 sphere, 2 capsule, 3 box, 5 convexmesh (real), 6 convex
    // fast path (mHullDataIndex invalid), 9 invalid shapeIndex entry
    static const UkShapeDef defs[UK_NB_SHAPES] = {
        { 0u, 1u, 0u },  { 0u, 1u, 3u },   // two shapes on one link
        { 2u, 2u, 2u },  { 2u, 0u, 5u },
        { 4u, 3u, 3u },  { 4u, 3u, 9u },   // invalid entry between valid ones
        { 5u, 0u, 5u },                    // single-link articulation, real hull
        { 6u, 1u, 6u },  { 6u, 2u, 0u },
        { 8u, 4u, 3u },  { 8u, 1u, 2u },
        { 0xFFu, 0u, 0u }, { 0xFFu, 0u, 3u },   // rigid entries
        { 10u, 1u, 0u }, { 10u, 0u, 6u },
        { 1u, 1u, 3u },                    // velocities-only record: untouched in !zeroSim
        { 3u, 1u, 0u },                    // no-eNEEDS record: never touched
        { 9u, 2u, 2u },
    };
    return defs[s];
}
static physx::PxNodeIndex ukNode(const UkShapeDef& d)
{
    if (d.rec == 0xFFu)
        return physx::PxNodeIndex(700u + d.link);            // rigid: mLinkID = 0
    return physx::PxNodeIndex(ukBodySim(d.rec), d.link);     // arti: (link<<1)|1
}

struct UkArena
{
    std::vector<uint8_t> rec, jointCores, jointData;
    std::vector<uint32_t> parents;
    std::vector<float> xforms, xformsA, motion, jpos, jvel, relq;
    std::vector<float> mvel, macc, incoming, jacc, tempQ, tempR;
    std::vector<physx::PxgShapeSim> shapeSims;
    std::vector<uint32_t> shapeOrder;        // sorted entry -> shape def index
    std::vector<physx::PxNodeIndex> nodeIndices;
    std::vector<uint32_t> shapeIndices;
    std::vector<uint8_t> convexShapes;       // PxgShape images, 48 B
    std::vector<uint32_t> convexBlob;
    std::vector<float> cache;                // PxsCachedTransform, 8 floats
    std::vector<float> bounds;               // PxBounds3, 6 floats
    std::vector<uint8_t> scDesc;
};

static void ukSeed(UkArena& A, uint32_t salt)
{
    A.rec.assign(size_t(UK_NB_RECORDS) * UK_RECORD_SIZE, 0xAB);
    A.jointCores.assign(size_t(UK_NB_RECORDS) * UK_MAX_LINKS * UK_JC_SIZE, 0xAB);
    A.jointData.assign(size_t(UK_NB_RECORDS) * UK_MAX_LINKS * 8, 0xAB);
    A.parents.assign(size_t(UK_NB_RECORDS) * UK_MAX_LINKS, 0xCDCDCDCDu);
    A.xforms.assign(size_t(UK_NB_RECORDS) * UK_MAX_LINKS * 7, -88.f);
    A.xformsA.assign(size_t(UK_NB_RECORDS) * UK_MAX_LINKS * 7, -87.f);
    A.motion.assign(size_t(UK_NB_RECORDS) * UK_MAX_LINKS * 20, -77.f);
    A.jpos.assign(size_t(UK_NB_RECORDS) * UK_MAX_DOFS, -66.f);
    A.jvel.assign(size_t(UK_NB_RECORDS) * UK_MAX_DOFS, -55.f);
    A.relq.assign(size_t(UK_NB_RECORDS) * UK_MAX_LINKS * 4, -44.f);
    A.mvel.assign(size_t(UK_NB_RECORDS) * UK_MAX_LINKS * 6, 0.f);
    A.macc.assign(size_t(UK_NB_RECORDS) * UK_MAX_LINKS * 6, 0.f);
    A.incoming.assign(size_t(UK_NB_RECORDS) * UK_MAX_LINKS * 6, 0.f);
    A.jacc.assign(size_t(UK_NB_RECORDS) * UK_MAX_DOFS, 0.f);
    A.tempQ.assign(size_t(UK_NB_RECORDS) * UK_MAX_LINKS * 4, 0.f);
    A.tempR.assign(size_t(UK_NB_RECORDS) * UK_MAX_LINKS * 3, 0.f);
    std::memset(A.macc.data(), 0xAB, A.macc.size() * 4);
    std::memset(A.incoming.data(), 0xAB, A.incoming.size() * 4);
    std::memset(A.jacc.data(), 0xAB, A.jacc.size() * 4);
    std::memset(A.tempQ.data(), 0xAB, A.tempQ.size() * 4);
    std::memset(A.tempR.data(), 0xAB, A.tempR.size() * 4);

    for (uint32_t r = 0; r < UK_NB_RECORDS; ++r)
    {
        const uint32_t nl = ukNumLinks(r);
        uint8_t* R = A.rec.data() + size_t(r) * UK_RECORD_SIZE;
        const uint32_t idx = r, bsi = ukBodySim(r), dirty = ukDirtyWord(r);
        std::memcpy(R + 0, &idx, 4);
        std::memcpy(R + 4, &bsi, 4);
        std::memcpy(R + 8, &dirty, 4);
        const uint16_t nl16 = uint16_t(nl), nd16 = uint16_t(ukDofCount(r));
        std::memcpy(R + 16, &nl16, 2);
        std::memcpy(R + 18, &nd16, 2);

        for (uint32_t d = 0; d < ukDofCount(r); ++d)
        {
            A.jpos[size_t(r) * UK_MAX_DOFS + d] = ukJPos(r, d);
            A.jvel[size_t(r) * UK_MAX_DOFS + d] = ukf(r, 900u + d + salt);
        }

        for (uint32_t l = 0; l < nl; ++l)
        {
            uint8_t* jc = A.jointCores.data() + (size_t(r) * UK_MAX_LINKS + l) * UK_JC_SIZE;
            const PxQuat pq = ukQuat(r, 30u + l), cq = ukQuat(r, 60u + l);
            float qq[7] = { pq.x, pq.y, pq.z, pq.w,
                            ukf(r, 10u + l * 3u), ukf(r, 11u + l * 3u), ukf(r, 12u + l * 3u) };
            std::memcpy(jc + 0, qq, 28);   // parentPose
            float cc[7] = { cq.x, cq.y, cq.z, cq.w,
                            ukf(r, 20u + l * 3u), ukf(r, 21u + l * 3u), ukf(r, 22u + l * 3u) };
            std::memcpy(jc + 28, cc, 28);  // childPose
            jc[539] = uint8_t(ukJointType(r, l));

            uint8_t* jd = A.jointData.data() + (size_t(r) * UK_MAX_LINKS + l) * 8;
            const uint32_t jo = (l == 0) ? 0u : ukJointOffset(r, l);
            std::memcpy(jd, &jo, 4);
            jd[4] = uint8_t((l == 0) ? 0u : ukDofs(r, l));
            A.parents[size_t(r) * UK_MAX_LINKS + l] = (l == 0) ? 0u : ukParent(l);

            float* xf = &A.xforms[(size_t(r) * UK_MAX_LINKS + l) * 7];
            const PxQuat q = ukQuat(r, l);
            xf[0] = q.x; xf[1] = q.y; xf[2] = q.z; xf[3] = q.w;
            xf[4] = ukf(r, 100u + l * 3u); xf[5] = ukf(r, 101u + l * 3u);
            xf[6] = ukf(r, 102u + l * 3u);
            float* xa = &A.xformsA[(size_t(r) * UK_MAX_LINKS + l) * 7];
            const PxQuat qa = ukQuat(r, 90u + l);
            xa[0] = qa.x; xa[1] = qa.y; xa[2] = qa.z; xa[3] = qa.w;
            xa[4] = ukf(r, 130u + l * 3u); xa[5] = ukf(r, 131u + l * 3u);
            xa[6] = ukf(r, 132u + l * 3u);

            // motion matrix: unit-ish axes so PxQuat(angle, axis) sees a unit
            // axis, per its contract; no exact-zero components (P8.87 fold)
            for (uint32_t d = 0; d < 3; ++d)
            {
                PxVec3 ax(0.31f + 0.17f * float((r + l + d) % 5u),
                          0.42f - 0.13f * float((r * 2u + l + d) % 4u),
                          0.53f + 0.07f * float((r + 2u * l + d) % 6u));
                ax.normalize();
                float* mm = &A.motion[(size_t(r) * UK_MAX_LINKS + l) * 20 + d * 6];
                mm[0] = ax.x; mm[1] = ax.y; mm[2] = ax.z;                 // top
                PxVec3 bx(ukf(r, 300u + l * 20u + d * 6u), ukf(r, 301u + l * 20u + d * 6u),
                          ukf(r, 302u + l * 20u + d * 6u));
                mm[3] = bx.x; mm[4] = bx.y; mm[5] = bx.z;                 // bottom
            }
            const PxQuat rq = ukQuat(r, 120u + l);
            float* rr = &A.relq[(size_t(r) * UK_MAX_LINKS + l) * 4];
            rr[0] = rq.x; rr[1] = rq.y; rr[2] = rq.z; rr[3] = rq.w;

            for (uint32_t c = 0; c < 6; ++c)
                A.mvel[(size_t(r) * UK_MAX_LINKS + l) * 6 + c] = ukf(r, 700u + l * 6u + c + salt);
        }
    }

    // ---- the shape tables ----
    A.shapeSims.resize(UK_NB_SHAPES);
    std::memset(A.shapeSims.data(), 0xAB, A.shapeSims.size() * sizeof(physx::PxgShapeSim));
    A.convexShapes.assign(2 * 48, 0);
    // convex blob: 12 header words + 5 float4 vertices; count in bits [8,16) of word 4
    A.convexBlob.assign(12 + 5 * 4, 0);
    A.convexBlob[4] = (5u << 8);
    for (uint32_t v = 0; v < 5; ++v)
    {
        float vx = ukf(77u, v * 3u), vy = ukf(77u, v * 3u + 1u), vz = ukf(77u, v * 3u + 2u);
        std::memcpy(&A.convexBlob[12 + v * 4 + 0], &vx, 4);
        std::memcpy(&A.convexBlob[12 + v * 4 + 1], &vy, 4);
        std::memcpy(&A.convexBlob[12 + v * 4 + 2], &vz, 4);
    }
    // PxgShape 0: identity scale; PxgShape 1: non-identity scale + rotation
    {
        float ident[7] = { 1.f, 1.f, 1.f, 0.f, 0.f, 0.f, 1.f };
        std::memcpy(A.convexShapes.data() + 0, ident, 28);
        const PxQuat sr = ukQuat(3u, 9u);
        float scaled[7] = { 1.25f, 0.75f, 1.5f, sr.x, sr.y, sr.z, sr.w };
        std::memcpy(A.convexShapes.data() + 48, scaled, 28);
    }

    for (uint32_t s = 0; s < UK_NB_SHAPES; ++s)
    {
        const UkShapeDef d = ukShape(s);
        physx::PxgShapeSim& ss = A.shapeSims[s];
        ss.mTransform = PxTransform(PxVec3(ukf(s, 500u), ukf(s, 501u), ukf(s, 502u)),
                                    ukQuat(s, 77u));
        ss.mLocalBounds = physx::PxBounds3(
            PxVec3(-0.5f - 0.03125f * float(s), -0.75f, -0.25f - 0.0625f * float(s % 3u)),
            PxVec3(1.0f + 0.03125f * float(s), 0.4f + 0.0625f * float(s % 4u), 0.6f));
        ss.mBodySimIndex = ukNode(d);
        ss.mHullDataIndex = (d.geom == 5u) ? (s & 1u) : 0xFFFFFFFFu;
        ss.mShapeFlags = 0;
        ss.mShapeType = uint16_t(d.geom == 6u ? 5u : (d.geom == 9u ? 0u : d.geom));
    }
    // sort entries by node index (u64 ascending), ties keep def order
    A.shapeOrder.resize(UK_NB_SHAPES);
    for (uint32_t s = 0; s < UK_NB_SHAPES; ++s) A.shapeOrder[s] = s;
    std::stable_sort(A.shapeOrder.begin(), A.shapeOrder.end(), [&](uint32_t a, uint32_t b) {
        return ukNode(ukShape(a)).getInd() < ukNode(ukShape(b)).getInd();
    });
    A.nodeIndices.resize(UK_NB_SHAPES);
    A.shapeIndices.resize(UK_NB_SHAPES);
    for (uint32_t k = 0; k < UK_NB_SHAPES; ++k)
    {
        const uint32_t s = A.shapeOrder[k];
        A.nodeIndices[k] = ukNode(ukShape(s));
        A.shapeIndices[k] = (ukShape(s).geom == 9u) ? 0xFFFFFFFFu : s;
    }

    A.cache.assign(size_t(UK_NB_SHAPES) * 8, 0.f);
    A.bounds.assign(size_t(UK_NB_SHAPES) * 6, 0.f);
    std::memset(A.cache.data(), 0xAB, A.cache.size() * 4);
    std::memset(A.bounds.data(), 0xAB, A.bounds.size() * 4);

    A.scDesc.assign(96, 0xAB);
    const uint32_t nb = UK_NB_LIVE;
    std::memcpy(A.scDesc.data() + 44, &nb, 4);
}

// ---------------------------------------------------------------------------
// the host mirror
// ---------------------------------------------------------------------------

static PxVec3 ukBasisExtent(const PxMat33& basis, const PxVec3& e)
{
    const PxVec3 c0 = basis.column0 * e.x, c1 = basis.column1 * e.y, c2 = basis.column2 * e.z;
    return PxVec3(std::fabs(c0.x) + std::fabs(c1.x) + std::fabs(c2.x),
                  std::fabs(c0.y) + std::fabs(c1.y) + std::fabs(c2.y),
                  std::fabs(c0.z) + std::fabs(c1.z) + std::fabs(c2.z));
}

static void ukMirror(UkArena& W, bool zeroSim, bool haveIndices,
                     const std::vector<uint32_t>& jobs)
{
    using physx::Dy::FeatherstoneArticulation;
    const uint32_t nbJobs = uint32_t(jobs.size());
    for (uint32_t j = 0; j < nbJobs; ++j)
    {
        const uint32_t r = haveIndices ? jobs[j] : j;
        uint8_t* R = W.rec.data() + size_t(r) * UK_RECORD_SIZE;
        uint32_t dirty;
        std::memcpy(&dirty, R + 8, 4);
        if (!(dirty & (1u << 20)))
            continue;
        dirty &= ~(1u << 20);
        std::memcpy(R + 8, &dirty, 4);   // the clear happens even when skipped
        const bool positionsDirty = (dirty & (16u | 2u)) != 0u;
        if (!zeroSim && !positionsDirty)
            continue;

        const uint32_t linkCount = ukNumLinks(r);
        const uint32_t dofCount = ukDofCount(r);
        float* xf = &W.xforms[size_t(r) * UK_MAX_LINKS * 7];
        float* tq = &W.tempQ[size_t(r) * UK_MAX_LINKS * 4];
        float* tr = &W.tempR[size_t(r) * UK_MAX_LINKS * 3];
        float* mvel = &W.mvel[size_t(r) * UK_MAX_LINKS * 6];
        const float* jp = &W.jpos[size_t(r) * UK_MAX_DOFS];
        const float* jv = &W.jvel[size_t(r) * UK_MAX_DOFS];

        auto motionVec = [&](uint32_t l, uint32_t d) {
            const float* mm = &W.motion[(size_t(r) * UK_MAX_LINKS + l) * 20 + d * 6];
            return UnAlignedSpatialVector(PxVec3(mm[0], mm[1], mm[2]), PxVec3(mm[3], mm[4], mm[5]));
        };

        // phase A
        for (uint32_t l = 1; l < linkCount; ++l)
        {
            const uint8_t* jc = W.jointCores.data() + (size_t(r) * UK_MAX_LINKS + l) * UK_JC_SIZE;
            PxVec3 parentOffset, childP;
            std::memcpy(&parentOffset, jc + 16, 12);
            std::memcpy(&childP, jc + 44, 12);
            const PxVec3 childOffset = -childP;
            const float* rqf = &W.relq[(size_t(r) * UK_MAX_LINKS + l) * 4];
            const PxQuat relativeQuat(rqf[0], rqf[1], rqf[2], rqf[3]);
            const uint32_t jo = ukJointOffset(r, l);
            const uint32_t nd = ukDofs(r, l);
            const uint32_t type = ukJointType(r, l);

            PxQuat ntc;
            PxVec3 rv;
            if (type == 1u)   // prismatic
            {
                ntc = relativeQuat;
                const PxVec3 u = motionVec(l, 0).bottom;
                rv = ntc.rotate(parentOffset) + childOffset + u * jp[jo];
            }
            else if (type == 2u || type == 3u)   // revolute
            {
                const PxVec3 u = motionVec(l, 0).top;
                PxQuat jointRotation(-jp[jo], u);
                if (jointRotation.w < 0.f)
                    jointRotation = -jointRotation;
                ntc = (jointRotation * relativeQuat).getNormalized();
                rv = ntc.rotate(parentOffset) + childOffset;
            }
            else if (type == 4u)   // spherical
            {
                PxVec3 ang(0.f);
                for (uint32_t d = 0; d < nd; ++d)
                    ang += motionVec(l, d).top * -jp[jo + d];
                const float angle = ang.normalize();
                PxQuat jointRotation = angle < 1e-10f ? PxQuat(physx::PxIdentity)
                                                      : PxQuat(angle, ang);
                if (jointRotation.w < 0.f)
                    jointRotation = -jointRotation;
                ntc = (jointRotation * relativeQuat).getNormalized();
                rv = ntc.rotate(parentOffset) + childOffset;
            }
            else   // fix
            {
                ntc = relativeQuat;
                rv = ntc.rotate(parentOffset) + childOffset;
            }
            tq[l * 4 + 0] = ntc.x; tq[l * 4 + 1] = ntc.y;
            tq[l * 4 + 2] = ntc.z; tq[l * 4 + 3] = ntc.w;
            tr[l * 3 + 0] = rv.x; tr[l * 3 + 1] = rv.y; tr[l * 3 + 2] = rv.z;
        }

        // phase B
        for (uint32_t l = 1; l < linkCount; ++l)
        {
            const uint32_t parent = ukParent(l);
            const PxQuat pQ(xf[parent * 7 + 0], xf[parent * 7 + 1], xf[parent * 7 + 2],
                            xf[parent * 7 + 3]);
            const PxVec3 pP(xf[parent * 7 + 4], xf[parent * 7 + 5], xf[parent * 7 + 6]);
            const PxQuat ntc(tq[l * 4 + 0], tq[l * 4 + 1], tq[l * 4 + 2], tq[l * 4 + 3]);
            const PxVec3 rv(tr[l * 3 + 0], tr[l * 3 + 1], tr[l * 3 + 2]);
            const PxQuat q = (pQ * ntc.getConjugate()).getNormalized();
            const PxVec3 p = pP + q.rotate(rv);
            xf[l * 7 + 0] = q.x; xf[l * 7 + 1] = q.y; xf[l * 7 + 2] = q.z; xf[l * 7 + 3] = q.w;
            xf[l * 7 + 4] = p.x; xf[l * 7 + 5] = p.y; xf[l * 7 + 6] = p.z;

            if (zeroSim)
            {
                UnAlignedSpatialVector parentVel(
                    PxVec3(mvel[parent * 6 + 0], mvel[parent * 6 + 1], mvel[parent * 6 + 2]),
                    PxVec3(mvel[parent * 6 + 3], mvel[parent * 6 + 4], mvel[parent * 6 + 5]));
                const PxVec3 c2p = p - pP;
                UnAlignedSpatialVector linkVelocity =
                    FeatherstoneArticulation::translateSpatialVector(-c2p, parentVel);
                const uint32_t jo = ukJointOffset(r, l);
                const PxTransform b2w(p, q);
                for (uint32_t ind = 0; ind < ukDofs(r, l); ++ind)
                {
                    const UnAlignedSpatialVector worldCol = motionVec(l, ind).rotate(b2w);
                    linkVelocity += worldCol * jv[jo + ind];
                }
                mvel[l * 6 + 0] = linkVelocity.top.x; mvel[l * 6 + 1] = linkVelocity.top.y;
                mvel[l * 6 + 2] = linkVelocity.top.z;
                mvel[l * 6 + 3] = linkVelocity.bottom.x; mvel[l * 6 + 4] = linkVelocity.bottom.y;
                mvel[l * 6 + 5] = linkVelocity.bottom.z;
            }
        }

        // phase C
        if (zeroSim)
        {
            for (uint32_t i = 0; i < linkCount * 6; ++i)
            {
                W.macc[size_t(r) * UK_MAX_LINKS * 6 + i] = 0.f;
                W.incoming[size_t(r) * UK_MAX_LINKS * 6 + i] = 0.f;
            }
            for (uint32_t d = 0; d < dofCount; ++d)
                W.jacc[size_t(r) * UK_MAX_DOFS + d] = 0.f;
        }

        // phase D: every table entry whose node index names one of this
        // articulation's links (the sorted search + backward walk visits
        // exactly the contiguous run of matches)
        for (uint32_t k = 0; k < UK_NB_SHAPES; ++k)
        {
            const uint32_t s = W.shapeOrder[k];
            const UkShapeDef d = ukShape(s);
            if (d.rec != r || W.shapeIndices[k] == 0xFFFFFFFFu)
                continue;
            const uint32_t l = d.link;
            const PxTransform body2World(
                PxVec3(xf[l * 7 + 4], xf[l * 7 + 5], xf[l * 7 + 6]),
                PxQuat(xf[l * 7 + 0], xf[l * 7 + 1], xf[l * 7 + 2], xf[l * 7 + 3]));
            const float* xa = &W.xformsA[(size_t(r) * UK_MAX_LINKS + l) * 7];
            const PxTransform body2Actor(PxVec3(xa[4], xa[5], xa[6]),
                                         PxQuat(xa[0], xa[1], xa[2], xa[3]));
            const physx::PxgShapeSim& ss = W.shapeSims[s];
            const PxTransform absPos =
                body2World.transform(body2Actor.transformInv(ss.mTransform));

            float* ce = &W.cache[size_t(s) * 8];
            ce[0] = absPos.q.x; ce[1] = absPos.q.y; ce[2] = absPos.q.z; ce[3] = absPos.q.w;
            ce[4] = absPos.p.x; ce[5] = absPos.p.y; ce[6] = absPos.p.z;
            uint32_t zero = 0;
            std::memcpy(&ce[7], &zero, 4);

            PxVec3 outMin, outMax;
            const PxVec3 lmin = ss.mLocalBounds.minimum, lmax = ss.mLocalBounds.maximum;
            const uint32_t shapeType = ss.mShapeType;
            if (shapeType == 0u)          // sphere
            {
                outMin = absPos.p + lmin;
                outMax = absPos.p + lmax;
            }
            else if (shapeType == 2u)     // capsule
            {
                const float radius = lmax.y;
                const float halfHeight = lmax.x - radius;
                const PxVec3 dv = absPos.q.getBasisVector0();
                const PxVec3 extents = dv.abs() * halfHeight + PxVec3(radius);
                outMin = absPos.p - extents;
                outMax = absPos.p + extents;
            }
            else if (shapeType == 3u)     // box
            {
                const PxVec3 extents = ukBasisExtent(PxMat33(absPos.q), lmax);
                outMin = absPos.p - extents;
                outMax = absPos.p + extents;
            }
            else                          // convexmesh
            {
                if (ss.mHullDataIndex != 0xFFFFFFFFu)
                {
                    const uint8_t* sh = W.convexShapes.data() + size_t(ss.mHullDataIndex) * 48;
                    float sc[7];
                    std::memcpy(sc, sh, 28);
                    PxMat33 rot(absPos.q);
                    const bool ident = sc[0] == 1.f && sc[1] == 1.f && sc[2] == 1.f &&
                                       sc[3] == 0.f && sc[4] == 0.f && sc[5] == 0.f;
                    if (!ident)
                    {
                        const PxMat33 sr(PxQuat(sc[3], sc[4], sc[5], sc[6]));
                        const PxMat33 srT = sr.getTranspose();
                        PxMat33 diagTimesRot(sr.column0.multiply(PxVec3(sc[0], sc[1], sc[2])),
                                             sr.column1.multiply(PxVec3(sc[0], sc[1], sc[2])),
                                             sr.column2.multiply(PxVec3(sc[0], sc[1], sc[2])));
                        rot = rot * (srT * diagTimesRot);
                    }
                    PxVec3 minV(PX_MAX_F32), maxV(-PX_MAX_F32);
                    for (uint32_t v = 0; v < 5; ++v)
                    {
                        float vf[3];
                        std::memcpy(vf, &W.convexBlob[12 + v * 4], 12);
                        const PxVec3 tv = rot.transform(PxVec3(vf[0], vf[1], vf[2]));
                        minV = minV.minimum(tv);
                        maxV = maxV.maximum(tv);
                    }
                    outMin = minV + absPos.p;
                    outMax = maxV + absPos.p;
                }
                else                       // the fast path
                {
                    const PxMat33 rot(absPos.q);
                    const PxVec3 center = (lmax + lmin) * 0.5f;
                    const PxVec3 extents = (lmax - lmin) * 0.5f;
                    const PxVec3 nc = rot.transform(center) + absPos.p;
                    const PxVec3 ne = ukBasisExtent(rot, extents);
                    outMin = nc - ne;
                    outMax = nc + ne;
                }
            }
            float* be = &W.bounds[size_t(s) * 6];
            be[0] = outMin.x; be[1] = outMin.y; be[2] = outMin.z;
            be[3] = outMax.x; be[4] = outMax.y; be[5] = outMax.z;
        }
    }
}

// ---------------------------------------------------------------------------

static uint32_t ukUlp(float a, float b)
{
    int32_t ia, ib;
    std::memcpy(&ia, &a, 4);
    std::memcpy(&ib, &b, 4);
    if (ia < 0) ia = int32_t(0x80000000u) - ia;
    if (ib < 0) ib = int32_t(0x80000000u) - ib;
    const int64_t d = int64_t(ia) - int64_t(ib);
    const int64_t m = d < 0 ? -d : d;
    return m > 0xFFFFFFFFll ? 0xFFFFFFFFu : uint32_t(m);
}

struct UkStats { size_t bad = 0, loud = 0, tolerated = 0; uint32_t maxUlp = 0; float maxAbs = 0; };

// tolFor(k): whether word k belongs to a transcendental-tainted region
template <typename F>
static void ukCmp(UkStats& st, const char* what, const void* wantP, const void* gotP,
                  size_t bytes, F tolFor)
{
    const uint32_t* w = reinterpret_cast<const uint32_t*>(wantP);
    const uint32_t* g = reinterpret_cast<const uint32_t*>(gotP);
    for (size_t k = 0; k < bytes / 4; ++k)
    {
        if (w[k] == g[k]) continue;
        if (tolFor(k))
        {
            float x, y;
            std::memcpy(&x, &w[k], 4);
            std::memcpy(&y, &g[k], 4);
            const uint32_t u = ukUlp(x, y);
            const float e = std::fabs(x - y);
            st.maxUlp = st.maxUlp > u ? st.maxUlp : u;
            st.maxAbs = st.maxAbs > e ? st.maxAbs : e;
            if (u > 384 && e > 4e-6f)
            {
                if (st.loud < 8)
                    std::printf("MM uk %s tol word %zu: %u ulp, abs %.3g\n", what, k, u, double(e));
                ++st.loud;
            }
            ++st.tolerated;
            continue;
        }
        if (st.bad < 12)
            std::printf("MM uk %s word %zu  ref %08x  gpu %08x\n", what, k, w[k], g[k]);
        ++st.bad;
    }
}

static UkStats ukRun(PxgCudaShim& shim, bool zeroSim, bool haveIndices, uint32_t salt)
{
    UkArena A;
    ukSeed(A, salt);
    std::vector<uint32_t> jobs;
    if (haveIndices)
    {
        // a remap over the live records, so the index path differs from identity
        for (uint32_t j = 0; j < UK_NB_LIVE; ++j)
            jobs.push_back((j * 5u + 1u) % UK_NB_LIVE);
    }
    else
    {
        for (uint32_t j = 0; j < UK_NB_LIVE; ++j)
            jobs.push_back(j);
    }
    UkArena W = A;
    ukMirror(W, zeroSim, haveIndices, jobs);

    std::vector<PxgDevicePtr> bufs;
    auto up = [&](const void* p, size_t n) {
        PxgDevicePtr dp = 0;
        shim.memAlloc(&dp, n);
        if (dp && p) shim.memcpyHtoD(dp, p, n);
        bufs.push_back(dp);
        return dp;
    };
    PxgDevicePtr dJc = up(A.jointCores.data(), A.jointCores.size());
    PxgDevicePtr dJd = up(A.jointData.data(), A.jointData.size());
    PxgDevicePtr dPar = up(A.parents.data(), A.parents.size() * 4);
    PxgDevicePtr dXf = up(A.xforms.data(), A.xforms.size() * 4);
    PxgDevicePtr dXa = up(A.xformsA.data(), A.xformsA.size() * 4);
    PxgDevicePtr dMm = up(A.motion.data(), A.motion.size() * 4);
    PxgDevicePtr dJp = up(A.jpos.data(), A.jpos.size() * 4);
    PxgDevicePtr dJv = up(A.jvel.data(), A.jvel.size() * 4);
    PxgDevicePtr dRq = up(A.relq.data(), A.relq.size() * 4);
    PxgDevicePtr dMv = up(A.mvel.data(), A.mvel.size() * 4);
    PxgDevicePtr dMa = up(A.macc.data(), A.macc.size() * 4);
    PxgDevicePtr dIf = up(A.incoming.data(), A.incoming.size() * 4);
    PxgDevicePtr dJa = up(A.jacc.data(), A.jacc.size() * 4);
    PxgDevicePtr dTq = up(A.tempQ.data(), A.tempQ.size() * 4);
    PxgDevicePtr dTr = up(A.tempR.data(), A.tempR.size() * 4);

    for (uint32_t r = 0; r < UK_NB_RECORDS; ++r)
    {
        uint8_t* R = A.rec.data() + size_t(r) * UK_RECORD_SIZE;
        auto put = [&](uint32_t off, uint64_t v) { std::memcpy(R + off, &v, 8); };
        put(40, uint64_t(dJc) + uint64_t(r) * UK_MAX_LINKS * UK_JC_SIZE);
        put(48, uint64_t(dJd) + uint64_t(r) * UK_MAX_LINKS * 8u);
        put(208, uint64_t(dPar) + uint64_t(r) * UK_MAX_LINKS * 4u);
        put(184, uint64_t(dXf) + uint64_t(r) * UK_MAX_LINKS * 28u);
        put(200, uint64_t(dXa) + uint64_t(r) * UK_MAX_LINKS * 28u);
        put(216, uint64_t(dMm) + uint64_t(r) * UK_MAX_LINKS * 80u);
        put(112, uint64_t(dJp) + uint64_t(r) * UK_MAX_DOFS * 4u);
        put(120, uint64_t(dJv) + uint64_t(r) * UK_MAX_DOFS * 4u);
        put(280, uint64_t(dRq) + uint64_t(r) * UK_MAX_LINKS * 16u);
        put(56, uint64_t(dMv) + uint64_t(r) * UK_MAX_LINKS * 24u);
        put(64, uint64_t(dMa) + uint64_t(r) * UK_MAX_LINKS * 24u);
        put(72, uint64_t(dIf) + uint64_t(r) * UK_MAX_LINKS * 24u);
        put(128, uint64_t(dJa) + uint64_t(r) * UK_MAX_DOFS * 4u);
        put(288, uint64_t(dTq) + uint64_t(r) * UK_MAX_LINKS * 16u);
        put(296, uint64_t(dTr) + uint64_t(r) * UK_MAX_LINKS * 12u);
    }
    PxgDevicePtr dRec = up(A.rec.data(), A.rec.size());
    {
        const uint64_t v = uint64_t(dRec);
        std::memcpy(A.scDesc.data() + 32, &v, 8);
    }
    PxgDevicePtr dDesc = up(A.scDesc.data(), A.scDesc.size());

    PxgDevicePtr dNode = up(A.nodeIndices.data(), A.nodeIndices.size() * 8);
    PxgDevicePtr dSidx = up(A.shapeIndices.data(), A.shapeIndices.size() * 4);
    PxgDevicePtr dSims = up(A.shapeSims.data(), A.shapeSims.size() * sizeof(physx::PxgShapeSim));
    PxgDevicePtr dBlob = up(A.convexBlob.data(), A.convexBlob.size() * 4);
    // patch hullOrMeshPtr of both PxgShape images to the blob's device address
    {
        const uint64_t bp = uint64_t(dBlob);
        std::memcpy(A.convexShapes.data() + 32, &bp, 8);
        std::memcpy(A.convexShapes.data() + 48 + 32, &bp, 8);
    }
    PxgDevicePtr dShapes = up(A.convexShapes.data(), A.convexShapes.size());
    PxgDevicePtr dCache = up(A.cache.data(), A.cache.size() * 4);
    PxgDevicePtr dBounds = up(A.bounds.data(), A.bounds.size() * 4);
    std::vector<uint32_t> idxBuf = jobs;
    for (uint32_t j = UK_NB_LIVE; j < UK_NB_RECORDS; ++j)
        idxBuf.push_back(j);   // the guard records, reachable only by a dropped guard
    PxgDevicePtr dIdx = haveIndices ? up(idxBuf.data(), idxBuf.size() * 4) : 0;

    bool ok = true;
    for (PxgDevicePtr p : bufs) ok = ok && (p != 0);
    if (!ok)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
        return UkStats();
    }

    struct Params
    {
        uint64_t scDesc, shapeSims, convexShapes, transformCache, rigidNodeIndices, shapeIndices;
        uint32_t numShapes, pad0;
        uint64_t bounds, gpuIndices;
        uint32_t nbElements, zeroSimWord;
    } prm = { uint64_t(dDesc), uint64_t(dSims), uint64_t(dShapes), uint64_t(dCache),
              uint64_t(dNode), uint64_t(dSidx), UK_NB_SHAPES, 0u,
              uint64_t(dBounds), uint64_t(dIdx),
              haveIndices ? UK_NB_LIVE : 0u, zeroSim ? 1u : 0u };
    PXG_CHECK_MSG(shim.launchKernel(0, UK_GRID, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch artiUpdateKinematic");
    shim.streamSynchronize();

    // per-record trig classification for the tolerance regions
    bool trig[UK_NB_RECORDS];
    for (uint32_t r = 0; r < UK_NB_RECORDS; ++r) trig[r] = ukTrig(r) && r < UK_NB_LIVE;
    // shape entry -> tainted (its record has a transcendental)
    bool shapeTol[UK_NB_SHAPES];
    for (uint32_t s = 0; s < UK_NB_SHAPES; ++s)
    {
        const UkShapeDef d = ukShape(s);
        shapeTol[s] = (d.rec != 0xFFu) && trig[d.rec];
    }

    UkStats st;
    auto cmpBuf = [&](PxgDevicePtr dp, const std::vector<float>& want, const char* name,
                      uint32_t wordsPerRec) {
        std::vector<float> got(want.size());
        shim.memcpyDtoH(got.data(), dp, got.size() * 4);
        ukCmp(st, name, want.data(), got.data(), want.size() * 4,
              [&](size_t k) { return trig[uint32_t(k / wordsPerRec) % UK_NB_RECORDS]; });
    };
    cmpBuf(dXf, W.xforms, "body2Worlds", UK_MAX_LINKS * 7);
    cmpBuf(dTq, W.tempQ, "tempNtc", UK_MAX_LINKS * 4);
    cmpBuf(dTr, W.tempR, "tempRs", UK_MAX_LINKS * 3);
    cmpBuf(dMv, W.mvel, "motionVel", UK_MAX_LINKS * 6);
    // the zeroed sim outputs: bit-exact always
    auto cmpExact = [&](PxgDevicePtr dp, const std::vector<float>& want, const char* name) {
        std::vector<float> got(want.size());
        shim.memcpyDtoH(got.data(), dp, got.size() * 4);
        ukCmp(st, name, want.data(), got.data(), want.size() * 4, [](size_t) { return false; });
    };
    cmpExact(dMa, W.macc, "motionAccel");
    cmpExact(dIf, W.incoming, "incomingJF");
    cmpExact(dJa, W.jacc, "jointAccel");
    // the dirty-flag words (and the rest of the record image): compare the
    // flag words only -- the record blob also holds the patched pointers
    {
        std::vector<uint8_t> gotRec(A.rec.size());
        shim.memcpyDtoH(gotRec.data(), dRec, gotRec.size());
        for (uint32_t r = 0; r < UK_NB_RECORDS; ++r)
        {
            uint32_t wantW, gotW;
            std::memcpy(&wantW, W.rec.data() + size_t(r) * UK_RECORD_SIZE + 8, 4);
            std::memcpy(&gotW, gotRec.data() + size_t(r) * UK_RECORD_SIZE + 8, 4);
            if (wantW != gotW)
            {
                if (st.bad < 12)
                    std::printf("MM uk dirty rec %u  ref %08x  gpu %08x\n", r, wantW, gotW);
                ++st.bad;
            }
        }
    }
    // cache + bounds: per-shape tolerance mapping
    {
        std::vector<float> got(W.cache.size());
        shim.memcpyDtoH(got.data(), dCache, got.size() * 4);
        ukCmp(st, "cache", W.cache.data(), got.data(), got.size() * 4,
              [&](size_t k) { return shapeTol[uint32_t(k / 8)]; });
        std::vector<float> gotB(W.bounds.size());
        shim.memcpyDtoH(gotB.data(), dBounds, gotB.size() * 4);
        ukCmp(st, "bounds", W.bounds.data(), gotB.data(), gotB.size() * 4,
              [&](size_t k) { return shapeTol[uint32_t(k / 6)]; });
    }

    for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
    return st;
}

}  // namespace

PXG_TEST(artiUpdateKinematic)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiUpdateKin.spv", "artiUpdateKinematic", 80u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &artiUkLoader);

    uint32_t nTrig = 0, nExactRec = 0, nSkipped = 0, nUntouched = 0;
    for (uint32_t r = 0; r < UK_NB_LIVE; ++r)
    {
        if (!ukNeeds(r)) { ++nUntouched; continue; }
        if (!ukPosDirty(r)) ++nSkipped;
        if (ukTrig(r)) ++nTrig; else ++nExactRec;
    }
    std::printf("[ info ] uk: %u trig / %u exact records, %u velocity-only, %u untouched\n",
                nTrig, nExactRec, nSkipped, nUntouched);
    PXG_CHECK_MSG(nTrig > 2 && nExactRec > 2 && nSkipped > 1 && nUntouched > 0, "coverage hole");

    UkStats total;
    const struct { bool zeroSim, haveIndices; uint32_t salt; } rounds[3] = {
        { true, true, 3 }, { false, true, 7 }, { true, false, 11 },
    };
    for (const auto& rd : rounds)
    {
        const UkStats st = ukRun(shim, rd.zeroSim, rd.haveIndices, rd.salt);
        std::printf("[ info ] uk zeroSim=%d idx=%d: tolerated %zu words, max %u ulp, "
                    "max abs %.3g, loud %zu, bad %zu\n",
                    rd.zeroSim ? 1 : 0, rd.haveIndices ? 1 : 0, st.tolerated, st.maxUlp,
                    double(st.maxAbs), st.loud, st.bad);
        total.bad += st.bad;
        total.loud += st.loud;
    }
    PXG_CHECK_MSG(total.bad == 0, "%zu words outside the trig scope differ", total.bad);
    PXG_CHECK_MSG(total.loud == 0, "%zu tolerated words outside both bounds", total.loud);
}
