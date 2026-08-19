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


// P8.113: `computeArtiDenseJacobians` (inverseDynamic.cu:230) and
// `computeArtiCentroidalMomentumMatrices` (:1079) -- the two small
// direct-GPU-API kernels after computeArtiCOM's pattern: record-side, one
// warp per job, guarded (unlike the COM kernel), no atomics -- so the
// mirrors are plain transcriptions over engine primitives and every value
// is deterministic.
//
// The jacobian fixture pins: the MAX-sized output pitch against the
// ACTUAL-count row stride (maxLinks/maxDofs deliberately larger than any
// job's counts; untouched cells stay poisoned), fixed and floating bases
// (identity block present/absent, baseDofs 0/6), re-parented links (the
// parent band reads the PARENT's rows, not link-1's), the U8 nbDof read,
// and dead guard jobs.
//
// The centroidal fixture pins: the four MAX-sized buffer regions against
// ACTUAL-count element indexing, the three scattered root-inertia elements
// behind pg, the column-constructed skew matrix, the += onto the copied
// rows, the lane-0 bias tail, and fixed-base jobs that write NOTHING.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "foundation/PxVec3.h"
#include "foundation/PxQuat.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::PxQuat;
using physx::PxVec3;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

static std::vector<uint8_t> artiId2Loader(const char* file) { return loadShaderBlob(file); }

const uint32_t ID2_NB_RECORDS = 24;
const uint32_t ID2_RECORD_SIZE = 544;
const uint32_t ID2_R_NUM_LINKS = 16;    // u16
const uint32_t ID2_R_NUM_DOFS = 18;     // u16
const uint32_t ID2_R_FLAGS = 26;        // u8
const uint32_t ID2_R_JOINT_DATA = 48;
const uint32_t ID2_R_BODY2WORLDS = 184;
const uint32_t ID2_R_PARENTS = 208;
const uint32_t ID2_R_MOTION_MATRIX = 216;
const uint32_t ID2_MAX_LINKS = 7;       // pool sizes; actual counts are smaller
const uint32_t ID2_MAX_DOFS = 15;

static uint32_t id2NumLinks(uint32_t rec) { return 1u + (rec % 6u); }
static bool id2FixBase(uint32_t rec) { return (rec % 3u) == 1u; }
static uint32_t id2Dofs(uint32_t rec, uint32_t link) { return 1u + ((rec + link) % 3u); }
static uint32_t id2Parent(uint32_t link)
{
    if (link <= 1u) return 0u;
    return (link >= 3u && (link % 3u) == 0u) ? 1u : (link - 1u);
}
static uint32_t id2JointOffset(uint32_t rec, uint32_t link)
{
    uint32_t off = 0;
    for (uint32_t j = 1; j < link; ++j)
        off += id2Dofs(rec, j);
    return off;
}
static uint32_t id2DofCount(uint32_t rec)
{
    return id2JointOffset(rec, id2NumLinks(rec)) ;
}
static float id2(uint32_t i, uint32_t k)
{
    return (float((i * 83u + k * 43u) % 521u) + 1.0f) * 0.015625f - 3.5f;
}
static PxQuat id2Quat(uint32_t rec, uint32_t link)
{
    PxQuat q(0.15f + 0.02f * float((rec + link) % 8u), -0.25f + 0.04f * float((rec * 3u + link) % 6u),
             0.35f - 0.03f * float((rec + 2u * link) % 7u), 0.85f);
    return q.getNormalized();
}
static PxVec3 id2Pos(uint32_t rec, uint32_t link)
{
    return PxVec3(id2(rec, 100u + link * 3u), id2(rec, 101u + link * 3u), id2(rec, 102u + link * 3u));
}

// the shared record pool: numLinks/dofs/flags/jointData/parents/body2Worlds/
// motionMatrix, everything else 0xAB
struct Id2Pool
{
    std::vector<uint8_t> rec;
    std::vector<uint8_t> jointData;   // stride 8 per link
    std::vector<uint32_t> parents;
    std::vector<float> xforms;        // 7 floats per link (q, p)
    std::vector<float> motion;        // 80-byte link stride, 24-byte dof stride
};

static void id2BuildPool(Id2Pool& P)
{
    P.rec.assign(size_t(ID2_NB_RECORDS) * ID2_RECORD_SIZE, 0xAB);
    P.jointData.assign(size_t(ID2_NB_RECORDS) * ID2_MAX_LINKS * 8, 0xAB);
    P.parents.assign(size_t(ID2_NB_RECORDS) * ID2_MAX_LINKS, 0xCDCDCDCDu);
    P.xforms.assign(size_t(ID2_NB_RECORDS) * ID2_MAX_LINKS * 7, -88.0f);
    P.motion.assign(size_t(ID2_NB_RECORDS) * ID2_MAX_LINKS * 20, -77.0f);   // 80B = 20 floats

    for (uint32_t r = 0; r < ID2_NB_RECORDS; ++r)
    {
        const uint32_t nl = id2NumLinks(r);
        const uint16_t nl16 = uint16_t(nl);
        const uint16_t nd16 = uint16_t(id2DofCount(r));
        const uint8_t flags = uint8_t((id2FixBase(r) ? 1u : 0u) | ((r & 1u) << 1) | 4u);
        uint8_t* R = P.rec.data() + size_t(r) * ID2_RECORD_SIZE;
        std::memcpy(R + ID2_R_NUM_LINKS, &nl16, 2);
        std::memcpy(R + ID2_R_NUM_DOFS, &nd16, 2);
        R[ID2_R_FLAGS] = flags;

        for (uint32_t l = 0; l < nl; ++l)
        {
            uint8_t* jd = P.jointData.data() + (size_t(r) * ID2_MAX_LINKS + l) * 8;
            const uint32_t jo = (l == 0) ? 0u : id2JointOffset(r, l);
            std::memcpy(jd, &jo, 4);
            jd[4] = uint8_t((l == 0) ? 0u : id2Dofs(r, l));
            // bytes 5..7 stay 0xAB: the U8 read must mask them off
            P.parents[size_t(r) * ID2_MAX_LINKS + l] = (l == 0) ? 0u : id2Parent(l);
            float* xf = &P.xforms[(size_t(r) * ID2_MAX_LINKS + l) * 7];
            const PxQuat q = id2Quat(r, l);
            xf[0] = q.x; xf[1] = q.y; xf[2] = q.z; xf[3] = q.w;
            const PxVec3 p = id2Pos(r, l);
            xf[4] = p.x; xf[5] = p.y; xf[6] = p.z;
            for (uint32_t d = 0; d < 3; ++d)
                for (uint32_t c = 0; c < 6; ++c)
                    P.motion[(size_t(r) * ID2_MAX_LINKS + l) * 20 + d * 6 + c] =
                        id2(r, 300u + l * 20u + d * 6u + c);
        }
    }
}

// ---------------------------------------------------------------------------
// jacobians
// ---------------------------------------------------------------------------

const uint32_t DJ_NB_JOBS = 20;   // 2 blocks of 16 warps: 12 guard-dead jobs
// the launch spans 32 warps; the index buffer and the output region cover all
// 32 so a dropped guard writes into COMPARED poison instead of allocator
// slack (the first sweep run of this fixture proved that lesson)
const uint32_t DJ_NB_SLOTS = 32;
static uint32_t djArti(uint32_t job) { return (job * 7u + 5u) % ID2_NB_RECORDS; }

static void djMirror(const Id2Pool& P, std::vector<float>& out)
{
    const uint32_t maxCols = 6u + ID2_MAX_DOFS;
    const uint32_t maxRows = 6u + (ID2_MAX_LINKS - 1u) * 6u;
    for (uint32_t job = 0; job < DJ_NB_JOBS; ++job)
    {
        const uint32_t r = djArti(job);
        const uint32_t linkCount = id2NumLinks(r);
        const uint32_t dofCount = id2DofCount(r);
        const bool fixedBase = id2FixBase(r);
        const uint32_t baseDofs = fixedBase ? 0u : 6u;
        const uint32_t colCount = baseDofs + dofCount;
        float* jac = &out[size_t(job) * maxCols * maxRows];

        if (!fixedBase)
            for (uint32_t i = 0; i < 6u * colCount; ++i)
                jac[i] = (i / colCount == i % colCount) ? 1.0f : 0.0f;

        for (uint32_t link = 1; link < linkCount; ++link)
        {
            const uint32_t linkOffset = id2JointOffset(r, link) + baseDofs;
            const uint32_t linkDofs = id2Dofs(r, link);
            const uint32_t linkRow0 = (link - 1u) * 6u + baseDofs;
            const uint32_t parentLink = id2Parent(link);
            const uint32_t parentDofs =
                parentLink ? id2JointOffset(r, parentLink) + id2Dofs(r, parentLink) + baseDofs
                           : baseDofs;
            const PxQuat q = id2Quat(r, link);
            for (uint32_t dof = 0; dof < colCount; ++dof)
            {
                PxVec3 linkLin(0.f), linkAng(0.f);
                if (dof < parentDofs)
                {
                    const uint32_t parentRow0 =
                        parentLink ? (parentLink - 1u) * 6u + baseDofs : 0u;
                    const PxVec3 rw = id2Pos(r, link) - id2Pos(r, parentLink);
                    const PxVec3 pl(jac[(parentRow0 + 0) * colCount + dof],
                                    jac[(parentRow0 + 1) * colCount + dof],
                                    jac[(parentRow0 + 2) * colCount + dof]);
                    const PxVec3 pa(jac[(parentRow0 + 3) * colCount + dof],
                                    jac[(parentRow0 + 4) * colCount + dof],
                                    jac[(parentRow0 + 5) * colCount + dof]);
                    linkLin = pl + pa.cross(rw);
                    linkAng = pa;
                }
                else if (dof >= linkOffset && dof < linkOffset + linkDofs)
                {
                    const float* mm =
                        &P.motion[(size_t(r) * ID2_MAX_LINKS + link) * 20
                                  + (dof - linkOffset) * 6];
                    linkLin = q.rotate(PxVec3(mm[3], mm[4], mm[5]));
                    linkAng = q.rotate(PxVec3(mm[0], mm[1], mm[2]));
                }
                jac[(linkRow0 + 0) * colCount + dof] = linkLin.x;
                jac[(linkRow0 + 1) * colCount + dof] = linkLin.y;
                jac[(linkRow0 + 2) * colCount + dof] = linkLin.z;
                jac[(linkRow0 + 3) * colCount + dof] = linkAng.x;
                jac[(linkRow0 + 4) * colCount + dof] = linkAng.y;
                jac[(linkRow0 + 5) * colCount + dof] = linkAng.z;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// centroidal momentum
// ---------------------------------------------------------------------------

const uint32_t CMM_NB_JOBS = 12;   // 2 blocks of 8 warps: 4 guard-dead jobs
static uint32_t cmmArti(uint32_t job) { return (job * 11u + 2u) % ID2_NB_RECORDS; }

static void cmmMirror(const std::vector<float>& in, std::vector<float>& out)
{
    const uint32_t mc = ID2_MAX_DOFS + 6u;
    const uint32_t nb = CMM_NB_JOBS;
    const uint32_t startCoriolis = mc * mc * nb;
    const uint32_t startMatrix = startCoriolis + mc * nb;
    const uint32_t startBias = startMatrix + 6u * mc * nb;
    for (uint32_t job = 0; job < CMM_NB_JOBS; ++job)
    {
        const uint32_t r = cmmArti(job);
        if (id2FixBase(r))
            continue;   // writes NOTHING
        const uint32_t d6 = id2DofCount(r) + 6u;
        const float* mm = &in[size_t(job) * mc * mc];
        const float* cf = &in[startCoriolis + size_t(job) * mc];
        float* cm = &out[startMatrix + size_t(job) * 6u * mc];
        float* cb = &out[startBias + size_t(job) * 6u];

        for (uint32_t j = 0; j < 6u * d6; ++j)
            cm[j] = mm[j];

        const float totalMass = mm[2 * d6 + 2];
        const float invM = 1.0f / totalMass;
        const PxVec3 pg(mm[2 * d6 + 4] * invM, mm[5] * invM, mm[d6 + 3] * invM);
        const PxVec3 c0(0.f, pg.z, -pg.y), c1(-pg.z, 0.f, pg.x), c2(pg.y, -pg.x, 0.f);

        for (uint32_t col = 0; col < d6; ++col)
        {
            const PxVec3 lin(mm[0 * d6 + col], mm[1 * d6 + col], mm[2 * d6 + col]);
            const PxVec3 cr = c0 * lin.x + c1 * lin.y + c2 * lin.z;
            cm[col + 3 * d6] += cr.x;
            cm[col + 4 * d6] += cr.y;
            cm[col + 5 * d6] += cr.z;
        }
        for (uint32_t row = 0; row < 6; ++row)
            cb[row] = cf[row];
        const PxVec3 lin(cf[0], cf[1], cf[2]);
        const PxVec3 cr = c0 * lin.x + c1 * lin.y + c2 * lin.z;
        cb[3] += cr.x;
        cb[4] += cr.y;
        cb[5] += cr.z;
    }
}

// ---------------------------------------------------------------------------

PXG_TEST(artiInverseDyn2)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiInverseDyn.spv", "computeArtiDenseJacobians", 40u, 0 },
        { "npArtiInverseDyn.spv", "computeArtiCentroidalMomentumMatrices", 32u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 2, &artiId2Loader);

    Id2Pool P;
    id2BuildPool(P);

    // coverage
    {
        uint32_t nFixed = 0, nFree = 0, nBranch = 0;
        for (uint32_t job = 0; job < DJ_NB_JOBS; ++job)
        {
            const uint32_t r = djArti(job);
            if (id2FixBase(r)) ++nFixed; else ++nFree;
            for (uint32_t l = 2; l < id2NumLinks(r); ++l)
                if (id2Parent(l) != l - 1) ++nBranch;
        }
        std::printf("[ info ] id2/dj: %u fixed / %u free jobs, %u re-parented links\n",
                    nFixed, nFree, nBranch);
        PXG_CHECK_MSG(nFixed > 2 && nFree > 2 && nBranch > 2, "jacobian coverage hole");
    }

    // ---- upload the shared pool ----
    std::vector<PxgDevicePtr> bufs;
    auto up = [&](const void* p, size_t n) {
        PxgDevicePtr dp = 0;
        shim.memAlloc(&dp, n);
        if (dp && p) shim.memcpyHtoD(dp, p, n);
        bufs.push_back(dp);
        return dp;
    };
    PxgDevicePtr dJd = up(P.jointData.data(), P.jointData.size());
    PxgDevicePtr dPar = up(P.parents.data(), P.parents.size() * 4);
    PxgDevicePtr dXf = up(P.xforms.data(), P.xforms.size() * 4);
    PxgDevicePtr dMm = up(P.motion.data(), P.motion.size() * 4);
    for (uint32_t r = 0; r < ID2_NB_RECORDS; ++r)
    {
        uint8_t* R = P.rec.data() + size_t(r) * ID2_RECORD_SIZE;
        const uint64_t jd = uint64_t(dJd) + uint64_t(r) * ID2_MAX_LINKS * 8u;
        const uint64_t pr = uint64_t(dPar) + uint64_t(r) * ID2_MAX_LINKS * 4u;
        const uint64_t xf = uint64_t(dXf) + uint64_t(r) * ID2_MAX_LINKS * 28u;
        const uint64_t mo = uint64_t(dMm) + uint64_t(r) * ID2_MAX_LINKS * 80u;
        std::memcpy(R + ID2_R_JOINT_DATA, &jd, 8);
        std::memcpy(R + ID2_R_PARENTS, &pr, 8);
        std::memcpy(R + ID2_R_BODY2WORLDS, &xf, 8);
        std::memcpy(R + ID2_R_MOTION_MATRIX, &mo, 8);
    }
    PxgDevicePtr dRec = up(P.rec.data(), P.rec.size());

    // ---- jacobians ----
    const uint32_t maxCols = 6u + ID2_MAX_DOFS, maxRows = 6u + (ID2_MAX_LINKS - 1u) * 6u;
    std::vector<float> djOut(size_t(DJ_NB_SLOTS) * maxCols * maxRows);
    std::memset(djOut.data(), 0xAB, djOut.size() * 4);
    std::vector<float> djWant = djOut;
    djMirror(P, djWant);   // fills jobs 0..19; slots 20..31 must stay poison

    std::vector<uint32_t> djIdx(DJ_NB_SLOTS);
    for (uint32_t j = 0; j < DJ_NB_SLOTS; ++j) djIdx[j] = djArti(j);
    PxgDevicePtr dDjIdx = up(djIdx.data(), djIdx.size() * 4);
    PxgDevicePtr dDjOut = up(djOut.data(), djOut.size() * 4);

    // ---- centroidal ----
    const uint32_t mc = ID2_MAX_DOFS + 6u;
    const size_t cmmTotal = size_t(mc) * mc * CMM_NB_JOBS + size_t(mc) * CMM_NB_JOBS
                          + size_t(6) * mc * CMM_NB_JOBS + size_t(6) * CMM_NB_JOBS;
    std::vector<float> cmmData(cmmTotal);
    std::memset(cmmData.data(), 0xAB, cmmData.size() * 4);
    // the two INPUT regions get real values
    for (size_t i = 0; i < size_t(mc) * mc * CMM_NB_JOBS + size_t(mc) * CMM_NB_JOBS; ++i)
        cmmData[i] = id2(uint32_t(i % 997u), uint32_t(i / 997u)) * 0.5f;
    std::vector<float> cmmWant = cmmData;
    cmmMirror(cmmData, cmmWant);

    std::vector<uint32_t> cmmIdx(CMM_NB_JOBS);
    for (uint32_t j = 0; j < CMM_NB_JOBS; ++j) cmmIdx[j] = cmmArti(j);
    PxgDevicePtr dCmIdx = up(cmmIdx.data(), cmmIdx.size() * 4);
    PxgDevicePtr dCmData = up(cmmData.data(), cmmData.size() * 4);

    bool ok = true;
    for (PxgDevicePtr p : bufs) ok = ok && (p != 0);
    if (!ok)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
        return;
    }

    struct DjP
    {
        uint64_t data, idx;
        uint32_t nb, maxLinks, maxDofs, pad;
        uint64_t rec;
    } dj = { uint64_t(dDjOut), uint64_t(dDjIdx), DJ_NB_JOBS, ID2_MAX_LINKS, ID2_MAX_DOFS, 0u,
             uint64_t(dRec) };
    PXG_CHECK_MSG(shim.launchKernel(0, (DJ_NB_JOBS + 15) / 16, 1, 1, &dj, sizeof(dj)) ==
                      PxgResult::Success,
                  "launch computeArtiDenseJacobians");

    struct CmP
    {
        uint64_t data, idx;
        uint32_t nb, maxDofs;
        uint64_t rec;
    } cp = { uint64_t(dCmData), uint64_t(dCmIdx), CMM_NB_JOBS, ID2_MAX_DOFS, uint64_t(dRec) };
    PXG_CHECK_MSG(shim.launchKernel(1, (CMM_NB_JOBS + 7) / 8, 1, 1, &cp, sizeof(cp)) ==
                      PxgResult::Success,
                  "launch computeArtiCentroidalMomentumMatrices");
    shim.streamSynchronize();

    size_t bad = 0;
    auto cmp = [&](PxgDevicePtr dp, const std::vector<float>& want, const char* name) {
        std::vector<float> got(want.size());
        shim.memcpyDtoH(got.data(), dp, got.size() * 4);
        const uint32_t* w = reinterpret_cast<const uint32_t*>(want.data());
        const uint32_t* g = reinterpret_cast<const uint32_t*>(got.data());
        for (size_t k = 0; k < want.size(); ++k)
            if (w[k] != g[k])
            {
                if (bad < 10)
                    std::printf("MM id2/%s word %zu  ref %08x (%g)  gpu %08x (%g)\n", name, k,
                                w[k], want[k], g[k], got[k]);
                ++bad;
            }
    };
    cmp(dDjOut, djWant, "jacobians");
    cmp(dCmData, cmmWant, "centroidal");

    for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
    PXG_CHECK_MSG(bad == 0, "%zu mismatches", bad);
}

}  // namespace
