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


// P8.114: the `computeArtiGravityForces` engine entry
// (inverseDynamic.cu:593) -- the gravity-compensation direct-API kernel,
// both branches.  The mirror is the CUDA's structure over engine
// primitives: Cm::transformInertiaTensor, PxMat33(quat), the pxgref
// skew/translateInertia pair, Dy::SpatialMatrix operator+ / operator* /
// invertInertia, Dy::FeatherstoneArticulation::translateSpatialVector,
// UnAlignedSpatialVector::rotate(PxTransform) and the dual innerProduct --
// every one PX_CUDA_CALLABLE or already pinned by its own P8.86 fixture.
//
// Pins: the record-scratch reinterpretation (worldSpatialArticulatedInertia
// read at the Dy::SpatialMatrix stride of 112, NOT PxSpatialMatrix's 144 --
// the fixture pins both numbers by static_assert), the dof-index staging
// trick through the output buffer, rootDof vs bufferDof, per-component zero
// guards (a static link in every branch), motionVelocities[0] as a live
// INPUT, the negated root acceleration, the engine's own zA sign
// inconsistency between the branches, and two rounds (rootMotion on/off)
// with fixed and floating bases mixed in both.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"
#include "pxgArtiRefHelpers.h"

#include "foundation/PxVec3.h"
#include "foundation/PxQuat.h"
#include "foundation/PxTransform.h"
#include "CmUtils.h"
#include "PxSpatialMatrix.h"

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

static_assert(sizeof(physx::Dy::SpatialMatrix) == 112, "the reinterpret stride");
static_assert(sizeof(physx::PxSpatialMatrix) == 144, "the underlying buffer stride");
static_assert(offsetof(physx::PxgArticulation, worldSpatialArticulatedInertia) == 168,
              "..WORLD_SPATIAL_ARTICULATED_INERTIA");

static std::vector<uint8_t> artiGfLoader(const char* file) { return loadShaderBlob(file); }

const uint32_t GF_NB_JOBS = 12;      // 2 blocks of 8 warps: 4 guard-dead jobs
const uint32_t GF_NB_SLOTS = 16;     // index + output cover every launched warp
const uint32_t GF_NB_RECORDS = 18;
const uint32_t GF_RECORD_SIZE = 544;
const uint32_t GF_R_NUM_LINKS = 16, GF_R_NUM_DOFS = 18, GF_R_FLAGS = 26;
const uint32_t GF_R_JOINT_DATA = 48, GF_R_MOTION_VEL = 56, GF_R_MOTION_ACC = 64;
const uint32_t GF_R_ZA = 88, GF_R_WORLD_INERTIA = 168, GF_R_BODY2WORLDS = 184;
const uint32_t GF_R_PARENTS = 208, GF_R_MOTION_MATRIX = 216, GF_R_LINK_PROPS = 256;
const uint32_t GF_MAX_LINKS = 6;
const uint32_t GF_MAX_DOFS = 12;
const PxVec3 GF_GRAVITY(0.125f, -9.8125f, 0.375f);

static uint32_t gfArti(uint32_t job) { return (job * 5u + 1u) % GF_NB_RECORDS; }
static uint32_t gfNumLinks(uint32_t rec) { return 2u + (rec % 5u); }
static bool gfFixBase(uint32_t rec) { return (rec % 3u) == 1u; }
static uint32_t gfDofs(uint32_t rec, uint32_t link) { return 1u + ((rec + link) % 3u); }
static uint32_t gfParent(uint32_t link)
{
    if (link <= 1u) return 0u;
    return (link >= 3u && (link % 3u) == 0u) ? 1u : (link - 1u);
}
static uint32_t gfJointOffset(uint32_t rec, uint32_t link)
{
    uint32_t off = 0;
    for (uint32_t j = 1; j < link; ++j)
        off += gfDofs(rec, j);
    return off;
}
static uint32_t gfDofCount(uint32_t rec) { return gfJointOffset(rec, gfNumLinks(rec)); }
// link 2 of every fourth record is STATIC: invMass and invInertia all zero
static bool gfStatic(uint32_t rec, uint32_t link) { return (rec % 4u) == 2u && link == 2u; }
static float gf(uint32_t i, uint32_t k)
{
    return (float((i * 89u + k * 47u) % 523u) + 1.0f) * 0.015625f - 3.5f;
}
static PxQuat gfQuat(uint32_t rec, uint32_t link)
{
    PxQuat q(0.12f + 0.03f * float((rec + link) % 7u), -0.22f + 0.05f * float((rec * 2u + link) % 5u),
             0.31f - 0.04f * float((rec + 3u * link) % 6u), 0.9f);
    return q.getNormalized();
}
static PxVec3 gfPos(uint32_t rec, uint32_t link)
{
    return PxVec3(gf(rec, 100u + link * 3u), gf(rec, 101u + link * 3u), gf(rec, 102u + link * 3u));
}
static physx::PxVec4 gfInvIM(uint32_t rec, uint32_t link)
{
    if (gfStatic(rec, link))
        return physx::PxVec4(0.f, 0.f, 0.f, 0.f);
    return physx::PxVec4(0.25f + 0.125f * float((rec + link) % 5u),
                         0.5f + 0.0625f * float((rec * 2u + link) % 4u),
                         0.75f + 0.03125f * float((rec + 2u * link) % 6u),
                         0.5f + 0.25f * float((rec + link) % 3u));
}

struct GfArena
{
    std::vector<uint8_t> rec;
    std::vector<uint8_t> jointData;
    std::vector<uint32_t> parents;
    std::vector<float> xforms;   // 7 per link
    std::vector<float> motion;   // 20 per link
    std::vector<float> props;    // 4 per link
    std::vector<float> za;       // 6 per link (scratch, poisoned)
    std::vector<float> inertia;  // 28 per link (112 B, scratch, poisoned)
    std::vector<float> mvel;     // 6 per link (link 0 = LIVE INPUT, rest poison)
    std::vector<float> macc;     // 6 per link (scratch, poisoned)
    std::vector<float> out;      // per-job (maxDofs + bufferDof) floats
    uint32_t bufferDof;
};

static void gfSeed(GfArena& A, bool rootMotion, uint32_t salt)
{
    A.bufferDof = rootMotion ? 6u : 0u;
    A.rec.assign(size_t(GF_NB_RECORDS) * GF_RECORD_SIZE, 0xAB);
    A.jointData.assign(size_t(GF_NB_RECORDS) * GF_MAX_LINKS * 8, 0xAB);
    A.parents.assign(size_t(GF_NB_RECORDS) * GF_MAX_LINKS, 0xCDCDCDCDu);
    A.xforms.assign(size_t(GF_NB_RECORDS) * GF_MAX_LINKS * 7, -88.f);
    A.motion.assign(size_t(GF_NB_RECORDS) * GF_MAX_LINKS * 20, -77.f);
    A.props.assign(size_t(GF_NB_RECORDS) * GF_MAX_LINKS * 4, -66.f);
    A.za.assign(size_t(GF_NB_RECORDS) * GF_MAX_LINKS * 6, 0.f);
    A.inertia.assign(size_t(GF_NB_RECORDS) * GF_MAX_LINKS * 28, 0.f);
    A.mvel.assign(size_t(GF_NB_RECORDS) * GF_MAX_LINKS * 6, 0.f);
    A.macc.assign(size_t(GF_NB_RECORDS) * GF_MAX_LINKS * 6, 0.f);
    std::memset(A.za.data(), 0xAB, A.za.size() * 4);
    std::memset(A.inertia.data(), 0xAB, A.inertia.size() * 4);
    std::memset(A.mvel.data(), 0xAB, A.mvel.size() * 4);
    std::memset(A.macc.data(), 0xAB, A.macc.size() * 4);
    A.out.assign(size_t(GF_NB_SLOTS) * (GF_MAX_DOFS + A.bufferDof), 0.f);
    std::memset(A.out.data(), 0xAB, A.out.size() * 4);

    for (uint32_t r = 0; r < GF_NB_RECORDS; ++r)
    {
        const uint32_t nl = gfNumLinks(r);
        const uint16_t nl16 = uint16_t(nl), nd16 = uint16_t(gfDofCount(r));
        uint8_t* R = A.rec.data() + size_t(r) * GF_RECORD_SIZE;
        std::memcpy(R + GF_R_NUM_LINKS, &nl16, 2);
        std::memcpy(R + GF_R_NUM_DOFS, &nd16, 2);
        R[GF_R_FLAGS] = uint8_t((gfFixBase(r) ? 1u : 0u) | ((r & 1u) << 1) | 4u);

        // the root's motion velocity is a LIVE INPUT (branch B reads it)
        for (uint32_t c = 0; c < 6; ++c)
            A.mvel[(size_t(r) * GF_MAX_LINKS + 0) * 6 + c] = gf(r, 700u + c + salt);

        for (uint32_t l = 0; l < nl; ++l)
        {
            uint8_t* jd = A.jointData.data() + (size_t(r) * GF_MAX_LINKS + l) * 8;
            const uint32_t jo = (l == 0) ? 0u : gfJointOffset(r, l);
            std::memcpy(jd, &jo, 4);
            jd[4] = uint8_t((l == 0) ? 0u : gfDofs(r, l));
            A.parents[size_t(r) * GF_MAX_LINKS + l] = (l == 0) ? 0u : gfParent(l);
            float* xf = &A.xforms[(size_t(r) * GF_MAX_LINKS + l) * 7];
            const PxQuat q = gfQuat(r, l);
            xf[0] = q.x; xf[1] = q.y; xf[2] = q.z; xf[3] = q.w;
            const PxVec3 p = gfPos(r, l);
            xf[4] = p.x; xf[5] = p.y; xf[6] = p.z;
            const physx::PxVec4 im = gfInvIM(r, l);
            float* pp = &A.props[(size_t(r) * GF_MAX_LINKS + l) * 4];
            pp[0] = im.x; pp[1] = im.y; pp[2] = im.z; pp[3] = im.w;
            for (uint32_t d = 0; d < 3; ++d)
                for (uint32_t c = 0; c < 6; ++c)
                    A.motion[(size_t(r) * GF_MAX_LINKS + l) * 20 + d * 6 + c] =
                        gf(r, 300u + l * 20u + d * 6u + c + salt);
        }
    }
}

static UnAlignedSpatialVector gfLoadSV(const float* p)
{
    return UnAlignedSpatialVector(PxVec3(p[0], p[1], p[2]), PxVec3(p[3], p[4], p[5]));
}
static void gfStoreSV(float* p, const UnAlignedSpatialVector& v)
{
    p[0] = v.top.x; p[1] = v.top.y; p[2] = v.top.z;
    p[3] = v.bottom.x; p[4] = v.bottom.y; p[5] = v.bottom.z;
}
static physx::Dy::SpatialMatrix gfLoadSM(const float* p)
{
    physx::Dy::SpatialMatrix m;
    std::memcpy(&m, p, 108);
    return m;
}
static void gfStoreSM(float* p, const physx::Dy::SpatialMatrix& m)
{
    std::memcpy(p, &m, 108);
}

// the mirror: the kernel's structure, sequentially, over engine primitives
static void gfMirror(GfArena& W, bool rootMotion)
{
    using physx::Dy::FeatherstoneArticulation;
    const PxVec3 tGravity = -GF_GRAVITY;
    for (uint32_t job = 0; job < GF_NB_JOBS; ++job)
    {
        const uint32_t r = gfArti(job);
        const uint32_t linkCount = gfNumLinks(r);
        const uint32_t dofCount = gfDofCount(r);
        const bool fixBase = gfFixBase(r);
        const uint32_t rootDof = (rootMotion && !fixBase) ? 6u : 0u;
        float* gcf = &W.out[size_t(job) * (GF_MAX_DOFS + W.bufferDof)];
        float* za = &W.za[size_t(r) * GF_MAX_LINKS * 6];
        float* mvel = &W.mvel[size_t(r) * GF_MAX_LINKS * 6];
        float* macc = &W.macc[size_t(r) * GF_MAX_LINKS * 6];
        float* inr = &W.inertia[size_t(r) * GF_MAX_LINKS * 28];
        auto b2w = [&](uint32_t l) {
            const float* xf = &W.xforms[(size_t(r) * GF_MAX_LINKS + l) * 7];
            return PxTransform(PxVec3(xf[4], xf[5], xf[6]), PxQuat(xf[0], xf[1], xf[2], xf[3]));
        };
        auto invIM = [&](uint32_t l) { return gfInvIM(r, l); };
        auto jointOff = [&](uint32_t l) { return gfJointOffset(r, l); };
        auto nbDof = [&](uint32_t l) { return gfDofs(r, l); };
        auto motionVec = [&](uint32_t l, uint32_t d) {
            const float* mm = &W.motion[(size_t(r) * GF_MAX_LINKS + l) * 20 + d * 6];
            return UnAlignedSpatialVector(PxVec3(mm[0], mm[1], mm[2]), PxVec3(mm[3], mm[4], mm[5]));
        };

        if (rootMotion || fixBase)
        {
            for (uint32_t l = 0; l < linkCount; ++l)
            {
                const float w = invIM(l).w;
                const float mass = w == 0.f ? 0.f : 1.f / w;
                gfStoreSV(&za[l * 6], UnAlignedSpatialVector(tGravity * mass, PxVec3(0.f)));
            }
            for (uint32_t l = linkCount - 1; l > 0; --l)
            {
                const uint32_t pl = gfParent(l);
                const UnAlignedSpatialVector t = FeatherstoneArticulation::translateSpatialVector(
                    b2w(l).p - b2w(pl).p, gfLoadSV(&za[l * 6]));
                gfStoreSV(&za[pl * 6], gfLoadSV(&za[pl * 6]) + t);
                for (uint32_t d = 0; d < nbDof(l); ++d)
                    gcf[jointOff(l) + d + rootDof] = float(l);
            }
            for (uint32_t dof = 0; dof < dofCount; ++dof)
            {
                const uint32_t l = uint32_t(gcf[dof + rootDof]);
                const UnAlignedSpatialVector dm =
                    motionVec(l, dof - jointOff(l)).rotate(b2w(l));
                gcf[dof + rootDof] = dm.innerProduct(gfLoadSV(&za[l * 6]));
            }
            if (rootDof == 6)
            {
                const UnAlignedSpatialVector z0 = gfLoadSV(&za[0]);
                gcf[0] = z0.top.x; gcf[1] = z0.top.y; gcf[2] = z0.top.z;
                gcf[3] = z0.bottom.x; gcf[4] = z0.bottom.y; gcf[5] = z0.bottom.z;
            }
        }
        else
        {
            for (uint32_t l = 0; l < linkCount; ++l)
            {
                const physx::PxVec4 im = invIM(l);
                const float mass = im.w == 0.f ? 0.f : 1.f / im.w;
                const PxVec3 localInertia(im.x == 0.f ? 0.f : 1.f / im.x,
                                          im.y == 0.f ? 0.f : 1.f / im.y,
                                          im.z == 0.f ? 0.f : 1.f / im.z);
                physx::Dy::SpatialMatrix sm;
                sm.topLeft = PxMat33(physx::PxZero);
                sm.topRight = PxMat33::createDiagonal(PxVec3(mass));
                physx::Cm::transformInertiaTensor(localInertia, PxMat33(b2w(l).q), sm.bottomLeft);
                gfStoreSM(&inr[l * 28], sm);
            }
            for (uint32_t l = linkCount - 1; l > 0; --l)
            {
                const uint32_t pl = gfParent(l);
                const PxVec3 rw = b2w(l).p - b2w(pl).p;
                physx::Dy::SpatialMatrix parentSpace = gfLoadSM(&inr[l * 28]);
                pxgtest::refTranslateInertia(pxgtest::refSkewSymmetric(rw.x, rw.y, rw.z),
                                             parentSpace);
                gfStoreSM(&inr[pl * 28], gfLoadSM(&inr[pl * 28]) + parentSpace);
            }
            for (uint32_t l = 1; l < linkCount; ++l)
            {
                const uint32_t pl = gfParent(l);
                gfStoreSV(&mvel[l * 6], FeatherstoneArticulation::translateSpatialVector(
                                            b2w(pl).p - b2w(l).p, gfLoadSV(&mvel[pl * 6])));
            }
            for (uint32_t l = 0; l < linkCount; ++l)
            {
                const float w = invIM(l).w;
                const float mass = w == 0.f ? 0.f : 1.f / w;
                const PxMat33 I = gfLoadSM(&inr[l * 28]).bottomLeft;
                const PxVec3 vA = gfLoadSV(&mvel[l * 6]).top;
                gfStoreSV(&za[l * 6],
                          UnAlignedSpatialVector(-tGravity * mass, vA.cross(I * vA)));
            }
            for (uint32_t l = linkCount - 1; l > 0; --l)
            {
                const uint32_t pl = gfParent(l);
                const UnAlignedSpatialVector t = FeatherstoneArticulation::translateSpatialVector(
                    b2w(l).p - b2w(pl).p, gfLoadSV(&za[l * 6]));
                gfStoreSV(&za[pl * 6], gfLoadSV(&za[pl * 6]) + t);
            }
            {
                physx::Dy::SpatialMatrix invInertia = gfLoadSM(&inr[0]).invertInertia();
                const UnAlignedSpatialVector a0 = invInertia * gfLoadSV(&za[0]);
                gfStoreSV(&macc[0], UnAlignedSpatialVector(-a0.top, -a0.bottom));
            }
            for (uint32_t l = 1; l < linkCount; ++l)
            {
                const uint32_t pl = gfParent(l);
                const UnAlignedSpatialVector acc = FeatherstoneArticulation::translateSpatialVector(
                    b2w(pl).p - b2w(l).p, gfLoadSV(&macc[pl * 6]));
                gfStoreSV(&macc[l * 6], acc);
                gfStoreSV(&za[l * 6], gfLoadSM(&inr[l * 28]) * acc + gfLoadSV(&za[l * 6]));
                for (uint32_t d = 0; d < nbDof(l); ++d)
                    gcf[jointOff(l) + d] = float(l);
            }
            for (uint32_t dof = 0; dof < dofCount; ++dof)
            {
                const uint32_t l = uint32_t(gcf[dof]);
                const UnAlignedSpatialVector dm =
                    motionVec(l, dof - jointOff(l)).rotate(b2w(l));
                gcf[dof] = dm.innerProduct(gfLoadSV(&za[l * 6]));
            }
        }
    }
}

// ---------------------------------------------------------------------------

static size_t gfRun(PxgCudaShim& shim, bool rootMotion, uint32_t salt)
{
    GfArena A;
    gfSeed(A, rootMotion, salt);
    GfArena W = A;
    gfMirror(W, rootMotion);

    std::vector<PxgDevicePtr> bufs;
    auto up = [&](const void* p, size_t n) {
        PxgDevicePtr dp = 0;
        shim.memAlloc(&dp, n);
        if (dp && p) shim.memcpyHtoD(dp, p, n);
        bufs.push_back(dp);
        return dp;
    };
    PxgDevicePtr dJd = up(A.jointData.data(), A.jointData.size());
    PxgDevicePtr dPar = up(A.parents.data(), A.parents.size() * 4);
    PxgDevicePtr dXf = up(A.xforms.data(), A.xforms.size() * 4);
    PxgDevicePtr dMm = up(A.motion.data(), A.motion.size() * 4);
    PxgDevicePtr dPr = up(A.props.data(), A.props.size() * 4);
    PxgDevicePtr dZa = up(A.za.data(), A.za.size() * 4);
    PxgDevicePtr dIn = up(A.inertia.data(), A.inertia.size() * 4);
    PxgDevicePtr dMv = up(A.mvel.data(), A.mvel.size() * 4);
    PxgDevicePtr dMa = up(A.macc.data(), A.macc.size() * 4);

    for (uint32_t r = 0; r < GF_NB_RECORDS; ++r)
    {
        uint8_t* R = A.rec.data() + size_t(r) * GF_RECORD_SIZE;
        auto put = [&](uint32_t off, uint64_t v) { std::memcpy(R + off, &v, 8); };
        put(GF_R_JOINT_DATA, uint64_t(dJd) + uint64_t(r) * GF_MAX_LINKS * 8u);
        put(GF_R_PARENTS, uint64_t(dPar) + uint64_t(r) * GF_MAX_LINKS * 4u);
        put(GF_R_BODY2WORLDS, uint64_t(dXf) + uint64_t(r) * GF_MAX_LINKS * 28u);
        put(GF_R_MOTION_MATRIX, uint64_t(dMm) + uint64_t(r) * GF_MAX_LINKS * 80u);
        put(GF_R_LINK_PROPS, uint64_t(dPr) + uint64_t(r) * GF_MAX_LINKS * 16u);
        put(GF_R_ZA, uint64_t(dZa) + uint64_t(r) * GF_MAX_LINKS * 24u);
        put(GF_R_WORLD_INERTIA, uint64_t(dIn) + uint64_t(r) * GF_MAX_LINKS * 112u);
        put(GF_R_MOTION_VEL, uint64_t(dMv) + uint64_t(r) * GF_MAX_LINKS * 24u);
        put(GF_R_MOTION_ACC, uint64_t(dMa) + uint64_t(r) * GF_MAX_LINKS * 24u);
    }
    PxgDevicePtr dRec = up(A.rec.data(), A.rec.size());

    std::vector<uint32_t> idx(GF_NB_SLOTS);
    for (uint32_t j = 0; j < GF_NB_SLOTS; ++j) idx[j] = gfArti(j);
    PxgDevicePtr dIdx = up(idx.data(), idx.size() * 4);
    PxgDevicePtr dOut = up(A.out.data(), A.out.size() * 4);

    bool ok = true;
    for (PxgDevicePtr p : bufs) ok = ok && (p != 0);
    if (!ok)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
        return 0;
    }

    struct Params
    {
        uint32_t nb, pad0;
        uint64_t data, idx;
        uint32_t maxDofs, rootWord;
        uint64_t rec;
        float gx, gy, gz;
        uint32_t pad1;
    } prm = { GF_NB_JOBS, 0u, uint64_t(dOut), uint64_t(dIdx), GF_MAX_DOFS,
              rootMotion ? 1u : 0u, uint64_t(dRec), GF_GRAVITY.x, GF_GRAVITY.y, GF_GRAVITY.z, 0u };
    PXG_CHECK_MSG(shim.launchKernel(0, (GF_NB_JOBS + 7) / 8, 1, 1, &prm, sizeof(prm)) ==
                      PxgResult::Success,
                  "launch computeArtiGravityForces");
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
                    std::printf("MM gf rm=%d %s word %zu  ref %08x (%g)  gpu %08x (%g)\n",
                                rootMotion ? 1 : 0, name, k, w[k], want[k], g[k], got[k]);
                ++bad;
            }
    };
    cmp(dOut, W.out, "forces");
    cmp(dZa, W.za, "zaScratch");
    cmp(dIn, W.inertia, "inertiaScratch");
    cmp(dMv, W.mvel, "motionVel");
    cmp(dMa, W.macc, "motionAcc");

    for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
    return bad;
}

}  // namespace

PXG_TEST(artiGravityForces)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiInverseDyn.spv", "computeArtiGravityForces", 56u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &artiGfLoader);

    // coverage
    uint32_t nFixed = 0, nFree = 0, nStatic = 0;
    for (uint32_t j = 0; j < GF_NB_JOBS; ++j)
    {
        const uint32_t r = gfArti(j);
        if (gfFixBase(r)) ++nFixed; else ++nFree;
        for (uint32_t l = 0; l < gfNumLinks(r); ++l)
            if (gfStatic(r, l)) ++nStatic;
    }
    std::printf("[ info ] gf: %u fixed / %u free jobs, %u static links\n", nFixed, nFree, nStatic);
    PXG_CHECK_MSG(nFixed > 2 && nFree > 2 && nStatic > 0, "coverage hole");

    size_t bad = 0;
    bad += gfRun(shim, true, 3);    // branch A everywhere (rootMotion)
    bad += gfRun(shim, false, 7);   // branch A for fixed bases, branch B for floating
    PXG_CHECK_MSG(bad == 0, "%zu mismatches", bad);
}
