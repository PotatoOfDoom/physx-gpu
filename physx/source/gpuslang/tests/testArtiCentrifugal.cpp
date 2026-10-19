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


// P8.115: the `computeArtiCentrifugalForces` engine entry
// (inverseDynamic.cu:803) -- the coriolis/centrifugal twin of the gravity
// kernel.  The mirror is the CUDA's structure sequentially over engine
// primitives (translateSpatialVector, transformInertiaTensor, PxMat33(q),
// the pxgref skew/translateInertia pair, SpatialMatrix algebra,
// rotate(PxTransform), the dual innerProduct).
//
// Pins beyond the gravity fixture's: the corioliseVectors/
// "motionAccelerations" ALIAS (one record pointer, the acceleration chain
// overwriting the coriolis vectors slot by slot in exactly the order that
// makes the alias sound), the upper-only jVel clamp against
// maxJointVelocity (some seeded joint velocities EXCEED their cap, some are
// NEGATIVE and pass through untouched), motionVelocities[1..] as an
// OBSERVABLE OUTPUT (link 0 a live input), the crossed zA halves, the
// lane-0 staging.  5.12: the rootMotion argument is gone (rootDof = fixBase ?
// 0 : 6, output pitch maxDofs + 6, no root-acceleration recompute: the
// floating-base pass re-propagates the ZERO root acceleration), and every Rw
// comes from computeLinkRwInvDyn (joint anchors + the prismatic slide).

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"
#include "pxgArtiRefHelpers.h"

#include "foundation/PxVec3.h"
#include "foundation/PxQuat.h"
#include "foundation/PxTransform.h"
#include "CmUtils.h"
#include "pxgArtiLinkRwRef.h"

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

static std::vector<uint8_t> artiCfLoader(const char* file) { return loadShaderBlob(file); }

const uint32_t CF_NB_JOBS = 12;
const uint32_t CF_NB_SLOTS = 16;
const uint32_t CF_NB_RECORDS = 18;
const uint32_t CF_RECORD_SIZE = 496;
const uint32_t CF_R_NUM_LINKS = 16, CF_R_NUM_DOFS = 18, CF_R_FLAGS = 26;
const uint32_t CF_R_JOINT_POSITIONS = 104;   // 5.12: the prismatic slide
const uint32_t CF_R_JOINTS = 40, CF_R_JOINT_DATA = 48, CF_R_MOTION_VEL = 56;
const uint32_t CF_R_CORIOLIS = 80, CF_R_ZA = 88, CF_R_JOINT_VELOCITIES = 112;
const uint32_t CF_R_WORLD_INERTIA = 160, CF_R_BODY2WORLDS = 176, CF_R_PARENTS = 200;
const uint32_t CF_R_MOTION_MATRIX = 208, CF_R_LINK_PROPS = 240;
const uint32_t CF_JOINT_CORE_SIZE = 512;
const uint32_t CF_JOINT_MAX_VEL = 464;   // PXG_AR_JOINT_maxJointVelocity (5.12)
const uint32_t CF_MAX_LINKS = 6;
const uint32_t CF_MAX_DOFS = 12;

static uint32_t cfArti(uint32_t job) { return (job * 5u + 1u) % CF_NB_RECORDS; }
static uint32_t cfNumLinks(uint32_t rec) { return 2u + (rec % 5u); }
static bool cfFixBase(uint32_t rec) { return (rec % 3u) == 1u; }
static uint32_t cfDofs(uint32_t rec, uint32_t link) { return 1u + ((rec + link) % 3u); }
static uint32_t cfParent(uint32_t link)
{
    if (link <= 1u) return 0u;
    return (link >= 3u && (link % 3u) == 0u) ? 1u : (link - 1u);
}
static uint32_t cfJointOffset(uint32_t rec, uint32_t link)
{
    uint32_t off = 0;
    for (uint32_t j = 1; j < link; ++j)
        off += cfDofs(rec, j);
    return off;
}
static uint32_t cfDofCount(uint32_t rec) { return cfJointOffset(rec, cfNumLinks(rec)); }
static bool cfStatic(uint32_t rec, uint32_t link) { return (rec % 4u) == 2u && link == 2u; }
static float cf(uint32_t i, uint32_t k)
{
    return (float((i * 97u + k * 53u) % 541u) + 1.0f) * 0.015625f - 3.5f;
}
static PxQuat cfQuat(uint32_t rec, uint32_t link)
{
    PxQuat q(0.14f + 0.02f * float((rec + link) % 8u), -0.24f + 0.04f * float((rec * 3u + link) % 6u),
             0.33f - 0.05f * float((rec + 2u * link) % 5u), 0.88f);
    return q.getNormalized();
}
static PxVec3 cfPos(uint32_t rec, uint32_t link)
{
    return PxVec3(cf(rec, 100u + link * 3u), cf(rec, 101u + link * 3u), cf(rec, 102u + link * 3u));
}
static physx::PxVec4 cfInvIM(uint32_t rec, uint32_t link)
{
    if (cfStatic(rec, link))
        return physx::PxVec4(0.f, 0.f, 0.f, 0.f);
    return physx::PxVec4(0.25f + 0.125f * float((rec + link) % 5u),
                         0.5f + 0.0625f * float((rec * 2u + link) % 4u),
                         0.75f + 0.03125f * float((rec + 2u * link) % 6u),
                         0.5f + 0.25f * float((rec + link) % 3u));
}
// caps: modest positive values; velocities cycle above/below/negative so the
// upper-only clamp shows in all three regimes
static float cfMaxVel(uint32_t rec, uint32_t link, uint32_t dof)
{
    return 1.0f + 0.5f * float((rec + link + dof) % 4u);
}
static float cfJointVel(uint32_t rec, uint32_t gdof)
{
    const uint32_t m = (rec + gdof) % 3u;
    if (m == 0u) return 4.0f + 0.25f * float(gdof);    // above the cap
    if (m == 1u) return 0.5f + 0.125f * float(gdof);   // below
    return -2.0f - 0.25f * float(gdof);                // negative, untouched
}

struct CfArena
{
    std::vector<uint8_t> rec, jointData, jointCores;
    std::vector<uint32_t> parents;
    std::vector<float> xforms, motion, props, jvel, jointPositions;
    std::vector<float> za, inertia, mvel, coriolis;
    std::vector<float> out;
    uint32_t bufferDof;
};

static void cfSeed(CfArena& A, uint32_t salt)
{
    A.bufferDof = 6u;   // 5.12: always maxDofs + 6
    A.rec.assign(size_t(CF_NB_RECORDS) * CF_RECORD_SIZE, 0xAB);
    A.jointData.assign(size_t(CF_NB_RECORDS) * CF_MAX_LINKS * 8, 0xAB);
    // 5.12: real joint cores (anchors + type for computeLinkRwInvDyn), maxJointVelocity overlaid
    lrBuildJoints(A.jointCores, CF_NB_RECORDS, CF_MAX_LINKS);
    lrBuildJointPositions(A.jointPositions, CF_NB_RECORDS, CF_MAX_DOFS);
    A.parents.assign(size_t(CF_NB_RECORDS) * CF_MAX_LINKS, 0xCDCDCDCDu);
    A.xforms.assign(size_t(CF_NB_RECORDS) * CF_MAX_LINKS * 7, -88.f);
    A.motion.assign(size_t(CF_NB_RECORDS) * CF_MAX_LINKS * 20, -77.f);
    A.props.assign(size_t(CF_NB_RECORDS) * CF_MAX_LINKS * 4, -66.f);
    A.jvel.assign(size_t(CF_NB_RECORDS) * CF_MAX_DOFS, -55.f);
    A.za.assign(size_t(CF_NB_RECORDS) * CF_MAX_LINKS * 6, 0.f);
    A.inertia.assign(size_t(CF_NB_RECORDS) * CF_MAX_LINKS * 28, 0.f);
    A.mvel.assign(size_t(CF_NB_RECORDS) * CF_MAX_LINKS * 6, 0.f);
    A.coriolis.assign(size_t(CF_NB_RECORDS) * CF_MAX_LINKS * 6, 0.f);
    std::memset(A.za.data(), 0xAB, A.za.size() * 4);
    std::memset(A.inertia.data(), 0xAB, A.inertia.size() * 4);
    std::memset(A.mvel.data(), 0xAB, A.mvel.size() * 4);
    std::memset(A.coriolis.data(), 0xAB, A.coriolis.size() * 4);
    A.out.assign(size_t(CF_NB_SLOTS) * (CF_MAX_DOFS + A.bufferDof), 0.f);
    std::memset(A.out.data(), 0xAB, A.out.size() * 4);

    for (uint32_t r = 0; r < CF_NB_RECORDS; ++r)
    {
        const uint32_t nl = cfNumLinks(r);
        const uint16_t nl16 = uint16_t(nl), nd16 = uint16_t(cfDofCount(r));
        uint8_t* R = A.rec.data() + size_t(r) * CF_RECORD_SIZE;
        std::memcpy(R + CF_R_NUM_LINKS, &nl16, 2);
        std::memcpy(R + CF_R_NUM_DOFS, &nd16, 2);
        R[CF_R_FLAGS] = uint8_t((cfFixBase(r) ? 1u : 0u) | ((r & 1u) << 1) | 4u);

        for (uint32_t c = 0; c < 6; ++c)
            A.mvel[(size_t(r) * CF_MAX_LINKS + 0) * 6 + c] = cf(r, 700u + c + salt);
        for (uint32_t d = 0; d < cfDofCount(r); ++d)
            A.jvel[size_t(r) * CF_MAX_DOFS + d] = cfJointVel(r, d);

        for (uint32_t l = 0; l < nl; ++l)
        {
            uint8_t* jd = A.jointData.data() + (size_t(r) * CF_MAX_LINKS + l) * 8;
            const uint32_t jo = (l == 0) ? 0u : cfJointOffset(r, l);
            std::memcpy(jd, &jo, 4);
            jd[4] = uint8_t((l == 0) ? 0u : cfDofs(r, l));
            A.parents[size_t(r) * CF_MAX_LINKS + l] = (l == 0) ? 0u : cfParent(l);
            float* xf = &A.xforms[(size_t(r) * CF_MAX_LINKS + l) * 7];
            const PxQuat q = cfQuat(r, l);
            xf[0] = q.x; xf[1] = q.y; xf[2] = q.z; xf[3] = q.w;
            const PxVec3 p = cfPos(r, l);
            xf[4] = p.x; xf[5] = p.y; xf[6] = p.z;
            const physx::PxVec4 im = cfInvIM(r, l);
            float* pp = &A.props[(size_t(r) * CF_MAX_LINKS + l) * 4];
            pp[0] = im.x; pp[1] = im.y; pp[2] = im.z; pp[3] = im.w;
            for (uint32_t d = 0; d < 3; ++d)
            {
                for (uint32_t c = 0; c < 6; ++c)
                    A.motion[(size_t(r) * CF_MAX_LINKS + l) * 20 + d * 6 + c] =
                        cf(r, 300u + l * 20u + d * 6u + c + salt);
                float mv = cfMaxVel(r, l, d);
                std::memcpy(A.jointCores.data()
                                + (size_t(r) * CF_MAX_LINKS + l) * CF_JOINT_CORE_SIZE
                                + CF_JOINT_MAX_VEL + d * 4,
                            &mv, 4);
            }
        }
    }
}

static UnAlignedSpatialVector cfLoadSV(const float* p)
{
    return UnAlignedSpatialVector(PxVec3(p[0], p[1], p[2]), PxVec3(p[3], p[4], p[5]));
}
static void cfStoreSV(float* p, const UnAlignedSpatialVector& v)
{
    p[0] = v.top.x; p[1] = v.top.y; p[2] = v.top.z;
    p[3] = v.bottom.x; p[4] = v.bottom.y; p[5] = v.bottom.z;
}
static physx::Dy::SpatialMatrix cfLoadSM(const float* p)
{
    physx::Dy::SpatialMatrix m;
    std::memcpy(&m, p, 108);
    return m;
}
static void cfStoreSM(float* p, const physx::Dy::SpatialMatrix& m) { std::memcpy(p, &m, 108); }

static void cfMirror(CfArena& W)
{
    using physx::Dy::FeatherstoneArticulation;
    for (uint32_t job = 0; job < CF_NB_JOBS; ++job)
    {
        const uint32_t r = cfArti(job);
        const uint32_t linkCount = cfNumLinks(r);
        const uint32_t dofCount = cfDofCount(r);
        const bool fixBase = cfFixBase(r);
        const uint32_t rootDof = fixBase ? 0u : 6u;
        float* out = &W.out[size_t(job) * (CF_MAX_DOFS + 6u)];
        float* za = &W.za[size_t(r) * CF_MAX_LINKS * 6];
        float* mvel = &W.mvel[size_t(r) * CF_MAX_LINKS * 6];
        float* cor = &W.coriolis[size_t(r) * CF_MAX_LINKS * 6];   // the ALIAS: one buffer
        float* inr = &W.inertia[size_t(r) * CF_MAX_LINKS * 28];
        const float* jv = &W.jvel[size_t(r) * CF_MAX_DOFS];
        auto b2w = [&](uint32_t l) {
            const float* xf = &W.xforms[(size_t(r) * CF_MAX_LINKS + l) * 7];
            return PxTransform(PxVec3(xf[4], xf[5], xf[6]), PxQuat(xf[0], xf[1], xf[2], xf[3]));
        };
        auto jointOff = [&](uint32_t l) { return cfJointOffset(r, l); };
        auto nbDof = [&](uint32_t l) { return cfDofs(r, l); };
        auto motionVec = [&](uint32_t l, uint32_t d) {
            const float* mm = &W.motion[(size_t(r) * CF_MAX_LINKS + l) * 20 + d * 6];
            return UnAlignedSpatialVector(PxVec3(mm[0], mm[1], mm[2]), PxVec3(mm[3], mm[4], mm[5]));
        };
        // 5.12: computeLinkRwInvDyn
        auto linkRw = [&](uint32_t l) {
            return lrLinkRw(r, l, b2w(cfParent(l)).q, b2w(l).q, jointOff(l), motionVec(l, 0).bottom);
        };

        // S0: velocities + coriolis vectors
        for (uint32_t l = 1; l < linkCount; ++l)
        {
            const uint32_t pl = cfParent(l);
            const PxVec3 rw = linkRw(l);
            const UnAlignedSpatialVector pVel = cfLoadSV(&mvel[pl * 6]);
            UnAlignedSpatialVector vel =
                FeatherstoneArticulation::translateSpatialVector(-rw, pVel);
            UnAlignedSpatialVector deltaV = UnAlignedSpatialVector::Zero();
            for (uint32_t d = 0; d < nbDof(l); ++d)
            {
                const float maxV = cfMaxVel(r, l, d);
                const float v = jv[jointOff(l) + d];
                const float jVel = v < maxV ? v : maxV;
                deltaV += motionVec(l, d).rotate(b2w(l)) * jVel;
            }
            vel.top += deltaV.top;
            vel.bottom += deltaV.bottom;
            const PxVec3 aVec = deltaV.top;
            const PxVec3 force = pVel.top.cross(aVec);
            const PxVec3 lVel = deltaV.bottom;
            const PxVec3 torque = pVel.top.cross(pVel.top.cross(rw))
                                + 2.f * pVel.top.cross(lVel) + aVec.cross(lVel);
            cfStoreSV(&cor[l * 6], UnAlignedSpatialVector(force, torque));
            cfStoreSV(&mvel[l * 6], vel);
        }
        // S1: the acceleration chain through the alias
        cfStoreSV(&cor[0], UnAlignedSpatialVector::Zero());
        for (uint32_t l = 1; l < linkCount; ++l)
        {
            const uint32_t pl = cfParent(l);
            const UnAlignedSpatialVector pAcc =
                FeatherstoneArticulation::translateSpatialVector(-linkRw(l), cfLoadSV(&cor[pl * 6]));
            cfStoreSV(&cor[l * 6], pAcc + cfLoadSV(&cor[l * 6]));
        }
        // S2: zA init with crossed halves
        for (uint32_t l = 0; l < linkCount; ++l)
        {
            const physx::PxVec4 im = cfInvIM(r, l);
            const float mass = im.w == 0.f ? 0.f : 1.f / im.w;
            const PxVec3 localInertia(im.x == 0.f ? 0.f : 1.f / im.x,
                                      im.y == 0.f ? 0.f : 1.f / im.y,
                                      im.z == 0.f ? 0.f : 1.f / im.z);
            PxMat33 I;
            physx::Cm::transformInertiaTensor(localInertia, PxMat33(b2w(l).q), I);
            const PxVec3 vA = cfLoadSV(&mvel[l * 6]).top;
            const UnAlignedSpatialVector acc = cfLoadSV(&cor[l * 6]);
            cfStoreSV(&za[l * 6],
                      UnAlignedSpatialVector(acc.bottom * mass,
                                             vA.cross(I * vA) + I * acc.top));
        }
        // S3: zA accumulation + staging
        for (uint32_t l = linkCount - 1; l > 0; --l)
        {
            const uint32_t pl = cfParent(l);
            const UnAlignedSpatialVector t = FeatherstoneArticulation::translateSpatialVector(
                linkRw(l), cfLoadSV(&za[l * 6]));
            cfStoreSV(&za[pl * 6], cfLoadSV(&za[pl * 6]) + t);
            for (uint32_t d = 0; d < nbDof(l); ++d)
                out[jointOff(l) + d + rootDof] = float(l);
        }
        // S4..S5: the floating-base pass (5.12: no root-acceleration recompute)
        if (!fixBase)
        {
            for (uint32_t l = 0; l < linkCount; ++l)
            {
                const physx::PxVec4 im = cfInvIM(r, l);
                const float mass = im.w == 0.f ? 0.f : 1.f / im.w;
                const PxVec3 localInertia(im.x == 0.f ? 0.f : 1.f / im.x,
                                          im.y == 0.f ? 0.f : 1.f / im.y,
                                          im.z == 0.f ? 0.f : 1.f / im.z);
                physx::Dy::SpatialMatrix sm;
                sm.topLeft = PxMat33(physx::PxZero);
                sm.topRight = PxMat33::createDiagonal(PxVec3(mass));
                physx::Cm::transformInertiaTensor(localInertia, PxMat33(b2w(l).q), sm.bottomLeft);
                cfStoreSM(&inr[l * 28], sm);
            }
            for (uint32_t l = linkCount - 1; l > 0; --l)
            {
                const uint32_t pl = cfParent(l);
                const PxVec3 rw = linkRw(l);
                physx::Dy::SpatialMatrix parentSpace = cfLoadSM(&inr[l * 28]);
                pxgtest::refTranslateInertia(pxgtest::refSkewSymmetric(rw.x, rw.y, rw.z),
                                             parentSpace);
                cfStoreSM(&inr[pl * 28], dySmAdd(cfLoadSM(&inr[pl * 28]), parentSpace));
            }
            for (uint32_t l = 1; l < linkCount; ++l)
            {
                const uint32_t pl = cfParent(l);
                const UnAlignedSpatialVector acc = FeatherstoneArticulation::translateSpatialVector(
                    -linkRw(l), cfLoadSV(&cor[pl * 6]));
                cfStoreSV(&cor[l * 6], acc);
                cfStoreSV(&za[l * 6], cfLoadSM(&inr[l * 28]) * acc + cfLoadSV(&za[l * 6]));
            }
        }
        // S7: the projection + the root tail
        for (uint32_t dof = 0; dof < dofCount; ++dof)
        {
            const uint32_t l = uint32_t(out[dof + rootDof]);
            const UnAlignedSpatialVector dm = motionVec(l, dof - jointOff(l)).rotate(b2w(l));
            out[dof + rootDof] = dm.innerProduct(cfLoadSV(&za[l * 6]));
        }
        if (!fixBase)
        {
            const UnAlignedSpatialVector z0 = cfLoadSV(&za[0]);
            out[0] = z0.top.x; out[1] = z0.top.y; out[2] = z0.top.z;
            out[3] = z0.bottom.x; out[4] = z0.bottom.y; out[5] = z0.bottom.z;
        }
    }
}

// ---------------------------------------------------------------------------

static size_t cfRun(PxgCudaShim& shim, uint32_t salt)
{
    CfArena A;
    cfSeed(A, salt);
    CfArena W = A;
    cfMirror(W);

    std::vector<PxgDevicePtr> bufs;
    auto up = [&](const void* p, size_t n) {
        PxgDevicePtr dp = 0;
        shim.memAlloc(&dp, n);
        if (dp && p) shim.memcpyHtoD(dp, p, n);
        bufs.push_back(dp);
        return dp;
    };
    PxgDevicePtr dJd = up(A.jointData.data(), A.jointData.size());
    PxgDevicePtr dJc = up(A.jointCores.data(), A.jointCores.size());
    PxgDevicePtr dPar = up(A.parents.data(), A.parents.size() * 4);
    PxgDevicePtr dXf = up(A.xforms.data(), A.xforms.size() * 4);
    PxgDevicePtr dMm = up(A.motion.data(), A.motion.size() * 4);
    PxgDevicePtr dPr = up(A.props.data(), A.props.size() * 4);
    PxgDevicePtr dJv = up(A.jvel.data(), A.jvel.size() * 4);
    PxgDevicePtr dZa = up(A.za.data(), A.za.size() * 4);
    PxgDevicePtr dIn = up(A.inertia.data(), A.inertia.size() * 4);
    PxgDevicePtr dMv = up(A.mvel.data(), A.mvel.size() * 4);
    PxgDevicePtr dCo = up(A.coriolis.data(), A.coriolis.size() * 4);
    PxgDevicePtr dJp = up(A.jointPositions.data(), A.jointPositions.size() * 4);

    for (uint32_t r = 0; r < CF_NB_RECORDS; ++r)
    {
        uint8_t* R = A.rec.data() + size_t(r) * CF_RECORD_SIZE;
        auto put = [&](uint32_t off, uint64_t v) { std::memcpy(R + off, &v, 8); };
        put(CF_R_JOINT_DATA, uint64_t(dJd) + uint64_t(r) * CF_MAX_LINKS * 8u);
        put(CF_R_JOINTS, uint64_t(dJc) + uint64_t(r) * CF_MAX_LINKS * CF_JOINT_CORE_SIZE);
        put(CF_R_PARENTS, uint64_t(dPar) + uint64_t(r) * CF_MAX_LINKS * 4u);
        put(CF_R_BODY2WORLDS, uint64_t(dXf) + uint64_t(r) * CF_MAX_LINKS * 28u);
        put(CF_R_MOTION_MATRIX, uint64_t(dMm) + uint64_t(r) * CF_MAX_LINKS * 80u);
        put(CF_R_LINK_PROPS, uint64_t(dPr) + uint64_t(r) * CF_MAX_LINKS * 16u);
        put(CF_R_JOINT_VELOCITIES, uint64_t(dJv) + uint64_t(r) * CF_MAX_DOFS * 4u);
        put(CF_R_ZA, uint64_t(dZa) + uint64_t(r) * CF_MAX_LINKS * 24u);
        put(CF_R_WORLD_INERTIA, uint64_t(dIn) + uint64_t(r) * CF_MAX_LINKS * 112u);
        put(CF_R_MOTION_VEL, uint64_t(dMv) + uint64_t(r) * CF_MAX_LINKS * 24u);
        put(CF_R_CORIOLIS, uint64_t(dCo) + uint64_t(r) * CF_MAX_LINKS * 24u);
        put(CF_R_JOINT_POSITIONS, uint64_t(dJp) + uint64_t(r) * CF_MAX_DOFS * 4u);
    }
    PxgDevicePtr dRec = up(A.rec.data(), A.rec.size());

    std::vector<uint32_t> idx(CF_NB_SLOTS);
    for (uint32_t j = 0; j < CF_NB_SLOTS; ++j) idx[j] = cfArti(j);
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
        uint32_t maxDofs, pad1;   // (5.6.1: the rootMotion bool sat in pad1)
        uint64_t rec;
    } prm = { CF_NB_JOBS, 0u, uint64_t(dOut), uint64_t(dIdx), CF_MAX_DOFS,
              0xABu, uint64_t(dRec) };
    PXG_CHECK_MSG(shim.launchKernel(0, (CF_NB_JOBS + 7) / 8, 1, 1, &prm, sizeof(prm)) ==
                      PxgResult::Success,
                  "launch computeArtiCentrifugalForces");
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
                    std::printf("MM cf salt=%u %s word %zu  ref %08x (%g)  gpu %08x (%g)\n",
                                salt, name, k, w[k], want[k], g[k], got[k]);
                ++bad;
            }
    };
    cmp(dOut, W.out, "forces");
    cmp(dMv, W.mvel, "motionVel");
    cmp(dCo, W.coriolis, "coriolisAlias");
    cmp(dZa, W.za, "zaScratch");
    cmp(dIn, W.inertia, "inertiaScratch");

    for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
    return bad;
}

}  // namespace

PXG_TEST(artiCentrifugal)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiInverseDyn.spv", "computeArtiCentrifugalForces", 40u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &artiCfLoader);

    uint32_t nFixed = 0, nFree = 0, nAbove = 0, nNeg = 0;
    for (uint32_t j = 0; j < CF_NB_JOBS; ++j)
    {
        const uint32_t r = cfArti(j);
        if (cfFixBase(r)) ++nFixed; else ++nFree;
        for (uint32_t d = 0; d < cfDofCount(r); ++d)
        {
            const uint32_t m = (r + d) % 3u;
            if (m == 0u) ++nAbove;
            if (m == 2u) ++nNeg;
        }
    }
    std::printf("[ info ] cf: %u fixed / %u free jobs, %u over-cap dofs, %u negative dofs\n",
                nFixed, nFree, nAbove, nNeg);
    PXG_CHECK_MSG(nFixed > 2 && nFree > 2 && nAbove > 3 && nNeg > 3, "coverage hole");

    size_t bad = 0;
    bad += cfRun(shim, 3);
    bad += cfRun(shim, 7);
    PXG_CHECK_MSG(bad == 0, "%zu mismatches", bad);
}
