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


// P8.116: the `computeArtiMassMatrices` engine entry
// (inverseDynamic.cu:336) -- the CRB generalized mass matrix of the
// direct API.  The mirror is the CUDA's structure sequentially over engine
// primitives (transformInertiaTensor, PxMat33(q), the pxgref
// skew/translateInertia pair, Dy::SpatialMatrix +/*/invertInertia,
// FeatherstoneArticulation::translateSpatialVector, rotate(PxTransform),
// the dual innerProduct, UnAlignedSpatialVector::operator[]).
//
// Pins beyond the gravity/centrifugal fixtures':
//  * TWO scratch reinterpretations at once: worldSpatialArticulatedInertia
//    as Dy::SpatialMatrix* at stride 112 INCLUDING the padding word at
//    +108 (init writes 0, the level prepass mutates it into level
//    numbers, operator+= leaves it alone -- all three states compared),
//    and worldMotionMatrix as a FLAT UnAlignedSpatialVector* indexed
//    [jointOffset + dof] at stride 24;
//  * the sibling accumulation ORDER (a parent with two children whose
//    translated inertias sum non-associatively -- the level-scheduled
//    order provably equals descending link order, and the mirror does
//    descending);
//  * rootDof vs bufferDof as matrix PITCH vs INDEX (a fixed-base job in
//    the rootMotion round has matSize = dofCount but pitch (maxDofs+6)^2,
//    and only matSize^2 entries of the window are zeroed -- the rest of
//    the pitch must come back poisoned);
//  * the inward walk mutating the wm scratch hop by hop (final state =
//    root-translated I*m vectors), consumed by the !rootMotion free-base
//    correction (-=) over stride dofCount;
//  * the root 6x6 tail's raw-float linear/angular inversion index map;
//  * a single-link free-base articulation (rec 5: the root tail is the
//    whole matrix) and static links (all-zero inertia rows).

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"
#include "pxgArtiRefHelpers.h"

#include "foundation/PxVec3.h"
#include "foundation/PxQuat.h"
#include "foundation/PxTransform.h"
#include "CmUtils.h"

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
static_assert(offsetof(physx::Dy::SpatialMatrix, padding) == 108, "the level-scratch word");
static_assert(offsetof(physx::PxgArticulation, worldMotionMatrix) == 224,
              "..WORLD_MOTION_MATRIX");

static std::vector<uint8_t> artiMmLoader(const char* file) { return loadShaderBlob(file); }

const uint32_t MM_NB_JOBS = 12;
const uint32_t MM_NB_SLOTS = 16;
const uint32_t MM_NB_RECORDS = 18;
const uint32_t MM_RECORD_SIZE = 544;
const uint32_t MM_R_NUM_LINKS = 16, MM_R_NUM_DOFS = 18, MM_R_FLAGS = 26;
const uint32_t MM_R_JOINT_DATA = 48;
const uint32_t MM_R_WORLD_INERTIA = 168, MM_R_BODY2WORLDS = 184, MM_R_PARENTS = 208;
const uint32_t MM_R_MOTION_MATRIX = 216, MM_R_WORLD_MOTION_MATRIX = 224;
const uint32_t MM_R_LINK_PROPS = 256;
const uint32_t MM_MAX_LINKS = 6;
const uint32_t MM_MAX_DOFS = 12;

static uint32_t mmArti(uint32_t job) { return (job * 5u + 1u) % MM_NB_RECORDS; }
// rec 5 is the single-link free-base case (job 8): with rootMotion the root
// tail is the whole 6x6, without it matSize is 0 and the slot stays poison.
static uint32_t mmNumLinks(uint32_t rec) { return rec == 5u ? 1u : 2u + (rec % 5u); }
static bool mmFixBase(uint32_t rec) { return (rec % 3u) == 1u; }
static uint32_t mmDofs(uint32_t rec, uint32_t link) { return 1u + ((rec + link) % 3u); }
// link 1 has children {2, 3} whenever numLinks >= 4: the sibling-order case
static uint32_t mmParent(uint32_t link)
{
    if (link <= 1u) return 0u;
    return (link >= 3u && (link % 3u) == 0u) ? 1u : (link - 1u);
}
static uint32_t mmJointOffset(uint32_t rec, uint32_t link)
{
    uint32_t off = 0;
    for (uint32_t j = 1; j < link; ++j)
        off += mmDofs(rec, j);
    return off;
}
static uint32_t mmDofCount(uint32_t rec) { return mmJointOffset(rec, mmNumLinks(rec)); }
static bool mmStatic(uint32_t rec, uint32_t link) { return (rec % 4u) == 2u && link == 2u; }
static float mmf(uint32_t i, uint32_t k)
{
    return (float((i * 89u + k * 61u) % 547u) + 1.0f) * 0.015625f - 3.25f;
}
static PxQuat mmQuat(uint32_t rec, uint32_t link)
{
    PxQuat q(0.11f + 0.03f * float((rec + link) % 7u), -0.21f + 0.05f * float((rec * 3u + link) % 5u),
             0.36f - 0.04f * float((rec + 2u * link) % 6u), 0.85f);
    return q.getNormalized();
}
static PxVec3 mmPos(uint32_t rec, uint32_t link)
{
    return PxVec3(mmf(rec, 200u + link * 3u), mmf(rec, 201u + link * 3u), mmf(rec, 202u + link * 3u));
}
static physx::PxVec4 mmInvIM(uint32_t rec, uint32_t link)
{
    if (mmStatic(rec, link))
        return physx::PxVec4(0.f, 0.f, 0.f, 0.f);
    return physx::PxVec4(0.1875f + 0.125f * float((rec + link) % 5u),
                         0.4375f + 0.0625f * float((rec * 2u + link) % 4u),
                         0.8125f + 0.03125f * float((rec + 2u * link) % 6u),
                         0.375f + 0.25f * float((rec + link) % 3u));
}

struct MmArena
{
    std::vector<uint8_t> rec, jointData;
    std::vector<uint32_t> parents;
    std::vector<float> xforms, motion, props;
    std::vector<float> inertia, wms;
    std::vector<float> out;
    uint32_t bufferDof;
};

static void mmSeed(MmArena& A, bool rootMotion, uint32_t salt)
{
    A.bufferDof = rootMotion ? 6u : 0u;
    A.rec.assign(size_t(MM_NB_RECORDS) * MM_RECORD_SIZE, 0xAB);
    A.jointData.assign(size_t(MM_NB_RECORDS) * MM_MAX_LINKS * 8, 0xAB);
    A.parents.assign(size_t(MM_NB_RECORDS) * MM_MAX_LINKS, 0xCDCDCDCDu);
    A.xforms.assign(size_t(MM_NB_RECORDS) * MM_MAX_LINKS * 7, -88.f);
    A.motion.assign(size_t(MM_NB_RECORDS) * MM_MAX_LINKS * 20, -77.f);
    A.props.assign(size_t(MM_NB_RECORDS) * MM_MAX_LINKS * 4, -66.f);
    A.inertia.assign(size_t(MM_NB_RECORDS) * MM_MAX_LINKS * 28, 0.f);
    A.wms.assign(size_t(MM_NB_RECORDS) * MM_MAX_DOFS * 6, 0.f);
    std::memset(A.inertia.data(), 0xAB, A.inertia.size() * 4);
    std::memset(A.wms.data(), 0xAB, A.wms.size() * 4);
    const uint32_t pitch = MM_MAX_DOFS + A.bufferDof;
    A.out.assign(size_t(MM_NB_SLOTS) * pitch * pitch, 0.f);
    std::memset(A.out.data(), 0xAB, A.out.size() * 4);

    for (uint32_t r = 0; r < MM_NB_RECORDS; ++r)
    {
        const uint32_t nl = mmNumLinks(r);
        const uint16_t nl16 = uint16_t(nl), nd16 = uint16_t(mmDofCount(r));
        uint8_t* R = A.rec.data() + size_t(r) * MM_RECORD_SIZE;
        std::memcpy(R + MM_R_NUM_LINKS, &nl16, 2);
        std::memcpy(R + MM_R_NUM_DOFS, &nd16, 2);
        R[MM_R_FLAGS] = uint8_t((mmFixBase(r) ? 1u : 0u) | ((r & 1u) << 1) | 4u);

        for (uint32_t l = 0; l < nl; ++l)
        {
            uint8_t* jd = A.jointData.data() + (size_t(r) * MM_MAX_LINKS + l) * 8;
            const uint32_t jo = (l == 0) ? 0u : mmJointOffset(r, l);
            std::memcpy(jd, &jo, 4);
            jd[4] = uint8_t((l == 0) ? 0u : mmDofs(r, l));
            A.parents[size_t(r) * MM_MAX_LINKS + l] = (l == 0) ? 0u : mmParent(l);
            float* xf = &A.xforms[(size_t(r) * MM_MAX_LINKS + l) * 7];
            const PxQuat q = mmQuat(r, l);
            xf[0] = q.x; xf[1] = q.y; xf[2] = q.z; xf[3] = q.w;
            const PxVec3 p = mmPos(r, l);
            xf[4] = p.x; xf[5] = p.y; xf[6] = p.z;
            const physx::PxVec4 im = mmInvIM(r, l);
            float* pp = &A.props[(size_t(r) * MM_MAX_LINKS + l) * 4];
            pp[0] = im.x; pp[1] = im.y; pp[2] = im.z; pp[3] = im.w;
            for (uint32_t d = 0; d < 3; ++d)
                for (uint32_t c = 0; c < 6; ++c)
                    A.motion[(size_t(r) * MM_MAX_LINKS + l) * 20 + d * 6 + c] =
                        mmf(r, 400u + l * 20u + d * 6u + c + salt);
        }
    }
}

static UnAlignedSpatialVector mmLoadSV(const float* p)
{
    return UnAlignedSpatialVector(PxVec3(p[0], p[1], p[2]), PxVec3(p[3], p[4], p[5]));
}
static void mmStoreSV(float* p, const UnAlignedSpatialVector& v)
{
    p[0] = v.top.x; p[1] = v.top.y; p[2] = v.top.z;
    p[3] = v.bottom.x; p[4] = v.bottom.y; p[5] = v.bottom.z;
}
// 108 bytes: the padding word at +108 is handled EXPLICITLY by the mirror
static physx::Dy::SpatialMatrix mmLoadSM(const float* p)
{
    physx::Dy::SpatialMatrix m;
    std::memcpy(&m, p, 108);
    return m;
}
static void mmStoreSM(float* p, const physx::Dy::SpatialMatrix& m) { std::memcpy(p, &m, 108); }
static uint32_t mmGetPad(const float* base, uint32_t link)
{
    uint32_t v;
    std::memcpy(&v, base + link * 28 + 27, 4);
    return v;
}
static void mmSetPad(float* base, uint32_t link, uint32_t v)
{
    std::memcpy(base + link * 28 + 27, &v, 4);
}

static void mmMirror(MmArena& W, bool rootMotion)
{
    using physx::Dy::FeatherstoneArticulation;
    const uint32_t pitch = MM_MAX_DOFS + W.bufferDof;
    for (uint32_t job = 0; job < MM_NB_JOBS; ++job)
    {
        const uint32_t r = mmArti(job);
        const uint32_t linkCount = mmNumLinks(r);
        const uint32_t dofCount = mmDofCount(r);
        const bool fixBase = mmFixBase(r);
        const uint32_t rootDof = (rootMotion && !fixBase) ? 6u : 0u;
        const uint32_t matSize = dofCount + rootDof;
        float* out = &W.out[size_t(job) * pitch * pitch];
        float* inr = &W.inertia[size_t(r) * MM_MAX_LINKS * 28];
        float* wms = &W.wms[size_t(r) * MM_MAX_DOFS * 6];
        auto b2w = [&](uint32_t l) {
            const float* xf = &W.xforms[(size_t(r) * MM_MAX_LINKS + l) * 7];
            return PxTransform(PxVec3(xf[4], xf[5], xf[6]), PxQuat(xf[0], xf[1], xf[2], xf[3]));
        };
        auto jointOff = [&](uint32_t l) { return mmJointOffset(r, l); };
        auto nbDof = [&](uint32_t l) { return mmDofs(r, l); };
        auto motionVec = [&](uint32_t l, uint32_t d) {
            const float* mm = &W.motion[(size_t(r) * MM_MAX_LINKS + l) * 20 + d * 6];
            return UnAlignedSpatialVector(PxVec3(mm[0], mm[1], mm[2]), PxVec3(mm[3], mm[4], mm[5]));
        };

        // S0: the matSize^2 window is zeroed; the rest of the pitch stays
        for (uint32_t i = 0; i < matSize * matSize; ++i)
            out[i] = 0.f;

        // S1: per-link spatial inertias -- the struct copy includes padding = 0
        for (uint32_t l = 0; l < linkCount; ++l)
        {
            const physx::PxVec4 im = mmInvIM(r, l);
            const float mass = im.w == 0.f ? 0.f : 1.f / im.w;
            const PxVec3 localInertia(im.x == 0.f ? 0.f : 1.f / im.x,
                                      im.y == 0.f ? 0.f : 1.f / im.y,
                                      im.z == 0.f ? 0.f : 1.f / im.z);
            physx::Dy::SpatialMatrix sm;
            sm.topLeft = PxMat33(physx::PxZero);
            sm.topRight = PxMat33::createDiagonal(PxVec3(mass));
            physx::Cm::transformInertiaTensor(localInertia, PxMat33(b2w(l).q), sm.bottomLeft);
            mmStoreSM(&inr[l * 28], sm);
            mmSetPad(inr, l, 0u);
        }

        // S2a: the level prepass -- the padding words are observable state
        for (uint32_t l = linkCount - 1; l > 0; --l)
        {
            const uint32_t pl = mmParent(l);
            uint32_t s = mmGetPad(inr, l), ps = mmGetPad(inr, pl);
            if (s < ps) { s = ps; mmSetPad(inr, l, s); }
            mmSetPad(inr, pl, s + 1u);
        }
        // S2b: the accumulation in descending link order (== the level order)
        for (uint32_t l = linkCount - 1; l > 0; --l)
        {
            const uint32_t pl = mmParent(l);
            const PxVec3 rw = b2w(l).p - b2w(pl).p;
            physx::Dy::SpatialMatrix parentSpace = mmLoadSM(&inr[l * 28]);
            pxgtest::refTranslateInertia(pxgtest::refSkewSymmetric(rw.x, rw.y, rw.z), parentSpace);
            mmStoreSM(&inr[pl * 28], mmLoadSM(&inr[pl * 28]) + parentSpace);
        }

        // S3: the per-link bands + inward walks
        for (uint32_t link = 1; link < linkCount; ++link)
        {
            const uint32_t jo = jointOff(link);
            const uint32_t nd = nbDof(link);
            const physx::Dy::SpatialMatrix I = mmLoadSM(&inr[link * 28]);

            UnAlignedSpatialVector wm[3];
            for (uint32_t d = 0; d < nd; ++d)
            {
                wm[d] = motionVec(link, d).rotate(b2w(link));
                mmStoreSV(&wms[(jo + d) * 6], I * wm[d]);
            }
            for (uint32_t dof0 = 0; dof0 < nd; ++dof0)
            {
                const UnAlignedSpatialVector Im0 = mmLoadSV(&wms[(jo + dof0) * 6]);
                const uint32_t row = jo + dof0 + rootDof;
                for (uint32_t dof1 = 0; dof1 < nd; ++dof1)
                    out[row * matSize + (jo + dof1 + rootDof)] = wm[dof1].innerProduct(Im0);
            }

            PxVec3 childP = b2w(link).p;
            uint32_t pl = mmParent(link);
            for (;;)
            {
                const PxVec3 rw = childP - b2w(pl).p;
                for (uint32_t d = 0; d < nd; ++d)
                    mmStoreSV(&wms[(jo + d) * 6],
                              FeatherstoneArticulation::translateSpatialVector(
                                  rw, mmLoadSV(&wms[(jo + d) * 6])));
                childP = b2w(pl).p;

                if (pl == 0)
                {
                    if (!fixBase && rootMotion)
                    {
                        for (uint32_t dof0 = 0; dof0 < nd; ++dof0)
                        {
                            const UnAlignedSpatialVector Im0 = mmLoadSV(&wms[(jo + dof0) * 6]);
                            const uint32_t col = jo + dof0 + rootDof;
                            for (uint32_t row = 0; row < 6; ++row)
                            {
                                out[col * matSize + row] += Im0[row];
                                out[col + row * matSize] += Im0[row];
                            }
                        }
                    }
                    break;
                }

                const uint32_t pjo = jointOff(pl);
                const uint32_t pnd = nbDof(pl);
                UnAlignedSpatialVector pwm[3];
                for (uint32_t d = 0; d < pnd; ++d)
                {
                    const UnAlignedSpatialVector m = motionVec(pl, d);
                    pwm[d].top = b2w(pl).rotate(m.top);
                    pwm[d].bottom = b2w(pl).rotate(m.bottom);
                }
                for (uint32_t dof0 = 0; dof0 < nd; ++dof0)
                {
                    const UnAlignedSpatialVector Im0 = mmLoadSV(&wms[(jo + dof0) * 6]);
                    const uint32_t row = jo + dof0 + rootDof;
                    for (uint32_t dof1 = 0; dof1 < pnd; ++dof1)
                    {
                        const uint32_t col = pjo + dof1 + rootDof;
                        const float m = pwm[dof1].innerProduct(Im0);
                        out[row * matSize + col] = m;
                        out[row + col * matSize] = m;
                    }
                }
                pl = mmParent(pl);
            }
        }

        // the rootMotion root 6x6 tail: raw floats + the inversion index map
        if (!fixBase && rootMotion)
        {
            const float* rootI = &inr[0];
            for (uint32_t row = 0; row < 6; ++row)
            {
                const uint32_t rowSI = (row < 3) ? row : row - 3;
                for (uint32_t col = row; col < 6; ++col)
                {
                    const uint32_t offset = (row > 2) ? 18u : ((col < 3) ? 9u : 0u);
                    const uint32_t colSI = (col < 3) ? col : col - 3;
                    const float v = rootI[offset + colSI * 3 + rowSI];
                    out[row * matSize + col] = v;
                    out[col * matSize + row] = v;
                }
            }
        }
        // the !rootMotion free-base correction over the root-translated scratch
        else if (!fixBase && !rootMotion)
        {
            physx::Dy::SpatialMatrix baseInvI = mmLoadSM(&inr[0]).invertInertia();
            for (uint32_t row = 0; row < dofCount; ++row)
            {
                const UnAlignedSpatialVector m0 = mmLoadSV(&wms[row * 6]);
                for (uint32_t col = 0; col < dofCount; ++col)
                {
                    const UnAlignedSpatialVector m1 = mmLoadSV(&wms[col * 6]);
                    out[row * dofCount + col] -= m0.innerProduct(baseInvI * m1);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------

static size_t mmRun(PxgCudaShim& shim, bool rootMotion, uint32_t salt)
{
    MmArena A;
    mmSeed(A, rootMotion, salt);
    MmArena W = A;
    mmMirror(W, rootMotion);

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
    PxgDevicePtr dIn = up(A.inertia.data(), A.inertia.size() * 4);
    PxgDevicePtr dWm = up(A.wms.data(), A.wms.size() * 4);

    for (uint32_t r = 0; r < MM_NB_RECORDS; ++r)
    {
        uint8_t* R = A.rec.data() + size_t(r) * MM_RECORD_SIZE;
        auto put = [&](uint32_t off, uint64_t v) { std::memcpy(R + off, &v, 8); };
        put(MM_R_JOINT_DATA, uint64_t(dJd) + uint64_t(r) * MM_MAX_LINKS * 8u);
        put(MM_R_PARENTS, uint64_t(dPar) + uint64_t(r) * MM_MAX_LINKS * 4u);
        put(MM_R_BODY2WORLDS, uint64_t(dXf) + uint64_t(r) * MM_MAX_LINKS * 28u);
        put(MM_R_MOTION_MATRIX, uint64_t(dMm) + uint64_t(r) * MM_MAX_LINKS * 80u);
        put(MM_R_LINK_PROPS, uint64_t(dPr) + uint64_t(r) * MM_MAX_LINKS * 16u);
        put(MM_R_WORLD_INERTIA, uint64_t(dIn) + uint64_t(r) * MM_MAX_LINKS * 112u);
        put(MM_R_WORLD_MOTION_MATRIX, uint64_t(dWm) + uint64_t(r) * MM_MAX_DOFS * 24u);
    }
    PxgDevicePtr dRec = up(A.rec.data(), A.rec.size());

    std::vector<uint32_t> idx(MM_NB_SLOTS);
    for (uint32_t j = 0; j < MM_NB_SLOTS; ++j) idx[j] = mmArti(j);
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
    } prm = { MM_NB_JOBS, 0u, uint64_t(dOut), uint64_t(dIdx), MM_MAX_DOFS,
              rootMotion ? 1u : 0u, uint64_t(dRec) };
    PXG_CHECK_MSG(shim.launchKernel(0, (MM_NB_JOBS + 7) / 8, 1, 1, &prm, sizeof(prm)) ==
                      PxgResult::Success,
                  "launch computeArtiMassMatrices");
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
                    std::printf("MM mm rm=%d %s word %zu  ref %08x (%g)  gpu %08x (%g)\n",
                                rootMotion ? 1 : 0, name, k, w[k], want[k], g[k], got[k]);
                ++bad;
            }
    };
    cmp(dOut, W.out, "massMatrix");
    cmp(dIn, W.inertia, "inertiaScratch");
    cmp(dWm, W.wms, "wmScratch");

    for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
    return bad;
}

}  // namespace

PXG_TEST(artiMassMatrices)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiInverseDyn.spv", "computeArtiMassMatrices", 40u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &artiMmLoader);

    uint32_t nFixed = 0, nFree = 0, nMultiChild = 0, nSingle = 0;
    for (uint32_t j = 0; j < MM_NB_JOBS; ++j)
    {
        const uint32_t r = mmArti(j);
        if (mmFixBase(r)) ++nFixed; else ++nFree;
        if (mmNumLinks(r) >= 4) ++nMultiChild;   // link 1 then has children {2,3}
        if (mmNumLinks(r) == 1) ++nSingle;
    }
    std::printf("[ info ] mm: %u fixed / %u free jobs, %u multi-child trees, %u single-link\n",
                nFixed, nFree, nMultiChild, nSingle);
    PXG_CHECK_MSG(nFixed > 2 && nFree > 2 && nMultiChild > 3 && nSingle > 0, "coverage hole");

    size_t bad = 0;
    bad += mmRun(shim, true, 5);
    bad += mmRun(shim, false, 9);
    PXG_CHECK_MSG(bad == 0, "%zu mismatches", bad);
}
