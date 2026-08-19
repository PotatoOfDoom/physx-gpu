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


// P8.118: the nine non-tendon direct-API state get/set kernels
// (articulationDirectGpuApi.cu:350-819) in npArtiGetSet.slang.  Pure
// gather/scatter -- every comparison is BIT-EXACT (the only arithmetic is
// quat multiply/rotate and one componentwise scale; no transcendentals).
//
// Pins: the six-read / five-write enum-to-array maps (their orders DIFFER),
// the max-pitch vs real-count guards (dead pitch slots stay poison), the
// per-type dirty-bit sets (positions/velocities also raise
// eNEEDS_KINEMATIC_UPDATE, forces/targets do not; each set by exactly one
// thread), the ROOT_GLOBAL_POSE numLinks clamp with an unclamped output
// pitch, body2World * body2Actor.getInverse() vs actorPose * body2Actor
// (get vs set -- one inverts, the other does not), the angular(+0) vs
// linear(+3) spatial-vector halves at stride 6, externalAccelerations
// .top = force * invMass vs .bottom = the rotateInv/scale/rotate torque
// sandwich, and the launch overhang (grid 2 blocks of 512/480: every
// overhang groupIndex resolves through a 1024-entry index buffer onto four
// GUARD records whose dirty words and outputs must come back untouched --
// the P8.113 lesson applied to a 512-wide block).

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "foundation/PxVec3.h"
#include "foundation/PxQuat.h"
#include "foundation/PxTransform.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::PxQuat;
using physx::PxTransform;
using physx::PxVec3;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

static std::vector<uint8_t> artiGsLoader(const char* file) { return loadShaderBlob(file); }

const uint32_t GS_NB_LIVE = 12;
const uint32_t GS_NB_RECORDS = 16;   // + 4 guard records for the launch overhang
const uint32_t GS_MAX_LINKS = 6;
const uint32_t GS_MAX_DOFS = 12;
const uint32_t GS_RECORD_SIZE = 544;
const uint32_t GS_IDX_ENTRIES = 1024;   // covers every overhang groupIndex at grid 2
const uint32_t GS_MAX_SPT = 2;    // maxSpatialTendonsPerArticulation (desc @260)
const uint32_t GS_MAX_ATT = 3;    // maxAttachmentPerArticulation (desc @264)
const uint32_t GS_MAX_FT = 2;     // maxFixedTendonsPerArticulation (desc @268)
const uint32_t GS_MAX_TJ = 3;     // maxTendonJointPerArticulation (desc @272)

static uint32_t gsNumLinks(uint32_t r) { return r == 5u ? 1u : 2u + (r % 5u); }
static uint32_t gsDofCount(uint32_t r) { return 1u + ((r * 7u) % (GS_MAX_DOFS - 1u)); }
static float gsf(uint32_t i, uint32_t k)
{
    return (float((i * 79u + k * 67u) % 519u) + 1.0f) * 0.015625f - 3.5f;
}
static PxQuat gsQuat(uint32_t i, uint32_t k)
{
    PxQuat q(0.12f + 0.04f * float((i + k) % 6u), -0.22f + 0.06f * float((i * 3u + k) % 5u),
             0.35f - 0.05f * float((i + 2u * k) % 7u), 0.88f);
    return q.getNormalized();
}
static uint32_t gsDirtyWord(uint32_t r) { return 0x40000u | (r << 24) | 8u; }
static uint32_t gsNumSpt(uint32_t r) { return r % 3u; }        // 0..2 spatial tendons
static uint32_t gsNumFt(uint32_t r) { return (r + 1u) % 3u; }  // 0..2 fixed tendons
static uint32_t gsNbAtt(uint32_t r, uint32_t t) { return 1u + ((r + t) % GS_MAX_ATT); }
static uint32_t gsNbTj(uint32_t r, uint32_t t) { return 1u + ((r * 2u + t) % GS_MAX_TJ); }

struct GsArena
{
    std::vector<uint8_t> rec;
    std::vector<float> dofArr[6];   // jpos jvel jacc jforce jtvel jtpos
    std::vector<float> mvel, macc, incoming, xw, xa, props, extAccel;
    std::vector<float> sptParams, ftParams;         // 16 B / 32 B per tendon
    std::vector<uint8_t> sptTendons, ftTendons;     // PxgArticulationTendon images, 24 B
    std::vector<float> attMod, tjMod;               // mModElements blocks, 32 B / 16 B
    std::vector<uint32_t> idx;
};

static void gsSeed(GsArena& A, uint32_t salt)
{
    A.rec.assign(size_t(GS_NB_RECORDS) * GS_RECORD_SIZE, 0xAB);
    for (int a = 0; a < 6; ++a)
        A.dofArr[a].assign(size_t(GS_NB_RECORDS) * GS_MAX_DOFS, 0.f);
    A.mvel.assign(size_t(GS_NB_RECORDS) * GS_MAX_LINKS * 6, 0.f);
    A.macc.assign(size_t(GS_NB_RECORDS) * GS_MAX_LINKS * 6, 0.f);
    A.incoming.assign(size_t(GS_NB_RECORDS) * GS_MAX_LINKS * 6, 0.f);
    A.xw.assign(size_t(GS_NB_RECORDS) * GS_MAX_LINKS * 7, 0.f);
    A.xa.assign(size_t(GS_NB_RECORDS) * GS_MAX_LINKS * 7, 0.f);
    A.props.assign(size_t(GS_NB_RECORDS) * GS_MAX_LINKS * 4, 0.f);
    A.extAccel.assign(size_t(GS_NB_RECORDS) * GS_MAX_LINKS * 6, 0.f);
    std::memset(A.extAccel.data(), 0xAB, A.extAccel.size() * 4);
    A.sptParams.assign(size_t(GS_NB_RECORDS) * GS_MAX_SPT * 4, 0.f);
    A.ftParams.assign(size_t(GS_NB_RECORDS) * GS_MAX_FT * 8, 0.f);
    A.sptTendons.assign(size_t(GS_NB_RECORDS) * GS_MAX_SPT * 24, 0xAB);
    A.ftTendons.assign(size_t(GS_NB_RECORDS) * GS_MAX_FT * 24, 0xAB);
    A.attMod.assign(size_t(GS_NB_RECORDS) * GS_MAX_SPT * GS_MAX_ATT * 8, 0.f);
    A.tjMod.assign(size_t(GS_NB_RECORDS) * GS_MAX_FT * GS_MAX_TJ * 4, 0.f);

    for (uint32_t r = 0; r < GS_NB_RECORDS; ++r)
    {
        uint8_t* R = A.rec.data() + size_t(r) * GS_RECORD_SIZE;
        const uint32_t dirty = gsDirtyWord(r);
        std::memcpy(R + 8, &dirty, 4);
        const uint16_t nl16 = uint16_t(gsNumLinks(r)), nd16 = uint16_t(gsDofCount(r));
        std::memcpy(R + 16, &nl16, 2);
        std::memcpy(R + 18, &nd16, 2);
        const uint16_t nspt16 = uint16_t(gsNumSpt(r)), nft16 = uint16_t(gsNumFt(r));
        std::memcpy(R + 20, &nspt16, 2);
        std::memcpy(R + 22, &nft16, 2);
        for (int a = 0; a < 6; ++a)
            for (uint32_t d = 0; d < GS_MAX_DOFS; ++d)
                A.dofArr[a][size_t(r) * GS_MAX_DOFS + d] = gsf(r, 50u + a * 20u + d + salt);
        for (uint32_t l = 0; l < GS_MAX_LINKS; ++l)
        {
            for (uint32_t c = 0; c < 6; ++c)
            {
                A.mvel[(size_t(r) * GS_MAX_LINKS + l) * 6 + c] = gsf(r, 200u + l * 6u + c + salt);
                A.macc[(size_t(r) * GS_MAX_LINKS + l) * 6 + c] = gsf(r, 300u + l * 6u + c + salt);
                A.incoming[(size_t(r) * GS_MAX_LINKS + l) * 6 + c] =
                    gsf(r, 400u + l * 6u + c + salt);
            }
            const PxQuat qw = gsQuat(r, l), qa = gsQuat(r, 50u + l);
            float* xw = &A.xw[(size_t(r) * GS_MAX_LINKS + l) * 7];
            xw[0] = qw.x; xw[1] = qw.y; xw[2] = qw.z; xw[3] = qw.w;
            xw[4] = gsf(r, 500u + l); xw[5] = gsf(r, 510u + l); xw[6] = gsf(r, 520u + l);
            float* xa = &A.xa[(size_t(r) * GS_MAX_LINKS + l) * 7];
            xa[0] = qa.x; xa[1] = qa.y; xa[2] = qa.z; xa[3] = qa.w;
            xa[4] = gsf(r, 530u + l); xa[5] = gsf(r, 540u + l); xa[6] = gsf(r, 550u + l);
            float* pp = &A.props[(size_t(r) * GS_MAX_LINKS + l) * 4];
            pp[0] = 0.25f + 0.125f * float((r + l) % 5u);
            pp[1] = 0.5f + 0.0625f * float((r * 2u + l) % 4u);
            pp[2] = 0.75f + 0.03125f * float((r + 2u * l) % 6u);
            pp[3] = 0.375f + 0.25f * float((r + l) % 3u);
        }
        for (uint32_t t = 0; t < GS_MAX_SPT; ++t)
            for (uint32_t c = 0; c < 4; ++c)
                A.sptParams[(size_t(r) * GS_MAX_SPT + t) * 4 + c] =
                    gsf(r, 600u + t * 4u + c + salt);
        for (uint32_t t = 0; t < GS_MAX_FT; ++t)
            for (uint32_t c = 0; c < 8; ++c)
                A.ftParams[(size_t(r) * GS_MAX_FT + t) * 8 + c] =
                    gsf(r, 640u + t * 8u + c + salt);
        // mod-element content is per (record, element) -- IDENTICAL across
        // tendons, so the transcribed ninth-candidate behavior (the missing
        // tendonIndex in the user-buffer offset, incl. the get-side write
        // race) stays deterministic
        for (uint32_t t = 0; t < GS_MAX_SPT; ++t)
            for (uint32_t a = 0; a < GS_MAX_ATT; ++a)
                for (uint32_t c = 0; c < 8; ++c)
                    A.attMod[((size_t(r) * GS_MAX_SPT + t) * GS_MAX_ATT + a) * 8 + c] =
                        gsf(r, 700u + a * 8u + c + salt);
        for (uint32_t t = 0; t < GS_MAX_FT; ++t)
            for (uint32_t j2 = 0; j2 < GS_MAX_TJ; ++j2)
                for (uint32_t c = 0; c < 4; ++c)
                    A.tjMod[((size_t(r) * GS_MAX_FT + t) * GS_MAX_TJ + j2) * 4 + c] =
                        gsf(r, 760u + j2 * 4u + c + salt);
    }
    A.idx.resize(GS_IDX_ENTRIES);
    for (uint32_t j = 0; j < GS_IDX_ENTRIES; ++j)
        A.idx[j] = (j < GS_NB_LIVE) ? ((j * 5u + 1u) % GS_NB_LIVE)
                                    : (GS_NB_LIVE + (j % (GS_NB_RECORDS - GS_NB_LIVE)));
}

// device image of one seeded arena; records get their pointers patched in
struct GsDev
{
    std::vector<PxgDevicePtr> bufs;
    PxgDevicePtr dRec = 0, dDesc = 0, dIdx = 0;
    PxgDevicePtr dDof[6] = { 0, 0, 0, 0, 0, 0 };
    PxgDevicePtr dMv = 0, dMa = 0, dIn = 0, dXw = 0, dXa = 0, dPr = 0, dEa = 0;
    PxgDevicePtr dSp = 0, dFp = 0, dSt = 0, dFt = 0, dAm = 0, dJm = 0;
    bool ok = true;
};

static void gsUpload(PxgCudaShim& shim, GsArena& A, GsDev& D)
{
    auto up = [&](const void* p, size_t n) {
        PxgDevicePtr dp = 0;
        shim.memAlloc(&dp, n);
        if (dp && p) shim.memcpyHtoD(dp, p, n);
        D.bufs.push_back(dp);
        D.ok = D.ok && dp != 0;
        return dp;
    };
    for (int a = 0; a < 6; ++a)
        D.dDof[a] = up(A.dofArr[a].data(), A.dofArr[a].size() * 4);
    D.dMv = up(A.mvel.data(), A.mvel.size() * 4);
    D.dMa = up(A.macc.data(), A.macc.size() * 4);
    D.dIn = up(A.incoming.data(), A.incoming.size() * 4);
    D.dXw = up(A.xw.data(), A.xw.size() * 4);
    D.dXa = up(A.xa.data(), A.xa.size() * 4);
    D.dPr = up(A.props.data(), A.props.size() * 4);
    D.dEa = up(A.extAccel.data(), A.extAccel.size() * 4);
    D.dSp = up(A.sptParams.data(), A.sptParams.size() * 4);
    D.dFp = up(A.ftParams.data(), A.ftParams.size() * 4);
    D.dAm = up(A.attMod.data(), A.attMod.size() * 4);
    D.dJm = up(A.tjMod.data(), A.tjMod.size() * 4);
    // PxgArticulationTendon images: mFixedElements stays poison, mModElements
    // points into the mod blocks, mNbElements is the per-tendon count
    for (uint32_t r = 0; r < GS_NB_RECORDS; ++r)
    {
        for (uint32_t t = 0; t < GS_MAX_SPT; ++t)
        {
            uint8_t* T = A.sptTendons.data() + (size_t(r) * GS_MAX_SPT + t) * 24;
            const uint64_t mp = uint64_t(D.dAm)
                + ((size_t(r) * GS_MAX_SPT + t) * GS_MAX_ATT) * 32u;
            std::memcpy(T + 8, &mp, 8);
            const uint32_t nb = gsNbAtt(r, t);
            std::memcpy(T + 16, &nb, 4);
        }
        for (uint32_t t = 0; t < GS_MAX_FT; ++t)
        {
            uint8_t* T = A.ftTendons.data() + (size_t(r) * GS_MAX_FT + t) * 24;
            const uint64_t mp = uint64_t(D.dJm)
                + ((size_t(r) * GS_MAX_FT + t) * GS_MAX_TJ) * 16u;
            std::memcpy(T + 8, &mp, 8);
            const uint32_t nb = gsNbTj(r, t);
            std::memcpy(T + 16, &nb, 4);
        }
    }
    D.dSt = up(A.sptTendons.data(), A.sptTendons.size());
    D.dFt = up(A.ftTendons.data(), A.ftTendons.size());

    // record field offsets: jpos 112, jvel 120, jacc 128, jforce 136,
    // jtpos 144, jtvel 152 -- note dofArr order is jpos jvel jacc jforce
    // jtvel jtpos (the READ enum order)
    const uint32_t fieldOf[6] = { 112, 120, 128, 136, 152, 144 };
    for (uint32_t r = 0; r < GS_NB_RECORDS; ++r)
    {
        uint8_t* R = A.rec.data() + size_t(r) * GS_RECORD_SIZE;
        auto put = [&](uint32_t off, uint64_t v) { std::memcpy(R + off, &v, 8); };
        for (int a = 0; a < 6; ++a)
            put(fieldOf[a], uint64_t(D.dDof[a]) + uint64_t(r) * GS_MAX_DOFS * 4u);
        put(56, uint64_t(D.dMv) + uint64_t(r) * GS_MAX_LINKS * 24u);
        put(64, uint64_t(D.dMa) + uint64_t(r) * GS_MAX_LINKS * 24u);
        put(72, uint64_t(D.dIn) + uint64_t(r) * GS_MAX_LINKS * 24u);
        put(184, uint64_t(D.dXw) + uint64_t(r) * GS_MAX_LINKS * 28u);
        put(200, uint64_t(D.dXa) + uint64_t(r) * GS_MAX_LINKS * 28u);
        put(256, uint64_t(D.dPr) + uint64_t(r) * GS_MAX_LINKS * 16u);
        put(96, uint64_t(D.dEa) + uint64_t(r) * GS_MAX_LINKS * 24u);
        put(304, uint64_t(D.dSp) + uint64_t(r) * GS_MAX_SPT * 16u);
        put(312, uint64_t(D.dSt) + uint64_t(r) * GS_MAX_SPT * 24u);
        put(320, uint64_t(D.dFp) + uint64_t(r) * GS_MAX_FT * 32u);
        put(328, uint64_t(D.dFt) + uint64_t(r) * GS_MAX_FT * 24u);
    }
    D.dRec = up(A.rec.data(), A.rec.size());
    std::vector<uint8_t> desc(288, 0xAB);
    {
        const uint64_t v = uint64_t(D.dRec);
        std::memcpy(desc.data() + 32, &v, 8);
        const uint32_t nb = GS_NB_LIVE;
        std::memcpy(desc.data() + 44, &nb, 4);
        const uint32_t mspt = GS_MAX_SPT, matt = GS_MAX_ATT, mft = GS_MAX_FT, mtj = GS_MAX_TJ;
        std::memcpy(desc.data() + 260, &mspt, 4);
        std::memcpy(desc.data() + 264, &matt, 4);
        std::memcpy(desc.data() + 268, &mft, 4);
        std::memcpy(desc.data() + 272, &mtj, 4);
    }
    D.dDesc = up(desc.data(), desc.size());
    D.dIdx = up(A.idx.data(), A.idx.size() * 4);
}

static size_t gsBad = 0;

static void gsCmp(PxgCudaShim& shim, PxgDevicePtr dp, const std::vector<float>& want,
                  const char* what)
{
    std::vector<float> got(want.size());
    shim.memcpyDtoH(got.data(), dp, got.size() * 4);
    const uint32_t* w = reinterpret_cast<const uint32_t*>(want.data());
    const uint32_t* g = reinterpret_cast<const uint32_t*>(got.data());
    for (size_t k = 0; k < want.size(); ++k)
        if (w[k] != g[k])
        {
            if (gsBad < 10)
                std::printf("MM gs %s word %zu  ref %08x (%g)  gpu %08x (%g)\n", what, k, w[k],
                            want[k], g[k], got[k]);
            ++gsBad;
        }
}

static void gsCmpRecDirty(PxgCudaShim& shim, PxgDevicePtr dRec,
                          const uint32_t* wantDirty, const char* what)
{
    std::vector<uint8_t> got(size_t(GS_NB_RECORDS) * GS_RECORD_SIZE);
    shim.memcpyDtoH(got.data(), dRec, got.size());
    for (uint32_t r = 0; r < GS_NB_RECORDS; ++r)
    {
        uint32_t g;
        std::memcpy(&g, got.data() + size_t(r) * GS_RECORD_SIZE + 8, 4);
        if (g != wantDirty[r])
        {
            if (gsBad < 10)
                std::printf("MM gs %s dirty rec %u  ref %08x  gpu %08x\n", what, r, wantDirty[r],
                            g);
            ++gsBad;
        }
    }
}

struct P6 { uint64_t a, b, c; uint32_t n, m, t; };
struct P5 { uint64_t a, b, c; uint32_t n, m; };
struct P4 { uint64_t a, b, c; uint32_t n; };

static const PxQuat gsQ(const float* f) { return PxQuat(f[0], f[1], f[2], f[3]); }
static const PxVec3 gsP(const float* f) { return PxVec3(f[4], f[5], f[6]); }

}  // namespace

PXG_TEST(artiGetSet)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiGetSet.spv", "getArtiDofStates", 36u, 0 },
        { "npArtiGetSet.spv", "getArtiTransformStates", 36u, 0 },
        { "npArtiGetSet.spv", "getArtiVelocityStates", 36u, 0 },
        { "npArtiGetSet.spv", "getArtiSpatialForceStates", 32u, 0 },
        { "npArtiGetSet.spv", "setArtiDofStates", 36u, 0 },
        { "npArtiGetSet.spv", "setArtiRootGlobalPoseState", 28u, 0 },
        { "npArtiGetSet.spv", "setArtiRootVelocityState", 32u, 0 },
        { "npArtiGetSet.spv", "setArtiLinkForceState", 32u, 0 },
        { "npArtiGetSet.spv", "setArtiLinkTorqueState", 32u, 0 },
        { "npArtiGetSet.spv", "getArtiTendonState", 36u, 0 },
        { "npArtiGetSet.spv", "setArtiTendonState", 36u, 0 },
        { "npArtiGetSet.spv", "getArtiSpatialTendonAttachmentState", 32u, 0 },
        { "npArtiGetSet.spv", "setArtiSpatialTendonAttachmentState", 32u, 0 },
        { "npArtiGetSet.spv", "getArtiFixedTendonJointState", 28u, 0 },
        { "npArtiGetSet.spv", "setArtiFixedTendonJointState", 28u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 15, &artiGsLoader);
    gsBad = 0;

    auto jobsOf = [&](const GsArena& A, uint32_t j) { return A.idx[j]; };

    // ---- getArtiDofStates: all six read types ----
    for (uint32_t type = 0; type < 6; ++type)
    {
        GsArena A;
        gsSeed(A, 3u + type);
        GsDev D;
        gsUpload(shim, A, D);
        if (!D.ok) { for (auto p : D.bufs) if (p) shim.memFree(p); continue; }
        const uint32_t outGroups = (2u * 512u - 1u) / GS_MAX_DOFS + 1u;
        std::vector<float> out(size_t(outGroups) * GS_MAX_DOFS, 0.f);
        std::memset(out.data(), 0xAB, out.size() * 4);
        PxgDevicePtr dOut = 0;
        shim.memAlloc(&dOut, out.size() * 4);
        shim.memcpyHtoD(dOut, out.data(), out.size() * 4);
        P6 prm = { uint64_t(D.dDesc), uint64_t(dOut), uint64_t(D.dIdx), GS_NB_LIVE, GS_MAX_DOFS,
                   type };
        PXG_CHECK_MSG(shim.launchKernel(0, 2, 1, 1, &prm, 36u) == PxgResult::Success,
                      "launch getArtiDofStates");
        shim.streamSynchronize();
        std::vector<float> want = out;
        for (uint32_t j = 0; j < GS_NB_LIVE; ++j)
        {
            const uint32_t r = jobsOf(A, j);
            for (uint32_t d = 0; d < gsDofCount(r); ++d)
                want[size_t(j) * GS_MAX_DOFS + d] = A.dofArr[type][size_t(r) * GS_MAX_DOFS + d];
        }
        gsCmp(shim, dOut, want, "getDof");
        shim.memFree(dOut);
        for (auto p : D.bufs) if (p) shim.memFree(p);
    }

    // ---- getArtiTransformStates: eROOT_GLOBAL_POSE (6) and eLINK_GLOBAL_POSE (9) ----
    for (uint32_t type = 6; type <= 9; type += 3)
    {
        GsArena A;
        gsSeed(A, 11u + type);
        GsDev D;
        gsUpload(shim, A, D);
        if (!D.ok) { for (auto p : D.bufs) if (p) shim.memFree(p); continue; }
        const uint32_t outGroups = (2u * 512u - 1u) / GS_MAX_LINKS + 1u;
        std::vector<float> out(size_t(outGroups) * GS_MAX_LINKS * 7, 0.f);
        std::memset(out.data(), 0xAB, out.size() * 4);
        PxgDevicePtr dOut = 0;
        shim.memAlloc(&dOut, out.size() * 4);
        shim.memcpyHtoD(dOut, out.data(), out.size() * 4);
        P6 prm = { uint64_t(D.dDesc), uint64_t(dOut), uint64_t(D.dIdx), GS_NB_LIVE, GS_MAX_LINKS,
                   type };
        PXG_CHECK_MSG(shim.launchKernel(1, 2, 1, 1, &prm, 36u) == PxgResult::Success,
                      "launch getArtiTransformStates");
        shim.streamSynchronize();
        std::vector<float> want = out;
        for (uint32_t j = 0; j < GS_NB_LIVE; ++j)
        {
            const uint32_t r = jobsOf(A, j);
            const uint32_t nl = (type == 6u) ? 1u : gsNumLinks(r);
            for (uint32_t l = 0; l < nl; ++l)
            {
                const float* xw = &A.xw[(size_t(r) * GS_MAX_LINKS + l) * 7];
                const float* xa = &A.xa[(size_t(r) * GS_MAX_LINKS + l) * 7];
                const PxTransform t =
                    PxTransform(gsP(xw), gsQ(xw)) * PxTransform(gsP(xa), gsQ(xa)).getInverse();
                float* o = &want[(size_t(j) * GS_MAX_LINKS + l) * 7];
                o[0] = t.q.x; o[1] = t.q.y; o[2] = t.q.z; o[3] = t.q.w;
                o[4] = t.p.x; o[5] = t.p.y; o[6] = t.p.z;
            }
        }
        gsCmp(shim, dOut, want, "getTransform");
        shim.memFree(dOut);
        for (auto p : D.bufs) if (p) shim.memFree(p);
    }

    // ---- getArtiVelocityStates: the six vec3 read types ----
    {
        const uint32_t types[6] = { 7, 8, 10, 11, 12, 13 };
        for (uint32_t ti = 0; ti < 6; ++ti)
        {
            const uint32_t type = types[ti];
            GsArena A;
            gsSeed(A, 23u + type);
            GsDev D;
            gsUpload(shim, A, D);
            if (!D.ok) { for (auto p : D.bufs) if (p) shim.memFree(p); continue; }
            const uint32_t outGroups = (2u * 480u - 1u) / (3u * GS_MAX_LINKS) + 1u;
            std::vector<float> out(size_t(outGroups) * GS_MAX_LINKS * 3, 0.f);
            std::memset(out.data(), 0xAB, out.size() * 4);
            PxgDevicePtr dOut = 0;
            shim.memAlloc(&dOut, out.size() * 4);
            shim.memcpyHtoD(dOut, out.data(), out.size() * 4);
            P6 prm = { uint64_t(D.dDesc), uint64_t(dOut), uint64_t(D.dIdx), GS_NB_LIVE,
                       GS_MAX_LINKS, type };
            PXG_CHECK_MSG(shim.launchKernel(2, 2, 1, 1, &prm, 36u) == PxgResult::Success,
                          "launch getArtiVelocityStates");
            shim.streamSynchronize();
            std::vector<float> want = out;
            for (uint32_t j = 0; j < GS_NB_LIVE; ++j)
            {
                const uint32_t r = jobsOf(A, j);
                const bool acc = type >= 12u;
                const bool linear = (type == 7u || type == 10u || type == 12u);
                const uint32_t nl = (type <= 8u) ? 1u : gsNumLinks(r);
                const std::vector<float>& src = acc ? A.macc : A.mvel;
                for (uint32_t l = 0; l < nl; ++l)
                    for (uint32_t e = 0; e < 3; ++e)
                        want[(size_t(j) * GS_MAX_LINKS + l) * 3 + e] =
                            src[(size_t(r) * GS_MAX_LINKS + l) * 6 + (linear ? 3u : 0u) + e];
            }
            gsCmp(shim, dOut, want, "getVelocity");
            shim.memFree(dOut);
            for (auto p : D.bufs) if (p) shim.memFree(p);
        }
    }

    // ---- getArtiSpatialForceStates ----
    {
        GsArena A;
        gsSeed(A, 41u);
        GsDev D;
        gsUpload(shim, A, D);
        if (D.ok)
        {
            const uint32_t outGroups = (2u * 512u - 1u) / (6u * GS_MAX_LINKS) + 1u;
            std::vector<float> out(size_t(outGroups) * GS_MAX_LINKS * 6, 0.f);
            std::memset(out.data(), 0xAB, out.size() * 4);
            PxgDevicePtr dOut = 0;
            shim.memAlloc(&dOut, out.size() * 4);
            shim.memcpyHtoD(dOut, out.data(), out.size() * 4);
            P5 prm = { uint64_t(D.dDesc), uint64_t(dOut), uint64_t(D.dIdx), GS_NB_LIVE,
                       GS_MAX_LINKS };
            PXG_CHECK_MSG(shim.launchKernel(3, 2, 1, 1, &prm, 32u) == PxgResult::Success,
                          "launch getArtiSpatialForceStates");
            shim.streamSynchronize();
            std::vector<float> want = out;
            for (uint32_t j = 0; j < GS_NB_LIVE; ++j)
            {
                const uint32_t r = jobsOf(A, j);
                for (uint32_t l = 0; l < gsNumLinks(r); ++l)
                    for (uint32_t e = 0; e < 6; ++e)
                        want[(size_t(j) * GS_MAX_LINKS + l) * 6 + e] =
                            A.incoming[(size_t(r) * GS_MAX_LINKS + l) * 6 + e];
            }
            gsCmp(shim, dOut, want, "getSpatialForce");
            shim.memFree(dOut);
        }
        for (auto p : D.bufs) if (p) shim.memFree(p);
    }

    // ---- setArtiDofStates: five write types, each with its dirty-bit set ----
    {
        const uint32_t fieldOfW[5] = { 0, 1, 3, 4, 5 };   // dofArr index per write type
        const uint32_t dirtyOfW[5] = { 2u | (1u << 20), 4u | (1u << 20), 8u, 4096u, 8192u };
        for (uint32_t type = 0; type < 5; ++type)
        {
            GsArena A;
            gsSeed(A, 61u + type);
            GsDev D;
            gsUpload(shim, A, D);
            if (!D.ok) { for (auto p : D.bufs) if (p) shim.memFree(p); continue; }
            std::vector<float> in(size_t(GS_NB_LIVE) * GS_MAX_DOFS);
            for (size_t k = 0; k < in.size(); ++k) in[k] = gsf(uint32_t(k), 777u + type);
            PxgDevicePtr dIn = 0;
            shim.memAlloc(&dIn, in.size() * 4);
            shim.memcpyHtoD(dIn, in.data(), in.size() * 4);
            P6 prm = { uint64_t(D.dDesc), uint64_t(dIn), uint64_t(D.dIdx), GS_NB_LIVE,
                       GS_MAX_DOFS, type };
            PXG_CHECK_MSG(shim.launchKernel(4, 2, 1, 1, &prm, 36u) == PxgResult::Success,
                          "launch setArtiDofStates");
            shim.streamSynchronize();
            std::vector<float> want = A.dofArr[fieldOfW[type]];
            uint32_t wantDirty[GS_NB_RECORDS];
            for (uint32_t r = 0; r < GS_NB_RECORDS; ++r) wantDirty[r] = gsDirtyWord(r);
            for (uint32_t j = 0; j < GS_NB_LIVE; ++j)
            {
                const uint32_t r = jobsOf(A, j);
                for (uint32_t d = 0; d < gsDofCount(r); ++d)
                    want[size_t(r) * GS_MAX_DOFS + d] = in[size_t(j) * GS_MAX_DOFS + d];
                wantDirty[r] |= dirtyOfW[type];
            }
            gsCmp(shim, D.dDof[fieldOfW[type]], want, "setDof");
            gsCmpRecDirty(shim, D.dRec, wantDirty, "setDof");
            shim.memFree(dIn);
            for (auto p : D.bufs) if (p) shim.memFree(p);
        }
    }

    // ---- setArtiRootGlobalPoseState ----
    {
        GsArena A;
        gsSeed(A, 91u);
        GsDev D;
        gsUpload(shim, A, D);
        if (D.ok)
        {
            std::vector<float> in(size_t(GS_NB_LIVE) * 7);
            for (uint32_t j = 0; j < GS_NB_LIVE; ++j)
            {
                const PxQuat q = gsQuat(90u + j, 5u);
                in[j * 7 + 0] = q.x; in[j * 7 + 1] = q.y; in[j * 7 + 2] = q.z; in[j * 7 + 3] = q.w;
                in[j * 7 + 4] = gsf(j, 20u); in[j * 7 + 5] = gsf(j, 21u); in[j * 7 + 6] = gsf(j, 22u);
            }
            PxgDevicePtr dIn = 0;
            shim.memAlloc(&dIn, in.size() * 4);
            shim.memcpyHtoD(dIn, in.data(), in.size() * 4);
            P4 prm = { uint64_t(D.dDesc), uint64_t(dIn), uint64_t(D.dIdx), GS_NB_LIVE };
            PXG_CHECK_MSG(shim.launchKernel(5, 2, 1, 1, &prm, 28u) == PxgResult::Success,
                          "launch setArtiRootGlobalPoseState");
            shim.streamSynchronize();
            std::vector<float> want = A.xw;
            uint32_t wantDirty[GS_NB_RECORDS];
            for (uint32_t r = 0; r < GS_NB_RECORDS; ++r) wantDirty[r] = gsDirtyWord(r);
            for (uint32_t j = 0; j < GS_NB_LIVE; ++j)
            {
                const uint32_t r = jobsOf(A, j);
                const PxTransform actorPose(PxVec3(in[j * 7 + 4], in[j * 7 + 5], in[j * 7 + 6]),
                                            PxQuat(in[j * 7 + 0], in[j * 7 + 1], in[j * 7 + 2],
                                                   in[j * 7 + 3]));
                const float* xa = &A.xa[size_t(r) * GS_MAX_LINKS * 7];
                const PxTransform pose = actorPose * PxTransform(gsP(xa), gsQ(xa));
                float* o = &want[size_t(r) * GS_MAX_LINKS * 7];
                o[0] = pose.q.x; o[1] = pose.q.y; o[2] = pose.q.z; o[3] = pose.q.w;
                o[4] = pose.p.x; o[5] = pose.p.y; o[6] = pose.p.z;
                wantDirty[r] |= 16u | (1u << 20);
            }
            gsCmp(shim, D.dXw, want, "setRootPose");
            gsCmpRecDirty(shim, D.dRec, wantDirty, "setRootPose");
            shim.memFree(dIn);
        }
        for (auto p : D.bufs) if (p) shim.memFree(p);
    }

    // ---- setArtiRootVelocityState: linear (6) and angular (7) ----
    for (uint32_t type = 6; type <= 7; ++type)
    {
        GsArena A;
        gsSeed(A, 101u + type);
        GsDev D;
        gsUpload(shim, A, D);
        if (!D.ok) { for (auto p : D.bufs) if (p) shim.memFree(p); continue; }
        std::vector<float> in(size_t(GS_NB_LIVE) * 3);
        for (size_t k = 0; k < in.size(); ++k) in[k] = gsf(uint32_t(k), 888u + type);
        PxgDevicePtr dIn = 0;
        shim.memAlloc(&dIn, in.size() * 4);
        shim.memcpyHtoD(dIn, in.data(), in.size() * 4);
        P5 prm = { uint64_t(D.dDesc), uint64_t(dIn), uint64_t(D.dIdx), GS_NB_LIVE, type };
        PXG_CHECK_MSG(shim.launchKernel(6, 2, 1, 1, &prm, 32u) == PxgResult::Success,
                      "launch setArtiRootVelocityState");
        shim.streamSynchronize();
        std::vector<float> want = A.mvel;
        uint32_t wantDirty[GS_NB_RECORDS];
        for (uint32_t r = 0; r < GS_NB_RECORDS; ++r) wantDirty[r] = gsDirtyWord(r);
        for (uint32_t j = 0; j < GS_NB_LIVE; ++j)
        {
            const uint32_t r = jobsOf(A, j);
            for (uint32_t e = 0; e < 3; ++e)
                want[size_t(r) * GS_MAX_LINKS * 6 + (type == 6u ? 3u : 0u) + e] = in[j * 3 + e];
            wantDirty[r] |= 32u | (1u << 20);
        }
        gsCmp(shim, D.dMv, want, "setRootVel");
        gsCmpRecDirty(shim, D.dRec, wantDirty, "setRootVel");
        shim.memFree(dIn);
        for (auto p : D.bufs) if (p) shim.memFree(p);
    }

    // ---- setArtiLinkForceState + setArtiLinkTorqueState ----
    for (uint32_t which = 0; which < 2; ++which)
    {
        GsArena A;
        gsSeed(A, 121u + which);
        GsDev D;
        gsUpload(shim, A, D);
        if (!D.ok) { for (auto p : D.bufs) if (p) shim.memFree(p); continue; }
        std::vector<float> in(size_t(GS_NB_LIVE) * GS_MAX_LINKS * 3);
        for (size_t k = 0; k < in.size(); ++k) in[k] = gsf(uint32_t(k), 999u + which);
        PxgDevicePtr dIn = 0;
        shim.memAlloc(&dIn, in.size() * 4);
        shim.memcpyHtoD(dIn, in.data(), in.size() * 4);
        P5 prm = { uint64_t(D.dDesc), uint64_t(dIn), uint64_t(D.dIdx), GS_NB_LIVE, GS_MAX_LINKS };
        PXG_CHECK_MSG(shim.launchKernel(which == 0 ? 7 : 8, 2, 1, 1, &prm, 32u) ==
                          PxgResult::Success,
                      "launch setArtiLink*State");
        shim.streamSynchronize();
        std::vector<float> want = A.extAccel;
        for (uint32_t j = 0; j < GS_NB_LIVE; ++j)
        {
            const uint32_t r = jobsOf(A, j);
            for (uint32_t l = 0; l < gsNumLinks(r); ++l)
            {
                const float* src = &in[(size_t(j) * GS_MAX_LINKS + l) * 3];
                const float* pp = &A.props[(size_t(r) * GS_MAX_LINKS + l) * 4];
                float* dst = &want[(size_t(r) * GS_MAX_LINKS + l) * 6];
                if (which == 0)
                {
                    for (uint32_t e = 0; e < 3; ++e)
                        dst[e] = src[e] * pp[3];   // .top = force * invMass
                }
                else
                {
                    const float* xw = &A.xw[(size_t(r) * GS_MAX_LINKS + l) * 7];
                    const PxQuat q = gsQ(xw);
                    const PxVec3 localT = q.rotateInv(PxVec3(src[0], src[1], src[2]));
                    const PxVec3 localA = PxVec3(pp[0], pp[1], pp[2]).multiply(localT);
                    const PxVec3 worldA = q.rotate(localA);
                    dst[3] = worldA.x; dst[4] = worldA.y; dst[5] = worldA.z;   // .bottom
                }
            }
        }
        gsCmp(shim, D.dEa, want, which == 0 ? "setLinkForce" : "setLinkTorque");
        shim.memFree(dIn);
        for (auto p : D.bufs) if (p) shim.memFree(p);
    }

    // ---- get/setArtiTendonState: spatial (R17/W12) and fixed (R15/W10) ----
    for (uint32_t which = 0; which < 4; ++which)
    {
        const bool isGet = which < 2;
        const bool isSpt = (which % 2u) == 0u;
        const uint32_t type = isGet ? (isSpt ? 17u : 15u) : (isSpt ? 12u : 10u);
        const uint32_t maxT = isSpt ? GS_MAX_SPT : GS_MAX_FT;
        const uint32_t strideF = isSpt ? 4u : 8u;   // floats per tendon datum
        GsArena A;
        gsSeed(A, 141u + which);
        GsDev D;
        gsUpload(shim, A, D);
        if (!D.ok) { for (auto p : D.bufs) if (p) shim.memFree(p); continue; }
        const uint32_t bufGroups = (2u * 512u - 1u) / maxT + 1u;
        std::vector<float> user(size_t(bufGroups) * maxT * strideF, 0.f);
        if (isGet)
            std::memset(user.data(), 0xAB, user.size() * 4);
        else
            for (size_t k = 0; k < user.size(); ++k) user[k] = gsf(uint32_t(k), 811u + which);
        PxgDevicePtr dU = 0;
        shim.memAlloc(&dU, user.size() * 4);
        shim.memcpyHtoD(dU, user.data(), user.size() * 4);
        P6 prm = { uint64_t(D.dDesc), uint64_t(dU), uint64_t(D.dIdx), GS_NB_LIVE, maxT, type };
        PXG_CHECK_MSG(shim.launchKernel(isGet ? 9 : 10, 2, 1, 1, &prm, 36u) ==
                          PxgResult::Success,
                      "launch tendon state");
        shim.streamSynchronize();
        uint32_t wantDirty[GS_NB_RECORDS];
        for (uint32_t r = 0; r < GS_NB_RECORDS; ++r) wantDirty[r] = gsDirtyWord(r);
        if (isGet)
        {
            std::vector<float> want = user;
            const std::vector<float>& src = isSpt ? A.sptParams : A.ftParams;
            for (uint32_t j = 0; j < GS_NB_LIVE; ++j)
            {
                const uint32_t r = jobsOf(A, j);
                const uint32_t n = isSpt ? gsNumSpt(r) : gsNumFt(r);
                for (uint32_t t = 0; t < n; ++t)
                    for (uint32_t c = 0; c < strideF; ++c)
                        want[(size_t(j) * maxT + t) * strideF + c] =
                            src[(size_t(r) * maxT + t) * strideF + c];
            }
            gsCmp(shim, dU, want, "getTendon");
        }
        else
        {
            std::vector<float> want = isSpt ? A.sptParams : A.ftParams;
            for (uint32_t j = 0; j < GS_NB_LIVE; ++j)
            {
                const uint32_t r = jobsOf(A, j);
                const uint32_t n = isSpt ? gsNumSpt(r) : gsNumFt(r);
                for (uint32_t t = 0; t < n; ++t)
                    for (uint32_t c = 0; c < strideF; ++c)
                        want[(size_t(r) * maxT + t) * strideF + c] =
                            user[(size_t(j) * maxT + t) * strideF + c];
                wantDirty[r] |= isSpt ? 16384u : 65536u;   // set even when n == 0 (trap 8)
            }
            gsCmp(shim, isSpt ? D.dSp : D.dFp, want, "setTendon");
            gsCmpRecDirty(shim, D.dRec, wantDirty, "setTendon");
        }
        shim.memFree(dU);
        for (auto p : D.bufs) if (p) shim.memFree(p);
    }

    // ---- get/setArtiSpatialTendonAttachmentState + get/setArtiFixedTendonJointState ----
    for (uint32_t which = 0; which < 4; ++which)
    {
        const bool isGet = which < 2;
        const bool isAtt = (which % 2u) == 0u;
        const uint32_t maxT = isAtt ? (GS_MAX_SPT * GS_MAX_ATT) : (GS_MAX_FT * GS_MAX_TJ);
        const uint32_t elemF = isAtt ? 8u : 4u;   // floats per element
        GsArena A;
        gsSeed(A, 171u + which);
        GsDev D;
        gsUpload(shim, A, D);
        if (!D.ok) { for (auto p : D.bufs) if (p) shim.memFree(p); continue; }
        const uint32_t bufGroups = (2u * 512u - 1u) / maxT + 1u;
        std::vector<float> user(size_t(bufGroups) * maxT * elemF, 0.f);
        if (isGet)
            std::memset(user.data(), 0xAB, user.size() * 4);
        else
            for (size_t k = 0; k < user.size(); ++k) user[k] = gsf(uint32_t(k), 911u + which);
        PxgDevicePtr dU = 0;
        shim.memAlloc(&dU, user.size() * 4);
        shim.memcpyHtoD(dU, user.data(), user.size() * 4);
        if (isAtt)
        {
            P5 prm = { uint64_t(D.dDesc), uint64_t(dU), uint64_t(D.dIdx), GS_NB_LIVE, maxT };
            PXG_CHECK_MSG(shim.launchKernel(isGet ? 11 : 12, 2, 1, 1, &prm, 32u) ==
                              PxgResult::Success,
                          "launch attachment state");
        }
        else
        {
            P4 prm = { uint64_t(D.dDesc), uint64_t(dU), uint64_t(D.dIdx), GS_NB_LIVE };
            PXG_CHECK_MSG(shim.launchKernel(isGet ? 13 : 14, 2, 1, 1, &prm, 28u) ==
                              PxgResult::Success,
                          "launch tendon-joint state");
        }
        shim.streamSynchronize();
        const uint32_t nT = isAtt ? GS_MAX_SPT : GS_MAX_FT;
        const uint32_t nE = isAtt ? GS_MAX_ATT : GS_MAX_TJ;
        uint32_t wantDirty[GS_NB_RECORDS];
        for (uint32_t r = 0; r < GS_NB_RECORDS; ++r) wantDirty[r] = gsDirtyWord(r);
        if (isGet)
        {
            // the ninth candidate, transcribed: the user offset carries the
            // element index only; every live tendon writes the same slots
            // (equal content keeps it deterministic)
            std::vector<float> want = user;
            const std::vector<float>& mod = isAtt ? A.attMod : A.tjMod;
            for (uint32_t j = 0; j < GS_NB_LIVE; ++j)
            {
                const uint32_t r = jobsOf(A, j);
                const uint32_t n = isAtt ? gsNumSpt(r) : gsNumFt(r);
                for (uint32_t t = 0; t < n; ++t)
                {
                    const uint32_t nb = isAtt ? gsNbAtt(r, t) : gsNbTj(r, t);
                    for (uint32_t e = 0; e < nb; ++e)
                        for (uint32_t c = 0; c < elemF; ++c)
                            want[(size_t(j) * maxT + e) * elemF + c] =
                                mod[((size_t(r) * nT + t) * nE + e) * elemF + c];
                }
            }
            gsCmp(shim, dU, want, isAtt ? "getAttach" : "getTendonJoint");
        }
        else
        {
            std::vector<float> want = isAtt ? A.attMod : A.tjMod;
            for (uint32_t j = 0; j < GS_NB_LIVE; ++j)
            {
                const uint32_t r = jobsOf(A, j);
                const uint32_t n = isAtt ? gsNumSpt(r) : gsNumFt(r);
                for (uint32_t t = 0; t < n; ++t)
                {
                    const uint32_t nb = isAtt ? gsNbAtt(r, t) : gsNbTj(r, t);
                    for (uint32_t e = 0; e < nb; ++e)
                        for (uint32_t c = 0; c < elemF; ++c)
                            want[((size_t(r) * nT + t) * nE + e) * elemF + c] =
                                user[(size_t(j) * maxT + e) * elemF + c];
                }
                wantDirty[r] |= isAtt ? 32768u : 131072u;   // outside the guard (trap 8)
            }
            gsCmp(shim, isAtt ? D.dAm : D.dJm, want, isAtt ? "setAttach" : "setTendonJoint");
            gsCmpRecDirty(shim, D.dRec, wantDirty, isAtt ? "setAttach" : "setTendonJoint");
        }
        shim.memFree(dU);
        for (auto p : D.bufs) if (p) shim.memFree(p);
    }

    std::printf("[ info ] gs: 33 dispatches over 15 kernels, %zu mismatching words\n", gsBad);
    PXG_CHECK_MSG(gsBad == 0, "%zu mismatches", gsBad);
}
