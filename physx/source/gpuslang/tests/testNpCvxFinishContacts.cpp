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

// P5 Step 4: the COMPLETE finishContactsKernel on the ENGINE ABI, driven
// through PxgCudaShim exactly as PxgNarrowphaseCore.cpp launches it —
// 20-argument blob {4,8,8,8,8,8,8,8,8,8,1,8,8,8,8,8,8,4,4,4} (152 bytes),
// grid ceil(numTests/128) x (32,4) blocks.  Each launch runs ONE active
// warp (numTests <= 32, one block; the block's other 3 warps see all-invalid
// lanes and touch nothing) so the per-warp global atomics on
// PxgPatchAndContactCounters reserve deterministically — the CPU reference
// reproduces the exact serial reservation.  Coverage accumulates over many
// launches with fresh data + reset counters.
//
// The reference is the faithful COMPOSITION of the two bit-verified
// bound-buffer references — writeContactsToStream4threads (tranche 151) and
// the status/patch tail + writePatchesToStream4threads (tranche 152) — over
// the REAL engine structs and streams, keeping the same expected outputs:
// manifold invalidation-bit clear, contact-stream float4 writes (avg point
// first), cmOutputs point/force/patch offsets (+ the 0xFFFFFFFF sentinels
// and force-offset uint wraparound), allflags rewrite, nbContacts, touch/
// patch change flags, PxgPatchAndContactCounters reservations + overflow
// (contact/force >= limit, patch strictly >), rollback, combineMaterials,
// and the 64-byte PxContactPatch payloads.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "PxNodeIndex.h"
#include "PxgContactManager.h"
#include "PxgConvexConvexShape.h"
#include "PxgPersistentContactManifold.h"
#include "PxsContactManagerState.h"
#include "PxsMaterialCore.h"
#include "cudaNpCommon.h"
#include "PxsTransformCache.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };
struct V3 { float x, y, z; };

static const float FLTMAX = 3.4028235e38f;

static uint32_t popc(uint32_t m) { uint32_t c = 0; while (m) { c += m & 1u; m >>= 1; } return c; }
static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static V3 qRot(const F4& q, const V3& v) { const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z, w2 = q.w * q.w - 0.5f, d2 = q.x * vx + q.y * vy + q.z * vz; return V3{ vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d2, vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d2, vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d2 }; }
static V3 unit(const V3& a) { float l = std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z); return l > 0 ? V3{ a.x / l, a.y / l, a.z / l } : a; }

struct MatRef { float dyn, sta, rest, damp; uint32_t flags, fricMode, restMode, dampMode; };

static float combineScalars(float a, float b, uint32_t mode)
{
    if (mode == 0) return 0.5f * (a + b);
    if (mode == 1) return std::fmin(a, b);
    if (mode == 2) return a * b;
    if (mode == 3) return std::fmax(a, b);
    return 0.f;
}
static void combineMaterials(const MatRef& m0, const MatRef& m1,
                             uint32_t& matFlags, float& staF, float& dynF, float& rest, float& damp)
{
    const float r0 = m0.rest, r1 = m1.rest;
    const bool c0 = r0 < 0.f, c1 = r1 < 0.f;
    const bool oneC = c0 != c1, bothC = c0 && c1;
    const bool a0 = (m0.flags & 16u) != 0, a1 = (m1.flags & 16u) != 0;
    const bool oneA = a0 != a1;
    if (bothC && oneA) rest = a0 ? r0 : r1;
    else
    {
        const uint32_t mode = oneC ? 1u : std::max(m0.restMode, m1.restMode);
        const float flip = (bothC && mode == 2u) ? -1.f : 1.f;
        rest = flip * combineScalars(r0, r1, mode);
    }
    if (bothC && oneA) damp = a0 ? m0.damp : m1.damp;
    else
    {
        const uint32_t mode = oneC ? 3u : std::max(m0.dampMode, m1.dampMode);
        damp = combineScalars(m0.damp, m1.damp, mode);
    }
    const uint32_t cf = m0.flags | m1.flags;
    if ((cf & 1u) == 0)
    {
        const uint32_t mode = std::max(m0.fricMode, m1.fricMode);
        const float d = combineScalars(m0.dyn, m1.dyn, mode);
        const float s = combineScalars(m0.sta, m1.sta, mode);
        const float fd = std::fmax(d, 0.f);
        dynF = fd; staF = (s - fd) >= 0.f ? s : fd; matFlags = cf;
    }
    else { matFlags = cf | 2u; dynF = 0.f; staF = 0.f; }
}

// per-pair raw semantic fixture (pre-flip)
struct Pair
{
    uint32_t type0, type1, mat0, mat1;
    V3 scale0, scale1;
    F4 t0q, t1q; V3 t0p, t1p;
    float cd0, cd1;
    F4 normPen[4], pointB[4];
    uint32_t rawNb;          // manifold mNbContacts (maybe |0x80000000)
    uint32_t allflags;       // cmOutputs allflagsStart word (initial)
    uint32_t nbContactsFlags;// cmOutputs word36 (initial; high 16 = flags preserved)
};

struct PairOut
{
    uint32_t manifoldNb;
    uint32_t contactPointsOff, contactForcesOff;
    uint32_t allflagsWord, nbContactsFlags;
    uint32_t touchChange, patchChange;
    uint32_t contactPatchesOff;
    bool patchWritten;
};

struct FinishRef
{
    PairOut pairs[32];
    uint32_t patchesBytes, contactsBytes, forceBytes, overflowError;
    std::vector<uint8_t> points;
    std::vector<uint8_t> patch;
};

// Prior content of the contact and patch streams, by word index.  The engine never clears either
// stream, so every byte the CUDA does not store must come back as uploaded; the value varies per word
// so a store that copies ANOTHER word is visible, not just one that zeroes (P9.87).
static uint32_t finPoison(size_t word) { return 0xCDCDCDCDu ^ uint32_t(word * 0x00010001u); }
static void fillPoison(std::vector<uint8_t>& v, size_t bytes)
{
    v.resize(bytes);
    for (size_t w = 0; w < bytes / 4; ++w) { const uint32_t p = finPoison(w); std::memcpy(&v[w * 4], &p, 4); }
}

static void refFinish(const std::vector<Pair>& pr, uint32_t numTests, bool insertAvg,
                      uint32_t initContacts, uint32_t initForce, uint32_t initPatch,
                      uint32_t contactLimit, uint32_t forceLimit, uint32_t patchLimit,
                      const std::vector<MatRef>& mats, size_t bufBytes, FinishRef& o)
{
    fillPoison(o.points, bufBytes);
    fillPoison(o.patch, bufBytes);
    uint32_t contactsBytes = initContacts, forceBytes = initForce, patchesBytes = initPatch, ovf = 0;

    float scrNormX[32] = {}, scrNormY[32] = {}, scrNormZ[32] = {};
    uint32_t scrNb[32] = {};
    uint32_t scrMat0[32] = {}, scrMat1[32] = {};

    for (uint32_t baseOffs = 0; baseOffs < 32; baseOffs += 8)
    {
        bool valid[8], flip[8];
        float radius[8], cd[8];
        F4 t1q[8]; V3 t1p[8];
        uint32_t num0[8];
        for (uint32_t g = 0; g < 8; ++g)
        {
            const uint32_t elem = baseOffs + g;
            valid[g] = elem < numTests;
            flip[g] = false; radius[g] = 0; cd[g] = 0;
            t1q[g] = F4{ 0, 0, 0, 1 }; t1p[g] = V3{ 0, 0, 0 }; num0[g] = 0;
            if (valid[g])
            {
                const Pair& p = pr[elem];
                uint32_t type0 = p.type0, type1 = p.type1;
                V3 sc0 = p.scale0, sc1 = p.scale1;
                F4 tq0 = p.t0q, tq1 = p.t1q; V3 tp0 = p.t0p, tp1 = p.t1p;
                flip[g] = type1 < type0;
                if (flip[g]) { std::swap(type0, type1); std::swap(sc0, sc1); std::swap(tq0, tq1); std::swap(tp0, tp1); }
                if (type0 == 0 || type0 == 2) radius[g] = sc0.y;
                t1q[g] = tq1; t1p[g] = tp1;
                cd[g] = p.cd0 + p.cd1;
                scrMat0[baseOffs + g] = p.mat0; // UNSWAPPED
                scrMat1[baseOffs + g] = p.mat1;
                uint32_t raw = p.rawNb;
                uint32_t nb = raw & ~0x80000000u;
                if (raw & 0x80000000u) nb = 0;
                num0[g] = nb;
            }
        }

        F4 normPenL[32], pointBL[32];
        for (uint32_t L = 0; L < 32; ++L)
        {
            const uint32_t g = L / 4, j = L % 4;
            normPenL[L] = F4{ 0, 0, 0, 0 };
            F4 pb{ FLTMAX, FLTMAX, FLTMAX, FLTMAX };
            if (j < num0[g])
            {
                const Pair& p = pr[baseOffs + g];
                normPenL[L] = p.normPen[j];
                const float sep = normPenL[L].w - radius[g];
                const V3 q = qRot(t1q[g], V3{ p.pointB[j].x, p.pointB[j].y, p.pointB[j].z });
                pb = F4{ q.x + t1p[g].x, q.y + t1p[g].y, q.z + t1p[g].z, sep };
            }
            pointBL[L] = pb;
        }

        for (uint32_t g = 0; g < 8; ++g)
        {
            V3 sum{ (normPenL[4 * g].x + normPenL[4 * g + 2].x) + (normPenL[4 * g + 1].x + normPenL[4 * g + 3].x),
                    (normPenL[4 * g].y + normPenL[4 * g + 2].y) + (normPenL[4 * g + 1].y + normPenL[4 * g + 3].y),
                    (normPenL[4 * g].z + normPenL[4 * g + 2].z) + (normPenL[4 * g + 1].z + normPenL[4 * g + 3].z) };
            const float magSq = sum.x * sum.x + sum.y * sum.y + sum.z * sum.z;
            // PxVec3::operator/= is reciprocal-multiply (matches the shader / .cu)
            const float invMag = 1.f / std::sqrt(magSq);
            V3 avg = (magSq >= 1e-5f) ? V3{ sum.x * invMag, sum.y * invMag, sum.z * invMag }
                                      : V3{ normPenL[4 * g].x, normPenL[4 * g].y, normPenL[4 * g].z };
            avg = qRot(t1q[g], avg);
            if (flip[g]) avg = V3{ -avg.x, -avg.y, -avg.z };
            scrNormX[baseOffs + g] = avg.x; scrNormY[baseOffs + g] = avg.y; scrNormZ[baseOffs + g] = avg.z;
        }

        uint32_t keepW = 0;
        for (uint32_t L = 0; L < 32; ++L)
            if (pointBL[L].w <= cd[L / 4]) keepW |= (1u << L);
        uint32_t numKept[8];
        for (uint32_t g = 0; g < 8; ++g) numKept[g] = popc(keepW & (0xFu << (4 * g)));
        uint32_t extraW = 0;
        for (uint32_t g = 0; g < 8; ++g) if (insertAvg && numKept[g] > 1) extraW |= (1u << (4 * g));

        uint32_t cOff0 = 0xFFFFFFFFu, fOff0 = 0xFFFFFFFFu;
        const uint32_t total = popc(keepW) + popc(extraW);
        if (total)
        {
            cOff0 = contactsBytes; contactsBytes += 16 * total;
            fOff0 = forceBytes; forceBytes += 4 * total;
            if ((cOff0 + 16 * total) >= contactLimit) { ovf |= 1u; cOff0 = 0xFFFFFFFFu; }
            else if ((fOff0 + 4 * total) >= forceLimit) { ovf |= 2u; fOff0 = 0xFFFFFFFFu; }
        }

        for (uint32_t g = 0; g < 8; ++g)
        {
            const uint32_t elem = baseOffs + g;
            uint32_t numKeptG = numKept[g];
            uint32_t cbLane0 = cOff0, fbLane0 = fOff0;
            if (cOff0 != 0xFFFFFFFFu)
            {
                F4 pv[4];
                for (uint32_t j = 0; j < 4; ++j) pv[j] = (j < numKeptG) ? pointBL[4 * g + j] : F4{ 0, 0, 0, 0 };
                const F4 avgPt{ (pv[0].x + pv[2].x) + (pv[1].x + pv[3].x), (pv[0].y + pv[2].y) + (pv[1].y + pv[3].y),
                                (pv[0].z + pv[2].z) + (pv[1].z + pv[3].z), (pv[0].w + pv[2].w) + (pv[1].w + pv[3].w) };
                for (uint32_t j = 0; j < 4; ++j)
                {
                    const uint32_t L = 4 * g + j;
                    const uint32_t mask = (1u << L) - 1u;
                    const uint32_t byteOffset = popc(keepW & mask) + popc(extraW & mask);
                    const uint32_t cb = cOff0 + 16 * byteOffset;
                    const uint32_t fb = fOff0 + 4 * byteOffset;
                    if (j == 0) { cbLane0 = cb; fbLane0 = fb; }
                    uint32_t offset = 0;
                    if (insertAvg && numKeptG > 1 && j == 0)
                    {
                        const float inv = 1.f / float(numKeptG);
                        const F4 a{ avgPt.x * inv, avgPt.y * inv, avgPt.z * inv, avgPt.w * inv };
                        if (cb + 16 <= o.points.size()) std::memcpy(&o.points[cb], &a, 16);
                        offset = 16;
                    }
                    if (pointBL[L].w <= cd[g])
                        if (cb + offset + 16 <= o.points.size()) std::memcpy(&o.points[cb + offset], &pointBL[L], 16);
                }
            }
            else numKeptG = 0;

            if (numKeptG == 0) { cbLane0 = 0xFFFFFFFFu; fbLane0 = 0xFFFFFFFFu; }
            o.pairs[elem].contactPointsOff = valid[g] ? cbLane0 : 0xFFFFFFFFu;
            o.pairs[elem].contactForcesOff = valid[g] ? fbLane0 : 0xFFFFFFFFu;
            scrNb[baseOffs + g] = numKeptG + ((numKeptG > 1 && insertAvg) ? 1u : 0u);
        }
    }

    uint32_t statusL[32], prevP[32], nbL[32], origW36[32];
    bool prevTouch[32], validL[32];
    for (uint32_t L = 0; L < 32; ++L)
    {
        validL[L] = L < numTests;
        nbL[L] = validL[L] ? scrNb[L] : 0;
        uint32_t allflags = validL[L] ? pr[L].allflags : 0;
        origW36[L] = validL[L] ? pr[L].nbContactsFlags : 0;
        const uint32_t oldSt = (allflags >> 16) & 0xFF;
        uint32_t st = (oldSt & ~3u) | (nbL[L] != 0 ? 2u : 1u);
        prevP[L] = (allflags & 0xFFFF) >> 8;
        prevTouch[L] = (oldSt & 2u) != 0;
        const bool prevKnown = (oldSt & 3u) != 0;
        o.pairs[L].touchChange = 0; o.pairs[L].patchChange = 0;
        o.pairs[L].allflagsWord = allflags; o.pairs[L].nbContactsFlags = origW36[L];
        // valid pairs have the invalidation bit cleared; invalid pairs are
        // never touched by the shader, so they keep the uploaded raw value
        o.pairs[L].manifoldNb = validL[L] ? (pr[L].rawNb & ~0x80000000u) : pr[L].rawNb;
        o.pairs[L].contactPatchesOff = 0xFFFFFFFFu; o.pairs[L].patchWritten = false;
        if (validL[L])
        {
            const uint32_t nP = nbL[L] ? 1u : 0u;
            o.pairs[L].touchChange = ((prevTouch[L] != (nbL[L] != 0)) || !prevKnown) ? 1u : 0u;
            o.pairs[L].patchChange = (prevP[L] != nP) ? 1u : 0u;
            o.pairs[L].allflagsWord = (((prevP[L] << 8) | st) << 16) | (nP << 8);
            o.pairs[L].nbContactsFlags = (origW36[L] & 0xFFFF0000u) | (nbL[L] & 0xFFFFu);
        }
        statusL[L] = st;
    }

    uint32_t patchMask = 0;
    for (uint32_t L = 0; L < 32; ++L) if (nbL[L] > 0) patchMask |= (1u << L);
    uint32_t patchIndex = 0xFFFFFFFFu;
    if (patchMask)
    {
        const uint32_t totalSz = 64 * popc(patchMask);
        patchIndex = patchesBytes; patchesBytes += totalSz;
        if ((patchIndex + totalSz) > patchLimit) { ovf |= 4u; patchIndex = 0xFFFFFFFFu; }
    }
    if (patchIndex == 0xFFFFFFFFu && patchMask)
    {
        patchMask = 0;
        for (uint32_t L = 0; L < 32; ++L)
        {
            scrNb[L] = 0;
            if (validL[L])
            {
                uint32_t st = (statusL[L] & ~3u) | 1u;
                o.pairs[L].allflagsWord = ((prevP[L] << 8) | st) << 16;
                o.pairs[L].nbContactsFlags = origW36[L] & 0xFFFF0000u;
                o.pairs[L].touchChange = prevTouch[L] ? 1u : 0u;
                o.pairs[L].patchChange = (prevP[L] != 0) ? 1u : 0u;
                statusL[L] = st;
            }
        }
    }
    for (uint32_t L = 0; L < 32; ++L)
        if (validL[L] && patchIndex != 0xFFFFFFFFu)
        {
            o.pairs[L].contactPatchesOff = patchIndex + 64 * popc(patchMask & ((1u << L) - 1u));
            o.pairs[L].patchWritten = true;
        }

    float restL[32], dynL[32], staL[32], dampL[32]; uint32_t mfL[32];
    for (uint32_t L = 0; L < 32; ++L)
    {
        restL[L] = dynL[L] = staL[L] = dampL[L] = 0.f; mfL[L] = 0;
        if (validL[L])
        {
            uint32_t f; float sf, df, re, da;
            combineMaterials(mats[scrMat0[L]], mats[scrMat1[L]], f, sf, df, re, da);
            restL[L] = re; dynL[L] = df; staL[L] = sf; dampL[L] = da; mfL[L] = f & 0xFF;
        }
    }

    const uint32_t ONE = f2u(1.0f);
    for (uint32_t L = 0; L < 32; ++L)
        if (patchMask & (1u << L))
        {
            const uint32_t slot = popc(patchMask & ((1u << L) - 1u));
            const size_t base = size_t(patchIndex) + size_t(slot) * 64;
            const uint32_t contactUnion = ((scrNb[L] & 0xFFu) << 16) | (mfL[L] << 24);
            const uint32_t materialUnion = (scrMat0[L] & 0xFFFFu) << 16;
            // Bytes 54-55: the CUDA stores MaterialParams2::pad (cudaGJKEPA.cu:1579), never written in
            // the .cu -> indeterminate __shared__; pinned to the port's documented 0.
            const uint32_t materialUnion2 = scrMat1[L] & 0xFFFFu;
            // The CUDA stores the WHOLE struct (cudaGJKEPA.cu:2003-2004): words 14-15 are
            // patchConstants[14..15] = 0.0f (:1912-1913), a real store of 0 -- NOT the prior content.
            uint32_t w[16] = { ONE, ONE, ONE, ONE,
                               f2u(scrNormX[L]), f2u(scrNormY[L]), f2u(scrNormZ[L]), f2u(restL[L]),
                               f2u(dynL[L]), f2u(staL[L]), f2u(dampL[L]), contactUnion,
                               materialUnion, materialUnion2, 0u, 0u };
            if (base + 64 <= o.patch.size()) std::memcpy(&o.patch[base], w, 64);
        }

    o.contactsBytes = contactsBytes; o.forceBytes = forceBytes;
    o.patchesBytes = patchesBytes; o.overflowError = ovf;
}

static std::vector<uint8_t> finLoader(const char* file) { return loadShaderBlob(file); }

PXG_TEST(npCvxFinishContacts_engineAbi)
{
    using namespace physx;

    std::mt19937 rng(152001);
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto randQuat = [&]() { V3 ax = unit(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) }); const float a = rf(-3.1f, 3.1f); return F4{ ax.x * std::sin(a * .5f), ax.y * std::sin(a * .5f), ax.z * std::sin(a * .5f), std::cos(a * .5f) }; };

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npCvxFinishContacts.spv", "finishContactsKernel", 152u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &finLoader);

    const uint32_t NMAT = 64;
    std::vector<MatRef> mats(NMAT);
    for (auto& m : mats)
    {
        m.dyn = rf(0, 1.5f); m.sta = rf(0, 2.f); m.rest = rf(-1.f, 1.f); m.damp = rf(0, 2.f);
        m.flags = (rng() % 4 == 0 ? 1u : 0u) | (rng() % 3 == 0 ? 16u : 0u) | (rng() % 5 == 0 ? 256u : 0u);
        m.fricMode = rng() % 4; m.restMode = rng() % 4; m.dampMode = rng() % 4;
    }
    std::vector<PxsMaterialData> matData(NMAT);
    for (uint32_t i = 0; i < NMAT; ++i)
    {
        matData[i].dynamicFriction = mats[i].dyn; matData[i].staticFriction = mats[i].sta;
        matData[i].restitution = mats[i].rest; matData[i].damping = mats[i].damp;
        matData[i].flags = PxMaterialFlags(PxU16(mats[i].flags));
        matData[i].fricCombineMode = PxU8(mats[i].fricMode);
        matData[i].restCombineMode = PxU8(mats[i].restMode);
        matData[i].dampingCombineMode = PxU8(mats[i].dampMode);
    }

    const size_t BUF = 8192;
    std::vector<uint8_t> poisonImg;
    fillPoison(poisonImg, BUF);
    PxgDevicePtr dCmIn = 0, dShapes = 0, dCache = 0, dCd = 0, dMf = 0, dMat = 0;
    PxgDevicePtr dPoints = 0, dPatch = 0, dForce = 0, dCounters = 0, dTouch = 0, dPatchCh = 0, dCmOut = 0;
    if (shim.memAlloc(&dCmIn, 32 * sizeof(PxgContactManagerInput)) != PxgResult::Success || !dCmIn)
    { std::printf("  NOTE: no BDA -> skip\n"); return; }
    shim.memAlloc(&dShapes, 64 * sizeof(PxgShape));
    shim.memAlloc(&dCache, 64 * sizeof(PxsCachedTransform));
    shim.memAlloc(&dCd, 64 * 4);
    shim.memAlloc(&dMf, 32 * sizeof(PxgPersistentContactManifold));
    shim.memAlloc(&dMat, NMAT * sizeof(PxsMaterialData));
    shim.memAlloc(&dPoints, BUF);
    shim.memAlloc(&dPatch, BUF);
    shim.memAlloc(&dForce, BUF);
    shim.memAlloc(&dCounters, sizeof(PxgPatchAndContactCounters));
    shim.memAlloc(&dTouch, 32 * 4);
    shim.memAlloc(&dPatchCh, 32 * 4);
    shim.memAlloc(&dCmOut, 32 * sizeof(PxsContactManagerOutput));
    shim.memcpyHtoD(dMat, matData.data(), NMAT * sizeof(PxsMaterialData));

    const uint32_t NLAUNCH = 140;
    size_t cOvf = 0, cPatchOvf = 0, cInval = 0, cAvg = 0, cInvalid = 0, cEmpty = 0;
    size_t mmMf = 0, mmOut = 0, mmFlags = 0, mmCnt = 0, mmPoints = 0, mmPatch = 0;

    for (uint32_t launch = 0; launch < NLAUNCH; ++launch)
    {
        const uint32_t numTests = 24 + (rng() % 9);
        const bool insertAvg = (launch % 2 == 0);
        if (insertAvg) ++cAvg;

        std::vector<Pair> pr(32);
        uint32_t nPatch = 0;
        for (uint32_t i = 0; i < 32; ++i)
        {
            Pair& p = pr[i];
            const uint32_t kinds[4] = { 0u, 2u, 3u, 5u };
            p.type0 = kinds[rng() % 4]; p.type1 = kinds[rng() % 4];
            p.mat0 = rng() % NMAT; p.mat1 = rng() % NMAT;
            p.scale0 = V3{ rf(0.5f, 2), rf(0.5f, 2), rf(0.5f, 2) };
            p.scale1 = V3{ rf(0.5f, 2), rf(0.5f, 2), rf(0.5f, 2) };
            p.t0q = randQuat(); p.t1q = randQuat();
            p.t0p = V3{ rf(-2, 2), rf(-2, 2), rf(-2, 2) };
            p.t1p = V3{ rf(-2, 2), rf(-2, 2), rf(-2, 2) };
            p.cd0 = rf(0.05f, 0.3f); p.cd1 = rf(0.05f, 0.3f);
            uint32_t nb = rng() % 5;
            if (i < numTests && nb == 0) ++cEmpty;
            for (int j = 0; j < 4; ++j)
            {
                const V3 n = unit(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) });
                p.normPen[j] = F4{ n.x, n.y, n.z, rf(-0.6f, 0.5f) };
                p.pointB[j] = F4{ rf(-1.5f, 1.5f), rf(-1.5f, 1.5f), rf(-1.5f, 1.5f), 0 };
            }
            uint32_t raw = nb;
            if (rng() % 7 == 0) { raw |= 0x80000000u; ++cInval; }
            p.rawNb = raw;
            p.allflags = rng();
            p.nbContactsFlags = rng();
            if (i < numTests && (raw & ~0x80000000u) != 0) ++nPatch;
        }

        const uint32_t initC = 16 * (rng() % 16), initF = 4 * (rng() % 16), initP = 64 * (rng() % 4);
        uint32_t cLimit = initC + 16 * 200, fLimit = initF + 4 * 200, pLimit = initP + 64 * 40;
        if (launch % 11 == 1) { cLimit = initC + 16 * (rng() % 20); ++cOvf; }
        else if (launch % 11 == 2) { pLimit = initP + 64 * (rng() % (nPatch ? nPatch : 1)); ++cPatchOvf; }
        else if (launch % 11 == 3) { fLimit = initF + 4 * (rng() % 20); }

        cInvalid += (32 - numTests);

        FinishRef ref;
        refFinish(pr, numTests, insertAvg, initC, initF, initP, cLimit, fLimit, pLimit, mats, BUF, ref);

        std::vector<PxgContactManagerInput> cmIn(32);
        std::vector<PxgShape> shapes(64);
        std::vector<PxsCachedTransform> cache(64);
        std::vector<float> cd(64);
        std::vector<PxgPersistentContactManifold> mf(32);
        std::vector<PxsContactManagerOutput> cmOut(32);
        std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShape));
        std::memset(cache.data(), 0, cache.size() * sizeof(PxsCachedTransform));
        std::memset(mf.data(), 0, mf.size() * sizeof(PxgPersistentContactManifold));
        std::memset(cmOut.data(), 0, cmOut.size() * sizeof(PxsContactManagerOutput));
        for (uint32_t i = 0; i < 32; ++i)
        {
            const Pair& p = pr[i];
            cmIn[i].shapeRef0 = 2 * i; cmIn[i].shapeRef1 = 2 * i + 1;
            cmIn[i].transformCacheRef0 = 2 * i; cmIn[i].transformCacheRef1 = 2 * i + 1;
            shapes[2 * i].type = p.type0; shapes[2 * i].materialIndex = p.mat0;
            shapes[2 * i].scale.scale = PxVec3(p.scale0.x, p.scale0.y, p.scale0.z);
            shapes[2 * i].scale.rotation = PxQuat(PxIdentity);
            shapes[2 * i + 1].type = p.type1; shapes[2 * i + 1].materialIndex = p.mat1;
            shapes[2 * i + 1].scale.scale = PxVec3(p.scale1.x, p.scale1.y, p.scale1.z);
            shapes[2 * i + 1].scale.rotation = PxQuat(PxIdentity);
            cache[2 * i].transform.q = PxQuat(p.t0q.x, p.t0q.y, p.t0q.z, p.t0q.w);
            cache[2 * i].transform.p = PxVec3(p.t0p.x, p.t0p.y, p.t0p.z);
            cache[2 * i + 1].transform.q = PxQuat(p.t1q.x, p.t1q.y, p.t1q.z, p.t1q.w);
            cache[2 * i + 1].transform.p = PxVec3(p.t1p.x, p.t1p.y, p.t1p.z);
            cd[2 * i] = p.cd0; cd[2 * i + 1] = p.cd1;
            for (int j = 0; j < 4; ++j)
            {
                mf[i].mLocalNormal_PenW[j] = make_float4(p.normPen[j].x, p.normPen[j].y, p.normPen[j].z, p.normPen[j].w);
                mf[i].mLocalContactB[j] = make_float4(p.pointB[j].x, p.pointB[j].y, p.pointB[j].z, 0);
            }
            mf[i].mNbContacts = p.rawNb;
            std::memcpy(&cmOut[i].allflagsStart, &p.allflags, 4);
            std::memcpy(&cmOut[i].nbContacts, &p.nbContactsFlags, 4);
        }

        PxgPatchAndContactCounters ctr{};
        ctr.patchesBytes = initP; ctr.contactsBytes = initC; ctr.forceAndIndiceBytes = initF; ctr.overflowError = 0;

        shim.memcpyHtoD(dCmIn, cmIn.data(), 32 * sizeof(PxgContactManagerInput));
        shim.memcpyHtoD(dShapes, shapes.data(), 64 * sizeof(PxgShape));
        shim.memcpyHtoD(dCache, cache.data(), 64 * sizeof(PxsCachedTransform));
        shim.memcpyHtoD(dCd, cd.data(), 64 * 4);
        shim.memcpyHtoD(dMf, mf.data(), 32 * sizeof(PxgPersistentContactManifold));
        shim.memcpyHtoD(dCmOut, cmOut.data(), 32 * sizeof(PxsContactManagerOutput));
        shim.memcpyHtoD(dCounters, &ctr, sizeof(ctr));
        shim.memcpyHtoD(dPoints, poisonImg.data(), BUF);   // prior content (refFinish expects it)
        shim.memcpyHtoD(dPatch, poisonImg.data(), BUF);
        shim.memsetD32(dTouch, 0xEEEEEEEE, 32);
        shim.memsetD32(dPatchCh, 0xEEEEEEEE, 32);

        struct FinBlob
        {
            uint32_t numTests;
            uint64_t cmInputs, cmOutputs, cmManifolds, shapes, transformCache, contactDistance, materials,
                     contactStream, patchStream;
            uint32_t insertAveragePointWord;
            uint64_t patchAndContactCounters, touchChangeFlags, patchChangeFlags,
                     startContactPatches, startContactPoints, startContactForces;
            uint32_t patchBytesLimit, contactBytesLimit, forceBytesLimit;
        } blob = { numTests, dCmIn, dCmOut, dMf, dShapes, dCache, dCd, dMat, dPoints, dPatch,
                   insertAvg ? 1u : 0u, dCounters, dTouch, dPatchCh, dPatch, dPoints, dForce,
                   pLimit, cLimit, fLimit };
        static_assert(sizeof(FinBlob) == 152, "finishContacts engine blob is 152 bytes");

        PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &blob, sizeof(blob)) == PxgResult::Success,
                      "launch finishContacts (engine ABI)");
        shim.streamSynchronize();

        std::vector<PxgPersistentContactManifold> gMf(32);
        std::vector<PxsContactManagerOutput> gOut(32);
        std::vector<uint32_t> gTouch(32), gPatchCh(32);
        PxgPatchAndContactCounters gCtr{};
        std::vector<uint8_t> gPoints(BUF), gPatch(BUF);
        shim.memcpyDtoH(gMf.data(), dMf, 32 * sizeof(PxgPersistentContactManifold));
        shim.memcpyDtoH(gOut.data(), dCmOut, 32 * sizeof(PxsContactManagerOutput));
        shim.memcpyDtoH(gTouch.data(), dTouch, 32 * 4);
        shim.memcpyDtoH(gPatchCh.data(), dPatchCh, 32 * 4);
        shim.memcpyDtoH(&gCtr, dCounters, sizeof(gCtr));
        shim.memcpyDtoH(gPoints.data(), dPoints, BUF);
        shim.memcpyDtoH(gPatch.data(), dPatch, BUF);

        for (uint32_t i = 0; i < 32; ++i)
        {
            const PairOut& r = ref.pairs[i];
            if (gMf[i].mNbContacts != r.manifoldNb) ++mmMf;

            uint32_t gAll; std::memcpy(&gAll, &gOut[i].allflagsStart, 4);
            uint32_t gW36; std::memcpy(&gW36, &gOut[i].nbContacts, 4);
            const uint64_t gPts = uint64_t(reinterpret_cast<uintptr_t>(gOut[i].contactPoints));
            const uint64_t gFrc = uint64_t(reinterpret_cast<uintptr_t>(gOut[i].contactForces));
            const uint64_t gPch = uint64_t(reinterpret_cast<uintptr_t>(gOut[i].contactPatches));

            const bool valid = i < numTests;
            if (valid)
            {
                const uint64_t expPts = uint64_t(dPoints) + uint64_t(r.contactPointsOff);
                const uint64_t expFrc = uint64_t(dForce) + uint64_t(r.contactForcesOff);
                if (gPts != expPts || gFrc != expFrc) ++mmOut;
                if (gAll != r.allflagsWord || gW36 != r.nbContactsFlags) ++mmFlags;
                if (gTouch[i] != r.touchChange || gPatchCh[i] != r.patchChange) ++mmFlags;
                const uint64_t expPch = r.patchWritten ? (uint64_t(dPatch) + uint64_t(r.contactPatchesOff)) : 0ull;
                if (gPch != expPch) ++mmOut;
            }
        }
        if (gCtr.patchesBytes != ref.patchesBytes || gCtr.contactsBytes != ref.contactsBytes ||
            gCtr.forceAndIndiceBytes != ref.forceBytes || gCtr.overflowError != ref.overflowError) ++mmCnt;
        if (std::memcmp(gPoints.data(), ref.points.data(), BUF) != 0) ++mmPoints;
        // patch payloads: 64-byte chunks; integer words exact, float words
        // (norm x/y/z, restitution, dyn/sta friction, damping) to tolerance —
        // the norm carries a GPU/CPU 1/sqrt whose last bit can differ (mirrors
        // the writeStream port's tolerance-compared avgNorm).
        for (size_t off = 0; off + 64 <= BUF; off += 64)
        {
            const uint32_t* r = reinterpret_cast<const uint32_t*>(&ref.patch[off]);
            const uint32_t* g = reinterpret_cast<const uint32_t*>(&gPatch[off]);
            // A chunk no patch covers must come back exactly as uploaded.  A written patch can never
            // look like the poison: word 0 is 1.0f.
            bool untouched = true;
            for (int k = 0; k < 16; ++k) if (r[k] != finPoison(off / 4 + k)) { untouched = false; break; }
            if (untouched)
            {
                if (std::memcmp(r, g, 64) != 0) ++mmPatch;
                continue;
            }
            bool bad = false;
            for (int k = 0; k < 16; ++k)
            {
                const bool floatWord = (k >= 4 && k <= 10); // norm, rest, dyn, sta, damp
                if (floatWord)
                {
                    float rf2, gf2; std::memcpy(&rf2, &r[k], 4); std::memcpy(&gf2, &g[k], 4);
                    if (!almostEqual(rf2, gf2, 2e-4f, 2e-5f)) { bad = true; break; }
                }
                else if (r[k] != g[k]) { bad = true; break; }
            }
            if (bad) ++mmPatch;
        }
    }

    PXG_CHECK_MSG(mmMf == 0, "finishContacts: %zu manifold-nb mismatches", mmMf);
    PXG_CHECK_MSG(mmOut == 0, "finishContacts: %zu cmOutputs offset mismatches", mmOut);
    PXG_CHECK_MSG(mmFlags == 0, "finishContacts: %zu flag/allflags mismatches", mmFlags);
    PXG_CHECK_MSG(mmCnt == 0, "finishContacts: %zu counter mismatches", mmCnt);
    PXG_CHECK_MSG(mmPoints == 0, "finishContacts: %zu contact-stream mismatches", mmPoints);
    PXG_CHECK_MSG(mmPatch == 0, "finishContacts: %zu patch-payload mismatches", mmPatch);
    PXG_CHECK_MSG(cOvf > 0 && cPatchOvf > 0 && cInval > NLAUNCH && cAvg > NLAUNCH / 3 && cInvalid > 0 && cEmpty > 0,
                  "finishContacts: coverage thin (cOvf %zu patchOvf %zu inval %zu avg %zu invalid %zu empty %zu)",
                  cOvf, cPatchOvf, cInval, cAvg, cInvalid, cEmpty);

    for (PxgDevicePtr pfree : { dCmIn, dShapes, dCache, dCd, dMf, dMat, dPoints, dPatch, dForce,
                                dCounters, dTouch, dPatchCh, dCmOut })
        shim.memFree(pfree);
}

} // namespace
