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

// Verification of npCvxCollidePrep.slang: the collide() driver prep glue from
// cudaGJKEPA.cu — prepareVertices (vertex2Shape + aToB.transform for A),
// warm-start unpack/seed (lanes 13/14/15 -> CachedVertex(a, b, vA[a]-vB[b]),
// indices = 0x3f800000|b<<8|a), the runGJK warm-start re-pack OR-reduction, and
// the initialDir selection.  Exact CPU transcription; vertices/cached vectors
// to tolerance, all packed indices and sizes exact.

#include "pxgTestFramework.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct V3 { float x, y, z; };
struct PrepIn
{
    F4 rawA[32], rawB[32];
    F4 scale0, rot0, scale1, rot1, aToBq, aToBp;
    uint32_t wsLane13, wsLane14, wsLane15, pad0;
};
struct PrepOut
{
    F4 vA[32], vB[32];
    F4 cachedV[4];
    uint32_t cachedIdx[4];
    uint32_t cachedSize, rePack13, rePack14, rePack15;
    F4 initialDir;
};

static inline V3 v3(const F4& f) { return V3{ f.x, f.y, f.z }; }
static inline V3 add(const V3& a, const V3& b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static inline V3 sub(const V3& a, const V3& b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static inline V3 mul3(const V3& a, const V3& b) { return V3{ a.x * b.x, a.y * b.y, a.z * b.z }; }
static inline V3 mul(const V3& a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; }
static inline float dot3(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 unit(const V3& a) { float l = std::sqrt(dot3(a, a)); return l > 0 ? mul(a, 1.f / l) : a; }
static V3 qRot(const F4& q, const V3& v) { const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z, w2 = q.w * q.w - 0.5f, d2 = q.x * vx + q.y * vy + q.z * vz; return V3{ vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d2, vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d2, vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d2 }; }
static V3 qRotInv(const F4& q, const V3& v) { const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z, w2 = q.w * q.w - 0.5f, d2 = q.x * vx + q.y * vy + q.z * vz; return V3{ vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * d2, vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * d2, vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * d2 }; }
static V3 vertex2Shape(const V3& v, const V3& scale, const F4& rot) { return qRotInv(rot, mul3(qRot(rot, v), scale)); }

static void refPrep(const PrepIn& s, PrepOut& o)
{
    const uint32_t nbV0 = (uint32_t)s.scale0.w;
    const uint32_t nbV1 = (uint32_t)s.scale1.w;
    const V3 aToBp = v3(s.aToBp);

    V3 vA[32], vB[32];
    for (uint32_t i = 0; i < 32; ++i) { vA[i] = V3{ 0, 0, 0 }; vB[i] = V3{ 0, 0, 0 }; }
    for (uint32_t i = 0; i < nbV0; ++i)
        vA[i] = add(qRot(s.aToBq, vertex2Shape(v3(s.rawA[i]), v3(s.scale0), s.rot0)), aToBp);
    for (uint32_t i = 0; i < nbV1; ++i)
        vB[i] = vertex2Shape(v3(s.rawB[i]), v3(s.scale1), s.rot1);

    const uint32_t nWs = s.wsLane13;
    V3 cachedV[4]; uint32_t cachedIdx[4];
    for (uint32_t j = 0; j < 4; ++j) { cachedV[j] = V3{ 0, 0, 0 }; cachedIdx[j] = 0; }
    for (uint32_t tI = 0; tI < nWs; ++tI)
    {
        const uint32_t a = (s.wsLane14 >> (tI << 3)) & 0xFF;
        const uint32_t b = (s.wsLane15 >> (tI << 3)) & 0xFF;
        cachedV[tI] = sub(vA[a], vB[b]);
        cachedIdx[tI] = 0x3f800000u | (b << 8) | a;
    }

    uint32_t rePackA = 0, rePackB = 0;
    for (uint32_t tI = 0; tI < nWs; ++tI)
    {
        const uint32_t a = cachedIdx[tI] & 0xFF;
        const uint32_t b = (cachedIdx[tI] >> 8) & 0xFF;
        rePackA |= (a << (tI << 3));
        rePackB |= (b << (tI << 3));
    }

    V3 initialDir{ 1, 0, 0 };
    if (dot3(aToBp, aToBp) > 0.f) initialDir = aToBp;

    for (uint32_t i = 0; i < 32; ++i)
    {
        o.vA[i] = F4{ vA[i].x, vA[i].y, vA[i].z, 0 };
        o.vB[i] = F4{ vB[i].x, vB[i].y, vB[i].z, 0 };
    }
    for (uint32_t j = 0; j < 4; ++j) { o.cachedV[j] = F4{ cachedV[j].x, cachedV[j].y, cachedV[j].z, 0 }; o.cachedIdx[j] = cachedIdx[j]; }
    o.cachedSize = nWs;
    o.rePack13 = nWs; o.rePack14 = rePackA; o.rePack15 = rePackB;
    o.initialDir = F4{ initialDir.x, initialDir.y, initialDir.z, 0 };
}

static bool eqP(const F4& r, const F4& v) { return almostEqual(v.x, r.x, 1e-4f, 1e-5f) && almostEqual(v.y, r.y, 1e-4f, 1e-5f) && almostEqual(v.z, r.z, 1e-4f, 1e-5f); }

PXG_TEST(npCvxCollidePrep_prep)
{
    std::mt19937 rng(148001);
    const uint32_t N = 8000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto randQuat = [&]() { V3 ax = unit(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) }); const float a = rf(-3.1f, 3.1f); return F4{ ax.x * std::sin(a * .5f), ax.y * std::sin(a * .5f), ax.z * std::sin(a * .5f), std::cos(a * .5f) }; };

    std::vector<PrepIn> in(N);
    std::vector<PrepOut> ref(N);
    size_t zeroP = 0, ws0 = 0, wsFull = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        PrepIn s{};
        const uint32_t nbV0 = 4 + rng() % 29; // 4..32
        const uint32_t nbV1 = 4 + rng() % 29;
        for (uint32_t j = 0; j < 32; ++j)
        {
            s.rawA[j] = { rf(-2, 2), rf(-2, 2), rf(-2, 2), 0 };
            s.rawB[j] = { rf(-2, 2), rf(-2, 2), rf(-2, 2), 0 };
        }
        s.scale0 = { rf(0.5f, 2.f), rf(0.5f, 2.f), rf(0.5f, 2.f), float(nbV0) };
        s.scale1 = { rf(0.5f, 2.f), rf(0.5f, 2.f), rf(0.5f, 2.f), float(nbV1) };
        s.rot0 = randQuat();
        s.rot1 = randQuat();
        s.aToBq = randQuat();
        // exercise the |p|^2 == 0 initialDir fallback occasionally (exact zero)
        if (i % 16 == 0) { s.aToBp = { 0, 0, 0, 0 }; ++zeroP; }
        else s.aToBp = { rf(-2, 2), rf(-2, 2), rf(-2, 2), 0 };

        const uint32_t nWs = i % 5; // 0..4
        if (nWs == 0) ++ws0; else if (nWs == 4) ++wsFull;
        uint32_t wsA = 0, wsB = 0;
        for (uint32_t tI = 0; tI < 4; ++tI)
        {
            // garbage in unused high bytes must be masked out by the unpack
            const uint32_t a = (tI < nWs) ? (rng() % nbV0) : (rng() % 256);
            const uint32_t b = (tI < nWs) ? (rng() % nbV1) : (rng() % 256);
            wsA |= (a << (tI << 3));
            wsB |= (b << (tI << 3));
        }
        s.wsLane13 = nWs; s.wsLane14 = wsA; s.wsLane15 = wsB;
        in[i] = s;
        refPrep(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<PrepOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct PrepParams { uint32_t numTests; } prm = { N };
    Kernel k("npCvxCollidePrep.spv", "cvxCollidePrep",
             { (uint32_t)sizeof(PrepIn), (uint32_t)sizeof(PrepOut) }, sizeof(PrepParams));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<PrepOut>(dev, bOut, N);
    size_t mmV = 0, mmC = 0, mmP = 0, mmD = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const uint32_t nbV0 = (uint32_t)in[i].scale0.w, nbV1 = (uint32_t)in[i].scale1.w;
        for (uint32_t j = 0; j < nbV0; ++j) if (!eqP(ref[i].vA[j], g[i].vA[j])) { ++mmV; break; }
        for (uint32_t j = 0; j < nbV1; ++j) if (!eqP(ref[i].vB[j], g[i].vB[j])) { ++mmV; break; }
        // all 4 slots: seeded below cachedSize, deterministically zero above it
        for (uint32_t j = 0; j < 4; ++j)
            if (!eqP(ref[i].cachedV[j], g[i].cachedV[j]) || ref[i].cachedIdx[j] != g[i].cachedIdx[j]) { ++mmC; break; }
        if (g[i].cachedSize != ref[i].cachedSize || g[i].rePack13 != ref[i].rePack13 ||
            g[i].rePack14 != ref[i].rePack14 || g[i].rePack15 != ref[i].rePack15) ++mmP;
        if (!eqP(ref[i].initialDir, g[i].initialDir)) ++mmD;
    }
    PXG_CHECK_MSG(mmV == 0, "collidePrep: %zu prepared-vertex mismatches", mmV);
    PXG_CHECK_MSG(mmC == 0, "collidePrep: %zu cached-vertex mismatches", mmC);
    PXG_CHECK_MSG(mmP == 0, "collidePrep: %zu warm-start pack mismatches", mmP);
    PXG_CHECK_MSG(mmD == 0, "collidePrep: %zu initialDir mismatches", mmD);
    PXG_CHECK_MSG(zeroP > N / 32 && ws0 > N / 10 && wsFull > N / 10,
                  "collidePrep: coverage thin (zeroP %zu / ws0 %zu / wsFull %zu)", zeroP, ws0, wsFull);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
