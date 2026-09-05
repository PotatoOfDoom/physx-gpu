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

// Verification of npCvxHfNp.slang (sphereHeightfieldNarrowphaseCore +
// the warp output stage, convexHeightfield.cu, tranche 107) against an
// exact CPU transcription.  The heightfield twin of tranche 106: the
// triangle geometry is extracted from a shared integer-height sample
// grid (getTriangle with handedness), so verts, normals and closest
// points are all exact (integer cross products, power-of-two scales,
// IEEE sqrt).  The distinctive piece is the EDGE-CONCAVITY SWITCH: for
// an eE01/eE12/eE02 closest feature the adjacent triangle is
// re-extracted and the contact is regated by the sign of
// triNormalAdj . (oppositeVert - adjV0) — coplanar (flat) edges are
// rejected (dot==0, not <0), concave-valley edges are accepted.
//
// Fixtures place the sphere in shape space relative to the CPU-extracted
// triangle: face contacts (center = interior + t*triNormal, patch
// normal = triNormal exact), vertex deferrals (flat plateau, axis
// in-plane normals), flat-edge rejects (concavity dot==0), concave-edge
// accepts (valley), below-plane / too-far misses, and the strict accept
// boundary.  The warp output stage, second-pass and temp reservations,
// batch-drop quirk and disjoint-partition checks mirror tranche 106.
//
// KNOWN LIMIT: the 0.996 deferral boundary is unhittable exactly (as in
// tranche 103/106).  Capsule-heightfield is NOT ported (two confirmed
// upstream bugs: the flags read at convexHeightfield.cu:656-658 reads
// the low bits of the pointer variable, and p0/p1 mix world-space tmp
// with the mesh-space centre) — documented, deferred.

#include "pxgTestFramework.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };
struct ChCase
{
    F4 aQ, aP, mQ, mP, scale;
    U4 pair;
};
struct ChParams { uint32_t numCases, rows, columns, pad0; };

struct V3 { float x, y, z; };
static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 add(V3 a, V3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 mul(V3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
static V3 neg(V3 a) { return { -a.x, -a.y, -a.z }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b)
{
    return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}
static V3 v3(F4 f) { return { f.x, f.y, f.z }; }

static V3 quatRotate(F4 q, V3 v)
{
    const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d = q.x * vx + q.y * vy + q.z * vz;
    return { vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d,
             vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d,
             vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d };
}
static V3 quatRotateInv(F4 q, V3 v)
{
    const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d = q.x * vx + q.y * vy + q.z * vz;
    return { vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * d,
             vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * d,
             vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * d };
}
static F4 exactQuat(std::mt19937& rng)
{
    if (rng() % 4 == 0) return { 0, 0, 0, 1 };
    const uint32_t s = rng() % 16;
    return { (s & 1) ? -0.5f : 0.5f, (s & 2) ? -0.5f : 0.5f, (s & 4) ? -0.5f : 0.5f, (s & 8) ? -0.5f : 0.5f };
}

static const float FLT_BIG = 3.402823466e38f;
static const uint32_t BOUNDARY = 0xffffffffu;
static const uint32_t NONCONVEX = 0x80000000u;
static const uint32_t eV0 = 1u << 29, eV1 = 2u << 29, eV2 = 3u << 29;
static const uint32_t eE01 = 4u << 29, eE02 = 5u << 29, eE12 = 6u << 29;
static const uint32_t DEFER = 0x80000000u;
static const uint32_t FACE_FLAG = 0x80000000u;
static const uint32_t POISON = 0xCDCDCDCDu;

static V3 getNormalizedRef(V3 v)
{
    const float m = dot3(v, v);
    if (!(m > 0.0f)) return { 0, 0, 0 };
    return mul(v, 1.0f / std::sqrt(m));
}
static bool isEdgeNonconvexRef(uint32_t a) { return (a != BOUNDARY) && ((a & NONCONVEX) != 0u); }

// ---- heightfield ------------------------------------------------------------
// NON-SQUARE (rows != columns) so any row/column transposition in the index
// math (getVertex row=vi/cols col=vi%cols, the rows-2 / columns-2 adjacency
// guards) is observable; the pit/peak sit at ASYMMETRIC (row != col) cells so
// a transposed grid relocates them and changes the concavity structure.
static const uint32_t HF_ROWS = 9, HF_COLS = 6;
// pit/peak at ASYMMETRIC cells with the SAME (even) parity as the original
// square fixture, so the alternating tessellation rings them with the same mix
// of concave/convex edges across all three eE arms
static const uint32_t PIT_R = 2, PIT_C = 4, PEAK_R = 5, PEAK_C = 3;

static float hfHeight(const std::vector<uint32_t>& smp, uint32_t v) { return float(int16_t(smp[v] & 0xffffu)); }
static bool hfShared(const std::vector<uint32_t>& smp, uint32_t cell) { return (smp[cell] & 0x800000u) != 0u; }
static V3 hfVertex(const std::vector<uint32_t>& smp, uint32_t vi)
{
    return { float(vi / HF_COLS), hfHeight(smp, vi), float(vi % HF_COLS) };
}
static V3 hf2shape(V3 v, V3 s) { return { v.x * s.x, v.y * s.y, v.z * s.z }; }

static void triVertIdx(const std::vector<uint32_t>& smp, uint32_t tri, uint32_t& i0, uint32_t& i1, uint32_t& i2)
{
    const uint32_t cell = tri >> 1;
    const bool first = (tri & 1u) == 0u;
    if (hfShared(smp, cell))
    {
        if (first) { i0 = cell + HF_COLS; i1 = cell; i2 = cell + HF_COLS + 1u; }
        else       { i0 = cell + 1u; i1 = cell + HF_COLS + 1u; i2 = cell; }
    }
    else
    {
        if (first) { i0 = cell; i1 = cell + 1u; i2 = cell + HF_COLS; }
        else       { i0 = cell + HF_COLS + 1u; i1 = cell + HF_COLS; i2 = cell + 1u; }
    }
}
static void triAdjIdx(const std::vector<uint32_t>& smp, uint32_t tri, uint32_t& a0, uint32_t& a1, uint32_t& a2)
{
    const uint32_t cell = tri >> 1;
    const bool first = (tri & 1u) == 0u;
    if (hfShared(smp, cell))
    {
        if (first)
        {
            a0 = BOUNDARY; a1 = tri + 1u; a2 = BOUNDARY;
            if (cell % HF_COLS != 0u)         a0 = tri - 1u;
            if (cell / HF_COLS != HF_ROWS - 2u) a2 = ((cell + HF_COLS) * 2u) + 1u;
        }
        else
        {
            a0 = BOUNDARY; a1 = tri - 1u; a2 = BOUNDARY;
            if (cell % HF_COLS < HF_COLS - 2u) a0 = tri + 1u;
            if (cell >= HF_COLS - 1u)          a2 = (cell - HF_COLS) * 2u;
        }
    }
    else
    {
        if (first)
        {
            a0 = BOUNDARY; a1 = tri + 1u; a2 = BOUNDARY;
            if (cell >= HF_COLS - 1u)          a0 = ((cell - HF_COLS) * 2u) + 1u;
            if (cell % HF_COLS != 0u)          a2 = tri - 1u;
        }
        else
        {
            a0 = BOUNDARY; a1 = tri - 1u; a2 = BOUNDARY;
            if (cell / HF_COLS != HF_ROWS - 2u) a0 = (cell + HF_COLS) * 2u;
            if (cell % HF_COLS < HF_COLS - 2u) a2 = tri + 1u;
        }
    }
}
static void getTriangleHf(const std::vector<uint32_t>& smp, uint32_t tri, V3 scale,
                          V3& v0, V3& v1, V3& v2, U4& adj)
{
    bool wrong = false;
    if (scale.z < 0.0f) wrong = !wrong;
    if (scale.x < 0.0f) wrong = !wrong;
    uint32_t t0, t1, t2, a0, a1, a2;
    triVertIdx(smp, tri, t0, t1, t2);
    triAdjIdx(smp, tri, a0, a1, a2);
    adj = { wrong ? a2 : a0, a1, wrong ? a0 : a2, 0 };
    v0 = hf2shape(hfVertex(smp, t0), scale);
    v1 = hf2shape(hfVertex(smp, wrong ? t2 : t1), scale);
    v2 = hf2shape(hfVertex(smp, wrong ? t1 : t2), scale);
}

static float distPointTriRef(V3 p, V3 a, V3 b, V3 c, U4 adj, V3& closestP,
                             bool& gen, bool& face, uint32_t& mask)
{
    face = false;
    const V3 ab = sub(b, a), ac = sub(c, a), ap = sub(p, a);
    const float d1 = dot3(ab, ap), d2 = dot3(ac, ap);
    const V3 bp = sub(p, b);
    const float d3 = dot3(ab, bp), d4 = dot3(ac, bp);
    const V3 cp = sub(p, c);
    const float d5 = dot3(ab, cp), d6 = dot3(ac, cp);
    const float vc = d1 * d4 - d3 * d2, vb = d5 * d2 - d1 * d6, va = d3 * d6 - d5 * d4;
    const float edgeTol = (va + vb + vc) * 1e-5f;
    if (d1 < 0.0f && d2 < 0.0f) { mask = eV0; gen = (!isEdgeNonconvexRef(adj.x)) || (!isEdgeNonconvexRef(adj.z)); closestP = a; return dot3(ap, ap); }
    if (d3 > 0.0f && d4 < d3)   { mask = eV1; gen = (!isEdgeNonconvexRef(adj.x)) || (!isEdgeNonconvexRef(adj.y)); closestP = b; return dot3(bp, bp); }
    if (d6 > 0.0f && d5 < d6)   { mask = eV2; gen = (!isEdgeNonconvexRef(adj.y)) || (!isEdgeNonconvexRef(adj.z)); closestP = c; return dot3(cp, cp); }
    if (vc < 0.0f && d1 >= 0.0f && d3 <= 0.0f) { mask = eE01; gen = !isEdgeNonconvexRef(adj.x) || vc > -edgeTol; const float v = d1 / (d1 - d3); closestP = add(a, mul(ab, v)); const V3 d = sub(closestP, p); return dot3(d, d); }
    if (vb < 0.0f && d2 >= 0.0f && d6 <= 0.0f) { mask = eE02; gen = !isEdgeNonconvexRef(adj.z) || vb > -edgeTol; const float w = d2 / (d2 - d6); closestP = add(a, mul(ac, w)); const V3 d = sub(closestP, p); return dot3(d, d); }
    if (va < 0.0f && d4 >= d3 && d5 >= d6)     { mask = eE12; gen = !isEdgeNonconvexRef(adj.y) || va > -edgeTol; const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6)); closestP = add(b, mul(sub(c, b), w)); const V3 d = sub(closestP, p); return dot3(d, d); }
    gen = true; face = true; mask = 0u;
    // THE MIRROR IS WRITTEN FROM THE CUDA, NOT FROM THE SHADER (P9.53).
    // sphereTriangle.cuh:138 is
    //     closestP = a + (ab * vb + ac * vc) / (va + vb + vc);
    // and PxVec3::operator/(Type f) (PxVec3.h:218-223) is `f = 1/f; return
    // (x*f, y*f, z*f)`.  So: ONE vector sum, ONE reciprocal, three multiplies,
    // and ONE add onto `a` -- not two adds, and not two scaled scalars.
    //
    // P9.36 rewrote this mirror to GuDistancePointTriangle.h's tail in the same
    // commit that rewrote the shader, AFTER the shader change turned this fixture
    // red on 123 cases.  That is the trap the same commit message names: a mirror
    // written from the Slang by the same hand at the same time proves only that
    // shader and mirror agree with each other.  Restoring the shader to the CUDA
    // turns this fixture red on the SAME 123 cases -- measured, not assumed.
    const float denom = 1.0f / (va + vb + vc);
    closestP = add(a, mul(add(mul(ab, vb), mul(ac, vc)), denom));
    const V3 d = sub(closestP, p); return dot3(d, d);
}

struct ChRef
{
    uint32_t nb = 0, om = 0, niIndex = 0, ordered = 0, pairOffset = 0, padOffset = 0, gpuTri = 0;
    float sep = FLT_BIG;
    V3 niNormal{};
    bool delay = false;
    V3 v0{}, v1{}, v2{};
    V3 triNormal{};
};

static ChRef refSphere(const ChCase& cs, const std::vector<uint32_t>& smp, const std::vector<U4>& pairRec)
{
    ChRef ro;
    const uint32_t triangleIdx = cs.pair.y;
    ro.gpuTri = triangleIdx;
    const U4 rec = pairRec[cs.pair.x];
    ro.pairOffset = rec.x + cs.pair.z;
    ro.padOffset = rec.y + cs.pair.z;
    const float contactDist = cs.aP.w, radius = cs.scale.w;
    const V3 scale = v3(cs.scale);
    V3 v0, v1, v2; U4 adj;
    getTriangleHf(smp, triangleIdx, scale, v0, v1, v2, adj);
    ro.v0 = v0; ro.v1 = v1; ro.v2 = v2;
    const V3 triNormal = getNormalizedRef(cross3(sub(v1, v0), sub(v2, v0)));
    ro.triNormal = triNormal;
    const V3 center = quatRotateInv(cs.mQ, sub(v3(cs.aP), v3(cs.mP)));
    const float inflated = radius + contactDist;
    const float dist = dot3(sub(center, v0), triNormal);
    float separation = FLT_BIG;
    V3 patchNormal = { 0, 0, 0 };
    V3 closestP;
    bool gen = false, face = false;
    uint32_t mask;
    const float sqDist = distPointTriRef(center, v0, v1, v2, adj, closestP, gen, face, mask);
    const float sqInflated = inflated * inflated;
    if (dist >= 0.0f && sqInflated > sqDist && gen)
    {
        auto adjNormal = [&](uint32_t at) { V3 a0, a1, a2; U4 d; getTriangleHf(smp, at, scale, a0, a1, a2, d); return getNormalizedRef(cross3(sub(a1, a0), sub(a2, a0))); };
        auto adjV0 = [&](uint32_t at) { V3 a0, a1, a2; U4 d; getTriangleHf(smp, at, scale, a0, a1, a2, d); return a0; };
        if (mask == eE01 && adj.x != BOUNDARY) gen = dot3(adjNormal(adj.x), sub(v2, adjV0(adj.x))) < 0.0f;
        else if (mask == eE12 && adj.y != BOUNDARY) gen = dot3(adjNormal(adj.y), sub(v0, adjV0(adj.y))) < 0.0f;
        else if (mask == eE02 && adj.z != BOUNDARY) gen = dot3(adjNormal(adj.z), sub(v1, adjV0(adj.z))) < 0.0f;

        if (gen)
        {
            patchNormal = triNormal;
            if (!face) patchNormal = getNormalizedRef(sub(center, closestP));
            const float cosTheta = dot3(patchNormal, triNormal);
            ro.delay = cosTheta <= 0.996f;
            if (ro.delay) ro.om = cs.pair.w | mask;
            ro.nb = 1;
            separation = std::sqrt(sqDist);
        }
    }
    ro.sep = separation;
    ro.ordered = (ro.nb != 0 && !ro.delay) ? (FACE_FLAG | triangleIdx) : triangleIdx;
    const V3 worldNormal = quatRotate(cs.mQ, neg(patchNormal));
    ro.niNormal = quatRotateInv(cs.aQ, worldNormal);
    ro.niIndex = (ro.delay ? DEFER : 0u) + (ro.nb << 27) + triangleIdx;
    return ro;
}

struct ChRun
{
    std::vector<F4> ni, temp, dbg;
    std::vector<float> depth;
    std::vector<uint32_t> interm, ordered, second, ctsIdx;
    uint32_t counters[2];
};
static ChRun runCh(const std::vector<ChCase>& cases, const std::vector<uint32_t>& samples,
                   const std::vector<U4>& pairRec, uint32_t poolN, uint32_t spCap, uint32_t tcCap)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t n = (uint32_t)cases.size();
    std::vector<F4> niInit(poolN, { -7, -7, -7, -7 });
    std::vector<float> depthInit(poolN, -7.0f);
    std::vector<uint32_t> intermInit(poolN, POISON), orderedInit(poolN, POISON), ctsIdxInit(poolN, POISON);
    std::vector<uint32_t> spInit(spCap, POISON);
    std::vector<F4> tcInit(tcCap, { -777, -777, -777, -777 });
    std::vector<uint32_t> ctrInit = { 0, 0 };
    std::vector<F4> dbgInit(n, { -7, -7, -7, -7 });

    PxgBufferHandle bC = createBufferFrom(dev, cases);
    PxgBufferHandle bS = createBufferFrom(dev, samples);
    PxgBufferHandle bP = createBufferFrom(dev, pairRec);
    PxgBufferHandle bNI = createBufferFrom(dev, niInit);
    PxgBufferHandle bD = createBufferFrom(dev, depthInit);
    PxgBufferHandle bI = createBufferFrom(dev, intermInit);
    PxgBufferHandle bO = createBufferFrom(dev, orderedInit);
    PxgBufferHandle bSp = createBufferFrom(dev, spInit);
    PxgBufferHandle bTC = createBufferFrom(dev, tcInit);
    PxgBufferHandle bCI = createBufferFrom(dev, ctsIdxInit);
    PxgBufferHandle bCt = createBufferFrom(dev, ctrInit);
    PxgBufferHandle bDbg = createBufferFrom(dev, dbgInit);
    ChParams prm = { n, HF_ROWS, HF_COLS, 0 };
    Kernel k("npCvxHfNp.spv", "chSphereTri",
             { (uint32_t)sizeof(ChCase), 4, 16, 16, 4, 4, 4, 4, 16, 4, 4, 16 },
             sizeof(ChParams));
    const uint32_t batches = (n + 31) / 32;
    k.dispatch((batches + 63) / 64,
               { bC, bS, bP, bNI, bD, bI, bO, bSp, bTC, bCI, bCt, bDbg }, prm);
    dev->flush();
    ChRun r;
    r.ni = downloadBuffer<F4>(dev, bNI, poolN);
    r.depth = downloadBuffer<float>(dev, bD, poolN);
    r.interm = downloadBuffer<uint32_t>(dev, bI, poolN);
    r.ordered = downloadBuffer<uint32_t>(dev, bO, poolN);
    r.second = downloadBuffer<uint32_t>(dev, bSp, spCap);
    r.temp = downloadBuffer<F4>(dev, bTC, tcCap);
    r.ctsIdx = downloadBuffer<uint32_t>(dev, bCI, poolN);
    const auto ctr = downloadBuffer<uint32_t>(dev, bCt, 2);
    r.counters[0] = ctr[0]; r.counters[1] = ctr[1];
    r.dbg = downloadBuffer<F4>(dev, bDbg, n);
    for (PxgBufferHandle h : { bC, bS, bP, bNI, bD, bI, bO, bSp, bTC, bCI, bCt, bDbg })
        dev->destroyBuffer(h);
    return r;
}

static uint32_t asU32(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

PXG_TEST(npCvxHfNp_sphere)
{
    // NC deliberately NOT a multiple of 32 so the last warp is PARTIAL: its
    // high lanes are inactive (ci >= numCases), exercising the laneActive
    // masking in warpOutputStage (a bug counting inactive lanes into
    // count/total would otherwise slip through a full-warp-only test).
    const uint32_t NC = 1000;
    std::mt19937 rng(1070701);

    // heightfield: left half flat plateau at height 4 (tessFlag varied),
    // right half a stepped ramp creating tilted triangles + concave valleys.
    // heightfield: FLAT height 0 everywhere except a PIT (one vertex pushed
    // down) and a PEAK (one vertex pushed up) — the pit rings its triangles
    // with CONCAVE edges (accept), the peak with CONVEX edges (reject).  The
    // tessellation flag alternates to exercise both diagonal orientations.
    std::vector<uint32_t> samples(HF_ROWS * HF_COLS, 0);
    for (uint32_t rIdx = 0; rIdx < HF_ROWS; ++rIdx)
        for (uint32_t c = 0; c < HF_COLS; ++c)
        {
            const uint32_t vi = rIdx * HF_COLS + c;
            int16_t h = 0;
            if (rIdx == PIT_R && c == PIT_C) h = -3;    // pit
            if (rIdx == PEAK_R && c == PEAK_C) h = 3;   // peak
            const uint32_t tess = ((rIdx + c) & 1u) ? 0x80u : 0u;
            samples[vi] = uint32_t(uint16_t(h)) | (tess << 16);
        }

    const uint32_t NCM = NC / 4;
    std::vector<U4> pairRec(NCM);
    for (uint32_t cm = 0; cm < NCM; ++cm)
        pairRec[cm] = { ((cm * 7 + 3) % NCM) * 4, ((cm * 11 + 5) % NCM) * 4, 0, 0 };
    const uint32_t poolN = NC + 16;

    // concavity of an edge (edge 0=e01,1=e12,2=e02) of a triangle, at scale
    // (1,1,1): sign of triNormalAdj . (oppositeVert - adjV0); <0 concave,
    // >0 convex, ==0 flat (matches the shader switch)
    auto concavity = [&](uint32_t tri, uint32_t edge) -> float {
        V3 v0, v1, v2; U4 adj;
        getTriangleHf(samples, tri, { 1, 1, 1 }, v0, v1, v2, adj);
        const uint32_t at = (edge == 0) ? adj.x : (edge == 1) ? adj.y : adj.z;
        if (at == BOUNDARY) return 0.0f;
        V3 a0, a1, a2; U4 d;
        getTriangleHf(samples, at, { 1, 1, 1 }, a0, a1, a2, d);
        const V3 nAdj = getNormalizedRef(cross3(sub(a1, a0), sub(a2, a0)));
        const V3 vk = (edge == 0) ? v2 : (edge == 1) ? v0 : v1;
        return dot3(nAdj, sub(vk, a0));
    };
    // identity-frame placement that pushes a sphere PAST an edge midpoint (out
    // of the triangle along mid-centroid, lifted along the normal) so the
    // closest feature is that EDGE — returns the placement point C
    auto edgePlace = [&](uint32_t tri, uint32_t e) -> V3 {
        V3 v0, v1, v2; U4 adj;
        getTriangleHf(samples, tri, { 1, 1, 1 }, v0, v1, v2, adj);
        const V3 tn = getNormalizedRef(cross3(sub(v1, v0), sub(v2, v0)));
        const V3 centroid = mul(add(add(v0, v1), v2), 1.0f / 3.0f);
        const V3 mid = (e == 0) ? mul(add(v0, v1), 0.5f)
                     : (e == 1) ? mul(add(v1, v2), 0.5f) : mul(add(v0, v2), 0.5f);
        const V3 outward = getNormalizedRef(sub(mid, centroid));
        return add(add(mid, mul(outward, 0.25f)), mul(tn, 0.25f));
    };
    // Bucket edge placements by the arm the placement ACTUALLY produces (the
    // outward push can land in a neighbouring edge's Voronoi region, so we
    // classify by the realised closest feature, not the intended edge) and by
    // the concavity SIGN of that realised edge: concaveArm/convexArm[arm] where
    // arm 0=eE01, 1=eE12, 2=eE02.  This guarantees the concavity switch is
    // exercised on all three arms with BOTH accept (concave) and reject (convex)
    // outcomes.  maskIdx>>29: 4=eE01,6=eE12,5=eE02; lambda edge 0=adj.x(eE01),
    // 1=adj.y(eE12), 2=adj.z(eE02).
    // flatEdgeArm[arm]: a COPLANAR interior edge (concavity dot EXACTLY 0), so
    // the switch's `dot < 0` rejects it — pinning the strict `<` (a `<=` mutant
    // would flip the flat edge from reject to accept).  Bucketed per arm so all
    // three switch arms get a flat-edge reject.
    std::vector<std::pair<uint32_t, V3>> concaveArm[3], convexArm[3], flatEdgeArm[3];
    for (uint32_t r = 0; r + 1 < HF_ROWS; ++r)
        for (uint32_t c = 0; c + 1 < HF_COLS; ++c)
            for (uint32_t sub = 0; sub < 2; ++sub)
            {
                const uint32_t tri = (r * HF_COLS + c) * 2u + sub;
                for (uint32_t e = 0; e < 3; ++e)
                {
                    V3 v0, v1, v2; U4 adj;
                    getTriangleHf(samples, tri, { 1, 1, 1 }, v0, v1, v2, adj);
                    const V3 C = edgePlace(tri, e);
                    V3 cp; bool g, f; uint32_t mk;
                    distPointTriRef(C, v0, v1, v2, adj, cp, g, f, mk);
                    const uint32_t mi = mk >> 29;
                    if (mi != 4 && mi != 5 && mi != 6) continue; // not an edge feature
                    const uint32_t arm = (mi == 4) ? 0 : (mi == 6) ? 1 : 2;
                    const uint32_t lamEdge = (mi == 4) ? 0 : (mi == 6) ? 1 : 2;
                    const uint32_t at = (lamEdge == 0) ? adj.x : (lamEdge == 1) ? adj.y : adj.z;
                    if (at == BOUNDARY) continue;
                    const float cc = concavity(tri, lamEdge);
                    if (cc < -0.05f) concaveArm[arm].push_back({ tri, C });
                    else if (cc > 0.05f) convexArm[arm].push_back({ tri, C });
                    else if (cc == 0.0f) flatEdgeArm[arm].push_back({ tri, C });
                }
            }
    for (uint32_t a = 0; a < 3; ++a)
        PXG_CHECK_MSG(!concaveArm[a].empty() && !convexArm[a].empty() && !flatEdgeArm[a].empty(),
                      "chSphere: switch arm %u coverage thin (concave %zu convex %zu flat %zu)",
                      a, concaveArm[a].size(), convexArm[a].size(), flatEdgeArm[a].size());

    const uint32_t pitVi = PIT_R * HF_COLS + PIT_C, peakVi = PEAK_R * HF_COLS + PEAK_C;

    // BOUNDARY-edge placements: an edge closest-feature whose adjacency IS
    // BOUNDARY, so the concavity switch takes its "keep the contact" branch
    // (the shader's `&& adj.x != BOUNDARY` guard — the DELIBERATE resolution of
    // the upstream unguarded getTriangle(0xFFFFFFFF) read).  Restricted to flat
    // grid-edge triangles for clean geometry; IDENTITY transform keeps the
    // irrational edge-contact patch normal bit-exact CPU vs GPU.
    std::vector<std::pair<uint32_t, V3>> boundaryEdge;
    for (uint32_t r = 0; r + 1 < HF_ROWS; ++r)
        for (uint32_t c = 0; c + 1 < HF_COLS; ++c)
            for (uint32_t sub = 0; sub < 2; ++sub)
            {
                const uint32_t tri = (r * HF_COLS + c) * 2u + sub;
                uint32_t t0, t1, t2; triVertIdx(samples, tri, t0, t1, t2);
                const bool flatTri = t0 != pitVi && t1 != pitVi && t2 != pitVi &&
                                     t0 != peakVi && t1 != peakVi && t2 != peakVi;
                if (!flatTri) continue;
                for (uint32_t e = 0; e < 3; ++e)
                {
                    V3 v0, v1, v2; U4 adj;
                    getTriangleHf(samples, tri, { 1, 1, 1 }, v0, v1, v2, adj);
                    const V3 C = edgePlace(tri, e);
                    V3 cp; bool g, f; uint32_t mk;
                    distPointTriRef(C, v0, v1, v2, adj, cp, g, f, mk);
                    const uint32_t mi = mk >> 29;
                    if (mi != 4 && mi != 5 && mi != 6) continue;
                    const uint32_t lamEdge = (mi == 4) ? 0 : (mi == 6) ? 1 : 2;
                    const uint32_t at = (lamEdge == 0) ? adj.x : (lamEdge == 1) ? adj.y : adj.z;
                    if (at == BOUNDARY) boundaryEdge.push_back({ tri, C });
                }
            }
    PXG_CHECK_MSG(!boundaryEdge.empty(), "chSphere: no boundary-edge placements found");

    // fully-flat cells: every one of the four 2x2 vertices is at height 0 (no
    // pit (2,2) / peak (5,5)), so the triangle normal is axis-aligned (0,+-1,0)
    // and stays exact under a signed-permutation rotation
    auto vertexIsFlat = [&](uint32_t vr, uint32_t vc) {
        return !((vr == PIT_R && vc == PIT_C) || (vr == PEAK_R && vc == PEAK_C));
    };
    std::vector<uint32_t> flatCells;
    for (uint32_t r = 0; r + 1 < HF_ROWS; ++r)
        for (uint32_t c = 0; c + 1 < HF_COLS; ++c)
            if (vertexIsFlat(r, c) && vertexIsFlat(r, c + 1) &&
                vertexIsFlat(r + 1, c) && vertexIsFlat(r + 1, c + 1))
                flatCells.push_back(r * HF_COLS + c);
    PXG_CHECK_MSG(flatCells.size() > 8, "chSphere: too few flat cells (%zu)", flatCells.size());
    auto flatCell = [&](std::mt19937& g) {
        return flatCells[g() % flatCells.size()];
    };

    // TILTED triangles: any triangle touching the pit or peak vertex has an
    // irrational (non-axis-aligned) normal and a general (non-power-of-two)
    // barycentric denominator, so a FACE contact on one exercises the
    // reciprocal-then-multiply of the face-region closest point (the divide-vs-
    // reciprocal difference is invisible on flat power-of-two triangles)
    std::vector<uint32_t> tiltedTris;
    for (uint32_t r = 0; r + 1 < HF_ROWS; ++r)
        for (uint32_t c = 0; c + 1 < HF_COLS; ++c)
            for (uint32_t sub = 0; sub < 2; ++sub)
            {
                const uint32_t tri = (r * HF_COLS + c) * 2u + sub;
                uint32_t t0, t1, t2; triVertIdx(samples, tri, t0, t1, t2);
                if (t0 == pitVi || t1 == pitVi || t2 == pitVi ||
                    t0 == peakVi || t1 == peakVi || t2 == peakVi)
                    tiltedTris.push_back(tri);
            }
    PXG_CHECK_MSG(tiltedTris.size() > 4, "chSphere: too few tilted tris (%zu)", tiltedTris.size());

    std::vector<ChCase> cases(NC);
    for (uint32_t i = 0; i < NC; ++i)
    {
        ChCase& cs = cases[i];
        std::memset(&cs, 0, sizeof(cs));
        const float cd = 0.25f;
        const uint32_t fam = i % 9;
        const bool flatEdgeFam = (fam == 2);
        const bool concavityFam = (fam == 5 || fam == 6);
        const bool tiltedFaceFam = (fam == 7);
        const bool boundaryFam = (fam == 8);
        // families whose geometry is an EDGE or a tilted face MUST use an
        // IDENTITY transform: the edge patch normal / tilted normal is
        // irrational and NI = -patchNormal has no rotation rounding, so it
        // stays bit-exact CPU vs GPU.  Flat FACE/VERTEX families use signed-perm
        // transforms (axis-aligned normals -> rotation is an exact permutation)
        // with +-scales for handedness.  (flatEdgeFam is rejected, so its
        // outputs are integer/clean, but its edge placement is bucketed at
        // scale 1 so it also takes the identity frame.)
        // Residual coverage note: the concavity switch's adjTriNormal/adjTriV0
        // take `scale`, but the switch families run at scale (1,1,1); a mutant
        // that substituted a DIFFERENT non-unit scale there is still caught (the
        // adjacent geometry changes), only one substituting another variable
        // that also equals (1,1,1) would survive — a scaled-switch fixture is
        // deferred because a tilted edge under a non-identity rotation loses
        // bit-exactness.
        const bool identityFam = flatEdgeFam || concavityFam || tiltedFaceFam || boundaryFam;

        const float sc[3] = { 0.5f, 1.0f, 2.0f };
        if (identityFam)
        {
            cs.mQ = { 0, 0, 0, 1 }; cs.aQ = { 0, 0, 0, 1 };
            cs.mP = { 0, 0, 0, 0 };
            cs.scale = { 1, 1, 1, 0.5f };
        }
        else
        {
            cs.mQ = exactQuat(rng);
            cs.aQ = exactQuat(rng);
            cs.mP = { float(int(rng() % 9) - 4), float(int(rng() % 9) - 4), float(int(rng() % 9) - 4), 0 };
            float rsx = sc[rng() % 3], csz = sc[rng() % 3];
            if (i % 4 == 1) rsx = -rsx;   // negative row scale -> handedness flip
            if (i % 4 == 3) csz = -csz;   // negative column scale -> handedness flip
            cs.scale = { rsx, sc[rng() % 3], csz, 0.5f };
        }

        // concavity families select a pre-bucketed edge placement (arm cycles
        // 0/1/2 so all three switch arms are hit; if an arm happens to be empty
        // for this grid the next non-empty arm is used, never a %0); the
        // boundary family a grid-edge placement; the tilted-face family a
        // pit/peak triangle; flat families a flat cell.  specialC carries the
        // pre-computed identity-frame placement for the edge/boundary families.
        uint32_t tri;
        V3 specialC{};
        if (concavityFam)
        {
            auto* arms = (fam == 5) ? concaveArm : convexArm;
            const uint32_t sel = i / 9;
            std::pair<uint32_t, V3> pick{ 0u, V3{ 0, 0, 0 } };
            for (uint32_t k = 0; k < 3; ++k)
            {
                auto& b = arms[(sel + k) % 3];
                if (!b.empty()) { pick = b[(sel / 3) % b.size()]; break; }
            }
            tri = pick.first;
            specialC = pick.second;
        }
        else if (flatEdgeFam)
        {
            auto& b = flatEdgeArm[(i / 9) % 3];
            const auto& pick = b[(i / 27) % b.size()];
            tri = pick.first;
            specialC = pick.second;
        }
        else if (boundaryFam)
        {
            const auto& pick = boundaryEdge[(i / 9) % boundaryEdge.size()];
            tri = pick.first;
            specialC = pick.second;
        }
        else if (tiltedFaceFam)
        {
            tri = tiltedTris[(i / 9) % tiltedTris.size()];
        }
        else
        {
            const uint32_t cell = flatCell(rng);
            tri = cell * 2u + (rng() % 2u);
        }
        cs.pair = { i / 4, tri, i % 4, (1u << 28) | ((i * 11 + 3) % 4096) };

        V3 v0, v1, v2; U4 adj;
        getTriangleHf(samples, tri, v3(cs.scale), v0, v1, v2, adj);
        const V3 tn = getNormalizedRef(cross3(sub(v1, v0), sub(v2, v0)));
        const V3 centroid = mul(add(add(v0, v1), v2), 1.0f / 3.0f);

        V3 C;
        switch (fam)
        {
        case 0: case 7: // face contact above the centroid along the normal
            C = add(centroid, mul(tn, 0.25f));
            break;
        case 1: // flat vertex deferral: beside v0 along (v0 - centroid) in-plane
            C = add(v0, mul(getNormalizedRef(sub(v0, centroid)), 0.375f));
            break;
        case 2: // flat INTERIOR (coplanar) edge -> concavity dot EXACTLY 0 ->
                // REJECTED by the switch's strict `< 0`; the per-arm flatEdgeArm
                // bucket cycles all three arms so `<` vs `<=` is pinned on each.
            C = specialC;
            break;
        case 3: // just BELOW the plane (dist<0) but WITHIN the inflated radius
                // (sqDist 0.25 < sqInflated 0.5625): the dist>=0 one-sided gate
                // is the ONLY thing rejecting it, so dropping that gate flips nb
            C = add(centroid, mul(tn, -0.5f));
            break;
        case 4: // too far above
            C = add(centroid, mul(tn, 4.0f));
            break;
        case 5: case 6: // concavity: pre-bucketed edge placement (sphere pushed
                        // PAST an edge so the closest feature is genuinely that
                        // EDGE) exercising the concavity switch on all three
                        // arms; fam5 concave->accept, fam6 convex->reject.  The
                        // IDENTITY transform keeps the irrational edge geometry
                        // bit-exact CPU vs GPU.
        case 8:         // boundary edge: closest feature is a grid-edge whose
                        // adjacency is BOUNDARY -> the switch's keep-contact
                        // guard fires (accepted, delayed edge contact).
            C = specialC;
            break;
        default: C = add(centroid, mul(tn, 0.25f)); break;
        }

        const V3 world = add(quatRotate(cs.mQ, C), v3(cs.mP));
        cs.aP = { world.x, world.y, world.z, cd };
    }

    std::vector<ChRef> ref(NC);
    for (uint32_t i = 0; i < NC; ++i)
        ref[i] = refSphere(cases[i], samples, pairRec);

    // batch aggregation (ceil: the last warp is partial when NC % 32 != 0)
    const uint32_t NB = (NC + 31) / 32;
    std::vector<uint32_t> batchContacts(NB, 0), batchMasks(NB, 0);
    for (uint32_t i = 0; i < NC; ++i)
    {
        batchContacts[i / 32] += ref[i].nb;
        if (ref[i].om != 0) batchMasks[i / 32]++;
    }
    uint32_t totalSP = 0, totalTC = 0;
    for (uint32_t b = 0; b < NB; ++b)
        if (batchContacts[b] != 0) { totalSP += batchMasks[b]; totalTC += batchContacts[b]; }

    size_t nHit = 0, nDelay = 0, nZero = 0, nHand = 0, nFace = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        if (ref[i].nb) ++nHit; else ++nZero;
        if (ref[i].delay) ++nDelay;
        if (ref[i].nb && !ref[i].delay) ++nFace;
        if ((i % 4 == 1 || i % 4 == 3)) ++nHand;
    }
    PXG_CHECK_MSG(nHit > 200 && nDelay > 40 && nZero > 100 && nFace > 100 && nHand > 300,
                  "chSphere coverage thin (hit=%zu delay=%zu zero=%zu face=%zu hand=%zu)",
                  nHit, nDelay, nZero, nFace, nHand);

    const uint32_t spCap = totalSP + 32, tcCap = totalTC + 32;
    const ChRun r = runCh(cases, samples, pairRec, poolN, spCap, tcCap);

    PXG_CHECK_MSG(r.counters[0] == totalSP && r.counters[1] == totalTC,
                  "chSphere counters %u/%u (want %u/%u)", r.counters[0], r.counters[1], totalSP, totalTC);

    std::vector<bool> usedPo(poolN, false), usedPad(poolN, false);
    size_t mm = 0;
    std::vector<std::pair<uint32_t, uint32_t>> ranges;
    for (uint32_t i = 0; i < NC; ++i)
    {
        const ChRef& ro = ref[i];
        usedPo[ro.pairOffset] = true;
        usedPad[ro.padOffset] = true;
        const F4 ni = r.ni[ro.pairOffset];
        if (ni.x != ro.niNormal.x || ni.y != ro.niNormal.y || ni.z != ro.niNormal.z ||
            asU32(ni.w) != ro.niIndex) ++mm;
        if (r.depth[ro.pairOffset] != ro.sep) ++mm;
        if (r.interm[ro.pairOffset] != ro.gpuTri) ++mm;
        if (r.ordered[ro.padOffset] != ro.ordered) ++mm;
        if (r.dbg[i].x != float(ro.nb) || r.dbg[i].y != ro.sep ||
            r.dbg[i].z != (ro.delay ? 1.0f : 0.0f) || r.dbg[i].w != float(ro.gpuTri)) ++mm;
        if (ro.nb)
        {
            const uint32_t idx = r.ctsIdx[ro.pairOffset];
            if (idx == POISON || idx + ro.nb > totalTC) { ++mm; continue; }
            const F4 got = r.temp[idx];
            if (got.x != 0.0f || got.y != 0.0f || got.z != 0.0f || got.w != ro.sep) ++mm;
            ranges.push_back({ idx, ro.nb });
        }
        else if (r.ctsIdx[ro.pairOffset] != POISON) ++mm;
    }
    for (uint32_t s = 0; s < poolN; ++s)
    {
        if (!usedPo[s] && (asU32(r.ni[s].x) != asU32(-7.0f) || r.depth[s] != -7.0f ||
                           r.interm[s] != POISON || r.ctsIdx[s] != POISON)) ++mm;
        if (!usedPad[s] && r.ordered[s] != POISON) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "chSphere: %zu mismatches", mm);

    std::sort(ranges.begin(), ranges.end());
    {
        uint32_t cursor = 0; size_t bad = 0;
        for (const auto& pr : ranges) { if (pr.first != cursor) ++bad; cursor = pr.first + pr.second; }
        if (cursor != totalTC) ++bad;
        PXG_CHECK_MSG(bad == 0, "chSphere: temp ranges do not partition [0, %u)", totalTC);
    }
    for (uint32_t s = totalTC; s < tcCap; ++s)
        PXG_CHECK_MSG(r.temp[s].x == -777.0f, "chSphere: dirty temp slot %u", s);
    for (uint32_t s = totalSP; s < spCap; ++s)
        PXG_CHECK_MSG(r.second[s] == POISON, "chSphere: dirty second-pass slot %u", s);

    // the appended second-pass outMasks (gti | mask) must match the expected
    // deferred set as a MULTISET — the per-warp InterlockedAdd makes the
    // ordering non-deterministic, so compare sorted.  (Every deferred contact
    // has nb==1 -> its warp always has total>0 -> never batch-dropped.)
    {
        std::vector<uint32_t> wantOm, gotOm;
        for (uint32_t i = 0; i < NC; ++i)
            if (ref[i].om != 0) wantOm.push_back(ref[i].om);
        for (uint32_t s = 0; s < totalSP; ++s) gotOm.push_back(r.second[s]);
        std::sort(wantOm.begin(), wantOm.end());
        std::sort(gotOm.begin(), gotOm.end());
        PXG_CHECK_MSG(wantOm == gotOm, "chSphere: second-pass outMask multiset mismatch (%zu vs %zu)",
                      wantOm.size(), gotOm.size());
    }
}

} // namespace
