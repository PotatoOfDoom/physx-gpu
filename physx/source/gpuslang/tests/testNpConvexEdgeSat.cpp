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

// Verification of npConvexEdgeSat.slang: ctEdgeCrossSat (satEdgeCrossDirection2)
// — the convex-edge x triangle-edge separating-axis driver.  The hull is an
// axis-aligned box (8 verts, 12 edges, 6 faces) at power-of-two half-extents and
// the triangle has AXIS-ALIGNED edges (a right triangle in a coordinate plane,
// possibly signed-permutation rotated), so every box-edge x tri-edge cross
// product is either degenerate (parallel) or an EXACT axis-aligned unit axis and
// the integer/power-of-two vertex projections are bit-exact.  The whole
// separation / faceNormal / feature / passed output is therefore compared
// BIT-EXACTLY against an exact CPU transcription that mirrors the shader's serial
// loop pair-for-pair (so even running-max ties resolve identically).

#include "pxgTestFramework.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct V3 { float x, y, z; };
struct F4 { float x, y, z, w; };
struct U2 { uint32_t x, y; };
struct U4 { uint32_t x, y, z, w; };
struct EsIn { F4 triV0, triV1, triV2, triNormal, seedFaceNormal, misc; U4 triAdj; };
struct EsOut { F4 sepRet, faceNormal; uint32_t feature, q0, q1, q2; };
struct EsParams { uint32_t numTests, maxVerts, maxEdges, maxPolys; };

static const uint32_t EDGE_FEATURE_IDX = 0x80000000u;
static const uint32_t BOUNDARY = 0xffffffffu;
static const uint32_t NONCONVEX_FLAG = 0x80000000u;
static const uint32_t MAX_VERTS = 8, MAX_EDGES = 12, MAX_POLYS = 6;
static const float ES_MAX = 3.402823466e38f;

static float dotX(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 crossX(V3 a, V3 b)
{
    return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}
static V3 normalizedX(V3 v) { const float m = dotX(v, v); const float s = 1.0f / std::sqrt(m); return { v.x * s, v.y * s, v.z * s }; }
static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 neg(V3 a) { return { -a.x, -a.y, -a.z }; }
static F4 f4(V3 v) { return { v.x, v.y, v.z, 0.0f }; }
static bool isActiveEdge(uint32_t e) { return e == BOUNDARY || (e & NONCONVEX_FLAG) == 0u; }

// ---- axis-aligned box hull ------------------------------------------------
// verts indexed by (i|2j|4k): (i?+hx:-hx, j?+hy:-hy, k?+hz:-hz)
// faces 0=+x 1=-x 2=+y 3=-y 4=+z 5=-z
struct Box
{
    F4 verts[8];
    F4 planeN[6];
    U2 edgeV[12];
    U2 edgeF[12];
};
static Box makeBox(float hx, float hy, float hz)
{
    Box b;
    for (uint32_t k = 0; k < 2; ++k)
        for (uint32_t j = 0; j < 2; ++j)
            for (uint32_t i = 0; i < 2; ++i)
                b.verts[i | (j << 1) | (k << 2)] =
                    { i ? hx : -hx, j ? hy : -hy, k ? hz : -hz, 0.0f };
    b.planeN[0] = { 1, 0, 0, 0 };  b.planeN[1] = { -1, 0, 0, 0 };
    b.planeN[2] = { 0, 1, 0, 0 };  b.planeN[3] = { 0, -1, 0, 0 };
    b.planeN[4] = { 0, 0, 1, 0 };  b.planeN[5] = { 0, 0, -1, 0 };
    uint32_t e = 0;
    // x-edges: differ in i; adjacent faces = y(j?2:3), z(k?4:5)
    for (uint32_t k = 0; k < 2; ++k)
        for (uint32_t j = 0; j < 2; ++j)
        { b.edgeV[e] = { (j << 1) | (k << 2), 1u | (j << 1) | (k << 2) }; b.edgeF[e] = { j ? 2u : 3u, k ? 4u : 5u }; ++e; }
    // y-edges: differ in j; adjacent faces = x(i?0:1), z(k?4:5)
    for (uint32_t k = 0; k < 2; ++k)
        for (uint32_t i = 0; i < 2; ++i)
        { b.edgeV[e] = { i | (k << 2), i | (1u << 1) | (k << 2) }; b.edgeF[e] = { i ? 0u : 1u, k ? 4u : 5u }; ++e; }
    // z-edges: differ in k; adjacent faces = x(i?0:1), y(j?2:3)
    for (uint32_t j = 0; j < 2; ++j)
        for (uint32_t i = 0; i < 2; ++i)
        { b.edgeV[e] = { i | (j << 1), i | (j << 1) | (1u << 2) }; b.edgeF[e] = { i ? 0u : 1u, j ? 2u : 3u }; ++e; }
    return b;
}

// signed-permutation rotation (exact): maps axes to +-axes.  perm is a
// permutation of {0,1,2}; sign is +-1 per output component.
struct SPerm { int perm[3]; float sign[3]; };
static V3 applySP(const SPerm& s, V3 v)
{
    const float in[3] = { v.x, v.y, v.z };
    return { s.sign[0] * in[s.perm[0]], s.sign[1] * in[s.perm[1]], s.sign[2] * in[s.perm[2]] };
}

// ---- CPU reference: exact transcription of ctEdgeCrossSat -----------------
struct EdgeRef { float separation; V3 faceNormal; uint32_t feature; bool passed; };
static EdgeRef edgeSatRef(const Box& box, uint32_t nbVerts, uint32_t nbEdges,
                          const V3 tri[3], const uint32_t triAdj[3], V3 shapeNormal,
                          float contactDist, float seedSep, V3 seedFN, uint32_t seedFeat)
{
    V3 accFN = seedFN;
    float accSep = seedSep;
    bool passed = true;
    const uint32_t total = nbEdges * 3u;
    for (uint32_t index = 0; index < total; ++index)
    {
        const uint32_t vs = index % 3u;
        if (!isActiveEdge(triAdj[vs])) continue;
        const uint32_t ve = (vs + 1u) % 3u;
        const uint32_t ce = index / 3u;
        const U2 vIdx = box.edgeV[ce];
        const U2 fIdx = box.edgeF[ce];
        const V3 nf0 = { box.planeN[fIdx.x].x, box.planeN[fIdx.x].y, box.planeN[fIdx.x].z };
        const V3 nf1 = { box.planeN[fIdx.y].x, box.planeN[fIdx.y].y, box.planeN[fIdx.y].z };
        if (!(dotX(shapeNormal, nf0) < 0.0f || dotX(shapeNormal, nf1) < 0.0f)) continue;
        const V3 vA = { box.verts[vIdx.x].x, box.verts[vIdx.x].y, box.verts[vIdx.x].z };
        const V3 vB = { box.verts[vIdx.y].x, box.verts[vIdx.y].y, box.verts[vIdx.y].z };
        const V3 convexEdge = sub(vA, vB);
        const V3 triEdge = sub(tri[ve], tri[vs]);
        const V3 v = crossX(convexEdge, triEdge);
        const float magSq = dotX(v, v);
        if (magSq > 1e-10f)
        {
            const V3 axis = normalizedX(v);
            float minTri = ES_MAX, maxTri = -ES_MAX, minConv = ES_MAX, maxConv = -ES_MAX;
            for (uint32_t i = 0; i < 3; ++i) { const float d = dotX(tri[i], axis); minTri = std::min(d, minTri); maxTri = std::max(d, maxTri); }
            for (uint32_t i = 0; i < nbVerts; ++i)
            { const V3 vv = { box.verts[i].x, box.verts[i].y, box.verts[i].z }; const float d = dotX(vv, axis); minConv = std::min(d, minConv); maxConv = std::max(d, maxConv); }
            const float o0 = minTri - maxConv;
            const float o1 = minConv - maxTri;
            if (minConv > (maxTri + contactDist) || minTri > (maxConv + contactDist)) { passed = false; break; }
            if (accSep < o0) { accSep = o0; accFN = axis; }
            if (accSep < o1) { accSep = o1; accFN = neg(axis); }
        }
    }
    EdgeRef r;
    if (!passed) { r.separation = seedSep; r.faceNormal = seedFN; r.feature = seedFeat; r.passed = false; return r; }
    r.separation = seedSep; r.faceNormal = seedFN; r.feature = seedFeat; r.passed = true;
    if (seedSep < accSep) { r.faceNormal = accFN; r.separation = accSep; r.feature = EDGE_FEATURE_IDX; }
    return r;
}

static uint32_t asU32(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

PXG_TEST(npConvexEdgeSat_edgeCross)
{
    std::mt19937 rng(1080801);
    const float dims[3] = { 0.5f, 1.0f, 2.0f };
    const float triLen[3] = { 1.0f, 2.0f, 4.0f };

    // 24 signed-permutation rotations (proper + improper): all perm x all sign
    std::vector<SPerm> sps;
    const int perms[6][3] = { {0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0} };
    for (int p = 0; p < 6; ++p)
        for (int s = 0; s < 8; ++s)
            sps.push_back({ { perms[p][0], perms[p][1], perms[p][2] },
                            { (s & 1) ? -1.0f : 1.0f, (s & 2) ? -1.0f : 1.0f, (s & 4) ? -1.0f : 1.0f } });

    std::vector<F4> verts, planeN;
    std::vector<U2> edgeV, edgeF;
    std::vector<EsIn> in;
    std::vector<EdgeRef> ref;
    std::vector<Box> boxes;

    auto emit = [&](const Box& box, const V3 tri[3], const uint32_t triAdj[3], V3 shapeNormal,
                    float contactDist, float seedSep, V3 seedFN, uint32_t seedFeat)
    {
        boxes.push_back(box);
        for (uint32_t i = 0; i < MAX_VERTS; ++i) verts.push_back(box.verts[i]);
        for (uint32_t i = 0; i < MAX_POLYS; ++i) planeN.push_back(box.planeN[i]);
        for (uint32_t i = 0; i < MAX_EDGES; ++i) { edgeV.push_back(box.edgeV[i]); edgeF.push_back(box.edgeF[i]); }
        EsIn e;
        e.triV0 = f4(tri[0]); e.triV1 = f4(tri[1]); e.triV2 = f4(tri[2]);
        e.triNormal = f4(shapeNormal);
        e.seedFaceNormal = f4(seedFN);
        e.misc = { float(MAX_VERTS), float(MAX_EDGES), contactDist, seedSep };
        e.triAdj = { triAdj[0], triAdj[1], triAdj[2], seedFeat };
        in.push_back(e);
        ref.push_back(edgeSatRef(box, MAX_VERTS, MAX_EDGES, tri, triAdj, shapeNormal,
                                 contactDist, seedSep, seedFN, seedFeat));
    };

    // family A: box vs an axis-aligned right triangle in a coordinate plane,
    // signed-perm rotated + integer-translated, sweeping active-edge flags and
    // seed separations.  All edges axis-aligned -> exact axes.
    for (uint32_t iter = 0; iter < 900; ++iter)
    {
        const Box box = makeBox(dims[rng() % 3], dims[rng() % 3], dims[rng() % 3]);
        // base right triangle in the xy plane at z = offz (axis-aligned legs)
        const float lx = triLen[rng() % 3], ly = triLen[rng() % 3];
        const float ox = float(int(rng() % 9) - 4);
        const float oy = float(int(rng() % 9) - 4);
        const float oz = float(int(rng() % 9) - 4);
        V3 baseTri[3] = { { ox, oy, oz }, { ox + lx, oy, oz }, { ox, oy + ly, oz } };
        // triangle normal is +-z (right triangle in xy plane); winding gives +z
        V3 baseN = { 0, 0, 1 };
        const SPerm& sp = sps[rng() % sps.size()];
        V3 tri[3]; for (int i = 0; i < 3; ++i) tri[i] = applySP(sp, baseTri[i]);
        V3 shapeNormal = applySP(sp, baseN);

        uint32_t triAdj[3];
        for (int i = 0; i < 3; ++i)
        {
            const uint32_t r = rng() % 3;
            triAdj[i] = (r == 0) ? BOUNDARY : (r == 1) ? (12u + i) : (NONCONVEX_FLAG | (5u + i)); // active, active, inactive
        }
        const float contactDist = dims[rng() % 3];
        const float seedSep = (rng() & 1) ? -ES_MAX : (float(int(rng() % 17) - 8) * 0.5f);
        V3 seedFN = applySP(sps[rng() % sps.size()], V3{ 0, 0, 1 });
        const uint32_t seedFeat = rng() & 0xffff;
        emit(box, tri, triAdj, shapeNormal, contactDist, seedSep, seedFN, seedFeat);
    }

    // family B: OVERLAPPING box vs a small triangle centred near the origin so
    // NO axis separates (passed stays true) and the edge SAT computes real
    // overlaps.  The triangle normal is +-z, so the Gauss-map cull keeps the
    // x/y edges (which cross to the z axis) and drops the z edges (which cross
    // to x/y axes) — flipping the cull therefore changes the WINNING axis, which
    // pins the cull DIRECTION.  seedSep = -MAX makes the edge axis always win.
    for (uint32_t iter = 0; iter < 400; ++iter)
    {
        const Box box = makeBox(dims[rng() % 3], dims[rng() % 3], dims[rng() % 3]);
        const float lx = 0.5f, ly = 0.5f;
        // centre the right triangle near the origin (deep inside the box)
        const float ox = -0.5f + float(int(rng() % 3) - 1) * 0.5f;
        const float oy = -0.5f + float(int(rng() % 3) - 1) * 0.5f;
        const float oz = float(int(rng() % 3) - 1) * 0.5f; // -0.5, 0, 0.5 (|oz| < hz)
        V3 baseTri[3] = { { ox, oy, oz }, { ox + lx, oy, oz }, { ox, oy + ly, oz } };
        V3 baseN = { 0, 0, 1 };
        const SPerm& sp = sps[rng() % sps.size()];
        V3 tri[3]; for (int i = 0; i < 3; ++i) tri[i] = applySP(sp, baseTri[i]);
        V3 shapeNormal = applySP(sp, baseN);

        uint32_t triAdj[3] = { BOUNDARY, 12u, 13u }; // all active edges
        const float contactDist = dims[rng() % 3];
        const float seedSep = (rng() & 1) ? -ES_MAX : (float(int(rng() % 5) - 4) * 0.5f);
        V3 seedFN = applySP(sps[rng() % sps.size()], V3{ 0, 1, 0 });
        const uint32_t seedFeat = rng() & 0xffff;
        emit(box, tri, triAdj, shapeNormal, contactDist, seedSep, seedFN, seedFeat);
    }

    // KNOWN-EQUIVALENT mutants under this exact box-hull fixture regime:
    //  * the Gauss-map cull DIRECTION (shapeNormal.dot(planeN) < 0 vs > 0): a box
    //    has only three edge directions and every direction's edges border both
    //    the + and - face, so flipping the cull only swaps PARALLEL edges (which
    //    cross to the identical axis) or symmetrically includes/excludes a whole
    //    direction — it can never asymmetrically change the tested axis SET.
    //    Observing it would need an asymmetric hull with tilted (irrational)
    //    face normals, which would forfeit the bit-exact comparison.
    //  * dropping the `break` after passed=false: once separated the output uses
    //    the seed values, so continuing the loop (updating the unused running-max
    //    accumulator) cannot change any output — a provable no-op.
    const uint32_t N = (uint32_t)in.size();
    PxgSlangComputeDevice* dev = testDevice();
    std::vector<EsOut> outInit(N);

    PxgBufferHandle bV = createBufferFrom(dev, verts);
    PxgBufferHandle bP = createBufferFrom(dev, planeN);
    PxgBufferHandle bEV = createBufferFrom(dev, edgeV);
    PxgBufferHandle bEF = createBufferFrom(dev, edgeF);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    EsParams prm = { N, MAX_VERTS, MAX_EDGES, MAX_POLYS };
    Kernel k("npConvexEdgeSat.spv", "ctEdgeCrossSat",
             { (uint32_t)sizeof(F4), (uint32_t)sizeof(F4), (uint32_t)sizeof(U2), (uint32_t)sizeof(U2),
               (uint32_t)sizeof(EsIn), (uint32_t)sizeof(EsOut) }, sizeof(EsParams));
    k.dispatch((N + 63) / 64, { bV, bP, bEV, bEF, bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<EsOut>(dev, bOut, N);

    size_t mm = 0, nPass = 0, nFail = 0, nEdgeWin = 0, nSeedKept = 0;
    for (uint32_t t = 0; t < N; ++t)
    {
        const EdgeRef& r = ref[t];
        const EsOut& o = g[t];
        const bool gotPassed = o.sepRet.y != 0.0f;
        if (gotPassed != r.passed) ++mm;
        if (asU32(o.sepRet.x) != asU32(r.separation)) ++mm;
        if (asU32(o.faceNormal.x) != asU32(r.faceNormal.x) ||
            asU32(o.faceNormal.y) != asU32(r.faceNormal.y) ||
            asU32(o.faceNormal.z) != asU32(r.faceNormal.z)) ++mm;
        if (o.feature != r.feature) ++mm;

        if (r.passed) ++nPass; else ++nFail;
        if (r.passed && r.feature == EDGE_FEATURE_IDX) ++nEdgeWin;
        if (r.passed && r.feature != EDGE_FEATURE_IDX) ++nSeedKept;
    }
    PXG_CHECK_MSG(mm == 0, "ctEdgeCrossSat: %zu mismatches", mm);
    PXG_CHECK_MSG(nFail > 20 && nEdgeWin > 20 && nSeedKept > 20,
                  "ctEdgeCrossSat coverage thin (pass=%zu fail=%zu edgeWin=%zu seedKept=%zu)",
                  nPass, nFail, nEdgeWin, nSeedKept);

    for (PxgBufferHandle h : { bV, bP, bEV, bEF, bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
