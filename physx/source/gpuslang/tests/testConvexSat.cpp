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

// P8.40 — the first two convex-vs-triangle SAT stages (pxgSlangConvexSat).
//
// The oracle is a scalar 32-lane simulation, the same shape as P8.38's: it shares
// the maths with the shader but not the warp machinery, so a wrong argmax tie-break
// or a missed break shows up as a disagreement rather than cancelling out.
//
// Suite (ctest filter "convexSat"):
//   convexSatStages — separated and overlapping boxes, a scaled hull, the tie-break
//                     between two equally-separating faces, and a degenerate plane.
//   convexSatSupportFold — P9.89: the stage-3 support folds read out per lane.

#include "pxgTestFramework.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace
{
using namespace pxgtest;

const uint32_t LANES = 32u;
const uint32_t MAXP = 64u;
const float MAXF = 3.4028234663852886e+38f;
const uint32_t TRI_FEATURE_IDX = 0x7FFFFFFFu;
const uint32_t EDGE_FEATURE_IDX = 0x80000000u;
const uint32_t BOUNDARY = 0xffffffffu;
const uint32_t NONCONVEX_FLAG = 0x80000000u;

struct V3 { float x, y, z; };
V3 mk(float x, float y, float z) { return V3{ x, y, z }; }
V3 sub(V3 a, V3 b) { return mk(a.x - b.x, a.y - b.y, a.z - b.z); }
V3 add(V3 a, V3 b) { return mk(a.x + b.x, a.y + b.y, a.z + b.z); }
float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
V3 cross3(V3 a, V3 b)
{ return mk(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }

// npAbiQuatRotate / RotateInv, transcribed operand-for-operand from
// pxgSlangNpAbi.slang:157 so the float order matches.
V3 quatRotate(const float q[4], V3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q[3] * q[3] - 0.5f;
    const float d2 = q[0] * vx + q[1] * vy + q[2] * vz;
    return mk(vx * w2 + (q[1] * vz - q[2] * vy) * q[3] + q[0] * d2,
              vy * w2 + (q[2] * vx - q[0] * vz) * q[3] + q[1] * d2,
              vz * w2 + (q[0] * vy - q[1] * vx) * q[3] + q[2] * d2);
}
V3 quatRotateInv(const float q[4], V3 v)
{
    const float qi[4] = { -q[0], -q[1], -q[2], q[3] };
    return quatRotate(qi, v);
}
V3 vertex2Shape(V3 v, V3 s, const float q[4])
{
    const V3 r = quatRotate(q, v);
    return quatRotateInv(q, mk(r.x * s.x, r.y * s.y, r.z * s.z));
}
V3 shape2Vertex(V3 v, V3 s, const float q[4])
{
    const V3 r = quatRotate(q, v);
    return quatRotateInv(q, mk(r.x / s.x, r.y / s.y, r.z / s.z));
}
// PxVec3::normalize(): returns the magnitude, divides in place only when positive.
// P9.36: PxVec3::normalize scales through operator/= (PxVec3.h:260) -- ONE reciprocal
// then three multiplies.  convexTriangle.cuh:246 reaches it via
// `shapeSpacePlaneNormal.normalize()`.
float normalizeInPlace(V3& v)
{
    const float m = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (m > 0.0f) { const float r = 1.0f / m; v.x *= r; v.y *= r; v.z *= r; }
    return m;
}

// ---- P9.89: the stage-3 support folds -------------------------------------
// PxMin/PxMax<float> are intrinsics::selectMin/selectMax (PxMath.h:82-100,
// PxUnixMathIntrinsics.h:116-125): `a < b ? a : b` and `a > b ? a : b`.  On an
// equal compare they return the SECOND operand, which in supportTriangle and
// supportConvex (`PxMin(d, minVal)`) is the running value: a +-0 tie keeps the
// FIRST-encountered zero, and a NaN projection leaves the running value alone.
float selMin(float a, float b) { return a < b ? a : b; }
float selMax(float a, float b) { return a > b ? a : b; }

// llvmpipe's GLSL FMin/FMax in a SignedZeroInfNanPreserve-stamped shader (P9.86c,
// pinned by p86SignedZero): a NaN operand is dropped, and a +-0 tie gives -0 for
// min and +0 for max in EITHER operand order.  That zero rule is also NVIDIA's
// fminf/fmaxf, i.e. warpReduction<MaxOpFloat>'s combine, so the warp arg-max below
// uses lpFMax rather than std::fmax, which leaves a zero tie to the C library.
float lpFMin(float a, float b)
{
    if (std::isnan(a)) return b;
    if (std::isnan(b)) return a;
    return (a == b) ? (std::signbit(a) ? a : b) : (a < b ? a : b);
}
float lpFMax(float a, float b)
{
    if (std::isnan(a)) return b;
    if (std::isnan(b)) return a;
    return (a == b) ? (std::signbit(a) ? b : a) : (a > b ? a : b);
}

// gSupVariant replays lines of the PRE-P9.89 port, which the fixtures below must
// tell apart from the CUDA.  A bit mask: 1 = csSupportTriangle's min as the
// unseeded min(d0, min(d1, d2)), 2 = its max likewise, 4 = csSupportConvex's min
// as min(d, minVal), 8 = its max as max(d, maxVal).  0 = the CUDA.
int gSupVariant = 0;

// supportTriangle, convexTriangle.cuh:507-518: seeded +-PX_MAX_F32, i = 0, 1, 2.
void refSupportTriangle(const V3 t[3], V3 dir, float& minVal, float& maxVal)
{
    minVal = MAXF;
    maxVal = -MAXF;
    float d[3];
    for (uint32_t i = 0; i < 3; ++i)
    {
        d[i] = dot3(t[i], dir);
        minVal = selMin(d[i], minVal);
        maxVal = selMax(d[i], maxVal);
    }
    if (gSupVariant & 1) minVal = lpFMin(d[0], lpFMin(d[1], d[2]));
    if (gSupVariant & 2) maxVal = lpFMax(d[0], lpFMax(d[1], d[2]));
}

// supportConvex, convexTriangle.cuh:494-505: seeded +-PX_MAX_F32, i = 0..n-1.
void refSupportConvex(const std::vector<V3>& verts, uint32_t n, V3 dir, float& minVal, float& maxVal)
{
    minVal = MAXF;
    maxVal = -MAXF;
    for (uint32_t i = 0; i < n; ++i)
    {
        const float d = dot3(verts[i], dir);
        minVal = (gSupVariant & 4) ? lpFMin(d, minVal) : selMin(d, minVal);
        maxVal = (gSupVariant & 8) ? lpFMax(d, maxVal) : selMax(d, maxVal);
    }
}

struct SatCase
{
    std::vector<uint8_t> hull;
    uint32_t nbVerts = 0, nbPolygons = 0, nbEdges = 0;
    std::vector<V3> verts;      // vertex-space hull vertices
    std::vector<float> planes;  // 4 per polygon
    std::vector<uint32_t> polyDescs;
    std::vector<uint32_t> vbe;   // verticesByEdges16, 2 per edge
    std::vector<uint32_t> fbe;   // facesByEdges8, 2 per edge
    V3 scale{}; float rot[4]{};
    float contactDist = 0.0f;
    V3 t[3]{}; V3 triN{};
    uint32_t probeIndex = 0;
    uint32_t adj[3] = { 0xffffffffu, 0xffffffffu, 0xffffffffu };   // BOUNDARY = active
    const char* name = "";
};

uint32_t packHullDesc(uint32_t e, uint32_t v, uint32_t p) { return (e << 16) | (v << 8) | p; }
uint32_t packPolyDesc(uint32_t vRef8, uint32_t nbV, uint32_t minIdx)
{ return (vRef8 << 16) | (nbV << 8) | minIdx; }

void buildBlob(SatCase& c)
{
    std::vector<uint8_t>& b = c.hull;
    auto u32 = [&](uint32_t v) { for (int i = 0; i < 4; ++i) b.push_back(uint8_t(v >> (8 * i))); };
    auto f32 = [&](float v) { uint32_t u; std::memcpy(&u, &v, 4); u32(u); };

    f32(0.0f); f32(0.0f); f32(0.0f); f32(0.0f);            // centre of mass
    u32(packHullDesc(c.nbEdges, c.nbVerts, c.nbPolygons));
    u32(0); u32(0); u32(0);
    f32(0.0f); f32(0.0f); f32(0.0f); f32(0.0f);            // unread float4

    for (uint32_t i = 0; i < c.nbVerts; ++i)
    { f32(c.verts[i].x); f32(c.verts[i].y); f32(c.verts[i].z); f32(0.0f); }
    for (uint32_t i = 0; i < 4u * c.nbPolygons; ++i) f32(c.planes[i]);
    for (uint32_t i = 0; i < c.nbPolygons; ++i) u32(c.polyDescs[i]);
    // verticesByEdges16: two ushorts per edge
    for (uint32_t e = 0; e < c.nbEdges; ++e)
    { const uint32_t w = (c.vbe[2*e+1] << 16) | c.vbe[2*e+0]; u32(w); }
    // facesByEdges8: two bytes per edge
    for (uint32_t e = 0; e < c.nbEdges; ++e)
    { b.push_back(uint8_t(c.fbe[2*e+0])); b.push_back(uint8_t(c.fbe[2*e+1])); }
    for (uint32_t i = 0; i < 3u * c.nbVerts + 64u; ++i) b.push_back(0);
}

struct SatResult
{
    bool p1 = false, p2 = false, p3 = false;
    uint32_t feature = 0xffffffffu;
    V3 faceNormal{};
    float separation = -MAXF;
    V3 planeN{}; float planeD = 0.0f;
    // the whole shape-space plane-normal array stage 1 left in groupshared, which
    // selectPolygonFeatureIndex reads back -- not just the probed entry
    std::vector<V3> planeAll;
    V3 scaledVert{};
};

SatResult refSat(const SatCase& c)
{
    SatResult r;

    // prologue
    std::vector<V3> sv(c.nbVerts);
    for (uint32_t i = 0; i < c.nbVerts; ++i) sv[i] = vertex2Shape(c.verts[i], c.scale, c.rot);

    // ---- stage 1: satConvexFaceNormals -------------------------------------
    float lSep[LANES]; uint32_t lFeat[LANES]; V3 lNorm[LANES]; bool lDisjoint[LANES];
    std::vector<V3> planeN(MAXP); std::vector<float> planeD(MAXP);
    for (uint32_t l = 0; l < LANES; ++l)
    { lSep[l] = -MAXF; lFeat[l] = 0; lNorm[l] = mk(0.0f, 0.0f, 0.0f); lDisjoint[l] = false; }

    for (uint32_t l = 0; l < LANES; ++l)
        for (uint32_t i = l; i < c.nbPolygons; i += LANES)
        {
            const uint32_t polygonData = c.polyDescs[i];
            const V3 minVertConv = sv[polygonData & 0xffu];
            const float planeDist = c.planes[4 * i + 3];
            V3 n = shape2Vertex(mk(c.planes[4 * i + 0], c.planes[4 * i + 1], c.planes[4 * i + 2]),
                                c.scale, c.rot);
            const float invMagnitude = 1.0f / normalizeInPlace(n);
            const float min0 = dot3(n, minVertConv);
            const float max0 = -planeDist * invMagnitude;
            const float d0 = dot3(c.t[0], n), d1 = dot3(c.t[1], n), d2 = dot3(c.t[2], n);
            const float max1 = std::fmax(d0, std::fmax(d1, d2));
            const float min1 = std::fmin(d0, std::fmin(d1, d2));

            if ((min1 > max0 + c.contactDist) || (min0 > max1 + c.contactDist))
            { lDisjoint[l] = true; break; }

            planeN[i] = n;
            planeD[i] = -max0;

            const float tempSeparation = min1 - max0;
            if (lSep[l] < tempSeparation) { lSep[l] = tempSeparation; lFeat[l] = i; lNorm[l] = n; }
        }

    bool anyDisjoint = false;
    for (uint32_t l = 0; l < LANES; ++l) anyDisjoint = anyDisjoint || lDisjoint[l];

    r.planeAll = planeN;
    r.planeN = planeN[c.probeIndex];
    r.planeD = planeD[c.probeIndex];
    r.scaledVert = (c.probeIndex < c.nbVerts) ? sv[c.probeIndex] : mk(0.0f, 0.0f, 0.0f);

    if (anyDisjoint)
        return r;
    r.p1 = true;

    float best = -MAXF;
    for (uint32_t l = 0; l < LANES; ++l) best = lpFMax(best, lSep[l]);
    uint32_t winner = 0;                       // lowest lane holding the max
    for (uint32_t l = 0; l < LANES; ++l) if (lSep[l] == best) { winner = l; break; }

    if (r.separation < best)
    { r.faceNormal = lNorm[winner]; r.separation = best; r.feature = lFeat[winner]; }

    // ---- stage 2: satTriangleNormal ----------------------------------------
    const float triProj = dot3(c.triN, c.t[0]);
    float min1 = MAXF, max1 = -MAXF;
    for (uint32_t i = 0; i < c.nbVerts; ++i)
    { const float d = dot3(sv[i], c.triN); max1 = std::fmax(d, max1); min1 = std::fmin(d, min1); }

    if ((min1 > triProj + c.contactDist) || (triProj > max1 + c.contactDist))
        return r;
    r.p2 = true;

    const float sep2 = min1 - triProj;
    if (r.separation < sep2)
    {
        r.faceNormal = mk(-c.triN.x, -c.triN.y, -c.triN.z);
        r.separation = sep2;
        r.feature = TRI_FEATURE_IDX;
    }

    // ---- stage 3: satEdgeCrossDirection2 -----------------------------------
    // The plane normals stage 1 stored are the gate, so this stage is simulated
    // against `planeN`, not recomputed.
    {
        float lSep3[LANES]; V3 lNorm3[LANES]; bool lPassed[LANES];
        for (uint32_t l = 0; l < LANES; ++l)
        { lSep3[l] = r.separation; lNorm3[l] = r.faceNormal; lPassed[l] = true; }

        const uint32_t nbEdges = c.nbEdges;
        bool bailed = false;
        for (uint32_t it = 0; it < nbEdges * 3u && !bailed; it += LANES)
        {
            for (uint32_t l = 0; l < LANES; ++l)
            {
                const uint32_t index = it + l;
                if (index >= nbEdges * 3u) continue;
                const uint32_t vStart = index % 3u;
                const uint32_t adj = c.adj[vStart];
                const bool active = (adj == BOUNDARY) || ((adj & NONCONVEX_FLAG) == 0u);
                if (!active) continue;
                const uint32_t vEnd = (vStart + 1u) % 3u;
                const uint32_t e = index / 3u;
                const uint32_t v0 = c.vbe[2 * e + 0], v1 = c.vbe[2 * e + 1];
                const uint32_t f0 = c.fbe[2 * e + 0], f1 = c.fbe[2 * e + 1];
                if (!(dot3(c.triN, planeN[f0]) < 0.0f || dot3(c.triN, planeN[f1]) < 0.0f)) continue;
                const V3 convexEdge = sub(sv[v0], sv[v1]);
                const V3 triEdge = sub(c.t[vEnd], c.t[vStart]);
                const V3 v = cross3(convexEdge, triEdge);
                if (!(dot3(v, v) > 1e-10f)) continue;
                // P9.36: convexTriangle.cuh:639 is `v.getNormalized()` --
                // `v * PxRecipSqrt(magnitudeSquared())`, a multiply (PxVec3.h:286).
                const float m = std::sqrt(dot3(v, v));
                const float rm = 1.0f / m;
                const V3 sv3 = mk(v.x * rm, v.y * rm, v.z * rm);
                // P9.89: the CUDA's select folds, not std::fmin/std::fmax -- the old
                // oracle mirrored the port's FMin/FMax and so could not see them.
                float minTri, maxTri, minConv, maxConv;
                refSupportTriangle(c.t, sv3, minTri, maxTri);
                refSupportConvex(sv, c.nbVerts, sv3, minConv, maxConv);
                const float o0 = minTri - maxConv, o1 = minConv - maxTri;
                if (minConv > (maxTri + c.contactDist) || minTri > (maxConv + c.contactDist))
                    lPassed[l] = false;
                else
                {
                    if (lSep3[l] < o0) { lSep3[l] = o0; lNorm3[l] = sv3; }
                    if (lSep3[l] < o1) { lSep3[l] = o1; lNorm3[l] = mk(-sv3.x, -sv3.y, -sv3.z); }
                }
            }
            for (uint32_t l = 0; l < LANES; ++l) if (!lPassed[l]) bailed = true;
        }
        if (bailed)
            return r;
        r.p3 = true;

        float best3 = -MAXF;
        for (uint32_t l = 0; l < LANES; ++l) best3 = lpFMax(best3, lSep3[l]);
        uint32_t w3 = 0;
        for (uint32_t l = 0; l < LANES; ++l) if (lSep3[l] == best3) { w3 = l; break; }
        if (r.separation < best3)
        { r.faceNormal = lNorm3[w3]; r.separation = best3; r.feature = EDGE_FEATURE_IDX; }
    }
    return r;
}

// An axis-aligned box hull: 8 vertices, 6 polygons, 12 edges, planes outward.
void makeBox(SatCase& c, float hx, float hy, float hz)
{
    c.nbVerts = 8; c.nbPolygons = 6; c.nbEdges = 12;
    c.verts = { mk(-hx,-hy,-hz), mk(hx,-hy,-hz), mk(hx,hy,-hz), mk(-hx,hy,-hz),
                mk(-hx,-hy,hz), mk(hx,-hy,hz), mk(hx,hy,hz), mk(-hx,hy,hz) };
    const float n[6][4] = {
        { 1, 0, 0, -hx }, { -1, 0, 0, -hx },
        { 0, 1, 0, -hy }, { 0, -1, 0, -hy },
        { 0, 0, 1, -hz }, { 0, 0, -1, -hz } };
    // minIndex is the vertex with the SMALLEST projection along that face's
    // normal, not one lying on the face.  satConvexFaceNormals uses it as `min0`,
    // the hull's near extent along that axis; seeding it with an on-plane vertex
    // makes min0 == max0 and every overlapping pair reports as separated.  That
    // is what the first version of this fixture did, and the symptom was that no
    // case reached stage 2 at all.
    const uint32_t minIdx[6] = { 0, 1, 0, 2, 0, 4 };
    // The 12 box edges: two vertices and the two faces meeting there.  The edge
    // stage needs a real table -- with zeros every edge degenerates to vertex 0
    // and the whole stage tests nothing.
    const uint32_t E[12][4] = {
        {0,1,3,5},{1,2,0,5},{2,3,2,5},{3,0,1,5},
        {4,5,3,4},{5,6,0,4},{6,7,2,4},{7,4,1,4},
        {0,4,1,3},{1,5,0,3},{2,6,0,2},{3,7,1,2} };
    c.vbe.clear(); c.fbe.clear();
    for (uint32_t e = 0; e < 12; ++e)
    { c.vbe.push_back(E[e][0]); c.vbe.push_back(E[e][1]); c.fbe.push_back(E[e][2]); c.fbe.push_back(E[e][3]); }
    c.planes.clear(); c.polyDescs.clear();
    for (uint32_t i = 0; i < 6; ++i)
    {
        for (uint32_t k = 0; k < 4; ++k) c.planes.push_back(n[i][k]);
        c.polyDescs.push_back(packPolyDesc(4u * i, 4u, minIdx[i]));
    }
}


// P9.89: an axis-aligned box between two arbitrary corners, with makeBox's vertex,
// face and edge numbering.  The cases that pin the support folds need a box whose
// vertical corner edge lies ON the z axis, which a centred box cannot provide.
void makeBoxAt(SatCase& c, V3 lo, V3 hi)
{
    c.nbVerts = 8; c.nbPolygons = 6; c.nbEdges = 12;
    c.verts = { mk(lo.x, lo.y, lo.z), mk(hi.x, lo.y, lo.z), mk(hi.x, hi.y, lo.z), mk(lo.x, hi.y, lo.z),
                mk(lo.x, lo.y, hi.z), mk(hi.x, lo.y, hi.z), mk(hi.x, hi.y, hi.z), mk(lo.x, hi.y, hi.z) };
    const float n[6][4] = {
        { 1, 0, 0, -hi.x }, { -1, 0, 0, lo.x },
        { 0, 1, 0, -hi.y }, { 0, -1, 0, lo.y },
        { 0, 0, 1, -hi.z }, { 0, 0, -1, lo.z } };
    const uint32_t minIdx[6] = { 0, 1, 0, 2, 0, 4 };
    const uint32_t E[12][4] = {
        {0,1,3,5},{1,2,0,5},{2,3,2,5},{3,0,1,5},
        {4,5,3,4},{5,6,0,4},{6,7,2,4},{7,4,1,4},
        {0,4,1,3},{1,5,0,3},{2,6,0,2},{3,7,1,2} };
    c.vbe.clear(); c.fbe.clear();
    for (uint32_t e = 0; e < 12; ++e)
    { c.vbe.push_back(E[e][0]); c.vbe.push_back(E[e][1]); c.fbe.push_back(E[e][2]); c.fbe.push_back(E[e][3]); }
    c.planes.clear(); c.polyDescs.clear();
    for (uint32_t i = 0; i < 6; ++i)
    {
        for (uint32_t k = 0; k < 4; ++k) c.planes.push_back(n[i][k]);
        c.polyDescs.push_back(packPolyDesc(4u * i, 4u, minIdx[i]));
    }
}

// A triangle normal from the winding, as the other cases compute it.
V3 windingNormal(const V3 t[3])
{
    const V3 e1 = sub(t[1], t[0]), e2 = sub(t[2], t[0]);
    const V3 nn = cross3(e1, e2);
    const float m = std::sqrt(dot3(nn, nn));
    return mk(nn.x / m, nn.y / m, nn.z / m);
}

// A drum with `sides` side faces plus two caps: nbPolygons > 32, so a lane owns
// SEVERAL polygons and the strided loop actually iterates.  Without that, lane ==
// polygon index and two whole classes of mistake are invisible: a `break` that
// should stop a lane scanning its remaining faces, and a feature index that
// records the lane instead of the polygon.
void makeDrum(SatCase& c, uint32_t sides, float r, float hz)
{
    c.nbVerts = 2u * sides; c.nbPolygons = sides + 2u; c.nbEdges = 3u * sides;
    c.verts.clear();
    for (uint32_t i = 0; i < sides; ++i)
    {
        const float a = 6.28318530718f * float(i) / float(sides);
        c.verts.push_back(mk(r * std::cos(a), r * std::sin(a), -hz));
    }
    for (uint32_t i = 0; i < sides; ++i)
    {
        const float a = 6.28318530718f * float(i) / float(sides);
        c.verts.push_back(mk(r * std::cos(a), r * std::sin(a), hz));
    }
    c.planes.clear(); c.polyDescs.clear();
    for (uint32_t i = 0; i < sides; ++i)
    {
        const float a = 6.28318530718f * (float(i) + 0.5f) / float(sides);
        const float nx = std::cos(a), ny = std::sin(a);
        c.planes.push_back(nx); c.planes.push_back(ny); c.planes.push_back(0.0f); c.planes.push_back(-r * 0.9f);
        // min index: the vertex furthest AGAINST this face normal
        uint32_t best = 0; float bestD = 1e30f;
        for (uint32_t v = 0; v < c.nbVerts; ++v)
        { const float d = c.verts[v].x * nx + c.verts[v].y * ny; if (d < bestD) { bestD = d; best = v; } }
        c.polyDescs.push_back(packPolyDesc(4u * i, 4u, best & 0xffu));
    }
    // edges: the two rings plus the verticals, 3*sides in total
    c.vbe.clear(); c.fbe.clear();
    for (uint32_t i = 0; i < sides; ++i)   // bottom ring: side face i and the -z cap
    { c.vbe.push_back(i); c.vbe.push_back((i + 1u) % sides); c.fbe.push_back(i); c.fbe.push_back(sides + 1u); }
    for (uint32_t i = 0; i < sides; ++i)   // top ring: side face i and the +z cap
    { c.vbe.push_back(sides + i); c.vbe.push_back(sides + (i + 1u) % sides); c.fbe.push_back(i); c.fbe.push_back(sides); }
    for (uint32_t i = 0; i < sides; ++i)   // verticals: two neighbouring side faces
    { c.vbe.push_back(i); c.vbe.push_back(sides + i); c.fbe.push_back((i + sides - 1u) % sides); c.fbe.push_back(i); }
    // +z and -z caps
    c.planes.push_back(0.0f); c.planes.push_back(0.0f); c.planes.push_back(1.0f); c.planes.push_back(-hz);
    c.polyDescs.push_back(packPolyDesc(0u, 4u, 0u));
    c.planes.push_back(0.0f); c.planes.push_back(0.0f); c.planes.push_back(-1.0f); c.planes.push_back(-hz);
    c.polyDescs.push_back(packPolyDesc(0u, 4u, sides & 0xffu));
}

struct SatPush { uint64_t input, output; uint32_t numCases, pad; };

// The case list is shared by both tests in this file: convexSatStages checks the
// three SAT stages one at a time, convexSelectFace checks the driver plus the
// face selection that runs on their result.  Sharing it means a case added to
// sharpen one is automatically under the other's oracle too.
void buildCases(std::vector<SatCase>& cases)
{
    {   // 1: the triangle sits well above the box -> stage 1 separates
        SatCase c; makeBox(c, 1.0f, 1.0f, 1.0f);
        c.scale = mk(1, 1, 1); c.rot[3] = 1.0f; c.contactDist = 0.01f;
        c.t[0] = mk(0, 0, 5); c.t[1] = mk(1, 0, 5); c.t[2] = mk(0, 1, 5);
        c.triN = mk(0, 0, 1); c.probeIndex = 0; c.name = "separated in z";
        cases.push_back(c);
    }
    {   // 2: the triangle cuts through the box -> both stages pass
        SatCase c; makeBox(c, 1.0f, 1.0f, 1.0f);
        c.scale = mk(1, 1, 1); c.rot[3] = 1.0f; c.contactDist = 0.01f;
        c.t[0] = mk(-0.5f, -0.5f, 0.2f); c.t[1] = mk(0.5f, -0.5f, 0.2f); c.t[2] = mk(0, 0.5f, 0.2f);
        c.triN = mk(0, 0, 1); c.probeIndex = 4; c.name = "triangle inside the box";
        cases.push_back(c);
    }
    {   // 3: a non-unit scale, which sends the plane normals through the divide
        //    in shape2Vertex and makes invMagnitude != 1
        SatCase c; makeBox(c, 1.0f, 1.0f, 1.0f);
        c.scale = mk(2.0f, 0.5f, 3.0f); c.rot[3] = 1.0f; c.contactDist = 0.02f;
        c.t[0] = mk(-0.4f, -0.2f, 1.0f); c.t[1] = mk(0.6f, -0.2f, 1.0f); c.t[2] = mk(0, 0.3f, 1.0f);
        c.triN = mk(0, 0, 1); c.probeIndex = 2; c.name = "scaled hull";
        cases.push_back(c);
    }
    {   // 4: a rotated scale, so quatRotate is not the identity
        SatCase c; makeBox(c, 1.0f, 1.0f, 1.0f);
        const float s = std::sin(0.3f), co = std::cos(0.3f);
        c.scale = mk(1.5f, 0.75f, 1.0f);
        c.rot[0] = 0.0f; c.rot[1] = 0.0f; c.rot[2] = s; c.rot[3] = co;
        c.contactDist = 0.02f;
        c.t[0] = mk(-0.4f, -0.2f, 0.9f); c.t[1] = mk(0.6f, -0.2f, 0.9f); c.t[2] = mk(0, 0.3f, 0.9f);
        c.triN = mk(0, 0, 1); c.probeIndex = 3; c.name = "scaled and rotated hull";
        cases.push_back(c);
    }
    {   // 5: THE TIE-BREAK.  A symmetric box and a triangle whose projection is
        //    identical against +x and -x, so two lanes report the same separation
        //    and only the lowest-lane rule decides which face wins.  Without it
        //    the reported feature index is the other face.
        SatCase c; makeBox(c, 1.0f, 1.0f, 1.0f);
        c.scale = mk(1, 1, 1); c.rot[3] = 1.0f; c.contactDist = 0.01f;
        c.t[0] = mk(0, 0, 0); c.t[1] = mk(0, 0.5f, 0); c.t[2] = mk(0, 0, 0.5f);
        c.triN = mk(1, 0, 0); c.probeIndex = 1; c.name = "tie between +x and -x faces";
        cases.push_back(c);
    }
    {   // 6: a degenerate plane normal (0,0,0), which drives normalize() down its
        //    zero branch: the vector stays unchanged and invMagnitude is +inf.
        SatCase c; makeBox(c, 1.0f, 1.0f, 1.0f);
        c.planes[4 * 5 + 0] = 0.0f; c.planes[4 * 5 + 1] = 0.0f; c.planes[4 * 5 + 2] = 0.0f;
        c.scale = mk(1, 1, 1); c.rot[3] = 1.0f; c.contactDist = 0.01f;
        c.t[0] = mk(-0.5f, -0.5f, 0.2f); c.t[1] = mk(0.5f, -0.5f, 0.2f); c.t[2] = mk(0, 0.5f, 0.2f);
        c.triN = mk(0, 0, 1); c.probeIndex = 5; c.name = "degenerate plane normal";
        cases.push_back(c);
    }

    {   // 7: a 34-polygon drum, so a lane owns more than one face
        SatCase c; makeDrum(c, 32u, 1.0f, 1.0f);
        c.scale = mk(1, 1, 1); c.rot[3] = 1.0f; c.contactDist = 0.02f;
        c.t[0] = mk(-0.3f, -0.3f, 0.5f); c.t[1] = mk(0.4f, -0.2f, 0.5f); c.t[2] = mk(0.0f, 0.5f, 0.5f);
        c.triN = mk(0, 0, 1); c.probeIndex = 33; c.name = "34-polygon drum";
        cases.push_back(c);
    }
    {   // 8: separated by the SECOND half of the test only.  For the +z face,
        //    min0 > max1 + contactDist fires when the triangle lies well BELOW the
        //    hull; the first half (min1 > max0 + contactDist) never does.
        SatCase c; makeBox(c, 1.0f, 1.0f, 1.0f);
        c.scale = mk(1, 1, 1); c.rot[3] = 1.0f; c.contactDist = 0.01f;
        c.t[0] = mk(-0.5f, -0.5f, -5.0f); c.t[1] = mk(0.5f, -0.5f, -5.0f); c.t[2] = mk(0, 0.5f, -5.0f);
        c.triN = mk(0, 0, 1); c.probeIndex = 0; c.name = "separated below (second test)";
        cases.push_back(c);
    }
    {   // 9: the TRIANGLE NORMAL wins the axis.  A tilted triangle cutting the box
        //    so that min1 - triProj beats every face's min1 - max0; without such a
        //    case the sign of the committed normal (-n0) is never observed.
        SatCase c; makeBox(c, 1.0f, 1.0f, 1.0f);
        c.scale = mk(1, 1, 1); c.rot[3] = 1.0f; c.contactDist = 0.02f;
        c.t[0] = mk(-0.9f, -0.9f, 0.9f); c.t[1] = mk(0.9f, -0.9f, -0.9f); c.t[2] = mk(0.0f, 0.9f, 0.0f);
        {   // the true unit normal of that triangle
            const V3 e1 = sub(c.t[1], c.t[0]), e2 = sub(c.t[2], c.t[0]);
            V3 n = cross3(e1, e2);
            const float m = std::sqrt(dot3(n, n));
            c.triN = mk(n.x / m, n.y / m, n.z / m);
        }
        c.probeIndex = 2; c.name = "triangle normal wins";
        cases.push_back(c);
    }

    {   // 10: THE EDGE-CROSS AXIS WINS.  Without this case stage 3 runs but never
        //     commits, so the whole tail of satEdgeCrossDirection2 -- the argmax
        //     across lanes, the sign choice between the two overlaps, and the
        //     EDGE_FEATURE_IDX write -- is dead code the fixture cannot see.
        //
        //     The triangle's first edge, (1.8,0,-1) -> (0,1.8,1), passes diagonally
        //     across the corner of the box near x=y=1.  Crossed with any of the
        //     box's vertical edges it yields the pure xy diagonal (1,1,0)/sqrt(2),
        //     along which the two hulls overlap by only 0.141 -- far less than the
        //     1.0 that the best face normal gives, so the edge axis takes the lead.
        //
        //     THE GATE MATTERS HERE.  The vertical edge closest to the crossing,
        //     edge 10 at (x=1,y=1), is REJECTED: its two faces are +x and +y and
        //     the triangle normal has a positive dot with both.  The axis arrives
        //     from the three other vertical edges, whose faces point away.  All
        //     three produce the identical vector (0,0,-2), so their separations are
        //     bitwise equal and the lowest-lane rule decides -- and the lowest of
        //     the three is lane 1, holding index 33, i.e. a value carried out of
        //     the SECOND pass of the strided loop.
        SatCase c; makeBox(c, 1.0f, 1.0f, 1.0f);
        c.scale = mk(1, 1, 1); c.rot[3] = 1.0f; c.contactDist = 0.02f;
        c.t[0] = mk(1.8f, 0.0f, -1.0f); c.t[1] = mk(0.0f, 1.8f, 1.0f); c.t[2] = mk(2.0f, 2.0f, 2.0f);
        {
            const V3 e1 = sub(c.t[1], c.t[0]), e2 = sub(c.t[2], c.t[0]);
            V3 nn = cross3(e1, e2);
            const float m = std::sqrt(dot3(nn, nn));
            c.triN = mk(nn.x / m, nn.y / m, nn.z / m);
        }
        c.probeIndex = 1; c.name = "edge-cross axis wins";
        cases.push_back(c);
    }

    {   // 11: SEPARATED BY AN EDGE-CROSS AXIS ALONE.  All six face normals and the
        //     triangle normal overlap; only cross(box vertical edge, triangle edge)
        //     -- again the xy diagonal -- pulls the two apart.  This is the only
        //     case that drives `isPassed = false` and therefore the only one that
        //     exercises the __any_sync early-out INSIDE the iteration loop.
        //
        //     The triangle passes diagonally outside the corner at x=y=1: its
        //     nearest projection along (1,1,0)/sqrt(2) is 1.838, the box reaches
        //     1.414, and contactDist is 0.02.  Three vertical edges report it, the
        //     first at index 24 -- inside the FIRST pass -- so the loop leaves
        //     without ever looking at indices 32..35.
        SatCase c; makeBox(c, 1.0f, 1.0f, 1.0f);
        c.scale = mk(1, 1, 1); c.rot[3] = 1.0f; c.contactDist = 0.02f;
        c.t[0] = mk(2.0f, 0.6f, -2.0f); c.t[1] = mk(0.6f, 2.0f, 2.0f); c.t[2] = mk(2.0f, 2.0f, 0.0f);
        {
            const V3 e1 = sub(c.t[1], c.t[0]), e2 = sub(c.t[2], c.t[0]);
            V3 nn = cross3(e1, e2);
            const float m = std::sqrt(dot3(nn, nn));
            c.triN = mk(nn.x / m, nn.y / m, nn.z / m);
        }
        c.probeIndex = 3; c.name = "separated by an edge-cross axis only";
        cases.push_back(c);
    }

    {   // 12: THE ADJACENCY FLAGS.  Case 10's geometry again, but the triangle's
        //     second edge is marked non-convex and its third carries a plain
        //     neighbour index.  Every other case leaves all three at BOUNDARY, so
        //     without this one `csIsActiveEdge` is only ever asked the BOUNDARY
        //     question and the vStart -> adj selection is untested.
        //
        //     It also pins the pairing between an edge's OWN flag and the edge it
        //     spans.  Closing the triangle the other way round -- (vStart+2)%3 --
        //     yields the same three edge directions negated, and the stage tries
        //     both signs anyway, so with all-BOUNDARY adjacency that mistake is
        //     invisible.  Here it is not: vStart 0 pairs with t0->t2 instead of
        //     t0->t1, the winning xy diagonal is never formed, and the committed
        //     axis moves to (0, 0.832, -0.555).
        SatCase c; makeBox(c, 1.0f, 1.0f, 1.0f);
        c.scale = mk(1, 1, 1); c.rot[3] = 1.0f; c.contactDist = 0.02f;
        c.t[0] = mk(1.8f, 0.0f, -1.0f); c.t[1] = mk(0.0f, 1.8f, 1.0f); c.t[2] = mk(2.0f, 2.0f, 2.0f);
        {
            const V3 e1 = sub(c.t[1], c.t[0]), e2 = sub(c.t[2], c.t[0]);
            V3 nn = cross3(e1, e2);
            const float m = std::sqrt(dot3(nn, nn));
            c.triN = mk(nn.x / m, nn.y / m, nn.z / m);
        }
        c.adj[0] = BOUNDARY;                  // active: the boundary case
        c.adj[1] = NONCONVEX_FLAG | 7u;       // inactive: a flagged neighbour
        c.adj[2] = 5u;                        // active: a plain neighbour index
        c.probeIndex = 1; c.name = "edge-cross with adjacency flags";
        cases.push_back(c);
    }

    {   // 13: THE OTHER ARM OF selectPolygonFeatureIndex.  When the SAT winner is
        //     the triangle normal or an edge cross, the selection first picks the
        //     best-aligned convex face and then asks whether some EDGE normal is
        //     better still.  In every case above the answer is yes, so the arm that
        //     instead keeps the face and looks for its best NEIGHBOUR never runs.
        //
        //     Which arm wins is a cone, and for a box it is a sharp one.  Write t
        //     for the angle between the winning axis and the nearest face normal.
        //     A box edge normal is the sum of two perpendicular face normals, so
        //     its squared normalised alignment is (1 + sin 2t)/2 against the face's
        //     cos^2 t = (1 + cos 2t)/2: the edge wins exactly when t > 22.5 deg.
        //     This triangle is nearly parallel to the +x face, which puts the
        //     triangle-normal axis about 9 deg off it -- inside the cone, so no
        //     edge can displace the face and the neighbour arm runs.
        SatCase c; makeBox(c, 1.0f, 1.0f, 1.0f);
        c.scale = mk(1, 1, 1); c.rot[3] = 1.0f; c.contactDist = 0.02f;
        c.t[0] = mk(1.0f, -0.5f, 2.0f); c.t[1] = mk(0.5f, 1.0f, -2.0f); c.t[2] = mk(0.5f, -2.0f, -0.5f);
        {
            const V3 e1 = sub(c.t[1], c.t[0]), e2 = sub(c.t[2], c.t[0]);
            V3 nn = cross3(e1, e2);
            const float m = std::sqrt(dot3(nn, nn));
            c.triN = mk(nn.x / m, nn.y / m, nn.z / m);
        }
        c.probeIndex = 0; c.name = "face keeps its best neighbour";
        cases.push_back(c);
    }

    {   // 14: REJECTED BY THE FACE STAGE AND BY NOTHING ELSE.  A small triangle
        //     sitting out at x in [-3, -1.5]: the -x face separates the pair, but
        //     the triangle's own normal axis still overlaps the box, so a driver
        //     that ran stage 2 anyway would come back with the same verdict for
        //     the wrong reason.  Every earlier stage-1 rejection is also a stage-2
        //     rejection, which is why this case exists.
        SatCase c; makeBox(c, 1.0f, 1.0f, 1.0f);
        c.scale = mk(1, 1, 1); c.rot[3] = 1.0f; c.contactDist = 0.02f;
        c.t[0] = mk(-3.0f, 0.0f, -1.5f); c.t[1] = mk(-2.5f, 1.0f, -1.5f); c.t[2] = mk(-1.5f, 1.0f, -0.5f);
        {
            const V3 e1 = sub(c.t[1], c.t[0]), e2 = sub(c.t[2], c.t[0]);
            V3 nn = cross3(e1, e2);
            const float m = std::sqrt(dot3(nn, nn));
            c.triN = mk(nn.x / m, nn.y / m, nn.z / m);
        }
        c.probeIndex = 1; c.name = "rejected by the face stage alone";
        cases.push_back(c);
    }

    {   // 15: REJECTED BY THE TRIANGLE-NORMAL STAGE AND BY NOTHING ELSE.  Every
        //     face axis overlaps and every edge-cross axis overlaps; only the
        //     triangle's own plane separates the pair.  The mirror of case 14, and
        //     the reason both exist is the same: without them a driver that
        //     ignored a stage's verdict would still report the right answer.
        SatCase c; makeBox(c, 1.0f, 1.0f, 1.0f);
        c.scale = mk(1, 1, 1); c.rot[3] = 1.0f; c.contactDist = 0.02f;
        c.t[0] = mk(-1.0f, -3.0f, -2.5f); c.t[1] = mk(1.0f, 0.5f, -1.5f); c.t[2] = mk(2.5f, -2.5f, 1.0f);
        {
            const V3 e1 = sub(c.t[1], c.t[0]), e2 = sub(c.t[2], c.t[0]);
            V3 nn = cross3(e1, e2);
            const float m = std::sqrt(dot3(nn, nn));
            c.triN = mk(nn.x / m, nn.y / m, nn.z / m);
        }
        c.probeIndex = 2; c.name = "rejected by the triangle stage alone";
        cases.push_back(c);
    }

    // Cases 16-18 all run the face SELECTION against the 34-polygon drum rather
    // than the box, and the reason is one property the box does not have.  Every
    // box edge joins two perpendicular faces, so every edge normal has the same
    // squared length, 2.  selectPolygonFeatureIndex compares `sqDp > maxDpSq *
    // enMagSq` and stores `sqDp / enMagSq`; with a single constant enMagSq the
    // multiply and the divide are interchangeable and so is the seed's scale.
    // The drum's edges join two side faces 11.25 degrees apart (enMagSq ~ 3.92)
    // or a side face and a cap (enMagSq = 2), so the two are told apart.
    {   // 16: the seed's scale and the multiply/divide asymmetry, plus the order
        //     of the winning edge's two faces
        SatCase c; makeDrum(c, 32u, 1.0f, 1.0f);
        c.scale = mk(1, 1, 1); c.rot[3] = 1.0f; c.contactDist = 0.02f;
        c.t[0] = mk(1.5f, 0.0f, -2.5f); c.t[1] = mk(0.5f, -2.0f, 2.0f); c.t[2] = mk(-1.5f, 0.5f, 3.0f);
        {
            const V3 e1 = sub(c.t[1], c.t[0]), e2 = sub(c.t[2], c.t[0]);
            V3 nn = cross3(e1, e2);
            const float m = std::sqrt(dot3(nn, nn));
            c.triN = mk(nn.x / m, nn.y / m, nn.z / m);
        }
        c.probeIndex = 33; c.name = "drum: triangle axis selects a face";
        cases.push_back(c);
    }
    {   // 17: the winning face is owned by a lane other than 0, which a 6-polygon
        //     box cannot show once the edge arm overwrites the answer anyway
        SatCase c; makeDrum(c, 32u, 1.0f, 1.0f);
        c.scale = mk(1, 1, 1); c.rot[3] = 1.0f; c.contactDist = 0.02f;
        c.t[0] = mk(2.5f, 1.5f, 0.5f); c.t[1] = mk(-1.0f, 2.5f, 0.5f); c.t[2] = mk(-0.5f, -2.0f, -0.5f);
        {
            const V3 e1 = sub(c.t[1], c.t[0]), e2 = sub(c.t[2], c.t[0]);
            V3 nn = cross3(e1, e2);
            const float m = std::sqrt(dot3(nn, nn));
            c.triN = mk(nn.x / m, nn.y / m, nn.z / m);
        }
        c.probeIndex = 20; c.name = "drum: the winner is not lane 0";
        cases.push_back(c);
    }
    {   // 18: an EDGE feature into the selection, with the edge comparison's seed
        //     doing real work -- from -PX_MAX_REAL a different edge wins
        SatCase c; makeDrum(c, 32u, 1.0f, 1.0f);
        c.scale = mk(1, 1, 1); c.rot[3] = 1.0f; c.contactDist = 0.02f;
        c.t[0] = mk(3.0f, -0.5f, -1.0f); c.t[1] = mk(1.5f, 0.5f, 2.0f); c.t[2] = mk(0.0f, -0.5f, -2.5f);
        {
            const V3 e1 = sub(c.t[1], c.t[0]), e2 = sub(c.t[2], c.t[0]);
            V3 nn = cross3(e1, e2);
            const float m = std::sqrt(dot3(nn, nn));
            c.triN = mk(nn.x / m, nn.y / m, nn.z / m);
        }
        c.probeIndex = 5; c.name = "drum: edge axis selects two faces";
        cases.push_back(c);
    }

    // Cases 19-21 (P9.89) pin the stage-3 support folds.  supportTriangle and
    // supportConvex are serial PxMin/PxMax SELECT folds seeded with +-PX_MAX_F32;
    // the port had them as GLSL FMin/FMax (and supportTriangle unseeded).  The two
    // spellings agree on every finite projection except a +-0 TIE, where the select
    // keeps the first-encountered zero and FMin/FMax give -0/+0, and they disagree
    // when the projections are NaN, where the CUDA keeps the seed.
    //
    // A zero's sign reaches an output only when BOTH operands of tempOverlap0/1
    // (minTri - maxConv, minConv - maxTri) are zeros, the edge axis is the stage's
    // best, and every lane that reports a zero agrees -- fmaxf gives +0 as soon as
    // one lane has +0.  Exact signed-zero projections need a vertex whose nonzero
    // coordinates meet zero components of the axis.  Both cases therefore put the
    // box's vertical corner edge ON the z axis, box vertices 2 and 6 at (0,0,zlo)
    // and (0,0,zhi), and use an axis in the plane x + y = 0.
    {   // 19: THE CONVEX MIN FOLD, at a genuine edge-edge contact.  Box
        //     [-2,0]^2 x [-1,1]; the triangle edge t0->t1, direction (1,-1,-0.5),
        //     lies in the plane x + y = 0 and crosses the box's corner edge at
        //     (0,0,0.05).  Its cross with the box edge (0,0,-2) is (-2,-2,-0), so the
        //     axis is (-a,-a,-0) and the box's zero projections are vertex 2 ->
        //     -0 + -0 + (-1)(-0) = +0 and vertex 6 -> -0 + -0 + (1)(-0) = -0.  The
        //     CUDA's select keeps vertex 2's +0; FMin gives -0.  Both triangle zeros
        //     are +0 (0.375a - 0.375a), so minConv - maxTri is +0 in the CUDA and -0
        //     in the old port -- and it is the stage's maximum, because every face
        //     and the triangle normal overlap.
        SatCase c; makeBoxAt(c, mk(-2.0f, -2.0f, -1.0f), mk(0.0f, 0.0f, 1.0f));
        c.scale = mk(1, 1, 1); c.rot[3] = 1.0f; c.contactDist = 0.02f;
        c.t[0] = mk(-0.375f, 0.375f, 0.3f); c.t[1] = mk(0.625f, -0.625f, -0.2f); c.t[2] = mk(0.8f, 0.6f, 0.3f);
        c.triN = windingNormal(c.t);
        c.probeIndex = 2; c.name = "P9.89 convex fold keeps the first zero";
        cases.push_back(c);
    }
    {   // 20: THE TRIANGLE MIN FOLD.  Box [-2,0]^2 x [0.1875,1.625]; the triangle
        //     edge t0->t1, direction (-0.625,0.625,-2), lies in the plane x + y = 0
        //     and meets the box edge's LINE at t1 = (-0,-0,-0.75), below the box.
        //     The axis is (a,a,+0): t0 projects to 0.625a - 0.625a = +0, t1 to
        //     -0 + -0 + (-0.75)(+0) = -0, and every box zero is +0.  The CUDA fold
        //     meets t0 first and keeps +0; min(d0, min(d1, d2)) gives -0, so
        //     minTri - maxConv is +0 against -0.  (The pair is not touching -- the
        //     gate on dot(triN, planeN) < 0 prunes the axes that would see the gap
        //     -- which is exactly what the CUDA reports, and what the port must.)
        SatCase c; makeBoxAt(c, mk(-2.0f, -2.0f, 0.1875f), mk(0.0f, 0.0f, 1.625f));
        c.scale = mk(1, 1, 1); c.rot[3] = 1.0f; c.contactDist = 0.02f;
        c.t[0] = mk(0.625f, -0.625f, 1.25f); c.t[1] = mk(-0.0f, -0.0f, -0.75f); c.t[2] = mk(-0.4375f, 0.5f, 0.3125f);
        c.triN = windingNormal(c.t);
        c.probeIndex = 3; c.name = "P9.89 triangle fold keeps the first zero";
        cases.push_back(c);
    }
    {   // 21: THE TRIANGLE FOLD'S SEED.  A box of half-extent 1e19 and a triangle
        //     that cuts through it.  Box edges (2e19) crossed with triangle edges
        //     (~2.3e19) overflow a component of v to inf; |v|^2 > 1e-10 holds,
        //     1/sqrt(inf) = 0 and getNormalized() gives inf * 0 = NaN in that
        //     component, so every projection on that axis is NaN.  The CUDA's
        //     supportTriangle keeps its seeds (+PX_MAX_F32 / -PX_MAX_F32), so
        //     `minConv > maxTri + contactDist` is PX_MAX_F32 > -PX_MAX_F32 and the
        //     pair is REJECTED by stage 3.  The unseeded min(d0, min(d1, d2)) returns
        //     NaN, both rejection tests are false, and the old port passed it.
        //     (supportConvex is seeded in both spellings and keeps +-PX_MAX_F32.)
        //     The normal is computed in double: the float cross product overflows.
        const float B = 1e19f;
        SatCase c; makeBox(c, B, B, B);
        c.scale = mk(1, 1, 1); c.rot[3] = 1.0f; c.contactDist = 0.02f;
        c.t[0] = mk(-0.9f * B, -1.1f * B, 0.15f * B);
        c.t[1] = mk(1.3f * B, -0.7f * B, 0.3f * B);
        c.t[2] = mk(-0.2f * B, 1.2f * B, 0.1f * B);
        {
            const double ax = double(c.t[1].x) - c.t[0].x, ay = double(c.t[1].y) - c.t[0].y, az = double(c.t[1].z) - c.t[0].z;
            const double bx = double(c.t[2].x) - c.t[0].x, by = double(c.t[2].y) - c.t[0].y, bz = double(c.t[2].z) - c.t[0].z;
            const double nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
            const double m = std::sqrt(nx * nx + ny * ny + nz * nz);
            c.triN = mk(float(nx / m), float(ny / m), float(nz / m));
        }
        c.probeIndex = 2; c.name = "P9.89 overflowed axis hits the seed";
        cases.push_back(c);
    }
}

PXG_TEST(convexSatStages)
{
    PxgSlangComputeDevice* dev = testDevice();

    std::vector<SatCase> cases;
    buildCases(cases);

    const uint32_t n = uint32_t(cases.size());
    std::vector<PxgBufferHandle> hullBufs;
    std::vector<float> input;
    for (SatCase& c : cases)
    {
        // The probe reads convexPlaneN/D at `probeIndex`, and stage 1 only writes
        // the first nbPolygons entries of that groupshared array.  An out-of-range
        // probe therefore reads whatever the previous dispatch left behind -- which
        // happened to be zero in the host-visible memory mode and was NOT zero in
        // the device-local one, so the first version of the "separated by an edge-cross
        // axis only" case passed one memory mode and failed the other.
        PXG_CHECK_MSG(c.probeIndex < c.nbPolygons && c.probeIndex < c.nbVerts,
                      "[%s] probeIndex %u is inside the hull (%u polygons, %u vertices)",
                      c.name, c.probeIndex, c.nbPolygons, c.nbVerts);
        buildBlob(c);
        PxgBufferHandle hb = createBufferFrom(dev, c.hull);
        hullBufs.push_back(hb);
        const uint64_t addr = dev->getBufferDeviceAddress(hb);

        const size_t base = input.size();
        input.resize(base + 128 / 4, 0.0f);
        float* p = input.data() + base;
        std::memcpy(p, &addr, 8);
        std::memcpy(p + 2, &c.probeIndex, 4);
        p[4] = c.scale.x; p[5] = c.scale.y; p[6] = c.scale.z; p[7] = c.contactDist;
        p[8] = c.rot[0]; p[9] = c.rot[1]; p[10] = c.rot[2]; p[11] = c.rot[3];
        for (uint32_t k = 0; k < 3; ++k)
        { p[12 + 4 * k] = c.t[k].x; p[13 + 4 * k] = c.t[k].y; p[14 + 4 * k] = c.t[k].z; }
        p[24] = c.triN.x; p[25] = c.triN.y; p[26] = c.triN.z;
        std::memcpy(p + 28, c.adj, 12);   // @112 triAdjTrisIdx
    }

    PxgBufferHandle bIn = createBufferFrom(dev, input);
    std::vector<float> outInit(size_t(n) * (64 / 4), -12345.0f);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    SatPush push{};
    push.input = dev->getBufferDeviceAddress(bIn);
    push.output = dev->getBufferDeviceAddress(bOut);
    push.numCases = n;

    {
        Kernel k("convexSatBda.spv", "convexSatLaunch", {}, sizeof(SatPush));
        k.dispatch(n, {}, push, 1u, 1u);
        dev->flush();
    }

    const auto got = downloadBuffer<float>(dev, bOut, outInit.size());

    size_t bad = 0;
    auto same = [](float a, float b) { uint32_t x, y; std::memcpy(&x, &a, 4); std::memcpy(&y, &b, 4); return x == y; };

    bool sawStage1Reject = false, sawStage2Reject = false, sawBothPass = false, sawTriFeature = false;
    bool sawManyPolys = false, sawEdgeFeature = false, sawStage3 = false, sawStage3Reject = false;
    uint32_t gpuIsTwin = 0;
    for (uint32_t i = 0; i < n; ++i)
    {
        const SatResult e = refSat(cases[i]);
        const float* r = got.data() + size_t(i) * (64 / 4);
        uint32_t p1, p2, feat;
        std::memcpy(&p1, r + 0, 4); std::memcpy(&p2, r + 1, 4); std::memcpy(&feat, r + 2, 4);

        if (!e.p1) sawStage1Reject = true;
        else if (!e.p2) sawStage2Reject = true;
        else sawBothPass = true;
        if (e.feature == TRI_FEATURE_IDX) sawTriFeature = true;
        if (cases[i].nbPolygons > 32u && e.p1) sawManyPolys = true;
        if (e.feature == EDGE_FEATURE_IDX) sawEdgeFeature = true;
        if (e.p3) sawStage3 = true;
        else if (e.p2) sawStage3Reject = true;

        const char* nm = cases[i].name;
        auto chkU = [&](const char* w, uint32_t g, uint32_t x) {
            if (g != x) { ++bad; if (bad <= 10) std::printf("  [%s] %s: got=%u exp=%u\n", nm, w, g, x); } };
        auto chkF = [&](const char* w, float g, float x) {
            if (!same(g, x)) { ++bad; if (bad <= 10) std::printf("  [%s] %s: got=%.9g exp=%.9g\n", nm, w, g, x); } };

        chkU("stage1", p1, e.p1 ? 1u : 0u);
        chkU("stage2", p2, e.p2 ? 1u : 0u);
        uint32_t p3; std::memcpy(&p3, r + 3, 4);
        chkU("stage3", p3, e.p3 ? 1u : 0u);
        if (e.p1)
        {
            chkU("feature", feat, e.feature);
            chkF("faceNormal.x", r[4], e.faceNormal.x);
            chkF("faceNormal.y", r[5], e.faceNormal.y);
            chkF("faceNormal.z", r[6], e.faceNormal.z);
            chkF("separation", r[7], e.separation);
            chkF("planeN.x", r[8], e.planeN.x);
            chkF("planeD", r[11], e.planeD);
        }
        chkF("scaledVert.x", r[12], e.scaledVert.x);
        chkF("scaledVert.y", r[13], e.scaledVert.y);

        // P9.89 diagnostic: does the GPU reproduce the pre-P9.89 folds' answer?
        // (Printed, not checked: it is how the twin was validated against the old
        // shader, and after the fix it reads 18 of 21.)
        {
            gSupVariant = 15;
            const SatResult tw = refSat(cases[i]);
            gSupVariant = 0;
            bool eq = p1 == (tw.p1 ? 1u : 0u) && p2 == (tw.p2 ? 1u : 0u) && p3 == (tw.p3 ? 1u : 0u);
            if (tw.p1)
                eq = eq && feat == tw.feature && same(r[4], tw.faceNormal.x) && same(r[5], tw.faceNormal.y) &&
                     same(r[6], tw.faceNormal.z) && same(r[7], tw.separation);
            if (eq) ++gpuIsTwin;
        }
    }

    for (PxgBufferHandle h : hullBufs) dev->destroyBuffer(h);
    dev->destroyBuffer(bIn);
    dev->destroyBuffer(bOut);

    PXG_CHECK_MSG(sawStage1Reject, "some case is rejected by the face-normal stage");
    PXG_CHECK_MSG(sawBothPass, "some case passes both stages");
    PXG_CHECK_MSG(bad == 0, "the SAT stages match the scalar oracle (%zu mismatches)", bad);
    PXG_CHECK_MSG(sawTriFeature, "some case commits the triangle normal as the axis");
    PXG_CHECK_MSG(sawManyPolys, "some case has more than 32 polygons and passes stage 1");
    PXG_CHECK_MSG(sawStage3, "some case reaches the edge-cross stage");
    PXG_CHECK_MSG(sawEdgeFeature, "some case commits an edge-cross axis");
    PXG_CHECK_MSG(sawStage3Reject, "some case is rejected by the edge-cross stage alone");
    PXG_CHECK_MSG(sawStage2Reject, "some case is rejected by the triangle-normal stage");

    // P9.89: the pre-P9.89 support folds, one line at a time, against the CUDA
    // oracle over the same cases.  Each must lose somewhere, or the fixture above
    // could not have told the fix from the bug.  Only the two MIN folds can reach
    // these outputs (see cases 19-20); the two MAX folds are pinned by
    // convexSatSupportFold.
    {
        auto bitsOf = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; };
        auto differs = [&](const SatResult& a, const SatResult& b) {
            return a.p1 != b.p1 || a.p2 != b.p2 || a.p3 != b.p3 || a.feature != b.feature ||
                   bitsOf(a.separation) != bitsOf(b.separation) ||
                   bitsOf(a.faceNormal.x) != bitsOf(b.faceNormal.x) ||
                   bitsOf(a.faceNormal.y) != bitsOf(b.faceNormal.y) ||
                   bitsOf(a.faceNormal.z) != bitsOf(b.faceNormal.z);
        };
        // 3 = the whole old csSupportTriangle: the seed only shows when BOTH its
        // folds are unseeded, since either seed alone still trips a rejection test
        const int variants[6] = { 1, 2, 3, 4, 8, 15 };
        uint32_t lost[6] = {};
        bool seedFlip = false, zeroTri = false, zeroConv = false;
        for (uint32_t i = 0; i < n; ++i)
        {
            gSupVariant = 0;
            const SatResult cuda = refSat(cases[i]);
            for (uint32_t v = 0; v < 6; ++v)
            {
                gSupVariant = variants[v];
                const SatResult twin = refSat(cases[i]);
                if (!differs(cuda, twin)) continue;
                ++lost[v];
                if (variants[v] == 3 && cuda.p2 && !cuda.p3 && twin.p3) seedFlip = true;
                if (variants[v] == 1 && cuda.p3 && twin.p3 && cuda.separation == 0.0f &&
                    bitsOf(cuda.separation) != bitsOf(twin.separation)) zeroTri = true;
                if (variants[v] == 4 && cuda.p3 && twin.p3 && cuda.separation == 0.0f &&
                    bitsOf(cuda.separation) != bitsOf(twin.separation)) zeroConv = true;
            }
        }
        gSupVariant = 0;
        std::printf("  P9.89 old-fold twins that disagree with the CUDA oracle, cases of %u: "
                    "triMin %u, triMax %u, triangle %u, convMin %u, convMax %u, all four %u; "
                    "the GPU reproduces the all-four twin in %u\n",
                    n, lost[0], lost[1], lost[2], lost[3], lost[4], lost[5], gpuIsTwin);
        PXG_CHECK_MSG(zeroConv, "the old csSupportConvex min fold commits -0 where the CUDA commits +0");
        PXG_CHECK_MSG(zeroTri, "the old csSupportTriangle min fold commits -0 where the CUDA commits +0");
        PXG_CHECK_MSG(seedFlip, "the unseeded old csSupportTriangle passes a pair the CUDA rejects on a NaN axis");
        PXG_CHECK_MSG(lost[5] >= 3u, "the whole old port loses on the three P9.89 cases (%u)", lost[5]);
    }
}

// ===========================================================================
// convexMeshSAT + delayGenerateContact + selectPolygonFeatureIndex
// ===========================================================================

struct SelectResult
{
    bool passed = false, delayContacts = false;
    // which arm of selectPolygonFeatureIndex produced faceIndex2: an edge that
    // beat every face, or the best neighbour of the winning face
    bool ranSelect = false, edgeBeatFace = false;
    uint32_t featureIndex = 0xffffffffu;
    uint32_t faceIndex1 = 0xFFFFFFFFu, faceIndex2 = 0xFFFFFFFFu;
    float separation = -MAXF;
    V3 minNormal{};
};

// The scalar 32-lane simulation of the driver's tail.  refSat already reproduces
// the three stages, so this reuses its result rather than re-deriving it: the
// point under test here is what happens AFTER the SAT, and re-simulating the SAT
// would let a mistake in the copy hide a mistake in the port.
SelectResult refSelect(const SatCase& c)
{
    SelectResult r;
    const SatResult sat = refSat(c);

    r.passed = sat.p1 && sat.p2 && sat.p3;

    // WHAT THE DRIVER LEAVES BEHIND WHEN THE SAT FAILS IS OBSERVABLE, and the
    // fixture asserts it.  convexMeshSAT returns by reference and does not undo a
    // commit: a pair rejected by stage 2 or stage 3 still carries whatever stage 1
    // wrote, and only a stage-1 rejection leaves the caller's seeds untouched.
    // Those seeds are the driver's own -- separation -PX_MAX_REAL, feature
    // 0xffffffff, and minNormal +PX_MAX_REAL in ALL THREE components, which is not
    // the zero vector the per-stage fixture uses.
    //
    // Asserting them is what makes a driver that ignores a stage's verdict
    // visible.  Without it, running stage 2 after stage 1 already rejected still
    // yields "did not pass", and the mistake reads as correct.
    if (sat.p1)
    {
        r.featureIndex = sat.feature;
        r.separation = sat.separation;
        r.minNormal = sat.faceNormal;
    }
    else
    {
        r.featureIndex = 0xffffffffu;
        r.separation = -MAXF;
        r.minNormal = mk(MAXF, MAXF, MAXF);
    }

    if (!r.passed)
        return r;

    // delayGenerateContact
    if (r.featureIndex != EDGE_FEATURE_IDX && r.featureIndex != TRI_FEATURE_IDX)
    {
        const V3 triLocNormal = mk(-c.triN.x, -c.triN.y, -c.triN.z);
        r.delayContacts = dot3(r.minNormal, triLocNormal) < 0.707106781f;
    }

    // selectPolygonFeatureIndex.  The plane normals are the ones stage 1 stored,
    // which refSat hands back through sat.planeAll -- the same array the shader
    // reads out of groupshared.
    r.faceIndex1 = r.featureIndex;
    if (r.featureIndex == TRI_FEATURE_IDX || r.featureIndex == EDGE_FEATURE_IDX)
    {
        r.ranSelect = true;
        float lProj[LANES]; uint32_t lBest[LANES];
        for (uint32_t l = 0; l < LANES; ++l) { lProj[l] = -MAXF; lBest[l] = 0; }
        for (uint32_t l = 0; l < LANES; ++l)
            for (uint32_t i = l; i < c.nbPolygons; i += LANES)
            {
                const float proj = dot3(r.minNormal, sat.planeAll[i]);
                if (lProj[l] < proj) { lProj[l] = proj; lBest[l] = i; }
            }
        float minProj = -MAXF;
        for (uint32_t l = 0; l < LANES; ++l) minProj = std::fmax(minProj, lProj[l]);
        uint32_t w = 0;
        for (uint32_t l = 0; l < LANES; ++l) if (lProj[l] == minProj) { w = l; break; }
        r.faceIndex1 = lBest[w];

        float lDp[LANES], lDp2[LANES]; uint32_t lEdge[LANES], lP2[LANES];
        for (uint32_t l = 0; l < LANES; ++l)
        { lDp[l] = minProj * minProj; lDp2[l] = -10.0f; lEdge[l] = 0xFFFFFFFFu; lP2[l] = 0xFFFFFFFFu; }

        for (uint32_t l = 0; l < LANES; ++l)
            for (uint32_t i = l; i < c.nbEdges; i += LANES)
            {
                const uint32_t f0 = c.fbe[2 * i + 0], f1 = c.fbe[2 * i + 1];
                const V3 edgeN = add(sat.planeAll[f0], sat.planeAll[f1]);
                const float enMagSq = dot3(edgeN, edgeN);
                const float dp = dot3(edgeN, r.minNormal);
                const float sqDp = dp * dp;
                if ((f0 == r.faceIndex1 || f1 == r.faceIndex1) && (sqDp > lDp2[l]))
                { lDp2[l] = sqDp; lP2[l] = (f0 == r.faceIndex1) ? f1 : f0; }
                if (dp > 0.0f && sqDp > (lDp[l] * enMagSq))
                { lDp[l] = sqDp / enMagSq; lEdge[l] = i; }
            }

        float maxDpSq = -MAXF;
        for (uint32_t l = 0; l < LANES; ++l) maxDpSq = std::fmax(maxDpSq, lDp[l]);
        w = 0;
        for (uint32_t l = 0; l < LANES; ++l) if (lDp[l] == maxDpSq) { w = l; break; }
        const uint32_t closestEdge = lEdge[w];

        if (closestEdge != 0xFFFFFFFFu)
        {
            r.edgeBeatFace = true;
            const uint32_t f0 = c.fbe[2 * closestEdge + 0], f1 = c.fbe[2 * closestEdge + 1];
            const float dp0 = dot3(sat.planeAll[f0], r.minNormal);
            const float dp1 = dot3(sat.planeAll[f1], r.minNormal);
            if (dp0 > dp1) { r.faceIndex1 = f0; r.faceIndex2 = f1; }
            else           { r.faceIndex1 = f1; r.faceIndex2 = f0; }
        }
        else
        {
            float maxDpSq2 = -MAXF;
            for (uint32_t l = 0; l < LANES; ++l) maxDpSq2 = std::fmax(maxDpSq2, lDp2[l]);
            w = 0;
            for (uint32_t l = 0; l < LANES; ++l) if (lDp2[l] == maxDpSq2) { w = l; break; }
            r.faceIndex2 = lP2[w];
        }
    }
    return r;
}

PXG_TEST(convexSelectFace)
{
    PxgSlangComputeDevice* dev = testDevice();

    std::vector<SatCase> cases;
    buildCases(cases);

    const uint32_t n = uint32_t(cases.size());
    std::vector<PxgBufferHandle> hullBufs;
    std::vector<float> input;
    for (SatCase& c : cases)
    {
        buildBlob(c);
        PxgBufferHandle hb = createBufferFrom(dev, c.hull);
        hullBufs.push_back(hb);
        const uint64_t addr = dev->getBufferDeviceAddress(hb);

        const size_t base = input.size();
        input.resize(base + 128 / 4, 0.0f);
        float* p = input.data() + base;
        std::memcpy(p, &addr, 8);
        std::memcpy(p + 2, &c.probeIndex, 4);
        p[4] = c.scale.x; p[5] = c.scale.y; p[6] = c.scale.z; p[7] = c.contactDist;
        p[8] = c.rot[0]; p[9] = c.rot[1]; p[10] = c.rot[2]; p[11] = c.rot[3];
        for (uint32_t k = 0; k < 3; ++k)
        { p[12 + 4 * k] = c.t[k].x; p[13 + 4 * k] = c.t[k].y; p[14 + 4 * k] = c.t[k].z; }
        p[24] = c.triN.x; p[25] = c.triN.y; p[26] = c.triN.z;
        std::memcpy(p + 28, c.adj, 12);
    }

    PxgBufferHandle bIn = createBufferFrom(dev, input);
    std::vector<float> outInit(size_t(n) * (32 / 4), -12345.0f);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    SatPush push{};
    push.input = dev->getBufferDeviceAddress(bIn);
    push.output = dev->getBufferDeviceAddress(bOut);
    push.numCases = n;

    {
        Kernel k("convexSatBda.spv", "convexSelectLaunch", {}, sizeof(SatPush));
        k.dispatch(n, {}, push, 1u, 1u);
        dev->flush();
    }

    const auto got = downloadBuffer<float>(dev, bOut, outInit.size());

    size_t bad = 0;
    auto same = [](float a, float b) { uint32_t x, y; std::memcpy(&x, &a, 4); std::memcpy(&y, &b, 4); return x == y; };

    bool sawDelay = false, sawNoDelay = false, sawTwoFaces = false, sawEdgePick = false;
    bool sawFacePassthrough = false, sawSelectRun = false, sawNeighbourPick = false;
    for (uint32_t i = 0; i < n; ++i)
    {
        const SelectResult e = refSelect(cases[i]);
        const float* r = got.data() + size_t(i) * (32 / 4);
        uint32_t passed, feat, delay, f1, f2;
        std::memcpy(&passed, r + 0, 4); std::memcpy(&feat, r + 1, 4);
        std::memcpy(&delay, r + 2, 4);  std::memcpy(&f1, r + 3, 4);
        std::memcpy(&f2, r + 4, 4);

        if (e.passed)
        {
            if (e.delayContacts) sawDelay = true; else sawNoDelay = true;
            if (e.ranSelect) sawSelectRun = true; else sawFacePassthrough = true;
            if (e.faceIndex2 != 0xFFFFFFFFu) sawTwoFaces = true;
            if (e.ranSelect) { if (e.edgeBeatFace) sawEdgePick = true; else sawNeighbourPick = true; }
        }

        const char* nm = cases[i].name;
        auto chkU = [&](const char* w, uint32_t g, uint32_t x) {
            if (g != x) { ++bad; if (bad <= 12) std::printf("  [%s] %s: got=%u exp=%u\n", nm, w, g, x); } };
        auto chkF = [&](const char* w, float g, float x) {
            if (!same(g, x)) { ++bad; if (bad <= 12) std::printf("  [%s] %s: got=%.9g exp=%.9g\n", nm, w, g, x); } };

        chkU("satPassed", passed, e.passed ? 1u : 0u);
        chkU("featureIndex", feat, e.featureIndex);
        chkF("separation", r[5], e.separation);
        chkF("minNormal.x", r[6], e.minNormal.x);
        chkF("minNormal.y", r[7], e.minNormal.y);
        if (e.passed)
        {
            chkU("delayContacts", delay, e.delayContacts ? 1u : 0u);
            chkU("faceIndex1", f1, e.faceIndex1);
            chkU("faceIndex2", f2, e.faceIndex2);
        }
    }
    for (PxgBufferHandle h : hullBufs) dev->destroyBuffer(h);
    dev->destroyBuffer(bIn);
    dev->destroyBuffer(bOut);

    PXG_CHECK_MSG(bad == 0, "the SAT driver and face selection match the scalar oracle (%zu mismatches)", bad);
    PXG_CHECK_MSG(sawSelectRun, "some case selects a face from a tri or edge axis");
    PXG_CHECK_MSG(sawFacePassthrough, "some case already has a face index and passes it through");
    PXG_CHECK_MSG(sawTwoFaces, "some case comes back with a second face");
    PXG_CHECK_MSG(sawDelay, "some case delays its contacts");
    PXG_CHECK_MSG(sawNoDelay, "some case does not delay its contacts");
    PXG_CHECK_MSG(sawEdgePick, "some case picks both faces of an edge that beats every face");
    PXG_CHECK_MSG(sawNeighbourPick, "some case picks the best neighbour of the winning face");
}

// ===========================================================================
// P9.89: convexSatSupportFold -- csSupportTriangle and csSupportConvex, per lane
// ===========================================================================
//
// supportTriangle (convexTriangle.cuh:507-518) and supportConvex (:494-505) are
// serial PxMin/PxMax select folds, seeded +-PX_MAX_F32, in index order.  The port
// had FMin/FMax (and no seed in the triangle fold).  The two agree on distinct
// finite projections, so the lanes below are built to make them disagree, and
// every word is compared bit for bit against the CUDA transcription:
//
//   lanes  0-7   the axis is one of the eight signed-zero vectors: every
//                projection is a signed zero, so every fold is one long zero tie
//                and the answer is the FIRST vertex's zero (the select) against
//                -0 / +0 (FMin / FMax);
//   lanes  8-13  one non-round axis component and two signed zeros: ties only
//                among the vertices with a zero coordinate on that axis;
//   lanes 14-21  a generic non-round axis -- the folds agree, and this checks the
//                ordinary path;
//   lanes 22-25  a NaN axis component: every projection is NaN, the CUDA keeps
//                both seeds, FMin(NaN, FMin(NaN, NaN)) is NaN;
//   lanes 26-27  (+inf, +-0, +-0) with the triangle at x > 0: every triangle
//                projection is +inf, and selectMin(+inf, PX_MAX_F32) keeps the seed;
//   lanes 28-29  a huge axis that overflows every triangle projection to -inf,
//                the mirror for the max fold;
//   lanes 30-31  the FIRST triangle projection alone is NaN (t0 carries an inf
//                that meets a zero axis component): the seeded fold skips it,
//                an unseeded select would keep it.
//
// Eight hulls: a 27-point grid with zero coordinates, 64 pseudo-random points on a
// zero-rich pool, 48 non-round points, the second pool again through a non-unit
// scale and a rotation (so shape space is not the vertex space), and those four
// with their vertex order reversed.
namespace
{
struct Lcg
{
    uint32_t s;
    uint32_t next() { s = s * 1664525u + 1013904223u; return s; }
    float uni(float lo, float hi) { return lo + (hi - lo) * float(next() >> 8) * (1.0f / 16777216.0f); }
    float sgn() { return (next() >> 31) ? -1.0f : 1.0f; }
    float zero() { return (next() >> 31) ? -0.0f : 0.0f; }
};
} // namespace

PXG_TEST(convexSatSupportFold)
{
    PxgSlangComputeDevice* dev = testDevice();

    const uint32_t GROUPS = 8u;
    const size_t IN_FLOATS = (64u + 32u * 64u) / 4u;
    const size_t OUT_FLOATS = 32u * 4u;
    const float QNAN = std::numeric_limits<float>::quiet_NaN();
    const float INF = std::numeric_limits<float>::infinity();

    Lcg rng{ 0x9e3779b9u };

    // ---- the four hulls (vertices only: csScaleVerts reads nothing else) -----
    std::vector<SatCase> hulls(GROUPS);
    {
        SatCase& h = hulls[0];
        const float g[3] = { 0.6875f, 0.0f, -1.375f };
        for (uint32_t a = 0; a < 3; ++a)
            for (uint32_t b = 0; b < 3; ++b)
                for (uint32_t k = 0; k < 3; ++k)
                    h.verts.push_back(mk(g[(a + k) % 3], g[b], g[(k + 2u * a) % 3]));
    }
    {
        const float pool[8] = { 0.0f, 0.0f, 0.4375f, -0.8173541f, 1.3125f, -2.71875f, 0.1234567f, -0.0f };
        for (uint32_t i = 0; i < 64u; ++i)
            hulls[1].verts.push_back(mk(pool[rng.next() % 8u], pool[rng.next() % 8u], pool[rng.next() % 8u]));
        hulls[3].verts = hulls[1].verts;
    }
    for (uint32_t i = 0; i < 48u; ++i)
        hulls[2].verts.push_back(mk(rng.uni(-1.9f, 1.9f), rng.uni(-1.9f, 1.9f), rng.uni(-1.9f, 1.9f)));
    for (uint32_t gi = 0; gi < GROUPS; ++gi)
    {
        SatCase& h = hulls[gi];
        h.scale = mk(1, 1, 1); h.rot[0] = h.rot[1] = h.rot[2] = 0.0f; h.rot[3] = 1.0f;
    }
    {
        SatCase& h = hulls[3];
        h.scale = mk(1.3f, 0.7f, 2.1f);
        const float qa = 0.37f, qb = -0.22f, qc = 0.61f, qd = 0.66f;
        const float qm = std::sqrt(qa * qa + qb * qb + qc * qc + qd * qd);
        h.rot[0] = qa / qm; h.rot[1] = qb / qm; h.rot[2] = qc / qm; h.rot[3] = qd / qm;
    }
    // Hulls 4-7 are 0-3 with the vertex order REVERSED.  Which zero wins a tie is
    // the first one met, and the first vertex fixes it for a whole group, so the
    // reversed copies both double the tie coverage and pin the fold's direction.
    for (uint32_t gi = 4; gi < GROUPS; ++gi)
    {
        hulls[gi] = hulls[gi - 4u];
        std::vector<V3>& v = hulls[gi].verts;
        for (size_t a = 0, b = v.size() - 1u; a < b; ++a, --b) std::swap(v[a], v[b]);
    }
    for (uint32_t gi = 0; gi < GROUPS; ++gi)
    { hulls[gi].nbVerts = uint32_t(hulls[gi].verts.size()); hulls[gi].nbPolygons = 0; hulls[gi].nbEdges = 0; }

    // ---- the lanes -------------------------------------------------------------
    struct Lane { V3 t[3]; V3 dir; };
    std::vector<Lane> lanes(size_t(GROUPS) * 32u);
    for (uint32_t gi = 0; gi < GROUPS; ++gi)
        for (uint32_t l = 0; l < 32u; ++l)
        {
            Lane& L = lanes[size_t(gi) * 32u + l];
            auto rv = [&]() { return mk(rng.sgn() * rng.uni(0.05f, 2.3f), rng.sgn() * rng.uni(0.05f, 2.3f),
                                        rng.sgn() * rng.uni(0.05f, 2.3f)); };
            for (uint32_t k = 0; k < 3; ++k) L.t[k] = rv();
            if (l < 8u)
            {
                const bool neg[3] = { (l & 1u) != 0u, (l & 2u) != 0u, (l & 4u) != 0u };
                L.dir = mk(neg[0] ? -0.0f : 0.0f, neg[1] ? -0.0f : 0.0f, neg[2] ? -0.0f : 0.0f);
                // Plant the triangle's zero signs.  A projection is -0 only when all
                // three products are -0, i.e. every coordinate's sign is opposite to
                // its axis zero's; one coordinate of the same sign makes it +0.
                // Every pattern mixes both zeros, so either the min or the max fold
                // disagrees with FMin/FMax on it.
                static const char* pat[6] = { "+-+", "-+-", "++-", "--+", "+--", "-++" };
                const char* P = pat[(l + gi) % 6u];
                for (uint32_t k = 0; k < 3; ++k)
                {
                    float* c[3] = { &L.t[k].x, &L.t[k].y, &L.t[k].z };
                    for (uint32_t a = 0; a < 3; ++a)
                    {
                        const float m = std::fabs(*c[a]);
                        const bool wantNeg = (P[k] == '-') ? !neg[a] : (a == 0u ? neg[a] : (*c[a] < 0.0f));
                        *c[a] = wantNeg ? -m : m;
                    }
                }
            }
            else if (l < 14u)
            {
                const uint32_t axis = l % 3u;
                float d[3] = { rng.zero(), rng.zero(), rng.zero() };
                d[axis] = rng.sgn() * rng.uni(0.2f, 1.6f);
                L.dir = mk(d[0], d[1], d[2]);
                // two of the three triangle vertices get a zero on that axis
                float* c0 = (axis == 0) ? &L.t[0].x : (axis == 1) ? &L.t[0].y : &L.t[0].z;
                float* c2 = (axis == 0) ? &L.t[2].x : (axis == 1) ? &L.t[2].y : &L.t[2].z;
                *c0 = rng.zero(); *c2 = rng.zero();
            }
            else if (l < 22u)
                L.dir = mk(rng.uni(-1.7f, 1.7f), rng.uni(-1.7f, 1.7f), rng.uni(-1.7f, 1.7f));
            else if (l < 26u)
            {
                L.dir = mk(rng.uni(-1.7f, 1.7f), rng.uni(-1.7f, 1.7f), rng.uni(-1.7f, 1.7f));
                if (l == 22u) L.dir.x = QNAN; else if (l == 23u) L.dir.y = QNAN; else if (l == 24u) L.dir.z = QNAN;
                else L.dir = mk(QNAN, QNAN, QNAN);
            }
            else if (l < 28u)
            {
                L.dir = mk(INF, rng.zero(), rng.zero());
                for (uint32_t k = 0; k < 3; ++k) L.t[k].x = std::fabs(L.t[k].x);
            }
            else if (l < 30u)
            {
                L.dir = mk(3.0e38f, 3.0e38f, rng.uni(-1.0f, 1.0f));
                for (uint32_t k = 0; k < 3; ++k)
                { L.t[k].x = -1.0f - std::fabs(L.t[k].x); L.t[k].y = -1.0f - std::fabs(L.t[k].y); }
            }
            else
            {
                L.dir = mk(rng.uni(0.2f, 1.7f), rng.zero(), -rng.uni(0.2f, 1.7f));
                L.t[0].y = (l & 1u) ? -INF : INF;
            }
        }

    // ---- upload and dispatch ---------------------------------------------------
    std::vector<PxgBufferHandle> hullBufs;
    std::vector<float> input(size_t(GROUPS) * IN_FLOATS, 0.0f);
    for (uint32_t gi = 0; gi < GROUPS; ++gi)
    {
        SatCase& h = hulls[gi];
        buildBlob(h);
        PxgBufferHandle hb = createBufferFrom(dev, h.hull);
        hullBufs.push_back(hb);
        const uint64_t addr = dev->getBufferDeviceAddress(hb);
        float* p = input.data() + size_t(gi) * IN_FLOATS;
        std::memcpy(p, &addr, 8);
        p[4] = h.scale.x; p[5] = h.scale.y; p[6] = h.scale.z;
        p[8] = h.rot[0]; p[9] = h.rot[1]; p[10] = h.rot[2]; p[11] = h.rot[3];
        for (uint32_t l = 0; l < 32u; ++l)
        {
            const Lane& L = lanes[size_t(gi) * 32u + l];
            float* q = p + 16 + 16 * l;
            for (uint32_t k = 0; k < 3; ++k) { q[4 * k] = L.t[k].x; q[4 * k + 1] = L.t[k].y; q[4 * k + 2] = L.t[k].z; }
            q[12] = L.dir.x; q[13] = L.dir.y; q[14] = L.dir.z;
        }
    }
    PxgBufferHandle bIn = createBufferFrom(dev, input);
    std::vector<float> outInit(size_t(GROUPS) * OUT_FLOATS, -12345.0f);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    SatPush push{};
    push.input = dev->getBufferDeviceAddress(bIn);
    push.output = dev->getBufferDeviceAddress(bOut);
    push.numCases = GROUPS;
    {
        Kernel k("convexSatBda.spv", "convexSupportLaunch", {}, sizeof(SatPush));
        k.dispatch(GROUPS, {}, push, 1u, 1u);
        dev->flush();
    }
    const auto got = downloadBuffer<float>(dev, bOut, outInit.size());
    for (PxgBufferHandle h : hullBufs) dev->destroyBuffer(h);
    dev->destroyBuffer(bIn);
    dev->destroyBuffer(bOut);

    // ---- compare against the CUDA transcription, and the old fold against it ----
    auto bitsOf = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; };
    const char* word[4] = { "minTri", "maxTri", "minConv", "maxConv" };
    size_t bad = 0;
    uint32_t twinLost[4] = {}, twinZero[4] = {}, twinSeed[4] = {}, gpuIsTwin = 0, nanOut = 0;
    for (uint32_t gi = 0; gi < GROUPS; ++gi)
    {
        const SatCase& h = hulls[gi];
        std::vector<V3> sv(h.nbVerts);
        for (uint32_t i = 0; i < h.nbVerts; ++i) sv[i] = vertex2Shape(h.verts[i], h.scale, h.rot);
        for (uint32_t l = 0; l < 32u; ++l)
        {
            const Lane& L = lanes[size_t(gi) * 32u + l];
            float cuda[4], twin[4];
            gSupVariant = 0;
            refSupportTriangle(L.t, L.dir, cuda[0], cuda[1]);
            refSupportConvex(sv, h.nbVerts, L.dir, cuda[2], cuda[3]);
            gSupVariant = 15;
            refSupportTriangle(L.t, L.dir, twin[0], twin[1]);
            refSupportConvex(sv, h.nbVerts, L.dir, twin[2], twin[3]);
            gSupVariant = 0;
            const float* r = got.data() + size_t(gi) * OUT_FLOATS + 4u * l;
            for (uint32_t w = 0; w < 4; ++w)
            {
                if (std::isnan(cuda[w])) ++nanOut;
                if (bitsOf(r[w]) != bitsOf(cuda[w]))
                {
                    ++bad;
                    if (bad <= 12)
                        std::printf("  [hull %u lane %2u] %s: got=%08x (%.9g) exp=%08x (%.9g)\n", gi, l, word[w],
                                    bitsOf(r[w]), r[w], bitsOf(cuda[w]), cuda[w]);
                }
                const bool sameTwin = bitsOf(twin[w]) == bitsOf(cuda[w]);
                if (bitsOf(r[w]) == bitsOf(twin[w]) || (std::isnan(r[w]) && std::isnan(twin[w]))) ++gpuIsTwin;
                if (sameTwin) continue;
                ++twinLost[w];
                if (cuda[w] == 0.0f && twin[w] == 0.0f) ++twinZero[w];
                else ++twinSeed[w];
            }
        }
    }
    std::printf("  old-fold twin loses: minTri %u (zero %u, seed %u), maxTri %u (zero %u, seed %u), "
                "minConv %u (zero %u), maxConv %u (zero %u); GPU equals the twin in %u of %u words\n",
                twinLost[0], twinZero[0], twinSeed[0], twinLost[1], twinZero[1], twinSeed[1],
                twinLost[2], twinZero[2], twinLost[3], twinZero[3], gpuIsTwin, GROUPS * 32u * 4u);

    PXG_CHECK_MSG(bad == 0, "the support folds match the CUDA's select folds bit for bit (%zu mismatches)", bad);
    PXG_CHECK_MSG(nanOut == 0, "the CUDA's seeded select folds never return NaN (%u)", nanOut);
    for (uint32_t w = 0; w < 4; ++w)
        PXG_CHECK_MSG(twinZero[w] >= 8u, "the FMin/FMax twin loses %s on a +-0 tie in at least 8 lanes (%u)",
                      word[w], twinZero[w]);
    PXG_CHECK_MSG(twinSeed[0] >= 4u && twinSeed[1] >= 4u,
                  "the unseeded triangle twin loses both seeds on NaN/inf projections (%u / %u)",
                  twinSeed[0], twinSeed[1]);
}

} // namespace
