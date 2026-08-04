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

#include "pxgTestFramework.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
float normalizeInPlace(V3& v)
{
    const float m = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (m > 0.0f) { v.x /= m; v.y /= m; v.z /= m; }
    return m;
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

    r.planeN = planeN[c.probeIndex];
    r.planeD = planeD[c.probeIndex];
    r.scaledVert = (c.probeIndex < c.nbVerts) ? sv[c.probeIndex] : mk(0.0f, 0.0f, 0.0f);

    if (anyDisjoint)
        return r;
    r.p1 = true;

    float best = -MAXF;
    for (uint32_t l = 0; l < LANES; ++l) best = std::fmax(best, lSep[l]);
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
                const float m = std::sqrt(dot3(v, v));
                const V3 sv3 = mk(v.x / m, v.y / m, v.z / m);
                float minTri = MAXF, maxTri = -MAXF, minConv = MAXF, maxConv = -MAXF;
                for (uint32_t k = 0; k < 3; ++k)
                { const float d = dot3(c.t[k], sv3); minTri = std::fmin(d, minTri); maxTri = std::fmax(d, maxTri); }
                for (uint32_t k = 0; k < c.nbVerts; ++k)
                { const float d = dot3(sv[k], sv3); minConv = std::fmin(d, minConv); maxConv = std::fmax(d, maxConv); }
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
        for (uint32_t l = 0; l < LANES; ++l) best3 = std::fmax(best3, lSep3[l]);
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

PXG_TEST(convexSatStages)
{
    PxgSlangComputeDevice* dev = testDevice();

    std::vector<SatCase> cases;

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

    const uint32_t n = uint32_t(cases.size());
    std::vector<PxgBufferHandle> hullBufs;
    std::vector<float> input;
    for (SatCase& c : cases)
    {
        // The probe reads convexPlaneN/D at `probeIndex`, and stage 1 only writes
        // the first nbPolygons entries of that groupshared array.  An out-of-range
        // probe therefore reads whatever the previous dispatch left behind -- which
        // happened to be zero in the default allocator and was NOT zero under
        // PXG_DEVICE_V2, so the first version of the "separated by an edge-cross
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
    (void)sawStage2Reject;
}

} // namespace
