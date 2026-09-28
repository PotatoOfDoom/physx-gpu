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

// Verification of fcConvexTri.slang (cloth convexTriangleContactGen:
// convexMeshSAT<false> + selectPolygonFeatureIndex + polyClip4 +
// computeBarycentric gate + contactReduce<false,true,5> + PxgOutputIndex +
// writeRigidVsDeformableContact).  Synthetic hulls (a box and a hexagonal
// prism) are authored analytically in the REAL ConvexScratch blob layout.
// The CPU mirror replicates the faithful warp serialization (per-lane
// strided argmax scans with strict-greater update, then lowest-lane tie):
// - gDebug pins the SAT verdict, featureIndex and both selected faces;
// - writer slots: the face-1 batch reserves BEFORE the face-2 batch
//   (single thread) and within a batch slots follow lane order, so the
//   per-case slot sequence sorted by index is deterministic; triangleId ==
//   case index maps every slot back to its case;
// - the info fourth dword must equal rigidMaterialIndex with bit 31
//   CLEARED (fixtures include materials with bit 31 set);
// - a case whose reduced mask has bit 31 set (signed mask > 0 quirk:
//   reservation advances, nothing written) is tainted — practically
//   unreachable with <= 8-vert reference polygons.
// Slack taints: SAT disjoint/separation margins, near-ties in every argmax
// (face separation, alignment, edge metrics, reduce anchors), polyClip
// gates (tranche-95 notes), barycentric eps-window boundaries and the
// dupe-cull threshold.
//
// KNOWN LIMITS (verified textually instead): PxPlane::normalize on the
// reference plane is output-neutral here (plane0 only feeds sign tests,
// positive scaling); the contactReduce PCM extra-deepest step equals the
// first fill-loop iteration at maxPoints = 5 (its guard folds away); the
// exact bary eps-window VALUE is essentially unpinned (any admitted
// boundary contact classifies as soft and is only counted, not compared —
// the window's EXISTENCE is pinned, its width verified textually); the
// firm/soft scheme absorbs boundary-hovering membership, which is
// genuinely bistable across implementations; the dupe cull only acts on
// coincident
// clip points, which are always taint-flagged.

#include "pxgTestFramework.h"

// P9.15d drives the engine-ABI kernel beside the Phase-1 core, so it needs the
// real narrowphase structs.
#include "host_defines.h"
#include "PxgFEMCloth.h"
#include "PxgContactManager.h"
#include "PxgConvexConvexShape.h"
#include "PxgSimulationCoreDesc.h"
#include "PxsTransformCache.h"
#include "PxNodeIndex.h"
#include "geometry/PxGeometry.h"
#include <cstring>

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;
using namespace physx;   // P9.15d drives the engine-ABI kernel beside the Phase-1 core

struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };
struct FctCase { F4 convexScale, convexScaleRot, t0, t1, t2, triNormal, convexP, convexQ; U4 ids, ids2; };
// P9.15c: the hull blob and the whole contact writer moved from bindings to BDA
// addresses, so the shader body can be called by a kernel that has no descriptor
// set to bind them into.  Only gCase and gDebug -- both test scaffolding, neither
// part of the engine ABI -- are still bound.
struct FctParams
{
    uint32_t numCases, maxContacts, pad0, pad1;
    uint64_t hullAddr;
    uint64_t wPoint, wNormalPen, wBary, wInfo, wCount, wByX, wTempByX, wIdxByX;
};
static_assert(sizeof(FctParams) == 88, "FctParams must match the shader blob");

struct V3 { float x, y, z; };
static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 add(V3 a, V3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 mul(V3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
static V3 divv(V3 a, V3 b) { return { a.x / b.x, a.y / b.y, a.z / b.z }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 crossp(V3 a, V3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static V3 normalize3(V3 v) { const float m = std::sqrt(dot3(v, v)); return mul(v, 1.0f / m); }
static V3 getNorm0(V3 v) { const float m = dot3(v, v); return m > 0 ? mul(v, 1.0f / std::sqrt(m)) : V3{ 0, 0, 0 }; }
static V3 v3f(F4 f) { return { f.x, f.y, f.z }; }

static const uint32_t NUMTMP = 32;
static const uint32_t TRI_FEAT = 0x7FffFFffu;
static const float EPSR = 1.192092896e-07f;
static const float MAXF = 3.4028235e38f;

struct Slack
{
    bool near = false;
    void noteSc(float l, float r, float scale, int = 0) { if (std::fabs(l - r) < 1e-5f * scale) near = true; }
};

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
static V3 vertex2Shape(V3 v, V3 scale, F4 rot)
{
    V3 t = quatRotate(rot, v);
    t = { t.x * scale.x, t.y * scale.y, t.z * scale.z };
    return quatRotateInv(rot, t);
}
static V3 v2sNormal(V3 n, V3 scale, F4 rot)
{
    V3 t = quatRotate(rot, n);
    t = divv(t, scale);
    return quatRotateInv(rot, t);
}

// ---- hull blob (ConvexScratch layout) ---------------------------------------
struct Hull
{
    std::vector<V3> verts;
    std::vector<F4> planes;
    std::vector<std::vector<uint32_t>> faces;
    std::vector<std::pair<uint32_t, uint32_t>> edgeFaces;

    void serialize(std::vector<uint32_t>& blob, uint32_t& offset) const
    {
        const uint32_t nbV = (uint32_t)verts.size();
        const uint32_t nbP = (uint32_t)faces.size();
        const uint32_t nbE = (uint32_t)edgeFaces.size();
        std::vector<uint8_t> vertexData8;
        std::vector<uint32_t> polyDescs;
        for (uint32_t p = 0; p < nbP; ++p)
        {
            uint32_t minIdx = 0;
            float best = MAXF;
            const V3 n = { planes[p].x, planes[p].y, planes[p].z };
            for (uint32_t v = 0; v < nbV; ++v)
            {
                const float d = dot3(n, verts[v]);
                if (d < best) { best = d; minIdx = v; }
            }
            polyDescs.push_back(((uint32_t)vertexData8.size() << 16) | ((uint32_t)faces[p].size() << 8) | minIdx);
            for (uint32_t idx : faces[p]) vertexData8.push_back((uint8_t)idx);
        }

        offset = (uint32_t)blob.size();
        auto pushF4 = [&](float a, float b, float c, float d) {
            blob.push_back(*(uint32_t*)&a); blob.push_back(*(uint32_t*)&b);
            blob.push_back(*(uint32_t*)&c); blob.push_back(*(uint32_t*)&d);
        };
        pushF4(0, 0, 0, 0);                              // center of mass
        blob.push_back((nbE << 16) | (nbV << 8) | nbP);  // header.x
        blob.push_back(0); blob.push_back(0); blob.push_back(0);
        pushF4(1, 1, 1, 0);                              // extents (unused)
        for (const V3& v : verts) pushF4(v.x, v.y, v.z, 0);
        for (const F4& p : planes) pushF4(p.x, p.y, p.z, p.w);
        for (uint32_t d : polyDescs) blob.push_back(d);
        std::vector<uint8_t> bytes;
        for (uint32_t e = 0; e < nbE * 4; ++e) bytes.push_back(0); // verticesByEdges16 (unused here)
        for (const auto& ef : edgeFaces) { bytes.push_back((uint8_t)ef.first); bytes.push_back((uint8_t)ef.second); }
        for (uint32_t v = 0; v < nbV * 3; ++v) bytes.push_back(0); // facesByVertices8 (unused)
        for (uint8_t b : vertexData8) bytes.push_back(b);
        while (bytes.size() % 4) bytes.push_back(0);
        for (size_t i = 0; i < bytes.size(); i += 4)
            blob.push_back((uint32_t)bytes[i] | ((uint32_t)bytes[i + 1] << 8) |
                           ((uint32_t)bytes[i + 2] << 16) | ((uint32_t)bytes[i + 3] << 24));
    }
};

static Hull makeBox(float hx, float hy, float hz)
{
    Hull h;
    for (int i = 0; i < 8; ++i)
        h.verts.push_back({ (i & 1) ? hx : -hx, (i & 2) ? hy : -hy, (i & 4) ? hz : -hz });
    h.planes = { { 1, 0, 0, -hx }, { -1, 0, 0, -hx }, { 0, 1, 0, -hy },
                 { 0, -1, 0, -hy }, { 0, 0, 1, -hz }, { 0, 0, -1, -hz } };
    h.faces = { { 1, 3, 7, 5 }, { 0, 4, 6, 2 }, { 2, 6, 7, 3 },
                { 0, 1, 5, 4 }, { 4, 5, 7, 6 }, { 0, 2, 3, 1 } };
    h.edgeFaces = { { 5, 3 }, { 5, 1 }, { 5, 0 }, { 5, 2 },
                    { 4, 3 }, { 4, 1 }, { 4, 0 }, { 4, 2 },
                    { 1, 3 }, { 0, 3 }, { 1, 2 }, { 0, 2 } };
    return h;
}

static Hull makePrism(uint32_t n, const float* radii, float hh)
{
    // IRREGULAR extruded polygon: distinct side normals and vertex spacings
    // break the argmax ties a regular prism produces in the feature-select
    // and contact-reduce stages
    Hull h;
    for (uint32_t k = 0; k < n; ++k)
    {
        const float a = 2.0f * 3.14159265f * k / n;
        h.verts.push_back({ radii[k] * std::cos(a), radii[k] * std::sin(a), -hh });
    }
    for (uint32_t k = 0; k < n; ++k)
    {
        const float a = 2.0f * 3.14159265f * k / n;
        h.verts.push_back({ radii[k] * std::cos(a), radii[k] * std::sin(a), hh });
    }
    h.planes.push_back({ 0, 0, 1, -hh });
    h.planes.push_back({ 0, 0, -1, -hh });
    std::vector<uint32_t> top, bot;
    for (uint32_t k = 0; k < n; ++k) top.push_back(n + k);
    for (uint32_t k = 0; k < n; ++k) bot.push_back(n - 1 - k);
    h.faces.push_back(top);
    h.faces.push_back(bot);
    for (uint32_t k = 0; k < n; ++k)
    {
        const uint32_t k1 = (k + 1) % n;
        const V3 e = sub(h.verts[k1], h.verts[k]);
        V3 sn = normalize3({ e.y, -e.x, 0 }); // outward for CCW ring
        h.planes.push_back({ sn.x, sn.y, sn.z, -dot3(sn, h.verts[k]) });
        h.faces.push_back({ k, k1, n + k1, n + k });
        h.edgeFaces.push_back({ 1, 2 + k });
        h.edgeFaces.push_back({ 0, 2 + k });
        h.edgeFaces.push_back({ 2 + k, 2 + (k + n - 1) % n });
    }
    return h;
}

// ---- polyClip4 mirror (tranche 95) ------------------------------------------
struct M33 { V3 c0, c1, c2; };
static V3 mT(const M33& m, V3 v) { return add(add(mul(m.c0, v.x), mul(m.c1, v.y)), mul(m.c2, v.z)); }
static V3 mTT(const M33& m, V3 v) { return { dot3(m.c0, v), dot3(m.c1, v), dot3(m.c2, v) }; }

// P9.86c: mutant twins of the port's findRotationMatrixFromZ as llvmpipe would run it
// without the P9.86 spellings (see PXG_TEST fcConvexTri_contactGenRotLaneZeroSign).
// 0 -- the default, and the only value outside that test -- is the CUDA read as IEEE.
//   1: col2 without the x lane of `+ u * c1u.z` (a literal +0 lane: F3 folds it away)
//   2: col0 without the z lane of `+ u * c1u.x` (the same fold, the other select)
//   4: v.x = -to.x instead of 0 - to.x (F2: OpFSub(literal 0, x) becomes -x)
static int gRotZLaneVariant = 0;

static M33 findRotZ(V3 to, Slack& sl)
{
    sl.noteSc(std::fabs(to.z), 0.9999f, 1.0f, 21);
    const float e = to.z, f = std::fabs(e);
    M33 m;
    if (f <= 0.9999f)
    {
        const float vx = -to.y, vy = to.x;
        const float h = 1.0f / (1.0f + e);
        const float hvx = h * vx, hvxy = hvx * vy;
        m.c0 = { e + hvx * vx, hvxy, vy };
        m.c1 = { hvxy, e + h * vy * vy, -vx };
        m.c2 = { -vy, vx, e };
    }
    else
    {
        const V3 from = { 0, 0, 1 }, absFrom = { 0, 1, 0 };
        const V3 u = sub(absFrom, from);
        V3 v = sub(absFrom, to);
        if (gRotZLaneVariant & 4) v.x = -to.x;
        const float c1 = -(2.0f / dot3(u, u));
        const float c2 = -(2.0f / dot3(v, v));
        const float c3 = c1 * c2 * dot3(u, v);
        const V3 c1u = mul(u, c1), c2v = mul(v, c2), c3v = mul(v, c3);
        V3 col0 = add(add(mul(u, c3v.x), mul(v, c2v.x)), mul(u, c1u.x));
        if (gRotZLaneVariant & 2) col0.z = u.z * c3v.x + v.z * c2v.x;
        col0.x += 1.0f;
        V3 col1 = add(add(mul(u, c3v.y), mul(v, c2v.y)), mul(u, c1u.y)); col1.y += 1.0f;
        V3 col2 = add(add(mul(u, c3v.z), mul(v, c2v.z)), mul(u, c1u.z));
        if (gRotZLaneVariant & 1) col2.x = u.x * c3v.z + v.x * c2v.z;
        col2.z += 1.0f;
        m.c0 = col0; m.c1 = col1; m.c2 = col2;
    }
    return m;
}

static bool refContains(const V3* verts, uint32_t numVerts, V3 p, Slack& sl)
{
    const float tx = p.x, ty = p.y;
    uint32_t numIntersects = 0;
    for (uint32_t tI = 0; tI < numVerts; ++tI)
    {
        const V3 start = verts[tI];
        const V3 endP = verts[tI + 1 == numVerts ? 0 : tI + 1];
        const float jy = endP.y, iy = start.y;
        const float minY = std::min(jy, iy), maxY = std::max(jy, iy);
        sl.noteSc(ty, minY, 20.0f, 1);
        sl.noteSc(ty, maxY, 20.0f, 1);
        if ((minY != maxY) && ty >= minY && ty <= maxY)
        {
            const float jx = endP.x, ix = start.x;
            const float jix = ix - jx, jiy = iy - jy, jty = ty - jy;
            const float part1 = jty * jix;
            const float part2 = (jx + EPSR) * jiy;
            const float part3 = tx * jiy;
            const bool comp = jiy > 0.0f;
            const float tmp = part1 + part2;
            const float comp1 = comp ? tmp : part3;
            const float comp2 = comp ? part3 : tmp;
            sl.noteSc(comp1, comp2, std::fabs(comp1) + std::fabs(comp2) + 1.0f, 2);
            if (comp1 >= comp2) ++numIntersects;
        }
    }
    return (numIntersects % 2) == 1;
}

static float area2D(V3 a, V3 b, V3 c)
{
    const V3 ca = sub(a, c), cb = sub(b, c);
    return ca.x * cb.y - ca.y * cb.x;
}

static bool refSegIntersect(V3 ipA, V3 ipB, V3 rpA, V3 rpB, V3 rMin, V3 rMax,
                            V3& pAA, V3& pBB, Slack& sl)
{
    const V3 iMin = { std::min(ipA.x, ipB.x), std::min(ipA.y, ipB.y), std::min(ipA.z, ipB.z) };
    const V3 iMax = { std::max(ipA.x, ipB.x), std::max(ipA.y, ipB.y), std::max(ipA.z, ipB.z) };
    sl.noteSc(iMin.x, rMax.x, 20.0f, 3);
    sl.noteSc(rMin.x, iMax.x, 20.0f, 3);
    sl.noteSc(iMin.y, rMax.y, 20.0f, 3);
    sl.noteSc(rMin.y, iMax.y, 20.0f, 3);
    if (iMin.x > rMax.x || rMin.x > iMax.x || iMin.y > rMax.y || rMin.y > iMax.y) return false;
    const float a1 = area2D(rpA, rpB, ipA);
    const float a2 = area2D(rpA, rpB, ipB);
    const float segMag = dot3(sub(rpB, rpA), sub(rpB, rpA)) + dot3(sub(ipB, ipA), sub(ipB, ipA));
    if (std::fabs(a1) + std::fabs(a2) < 5e-3f * segMag) sl.near = true; // area cancellation
    sl.noteSc(a1 * a2, 0.0f, std::fabs(a1 * a2) + std::fabs(a1) + std::fabs(a2) + 1.0f, 4);
    if ((a1 * a2) < 0.0f)
    {
        const float a3 = area2D(ipA, ipB, rpA);
        const float a4 = area2D(ipA, ipB, rpB);
        if (std::fabs(a3) + std::fabs(a4) < 5e-3f * segMag) sl.near = true;
        sl.noteSc(a3 * a4, 0.0f, std::fabs(a3 * a4) + std::fabs(a3) + std::fabs(a4) + 1.0f, 4);
        if ((a3 * a4) < 0.0f)
        {
            // ill-conditioned intersections drift along the edges: taint when
            // a division denominator is small relative to the operands (the
            // 2D areas suffer catastrophic cancellation for near-parallel
            // segments) — margin 1e-2 * magnitude
            sl.noteSc(a2 - a1, 0.0f, (std::fabs(a1) + std::fabs(a2)) * 1000.0f);
            sl.noteSc(a4 - a3, 0.0f, (std::fabs(a3) + std::fabs(a4)) * 1000.0f);
            const float t0 = a1 / (a2 - a1);
            pAA = sub(ipA, mul(sub(ipB, ipA), t0));
            const float t1 = a3 / (a4 - a3);
            pBB = sub(rpA, mul(sub(rpB, rpA), t1));
            return true;
        }
    }
    return false;
}

static uint32_t refPolyClip(const std::vector<V3>& v0, uint32_t nbVerts0, V3 axis, F4 plane0, F4 plane1,
                            const V3* v1, float contactDist, float* ct, Slack& sl)
{
    const M33 rot = findRotZ(axis, sl);
    V3 triangleP[3], polyPoint[32] = {};
    for (uint32_t l = 0; l < 3; ++l) triangleP[l] = mT(rot, v1[l]);
    for (uint32_t l = 0; l < nbVerts0; ++l) polyPoint[l] = mT(rot, v0[l]);
    const float sqCd = contactDist * contactDist;
    const V3 p0n = { plane0.x, plane0.y, plane0.z };
    const V3 p1n = { plane1.x, plane1.y, plane1.z };
    uint32_t num = 0;

    uint32_t mask = 0;
    for (uint32_t j = 0; j < nbVerts0; ++j)
        if (refContains(triangleP, 3, polyPoint[j], sl)) mask |= 1u << j;
    {
        float dist[32];
        V3 nor[32], cp[32];
        uint32_t keep = 0;
        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            if (!(mask & (1u << lane))) continue;
            const V3 proj = sub(v0[lane], mul(p1n, dot3(p1n, v0[lane]) + plane1.w));
            const V3 dir = sub(v0[lane], proj);
            const float cosT = dot3(dir, p0n);
            const float sq = dot3(dir, dir);
            sl.noteSc(cosT, 0.0f, std::fabs(cosT) + std::sqrt(sq) + 1.0f, 5);
            sl.noteSc(sq, sqCd, sq + sqCd + 1.0f, 6);
            sl.noteSc(sq, 0.0f, 10.0f, 7);
            if (cosT < 0.0f)
            {
                if (sq > sqCd) continue;
                dist[lane] = std::sqrt(sq);
                nor[lane] = mul(getNorm0(dir), -1.0f);
            }
            else
            {
                dist[lane] = -std::sqrt(sq);
                nor[lane] = getNorm0(dir);
            }
            cp[lane] = proj;
            keep |= 1u << lane;
        }
        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            if (!(keep & (1u << lane))) continue;
            const uint32_t idx = (uint32_t)__builtin_popcount(keep & ((1u << lane) - 1)) + num;
            if (idx < NUMTMP)
            {
                ct[8 * idx + 0] = cp[lane].x; ct[8 * idx + 1] = cp[lane].y;
                ct[8 * idx + 2] = cp[lane].z; ct[8 * idx + 3] = dist[lane];
                ct[8 * idx + 4] = nor[lane].x; ct[8 * idx + 5] = nor[lane].y;
                ct[8 * idx + 6] = nor[lane].z; ct[8 * idx + 7] = 0.0f;
            }
        }
        num += (uint32_t)__builtin_popcount(keep);
    }

    for (uint32_t rStart = 0, rEnd = 2; rStart < 3; rEnd = rStart++)
    {
        const V3 rpA = triangleP[rStart], rpB = triangleP[rEnd];
        const V3 rMin = { std::min(rpA.x, rpB.x), std::min(rpA.y, rpB.y), std::min(rpA.z, rpA.z) };
        const V3 rMax = { std::max(rpA.x, rpB.x), std::max(rpA.y, rpB.y), std::max(rpA.z, rpA.z) };
        float dist[32];
        V3 nor[32], cp[32];
        uint32_t m3 = 0;
        for (uint32_t lane = 0; lane < 32 && lane < nbVerts0; ++lane)
        {
            const V3 ipA = polyPoint[lane];
            const V3 ipB = polyPoint[lane + 1 == nbVerts0 ? 0 : lane + 1];
            V3 pAA, pBB;
            if (!refSegIntersect(ipA, ipB, rpA, rpB, rMin, rMax, pAA, pBB, sl)) continue;
            const V3 pA = mTT(rot, pAA);
            const V3 cpt = mTT(rot, pBB);
            const V3 dir = sub(pA, cpt);
            const float cosT = dot3(dir, p0n);
            const float sq = dot3(dir, dir);
            sl.noteSc(cosT, 0.0f, std::fabs(cosT) + std::sqrt(sq) + 1.0f, 5);
            sl.noteSc(sq, sqCd, sq + sqCd + 1.0f, 6);
            sl.noteSc(sq, 0.0f, 10.0f, 7);
            if (cosT < 0.0f)
            {
                if (sq > sqCd) continue;
                dist[lane] = std::sqrt(sq);
                nor[lane] = mul(getNorm0(dir), -1.0f);
            }
            else
            {
                dist[lane] = -std::sqrt(sq);
                nor[lane] = getNorm0(dir);
            }
            cp[lane] = cpt;
            m3 |= 1u << lane;
        }
        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            if (!(m3 & (1u << lane))) continue;
            const uint32_t idx = (uint32_t)__builtin_popcount(m3 & ((1u << lane) - 1)) + num;
            if (idx < NUMTMP)
            {
                ct[8 * idx + 0] = cp[lane].x; ct[8 * idx + 1] = cp[lane].y;
                ct[8 * idx + 2] = cp[lane].z; ct[8 * idx + 3] = dist[lane];
                ct[8 * idx + 4] = nor[lane].x; ct[8 * idx + 5] = nor[lane].y;
                ct[8 * idx + 6] = nor[lane].z; ct[8 * idx + 7] = 0.0f;
            }
        }
        num += (uint32_t)__builtin_popcount(m3);
    }
    return num;
}

// firm: bary comfortably inside the eps window — membership is stable
// across implementations; soft (edge-hovering) contacts may legitimately
// appear on one side only (phase-3 points sit ~1e-4 from the boundary and
// their bary drifts with the 2D intersection conditioning)
struct RefContact { V3 p, n; float sep; F4 bary; bool firm; };

// ---------------------------------------------------------------------------
// P9.15d -- cloth_boxTriangleContactGenLaunch, verified as a GPU DIFFERENTIAL.
//
// THE ORACLE.  This kernel's contact machine -- SAT, feature selection, polygon
// clipping, contact reduction, the writer -- is not its own: P9.15c lifted it into
// pxgSlangConvexTriCloth, where fcConvexTri's Phase-1 battery verifies it against a
// full CPU transcription.  Re-transcribing it here would be a third copy of the
// thing that is already checked.
//
// What P9.15d actually adds is the CALLER GLUE, and glue is exactly what a
// differential can pin: seed one arena, run the REAL kernel on it, then run the
// ALREADY-SWEPT core entry point (ctFcConvexTri) over FctCases a host mirror of the
// glue built, and compare the two contact streams.  The core is identical on both
// sides by construction, so any difference is the glue -- and the mirror is fifty
// lines of transforms rather than a second SAT.  Same arrangement as P8.97d.
//
// The comparison is a MULTISET: both sides reserve their slots with an atomicAdd
// whose order across pairs is a race (the P9.11 rule), so contacts are matched by
// content and claimed by index.
//
// WHAT THE RIG IS BUILT TO DISCRIMINATE
//   * a cloth transform that is NOT the identity and NOT the rigid's, so B5's
//     composition (conj(rigidQ) * clothQ) cannot collapse into either factor;
//   * a rigid box with a non-uniform scale AND a non-identity scale rotation, so
//     B2's "the cloth is not scaled" is separable from "nothing is scaled";
//   * TWO contact-manager pairs with DIFFERENT rest distances, so B6's cmIdx index
//     is separable from a cache ref;
//   * a filter list that names TWO of one triangle's three corners and ALL THREE
//     of another's -- B1's AND: the first triangle must still produce contacts,
//     the second must not;
//   * a midphase counter that OVERSHOOTS the stack, so B3's clamp and B4's
//     unclamped atomicMax are measured in the same run and pull opposite ways.
// ---------------------------------------------------------------------------
namespace btri
{
// FctCase / FctParams are this file's own -- one definition, two users.
struct V3c { float x, y, z; };
struct Tri4 { uint32_t x, y, z, w; };
struct FemInfoHost { uint64_t pairInd0; uint32_t pairInd1, rigidMatInd; };
static_assert(sizeof(FemInfoHost) == 16, "PxgFemOtherContactInfo mirror");

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
struct DevBuf { PxgBufferHandle h = 0; uint64_t addr = 0; };
template <typename T>
static DevBuf mkBuf(PxgSlangComputeDevice* dev, const std::vector<T>& d)
{
    DevBuf b; b.h = createBufferFrom(dev, d); b.addr = dev->getBufferDeviceAddress(b.h);
    return b;
}
struct WBufs { DevBuf point, normalPen, bary, info, count, byX, tempByX, idxByX, sortedByX; };
static WBufs mkWriter(PxgSlangComputeDevice* dev, uint32_t cap)
{
    WBufs w;
    w.point = mkBuf(dev, std::vector<F4>(cap, F4{ -9, -9, -9, -9 }));
    w.normalPen = mkBuf(dev, std::vector<F4>(cap, F4{ -9, -9, -9, -9 }));
    w.bary = mkBuf(dev, std::vector<F4>(cap, F4{ -9, -9, -9, -9 }));
    w.info = mkBuf(dev, std::vector<FemInfoHost>(cap, FemInfoHost{ 0xdeadbeefull, 0xdeadbeefu, 0xdeadbeefu }));
    w.count = mkBuf(dev, std::vector<uint32_t>{ 0u });
    w.byX = mkBuf(dev, std::vector<uint64_t>(cap, 0xffffffffffffffffull));
    w.tempByX = mkBuf(dev, std::vector<uint32_t>(cap, 0xeeeeeeeeu));
    w.idxByX = mkBuf(dev, std::vector<uint32_t>(cap, 0xeeeeeeeeu));
    w.sortedByX = mkBuf(dev, std::vector<uint32_t>(cap, 0xeeeeeeeeu));
    return w;
}
static void killWriter(PxgSlangComputeDevice* dev, const WBufs& w)
{
    for (PxgBufferHandle h : { w.point.h, w.normalPen.h, w.bary.h, w.info.h, w.count.h,
                               w.byX.h, w.tempByX.h, w.idxByX.h, w.sortedByX.h })
        dev->destroyBuffer(h);
}

struct BTriPush
{
    float toleranceLength; uint32_t pad4;
    uint64_t cmInputs, transformCache, contactDistance, restDistance, gpuShapes, femClothes;
    uint64_t shapeToRigidRemap, stackPtr, midphasePairsNum, filterPairs;
    uint32_t nbFilterPairs, stackSizeBytes;
    uint64_t stackSizeNeeded;
    uint64_t pad104;   // paramAlign(80) == 16, so the writer lands at 112, not 104
    uint64_t bPoint, bNormalPen, bBary, bInfo, bCount, bByX, bTempByX, bIdxByX, bSortedByX;
    uint32_t bMaxNumContacts, pad188;
};
static_assert(sizeof(BTriPush) == 192, "cloth box-triangle blob must be 192 bytes");
static_assert(offsetof(BTriPush, stackSizeNeeded) == 96, "stackSizeNeeded @96");
static_assert(offsetof(BTriPush, bPoint) == 112, "writer @112 -- paramAlign(80) is 16");

struct Q { float x, y, z, w; };
static Q qmul(Q a, Q b)
{
    return Q{ a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
              a.w * b.y + a.y * b.w + a.z * b.x - a.x * b.z,
              a.w * b.z + a.z * b.w + a.x * b.y - a.y * b.x,
              a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z };
}
static Q qconj(Q q) { return Q{ -q.x, -q.y, -q.z, q.w }; }
static V3c qrot(Q q, V3c v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d2 = (q.x * vx + q.y * vy + q.z * vz);
    return V3c{ vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d2,
                vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d2,
                vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d2 };
}
static V3c qrotInv(Q q, V3c v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d2 = (q.x * vx + q.y * vy + q.z * vz);
    return V3c{ vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * d2,
                vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * d2,
                vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * d2 };
}
} // namespace btri

PXG_TEST(femClothSimBoxTriangleContactGen)
{
    using namespace btri;
    PxgSlangComputeDevice* dev = testDevice();

    // ---- geometry ----
    const V3c boxExt = { 1.1f, 0.6f, 1.7f };
    const V3c boxP = { 0.3f, -0.4f, 0.2f };
    const float bh = std::sin(0.2f), bc = std::cos(0.2f);
    const Q boxQ = { 0.0f, bh, 0.0f, bc };                  // 0.4 rad about +y
    // The CLOTH's own transform: neither identity nor the rigid's (B5).
    // A SMALL cloth rotation: enough that conj(rigidQ)*clothQ is neither factor, and
    // small enough that the triangles stay near-parallel to the face they rest on.
    const float ch = std::sin(0.05f), cc = std::cos(0.05f);
    const Q clothQ = { ch * 0.6f, ch * 0.8f, 0.0f, cc };
    // Placed so the triangles STRADDLE the box's top face: the box centre is at
    // y = -0.4 with a half-extent of 0.6, so the face sits near y = 0.2, and the
    // cloth's own local y runs +-0.05 about clothP.y.
    // The offset from the box to the cloth must NOT lie along the box's rotation
    // axis.  It did: boxP and clothP shared x and z while boxQ turns about +y, so
    // rotating (0, dy, 0) by that quaternion or by its inverse gives the same vector
    // and `b5-toconv-rotate-forward` was indistinguishable.  Now it has x and z too.
    const V3c clothP = { 0.55f, 0.26f, 0.45f };

    // The RIGID's contribution is the LARGER half on purpose.  With 0.22 + 0.08 the
    // band a mutant closes by dropping it was only 0.08 wide and no triangle sat in
    // it; at 0.22 + 0.50 the band is 0.50 wide and triangle 5 lives squarely inside.
    const float cdCloth = 0.22f, cdRigid = 0.50f;
    const float rest0 = 0.017f, rest1 = 0.041f;             // B6: per contact-manager pair
    const uint32_t materialIndex = 12u;
    const uint64_t rigidNodeInd = 0x0000000700000031ull;
    const uint32_t CLOTH_ID = 1u;

    // A cloth of six triangles straddling the box, plus two the filter will act on.
    std::vector<F4> cverts;
    std::vector<Tri4> ctris;
    auto addTri = [&](V3c a, V3c b, V3c c) {
        const uint32_t base = uint32_t(cverts.size());
        cverts.push_back(F4{ a.x, a.y, a.z, 1.0f });
        cverts.push_back(F4{ b.x, b.y, b.z, 1.0f });
        cverts.push_back(F4{ c.x, c.y, c.z, 1.0f });
        ctris.push_back(Tri4{ base, base + 1u, base + 2u, 0u });
    };
    // The cloth's LOCAL frame; the kernel moves it by clothTransform then into box space.
    // Six triangles laid flat, each LARGER than the box face they rest on.
    //
    // Size matters here and the first draft had it backwards.  The contact stage
    // clips the reference FACE polygon against the triangle, so a triangle that
    // fits inside the face leaves the clipper nothing to cut and emits nothing --
    // measured, three times, before reading how the Phase-1 battery builds its own
    // cases: its triangles run 4.5 to 7 units against a box of half-extent ~1.
    // These span about 6 against a 2.2 x 3.4 face, and each is shifted in z so the
    // six are not the same case six times.
    for (int i = 0; i < 6; ++i)
    {
        const float z = -0.5f + 0.2f * float(i);
        // Triangle 5 is lifted into the band that only the SUMMED contact distance
        // opens: the others sit ~0.06 above the face, this one ~0.26 -- inside
        // 0.22 + 0.08 = 0.30 and outside the cloth's 0.22 alone.  Without it,
        // dropping the rigid's contribution changed nothing.
        const float y = (i == 5) ? 0.34f : 0.0f;   // separation ~0.40: inside 0.72, outside 0.22
        addTri(V3c{ -3.0f, y, z - 3.0f }, V3c{ 3.0f, y, z - 3.0f }, V3c{ 0.0f, y, z + 3.0f });
    }
    const uint32_t NB_TRIS = uint32_t(ctris.size());

    DevBuf bCVerts = mkBuf(dev, cverts);
    DevBuf bCTris = mkBuf(dev, ctris);

    // ---- the hull: a unit box, cooked by hand (the layout pxgSlangConvexHull reads) ----
    // The hull blob comes from THIS file's Hull/makeBox -- the same authored
    // ConvexScratch layout the Phase-1 battery drives, not a second one.  makeBox
    // takes half-extents of 1: the box's real size arrives through the SHAPE SCALE,
    // which is what B2 is about.
    std::vector<uint32_t> blob;
    uint32_t hullOffset = 0;
    makeBox(1.0f, 1.0f, 1.0f).serialize(blob, hullOffset);
    PXG_CHECK_MSG(hullOffset == 0u, "the hull must start at offset 0 for this rig");
    DevBuf bHull = mkBuf(dev, blob);

    PxgFEMCloth fc; std::memset(&fc, 0, sizeof(fc));
    fc.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bCVerts.addr));
    fc.mTriangleVertexIndices = reinterpret_cast<uint4*>(size_t(bCTris.addr));
    fc.mNbVerts = uint32_t(cverts.size());
    fc.mNbTriangles = NB_TRIS;
    PxgFEMCloth decoy; std::memset(&decoy, 0, sizeof(decoy));
    std::vector<PxgFEMCloth> descArr = { decoy, fc };
    DevBuf bDesc = mkBuf(dev, descArr);

    // Two contact-manager pairs over the same cloth and box, differing only in rest
    // distance -- so restDistance[cmIdx] is separable from restDistance[cacheRef].
    std::vector<PxgContactManagerInput> cmInputs(2);
    std::memset(cmInputs.data(), 0, cmInputs.size() * sizeof(PxgContactManagerInput));
    for (uint32_t p = 0; p < 2u; ++p)
    { cmInputs[p].shapeRef0 = 0u; cmInputs[p].shapeRef1 = 1u;
      cmInputs[p].transformCacheRef0 = 0u; cmInputs[p].transformCacheRef1 = 1u; }
    DevBuf bCin = mkBuf(dev, cmInputs);

    std::vector<PxgShape> shapes(2);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShape));
    shapes[0].type = PxU32(PxGeometryType::eTRIANGLEMESH);
    shapes[0].particleOrSoftbodyId = CLOTH_ID;
    shapes[1].type = PxU32(PxGeometryType::eBOX);
    shapes[1].materialIndex = materialIndex;
    // B2: a non-uniform scale in a ROTATED frame on the RIGID only.
    // A non-uniform scale on the RIGID only -- that is what B2 separates.  The scale
    // ROTATION is left identity here: the hull decode it exercises is already pinned
    // by the Phase-1 battery, and tilting the solid only moves the contact band.
    shapes[1].scale = PxMeshScale(PxVec3(boxExt.x, boxExt.y, boxExt.z), PxQuat(PxIdentity));
    shapes[1].hullOrMeshPtr = size_t(bHull.addr);
    DevBuf bShapes = mkBuf(dev, shapes);

    std::vector<PxsCachedTransform> tcache(2);
    std::memset(tcache.data(), 0, tcache.size() * sizeof(PxsCachedTransform));
    tcache[0].transform = PxTransform(PxVec3(clothP.x, clothP.y, clothP.z),
                                      PxQuat(clothQ.x, clothQ.y, clothQ.z, clothQ.w));
    tcache[1].transform = PxTransform(PxVec3(boxP.x, boxP.y, boxP.z),
                                      PxQuat(boxQ.x, boxQ.y, boxQ.z, boxQ.w));
    DevBuf bTc = mkBuf(dev, tcache);

    DevBuf bCd = mkBuf(dev, std::vector<float>{ cdCloth, cdRigid });
    DevBuf bRd = mkBuf(dev, std::vector<float>{ rest0, rest1 });
    DevBuf bRemap = mkBuf(dev, std::vector<uint64_t>{ 0ull, rigidNodeInd });

    // B1: triangle 1 has TWO filtered corners (must still contact); triangle 4 has
    // all THREE (must be skipped).
    const uint32_t TRI_PARTIAL = 1u, TRI_FULL = 4u;
    std::vector<PxgRigidFilterPair> fpairs;
    auto addFilter = [&](uint32_t vert) {
        PxgRigidFilterPair fp; std::memset(&fp, 0, sizeof(fp));
        fp.index0 = rigidNodeInd;
        fp.index1 = uint64_t((CLOTH_ID << 20) | vert);
        fpairs.push_back(fp);
    };
    addFilter(ctris[TRI_PARTIAL].x); addFilter(ctris[TRI_PARTIAL].y);
    addFilter(ctris[TRI_FULL].x); addFilter(ctris[TRI_FULL].y); addFilter(ctris[TRI_FULL].z);
    // find() expects the pair list SORTED by (index0, index1).
    std::sort(fpairs.begin(), fpairs.end(), [](const PxgRigidFilterPair& a, const PxgRigidFilterPair& b) {
        return a.index0 != b.index0 ? a.index0 < b.index0 : a.index1 < b.index1; });
    DevBuf bFilters = mkBuf(dev, fpairs);

    // ---- the midphase pair stack: every triangle, alternating cm pair ----
    std::vector<U4> stack;
    for (uint32_t t = 0; t < NB_TRIS; ++t) stack.push_back(U4{ t & 1u, t, 0u, 0u });
    // B3/B4: the counter OVERSHOOTS the stack.
    const uint32_t OVERSHOOT = 3u;
    const uint32_t stackPairs = uint32_t(stack.size());
    // The buffer is PADDED past stackSizeBytes with valid-but-extra pairs, so a port
    // that consumes the counter unclamped reads real work rather than running off the
    // allocation -- a wrong ANSWER instead of a crash, which is the stronger kill.
    for (uint32_t k = 0; k < OVERSHOOT; ++k) stack.push_back(U4{ 0u, k, 0u, 0u });
    DevBuf bStack = mkBuf(dev, stack);
    DevBuf bPairsNum = mkBuf(dev, std::vector<uint32_t>{ stackPairs + OVERSHOOT });
    DevBuf bNeeded = mkBuf(dev, std::vector<uint32_t>{ 0u });

    const uint32_t CAP = 256u;
    WBufs wa = mkWriter(dev, CAP);

    BTriPush push = {};
    push.toleranceLength = 1.0f;
    push.cmInputs = bCin.addr; push.transformCache = bTc.addr;
    push.contactDistance = bCd.addr; push.restDistance = bRd.addr;
    push.gpuShapes = bShapes.addr; push.femClothes = bDesc.addr;
    push.shapeToRigidRemap = bRemap.addr; push.stackPtr = bStack.addr;
    push.midphasePairsNum = bPairsNum.addr; push.filterPairs = bFilters.addr;
    push.nbFilterPairs = uint32_t(fpairs.size());
    push.stackSizeBytes = stackPairs * 16u;
    push.stackSizeNeeded = bNeeded.addr;
    push.bPoint = wa.point.addr; push.bNormalPen = wa.normalPen.addr;
    push.bBary = wa.bary.addr; push.bInfo = wa.info.addr;
    push.bCount = wa.count.addr; push.bByX = wa.byX.addr;
    push.bTempByX = wa.tempByX.addr; push.bIdxByX = wa.idxByX.addr;
    push.bSortedByX = wa.sortedByX.addr; push.bMaxNumContacts = CAP;

    Kernel ka("femClothContactGenBda.spv", "cloth_boxTriangleContactGenLaunch", {}, sizeof(BTriPush));
    PXG_CHECK_MSG(ka.isValid(), "the cloth box-triangle pipeline must build");
    if (!ka.isValid()) { killWriter(dev, wa); return; }
    ka.dispatch(/*gridX*/ 4096u, {}, push, 1u, 1u);
    dev->flush();

    // ---- the host mirror of the GLUE, and the differential dispatch ----
    std::vector<FctCase> cases;
    uint32_t skipped = 0, partialKept = 0;
    for (uint32_t i = 0; i < stackPairs; ++i)
    {
        const uint32_t cmIdx = stack[i].x, triangleIdx = stack[i].y;
        const Tri4& t = ctris[triangleIdx];
        auto filtered = [&](uint32_t v) {
            const uint64_t key = uint64_t((CLOTH_ID << 20) | v);
            for (const auto& fp : fpairs) if (fp.index0 == rigidNodeInd && fp.index1 == key) return true;
            return false;
        };
        if (filtered(t.x) && filtered(t.y) && filtered(t.z)) { ++skipped; continue; }   // B1
        if (triangleIdx == TRI_PARTIAL) ++partialKept;

        const Q toConvQ = qmul(qconj(boxQ), clothQ);                                   // B5
        const V3c toConvP = qrotInv(boxQ, V3c{ clothP.x - boxP.x, clothP.y - boxP.y, clothP.z - boxP.z });
        auto toConv = [&](const F4& v) {
            const V3c r = qrot(toConvQ, V3c{ v.x, v.y, v.z });
            return V3c{ r.x + toConvP.x, r.y + toConvP.y, r.z + toConvP.z };
        };
        const V3c a = toConv(cverts[t.x]), b = toConv(cverts[t.y]), c = toConv(cverts[t.z]);
        const V3c e0 = { b.x - a.x, b.y - a.y, b.z - a.z };
        const V3c e1 = { c.x - a.x, c.y - a.y, c.z - a.z };
        V3c n = { e0.y * e1.z - e0.z * e1.y, e0.z * e1.x - e0.x * e1.z, e0.x * e1.y - e0.y * e1.x };
        // P9.47: the engine takes this triangle normal with PxVec3::getNormalized()
        // (PxVec3.h:286-290) -- m = magnitudeSquared(), then v * PxRecipSqrt(m), i.e.
        // ONE reciprocal of an exactly specified square root and three multiplies,
        // with the zero vector for a degenerate triangle.  This mirror used to divide
        // each component by the magnitude separately, which agreed with the shader
        // only while the shader spelled the normalisation as the GLSL.std.450
        // Normalize INTRINSIC -- two wrong readings that happened to match.  It is
        // also the P9.45 componentwise-divide class, on the host side where
        // gpuslang.componentDivide cannot see it.
        {
            const float nm = n.x * n.x + n.y * n.y + n.z * n.z;
            if (nm > 0.0f)
            {
                const float r = 1.0f / std::sqrt(nm);
                n = V3c{ n.x * r, n.y * r, n.z * r };
            }
            else
            {
                n = V3c{ 0.0f, 0.0f, 0.0f };
            }
        }

        FctCase cs{};
        cs.convexScale = F4{ boxExt.x, boxExt.y, boxExt.z, cdCloth + cdRigid };
        const PxQuat sr = shapes[1].scale.rotation;
        cs.convexScaleRot = F4{ sr.x, sr.y, sr.z, sr.w };
        cs.t0 = F4{ a.x, a.y, a.z, (cmIdx == 0u) ? rest0 : rest1 };                     // B6
        cs.t1 = F4{ b.x, b.y, b.z, 0.0f };
        cs.t2 = F4{ c.x, c.y, c.z, 0.0f };
        cs.triNormal = F4{ n.x, n.y, n.z, 0.0f };
        cs.convexP = F4{ boxP.x, boxP.y, boxP.z, 0.0f };
        cs.convexQ = F4{ boxQ.x, boxQ.y, boxQ.z, boxQ.w };
        cs.ids.x = uint32_t(rigidNodeInd & 0xffffffffu);
        cs.ids.y = uint32_t(rigidNodeInd >> 32);
        cs.ids.z = CLOTH_ID; cs.ids.w = triangleIdx;
        cs.ids2.x = materialIndex; cs.ids2.y = 0u;
        cases.push_back(cs);
    }
    PXG_CHECK_MSG(skipped == 1u, "B1: exactly one triangle must be skipped by the AND filter, "
                  "%u were", skipped);
    PXG_CHECK_MSG(partialKept == 1u, "B1: the two-corner-filtered triangle must be KEPT, or an "
                  "`||` spelling is invisible");

    DevBuf bCases = mkBuf(dev, cases);
    WBufs wb = mkWriter(dev, CAP);
    FctParams fp2 = {};
    fp2.numCases = uint32_t(cases.size()); fp2.maxContacts = CAP;
    fp2.hullAddr = bHull.addr;
    fp2.wPoint = wb.point.addr; fp2.wNormalPen = wb.normalPen.addr; fp2.wBary = wb.bary.addr;
    fp2.wInfo = wb.info.addr; fp2.wCount = wb.count.addr; fp2.wByX = wb.byX.addr;
    fp2.wTempByX = wb.tempByX.addr; fp2.wIdxByX = wb.idxByX.addr;
    std::vector<U4> dbg(cases.size(), U4{ 7, 7, 7, 7 });
    DevBuf bDbg = mkBuf(dev, dbg);
    Kernel kb("fcConvexTri.spv", "ctFcConvexTri",
              { (uint32_t)sizeof(FctCase), 16 }, sizeof(FctParams));
    PXG_CHECK_MSG(kb.isValid(), "the reference core pipeline must build");
    if (kb.isValid())
    {
        kb.dispatch((uint32_t(cases.size()) + 63u) / 64u, { bCases.h, bDbg.h }, fp2);
        dev->flush();
    }

    // ---- compare the two contact streams as multisets ----
    const uint32_t cntA = downloadBuffer<uint32_t>(dev, wa.count.h, 1)[0];
    const uint32_t cntB = downloadBuffer<uint32_t>(dev, wb.count.h, 1)[0];
    PXG_CHECK_MSG(cntA == cntB,
                  "the kernel emitted %u contacts, the reference core over the mirrored glue "
                  "emitted %u", cntA, cntB);
    PXG_CHECK_MSG(cntA > 0u, "no contacts at all -- the rig measures nothing");

    if (cntA == cntB && cntA > 0u)
    {
        const uint32_t n = std::min(cntA, CAP);
        const auto pA = downloadBuffer<F4>(dev, wa.point.h, n);
        const auto nA = downloadBuffer<F4>(dev, wa.normalPen.h, n);
        const auto yA = downloadBuffer<F4>(dev, wa.bary.h, n);
        const auto iA = downloadBuffer<FemInfoHost>(dev, wa.info.h, n);
        const auto pB = downloadBuffer<F4>(dev, wb.point.h, n);
        const auto nB = downloadBuffer<F4>(dev, wb.normalPen.h, n);
        const auto yB = downloadBuffer<F4>(dev, wb.bary.h, n);
        const auto iB = downloadBuffer<FemInfoHost>(dev, wb.info.h, n);
        std::vector<bool> used(n, false);
        size_t mm = 0;
        for (uint32_t a = 0; a < n; ++a)
        {
            bool found = false;
            for (uint32_t b = 0; b < n && !found; ++b)
            {
                if (used[b]) continue;
                if (f2u(pA[a].x) != f2u(pB[b].x) || f2u(pA[a].y) != f2u(pB[b].y) ||
                    f2u(pA[a].z) != f2u(pB[b].z) || f2u(pA[a].w) != f2u(pB[b].w)) continue;
                if (f2u(nA[a].x) != f2u(nB[b].x) || f2u(nA[a].y) != f2u(nB[b].y) ||
                    f2u(nA[a].z) != f2u(nB[b].z) || f2u(nA[a].w) != f2u(nB[b].w)) continue;
                if (f2u(yA[a].x) != f2u(yB[b].x) || f2u(yA[a].y) != f2u(yB[b].y) ||
                    f2u(yA[a].z) != f2u(yB[b].z)) continue;
                if (iA[a].pairInd0 != iB[b].pairInd0 || iA[a].pairInd1 != iB[b].pairInd1 ||
                    iA[a].rigidMatInd != iB[b].rigidMatInd) continue;
                used[b] = true; found = true;
            }
            if (!found)
            { if (mm < 4) std::printf("  btri contact %u (pt %g,%g,%g w %g / np %g,%g,%g,%g / "
                                      "pair1 %u) has no match in the reference stream\n",
                                      a, pA[a].x, pA[a].y, pA[a].z, pA[a].w,
                                      nA[a].x, nA[a].y, nA[a].z, nA[a].w, iA[a].pairInd1);
              ++mm; }
        }
        PXG_CHECK_MSG(mm == 0, "cloth_boxTriangleContactGen differs from the reference core over "
                      "the mirrored glue in %zu contacts", mm);
    }

    // B4: the atomicMax carries the UNCLAMPED count, in bytes.
    const uint32_t needed = downloadBuffer<uint32_t>(dev, bNeeded.h, 1)[0];
    PXG_CHECK_MSG(needed == (stackPairs + OVERSHOOT) * 16u,
                  "stackSizeNeeded reads %u, expected %u -- it must carry the UNCLAMPED pair "
                  "count times 16, which is how the engine grows next frame's stack",
                  needed, (stackPairs + OVERSHOOT) * 16u);

    std::printf("    [ note ] %u contacts over %u triangles / %u pairs (1 skipped by the AND "
                "filter, 1 kept with two corners filtered); stackNeeded %u vs stack %u\n",
                cntA, NB_TRIS, stackPairs, needed, stackPairs * 16u);

    killWriter(dev, wa); killWriter(dev, wb);
    dev->destroyBuffer(bCVerts.h); dev->destroyBuffer(bCTris.h); dev->destroyBuffer(bHull.h);
    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bCin.h); dev->destroyBuffer(bShapes.h);
    dev->destroyBuffer(bTc.h); dev->destroyBuffer(bCd.h); dev->destroyBuffer(bRd.h);
    dev->destroyBuffer(bRemap.h); dev->destroyBuffer(bFilters.h); dev->destroyBuffer(bStack.h);
    dev->destroyBuffer(bPairsNum.h); dev->destroyBuffer(bNeeded.h);
    dev->destroyBuffer(bCases.h); dev->destroyBuffer(bDbg.h);
}

PXG_TEST(fcConvexTri_contactGen)
{
    std::mt19937 rng(96031);
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<uint32_t> blob;
    uint32_t boxOfs, prismOfs;
    const Hull box = makeBox(0.8f, 0.6f, 1.0f);
    const float prismRadii[6] = { 0.92f, 0.78f, 1.02f, 0.85f, 0.97f, 0.72f };
    const Hull prism = makePrism(6, prismRadii, 0.7f);
    float bigRadii[24];
    for (int k = 0; k < 24; ++k)
        bigRadii[k] = 0.85f + 0.18f * std::sin(k * 1.7f) * std::cos(k * 0.9f);
    const Hull bigPrism = makePrism(24, bigRadii, 0.6f); // 72 edges: the
    // per-lane strided reduction loops take 2-3 steps like production hulls
    uint32_t bigOfs;
    box.serialize(blob, boxOfs);
    prism.serialize(blob, prismOfs);
    bigPrism.serialize(blob, bigOfs);

    const uint32_t NC = 1200, CAP = 8192, POISON = 0xeeeeeeeeu, INFOPOISON = 0xdeadbeefu;
    const uint32_t CLOTH = 17;
    std::vector<FctCase> cases(NC);
    std::vector<std::vector<RefContact>> refCt(NC);
    std::vector<U4> refDbg(NC);
    std::vector<uint8_t> taint(NC, 0);
    size_t satFail = 0, triFeat = 0, faceFeat = 0, twoFace = 0, reduced = 0, reducedClean = 0, baryRej = 0, edgeBranchClean = 0, poly2Clean = 0;

    for (uint32_t i = 0; i < NC; ++i)
    {
        const bool useBox = (i & 1) == 0;
        const bool useBig = !useBox && (i % 18 == 5); // avoids the prism-indexed families
        const Hull& H = useBox ? box : (useBig ? bigPrism : prism);
        FctCase& cs = cases[i];
        const V3 scale = { rf(0.7f, 1.4f), rf(0.7f, 1.4f), rf(0.7f, 1.4f) };
        F4 srot = { 0, 0, 0, 1 };
        if (i % 5 == 2)
        {
            srot = { rf(-1, 1), rf(-1, 1), rf(-1, 1), rf(-1, 1) };
            const float m = std::sqrt(srot.x * srot.x + srot.y * srot.y + srot.z * srot.z + srot.w * srot.w);
            srot = { srot.x / m, srot.y / m, srot.z / m, srot.w / m };
        }
        const float cd = rf(0.05f, 0.3f), rest = rf(0.0f, 0.1f);

        V3 t[3];
        const V3 dir = normalize3({ rf(-1, 1), rf(-1, 1), rf(-1, 1) });
        V3 nT;
        if (i % 6 == 4)
        {
            // vertex-vs-triangle-face family: triangle centered beyond the
            // hull along dir, facing back — the TRIANGLE-normal SAT axis wins
            const V3 u = normalize3(crossp(dir, std::fabs(dir.x) < 0.7f ? V3{ 1, 0, 0 } : V3{ 0, 1, 0 }));
            const V3 v = crossp(dir, u);
            const V3 center = mul(dir, rf(1.0f, 1.6f));
            const float tr = rf(0.8f, 1.8f);
            t[0] = add(center, mul(u, tr));
            t[1] = add(center, add(mul(u, -0.5f * tr), mul(v, 0.87f * tr)));
            t[2] = add(center, add(mul(u, -0.5f * tr), mul(v, -0.87f * tr)));
            nT = getNorm0(crossp(sub(t[1], t[0]), sub(t[2], t[0])));
            if (dot3(nT, dir) > 0.0f) { std::swap(t[1], t[2]); nT = mul(nT, -1.0f); }
        }
        else if (i % 12 == 7) // prism-only: small triangle over a TOP VERTEX
        {                     // facing down — TRI feature with minNormal
                              // nearly the top-face normal, so no edge beats
                              // the face alignment (poly2 else-branch)
            const V3 pv = vertex2Shape(H.verts[6 + rng() % 6], scale, srot);
            V3 fnTop = normalize3(v2sNormal({ 0, 0, 1 }, scale, srot));
            // tilt TOWARD the corner's outward direction so the hovered
            // vertex is the hull support along the triangle normal (no
            // far-side dip: the TRI axis wins while the face alignment
            // stays high -> no edge beats it -> poly2 else-branch)
            V3 fc2 = { 0, 0, 0 };
            for (uint32_t kk = 6; kk < 12; ++kk)
                fc2 = add(fc2, vertex2Shape(H.verts[kk], scale, srot));
            fc2 = mul(fc2, 1.0f / 6.0f);
            V3 outw = sub(pv, fc2);
            outw = sub(outw, mul(fnTop, dot3(outw, fnTop)));
            outw = normalize3(outw);
            const V3 center = add(pv, mul(fnTop, rf(0.05f, 0.2f)));
            V3 down = normalize3(add(fnTop, mul(outw, rf(0.1f, 0.18f))));
            const V3 u = normalize3(crossp(down, std::fabs(down.x) < 0.7f ? V3{ 1, 0, 0 } : V3{ 0, 1, 0 }));
            const V3 v = crossp(down, u);
            const float tr = rf(0.3f, 0.55f);
            t[0] = add(center, mul(u, tr));
            t[1] = add(center, add(mul(u, -0.5f * tr), mul(v, 0.87f * tr)));
            t[2] = add(center, add(mul(u, -0.5f * tr), mul(v, -0.87f * tr)));
            nT = getNorm0(crossp(sub(t[1], t[0]), sub(t[2], t[0])));
            if (dot3(nT, down) > 0.0f) { std::swap(t[1], t[2]); nT = mul(nT, -1.0f); }
        }
        else if (i % 6 == 1) // prism-only (odd i): huge triangle covering the
        {                    // top face -> 6 clip candidates -> contactReduce
            V3 fc = { 0, 0, 0 };
            for (uint32_t kk = 6; kk < 12; ++kk)
                fc = add(fc, vertex2Shape(H.verts[kk], scale, srot));
            fc = mul(fc, 1.0f / 6.0f);
            V3 fn = normalize3(v2sNormal({ 0, 0, 1 }, scale, srot));
            const V3 center = add(fc, mul(fn, rf(0.02f, 0.2f)));
            // TILT the plane so the six covered verts get DISTINCT separations
            // (a parallel cover ties every reduce anchor and taints the case)
            // small tilts keep the face alignment high enough that no edge
            // beats it (the poly2-neighbor else-branch); large tilts hand the
            // selection to the edge branch
            const float tiltMag = (i % 12 == 1) ? 0.12f : 0.35f;
            fn = normalize3(add(fn, { rf(-tiltMag, tiltMag), rf(-tiltMag, tiltMag), rf(-tiltMag, tiltMag) }));
            const V3 u = normalize3(crossp(fn, std::fabs(fn.x) < 0.7f ? V3{ 1, 0, 0 } : V3{ 0, 1, 0 }));
            const V3 v = crossp(fn, u);
            const float tr = rf(4.5f, 7.0f);
            t[0] = add(center, mul(u, tr));
            t[1] = add(center, add(mul(u, -0.5f * tr), mul(v, 0.87f * tr)));
            t[2] = add(center, add(mul(u, -0.5f * tr), mul(v, -0.87f * tr)));
            nT = getNorm0(crossp(sub(t[1], t[0]), sub(t[2], t[0])));
        }
        else
        {
            const V3 anchor = mul(dir, rf(0.2f, 1.7f));
            const float ts = (i % 7 == 3 || i % 7 == 5) ? rf(2.0f, 3.5f) : rf(0.4f, 1.4f);
            for (int k = 0; k < 3; ++k)
                t[k] = add(anchor, { rf(-ts, ts), rf(-ts, ts), rf(-ts, ts) });
            nT = getNorm0(crossp(sub(t[1], t[0]), sub(t[2], t[0])));
        }

        cs.convexScale = { scale.x, scale.y, scale.z, cd };
        cs.convexScaleRot = srot;
        cs.t0 = { t[0].x, t[0].y, t[0].z, rest };
        cs.t1 = { t[1].x, t[1].y, t[1].z, 0 };
        cs.t2 = { t[2].x, t[2].y, t[2].z, 0 };
        cs.triNormal = { nT.x, nT.y, nT.z, 0 };
        cs.convexP = { rf(-1, 1), rf(-1, 1), rf(-1, 1), 0 };
        {
            F4 q = { rf(-1, 1), rf(-1, 1), rf(-1, 1), rf(-1, 1) };
            const float m = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
            cs.convexQ = { q.x / m, q.y / m, q.z / m, q.w / m };
        }
        cs.ids = { 0xA0u + i % 7, 3u + i % 3, CLOTH, i };
        cs.ids2 = { (i % 4 == 1) ? (0x80000000u | (i & 0xfffu)) : (i & 0xfffu),
                    useBox ? boxOfs : (useBig ? bigOfs : prismOfs), 0, 0 };

        // ---- reference ----
        Slack sl;
        bool hadReduce = false, hadEdgeBranch = false, hadPoly2 = false;
        const uint32_t nbV = (uint32_t)H.verts.size(), nbP = (uint32_t)H.faces.size(), nbE = (uint32_t)H.edgeFaces.size();
        std::vector<V3> scaledVerts(nbV), planeN(nbP);
        for (uint32_t v = 0; v < nbV; ++v) scaledVerts[v] = vertex2Shape(H.verts[v], scale, srot);

        float separation = -MAXF;
        V3 minNormal = { MAXF, MAXF, MAXF };
        uint32_t featureIndex = 0xffffffffu;
        bool satPassed = true;

        // satConvexFaceNormals (strided lanes; nbP <= 32 so lane == index)
        {
            float laneSep[32];
            uint32_t laneFeat[32];
            V3 laneNrm[32];
            for (int l = 0; l < 32; ++l) { laneSep[l] = -MAXF; laneFeat[l] = 0; laneNrm[l] = minNormal; }
            bool disjoint = false;
            for (uint32_t p = 0; p < nbP; ++p)
            {
                V3 n = v2sNormal({ H.planes[p].x, H.planes[p].y, H.planes[p].z }, scale, srot);
                const float mag = std::sqrt(dot3(n, n));
                const float invMag = 1.0f / mag;
                n = mul(n, invMag);
                uint32_t minIdx = 0;
                float bestd = MAXF;
                for (uint32_t v = 0; v < nbV; ++v)
                {
                    const float d = dot3({ H.planes[p].x, H.planes[p].y, H.planes[p].z }, H.verts[v]);
                    if (d < bestd) { bestd = d; minIdx = v; }
                }
                const float min0 = dot3(n, scaledVerts[minIdx]);
                const float max0 = -H.planes[p].w * invMag;
                const float d0 = dot3(t[0], n), d1 = dot3(t[1], n), d2 = dot3(t[2], n);
                const float max1 = std::max(d0, std::max(d1, d2));
                const float min1 = std::min(d0, std::min(d1, d2));
                sl.noteSc(min1, max0 + cd, 20.0f, 8);
                sl.noteSc(min0, max1 + cd, 20.0f, 8);
                if ((min1 > max0 + cd) || (min0 > max1 + cd)) { disjoint = true; break; }
                planeN[p] = n;
                const uint32_t lane = p % 32;
                const float tempSep = min1 - max0;
                if (laneSep[lane] < tempSep) { laneSep[lane] = tempSep; laneFeat[lane] = p; laneNrm[lane] = n; }
            }
            if (disjoint)
                satPassed = false;
            else
            {
                float best = -MAXF;
                for (int l = 0; l < 32; ++l) best = std::max(best, laneSep[l]);
                uint32_t w = 0;
                for (int l = 0; l < 32; ++l)
                    if (laneSep[l] == best) { w = (uint32_t)l; break; }
                for (int l = 0; l < 32; ++l)
                    if ((uint32_t)l != w && laneSep[l] > -MAXF) sl.noteSc(laneSep[l], best, 20.0f, 9);
                if (separation < best)
                {
                    separation = best;
                    featureIndex = laneFeat[w];
                    minNormal = laneNrm[w];
                }
            }
        }
        // satTriangleNormal
        if (satPassed)
        {
            const float triProj = dot3(nT, t[0]);
            float min1 = MAXF, max1 = -MAXF;
            for (uint32_t v = 0; v < nbV; ++v)
            {
                const float d = dot3(scaledVerts[v], nT);
                max1 = std::max(d, max1);
                min1 = std::min(d, min1);
            }
            sl.noteSc(min1, triProj + cd, 20.0f, 10);
            sl.noteSc(triProj, max1 + cd, 20.0f, 10);
            if ((min1 > triProj + cd) || (triProj > max1 + cd))
                satPassed = false;
            else
            {
                const float s2 = min1 - triProj;
                sl.noteSc(s2, separation, 20.0f, 11);
                if (separation < s2)
                {
                    minNormal = mul(nT, -1.0f);
                    separation = s2;
                    featureIndex = TRI_FEAT;
                }
            }
        }

        uint32_t face1 = featureIndex, face2 = 0xFFFFFFFFu;
        if (satPassed && featureIndex == TRI_FEAT)
        {
            float laneProj[32];
            uint32_t laneFace[32];
            for (int l = 0; l < 32; ++l) { laneProj[l] = -MAXF; laneFace[l] = 0; }
            for (uint32_t p = 0; p < nbP; ++p)
            {
                const uint32_t lane = p % 32;
                const float proj = dot3(minNormal, planeN[p]);
                if (laneProj[lane] < proj) { laneProj[lane] = proj; laneFace[lane] = p; }
            }
            float minProj = -MAXF;
            for (int l = 0; l < 32; ++l) minProj = std::max(minProj, laneProj[l]);
            uint32_t w = 0;
            for (int l = 0; l < 32; ++l)
                if (laneProj[l] == minProj) { w = (uint32_t)l; break; }
            for (int l = 0; l < 32; ++l)
                if ((uint32_t)l != w && laneProj[l] > -MAXF) sl.noteSc(laneProj[l], minProj, 4.0f, 12);
            uint32_t polyFace = laneFace[w];

            float laneDpSq[32], laneDpSq2[32];
            uint32_t laneEdge[32], lanePoly2[32];
            for (int l = 0; l < 32; ++l)
            {
                laneDpSq[l] = minProj * minProj;
                laneEdge[l] = 0xFFFFFFFFu;
                laneDpSq2[l] = -10.0f;
                lanePoly2[l] = 0xFFFFFFFFu;
            }
            for (uint32_t e = 0; e < nbE; ++e)
            {
                const uint32_t lane = e % 32;
                const uint32_t f0 = H.edgeFaces[e].first, f1 = H.edgeFaces[e].second;
                const V3 edgeN = add(planeN[f0], planeN[f1]);
                const float enMagSq = dot3(edgeN, edgeN);
                const float dp = dot3(edgeN, minNormal);
                const float sqDp = dp * dp;
                sl.noteSc(dp, 0.0f, 4.0f, 13);
                if ((f0 == polyFace || f1 == polyFace) && sqDp > laneDpSq2[lane])
                {
                    laneDpSq2[lane] = sqDp;
                    lanePoly2[lane] = f0 == polyFace ? f1 : f0;
                }
                sl.noteSc(sqDp, laneDpSq[lane] * enMagSq, 8.0f, 14);
                if (dp > 0.0f && sqDp > laneDpSq[lane] * enMagSq)
                {
                    laneDpSq[lane] = sqDp / enMagSq;
                    laneEdge[lane] = e;
                }
            }
            float maxDpSq = -MAXF;
            for (int l = 0; l < 32; ++l) maxDpSq = std::max(maxDpSq, laneDpSq[l]);
            uint32_t w2 = 0;
            for (int l = 0; l < 32; ++l)
                if (laneDpSq[l] == maxDpSq) { w2 = (uint32_t)l; break; }
            // seed-vs-seed ties resolve identically on both sides; only a REAL
            // edge candidate near the winner is outcome-unstable
            for (int l = 0; l < 32; ++l)
                if (laneEdge[l] != 0xFFFFFFFFu && (uint32_t)l != w2) sl.noteSc(laneDpSq[l], maxDpSq, 8.0f, 15);
            if (laneEdge[w2] == 0xFFFFFFFFu)
                for (int l = 0; l < 32; ++l)
                    if (laneEdge[l] != 0xFFFFFFFFu) sl.noteSc(laneDpSq[l], maxDpSq, 8.0f, 15);
            const uint32_t closestEdge = laneEdge[w2];
            if (closestEdge != 0xFFFFFFFFu)
            {
                hadEdgeBranch = true;
                const uint32_t f0 = H.edgeFaces[closestEdge].first, f1 = H.edgeFaces[closestEdge].second;
                const float dp0 = dot3(planeN[f0], minNormal);
                const float dp1 = dot3(planeN[f1], minNormal);
                sl.noteSc(dp0, dp1, 4.0f, 16);
                if (dp0 > dp1) { polyFace = f0; face2 = f1; }
                else           { polyFace = f1; face2 = f0; }
            }
            else
            {
                float maxDpSq2 = -MAXF;
                for (int l = 0; l < 32; ++l) maxDpSq2 = std::max(maxDpSq2, laneDpSq2[l]);
                uint32_t w3 = 0;
                for (int l = 0; l < 32; ++l)
                    if (laneDpSq2[l] == maxDpSq2) { w3 = (uint32_t)l; break; }
                for (int l = 0; l < 32; ++l)
                    if (laneDpSq2[l] > -10.0f && (uint32_t)l != w3) sl.noteSc(laneDpSq2[l], maxDpSq2, 8.0f, 17);
                face2 = lanePoly2[w3];
                hadPoly2 = true;
            }
            face1 = polyFace;
        }

        refDbg[i] = { satPassed ? 1u : 0u, featureIndex, face1, face2 };

        if (satPassed)
        {
            const F4 triPlane = { nT.x, nT.y, nT.z, -dot3(t[0], nT) };
            uint32_t fi2 = face2;
            for (uint32_t face = face1; face != 0xFFFFFFFFu;)
            {
                V3 p0n = v2sNormal({ H.planes[face].x, H.planes[face].y, H.planes[face].z }, scale, srot);
                float p0d = H.planes[face].w;
                const float dn = 1.0f / std::sqrt(dot3(p0n, p0n));
                p0n = mul(p0n, dn);
                p0d *= dn;
                const F4 plane0 = { p0n.x, p0n.y, p0n.z, p0d };
                std::vector<V3> pverts;
                for (uint32_t idx : H.faces[face])
                    pverts.push_back(vertex2Shape(H.verts[idx], scale, srot));

                float ct[NUMTMP * 8] = {};
                uint32_t num = refPolyClip(pverts, (uint32_t)pverts.size(), minNormal, plane0, triPlane, t, cd, ct, sl);
                if (num)
                {
                    V3 pa[32] = {};
                    float sep[32] = {};
                    F4 bary[32] = {};
                    uint32_t validMask = 0;
                    for (uint32_t lane = 0; lane < 32 && lane < num; ++lane)
                    {
                        pa[lane] = { ct[8 * lane], ct[8 * lane + 1], ct[8 * lane + 2] };
                        sep[lane] = ct[8 * lane + 3];
                        const V3 e0 = sub(t[1], t[0]), e1 = sub(t[2], t[0]), e2 = sub(pa[lane], t[0]);
                        const float d00 = dot3(e0, e0), d01 = dot3(e0, e1), d11 = dot3(e1, e1);
                        const float d20 = dot3(e2, e0), d21 = dot3(e2, e1);
                        const float dnm = d00 * d11 - d01 * d01;
                        const float bv = (d11 * d20 - d01 * d21) / dnm;
                        const float bw = (d00 * d21 - d01 * d20) / dnm;
                        const float bu = 1.0f - bv - bw;
                        bary[lane] = { bu, bv, bw, 0 };
                        sl.noteSc(bu, -1e-4f, 4.0f, 18); sl.noteSc(bu, 1.0f + 1e-4f, 4.0f, 18);
                        sl.noteSc(bv, -1e-4f, 4.0f, 18); sl.noteSc(bv, 1.0f + 1e-4f, 4.0f, 18);
                        sl.noteSc(bw, -1e-4f, 4.0f, 18); sl.noteSc(bw, 1.0f + 1e-4f, 4.0f, 18);
                        const bool ok = bu >= -1e-4f && bu <= 1.0f + 1e-4f && bv >= -1e-4f && bv <= 1.0f + 1e-4f &&
                                        bw >= -1e-4f && bw <= 1.0f + 1e-4f;
                        if (ok) validMask |= 1u << lane;
                        else ++baryRej;
                    }
                    const uint32_t ng = (uint32_t)__builtin_popcount(validMask);
                    if (ng)
                    {
                        uint32_t mask = validMask;
                        if (ng > 5)
                        {
                            ++reduced;
                            hadReduce = true;
                            V3 point[32];
                            for (int j = 0; j < 32; ++j)
                                point[j] = sub(pa[j], mul(minNormal, dot3(pa[j], minNormal)));
                            uint32_t dupe = 0;
                            for (int j = 0; j < 32; ++j)
                            {
                                if (!((mask >> j) & 1)) continue;
                                for (int k = 0; k < 32; ++k)
                                {
                                    if (!((mask >> k) & 1)) continue;
                                    const V3 dv = sub(pa[k], pa[j]);
                                    const float dd = dot3(dv, dv);
                                    if (k != j) sl.noteSc(dd, 1e-8f, 0.01f, 19); // self-pair exact both sides
                                    if (dd < 1e-8f && (sep[k] < sep[j] || (sep[k] == sep[j] && k > j)))
                                        { dupe |= 1u << j; break; }
                                }
                            }
                            mask &= ~dupe;
                            if ((uint32_t)__builtin_popcount(mask) > 5)
                            {
                                auto argMaxN = [&](const float* vals, uint32_t m) {
                                    float best = -MAXF; uint32_t bi = 0;
                                    for (uint32_t j = 0; j < 32; ++j)
                                        if (((m >> j) & 1) && vals[j] > best) { best = vals[j]; bi = j; }
                                    for (uint32_t j = 0; j < 32; ++j)
                                        if (((m >> j) & 1) && j != bi) sl.noteSc(vals[j], best, 8.0f, 20);
                                    return bi;
                                };
                                auto argMinN = [&](const float* vals, uint32_t m) {
                                    float best = MAXF; uint32_t bi = 0;
                                    for (uint32_t j = 0; j < 32; ++j)
                                        if (((m >> j) & 1) && vals[j] < best) { best = vals[j]; bi = j; }
                                    for (uint32_t j = 0; j < 32; ++j)
                                        if (((m >> j) & 1) && j != bi) sl.noteSc(vals[j], best, 8.0f, 20);
                                    return bi;
                                };
                                const uint32_t allM = mask;
                                const uint32_t i0 = argMaxN(sep, allM);
                                uint32_t m2 = 1u << i0;
                                float vals[32];
                                for (int j = 0; j < 32; ++j)
                                { const V3 dv = sub(point[i0], point[j]); vals[j] = std::sqrt(dot3(dv, dv)); }
                                const uint32_t i1 = argMaxN(vals, allM & ~m2);
                                m2 |= 1u << i1;
                                const V3 dirv = crossp(minNormal, sub(point[i1], point[i0]));
                                float dval[32];
                                for (int j = 0; j < 32; ++j) dval[j] = dot3(dirv, sub(point[j], point[i0]));
                                m2 |= 1u << argMaxN(dval, allM & ~m2);
                                m2 |= 1u << argMinN(dval, allM & ~m2);
                                if (__builtin_popcount(m2) == 4)
                                    m2 |= 1u << argMinN(sep, allM & ~m2);
                                for (uint32_t c = (uint32_t)__builtin_popcount(m2); c < 5; ++c)
                                    m2 |= 1u << argMinN(sep, allM & ~m2);
                                mask = m2;
                            }
                        }
                        if ((int32_t)mask > 0)
                        {
                            for (uint32_t lane = 0; lane < 32; ++lane)
                            {
                                if (!((mask >> lane) & 1)) continue;
                                RefContact rc;
                                rc.p = add(quatRotate(cs.convexQ, pa[lane]), v3f(cs.convexP));
                                rc.n = quatRotate(cs.convexQ, { ct[8 * lane + 4], ct[8 * lane + 5], ct[8 * lane + 6] });
                                rc.sep = sep[lane];
                                rc.bary = bary[lane];
                                const float B = 3e-3f;
                                rc.firm = bary[lane].x > -1e-4f + B && bary[lane].x < 1.0001f - B &&
                                          bary[lane].y > -1e-4f + B && bary[lane].y < 1.0001f - B &&
                                          bary[lane].z > -1e-4f + B && bary[lane].z < 1.0001f - B;
                                refCt[i].push_back(rc);
                            }
                        }
                        else if (mask != 0)
                            sl.near = true; // signed-gate quirk: reservation hole
                    }
                }
                face = fi2;
                fi2 = 0xFFFFFFFFu;
            }
        }

        taint[i] = sl.near ? 1 : 0;
        if (!taint[i] && hadReduce) ++reducedClean;
        if (!taint[i] && hadEdgeBranch) ++edgeBranchClean;
        if (!taint[i] && hadPoly2) ++poly2Clean;
        if (!taint[i])
        {
            if (!satPassed) ++satFail;
            else if (featureIndex == TRI_FEAT) ++triFeat;
            else ++faceFeat;
            if (satPassed && face2 != 0xFFFFFFFFu) ++twoFace;
        }
    }

    // ---- GPU ----
    PxgSlangComputeDevice* dev = testDevice();
    std::vector<F4> femF4Init(CAP, { -9, -9, -9, -9 });
    std::vector<U4> infoInit(CAP, { INFOPOISON, INFOPOISON, INFOPOISON, INFOPOISON });
    std::vector<uint32_t> u1Init(CAP, POISON), u2Init(CAP * 2, POISON), wc0(1, 0);
    std::vector<U4> dbgInit(NC, { 7, 7, 7, 7 });
    PxgBufferHandle bC = createBufferFrom(dev, cases);
    PxgBufferHandle bH = createBufferFrom(dev, blob);
    PxgBufferHandle bPt = createBufferFrom(dev, femF4Init);
    PxgBufferHandle bNp = createBufferFrom(dev, femF4Init);
    PxgBufferHandle bBc = createBufferFrom(dev, femF4Init);
    PxgBufferHandle bIn = createBufferFrom(dev, infoInit);
    PxgBufferHandle bWc = createBufferFrom(dev, wc0);
    PxgBufferHandle bBy = createBufferFrom(dev, u2Init);
    PxgBufferHandle bTb = createBufferFrom(dev, u1Init);
    PxgBufferHandle bIx = createBufferFrom(dev, u1Init);
    PxgBufferHandle bDb = createBufferFrom(dev, dbgInit);
    FctParams prm = {};
    prm.numCases = NC; prm.maxContacts = CAP;
    prm.hullAddr = dev->getBufferDeviceAddress(bH);
    prm.wPoint = dev->getBufferDeviceAddress(bPt);
    prm.wNormalPen = dev->getBufferDeviceAddress(bNp);
    prm.wBary = dev->getBufferDeviceAddress(bBc);
    prm.wInfo = dev->getBufferDeviceAddress(bIn);
    prm.wCount = dev->getBufferDeviceAddress(bWc);
    prm.wByX = dev->getBufferDeviceAddress(bBy);
    prm.wTempByX = dev->getBufferDeviceAddress(bTb);
    prm.wIdxByX = dev->getBufferDeviceAddress(bIx);
    Kernel k("fcConvexTri.spv", "ctFcConvexTri",
             { (uint32_t)sizeof(FctCase), 16 }, sizeof(FctParams));
    k.dispatch((NC + 63) / 64, { bC, bDb }, prm);
    dev->flush();
    const auto gDbg = downloadBuffer<U4>(dev, bDb, NC);
    const auto gPt = downloadBuffer<F4>(dev, bPt, CAP);
    const auto gNp = downloadBuffer<F4>(dev, bNp, CAP);
    const auto gBc = downloadBuffer<F4>(dev, bBc, CAP);
    const auto gIn = downloadBuffer<U4>(dev, bIn, CAP);
    const auto gBy = downloadBuffer<uint32_t>(dev, bBy, CAP * 2);
    const auto gTb = downloadBuffer<uint32_t>(dev, bTb, CAP);
    const auto gIx = downloadBuffer<uint32_t>(dev, bIx, CAP);
    const uint32_t gWc = downloadBuffer<uint32_t>(dev, bWc, 1)[0];

    // debug comparison (SAT verdict, feature, faces)
    size_t mmDbg = 0, checked = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        if (taint[i]) continue;
        ++checked;
        if (gDbg[i].x != refDbg[i].x || gDbg[i].y != refDbg[i].y ||
            gDbg[i].z != refDbg[i].z || gDbg[i].w != refDbg[i].w)
            ++mmDbg;
    }
    PXG_CHECK_MSG(mmDbg == 0, "fcConvexTri: %zu SAT/feature mismatches", mmDbg);
    PXG_CHECK_MSG(checked > NC * 3 / 5, "fcConvexTri: too many tainted (%zu)", NC - checked);

    // writer slots: group by case via pairInd1 (= (CLOTH << 20) | caseIdx)
    std::vector<std::vector<uint32_t>> slots(NC);
    size_t mmW = 0;
    const uint32_t written = std::min(gWc, CAP);
    for (uint32_t s = 0; s < written; ++s)
    {
        const uint32_t caseIdx = gIn[s].z & 0xFFFFFu;
        if ((gIn[s].z >> 20) != CLOTH || caseIdx >= NC) { ++mmW; continue; }
        slots[caseIdx].push_back(s);
        const uint32_t rl = cases[caseIdx].ids.x, rh = cases[caseIdx].ids.y;
        if (gIn[s].x != rl || gIn[s].y != rh ||
            gBy[s * 2] != rl || gBy[s * 2 + 1] != rh || gTb[s] != rl || gIx[s] != s)
            ++mmW;
        const uint32_t expectMat = cases[caseIdx].ids2.x & 0x7FFFFFFFu;
        if (gIn[s].w != expectMat) ++mmW;
    }
    uint32_t sumChecked = 0, sumSoftRef = 0, numTainted = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        if (taint[i]) { ++numTainted; continue; }
        std::vector<const RefContact*> firm;
        uint32_t softRef = 0;
        for (const RefContact& rc : refCt[i])
            if (rc.firm) firm.push_back(&rc); else ++softRef;
        sumChecked += (uint32_t)firm.size();
        sumSoftRef += softRef;
        std::sort(slots[i].begin(), slots[i].end());
        size_t fi = 0;
        uint32_t softGpu = 0;
        bool bad = false;
        for (uint32_t gi2 : slots[i])
        {
            const float B = 3e-3f;
            const bool gpuFirm = gBc[gi2].x > -1e-4f + B && gBc[gi2].x < 1.0001f - B &&
                                 gBc[gi2].y > -1e-4f + B && gBc[gi2].y < 1.0001f - B &&
                                 gBc[gi2].z > -1e-4f + B && gBc[gi2].z < 1.0001f - B;
            if (!gpuFirm) { ++softGpu; continue; }
            if (fi >= firm.size()) { bad = true; break; }
            const RefContact& rc = *firm[fi++];
            if (!almostEqual(gPt[gi2].x, rc.p.x, 2e-3f, 2e-3f) ||
                !almostEqual(gPt[gi2].y, rc.p.y, 2e-3f, 2e-3f) ||
                !almostEqual(gPt[gi2].z, rc.p.z, 2e-3f, 2e-3f) ||
                gPt[gi2].w != cases[i].t0.w ||
                !almostEqual(gNp[gi2].x, rc.n.x, 2e-3f, 2e-3f) ||
                !almostEqual(gNp[gi2].y, rc.n.y, 2e-3f, 2e-3f) ||
                !almostEqual(gNp[gi2].z, rc.n.z, 2e-3f, 2e-3f) ||
                !almostEqual(gNp[gi2].w, rc.sep, 2e-4f, 2e-4f) ||
                !almostEqual(gBc[gi2].x, rc.bary.x, 5e-3f, 5e-4f) ||
                !almostEqual(gBc[gi2].y, rc.bary.y, 5e-3f, 5e-4f) ||
                !almostEqual(gBc[gi2].z, rc.bary.z, 5e-3f, 5e-4f))
            { bad = true; break; }
        }
        if (fi != firm.size() || softGpu > softRef + 2) bad = true;
        if (bad) ++mmW;
    }
    PXG_CHECK_MSG(mmW == 0, "fcConvexTri: %zu writer slot mismatches", mmW);
    PXG_CHECK_MSG(gWc >= sumChecked && gWc <= sumChecked + sumSoftRef + 2 * NC / 10 + 10 * numTainted,
                  "fcConvexTri: counter %u outside [%u, %u]", gWc, sumChecked,
                  sumChecked + sumSoftRef + 2 * NC / 10 + 10 * numTainted);
    PXG_CHECK_MSG(satFail > 100 && triFeat > 100 && faceFeat > 100 && twoFace > 100,
                  "fcConvexTri: SAT coverage thin (f=%zu t=%zu c=%zu 2f=%zu)", satFail, triFeat, faceFeat, twoFace);
    PXG_CHECK_MSG(reducedClean > 20 && baryRej > 12,
                  "fcConvexTri: reduce/bary coverage thin (rc=%zu of %zu, b=%zu)", reducedClean, reduced, baryRej);
    PXG_CHECK_MSG(edgeBranchClean > 20 && poly2Clean > 8,
                  "fcConvexTri: feature-select coverage thin (e=%zu p2=%zu)", edgeBranchClean, poly2Clean);

    // ---- P9.15c: THE CAPACITY GATE, which nothing here was probing -------------
    //
    // This file's header has always claimed "a small-capacity re-dispatch checks
    // the capacity gate (reservation ungated, stores dropped)".  It was not here:
    // CAP is 8192 and the run emits far fewer, so `index >= maxContacts` never
    // fired.  The P9.15c sweep found it -- two different mutants that delete the
    // gate both SURVIVED -- so the re-dispatch the comment promises is added now.
    //
    // The contract has two halves and they pull opposite ways: the RESERVATION is
    // ungated, so the counter must reach the same total as the full-capacity run,
    // while every store at or past the cap must be dropped.
    {
        const uint32_t SMALL = 5u;
        dev->upload(bPt, femF4Init.data(), femF4Init.size() * sizeof(F4));
        dev->upload(bNp, femF4Init.data(), femF4Init.size() * sizeof(F4));
        dev->upload(bBc, femF4Init.data(), femF4Init.size() * sizeof(F4));
        dev->upload(bIn, infoInit.data(), infoInit.size() * sizeof(U4));
        dev->upload(bTb, u1Init.data(), u1Init.size() * sizeof(uint32_t));
        dev->upload(bIx, u1Init.data(), u1Init.size() * sizeof(uint32_t));
        dev->upload(bBy, u2Init.data(), u2Init.size() * sizeof(uint32_t));
        const uint32_t zero = 0u;
        dev->upload(bWc, &zero, sizeof(uint32_t));

        FctParams small = prm;
        small.maxContacts = SMALL;
        k.dispatch((NC + 63) / 64, { bC, bDb }, small);
        dev->flush();

        const uint32_t sWc = downloadBuffer<uint32_t>(dev, bWc, 1)[0];
        PXG_CHECK_MSG(sWc == gWc,
                      "the small-capacity run reserved %u contacts, the full-capacity run %u -- "
                      "the reservation must be UNGATED, only the stores are capped", sWc, gWc);
        PXG_CHECK_MSG(gWc > SMALL,
                      "the run emits only %u contacts against a cap of %u -- the gate never "
                      "fires and deleting it is invisible", gWc, SMALL);

        const auto sPt = downloadBuffer<F4>(dev, bPt, CAP);
        const auto sIx = downloadBuffer<uint32_t>(dev, bIx, CAP);
        const auto sTb = downloadBuffer<uint32_t>(dev, bTb, CAP);
        size_t leaked = 0;
        for (uint32_t i = SMALL; i < CAP; ++i)
        {
            if (sPt[i].x != -9.0f || sPt[i].y != -9.0f) ++leaked;
            if (sIx[i] != POISON || sTb[i] != POISON) ++leaked;
        }
        PXG_CHECK_MSG(leaked == 0,
                      "%zu writer slots at or past the %u-contact cap were written", leaked, SMALL);
    }

    for (PxgBufferHandle h : { bC, bH, bPt, bNp, bBc, bIn, bWc, bBy, bTb, bIx, bDb })
        dev->destroyBuffer(h);
}

// ---------------------------------------------------------------------------
// P9.86c -- THE SIGNED ZEROS OF findRotationMatrixFromZ (femClothPrimitives.cu:988-1050).
//
// In the |to.z| > 0.9999 branch the rotation adds two constant vectors with literal
// zero lanes: col0 gets u * c1u.x = (-0, -0, +0) (:1031) and col2 gets u * c1u.z =
// (+0, 1, -1) (:1046).  IEEE turns a zero sum into +0 when it adds that +0; llvmpipe
// drops the +0 lanes (F3), so the port spells col0.z and col2.x as selects.  And
// v = absFrom - to (:1016) is 0 - to.x in lane x, which llvmpipe rewrites to -to.x
// (F2): -0 where the CUDA has +0 when to.x = +0.
//
// The random battery above never sees any of it.  The rotation's zero lanes are
// signed differently only when the SAT axis has x = +-0, and even then they reach an
// output in two narrow ways only -- the rotated points feed comparisons and the
// segment intersection, whose zero components come out +0:
//   * the transpose dot  contactPoint = rot^T * pBB  of a phase-3 crossing whose
//     other terms are zeros as well: a triangle edge ON the rotated x axis (case A);
//   * a crossing point whose x is -0 because t1 * (rpB.x - rpA.x) underflows to +0:
//     a triangle vertex at x = -0 within 2^-24 (in t1) of the polygon edge it crosses,
//     and the edge's other end at x = -FLT_MIN (cases C and D).
// Every input is a NORMAL float.  The world transform keeps the sign only when its
// other terms are zeros of the same sign, hence a convexQ with -0 imaginary parts (the
// conjugate of the identity, or one negated zero) and convexP lanes of -0.
//
//   A  box z in [0.05, 2.05]; TRI axis to = (-0, +0.01, -0.99995), the triangle edge
//      t0-t1 on the local x axis.  CUDA col2.x = +0, -0 without the select: slot 2's
//      world z flips from -0 to +0.                                     (twin bit 1)
//   C  box y in [-2, ~2^-11]; TRI axis to = (-0, -0.0078, -0.99997).  CUDA col0.z = +0,
//      -0 without the select: slot 1's world x flips from +0 to -0.     (twin bit 2)
//   D  a bottom face tilted about x, normal (+0, s, -c), wins the SAT, so to.x = +0.
//      CUDA v.x = +0, F2 gives -0: col0.y and col1.x turn +0, triangle vertex t0 =
//      (-0, +0, -0) rotates to x = +0, and slot 1's world x flips from -0 to +0.
//      (twin bit 4; the col2.x twin flips it too, through rot * t0)
// The polygon-edge positions of C and D (their hulls' yhi) sit inside windows of about
// 2000 ulps where the underflow happens; they were chosen in the middle.
//
// Every lane these cases write is compared BITWISE -- the directed inputs are exact,
// so nothing here needs the battery's tolerances or its firm/soft split.
// ---------------------------------------------------------------------------

// The box's vertex/face/edge numbering: x in [xlo, xhi], y in [ylo, yhi], top face z =
// ztop, bottom face (index 5) the plane {nb, wb} with nb.x = 0 (a tilt about x) and its
// four vertices on it.  nb = (0, 0, -1) is an axis-aligned box.
static Hull makeRotLaneHull(float xlo, float xhi, float ylo, float yhi, float ztop, V3 nb, float wb)
{
    Hull h = makeBox(1, 1, 1); // faces and edgeFaces
    h.verts.clear();
    for (int i = 0; i < 8; ++i)
    {
        const float y = (i & 2) ? yhi : ylo;
        h.verts.push_back({ (i & 1) ? xhi : xlo, y, (i & 4) ? ztop : (nb.y * y + wb) / -nb.z });
    }
    h.planes = { { 1, 0, 0, -xhi }, { -1, 0, 0, xlo }, { 0, 1, 0, -yhi },
                 { 0, -1, 0, ylo }, { 0, 0, 1, -ztop }, { nb.x, nb.y, nb.z, wb } };
    return h;
}

// The SAT axis the battery's reference derives for a verdict (featureIndex).
static V3 rotLaneMinNormal(const Hull& H, const FctCase& cs, uint32_t feature)
{
    if (feature == TRI_FEAT)
        return mul(v3f(cs.triNormal), -1.0f);
    const V3 scale = { cs.convexScale.x, cs.convexScale.y, cs.convexScale.z };
    const V3 n = v2sNormal({ H.planes[feature].x, H.planes[feature].y, H.planes[feature].z }, scale,
                           cs.convexScaleRot);
    const float invMag = 1.0f / std::sqrt(dot3(n, n));
    return mul(n, invMag);
}

// The battery's per-face block (plane0, polyClip4, the barycentric gate, the world
// transform) for a face that emits at most 5 contacts, so contactReduce never runs.
// Returns false if it would.
static bool rotLaneFaceContacts(const Hull& H, const FctCase& cs, V3 minNormal, uint32_t face,
                                std::vector<RefContact>& out)
{
    const V3 scale = { cs.convexScale.x, cs.convexScale.y, cs.convexScale.z };
    const F4 srot = cs.convexScaleRot;
    const V3 t[3] = { v3f(cs.t0), v3f(cs.t1), v3f(cs.t2) };
    const V3 nT = v3f(cs.triNormal);
    const F4 triPlane = { nT.x, nT.y, nT.z, -dot3(t[0], nT) };
    V3 p0n = v2sNormal({ H.planes[face].x, H.planes[face].y, H.planes[face].z }, scale, srot);
    float p0d = H.planes[face].w;
    const float dn = 1.0f / std::sqrt(dot3(p0n, p0n));
    p0n = mul(p0n, dn);
    p0d *= dn;
    const F4 plane0 = { p0n.x, p0n.y, p0n.z, p0d };
    std::vector<V3> pverts;
    for (uint32_t idx : H.faces[face])
        pverts.push_back(vertex2Shape(H.verts[idx], scale, srot));
    float ct[NUMTMP * 8] = {};
    Slack sl; // near-ties are inherent to C and D; the GPU compare is the arbiter
    const uint32_t num = refPolyClip(pverts, (uint32_t)pverts.size(), minNormal, plane0, triPlane, t,
                                     cs.convexScale.w, ct, sl);
    std::vector<RefContact> faceCt;
    for (uint32_t lane = 0; lane < 32 && lane < num; ++lane)
    {
        const V3 pa = { ct[8 * lane], ct[8 * lane + 1], ct[8 * lane + 2] };
        const V3 e0 = sub(t[1], t[0]), e1 = sub(t[2], t[0]), e2 = sub(pa, t[0]);
        const float d00 = dot3(e0, e0), d01 = dot3(e0, e1), d11 = dot3(e1, e1);
        const float d20 = dot3(e2, e0), d21 = dot3(e2, e1);
        const float dnm = d00 * d11 - d01 * d01;
        const float bv = (d11 * d20 - d01 * d21) / dnm;
        const float bw = (d00 * d21 - d01 * d20) / dnm;
        const float bu = 1.0f - bv - bw;
        if (!(bu >= -1e-4f && bu <= 1.0f + 1e-4f && bv >= -1e-4f && bv <= 1.0f + 1e-4f &&
              bw >= -1e-4f && bw <= 1.0f + 1e-4f))
            continue;
        RefContact rc;
        rc.p = add(quatRotate(cs.convexQ, pa), v3f(cs.convexP));
        rc.n = quatRotate(cs.convexQ, { ct[8 * lane + 4], ct[8 * lane + 5], ct[8 * lane + 6] });
        rc.sep = ct[8 * lane + 3];
        rc.bary = { bu, bv, bw, 0 };
        rc.firm = true;
        faceCt.push_back(rc);
    }
    if (faceCt.size() > 5)
        return false;
    out.insert(out.end(), faceCt.begin(), faceCt.end());
    return true;
}

PXG_TEST(fcConvexTri_contactGenRotLaneZeroSign)
{
    using btri::f2u;
    auto u2f = [](uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; };
    auto laneOf = [](const V3& v, int l) { return l == 0 ? v.x : (l == 1 ? v.y : v.z); };
    const uint32_t NC = 3, CAP = 64, CLOTH = 17, POISON = 0xeeeeeeeeu, INFOPOISON = 0xdeadbeefu;

    const V3 nD = getNorm0({ 0.0f, 1.0f / 128.0f, -1.0f });
    const Hull hulls[NC] = {
        makeRotLaneHull(-1, 1, -1, 1, 2.05f, { 0, 0, -1 }, 0.05f),                 // A
        makeRotLaneHull(-1, 1, -2, u2f(0x3a000010u), 1, { 0, 0, -1 }, -1),          // C
        makeRotLaneHull(-1, 1, -2, u2f(0xb9cccf33u), 2, nD, 0.05f),                  // D
    };
    std::vector<uint32_t> blob;
    uint32_t ofs[NC];
    for (uint32_t i = 0; i < NC; ++i)
        hulls[i].serialize(blob, ofs[i]);

    struct Directed
    {
        const char* name;
        V3 t0, t1, t2;
        F4 convexP, convexQ;
        U4 dbg;             // expected (isSATPassed, featureIndex, face1, face2)
        uint32_t minNx;     // minNormal.x the case is built on
        float minNySign;    // sign of minNormal.y: c3 < 0 iff to.y > 0
        uint32_t slot;      // the slot whose world point carries the sign
        int lane;           // ... in this lane
        uint32_t cudaBits;  // ... with these bits under the CUDA semantics
        int twin;           // the gRotZLaneVariant bit that flips it
    };
    const float fmin = u2f(0x00800000u); // FLT_MIN
    const Directed dc[NC] = {
        { "A", { -3, 0, 0 }, { 3, 0, 0 }, { 0, 5, 0.05f }, { 0.25f, -0.5f, -0.0f, 0 }, { -0.0f, -0.0f, -0.0f, 1 },
          { 1, TRI_FEAT, 5, 2 }, 0x80000000u, 1.0f, 2, 2, 0x80000000u, 1 },
        { "C", { -0.0f, -0.0f, -1.0625f }, { 4, -4, -1.03125f }, { -fmin, 4, -1.09375f }, { -0.0f, 0.5f, 0.25f, 0 },
          { -0.0f, 0.0f, 0.0f, 1 }, { 1, TRI_FEAT, 5, 3 }, 0x80000000u, -1.0f, 1, 0, 0x00000000u, 2 },
        { "D", { -0.0f, 0.0f, -0.0f }, { 1.5f, 0.3f, 0 }, { -fmin, -1, 0.02f }, { -0.0f, 0.5f, 0.25f, 0 },
          { -0.0f, -0.0f, -0.0f, 1 }, { 1, 5, 5, 0xFFFFFFFFu }, 0x00000000u, 1.0f, 1, 0, 0x80000000u, 4 },
    };
    std::vector<FctCase> cases(NC);
    for (uint32_t i = 0; i < NC; ++i)
    {
        const Directed& d = dc[i];
        FctCase& cs = cases[i];
        const V3 nT = getNorm0(crossp(sub(d.t1, d.t0), sub(d.t2, d.t0)));
        cs.convexScale = { 1, 1, 1, 0.1f };
        cs.convexScaleRot = { 0, 0, 0, 1 };
        cs.t0 = { d.t0.x, d.t0.y, d.t0.z, 0.01f };
        cs.t1 = { d.t1.x, d.t1.y, d.t1.z, 0 };
        cs.t2 = { d.t2.x, d.t2.y, d.t2.z, 0 };
        cs.triNormal = { nT.x, nT.y, nT.z, 0 };
        cs.convexP = d.convexP;
        cs.convexQ = d.convexQ;
        cs.ids = { 0xA0u + i, 3u, CLOTH, i };
        cs.ids2 = { 0x40u + i, ofs[i], 0, 0 };
    }

    // ---- reference: the CUDA semantics, then each case's twin ----
    auto reference = [&](uint32_t i, std::vector<RefContact>& out) {
        const V3 mn = rotLaneMinNormal(hulls[i], cases[i], dc[i].dbg.y);
        bool ok = true;
        for (uint32_t face : { dc[i].dbg.z, dc[i].dbg.w })
            if (face != 0xFFFFFFFFu)
                ok = rotLaneFaceContacts(hulls[i], cases[i], mn, face, out) && ok;
        return ok;
    };
    std::vector<RefContact> ref[NC];
    uint32_t twinFlips = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        const Directed& d = dc[i];
        const V3 mn = rotLaneMinNormal(hulls[i], cases[i], d.dbg.y);
        PXG_CHECK_MSG(f2u(mn.x) == d.minNx && std::fabs(mn.z) > 0.9999f && mn.y * d.minNySign > 0.0f,
                      "case %s: SAT axis (%08x, %g, %g) is not the one the case is built on", d.name,
                      f2u(mn.x), mn.y, mn.z);
        PXG_CHECK_MSG(reference(i, ref[i]), "case %s: a face emits more than 5 contacts", d.name);
        PXG_CHECK_MSG(ref[i].size() > d.slot && f2u(laneOf(ref[i][d.slot].p, d.lane)) == d.cudaBits,
                      "case %s: the reference no longer puts %08x in slot %u lane %d", d.name, d.cudaBits,
                      d.slot, d.lane);
        gRotZLaneVariant = d.twin;
        std::vector<RefContact> tw;
        const bool ok = reference(i, tw);
        gRotZLaneVariant = 0;
        if (ok && tw.size() == ref[i].size() && tw.size() > d.slot &&
            f2u(laneOf(tw[d.slot].p, d.lane)) == (d.cudaBits ^ 0x80000000u))
            ++twinFlips;
    }
    PXG_CHECK_MSG(twinFlips == NC,
                  "only %u of %u directed cases flip their sign under the P9.86 twin -- the inputs no longer "
                  "exercise the rotation's zero lanes", twinFlips, NC);

    // ---- GPU ----
    PxgSlangComputeDevice* dev = testDevice();
    std::vector<F4> f4Init(CAP, { -9, -9, -9, -9 });
    std::vector<U4> infoInit(CAP, { INFOPOISON, INFOPOISON, INFOPOISON, INFOPOISON });
    std::vector<uint32_t> u1Init(CAP, POISON), u2Init(CAP * 2, POISON), wc0(1, 0);
    std::vector<U4> dbgInit(NC, { 7, 7, 7, 7 });
    PxgBufferHandle bC = createBufferFrom(dev, cases);
    PxgBufferHandle bH = createBufferFrom(dev, blob);
    PxgBufferHandle bPt = createBufferFrom(dev, f4Init);
    PxgBufferHandle bNp = createBufferFrom(dev, f4Init);
    PxgBufferHandle bBc = createBufferFrom(dev, f4Init);
    PxgBufferHandle bIn = createBufferFrom(dev, infoInit);
    PxgBufferHandle bWc = createBufferFrom(dev, wc0);
    PxgBufferHandle bBy = createBufferFrom(dev, u2Init);
    PxgBufferHandle bTb = createBufferFrom(dev, u1Init);
    PxgBufferHandle bIx = createBufferFrom(dev, u1Init);
    PxgBufferHandle bDb = createBufferFrom(dev, dbgInit);
    FctParams prm = {};
    prm.numCases = NC; prm.maxContacts = CAP;
    prm.hullAddr = dev->getBufferDeviceAddress(bH);
    prm.wPoint = dev->getBufferDeviceAddress(bPt);
    prm.wNormalPen = dev->getBufferDeviceAddress(bNp);
    prm.wBary = dev->getBufferDeviceAddress(bBc);
    prm.wInfo = dev->getBufferDeviceAddress(bIn);
    prm.wCount = dev->getBufferDeviceAddress(bWc);
    prm.wByX = dev->getBufferDeviceAddress(bBy);
    prm.wTempByX = dev->getBufferDeviceAddress(bTb);
    prm.wIdxByX = dev->getBufferDeviceAddress(bIx);
    Kernel k("fcConvexTri.spv", "ctFcConvexTri", { (uint32_t)sizeof(FctCase), 16 }, sizeof(FctParams));
    k.dispatch(1, { bC, bDb }, prm);
    dev->flush();
    const auto gDbg = downloadBuffer<U4>(dev, bDb, NC);
    const auto gPt = downloadBuffer<F4>(dev, bPt, CAP);
    const auto gNp = downloadBuffer<F4>(dev, bNp, CAP);
    const auto gBc = downloadBuffer<F4>(dev, bBc, CAP);
    const auto gIn = downloadBuffer<U4>(dev, bIn, CAP);
    const uint32_t gWc = downloadBuffer<uint32_t>(dev, bWc, 1)[0];

    size_t refTotal = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        refTotal += ref[i].size();
        PXG_CHECK_MSG(gDbg[i].x == dc[i].dbg.x && gDbg[i].y == dc[i].dbg.y && gDbg[i].z == dc[i].dbg.z &&
                      gDbg[i].w == dc[i].dbg.w,
                      "case %s: SAT/feature verdict (%u, %08x, %u, %08x), expected (%u, %08x, %u, %08x)", dc[i].name,
                      gDbg[i].x, gDbg[i].y, gDbg[i].z, gDbg[i].w, dc[i].dbg.x, dc[i].dbg.y, dc[i].dbg.z, dc[i].dbg.w);
    }
    PXG_CHECK_MSG(gWc == refTotal, "%u contacts written, the reference has %zu", gWc, refTotal);

    // one thread per case: its slots ascend in the reference's order (face 1, then face 2)
    std::vector<std::vector<uint32_t>> slots(NC);
    size_t mmInfo = 0;
    for (uint32_t s = 0; s < std::min(gWc, CAP); ++s)
    {
        const uint32_t i = gIn[s].z & 0xFFFFFu;
        if ((gIn[s].z >> 20) != CLOTH || i >= NC) { ++mmInfo; continue; }
        slots[i].push_back(s);
        if (gIn[s].x != cases[i].ids.x || gIn[s].y != cases[i].ids.y || gIn[s].w != (cases[i].ids2.x & 0x7FFFFFFFu))
            ++mmInfo;
    }
    PXG_CHECK_MSG(mmInfo == 0, "%zu contacts carry the wrong pair/material words", mmInfo);

    size_t mmBits = 0;
    uint32_t signLanes = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        const Directed& d = dc[i];
        std::sort(slots[i].begin(), slots[i].end());
        PXG_CHECK_MSG(slots[i].size() == ref[i].size(), "case %s: %zu contacts, the reference has %zu", d.name,
                      slots[i].size(), ref[i].size());
        for (size_t kk = 0; kk < slots[i].size() && kk < ref[i].size(); ++kk)
        {
            const uint32_t s = slots[i][kk];
            const RefContact& rc = ref[i][kk];
            const float want[12] = { rc.p.x, rc.p.y, rc.p.z, cases[i].t0.w, rc.n.x, rc.n.y, rc.n.z, rc.sep,
                                     rc.bary.x, rc.bary.y, rc.bary.z, rc.bary.w };
            const float got[12] = { gPt[s].x, gPt[s].y, gPt[s].z, gPt[s].w, gNp[s].x, gNp[s].y, gNp[s].z, gNp[s].w,
                                    gBc[s].x, gBc[s].y, gBc[s].z, gBc[s].w };
            for (int j = 0; j < 12; ++j)
                if (f2u(got[j]) != f2u(want[j]))
                {
                    if (mmBits < 8)
                        std::printf("  case %s contact %zu word %d: GPU %08x, reference %08x\n", d.name, kk, j,
                                    f2u(got[j]), f2u(want[j]));
                    ++mmBits;
                }
            if (kk == d.slot && f2u(laneOf(V3{ gPt[s].x, gPt[s].y, gPt[s].z }, d.lane)) == d.cudaBits)
                ++signLanes;
        }
    }
    PXG_CHECK_MSG(mmBits == 0, "%zu written words differ bitwise from the reference", mmBits);
    PXG_CHECK_MSG(signLanes == NC, "only %u of %u directed signed-zero lanes were written as the CUDA has them",
                  signLanes, NC);

    for (PxgBufferHandle h : { bC, bH, bPt, bNp, bBc, bIn, bWc, bBy, bTb, bIx, bDb })
        dev->destroyBuffer(h);
}

} // namespace
