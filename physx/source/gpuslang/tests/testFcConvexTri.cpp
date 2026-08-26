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

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

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
        const V3 u = sub(absFrom, from), v = sub(absFrom, to);
        const float c1 = -(2.0f / dot3(u, u));
        const float c2 = -(2.0f / dot3(v, v));
        const float c3 = c1 * c2 * dot3(u, v);
        const V3 c1u = mul(u, c1), c2v = mul(v, c2), c3v = mul(v, c3);
        V3 col0 = add(add(mul(u, c3v.x), mul(v, c2v.x)), mul(u, c1u.x)); col0.x += 1.0f;
        V3 col1 = add(add(mul(u, c3v.y), mul(v, c2v.y)), mul(u, c1u.y)); col1.y += 1.0f;
        V3 col2 = add(add(mul(u, c3v.z), mul(v, c2v.z)), mul(u, c1u.z)); col2.z += 1.0f;
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

} // namespace
