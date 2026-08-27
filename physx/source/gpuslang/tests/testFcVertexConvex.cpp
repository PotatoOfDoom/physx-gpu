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

// Verification of fcVertexConvex.slang (clothVertexConvexCollision:
// PxgRigidFilterPair attachment filter + closestVertToConvex +
// computeClosestPolygon + writeRigidVsDeformableContact).  Hull blobs are
// the tranche-96 analytic builders (box, irregular hex prism, 24-gon
// prism).  The sorted filter table mixes real hits (~10% of cases),
// near-miss entries (same rigid id, adjacent vertex mask; same mask,
// different rigid id) that must NOT match, and index2-nonzero entries
// accepted through the ignoreIndex2 rule.  gDebug pins the filter
// verdict, the raw pen, the write decision and the all-inside branch;
// writer slots map injectively via pairInd1 = (clothId << 20) | caseIdx
// and verify every field including the (0,0,0,1) VERTEX barycentric
// marker and the material dword with bit 31 cleared.  Slack taints:
// inside/outside membership boundaries (scaled and unscaled plane
// distances near zero), argmax/min near-ties (winner excluded), edge
// plane membership and degenerate edge cross products, segment-distance
// ties, the polygon running-min ties and the pen-vs-contactDist gate.

#include "pxgTestFramework.h"

// P9.21 drives the engine-ABI kernel beside the Phase-1 core, so it needs the
// real narrowphase structs.
#include "host_defines.h"
#include "PxgFEMCloth.h"
#include "PxgContactManager.h"
#include "PxgConvexConvexShape.h"
#include "PxgSimulationCoreDesc.h"
#include "PxsTransformCache.h"
#include "PxNodeIndex.h"
#include "geometry/PxGeometry.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };
struct FvcCase { F4 clothP, rigidP, rigidQ, convexScale, convexScaleRot; U4 ids, ids2; };
// P9.21: the hull and the writer moved to BDA addresses when the core was
// lifted into pxgSlangVertexConvexCloth -- an engine-ABI kernel has no
// descriptor set to bind them into, so the module can only take addresses, and
// this battery hands it the same kind.
struct FvcParams
{
    uint32_t numCases, maxContacts, nbFilterPairs, pad0;   // @0
    uint64_t hullAddr;                                     // @16
    uint64_t wPoint, wNormalPen, wBary, wInfo, wCount, wByX, wTempByX, wIdxByX;  // @24
};
static_assert(sizeof(FvcParams) == 88, "FvcParams must match the shader blob");
struct FilterPair { uint64_t index0, index1; uint32_t index2, pad0, pad1, pad2; };

struct V3 { float x, y, z; };
static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 add(V3 a, V3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 mul(V3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
static V3 mulv(V3 a, V3 b) { return { a.x * b.x, a.y * b.y, a.z * b.z }; }
static V3 divv(V3 a, V3 b) { return { a.x / b.x, a.y / b.y, a.z / b.z }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 crossp(V3 a, V3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static V3 normalize3(V3 v) { const float m = std::sqrt(dot3(v, v)); return mul(v, 1.0f / m); }
static V3 getNorm0(V3 v) { const float m = dot3(v, v); return m > 0 ? mul(v, 1.0f / std::sqrt(m)) : V3{ 0, 0, 0 }; }
static V3 v3f(F4 f) { return { f.x, f.y, f.z }; }

static const float MAXF = 3.4028235e38f;

struct Slack
{
    bool near = false;
    void noteSc(float l, float r, float scale) { if (std::fabs(l - r) < 1e-5f * scale) near = true; }
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
static V3 vertex2Shape(V3 v, V3 scale, F4 rot) { return quatRotateInv(rot, mulv(quatRotate(rot, v), scale)); }
static V3 shape2Vertex(V3 v, V3 scale, F4 rot) { return quatRotateInv(rot, divv(quatRotate(rot, v), scale)); }

// ---- hull blob builders (tranche 96) ----------------------------------------
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
        pushF4(0, 0, 0, 0);
        blob.push_back((nbE << 16) | (nbV << 8) | nbP);
        blob.push_back(0); blob.push_back(0); blob.push_back(0);
        pushF4(1, 1, 1, 0);
        for (const V3& v : verts) pushF4(v.x, v.y, v.z, 0);
        for (const F4& p : planes) pushF4(p.x, p.y, p.z, p.w);
        for (uint32_t d : polyDescs) blob.push_back(d);
        std::vector<uint8_t> bytes;
        for (uint32_t e = 0; e < nbE * 4; ++e) bytes.push_back(0);
        for (const auto& ef : edgeFaces) { bytes.push_back((uint8_t)ef.first); bytes.push_back((uint8_t)ef.second); }
        for (uint32_t v = 0; v < nbV * 3; ++v) bytes.push_back(0);
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
        V3 sn = normalize3({ e.y, -e.x, 0 });
        h.planes.push_back({ sn.x, sn.y, sn.z, -dot3(sn, h.verts[k]) });
        h.faces.push_back({ k, k1, n + k1, n + k });
        h.edgeFaces.push_back({ 1, 2 + k });
        h.edgeFaces.push_back({ 0, 2 + k });
        h.edgeFaces.push_back({ 2 + k, 2 + (k + n - 1) % n });
    }
    return h;
}

// ---- reference --------------------------------------------------------------
static V3 distancePointSegment(V3 a, V3 b, V3 p)
{
    const V3 ap = sub(p, a), ab = sub(b, a);
    const float nom = dot3(ap, ab), denom = dot3(ab, ab);
    const float tValue = std::min(std::max(nom / denom, 0.0f), 1.0f);
    const float t = denom > 0.0f ? tValue : 0.0f;
    return add(a, mul(ab, t));
}

struct RefOut { bool filtered, wrote, inside; float pen; V3 p, n; };

PXG_TEST(fcVertexConvex_closest)
{
    std::mt19937 rng(97043);
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<uint32_t> blob;
    uint32_t boxOfs, prismOfs, bigOfs;
    const Hull box = makeBox(0.8f, 0.6f, 1.0f);
    const float prismRadii[6] = { 0.92f, 0.78f, 1.02f, 0.85f, 0.97f, 0.72f };
    const Hull prism = makePrism(6, prismRadii, 0.7f);
    float bigRadii[24];
    for (int k = 0; k < 24; ++k)
        bigRadii[k] = 0.85f + 0.18f * std::sin(k * 1.7f) * std::cos(k * 0.9f);
    const Hull bigPrism = makePrism(24, bigRadii, 0.6f);
    float hugeRadii[31];
    for (int k = 0; k < 31; ++k)
        hugeRadii[k] = 0.9f + 0.12f * std::sin(k * 2.1f);
    const Hull hugePrism = makePrism(31, hugeRadii, 0.55f); // 33 planes: the
    // per-lane strided loops take a SECOND chunk (nbPolygons > 32)
    uint32_t hugeOfs;
    box.serialize(blob, boxOfs);
    prism.serialize(blob, prismOfs);
    bigPrism.serialize(blob, bigOfs);
    hugePrism.serialize(blob, hugeOfs);

    const uint32_t NC = 1500, CAP = 2048, POISON = 0xeeeeeeeeu, INFOPOISON = 0xdeadbeefu;
    const uint32_t CLOTH = 21, SHIFT = 20;

    // filter table: real hits for i%10==3, near-misses for i%10==6, plus
    // index2-nonzero entries (matched through ignoreIndex2)
    std::vector<FilterPair> filters;
    auto rigidOf = [&](uint32_t i) { return (uint64_t(3 + i % 3) << 32) | (0xA0u + i % 7); };
    for (uint32_t i = 0; i < NC; ++i)
    {
        const uint64_t mask = (uint64_t)((CLOTH << SHIFT) | i);
        if (i % 10 == 3)
            filters.push_back({ rigidOf(i), mask, (i % 20 == 13) ? 7u : 0u, 0, 0, 0 });
        if (i % 10 == 6)
        {
            filters.push_back({ rigidOf(i), mask + 1, 0, 0, 0, 0 });          // adjacent vertex
            filters.push_back({ rigidOf(i) ^ 0x100u, mask, 0, 0, 0, 0 });     // different rigid
        }
    }
    std::sort(filters.begin(), filters.end(), [](const FilterPair& a, const FilterPair& b) {
        if (a.index0 != b.index0) return a.index0 < b.index0;
        if (a.index1 != b.index1) return a.index1 < b.index1;
        return a.index2 < b.index2;
    });

    std::vector<FvcCase> cases(NC);
    std::vector<RefOut> ref(NC);
    std::vector<uint8_t> taint(NC, 0);
    size_t filteredN = 0, insideN = 0, outsideEdgeN = 0, gateRejN = 0, wroteN = 0;

    for (uint32_t i = 0; i < NC; ++i)
    {
        const uint32_t which = i % 3;
        const bool useHuge = (which == 2) && (i % 6 == 5);
        const Hull& H = which == 0 ? box : (which == 1 ? prism : (useHuge ? hugePrism : bigPrism));
        FvcCase& cs = cases[i];
        const V3 scale = { rf(0.7f, 1.4f), rf(0.7f, 1.4f), rf(0.7f, 1.4f) };
        F4 srot = { 0, 0, 0, 1 };
        if (i % 5 == 2)
        {
            srot = { rf(-1, 1), rf(-1, 1), rf(-1, 1), rf(-1, 1) };
            const float m = std::sqrt(srot.x * srot.x + srot.y * srot.y + srot.z * srot.z + srot.w * srot.w);
            srot = { srot.x / m, srot.y / m, srot.z / m, srot.w / m };
        }
        const float cd = rf(0.05f, 0.4f), rest = rf(0.0f, 0.1f);

        // vertex placed in SHAPE space then transformed to world
        const V3 dir = normalize3({ rf(-1, 1), rf(-1, 1), rf(-1, 1) });
        float rad;
        if (i % 7 == 1) rad = rf(0.05f, 0.45f);       // deep inside
        else if (i % 7 == 4) rad = rf(2.5f, 4.0f);    // far (gate reject)
        else rad = rf(0.6f, 1.9f);                    // near the surface
        const V3 shapeP = mul(dir, rad);

        F4 rq = { rf(-1, 1), rf(-1, 1), rf(-1, 1), rf(-1, 1) };
        const float qm = std::sqrt(rq.x * rq.x + rq.y * rq.y + rq.z * rq.z + rq.w * rq.w);
        rq = { rq.x / qm, rq.y / qm, rq.z / qm, rq.w / qm };
        const V3 rp = { rf(-1, 1), rf(-1, 1), rf(-1, 1) };
        const V3 world = add(quatRotate(rq, shapeP), rp);

        cs.clothP = { world.x, world.y, world.z, cd };
        cs.rigidP = { rp.x, rp.y, rp.z, rest };
        cs.rigidQ = rq;
        cs.convexScale = { scale.x, scale.y, scale.z, 0 };
        cs.convexScaleRot = srot;
        cs.ids = { 0xA0u + i % 7, 3u + i % 3, CLOTH, i };
        cs.ids2 = { (i % 4 == 1) ? (0x80000000u | (i & 0xfffu)) : (i & 0xfffu),
                    which == 0 ? boxOfs : (which == 1 ? prismOfs : (useHuge ? hugeOfs : bigOfs)), 0, 0 };

        // ---- reference ----
        Slack sl;
        RefOut& ro = ref[i];
        ro = {};

        const uint64_t rid = rigidOf(i);
        const uint64_t mask = (uint64_t)((CLOTH << SHIFT) | i);
        bool filtered = false;
        {
            // binary search with 3-key compare, ignoreIndex2 acceptance
            uint32_t l = 0, r = (uint32_t)filters.size();
            while (l < r)
            {
                const uint32_t mid = (l + r) / 2;
                const FilterPair& p = filters[mid];
                int cmp;
                if (rid != p.index0) cmp = rid < p.index0 ? -1 : 1;
                else if (mask != p.index1) cmp = mask < p.index1 ? -1 : 1;
                else if (0u != p.index2) cmp = -1;
                else cmp = 0;
                if (cmp == 0 || (rid == p.index0 && mask == p.index1)) { filtered = true; break; }
                if (cmp < 0) r = mid; else l = mid + 1;
            }
        }
        ro.filtered = filtered;
        if (filtered) { ++filteredN; taint[i] = 0; continue; }

        const uint32_t nbP = (uint32_t)H.faces.size();
        const V3 pVertexSpace = shape2Vertex(shapeP, scale, srot);

        float laneMax[32];
        uint32_t laneIdx[32];
        bool outside = false;
        for (int l = 0; l < 32; ++l) { laneMax[l] = -MAXF; laneIdx[l] = 0; }
        for (uint32_t p = 0; p < nbP; ++p)
        {
            const uint32_t lane = p % 32;
            const F4& pl = H.planes[p];
            const float sdv = dot3({ pl.x, pl.y, pl.z }, pVertexSpace) + pl.w;
            const V3 sn = mulv(quatRotate(srot, { pl.x, pl.y, pl.z }), scale);
            const float scaleCoef = std::sqrt(dot3(sn, sn));
            const float signDist = sdv * scaleCoef;
            sl.noteSc(signDist, 0.0f, std::fabs(signDist) + 1.0f); // membership boundary
            outside = outside || signDist > 0.0f;
            if (signDist > laneMax[lane]) { laneMax[lane] = signDist; laneIdx[lane] = p; }
        }

        V3 closestP{}, normal{};
        float maxVal;
        if (outside)
        {
            float minDist = MAXF;
            for (uint32_t idx = 0; idx < nbP; ++idx)
            {
                const F4& pl = H.planes[idx];
                const float sd = dot3({ pl.x, pl.y, pl.z }, pVertexSpace) + pl.w;
                sl.noteSc(sd, 0.0f, std::fabs(sd) + 1.0f);
                if (!(sd > 0.0f)) continue;

                const auto& face = H.faces[idx];
                const uint32_t nbVerts = (uint32_t)face.size();
                V3 polyClosestP{};
                float polyDist;
                {
                    float laneSign[32];
                    bool anyOutside = false;
                    for (uint32_t l = 0; l < 32; ++l)
                    {
                        laneSign[l] = -MAXF;
                        if (l < nbVerts)
                        {
                            const V3 rpA = H.verts[face[l]];
                            const V3 rpB = H.verts[face[l + 1 == nbVerts ? 0 : l + 1]];
                            const V3 cr = crossp({ pl.x, pl.y, pl.z }, sub(rpA, rpB));
                            sl.noteSc(std::sqrt(dot3(cr, cr)), 0.0f, 1.0f); // degenerate edge plane
                            const V3 epn = getNorm0(cr);
                            laneSign[l] = dot3(sub(pVertexSpace, rpA), epn);
                            sl.noteSc(laneSign[l], 0.0f, std::fabs(laneSign[l]) + 0.5f);
                            anyOutside = anyOutside || laneSign[l] > 0.0f;
                        }
                    }
                    if (anyOutside)
                    {
                        ++outsideEdgeN;
                        float best = MAXF;
                        uint32_t winner = 0;
                        float dists[32];
                        V3 cps[32];
                        for (uint32_t l = 0; l < 32; ++l)
                        {
                            dists[l] = MAXF;
                            cps[l] = {};
                            if (l < nbVerts && laneSign[l] > 0.0f)
                            {
                                const V3 rpA = H.verts[face[l]];
                                const V3 rpB = H.verts[face[l + 1 == nbVerts ? 0 : l + 1]];
                                cps[l] = distancePointSegment(rpA, rpB, pVertexSpace);
                                const V3 d = sub(cps[l], pVertexSpace);
                                dists[l] = dot3(d, d);
                            }
                            if (dists[l] < best) best = dists[l];
                        }
                        for (uint32_t l = 0; l < 32; ++l)
                            if (dists[l] == best) { winner = l; break; }
                        // a tie is only unstable if the tied edges disagree
                        // on the closest point (shared-vertex ties are benign)
                        for (uint32_t l = 0; l < 32; ++l)
                        {
                            if (l == winner || dists[l] >= MAXF) continue;
                            const V3 dcp = sub(cps[l], cps[winner]);
                            if (dot3(dcp, dcp) > 1e-6f) sl.noteSc(dists[l], best, 8.0f);
                        }
                        polyClosestP = cps[winner];
                        polyDist = std::sqrt(best);
                    }
                    else
                    {
                        polyDist = sd;
                        polyClosestP = sub(pVertexSpace, mul({ pl.x, pl.y, pl.z }, sd));
                    }
                }
                if (polyDist < minDist)
                {
                    minDist = polyDist;
                    closestP = polyClosestP;
                }
                else
                {
                    // adjacent faces sharing the closest edge tie with the
                    // SAME point — only point-disagreeing ties are unstable
                    const V3 dcp = sub(polyClosestP, closestP);
                    if (dot3(dcp, dcp) > 1e-6f) sl.noteSc(polyDist, minDist, 8.0f);
                }
            }
            normal = sub(pVertexSpace, closestP);
            const V3 sn = mulv(quatRotate(srot, normal), scale);
            maxVal = std::sqrt(dot3(sn, sn));
        }
        else
        {
            ro.inside = true;
            ++insideN;
            float best = -MAXF;
            for (int l = 0; l < 32; ++l) best = std::max(best, laneMax[l]);
            uint32_t winner = 0;
            for (int l = 0; l < 32; ++l)
                if (laneMax[l] == best) { winner = (uint32_t)l; break; }
            for (int l = 0; l < 32; ++l)
                if ((uint32_t)l != winner && laneMax[l] > -MAXF) sl.noteSc(laneMax[l], best, 8.0f);
            const uint32_t polyFaceIndex = laneIdx[winner];
            const F4& pl = H.planes[polyFaceIndex];
            const float sd = dot3({ pl.x, pl.y, pl.z }, pVertexSpace) + pl.w;
            closestP = sub(pVertexSpace, mul({ pl.x, pl.y, pl.z }, sd));
            normal = { pl.x, pl.y, pl.z };
            maxVal = best;
        }

        V3 outContact = vertex2Shape(closestP, scale, srot);
        V3 outNormal = shape2Vertex(normal, scale, srot);
        const float mag = std::sqrt(dot3(outNormal, outNormal));
        if (mag >= 1e-20f) outNormal = mul(outNormal, 1.0f / mag); // normalizeSafe leaves it unchanged below eps
        const float pen = maxVal;

        ro.pen = pen;
        ro.p = add(quatRotate(rq, outContact), rp);
        ro.n = getNorm0(quatRotate(rq, outNormal));
        sl.noteSc(pen, cd, std::fabs(pen) + cd + 1.0f); // gate
        ro.wrote = pen < cd;
        taint[i] = sl.near ? 1 : 0;
        if (!taint[i])
        {
            if (ro.wrote) ++wroteN; else ++gateRejN;
        }
    }

    // ---- GPU ----
    PxgSlangComputeDevice* dev = testDevice();
    std::vector<F4> femF4Init(CAP, { -9, -9, -9, -9 });
    std::vector<U4> infoInit(CAP, { INFOPOISON, INFOPOISON, INFOPOISON, INFOPOISON });
    std::vector<uint32_t> u1Init(CAP, POISON), u2Init(CAP * 2, POISON), wc0(1, 0);
    std::vector<F4> dbgInit(NC, { -7, -7, -7, -7 });
    std::vector<uint32_t> filterRaw(filters.size() * 8);
    std::memcpy(filterRaw.data(), filters.data(), filters.size() * sizeof(FilterPair));
    PxgBufferHandle bC = createBufferFrom(dev, cases);
    PxgBufferHandle bH = createBufferFrom(dev, blob);
    PxgBufferHandle bF = createBufferFrom(dev, filterRaw);
    PxgBufferHandle bPt = createBufferFrom(dev, femF4Init);
    PxgBufferHandle bNp = createBufferFrom(dev, femF4Init);
    PxgBufferHandle bBc = createBufferFrom(dev, femF4Init);
    PxgBufferHandle bIn = createBufferFrom(dev, infoInit);
    PxgBufferHandle bWc = createBufferFrom(dev, wc0);
    PxgBufferHandle bBy = createBufferFrom(dev, u2Init);
    PxgBufferHandle bTb = createBufferFrom(dev, u1Init);
    PxgBufferHandle bIx = createBufferFrom(dev, u1Init);
    PxgBufferHandle bDb = createBufferFrom(dev, dbgInit);
    FvcParams prm = {};
    prm.numCases = NC; prm.maxContacts = CAP; prm.nbFilterPairs = (uint32_t)filters.size();
    prm.hullAddr = dev->getBufferDeviceAddress(bH);
    prm.wPoint = dev->getBufferDeviceAddress(bPt);
    prm.wNormalPen = dev->getBufferDeviceAddress(bNp);
    prm.wBary = dev->getBufferDeviceAddress(bBc);
    prm.wInfo = dev->getBufferDeviceAddress(bIn);
    prm.wCount = dev->getBufferDeviceAddress(bWc);
    prm.wByX = dev->getBufferDeviceAddress(bBy);
    prm.wTempByX = dev->getBufferDeviceAddress(bTb);
    prm.wIdxByX = dev->getBufferDeviceAddress(bIx);
    Kernel k("fcVertexConvex.spv", "ctFcVertexConvex",
             { (uint32_t)sizeof(FvcCase), (uint32_t)sizeof(FilterPair), 16 },
             sizeof(FvcParams));
    k.dispatch((NC + 63) / 64, { bC, bF, bDb }, prm);
    dev->flush();
    const auto gDbg = downloadBuffer<F4>(dev, bDb, NC);
    const auto gPt = downloadBuffer<F4>(dev, bPt, CAP);
    const auto gNp = downloadBuffer<F4>(dev, bNp, CAP);
    const auto gBc = downloadBuffer<F4>(dev, bBc, CAP);
    const auto gIn = downloadBuffer<U4>(dev, bIn, CAP);
    const auto gBy = downloadBuffer<uint32_t>(dev, bBy, CAP * 2);
    const auto gTb = downloadBuffer<uint32_t>(dev, bTb, CAP);
    const auto gIx = downloadBuffer<uint32_t>(dev, bIx, CAP);
    const uint32_t gWc = downloadBuffer<uint32_t>(dev, bWc, 1)[0];

    // filter verdicts are integer-exact: compare for ALL cases
    size_t mmF = 0;
    for (uint32_t i = 0; i < NC; ++i)
        if ((gDbg[i].x != 0.0f) != ref[i].filtered) ++mmF;
    PXG_CHECK_MSG(mmF == 0, "fcVertexConvex: %zu filter mismatches", mmF);

    // per-case slot map via pairInd1
    std::vector<int> slotOf(NC, -1);
    size_t mmW = 0;
    for (uint32_t s = 0; s < std::min(gWc, CAP); ++s)
    {
        const uint32_t caseIdx = gIn[s].z & 0xFFFFFu;
        if ((gIn[s].z >> 20) != CLOTH || caseIdx >= NC || slotOf[caseIdx] != -1) { ++mmW; continue; }
        slotOf[caseIdx] = (int)s;
        const uint32_t rl = cases[caseIdx].ids.x, rh = cases[caseIdx].ids.y;
        if (gIn[s].x != rl || gIn[s].y != rh ||
            gBy[s * 2] != rl || gBy[s * 2 + 1] != rh || gTb[s] != rl || gIx[s] != s)
            ++mmW;
        if (gIn[s].w != (cases[caseIdx].ids2.x & 0x7FFFFFFFu)) ++mmW;
    }
    uint32_t sumChecked = 0, numTainted = 0;
    size_t mm = 0, checked = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        if (ref[i].filtered)
        {
            if (slotOf[i] != -1) ++mm; // filtered must not write
            continue;
        }
        if (taint[i]) { ++numTainted; continue; }
        ++checked;
        const RefOut& ro = ref[i];
        // debug: pen + branch + write decision
        if ((gDbg[i].z != 0.0f) != ro.wrote || (gDbg[i].w != 0.0f) != ro.inside ||
            !almostEqual(gDbg[i].y, ro.pen, 2e-4f, 2e-4f)) { ++mm; continue; }
        if (!ro.wrote) { if (slotOf[i] != -1) ++mm; continue; }
        sumChecked += 1;
        if (slotOf[i] == -1) { ++mm; continue; }
        const uint32_t s = (uint32_t)slotOf[i];
        if (!almostEqual(gPt[s].x, ro.p.x, 2e-3f, 2e-3f) ||
            !almostEqual(gPt[s].y, ro.p.y, 2e-3f, 2e-3f) ||
            !almostEqual(gPt[s].z, ro.p.z, 2e-3f, 2e-3f) ||
            gPt[s].w != cases[i].rigidP.w ||
            !almostEqual(gNp[s].x, ro.n.x, 2e-3f, 2e-3f) ||
            !almostEqual(gNp[s].y, ro.n.y, 2e-3f, 2e-3f) ||
            !almostEqual(gNp[s].z, ro.n.z, 2e-3f, 2e-3f) ||
            !almostEqual(gNp[s].w, ro.pen, 2e-4f, 2e-4f) ||
            gBc[s].x != 0.0f || gBc[s].y != 0.0f || gBc[s].z != 0.0f || gBc[s].w != 1.0f)
            ++mm;
    }
    PXG_CHECK_MSG(mm == 0 && mmW == 0, "fcVertexConvex: %zu content + %zu slot mismatches", mm, mmW);
    PXG_CHECK_MSG(gWc >= sumChecked && gWc <= sumChecked + numTainted,
                  "fcVertexConvex: counter %u outside [%u, %u]", gWc, sumChecked, sumChecked + numTainted);
    PXG_CHECK_MSG(checked > NC * 3 / 5, "fcVertexConvex: too many tainted (%u)", numTainted);
    PXG_CHECK_MSG(filteredN > 100 && insideN > 80 && outsideEdgeN > 200 && gateRejN > 150 && wroteN > 300,
                  "fcVertexConvex: coverage thin (f=%zu i=%zu e=%zu r=%zu w=%zu)",
                  filteredN, insideN, outsideEdgeN, gateRejN, wroteN);

    for (PxgBufferHandle h : { bC, bH, bF, bPt, bNp, bBc, bIn, bWc, bBy, bTb, bIx, bDb })
        dev->destroyBuffer(h);
}

// ===========================================================================
// P9.21 -- cloth_convexVertexContactGenLaunch, the ENGINE-ABI kernel.
//
// WHAT THIS MEASURES, and what it deliberately does not.
//
// The closest-point core is the shared pxgSlangVertexConvexCloth module, which
// the battery above verifies against a C++ host mirror of the CUDA.  Running
// the engine kernel against that same module would measure nothing about the
// core -- both sides would be the same instructions -- so this test does not
// try to.  It measures the GLUE, which is the only part the battery above has
// no opinion about:
//
//   * the (cmIdx, VERTEX index) pair stack and its min(count, stackSize) clamp;
//   * LoadShapePair<eTRIANGLEMESH>, including the branch that fires when the
//     cloth is shape1 rather than shape0;
//   * the SINGLE-match attachment filter (W3);
//   * restDistance[cmIdx] vs contactDistance[cacheRef0] + contactDistance[cacheRef1];
//   * the cloth vertex load, the rigid pose and the hull address;
//   * the unclamped stackSizeNeeded atomicMax.
//
// The method is the P9.15d one: dispatch the engine kernel, mirror ONLY the
// glue on the host into a list of FvcCases, dispatch the Phase-1 entry point
// over those, and compare the two contact streams as multisets, BITWISE.  Any
// glue field the kernel reads from the wrong place lands in a different case
// and the streams diverge.
//
// Both shaders carry NoContraction (tests/CMakeLists.txt) for the reason P9.15d
// found the hard way: with only one of the two stamped, the SHARED module gets
// contracted differently in each and the differential measures the fp policy
// instead of the glue.
// ===========================================================================
namespace fvc2
{
using namespace physx;

struct V3c { float x, y, z; };
struct Qc { float x, y, z, w; };
static V3c qrotc(Qc q, V3c v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d2 = (q.x * vx + q.y * vy + q.z * vz);
    return V3c{ vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d2,
                vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d2,
                vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d2 };
}

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

// W1: byte-identical to the box-triangle blob -- same fifteen arguments, same
// order, same ABI row.  The writer sits at 112 because paramAlign(80) is 16.
struct CvxVertPush
{
    float toleranceLength; uint32_t pad4;
    uint64_t cmInputs, transformCache, contactDistance, restDistance, gpuShapes, femClothes;
    uint64_t shapeToRigidRemap, stackPtr, midphasePairsNum, filterPairs;
    uint32_t nbFilterPairs, stackSizeBytes;
    uint64_t stackSizeNeeded;
    uint64_t pad104;
    uint64_t bPoint, bNormalPen, bBary, bInfo, bCount, bByX, bTempByX, bIdxByX, bSortedByX;
    uint32_t bMaxNumContacts, pad188;
};
static_assert(sizeof(CvxVertPush) == 192, "cloth convex-vertex blob must be 192 bytes");
static_assert(offsetof(CvxVertPush, stackSizeNeeded) == 96, "stackSizeNeeded @96");
static_assert(offsetof(CvxVertPush, bPoint) == 112, "writer @112 -- paramAlign(80) is 16");
} // namespace fvc2

PXG_TEST(femClothSimConvexVertexContactGen)
{
    using namespace fvc2;
    PxgSlangComputeDevice* dev = testDevice();

    // ---- the rigid: an irregular hex prism under a NON-UNIFORM, ROTATED scale ----
    // W5 needs the scale to be non-uniform: with a uniform scale the point on the
    // convex and `clothP - pen * normal` agree, and the two constructions become
    // indistinguishable.  The scale ROTATION is non-identity for the same reason --
    // it is the frame the non-uniformity lives in.
    const float radii[6] = { 1.00f, 1.35f, 0.80f, 1.20f, 0.95f, 1.45f };
    std::vector<uint32_t> blob;
    uint32_t hullOffset = 0;
    makePrism(6u, radii, 0.9f).serialize(blob, hullOffset);
    PXG_CHECK_MSG(hullOffset == 0u, "the hull must start at offset 0 for this rig");
    DevBuf bHull = mkBuf(dev, blob);

    const V3c rigidPos = { 0.35f, -0.20f, 0.10f };
    const float rh = std::sin(0.23f), rc = std::cos(0.23f);
    const Qc rigidQ = { rh * 0.6f, rh * 0.48f, rh * 0.64f, rc };
    const V3c cvxScale = { 1.30f, 0.70f, 1.05f };
    const float sh = std::sin(0.31f), sc = std::cos(0.31f);
    const Qc cvxScaleRot = { 0.0f, sh, 0.0f, sc };

    const float cdCloth = 0.18f, cdRigid = 0.34f;
    const float rest0 = 0.013f, rest1 = 0.047f;         // W4: per contact-manager pair
    const uint32_t materialIndex = 21u;
    const uint64_t rigidNodeInd = 0x0000000500000029ull;
    const uint32_t CLOTH_ID = 2u;

    // ---- the cloth: eleven vertices spread over the three outcomes ----
    // Deep inside (all-inside branch), just outside a side face, outside an edge,
    // and two far enough away that the pen < contactDist gate rejects them.
    std::vector<F4> cverts = {
        F4{ 0.35f, -0.20f,  0.10f, 1.0f },   // 0: the hull centre -- inside
        F4{ 0.60f, -0.10f,  0.30f, 1.0f },   // 1: inside, off-centre
        F4{ 1.95f, -0.15f,  0.20f, 1.0f },   // 2: just outside a side face
        F4{ 0.20f, -0.30f,  1.35f, 1.0f },   // 3: just outside the +z cap
        F4{ 1.70f, -0.10f,  1.20f, 1.0f },   // 4: outside a cap/side EDGE
        F4{ -1.35f, -0.25f, 0.05f, 1.0f },   // 5: just outside the far side
        // 6 and 10 sit in the two GATE BANDS -- see the band check below.  Their
        // exact heights were measured, not guessed: 6 lands at pen ~0.44 and 10
        // at pen ~0.60 against a summed contact distance of 0.52.
        F4{ 0.40f,  1.03f,  0.15f, 1.0f },   // 6: outside along +y, pen in (2*cd0, cdSum)
        F4{ 6.00f,  4.00f,  5.00f, 1.0f },   // 7: far -- gate reject
        F4{ -5.50f, -4.20f, -3.10f, 1.0f },  // 8: far -- gate reject
        F4{ 0.90f,  0.35f,  0.95f, 1.0f },   // 9: near a corner
        F4{ 0.05f, -0.55f, -1.61f, 1.0f },   // 10: -z cap, pen in (cdSum, 2*cd1)
    };
    const uint32_t NB_VERTS = uint32_t(cverts.size());
    DevBuf bCVerts = mkBuf(dev, cverts);
    // The triangle array is never read by this kernel; it exists so the descriptor
    // is a real PxgFEMCloth rather than a half-filled one.
    std::vector<U4> ctris = { U4{ 0u, 1u, 2u, 0u } };
    DevBuf bCTris = mkBuf(dev, ctris);

    PxgFEMCloth fc; std::memset(&fc, 0, sizeof(fc));
    fc.mPosition_InvMass = reinterpret_cast<float4*>(size_t(bCVerts.addr));
    fc.mTriangleVertexIndices = reinterpret_cast<uint4*>(size_t(bCTris.addr));
    fc.mNbVerts = NB_VERTS;
    fc.mNbTriangles = 1u;
    PxgFEMCloth decoy0; std::memset(&decoy0, 0, sizeof(decoy0));
    PxgFEMCloth decoy1; std::memset(&decoy1, 0, sizeof(decoy1));
    std::vector<PxgFEMCloth> descArr = { decoy0, decoy1, fc };   // CLOTH_ID == 2
    DevBuf bDesc = mkBuf(dev, descArr);

    // Two contact-manager pairs.  Pair 0 puts the cloth in shape0; pair 1 SWAPS
    // them, so LoadShapePair<eTRIANGLEMESH>'s reorder branch is live and the
    // cache refs must travel with their shapes.
    std::vector<PxgContactManagerInput> cmInputs(2);
    std::memset(cmInputs.data(), 0, cmInputs.size() * sizeof(PxgContactManagerInput));
    cmInputs[0].shapeRef0 = 0u; cmInputs[0].shapeRef1 = 1u;
    cmInputs[0].transformCacheRef0 = 0u; cmInputs[0].transformCacheRef1 = 1u;
    cmInputs[1].shapeRef0 = 1u; cmInputs[1].shapeRef1 = 0u;
    cmInputs[1].transformCacheRef0 = 1u; cmInputs[1].transformCacheRef1 = 0u;
    DevBuf bCin = mkBuf(dev, cmInputs);

    std::vector<PxgShape> shapes(2);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShape));
    shapes[0].type = PxU32(PxGeometryType::eTRIANGLEMESH);
    shapes[0].particleOrSoftbodyId = CLOTH_ID;
    shapes[1].type = PxU32(PxGeometryType::eCONVEXMESH);
    shapes[1].materialIndex = materialIndex;
    shapes[1].scale = PxMeshScale(PxVec3(cvxScale.x, cvxScale.y, cvxScale.z),
                                  PxQuat(cvxScaleRot.x, cvxScaleRot.y, cvxScaleRot.z, cvxScaleRot.w));
    shapes[1].hullOrMeshPtr = size_t(bHull.addr);
    DevBuf bShapes = mkBuf(dev, shapes);

    // The CLOTH's transform cache entry is never read by this kernel -- the cloth
    // vertices are already world-space (the CUDA loads them raw).  It is filled
    // with a NON-identity pose anyway, so a port that "restores the symmetry" by
    // transforming the cloth is visible rather than a no-op.
    std::vector<PxsCachedTransform> tcache(2);
    std::memset(tcache.data(), 0, tcache.size() * sizeof(PxsCachedTransform));
    tcache[0].transform = PxTransform(PxVec3(-0.7f, 0.9f, 0.4f),
                                      PxQuat(0.0f, std::sin(0.4f), 0.0f, std::cos(0.4f)));
    tcache[1].transform = PxTransform(PxVec3(rigidPos.x, rigidPos.y, rigidPos.z),
                                      PxQuat(rigidQ.x, rigidQ.y, rigidQ.z, rigidQ.w));
    DevBuf bTc = mkBuf(dev, tcache);

    DevBuf bCd = mkBuf(dev, std::vector<float>{ cdCloth, cdRigid });
    DevBuf bRd = mkBuf(dev, std::vector<float>{ rest0, rest1 });
    DevBuf bRemap = mkBuf(dev, std::vector<uint64_t>{ 0ull, rigidNodeInd });

    // W3: a SINGLE match suppresses the vertex.  Two vertices are filtered, and
    // both are ones that would otherwise write; a near-miss entry on the SAME
    // rigid with an adjacent mask, and one on a DIFFERENT rigid with a live mask,
    // must not suppress anything.
    const uint32_t FILT_A = 2u, FILT_B = 5u;
    std::vector<PxgRigidFilterPair> fpairs;
    auto addFilter = [&](uint64_t rigid, uint32_t vert) {
        PxgRigidFilterPair fp; std::memset(&fp, 0, sizeof(fp));
        fp.index0 = rigid;
        fp.index1 = uint64_t((CLOTH_ID << 20) | vert);
        fpairs.push_back(fp);
    };
    addFilter(rigidNodeInd, FILT_A);
    addFilter(rigidNodeInd, FILT_B);
    addFilter(rigidNodeInd, NB_VERTS + 3u);              // adjacent mask, no such vertex
    addFilter(rigidNodeInd ^ 0x10ull, 1u);               // live mask, wrong rigid
    std::sort(fpairs.begin(), fpairs.end(), [](const PxgRigidFilterPair& a, const PxgRigidFilterPair& b) {
        return a.index0 != b.index0 ? a.index0 < b.index0 : a.index1 < b.index1; });
    DevBuf bFilters = mkBuf(dev, fpairs);

    // ---- the midphase pair stack: (cmIdx, VERTEX index) ----
    // W2 is only visible if the vertex index cannot be reconstructed from
    // anything else the loop has, so the table below is written out rather than
    // generated, and the three properties it has to have are CHECKED rather than
    // asserted in a comment:
    //   * every vertex appears exactly ONCE -- which is what makes the
    //     pairInd1 = (clothId << 20) | vertexIdx slot map injective;
    //   * NO slot maps to its own index -- so reading the loop counter, or
    //     curPair.x, in place of curPair.y changes every contact;
    //   * cmIdx is neither the slot's parity nor the vertex's -- so W4's
    //     restDistance[cmIdx] cannot be satisfied by either of them.
    // An arithmetic permutation `(a*t + b) mod p` cannot have the second
    // property unless a == 1, and a == 1 is a shift, which leaves the visited
    // SET unchanged -- and this test compares the streams as a multiset.
    static const uint32_t STACK_VERT[11] = { 4, 9, 0, 6, 10, 2, 7, 1, 5, 3, 8 };
    static const uint32_t STACK_CM[11]   = { 0, 1, 1, 0,  1, 0, 0, 1, 0, 1, 1 };
    PXG_CHECK_MSG(NB_VERTS == 11u, "the stack table is written for 11 vertices, not %u", NB_VERTS);
    std::vector<U4> stack;
    for (uint32_t t = 0; t < NB_VERTS; ++t)
        stack.push_back(U4{ STACK_CM[t], STACK_VERT[t], 0u, 0u });
    {
        std::vector<uint32_t> seen(NB_VERTS, 0u);
        uint32_t fixedPoints = 0, cmIsSlotParity = 0, cmIsVertParity = 0;
        for (uint32_t t = 0; t < NB_VERTS; ++t)
        {
            ++seen[STACK_VERT[t]];
            if (STACK_VERT[t] == t) ++fixedPoints;
            if (STACK_CM[t] == (t & 1u)) ++cmIsSlotParity;
            if (STACK_CM[t] == (STACK_VERT[t] & 1u)) ++cmIsVertParity;
        }
        uint32_t missing = 0;
        for (uint32_t v = 0; v < NB_VERTS; ++v) if (seen[v] != 1u) ++missing;
        PXG_CHECK_MSG(missing == 0u, "the stack must visit every vertex exactly once (%u do not)",
                      missing);
        PXG_CHECK_MSG(fixedPoints == 0u,
                      "%u stack slots carry their own index as the vertex -- reading the loop "
                      "counter for curPair.y would be invisible there", fixedPoints);
        PXG_CHECK_MSG(cmIsSlotParity < NB_VERTS && cmIsVertParity < NB_VERTS,
                      "cmIdx must not be a function of the slot (%u/%u) or of the vertex (%u/%u)",
                      cmIsSlotParity, NB_VERTS, cmIsVertParity, NB_VERTS);
    }
    const uint32_t stackPairs = uint32_t(stack.size());
    // The counter OVERSHOOTS, and the buffer is padded past stackSizeBytes with
    // valid-but-extra work, so an unclamped consumer gets a wrong ANSWER rather
    // than a crash.
    const uint32_t OVERSHOOT = 3u;
    for (uint32_t k = 0; k < OVERSHOOT; ++k) stack.push_back(U4{ 0u, k, 0u, 0u });
    DevBuf bStack = mkBuf(dev, stack);
    DevBuf bPairsNum = mkBuf(dev, std::vector<uint32_t>{ stackPairs + OVERSHOOT });
    DevBuf bNeeded = mkBuf(dev, std::vector<uint32_t>{ 0u });

    const uint32_t CAP = 128u;
    WBufs wa = mkWriter(dev, CAP);

    CvxVertPush push = {};
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

    Kernel ka("femClothContactGenBda.spv", "cloth_convexVertexContactGenLaunch", {}, sizeof(CvxVertPush));
    PXG_CHECK_MSG(ka.isValid(), "the cloth convex-vertex pipeline must build");
    if (!ka.isValid()) { killWriter(dev, wa); return; }
    ka.dispatch(/*gridX*/ 4096u, {}, push, 1u, 1u);
    dev->flush();

    // ---- the host mirror of the GLUE, and the differential dispatch ----
    std::vector<FvcCase> cases;
    std::vector<uint32_t> caseCm;
    uint32_t skipped = 0;
    for (uint32_t i = 0; i < stackPairs; ++i)
    {
        const uint32_t cmIdx = stack[i].x, vertexIdx = stack[i].y;
        const uint64_t key = uint64_t((CLOTH_ID << 20) | vertexIdx);
        bool filtered = false;
        for (const auto& fp : fpairs)
            if (fp.index0 == rigidNodeInd && fp.index1 == key) filtered = true;
        if (filtered) { ++skipped; continue; }                     // W3

        FvcCase cs{};
        cs.clothP = F4{ cverts[vertexIdx].x, cverts[vertexIdx].y, cverts[vertexIdx].z,
                        cdCloth + cdRigid };                       // W4
        cs.rigidP = F4{ rigidPos.x, rigidPos.y, rigidPos.z, (cmIdx == 0u) ? rest0 : rest1 };
        cs.rigidQ = F4{ rigidQ.x, rigidQ.y, rigidQ.z, rigidQ.w };
        cs.convexScale = F4{ cvxScale.x, cvxScale.y, cvxScale.z, 0.0f };
        cs.convexScaleRot = F4{ cvxScaleRot.x, cvxScaleRot.y, cvxScaleRot.z, cvxScaleRot.w };
        cs.ids.x = uint32_t(rigidNodeInd & 0xffffffffu);
        cs.ids.y = uint32_t(rigidNodeInd >> 32);
        cs.ids.z = CLOTH_ID; cs.ids.w = vertexIdx;
        cs.ids2.x = materialIndex; cs.ids2.y = 0u;
        cases.push_back(cs);
        caseCm.push_back(cmIdx);
    }
    PXG_CHECK_MSG(skipped == 2u, "W3: exactly the two filtered vertices must be skipped, %u were",
                  skipped);

    DevBuf bCases = mkBuf(dev, cases);
    // The reference dispatch runs with an EMPTY filter list: the host mirror has
    // already applied the single-match skip, and handing it the real list would
    // let a reference-side filter bug cancel an engine-side one.
    DevBuf bNoFilter = mkBuf(dev, std::vector<uint32_t>(8, 0u));
    WBufs wb = mkWriter(dev, CAP);
    std::vector<F4> dbg(cases.size(), F4{ -7, -7, -7, -7 });
    DevBuf bDbg = mkBuf(dev, dbg);
    FvcParams fp2 = {};
    fp2.numCases = uint32_t(cases.size()); fp2.maxContacts = CAP; fp2.nbFilterPairs = 0u;
    fp2.hullAddr = bHull.addr;
    fp2.wPoint = wb.point.addr; fp2.wNormalPen = wb.normalPen.addr; fp2.wBary = wb.bary.addr;
    fp2.wInfo = wb.info.addr; fp2.wCount = wb.count.addr; fp2.wByX = wb.byX.addr;
    fp2.wTempByX = wb.tempByX.addr; fp2.wIdxByX = wb.idxByX.addr;
    Kernel kb("fcVertexConvex.spv", "ctFcVertexConvex",
              { (uint32_t)sizeof(FvcCase), (uint32_t)sizeof(FilterPair), 16 }, sizeof(FvcParams));
    PXG_CHECK_MSG(kb.isValid(), "the reference core pipeline must build");
    if (kb.isValid())
    {
        kb.dispatch((uint32_t(cases.size()) + 63u) / 64u, { bCases.h, bNoFilter.h, bDbg.h }, fp2);
        dev->flush();
    }

    // ---- the rig must actually reach all three outcomes ----
    const auto gDbg = downloadBuffer<F4>(dev, bDbg.h, cases.size());
    uint32_t insideN = 0, wroteN = 0, gateRejN = 0;
    for (size_t i = 0; i < cases.size(); ++i)
    {
        if (gDbg[i].w != 0.0f) ++insideN;
        if (gDbg[i].z != 0.0f) ++wroteN; else ++gateRejN;
    }
    PXG_CHECK_MSG(insideN >= 2u && wroteN >= 5u && gateRejN >= 2u,
                  "the rig must reach all three outcomes (inside=%u wrote=%u gateRej=%u) or a "
                  "glue mutation has nowhere to show", insideN, wroteN, gateRejN);

    // W4, THE HALF THE FIRST SWEEP DID NOT COVER.
    //
    // `w4-contactdist-by-cmidx` -- contactDistance[cmIdx] + contactDistance[cmIdx]
    // instead of contactDistance[clothCacheRef] + contactDistance[rigidCacheRef] --
    // SURVIVED the first sweep, and the reason is worth stating because it is a
    // property of the fixture, not of the kernel.  The contact distance is never
    // STORED in a contact; it only decides the `pen < contactDist` gate.  With both
    // cm pairs resolving to the same two cache refs, the correct sum is 0.52 for
    // every case, while the mutant gives 2*cd0 = 0.36 for cm-0 cases and 2*cd1 =
    // 0.68 for cm-1 ones.  Those are different NUMBERS, but no case's pen fell in
    // either band, so every gate verdict was unchanged and the mutant was
    // output-identical.
    //
    // Two vertices were therefore moved into the bands, and this check is what
    // keeps them there: an edit that drifts them out reopens the hole silently,
    // which is exactly how the hole got there.
    {
        const float cd0 = cdCloth, cd1 = cdRigid, cdSum = cdCloth + cdRigid;
        PXG_CHECK_MSG(cd0 < cd1, "the band argument needs the two contact distances to differ");
        uint32_t lowBand = 0, highBand = 0;
        for (size_t i = 0; i < cases.size(); ++i)
        {
            const float pen = gDbg[i].y;
            if (caseCm[i] == 0u && pen > 2.0f * cd0 && pen < cdSum) ++lowBand;
            if (caseCm[i] == 1u && pen > cdSum && pen < 2.0f * cd1) ++highBand;
        }
        PXG_CHECK_MSG(lowBand >= 1u && highBand >= 1u,
                      "the gate bands are empty (low=%u in (%g, %g) on cm 0, high=%u in (%g, %g) "
                      "on cm 1) -- without a case in each, reading contactDistance[cmIdx] twice "
                      "instead of summing the two cache refs changes no gate verdict and the "
                      "differential cannot see it",
                      lowBand, 2.0f * cd0, cdSum, highBand, cdSum, 2.0f * cd1);
    }

    // ---- compare the two contact streams as multisets, bitwise ----
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
                    f2u(yA[a].z) != f2u(yB[b].z) || f2u(yA[a].w) != f2u(yB[b].w)) continue;
                if (iA[a].pairInd0 != iB[b].pairInd0 || iA[a].pairInd1 != iB[b].pairInd1 ||
                    iA[a].rigidMatInd != iB[b].rigidMatInd) continue;
                used[b] = true; found = true;
            }
            if (!found)
            { if (mm < 4) std::printf("  cvxvert contact %u (pt %g,%g,%g w %g / np %g,%g,%g,%g / "
                                      "pair1 %u) has no match in the reference stream\n",
                                      a, pA[a].x, pA[a].y, pA[a].z, pA[a].w,
                                      nA[a].x, nA[a].y, nA[a].z, nA[a].w, iA[a].pairInd1);
              ++mm; }
        }
        PXG_CHECK_MSG(mm == 0, "cloth_convexVertexContactGen differs from the reference core over "
                      "the mirrored glue in %zu contacts", mm);

        // the (0,0,0,1) VERTEX marker -- the one field that distinguishes this
        // kernel's contacts from the triangle kernel's downstream
        size_t mmB = 0;
        for (uint32_t a = 0; a < n; ++a)
            if (yA[a].x != 0.0f || yA[a].y != 0.0f || yA[a].z != 0.0f || yA[a].w != 1.0f) ++mmB;
        PXG_CHECK_MSG(mmB == 0, "%zu contacts carry a barycentric other than (0,0,0,1) -- the 1 in "
                      "w is what marks a VERTEX contact downstream", mmB);
    }

    // W7: the atomicMax carries the UNCLAMPED count, in bytes.
    const uint32_t needed = downloadBuffer<uint32_t>(dev, bNeeded.h, 1)[0];
    PXG_CHECK_MSG(needed == (stackPairs + OVERSHOOT) * 16u,
                  "stackSizeNeeded reads %u, expected %u -- it must carry the UNCLAMPED pair "
                  "count times 16, which is how the engine grows next frame's stack",
                  needed, (stackPairs + OVERSHOOT) * 16u);

    std::printf("    [ note ] %u contacts over %u vertices / %u pairs (2 filtered, %u inside, "
                "%u gate-rejected); stackNeeded %u vs stack %u\n",
                cntA, NB_VERTS, stackPairs, insideN, gateRejN, needed, stackPairs * 16u);

    killWriter(dev, wa); killWriter(dev, wb);
    dev->destroyBuffer(bHull.h); dev->destroyBuffer(bCVerts.h); dev->destroyBuffer(bCTris.h);
    dev->destroyBuffer(bDesc.h); dev->destroyBuffer(bCin.h); dev->destroyBuffer(bShapes.h);
    dev->destroyBuffer(bTc.h); dev->destroyBuffer(bCd.h); dev->destroyBuffer(bRd.h);
    dev->destroyBuffer(bRemap.h); dev->destroyBuffer(bFilters.h); dev->destroyBuffer(bStack.h);
    dev->destroyBuffer(bPairsNum.h); dev->destroyBuffer(bNeeded.h);
    dev->destroyBuffer(bCases.h); dev->destroyBuffer(bNoFilter.h); dev->destroyBuffer(bDbg.h);
}

} // namespace
