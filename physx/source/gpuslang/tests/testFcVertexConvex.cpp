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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };
struct FvcCase { F4 clothP, rigidP, rigidQ, convexScale, convexScaleRot; U4 ids, ids2; };
struct FvcParams { uint32_t numCases, maxContacts, nbFilterPairs, pad0; };
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
    FvcParams prm = { NC, CAP, (uint32_t)filters.size(), 0 };
    Kernel k("fcVertexConvex.spv", "ctFcVertexConvex",
             { (uint32_t)sizeof(FvcCase), 4, (uint32_t)sizeof(FilterPair),
               16, 16, 16, 16, 4, 8, 4, 4, 16 }, sizeof(FvcParams));
    k.dispatch((NC + 63) / 64, { bC, bH, bF, bPt, bNp, bBc, bIn, bWc, bBy, bTb, bIx, bDb }, prm);
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

} // namespace
