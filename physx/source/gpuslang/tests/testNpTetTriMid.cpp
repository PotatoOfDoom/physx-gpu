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

// Verification of npTetTriMid.slang (TriangleLeafBoundMinMaxTraverser + the
// tet-vs-triangle satIntersect and the driver's top-4 pair extraction from
// softbodySoftbodyMidPhase.cu).  One query tetrahedron per 32-thread group
// walks softbody-vs-trimesh BV32 with the AABB-displacement box callback;
// per lane: triangle plane, OUTER front-face gate on the tet centroid, then
// the plane-only SAT (tri plane term + 4 tet planes, max of min projections)
// with the EVOLVING early-out tolerance min(bestDistance, contactDistance);
// finally up to 4 winners per query are extracted by repeated warp argmin
// (lowest lane wins ties) behind ONE atomicAdd of the qualifying count.
//
// Verification compares PAIR SETS per query (order-free: the extraction
// order is tie-dependent, membership is not, except at the count cutoff
// which is slack-noted).  Slack machinery follows testNpSbSelf.cpp:
// error-scaled margins, per-triangle slack promoted only when the triangle
// is result-relevant (its separation - or the separation it would have had
// were it gate-blocked within rounding - lies below contactDistance + pad).
// The early-out chain is provably output-neutral away from comparison
// boundaries: the chain is non-decreasing, so a triangle whose full
// separation qualifies never early-outs (sep_k <= sep_full < contactDist <=
// tolerance); early-outed partial values CAN be stored as a lane best while
// the best is still above contactDistance, but such bests never qualify in
// the extraction ballot and leave the clamped tolerance
// min(best, contactDistance) = contactDistance unchanged.
// Invalid (flipped/degenerate) tets must produce no pairs at all.

#include "pxgTestFramework.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <random>
#include <set>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };
struct Bv32Node
{
    F4 mMin[32];
    F4 mMax[32];
    uint32_t mData[32];
    uint32_t mNbNodes, mDepth, pad0, pad1;
};
struct TtWork { F4 v0, v1, v2, v3, params; };
struct TtParams { uint32_t numWorkItems, maxPairs, pad0, pad1; };

struct V3 { float x, y, z; };
static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 add(V3 a, V3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 muls(V3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
static V3 neg(V3 a) { return { -a.x, -a.y, -a.z }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static V3 v3(F4 f) { return { f.x, f.y, f.z }; }
static F4 f4(V3 v) { return { v.x, v.y, v.z, 0.0f }; }

static const float MAXF = 3.4028235e38f;

struct Slack { bool near = false; void noteSc(float l, float r, float scale) { if (std::fabs(l - r) < 1e-5f * scale) near = true; } };

static float l1n(V3 v) { return std::fabs(v.x) + std::fabs(v.y) + std::fabs(v.z); }

static V3 getNormalized(V3 v)
{
    const float m = dot3(v, v);
    return m > 0 ? muls(v, 1.0f / std::sqrt(m)) : V3{ 0, 0, 0 };
}
static bool refValidTet(V3 a, V3 b, V3 c, V3 d) { return dot3(sub(b, a), cross3(sub(c, a), sub(d, a))) > 1e-9f; }

// PxPlane(p0,p1,p2); returns the normalize blow factor for slack scales
static float makePlane(V3 p0, V3 p1, V3 p2, V3& n, float& d)
{
    const V3 e0 = sub(p1, p0), e1 = sub(p2, p0);
    const V3 cn = cross3(e0, e1);
    n = getNormalized(cn);
    d = -dot3(p0, n);
    return l1n(e0) * l1n(e1) / (std::sqrt(dot3(cn, cn)) + 1e-30f);
}

static float createPlane(V3 pa, V3 pb, V3 pc, V3 pd, V3& n, float& d)
{
    const float blow = makePlane(pa, pb, pc, n, d);
    const float distance = dot3(n, pd) + d;
    if (distance > 0.0f) { n = neg(n); d = -d; }
    return blow;
}

static float minProject3(V3 n, float d, V3 v0, V3 v1, V3 v2)
{
    return std::min(dot3(n, v0) + d, std::min(dot3(n, v1) + d, dot3(n, v2) + d));
}

// satIntersect (plane-only tet-vs-tri), slack-noting each early-out and the
// internal (caller-dead) back-face cull.  Mirrors the shader expression-for-
// expression.
// skipCull: the gate-blocked promotion probe must bypass the internal cull —
// it evaluates what a within-rounding gate flip WOULD contribute, and the
// cull recomputes the very expression that just blocked the caller.
static float refSat(const V3 pn[4], const float pd[4], V3 centroid,
                    V3 v0, V3 v1, V3 v2, float tolerance, Slack& sl, float blowT,
                    bool skipCull = false)
{
    V3 n;
    float d;
    const float blow = makePlane(v0, v1, v2, n, d);
    const float scale = (l1n(centroid) + l1n(v0)) * (1.0f + std::max(blow, blowT)) + 1.0f;
    if (!skipCull && dot3(n, centroid) + d < 0.0f)
        return MAXF;

    float sep = minProject3(n, d, v0, v1, v2);
    sl.noteSc(sep, tolerance, scale);
    if (sep > tolerance) return sep;
    for (int k = 0; k < 4; ++k)
    {
        sep = std::max(sep, minProject3(pn[k], pd[k], v0, v1, v2));
        sl.noteSc(sep, tolerance, scale);
        if (sep > tolerance) return sep;
    }
    return sep;
}

// ---- builder ----------------------------------------------------------
struct AABB { float mn[3], mx[3]; };
static AABB merge(const AABB& a, const AABB& b)
{
    AABB r;
    for (int k = 0; k < 3; ++k) { r.mn[k] = std::min(a.mn[k], b.mn[k]); r.mx[k] = std::max(a.mx[k], b.mx[k]); }
    return r;
}
static AABB rangeBounds(const std::vector<AABB>& prims, uint32_t begin, uint32_t end)
{
    AABB r = prims[begin];
    for (uint32_t i = begin + 1; i < end; ++i) r = merge(r, prims[i]);
    return r;
}
static uint32_t buildNode(std::vector<Bv32Node>& nodes, const std::vector<AABB>& prims,
                          uint32_t begin, uint32_t end, std::mt19937& rng, int depth)
{
    const uint32_t nodeIdx = (uint32_t)nodes.size();
    nodes.push_back(Bv32Node());
    const uint32_t count = end - begin;
    const uint32_t nbChunks = std::min<uint32_t>(2 + rng() % 5, count);
    std::vector<uint32_t> cuts = { begin, end };
    while (cuts.size() < nbChunks + 1)
        cuts.push_back(begin + 1 + rng() % (count - 1));
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
    uint32_t slot = 0;
    for (size_t c = 0; c + 1 < cuts.size(); ++c, ++slot)
    {
        const uint32_t b = cuts[c], e = cuts[c + 1];
        const AABB bounds = rangeBounds(prims, b, e);
        if (e - b <= 12 || (depth >= 4 && e - b <= 32))
            nodes[nodeIdx].mData[slot] = (b << 7) | ((e - b) << 1) | 1;
        else
        {
            const uint32_t child = buildNode(nodes, prims, b, e, rng, depth + 1);
            nodes[nodeIdx].mData[slot] = child << 11;
        }
        nodes[nodeIdx].mMin[slot] = { bounds.mn[0], bounds.mn[1], bounds.mn[2], 0.0f };
        nodes[nodeIdx].mMax[slot] = { bounds.mx[0], bounds.mx[1], bounds.mx[2], 0.0f };
    }
    nodes[nodeIdx].mNbNodes = slot;
    return nodeIdx;
}

PXG_TEST(npTetTriMid_traverser)
{
    std::mt19937 rng(86011);
    std::uniform_real_distribution<float> uv(-1, 1);
    std::uniform_real_distribution<float> u01(0, 1);

    // triangle mesh: 250 local triangles, 3 dedicated verts each
    const uint32_t NR = 250, NQ = 96;
    std::vector<F4> verts(NR * 3);
    std::vector<U4> tris(NR);
    std::vector<AABB> bounds(NR);
    for (uint32_t t = 0; t < NR; ++t)
    {
        const V3 c0 = { 1.2f * uv(rng), 1.2f * uv(rng), 1.2f * uv(rng) };
        for (;;)
        {
            for (int k = 0; k < 3; ++k)
                verts[t * 3 + k] = { c0.x + 0.12f * uv(rng), c0.y + 0.12f * uv(rng), c0.z + 0.12f * uv(rng), 0 };
            const V3 e0 = sub(v3(verts[t * 3 + 1]), v3(verts[t * 3]));
            const V3 e1 = sub(v3(verts[t * 3 + 2]), v3(verts[t * 3]));
            const V3 cn = cross3(e0, e1);
            // healthy triangles only: normalize blow factor < 3 keeps the
            // error-scaled slack margins tight
            if (std::sqrt(dot3(cn, cn)) > l1n(e0) * l1n(e1) / 3.0f) break;
        }
        tris[t] = { t * 3, t * 3 + 1, t * 3 + 2, 0 };
        AABB bb = { { 1e30f, 1e30f, 1e30f }, { -1e30f, -1e30f, -1e30f } };
        for (int k = 0; k < 3; ++k)
        {
            const float* p = &verts[t * 3 + k].x;
            for (int j = 0; j < 3; ++j) { bb.mn[j] = std::min(bb.mn[j], p[j]); bb.mx[j] = std::max(bb.mx[j], p[j]); }
        }
        bounds[t] = bb;
    }

    // Morton sort
    {
        auto expand = [](uint32_t v) {
            v &= 0x3ff; v = (v | (v << 16)) & 0x30000ff; v = (v | (v << 8)) & 0x300f00f;
            v = (v | (v << 4)) & 0x30c30c3; v = (v | (v << 2)) & 0x9249249; return v;
        };
        std::vector<uint32_t> order(NR);
        std::vector<uint64_t> code(NR);
        for (uint32_t t = 0; t < NR; ++t)
        {
            order[t] = t;
            uint32_t m[3];
            for (int k = 0; k < 3; ++k)
            {
                const float c = 0.5f * (bounds[t].mn[k] + bounds[t].mx[k]);
                m[k] = (uint32_t)std::min(1023.0f, std::max(0.0f, (c + 1.5f) / 3.0f * 1023.0f));
            }
            code[t] = ((uint64_t)expand(m[0]) << 2) | ((uint64_t)expand(m[1]) << 1) | expand(m[2]);
        }
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return code[a] < code[b]; });
        std::vector<U4> t2(NR); std::vector<AABB> b2(NR);
        for (uint32_t t = 0; t < NR; ++t) { t2[t] = tris[order[t]]; b2[t] = bounds[order[t]]; }
        tris.swap(t2); bounds.swap(b2);
    }
    std::vector<Bv32Node> nodes;
    buildNode(nodes, bounds, 0, NR, rng, 0);

    // query tets: overlapping / near / far placements + invalid fixtures
    std::vector<TtWork> work(NQ);
    size_t invalidCount = 0;
    for (uint32_t q = 0; q < NQ; ++q)
    {
        TtWork& w = work[q];
        // q%16==2: wide contact distance so >4 triangles qualify and the
        // MaxContacts clamp truncates (separations distinct -> stable sets)
        const float contactDist = (q % 16 == 2) ? 0.35f : 0.06f + 0.06f * u01(rng);
        w.params = { contactDist, 0, 0, 0 };
        V3 cen;
        if (q % 3 == 2)
            cen = { 1.3f * uv(rng), 1.3f * uv(rng), 1.3f * uv(rng) };
        else
        {
            // anchor near a triangle: overlapping (q%6==0, kept a minority -
            // penetrating separations tie at ~0 across triangles, a genuine
            // instability the tracker must skip) or separated within
            // ~contactDist of the mesh surface
            const uint32_t t = (q * 89) % NR;
            V3 tc = { 0, 0, 0 };
            for (int k = 0; k < 3; ++k) tc = add(tc, v3(verts[tris[t].x + k]));
            tc = muls(tc, 1.0f / 3.0f);
            const V3 dir = getNormalized(V3{ uv(rng), uv(rng), uv(rng) });
            const float off = (q % 6 == 0) ? 0.02f * u01(rng) : 0.24f + 0.6f * contactDist * u01(rng);
            cen = add(tc, muls(dir, off));
        }
        V3 t0, t1, t2, t3;
        for (;;)
        {
            t0 = add(cen, V3{ 0.10f * uv(rng), 0.10f * uv(rng), 0.10f * uv(rng) });
            t1 = add(cen, V3{ 0.10f * uv(rng), 0.10f * uv(rng), 0.10f * uv(rng) });
            t2 = add(cen, V3{ 0.10f * uv(rng), 0.10f * uv(rng), 0.10f * uv(rng) });
            t3 = add(cen, V3{ 0.10f * uv(rng), 0.10f * uv(rng), 0.10f * uv(rng) });
            const float det = dot3(sub(t1, t0), cross3(sub(t2, t0), sub(t3, t0)));
            if (det <= 2e-4f) continue; // healthy volume, well clear of 1e-9
            V3 n;
            float d;
            float bmax = 0.0f;
            bmax = std::max(bmax, createPlane(t0, t1, t2, t3, n, d));
            bmax = std::max(bmax, createPlane(t0, t1, t3, t2, n, d));
            bmax = std::max(bmax, createPlane(t0, t2, t3, t1, n, d));
            bmax = std::max(bmax, createPlane(t1, t2, t3, t0, n, d));
            if (bmax < 4.0f) break; // non-sliver faces -> bounded blow
        }
        if (q % 13 == 7)
        {
            // invalid fixture: flip the winding so the determinant is
            // strongly NEGATIVE (well clear of the 1e-9 threshold)
            std::swap(t0, t1);
            ++invalidCount;
        }
        w.v0 = f4(t0); w.v1 = f4(t1); w.v2 = f4(t2); w.v3 = f4(t3);
    }

    // ---- reference ----
    const uint32_t MaxContacts = 4;
    std::vector<std::set<uint32_t>> refPairs(NQ);
    std::vector<uint8_t> skip(NQ, 0);
    size_t sumPairs = 0, gateBlocked = 0, qWithPairs = 0, qTruncated = 0, penetrating = 0;

    for (uint32_t q = 0; q < NQ; ++q)
    {
        const TtWork& w = work[q];
        const V3 t0 = v3(w.v0), t1 = v3(w.v1), t2 = v3(w.v2), t3 = v3(w.v3);
        const float contactDist = w.params.x;
        Slack sl;

        {
            // validity gate: exact for these fixtures (valid tets are
            // regenerated until det > 1e-9 with det typically ~1e-3;
            // invalid fixtures have det strongly negative), but note it
            const float det = dot3(sub(t1, t0), cross3(sub(t2, t0), sub(t3, t0)));
            if (std::fabs(det - 1e-9f) < 3e-5f * (l1n(sub(t1, t0)) * l1n(sub(t2, t0)) * l1n(sub(t3, t0))))
                sl.near = true;
        }
        if (!refValidTet(t0, t1, t2, t3)) { skip[q] = sl.near ? 1 : 0; continue; }

        const V3 tMin = { std::min(std::min(t0.x, t1.x), std::min(t2.x, t3.x)),
                          std::min(std::min(t0.y, t1.y), std::min(t2.y, t3.y)),
                          std::min(std::min(t0.z, t1.z), std::min(t2.z, t3.z)) };
        const V3 tMax = { std::max(std::max(t0.x, t1.x), std::max(t2.x, t3.x)),
                          std::max(std::max(t0.y, t1.y), std::max(t2.y, t3.y)),
                          std::max(std::max(t0.z, t1.z), std::max(t2.z, t3.z)) };

        V3 pn[4];
        float pd[4];
        float blowT = 0.0f;
        blowT = std::max(blowT, createPlane(t0, t1, t2, t3, pn[0], pd[0]));
        blowT = std::max(blowT, createPlane(t0, t1, t3, t2, pn[1], pd[1]));
        blowT = std::max(blowT, createPlane(t0, t2, t3, t1, pn[2], pd[2]));
        blowT = std::max(blowT, createPlane(t1, t2, t3, t0, pn[3], pd[3]));
        const V3 centroid = muls(add(add(t0, t1), t2), 1.0f / 3.0f);

        // slack-tracked traversal (AABB displacement callback)
        std::vector<std::pair<uint32_t, uint32_t>> rounds;
        {
            std::vector<uint32_t> stack;
            uint32_t cur = 0;
            bool hasNode = true;
            while (hasNode)
            {
                const Bv32Node& node = nodes[cur >> 6];
                if (cur & 1)
                {
                    const uint32_t data = node.mData[(cur >> 1) & 31];
                    rounds.push_back({ data >> 7, (data >> 1) & 63 });
                }
                else
                {
                    for (uint32_t i = 0; i < node.mNbNodes; ++i)
                    {
                        V3 disp;
                        const float* mn = &node.mMin[i].x;
                        const float* mx = &node.mMax[i].x;
                        const float* tmn = &tMin.x;
                        const float* tmx = &tMax.x;
                        float* dp = &disp.x;
                        for (int k = 0; k < 3; ++k)
                            dp[k] = (mn[k] > tmx[k]) ? (mn[k] - tmx[k]) : ((mx[k] < tmn[k]) ? (mx[k] - tmn[k]) : 0.0f);
                        const float d2 = dot3(disp, disp);
                        const float dd = contactDist * contactDist;
                        sl.noteSc(d2, dd, l1n(disp) * (l1n(tMin) + l1n(tMax) + 6.0f) + d2 + dd);
                        if (!(d2 < dd)) continue;
                        const uint32_t data = node.mData[i];
                        stack.push_back((data & 1) ? ((cur & 0xFFFFFFC0u) | (i << 1) | 1) : ((data >> 11) << 6));
                    }
                }
                if (!stack.empty()) { cur = stack.back(); stack.pop_back(); }
                else hasNode = false;
            }
        }

        // per-lane traverser
        float laneBest[32];
        uint32_t lanePrim[32];
        for (int j = 0; j < 32; ++j) { laneBest[j] = MAXF; lanePrim[j] = 0xffffffff; }

        const float PAD = 0.02f;
        for (const auto& rd : rounds)
        {
            for (uint32_t i = 0; i < rd.second && i < 32; ++i)
            {
                const uint32_t t = rd.first + i;
                const V3 v0 = v3(verts[tris[t].x]), v1 = v3(verts[tris[t].y]), v2 = v3(verts[tris[t].z]);

                Slack triSl;
                V3 tn;
                float td;
                const float blow = makePlane(v0, v1, v2, tn, td);
                const float gd = dot3(tn, centroid) + td;
                triSl.noteSc(gd, 0.0f, (l1n(centroid) + l1n(v0)) * (1.0f + blow) + 1e-3f);

                float sepEval = MAXF;
                if (gd >= 0.0f)
                {
                    const float tolerance = std::min(laneBest[i], contactDist);
                    const float sep = refSat(pn, pd, centroid, v0, v1, v2, tolerance, triSl, blowT);
                    sepEval = sep;
                    if (sep < 1e-3f) ++penetrating;
                    if (std::min(sep, laneBest[i]) < contactDist + PAD)
                        triSl.noteSc(sep, laneBest[i], 8.0f * (1.0f + blowT));
                    if (sep < laneBest[i])
                    {
                        laneBest[i] = sep;
                        lanePrim[i] = t;
                    }
                }
                else
                {
                    ++gateBlocked;
                    if (triSl.near)
                    {
                        // gate within rounding: evaluate what the flipped-in
                        // triangle would contribute to decide promotion
                        Slack dummy;
                        sepEval = refSat(pn, pd, centroid, v0, v1, v2, contactDist, dummy, blowT, true);
                    }
                }

                // promote only when the triangle itself is result-relevant
                if (triSl.near && sepEval < contactDist + PAD)
                    sl.near = true;
            }
        }

        // driver-tail extraction: min(qualifying, 4) smallest bests
        std::vector<std::pair<float, uint32_t>> qual; // (best, lane)
        for (uint32_t j = 0; j < 32; ++j)
        {
            if (laneBest[j] < MAXF)
                sl.noteSc(laneBest[j], contactDist, 8.0f * (1.0f + blowT));
            if (laneBest[j] < contactDist)
                qual.push_back({ laneBest[j], j });
        }
        std::sort(qual.begin(), qual.end()); // (value, lane) — lowest lane wins ties
        const uint32_t count = std::min<uint32_t>((uint32_t)qual.size(), MaxContacts);
        if (qual.size() > MaxContacts)
        {
            ++qTruncated;
            sl.noteSc(qual[MaxContacts - 1].first, qual[MaxContacts].first, 8.0f * (1.0f + blowT));
        }
        for (uint32_t j = 0; j < count; ++j)
            refPairs[q].insert(lanePrim[qual[j].second]);
        sumPairs += count;
        if (count) ++qWithPairs;
        skip[q] = sl.near ? 1 : 0;
    }

    // ---- GPU run ----
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t CAP = NQ * 4 + 64;
    std::vector<U4> pairInit(CAP, { 0xffffffff, 0xffffffff, 0, 0 });
    std::vector<uint32_t> cnt(1, 0);
    PxgBufferHandle bN = createBufferFrom(dev, nodes);
    PxgBufferHandle bT = createBufferFrom(dev, tris);
    PxgBufferHandle bV = createBufferFrom(dev, verts);
    PxgBufferHandle bW = createBufferFrom(dev, work);
    PxgBufferHandle bP = createBufferFrom(dev, pairInit);
    PxgBufferHandle bC = createBufferFrom(dev, cnt);

    TtParams prm = { NQ, CAP, 0, 0 };
    Kernel k("npTetTriMid.spv", "ctTetTriMid",
             { (uint32_t)sizeof(Bv32Node), 16, 16, (uint32_t)sizeof(TtWork), 16, 4 },
             sizeof(TtParams));
    k.dispatch(NQ, { bN, bT, bV, bW, bP, bC }, prm);
    dev->flush();

    const auto gP = downloadBuffer<U4>(dev, bP, CAP);
    const uint32_t gCount = downloadBuffer<uint32_t>(dev, bC, 1)[0];
    dev->destroyBuffer(bN); dev->destroyBuffer(bT); dev->destroyBuffer(bV);
    dev->destroyBuffer(bW); dev->destroyBuffer(bP); dev->destroyBuffer(bC);

    std::vector<std::set<uint32_t>> gpuPairs(NQ);
    size_t bogus = 0;
    for (uint32_t i = 0; i < gCount && i < CAP; ++i)
    {
        if (gP[i].x < NQ) gpuPairs[gP[i].x].insert(gP[i].y);
        else ++bogus;
    }

    size_t mm = 0, checked = 0, skipped = 0, sumNonSkipped = 0, checkedMulti3 = 0;
    for (uint32_t q = 0; q < NQ; ++q)
    {
        if (skip[q]) { ++skipped; continue; }
        ++checked;
        sumNonSkipped += refPairs[q].size();
        if (refPairs[q].size() >= 3) ++checkedMulti3;
        if (gpuPairs[q] != refPairs[q]) ++mm;
    }
    PXG_CHECK_MSG(bogus == 0, "npTetTriMid: %zu pairs with out-of-range workId", bogus);
    PXG_CHECK_MSG(mm == 0, "npTetTriMid: %zu pair-set mismatches", mm);
    PXG_CHECK_MSG(gCount >= sumNonSkipped && gCount <= sumNonSkipped + 4 * skipped,
                  "npTetTriMid: counter %u outside [%zu, %zu]", gCount, sumNonSkipped, sumNonSkipped + 4 * skipped);
    PXG_CHECK_MSG(checked > NQ * 2 / 3, "npTetTriMid: too many skipped (%zu)", skipped);
    PXG_CHECK_MSG(qWithPairs > 15 && qTruncated >= 1 && checkedMulti3 >= 2 && gateBlocked > 100 && penetrating > 5 && invalidCount >= 4,
                  "npTetTriMid: coverage thin (qp=%zu trunc=%zu m3=%zu gate=%zu pen=%zu inv=%zu)",
                  qWithPairs, qTruncated, checkedMulti3, gateBlocked, penetrating, invalidCount);
    // invalid tets must contribute nothing
    size_t invPairs = 0;
    for (uint32_t q = 0; q < NQ; ++q)
        if (q % 13 == 7) invPairs += gpuPairs[q].size();
    PXG_CHECK_MSG(invPairs == 0, "npTetTriMid: invalid tets produced %zu pairs", invPairs);

    // capacity-overflow dispatch: the counter legitimately overshoots (the
    // CUDA atomicAdd happens before the capacity gate) while out-of-range
    // slots stay untouched
    {
        const uint32_t TINY = 3;
        std::vector<U4> pairInit2(CAP, { 0xffffffff, 0xffffffff, 0, 0 });
        std::vector<uint32_t> cnt2(1, 0);
        PxgBufferHandle bN2 = createBufferFrom(dev, nodes);
        PxgBufferHandle bT2 = createBufferFrom(dev, tris);
        PxgBufferHandle bV2 = createBufferFrom(dev, verts);
        PxgBufferHandle bW2 = createBufferFrom(dev, work);
        PxgBufferHandle bP2 = createBufferFrom(dev, pairInit2);
        PxgBufferHandle bC2 = createBufferFrom(dev, cnt2);
        TtParams prm2 = { NQ, TINY, 0, 0 };
        Kernel k2("npTetTriMid.spv", "ctTetTriMid",
                  { (uint32_t)sizeof(Bv32Node), 16, 16, (uint32_t)sizeof(TtWork), 16, 4 },
                  sizeof(TtParams));
        k2.dispatch(NQ, { bN2, bT2, bV2, bW2, bP2, bC2 }, prm2);
        dev->flush();
        const auto gP2 = downloadBuffer<U4>(dev, bP2, CAP);
        const uint32_t gCount2 = downloadBuffer<uint32_t>(dev, bC2, 1)[0];
        dev->destroyBuffer(bN2); dev->destroyBuffer(bT2); dev->destroyBuffer(bV2);
        dev->destroyBuffer(bW2); dev->destroyBuffer(bP2); dev->destroyBuffer(bC2);
        PXG_CHECK_MSG(gCount2 == gCount, "npTetTriMid: overflow counter %u != %u", gCount2, gCount);
        PXG_CHECK_MSG(gCount2 > TINY, "npTetTriMid: overflow run did not overflow (%u)", gCount2);
        size_t written = 0, dirtyBeyond = 0;
        for (uint32_t i = 0; i < CAP; ++i)
        {
            if (i < TINY) { if (gP2[i].x < NQ) ++written; }
            else if (gP2[i].x != 0xffffffff) ++dirtyBeyond;
        }
        PXG_CHECK_MSG(written == TINY && dirtyBeyond == 0,
                      "npTetTriMid: overflow slots wrong (written=%zu dirty=%zu)", written, dirtyBeyond);
    }
}

} // namespace
