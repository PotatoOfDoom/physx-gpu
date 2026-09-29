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
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <string>
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

// P9.89: PxMin/PxMax<float> are selectMin(a, b) = a < b ? a : b and selectMax(a, b) =
// a > b ? a : b (PxMath.h:82-100, PxUnixMathIntrinsics.h:116-125): an equal compare returns b,
// and a NaN in either operand returns b.  This mirror used std::min/std::max before, which keep
// a on a tie; the pair sets below cannot see the difference (finite operands only), but the
// mirror transcribes the CUDA, so it spells the selects.
static float selMin(float a, float b) { return a < b ? a : b; }
static float selMax(float a, float b) { return a > b ? a : b; }

// minProject (softbodySoftbodyMidPhase.cu:72-76): PxMin(d0, PxMin(d1, d2)), `+ d` per distance
static float minProject3(V3 n, float d, V3 v0, V3 v1, V3 v2)
{
    return selMin(dot3(n, v0) + d, selMin(dot3(n, v1) + d, dot3(n, v2) + d));
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
        sep = selMax(sep, minProject3(pn[k], pd[k], v0, v1, v2));   // :93/:96/:99/:102
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
                    const float tolerance = selMin(laneBest[i], contactDist);   // :153
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


// ===========================================================================
// P9.89: satIntersect's folds are the CUDA's selects, not FMin/FMax.
//
// PxMin/PxMax<float> are selectMin(a, b) = a < b ? a : b and selectMax(a, b) = a > b ? a : b
// (PxMath.h:82-100, PxUnixMathIntrinsics.h:116-125).  softbodySoftbodyMidPhase.cu folds with
// them at minProject (:74-75, two), at satIntersect's four sep = PxMax(sep, minProject(
// tet0.planes[k], ...)) (:93, :96, :99, :102) and at the caller's PxMin(mBestDistance,
// mContactDistance) (:153).  The port spelled all seven min()/max() -- GLSL.std.450 FMin/FMax,
// which llvmpipe evaluates as "the non-NaN operand" with -0/+0 on a zero tie.  The tree shape
// and operand order were already the CUDA's; only the operator differed.
//
// p89tt below is the kernel transcribed with the selects; `revert` replays the OLD spelling one
// site at a time (bits 0..6), and bits 8..14 replay an operand SWAP of the same site, so the
// fixtures can prove that they see each site and each operand order.
//
// npTetTriMid_p89NaNSelect -- through ctTetTriMid itself, emission order compared.  A triangle
// whose THIRD vertex has an infinite coordinate along axis A passes the front-face gate: its
// edge cross holds 0 * inf = NaN, so the normal is the zero vector and the gate reads +-0 >= 0.
// Every tet plane whose normal has an exact zero A component then projects that vertex as
// 0 * inf = NaN, and the CUDA's PxMin(d1, NaN) = NaN sticks.  satIntersect's
// PxMax(sep, NaN) = NaN wipes the running separation, and the next plane's PxMax(NaN, x) = x
// RESETS it.  The old FMin/FMax skipped each NaN instead.  Families (one query tet each, one
// BV32 leaf per query, lane 0 the planted triangle, lanes 1.. ordinary competitors):
//   * INF3  -- a tet with exactly one face parallel to A, face 3.  The CUDA returns NaN and
//             never records the triangle; the old port and the plane-3 revert record it.
//   * INF2  -- the parallel face is face 2: the CUDA's separation is planes[3]'s alone, the
//             plane-2 revert keeps max(planes[0], planes[1]) -- a different rank.
//   * INF1  -- face 1: the CUDA keeps max(planes[2], planes[3]), the revert adds planes[0].
//   * NANV  -- the third vertex has a NaN coordinate: every projection of it is NaN, the CUDA
//             drops the triangle, the old minProject (either fold) returns a finite minimum.
//   * OVF   -- ALL-FINITE input: a tet with two ~2e19 edges, so the cross products of faces 0 and
//             3 overflow and their normals come out (0, 0, NaN).  planes[3]'s NaN is the last
//             fold, so the CUDA drops every triangle; the old port kept the running maximum.
//   * INF0  -- control: face 0 parallel.  The tri-plane term is already NaN there, so
//             PxMax(NaN, NaN) matches FMax(NaN, NaN): the planes[0] fold is kernel-blind (the
//             probe reads it), and the query only has to match the CUDA.
//   * ORD   -- control: finite, ordinary geometry; every twin agrees with the CUDA.
// The kernel is not in PXG_NOCONTRACT_SHADERS, so a mul-add may fuse on the device: every
// comparison whose outcome depends on inexact operands is margin-checked (Frag) for the CUDA
// order AND for every twin, and a candidate that is close anywhere is redrawn.  Non-finite and
// exact-zero operands are decided identically on both sides.
//
// npTetTriMid_p89SatProbe -- ctTetTriSatProbe returns minProject / satIntersect on caller
// operands, bit for bit.  Every record is EXACT arithmetic (axis-aligned or NaN/zero planes,
// dyadic coordinates, degenerate or axis-aligned triangles), so contraction cannot touch it,
// and it plants the +-0 ties and NaN/inf operands whose single-site revert (or swap) changes a
// word -- including the planes[0] fold the kernel cannot see.
//
// The tolerance fold PxMin(mBestDistance, mContactDistance) is OUTPUT-EQUIVALENT and has no
// fixture: bestDist is never NaN (it only takes a separation that passed `< bestDist`), so the
// select and FMin differ only for a NaN contactDist -- and then the box test
// `|d|^2 < contactDist^2` fails for every child of the root, no leaf is ever visited and the
// tolerance is never evaluated -- or on a +-0 tie, which `sep > tolerance` cannot see.
// ===========================================================================
namespace p89tt
{
static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static const float INF = std::numeric_limits<float>::infinity();
static const float QNAN = std::numeric_limits<float>::quiet_NaN();

// the OLD port's spelling: GLSL.std.450 FMin/FMax as llvmpipe evaluates them
static float oldFMin(float a, float b)
{
    if (std::isnan(a)) return b;
    if (std::isnan(b)) return a;
    if (a == 0.0f && b == 0.0f) return (std::signbit(a) || std::signbit(b)) ? -0.0f : 0.0f;
    return a < b ? a : b;
}
static float oldFMax(float a, float b)
{
    if (std::isnan(a)) return b;
    if (std::isnan(b)) return a;
    if (a == 0.0f && b == 0.0f) return (std::signbit(a) && std::signbit(b)) ? -0.0f : 0.0f;
    return a > b ? a : b;
}

enum Site : uint32_t
{
    MP_INNER = 1u << 0, // minProject  PxMin(plane.distance(v1), plane.distance(v2))   (:74-75)
    MP_OUTER = 1u << 1, // minProject  PxMin(plane.distance(v0), inner)                 (:74)
    SEP0 = 1u << 2,     // satIntersect sep = PxMax(sep, minProject(tet0.planes[0]))    (:93)
    SEP1 = 1u << 3,     //                                              planes[1]       (:96)
    SEP2 = 1u << 4,     //                                              planes[2]       (:99)
    SEP3 = 1u << 5,     //                                              planes[3]       (:102)
    TOL = 1u << 6,      // PxMin(mBestDistance, mContactDistance)                        (:153)
    SITE_COUNT = 7,
    SITE_ALL = (1u << 7) - 1u,
    SWAP_SHIFT = 8      // bit (site << 8): the same select with its operands swapped
};
static const char* const kSiteName[SITE_COUNT] = { "minProject inner", "minProject outer", "sep planes[0]",
                                                   "sep planes[1]", "sep planes[2]", "sep planes[3]",
                                                   "tolerance" };
static float foldMin(uint32_t rv, uint32_t site, float a, float b)
{
    if (rv & site) return oldFMin(a, b);
    if (rv & (site << SWAP_SHIFT)) return selMin(b, a);
    return selMin(a, b);
}
static float foldMax(uint32_t rv, uint32_t site, float a, float b)
{
    if (rv & site) return oldFMax(a, b);
    if (rv & (site << SWAP_SHIFT)) return selMax(b, a);
    return selMax(a, b);
}

// A comparison host and device could decide differently (see the header): finite, inexact
// operands closer than this.
struct Frag
{
    bool hit = false;
    void note(float a, float b)
    {
        if (std::isfinite(a) && std::isfinite(b) && std::fabs(a - b) <= 1e-4f * (1.0f + std::fabs(a) + std::fabs(b)))
            hit = true;
    }
};

static float comp(const V3& v, int i) { return (&v.x)[i]; }
static void setComp(V3& v, int i, float f) { (&v.x)[i] = f; }
static bool isZeroVec(const V3& v) { return v.x == 0.0f && v.y == 0.0f && v.z == 0.0f; }

// PxPlane(p0, p1, p2) with getNormalized's `m > 0 ? v * PxRecipSqrt(m) : 0`
// (npFloatIntrinRecipSqrt measures the device's rsqrt == 1 / sqrt)
static void plane3(V3 p0, V3 p1, V3 p2, V3& n, float& d)
{
    const V3 c = cross3(sub(p1, p0), sub(p2, p0));
    const float m = dot3(c, c);
    n = (m > 0.0f) ? muls(c, 1.0f / std::sqrt(m)) : V3{ 0.0f, 0.0f, 0.0f };
    d = -dot3(p0, n);
}
// Gu::createPlane; `dist` is the fourth vertex's signed distance before the flip
static void createPlane4(V3 pa, V3 pb, V3 pc, V3 pd, V3& n, float& d, float* dist = nullptr)
{
    plane3(pa, pb, pc, n, d);
    const float s = dot3(n, pd) + d;
    if (dist) *dist = s;
    if (s > 0.0f) { n = neg(n); d = -d; }
}
static void tetPlanes(const V3 t[4], V3 pn[4], float pd[4], float dist[4] = nullptr)
{
    createPlane4(t[0], t[1], t[2], t[3], pn[0], pd[0], dist ? &dist[0] : nullptr);
    createPlane4(t[0], t[1], t[3], t[2], pn[1], pd[1], dist ? &dist[1] : nullptr);
    createPlane4(t[0], t[2], t[3], t[1], pn[2], pd[2], dist ? &dist[2] : nullptr);
    createPlane4(t[1], t[2], t[3], t[0], pn[3], pd[3], dist ? &dist[3] : nullptr);
}

// minProject (:72-76)
static float minProject(V3 n, float d, V3 v0, V3 v1, V3 v2, uint32_t rv)
{
    return foldMin(rv, MP_OUTER, dot3(n, v0) + d, foldMin(rv, MP_INNER, dot3(n, v1) + d, dot3(n, v2) + d));
}

// satIntersect<BackFaceCull> (:78-106)
static float satIntersect(const V3 pn[4], const float pd[4], V3 centroid, V3 v0, V3 v1, V3 v2,
                          float tolerance, uint32_t rv, Frag* fr)
{
    V3 n;
    float d;
    plane3(v0, v1, v2, n, d);
    if (dot3(n, centroid) + d < 0.0f)
        return MAXF;
    float sep = minProject(n, d, v0, v1, v2, rv);
    if (fr) fr->note(sep, tolerance);
    if (sep > tolerance) return sep;
    static const uint32_t kSep[4] = { SEP0, SEP1, SEP2, SEP3 };
    for (int k = 0; k < 4; ++k)
    {
        sep = foldMax(rv, kSep[k], sep, minProject(pn[k], pd[k], v0, v1, v2, rv));
        if (fr) fr->note(sep, tolerance);
        if (sep > tolerance) return sep;
    }
    return sep;
}

struct Scene
{
    std::vector<Bv32Node> nodes;
    std::vector<U4> tris;
    std::vector<F4> verts;
};

// ctTetTriMid for one query: the BV32 walk, the per-lane best and the driver tail, returning
// the primitive indices in EMISSION order (slot index + i).
static std::vector<uint32_t> runQuery(const Scene& sc, const TtWork& w, uint32_t rv, Frag& fr)
{
    std::vector<uint32_t> out;
    V3 t[4] = { v3(w.v0), v3(w.v1), v3(w.v2), v3(w.v3) };
    const float cd = w.params.x;
    const float det = dot3(sub(t[1], t[0]), cross3(sub(t[2], t[0]), sub(t[3], t[0])));
    fr.note(det, 1e-9f);
    if (!(det > 1e-9f)) return out;

    // tetBoundingBox (deformableCollision.cuh:64-94), selects
    V3 tMin, tMax;
    for (int i = 0; i < 3; ++i)
    {
        setComp(tMin, i, selMin(selMin(comp(t[0], i), comp(t[1], i)), selMin(comp(t[2], i), comp(t[3], i))));
        setComp(tMax, i, selMax(selMax(comp(t[0], i), comp(t[1], i)), selMax(comp(t[2], i), comp(t[3], i))));
    }
    V3 pn[4];
    float pd[4], dist[4];
    tetPlanes(t, pn, pd, dist);
    for (int k = 0; k < 4; ++k)
        if (!isZeroVec(pn[k])) fr.note(dist[k], 0.0f);
    const V3 centroid = muls(add(add(t[0], t[1]), t[2]), 1.0f / 3.0f);

    float best[32];
    uint32_t prim[32];
    for (int j = 0; j < 32; ++j) { best[j] = MAXF; prim[j] = 0xffffffffu; }

    std::vector<uint32_t> stack;
    uint32_t cur = 0;
    for (;;)
    {
        const Bv32Node& node = sc.nodes[cur >> 6];
        if (cur & 1u)
        {
            const uint32_t data = node.mData[(cur >> 1) & 31u];
            const uint32_t nb = (data >> 1) & 63u, start = data >> 7;
            for (uint32_t lane = 0; lane < 32 && lane < nb; ++lane)
            {
                const uint32_t p = start + lane;
                const V3 v0 = v3(sc.verts[sc.tris[p].x]), v1 = v3(sc.verts[sc.tris[p].y]), v2 = v3(sc.verts[sc.tris[p].z]);
                V3 tn;
                float td;
                plane3(v0, v1, v2, tn, td);
                const float g = dot3(tn, centroid) + td;
                if (!isZeroVec(tn)) fr.note(g, 0.0f);
                if (!(g >= 0.0f)) continue;
                const float tol = foldMin(rv, TOL, best[lane], cd);
                const float sep = satIntersect(pn, pd, centroid, v0, v1, v2, tol, rv, &fr);
                if (best[lane] != MAXF) fr.note(sep, best[lane]);
                if (sep < best[lane]) { best[lane] = sep; prim[lane] = p; }
            }
        }
        else
        {
            for (uint32_t i = 0; i < node.mNbNodes; ++i)
            {
                float dd[3];
                for (int a = 0; a < 3; ++a)
                {
                    const float mn = (&node.mMin[i].x)[a], mx = (&node.mMax[i].x)[a];
                    dd[a] = (mn > comp(tMax, a)) ? (mn - comp(tMax, a)) : ((mx < comp(tMin, a)) ? (mx - comp(tMin, a)) : 0.0f);
                }
                const V3 disp = { dd[0], dd[1], dd[2] };
                const float d2 = dot3(disp, disp), cd2 = cd * cd;
                fr.note(d2, cd2);
                if (!(d2 < cd2)) continue;
                const uint32_t data = node.mData[i];
                stack.push_back((data & 1u) ? ((cur & 0xFFFFFFC0u) | (i << 1) | 1u) : ((data >> 11) << 6));
            }
        }
        if (stack.empty()) break;
        cur = stack.back();
        stack.pop_back();
    }

    // driver tail (:1403-1431): ballot, count = min(popc, 4), `count` rounds of minIndex
    uint32_t nq = 0;
    for (int j = 0; j < 32; ++j)
    {
        if (best[j] != MAXF) fr.note(best[j], cd);
        if (best[j] < cd) ++nq;
    }
    for (int j = 0; j < 32; ++j)
        for (int k = j + 1; k < 32; ++k)
            if (best[j] < cd && best[k] < cd) fr.note(best[j], best[k]);
    const uint32_t count = std::min<uint32_t>(nq, 4u);
    float bestSep = -1.0f;
    for (uint32_t i = 0; i < count && bestSep < cd; ++i)
    {
        uint32_t win = 0;
        float b = MAXF;
        for (uint32_t j = 0; j < 32; ++j)
            if (best[j] < b) { b = best[j]; win = j; }
        bestSep = b;
        if (best[win] < cd) { out.push_back(prim[win]); best[win] = MAXF; }
    }
    return out;
}

// ---- scene assembly: one BV32 leaf per region, regions grouped 32 to an inner node --------
struct Region
{
    V3 O;                                // box = O +- 3
    std::vector<std::array<V3, 3>> tris; // lane order
};
static Scene buildScene(const std::vector<Region>& regions)
{
    Scene sc;
    const uint32_t R = uint32_t(regions.size());
    const uint32_t G = (R + 31u) / 32u;
    sc.nodes.assign(1 + G, Bv32Node());
    for (auto& n : sc.nodes) std::memset(&n, 0, sizeof(n));
    sc.nodes[0].mNbNodes = G;
    for (uint32_t g = 0; g < G; ++g)
    {
        Bv32Node& gn = sc.nodes[1 + g];
        F4 gmin = { MAXF, MAXF, MAXF, 0 }, gmax = { -MAXF, -MAXF, -MAXF, 0 };
        for (uint32_t j = 0; j < 32 && g * 32 + j < R; ++j)
        {
            const Region& rg = regions[g * 32 + j];
            const uint32_t start = uint32_t(sc.tris.size());
            for (const auto& tr : rg.tris)
            {
                const uint32_t b = uint32_t(sc.verts.size());
                for (int k = 0; k < 3; ++k) sc.verts.push_back(f4(tr[k]));
                sc.tris.push_back({ b, b + 1, b + 2, 0 });
            }
            gn.mData[j] = (start << 7) | (uint32_t(rg.tris.size()) << 1) | 1u;
            gn.mMin[j] = { rg.O.x - 3.0f, rg.O.y - 3.0f, rg.O.z - 3.0f, 0 };
            gn.mMax[j] = { rg.O.x + 3.0f, rg.O.y + 3.0f, rg.O.z + 3.0f, 0 };
            gmin = { std::min(gmin.x, gn.mMin[j].x), std::min(gmin.y, gn.mMin[j].y), std::min(gmin.z, gn.mMin[j].z), 0 };
            gmax = { std::max(gmax.x, gn.mMax[j].x), std::max(gmax.y, gn.mMax[j].y), std::max(gmax.z, gn.mMax[j].z), 0 };
            gn.mNbNodes = j + 1;
        }
        sc.nodes[0].mData[g] = (1u + g) << 11;
        sc.nodes[0].mMin[g] = gmin;
        sc.nodes[0].mMax[g] = gmax;
    }
    return sc;
}

// the twins a kernel query is evaluated against: each single-site revert, the whole old
// spelling, and each single-site operand swap
static std::vector<uint32_t> twinMasks()
{
    std::vector<uint32_t> m;
    for (uint32_t s = 0; s < SITE_COUNT; ++s) m.push_back(1u << s);
    m.push_back(SITE_ALL);
    for (uint32_t s = 0; s < SITE_COUNT; ++s) m.push_back((1u << s) << SWAP_SHIFT);
    return m;
}
static std::string twinName(uint32_t m)
{
    if (m == SITE_ALL) return "old FMin/FMax";
    for (uint32_t s = 0; s < SITE_COUNT; ++s)
    {
        if (m == (1u << s)) return std::string(kSiteName[s]) + " revert";
        if (m == ((1u << s) << SWAP_SHIFT)) return std::string(kSiteName[s]) + " swap";
    }
    return "?";
}
} // namespace p89tt

PXG_TEST(npTetTriMid_p89NaNSelect)
{
    using namespace p89tt;
    std::mt19937 rng(89089u);
    auto D = [&](int lo, int hi, int den) { return float(int(rng() % uint32_t(hi - lo + 1)) + lo) / float(den); };
    auto u01 = [&]() { return float(rng() % 1000001u) / 1000000.0f; };
    auto axisVec = [](int a, float s) { V3 v = { 0.0f, 0.0f, 0.0f }; setComp(v, a, s); return v; };

    enum Fam { INF3 = 0, INF2, INF1, NANV, OVF, INF0, ORD, FAM_COUNT };
    const char* famName[FAM_COUNT] = { "INF3", "INF2", "INF1", "NANV", "OVF", "INF0", "ORD" };
    const int famWant[FAM_COUNT] = { 8, 8, 8, 6, 6, 4, 6 };
    const std::vector<uint32_t> twins = twinMasks();

    // ordinary triangles around c (lanes 1..)
    auto addCompetitors = [&](Region& rg, V3 c, float spread, int m) {
        for (int i = 0; i < m; ++i)
        {
            for (;;)
            {
                const V3 cc = add(c, V3{ spread * D(-64, 64, 64), spread * D(-64, 64, 64), spread * D(-64, 64, 64) });
                std::array<V3, 3> tr;
                for (int k = 0; k < 3; ++k) tr[k] = add(cc, V3{ D(-16, 16, 64), D(-16, 16, 64), D(-16, 16, 64) });
                const V3 cn = cross3(sub(tr[1], tr[0]), sub(tr[2], tr[0]));
                if (dot3(cn, cn) < 1e-4f) continue;
                rg.tris.push_back(tr);
                break;
            }
        }
    };
    // a generic tet around O (for NANV / ORD)
    auto genericTet = [&](V3 O, V3 t[4]) {
        for (;;)
        {
            for (int k = 0; k < 4; ++k) t[k] = add(O, V3{ D(-48, 48, 64), D(-48, 48, 64), D(-48, 48, 64) });
            const float det = dot3(sub(t[1], t[0]), cross3(sub(t[2], t[0]), sub(t[3], t[0])));
            if (det < 0.0f) std::swap(t[0], t[1]);
            if (std::fabs(det) < 0.05f) continue;
            V3 pn[4]; float pd[4], dist[4];
            tetPlanes(t, pn, pd, dist);
            bool ok = true;
            for (int k = 0; k < 4; ++k) ok = ok && std::fabs(dist[k]) > 0.05f;
            if (ok) return;
        }
    };
    // a tet whose face k (the one without vertex 3 - k) contains the direction e_A, so
    // planes[k].n has an EXACT zero A component and the other three do not
    auto parallelFaceTet = [&](V3 O, int A, int k, V3 t[4]) -> bool {
        const int B = (A + 1) % 3, C = (A + 2) % 3;
        V3 u = { 0.0f, 0.0f, 0.0f };
        setComp(u, B, D(-4, 4, 4));
        setComp(u, C, D(-4, 4, 4));
        if (comp(u, B) == 0.0f && comp(u, C) == 0.0f) return false;
        const V3 P1 = add(O, V3{ D(-16, 16, 64), D(-16, 16, 64), D(-16, 16, 64) });
        const float al = D(-16, 16, 16), be = D(-8, 8, 8), ga = D(-16, 16, 16), de = D(-8, 8, 8);
        if (std::fabs(al * de - be * ga) < 0.25f) return false;
        const V3 P2 = add(add(P1, axisVec(A, al)), muls(u, be));
        const V3 P3 = add(add(P1, axisVec(A, ga)), muls(u, de));
        const V3 Q = add(P1, V3{ D(-32, 32, 32), D(-32, 32, 32), D(-32, 32, 32) });
        const int excl = 3 - k;
        int fi[3], n = 0;
        for (int i = 0; i < 4; ++i) if (i != excl) fi[n++] = i;
        V3 f[3] = { P1, P2, P3 };
        const int perm = int(rng() % 6u);
        static const int P[6][3] = { { 0, 1, 2 }, { 0, 2, 1 }, { 1, 0, 2 }, { 1, 2, 0 }, { 2, 0, 1 }, { 2, 1, 0 } };
        t[excl] = Q;
        for (int i = 0; i < 3; ++i) t[fi[i]] = f[P[perm][i]];
        float det = dot3(sub(t[1], t[0]), cross3(sub(t[2], t[0]), sub(t[3], t[0])));
        if (det < 0.0f)
        {
            std::swap(t[fi[0]], t[fi[1]]);
            det = dot3(sub(t[1], t[0]), cross3(sub(t[2], t[0]), sub(t[3], t[0])));
        }
        if (!(det > 0.05f)) return false;
        V3 pn[4]; float pd[4], dist[4];
        tetPlanes(t, pn, pd, dist);
        for (int j = 0; j < 4; ++j)
        {
            if (std::fabs(dist[j]) < 0.05f) return false;
            if (j == k ? (comp(pn[j], A) != 0.0f) : (std::fabs(comp(pn[j], A)) < 0.1f)) return false;
        }
        return true;
    };

    std::vector<Region> regions;
    std::vector<TtWork> work;
    std::vector<int> fam;
    int famHave[FAM_COUNT] = {};
    uint32_t drawn = 0, fragile = 0, unplanted = 0;
    uint32_t grid = 0, ovfLevel = 0;

    for (int f = 0; f < FAM_COUNT; ++f)
    {
        for (uint32_t tries = 0; tries < 200000u && famHave[f] < famWant[f]; ++tries)
        {
            ++drawn;
            Region rg;
            V3 t[4];
            float cd = 0.12f + 0.3f * u01();
            if (f == OVF)
            {
                rg.O = { -64.0f, -64.0f, -64.0f * float(ovfLevel + 1) };
                const float L1 = 1.9e19f + 0.6e19f * u01(), L2 = 1.9e19f + 0.6e19f * u01();
                t[0] = rg.O;
                t[1] = add(rg.O, V3{ -L1, 0.0f, 0.0f });
                t[2] = add(rg.O, V3{ 0.0f, -L2, 0.0f });
                t[3] = add(rg.O, V3{ D(-8, 8, 32), D(-8, 8, 32), D(8, 16, 32) });
                // lane 0 and the rest: small finite triangles in the corner wedge
                addCompetitors(rg, add(rg.O, V3{ -0.25f, -0.25f, 0.15f }), 0.35f, 1 + int(rng() % 6u));
            }
            else
            {
                rg.O = { 32.0f * float(grid % 8u), 32.0f * float(grid / 8u), 0.0f };
                if (f == NANV || f == ORD)
                    genericTet(rg.O, t);
                else
                {
                    const int k = (f == INF3) ? 3 : (f == INF2) ? 2 : (f == INF1) ? 1 : 0;
                    const int A = int(rng() % 3u);
                    if (!parallelFaceTet(rg.O, A, k, t)) continue;
                    // lane 0: v0, v1 finite with a shared coordinate off A, v2 infinite along A
                    const float s = (rng() & 1u) ? 1.0f : -1.0f;
                    const V3 v0 = add(rg.O, V3{ D(-40, 40, 64), D(-40, 40, 64), D(-40, 40, 64) });
                    V3 e0 = { D(-24, 24, 64), D(-24, 24, 64), D(-24, 24, 64) };
                    setComp(e0, (A + 1 + int(rng() & 1u)) % 3, 0.0f);
                    if (isZeroVec(e0)) continue;
                    V3 v2 = add(v0, V3{ D(-24, 24, 64), D(-24, 24, 64), D(-24, 24, 64) });
                    setComp(v2, A, s * INF);
                    rg.tris.push_back({ v0, add(v0, e0), v2 });
                }
                if (f == NANV)
                {
                    const V3 v0 = add(rg.O, V3{ D(-40, 40, 64), D(-40, 40, 64), D(-40, 40, 64) });
                    const V3 v1 = add(v0, V3{ D(-24, 24, 64), D(-24, 24, 64), D(-24, 24, 64) });
                    V3 v2 = add(v0, V3{ D(-24, 24, 64), D(-24, 24, 64), D(-24, 24, 64) });
                    if (rng() & 1u) v2 = { QNAN, QNAN, QNAN };
                    else setComp(v2, int(rng() % 3u), QNAN);
                    rg.tris.push_back({ v0, v1, v2 });
                }
                addCompetitors(rg, rg.O, 0.8f, (f == ORD ? 3 : 2) + int(rng() % 6u));
            }
            TtWork w;
            w.v0 = f4(t[0]); w.v1 = f4(t[1]); w.v2 = f4(t[2]); w.v3 = f4(t[3]);
            w.params = { cd, 0, 0, 0 };

            // evaluate alone (regions never see each other: boxes are 32 apart, cd < 0.5)
            const Scene one = buildScene({ rg });
            Frag fr;
            const std::vector<uint32_t> ref = runQuery(one, w, 0u, fr);
            std::vector<std::vector<uint32_t>> tw(twins.size());
            for (size_t i = 0; i < twins.size(); ++i) tw[i] = runQuery(one, w, twins[i], fr);
            if (fr.hit) { ++fragile; continue; }
            auto differs = [&](uint32_t m) {
                for (size_t i = 0; i < twins.size(); ++i) if (twins[i] == m) return tw[i] != ref;
                return false;
            };
            bool planted = true;
            switch (f)
            {
            case INF3: planted = differs(SEP3) && differs(SITE_ALL); break;
            case INF2: planted = differs(SEP2) && differs(SITE_ALL); break;
            case INF1: planted = differs(SEP1) && differs(SITE_ALL); break;
            case NANV: planted = differs(MP_INNER) && differs(MP_OUTER) && differs(SITE_ALL); break;
            case OVF: planted = differs(SEP3) && differs(SITE_ALL) && ref.empty(); break;
            default: break;
            }
            if (!planted) { ++unplanted; continue; }
            regions.push_back(rg);
            work.push_back(w);
            fam.push_back(f);
            ++famHave[f];
            if (f == OVF) ++ovfLevel; else ++grid;
        }
    }
    bool famShort = false;
    for (int f = 0; f < FAM_COUNT; ++f) famShort = famShort || famHave[f] < famWant[f];
    PXG_CHECK_MSG(!famShort, "p89 NaN select: a family came up short (INF3 %d INF2 %d INF1 %d NANV %d OVF %d INF0 %d ORD %d)",
                  famHave[INF3], famHave[INF2], famHave[INF1], famHave[NANV], famHave[OVF], famHave[INF0], famHave[ORD]);
    if (famShort) return;

    // ---- the full scene: the reference and every twin, re-evaluated in place ----------------
    const Scene sc = buildScene(regions);
    const uint32_t NQ = uint32_t(work.size());
    std::vector<std::vector<uint32_t>> ref(NQ);
    std::vector<std::vector<std::vector<uint32_t>>> tw(NQ, std::vector<std::vector<uint32_t>>(twins.size()));
    size_t refPairs = 0, frag = 0;
    for (uint32_t q = 0; q < NQ; ++q)
    {
        Frag fr;
        ref[q] = runQuery(sc, work[q], 0u, fr);
        for (size_t i = 0; i < twins.size(); ++i) tw[q][i] = runQuery(sc, work[q], twins[i], fr);
        frag += fr.hit ? 1u : 0u;
        refPairs += ref[q].size();
    }
    PXG_CHECK_MSG(frag == 0, "p89 NaN select: %zu queries became margin-fragile in the assembled scene", frag);

    std::printf("    [p89 NaN select] %u queries (%u drawn, %u margin-fragile, %u unplanted redrawn), %zu CUDA pairs\n",
                NQ, drawn, fragile, unplanted, refPairs);
    std::vector<uint32_t> twinDiff(twins.size(), 0);
    for (size_t i = 0; i < twins.size(); ++i)
    {
        uint32_t byFam[FAM_COUNT] = {};
        for (uint32_t q = 0; q < NQ; ++q)
            if (tw[q][i] != ref[q]) { ++twinDiff[i]; ++byFam[fam[q]]; }
        std::printf("      %-24s differs on %2u queries (INF3 %u INF2 %u INF1 %u NANV %u OVF %u INF0 %u ORD %u)\n",
                    twinName(twins[i]).c_str(), twinDiff[i], byFam[INF3], byFam[INF2], byFam[INF1], byFam[NANV],
                    byFam[OVF], byFam[INF0], byFam[ORD]);
    }
    // the twins that must SEE their site through the kernel (the planes[0] fold and the
    // tolerance cannot -- see the header; the probe reads planes[0])
    const uint32_t mustSee[5] = { MP_INNER, MP_OUTER, SEP1, SEP2, SEP3 };
    for (uint32_t m : mustSee)
        for (size_t i = 0; i < twins.size(); ++i)
            if (twins[i] == m)
                PXG_CHECK_MSG(twinDiff[i] >= 4u, "p89 NaN select: the %s twin differs on only %u queries",
                              twinName(m).c_str(), twinDiff[i]);
    uint32_t oldSame = 0;
    for (uint32_t q = 0; q < NQ; ++q)
        if (fam[q] <= OVF && tw[q][SITE_COUNT] == ref[q]) ++oldSame;
    PXG_CHECK_MSG(oldSame == 0u, "p89 NaN select: the old twin matches the CUDA on %u planted queries", oldSame);
    uint32_t ordDiff = 0;
    for (uint32_t q = 0; q < NQ; ++q)
        if (fam[q] == ORD)
            for (size_t i = 0; i < twins.size(); ++i) ordDiff += (tw[q][i] != ref[q]) ? 1u : 0u;
    PXG_CHECK_MSG(ordDiff == 0u, "p89 NaN select: %u (ORD query, twin) pairs differ -- the control is not one", ordDiff);

    // ---- GPU ----------------------------------------------------------------------------------
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t CAP = NQ * 4 + 16;
    std::vector<U4> pairInit(CAP, { 0xffffffffu, 0xffffffffu, 0, 0 });
    std::vector<uint32_t> cnt(1, 0);
    PxgBufferHandle bN = createBufferFrom(dev, sc.nodes);
    PxgBufferHandle bT = createBufferFrom(dev, sc.tris);
    PxgBufferHandle bV = createBufferFrom(dev, sc.verts);
    PxgBufferHandle bW = createBufferFrom(dev, work);
    PxgBufferHandle bP = createBufferFrom(dev, pairInit);
    PxgBufferHandle bC = createBufferFrom(dev, cnt);
    TtParams prm = { NQ, CAP, 0, 0 };
    Kernel k("npTetTriMid.spv", "ctTetTriMid",
             { (uint32_t)sizeof(Bv32Node), 16, 16, (uint32_t)sizeof(TtWork), 16, 4 }, sizeof(TtParams));
    k.dispatch(NQ, { bN, bT, bV, bW, bP, bC }, prm);
    dev->flush();
    const auto gP = downloadBuffer<U4>(dev, bP, CAP);
    const uint32_t gCount = downloadBuffer<uint32_t>(dev, bC, 1)[0];
    dev->destroyBuffer(bN); dev->destroyBuffer(bT); dev->destroyBuffer(bV);
    dev->destroyBuffer(bW); dev->destroyBuffer(bP); dev->destroyBuffer(bC);

    std::vector<std::vector<uint32_t>> gpu(NQ);
    size_t bogus = 0;
    for (uint32_t i = 0; i < gCount && i < CAP; ++i)
    {
        if (gP[i].x < NQ) gpu[gP[i].x].push_back(gP[i].y);
        else ++bogus;
    }
    uint32_t bad[FAM_COUNT] = {}, badTotal = 0, asOld = 0, shown = 0;
    for (uint32_t q = 0; q < NQ; ++q)
    {
        if (gpu[q] == ref[q]) continue;
        ++badTotal;
        ++bad[fam[q]];
        if (gpu[q] == tw[q][SITE_COUNT]) ++asOld;
        if (shown++ < 6)
        {
            auto str = [](const std::vector<uint32_t>& v) {
                std::string s = "[";
                for (size_t i = 0; i < v.size(); ++i) s += (i ? " " : "") + std::to_string(v[i]);
                return s + "]";
            };
            std::printf("    p89 NaN select q%u (%s): gpu %s cuda %s old twin %s\n", q, famName[fam[q]],
                        str(gpu[q]).c_str(), str(ref[q]).c_str(), str(tw[q][SITE_COUNT]).c_str());
        }
    }
    PXG_CHECK_MSG(bogus == 0, "p89 NaN select: %zu pairs with an out-of-range workId", bogus);
    PXG_CHECK_MSG(gCount == refPairs, "p89 NaN select: counter %u, the CUDA emits %zu", gCount, refPairs);
    PXG_CHECK_MSG(badTotal == 0u,
                  "p89 NaN select: %u queries emit a different list than the CUDA's selects (%u of them the old "
                  "FMin/FMax twin's) -- INF3 %u INF2 %u INF1 %u NANV %u OVF %u INF0 %u ORD %u",
                  badTotal, asOld, bad[INF3], bad[INF2], bad[INF1], bad[NANV], bad[OVF], bad[INF0], bad[ORD]);
}

// ---------------------------------------------------------------------------
// npTetTriMid_p89SatProbe: ctTetTriSatProbe, every fold bit for bit.  Records are exact
// arithmetic -- see the p89tt header.  NaN compares as NaN (which NaN a select hands on is
// llvmpipe's business, and the CUDA's is canonical); every other word compares by bits.
// ---------------------------------------------------------------------------
namespace p89tt
{
struct TpRec { F4 v0, v1, v2, pl[4], ct, lone; };
static_assert(sizeof(TpRec) == 9 * 16, "probe record must be TP_REC_FLOAT4S = 9 float4s");

static void probeRef(const TpRec& r, uint32_t rv, float o[4])
{
    const V3 v0 = v3(r.v0), v1 = v3(r.v1), v2 = v3(r.v2);
    V3 pn[4];
    float pd[4];
    for (int k = 0; k < 4; ++k) { pn[k] = v3(r.pl[k]); pd[k] = r.pl[k].w; }
    o[0] = minProject(v3(r.lone), r.lone.w, v0, v1, v2, rv);
    o[1] = satIntersect(pn, pd, v3(r.ct), v0, v1, v2, r.ct.w, rv, nullptr);
    o[2] = satIntersect(pn, pd, v3(r.ct), v0, v1, v2, MAXF, rv, nullptr);
    V3 tn;
    float td;
    plane3(v0, v1, v2, tn, td);
    o[3] = minProject(tn, td, v0, v1, v2, rv);
}
static bool sameWord(float a, float b)
{
    return (std::isnan(a) && std::isnan(b)) || (!std::isnan(a) && !std::isnan(b) && f2u(a) == f2u(b));
}
static uint32_t probeDiffWords(const TpRec& r, uint32_t rv)
{
    float c[4], m[4];
    probeRef(r, 0u, c);
    probeRef(r, rv, m);
    uint32_t n = 0;
    for (int w = 0; w < 4; ++w) n += sameWord(c[w], m[w]) ? 0u : 1u;
    return n;
}
} // namespace p89tt

PXG_TEST(npTetTriMid_p89SatProbe)
{
    using namespace p89tt;
    uint64_t rs = 0x89e7a7e1ull;
    auto u = [&]() { rs = rs * 6364136223846793005ull + 1442695040888963407ull; return float(uint32_t(rs >> 40)) / 16777216.0f; };
    auto pick = [&](int n) { return int(u() * float(n)) % n; };
    auto dy = [&](int lo, int hi, int den) { return float(pick(hi - lo + 1) + lo) / float(den); };
    auto sgnOf = [&]() { return (u() < 0.5f) ? 1.0f : -1.0f; };
    auto axisF4 = [](int k, float s, float d) { F4 p = { 0.0f, 0.0f, 0.0f, d }; (&p.x)[k] = s; return p; };

    // values a vertex coordinate is planted with; projections on a +-e_k plane are exact
    const float PAT[10] = { -0.0f, 0.0f, -1.25f, -0.75f, 0.5f, 1.5f, 0.25f, QNAN, INF, -INF };
    auto patFinite = [&](bool zeroOnly) { return zeroOnly ? PAT[pick(2)] : PAT[pick(7)]; };
    auto patAny = [&]() { return PAT[pick(10)]; };
    const float DP[5] = { 0.0f, -0.0f, 0.25f, -0.5f, 1.0f };
    const float TOLS[6] = { MAXF, 0.25f, 0.0f, -0.0f, -0.5f, 1.0f };

    // a plane from the menu: +-e_j with a planted d, the zero plane, or makePlane's overflow
    // plane (0, 0, NaN | NaN)
    auto menuPlane = [&](int kind) {
        if (kind == 0) return axisF4(pick(3), sgnOf(), DP[pick(5)]);
        if (kind == 1) { F4 p = { u() < 0.5f ? 0.0f : -0.0f, u() < 0.5f ? 0.0f : -0.0f, u() < 0.5f ? 0.0f : -0.0f, DP[pick(2)] }; return p; }
        F4 p = { 0.0f, 0.0f, 0.0f, QNAN }; (&p.x)[pick(3)] = QNAN; return p;
    };
    // a triangle collinear along e_a: the three vertices share their other two coordinates, so
    // every edge cross is exactly zero (or NaN / inf once a planted value is non-finite) and
    // makePlane's normal is the zero vector (or NaN) with no rounding anywhere
    auto collinearTri = [&](TpRec& r, int a, const float c[3]) {
        V3 base = { dy(-48, 48, 32), dy(-48, 48, 32), dy(-48, 48, 32) };
        V3 vv[3];
        for (int i = 0; i < 3; ++i) { vv[i] = base; setComp(vv[i], a, c[i]); }
        r.v0 = f4(vv[0]); r.v1 = f4(vv[1]); r.v2 = f4(vv[2]);
    };

    enum Fam { MPPAT = 0, SEPPAT, AXTRI, RANDOM, PFAM_COUNT };
    std::vector<TpRec> recs;
    std::vector<int> fam;
    uint32_t missing = 0;

    // ---- MPPAT: minProject's two folds, x word (and w) ----------------------------------------
    {
        const uint32_t sites[4] = { MP_INNER, MP_OUTER, MP_INNER << SWAP_SHIFT, MP_OUTER << SWAP_SHIFT };
        for (uint32_t site : sites)
            for (int nonfin = 0; nonfin < 2; ++nonfin)
            {
                int found = 0;
                for (int tries = 0; tries < 400000 && found < 4; ++tries)
                {
                    TpRec r;
                    const int k = pick(3);
                    const float s = sgnOf();
                    float c[3];
                    bool anyNonfin = false;
                    for (int i = 0; i < 3; ++i)
                    {
                        const float p = nonfin ? patAny() : patFinite(u() < 0.6f);
                        c[i] = std::isnan(p) ? p : p * s;   // projection on s * e_k is exactly p
                        anyNonfin = anyNonfin || !std::isfinite(p);
                    }
                    if (bool(nonfin) != anyNonfin) continue;
                    collinearTri(r, k, c);
                    r.lone = axisF4(k, s, DP[pick(5)]);
                    for (int j = 0; j < 4; ++j) r.pl[j] = menuPlane(pick(3));
                    r.ct = { dy(-32, 32, 16), dy(-32, 32, 16), dy(-32, 32, 16), TOLS[pick(6)] };
                    float cw[4], mw[4];
                    probeRef(r, 0u, cw);
                    probeRef(r, site, mw);
                    if (sameWord(cw[0], mw[0])) continue;
                    recs.push_back(r); fam.push_back(MPPAT); ++found;
                }
                if (found < 4) ++missing;
            }
    }
    // ---- SEPPAT: satIntersect's four PxMax folds, y / z words -----------------------------------
    {
        const uint32_t base[4] = { SEP0, SEP1, SEP2, SEP3 };
        for (int bi = 0; bi < 8; ++bi)
        {
            const uint32_t site = (bi < 4) ? base[bi] : (base[bi - 4] << SWAP_SHIFT);
            for (int nonfin = 0; nonfin < 2; ++nonfin)
            {
                int found = 0;
                for (int tries = 0; tries < 400000 && found < 4; ++tries)
                {
                    TpRec r;
                    // a FINITE collinear triangle: the tri-plane term is an exact +-0
                    const int a = pick(3);
                    float c[3];
                    for (int i = 0; i < 3; ++i) c[i] = dy(-48, 48, 32);
                    collinearTri(r, a, c);
                    bool anyNan = false;
                    for (int j = 0; j < 4; ++j)
                    {
                        const int kind = nonfin ? pick(3) : pick(2);
                        r.pl[j] = menuPlane(kind);
                        anyNan = anyNan || kind == 2;
                    }
                    if (bool(nonfin) != anyNan) continue;
                    r.lone = menuPlane(0);
                    r.ct = { dy(-32, 32, 16), dy(-32, 32, 16), dy(-32, 32, 16), TOLS[pick(6)] };
                    float cw[4], mw[4];
                    probeRef(r, 0u, cw);
                    probeRef(r, site, mw);
                    if (sameWord(cw[1], mw[1]) && sameWord(cw[2], mw[2])) continue;
                    recs.push_back(r); fam.push_back(SEPPAT); ++found;
                }
                if (found < 4) ++missing;
            }
        }
    }
    // ---- AXTRI: non-degenerate axis-aligned triangles -- makePlane's normalize on an exact
    // power-of-two cross (npFloatIntrinRecipSqrt: rsqrt == 1 / sqrt), so the tri-plane term and
    // the internal back-face cull run through a unit normal with no rounding -------------------
    for (int t = 0; t < 48; ++t)
    {
        TpRec r;
        const int j = pick(3), a = (j + 1 + pick(2)) % 3, b = 3 - j - a;
        V3 v0 = { dy(-32, 32, 16), dy(-32, 32, 16), dy(-32, 32, 16) };
        if (u() < 0.3f) setComp(v0, j, u() < 0.5f ? 0.0f : -0.0f);
        V3 v1 = v0, v2 = v0;
        setComp(v1, a, comp(v0, a) + sgnOf() * float(1 << pick(3)) * 0.25f);
        setComp(v2, a, comp(v0, a) + dy(-8, 8, 8));
        setComp(v2, b, comp(v0, b) + sgnOf() * float(1 << pick(3)) * 0.5f);
        r.v0 = f4(v0); r.v1 = f4(v1); r.v2 = f4(v2);
        for (int k = 0; k < 4; ++k) r.pl[k] = menuPlane(pick(10) < 8 ? 0 : pick(3));
        r.lone = menuPlane(pick(2));
        r.ct = { dy(-32, 32, 16), dy(-32, 32, 16), dy(-32, 32, 16), TOLS[pick(6)] };
        recs.push_back(r); fam.push_back(AXTRI);
    }
    // ---- RANDOM: collinear triangles with any planted values, any menu planes -----------------
    for (int t = 0; t < 128; ++t)
    {
        TpRec r;
        const int a = pick(3);
        float c[3];
        for (int i = 0; i < 3; ++i) c[i] = (u() < 0.5f) ? patAny() : dy(-48, 48, 32);
        collinearTri(r, a, c);
        for (int k = 0; k < 4; ++k) r.pl[k] = menuPlane(pick(3));
        r.lone = menuPlane(pick(3));
        r.ct = { dy(-32, 32, 16), dy(-32, 32, 16), dy(-32, 32, 16), TOLS[pick(6)] };
        recs.push_back(r); fam.push_back(RANDOM);
    }
    PXG_CHECK_MSG(missing == 0u, "p89 SAT probe: %u planted (site, zero/non-finite) classes found fewer than four records",
                  missing);
    const uint32_t N = uint32_t(recs.size());

    // ---- the battery has to SEE every fold and every operand order ----------------------------
    uint32_t blind = 0, twinWords = 0;
    std::printf("    [p89 SAT probe] %u records; words each single-site twin changes:\n", N);
    for (uint32_t s = 0; s < 6; ++s)
    {
        uint32_t rvW = 0, swW = 0;
        for (uint32_t i = 0; i < N; ++i)
        {
            rvW += probeDiffWords(recs[i], 1u << s);
            swW += probeDiffWords(recs[i], (1u << s) << SWAP_SHIFT);
        }
        std::printf("      %-18s revert %3u  swap %3u\n", kSiteName[s], rvW, swW);
        if (rvW < 4u || swW < 4u) ++blind;
    }
    for (uint32_t i = 0; i < N; ++i) twinWords += probeDiffWords(recs[i], SITE_ALL);
    std::printf("      old FMin/FMax twin: %u words\n", twinWords);
    PXG_CHECK_MSG(blind == 0u, "p89 SAT probe: %u fold sites are changed on fewer than four words by their revert or swap",
                  blind);

    // ---- run the probe on ctTetTriMid's bindings ------------------------------------------------
    PxgSlangComputeDevice* dev = testDevice();
    std::vector<F4> in(size_t(N) * 9);
    std::memcpy(in.data(), recs.data(), sizeof(TpRec) * N);
    std::vector<U4> outInit(N, { 0x7f800001u, 1u, 2u, 3u });
    std::vector<Bv32Node> dNodes(1);
    std::memset(dNodes.data(), 0, sizeof(Bv32Node));
    std::vector<U4> dTris(1, { 0, 0, 0, 0 });
    std::vector<TtWork> dWork(1);
    std::memset(dWork.data(), 0, sizeof(TtWork));
    std::vector<uint32_t> dCnt(1, 0);
    PxgBufferHandle bN = createBufferFrom(dev, dNodes);
    PxgBufferHandle bT = createBufferFrom(dev, dTris);
    PxgBufferHandle bV = createBufferFrom(dev, in);
    PxgBufferHandle bW = createBufferFrom(dev, dWork);
    PxgBufferHandle bP = createBufferFrom(dev, outInit);
    PxgBufferHandle bC = createBufferFrom(dev, dCnt);
    TtParams prm = { N, 0, 0, 0 };
    Kernel k("npTetTriMid.spv", "ctTetTriSatProbe",
             { (uint32_t)sizeof(Bv32Node), 16, 16, (uint32_t)sizeof(TtWork), 16, 4 }, sizeof(TtParams));
    k.dispatch((N + 31u) / 32u, { bN, bT, bV, bW, bP, bC }, prm);
    dev->flush();
    const auto g = downloadBuffer<U4>(dev, bP, N);
    dev->destroyBuffer(bN); dev->destroyBuffer(bT); dev->destroyBuffer(bV);
    dev->destroyBuffer(bW); dev->destroyBuffer(bP); dev->destroyBuffer(bC);

    uint32_t bad[PFAM_COUNT] = {}, badWords = 0, asOld = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        float c[4], t[4];
        probeRef(recs[i], 0u, c);
        probeRef(recs[i], SITE_ALL, t);
        const uint32_t gw[4] = { g[i].x, g[i].y, g[i].z, g[i].w };
        bool recBad = false;
        for (int w = 0; w < 4; ++w)
        {
            float gv;
            std::memcpy(&gv, &gw[w], 4);
            if (sameWord(gv, c[w])) continue;
            recBad = true;
            ++badWords;
            if (sameWord(gv, t[w])) ++asOld;
            if (badWords <= 6)
                std::printf("    p89 SAT probe rec %u (fam %d) word %c: gpu %08x (%g) cuda %08x (%g) old twin %08x (%g)\n",
                            i, fam[i], "xyzw"[w], gw[w], gv, f2u(c[w]), c[w], f2u(t[w]), t[w]);
        }
        if (recBad) ++bad[fam[i]];
    }
    PXG_CHECK_MSG(badWords == 0u,
                  "p89 SAT probe: %u words differ from the CUDA's select folds (%u of them equal the old FMin/FMax "
                  "twin) -- records by family: minProject %u, sep %u, axis-tri %u, random %u",
                  badWords, asOld, bad[MPPAT], bad[SEPPAT], bad[AXTRI], bad[RANDOM]);
}

} // namespace
