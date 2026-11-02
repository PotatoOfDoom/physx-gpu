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

// P9.89: softBodySimClothMidphase_p89NaNSelect drives the ENGINE kernel
// sb_clothMidphaseGeneratePairsLaunch, which reads the real engine descriptors.  It lives in
// this file because its reference is the p89tt model below -- the kernel's traverser is the
// same TriangleLeafBoundMinMaxTraverser<false> ctTetTriMid ports.
#include "geometry/PxGeometry.h"
// no-CUDA-toolkit stand-in for the __device__/__align__ qualifiers; must precede PxgSoftBody.h.
#include "host_defines.h"
#include "PxgSoftBody.h"
#include "PxgFEMCloth.h"

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
// tetBoundingBox (deformableCollision.cuh:64-94), per axis
//   minimum = PxMin(PxMin(v0, v1), PxMin(v2, v3)), maximum = PxMax(PxMax(v0, v1), PxMax(v2, v3)).
// Its own mask space (bits 16..21, swaps at 24..29), so SITE_ALL and twinMasks() above keep
// meaning the SAT alone.
enum TbSite : uint32_t
{
    TB_MIN01 = 1u << 16,  // tX0 = PxMin(worldV0.x, worldV1.x)   (:70-72)
    TB_MIN23 = 1u << 17,  // tX1 = PxMin(worldV2.x, worldV3.x)   (:74-76)
    TB_MINOUT = 1u << 18, // minimum.x = PxMin(tX0, tX1)         (:78-80)
    TB_MAX01 = 1u << 19,  // tX0 = PxMax(worldV0.x, worldV1.x)   (:83-85)
    TB_MAX23 = 1u << 20,  // tX1 = PxMax(worldV2.x, worldV3.x)   (:87-89)
    TB_MAXOUT = 1u << 21, // maximum.x = PxMax(tX0, tX1)         (:91-93)
    TB_COUNT = 6,
    TB_SHIFT = 16,
    TB_ALL = 0x3Fu << 16
};
static const char* const kTbSiteName[TB_COUNT] = { "tetBound min(v0,v1)", "tetBound min(v2,v3)", "tetBound min outer",
                                                   "tetBound max(v0,v1)", "tetBound max(v2,v3)", "tetBound max outer" };
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

// satIntersect<BackFaceCull> (:78-106).  `cull` = BackFaceCull: ctTetTriMid keeps the internal
// cull (it stands in for both instantiations), softBodyClothMidphaseBda is the <false>
// instantiation the engine calls.  Behind the caller's front-face gate the two agree.
static float satIntersect(const V3 pn[4], const float pd[4], V3 centroid, V3 v0, V3 v1, V3 v2,
                          float tolerance, uint32_t rv, Frag* fr, bool cull = true)
{
    V3 n;
    float d;
    plane3(v0, v1, v2, n, d);
    if (cull && dot3(n, centroid) + d < 0.0f)
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

// tetBoundingBox (deformableCollision.cuh:64-94): the pairwise tree, per axis, as selects
static void tetBound(const V3 t[4], V3& tMin, V3& tMax, uint32_t rv)
{
    for (int i = 0; i < 3; ++i)
    {
        setComp(tMin, i, foldMin(rv, TB_MINOUT, foldMin(rv, TB_MIN01, comp(t[0], i), comp(t[1], i)),
                                 foldMin(rv, TB_MIN23, comp(t[2], i), comp(t[3], i))));
        setComp(tMax, i, foldMax(rv, TB_MAXOUT, foldMax(rv, TB_MAX01, comp(t[0], i), comp(t[1], i)),
                                 foldMax(rv, TB_MAX23, comp(t[2], i), comp(t[3], i))));
    }
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

    V3 tMin, tMax;
    tetBound(t, tMin, tMax, rv);
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
// ---- npTetTriMid_p89NaNSelect and softBodySimClothMidphase_p89NaNSelect share the generator,
// the host evaluation and the GPU comparison below; only the kernel launch differs. ------------
enum NsFam { INF3 = 0, INF2, INF1, NANV, OVF, INF0, ORD, FAM_COUNT };
static const char* const famName[FAM_COUNT] = { "INF3", "INF2", "INF1", "NANV", "OVF", "INF0", "ORD" };

struct NsSet
{
    std::vector<Region> regions;
    std::vector<TtWork> work;
    std::vector<int> fam;
    int famHave[FAM_COUNT] = {};
    uint32_t drawn = 0, fragile = 0, unplanted = 0;
    bool famShort = false;
};

// Draws the families (see the header) one query at a time, each evaluated in a scene of its own:
// a margin-fragile candidate, or one whose planted twin does not differ, is redrawn.
static NsSet genNaNSelect(uint32_t seed)
{
    NsSet gs;
    std::mt19937 rng(seed);
    auto D = [&](int lo, int hi, int den) { return float(int(rng() % uint32_t(hi - lo + 1)) + lo) / float(den); };
    auto u01 = [&]() { return float(rng() % 1000001u) / 1000000.0f; };
    auto axisVec = [](int a, float s) { V3 v = { 0.0f, 0.0f, 0.0f }; setComp(v, a, s); return v; };

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

    std::vector<Region>& regions = gs.regions;
    std::vector<TtWork>& work = gs.work;
    std::vector<int>& fam = gs.fam;
    int* famHave = gs.famHave;
    uint32_t& drawn = gs.drawn;
    uint32_t& fragile = gs.fragile;
    uint32_t& unplanted = gs.unplanted;
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
    for (int f = 0; f < FAM_COUNT; ++f) gs.famShort = gs.famShort || gs.famHave[f] < famWant[f];
    return gs;
}

static bool famOk(const char* tag, const NsSet& g)
{
    PXG_CHECK_MSG(!g.famShort, "%s: a family came up short (INF3 %d INF2 %d INF1 %d NANV %d OVF %d INF0 %d ORD %d)", tag,
                  g.famHave[INF3], g.famHave[INF2], g.famHave[INF1], g.famHave[NANV], g.famHave[OVF], g.famHave[INF0],
                  g.famHave[ORD]);
    return !g.famShort;
}

struct NsEval
{
    std::vector<uint32_t> twins;
    std::vector<std::vector<uint32_t>> ref;
    std::vector<std::vector<std::vector<uint32_t>>> tw;
    size_t refPairs = 0;
};

// The full scene: the reference and every twin, re-evaluated in place, and the host-side proof
// that the battery sees each site it claims to see.
static NsEval evalNaNSelect(const char* tag, const Scene& sc, const std::vector<TtWork>& work,
                            const std::vector<int>& fam, const NsSet& g)
{
    NsEval ev;
    ev.twins = twinMasks();
    const std::vector<uint32_t>& twins = ev.twins;
    const uint32_t NQ = uint32_t(work.size());
    std::vector<std::vector<uint32_t>>& ref = ev.ref;
    std::vector<std::vector<std::vector<uint32_t>>>& tw = ev.tw;
    ref.assign(NQ, {});
    tw.assign(NQ, std::vector<std::vector<uint32_t>>(twins.size()));
    size_t frag = 0;
    for (uint32_t q = 0; q < NQ; ++q)
    {
        Frag fr;
        ref[q] = runQuery(sc, work[q], 0u, fr);
        for (size_t i = 0; i < twins.size(); ++i) tw[q][i] = runQuery(sc, work[q], twins[i], fr);
        frag += fr.hit ? 1u : 0u;
        ev.refPairs += ref[q].size();
    }
    PXG_CHECK_MSG(frag == 0, "%s: %zu queries became margin-fragile in the assembled scene", tag, frag);

    std::printf("    [%s] %u queries (%u drawn, %u margin-fragile, %u unplanted redrawn), %zu CUDA pairs\n",
                tag, NQ, g.drawn, g.fragile, g.unplanted, ev.refPairs);
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
                PXG_CHECK_MSG(twinDiff[i] >= 4u, "%s: the %s twin differs on only %u queries", tag,
                              twinName(m).c_str(), twinDiff[i]);
    uint32_t oldSame = 0;
    for (uint32_t q = 0; q < NQ; ++q)
        if (fam[q] <= OVF && tw[q][SITE_COUNT] == ref[q]) ++oldSame;
    PXG_CHECK_MSG(oldSame == 0u, "%s: the old twin matches the CUDA on %u planted queries", tag, oldSame);
    uint32_t ordDiff = 0;
    for (uint32_t q = 0; q < NQ; ++q)
        if (fam[q] == ORD)
            for (size_t i = 0; i < twins.size(); ++i) ordDiff += (tw[q][i] != ref[q]) ? 1u : 0u;
    PXG_CHECK_MSG(ordDiff == 0u, "%s: %u (ORD query, twin) pairs differ -- the control is not one", tag, ordDiff);
    return ev;
}

// `gpu[q]` is query q's primitive list in EMISSION order (slot index + i).
static void checkNaNSelectGpu(const char* tag, const NsEval& ev, const std::vector<int>& fam,
                              const std::vector<std::vector<uint32_t>>& gpu, uint32_t gCount, size_t bogus)
{
    const std::vector<std::vector<uint32_t>>& ref = ev.ref;
    const std::vector<std::vector<std::vector<uint32_t>>>& tw = ev.tw;
    const uint32_t NQ = uint32_t(ref.size());
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
            std::printf("    %s q%u (%s): gpu %s cuda %s old twin %s\n", tag, q, famName[fam[q]],
                        str(gpu[q]).c_str(), str(ref[q]).c_str(), str(tw[q][SITE_COUNT]).c_str());
        }
    }
    PXG_CHECK_MSG(bogus == 0, "%s: %zu pairs with an out-of-range workId", tag, bogus);
    PXG_CHECK_MSG(gCount == ev.refPairs, "%s: counter %u, the CUDA emits %zu", tag, gCount, ev.refPairs);
    PXG_CHECK_MSG(badTotal == 0u,
                  "%s: %u queries emit a different list than the CUDA's selects (%u of them the old "
                  "FMin/FMax twin's) -- INF3 %u INF2 %u INF1 %u NANV %u OVF %u INF0 %u ORD %u", tag,
                  badTotal, asOld, bad[INF3], bad[INF2], bad[INF1], bad[NANV], bad[OVF], bad[INF0], bad[ORD]);
}
} // namespace p89tt

PXG_TEST(npTetTriMid_p89NaNSelect)
{
    using namespace p89tt;
    const char* tag = "p89 NaN select";
    const NsSet gen = genNaNSelect(89089u);
    if (!famOk(tag, gen)) return;
    const std::vector<TtWork>& work = gen.work;
    const Scene sc = buildScene(gen.regions);
    const NsEval ev = evalNaNSelect(tag, sc, work, gen.fam, gen);
    const uint32_t NQ = uint32_t(work.size());

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
    checkNaNSelectGpu(tag, ev, gen.fam, gpu, gCount, bogus);
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

// `cull`: see satIntersect -- ctTetTriSatProbe runs the culling copy, sb_clothMidphaseSatProbe
// the engine's <false> instantiation.
static void probeRef(const TpRec& r, uint32_t rv, float o[4], bool cull = true)
{
    const V3 v0 = v3(r.v0), v1 = v3(r.v1), v2 = v3(r.v2);
    V3 pn[4];
    float pd[4];
    for (int k = 0; k < 4; ++k) { pn[k] = v3(r.pl[k]); pd[k] = r.pl[k].w; }
    o[0] = minProject(v3(r.lone), r.lone.w, v0, v1, v2, rv);
    o[1] = satIntersect(pn, pd, v3(r.ct), v0, v1, v2, r.ct.w, rv, nullptr, cull);
    o[2] = satIntersect(pn, pd, v3(r.ct), v0, v1, v2, MAXF, rv, nullptr, cull);
    V3 tn;
    float td;
    plane3(v0, v1, v2, tn, td);
    o[3] = minProject(tn, td, v0, v1, v2, rv);
}
static bool sameWord(float a, float b)
{
    return (std::isnan(a) && std::isnan(b)) || (!std::isnan(a) && !std::isnan(b) && f2u(a) == f2u(b));
}
static uint32_t probeDiffWords(const TpRec& r, uint32_t rv, bool cull = true)
{
    float c[4], m[4];
    probeRef(r, 0u, c, cull);
    probeRef(r, rv, m, cull);
    uint32_t n = 0;
    for (int w = 0; w < 4; ++w) n += sameWord(c[w], m[w]) ? 0u : 1u;
    return n;
}
enum PFam { MPPAT = 0, SEPPAT, AXTRI, RANDOM, PFAM_COUNT };

// The SAT probe battery (see npTetTriMid_p89SatProbe): exact-arithmetic records, planted per
// fold, per operand order and per class (signed zero / non-finite), then ordinary fillers.
// `cull` picks the satIntersect the planted records are checked against.
static void genSatProbeRecords(uint64_t seed, bool cull, std::vector<TpRec>& recs, std::vector<int>& fam,
                               uint32_t& missing)
{
    uint64_t rs = seed;
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
                    probeRef(r, 0u, cw, cull);
                    probeRef(r, site, mw, cull);
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
                    probeRef(r, 0u, cw, cull);
                    probeRef(r, site, mw, cull);
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
}
} // namespace p89tt

PXG_TEST(npTetTriMid_p89SatProbe)
{
    using namespace p89tt;
    std::vector<TpRec> recs;
    std::vector<int> fam;
    uint32_t missing = 0;
    genSatProbeRecords(0x89e7a7e1ull, /*cull*/ true, recs, fam, missing);
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

// ===========================================================================
// P9.89 -- tetBoundingBox (deformableCollision.cuh:64-94), called by computeTetBoundWithTetrahedron
// (softbodySoftbodyMidPhase.cu:655-670): per axis PxMin(PxMin(v0, v1), PxMin(v2, v3)) and the PxMax
// twin, the pairwise tree of selects.  Both ports spelled it min()/max() (FMin/FMax).  It is
// output-equivalent in both kernels -- a NaN corner fails isValidTet before the bound is taken,
// and the box test reads the bound through `>`/`<` and through mn - tMax / mx - tMin taken only
// when that difference is non-zero, none of which sees a zero's sign -- so it is read out through
// a probe: ctTetTriBoundProbe here (npTetTriMid_p89BoundProbe), sb_clothMidphaseSatProbe for the
// engine port (softBodySimClothMidphase_p89SatProbe).  The records are +-0 / non-finite
// coordinate patterns and dyadic values, so every word is exact.
// ===========================================================================
namespace p89tt
{
struct TbRec { F4 t[4]; };
static_assert(sizeof(TbRec) == 4 * 16, "bound record must be TB_REC_FLOAT4S = 4 float4s");
enum TbFam { TBPAT = 0, TBRANDOM, TBFAM_COUNT };

static void tbRef(const TbRec& r, uint32_t rv, float o[6])
{
    const V3 t[4] = { v3(r.t[0]), v3(r.t[1]), v3(r.t[2]), v3(r.t[3]) };
    V3 mn, mx;
    tetBound(t, mn, mx, rv);
    o[0] = mn.x; o[1] = mn.y; o[2] = mn.z;
    o[3] = mx.x; o[4] = mx.y; o[5] = mx.z;
}
static uint32_t tbDiffWords(const TbRec& r, uint32_t rv)
{
    float c[6], m[6];
    tbRef(r, 0u, c);
    tbRef(r, rv, m);
    uint32_t n = 0;
    for (int w = 0; w < 6; ++w) n += sameWord(c[w], m[w]) ? 0u : 1u;
    return n;
}
// tetBoundingBox corners: for each of the six selects, each operand order and each class
// (signed-zero ties / non-finite), four records whose single-site twin changes a word; then
// fillers up to `fillTo`.  Every coordinate is a planted pattern or a dyadic value.
static void genTbRecords(uint64_t seed, size_t fillTo, std::vector<TbRec>& recs, std::vector<int>& fam,
                         uint32_t& missing)
{
    uint64_t rs = seed;
    auto u = [&]() { rs = rs * 6364136223846793005ull + 1442695040888963407ull; return float(uint32_t(rs >> 40)) / 16777216.0f; };
    auto pick = [&](int n) { return int(u() * float(n)) % n; };
    auto dy = [&](int lo, int hi, int den) { return float(pick(hi - lo + 1) + lo) / float(den); };
    const float PAT[10] = { -0.0f, 0.0f, -1.25f, -0.75f, 0.5f, 1.5f, 0.25f, QNAN, INF, -INF };
    auto patFinite = [&](bool zeroOnly) { return zeroOnly ? PAT[pick(2)] : PAT[pick(7)]; };
    auto patAny = [&]() { return PAT[pick(10)]; };

    for (uint32_t s = 0; s < TB_COUNT; ++s)
        for (int sw = 0; sw < 2; ++sw)
        {
            const uint32_t site = sw ? ((1u << (TB_SHIFT + s)) << SWAP_SHIFT) : (1u << (TB_SHIFT + s));
            for (int nonfin = 0; nonfin < 2; ++nonfin)
            {
                int found = 0;
                for (int tries = 0; tries < 400000 && found < 4; ++tries)
                {
                    TbRec r;
                    bool anyNonfin = false;
                    for (int k = 0; k < 4; ++k)
                    {
                        float* c = &r.t[k].x;
                        for (int i = 0; i < 3; ++i)
                        {
                            c[i] = nonfin ? patAny() : patFinite(u() < 0.6f);
                            anyNonfin = anyNonfin || !std::isfinite(c[i]);
                        }
                        r.t[k].w = 0.0f;
                    }
                    if (bool(nonfin) != anyNonfin) continue;
                    if (tbDiffWords(r, site) == 0u) continue;
                    recs.push_back(r); fam.push_back(TBPAT); ++found;
                }
                if (found < 4) ++missing;
            }
        }
    while (recs.size() < fillTo)
    {
        TbRec r;
        for (int k = 0; k < 4; ++k)
        {
            float* c = &r.t[k].x;
            for (int i = 0; i < 3; ++i) c[i] = (u() < 0.3f) ? patAny() : dy(-48, 48, 32);
            r.t[k].w = 0.0f;
        }
        recs.push_back(r); fam.push_back(TBRANDOM);
    }
}
} // namespace p89tt

PXG_TEST(npTetTriMid_p89BoundProbe)
{
    using namespace p89tt;
    const char* tag = "p89 tetBound probe";
    std::vector<TbRec> recs;
    std::vector<int> fam;
    uint32_t missing = 0;
    genTbRecords(0x7b0d89a1ull, 192u, recs, fam, missing);
    PXG_CHECK_MSG(missing == 0u, "%s: %u planted (site, zero/non-finite) classes found fewer than four records", tag,
                  missing);
    const uint32_t N = uint32_t(recs.size());

    // ---- the battery has to SEE every select and every operand order --------------------------
    uint32_t blind = 0, twinWords = 0;
    std::printf("    [%s] %u records; words each single-site twin changes:\n", tag, N);
    for (uint32_t s = 0; s < TB_COUNT; ++s)
    {
        const uint32_t site = 1u << (TB_SHIFT + s);
        uint32_t rvW = 0, swW = 0;
        for (uint32_t i = 0; i < N; ++i)
        {
            rvW += tbDiffWords(recs[i], site);
            swW += tbDiffWords(recs[i], site << SWAP_SHIFT);
        }
        std::printf("      %-20s revert %3u  swap %3u\n", kTbSiteName[s], rvW, swW);
        if (rvW < 4u || swW < 4u) ++blind;
    }
    for (uint32_t i = 0; i < N; ++i) twinWords += tbDiffWords(recs[i], TB_ALL);
    std::printf("      old FMin/FMax twin: %u words\n", twinWords);
    PXG_CHECK_MSG(blind == 0u, "%s: %u selects are changed on fewer than four words by their revert or swap", tag, blind);

    // ---- run the probe on ctTetTriMid's bindings ------------------------------------------------
    PxgSlangComputeDevice* dev = testDevice();
    std::vector<F4> in(size_t(N) * 4);
    std::memcpy(in.data(), recs.data(), sizeof(TbRec) * N);
    std::vector<U4> outInit(size_t(N) * 2, { 0x7f800001u, 1u, 2u, 3u });
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
    Kernel k("npTetTriMid.spv", "ctTetTriBoundProbe",
             { (uint32_t)sizeof(Bv32Node), 16, 16, (uint32_t)sizeof(TtWork), 16, 4 }, sizeof(TtParams));
    k.dispatch((N + 31u) / 32u, { bN, bT, bV, bW, bP, bC }, prm);
    dev->flush();
    const auto g = downloadBuffer<U4>(dev, bP, size_t(N) * 2);
    dev->destroyBuffer(bN); dev->destroyBuffer(bT); dev->destroyBuffer(bV);
    dev->destroyBuffer(bW); dev->destroyBuffer(bP); dev->destroyBuffer(bC);

    uint32_t bad[TBFAM_COUNT] = {}, badWords = 0, asOld = 0, badPad = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        float c[6], t[6];
        tbRef(recs[i], 0u, c);
        tbRef(recs[i], TB_ALL, t);
        const uint32_t gw[8] = { g[2 * i].x, g[2 * i].y, g[2 * i].z, g[2 * i].w,
                                 g[2 * i + 1].x, g[2 * i + 1].y, g[2 * i + 1].z, g[2 * i + 1].w };
        badPad += (gw[6] != 0u || gw[7] != 0u) ? 1u : 0u;
        bool recBad = false;
        for (int w = 0; w < 6; ++w)
        {
            float gv;
            std::memcpy(&gv, &gw[w], 4);
            if (sameWord(gv, c[w])) continue;
            recBad = true;
            ++badWords;
            if (sameWord(gv, t[w])) ++asOld;
            if (badWords <= 6)
                std::printf("    %s rec %u word %d: gpu %08x (%g) cuda %08x (%g) old twin %08x (%g)\n", tag, i, w, gw[w],
                            gv, f2u(c[w]), c[w], f2u(t[w]), t[w]);
        }
        if (recBad) ++bad[fam[i]];
    }
    PXG_CHECK_MSG(badPad == 0u, "%s: %u records with a non-zero pad word", tag, badPad);
    PXG_CHECK_MSG(badWords == 0u,
                  "%s: %u words differ from the CUDA's select tree (%u of them equal the old FMin/FMax twin) -- "
                  "records by family: planted %u, random %u",
                  tag, badWords, asOld, bad[TBPAT], bad[TBRANDOM]);
}

// ===========================================================================
// P9.89 -- the same folds in the ENGINE kernel: sb_clothMidphaseGeneratePairsLaunch
// (softBodyClothMidphaseBda.slang, SB_CLOTH_MIDPHASE; softbodySoftbodyMidPhase.cu:1299-1444).
//
// Its traverser is TriangleLeafBoundMinMaxTraverser<false> -- the one ctTetTriMid ports -- so the
// p89tt model above IS its reference: minProject (:72-76), satIntersect's four
// sep = PxMax(sep, minProject(tet0.planes[k], ...)) (:93, :96, :99, :102), the caller's
// PxMin(mBestDistance, mContactDistance) (:153) and tetBoundingBox (deformableCollision.cuh:64-94),
// all PxMin/PxMax<float> selects.  The port spelled every one of them min()/max() (GLSL.std.450
// FMin/FMax).  The tests live in this file for that reason; they carry the softBodySim suite name
// because the kernel is that suite's.
//
// softBodySimClothMidphase_p89NaNSelect -- the NaN-select battery of npTetTriMid_p89NaNSelect
// (drawn with its own seed), through the engine kernel and its driver:
//   * one contact manager per query (grid y = cmIdx), each pairing its OWN one-tet softbody with
//     the ONE cloth whose BV32 tree holds every region; odd managers list the cloth first, so
//     LoadShapePairWarp finds the tet mesh by type;
//   * the query's contact distance is contactDistance[sb] + contactDistance[cloth] = cd_q + 0;
//   * the tets are handed over in WORLD space (cloth-local + T, identity rotation) and the
//     reference evaluates what PxTransform::transformInv (PxTransform.h:134-138,
//     PxQuat::rotateInv PxQuat.h:324-333) hands the traverser;
//   * the engine grid, 1024 x numTests, and every surface hint set.
// Each query's pairs are compared IN EMISSION ORDER (one atomicAdd of the count, then slot
// index + i).  The kernel is in PXG_NOCONTRACT_SHADERS; the Frag margins are kept anyway.
//
// softBodySimClothMidphase_p89SatProbe -- sb_clothMidphaseSatProbe (test-only, not in the
// manifest) returns minProject3, satIntersectTetTri (the <false> instantiation: no internal cull)
// and tetBoundingBox on caller operands, bit for bit.  It is the only way to see the planes[0]
// fold (kernel-blind, see the p89tt header) and tetBoundingBox: in the kernel a NaN corner fails
// isValidTet before the bound is taken, and the box test reads the bound through `>`/`<` and
// through mn - tMax / mx - tMin taken only when that difference is non-zero, none of which sees a
// zero's sign.  The tetBoundingBox records are +-0 / non-finite coordinate patterns, so every
// value is exact.
//
// The tolerance fold stays output-equivalent here too (see the p89tt header): bestDist is never
// NaN, and a NaN contactDist fails `|d|^2 < contactDist^2` for every child of the root.
// ===========================================================================
namespace p89sbc
{
using namespace p89tt;

// { PxReal toleranceLength, cmInputs*, transformCache*, bounds*, contactDistance*, gpuShapes*,
//   softbodies*, clothes*, PxU32 stackSizeBytes, stackPtr*, midphasePairsNum* }
struct SbClothMidPush
{
    float toleranceLength; uint32_t pad4;
    uint64_t cmInputs, transformCache, bounds, contactDistance, gpuShapes, softbodies, clothes;
    uint32_t stackSizeBytes, pad68;
    uint64_t stackPtr, midphasePairsNum;
};
static_assert(sizeof(SbClothMidPush) == 88, "sb_clothMidphase push blob must be 88 bytes");
static_assert(sizeof(Bv32Node) == 1168, "Gu::BV32DataPacked is 1168 bytes");

struct CmIn { uint32_t shapeRef0, shapeRef1, cacheRef0, cacheRef1; };
struct ShapeH
{
    float sx, sy, sz, srx, sry, srz, srw;
    uint32_t materialIndex, hullLo, hullHi, type, particleOrSoftbodyId;
};
static_assert(sizeof(ShapeH) == 48, "PxgShape mirror must be 48 bytes");

struct DevBuf { PxgBufferHandle h = 0; uint64_t addr = 0; };
template <typename T>
static DevBuf makeDev(PxgSlangComputeDevice* dev, const std::vector<T>& data)
{
    DevBuf b;
    b.h = createBufferFrom(dev, data);
    b.addr = dev->getBufferDeviceAddress(b.h);
    return b;
}

// PxQuat::rotateInv (PxQuat.h:324-333), expression for expression
static V3 quatRotateInv(const float q[4], V3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q[3] * q[3] - 0.5f;
    const float dot2 = (q[0] * vx + q[1] * vy + q[2] * vz);
    return { vx * w2 - (q[1] * vz - q[2] * vy) * q[3] + q[0] * dot2,
             vy * w2 - (q[2] * vx - q[0] * vz) * q[3] + q[1] * dot2,
             vz * w2 - (q[0] * vy - q[1] * vx) * q[3] + q[2] * dot2 };
}

// ---- the probe: SAT record + tetBoundingBox corners (p89tt::TbRec) ------------------------
struct SbcRec { TpRec sat; TbRec tb; };
static_assert(sizeof(SbcRec) == 13 * 16, "probe record must be SBC_PROBE_IN_FLOAT4S = 13 float4s");

// words 0..3 = the SAT (satIntersect<false>), 4..9 = tMin.xyz, tMax.xyz
static void sbcProbeRef(const SbcRec& r, uint32_t rv, float o[10])
{
    probeRef(r.sat, rv, o, /*cull*/ false);
    tbRef(r.tb, rv, o + 4);
}
static uint32_t sbcDiffWords(const SbcRec& r, uint32_t rv)
{
    float c[10], m[10];
    sbcProbeRef(r, 0u, c);
    sbcProbeRef(r, rv, m);
    uint32_t n = 0;
    for (int w = 0; w < 10; ++w) n += sameWord(c[w], m[w]) ? 0u : 1u;
    return n;
}
} // namespace p89sbc

PXG_TEST(softBodySimClothMidphase_p89NaNSelect)
{
    using namespace p89tt;
    using namespace p89sbc;
    const char* tag = "p89 sbCloth NaN select";
    const NsSet gen = genNaNSelect(89188u);
    if (!famOk(tag, gen)) return;
    const uint32_t NQ = uint32_t(gen.work.size());

    // ---- the cloth frame and the tets the kernel's traverser sees ----------------------------
    const float Q[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
    const V3 T = { 8.0f, -16.0f, 4.0f };
    std::vector<F4> tetWorld;
    std::vector<U4> tetIdx;
    std::vector<TtWork> seen(NQ);
    uint32_t moved = 0;
    for (uint32_t q = 0; q < NQ; ++q)
    {
        const F4* lv[4] = { &gen.work[q].v0, &gen.work[q].v1, &gen.work[q].v2, &gen.work[q].v3 };
        F4* sv[4] = { &seen[q].v0, &seen[q].v1, &seen[q].v2, &seen[q].v3 };
        for (int k = 0; k < 4; ++k)
        {
            const V3 w = add(v3(*lv[k]), T);
            tetWorld.push_back({ w.x, w.y, w.z, 1.0f });
            const V3 c = quatRotateInv(Q, sub(w, T));   // clothToWorld.transformInv(worldV)
            *sv[k] = f4(c);
            if (f2u(c.x) != f2u(lv[k]->x) || f2u(c.y) != f2u(lv[k]->y) || f2u(c.z) != f2u(lv[k]->z)) ++moved;
        }
        seen[q].params = gen.work[q].params;
        tetIdx.push_back({ 4u * q, 4u * q + 1u, 4u * q + 2u, 4u * q + 3u });
    }
    std::printf("    [%s] %u of %u tet corners change bits through transformInv (signs of zero only)\n", tag, moved, 4u * NQ);

    const Scene sc = buildScene(gen.regions);
    const NsEval ev = evalNaNSelect(tag, sc, seen, gen.fam, gen);
    // tetBoundingBox is kernel-blind (see the header), and this battery agrees: its whole old
    // spelling, and each of its single-site twins, changes no query
    uint32_t tbSeen = 0;
    for (uint32_t q = 0; q < NQ; ++q)
    {
        Frag fr;
        tbSeen += (runQuery(sc, seen[q], TB_ALL, fr) != ev.ref[q]) ? 1u : 0u;
        for (uint32_t s = 0; s < TB_COUNT; ++s)
        {
            const uint32_t site = 1u << (TB_SHIFT + s);
            tbSeen += (runQuery(sc, seen[q], site, fr) != ev.ref[q]) ? 1u : 0u;
            tbSeen += (runQuery(sc, seen[q], site << SWAP_SHIFT, fr) != ev.ref[q]) ? 1u : 0u;
        }
    }
    PXG_CHECK_MSG(tbSeen == 0u, "%s: %u (query, tetBound twin) pairs differ -- the bound is not kernel-blind", tag, tbSeen);

    // ---- GPU: the engine kernel on the engine descriptors -----------------------------------
    PxgSlangComputeDevice* dev = testDevice();
    std::vector<uint32_t> treeBlob(4 + sc.nodes.size() * (sizeof(Bv32Node) / 4), 0u);
    treeBlob[0] = uint32_t(sc.verts.size());
    treeBlob[1] = uint32_t(sc.tris.size());
    treeBlob[2] = 2u;
    treeBlob[3] = uint32_t(sc.nodes.size());
    std::memcpy(treeBlob.data() + 4, sc.nodes.data(), sc.nodes.size() * sizeof(Bv32Node));
    DevBuf bTree = makeDev(dev, treeBlob), bTriV = makeDev(dev, sc.verts), bTriI = makeDev(dev, sc.tris);
    if (!bTree.addr)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        for (PxgBufferHandle h : { bTree.h, bTriV.h, bTriI.h }) dev->destroyBuffer(h);
        return;
    }
    physx::PxgFEMCloth cloth;
    std::memset(&cloth, 0, sizeof(cloth));
    cloth.mTriMeshData = reinterpret_cast<void*>(size_t(bTree.addr));
    cloth.mPosition_InvMass = reinterpret_cast<decltype(cloth.mPosition_InvMass)>(size_t(bTriV.addr));
    cloth.mTriangleVertexIndices_ownedEdges = reinterpret_cast<decltype(cloth.mTriangleVertexIndices_ownedEdges)>(size_t(bTriI.addr));
    cloth.mNbVerts = uint32_t(sc.verts.size());
    cloth.mNbTriangles = uint32_t(sc.tris.size());
    const std::vector<physx::PxgFEMCloth> clothArr = { cloth };
    DevBuf bCloth = makeDev(dev, clothArr);

    DevBuf bTetV = makeDev(dev, tetWorld), bTetI = makeDev(dev, tetIdx);
    const std::vector<uint32_t> hintWords((NQ + 3u) / 4u + 1u, 0x01010101u);
    DevBuf bHint = makeDev(dev, hintWords);
    std::vector<physx::PxgSoftBody> sbs(NQ);
    for (uint32_t q = 0; q < NQ; ++q)
    {
        physx::PxgSoftBody& sb = sbs[q];
        std::memset(&sb, 0, sizeof(sb));
        sb.mTetIndices = reinterpret_cast<decltype(sb.mTetIndices)>(size_t(bTetI.addr + 16ull * q));
        sb.mPosition_InvMass = reinterpret_cast<decltype(sb.mPosition_InvMass)>(size_t(bTetV.addr));
        sb.mTetMeshSurfaceHint = reinterpret_cast<decltype(sb.mTetMeshSurfaceHint)>(size_t(bHint.addr + q));
        sb.mNumTets = 1u;
        sb.mNumVerts = 4u * NQ;
    }
    DevBuf bSb = makeDev(dev, sbs);

    // shape 0 = the cloth, shape 1 + q = softbody q; transform/contact-distance slot q = softbody
    // q, slot NQ = the cloth
    std::vector<ShapeH> shapes(NQ + 1);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(ShapeH));
    shapes[0].type = uint32_t(physx::PxGeometryType::eTRIANGLEMESH);
    shapes[0].srw = 1.0f;
    for (uint32_t q = 0; q < NQ; ++q)
    {
        shapes[1 + q].type = uint32_t(physx::PxGeometryType::eTETRAHEDRONMESH);
        shapes[1 + q].particleOrSoftbodyId = q;
        shapes[1 + q].srw = 1.0f;
    }
    std::vector<CmIn> cms(NQ);
    for (uint32_t q = 0; q < NQ; ++q)
        cms[q] = (q & 1u) ? CmIn{ 0u, 1u + q, NQ, q } : CmIn{ 1u + q, 0u, q, NQ };
    std::vector<float> xf(size_t(NQ + 1) * 8u, 0.0f);   // PxsCachedTransform: q @0, p @16, flags @28
    for (uint32_t i = 0; i <= NQ; ++i) xf[8u * i + 3u] = 1.0f;
    for (int k = 0; k < 4; ++k) xf[8u * NQ + k] = Q[k];
    xf[8u * NQ + 4u] = T.x; xf[8u * NQ + 5u] = T.y; xf[8u * NQ + 6u] = T.z;
    std::vector<float> cd(NQ + 1, 0.0f);
    for (uint32_t q = 0; q < NQ; ++q) cd[q] = gen.work[q].params.x;
    DevBuf bShapes = makeDev(dev, shapes), bCms = makeDev(dev, cms), bXf = makeDev(dev, xf), bCd = makeDev(dev, cd);

    const uint32_t CAP = NQ * 4 + 16;
    const std::vector<U4> stackInit(CAP, { 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu });
    const std::vector<uint32_t> cnt(1, 0u);
    DevBuf bStack = makeDev(dev, stackInit), bCnt = makeDev(dev, cnt);

    SbClothMidPush push = {};
    push.toleranceLength = 1.0f;
    push.cmInputs = bCms.addr;
    push.transformCache = bXf.addr;
    push.bounds = 0ull;
    push.contactDistance = bCd.addr;
    push.gpuShapes = bShapes.addr;
    push.softbodies = bSb.addr;
    push.clothes = bCloth.addr;
    push.stackSizeBytes = CAP * 16u;
    push.stackPtr = bStack.addr;
    push.midphasePairsNum = bCnt.addr;
    {
        Kernel k("softBodyClothMidphaseBda.spv", "sb_clothMidphaseGeneratePairsLaunch", {}, sizeof(SbClothMidPush));
        // the engine grid (PxgNarrowphaseCore.cpp): x = SB_SBMIDPHASE = 1024 blocks of two warps,
        // y = numTests; the port launches one group per CUDA warp (PXG_NO_CUDA_TOOLKIT), x = 2048
        k.dispatch(2048u, {}, push, NQ, 1u);
        dev->flush();
    }
    const auto gS = downloadBuffer<U4>(dev, bStack.h, CAP);
    const uint32_t gCount = downloadBuffer<uint32_t>(dev, bCnt.h, 1)[0];
    for (PxgBufferHandle h : { bTree.h, bTriV.h, bTriI.h, bCloth.h, bTetV.h, bTetI.h, bHint.h, bSb.h, bShapes.h,
                               bCms.h, bXf.h, bCd.h, bStack.h, bCnt.h })
        dev->destroyBuffer(h);

    // the stack in slot order: each query's reservation is contiguous, so this is emission order
    std::vector<std::vector<uint32_t>> gpu(NQ);
    size_t bogus = 0;
    for (uint32_t i = 0; i < gCount && i < CAP; ++i)
    {
        if (gS[i].x < NQ && gS[i].y == 0u && gS[i].w == 0u) gpu[gS[i].x].push_back(gS[i].z);
        else ++bogus;
    }
    checkNaNSelectGpu(tag, ev, gen.fam, gpu, gCount, bogus);
}

PXG_TEST(softBodySimClothMidphase_p89SatProbe)
{
    using namespace p89tt;
    using namespace p89sbc;
    const char* tag = "p89 sbCloth SAT probe";
    std::vector<TpRec> sat;
    std::vector<int> satFam;
    uint32_t satMissing = 0;
    genSatProbeRecords(0x5bc1a7e1ull, /*cull*/ false, sat, satFam, satMissing);
    std::vector<TbRec> tb;
    std::vector<int> tbFam;
    uint32_t tbMissing = 0;
    genTbRecords(0x7e7b0dd1ull, sat.size(), tb, tbFam, tbMissing);
    PXG_CHECK_MSG(satMissing == 0u && tbMissing == 0u,
                  "%s: planted (site, zero/non-finite) classes with fewer than four records -- SAT %u, tetBound %u", tag,
                  satMissing, tbMissing);
    PXG_CHECK_MSG(tb.size() == sat.size(), "%s: %zu tetBound records against %zu SAT records", tag, tb.size(), sat.size());
    if (tb.size() != sat.size()) return;
    const uint32_t N = uint32_t(sat.size());
    std::vector<SbcRec> recs(N);
    for (uint32_t i = 0; i < N; ++i) { recs[i].sat = sat[i]; recs[i].tb = tb[i]; }

    // ---- the battery has to SEE every fold and every operand order ----------------------------
    uint32_t blind = 0, twinWords = 0;
    std::printf("    [%s] %u records; words each single-site twin changes:\n", tag, N);
    for (uint32_t s = 0; s < 6; ++s)
    {
        uint32_t rvW = 0, swW = 0;
        for (uint32_t i = 0; i < N; ++i)
        {
            rvW += sbcDiffWords(recs[i], 1u << s);
            swW += sbcDiffWords(recs[i], (1u << s) << SWAP_SHIFT);
        }
        std::printf("      %-20s revert %3u  swap %3u\n", kSiteName[s], rvW, swW);
        if (rvW < 4u || swW < 4u) ++blind;
    }
    for (uint32_t s = 0; s < TB_COUNT; ++s)
    {
        const uint32_t site = 1u << (TB_SHIFT + s);
        uint32_t rvW = 0, swW = 0;
        for (uint32_t i = 0; i < N; ++i)
        {
            rvW += sbcDiffWords(recs[i], site);
            swW += sbcDiffWords(recs[i], site << SWAP_SHIFT);
        }
        std::printf("      %-20s revert %3u  swap %3u\n", kTbSiteName[s], rvW, swW);
        if (rvW < 4u || swW < 4u) ++blind;
    }
    const uint32_t OLD = SITE_ALL | TB_ALL;
    for (uint32_t i = 0; i < N; ++i) twinWords += sbcDiffWords(recs[i], OLD);
    std::printf("      old FMin/FMax twin: %u words\n", twinWords);
    PXG_CHECK_MSG(blind == 0u, "%s: %u fold sites are changed on fewer than four words by their revert or swap", tag,
                  blind);

    // ---- run the probe ---------------------------------------------------------------------------
    PxgSlangComputeDevice* dev = testDevice();
    std::vector<F4> in(size_t(N) * 13u);
    std::memcpy(in.data(), recs.data(), sizeof(SbcRec) * N);
    const std::vector<U4> outInit(size_t(N) * 3u, { 0x7f800001u, 1u, 2u, 3u });
    DevBuf bIn = makeDev(dev, in), bOut = makeDev(dev, outInit);
    if (!bIn.addr)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        dev->destroyBuffer(bIn.h); dev->destroyBuffer(bOut.h);
        return;
    }
    struct ProbePush { uint64_t input, output; uint32_t numRecords, pad20, pad24, pad28; };
    static_assert(sizeof(ProbePush) == 32, "SbClothSatProbeParams is 32 bytes");
    const ProbePush pp = { bIn.addr, bOut.addr, N, 0u, 0u, 0u };
    {
        Kernel k("softBodyClothMidphaseBda.spv", "sb_clothMidphaseSatProbe", {}, sizeof(ProbePush));
        k.dispatch((N + 31u) / 32u, {}, pp);
        dev->flush();
    }
    const auto g = downloadBuffer<U4>(dev, bOut.h, size_t(N) * 3u);
    dev->destroyBuffer(bIn.h); dev->destroyBuffer(bOut.h);

    uint32_t badSat[PFAM_COUNT] = {}, badTb[TBFAM_COUNT] = {}, badWords = 0, asOld = 0, badPad = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        float c[10], t[10];
        sbcProbeRef(recs[i], 0u, c);
        sbcProbeRef(recs[i], OLD, t);
        const uint32_t gw[12] = { g[3 * i].x, g[3 * i].y, g[3 * i].z, g[3 * i].w,
                                  g[3 * i + 1].x, g[3 * i + 1].y, g[3 * i + 1].z, g[3 * i + 1].w,
                                  g[3 * i + 2].x, g[3 * i + 2].y, g[3 * i + 2].z, g[3 * i + 2].w };
        badPad += (gw[10] != 0u || gw[11] != 0u) ? 1u : 0u;
        bool satBad = false, tbBad = false;
        for (int w = 0; w < 10; ++w)
        {
            float gv;
            std::memcpy(&gv, &gw[w], 4);
            if (sameWord(gv, c[w])) continue;
            (w < 4 ? satBad : tbBad) = true;
            ++badWords;
            if (sameWord(gv, t[w])) ++asOld;
            if (badWords <= 6)
                std::printf("    %s rec %u word %d: gpu %08x (%g) cuda %08x (%g) old twin %08x (%g)\n", tag, i, w, gw[w],
                            gv, f2u(c[w]), c[w], f2u(t[w]), t[w]);
        }
        if (satBad) ++badSat[satFam[i]];
        if (tbBad) ++badTb[tbFam[i]];
    }
    PXG_CHECK_MSG(badPad == 0u, "%s: %u records with a non-zero pad word", tag, badPad);
    PXG_CHECK_MSG(badWords == 0u,
                  "%s: %u words differ from the CUDA's select folds (%u of them equal the old FMin/FMax twin) -- "
                  "records by family: minProject %u, sep %u, axis-tri %u, random %u; tetBound planted %u, random %u",
                  tag, badWords, asOld, badSat[MPPAT], badSat[SEPPAT], badSat[AXTRI], badSat[RANDOM], badTb[TBPAT],
                  badTb[TBRANDOM]);
}

} // namespace
