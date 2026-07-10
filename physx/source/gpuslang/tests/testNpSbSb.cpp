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

// Verification of npSbSb.slang (SbSbMeshTreeTraverser - the softbody-vs-
// softbody midphase on the BV32 walk: query vertices of softbody0 against
// the tet mesh + BV32 tree of softbody1).  Same traversal and slack
// machinery as testNpSbSelf.cpp (error-scaled margins, candidate-gated
// promotion, unconditional promotion of an ambiguous inside flag), but the
// contact pipeline follows the SbSb form: no neighbour exclusion, no
// rest-pose recheck, no stress gate; the outside branch is gated on
// v.dot(normal) < 0 with the UN-negated normal and picks
// (v_hat.dot(normal) < -0.707) ? v_hat : -normal.
//
// Queries mix directed centroid points (inside-path coverage), near-surface
// points sampled off hinted AND non-hinted faces (outside contacts, both
// ternary branches, vd-gate blocks) and random points.  Filter pairs target
// the tets NEAR the query point so a broken lookup manufactures contacts.
// Contacts are matched exactly by the injective encoded pairId0; the
// atomic counter is bounded by the reference contact count +/- skips.

#include "pxgTestFramework.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <random>
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
struct PairNR { uint32_t i0lo, i0hi, index1, index2; };
struct SsbWork { F4 point; U4 ids; F4 params; };
struct SsbContact { F4 closestP0, normal_pen, point, closestP1; U4 pairIds; };
struct SsbParams { uint32_t numWorkItems, numFilterPairs, maxContacts, pad0; };

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

// Error-scaled slack: a comparison is "near" when |l - r| falls within ~250x
// the float rounding error of the computation, estimated from the L1
// magnitudes of the inputs (an absolute margin would systematically flag the
// tiny plane-products of these small tets).
struct Slack { bool near = false; void noteSc(float l, float r, float scale) { if (std::fabs(l - r) < 3e-5f * scale) near = true; } };

static float l1n(V3 v) { return std::fabs(v.x) + std::fabs(v.y) + std::fabs(v.z); }

static V3 getNormalized(V3 v)
{
    const float m = dot3(v, v);
    return m > 0 ? muls(v, 1.0f / std::sqrt(m)) : V3{ 0, 0, 0 };
}
static bool refValidTet(V3 a, V3 b, V3 c, V3 d) { return dot3(sub(b, a), cross3(sub(c, a), sub(d, a))) > 1e-9f; }

// ambiguous: some plane product is within rounding of zero AND no other
// product is certainly negative — i.e. the inside FLAG itself could flip
// between the GPU and CPU evaluations.  A near-zero product next to a
// strongly negative one cannot change the conjunction, so it is not flagged
// (an absolute per-product flag would skip nearly every query).  The CUDA
// inside path has NO distance gate, so an ambiguous inside flag must be
// promoted to a query skip UNCONDITIONALLY by the caller (a flip to inside
// on a far tet still produces a contact candidate).
static bool refInside(V3 p, V3 a, V3 b, V3 c, V3 d, bool& ambiguous)
{
    const V3 ap = sub(p, a), ab = sub(b, a), ac = sub(c, a), ad = sub(d, a);
    const float lab = l1n(ab), lac = l1n(ac), lad = l1n(ad), lap = l1n(ap);
    const V3 v0 = cross3(ab, ac);
    const float s0 = dot3(v0, ap) * dot3(v0, ad);
    const V3 v1 = cross3(ac, ad);
    const float s1 = dot3(v1, ap) * dot3(v1, ab);
    const V3 v2 = cross3(ad, ab);
    const float s2 = dot3(v2, ap) * dot3(v2, ac);
    const V3 bd = sub(d, b), bc = sub(c, b), bp = sub(p, b), ba = sub(a, b);
    const V3 v3v = cross3(bd, bc);
    const float s3 = dot3(v3v, bp) * dot3(v3v, ba);
    // per-product error scale: |cross| bound * factor bounds on each side
    const float s[4] = { s0, s1, s2, s3 };
    const float sc[4] = {
        (lab * lac * lap) * (lab * lac * lad),
        (lac * lad * lap) * (lac * lad * lab),
        (lad * lab * lap) * (lad * lab * lac),
        (l1n(bd) * l1n(bc) * l1n(bp)) * (l1n(bd) * l1n(bc) * l1n(ba)),
    };
    bool nearAny = false, certainNeg = false;
    for (int k = 0; k < 4; ++k)
    {
        if (std::fabs(s[k]) < 3e-5f * sc[k]) nearAny = true;
        else if (s[k] < 0) certainNeg = true;
    }
    ambiguous = nearAny && !certainNeg;
    return s0 >= 0 && s1 >= 0 && s2 >= 0 && s3 >= 0;
}

static V3 refTriClosest(V3 p, V3 a, V3 b, V3 c, V3 ab, V3 ac)
{
    const V3 ap = sub(p, a);
    const float d1 = dot3(ab, ap), d2 = dot3(ac, ap);
    if (d1 <= 0 && d2 <= 0) return a;
    const V3 bp = sub(p, b);
    const float d3 = dot3(ab, bp), d4 = dot3(ac, bp);
    if (d3 >= 0 && d4 <= d3) return b;
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) { const float v = d1 / (d1 - d3); return add(a, muls(ab, v)); }
    const V3 cp = sub(p, c);
    const float d5 = dot3(ab, cp), d6 = dot3(ac, cp);
    if (d6 >= 0 && d5 <= d6) return c;
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) { const float w = d2 / (d2 - d6); return add(a, muls(ac, w)); }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0)
    { const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6)); return add(b, muls(sub(c, b), w)); }
    const float denom = 1.0f / (va + vb + vc);
    return add(add(a, muls(ab, vb * denom)), muls(ac, vc * denom));
}

// outBlow: max over the evaluated faces of the normalization error
// amplification |e0|*|e1|/|cross| (feeds the downstream normal-dependent
// slack scales).  Face ties and the orientation dot are only noted when the
// caller USES the winning face's normal (the rest-pose recheck does not — it
// only consumes the continuous min distance and the inside flag — so it
// passes a dummy Slack).
static void refClosestTet(V3 point, uint32_t hint, V3 a, V3 b, V3 c, V3 d,
                          V3& normal, V3& closestP, float& sqDist, Slack& sl, float& outBlow)
{
    const V3 ac = sub(c, a), ab = sub(b, a), ad = sub(d, a);
    struct FaceDef { V3 fa, fb, fc, e0, e1, cn, opp; bool flipGt; };
    const V3 bd = sub(d, b), bc = sub(c, b);
    const FaceDef faces[4] = {
        { a, b, c, ab, ac, cross3(ac, ab), ad, false },
        { a, b, d, ab, ad, cross3(ab, ad), ac, false },
        { a, c, d, ac, ad, cross3(ac, ad), ab, false },
        { b, c, d, bc, bd, cross3(bc, bd), ab, true },
    };
    for (int k = 0; k < 4; ++k)
    {
        if (!((hint >> k) & 1)) continue;
        const FaceDef& f = faces[k];
        const float blow = l1n(f.e0) * l1n(f.e1) / (std::sqrt(dot3(f.cn, f.cn)) + 1e-30f);
        outBlow = std::max(outBlow, blow);
        const V3 tp = refTriClosest(point, f.fa, f.fb, f.fc, f.e0, f.e1);
        const V3 dv = sub(tp, point);
        const float tSq = dot3(dv, dv);
        if (sqDist < 1e30f)
            sl.noteSc(tSq, sqDist, l1n(dv) * (l1n(tp) + l1n(point)) + tSq + sqDist);
        if (tSq < sqDist)
        {
            V3 n = getNormalized(f.cn);
            const float od = dot3(f.opp, n);
            sl.noteSc(od, 0.0f, l1n(f.opp) * (1.0f + blow));
            if (f.flipGt ? od > 0.0f : od < 0.0f) n = neg(n);
            normal = n;
            closestP = tp;
            sqDist = tSq;
        }
    }
}

static uint32_t encodeSb(uint32_t id, uint32_t tet) { return (id << 20) | tet; }
static bool refFind(const std::vector<PairNR>& pairs, uint32_t id0, uint32_t id1)
{
    uint32_t l = 0, r = (uint32_t)pairs.size();
    while (l < r)
    {
        const uint32_t mid = (l + r) / 2;
        const PairNR& p = pairs[mid];
        const uint64_t p0 = ((uint64_t)p.i0hi << 32) | p.i0lo;
        int cmp = 0;
        if ((uint64_t)id0 < p0) cmp = -1;
        else if ((uint64_t)id0 > p0) cmp = 1;
        else if (id1 < p.index1) cmp = -1;
        else if (id1 > p.index1) cmp = 1;
        else if (0u < p.index2) cmp = -1;
        if (cmp == 0 || ((uint64_t)id0 == p0 && id1 == p.index1)) return true;
        if (cmp < 0) r = mid; else l = mid + 1;
    }
    return false;
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


PXG_TEST(npSbSb_traverser)
{
    std::mt19937 rng(85017);
    std::uniform_real_distribution<float> uv(-1, 1);
    std::uniform_real_distribution<float> u01(0, 1);

    // softbody1: 500 local tets, 4 dedicated verts each
    const uint32_t NT = 500, NQ = 96, NV0 = 2000;
    std::vector<F4> verts(NT * 4);
    std::vector<U4> tets(NT);
    std::vector<uint32_t> hints(NT), v2t0(NV0);
    std::vector<AABB> bounds(NT);
    for (uint32_t t = 0; t < NT; ++t)
    {
        const V3 c0 = { uv(rng), uv(rng), uv(rng) };
        for (;;)
        {
            for (int k = 0; k < 4; ++k)
                verts[t * 4 + k] = { c0.x + 0.18f * uv(rng), c0.y + 0.18f * uv(rng), c0.z + 0.18f * uv(rng), 0 };
            if (refValidTet(v3(verts[t * 4]), v3(verts[t * 4 + 1]), v3(verts[t * 4 + 2]), v3(verts[t * 4 + 3]))) break;
        }
        tets[t] = { t * 4, t * 4 + 1, t * 4 + 2, t * 4 + 3 };
        // mostly single-face hints (multi-bit face ties on shared closest
        // features are genuine instabilities - see testNpSbSelf.cpp)
        hints[t] = (t % 7 == 6) ? 0u : ((t % 11 == 5) ? (1u + rng() % 15) : (1u << (rng() % 4)));
        AABB bb = { { 1e30f, 1e30f, 1e30f }, { -1e30f, -1e30f, -1e30f } };
        for (int k = 0; k < 4; ++k)
        {
            const float* p = &verts[t * 4 + k].x;
            for (int j = 0; j < 3; ++j) { bb.mn[j] = std::min(bb.mn[j], p[j]); bb.mx[j] = std::max(bb.mx[j], p[j]); }
        }
        bounds[t] = bb;
    }
    for (uint32_t v = 0; v < NV0; ++v) v2t0[v] = v % NT; // softbody0 surface remap (injective per query)

    // Morton sort tets
    {
        auto expand = [](uint32_t v) {
            v &= 0x3ff; v = (v | (v << 16)) & 0x30000ff; v = (v | (v << 8)) & 0x300f00f;
            v = (v | (v << 4)) & 0x30c30c3; v = (v | (v << 2)) & 0x9249249; return v;
        };
        std::vector<uint32_t> order(NT);
        std::vector<uint64_t> code(NT);
        for (uint32_t t = 0; t < NT; ++t)
        {
            order[t] = t;
            uint32_t m[3];
            for (int k = 0; k < 3; ++k)
            {
                const float c = 0.5f * (bounds[t].mn[k] + bounds[t].mx[k]);
                m[k] = (uint32_t)std::min(1023.0f, std::max(0.0f, (c + 1.4f) / 2.8f * 1023.0f));
            }
            code[t] = ((uint64_t)expand(m[0]) << 2) | ((uint64_t)expand(m[1]) << 1) | expand(m[2]);
        }
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return code[a] < code[b]; });
        std::vector<U4> t2(NT); std::vector<AABB> b2(NT); std::vector<uint32_t> h2(NT);
        for (uint32_t t = 0; t < NT; ++t) { t2[t] = tets[order[t]]; b2[t] = bounds[order[t]]; h2[t] = hints[order[t]]; }
        tets.swap(t2); bounds.swap(b2); hints.swap(h2);
    }
    std::vector<Bv32Node> nodes;
    buildNode(nodes, bounds, 0, NT, rng, 0);

    // queries
    const uint32_t sbId0 = 7, sbId1 = 13;
    std::vector<SsbWork> work(NQ);
    std::vector<PairNR> filters;
    {
        auto popcount4 = [](uint32_t v) { int n = 0; for (; v; v >>= 1) n += v & 1; return n; };
        std::vector<std::pair<uint32_t, uint32_t>> fset;
        for (uint32_t q = 0; q < NQ; ++q)
        {
            const uint32_t vertIdx = (q * 977) % NV0;
            SsbWork& w = work[q];
            w.ids = { vertIdx, sbId0, sbId1, 0 };
            w.params = { 0.08f + 0.15f * u01(rng), 0, 0, 0 };
            V3 p = { 1.2f * uv(rng), 1.2f * uv(rng), 1.2f * uv(rng) };
            if (q % 8 == 5)
            {
                // directed inside-path fixture: centroid of a single-hint tet
                for (uint32_t k = 0, t = (q * 41) % NT; k < NT; ++k, t = (t + 1) % NT)
                {
                    if (hints[t] == 0 || popcount4(hints[t]) != 1) continue;
                    const V3 a = v3(verts[tets[t].x]), b = v3(verts[tets[t].y]), c = v3(verts[tets[t].z]), d = v3(verts[tets[t].w]);
                    if (!refValidTet(a, b, c, d)) continue;
                    const V3 cen = muls(add(add(a, b), add(c, d)), 0.25f);
                    bool amb = false;
                    if (!refInside(cen, a, b, c, d, amb) || amb) continue;
                    p = cen;
                    break;
                }
            }
            else if (q % 2 == 1)
            {
                // near-surface point: face point + outward offset off a
                // random face (hinted or not -> outside contacts, both
                // ternary branches, vd-gate blocks)
                const uint32_t t = (q * 131) % NT;
                const V3 vs[4] = { v3(verts[tets[t].x]), v3(verts[tets[t].y]), v3(verts[tets[t].z]), v3(verts[tets[t].w]) };
                const int fj[4][4] = { {0,1,2,3}, {0,1,3,2}, {0,2,3,1}, {1,2,3,0} };
                const int j = rng() % 4;
                float u = 0.1f + 0.8f * u01(rng), vb = 0.1f + 0.8f * u01(rng);
                if (u + vb > 0.9f) { u *= 0.45f; vb *= 0.45f; }
                const V3 fa = vs[fj[j][0]], fb = vs[fj[j][1]], fc = vs[fj[j][2]], fo = vs[fj[j][3]];
                const V3 fp = add(fa, add(muls(sub(fb, fa), u), muls(sub(fc, fa), vb)));
                V3 no = getNormalized(cross3(sub(fb, fa), sub(fc, fa)));
                if (dot3(no, sub(fo, fp)) > 0) no = neg(no);
                p = add(fp, muls(no, 0.005f + 0.1f * u01(rng)));
            }
            w.point = f4(p);
            if (q % 6 == 3)
            {
                // filter the tets NEAR the query point (would-be winners)
                uint32_t added = 0;
                for (uint32_t t = 0; t < NT && added < 40; ++t)
                {
                    float d2 = 0;
                    const float pc[3] = { p.x, p.y, p.z };
                    for (int jj = 0; jj < 3; ++jj)
                    {
                        const float cl = std::max(bounds[t].mn[jj], std::min(bounds[t].mx[jj], pc[jj]));
                        d2 += (cl - pc[jj]) * (cl - pc[jj]);
                    }
                    if (d2 < w.params.x * w.params.x)
                    {
                        fset.push_back({ encodeSb(sbId0, vertIdx), encodeSb(sbId1, t) });
                        ++added;
                    }
                }
            }
        }
        std::sort(fset.begin(), fset.end());
        fset.erase(std::unique(fset.begin(), fset.end()), fset.end());
        // the CUDA passes (mask0, mask1) UNordered to find() - build with that key order
        for (const auto& f : fset)
            filters.push_back({ f.first, 0, f.second, 0 });
        std::sort(filters.begin(), filters.end(), [](const PairNR& a, const PairNR& b) {
            if (a.i0lo != b.i0lo) return a.i0lo < b.i0lo;
            return a.index1 < b.index1; });
    }

    // ---- reference ----
    struct RefContact { bool has = false; V3 closestP, normal; float pen; uint32_t pairId1; int tag = -1; bool obs = false, vdClose = false; };
    std::vector<RefContact> refC(NQ);
    std::vector<uint8_t> skip(NQ, 0);
    size_t hintBlocked = 0, filtered = 0, contacts = 0;
    size_t insideContacts = 0, outsideContacts = 0, vdBlocked = 0, normReplaced = 0, normKept = 0;

    for (uint32_t q = 0; q < NQ; ++q)
    {
        const SsbWork& w = work[q];
        const V3 point = v3(w.point);
        const uint32_t vertIdx = w.ids.x;
        const float distance = w.params.x;
        Slack sl;

        // slack-tracked traversal
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
                        V3 cl;
                        cl.x = std::max(node.mMin[i].x, std::min(node.mMax[i].x, point.x));
                        cl.y = std::max(node.mMin[i].y, std::min(node.mMax[i].y, point.y));
                        cl.z = std::max(node.mMin[i].z, std::min(node.mMax[i].z, point.z));
                        const V3 dv = sub(cl, point);
                        const float d2 = dot3(dv, dv);
                        const float dd = distance * distance;
                        sl.noteSc(d2, dd, l1n(dv) * (l1n(point) + l1n(cl)) + d2 + dd);
                        if (!(d2 < dd)) continue;
                        const uint32_t data = node.mData[i];
                        stack.push_back((data & 1) ? ((cur & 0xFFFFFFC0u) | (i << 1) | 1) : ((data >> 11) << 6));
                    }
                }
                if (!stack.empty()) { cur = stack.back(); stack.pop_back(); }
                else hasNode = false;
            }
        }

        // per-lane pipeline (SbSb form)
        float laneBest[32];
        V3 laneP[32], laneN[32];
        uint32_t lanePrim[32];
        int laneTag[32];      // 0 = inside, 1 = outside/normal-replaced, 2 = outside/normal-kept
        bool laneObs[32];     // replaced-branch winners: outcome differs observably from the kept branch
        for (int j = 0; j < 32; ++j)
        { laneBest[j] = MAXF; lanePrim[j] = 0; laneP[j] = { 0, 0, 0 }; laneN[j] = { 0, 0, 0 }; laneTag[j] = -1; laneObs[j] = false; }
        bool anyVdClose = false; // a vd-blocked tet closer than the contact distance

        for (const auto& rd : rounds)
        {
            for (uint32_t i = 0; i < rd.second && i < 32; ++i)
            {
                const uint32_t t = rd.first + i;
                const U4& ti = tets[t];
                const uint32_t hint = hints[t];
                if (hint == 0) { ++hintBlocked; continue; }
                if (refFind(filters, encodeSb(sbId0, vertIdx), encodeSb(sbId1, t))) { ++filtered; continue; }
                const V3 a = v3(verts[ti.x]), b = v3(verts[ti.y]), c = v3(verts[ti.z]), d = v3(verts[ti.w]);
                {
                    const float det = dot3(sub(b, a), cross3(sub(c, a), sub(d, a)));
                    if (std::fabs(det - 1e-9f) < 3e-5f * (l1n(sub(b, a)) * l1n(sub(c, a)) * l1n(sub(d, a))))
                        sl.near = true;
                }
                if (!refValidTet(a, b, c, d)) continue;

                Slack tetSl;
                bool insideAmb = false;
                const bool inside = refInside(point, a, b, c, d, insideAmb);
                if (insideAmb) sl.near = true; // the inside path has no distance gate
                V3 normal = { 0, 0, 0 }, closestP = { 0, 0, 0 };
                float sqDist2 = MAXF, blow = 0.0f;
                refClosestTet(point, hint, a, b, c, d, normal, closestP, sqDist2, tetSl, blow);
                V3 v = sub(point, closestP);
                const float magP = l1n(point) + l1n(closestP);

                bool bHasCollision = false;
                float dist = MAXF;
                int tag = 0;
                bool obs = false;
                if (inside)
                {
                    normal = neg(normal);
                    dist = dot3(v, normal);
                    bHasCollision = true;
                    ++insideContacts;
                }
                else
                {
                    const float vd = dot3(v, normal); // UN-negated normal
                    tetSl.noteSc(vd, 0.0f, magP * (1.0f + blow));
                    if (vd < 0.0f)
                    {
                        dist = std::sqrt(dot3(v, v));
                        if (dist > 0.0f) v = muls(v, 1.0f / dist); // PxVec3::normalize guard
                        const float nd = dot3(v, normal);
                        tetSl.noteSc(nd, -0.707f, 1.0f + (magP / std::max(dist, 1e-9f)) * (1.0f + blow));
                        if (nd < -0.707f)
                        {
                            // observable iff the replaced normal differs from
                            // the kept alternative beyond the compare tolerance
                            obs = l1n(add(v, normal)) > 2e-3f;
                            normal = v;
                            tag = 1;
                            ++normReplaced;
                        }
                        else { normal = neg(normal); tag = 2; ++normKept; }
                        tetSl.noteSc(dist, distance, magP + dist + distance);
                        bHasCollision = dist < distance;
                        if (bHasCollision) ++outsideContacts;
                    }
                    else
                    {
                        ++vdBlocked;
                        if (std::sqrt(dot3(v, v)) < distance) anyVdClose = true;
                    }
                }

                if (bHasCollision)
                {
                    tetSl.noteSc(dist, laneBest[i], magP * (1.0f + blow) + dist + std::min(laneBest[i], 100.0f));
                    if (dist < laneBest[i])
                    {
                        laneBest[i] = dist;
                        laneN[i] = normal;
                        laneP[i] = closestP;
                        lanePrim[i] = t;
                        laneTag[i] = tag;
                        laneObs[i] = obs;
                    }
                }

                const float gate = distance + 0.02f;
                if (tetSl.near && (inside || bHasCollision || sqDist2 < gate * gate))
                    sl.near = true;
            }
        }

        // finalize: lowest-lane argmin
        float best = MAXF;
        uint32_t winner = 0;
        for (int j = 0; j < 32; ++j)
        {
            if (laneBest[j] < best)
            {
                if (best < MAXF) sl.noteSc(laneBest[j], best, 20.0f);
                best = laneBest[j];
                winner = j;
            }
            else if (laneBest[j] < MAXF) sl.noteSc(laneBest[j], best, 20.0f);
        }
        if (best < MAXF) sl.noteSc(best, work[q].params.x, 20.0f);
        RefContact& rc = refC[q];
        if (best < work[q].params.x)
        {
            rc.has = true;
            rc.closestP = laneP[winner];
            rc.normal = laneN[winner];
            rc.pen = best;
            rc.pairId1 = encodeSb(sbId1, lanePrim[winner]);
            rc.tag = laneTag[winner];
            rc.obs = laneObs[winner];
            rc.vdClose = anyVdClose;
            ++contacts;
        }
        skip[q] = sl.near ? 1 : 0;
    }

    // ---- GPU run ----
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t CAP = NQ + 32;
    std::vector<SsbContact> ctInit(CAP);
    std::vector<uint32_t> cnt(1, 0);
    PxgBufferHandle bN = createBufferFrom(dev, nodes);
    PxgBufferHandle bT = createBufferFrom(dev, tets);
    PxgBufferHandle bV = createBufferFrom(dev, verts);
    PxgBufferHandle bH = createBufferFrom(dev, hints);
    PxgBufferHandle bM = createBufferFrom(dev, v2t0);
    PxgBufferHandle bF = createBufferFrom(dev, filters);
    PxgBufferHandle bW = createBufferFrom(dev, work);
    PxgBufferHandle bC = createBufferFrom(dev, ctInit);
    PxgBufferHandle bCc = createBufferFrom(dev, cnt);

    SsbParams prm = { NQ, (uint32_t)filters.size(), CAP, 0 };
    Kernel k("npSbSb.spv", "ctSbSbQuery",
             { (uint32_t)sizeof(Bv32Node), 16, 16, 4, 4, 16, (uint32_t)sizeof(SsbWork), (uint32_t)sizeof(SsbContact), 4 },
             sizeof(SsbParams));
    k.dispatch(NQ, { bN, bT, bV, bH, bM, bF, bW, bC, bCc }, prm);
    dev->flush();

    const auto gC = downloadBuffer<SsbContact>(dev, bC, CAP);
    const uint32_t gCount = downloadBuffer<uint32_t>(dev, bCc, 1)[0];
    dev->destroyBuffer(bN); dev->destroyBuffer(bT); dev->destroyBuffer(bV); dev->destroyBuffer(bH);
    dev->destroyBuffer(bM); dev->destroyBuffer(bF); dev->destroyBuffer(bW); dev->destroyBuffer(bC);
    dev->destroyBuffer(bCc);

    // group GPU contacts by pairId0 (injective per query)
    std::map<uint32_t, const SsbContact*> byPair;
    for (uint32_t i = 0; i < gCount && i < CAP; ++i)
        byPair[gC[i].pairIds.x] = &gC[i];

    size_t mm = 0, checked = 0, skipped = 0;
    size_t refSkippedContacts = 0;
    size_t checkedInsideWin = 0, checkedReplacedWin = 0, checkedKeptWin = 0, checkedVdClose = 0;
    for (uint32_t q = 0; q < NQ; ++q)
    {
        if (skip[q]) { ++skipped; if (refC[q].has) ++refSkippedContacts; continue; }
        ++checked;
        const RefContact& rc = refC[q];
        const uint32_t pid0 = encodeSb(sbId0, v2t0[work[q].ids.x]);
        auto it = byPair.find(pid0);
        if (rc.has)
        {
            if (rc.tag == 0) ++checkedInsideWin;
            if (rc.tag == 1 && rc.obs) ++checkedReplacedWin;
            if (rc.tag == 2) ++checkedKeptWin;
            if (rc.vdClose) ++checkedVdClose;
            if (it == byPair.end()) { ++mm; continue; }
            const SsbContact& g = *it->second;
            if (g.pairIds.y != rc.pairId1 ||
                !almostEqual(g.normal_pen.w, rc.pen, 1e-4f, 1e-4f) ||
                !almostEqual(g.closestP0.x, rc.closestP.x, 1e-4f, 1e-4f) ||
                !almostEqual(g.closestP0.y, rc.closestP.y, 1e-4f, 1e-4f) ||
                !almostEqual(g.closestP0.z, rc.closestP.z, 1e-4f, 1e-4f) ||
                !almostEqual(g.normal_pen.x, rc.normal.x, 1e-4f, 1e-4f) ||
                !almostEqual(g.normal_pen.y, rc.normal.y, 1e-4f, 1e-4f) ||
                !almostEqual(g.normal_pen.z, rc.normal.z, 1e-4f, 1e-4f) ||
                !almostEqual(g.closestP1.x, rc.closestP.x, 1e-4f, 1e-4f) ||
                !almostEqual(g.closestP1.y, rc.closestP.y, 1e-4f, 1e-4f) ||
                !almostEqual(g.closestP1.z, rc.closestP.z, 1e-4f, 1e-4f) ||
                g.point.x != work[q].point.x ||
                g.point.y != work[q].point.y ||
                g.point.z != work[q].point.z) ++mm;
        }
        else if (it != byPair.end()) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "npSbSb: %zu contact mismatches", mm);
    PXG_CHECK_MSG(gCount >= contacts - refSkippedContacts && gCount <= contacts - refSkippedContacts + skipped,
                  "npSbSb: counter %u outside [%zu, %zu]", gCount, contacts - refSkippedContacts,
                  contacts - refSkippedContacts + skipped);
    PXG_CHECK_MSG(checked > NQ * 2 / 3, "npSbSb: too many skipped (%zu)", skipped);
    PXG_CHECK_MSG(contacts > 15 && hintBlocked > 50 && filtered > 5,
                  "npSbSb: coverage thin (ct=%zu hint=%zu filt=%zu)", contacts, hintBlocked, filtered);
    PXG_CHECK_MSG(insideContacts > 3 && outsideContacts > 10 && vdBlocked > 10 && normReplaced > 3 && normKept > 3,
                  "npSbSb: branch coverage thin (in=%zu out=%zu vd=%zu rep=%zu kept=%zu)",
                  insideContacts, outsideContacts, vdBlocked, normReplaced, normKept);
    // the floors above count reference-side occurrences; these pin each branch
    // on a CHECKED winning contact that the GPU comparison actually verifies
    // (replaced-branch winners must differ observably from the kept branch)
    PXG_CHECK_MSG(checkedInsideWin >= 2 && checkedReplacedWin >= 2 && checkedKeptWin >= 2 && checkedVdClose >= 1,
                  "npSbSb: checked-winner coverage thin (in=%zu rep=%zu kept=%zu vdClose=%zu)",
                  checkedInsideWin, checkedReplacedWin, checkedKeptWin, checkedVdClose);
}

} // namespace
