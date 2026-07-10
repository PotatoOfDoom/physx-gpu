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

// Verification of npSbSelf.slang (SbMeshTreeTraverser — the softbody
// SELF-collision midphase on the BV32 walk).  Unlike the earlier traversers,
// the box callback here is ARITHMETIC (point-to-box clamped distance vs the
// contact distance), so the traversal itself is slack-tracked per query; the
// per-lane contact pipeline tracks every float gate (the inside/outside
// plane-products, v.dot(normal), the 0.707 normal replacement, dist <
// distance, the rest-pose newSqDist > filterDistanceSq gate, the
// closestPtTetrahedron running-min ties and the finalize argmin ties).
// Integer gates (surface hint, neighbour-vertex exclusion, filter pairs) and
// raw uploads (stress vs tolerance, both uploaded) are exact.
//
// Slack margins are ERROR-SCALED (L1-magnitude based, not a blunt absolute
// margin: the plane-products of these small tets are ~1e-4, an absolute 1e-4
// margin would skip every query), and per-tet events are only PROMOTED to the
// query's skip flag when the tet is a contact candidate (inside, colliding,
// or hinted-face distance within the contact distance + pad) — a near event
// on a clearly-out-of-range tet cannot change the output since the collision
// gate rejects the tet under either resolution of the ambiguity.  Queries
// with a promoted near event are skipped; per non-skipped query the GPU must
// emit exactly the reference contact (matched by the injective pairId0) or
// none.
//
// Directed rest-gate fixtures (adversarial-review hardening): with a uniform
// filterDistanceSq and rigid-copy rest poses both the !inside2 term and the
// 0xf all-faces constant of the rest recheck are dead code as far as random
// data can observe.  q%12==5 queries place the point at the centroid of a
// rest-identical tet with a tiny filter distance (!inside2 becomes the sole
// blocker); q%12==11 queries search for a point just outside a non-hinted
// face where the full-face rest distance falls below the filter gate while
// the hinted-face distance lies above it (a 0xf->hint mutation un-blocks).
// Both fixture kinds are coverage-asserted and their mutations verified
// caught.
//
// KNOWN DATA INVARIANT: the neighbour-vertex exclusion (skip tets containing
// the query vertex) is NOT independently observable here — such tets produce
// dist = 0 / inside = true, but the rest-pose recheck then sees restP as a
// vertex of the rest tet (all four plane products exactly 0 => inside2 =
// true) and blocks the contact, for any mesh whose rest state mirrors the
// topology.  A mutation dropping the exclusion therefore passes; the
// exclusion is exercised as a fast path (neighborBlocked coverage floor) but
// its absence is masked by the rest gate.

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
struct SsWork { F4 point, restP; U4 ids; F4 params; };
struct SsContact { F4 closestP0, normal_pen, point, closestP1; U4 pairIds; };
struct SsParams { uint32_t numWorkItems, numFilterPairs, maxContacts, pad0; };

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

PXG_TEST(npSbSelf_traverser)
{
    std::mt19937 rng(84013);
    std::uniform_real_distribution<float> uv(-1, 1);
    std::uniform_real_distribution<float> u01(0, 1);

    // tet mesh: 500 local tets, 4 dedicated verts each
    const uint32_t NT = 500, NQ = 96;
    std::vector<F4> verts(NT * 4), restVerts(NT * 4);
    std::vector<U4> tets(NT);
    std::vector<uint32_t> hints(NT), v2t(NT * 4);
    std::vector<float> stress(NT);
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
        for (int k = 0; k < 4; ++k)
        {
            // rest pose: half far from the deformed pose (rest gate passes), half close (gate blocks)
            const float off = (t % 2 == 0) ? 2.0f : 0.0f;
            restVerts[t * 4 + k] = { verts[t * 4 + k].x + off, verts[t * 4 + k].y, verts[t * 4 + k].z, 0 };
        }
        tets[t] = { t * 4, t * 4 + 1, t * 4 + 2, t * 4 + 3 };
        // Mostly single-face hints: with two hinted faces, a query point whose
        // closest feature is the shared edge produces an EXACT tSq tie whose
        // winner is float-order dependent — a genuine instability the slack
        // tracker must skip, so multi-bit hints are kept to a small subset.
        hints[t] = (t % 7 == 6) ? 0u : ((t % 11 == 5) ? (1u + rng() % 15) : (1u << (rng() % 4)));
        stress[t] = (t % 5 == 4) ? 2.0f : 0.1f * u01(rng); // some over the tolerance
        AABB bb = { { 1e30f, 1e30f, 1e30f }, { -1e30f, -1e30f, -1e30f } };
        for (int k = 0; k < 4; ++k)
        {
            const float* p = &verts[t * 4 + k].x;
            for (int j = 0; j < 3; ++j) { bb.mn[j] = std::min(bb.mn[j], p[j]); bb.mx[j] = std::max(bb.mx[j], p[j]); }
        }
        bounds[t] = bb;
    }
    for (uint32_t v = 0; v < NT * 4; ++v) v2t[v] = v % NT; // injective enough per query (queries use distinct verts)

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
        std::vector<U4> t2(NT); std::vector<AABB> b2(NT); std::vector<uint32_t> h2(NT); std::vector<float> s2(NT);
        for (uint32_t t = 0; t < NT; ++t) { t2[t] = tets[order[t]]; b2[t] = bounds[order[t]]; h2[t] = hints[order[t]]; s2[t] = stress[order[t]]; }
        tets.swap(t2); bounds.swap(b2); hints.swap(h2); stress.swap(s2);
    }
    std::vector<Bv32Node> nodes;
    buildNode(nodes, bounds, 0, NT, rng, 0);

    // queries: existing mesh vertices (neighbour exclusion fires) at various distances
    const uint32_t softbodyId = 42;
    const float stressTol = 1.0f;
    std::vector<SsWork> work(NQ);
    std::vector<PairNR> filters;
    size_t med3Fixtures = 0;
    {
        std::vector<std::pair<uint32_t, uint32_t>> fset;
        for (uint32_t q = 0; q < NQ; ++q)
        {
            const uint32_t vertIdx = (q * 977) % (NT * 4);
            SsWork& w = work[q];
            w.point = verts[vertIdx];
            w.restP = restVerts[vertIdx];
            w.ids = { vertIdx, softbodyId, 0, 0 };
            w.params = { 0.08f + 0.15f * u01(rng), 0.25f, stressTol, 0 };
            auto popcount4 = [](uint32_t v) { int n = 0; for (; v; v >>= 1) n += v & 1; return n; };
            auto sameParity = [&](uint32_t t) { return restVerts[tets[t].x].x == verts[tets[t].x].x; };
            auto touchesVert = [&](uint32_t t) {
                return tets[t].x == vertIdx || tets[t].y == vertIdx || tets[t].z == vertIdx || tets[t].w == vertIdx; };
            if (q % 12 == 5)
            {
                // Directed fixture (rest gate, !inside2 term): point = centroid
                // of a rest-identical tet -> inside in BOTH poses; filter
                // distance made tiny so the rest gate blocks through !inside2
                // ALONE (with the default 0.25 the nsq term always co-blocks
                // and a dropped !inside2 is unobservable).
                for (uint32_t k = 0, t = (q * 37) % NT; k < NT; ++k, t = (t + 1) % NT)
                {
                    if (hints[t] == 0 || popcount4(hints[t]) != 1 || stress[t] > stressTol) continue;
                    if (!sameParity(t) || touchesVert(t)) continue;
                    const V3 a = v3(verts[tets[t].x]), b = v3(verts[tets[t].y]), c = v3(verts[tets[t].z]), d = v3(verts[tets[t].w]);
                    if (!refValidTet(a, b, c, d)) continue;
                    const V3 cen = muls(add(add(a, b), add(c, d)), 0.25f);
                    bool amb = false;
                    if (!refInside(cen, a, b, c, d, amb) || amb) continue;
                    V3 nn = { 0, 0, 0 }, ncp = { 0, 0, 0 };
                    float nsq = MAXF, blow = 0.0f;
                    Slack dsl;
                    refClosestTet(cen, 0xf, a, b, c, d, nn, ncp, nsq, dsl, blow);
                    if (nsq < 4e-4f) continue; // keep nsq > filterDistSq robust
                    w.point = f4(cen);
                    w.restP = f4(cen);
                    w.params.y = 1e-4f;
                    break;
                }
            }
            if (q % 12 == 11)
            {
                // Directed fixture (rest gate, 0xf face set): point outside a
                // rest-identical single-hint tet such that the FULL-face rest
                // distance falls below the filter gate while the hinted-face
                // distance (what a 0xf->hint mutation would use) lies above it.
                for (uint32_t trial = 0; trial < 60000; ++trial)
                {
                    const uint32_t t = rng() % NT;
                    if (hints[t] == 0 || popcount4(hints[t]) != 1 || stress[t] > stressTol) continue;
                    if (!sameParity(t) || touchesVert(t)) continue;
                    const V3 a = v3(verts[tets[t].x]), b = v3(verts[tets[t].y]), c = v3(verts[tets[t].z]), d = v3(verts[tets[t].w]);
                    if (!refValidTet(a, b, c, d)) continue;
                    // sample p just OUTSIDE a non-hinted face: interior
                    // barycentric point + small offset along the outward normal
                    const V3 fv[4][4] = { // face verts + opposite vertex
                        { a, b, c, d }, { a, b, d, c }, { a, c, d, b }, { b, c, d, a } };
                    uint32_t hk = 0;
                    while (!((hints[t] >> hk) & 1)) ++hk;
                    const uint32_t j = (hk + 1 + rng() % 3) & 3; // any non-hinted face
                    if (j == hk) continue;
                    float u = 0.15f + 0.7f * u01(rng), vB = 0.15f + 0.7f * u01(rng);
                    if (u + vB > 0.85f) { u *= 0.4f; vB *= 0.4f; }
                    const V3 fp = add(fv[j][0], add(muls(sub(fv[j][1], fv[j][0]), u), muls(sub(fv[j][2], fv[j][0]), vB)));
                    V3 no = getNormalized(cross3(sub(fv[j][1], fv[j][0]), sub(fv[j][2], fv[j][0])));
                    if (dot3(no, sub(fv[j][3], fp)) > 0) no = neg(no); // orient away from the opposite vertex
                    const float eps = 0.003f + 0.009f * u01(rng);
                    const V3 p = add(fp, muls(no, eps));
                    bool amb = false;
                    if (refInside(p, a, b, c, d, amb) || amb) continue;
                    V3 n = { 0, 0, 0 }, cp = { 0, 0, 0 };
                    float sq = MAXF, blow = 0.0f;
                    Slack dsl;
                    refClosestTet(p, hints[t], a, b, c, d, n, cp, sq, dsl, blow);
                    const V3 v = sub(p, cp);
                    n = neg(n);
                    if (dot3(v, n) < 1e-4f) continue;
                    const float dist = std::sqrt(dot3(v, v));
                    if (dist < 0.02f || dist > 0.045f) continue;
                    const float nd = dot3(n, muls(v, 1.0f / dist));
                    if (std::fabs(nd - 0.707f) < 0.05f) continue;
                    V3 nn = { 0, 0, 0 }, ncp = { 0, 0, 0 };
                    float nsqF = MAXF, blow2 = 0.0f;
                    Slack d2sl;
                    refClosestTet(p, 0xf, a, b, c, d, nn, ncp, nsqF, d2sl, blow2);
                    // need full-face rest distance well below the hinted-face
                    // distance: fds = 3*nsqF gives nsqF < fds/3 (gate blocks
                    // robustly) and dist^2 > (10/3)*fds (a 0xf->hint mutation
                    // robustly un-blocks)
                    if (!(nsqF > 1e-8f && nsqF < 0.1f * dist * dist)) continue;
                    const float fds = 3.0f * nsqF;
                    w.point = f4(p);
                    w.restP = f4(p);
                    w.params.x = 0.05f;
                    w.params.y = fds;
                    ++med3Fixtures;
                    break;
                }
            }
            if (q % 6 == 3)
            {
                // Filter the tets NEAR the query point (the would-be contact
                // winners), so a broken filter lookup manufactures contacts.
                const V3 p = v3(w.point);
                uint32_t added = 0;
                for (uint32_t t = 0; t < NT && added < 40; ++t)
                {
                    float d2 = 0;
                    const float pc[3] = { p.x, p.y, p.z };
                    for (int j = 0; j < 3; ++j)
                    {
                        const float cl = std::max(bounds[t].mn[j], std::min(bounds[t].mx[j], pc[j]));
                        d2 += (cl - pc[j]) * (cl - pc[j]);
                    }
                    if (d2 < w.params.x * w.params.x)
                    {
                        fset.push_back({ encodeSb(softbodyId, vertIdx), encodeSb(softbodyId, t) });
                        ++added;
                    }
                }
            }
        }
        std::sort(fset.begin(), fset.end());
        fset.erase(std::unique(fset.begin(), fset.end()), fset.end());
        // NOTE: the CUDA passes (mask0, mask1) UNordered to find() here — transcribe faithfully:
        // SbMeshTreeTraverser calls find(pairs, nbPairs, softBodyMask0, softBodyMask1) directly,
        // so the pair list must be built with (mask0, mask1) key order, not min/max.
        for (const auto& f : fset)
            filters.push_back({ f.first, 0, f.second, 0 });
        std::sort(filters.begin(), filters.end(), [](const PairNR& a, const PairNR& b) {
            if (a.i0lo != b.i0lo) return a.i0lo < b.i0lo;
            return a.index1 < b.index1; });
    }

    // ---- reference ----
    struct RefContact { bool has = false; V3 closestP, normal; float pen; uint32_t pairId0, pairId1; };
    std::vector<RefContact> refC(NQ);
    std::vector<uint8_t> skip(NQ, 0);
    size_t hintBlocked = 0, neighborBlocked = 0, filtered = 0, restBlocked = 0, stressBlocked = 0, contacts = 0;
    size_t restInsideBlocked = 0; // blocked by !inside2 ALONE (nsq term passing)

    for (uint32_t q = 0; q < NQ; ++q)
    {
        const SsWork& w = work[q];
        const V3 point = v3(w.point), restP = v3(w.restP);
        const uint32_t vertIdx = w.ids.x;
        const float distance = w.params.x, filterDistSq = w.params.y;
        Slack sl;

        // slack-tracked traversal (box decisions are arithmetic here)
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

        // per-lane pipeline
        float laneBest[32];
        V3 laneP[32], laneN[32];
        uint32_t lanePrim[32];
        for (int j = 0; j < 32; ++j) { laneBest[j] = MAXF; lanePrim[j] = 0; laneP[j] = { 0, 0, 0 }; laneN[j] = { 0, 0, 0 }; }

        for (const auto& rd : rounds)
        {
            for (uint32_t i = 0; i < rd.second && i < 32; ++i)
            {
                const uint32_t t = rd.first + i;
                const U4& ti = tets[t];
                const uint32_t hint = hints[t];
                if (hint == 0) { ++hintBlocked; continue; }
                if (ti.x == vertIdx || ti.y == vertIdx || ti.z == vertIdx || ti.w == vertIdx) { ++neighborBlocked; continue; }
                if (refFind(filters, encodeSb(softbodyId, vertIdx), encodeSb(softbodyId, t))) { ++filtered; continue; }
                const V3 a = v3(verts[ti.x]), b = v3(verts[ti.y]), c = v3(verts[ti.z]), d = v3(verts[ti.w]);
                {
                    // validity gate: a flip admits/removes the whole tet, so a
                    // near event promotes unconditionally (never fires with
                    // this generator, which regenerates until det > 1e-9)
                    const float det = dot3(sub(b, a), cross3(sub(c, a), sub(d, a)));
                    if (std::fabs(det - 1e-9f) < 3e-5f * (l1n(sub(b, a)) * l1n(sub(c, a)) * l1n(sub(d, a))))
                        sl.near = true;
                }
                if (!refValidTet(a, b, c, d)) continue;

                // Per-tet slack, promoted to the query slack only when the tet
                // is a contact CANDIDATE (near events on a tet whose closest
                // hinted-face distance clearly exceeds the contact distance
                // cannot change the output — the collision gate rejects it for
                // either resolution of the ambiguity).
                Slack tetSl;
                bool insideAmb = false;
                const bool inside = refInside(point, a, b, c, d, insideAmb);
                if (insideAmb) sl.near = true; // unconditional: the inside path has no distance gate
                V3 normal = { 0, 0, 0 }, closestP = { 0, 0, 0 };
                float sqDist2 = MAXF, blow = 0.0f;
                refClosestTet(point, hint, a, b, c, d, normal, closestP, sqDist2, tetSl, blow);
                V3 v = sub(point, closestP);
                normal = neg(normal);
                const float magP = l1n(point) + l1n(closestP); // cancellation scale of v

                bool bHasCollision = false;
                float dist = MAXF;
                if (inside)
                {
                    bHasCollision = true;
                    dist = dot3(v, normal);
                }
                else
                {
                    const float vd = dot3(v, normal);
                    tetSl.noteSc(vd, 0.0f, magP * (1.0f + blow));
                    if (vd > 0.0f)
                    {
                        dist = std::sqrt(dot3(v, v));
                        v = muls(v, 1.0f / dist);
                        const float nd = dot3(normal, v);
                        tetSl.noteSc(nd, 0.707f, 1.0f + (magP / std::max(dist, 1e-9f)) * (1.0f + blow));
                        if (nd > 0.707f) normal = v;
                        tetSl.noteSc(dist, distance, magP + dist + distance);
                        bHasCollision = dist < distance;
                    }
                }

                if (bHasCollision)
                {
                    const V3 ra = v3(restVerts[ti.x]), rb = v3(restVerts[ti.y]), rc = v3(restVerts[ti.z]), rd2 = v3(restVerts[ti.w]);
                    V3 nn = { 0, 0, 0 }, ncp = { 0, 0, 0 };
                    float nsq = MAXF, restBlow = 0.0f;
                    Slack dummy; // face ties/orientation are output-neutral here (nn unused)
                    bool inside2Amb = false;
                    const bool inside2 = refInside(restP, ra, rb, rc, rd2, inside2Amb);
                    if (inside2Amb) tetSl.near = true; // promoted below (bHasCollision holds here)
                    refClosestTet(restP, 0xf, ra, rb, rc, rd2, nn, ncp, nsq, dummy, restBlow);
                    tetSl.noteSc(nsq, filterDistSq, l1n(sub(ncp, restP)) * (l1n(restP) + l1n(ncp)) + nsq + filterDistSq);
                    bHasCollision = !inside2 && nsq > filterDistSq;
                    if (!bHasCollision)
                    {
                        ++restBlocked;
                        if (inside2 && nsq > filterDistSq) ++restInsideBlocked;
                    }
                    if (bHasCollision && stress[t] > stressTol) { bHasCollision = false; ++stressBlocked; }
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
            rc.pairId0 = encodeSb(softbodyId, v2t[work[q].ids.x]);
            rc.pairId1 = encodeSb(softbodyId, lanePrim[winner]);
            ++contacts;
        }
        skip[q] = sl.near ? 1 : 0;
    }

    // ---- GPU run ----
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t CAP = NQ + 32;
    std::vector<SsContact> ctInit(CAP);
    std::vector<uint32_t> cnt(1, 0);
    PxgBufferHandle bN = createBufferFrom(dev, nodes);
    PxgBufferHandle bT = createBufferFrom(dev, tets);
    PxgBufferHandle bV = createBufferFrom(dev, verts);
    PxgBufferHandle bR = createBufferFrom(dev, restVerts);
    PxgBufferHandle bH = createBufferFrom(dev, hints);
    PxgBufferHandle bS = createBufferFrom(dev, stress);
    PxgBufferHandle bM = createBufferFrom(dev, v2t);
    PxgBufferHandle bF = createBufferFrom(dev, filters);
    PxgBufferHandle bW = createBufferFrom(dev, work);
    PxgBufferHandle bC = createBufferFrom(dev, ctInit);
    PxgBufferHandle bCc = createBufferFrom(dev, cnt);

    SsParams prm = { NQ, (uint32_t)filters.size(), CAP, 0 };
    Kernel k("npSbSelf.spv", "ctSbSelfQuery",
             { (uint32_t)sizeof(Bv32Node), 16, 16, 16, 4, 4, 4, 16, (uint32_t)sizeof(SsWork), (uint32_t)sizeof(SsContact), 4 },
             sizeof(SsParams));
    k.dispatch(NQ, { bN, bT, bV, bR, bH, bS, bM, bF, bW, bC, bCc }, prm);
    dev->flush();

    const auto gC = downloadBuffer<SsContact>(dev, bC, CAP);
    const uint32_t gCount = downloadBuffer<uint32_t>(dev, bCc, 1)[0];
    dev->destroyBuffer(bN); dev->destroyBuffer(bT); dev->destroyBuffer(bV); dev->destroyBuffer(bR);
    dev->destroyBuffer(bH); dev->destroyBuffer(bS); dev->destroyBuffer(bM); dev->destroyBuffer(bF);
    dev->destroyBuffer(bW); dev->destroyBuffer(bC); dev->destroyBuffer(bCc);

    // group GPU contacts by pairId0 (injective per query: distinct verts -> distinct v2t)
    std::map<uint32_t, const SsContact*> byPair;
    for (uint32_t i = 0; i < gCount && i < CAP; ++i)
        byPair[gC[i].pairIds.x] = &gC[i];

    size_t mm = 0, checked = 0, skipped = 0, sk = 0;
    size_t refSkippedContacts = 0;
    for (uint32_t q = 0; q < NQ; ++q)
    {
        if (skip[q]) { ++skipped; if (refC[q].has) ++refSkippedContacts; continue; }
        ++checked;
        const RefContact& rc = refC[q];
        const uint32_t pid0 = encodeSb(softbodyId, v2t[work[q].ids.x]);
        auto it = byPair.find(pid0);
        if (rc.has)
        {
            if (it == byPair.end()) { ++mm; continue; }
            const SsContact& g = *it->second;
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
    (void)sk;
    PXG_CHECK_MSG(mm == 0, "npSbSelf: %zu contact mismatches", mm);
    PXG_CHECK_MSG(gCount >= contacts - refSkippedContacts && gCount <= contacts - refSkippedContacts + skipped,
                  "npSbSelf: counter %u outside [%zu, %zu]", gCount, contacts - refSkippedContacts,
                  contacts - refSkippedContacts + skipped);
    PXG_CHECK_MSG(checked > NQ * 2 / 3, "npSbSelf: too many skipped (%zu)", skipped);
    PXG_CHECK_MSG(contacts > 15 && hintBlocked > 50 && neighborBlocked > 20 && filtered > 5 && restBlocked > 5 && stressBlocked > 2,
                  "npSbSelf: coverage thin (ct=%zu hint=%zu nb=%zu filt=%zu rest=%zu stress=%zu)",
                  contacts, hintBlocked, neighborBlocked, filtered, restBlocked, stressBlocked);
    PXG_CHECK_MSG(restInsideBlocked >= 3 && med3Fixtures >= 4,
                  "npSbSelf: rest-gate fixtures thin (insideBlocked=%zu med3=%zu)", restInsideBlocked, med3Fixtures);
}

} // namespace
