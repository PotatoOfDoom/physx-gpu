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

// Verification of npClothCloth.slang (TriangleLeafBoundMinMaxTraverser +
// computeTriangleBound2 on the BV32 walk).  The traversal and filter
// decisions are exact (pure comparisons / integer binary search); the
// per-triangle float chain (plane, back-face cull, satIntersect with the
// EVOLVING tolerance min(best, contactDistance), the running-min update) is
// slack-tracked by the C++ reference per LANE: any evaluated comparison
// within MARG of flipping marks that lane skipped, all other lanes must
// match exactly in prim index and to tolerance in best distance.  Both the
// mesh (back-face cull) and cloth-cloth (double filter lookup) paths are
// exercised, including full-mask and exact filter hits.

#include "pxgTestFramework.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
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
struct PairNR { uint32_t i0lo, i0hi, index1, index2; };
struct CcWork { U4 ids; F4 centroid_cd; };
struct CcParams { uint32_t numWorkItems, numFilterPairs, pad0, pad1; };

struct V3 { float x, y, z; };
static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static V3 muls(V3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
static V3 v3(F4 f) { return { f.x, f.y, f.z }; }
static F4 f4(V3 v) { return { v.x, v.y, v.z, 0.0f }; }

static const float MARG = 1e-4f;
static const float MARG_MAGSQ = 1e-6f;
static const float MAXF = 3.4028235e38f;

struct Slack
{
    bool near = false;
    void note(float lhs, float rhs) { if (std::fabs(lhs - rhs) < MARG) near = true; }
    void noteMag(float m) { if (std::fabs(m - 1e-5f) < MARG_MAGSQ) near = true; }
};

static void makePlane(V3 p0, V3 p1, V3 p2, V3& n, float& d)
{
    const V3 c = cross3(sub(p1, p0), sub(p2, p0));
    const float m = dot3(c, c);
    n = m > 0 ? muls(c, 1.0f / std::sqrt(m)) : V3{ 0, 0, 0 };
    d = -dot3(p0, n);
}

static float minProject(V3 n, float d, V3 v0, V3 v1, V3 v2)
{
    return std::min(dot3(n, v0), std::min(dot3(n, v1), dot3(n, v2))) + d;
}

static float testAxis(V3 axis, V3 a0, V3 a1, V3 a2, V3 b0, V3 b1, V3 b2)
{
    float min0 = dot3(a0, axis), max0 = min0;
    float min1 = dot3(b0, axis), max1 = min1;
    const float pa1 = dot3(a1, axis), pb1 = dot3(b1, axis);
    min0 = std::min(pa1, min0); max0 = std::max(pa1, max0);
    min1 = std::min(pb1, min1); max1 = std::max(pb1, max1);
    const float pa2 = dot3(a2, axis), pb2 = dot3(b2, axis);
    min0 = std::min(pa2, min0); max0 = std::max(pa2, max0);
    min1 = std::min(pb2, min1); max1 = std::max(pb2, max1);
    return std::max(min1 - max0, min0 - max1);
}

static float refSatIntersect(V3 a0, V3 a1, V3 a2, V3 nA, float dA,
                             V3 b0, V3 b1, V3 b2, V3 nB, float dB,
                             float tolerance, Slack& sl)
{
    float sep = minProject(nA, dA, b0, b1, b2);
    sl.note(sep, tolerance);
    if (sep > tolerance) return sep;
    sep = std::max(sep, minProject(nB, dB, a0, a1, a2));
    sl.note(sep, tolerance);
    if (sep > tolerance) return sep;
    const V3 axes0[3] = { sub(a1, a0), sub(a2, a0), sub(a2, a1) };
    const V3 axes1[3] = { sub(b1, b0), sub(b2, b0), sub(b2, b1) };
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            V3 ax = cross3(axes0[i], axes1[j]);
            const float magSq = dot3(ax, ax);
            sl.noteMag(magSq);
            if (magSq > 1e-5f)
            {
                ax = muls(ax, 1.0f / std::sqrt(magSq));
                sep = std::max(sep, testAxis(ax, a0, a1, a2, b0, b1, b2));
                sl.note(sep, tolerance);
                if (sep > tolerance) return sep;
            }
        }
    return sep;
}

// ---- filter reference (integer, exact) --------------------------------
static uint32_t encodeCloth(uint32_t id, uint32_t elem) { return (id << 20) | elem; }
static bool refFind(const std::vector<PairNR>& pairs, uint32_t id0, uint32_t id1)
{
    const uint64_t q0 = id0; // zero-extended, as the 2-arg CUDA wrapper does
    uint32_t l = 0, r = (uint32_t)pairs.size();
    while (l < r)
    {
        const uint32_t mid = (l + r) / 2;
        const PairNR& p = pairs[mid];
        const uint64_t p0 = ((uint64_t)p.i0hi << 32) | p.i0lo;
        int cmp = 0;
        if (q0 < p0) cmp = -1;
        else if (q0 > p0) cmp = 1;
        else if (id1 < p.index1) cmp = -1;
        else if (id1 > p.index1) cmp = 1;
        else if (0u < p.index2) cmp = -1;
        if (cmp == 0 || (q0 == p0 && id1 == p.index1))
            return true;
        if (cmp < 0) r = mid; else l = mid + 1;
    }
    return false;
}

// ---- BVH builder (as prior tranches) ----------------------------------
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

struct Scene
{
    std::vector<Bv32Node> nodes;
    std::vector<U4> tris1;
    std::vector<F4> verts1;
    std::vector<U4> tris0;
    std::vector<F4> verts0;
    std::vector<PairNR> filters;
    std::vector<CcWork> work;
};

static Scene buildScene()
{
    std::mt19937 rng(80017);
    std::uniform_real_distribution<float> uv(-1, 1);
    std::uniform_real_distribution<float> u01(0, 1);

    Scene s;
    const uint32_t NT = 500, NQ = 96;
    s.verts1.resize(NT * 3);
    s.tris1.resize(NT);
    std::vector<AABB> bounds(NT);
    for (uint32_t t = 0; t < NT; ++t)
    {
        const V3 c0 = { uv(rng), uv(rng), uv(rng) };
        for (uint32_t k = 0; k < 3; ++k)
            s.verts1[t * 3 + k] = { c0.x + 0.15f * uv(rng), c0.y + 0.15f * uv(rng), c0.z + 0.15f * uv(rng), 0 };
        s.tris1[t] = { t * 3, t * 3 + 1, t * 3 + 2, 0 };
        AABB bb = { { 1e30f, 1e30f, 1e30f }, { -1e30f, -1e30f, -1e30f } };
        for (uint32_t k = 0; k < 3; ++k)
        {
            const float* p = &s.verts1[t * 3 + k].x;
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
        std::vector<U4> t2(NT);
        std::vector<AABB> b2(NT);
        for (uint32_t t = 0; t < NT; ++t) { t2[t] = s.tris1[order[t]]; b2[t] = bounds[order[t]]; }
        s.tris1.swap(t2);
        bounds.swap(b2);
    }
    buildNode(s.nodes, bounds, 0, NT, rng, 0);

    // query triangles (mesh 0)
    s.verts0.resize(NQ * 3);
    s.tris0.resize(NQ);
    s.work.resize(NQ);
    for (uint32_t q = 0; q < NQ; ++q)
    {
        const V3 c0 = { uv(rng), uv(rng), uv(rng) };
        for (uint32_t k = 0; k < 3; ++k)
            s.verts0[q * 3 + k] = { c0.x + 0.2f * uv(rng), c0.y + 0.2f * uv(rng), c0.z + 0.2f * uv(rng), 0 };
        s.tris0[q] = { q * 3, q * 3 + 1, q * 3 + 2, 0 };

        CcWork& w = s.work[q];
        const bool isMesh = (q & 1) != 0;
        w.ids = { q, isMesh ? 1u : 0u, 3 + q % 7, 20 + q % 5 }; // cloth ids < 0xfff
        const V3 cen = { c0.x + 0.5f * uv(rng), c0.y + 0.5f * uv(rng), c0.z + 0.5f * uv(rng) };
        w.centroid_cd = { cen.x, cen.y, cen.z, 0.05f + 0.25f * u01(rng) };
    }

    // filters (cloth-cloth path): exact pairs + full-mask entries over the
    // actual id combinations, sorted by the 3-key compare (index0 u64 asc,
    // index1, index2; index2 always 0 here)
    std::set<std::pair<uint32_t, uint32_t>> fset;
    for (uint32_t q = 0; q < NQ; q += 2) // cloth-cloth work items are the even ones
    {
        const uint32_t id0 = 3 + q % 7, id1 = 20 + q % 5;
        if (q % 8 == 0)
        {
            // full-mask filter: suppress a handful of specific mesh1 triangles
            const uint32_t full0 = encodeCloth(id0, 0xfffff);
            for (uint32_t t = 0; t < 40; ++t)
            {
                const uint32_t m1 = encodeCloth(id1, (q * 13 + t * 7) % 500);
                fset.insert({ std::min(full0, m1), std::max(full0, m1) });
            }
        }
        if (q % 8 == 2)
        {
            // exact filters: this query triangle against specific mesh1 triangles
            const uint32_t m0 = encodeCloth(id0, q);
            for (uint32_t t = 0; t < 40; ++t)
            {
                const uint32_t m1 = encodeCloth(id1, (q * 11 + t * 5) % 500);
                fset.insert({ std::min(m0, m1), std::max(m0, m1) });
            }
        }
    }
    for (const auto& f : fset)
        s.filters.push_back({ f.first, 0, f.second, 0 });
    // already sorted: set orders by (first, second) == (index0, index1) with index2 == 0
    return s;
}

// exact reference walk: which prims lane L tests, in round order
static void refLanePrims(const std::vector<Bv32Node>& nodes, V3 tMin, V3 tMax,
                         std::vector<std::vector<uint32_t>>& lanePrims)
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
            const uint32_t nb = (data >> 1) & 63, start = data >> 7;
            for (uint32_t i = 0; i < nb && i < 32; ++i)
                lanePrims[i].push_back(start + i);
        }
        else
        {
            for (uint32_t i = 0; i < node.mNbNodes; ++i)
            {
                const bool hit = !(node.mMin[i].x > tMax.x || tMin.x > node.mMax[i].x ||
                                   node.mMin[i].y > tMax.y || tMin.y > node.mMax[i].y ||
                                   node.mMin[i].z > tMax.z || tMin.z > node.mMax[i].z);
                if (!hit) continue;
                const uint32_t data = node.mData[i];
                stack.push_back((data & 1) ? ((cur & 0xFFFFFFC0u) | (i << 1) | 1) : ((data >> 11) << 6));
            }
        }
        if (!stack.empty()) { cur = stack.back(); stack.pop_back(); }
        else hasNode = false;
    }
}

PXG_TEST(npClothCloth_traverser)
{
    const Scene s = buildScene();
    const uint32_t NQ = (uint32_t)s.work.size();

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<float> bestInit(NQ * 32, -1.0f);
    std::vector<uint32_t> primInit(NQ * 32, 0xdeadbeefu);
    PxgBufferHandle bN = createBufferFrom(dev, s.nodes);
    PxgBufferHandle bT1 = createBufferFrom(dev, s.tris1);
    PxgBufferHandle bV1 = createBufferFrom(dev, s.verts1);
    PxgBufferHandle bT0 = createBufferFrom(dev, s.tris0);
    PxgBufferHandle bV0 = createBufferFrom(dev, s.verts0);
    PxgBufferHandle bF = createBufferFrom(dev, s.filters);
    PxgBufferHandle bW = createBufferFrom(dev, s.work);
    PxgBufferHandle bB = createBufferFrom(dev, bestInit);
    PxgBufferHandle bP = createBufferFrom(dev, primInit);

    CcParams prm = { NQ, (uint32_t)s.filters.size(), 0, 0 };
    Kernel k("npClothCloth.spv", "ctClothClothQuery",
             { (uint32_t)sizeof(Bv32Node), 16, 16, 16, 16, 16, (uint32_t)sizeof(CcWork), 4, 4 },
             sizeof(CcParams));
    k.dispatch(NQ, { bN, bT1, bV1, bT0, bV0, bF, bW, bB, bP }, prm);
    dev->flush();

    const auto gBest = downloadBuffer<float>(dev, bB, NQ * 32);
    const auto gPrim = downloadBuffer<uint32_t>(dev, bP, NQ * 32);
    dev->destroyBuffer(bN); dev->destroyBuffer(bT1); dev->destroyBuffer(bV1); dev->destroyBuffer(bT0);
    dev->destroyBuffer(bV0); dev->destroyBuffer(bF); dev->destroyBuffer(bW); dev->destroyBuffer(bB); dev->destroyBuffer(bP);

    size_t mmPrim = 0, mmBest = 0, skipped = 0, checked = 0, checkedNonEmpty = 0, nonEmpty = 0;
    size_t updates = 0, culled = 0, filteredFull = 0, filteredExact = 0, multiUpdate = 0;
    for (uint32_t q = 0; q < NQ; ++q)
    {
        const CcWork& w = s.work[q];
        const bool isMesh = w.ids.y != 0;
        const uint32_t id0 = w.ids.z, id1 = w.ids.w;
        const float cd = w.centroid_cd.w;
        const V3 cen = v3(w.centroid_cd);

        const U4& ti0 = s.tris0[w.ids.x];
        const V3 a0 = v3(s.verts0[ti0.x]), a1 = v3(s.verts0[ti0.y]), a2 = v3(s.verts0[ti0.z]);
        V3 nA; float dA;
        makePlane(a0, a1, a2, nA, dA);
        // computeTriangleBound2 + the callers' fattenFast(contactDist); the same
        // single subtract/add on both sides keeps the box tests bit-exact
        const V3 tMin = { std::min(std::min(a0.x, a1.x), a2.x) - cd, std::min(std::min(a0.y, a1.y), a2.y) - cd, std::min(std::min(a0.z, a1.z), a2.z) - cd };
        const V3 tMax = { std::max(std::max(a0.x, a1.x), a2.x) + cd, std::max(std::max(a0.y, a1.y), a2.y) + cd, std::max(std::max(a0.z, a1.z), a2.z) + cd };

        std::vector<std::vector<uint32_t>> lanePrims(32);
        refLanePrims(s.nodes, tMin, tMax, lanePrims);

        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            Slack sl;
            float best = MAXF;
            uint32_t prim = 0xffffffff;
            uint32_t laneUpdates = 0;
            for (uint32_t t : lanePrims[lane])
            {
                const U4& ti1 = s.tris1[t];
                const V3 b0 = v3(s.verts1[ti1.x]), b1 = v3(s.verts1[ti1.y]), b2 = v3(s.verts1[ti1.z]);
                V3 nB; float dB;
                makePlane(b0, b1, b2, nB, dB);

                bool process;
                if (isMesh)
                {
                    const float dist = dot3(nB, cen) + dB;
                    sl.note(dist, 0.0f);
                    process = dist >= 0.0f;
                    if (!process) ++culled;
                }
                else
                {
                    const uint32_t m0 = encodeCloth(id0, w.ids.x);
                    const uint32_t m1 = encodeCloth(id1, t);
                    const uint32_t f0 = encodeCloth(id0, 0xfffff);
                    const bool hitFull = refFind(s.filters, std::min(f0, m1), std::max(f0, m1));
                    const bool hitExact = !hitFull && refFind(s.filters, std::min(m0, m1), std::max(m0, m1));
                    process = !hitFull && !hitExact;
                    if (hitFull) ++filteredFull;
                    if (hitExact) ++filteredExact;
                }
                if (process)
                {
                    const float separation = refSatIntersect(a0, a1, a2, nA, dA, b0, b1, b2, nB, dB,
                                                             std::min(best, cd), sl);
                    sl.note(separation, best);
                    if (separation < best)
                    {
                        best = separation;
                        prim = t;
                        ++updates;
                        ++laneUpdates;
                    }
                }
            }
            if (laneUpdates >= 2) ++multiUpdate;

            const float gB = gBest[q * 32 + lane];
            const uint32_t gP = gPrim[q * 32 + lane];
            if (!lanePrims[lane].empty()) ++nonEmpty;
            if (sl.near) { ++skipped; continue; }
            ++checked;
            if (!lanePrims[lane].empty()) ++checkedNonEmpty;
            if (gP != prim) ++mmPrim;
            if (prim == 0xffffffff) { if (gB != MAXF) ++mmBest; }
            else if (!almostEqual(gB, best, 1e-4f, 1e-4f)) ++mmBest;
        }
    }

    PXG_CHECK_MSG(mmPrim == 0, "npClothCloth: %zu prim mismatches", mmPrim);
    PXG_CHECK_MSG(mmBest == 0, "npClothCloth: %zu best-distance mismatches", mmBest);
    PXG_CHECK_MSG(checked > NQ * 32 * 3 / 4 && checkedNonEmpty > nonEmpty * 3 / 4,
                  "npClothCloth: too many skipped lanes (%zu skipped, %zu checked, %zu/%zu non-empty)",
                  skipped, checked, checkedNonEmpty, nonEmpty);
    PXG_CHECK_MSG(updates > 300 && culled > 100 && filteredFull > 20 && filteredExact > 10 && multiUpdate > 20,
                  "npClothCloth: coverage thin (updates=%zu culled=%zu filtFull=%zu filtExact=%zu multiUpdate=%zu)",
                  updates, culled, filteredFull, filteredExact, multiUpdate);
}

} // namespace
