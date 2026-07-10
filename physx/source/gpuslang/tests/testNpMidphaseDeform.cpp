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

// Verification of npMidphaseDeform.slang: the three deformable midphase
// traversers (ClothTreeTraverser / SoftbodyTreeTraverser /
// SoftbodyBoxTraverser) on the BV32 walk.  The tri/tet bounding boxes are
// componentwise min/max SELECTIONS of uploaded vertices and
// PxBounds3::intersects is pure comparisons, so those decisions are bit-exact.
// isValidTet (signed volume > 1e-9) is the only arithmetic gate; the tet
// generator REJECTS tets whose signed volume is within 1e-4 of the threshold,
// so its outcome is also deterministic — the whole expected pair set is exact,
// including the global counter.  Output order across queries is
// atomic-nondeterministic, so pair sets are compared sorted.  Degenerate /
// inverted tets (robustly negative volume) and zero-hint tets are seeded to
// exercise both gates; a capacity-limited run checks the drop rule.

#include "pxgTestFramework.h"

#include <algorithm>
#include <array>
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
struct DmWork { F4 boxMin, boxMax; U4 ids; };
struct DmParams { uint32_t numWorkItems, stackSize, pad0, pad1; };

struct V3 { float x, y, z; };
static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static V3 v3(F4 f) { return { f.x, f.y, f.z }; }

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
static bool bIntersect(const AABB& a, const AABB& b)
{
    return !(b.mn[0] > a.mx[0] || a.mn[0] > b.mx[0] || b.mn[1] > a.mx[1] || a.mn[1] > b.mx[1] ||
             b.mn[2] > a.mx[2] || a.mn[2] > b.mx[2]);
}

static uint32_t buildNode(std::vector<Bv32Node>& nodes, const std::vector<AABB>& prims,
                          uint32_t begin, uint32_t end, std::mt19937& rng, int depth)
{
    const uint32_t nodeIdx = (uint32_t)nodes.size();
    nodes.push_back(Bv32Node());
    nodes.back().mDepth = (uint32_t)depth;

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

static void mortonSort(std::vector<U4>& prims, std::vector<AABB>& bounds, std::vector<uint32_t>* hints)
{
    const uint32_t N = (uint32_t)prims.size();
    auto expand = [](uint32_t v) {
        v &= 0x3ff; v = (v | (v << 16)) & 0x30000ff; v = (v | (v << 8)) & 0x300f00f;
        v = (v | (v << 4)) & 0x30c30c3; v = (v | (v << 2)) & 0x9249249; return v;
    };
    std::vector<uint32_t> order(N);
    std::vector<uint64_t> code(N);
    for (uint32_t t = 0; t < N; ++t)
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
    std::vector<U4> p2(N);
    std::vector<AABB> b2(N);
    std::vector<uint32_t> h2(N);
    for (uint32_t t = 0; t < N; ++t)
    {
        p2[t] = prims[order[t]];
        b2[t] = bounds[order[t]];
        if (hints) h2[t] = (*hints)[order[t]];
    }
    prims.swap(p2);
    bounds.swap(b2);
    if (hints) hints->swap(h2);
}

// exact reference traversal (all decisions are pure comparisons / gated
// arithmetic whose inputs were generated away from the threshold)
static void refTraverse(const std::vector<Bv32Node>& nodes, const std::vector<U4>& prims,
                        const std::vector<F4>& verts, const std::vector<uint32_t>& hints,
                        const DmWork& w, int mode,
                        std::vector<U4>& out)
{
    const AABB qb = { { w.boxMin.x, w.boxMin.y, w.boxMin.z }, { w.boxMax.x, w.boxMax.y, w.boxMax.z } };
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
            for (uint32_t i = 0; i < nb; ++i)
            {
                const uint32_t t = start + i;
                bool hit = false;
                const U4& pi = prims[t];
                if (mode == 0)
                {
                    AABB tb;
                    const V3 a = v3(verts[pi.x]), b = v3(verts[pi.y]), c = v3(verts[pi.z]);
                    tb.mn[0] = std::min(std::min(a.x, b.x), c.x); tb.mx[0] = std::max(std::max(a.x, b.x), c.x);
                    tb.mn[1] = std::min(std::min(a.y, b.y), c.y); tb.mx[1] = std::max(std::max(a.y, b.y), c.y);
                    tb.mn[2] = std::min(std::min(a.z, b.z), c.z); tb.mx[2] = std::max(std::max(a.z, b.z), c.z);
                    hit = bIntersect(tb, qb);
                }
                else if (mode != 1 || hints[t] != 0)
                {
                    const V3 a = v3(verts[pi.x]), b = v3(verts[pi.y]), c = v3(verts[pi.z]), d = v3(verts[pi.w]);
                    if (dot3(sub(b, a), cross3(sub(c, a), sub(d, a))) > 1e-9f)
                    {
                        AABB tb;
                        tb.mn[0] = std::min(std::min(a.x, b.x), std::min(c.x, d.x)); tb.mx[0] = std::max(std::max(a.x, b.x), std::max(c.x, d.x));
                        tb.mn[1] = std::min(std::min(a.y, b.y), std::min(c.y, d.y)); tb.mx[1] = std::max(std::max(a.y, b.y), std::max(c.y, d.y));
                        tb.mn[2] = std::min(std::min(a.z, b.z), std::min(c.z, d.z)); tb.mx[2] = std::max(std::max(a.z, b.z), std::max(c.z, d.z));
                        hit = bIntersect(tb, qb);
                    }
                }
                if (hit)
                    out.push_back(mode == 2 ? U4{ w.ids.x, w.ids.y, t, 0 } : U4{ w.ids.x, t, 0, 0 });
            }
        }
        else
        {
            for (uint32_t i = 0; i < node.mNbNodes; ++i)
            {
                const AABB cb = { { node.mMin[i].x, node.mMin[i].y, node.mMin[i].z },
                                  { node.mMax[i].x, node.mMax[i].y, node.mMax[i].z } };
                if (!bIntersect(qb, cb)) continue;
                const uint32_t data = node.mData[i];
                stack.push_back((data & 1) ? ((cur & 0xFFFFFFC0u) | (i << 1) | 1) : ((data >> 11) << 6));
            }
        }
        if (!stack.empty()) { cur = stack.back(); stack.pop_back(); }
        else hasNode = false;
    }
}

struct Scene
{
    std::vector<Bv32Node> nodes;
    std::vector<U4> prims;
    std::vector<F4> verts;
    std::vector<uint32_t> hints;
    std::vector<DmWork> work;
    size_t validityRejected = 0, hintZero = 0;
};

// isTet: tets with the signed-volume near-threshold band rejected at
// generation; ~1/5 tets robustly inverted (volume < -1e-3), ~1/6 hint 0.
static Scene buildScene(bool isTet, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uv(-1, 1);
    std::uniform_real_distribution<float> u01(0, 1);

    Scene s;
    const uint32_t NP = 500, NQ = 96;
    const uint32_t K = isTet ? 4 : 3;
    s.verts.resize(NP * 4);
    s.prims.resize(NP);
    s.hints.resize(NP, 1);
    std::vector<AABB> bounds(NP);
    for (uint32_t t = 0; t < NP; ++t)
    {
        const V3 c0 = { uv(rng), uv(rng), uv(rng) };
        for (;;)
        {
            for (uint32_t k = 0; k < K; ++k)
                s.verts[t * 4 + k] = { c0.x + 0.15f * uv(rng), c0.y + 0.15f * uv(rng), c0.z + 0.15f * uv(rng), 0 };
            if (!isTet) break;
            V3 a = v3(s.verts[t * 4]), b = v3(s.verts[t * 4 + 1]), c = v3(s.verts[t * 4 + 2]), d = v3(s.verts[t * 4 + 3]);
            float vol = dot3(sub(b, a), cross3(sub(c, a), sub(d, a)));
            if (t % 5 == 4)
            {
                // force a robustly INVERTED tet (swap two verts if needed)
                if (vol > 0) { std::swap(s.verts[t * 4 + 2], s.verts[t * 4 + 3]); vol = -vol; }
                if (vol < -1e-3f) { ++s.validityRejected; break; }
            }
            else if (vol > 1e-9f + 1e-4f)
                break; // robustly valid
            // else near the threshold or wrong sign: regenerate
        }
        if (isTet && t % 6 == 5) { s.hints[t] = 0; ++s.hintZero; }
        else if (isTet && t % 6 == 2) s.hints[t] = 2 + t % 254; // nonzero values beyond 1: gate is != 0, not == 1
        s.prims[t] = { t * 4, t * 4 + 1, t * 4 + 2, isTet ? t * 4 + 3 : 0 };
        AABB bb = { { 1e30f, 1e30f, 1e30f }, { -1e30f, -1e30f, -1e30f } };
        for (uint32_t k = 0; k < K; ++k)
        {
            const float* p = &s.verts[t * 4 + k].x;
            for (int j = 0; j < 3; ++j) { bb.mn[j] = std::min(bb.mn[j], p[j]); bb.mx[j] = std::max(bb.mx[j], p[j]); }
        }
        bounds[t] = bb;
    }
    mortonSort(s.prims, bounds, &s.hints);
    buildNode(s.nodes, bounds, 0, NP, rng, 0);

    s.work.resize(NQ);
    for (uint32_t q = 0; q < NQ; ++q)
    {
        DmWork& w = s.work[q];
        if (q % 8 == 7)
        {
            w.boxMin = { 60.0f + q, 60, 60, 0 };
            w.boxMax = { 61.0f + q, 61, 61, 0 };
        }
        else
        {
            const V3 c = { uv(rng), uv(rng), uv(rng) };
            const V3 e = { 0.08f + 0.2f * u01(rng), 0.08f + 0.2f * u01(rng), 0.08f + 0.2f * u01(rng) };
            w.boxMin = { c.x - e.x, c.y - e.y, c.z - e.z, 0 };
            w.boxMax = { c.x + e.x, c.y + e.y, c.z + e.z, 0 };
        }
        w.ids = { q, 7000 + q, 0, 0 };
    }
    return s;
}

struct GpuOut { std::vector<U4> pairs; uint32_t counter; };
static GpuOut runEntry(const Scene& s, const char* entry, uint32_t stackSize, uint32_t bufCap)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t NQ = (uint32_t)s.work.size();
    std::vector<U4> outInit(bufCap, { 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu });
    std::vector<uint32_t> cnt(1, 0);
    PxgBufferHandle bN = createBufferFrom(dev, s.nodes);
    PxgBufferHandle bP = createBufferFrom(dev, s.prims);
    PxgBufferHandle bV = createBufferFrom(dev, s.verts);
    PxgBufferHandle bH = createBufferFrom(dev, s.hints);
    PxgBufferHandle bW = createBufferFrom(dev, s.work);
    PxgBufferHandle bO = createBufferFrom(dev, outInit);
    PxgBufferHandle bC = createBufferFrom(dev, cnt);

    DmParams prm = { NQ, stackSize, 0, 0 };
    Kernel k("npMidphaseDeform.spv", entry,
             { (uint32_t)sizeof(Bv32Node), 16, 16, 4, (uint32_t)sizeof(DmWork), 16, 4 },
             sizeof(DmParams));
    k.dispatch(NQ, { bN, bP, bV, bH, bW, bO, bC }, prm);
    dev->flush();

    GpuOut g;
    g.pairs = downloadBuffer<U4>(dev, bO, bufCap);
    g.counter = downloadBuffer<uint32_t>(dev, bC, 1)[0];
    dev->destroyBuffer(bN); dev->destroyBuffer(bP); dev->destroyBuffer(bV); dev->destroyBuffer(bH);
    dev->destroyBuffer(bW); dev->destroyBuffer(bO); dev->destroyBuffer(bC);
    return g;
}

static std::vector<std::array<uint32_t, 4>> sorted(const std::vector<U4>& v, size_t n)
{
    std::vector<std::array<uint32_t, 4>> r;
    for (size_t i = 0; i < n; ++i) r.push_back({ v[i].x, v[i].y, v[i].z, v[i].w });
    std::sort(r.begin(), r.end());
    return r;
}

static void checkExact(const Scene& s, const char* entry, int mode)
{
    std::vector<U4> expect;
    for (const auto& w : s.work)
        refTraverse(s.nodes, s.prims, s.verts, s.hints, w, mode, expect);

    const uint32_t total = (uint32_t)expect.size();
    PXG_CHECK_MSG(total > 300, "%s: scene too sparse (%u)", entry, total);
    const GpuOut g = runEntry(s, entry, total + 64, total + 64);
    PXG_CHECK_MSG(g.counter == total, "%s: counter %u != expected %u", entry, g.counter, total);

    const auto ge = sorted(g.pairs, total);
    const auto re = sorted(expect, total);
    size_t mm = 0;
    for (uint32_t i = 0; i < total; ++i) if (ge[i] != re[i]) ++mm;
    PXG_CHECK_MSG(mm == 0, "%s: %zu pair mismatches", entry, mm);

    size_t dirty = 0;
    for (uint32_t i = total; i < total + 64; ++i) if (g.pairs[i].x != 0xffffffffu) ++dirty;
    PXG_CHECK_MSG(dirty == 0, "%s: %zu writes past counter", entry, dirty);
}

PXG_TEST(npMidphaseDeform_clothTri)
{
    const Scene s = buildScene(false, 78011);
    checkExact(s, "ctClothTri", 0);

    // capacity-limited run: drop rule
    std::vector<U4> expect;
    for (const auto& w : s.work) refTraverse(s.nodes, s.prims, s.verts, s.hints, w, 0, expect);
    const uint32_t total = (uint32_t)expect.size();
    const uint32_t cap = total / 3;
    const GpuOut g = runEntry(s, "ctClothTri", cap, total + 64);
    PXG_CHECK_MSG(g.counter == total, "ctClothTri overflow: counter %u != %u", g.counter, total);
    std::set<std::array<uint32_t, 4>> expSet;
    for (const auto& p : expect) expSet.insert({ p.x, p.y, p.z, p.w });
    std::set<std::array<uint32_t, 4>> gotSet;
    size_t bad = 0;
    for (uint32_t i = 0; i < cap; ++i)
    {
        const std::array<uint32_t, 4> p = { g.pairs[i].x, g.pairs[i].y, g.pairs[i].z, g.pairs[i].w };
        if (!expSet.count(p)) ++bad;
        gotSet.insert(p);
    }
    PXG_CHECK_MSG(bad == 0 && gotSet.size() == cap, "ctClothTri overflow: bad=%zu distinct=%zu want %u", bad, gotSet.size(), cap);
    // the capacity clamp itself: nothing may be stored at or past the cap
    size_t dirty = 0;
    for (uint32_t i = cap; i < total + 64; ++i)
        if (g.pairs[i].x != 0xffffffffu) ++dirty;
    PXG_CHECK_MSG(dirty == 0, "ctClothTri overflow: %zu stores past the capacity (drop guard broken)", dirty);
}

PXG_TEST(npMidphaseDeform_softbodyTet)
{
    const Scene s = buildScene(true, 78029);
    checkExact(s, "ctSoftbodyTet", 1);
    // both gates genuinely exercised
    PXG_CHECK_MSG(s.validityRejected > 50 && s.hintZero > 50,
                  "ctSoftbodyTet: gate coverage thin (invalid=%zu hint0=%zu)", s.validityRejected, s.hintZero);

    // the hint gate must change the outcome vs the ungated entry on this scene
    std::vector<U4> gated, ungated;
    for (const auto& w : s.work)
    {
        refTraverse(s.nodes, s.prims, s.verts, s.hints, w, 1, gated);
        refTraverse(s.nodes, s.prims, s.verts, s.hints, w, 2, ungated);
    }
    PXG_CHECK_MSG(gated.size() < ungated.size(), "ctSoftbodyTet: hint gate never engaged (%zu vs %zu)", gated.size(), ungated.size());

    // the validity gate must also be outcome-changing: with every tet treated
    // as valid, the ref must produce MORE pairs (i.e. some visited, box-hitting
    // tet is actually being rejected by isValidTet on this scene)
    {
        Scene s2 = s; // volume gate bypassed by flipping inverted tets' vert order
        for (uint32_t t = 0; t < (uint32_t)s2.prims.size(); ++t)
        {
            const U4& pi = s2.prims[t];
            const V3 a = v3(s2.verts[pi.x]), b = v3(s2.verts[pi.y]), c = v3(s2.verts[pi.z]), d = v3(s2.verts[pi.w]);
            if (dot3(sub(b, a), cross3(sub(c, a), sub(d, a))) <= 1e-9f)
                std::swap(s2.prims[t].z, s2.prims[t].w); // make it valid, same bounds
        }
        std::vector<U4> fixedValid;
        for (const auto& w : s2.work)
            refTraverse(s2.nodes, s2.prims, s2.verts, s2.hints, w, 2, fixedValid);
        PXG_CHECK_MSG(ungated.size() < fixedValid.size(),
                      "ctSoftbodyTet: validity gate never engaged (%zu vs %zu)", ungated.size(), fixedValid.size());
    }
}

PXG_TEST(npMidphaseDeform_softbodyBoxTet)
{
    const Scene s = buildScene(true, 78041);
    checkExact(s, "ctSoftbodyBoxTet", 2);
}

} // namespace
