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

// Verification of npBv32.slang (bv32TreeTraversal + pushOntoStackFullWarp with
// an AABB-overlap query callback).  A synthetic BV32 tree is built on the CPU
// (packed exactly like Gu::BV32DataPacked: leaf mData = (start<<7)|(nb<<1)|1
// held in the parent's slot, internal mData = (childNodeIndex<<11), packed
// node index = nodeIdx<<6) and traversed by a C++ transcription of the CUDA
// walk.  The box-overlap decisions are pure comparisons on uploaded values,
// so the visited set is bit-exact deterministic; the output order across
// queries is atomic-nondeterministic, so the (queryId, primitiveIndex) pair
// sets are compared sorted.  A second run with a small output capacity
// verifies the drop-on-overflow path: the counter still counts every push and
// exactly stackSize distinct expected pairs land in the buffer.

#include "pxgTestFramework.h"

#include <algorithm>
#include <cstdint>
#include <random>
#include <set>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct Bv32Node
{
    F4 mMin[32];
    F4 mMax[32];
    uint32_t mData[32];
    uint32_t mNbNodes, mDepth, pad0, pad1;
};
struct QueryBox { F4 mn, mx; };
struct U4 { uint32_t x, y, z, w; };
struct Bv32Params { uint32_t numQueries, stackSize, pad0, pad1; };

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
static bool overlap(const QueryBox& q, const AABB& b)
{
    return q.mn.x <= b.mx[0] && q.mn.y <= b.mx[1] && q.mn.z <= b.mx[2] &&
           b.mn[0] <= q.mx.x && b.mn[1] <= q.mx.y && b.mn[2] <= q.mx.z;
}

// Recursive builder: packs [begin,end) into a new BV32DataPacked node (returns
// its index).  Children are contiguous chunks; small chunks become leaf slots
// (bit0 = 1), larger ones become recursively built internal children.
static uint32_t buildNode(std::vector<Bv32Node>& nodes, const std::vector<AABB>& prims,
                          uint32_t begin, uint32_t end, std::mt19937& rng, int depth)
{
    const uint32_t nodeIdx = (uint32_t)nodes.size();
    nodes.push_back(Bv32Node());
    Bv32Node& stub = nodes.back();
    stub.mDepth = (uint32_t)depth;

    const uint32_t count = end - begin;
    const uint32_t nbChunks = std::min<uint32_t>(2 + rng() % 5, count); // 2..6 children
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
        // capture by index — recursion may reallocate `nodes`.
        // A leaf never exceeds 32 primitives (the warp width the CUDA
        // traversal hands to intersectPrimitiveFullWarp).
        if (e - b <= 12 || (depth >= 4 && e - b <= 32))
        {
            nodes[nodeIdx].mData[slot] = (b << 7) | ((e - b) << 1) | 1; // leaf
            if (rng() % 6 == 0 && slot + 2 < 32)
            {
                // occasionally add an EMPTY leaf slot with the same bounds: a
                // visited zero-primitive leaf exercises the validCount == 0
                // sentinel branch of pushOntoStackFullWarp (no atomicAdd, no store)
                nodes[nodeIdx].mMin[slot] = { bounds.mn[0], bounds.mn[1], bounds.mn[2], 0.0f };
                nodes[nodeIdx].mMax[slot] = { bounds.mx[0], bounds.mx[1], bounds.mx[2], 0.0f };
                ++slot;
                nodes[nodeIdx].mData[slot] = (b << 7) | (0 << 1) | 1;
            }
        }
        else
        {
            const uint32_t child = buildNode(nodes, prims, b, e, rng, depth + 1);
            nodes[nodeIdx].mData[slot] = child << 11; // internal (bit0 = 0)
        }
        nodes[nodeIdx].mMin[slot] = { bounds.mn[0], bounds.mn[1], bounds.mn[2], 0.0f };
        nodes[nodeIdx].mMax[slot] = { bounds.mx[0], bounds.mx[1], bounds.mx[2], 0.0f };
    }
    nodes[nodeIdx].mNbNodes = slot;
    return nodeIdx;
}

// C++ transcription of bv32TreeTraversal with the AABB-overlap callback.
static void refTraverse(const std::vector<Bv32Node>& nodes, const QueryBox& q, uint32_t queryId,
                        std::vector<std::pair<uint32_t, uint32_t>>& out, size_t& maxStack)
{
    std::vector<uint32_t> stack;
    uint32_t cur = 0;
    bool hasNode = true;
    while (hasNode)
    {
        const Bv32Node& node = nodes[cur >> 6];
        if (cur & 1)
        {
            const uint32_t leafOffset = (cur >> 1) & 31;
            const uint32_t data = node.mData[leafOffset];
            const uint32_t nb = (data >> 1) & 63;
            const uint32_t start = data >> 7;
            for (uint32_t i = 0; i < nb; ++i)
                out.push_back({ queryId, start + i });
        }
        else
        {
            for (uint32_t i = 0; i < node.mNbNodes; ++i)
            {
                AABB b;
                b.mn[0] = node.mMin[i].x; b.mn[1] = node.mMin[i].y; b.mn[2] = node.mMin[i].z;
                b.mx[0] = node.mMax[i].x; b.mx[1] = node.mMax[i].y; b.mx[2] = node.mMax[i].z;
                if (!overlap(q, b))
                    continue;
                const uint32_t data = node.mData[i];
                if (data & 1)
                    stack.push_back((cur & 0xFFFFFFC0u) | (i << 1) | 1);
                else
                    stack.push_back((data >> 11) << 6);
            }
            maxStack = std::max(maxStack, stack.size());
        }
        if (!stack.empty()) { cur = stack.back(); stack.pop_back(); }
        else hasNode = false;
    }
}

struct Scene
{
    std::vector<Bv32Node> nodes;
    std::vector<QueryBox> queries;
    std::vector<std::pair<uint32_t, uint32_t>> expected; // sorted (query, prim) pairs
};

static Scene buildScene()
{
    std::mt19937 rng(76013);
    std::uniform_real_distribution<float> uc(-10, 10);
    std::uniform_real_distribution<float> ue(0.1f, 1.2f);

    const uint32_t NP = 600;
    std::vector<AABB> prims(NP);
    for (auto& p : prims)
    {
        const float cx = uc(rng), cy = uc(rng), cz = uc(rng);
        const float ex = ue(rng), ey = ue(rng), ez = ue(rng);
        p = { { cx - ex, cy - ey, cz - ez }, { cx + ex, cy + ey, cz + ez } };
    }

    Scene s;
    buildNode(s.nodes, prims, 0, NP, rng, 0);

    // Enforce the invariant the C++ reference relies on: the CUDA warp emits
    // at most 32 primitives per leaf, so no leaf may reference more.
    size_t emptyLeaves = 0;
    for (const auto& n : s.nodes)
        for (uint32_t i = 0; i < n.mNbNodes; ++i)
            if (n.mData[i] & 1)
            {
                const uint32_t nb = (n.mData[i] >> 1) & 63;
                PXG_CHECK_MSG(nb <= 32, "scene: leaf with %u > 32 primitives", nb);
                if (nb == 0) ++emptyLeaves;
            }
    PXG_CHECK_MSG(emptyLeaves > 3, "scene: too few empty leaves (%zu) to exercise the validCount==0 path", emptyLeaves);

    const uint32_t NQ = 128;
    s.queries.resize(NQ);
    std::uniform_real_distribution<float> uq(0.3f, 4.0f);
    for (uint32_t i = 0; i < NQ; ++i)
    {
        if (i % 8 == 7) // guaranteed miss: far outside the [-12,12] world
        {
            s.queries[i] = { { 100.0f + i, 100, 100, 0 }, { 101.0f + i, 101, 101, 0 } };
            continue;
        }
        const float cx = uc(rng), cy = uc(rng), cz = uc(rng);
        const float ex = uq(rng), ey = uq(rng), ez = uq(rng);
        s.queries[i] = { { cx - ex, cy - ey, cz - ez, 0 }, { cx + ex, cy + ey, cz + ez, 0 } };
    }

    size_t maxStack = 0;
    for (uint32_t i = 0; i < NQ; ++i)
        refTraverse(s.nodes, s.queries[i], i, s.expected, maxStack);
    std::sort(s.expected.begin(), s.expected.end());
    PXG_CHECK_MSG(maxStack <= 1024, "scene: ref stack watermark %zu exceeds shader stack", maxStack);
    return s;
}

static std::vector<U4> runQueries(const Scene& s, uint32_t stackCapacity, uint32_t& counterOut)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t NQ = (uint32_t)s.queries.size();
    std::vector<U4> outInit(stackCapacity, { 0xffffffffu, 0xffffffffu, 0, 0 });
    std::vector<uint32_t> cnt(1, 0);
    PxgBufferHandle bN = createBufferFrom(dev, s.nodes);
    PxgBufferHandle bQ = createBufferFrom(dev, s.queries);
    PxgBufferHandle bO = createBufferFrom(dev, outInit);
    PxgBufferHandle bC = createBufferFrom(dev, cnt);

    Bv32Params prm = { NQ, stackCapacity, 0, 0 };
    Kernel k("npBv32.spv", "ctBv32Query",
             { (uint32_t)sizeof(Bv32Node), (uint32_t)sizeof(QueryBox), (uint32_t)sizeof(U4), 4 },
             sizeof(Bv32Params));
    k.dispatch(NQ, { bN, bQ, bO, bC }, prm);
    dev->flush();

    const auto res = downloadBuffer<U4>(dev, bO, stackCapacity);
    counterOut = downloadBuffer<uint32_t>(dev, bC, 1)[0];
    dev->destroyBuffer(bN); dev->destroyBuffer(bQ); dev->destroyBuffer(bO); dev->destroyBuffer(bC);
    return res;
}

PXG_TEST(npBv32_traversal)
{
    const Scene s = buildScene();
    const uint32_t total = (uint32_t)s.expected.size();
    PXG_CHECK_MSG(total > 500, "npBv32: scene too sparse (%u hits)", total);

    uint32_t counter = 0;
    const auto g = runQueries(s, total + 64, counter);
    PXG_CHECK_MSG(counter == total, "npBv32: counter %u != expected %u", counter, total);

    std::vector<std::pair<uint32_t, uint32_t>> got;
    for (uint32_t i = 0; i < total; ++i)
        got.push_back({ g[i].x, g[i].y });
    std::sort(got.begin(), got.end());
    size_t mm = 0;
    for (uint32_t i = 0; i < total; ++i)
        if (got[i] != s.expected[i]) ++mm;
    PXG_CHECK_MSG(mm == 0, "npBv32: %zu pair mismatches", mm);

    // slots past the counter stay untouched
    size_t dirty = 0;
    for (uint32_t i = total; i < total + 64; ++i)
        if (g[i].x != 0xffffffffu || g[i].y != 0xffffffffu) ++dirty;
    PXG_CHECK_MSG(dirty == 0, "npBv32: %zu writes past the counter", dirty);

    // coverage: some queries hit nothing, some hit many
    std::set<uint32_t> hitQueries;
    for (const auto& p : s.expected) hitQueries.insert(p.first);
    PXG_CHECK_MSG(hitQueries.size() > 40 && hitQueries.size() < s.queries.size(),
                  "npBv32: query coverage thin (%zu of %zu hit)", hitQueries.size(), s.queries.size());
}

PXG_TEST(npBv32_overflow)
{
    const Scene s = buildScene();
    const uint32_t total = (uint32_t)s.expected.size();
    const uint32_t cap = total / 3;

    uint32_t counter = 0;
    const auto g = runQueries(s, cap, counter);
    // the counter counts every attempted push, dropped or not
    PXG_CHECK_MSG(counter == total, "npBv32 overflow: counter %u != expected %u", counter, total);

    // exactly `cap` pairs stored (indices 0..counter-1 partition the pushes),
    // each a distinct member of the expected set
    std::set<std::pair<uint32_t, uint32_t>> expectedSet(s.expected.begin(), s.expected.end());
    std::set<std::pair<uint32_t, uint32_t>> gotSet;
    size_t bad = 0;
    for (uint32_t i = 0; i < cap; ++i)
    {
        const std::pair<uint32_t, uint32_t> p = { g[i].x, g[i].y };
        if (!expectedSet.count(p)) ++bad;
        gotSet.insert(p);
    }
    PXG_CHECK_MSG(bad == 0, "npBv32 overflow: %zu stored pairs not in expected set", bad);
    PXG_CHECK_MSG(gotSet.size() == cap, "npBv32 overflow: %zu distinct pairs stored, want %u", gotSet.size(), cap);
}

} // namespace
