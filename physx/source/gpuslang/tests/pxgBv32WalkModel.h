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

// pxgBv32WalkModel.h
//
// The ORDER oracle for the batched deformable-vs-rigid midphase walk
// (shaders/pxgSlangBv32Batched.slang: cloth_midphaseGeneratePairsLaunch,
// sb_midphaseGeneratePairsLaunch).  The batched walk claims to emit exactly the
// slot sequence of the CUDA's one-warp bv32TreeTraversal, so the oracle here IS
// that one-warp walk, transcribed as plainly as possible: pop a node; an internal
// node pushes its valid children in ascending slot (they pop descending); a leaf
// emits its overlapping primitives of lanes 0..31 in lane order.  It shares no
// code with the shader, which walks the tree in a different shape altogether.
//
// Trees are authored as node lists that may SHARE nodes (a DAG): the traversal
// never notices, and it lets a small blob drive the walk's stack far past what a
// tree of the same size could (the W = 1 fallback of the batched walk).
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <vector>

namespace pxgbv32walk
{
static const uint32_t NODE_WORDS = 292u;   // sizeof(Gu::BV32DataPacked) / 4
static const uint32_t HDR_WORDS = 4u;      // the uint4 the kernels skip

struct Box { float lo[3], hi[3]; };

struct Child
{
    Box box;
    bool leaf;
    uint32_t primStart;   // leaf
    uint32_t nbField;     // leaf: the raw 6-bit count written to mData (may exceed 32)
    uint32_t node;        // internal: index of the child node in the list
};

inline uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

inline bool intersects(const Box& a, const Box& b)   // PxBounds3::intersects
{
    return !(b.lo[0] > a.hi[0] || a.lo[0] > b.hi[0] ||
             b.lo[1] > a.hi[1] || a.lo[1] > b.hi[1] ||
             b.lo[2] > a.hi[2] || a.lo[2] > b.hi[2]);
}

// The blob: header, the nodes, a 4-word trailer.  Unused child slots are poisoned.
inline std::vector<uint32_t> buildBlob(const std::vector<std::vector<Child>>& nodes, uint32_t nbPrims)
{
    const uint32_t n = uint32_t(nodes.size());
    std::vector<uint32_t> w(HDR_WORDS + n * NODE_WORDS + 4u, 0u);
    w[0] = 0u; w[1] = nbPrims; w[2] = 1u; w[3] = n;
    const uint32_t P = f2u(-7.5e37f);
    for (uint32_t k = 0; k < n; ++k)
    {
        const uint32_t base = HDR_WORDS + k * NODE_WORDS;
        for (uint32_t s = 0; s < 256u; ++s) w[base + s] = P;
        for (uint32_t j = 0; j < uint32_t(nodes[k].size()); ++j)
        {
            const Child& c = nodes[k][j];
            for (int a = 0; a < 3; ++a)
            {
                w[base + j * 4u + uint32_t(a)] = f2u(c.box.lo[a]);
                w[base + 128u + j * 4u + uint32_t(a)] = f2u(c.box.hi[a]);
            }
            w[base + j * 4u + 3u] = 0u;
            w[base + 128u + j * 4u + 3u] = 0u;
            w[base + 256u + j] = c.leaf ? ((c.primStart << 7) | ((c.nbField & 63u) << 1) | 1u)
                                        : (c.node << 11);
        }
        w[base + 288u] = uint32_t(nodes[k].size());
        w[base + 289u] = 0u;
    }
    return w;
}

// The CUDA one-warp walk.  Returns the emitted primitives in slot order and the
// deepest stack it needed (the CUDA keeps 192 entries).
inline std::vector<uint32_t> walkOneWarp(const std::vector<std::vector<Child>>& nodes, const Box& rigid,
                                         const std::function<bool(uint32_t)>& overlaps,
                                         uint32_t* maxStackOut = nullptr)
{
    std::vector<uint32_t> out;
    std::vector<uint32_t> stack;
    uint32_t maxStack = 0u;
    uint32_t cur = 0u;   // root
    for (;;)
    {
        const uint32_t node = cur >> 6;
        if (cur & 1u)
        {
            const Child& c = nodes[node][(cur >> 1) & 31u];
            const uint32_t nb = c.nbField & 63u;
            for (uint32_t lane = 0; lane < 32u; ++lane)
                if (lane < nb && overlaps(c.primStart + lane))
                    out.push_back(c.primStart + lane);
        }
        else
        {
            const std::vector<Child>& kids = nodes[node];
            for (uint32_t lane = 0; lane < uint32_t(kids.size()); ++lane)
            {
                if (!intersects(rigid, kids[lane].box)) continue;
                stack.push_back(kids[lane].leaf ? ((cur & ~63u) | (lane << 1) | 1u) : (kids[lane].node << 6));
            }
            maxStack = std::max(maxStack, uint32_t(stack.size()));
        }
        if (stack.empty()) break;
        cur = stack.back();
        stack.pop_back();
    }
    if (maxStackOut) *maxStackOut = maxStack;
    return out;
}

struct Rng
{
    uint32_t s;
    uint32_t next() { s = s * 1664525u + 1013904223u; return s >> 8; }
    uint32_t below(uint32_t n) { return next() % n; }
    float unit() { return float(next() % 1000001u) / 1000000.0f; }
};

inline Box hullOf(const Box& a, const Box& b)
{
    Box r;
    for (int i = 0; i < 3; ++i) { r.lo[i] = std::min(a.lo[i], b.lo[i]); r.hi[i] = std::max(a.hi[i], b.hi[i]); }
    return r;
}

// A random tree over primitives [0, nbPrims) whose bounds are `primBox(i)`.
// Shapes: `levels` internal levels below the root with fan-out in [minFan, 32],
// leaf children carry 1..maxLeafPrims primitives in order.  Child boxes are the
// honest hull of what they hold, except with probability lieChance in 1000, when
// they are moved far away (a child the walk must cull although its primitives
// overlap) or blown up to cover everything.
struct TreeShape
{
    uint32_t levels;          // internal levels below the root
    uint32_t minFan;          // children per internal node: minFan..32
    uint32_t maxLeafPrims;    // 1..32
    uint32_t leafChance;      // in 1000: an internal-level child is a leaf anyway
    uint32_t lieChance;       // in 1000
};

inline Box buildSubtree(std::vector<std::vector<Child>>& nodes, uint32_t node, uint32_t depth,
                        const TreeShape& sh, Rng& rng, uint32_t& nextPrim, uint32_t nbPrims,
                        const std::function<Box(uint32_t)>& primBox)
{
    const uint32_t fan = sh.minFan + rng.below(33u - sh.minFan);
    Box nodeBox = { { 3e38f, 3e38f, 3e38f }, { -3e38f, -3e38f, -3e38f } };
    for (uint32_t k = 0; k < fan && nextPrim < nbPrims; ++k)
    {
        Child c;
        std::memset(&c, 0, sizeof(c));
        const bool leaf = depth >= sh.levels || rng.below(1000u) < sh.leafChance;
        Box b;
        if (leaf)
        {
            const uint32_t cnt = std::min(1u + rng.below(sh.maxLeafPrims), nbPrims - nextPrim);
            c.leaf = true; c.primStart = nextPrim; c.nbField = cnt;
            b = primBox(nextPrim);
            for (uint32_t i = 1; i < cnt; ++i) b = hullOf(b, primBox(nextPrim + i));
            nextPrim += cnt;
        }
        else
        {
            c.leaf = false;
            c.node = uint32_t(nodes.size());
            nodes.push_back(std::vector<Child>());
            b = buildSubtree(nodes, c.node, depth + 1u, sh, rng, nextPrim, nbPrims, primBox);
            if (nodes[c.node].empty())   // ran out of primitives: make it a 1-prim leaf instead
            {
                nodes.pop_back();
                if (nextPrim >= nbPrims) break;
                c.leaf = true; c.primStart = nextPrim; c.nbField = 1u; c.node = 0u;
                b = primBox(nextPrim);
                ++nextPrim;
            }
        }
        nodeBox = hullOf(nodeBox, b);
        const uint32_t r = rng.below(1000u);
        if (r < sh.lieChance)
        {
            if (r & 1u) b = Box{ { 9000.0f, 9000.0f, 9000.0f }, { 9001.0f, 9001.0f, 9001.0f } };
            else b = Box{ { -1e6f, -1e6f, -1e6f }, { 1e6f, 1e6f, 1e6f } };
        }
        c.box = b;
        nodes[node].push_back(c);
    }
    return nodeBox;
}

inline std::vector<std::vector<Child>> randomTree(const TreeShape& sh, Rng& rng, uint32_t nbPrims,
                                                  const std::function<Box(uint32_t)>& primBox,
                                                  uint32_t* nbUsedPrims)
{
    std::vector<std::vector<Child>> nodes(1);
    uint32_t next = 0u;
    // the root always has 32 children (or as many as the primitives allow)
    TreeShape rootShape = sh;
    rootShape.minFan = 32u;
    buildSubtree(nodes, 0u, 0u, rootShape, rng, next, nbPrims, primBox);
    *nbUsedPrims = next;
    return nodes;
}

// The DAG that drives the batched walk's stack past its reserve line: four levels
// that all share their children (root -> 32 x node 1 -> 32 x node 2 -> 32 x node 3),
// node 3 holding 32 leaves of which only those in `hitSlots` meet the rigid box.
// The CUDA's stack peaks at 31 * 3 + 32 = 125 here, inside its 192.
inline std::vector<std::vector<Child>> sharedDag(const Box& everywhere, const Box& nowhere,
                                                 uint32_t hitMask, uint32_t primsPerLeaf)
{
    std::vector<std::vector<Child>> nodes(4);
    for (uint32_t lvl = 0; lvl < 3u; ++lvl)
        for (uint32_t k = 0; k < 32u; ++k)
        {
            Child c; std::memset(&c, 0, sizeof(c));
            c.box = everywhere; c.leaf = false; c.node = lvl + 1u;
            nodes[lvl].push_back(c);
        }
    for (uint32_t k = 0; k < 32u; ++k)
    {
        Child c; std::memset(&c, 0, sizeof(c));
        c.box = ((hitMask >> k) & 1u) ? everywhere : nowhere;
        c.leaf = true; c.primStart = k * primsPerLeaf; c.nbField = primsPerLeaf;
        nodes[3].push_back(c);
    }
    return nodes;
}
} // namespace pxgbv32walk
