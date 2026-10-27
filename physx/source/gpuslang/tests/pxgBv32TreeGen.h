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

// pxgBv32TreeGen.h
//
// Synthetic Gu::BV32DataPacked trees for the multi-workgroup BVH refit fixtures
// (femClothSimRefitBoundLastArriver, softBodySimRefitBoundLastArriver).  The refit
// kernels (femClothRefitBda / softBodyRefitBda) schedule a node's refit on the LAST of
// its internal children to finish, so the shapes that matter are the ones that stress
// that schedule: levels with one node and levels with more than a thousand, deep and
// unbalanced chains, shallow nodes with only leaf children (refit at launch start next
// to the deepest ones), nodes with a single internal child, and a node filed one level
// deeper than its depth (the depth info only promises "children are deeper").
//
// The blob layout is the engine's mTriMeshData / mTetMeshData one: a uint4 header
// (nbVerts, nbPrims, maxDepth, nbNodes), nbNodes x BV32DataPacked (292 words), maxDepth
// x BV32DataDepthInfo, nbNodes x remapPackedNodeIndex, plus a 4-word trailer (the
// llvmpipe load-width precedent of femClothSimRefitBound).
//
// The node words follow what PhysX cooks (BV32Tree::createSOAformatNode): the w word of
// every VALID child slot of mMin/mMax is 0.0f (the refit's arrival counter lives in
// mMin[0].w and relies on it), every other mMin/mMax word is poisoned, and padding[0]
// (never serialised by the cooker, so arbitrary on the device) is garbage — random
// words, the empty-link value, and links that name a real internal slot of a WRONG
// node — to prove the kernel's parent-link cache can only miss, never mislead.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace pxgbv32gen
{
static const uint32_t BV32_WORDS = 292u;   // Gu::BV32DataPacked = 1168 B
static const uint32_t W_MMIN = 0u;
static const uint32_t W_MMAX = 128u;
static const uint32_t W_MDATA = 256u;
static const uint32_t W_NBNODES = 288u;
static const uint32_t W_DEPTH = 289u;
static const uint32_t W_PADDING0 = 290u;
static const uint32_t W_PADDING1 = 291u;

struct Child
{
    bool leaf;
    uint32_t node;     // internal: spec index of the child node
    uint32_t nbPrims;  // leaf: 1..32 primitives
};

// spec.nodes[0] is the root; levelOverride[i] (when not ~0u) files node i at that depth
// level instead of its tree depth (it must stay deeper than its parent and shallower
// than its children).
struct Spec
{
    std::vector<std::vector<Child>> nodes;
    std::vector<uint32_t> levelOverride;
};

struct Rng
{
    uint32_t s;
    uint32_t next() { s = s * 1664525u + 1013904223u; return s >> 8; }
    uint32_t below(uint32_t n) { return next() % n; }
    float unit() { return float(next() % 200001u) / 100000.0f - 1.0f; }
};

struct Tree
{
    std::vector<uint32_t> w;              // the whole blob (header .. trailer)
    uint32_t nbNodes = 0, maxDepth = 0, nbPrims = 0;
    uint32_t wDepthInfo = 0, wRemap = 0;
    std::vector<uint32_t> packedOf;       // spec index -> packed index (root -> 0)
    std::vector<uint32_t> level;          // packed index -> depth level
    std::vector<uint32_t> parentLink;     // packed index -> (parent << 5) | slot, ~0u for the root
    std::vector<uint32_t> levelCount;     // nodes per level
    uint32_t nodeWord(uint32_t node) const { return 4u + node * BV32_WORDS; }
};

inline uint32_t leafWord(uint32_t startIndex, uint32_t nbPrims) { return (startIndex << 7) | (nbPrims << 1) | 1u; }
inline uint32_t internalWord(uint32_t childOffset) { return childOffset << 11; }
inline uint32_t poisonBits(float p) { uint32_t u; std::memcpy(&u, &p, 4); return u; }

// Builds the blob; primitives are numbered in spec order (leaf by leaf), the caller
// creates nbPrims of them.
inline Tree build(const Spec& spec, Rng& rng, float poison)
{
    Tree t;
    const uint32_t n = uint32_t(spec.nodes.size());
    t.nbNodes = n;

    // depth by BFS from the root, then the overrides
    std::vector<uint32_t> depth(n, ~0u), parentSpec(n, ~0u), parentSlot(n, ~0u);
    depth[0] = 0u;
    std::vector<uint32_t> queue(1, 0u);
    for (size_t q = 0; q < queue.size(); ++q)
    {
        const uint32_t s = queue[q];
        for (uint32_t k = 0; k < uint32_t(spec.nodes[s].size()); ++k)
        {
            const Child& c = spec.nodes[s][k];
            if (c.leaf) continue;
            depth[c.node] = depth[s] + 1u;
            parentSpec[c.node] = s;
            parentSlot[c.node] = k;
            queue.push_back(c.node);
        }
    }
    for (uint32_t s = 0; s < n && s < uint32_t(spec.levelOverride.size()); ++s)
        if (spec.levelOverride[s] != ~0u) depth[s] = spec.levelOverride[s];
    for (uint32_t s = 0; s < n; ++s) t.maxDepth = std::max(t.maxDepth, depth[s] + 1u);

    // packed numbering: root 0, the rest shuffled
    t.packedOf.resize(n);
    std::vector<uint32_t> ids;
    for (uint32_t i = 1; i < n; ++i) ids.push_back(i);
    for (uint32_t i = uint32_t(ids.size()); i > 1u; --i) std::swap(ids[i - 1u], ids[rng.below(i)]);
    t.packedOf[0] = 0u;
    for (uint32_t i = 1; i < n; ++i) t.packedOf[i] = ids[i - 1u];

    t.wDepthInfo = 4u + n * BV32_WORDS;
    t.wRemap = t.wDepthInfo + t.maxDepth * 2u;
    t.w.assign(t.wRemap + n + 4u, 0u);
    t.level.assign(n, 0u);
    t.parentLink.assign(n, ~0u);

    const uint32_t P = poisonBits(poison);
    uint32_t prim = 0u;
    for (uint32_t s = 0; s < n; ++s)
    {
        const uint32_t p = t.packedOf[s];
        const uint32_t nw = t.nodeWord(p);
        const uint32_t nb = uint32_t(spec.nodes[s].size());
        for (uint32_t k = 0; k < 32u; ++k)
            for (uint32_t c = 0; c < 4u; ++c)
            {
                const bool validW = (c == 3u) && k < nb;
                t.w[nw + W_MMIN + k * 4u + c] = validW ? 0u : P;
                t.w[nw + W_MMAX + k * 4u + c] = validW ? 0u : P;
            }
        for (uint32_t k = 0; k < nb; ++k)
        {
            const Child& c = spec.nodes[s][k];
            if (c.leaf) { t.w[nw + W_MDATA + k] = leafWord(prim, c.nbPrims); prim += c.nbPrims; }
            else t.w[nw + W_MDATA + k] = internalWord(t.packedOf[c.node]);
        }
        t.w[nw + W_NBNODES] = nb;
        t.w[nw + W_DEPTH] = depth[s];
        t.w[nw + W_PADDING1] = rng.next();
        t.level[p] = depth[s];
        if (s != 0u) t.parentLink[p] = (t.packedOf[parentSpec[s]] << 5) | parentSlot[s];
    }
    t.nbPrims = prim;

    // padding[0]: garbage the parent-link cache must reject
    std::vector<uint32_t> internalSlots;   // (packed << 5) | slot of every internal child slot
    for (uint32_t p = 0; p < n; ++p)
        for (uint32_t k = 0; k < t.w[t.nodeWord(p) + W_NBNODES]; ++k)
            if ((t.w[t.nodeWord(p) + W_MDATA + k] & 1u) == 0u) internalSlots.push_back((p << 5) | k);
    for (uint32_t p = 0; p < n; ++p)
    {
        uint32_t g;
        switch (rng.below(4u))
        {
        case 0: g = rng.next() * 2654435761u; break;
        case 1: g = 0xFFFFFFFFu; break;
        case 2: g = 0u; break;
        default:
            g = internalSlots.empty() ? 0u : internalSlots[rng.below(uint32_t(internalSlots.size()))];
            if (g == t.parentLink[p]) g ^= 0x20u;   // a real slot, but never c's own
            break;
        }
        t.w[t.nodeWord(p) + W_PADDING0] = g;
    }

    // depth info and the remap, shuffled within each level
    t.levelCount.assign(t.maxDepth, 0u);
    std::vector<std::vector<uint32_t>> perLevel(t.maxDepth);
    for (uint32_t p = 0; p < n; ++p) perLevel[t.level[p]].push_back(p);
    uint32_t off = 0u;
    for (uint32_t l = 0; l < t.maxDepth; ++l)
    {
        std::vector<uint32_t>& v = perLevel[l];
        for (uint32_t i = uint32_t(v.size()); i > 1u; --i) std::swap(v[i - 1u], v[rng.below(i)]);
        t.w[t.wDepthInfo + l * 2u + 0] = off;
        t.w[t.wDepthInfo + l * 2u + 1] = uint32_t(v.size());
        t.levelCount[l] = uint32_t(v.size());
        for (uint32_t i = 0; i < uint32_t(v.size()); ++i) t.w[t.wRemap + off + i] = v[i];
        off += uint32_t(v.size());
    }

    t.w[1] = prim;
    t.w[2] = t.maxDepth;
    t.w[3] = n;
    return t;
}

// --- spec shapes --------------------------------------------------------------------

// Root -> `mid` internal nodes -> `perMid` leaf-only nodes each: the SnippetDeformable-
// Surface shape (1 / 32 / 1024).  Bottom nodes carry 1..4 leaves of 1..3 primitives.
inline Spec wideSpec(uint32_t mid, uint32_t perMid, Rng& rng)
{
    Spec s;
    s.nodes.resize(1u + mid + mid * perMid);
    for (uint32_t m = 0; m < mid; ++m)
    {
        s.nodes[0].push_back(Child{ false, 1u + m, 0u });
        for (uint32_t b = 0; b < perMid; ++b)
        {
            const uint32_t node = 1u + mid + m * perMid + b;
            s.nodes[1u + m].push_back(Child{ false, node, 0u });
            const uint32_t nbLeaves = 1u + rng.below(4u);
            for (uint32_t l = 0; l < nbLeaves; ++l) s.nodes[node].push_back(Child{ true, 0u, 1u + rng.below(3u) });
        }
    }
    return s;
}

// A deep unbalanced chain: every chain node has ONE internal child (the next chain
// node) mixed in with leaves, plus — every other level — a leaf-only side node, i.e. a
// node refit at launch start next to the deepest one.  One side node is filed one level
// deeper than its depth.
inline Spec chainSpec(uint32_t depth, Rng& rng)
{
    Spec s;
    s.nodes.resize(depth);
    for (uint32_t d = 0; d + 1u < depth; ++d)
    {
        const uint32_t nbLeaves = rng.below(4u);
        const uint32_t slot = rng.below(nbLeaves + 1u);
        for (uint32_t l = 0; l <= nbLeaves; ++l)
        {
            if (l == slot) s.nodes[d].push_back(Child{ false, d + 1u, 0u });
            if (l < nbLeaves) s.nodes[d].push_back(Child{ true, 0u, 1u + rng.below(32u) });
        }
        if ((d & 1u) == 1u)
        {
            const uint32_t side = uint32_t(s.nodes.size());
            s.nodes.push_back(std::vector<Child>());
            const uint32_t nbSideLeaves = 1u + rng.below(32u);
            for (uint32_t l = 0; l < nbSideLeaves; ++l) s.nodes[side].push_back(Child{ true, 0u, 1u + rng.below(4u) });
            s.nodes[d].push_back(Child{ false, side, 0u });
        }
    }
    s.nodes[depth - 1u].push_back(Child{ true, 0u, 32u });   // the deepest node: one full leaf
    // file the side node of level 1 (tree depth 2) two levels deeper
    s.levelOverride.assign(s.nodes.size(), ~0u);
    for (uint32_t i = depth; i < uint32_t(s.nodes.size()); ++i)
    {
        s.levelOverride[i] = 4u;   // its parent is chain node 1 (level 1); depth >= 6 keeps level 4 valid
        break;
    }
    return s;
}

// Random fan-out (1..32 children), internal with a probability that falls with depth.
inline void randomSubtree(Spec& s, uint32_t node, uint32_t depth, uint32_t maxDepth, Rng& rng)
{
    const uint32_t nb = 1u + rng.below(depth == 0u ? 32u : 12u);
    for (uint32_t k = 0; k < nb; ++k)
    {
        const int pct = 70 - 15 * int(depth);
        const bool internal = depth + 1u < maxDepth && int(rng.below(100u)) < pct;
        if (internal)
        {
            const uint32_t child = uint32_t(s.nodes.size());
            s.nodes.push_back(std::vector<Child>());
            s.nodes[node].push_back(Child{ false, child, 0u });
            randomSubtree(s, child, depth + 1u, maxDepth, rng);
        }
        else
            s.nodes[node].push_back(Child{ true, 0u, 1u + rng.below(32u) });
    }
}
inline Spec randomSpec(uint32_t maxDepth, Rng& rng)
{
    Spec s;
    s.nodes.resize(1);
    randomSubtree(s, 0u, 0u, maxDepth, rng);
    return s;
}

// The whole tree in one node: the root refits itself at launch start.
inline Spec singleSpec(Rng& rng)
{
    Spec s;
    s.nodes.resize(1);
    const uint32_t nb = 1u + rng.below(32u);
    for (uint32_t k = 0; k < nb; ++k) s.nodes[0].push_back(Child{ true, 0u, 1u + rng.below(32u) });
    return s;
}
} // namespace pxgbv32gen
