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

// Verification of fcMidphase.slang against exact CPU transcriptions.  All
// fixtures use EXACT float arithmetic — vertices/bounds on a 1/64 grid,
// quaternions from the (+-1/2, +-1/2, +-1/2, +-1/2) family plus identity,
// power-of-two mesh/heightfield scales and integer translations — so every
// comparison in all three cores is bit-deterministic across CPU and GPU
// (all products are exact halvings, immune to fma contraction) and the
// references compare EXACTLY, with boundary cases placed deliberately ON
// the thresholds (e.g. dist^2 == tolerance pins the inclusive <=).
//
// fmVertexAabb: closestPoint clamp + inclusive squared-distance gate +
// ballot pack, UNCAPPED stack writes (faithful).  fmSphereTree: the
// ClothParticleTreeTraverser BV32 walk (world-space spheres AND the
// vertex-mesh flavour's shape2Vertex(transformInv(p)) pre-transform),
// counter-vs-multiset equality plus a small-stack run pinning the
// pushOntoStackFullWarp capacity gate (reservation ungated).  fmHfBound:
// the world->heightfield transformFast(getInverse) bound chain, the
// FAITHFUL mis-parenthesised culling gate (off-grid-in-+z and
// off-grid-in-+x-and-+z bounds are admitted and produce real clamped-edge
// pairs — a corrected gate would cull them), the float->uint saturating
// y-range conversion (negative bounds saturate to miny = 0), hole/height
// culling, per-lane multi-pair warp packing and the global stack cap.

#include "pxgTestFramework.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };
struct FmCase { F4 a, b, c, d, e, f, g; U4 ids; };
struct FmParams { uint32_t numCases, stackSize, nbRows, nbCols; };
struct Bv32Node
{
    F4 mMin[32];
    F4 mMax[32];
    uint32_t mData[32];
    uint32_t mNbNodes, mDepth, pad0, pad1;
};

struct V3 { float x, y, z; };
static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 add(V3 a, V3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 mul(V3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
static V3 divv(V3 a, V3 b) { return { a.x / b.x, a.y / b.y, a.z / b.z }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 v3(F4 f) { return { f.x, f.y, f.z }; }

static V3 quatRotate(F4 q, V3 v)
{
    const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d = q.x * vx + q.y * vy + q.z * vz;
    return { vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d,
             vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d,
             vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d };
}
static V3 quatRotateInv(F4 q, V3 v)
{
    const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d = q.x * vx + q.y * vy + q.z * vz;
    return { vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * d,
             vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * d,
             vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * d };
}

// exact quaternion family: identity or all sign choices of (1/2,1/2,1/2,1/2)
static F4 exactQuat(std::mt19937& rng)
{
    if (rng() % 4 == 0) return { 0, 0, 0, 1 };
    const uint32_t s = rng() % 16;
    return { (s & 1) ? -0.5f : 0.5f, (s & 2) ? -0.5f : 0.5f,
             (s & 4) ? -0.5f : 0.5f, (s & 8) ? -0.5f : 0.5f };
}
static float grid64(std::mt19937& rng, float lo, float hi) // 1/64-grid value in [lo, hi]
{
    const int n = int((hi - lo) * 64.0f);
    return lo + float(int(rng() % uint32_t(n + 1))) / 64.0f;
}

struct AABB { float mn[3], mx[3]; };
static bool bIntersect(const AABB& a, const AABB& b)
{
    return !(b.mn[0] > a.mx[0] || a.mn[0] > b.mx[0] || b.mn[1] > a.mx[1] || a.mn[1] > b.mx[1] ||
             b.mn[2] > a.mx[2] || a.mn[2] > b.mx[2]);
}

static const uint32_t PAIRPOISON = 0xffffffffu;

// ============================================================================
// PXG_TEST 1: fmVertexAabb
// ============================================================================
PXG_TEST(fcMidphase_vertexAabb)
{
    const uint32_t NC = 2011; // deliberately not a multiple of 32
    std::mt19937 rng(990701);
    std::vector<FmCase> cases(NC);
    size_t boundaryN = 0, insideN = 0, rejectN = 0;
    std::vector<char> expectHit(NC);
    for (uint32_t i = 0; i < NC; ++i)
    {
        FmCase& cs = cases[i];
        std::memset(&cs, 0, sizeof(cs));
        const float cd = 0.25f + float(i % 5) * 0.125f; // exact
        V3 bmn = { grid64(rng, -4, 2), grid64(rng, -4, 2), grid64(rng, -4, 2) };
        V3 bmx = add(bmn, V3{ grid64(rng, 0.5f, 3), grid64(rng, 0.5f, 3), grid64(rng, 0.5f, 3) });
        V3 v;
        bool hit;
        switch (i % 4)
        {
        case 0: // inside the box: closest == v, dist 0
            v = add(bmn, V3{ 0.25f, 0.25f, 0.25f });
            hit = true; ++insideN;
            break;
        case 1: // outside one face by EXACTLY cd: dist^2 == cd^2, inclusive gate hits
            v = { bmx.x + cd, bmn.y + 0.25f, bmn.z + 0.25f };
            hit = true; ++boundaryN;
            break;
        case 2: // outside by less than cd (exact grid offsets)
            v = { bmn.x - 0.125f, bmn.y + 0.5f, bmx.z + 0.0625f };
            hit = (0.125f * 0.125f + 0.0625f * 0.0625f) <= cd * cd;
            break;
        default: // far outside
            v = { bmx.x + cd + 1.0f, bmx.y + 2.0f, bmn.z };
            hit = false; ++rejectN;
            break;
        }
        expectHit[i] = hit ? 1 : 0;
        cs.a = { v.x, v.y, v.z, 0 };
        cs.b = { bmn.x, bmn.y, bmn.z, 0 };
        cs.c = { bmx.x, bmx.y, bmx.z, cd * cd };
        cs.ids = { i, 5000 + i, 0, 0 };
    }
    uint32_t total = 0;
    for (uint32_t i = 0; i < NC; ++i) total += expectHit[i];

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t CAP = total + 64;
    std::vector<U4> outInit(CAP, { PAIRPOISON, PAIRPOISON, PAIRPOISON, PAIRPOISON });
    std::vector<uint32_t> cnt0(1, 0);
    std::vector<F4> dbgInit(NC, { -7, -7, -7, -7 });
    std::vector<F4> dumF4(4, { 0, 0, 0, 0 });
    std::vector<U4> dumU4(4, { 0, 0, 0, 0 });
    std::vector<Bv32Node> dumNodes(1);
    std::memset(dumNodes.data(), 0, sizeof(Bv32Node));
    std::vector<uint32_t> dumU(4, 0);

    Kernel k("fcMidphase.spv", "fmVertexAabb",
             { (uint32_t)sizeof(FmCase), 16, 16, (uint32_t)sizeof(Bv32Node), 4, 16, 4, 16 },
             sizeof(FmParams));
    PxgBufferHandle bC = createBufferFrom(dev, cases);
    PxgBufferHandle bV = createBufferFrom(dev, dumF4);
    PxgBufferHandle bP = createBufferFrom(dev, dumU4);
    PxgBufferHandle bN = createBufferFrom(dev, dumNodes);
    PxgBufferHandle bS = createBufferFrom(dev, dumU);
    PxgBufferHandle bO = createBufferFrom(dev, outInit);
    PxgBufferHandle bW = createBufferFrom(dev, cnt0);
    PxgBufferHandle bD = createBufferFrom(dev, dbgInit);
    FmParams prm = { NC, CAP, 0, 0 };
    const uint32_t batches = (NC + 31) / 32;
    k.dispatch((batches + 63) / 64, { bC, bV, bP, bN, bS, bO, bW, bD }, prm);
    dev->flush();
    const auto gOut = downloadBuffer<U4>(dev, bO, CAP);
    const auto gDbg = downloadBuffer<F4>(dev, bD, NC);
    const uint32_t gWc = downloadBuffer<uint32_t>(dev, bW, 1)[0];

    PXG_CHECK_MSG(gWc == total, "vertexAabb: counter %u != %u", gWc, total);
    // slot map via unique payload .y
    std::vector<int> slotOf(NC, -1);
    size_t mm = 0;
    for (uint32_t s = 0; s < std::min(gWc, CAP); ++s)
    {
        const uint32_t caseIdx = gOut[s].y - 5000;
        if (caseIdx >= NC || slotOf[caseIdx] != -1 || gOut[s].x != caseIdx ||
            gOut[s].z != 0 || gOut[s].w != 0) { ++mm; continue; }
        slotOf[caseIdx] = (int)s;
    }
    for (uint32_t i = 0; i < NC; ++i)
    {
        if ((slotOf[i] != -1) != (expectHit[i] != 0)) ++mm;
        if ((gDbg[i].x != 0.0f) != (expectHit[i] != 0)) ++mm;
    }
    // batch contiguity (exact refs: every batch is clean)
    size_t mmB = 0;
    for (uint32_t b = 0; b * 32 < NC; ++b)
    {
        int prev = -1;
        for (uint32_t l = 0; l < 32 && b * 32 + l < NC; ++l)
        {
            const uint32_t i = b * 32 + l;
            if (!expectHit[i] || slotOf[i] == -1) continue;
            if (prev != -1 && slotOf[i] != prev + 1) ++mmB;
            prev = slotOf[i];
        }
    }
    size_t dirty = 0;
    for (uint32_t s = gWc; s < CAP; ++s) if (gOut[s].x != PAIRPOISON) ++dirty;
    PXG_CHECK_MSG(mm == 0 && mmB == 0 && dirty == 0,
                  "vertexAabb: %zu mismatches, %zu contiguity breaks, %zu dirty", mm, mmB, dirty);
    PXG_CHECK_MSG(boundaryN > 400 && insideN > 400 && rejectN > 400,
                  "vertexAabb coverage thin (b=%zu i=%zu r=%zu)", boundaryN, insideN, rejectN);

    for (PxgBufferHandle h : { bC, bV, bP, bN, bS, bO, bW, bD }) dev->destroyBuffer(h);
}

// ============================================================================
// PXG_TEST 2: fmSphereTree
// ============================================================================
static uint32_t buildNode(std::vector<Bv32Node>& nodes, const std::vector<AABB>& prims,
                          uint32_t begin, uint32_t end, std::mt19937& rng, int depth)
{
    const uint32_t nodeIdx = (uint32_t)nodes.size();
    nodes.push_back(Bv32Node());
    std::memset(&nodes.back(), 0, sizeof(Bv32Node));
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
        AABB bounds = prims[b];
        for (uint32_t i = b + 1; i < e; ++i)
            for (int j = 0; j < 3; ++j)
            {
                bounds.mn[j] = std::min(bounds.mn[j], prims[i].mn[j]);
                bounds.mx[j] = std::max(bounds.mx[j], prims[i].mx[j]);
            }
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

static void refSphereTraverse(const std::vector<Bv32Node>& nodes, const std::vector<U4>& prims,
                              const std::vector<F4>& verts, const FmCase& w, std::vector<U4>& out)
{
    V3 center = v3(w.a);
    if (w.ids.z != 0)
    {
        const V3 local = quatRotateInv(w.d, sub(center, v3(w.e)));
        center = quatRotateInv(w.g, divv(quatRotate(w.g, local), v3(w.f)));
    }
    const float r = w.b.x;
    const AABB qb = { { center.x - r, center.y - r, center.z - r },
                      { center.x + r, center.y + r, center.z + r } };
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
                const U4& pi = prims[t];
                AABB tb;
                const V3 a = v3(verts[pi.x]), b = v3(verts[pi.y]), c = v3(verts[pi.z]);
                tb.mn[0] = std::min(std::min(a.x, b.x), c.x); tb.mx[0] = std::max(std::max(a.x, b.x), c.x);
                tb.mn[1] = std::min(std::min(a.y, b.y), c.y); tb.mx[1] = std::max(std::max(a.y, b.y), c.y);
                tb.mn[2] = std::min(std::min(a.z, b.z), c.z); tb.mx[2] = std::max(std::max(a.z, b.z), c.z);
                if (bIntersect(tb, qb))
                    out.push_back({ w.ids.x, w.ids.y, t, 0 });
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

static std::vector<std::array<uint32_t, 4>> sortedPairs(const std::vector<U4>& v, size_t n)
{
    std::vector<std::array<uint32_t, 4>> r;
    for (size_t i = 0; i < n; ++i) r.push_back({ v[i].x, v[i].y, v[i].z, v[i].w });
    std::sort(r.begin(), r.end());
    return r;
}

PXG_TEST(fcMidphase_sphereTree)
{
    std::mt19937 rng(990702);
    const uint32_t NP = 500, NQ = 96;
    std::vector<F4> verts(NP * 4);
    std::vector<U4> prims(NP);
    std::vector<AABB> bounds(NP);
    for (uint32_t t = 0; t < NP; ++t)
    {
        const V3 c0 = { grid64(rng, -2, 2), grid64(rng, -2, 2), grid64(rng, -2, 2) };
        for (uint32_t kk = 0; kk < 3; ++kk)
            verts[t * 4 + kk] = { c0.x + grid64(rng, -0.25f, 0.25f), c0.y + grid64(rng, -0.25f, 0.25f),
                                  c0.z + grid64(rng, -0.25f, 0.25f), 0 };
        prims[t] = { t * 4, t * 4 + 1, t * 4 + 2, 0 };
        AABB bb = { { 1e30f, 1e30f, 1e30f }, { -1e30f, -1e30f, -1e30f } };
        for (uint32_t kk = 0; kk < 3; ++kk)
        {
            const float* p = &verts[t * 4 + kk].x;
            for (int j = 0; j < 3; ++j) { bb.mn[j] = std::min(bb.mn[j], p[j]); bb.mx[j] = std::max(bb.mx[j], p[j]); }
        }
        bounds[t] = bb;
    }
    std::vector<Bv32Node> nodes;
    buildNode(nodes, bounds, 0, NP, rng, 0);

    std::vector<FmCase> work(NQ);
    size_t transformedN = 0, touchN = 0;
    for (uint32_t q = 0; q < NQ; ++q)
    {
        FmCase& w = work[q];
        std::memset(&w, 0, sizeof(w));
        const float r = 0.25f + float(q % 4) * 0.125f;
        w.b = { r, 0, 0, 0 };
        w.ids = { q, 7000 + q, 0, 0 };
        V3 target; // intended query center in tree space
        if (q % 8 == 7)
            target = { 60.0f + q, 60, 60 }; // no-hit query
        else if (q % 8 == 3)
        {
            // exact-touch query: bmx.x == chosen triangle's bounds.mn[0] exactly
            // ((a - r) + r == a for grid-exact a, r) — pins the touching-counts-
            // as-intersecting convention of the strict > comparisons
            const uint32_t tq = (q * 7 + 1) % NP;
            target = { bounds[tq].mn[0] - r,
                       (bounds[tq].mn[1] + bounds[tq].mx[1]) * 0.5f,
                       (bounds[tq].mn[2] + bounds[tq].mx[2]) * 0.5f };
            ++touchN;
        }
        else
            target = { grid64(rng, -2, 2), grid64(rng, -2, 2), grid64(rng, -2, 2) };
        if (q % 3 == 1 && q % 8 != 3)
        {
            // vertex-mesh flavour: world point -> transformInv -> shape2Vertex,
            // exact quats + power-of-two scales keep the chain bit-exact
            w.ids.z = 1;
            ++transformedN;
            w.d = exactQuat(rng);
            w.e = { float(int(rng() % 9)) - 4.0f, float(int(rng() % 9)) - 4.0f, float(int(rng() % 9)) - 4.0f, 0 };
            const float sc[3] = { 0.5f, 1.0f, 2.0f };
            w.f = { sc[rng() % 3], sc[rng() % 3], sc[rng() % 3], 0 };
            w.g = exactQuat(rng);
            // world = inverse chain of the intended tree-space target:
            // target = shape2Vertex(transformInv(world)) => world = transform(vertex2Shape(target));
            // signed-permutation quats + power-of-two scales make the round trip bit-exact
            const V3 rt = quatRotate(w.g, target);
            const V3 v2s = quatRotateInv(w.g, V3{ rt.x * w.f.x, rt.y * w.f.y, rt.z * w.f.z });
            const V3 world = add(quatRotate(w.d, v2s), v3(w.e));
            w.a = { world.x, world.y, world.z, 0 };
        }
        else
            w.a = { target.x, target.y, target.z, 0 };
    }

    std::vector<U4> expect;
    for (const auto& w : work)
        refSphereTraverse(nodes, prims, verts, w, expect);
    const uint32_t total = (uint32_t)expect.size();
    PXG_CHECK_MSG(total > 300, "sphereTree: scene too sparse (%u)", total);

    PxgSlangComputeDevice* dev = testDevice();
    auto run = [&](uint32_t stackSize, uint32_t bufCap) -> std::pair<std::vector<U4>, uint32_t>
    {
        std::vector<U4> outInit(bufCap, { PAIRPOISON, PAIRPOISON, PAIRPOISON, PAIRPOISON });
        std::vector<uint32_t> cnt0(1, 0);
        std::vector<F4> dbgInit(NQ, { -7, -7, -7, -7 });
        std::vector<uint32_t> dumU(4, 0);
        PxgBufferHandle bC = createBufferFrom(dev, work);
        PxgBufferHandle bV = createBufferFrom(dev, verts);
        PxgBufferHandle bP = createBufferFrom(dev, prims);
        PxgBufferHandle bN = createBufferFrom(dev, nodes);
        PxgBufferHandle bS = createBufferFrom(dev, dumU);
        PxgBufferHandle bO = createBufferFrom(dev, outInit);
        PxgBufferHandle bW = createBufferFrom(dev, cnt0);
        PxgBufferHandle bD = createBufferFrom(dev, dbgInit);
        Kernel k("fcMidphase.spv", "fmSphereTree",
                 { (uint32_t)sizeof(FmCase), 16, 16, (uint32_t)sizeof(Bv32Node), 4, 16, 4, 16 },
                 sizeof(FmParams));
        FmParams prm = { NQ, stackSize, 0, 0 };
        k.dispatch(NQ, { bC, bV, bP, bN, bS, bO, bW, bD }, prm);
        dev->flush();
        auto pairs = downloadBuffer<U4>(dev, bO, bufCap);
        const uint32_t wc = downloadBuffer<uint32_t>(dev, bW, 1)[0];
        for (PxgBufferHandle h : { bC, bV, bP, bN, bS, bO, bW, bD }) dev->destroyBuffer(h);
        return { pairs, wc };
    };

    const auto full = run(total + 64, total + 64);
    PXG_CHECK_MSG(full.second == total, "sphereTree: counter %u != %u", full.second, total);
    const auto ge = sortedPairs(full.first, total);
    const auto re = sortedPairs(expect, total);
    size_t mm = 0;
    for (uint32_t i = 0; i < total; ++i) if (ge[i] != re[i]) ++mm;
    size_t dirty = 0;
    for (uint32_t i = total; i < total + 64; ++i) if (full.first[i].x != PAIRPOISON) ++dirty;
    PXG_CHECK_MSG(mm == 0 && dirty == 0, "sphereTree: %zu mismatches, %zu dirty", mm, dirty);

    // capacity run: reservation ungated (counter = total), stores capped
    const uint32_t TINY = total / 3;
    const auto capped = run(TINY, total + 64);
    PXG_CHECK_MSG(capped.second == total, "sphereTree cap: counter %u != %u", capped.second, total);
    std::map<std::array<uint32_t, 4>, int> mult;
    for (const auto& p : re) ++mult[p];
    size_t bad = 0;
    for (uint32_t s = 0; s < TINY; ++s)
    {
        const std::array<uint32_t, 4> p = { capped.first[s].x, capped.first[s].y, capped.first[s].z, capped.first[s].w };
        auto it = mult.find(p);
        if (it == mult.end() || it->second == 0) ++bad;
        else --it->second;
    }
    for (uint32_t s = TINY; s < total + 64; ++s) if (capped.first[s].x != PAIRPOISON) ++bad;
    PXG_CHECK_MSG(bad == 0, "sphereTree cap: %zu bad slots", bad);
    // every exact-touch query must have found its target triangle
    size_t touchHits = 0;
    for (uint32_t q = 0; q < NQ; ++q)
    {
        if (q % 8 != 3 || q % 8 == 7) continue;
        const uint32_t tq = (q * 7 + 1) % NP;
        for (const auto& e : expect)
            if (e.x == q && e.y == 7000 + q && e.z == tq) { ++touchHits; break; }
    }
    PXG_CHECK_MSG(touchHits == touchN && touchN >= 10,
                  "sphereTree: touch coverage %zu/%zu", touchHits, touchN);
    PXG_CHECK_MSG(transformedN > 25, "sphereTree: too few transformed queries (%zu)", transformedN);
}

// ============================================================================
// PXG_TEST 3: fmHfBound
// ============================================================================
static const uint32_t HF_ROWS = 30, HF_COLS = 22;

static float hfHeightRef(const std::vector<uint32_t>& smp, uint32_t v)
{
    return float(int16_t(smp[v] & 0xffffu));
}
static uint32_t hfMat0(const std::vector<uint32_t>& smp, uint32_t v) { return (smp[v] >> 16) & 0x7fu; }
static uint32_t hfMat1(const std::vector<uint32_t>& smp, uint32_t v) { return (smp[v] >> 24) & 0x7fu; }
static uint32_t clampI(int v, int lo, int hi) { return uint32_t(std::min(std::max(v, lo), hi)); }
static uint32_t f2uSatRef(float x)
{
    if (!(x > 0.0f)) return 0u;
    if (x >= 4294967296.0f) return 0xffffffffu;
    return uint32_t(x);
}

struct HfRangeRef
{
    uint32_t minRow = 0, maxRow = 0, minColumn = 0, maxColumn = 0;
    float minyF = 0, maxyF = 0;
    uint32_t nbPairs = 0;
};

static HfRangeRef refHfRange(const FmCase& cs, const std::vector<uint32_t>& smp, std::vector<U4>* pairs)
{
    HfRangeRef r;
    if (cs.ids.y >= cs.ids.w) return r; // element gate

    V3 bmn, bmx;
    const float cd = cs.a.w;
    if (cs.ids.z == 0)
    {
        const V3 a = v3(cs.a), b = v3(cs.b), c = v3(cs.c);
        bmn = { std::min(std::min(a.x, b.x), c.x) - cd, std::min(std::min(a.y, b.y), c.y) - cd,
                std::min(std::min(a.z, b.z), c.z) - cd };
        bmx = { std::max(std::max(a.x, b.x), c.x) + cd, std::max(std::max(a.y, b.y), c.y) + cd,
                std::max(std::max(a.z, b.z), c.z) + cd };
    }
    else
    {
        bmn = { cs.a.x - cd, cs.a.y - cd, cs.a.z - cd };
        bmx = { cs.a.x + cd, cs.a.y + cd, cs.a.z + cd };
    }

    const F4 q = cs.d;
    const F4 qinv = { -q.x, -q.y, -q.z, q.w };
    const V3 centerW = mul(add(bmn, bmx), 0.5f);
    const V3 extentW = mul(sub(bmx, bmn), 0.5f);
    // getInverse() materialized first, then inv.transform(c): two rotations
    // plus a vector add (matches CUDA op order; NOT rotateInv(c - p))
    const V3 invP = quatRotateInv(q, { -cs.e.x, -cs.e.y, -cs.e.z });
    const V3 center = add(quatRotateInv(q, centerW), invP);
    // PxMat33(qinv) columns
    const float x = qinv.x, y = qinv.y, z = qinv.z, w = qinv.w;
    const float x2 = x + x, y2 = y + y, z2 = z + z;
    const float xx = x2 * x, yy = y2 * y, zz = z2 * z;
    const float xy = x2 * y, xz = x2 * z, xw = x2 * w;
    const float yz = y2 * z, yw = y2 * w, zw = z2 * w;
    const V3 c0 = { 1.0f - yy - zz, xy + zw, xz - yw };
    const V3 c1 = { xy - zw, 1.0f - xx - zz, yz + xw };
    const V3 c2 = { xz + yw, yz - xw, 1.0f - xx - yy };
    const V3 e0 = mul(c0, extentW.x), e1 = mul(c1, extentW.y), e2 = mul(c2, extentW.z);
    const V3 wext = { std::fabs(e0.x) + std::fabs(e1.x) + std::fabs(e2.x),
                      std::fabs(e0.y) + std::fabs(e1.y) + std::fabs(e2.y),
                      std::fabs(e0.z) + std::fabs(e1.z) + std::fabs(e2.z) };
    V3 lmn = sub(center, wext);
    V3 lmx = add(center, wext);
    lmn = { lmn.x / std::fabs(cs.f.x), lmn.y / cs.f.y, lmn.z / std::fabs(cs.f.z) };
    lmx = { lmx.x / std::fabs(cs.f.x), lmx.y / cs.f.y, lmx.z / std::fabs(cs.f.z) };

    // FAITHFUL: ! binds only to the first comparison
    if (!(lmn.x > float(HF_ROWS - 1)) || (lmn.z > float(HF_COLS - 1)) || (lmx.x < 0.0f) || (lmx.z < 0.0f))
    {
        r.minRow = clampI(int(std::floor(lmn.x)), 0, int(HF_ROWS) - 2);
        r.maxRow = clampI(int(std::ceil(lmx.x)), 0, int(HF_ROWS) - 1);
        r.minColumn = clampI(int(std::floor(lmn.z)), 0, int(HF_COLS) - 2);
        r.maxColumn = clampI(int(std::ceil(lmx.z)), 0, int(HF_COLS) - 1);
        r.minyF = float(f2uSatRef(lmn.y));
        r.maxyF = float(f2uSatRef(lmx.y));
        if (2u * (r.maxColumn - r.minColumn) * (r.maxRow - r.minRow) > 0u)
        {
            const uint32_t columnSpan = r.maxColumn - r.minColumn;
            const uint32_t totalNumProcessed = (r.maxRow - r.minRow) * columnSpan * 2u;
            for (uint32_t i = 0; i < totalNumProcessed; ++i)
            {
                const uint32_t index = i / 2u;
                const uint32_t vertexIndex = (r.minRow + index / columnSpan) * HF_COLS + (r.minColumn + index % columnSpan);
                const float h0 = hfHeightRef(smp, vertexIndex);
                const float h1 = hfHeightRef(smp, vertexIndex + 1u);
                const float h2 = hfHeightRef(smp, vertexIndex + HF_COLS);
                const float h3 = hfHeightRef(smp, vertexIndex + HF_COLS + 1u);
                const bool con0 = r.maxyF < h0 && r.maxyF < h1 && r.maxyF < h2 && r.maxyF < h3;
                const bool con1 = r.minyF > h0 && r.minyF > h1 && r.minyF > h2 && r.minyF > h3;
                if (!(con0 || con1))
                {
                    const bool isM1 = (i & 1u) != 0u;
                    const uint32_t material = isM1 ? hfMat1(smp, vertexIndex) : hfMat0(smp, vertexIndex);
                    if (material != 127u)
                    {
                        ++r.nbPairs;
                        if (pairs)
                            pairs->push_back({ cs.ids.x, cs.ids.y,
                                               isM1 ? ((vertexIndex << 1) + 1u) : (vertexIndex << 1), 0 });
                    }
                }
            }
        }
    }
    return r;
}

PXG_TEST(fcMidphase_hfBound)
{
    std::mt19937 rng(990703);
    // rectangular grid with signed heights, holes and flag bits in the material bytes
    std::vector<uint32_t> samples(HF_ROWS * HF_COLS);
    for (uint32_t s = 0; s < samples.size(); ++s)
    {
        const int16_t h = int16_t(int(rng() % 101) - 40); // heights in [-40, 60]
        uint32_t m0 = (s % 13 == 3) ? 127u : (rng() % 5);
        uint32_t m1 = (s % 17 == 5) ? 127u : (rng() % 5);
        if (rng() % 3 == 0) m0 |= 0x80u; // tess/reserved flag bits: must be masked off
        if (rng() % 4 == 0) m1 |= 0x80u;
        samples[s] = (uint32_t(uint16_t(h))) | (m0 << 16) | (m1 << 24);
    }

    const uint32_t NC = 1024;
    std::vector<FmCase> cases(NC);
    size_t offGridZ = 0, offGridXZ = 0, negY = 0, gatedN = 0, culledXN = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        FmCase& cs = cases[i];
        std::memset(&cs, 0, sizeof(cs));
        cs.d = exactQuat(rng);
        cs.e = { float(int(rng() % 9)) - 4.0f, float(int(rng() % 9)) - 4.0f, float(int(rng() % 9)) - 4.0f, 0 };
        const float sc[3] = { 0.5f, 1.0f, 2.0f };
        cs.f = { sc[rng() % 3], sc[rng() % 3], sc[rng() % 3], 0 };
        const float cd = 0.25f + float(i % 3) * 0.25f;

        // pick a target range in heightfield space, then take the element to
        // world through the exact transform so the GPU recovers it exactly
        V3 hfP;
        switch (i % 8)
        {
        case 3: hfP = { grid64(rng, 2, 20), grid64(rng, -20, -1), grid64(rng, 2, 14) }; ++negY; break; // below all heights / negative y -> miny saturates to 0
        case 5: hfP = { grid64(rng, 2, 20), grid64(rng, -5, 55), float(HF_COLS) + grid64(rng, 1, 4) }; ++offGridZ; break; // off-grid +z: buggy gate admits
        case 6: hfP = { float(HF_ROWS) + grid64(rng, 1, 3), grid64(rng, -5, 55), float(HF_COLS) + grid64(rng, 1, 3) }; ++offGridXZ; break; // off-grid +x AND +z
        case 7: // off-grid +x with IN-GRID z: the gate's only culling branch fires
            hfP = { float(HF_ROWS) + grid64(rng, 2, 4), grid64(rng, -5, 55), grid64(rng, 2, 14) };
            cs.f.x = 1.0f + float(rng() % 2); // keep lmn.x > nbRows-1 after the 1/scale multiply
            ++culledXN;
            break;
        default: hfP = { grid64(rng, 1, 25), grid64(rng, -5, 55), grid64(rng, 1, 18) }; break;
        }
        // heightfield-space -> world: w = hfT.transform(scale * p)
        const V3 scaled = { hfP.x * cs.f.x, hfP.y * cs.f.y, hfP.z * cs.f.z };
        const V3 world = add(quatRotate(cs.d, scaled), v3(cs.e));

        const uint32_t kind = (i % 2);
        cs.ids = { i, (i % 16 == 9) ? 100u + i : i, kind, (i % 16 == 9) ? 50u : NC + 100u };
        if (i % 16 == 9) ++gatedN; // elementIdx >= numElements
        if (kind == 0)
        {
            const V3 dx = { grid64(rng, 0.25f, 2), 0, 0 };
            const V3 dz = { 0, grid64(rng, -1, 1), grid64(rng, 0.25f, 2) };
            const V3 b = add(world, quatRotate(cs.d, V3{ dx.x * cs.f.x, 0, 0 }));
            const V3 c = add(world, quatRotate(cs.d, V3{ 0, dz.y * cs.f.y, dz.z * cs.f.z }));
            cs.a = { world.x, world.y, world.z, cd };
            cs.b = { b.x, b.y, b.z, 0 };
            cs.c = { c.x, c.y, c.z, 0 };
        }
        else
            cs.a = { world.x, world.y, world.z, cd };
    }

    std::vector<U4> expect;
    std::vector<HfRangeRef> refR(NC);
    std::vector<std::pair<size_t, size_t>> caseRange(NC);
    for (uint32_t i = 0; i < NC; ++i)
    {
        const size_t b = expect.size();
        refR[i] = refHfRange(cases[i], samples, &expect);
        caseRange[i] = { b, expect.size() };
    }
    const uint32_t total = (uint32_t)expect.size();
    PXG_CHECK_MSG(total > 800, "hfBound: scene too sparse (%u)", total);
    size_t offZPairs = 0;
    for (uint32_t i = 0; i < NC; ++i)
        if (i % 8 == 5 || i % 8 == 6) offZPairs += refR[i].nbPairs;

    PxgSlangComputeDevice* dev = testDevice();
    auto run = [&](uint32_t stackSize, uint32_t bufCap) -> std::tuple<std::vector<U4>, uint32_t, std::vector<F4>>
    {
        std::vector<U4> outInit(bufCap, { PAIRPOISON, PAIRPOISON, PAIRPOISON, PAIRPOISON });
        std::vector<uint32_t> cnt0(1, 0);
        std::vector<F4> dbgInit(NC, { -7, -7, -7, -7 });
        std::vector<F4> dumF4(4, { 0, 0, 0, 0 });
        std::vector<U4> dumU4(4, { 0, 0, 0, 0 });
        std::vector<Bv32Node> dumNodes(1);
        std::memset(dumNodes.data(), 0, sizeof(Bv32Node));
        PxgBufferHandle bC = createBufferFrom(dev, cases);
        PxgBufferHandle bV = createBufferFrom(dev, dumF4);
        PxgBufferHandle bP = createBufferFrom(dev, dumU4);
        PxgBufferHandle bN = createBufferFrom(dev, dumNodes);
        PxgBufferHandle bS = createBufferFrom(dev, samples);
        PxgBufferHandle bO = createBufferFrom(dev, outInit);
        PxgBufferHandle bW = createBufferFrom(dev, cnt0);
        PxgBufferHandle bD = createBufferFrom(dev, dbgInit);
        Kernel k("fcMidphase.spv", "fmHfBound",
                 { (uint32_t)sizeof(FmCase), 16, 16, (uint32_t)sizeof(Bv32Node), 4, 16, 4, 16 },
                 sizeof(FmParams));
        FmParams prm = { NC, stackSize, HF_ROWS, HF_COLS };
        const uint32_t batches = (NC + 31) / 32;
        k.dispatch((batches + 63) / 64, { bC, bV, bP, bN, bS, bO, bW, bD }, prm);
        dev->flush();
        auto pairs = downloadBuffer<U4>(dev, bO, bufCap);
        const uint32_t wc = downloadBuffer<uint32_t>(dev, bW, 1)[0];
        auto dbg = downloadBuffer<F4>(dev, bD, NC);
        for (PxgBufferHandle h : { bC, bV, bP, bN, bS, bO, bW, bD }) dev->destroyBuffer(h);
        return { pairs, wc, dbg };
    };

    const uint32_t CAP = total + 64;
    const auto full = run(CAP, CAP);
    const auto& gOut = std::get<0>(full);
    const uint32_t gWc = std::get<1>(full);
    const auto& gDbg = std::get<2>(full);
    PXG_CHECK_MSG(gWc == total, "hfBound: counter %u != %u", gWc, total);

    // group output rows by cmIdx; each case's run must be contiguous
    // ascending and equal its expected sequence
    std::vector<std::vector<std::pair<uint32_t, U4>>> byCase(NC);
    size_t mm = 0;
    for (uint32_t s = 0; s < std::min(gWc, CAP); ++s)
    {
        if (gOut[s].x >= NC) { ++mm; continue; }
        byCase[gOut[s].x].push_back({ s, gOut[s] });
    }
    size_t mmB = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        const size_t n = caseRange[i].second - caseRange[i].first;
        if (byCase[i].size() != n) { ++mm; continue; }
        if ((gDbg[i].x < 0.0f) != (cases[i].ids.y >= cases[i].ids.w)) ++mm;
        if (gDbg[i].x >= 0.0f && uint32_t(gDbg[i].x) != refR[i].nbPairs) ++mm;
        for (size_t j = 0; j < n; ++j)
        {
            const U4& e = expect[caseRange[i].first + j];
            const U4& g = byCase[i][j].second;
            if (g.x != e.x || g.y != e.y || g.z != e.z || g.w != e.w) ++mm;
            if (j > 0 && byCase[i][j].first != byCase[i][j - 1].first + 1) ++mmB;
        }
    }
    // batch contiguity: cases in one 32-batch share a single reservation
    for (uint32_t b = 0; b * 32 < NC; ++b)
    {
        long prevEnd = -1;
        for (uint32_t l = 0; l < 32 && b * 32 + l < NC; ++l)
        {
            const uint32_t i = b * 32 + l;
            if (byCase[i].empty()) continue;
            if (prevEnd != -1 && (long)byCase[i].front().first != prevEnd) ++mmB;
            prevEnd = (long)byCase[i].back().first + 1;
        }
    }
    size_t dirty = 0;
    for (uint32_t s = gWc; s < CAP; ++s) if (gOut[s].x != PAIRPOISON) ++dirty;
    PXG_CHECK_MSG(mm == 0 && mmB == 0 && dirty == 0,
                  "hfBound: %zu mismatches, %zu contiguity breaks, %zu dirty", mm, mmB, dirty);

    // capacity run: counter unchanged, stores capped at TINY
    const uint32_t TINY = total / 4;
    const auto capped = run(TINY, CAP);
    size_t bad = 0;
    if (std::get<1>(capped) != total) ++bad;
    for (uint32_t s = TINY; s < CAP; ++s) if (std::get<0>(capped)[s].x != PAIRPOISON) ++bad;
    PXG_CHECK_MSG(bad == 0, "hfBound cap: %zu capacity leaks", bad);

    size_t culledPairs = 0;
    for (uint32_t i = 0; i < NC; ++i)
        if (i % 8 == 7) culledPairs += refR[i].nbPairs;
    PXG_CHECK_MSG(culledPairs == 0, "hfBound: off-grid +x cases must be culled (%zu pairs)", culledPairs);
    PXG_CHECK_MSG(offZPairs > 30 && negY > 100 && gatedN > 50 && culledXN > 100,
                  "hfBound coverage thin (offZ=%zu negY=%zu gated=%zu culled=%zu)",
                  offZPairs, negY, gatedN, culledXN);
}

// ============================================================================
// PXG_TEST 4: sbHfBound (sbHeightfieldMidphaseCore, tranche 105)
// ============================================================================
// The softbody twin of hfBound: TET bound (min/max of four world verts,
// fattened by contactDist) and the surface-hint gate `hint &&
// tetrahedronIdx < numTets` (PxU8 truthiness: hint 255 counts, hint 0
// culls).  The element-gate boundary is pinned with tetIdx == numTets
// exactly.  The transform chain, quirk gate, clamps, y-conversion and
// pair emission reuse the fmHfBound machinery (same CPU transcription).
static HfRangeRef refSbHfRange(const FmCase& cs, const std::vector<uint32_t>& smp, std::vector<U4>* pairs)
{
    HfRangeRef r;
    if (!(cs.g.w != 0.0f) || cs.ids.y >= cs.ids.w) return r; // hint + element gate

    const float cd = cs.a.w;
    const V3 v0 = v3(cs.a), v1 = v3(cs.b), v2 = v3(cs.c), vv3 = v3(cs.g);
    V3 bmn = { std::min(std::min(v0.x, v1.x), std::min(v2.x, vv3.x)) - cd,
               std::min(std::min(v0.y, v1.y), std::min(v2.y, vv3.y)) - cd,
               std::min(std::min(v0.z, v1.z), std::min(v2.z, vv3.z)) - cd };
    V3 bmx = { std::max(std::max(v0.x, v1.x), std::max(v2.x, vv3.x)) + cd,
               std::max(std::max(v0.y, v1.y), std::max(v2.y, vv3.y)) + cd,
               std::max(std::max(v0.z, v1.z), std::max(v2.z, vv3.z)) + cd };

    const F4 q = cs.d;
    const F4 qinv = { -q.x, -q.y, -q.z, q.w };
    const V3 centerW = mul(add(bmn, bmx), 0.5f);
    const V3 extentW = mul(sub(bmx, bmn), 0.5f);
    const V3 invP = quatRotateInv(q, { -cs.e.x, -cs.e.y, -cs.e.z });
    const V3 center = add(quatRotateInv(q, centerW), invP);
    const float x = qinv.x, y = qinv.y, z = qinv.z, w = qinv.w;
    const float x2 = x + x, y2 = y + y, z2 = z + z;
    const float xx = x2 * x, yy = y2 * y, zz = z2 * z;
    const float xy = x2 * y, xz = x2 * z, xw = x2 * w;
    const float yz = y2 * z, yw = y2 * w, zw = z2 * w;
    const V3 c0 = { 1.0f - yy - zz, xy + zw, xz - yw };
    const V3 c1 = { xy - zw, 1.0f - xx - zz, yz + xw };
    const V3 c2 = { xz + yw, yz - xw, 1.0f - xx - yy };
    const V3 e0 = mul(c0, extentW.x), e1 = mul(c1, extentW.y), e2 = mul(c2, extentW.z);
    const V3 wext = { std::fabs(e0.x) + std::fabs(e1.x) + std::fabs(e2.x),
                      std::fabs(e0.y) + std::fabs(e1.y) + std::fabs(e2.y),
                      std::fabs(e0.z) + std::fabs(e1.z) + std::fabs(e2.z) };
    V3 lmn = sub(center, wext);
    V3 lmx = add(center, wext);
    lmn = { lmn.x / std::fabs(cs.f.x), lmn.y / cs.f.y, lmn.z / std::fabs(cs.f.z) };
    lmx = { lmx.x / std::fabs(cs.f.x), lmx.y / cs.f.y, lmx.z / std::fabs(cs.f.z) };

    if (!(lmn.x > float(HF_ROWS - 1)) || (lmn.z > float(HF_COLS - 1)) || (lmx.x < 0.0f) || (lmx.z < 0.0f))
    {
        r.minRow = clampI(int(std::floor(lmn.x)), 0, int(HF_ROWS) - 2);
        r.maxRow = clampI(int(std::ceil(lmx.x)), 0, int(HF_ROWS) - 1);
        r.minColumn = clampI(int(std::floor(lmn.z)), 0, int(HF_COLS) - 2);
        r.maxColumn = clampI(int(std::ceil(lmx.z)), 0, int(HF_COLS) - 1);
        r.minyF = float(f2uSatRef(lmn.y));
        r.maxyF = float(f2uSatRef(lmx.y));
        if (2u * (r.maxColumn - r.minColumn) * (r.maxRow - r.minRow) > 0u)
        {
            const uint32_t columnSpan = r.maxColumn - r.minColumn;
            const uint32_t totalNumProcessed = (r.maxRow - r.minRow) * columnSpan * 2u;
            for (uint32_t i = 0; i < totalNumProcessed; ++i)
            {
                const uint32_t index = i / 2u;
                const uint32_t vertexIndex = (r.minRow + index / columnSpan) * HF_COLS + (r.minColumn + index % columnSpan);
                const float h0 = hfHeightRef(smp, vertexIndex);
                const float h1 = hfHeightRef(smp, vertexIndex + 1u);
                const float h2 = hfHeightRef(smp, vertexIndex + HF_COLS);
                const float h3 = hfHeightRef(smp, vertexIndex + HF_COLS + 1u);
                const bool con0 = r.maxyF < h0 && r.maxyF < h1 && r.maxyF < h2 && r.maxyF < h3;
                const bool con1 = r.minyF > h0 && r.minyF > h1 && r.minyF > h2 && r.minyF > h3;
                if (!(con0 || con1))
                {
                    const bool isM1 = (i & 1u) != 0u;
                    const uint32_t material = isM1 ? hfMat1(smp, vertexIndex) : hfMat0(smp, vertexIndex);
                    if (material != 127u)
                    {
                        ++r.nbPairs;
                        if (pairs)
                            pairs->push_back({ cs.ids.x, cs.ids.y,
                                               isM1 ? ((vertexIndex << 1) + 1u) : (vertexIndex << 1), 0 });
                    }
                }
            }
        }
    }
    return r;
}

PXG_TEST(fcMidphase_sbHfBound)
{
    std::mt19937 rng(1050701);
    std::vector<uint32_t> samples(HF_ROWS * HF_COLS);
    for (uint32_t s = 0; s < samples.size(); ++s)
    {
        const int16_t h = int16_t(int(rng() % 101) - 40);
        uint32_t m0 = (s % 13 == 3) ? 127u : (rng() % 5);
        uint32_t m1 = (s % 17 == 5) ? 127u : (rng() % 5);
        if (rng() % 3 == 0) m0 |= 0x80u;
        if (rng() % 4 == 0) m1 |= 0x80u;
        samples[s] = (uint32_t(uint16_t(h))) | (m0 << 16) | (m1 << 24);
    }

    const uint32_t NC = 1024;
    std::vector<FmCase> cases(NC);
    size_t offGridZ = 0, offGridXZ = 0, negY = 0, gatedN = 0, culledXN = 0, hint0N = 0, hint255N = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        FmCase& cs = cases[i];
        std::memset(&cs, 0, sizeof(cs));
        cs.d = exactQuat(rng);
        cs.e = { float(int(rng() % 9)) - 4.0f, float(int(rng() % 9)) - 4.0f, float(int(rng() % 9)) - 4.0f, 0 };
        const float sc[3] = { 0.5f, 1.0f, 2.0f };
        cs.f = { sc[rng() % 3], sc[rng() % 3], sc[rng() % 3], 0 };
        const float cd = 0.25f + float(i % 3) * 0.25f;

        V3 hfP;
        switch (i % 8)
        {
        case 3: hfP = { grid64(rng, 2, 20), grid64(rng, -20, -1), grid64(rng, 2, 14) }; ++negY; break;
        case 5: hfP = { grid64(rng, 2, 20), grid64(rng, -5, 55), float(HF_COLS) + grid64(rng, 1, 4) }; ++offGridZ; break;
        case 6: // off-grid +x AND +z: ONLY the (lmn.z > cols-1) disjunct admits
                // (pins the sb copy of the mis-parenthesised gate's B term:
                // clamped-corner pairs at rows 28-29 x cols 20-21 vanish if
                // that disjunct is dropped)
            hfP = { float(HF_ROWS) + grid64(rng, 1, 3), grid64(rng, -5, 55), float(HF_COLS) + grid64(rng, 1, 3) };
            ++offGridXZ;
            break;
        case 7: // off-grid +x with IN-GRID z: the only culling branch fires
            hfP = { float(HF_ROWS) + grid64(rng, 2, 4), grid64(rng, -5, 55), grid64(rng, 2, 14) };
            cs.f.x = 1.0f + float(rng() % 2);
            ++culledXN;
            break;
        default: hfP = { grid64(rng, 1, 25), grid64(rng, -5, 55), grid64(rng, 1, 18) }; break;
        }
        const V3 scaled = { hfP.x * cs.f.x, hfP.y * cs.f.y, hfP.z * cs.f.z };
        const V3 world = add(quatRotate(cs.d, scaled), v3(cs.e));

        // hint families: 0 culls, 255 counts (PxU8 truthiness), 1 default;
        // element-gate boundary pinned with tetIdx == numTets EXACTLY
        float hint = 1.0f;
        if (i % 16 == 11) { hint = 0.0f; ++hint0N; }
        if (i % 16 == 13) { hint = 255.0f; ++hint255N; }
        cs.ids = { i, (i % 16 == 9) ? 50u : i, 0, (i % 16 == 9) ? 50u : NC + 100u };
        if (i % 16 == 9) ++gatedN;

        // tet verts: world + three exact rotated/scaled hf-space offsets;
        // v3 is the UNIQUE y extreme so a 3-vert-bound mutant shrinks rows
        const V3 b = add(world, quatRotate(cs.d, V3{ grid64(rng, 0.25f, 2) * cs.f.x, 0, 0 }));
        const V3 c = add(world, quatRotate(cs.d, V3{ 0, 0, grid64(rng, 0.25f, 2) * cs.f.z }));
        const V3 dl = add(world, quatRotate(cs.d, V3{ 0, grid64(rng, 1, 3) * cs.f.y, 0 }));
        cs.a = { world.x, world.y, world.z, cd };
        cs.b = { b.x, b.y, b.z, 0 };
        cs.c = { c.x, c.y, c.z, 0 };
        cs.g = { dl.x, dl.y, dl.z, hint };
    }

    std::vector<U4> expect;
    std::vector<HfRangeRef> refR(NC);
    std::vector<std::pair<size_t, size_t>> caseRange(NC);
    for (uint32_t i = 0; i < NC; ++i)
    {
        const size_t b = expect.size();
        refR[i] = refSbHfRange(cases[i], samples, &expect);
        caseRange[i] = { b, expect.size() };
    }
    const uint32_t total = (uint32_t)expect.size();
    PXG_CHECK_MSG(total > 800, "sbHfBound: scene too sparse (%u)", total);
    size_t hint255Pairs = 0, offZPairs = 0, offXZPairs = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        if (i % 16 == 13) hint255Pairs += refR[i].nbPairs;
        if (i % 8 == 5) offZPairs += refR[i].nbPairs;
        if (i % 8 == 6) offXZPairs += refR[i].nbPairs;
    }

    PxgSlangComputeDevice* dev = testDevice();
    auto run = [&](uint32_t stackSize, uint32_t bufCap) -> std::tuple<std::vector<U4>, uint32_t, std::vector<F4>>
    {
        std::vector<U4> outInit(bufCap, { PAIRPOISON, PAIRPOISON, PAIRPOISON, PAIRPOISON });
        std::vector<uint32_t> cnt0(1, 0);
        std::vector<F4> dbgInit(NC, { -7, -7, -7, -7 });
        std::vector<F4> dumF4(4, { 0, 0, 0, 0 });
        std::vector<U4> dumU4(4, { 0, 0, 0, 0 });
        std::vector<Bv32Node> dumNodes(1);
        std::memset(dumNodes.data(), 0, sizeof(Bv32Node));
        PxgBufferHandle bC = createBufferFrom(dev, cases);
        PxgBufferHandle bV = createBufferFrom(dev, dumF4);
        PxgBufferHandle bP = createBufferFrom(dev, dumU4);
        PxgBufferHandle bN = createBufferFrom(dev, dumNodes);
        PxgBufferHandle bS = createBufferFrom(dev, samples);
        PxgBufferHandle bO = createBufferFrom(dev, outInit);
        PxgBufferHandle bW = createBufferFrom(dev, cnt0);
        PxgBufferHandle bD = createBufferFrom(dev, dbgInit);
        Kernel k("fcMidphase.spv", "sbHfBound",
                 { (uint32_t)sizeof(FmCase), 16, 16, (uint32_t)sizeof(Bv32Node), 4, 16, 4, 16 },
                 sizeof(FmParams));
        FmParams prm = { NC, stackSize, HF_ROWS, HF_COLS };
        const uint32_t batches = (NC + 31) / 32;
        k.dispatch((batches + 63) / 64, { bC, bV, bP, bN, bS, bO, bW, bD }, prm);
        dev->flush();
        auto pairs = downloadBuffer<U4>(dev, bO, bufCap);
        const uint32_t wc = downloadBuffer<uint32_t>(dev, bW, 1)[0];
        auto dbg = downloadBuffer<F4>(dev, bD, NC);
        for (PxgBufferHandle h : { bC, bV, bP, bN, bS, bO, bW, bD }) dev->destroyBuffer(h);
        return { pairs, wc, dbg };
    };

    const uint32_t CAP = total + 64;
    const auto full = run(CAP, CAP);
    const auto& gOut = std::get<0>(full);
    const uint32_t gWc = std::get<1>(full);
    const auto& gDbg = std::get<2>(full);
    PXG_CHECK_MSG(gWc == total, "sbHfBound: counter %u != %u", gWc, total);

    std::vector<std::vector<std::pair<uint32_t, U4>>> byCase(NC);
    size_t mm = 0;
    for (uint32_t s = 0; s < std::min(gWc, CAP); ++s)
    {
        if (gOut[s].x >= NC) { ++mm; continue; }
        byCase[gOut[s].x].push_back({ s, gOut[s] });
    }
    size_t mmB = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        const size_t n = caseRange[i].second - caseRange[i].first;
        if (byCase[i].size() != n) { ++mm; continue; }
        const bool gatedOut = !(cases[i].g.w != 0.0f) || cases[i].ids.y >= cases[i].ids.w;
        if ((gDbg[i].x < 0.0f) != gatedOut) ++mm;
        if (gDbg[i].x >= 0.0f && uint32_t(gDbg[i].x) != refR[i].nbPairs) ++mm;
        for (size_t j = 0; j < n; ++j)
        {
            const U4& e = expect[caseRange[i].first + j];
            const U4& g = byCase[i][j].second;
            if (g.x != e.x || g.y != e.y || g.z != e.z || g.w != e.w) ++mm;
            if (j > 0 && byCase[i][j].first != byCase[i][j - 1].first + 1) ++mmB;
        }
    }
    for (uint32_t b = 0; b * 32 < NC; ++b)
    {
        long prevEnd = -1;
        for (uint32_t l = 0; l < 32 && b * 32 + l < NC; ++l)
        {
            const uint32_t i = b * 32 + l;
            if (byCase[i].empty()) continue;
            if (prevEnd != -1 && (long)byCase[i].front().first != prevEnd) ++mmB;
            prevEnd = (long)byCase[i].back().first + 1;
        }
    }
    size_t dirty = 0;
    for (uint32_t s = gWc; s < CAP; ++s) if (gOut[s].x != PAIRPOISON) ++dirty;
    PXG_CHECK_MSG(mm == 0 && mmB == 0 && dirty == 0,
                  "sbHfBound: %zu mismatches, %zu contiguity breaks, %zu dirty", mm, mmB, dirty);

    const uint32_t TINY = total / 4;
    const auto capped = run(TINY, CAP);
    size_t bad = 0;
    if (std::get<1>(capped) != total) ++bad;
    for (uint32_t s = TINY; s < CAP; ++s) if (std::get<0>(capped)[s].x != PAIRPOISON) ++bad;
    // the written prefix is a per-case PREFIX of the expected sequence
    // (batch order is nondeterministic, per-case emission order is not)
    {
        std::vector<size_t> seen(NC, 0);
        for (uint32_t s = 0; s < TINY; ++s)
        {
            const U4 g = std::get<0>(capped)[s];
            if (g.x == PAIRPOISON) continue; // slot cut by the cap mid-batch
            if (g.x >= NC) { ++bad; continue; }
            const size_t j = seen[g.x]++;
            if (caseRange[g.x].first + j >= caseRange[g.x].second) { ++bad; continue; }
            const U4& e = expect[caseRange[g.x].first + j];
            if (g.y != e.y || g.z != e.z || g.w != e.w) ++bad;
        }
    }
    PXG_CHECK_MSG(bad == 0, "sbHfBound cap: %zu capacity leaks", bad);

    size_t culledPairs = 0, hint0Pairs = 0, gatedPairs = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        if (i % 8 == 7) culledPairs += refR[i].nbPairs;
        if (i % 16 == 11) hint0Pairs += refR[i].nbPairs;
        if (i % 16 == 9) gatedPairs += refR[i].nbPairs;
    }
    PXG_CHECK_MSG(culledPairs == 0 && hint0Pairs == 0 && gatedPairs == 0,
                  "sbHfBound: culled families leaked (%zu %zu %zu)", culledPairs, hint0Pairs, gatedPairs);
    PXG_CHECK_MSG(offZPairs > 30 && offXZPairs > 30 && offGridXZ > 100 && negY > 100 &&
                  gatedN > 50 && culledXN > 100 && hint0N > 50 && hint255N > 50 && hint255Pairs > 30,
                  "sbHfBound coverage thin (offZ=%zu offXZ=%zu negY=%zu gated=%zu culled=%zu h0=%zu h255=%zu h255p=%zu)",
                  offZPairs, offXZPairs, negY, gatedN, culledXN, hint0N, hint255N, hint255Pairs);
}

} // namespace
