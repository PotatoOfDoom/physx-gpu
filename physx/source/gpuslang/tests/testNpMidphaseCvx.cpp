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

// Verification of npMidphaseCvx.slang (the convex-vs-trianglemesh
// MidphaseTreeTraverser on the BV32 walk).  The BV32 traversal decisions are
// pure comparisons on uploaded bounds (bit-exact), so the set of TESTED
// triangles per work item is deterministic; the per-triangle decision
// (back-face cull -> world transform -> triangle-box SAT) involves float
// arithmetic, so the C++ reference tracks the minimum slack of every evaluated
// comparison and flags near-boundary triangles (MARG).  Work items with no
// near triangle are compared exactly — (triangleIndex, z) pair sets and the
// per-CM count; items with near triangles are compared by membership (decided
// non-near triangles must appear, undecided non-near must not, near ones may
// go either way; z unchecked).  A capacity-limited run verifies the maxPairs
// drop rule (counter still counts everything; exactly maxPairs distinct valid
// pairs stored).

#include "pxgTestFramework.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
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
struct MpWork
{
    F4 pMin, pMax;
    F4 shapeCoM;
    F4 center, inflatedExtents;
    F4 meshToWorldQ, meshToWorldP;
    F4 scale, scaleRotQ;
    U4 ids;
};
struct MpParams { uint32_t numWorkItems, maxPairs, pad0, pad1; };

struct V3 { float x, y, z; };
static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 add(V3 a, V3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 mulc(V3 a, V3 b) { return { a.x * b.x, a.y * b.y, a.z * b.z }; }
static V3 muls(V3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static F4 f4(V3 v) { return { v.x, v.y, v.z, 0.0f }; }
static V3 v3(F4 f) { return { f.x, f.y, f.z }; }

struct Quat { float x, y, z, w; };
static V3 quatRotate(Quat q, V3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = q.x * vx + q.y * vy + q.z * vz;
    return { vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * dot2,
             vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * dot2,
             vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * dot2 };
}
static V3 quatRotateInv(Quat q, V3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = q.x * vx + q.y * vy + q.z * vz;
    return { vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * dot2,
             vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * dot2,
             vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * dot2 };
}
static V3 vertex2Shape(V3 v, V3 scale, Quat rot) { return quatRotateInv(rot, mulc(quatRotate(rot, v), scale)); }

static const float MARG = 1e-4f; // ~1000x the FMA/rounding divergence at O(1) magnitudes

// Slack tracker: the smallest |lhs-rhs| over every comparison the algorithm
// evaluated; a triangle is "near" a decision boundary when that minimum is
// below MARG (a GPU/CPU float divergence could flip it).
struct Slack
{
    float m = 1e30f;
    void note(float lhs, float rhs) { m = std::min(m, std::fabs(lhs - rhs)); }
};

static bool refPlaneBoxOverlap(V3 normal, float d, V3 maxbox, Slack& sl)
{
    V3 vmin, vmax;
    vmin.x = normal.x > 0 ? -maxbox.x : maxbox.x; vmax.x = normal.x > 0 ? maxbox.x : -maxbox.x;
    vmin.y = normal.y > 0 ? -maxbox.y : maxbox.y; vmax.y = normal.y > 0 ? maxbox.y : -maxbox.y;
    vmin.z = normal.z > 0 ? -maxbox.z : maxbox.z; vmax.z = normal.z > 0 ? maxbox.z : -maxbox.z;
    sl.note(dot3(normal, vmin) + d, 0.0f);
    if (dot3(normal, vmin) + d > 0.0f) return false;
    sl.note(dot3(normal, vmax) + d, 0.0f);
    if (dot3(normal, vmax) + d >= 0.0f) return true;
    return false;
}

static bool refTriBox(V3 boxcenter, V3 extents, V3 tp0, V3 tp1, V3 tp2, Slack& sl)
{
    const V3 v0 = sub(tp0, boxcenter), v1 = sub(tp1, boxcenter), v2 = sub(tp2, boxcenter);

    const V3 vs[3] = { v0, v1, v2 };
    for (int k = 0; k < 3; ++k)
    {
        sl.note(std::fabs(vs[k].x), extents.x);
        sl.note(std::fabs(vs[k].y), extents.y);
        sl.note(std::fabs(vs[k].z), extents.z);
        if (std::fabs(vs[k].x) <= extents.x && std::fabs(vs[k].y) <= extents.y && std::fabs(vs[k].z) <= extents.z)
            return true;
    }

    const V3 e0 = sub(v1, v0), e1 = sub(v2, v1), e2 = sub(v0, v2);
    float mn, mx, rad, p0, p1, p2;

    // P9.86f: AXISTEST_* / FINDMINMAX are intrinsics::selectMin/selectMax (GuIntersectionTriangleBoxRef.h,
    // PxUnixMathIntrinsics.h:116-125): a NaN SECOND operand survives.  std::min/std::max drop it.
    auto selMin = [](float a, float b) { return (a < b) ? a : b; };
    auto selMax = [](float a, float b) { return (a > b) ? a : b; };
    #define AX(pa, pb, radv) \
        mn = selMin(pa, pb); mx = selMax(pa, pb); rad = radv; \
        sl.note(mn, rad); sl.note(mx, -rad); \
        if (mn > rad || mx < -rad) return false;

    float fex = std::fabs(e0.x), fey = std::fabs(e0.y), fez = std::fabs(e0.z);
    p0 = e0.z * v0.y - e0.y * v0.z; p2 = e0.z * v2.y - e0.y * v2.z; AX(p0, p2, fez * extents.y + fey * extents.z);
    p0 = -e0.z * v0.x + e0.x * v0.z; p2 = -e0.z * v2.x + e0.x * v2.z; AX(p0, p2, fez * extents.x + fex * extents.z);
    p1 = e0.y * v1.x - e0.x * v1.y; p2 = e0.y * v2.x - e0.x * v2.y; AX(p1, p2, fey * extents.x + fex * extents.y);

    fex = std::fabs(e1.x); fey = std::fabs(e1.y); fez = std::fabs(e1.z);
    p0 = e1.z * v0.y - e1.y * v0.z; p2 = e1.z * v2.y - e1.y * v2.z; AX(p0, p2, fez * extents.y + fey * extents.z);
    p0 = -e1.z * v0.x + e1.x * v0.z; p2 = -e1.z * v2.x + e1.x * v2.z; AX(p0, p2, fez * extents.x + fex * extents.z);
    p0 = e1.y * v0.x - e1.x * v0.y; p1 = e1.y * v1.x - e1.x * v1.y; AX(p0, p1, fey * extents.x + fex * extents.y);

    fex = std::fabs(e2.x); fey = std::fabs(e2.y); fez = std::fabs(e2.z);
    p0 = e2.z * v0.y - e2.y * v0.z; p1 = e2.z * v1.y - e2.y * v1.z; AX(p0, p1, fez * extents.y + fey * extents.z);
    p0 = -e2.z * v0.x + e2.x * v0.z; p1 = -e2.z * v1.x + e2.x * v1.z; AX(p0, p1, fez * extents.x + fex * extents.z);
    p1 = e2.y * v1.x - e2.x * v1.y; p2 = e2.y * v2.x - e2.x * v2.y; AX(p1, p2, fey * extents.x + fex * extents.y);
    #undef AX

    mn = selMin(selMin(v0.x, v1.x), v2.x); mx = selMax(selMax(v0.x, v1.x), v2.x);
    sl.note(mn, extents.x); sl.note(mx, -extents.x);
    if (mn > extents.x || mx < -extents.x) return false;
    mn = selMin(selMin(v0.y, v1.y), v2.y); mx = selMax(selMax(v0.y, v1.y), v2.y);
    sl.note(mn, extents.y); sl.note(mx, -extents.y);
    if (mn > extents.y || mx < -extents.y) return false;
    mn = selMin(selMin(v0.z, v1.z), v2.z); mx = selMax(selMax(v0.z, v1.z), v2.z);
    sl.note(mn, extents.z); sl.note(mx, -extents.z);
    if (mn > extents.z || mx < -extents.z) return false;

    const V3 normal = cross3(e0, e1);
    const float d = -dot3(normal, v0);
    return refPlaneBoxOverlap(normal, d, extents, sl);
}

// Full per-triangle decision of intersectPrimitiveFullWarp (cull + transform
// + SAT).  Returns intersects; `near` set when any evaluated comparison is
// within MARG of flipping.
static bool refTriDecision(const MpWork& w, V3 tv0, V3 tv1, V3 tv2, bool& near, bool& culled)
{
    Slack sl;
    const V3 c = cross3(sub(tv1, tv0), sub(tv2, tv0));
    const float m = dot3(c, c);
    const V3 n = m > 0 ? muls(c, 1.0f / std::sqrt(m)) : V3{ 0, 0, 0 };
    const float pd = -dot3(tv0, n);
    const float dist = dot3(n, v3(w.shapeCoM)) + pd;
    // An exactly degenerate triangle (m == 0) yields n = 0 and dist = +-0 with
    // NO arithmetic divergence — the >= 0 tie is bit-exact on both sides, so
    // it is NOT a near-boundary case (this deliberately pins the CUDA's >=
    // tie semantics: a port using > would cull it).
    if (m > 0)
        sl.note(dist, 0.0f);
    culled = dist < 0.0f;
    bool result = false;
    if (!culled)
    {
        const Quat mq = { w.meshToWorldQ.x, w.meshToWorldQ.y, w.meshToWorldQ.z, w.meshToWorldQ.w };
        const Quat sq = { w.scaleRotQ.x, w.scaleRotQ.y, w.scaleRotQ.z, w.scaleRotQ.w };
        const V3 sc = v3(w.scale);
        const V3 p0 = add(quatRotate(mq, vertex2Shape(tv0, sc, sq)), v3(w.meshToWorldP));
        const V3 p1 = add(quatRotate(mq, vertex2Shape(tv1, sc, sq)), v3(w.meshToWorldP));
        const V3 p2 = add(quatRotate(mq, vertex2Shape(tv2, sc, sq)), v3(w.meshToWorldP));
        result = refTriBox(v3(w.center), v3(w.inflatedExtents), p0, p1, p2, sl);
    }
    near = sl.m < MARG;
    return result;
}

// ---- scene ------------------------------------------------------------

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
        {
            nodes[nodeIdx].mData[slot] = (b << 7) | ((e - b) << 1) | 1;
        }
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

// C++ transcription of the traversal: which leaf ROUNDS get tested, in visit
// order (exact — the box tests are comparisons on uploaded values).  Each
// round is one intersectPrimitiveFullWarp call: (primStart, nbPrimitives).
static void refTestedRounds(const std::vector<Bv32Node>& nodes, const MpWork& w,
                            std::vector<std::pair<uint32_t, uint32_t>>& rounds)
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
                const bool hit = !(node.mMin[i].x > w.pMax.x || w.pMin.x > node.mMax[i].x ||
                                   node.mMin[i].y > w.pMax.y || w.pMin.y > node.mMax[i].y ||
                                   node.mMin[i].z > w.pMax.z || w.pMin.z > node.mMax[i].z);
                if (!hit) continue;
                const uint32_t data = node.mData[i];
                stack.push_back((data & 1) ? ((cur & 0xFFFFFFC0u) | (i << 1) | 1) : ((data >> 11) << 6));
            }
        }
        if (!stack.empty()) { cur = stack.back(); stack.pop_back(); }
        else hasNode = false;
    }
}

struct QueryRef
{
    std::vector<std::pair<uint32_t, uint32_t>> pairs; // (tri, z) in emission order
    std::set<uint32_t> nearTris;
    std::set<uint32_t> decided; // ref-intersecting tris
    bool anyNear = false;
    size_t culls = 0, rejects = 0;
    uint32_t hitRounds = 0; // leaf rounds contributing >= 1 hit (cross-round z accumulation)
};

struct Scene
{
    std::vector<Bv32Node> nodes;
    std::vector<U4> triIndices;
    std::vector<F4> verts;
    std::vector<MpWork> work;
    std::vector<QueryRef> refs;
    uint32_t totalDecided = 0;
    uint32_t fixtureTri[4];   // Morton-order indices of the degenerate point triangles
    uint32_t fixtureQuery[4]; // queries centered on them (identity transform)
};

static Scene buildScene()
{
    std::mt19937 rng(77017);
    std::uniform_real_distribution<float> uv(-1, 1);
    std::uniform_real_distribution<float> u01(0, 1);

    Scene s;
    const uint32_t NT = 600, NQ = 96;
    const uint32_t NV = NT * 3;
    // LOCAL triangles (3 dedicated verts within a small radius of a random
    // center) so triangle AABBs are small and each query tests few triangles.
    s.verts.resize(NV);
    s.triIndices.resize(NT);
    std::vector<AABB> triBounds(NT);
    for (uint32_t t = 0; t < NT; ++t)
    {
        const V3 c0 = { uv(rng), uv(rng), uv(rng) };
        for (uint32_t k = 0; k < 3; ++k)
            s.verts[t * 3 + k] = { c0.x + 0.15f * uv(rng), c0.y + 0.15f * uv(rng), c0.z + 0.15f * uv(rng), 0 };
        const uint32_t a = t * 3, b = t * 3 + 1, c = t * 3 + 2;
        s.triIndices[t] = { a, b, c, 0 };
        // Triangles 0..3 are exactly degenerate POINT triangles at fixed
        // coordinates: cross == 0 exactly, so the back-face distance is +-0
        // bit-exactly on CPU and GPU — pinning the cull's >= tie semantics.
        if (t < 4)
        {
            static const V3 F[4] = { { 0, 0, 0 }, { 0.5f, 0.5f, 0.5f }, { -0.5f, 0.5f, -0.5f }, { 0.3f, -0.4f, 0.2f } };
            s.verts[a] = f4(F[t]); s.verts[b] = f4(F[t]); s.verts[c] = f4(F[t]);
        }
        AABB bb;
        for (int k = 0; k < 3; ++k)
        {
            const float* pa = &s.verts[a].x; const float* pb = &s.verts[b].x; const float* pc = &s.verts[c].x;
            bb.mn[k] = std::min(pa[k], std::min(pb[k], pc[k]));
            bb.mx[k] = std::max(pa[k], std::max(pb[k], pc[k]));
        }
        triBounds[t] = bb;
    }
    // spatially sort the triangles (Morton order of the AABB centers) so the
    // builder's contiguous chunks are spatially coherent and the tree culls
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
                const float cmid = 0.5f * (triBounds[t].mn[k] + triBounds[t].mx[k]);
                m[k] = (uint32_t)std::min(1023.0f, std::max(0.0f, (cmid + 1.3f) / 2.6f * 1023.0f));
            }
            code[t] = ((uint64_t)expand(m[0]) << 2) | ((uint64_t)expand(m[1]) << 1) | expand(m[2]);
        }
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return code[a] < code[b]; });
        std::vector<U4> ti2(NT);
        std::vector<AABB> tb2(NT);
        for (uint32_t t = 0; t < NT; ++t)
        {
            ti2[t] = s.triIndices[order[t]];
            tb2[t] = triBounds[order[t]];
            if (order[t] < 4) s.fixtureTri[order[t]] = t;
        }
        s.triIndices.swap(ti2);
        triBounds.swap(tb2);
    }
    buildNode(s.nodes, triBounds, 0, NT, rng, 0);

    auto randQuat = [&]() -> Quat {
        std::normal_distribution<float> g(0, 1);
        float x = g(rng), y = g(rng), z = g(rng), w = g(rng);
        const float n = 1.0f / std::sqrt(x * x + y * y + z * z + w * w);
        return { x * n, y * n, z * n, w * n };
    };

    s.work.resize(NQ);
    for (uint32_t q = 0; q < NQ; ++q)
    {
        MpWork& w = s.work[q];
        // queries 88..91 are FIXTURES: identity transforms, box centered on a
        // degenerate point triangle — its emission pins the >= tie semantics.
        if (q >= 88 && q < 92)
        {
            static const V3 F[4] = { { 0, 0, 0 }, { 0.5f, 0.5f, 0.5f }, { -0.5f, 0.5f, -0.5f }, { 0.3f, -0.4f, 0.2f } };
            const V3 c = F[q - 88];
            const V3 e = { 0.12f, 0.12f, 0.12f };
            w.pMin = f4(sub(c, e)); w.pMax = f4(add(c, e));
            w.shapeCoM = { 0, 0, 0, 0 };
            w.meshToWorldQ = { 0, 0, 0, 1 };
            w.meshToWorldP = { 0, 0, 0, 0 };
            w.scale = { 1, 1, 1, 0 };
            w.scaleRotQ = { 0, 0, 0, 1 };
            w.center = f4(c);
            w.inflatedExtents = { 0.17f, 0.17f, 0.17f, 0 };
            w.ids = { q, 1000 + q, 0, 0 };
            s.fixtureQuery[q - 88] = q;
            continue;
        }
        // vertex-space query aabb (misses for every 8th)
        if (q % 8 == 7)
        {
            w.pMin = { 50.0f + q, 50, 50, 0 };
            w.pMax = { 51.0f + q, 51, 51, 0 };
        }
        else
        {
            // small boxes keep the per-query tested-triangle count low, so most
            // queries stay free of near-boundary triangles (exact-z checkable)
            const V3 c = { uv(rng), uv(rng), uv(rng) };
            const V3 e = { 0.06f + 0.18f * u01(rng), 0.06f + 0.18f * u01(rng), 0.06f + 0.18f * u01(rng) };
            w.pMin = f4(sub(c, e)); w.pMax = f4(add(c, e));
        }
        w.shapeCoM = { 1.5f * uv(rng), 1.5f * uv(rng), 1.5f * uv(rng), 0 };
        const Quat mq = randQuat(), sq = randQuat();
        w.meshToWorldQ = { mq.x, mq.y, mq.z, mq.w };
        w.meshToWorldP = { 1.5f * uv(rng), 1.5f * uv(rng), 1.5f * uv(rng), 0 };
        V3 sc = { 0.6f + 0.8f * u01(rng), 0.6f + 0.8f * u01(rng), 0.6f + 0.8f * u01(rng) };
        if (q % 5 == 4) sc.x = -sc.x; // some negative scale
        w.scale = f4(sc);
        w.scaleRotQ = { sq.x, sq.y, sq.z, sq.w };
        // roughly consistent world box: transformed vertex-space box center
        const V3 vc = muls(add(v3(w.pMin), v3(w.pMax)), 0.5f);
        const V3 wc = add(quatRotate(mq, vertex2Shape(vc, sc, sq)), v3(w.meshToWorldP));
        const V3 ve = muls(sub(v3(w.pMax), v3(w.pMin)), 0.5f);
        const float mscale = std::max(std::fabs(sc.x), std::max(std::fabs(sc.y), std::fabs(sc.z)));
        const float inflate = 0.9f + 0.8f * u01(rng);
        w.center = f4(wc);
        w.inflatedExtents = { ve.x * mscale * inflate + 0.05f, ve.y * mscale * inflate + 0.05f, ve.z * mscale * inflate + 0.05f, 0 };
        w.ids = { q, 1000 + q, 0, 0 };
    }

    // references
    s.refs.resize(NQ);
    for (uint32_t q = 0; q < NQ; ++q)
    {
        QueryRef& r = s.refs[q];
        std::vector<std::pair<uint32_t, uint32_t>> rounds;
        refTestedRounds(s.nodes, s.work[q], rounds);
        // z = nbPairsPerCM before the round + offset among the round's decided
        // tris in lane order == the running hit index in flattened visit order.
        uint32_t count = 0;
        for (const auto& rd : rounds)
        {
            uint32_t roundHits = 0;
            for (uint32_t i = 0; i < rd.second; ++i)
            {
                const uint32_t t = rd.first + i;
                const U4& ti = s.triIndices[t];
                bool near = false, culled = false;
                const bool hit = refTriDecision(s.work[q], v3(s.verts[ti.x]), v3(s.verts[ti.y]), v3(s.verts[ti.z]), near, culled);
                if (near) { r.nearTris.insert(t); r.anyNear = true; }
                if (culled) ++r.culls; else if (!hit) ++r.rejects;
                if (hit)
                {
                    r.pairs.push_back({ t, count });
                    r.decided.insert(t);
                    ++count;
                    ++roundHits;
                }
            }
            if (roundHits) ++r.hitRounds;
        }
        s.totalDecided += (uint32_t)r.decided.size();
    }
    return s;
}

struct GpuOut { std::vector<U4> pairs; uint32_t counter; std::vector<uint32_t> perCM; };
static GpuOut runMidphase(const Scene& s, uint32_t maxPairs, uint32_t bufCap)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t NQ = (uint32_t)s.work.size();
    std::vector<U4> outInit(bufCap, { 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu });
    std::vector<uint32_t> cnt(1, 0), percm(NQ, 0xffffffffu);
    PxgBufferHandle bN = createBufferFrom(dev, s.nodes);
    PxgBufferHandle bT = createBufferFrom(dev, s.triIndices);
    PxgBufferHandle bV = createBufferFrom(dev, s.verts);
    PxgBufferHandle bW = createBufferFrom(dev, s.work);
    PxgBufferHandle bO = createBufferFrom(dev, outInit);
    PxgBufferHandle bC = createBufferFrom(dev, cnt);
    PxgBufferHandle bP = createBufferFrom(dev, percm);

    MpParams prm = { NQ, maxPairs, 0, 0 };
    Kernel k("npMidphaseCvx.spv", "ctMidphaseCvx",
             { (uint32_t)sizeof(Bv32Node), 16, 16, (uint32_t)sizeof(MpWork), 16, 4, 4 },
             sizeof(MpParams));
    k.dispatch(NQ, { bN, bT, bV, bW, bO, bC, bP }, prm);
    dev->flush();

    GpuOut g;
    g.pairs = downloadBuffer<U4>(dev, bO, bufCap);
    g.counter = downloadBuffer<uint32_t>(dev, bC, 1)[0];
    g.perCM = downloadBuffer<uint32_t>(dev, bP, NQ);
    dev->destroyBuffer(bN); dev->destroyBuffer(bT); dev->destroyBuffer(bV); dev->destroyBuffer(bW);
    dev->destroyBuffer(bO); dev->destroyBuffer(bC); dev->destroyBuffer(bP);
    return g;
}

PXG_TEST(npMidphaseCvx_traversal)
{
    const Scene s = buildScene();
    const uint32_t NQ = (uint32_t)s.work.size();
    const uint32_t cap = s.totalDecided + 256;
    const GpuOut g = runMidphase(s, cap, cap);

    // counter == sum of per-CM counts (no overflow in this run)
    uint64_t sumPerCM = 0;
    for (uint32_t q = 0; q < NQ; ++q) sumPerCM += g.perCM[q];
    PXG_CHECK_MSG(g.counter == sumPerCM, "npMidphaseCvx: counter %u != sum(perCM) %llu", g.counter, (unsigned long long)sumPerCM);

    // group GPU pairs by cmIdx
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> gp(NQ); // (tri, z)
    size_t badMesh = 0;
    for (uint32_t i = 0; i < g.counter; ++i)
    {
        const U4& p = g.pairs[i];
        PXG_CHECK_MSG(p.x < NQ, "npMidphaseCvx: bad cmIdx %u", p.x);
        if (p.w != 1000 + p.x) ++badMesh;
        gp[p.x].push_back({ p.y, p.z });
    }
    PXG_CHECK_MSG(badMesh == 0, "npMidphaseCvx: %zu wrong shapeMeshIndex payloads", badMesh);

    size_t exactQ = 0, nearQ = 0, mmExact = 0, mmMember = 0, mmCount = 0;
    for (uint32_t q = 0; q < NQ; ++q)
    {
        const QueryRef& r = s.refs[q];
        auto got = gp[q];
        if (!r.anyNear)
        {
            ++exactQ;
            auto want = r.pairs;
            std::sort(got.begin(), got.end());
            std::sort(want.begin(), want.end());
            if (got != want) ++mmExact;
            if (g.perCM[q] != r.decided.size()) ++mmCount;
        }
        else
        {
            ++nearQ;
            std::set<uint32_t> gotTris;
            for (const auto& p : got) gotTris.insert(p.first);
            // decided non-near must appear; non-decided non-near must not
            for (uint32_t t : r.decided)
                if (!r.nearTris.count(t) && !gotTris.count(t)) ++mmMember;
            for (uint32_t t : gotTris)
                if (!r.decided.count(t) && !r.nearTris.count(t)) ++mmMember;
            if (g.perCM[q] != gotTris.size()) ++mmCount; // per-CM must equal this query's emissions
        }
    }
    PXG_CHECK_MSG(mmExact == 0, "npMidphaseCvx: %zu exact-query mismatches", mmExact);
    PXG_CHECK_MSG(mmMember == 0, "npMidphaseCvx: %zu membership violations on near-queries", mmMember);
    PXG_CHECK_MSG(mmCount == 0, "npMidphaseCvx: %zu per-CM count mismatches", mmCount);

    // fixtures: the degenerate point triangles (dist == +0, kept by the >= tie)
    // must be decided, not near, and actually emitted by the GPU
    for (int k = 0; k < 4; ++k)
    {
        const uint32_t fq = s.fixtureQuery[k], ft = s.fixtureTri[k];
        const QueryRef& r = s.refs[fq];
        PXG_CHECK_MSG(r.decided.count(ft) && !r.nearTris.count(ft),
                      "npMidphaseCvx: fixture %d not decided/exact (q=%u t=%u)", k, fq, ft);
        bool emitted = false;
        for (const auto& p : gp[fq]) if (p.first == ft) emitted = true;
        PXG_CHECK_MSG(emitted, "npMidphaseCvx: fixture %d not emitted by GPU (>= tie broken?)", k);
    }

    // coverage (incl. at least one exact query accumulating hits across >= 2
    // leaf rounds, pinning the cross-round nbPairsPerCM/z accumulation)
    size_t culls = 0, rejects = 0, hitQ = 0, multiRound = 0;
    for (uint32_t q = 0; q < NQ; ++q)
    {
        const QueryRef& r = s.refs[q];
        culls += r.culls; rejects += r.rejects;
        if (!r.decided.empty()) ++hitQ;
        if (!r.anyNear && r.hitRounds >= 2) ++multiRound;
    }
    PXG_CHECK_MSG(s.totalDecided > 400 && hitQ > 30 && hitQ < NQ && culls > 300 && rejects > 300 && exactQ > NQ / 3 && multiRound > 0,
                  "npMidphaseCvx: coverage thin (decided=%u hitQ=%zu culls=%zu rejects=%zu exactQ=%zu nearQ=%zu multiRound=%zu)",
                  s.totalDecided, hitQ, culls, rejects, exactQ, nearQ, multiRound);
}

// Pins the EXACT overflow-correction semantics deterministically: a single
// work item = a single warp, so the atomic interleaving is trivial and the
// stored pairs (incl. their z payloads), the counter and the corrected per-CM
// count are all exactly predictable.
PXG_TEST(npMidphaseCvx_overflowExact)
{
    const Scene s = buildScene();
    int pick = -1;
    for (uint32_t q = 0; q < s.work.size(); ++q)
        if (!s.refs[q].anyNear && s.refs[q].decided.size() >= 6 && s.refs[q].hitRounds >= 2) { pick = (int)q; break; }
    PXG_CHECK_MSG(pick >= 0, "npMidphaseCvx overflowExact: no exact multi-round query found");

    Scene s1;
    s1.nodes = s.nodes; s1.triIndices = s.triIndices; s1.verts = s.verts;
    s1.work.push_back(s.work[pick]);
    const uint32_t cap = (uint32_t)s.refs[pick].decided.size() / 2;

    const GpuOut g = runMidphase(s1, cap, cap + 32);

    // reference simulation of the per-round push + correction
    std::vector<std::pair<uint32_t, uint32_t>> rounds;
    refTestedRounds(s.nodes, s.work[pick], rounds);
    std::vector<U4> expect(cap, { 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu });
    uint32_t prev = 0, percm = 0;
    for (const auto& rd : rounds)
    {
        std::vector<uint32_t> hits;
        for (uint32_t i = 0; i < rd.second; ++i)
        {
            const uint32_t t = rd.first + i;
            const U4& ti = s.triIndices[t];
            bool near = false, culled = false;
            if (refTriDecision(s.work[pick], v3(s.verts[ti.x]), v3(s.verts[ti.y]), v3(s.verts[ti.z]), near, culled))
                hits.push_back(t);
        }
        uint32_t v = (uint32_t)hits.size();
        for (uint32_t i = 0; i < v; ++i)
            if (prev + i < cap)
                expect[prev + i] = { (uint32_t)pick, hits[i], percm + i, 1000u + (uint32_t)pick };
        if (v > 0 && v + prev >= cap)
            v = std::max(cap, prev) - prev;
        percm += v;
        prev += (uint32_t)hits.size();
    }

    PXG_CHECK_MSG(g.counter == prev, "npMidphaseCvx overflowExact: counter %u != %u", g.counter, prev);
    PXG_CHECK_MSG(g.perCM[0] == percm, "npMidphaseCvx overflowExact: perCM %u != corrected %u", g.perCM[0], percm);
    size_t mm = 0, dirty = 0;
    for (uint32_t i = 0; i < cap; ++i)
    {
        const U4& a = g.pairs[i]; const U4& b = expect[i];
        if (a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w) ++mm;
    }
    for (uint32_t i = cap; i < cap + 32; ++i)
        if (g.pairs[i].x != 0xffffffffu) ++dirty;
    PXG_CHECK_MSG(mm == 0, "npMidphaseCvx overflowExact: %zu slot mismatches (correction formula or z broken)", mm);
    PXG_CHECK_MSG(dirty == 0, "npMidphaseCvx overflowExact: %zu writes past cap", dirty);
    PXG_CHECK_MSG(percm < s.refs[pick].decided.size(), "npMidphaseCvx overflowExact: correction never engaged (percm=%u)", percm);
}

PXG_TEST(npMidphaseCvx_overflow)
{
    const Scene s = buildScene();
    const uint32_t NQ = (uint32_t)s.work.size();

    // reference full run to learn the GPU's own total
    const uint32_t bigCap = s.totalDecided + 256;
    const GpuOut full = runMidphase(s, bigCap, bigCap);

    const uint32_t cap = full.counter / 3;
    const GpuOut g = runMidphase(s, cap, bigCap);

    // the counter counts every attempted push, dropped or not
    PXG_CHECK_MSG(g.counter == full.counter, "npMidphaseCvx overflow: counter %u != full %u", g.counter, full.counter);

    // slots 0..cap-1 are all written, each a distinct (cm, tri) pair that the
    // full run also produced; slots past cap stay untouched
    std::set<std::pair<uint32_t, uint32_t>> fullSet;
    for (uint32_t i = 0; i < full.counter; ++i) fullSet.insert({ full.pairs[i].x, full.pairs[i].y });
    std::set<std::pair<uint32_t, uint32_t>> gotSet;
    size_t bad = 0, dirty = 0;
    for (uint32_t i = 0; i < cap; ++i)
    {
        const std::pair<uint32_t, uint32_t> p = { g.pairs[i].x, g.pairs[i].y };
        if (!fullSet.count(p)) ++bad;
        gotSet.insert(p);
    }
    for (uint32_t i = cap; i < bigCap; ++i)
        if (g.pairs[i].x != 0xffffffffu) ++dirty;
    PXG_CHECK_MSG(bad == 0, "npMidphaseCvx overflow: %zu stored pairs not in full set", bad);
    PXG_CHECK_MSG(gotSet.size() == cap, "npMidphaseCvx overflow: %zu distinct stored, want %u", gotSet.size(), cap);
    PXG_CHECK_MSG(dirty == 0, "npMidphaseCvx overflow: %zu writes past cap", dirty);

    // per-CM totals never exceed the uncapped totals
    size_t badCM = 0;
    for (uint32_t q = 0; q < NQ; ++q)
        if (g.perCM[q] > full.perCM[q]) ++badCM;
    PXG_CHECK_MSG(badCM == 0, "npMidphaseCvx overflow: %zu per-CM counts exceed uncapped", badCM);
}

} // namespace
