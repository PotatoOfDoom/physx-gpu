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

// Verification of psMeshMid.slang (PsTreeContactGenTraverser - the
// particle-vs-triangle-mesh midphase of particleSystemMeshMidphase.cu).
// One 32-thread group = 32 particles walking one mesh BV32; the reference
// replays the walk (box culling with the ANY-active-particle vertex-space
// clamp test, slack-noted), the per-leaf serialized particle rounds with
// each lane's triangle, the ordered compaction into per-particle 6-slot
// lists and the WORST-CONTACT REPLACEMENT pass, then both finalize paths
// (static one-way lists with the 12 cap / the shared writer).  Slot
// contents within a warp are DETERMINISTIC (ordered ballot offsets and a
// serialized replacement), so per non-skipped particle the slot lists
// compare directly (normal/depth 1e-4); slack taints a particle when any
// of its triangle gates, box tests or replacement depth comparisons sit
// within rounding.  Contact identity in the writer path is the compressed
// particle index (unique per particle).

#include "pxgTestFramework.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };
struct PmMesh { F4 meshP, meshQ, scale, scaleRot, misc; U4 rigid; };
struct Bv32Node
{
    F4 mMin[32];
    F4 mMax[32];
    uint32_t mData[32];
    uint32_t mNbNodes, mDepth, pad0, pad1;
};
struct PsC { F4 normal_pen; uint32_t rLo, rHi, pLo, pHi; };
struct PmParams { uint32_t numParticles, maxContacts, flags, particleSystemId; };

struct V3 { float x, y, z; };
static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 add(V3 a, V3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 muls(V3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static V3 v3(F4 f) { return { f.x, f.y, f.z }; }

static const float MAXF = 3.4028235e38f;
static const uint32_t MAXMESH = 6, MAXPART = 12;

struct Slack { bool near = false; void noteSc(float l, float r, float scale) { if (std::fabs(l - r) < 1e-5f * scale) near = true; } };
static float l1n(V3 v) { return std::fabs(v.x) + std::fabs(v.y) + std::fabs(v.z); }

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
static V3 vertex2Shape(V3 v, V3 sc, F4 rot)
{
    V3 t = quatRotate(rot, v);
    t = { t.x * sc.x, t.y * sc.y, t.z * sc.z };
    return quatRotateInv(rot, t);
}
static V3 shape2Vertex(V3 v, V3 sc, F4 rot)
{
    V3 t = quatRotate(rot, v);
    t = { t.x / sc.x, t.y / sc.y, t.z / sc.z };
    return quatRotateInv(rot, t);
}

static V3 refTriClosest(V3 p, V3 a, V3 b, V3 c)
{
    const V3 ab = sub(b, a), ac = sub(c, a), ap = sub(p, a);
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

// slack-noted particleTriangleTest mirror
static bool refPTT(V3& normal, float& distance, V3 pp, V3 cvp, float cvr,
                   V3 a, V3 b, V3 c, bool ccd, Slack& sl)
{
    const V3 cn = cross3(sub(b, a), sub(c, a));
    const float m = dot3(cn, cn);
    const float blow = l1n(sub(b, a)) * l1n(sub(c, a)) / (std::sqrt(m) + 1e-30f);
    const V3 tn = m > 0 ? muls(cn, 1.0f / std::sqrt(m)) : V3{ 0, 0, 0 };
    const float pd = dot3(sub(cvp, a), tn);
    sl.noteSc(pd, cvr, (l1n(cvp) + l1n(a)) * (1.0f + blow) + 1.0f);
    if (pd < cvr)
    {
        V3 closest = refTriClosest(cvp, a, b, c);
        V3 dv0 = sub(cvp, closest);
        float distSq = dot3(dv0, dv0);
        sl.noteSc(distSq, cvr * cvr, l1n(dv0) * (l1n(cvp) + l1n(closest)) + distSq + cvr * cvr + 1e-3f);
        if (distSq <= cvr * cvr)
        {
            const float ppd = dot3(sub(pp, a), tn);
            sl.noteSc(ppd, 0.0f, (l1n(pp) + l1n(a)) * (1.0f + blow) + 1e-3f);
            if (ppd > 0.0f)
            {
                if (ccd)
                {
                    closest = refTriClosest(pp, a, b, c);
                    const V3 dv = sub(pp, closest);
                    distSq = dot3(dv, dv);
                }
                const V3 dv = sub(pp, closest);
                if (distSq != 0.0f) sl.noteSc(std::sqrt(distSq), 0.0f, 1000.0f);
                normal = (distSq > 0.0f) ? muls(dv, 1.0f / std::sqrt(distSq)) : tn;
                distance = std::sqrt(distSq);
                return true;
            }
        }
    }
    return false;
}

// ---- builder (Morton + BV32, as in testNpTetTriMid) -----------------------
struct AABB { float mn[3], mx[3]; };
static AABB merge(const AABB& a, const AABB& b)
{
    AABB r;
    for (int k = 0; k < 3; ++k) { r.mn[k] = std::min(a.mn[k], b.mn[k]); r.mx[k] = std::max(a.mx[k], b.mx[k]); }
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
    for (size_t cc = 0; cc + 1 < cuts.size(); ++cc, ++slot)
    {
        const uint32_t b = cuts[cc], e = cuts[cc + 1];
        AABB bounds = prims[b];
        for (uint32_t j = b + 1; j < e; ++j) bounds = merge(bounds, prims[j]);
        if (e - b <= 12 || (depth >= 4 && e - b <= 32))
            nodes[nodeIdx].mData[slot] = (b << 7) | ((e - b) << 1) | 1;
        else
            nodes[nodeIdx].mData[slot] = buildNode(nodes, prims, b, e, rng, depth + 1) << 11;
        nodes[nodeIdx].mMin[slot] = { bounds.mn[0], bounds.mn[1], bounds.mn[2], 0.0f };
        nodes[nodeIdx].mMax[slot] = { bounds.mx[0], bounds.mx[1], bounds.mx[2], 0.0f };
    }
    nodes[nodeIdx].mNbNodes = slot;
    return nodeIdx;
}

PXG_TEST(psMeshMid_traverser)
{
    std::mt19937 rng(92003);
    std::uniform_real_distribution<float> u01(0, 1);
    std::uniform_real_distribution<float> uv(-1, 1);

    const uint32_t NR = 250, NP = 97, SYS = 5; // non-multiple of 32: partial last group
    const bool CCD = true, TGS = true, STATIC_RIGID = true;

    // mesh (VERTEX space) + Morton + BV32
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
            uint32_t mm[3];
            for (int k = 0; k < 3; ++k)
            {
                const float c = 0.5f * (bounds[t].mn[k] + bounds[t].mx[k]);
                mm[k] = (uint32_t)std::min(1023.0f, std::max(0.0f, (c + 1.5f) / 3.0f * 1023.0f));
            }
            code[t] = ((uint64_t)expand(mm[0]) << 2) | ((uint64_t)expand(mm[1]) << 1) | expand(mm[2]);
        }
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return code[a] < code[b]; });
        std::vector<U4> t2(NR); std::vector<AABB> b2(NR);
        for (uint32_t t = 0; t < NR; ++t) { t2[t] = tris[order[t]]; b2[t] = bounds[order[t]]; }
        tris.swap(t2); bounds.swap(b2);
    }
    std::vector<Bv32Node> nodes;
    buildNode(nodes, bounds, 0, NR, rng, 0);

    // mesh pose/scale: mild anisotropic NEGATIVE-determinant scale exercises
    // flipNormal; rotation kept identity for scale frame
    PmMesh mesh;
    mesh.meshP = { 0.3f, -0.2f, 0.1f, 0 };
    {
        F4 q = { 0.2f, -0.1f, 0.3f, 1.0f };
        const float m = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
        mesh.meshQ = { q.x / m, q.y / m, q.z / m, q.w / m };
    }
    mesh.scale = { 1.1f, -0.9f, 1.05f, 0.22f }; // w = contactDist; det < 0
    mesh.scaleRot = { 0, 0, 0, 1 };
    mesh.misc = { 0.03f, 0, 0, 0 }; // restDist
    mesh.rigid = { 0x77u, 9u, 0, 0 };
    const float contactDist = mesh.scale.w, restDist = mesh.misc.x;
    const bool flipNormal = (mesh.scale.x * mesh.scale.y * mesh.scale.z) < 0.0f;

    // densest mesh spot (vertex space): anchor for the overflow fixtures
    uint32_t denseTri = 0;
    {
        size_t bestCnt = 0;
        for (uint32_t t = 0; t < NR; ++t)
        {
            V3 tc = { 0, 0, 0 };
            for (int k = 0; k < 3; ++k) tc = add(tc, v3(verts[tris[t].x + k]));
            tc = muls(tc, 1.0f / 3.0f);
            size_t cnt = 0;
            for (uint32_t u = 0; u < NR; ++u)
            {
                V3 uc = { 0, 0, 0 };
                for (int k = 0; k < 3; ++k) uc = add(uc, v3(verts[tris[u].x + k]));
                uc = muls(uc, 1.0f / 3.0f);
                const V3 d = sub(uc, tc);
                if (dot3(d, d) < 0.35f * 0.35f) ++cnt;
            }
            if (cnt > bestCnt) { bestCnt = cnt; denseTri = t; }
        }
    }

    // particles (world): placed near shape-space triangle positions
    const uint32_t NPPAD = ((NP + 31) / 32) * 32;
    std::vector<F4> curPos(NPPAD, { 100, 100, 100, 1 }), predPos(NPPAD, { 100, 100, 100, 1 });
    for (uint32_t p = 0; p < NP; ++p)
    {
        const uint32_t t = (p % 5 == 2) ? denseTri : (p * 29) % NR;
        V3 tc = { 0, 0, 0 };
        for (int k = 0; k < 3; ++k) tc = add(tc, v3(verts[tris[t].x + k]));
        tc = muls(tc, 1.0f / 3.0f);
        // vertex -> shape -> world
        const V3 sp = vertex2Shape(tc, v3(mesh.scale), mesh.scaleRot);
        const V3 wp = add(quatRotate(mesh.meshQ, sp), v3(mesh.meshP));
        V3 dir = { uv(rng), uv(rng), uv(rng) };
        const float m = std::sqrt(std::max(dot3(dir, dir), 1e-3f));
        dir = muls(dir, 1.0f / m);
        // p%12==7: dense fixtures sit ON the surface amid a Morton-adjacent
        // cluster so more than MaxStaticContactsPerMesh triangles qualify
        const float off = (p % 5 == 2) ? 0.02f + 0.13f * u01(rng)
                          : (p % 4 == 0) ? 0.9f * u01(rng) : 0.25f * u01(rng);
        const V3 c = add(wp, muls(dir, off));
        curPos[p] = { c.x, c.y, c.z, 1 };
        predPos[p] = { c.x + 0.15f * uv(rng), c.y + 0.15f * uv(rng), c.z + 0.15f * uv(rng), 1 };
    }

    // ---- reference ----
    struct PRef { std::vector<V3> n; std::vector<float> d; uint32_t contactCount = 0; uint32_t overflowRounds = 0, rejects = 0; };
    std::vector<PRef> pref(NP);
    std::vector<uint8_t> taint(NP, 0);

    const uint32_t numGroups = (NP + 31) / 32;
    for (uint32_t g = 0; g < numGroups; ++g)
    {
        // per-lane state
        V3 pShape[32], cvShape[32], cvVert[32];
        float cvR[32];
        Slack laneSl[32];
        bool valid[32];
        for (uint32_t l = 0; l < 32; ++l)
        {
            const uint32_t gp = g * 32 + l;
            valid[l] = gp < NP;
            if (!valid[l]) { cvR[l] = 0; cvVert[l] = { 0, 0, 0 }; continue; }
            const V3 cur = v3(curPos[gp]);
            pShape[l] = quatRotateInv(mesh.meshQ, sub(cur, v3(mesh.meshP)));
            if (CCD)
            {
                const V3 off = muls(sub(v3(predPos[gp]), cur), (TGS ? 2.0f : 1.0f) * 0.5f);
                const V3 cvw = add(cur, off);
                cvR[l] = std::sqrt(dot3(off, off)) + contactDist;
                cvShape[l] = quatRotateInv(mesh.meshQ, sub(cvw, v3(mesh.meshP)));
            }
            else
            {
                cvShape[l] = pShape[l];
                cvR[l] = contactDist;
            }
            cvVert[l] = shape2Vertex(cvShape[l], v3(mesh.scale), mesh.scaleRot);
        }

        // walk (box culling: ANY particle; near events taint the WHOLE group
        // via all lanes since traversal visits are shared)
        bool groupNear = false;
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
                        bool any = false;
                        for (uint32_t l = 0; l < 32; ++l)
                        {
                            if (!valid[l]) continue; // not in the active mask
                            const V3 mn = v3(node.mMin[i]), mx = v3(node.mMax[i]);
                            V3 cl = { std::max(mn.x, std::min(mx.x, cvVert[l].x)),
                                      std::max(mn.y, std::min(mx.y, cvVert[l].y)),
                                      std::max(mn.z, std::min(mx.z, cvVert[l].z)) };
                            const V3 dvv = sub(cl, cvVert[l]);
                            const V3 dv = vertex2Shape(dvv, v3(mesh.scale), mesh.scaleRot);
                            const float d2 = dot3(dv, dv);
                            Slack bs;
                            bs.noteSc(d2, cvR[l] * cvR[l], l1n(dv) * (l1n(cvVert[l]) + 3.0f) + d2 + cvR[l] * cvR[l]);
                            if (bs.near) groupNear = true;
                            if (d2 < cvR[l] * cvR[l]) any = true;
                        }
                        const uint32_t data = node.mData[i];
                        if (!any) continue;
                        stack.push_back((data & 1) ? ((cur & 0xFFFFFFC0u) | (i << 1) | 1) : ((data >> 11) << 6));
                    }
                }
                if (!stack.empty()) { cur = stack.back(); stack.pop_back(); }
                else hasNode = false;
            }
        }

        // leaves: serialized particle rounds
        for (const auto& rd : rounds)
        {
            for (uint32_t p = 0; p < 32; ++p) // particle order = lowest lane first
            {
                if (!valid[p]) continue;
                // each lane evaluates its triangle vs particle p
                bool inter[32] = {};
                V3 nrm[32];
                float dst[32];
                for (uint32_t l = 0; l < rd.second && l < 32; ++l)
                {
                    const uint32_t t = rd.first + l;
                    V3 a = vertex2Shape(v3(verts[tris[t].x]), v3(mesh.scale), mesh.scaleRot);
                    V3 b = vertex2Shape(v3(verts[tris[t].y]), v3(mesh.scale), mesh.scaleRot);
                    V3 c = vertex2Shape(v3(verts[tris[t].z]), v3(mesh.scale), mesh.scaleRot);
                    if (flipNormal) std::swap(b, c);
                    V3 nss;
                    float dd;
                    if (refPTT(nss, dd, pShape[p], cvShape[p], cvR[p], a, b, c, CCD, laneSl[p]))
                    {
                        inter[l] = true;
                        nrm[l] = quatRotate(mesh.meshQ, nss);
                        dst[l] = dd;
                    }
                }
                PRef& pr = pref[g * 32 + p];
                const uint32_t startCnt = pr.contactCount;
                uint32_t valid = 0;
                bool anyFail = false;
                for (uint32_t l = 0; l < 32; ++l)
                {
                    if (!inter[l]) continue;
                    const uint32_t index = startCnt + valid;
                    ++valid;
                    if (index < MAXMESH)
                    {
                        if (pr.n.size() <= index) { pr.n.resize(index + 1); pr.d.resize(index + 1); }
                        pr.n[index] = nrm[l];
                        pr.d[index] = dst[l];
                    }
                    else anyFail = true;
                }
                pr.contactCount += valid;
                if (anyFail)
                {
                    // replacement pass, fail lanes in order
                    float outSep[MAXMESH];
                    for (uint32_t sIdx = 0; sIdx < MAXMESH; ++sIdx) outSep[sIdx] = pr.d[sIdx];
                    uint32_t valid2 = 0;
                    for (uint32_t l = 0; l < 32; ++l)
                    {
                        if (!inter[l]) continue;
                        const uint32_t index = startCnt + valid2;
                        ++valid2;
                        if (index < MAXMESH) continue;
                        // failed writer: find current worst slot
                        uint32_t best = 0;
                        float bv = -MAXF;
                        for (uint32_t sIdx = 0; sIdx < MAXMESH; ++sIdx)
                            if (outSep[sIdx] > bv) { bv = outSep[sIdx]; best = sIdx; }
                        laneSl[p].noteSc(dst[l], bv, 8.0f);
                        // ties among slot depths also decide 'best'
                        for (uint32_t sIdx = 0; sIdx < MAXMESH; ++sIdx)
                            if (sIdx != best) laneSl[p].noteSc(outSep[sIdx], bv, 8.0f);
                        if (dst[l] < bv)
                        {
                            pr.n[best] = nrm[l];
                            pr.d[best] = dst[l];
                            outSep[best] = dst[l];
                        }
                        else
                            pr.rejects++;
                    }
                    pr.contactCount = MAXMESH;
                    pr.overflowRounds++;
                }
            }
        }
        for (uint32_t l = 0; l < 32 && g * 32 + l < NP; ++l)
            taint[g * 32 + l] = (laneSl[l].near || groupNear) ? 1 : 0;
    }

    // expected one-way lists (static path; counts capped at 12 per particle,
    // single mesh -> startIndex 0)
    // ---- GPU ----
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t CAP = 2048;
    std::vector<PmMesh> meshV(1, mesh);
    std::vector<F4> infoInit(NP * MAXPART, { -9, -9, -9, -9 });
    std::vector<uint32_t> nodeInit(NP * MAXPART * 2, 0xcccccccc), cnt0(NP, 0), wcnt(1, 0);
    std::vector<PsC> ctInit(CAP);
    std::vector<uint32_t> z2(CAP * 2, 0), z1(CAP, 0);
    PxgBufferHandle bM = createBufferFrom(dev, meshV);
    PxgBufferHandle bN = createBufferFrom(dev, nodes);
    PxgBufferHandle bT = createBufferFrom(dev, tris);
    PxgBufferHandle bV = createBufferFrom(dev, verts);
    PxgBufferHandle bCur = createBufferFrom(dev, curPos);
    PxgBufferHandle bPred = createBufferFrom(dev, predPos);
    PxgBufferHandle bI = createBufferFrom(dev, infoInit);
    PxgBufferHandle bNi = createBufferFrom(dev, nodeInit);
    PxgBufferHandle bC = createBufferFrom(dev, cnt0);
    PxgBufferHandle bW = createBufferFrom(dev, ctInit);
    PxgBufferHandle bBR = createBufferFrom(dev, z2);
    PxgBufferHandle bTBR = createBufferFrom(dev, z1);
    PxgBufferHandle bIBR = createBufferFrom(dev, z1);
    PxgBufferHandle bSP = createBufferFrom(dev, z2);
    PxgBufferHandle bTBP = createBufferFrom(dev, z1);
    PxgBufferHandle bIBP = createBufferFrom(dev, z1);
    PxgBufferHandle bWc = createBufferFrom(dev, wcnt);

    const uint32_t flags = (CCD ? 1u : 0) | (TGS ? 2u : 0) | (STATIC_RIGID ? 8u : 0);
    PmParams prm = { NP, CAP, flags, SYS };
    const std::vector<uint32_t> strides = { (uint32_t)sizeof(PmMesh), (uint32_t)sizeof(Bv32Node), 16, 16, 16, 16,
                                            16, 8, 4, (uint32_t)sizeof(PsC), 8, 4, 4, 8, 4, 4, 4 };
    Kernel k("psMeshMid.spv", "ctPsMeshMid", strides, sizeof(PmParams));
    k.dispatch((NP + 31) / 32, { bM, bN, bT, bV, bCur, bPred, bI, bNi, bC, bW, bBR, bTBR, bIBR, bSP, bTBP, bIBP, bWc }, prm);
    dev->flush();

    const auto gI = downloadBuffer<F4>(dev, bI, NP * MAXPART);
    const auto gNi = downloadBuffer<uint32_t>(dev, bNi, NP * MAXPART * 2);
    const auto gC = downloadBuffer<uint32_t>(dev, bC, NP);
    const uint32_t gWCount = downloadBuffer<uint32_t>(dev, bWc, 1)[0];
    for (PxgBufferHandle h : { bM, bN, bT, bV, bCur, bPred, bI, bNi, bC, bW, bBR, bTBR, bIBR, bSP, bTBP, bIBP, bWc })
        dev->destroyBuffer(h);

    size_t mm = 0, checked = 0, skipped = 0, contactsP = 0, replaced = 0, rejectsN = 0;
    for (uint32_t p = 0; p < NP; ++p)
    {
        if (taint[p]) { ++skipped; continue; }
        ++checked;
        const PRef& pr = pref[p];
        // static path: counts, then the stored slots (single mesh -> start 0)
        if (gC[p] != pr.contactCount) { ++mm; continue; }
        if (pr.contactCount) ++contactsP;
        if (pr.overflowRounds) ++replaced;
        rejectsN += pr.rejects;
        const uint32_t stored = std::min<uint32_t>(std::min<uint32_t>(pr.contactCount, MAXMESH), MAXPART);
        bool ok = true;
        for (uint32_t s = 0; s < stored && ok; ++s)
        {
            const uint32_t oi = p + s * NP;
            ok = almostEqual(gI[oi].w, pr.d[s] - restDist, 1e-4f, 1e-4f) &&
                 almostEqual(gI[oi].x, pr.n[s].x, 1e-4f, 1e-4f) &&
                 almostEqual(gI[oi].y, pr.n[s].y, 1e-4f, 1e-4f) &&
                 almostEqual(gI[oi].z, pr.n[s].z, 1e-4f, 1e-4f) &&
                 gNi[oi * 2] == 0x77u && gNi[oi * 2 + 1] == 9u;
        }
        // untouched sentinels beyond
        for (uint32_t s = stored; s < MAXPART && ok; ++s)
            ok = gI[p + s * NP].x == -9.0f;
        if (!ok) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "psMeshMid: %zu mismatches", mm);
    // static rigid + per-mesh cap 6 < per-particle cap 12: nothing remains
    // for the writer
    PXG_CHECK_MSG(gWCount == 0, "psMeshMid: writer path unexpectedly used (%u)", gWCount);
    PXG_CHECK_MSG(checked > NP * 3 / 4, "psMeshMid: too many tainted (%zu)", skipped);
    (void)rejectsN; // the deterministic micro-scene below pins the reject branch
    PXG_CHECK_MSG(contactsP > 30 && replaced >= 1,
                  "psMeshMid: coverage thin (cp=%zu repl=%zu)", contactsP, replaced);

    // ---- deterministic replacement micro-scene: one leaf, one particle ----
    // Triangles at plane distances 0.05..0.10 (slots), then 0.30 (MUST be
    // REJECTED by the depth < worst gate) and 0.07 (replaces the 0.10 slot).
    // Identity pose, unit scale, no CCD (also exercises the non-CCD volume).
    {
        const float dep[8] = { 0.05f, 0.06f, 0.08f, 0.09f, 0.095f, 0.10f, 0.07f, 0.30f };
        std::vector<F4> mv(24);
        std::vector<U4> mt(8);
        for (uint32_t t = 0; t < 8; ++t)
        {
            // big triangle in the plane z = -dep[t], normal +z (towards the particle)
            mv[t * 3 + 0] = { -2.0f, -2.0f, -dep[t], 0 };
            mv[t * 3 + 1] = { 3.0f, -2.0f, -dep[t], 0 };
            mv[t * 3 + 2] = { -2.0f, 3.0f, -dep[t], 0 };
            mt[t] = { t * 3, t * 3 + 1, t * 3 + 2, 0 };
        }
        std::vector<Bv32Node> mn(1);
        mn[0].mNbNodes = 1;
        mn[0].mData[0] = (0u << 7) | (8u << 1) | 1u; // one leaf, prims [0,8)
        mn[0].mMin[0] = { -2, -2, -0.4f, 0 };
        mn[0].mMax[0] = { 3, 3, 0, 0 };
        PmMesh mm2;
        mm2.meshP = { 0, 0, 0, 0 };
        mm2.meshQ = { 0, 0, 0, 1 };
        mm2.scale = { 1, 1, 1, 0.5f };
        mm2.scaleRot = { 0, 0, 0, 1 };
        mm2.misc = { 0.01f, 0, 0, 0 };
        mm2.rigid = { 0x55u, 3u, 0, 0 };
        std::vector<PmMesh> mmV(1, mm2);
        std::vector<F4> mcur(32, { 50, 50, 50, 1 }), mpred(32, { 50, 50, 50, 1 });
        mcur[0] = { 0, 0, 0, 1 }; // particle 0 at the origin
        std::vector<F4> mInfo(1 * MAXPART, { -9, -9, -9, -9 });
        std::vector<uint32_t> mNode(1 * MAXPART * 2, 0xcccccccc), mCnt(1, 0), mW(1, 0);
        std::vector<PsC> mCt(64);
        std::vector<uint32_t> mz2(64 * 2, 0), mz1(64, 0);
        PxgBufferHandle qM = createBufferFrom(dev, mmV);
        PxgBufferHandle qN = createBufferFrom(dev, mn);
        PxgBufferHandle qT = createBufferFrom(dev, mt);
        PxgBufferHandle qV = createBufferFrom(dev, mv);
        PxgBufferHandle qCur = createBufferFrom(dev, mcur);
        PxgBufferHandle qPred = createBufferFrom(dev, mpred);
        PxgBufferHandle qI = createBufferFrom(dev, mInfo);
        PxgBufferHandle qNi = createBufferFrom(dev, mNode);
        PxgBufferHandle qC = createBufferFrom(dev, mCnt);
        PxgBufferHandle qW = createBufferFrom(dev, mCt);
        PxgBufferHandle qBR = createBufferFrom(dev, mz2);
        PxgBufferHandle qTBR = createBufferFrom(dev, mz1);
        PxgBufferHandle qIBR = createBufferFrom(dev, mz1);
        PxgBufferHandle qSP = createBufferFrom(dev, mz2);
        PxgBufferHandle qTBP = createBufferFrom(dev, mz1);
        PxgBufferHandle qIBP = createBufferFrom(dev, mz1);
        PxgBufferHandle qWc = createBufferFrom(dev, mW);
        PmParams mprm = { 1, 64, 8u /* static, no CCD */, SYS };
        Kernel kq("psMeshMid.spv", "ctPsMeshMid", strides, sizeof(PmParams));
        kq.dispatch(1, { qM, qN, qT, qV, qCur, qPred, qI, qNi, qC, qW, qBR, qTBR, qIBR, qSP, qTBP, qIBP, qWc }, mprm);
        dev->flush();
        const auto qGI = downloadBuffer<F4>(dev, qI, MAXPART);
        const auto qGC = downloadBuffer<uint32_t>(dev, qC, 1);
        // expected: slots 0-4 = dep[0..4], slot 5 = 0.07 (replaced the 0.10);
        // the trailing 0.30 REJECTED (a flipped reject gate would replace the
        // 0.095 slot with it); count saturated at 6
        const float expDep[6] = { 0.05f, 0.06f, 0.08f, 0.09f, 0.095f, 0.07f };
        size_t mmq = 0;
        if (qGC[0] != 6) ++mmq;
        for (uint32_t sIdx = 0; sIdx < 6; ++sIdx)
            if (!almostEqual(qGI[sIdx].w, expDep[sIdx] - 0.01f, 1e-4f, 1e-4f)) ++mmq;
        PXG_CHECK_MSG(mmq == 0, "psMeshMid: micro replacement scene wrong (%zu)", mmq);
        for (PxgBufferHandle h : { qM, qN, qT, qV, qCur, qPred, qI, qNi, qC, qW, qBR, qTBR, qIBR, qSP, qTBP, qIBP, qWc })
            dev->destroyBuffer(h);
    }

    // ---- dispatch 2: SAME one-way buffers again (static path accumulation:
    // nonzero startIndex, the 12 cap binding when 2*count > 12) ----
    // ---- dispatch 3: dynamic rigid -> EVERYTHING flows through the writer ----
    {
        std::vector<F4> infoInit2(NPPAD * MAXPART, { -9, -9, -9, -9 });
        std::vector<uint32_t> nodeInit2(NPPAD * MAXPART * 2, 0xcccccccc), cnt2(NPPAD, 0), wcnt2(1, 0);
        std::vector<PsC> ctInit2(CAP);
        std::vector<uint32_t> zz2(CAP * 2, 0xabababab), zz1(CAP, 0xabababab);
        PxgBufferHandle dM = createBufferFrom(dev, meshV);
        PxgBufferHandle dN = createBufferFrom(dev, nodes);
        PxgBufferHandle dT = createBufferFrom(dev, tris);
        PxgBufferHandle dV = createBufferFrom(dev, verts);
        PxgBufferHandle dCur = createBufferFrom(dev, curPos);
        PxgBufferHandle dPred = createBufferFrom(dev, predPos);
        PxgBufferHandle dI = createBufferFrom(dev, infoInit2);
        PxgBufferHandle dNi = createBufferFrom(dev, nodeInit2);
        PxgBufferHandle dC = createBufferFrom(dev, cnt2);
        PxgBufferHandle dW = createBufferFrom(dev, ctInit2);
        PxgBufferHandle dBR = createBufferFrom(dev, zz2);
        PxgBufferHandle dTBR = createBufferFrom(dev, zz1);
        PxgBufferHandle dIBR = createBufferFrom(dev, zz1);
        PxgBufferHandle dSP = createBufferFrom(dev, zz2);
        PxgBufferHandle dTBP = createBufferFrom(dev, zz1);
        PxgBufferHandle dIBP = createBufferFrom(dev, zz1);
        PxgBufferHandle dWc = createBufferFrom(dev, wcnt2);
        Kernel k2("psMeshMid.spv", "ctPsMeshMid", strides, sizeof(PmParams));
        // twice static (accumulation)
        k2.dispatch((NP + 31) / 32, { dM, dN, dT, dV, dCur, dPred, dI, dNi, dC, dW, dBR, dTBR, dIBR, dSP, dTBP, dIBP, dWc }, prm);
        k2.dispatch((NP + 31) / 32, { dM, dN, dT, dV, dCur, dPred, dI, dNi, dC, dW, dBR, dTBR, dIBR, dSP, dTBP, dIBP, dWc }, prm);
        dev->flush();
        const auto dGI = downloadBuffer<F4>(dev, dI, NPPAD * MAXPART);
        const auto dGC = downloadBuffer<uint32_t>(dev, dC, NPPAD);
        size_t mm2 = 0;
        for (uint32_t p = 0; p < NP; ++p)
        {
            if (taint[p]) continue;
            const PRef& pr = pref[p];
            if (dGC[p] != 2 * pr.contactCount) { ++mm2; continue; }
            const uint32_t storedEach = std::min<uint32_t>(pr.contactCount, MAXMESH);
            const uint32_t total = std::min<uint32_t>(2 * storedEach, MAXPART);
            bool ok = true;
            for (uint32_t sIdx = 0; sIdx < total && ok; ++sIdx)
            {
                const uint32_t src = sIdx < storedEach ? sIdx : sIdx - storedEach;
                const uint32_t oi = p + sIdx * NPPAD;
                // NOTE: strided by numParticles = NP in the shader
                const uint32_t oiNP = p + sIdx * NP;
                ok = almostEqual(dGI[oiNP].w, pr.d[src] - restDist, 1e-4f, 1e-4f);
                (void)oi;
            }
            if (!ok) ++mm2;
        }
        PXG_CHECK_MSG(mm2 == 0, "psMeshMid: %zu accumulation mismatches", mm2);
        // dynamic rigid: writer path
        std::vector<uint32_t> cnt3(NPPAD, 0), wcnt3(1, 0);
        PxgBufferHandle eC = createBufferFrom(dev, cnt3);
        PxgBufferHandle eW = createBufferFrom(dev, ctInit2);
        PxgBufferHandle eWc = createBufferFrom(dev, wcnt3);
        PmParams prmDyn = { NP, CAP, (CCD ? 1u : 0) | (TGS ? 2u : 0), SYS }; // static bit clear
        k2.dispatch((NP + 31) / 32, { dM, dN, dT, dV, dCur, dPred, dI, dNi, eC, eW, dBR, dTBR, dIBR, dSP, dTBP, dIBP, eWc }, prmDyn);
        dev->flush();
        const auto eGW = downloadBuffer<PsC>(dev, eW, CAP);
        const uint32_t eCount = downloadBuffer<uint32_t>(dev, eWc, 1)[0];
        // group writer contacts by particle (unique pLo), compare in order
        uint32_t sumChecked = 0, sumAll = 0;
        for (uint32_t p = 0; p < NP; ++p) { sumAll += std::min<uint32_t>(pref[p].contactCount, MAXMESH); }
        size_t mm3 = 0;
        std::vector<std::vector<uint32_t>> slots(NP);
        for (uint32_t i = 0; i < std::min(eCount, CAP); ++i)
            if (eGW[i].pLo < NP && eGW[i].pHi == SYS && eGW[i].rLo == 0x77u && eGW[i].rHi == 9u)
                slots[eGW[i].pLo].push_back(i);
            else ++mm3;
        for (uint32_t p = 0; p < NP; ++p)
        {
            if (taint[p]) continue;
            const PRef& pr = pref[p];
            const uint32_t expect = std::min<uint32_t>(pr.contactCount, MAXMESH);
            sumChecked += expect;
            if (slots[p].size() != expect) { ++mm3; continue; }
            std::sort(slots[p].begin(), slots[p].end()); // sequential per particle
            for (uint32_t sIdx = 0; sIdx < expect; ++sIdx)
            {
                const PsC& g2 = eGW[slots[p][sIdx]];
                if (!almostEqual(g2.normal_pen.w, pr.d[sIdx] - restDist, 1e-4f, 1e-4f) ||
                    !almostEqual(g2.normal_pen.x, pr.n[sIdx].x, 1e-4f, 1e-4f) ||
                    !almostEqual(g2.normal_pen.y, pr.n[sIdx].y, 1e-4f, 1e-4f) ||
                    !almostEqual(g2.normal_pen.z, pr.n[sIdx].z, 1e-4f, 1e-4f)) { ++mm3; break; }
            }
        }
        PXG_CHECK_MSG(mm3 == 0, "psMeshMid: %zu writer-path mismatches", mm3);
        PXG_CHECK_MSG(eCount >= sumChecked && eCount <= sumAll,
                      "psMeshMid: writer counter %u outside [%u, %u]", eCount, sumChecked, sumAll);
        PXG_CHECK_MSG(eCount > 40, "psMeshMid: writer path thin (%u)", eCount);
        for (PxgBufferHandle h : { dM, dN, dT, dV, dCur, dPred, dI, dNi, dC, dW, dBR, dTBR, dIBR, dSP, dTBP, dIBP, dWc, eC, eW, eWc })
            dev->destroyBuffer(h);
    }
}

} // namespace
