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

// Verification of psPrimCollide.slang (psPrimitivesCollision - the
// particle-vs-primitive collision warp driver of cudaParticleSystem.cu).
// One work item per thread (pair x grid cell); the warp iterates the max
// cell population; per particle: speculative CCD contact volume or plain
// contact distance, rigid-filter lookup (unsorted id localized through the
// particle-buffer runsum into its sim buffer's PxgRigidFilterPair list),
// the tranche-87 contact core, then warp-aggregated writes (ballot prefix +
// one lane-0 atomicAdd) through PxgParticleContactWriter (contact + six
// radix staging arrays, capacity-gated).
//
// Contacts are grouped by the injective per-work-item rigidId and compared
// as sets keyed by the compressed particle index; normal/pen(-restDistance)
// to 1e-4.  Slack: the tranche-87 gate notes (maxDist rejections, box
// inside/outside and face ties, eps fallbacks, grazing shields) mark a work
// item skipped; integer plumbing (cells, ranges, encode, filter, staging
// arrays) is exact.

#include "pxgTestFramework.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };
struct V3 { float x, y, z; };
static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 add(V3 a, V3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 muls(V3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
static V3 neg(V3 a) { return { -a.x, -a.y, -a.z }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static V3 v3(F4 f) { return { f.x, f.y, f.z }; }
static F4 f4(V3 v) { return { v.x, v.y, v.z, 0.0f }; }
static float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

static const float MAXF = 3.4028235e38f;
static const float EPSR = 1.192092896e-07f;

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
static V3 basis0(F4 q)
{
    const float x2 = q.x * 2, w2 = q.w * 2;
    return { (q.w * w2) - 1.0f + q.x * x2, (q.z * w2) + q.y * x2, (-q.y * w2) + q.z * x2 };
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

static float distPtSegSq(V3 a, V3 b, V3 p, float& param)
{
    const V3 ap = sub(p, a), ab = sub(b, a);
    const float nom = dot3(ap, ab);
    const float denom = dot3(ab, ab);
    const float tValue = clampf(nom / denom, 0.0f, 1.0f);
    const float t = denom > 0.0f ? tValue : 0.0f;
    const V3 v = sub(ap, muls(ab, t));
    param = t;
    return dot3(v, v);
}

static float distPtTriSq(V3 p, V3 a, V3 b, V3 c, V3& closestP)
{
    const V3 ab = sub(b, a), ac = sub(c, a);
    closestP = refTriClosest(p, a, b, c, ab, ac);
    const V3 v = sub(p, closestP);
    return dot3(v, v);
}

// ---- reference contact functions (mirroring the shader, slack-noted) ----

static bool refBox(V3& normal, float& dist, V3 pointPos, V3 boxP, F4 boxQ, V3 he, float maxDist, Slack& sl)
{
    const V3 rel = quatRotateInv(boxQ, sub(pointPos, boxP));
    const V3 closest = { clampf(rel.x, -he.x, he.x), clampf(rel.y, -he.y, he.y), clampf(rel.z, -he.z, he.z) };
    if (maxDist < MAXF)
    {
        const float maxDistSq = maxDist * maxDist;
        const V3 dv = sub(rel, closest);
        const float distSq = dot3(dv, dv);
        sl.noteSc(distSq, maxDistSq, l1n(dv) * (l1n(rel) + l1n(closest)) + distSq + maxDistSq + 1e-3f);
        if (distSq > maxDistSq) return false;
    }
    // inside/outside per-axis decisions
    sl.noteSc(he.x, std::fabs(rel.x), he.x + std::fabs(rel.x) + 1.0f);
    sl.noteSc(he.y, std::fabs(rel.y), he.y + std::fabs(rel.y) + 1.0f);
    sl.noteSc(he.z, std::fabs(rel.z), he.z + std::fabs(rel.z) + 1.0f);
    const bool isInside = he.x >= std::fabs(rel.x) && he.y >= std::fabs(rel.y) && he.z >= std::fabs(rel.z);
    if (isInside)
    {
        const V3 dts = { he.x - std::fabs(closest.x), he.y - std::fabs(closest.y), he.z - std::fabs(closest.z) };
        sl.noteSc(dts.x, dts.y, 2.0f);
        sl.noteSc(dts.x, dts.z, 2.0f);
        sl.noteSc(dts.y, dts.z, 2.0f);
        normal = closest.z >= 0.0f ? V3{ 0, 0, 1 } : V3{ 0, 0, -1 };
        float sel = closest.z;
        dist = -dts.z;
        if (dts.x <= dts.y && dts.x <= dts.z)
        {
            normal = closest.x >= 0.0f ? V3{ 1, 0, 0 } : V3{ -1, 0, 0 };
            dist = -dts.x;
            sel = closest.x;
        }
        else if (dts.y <= dts.z)
        {
            normal = closest.y >= 0.0f ? V3{ 0, 1, 0 } : V3{ 0, -1, 0 };
            dist = -dts.y;
            sel = closest.y;
        }
        sl.noteSc(sel, 0.0f, 1.0f); // face-sign tie of the selected axis
    }
    else
    {
        normal = sub(rel, closest);
        const float mag = std::sqrt(dot3(normal, normal));
        sl.noteSc(mag, 0.0f, 1000.0f); // grazing: direction ill-conditioned
        normal = muls(normal, 1.0f / mag);
        dist = mag;
    }
    normal = quatRotate(boxQ, normal);
    return true;
}

static bool refSphere(V3& normal, float& dist, V3 pointPos, V3 spherePos, float radius, float maxDist, Slack& sl)
{
    const V3 rel = sub(pointPos, spherePos);
    const float dcSq = dot3(rel, rel);
    if (maxDist < MAXF)
    {
        const float mdc = maxDist + radius;
        sl.noteSc(dcSq, mdc * mdc, l1n(rel) * l1n(rel) + dcSq + mdc * mdc + 1e-3f);
        if (dcSq > mdc * mdc) return false;
    }
    const float dc = std::sqrt(dcSq);
    // grazing shield consistent with capsule/box (~1e-2 window); exact zero
    // stays checked - deterministic both sides
    if (dc != 0.0f) sl.noteSc(dc, 0.0f, 1000.0f);
    normal = (dc > EPSR) ? muls(rel, 1.0f / dc) : V3{ 1, 0, 0 };
    dist = dc - radius;
    return true;
}

static bool refCapsule(V3& normal, float& dist, V3 pointPos, V3 capsulePos, V3 capsuleDir,
                       float radius, float halfHeight, float maxDist, Slack& sl)
{
    const V3 axis = muls(capsuleDir, halfHeight);
    const V3 rel = sub(pointPos, capsulePos);
    float t;
    const float dsSq = distPtSegSq(neg(axis), axis, rel, t);
    if (maxDist < MAXF)
    {
        const float mds = maxDist + radius;
        sl.noteSc(dsSq, mds * mds, l1n(rel) * l1n(rel) + dsSq + mds * mds + 1e-3f);
        if (dsSq > mds * mds) return false;
    }
    const V3 dsVec = sub(rel, muls(axis, 2.0f * t - 1.0f));
    const float ds = std::sqrt(dsSq);
    // grazing contacts: the normal direction error scales as (float
    // divergence)/ds against the fixed 1e-4 compare tolerance, so shield
    // everything below ~1e-2 (exact zero stays checked - deterministic)
    if (ds != 0.0f) sl.noteSc(ds, 0.0f, 1000.0f);
    normal = (ds > EPSR) ? muls(dsVec, 1.0f / ds) : V3{ 1, 0, 0 };
    dist = ds - radius;
    return true;
}


struct PclWork { F4 shapeP, shapeQ, scale; U4 ids, rigid; };
struct PairR { uint32_t i0lo, i0hi, i1lo, i1hi, index2, p0, p1, p2; };
struct PsC { F4 normal_pen; uint32_t rLo, rHi, pLo, pHi; };
struct PclParams { uint32_t numWork, numBuffers, maxContacts, isTGS; };

PXG_TEST(psPrimCollide_driver)
{
    std::mt19937 rng(89021);
    std::uniform_real_distribution<float> u01(0, 1);
    std::uniform_real_distribution<float> uv(-1, 1);

    const uint32_t NP = 1200, NCELLS = 200, NW = 773, SYS = 3; // non-multiple of 32: idle-lane tail
    const bool TGS = true;

    // cells: contiguous runs of 0..8 particles in the sorted arrays
    // cellEnd of EMPTY cells is POISONED (the engine leaves it garbage; the
    // CUDA reads it only behind the EMPTY_CELL guard, so a dropped guard
    // must produce bogus contacts here rather than a benign wrapped range)
    std::vector<uint32_t> cellStart(256, 0xffffffffu), cellEnd(256, 5);
    std::vector<F4> curPos(NP), predPos(NP);
    std::vector<V3> cellCenter(NCELLS);
    {
        uint32_t p = 0;
        for (uint32_t c = 0; c < NCELLS && p < NP; ++c)
        {
            const uint32_t n = std::min<uint32_t>(2 + rng() % 9, NP - p);
            cellCenter[c] = { 4.0f * uv(rng), 4.0f * uv(rng), 4.0f * uv(rng) };
            if (n > 0)
            {
                cellStart[c] = p;
                cellEnd[c] = p + n;
                for (uint32_t j = 0; j < n; ++j, ++p)
                {
                    curPos[p] = { cellCenter[c].x + 0.5f * uv(rng), cellCenter[c].y + 0.5f * uv(rng), cellCenter[c].z + 0.5f * uv(rng), 1.0f };
                    predPos[p] = { curPos[p].x + 0.2f * uv(rng), curPos[p].y + 0.2f * uv(rng), curPos[p].z + 0.2f * uv(rng), 1.0f };
                }
            }
        }
        // trim NP to what was placed
        PXG_CHECK(p > 900);
        for (; p < NP; ++p) { curPos[p] = { 100, 100, 100, 1 }; predPos[p] = curPos[p]; }
    }
    std::vector<uint32_t> s2u(NP);
    std::iota(s2u.begin(), s2u.end(), 0);
    std::shuffle(s2u.begin(), s2u.end(), rng);

    // sim buffers + filters
    const uint32_t runsum[3] = { 0, 400, 900 };
    std::vector<uint32_t> runsumV(runsum, runsum + 3);
    std::vector<PairR> filters;
    std::vector<U4> filterTable(3);
    std::vector<PclWork> work(NW);
    std::vector<std::vector<std::pair<uint64_t, uint32_t>>> perBuf(3); // (rigidId, localId)

    for (uint32_t i = 0; i < NW; ++i)
    {
        PclWork& w = work[i];
        const uint32_t cell = (i * 13) % 256; // cells >= NCELLS are EMPTY
        const uint32_t type = i % 4;
        const bool ccd = (i / 4) % 2 == 1;
        const float cd = 0.15f + 0.15f * u01(rng);
        const float rd = 0.02f + 0.08f * u01(rng);
        w.ids = { cell, type, ccd ? 1u : 0u, SYS };
        w.rigid = { 0x40u + i, 7u, 0, 0 };
        const V3 cc = cell < NCELLS ? cellCenter[cell] : V3{ 0, 0, 0 };
        const V3 dir = [&]() { for (;;) { V3 d = { uv(rng), uv(rng), uv(rng) }; float m = dot3(d, d); if (m > 0.05f) return muls(d, 1.0f / std::sqrt(m)); } }();
        const V3 sp = add(cc, muls(dir, 0.4f * u01(rng)));
        w.shapeP = { sp.x, sp.y, sp.z, cd };
        F4 q = { uv(rng), uv(rng), uv(rng), 1.0f + uv(rng) };
        const float qm = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
        q.x /= qm; q.y /= qm; q.z /= qm; q.w /= qm;
        w.shapeQ = q;
        w.scale = { 0.3f + 0.5f * u01(rng), 0.25f + 0.4f * u01(rng), 0.3f + 0.4f * u01(rng), rd };

        // filters: block roughly half of this cell's particles (by UNSORTED id, localized)
        if (i % 5 == 2 && cell < NCELLS && cellStart[cell] != 0xffffffffu)
        {
            for (uint32_t pi = cellStart[cell]; pi < cellEnd[cell]; ++pi)
            {
                if ((pi & 1) == 0) continue;
                const uint32_t unsorted = s2u[pi];
                uint32_t buf = unsorted < 400 ? 0 : (unsorted < 900 ? 1 : 2);
                perBuf[buf].push_back({ (uint64_t(7) << 32) | (0x40u + i), unsorted - runsum[buf] });
            }
        }
    }
    for (int b = 0; b < 3; ++b)
    {
        std::sort(perBuf[b].begin(), perBuf[b].end());
        filterTable[b] = { (uint32_t)filters.size(), (uint32_t)perBuf[b].size(), 0, 0 };
        for (const auto& f : perBuf[b])
            filters.push_back({ (uint32_t)(f.first & 0xffffffffu), (uint32_t)(f.first >> 32), f.second, 0, 0, 0, 0, 0 });
    }
    if (filters.empty()) filters.push_back({ 0, 0, 0, 0, 0, 0, 0, 0 });

    // ---- reference ----
    struct RefC { uint32_t particle; V3 normal; float pen; };
    std::vector<std::vector<RefC>> refC(NW);
    std::vector<uint8_t> skip(NW, 0);
    size_t hitT[4] = { 0 }, emptyItems = 0, filteredBlocks = 0, ccdHits = 0, multiCell = 0;
    size_t bufBlocked[3] = { 0 }, bufBlockedWouldHit[3] = { 0 };
    uint32_t sumNonSkipped = 0, sumSkippedCap = 0;

    auto whichBuf = [&](uint32_t unsorted) { return unsorted < 400 ? 0 : (unsorted < 900 ? 1 : 2); };
    auto refFind = [&](uint64_t rigid, uint32_t unsorted) {
        const uint32_t buf = whichBuf(unsorted);
        const uint32_t local = unsorted - runsum[buf];
        const auto& v = perBuf[buf];
        return std::binary_search(v.begin(), v.end(), std::make_pair(rigid, local));
    };

    for (uint32_t i = 0; i < NW; ++i)
    {
        const PclWork& w = work[i];
        const uint32_t cell = w.ids.x;
        Slack sl;
        if (cell >= NCELLS || cellStart[cell] == 0xffffffffu) { ++emptyItems; continue; }
        const uint32_t start = cellStart[cell], end = cellEnd[cell];
        size_t itemHits = 0, itemCcdHits = 0, itemFiltered = 0;
        size_t itemBufBlocked[3] = { 0 }, itemBufWouldHit[3] = { 0 };
        const bool ccd = w.ids.z != 0;
        const uint64_t rigid = (uint64_t(7) << 32) | (0x40u + i);

        for (uint32_t pi = start; pi < end; ++pi)
        {
            const V3 cur = v3(curPos[pi]);
            V3 cvp = cur;
            float cvr = w.shapeP.w;
            if (ccd)
            {
                // getParticleSpeculativeContactVolume(isDiffuse=false, TGS)
                const V3 pred = v3(predPos[pi]);
                const float scaleInv = TGS ? 2.0f : 1.0f; // 1/0.5, 1/1
                const V3 off = muls(sub(pred, cur), scaleInv * 0.5f);
                cvp = add(cur, off);
                cvr = std::sqrt(dot3(off, off)) + w.shapeP.w;
            }
            const bool blocked = refFind(rigid, s2u[pi]);
            if (blocked) { ++itemFiltered; ++itemBufBlocked[whichBuf(s2u[pi])]; }

            bool intersect = false;
            V3 normal = { 0, 0, 0 };
            float dist = 0;
            switch (w.ids.y)
            {
            case 0:
            {
                const V3 pn = quatRotate(w.shapeQ, V3{ 1, 0, 0 });
                const float pdd = -dot3(pn, v3(w.shapeP));
                dist = dot3(pn, cvp) + pdd;
                sl.noteSc(dist, cvr, l1n(cvp) + l1n(v3(w.shapeP)) + 1.0f);
                if (dist <= cvr)
                {
                    if (ccd) dist = dot3(pn, cur) + pdd;
                    normal = pn;
                    intersect = true;
                }
                break;
            }
            case 1:
                intersect = refSphere(normal, dist, cvp, v3(w.shapeP), w.scale.x, cvr, sl);
                if (intersect && ccd) refSphere(normal, dist, cur, v3(w.shapeP), w.scale.x, MAXF, sl);
                break;
            case 2:
                intersect = refCapsule(normal, dist, cvp, v3(w.shapeP), basis0(w.shapeQ), w.scale.y, w.scale.x, cvr, sl);
                if (intersect && ccd) refCapsule(normal, dist, cur, v3(w.shapeP), basis0(w.shapeQ), w.scale.y, w.scale.x, MAXF, sl);
                break;
            default:
                intersect = refBox(normal, dist, cvp, v3(w.shapeP), w.shapeQ, v3(w.scale), cvr, sl);
                if (intersect && ccd) refBox(normal, dist, cur, v3(w.shapeP), w.shapeQ, v3(w.scale), MAXF, sl);
                break;
            }
            if (intersect)
            {
                if (blocked)
                {
                    // a filtered particle that WOULD contact: pins the buffer
                    // localization (a broken '- runsum' silently un-blocks)
                    ++itemBufWouldHit[whichBuf(s2u[pi])];
                }
                else
                {
                    refC[i].push_back({ pi, normal, dist - w.scale.w });
                    ++itemHits;
                    if (ccd) ++itemCcdHits;
                }
            }
        }
        skip[i] = sl.near ? 1 : 0;
        if (skip[i]) sumSkippedCap += end - start;
        else
        {
            sumNonSkipped += (uint32_t)refC[i].size();
            hitT[w.ids.y] += itemHits;
            ccdHits += itemCcdHits;
            filteredBlocks += itemFiltered;
            if (end - start >= 4) ++multiCell;
            for (int b2 = 0; b2 < 3; ++b2)
            {
                bufBlocked[b2] += itemBufBlocked[b2];
                bufBlockedWouldHit[b2] += itemBufWouldHit[b2];
            }
        }
    }

    // ---- GPU ----
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t CAP = 4096;
    std::vector<PsC> ctInit(CAP);
    std::vector<uint32_t> u2z(CAP * 2, 0), uz(CAP, 0), cnt(1, 0);
    PxgBufferHandle bW = createBufferFrom(dev, work);
    PxgBufferHandle bCS = createBufferFrom(dev, cellStart);
    PxgBufferHandle bCE = createBufferFrom(dev, cellEnd);
    PxgBufferHandle bS2U = createBufferFrom(dev, s2u);
    PxgBufferHandle bCur = createBufferFrom(dev, curPos);
    PxgBufferHandle bPred = createBufferFrom(dev, predPos);
    PxgBufferHandle bRun = createBufferFrom(dev, runsumV);
    PxgBufferHandle bFT = createBufferFrom(dev, filterTable);
    PxgBufferHandle bFP = createBufferFrom(dev, filters);
    PxgBufferHandle bC = createBufferFrom(dev, ctInit);
    PxgBufferHandle bBR = createBufferFrom(dev, u2z);
    PxgBufferHandle bTBR = createBufferFrom(dev, uz);
    PxgBufferHandle bIBR = createBufferFrom(dev, uz);
    PxgBufferHandle bSP = createBufferFrom(dev, u2z);
    PxgBufferHandle bTBP = createBufferFrom(dev, uz);
    PxgBufferHandle bIBP = createBufferFrom(dev, uz);
    PxgBufferHandle bCc = createBufferFrom(dev, cnt);

    PclParams prm = { NW, 3, CAP, TGS ? 1u : 0u };
    Kernel k("psPrimCollide.spv", "ctPsPrimCollide",
             { (uint32_t)sizeof(PclWork), 4, 4, 4, 16, 16, 4, 16, (uint32_t)sizeof(PairR),
               (uint32_t)sizeof(PsC), 8, 4, 4, 8, 4, 4, 4 },
             sizeof(PclParams));
    k.dispatch((NW + 31) / 32, { bW, bCS, bCE, bS2U, bCur, bPred, bRun, bFT, bFP, bC,
                          bBR, bTBR, bIBR, bSP, bTBP, bIBP, bCc }, prm);
    dev->flush();

    const auto gC = downloadBuffer<PsC>(dev, bC, CAP);
    const auto gBR = downloadBuffer<uint32_t>(dev, bBR, CAP * 2);
    const auto gTBR = downloadBuffer<uint32_t>(dev, bTBR, CAP);
    const auto gIBR = downloadBuffer<uint32_t>(dev, bIBR, CAP);
    const auto gSP = downloadBuffer<uint32_t>(dev, bSP, CAP * 2);
    const auto gTBP = downloadBuffer<uint32_t>(dev, bTBP, CAP);
    const auto gIBP = downloadBuffer<uint32_t>(dev, bIBP, CAP);
    const uint32_t gCount = downloadBuffer<uint32_t>(dev, bCc, 1)[0];
    for (PxgBufferHandle h : { bW, bCS, bCE, bS2U, bCur, bPred, bRun, bFT, bFP, bC, bBR, bTBR, bIBR, bSP, bTBP, bIBP, bCc })
        dev->destroyBuffer(h);

    // group contacts by rigid low word (injective per work item)
    std::vector<std::vector<uint32_t>> byItem(NW);
    size_t bogus = 0;
    const uint32_t nUsed = std::min(gCount, CAP);
    for (uint32_t i = 0; i < nUsed; ++i)
    {
        const uint32_t wi = gC[i].rLo - 0x40u;
        if (wi < NW && gC[i].rHi == 7u && gC[i].pHi == SYS) byItem[wi].push_back(i);
        else ++bogus;
    }
    PXG_CHECK_MSG(bogus == 0, "psPrimCollide: %zu bogus contacts", bogus);

    size_t mm = 0, checked = 0, skipped = 0;
    for (uint32_t i = 0; i < NW; ++i)
    {
        if (skip[i]) { ++skipped; continue; }
        ++checked;
        if (byItem[i].size() != refC[i].size()) { ++mm; continue; }
        for (const RefC& r : refC[i])
        {
            bool found = false;
            for (uint32_t slot : byItem[i])
            {
                if (gC[slot].pLo != r.particle) continue;
                found = almostEqual(gC[slot].normal_pen.w, r.pen, 1e-4f, 1e-4f) &&
                        almostEqual(gC[slot].normal_pen.x, r.normal.x, 1e-4f, 1e-4f) &&
                        almostEqual(gC[slot].normal_pen.y, r.normal.y, 1e-4f, 1e-4f) &&
                        almostEqual(gC[slot].normal_pen.z, r.normal.z, 1e-4f, 1e-4f);
                break;
            }
            if (!found) { ++mm; break; }
        }
    }
    PXG_CHECK_MSG(mm == 0, "psPrimCollide: %zu mismatches", mm);
    PXG_CHECK_MSG(gCount >= sumNonSkipped && gCount <= sumNonSkipped + sumSkippedCap,
                  "psPrimCollide: counter %u outside [%u, %u]", gCount, sumNonSkipped, sumNonSkipped + sumSkippedCap);
    PXG_CHECK_MSG(checked > NW * 8 / 10, "psPrimCollide: too many skipped (%zu)", skipped);

    // radix staging arrays: self-consistent for every written slot
    size_t mmS = 0;
    for (uint32_t i = 0; i < nUsed; ++i)
    {
        if (gBR[i * 2] != gC[i].rLo || gBR[i * 2 + 1] != gC[i].rHi) ++mmS;
        if (gTBR[i] != gC[i].rLo || gIBR[i] != i) ++mmS;
        if (gSP[i * 2] != gC[i].pLo || gSP[i * 2 + 1] != gC[i].pHi) ++mmS;
        if (gTBP[i] != gC[i].pLo || gIBP[i] != i) ++mmS;
    }
    PXG_CHECK_MSG(mmS == 0, "psPrimCollide: %zu staging inconsistencies", mmS);

    // capacity-overflow dispatch: the counter overshoots (atomicAdd is not
    // gated) while slots >= maxContacts stay untouched
    {
        const uint32_t TINY = 64;
        std::vector<PsC> poison(CAP);
        for (auto& c : poison) { c.normal_pen = { -9, -9, -9, -9 }; c.rLo = 0xdeadbeef; c.rHi = 0xdeadbeef; c.pLo = 0xdeadbeef; c.pHi = 0xdeadbeef; }
        std::vector<uint32_t> z2(CAP * 2, 0xabababab), z1(CAP, 0xabababab), c0(1, 0);
        PxgBufferHandle oW = createBufferFrom(dev, work);
        PxgBufferHandle oCS = createBufferFrom(dev, cellStart);
        PxgBufferHandle oCE = createBufferFrom(dev, cellEnd);
        PxgBufferHandle oS2U = createBufferFrom(dev, s2u);
        PxgBufferHandle oCur = createBufferFrom(dev, curPos);
        PxgBufferHandle oPred = createBufferFrom(dev, predPos);
        PxgBufferHandle oRun = createBufferFrom(dev, runsumV);
        PxgBufferHandle oFT = createBufferFrom(dev, filterTable);
        PxgBufferHandle oFP = createBufferFrom(dev, filters);
        PxgBufferHandle oC = createBufferFrom(dev, poison);
        PxgBufferHandle oBR = createBufferFrom(dev, z2);
        PxgBufferHandle oTBR = createBufferFrom(dev, z1);
        PxgBufferHandle oIBR = createBufferFrom(dev, z1);
        PxgBufferHandle oSP = createBufferFrom(dev, z2);
        PxgBufferHandle oTBP = createBufferFrom(dev, z1);
        PxgBufferHandle oIBP = createBufferFrom(dev, z1);
        PxgBufferHandle oCc = createBufferFrom(dev, c0);
        PclParams prm2 = { NW, 3, TINY, TGS ? 1u : 0u };
        Kernel k2("psPrimCollide.spv", "ctPsPrimCollide",
                  { (uint32_t)sizeof(PclWork), 4, 4, 4, 16, 16, 4, 16, (uint32_t)sizeof(PairR),
                    (uint32_t)sizeof(PsC), 8, 4, 4, 8, 4, 4, 4 },
                  sizeof(PclParams));
        k2.dispatch((NW + 31) / 32, { oW, oCS, oCE, oS2U, oCur, oPred, oRun, oFT, oFP, oC,
                                      oBR, oTBR, oIBR, oSP, oTBP, oIBP, oCc }, prm2);
        dev->flush();
        const auto oGC = downloadBuffer<PsC>(dev, oC, CAP);
        const uint32_t oCount = downloadBuffer<uint32_t>(dev, oCc, 1)[0];
        size_t dirty = 0, writtenBelow = 0;
        for (uint32_t i = 0; i < CAP; ++i)
        {
            if (i < TINY) { if (oGC[i].rLo != 0xdeadbeef) ++writtenBelow; }
            else if (oGC[i].rLo != 0xdeadbeef) ++dirty;
        }
        PXG_CHECK_MSG(oCount == gCount && oCount > TINY && dirty == 0 && writtenBelow == TINY,
                      "psPrimCollide: overflow run wrong (cnt=%u dirty=%zu below=%zu)", oCount, dirty, writtenBelow);
        for (PxgBufferHandle h : { oW, oCS, oCE, oS2U, oCur, oPred, oRun, oFT, oFP, oC, oBR, oTBR, oIBR, oSP, oTBP, oIBP, oCc })
            dev->destroyBuffer(h);
    }

    PXG_CHECK_MSG(hitT[0] > 15 && hitT[1] > 15 && hitT[2] > 15 && hitT[3] > 15 &&
                  emptyItems > 40 && filteredBlocks > 10 && ccdHits > 30 && multiCell > 10 &&
                  bufBlocked[0] > 0 && bufBlocked[1] > 0 && bufBlocked[2] > 0 &&
                  bufBlockedWouldHit[0] > 0 && bufBlockedWouldHit[1] > 0 && bufBlockedWouldHit[2] > 0,
                  "psPrimCollide: coverage thin (t %zu/%zu/%zu/%zu e=%zu f=%zu c=%zu m=%zu b=%zu/%zu/%zu w=%zu/%zu/%zu)",
                  hitT[0], hitT[1], hitT[2], hitT[3], emptyItems, filteredBlocks, ccdHits, multiCell,
                  bufBlocked[0], bufBlocked[1], bufBlocked[2],
                  bufBlockedWouldHit[0], bufBlockedWouldHit[1], bufBlockedWouldHit[2]);
}

} // namespace
