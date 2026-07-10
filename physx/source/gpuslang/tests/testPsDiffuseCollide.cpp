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

// Verification of psDiffuseCollide.slang (psDiffusePrimitivesCollision -
// the DIFFUSE-particle variant of the particle-vs-primitive collision
// driver).  No filter, no warp aggregation: per intersecting particle an
// atomic slot is claimed in that particle's one-way contact list (cap 12),
// writing normal/pen(-restDistance), a zeroed impulse and the rigid node
// index STRIDED by the diffuse particle count.
//
// Multiple work items hit the same cell CONCURRENTLY, so slot ORDER per
// particle is nondeterministic; verification is per PARTICLE: the count
// must equal the expected contact count exactly; each written slot's node
// index identifies its source work item (injective rigid ids), whose
// expected contact for this particle must match to 1e-4 with a zero
// impulse and no duplicate sources; when the count is within the cap the
// slot set must be the complete expected set.  A particle is skipped when
// ANY evaluation against it had a near event (slack taints the particle,
// not the work item, since slots are shared).  Unwritten slots keep their
// sentinels; a maxDiffuseParticles == 0 dispatch must leave everything
// untouched.

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



struct PdWork { F4 shapeP, shapeQ, scale; U4 ids, rigid; };
struct PdParams { uint32_t numWork, numParticles, maxDiffuseParticles, isTGS; };

PXG_TEST(psDiffuseCollide_driver)
{
    std::mt19937 rng(90017);
    std::uniform_real_distribution<float> u01(0, 1);
    std::uniform_real_distribution<float> uv(-1, 1);

    const uint32_t NP = 900, NCELLS = 160, NW = 900, SYS = 3;
    const uint32_t MAXSLOTS = 12;
    const bool TGS = true;

    std::vector<uint32_t> cellStart(256, 0xffffffffu), cellEnd(256, 7); // poisoned empty cellEnd
    std::vector<F4> curPos(NP), predPos(NP);
    std::vector<V3> cellCenter(NCELLS);
    {
        uint32_t p = 0;
        for (uint32_t c = 0; c < NCELLS && p < NP; ++c)
        {
            const uint32_t n = std::min<uint32_t>(2 + rng() % 8, NP - p);
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
        PXG_CHECK(p > 700);
        for (; p < NP; ++p) { curPos[p] = { 100, 100, 100, 1 }; predPos[p] = curPos[p]; }
    }

    std::vector<PdWork> work(NW);
    for (uint32_t i = 0; i < NW; ++i)
    {
        PdWork& w = work[i];
        const uint32_t cell = (i * 13) % 256; // repeats -> concurrent same-cell items; >= NCELLS EMPTY
        const uint32_t type = i % 4;
        const bool ccd = (i / 4) % 2 == 1;
        const float cd = 0.2f + 0.2f * u01(rng);
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
        if (i >= NW - 20)
        {
            // slot-overflow fixtures: 20 covering spheres on cell 0 push its
            // particles past MaxStaticContactsPerParticle
            w.ids = { 0, 1, 0, SYS };
            w.shapeP = { cellCenter[0].x, cellCenter[0].y, cellCenter[0].z, 0.4f };
            w.scale.x = 0.8f;
        }
    }

    // ---- reference: expected contacts per PARTICLE ----
    struct Exp { uint32_t item; V3 normal; float pen; };
    std::vector<std::vector<Exp>> expP(NP);
    std::vector<uint8_t> taint(NP, 0);
    size_t hitT[4] = { 0 }, emptyItems = 0, ccdHits = 0;

    for (uint32_t i = 0; i < NW; ++i)
    {
        const PdWork& w = work[i];
        const uint32_t cell = w.ids.x;
        if (cell >= NCELLS || cellStart[cell] == 0xffffffffu) { ++emptyItems; continue; }
        const bool ccd = w.ids.z != 0;
        for (uint32_t pi = cellStart[cell]; pi < cellEnd[cell]; ++pi)
        {
            Slack sl;
            const V3 cur = v3(curPos[pi]);
            V3 cvp = cur;
            float cvr = w.shapeP.w;
            if (ccd)
            {
                // isDiffuse = true -> scaleInv = 1
                const V3 off = muls(sub(v3(predPos[pi]), cur), 1.0f * 0.5f);
                cvp = add(cur, off);
                cvr = std::sqrt(dot3(off, off)) + w.shapeP.w;
            }
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
            if (sl.near) taint[pi] = 1; // slots are shared across items -> particle-level skip
            if (intersect)
                expP[pi].push_back({ i, normal, dist - w.scale.w });
        }
    }
    // per-type/CCD floors over UNTAINTED particles only (tainted ones are
    // excluded from comparison, so they must not satisfy coverage)
    for (uint32_t p = 0; p < NP; ++p)
    {
        if (taint[p]) continue;
        for (const Exp& e : expP[p])
        {
            ++hitT[work[e.item].ids.y];
            if (work[e.item].ids.z != 0) ++ccdHits;
        }
    }

    // ---- GPU ----
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t SLOTS = NP * MAXSLOTS;
    std::vector<F4> infoInit(SLOTS, { -9, -9, -9, -9 });
    std::vector<float> impInit(SLOTS, 5.0f);
    std::vector<uint32_t> nodeInit(SLOTS * 2, 0xcccccccc), cnt0(NP, 0);
    PxgBufferHandle bW = createBufferFrom(dev, work);
    PxgBufferHandle bCS = createBufferFrom(dev, cellStart);
    PxgBufferHandle bCE = createBufferFrom(dev, cellEnd);
    PxgBufferHandle bCur = createBufferFrom(dev, curPos);
    PxgBufferHandle bPred = createBufferFrom(dev, predPos);
    PxgBufferHandle bI = createBufferFrom(dev, infoInit);
    PxgBufferHandle bImp = createBufferFrom(dev, impInit);
    PxgBufferHandle bN = createBufferFrom(dev, nodeInit);
    PxgBufferHandle bC = createBufferFrom(dev, cnt0);

    PdParams prm = { NW, NP, 64, TGS ? 1u : 0u };
    Kernel k("psDiffuseCollide.spv", "ctPsDiffuseCollide",
             { (uint32_t)sizeof(PdWork), 4, 4, 16, 16, 16, 4, 8, 4 }, sizeof(PdParams));
    k.dispatch((NW + 255) / 256, { bW, bCS, bCE, bCur, bPred, bI, bImp, bN, bC }, prm);
    dev->flush();
    const auto gI = downloadBuffer<F4>(dev, bI, SLOTS);
    const auto gImp = downloadBuffer<float>(dev, bImp, SLOTS);
    const auto gN = downloadBuffer<uint32_t>(dev, bN, SLOTS * 2);
    const auto gC = downloadBuffer<uint32_t>(dev, bC, NP);

    size_t mm = 0, checkedP = 0, taintedP = 0, overflowP = 0, contactedP = 0;
    for (uint32_t p = 0; p < NP; ++p)
    {
        if (taint[p]) { ++taintedP; continue; }
        ++checkedP;
        const auto& exp = expP[p];
        if (gC[p] != exp.size()) { ++mm; continue; }
        if (!exp.empty()) ++contactedP;
        if (exp.size() > MAXSLOTS) ++overflowP;
        const uint32_t written = std::min<uint32_t>((uint32_t)exp.size(), MAXSLOTS);
        std::vector<uint8_t> used(exp.size(), 0);
        for (uint32_t s = 0; s < written; ++s)
        {
            const uint32_t slot = p + s * NP;
            const uint32_t item = gN[slot * 2] - 0x40u;
            bool ok = gN[slot * 2 + 1] == 7u && gImp[slot] == 0.0f;
            bool matched = false;
            for (uint32_t e = 0; e < exp.size() && ok; ++e)
            {
                if (used[e] || exp[e].item != item) continue;
                matched = almostEqual(gI[slot].w, exp[e].pen, 1e-4f, 1e-4f) &&
                          almostEqual(gI[slot].x, exp[e].normal.x, 1e-4f, 1e-4f) &&
                          almostEqual(gI[slot].y, exp[e].normal.y, 1e-4f, 1e-4f) &&
                          almostEqual(gI[slot].z, exp[e].normal.z, 1e-4f, 1e-4f);
                used[e] = 1;
                break;
            }
            if (!ok || !matched) { ++mm; break; }
        }
        // slots beyond the written window keep their sentinels
        for (uint32_t s = written; s < MAXSLOTS; ++s)
        {
            const uint32_t slot = p + s * NP;
            if (gI[slot].x != -9.0f || gImp[slot] != 5.0f || gN[slot * 2] != 0xcccccccc) { ++mm; break; }
        }
    }
    PXG_CHECK_MSG(mm == 0, "psDiffuseCollide: %zu mismatches", mm);
    PXG_CHECK_MSG(checkedP > NP * 8 / 10, "psDiffuseCollide: too many tainted (%zu)", taintedP);
    PXG_CHECK_MSG(hitT[0] > 20 && hitT[1] > 20 && hitT[2] > 20 && hitT[3] > 20 &&
                  emptyItems > 100 && ccdHits > 60 && overflowP >= 2 && contactedP > 100,
                  "psDiffuseCollide: coverage thin (t %zu/%zu/%zu/%zu e=%zu c=%zu o=%zu p=%zu)",
                  hitT[0], hitT[1], hitT[2], hitT[3], emptyItems, ccdHits, overflowP, contactedP);

    // maxDiffuseParticles == 0: everything untouched
    {
        std::vector<uint32_t> cnt1(NP, 0);
        PxgBufferHandle zI = createBufferFrom(dev, infoInit);
        PxgBufferHandle zImp = createBufferFrom(dev, impInit);
        PxgBufferHandle zN = createBufferFrom(dev, nodeInit);
        PxgBufferHandle zC = createBufferFrom(dev, cnt1);
        PdParams prm0 = { NW, NP, 0, TGS ? 1u : 0u };
        Kernel k0("psDiffuseCollide.spv", "ctPsDiffuseCollide",
                  { (uint32_t)sizeof(PdWork), 4, 4, 16, 16, 16, 4, 8, 4 }, sizeof(PdParams));
        k0.dispatch((NW + 255) / 256, { bW, bCS, bCE, bCur, bPred, zI, zImp, zN, zC }, prm0);
        dev->flush();
        const auto zGC = downloadBuffer<uint32_t>(dev, zC, NP);
        const auto zGI = downloadBuffer<F4>(dev, zI, SLOTS);
        const auto zGImp = downloadBuffer<float>(dev, zImp, SLOTS);
        const auto zGN = downloadBuffer<uint32_t>(dev, zN, SLOTS * 2);
        size_t touched = 0;
        for (uint32_t p = 0; p < NP; ++p) if (zGC[p] != 0) ++touched;
        for (uint32_t s = 0; s < SLOTS; ++s)
            if (zGI[s].x != -9.0f || zGImp[s] != 5.0f || zGN[s * 2] != 0xcccccccc) ++touched;
        PXG_CHECK_MSG(touched == 0, "psDiffuseCollide: maxDiffuse==0 touched %zu", touched);
        dev->destroyBuffer(zI); dev->destroyBuffer(zImp); dev->destroyBuffer(zN); dev->destroyBuffer(zC);
    }

    for (PxgBufferHandle h : { bW, bCS, bCE, bCur, bPred, bI, bImp, bN, bC })
        dev->destroyBuffer(h);
}

} // namespace
