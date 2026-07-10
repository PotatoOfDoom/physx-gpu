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

// Verification of npPolyClip4.slang (femClothPrimitives.cu polyClip4 —
// convex polygon vs triangle clipping).  Contact ORDER is deterministic
// (phase 2 appends in polygon-vertex order, phase 3 per triangle-edge round
// in polygon-edge order), so slots compare directly against a scalar CPU
// mirror.  Slack taints a case when: a containment ray-crossing is within
// rounding of an edge/vertex, a segment-intersection sign product is near
// zero, a bbox reject is marginal, a phase gate (cosTheta sign,
// contactDist) is marginal, the witness distance is near zero (normal
// direction unstable), or the axis sits at the findRotationMatrixFromZ
// branch boundary (|z| ~ 0.9999).  Fixtures include near-z axes (the
// double-reflection branch), 10-16-vert polygons, and a deterministic
// 32-gon family whose clip yields 33 candidates — pinning the
// NUM_TMP_CONTACTS_PER_PAIR = 32 write cap against the UNCAPPED counter.
//
// KNOWN LIMITS: the containsParallel eps shift ((jx + PX_EPS_REAL) * jiy)
// and every strict-vs-nonstrict comparison direction (comp1 >= comp2,
// sign products < 0) are unpinnable — they only matter in exact-tie
// neighborhoods that random fixtures never hit and the gate taints
// exclude by design; those tokens were verified character-by-character
// against the CUDA instead.

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
struct PcCase { F4 axis, plane0, plane1, t0, t1, t2, misc; U4 counts; };
struct PcParams { uint32_t numCases, pad0, pad1, pad2; };

struct V3 { float x, y, z; };
static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 add(V3 a, V3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 mul(V3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 crossp(V3 a, V3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static V3 normalize3(V3 v) { const float m = std::sqrt(dot3(v, v)); return mul(v, 1.0f / m); }
static V3 getNormalized(V3 v) { const float m = dot3(v, v); return m > 0 ? mul(v, 1.0f / std::sqrt(m)) : V3{ 0, 0, 0 }; }

static const uint32_t NUMTMP = 32;
static const float EPSR = 1.192092896e-07f;
static const float MAXF = 3.4028235e38f;

struct Slack { bool near = false; void noteSc(float l, float r, float scale) { if (std::fabs(l - r) < 1e-5f * scale) near = true; } };

struct M33 { V3 c0, c1, c2; };
static V3 mT(const M33& m, V3 v) { return add(add(mul(m.c0, v.x), mul(m.c1, v.y)), mul(m.c2, v.z)); }
static V3 mTT(const M33& m, V3 v) { return { dot3(m.c0, v), dot3(m.c1, v), dot3(m.c2, v) }; }

static M33 findRotZ(V3 to, Slack& sl)
{
    sl.noteSc(std::fabs(to.z), 0.9999f, 1.0f);
    const float e = to.z, f = std::fabs(e);
    M33 m;
    if (f <= 0.9999f)
    {
        const float vx = -to.y, vy = to.x;
        const float h = 1.0f / (1.0f + e);
        const float hvx = h * vx, hvxy = hvx * vy;
        m.c0 = { e + hvx * vx, hvxy, vy };
        m.c1 = { hvxy, e + h * vy * vy, -vx };
        m.c2 = { -vy, vx, e };
    }
    else
    {
        const V3 from = { 0, 0, 1 }, absFrom = { 0, 1, 0 };
        const V3 u = sub(absFrom, from), v = sub(absFrom, to);
        const float c1 = -(2.0f / dot3(u, u));
        const float c2 = -(2.0f / dot3(v, v));
        const float c3 = c1 * c2 * dot3(u, v);
        const V3 c1u = mul(u, c1), c2v = mul(v, c2), c3v = mul(v, c3);
        V3 col0 = add(add(mul(u, c3v.x), mul(v, c2v.x)), mul(u, c1u.x)); col0.x += 1.0f;
        V3 col1 = add(add(mul(u, c3v.y), mul(v, c2v.y)), mul(u, c1u.y)); col1.y += 1.0f;
        V3 col2 = add(add(mul(u, c3v.z), mul(v, c2v.z)), mul(u, c1u.z)); col2.z += 1.0f;
        m.c0 = col0; m.c1 = col1; m.c2 = col2;
    }
    return m;
}

static bool refContains(const V3* verts, uint32_t numVerts, V3 p, Slack& sl)
{
    const float tx = p.x, ty = p.y;
    uint32_t numIntersects = 0;
    for (uint32_t tI = 0; tI < numVerts; ++tI)
    {
        const V3 start = verts[tI];
        const V3 endP = verts[tI + 1 == numVerts ? 0 : tI + 1];
        const float jy = endP.y, iy = start.y;
        const float minY = std::min(jy, iy), maxY = std::max(jy, iy);
        sl.noteSc(ty, minY, 20.0f);
        sl.noteSc(ty, maxY, 20.0f);
        if ((minY != maxY) && ty >= minY && ty <= maxY)
        {
            const float jx = endP.x, ix = start.x;
            const float jix = ix - jx, jiy = iy - jy, jty = ty - jy;
            const float part1 = jty * jix;
            const float part2 = (jx + EPSR) * jiy;
            const float part3 = tx * jiy;
            const bool comp = jiy > 0.0f;
            const float tmp = part1 + part2;
            const float comp1 = comp ? tmp : part3;
            const float comp2 = comp ? part3 : tmp;
            sl.noteSc(comp1, comp2, 40.0f);
            if (comp1 >= comp2) ++numIntersects;
        }
    }
    return (numIntersects % 2) == 1;
}

static float area2D(V3 a, V3 b, V3 c)
{
    const V3 ca = sub(a, c), cb = sub(b, c);
    return ca.x * cb.y - ca.y * cb.x;
}

static bool refSegIntersect(V3 ipA, V3 ipB, V3 rpA, V3 rpB, V3 rMin, V3 rMax,
                            V3& pAA, V3& pBB, Slack& sl)
{
    const V3 iMin = { std::min(ipA.x, ipB.x), std::min(ipA.y, ipB.y), std::min(ipA.z, ipB.z) };
    const V3 iMax = { std::max(ipA.x, ipB.x), std::max(ipA.y, ipB.y), std::max(ipA.z, ipB.z) };
    sl.noteSc(iMin.x, rMax.x, 20.0f);
    sl.noteSc(rMin.x, iMax.x, 20.0f);
    sl.noteSc(iMin.y, rMax.y, 20.0f);
    sl.noteSc(rMin.y, iMax.y, 20.0f);
    const bool xCon = iMin.x > rMax.x || rMin.x > iMax.x;
    const bool yCon = iMin.y > rMax.y || rMin.y > iMax.y;
    if (xCon || yCon) return false;
    const float a1 = area2D(rpA, rpB, ipA);
    const float a2 = area2D(rpA, rpB, ipB);
    sl.noteSc(a1 * a2, 0.0f, 400.0f);
    if ((a1 * a2) < 0.0f)
    {
        const float a3 = area2D(ipA, ipB, rpA);
        const float a4 = area2D(ipA, ipB, rpB);
        sl.noteSc(a3 * a4, 0.0f, 400.0f);
        if ((a3 * a4) < 0.0f)
        {
            const float t0 = a1 / (a2 - a1);
            pAA = sub(ipA, mul(sub(ipB, ipA), t0));
            const float t1 = a3 / (a4 - a3);
            pBB = sub(rpA, mul(sub(rpB, rpA), t1));
            return true;
        }
    }
    return false;
}

// full polyClip4 mirror; ct is 32 slots x 8 floats
static uint32_t refPolyClip(const V3* v0, uint32_t nbVerts0, V3 axis, F4 plane0, F4 plane1,
                            const V3* v1, float contactDist, float* ct, Slack& sl)
{
    const M33 rot = findRotZ(axis, sl);
    V3 triangleP[3], polyPoint[32];
    for (uint32_t l = 0; l < 3; ++l) triangleP[l] = mT(rot, v1[l]);
    for (uint32_t l = 0; l < nbVerts0; ++l) polyPoint[l] = mT(rot, v0[l]);
    const float sqCd = contactDist * contactDist;
    const V3 p0n = { plane0.x, plane0.y, plane0.z };
    const V3 p1n = { plane1.x, plane1.y, plane1.z };
    uint32_t num = 0;

    // phase 2
    uint32_t mask = 0;
    for (uint32_t j = 0; j < nbVerts0; ++j)
        if (refContains(triangleP, 3, polyPoint[j], sl)) mask |= 1u << j;
    {
        float dist[32];
        V3 nor[32], cp[32];
        uint32_t keep = 0;
        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            if (!(mask & (1u << lane))) continue;
            const V3 proj = sub(v0[lane], mul(p1n, dot3(p1n, v0[lane]) + plane1.w));
            const V3 dir = sub(v0[lane], proj);
            const float cosT = dot3(dir, p0n);
            const float sq = dot3(dir, dir);
            sl.noteSc(cosT, 0.0f, 20.0f);
            sl.noteSc(sq, sqCd, 20.0f);
            sl.noteSc(sq, 0.0f, 10.0f); // getNormalized direction unstable near 0
            if (cosT < 0.0f)
            {
                if (sq > sqCd) continue;
                dist[lane] = std::sqrt(sq);
                nor[lane] = mul(getNormalized(dir), -1.0f);
            }
            else
            {
                dist[lane] = -std::sqrt(sq);
                nor[lane] = getNormalized(dir);
            }
            cp[lane] = proj;
            keep |= 1u << lane;
        }
        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            if (!(keep & (1u << lane))) continue;
            const uint32_t idx = (uint32_t)__builtin_popcount(keep & ((1u << lane) - 1)) + num;
            if (idx < NUMTMP)
            {
                ct[8 * idx + 0] = cp[lane].x; ct[8 * idx + 1] = cp[lane].y;
                ct[8 * idx + 2] = cp[lane].z; ct[8 * idx + 3] = dist[lane];
                ct[8 * idx + 4] = nor[lane].x; ct[8 * idx + 5] = nor[lane].y;
                ct[8 * idx + 6] = nor[lane].z; ct[8 * idx + 7] = 0.0f;
            }
        }
        num += (uint32_t)__builtin_popcount(keep);
    }

    // phase 3
    for (uint32_t rStart = 0, rEnd = 2; rStart < 3; rEnd = rStart++)
    {
        const V3 rpA = triangleP[rStart], rpB = triangleP[rEnd];
        const V3 rMin = { std::min(rpA.x, rpB.x), std::min(rpA.y, rpB.y), std::min(rpA.z, rpA.z) };
        const V3 rMax = { std::max(rpA.x, rpB.x), std::max(rpA.y, rpB.y), std::max(rpA.z, rpA.z) };
        float dist[32];
        V3 nor[32], cp[32];
        uint32_t mask3 = 0;
        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            if (lane >= nbVerts0) continue;
            const V3 ipA = polyPoint[lane];
            const V3 ipB = polyPoint[lane + 1 == nbVerts0 ? 0 : lane + 1];
            V3 pAA, pBB;
            if (!refSegIntersect(ipA, ipB, rpA, rpB, rMin, rMax, pAA, pBB, sl)) continue;
            const V3 pA = mTT(rot, pAA);
            const V3 cpt = mTT(rot, pBB);
            const V3 dir = sub(pA, cpt);
            const float cosT = dot3(dir, p0n);
            const float sq = dot3(dir, dir);
            sl.noteSc(cosT, 0.0f, 20.0f);
            sl.noteSc(sq, sqCd, 20.0f);
            sl.noteSc(sq, 0.0f, 10.0f);
            if (cosT < 0.0f)
            {
                if (sq > sqCd) continue;
                dist[lane] = std::sqrt(sq);
                nor[lane] = mul(getNormalized(dir), -1.0f);
            }
            else
            {
                dist[lane] = -std::sqrt(sq);
                nor[lane] = getNormalized(dir);
            }
            cp[lane] = cpt;
            mask3 |= 1u << lane;
        }
        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            if (!(mask3 & (1u << lane))) continue;
            const uint32_t idx = (uint32_t)__builtin_popcount(mask3 & ((1u << lane) - 1)) + num;
            if (idx < NUMTMP)
            {
                ct[8 * idx + 0] = cp[lane].x; ct[8 * idx + 1] = cp[lane].y;
                ct[8 * idx + 2] = cp[lane].z; ct[8 * idx + 3] = dist[lane];
                ct[8 * idx + 4] = nor[lane].x; ct[8 * idx + 5] = nor[lane].y;
                ct[8 * idx + 6] = nor[lane].z; ct[8 * idx + 7] = 0.0f;
            }
        }
        num += (uint32_t)__builtin_popcount(mask3);
    }
    return num;
}

PXG_TEST(npPolyClip4_clip)
{
    std::mt19937 rng(95023);
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    const uint32_t NC = 3000;
    std::vector<PcCase> cases(NC);
    std::vector<F4> poly(NC * 32, { 0, 0, 0, 0 });
    std::vector<uint32_t> refNum(NC);
    std::vector<std::vector<float>> refCt(NC);
    std::vector<uint8_t> taint(NC, 0);
    size_t hits = 0, misses = 0, phase3Cases = 0, capOver = 0, zBranch = 0;

    for (uint32_t i = 0; i < NC; ++i)
    {
        V3 axis, nP, C;
        uint32_t n;
        float radius[32], theta[32];
        const bool zCase = (i % 11 == 5);
        const bool capCase = (i % 29 == 7);
        if (zCase)
        {
            const float sgn = (i & 1) ? 1.0f : -1.0f;
            axis = normalize3({ rf(-1e-3f, 1e-3f), rf(-1e-3f, 1e-3f), sgn });
            nP = normalize3(add(axis, { rf(-0.05f, 0.05f), rf(-0.05f, 0.05f), 0 }));
        }
        else
        {
            nP = normalize3({ rf(-1, 1), rf(-1, 1), rf(-1, 1) });
            axis = normalize3(add(nP, { rf(-0.35f, 0.35f), rf(-0.35f, 0.35f), rf(-0.35f, 0.35f) }));
        }
        C = { rf(-2, 2), rf(-2, 2), rf(-2, 2) };
        n = capCase ? 32 : ((i % 7 == 2) ? 10 + rng() % 7 : 3 + rng() % 6);

        // orthonormal basis in the polygon plane
        const V3 anyV = std::fabs(nP.x) < 0.7f ? V3{ 1, 0, 0 } : V3{ 0, 1, 0 };
        const V3 b1 = normalize3(crossp(nP, anyV));
        const V3 b2 = crossp(nP, b1);

        for (uint32_t k = 0; k < n; ++k)
        {
            theta[k] = (2.0f * 3.14159265f * k) / n + (capCase ? 0.0f : rf(0.0f, 0.6f / n));
            radius[k] = capCase ? 1.0f : rf(0.5f, 1.6f);
        }
        for (uint32_t k = 0; k < n; ++k)
        {
            const V3 v = add(C, add(mul(b1, radius[k] * std::cos(theta[k])),
                                    mul(b2, radius[k] * std::sin(theta[k]))));
            poly[i * 32 + k] = { v.x, v.y, v.z, 0 };
        }

        V3 t[3];
        if (capCase)
        {
            // huge triangle covering the 32-gon except vertex 0 (at C + b1):
            // its edge crosses just inside that vertex -> 31 verts contained,
            // 2 polygon-edge crossings -> 33 candidates, write cap at 32
            const V3 up = mul(nP, 0.02f);
            t[0] = add(add(C, mul(b1, 0.99f)), add(mul(b2, -8.0f), up));
            t[1] = add(add(C, mul(b1, 0.99f)), add(mul(b2, 8.0f), up));
            t[2] = add(add(C, mul(b1, -8.0f)), up);
        }
        else
        {
            for (uint32_t k = 0; k < 3; ++k)
                t[k] = add(C, add(mul(b1, rf(-1.8f, 1.8f)),
                                  add(mul(b2, rf(-1.8f, 1.8f)), mul(nP, rf(-0.6f, 0.6f)))));
        }

        PcCase& cs = cases[i];
        cs.axis = { axis.x, axis.y, axis.z, 0 };
        cs.plane0 = { nP.x, nP.y, nP.z, -dot3(C, nP) };
        const V3 nT = getNormalized(crossp(sub(t[1], t[0]), sub(t[2], t[0])));
        cs.plane1 = { nT.x, nT.y, nT.z, -dot3(t[0], nT) };
        cs.t0 = { t[0].x, t[0].y, t[0].z, 0 };
        cs.t1 = { t[1].x, t[1].y, t[1].z, 0 };
        cs.t2 = { t[2].x, t[2].y, t[2].z, 0 };
        cs.misc = { capCase ? 0.3f : rf(0.05f, 0.4f), 0, 0, 0 };
        cs.counts = { n, 0, 0, 0 };

        V3 pv[32];
        for (uint32_t k = 0; k < n; ++k) pv[k] = { poly[i * 32 + k].x, poly[i * 32 + k].y, poly[i * 32 + k].z };
        refCt[i].assign(NUMTMP * 8, -8.0f);
        Slack sl;
        refNum[i] = refPolyClip(pv, n, axis, cs.plane0, cs.plane1, t, cs.misc.x, refCt[i].data(), sl);
        taint[i] = sl.near ? 1 : 0;
        if (!taint[i])
        {
            if (refNum[i]) ++hits; else ++misses;
            if (refNum[i] > NUMTMP) ++capOver;
            if (zCase) ++zBranch;
        }
    }
    // phase-3 coverage: count non-tainted cases whose contact list includes a
    // slot produced after fewer phase-2 contacts than refNum (approximated by
    // re-running with a phase-2-only mirror would be heavy; instead require a
    // healthy mix of hit counts)
    for (uint32_t i = 0; i < NC; ++i)
        if (!taint[i] && refNum[i] >= 2) ++phase3Cases;

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<uint32_t> numInit(NC, 0xffffffffu);
    std::vector<F4> ctInit(NC * NUMTMP * 2, { -5, -5, -5, -5 });
    PxgBufferHandle bC = createBufferFrom(dev, cases);
    PxgBufferHandle bP = createBufferFrom(dev, poly);
    PxgBufferHandle bN = createBufferFrom(dev, numInit);
    PxgBufferHandle bT = createBufferFrom(dev, ctInit);
    PcParams prm = { NC, 0, 0, 0 };
    Kernel k("npPolyClip4.spv", "ctPolyClip4",
             { (uint32_t)sizeof(PcCase), 16, 4, 16 }, sizeof(PcParams));
    k.dispatch((NC + 63) / 64, { bC, bP, bN, bT }, prm);
    dev->flush();
    const auto gN = downloadBuffer<uint32_t>(dev, bN, NC);
    const auto gT = downloadBuffer<F4>(dev, bT, NC * NUMTMP * 2);

    size_t mm = 0, mmSlots = 0, checked = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        if (taint[i]) continue;
        ++checked;
        if (gN[i] != refNum[i]) { ++mm; continue; }
        const uint32_t nw = std::min(refNum[i], NUMTMP);
        bool ok = true;
        for (uint32_t s = 0; s < NUMTMP && ok; ++s)
        {
            const F4& pd = gT[(i * NUMTMP + s) * 2 + 0];
            const F4& nr = gT[(i * NUMTMP + s) * 2 + 1];
            if (s < nw)
            {
                const float* r = &refCt[i][8 * s];
                ok = almostEqual(pd.x, r[0], 1e-4f, 1e-4f) && almostEqual(pd.y, r[1], 1e-4f, 1e-4f) &&
                     almostEqual(pd.z, r[2], 1e-4f, 1e-4f) && almostEqual(pd.w, r[3], 1e-4f, 1e-4f) &&
                     almostEqual(nr.x, r[4], 1e-4f, 1e-4f) && almostEqual(nr.y, r[5], 1e-4f, 1e-4f) &&
                     almostEqual(nr.z, r[6], 1e-4f, 1e-4f) && nr.w == 0.0f;
            }
            else
                ok = pd.x == -8.0f && nr.x == -8.0f; // untouched sentinel
        }
        if (!ok) ++mmSlots;
    }
    PXG_CHECK_MSG(mm == 0, "polyClip4: %zu count mismatches", mm);
    PXG_CHECK_MSG(mmSlots == 0, "polyClip4: %zu slot mismatches", mmSlots);
    PXG_CHECK_MSG(checked > NC * 3 / 4, "polyClip4: too many tainted (%zu)", NC - checked);
    PXG_CHECK_MSG(hits > 800 && misses > 300, "polyClip4: coverage thin (h=%zu m=%zu)", hits, misses);
    PXG_CHECK_MSG(phase3Cases > 400, "polyClip4: multi-contact coverage thin (%zu)", phase3Cases);
    PXG_CHECK_MSG(capOver >= 15, "polyClip4: cap-overflow coverage thin (%zu)", capOver);
    PXG_CHECK_MSG(zBranch >= 40, "polyClip4: near-z rotation branch coverage thin (%zu)", zBranch);

    dev->destroyBuffer(bC); dev->destroyBuffer(bP); dev->destroyBuffer(bN); dev->destroyBuffer(bT);
}

} // namespace
