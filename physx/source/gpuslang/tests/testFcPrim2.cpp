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

// Verification of fcPrim2.slang (femClothPrimitives.cu tranche 2:
// fcCapsuleCollision live path + clothParticleCollision core).
//
// Mode 0 (capsule vs cloth triangle) uses WITNESS verification for the
// closest-feature outputs: distanceSegmentTriangleSquared resolves feature
// ties arbitrarily, but any returned (t, u, v) must reconstruct the returned
// distance, the barycentric triangle point and the on-segment point must
// reproduce sqrt(sqDist) (tranche-55 strategy), the normal must be unit (or
// zero via the getNormalized fallback), the contact point must equal
// pointOnSegment + radius * normal, and pen must equal dist - radius.  Only
// the acceptance gate needs slack taint.  Crossing fixtures pin the
// sqDist == 0 path (pen == -radius exactly).
//
// Mode 1 (particle vs cloth triangle) compares directly against a CPU
// mirror (the closest point is unique, so outputs are continuous): normal
// NEGATED in the writer, pen = dist - restDistance, contact = p - n *
// restDistance, Cramer barycentric of the closest point (the eps-window
// bool is discarded upstream — pinned arithmetically via the barycentric
// values).  The PxgFEMContactWriter is verified slot by slot (particleId is
// the item index, so every slot maps back to its item): contactByX ==
// pairInd0, tempContactByX == its low 32 bits, contactIndexSortedByX ==
// slot, PxgFemOtherContactInfo pairInd0/pairInd1 with the FOURTH DWORD
// UNTOUCHED (poison-checked), and every non-tainted hit appears exactly
// once.  A small-capacity dispatch pins the ungated atomicAdd reservation
// (counter overshoots to the full total) and the capacity gate (slots
// beyond maxContacts keep their poison).
//
// KNOWN LIMITS: gate strictness (<= vs <, > vs >=) is unpinnable — exact
// float ties never occur in random fixtures and near-boundary items are
// tainted by design; the segSeg parallel fallback (denom < 1e-6 -> s=0.5)
// is never reached by random capsule/edge pairs.

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
struct Fc2Work { F4 a, b, c, shapeP, shapeQ; U4 ids, ids2; };
struct Fc2Out { F4 point, normalPen, barycentric; U4 count; };
struct Fc2Params { uint32_t numCases, maxContacts, pad0, pad1; };

struct V3 { float x, y, z; };
static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 add(V3 a, V3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 mul(V3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 crossp(V3 a, V3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static V3 normalize3(V3 v) { const float m = std::sqrt(dot3(v, v)); return mul(v, 1.0f / m); }
static float l1n(V3 v) { return std::fabs(v.x) + std::fabs(v.y) + std::fabs(v.z); }
static V3 v3(F4 f) { return { f.x, f.y, f.z }; }

struct Slack { bool near = false; void noteSc(float l, float r, float scale) { if (std::fabs(l - r) < 1e-5f * scale) near = true; } };

static const uint32_t SYSID = 9, CLOTHID = 42, SHIFT = 20;

static V3 quatRotate(F4 q, V3 v)
{
    const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d = q.x * vx + q.y * vy + q.z * vz;
    return { vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d,
             vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d,
             vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d };
}
static V3 basis0(F4 q)
{
    const float x2 = q.x * 2, w2 = q.w * 2;
    return { (q.w * w2) - 1.0f + q.x * x2, (q.z * w2) + q.y * x2, (-q.y * w2) + q.z * x2 };
}

// ---- distanceSegmentTriangleSquared reference (tranche-55 mirror) ----------
static bool isValidBary(float v, float w)
{
    const float eps = 1e-6f;
    return (v >= -eps && v <= 1 + eps) && (w >= -eps && w <= 1 + eps) && ((v + w) <= 1 + eps);
}
static void segSeg(V3 p1, V3 d1, V3 p2, V3 d2, float& s, float& t)
{
    const float eps = 1e-6f;
    const V3 r = sub(p1, p2);
    const float a = dot3(d1, d1), e = dot3(d2, d2), b = dot3(d1, d2), c = dot3(d1, r);
    const float aRecip = a > eps ? 1.0f / a : 0.0f, eRecip = e > eps ? 1.0f / e : 0.0f;
    const float f = dot3(d2, r);
    const float denom = a * e - b * b;
    const float s0 = std::min(std::max((b * f - c * e) / denom, 0.0f), 1.0f);
    const float sTmp = eps > denom ? 0.5f : s0;
    const float t2 = std::min(std::max((b * sTmp + f) * eRecip, 0.0f), 1.0f);
    s = std::min(std::max((b * t2 - c) * aRecip, 0.0f), 1.0f);
    t = t2;
}
static float refSegTri(V3 p, V3 q, V3 a, V3 b, V3 c)
{
    const float MAXF = 3.4028235e38f;
    const V3 pq = sub(q, p), ab = sub(b, a), ac = sub(c, a), bc = sub(c, b), ap = sub(p, a), aq = sub(q, a);
    const float d00 = dot3(ab, ab), d01 = dot3(ab, ac), d11 = dot3(ac, ac);
    const float tDenom = d00 * d11 - d01 * d01;
    const float bdenom = tDenom > 0 ? 1.0f / tDenom : MAXF;
    const V3 n = normalize3(crossp(ab, ac));
    const float dist3 = dot3(ap, n), dist4 = dot3(aq, n);
    const float sqDist3 = dist3 * dist3, sqDist4 = dist4 * dist4;

    if (dist3 * dist4 <= 0.0f)
    {
        const float t0 = -dot3(n, ap) * (1.0f / dot3(n, pq));
        const V3 ip = add(p, mul(pq, t0)), v2 = sub(ip, a);
        const float d20 = dot3(v2, ab), d21 = dot3(v2, ac);
        const float v00 = (d11 * d20 - d01 * d21) * bdenom, w00 = (d00 * d21 - d01 * d20) * bdenom;
        if (isValidBary(v00, w00)) return 0.0f;
    }
    const V3 pV20 = sub(sub(p, mul(n, dist3)), a);
    const float v0 = (d11 * dot3(pV20, ab) - d01 * dot3(pV20, ac)) * bdenom;
    const float w0 = (d00 * dot3(pV20, ac) - d01 * dot3(pV20, ab)) * bdenom;
    const bool con0 = isValidBary(v0, w0);
    const V3 qV20 = sub(sub(q, mul(n, dist4)), a);
    const float v1 = (d11 * dot3(qV20, ab) - d01 * dot3(qV20, ac)) * bdenom;
    const float w1 = (d00 * dot3(qV20, ac) - d01 * dot3(qV20, ab)) * bdenom;
    const bool con1 = isValidBary(v1, w1);
    if (con0 && con1) return sqDist4 > sqDist3 ? sqDist3 : sqDist4;

    float t00, t10, t01, t11, t02, t12;
    segSeg(p, pq, a, ab, t00, t10);
    segSeg(p, pq, b, bc, t01, t11);
    segSeg(p, pq, a, ac, t02, t12);
    V3 tv = sub(add(p, mul(pq, t00)), add(a, mul(ab, t10))); const float sqDist0 = dot3(tv, tv);
    tv = sub(add(p, mul(pq, t01)), add(b, mul(bc, t11))); const float sqDist1 = dot3(tv, tv);
    tv = sub(add(p, mul(pq, t02)), add(a, mul(ac, t12))); const float sqDist2 = dot3(tv, tv);
    float sqDistPE;
    if (sqDist1 > sqDist0 && sqDist2 > sqDist0) sqDistPE = sqDist0;
    else if (sqDist2 > sqDist1) sqDistPE = sqDist1;
    else sqDistPE = sqDist2;
    if (con0) return sqDistPE > sqDist3 ? sqDist3 : sqDistPE;
    if (con1) return sqDistPE > sqDist4 ? sqDist4 : sqDistPE;
    return sqDistPE;
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
    if (vc <= 0 && d1 >= 0 && d3 <= 0) { const float v = d1 / (d1 - d3); return add(a, mul(ab, v)); }
    const V3 cp = sub(p, c);
    const float d5 = dot3(ab, cp), d6 = dot3(ac, cp);
    if (d6 >= 0 && d5 <= d6) return c;
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) { const float w = d2 / (d2 - d6); return add(a, mul(ac, w)); }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0)
    { const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6)); return add(b, mul(sub(c, b), w)); }
    const float denom = 1.0f / (va + vb + vc);
    return add(add(a, mul(ab, vb * denom)), mul(ac, vc * denom));
}

static void refBary(V3 a, V3 b, V3 c, V3 p, F4& bary)
{
    const V3 v0 = sub(b, a), v1 = sub(c, a), v2 = sub(p, a);
    const float d00 = dot3(v0, v0), d01 = dot3(v0, v1), d11 = dot3(v1, v1);
    const float d20 = dot3(v2, v0), d21 = dot3(v2, v1);
    const float denom = d00 * d11 - d01 * d01;
    const float v = (d11 * d20 - d01 * d21) / denom;
    const float w = (d00 * d21 - d01 * d20) / denom;
    bary = { 1.0f - v - w, v, w, 0.0f };
}

PXG_TEST(fcPrim2_capsuleParticle)
{
    std::mt19937 rng(94017);
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    const uint32_t NC = 4000, CAP = 4096, POISON = 0xeeeeeeeeu, INFOPOISON = 0xdeadbeefu;
    std::vector<Fc2Work> work(NC);
    std::vector<uint8_t> mode(NC), taint(NC, 0), refHit(NC, 0);
    std::vector<float> refSq(NC, 0);     // mode 0: gate distance
    std::vector<F4> refNP(NC), refPt(NC), refBc(NC); // mode 1 mirrors
    size_t m0Hits = 0, m0Miss = 0, m0Cross = 0, m1Hits = 0, m1Miss = 0;

    for (uint32_t i = 0; i < NC; ++i)
    {
        // healthy triangle
        V3 a, e0, e1;
        do
        {
            a = { rf(-2, 2), rf(-2, 2), rf(-2, 2) };
            e0 = { rf(-1.5f, 1.5f), rf(-1.5f, 1.5f), rf(-1.5f, 1.5f) };
            e1 = { rf(-1.5f, 1.5f), rf(-1.5f, 1.5f), rf(-1.5f, 1.5f) };
        } while (dot3(crossp(e0, e1), crossp(e0, e1)) < 0.25f);
        const V3 b = add(a, e0), c = add(a, e1);
        const V3 n = normalize3(crossp(e0, e1));
        const float w0 = rf(0.15f, 0.5f), w1 = rf(0.1f, 0.4f);
        const V3 inside = add(a, add(mul(e0, w0), mul(e1, w1)));

        Fc2Work& w = work[i];
        w.a = { a.x, a.y, a.z, 0 };
        w.b = { b.x, b.y, b.z, 0 };
        w.c = { c.x, c.y, c.z, 0 };
        w.ids2 = { 0, 0, 0, 0 };
        mode[i] = (uint8_t)(i & 1);
        Slack sl;

        if (mode[i] == 0) // capsule
        {
            F4 q = { rf(-1, 1), rf(-1, 1), rf(-1, 1), rf(-1, 1) };
            const float qm = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
            q = { q.x / qm, q.y / qm, q.z / qm, q.w / qm };
            const float hh = rf(0.2f, 1.0f), radius = rf(0.05f, 0.35f), cd = rf(0.05f, 0.3f);
            V3 pose;
            if (i % 6 == 0) // crossing fixture: segment through the interior
            {
                const V3 dir = basis0(q);
                pose = sub(inside, mul(dir, rf(-0.8f, 0.8f) * hh));
            }
            else
                pose = add(inside, add(mul(n, rf(-1.2f, 1.2f)), V3{ rf(-0.6f, 0.6f), rf(-0.6f, 0.6f), rf(-0.6f, 0.6f) }));
            w.a.w = cd;
            w.shapeP = { pose.x, pose.y, pose.z, hh };
            w.shapeQ = q;
            w.ids = { 0, 0, 0, 0 };
            w.ids.y = *(const uint32_t*)&radius;

            const V3 dir = basis0(q);
            const V3 p = add(pose, mul(dir, hh)), qq = sub(pose, mul(dir, hh));
            refSq[i] = refSegTri(p, qq, a, b, c);
            const float infl = radius + cd;
            const float S = l1n(p) + l1n(qq) + l1n(a) + l1n(b) + l1n(c) + 1.0f;
            sl.noteSc(refSq[i], infl * infl, S * S);
            refHit[i] = infl * infl > refSq[i] ? 1 : 0;
            taint[i] = sl.near ? 1 : 0;
            if (!taint[i] && refHit[i]) { ++m0Hits; if (refSq[i] == 0.0f) ++m0Cross; }
            if (!taint[i] && !refHit[i]) ++m0Miss;
        }
        else // particle
        {
            const float cd = rf(0.1f, 0.4f), rest = rf(0.0f, 0.12f);
            V3 p;
            if (i % 9 == 1) // outer region beyond vertex a
                p = add(a, add(mul(sub(a, inside), rf(0.2f, 1.0f)), mul(n, rf(-0.4f, 0.4f))));
            else if (i % 9 == 4) // beyond vertex b
                p = add(b, add(mul(sub(b, inside), rf(0.2f, 1.0f)), mul(n, rf(-0.4f, 0.4f))));
            else if (i % 9 == 7) // beyond vertex c
                p = add(c, add(mul(sub(c, inside), rf(0.2f, 1.0f)), mul(n, rf(-0.4f, 0.4f))));
            else
                p = add(inside, add(mul(n, rf(-0.5f, 0.5f)), V3{ rf(-0.3f, 0.3f), rf(-0.3f, 0.3f), rf(-0.3f, 0.3f) }));
            w.a.w = cd;
            w.b.w = rest;
            w.shapeP = { p.x, p.y, p.z, 0 };
            w.shapeQ = { 0, 0, 0, 1 };
            w.ids = { 1, i, SYSID, CLOTHID };
            w.ids2 = { rng() % (1u << SHIFT), 0, 0, 0 };

            const V3 closest = refTriClosest(p, a, b, c);
            const V3 v = sub(p, closest);
            const float sqDist = dot3(v, v);
            const float S = l1n(p) + l1n(a) + l1n(b) + l1n(c) + 1.0f;
            sl.noteSc(sqDist, cd * cd, S * S);
            refHit[i] = sqDist <= cd * cd ? 1 : 0;
            if (refHit[i])
            {
                const float m = std::sqrt(sqDist);
                sl.noteSc(m, 0.0f, 1000.0f); // grazing shield: n = v/m unguarded
                const V3 nn = mul(v, 1.0f / m);
                const V3 contact = sub(p, mul(nn, rest));
                refNP[i] = { -nn.x, -nn.y, -nn.z, m - rest };
                refPt[i] = { contact.x, contact.y, contact.z, 0 };
                refBary(a, b, c, closest, refBc[i]);
            }
            taint[i] = sl.near ? 1 : 0;
            if (!taint[i] && refHit[i]) ++m1Hits;
            if (!taint[i] && !refHit[i]) ++m1Miss;
        }
    }

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<Fc2Out> outInit(NC);
    for (auto& o : outInit) { o.point = { -7, -7, -7, -7 }; o.count = { 77, 0, 0, 0 }; }
    std::vector<F4> femF4Init(CAP, { -9, -9, -9, -9 });
    std::vector<U4> infoInit(CAP, { INFOPOISON, INFOPOISON, INFOPOISON, INFOPOISON });
    std::vector<uint32_t> u1Init(CAP, POISON), u2Init(CAP * 2, POISON), wc0(1, 0);
    const std::vector<uint32_t> strides = { (uint32_t)sizeof(Fc2Work), (uint32_t)sizeof(Fc2Out),
                                            16, 16, 16, 16, 4, 8, 4, 4 };

    struct RunOut
    {
        std::vector<Fc2Out> out;
        std::vector<F4> pt, np, bc;
        std::vector<U4> info;
        std::vector<uint32_t> byX, tByX, idxX;
        uint32_t wc;
    };
    auto run = [&](uint32_t cap, RunOut& o) {
        PxgBufferHandle bW = createBufferFrom(dev, work);
        PxgBufferHandle bO = createBufferFrom(dev, outInit);
        PxgBufferHandle bPt = createBufferFrom(dev, femF4Init);
        PxgBufferHandle bNp = createBufferFrom(dev, femF4Init);
        PxgBufferHandle bBc = createBufferFrom(dev, femF4Init);
        PxgBufferHandle bIn = createBufferFrom(dev, infoInit);
        PxgBufferHandle bWc = createBufferFrom(dev, wc0);
        PxgBufferHandle bBy = createBufferFrom(dev, u2Init);
        PxgBufferHandle bTb = createBufferFrom(dev, u1Init);
        PxgBufferHandle bIx = createBufferFrom(dev, u1Init);
        Fc2Params prm = { NC, cap, 0, 0 };
        Kernel k("fcPrim2.spv", "ctFcPrim2", strides, sizeof(Fc2Params));
        k.dispatch((NC + 255) / 256, { bW, bO, bPt, bNp, bBc, bIn, bWc, bBy, bTb, bIx }, prm);
        dev->flush();
        o.out = downloadBuffer<Fc2Out>(dev, bO, NC);
        o.pt = downloadBuffer<F4>(dev, bPt, CAP);
        o.np = downloadBuffer<F4>(dev, bNp, CAP);
        o.bc = downloadBuffer<F4>(dev, bBc, CAP);
        o.info = downloadBuffer<U4>(dev, bIn, CAP);
        o.byX = downloadBuffer<uint32_t>(dev, bBy, CAP * 2);
        o.tByX = downloadBuffer<uint32_t>(dev, bTb, CAP);
        o.idxX = downloadBuffer<uint32_t>(dev, bIx, CAP);
        o.wc = downloadBuffer<uint32_t>(dev, bWc, 1)[0];
        for (PxgBufferHandle h : { bW, bO, bPt, bNp, bBc, bIn, bWc, bBy, bTb, bIx })
            dev->destroyBuffer(h);
    };

    RunOut A;
    run(CAP, A);

    // ---- mode 0: witness verification --------------------------------------
    size_t mm0 = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        if (mode[i] != 0 || taint[i]) continue;
        const Fc2Out& o = A.out[i];
        if (o.count.x != refHit[i]) { ++mm0; continue; }
        if (!refHit[i]) { if (o.point.x != 0.0f || o.normalPen.w != 0.0f) ++mm0; continue; }

        const V3 a = v3(work[i].a), b = v3(work[i].b), c = v3(work[i].c);
        const float radius = *(const float*)&work[i].ids.y;
        const V3 dir = basis0(work[i].shapeQ);
        const V3 pose = v3(work[i].shapeP);
        const float hh = work[i].shapeP.w;
        const V3 p = add(pose, mul(dir, hh)), q = sub(pose, mul(dir, hh));

        const float dist = o.normalPen.w + radius; // pen = dist - radius
        const V3 nrm = v3(o.normalPen);
        const float nl = dot3(nrm, nrm);
        // dist must be the NON-NEGATIVE root (pen = +sqrt(sqDist) - radius)
        bool ok = dist > -1e-3f &&
                  almostEqual(dist * dist, refSq[i], 2e-3f, 2e-4f) &&
                  (std::fabs(nl - 1.0f) < 1e-3f || nl < 1e-6f);
        // barycentric witness: triangle point + on-segment point reproduce dist
        const float wB = o.barycentric.x, uB = o.barycentric.y, vB = o.barycentric.z;
        ok = ok && std::fabs(wB + uB + vB - 1.0f) < 1e-5f &&
             uB > -1e-4f && vB > -1e-4f && uB + vB < 1.0f + 1e-4f;
        const V3 triPt = add(add(mul(a, wB), mul(b, uB)), mul(c, vB));
        const V3 segPt = sub(v3(o.point), mul(nrm, radius)); // point = seg + r*n
        const V3 d = sub(triPt, segPt);
        ok = ok && almostEqual(dot3(d, d), refSq[i], 2e-3f, 2e-4f);
        // the normal must POINT from the segment witness to the triangle
        // witness (normal = normalize(pointOnTriangle - pointOnSegment)) —
        // magnitude alone would accept any unit vector
        ok = ok && (dist < 1e-3f || std::fabs(dot3(nrm, d) - dist) < 2e-3f * (1.0f + dist));
        // segPt must lie ON the capsule segment
        const V3 pq = sub(q, p);
        const float tRec = std::min(std::max(dot3(sub(segPt, p), pq) / dot3(pq, pq), 0.0f), 1.0f);
        const V3 offSeg = sub(segPt, add(p, mul(pq, tRec)));
        ok = ok && dot3(offSeg, offSeg) < 1e-5f * (1.0f + dot3(pq, pq));
        if (!ok) ++mm0;
    }
    PXG_CHECK_MSG(mm0 == 0, "fcPrim2: %zu capsule witness mismatches", mm0);
    PXG_CHECK_MSG(m0Hits > 400 && m0Miss > 300 && m0Cross > 40,
                  "fcPrim2: capsule coverage thin (h=%zu m=%zu x=%zu)", m0Hits, m0Miss, m0Cross);

    // ---- mode 1: direct comparison + writer slots ---------------------------
    size_t mm1 = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        if (mode[i] != 1 || taint[i]) continue;
        const Fc2Out& o = A.out[i];
        if (o.count.x != refHit[i]) { ++mm1; continue; }
        if (!refHit[i]) continue;
        if (!almostEqual(o.normalPen.x, refNP[i].x, 1e-4f, 1e-4f) ||
            !almostEqual(o.normalPen.y, refNP[i].y, 1e-4f, 1e-4f) ||
            !almostEqual(o.normalPen.z, refNP[i].z, 1e-4f, 1e-4f) ||
            !almostEqual(o.normalPen.w, refNP[i].w, 1e-4f, 1e-4f) ||
            !almostEqual(o.point.x, refPt[i].x, 1e-4f, 1e-4f) ||
            !almostEqual(o.point.z, refPt[i].z, 1e-4f, 1e-4f) ||
            !almostEqual(o.barycentric.x, refBc[i].x, 5e-3f, 5e-4f) ||
            !almostEqual(o.barycentric.y, refBc[i].y, 5e-3f, 5e-4f) ||
            !almostEqual(o.barycentric.z, refBc[i].z, 5e-3f, 5e-4f) ||
            o.barycentric.w != 0.0f)
            ++mm1;
    }
    PXG_CHECK_MSG(mm1 == 0, "fcPrim2: %zu particle mismatches", mm1);
    PXG_CHECK_MSG(m1Hits > 400 && m1Miss > 200,
                  "fcPrim2: particle coverage thin (h=%zu m=%zu)", m1Hits, m1Miss);

    // writer slots: map back via particleId == item index
    size_t mmW = 0;
    std::vector<uint32_t> seen(NC, 0);
    const uint32_t written = std::min(A.wc, CAP);
    for (uint32_t s = 0; s < written; ++s)
    {
        const uint32_t item = A.info[s].x;
        if (item >= NC || mode[item] != 1) { ++mmW; continue; }
        ++seen[item];
        const uint32_t pairInd1 = (CLOTHID << SHIFT) | work[item].ids2.x;
        if (A.info[s].y != SYSID || A.info[s].z != pairInd1 || A.info[s].w != INFOPOISON ||
            A.byX[s * 2] != item || A.byX[s * 2 + 1] != SYSID || A.tByX[s] != item || A.idxX[s] != s)
            { ++mmW; continue; }
        if (taint[item]) continue;
        if (!almostEqual(A.np[s].x, refNP[item].x, 1e-4f, 1e-4f) ||
            !almostEqual(A.np[s].y, refNP[item].y, 1e-4f, 1e-4f) ||
            !almostEqual(A.np[s].z, refNP[item].z, 1e-4f, 1e-4f) ||
            !almostEqual(A.np[s].w, refNP[item].w, 1e-4f, 1e-4f) ||
            !almostEqual(A.pt[s].y, refPt[item].y, 1e-4f, 1e-4f) ||
            !almostEqual(A.bc[s].y, refBc[item].y, 5e-3f, 5e-4f))
            ++mmW;
    }
    uint32_t hitsChecked = 0, taintedM1 = 0;
    for (uint32_t i = 0; i < NC; ++i)
    {
        if (mode[i] != 1) continue;
        if (taint[i]) { ++taintedM1; continue; }
        hitsChecked += refHit[i];
        if (seen[i] != (uint32_t)refHit[i]) ++mmW;
    }
    PXG_CHECK_MSG(mmW == 0, "fcPrim2: %zu writer slot mismatches", mmW);
    PXG_CHECK_MSG(A.wc >= hitsChecked && A.wc <= hitsChecked + taintedM1,
                  "fcPrim2: writer counter %u outside [%u, %u]", A.wc, hitsChecked, hitsChecked + taintedM1);

    // ---- small capacity: ungated reservation, gated stores ------------------
    {
        const uint32_t TINY = 32;
        RunOut H;
        run(TINY, H);
        size_t mmH = 0;
        for (uint32_t s = 0; s < TINY; ++s)
        {
            const uint32_t item = H.info[s].x;
            if (item >= NC || mode[item] != 1 || H.idxX[s] != s || H.tByX[s] != item)
                ++mmH;
        }
        for (uint32_t s = TINY; s < CAP; ++s)
            if (H.info[s].x != INFOPOISON || H.idxX[s] != POISON || H.byX[s * 2] != POISON ||
                H.byX[s * 2 + 1] != POISON || H.tByX[s] != POISON ||
                H.pt[s].x != -9.0f || H.np[s].x != -9.0f || H.bc[s].x != -9.0f)
                ++mmH;
        PXG_CHECK_MSG(mmH == 0, "fcPrim2: %zu capacity-gate violations", mmH);
        PXG_CHECK_MSG(H.wc == A.wc && H.wc > TINY,
                      "fcPrim2: overflow counter %u (expected %u > %u)", H.wc, A.wc, TINY);
    }
}

} // namespace
