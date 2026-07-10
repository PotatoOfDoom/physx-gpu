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

// Verification of psPrimContact.slang (the particle-vs-primitive contact
// core: particleCollision.cuh contactPointBox/Sphere/Capsule +
// particleTriangleTest, and the particlePrimitiveCollision dispatcher's
// plane case with the CCD replacement semantics).  Per case the intersect
// flag must match exactly and, when intersecting, normal/dist to 1e-4.
// Slack-tracked float gates: the plane/sphere/capsule/box maxDist
// rejections, the box inside/outside per-axis decisions, the inside-face
// selection ties and face-sign ties, the sphere/capsule PX_EPS_REAL
// normal fallbacks (points placed EXACTLY at the sphere center/capsule
// center produce an exact 0 distance deterministically on both sides and
// stay checked), and the triangle gates (plane distance, closest distance,
// back-face cull, zero-distance normal fallback).  Quaternions are
// uploaded raw, so the rotation arithmetic mirrors the shader identically;
// the capsule axis uses PxQuat::getBasisVector0's specialized form.
// The dispatcher's eCONVEXCORE case (GJK/EPA) is deferred - see
// MIGRATION_STATUS.md.

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
struct PcWork { F4 particlePos, cVolume, shapeP, shapeQ, scale; U4 ids; };
struct PcOut { F4 normal_dist; U4 flags; };
struct PcParams { uint32_t numCases, pad0, pad1, pad2; };

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
    if (dc != 0.0f) sl.noteSc(dc, EPSR, 1.0f); // exact zero is deterministic both sides
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

static bool refTriangle(V3& normal, float& distance, V3 particlePos, V3 cVolumePos, float cVolumeRadius,
                        V3 a, V3 b, V3 c, bool ccd, Slack& sl)
{
    const V3 cn = cross3(sub(b, a), sub(c, a));
    const float m = dot3(cn, cn);
    const float blow = l1n(sub(b, a)) * l1n(sub(c, a)) / (std::sqrt(m) + 1e-30f);
    const V3 tn = m > 0 ? muls(cn, 1.0f / std::sqrt(m)) : V3{ 0, 0, 0 };
    const float rSq = cVolumeRadius * cVolumeRadius;

    const float pd = dot3(sub(cVolumePos, a), tn);
    sl.noteSc(pd, cVolumeRadius, (l1n(cVolumePos) + l1n(a)) * (1.0f + blow) + 1.0f);
    V3 closest;
    if (pd < cVolumeRadius)
    {
        float distSq = distPtTriSq(cVolumePos, a, b, c, closest);
        sl.noteSc(distSq, rSq, l1n(sub(cVolumePos, closest)) * (l1n(cVolumePos) + l1n(closest)) + distSq + rSq + 1e-3f);
        if (distSq <= rSq)
        {
            const float ppd = dot3(sub(particlePos, a), tn);
            sl.noteSc(ppd, 0.0f, (l1n(particlePos) + l1n(a)) * (1.0f + blow) + 1e-3f);
            if (ppd > 0.0f)
            {
                if (ccd)
                    distSq = distPtTriSq(particlePos, a, b, c, closest);
                const V3 dv = sub(particlePos, closest);
                // grazing shield (also covers the defensively-dead exact-zero
                // fallback: distSq == 0 with the back-face cull passed needs
                // closest == particlePos bitwise while the plane distance
                // rounds positive)
                if (distSq != 0.0f) sl.noteSc(std::sqrt(distSq), 0.0f, 1000.0f);
                normal = (distSq > 0.0f) ? muls(dv, 1.0f / std::sqrt(distSq)) : tn;
                distance = std::sqrt(distSq);
                return true;
            }
        }
    }
    return false;
}

PXG_TEST(psPrimContact_all)
{
    std::mt19937 rng(87019);
    std::uniform_real_distribution<float> uv(-1, 1);
    std::uniform_real_distribution<float> u01(0, 1);

    const uint32_t N = 4096;
    std::vector<PcWork> work(N);

    auto randQ = [&]() {
        F4 q = { uv(rng), uv(rng), uv(rng), 1.0f + uv(rng) };
        const float m = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
        q.x /= m; q.y /= m; q.z /= m; q.w /= m;
        return q;
    };
    auto randDir = [&]() {
        for (;;)
        {
            V3 d = { uv(rng), uv(rng), uv(rng) };
            const float m = dot3(d, d);
            if (m > 0.05f) return muls(d, 1.0f / std::sqrt(m));
        }
    };

    for (uint32_t i = 0; i < N; ++i)
    {
        PcWork& w = work[i];
        const uint32_t type = i % 5;
        const bool ccd = (i / 5) % 2 == 1;
        const float cvr = 0.05f + 0.3f * u01(rng);
        w.ids = { type, ccd ? 1u : 0u, 0, 0 };
        w.shapeP = { uv(rng), uv(rng), uv(rng), 0 };
        w.shapeQ = randQ();
        w.scale = { 0.2f + 0.8f * u01(rng), 0.2f + 0.6f * u01(rng), 0.2f + 0.6f * u01(rng), 0 };

        // contact-volume position placed around the shape surface so both
        // intersect outcomes occur; the CCD particle position jitters nearby
        V3 cv;
        const V3 dir = randDir();
        const float surfOff = -0.3f + 1.2f * u01(rng); // signed offset from the surface
        switch (type)
        {
        case 0: // plane through shapeP with normal = rotated x-axis
        {
            const V3 pn = quatRotate(w.shapeQ, V3{ 1, 0, 0 });
            V3 tang = cross3(pn, dir);
            cv = add(add(v3(w.shapeP), muls(pn, surfOff)), muls(tang, u01(rng)));
            break;
        }
        case 1: // sphere, radius scale.x
            cv = add(v3(w.shapeP), muls(dir, std::max(0.0f, w.scale.x + surfOff)));
            if (i % 50 == 1) cv = v3(w.shapeP); // exact center: deterministic eps fallback
            break;
        case 2: // capsule, halfHeight scale.x radius scale.y
        {
            const V3 ax = basis0(w.shapeQ);
            cv = add(add(v3(w.shapeP), muls(ax, w.scale.x * uv(rng))), muls(dir, std::max(0.0f, w.scale.y + surfOff)));
            if (i % 50 == 2) cv = v3(w.shapeP); // exact center
            break;
        }
        case 3: // box, half extents scale.xyz
        {
            if (i % 10 == 3)
            {
                // strictly inside
                cv = add(v3(w.shapeP), quatRotate(w.shapeQ, V3{ 0.8f * w.scale.x * uv(rng), 0.8f * w.scale.y * uv(rng), 0.8f * w.scale.z * uv(rng) }));
            }
            else
            {
                const V3 lp = { w.scale.x * uv(rng), w.scale.y * uv(rng), w.scale.z * uv(rng) };
                cv = add(v3(w.shapeP), quatRotate(w.shapeQ, add(lp, muls(dir, std::max(0.05f, surfOff + 0.3f)))));
            }
            break;
        }
        default: // triangle: verts around shapeP (healthy, blow-bounded)
        {
            V3 a, b, c;
            for (;;)
            {
                a = add(v3(w.shapeP), V3{ 0.4f * uv(rng), 0.4f * uv(rng), 0.4f * uv(rng) });
                b = add(v3(w.shapeP), V3{ 0.4f * uv(rng), 0.4f * uv(rng), 0.4f * uv(rng) });
                c = add(v3(w.shapeP), V3{ 0.4f * uv(rng), 0.4f * uv(rng), 0.4f * uv(rng) });
                const V3 e0 = sub(b, a), e1 = sub(c, a);
                const V3 cn = cross3(e0, e1);
                if (std::sqrt(dot3(cn, cn)) > l1n(e0) * l1n(e1) / 3.0f) break;
            }
            w.shapeP = f4(a);
            w.shapeQ = f4(b);
            w.scale = f4(c);
            // both sides of the plane so the back-face cull fires
            const V3 tc = muls(add(add(a, b), c), 1.0f / 3.0f);
            cv = add(tc, muls(randDir(), 0.4f * u01(rng)));
            break;
        }
        }
        w.cVolume = { cv.x, cv.y, cv.z, cvr };
        const V3 pp = ccd ? add(cv, muls(randDir(), 0.15f * u01(rng))) : cv;
        w.particlePos = f4(pp);
        if ((i % 50 == 1 && type == 1) || (i % 50 == 2 && type == 2))
            w.particlePos = w.cVolume; // keep the exact-center case exact
    }

    // ---- reference ----
    struct Ref { bool intersect = false; V3 normal = { 0, 0, 0 }; float dist = 0;
                 int faceAxis = -1; bool backface = false, epsFb = false; };
    std::vector<Ref> ref(N);
    std::vector<uint8_t> skip(N, 0);

    for (uint32_t i = 0; i < N; ++i)
    {
        const PcWork& w = work[i];
        const uint32_t type = w.ids.x;
        const bool ccd = w.ids.y != 0;
        const V3 pp = v3(w.particlePos), cv = v3(w.cVolume);
        const float cvr = w.cVolume.w;
        Slack sl;
        Ref& r = ref[i];

        switch (type)
        {
        case 0:
        {
            const V3 pn = quatRotate(w.shapeQ, V3{ 1, 0, 0 });
            const float pdd = -dot3(pn, v3(w.shapeP));
            float dist = dot3(pn, cv) + pdd;
            sl.noteSc(dist, cvr, l1n(cv) + l1n(v3(w.shapeP)) + 1.0f);
            if (dist <= cvr)
            {
                if (ccd) dist = dot3(pn, pp) + pdd;
                r.normal = pn;
                r.dist = dist;
                r.intersect = true;
            }
            break;
        }
        case 1:
        {
            r.intersect = refSphere(r.normal, r.dist, cv, v3(w.shapeP), w.scale.x, cvr, sl);
            if (r.intersect && ccd)
                refSphere(r.normal, r.dist, pp, v3(w.shapeP), w.scale.x, MAXF, sl);
            if (r.intersect && r.dist == -w.scale.x) r.epsFb = true;
            break;
        }
        case 2:
        {
            const V3 dirAx = basis0(w.shapeQ);
            r.intersect = refCapsule(r.normal, r.dist, cv, v3(w.shapeP), dirAx, w.scale.y, w.scale.x, cvr, sl);
            if (r.intersect && ccd)
                refCapsule(r.normal, r.dist, pp, v3(w.shapeP), dirAx, w.scale.y, w.scale.x, MAXF, sl);
            if (r.intersect && r.dist == -w.scale.y) r.epsFb = true;
            break;
        }
        case 3:
        {
            r.intersect = refBox(r.normal, r.dist, cv, v3(w.shapeP), w.shapeQ, v3(w.scale), cvr, sl);
            if (r.intersect && ccd)
                refBox(r.normal, r.dist, pp, v3(w.shapeP), w.shapeQ, v3(w.scale), MAXF, sl);
            if (r.intersect && r.dist < 0.0f)
            {
                // record the selected LOCAL face axis
                const V3 ln = quatRotateInv(w.shapeQ, r.normal);
                r.faceAxis = std::fabs(ln.x) > 0.9f ? 0 : (std::fabs(ln.y) > 0.9f ? 1 : 2);
            }
            break;
        }
        default:
        {
            const bool wouldFrontface = [&]() {
                // count back-face blocks for coverage: volume test passes but cull fires
                const V3 a = v3(w.shapeP), b = v3(w.shapeQ), c = v3(w.scale);
                const V3 cn = cross3(sub(b, a), sub(c, a));
                const float m = dot3(cn, cn);
                const V3 tn = m > 0 ? muls(cn, 1.0f / std::sqrt(m)) : V3{ 0, 0, 0 };
                if (!(dot3(sub(cv, a), tn) < cvr)) return false;
                V3 cl;
                if (!(distPtTriSq(cv, a, b, c, cl) <= cvr * cvr)) return false;
                return !(dot3(sub(pp, a), tn) > 0.0f);
            }();
            r.backface = wouldFrontface;
            r.intersect = refTriangle(r.normal, r.dist, pp, cv, cvr, v3(w.shapeP), v3(w.shapeQ), v3(w.scale), ccd, sl);
            break;
        }
        }
        skip[i] = sl.near ? 1 : 0;
    }

    // ---- GPU run ----
    PxgSlangComputeDevice* dev = testDevice();
    std::vector<PcOut> outInit(N);
    PxgBufferHandle bW = createBufferFrom(dev, work);
    PxgBufferHandle bO = createBufferFrom(dev, outInit);
    PcParams prm = { N, 0, 0, 0 };
    Kernel k("psPrimContact.spv", "ctPsPrimContact",
             { (uint32_t)sizeof(PcWork), (uint32_t)sizeof(PcOut) }, sizeof(PcParams));
    k.dispatch((N + 255) / 256, { bW, bO }, prm);
    dev->flush();
    const auto gO = downloadBuffer<PcOut>(dev, bO, N);
    dev->destroyBuffer(bW);
    dev->destroyBuffer(bO);

    size_t mm = 0, checked = 0, skipped = 0;
    size_t hit[5] = { 0 }, miss[5] = { 0 }, ccdHit[5] = { 0 };
    size_t boxInside = 0, boxFace[3] = { 0, 0, 0 }, triBackface = 0, epsFallback = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (skip[i]) { ++skipped; continue; }
        ++checked;
        const Ref& r = ref[i];
        const uint32_t type = work[i].ids.x;
        const bool ccd = work[i].ids.y != 0;
        if (r.intersect) { ++hit[type]; if (ccd) ++ccdHit[type]; }
        else ++miss[type];
        if (r.faceAxis >= 0) { ++boxInside; ++boxFace[r.faceAxis]; }
        if (r.backface) ++triBackface;
        if (r.epsFb) ++epsFallback;
        const bool gi = gO[i].flags.x != 0;
        if (gi != r.intersect) { ++mm; continue; }
        if (r.intersect)
        {
            if (!almostEqual(gO[i].normal_dist.w, r.dist, 1e-4f, 1e-4f) ||
                !almostEqual(gO[i].normal_dist.x, r.normal.x, 1e-4f, 1e-4f) ||
                !almostEqual(gO[i].normal_dist.y, r.normal.y, 1e-4f, 1e-4f) ||
                !almostEqual(gO[i].normal_dist.z, r.normal.z, 1e-4f, 1e-4f)) ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "psPrimContact: %zu mismatches", mm);
    PXG_CHECK_MSG(checked > N * 9 / 10, "psPrimContact: too many skipped (%zu)", skipped);
    bool cov = true;
    for (int t = 0; t < 5; ++t)
        cov = cov && hit[t] > 40 && miss[t] > 40 && ccdHit[t] > 15;
    PXG_CHECK_MSG(cov, "psPrimContact: type coverage thin (h %zu/%zu/%zu/%zu/%zu m %zu/%zu/%zu/%zu/%zu c %zu/%zu/%zu/%zu/%zu)",
                  hit[0], hit[1], hit[2], hit[3], hit[4], miss[0], miss[1], miss[2], miss[3], miss[4],
                  ccdHit[0], ccdHit[1], ccdHit[2], ccdHit[3], ccdHit[4]);
    PXG_CHECK_MSG(boxInside > 10 && boxFace[0] >= 6 && boxFace[1] >= 6 && boxFace[2] >= 6 &&
                  triBackface > 10 && epsFallback >= 2,
                  "psPrimContact: branch coverage thin (in=%zu f=%zu/%zu/%zu bf=%zu eps=%zu)",
                  boxInside, boxFace[0], boxFace[1], boxFace[2], triBackface, epsFallback);
}

} // namespace
