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

// Verification of npCvxCoreGjk.slang: Gu::RefGjkEpa::computeGjkDistance
// (GuRefGjkEpa.h L44-419) over ConvexShape::supportLocal (margin-less).
// Exact CPU transcription: the Convex<Support> weld cache (MAX_VERTS 32,
// accuracy = max(1% local max extent, FLT_EPSILON), BACKWARD weld scan with
// early-out, strict closest tracking, full-cache fallback), GjkDistance
// (seed dir from bounds centers with the (1,0,0) fallback; addPoint's
// iteration cap 100, duplicate-(aI,bI) FLT_MAX out; the four Johnson
// computeClosest cases with the abc.dot(a) > 0 swap; the improvement test
// |closest| < |mClosest| - FLT_EPSILON with the --mNumVerts backtrack) and
// the driver loop (distEps = 0.01 * min accuracy; 0 / dist / FLT_MAX).
// All arithmetic is add/mul/div/sqrt: outputs compare bitwise, except the
// signs of exact zeros (the tranche-164 llvmpipe NSZ exception class: the
// reference carries CUDA's PxMat33 diagonal ellipsoid form and IEEE signed
// zeros that llvmpipe's NSZ folding cannot reproduce -- every magnitude bit
// must still match, only the sign of exact 0.0f is tolerated).

#include "pxgTestFramework.h"

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };
struct V3 { float x, y, z; };

struct GjkIn
{
    U4 metaA;
    U4 coreA[3];
    F4 poseAq, poseAp;
    U4 metaB;
    U4 coreB[3];
    F4 poseBq, poseBp;
    F4 cfg;       // x = maxDist
    F4 ptsA[12];
    F4 ptsB[12];
};
struct GjkOut
{
    F4 result;    // x = return value, y = tag bits (0 overlap, 1 dist, 2 maxDist)
    F4 pointA, pointB, axis;
};

const uint32_t CORE_POINT = 0, CORE_SEGMENT = 1, CORE_BOX = 2, CORE_ELLIPSOID = 3,
               CORE_CYLINDER = 4, CORE_CONE = 5, CORE_POINTS = 6;
const uint32_t MAX_VERTS = 32, MAX_ITERATIONS = 100;

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static float pxSign(float a) { return (a >= 0.f) ? 1.f : -1.f; }
static V3 vadd(V3 a, V3 b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 vsub(V3 a, V3 b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 vscl(V3 a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; }
static V3 vneg(V3 a) { return V3{ -a.x, -a.y, -a.z }; }
static float vdot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 vcross(V3 a, V3 b)
{ return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static float magSq(V3 v) { return v.x * v.x + v.y * v.y + v.z * v.z; }
static float mag(V3 v) { return std::sqrt(magSq(v)); }
static V3 nrmz(V3 v) // PxVec3::getNormalized
{
    const float m = magSq(v);
    if (m > 0) { const float i = 1.f / std::sqrt(m); return V3{ v.x * i, v.y * i, v.z * i }; }
    return V3{ 0, 0, 0 };
}
static float normalizeIn(V3& v) // PxVec3::normalize: guarded reciprocal multiply
{
    const float m = mag(v);
    if (m > 0) { const float i = 1.f / m; v = V3{ v.x * i, v.y * i, v.z * i }; }
    return m;
}
static V3 qRot(const F4& q, V3 v)
{
    const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z;
    const float w2 = q.w * q.w - 0.5f, d2 = q.x * vx + q.y * vy + q.z * vz;
    return V3{ vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d2,
               vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d2,
               vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d2 };
}
static V3 qRotInv(const F4& q, V3 v)
{
    const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z;
    const float w2 = q.w * q.w - 0.5f, d2 = q.x * vx + q.y * vy + q.z * vz;
    return V3{ vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * d2,
               vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * d2,
               vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * d2 };
}
// PxQuat::getInvBasisVector0/1/2 (PxQuat.h L298-319)
static V3 invBasis0(const F4& q)
{
    const float x2 = q.x * 2, w2 = q.w * 2;
    return V3{ (q.w * w2) - 1.f + q.x * x2, (-q.z * w2) + q.y * x2, (q.y * w2) + q.z * x2 };
}
static V3 invBasis1(const F4& q)
{
    const float y2 = q.y * 2, w2 = q.w * 2;
    return V3{ (q.z * w2) + q.x * y2, (q.w * w2) - 1.f + q.y * y2, (-q.x * w2) + q.z * y2 };
}
static V3 invBasis2(const F4& q)
{
    const float z2 = q.z * 2, w2 = q.w * 2;
    return V3{ (-q.y * w2) + q.x * z2, (q.x * w2) + q.y * z2, (q.w * w2) - 1.f + q.z * z2 };
}

struct ShapeRefC
{
    uint32_t type;
    const U4* core;
    F4 poseQ;
    V3 poseP;
    const F4* pool;
};

static float coreF(const ShapeRefC& s, uint32_t w) { return u2f((&s.core[w / 4].x)[w & 3]); }

// ConvexShape::supportLocal == ConvexCore::localSupport (no margin) -- the
// exact tranche-164 transcription incl. the PxMat33 diagonal ellipsoid form
static V3 supportLocal(const ShapeRefC& s, V3 dir)
{
    const uint32_t type = s.type;
    if (type == CORE_SEGMENT)
        return V3{ pxSign(dir.x) * coreF(s, 0) * 0.5f, 0, 0 };
    if (type == CORE_BOX)
        return V3{ pxSign(dir.x) * coreF(s, 0) * 0.5f,
                   pxSign(dir.y) * coreF(s, 1) * 0.5f,
                   pxSign(dir.z) * coreF(s, 2) * 0.5f };
    if (type == CORE_ELLIPSOID)
    {
        const V3 r{ coreF(s, 0), coreF(s, 1), coreF(s, 2) };
        const V3 tt{ (r.x * dir.x + 0.0f * dir.y) + 0.0f * dir.z,
                     (0.0f * dir.x + r.y * dir.y) + 0.0f * dir.z,
                     (0.0f * dir.x + 0.0f * dir.y) + r.z * dir.z };
        const V3 d1 = nrmz(tt);
        return V3{ (r.x * d1.x + 0.0f * d1.y) + 0.0f * d1.z,
                   (0.0f * d1.x + r.y * d1.y) + 0.0f * d1.z,
                   (0.0f * d1.x + 0.0f * d1.y) + r.z * d1.z };
    }
    if (type == CORE_CYLINDER)
    {
        const float h = coreF(s, 0) * 0.5f, r = coreF(s, 1);
        const V3 d = nrmz(dir);
        const float eps = 1e-5f;
        if (std::fabs(d.y) < eps && std::fabs(d.z) < eps) return V3{ pxSign(d.x) * h, r, 0 };
        const V3 dr = nrmz(V3{ 0, d.y, d.z });
        // CUDA form is PxVec3(sign*h,0,0) + dr*r: keep the 0 + adds (review
        // 169: they flip -0 radial components to +0, the eqFZ-visible side)
        return V3{ pxSign(d.x) * h + dr.x * r, 0.0f + dr.y * r, 0.0f + dr.z * r };
    }
    if (type == CORE_CONE)
    {
        const float h = coreF(s, 0), r = coreF(s, 1);
        const float sinA = r / std::sqrt(h * h + r * r);
        const float halfH = h * 0.5f;
        const V3 d = nrmz(dir);
        const float eps = 1e-5f;
        if (d.x > sinA) return V3{ halfH, 0, 0 };
        if (std::fabs(d.y) < eps && std::fabs(d.z) < eps) return V3{ -halfH, r, 0 };
        const V3 dr = nrmz(V3{ 0, d.y, d.z });
        return V3{ -halfH + dr.x * r, 0.0f + dr.y * r, 0.0f + dr.z * r };
    }
    if (type == CORE_POINTS)
    {
        const uint32_t w9 = s.core[2].y;
        const uint32_t numPoints = w9 & 0xFF, stride = (w9 >> 8) & 0xFF;
        if ((s.core[0].x == 0 && s.core[0].y == 0) || numPoints == 0) return V3{ 0, 0, 0 };
        const V3 S{ u2f(s.core[0].z), u2f(s.core[0].w), u2f(s.core[1].x) };
        const F4 R{ u2f(s.core[1].y), u2f(s.core[1].z), u2f(s.core[1].w), u2f(s.core[2].x) };

        const V3 rd = qRot(R, dir);
        const V3 d = qRotInv(R, V3{ S.x * rd.x, S.y * rd.y, S.z * rd.z });
        if (mag(d) < FLT_EPSILON) return V3{ 0, 0, 0 };

        uint32_t index = 0;
        float maxDot = -FLT_MAX;
        for (uint32_t i = 0; i < numPoints; ++i)
        {
            const F4& p = s.pool[(i * stride) / 16];
            const float dot = d.x * p.x + d.y * p.y + d.z * p.z;
            if (dot > maxDot) { maxDot = dot; index = i; }
        }
        const F4& pt = s.pool[(index * stride) / 16];
        const V3 rp = qRot(R, V3{ pt.x, pt.y, pt.z });
        return qRotInv(R, V3{ S.x * rp.x, S.y * rp.y, S.z * rp.z });
    }
    return V3{ 0, 0, 0 }; // CORE_POINT
}

static V3 poseTf(const ShapeRefC& s, V3 v) { return vadd(qRot(s.poseQ, v), s.poseP); }

struct RefStats
{
    uint32_t tag = 0;
    bool fallback = false, weld = false, dup = false, iterCap = false, cacheFull = false;
    uint32_t iters = 0;
    uint32_t maxCache = 0; // probe only
};

struct ConvexC
{
    ShapeRefC shape;
    float accuracy;
    uint32_t numVerts;
    V3 verts[MAX_VERTS];
};

// Convex<Support> ctor (GuRefGjkEpa.h L49-59)
static float convexAccuracy(const ShapeRefC& s)
{
    const V3 X{ 1, 0, 0 }, Y{ 0, 1, 0 }, Z{ 0, 0, 1 };
    const V3 mn{ supportLocal(s, vneg(X)).x, supportLocal(s, vneg(Y)).y, supportLocal(s, vneg(Z)).z };
    const V3 mx{ supportLocal(s, X).x, supportLocal(s, Y).y, supportLocal(s, Z).z };
    const V3 dim = vsub(mx, mn);
    const float maxExtent = std::fmax(dim.x, std::fmax(dim.y, dim.z));
    return std::fmax(maxExtent * 0.01f, FLT_EPSILON);
}

// Convex<Support>::getBounds (L65-78)
static void convexBounds(const ShapeRefC& s, V3& mn, V3& mx)
{
    const V3 X = invBasis0(s.poseQ), Y = invBasis1(s.poseQ), Z = invBasis2(s.poseQ);
    mn = V3{ poseTf(s, supportLocal(s, vneg(X))).x,
             poseTf(s, supportLocal(s, vneg(Y))).y,
             poseTf(s, supportLocal(s, vneg(Z))).z };
    mx = V3{ poseTf(s, supportLocal(s, X)).x,
             poseTf(s, supportLocal(s, Y)).y,
             poseTf(s, supportLocal(s, Z)).z };
}

// Convex<Support>::supportIndex (L80-110)
static uint32_t supportIndex(ConvexC& c, V3 dir, RefStats& st)
{
    const V3 v = poseTf(c.shape, supportLocal(c.shape, qRotInv(c.shape.poseQ, dir)));
    uint32_t closest = MAX_VERTS;
    const float WELD_DIST_SQ = c.accuracy * c.accuracy;
    float distSq = FLT_MAX;
    for (int32_t i = int32_t(c.numVerts) - 1; i >= 0; --i)
    {
        const float dSq = magSq(vsub(c.verts[i], v));
        if (dSq < WELD_DIST_SQ) { st.weld = true; return uint32_t(i); }
        if (distSq > dSq) { distSq = dSq; closest = uint32_t(i); }
    }
    if (c.numVerts == MAX_VERTS) { st.cacheFull = true; return closest; }
    c.verts[c.numVerts++] = v;
    return c.numVerts - 1;
}

struct Vert { V3 p; float s; uint32_t aI, bI; };

// GjkDistance state + computeClosest + addPoint + driver (L129-419)
static void refGjk(const GjkIn& in, GjkOut& o, RefStats& st)
{
    std::memset(&o, 0, sizeof(o));

    ShapeRefC sA{ in.metaA.x, in.coreA, in.poseAq, V3{ in.poseAp.x, in.poseAp.y, in.poseAp.z }, in.ptsA };
    ShapeRefC sB{ in.metaB.x, in.coreB, in.poseBq, V3{ in.poseBp.x, in.poseBp.y, in.poseBp.z }, in.ptsB };

    ConvexC cA, cB;
    cA.shape = sA; cA.numVerts = 0; cA.accuracy = convexAccuracy(sA);
    cB.shape = sB; cB.numVerts = 0; cB.accuracy = convexAccuracy(sB);

    Vert verts[4];
    for (int i = 0; i < 4; ++i) verts[i] = Vert{ V3{ 0, 0, 0 }, 0, 0, 0 };
    uint32_t numVerts = 0;
    uint32_t iteration = 0;
    V3 mClosest{ FLT_MAX, FLT_MAX, FLT_MAX };

    V3 mnA, mxA, mnB, mxB;
    convexBounds(sA, mnA, mxA);
    convexBounds(sB, mnB, mxB);
    V3 dir0 = vsub(vscl(vadd(mnB, mxB), 0.5f), vscl(vadd(mnA, mxA), 0.5f));
    if (normalizeIn(dir0) < FLT_EPSILON) { dir0 = V3{ 1, 0, 0 }; st.fallback = true; }

    // addPoint (L155-173)
    auto addPoint = [&](V3 dir) -> float
    {
        if (++iteration > MAX_ITERATIONS) { st.iterCap = true; return FLT_MAX; }
        const uint32_t aI = supportIndex(cA, dir, st);
        const uint32_t bI = supportIndex(cB, vneg(dir), st);
        const V3 p = vsub(cA.verts[aI], cB.verts[bI]);
        for (uint32_t i = 0; i < numVerts; ++i)
            if (verts[i].aI == aI && verts[i].bI == bI) { st.dup = true; return FLT_MAX; }
        verts[numVerts++] = Vert{ p, 0.0f, aI, bI };
        return vdot(p, vneg(dir));
    };

    addPoint(dir0); // ctor: result discarded

    const float distEps = 0.01f * std::fmin(cA.accuracy, cB.accuracy);
    const float maxDist = in.cfg.x;

    while (true)
    {
        ++st.iters;
        // computeClosest (L175-342)
        Vert w[4] = { verts[0], verts[1], verts[2], verts[3] };
        w[0].s = w[1].s = w[2].s = w[3].s = 0;
        uint32_t nv = numVerts;

        switch (nv)
        {
        case 1:
        {
            w[0].s = 1.0f;
            break;
        }
        case 2:
        {
            const V3 a = w[0].p, b = w[1].p;
            const V3 ba = vsub(a, b);
            const float sba = vdot(ba, vneg(b));
            if (sba <= 0)
            {
                w[0] = w[1];
                w[0].s = 1.0f;
                nv = 1;
                break;
            }
            w[0].s = sba / magSq(ba);
            w[1].s = 1.0f - w[0].s;
            break;
        }
        case 3:
        {
            const V3 a = w[0].p, b = w[1].p, c = w[2].p;
            const V3 ca = vsub(a, c), cb = vsub(b, c);
            const float sca = vdot(ca, vneg(c)), scb = vdot(cb, vneg(c));
            if (sca <= 0 && scb <= 0)
            {
                w[0] = w[2];
                w[0].s = 1.0f;
                nv = 1;
                break;
            }
            const V3 abc = vcross(vsub(b, a), vsub(c, a));
            const float iabc = 1.0f / magSq(abc);
            const V3 pabc = vscl(vscl(abc, vdot(abc, a)), iabc);
            const float tca = vdot(abc, vcross(vsub(c, pabc), vsub(a, pabc)));
            if (sca > 0 && tca <= 0)
            {
                w[1] = w[2];
                w[0].s = sca / magSq(ca);
                w[1].s = 1.0f - w[0].s;
                nv = 2;
                break;
            }
            const float tbc = vdot(abc, vcross(vsub(b, pabc), vsub(c, pabc)));
            if (scb > 0 && tbc <= 0)
            {
                w[0] = w[1];
                w[1] = w[2];
                w[0].s = scb / magSq(cb);
                w[1].s = 1.0f - w[0].s;
                nv = 2;
                break;
            }
            w[0].s = tbc * iabc;
            w[1].s = tca * iabc;
            w[2].s = 1.0f - w[0].s - w[1].s;
            if (vdot(abc, a) > 0) { const Vert tv = w[0]; w[0] = w[1]; w[1] = tv; }
            break;
        }
        case 4:
        {
            const V3 a = w[0].p, b = w[1].p, c = w[2].p, d = w[3].p;
            const V3 da = vsub(a, d), db = vsub(b, d), dc = vsub(c, d);
            const float sda = vdot(da, vneg(d)), sdb = vdot(db, vneg(d)), sdc = vdot(dc, vneg(d));
            if (sda <= 0 && sdb <= 0 && sdc <= 0)
            {
                w[0] = w[3];
                w[0].s = 1.0f;
                nv = 1;
                break;
            }
            const V3 dab = vcross(vsub(a, d), vsub(b, d));
            const float idab = 1.0f / magSq(dab);
            const V3 pdab = vscl(vscl(dab, vdot(dab, d)), idab);
            const V3 dbc = vcross(vsub(b, d), vsub(c, d));
            const float idbc = 1.0f / magSq(dbc);
            const V3 pdbc = vscl(vscl(dbc, vdot(dbc, d)), idbc);
            const V3 dca = vcross(vsub(c, d), vsub(a, d));
            const float idca = 1.0f / magSq(dca);
            const V3 pdca = vscl(vscl(dca, vdot(dca, d)), idca);
            const float tda = vdot(dab, vcross(vsub(d, pdab), vsub(a, pdab)));
            const float tad = vdot(dca, vcross(vsub(a, pdca), vsub(d, pdca)));
            if (sda > 0 && tda <= 0 && tad <= 0)
            {
                w[1] = w[3];
                w[0].s = sda / magSq(da);
                w[1].s = 1.0f - w[0].s;
                nv = 2;
                break;
            }
            const float tdb = vdot(dbc, vcross(vsub(d, pdbc), vsub(b, pdbc)));
            const float tbd = vdot(dab, vcross(vsub(b, pdab), vsub(d, pdab)));
            if (sdb > 0 && tdb <= 0 && tbd <= 0)
            {
                w[0] = w[1];
                w[1] = w[3];
                w[0].s = sdb / magSq(db);
                w[1].s = 1.0f - w[0].s;
                nv = 2;
                break;
            }
            const float tcd = vdot(dbc, vcross(vsub(c, pdbc), vsub(d, pdbc)));
            const float tdc = vdot(dca, vcross(vsub(d, pdca), vsub(c, pdca)));
            if (sdc > 0 && tdc <= 0 && tcd <= 0)
            {
                w[0] = w[2];
                w[1] = w[3];
                w[0].s = sdc / magSq(dc);
                w[1].s = 1.0f - w[0].s;
                nv = 2;
                break;
            }
            const V3 abc = vcross(vsub(b, a), vsub(c, a));
            if (tda > 0 && tbd > 0 && vdot(dab, d) < 0 && vdot(abc, dab) > 0)
            {
                w[2] = w[3];
                w[0].s = tbd * idab;
                w[1].s = tda * idab;
                w[2].s = 1.0f - w[0].s - w[1].s;
                nv = 3;
                break;
            }
            if (tdb > 0 && tcd > 0 && vdot(dbc, d) < 0 && vdot(abc, dbc) > 0)
            {
                w[0] = w[1];
                w[1] = w[2];
                w[2] = w[3];
                w[0].s = tcd * idbc;
                w[1].s = tdb * idbc;
                w[2].s = 1.0f - w[0].s - w[1].s;
                nv = 3;
                break;
            }
            if (tdc > 0 && tad > 0 && vdot(dca, d) < 0 && vdot(abc, dca) > 0)
            {
                w[1] = w[3];
                w[0].s = tdc * idca;
                w[2].s = tad * idca;
                w[1].s = 1.0f - w[0].s - w[2].s;
                nv = 3;
                break;
            }
            break;
        }
        }

        // (t0+t1)+(t2+t3) vs CUDA's ((t0+t1)+t2)+t3: w[3].s is never set by
        // any case (always 0), so t3 is an exact +-0 and both associations
        // agree bitwise in every reachable configuration (review 169)
        const V3 closest = vadd(vadd(vscl(w[0].p, w[0].s), vscl(w[1].p, w[1].s)),
                                vadd(vscl(w[2].p, w[2].s), vscl(w[3].p, w[3].s)));
        if (mag(closest) < mag(mClosest) - FLT_EPSILON)
        {
            mClosest = closest;
            verts[0] = w[0]; verts[1] = w[1]; verts[2] = w[2]; verts[3] = w[3];
            numVerts = nv;
        }
        else
            --numVerts;

        // driver (L396-418)
        const float closestDist = mag(mClosest);
        if (closestDist < distEps)
        {
            o.result = F4{ 0.0f, u2f(0), 0, 0 };
            st.tag = 0;
            return;
        }

        const V3 dir = vneg(nrmz(mClosest));
        const float proj = addPoint(dir);

        if (proj >= closestDist - distEps)
        {
            V3 pA{ 0, 0, 0 }, pB{ 0, 0, 0 };
            for (uint32_t i = 0; i < numVerts; ++i)
            {
                pA = vadd(pA, vscl(cA.verts[verts[i].aI], verts[i].s));
                pB = vadd(pB, vscl(cB.verts[verts[i].bI], verts[i].s));
            }
            o.result = F4{ closestDist, u2f(1), 0, 0 };
            o.pointA = F4{ pA.x, pA.y, pA.z, 0 };
            o.pointB = F4{ pB.x, pB.y, pB.z, 0 };
            o.axis = F4{ -dir.x, -dir.y, -dir.z, 0 };
            st.tag = 1;
            return;
        }

        st.maxCache = cA.numVerts > cB.numVerts ? cA.numVerts : cB.numVerts;
        if (proj > maxDist)
        {
            o.result = F4{ FLT_MAX, u2f(2), 0, 0 };
            st.tag = 2;
            return;
        }
    }
}

// bitwise, tolerating only the SIGN of exact zeros (tranche-164 NSZ class)
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b) || (a == 0.f && b == 0.f); }
static bool eqF4Z(const F4& a, const F4& b)
{ return eqFZ(a.x, b.x) && eqFZ(a.y, b.y) && eqFZ(a.z, b.z) && eqFZ(a.w, b.w); }

static void fillCore(std::mt19937& rng, uint32_t type, U4 core[3], F4 pts[12],
                     std::function<float(float, float)> rf)
{
    std::memset(core, 0, sizeof(U4) * 3);
    if (type == CORE_SEGMENT) core[0].x = f2u(rf(0.2f, 4));
    else if (type == CORE_BOX || type == CORE_ELLIPSOID)
    { core[0].x = f2u(rf(0.1f, 3)); core[0].y = f2u(rf(0.1f, 3)); core[0].z = f2u(rf(0.1f, 3)); }
    else if (type == CORE_CYLINDER || type == CORE_CONE)
    { core[0].x = f2u(rf(0.2f, 3)); core[0].y = f2u(rf(0.1f, 2)); }
    else if (type == CORE_POINTS)
    {
        const uint32_t np = 3 + rng() % 10; // <= 12: the GjkIn pool is 12 float4s
        core[0].x = 48; // non-NULL pointer image
        core[0].z = f2u(rf(0.3f, 2)); core[0].w = f2u(rf(0.3f, 2));
        core[1].x = f2u(rf(0.3f, 2));
        float a[4]; float m2 = 0;
        for (int k = 0; k < 4; ++k) { a[k] = rf(-1, 1); m2 += a[k] * a[k]; }
        const float qi = 1.f / std::sqrt(m2 > 0 ? m2 : 1.f);
        core[1].y = f2u(a[0] * qi); core[1].z = f2u(a[1] * qi); core[1].w = f2u(a[2] * qi);
        core[2].x = f2u(a[3] * qi);
        core[2].y = np | (16u << 8); // stride 16
        for (uint32_t p = 0; p < 12; ++p)
            pts[p] = F4{ rf(-1.5f, 1.5f), rf(-1.5f, 1.5f), rf(-1.5f, 1.5f), 0 };
    }
}

PXG_TEST(npCvxCoreGjk_distance)
{
    std::mt19937 rng(169001);
    const uint32_t N = 4000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto randQuat = [&]() { float a[4]; float m2 = 0; for (int k = 0; k < 4; ++k) { a[k] = rf(-1, 1); m2 += a[k] * a[k]; } const float i = 1.f / std::sqrt(m2 > 0 ? m2 : 1.f); return F4{ a[0] * i, a[1] * i, a[2] * i, a[3] * i }; };

    std::vector<GjkIn> in(N);
    std::vector<GjkOut> ref(N);
    size_t cType[7] = { 0, 0, 0, 0, 0, 0, 0 };
    size_t cTag[3] = { 0, 0, 0 };
    size_t cFallback = 0, cWeld = 0, cDup = 0, cIterCap = 0, cFull = 0, cNear = 0, cPtPt = 0;
    size_t cNullPts = 0, cTinyS = 0;
    uint32_t gMaxCache = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        GjkIn s{};
        uint32_t typeA = rng() % 7, typeB = rng() % 7;

        s.poseAq = randQuat();
        s.poseBq = randQuat();
        const V3 pA{ rf(-1, 1), rf(-1, 1), rf(-1, 1) };
        V3 pB;

        // separation classes: overlap / gap within maxDist / beyond maxDist /
        // near-touch around the distEps convergence window
        const uint32_t cls = i % 4;
        float maxDist = 1e6f;
        float sep = 0;
        if (cls == 0) sep = rf(0.f, 0.4f);           // deep overlap of typical cores
        else if (cls == 1) sep = rf(4.5f, 8.f);      // clear gap
        else if (cls == 2) { sep = rf(5.f, 9.f); maxDist = rf(0.005f, 0.05f); }
        else { sep = rf(3.8f, 4.4f); ++cNear; }      // shells graze near the surface sum
        V3 sepDir{ rf(-1, 1), rf(-1, 1), rf(-1, 1) };
        if (normalizeIn(sepDir) == 0) sepDir = V3{ 1, 0, 0 };
        pB = vadd(pA, vscl(sepDir, sep));

        if (i % 19 == 7)
        {
            // identical pose + shape: zero center difference -> the (1,0,0)
            // seed fallback (and overlap)
            typeB = typeA;
            s.poseBq = s.poseAq;
            pB = pA;
        }
        if (i % 23 == 11)
        {
            typeA = typeB = CORE_POINT; // point vs point: pure pose distance
            ++cPtPt;
        }
        if (i % 13 == 5)
        {
            // sphere-sphere graze: known surface distance sep - rA - rB
            typeA = typeB = CORE_ELLIPSOID;
            const float rA = rf(0.5f, 1.5f), rB = rf(0.5f, 1.5f);
            fillCore(rng, typeA, s.coreA, s.ptsA, rf);
            fillCore(rng, typeB, s.coreB, s.ptsB, rf);
            s.coreA[0].x = s.coreA[0].y = s.coreA[0].z = f2u(rA);
            s.coreB[0].x = s.coreB[0].y = s.coreB[0].z = f2u(rB);
            pB = vadd(pA, vscl(sepDir, rA + rB + rf(-2e-3f, 2e-2f)));
            s.metaA = U4{ typeA, 0, 0, 0 };
            s.metaB = U4{ typeB, 0, 0, 0 };
            s.poseAp = F4{ pA.x, pA.y, pA.z, 0 };
            s.poseBp = F4{ pB.x, pB.y, pB.z, 0 };
            s.cfg = F4{ maxDist, 0, 0, 0 };
            in[i] = s;
            ++cType[s.metaA.x]; ++cType[s.metaB.x];
            RefStats st;
            refGjk(s, ref[i], st);
            ++cTag[st.tag];
            if (st.fallback) ++cFallback;
            if (st.weld) ++cWeld;
            if (st.dup) ++cDup;
            if (st.iterCap) ++cIterCap;
            if (st.cacheFull) ++cFull; if (st.maxCache > gMaxCache) gMaxCache = st.maxCache;
            continue;
        }

        s.metaA = U4{ typeA, 0, 0, 0 };
        s.metaB = U4{ typeB, 0, 0, 0 };
        fillCore(rng, typeA, s.coreA, s.ptsA, rf);
        fillCore(rng, typeB, s.coreB, s.ptsB, rf);
        if (i % 19 == 7)
        { std::memcpy(s.coreB, s.coreA, sizeof(s.coreA)); std::memcpy(s.ptsB, s.ptsA, sizeof(s.ptsA)); }

        s.poseAp = F4{ pA.x, pA.y, pA.z, 0 };
        s.poseBp = F4{ pB.x, pB.y, pB.z, 0 };
        s.cfg = F4{ maxDist, 0, 0, 0 };

        if (i % 61 == 13)
        {
            // review-169 gap: the ePOINTS 64-bit NULL-pointer image branch
            typeA = CORE_POINTS;
            s.metaA = U4{ typeA, 0, 0, 0 };
            fillCore(rng, typeA, s.coreA, s.ptsA, rf);
            s.coreA[0].x = 0; s.coreA[0].y = 0; // NULL: supportLocal -> (0,0,0)
            ++cNullPts;
        }
        if (i % 67 == 29)
        {
            // review-169 gap: the |d| < FLT_EPSILON degenerate-scale guard --
            // S ~ 1e-8 collapses S*(R*dir) below FLT_EPSILON for every unit
            // query dir, so the guard (not the argmax) must answer
            typeB = CORE_POINTS;
            s.metaB = U4{ typeB, 0, 0, 0 };
            fillCore(rng, typeB, s.coreB, s.ptsB, rf);
            s.coreB[0].z = f2u(1e-8f); s.coreB[0].w = f2u(1e-8f);
            s.coreB[1].x = f2u(1e-8f);
            ++cTinyS;
        }
        if (i % 97 == 31)
        {
            // fat coaxial cylinder/cone caps facing across a tiny gap, with a
            // micro tilt: the near-parallel flat faces make the support dir
            // wander around the cap rim, piling distinct verts into the weld
            // cache (the MAX_VERTS == 32 fallback path)
            typeA = CORE_CYLINDER;
            typeB = (rng() % 2) ? CORE_CYLINDER : CORE_CONE;
            s.metaA = U4{ typeA, 0, 0, 0 };
            s.metaB = U4{ typeB, 0, 0, 0 };
            std::memset(s.coreA, 0, sizeof(s.coreA));
            std::memset(s.coreB, 0, sizeof(s.coreB));
            const float hA = rf(0.5f, 1.f), rA = rf(0.8f, 1.6f);
            const float hB = rf(0.5f, 1.f), rB = rf(0.8f, 1.6f);
            s.coreA[0].x = f2u(hA); s.coreA[0].y = f2u(rA);
            s.coreB[0].x = f2u(hB); s.coreB[0].y = f2u(rB);
            const float tilt = rf(2e-4f, 2e-3f); // ~radians, around z
            s.poseAq = F4{ 0, 0, 0, 1 };
            s.poseBq = F4{ 0, 0, std::sin(tilt * 0.5f), std::cos(tilt * 0.5f) };
            const float gap = rf(1e-4f, 2e-3f);
            s.poseAp = F4{ pA.x, pA.y, pA.z, 0 };
            s.poseBp = F4{ pA.x + hA * 0.5f + hB * 0.5f + gap, pA.y, pA.z, 0 };
            s.cfg = F4{ 1e6f, 0, 0, 0 };
        }
        if (i % 1499 == 700)
        {
            // hill-climbed ellipsoid pair (scratchpad gjkprobe3, seed 1234)
            // whose GJK trajectory pushes a convex's weld cache to ALL 32
            // slots: the only known fixture reaching the numVerts ==
            // MAX_VERTS closest-fallback branch of supportIndex.  Exact bit
            // pattern embedded -- any perturbation loses the property.
            static const uint32_t CF32[52] = {
                0x00000003u, 0u, 0u, 0u,
                0x3f302eb1u, 0x3fee0a4cu, 0x404aec78u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                0xbf5abdeeu, 0x3eea2676u, 0xbcdc360bu, 0xbe7ae34cu,
                0xbf6de52au, 0x3f798d8au, 0xbf025a8au, 0u,
                0x00000003u, 0u, 0u, 0u,
                0x3f1c37b0u, 0x40bc5838u, 0x40efc7c0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                0xbf73389bu, 0xbe3d4d4eu, 0x3ca9b731u, 0x3e803e8bu,
                0x3f901e37u, 0x40391c0au, 0x3e8a1ed0u, 0u,
                0x49742400u, 0u, 0u, 0u };
            std::memset(&s, 0, sizeof(s));
            std::memcpy(&s, CF32, sizeof(CF32));
        }
        if (i % 1499 == 1100)
        {
            // CF32 sibling (scratchpad gjkprobe5, seed 99): the weld cache
            // fills and the full-cache CLOSEST fallback then executes on a
            // LIVE call -- its returned index feeds a vert that changes the
            // remaining trajectory, so both the closest-tracking and the
            // return-closest form are pinned (fullExec = 2).
            static const uint32_t CF32B[52] = {
                0x00000003u, 0u, 0u, 0u,
                0x3f302eb1u, 0x3fee0a4cu, 0x404aec78u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                0xbf5abdf5u, 0x3eea267eu, 0xbcdc0ad8u, 0xbe7ae354u,
                0xbf6de52au, 0x3f798d8au, 0xbf025a8au, 0u,
                0x00000003u, 0u, 0u, 0u,
                0x3f1c37b0u, 0x40bc5838u, 0x40efc7c0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                0xbf73389cu, 0xbe3d4d4fu, 0x3ca9b732u, 0x3e803e8cu,
                0x3f901e37u, 0x40391c0au, 0x3e8a1ed0u, 0u,
                0x49742400u, 0u, 0u, 0u };
            std::memset(&s, 0, sizeof(s));
            std::memcpy(&s, CF32B, sizeof(CF32B));
        }
        if (i % 1499 == 250)
        {
            // the CF32 pair again with maxDist INSIDE its proj ramp: the
            // final distance is D = 0.00222044718 and the slow convergence
            // pushes proj through (0.7 D, D) over many passes, so the run
            // exits via proj > maxDist (tag 2) at 0.7 D exactly where a
            // loosened maxDist test would instead converge to tag 1.
            static const uint32_t CF32M[52] = {
                0x00000003u, 0u, 0u, 0u,
                0x3f302eb1u, 0x3fee0a4cu, 0x404aec78u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                0xbf5abdeeu, 0x3eea2676u, 0xbcdc360bu, 0xbe7ae34cu,
                0xbf6de52au, 0x3f798d8au, 0xbf025a8au, 0u,
                0x00000003u, 0u, 0u, 0u,
                0x3f1c37b0u, 0x40bc5838u, 0x40efc7c0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                0xbf73389bu, 0xbe3d4d4eu, 0x3ca9b731u, 0x3e803e8bu,
                0x3f901e37u, 0x40391c0au, 0x3e8a1ed0u, 0u,
                0u, 0u, 0u, 0u };
            std::memset(&s, 0, sizeof(s));
            std::memcpy(&s, CF32M, sizeof(CF32M));
            s.cfg = F4{ 0.0015543f, 0, 0, 0 }; // 0.7 * D
        }
        if (i % 89 == 47)
        {
            // near-parallel skew NEEDLE ellipsoids with a tiny lateral gap:
            // the support point is extremely sensitive to the direction's
            // x-component and slides along the long axis by more than the
            // weld radius each iteration -- the creep piles distinct verts
            // into the cache toward the MAX_VERTS == 32 fallback
            typeA = typeB = CORE_ELLIPSOID;
            s.metaA = U4{ typeA, 0, 0, 0 };
            s.metaB = U4{ typeB, 0, 0, 0 };
            std::memset(s.coreA, 0, sizeof(s.coreA));
            std::memset(s.coreB, 0, sizeof(s.coreB));
            const float LA = rf(2.5f, 3.5f), rA = rf(0.03f, 0.06f);
            const float LB = rf(2.5f, 3.5f), rB = rf(0.03f, 0.06f);
            s.coreA[0].x = f2u(LA); s.coreA[0].y = f2u(rA); s.coreA[0].z = f2u(rA);
            s.coreB[0].x = f2u(LB); s.coreB[0].y = f2u(rB); s.coreB[0].z = f2u(rB);
            const float skew = rf(5e-3f, 3e-2f); // around z: needles nearly parallel
            s.poseAq = F4{ 0, 0, 0, 1 };
            s.poseBq = F4{ 0, 0, std::sin(skew * 0.5f), std::cos(skew * 0.5f) };
            const float gap = rf(1e-4f, 1e-3f);
            s.poseAp = F4{ pA.x, pA.y, pA.z, 0 };
            s.poseBp = F4{ pA.x, pA.y + rA + rB + gap, pA.z, 0 };
            s.cfg = F4{ 1e6f, 0, 0, 0 };
        }

        if (i % 199 == 23 || i % 199 == 87)
        {
            // DISJOINT ePOINTS hulls with EXACTLY equal bounds centers: the
            // corner simplex {0, 2ex, 2.5ey, 3ez} vs its point-mirror around
            // (2, 2.5, 3) -- x/2 + y/2.5 + z/3 <= 1 vs >= 2, positive gap,
            // both bounds [0,2]x[0,2.5]x[0,3].  The seed dir0 is exactly
            // zero, so the (1,0,0) fallback fires on a SEPARATED pair whose
            // closest face pair is parallel (non-unique witness points): the
            // fallback CHOICE shows in the tag-1 outputs, and the asymmetric
            // extents break any coordinate-permutation symmetry.
            // i%199==87 instead nudges the B center by a tiny y-dominant
            // offset: above FLT_EPSILON (no fallback), yet its normalized
            // seed picks a DIFFERENT first support than (1,0,0) would --
            // pinning the fallback threshold from the non-fallback side.
            typeA = typeB = CORE_POINTS;
            s.metaA = U4{ typeA, 0, 0, 0 };
            s.metaB = U4{ typeB, 0, 0, 0 };
            std::memset(s.coreA, 0, sizeof(s.coreA));
            std::memset(s.coreB, 0, sizeof(s.coreB));
            const V3 base{ rf(-2, 2), rf(-2, 2), rf(-2, 2) };
            for (U4* core : { s.coreA, s.coreB })
            {
                core[0].x = 48;
                core[0].z = f2u(1.f); core[0].w = f2u(1.f);
                core[1].x = f2u(1.f);
                core[1].y = f2u(0.f); core[1].z = f2u(0.f); core[1].w = f2u(0.f);
                core[2].x = f2u(1.f); // identity R, unit S
                core[2].y = 4 | (16u << 8);
            }
            s.ptsA[0] = F4{ 0, 0, 0, 0 };    s.ptsA[1] = F4{ 2, 0, 0, 0 };
            s.ptsA[2] = F4{ 0, 2.5f, 0, 0 }; s.ptsA[3] = F4{ 0, 0, 3, 0 };
            s.ptsB[0] = F4{ 2, 2.5f, 3, 0 }; s.ptsB[1] = F4{ 0, 2.5f, 3, 0 };
            s.ptsB[2] = F4{ 2, 0, 3, 0 };    s.ptsB[3] = F4{ 2, 2.5f, 0, 0 };
            s.poseAq = F4{ 0, 0, 0, 1 };
            s.poseBq = F4{ 0, 0, 0, 1 };
            const float nudge = (i % 199 == 87) ? 5e-5f : 0.f;
            s.poseAp = F4{ base.x, base.y, base.z, 0 };
            s.poseBp = F4{ base.x + 0.6f * nudge, base.y + nudge, base.z, 0 };
            s.cfg = F4{ 1e6f, 0, 0, 0 };
        }

        in[i] = s;
        ++cType[s.metaA.x]; ++cType[s.metaB.x]; // counted from what dispatches
        RefStats st;
        refGjk(s, ref[i], st);
        ++cTag[st.tag];
        if (st.fallback) ++cFallback;
        if (st.weld) ++cWeld;
        if (st.dup) ++cDup;
        if (st.iterCap) ++cIterCap;
        if (st.cacheFull) ++cFull; if (st.maxCache > gMaxCache) gMaxCache = st.maxCache;
    }

    std::printf("[ info ] gjk cov: tags %zu/%zu/%zu fb %zu weld %zu dup %zu iterCap %zu cacheFull %zu maxCache %u\n",
                cTag[0], cTag[1], cTag[2], cFallback, cWeld, cDup, cIterCap, cFull, gMaxCache);

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 31) / 32;
    std::vector<GjkOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npCvxCoreGjk.spv", "cgjDistance",
             { (uint32_t)sizeof(GjkIn), (uint32_t)sizeof(GjkOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<GjkOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const GjkOut& r = ref[i]; const GjkOut& q = g[i];
        const bool ok = f2u(r.result.y) == f2u(q.result.y) && eqFZ(r.result.x, q.result.x) &&
                        eqF4Z(r.pointA, q.pointA) && eqF4Z(r.pointB, q.pointB) && eqF4Z(r.axis, q.axis);
        if (!ok && mm < 5)
            std::printf("MM %u tagR %u tagG %u dR %.9g dG %.9g axR (%g %g %g) axG (%g %g %g)\n",
                        i, f2u(r.result.y), f2u(q.result.y), r.result.x, q.result.x,
                        r.axis.x, r.axis.y, r.axis.z, q.axis.x, q.axis.y, q.axis.z);
        if (!ok) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "cgjDistance: %zu mismatches", mm);

    bool cov = cTag[0] > N / 10 && cTag[1] > N / 10 && cTag[2] > N / 20 &&
               cFallback > 8 && cWeld > N / 20 && cDup > N / 20 && cPtPt > 8 && cNullPts > 8 && cTinyS > 8 &&
               cFull >= 1 && cIterCap >= 1;
    for (int k2 = 0; k2 < 7; ++k2) cov = cov && cType[k2] > N / 16;
    PXG_CHECK_MSG(cov, "cgjDistance: coverage thin (tags %zu/%zu/%zu fb %zu weld %zu dup %zu cap %zu full %zu)",
                  cTag[0], cTag[1], cTag[2], cFallback, cWeld, cDup, cIterCap, cFull);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
