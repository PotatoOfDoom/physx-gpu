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

// Verification of npCvxCoreCvxGen.slang: Gu::generateContacts(ConvexShape,
// ConvexShape) (GuConvexSupport.h L1162-1207).  The reference composes the
// tranche-169/170/167/165/166 CPU transcriptions VERBATIM, each in its own
// namespace: gjkref (computeGjkDistance), eparef (computeEpaDepth), faceref
// (the contactFace wrapper family) and clipref (FaceClipper + getContact),
// glued by the driver: maxDist gate, GJK -> EPA on overlap, cull test,
// margin pushback, generateContactPatch.  Compare doctrine inherited:
// counts/normal/non-trig outputs BITWISE (modulo the documented +-0 class),
// contactFace cap points behind the deterministic trig flags to tolerance.
//
// P9.86c adds npCvxCoreCvxGen_triNpAtoBSign at the end of the file: the engine
// kernel convexCoreTrimeshNphase_Kernel32 (cvxCoreTriNpBda.slang) run end to end
// against refCvxGen composed with the Contact32 accumulator and the multi-patch
// tail -- see the block comment there.

#include "pxgTestFramework.h"

#include "geometry/PxGeometry.h"
#include "geometry/PxConvexCoreGeometry.h"
#include "GuBV32.h"

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

namespace gjkref {


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

// P9.86c: mutant twin of the pre-fix port as llvmpipe compiles it.  0 = the CUDA
// (GuRefGjkEpa.h:353-360 computePoints, an IEEE chain seeded with +0); 1 = the first
// unrolled `0 + t` of computePoints dropped (the F3 unrolled-seed fold), so a lane
// whose terms are all -0 ends at -0 instead of +0.
static int gSeedFoldVariant = 0;

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
                const V3 ta = vscl(cA.verts[verts[i].aI], verts[i].s);
                const V3 tb = vscl(cB.verts[verts[i].bI], verts[i].s);
                if (gSeedFoldVariant == 1 && i == 0) { pA = ta; pB = tb; }
                else { pA = vadd(pA, ta); pB = vadd(pB, tb); }
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
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed
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


} // namespace gjkref

namespace eparef {


struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };
struct V3 { float x, y, z; };

struct EpaTIn
{
    U4 metaA;
    U4 coreA[3];
    F4 poseAq, poseAp;
    U4 metaB;
    U4 coreB[3];
    F4 poseBq, poseBp;
    F4 cfg;
    F4 ptsA[12];
    F4 ptsB[12];
};
struct EpaTOut
{
    F4 result;    // x = return value, y = tag bits (0 invalid, 1 converged)
    F4 pointA, pointB, axis;
};

const uint32_t CORE_POINT = 0, CORE_SEGMENT = 1, CORE_BOX = 2, CORE_ELLIPSOID = 3,
               CORE_CYLINDER = 4, CORE_CONE = 5, CORE_POINTS = 6;
const uint32_t MAX_VERTS = 32, MAX_FACES = 2 * MAX_VERTS - 4, MAX_EDGES = MAX_VERTS + MAX_FACES - 2;

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
// PxComputeBasisVectors (PxMathUtils.h L297-316)
static void computeBasis(V3 dir, V3& right, V3& up)
{
    if (std::fabs(dir.y) <= 0.9999f)
    {
        right = V3{ dir.z, 0.f, -dir.x };
        normalizeIn(right);
        up = V3{ dir.y * right.z, dir.z * right.x - dir.x * right.z, -dir.y * right.x };
    }
    else
    {
        right = V3{ 1.f, 0.f, 0.f };
        up = V3{ 0.f, dir.z, -dir.y };
        normalizeIn(up);
    }
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

// ConvexShape::supportLocal (no margin) -- the exact tranche-164 forms
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
    bool fallback = false, basisElse = false, invalid = false;
    bool dup = false, cap = false, planeFail = false, edge0 = false, faceOver = false;
    uint32_t iters = 0;
};

struct ConvexC
{
    ShapeRefC shape;
    float accuracy;
    uint32_t numVerts;
    V3 verts[MAX_VERTS];
};

static float convexAccuracy(const ShapeRefC& s)
{
    const V3 X{ 1, 0, 0 }, Y{ 0, 1, 0 }, Z{ 0, 0, 1 };
    const V3 mn{ supportLocal(s, vneg(X)).x, supportLocal(s, vneg(Y)).y, supportLocal(s, vneg(Z)).z };
    const V3 mx{ supportLocal(s, X).x, supportLocal(s, Y).y, supportLocal(s, Z).z };
    const V3 dim = vsub(mx, mn);
    const float maxExtent = std::fmax(dim.x, std::fmax(dim.y, dim.z));
    return std::fmax(maxExtent * 0.01f, FLT_EPSILON);
}

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

static uint32_t supportIndex(ConvexC& c, V3 dir)
{
    const V3 v = poseTf(c.shape, supportLocal(c.shape, qRotInv(c.shape.poseQ, dir)));
    uint32_t closest = MAX_VERTS;
    const float WELD_DIST_SQ = c.accuracy * c.accuracy;
    float distSq = FLT_MAX;
    for (int32_t i = int32_t(c.numVerts) - 1; i >= 0; --i)
    {
        const float dSq = magSq(vsub(c.verts[i], v));
        if (dSq < WELD_DIST_SQ) return uint32_t(i);
        if (distSq > dSq) { distSq = dSq; closest = uint32_t(i); }
    }
    if (c.numVerts == MAX_VERTS) return closest;
    c.verts[c.numVerts++] = v;
    return c.numVerts - 1;
}

// EpaDepth::makePlane (L455-472)
static bool makePlane(V3 a, V3 b, V3 c, F4& plane)
{
    const V3 abc = vadd(vadd(vcross(vsub(b, a), vsub(c, a)), vcross(vsub(c, b), vsub(a, b))),
                        vcross(vsub(a, c), vsub(b, c)));
    const float labc = mag(abc);
    if (labc < FLT_EPSILON) // normalEps
        return false;
    const V3 n = vscl(abc, 1.0f / labc); // PxVec3 operator/: reciprocal multiply
    const float d = vdot(n, vneg(a));
    plane = F4{ n.x, n.y, n.z, d };
    return true;
}

struct EpaPolyC
{
    uint32_t vaI[MAX_VERTS], vbI[MAX_VERTS];
    F4 fpl[MAX_FACES];
    uint32_t fv[MAX_FACES][3];
    uint32_t numVerts = 0, numFaces = 0;
    F4 closestPlane{ 0, 0, 0, 0 }; uint32_t closestV[3] = { 0, 0, 0 };
    F4 bestPlane{ 0, 0, 0, 0 };    uint32_t bestV[3] = { 0, 0, 0 };
    float mProj = -FLT_MAX;
};

// EpaDepth::addPoint (L480-565)
static float epaAddPoint(EpaPolyC& P, ConvexC& cA, ConvexC& cB, V3 dir, float accuracy,
                         bool& fltMax, RefStats& st)
{
    fltMax = false;
    if (P.numVerts > MAX_VERTS - 1) { fltMax = true; st.cap = true; return 0.0f; }

    const uint32_t aI = supportIndex(cA, dir);
    const uint32_t bI = supportIndex(cB, vneg(dir));
    const V3 p = vsub(cA.verts[aI], cB.verts[bI]);

    const float proj = vdot(p, vneg(dir));
    if (proj > P.mProj)
    {
        P.mProj = proj;
        P.bestPlane = P.closestPlane;
        for (int k = 0; k < 3; ++k) P.bestV[k] = P.closestV[k];
    }

    for (uint32_t i = 0; i < P.numVerts; ++i)
        if (P.vaI[i] == aI && P.vbI[i] == bI) { fltMax = true; st.dup = true; return 0.0f; }

    P.vaI[P.numVerts] = aI; P.vbI[P.numVerts] = bI;
    ++P.numVerts;

    if (P.numVerts < 3)
        return 0.0f;

    if (P.numVerts == 3)
    {
        const V3 a = vsub(cA.verts[P.vaI[0]], cB.verts[P.vbI[0]]);
        const V3 b = vsub(cA.verts[P.vaI[1]], cB.verts[P.vbI[1]]);
        const V3 c = vsub(cA.verts[P.vaI[2]], cB.verts[P.vbI[2]]);
        F4 plane0, plane1;
        if (!makePlane(a, b, c, plane0) || !makePlane(b, a, c, plane1))
        { fltMax = true; st.planeFail = true; return 0.0f; }

        P.fpl[0] = plane0; P.fv[0][0] = 0; P.fv[0][1] = 1; P.fv[0][2] = 2;
        P.fpl[1] = plane1; P.fv[1][0] = 1; P.fv[1][1] = 0; P.fv[1][2] = 2;
        P.numFaces = 2;
        return 0.0f;
    }

    const float testEps = 0.01f * accuracy;
    uint32_t edges[MAX_EDGES][2];
    uint32_t numEdges = 0;
    for (int32_t i = int32_t(P.numFaces) - 1; i >= 0; --i)
    {
        const F4& pl = P.fpl[i];
        if (pl.x * p.x + pl.y * p.y + pl.z * p.z + pl.w * 1.0f > testEps)
        {
            for (uint32_t j = 0; j < 3; ++j)
            {
                bool add = true;
                const uint32_t v0 = P.fv[i][j], v1 = P.fv[i][(j + 1) % 3];
                for (int32_t k = int32_t(numEdges) - 1; k >= 0; --k)
                {
                    if (v0 == edges[k][1] && v1 == edges[k][0])
                    {
                        add = false;
                        edges[k][0] = edges[numEdges - 1][0];
                        edges[k][1] = edges[numEdges - 1][1];
                        --numEdges;
                        break;
                    }
                }
                if (add)
                {
                    edges[numEdges][0] = v0; edges[numEdges][1] = v1;
                    ++numEdges;
                }
            }
            P.fpl[i] = P.fpl[P.numFaces - 1];
            for (int k = 0; k < 3; ++k) P.fv[i][k] = P.fv[P.numFaces - 1][k];
            --P.numFaces;
        }
    }

    if (numEdges == 0) { fltMax = true; st.edge0 = true; return 0.0f; }
    if (P.numFaces > MAX_FACES - numEdges) { fltMax = true; st.faceOver = true; return 0.0f; }

    for (uint32_t i = 0; i < numEdges; ++i)
    {
        const uint32_t i0 = edges[i][0], i1 = edges[i][1], i2 = P.numVerts - 1;
        const V3 a = vsub(cA.verts[P.vaI[i0]], cB.verts[P.vbI[i0]]);
        const V3 b = vsub(cA.verts[P.vaI[i1]], cB.verts[P.vbI[i1]]);
        const V3 c = vsub(cA.verts[P.vaI[i2]], cB.verts[P.vbI[i2]]);
        F4 plane;
        if (!makePlane(a, b, c, plane)) { fltMax = true; st.planeFail = true; return 0.0f; }
        P.fpl[P.numFaces] = plane;
        P.fv[P.numFaces][0] = i0; P.fv[P.numFaces][1] = i1; P.fv[P.numFaces][2] = i2;
        ++P.numFaces;
    }

    return proj;
}

// computeEpaDepth driver (L671-699)
static void refEpa(const EpaTIn& in, EpaTOut& o, RefStats& st)
{
    std::memset(&o, 0, sizeof(o));

    ShapeRefC sA{ in.metaA.x, in.coreA, in.poseAq, V3{ in.poseAp.x, in.poseAp.y, in.poseAp.z }, in.ptsA };
    ShapeRefC sB{ in.metaB.x, in.coreB, in.poseBq, V3{ in.poseBp.x, in.poseBp.y, in.poseBp.z }, in.ptsB };

    ConvexC cA, cB;
    cA.shape = sA; cA.numVerts = 0; cA.accuracy = convexAccuracy(sA);
    cB.shape = sB; cB.numVerts = 0; cB.accuracy = convexAccuracy(sB);
    const float accuracy = std::fmin(cA.accuracy, cB.accuracy);

    EpaPolyC P;

    // ctor (L424-440)
    V3 mnA, mxA, mnB, mxB;
    convexBounds(sA, mnA, mxA);
    convexBounds(sB, mnB, mxB);
    V3 dir0 = vsub(vscl(vadd(mnB, mxB), 0.5f), vscl(vadd(mnA, mxA), 0.5f));
    if (normalizeIn(dir0) < FLT_EPSILON) { dir0 = V3{ 1, 0, 0 }; st.fallback = true; }
    if (std::fabs(dir0.y) > 0.9999f) st.basisElse = true;
    V3 dir1, dir2;
    computeBasis(dir0, dir1, dir2);

    bool fm;
    epaAddPoint(P, cA, cB, dir0, accuracy, fm, st);
    epaAddPoint(P, cA, cB, vneg(dir0), accuracy, fm, st);
    epaAddPoint(P, cA, cB, dir1, accuracy, fm, st);
    if (P.numVerts < 3) epaAddPoint(P, cA, cB, vneg(dir1), accuracy, fm, st);
    if (P.numVerts < 3) epaAddPoint(P, cA, cB, dir2, accuracy, fm, st);
    if (P.numVerts < 3) epaAddPoint(P, cA, cB, vneg(dir2), accuracy, fm, st);
    P.mProj = -FLT_MAX;

    if (P.numFaces == 0) // !isValid()
    {
        o.result = F4{ FLT_MAX, u2f(0), 0, 0 };
        st.tag = 0;
        st.invalid = true;
        return;
    }

    const float distEps = 0.01f * accuracy;
    while (true)
    {
        ++st.iters;
        // computeClosest (L568-590)
        uint32_t closest = 0;
        float closestW = -FLT_MAX;
        for (uint32_t i = 0; i < P.numFaces; ++i)
        {
            const float w = P.fpl[i].w;
            if (w > closestW) { closest = i; closestW = w; }
        }
        P.closestPlane = P.fpl[closest];
        for (int k = 0; k < 3; ++k) P.closestV[k] = P.fv[closest][k];
        if (P.numFaces == 2)
        {
            P.bestPlane = P.closestPlane;
            for (int k = 0; k < 3; ++k) P.bestV[k] = P.closestV[k];
        }

        const float closestDist = P.closestPlane.w;
        const V3 dir{ P.closestPlane.x, P.closestPlane.y, P.closestPlane.z };

        float proj = epaAddPoint(P, cA, cB, dir, accuracy, fm, st);
        if (fm) proj = FLT_MAX;

        if (proj >= closestDist - distEps)
        {
            // computePoints (L609-624)
            const uint32_t a0 = P.vaI[P.bestV[0]], a1 = P.vaI[P.bestV[1]], a2 = P.vaI[P.bestV[2]];
            const uint32_t b0 = P.vbI[P.bestV[0]], b1 = P.vbI[P.bestV[1]], b2 = P.vbI[P.bestV[2]];
            const V3 va0 = cA.verts[a0], va1 = cA.verts[a1], va2 = cA.verts[a2];
            const V3 vb0 = cB.verts[b0], vb1 = cB.verts[b1], vb2 = cB.verts[b2];
            const V3 a = vsub(va0, vb0), b = vsub(va1, vb1), c = vsub(va2, vb2);
            const V3 abc = vcross(vsub(b, a), vsub(c, a));
            const float iabc = 1.0f / magSq(abc);
            const V3 pabc = vscl(vscl(abc, vdot(abc, a)), iabc);
            const float tbc = vdot(abc, vcross(vsub(b, pabc), vsub(c, pabc)));
            const float tca = vdot(abc, vcross(vsub(c, pabc), vsub(a, pabc)));
            const float tab = vdot(abc, vcross(vsub(a, pabc), vsub(b, pabc)));
            const float sa = tbc * iabc, sb = tca * iabc, sc = tab * iabc;
            const V3 pA = vadd(vadd(vscl(va0, sa), vscl(va1, sb)), vscl(va2, sc));
            const V3 pB = vadd(vadd(vscl(vb0, sa), vscl(vb1, sb)), vscl(vb2, sc));
            o.result = F4{ P.bestPlane.w, u2f(1), 0, 0 };
            o.pointA = F4{ pA.x, pA.y, pA.z, 0 };
            o.pointB = F4{ pB.x, pB.y, pB.z, 0 };
            o.axis = F4{ -P.bestPlane.x, -P.bestPlane.y, -P.bestPlane.z, 0 };
            st.tag = 1;
            return;
        }
    }
}

// bitwise, tolerating only the SIGN of exact zeros (tranche-164 NSZ class)
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed
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
        const uint32_t np = 3 + rng() % 10;
        core[0].x = 48;
        core[0].z = f2u(rf(0.3f, 2)); core[0].w = f2u(rf(0.3f, 2));
        core[1].x = f2u(rf(0.3f, 2));
        float a[4]; float m2 = 0;
        for (int k = 0; k < 4; ++k) { a[k] = rf(-1, 1); m2 += a[k] * a[k]; }
        const float qi = 1.f / std::sqrt(m2 > 0 ? m2 : 1.f);
        core[1].y = f2u(a[0] * qi); core[1].z = f2u(a[1] * qi); core[1].w = f2u(a[2] * qi);
        core[2].x = f2u(a[3] * qi);
        core[2].y = np | (16u << 8);
        for (uint32_t p = 0; p < 12; ++p)
            pts[p] = F4{ rf(-1.5f, 1.5f), rf(-1.5f, 1.5f), rf(-1.5f, 1.5f), 0 };
    }
}


} // namespace eparef

namespace clipref {


struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };
struct V3 { float x, y, z; };
struct ClipIn
{
    F4 p0, p1, axis, n0, n1;
    U4 counts;
    F4 fp0[4], fp1[4];
};
struct ClipOut
{
    U4 counts;
    F4 normal;
    F4 pts[4];
};

const uint32_t NOIDX = 0xFFFFFFFFu;
const uint32_t MAX_CLIP_PLANES = 8, MAX_POLY = 13;

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b) { return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static V3 sub(V3 a, V3 b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 add(V3 a, V3 b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 mul(V3 a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; }
static V3 neg(V3 a) { return V3{ -a.x, -a.y, -a.z }; }
static float magSq(V3 v) { return v.x * v.x + v.y * v.y + v.z * v.z; }
static float dot4(const F4& a, const F4& b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
static void normRecip(V3& v) // PxVec3::normalize (guarded reciprocal multiply)
{
    const float m = std::sqrt(magSq(v));
    if (m > 0) { const float inv = 1.0f / m; v.x *= inv; v.y *= inv; v.z *= inv; }
}
static void normFast(V3& v) // PxVec3::normalizeFast (unguarded)
{
    const float inv = 1.0f / std::sqrt(magSq(v));
    v.x *= inv; v.y *= inv; v.z *= inv;
}

static uint32_t reduce4(const F4 pts[MAX_POLY], uint32_t count, V3 normal, uint32_t inds[4])
{
    inds[0] = inds[1] = inds[2] = inds[3] = NOIDX;
    uint32_t pi0 = 0, pi1 = NOIDX;
    {
        float maxDist = -FLT_MAX;
        const V3 p0{ pts[0].x, pts[0].y, pts[0].z };
        for (uint32_t j = 0; j < count; ++j)
        {
            const float dist = magSq(sub(V3{ pts[j].x, pts[j].y, pts[j].z }, p0));
            if (dist > maxDist + FLT_EPSILON) { maxDist = dist; pi1 = j; }
        }
    }
    if (pi0 == NOIDX || pi1 == NOIDX) return 0;
    if (pi0 == pi1) { inds[0] = pi0; return 1; }

    uint32_t pi2 = NOIDX, pi3 = NOIDX;
    {
        const V3 p0{ pts[pi0].x, pts[pi0].y, pts[pi0].z };
        const V3 p1{ pts[pi1].x, pts[pi1].y, pts[pi1].z };
        const V3 n01 = cross3(normal, sub(p1, p0));
        float minDist = 0, maxDist = 0;
        for (uint32_t i = 0; i < count; ++i)
        {
            const float d = dot3(n01, sub(V3{ pts[i].x, pts[i].y, pts[i].z }, p0));
            if (d < minDist - FLT_EPSILON) { pi2 = i; minDist = d; }
            if (d > maxDist + FLT_EPSILON) { pi3 = i; maxDist = d; }
        }
    }
    uint32_t n = 0;
    inds[n++] = pi0;
    inds[n++] = pi1;
    if (pi2 != NOIDX && pi2 != pi0 && pi2 != pi1) inds[n++] = pi2;
    if (pi3 != NOIDX && pi3 != pi0 && pi3 != pi1 && pi3 != pi2) inds[n++] = pi3;
    return n;
}

static void makePlanesRef(const V3 face[4], uint32_t numPoints, V3 axis, F4 planes[MAX_CLIP_PLANES], uint32_t& numPlanes)
{
    if (numPoints < 3) return;
    const float eps = 1e-5f;
    for (uint32_t i = 0; i < numPoints; ++i)
    {
        const V3 s = face[i];
        for (uint32_t j = i + 1; j < numPoints; ++j)
        {
            const V3 e = face[j];
            V3 n = cross3(sub(e, s), axis);
            if (magSq(n) < eps * eps) continue;
            normFast(n);
            uint32_t posCount = 0, negCount = 0;
            for (uint32_t k = 0; k < numPoints; ++k)
            {
                if (k == i || k == j) continue;
                const float d = dot3(sub(face[k], s), n);
                if (d > eps) posCount++;
                else negCount++;
            }
            if (posCount && negCount) continue;
            if (posCount) n = neg(n);
            planes[numPlanes++] = F4{ -n.x, -n.y, -n.z, dot3(n, s) };
        }
    }
}

static void refClip(const ClipIn& s, ClipOut& o)
{
    std::memset(&o, 0, sizeof(o));
    const V3 point0{ s.p0.x, s.p0.y, s.p0.z }, point1{ s.p1.x, s.p1.y, s.p1.z };
    const V3 axis{ s.axis.x, s.axis.y, s.axis.z };
    const uint32_t n0 = s.counts.x, n1 = s.counts.y;
    V3 face0[4], face1[4];
    for (int i = 0; i < 4; ++i)
    {
        face0[i] = V3{ s.fp0[i].x, s.fp0[i].y, s.fp0[i].z };
        face1[i] = V3{ s.fp1[i].x, s.fp1[i].y, s.fp1[i].z };
    }

    F4 outPts[4] = {};
    uint32_t outNum = 0;
    bool doneNone = false;

    if (n0 < 2 || n1 < 2)
        doneNone = true;
    else if (n0 == 2 && n1 == 2)
    {
        const float eps = 1e-5f;
        const V3 a = face0[0], b = face0[1], c = face1[0], d = face1[1];
        if (std::fabs(dot3(cross3(sub(b, a), sub(d, c)), axis)) > eps)
            doneNone = true;
        else
        {
            uint32_t num = 0;
            F4 points[4];
            auto proj = [&](V3 p, V3 s0, V3 e0) {
                return add(s0, mul(mul(sub(e0, s0), dot3(sub(e0, s0), sub(p, s0))), 1.0f / magSq(sub(e0, s0))));
            };
            if (dot3(sub(a, c), sub(d, c)) > -eps && dot3(sub(a, d), sub(c, d)) > -eps)
            {
                const V3 a1 = proj(a, c, d);
                points[num++] = F4{ (a.x + a1.x) * 0.5f, (a.y + a1.y) * 0.5f, (a.z + a1.z) * 0.5f, dot3(axis, sub(a, a1)) };
            }
            if (dot3(sub(b, c), sub(d, c)) > -eps && dot3(sub(b, d), sub(c, d)) > -eps)
            {
                const V3 b1 = proj(b, c, d);
                points[num++] = F4{ (b.x + b1.x) * 0.5f, (b.y + b1.y) * 0.5f, (b.z + b1.z) * 0.5f, dot3(axis, sub(b, b1)) };
            }
            if (dot3(sub(c, a), sub(b, a)) > -eps && dot3(sub(c, b), sub(a, b)) > -eps)
            {
                const V3 c1 = proj(c, a, b);
                points[num++] = F4{ (c1.x + c.x) * 0.5f, (c1.y + c.y) * 0.5f, (c1.z + c.z) * 0.5f, dot3(axis, sub(c1, c)) };
            }
            if (dot3(sub(d, a), sub(b, a)) > -eps && dot3(sub(d, b), sub(a, b)) > -eps)
            {
                const V3 d1 = proj(d, a, b);
                points[num++] = F4{ (d1.x + d.x) * 0.5f, (d1.y + d.y) * 0.5f, (d1.z + d.z) * 0.5f, dot3(axis, sub(d1, d)) };
            }

            const float mergeEps = 1e-6f;
            for (int i = int(num) - 1; i >= 0; --i)
                for (int j = i - 1; j >= 0; --j)
                    if (magSq(V3{ points[i].x - points[j].x, points[i].y - points[j].y, points[i].z - points[j].z }) < mergeEps)
                    {
                        if (points[i].w < points[j].w) points[j] = points[i];
                        points[i] = points[--num];
                        break;
                    }

            for (uint32_t i = 0; i < num; ++i) outPts[i] = points[i];
            outNum = num;
        }
    }
    else
    {
        uint32_t numPlanes = 0;
        F4 planes[MAX_CLIP_PLANES];
        makePlanesRef(face0, n0, axis, planes, numPlanes);
        makePlanesRef(face1, n1, axis, planes, numPlanes);

        uint32_t numPoly = 0;
        V3 poly[MAX_POLY];
        if (n0 == 2) { poly[numPoly++] = face0[0]; poly[numPoly++] = face0[1]; }
        else if (n1 == 2) { poly[numPoly++] = face1[0]; poly[numPoly++] = face1[1]; }
        else
        {
            V3 X, Y;
            if (std::fabs(axis.y) <= 0.9999f)
            {
                X = V3{ axis.z, 0.f, -axis.x };
                normRecip(X);
                Y = V3{ axis.y * X.z, axis.z * X.x - axis.x * X.z, -axis.y * X.x };
            }
            else
            {
                X = V3{ 1, 0, 0 };
                Y = V3{ 0.f, axis.z, -axis.y };
                normRecip(Y);
            }
            float minX = FLT_MAX, maxX = -FLT_MAX, minY = FLT_MAX, maxY = -FLT_MAX;
            for (uint32_t i = 0; i < n0; ++i)
            {
                const float pX = dot3(face0[i], X), pY = dot3(face0[i], Y);
                minX = std::fmin(minX, pX); maxX = std::fmax(maxX, pX);
                minY = std::fmin(minY, pY); maxY = std::fmax(maxY, pY);
            }
            for (uint32_t i = 0; i < n1; ++i)
            {
                const float pX = dot3(face1[i], X), pY = dot3(face1[i], Y);
                minX = std::fmin(minX, pX); maxX = std::fmax(maxX, pX);
                minY = std::fmin(minY, pY); maxY = std::fmax(maxY, pY);
            }
            const V3 ref = mul(add(point0, point1), 0.5f);
            const float refX = dot3(ref, X), refY = dot3(ref, Y);
            poly[numPoly++] = add(add(ref, mul(X, minX - refX)), mul(Y, minY - refY));
            poly[numPoly++] = add(add(ref, mul(X, maxX - refX)), mul(Y, minY - refY));
            poly[numPoly++] = add(add(ref, mul(X, maxX - refX)), mul(Y, maxY - refY));
            poly[numPoly++] = add(add(ref, mul(X, minX - refX)), mul(Y, maxY - refY));
        }

        for (uint32_t pi = 0; pi < numPlanes; ++pi)
        {
            const F4 plane = planes[pi];
            float dist[MAX_POLY];
            for (uint32_t i = 0; i < numPoly; ++i)
                dist[i] = dot4(plane, F4{ poly[i].x, poly[i].y, poly[i].z, 1.f });
            const float eps = 1e-5f;
            uint32_t newNum = 0;
            V3 newPts[MAX_POLY];
            for (uint32_t i = 0; i < numPoly; ++i)
            {
                const uint32_t i0 = i, i1 = (i + 1) % numPoly;
                const float d0 = dist[i0], d1 = dist[i1];
                const V3 q0 = poly[i0], q1 = poly[i1];
                if (d0 > -eps)
                {
                    newPts[newNum++] = q0;
                    if (d1 < -eps && d0 - d1 > eps)
                        newPts[newNum++] = add(q0, mul(sub(q1, q0), d0 / (d0 - d1)));
                }
                else if (d1 > -eps && d1 - d0 > eps)
                    newPts[newNum++] = add(q0, mul(sub(q1, q0), d0 / (d0 - d1)));
            }
            numPoly = newNum;
            for (uint32_t i = 0; i < newNum; ++i) poly[i] = newPts[i];
        }

        poly[numPoly++] = mul(add(point0, point1), 0.5f);

        const F4 plane0{ s.n0.x, s.n0.y, s.n0.z, -dot3(V3{ s.n0.x, s.n0.y, s.n0.z }, face0[0]) };
        const F4 plane1{ s.n1.x, s.n1.y, s.n1.z, -dot3(V3{ s.n1.x, s.n1.y, s.n1.z }, face1[0]) };

        F4 polyDists[MAX_POLY];
        for (uint32_t i = 0; i < numPoly; ++i)
        {
            const V3 p = poly[i];
            const V3 pr0 = sub(p, mul(mul(axis, dot4(plane0, F4{ p.x, p.y, p.z, 1.f })), 1.0f / dot3(axis, V3{ plane0.x, plane0.y, plane0.z })));
            const V3 pr1 = sub(p, mul(mul(axis, dot4(plane1, F4{ p.x, p.y, p.z, 1.f })), 1.0f / dot3(axis, V3{ plane1.x, plane1.y, plane1.z })));
            polyDists[i] = F4{ (pr0.x + pr1.x) * 0.5f, (pr0.y + pr1.y) * 0.5f, (pr0.z + pr1.z) * 0.5f, dot3(axis, sub(pr0, pr1)) };
        }

        uint32_t minIdx = 0;
        float minDist = polyDists[0].w;
        for (uint32_t i = 0; i < numPoly; ++i)
            if (polyDists[i].w < minDist) { minDist = polyDists[i].w; minIdx = i; }
        std::swap(polyDists[0], polyDists[minIdx]);

        uint32_t inds[4];
        const uint32_t num = reduce4(polyDists, numPoly, axis, inds);
        for (uint32_t i = 0; i < num; ++i) outPts[i] = polyDists[inds[i]];
        outNum = num;
    }

    if (doneNone)
    {
        outPts[0] = F4{ (point0.x + point1.x) * 0.5f, (point0.y + point1.y) * 0.5f, (point0.z + point1.z) * 0.5f,
                        dot3(axis, sub(point0, point1)) };
        outNum = 1;
    }

    o.counts = U4{ outNum, 0, 0, 0 };
    o.normal = F4{ axis.x, axis.y, axis.z, 0 };
    for (uint32_t i = 0; i < 4; ++i) o.pts[i] = outPts[i];
}

static bool eqF4(const F4& a, const F4& b)
{ return f2u(a.x) == f2u(b.x) && f2u(a.y) == f2u(b.y) && f2u(a.z) == f2u(b.z) && f2u(a.w) == f2u(b.w); }


} // namespace clipref

namespace faceref {


struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };
struct V3 { float x, y, z; };

struct PgIn
{
    F4 plane;
    U4 meta;
    U4 core[3];
    F4 poseQ, poseP;  // poseP.w = margin
    F4 cfg;           // x = contactDist
    F4 pts[12];
};
struct PgOut
{
    U4 counts;        // x = numContacts, y = trig
    F4 normal;
    F4 points[4];     // w = dist
};

struct FaceIn
{
    U4 meta;
    U4 core[3];
    F4 poseQ, poseP;
    F4 dir;
    F4 refp;
};
struct FaceOut
{
    U4 counts;
    F4 normal;
    F4 fp[4];
};

const uint32_t CORE_POINT = 0, CORE_SEGMENT = 1, CORE_BOX = 2, CORE_ELLIPSOID = 3,
               CORE_CYLINDER = 4, CORE_CONE = 5, CORE_POINTS = 6;
const uint32_t NOIDX = 0xFFFFFFFFu;

struct PtsIn
{
    U4 meta;
    U4 core[3];
    F4 poseQ, poseP;
    F4 dir;
    F4 refp;
    F4 pts[32];
};
struct PtsOut
{
    U4 counts;
    U4 inds;
    F4 normal;
    F4 fp[4];
};


static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static float pxSign(float a) { return (a >= 0.f) ? 1.f : -1.f; }
static V3 nrmz(V3 v)
{
    const float m = v.x * v.x + v.y * v.y + v.z * v.z;
    if (m > 0) { const float i = 1.f / std::sqrt(m); return V3{ v.x * i, v.y * i, v.z * i }; }
    return V3{ 0, 0, 0 };
}
static float normDiv(V3& v) // PxVec3::normalize -> operator/=: one division then
{                            // three multiplies (PxVec3.h L260-267), review 165
    const float m = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (m > 0) { const float inv = 1.0f / m; v.x *= inv; v.y *= inv; v.z *= inv; }
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
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b) { return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static V3 sub(V3 a, V3 b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 add(V3 a, V3 b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 mul(V3 a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; }
static V3 neg(V3 a) { return V3{ -a.x, -a.y, -a.z }; }
static float comp(V3 v, uint32_t i) { return i == 0 ? v.x : (i == 1 ? v.y : v.z); }
static void setComp(V3& v, uint32_t i, float x) { if (i == 0) v.x = x; else if (i == 1) v.y = x; else v.z = x; }

static float coreF(const FaceIn& s, uint32_t w) { return u2f((&s.core[w / 4].x)[w & 3]); }

static bool rotatePointsRef(V3 center, V3 refp, V3 normal, V3 pts[4], uint32_t count)
{
    V3 dir0 = sub(pts[0], center);
    if (normDiv(dir0) < FLT_EPSILON) return false;
    V3 dir1 = add(sub(refp, center), mul(normal, dot3(normal, sub(refp, center))));
    if (normDiv(dir1) < FLT_EPSILON) return false;
    V3 axis = cross3(dir0, dir1);
    if (normDiv(axis) < FLT_EPSILON) return false;
    axis = dot3(normal, axis) > 0 ? normal : neg(normal);
    const float d01 = dot3(dir0, dir1);
    const float angle = std::acos(d01 < -1.f ? -1.f : (d01 > 1.f ? 1.f : d01)); // PxAcos clamps (review 165)
    const float a = angle * 0.5f;
    const float sh = std::sin(a), ch = std::cos(a);
    const F4 rot{ axis.x * sh, axis.y * sh, axis.z * sh, ch };
    for (uint32_t i = 0; i < count; ++i)
        pts[i] = add(qRot(rot, sub(pts[i], center)), center);
    return true;
}

static V3 vadd(V3 a, V3 b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 vsub(V3 a, V3 b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 vscl(V3 a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; }
static V3 vneg(V3 a) { return V3{ -a.x, -a.y, -a.z }; }
static float vdot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 vcross(V3 a, V3 b)
{ return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static float magSq(V3 v) { return v.x * v.x + v.y * v.y + v.z * v.z; }
static float mag(V3 v) { return std::sqrt(magSq(v)); }
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

static void refFace(const FaceIn& s, FaceOut& o)
{
    std::memset(&o, 0, sizeof(o));
    const uint32_t type = s.meta.x;
    const F4 poseQ = s.poseQ;
    const V3 poseP{ s.poseP.x, s.poseP.y, s.poseP.z };
    const float margin = s.poseP.w;

    const V3 dir = qRotInv(poseQ, V3{ s.dir.x, s.dir.y, s.dir.z });
    const V3 refLocal = qRotInv(poseQ, sub(V3{ s.refp.x, s.refp.y, s.refp.z }, poseP));

    uint32_t numPoints = 0, trig = 0;
    V3 faceNormal{ 0, 0, 0 };
    V3 fp[4] = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } };
    const float dEps1 = 0.99f, dEps2 = 0.14f;

    if (type == CORE_SEGMENT)
    {
        const float len = coreF(s, 0);
        const V3 d = nrmz(dir);
        if (std::fabs(d.x) < dEps2)
        {
            faceNormal = nrmz(V3{ 0, d.y, d.z });
            fp[0] = V3{ len * 0.5f, 0, 0 };
            fp[1] = V3{ -len * 0.5f, 0, 0 };
            numPoints = 2;
        }
    }
    else if (type == CORE_BOX)
    {
        const V3 ext{ coreF(s, 0), coreF(s, 1), coreF(s, 2) };
        const V3 d = nrmz(dir);
        for (uint32_t a0 = 0; a0 < 3 && numPoints == 0; ++a0)
        {
            const uint32_t a1 = (a0 + 1) % 3, a2 = (a0 + 2) % 3;
            const float d0 = comp(d, a0), d1 = comp(d, a1), d2 = comp(d, a2);
            const float e0 = comp(ext, a0), e1 = comp(ext, a1), e2 = comp(ext, a2);
            if (std::fabs(d0) > dEps1 && (e1 > FLT_EPSILON || e2 > FLT_EPSILON))
            {
                setComp(faceNormal, a0, pxSign(d0));
                setComp(fp[0], a0, pxSign(d0) * e0 * 0.5f); setComp(fp[0], a1, e1 * 0.5f); setComp(fp[0], a2, e2 * 0.5f);
                setComp(fp[1], a0, pxSign(d0) * e0 * 0.5f); setComp(fp[1], a1, -e1 * 0.5f); setComp(fp[1], a2, e2 * 0.5f);
                setComp(fp[2], a0, pxSign(d0) * e0 * 0.5f); setComp(fp[2], a1, -e1 * 0.5f); setComp(fp[2], a2, -e2 * 0.5f);
                setComp(fp[3], a0, pxSign(d0) * e0 * 0.5f); setComp(fp[3], a1, e1 * 0.5f); setComp(fp[3], a2, -e2 * 0.5f);
                numPoints = 4;
            }
            else if (std::fabs(d0) < dEps2 && e0 > FLT_EPSILON &&
                (std::fabs(d1) < dEps1 || e2 < FLT_EPSILON) &&
                (std::fabs(d2) < dEps1 || e1 < FLT_EPSILON))
            {
                V3 n{ 0, 0, 0 };
                setComp(n, a1, d1); setComp(n, a2, d2);
                normDiv(n);
                faceNormal = n;
                setComp(fp[0], a0, e0 * 0.5f); setComp(fp[0], a1, pxSign(d1) * e1 * 0.5f); setComp(fp[0], a2, pxSign(d2) * e2 * 0.5f);
                setComp(fp[1], a0, -e0 * 0.5f); setComp(fp[1], a1, pxSign(d1) * e1 * 0.5f); setComp(fp[1], a2, pxSign(d2) * e2 * 0.5f);
                numPoints = 2;
            }
        }
    }
    else if (type == CORE_ELLIPSOID)
    {
        const V3 radii{ coreF(s, 0), coreF(s, 1), coreF(s, 2) };
        const V3 d = nrmz(dir);
        for (uint32_t a0 = 0; a0 < 3 && numPoints == 0; ++a0)
        {
            const uint32_t a1 = (a0 + 1) % 3, a2 = (a0 + 2) % 3;
            const float d0 = comp(d, a0), d1 = comp(d, a1), d2 = comp(d, a2);
            const float r0 = comp(radii, a0), r1 = comp(radii, a1), r2 = comp(radii, a2);
            if (std::fabs(d0) > dEps1 && r0 < FLT_EPSILON && r1 > FLT_EPSILON && r2 > FLT_EPSILON)
            {
                setComp(faceNormal, a0, pxSign(d0));
                setComp(fp[0], a1, r1); setComp(fp[1], a1, -r1);
                setComp(fp[2], a2, r2); setComp(fp[3], a2, -r2);
                numPoints = 4;
            }
            else if (std::fabs(d0) < dEps2 && r0 > FLT_EPSILON && r1 < FLT_EPSILON && r2 < FLT_EPSILON)
            {
                V3 n{ 0, 0, 0 };
                setComp(n, a1, d1); setComp(n, a2, d2);
                normDiv(n);
                faceNormal = n;
                setComp(fp[0], a0, r0); setComp(fp[1], a0, -r0);
                numPoints = 2;
            }
        }
    }
    else if (type == CORE_CYLINDER)
    {
        const float height = coreF(s, 0), radius = coreF(s, 1);
        const V3 d = nrmz(dir);
        if (std::fabs(d.x) > dEps1 && radius > 0)
        {
            faceNormal = V3{ pxSign(d.x), 0, 0 };
            fp[0] = V3{ pxSign(d.x) * height * 0.5f, radius, 0 };
            fp[1] = V3{ pxSign(d.x) * height * 0.5f, 0, radius };
            fp[2] = V3{ pxSign(d.x) * height * 0.5f, -radius, 0 };
            fp[3] = V3{ pxSign(d.x) * height * 0.5f, 0, -radius };
            numPoints = 4;
            trig = rotatePointsRef(V3{ pxSign(d.x) * height * 0.5f, 0, 0 }, refLocal, faceNormal, fp, 4) ? 1 : 0;
        }
        else if (std::fabs(d.x) < dEps2 && height > 0)
        {
            const V3 dr = nrmz(V3{ 0, d.y, d.z });
            faceNormal = dr;
            fp[0] = add(V3{ height * 0.5f, 0, 0 }, mul(dr, radius));
            fp[1] = add(V3{ -height * 0.5f, 0, 0 }, mul(dr, radius));
            numPoints = 2;
        }
    }
    else if (type == CORE_CONE)
    {
        const float height = coreF(s, 0), radius = coreF(s, 1);
        if (height + radius > 0)
        {
            const V3 d = nrmz(dir);
            const float eps = 1e-5f;
            if ((d.x < -dEps1 || (d.x > dEps1 && height < eps)) && radius > 0)
            {
                faceNormal = V3{ pxSign(d.x), 0, 0 };
                fp[0] = V3{ pxSign(d.x) * height * 0.5f, radius, 0 };
                fp[1] = V3{ pxSign(d.x) * height * 0.5f, 0, radius };
                fp[2] = V3{ pxSign(d.x) * height * 0.5f, -radius, 0 };
                fp[3] = V3{ pxSign(d.x) * height * 0.5f, 0, -radius };
                numPoints = 4;
                trig = rotatePointsRef(V3{ pxSign(d.x) * height * 0.5f, 0, 0 }, refLocal, faceNormal, fp, 4) ? 1 : 0;
            }
            else if (height > 0)
            {
                const float sinA = radius / std::sqrt(height * height + radius * radius);
                const float dX = d.x * std::sqrt(1.0f - sinA * sinA) - std::sqrt(1.0f - d.x * d.x) * sinA;
                if (std::fabs(dX) < dEps2)
                {
                    const V3 dr = nrmz(V3{ 0, d.y, d.z });
                    faceNormal = nrmz(V3{ radius / height, dr.y, dr.z });
                    fp[0] = V3{ height * 0.5f, 0, 0 };
                    fp[1] = add(V3{ -height * 0.5f, 0, 0 }, mul(dr, radius));
                    numPoints = 2;
                }
            }
        }
    }

    const V3 worldNormal = qRot(poseQ, faceNormal);
    o.counts = U4{ numPoints, trig, 0, 0 };
    o.normal = F4{ worldNormal.x, worldNormal.y, worldNormal.z, 0 };
    for (uint32_t i = 0; i < 4; ++i)
    {
        if (i < numPoints)
        {
            const V3 wp = add(add(qRot(poseQ, fp[i]), poseP), mul(worldNormal, margin));
            o.fp[i] = F4{ wp.x, wp.y, wp.z, 0 };
        }
    }
}


static uint32_t reduceRef(const V3 pts[8], uint32_t count, V3 normal, bool keep1st, uint32_t inds[4])
{
    inds[0] = inds[1] = inds[2] = inds[3] = NOIDX;
    uint32_t pi0 = NOIDX, pi1 = NOIDX;
    if (keep1st)
    {
        pi0 = 0;
        float maxDist = -FLT_MAX;
        const V3 p0 = pts[0];
        for (uint32_t j = 0; j < count; ++j)
        {
            const float dist = magSq(sub(pts[j], p0));
            if (dist > maxDist + FLT_EPSILON) { maxDist = dist; pi1 = j; }
        }
    }
    else
    {
        float maxDist = -FLT_MAX;
        for (uint32_t i = 0; i < count; ++i)
        {
            const V3 p0 = pts[i];
            for (uint32_t j = 0; j < count; ++j)
            {
                const float dist = magSq(sub(pts[j], p0));
                if (dist > maxDist + FLT_EPSILON) { maxDist = dist; pi0 = i; pi1 = j; }
            }
        }
    }

    if (pi0 == NOIDX || pi1 == NOIDX) return 0;
    if (pi0 == pi1) { inds[0] = pi0; return 1; }

    uint32_t pi2 = NOIDX, pi3 = NOIDX;
    {
        const V3 p0 = pts[pi0], p1 = pts[pi1];
        const V3 n01 = cross3(normal, sub(p1, p0));
        float minDist = 0, maxDist = 0;
        for (uint32_t i = 0; i < count; ++i)
        {
            const float d = dot3(n01, sub(pts[i], p0));
            if (d < minDist - FLT_EPSILON) { pi2 = i; minDist = d; }
            if (d > maxDist + FLT_EPSILON) { pi3 = i; maxDist = d; }
        }
    }

    uint32_t n = 0;
    inds[n++] = pi0;
    inds[n++] = pi1;
    if (pi2 != NOIDX && pi2 != pi0 && pi2 != pi1) inds[n++] = pi2;
    if (pi3 != NOIDX && pi3 != pi0 && pi3 != pi1 && pi3 != pi2) inds[n++] = pi3;
    return n;
}


static V3 getPoolPoint(const PtsIn& s, uint32_t i, uint32_t stride, const V3& S, const F4& R)
{
    const F4& raw4 = s.pts[(i * stride) / 16];
    const V3 rp = qRot(R, V3{ raw4.x, raw4.y, raw4.z });
    return qRotInv(R, V3{ S.x * rp.x, S.y * rp.y, S.z * rp.z });
}


static void refPointsFace(const PtsIn& s, PtsOut& o)
{
    std::memset(&o, 0, sizeof(o));
    o.inds = U4{ NOIDX, NOIDX, NOIDX, NOIDX };
    const F4 poseQ = s.poseQ;
    const V3 poseP{ s.poseP.x, s.poseP.y, s.poseP.z };
    const float margin = s.poseP.w;
    const V3 dir = qRotInv(poseQ, V3{ s.dir.x, s.dir.y, s.dir.z });

    uint32_t numPoints = 0;
    V3 faceNormal{ 0, 0, 0 };
    V3 fp[4] = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } };

    const uint32_t w9 = s.core[2].y;
    const uint32_t poolN = w9 & 0xFF, stride = (w9 >> 8) & 0xFF;
    const V3 S{ u2f(s.core[0].z), u2f(s.core[0].w), u2f(s.core[1].x) };
    const F4 R{ u2f(s.core[1].y), u2f(s.core[1].z), u2f(s.core[1].w), u2f(s.core[2].x) };

    bool ok = !((s.core[0].x == 0 && s.core[0].y == 0) || poolN == 0);
    if (ok)
    {
        const float dEps1 = 0.99f;
        uint32_t index[3] = { NOIDX, NOIDX, NOIDX };
        float maxDot[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
        for (uint32_t i = 0; i < poolN; ++i)
        {
            const V3 p = getPoolPoint(s, i, stride, S, R);
            float dot = dot3(dir, p);
            uint32_t ind = i;
            for (uint32_t j = 0; j < 3; ++j)
                if (dot > maxDot[j])
                {
                    std::swap(maxDot[j], dot);
                    std::swap(index[j], ind);
                }
        }

        if (index[1] == NOIDX)
            ok = false;

        V3 p0{ 0, 0, 0 };
        if (ok)
        {
            p0 = getPoolPoint(s, index[0], stride, S, R);
            if (index[2] == NOIDX)
            {
                const V3 p1 = getPoolPoint(s, index[1], stride, S, R);
                const V3 n = nrmz(cross3(cross3(dir, sub(p1, p0)), sub(p1, p0)));
                const float dot = dot3(dir, n);
                if (dot > dEps1) faceNormal = n;
                else ok = false;
            }
            else
            {
                const V3 p1 = getPoolPoint(s, index[1], stride, S, R);
                const V3 p2 = getPoolPoint(s, index[2], stride, S, R);
                const V3 n0 = nrmz(cross3(sub(p1, p0), sub(p2, p0)));
                const float dot0 = dot3(dir, n0);
                if (dot0 > dEps1) faceNormal = n0;
                else if (dot0 < -dEps1) faceNormal = neg(n0);
                else
                {
                    const V3 n1 = nrmz(cross3(cross3(sub(p1, p0), dir), sub(p1, p0)));
                    const float dot1 = dot3(dir, n1);
                    if (dot1 > dEps1) faceNormal = n1;
                    else
                    {
                        const V3 n2 = nrmz(cross3(cross3(sub(p2, p0), dir), sub(p2, p0)));
                        const float dot2 = dot3(dir, n2);
                        if (dot2 > dEps1) faceNormal = n2;
                        else ok = false;
                    }
                }
            }
        }

        if (ok)
        {
            const float maxP = dot3(faceNormal, p0);
            for (uint32_t i = 0; i < poolN; ++i)
            {
                const V3 p = getPoolPoint(s, i, stride, S, R);
                const float pP = dot3(faceNormal, p);
                if (std::fabs(pP - maxP) < 1e-4f)
                {
                    V3 nfp[8];
                    nfp[0] = p;
                    for (uint32_t j = 0; j < numPoints; ++j) nfp[j + 1] = fp[j];
                    for (uint32_t j = 5; j < 8; ++j) nfp[j] = V3{ 0, 0, 0 };
                    for (uint32_t j = 0; j < numPoints; ++j)
                        if (dot3(nfp[0], dir) < dot3(nfp[j + 1], dir))
                            std::swap(nfp[0], nfp[j + 1]);
                    uint32_t inds[4];
                    numPoints = reduceRef(nfp, numPoints + 1, dir, true, inds);
                    V3 gathered[4];
                    for (uint32_t j = 0; j < numPoints; ++j) gathered[j] = nfp[inds[j]];
                    for (uint32_t j = 0; j < numPoints; ++j) fp[j] = gathered[j];
                }
            }
            if (numPoints < 2) { numPoints = 0; ok = false; }
        }
    }
    if (!ok) numPoints = 0;

    const V3 worldNormal = qRot(poseQ, faceNormal);
    o.counts = U4{ numPoints, 0, 0, 0 };
    o.normal = F4{ worldNormal.x, worldNormal.y, worldNormal.z, 0 };
    for (uint32_t i = 0; i < 4; ++i)
    {
        if (i < numPoints)
        {
            const V3 wp = add(add(qRot(poseQ, fp[i]), poseP), mul(worldNormal, margin));
            o.fp[i] = F4{ wp.x, wp.y, wp.z, 0 };
        }
    }
}


static bool eqBits(float a, float b) { return f2u(a) == f2u(b); }
static bool eqTol(float a, float b)
{
    // trig-path points only: Vulkan permits sin/cos absolute error up to
    // 2^-11 (~4.9e-4), amplified by the point magnitudes in the rotation
    if (std::isnan(a) && std::isnan(b)) return true;
    const float diff = std::fabs(a - b);
    return diff <= 2e-3f + 1e-3f * std::fmax(std::fabs(a), std::fabs(b));
}
static bool eqF4Bits(const F4& a, const F4& b)
{ return eqBits(a.x, b.x) && eqBits(a.y, b.y) && eqBits(a.z, b.z) && eqBits(a.w, b.w); }
static bool eqF4Tol(const F4& a, const F4& b)
{ return eqTol(a.x, b.x) && eqTol(a.y, b.y) && eqTol(a.z, b.z) && eqTol(a.w, b.w); }

static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed


} // namespace faceref


struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };
struct V3 { float x, y, z; };

struct CvxIn
{
    U4 metaA;
    U4 coreA[3];
    F4 poseAq, poseAp;   // poseAp.w = margin0
    U4 metaB;
    U4 coreB[3];
    F4 poseBq, poseBp;   // poseBp.w = margin1
    F4 cfg;              // x = contactDist, yzw = cullDir
    F4 ptsA[12];         // the host reference (and its memcpy into gjkref::GjkIn /
    F4 ptsB[12];         // eparef::EpaTIn) still reads these directly
    // P9.69b: the SHADER reads the hulls through Ptr<float4> at these addresses.
    // A cooked convex hull carries up to 255 points; the inline pool capped every
    // consumer of the shared shape layer at 12, which is why the engine kernels
    // could not take the eCONVEXMESH arm at all.  The arrays above stay because
    // the sibling reference layouts are memcpy'd out of this record.
    U4 ptsAddrA;      // x = lo, y = hi
    U4 ptsAddrB;
};
struct CvxOut
{
    U4 counts;           // x = numPoints, y = trig flags, z = emit tag
    F4 normal;
    F4 points[4];        // w = dist
};

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b); }   // P9.86: bitwise -- the +-0 class is closed

// Gu::generateContacts(ConvexShape, ConvexShape) (GuConvexSupport.h L1169-1201)
static void refCvxGen(const CvxIn& s, CvxOut& o)
{
    std::memset(&o, 0, sizeof(o));

    const float margin0 = s.poseAp.w, margin1 = s.poseBp.w;
    const float contactDist = s.cfg.x;
    const float maxDist = margin0 + margin1 + contactDist;

    gjkref::GjkIn gi;
    std::memcpy(&gi, &s, sizeof(gi));
    gi.cfg = gjkref::F4{ maxDist, 0, 0, 0 };
    gjkref::GjkOut go; gjkref::RefStats gst;
    gjkref::refGjk(gi, go, gst);

    float dist = go.result.x;
    V3 point0{ go.pointA.x, go.pointA.y, go.pointA.z };
    V3 point1{ go.pointB.x, go.pointB.y, go.pointB.z };
    V3 axis{ go.axis.x, go.axis.y, go.axis.z };

    if (dist < FLT_EPSILON)
    {
        eparef::EpaTIn ei;
        std::memcpy(&ei, &s, sizeof(ei));
        ei.cfg = eparef::F4{ 0, 0, 0, 0 };
        eparef::EpaTOut eo; eparef::RefStats est;
        eparef::refEpa(ei, eo, est);
        dist = eo.result.x;
        point0 = V3{ eo.pointA.x, eo.pointA.y, eo.pointA.z };
        point1 = V3{ eo.pointB.x, eo.pointB.y, eo.pointB.z };
        axis = V3{ eo.axis.x, eo.axis.y, eo.axis.z };
    }

    if (dist > maxDist)
        return;

    const float testEps = 1e-5f;
    const V3 cullDir{ s.cfg.y, s.cfg.z, s.cfg.w };
    if (cullDir.x * axis.x + cullDir.y * axis.y + cullDir.z * axis.z < -testEps)
        return;

    point0 = V3{ point0.x - axis.x * margin0, point0.y - axis.y * margin0, point0.z - axis.z * margin0 };
    point1 = V3{ point1.x + axis.x * margin1, point1.y + axis.y * margin1, point1.z + axis.z * margin1 };
    dist -= margin0 + margin1;
    (void)dist;

    // FaceClipper ctor: the two ConvexShape::contactFace wrapper calls
    uint32_t np[2] = { 0, 0 }, trig[2] = { 0, 0 };
    F4 wn[2]; F4 wfp[2][4];
    for (int sideI = 0; sideI < 2; ++sideI)
    {
        const bool B = sideI == 1;
        const V3 dirW = B ? axis : V3{ -axis.x, -axis.y, -axis.z };
        const V3 refW = B ? point1 : point0;
        const U4* core = B ? s.coreB : s.coreA;
        const F4 poseQ = B ? s.poseBq : s.poseAq;
        const F4 poseP = B ? s.poseBp : s.poseAp;
        const uint32_t type = B ? s.metaB.x : s.metaA.x;
        const F4* pool = B ? s.ptsB : s.ptsA;

        if (type == 6) // CORE_POINTS
        {
            faceref::PtsIn pi{};
            pi.meta = faceref::U4{ type, 0, 0, 0 };
            std::memcpy(pi.core, core, sizeof(pi.core));
            std::memcpy(&pi.poseQ, &poseQ, 16); std::memcpy(&pi.poseP, &poseP, 16);
            pi.dir = faceref::F4{ dirW.x, dirW.y, dirW.z, 0 };
            pi.refp = faceref::F4{ refW.x, refW.y, refW.z, 0 };
            std::memcpy(pi.pts, pool, 12 * 16);
            faceref::PtsOut po;
            faceref::refPointsFace(pi, po);
            np[sideI] = po.counts.x;
            std::memcpy(&wn[sideI], &po.normal, 16);
            std::memcpy(wfp[sideI], po.fp, sizeof(po.fp));
        }
        else
        {
            faceref::FaceIn fi{};
            fi.meta = faceref::U4{ type, 0, 0, 0 };
            std::memcpy(fi.core, core, sizeof(fi.core));
            std::memcpy(&fi.poseQ, &poseQ, 16); std::memcpy(&fi.poseP, &poseP, 16);
            fi.dir = faceref::F4{ dirW.x, dirW.y, dirW.z, 0 };
            fi.refp = faceref::F4{ refW.x, refW.y, refW.z, 0 };
            faceref::FaceOut fo;
            faceref::refFace(fi, fo);
            np[sideI] = fo.counts.x;
            trig[sideI] = fo.counts.y;
            std::memcpy(&wn[sideI], &fo.normal, 16);
            std::memcpy(wfp[sideI], fo.fp, sizeof(fo.fp));
        }
    }

    clipref::ClipIn ci{};
    ci.p0 = clipref::F4{ point0.x, point0.y, point0.z, 0 };
    ci.p1 = clipref::F4{ point1.x, point1.y, point1.z, 0 };
    ci.axis = clipref::F4{ axis.x, axis.y, axis.z, 0 };
    std::memcpy(&ci.n0, &wn[0], 16);
    std::memcpy(&ci.n1, &wn[1], 16);
    ci.counts = clipref::U4{ np[0], np[1], 0, 0 };
    std::memcpy(ci.fp0, wfp[0], sizeof(ci.fp0));
    std::memcpy(ci.fp1, wfp[1], sizeof(ci.fp1));
    clipref::ClipOut co;
    clipref::refClip(ci, co);

    o.counts = U4{ co.counts.x, trig[0] | (trig[1] << 1), 1, 0 };
    std::memcpy(&o.normal, &co.normal, 16);
    for (uint32_t i = 0; i < 4; ++i)
        if (i < co.counts.x) std::memcpy(&o.points[i], &co.pts[i], 16);
}

static void fillCoreCvx(std::mt19937& rng, uint32_t type, U4 core[3], F4 pts[12],
                        std::function<float(float, float)> rf)
{
    std::memset(core, 0, sizeof(U4) * 3);
    if (type == 1) core[0].x = f2u(rf(0.2f, 4));
    else if (type == 2 || type == 3)
    { core[0].x = f2u(rf(0.1f, 3)); core[0].y = f2u(rf(0.1f, 3)); core[0].z = f2u(rf(0.1f, 3)); }
    else if (type == 4 || type == 5)
    { core[0].x = f2u(rf(0.2f, 3)); core[0].y = f2u(rf(0.1f, 2)); }
    else if (type == 6)
    {
        const uint32_t np = 3 + rng() % 10;
        core[0].x = 48;
        core[0].z = f2u(rf(0.3f, 2)); core[0].w = f2u(rf(0.3f, 2));
        core[1].x = f2u(rf(0.3f, 2));
        float a[4]; float m2 = 0;
        for (int k = 0; k < 4; ++k) { a[k] = rf(-1, 1); m2 += a[k] * a[k]; }
        const float qi = 1.f / std::sqrt(m2 > 0 ? m2 : 1.f);
        core[1].y = f2u(a[0] * qi); core[1].z = f2u(a[1] * qi); core[1].w = f2u(a[2] * qi);
        core[2].x = f2u(a[3] * qi);
        core[2].y = np | (16u << 8);
        for (uint32_t p = 0; p < 12; ++p)
            pts[p] = F4{ rf(-1.5f, 1.5f), rf(-1.5f, 1.5f), rf(-1.5f, 1.5f), 0 };
    }
}

PXG_TEST(npCvxCoreCvxGen_contacts)
{
    std::mt19937 rng(171501);
    const uint32_t N = 3000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto randQuat = [&]() { float a[4]; float m2 = 0; for (int k = 0; k < 4; ++k) { a[k] = rf(-1, 1); m2 += a[k] * a[k]; } const float i = 1.f / std::sqrt(m2 > 0 ? m2 : 1.f); return F4{ a[0] * i, a[1] * i, a[2] * i, a[3] * i }; };

    std::vector<CvxIn> in(N);
    std::vector<CvxOut> ref(N);
    size_t cNone = 0, cOne = 0, cMulti = 0, cTrig = 0, cCull = 0, cEpaPath = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        CvxIn s{};
        const uint32_t typeA = rng() % 7, typeB = rng() % 7;
        s.metaA = U4{ typeA, 0, 0, 0 };
        s.metaB = U4{ typeB, 0, 0, 0 };
        s.poseAq = randQuat();
        s.poseBq = randQuat();
        fillCoreCvx(rng, typeA, s.coreA, s.ptsA, rf);
        fillCoreCvx(rng, typeB, s.coreB, s.ptsB, rf);

        const V3 pA{ rf(-1, 1), rf(-1, 1), rf(-1, 1) };
        const uint32_t cls = i % 4;
        float sep;
        if (cls == 0) sep = rf(0.f, 0.4f);        // overlap -> EPA path
        else if (cls == 1) sep = rf(3.5f, 4.5f);  // graze
        else if (cls == 2) sep = rf(4.5f, 6.f);   // near miss (contactDist window)
        else sep = rf(7.f, 10.f);                 // clear miss
        V3 sd{ rf(-1, 1), rf(-1, 1), rf(-1, 1) };
        { const float m = std::sqrt(sd.x * sd.x + sd.y * sd.y + sd.z * sd.z); if (m > 0) { sd = V3{ sd.x / m, sd.y / m, sd.z / m }; } else sd = V3{ 1, 0, 0 }; }
        const float m0 = (rng() % 3 == 0) ? 0.f : rf(0.01f, 0.3f);
        const float m1 = (rng() % 3 == 0) ? 0.f : rf(0.01f, 0.3f);
        s.poseAp = F4{ pA.x, pA.y, pA.z, m0 };
        s.poseBp = F4{ pA.x + sd.x * sep, pA.y + sd.y * sep, pA.z + sd.z * sep, m1 };

        V3 cull{ 0, 0, 0 };
        if (i % 9 == 2)
        {
            // live cull direction (triangle backface-style): sometimes against the axis
            cull = V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) };
            const float m = std::sqrt(cull.x * cull.x + cull.y * cull.y + cull.z * cull.z);
            if (m > 0) cull = V3{ cull.x / m, cull.y / m, cull.z / m };
            ++cCull;
        }
        s.cfg = F4{ rf(0.05f, 1.0f), cull.x, cull.y, cull.z };

        if (i % 11 == 4)
        {
            // identity poses + axis-aligned cap contact: the trig patch path
            s.metaA = U4{ 4, 0, 0, 0 }; s.metaB = U4{ (rng() % 2) ? 4u : 2u, 0, 0, 0 };
            fillCoreCvx(rng, s.metaA.x, s.coreA, s.ptsA, rf);
            fillCoreCvx(rng, s.metaB.x, s.coreB, s.ptsB, rf);
            s.poseAq = F4{ 0, 0, 0, 1 };
            s.poseBq = F4{ 0, 0, 0, 1 };
            const float hA = faceref::u2f(s.coreA[0].x);
            const float wB = faceref::u2f(s.coreB[0].x);
            s.poseAp = F4{ 0.f + pA.x, pA.y, pA.z, m0 };
            s.poseBp = F4{ pA.x + hA * 0.5f + wB * 0.5f + rf(-0.05f, 0.3f), pA.y, pA.z, m1 };
        }

        in[i] = s;
        refCvxGen(s, ref[i]);
        if (ref[i].counts.x == 0) ++cNone;
        else if (ref[i].counts.x == 1) ++cOne;
        else ++cMulti;
        if (ref[i].counts.y != 0) ++cTrig;
        if (ref[i].counts.z != 0 && ref[i].counts.x != 0) {}
    }
    (void)cEpaPath;

    std::printf("[ info ] cvg cov: none %zu one %zu multi %zu trig %zu cull %zu\n",
                cNone, cOne, cMulti, cTrig, cCull);

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 31) / 32;
    std::vector<CvxOut> outInit(N);
    // The shader reads hull points by DEVICE ADDRESS (P9.69b), so stage them in
    // their own buffer: 24 slots per test, A's twelve then B's twelve.
    std::vector<F4> ptsHost(size_t(N) * 24);
    for (uint32_t i = 0; i < N; ++i)
    {
        std::memcpy(&ptsHost[size_t(i) * 24 + 0], in[i].ptsA, sizeof(F4) * 12);
        std::memcpy(&ptsHost[size_t(i) * 24 + 12], in[i].ptsB, sizeof(F4) * 12);
    }
    PxgBufferHandle bPts = createBufferFrom(dev, ptsHost);
    const uint64_t ptsBase = dev->getBufferDeviceAddress(bPts);
    for (uint32_t i = 0; i < N; ++i)
    {
        const uint64_t a = ptsBase + uint64_t(i) * 24ull * sizeof(F4);
        const uint64_t b = a + 12ull * sizeof(F4);
        in[i].ptsAddrA = U4{ uint32_t(a & 0xffffffffu), uint32_t(a >> 32), 0, 0 };
        in[i].ptsAddrB = U4{ uint32_t(b & 0xffffffffu), uint32_t(b >> 32), 0, 0 };
    }

    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npCvxCoreCvxGen.spv", "cvgContacts",
             { (uint32_t)sizeof(CvxIn), (uint32_t)sizeof(CvxOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<CvxOut>(dev, bOut, N);
    size_t mm = 0, trigUnstable = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const CvxOut& r = ref[i]; const CvxOut& q = g[i];
        // trig-flagged cases whose COUNTS differ: the rotatePoints
        // acos/sin/cos deviation (not bit-exact-portable, same on CUDA
        // itself) can flip a FaceClipper band decision downstream, changing
        // the emitted point count.  Those razor cases are excluded from the
        // bitwise compare but counted and CAPPED below.
        const bool trigMeta = r.counts.y == q.counts.y && r.counts.z == q.counts.z &&
                              eqFZ(r.normal.x, q.normal.x) && eqFZ(r.normal.y, q.normal.y) &&
                              eqFZ(r.normal.z, q.normal.z); // trig can't alter flags/normal (review 171)
        if (r.counts.y != 0 && trigMeta && r.counts.x != q.counts.x)
        {
            ++trigUnstable;
            continue;
        }
        bool ok = r.counts.x == q.counts.x && r.counts.y == q.counts.y && r.counts.z == q.counts.z &&
                  eqFZ(r.normal.x, q.normal.x) && eqFZ(r.normal.y, q.normal.y) && eqFZ(r.normal.z, q.normal.z);
        const bool trig = r.counts.y != 0;
        for (uint32_t j = 0; j < 4 && ok; ++j)
            ok = trig ? faceref::eqF4Tol(*(const faceref::F4*)&r.points[j], *(const faceref::F4*)&q.points[j])
                      : (eqFZ(r.points[j].x, q.points[j].x) && eqFZ(r.points[j].y, q.points[j].y) &&
                         eqFZ(r.points[j].z, q.points[j].z) && eqFZ(r.points[j].w, q.points[j].w));
        if (!ok && r.counts.y != 0 && trigMeta)
        {
            // same class: a trig-flipped reduce decision can also REORDER
            // the emitted points at equal counts
            ++trigUnstable;
            continue;
        }
        if (!ok && mm < 5)
            std::printf("MM %u nR %u nG %u tR %u tG %u p0R (%g %g %g %g) p0G (%g %g %g %g)\n",
                        i, r.counts.x, q.counts.x, r.counts.y, q.counts.y,
                        r.points[0].x, r.points[0].y, r.points[0].z, r.points[0].w,
                        q.points[0].x, q.points[0].y, q.points[0].z, q.points[0].w);
        if (!ok) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "cvgContacts: %zu mismatches", mm);
    PXG_CHECK_MSG(trigUnstable <= 8, "cvgContacts: %zu trig-unstable cases (cap 8)", trigUnstable);

    bool cov = cNone > N / 10 && cOne > N / 20 && cMulti > N / 20 && cTrig > 20 && cCull > N / 12;
    PXG_CHECK_MSG(cov, "cvgContacts: coverage thin (none %zu one %zu multi %zu trig %zu)",
                  cNone, cOne, cMulti, cTrig);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

// ===========================================================================
// P9.86c -- pxgSlangCvxCoreCvx gjkDistanceQ computePoints: the zero-seeded pA/pB.
//
// The CUDA (GuRefGjkEpa.h:353-360) seeds pA/pB with +0 and only adds, so they never
// hold -0.  llvmpipe partially unrolls the port's loop (NIR guesses 4 trips from
// GVert verts[4]) and drops the first `0 + t`; the port therefore canonicalises pA/pB
// with a select.  A point core's world vertex with a -0 lane makes the dropped seed
// visible: sphere at poseP = (-0, -e, -e) rotated by q = (-.5,-.5,.5,.5) gives
// vA = (-0, -e, -e); a zero-radius segment of half-length len/2 along x at the origin
// gives the 45-degree axis (+0, -r, -r), GJK ends on the 2-vertex simplex, and with
// fl(r*m0) == e exactly (m0 found by search) point0 = (pA.x, +0, +0), point1 = 0.
// clipNone's separation w = dot(axis, point0 - point1) is +0 under the CUDA and -0
// when pA.x = -0 (every product -0).  Everything else in the record is bit-identical.
// ===========================================================================
PXG_TEST(npCvxCoreCvxGen_seedFoldZeroSign)
{
    struct Fam { float e, len; };
    const Fam fams[] = { { 0.25f, 2.0f }, { 0.125f, 1.0f }, { 0.5f, 4.0f } };
    std::vector<CvxIn> in;
    std::vector<CvxOut> ref;
    std::vector<uint32_t> mBits;
    size_t cSeedFoldSeen = 0, cOnlyW = 0;
    for (const Fam& f : fams)
    {
        CvxIn s{};
        s.metaA = U4{ 0, 0, 0, 0 };                       // CORE_POINT (sphere core)
        s.poseAq = F4{ -0.5f, -0.5f, 0.5f, 0.5f };
        s.metaB = U4{ 1, 0, 0, 0 };                       // CORE_SEGMENT, zero margin
        s.coreB[0].x = f2u(f.len);
        s.poseBq = F4{ 0, 0, 0, 1 };
        s.poseBp = F4{ 0, 0, 0, 0 };
        s.cfg = F4{ 0.1f, 0, 0, 0 };
        const float seed = std::sqrt(2.0f) * f.e;
        bool found = false;
        for (int k = 0; k <= 128 && !found; ++k)
        {
            float m = seed;
            const int steps = (k + 1) / 2;
            for (int j = 0; j < steps; ++j) m = std::nextafter(m, (k & 1) ? 1e30f : -1e30f);
            s.poseAp = F4{ -0.0f, -f.e, -f.e, m };
            CvxOut a, t;
            refCvxGen(s, a);
            gjkref::gSeedFoldVariant = 1;
            refCvxGen(s, t);
            gjkref::gSeedFoldVariant = 0;
            if (a.counts.x == 1 && a.counts.y == 0 && f2u(a.points[0].w) == 0u &&
                f2u(t.points[0].w) == 0x80000000u)
            {
                found = true;
                ++cSeedFoldSeen;
                // the twin must change ONLY the separation word
                bool same = std::memcmp(&a.counts, &t.counts, sizeof(a.counts)) == 0 &&
                            std::memcmp(&a.normal, &t.normal, sizeof(a.normal)) == 0 &&
                            f2u(a.points[0].x) == f2u(t.points[0].x) &&
                            f2u(a.points[0].y) == f2u(t.points[0].y) &&
                            f2u(a.points[0].z) == f2u(t.points[0].z) &&
                            std::memcmp(&a.points[1], &t.points[1], sizeof(F4) * 3) == 0;
                if (same) ++cOnlyW;
                in.push_back(s);
                ref.push_back(a);
                mBits.push_back(f2u(m));
            }
        }
    }
    PXG_CHECK_MSG(cSeedFoldSeen == 3, "cvg seed-fold: only %zu of 3 directed cases change under the twin",
                  cSeedFoldSeen);
    PXG_CHECK_MSG(cOnlyW == cSeedFoldSeen, "cvg seed-fold: twin changed more than points[0].w (%zu of %zu)",
                  cOnlyW, cSeedFoldSeen);
    for (size_t i = 0; i < mBits.size(); ++i)
        std::printf("[ info ] cvg seed-fold case %zu: m0 0x%08X\n", i, mBits[i]);
    if (in.empty()) return;

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t N = uint32_t(in.size());
    std::vector<F4> ptsHost(size_t(N) * 24, F4{ 0, 0, 0, 0 });
    PxgBufferHandle bPts = createBufferFrom(dev, ptsHost);
    const uint64_t ptsBase = dev->getBufferDeviceAddress(bPts);
    for (uint32_t i = 0; i < N; ++i)
    {
        const uint64_t a = ptsBase + uint64_t(i) * 24ull * sizeof(F4);
        const uint64_t b = a + 12ull * sizeof(F4);
        in[i].ptsAddrA = U4{ uint32_t(a & 0xffffffffu), uint32_t(a >> 32), 0, 0 };
        in[i].ptsAddrB = U4{ uint32_t(b & 0xffffffffu), uint32_t(b >> 32), 0, 0 };
    }
    std::vector<CvxOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);
    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npCvxCoreCvxGen.spv", "cvgContacts",
             { (uint32_t)sizeof(CvxIn), (uint32_t)sizeof(CvxOut) }, sizeof(P));
    k.dispatch((N + 31) / 32, { bIn, bOut }, prm);
    dev->flush();
    const auto g = downloadBuffer<CvxOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const CvxOut& r = ref[i]; const CvxOut& q = g[i];
        bool ok = r.counts.x == q.counts.x && r.counts.y == q.counts.y && r.counts.z == q.counts.z &&
                  eqFZ(r.normal.x, q.normal.x) && eqFZ(r.normal.y, q.normal.y) && eqFZ(r.normal.z, q.normal.z);
        for (uint32_t j = 0; j < 4; ++j)
            ok = ok && eqFZ(r.points[j].x, q.points[j].x) && eqFZ(r.points[j].y, q.points[j].y) &&
                 eqFZ(r.points[j].z, q.points[j].z) && eqFZ(r.points[j].w, q.points[j].w);
        if (!ok)
        {
            ++mm;
            std::printf("MM seed-fold %u n %u/%u w 0x%08X/0x%08X p0 (%g %g %g)/(%g %g %g)\n", i,
                        r.counts.x, q.counts.x, f2u(r.points[0].w), f2u(q.points[0].w),
                        r.points[0].x, r.points[0].y, r.points[0].z, q.points[0].x, q.points[0].y, q.points[0].z);
        }
    }
    PXG_CHECK_MSG(mm == 0, "cvg seed-fold: %zu mismatches", mm);
    for (PxgBufferHandle h : { bIn, bOut, bPts }) dev->destroyBuffer(h);
}

// ===========================================================================
// P9.86c -- convexCoreTrimeshNphase_Kernel32 (cvxCoreTriNpBda.slang), the SHIPPED
// kernel, run END TO END against a host mirror built from the verified layers.
//
// WHY IT EXISTS.  Commit 24a7397 changed the kernel's transform0in1 from
// rotateInv(q1, p0 - p1) to rotate(conj(q1), p0 - p1), which is what the CUDA
// spells (convexCoreCollision.cu:980 -> PxTransform::transformInv, PxTransform.h:
// 167-168: `qinv = q.getConjugate(); p = qinv.rotate(src.p - p)`).  The two are
// equal on every non-zero value (round-to-nearest is symmetric under negation) and
// differ only in the SIGN OF A ZERO component: rotateInv subtracts the cross term
// that rotate(conj) adds with its operands negated, so an all-zero component comes
// out -0 where the CUDA gives +0.  Until this test no bitwise fixture loaded
// cvxCoreTriNpBda.spv at all, so the revert could not die.
//
// HOW THE SIGN REACHES A STORED WORD -- and why the obvious inputs cannot carry it.
// A point core (sphere) never shows it: GJK's closest point always adds slot 3's
// (+0)*0 and computePoints starts from +0, and the single clipNone point
// (point0+point1)*0.5 absorbs a lone signed zero.  What does carry it:
//   * a SEGMENT core (2-point contact face) that PIERCES the triangle, so EPA
//     runs and ends on the facet whose three Minkowski points share one y --
//     its normal's z is then an exact zero and the axis gets z = -0;
//   * transform0in1 whose rotation is about the mesh z axis with zero x/y
//     components, so the segment's +len/2 end keeps z = (-0) + aToBp.z = aToBp.z;
//   * a mesh rotation q1 = PxQuat(-PxHalfPi, (-0,0,1)) = (+0,-0,-h,h) and a mesh
//     position whose z is -0 (the `+ p1` in addPoints absorbs any other value).
// The face point of that end then clips unchanged (t0 = +0 exactly, t1 < 0 because
// the margin 0.5 exceeds the end's height above the plane), survives both
// reducePoints as the point farthest from the deepest one, and addPoints stores
// z = -c for c = aToBp.z: 0x80000000 under the CUDA, 0x00000000 under the revert.
// Everything else is bit-identical.  The directed family below is exactly that
// geometry; the host mirror is run TWICE (CUDA and revert spelling) and a coverage
// counter requires the two expected images to differ, so a later edit cannot make
// the input stop exercising the fix without failing loudly.
//
// THE MIRROR is not new arithmetic: generateContacts is refCvxGen above (verified
// bitwise against the same shape layers by npCvxCoreCvxGen_contacts), the
// accumulator is testNpCvxCoreWarpAcc.cpp's refWarpAcc verbatim (npCvxCoreWarpAcc
// verifies it against pxgSlangCvxCoreAcc.slang, which this kernel includes), and
// the bounds / mesh-scale glue is testNpCvxBvhGlue.cpp's (npCvxBvhGlue).  What is
// new is the wiring those fixtures cannot see: transform0in1, makeConvexShape's
// eCONVEXCORE arm, computeBounds + fattenFast, the one-leaf BV32 walk, the lane
// staging, and the multi-patch tail's stream layout.  A GENERIC set (point / segment
// / box / ellipsoid cores over a two-triangle quad, rotated and scaled meshes, both
// slot orders) compares that wiring bitwise too.
// ===========================================================================

namespace trinpacc {

// Gu::Contact32::addPoints -- testNpCvxCoreWarpAcc.cpp's refWarpAcc, VERBATIM.
struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };
struct V3 { float x, y, z; };

const uint32_t WARP_SIZE = 32, MAX_PATCHES = 16, MAX_PATCH_POINTS = 4, MAX_CC = 4;
const uint32_t NOIDX = 0xFFFFFFFFu;
const float SAME_NORMAL = 0.999f;

struct WIn
{
    U4 meta;
    F4 transformQ, transformP;
    U4 nums[2][8];
    F4 pts[2][128];
    F4 nrm[2][32];
};
struct WOut
{
    U4 counts;
    F4 patchN[MAX_PATCHES];
    F4 pool[MAX_PATCHES * MAX_PATCH_POINTS];
};

static V3 vadd(V3 a, V3 b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 vsub(V3 a, V3 b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static float vdot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 vcross(V3 a, V3 b)
{ return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static float vmag(V3 v) { return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }
static V3 qRot(const F4& q, V3 v)
{
    const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z;
    const float w2 = q.w * q.w - 0.5f, d2 = q.x * vx + q.y * vy + q.z * vz;
    return V3{ vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d2,
               vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d2,
               vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d2 };
}
static uint32_t popc(uint32_t v) { uint32_t c = 0; while (v) { v &= v - 1; ++c; } return c; }

static uint32_t minIdx(const float v[WARP_SIZE], uint32_t mask, float& best)
{
    best = FLT_MAX; uint32_t idx = NOIDX;
    for (uint32_t l = 0; l < WARP_SIZE; ++l)
        if ((mask & (1u << l)) && v[l] < best) { best = v[l]; idx = l; }
    return idx;
}
static uint32_t maxIdx(const float v[WARP_SIZE], uint32_t mask, float& best)
{
    best = -FLT_MAX; uint32_t idx = NOIDX;
    for (uint32_t l = 0; l < WARP_SIZE; ++l)
        if ((mask & (1u << l)) && v[l] > best) { best = v[l]; idx = l; }
    return idx;
}

static void refWarpAcc(const WIn& s, WOut& o)
{
    std::memset(&o, 0, sizeof(o));

    uint32_t mNumPatches = 0, mNumPoints = 0;
    V3 patchNormal[MAX_PATCHES];
    uint32_t patchCount[MAX_PATCHES];
    F4 mPoints[MAX_PATCHES * MAX_PATCH_POINTS];
    for (uint32_t i = 0; i < MAX_PATCHES; ++i) { patchNormal[i] = V3{ 0, 0, 0 }; patchCount[i] = 0; }
    std::memset(mPoints, 0, sizeof(mPoints));

    const uint32_t numCalls = s.meta.x < 2 ? s.meta.x : 2;
    for (uint32_t call = 0; call < numCalls; ++call)
    {
        uint32_t laneNum[WARP_SIZE];
        for (uint32_t l = 0; l < WARP_SIZE; ++l)
        {
            const uint32_t n = (&s.nums[call][l / 4].x)[l & 3];
            laneNum[l] = n < MAX_CC ? n : MAX_CC;
        }

        F4 newPts[WARP_SIZE * MAX_CC];
        V3 newNrm[WARP_SIZE * MAX_CC];
        for (uint32_t i = 0; i < WARP_SIZE * MAX_CC; ++i)
        { newPts[i] = F4{ 0, 0, 0, FLT_MAX }; newNrm[i] = V3{ 0, 0, 0 }; }

        uint32_t prefix = 0, totalPoints = 0;
        for (uint32_t l = 0; l < WARP_SIZE; ++l)
        {
            prefix += laneNum[l];
            const uint32_t off = prefix - laneNum[l];
            const V3 worldNormal = qRot(s.transformQ, V3{ s.nrm[call][l].x, s.nrm[call][l].y, s.nrm[call][l].z });
            for (uint32_t j = 0; j < laneNum[l]; ++j)
            {
                const F4 lp = s.pts[call][l * MAX_CC + j];
                const V3 wp = vadd(qRot(s.transformQ, V3{ lp.x, lp.y, lp.z }),
                                   V3{ s.transformP.x, s.transformP.y, s.transformP.z });
                newPts[off + j] = F4{ wp.x, wp.y, wp.z, lp.w };
                newNrm[off + j] = worldNormal;
            }
            if (l == WARP_SIZE - 1) totalPoints = prefix;
        }

        const uint32_t numFullWarps = (totalPoints + WARP_SIZE - 1) / WARP_SIZE;
        for (uint32_t round = 0; round < numFullWarps; ++round)
        {
            F4 lp[WARP_SIZE]; V3 ln[WARP_SIZE];
            uint32_t allMask = 0;
            for (uint32_t l = 0; l < WARP_SIZE; ++l)
            {
                const uint32_t i = round * WARP_SIZE + l;
                if (i < totalPoints) { allMask |= (1u << l); lp[l] = newPts[i]; ln[l] = newNrm[i]; }
                else { lp[l] = F4{ 0, 0, 0, FLT_MAX }; ln[l] = V3{ 0, 0, 0 }; }
            }

            uint32_t guard = 0;
            while (allMask != 0 && guard < 40)
            {
                ++guard;
                float dists[WARP_SIZE];
                for (uint32_t l = 0; l < WARP_SIZE; ++l) dists[l] = lp[l].w;
                float minDist;
                const uint32_t minDistThread = minIdx(dists, allMask, minDist);
                const V3 minDistNormal = ln[minDistThread];

                uint32_t sameNormalMask = 0;
                for (uint32_t l = 0; l < WARP_SIZE; ++l)
                    if (vdot(ln[l], minDistNormal) > SAME_NORMAL) sameNormalMask |= (1u << l);
                sameNormalMask &= allMask;

                float tmpv[WARP_SIZE]; float md;
                for (uint32_t l = 0; l < WARP_SIZE; ++l) tmpv[l] = lp[l].w;
                const uint32_t pi0 = minIdx(tmpv, sameNormalMask, md);
                const V3 p0{ lp[pi0].x, lp[pi0].y, lp[pi0].z };
                for (uint32_t l = 0; l < WARP_SIZE; ++l) tmpv[l] = vmag(vsub(p0, V3{ lp[l].x, lp[l].y, lp[l].z }));
                const uint32_t pi1 = maxIdx(tmpv, sameNormalMask, md);
                const V3 p1{ lp[pi1].x, lp[pi1].y, lp[pi1].z };
                const V3 sideN = vcross(minDistNormal, vsub(p1, p0));
                for (uint32_t l = 0; l < WARP_SIZE; ++l) tmpv[l] = vdot(sideN, vsub(V3{ lp[l].x, lp[l].y, lp[l].z }, p0));
                const uint32_t pi2 = minIdx(tmpv, sameNormalMask, md);
                const uint32_t pi3 = maxIdx(tmpv, sameNormalMask, md);
                uint32_t bestPointsMask = (1u << pi0) | (1u << pi1) | (1u << pi2) | (1u << pi3);

                F4 tmpBuffer[8];
                std::memset(tmpBuffer, 0, sizeof(tmpBuffer));
                uint32_t count = popc(bestPointsMask);
                for (uint32_t l = 0; l < WARP_SIZE; ++l)
                    if (bestPointsMask & (1u << l))
                        tmpBuffer[popc(bestPointsMask & ((1u << l) - 1))] = lp[l];

                allMask &= ~sameNormalMask;

                uint32_t patchIndex = NOIDX;
                {
                    float dots[WARP_SIZE];
                    for (uint32_t l = 0; l < WARP_SIZE; ++l)
                    {
                        const V3 pn = l < mNumPatches ? patchNormal[l] : V3{ 0, 0, 0 };
                        dots[l] = vdot(minDistNormal, pn);
                    }
                    float maxDot;
                    const uint32_t maxDotThread = maxIdx(dots, 0xFFFFFFFFu, maxDot);

                    if (maxDot > SAME_NORMAL)
                        patchIndex = maxDotThread;
                    else if (mNumPatches < MAX_PATCHES)
                    {
                        patchNormal[mNumPatches] = minDistNormal;
                        patchCount[mNumPatches] = 0;
                        ++mNumPatches;
                        patchIndex = mNumPatches - 1;
                    }
                    else
                    {
                        float pMax[WARP_SIZE]; uint32_t pClosest[WARP_SIZE];
                        for (uint32_t l = 0; l < WARP_SIZE; ++l)
                        {
                            const V3 pn = l < mNumPatches ? patchNormal[l] : V3{ 0, 0, 0 };
                            float patchMaxDot = -FLT_MAX;
                            uint32_t closestPatchThread = NOIDX;
                            for (uint32_t i = l + 1; i < MAX_PATCHES; ++i)
                            {
                                const float d = vdot(pn, patchNormal[i]);
                                if (patchMaxDot < d) { patchMaxDot = d; closestPatchThread = i; }
                            }
                            pMax[l] = patchMaxDot; pClosest[l] = closestPatchThread;
                        }
                        float patchMaxDot;
                        const uint32_t patchMaxDotThread = maxIdx(pMax, 0xFFFFFFFFu, patchMaxDot);

                        if (patchMaxDot > maxDot)
                        {
                            const uint32_t closestPatchThread = pClosest[patchMaxDotThread];
                            const float dist0 = mPoints[patchMaxDotThread * MAX_PATCH_POINTS].w;
                            const float dist1 = mPoints[closestPatchThread * MAX_PATCH_POINTS].w;
                            const uint32_t pi = dist0 > dist1 ? patchMaxDotThread : closestPatchThread;
                            mNumPoints -= patchCount[pi];
                            patchCount[pi] = 0;
                            patchNormal[pi] = minDistNormal;
                            patchIndex = pi;
                        }
                    }
                }

                if (patchIndex == NOIDX)
                    continue;

                for (uint32_t j = 0; j < patchCount[patchIndex]; ++j)
                    tmpBuffer[count + j] = mPoints[patchIndex * MAX_PATCH_POINTS + j];
                count += patchCount[patchIndex];

                const uint32_t patchPointMask = count >= 32 ? 0xFFFFFFFFu : ((1u << count) - 1);
                F4 pp[WARP_SIZE];
                for (uint32_t l = 0; l < WARP_SIZE; ++l)
                    pp[l] = l < count ? tmpBuffer[l] : F4{ 0, 0, 0, 0 };
                for (uint32_t l = 0; l < WARP_SIZE; ++l) tmpv[l] = pp[l].w;
                float md2;
                const uint32_t qi0 = minIdx(tmpv, patchPointMask, md2);
                const uint32_t deepest2 = qi0;
                const V3 q0{ pp[qi0].x, pp[qi0].y, pp[qi0].z };
                for (uint32_t l = 0; l < WARP_SIZE; ++l) tmpv[l] = vmag(vsub(q0, V3{ pp[l].x, pp[l].y, pp[l].z }));
                const uint32_t qi1 = maxIdx(tmpv, patchPointMask, md2);
                const V3 q1{ pp[qi1].x, pp[qi1].y, pp[qi1].z };
                const V3 sideN2 = vcross(minDistNormal, vsub(q1, q0));
                for (uint32_t l = 0; l < WARP_SIZE; ++l) tmpv[l] = vdot(sideN2, vsub(V3{ pp[l].x, pp[l].y, pp[l].z }, q0));
                const uint32_t qi2 = minIdx(tmpv, patchPointMask, md2);
                const uint32_t qi3 = maxIdx(tmpv, patchPointMask, md2);
                bestPointsMask = (1u << qi0) | (1u << qi1) | (1u << qi2) | (1u << qi3);

                const uint32_t newCount = popc(bestPointsMask);
                for (uint32_t l = 0; l < WARP_SIZE; ++l)
                    if (bestPointsMask & (1u << l))
                        mPoints[patchIndex * MAX_PATCH_POINTS + popc(bestPointsMask & ((1u << l) - 1))] = pp[l];

                const uint32_t dOff = popc(bestPointsMask & ((1u << deepest2) - 1));
                const F4 tv = mPoints[patchIndex * MAX_PATCH_POINTS];
                mPoints[patchIndex * MAX_PATCH_POINTS] = mPoints[patchIndex * MAX_PATCH_POINTS + dOff];
                mPoints[patchIndex * MAX_PATCH_POINTS + dOff] = tv;

                mNumPoints -= patchCount[patchIndex];
                patchCount[patchIndex] = newCount;
                mNumPoints += patchCount[patchIndex];
            }
        }
    }

    o.counts = U4{ mNumPatches, mNumPoints, 0, 0 };
    for (uint32_t i = 0; i < MAX_PATCHES; ++i)
    {
        const V3 n = i < mNumPatches ? patchNormal[i] : V3{ 0, 0, 0 };
        uint32_t c = i < mNumPatches ? patchCount[i] : 0;
        float cf; std::memcpy(&cf, &c, 4);
        o.patchN[i] = F4{ n.x, n.y, n.z, cf };
    }
    for (uint32_t i = 0; i < MAX_PATCHES * MAX_PATCH_POINTS; ++i)
    {
        const uint32_t pi = i / MAX_PATCH_POINTS;
        const bool live = pi < mNumPatches && (i % MAX_PATCH_POINTS) < patchCount[pi];
        o.pool[i] = live ? mPoints[i] : F4{ 0, 0, 0, 0 };
    }
}

} // namespace trinpacc

namespace trinp {

// The engine ABI rows this kernel reads, mirrored with their sizes pinned.
struct GpuShape                 // PxgShape, 48 B
{
    float sx, sy, sz;           // PxMeshScale::scale -- for eCONVEXCORE the core data
    float rx, ry, rz, rw;       // PxMeshScale::rotation -- for eCONVEXCORE .w is the margin
    uint32_t materialIndex;
    uint32_t hullLo, hullHi;    // eCONVEXCORE: the PxConvexCore::Type; mesh: the blob
    uint32_t type;
    uint32_t particleOrSoftbodyId;
};
static_assert(sizeof(GpuShape) == 48, "PxgShape mirror");
struct CmInput { uint32_t shapeRef0, shapeRef1, tcRef0, tcRef1; };
static_assert(sizeof(CmInput) == 16, "PxgContactManagerInput mirror");
struct CachedXf { float qx, qy, qz, qw, px, py, pz; uint32_t flags; };
static_assert(sizeof(CachedXf) == 32, "PxsCachedTransform mirror");
struct MaterialH { float dyn, sta, rest, damping; uint32_t flagsModes0, modes1; };
static_assert(sizeof(MaterialH) == 24, "PxsMaterialData mirror");

// cvxCoreTriNpBda.slang's CvxCoreTriParams == the CUDA parameter list (:919-937).
struct TriPush
{
    uint32_t numTests, pad4;
    uint64_t cmInputs, cmOutputs, shapes, transformCache, contactDistance, materials;
    uint64_t contactStream, patchStream, counters, touchFlags, patchFlags;
    uint64_t startPatches, startPoints, startForces;
    uint32_t patchBytesLimit, contactBytesLimit, forceBytesLimit, pad132;
};
static_assert(sizeof(TriPush) == 136, "the kCvxCorePlane ABI row is 136 bytes");
static_assert(offsetof(TriPush, cmInputs) == 8 && offsetof(TriPush, materials) == 48 &&
              offsetof(TriPush, counters) == 72 && offsetof(TriPush, startForces) == 112 &&
              offsetof(TriPush, patchBytesLimit) == 120 && offsetof(TriPush, forceBytesLimit) == 128,
              "CvxCoreTriParams offsets");

const uint32_t CORE_POINT = 0, CORE_SEGMENT = 1, CORE_BOX = 2, CORE_ELLIPSOID = 3;   // PxConvexCore::Type
const uint32_t MAX_PATCHES = 16, MAX_PATCH_POINTS = 4;

// A cooked mesh, by hand: ONE BV32 node whose child 0 is a leaf holding every
// triangle (readTriangleMesh, triangleMesh.cuh:68-92).
struct MeshH
{
    std::vector<F4> verts;
    std::vector<U4> tris;
    F4 nodeMin, nodeMax;
};

struct TriCase
{
    uint32_t coreType;          // PxConvexCore::Type
    V3 coreData;                // PxMeshScale::scale bytes (length / extents / radii)
    float margin;               // PxMeshScale::rotation.w
    F4 q0; V3 p0;               // the core's pose
    F4 q1; V3 p1;               // the mesh's pose
    V3 meshScale; F4 meshRot;   // PxMeshScale of the mesh shape
    uint32_t mesh;              // index into the mesh list
    float cd0, cd1;             // contactDistance of the two cache entries
    bool swapped;               // the pair arrives mesh-first (checkTypes swaps it)
    uint32_t kind;              // 0 the design case, 1 the directed family, 2 generic
};

struct TriExpect
{
    uint32_t numPatches, numPoints;
    V3 normal[MAX_PATCHES];
    uint32_t count[MAX_PATCHES];
    F4 contacts[MAX_PATCHES * MAX_PATCH_POINTS];   // the contact stream, patch-major
    // what the directed counters look at
    F4 aToBq; V3 aToBp;
    bool intersect;
    CvxOut gen[4];
};

static float u2fT(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static gjkref::V3 toG(V3 v) { return gjkref::V3{ v.x, v.y, v.z }; }
static V3 fromG(gjkref::V3 v) { return V3{ v.x, v.y, v.z }; }
static gjkref::F4 toG(F4 q) { return gjkref::F4{ q.x, q.y, q.z, q.w }; }

// PxQuat operator* (PxQuat.h:388-392)
static F4 quatMul(F4 a, F4 b)
{
    return F4{ a.w * b.x + b.w * a.x + a.y * b.z - b.y * a.z,
               a.w * b.y + b.w * a.y + a.z * b.x - b.z * a.x,
               a.w * b.z + b.w * a.z + a.x * b.y - b.x * a.y,
               a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z };
}

// PxMeshScale::toMat33 -- testNpCvxBvhGlue.cpp's quatToMat33 / meshScaleMat / matT.
static void quatToMat33(F4 q, V3& c0, V3& c1, V3& c2)
{
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    const float x2 = x + x, y2 = y + y, z2 = z + z;
    const float xx = x2 * x, yy = y2 * y, zz = z2 * z;
    const float xy = x2 * y, xz = x2 * z, xw = x2 * w;
    const float yz = y2 * z, yw = y2 * w, zw = z2 * w;
    c0 = { 1.0f - yy - zz, xy + zw, xz - yw };
    c1 = { xy - zw, 1.0f - xx - zz, yz + xw };
    c2 = { xz + yw, yz - xw, 1.0f - xx - yy };
}
static V3 addv(V3 a, V3 b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 subv(V3 a, V3 b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 mulv(V3 a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; }
static void meshScaleMat(V3 scale, F4 rot, V3& m0, V3& m1, V3& m2)
{
    V3 r0, r1, r2; quatToMat33(rot, r0, r1, r2);
    const V3 t0 = { r0.x * scale.x, r1.x * scale.x, r2.x * scale.x };
    const V3 t1 = { r0.y * scale.y, r1.y * scale.y, r2.y * scale.y };
    const V3 t2 = { r0.z * scale.z, r1.z * scale.z, r2.z * scale.z };
    m0 = addv(addv(mulv(t0, r0.x), mulv(t1, r0.y)), mulv(t2, r0.z));
    m1 = addv(addv(mulv(t0, r1.x), mulv(t1, r1.y)), mulv(t2, r1.z));
    m2 = addv(addv(mulv(t0, r2.x), mulv(t1, r2.y)), mulv(t2, r2.z));
}
static V3 matT(V3 m0, V3 m1, V3 m2, V3 v) { return addv(addv(mulv(m0, v.x), mulv(m1, v.y)), mulv(m2, v.z)); }

// The kernel, one test, host side.  `revert` selects the pre-24a7397 spelling of
// transform0in1's position, rotateInv(q1, p0 - p1); everything else is shared.
static void refTriNp(const TriCase& c, const MeshH& m, bool revert, TriExpect& e)
{
    std::memset(&e, 0, sizeof(e));

    // ---- transform0in1 = transform1.transformInv(transform0) (:980) ----------
    const F4 conj{ -c.q1.x, -c.q1.y, -c.q1.z, c.q1.w };
    const F4 aToBq = quatMul(conj, c.q0);
    const gjkref::V3 v = gjkref::vsub(toG(c.p0), toG(c.p1));
    const V3 aToBp = fromG(revert ? gjkref::qRotInv(toG(c.q1), v) : gjkref::qRot(toG(conj), v));
    e.aToBq = aToBq; e.aToBp = aToBp;

    const float contactDist = c.cd0 + c.cd1;

    // ---- makeConvexShape, eCONVEXCORE arm (:120-127): words 0..5 = scale, rot.xyz
    U4 core[3] = { { f2u(c.coreData.x), f2u(c.coreData.y), f2u(c.coreData.z), 0u },
                   { 0u, 0u, 0u, 0u }, { 0u, 0u, 0u, 0u } };
    gjkref::ShapeRefC sA{ c.coreType, reinterpret_cast<const gjkref::U4*>(core), toG(aToBq), toG(aToBp), nullptr };

    // ---- convex.computeBounds() (GuConvexSupport.h:555-561) + fattenFast ------
    auto support = [&](gjkref::V3 dir) {   // ConvexShape::support (:539-540), WITH the margin
        return gjkref::vadd(gjkref::poseTf(sA, gjkref::supportLocal(sA, gjkref::qRotInv(sA.poseQ, dir))),
                            gjkref::vscl(dir, c.margin));
    };
    const gjkref::V3 X{ 1, 0, 0 }, Y{ 0, 1, 0 }, Z{ 0, 0, 1 };
    V3 bmin{ support(gjkref::vneg(X)).x, support(gjkref::vneg(Y)).y, support(gjkref::vneg(Z)).z };
    V3 bmax{ support(X).x, support(Y).y, support(Z).z };
    bmin = subv(bmin, V3{ contactDist, contactDist, contactDist });
    bmax = addv(bmax, V3{ contactDist, contactDist, contactDist });

    // ---- TrimeshBvh::scaleBounds (PxBounds3::transformFast over the inverse scale)
    V3 im0, im1, im2;
    meshScaleMat(V3{ 1.0f / c.meshScale.x, 1.0f / c.meshScale.y, 1.0f / c.meshScale.z }, c.meshRot, im0, im1, im2);
    const V3 center = mulv(addv(bmin, bmax), 0.5f);
    const V3 extents = mulv(subv(bmax, bmin), 0.5f);
    const V3 mc = matT(im0, im1, im2, center);
    const V3 e0 = mulv(im0, extents.x), e1 = mulv(im1, extents.y), e2 = mulv(im2, extents.z);
    const V3 w{ std::fabs(e0.x) + std::fabs(e1.x) + std::fabs(e2.x),
                std::fabs(e0.y) + std::fabs(e1.y) + std::fabs(e2.y),
                std::fabs(e0.z) + std::fabs(e1.z) + std::fabs(e2.z) };
    const V3 vMin = subv(mc, w), vMax = addv(mc, w);
    const V3 dims = subv(vMax, vMin);
    // margin(_bounds.getDimensions().minElement() * 0.0001f) (:570-573); dims > 0 here
    const float triMargin = std::fmin(dims.x, std::fmin(dims.y, dims.z)) * 0.0001f;

    // ---- bv32TreeTraversal over one root with one leaf child ----------------
    e.intersect = !(m.nodeMin.x > vMax.x || vMin.x > m.nodeMax.x ||
                    m.nodeMin.y > vMax.y || vMin.y > m.nodeMax.y ||
                    m.nodeMin.z > vMax.z || vMin.z > m.nodeMax.z);
    if (!e.intersect)
        return;   // no leaf visited: addPoints is never called, Contact32 stays empty

    V3 m0, m1, m2;
    meshScaleMat(c.meshScale, c.meshRot, m0, m1, m2);

    trinpacc::WIn wi;
    std::memset(&wi, 0, sizeof(wi));
    wi.meta = trinpacc::U4{ 1, 0, 0, 0 };      // one leaf -> one addPoints call (:1009)
    wi.transformQ = trinpacc::F4{ c.q1.x, c.q1.y, c.q1.z, c.q1.w };
    wi.transformP = trinpacc::F4{ c.p1.x, c.p1.y, c.p1.z, 0 };

    const uint32_t nbPrims = uint32_t(m.tris.size());
    for (uint32_t lane = 0; lane < nbPrims && lane < 4; ++lane)
    {
        const U4 t = m.tris[lane];
        const V3 rv[3] = { V3{ m.verts[t.x].x, m.verts[t.x].y, m.verts[t.x].z },
                           V3{ m.verts[t.y].x, m.verts[t.y].y, m.verts[t.y].z },
                           V3{ m.verts[t.z].x, m.verts[t.z].y, m.verts[t.z].z } };
        V3 tv[3];
        for (int k = 0; k < 3; ++k) tv[k] = matT(m0, m1, m2, rv[k]);
        // getTriNormal (:648-656): (v1 - v0).cross(v2 - v0).getNormalized()
        const V3 triN = fromG(gjkref::nrmz(gjkref::vcross(gjkref::vsub(toG(tv[1]), toG(tv[0])),
                                                          gjkref::vsub(toG(tv[2]), toG(tv[0])))));

        CvxIn s;
        std::memset(&s, 0, sizeof(s));
        s.metaA = U4{ c.coreType, 0, 0, 0 };
        std::memcpy(s.coreA, core, sizeof(core));
        s.poseAq = aToBq;
        s.poseAp = F4{ aToBp.x, aToBp.y, aToBp.z, c.margin };
        // createConvex (:628-646): a PointsCore of the three scaled vertices,
        // S = 1, R = identity, identity pose, the callback's margin
        s.metaB = U4{ 6, 0, 0, 0 };
        s.coreB[0] = U4{ 0x30u, 0u, f2u(1.0f), f2u(1.0f) };
        s.coreB[1] = U4{ f2u(1.0f), f2u(0.0f), f2u(0.0f), f2u(0.0f) };
        s.coreB[2] = U4{ f2u(1.0f), 3u | (16u << 8), 0u, 0u };
        s.poseBq = F4{ 0, 0, 0, 1 };
        s.poseBp = F4{ 0, 0, 0, triMargin };
        s.cfg = F4{ contactDist, triN.x, triN.y, triN.z };   // cullDir = the triangle normal
        for (int k = 0; k < 3; ++k) s.ptsB[k] = F4{ tv[k].x, tv[k].y, tv[k].z, 1.0f };
        CvxOut& g = e.gen[lane];
        refCvxGen(s, g);

        (&wi.nums[0][lane / 4].x)[lane & 3] = g.counts.x;
        wi.nrm[0][lane] = trinpacc::F4{ g.normal.x, g.normal.y, g.normal.z, 0 };
        for (uint32_t j = 0; j < 4; ++j)
            wi.pts[0][lane * 4 + j] = trinpacc::F4{ g.points[j].x, g.points[j].y, g.points[j].z, g.points[j].w };
    }

    trinpacc::WOut wo;
    trinpacc::refWarpAcc(wi, wo);
    e.numPatches = wo.counts.x;
    e.numPoints = wo.counts.y;
    uint32_t slot = 0;
    for (uint32_t i = 0; i < e.numPatches; ++i)
    {
        uint32_t cnt; std::memcpy(&cnt, &wo.patchN[i].w, 4);
        e.count[i] = cnt;
        e.normal[i] = V3{ wo.patchN[i].x, wo.patchN[i].y, wo.patchN[i].z };
        for (uint32_t j = 0; j < cnt; ++j, ++slot)
            e.contacts[slot] = F4{ wo.pool[i * 4 + j].x, wo.pool[i * 4 + j].y, wo.pool[i * 4 + j].z, wo.pool[i * 4 + j].w };
    }
}

// Words that differ between two expected images (contact stream + patch normals);
// ~0u when the SHAPE of the output differs (counts).
static uint32_t diffWords(const TriExpect& a, const TriExpect& b, uint32_t& word, uint32_t& aBits, uint32_t& bBits)
{
    if (a.numPatches != b.numPatches || a.numPoints != b.numPoints) return ~0u;
    uint32_t d = 0;
    for (uint32_t i = 0; i < a.numPatches; ++i)
    {
        if (a.count[i] != b.count[i]) return ~0u;
        const float* na = &a.normal[i].x; const float* nb = &b.normal[i].x;
        for (uint32_t k = 0; k < 3; ++k)
            if (f2u(na[k]) != f2u(nb[k])) { ++d; word = 1000 + i * 3 + k; aBits = f2u(na[k]); bBits = f2u(nb[k]); }
    }
    for (uint32_t s = 0; s < a.numPoints; ++s)
    {
        const float* pa = &a.contacts[s].x; const float* pb = &b.contacts[s].x;
        for (uint32_t k = 0; k < 4; ++k)
            if (f2u(pa[k]) != f2u(pb[k])) { ++d; word = s * 4 + k; aBits = f2u(pa[k]); bBits = f2u(pb[k]); }
    }
    return d;
}

static F4 axisAngle(float angle, V3 axis)   // PxQuat(angle, unitAxis)
{
    const float a = angle * 0.5f;
    const float s = std::sin(a);
    return F4{ axis.x * s, axis.y * s, axis.z * s, std::cos(a) };
}

static void buildMeshes(std::vector<MeshH>& meshes)
{
    meshes.resize(2);
    // Mesh 0: ONE triangle in the plane y = -0.05, normal +y (the design case).
    MeshH& a = meshes[0];
    a.verts = { F4{ -10.0f, -0.05f, -10.0f, 1.0f }, F4{ 0.0f, -0.05f, 10.0f, 1.0f }, F4{ 10.0f, -0.05f, -10.0f, 1.0f } };
    a.tris = { U4{ 0, 1, 2, 0 } };
    a.nodeMin = F4{ -10.0f, -0.05f, -10.0f, 0 };
    a.nodeMax = F4{ 10.0f, -0.05f, 10.0f, 0 };
    // Mesh 1: a two-triangle quad at y = 0, both normals +y, one leaf of two.
    MeshH& b = meshes[1];
    b.verts = { F4{ -4.0f, 0.0f, -4.0f, 1.0f }, F4{ -4.0f, 0.0f, 4.0f, 1.0f },
                F4{ 4.0f, 0.0f, 4.0f, 1.0f }, F4{ 4.0f, 0.0f, -4.0f, 1.0f } };
    b.tris = { U4{ 0, 1, 2, 0 }, U4{ 0, 2, 3, 0 } };
    b.nodeMin = F4{ -4.0f, 0.0f, -4.0f, 0 };
    b.nodeMax = F4{ 4.0f, 0.0f, 4.0f, 0 };
}

static void buildCases(std::vector<TriCase>& cases)
{
    const float h = 0.70710677f;   // sin/cos(PxHalfPi / 2)
    // q1 = PxQuat(-PxHalfPi, PxVec3(-0.0f, 0.0f, 1.0f)) = (+0, -0, -h, h): the mesh is
    // yawed -90 deg about z; the zero components carry the signs the axis gives them.
    const F4 q1{ 0.0f, -0.0f, -h, h };
    const V3 p1{ 0.0f, 0.0f, -0.0f };   // a mesh at the origin whose z is -0

    // ---- the design case: 5 deg tilt, length 3, margin 0.5, v = (0.05, -0.05, +0)
    // q0 = -PxQuat(PxHalfPi - 5 deg, Z): the same rotation as the positive quaternion,
    // written with w < 0 (both are unit and valid), and x/y = -0.
    {
        TriCase c{};
        c.coreType = CORE_SEGMENT; c.coreData = V3{ 3.0f, 0.0f, 0.0f }; c.margin = 0.5f;
        c.q0 = F4{ -0.0f, -0.0f, -u2fT(0x3f2cf37bu), -u2fT(0x3f3cbe35u) };   // -sin/-cos(0.741764963)
        c.p0 = V3{ 0.05f, -0.05f, 0.0f };
        c.q1 = q1; c.p1 = p1;
        c.meshScale = V3{ 1, 1, 1 }; c.meshRot = F4{ 0, 0, 0, 1 };
        c.mesh = 0; c.cd0 = 0.01f; c.cd1 = 0.01f; c.swapped = false; c.kind = 0;
        cases.push_back(c);
    }
    // ---- the directed family: the same construction over tilt, pierce depth, the
    // core's mesh-frame height and the sign of its mesh-frame x offset.  The
    // segment's lower end pierces the triangle plane (y = -0.05) by `depth`.
    const float PxHalfPi = 1.57079632679489661923f;
    const float degf = 3.14159265358979323846f / 180.0f;
    for (float tilt : { 3.0f, 5.0f, 7.0f })
        for (float depth : { 0.01f, 0.03f })
            for (float vy : { -0.05f, 0.05f })
                for (uint32_t sw = 0; sw < 2; ++sw)
                {
                    const float a = 0.5f * (PxHalfPi - tilt * degf);
                    TriCase c{};
                    c.coreType = CORE_SEGMENT; c.coreData = V3{ 3.0f, 0.0f, 0.0f }; c.margin = 0.5f;
                    c.q0 = F4{ -0.0f, -0.0f, -std::sin(a), -std::cos(a) };
                    const float y0 = -0.05f + 1.5f * std::sin(tilt * degf) - depth;
                    c.p0 = V3{ y0, vy, 0.0f };
                    c.q1 = q1; c.p1 = p1;
                    c.meshScale = V3{ 1, 1, 1 }; c.meshRot = F4{ 0, 0, 0, 1 };
                    c.mesh = 0; c.cd0 = 0.01f; c.cd1 = 0.01f; c.swapped = sw != 0; c.kind = 1;
                    cases.push_back(c);
                }

    // ---- the generic set: every wiring step, compared bitwise ---------------
    std::mt19937 rng(986301);
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto randQuat = [&]() {
        float q[4]; float m2 = 0;
        for (int k = 0; k < 4; ++k) { q[k] = rf(-1, 1); m2 += q[k] * q[k]; }
        const float i = 1.f / std::sqrt(m2 > 0 ? m2 : 1.f);
        return F4{ q[0] * i, q[1] * i, q[2] * i, q[3] * i };
    };
    for (uint32_t i = 0; i < 48; ++i)
    {
        TriCase c{};
        c.coreType = i % 4;   // point, segment, box, ellipsoid
        switch (c.coreType)
        {
        case CORE_POINT:     c.coreData = V3{ 0, 0, 0 }; c.margin = rf(0.2f, 0.6f); break;
        case CORE_SEGMENT:   c.coreData = V3{ rf(0.5f, 2.5f), 0, 0 }; c.margin = rf(0.1f, 0.5f); break;
        case CORE_BOX:       c.coreData = V3{ rf(0.3f, 1.5f), rf(0.3f, 1.5f), rf(0.3f, 1.5f) };
                             c.margin = (i % 8 < 4) ? 0.0f : rf(0.02f, 0.2f); break;
        default:             c.coreData = V3{ rf(0.2f, 0.8f), rf(0.2f, 0.8f), rf(0.2f, 0.8f) }; c.margin = rf(0.0f, 0.1f); break;
        }
        // the mesh pose: identity, a quarter turn, or arbitrary
        const uint32_t mk = (i / 4) % 3;
        c.q1 = mk == 0 ? F4{ 0, 0, 0, 1 } : (mk == 1 ? axisAngle(PxHalfPi, V3{ 1, 0, 0 }) : randQuat());
        c.p1 = V3{ rf(-3, 3), rf(-3, 3), rf(-3, 3) };
        // mesh scale: identity half the time, else non-uniform (positive)
        if ((i / 12) % 2 == 0) { c.meshScale = V3{ 1, 1, 1 }; c.meshRot = F4{ 0, 0, 0, 1 }; }
        else { c.meshScale = V3{ rf(0.6f, 1.8f), rf(0.6f, 1.8f), rf(0.6f, 1.8f) };
               c.meshRot = (i % 3 == 0) ? randQuat() : F4{ 0, 0, 0, 1 }; }
        // the core, placed in the MESH frame near the quad and moved to world
        const F4 lq = (i % 5 == 0) ? F4{ 0, 0, 0, 1 } : randQuat();
        const V3 lp{ rf(-2.5f, 2.5f), rf(-0.2f, 0.7f), rf(-2.5f, 2.5f) };
        c.q0 = quatMul(c.q1, lq);
        c.p0 = addv(fromG(gjkref::qRot(toG(c.q1), toG(lp))), c.p1);
        c.mesh = 1; c.cd0 = rf(0.0f, 0.05f); c.cd1 = rf(0.0f, 0.05f);
        c.swapped = (i % 7) == 3; c.kind = 2;
        cases.push_back(c);
    }
}

} // namespace trinp

// Build one hand-cooked mesh blob: header (nbVerts, nbTris, 0, nbPackedNodes = 1),
// ONE BV32DataPacked whose child 0 is a leaf of every triangle, the float4 vertices,
// the uint4 indices (triangleMesh.cuh:68-92, the layout the kernel reads at :391-398).
static std::vector<uint8_t> triNpMeshBlob(const trinp::MeshH& m)
{
    const uint32_t BV32_STRIDE = uint32_t(sizeof(physx::Gu::BV32DataPacked));
    std::vector<uint8_t> blob(16 + BV32_STRIDE + m.verts.size() * 16 + m.tris.size() * 16, 0);
    const uint32_t hdr[4] = { uint32_t(m.verts.size()), uint32_t(m.tris.size()), 0u, 1u };
    std::memcpy(&blob[0], hdr, 16);
    uint8_t* node = &blob[16];
    std::memcpy(node + 0, &m.nodeMin, 16);                        // mMin[0]
    std::memcpy(node + 512, &m.nodeMax, 16);                      // mMax[0]
    const uint32_t leaf = (0u << 7) | (uint32_t(m.tris.size()) << 1) | 1u;   // start 0, n prims, leaf
    std::memcpy(node + 1024, &leaf, 4);                           // mData[0]
    const uint32_t nbNodes = 1u;
    std::memcpy(node + 1152, &nbNodes, 4);                        // mNbNodes
    std::memcpy(&blob[16 + BV32_STRIDE], m.verts.data(), m.verts.size() * 16);
    std::memcpy(&blob[16 + BV32_STRIDE + m.verts.size() * 16], m.tris.data(), m.tris.size() * 16);
    return blob;
}

PXG_TEST(npCvxCoreCvxGen_triNpAtoBSign)
{
    using namespace trinp;
    static_assert(sizeof(physx::Gu::BV32DataPacked) == 1168, "BV32DataPacked (pxgSlangBv32's BV32_STRIDE)");
    static_assert(uint32_t(physx::PxGeometryType::eCONVEXCORE) == 4u &&
                  uint32_t(physx::PxGeometryType::eTRIANGLEMESH) == 8u, "PxGeometryType numbering");
    static_assert(uint32_t(physx::PxConvexCore::ePOINT) == CORE_POINT &&
                  uint32_t(physx::PxConvexCore::eSEGMENT) == CORE_SEGMENT &&
                  uint32_t(physx::PxConvexCore::eBOX) == CORE_BOX &&
                  uint32_t(physx::PxConvexCore::eELLIPSOID) == CORE_ELLIPSOID, "PxConvexCore::Type numbering");

    std::vector<MeshH> meshes; buildMeshes(meshes);
    std::vector<TriCase> cases; buildCases(cases);
    const uint32_t N = uint32_t(cases.size());

    // ---- the host mirror, CUDA spelling and the pre-24a7397 twin ------------
    std::vector<TriExpect> ref(N);
    size_t atobZeroSign = 0, epaAxisNegZero = 0, carrierStored = 0, twinOther = 0;
    bool designKilled = false;
    size_t gContacts = 0, gNone = 0, gMultiPatch = 0, gTwoLanes = 0, gSwapped = 0, gScaled = 0, gMiss = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const TriCase& c = cases[i];
        TriExpect twin;
        refTriNp(c, meshes[c.mesh], false, ref[i]);
        refTriNp(c, meshes[c.mesh], true, twin);
        const TriExpect& e = ref[i];
        uint32_t word = 0, bitsCuda = 0, bitsTwin = 0;
        const uint32_t d = diffWords(e, twin, word, bitsCuda, bitsTwin);
        if (c.kind != 2)
        {
            // transform0in1 as constructed: q = (+0, -0, qz < 0, w < 0), and the two
            // spellings of p differing in exactly the sign of the zero z
            const bool q = f2u(e.aToBq.x) == 0x00000000u && f2u(e.aToBq.y) == 0x80000000u &&
                           e.aToBq.z < 0.0f && e.aToBq.w < 0.0f;
            const bool p = f2u(e.aToBp.x) == f2u(twin.aToBp.x) && f2u(e.aToBp.y) == f2u(twin.aToBp.y) &&
                           f2u(e.aToBp.z) == 0x00000000u && f2u(twin.aToBp.z) == 0x80000000u;
            if (q && p) ++atobZeroSign;
            // EPA ended on the cap facet (axis z == -0) with the segment's 2-point face
            if (e.gen[0].counts.x >= 2 && f2u(e.gen[0].normal.z) == 0x80000000u) ++epaAxisNegZero;
            // THE KILL: the two images differ in exactly one stored word, the carrier's z
            if (d == 1 && word < 1000u && (word % 4u) == 2u && bitsCuda == 0x80000000u && bitsTwin == 0x00000000u)
            {
                ++carrierStored;
                if (c.kind == 0) designKilled = true;
            }
            else if (d != 0)
                ++twinOther;
        }
        else
        {
            PXG_CHECK_MSG(d == 0, "generic case %u: the twin differs (%u words) -- the generic set "
                          "was built with no zero-sign structure", i, d);
            if (!e.intersect) ++gMiss;
            if (e.numPoints) ++gContacts; else ++gNone;
            if (e.numPatches >= 2) ++gMultiPatch;
            if (e.gen[0].counts.x && e.gen[1].counts.x) ++gTwoLanes;
            if (e.numPoints && c.swapped) ++gSwapped;
            if (e.numPoints && (c.meshScale.x != 1.0f || c.meshScale.y != 1.0f || c.meshScale.z != 1.0f)) ++gScaled;
        }
    }
    std::printf("[ info ] triNp cov: atobZeroSign %zu epaAxisNegZero %zu carrierStored %zu (design %d) twinOther %zu | "
                "generic contacts %zu none %zu miss %zu multiPatch %zu twoLanes %zu swapped %zu scaled %zu\n",
                atobZeroSign, epaAxisNegZero, carrierStored, designKilled ? 1 : 0, twinOther,
                gContacts, gNone, gMiss, gMultiPatch, gTwoLanes, gSwapped, gScaled);
    // The coverage the kill rests on.  If a later edit moves the geometry off the
    // construction, THESE fail -- the input cannot silently stop exercising the fix.
    PXG_CHECK_MSG(designKilled, "the design case no longer separates rotate(conj(q1)) from rotateInv(q1) "
                  "in a stored word");
    PXG_CHECK_MSG(carrierStored >= 16 && atobZeroSign >= carrierStored && epaAxisNegZero >= carrierStored &&
                  twinOther == 0,
                  "directed family: carrierStored %zu atobZeroSign %zu epaAxisNegZero %zu twinOther %zu",
                  carrierStored, atobZeroSign, epaAxisNegZero, twinOther);
    PXG_CHECK_MSG(gContacts >= 20 && gNone >= 8 && gMiss >= 1 && gMultiPatch >= 4 && gTwoLanes >= 4 &&
                  gSwapped >= 2 && gScaled >= 6, "generic coverage thin");

    // ---- the device run ------------------------------------------------------
    PxgSlangComputeDevice* dev = testDevice();
    std::vector<PxgBufferHandle> meshBufs;
    std::vector<uint64_t> meshAddr;
    for (const MeshH& m : meshes)
    {
        meshBufs.push_back(createBufferFrom(dev, triNpMeshBlob(m)));
        meshAddr.push_back(dev->getBufferDeviceAddress(meshBufs.back()));
    }

    const uint32_t MAT_CORE = 1u, MAT_MESH = 2u;
    std::vector<MaterialH> mats(3);
    std::memset(mats.data(), 0, mats.size() * sizeof(MaterialH));
    mats[MAT_CORE] = MaterialH{ 0.5f, 0.75f, 0.25f, 0.125f, 0u, 0u };   // average combine everywhere
    mats[MAT_MESH] = MaterialH{ 0.25f, 0.5f, 0.5f, 0.375f, 0u, 0u };

    std::vector<GpuShape> shapes(2 * N);
    std::vector<CachedXf> xf(2 * N);
    std::vector<float> cdist(2 * N);
    std::vector<CmInput> cms(N);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(GpuShape));
    std::memset(xf.data(), 0, xf.size() * sizeof(CachedXf));
    for (uint32_t i = 0; i < N; ++i)
    {
        const TriCase& c = cases[i];
        GpuShape& s0 = shapes[2 * i];
        s0.sx = c.coreData.x; s0.sy = c.coreData.y; s0.sz = c.coreData.z;
        s0.rx = 0.0f; s0.ry = 0.0f; s0.rz = 0.0f; s0.rw = c.margin;
        s0.materialIndex = MAT_CORE;
        s0.hullLo = c.coreType; s0.hullHi = 0u;
        s0.type = uint32_t(physx::PxGeometryType::eCONVEXCORE);
        GpuShape& s1 = shapes[2 * i + 1];
        s1.sx = c.meshScale.x; s1.sy = c.meshScale.y; s1.sz = c.meshScale.z;
        s1.rx = c.meshRot.x; s1.ry = c.meshRot.y; s1.rz = c.meshRot.z; s1.rw = c.meshRot.w;
        s1.materialIndex = MAT_MESH;
        s1.hullLo = uint32_t(meshAddr[c.mesh] & 0xffffffffu); s1.hullHi = uint32_t(meshAddr[c.mesh] >> 32);
        s1.type = uint32_t(physx::PxGeometryType::eTRIANGLEMESH);
        xf[2 * i] = CachedXf{ c.q0.x, c.q0.y, c.q0.z, c.q0.w, c.p0.x, c.p0.y, c.p0.z, 0u };
        xf[2 * i + 1] = CachedXf{ c.q1.x, c.q1.y, c.q1.z, c.q1.w, c.p1.x, c.p1.y, c.p1.z, 0u };
        cdist[2 * i] = c.cd0; cdist[2 * i + 1] = c.cd1;
        // checkTypes (:98-108) swaps a mesh-first pair back
        cms[i] = c.swapped ? CmInput{ 2 * i + 1, 2 * i, 2 * i + 1, 2 * i } : CmInput{ 2 * i, 2 * i + 1, 2 * i, 2 * i + 1 };
    }

    const uint32_t PATCH_WORDS = 16u * MAX_PATCHES, CONTACT_WORDS = 4u * MAX_PATCHES * MAX_PATCH_POINTS;
    const uint64_t START_PATCHES = 0x0000100000000000ull, START_POINTS = 0x0000200000000040ull,
                   START_FORCES = 0x0000300000000080ull;   // never dereferenced by the tail
    std::vector<uint32_t> patchInit(size_t(N) * PATCH_WORDS, 0u), contactInit(size_t(N) * CONTACT_WORDS, 0xCDCDCDCDu);
    std::vector<uint32_t> outInit(size_t(N) * 12u, 0u), flagInit(N, 0xFFFFFFFFu), ctrInit(4, 0u);

    PxgBufferHandle bShapes = createBufferFrom(dev, shapes);
    PxgBufferHandle bXf = createBufferFrom(dev, xf);
    PxgBufferHandle bCd = createBufferFrom(dev, cdist);
    PxgBufferHandle bCms = createBufferFrom(dev, cms);
    PxgBufferHandle bMat = createBufferFrom(dev, mats);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);
    PxgBufferHandle bContact = createBufferFrom(dev, contactInit);
    PxgBufferHandle bPatch = createBufferFrom(dev, patchInit);
    PxgBufferHandle bCtr = createBufferFrom(dev, ctrInit);
    PxgBufferHandle bTouch = createBufferFrom(dev, flagInit);
    PxgBufferHandle bPchg = createBufferFrom(dev, flagInit);

    TriPush push{};
    push.numTests = N;
    push.cmInputs = dev->getBufferDeviceAddress(bCms);
    push.cmOutputs = dev->getBufferDeviceAddress(bOut);
    push.shapes = dev->getBufferDeviceAddress(bShapes);
    push.transformCache = dev->getBufferDeviceAddress(bXf);
    push.contactDistance = dev->getBufferDeviceAddress(bCd);
    push.materials = dev->getBufferDeviceAddress(bMat);
    push.contactStream = dev->getBufferDeviceAddress(bContact);
    push.patchStream = dev->getBufferDeviceAddress(bPatch);
    push.counters = dev->getBufferDeviceAddress(bCtr);
    push.touchFlags = dev->getBufferDeviceAddress(bTouch);
    push.patchFlags = dev->getBufferDeviceAddress(bPchg);
    push.startPatches = START_PATCHES;
    push.startPoints = START_POINTS;
    push.startForces = START_FORCES;
    push.patchBytesLimit = N * PATCH_WORDS * 4u;
    push.contactBytesLimit = N * CONTACT_WORDS * 4u;
    push.forceBytesLimit = N * 256u;

    // grid = ceil(numTests / 4): one group walks the block's four warps (tests) serially
    Kernel k("cvxCoreTriNpBda.spv", "convexCoreTrimeshNphase_Kernel32", {}, sizeof(TriPush));
    k.dispatch((N + 3u) / 4u, {}, push, 1u, 1u);
    dev->flush();

    const auto gOut = downloadBuffer<uint32_t>(dev, bOut, outInit.size());
    const auto gContact = downloadBuffer<uint32_t>(dev, bContact, contactInit.size());
    const auto gPatch = downloadBuffer<uint32_t>(dev, bPatch, patchInit.size());
    const auto gCtr = downloadBuffer<uint32_t>(dev, bCtr, 4);
    const auto gTouch = downloadBuffer<uint32_t>(dev, bTouch, N);
    const auto gPchg = downloadBuffer<uint32_t>(dev, bPchg, N);

    // ---- expected tail values that do not depend on the contacts -------------
    // combine (PxsMaterialCombiner, average mode on both sides), transcribed
    const MaterialH& m0 = mats[MAT_CORE]; const MaterialH& m1 = mats[MAT_MESH];
    const float eRest = 1.0f * (0.5f * (m0.rest + m1.rest));
    const float eDamp = 0.5f * (m0.damping + m1.damping);
    const float eDyn = 0.5f * (m0.dyn + m1.dyn);
    const float eStaF = 0.5f * (m0.sta + m1.sta);
    const float eSta = (eStaF - eDyn) >= 0.0f ? eStaF : eDyn;

    uint64_t sumPoints = 0, sumPatches = 0;
    size_t mm = 0, carrierChecked = 0;
    auto bad = [&](uint32_t i, const char* what, uint32_t got, uint32_t want) {
        if (mm < 12)
            std::printf("  triNp case %u (kind %u): %s got 0x%08x want 0x%08x\n", i, cases[i].kind, what, got, want);
        ++mm;
    };
    for (uint32_t i = 0; i < N; ++i)
    {
        const TriExpect& e = ref[i];
        sumPoints += e.numPoints; sumPatches += e.numPatches;
        const uint32_t* o = &gOut[size_t(i) * 12u];
        const uint32_t status = e.numPoints ? 2u : 1u;    // eHAS_TOUCH / eHAS_NO_TOUCH, prior 0
        if (o[8] != ((status << 16) | (e.numPatches << 8))) bad(i, "allflags", o[8], (status << 16) | (e.numPatches << 8));
        if ((o[9] & 0xFFFFu) != e.numPoints) bad(i, "nbContacts", o[9] & 0xFFFFu, e.numPoints);
        if (gTouch[i] != 1u) bad(i, "touchChange", gTouch[i], 1u);   // prior touch unknown
        if (gPchg[i] != (e.numPatches ? 1u : 0u)) bad(i, "patchChange", gPchg[i], e.numPatches ? 1u : 0u);

        const uint64_t cp = uint64_t(o[2]) | (uint64_t(o[3]) << 32);
        if (e.numPoints == 0)
        {
            if (cp != 0) bad(i, "contactPoints (none expected)", uint32_t(cp), 0u);
            continue;
        }
        const uint64_t cOff = cp - START_POINTS;
        if ((cOff & 15u) != 0 || cOff + 16u * e.numPoints > uint64_t(contactInit.size()) * 4u)
        { bad(i, "contactPoints offset", uint32_t(cOff), 0u); continue; }
        for (uint32_t s = 0; s < e.numPoints; ++s)
            for (uint32_t kk = 0; kk < 4; ++kk)
            {
                const uint32_t got = gContact[size_t(cOff / 4u) + s * 4u + kk];
                const uint32_t want = f2u((&e.contacts[s].x)[kk]);
                if (got != want) bad(i, "contact word", got, want);
            }
        const uint64_t pp = uint64_t(o[0]) | (uint64_t(o[1]) << 32);
        const uint64_t pOff = pp - START_PATCHES;
        if ((pOff & 63u) != 0 || pOff + 64u * e.numPatches > uint64_t(patchInit.size()) * 4u)
        { bad(i, "contactPatches offset", uint32_t(pOff), 0u); continue; }
        uint32_t start = 0;
        for (uint32_t pi = 0; pi < e.numPatches; ++pi)
        {
            const uint32_t* w = &gPatch[size_t(pOff / 4u) + pi * 16u];
            const uint32_t want[14] = { f2u(1.0f), f2u(1.0f), f2u(1.0f), f2u(1.0f),
                                        f2u(e.normal[pi].x), f2u(e.normal[pi].y), f2u(e.normal[pi].z),
                                        f2u(eRest), f2u(eDyn), f2u(eSta), f2u(eDamp),
                                        (start & 0xFFFFu) | ((e.count[pi] & 0xFFu) << 16),   // matFlags 0
                                        MAT_CORE << 16, MAT_MESH };                           // shape0 = the core
            for (uint32_t kk = 0; kk < 14; ++kk)
                if (w[kk] != want[kk]) bad(i, "patch word", w[kk], want[kk]);
            start += e.count[pi];
        }
    }
    // THE DIRECTED CHECK, spelled out: on every family case the mirror marks, the
    // device's carrier z is the CUDA's -0 (the revert stores +0).
    for (uint32_t i = 0; i < N; ++i)
    {
        if (cases[i].kind == 2) continue;
        TriExpect twin;
        refTriNp(cases[i], meshes[cases[i].mesh], true, twin);
        uint32_t word = 0, bc = 0, bt = 0;
        if (diffWords(ref[i], twin, word, bc, bt) != 1u || word >= 1000u) continue;
        const uint32_t* o = &gOut[size_t(i) * 12u];
        const uint64_t cOff = (uint64_t(o[2]) | (uint64_t(o[3]) << 32)) - START_POINTS;
        if (cOff + 16u * ref[i].numPoints > uint64_t(contactInit.size()) * 4u) continue;
        const uint32_t got = gContact[size_t(cOff / 4u) + word];
        PXG_CHECK_MSG(got == 0x80000000u, "case %u: carrier contact z = 0x%08x, CUDA (qinv.rotate) gives "
                      "0x80000000, the pre-24a7397 rotateInv 0x00000000", i, got);
        ++carrierChecked;
    }
    PXG_CHECK_MSG(carrierChecked == carrierStored, "carrier checks %zu of %zu", carrierChecked, carrierStored);
    PXG_CHECK_MSG(gCtr[0] == 64u * sumPatches && gCtr[1] == 16u * sumPoints && gCtr[2] == 4u * sumPoints && gCtr[3] == 0u,
                  "counters: patches %u contacts %u forces %u overflow %u (want %llu %llu %llu 0)",
                  gCtr[0], gCtr[1], gCtr[2], gCtr[3], (unsigned long long)(64u * sumPatches),
                  (unsigned long long)(16u * sumPoints), (unsigned long long)(4u * sumPoints));
    PXG_CHECK_MSG(mm == 0, "cvxCoreTriNpBda: %zu mismatching words against the CUDA-semantics mirror", mm);

    for (PxgBufferHandle hb : meshBufs) dev->destroyBuffer(hb);
    for (PxgBufferHandle hb : { bShapes, bXf, bCd, bCms, bMat, bOut, bContact, bPatch, bCtr, bTouch, bPchg })
        dev->destroyBuffer(hb);
}

} // namespace
