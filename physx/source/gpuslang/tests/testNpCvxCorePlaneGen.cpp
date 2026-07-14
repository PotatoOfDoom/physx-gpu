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

// Verification of npCvxCorePlaneGen.slang: Gu::generateContacts(PxPlane,
// ConvexShape) (GuConvexSupport.cpp L41-75) -- support (with margin) +
// plane distance gate + the ConvexShape::contactFace wrapper (the
// tranche-165/166 reference transcriptions reused verbatim below) + the
// midpoint / face-point contact emission.  Compare doctrine: numContacts,
// normal and every non-trig output BITWISE (modulo the documented +-0
// class); points/dists that passed through rotatePoints' acos/sin/cos
// (cylinder/cone cap branches, flagged deterministically) to tolerance.

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

static bool eqFZ(float a, float b) { return f2u(a) == f2u(b) || (a == 0.f && b == 0.f); }

// Gu::generateContacts(PxPlane, ConvexShape) (GuConvexSupport.cpp L41-75)
static void refPlaneGen(const PgIn& s, PgOut& o)
{
    std::memset(&o, 0, sizeof(o));

    const V3 pn{ s.plane.x, s.plane.y, s.plane.z };
    const float pd = s.plane.w;
    const V3 normal = neg(pn);
    const float margin = s.poseP.w;
    const float contactDist = s.cfg.x;

    ShapeRefC sh{ s.meta.x, s.core, s.poseQ, V3{ s.poseP.x, s.poseP.y, s.poseP.z }, s.pts };

    // ConvexShape::support = pose.transform(supportLocal(pose.rotateInv(dir))) + dir * margin
    const V3 sl = supportLocal(sh, qRotInv(s.poseQ, normal));
    const V3 point1 = add(add(qRot(s.poseQ, sl), sh.poseP), mul(normal, margin));
    const float dist = dot3(point1, pn) + pd;

    uint32_t numContacts = 0, trig = 0;

    if (dist < contactDist)
    {
        // ConvexShape::contactFace via the tranche-165/166 references
        F4 wfp[4]; V3 wnrm{ 0, 0, 0 };
        uint32_t numPoints = 0;
        if (s.meta.x == CORE_POINTS)
        {
            PtsIn pi{};
            pi.meta = s.meta;
            std::memcpy(pi.core, s.core, sizeof(pi.core));
            pi.poseQ = s.poseQ; pi.poseP = s.poseP;
            pi.dir = F4{ normal.x, normal.y, normal.z, 0 };
            pi.refp = F4{ point1.x, point1.y, point1.z, 0 };
            std::memcpy(pi.pts, s.pts, sizeof(s.pts)); // pool words 0..11
            PtsOut po;
            refPointsFace(pi, po);
            numPoints = po.counts.x;
            wnrm = V3{ po.normal.x, po.normal.y, po.normal.z };
            for (int k = 0; k < 4; ++k) wfp[k] = po.fp[k];
        }
        else
        {
            FaceIn fi{};
            fi.meta = s.meta;
            std::memcpy(fi.core, s.core, sizeof(fi.core));
            fi.poseQ = s.poseQ; fi.poseP = s.poseP;
            fi.dir = F4{ normal.x, normal.y, normal.z, 0 };
            fi.refp = F4{ point1.x, point1.y, point1.z, 0 };
            FaceOut fo;
            refFace(fi, fo);
            numPoints = fo.counts.x;
            trig = fo.counts.y;
            wnrm = V3{ fo.normal.x, fo.normal.y, fo.normal.z };
            for (int k = 0; k < 4; ++k) wfp[k] = fo.fp[k];
        }
        (void)wnrm;

        if (numPoints == 0)
        {
            const V3 point = add(point1, mul(mul(normal, dist), 0.5f));
            o.points[numContacts] = F4{ point.x, point.y, point.z, dist };
            ++numContacts;
        }

        for (uint32_t i = 0; i < numPoints; ++i)
        {
            const V3 p1{ wfp[i].x, wfp[i].y, wfp[i].z };
            const float d = dot3(p1, pn) + pd;
            const V3 pt = add(p1, mul(mul(normal, d), 0.5f));
            o.points[numContacts] = F4{ pt.x, pt.y, pt.z, d };
            ++numContacts;
        }
    }

    o.counts = U4{ numContacts, trig, 0, 0 };
    o.normal = F4{ normal.x, normal.y, normal.z, 0 };
}

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

PXG_TEST(npCvxCorePlaneGen_contacts)
{
    std::mt19937 rng(171001);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto randQuat = [&]() { float a[4]; float m2 = 0; for (int k = 0; k < 4; ++k) { a[k] = rf(-1, 1); m2 += a[k] * a[k]; } const float i = 1.f / std::sqrt(m2 > 0 ? m2 : 1.f); return F4{ a[0] * i, a[1] * i, a[2] * i, a[3] * i }; };

    std::vector<PgIn> in(N);
    std::vector<PgOut> ref(N);
    size_t cType[7] = { 0, 0, 0, 0, 0, 0, 0 };
    size_t cHit = 0, cMiss = 0, cMid = 0, cFace = 0, cTrig = 0, cDegen = 0, cAxis = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        PgIn s{};
        uint32_t type = rng() % 7;

        // plane through a random point with a random normal
        V3 pn{ rf(-1, 1), rf(-1, 1), rf(-1, 1) };
        {
            const float m = std::sqrt(dot3(pn, pn));
            pn = m > 0 ? mul(pn, 1.f / m) : V3{ 0, 1, 0 };
        }
        if (i % 7 == 3)
        {
            // AXIS-ALIGNED planes: the -plane.n contact dir hits the exact
            // cap/face branches of contactFace (the trig paths)
            pn = V3{ 0, 0, 0 };
            const int ax = rng() % 3;
            (&pn.x)[ax] = (rng() % 2) ? 1.f : -1.f;
            ++cAxis;
        }
        const V3 pp{ rf(-2, 2), rf(-2, 2), rf(-2, 2) };
        const float pd = -dot3(pp, pn); // PxPlane(point, normal)
        s.plane = F4{ pn.x, pn.y, pn.z, pd };

        s.meta = U4{ type, 0, 0, 0 };
        fillCore(rng, type, s.core, s.pts, rf);
        s.poseQ = (i % 11 == 5) ? F4{ 0, 0, 0, 1 } : randQuat();
        const float margin = (rng() % 3 == 0) ? 0.f : rf(0.01f, 0.3f);
        // place the shape near the plane point: above/straddling/below
        const float off = (i % 3 == 0) ? rf(-1.f, 0.5f) : rf(0.2f, 3.f);
        const V3 cpos = add(pp, mul(pn, off));
        s.poseP = F4{ cpos.x, cpos.y, cpos.z, margin };
        s.cfg = F4{ (i % 5 == 0) ? rf(0.f, 0.05f) : rf(0.1f, 1.5f), 0, 0, 0 };

        if (i % 9 == 4)
        {
            // cap-branch fixture: identity pose + x-aligned plane so the
            // LOCAL contact dir is exactly (+-1, 0, 0) -- the cylinder cap /
            // cone base contactFace branches run rotatePoints (the trig path)
            type = (rng() % 2) ? CORE_CYLINDER : CORE_CONE;
            s.meta = U4{ type, 0, 0, 0 };
            fillCore(rng, type, s.core, s.pts, rf);
            s.poseQ = F4{ 0, 0, 0, 1 };
            pn = V3{ (type == CORE_CONE) ? 1.f : ((rng() % 2) ? 1.f : -1.f), 0.f, 0.f };
            const float pdc = -dot3(pp, pn);
            s.plane = F4{ pn.x, pn.y, pn.z, pdc };
            const float off2 = rf(-0.5f, 1.f);
            const V3 cp2 = add(pp, mul(pn, off2));
            s.poseP = F4{ cp2.x, cp2.y, cp2.z, s.poseP.w };
            s.cfg = F4{ rf(0.3f, 1.5f), 0, 0, 0 };
        }
        if (i % 13 == 9)
        {
            // degenerate radii/extents: the < FLT_EPSILON contactFace branches
            if (type == CORE_BOX || type == CORE_ELLIPSOID)
            { s.core[0].y = f2u((rng() % 2) ? 0.f : 1e-8f); ++cDegen; }
            else if (type == CORE_CYLINDER || type == CORE_CONE)
            { s.core[0].x = f2u(0.f); ++cDegen; }
        }

        in[i] = s;
        ++cType[type];
        refPlaneGen(s, ref[i]);
        if (ref[i].counts.x == 0) ++cMiss; else ++cHit;
        if (ref[i].counts.x == 1) ++cMid;
        if (ref[i].counts.x >= 2) ++cFace;
        if (ref[i].counts.y != 0) ++cTrig;
    }

    std::printf("[ info ] pg cov: hit %zu miss %zu mid %zu face %zu trig %zu degen %zu axis %zu\n",
                cHit, cMiss, cMid, cFace, cTrig, cDegen, cAxis);

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<PgOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npCvxCorePlaneGen.spv", "cpgContacts",
             { (uint32_t)sizeof(PgIn), (uint32_t)sizeof(PgOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<PgOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const PgOut& r = ref[i]; const PgOut& q = g[i];
        bool ok = r.counts.x == q.counts.x && r.counts.y == q.counts.y &&
                  eqFZ(r.normal.x, q.normal.x) && eqFZ(r.normal.y, q.normal.y) && eqFZ(r.normal.z, q.normal.z);
        const bool trig = r.counts.y != 0;
        for (uint32_t j = 0; j < 4 && ok; ++j)
            ok = trig ? eqF4Tol(r.points[j], q.points[j])
                      : (eqFZ(r.points[j].x, q.points[j].x) && eqFZ(r.points[j].y, q.points[j].y) &&
                         eqFZ(r.points[j].z, q.points[j].z) && eqFZ(r.points[j].w, q.points[j].w));
        if (!ok && mm < 5)
            std::printf("MM %u nR %u nG %u tR %u tG %u p0R (%g %g %g %g) p0G (%g %g %g %g)\n",
                        i, r.counts.x, q.counts.x, r.counts.y, q.counts.y,
                        r.points[0].x, r.points[0].y, r.points[0].z, r.points[0].w,
                        q.points[0].x, q.points[0].y, q.points[0].z, q.points[0].w);
        if (!ok) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "cpgContacts: %zu mismatches", mm);

    bool cov = cHit > N / 4 && cMiss > N / 10 && cMid > N / 20 && cFace > N / 20 &&
               cTrig > 20 && cDegen > 20 && cAxis > N / 8;
    for (int k2 = 0; k2 < 7; ++k2) cov = cov && cType[k2] > N / 16;
    PXG_CHECK_MSG(cov, "cpgContacts: coverage thin (hit %zu mid %zu face %zu trig %zu)",
                  cHit, cMid, cFace, cTrig);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
