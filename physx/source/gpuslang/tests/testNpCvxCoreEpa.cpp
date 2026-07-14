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

// Verification of npCvxCoreEpa.slang: Gu::RefGjkEpa::computeEpaDepth
// (GuRefGjkEpa.h L421-699) over the margin-less ConvexShape::supportLocal.
// Exact CPU transcription: the EpaDepth ctor (bounds-center seed +
// PxComputeBasisVectors + the <3-vert conditional seeding + mProj reset),
// addPoint (MAX_VERTS cap, mProj/mBest update BEFORE the duplicate out,
// the two seed faces at numVerts==3, the backward face-removal scan with
// reversed-twin edge cancellation and remove-by-swap, numEdges==0 and
// MAX_FACES overflow outs, the new-face fan), makePlane's symmetrized
// triple-cross with the FLT_EPSILON degeneracy out and reciprocal-multiply
// normalization, computeClosest's strict plane.w argmax with the
// numFaces==2 mBest latch, computePoints' barycentric split accumulation,
// and the driver (isValid else FLT_MAX; distEps = 0.01*accuracy;
// proj >= closestDist - distEps -> best face outputs).  All arithmetic is
// add/mul/div/sqrt: outputs compare bitwise except the signs of exact
// zeros (the documented llvmpipe NSZ exception class).

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

PXG_TEST(npCvxCoreEpa_depth)
{
    std::mt19937 rng(170001);
    const uint32_t N = 4000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto randQuat = [&]() { float a[4]; float m2 = 0; for (int k = 0; k < 4; ++k) { a[k] = rf(-1, 1); m2 += a[k] * a[k]; } const float i = 1.f / std::sqrt(m2 > 0 ? m2 : 1.f); return F4{ a[0] * i, a[1] * i, a[2] * i, a[3] * i }; };

    std::vector<EpaTIn> in(N);
    std::vector<EpaTOut> ref(N);
    size_t cType[7] = { 0, 0, 0, 0, 0, 0, 0 };
    size_t cTag[2] = { 0, 0 };
    size_t cInvalid = 0, cFallback = 0, cBasisElse = 0, cDup = 0, cCap = 0;
    size_t cPlaneFail = 0, cEdge0 = 0, cFaceOver = 0;
    size_t cNullPts = 0, cTinyS = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        EpaTIn s{};
        uint32_t typeA = rng() % 7, typeB = rng() % 7;

        s.poseAq = randQuat();
        s.poseBq = randQuat();
        const V3 pA{ rf(-1, 1), rf(-1, 1), rf(-1, 1) };

        // EPA is the penetration path: bias to overlap, keep some graze and
        // separated cases (the driver still runs to a dup/degenerate out)
        const uint32_t cls = i % 4;
        float sep = 0;
        if (cls == 0) sep = rf(0.f, 0.3f);            // deep overlap
        else if (cls == 1) sep = rf(0.3f, 1.5f);      // partial overlap
        else if (cls == 2) sep = rf(3.6f, 4.4f);      // graze band
        else sep = rf(4.5f, 7.f);                     // separated
        V3 sepDir{ rf(-1, 1), rf(-1, 1), rf(-1, 1) };
        if (normalizeIn(sepDir) == 0) sepDir = V3{ 1, 0, 0 };
        V3 pB = vadd(pA, vscl(sepDir, sep));

        if (i % 19 == 7)
        {
            typeB = typeA;
            s.poseBq = s.poseAq;
            pB = pA; // identical: max penetration + the (1,0,0) seed fallback
        }
        if (i % 23 == 11)
            typeA = typeB = CORE_POINT; // degenerate: no polytope, isValid false

        s.metaA = U4{ typeA, 0, 0, 0 };
        s.metaB = U4{ typeB, 0, 0, 0 };
        ++cType[typeA]; ++cType[typeB];
        fillCore(rng, typeA, s.coreA, s.ptsA, rf);
        fillCore(rng, typeB, s.coreB, s.ptsB, rf);
        if (i % 19 == 7)
        { std::memcpy(s.coreB, s.coreA, sizeof(s.coreA)); std::memcpy(s.ptsB, s.ptsA, sizeof(s.ptsA)); }

        s.poseAp = F4{ pA.x, pA.y, pA.z, 0 };
        s.poseBp = F4{ pB.x, pB.y, pB.z, 0 };
        s.cfg = F4{ 0, 0, 0, 0 };

        if (i % 61 == 13)
        {
            // ePOINTS 64-bit NULL-pointer image branch (review-169 gap)
            s.metaA = U4{ CORE_POINTS, 0, 0, 0 };
            fillCore(rng, CORE_POINTS, s.coreA, s.ptsA, rf);
            s.coreA[0].x = 0; s.coreA[0].y = 0;
            ++cNullPts;
        }
        if (i % 67 == 29)
        {
            // |d| < FLT_EPSILON degenerate-scale guard (review-169 gap)
            s.metaB = U4{ CORE_POINTS, 0, 0, 0 };
            fillCore(rng, CORE_POINTS, s.coreB, s.ptsB, rf);
            s.coreB[0].z = f2u(1e-8f); s.coreB[0].w = f2u(1e-8f);
            s.coreB[1].x = f2u(1e-8f);
            ++cTinyS;
        }
        if (i % 17 == 9)
        {
            // center offset along +-y only: |dir0.y| > 0.9999 drives the
            // PxComputeBasisVectors else-branch
            const float d = ((rng() % 2) ? 1.f : -1.f) * rf(0.05f, 1.5f);
            s.poseBp = F4{ pA.x, pA.y + d, pA.z, 0 };
        }

        if (i % 1499 == 400)
        {
            // review-170 m16 kill (scratchpad fix743): CORE_POINT vs thin
            // rotated box -- accuracy = FLT_EPSILON makes testEps ~1.2e-9,
            // below dot rounding noise, so the pass-1 point lands above BOTH
            // seed faces: all faces removed, numEdges == 0 in a fully
            // Euler-consistent state; without the numEdges == 0 out, mBest
            // is later reassigned from the stale face slot (axis flips)
            static const uint32_t KILL16[148] = {
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x3f800000u,0x3e5659c6u,0xbf116428u,0x3e1480b5u,0x00000000u,
                0x00000002u,0x00000000u,0x00000000u,0x00000000u,0x402c1ca4u,0x3fab8067u,0x328d4c67u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x3f084176u,0xbe8ad1c9u,0x3f2ae00du,0xbee3a25eu,0x3e2fe063u,0xbe4a72ebu,0x3e00fab7u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u };
            std::memcpy(&s, KILL16, sizeof(KILL16));
        }
        if (i % 1499 == 900)
        {
            // review-170 m17 kill (fix1609): DISCONNECTED visible regions
            // (numFaces + numEdges = F + 2C - 2I > 60) fire the MAX_FACES
            // overflow out in an Euler-consistent polytope; without it the
            // fan writes past fpl[59] and the run diverges
            static const uint32_t KILL17[148] = {
                0x00000004u,0x00000000u,0x00000000u,0x00000000u,0x402ffe85u,0x3f5468c3u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x3f2c3d4fu,0xbf10f466u,0x3eb0b71du,0x3ea7ef9bu,0xbf128630u,0xbf7e7fd4u,0xbf3a52f0u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x3de22b9bu,0x3dbf3256u,0x3f70820au,0x3e9f03e3u,0xbefa4ecau,0xbf428a48u,0xbfa79372u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u };
            std::memcpy(&s, KILL17, sizeof(KILL17));
        }
        if (i % 1499 == 1300)
        {
            // review-170 m17b kill (fix36249): the sum in {59,60} with
            // numVerts < 32 (after a disconnected removal) -- pins the exact
            // overflow threshold
            static const uint32_t KILL17B[148] = {
                0x00000001u,0x00000000u,0x00000000u,0x00000000u,0x403337a7u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x3f00f46du,0x3f0ef609u,0x3f008d24u,0x3eda9243u,0x3c82a140u,0xbf1ca600u,0xbf13150eu,0x00000000u,
                0x00000003u,0x00000000u,0x00000000u,0x00000000u,0x3f5226d8u,0x3f2c855eu,0x4031bb4cu,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0xbf213ed4u,0xbf23c348u,0x3d846dccu,0x3edf1a25u,0x3cb61164u,0xbf7148deu,0xbe641a3au,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u };
            std::memcpy(&s, KILL17B, sizeof(KILL17B));
        }
        if (i % 1999 == 555)
        {
            // review-170 planeFail fixture (fixpf): 3 collinear ePOINTS vs a
            // CORE_POINT give exactly collinear CSO seed verts at
            // numVerts == 3 -- the two-seed-face makePlane FLT_MAX path,
            // distinct from the CORE_POINT isValid path
            static const uint32_t PLANEFAIL[148] = {
                0x00000006u,0x00000000u,0x00000000u,0x00000000u,0x00000030u,0x00000000u,0x3f800000u,0x3f800000u,
                0x3f800000u,0x00000000u,0x00000000u,0x00000000u,0x3f800000u,0x00001003u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x3f800000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x3f800000u,0x00000000u,0x40000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0xbf800000u,0x00000000u,0x00000000u,0x00000000u,0x3f800000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,
                0x00000000u,0x00000000u,0x00000000u,0x00000000u };
            std::memcpy(&s, PLANEFAIL, sizeof(PLANEFAIL));
        }
        in[i] = s;
        RefStats st;
        refEpa(s, ref[i], st);
        ++cTag[st.tag];
        if (st.invalid) ++cInvalid;
        if (st.fallback) ++cFallback;
        if (st.basisElse) ++cBasisElse;
        if (st.dup) ++cDup;
        if (st.cap) ++cCap;
        if (st.planeFail) ++cPlaneFail;
        if (st.edge0) ++cEdge0;
        if (st.faceOver) ++cFaceOver;
    }

    std::printf("[ info ] epa cov: tags %zu/%zu invalid %zu fb %zu basisElse %zu dup %zu cap %zu planeFail %zu edge0 %zu faceOver %zu\n",
                cTag[0], cTag[1], cInvalid, cFallback, cBasisElse, cDup, cCap, cPlaneFail, cEdge0, cFaceOver);

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 31) / 32;
    std::vector<EpaTOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npCvxCoreEpa.spv", "cepDepth",
             { (uint32_t)sizeof(EpaTIn), (uint32_t)sizeof(EpaTOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<EpaTOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const EpaTOut& r = ref[i]; const EpaTOut& q = g[i];
        const bool ok = f2u(r.result.y) == f2u(q.result.y) && eqFZ(r.result.x, q.result.x) &&
                        eqF4Z(r.pointA, q.pointA) && eqF4Z(r.pointB, q.pointB) && eqF4Z(r.axis, q.axis);
        if (!ok && mm < 5)
            std::printf("MM %u tagR %u tagG %u dR %.9g dG %.9g axR (%g %g %g) axG (%g %g %g)\n",
                        i, f2u(r.result.y), f2u(q.result.y), r.result.x, q.result.x,
                        r.axis.x, r.axis.y, r.axis.z, q.axis.x, q.axis.y, q.axis.z);
        if (!ok) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "cepDepth: %zu mismatches", mm);

    bool cov = cTag[1] > N / 2 && cInvalid > 8 && cFallback > 8 && cBasisElse > N / 32 &&
               cDup > N / 10 && cEdge0 > 8 && cNullPts > 8 && cTinyS > 8 && cPlaneFail >= 1 && cFaceOver >= 1;
    for (int k2 = 0; k2 < 7; ++k2) cov = cov && cType[k2] > N / 16;
    PXG_CHECK_MSG(cov, "cepDepth: coverage thin (t1 %zu inv %zu fb %zu bE %zu dup %zu e0 %zu)",
                  cTag[1], cInvalid, cFallback, cBasisElse, cDup, cEdge0);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
