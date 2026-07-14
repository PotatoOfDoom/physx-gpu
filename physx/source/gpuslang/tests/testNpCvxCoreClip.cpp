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

// Verification of npCvxCoreClip.slang: Gu::FaceClipper (clipNone/clip2x2/
// clipNxN with makePlanes/makePolygon/clipPoly and the PxVec4 reducePolygon
// call site) behind generateContactPatch.  Exact CPU transcription; every
// output compares BITWISE (sqrt/div-only arithmetic; the vector-by-scalar
// divisions use PxVec3::operator/'s reciprocal-multiply form).

#include "pxgTestFramework.h"

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

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

PXG_TEST(npCvxCoreClip_clip)
{
    std::mt19937 rng(167001);
    const uint32_t N = 8000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<ClipIn> in(N);
    std::vector<ClipOut> ref(N);
    size_t cNone = 0, c22 = 0, cNN = 0, c2N = 0, cPar = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        ClipIn s{};
        // a contact frame: axis near +Z rotated arbitrarily is unnecessary --
        // keep axis random-ish but the faces roughly perpendicular so the
        // plane projections do not blow up (axis.dot(faceNormal) well away
        // from zero, as GJKEPA guarantees in practice)
        V3 axis{ rf(-0.3f, 0.3f), rf(-0.3f, 0.3f), (rng() % 2) ? 1.f : -1.f };
        if (i % 9 == 4) axis = V3{ rf(-0.01f, 0.01f), (rng() % 2) ? 1.f : -1.f, rf(-0.01f, 0.01f) }; // |axis.y| > 0.9999 basis branch
        if (i % 9 == 7) axis = V3{ rf(-0.5f, 0.5f), (rng() % 2 ? 0.7f : -0.7f) + rf(-0.1f, 0.1f), rf(-0.5f, 0.5f) }; // |axis.y| in (0.5, 0.9999): pins the 0.9999 basis threshold
        normRecip(axis);
        s.axis = F4{ axis.x, axis.y, axis.z, 0 };
        s.p0 = F4{ rf(-1, 1), rf(-1, 1), rf(0.f, 0.4f), 0 };
        s.p1 = F4{ rf(-1, 1), rf(-1, 1), rf(-0.4f, 0.f), 0 };

        const uint32_t mode = uint32_t(rng() % 8);
        uint32_t n0, n1;
        if (mode == 0) { n0 = uint32_t(rng() % 2); n1 = 2 + uint32_t(rng() % 3); ++cNone; }
        else if (mode == 1) { n0 = 2 + uint32_t(rng() % 3); n1 = uint32_t(rng() % 2); ++cNone; }
        else if (mode <= 3) { n0 = 2; n1 = 2; ++c22; }
        else if (mode == 4) { n0 = 2; n1 = 4; ++c2N; }
        else if (mode == 5) { n0 = 4; n1 = 2; ++c2N; }
        else { n0 = (rng() % 2) ? 3 : 4; n1 = (rng() % 2) ? 3 : 4; ++cNN; }
        s.counts = U4{ n0, n1, 0, 0 };

        // face normals: near +axis for face0, near -axis for face1
        V3 fn0 = add(axis, V3{ rf(-0.2f, 0.2f), rf(-0.2f, 0.2f), rf(-0.2f, 0.2f) }); normRecip(fn0);
        V3 fn1 = add(neg(axis), V3{ rf(-0.2f, 0.2f), rf(-0.2f, 0.2f), rf(-0.2f, 0.2f) }); normRecip(fn1);
        s.n0 = F4{ fn0.x, fn0.y, fn0.z, 0 };
        s.n1 = F4{ fn1.x, fn1.y, fn1.z, 0 };

        // face points: planar rings perpendicular to the face normal
        auto makeFace = [&](F4* fp, uint32_t n, V3 fn, float offset) {
            V3 t0 = cross3(fn, V3{ 0.31f, 0.77f, -0.53f }); normRecip(t0);
            V3 t1 = cross3(fn, t0);
            const V3 c{ rf(-0.3f, 0.3f) + fn.x * offset, rf(-0.3f, 0.3f) + fn.y * offset, rf(-0.3f, 0.3f) + fn.z * offset };
            for (uint32_t k = 0; k < n; ++k)
            {
                const float ang = 6.2831853f * (float(k) + rf(0.f, 0.6f)) / float(n < 2 ? 2 : n);
                const float rad = rf(0.4f, 1.4f);
                const V3 p = add(add(c, mul(t0, rad * std::cos(ang))), mul(t1, rad * std::sin(ang)));
                fp[k] = F4{ p.x, p.y, p.z, 0 };
            }
        };
        makeFace(s.fp0, n0, fn0, rf(0.f, 0.3f));
        if (i % 21 == 9 && n0 >= 3)
        {
            // one face edge nearly along the axis: |cross(e, axis)| lands in
            // (1e-5, 3e-3) so |n|^2 sits between eps^2 and eps -- pins the
            // makePlanes viability threshold
            V3 perp = cross3(axis, V3{ 0.6f, -0.2f, 0.75f }); normRecip(perp);
            const float tiny = rf(2e-4f, 2e-3f);
            const V3 e = add(axis, mul(perp, tiny));
            s.fp0[1] = F4{ s.fp0[0].x + e.x, s.fp0[0].y + e.y, s.fp0[0].z + e.z, 0 };
        }
        makeFace(s.fp1, n1, fn1, rf(-0.3f, 0.f));

        if (n0 == 2 && n1 == 2 && (i % 5 < 3))
        {
            // parallel segments so clip2x2 proceeds past the gate: both along
            // a shared direction perpendicular to the axis
            V3 t0 = cross3(axis, V3{ 0.4f, -0.8f, 0.45f }); normRecip(t0);
            const V3 base0{ rf(-0.5f, 0.5f), rf(-0.5f, 0.5f), rf(0.f, 0.2f) };
            const V3 base1{ base0.x + rf(-0.2f, 0.2f), base0.y + rf(-0.2f, 0.2f), base0.z - rf(0.f, 0.4f) };
            const float l0 = rf(0.5f, 2.f), o0 = rf(-1.f, 0.5f);
            const float l1 = rf(0.5f, 2.f), o1 = rf(-1.f, 0.5f);
            s.fp0[0] = F4{ base0.x + t0.x * o0, base0.y + t0.y * o0, base0.z + t0.z * o0, 0 };
            s.fp0[1] = F4{ base0.x + t0.x * (o0 + l0), base0.y + t0.y * (o0 + l0), base0.z + t0.z * (o0 + l0), 0 };
            s.fp1[0] = F4{ base1.x + t0.x * o1, base1.y + t0.y * o1, base1.z + t0.z * o1, 0 };
            s.fp1[1] = F4{ base1.x + t0.x * (o1 + l1), base1.y + t0.y * (o1 + l1), base1.z + t0.z * (o1 + l1), 0 };
            if (i % 15 == 6) { s.fp1[0] = s.fp0[0]; s.fp1[1] = s.fp0[1]; } // exact overlap: duplicate merge
            if (i % 15 == 11)
            {
                // NEAR-parallel: twist segment 1 so the cross-axis test lands
                // in (1e-5, 1e-3) -- pins the 1e-5 parallel gate
                V3 t1 = cross3(axis, t0); normRecip(t1);
                const float tw = rf(2e-4f, 8e-4f);
                const V3 dir1 = add(t0, mul(t1, tw));
                s.fp1[1] = F4{ s.fp1[0].x + dir1.x * l1, s.fp1[0].y + dir1.y * l1, s.fp1[0].z + dir1.z * l1, 0 };
            }
            ++cPar;
        }

        in[i] = s;
        refClip(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<ClipOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npCvxCoreClip.spv", "cccClip",
             { (uint32_t)sizeof(ClipIn), (uint32_t)sizeof(ClipOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<ClipOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const ClipOut& r = ref[i]; const ClipOut& q = g[i];
        bool ok = r.counts.x == q.counts.x && eqF4(r.normal, q.normal);
        for (int j = 0; j < 4 && ok; ++j) ok = eqF4(r.pts[j], q.pts[j]);
        if (!ok) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "cccClip: %zu mismatches", mm);
    PXG_CHECK_MSG(cNone > N / 16 && c22 > N / 8 && cNN > N / 8 && c2N > N / 8 && cPar > N / 16,
                  "cccClip: coverage thin (none %zu 22 %zu NN %zu 2N %zu par %zu)", cNone, c22, cNN, c2N, cPar);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
