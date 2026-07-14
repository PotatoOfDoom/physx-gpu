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

// Verification of npCvxCorePtsFace.slang: Gu::reducePolygon and the ePOINTS
// contactFace with the ConvexShape wrapper.  Exact CPU transcription; every
// output compares BITWISE (the whole unit is sqrt/div-only arithmetic).

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

const uint32_t NOIDX = 0xFFFFFFFFu;

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static V3 nrmz(V3 v)
{
    const float m = v.x * v.x + v.y * v.y + v.z * v.z;
    if (m > 0) { const float i = 1.f / std::sqrt(m); return V3{ v.x * i, v.y * i, v.z * i }; }
    return V3{ 0, 0, 0 };
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
static float magSq(V3 v) { return v.x * v.x + v.y * v.y + v.z * v.z; }

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

static void refReduce(const PtsIn& s, PtsOut& o)
{
    std::memset(&o, 0, sizeof(o));
    V3 pts[8];
    for (int i = 0; i < 8; ++i) pts[i] = V3{ s.pts[i].x, s.pts[i].y, s.pts[i].z };
    const uint32_t count = std::min(s.meta.y, 8u);
    uint32_t inds[4];
    const uint32_t n = reduceRef(pts, count, V3{ s.dir.x, s.dir.y, s.dir.z }, s.meta.x != 0, inds);
    o.counts = U4{ n, 0, 0, 0 };
    o.inds = U4{ inds[0], inds[1], inds[2], inds[3] };
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

static bool eqF4(const F4& a, const F4& b)
{ return f2u(a.x) == f2u(b.x) && f2u(a.y) == f2u(b.y) && f2u(a.z) == f2u(b.z) && f2u(a.w) == f2u(b.w); }
static bool eqU4(const U4& a, const U4& b) { return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w; }

PXG_TEST(npCvxCorePtsFace_reduce)
{
    std::mt19937 rng(166001);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<PtsIn> in(N);
    std::vector<PtsOut> ref(N);
    size_t cKeep = 0, cDup = 0, cOne = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        PtsIn s{};
        const uint32_t count = 1 + rng() % 8;
        const bool keep1st = (rng() % 2) == 0;
        if (keep1st) ++cKeep;
        s.meta = U4{ keep1st ? 1u : 0u, count, 0, 0 };
        s.dir = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), 0 };

        if (i % 7 == 3)
        {
            // coplanar polygons: points in a plane through the origin so the
            // band scan sees real min/max spreads
            const V3 a = nrmz(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) });
            const V3 b = nrmz(cross3(a, V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) }));
            for (uint32_t p = 0; p < count; ++p)
            {
                const float u = rf(-2, 2), v = rf(-2, 2);
                s.pts[p] = F4{ a.x * u + b.x * v, a.y * u + b.y * v, a.z * u + b.z * v, 0 };
            }
        }
        else
            for (uint32_t p = 0; p < count; ++p)
                s.pts[p] = F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), 0 };

        if (i % 11 == 5 && count >= 2) { s.pts[1] = s.pts[0]; ++cDup; }         // duplicates
        if (i % 13 == 7) { for (uint32_t p = 1; p < count; ++p) s.pts[p] = s.pts[0]; ++cOne; } // all identical -> pi0 == pi1

        in[i] = s;
        refReduce(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<PtsOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npCvxCorePtsFace.spv", "ccpReduce",
             { (uint32_t)sizeof(PtsIn), (uint32_t)sizeof(PtsOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<PtsOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
        if (!(eqU4(ref[i].counts, g[i].counts) && eqU4(ref[i].inds, g[i].inds))) ++mm;
    PXG_CHECK_MSG(mm == 0, "ccpReduce: %zu mismatches", mm);
    PXG_CHECK_MSG(cKeep > N / 3 && cDup > N / 16 && cOne > N / 16,
                  "ccpReduce: coverage thin (keep %zu dup %zu one %zu)", cKeep, cDup, cOne);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

PXG_TEST(npCvxCorePtsFace_pointsFace)
{
    std::mt19937 rng(166002);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto randQuat = [&]() { float a[4]; float m2 = 0; for (int k = 0; k < 4; ++k) { a[k] = rf(-1, 1); m2 += a[k] * a[k]; } const float i = 1.f / std::sqrt(m2 > 0 ? m2 : 1.f); return F4{ a[0] * i, a[1] * i, a[2] * i, a[3] * i }; };

    std::vector<PtsIn> in(N);
    std::vector<PtsOut> ref(N);
    size_t cHit = 0, cPlanar = 0, cNull = 0, cTinyPool = 0, cLone = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        PtsIn s{};
        s.poseQ = randQuat();
        const float margin = (rng() % 3 == 0) ? 0.f : rf(0.01f, 0.4f);
        s.poseP = F4{ rf(-3, 3), rf(-3, 3), rf(-3, 3), margin };
        s.refp = F4{ rf(-4, 4), rf(-4, 4), rf(-4, 4), 0 };

        const uint32_t poolN = (i % 19 == 3) ? 0
                             : ((i % 17 == 5) ? 1 + uint32_t(rng() % 3) : 4 + uint32_t(rng() % 12)); // tiny pools reach the 2-point candidate and the <2 gate
        const uint32_t ptrLo = (i % 23 == 7) ? 0 : 48;
        const uint32_t ptrHi = (i % 29 == 11) ? 1 : 0;
        if ((ptrLo == 0 && ptrHi == 0) || poolN == 0) ++cNull;
        if (poolN >= 1 && poolN <= 3) ++cTinyPool;
        s.core[0].x = ptrLo; s.core[0].y = ptrHi;
        // identity-leaning frames keep the planar construction planar after
        // the S/R map on a third of the cases; the rest are fully random
        const bool ident = (rng() % 3 == 0);
        s.core[0].z = f2u(ident ? 1.f : rf(0.5f, 1.5f));
        s.core[0].w = f2u(ident ? 1.f : rf(0.5f, 1.5f));
        s.core[1].x = f2u(ident ? 1.f : rf(0.5f, 1.5f));
        const F4 R = ident ? F4{ 0, 0, 0, 1 } : randQuat();
        s.core[1].y = f2u(R.x); s.core[1].z = f2u(R.y); s.core[1].w = f2u(R.z);
        s.core[2].x = f2u(R.w);
        s.core[2].y = poolN | (16u << 8);

        // pool: planar clusters (a polygon slab the accumulate loop collects)
        // vs random clouds; the local dir aims near the slab normal so the
        // normal-candidate chain accepts
        V3 localDir{ rf(-1, 1), rf(-1, 1), rf(-1, 1) };
        if (i % 2 == 0 && poolN >= 4)
        {
            ++cPlanar;
            const V3 nrm = nrmz(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) });
            V3 a = nrmz(cross3(nrm, V3{ 0.3f, 0.7f, -0.2f }));
            const V3 b = cross3(nrm, a);
            const float off = rf(-1, 1);
            for (uint32_t p = 0; p < poolN; ++p)
            {
                const float u = rf(-1.5f, 1.5f), v = rf(-1.5f, 1.5f);
                const float w = (p % 3 == 2) ? rf(-1.f, -0.3f)
                              : ((p % 5 == 4) ? -rf(2e-4f, 8e-4f) : 0.f); // deep, near-band, in-plane
                s.pts[p] = F4{ nrm.x * off + a.x * u + b.x * v + nrm.x * w,
                               nrm.y * off + a.y * u + b.y * v + nrm.y * w,
                               nrm.z * off + a.z * u + b.z * v + nrm.z * w, 0 };
            }
            localDir = V3{ nrm.x + rf(-0.05f, 0.05f), nrm.y + rf(-0.05f, 0.05f), nrm.z + rf(-0.05f, 0.05f) };
        }
        else
            for (uint32_t p = 0; p < poolN; ++p)
                s.pts[p] = F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), 0 };

        if (i % 31 == 13 && poolN >= 4)
        {
            // tilted top triangle: the accepted n0 is the plane THROUGH the
            // top three points, so exactly those three land in the 1e-4 band
            // while the 3e-4 dir-offsets keep the plane visibly tilted (band
            // anchor geometry; the numPoints >= 2 gate itself is provably
            // dead -- every accepted candidate is orthogonal to a top edge)
            s.core[0].z = f2u(1.f); s.core[0].w = f2u(1.f); s.core[1].x = f2u(1.f);
            s.core[1].y = f2u(0.f); s.core[1].z = f2u(0.f); s.core[1].w = f2u(0.f);
            s.core[2].x = f2u(1.f);
            const V3 nrm = nrmz(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) });
            V3 a = nrmz(cross3(nrm, V3{ 0.3f, 0.7f, -0.2f }));
            const V3 b = cross3(nrm, a);
            const float off = rf(-0.5f, 0.5f);
            s.pts[0] = F4{ nrm.x * off, nrm.y * off, nrm.z * off, 0 };
            for (uint32_t p = 1; p < poolN; ++p)
            {
                const float u = rf(-1.f, 1.f), v = rf(-1.f, 1.f);
                const float w = (p <= 2) ? -rf(3e-4f, 8e-4f) : -rf(0.3f, 1.f);
                s.pts[p] = F4{ nrm.x * (off + w) + a.x * u + b.x * v,
                               nrm.y * (off + w) + a.y * u + b.y * v,
                               nrm.z * (off + w) + a.z * u + b.z * v, 0 };
            }
            localDir = nrm;
            ++cLone;
        }

        const V3 world = qRot(s.poseQ, localDir);
        s.dir = F4{ world.x, world.y, world.z, 0 };

        in[i] = s;
        refPointsFace(s, ref[i]);
        if (ref[i].counts.x >= 2) ++cHit;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<PtsOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npCvxCorePtsFace.spv", "ccpPointsFace",
             { (uint32_t)sizeof(PtsIn), (uint32_t)sizeof(PtsOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<PtsOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const PtsOut& r = ref[i]; const PtsOut& q = g[i];
        bool okc = r.counts.x == q.counts.x && eqF4(r.normal, q.normal);
        for (int j = 0; j < 4 && okc; ++j) okc = eqF4(r.fp[j], q.fp[j]);
        if (!okc) ++mm;
        if (i % 31 == 13 && ref[i].counts.x == 0 && (f2u(ref[i].normal.x) | f2u(ref[i].normal.y) | f2u(ref[i].normal.z)) != 0)
            std::printf("GATE i=%u refc=%u gpuc=%u refn=(%g %g %g)\n", i, ref[i].counts.x, g[i].counts.x,
                        ref[i].normal.x, ref[i].normal.y, ref[i].normal.z);
    }
    PXG_CHECK_MSG(mm == 0, "ccpPointsFace: %zu mismatches", mm);
    PXG_CHECK_MSG(cHit > N / 8 && cPlanar > N / 3 && cNull > 10 && cTinyPool > N / 32 && cLone > 20,
                  "ccpPointsFace: coverage thin (hit %zu planar %zu null %zu tiny %zu lone %zu)", cHit, cPlanar, cNull, cTinyPool, cLone);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
