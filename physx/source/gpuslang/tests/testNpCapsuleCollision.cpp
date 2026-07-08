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

// Verification of npCapsuleCollision.slang: planeCapsule and capsuleCapsule.
// Randomized transforms drive general (single-contact), near-parallel (two
// contacts via endpoint projection) and separated cases; the GPU
// point/penetration/normal/contact-count outputs are compared against a C++
// reference mirroring sphereCollision.cuh + distanceSegmentSegment.cuh
// (almostEqual tolerance), with counters asserting each contact-count path fires.

#include "pxgTestFramework.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct XformC { float px, py, pz, pw, qx, qy, qz, qw; };
struct CapsuleTest { float radius0, halfHeight0, radius1, halfHeight1, contactDist, p0, p1, p2; };
struct F4 { float x, y, z, w; };
struct V3 { float x, y, z; };

V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
V3 add(V3 a, V3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
V3 mul(V3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
V3 cross(V3 a, V3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
V3 normalize(V3 v) { float r = 1.f / std::sqrt(dot(v, v)); return mul(v, r); }
float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

struct Quat { float x, y, z, w; };
Quat conj(Quat q) { return { -q.x, -q.y, -q.z, q.w }; }
Quat qmul(Quat a, Quat b)
{
    return { a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
             a.w * b.y + a.y * b.w + a.z * b.x - a.x * b.z,
             a.w * b.z + a.z * b.w + a.x * b.y - a.y * b.x,
             a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z };
}
V3 qrot(Quat q, V3 v)
{
    const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d = q.x * vx + q.y * vy + q.z * vz;
    return { vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d,
             vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d,
             vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d };
}
V3 qrotInv(Quat q, V3 v)
{
    const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d = q.x * vx + q.y * vy + q.z * vz;
    return { vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * d,
             vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * d,
             vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * d };
}
V3 basis0(Quat q)
{
    const float x2 = q.x * 2, w2 = q.w * 2;
    return { (q.w * w2) - 1.f + q.x * x2, (q.z * w2) + q.y * x2, (-q.y * w2) + q.z * x2 };
}
Quat qof(const XformC& t) { return { t.qx, t.qy, t.qz, t.qw }; }
V3 pof(const XformC& t) { return { t.px, t.py, t.pz }; }
V3 xformPt(const XformC& t, V3 p) { return add(qrot(qof(t), p), pof(t)); }

void distSegSegSq(V3 p1, V3 d1, V3 p2, V3 d2, float& s, float& t)
{
    const float eps = 1e-6f;
    V3 r = sub(p1, p2);
    float a = dot(d1, d1), e = dot(d2, d2), b = dot(d1, d2), c = dot(d1, r);
    float aRecip = a > eps ? 1.f / a : 0.f, eRecip = e > eps ? 1.f / e : 0.f, f = dot(d2, r);
    float denom = a * e - b * b, temp = b * f - c * e;
    float s0 = clampf(temp / denom, 0.f, 1.f);
    float sTmp = eps > denom ? 0.5f : s0;
    float tTmp = (b * sTmp + f) * eRecip;
    float t2 = clampf(tTmp, 0.f, 1.f);
    float comp = (b * t2 - c) * aRecip;
    s = clampf(comp, 0.f, 1.f);
    t = t2;
}
void pcm22(V3 a0, V3 b0, V3 a1, V3 b1, V3 p0, V3 p1, V3 p2, V3 p3,
           float& t0, float& t1, float& t2, float& t3)
{
    V3 ab0 = sub(b0, a0), ab1 = sub(b1, a1);
    float denom0 = dot(ab0, ab0), denom1 = dot(ab1, ab1), eps = 1e-6f;
    t0 = std::fabs(denom0) < eps ? 0.f : dot(sub(p0, a0), ab0) / denom0;
    t1 = std::fabs(denom0) < eps ? 0.f : dot(sub(p1, a0), ab0) / denom0;
    t2 = std::fabs(denom1) < eps ? 0.f : dot(sub(p2, a1), ab1) / denom1;
    t3 = std::fabs(denom1) < eps ? 0.f : dot(sub(p3, a1), ab1) / denom1;
}

struct Result { uint32_t nb; V3 pt[2]; float pen[2]; V3 normal; };

Result refPlaneCapsule(const XformC& t0, const XformC& t1, const CapsuleTest& c)
{
    Result r{}; r.nb = 0;
    Quat q0 = qof(t0);
    Quat bToAq = qmul(conj(q0), qof(t1));
    V3 bToAp = qrotInv(q0, sub(pof(t1), pof(t0)));
    V3 planeNormal = normalize(basis0(q0));
    r.normal = mul(planeNormal, -1.f);
    V3 bv = basis0(bToAq);
    V3 tmp = mul(bv, c.halfHeight0);
    V3 segS = add(bToAp, tmp), segE = sub(bToAp, tmp);
    float inflatedRadius = c.radius0 + c.contactDist;
    if (inflatedRadius >= segS.x) { r.pt[r.nb] = sub(xformPt(t0, segS), mul(planeNormal, c.radius0)); r.pen[r.nb] = segS.x - c.radius0; r.nb++; }
    if (inflatedRadius >= segE.x) { r.pt[r.nb] = sub(xformPt(t0, segE), mul(planeNormal, c.radius0)); r.pen[r.nb] = segE.x - c.radius0; r.nb++; }
    return r;
}

Result refCapsuleCapsule(const XformC& t0, const XformC& t1, const CapsuleTest& c)
{
    Result r{}; r.nb = 0;
    V3 posOff = mul(add(pof(t0), pof(t1)), 0.5f);
    V3 p0 = sub(pof(t0), posOff), p1 = sub(pof(t1), posOff);
    V3 tmp0 = mul(basis0(qof(t0)), c.halfHeight0);
    V3 s0 = add(p0, tmp0), e0 = sub(p0, tmp0), d0 = sub(e0, s0);
    V3 tmp1 = mul(basis0(qof(t1)), c.halfHeight1);
    V3 s1 = add(p1, tmp1), e1 = sub(p1, tmp1), d1 = sub(e1, s1);
    float sumRadius = c.radius0 + c.radius1;
    float inflatedSum = sumRadius + c.contactDist;
    float inflatedSumSquared = inflatedSum * inflatedSum;
    float a = dot(d0, d0), e = dot(d1, d1), eps = 1e-6f;
    float t0v, t1v; distSegSegSq(s0, d0, s1, d1, t0v, t1v);
    V3 closestA = add(s0, mul(d0, t0v)), closestB = add(s1, mul(d1, t1v));
    V3 vv = sub(closestA, closestB);
    float sqDist0 = dot(vv, vv);

    if (eps > sqDist0)
    {
        V3 normal = normalize(qrotInv(qof(t0), V3{ 0, 1, 0 }));
        r.pt[0] = add(sub(closestA, mul(normal, c.radius0)), posOff);
        r.pen[0] = -c.radius0; r.normal = normal; r.nb = 1;
    }
    else if (inflatedSumSquared >= sqDist0)
    {
        float parallelTolerance = 0.9998f;
        V3 dir0 = (eps > a) ? V3{ 0, 0, 0 } : mul(d0, 1.f / std::sqrt(a));
        V3 dir1 = (eps > e) ? V3{ 0, 0, 0 } : mul(d1, 1.f / std::sqrt(e));
        float cosv = std::fabs(dot(dir0, dir1));
        if (cosv > parallelTolerance)
        {
            float tV0, tV1, tV2, tV3;
            pcm22(s0, e0, s1, e1, s1, e1, s0, e0, tV0, tV1, tV2, tV3);
            auto emit = [&](V3 proj, V3 vminus) {
                V3 v = vminus; float dist = std::sqrt(dot(v, v)); V3 normal = mul(v, 1.f / dist);
                r.pt[r.nb] = add(sub(proj, mul(normal, c.radius0)), posOff); r.pen[r.nb] = dist - sumRadius; r.normal = normal; r.nb++;
            };
            if (tV0 >= 0.f && tV0 <= 1.f) { V3 projS1 = add(s0, mul(d0, tV0)); emit(projS1, sub(projS1, s1)); }
            if (tV1 >= 0.f && tV1 <= 1.f) { V3 projE1 = add(s0, mul(d0, tV1)); emit(projE1, sub(projE1, e1)); }
            if (r.nb < 2 && tV2 >= 0.f && tV2 <= 1.f) { V3 projS0 = add(s1, mul(d1, tV2)); emit(s0, sub(s0, projS0)); }
            if (r.nb < 2 && tV3 >= 0.f && tV3 <= 1.f) { V3 projE0 = add(s1, mul(d1, tV3)); emit(e0, sub(e0, projE0)); }
        }
        if (r.nb == 0)
        {
            V3 normal = normalize(vv);
            r.pt[0] = add(sub(closestA, mul(normal, c.radius0)), posOff);
            r.pen[0] = std::sqrt(sqDist0) - sumRadius; r.normal = normal; r.nb = 1;
        }
    }
    return r;
}

XformC randXform(std::mt19937& rng, float posRange)
{
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 100000) / 100000.f); };
    float qx = rf(-1, 1), qy = rf(-1, 1), qz = rf(-1, 1), qw = rf(-1, 1);
    float inv = 1.f / std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw + 1e-20f);
    return { rf(-posRange, posRange), rf(-posRange, posRange), rf(-posRange, posRange), 0, qx * inv, qy * inv, qz * inv, qw * inv };
}

bool eq(float a, float b) { return almostEqual(a, b, 2e-3f, 2e-3f); }

PXG_TEST(npCapsuleCollision_planeCapsule)
{
    std::mt19937 rng(42010);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 100000) / 100000.f); };
    std::vector<XformC> x0(N), x1(N);
    std::vector<CapsuleTest> tests(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        x0[i] = randXform(rng, 0.5f); // plane
        CapsuleTest c{}; c.radius0 = rf(0.3f, 1.0f); c.halfHeight0 = rf(0.5f, 2.0f); c.contactDist = rf(0.f, 0.2f); tests[i] = c;
        // place capsule near the plane: along the plane normal at a small signed distance
        XformC pl = x0[i];
        V3 n = normalize(basis0(qof(pl)));
        x1[i] = randXform(rng, 0.3f);
        float sd = rf(-1.0f, 2.5f);
        V3 base = add(pof(pl), mul(n, sd));
        x1[i].px = base.x; x1[i].py = base.y; x1[i].pz = base.z;
    }
    std::vector<Result> refs(N);
    size_t c1 = 0, c2 = 0;
    for (uint32_t i = 0; i < N; ++i) { refs[i] = refPlaneCapsule(x0[i], x1[i], tests[i]); if (refs[i].nb == 1) ++c1; if (refs[i].nb == 2) ++c2; }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<F4> outPP(2 * N, F4{ 9, 9, 9, 9 }), outN(N, F4{ 9, 9, 9, 9 });
    std::vector<uint32_t> outNb(N, 7);
    PxgBufferHandle b0 = createBufferFrom(dev, x0), b1 = createBufferFrom(dev, x1), bt = createBufferFrom(dev, tests);
    PxgBufferHandle bpp = createBufferFrom(dev, outPP), bn = createBufferFrom(dev, outN), bnb = createBufferFrom(dev, outNb);
    struct CapsuleParams { uint32_t numTests; } p = { N };
    Kernel k("npCapsuleCollision.spv", "planeCapsule", { 32, 32, 32, 16, 16, 4 }, sizeof(CapsuleParams));
    k.dispatch(GROUPS, { b0, b1, bt, bpp, bn, bnb }, p); dev->flush();
    const auto gPP = downloadBuffer<F4>(dev, bpp, 2 * N);
    const auto gN = downloadBuffer<F4>(dev, bn, N);
    const auto gNb = downloadBuffer<uint32_t>(dev, bnb, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const Result& r = refs[i];
        if (gNb[i] != r.nb) { ++mm; continue; }
        if (r.nb) { if (!eq(gN[i].x, r.normal.x) || !eq(gN[i].y, r.normal.y) || !eq(gN[i].z, r.normal.z)) ++mm; }
        for (uint32_t cc = 0; cc < r.nb; ++cc)
            if (!eq(gPP[2 * i + cc].x, r.pt[cc].x) || !eq(gPP[2 * i + cc].y, r.pt[cc].y) ||
                !eq(gPP[2 * i + cc].z, r.pt[cc].z) || !eq(gPP[2 * i + cc].w, r.pen[cc])) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "planeCapsule: %zu mismatches", mm);
    PXG_CHECK_MSG(c1 > N / 20 && c2 > N / 20, "planeCapsule: paths under-exercised (nb1=%zu nb2=%zu)", c1, c2);
    for (PxgBufferHandle h : { b0, b1, bt, bpp, bn, bnb }) dev->destroyBuffer(h);
}

PXG_TEST(npCapsuleCollision_capsuleCapsule)
{
    std::mt19937 rng(42020);
    const uint32_t N = 8000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 100000) / 100000.f); };
    std::vector<XformC> x0(N), x1(N);
    std::vector<CapsuleTest> tests(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        CapsuleTest c{}; c.radius0 = rf(0.3f, 0.9f); c.halfHeight0 = rf(0.5f, 2.0f);
        c.radius1 = rf(0.3f, 0.9f); c.halfHeight1 = rf(0.5f, 2.0f); c.contactDist = rf(0.f, 0.2f);
        tests[i] = c;
        float sumR = c.radius0 + c.radius1;
        x0[i] = randXform(rng, 0.5f);
        if (i % 2 == 0)
        {
            // parallel: same orientation, side-by-side within sum radius
            x1[i] = x0[i];
            V3 axis = normalize(basis0(qof(x0[i])));
            V3 rnd = { rf(-1, 1), rf(-1, 1), rf(-1, 1) };
            V3 perp = cross(axis, rnd);
            if (dot(perp, perp) < 1e-4f) perp = cross(axis, V3{ 1, 0, 0 });
            perp = normalize(perp);
            V3 base = add(pof(x0[i]), add(mul(perp, sumR * rf(0.4f, 0.95f)), mul(axis, rf(-1.f, 1.f) * c.halfHeight0)));
            x1[i].px = base.x; x1[i].py = base.y; x1[i].pz = base.z;
        }
        else
        {
            // general: random orientation, placed near
            x1[i] = randXform(rng, 0.5f);
            V3 dir = normalize(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) });
            V3 base = add(pof(x0[i]), mul(dir, rf(0.f, sumR + 0.5f)));
            x1[i].px = base.x; x1[i].py = base.y; x1[i].pz = base.z;
        }
    }
    std::vector<Result> refs(N);
    size_t c1 = 0, c2 = 0;
    for (uint32_t i = 0; i < N; ++i) { refs[i] = refCapsuleCapsule(x0[i], x1[i], tests[i]); if (refs[i].nb == 1) ++c1; if (refs[i].nb == 2) ++c2; }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<F4> outPP(2 * N, F4{ 9, 9, 9, 9 }), outN(N, F4{ 9, 9, 9, 9 });
    std::vector<uint32_t> outNb(N, 7);
    PxgBufferHandle b0 = createBufferFrom(dev, x0), b1 = createBufferFrom(dev, x1), bt = createBufferFrom(dev, tests);
    PxgBufferHandle bpp = createBufferFrom(dev, outPP), bn = createBufferFrom(dev, outN), bnb = createBufferFrom(dev, outNb);
    struct CapsuleParams { uint32_t numTests; } p = { N };
    Kernel k("npCapsuleCollision.spv", "capsuleCapsule", { 32, 32, 32, 16, 16, 4 }, sizeof(CapsuleParams));
    k.dispatch(GROUPS, { b0, b1, bt, bpp, bn, bnb }, p); dev->flush();
    const auto gPP = downloadBuffer<F4>(dev, bpp, 2 * N);
    const auto gN = downloadBuffer<F4>(dev, bn, N);
    const auto gNb = downloadBuffer<uint32_t>(dev, bnb, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const Result& r = refs[i];
        if (gNb[i] != r.nb) { ++mm; continue; }
        if (r.nb) { if (!eq(gN[i].x, r.normal.x) || !eq(gN[i].y, r.normal.y) || !eq(gN[i].z, r.normal.z)) ++mm; }
        for (uint32_t cc = 0; cc < r.nb; ++cc)
            if (!eq(gPP[2 * i + cc].x, r.pt[cc].x) || !eq(gPP[2 * i + cc].y, r.pt[cc].y) ||
                !eq(gPP[2 * i + cc].z, r.pt[cc].z) || !eq(gPP[2 * i + cc].w, r.pen[cc])) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "capsuleCapsule: %zu mismatches", mm);
    PXG_CHECK_MSG(c1 > N / 20 && c2 > N / 20, "capsuleCapsule: paths under-exercised (nb1=%zu nb2=%zu)", c1, c2);
    for (PxgBufferHandle h : { b0, b1, bt, bpp, bn, bnb }) dev->destroyBuffer(h);
}

} // namespace
