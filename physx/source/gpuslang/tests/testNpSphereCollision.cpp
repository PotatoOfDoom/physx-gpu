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

// Verification of npSphereCollision.slang: the single-contact sphere-primitive
// routines (spheresphere / sphereplane / spherecapsule / spherebox).  Randomized
// transforms drive both touching and separated cases (and, for the box, both the
// inside-box deep-penetration branch and the surface branch); the GPU
// point/penetration/normal/contact-count outputs are compared against a C++
// reference mirroring sphereCollision.cuh with an almostEqual tolerance.

#include "pxgTestFramework.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct XformC { float px, py, pz, pw, qx, qy, qz, qw; };
struct STest { float sphereRadius, contactDist, radius1, halfHeight, bx, by, bz, bw; };
struct F4 { float x, y, z, w; };

struct V3 { float x, y, z; };
V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
V3 add(V3 a, V3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
V3 mul(V3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

V3 quatRotate(const XformC& t, V3 v)
{
    const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z;
    const float w2 = t.qw * t.qw - 0.5f;
    const float d = t.qx * vx + t.qy * vy + t.qz * vz;
    return { vx * w2 + (t.qy * vz - t.qz * vy) * t.qw + t.qx * d,
             vy * w2 + (t.qz * vx - t.qx * vz) * t.qw + t.qy * d,
             vz * w2 + (t.qx * vy - t.qy * vx) * t.qw + t.qz * d };
}
V3 quatRotateInv(const XformC& t, V3 v)
{
    const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z;
    const float w2 = t.qw * t.qw - 0.5f;
    const float d = t.qx * vx + t.qy * vy + t.qz * vz;
    return { vx * w2 - (t.qy * vz - t.qz * vy) * t.qw + t.qx * d,
             vy * w2 - (t.qz * vx - t.qx * vz) * t.qw + t.qy * d,
             vz * w2 - (t.qx * vy - t.qy * vx) * t.qw + t.qz * d };
}
V3 basis0(const XformC& t)
{
    const float x2 = t.qx * 2, w2 = t.qw * 2;
    return { (t.qw * w2) - 1.0f + t.qx * x2, (t.qz * w2) + t.qy * x2, (-t.qy * w2) + t.qz * x2 };
}
V3 pos(const XformC& t) { return { t.px, t.py, t.pz }; }
V3 xformInv(const XformC& t, V3 pt) { return quatRotateInv(t, sub(pt, pos(t))); }
V3 xformPt(const XformC& t, V3 pt) { return add(quatRotate(t, pt), pos(t)); }
float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

struct Result { uint32_t nb; V3 point; float pen; V3 normal; };

Result refSphereSphere(const XformC& t0, const XformC& t1, const STest& s)
{
    Result r{}; r.nb = 0;
    V3 p0 = pos(t0), p1 = pos(t1);
    V3 delta = sub(p0, p1);
    float distanceSq = dot(delta, delta);
    float radiusSum = s.sphereRadius + s.radius1;
    float inflatedSum = radiusSum + s.contactDist;
    if (inflatedSum * inflatedSum > distanceSq)
    {
        float dist = std::sqrt(distanceSq);
        V3 normal = (1e-4f >= dist) ? V3{ 1, 0, 0 } : mul(delta, 1.f / dist);
        r.point = add(p1, mul(normal, s.radius1));
        r.pen = dist - radiusSum;
        r.normal = normal; r.nb = 1;
    }
    return r;
}
Result refSpherePlane(const XformC& t0, const XformC& t1, const STest& s)
{
    Result r{}; r.nb = 0;
    V3 cInPlane = xformInv(t1, pos(t0));
    float separation = cInPlane.x - s.sphereRadius;
    if (s.contactDist >= separation)
    {
        V3 n = basis0(t1);
        r.point = sub(pos(t0), mul(n, s.sphereRadius));
        r.pen = separation; r.normal = n; r.nb = 1;
    }
    return r;
}
Result refSphereCapsule(const XformC& t0, const XformC& t1, const STest& s)
{
    Result r{}; r.nb = 0;
    V3 c = pos(t0);
    V3 b0 = basis0(t1);
    V3 tmp0 = mul(b0, s.halfHeight);
    V3 segS = add(pos(t1), tmp0), segE = sub(pos(t1), tmp0);
    float radiusSum = s.sphereRadius + s.radius1;
    float inflatedSum = radiusSum + s.contactDist;
    // distancePointSegmentSquared
    V3 ap = sub(c, segS), ab = sub(segE, segS);
    float nom = dot(ap, ab), denom = dot(ab, ab);
    float tVal = clampf(nom / denom, 0.f, 1.f);
    float t = denom > 0.f ? tVal : 0.f;
    V3 vv = sub(ap, mul(ab, t));
    float squareDist = dot(vv, vv);
    if (inflatedSum * inflatedSum > squareDist)
    {
        V3 p = add(segS, mul(sub(segE, segS), t));
        V3 dir = sub(c, p);
        float length = std::sqrt(dot(dir, dir));
        V3 normal = length > 1.192092896e-07f ? mul(dir, 1.f / length) : V3{ 1, 0, 0 };
        r.point = sub(c, mul(normal, s.sphereRadius));
        r.pen = std::sqrt(squareDist) - radiusSum;
        r.normal = normal; r.nb = 1;
    }
    return r;
}
Result refSphereBox(const XformC& t0, const XformC& t1, const STest& s)
{
    Result r{}; r.nb = 0;
    V3 ext = { s.bx, s.by, s.bz };
    V3 sphereOrigin = pos(t0);
    V3 c = xformInv(t1, sphereOrigin);
    float inflatedSum = s.sphereRadius + s.contactDist;
    V3 p = { clampf(c.x, -ext.x, ext.x), clampf(c.y, -ext.y, ext.y), clampf(c.z, -ext.z, ext.z) };
    V3 v = sub(c, p);
    float lengthSq = dot(v, v);
    if (inflatedSum * inflatedSum > lengthSq)
    {
        bool inside = ext.x >= std::fabs(c.x) && ext.y >= std::fabs(c.y) && ext.z >= std::fabs(c.z);
        if (inside)
        {
            V3 absP = { std::fabs(p.x), std::fabs(p.y), std::fabs(p.z) };
            V3 d2s = sub(ext, absP);
            V3 localNorm = p.z >= 0.f ? V3{ 0, 0, 1 } : V3{ 0, 0, -1 };
            float dist = -d2s.z;
            if (d2s.x <= d2s.y && d2s.x <= d2s.z) { localNorm = p.x >= 0.f ? V3{ 1, 0, 0 } : V3{ -1, 0, 0 }; dist = -d2s.x; }
            else if (d2s.y <= d2s.z) { localNorm = p.y >= 0.f ? V3{ 0, 1, 0 } : V3{ 0, -1, 0 }; dist = -d2s.y; }
            V3 normal = quatRotate(t1, localNorm);
            r.pen = dist - s.sphereRadius;
            r.point = sub(sphereOrigin, mul(normal, dist));
            r.normal = normal;
        }
        else
        {
            float recipLength = 1.f / std::sqrt(lengthSq);
            float length = 1.f / recipLength;
            V3 locNorm = mul(v, recipLength);
            r.pen = length - s.sphereRadius;
            r.normal = quatRotate(t1, locNorm);
            r.point = xformPt(t1, p);
        }
        r.nb = 1;
    }
    return r;
}

XformC randXform(std::mt19937& rng, float posRange)
{
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 100000) / 100000.f); };
    float qx = rf(-1, 1), qy = rf(-1, 1), qz = rf(-1, 1), qw = rf(-1, 1);
    float inv = 1.f / std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw + 1e-20f);
    return { rf(-posRange, posRange), rf(-posRange, posRange), rf(-posRange, posRange), 0,
             qx * inv, qy * inv, qz * inv, qw * inv };
}

bool eq(float a, float b) { return almostEqual(a, b, 1e-4f, 1e-4f); }

template <class RefFn>
void runPrimitive(const char* entry, RefFn ref, int kind)
{
    std::mt19937 rng(41010 + kind);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 100000) / 100000.f); };

    std::vector<XformC> x0(N), x1(N);
    std::vector<STest> tests(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        x1[i] = randXform(rng, 3.0f);
        STest s{};
        s.sphereRadius = rf(0.3f, 1.5f);
        s.contactDist = rf(0.0f, 0.2f);
        s.radius1 = rf(0.3f, 1.2f);
        s.halfHeight = rf(0.5f, 2.0f);
        s.bx = rf(0.4f, 1.5f); s.by = rf(0.4f, 1.5f); s.bz = rf(0.4f, 1.5f);
        tests[i] = s;

        // place the sphere near shape1 so ~half collide; for the box, sometimes inside
        x0[i] = randXform(rng, 0.1f);
        V3 dir = { rf(-1, 1), rf(-1, 1), rf(-1, 1) };
        float dl = std::sqrt(dot(dir, dir)) + 1e-6f;
        dir = mul(dir, 1.f / dl);
        float reach = (kind == 3) ? rf(0.0f, 2.5f) : (s.sphereRadius + s.radius1 + rf(-0.4f, 0.6f));
        V3 base = pos(x1[i]);
        x0[i].px = base.x + dir.x * reach;
        x0[i].py = base.y + dir.y * reach;
        x0[i].pz = base.z + dir.z * reach;
    }

    // reference
    std::vector<Result> refs(N);
    for (uint32_t i = 0; i < N; ++i) refs[i] = ref(x0[i], x1[i], tests[i]);

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;

    std::vector<F4> outPP(N, F4{ 9, 9, 9, 9 }), outN(N, F4{ 9, 9, 9, 9 });
    std::vector<uint32_t> outNb(N, 7);

    PxgBufferHandle b0 = createBufferFrom(dev, x0);
    PxgBufferHandle b1 = createBufferFrom(dev, x1);
    PxgBufferHandle bt = createBufferFrom(dev, tests);
    PxgBufferHandle bpp = createBufferFrom(dev, outPP);
    PxgBufferHandle bn = createBufferFrom(dev, outN);
    PxgBufferHandle bnb = createBufferFrom(dev, outNb);

    struct SphereParams { uint32_t numTests; } p = { N };
    Kernel k("npSphereCollision.spv", entry, { 32, 32, 32, 16, 16, 4 }, sizeof(SphereParams));
    k.dispatch(GROUPS, { b0, b1, bt, bpp, bn, bnb }, p);
    dev->flush();

    const auto gPP = downloadBuffer<F4>(dev, bpp, N);
    const auto gN = downloadBuffer<F4>(dev, bn, N);
    const auto gNb = downloadBuffer<uint32_t>(dev, bnb, N);

    size_t mm = 0, nbHit = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const Result& r = refs[i];
        if (gNb[i] != r.nb) { ++mm; continue; }
        if (r.nb)
        {
            ++nbHit;
            if (!eq(gPP[i].x, r.point.x) || !eq(gPP[i].y, r.point.y) || !eq(gPP[i].z, r.point.z) ||
                !eq(gPP[i].w, r.pen) || !eq(gN[i].x, r.normal.x) || !eq(gN[i].y, r.normal.y) ||
                !eq(gN[i].z, r.normal.z))
                ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "%s: %zu mismatches", entry, mm);
    PXG_CHECK_MSG(nbHit > N / 10, "%s: too few contacts (%zu) — test may be vacuous", entry, nbHit);

    for (PxgBufferHandle h : { b0, b1, bt, bpp, bn, bnb }) dev->destroyBuffer(h);
}

PXG_TEST(npSphereCollision_sphereSphere)  { runPrimitive("sphereNphase_sphereSphere", refSphereSphere, 0); }
PXG_TEST(npSphereCollision_spherePlane)   { runPrimitive("sphereNphase_spherePlane", refSpherePlane, 1); }
PXG_TEST(npSphereCollision_sphereCapsule) { runPrimitive("sphereNphase_sphereCapsule", refSphereCapsule, 2); }
PXG_TEST(npSphereCollision_sphereBox)     { runPrimitive("sphereNphase_sphereBox", refSphereBox, 3); }

} // namespace
