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

// Verification of npCvxCoreFace.slang: the Gu::ConvexCore contactFace family
// for the six analytic core types + rotatePoints + the ConvexShape wrapper.
// Exact CPU transcription.  Branch decisions ahead of any trig are sqrt/div
// only and every non-trig output compares BITWISE (numPoints, faceNormal,
// and facePoints of all segment/box/ellipsoid/side branches).  The cylinder/
// cone cap branches run rotatePoints, whose acos/sin/cos are not bit-exact
// primitives (CUDA's own acosf differs in ulps from any libm) -- those
// facePoints compare to tolerance with NaN==NaN (an out-of-domain acos input
// NaN-poisons both sides identically); whether the rotation applied at all
// (the three FLT_EPSILON normalize guards) is deterministic and compares
// exactly via the trig flag.

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
               CORE_CYLINDER = 4, CORE_CONE = 5;

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static float pxSign(float a) { return (a >= 0.f) ? 1.f : -1.f; }
static V3 nrmz(V3 v)
{
    const float m = v.x * v.x + v.y * v.y + v.z * v.z;
    if (m > 0) { const float i = 1.f / std::sqrt(m); return V3{ v.x * i, v.y * i, v.z * i }; }
    return V3{ 0, 0, 0 };
}
static float normDiv(V3& v) // PxVec3::normalize -- division form
{
    const float m = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (m > 0) { v.x /= m; v.y /= m; v.z /= m; }
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
    const float angle = std::acos(dot3(dir0, dir1));
    const float a = angle * 0.5f;
    const float sh = std::sin(a), ch = std::cos(a);
    const F4 rot{ axis.x * sh, axis.y * sh, axis.z * sh, ch };
    for (uint32_t i = 0; i < count; ++i)
        pts[i] = add(qRot(rot, sub(pts[i], center)), center);
    return true;
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

PXG_TEST(npCvxCoreFace_contactFace)
{
    std::mt19937 rng(165001);
    const uint32_t N = 9000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto randQuat = [&]() { float a[4]; float m2 = 0; for (int k = 0; k < 4; ++k) { a[k] = rf(-1, 1); m2 += a[k] * a[k]; } const float i = 1.f / std::sqrt(m2 > 0 ? m2 : 1.f); return F4{ a[0] * i, a[1] * i, a[2] * i, a[3] * i }; };

    std::vector<FaceIn> in(N);
    std::vector<FaceOut> ref(N);
    size_t cTrig = 0, c4 = 0, c2 = 0, c0 = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        FaceIn s{};
        const uint32_t type = rng() % 6;
        s.meta = U4{ type, 0, 0, 0 };
        s.poseQ = randQuat();
        const float margin = (rng() % 3 == 0) ? 0.f : rf(0.01f, 0.4f);
        s.poseP = F4{ rf(-3, 3), rf(-3, 3), rf(-3, 3), margin };
        s.refp = F4{ rf(-4, 4), rf(-4, 4), rf(-4, 4), 0 };

        // build the query in LOCAL space so the branch thresholds are
        // controllable, then rotate to world (both sides recover the same
        // local dir bitwise)
        V3 local{};
        const uint32_t mode = rng() % 4;
        if (mode == 0) // near an axis (inside the 8-degree face cone)
        {
            const uint32_t ax = rng() % 3;
            local = V3{ rf(-0.05f, 0.05f), rf(-0.05f, 0.05f), rf(-0.05f, 0.05f) };
            setComp(local, ax, (rng() % 2) ? 1.f : -1.f);
        }
        else if (mode == 1) // near a coordinate plane (perpendicular band)
        {
            local = V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) };
            setComp(local, rng() % 3, rf(-0.05f, 0.05f));
        }
        else if (mode == 2) // exact axis
        {
            local = V3{ 0, 0, 0 };
            setComp(local, rng() % 3, (rng() % 2) ? 1.f : -1.f);
        }
        else
            local = V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) };
        const V3 world = qRot(s.poseQ, local);
        s.dir = F4{ world.x, world.y, world.z, 0 };

        if (type == CORE_SEGMENT) s.core[0].x = f2u(rf(0.2f, 4));
        else if (type == CORE_BOX)
            for (int c = 0; c < 3; ++c)
                (&s.core[0].x)[c] = f2u((rng() % 5 == 0) ? 0.f : rf(0.1f, 3)); // zero extents drive the degeneracy outs
        else if (type == CORE_ELLIPSOID)
        {
            // the branches REQUIRE degenerate radii: flat (one zero) and
            // needle (two zeros) alongside fully round ones
            const uint32_t pat = rng() % 4;
            for (int c = 0; c < 3; ++c) (&s.core[0].x)[c] = f2u(rf(0.1f, 2));
            if (pat == 1) s.core[0].x = f2u(0.f);
            else if (pat == 2) { s.core[0].y = f2u(0.f); s.core[0].z = f2u(0.f); }
            else if (pat == 3) { s.core[0].x = f2u(0.f); s.core[0].z = f2u(0.f); }
        }
        else if (type == CORE_CYLINDER || type == CORE_CONE)
        {
            s.core[0].x = f2u((rng() % 7 == 0) ? 0.f : rf(0.2f, 3));
            s.core[0].y = f2u((rng() % 7 == 3) ? 0.f : rf(0.1f, 2));
        }

        in[i] = s;
        refFace(s, ref[i]);
        if (ref[i].counts.y) ++cTrig;
        if (ref[i].counts.x == 4) ++c4;
        else if (ref[i].counts.x == 2) ++c2;
        else ++c0;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<FaceOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npCvxCoreFace.spv", "ccfContactFace",
             { (uint32_t)sizeof(FaceIn), (uint32_t)sizeof(FaceOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<FaceOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const FaceOut& r = ref[i]; const FaceOut& q = g[i];
        bool ok = r.counts.x == q.counts.x && r.counts.y == q.counts.y &&
                  eqF4Bits(r.normal, q.normal);
        const bool trig = r.counts.y != 0;
        for (int j = 0; j < 4 && ok; ++j)
            ok = trig ? eqF4Tol(r.fp[j], q.fp[j]) : eqF4Bits(r.fp[j], q.fp[j]);
        if (!ok) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "ccfContactFace: %zu mismatches", mm);
    PXG_CHECK_MSG(cTrig > N / 40 && c4 > N / 20 && c2 > N / 20 && c0 > N / 20,
                  "ccfContactFace: coverage thin (trig %zu c4 %zu c2 %zu c0 %zu)", cTrig, c4, c2, c0);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
