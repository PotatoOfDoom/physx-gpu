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

// Verification of fcPrim.slang (FEM-cloth vertex/triangle primitive contact
// cores from femClothPrimitives.cu: vertPlaneCollide, triSphereCollide,
// vertCapsuleCollide).  Per case: count exact on non-skipped cases, and on
// contact the point/normal/pen/barycentric to 1e-4.  Slack-tracked gates:
// plane contactDist >= x, sphere sqInflatedRadius > sqDist with the 1e-8
// degenerate-normal branch (grazing shield: normal direction error scales
// with 1/dist), capsule dist < inflatedRadius.  Directed fixtures put the
// sphere center EXACTLY on a triangle (bit-identical closest point, exact
// zero distance) to pin the surface-normal fallback deterministically.

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
struct FcWork { F4 a, b, c, shapeP, shapeQ; U4 ids; };
struct FcOut { F4 point, normalPen, barycentric; U4 count; };
struct FcParams { uint32_t numCases, pad0, pad1, pad2; };

struct V3 { float x, y, z; };
static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 add(V3 a, V3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 muls(V3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static V3 v3(F4 f) { return { f.x, f.y, f.z }; }
static float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

struct Slack { bool near = false; void noteSc(float l, float r, float scale) { if (std::fabs(l - r) < 1e-5f * scale) near = true; } };
static float l1n(V3 v) { return std::fabs(v.x) + std::fabs(v.y) + std::fabs(v.z); }

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
static V3 getNormalized(V3 v)
{
    const float m = dot3(v, v);
    return m > 0 ? muls(v, 1.0f / std::sqrt(m)) : V3{ 0, 0, 0 };
}

static V3 refTriClosestST(V3 p, V3 a, V3 b, V3 c, float& s, float& t)
{
    const V3 ab = sub(b, a), ac = sub(c, a), ap = sub(p, a);
    const float d1 = dot3(ab, ap), d2 = dot3(ac, ap);
    if (d1 <= 0 && d2 <= 0) { s = 0; t = 0; return a; }
    const V3 bp = sub(p, b);
    const float d3 = dot3(ab, bp), d4 = dot3(ac, bp);
    if (d3 >= 0 && d4 <= d3) { s = 1; t = 0; return b; }
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) { const float v = d1 / (d1 - d3); s = v; t = 0; return add(a, muls(ab, v)); }
    const V3 cp = sub(p, c);
    const float d5 = dot3(ab, cp), d6 = dot3(ac, cp);
    if (d6 >= 0 && d5 <= d6) { s = 0; t = 1; return c; }
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) { const float w = d2 / (d2 - d6); s = 0; t = w; return add(a, muls(ac, w)); }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0)
    { const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6)); s = 1 - w; t = w; return add(b, muls(sub(c, b), w)); }
    const float denom = 1.0f / (va + vb + vc);
    const float v = vb * denom, w = vc * denom;
    s = v; t = w;
    return add(add(a, muls(ab, v)), muls(ac, w));
}

PXG_TEST(fcPrim_all)
{
    std::mt19937 rng(91009);
    std::uniform_real_distribution<float> u01(0, 1);
    std::uniform_real_distribution<float> uv(-1, 1);

    const uint32_t N = 3072;
    std::vector<FcWork> work(N);
    auto randQ = [&]() {
        F4 q = { uv(rng), uv(rng), uv(rng), 1.0f + uv(rng) };
        const float m = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
        q.x /= m; q.y /= m; q.z /= m; q.w /= m;
        return q;
    };

    for (uint32_t i = 0; i < N; ++i)
    {
        FcWork& w = work[i];
        const uint32_t mode = i % 3;
        const float cd = 0.05f + 0.2f * u01(rng);
        w.ids = { mode, 0, 0, 0 };
        w.shapeP = { uv(rng), uv(rng), uv(rng), 0 };
        w.shapeQ = randQ();
        switch (mode)
        {
        case 0: // vertex near the plane (both sides)
        {
            const V3 pn = basis0(w.shapeQ);
            const V3 vtx = add(add(v3(w.shapeP), muls(pn, -0.4f + 1.0f * u01(rng))),
                               V3{ 0.5f * uv(rng), 0.5f * uv(rng), 0.5f * uv(rng) });
            w.a = { vtx.x, vtx.y, vtx.z, cd };
            break;
        }
        case 1: // triangle near a sphere
        {
            const float radius = 0.2f + 0.4f * u01(rng);
            w.shapeP.w = radius;
            const V3 c0 = add(v3(w.shapeP), V3{ (radius + 0.5f * cd) * uv(rng) * 2.0f, (radius + 0.5f * cd) * uv(rng) * 2.0f, (radius + 0.5f * cd) * uv(rng) * 2.0f });
            for (;;)
            {
                const V3 a = add(c0, V3{ 0.4f * uv(rng), 0.4f * uv(rng), 0.4f * uv(rng) });
                const V3 b = add(c0, V3{ 0.4f * uv(rng), 0.4f * uv(rng), 0.4f * uv(rng) });
                const V3 cc = add(c0, V3{ 0.4f * uv(rng), 0.4f * uv(rng), 0.4f * uv(rng) });
                const V3 e0 = sub(b, a), e1 = sub(cc, a);
                const V3 cn = cross3(e0, e1);
                if (std::sqrt(dot3(cn, cn)) < l1n(e0) * l1n(e1) / 3.0f) continue; // healthy
                w.a = { a.x, a.y, a.z, cd };
                w.b = { b.x, b.y, b.z, 0 };
                w.c = { cc.x, cc.y, cc.z, 0 };
                break;
            }
            if (i % 24 == 1)
            {
                // degenerate-fallback fixture: sphere center EXACTLY on a
                // triangle vertex -> closest == center bitwise, dist == 0
                w.shapeP.x = w.a.x; w.shapeP.y = w.a.y; w.shapeP.z = w.a.z;
            }
            break;
        }
        default: // vertex near a capsule
        {
            const float hh = 0.3f + 0.4f * u01(rng);
            const float radius = 0.15f + 0.3f * u01(rng);
            w.shapeP.w = hh;
            w.ids.y = *(const uint32_t*)&radius;
            const V3 ax = basis0(w.shapeQ);
            const V3 onAxis = add(v3(w.shapeP), muls(ax, hh * 1.4f * uv(rng)));
            V3 dir = { uv(rng), uv(rng), uv(rng) };
            const float m = std::sqrt(std::max(dot3(dir, dir), 1e-3f));
            dir = muls(dir, 1.0f / m);
            // keep strictly off-axis: on-axis points make vtx - closest pure
            // rounding noise (unstable normal) - shielded below but wasteful
            const V3 vtx = add(onAxis, muls(dir, std::max(0.2f * radius, radius - 0.3f * cd + (radius + cd) * uv(rng))));
            w.a = { vtx.x, vtx.y, vtx.z, cd };
            if (i % 24 == 11)
            {
                // axis-aligned exact-zero fixture: identity pose rotated so
                // basis0 = (1,0,0) exactly, q.x = hh - hh = +0, vertex ON the
                // segment -> closest == vertex bitwise, dist == 0 on both
                // sides -> normal stays (0,0,0), pen = -radius (pins the
                // normalize zero-guard)
                w.shapeQ = { 0, 0, 0, 1 };
                w.shapeP = { hh, w.shapeP.y, w.shapeP.z, hh };
                w.a = { 0.7f * hh, w.shapeP.y, w.shapeP.z, cd };
            }
            break;
        }
        }
    }

    // ---- reference ----
    struct Ref { uint32_t count = 0; V3 point = { 0, 0, 0 }, normal = { 0, 0, 0 }; float pen = 0; F4 bary = { 0, 0, 0, 0 }; };
    std::vector<Ref> ref(N);
    std::vector<uint8_t> skip(N, 0);
    size_t hit[3] = { 0 }, miss[3] = { 0 }, degen = 0, axisZero = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        const FcWork& w = work[i];
        const float cd = w.a.w;
        Slack sl;
        Ref& r = ref[i];
        switch (w.ids.x)
        {
        case 0:
        {
            const V3 pIn = quatRotateInv(w.shapeQ, sub(v3(w.a), v3(w.shapeP)));
            sl.noteSc(cd, pIn.x, l1n(v3(w.a)) + l1n(v3(w.shapeP)) + 1.0f);
            if (cd >= pIn.x)
            {
                r.normal = basis0(w.shapeQ);
                r.pen = pIn.x;
                r.point = sub(v3(w.a), muls(r.normal, pIn.x)); // caller's rigidP
                r.count = 1;
            }
            break;
        }
        case 1:
        {
            const V3 center = v3(w.shapeP);
            const float radius = w.shapeP.w;
            const float infl = radius + cd;
            float s, t;
            const V3 cp = refTriClosestST(center, v3(w.a), v3(w.b), v3(w.c), s, t);
            const V3 v = sub(cp, center);
            const float sqDist = dot3(v, v);
            sl.noteSc(infl * infl, sqDist, l1n(v) * (l1n(cp) + l1n(center)) + infl * infl + sqDist + 1e-3f);
            if (infl * infl > sqDist)
            {
                V3 n;
                const float dist = std::sqrt(sqDist);
                if (dist != 0.0f) sl.noteSc(dist, 0.0f, 1000.0f); // grazing shield; exact zero deterministic
                sl.noteSc(dist, 1e-8f, 1e-3f);
                if (dist > 1e-8f)
                    n = muls(v, 1.0f / dist);
                else
                {
                    n = getNormalized(cross3(sub(v3(w.b), v3(w.a)), sub(v3(w.c), v3(w.a))));
                    ++degen;
                }
                r.point = add(center, muls(n, radius));
                r.normal = n;
                r.pen = dist - radius;
                r.bary = { 1.0f - s - t, s, t, 0.0f };
                r.count = 1;
            }
            break;
        }
        default:
        {
            const float hh = w.shapeP.w;
            const float radius = *(const float*)&w.ids.y;
            const V3 dir = basis0(w.shapeQ);
            const V3 q = sub(v3(w.shapeP), muls(dir, hh));
            const float infl = radius + cd;
            const float len = hh * 2.0f;
            float diff = dot3(sub(v3(w.a), q), dir);
            diff = clampf(diff, 0.0f, len);
            const V3 closest = add(q, muls(dir, diff));
            V3 nor = sub(v3(w.a), closest);
            const float dist = std::sqrt(dot3(nor, nor));
            // near-zero shield; exact zeros stay checked - they only arise
            // from the axis-aligned fixtures where q + dir*diff reproduces the
            // vertex bitwise on both sides (random cases are kept off-axis)
            if (dist != 0.0f) sl.noteSc(dist, 0.0f, 1000.0f);
            if (dist > 0.0f) nor = muls(nor, 1.0f / dist);
            sl.noteSc(dist, infl, l1n(v3(w.a)) + l1n(closest) + infl + 1.0f);
            if (dist < infl)
            {
                r.point = v3(w.a);
                r.normal = nor;
                r.pen = dist - radius;
                r.count = 1;
                if (dist == 0.0f) ++axisZero;
            }
            break;
        }
        }
        if (r.count) ++hit[w.ids.x]; else ++miss[w.ids.x];
        skip[i] = sl.near ? 1 : 0;
    }

    // ---- GPU ----
    PxgSlangComputeDevice* dev = testDevice();
    std::vector<FcOut> outInit(N);
    PxgBufferHandle bW = createBufferFrom(dev, work);
    PxgBufferHandle bO = createBufferFrom(dev, outInit);
    FcParams prm = { N, 0, 0, 0 };
    Kernel k("fcPrim.spv", "ctFcPrim", { (uint32_t)sizeof(FcWork), (uint32_t)sizeof(FcOut) }, sizeof(FcParams));
    k.dispatch((N + 255) / 256, { bW, bO }, prm);
    dev->flush();
    const auto gO = downloadBuffer<FcOut>(dev, bO, N);
    dev->destroyBuffer(bW);
    dev->destroyBuffer(bO);

    size_t mm = 0, checked = 0, skipped = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (skip[i]) { ++skipped; continue; }
        ++checked;
        const Ref& r = ref[i];
        if (gO[i].count.x != r.count) { ++mm; continue; }
        if (r.count)
        {
            bool ok = almostEqual(gO[i].normalPen.w, r.pen, 1e-4f, 1e-4f) &&
                      almostEqual(gO[i].normalPen.x, r.normal.x, 1e-4f, 1e-4f) &&
                      almostEqual(gO[i].normalPen.y, r.normal.y, 1e-4f, 1e-4f) &&
                      almostEqual(gO[i].normalPen.z, r.normal.z, 1e-4f, 1e-4f);
            ok = ok && almostEqual(gO[i].point.x, r.point.x, 1e-4f, 1e-4f) &&
                 almostEqual(gO[i].point.y, r.point.y, 1e-4f, 1e-4f) &&
                 almostEqual(gO[i].point.z, r.point.z, 1e-4f, 1e-4f);
            if (work[i].ids.x == 1)
                ok = ok && almostEqual(gO[i].barycentric.x, r.bary.x, 1e-4f, 1e-4f) &&
                     almostEqual(gO[i].barycentric.y, r.bary.y, 1e-4f, 1e-4f) &&
                     almostEqual(gO[i].barycentric.z, r.bary.z, 1e-4f, 1e-4f) &&
                     gO[i].barycentric.w == 0.0f; // the vertex-contact discriminator stays 0 here
            if (!ok) ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "fcPrim: %zu mismatches", mm);
    PXG_CHECK_MSG(checked > N * 9 / 10, "fcPrim: too many skipped (%zu)", skipped);
    PXG_CHECK_MSG(hit[0] > 100 && miss[0] > 100 && hit[1] > 100 && miss[1] > 100 &&
                  hit[2] > 100 && miss[2] > 100 && degen >= 10 && axisZero >= 10,
                  "fcPrim: coverage thin (h %zu/%zu/%zu m %zu/%zu/%zu d=%zu z=%zu)",
                  hit[0], hit[1], hit[2], miss[0], miss[1], miss[2], degen, axisZero);
}

} // namespace
