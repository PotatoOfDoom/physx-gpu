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

// Verification of npBoxClip.slang: contains (2D point-in-polygon) and
// intersectSegmentAABB (segment-AABB slab test).  Randomized convex quads +
// query points (inside / outside / on-vertex) and randomized segments vs boxes
// (through / missing / parallel) are compared against C++ references transcribed
// from cudaBox.cu.

#include "pxgTestFramework.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

const uint32_t MAX_VERTS = 8;

struct F4 { float x, y, z, w; };
struct ContainsTest { float px, py, bminx, bminy, bmaxx, bmaxy; uint32_t numVerts, pad; };
struct SegTest { F4 p0, d, mn, mx; };

// mirrors cudaBox.cu contains()
bool refContains(const std::vector<F4>& verts, uint32_t base, uint32_t n, const ContainsTest& c)
{
    const float tx = c.px, ty = c.py;
    if (tx < c.bminx || tx > c.bmaxx || ty < c.bminy || ty > c.bmaxy) return false;
    const float eps = 1e-6f, zero = 0.f;
    uint32_t ip = 0;
    uint32_t i = 0, j = n - 1;
    for (; i < n; j = i++)
    {
        const float jy = verts[base + j].y, iy = verts[base + i].y;
        const float jx = verts[base + j].x, ix = verts[base + i].x;
        if ((tx == jx && ty == jy) || (tx == ix && ty == iy)) return true;
        const bool yflag0 = jy > ty, yflag1 = iy > ty;
        if (yflag0 != yflag1)
        {
            const float jix = ix - jx, jiy = iy - jy, jty = ty - jy;
            const float part1 = jty * jix, part2 = (jx + eps) * jiy, part3 = tx * jiy;
            const bool comp = jiy > zero;
            const float tmp = part1 + part2;
            const float comp1 = comp ? tmp : part3, comp2 = comp ? part3 : tmp;
            if (comp1 >= comp2) { if (ip == 1) return false; ip++; }
        }
    }
    return ip > 0;
}

// mirrors cudaBox.cu intersectSegmentAABB()
bool refSeg(const SegTest& s, float& tmin, float& tmax)
{
    const float eps = 1e-6f, fMax = 3.4028235e38f;
    const float dx = s.d.x, dy = s.d.y, dz = s.d.z;
    const float ax = std::fabs(dx), ay = std::fabs(dy), az = std::fabs(dz);
    float tminf = 0.f, tmaxf = 1.f;
    const bool pX = eps > ax, pY = eps > ay, pZ = eps > az;
    if ((pX && (s.p0.x > s.mx.x || s.p0.x < s.mn.x)) ||
        (pY && (s.p0.y > s.mx.y || s.p0.y < s.mn.y)) ||
        (pZ && (s.p0.z > s.mx.z || s.p0.z < s.mn.z))) { tmin = 0; tmax = 1; return false; }
    const float ox = 1.f / dx, oy = 1.f / dy, oz = 1.f / dz;
    const float t1x = pX ? 0.f : (s.mn.x - s.p0.x) * ox, t1y = pY ? 0.f : (s.mn.y - s.p0.y) * oy, t1z = pZ ? 0.f : (s.mn.z - s.p0.z) * oz;
    const float t2x = pX ? fMax : (s.mx.x - s.p0.x) * ox, t2y = pY ? fMax : (s.mx.y - s.p0.y) * oy, t2z = pZ ? fMax : (s.mx.z - s.p0.z) * oz;
    const float ft1 = std::max(std::min(t1x, t2x), std::max(std::min(t1y, t2y), std::min(t1z, t2z)));
    const float ft2 = std::min(std::max(t1x, t2x), std::min(std::max(t1y, t2y), std::max(t1z, t2z)));
    tminf = std::max(ft1, tminf);
    tmaxf = std::min(tmaxf, ft2);
    tmin = tminf; tmax = tmaxf;
    return !((tminf > tmaxf) || (tminf > 1.f));
}

PXG_TEST(npBoxClip_contains)
{
    std::mt19937 rng(47010);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 100000) / 100000.f); };

    std::vector<F4> verts(N * MAX_VERTS, F4{ 0, 0, 0, 0 });
    std::vector<ContainsTest> tests(N);
    size_t inCount = 0, outCount = 0, cornerOut = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        // a ROTATED convex quad: rotating the box means points can land inside the
        // AABB early-out yet outside the polygon, exercising the 0/2-crossing paths.
        const float a = rf(0.5f, 2.0f), b = rf(0.5f, 2.0f), cx = rf(-3, 3), cy = rf(-3, 3);
        const float ang = rf(0.05f, 1.5f); // avoid exact axis alignment
        const float ca = std::cos(ang), sa = std::sin(ang);
        auto rot = [&](float lx, float ly) { return F4{ cx + ca * lx - sa * ly, cy + sa * lx + ca * ly, 0, 0 }; };
        F4 v0 = rot(-a, -b), v1 = rot(a, -b), v2 = rot(a, b), v3 = rot(-a, b);
        verts[i * MAX_VERTS + 0] = v0; verts[i * MAX_VERTS + 1] = v1; verts[i * MAX_VERTS + 2] = v2; verts[i * MAX_VERTS + 3] = v3;
        ContainsTest c{}; c.numVerts = 4;
        c.bminx = std::min(std::min(v0.x, v1.x), std::min(v2.x, v3.x));
        c.bminy = std::min(std::min(v0.y, v1.y), std::min(v2.y, v3.y));
        c.bmaxx = std::max(std::max(v0.x, v1.x), std::max(v2.x, v3.x));
        c.bmaxy = std::max(std::max(v0.y, v1.y), std::max(v2.y, v3.y));
        if (i % 3 == 0) { c.px = cx + ca * rf(-0.8f, 0.8f) * a - sa * rf(-0.8f, 0.8f) * b; c.py = cy + sa * a * rf(-0.8f, 0.8f) + ca * b * rf(-0.8f, 0.8f); } // inside (rotated frame)
        else if (i % 3 == 1) { c.px = rf(c.bminx, c.bmaxx); c.py = rf(c.bminy, c.bmaxy); } // anywhere in AABB (inside or corner-outside)
        else { c.px = c.bmaxx + rf(0.2f, 1.5f); c.py = rf(c.bminy - 1.f, c.bmaxy + 1.f); } // outside the AABB
        tests[i] = c;
        const bool inside = refContains(verts, i * MAX_VERTS, 4, c);
        if (inside) ++inCount; else ++outCount;
        // count points inside the AABB but outside the polygon (the crossing paths)
        if (!inside && c.px >= c.bminx && c.px <= c.bmaxx && c.py >= c.bminy && c.py <= c.bmaxy) ++cornerOut;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<uint32_t> outInit(N, 9), dummyU(16, 0);
    std::vector<F4> dummyF(16, F4{ 0, 0, 0, 0 });
    std::vector<SegTest> dummySeg(16);
    PxgBufferHandle bVerts = createBufferFrom(dev, verts);
    PxgBufferHandle bIn = createBufferFrom(dev, tests);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);
    PxgBufferHandle bSeg = createBufferFrom(dev, dummySeg);
    PxgBufferHandle bSegOut = createBufferFrom(dev, dummyF);

    struct BoxClipParams { uint32_t numTests; } p = { N };
    Kernel k("npBoxClip.spv", "boxContains", { 16, 32, 4, 64, 16 }, sizeof(BoxClipParams));
    k.dispatch(GROUPS, { bVerts, bIn, bOut, bSeg, bSegOut }, p);
    dev->flush();

    const auto g = downloadBuffer<uint32_t>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const uint32_t r = refContains(verts, i * MAX_VERTS, 4, tests[i]) ? 1u : 0u;
        if (g[i] != r) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "boxContains: %zu mismatches", mm);
    PXG_CHECK_MSG(inCount > N / 5 && outCount > N / 5 && cornerOut > N / 20,
                  "contains: mix under-exercised (in=%zu out=%zu cornerOut=%zu)", inCount, outCount, cornerOut);

    for (PxgBufferHandle h : { bVerts, bIn, bOut, bSeg, bSegOut }) dev->destroyBuffer(h);
}

PXG_TEST(npBoxClip_intersectSegmentAABB)
{
    std::mt19937 rng(47020);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 100000) / 100000.f); };

    std::vector<SegTest> segs(N);
    size_t hitCount = 0, missCount = 0, parallelCount = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        SegTest s{};
        const float ex = rf(0.5f, 2.f), ey = rf(0.5f, 2.f), ez = rf(0.5f, 2.f);
        s.mn = { -ex, -ey, -ez, 0 }; s.mx = { ex, ey, ez, 0 };
        F4 a{ rf(-4, 4), rf(-4, 4), rf(-4, 4), 0 };
        F4 b{ rf(-4, 4), rf(-4, 4), rf(-4, 4), 0 };
        if (i % 3 == 0) { a = { rf(-4, 4), rf(-4, 4), rf(3, 4), 0 }; b = { a.x, a.y, rf(3, 4), 0 }; ++parallelCount; } // parallel-ish (tiny dz)
        s.p0 = a; s.d = { b.x - a.x, b.y - a.y, b.z - a.z, 0 };
        segs[i] = s;
        float tmn, tmx;
        if (refSeg(s, tmn, tmx)) ++hitCount; else ++missCount;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<F4> outInit(N, F4{ 9, 9, 9, 9 }), dummyF(16, F4{ 0, 0, 0, 0 });
    std::vector<uint32_t> dummyU(16, 0);
    std::vector<ContainsTest> dummyC(16);
    PxgBufferHandle bVerts = createBufferFrom(dev, dummyF);
    PxgBufferHandle bIn = createBufferFrom(dev, dummyC);
    PxgBufferHandle bOut = createBufferFrom(dev, dummyU);
    PxgBufferHandle bSeg = createBufferFrom(dev, segs);
    PxgBufferHandle bSegOut = createBufferFrom(dev, outInit);

    struct BoxClipParams { uint32_t numTests; } p = { N };
    Kernel k("npBoxClip.spv", "boxIntersectSegmentAABB", { 16, 32, 4, 64, 16 }, sizeof(BoxClipParams));
    k.dispatch(GROUPS, { bVerts, bIn, bOut, bSeg, bSegOut }, p);
    dev->flush();

    const auto g = downloadBuffer<F4>(dev, bSegOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        float tmn, tmx;
        const bool hit = refSeg(segs[i], tmn, tmx);
        if ((g[i].z != 0.f) != hit) { ++mm; continue; }
        if (hit && (!almostEqual(g[i].x, tmn, 1e-4f, 1e-5f) || !almostEqual(g[i].y, tmx, 1e-4f, 1e-5f))) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "boxIntersectSegmentAABB: %zu mismatches", mm);
    PXG_CHECK_MSG(hitCount > N / 8 && missCount > N / 8 && parallelCount > N / 8, "seg mix under-exercised (%zu/%zu/%zu)", hitCount, missCount, parallelCount);

    for (PxgBufferHandle h : { bVerts, bIn, bOut, bSeg, bSegOut }) dev->destroyBuffer(h);
}

} // namespace
