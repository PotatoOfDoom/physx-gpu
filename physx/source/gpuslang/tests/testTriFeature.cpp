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

// P8.30 — the second pass's triangle-feature classifier (triFeatureBda.spv /
// pxgSlangTriFeature).
//
// Suite (ctest filter "triFeature"):
//   triFeatureClassify — the barycentric classifier over a DENSE sweep of (v, w)
//                        plus the boundary cases the order and the pairings turn
//                        on, and the point overload driven through
//                        barycentricCoordinates on a real triangle.

#include "pxgTestFramework.h"

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>

namespace
{
using namespace pxgtest;

struct FeaturePush { uint64_t bary, pts, outB, outP; uint32_t count, pad36; };
static_assert(sizeof(FeaturePush) == 40, "feature blob must be 40 bytes");

static const uint32_t NONE = 0u, V0 = 1u << 29, V1 = 2u << 29, V2 = 3u << 29,
                      E01 = 4u << 29, E02 = 5u << 29, E12 = 6u << 29;

// convexTriangle.cuh:1562, verbatim.
static uint32_t cpuMask(float v, float w)
{
    const float UB = 0.97f;
    const float u = 1.0f - v - w;
    if (v > UB) return V1;
    else if (w > UB) return V2;
    else if (u > UB) return V0;
    else if ((v + u) > UB) return E01;
    else if ((u + w) > UB) return E02;
    else if ((w + v) > UB) return E12;
    return NONE;
}

struct F3 { float x, y, z; };
static F3 sub3(F3 a, F3 b) { return { a.x-b.x, a.y-b.y, a.z-b.z }; }
static float dot3(F3 a, F3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static F3 cross3(F3 a, F3 b)
{ return { a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x }; }

// convexTriangle.cuh:1538, verbatim, including the exact == 0 degenerate pin.
static void cpuBary(F3 p, F3 a, F3 b, F3 c, float& v, float& w)
{
    const F3 ab = sub3(b, a), ac = sub3(c, a);
    const F3 n = cross3(ab, ac);
    const F3 ap = sub3(a, p), bp = sub3(b, p), cp = sub3(c, p);
    const float va = dot3(n, cross3(bp, cp));
    const float vb = dot3(n, cross3(cp, ap));
    const float vc = dot3(n, cross3(ap, bp));
    const float total = va + vb + vc;
    const float denom = total == 0.0f ? 0.0f : 1.0f / total;
    v = vb * denom;
    w = vc * denom;
}

PXG_TEST(triFeatureClassify)
{
    PxgSlangComputeDevice* dev = testDevice();

    std::vector<float> bary;      // (v, w) pairs
    std::vector<float> pts;       // pa, v0, v1, v2 as float4s

    // A dense sweep of barycentric space, INCLUDING outside the triangle: the
    // classifier is handed whatever the contact produced and clamps nothing, and
    // a port that added a clamp would look identical on the inside.
    for (int i = -8; i <= 108; ++i)
        for (int j = -8; j <= 108; ++j)
        { bary.push_back(float(i) * 0.01f); bary.push_back(float(j) * 0.01f); }

    // The cases the ORDER and the PAIRINGS turn on, called out one by one.
    const float edge[][2] = {
        { 0.98f, 0.00f },   // V1 wins over E01 and E12, both of which are also true
        { 0.00f, 0.98f },   // V2 wins over E02 and E12
        { 0.00f, 0.00f },   // u = 1 -> V0 wins over E01 and E02
        { 0.97f, 0.00f },   // EXACTLY the bound: `>` not `>=`, so NOT V1
        { 0.00f, 0.97f },   // exactly the bound on w
        { 0.015f, 0.015f }, // u = 0.97 exactly -> not V0
        { 0.50f, 0.00f },   // w = 0 -> v + u = 1 -> E01
        { 0.00f, 0.50f },   // v = 0 -> u + w = 1 -> E02
        { 0.50f, 0.50f },   // u = 0 -> w + v = 1 -> E12
        { 0.34f, 0.33f },   // deep interior -> NONE
        // Three DEGENERATE-triangle cases (see the pts loop): upstream tests
        // `totalArea == 0.f` exactly and substitutes a ZERO reciprocal, so v and
        // w both come out 0 and the classifier reports V0.  A port that just
        // divided would produce NaN, every comparison would be false, and the
        // answer would silently become NONE.
        { 0.20f, 0.30f }, { 0.60f, 0.10f }, { 0.05f, 0.05f },
    };
    for (const auto& e : edge) { bary.push_back(e[0]); bary.push_back(e[1]); }

    const uint32_t N = uint32_t(bary.size() / 2);

    // The point overload: drive each (v, w) back through a real triangle, so
    // barycentricCoordinates has to reproduce the pair it was built from.  The
    // triangle is deliberately NOT axis-aligned and NOT unit-sized.
    const F3 t0{ -0.3f, 0.2f, 1.1f }, t1{ 2.1f, -0.4f, 0.7f }, t2{ 0.5f, 1.9f, -1.3f };
    // The last three cases get a DEGENERATE triangle instead -- collinear,
    // duplicated and fully coincident -- so the exact `totalArea == 0` pin in
    // barycentricCoordinates is exercised rather than assumed.
    const F3 d0{ 1.0f, 1.0f, 1.0f }, d1{ 2.0f, 2.0f, 2.0f }, d2{ 3.0f, 3.0f, 3.0f };
    for (uint32_t i = 0; i < N; ++i)
    {
        const bool degen = (i + 3u >= N);
        const F3 a = degen ? d0 : t0;
        const F3 b = degen ? d1 : t1;
        const F3 c = degen ? (i + 1u == N ? d0 : d2) : t2;
        const float v = bary[2*i], w = bary[2*i + 1], u = 1.0f - v - w;
        const F3 p{ u*a.x + v*b.x + w*c.x, u*a.y + v*b.y + w*c.y, u*a.z + v*b.z + w*c.z };
        const F3 q[4] = { p, a, b, c };
        for (int k = 0; k < 4; ++k)
        { pts.push_back(q[k].x); pts.push_back(q[k].y); pts.push_back(q[k].z); pts.push_back(0.0f); }
    }

    std::vector<uint32_t> zero(N, 0xFFFFFFFFu);
    PxgBufferHandle bB = createBufferFrom(dev, bary);
    PxgBufferHandle bP = createBufferFrom(dev, pts);
    PxgBufferHandle oB = createBufferFrom(dev, zero);
    PxgBufferHandle oP = createBufferFrom(dev, zero);

    FeaturePush push{};
    push.bary = dev->getBufferDeviceAddress(bB);
    push.pts  = dev->getBufferDeviceAddress(bP);
    push.outB = dev->getBufferDeviceAddress(oB);
    push.outP = dev->getBufferDeviceAddress(oP);
    push.count = N;

    Kernel k("triFeatureBda.spv", "triFeatureLaunch", {}, sizeof(FeaturePush));
    k.dispatch((N + 63u) / 64u, {}, push, 1u, 1u);
    dev->flush();

    const auto gotB = downloadBuffer<uint32_t>(dev, oB, N);
    const auto gotP = downloadBuffer<uint32_t>(dev, oP, N);

    size_t mmB = 0, mmP = 0;
    uint32_t seen[7] = {};
    for (uint32_t i = 0; i < N; ++i)
    {
        const float v = bary[2*i], w = bary[2*i + 1];
        const uint32_t exp = cpuMask(v, w);
        if (gotB[i] != exp)
        {
            ++mmB;
            if (mmB <= 8) std::printf("  bary (%.3f, %.3f): got=%u exp=%u\n", v, w, gotB[i] >> 29, exp >> 29);
        }
        seen[exp >> 29] += 1u;

        // The point overload must agree with the barycentrics it was built from.
        float rv, rw;
        const F3 p{ pts[i*16 + 0], pts[i*16 + 1], pts[i*16 + 2] };
        const F3 a{ pts[i*16 + 4], pts[i*16 + 5], pts[i*16 + 6] };
        const F3 b{ pts[i*16 + 8], pts[i*16 + 9], pts[i*16 + 10] };
        const F3 c{ pts[i*16 + 12], pts[i*16 + 13], pts[i*16 + 14] };
        cpuBary(p, a, b, c, rv, rw);
        const uint32_t expP = cpuMask(rv, rw);
        if (gotP[i] != expP)
        {
            ++mmP;
            if (mmP <= 8) std::printf("  point (%.3f, %.3f): got=%u exp=%u\n", v, w, gotP[i] >> 29, expP >> 29);
        }
    }

    // Every one of the seven outcomes must actually occur, or the sweep proves
    // less than it looks like it does.
    size_t missing = 0;
    const char* names[7] = { "NONE", "V0", "V1", "V2", "E01", "E02", "E12" };
    for (int s = 0; s < 7; ++s)
        if (seen[s] == 0u) { ++missing; std::printf("  outcome %s never occurs in the sweep\n", names[s]); }

    PXG_CHECK_MSG(mmB == 0, "classifier matches CPU over %u cases (%zu mismatches)", N, mmB);
    PXG_CHECK_MSG(mmP == 0, "point overload matches CPU over %u cases (%zu mismatches)", N, mmP);
    PXG_CHECK_MSG(missing == 0, "every classification outcome is exercised (%zu missing)", missing);

    for (PxgBufferHandle h : { bB, bP, oB, oP }) dev->destroyBuffer(h);
}

} // namespace
