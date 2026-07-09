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

// Verification of npSegTriangle.slang: distanceSegmentTriangleSquared.  The
// squared distance is compared against a C++ transcription (it is continuous
// across the branch boundaries, so tolerance-robust without margin-skip).  The
// returned witness (t,u,v) is independently validated by reconstructing the
// closest points — segPt = p + t*(q-p), triPt = a + u*(b-a) + v*(c-a) — and
// confirming |segPt - triPt|^2 equals the returned distance: this catches a
// broken witness while remaining robust to arbitrary tie-breaks between two
// equidistant closest features.

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
struct V3 { float x, y, z; };
struct SegTriIn { F4 p, q, a, b, c; };

const float MAXF = 3.4028235e38f;

static inline V3 sub(const V3& a, const V3& b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static inline V3 add(const V3& a, const V3& b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static inline V3 mul(const V3& a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; }
static inline float dot3(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 crossp(const V3& a, const V3& b) { return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static inline V3 normalize3(const V3& a) { float l = std::sqrt(dot3(a, a)); return l > 0 ? mul(a, 1.f / l) : a; }
static inline float clampf(float x) { return std::max(0.f, std::min(1.f, x)); }

bool isValidBary(float v, float w)
{
    const float eps = 1e-6f, zero = -eps, one = 1.f + eps;
    return (v >= zero && v <= one) && (w >= zero && w <= one) && ((v + w) <= one);
}
void segSeg(const V3& p1, const V3& d1, const V3& p2, const V3& d2, float& s, float& t)
{
    const float eps = 1e-6f;
    const V3 r = sub(p1, p2);
    const float a = dot3(d1, d1), e = dot3(d2, d2), b = dot3(d1, d2), c = dot3(d1, r);
    const float aRecip = a > eps ? 1.f / a : 0.f, eRecip = e > eps ? 1.f / e : 0.f;
    const float f = dot3(d2, r);
    const float denom = a * e - b * b, temp = b * f - c * e;
    const float s0 = clampf(temp / denom);
    const float sTmp = eps > denom ? 0.5f : s0;
    const float tTmp = (b * sTmp + f) * eRecip;
    const float t2 = clampf(tTmp);
    const float comp = (b * t2 - c) * aRecip;
    s = clampf(comp); t = t2;
}

// mirrors capsuleTriangle.cuh distanceSegmentTriangleSquared
float refSegTri(const V3& p, const V3& q, const V3& a, const V3& b, const V3& c, float& t, float& u, float& v)
{
    const V3 pq = sub(q, p), ab = sub(b, a), ac = sub(c, a), bc = sub(c, b), ap = sub(p, a), aq = sub(q, a);
    const float d00 = dot3(ab, ab), d01 = dot3(ab, ac), d11 = dot3(ac, ac);
    const float tDenom = d00 * d11 - d01 * d01;
    const float bdenom = tDenom > 0.f ? 1.f / tDenom : MAXF;
    const V3 n = normalize3(crossp(ab, ac));
    const float dist3 = dot3(ap, n), sqDist3 = dist3 * dist3;
    const float dist4 = dot3(aq, n), sqDist4 = dist4 * dist4;
    const float dMul = dist3 * dist4;

    if (dMul <= 0.f)
    {
        const float nom = -dot3(n, ap), denom = 1.f / dot3(n, pq), t0 = nom * denom;
        const V3 ip = add(p, mul(pq, t0)), v2 = sub(ip, a);
        const float d20 = dot3(v2, ab), d21 = dot3(v2, ac);
        const float v00 = (d11 * d20 - d01 * d21) * bdenom, w00 = (d00 * d21 - d01 * d20) * bdenom;
        if (isValidBary(v00, w00)) { t = t0; u = v00; v = w00; return 0.f; }
    }

    const V3 cp31 = sub(p, mul(n, dist3)), pV20 = sub(cp31, a);
    const float v0 = (d11 * dot3(pV20, ab) - d01 * dot3(pV20, ac)) * bdenom;
    const float w0 = (d00 * dot3(pV20, ac) - d01 * dot3(pV20, ab)) * bdenom;
    const bool con0 = isValidBary(v0, w0);
    const V3 cp41 = sub(q, mul(n, dist4)), qV20 = sub(cp41, a);
    const float v1 = (d11 * dot3(qV20, ab) - d01 * dot3(qV20, ac)) * bdenom;
    const float w1 = (d00 * dot3(qV20, ac) - d01 * dot3(qV20, ab)) * bdenom;
    const bool con1 = isValidBary(v1, w1);

    if (con0 && con1)
    {
        if (sqDist4 > sqDist3) { t = 0.f; u = v0; v = w0; return sqDist3; }
        t = 1.f; u = v1; v = w1; return sqDist4;
    }

    float t00, t10, t01, t11, t02, t12;
    segSeg(p, pq, a, ab, t00, t10);
    segSeg(p, pq, b, bc, t01, t11);
    segSeg(p, pq, a, ac, t02, t12);
    V3 tv = sub(add(p, mul(pq, t00)), add(a, mul(ab, t10))); const float sqDist0 = dot3(tv, tv);
    tv = sub(add(p, mul(pq, t01)), add(b, mul(bc, t11))); const float sqDist1 = dot3(tv, tv);
    tv = sub(add(p, mul(pq, t02)), add(a, mul(ac, t12))); const float sqDist2 = dot3(tv, tv);
    const float u10 = t10, v10 = 0.f, u11 = 1.f - t11, v11 = t11, u12 = 0.f, v12 = t12;

    float sqDistPE, uEdge, vEdge, tSeg;
    if (sqDist1 > sqDist0 && sqDist2 > sqDist0) { sqDistPE = sqDist0; uEdge = u10; vEdge = v10; tSeg = t00; }
    else if (sqDist2 > sqDist1) { sqDistPE = sqDist1; uEdge = u11; vEdge = v11; tSeg = t01; }
    else { sqDistPE = sqDist2; uEdge = u12; vEdge = v12; tSeg = t02; }

    if (con0)
    {
        if (sqDistPE > sqDist3) { t = 0.f; u = v0; v = w0; return sqDist3; }
        t = tSeg; u = uEdge; v = vEdge; return sqDistPE;
    }
    if (con1)
    {
        if (sqDistPE > sqDist4) { t = 1.f; u = v1; v = w1; return sqDist4; }
        t = tSeg; u = uEdge; v = vEdge; return sqDistPE;
    }
    t = tSeg; u = uEdge; v = vEdge; return sqDistPE;
}

PXG_TEST(npSegTriangle_distance)
{
    std::mt19937 rng(55010);
    const uint32_t N = 14000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<SegTriIn> in(N);
    std::vector<float> refSq(N);
    size_t zeroCount = 0, posCount = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        V3 a{ rf(-2, 2), rf(-2, 2), rf(-2, 2) };
        V3 b = add(a, mul(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) }, 2.f));
        V3 cc = add(a, mul(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) }, 2.f));
        V3 p, q;
        if (i % 3 == 0)
        {
            // segment crossing the triangle plane near the interior
            float w0 = rf(0.15f, 0.7f), w1 = rf(0.1f, 0.6f); if (w0 + w1 > 0.9f) { w0 *= 0.5f; w1 *= 0.5f; }
            const V3 inside = add(a, add(mul(sub(b, a), w0), mul(sub(cc, a), w1)));
            const V3 nrm = normalize3(crossp(sub(b, a), sub(cc, a)));
            p = add(inside, mul(nrm, rf(0.2f, 1.5f))); q = sub(inside, mul(nrm, rf(0.2f, 1.5f)));
        }
        else { p = V3{ rf(-4, 4), rf(-4, 4), rf(-4, 4) }; q = V3{ rf(-4, 4), rf(-4, 4), rf(-4, 4) }; }
        SegTriIn s{}; s.p = { p.x, p.y, p.z, 0 }; s.q = { q.x, q.y, q.z, 0 };
        s.a = { a.x, a.y, a.z, 0 }; s.b = { b.x, b.y, b.z, 0 }; s.c = { cc.x, cc.y, cc.z, 0 };
        in[i] = s;
        float t, u, v; refSq[i] = refSegTri(p, q, a, b, cc, t, u, v);
        if (refSq[i] < 1e-8f) ++zeroCount; else ++posCount;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<F4> outInit(N, F4{ 9, 9, 9, 9 });
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct SegTriParams { uint32_t numTests; } prm = { N };
    Kernel k("npSegTriangle.spv", "segTriDistance", { (uint32_t)sizeof(SegTriIn), 16 }, sizeof(SegTriParams));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<F4>(dev, bOut, N);
    size_t mmDist = 0, mmWitness = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const V3 p{ in[i].p.x, in[i].p.y, in[i].p.z }, q{ in[i].q.x, in[i].q.y, in[i].q.z };
        const V3 a{ in[i].a.x, in[i].a.y, in[i].a.z }, b{ in[i].b.x, in[i].b.y, in[i].b.z }, cc{ in[i].c.x, in[i].c.y, in[i].c.z };
        const float gSq = g[i].x, gt = g[i].y, gu = g[i].z, gv = g[i].w;
        // 1. squared distance matches the reference (continuous -> tolerance-robust)
        if (!almostEqual(gSq, refSq[i], 1e-3f, 1e-4f)) { ++mmDist; continue; }
        // 2. witness reconstructs the returned distance
        const V3 segPt = add(p, mul(sub(q, p), gt));
        const V3 triPt = add(a, add(mul(sub(b, a), gu), mul(sub(cc, a), gv)));
        const V3 d = sub(segPt, triPt);
        if (!almostEqual(dot3(d, d), gSq, 2e-3f, 2e-4f)) ++mmWitness;
    }
    PXG_CHECK_MSG(mmDist == 0, "segTriDistance: %zu squared-distance mismatches", mmDist);
    PXG_CHECK_MSG(mmWitness == 0, "segTriDistance: %zu witness reconstruction mismatches", mmWitness);
    PXG_CHECK_MSG(zeroCount > N / 20 && posCount > N / 4, "segTriDistance: cross/miss mix thin (%zu/%zu)", zeroCount, posCount);

    dev->destroyBuffer(bIn); dev->destroyBuffer(bOut);
}

} // namespace
