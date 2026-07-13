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

// Verification of npCvxPlaneRefresh.slang: the convex-plane manifold refresh
// (reversed v, transformInv on the plane-space A points, x-axis working
// normal, NO compaction), the rerun decision (invalidate_BoxConvex with
// radiusA = radiusB = |RAW blob extents| and the ConvexPlane 0.99996 quat
// table), and the
// no-rerun contact reload.  Reference transcribes the CUDA; counts/flags
// exact, floats to tolerance, all boundaries margin-skipped.

#include "pxgTestFramework.h"

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
struct PrIn { F4 planeTq, planeTp, convexTq, convexTp, extents, scale, mA[4], mB[4]; U4 md0, md1, md2, misc; };
struct PrOut { F4 n, bToAq, bToAp, p[32]; U4 result; };

const float TIE = 2e-4f;

static inline V3 v3(const F4& f) { return V3{ f.x, f.y, f.z }; }
static inline V3 add(const V3& a, const V3& b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static inline V3 sub(const V3& a, const V3& b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static inline V3 mul(const V3& a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; }
static inline float dot3(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 unit(const V3& a) { float l = std::sqrt(dot3(a, a)); return l > 0 ? mul(a, 1.f / l) : a; }
static V3 qRot(const F4& q, const V3& v) { const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z, w2 = q.w * q.w - 0.5f, d2 = q.x * vx + q.y * vy + q.z * vz; return V3{ vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d2, vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d2, vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d2 }; }
static V3 qRotInv(const F4& q, const V3& v) { const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z, w2 = q.w * q.w - 0.5f, d2 = q.x * vx + q.y * vy + q.z * vz; return V3{ vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * d2, vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * d2, vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * d2 }; }
static F4 qMul(const F4& a, const F4& b)
{
    return F4{ a.w * b.x + b.w * a.x + a.y * b.z - a.z * b.y,
               a.w * b.y + b.w * a.y + a.z * b.x - a.x * b.z,
               a.w * b.z + b.w * a.z + a.x * b.y - a.y * b.x,
               a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z };
}
static float qDot(const F4& a, const F4& b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
static V3 getNormalized(const V3& v) { const float m = dot3(v, v); return m > 0 ? mul(v, 1.f / std::sqrt(m)) : V3{ 0, 0, 0 }; }
static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static uint32_t popc(uint32_t m) { uint32_t c = 0; while (m) { c += m & 1u; m >>= 1; } return c; }

static const float thC[5] = { 0.5f, 0.125f, 0.25f, 0.375f, 0.375f };
static const float qthCP[5] = { 0.99996f, 0.99996f, 0.99996f, 0.99996f, 0.99996f };

static void refPlaneRefresh(const PrIn& s, PrOut& o, bool& marg)
{
    const F4 planeTq = s.planeTq; const V3 planeTp = v3(s.planeTp);
    const F4 convexTq = s.convexTq; const V3 convexTp = v3(s.convexTp);
    const float tol = s.scale.w;

    const F4 c{ -planeTq.x, -planeTq.y, -planeTq.z, planeTq.w };
    const F4 bToAq = qMul(c, convexTq);
    const V3 bToAp = qRotInv(planeTq, sub(convexTp, planeTp));

    const V3 n = getNormalized(qRotInv(bToAq, V3{ 1, 0, 0 }));

    const V3 ext{ s.extents.x * s.scale.x, s.extents.y * s.scale.y, s.extents.z * s.scale.z };
    const float minMargin = std::fmin(std::fmin(std::fmin(ext.x, ext.y), ext.z) * 0.25f, tol * 0.05f);

    const uint32_t initNb = s.misc.x;

    uint32_t keepW = 0;
    float distL[32] = { 0 };
    for (uint32_t L = 0; L < 4 && L < initNb; ++L)
    {
        const V3 localAInB = qRotInv(bToAq, sub(v3(s.mA[L]), bToAp));
        const V3 v = sub(v3(s.mB[L]), localAInB);
        const float dist = dot3(v, n);
        distL[L] = dist;
        const V3 pd = sub(v, mul(n, dist));
        const float d2 = dot3(pd, pd);
        const float th = minMargin * 0.8f;
        if (std::fabs(d2 - th * th) < TIE * (1.f + th * th)) marg = true;
        if (d2 <= th * th) keepW |= (1u << L);
    }
    const uint32_t nbContacts = popc(keepW);

    bool bRun = nbContacts != initNb;
    if (!bRun)
    {
        const V3 preP{ u2f(s.md0.x), u2f(s.md0.y), u2f(s.md0.z) };
        const F4 preQA{ u2f(s.md1.x), u2f(s.md1.y), u2f(s.md1.z), u2f(s.md1.w) };
        const F4 preQB{ u2f(s.md2.x), u2f(s.md2.y), u2f(s.md2.z), u2f(s.md2.w) };
        // RAW blob extents (cu L2946), not the scaled margin extents
        const V3 re = v3(s.extents);
        const float radiusB = std::sqrt(re.x * re.x + re.y * re.y + re.z * re.z);
        const float ratio = thC[nbContacts], thQ = qthCP[nbContacts];
        const float thP = minMargin * ratio;
        const V3 curRel = mul(bToAp, -1.f);
        const V3 dp = sub(curRel, preP);
        const float deltaP = std::fmax(std::fabs(dp.x), std::fmax(std::fabs(dp.y), std::fabs(dp.z)));
        const float dQA = qDot(planeTq, preQA);
        const float dQB = qDot(convexTq, preQB);
        if (std::fabs(deltaP - thP) < TIE) marg = true;
        if (std::fabs(dQA - thQ) < 1e-6f || std::fabs(dQB - thQ) < 1e-6f) marg = true;
        bool gen = (deltaP > thP) || (thQ > dQA) || (thQ > dQB);
        if (!gen)
        {
            const float aR = dQA < 1.f ? std::acos(dQA) : 0.f;
            const float bR = dQB < 1.f ? std::acos(dQB) : 0.f;
            const float tA = aR * radiusB, tB = bR * radiusB;
            if (std::fabs(tA - thP) < TIE || std::fabs(tB - thP) < TIE) marg = true;
            gen = (tA > thP) || (tB > thP);
        }
        bRun = gen;
    }

    for (int L = 0; L < 32; ++L) o.p[L] = F4{ 0, 0, 0, 0 };
    uint32_t mask = 0;
    if (!bRun)
    {
        mask = (1u << nbContacts) - 1;
        for (uint32_t L = 0; L < 4 && L < nbContacts; ++L)
        {
            const V3 p = add(qRot(bToAq, v3(s.mB[L])), bToAp);
            o.p[L] = F4{ p.x, p.y, p.z, distL[L] };
        }
    }

    o.n = F4{ n.x, n.y, n.z, minMargin };
    o.bToAq = bToAq;
    o.bToAp = F4{ bToAp.x, bToAp.y, bToAp.z, 0 };
    o.result = U4{ bRun ? 1u : 0u, nbContacts, mask, keepW };
}

static bool eqF4(const F4& a, const F4& b) { return almostEqual(a.x, b.x, 2e-4f, 2e-5f) && almostEqual(a.y, b.y, 2e-4f, 2e-5f) && almostEqual(a.z, b.z, 2e-4f, 2e-5f) && almostEqual(a.w, b.w, 2e-4f, 2e-4f); }

PXG_TEST(npCvxPlaneRefresh_refresh)
{
    std::mt19937 rng(155101);
    const uint32_t N = 12000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto randQuat = [&]() { V3 ax = unit(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) }); const float a = rf(-3.1f, 3.1f); return F4{ ax.x * std::sin(a * .5f), ax.y * std::sin(a * .5f), ax.z * std::sin(a * .5f), std::cos(a * .5f) }; };

    std::vector<PrIn> in(N);
    std::vector<uint8_t> skip(N, 0);
    std::vector<PrOut> ref(N);
    size_t nSkip = 0, cRun = 0, cNoRun = 0, cInval = 0, cDrop = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        PrIn s{};
        s.planeTq = randQuat();
        s.planeTp = F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), 0 };
        s.convexTq = randQuat();
        s.convexTp = F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), 0 };
        s.extents = F4{ rf(0.3f, 2), rf(0.3f, 2), rf(0.3f, 2), 0 };
        s.scale = F4{ rf(0.5f, 2), rf(0.5f, 2), rf(0.5f, 2), rf(0.5f, 4) };

        // manifold: pointA constructed through bToA so keeps occur; some slots
        // get a large in-plane offset so the refresh drops them
        const F4 c{ -s.planeTq.x, -s.planeTq.y, -s.planeTq.z, s.planeTq.w };
        const F4 bToAq = qMul(c, s.convexTq);
        const V3 bToAp = qRotInv(s.planeTq, sub(v3(s.convexTp), v3(s.planeTp)));
        const V3 n = getNormalized(qRotInv(bToAq, V3{ 1, 0, 0 }));
        V3 u = unit(V3{ n.y, -n.x + 0.3f, n.z + 0.1f });
        u = unit(sub(u, mul(n, dot3(u, n)))); // in-plane direction

        const uint32_t initNb = rng() % 5;
        for (uint32_t L = 0; L < 4; ++L)
        {
            const V3 pB{ rf(-1, 1), rf(-1, 1), rf(-1, 1) };
            s.mB[L] = F4{ pB.x, pB.y, pB.z, 0 };
            const bool dropIt = (rng() % 4 == 0);
            if (dropIt && L < initNb) ++cDrop;
            const float perp = dropIt ? rf(0.3f, 1.f) : rf(0.f, 0.01f);
            const float along = rf(-0.5f, 0.5f);
            const V3 tgt = add(pB, add(mul(n, along), mul(u, perp)));
            const V3 pA = add(qRot(bToAq, tgt), bToAp); // pointA in plane space
            s.mA[L] = F4{ pA.x, pA.y, pA.z, 0 };
        }

        // previous-frame data: half seeded from current (invalidate path decides,
        // with occasional tiny perturbations against the tight 0.99996 table)
        if (i % 2 == 0)
        {
            const V3 curRel = mul(bToAp, -1.f);
            const float eps = rf(0, 0.01f);
            s.md0 = U4{ f2u(curRel.x + rf(-eps, eps)), f2u(curRel.y + rf(-eps, eps)), f2u(curRel.z + rf(-eps, eps)), 0 };
            F4 pqa = s.planeTq, pqb = s.convexTq;
            if (i % 8 == 4)
            {
                const V3 ax = unit(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) });
                const float a = rf(0.002f, 0.05f); // sweeps the very tight 0.99996 band
                const F4 rot{ ax.x * std::sin(a * .5f), ax.y * std::sin(a * .5f), ax.z * std::sin(a * .5f), std::cos(a * .5f) };
                if (i % 16 == 4) pqa = qMul(rot, pqa); else pqb = qMul(rot, pqb);
            }
            s.md1 = U4{ f2u(pqa.x), f2u(pqa.y), f2u(pqa.z), f2u(pqa.w) };
            s.md2 = U4{ f2u(pqb.x), f2u(pqb.y), f2u(pqb.z), f2u(pqb.w) };
        }
        else
        {
            const F4 qa = randQuat(), qb = randQuat();
            s.md0 = U4{ f2u(rf(-2, 2)), f2u(rf(-2, 2)), f2u(rf(-2, 2)), 0 };
            s.md1 = U4{ f2u(qa.x), f2u(qa.y), f2u(qa.z), f2u(qa.w) };
            s.md2 = U4{ f2u(qb.x), f2u(qb.y), f2u(qb.z), f2u(qb.w) };
        }
        s.misc = U4{ initNb, 0, 0, 0 };
        in[i] = s;

        bool marg = false;
        refPlaneRefresh(s, ref[i], marg);
        skip[i] = marg ? 1u : 0u;
        if (marg) { ++nSkip; continue; }
        if (ref[i].result.x) ++cRun; else ++cNoRun;
        if (ref[i].result.y == initNb && initNb != 0) ++cInval;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<PrOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct PrParams { uint32_t numTests; } prm = { N };
    Kernel k("npCvxPlaneRefresh.spv", "cvxPlaneRefresh",
             { (uint32_t)sizeof(PrIn), (uint32_t)sizeof(PrOut) }, sizeof(PrParams));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<PrOut>(dev, bOut, N);
    size_t mmR = 0, mmF = 0, mmP = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (skip[i]) continue;
        if (g[i].result.x != ref[i].result.x || g[i].result.y != ref[i].result.y ||
            g[i].result.z != ref[i].result.z || g[i].result.w != ref[i].result.w) { ++mmR; continue; }
        if (!eqF4(ref[i].n, g[i].n) || !eqF4(ref[i].bToAq, g[i].bToAq) || !eqF4(ref[i].bToAp, g[i].bToAp)) ++mmF;
        for (int L = 0; L < 32; ++L) if (!eqF4(ref[i].p[L], g[i].p[L])) { ++mmP; break; }
    }
    PXG_CHECK_MSG(mmR == 0, "planeRefresh: %zu result mismatches", mmR);
    PXG_CHECK_MSG(mmF == 0, "planeRefresh: %zu frame mismatches", mmF);
    PXG_CHECK_MSG(mmP == 0, "planeRefresh: %zu reload mismatches", mmP);
    PXG_CHECK_MSG(nSkip < N / 3, "planeRefresh: too many marginal (%zu/%u)", nSkip, N);
    PXG_CHECK_MSG(cRun > N / 10 && cNoRun > N / 20 && cInval > N / 20 && cDrop > N / 10,
                  "planeRefresh: coverage thin (run %zu noRun %zu inval %zu drop %zu)", cRun, cNoRun, cInval, cDrop);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
