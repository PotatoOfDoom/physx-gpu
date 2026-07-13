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

// Verification of npCvxStage1Refresh.slang: convexConvexNphase_stage1Kernel --
// PCM margins, aToB, manifold refresh + in-place compaction, the rerun decision
// (count change or invalidate_BoxConvex), counts word and moreData rewrite.
// Reference transcribes the CUDA; manifold floats to tolerance, counts/flags
// exact, keep/invalidate boundary near-ties margin-skipped.  Sphere/capsule
// fixtures cap contacts at 2 (the sphere threshold tables have 3 entries).

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
struct S1In { U4 meta; F4 scale0, scale1, extents0, extents1, t0q, t0p, t1q, t1p, mNp[4], mA[4], mB[4]; U4 md0, md1, md2, md3; };
struct S1Out { F4 mNp[4], mA[4], mB[4]; U4 md0, md1, md2, md3, result; };

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
static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

static const float thC[5] = { 0.5f, 0.125f, 0.25f, 0.375f, 0.375f };
static const float qthC[5] = { 0.9998f, 0.9999f, 0.9999f, 0.9999f, 0.9999f };
static const float thS[3] = { 0.5f, 0.1f, 0.75f };
static const float qthS[3] = { 0.9995f, 0.9999f, 0.9997f };

static float pcmMargin(const V3& e, float tol) { return std::fmin(std::fmin(std::fmin(e.x, e.y), e.z) * 0.25f, tol * 0.05f); }

static void refStage1(const S1In& s, S1Out& o, bool& marg)
{
    uint32_t type0 = s.meta.x, type1 = s.meta.y;
    V3 sc0 = v3(s.scale0), sc1 = v3(s.scale1);
    V3 e0r = v3(s.extents0), e1r = v3(s.extents1);
    const float tol = s.scale0.w;
    F4 t0q = s.t0q, t1q = s.t1q;
    V3 t0p = v3(s.t0p), t1p = v3(s.t1p);
    if (type1 < type0)
    {
        std::swap(type0, type1); std::swap(sc0, sc1); std::swap(e0r, e1r);
        std::swap(t0q, t1q); std::swap(t0p, t1p);
    }

    float m0; V3 e0;
    if (type0 == 0 || type0 == 2) { m0 = sc0.y; e0 = V3{ 0, 0, 0 }; }
    else { e0 = V3{ e0r.x * sc0.x, e0r.y * sc0.y, e0r.z * sc0.z }; m0 = pcmMargin(e0, tol); }
    const V3 e1{ e1r.x * sc1.x, e1r.y * sc1.y, e1r.z * sc1.z };
    const float m1 = pcmMargin(e1, tol);
    const float minMargin = std::fmin(m0, m1);

    const F4 t1qc{ -t1q.x, -t1q.y, -t1q.z, t1q.w };
    const F4 aToBq = qMul(t1qc, t0q);
    const V3 aToBp = qRotInv(t1q, sub(t0p, t1p));

    const uint32_t initNb = s.md3.x;

    for (int j = 0; j < 4; ++j) { o.mNp[j] = s.mNp[j]; o.mA[j] = s.mA[j]; o.mB[j] = s.mB[j]; }

    F4 kNp[4], kA[4], kB[4];
    uint32_t newNb = 0;
    for (uint32_t j = 0; j < 4 && j < initNb; ++j)
    {
        const F4 pA = s.mA[j], pB = s.mB[j];
        F4 np = s.mNp[j];
        const V3 lA = add(qRot(aToBq, v3(pA)), aToBp);
        const V3 v = sub(lA, v3(pB));
        const V3 n = v3(np);
        const float dist = dot3(v, n);
        np.w = dist;
        const V3 proj = sub(lA, mul(n, dist));
        const V3 pd = sub(v3(pB), proj);
        const float d2 = dot3(pd, pd);
        const float th = minMargin * 0.8f;
        const float sqTh = th * th;
        if (std::fabs(d2 - sqTh) < TIE * (1.f + sqTh)) marg = true;
        if (d2 <= sqTh) { kNp[newNb] = np; kA[newNb] = pA; kB[newNb] = pB; ++newNb; }
    }
    for (uint32_t j = 0; j < newNb; ++j) { o.mNp[j] = kNp[j]; o.mA[j] = kA[j]; o.mB[j] = kB[j]; }

    bool bRun = newNb != initNb;
    if (!bRun)
    {
        const V3 preP{ u2f(s.md0.x), u2f(s.md0.y), u2f(s.md0.z) };
        const F4 preQA{ u2f(s.md1.x), u2f(s.md1.y), u2f(s.md1.z), u2f(s.md1.w) };
        const F4 preQB{ u2f(s.md2.x), u2f(s.md2.y), u2f(s.md2.z), u2f(s.md2.w) };
        const float radiusB = std::sqrt(e1.x * e1.x + e1.y * e1.y + e1.z * e1.z);
        float ratio, thQ, radiusA;
        if (type0 == 0 || type0 == 2) { radiusA = e0.y; ratio = thS[newNb]; thQ = qthS[newNb]; }
        else { radiusA = std::sqrt(e0.x * e0.x + e0.y * e0.y + e0.z * e0.z); ratio = thC[newNb]; thQ = qthC[newNb]; }

        const float thP = minMargin * ratio;
        const V3 dp = sub(aToBp, preP);
        const float deltaP = std::fmax(std::fabs(dp.x), std::fmax(std::fabs(dp.y), std::fabs(dp.z)));
        const float dQA = qDot(t0q, preQA);
        const float dQB = qDot(t1q, preQB);
        if (std::fabs(deltaP - thP) < TIE) marg = true;
        if (std::fabs(dQA - thQ) < 1e-5f || std::fabs(dQB - thQ) < 1e-5f) marg = true;
        bool gen = (deltaP > thP) || (thQ > dQA) || (thQ > dQB);
        if (!gen)
        {
            const float aR = dQA < 1.f ? std::acos(dQA) : 0.f;
            const float bR = dQB < 1.f ? std::acos(dQB) : 0.f;
            const float tA = aR * radiusA, tB = bR * radiusB;
            if (std::fabs(tA - thP) < TIE || std::fabs(tB - thP) < TIE) marg = true;
            gen = (tA > thP) || (tB > thP);
        }
        bRun = gen;
    }

    o.md0 = s.md0; o.md1 = s.md1; o.md2 = s.md2; o.md3 = s.md3;
    o.md3.x = newNb | (bRun ? (initNb << 8) : 0);
    if (bRun)
    {
        o.md0 = U4{ f2u(aToBp.x), f2u(aToBp.y), f2u(aToBp.z), 0 };
        o.md1 = U4{ f2u(t0q.x), f2u(t0q.y), f2u(t0q.z), f2u(t0q.w) };
        o.md2 = U4{ f2u(t1q.x), f2u(t1q.y), f2u(t1q.z), f2u(t1q.w) };
    }
    o.result = U4{ bRun ? 1u : 0u, newNb, 0, 0 };
}

static bool eqF4(const F4& a, const F4& b) { return almostEqual(a.x, b.x, 2e-4f, 2e-5f) && almostEqual(a.y, b.y, 2e-4f, 2e-5f) && almostEqual(a.z, b.z, 2e-4f, 2e-5f) && almostEqual(a.w, b.w, 2e-4f, 2e-4f); }
static bool eqU4f(const U4& a, const U4& b) // uint4 holding float bits: compare as floats to tolerance
{ return almostEqual(u2f(a.x), u2f(b.x), 2e-4f, 2e-5f) && almostEqual(u2f(a.y), u2f(b.y), 2e-4f, 2e-5f) && almostEqual(u2f(a.z), u2f(b.z), 2e-4f, 2e-5f); }

PXG_TEST(npCvxStage1Refresh_refresh)
{
    std::mt19937 rng(153001);
    const uint32_t N = 12000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto randQuat = [&]() { V3 ax = unit(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) }); const float a = rf(-3.1f, 3.1f); return F4{ ax.x * std::sin(a * .5f), ax.y * std::sin(a * .5f), ax.z * std::sin(a * .5f), std::cos(a * .5f) }; };

    std::vector<S1In> in(N);
    std::vector<uint8_t> skip(N, 0);
    std::vector<S1Out> ref(N);
    size_t nSkip = 0, cRun = 0, cNoRun = 0, cInvalPath = 0, cSphere = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        S1In s{};
        const bool sphereish = (i % 4 == 0);
        if (sphereish) ++cSphere;
        const bool flipped = (i % 5 == 1); // type1 < type0 -> flip path
        const uint32_t tA = sphereish ? (i % 8 == 0 ? 0u : 2u) : 4u;
        s.meta = flipped ? U4{ 4u, tA == 4u ? 3u : tA, 0, 0 } : U4{ tA, 4u, 0, 0 };
        s.scale0 = F4{ rf(0.5f, 2), rf(0.5f, 2), rf(0.5f, 2), rf(0.5f, 4) }; // .w = toleranceLength
        s.scale1 = F4{ rf(0.5f, 2), rf(0.5f, 2), rf(0.5f, 2), 0 };
        s.extents0 = F4{ rf(0.3f, 2), rf(0.3f, 2), rf(0.3f, 2), 0 };
        s.extents1 = F4{ rf(0.3f, 2), rf(0.3f, 2), rf(0.3f, 2), 0 };
        s.t0q = randQuat(); s.t1q = randQuat();
        s.t0p = F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), 0 };
        s.t1p = F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), 0 };

        const uint32_t nb = sphereish ? (rng() % 3) : (rng() % 5); // sphere tables have 3 entries
        // post-flip transforms determine aToB; construct pointA consistently so
        // a controlled fraction of slots survives the refresh keep test
        {
            F4 ft0q = s.t0q, ft1q = s.t1q; V3 ft0p = v3(s.t0p), ft1p = v3(s.t1p);
            uint32_t ty0 = s.meta.x, ty1 = s.meta.y;
            if (ty1 < ty0) { std::swap(ft0q, ft1q); std::swap(ft0p, ft1p); }
            const F4 fc{ -ft1q.x, -ft1q.y, -ft1q.z, ft1q.w };
            const F4 aq = qMul(fc, ft0q);
            const V3 ap = qRotInv(ft1q, sub(ft0p, ft1p));
            const F4 aqc{ -aq.x, -aq.y, -aq.z, aq.w };
            for (int j = 0; j < 4; ++j)
            {
                const V3 n = unit(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) });
                s.mNp[j] = F4{ n.x, n.y, n.z, rf(-0.5f, 0.2f) };
                const V3 b{ rf(-1, 1), rf(-1, 1), rf(-1, 1) };
                s.mB[j] = F4{ b.x, b.y, b.z, 0 };
                if (rng() % 5 == 0)
                {
                    // unrelated pointA -> almost surely dropped
                    s.mA[j] = F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), 0 };
                }
                else
                {
                    // pointA = aToB^-1(pointB + small in-plane delta) -> often kept
                    const float d = rf(0.f, 0.05f);
                    const V3 tgt = add(b, V3{ rf(-d, d), rf(-d, d), rf(-d, d) });
                    const V3 pa = qRot(aqc, sub(tgt, ap));
                    s.mA[j] = F4{ pa.x, pa.y, pa.z, 0 };
                }
            }
        }

        // previous-frame data: half the time seeded from the CURRENT transforms
        // (small delta -> the invalidate path decides), else random (rerun likely)
        const F4 t1qc{ -s.t1q.x, -s.t1q.y, -s.t1q.z, s.t1q.w };
        const V3 curRel = qRotInv(s.t1q, sub(v3(s.t0p), v3(s.t1p)));
        if (i % 2 == 0)
        {
            const float eps = rf(0, 0.02f);
            s.md0 = U4{ f2u(curRel.x + rf(-eps, eps)), f2u(curRel.y + rf(-eps, eps)), f2u(curRel.z + rf(-eps, eps)), 0 };
            s.md1 = U4{ f2u(s.t0q.x), f2u(s.t0q.y), f2u(s.t0q.z), f2u(s.t0q.w) };
            s.md2 = U4{ f2u(s.t1q.x), f2u(s.t1q.y), f2u(s.t1q.z), f2u(s.t1q.w) };
        }
        else
        {
            const F4 qa = randQuat(), qb = randQuat();
            s.md0 = U4{ f2u(rf(-2, 2)), f2u(rf(-2, 2)), f2u(rf(-2, 2)), 0 };
            s.md1 = U4{ f2u(qa.x), f2u(qa.y), f2u(qa.z), f2u(qa.w) };
            s.md2 = U4{ f2u(qb.x), f2u(qb.y), f2u(qb.z), f2u(qb.w) };
        }
        s.md3 = U4{ nb, rng() % 5, rng(), rng() };
        in[i] = s;

        bool marg = false;
        refStage1(s, ref[i], marg);
        skip[i] = marg ? 1u : 0u;
        if (marg) { ++nSkip; continue; }
        if (ref[i].result.x) ++cRun; else ++cNoRun;
        if (ref[i].result.y == s.md3.x && s.md3.x != 0) ++cInvalPath;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<S1Out> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct S1Params { uint32_t numTests; } prm = { N };
    Kernel k("npCvxStage1Refresh.spv", "cvxStage1Refresh",
             { (uint32_t)sizeof(S1In), (uint32_t)sizeof(S1Out) }, sizeof(S1Params));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<S1Out>(dev, bOut, N);
    size_t mmM = 0, mmD = 0, mmR = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (skip[i]) continue;
        if (g[i].result.x != ref[i].result.x || g[i].result.y != ref[i].result.y) { ++mmR; continue; }
        for (int j = 0; j < 4; ++j)
            if (!eqF4(ref[i].mNp[j], g[i].mNp[j]) || !eqF4(ref[i].mA[j], g[i].mA[j]) || !eqF4(ref[i].mB[j], g[i].mB[j])) { ++mmM; break; }
        if (g[i].md3.x != ref[i].md3.x || g[i].md3.y != ref[i].md3.y || g[i].md3.z != ref[i].md3.z || g[i].md3.w != ref[i].md3.w) ++mmD;
        else if (ref[i].result.x) { if (!eqU4f(ref[i].md0, g[i].md0) || g[i].md1.x != ref[i].md1.x || g[i].md2.x != ref[i].md2.x || g[i].md0.w != 0) ++mmD; }
        else { if (g[i].md0.x != ref[i].md0.x || g[i].md1.x != ref[i].md1.x || g[i].md2.x != ref[i].md2.x) ++mmD; }
    }
    PXG_CHECK_MSG(mmR == 0, "stage1: %zu result mismatches", mmR);
    PXG_CHECK_MSG(mmM == 0, "stage1: %zu manifold mismatches", mmM);
    PXG_CHECK_MSG(mmD == 0, "stage1: %zu moreData mismatches", mmD);
    PXG_CHECK_MSG(nSkip < N / 3, "stage1: too many marginal (%zu/%u)", nSkip, N);
    PXG_CHECK_MSG(cRun > N / 10 && cNoRun > N / 10 && cInvalPath > N / 20 && cSphere > N / 8,
                  "stage1: coverage thin (run %zu noRun %zu inval %zu sphere %zu)", cRun, cNoRun, cInvalPath, cSphere);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
