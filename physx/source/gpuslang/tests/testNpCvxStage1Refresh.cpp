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

// P5 Step 4: convexConvexNphase_stage1Kernel on the ENGINE ABI, driven
// through PxgCudaShim exactly as PxgNarrowphaseCore.cpp launches it —
// argument blob {4,4,8,8,8,8,8} (48 bytes), grid ceil(numTests*4/128) x
// (32,4) blocks, REAL PxgContactManagerInput/PxgShape/PxsCachedTransform/
// PxgPersistentContactManifold arrays and real cooked-hull blobs (extents
// float4 at hullPtr + 32) at live device addresses.  The CPU reference is
// the tranche-153 transcription of the CUDA (PCM margins, aToB, manifold
// refresh + in-place compaction, the rerun decision, counts word + moreData
// rewrite); manifold floats to tolerance, counts/flags exact, keep/
// invalidate near-ties margin-skipped.  The last GUARD pairs sit beyond
// numTests: their manifolds and keep slots must keep their sentinels.
// P9.86c: three directed fresh-manifold pairs (kDirZ) pin the SIGN of the
// zero mRelativePos word that aToB = t1.transformInv(t0) stores, bitwise.
// P9.90: the shader is NoContraction-stamped; npCvxStage1Refresh_p90DotOrder compares a
// general-pose refresh bitwise, stored separations v.dot(n) included, in the CUDA dot order.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "PxNodeIndex.h"
#include "PxgContactManager.h"
#include "PxgConvexConvexShape.h"
#include "PxgPersistentContactManifold.h"
#include "PxsTransformCache.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

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

// P9.90: the two PxVec3 dots the refresh stores or tests -- dist = v.dot(localNormal) and
// projectedDifference.magnitudeSquared() (cudaGJKEPA.cu:2278/:2286) -- are (x + y) + z in the CUDA
// (PxVec3.h:173/:274).  gS1DotOrder = 1 re-runs the mirror in the old llvmpipe order x + (y + z),
// which an UNSTAMPED OpDot evaluates; used only to prove the p90 fixture separates the two.
static int gS1DotOrder = 0;
static inline float s1Dot(const V3& a, const V3& b)
{
    return gS1DotOrder ? a.x * b.x + (a.y * b.y + a.z * b.z) : (a.x * b.x + a.y * b.y) + a.z * b.z;
}

static float pcmMargin(const V3& e, float tol) { return std::fmin(std::fmin(std::fmin(e.x, e.y), e.z) * 0.25f, tol * 0.05f); }

// P9.86c mutant twin: true selects the port's PRE-FIX aToB.p spelling q.rotateInv(v) instead
// of the CUDA's t1.transformInv(t0) = q.getConjugate().rotate(v).  Used only to prove that the
// directed zero-sign pairs below separate the two forms.
static bool gAToBRotInvVariant = false;

// CUDA transcription (bound-buffer tranche; aToB.p corrected in P9.86c).
static void refStage1(const S1In& s, float tol, S1Out& o, bool& marg)
{
    uint32_t type0 = s.meta.x, type1 = s.meta.y;
    V3 sc0 = v3(s.scale0), sc1 = v3(s.scale1);
    V3 e0r = v3(s.extents0), e1r = v3(s.extents1);
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
    // cudaGJKEPA.cu:2238 aToB = t1.transformInv(t0) -> PxTransform.h:162-168 qinv = q.getConjugate();
    // qinv.rotate(src.p - p).  NOT q.rotateInv(src.p - p): same bits for every nonzero value, but a
    // component that is a sum of exact zeros can take the other sign (the pre-P9.86c mirror shared
    // the port's old rotateInv form and so could never see the difference).
    const V3 aToBp = gAToBRotInvVariant ? qRotInv(t1q, sub(t0p, t1p)) : qRot(t1qc, sub(t0p, t1p));

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
        const float dist = s1Dot(v, n);
        np.w = dist;
        const V3 proj = sub(lA, mul(n, dist));
        const V3 pd = sub(v3(pB), proj);
        const float d2 = s1Dot(pd, pd);
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

static uint32_t u4w(const U4& u, int k) { return k == 0 ? u.x : k == 1 ? u.y : k == 2 ? u.z : u.w; }

// P9.86c directed zero-sign pairs for aToB.p (fresh manifold, so bRunNarrowphase stores it):
//   D0 (i=7):  t1 = PxQuat(PxHalfPi, x) = (s,+0,+0,s) at (0,3,0), t0 identity at the origin.
//              qinv = (-s,-0,-0,s), v = (+0,-3,+0): CUDA x = -0 + -0 + (-s)(+0) = -0;
//              rotateInv(q) gives (-0 - +0) + s(+0) = +0.
//   D1 (i=17): t1 = PxQuat(-PxHalfPi, z) = (-0,-0,-s,s), same positions: CUDA z = +0, rotateInv -0.
//   D2 (i=27): D0 through the flip path (meta {5,3}; the kernel swaps the cache refs).
// Every term of the diverging component is an exact zero product and w*w-0.5 < 0 fused or not,
// so the sign is contraction-invariant on both sides (stamped NoContraction since P9.90 or not).
struct DirZ { uint32_t idx; int word; uint32_t cudaBits; };
static const DirZ kDirZ[3] = { { 7u, 0, 0x80000000u }, { 17u, 2, 0x00000000u }, { 27u, 0, 0x80000000u } };
static int dirZIndex(uint32_t i) { for (int k = 0; k < 3; ++k) if (i == kDirZ[k].idx) return k; return -1; }

static bool eqF4(const F4& a, const F4& b) { return almostEqual(a.x, b.x, 2e-4f, 2e-5f) && almostEqual(a.y, b.y, 2e-4f, 2e-5f) && almostEqual(a.z, b.z, 2e-4f, 2e-5f) && almostEqual(a.w, b.w, 2e-4f, 2e-4f); }
static bool eqU4f(const U4& a, const U4& b)
{ return almostEqual(u2f(a.x), u2f(b.x), 2e-4f, 2e-5f) && almostEqual(u2f(a.y), u2f(b.y), 2e-4f, 2e-5f) && almostEqual(u2f(a.z), u2f(b.z), 2e-4f, 2e-5f); }

static std::vector<uint8_t> s1Loader(const char* file) { return loadShaderBlob(file); }

PXG_TEST(npCvxStage1Refresh_engineAbi)
{
    using namespace physx;

    std::mt19937 rng(153001);
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto randQuat = [&]() { V3 ax = unit(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) }); const float a = rf(-3.1f, 3.1f); return F4{ ax.x * std::sin(a * .5f), ax.y * std::sin(a * .5f), ax.z * std::sin(a * .5f), std::cos(a * .5f) }; };

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npCvxStage1Refresh.spv", "convexConvexNphase_stage1Kernel", 48u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &s1Loader);

    // toleranceLength is one PUSH scalar per launch (the fixture's only
    // per-test scalar in the bound-buffer form): three launches sweep it.
    const float tols[3] = { 0.75f, 1.6f, 3.2f };
    const uint32_t N = 4000;   // pairs per launch (incl. GUARD tail)
    const uint32_t GUARD = 5;  // beyond numTests: must stay untouched

    size_t nSkip = 0, cRun = 0, cNoRun = 0, cInvalPath = 0, cSphere = 0;
    size_t cZeroSignAToB = 0; // directed pairs on which the CUDA form and the rotateInv twin disagree in the stored zero word
    const float kS45 = u2f(0x3F3504F3u); // sin/cos(PxHalfPi * 0.5f) = 0.70710677f (PxQuat(PxHalfPi, axis))

    for (uint32_t launch = 0; launch < 3; ++launch)
    {
        const float tol = tols[launch];
        const uint32_t numTests = N - GUARD;

        std::vector<S1In> in(N);
        std::vector<uint8_t> skip(N, 0);
        std::vector<S1Out> ref(N);

        for (uint32_t i = 0; i < N; ++i)
        {
            S1In s{};
            const bool sphereish = (i % 4 == 0);
            if (sphereish) ++cSphere;
            const bool flipped = (i % 5 == 1); // type1 < type0 -> flip path
            const uint32_t tA = sphereish ? (i % 8 == 0 ? 0u : 2u) : 4u;
            s.meta = flipped ? U4{ 4u, tA == 4u ? 3u : tA, 0, 0 } : U4{ tA, 4u, 0, 0 };
            s.scale0 = F4{ rf(0.5f, 2), rf(0.5f, 2), rf(0.5f, 2), 0 };
            s.scale1 = F4{ rf(0.5f, 2), rf(0.5f, 2), rf(0.5f, 2), 0 };
            s.extents0 = F4{ rf(0.3f, 2), rf(0.3f, 2), rf(0.3f, 2), 0 };
            s.extents1 = F4{ rf(0.3f, 2), rf(0.3f, 2), rf(0.3f, 2), 0 };
            s.t0q = randQuat(); s.t1q = randQuat();
            s.t0p = F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), 0 };
            s.t1p = F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), 0 };

            const uint32_t nb = sphereish ? (rng() % 3) : (rng() % 5); // sphere tables have 3 entries
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
                        s.mA[j] = F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), 0 };
                    }
                    else
                    {
                        const float d = rf(0.f, 0.05f);
                        const V3 tgt = add(b, V3{ rf(-d, d), rf(-d, d), rf(-d, d) });
                        const V3 pa = qRot(aqc, sub(tgt, ap));
                        s.mA[j] = F4{ pa.x, pa.y, pa.z, 0 };
                    }
                }
            }

            const V3 curRel = qRotInv(s.t1q, sub(v3(s.t0p), v3(s.t1p)));
            if (i % 2 == 0)
            {
                const float eps = rf(0, 0.02f);
                s.md0 = U4{ f2u(curRel.x + rf(-eps, eps)), f2u(curRel.y + rf(-eps, eps)), f2u(curRel.z + rf(-eps, eps)), 0 };
                F4 pqa = s.t0q, pqb = s.t1q;
                if (i % 8 == 4)
                {
                    const V3 ax = unit(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) });
                    const float a = rf(0.004f, 0.07f);
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
            s.md3 = U4{ nb, rng() % 5, rng(), rng() };

            // P9.86c directed zero-sign pairs (see kDirZ): consume no RNG, so every other pair keeps its bits.
            // A fresh manifold as PxgPersistentContactManifold::clear() leaves it (mRelativePos = PX_MAX_REAL,
            // identity quats, no contacts) forces bRunNarrowphase, so aToB.p is stored into mRelativePos.
            const int dz = dirZIndex(i);
            if (dz >= 0)
            {
                const F4 qI{ 0.f, 0.f, 0.f, 1.f };
                const F4 qR = (dz == 1) ? F4{ -0.f, -0.f, -kS45, kS45 } : F4{ kS45, 0.f, 0.f, kS45 };
                const F4 pO{ 0.f, 0.f, 0.f, 0.f }, pU{ 0.f, 3.f, 0.f, 0.f };
                if (dz == 2) { s.meta = U4{ 5u, 3u, 0, 0 }; s.t0q = qR; s.t0p = pU; s.t1q = qI; s.t1p = pO; }
                else         { s.meta = U4{ 3u, 5u, 0, 0 }; s.t0q = qI; s.t0p = pO; s.t1q = qR; s.t1p = pU; }
                s.md0 = U4{ 0x7F7FFFFFu, 0x7F7FFFFFu, 0x7F7FFFFFu, 0x7F7FFFFFu };
                s.md1 = s.md2 = U4{ 0u, 0u, 0u, 0x3F800000u };
                s.md3 = U4{ 0u, 0u, s.md3.z, s.md3.w };
            }
            in[i] = s;

            if (i < numTests)
            {
                bool marg = false;
                refStage1(s, tol, ref[i], marg);
                if (dz >= 0)
                {
                    S1Out tw; bool m2 = false;
                    gAToBRotInvVariant = true; refStage1(s, tol, tw, m2); gAToBRotInvVariant = false;
                    const int w = kDirZ[dz].word; const uint32_t b = kDirZ[dz].cudaBits;
                    if (!marg && !m2 && ref[i].result.x == 1u && tw.result.x == 1u &&
                        u4w(ref[i].md0, w) == b && u4w(tw.md0, w) == (b ^ 0x80000000u))
                        ++cZeroSignAToB;
                }
                skip[i] = marg ? 1u : 0u;
                if (marg) { ++nSkip; continue; }
                if (ref[i].result.x) ++cRun; else ++cNoRun;
                if (ref[i].result.y == s.md3.x && s.md3.x != 0) ++cInvalPath;
            }
        }

        // ---- marshal the fixtures into REAL engine arrays ----
        std::vector<PxgContactManagerInput> cmIn(N);
        std::vector<PxgShape> shapes(2 * N);
        std::vector<PxsCachedTransform> cache(2 * N);
        std::vector<PxgPersistentContactManifold> mfs(N);
        std::vector<uint8_t> blobs(size_t(2) * N * 64, 0);
        std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShape));
        std::memset(cache.data(), 0, cache.size() * sizeof(PxsCachedTransform));
        std::memset(mfs.data(), 0xCD, mfs.size() * sizeof(PxgPersistentContactManifold));

        PxgDevicePtr dBlobs = 0;
        PXG_CHECK_MSG(shim.memAlloc(&dBlobs, blobs.size()) == PxgResult::Success, "memAlloc blobs");
        if (!dBlobs) { std::printf("  NOTE: no BDA -> skip\n"); return; }

        for (uint32_t i = 0; i < N; ++i)
        {
            const S1In& s = in[i];
            cmIn[i].shapeRef0 = 2 * i;
            cmIn[i].shapeRef1 = 2 * i + 1;
            cmIn[i].transformCacheRef0 = 2 * i;
            cmIn[i].transformCacheRef1 = 2 * i + 1;

            PxgShape& sh0 = shapes[2 * i];
            PxgShape& sh1 = shapes[2 * i + 1];
            sh0.type = s.meta.x;
            sh1.type = s.meta.y;
            sh0.scale.scale = PxVec3(s.scale0.x, s.scale0.y, s.scale0.z);
            sh1.scale.scale = PxVec3(s.scale1.x, s.scale1.y, s.scale1.z);
            sh0.scale.rotation = PxQuat(PxIdentity);
            sh1.scale.rotation = PxQuat(PxIdentity);
            sh0.materialIndex = 0;
            sh1.materialIndex = 0;
            sh0.hullOrMeshPtr = size_t(dBlobs + uint64_t(2 * i) * 64u);
            sh1.hullOrMeshPtr = size_t(dBlobs + uint64_t(2 * i + 1) * 64u);
            // blob extents float4 at +32
            const F4 e0{ s.extents0.x, s.extents0.y, s.extents0.z, 0 };
            const F4 e1{ s.extents1.x, s.extents1.y, s.extents1.z, 0 };
            std::memcpy(&blobs[size_t(2 * i) * 64 + 32], &e0, 16);
            std::memcpy(&blobs[size_t(2 * i + 1) * 64 + 32], &e1, 16);

            cache[2 * i].transform.q = PxQuat(s.t0q.x, s.t0q.y, s.t0q.z, s.t0q.w);
            cache[2 * i].transform.p = PxVec3(s.t0p.x, s.t0p.y, s.t0p.z);
            cache[2 * i + 1].transform.q = PxQuat(s.t1q.x, s.t1q.y, s.t1q.z, s.t1q.w);
            cache[2 * i + 1].transform.p = PxVec3(s.t1p.x, s.t1p.y, s.t1p.z);

            PxgPersistentContactManifold& m = mfs[i];
            for (int j = 0; j < 4; ++j)
            {
                m.mLocalNormal_PenW[j] = make_float4(s.mNp[j].x, s.mNp[j].y, s.mNp[j].z, s.mNp[j].w);
                m.mLocalContactA[j] = make_float4(s.mA[j].x, s.mA[j].y, s.mA[j].z, s.mA[j].w);
                m.mLocalContactB[j] = make_float4(s.mB[j].x, s.mB[j].y, s.mB[j].z, s.mB[j].w);
            }
            m.mRelativePos = make_float4(u2f(s.md0.x), u2f(s.md0.y), u2f(s.md0.z), u2f(s.md0.w));
            m.mQuatA = PxAlignedQuat(u2f(s.md1.x), u2f(s.md1.y), u2f(s.md1.z), u2f(s.md1.w));
            m.mQuatB = PxAlignedQuat(u2f(s.md2.x), u2f(s.md2.y), u2f(s.md2.z), u2f(s.md2.w));
            m.mNbContacts = s.md3.x;
            m.mNbWarmStartPoints = s.md3.y;
            m.mWarmStartA = s.md3.z;
            m.mWarmStartB = s.md3.w;
        }

        PxgDevicePtr dCmIn = 0, dShapes = 0, dCache = 0, dMf = 0, dKeep = 0;
        shim.memAlloc(&dCmIn, cmIn.size() * sizeof(PxgContactManagerInput));
        shim.memAlloc(&dShapes, shapes.size() * sizeof(PxgShape));
        shim.memAlloc(&dCache, cache.size() * sizeof(PxsCachedTransform));
        shim.memAlloc(&dMf, mfs.size() * sizeof(PxgPersistentContactManifold));
        shim.memAlloc(&dKeep, N * 4);
        shim.memcpyHtoD(dBlobs, blobs.data(), blobs.size());
        shim.memcpyHtoD(dCmIn, cmIn.data(), cmIn.size() * sizeof(PxgContactManagerInput));
        shim.memcpyHtoD(dShapes, shapes.data(), shapes.size() * sizeof(PxgShape));
        shim.memcpyHtoD(dCache, cache.data(), cache.size() * sizeof(PxsCachedTransform));
        shim.memcpyHtoD(dMf, mfs.data(), mfs.size() * sizeof(PxgPersistentContactManifold));
        shim.memsetD32(dKeep, 0xCCCCCCCC, N);

        // ---- the ENGINE launch: blob {4,4,8,8,8,8,8}, grid ceil(numTests*4/128) x (32,4) ----
        struct Stage1Blob
        {
            uint32_t numTests;
            float toleranceLength;
            uint64_t cmInputs, transformCache, shapes, cmManifolds, tempKeepPairArray;
        } blob = { numTests, tol, dCmIn, dCache, dShapes, dMf, dKeep };
        static_assert(sizeof(Stage1Blob) == 48, "stage1 engine blob is 48 bytes");

        const uint32_t numBlocks = (numTests * 4 + 127) / 128;
        PXG_CHECK_MSG(shim.launchKernel(0, numBlocks, 1, 1, &blob, sizeof(blob)) == PxgResult::Success,
                      "launch stage1 (engine ABI)");
        shim.streamSynchronize();

        // ---- compare ----
        std::vector<PxgPersistentContactManifold> gMf(N);
        std::vector<uint32_t> gKeep(N);
        shim.memcpyDtoH(gMf.data(), dMf, gMf.size() * sizeof(PxgPersistentContactManifold));
        shim.memcpyDtoH(gKeep.data(), dKeep, N * 4);

        size_t mmM = 0, mmD = 0, mmR = 0, mmG = 0, mmZ = 0;
        for (uint32_t i = 0; i < numTests; ++i)
        {
            if (skip[i]) continue;
            const PxgPersistentContactManifold& g = gMf[i];
            S1Out got;
            for (int j = 0; j < 4; ++j)
            {
                std::memcpy(&got.mNp[j], &g.mLocalNormal_PenW[j], 16);
                std::memcpy(&got.mA[j], &g.mLocalContactA[j], 16);
                std::memcpy(&got.mB[j], &g.mLocalContactB[j], 16);
            }
            std::memcpy(&got.md0, &g.mRelativePos, 16);
            std::memcpy(&got.md1, &g.mQuatA, 16);
            std::memcpy(&got.md2, &g.mQuatB, 16);
            got.md3 = U4{ g.mNbContacts, g.mNbWarmStartPoints, g.mWarmStartA, g.mWarmStartB };

            // P9.86c: the directed pairs' stored zero word of mRelativePos, BITWISE (sign included),
            // against both the corrected mirror and the hard-coded CUDA bits.  The rest of md0 stays
            // tolerant here; the P9.90 case below compares a whole refresh bitwise.
            const int dz = dirZIndex(i);
            if (dz >= 0)
            {
                const int w = kDirZ[dz].word;
                if (u4w(got.md0, w) != u4w(ref[i].md0, w) || u4w(got.md0, w) != kDirZ[dz].cudaBits) ++mmZ;
            }

            if (gKeep[i] != ref[i].result.x) { ++mmR; continue; }
            for (int j = 0; j < 4; ++j)
                if (!eqF4(ref[i].mNp[j], got.mNp[j]) || !eqF4(ref[i].mA[j], got.mA[j]) || !eqF4(ref[i].mB[j], got.mB[j])) { ++mmM; break; }
            if (got.md3.x != ref[i].md3.x || got.md3.y != ref[i].md3.y || got.md3.z != ref[i].md3.z || got.md3.w != ref[i].md3.w) ++mmD;
            else if (ref[i].result.x)
            {
                if (!eqU4f(ref[i].md0, got.md0) || got.md0.w != 0 ||
                    got.md1.x != ref[i].md1.x || got.md1.y != ref[i].md1.y || got.md1.z != ref[i].md1.z || got.md1.w != ref[i].md1.w ||
                    got.md2.x != ref[i].md2.x || got.md2.y != ref[i].md2.y || got.md2.z != ref[i].md2.z || got.md2.w != ref[i].md2.w) ++mmD;
            }
            else
            {
                if (got.md0.x != ref[i].md0.x || got.md1.x != ref[i].md1.x || got.md2.x != ref[i].md2.x) ++mmD;
            }
        }
        // GUARD pairs beyond numTests: manifold + keep sentinels untouched
        for (uint32_t i = numTests; i < N; ++i)
        {
            if (std::memcmp(&gMf[i], &mfs[i], sizeof(PxgPersistentContactManifold)) != 0) ++mmG;
            if (gKeep[i] != 0xCCCCCCCCu) ++mmG;
        }
        PXG_CHECK_MSG(mmR == 0, "stage1 launch %u: %zu keep-flag mismatches", launch, mmR);
        PXG_CHECK_MSG(mmM == 0, "stage1 launch %u: %zu manifold mismatches", launch, mmM);
        PXG_CHECK_MSG(mmD == 0, "stage1 launch %u: %zu moreData mismatches", launch, mmD);
        PXG_CHECK_MSG(mmG == 0, "stage1 launch %u: %zu guard-tail violations", launch, mmG);
        PXG_CHECK_MSG(mmZ == 0, "stage1 launch %u: %zu aToB zero-sign words differ (transformInv = getConjugate().rotate, cudaGJKEPA.cu:2238)", launch, mmZ);

        for (PxgDevicePtr pfree : { dBlobs, dCmIn, dShapes, dCache, dMf, dKeep })
            shim.memFree(pfree);
    }

    const size_t total = size_t(3) * (N - GUARD);
    PXG_CHECK_MSG(nSkip < total / 3, "stage1: too many marginal (%zu/%zu)", nSkip, total);
    PXG_CHECK_MSG(cRun > total / 10 && cNoRun > total / 10 && cInvalPath > total / 20 && cSphere > total / 8,
                  "stage1: coverage thin (run %zu noRun %zu inval %zu sphere %zu)", cRun, cNoRun, cInvalPath, cSphere);
    PXG_CHECK_MSG(cZeroSignAToB == 9, "stage1: aToB zero-sign directed pairs lost (%zu/9)", cZeroSignAToB);
}

// P9.90: the NoContraction stamp, observed.  64 convex-convex pairs (16 through the flip
// path) in general orientation with non-round extents, scales and poses; each manifold holds
// four contacts whose A point maps to pB + n * pen + (a ~1e-3 tangential offset), so every
// contact is kept far from the (0.8 * minMargin)^2 edge and the refreshed separation
// dist = v.dot(n) is stored in mLocalNormal_PenW[j].w.  mRelativePos = PX_MAX_REAL forces
// bRunNarrowphase, so aToB.p and both quats are stored too.  The whole refreshed manifold
// and the keep flags are compared BITWISE against refStage1 (-ffp-contract=off, CUDA order
// (x + y) + z); the old-order twin (gS1DotOrder = 1) must move >= 16 stored separations.
// Unstamped, llvmpipe's OpDot sums x + (y + z) and the stored separations are the twin's.
PXG_TEST(npCvxStage1Refresh_p90DotOrder)
{
    using namespace physx;

    std::mt19937 rng(990153);
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000003) / 1000003.f); };
    auto randQuat = [&]() { V3 ax = unit(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) }); const float a = rf(-3.1f, 3.1f); return F4{ ax.x * std::sin(a * .5f), ax.y * std::sin(a * .5f), ax.z * std::sin(a * .5f), std::cos(a * .5f) }; };

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npCvxStage1Refresh.spv", "convexConvexNphase_stage1Kernel", 48u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &s1Loader);

    const float tol = 1.6f;
    const uint32_t N = 64;
    std::vector<S1In> in(N);
    std::vector<S1Out> ref(N), twin(N);
    size_t nMarg = 0, nTwinW = 0, nTwinOther = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        S1In s{};
        const bool flipped = (i % 4 == 1);
        s.meta = flipped ? U4{ 4u, 3u, 0, 0 } : U4{ 3u, 4u, 0, 0 };
        s.scale0 = F4{ rf(0.6f, 1.9f), rf(0.6f, 1.9f), rf(0.6f, 1.9f), 0 };
        s.scale1 = F4{ rf(0.6f, 1.9f), rf(0.6f, 1.9f), rf(0.6f, 1.9f), 0 };
        s.extents0 = F4{ rf(0.4f, 1.7f), rf(0.4f, 1.7f), rf(0.4f, 1.7f), 0 };
        s.extents1 = F4{ rf(0.4f, 1.7f), rf(0.4f, 1.7f), rf(0.4f, 1.7f), 0 };
        s.t0q = randQuat(); s.t1q = randQuat();
        s.t0p = F4{ rf(-2.3f, 2.3f), rf(-2.3f, 2.3f), rf(-2.3f, 2.3f), 0 };
        s.t1p = F4{ rf(-2.3f, 2.3f), rf(-2.3f, 2.3f), rf(-2.3f, 2.3f), 0 };

        F4 ft0q = s.t0q, ft1q = s.t1q; V3 ft0p = v3(s.t0p), ft1p = v3(s.t1p);
        if (flipped) { std::swap(ft0q, ft1q); std::swap(ft0p, ft1p); }
        const F4 fc{ -ft1q.x, -ft1q.y, -ft1q.z, ft1q.w };
        const F4 aq = qMul(fc, ft0q);
        const V3 ap = qRot(fc, sub(ft0p, ft1p));
        const F4 aqc{ -aq.x, -aq.y, -aq.z, aq.w };
        for (int j = 0; j < 4; ++j)
        {
            const V3 n = unit(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) });
            s.mNp[j] = F4{ n.x, n.y, n.z, rf(-0.4f, 0.1f) };
            const V3 b{ rf(-1.3f, 1.3f), rf(-1.3f, 1.3f), rf(-1.3f, 1.3f) };
            s.mB[j] = F4{ b.x, b.y, b.z, 0 };
            const V3 tgt = add(add(b, mul(n, rf(-0.35f, 0.15f))), V3{ rf(-1e-3f, 1e-3f), rf(-1e-3f, 1e-3f), rf(-1e-3f, 1e-3f) });
            const V3 pa = qRot(aqc, sub(tgt, ap));
            s.mA[j] = F4{ pa.x, pa.y, pa.z, 0 };
        }
        // stale moreData: PX_MAX_REAL relative position forces bRunNarrowphase (deltaP > thresholdP)
        s.md0 = U4{ 0x7F7FFFFFu, 0x7F7FFFFFu, 0x7F7FFFFFu, 0u };
        const F4 qa = randQuat(), qb = randQuat();
        s.md1 = U4{ f2u(qa.x), f2u(qa.y), f2u(qa.z), f2u(qa.w) };
        s.md2 = U4{ f2u(qb.x), f2u(qb.y), f2u(qb.z), f2u(qb.w) };
        s.md3 = U4{ 4u, rng() % 5, rng(), rng() };
        in[i] = s;

        bool marg = false, m2 = false;
        refStage1(s, tol, ref[i], marg);
        gS1DotOrder = 1; refStage1(s, tol, twin[i], m2); gS1DotOrder = 0;
        if (marg || m2) ++nMarg;
        for (int j = 0; j < 4; ++j)
        {
            if (f2u(ref[i].mNp[j].w) != f2u(twin[i].mNp[j].w)) ++nTwinW;
            if (std::memcmp(&ref[i].mNp[j], &twin[i].mNp[j], 12) || std::memcmp(&ref[i].mA[j], &twin[i].mA[j], 16) ||
                std::memcmp(&ref[i].mB[j], &twin[i].mB[j], 16)) ++nTwinOther;
        }
        if (std::memcmp(&ref[i].md0, &twin[i].md0, 64) || std::memcmp(&ref[i].result, &twin[i].result, 16)) ++nTwinOther;
    }
    std::printf("  p90: the old-order twin moves %zu/256 stored separations\n", nTwinW);
    PXG_CHECK_MSG(nMarg == 0, "p90: %zu pairs sit on a keep/invalidate edge", nMarg);
    PXG_CHECK_MSG(nTwinOther == 0, "p90: the old-order twin changed %zu words other than the stored separations", nTwinOther);
    PXG_CHECK_MSG(nTwinW >= 16, "p90: the old-order twin x + (y + z) moves only %zu/256 stored separations", nTwinW);
    for (uint32_t i = 0; i < N; ++i)
        PXG_CHECK_MSG(ref[i].result.x == 1u && ref[i].result.y == 4u, "p90: pair %u not kept/rerun as planted", i);

    // ---- marshal into engine arrays ----
    std::vector<PxgContactManagerInput> cmIn(N);
    std::vector<PxgShape> shapes(2 * N);
    std::vector<PxsCachedTransform> cache(2 * N);
    std::vector<PxgPersistentContactManifold> mfs(N);
    std::vector<uint8_t> blobs(size_t(2) * N * 64, 0);
    std::memset(shapes.data(), 0, shapes.size() * sizeof(PxgShape));
    std::memset(cache.data(), 0, cache.size() * sizeof(PxsCachedTransform));
    std::memset(mfs.data(), 0xCD, mfs.size() * sizeof(PxgPersistentContactManifold));

    PxgDevicePtr dBlobs = 0;
    PXG_CHECK_MSG(shim.memAlloc(&dBlobs, blobs.size()) == PxgResult::Success, "memAlloc blobs");
    if (!dBlobs) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    for (uint32_t i = 0; i < N; ++i)
    {
        const S1In& s = in[i];
        cmIn[i].shapeRef0 = 2 * i; cmIn[i].shapeRef1 = 2 * i + 1;
        cmIn[i].transformCacheRef0 = 2 * i; cmIn[i].transformCacheRef1 = 2 * i + 1;
        PxgShape& sh0 = shapes[2 * i];
        PxgShape& sh1 = shapes[2 * i + 1];
        sh0.type = s.meta.x; sh1.type = s.meta.y;
        sh0.scale.scale = PxVec3(s.scale0.x, s.scale0.y, s.scale0.z);
        sh1.scale.scale = PxVec3(s.scale1.x, s.scale1.y, s.scale1.z);
        sh0.scale.rotation = PxQuat(PxIdentity); sh1.scale.rotation = PxQuat(PxIdentity);
        sh0.hullOrMeshPtr = size_t(dBlobs + uint64_t(2 * i) * 64u);
        sh1.hullOrMeshPtr = size_t(dBlobs + uint64_t(2 * i + 1) * 64u);
        const F4 e0{ s.extents0.x, s.extents0.y, s.extents0.z, 0 };
        const F4 e1{ s.extents1.x, s.extents1.y, s.extents1.z, 0 };
        std::memcpy(&blobs[size_t(2 * i) * 64 + 32], &e0, 16);
        std::memcpy(&blobs[size_t(2 * i + 1) * 64 + 32], &e1, 16);
        cache[2 * i].transform.q = PxQuat(s.t0q.x, s.t0q.y, s.t0q.z, s.t0q.w);
        cache[2 * i].transform.p = PxVec3(s.t0p.x, s.t0p.y, s.t0p.z);
        cache[2 * i + 1].transform.q = PxQuat(s.t1q.x, s.t1q.y, s.t1q.z, s.t1q.w);
        cache[2 * i + 1].transform.p = PxVec3(s.t1p.x, s.t1p.y, s.t1p.z);
        PxgPersistentContactManifold& m = mfs[i];
        for (int j = 0; j < 4; ++j)
        {
            m.mLocalNormal_PenW[j] = make_float4(s.mNp[j].x, s.mNp[j].y, s.mNp[j].z, s.mNp[j].w);
            m.mLocalContactA[j] = make_float4(s.mA[j].x, s.mA[j].y, s.mA[j].z, s.mA[j].w);
            m.mLocalContactB[j] = make_float4(s.mB[j].x, s.mB[j].y, s.mB[j].z, s.mB[j].w);
        }
        m.mRelativePos = make_float4(u2f(s.md0.x), u2f(s.md0.y), u2f(s.md0.z), u2f(s.md0.w));
        m.mQuatA = PxAlignedQuat(u2f(s.md1.x), u2f(s.md1.y), u2f(s.md1.z), u2f(s.md1.w));
        m.mQuatB = PxAlignedQuat(u2f(s.md2.x), u2f(s.md2.y), u2f(s.md2.z), u2f(s.md2.w));
        m.mNbContacts = s.md3.x; m.mNbWarmStartPoints = s.md3.y;
        m.mWarmStartA = s.md3.z; m.mWarmStartB = s.md3.w;
    }

    PxgDevicePtr dCmIn = 0, dShapes = 0, dCache = 0, dMf = 0, dKeep = 0;
    shim.memAlloc(&dCmIn, cmIn.size() * sizeof(PxgContactManagerInput));
    shim.memAlloc(&dShapes, shapes.size() * sizeof(PxgShape));
    shim.memAlloc(&dCache, cache.size() * sizeof(PxsCachedTransform));
    shim.memAlloc(&dMf, mfs.size() * sizeof(PxgPersistentContactManifold));
    shim.memAlloc(&dKeep, N * 4);
    shim.memcpyHtoD(dBlobs, blobs.data(), blobs.size());
    shim.memcpyHtoD(dCmIn, cmIn.data(), cmIn.size() * sizeof(PxgContactManagerInput));
    shim.memcpyHtoD(dShapes, shapes.data(), shapes.size() * sizeof(PxgShape));
    shim.memcpyHtoD(dCache, cache.data(), cache.size() * sizeof(PxsCachedTransform));
    shim.memcpyHtoD(dMf, mfs.data(), mfs.size() * sizeof(PxgPersistentContactManifold));
    shim.memsetD32(dKeep, 0xCCCCCCCC, N);

    struct Stage1Blob
    {
        uint32_t numTests;
        float toleranceLength;
        uint64_t cmInputs, transformCache, shapes, cmManifolds, tempKeepPairArray;
    } blob = { N, tol, dCmIn, dCache, dShapes, dMf, dKeep };
    static_assert(sizeof(Stage1Blob) == 48, "stage1 engine blob is 48 bytes");
    PXG_CHECK_MSG(shim.launchKernel(0, (N * 4 + 127) / 128, 1, 1, &blob, sizeof(blob)) == PxgResult::Success,
                  "launch stage1 (p90)");
    shim.streamSynchronize();

    std::vector<PxgPersistentContactManifold> gMf(N);
    std::vector<uint32_t> gKeep(N);
    shim.memcpyDtoH(gMf.data(), dMf, gMf.size() * sizeof(PxgPersistentContactManifold));
    shim.memcpyDtoH(gKeep.data(), dKeep, N * 4);

    size_t mmW = 0, mmWTwin = 0, mmOther = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const PxgPersistentContactManifold& g = gMf[i];
        S1Out got;
        for (int j = 0; j < 4; ++j)
        {
            std::memcpy(&got.mNp[j], &g.mLocalNormal_PenW[j], 16);
            std::memcpy(&got.mA[j], &g.mLocalContactA[j], 16);
            std::memcpy(&got.mB[j], &g.mLocalContactB[j], 16);
            if (f2u(got.mNp[j].w) != f2u(ref[i].mNp[j].w))
            {
                ++mmW;
                if (f2u(got.mNp[j].w) == f2u(twin[i].mNp[j].w)) ++mmWTwin;
            }
            if (std::memcmp(&got.mNp[j], &ref[i].mNp[j], 12) || std::memcmp(&got.mA[j], &ref[i].mA[j], 16) ||
                std::memcmp(&got.mB[j], &ref[i].mB[j], 16)) ++mmOther;
        }
        std::memcpy(&got.md0, &g.mRelativePos, 16);
        std::memcpy(&got.md1, &g.mQuatA, 16);
        std::memcpy(&got.md2, &g.mQuatB, 16);
        got.md3 = U4{ g.mNbContacts, g.mNbWarmStartPoints, g.mWarmStartA, g.mWarmStartB };
        if (std::memcmp(&got.md0, &ref[i].md0, 64) || gKeep[i] != ref[i].result.x) ++mmOther;
    }
    PXG_CHECK_MSG(mmW == 0, "p90: %zu/256 stored separations v.dot(n) differ bitwise from the CUDA order (x + y) + z "
                            "(%zu equal the old-order twin x + (y + z): is npCvxStage1Refresh NoContraction-stamped?)", mmW, mmWTwin);
    PXG_CHECK_MSG(mmOther == 0, "p90: %zu contact/moreData/keep words differ bitwise", mmOther);

    for (PxgDevicePtr pfree : { dBlobs, dCmIn, dShapes, dCache, dMf, dKeep })
        shim.memFree(pfree);
}

} // namespace
