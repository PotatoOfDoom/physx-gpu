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

// Verification of npCvxStage2Setup.slang: convexConvexNphase_stage2Kernel_core
// setup/teardown glue (shape flip, synthetic vertices, margins, blob header
// decode, the shape1-gated minScale quirk, contactDistance + expandedMargin,
// replaceBreakingThreshold, counts unpack, invalidation-bit writeback).
// Reference transcribes the CUDA; integers exact, floats to tight tolerance.

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
struct S2In { U4 meta0, meta1; F4 scale0, rot0, scale1, rot1, extents0, extents1, t0q, t0p, t1q, t1p; U4 misc; };
struct S2Out { F4 rotOut0, rotOut1, tBq, tBp, aToBq, aToBp, verts0[2]; U4 counts0, counts1; F4 params; U4 result, materials; };

static inline V3 v3(const F4& f) { return V3{ f.x, f.y, f.z }; }
static inline V3 sub(const V3& a, const V3& b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static inline V3 mul(const V3& a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; }
static inline float dot3(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 unit(const V3& a) { float l = std::sqrt(dot3(a, a)); return l > 0 ? mul(a, 1.f / l) : a; }
static V3 qRotInv(const F4& q, const V3& v) { const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z, w2 = q.w * q.w - 0.5f, d2 = q.x * vx + q.y * vy + q.z * vz; return V3{ vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * d2, vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * d2, vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * d2 }; }
static F4 qMul(const F4& a, const F4& b)
{
    return F4{ a.w * b.x + b.w * a.x + a.y * b.z - a.z * b.y,
               a.w * b.y + b.w * a.y + a.z * b.x - a.x * b.z,
               a.w * b.z + b.w * a.z + a.x * b.y - a.y * b.x,
               a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z };
}
static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float pcmMargin(const V3& e, float tol) { return std::fmin(std::fmin(std::fmin(e.x, e.y), e.z) * 0.25f, tol * 0.05f); }

static void refSetup(const S2In& s, S2Out& o)
{
    uint32_t type0 = s.meta0.x, type1 = s.meta1.x, hdr0 = s.meta0.z, hdr1 = s.meta1.z;
    uint32_t mat0 = s.meta0.y, mat1 = s.meta1.y;
    F4 rot0 = s.rot0, rot1 = s.rot1;
    V3 sc0 = v3(s.scale0), sc1 = v3(s.scale1);
    F4 e0 = s.extents0, e1 = s.extents1;
    F4 t0q = s.t0q, t1q = s.t1q;
    V3 t0p = v3(s.t0p), t1p = v3(s.t1p);
    float cd0 = s.t0p.w, cd1 = s.t1p.w;
    float tol; std::memcpy(&tol, &s.misc.w, 4);

    const bool flip = type1 < type0;
    if (flip)
    {
        std::swap(type0, type1); std::swap(hdr0, hdr1); std::swap(sc0, sc1);
        std::swap(mat0, mat1); std::swap(rot0, rot1);
        std::swap(e0, e1); std::swap(t0q, t1q); std::swap(t0p, t1p); std::swap(cd0, cd1);
    }

    const F4 c{ -t1q.x, -t1q.y, -t1q.z, t1q.w };
    const F4 aq = qMul(c, t0q);
    const V3 ap = qRotInv(t1q, sub(t0p, t1p));

    o.verts0[0] = F4{ 0, 0, 0, 0 }; o.verts0[1] = F4{ 0, 0, 0, 0 };
    float em, cm0, is0;
    uint32_t nv0, np0, ne0;
    if (type0 == 0)
    {
        em = sc0.x; cm0 = sc0.x; nv0 = 1; np0 = 0; ne0 = 0; is0 = 0.f;
    }
    else if (type0 == 2)
    {
        o.verts0[0] = F4{ 1, 0, 0, 0 }; o.verts0[1] = F4{ -1, 0, 0, 0 };
        em = sc0.y; cm0 = sc0.y; nv0 = 2; np0 = 0; ne0 = 0; is0 = 0.f;
    }
    else
    {
        em = 0.f;
        cm0 = pcmMargin(V3{ e0.x * sc0.x, e0.y * sc0.y, e0.z * sc0.z }, tol);
        nv0 = (hdr0 & 0xFFFF) >> 8; np0 = hdr0 & 0xFF; ne0 = hdr0 >> 16;
        float ms0 = 1.f;
        if (type1 == 3) ms0 = std::fmin(sc0.z, std::fmin(sc0.x, sc0.y)); // shape1-gated quirk
        is0 = e0.w * ms0;
    }
    const uint32_t nv1 = (hdr1 & 0xFFFF) >> 8, np1 = hdr1 & 0xFF, ne1 = hdr1 >> 16;
    float ms1 = 1.f;
    if (type1 == 3) ms1 = std::fmin(sc1.z, std::fmin(sc1.x, sc1.y));
    const float is1 = e1.w * ms1;

    const float cm1 = pcmMargin(V3{ e1.x * sc1.x, e1.y * sc1.y, e1.z * sc1.z }, tol);
    const float rbt = std::fmin(cm0, cm1) * 0.05f;
    const float cd = cd0 + cd1 + em;

    const uint32_t cw = s.misc.x;
    const uint32_t writeback = s.misc.z | (s.misc.y ? 0u : 0x80000000u);

    o.rotOut0 = rot0; o.rotOut1 = rot1;
    o.tBq = t1q; o.tBp = F4{ t1p.x, t1p.y, t1p.z, 0 };
    o.materials = U4{ mat0, mat1, 0, 0 };
    o.aToBq = aq;
    o.aToBp = F4{ ap.x, ap.y, ap.z, cd };
    o.counts0 = U4{ nv0, np0, ne0, type0 };
    o.counts1 = U4{ nv1, np1, ne1, type1 };
    o.params = F4{ is0, is1, rbt, em };
    o.result = U4{ cw & 0xFF, cw >> 8, writeback, flip ? 1u : 0u };
}

static bool eqF4(const F4& a, const F4& b) { return almostEqual(a.x, b.x, 1e-5f, 1e-6f) && almostEqual(a.y, b.y, 1e-5f, 1e-6f) && almostEqual(a.z, b.z, 1e-5f, 1e-6f) && almostEqual(a.w, b.w, 1e-5f, 1e-6f); }
static bool eqU4(const U4& a, const U4& b) { return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w; }

PXG_TEST(npCvxStage2Setup_setup)
{
    std::mt19937 rng(154001);
    const uint32_t N = 10000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto randQuat = [&]() { V3 ax = unit(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) }); const float a = rf(-3.1f, 3.1f); return F4{ ax.x * std::sin(a * .5f), ax.y * std::sin(a * .5f), ax.z * std::sin(a * .5f), std::cos(a * .5f) }; };

    std::vector<S2In> in(N);
    std::vector<S2Out> ref(N);
    size_t cSphere = 0, cCapsule = 0, cBox1 = 0, cFlip = 0, cNoInt = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        S2In s{};
        const uint32_t kinds[4] = { 0u, 2u, 3u, 5u }; // sphere, capsule, box, convex mesh
        const uint32_t ty0 = kinds[rng() % 4], ty1 = kinds[rng() % 4];
        if (ty0 == 0 || ty1 == 0) ++cSphere;
        if (ty0 == 2 || ty1 == 2) ++cCapsule;
        if (ty1 == 3 || (ty1 < ty0 && ty0 == 3)) ++cBox1;
        if (ty1 < ty0) ++cFlip;
        s.meta0 = U4{ ty0, rng() % 512, (rng() % 65536) << 16 | (rng() % 64) << 8 | (rng() % 64), 0 };
        s.meta1 = U4{ ty1, rng() % 512, (rng() % 65536) << 16 | (rng() % 64) << 8 | (rng() % 64), 0 };
        s.scale0 = F4{ rf(0.5f, 2), rf(0.5f, 2), rf(0.5f, 2), 0 };
        s.scale1 = F4{ rf(0.5f, 2), rf(0.5f, 2), rf(0.5f, 2), 0 };
        s.rot0 = randQuat(); s.rot1 = randQuat();
        s.extents0 = F4{ rf(0.3f, 2), rf(0.3f, 2), rf(0.3f, 2), rf(0.1f, 1) };
        s.extents1 = F4{ rf(0.3f, 2), rf(0.3f, 2), rf(0.3f, 2), rf(0.1f, 1) };
        s.t0q = randQuat(); s.t1q = randQuat();
        s.t0p = F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), rf(0.02f, 0.3f) };
        s.t1p = F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), rf(0.02f, 0.3f) };
        const uint32_t refreshed = rng() % 5, initial = rng() % 5;
        const uint32_t bInt = (i % 6 == 0) ? 0u : 1u;
        if (!bInt) ++cNoInt;
        s.misc = U4{ refreshed | (initial << 8), bInt, rng() % 5, f2u(rf(0.5f, 4)) };
        in[i] = s;
        refSetup(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<S2Out> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct S2Params { uint32_t numTests; } prm = { N };
    Kernel k("npCvxStage2Setup.spv", "cvxStage2Setup",
             { (uint32_t)sizeof(S2In), (uint32_t)sizeof(S2Out) }, sizeof(S2Params));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<S2Out>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (!eqF4(ref[i].rotOut0, g[i].rotOut0) || !eqF4(ref[i].rotOut1, g[i].rotOut1) ||
            !eqF4(ref[i].tBq, g[i].tBq) || !eqF4(ref[i].tBp, g[i].tBp) ||
            !eqU4(ref[i].materials, g[i].materials) ||
            !eqF4(ref[i].aToBq, g[i].aToBq) || !eqF4(ref[i].aToBp, g[i].aToBp) ||
            !eqF4(ref[i].verts0[0], g[i].verts0[0]) || !eqF4(ref[i].verts0[1], g[i].verts0[1]) ||
            !eqU4(ref[i].counts0, g[i].counts0) || !eqU4(ref[i].counts1, g[i].counts1) ||
            !eqF4(ref[i].params, g[i].params) || !eqU4(ref[i].result, g[i].result)) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "stage2Setup: %zu mismatches", mm);
    PXG_CHECK_MSG(cSphere > N / 10 && cCapsule > N / 10 && cBox1 > N / 10 && cFlip > N / 10 && cNoInt > N / 10,
                  "stage2Setup: coverage thin (sph %zu cap %zu box1 %zu flip %zu noInt %zu)",
                  cSphere, cCapsule, cBox1, cFlip, cNoInt);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
