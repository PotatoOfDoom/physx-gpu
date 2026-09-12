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

// Verification of npCvxBvhGlue.slang: the TrimeshBvh scalar glue from
// convexCoreCollision.cu -- scaleBounds (PxBounds3::transformFast over the
// inverse mesh scale), the traversal-callback margin (dimensions min
// element * 1e-4), the PxBounds3::intersects box test, and createConvex's
// scale-transformed vertices + PointsCore image.  Exact CPU transcription;
// ALL outputs compare bitwise.

#include "pxgTestFramework.h"

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };
struct V3 { float x, y, z; };

struct BvhIn
{
    F4 scale, scaleRot;
    F4 bmin, bmax;
    F4 pmin, pmax;
    F4 v0, v1, v2;
    F4 v3;      // the fourth tetrahedron vertex (P9.73)
    F4 dir;     // support direction for the tet support query (P9.73)
};
struct BvhOut
{
    F4 smin, smax;
    F4 marginOut;
    F4 verts[3];
    U4 core[3];
    F4 prim;
    F4 tetVerts[4];
    U4 tetCore[3];
    F4 tetSup;
};

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static V3 addv(V3 a, V3 b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 subv(V3 a, V3 b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 mulv(V3 a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; }

static void quatToMat33(F4 q, V3& c0, V3& c1, V3& c2)
{
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    const float x2 = x + x, y2 = y + y, z2 = z + z;
    const float xx = x2 * x, yy = y2 * y, zz = z2 * z;
    const float xy = x2 * y, xz = x2 * z, xw = x2 * w;
    const float yz = y2 * z, yw = y2 * w, zw = z2 * w;
    c0 = { 1.0f - yy - zz, xy + zw, xz - yw };
    c1 = { xy - zw, 1.0f - xx - zz, yz + xw };
    c2 = { xz + yw, yz - xw, 1.0f - xx - yy };
}
static void meshScaleMat(V3 scale, F4 rot, V3& m0, V3& m1, V3& m2)
{
    V3 r0, r1, r2; quatToMat33(rot, r0, r1, r2);
    const V3 t0 = { r0.x * scale.x, r1.x * scale.x, r2.x * scale.x };
    const V3 t1 = { r0.y * scale.y, r1.y * scale.y, r2.y * scale.y };
    const V3 t2 = { r0.z * scale.z, r1.z * scale.z, r2.z * scale.z };
    auto lc = [](V3 a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; };
    auto add = [](V3 a, V3 b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; };
    m0 = add(add(lc(t0, r0.x), lc(t1, r0.y)), lc(t2, r0.z));
    m1 = add(add(lc(t0, r1.x), lc(t1, r1.y)), lc(t2, r1.z));
    m2 = add(add(lc(t0, r2.x), lc(t1, r2.y)), lc(t2, r2.z));
}

static V3 matT(V3 m0, V3 m1, V3 m2, V3 v)
{ return addv(addv(mulv(m0, v.x), mulv(m1, v.y)), mulv(m2, v.z)); }

static void refBvh(const BvhIn& s, BvhOut& o)
{
    std::memset(&o, 0, sizeof(o));
    const V3 bmin{ s.bmin.x, s.bmin.y, s.bmin.z }, bmax{ s.bmax.x, s.bmax.y, s.bmax.z };

    const V3 invScale{ 1.0f / s.scale.x, 1.0f / s.scale.y, 1.0f / s.scale.z };
    V3 im0, im1, im2;
    meshScaleMat(invScale, s.scaleRot, im0, im1, im2);
    const V3 center = mulv(addv(bmin, bmax), 0.5f);
    const V3 extents = mulv(subv(bmax, bmin), 0.5f);
    const V3 mc = matT(im0, im1, im2, center);
    const V3 c0 = mulv(im0, extents.x), c1 = mulv(im1, extents.y), c2 = mulv(im2, extents.z);
    const V3 w{ std::fabs(c0.x) + std::fabs(c1.x) + std::fabs(c2.x),
                std::fabs(c0.y) + std::fabs(c1.y) + std::fabs(c2.y),
                std::fabs(c0.z) + std::fabs(c1.z) + std::fabs(c2.z) };
    const V3 smin = subv(mc, w), smax = addv(mc, w);

    const V3 dims = subv(smax, smin);
    const float margin = std::fmin(dims.x, std::fmin(dims.y, dims.z)) * 0.0001f;

    const V3 pmin{ s.pmin.x, s.pmin.y, s.pmin.z }, pmax{ s.pmax.x, s.pmax.y, s.pmax.z };
    const bool isect = !(pmin.x > smax.x || smin.x > pmax.x || pmin.y > smax.y || smin.y > pmax.y ||
                         pmin.z > smax.z || smin.z > pmax.z);

    V3 m0, m1, m2;
    meshScaleMat(V3{ s.scale.x, s.scale.y, s.scale.z }, s.scaleRot, m0, m1, m2);
    const V3 tv0 = matT(m0, m1, m2, V3{ s.v0.x, s.v0.y, s.v0.z });
    const V3 tv1 = matT(m0, m1, m2, V3{ s.v1.x, s.v1.y, s.v1.z });
    const V3 tv2 = matT(m0, m1, m2, V3{ s.v2.x, s.v2.y, s.v2.z });

    o.smin = F4{ smin.x, smin.y, smin.z, 0 };
    o.smax = F4{ smax.x, smax.y, smax.z, 0 };
    o.marginOut = F4{ margin, isect ? 1.0f : 0.0f, 0, 0 };
    o.verts[0] = F4{ tv0.x, tv0.y, tv0.z, 1.0f };
    o.verts[1] = F4{ tv1.x, tv1.y, tv1.z, 1.0f };
    o.verts[2] = F4{ tv2.x, tv2.y, tv2.z, 1.0f };
    o.core[0] = U4{ 0x30u, 0u, f2u(1.0f), f2u(1.0f) };
    o.core[1] = U4{ f2u(1.0f), f2u(0.0f), f2u(0.0f), f2u(0.0f) };
    o.core[2] = U4{ f2u(1.0f), 3u | (16u << 8), 0u, 0u };
    uint32_t six = 6;
    float sixf; std::memcpy(&sixf, &six, 4);
    o.prim = F4{ margin, sixf, 0, 0 };

    // ---- the TETRAHEDRON arm (P9.73) --------------------------------------
    // createConvexTet takes the vertices RAW: no mesh-scale transform anywhere
    // (convexCoreCollision.cu:1075-1079).  So unlike the triangle arm above, the
    // four verts come straight through.
    o.tetVerts[0] = F4{ s.v0.x, s.v0.y, s.v0.z, 1.0f };
    o.tetVerts[1] = F4{ s.v1.x, s.v1.y, s.v1.z, 1.0f };
    o.tetVerts[2] = F4{ s.v2.x, s.v2.y, s.v2.z, 1.0f };
    o.tetVerts[3] = F4{ s.v3.x, s.v3.y, s.v3.z, 1.0f };
    o.tetCore[0] = U4{ 0x30u, 0u, f2u(1.0f), f2u(1.0f) };
    o.tetCore[1] = U4{ f2u(1.0f), f2u(0.0f), f2u(0.0f), f2u(0.0f) };
    o.tetCore[2] = U4{ f2u(1.0f), 4u | (16u << 8), 0u, 0u };

    // supportLocalQ's CORE_POINTS arm, written from its SEMANTICS and not copied
    // from the shader.  With S = (1,1,1) and R = identity the whole prologue --
    //     d = R.rotateInv(S.multiply(R.rotate(dir)))
    // -- collapses to d = dir, and the epilogue collapses the same way, so what is
    // left is: pick the vertex maximising dot(dir, v_i) over i in [0, numPoints).
    //
    // THE TIE-BREAK IS LOAD-BEARING.  The comparison is strictly `>` starting from
    // -FLT_MAX with index 0, so the LOWEST index wins a tie.  A mirror using >= would
    // disagree on every degenerate tetrahedron.
    //
    // The mag < FLT_EPS early-out returns (0,0,0); the test feeds unit directions, so
    // it is not reachable here and is not mirrored.
    {
        const V3 d{ s.dir.x, s.dir.y, s.dir.z };
        const F4* v[4] = { &s.v0, &s.v1, &s.v2, &s.v3 };
        uint32_t best = 0;
        float maxDot = -3.402823466e+38f;
        for (uint32_t k = 0; k < 4; ++k)
        {
            const float dp = d.x * v[k]->x + d.y * v[k]->y + d.z * v[k]->z;
            if (dp > maxDot) { maxDot = dp; best = k; }
        }
        o.tetSup = F4{ v[best]->x, v[best]->y, v[best]->z, 0.0f };
    }
}

PXG_TEST(npCvxBvhGlue_glue)
{
    std::mt19937 rng(172001);
    const uint32_t N = 8000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto randQuat = [&]() { float a[4]; float m2 = 0; for (int k = 0; k < 4; ++k) { a[k] = rf(-1, 1); m2 += a[k] * a[k]; } const float i = 1.f / std::sqrt(m2 > 0 ? m2 : 1.f); return F4{ a[0] * i, a[1] * i, a[2] * i, a[3] * i }; };

    std::vector<BvhIn> in(N);
    std::vector<BvhOut> ref(N);
    size_t cIdent = 0, cNeg = 0, cHit = 0, cMiss = 0, cTouch = 0, cV3 = 0, cTie = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        BvhIn s{};
        // scales: identity / uniform / non-uniform / NEGATIVE components
        if (i % 5 == 0) { s.scale = F4{ 1, 1, 1, 0 }; s.scaleRot = F4{ 0, 0, 0, 1 }; ++cIdent; }
        else
        {
            s.scale = F4{ rf(0.2f, 3), rf(0.2f, 3), rf(0.2f, 3), 0 };
            if (i % 7 == 3) { (&s.scale.x)[rng() % 3] *= -1.f; ++cNeg; }
            s.scaleRot = (i % 3 == 0) ? F4{ 0, 0, 0, 1 } : randQuat();
        }
        const V3 c{ rf(-5, 5), rf(-5, 5), rf(-5, 5) };
        const V3 e{ rf(0.05f, 2), rf(0.05f, 2), rf(0.05f, 2) };
        s.bmin = F4{ c.x - e.x, c.y - e.y, c.z - e.z, 0 };
        s.bmax = F4{ c.x + e.x, c.y + e.y, c.z + e.z, 0 };
        // node boxes: overlapping / separated / exactly touching faces
        const float off = (i % 4 == 1) ? rf(6.f, 12.f) : rf(-1.f, 1.f);
        s.pmin = F4{ c.x + off - 1, c.y - 1, c.z - 1, 0 };
        s.pmax = F4{ c.x + off + 1, c.y + 1, c.z + 1, 0 };
        if (i % 17 == 8)
        {
            // exact face touch in scaled space: pmin.x == smax.x (the > is
            // strict, touching still intersects)
            BvhOut tmp; refBvh(s, tmp);
            s.pmin.x = tmp.smax.x;
            s.pmax.x = s.pmin.x + 2;
            ++cTouch;
        }
        for (F4* v : { &s.v0, &s.v1, &s.v2, &s.v3 })
            *v = F4{ rf(-3, 3), rf(-3, 3), rf(-3, 3), 0 };

        // A UNIT support direction, so supportLocalQ's mag < FLT_EPS early-out is
        // never taken and the argmax is always the thing under test.
        {
            V3 d{ rf(-1, 1), rf(-1, 1), rf(-1, 1) };
            float m2 = d.x * d.x + d.y * d.y + d.z * d.z;
            if (m2 < 1e-6f) d = V3{ 1, 0, 0 }, m2 = 1.f;
            const float inv = 1.f / std::sqrt(m2);
            s.dir = F4{ d.x * inv, d.y * inv, d.z * inv, 0 };
        }

        // EXACT TIES IN THE SUPPORT DOT, constructed on purpose (P9.73).  The mirror's
        // comment calls the strict `>` tie-break load-bearing, and the first sweep
        // refuted that claim: `support-tie-ge` SURVIVED, because two random vertices
        // never produce an exactly equal dot.  A tie needs two DISTINCT vertices at
        // the same projection, i.e. an edge PERPENDICULAR to the query direction --
        // which is not exotic at all, it is what a flat face against a flat floor
        // looks like.  So build one: move v1 off v0 along a direction perpendicular to
        // dir, and the two project identically while staying different points.
        if (i % 6 == 2)
        {
            const V3 d{ s.dir.x, s.dir.y, s.dir.z };
            // any vector not parallel to d, made perpendicular by Gram-Schmidt
            V3 t{ 1, 0, 0 };
            if (std::fabs(d.x) > 0.9f) t = V3{ 0, 1, 0 };
            const float dt = d.x * t.x + d.y * t.y + d.z * t.z;
            const V3 perp{ t.x - d.x * dt, t.y - d.y * dt, t.z - d.z * dt };
            const float pm = std::sqrt(perp.x * perp.x + perp.y * perp.y + perp.z * perp.z);
            const float k = rf(0.5f, 2.0f) / (pm > 0 ? pm : 1.f);
            s.v1 = F4{ s.v0.x + perp.x * k, s.v0.y + perp.y * k, s.v0.z + perp.z * k, 0 };
            ++cTie;
        }

        in[i] = s;
        refBvh(s, ref[i]);
        if (ref[i].marginOut.y != 0) ++cHit; else ++cMiss;
        // NON-TRIVIALITY GUARD.  The support query only discriminates the fourth
        // vertex on the cases where v3 IS the argmax; on the other three quarters a
        // port that dropped it would agree anyway.  Count them, and fail below if the
        // count is too low to mean anything.
        if (ref[i].tetSup.x == s.v3.x && ref[i].tetSup.y == s.v3.y &&
            ref[i].tetSup.z == s.v3.z) ++cV3;
    }

    std::printf("[ info ] bvh cov: ident %zu neg %zu hit %zu miss %zu touch %zu v3wins %zu tie %zu\n",
                cIdent, cNeg, cHit, cMiss, cTouch, cV3, cTie);
    PXG_CHECK_MSG(cV3 > N / 8, "the fourth tet vertex must win the support query often "
                  "enough to discriminate (v3 won %zu of %u)", cV3, N);

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<BvhOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npCvxBvhGlue.spv", "cbgGlue",
             { (uint32_t)sizeof(BvhIn), (uint32_t)sizeof(BvhOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<BvhOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const BvhOut& r = ref[i]; const BvhOut& q = g[i];
        const bool ok = std::memcmp(&r, &q, sizeof(BvhOut)) == 0;
        if (!ok && mm < 5)
            std::printf("MM %u sminR (%g %g %g) sminG (%g %g %g) mR %.9g mG %.9g iR %g iG %g\n",
                        i, r.smin.x, r.smin.y, r.smin.z, q.smin.x, q.smin.y, q.smin.z,
                        r.marginOut.x, q.marginOut.x, r.marginOut.y, q.marginOut.y);
        if (!ok) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "cbgGlue: %zu mismatches", mm);
    PXG_CHECK_MSG(cIdent > N / 8 && cNeg > N / 16 && cHit > N / 8 && cMiss > N / 8 && cTouch > 100,
                  "cbgGlue: coverage thin");

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
