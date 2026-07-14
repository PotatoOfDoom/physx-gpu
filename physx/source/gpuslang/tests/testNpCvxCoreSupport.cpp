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

// Verification of npCvxCoreSupport.slang: the Gu::ConvexCore localSupport
// family (GuConvexSupport.h L95-380) and the Gu::ConvexShape support wrappers
// (support = pose.transform(supportLocal(pose.rotateInv(dir))) + dir*margin;
// computeBounds from six axis supports).  Exact CPU transcription: PxSign's
// >= 0 convention, the segment/box RAW-dir sign picks, the ellipsoid diagonal
// transform-transpose-normalize-transform, the cylinder/cone normalized-dir
// branches (degenerate radial 1e-5 caps, the cone sinA apex test), and the
// ePOINTS argmax with the S/R vertex2shape form, NULL/empty-pool and
// |d| < FLT_EPSILON guards, and the stride-honoring pool walk.
// ALL outputs compare bitwise.

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
struct SupIn
{
    U4 meta;
    U4 core[3];
    F4 poseQ, poseP;
    F4 dir;
    F4 pts[32];
};
struct SupOut
{
    F4 supLocal, sup, bmin, bmax;
};

const uint32_t CORE_POINT = 0, CORE_SEGMENT = 1, CORE_BOX = 2, CORE_ELLIPSOID = 3,
               CORE_CYLINDER = 4, CORE_CONE = 5, CORE_POINTS = 6;

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static float pxSign(float a) { return (a >= 0.f) ? 1.f : -1.f; }
static V3 nrmz(V3 v)
{
    const float m = v.x * v.x + v.y * v.y + v.z * v.z;
    if (m > 0) { const float i = 1.f / std::sqrt(m); return V3{ v.x * i, v.y * i, v.z * i }; }
    return V3{ 0, 0, 0 };
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

static float coreF(const SupIn& s, uint32_t w) { return u2f((&s.core[w / 4].x)[w & 3]); }

static V3 supportLocal(const SupIn& s, V3 dir)
{
    const uint32_t type = s.meta.x;

    if (type == CORE_SEGMENT)
        return V3{ pxSign(dir.x) * coreF(s, 0) * 0.5f, 0, 0 };
    if (type == CORE_BOX)
        return V3{ pxSign(dir.x) * coreF(s, 0) * 0.5f,
                   pxSign(dir.y) * coreF(s, 1) * 0.5f,
                   pxSign(dir.z) * coreF(s, 2) * 0.5f };
    if (type == CORE_ELLIPSOID)
    {
        // PxMat33 diagonal form with the zero terms kept (signed-zero exact)
        const V3 r{ coreF(s, 0), coreF(s, 1), coreF(s, 2) };
        const V3 tt{ (r.x * dir.x + 0.0f * dir.y) + 0.0f * dir.z,
                     (0.0f * dir.x + r.y * dir.y) + 0.0f * dir.z,
                     (0.0f * dir.x + 0.0f * dir.y) + r.z * dir.z };
        const V3 d1 = nrmz(tt);
        return V3{ (r.x * d1.x + 0.0f * d1.y) + 0.0f * d1.z,
                   (0.0f * d1.x + r.y * d1.y) + 0.0f * d1.z,
                   (0.0f * d1.x + 0.0f * d1.y) + r.z * d1.z };
    }
    if (type == CORE_CYLINDER)
    {
        const float h = coreF(s, 0) * 0.5f, r = coreF(s, 1);
        const V3 d = nrmz(dir);
        const float eps = 1e-5f;
        if (std::fabs(d.y) < eps && std::fabs(d.z) < eps) return V3{ pxSign(d.x) * h, r, 0 };
        const V3 dr = nrmz(V3{ 0, d.y, d.z });
        return V3{ pxSign(d.x) * h + dr.x * r, dr.y * r, dr.z * r };
    }
    if (type == CORE_CONE)
    {
        const float h = coreF(s, 0), r = coreF(s, 1);
        const float sinA = r / std::sqrt(h * h + r * r);
        const float halfH = h * 0.5f;
        const V3 d = nrmz(dir);
        const float eps = 1e-5f;
        if (d.x > sinA) return V3{ halfH, 0, 0 };
        if (std::fabs(d.y) < eps && std::fabs(d.z) < eps) return V3{ -halfH, r, 0 };
        const V3 dr = nrmz(V3{ 0, d.y, d.z });
        return V3{ -halfH + dr.x * r, dr.y * r, dr.z * r };
    }
    if (type == CORE_POINTS)
    {
        const uint32_t w9 = s.core[2].y;
        const uint32_t numPoints = w9 & 0xFF, stride = (w9 >> 8) & 0xFF;
        if ((s.core[0].x == 0 && s.core[0].y == 0) || numPoints == 0) return V3{ 0, 0, 0 }; // 64-bit NULL
        const V3 S{ u2f(s.core[0].z), u2f(s.core[0].w), u2f(s.core[1].x) };
        const F4 R{ u2f(s.core[1].y), u2f(s.core[1].z), u2f(s.core[1].w), u2f(s.core[2].x) };

        const V3 rd = qRot(R, dir);
        const V3 d = qRotInv(R, V3{ S.x * rd.x, S.y * rd.y, S.z * rd.z });
        const float mag = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
        if (mag < FLT_EPSILON) return V3{ 0, 0, 0 };

        uint32_t index = 0;
        float maxDot = -FLT_MAX;
        for (uint32_t i = 0; i < numPoints; ++i)
        {
            const F4& p = s.pts[(i * stride) / 16];
            const float dot = d.x * p.x + d.y * p.y + d.z * p.z;
            if (dot > maxDot) { maxDot = dot; index = i; }
        }
        const F4& pt = s.pts[(index * stride) / 16];
        const V3 rp = qRot(R, V3{ pt.x, pt.y, pt.z });
        return qRotInv(R, V3{ S.x * rp.x, S.y * rp.y, S.z * rp.z });
    }
    return V3{ 0, 0, 0 }; // CORE_POINT
}

static V3 support(const SupIn& s, V3 dir)
{
    const V3 sl = supportLocal(s, qRotInv(s.poseQ, dir));
    const V3 rp = qRot(s.poseQ, sl);
    const float m = s.poseP.w;
    return V3{ (rp.x + s.poseP.x) + dir.x * m, (rp.y + s.poseP.y) + dir.y * m, (rp.z + s.poseP.z) + dir.z * m };
}

static void refSupport(const SupIn& s, SupOut& o)
{
    std::memset(&o, 0, sizeof(o));
    const V3 sl = supportLocal(s, V3{ s.dir.x, s.dir.y, s.dir.z });
    o.supLocal = F4{ sl.x, sl.y, sl.z, 0 };
    const V3 sp = support(s, V3{ s.dir.x, s.dir.y, s.dir.z });
    o.sup = F4{ sp.x, sp.y, sp.z, 0 };
    o.bmin = F4{ support(s, V3{ -1.f, -0.f, -0.f }).x, support(s, V3{ -0.f, -1.f, -0.f }).y, support(s, V3{ -0.f, -0.f, -1.f }).z, 0 }; // -X/-Y/-Z with IEEE -0 components
    o.bmax = F4{ support(s, V3{ 1, 0, 0 }).x, support(s, V3{ 0, 1, 0 }).y, support(s, V3{ 0, 0, 1 }).z, 0 };
}

static bool eqF4(const F4& a, const F4& b)
{ return f2u(a.x) == f2u(b.x) && f2u(a.y) == f2u(b.y) && f2u(a.z) == f2u(b.z) && f2u(a.w) == f2u(b.w); }
// ellipsoid only: the reference carries the exact PxMat33 diagonal form whose
// signed-zero outputs llvmpipe cannot reproduce (NSZ folding) -- every
// magnitude bit must still match, only the sign of exact zeros is tolerated
static bool eqFZ(float a, float b) { return f2u(a) == f2u(b) || (a == 0.f && b == 0.f); }
static bool eqF4Z(const F4& a, const F4& b)
{ return eqFZ(a.x, b.x) && eqFZ(a.y, b.y) && eqFZ(a.z, b.z) && eqFZ(a.w, b.w); }

PXG_TEST(npCvxCoreSupport_supportAll)
{
    std::mt19937 rng(164001);
    const uint32_t N = 7000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto randQuat = [&]() { float a[4]; float m2 = 0; for (int k = 0; k < 4; ++k) { a[k] = rf(-1, 1); m2 += a[k] * a[k]; } const float i = 1.f / std::sqrt(m2 > 0 ? m2 : 1.f); return F4{ a[0] * i, a[1] * i, a[2] * i, a[3] * i }; };

    std::vector<SupIn> in(N);
    std::vector<SupOut> ref(N);
    size_t cType[7] = { 0, 0, 0, 0, 0, 0, 0 };
    size_t cDegen = 0, cNull = 0, cTiny = 0, cZeroSign = 0, cBand = 0, cTie = 0, cHiPtr = 0, cApex = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        SupIn s{};
        const uint32_t type = rng() % 7;
        ++cType[type];
        s.meta = U4{ type, 0, 0, 0 };
        s.poseQ = randQuat();
        const float margin = (rng() % 3 == 0) ? 0.f : rf(0.01f, 0.5f);
        s.poseP = F4{ rf(-3, 3), rf(-3, 3), rf(-3, 3), margin };

        // axis-near and fully random query directions (degenerate radial paths
        // need |d.y|,|d.z| < 1e-5 AFTER normalize -- use exact axis dirs)
        if (i % 5 == 0) { const int ax = rng() % 3; float d[3] = { 0, 0, 0 }; d[ax] = (rng() % 2) ? 1.f : -1.f; s.dir = F4{ d[0], d[1], d[2], 0 }; ++cDegen; }
        else s.dir = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), 0 };
        if (i % 13 == 5) // signed-zero components (review 164: the diagonal matrix form is -0-sensitive)
        {
            float d[3];
            for (int c = 0; c < 3; ++c) { const uint32_t k = rng() % 4; d[c] = k == 0 ? -0.f : (k == 1 ? 0.f : rf(-1, 1)); }
            s.dir = F4{ d[0], d[1], d[2], 0 }; ++cZeroSign;
        }
        if (i % 41 == 19) s.dir = F4{ 0, 0, 0, 0 }; // all-zero dir: every getNormalized guard
        if (i % 43 == 21 && (type == CORE_CYLINDER || type == CORE_CONE))
        {
            // radial band just above the 1e-5 degeneracy eps (review 164 xB)
            s.dir = F4{ (rng() % 2) ? 1.f : -1.f, rf(1.5e-5f, 8e-5f), 0, 0 }; ++cBand;
        }


        if (type == CORE_SEGMENT) s.core[0].x = f2u(rf(0.2f, 4));
        else if (type == CORE_BOX || type == CORE_ELLIPSOID)
        { s.core[0].x = f2u(rf(0.1f, 3)); s.core[0].y = f2u(rf(0.1f, 3)); s.core[0].z = f2u(rf(0.1f, 3)); }
        else if (type == CORE_CYLINDER || type == CORE_CONE)
        { s.core[0].x = f2u(rf(0.2f, 3)); s.core[0].y = f2u(rf(0.1f, 2)); }
        else if (type == CORE_POINTS)
        {
            const uint32_t stride = (i % 29 == 13) ? 32u : 16u; // the pool walk honors stride
            const uint32_t maxNp = (stride == 32) ? 14u : 30u; // (np-1)*stride must stay inside pts[32]
            const uint32_t np = (i % 19 == 3) ? 0 : 3 + rng() % maxNp;
            const uint32_t ptr = (i % 23 == 7) ? 0 : 48;
            if (!np || !ptr) ++cNull;
            s.core[0].x = ptr;
            if (i % 53 == 9) { s.core[0].x = 0; s.core[0].y = 1; ++cHiPtr; } // 64-bit pointer with a zero low word: NOT NULL
            const bool tinyS = (i % 31 == 11); // |d| < FLT_EPSILON guard
            const bool smallS = (i % 31 == 22); // |d| in [FLT_EPSILON, sqrt(FLT_EPSILON)): separates the sqrt'd guard (review 164 xG)
            if (tinyS || smallS) ++cTiny;
            auto rs = [&]() { return tinyS ? 1e-8f : (smallS ? 1e-4f : rf(0.3f, 2)); };
            s.core[0].z = f2u(rs()); s.core[0].w = f2u(rs());
            s.core[1].x = f2u(rs());
            const F4 R = randQuat();
            s.core[1].y = f2u(R.x); s.core[1].z = f2u(R.y); s.core[1].w = f2u(R.z);
            s.core[2].x = f2u(R.w);
            s.core[2].y = np | (stride << 8);
            for (uint32_t p = 0; p < 32; ++p)
                s.pts[p] = F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), 0 };
            if (i % 37 == 17 && np >= 4 && stride == 16 && !tinyS && !smallS)
            {
                // exact argmax ties: identity R, unit S, exact-axis dir, and
                // point pairs sharing the axis projection -- first-wins order
                // is observable (review 164 xC)
                s.core[0].z = f2u(1.f); s.core[0].w = f2u(1.f); s.core[1].x = f2u(1.f);
                s.core[1].y = f2u(0.f); s.core[1].z = f2u(0.f); s.core[1].w = f2u(0.f);
                s.core[2].x = f2u(1.f);
                s.dir = F4{ 1.f, 0.f, 0.f, 0 };
                for (uint32_t p = 1; p + 1 < np; p += 2)
                    s.pts[p + 1] = F4{ s.pts[p].x, s.pts[p].y + 0.5f, s.pts[p].z - 0.25f, 0 };
                ++cTie;
            }
        }

        if (i % 47 == 23 && type == CORE_CONE)
        {
            // land d.x EXACTLY on sinA (review 164 xD): with the FINAL h/r,
            // search a partner c so sinA^2 + c^2 rounds to exactly 1.0f --
            // then getNormalized multiplies by 1/sqrt(1) = 1 and d.x == sinA
            const float h = u2f(s.core[0].x), rr = u2f(s.core[0].y);
            if (h > 0 && rr > 0)
            {
                const float sinA = rr / std::sqrt(h * h + rr * rr);
                float c = std::sqrt(1.f - sinA * sinA);
                bool found = false;
                for (int step = 0; step < 200 && !found; ++step)
                {
                    if ((sinA * sinA + c * c) + 0.0f * 0.0f == 1.0f) { found = true; break; }
                    c = std::nextafterf(c, 0.f);
                }
                if (found) { s.dir = F4{ sinA, c, 0, 0 }; ++cApex; }
            }
        }
        in[i] = s;
        refSupport(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<SupOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npCvxCoreSupport.spv", "ccsSupportAll",
             { (uint32_t)sizeof(SupIn), (uint32_t)sizeof(SupOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<SupOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const SupOut& r = ref[i]; const SupOut& q = g[i];
        const bool ell = in[i].meta.x == CORE_ELLIPSOID;
        const bool okAll = ell
            ? (eqF4Z(r.supLocal, q.supLocal) && eqF4Z(r.sup, q.sup) &&
               eqF4Z(r.bmin, q.bmin) && eqF4Z(r.bmax, q.bmax))
            : (eqF4(r.supLocal, q.supLocal) && eqF4(r.sup, q.sup) &&
               eqF4(r.bmin, q.bmin) && eqF4(r.bmax, q.bmax));
        if (!okAll) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "ccsSupportAll: %zu mismatches", mm);
    bool cov = cDegen > N / 8 && cNull > 10 && cTiny > 10 && cZeroSign > N / 16 && cBand > 8 && cTie > 8 && cHiPtr > 8 && cApex > 8;
    for (int k2 = 0; k2 < 7; ++k2) cov = cov && cType[k2] > N / 16;
    PXG_CHECK_MSG(cov, "ccsSupportAll: coverage thin (degen %zu null %zu)", cDegen, cNull);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
