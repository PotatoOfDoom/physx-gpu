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

// Verification of npCvxCoreSetup.slang: the convex-core kernel setup glue
// (convexCoreCollision.cu L66-171 + the kernel preps L475-480/L523-532).
// Exact CPU transcription: TestInput's contactDist sum and conditional
// restDist, checkTypes' reverse-match swap, makeConvexShape's five core
// mappings into the Gu::ConvexShape 40-byte coreData image (the eCONVEXCORE
// 24-byte memcpy from &scale.scale.x, the SegmentCore length*2, the BoxCore
// extents*2, the PointsCore pointer/S/R/numPoints/stride packing with
// numPoints = (blobHead.x >> 8) & 0xff), the PxPlane from getBasisVector0
// (exact PxQuat.h term order), and the convex-convex mid-point shift.
// ALL outputs compare bitwise.

#include "pxgTestFramework.h"

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };
struct CssIn
{
    F4 t0q, t0p, t1q, t1p;
    F4 s0ScaleS, s0ScaleR, s1ScaleS, s1ScaleR;
    U4 s0Info, s1Info;
    F4 dists;
    U4 misc;
};
struct CvxModel
{
    U4 meta;
    U4 core[3];
    F4 poseQ, poseP;
};
struct CssOut
{
    U4 flags;
    F4 dists;
    F4 aux;
    CvxModel c0, c1;
};

const uint32_t GEOM_SPHERE = 0, GEOM_PLANE = 1, GEOM_CAPSULE = 2, GEOM_BOX = 3,
               GEOM_CONVEXCORE = 4, GEOM_CONVEXMESH = 5;
const uint32_t CORE_POINT = 0, CORE_SEGMENT = 1, CORE_BOX = 2, CORE_POINTS = 6;

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

static CvxModel makeConvex(uint32_t type, uint32_t hullLow, uint32_t blobHeadX,
                           const F4& scS, const F4& scR, const F4& poseQ, const F4& poseP)
{
    CvxModel m{};
    m.meta.x = 0xFFFFFFFFu;
    float margin = 0.f;

    if (type == GEOM_CONVEXCORE)
    {
        m.meta.x = hullLow;
        m.core[0] = U4{ f2u(scS.x), f2u(scS.y), f2u(scS.z), f2u(scR.x) };
        m.core[1] = U4{ f2u(scR.y), f2u(scR.z), 0, 0 };
        margin = scR.w;
        m.meta.y = 1;
    }
    else if (type == GEOM_SPHERE)
    {
        m.meta.x = CORE_POINT;
        margin = scS.x;
        m.meta.y = 1;
    }
    else if (type == GEOM_CAPSULE)
    {
        m.meta.x = CORE_SEGMENT;
        m.core[0].x = f2u(scS.x * 2.0f);
        margin = scS.y;
        m.meta.y = 1;
    }
    else if (type == GEOM_BOX)
    {
        m.meta.x = CORE_BOX;
        m.core[0] = U4{ f2u(scS.x * 2.0f), f2u(scS.y * 2.0f), f2u(scS.z * 2.0f), 0 };
        margin = 0.f;
        m.meta.y = 1;
    }
    else if (type == GEOM_CONVEXMESH)
    {
        m.meta.x = CORE_POINTS;
        const uint32_t numPoints = (blobHeadX >> 8) & 0xFF;
        m.core[0] = U4{ 48, 0, f2u(scS.x), f2u(scS.y) };
        m.core[1] = U4{ f2u(scS.z), f2u(scR.x), f2u(scR.y), f2u(scR.z) };
        m.core[2] = U4{ f2u(scR.w), numPoints | (16u << 8), 0, 0 };
        margin = 0.f;
        m.meta.y = 1;
    }

    if (m.meta.y)
    {
        m.poseQ = poseQ;
        m.poseP = F4{ poseP.x, poseP.y, poseP.z, margin };
    }
    else
        m.poseP.w = margin;
    return m;
}

static void refPlanePrep(const CssIn& s, CssOut& o)
{
    std::memset(&o, 0, sizeof(o));
    uint32_t cr0 = s.misc.x, cr1 = s.misc.y;
    F4 t0q = s.t0q, t1q = s.t1q, t0p = s.t0p, t1p = s.t1p;
    U4 i0 = s.s0Info, i1 = s.s1Info;
    F4 sc0S = s.s0ScaleS, sc0R = s.s0ScaleR, sc1S = s.s1ScaleS, sc1R = s.s1ScaleR;
    const float contactDist = s.dists.x + s.dists.y;
    const float restDist = s.misc.z ? s.dists.z : 0.f;
    uint32_t swapped = 0;

    if (GEOM_PLANE == i1.x && GEOM_CONVEXCORE == i0.x)
    {
        std::swap(i0, i1); std::swap(sc0S, sc1S); std::swap(sc0R, sc1R);
        std::swap(cr0, cr1); std::swap(t0q, t1q); std::swap(t0p, t1p);
        swapped = 1;
    }

    // getBasisVector0 (PxQuat.h L260-265)
    const float x2 = t0q.x * 2.0f, w2 = t0q.w * 2.0f;
    const float nx = (t0q.w * w2) - 1.0f + t0q.x * x2;
    const float ny = (t0q.z * w2) + t0q.y * x2;
    const float nz = (-t0q.y * w2) + t0q.z * x2;
    const float d = -(t0p.x * nx + t0p.y * ny + t0p.z * nz);

    o.flags = U4{ swapped, cr0, cr1, 0 };
    o.dists = F4{ contactDist, restDist, 0, 0 };
    o.aux = F4{ nx, ny, nz, d };
    o.c1 = makeConvex(i1.x, i1.y, i1.z, sc1S, sc1R, t1q, t1p);
}

static void refConvexPrep(const CssIn& s, CssOut& o)
{
    std::memset(&o, 0, sizeof(o));
    const float contactDist = s.dists.x + s.dists.y;
    const float restDist = s.misc.z ? s.dists.z : 0.f;

    const float sx = (s.t0p.x + s.t1p.x) * 0.5f;
    const float sy = (s.t0p.y + s.t1p.y) * 0.5f;
    const float sz = (s.t0p.z + s.t1p.z) * 0.5f;
    const F4 p0{ s.t0p.x - sx, s.t0p.y - sy, s.t0p.z - sz, 0 };
    const F4 p1{ s.t1p.x - sx, s.t1p.y - sy, s.t1p.z - sz, 0 };

    o.flags = U4{ 0, s.misc.x, s.misc.y, 0 };
    o.dists = F4{ contactDist, restDist, 0, 0 };
    o.aux = F4{ sx, sy, sz, 0 };
    o.c0 = makeConvex(s.s0Info.x, s.s0Info.y, s.s0Info.z, s.s0ScaleS, s.s0ScaleR, s.t0q, p0);
    o.c1 = makeConvex(s.s1Info.x, s.s1Info.y, s.s1Info.z, s.s1ScaleS, s.s1ScaleR, s.t1q, p1);
}

static bool eqU4(const U4& a, const U4& b) { return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w; }
static bool eqF4(const F4& a, const F4& b)
{ return f2u(a.x) == f2u(b.x) && f2u(a.y) == f2u(b.y) && f2u(a.z) == f2u(b.z) && f2u(a.w) == f2u(b.w); }
static bool eqCvx(const CvxModel& a, const CvxModel& b)
{
    return eqU4(a.meta, b.meta) && eqU4(a.core[0], b.core[0]) && eqU4(a.core[1], b.core[1]) &&
           eqU4(a.core[2], b.core[2]) && eqF4(a.poseQ, b.poseQ) && eqF4(a.poseP, b.poseP);
}
static bool eqOut(const CssOut& a, const CssOut& b)
{
    return eqU4(a.flags, b.flags) && eqF4(a.dists, b.dists) && eqF4(a.aux, b.aux) &&
           eqCvx(a.c0, b.c0) && eqCvx(a.c1, b.c1);
}

static F4 randQuat(std::mt19937& rng)
{
    auto rf = [&]() { return -1.f + 2.f * (float(rng() % 1000000) / 1000000.f); };
    float a[4]; float m2 = 0;
    for (int k = 0; k < 4; ++k) { a[k] = rf(); m2 += a[k] * a[k]; }
    const float i = 1.f / std::sqrt(m2 > 0 ? m2 : 1.f);
    return F4{ a[0] * i, a[1] * i, a[2] * i, a[3] * i };
}

static void fillBase(CssIn& s, std::mt19937& rng)
{
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    s.t0q = randQuat(rng); s.t0p = F4{ rf(-10, 10), rf(-10, 10), rf(-10, 10), 0 };
    s.t1q = randQuat(rng); s.t1p = F4{ rf(-10, 10), rf(-10, 10), rf(-10, 10), 0 };
    s.s0ScaleS = F4{ rf(0.1f, 3), rf(0.1f, 3), rf(0.1f, 3), 0 }; s.s0ScaleR = randQuat(rng);
    s.s1ScaleS = F4{ rf(0.1f, 3), rf(0.1f, 3), rf(0.1f, 3), 0 }; s.s1ScaleR = randQuat(rng);
    s.dists = F4{ rf(0.01f, 0.5f), rf(0.01f, 0.5f), rf(0, 0.2f), 0 };
    s.misc = U4{ uint32_t(rng()), uint32_t(rng()), uint32_t(rng() % 2), 0 }; // full-width cacheRef passthrough (review 163)
}

PXG_TEST(npCvxCoreSetup_planePrep)
{
    std::mt19937 rng(163001);
    const uint32_t N = 4000;

    std::vector<CssIn> in(N);
    std::vector<CssOut> ref(N);
    size_t cSwap = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        CssIn s{};
        fillBase(s, rng);
        const bool planeFirst = (rng() % 2) == 0;
        U4 planeInfo{ GEOM_PLANE, uint32_t(rng()), uint32_t(rng()), 0 };
        U4 coreInfo{ GEOM_CONVEXCORE, uint32_t(rng()), uint32_t(rng()), 0 }; // hull ptr carries the core enum; full 32-bit passthrough (review 163)
        s.s0Info = planeFirst ? planeInfo : coreInfo;
        s.s1Info = planeFirst ? coreInfo : planeInfo;
        if (!planeFirst) ++cSwap;
        in[i] = s;
        refPlanePrep(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<CssOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npCvxCoreSetup.spv", "ccsPlanePrep",
             { (uint32_t)sizeof(CssIn), (uint32_t)sizeof(CssOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<CssOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
        if (!eqOut(ref[i], g[i])) ++mm;
    PXG_CHECK_MSG(mm == 0, "ccsPlanePrep: %zu mismatches", mm);
    PXG_CHECK_MSG(cSwap > N / 3, "ccsPlanePrep: swap coverage thin (%zu)", cSwap);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

PXG_TEST(npCvxCoreSetup_convexPrep)
{
    std::mt19937 rng(163002);
    const uint32_t N = 4000;

    std::vector<CssIn> in(N);
    std::vector<CssOut> ref(N);
    size_t cTypes[6] = { 0, 0, 0, 0, 0, 0 };
    size_t cBad = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        CssIn s{};
        fillBase(s, rng);
        // all makeConvexShape-supported types, plus an occasional unsupported
        // one (trimesh = 8) driving the return-false Enum(-1) path
        const uint32_t types[6] = { GEOM_SPHERE, GEOM_CAPSULE, GEOM_BOX, GEOM_CONVEXCORE, GEOM_CONVEXMESH, 8 };
        const uint32_t k0 = (i % 13 == 0) ? 5 : rng() % 5;
        const uint32_t k1 = (i % 17 == 0) ? 5 : rng() % 5;
        if (k0 < 5) ++cTypes[k0]; else ++cBad;
        if (k1 < 5) ++cTypes[k1]; else ++cBad;
        s.s0Info = U4{ types[k0], uint32_t(rng()), uint32_t(rng()), 0 };
        s.s1Info = U4{ types[k1], uint32_t(rng()), uint32_t(rng()), 0 };
        in[i] = s;
        refConvexPrep(s, ref[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<CssOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npCvxCoreSetup.spv", "ccsConvexPrep",
             { (uint32_t)sizeof(CssIn), (uint32_t)sizeof(CssOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<CssOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
        if (!eqOut(ref[i], g[i])) ++mm;
    PXG_CHECK_MSG(mm == 0, "ccsConvexPrep: %zu mismatches", mm);
    bool cov = cBad > N / 32;
    for (int k2 = 0; k2 < 5; ++k2) cov = cov && cTypes[k2] > N / 8;
    PXG_CHECK_MSG(cov, "ccsConvexPrep: type coverage thin (%zu %zu %zu %zu %zu bad %zu)",
                  cTypes[0], cTypes[1], cTypes[2], cTypes[3], cTypes[4], cBad);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
