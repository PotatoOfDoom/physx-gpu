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

// Verification of npConvexSat.slang: ctSupportConvex / ctSupportTriangle
// (supportConvex/supportTriangle) and ctSatTriangleNormal (satTriangleNormal) —
// the convex-collision projection / separating-axis primitives.  The support
// min/max and the SAT separation are continuous (dot products), compared with
// almostEqual.  satTriangleNormal's discrete outputs (the isSeparated early-out,
// the separation-update decision, and hence the feature and faceNormal) are
// gated on float comparisons, so cases within MARG of those boundaries are
// skipped; the feature index is a bit-exact integer when compared.

#include "pxgTestFramework.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct V3 { float x, y, z; };
struct F4 { float x, y, z, w; };
struct CsIn { F4 dir, triNormal, misc, seedFaceNormal; uint32_t seedFeature, p0, p1, p2; };
struct CsOut { F4 sepRet, faceNormal; uint32_t feature, q0, q1, q2; };
struct CsParams { uint32_t numTests, maxVerts, pad0, pad1; };

static const uint32_t TRI_FEATURE_IDX = 0x7FffFFffu;
static const uint32_t MAX_VERTS = 24;
static const float MARG = 1e-3f;

static float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static float len(V3 a) { return std::sqrt(dot(a, a)); }
static V3 norm(V3 a) { const float m = len(a); return { a.x / m, a.y / m, a.z / m }; }
static F4 f4(V3 v) { return { v.x, v.y, v.z, 0.0f }; }

static V3 randUnit(std::mt19937& rng)
{
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    V3 v; do { v = { u(rng), u(rng), u(rng) }; } while (dot(v, v) < 1e-4f);
    return norm(v);
}

static void supportRef(const std::vector<F4>& verts, uint32_t base, uint32_t nb, V3 dir, float& mn, float& mx)
{
    mn = FLT_MAX; mx = -FLT_MAX;
    for (uint32_t i = 0; i < nb; ++i)
    {
        const V3 v = { verts[base + i].x, verts[base + i].y, verts[base + i].z };
        const float d = dot(v, dir);
        mn = std::min(d, mn); mx = std::max(d, mx);
    }
}

PXG_TEST(npConvexSat_supportConvex)
{
    std::mt19937 rng(64011);
    const uint32_t N = 8000;
    std::vector<F4> verts(N * MAX_VERTS);
    std::vector<CsIn> in(N);
    std::vector<float> refMn(N), refMx(N);
    std::uniform_real_distribution<float> uc(-5.0f, 5.0f);
    std::uniform_int_distribution<uint32_t> un(1, MAX_VERTS);
    for (uint32_t t = 0; t < N; ++t)
    {
        const uint32_t nb = un(rng);
        for (uint32_t i = 0; i < MAX_VERTS; ++i)
            verts[t * MAX_VERTS + i] = { uc(rng), uc(rng), uc(rng), 0.0f };
        const V3 dir = randUnit(rng);
        in[t] = { f4(dir), {0,0,0,0}, { float(nb), 0, 0, 0 }, {0,0,0,0}, 0, 0, 0, 0 };
        supportRef(verts, t * MAX_VERTS, nb, dir, refMn[t], refMx[t]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<F4> triDummy(N * 3);
    std::vector<CsOut> outInit(N);
    PxgBufferHandle bV = createBufferFrom(dev, verts);
    PxgBufferHandle bT = createBufferFrom(dev, triDummy);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    CsParams prm = { N, MAX_VERTS, 0, 0 };
    Kernel k("npConvexSat.spv", "ctSupportConvex",
             { (uint32_t)sizeof(F4), (uint32_t)sizeof(F4), (uint32_t)sizeof(CsIn), (uint32_t)sizeof(CsOut) }, sizeof(CsParams));
    k.dispatch(GROUPS, { bV, bT, bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<CsOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t t = 0; t < N; ++t)
        if (!almostEqual(g[t].sepRet.x, refMn[t], 1e-5f, 1e-4f) || !almostEqual(g[t].sepRet.y, refMx[t], 1e-5f, 1e-4f)) ++mm;
    PXG_CHECK_MSG(mm == 0, "ctSupportConvex: %zu min/max mismatches", mm);

    dev->destroyBuffer(bV); dev->destroyBuffer(bT); dev->destroyBuffer(bIn); dev->destroyBuffer(bOut);
}

PXG_TEST(npConvexSat_supportTriangle)
{
    std::mt19937 rng(64019);
    const uint32_t N = 8000;
    std::vector<F4> tri(N * 3);
    std::vector<CsIn> in(N);
    std::vector<float> refMn(N), refMx(N);
    std::uniform_real_distribution<float> uc(-5.0f, 5.0f);
    for (uint32_t t = 0; t < N; ++t)
    {
        for (uint32_t i = 0; i < 3; ++i) tri[t * 3 + i] = { uc(rng), uc(rng), uc(rng), 0.0f };
        const V3 dir = randUnit(rng);
        in[t] = { f4(dir), {0,0,0,0}, {0,0,0,0}, {0,0,0,0}, 0, 0, 0, 0 };
        std::vector<F4> tv = { tri[t * 3], tri[t * 3 + 1], tri[t * 3 + 2] };
        supportRef(tv, 0, 3, dir, refMn[t], refMx[t]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<F4> vDummy(N * MAX_VERTS);
    std::vector<CsOut> outInit(N);
    PxgBufferHandle bV = createBufferFrom(dev, vDummy);
    PxgBufferHandle bT = createBufferFrom(dev, tri);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    CsParams prm = { N, MAX_VERTS, 0, 0 };
    Kernel k("npConvexSat.spv", "ctSupportTriangle",
             { (uint32_t)sizeof(F4), (uint32_t)sizeof(F4), (uint32_t)sizeof(CsIn), (uint32_t)sizeof(CsOut) }, sizeof(CsParams));
    k.dispatch(GROUPS, { bV, bT, bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<CsOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t t = 0; t < N; ++t)
        if (!almostEqual(g[t].sepRet.x, refMn[t], 1e-5f, 1e-4f) || !almostEqual(g[t].sepRet.y, refMx[t], 1e-5f, 1e-4f)) ++mm;
    PXG_CHECK_MSG(mm == 0, "ctSupportTriangle: %zu min/max mismatches", mm);

    dev->destroyBuffer(bV); dev->destroyBuffer(bT); dev->destroyBuffer(bIn); dev->destroyBuffer(bOut);
}

PXG_TEST(npConvexSat_satTriangleNormal)
{
    std::mt19937 rng(64029);
    const uint32_t N = 16000;
    std::vector<F4> verts(N * MAX_VERTS);
    std::vector<F4> tri(N * 3);
    std::vector<CsIn> in(N);
    std::vector<uint8_t> refRet(N), skip(N, 0);
    std::vector<float> refSep(N);
    std::vector<V3> refFN(N);
    std::vector<uint32_t> refFeat(N);
    std::uniform_real_distribution<float> uc(-3.0f, 3.0f);
    std::uniform_real_distribution<float> ucd(0.02f, 0.4f);
    std::uniform_int_distribution<uint32_t> un(4, MAX_VERTS);
    size_t sepC = 0, overlapC = 0, updated = 0, notUpd = 0, skipped = 0;

    for (uint32_t t = 0; t < N; ++t)
    {
        const uint32_t nb = un(rng);
        for (uint32_t i = 0; i < MAX_VERTS; ++i) verts[t * MAX_VERTS + i] = { uc(rng), uc(rng), uc(rng), 0.0f };
        // triangle placed with a random offset so overlap/separated both occur
        const V3 off = { uc(rng), uc(rng), uc(rng) };
        for (uint32_t i = 0; i < 3; ++i) tri[t * 3 + i] = { uc(rng) * 0.5f + off.x, uc(rng) * 0.5f + off.y, uc(rng) * 0.5f + off.z, 0.0f };
        const V3 n0 = randUnit(rng);
        const float contactDist = ucd(rng);
        const float seedSep = (rng() & 1u) ? -1e30f : (uc(rng) * 0.3f);
        const V3 seedFN = randUnit(rng);
        const uint32_t seedFeat = 0x80000000u; // EDGE_FEATURE_IDX
        in[t] = { {0,0,0,0}, f4(n0), { float(nb), contactDist, seedSep, 0 }, f4(seedFN), seedFeat, 0, 0, 0 };

        // reference
        const V3 tv0 = { tri[t * 3].x, tri[t * 3].y, tri[t * 3].z };
        const float triProj = dot(n0, tv0);
        float min1, max1; supportRef(verts, t * MAX_VERTS, nb, n0, min1, max1);
        const float lo = triProj + contactDist, hi = max1 + contactDist;
        // skip near the two isSeparated comparison boundaries
        if (std::fabs(min1 - lo) < MARG || std::fabs(triProj - hi) < MARG) { skip[t] = 1; ++skipped; continue; }
        const bool isSeparated = (min1 > lo) || (triProj > hi);
        float sep = seedSep; V3 fn = seedFN; uint32_t feat = seedFeat; uint8_t ret = 1;
        if (isSeparated) { ret = 0; ++sepC; }
        else
        {
            ++overlapC;
            const float _sep = min1 - triProj;
            // skip near the separation-update boundary
            if (std::fabs(sep - _sep) < MARG) { skip[t] = 1; ++skipped; continue; }
            if (sep < _sep) { fn = { -n0.x, -n0.y, -n0.z }; sep = _sep; feat = TRI_FEATURE_IDX; ++updated; }
            else ++notUpd;
        }
        refRet[t] = ret; refSep[t] = sep; refFN[t] = fn; refFeat[t] = feat;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<CsOut> outInit(N);
    PxgBufferHandle bV = createBufferFrom(dev, verts);
    PxgBufferHandle bT = createBufferFrom(dev, tri);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    CsParams prm = { N, MAX_VERTS, 0, 0 };
    Kernel k("npConvexSat.spv", "ctSatTriangleNormal",
             { (uint32_t)sizeof(F4), (uint32_t)sizeof(F4), (uint32_t)sizeof(CsIn), (uint32_t)sizeof(CsOut) }, sizeof(CsParams));
    k.dispatch(GROUPS, { bV, bT, bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<CsOut>(dev, bOut, N);
    size_t mmRet = 0, mmSep = 0, mmFN = 0, mmFeat = 0;
    for (uint32_t t = 0; t < N; ++t)
    {
        if (skip[t]) continue;
        const CsOut& v = g[t];
        const uint8_t gret = (v.sepRet.y != 0.0f) ? 1 : 0;
        if (gret != refRet[t]) { ++mmRet; continue; }
        if (!almostEqual(v.sepRet.x, refSep[t], 1e-4f, 1e-4f)) { ++mmSep; continue; }
        if (v.feature != refFeat[t]) { ++mmFeat; continue; }
        if (!almostEqual(v.faceNormal.x, refFN[t].x, 1e-4f, 1e-4f) ||
            !almostEqual(v.faceNormal.y, refFN[t].y, 1e-4f, 1e-4f) ||
            !almostEqual(v.faceNormal.z, refFN[t].z, 1e-4f, 1e-4f)) ++mmFN;
    }
    PXG_CHECK_MSG(mmRet == 0, "satTriangleNormal: %zu return mismatches", mmRet);
    PXG_CHECK_MSG(mmSep == 0, "satTriangleNormal: %zu separation mismatches", mmSep);
    PXG_CHECK_MSG(mmFeat == 0, "satTriangleNormal: %zu feature mismatches", mmFeat);
    PXG_CHECK_MSG(mmFN == 0, "satTriangleNormal: %zu faceNormal mismatches", mmFN);
    PXG_CHECK_MSG(sepC > 100 && overlapC > N / 8 && updated > 100 && notUpd > 100,
                  "satTriangleNormal: coverage thin (sep=%zu overlap=%zu upd=%zu notUpd=%zu skip=%zu)",
                  sepC, overlapC, updated, notUpd, skipped);

    dev->destroyBuffer(bV); dev->destroyBuffer(bT); dev->destroyBuffer(bIn); dev->destroyBuffer(bOut);
}

} // namespace
