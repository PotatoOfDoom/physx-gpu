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

// Verification of npTriMask.slang: the triangle barycentric / active-edge
// classification helpers.  barycentricCoordinates' (v,w) are continuous
// (almostEqual).  The active-edge masks are discrete integers gated on float
// comparisons (against 0.97 for the barycentric variant, against +-0.0174 for
// the normal variant), so cases within MARG of any evaluated decision boundary
// are skipped; the surviving masks are compared bit-exactly.

#include "pxgTestFramework.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct V3 { float x, y, z; };
struct F4 { float x, y, z, w; };
struct BaryIn { F4 p, a, b, c; };
struct NormIn { F4 v0, v1, v2, n; };
struct PointIn { F4 pa, v0, v1, v2; };
struct TmParams { uint32_t numTests, pad0, pad1, pad2; };

static const uint32_t M_V0 = 1u << 29, M_V1 = 2u << 29, M_V2 = 3u << 29;
static const uint32_t M_E01 = 4u << 29, M_E02 = 5u << 29, M_E12 = 6u << 29;
static const float MARG = 2e-3f;

static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static V3 nrm(V3 a) { const float m = std::sqrt(dot3(a, a)); return { a.x / m, a.y / m, a.z / m }; }
static F4 f4(V3 v) { return { v.x, v.y, v.z, 0.0f }; }
static V3 randUnit(std::mt19937& rng) { std::uniform_real_distribution<float> u(-1, 1); V3 v; float m; do { v = { u(rng), u(rng), u(rng) }; m = std::sqrt(dot3(v, v)); } while (m < 1e-2f); return { v.x / m, v.y / m, v.z / m }; }

static void refBary(V3 p, V3 a, V3 b, V3 c, float& v, float& w)
{
    const V3 ab = sub(b, a), ac = sub(c, a), n = cross3(ab, ac);
    const V3 ap = sub(a, p), bp = sub(b, p), cp = sub(c, p);
    const float va = dot3(n, cross3(bp, cp)), vb = dot3(n, cross3(cp, ap)), vc = dot3(n, cross3(ap, bp));
    const float totalArea = va + vb + vc;
    const float denom = totalArea == 0.0f ? 0.0f : 1.0f / totalArea;
    v = vb * denom; w = vc * denom;
}

static uint32_t refMaskVW(float v, float w, bool& nearBoundary)
{
    const float ub = 0.97f, u = 1.0f - v - w;
    auto near = [&](float x) { return std::fabs(x - ub) < MARG; };
    nearBoundary = near(v) || near(w) || near(u) || near(v + u) || near(u + w) || near(w + v);
    uint32_t mask = 0;
    if (v > ub) mask = M_V1;
    else if (w > ub) mask = M_V2;
    else if (u > ub) mask = M_V0;
    else if ((v + u) > ub) mask = M_E01;
    else if ((u + w) > ub) mask = M_E02;
    else if ((w + v) > ub) mask = M_E12;
    return mask;
}

static uint32_t refMaskNormal(V3 v0, V3 v1, V3 v2, V3 n, bool& nearBoundary)
{
    const V3 e01 = nrm(sub(v1, v0)), e02 = nrm(sub(v2, v0)), e12 = nrm(sub(v2, v1));
    const float d01 = dot3(e01, n), d02 = dot3(e02, n), d12 = dot3(e12, n);
    const V3 tn = nrm(cross3(e01, e02));
    const float eps = 0.0174f;
    const float c01 = dot3(nrm(cross3(tn, e01)), n);
    const float c02 = dot3(nrm(cross3(tn, e02)), n);
    const float c12 = dot3(nrm(cross3(tn, e12)), n);
    auto near = [&](float x) { return std::fabs(std::fabs(x) - eps) < MARG; };
    nearBoundary = near(d01) || near(d02) || near(d12) || near(c01) || near(c02) || near(c12);
    if (d01 < -eps && d02 < -eps) return M_V0;
    if (d01 > eps && d12 < -eps) return M_V1;
    if (d02 > eps && d12 > eps) return M_V2;
    if (c01 < -eps) return M_E01;
    if (c02 > eps) return M_E02;
    if (c12 < -eps) return M_E12;
    return 0;
}

// build a non-degenerate triangle with a modest aspect ratio
static void randTri(std::mt19937& rng, V3& a, V3& b, V3& c)
{
    std::uniform_real_distribution<float> uc(-2, 2);
    do { a = { uc(rng), uc(rng), uc(rng) }; b = { uc(rng), uc(rng), uc(rng) }; c = { uc(rng), uc(rng), uc(rng) };
    } while (dot3(cross3(sub(b, a), sub(c, a)), cross3(sub(b, a), sub(c, a))) < 0.5f);
}

PXG_TEST(npTriMask_barycentric)
{
    std::mt19937 rng(68011);
    const uint32_t N = 12000;
    std::vector<BaryIn> in(N);
    std::vector<float> refV(N), refW(N);
    std::uniform_real_distribution<float> uab(-0.3f, 1.3f);
    for (uint32_t i = 0; i < N; ++i)
    {
        V3 a, b, c; randTri(rng, a, b, c);
        // point as a barycentric combination (in-plane) so v,w are meaningful and spread
        float s = uab(rng), t = uab(rng);
        const V3 ab = sub(b, a), ac = sub(c, a);
        const V3 p = { a.x + ab.x * s + ac.x * t, a.y + ab.y * s + ac.y * t, a.z + ab.z * s + ac.z * t };
        in[i] = { f4(p), f4(a), f4(b), f4(c) };
        refBary(p, a, b, c, refV[i], refW[i]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<F4> outInit(N);
    std::vector<F4> vwD(N); std::vector<NormIn> nD(N); std::vector<PointIn> pD(N); std::vector<uint32_t> mD(N);
    PxgBufferHandle bB = createBufferFrom(dev, in);
    PxgBufferHandle bBo = createBufferFrom(dev, outInit);
    PxgBufferHandle bVw = createBufferFrom(dev, vwD);
    PxgBufferHandle bN = createBufferFrom(dev, nD);
    PxgBufferHandle bP = createBufferFrom(dev, pD);
    PxgBufferHandle bM = createBufferFrom(dev, mD);

    TmParams prm = { N, 0, 0, 0 };
    Kernel k("npTriMask.spv", "ctBarycentric",
             { (uint32_t)sizeof(BaryIn), (uint32_t)sizeof(F4), (uint32_t)sizeof(F4), (uint32_t)sizeof(NormIn), (uint32_t)sizeof(PointIn), 4 }, sizeof(TmParams));
    k.dispatch(GROUPS, { bB, bBo, bVw, bN, bP, bM }, prm);
    dev->flush();

    const auto g = downloadBuffer<F4>(dev, bBo, N);
    size_t mmV = 0, mmW = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (!almostEqual(g[i].x, refV[i], 1e-4f, 1e-4f)) ++mmV;
        if (!almostEqual(g[i].y, refW[i], 1e-4f, 1e-4f)) ++mmW;
    }
    PXG_CHECK_MSG(mmV == 0, "ctBarycentric: %zu v mismatches", mmV);
    PXG_CHECK_MSG(mmW == 0, "ctBarycentric: %zu w mismatches", mmW);

    dev->destroyBuffer(bB); dev->destroyBuffer(bBo); dev->destroyBuffer(bVw); dev->destroyBuffer(bN); dev->destroyBuffer(bP); dev->destroyBuffer(bM);
}

PXG_TEST(npTriMask_activeMaskVW)
{
    std::mt19937 rng(68023);
    const uint32_t N = 20000;
    std::vector<F4> in(N);
    std::vector<uint32_t> ref(N);
    std::vector<uint8_t> skip(N, 0);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    std::uniform_real_distribution<float> ue(0.0f, 0.06f);
    size_t v0c = 0, v1c = 0, v2c = 0, e01c = 0, e02c = 0, e12c = 0, zeroc = 0, skipped = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        // mix: interior, vertex-dominant, edge-dominant (one coord near 0)
        float bu, bv, bw;
        const uint32_t cat = rng() % 4u;
        if (cat == 0) { bu = u01(rng); bv = u01(rng); bw = u01(rng); }              // random -> mostly interior
        else if (cat == 1) { const uint32_t k = rng() % 3u; bu = (k == 0) ? 5.0f : u01(rng) * 0.1f; bv = (k == 1) ? 5.0f : u01(rng) * 0.1f; bw = (k == 2) ? 5.0f : u01(rng) * 0.1f; } // vertex
        else { const uint32_t k = rng() % 3u; bu = (k == 0) ? ue(rng) : u01(rng); bv = (k == 1) ? ue(rng) : u01(rng); bw = (k == 2) ? ue(rng) : u01(rng); } // edge (one near 0)
        const float sum = bu + bv + bw;
        const float v = bv / sum, w = bw / sum;
        in[i] = { v, w, 0, 0 };
        bool nb; ref[i] = refMaskVW(v, w, nb);
        if (nb) { skip[i] = 1; ++skipped; continue; }
        if (ref[i] == M_V0) ++v0c; else if (ref[i] == M_V1) ++v1c; else if (ref[i] == M_V2) ++v2c;
        else if (ref[i] == M_E01) ++e01c; else if (ref[i] == M_E02) ++e02c; else if (ref[i] == M_E12) ++e12c; else ++zeroc;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<BaryIn> bD(N); std::vector<F4> boD(N); std::vector<NormIn> nD(N); std::vector<PointIn> pD(N);
    std::vector<uint32_t> outInit(N, 0xffffffffu);
    PxgBufferHandle bB = createBufferFrom(dev, bD);
    PxgBufferHandle bBo = createBufferFrom(dev, boD);
    PxgBufferHandle bVw = createBufferFrom(dev, in);
    PxgBufferHandle bN = createBufferFrom(dev, nD);
    PxgBufferHandle bP = createBufferFrom(dev, pD);
    PxgBufferHandle bM = createBufferFrom(dev, outInit);

    TmParams prm = { N, 0, 0, 0 };
    Kernel k("npTriMask.spv", "ctActiveMaskVW",
             { (uint32_t)sizeof(BaryIn), (uint32_t)sizeof(F4), (uint32_t)sizeof(F4), (uint32_t)sizeof(NormIn), (uint32_t)sizeof(PointIn), 4 }, sizeof(TmParams));
    k.dispatch(GROUPS, { bB, bBo, bVw, bN, bP, bM }, prm);
    dev->flush();

    const auto g = downloadBuffer<uint32_t>(dev, bM, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i) { if (skip[i]) continue; if (g[i] != ref[i]) ++mm; }
    PXG_CHECK_MSG(mm == 0, "ctActiveMaskVW: %zu mask mismatches", mm);
    PXG_CHECK_MSG(v0c > 50 && v1c > 50 && v2c > 50 && e01c > 50 && e02c > 50 && e12c > 50 && zeroc > 100,
                  "ctActiveMaskVW: coverage thin (v=%zu/%zu/%zu e=%zu/%zu/%zu z=%zu)", v0c, v1c, v2c, e01c, e02c, e12c, zeroc);

    dev->destroyBuffer(bB); dev->destroyBuffer(bBo); dev->destroyBuffer(bVw); dev->destroyBuffer(bN); dev->destroyBuffer(bP); dev->destroyBuffer(bM);
}

PXG_TEST(npTriMask_activeMaskFromNormal)
{
    std::mt19937 rng(68037);
    const uint32_t N = 20000;
    std::vector<NormIn> in(N);
    std::vector<uint32_t> ref(N);
    std::vector<uint8_t> skip(N, 0);
    size_t v0c = 0, v1c = 0, v2c = 0, e01c = 0, e02c = 0, e12c = 0, zeroc = 0, skipped = 0;
    std::uniform_real_distribution<float> up(-0.006f, 0.006f); // tiny: keep edge dots well within +-eps -> robust fall-through mask 0
    for (uint32_t i = 0; i < N; ++i)
    {
        V3 v0, v1, v2; randTri(rng, v0, v1, v2);
        V3 n;
        if (rng() % 3u == 0u)
        {
            // bias toward the face normal (+/-): all edge dots ~0 -> fall-through mask 0
            const V3 fn = nrm(cross3(sub(v1, v0), sub(v2, v0)));
            const float sgn = (rng() & 1u) ? 1.0f : -1.0f;
            n = nrm({ fn.x * sgn + up(rng), fn.y * sgn + up(rng), fn.z * sgn + up(rng) });
        }
        else n = randUnit(rng);
        in[i] = { f4(v0), f4(v1), f4(v2), f4(n) };
        bool nb; ref[i] = refMaskNormal(v0, v1, v2, n, nb);
        if (nb) { skip[i] = 1; ++skipped; continue; }
        if (ref[i] == M_V0) ++v0c; else if (ref[i] == M_V1) ++v1c; else if (ref[i] == M_V2) ++v2c;
        else if (ref[i] == M_E01) ++e01c; else if (ref[i] == M_E02) ++e02c; else if (ref[i] == M_E12) ++e12c; else ++zeroc;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<BaryIn> bD(N); std::vector<F4> boD(N); std::vector<F4> vwD(N); std::vector<PointIn> pD(N);
    std::vector<uint32_t> outInit(N, 0xffffffffu);
    PxgBufferHandle bB = createBufferFrom(dev, bD);
    PxgBufferHandle bBo = createBufferFrom(dev, boD);
    PxgBufferHandle bVw = createBufferFrom(dev, vwD);
    PxgBufferHandle bN = createBufferFrom(dev, in);
    PxgBufferHandle bP = createBufferFrom(dev, pD);
    PxgBufferHandle bM = createBufferFrom(dev, outInit);

    TmParams prm = { N, 0, 0, 0 };
    Kernel k("npTriMask.spv", "ctActiveMaskFromNormal",
             { (uint32_t)sizeof(BaryIn), (uint32_t)sizeof(F4), (uint32_t)sizeof(F4), (uint32_t)sizeof(NormIn), (uint32_t)sizeof(PointIn), 4 }, sizeof(TmParams));
    k.dispatch(GROUPS, { bB, bBo, bVw, bN, bP, bM }, prm);
    dev->flush();

    const auto g = downloadBuffer<uint32_t>(dev, bM, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i) { if (skip[i]) continue; if (g[i] != ref[i]) ++mm; }
    PXG_CHECK_MSG(mm == 0, "ctActiveMaskFromNormal: %zu mask mismatches", mm);
    PXG_CHECK_MSG(v0c > 50 && v1c > 50 && v2c > 50 && e01c > 50 && e02c > 50 && e12c > 50 && zeroc > N / 8,
                  "ctActiveMaskFromNormal: coverage thin (v=%zu/%zu/%zu e=%zu/%zu/%zu z=%zu skip=%zu)",
                  v0c, v1c, v2c, e01c, e02c, e12c, zeroc, skipped);

    dev->destroyBuffer(bB); dev->destroyBuffer(bBo); dev->destroyBuffer(bVw); dev->destroyBuffer(bN); dev->destroyBuffer(bP); dev->destroyBuffer(bM);
}

PXG_TEST(npTriMask_activeMaskPoint)
{
    std::mt19937 rng(68051);
    const uint32_t N = 16000;
    std::vector<PointIn> in(N);
    std::vector<uint32_t> ref(N);
    std::vector<uint8_t> skip(N, 0);
    std::uniform_real_distribution<float> uab(-0.2f, 1.2f);
    size_t nonzero = 0, zeroc = 0, skipped = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        V3 v0, v1, v2; randTri(rng, v0, v1, v2);
        const float s = uab(rng), t = uab(rng);
        const V3 e1 = sub(v1, v0), e2 = sub(v2, v0);
        const V3 pa = { v0.x + e1.x * s + e2.x * t, v0.y + e1.y * s + e2.y * t, v0.z + e1.z * s + e2.z * t };
        in[i] = { f4(pa), f4(v0), f4(v1), f4(v2) };
        float v, w; refBary(pa, v0, v1, v2, v, w);
        bool nb; ref[i] = refMaskVW(v, w, nb);
        if (nb) { skip[i] = 1; ++skipped; continue; }
        if (ref[i] == 0) ++zeroc; else ++nonzero;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<BaryIn> bD(N); std::vector<F4> boD(N); std::vector<F4> vwD(N); std::vector<NormIn> nD(N);
    std::vector<uint32_t> outInit(N, 0xffffffffu);
    PxgBufferHandle bB = createBufferFrom(dev, bD);
    PxgBufferHandle bBo = createBufferFrom(dev, boD);
    PxgBufferHandle bVw = createBufferFrom(dev, vwD);
    PxgBufferHandle bN = createBufferFrom(dev, nD);
    PxgBufferHandle bP = createBufferFrom(dev, in);
    PxgBufferHandle bM = createBufferFrom(dev, outInit);

    TmParams prm = { N, 0, 0, 0 };
    Kernel k("npTriMask.spv", "ctActiveMaskPoint",
             { (uint32_t)sizeof(BaryIn), (uint32_t)sizeof(F4), (uint32_t)sizeof(F4), (uint32_t)sizeof(NormIn), (uint32_t)sizeof(PointIn), 4 }, sizeof(TmParams));
    k.dispatch(GROUPS, { bB, bBo, bVw, bN, bP, bM }, prm);
    dev->flush();

    const auto g = downloadBuffer<uint32_t>(dev, bM, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i) { if (skip[i]) continue; if (g[i] != ref[i]) ++mm; }
    PXG_CHECK_MSG(mm == 0, "ctActiveMaskPoint: %zu mask mismatches", mm);
    PXG_CHECK_MSG(nonzero > N / 8 && zeroc > N / 8, "ctActiveMaskPoint: coverage thin (nonzero=%zu zero=%zu skip=%zu)", nonzero, zeroc, skipped);

    dev->destroyBuffer(bB); dev->destroyBuffer(bBo); dev->destroyBuffer(bVw); dev->destroyBuffer(bN); dev->destroyBuffer(bP); dev->destroyBuffer(bM);
}

} // namespace
