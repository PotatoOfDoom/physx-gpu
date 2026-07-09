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

// Verification of npTetra.slang: ctTriBBox / ctTetBBox (triBoundingBox /
// tetBoundingBox, a componentwise min/max over the vertices — exact) and
// ctTetBary (computeTetBarycentric, the Cramer's-rule tetrahedron barycentric
// with a clamp and an inside/outside boolean).  The bounds are compared
// bit-exactly; the (clamped) barycentric coordinates are continuous and
// compared to tolerance, while the inside/outside boolean is a discrete
// decision whose -eps / 1+eps boundaries are margin-skipped.  Both inside and
// outside points are exercised.

#include "pxgTestFramework.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct V3 { float x, y, z; };
struct F4 { float x, y, z, w; };
struct TriB { F4 v0, v1, v2; };
struct TetB { F4 v0, v1, v2, v3; };
struct BaryB { F4 v0, v1, v2, v3, p; };
struct BBox { F4 mn, mx; };
struct BaryO { F4 bary, extra; };
struct TetParams { uint32_t numTests, pad0, pad1, pad2; };

static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 cross3(V3 a, V3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static float minf(float a, float b) { return a < b ? a : b; }
static float maxf(float a, float b) { return a > b ? a : b; }
static float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
static F4 f4(V3 v) { return { v.x, v.y, v.z, 0.0f }; }

// Faithful transcription of computeTetBarycentric; also returns the four
// unclamped coords so the test can margin-skip the boolean boundary.
static bool refTetBary(V3 a, V3 b, V3 c, V3 d, V3 point, F4& bary, float u4[4], float eps)
{
    const float minEps = -eps, maxEps = 1.0f + eps;
    const V3 ba = sub(b, a), ca = sub(c, a), da = sub(d, a), pa = sub(point, a);
    const float detBcd = dot3(ba, cross3(ca, da));
    const float detPcd = dot3(pa, cross3(ca, da));
    const float v = detPcd / detBcd;
    const float detBpd = dot3(ba, cross3(pa, da));
    const float w = detBpd / detBcd;
    const float detBcp = dot3(ba, cross3(ca, pa));
    const float x = detBcp / detBcd;
    const float u = 1.0f - v - w - x;
    u4[0] = u; u4[1] = v; u4[2] = w; u4[3] = x;
    bary = { clampf(u, 0, 1), clampf(v, 0, 1), clampf(w, 0, 1), clampf(x, 0, 1) };
    return (u >= minEps && u <= maxEps) && (v >= minEps && v <= maxEps) && (w >= minEps && w <= maxEps) && (x >= minEps && x <= maxEps);
}

// Shared buffer set for the multi-entry module (dummies for unused buffers).
struct Bufs { PxgBufferHandle tri, triBB, tet, tetBB, bary, baryOut; };
static Bufs makeBufs(PxgSlangComputeDevice* dev, uint32_t N,
                     const std::vector<TriB>& tri, const std::vector<BBox>& triBB,
                     const std::vector<TetB>& tet, const std::vector<BBox>& tetBB,
                     const std::vector<BaryB>& bary, const std::vector<BaryO>& baryOut)
{
    return { createBufferFrom(dev, tri), createBufferFrom(dev, triBB),
             createBufferFrom(dev, tet), createBufferFrom(dev, tetBB),
             createBufferFrom(dev, bary), createBufferFrom(dev, baryOut) };
}
static void destroyBufs(PxgSlangComputeDevice* dev, Bufs& b)
{
    dev->destroyBuffer(b.tri); dev->destroyBuffer(b.triBB); dev->destroyBuffer(b.tet);
    dev->destroyBuffer(b.tetBB); dev->destroyBuffer(b.bary); dev->destroyBuffer(b.baryOut);
}
static Kernel makeKernel(const char* entry)
{
    return Kernel("npTetra.spv", entry,
                  { (uint32_t)sizeof(TriB), (uint32_t)sizeof(BBox), (uint32_t)sizeof(TetB),
                    (uint32_t)sizeof(BBox), (uint32_t)sizeof(BaryB), (uint32_t)sizeof(BaryO) },
                  sizeof(TetParams));
}

PXG_TEST(npTetra_triBBox)
{
    std::mt19937 rng(73011);
    const uint32_t N = 8000;
    std::uniform_real_distribution<float> uc(-5, 5);
    std::vector<TriB> in(N);
    std::vector<BBox> ref(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        const V3 v0 = { uc(rng), uc(rng), uc(rng) }, v1 = { uc(rng), uc(rng), uc(rng) }, v2 = { uc(rng), uc(rng), uc(rng) };
        in[i] = { f4(v0), f4(v1), f4(v2) };
        const V3 mn = { minf(minf(v0.x, v1.x), v2.x), minf(minf(v0.y, v1.y), v2.y), minf(minf(v0.z, v1.z), v2.z) };
        const V3 mx = { maxf(maxf(v0.x, v1.x), v2.x), maxf(maxf(v0.y, v1.y), v2.y), maxf(maxf(v0.z, v1.z), v2.z) };
        ref[i] = { f4(mn), f4(mx) };
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<BBox> triBB(N); std::vector<TetB> tet(N); std::vector<BBox> tetBB(N);
    std::vector<BaryB> bary(N); std::vector<BaryO> baryOut(N);
    Bufs b = makeBufs(dev, N, in, triBB, tet, tetBB, bary, baryOut);

    TetParams prm = { N, 0, 0, 0 };
    Kernel k = makeKernel("ctTriBBox");
    k.dispatch(GROUPS, { b.tri, b.triBB, b.tet, b.tetBB, b.bary, b.baryOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<BBox>(dev, b.triBB, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const BBox& r = ref[i]; const BBox& v = g[i];
        if (r.mn.x != v.mn.x || r.mn.y != v.mn.y || r.mn.z != v.mn.z || r.mx.x != v.mx.x || r.mx.y != v.mx.y || r.mx.z != v.mx.z) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "ctTriBBox: %zu bounds mismatches", mm);
    destroyBufs(dev, b);
}

PXG_TEST(npTetra_tetBBox)
{
    std::mt19937 rng(73019);
    const uint32_t N = 8000;
    std::uniform_real_distribution<float> uc(-5, 5);
    std::vector<TetB> in(N);
    std::vector<BBox> ref(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        const V3 v0 = { uc(rng), uc(rng), uc(rng) }, v1 = { uc(rng), uc(rng), uc(rng) };
        const V3 v2 = { uc(rng), uc(rng), uc(rng) }, v3 = { uc(rng), uc(rng), uc(rng) };
        in[i] = { f4(v0), f4(v1), f4(v2), f4(v3) };
        const V3 mn = { minf(minf(v0.x, v1.x), minf(v2.x, v3.x)), minf(minf(v0.y, v1.y), minf(v2.y, v3.y)), minf(minf(v0.z, v1.z), minf(v2.z, v3.z)) };
        const V3 mx = { maxf(maxf(v0.x, v1.x), maxf(v2.x, v3.x)), maxf(maxf(v0.y, v1.y), maxf(v2.y, v3.y)), maxf(maxf(v0.z, v1.z), maxf(v2.z, v3.z)) };
        ref[i] = { f4(mn), f4(mx) };
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<TriB> tri(N); std::vector<BBox> triBB(N); std::vector<BBox> tetBB(N);
    std::vector<BaryB> bary(N); std::vector<BaryO> baryOut(N);
    Bufs b = makeBufs(dev, N, tri, triBB, in, tetBB, bary, baryOut);

    TetParams prm = { N, 0, 0, 0 };
    Kernel k = makeKernel("ctTetBBox");
    k.dispatch(GROUPS, { b.tri, b.triBB, b.tet, b.tetBB, b.bary, b.baryOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<BBox>(dev, b.tetBB, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const BBox& r = ref[i]; const BBox& v = g[i];
        if (r.mn.x != v.mn.x || r.mn.y != v.mn.y || r.mn.z != v.mn.z || r.mx.x != v.mx.x || r.mx.y != v.mx.y || r.mx.z != v.mx.z) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "ctTetBBox: %zu bounds mismatches", mm);
    destroyBufs(dev, b);
}

PXG_TEST(npTetra_tetBary)
{
    std::mt19937 rng(73033);
    const uint32_t N = 16000;
    const float EPS = 1e-5f;
    const float MARG = 1e-3f; // margin-skip band around the -eps / 1+eps inside/outside boundary
    std::uniform_real_distribution<float> uc(-2, 2);
    std::uniform_real_distribution<float> uw(0.05f, 1.0f);
    std::uniform_real_distribution<float> ub(-1.0f, 2.0f); // wide barycentric range -> inside AND outside
    std::vector<BaryB> in(N);
    std::vector<BaryO> ref(N);
    std::vector<uint8_t> skip(N, 0);
    size_t insideC = 0, outsideC = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        V3 a, b, c, d;
        // Non-degenerate tetrahedron: |6*volume| = |detBcd| well away from 0.
        float detBcd;
        do {
            a = { uc(rng), uc(rng), uc(rng) }; b = { uc(rng), uc(rng), uc(rng) };
            c = { uc(rng), uc(rng), uc(rng) }; d = { uc(rng), uc(rng), uc(rng) };
            detBcd = dot3(sub(b, a), cross3(sub(c, a), sub(d, a)));
        } while (std::fabs(detBcd) < 1.5f);

        // Half the points are a random affine combination (spanning inside and
        // just-outside via the wide weight range); half are drawn from a box.
        V3 p;
        if (i & 1)
        {
            // Interior point: non-negative weights normalized to sum 1 -> inside == true.
            float wa = uw(rng), wb = uw(rng), wc = uw(rng), wd = uw(rng);
            const float s = 1.0f / (wa + wb + wc + wd);
            wa *= s; wb *= s; wc *= s; wd *= s;
            p = { wa * a.x + wb * b.x + wc * c.x + wd * d.x,
                  wa * a.y + wb * b.y + wc * c.y + wd * d.y,
                  wa * a.z + wb * b.z + wc * c.z + wd * d.z };
            (void)ub;
        }
        else
        {
            std::uniform_real_distribution<float> up(-3, 3);
            p = { up(rng), up(rng), up(rng) };
        }
        in[i] = { f4(a), f4(b), f4(c), f4(d), f4(p) };

        F4 bary; float u4[4];
        const bool inside = refTetBary(a, b, c, d, p, bary, u4, EPS);
        ref[i] = { bary, { inside ? 1.0f : 0.0f, 0, 0, 0 } };
        if (inside) ++insideC; else ++outsideC;
        // Skip the boolean comparison when any coord is within MARG of a decision boundary.
        for (int j = 0; j < 4; ++j)
            if (std::fabs(u4[j] - (-EPS)) < MARG || std::fabs(u4[j] - (1.0f + EPS)) < MARG) skip[i] = 1;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<TriB> tri(N); std::vector<BBox> triBB(N); std::vector<TetB> tet(N); std::vector<BBox> tetBB(N);
    std::vector<BaryO> baryOut(N);
    Bufs bb = makeBufs(dev, N, tri, triBB, tet, tetBB, in, baryOut);

    TetParams prm = { N, 0, 0, 0 };
    Kernel k = makeKernel("ctTetBary");
    k.dispatch(GROUPS, { bb.tri, bb.triBB, bb.tet, bb.tetBB, bb.bary, bb.baryOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<BaryO>(dev, bb.baryOut, N);
    size_t mmB = 0, mmI = 0, skipped = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const BaryO& r = ref[i]; const BaryO& v = g[i];
        if (!almostEqual(v.bary.x, r.bary.x, 1e-3f, 1e-4f) || !almostEqual(v.bary.y, r.bary.y, 1e-3f, 1e-4f) ||
            !almostEqual(v.bary.z, r.bary.z, 1e-3f, 1e-4f) || !almostEqual(v.bary.w, r.bary.w, 1e-3f, 1e-4f)) ++mmB;
        if (skip[i]) { ++skipped; continue; }
        if (v.extra.x != r.extra.x) ++mmI;
    }
    PXG_CHECK_MSG(mmB == 0, "ctTetBary: %zu barycentric mismatches", mmB);
    PXG_CHECK_MSG(mmI == 0, "ctTetBary: %zu inside/outside mismatches (skipped %zu)", mmI, skipped);
    PXG_CHECK_MSG(insideC > N / 8 && outsideC > N / 8, "ctTetBary: coverage thin (inside=%zu outside=%zu)", insideC, outsideC);

    destroyBufs(dev, bb);
}

} // namespace
