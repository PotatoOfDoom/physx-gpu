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

// Verification of npTriMesh.slang: ctTriNormal (getTriNormal) and
// ctTriBarycentric (getTriBarycentric) — the scaled-triangle accessors that
// transform the three triangle vertices by the mesh scale (PxMeshScale::toMat33)
// and return the normalized face normal / the Cramer's-rule barycentric
// coordinates of a query point.  All outputs are continuous, compared with
// almostEqual; the query point is placed as a barycentric combination of the
// scaled triangle so the coordinates are well-conditioned.

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
struct MeshScale { F4 scale, rot; };
struct TriIn { F4 v0, v1, v2, p; };
struct TmParams { uint32_t numTests, pad0, pad1, pad2; };

static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 cross3(V3 a, V3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static float len(V3 a) { return std::sqrt(dot3(a, a)); }
static V3 nrm(V3 a) { const float m = len(a); return { a.x / m, a.y / m, a.z / m }; }
static F4 f4(V3 v) { return { v.x, v.y, v.z, 0.0f }; }

// quaternion -> PxMat33 columns (matches PxMat33T(const PxQuatT&))
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
static V3 matT(V3 m0, V3 m1, V3 m2, V3 v) { return { m0.x * v.x + m1.x * v.y + m2.x * v.z, m0.y * v.x + m1.y * v.y + m2.y * v.z, m0.z * v.x + m1.z * v.y + m2.z * v.z }; }

static F4 randUnitQuat(std::mt19937& rng)
{
    std::uniform_real_distribution<float> u(-1, 1);
    F4 q; float m;
    do { q = { u(rng), u(rng), u(rng), u(rng) }; m = std::sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w); } while (m < 1e-3f);
    return { q.x / m, q.y / m, q.z / m, q.w / m };
}

// build a scaled triangle (non-degenerate) and return its scaled verts
static void makeCase(std::mt19937& rng, TriIn& tri, MeshScale& ms, V3& sv0, V3& sv1, V3& sv2)
{
    std::uniform_real_distribution<float> uc(-2, 2);
    std::uniform_real_distribution<float> usc(0.3f, 2.0f);
    V3 v0, v1, v2, m0, m1, m2;
    do {
        v0 = { uc(rng), uc(rng), uc(rng) }; v1 = { uc(rng), uc(rng), uc(rng) }; v2 = { uc(rng), uc(rng), uc(rng) };
        // scale magnitudes 0.3..2, ~30% of components negative
        V3 sc = { usc(rng), usc(rng), usc(rng) };
        if (rng() % 3u == 0u) sc.x = -sc.x;
        if (rng() % 3u == 0u) sc.z = -sc.z;
        ms.scale = { sc.x, sc.y, sc.z, 0.0f };
        ms.rot = randUnitQuat(rng);
        meshScaleMat({ sc.x, sc.y, sc.z }, ms.rot, m0, m1, m2);
        sv0 = matT(m0, m1, m2, v0); sv1 = matT(m0, m1, m2, v1); sv2 = matT(m0, m1, m2, v2);
    } while (len(cross3(sub(sv1, sv0), sub(sv2, sv0))) < 0.5f); // non-degenerate scaled area
    tri.v0 = f4(v0); tri.v1 = f4(v1); tri.v2 = f4(v2);
}

PXG_TEST(npTriMesh_triNormal)
{
    std::mt19937 rng(71011);
    const uint32_t N = 12000;
    std::vector<TriIn> tri(N);
    std::vector<MeshScale> ms(N);
    std::vector<F4> ref(N);
    for (uint32_t i = 0; i < N; ++i)
    {
        V3 sv0, sv1, sv2; makeCase(rng, tri[i], ms[i], sv0, sv1, sv2);
        tri[i].p = { 0, 0, 0, 0 };
        const V3 n = nrm(cross3(sub(sv1, sv0), sub(sv2, sv0)));
        ref[i] = { n.x, n.y, n.z, 0.0f };
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<F4> outInit(N), baryDummy(N);
    PxgBufferHandle bT = createBufferFrom(dev, tri);
    PxgBufferHandle bS = createBufferFrom(dev, ms);
    PxgBufferHandle bN = createBufferFrom(dev, outInit);
    PxgBufferHandle bB = createBufferFrom(dev, baryDummy);

    TmParams prm = { N, 0, 0, 0 };
    Kernel k("npTriMesh.spv", "ctTriNormal",
             { (uint32_t)sizeof(TriIn), (uint32_t)sizeof(MeshScale), (uint32_t)sizeof(F4), (uint32_t)sizeof(F4) }, sizeof(TmParams));
    k.dispatch(GROUPS, { bT, bS, bN, bB }, prm);
    dev->flush();

    const auto g = downloadBuffer<F4>(dev, bN, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
        if (!almostEqual(g[i].x, ref[i].x, 1e-4f, 1e-4f) || !almostEqual(g[i].y, ref[i].y, 1e-4f, 1e-4f) || !almostEqual(g[i].z, ref[i].z, 1e-4f, 1e-4f)) ++mm;
    PXG_CHECK_MSG(mm == 0, "ctTriNormal: %zu normal mismatches", mm);

    dev->destroyBuffer(bT); dev->destroyBuffer(bS); dev->destroyBuffer(bN); dev->destroyBuffer(bB);
}

PXG_TEST(npTriMesh_triBarycentric)
{
    std::mt19937 rng(71029);
    const uint32_t N = 12000;
    std::vector<TriIn> tri(N);
    std::vector<MeshScale> ms(N);
    std::vector<F4> ref(N);
    std::uniform_real_distribution<float> uab(-0.3f, 1.3f);
    size_t interior = 0, exterior = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        V3 sv0, sv1, sv2; makeCase(rng, tri[i], ms[i], sv0, sv1, sv2);
        // query point as a barycentric combination of the SCALED triangle
        const float s = uab(rng), t = uab(rng);
        const V3 e1 = sub(sv1, sv0), e2 = sub(sv2, sv0);
        const V3 p = { sv0.x + e1.x * s + e2.x * t, sv0.y + e1.y * s + e2.y * t, sv0.z + e1.z * s + e2.z * t };
        tri[i].p = f4(p);
        // reference Cramer's-rule barycentric
        const V3 w0 = sub(sv1, sv0), w1 = sub(sv2, sv0), w2 = sub(p, sv0);
        const float d00 = dot3(w0, w0), d01 = dot3(w0, w1), d11 = dot3(w1, w1), d20 = dot3(w2, w0), d21 = dot3(w2, w1);
        const float denom = d00 * d11 - d01 * d01;
        const float v = (d11 * d20 - d01 * d21) / denom;
        const float w = (d00 * d21 - d01 * d20) / denom;
        ref[i] = { 1.0f - v - w, v, w, 0.0f };
        if (v >= 0 && w >= 0 && (v + w) <= 1) ++interior; else ++exterior;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<F4> normDummy(N), outInit(N);
    PxgBufferHandle bT = createBufferFrom(dev, tri);
    PxgBufferHandle bS = createBufferFrom(dev, ms);
    PxgBufferHandle bN = createBufferFrom(dev, normDummy);
    PxgBufferHandle bB = createBufferFrom(dev, outInit);

    TmParams prm = { N, 0, 0, 0 };
    Kernel k("npTriMesh.spv", "ctTriBarycentric",
             { (uint32_t)sizeof(TriIn), (uint32_t)sizeof(MeshScale), (uint32_t)sizeof(F4), (uint32_t)sizeof(F4) }, sizeof(TmParams));
    k.dispatch(GROUPS, { bT, bS, bN, bB }, prm);
    dev->flush();

    const auto g = downloadBuffer<F4>(dev, bB, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
        if (!almostEqual(g[i].x, ref[i].x, 1e-3f, 1e-3f) || !almostEqual(g[i].y, ref[i].y, 1e-3f, 1e-3f) ||
            !almostEqual(g[i].z, ref[i].z, 1e-3f, 1e-3f) || g[i].w != 0.0f) ++mm;
    PXG_CHECK_MSG(mm == 0, "ctTriBarycentric: %zu barycentric mismatches", mm);
    PXG_CHECK_MSG(interior > N / 8 && exterior > N / 4, "ctTriBarycentric: coverage thin (interior=%zu exterior=%zu)", interior, exterior);

    dev->destroyBuffer(bT); dev->destroyBuffer(bS); dev->destroyBuffer(bN); dev->destroyBuffer(bB);
}

} // namespace
