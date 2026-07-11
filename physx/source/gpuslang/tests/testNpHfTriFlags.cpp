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

// Verification of npHfTriFlags.slang: ctHfTriFlags — the edge-classification core
// of computeTriangleFlags (convexHeightfield.cu :565-614).  One thread per test;
// the 8-bit triFlags is compared exactly against an exact CPU transcription.
// Integer current/adjacent triangles make projD an exact integer dot (so the
// `projD < 0` branch is deterministic), and a normalized triNormal input plus the
// normalize (v*(1/sqrt(dot))) + explicit dotX give a bit-exact proj, so the
// `proj < 0.997f` compare is reproducible.  Fixtures exercise all branches
// (proj<0.997 active, boundary active, silhouette, coplanar-inactive), both projD
// signs, both boundaryCollisions values, and adjacency-valid masks 0..7.

#include "pxgTestFramework.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct V3 { float x, y, z; };
struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };
struct HfIn { F4 cur[3]; F4 triNormal; F4 adj[9]; U4 cfg; };
struct HfParams { uint32_t numTests, pad0, pad1, pad2; };

static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static V3 muls(V3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
static V3 normalizeF(V3 v) { const float m2 = dot3(v, v); return m2 > 0 ? muls(v, 1.0f / std::sqrt(m2)) : v; }

static uint32_t triFlagsRef(const V3 cur[3], V3 triNormal, const V3 adj[3][3], uint32_t adjValidMask, bool boundaryCollisions)
{
    const uint32_t nextInd[3] = { 2, 0, 1 };
    uint32_t triFlags = 0;
    for (uint32_t a = 0; a < 3; ++a)
    {
        if (((adjValidMask >> a) & 1u) == 0u) continue;
        const V3 av0 = adj[a][0], av1 = adj[a][1], av2 = adj[a][2];
        V3 adjNormal = cross3(sub(av1, av0), sub(av2, av0));
        const uint32_t otherIndex = nextInd[a];
        const float projD = dot3(adjNormal, sub(cur[otherIndex], av0));
        if (projD < 0.0f)
        {
            const float m2 = dot3(adjNormal, adjNormal);
            if (m2 > 0.0f) adjNormal = muls(adjNormal, 1.0f / std::sqrt(m2));
            const float proj = dot3(adjNormal, triNormal);
            if (proj < 0.997f) triFlags |= (1u << (a + 3));
        }
        else if (boundaryCollisions) triFlags |= (1u << (a + 3));
        else triFlags |= (1u << a);
    }
    return triFlags;
}

PXG_TEST(npHfTriFlags_flags)
{
    std::mt19937 rng(1190119);

    std::vector<HfIn> in;
    std::vector<uint32_t> ref;
    size_t nActiveProj = 0, nBoundary = 0, nSilhouette = 0, nInactive = 0, nProjDneg = 0, nProjDpos = 0;

    auto ivec = [&](int lim) -> V3 { return { float(int(rng() % (2 * lim + 1)) - lim),
                                             float(int(rng() % (2 * lim + 1)) - lim),
                                             float(int(rng() % (2 * lim + 1)) - lim) }; };
    auto nonDegenTri = [&](int lim, V3 out[3]) {
        V3 n;
        do { out[0] = ivec(lim); out[1] = ivec(lim); out[2] = ivec(lim); n = cross3(sub(out[1], out[0]), sub(out[2], out[0])); }
        while (n.x == 0 && n.y == 0 && n.z == 0);
    };

    auto emit = [&](const V3 cur[3], V3 triNormal, const V3 adj[3][3], uint32_t mask, bool boundary)
    {
        HfIn e;
        for (int i = 0; i < 3; ++i) e.cur[i] = { cur[i].x, cur[i].y, cur[i].z, 0 };
        e.triNormal = { triNormal.x, triNormal.y, triNormal.z, 0 };
        for (int a = 0; a < 3; ++a) for (int vi = 0; vi < 3; ++vi)
            e.adj[a * 3 + vi] = { adj[a][vi].x, adj[a][vi].y, adj[a][vi].z, 0 };
        e.cfg = { mask, boundary ? 1u : 0u, 0, 0 };
        in.push_back(e);
        ref.push_back(triFlagsRef(cur, triNormal, adj, mask, boundary));
        // classify branches (coverage)
        const uint32_t nextInd[3] = { 2, 0, 1 };
        for (uint32_t a = 0; a < 3; ++a)
        {
            if (((mask >> a) & 1u) == 0u) continue;
            V3 aN = cross3(sub(adj[a][1], adj[a][0]), sub(adj[a][2], adj[a][0]));
            const float projD = dot3(aN, sub(cur[nextInd[a]], adj[a][0]));
            if (projD < 0.0f)
            {
                ++nProjDneg;
                V3 aNn = normalizeF(aN);
                if (dot3(aNn, triNormal) < 0.997f) ++nActiveProj; else ++nInactive;
            }
            else { ++nProjDpos; if (boundary) ++nBoundary; else ++nSilhouette; }
        }
    };

    // family A: random non-degenerate triangles, random normalized triNormal
    for (uint32_t iter = 0; iter < 900; ++iter)
    {
        V3 cur[3]; nonDegenTri(12, cur);
        V3 adj[3][3]; for (int a = 0; a < 3; ++a) nonDegenTri(12, adj[a]);
        V3 tn = normalizeF(ivec(8)); if (dot3(tn, tn) == 0) tn = { 1, 0, 0 };
        const uint32_t mask = rng() % 8u;
        emit(cur, tn, adj, mask, (rng() % 2) != 0);
    }

    // family B: coplanar-ish adjacent for edge 0 -> proj ~ 1 (inactive when projD<0).
    // set triNormal = normalize(adjNormal[0]) and give edge 0 a projD<0 by placing
    // cur[nextInd[0]=2] on the negative side of the adj plane.
    for (uint32_t iter = 0; iter < 300; ++iter)
    {
        V3 adj[3][3];
        for (int a = 0; a < 3; ++a) nonDegenTri(10, adj[a]);
        const V3 aN0 = cross3(sub(adj[0][1], adj[0][0]), sub(adj[0][2], adj[0][0]));
        const V3 tn = normalizeF(aN0); // triNormal parallel to edge-0 adj normal -> proj ~ +-1
        // build a current triangle whose vert 2 (=nextInd[0]) is behind adj[0] plane
        V3 cur[3]; nonDegenTri(6, cur);
        cur[2] = sub(adj[0][0], muls(aN0, 1.0f)); // cur[2] = av0 - aN0  -> projD = aN0.(cur2-av0) = -|aN0|^2 < 0
        // keep triangle non-degenerate
        V3 n = cross3(sub(cur[1], cur[0]), sub(cur[2], cur[0]));
        if (n.x == 0 && n.y == 0 && n.z == 0) cur[1].x += 1;
        emit(cur, tn, adj, 1u | (rng() % 8u), (rng() % 2) != 0);
    }

    // family C: pin the boundary/silhouette split with projD>=0 edges.  Give each
    // edge an adj normal that points so projD>=0 (cur vert in front), toggling
    // boundaryCollisions.
    for (uint32_t iter = 0; iter < 300; ++iter)
    {
        V3 cur[3]; nonDegenTri(8, cur);
        V3 adj[3][3];
        const uint32_t nextInd[3] = { 2, 0, 1 };
        for (int a = 0; a < 3; ++a)
        {
            nonDegenTri(8, adj[a]);
            // flip winding if needed so projD >= 0 for cur[nextInd[a]]
            V3 aN = cross3(sub(adj[a][1], adj[a][0]), sub(adj[a][2], adj[a][0]));
            if (dot3(aN, sub(cur[nextInd[a]], adj[a][0])) < 0.0f) std::swap(adj[a][1], adj[a][2]);
        }
        V3 tn = normalizeF(ivec(8)); if (dot3(tn, tn) == 0) tn = { 0, 1, 0 };
        emit(cur, tn, adj, 7u, (iter % 2) != 0);
    }

    const uint32_t N = (uint32_t)in.size();
    PxgSlangComputeDevice* dev = testDevice();
    std::vector<uint32_t> outInit(N, 0xEEEEEEEEu);

    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    HfParams prm = { N, 0, 0, 0 };
    Kernel k("npHfTriFlags.spv", "ctHfTriFlags",
             { (uint32_t)sizeof(HfIn), 4 }, sizeof(HfParams));
    k.dispatch((N + 63) / 64, { bIn, bOut }, prm);
    dev->flush();

    const auto gOut = downloadBuffer<uint32_t>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t t = 0; t < N; ++t) if (gOut[t] != ref[t]) ++mm;

    PXG_CHECK_MSG(mm == 0, "hfTriFlags: %zu mismatches", mm);
    PXG_CHECK_MSG(nActiveProj > 200 && nBoundary > 100 && nSilhouette > 100 && nInactive > 50 &&
                  nProjDneg > 200 && nProjDpos > 200,
                  "hfTriFlags coverage thin (activeProj=%zu boundary=%zu silhouette=%zu inactive=%zu projDneg=%zu projDpos=%zu)",
                  nActiveProj, nBoundary, nSilhouette, nInactive, nProjDneg, nProjDpos);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
