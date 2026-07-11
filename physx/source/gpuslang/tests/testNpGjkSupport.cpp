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

// Verification of npGjkSupport.slang: ctGjkSupportIndex — supportIndex
// (epa.cuh :281-302), the GJK/EPA support-mapping warp argmax.  One thread per
// test; the winning index is compared exactly against a CPU transcription of the
// per-32-chunk structure (plain max == butterfly fmaxf, lowest lane with d==max,
// strict cross-chunk `>` so the earliest chunk wins ties).  Integer vertex
// coordinates and integer directions keep every dot product exact, so the max
// and every equality — hence the lowest-index tie-break — are bit-for-bit
// reproducible.  Fixtures pin: unique-max clouds, coplanar-face TIES (lowest
// index wins), multi-chunk sets (>32 verts) including cross-chunk ties (earliest
// chunk wins), and chunk-boundary counts (1, 32, 33, 64).

#include "pxgTestFramework.h"

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct V3 { float x, y, z; };
struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };
struct GsIn { F4 dir; U4 cfg; };
struct GsParams { uint32_t numTests, maxV, pad0, pad1; };

static const uint32_t MAXV = 64u;
static const float FLT_MAX_R = 3.402823466e38f;

static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// exact transcription of supportIndex's per-32-chunk structure
static uint32_t supportIndexRef(const std::vector<V3>& v, uint32_t nb, V3 dir)
{
    uint32_t bestN = 0xFFFFFFFFu;
    float bestDist = -FLT_MAX_R;
    for (uint32_t i = 0; i < nb; i += 32)
    {
        float d[32];
        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            const uint32_t n = i + lane;
            d[lane] = (n < nb) ? dot3(dir, v[n]) : -FLT_MAX_R;
        }
        float chunkMax = -FLT_MAX_R;
        for (uint32_t lane = 0; lane < 32; ++lane) chunkMax = std::max(chunkMax, d[lane]);
        if (chunkMax > bestDist)
        {
            uint32_t lowestLane = 0;
            for (uint32_t lane = 0; lane < 32; ++lane) if (d[lane] == chunkMax) { lowestLane = lane; break; }
            bestDist = chunkMax;
            bestN = i + lowestLane;
        }
    }
    return bestN;
}

PXG_TEST(npGjkSupport_support)
{
    std::mt19937 rng(1160116);

    std::vector<F4> vertsFlat;
    std::vector<GsIn> in;
    std::vector<uint32_t> ref;

    size_t nTie = 0, nMultiChunk = 0, nCrossTie = 0, nUnique = 0;

    auto emit = [&](const std::vector<V3>& verts, V3 dir)
    {
        const uint32_t nb = (uint32_t)verts.size();
        for (uint32_t i = 0; i < MAXV; ++i)
            vertsFlat.push_back(i < nb ? F4{ verts[i].x, verts[i].y, verts[i].z, 0 } : F4{ 0, 0, 0, 0 });
        GsIn e; e.dir = { dir.x, dir.y, dir.z, 0 }; e.cfg = { nb, 0, 0, 0 };
        in.push_back(e);
        ref.push_back(supportIndexRef(verts, nb, dir));
    };

    auto randDir = [&]() -> V3 { return { float(int(rng() % 7) - 3), float(int(rng() % 7) - 3), float(int(rng() % 7) - 3) }; };

    // family A: random integer clouds (usually unique max)
    for (uint32_t iter = 0; iter < 700; ++iter)
    {
        const uint32_t nb = 1u + rng() % 32u;
        std::vector<V3> v;
        for (uint32_t i = 0; i < nb; ++i)
            v.push_back({ float(int(rng() % 41) - 20), float(int(rng() % 41) - 20), float(int(rng() % 41) - 20) });
        V3 dir = randDir();
        if (dir.x == 0 && dir.y == 0 && dir.z == 0) dir.x = 1;
        emit(v, dir);
        ++nUnique;
    }

    // family B: coplanar-face TIES (many verts share the exact max projection)
    for (uint32_t iter = 0; iter < 400; ++iter)
    {
        // dir axis-aligned so the max projection = the coordinate; put several
        // verts at the same extreme coordinate (a face) => a projection tie
        const uint32_t axis = rng() % 3;
        V3 dir = { 0, 0, 0 }; (&dir.x)[axis] = (rng() % 2) ? 1.0f : -1.0f;
        const uint32_t nb = 4u + rng() % 24u;
        const float extreme = float((rng() % 2) ? 15 : -15);
        std::vector<V3> v(nb);
        for (uint32_t i = 0; i < nb; ++i)
            v[i] = { float(int(rng() % 21) - 10), float(int(rng() % 21) - 10), float(int(rng() % 21) - 10) };
        // force >=3 verts onto the extreme plane (same projection along dir)
        const uint32_t nExtreme = 3u + rng() % 4u;
        for (uint32_t k = 0; k < nExtreme && k < nb; ++k)
            (&v[rng() % nb].x)[axis] = (&dir.x)[axis] > 0 ? extreme : -extreme;
        // ensure at least one vert sits exactly at the winning extreme in dir's sense
        (&v[rng() % nb].x)[axis] = (&dir.x)[axis] > 0 ? std::abs(extreme) : -std::abs(extreme);
        emit(v, dir);
        ++nTie;
    }

    // family C: multi-chunk (>32 verts) incl. cross-chunk ties (earliest wins)
    for (uint32_t iter = 0; iter < 400; ++iter)
    {
        const uint32_t nb = 33u + rng() % (MAXV - 33u + 1u);
        const uint32_t axis = rng() % 3;
        V3 dir = { 0, 0, 0 }; (&dir.x)[axis] = 1.0f;
        std::vector<V3> v(nb);
        for (uint32_t i = 0; i < nb; ++i)
            v[i] = { float(int(rng() % 21) - 10), float(int(rng() % 21) - 10), float(int(rng() % 21) - 10) };
        // seed the SAME global-max projection in BOTH chunk 0 and a later chunk
        const float ext = 14.0f;
        const uint32_t a = rng() % 32u;                 // chunk 0
        const uint32_t b = 32u + rng() % (nb - 32u);    // chunk 1+
        (&v[a].x)[axis] = ext;
        (&v[b].x)[axis] = ext;
        emit(v, dir);
        ++nMultiChunk;
        if (ref.back() == a) ++nCrossTie; // earliest-chunk winner observed
    }

    // family D: chunk-boundary counts 1, 32, 33, 64
    for (uint32_t iter = 0; iter < 120; ++iter)
    {
        const uint32_t counts[4] = { 1u, 32u, 33u, 64u };
        const uint32_t nb = counts[rng() % 4];
        std::vector<V3> v(nb);
        for (uint32_t i = 0; i < nb; ++i)
            v[i] = { float(int(rng() % 31) - 15), float(int(rng() % 31) - 15), float(int(rng() % 31) - 15) };
        V3 dir = randDir(); if (dir.x == 0 && dir.y == 0 && dir.z == 0) dir.y = 1;
        emit(v, dir);
        if (nb > 32) ++nMultiChunk;
    }

    const uint32_t N = (uint32_t)in.size();
    PxgSlangComputeDevice* dev = testDevice();
    std::vector<uint32_t> outInit(N, 0xEEEEEEEEu);

    PxgBufferHandle bV = createBufferFrom(dev, vertsFlat);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    GsParams prm = { N, MAXV, 0, 0 };
    Kernel k("npGjkSupport.spv", "ctGjkSupportIndex",
             { (uint32_t)sizeof(F4), (uint32_t)sizeof(GsIn), 4 }, sizeof(GsParams));
    k.dispatch((N + 63) / 64, { bV, bIn, bOut }, prm);
    dev->flush();

    const auto gOut = downloadBuffer<uint32_t>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t t = 0; t < N; ++t) if (gOut[t] != ref[t]) ++mm;

    PXG_CHECK_MSG(mm == 0, "gjkSupport: %zu mismatches", mm);
    PXG_CHECK_MSG(nUnique > 400 && nTie > 200 && nMultiChunk > 300 && nCrossTie > 50,
                  "gjkSupport coverage thin (unique=%zu tie=%zu multiChunk=%zu crossTie=%zu)",
                  nUnique, nTie, nMultiChunk, nCrossTie);

    for (PxgBufferHandle h : { bV, bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
