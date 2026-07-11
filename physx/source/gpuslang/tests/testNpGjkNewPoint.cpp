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

// Verification of npGjkNewPoint.slang: ctGjkNewPoint — newPoint (epa.cuh
// :306-325), the EPA rare-case new-support-point pick.  One thread per test; the
// chosen vertex and packed (indexB<<8)|indexA are compared bit-exactly against a
// CPU transcription that reuses the exact supportIndex reference.  Integer
// vertex/dir/ref coordinates keep every dot, Minkowski subtraction and abs exact,
// so the strict `>` pick and the tie (else -> vn) branch are reproducible.
// Fixtures drive the +/- pick both ways, exact ties, and multi-chunk vertex sets.

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
struct NpIn { F4 dir, ref; U4 cfg; };
struct NpOut { F4 vertex; uint32_t indices, pad0, pad1, pad2; };
struct NpParams { uint32_t numTests, maxA, maxB, pad0; };

static const uint32_t MAXA = 48u;
static const uint32_t MAXB = 48u;
static const float FLT_MAX_R = 3.402823466e38f;

static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 neg(V3 a) { return { -a.x, -a.y, -a.z }; }
static uint32_t asU32(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

static uint32_t supportIndexRef(const std::vector<V3>& v, uint32_t nb, V3 dir)
{
    uint32_t bestN = 0xFFFFFFFFu; float bestDist = -FLT_MAX_R;
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
            uint32_t ll = 0;
            for (uint32_t lane = 0; lane < 32; ++lane) if (d[lane] == chunkMax) { ll = lane; break; }
            bestDist = chunkMax; bestN = i + ll;
        }
    }
    return bestN;
}

struct NpRef { V3 vertex; uint32_t indices; };
static NpRef newPointRef(const std::vector<V3>& vA, const std::vector<V3>& vB, V3 dir, V3 ref)
{
    const uint32_t nbA = (uint32_t)vA.size(), nbB = (uint32_t)vB.size();
    const uint32_t iAp = supportIndexRef(vA, nbA, dir);
    const uint32_t iBp = supportIndexRef(vB, nbB, neg(dir));
    const uint32_t iAn = supportIndexRef(vA, nbA, neg(dir));
    const uint32_t iBn = supportIndexRef(vB, nbB, dir);
    const V3 vp = sub(vA[iAp], vB[iBp]);
    const V3 vn = sub(vA[iAn], vB[iBn]);
    NpRef r;
    if (std::abs(dot3(sub(vp, ref), dir)) > std::abs(dot3(sub(vn, ref), dir)))
    { r.vertex = vp; r.indices = (iBp << 8) | iAp; }
    else
    { r.vertex = vn; r.indices = (iBn << 8) | iAn; }
    return r;
}

PXG_TEST(npGjkNewPoint_newpoint)
{
    std::mt19937 rng(1170117);

    std::vector<F4> vertsFlat;
    std::vector<NpIn> in;
    std::vector<NpRef> ref;

    size_t nPickP = 0, nPickN = 0, nTie = 0, nMultiChunk = 0;

    auto emit = [&](const std::vector<V3>& vA, const std::vector<V3>& vB, V3 dir, V3 refPt)
    {
        for (uint32_t i = 0; i < MAXA; ++i)
            vertsFlat.push_back(i < vA.size() ? F4{ vA[i].x, vA[i].y, vA[i].z, 0 } : F4{ 0, 0, 0, 0 });
        for (uint32_t i = 0; i < MAXB; ++i)
            vertsFlat.push_back(i < vB.size() ? F4{ vB[i].x, vB[i].y, vB[i].z, 0 } : F4{ 0, 0, 0, 0 });
        NpIn e; e.dir = { dir.x, dir.y, dir.z, 0 }; e.ref = { refPt.x, refPt.y, refPt.z, 0 };
        e.cfg = { (uint32_t)vA.size(), (uint32_t)vB.size(), 0, 0 };
        in.push_back(e);
        const NpRef r = newPointRef(vA, vB, dir, refPt);
        ref.push_back(r);
    };

    auto ivec = [&](int lim) -> V3 { return { float(int(rng() % (2 * lim + 1)) - lim),
                                              float(int(rng() % (2 * lim + 1)) - lim),
                                              float(int(rng() % (2 * lim + 1)) - lim) }; };

    // family A: random integer hulls, random dir/ref
    for (uint32_t iter = 0; iter < 900; ++iter)
    {
        const uint32_t nbA = 1u + rng() % 20u, nbB = 1u + rng() % 20u;
        std::vector<V3> vA(nbA), vB(nbB);
        for (auto& p : vA) p = ivec(15);
        for (auto& p : vB) p = ivec(15);
        V3 dir = ivec(4); if (dir.x == 0 && dir.y == 0 && dir.z == 0) dir.x = 1;
        V3 refPt = ivec(6);
        emit(vA, vB, dir, refPt);
        // classify which branch the reference took (for coverage)
        const uint32_t iAp = supportIndexRef(vA, nbA, dir), iBp = supportIndexRef(vB, nbB, neg(dir));
        const uint32_t iAn = supportIndexRef(vA, nbA, neg(dir)), iBn = supportIndexRef(vB, nbB, dir);
        const V3 vp = sub(vA[iAp], vB[iBp]), vn = sub(vA[iAn], vB[iBn]);
        const float ap = std::abs(dot3(sub(vp, refPt), dir)), an = std::abs(dot3(sub(vn, refPt), dir));
        if (ap > an) ++nPickP; else ++nPickN;
        if (ap == an) ++nTie;
    }

    // family B: symmetric hulls about the origin with ref at origin so the +dir
    // and -dir projections tie in magnitude (drives the else branch / tie path)
    for (uint32_t iter = 0; iter < 300; ++iter)
    {
        const uint32_t nb = 2u + rng() % 8u;
        std::vector<V3> vA, vB;
        for (uint32_t i = 0; i < nb; ++i) { V3 p = ivec(10); vA.push_back(p); vA.push_back(neg(p)); }
        for (uint32_t i = 0; i < nb; ++i) { V3 p = ivec(10); vB.push_back(p); vB.push_back(neg(p)); }
        V3 dir = ivec(3); if (dir.x == 0 && dir.y == 0 && dir.z == 0) dir.y = 1;
        emit(vA, vB, dir, V3{ 0, 0, 0 }); // ref = origin -> symmetric magnitudes tie
        ++nTie;
    }

    // family C: multi-chunk (>32 verts on at least one side)
    for (uint32_t iter = 0; iter < 300; ++iter)
    {
        const uint32_t nbA = 33u + rng() % (MAXA - 33u + 1u), nbB = 1u + rng() % 40u;
        std::vector<V3> vA(nbA), vB(nbB);
        for (auto& p : vA) p = ivec(15);
        for (auto& p : vB) p = ivec(15);
        V3 dir = ivec(4); if (dir.x == 0 && dir.y == 0 && dir.z == 0) dir.z = 1;
        emit(vA, vB, dir, ivec(6));
        ++nMultiChunk;
    }

    const uint32_t N = (uint32_t)in.size();
    PxgSlangComputeDevice* dev = testDevice();
    std::vector<NpOut> outInit(N);
    for (auto& o : outInit) { o.vertex = { -777, -777, -777, -777 }; o.indices = 0xEEEEEEEEu; o.pad0 = o.pad1 = o.pad2 = 0; }

    PxgBufferHandle bV = createBufferFrom(dev, vertsFlat);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    NpParams prm = { N, MAXA, MAXB, 0 };
    Kernel k("npGjkNewPoint.spv", "ctGjkNewPoint",
             { (uint32_t)sizeof(F4), (uint32_t)sizeof(NpIn), (uint32_t)sizeof(NpOut) }, sizeof(NpParams));
    k.dispatch((N + 63) / 64, { bV, bIn, bOut }, prm);
    dev->flush();

    const auto gOut = downloadBuffer<NpOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t t = 0; t < N; ++t)
    {
        const NpRef& r = ref[t]; const NpOut& g = gOut[t];
        if (g.indices != r.indices) ++mm;
        if (asU32(g.vertex.x) != asU32(r.vertex.x) || asU32(g.vertex.y) != asU32(r.vertex.y) ||
            asU32(g.vertex.z) != asU32(r.vertex.z)) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "gjkNewPoint: %zu mismatches", mm);
    PXG_CHECK_MSG(nPickP > 200 && nPickN > 200 && nTie > 100 && nMultiChunk > 200,
                  "gjkNewPoint coverage thin (pickP=%zu pickN=%zu tie=%zu multiChunk=%zu)",
                  nPickP, nPickN, nTie, nMultiChunk);

    for (PxgBufferHandle h : { bV, bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
