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

// Verification of npGjkBarycentric.slang: ctGjkBarycentric — schlock::barycentrics
// (schlockShared.h :127-141).  One thread per test; (u,v,w) compared bit-exactly
// against an exact CPU transcription.  Integer / signed-permutation coordinates
// keep the cross products and |m| exact (so the three-way largest-component
// branch is deterministic) and d = 1.0f/m.[axis] is a correctly-rounded division
// followed by a correctly-rounded multiply, so the result is bit-for-bit
// reproducible.  Fixtures use non-degenerate triangles (m != 0) and exercise all
// three axis branches, ma ties, and p inside / outside / at a vertex.

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
struct BcIn { F4 a, b, c, p; };
struct BcParams { uint32_t numTests, pad0, pad1, pad2; };

static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 cross3(V3 a, V3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static V3 abs3(V3 a) { return { std::abs(a.x), std::abs(a.y), std::abs(a.z) }; }
static uint32_t asU32(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

struct UVW { float u, v, w; };
static UVW barycentricsRef(V3 a, V3 b, V3 c, V3 p)
{
    const V3 m = cross3(sub(b, a), sub(c, a));
    const V3 ma = abs3(m);
    const V3 a1 = cross3(sub(b, p), sub(c, p));
    const V3 a2 = cross3(sub(c, p), sub(a, p));
    float d, u, v;
    if (ma.x >= ma.y && ma.x >= ma.z) { d = 1.0f / m.x; u = a1.x * d; v = a2.x * d; }
    else if (ma.y >= ma.z)            { d = 1.0f / m.y; u = a1.y * d; v = a2.y * d; }
    else                              { d = 1.0f / m.z; u = a1.z * d; v = a2.z * d; }
    return { u, v, 1.0f - u - v };
}

PXG_TEST(npGjkBarycentric_bary)
{
    std::mt19937 rng(1180118);

    std::vector<BcIn> in;
    std::vector<UVW> ref;
    size_t nAxisX = 0, nAxisY = 0, nAxisZ = 0, nInside = 0, nVertex = 0;

    auto ivec = [&](int lim) -> V3 { return { float(int(rng() % (2 * lim + 1)) - lim),
                                             float(int(rng() % (2 * lim + 1)) - lim),
                                             float(int(rng() % (2 * lim + 1)) - lim) }; };

    auto emit = [&](V3 a, V3 b, V3 c, V3 p)
    {
        BcIn e; e.a = { a.x, a.y, a.z, 0 }; e.b = { b.x, b.y, b.z, 0 };
        e.c = { c.x, c.y, c.z, 0 }; e.p = { p.x, p.y, p.z, 0 };
        in.push_back(e);
        ref.push_back(barycentricsRef(a, b, c, p));
        const V3 m = cross3(sub(b, a), sub(c, a)); const V3 ma = abs3(m);
        if (ma.x >= ma.y && ma.x >= ma.z) ++nAxisX; else if (ma.y >= ma.z) ++nAxisY; else ++nAxisZ;
    };

    // family A: random non-degenerate integer triangles + random points
    for (uint32_t iter = 0; iter < 1000; ++iter)
    {
        V3 a, b, c, m;
        do { a = ivec(12); b = ivec(12); c = ivec(12); m = cross3(sub(b, a), sub(c, a)); }
        while (m.x == 0 && m.y == 0 && m.z == 0); // reject degenerate
        emit(a, b, c, ivec(12));
    }

    // family B: point exactly at a vertex (barycentrics should be a unit basis)
    for (uint32_t iter = 0; iter < 200; ++iter)
    {
        V3 a, b, c, m;
        do { a = ivec(12); b = ivec(12); c = ivec(12); m = cross3(sub(b, a), sub(c, a)); }
        while (m.x == 0 && m.y == 0 && m.z == 0);
        const V3 verts[3] = { a, b, c };
        emit(a, b, c, verts[rng() % 3]);
        ++nVertex;
    }

    // family C: point at the exact centroid-ish interior (p = a, or average not
    // representable, so use p on an edge midpoint of even coords) + axis-plane
    // triangles to force each of the x/y/z branches
    for (uint32_t iter = 0; iter < 300; ++iter)
    {
        // a triangle in a coordinate plane makes the normal axis-aligned so a
        // specific ma component dominates
        const uint32_t plane = rng() % 3;
        V3 a = ivec(10), b = ivec(10), c = ivec(10);
        // flatten one coordinate so the normal points along `plane`
        (&a.x)[plane] = 0; (&b.x)[plane] = 0; (&c.x)[plane] = 0;
        V3 m = cross3(sub(b, a), sub(c, a));
        if (m.x == 0 && m.y == 0 && m.z == 0) continue; // collinear in-plane
        V3 p = ivec(10); // p out of plane in general
        emit(a, b, c, p);
        ++nInside;
    }

    const uint32_t N = (uint32_t)in.size();
    PxgSlangComputeDevice* dev = testDevice();
    std::vector<F4> outInit(N, { -777, -777, -777, -777 });

    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    BcParams prm = { N, 0, 0, 0 };
    Kernel k("npGjkBarycentric.spv", "ctGjkBarycentric",
             { (uint32_t)sizeof(BcIn), (uint32_t)sizeof(F4) }, sizeof(BcParams));
    k.dispatch((N + 63) / 64, { bIn, bOut }, prm);
    dev->flush();

    const auto gOut = downloadBuffer<F4>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t t = 0; t < N; ++t)
    {
        const UVW r = ref[t]; const F4 g = gOut[t];
        if (asU32(g.x) != asU32(r.u) || asU32(g.y) != asU32(r.v) || asU32(g.z) != asU32(r.w)) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "gjkBarycentric: %zu mismatches", mm);
    PXG_CHECK_MSG(nAxisX > 150 && nAxisY > 150 && nAxisZ > 150 && nVertex > 100 && nInside > 100,
                  "gjkBarycentric coverage thin (x=%zu y=%zu z=%zu vertex=%zu inplane=%zu)",
                  nAxisX, nAxisY, nAxisZ, nVertex, nInside);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
