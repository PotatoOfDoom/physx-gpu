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

// Verification of npConvexPolyClip.slang: ctPolyClip — the rigid convex-polygon
// vs triangle warp-SIMD clipper.  One thread simulates the whole 32-lane warp,
// so the contact buffer order is fully deterministic (lane order within each
// addContacts call, calls in phase order); the buffer is compared
// POSITION-FOR-POSITION bit-exactly against an exact CPU transcription (identical
// per-lane arrays / reductions / ballots / addContacts scan).  Axis-aligned
// square/rect poly0 and a coplanar triangle offset along the axis keep the
// projections (s = +-t/dnom) exact and edge-edge crossings on representable
// coordinates; every arithmetic op has an identical counterpart in the shader,
// so the match is bit-exact regardless of whether a crossing is "clean".

#include "pxgTestFramework.h"

#include <algorithm>
#include <cfloat>
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
struct PcIn { F4 plane0, plane1, axis; U4 cfg; };
struct PcParams { uint32_t numTests, maxVerts0, pad0, pad1; };

static const uint32_t NUM_TMP = 32u;
static const uint32_t MAXV = 8u; // max poly0 verts per test in fixtures
static const float PC_MAX = 3.402823466e38f;
static const uint32_t POISONU = 0x7F800001u; // a quiet-NaN-ish sentinel bit pattern

static float dotX(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 crossX(V3 a, V3 b)
{
    return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}
static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 mulAdd(V3 p, float s, V3 d) { return { p.x + s * d.x, p.y + s * d.y, p.z + s * d.z }; }
static float planeDist(F4 pl, V3 p) { return pl.x * p.x + pl.y * p.y + pl.z * p.z + pl.w; }
static uint32_t popc(uint32_t x) { uint32_t c = 0; while (x) { c += x & 1u; x >>= 1; } return c; }
static uint32_t lowestSetIndex(uint32_t x) { uint32_t i = 0; while ((x & 1u) == 0u) { x >>= 1; ++i; } return i; }
static uint32_t asU32(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

// exact CPU transcription of ctPolyClip; appends (pos,sep) to `out`
struct WarpRef { std::vector<F4> out; uint32_t count; };
static void addContactsRef(WarpRef& wp, uint32_t flags, const V3 pos[32], const float sep[32], float maxSep)
{
    uint32_t keep = 0u;
    for (uint32_t L = 0; L < 32; ++L)
        if ((flags & (1u << L)) && !(sep[L] > maxSep)) keep |= (1u << L);
    const uint32_t entry = wp.count;
    uint32_t added = 0u;
    for (uint32_t L = 0; L < 32; ++L)
    {
        if ((keep & (1u << L)) == 0u) continue;
        const uint32_t index = popc(keep & ((1u << L) - 1u)) + entry;
        if (index < NUM_TMP) { wp.out[index] = { pos[L].x, pos[L].y, pos[L].z, sep[L] }; ++added; }
    }
    wp.count = entry + added;
}

static WarpRef polyClipRef(F4 plane0, F4 plane1, V3 axis, float maxSep, uint32_t nbVerts0,
                           uint32_t initContacts, const std::vector<V3>& poly0, const V3 tri[3])
{
    WarpRef wp; wp.out.assign(NUM_TMP, { -777, -777, -777, -777 }); wp.count = initContacts;
    V3 v0[32], v1[32];
    for (uint32_t L = 0; L < 32; ++L)
    {
        v0[L] = (L < nbVerts0) ? poly0[L] : V3{ 0, 0, 0 };
        v1[L] = (L < 3u) ? tri[L] : V3{ 0, 0, 0 };
    }
    const float dnom0 = dotX(V3{ plane1.x, plane1.y, plane1.z }, axis);
    const float dnom1 = dotX(V3{ plane0.x, plane0.y, plane0.z }, axis);
    float s0[32], s1[32];
    for (uint32_t L = 0; L < 32; ++L)
    {
        const float t0 = planeDist(plane1, v0[L]);
        const float t1 = planeDist(plane0, v1[L]);
        s0[L] = (t0 > maxSep || dnom0 == 0.0f) ? PC_MAX : -t0 / dnom0;
        s1[L] = (t1 > maxSep || dnom1 == 0.0f) ? PC_MAX : t1 / dnom1;
    }
    V3 e0[32], e0planeN[32]; float e0planeD[32];
    for (uint32_t L = 0; L < 32; ++L)
    {
        const uint32_t nxt = ((L + 1u == nbVerts0) ? 0u : (L + 1u)) & 31u; // match CUDA mod-32 shfl, avoid OOB
        e0[L] = sub(v0[nxt], v0[L]);
        e0planeN[L] = crossX(e0[L], axis);
        e0planeD[L] = dotX(e0planeN[L], v0[L]);
    }
    uint32_t in1[32], out1[32];
    for (uint32_t L = 0; L < 32; ++L) { in1[L] = 0; out1[L] = 0; }
    for (uint32_t j = 0; j < 3; ++j)
    {
        const uint32_t bj = 1u << j;
        for (uint32_t L = 0; L < 32; ++L)
        {
            const float d = dotX(v1[j], e0planeN[L]) - e0planeD[L];
            if (d < 0.0f) in1[L] |= bj;
            if (d > 0.0f) out1[L] |= bj;
        }
    }
    for (uint32_t L = 0; L < 32; ++L) if (L >= nbVerts0) { in1[L] = 0; out1[L] = 0; }
    uint32_t band = 0xffffffffu;
    for (uint32_t L = 0; L < 32; ++L) band &= ~out1[L];
    { V3 pos[32]; float sep[32]; for (uint32_t L = 0; L < 32; ++L) { pos[L] = mulAdd(v1[L], -s1[L], axis); sep[L] = s1[L]; } addContactsRef(wp, band & 7u, pos, sep, maxSep); }
    bool anyOut1 = false; for (uint32_t L = 0; L < 32; ++L) if (out1[L] != 0u) anyOut1 = true;
    if (!anyOut1) return wp;
    uint32_t e1cross[32];
    for (uint32_t L = 0; L < 32; ++L)
    {
        const uint32_t i1 = in1[L] | (in1[L] << 3);
        const uint32_t o1 = out1[L] | (out1[L] << 3);
        e1cross[L] = ((i1 & (o1 >> 1)) | ((i1 >> 1) & o1)) & 7u;
    }
    V3 e1[32], e1planeN[32]; float e1planeD[32];
    for (uint32_t L = 0; L < 32; ++L)
    {
        const uint32_t nxt = ((L + 1u == 3u) ? 0u : (L + 1u)) & 31u; // match CUDA mod-32 shfl, avoid OOB
        e1[L] = sub(v1[nxt], v1[L]);
        e1planeN[L] = crossX(axis, e1[L]);
        e1planeD[L] = dotX(e1planeN[L], v1[L]);
    }
    uint32_t in0[32], out0[32];
    for (uint32_t L = 0; L < 32; ++L) { in0[L] = 0; out0[L] = 0; }
    for (uint32_t j = 0; j < nbVerts0; ++j)
    {
        uint32_t inb = 0, outb = 0;
        for (uint32_t L = 0; L < 3; ++L)
        {
            const float d = dotX(v0[j], e1planeN[L]) - e1planeD[L];
            if (d < 0.0f) inb |= (1u << L);
            if (d > 0.0f) outb |= (1u << L);
        }
        in0[j] = inb; out0[j] = outb;
    }
    for (uint32_t L = 0; L < 32; ++L) { in0[L] &= 7u; out0[L] &= 7u; }
    uint32_t flag2 = 0u;
    for (uint32_t L = 0; L < nbVerts0; ++L) if (out0[L] == 0u) flag2 |= (1u << L);
    flag2 &= (1u << nbVerts0) - 1u;
    { V3 pos[32]; float sep[32]; for (uint32_t L = 0; L < 32; ++L) { pos[L] = v0[L]; sep[L] = s0[L]; } addContactsRef(wp, flag2, pos, sep, maxSep); }
    bool anyOut0 = false; for (uint32_t L = 0; L < nbVerts0; ++L) if (out0[L] != 0u) anyOut0 = true;
    if (!anyOut0) return wp;
    uint32_t e0cross[32];
    for (uint32_t L = 0; L < 32; ++L)
    {
        const uint32_t nxt = ((L + 1u == nbVerts0) ? 0u : (L + 1u)) & 31u; // match CUDA mod-32 shfl, avoid OOB
        e0cross[L] = (in0[L] & out0[nxt]) | (in0[nxt] & out0[L]);
    }
    uint32_t c[32]; uint32_t a = 0u;
    for (uint32_t L = 0; L < 32; ++L) { c[L] = e0cross[L] & e1cross[L]; a |= c[L]; }
    while (a != 0u)
    {
        const uint32_t index = lowestSetIndex(a);
        const uint32_t lsb = a & (0u - a);
        const V3 m = e1planeN[index];
        const V3 triVert0 = v1[index];
        V3 pos[32]; float sep[32];
        for (uint32_t L = 0; L < 32; ++L)
        {
            const float denom = dotX(m, e0[L]);
            const float lambda = dotX(sub(triVert0, v0[L]), m) / denom;
            const V3 p = mulAdd(v0[L], lambda, e0[L]);
            const float d = -dotX(sub(p, triVert0), V3{ plane1.x, plane1.y, plane1.z });
            sep[L] = (denom == 0.0f || dnom0 == 0.0f) ? PC_MAX : d / dnom0;
            pos[L] = p;
        }
        uint32_t flag3 = 0u;
        for (uint32_t L = 0; L < 32; ++L) if (c[L] & lsb) flag3 |= (1u << L);
        addContactsRef(wp, flag3, pos, sep, maxSep);
        a &= (a - 1u);
    }
    return wp;
}

PXG_TEST(npConvexPolyClip_clip)
{
    std::mt19937 rng(1110111);
    std::vector<F4> poly0Flat, triFlat;
    std::vector<PcIn> in;
    std::vector<WarpRef> ref;
    std::vector<uint32_t> nbList;

    size_t nP1 = 0, nP2 = 0, nP3 = 0, nEmpty = 0;

    auto emit = [&](const std::vector<V3>& poly0, const V3 tri[3], float zOff, float maxSep, uint32_t initC)
    {
        const uint32_t nb = (uint32_t)poly0.size();
        // poly0 at z=0, triangle at z=zOff, axis +z; planes are the shape planes
        const F4 plane0 = { 0, 0, 1, 0 };            // n=+z, d=0 (poly0 at z=0)
        const F4 plane1 = { 0, 0, 1, -zOff };        // n=+z, d=-zOff (triangle at z=zOff)
        const V3 axis = { 0, 0, 1 };
        V3 tri3[3] = { { tri[0].x, tri[0].y, zOff }, { tri[1].x, tri[1].y, zOff }, { tri[2].x, tri[2].y, zOff } };

        for (uint32_t i = 0; i < MAXV; ++i)
            poly0Flat.push_back(i < nb ? F4{ poly0[i].x, poly0[i].y, 0.0f, 0.0f } : F4{ 0, 0, 0, 0 });
        for (uint32_t i = 0; i < 3; ++i) triFlat.push_back({ tri3[i].x, tri3[i].y, tri3[i].z, 0.0f });

        PcIn e;
        e.plane0 = plane0; e.plane1 = plane1;
        e.axis = { axis.x, axis.y, axis.z, maxSep };
        e.cfg = { nb, initC, 0, 0 };
        in.push_back(e);
        ref.push_back(polyClipRef(plane0, plane1, axis, maxSep, nb, initC, poly0, tri3));
        nbList.push_back(nb);
    };

    // general emitter with EXPLICIT plane0/plane1/axis (non-degenerate dnoms)
    auto emitG = [&](const std::vector<V3>& poly0, const V3 tri[3], F4 plane0, F4 plane1,
                     V3 axis, float maxSep, uint32_t initC)
    {
        const uint32_t nb = (uint32_t)poly0.size();
        for (uint32_t i = 0; i < MAXV; ++i)
            poly0Flat.push_back(i < nb ? F4{ poly0[i].x, poly0[i].y, poly0[i].z, 0.0f } : F4{ 0, 0, 0, 0 });
        for (uint32_t i = 0; i < 3; ++i) triFlat.push_back({ tri[i].x, tri[i].y, tri[i].z, 0.0f });
        PcIn e;
        e.plane0 = plane0; e.plane1 = plane1;
        e.axis = { axis.x, axis.y, axis.z, maxSep };
        e.cfg = { nb, initC, 0, 0 };
        in.push_back(e);
        ref.push_back(polyClipRef(plane0, plane1, axis, maxSep, nb, initC, poly0, tri));
        nbList.push_back(nb);
    };

    // varied axis-aligned squares/rects vs triangles.  The triangle is anchored
    // NEAR the rectangle (its first vertex inside/around the rect centre) so most
    // configs partially overlap -> phase-3 edge-edge crossings, with some fully-
    // inside (phase1/phase2 only) and a minority disjoint.
    for (uint32_t iter = 0; iter < 1200; ++iter)
    {
        const float ox = float(int(rng() % 9) - 4), oy = float(int(rng() % 9) - 4);
        const float w = float(3 + rng() % 4), h = float(3 + rng() % 4);
        std::vector<V3> rect = { { ox, oy, 0 }, { ox + w, oy, 0 }, { ox + w, oy + h, 0 }, { ox, oy + h, 0 } };
        const float cx = ox + w * 0.5f, cy = oy + h * 0.5f; // rect centre
        // anchor the triangle's first vertex near the rect centre (jittered)
        const float tx = cx + float(int(rng() % 5) - 2);
        const float ty = cy + float(int(rng() % 5) - 2);
        const float leg = float(2 + rng() % 5);
        const int orient = rng() % 4;
        V3 tri[3];
        if (orient == 0) tri[0] = { tx, ty, 0 }, tri[1] = { tx + leg, ty, 0 }, tri[2] = { tx, ty + leg, 0 };
        else if (orient == 1) tri[0] = { tx, ty, 0 }, tri[1] = { tx - leg, ty, 0 }, tri[2] = { tx, ty - leg, 0 };
        else if (orient == 2) tri[0] = { tx, ty, 0 }, tri[1] = { tx + leg, ty, 0 }, tri[2] = { tx + leg, ty + leg, 0 };
        else tri[0] = { tx, ty, 0 }, tri[1] = { tx, ty + leg, 0 }, tri[2] = { tx + leg, ty + leg, 0 };
        const float zOff = float(int(rng() % 5) - 2) * 0.5f;   // projection offset (pow2)
        const float maxSep = float(1 + rng() % 8);
        const uint32_t initC = rng() % 3;
        emit(rect, tri, zOff, maxSep, initC);
    }

    // high-initContacts family: start the count near NUM_TMP so an overlapping
    // clip pushes the exclusive-scan index PAST the 32-contact cap, pinning the
    // `index < NUM_TMP` overflow guard (dropped contacts are not written/counted)
    for (uint32_t iter = 0; iter < 200; ++iter)
    {
        std::vector<V3> rect = { { -2, -2, 0 }, { 2, -2, 0 }, { 2, 2, 0 }, { -2, 2, 0 } };
        V3 tri[3] = { { 0, 0, 0 }, { 5, 0, 0 }, { 0, 5, 0 } }; // partial overlap
        const float zOff = float(int(rng() % 5) - 2) * 0.5f;
        const uint32_t initC = 28u + rng() % 5; // 28..32
        emit(rect, tri, zOff, 8.0f, initC);
    }

    // family H (non-degenerate projection): TILTED plane1 normal and/or axis so
    // dnom0 (=plane1.n.axis) != dnom1 (=plane0.n.axis) and neither is 1 -> the
    // separations are REAL divisions that vary per vertex (pinning a dnom0<->dnom1
    // or plane0<->plane1 swap), and a SMALL maxSep makes some t>maxSep / sep>maxSep
    // -> the PC_MAX gates and the addContacts sep>maxSep drop actually fire.  All
    // arithmetic still has an identical counterpart on both sides, so bit-exact.
    for (uint32_t iter = 0; iter < 300; ++iter)
    {
        std::vector<V3> rect = { { -3, -3, 0 }, { 3, -3, 0 }, { 3, 3, 0 }, { -3, 3, 0 } };
        const float ox = float(int(rng() % 5) - 2), oy = float(int(rng() % 5) - 2);
        const float leg = float(2 + rng() % 4);
        V3 tri[3] = { { ox, oy, 0 }, { ox + leg, oy, 0 }, { ox, oy + leg, 0 } };
        const F4 plane0 = { 0, 0, 1, 0 };
        // tilted plane1 normal (Pythagorean-ish, not exactly unit -> a genuine
        // non-1 dnom0); a few variants including an axis tilt for dnom1 != 1 too
        const uint32_t v = rng() % 3;
        F4 plane1; V3 axis;
        if (v == 0) { plane1 = { 0.6f, 0.0f, 0.8f, 0.0f }; axis = { 0, 0, 1 }; }        // dnom0=0.8, dnom1=1
        else if (v == 1) { plane1 = { 0.0f, 0.6f, 0.8f, -0.5f }; axis = { 0, 0, 1 }; }   // dnom0=0.8, dnom1=1
        else { plane1 = { 0.0f, 0.0f, 1.0f, 0.0f }; axis = { 0.0f, 0.6f, 0.8f }; }        // dnom0=0.8, dnom1=0.8
        const float maxSep = float(1 + rng() % 3) * 0.5f; // 0.5, 1.0, 1.5 -> some drops
        const uint32_t initC = rng() % 3;
        emitG(rect, tri, plane0, plane1, axis, maxSep, initC);
    }

    // family I (phase-2 poly0-vertex separations that genuinely vary per vertex
    // AND straddle maxSep): a SMALL square poly0 fully contained in a LARGE
    // coplanar triangle, so EVERY poly0 vertex classifies inside the triangle
    // (out0 == 0 => flag2 = all four) and becomes a phase-2 contact carrying
    // sep = s0 = -t0/dnom0.  With a TILTED plane1 each vertex's t0 = plane1.n.v0
    // differs, so (a) s0 is a real division by dnom0 (NOT dnom1) -> a
    // dnom0<->dnom1 swap changes a surviving contact, and (b) the far verts have
    // t0 > maxSep so the PC_MAX gate legitimately drops them while the near verts
    // survive -> widening/removing that gate changes the contact set.  The big
    // triangle is wound CW so its interior is the out0==0 side.  Everything stays
    // coplanar at z=0 with the same ops on both sides -> still bit-exact.
    for (uint32_t iter = 0; iter < 240; ++iter)
    {
        // small square, verts at differing x so plane1 distances differ and the
        // far edge reaches past maxSep
        const float x0 = float(int(rng() % 3) - 1);        // -1..1
        const float x1 = x0 + float(2 + rng() % 3);        // span 2..4
        const float y0 = float(int(rng() % 3) - 1);        // -1..1
        const float y1 = y0 + float(2 + rng() % 2);        // span 2..3
        std::vector<V3> rect = { { x0, y0, 0 }, { x1, y0, 0 }, { x1, y1, 0 }, { x0, y1, 0 } };
        // large CW triangle containing the square (interior => out0 == 0)
        V3 tri[3] = { { -12, -12, 0 }, { 0, 24, 0 }, { 12, -12, 0 } };
        const F4 plane0 = { 0, 0, 1, 0 };
        F4 plane1; V3 axis = { 0, 0, 1 };
        // tilted plane1 => dnom0 (=plane1.n.axis) in {0.8, 0.6} != dnom1 (=1)
        if (rng() % 2 == 0) plane1 = { 0.6f, 0.0f, 0.8f, 0.0f };  // dnom0 = 0.8
        else                plane1 = { 0.8f, 0.0f, 0.6f, 0.0f };  // dnom0 = 0.6
        const float maxSep = float(1 + rng() % 2);  // 1 or 2 => far verts (t0>maxSep) dropped
        const uint32_t initC = rng() % 2;
        emitG(rect, tri, plane0, plane1, axis, maxSep, initC);
    }

    const uint32_t N = (uint32_t)in.size();
    PxgSlangComputeDevice* dev = testDevice();
    std::vector<F4> cInit(N * NUM_TMP, { -777, -777, -777, -777 });
    std::vector<uint32_t> cntInit(N, 0xEEEEEEEEu);

    PxgBufferHandle bP0 = createBufferFrom(dev, poly0Flat);
    PxgBufferHandle bT = createBufferFrom(dev, triFlat);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bC = createBufferFrom(dev, cInit);
    PxgBufferHandle bCnt = createBufferFrom(dev, cntInit);

    PcParams prm = { N, MAXV, 0, 0 };
    Kernel k("npConvexPolyClip.spv", "ctPolyClip",
             { (uint32_t)sizeof(F4), (uint32_t)sizeof(F4), (uint32_t)sizeof(PcIn), (uint32_t)sizeof(F4), 4 },
             sizeof(PcParams));
    k.dispatch((N + 63) / 64, { bP0, bT, bIn, bC, bCnt }, prm);
    dev->flush();

    const auto gC = downloadBuffer<F4>(dev, bC, N * NUM_TMP);
    const auto gCnt = downloadBuffer<uint32_t>(dev, bCnt, N);

    size_t mm = 0;
    size_t nP1c = 0, nP2c = 0, nP3c = 0, nEmptyc = 0, nOverflow = 0;
    for (uint32_t p = 0; p < N; ++p)
    {
        const WarpRef& r = ref[p];
        if (gCnt[p] != r.count) { ++mm; continue; }
        for (uint32_t i = 0; i < NUM_TMP; ++i)
        {
            const F4 g = gC[p * NUM_TMP + i];
            const F4 e = r.out[i];
            if (asU32(g.x) != asU32(e.x) || asU32(g.y) != asU32(e.y) ||
                asU32(g.z) != asU32(e.z) || asU32(g.w) != asU32(e.w)) ++mm;
        }
        const uint32_t added = r.count - in[p].cfg.y;
        if (added == 0) ++nEmptyc;
        if (r.count >= NUM_TMP) ++nOverflow;
        // rough phase attribution for coverage: count nonzero contacts
        if (added >= 1) ++nP1c;
        if (added >= 4) ++nP2c;
        if (added >= 5) ++nP3c;
    }
    PXG_CHECK_MSG(mm == 0, "polyClip: %zu mismatches", mm);
    PXG_CHECK_MSG(nP1c > 100 && nP3c > 20 && nEmptyc > 20,
                  "polyClip coverage thin (>=1:%zu >=4:%zu >=5:%zu empty:%zu overflow:%zu)",
                  nP1c, nP2c, nP3c, nEmptyc, nOverflow);

    for (PxgBufferHandle h : { bP0, bT, bIn, bC, bCnt }) dev->destroyBuffer(h);
}

} // namespace
