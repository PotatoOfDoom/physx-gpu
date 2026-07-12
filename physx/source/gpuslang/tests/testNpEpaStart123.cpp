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

// Verification of npEpaStart123.slang: ctEpaStart123 — the EPA seed-triangle
// setup (epa.cuh :330-385).  One thread per test simulates the 32-lane warp; the
// filled-out triangle vertices/indices, the two-sided normal/distance and the
// fixed topology are compared bit-exactly against a CPU transcription.  Integer
// clouds + integer cached vertices keep the support argmax exact and
// magnitude/getNormalized correctly-rounded, so the output is bit-for-bit
// reproducible.  Fixtures cover simplex sizes 1/2/3 (two/one/zero newPoint calls).

#include "pxgTestFramework.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct V3 { float x, y, z; };
struct Es3Params { uint32_t numTests, maxV, pad0, pad1; };
struct Es3Scalars { uint32_t size, nbA, nbB, pad; };
struct Es3Out { float verts[9]; uint32_t indices[3]; float normal0[3]; float dist0; float normal1[3]; float dist1; uint32_t v0, v1, a0, a1; };

static const float MAXF = 3.402823466e38f;
static uint32_t asU32(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float asF32(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static V3 sub3(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 getNormalized(V3 v) { const float m = dot3(v, v); if (m > 0.0f) { const float r = 1.0f / std::sqrt(m); return { v.x * r, v.y * r, v.z * r }; } return { 0, 0, 0 }; }
static float magnitude3(V3 v) { return std::sqrt(dot3(v, v)); }

static uint32_t supportIdx(V3 dir, const std::vector<float>& V, uint32_t vbase, uint32_t nb)
{
    uint32_t bestN = 0xFFFFFFFFu; float bestDist = -MAXF;
    for (uint32_t i = 0; i < nb; i += 32)
    {
        float d[32];
        for (uint32_t lane = 0; lane < 32; ++lane) { const uint32_t n = i + lane; V3 v = { V[(vbase + n) * 4], V[(vbase + n) * 4 + 1], V[(vbase + n) * 4 + 2] }; d[lane] = (n < nb) ? dot3(dir, v) : -MAXF; }
        float cmax = -MAXF; for (uint32_t lane = 0; lane < 32; ++lane) cmax = std::fmax(cmax, d[lane]);
        if (cmax > bestDist) { uint32_t ll = 0; for (uint32_t lane = 0; lane < 32; ++lane) if (d[lane] == cmax) { ll = lane; break; } bestDist = cmax; bestN = i + ll; }
    }
    return bestN;
}
static void newPoint(V3 dir, V3 ref, const std::vector<float>& VA, const std::vector<float>& VB, uint32_t vbase, uint32_t nbA, uint32_t nbB, V3& vertex, uint32_t& indices)
{
    auto va = [&](uint32_t i) -> V3 { return { VA[(vbase + i) * 4], VA[(vbase + i) * 4 + 1], VA[(vbase + i) * 4 + 2] }; };
    auto vb = [&](uint32_t i) -> V3 { return { VB[(vbase + i) * 4], VB[(vbase + i) * 4 + 1], VB[(vbase + i) * 4 + 2] }; };
    const V3 ndir = { -dir.x, -dir.y, -dir.z };
    const uint32_t iAp = supportIdx(dir, VA, vbase, nbA), iBp = supportIdx(ndir, VB, vbase, nbB);
    const uint32_t iAn = supportIdx(ndir, VA, vbase, nbA), iBn = supportIdx(dir, VB, vbase, nbB);
    const V3 vp = sub3(va(iAp), vb(iBp)), vn = sub3(va(iAn), vb(iBn));
    if (std::fabs(dot3(sub3(vp, ref), dir)) > std::fabs(dot3(sub3(vn, ref), dir))) { vertex = vp; indices = (iBp << 8) | (iAp & 0xffu); }
    else { vertex = vn; indices = (iBn << 8) | (iAn & 0xffu); }
}

static Es3Out start123Ref(const float input[32], const Es3Scalars& s, const std::vector<float>& VA, const std::vector<float>& VB, uint32_t vbase)
{
    float verts[12]; uint32_t indices[4];
    for (uint32_t L = 0; L < 16; ++L) { if ((L & 3u) < 3u) verts[L - L / 4u] = input[L]; else indices[L >> 2] = asU32(input[L]) & 0xffffu; } // CUDA S.indices is PxU16

    V3 vertex0 = { verts[0], verts[1], verts[2] };
    V3 vertex1 = { verts[3], verts[4], verts[5] };
    if (s.size < 2u) { V3 nv; uint32_t ni; newPoint({ 1, 0, 0 }, vertex0, VA, VB, vbase, s.nbA, s.nbB, nv, ni); vertex1 = nv; verts[3] = nv.x; verts[4] = nv.y; verts[5] = nv.z; indices[1] = ni; }

    V3 vertex2 = { verts[6], verts[7], verts[8] };
    if (s.size < 3u)
    {
        const V3 dir = sub3(vertex1, vertex0);
        const V3 axis = (std::fabs(dir.x) < std::fabs(dir.y)) ? V3{ 1, 0, 0 } : V3{ 0, 1, 0 };
        const V3 sdir = cross3(axis, dir);
        V3 nv; uint32_t ni; newPoint(sdir, vertex0, VA, VB, vbase, s.nbA, s.nbB, nv, ni); vertex2 = nv; verts[6] = nv.x; verts[7] = nv.y; verts[8] = nv.z; indices[2] = ni;
    }

    const V3 e01 = sub3(vertex1, vertex0), e12 = sub3(vertex2, vertex1), e02 = sub3(vertex2, vertex0);
    V3 normal;
    if (magnitude3(e01) <= magnitude3(e02)) normal = getNormalized(cross3(e01, e12));
    else                                    normal = getNormalized(cross3(e02, e12));
    const float distance = dot3(normal, vertex0);

    Es3Out o;
    for (uint32_t i = 0; i < 9; ++i) o.verts[i] = verts[i];
    o.indices[0] = indices[0]; o.indices[1] = indices[1]; o.indices[2] = indices[2];
    o.normal0[0] = normal.x; o.normal0[1] = normal.y; o.normal0[2] = normal.z; o.dist0 = distance;
    o.normal1[0] = -normal.x; o.normal1[1] = -normal.y; o.normal1[2] = -normal.z; o.dist1 = -distance;
    o.v0 = 0x020100u; o.v1 = 0x020001u; o.a0 = 0x010101u; o.a1 = 0x000000u;
    return o;
}

PXG_TEST(npEpaStart123_setup)
{
    std::mt19937 rng(1310131);
    const uint32_t MAXV = 32;
    std::vector<float> Data, VA, VB;
    std::vector<Es3Scalars> S;
    std::vector<Es3Out> ref;
    size_t n1 = 0, n2 = 0, n3 = 0;

    auto ic = [&](int r) -> int { return int(rng() % (2 * r + 1)) - r; };
    auto enc = [&]() -> float { return asF32(0x3f800000u | ((rng() % 30u) << 8) | (rng() % 30u)); };

    auto emit = [&](uint32_t size)
    {
        const uint32_t vbase = (uint32_t)(VA.size() / 4);
        const uint32_t nbA = 5 + rng() % 20, nbB = 5 + rng() % 20;
        for (uint32_t k = 0; k < MAXV; ++k)
        {
            V3 a = (k < nbA) ? V3{ float(ic(10)), float(ic(10)), float(ic(10)) } : V3{ 0, 0, 0 };
            V3 b = (k < nbB) ? V3{ float(ic(10)), float(ic(10)), float(ic(10)) } : V3{ 0, 0, 0 };
            VA.push_back(a.x); VA.push_back(a.y); VA.push_back(a.z); VA.push_back(0);
            VB.push_back(b.x); VB.push_back(b.y); VB.push_back(b.z); VB.push_back(0);
        }
        float input[32];
        for (uint32_t L = 0; L < 32; ++L) input[L] = 0.0f;
        // cached vertices 0..size-1: integer xyz + encoded index; keep a non-degenerate spread
        for (uint32_t k = 0; k < size; ++k)
        {
            input[k * 4 + 0] = float(ic(12));
            input[k * 4 + 1] = float(ic(12));
            input[k * 4 + 2] = float(ic(12));
            input[k * 4 + 3] = enc();
        }
        for (uint32_t L = 0; L < 32; ++L) Data.push_back(input[L]);
        Es3Scalars s{}; s.size = size; s.nbA = nbA; s.nbB = nbB;
        S.push_back(s);
        ref.push_back(start123Ref(input, s, VA, VB, vbase));
        if (size == 1) ++n1; else if (size == 2) ++n2; else ++n3;
    };

    for (uint32_t it = 0; it < 500; ++it) emit(1u);
    for (uint32_t it = 0; it < 500; ++it) emit(2u);
    for (uint32_t it = 0; it < 500; ++it) emit(3u);

    // degenerate size-3 fixtures: collinear vertices -> the triangle normal cross is
    // exactly zero, pinning the getNormalized m>0 zero-guard (returns 0, not 0*inf=NaN).
    auto emitCollinear = [&]()
    {
        const uint32_t vbase = (uint32_t)(VA.size() / 4);
        const uint32_t nbA = 6, nbB = 6;
        for (uint32_t k = 0; k < MAXV; ++k)
        {
            VA.push_back(float(ic(10))); VA.push_back(float(ic(10))); VA.push_back(float(ic(10))); VA.push_back(0);
            VB.push_back(float(ic(10))); VB.push_back(float(ic(10))); VB.push_back(float(ic(10))); VB.push_back(0);
        }
        float input[32]; for (uint32_t L = 0; L < 32; ++L) input[L] = 0.0f;
        // three points on a line: base + k*step
        const V3 baseP = { float(ic(6)), float(ic(6)), float(ic(6)) };
        const V3 step = { float(1 + int(rng() % 4)), float(int(rng() % 5) - 2), float(int(rng() % 5) - 2) };
        for (uint32_t k = 0; k < 3; ++k)
        {
            input[k * 4 + 0] = baseP.x + float(k) * step.x;
            input[k * 4 + 1] = baseP.y + float(k) * step.y;
            input[k * 4 + 2] = baseP.z + float(k) * step.z;
            input[k * 4 + 3] = enc();
        }
        for (uint32_t L = 0; L < 32; ++L) Data.push_back(input[L]);
        Es3Scalars s{}; s.size = 3u; s.nbA = nbA; s.nbB = nbB;
        S.push_back(s);
        ref.push_back(start123Ref(input, s, VA, VB, vbase));
        ++n3;
    };
    for (uint32_t it = 0; it < 150; ++it) emitCollinear();

    const uint32_t N = (uint32_t)ref.size();
    PxgSlangComputeDevice* dev = testDevice();
    PxgBufferHandle bData = createBufferFrom(dev, Data);
    PxgBufferHandle bVA = createBufferFrom(dev, VA);
    PxgBufferHandle bVB = createBufferFrom(dev, VB);
    PxgBufferHandle bS = createBufferFrom(dev, S);
    std::vector<Es3Out> outInit(N);
    std::memset(outInit.data(), 0xEE, outInit.size() * sizeof(Es3Out));
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    Es3Params prm = { N, MAXV, 0, 0 };
    Kernel k("npEpaStart123.spv", "ctEpaStart123", { 4, 16, 16, (uint32_t)sizeof(Es3Scalars), (uint32_t)sizeof(Es3Out) }, sizeof(Es3Params));
    k.dispatch((N + 63) / 64, { bData, bVA, bVB, bS, bOut }, prm);
    dev->flush();

    const auto gOut = downloadBuffer<Es3Out>(dev, bOut, N);

    size_t mm = 0;
    for (uint32_t t = 0; t < N; ++t)
    {
        const Es3Out& r = ref[t]; const Es3Out& g = gOut[t];
        for (uint32_t i = 0; i < 9; ++i) if (asU32(g.verts[i]) != asU32(r.verts[i])) ++mm;
        for (uint32_t i = 0; i < 3; ++i) if (g.indices[i] != r.indices[i]) ++mm;
        for (uint32_t i = 0; i < 3; ++i) { if (asU32(g.normal0[i]) != asU32(r.normal0[i])) ++mm; if (asU32(g.normal1[i]) != asU32(r.normal1[i])) ++mm; }
        if (asU32(g.dist0) != asU32(r.dist0)) ++mm;
        if (asU32(g.dist1) != asU32(r.dist1)) ++mm;
        if (g.v0 != r.v0 || g.v1 != r.v1 || g.a0 != r.a0 || g.a1 != r.a1) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "epaStart123: %zu mismatches", mm);
    PXG_CHECK_MSG(n1 > 100 && n2 > 100 && n3 > 100, "epaStart123 coverage thin (s1=%zu s2=%zu s3=%zu)", n1, n2, n3);

    for (PxgBufferHandle h : { bData, bVA, bVB, bS, bOut }) dev->destroyBuffer(h);
}

} // namespace
