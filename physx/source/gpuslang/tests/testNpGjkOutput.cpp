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

// Verification of npGjkOutput.slang: ctGjkOutput — the gjk() final-output stage
// (gjk.cuh :383-417).  One thread per test classifies the GJK result and, when
// CLOSE, produces the witness points.  The result enum plus closestPointDir /
// closestPointA / closestPointB are compared bit-exactly against a CPU
// transcription.  Integer cached vertices / closest points / vA cloud keep every
// dot/cross/magnitudeSquared exact and the divisions (1/m.[axis], dot/magSq,
// 1/sqrt) correctly-rounded, so the output is bit-for-bit reproducible.  Fixtures
// sweep the three result branches (overlap / distant / close) and, within close,
// all three simplex sizes (1/2/3).

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
struct GoParams { uint32_t numTests, pad0, pad1, pad2; };
struct GoIn { float v0[4], v1[4], v2[4], cp[4], bounds[4]; };
struct GoOut { uint32_t ret, p0, p1, p2; float dir[4], cpA[4], cpB[4]; };

static const uint32_t GJK_DISTANT = 0u, GJK_CLOSE = 1u, GJK_OVERLAP = 2u;

static uint32_t asU32(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float asF32(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

static V3 crossX(V3 a, V3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static V3 absX(V3 a) { return { std::fabs(a.x), std::fabs(a.y), std::fabs(a.z) }; }
static float dotX(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }

static V3 getNormalized(V3 v)
{
    const float m = dotX(v, v);
    if (m > 0.0f) { const float r = 1.0f / std::sqrt(m); return { v.x * r, v.y * r, v.z * r }; }
    return { 0.0f, 0.0f, 0.0f };
}

struct Ref { uint32_t ret; V3 dir, cpA, cpB; };
static Ref outputRef(const GoIn& e, const std::vector<float>& VA)
{
    Ref r; r.dir = r.cpA = r.cpB = { 0, 0, 0 };
    const float upperBound = e.bounds[0], lowerBound = e.bounds[1], minSep = e.bounds[2];
    const uint32_t size = asU32(e.bounds[3]);

    if (upperBound == 0.0f) { r.ret = GJK_OVERLAP; return r; }
    if (lowerBound > minSep) { r.ret = GJK_DISTANT; return r; }

    const uint32_t a0 = asU32(e.v0[3]) & 0xffu, a1 = asU32(e.v1[3]) & 0xffu, a2 = asU32(e.v2[3]) & 0xffu;
    auto va = [&](uint32_t i) -> V3 { return { VA[i * 4], VA[i * 4 + 1], VA[i * 4 + 2] }; };

    const V3 cp = { e.cp[0], e.cp[1], e.cp[2] };
    const V3 gn = getNormalized(cp);
    const V3 dir = { -gn.x, -gn.y, -gn.z };
    r.dir = dir;

    V3 cpA;
    if (size == 3u)
    {
        const V3 A = { e.v0[0], e.v0[1], e.v0[2] }, B = { e.v1[0], e.v1[1], e.v1[2] }, C = { e.v2[0], e.v2[1], e.v2[2] };
        const V3 m = crossX(sub(B, A), sub(C, A));
        const V3 ma = absX(m);
        const V3 s1 = crossX(sub(B, cp), sub(C, cp));
        const V3 s2 = crossX(sub(C, cp), sub(A, cp));
        float d, u, v;
        if (ma.x >= ma.y && ma.x >= ma.z) { d = 1.0f / m.x; u = s1.x * d; v = s2.x * d; }
        else if (ma.y >= ma.z)            { d = 1.0f / m.y; u = s1.y * d; v = s2.y * d; }
        else                              { d = 1.0f / m.z; u = s1.z * d; v = s2.z * d; }
        const float w = 1.0f - u - v;
        const V3 p0 = va(a0), p1 = va(a1), p2 = va(a2);
        cpA = { u * p0.x + v * p1.x + w * p2.x, u * p0.y + v * p1.y + w * p2.y, u * p0.z + v * p1.z + w * p2.z };
    }
    else if (size == 2u)
    {
        const V3 V0 = { e.v0[0], e.v0[1], e.v0[2] }, V1 = { e.v1[0], e.v1[1], e.v1[2] };
        const V3 dv = sub(V1, V0);
        const float q = -dotX(V0, dv) / dotX(dv, dv);
        const V3 p0 = va(a0), p1 = va(a1);
        cpA = { (1.0f - q) * p0.x + q * p1.x, (1.0f - q) * p0.y + q * p1.y, (1.0f - q) * p0.z + q * p1.z };
    }
    else
    {
        cpA = va(a0);
    }
    r.cpA = cpA;
    r.cpB = { cpA.x + upperBound * dir.x, cpA.y + upperBound * dir.y, cpA.z + upperBound * dir.z };
    r.ret = GJK_CLOSE;
    return r;
}

PXG_TEST(npGjkOutput_stage)
{
    std::mt19937 rng(1250125);
    const uint32_t NVA = 64;
    std::vector<float> VA(NVA * 4, 0.0f);
    for (uint32_t i = 0; i < NVA; ++i)
    {
        VA[i * 4 + 0] = float(int(rng() % 21) - 10);
        VA[i * 4 + 1] = float(int(rng() % 21) - 10);
        VA[i * 4 + 2] = float(int(rng() % 21) - 10);
        VA[i * 4 + 3] = 0.0f;
    }

    std::vector<GoIn> in;
    std::vector<Ref> ref;
    size_t nOverlap = 0, nDistant = 0, nClose1 = 0, nClose2 = 0, nClose3 = 0;

    auto iv = [&](int r) -> V3 { return { float(int(rng() % (2 * r + 1)) - r), float(int(rng() % (2 * r + 1)) - r), float(int(rng() % (2 * r + 1)) - r) }; };
    auto idx = [&]() -> uint32_t { return rng() % NVA; };
    // encoded index word: 0x3f800000 | b<<8 | a  (a in low byte)
    auto enc = [&](uint32_t a, uint32_t b) -> float { return asF32(0x3f800000u | ((b & 0xffu) << 8) | (a & 0xffu)); };

    auto emit = [&](GoIn e)
    {
        in.push_back(e);
        Ref r = outputRef(e, VA);
        ref.push_back(r);
        if (r.ret == GJK_OVERLAP) ++nOverlap;
        else if (r.ret == GJK_DISTANT) ++nDistant;
        else { uint32_t s = asU32(e.bounds[3]); if (s == 3) ++nClose3; else if (s == 2) ++nClose2; else ++nClose1; }
    };

    auto mkClose = [&](uint32_t size)
    {
        GoIn e{};
        V3 cp; do { cp = iv(9); } while (cp.x == 0 && cp.y == 0 && cp.z == 0);
        V3 A = iv(9), B = iv(9), C = iv(9);
        uint32_t a0 = idx(), a1 = idx(), a2 = idx();
        e.v0[0] = A.x; e.v0[1] = A.y; e.v0[2] = A.z; e.v0[3] = enc(a0, idx());
        e.v1[0] = B.x; e.v1[1] = B.y; e.v1[2] = B.z; e.v1[3] = enc(a1, idx());
        e.v2[0] = C.x; e.v2[1] = C.y; e.v2[2] = C.z; e.v2[3] = enc(a2, idx());
        e.cp[0] = cp.x; e.cp[1] = cp.y; e.cp[2] = cp.z; e.cp[3] = 0;
        // upperBound > 0, lowerBound <= minSep  -> CLOSE
        e.bounds[0] = float(1 + rng() % 40);      // upperBound > 0
        e.bounds[1] = float(int(rng() % 11) - 20); // lowerBound (<= minSep)
        e.bounds[2] = float(1 + rng() % 5);        // minSep
        e.bounds[3] = asF32(size);
        // for size==2, ensure d = v1-v0 non-degenerate
        if (size == 2u && A.x == B.x && A.y == B.y && A.z == B.z) { e.v1[0] = B.x + 1.0f; }
        // for size==3, ensure triangle non-degenerate (m != 0)
        if (size == 3u)
        {
            V3 m = crossX(sub(B, A), sub(C, A));
            if (m.x == 0 && m.y == 0 && m.z == 0) { e.v2[1] = C.y + 3.0f; e.v2[0] = C.x + 1.0f; }
        }
        emit(e);
    };

    // close, all three sizes
    for (uint32_t i = 0; i < 700; ++i) mkClose(3u);
    for (uint32_t i = 0; i < 700; ++i) mkClose(2u);
    for (uint32_t i = 0; i < 400; ++i) mkClose(1u);
    // overlap: upperBound == 0
    for (uint32_t i = 0; i < 200; ++i)
    {
        GoIn e{}; e.v0[3] = enc(idx(), idx()); e.v1[3] = enc(idx(), idx()); e.v2[3] = enc(idx(), idx());
        e.cp[0] = iv(9).x; e.bounds[0] = 0.0f; e.bounds[1] = -5.0f; e.bounds[2] = 1.0f; e.bounds[3] = asF32(3u);
        emit(e);
    }
    // distant: lowerBound > minSep (upperBound != 0)
    for (uint32_t i = 0; i < 200; ++i)
    {
        GoIn e{}; e.v0[3] = enc(idx(), idx()); e.v1[3] = enc(idx(), idx()); e.v2[3] = enc(idx(), idx());
        e.cp[0] = iv(9).x; e.bounds[0] = 5.0f; e.bounds[1] = float(10 + rng() % 10); e.bounds[2] = float(1 + rng() % 5); e.bounds[3] = asF32(2u);
        emit(e);
    }
    // boundary: lowerBound == minSep exactly -> stays CLOSE under the faithful strict
    // `>` (a `>=` mutant would misclassify these as DISTANT).  Overriding the bounds
    // keeps the result CLOSE, so mkClose's size count remains valid.
    for (uint32_t i = 0; i < 150; ++i)
    {
        const uint32_t size = 1u + (i % 3u);
        mkClose(size);
        GoIn& e = in.back();
        const float sep = float(1 + rng() % 6);
        e.bounds[1] = sep; e.bounds[2] = sep;   // lowerBound == minSep
        ref.back() = outputRef(e, VA);
    }
    // zero closest point: exercises the getNormalized m>0 guard (returns 0, not the
    // mutant's 0*inf = NaN).  cp override keeps the CLOSE classification intact.
    for (uint32_t i = 0; i < 150; ++i)
    {
        const uint32_t size = 1u + (i % 3u);
        mkClose(size);
        GoIn& e = in.back();
        e.cp[0] = 0.0f; e.cp[1] = 0.0f; e.cp[2] = 0.0f;   // closest point at origin
        ref.back() = outputRef(e, VA);
    }

    const uint32_t N = (uint32_t)ref.size();
    PxgSlangComputeDevice* dev = testDevice();
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bVA = createBufferFrom(dev, VA);
    std::vector<GoOut> outInit(N);
    std::memset(outInit.data(), 0xEE, outInit.size() * sizeof(GoOut));
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    GoParams prm = { N, 0, 0, 0 };
    Kernel k("npGjkOutput.spv", "ctGjkOutput", { (uint32_t)sizeof(GoIn), 16, (uint32_t)sizeof(GoOut) }, sizeof(GoParams));
    k.dispatch((N + 63) / 64, { bIn, bVA, bOut }, prm);
    dev->flush();

    const auto gOut = downloadBuffer<GoOut>(dev, bOut, N);

    size_t mm = 0;
    auto ck = [&](const float g[4], const V3& r) { return asU32(g[0]) == asU32(r.x) && asU32(g[1]) == asU32(r.y) && asU32(g[2]) == asU32(r.z); };
    for (uint32_t t = 0; t < N; ++t)
    {
        const Ref& r = ref[t];
        if (gOut[t].ret != r.ret) { ++mm; continue; }
        if (r.ret == GJK_CLOSE)
        {
            if (!ck(gOut[t].dir, r.dir)) ++mm;
            if (!ck(gOut[t].cpA, r.cpA)) ++mm;
            if (!ck(gOut[t].cpB, r.cpB)) ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "gjkOutput: %zu mismatches", mm);
    PXG_CHECK_MSG(nOverlap > 100 && nDistant > 100 && nClose1 > 100 && nClose2 > 100 && nClose3 > 100,
                  "gjkOutput coverage thin (overlap=%zu distant=%zu close1=%zu close2=%zu close3=%zu)",
                  nOverlap, nDistant, nClose1, nClose2, nClose3);

    for (PxgBufferHandle h : { bIn, bVA, bOut }) dev->destroyBuffer(h);
}

} // namespace
