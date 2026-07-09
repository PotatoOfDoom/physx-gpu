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

// Verification of npTriTri.slang: satIntersect — the triangle-triangle SAT
// separation test.  The returned separation is compared to a C++ transcription
// (same axis order and early-outs); groups where any early-out sits within a
// small margin of the tolerance, or an edge-cross magnitude sits near the 1e-5
// validity gate, are skipped (a GPU-vs-CPU flip of the early-out path or the
// gate is measure-zero for the fixed seed).  Both separated and overlapping
// outcomes are exercised.

#include "pxgTestFramework.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct V3 { float x, y, z; };
struct TriTriIn { F4 v0a, v1a, v2a, planeA, v0b, v1b, v2b, planeB, misc; };

const float MARG = 1e-3f;

static inline V3 sub(const V3& a, const V3& b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static inline float dot3(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 crossp(const V3& a, const V3& b) { return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static inline V3 nrm(const V3& a) { float l = std::sqrt(dot3(a, a)); return l > 0 ? V3{ a.x / l, a.y / l, a.z / l } : a; }

static float minProject(const F4& plane, const V3& v0, const V3& v1, const V3& v2)
{
    const V3 n{ plane.x, plane.y, plane.z };
    return std::min(dot3(n, v0), std::min(dot3(n, v1), dot3(n, v2))) + plane.w;
}
static float testSep(const V3& axis, const V3& a0, const V3& a1, const V3& a2, const V3& b0, const V3& b1, const V3& b2)
{
    float min0, max0, min1, max1;
    min0 = max0 = dot3(a0, axis); min1 = max1 = dot3(b0, axis);
    const float pa1 = dot3(a1, axis), pa2 = dot3(a2, axis), pb1 = dot3(b1, axis), pb2 = dot3(b2, axis);
    min0 = std::min(pa1, min0); max0 = std::max(pa1, max0); min1 = std::min(pb1, min1); max1 = std::max(pb1, max1);
    min0 = std::min(pa2, min0); max0 = std::max(pa2, max0); min1 = std::min(pb2, min1); max1 = std::max(pb2, max1);
    return std::max(min1 - max0, min0 - max1);
}

// mirrors satIntersect<true>; sets marg if any early-out / gate is borderline
float refSat(const TriTriIn& s, float tol, bool& marg)
{
    const V3 a0{ s.v0a.x, s.v0a.y, s.v0a.z }, a1{ s.v1a.x, s.v1a.y, s.v1a.z }, a2{ s.v2a.x, s.v2a.y, s.v2a.z };
    const V3 b0{ s.v0b.x, s.v0b.y, s.v0b.z }, b1{ s.v1b.x, s.v1b.y, s.v1b.z }, b2{ s.v2b.x, s.v2b.y, s.v2b.z };

    float sep = minProject(s.planeA, b0, b1, b2);
    if (std::fabs(sep - tol) < MARG) marg = true;
    if (sep > tol) return sep;

    sep = std::max(sep, minProject(s.planeB, a0, a1, a2));
    if (std::fabs(sep - tol) < MARG) marg = true;
    if (sep > tol) return sep;

    const V3 ax0[3] = { sub(a1, a0), sub(a2, a0), sub(a2, a1) };
    const V3 ax1[3] = { sub(b1, b0), sub(b2, b0), sub(b2, b1) };
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            V3 sa = crossp(ax0[i], ax1[j]);
            const float magSq = dot3(sa, sa);
            if (std::fabs(magSq - 1e-5f) < 1e-4f) marg = true; // validity-gate boundary
            if (magSq > 1e-5f)
            {
                sa = V3{ sa.x / std::sqrt(magSq), sa.y / std::sqrt(magSq), sa.z / std::sqrt(magSq) };
                const float tSep = testSep(sa, a0, a1, a2, b0, b1, b2);
                sep = std::max(sep, tSep);
                if (std::fabs(sep - tol) < MARG) marg = true;
                if (sep > tol) return sep;
            }
        }
    return sep;
}

PXG_TEST(npTriTri_satIntersect)
{
    std::mt19937 rng(57010);
    const uint32_t N = 14000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto mkTri = [&](V3& v0, V3& v1, V3& v2, F4& plane, const V3& center, float scale)
    {
        v0 = V3{ center.x + rf(-scale, scale), center.y + rf(-scale, scale), center.z + rf(-scale, scale) };
        v1 = V3{ center.x + rf(-scale, scale), center.y + rf(-scale, scale), center.z + rf(-scale, scale) };
        v2 = V3{ center.x + rf(-scale, scale), center.y + rf(-scale, scale), center.z + rf(-scale, scale) };
        const V3 n = nrm(crossp(sub(v1, v0), sub(v2, v0)));
        plane = F4{ n.x, n.y, n.z, -dot3(n, v0) };
    };

    std::vector<TriTriIn> in(N);
    std::vector<uint8_t> skip(N, 0);
    std::vector<float> ref(N);
    size_t nSkip = 0, sepC = 0, ovlC = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        TriTriIn s{};
        V3 a0, a1, a2, b0, b1, b2;
        mkTri(a0, a1, a2, s.planeA, V3{ 0, 0, 0 }, 1.0f);
        // second triangle offset by a random amount spanning overlap / separation
        const V3 off{ rf(-1.3f, 1.3f), rf(-1.3f, 1.3f), rf(-1.3f, 1.3f) };
        mkTri(b0, b1, b2, s.planeB, off, 1.0f);
        const float tol = rf(-0.1f, 0.3f);
        s.v0a = { a0.x, a0.y, a0.z, 0 }; s.v1a = { a1.x, a1.y, a1.z, 0 }; s.v2a = { a2.x, a2.y, a2.z, 0 };
        s.v0b = { b0.x, b0.y, b0.z, 0 }; s.v1b = { b1.x, b1.y, b1.z, 0 }; s.v2b = { b2.x, b2.y, b2.z, 0 };
        s.misc = { tol, 0, 0, 0 };
        in[i] = s;
        bool marg = false;
        ref[i] = refSat(s, tol, marg);
        skip[i] = marg ? 1u : 0u;
        if (marg) { ++nSkip; continue; }
        if (ref[i] > tol) ++sepC; else ++ovlC;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<float> outInit(N, -999.f);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct TriTriParams { uint32_t numTests; } prm = { N };
    Kernel k("npTriTri.spv", "triTriSat", { (uint32_t)sizeof(TriTriIn), 4 }, sizeof(TriTriParams));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<float>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (skip[i]) continue;
        if (!almostEqual(g[i], ref[i], 1e-4f, 1e-4f)) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "triTriSat: %zu separation mismatches", mm);
    PXG_CHECK_MSG(nSkip < N / 3, "triTriSat: too many marginal (%zu/%u)", nSkip, N);
    PXG_CHECK_MSG(sepC > N / 20 && ovlC > N / 20, "triTriSat: separated/overlap mix thin (%zu/%zu)", sepC, ovlC);

    dev->destroyBuffer(bIn); dev->destroyBuffer(bOut);
}

} // namespace
