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

// Verification of npGjkClosestSeg.slang: ctClosestPointSeg — the GJK 1-simplex
// (line-segment) closest-point, the `case 1` inline of gjk() (gjk.cuh :323-329).
// One thread per test simulates the 32-lane warp; the return feature code and
// the 32-lane closest point are compared exactly against a CPU transcription
// that serialises the identical warp ops.  Integer coordinates keep every dot
// exact and the single division x/m correctly-rounded IEEE (matching the CUDA
// grouping (l*x)/m), so the whole output is bit-for-bit reproducible.  Fixtures
// place a base vertex a (slot 0) and a replicated support vertex c to hit all
// three regions: line (0x2102), new-vertex (0x2111) and degenerate (0xffff).

#include "pxgTestFramework.h"

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct V3 { float x, y, z; };
struct CtParams { uint32_t numTests, pad0, pad1, pad2; };
static const uint32_t DEGENERATE = 0xffffu;

static uint32_t asU32(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

static void dotW(const float a[32], const float b[32], float o[32])
{
    float x[32]; for (uint32_t L = 0; L < 32; ++L) x[L] = a[L] * b[L];
    for (uint32_t L = 0; L < 32; ++L)
    {
        const float s1 = (L + 1 < 32) ? x[L + 1] : x[L];
        const float s2 = (L + 2 < 32) ? x[L + 2] : x[L];
        o[L] = x[L] + s1 + s2;
    }
}
static void splatXW(const float a[32], float o[32]) { for (uint32_t L = 0; L < 32; ++L) o[L] = a[L & 28u]; }
static void splatVW(uint32_t index, const float a[32], float o[32])
{
    for (uint32_t L = 0; L < 32; ++L) o[L] = a[((index << 2) | (L & 3u)) & 31u];
}

struct CtRef { uint32_t ret; float cp[32]; };
static CtRef closestSegRef(const float zV[32], const float cV[32])
{
    CtRef r;
    float aV[32]; splatVW(0u, zV, aV);
    float lV[32]; for (uint32_t L = 0; L < 32; ++L) lV[L] = cV[L] - aV[L];
    float dll[32]; dotW(lV, lV, dll);
    float m[32]; splatXW(dll, m);
    float dal[32]; dotW(aV, lV, dal);
    float x[32]; splatXW(dal, x);
    for (uint32_t L = 0; L < 32; ++L) x[L] = -x[L];

    if (m[0] == 0.0f)
    {
        r.ret = DEGENERATE;
        for (uint32_t L = 0; L < 32; ++L) r.cp[L] = 0.0f;
    }
    else if (x[0] < m[0])
    {
        r.ret = 0x2102u;
        for (uint32_t L = 0; L < 32; ++L) r.cp[L] = aV[L] + lV[L] * x[L] / m[L];
    }
    else
    {
        r.ret = 0x2111u;
        for (uint32_t L = 0; L < 32; ++L) r.cp[L] = cV[L];
    }
    return r;
}

PXG_TEST(npGjkClosestSeg_closest)
{
    std::mt19937 rng(1240124);
    std::vector<float> Z, CV;
    std::vector<CtRef> ref;
    size_t nLine = 0, nNew = 0, nDegen = 0;

    auto iv = [&]() -> V3 { return { float(int(rng() % 13) - 6), float(int(rng() % 13) - 6), float(int(rng() % 13) - 6) }; };

    auto emit = [&](V3 a, V3 c)
    {
        float z[32], cv[32];
        for (uint32_t L = 0; L < 32; ++L) { z[L] = float(100 + L); cv[L] = 0.0f; } // distinct sentinels
        // slot 0 = base vertex a; other slots hold sentinels (irrelevant to case 1)
        z[0] = a.x; z[1] = a.y; z[2] = a.z; z[3] = 0;
        // support vertex c replicated across all 8 vector slots (component 3 = pad 0)
        for (uint32_t s = 0; s < 8; ++s) { cv[s * 4] = c.x; cv[s * 4 + 1] = c.y; cv[s * 4 + 2] = c.z; cv[s * 4 + 3] = 0; }
        for (uint32_t L = 0; L < 32; ++L) { Z.push_back(z[L]); CV.push_back(cv[L]); }
        CtRef r = closestSegRef(z, cv);
        ref.push_back(r);
        if (r.ret == DEGENERATE) ++nDegen;
        else if (r.ret == 0x2111u) ++nNew;
        else ++nLine;
    };

    // family A: random integer segments around the origin (line + new-vertex + occasional degenerate)
    for (uint32_t iter = 0; iter < 1500; ++iter) emit(iv(), iv());
    // family B: support vertex coincident with the base vertex (degenerate, m==0)
    for (uint32_t iter = 0; iter < 200; ++iter) { V3 a = iv(); emit(a, a); }
    // family C: new-vertex region — a = k*c on the same ray so t = k/(k-1) >= 1 (origin past c)
    for (uint32_t iter = 0; iter < 400; ++iter)
    {
        V3 c; do { c = { float(int(rng() % 9) - 4), float(int(rng() % 9) - 4), float(int(rng() % 9) - 4) }; }
        while (c.x == 0 && c.y == 0 && c.z == 0);
        const int k = 2 + int(rng() % 3); // 2,3,4
        emit({ float(k) * c.x, float(k) * c.y, float(k) * c.z }, c);
    }

    const uint32_t N = (uint32_t)ref.size();
    PxgSlangComputeDevice* dev = testDevice();
    PxgBufferHandle bZ = createBufferFrom(dev, Z);
    PxgBufferHandle bCV = createBufferFrom(dev, CV);
    std::vector<float> cpInit(N * 32, -777.0f);
    std::vector<uint32_t> retInit(N, 0xEEEEEEEEu);
    PxgBufferHandle bCP = createBufferFrom(dev, cpInit);
    PxgBufferHandle bRet = createBufferFrom(dev, retInit);

    CtParams prm = { N, 0, 0, 0 };
    Kernel k("npGjkClosestSeg.spv", "ctClosestPointSeg", { 4, 4, 4, 4 }, sizeof(CtParams));
    k.dispatch((N + 63) / 64, { bZ, bCV, bCP, bRet }, prm);
    dev->flush();

    const auto gCP = downloadBuffer<float>(dev, bCP, N * 32);
    const auto gRet = downloadBuffer<uint32_t>(dev, bRet, N);

    size_t mm = 0;
    for (uint32_t t = 0; t < N; ++t)
    {
        const CtRef& r = ref[t];
        if (gRet[t] != r.ret) ++mm;
        for (uint32_t L = 0; L < 32; ++L)
            if (asU32(gCP[t * 32 + L]) != asU32(r.cp[L])) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "gjkClosestSeg: %zu mismatches", mm);
    PXG_CHECK_MSG(nLine > 200 && nNew > 100 && nDegen > 50,
                  "gjkClosestSeg coverage thin (line=%zu new=%zu degen=%zu)",
                  nLine, nNew, nDegen);

    for (PxgBufferHandle h : { bZ, bCV, bCP, bRet }) dev->destroyBuffer(h);
}

} // namespace
