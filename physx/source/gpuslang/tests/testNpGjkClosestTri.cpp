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

// Verification of npGjkClosestTri.slang: ctClosestPointTri — the GJK 2-simplex
// closest-point core (gjk.cuh :75-110).  One thread per test simulates the
// 32-lane warp; the return feature code, the 32-lane closest point, and the
// 32-lane simplex (with a possibly-set normal) are compared exactly against a
// CPU transcription that serialises the identical warp ops.  Integer simplex
// coordinates keep every dot/cross exact and the two divisions correctly-rounded,
// so the whole output is bit-for-bit reproducible.  Fixtures place a line simplex
// (2 verts) + apex in the 8-vec4 layout and sweep integer triangles around the
// origin to hit all four regions (face / edge / apex / degenerate).

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
static uint32_t next3(uint32_t x) { return x == 2u ? 0u : x + 1u; }
static uint32_t prev3(uint32_t x) { return x == 0u ? 2u : x - 1u; }
static uint32_t ffs(uint32_t m) { if (!m) return 0; uint32_t i = 0; while (!((m >> i) & 1u)) ++i; return i + 1; }

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
static void crossW(const float a[32], const float b[32], const uint32_t c0[32], const uint32_t c1[32], float o[32])
{
    for (uint32_t L = 0; L < 32; ++L) o[L] = a[c0[L]] * b[c1[L]] - a[c1[L]] * b[c0[L]];
}
static void splatXW(const float a[32], float o[32]) { for (uint32_t L = 0; L < 32; ++L) o[L] = a[L & 28u]; }
static void splatVW(uint32_t index, const float a[32], float o[32])
{
    for (uint32_t L = 0; L < 32; ++L) o[L] = a[((index << 2) | (L & 3u)) & 31u];
}

struct CtRef { uint32_t ret; float cp[32]; float zOut[32]; };
static CtRef closestTriRef(const float zVin[32], const float cV[32])
{
    CtRef r;
    float zV[32]; for (uint32_t L = 0; L < 32; ++L) zV[L] = zVin[L];
    uint32_t cp0I[32], cp1I[32];
    for (uint32_t L = 0; L < 32; ++L) { cp0I[L] = ((L & 28u) | next3(L & 3u)) & 31u; cp1I[L] = ((L & 28u) | prev3(L & 3u)) & 31u; }
    for (uint32_t L = 0; L < 32; ++L) r.cp[L] = 0.0f;

    float eV[32]; for (uint32_t L = 0; L < 32; ++L) eV[L] = cV[L] - zV[L];
    float dee[32]; dotW(eV, eV, dee);
    float lD[32]; splatXW(dee, lD);

    if ((lD[0] == 0.0f) || (lD[4] == 0.0f))
    {
        for (uint32_t L = 0; L < 32; ++L) { r.cp[L] = 0.0f; r.zOut[L] = zV[L]; }
        r.ret = DEGENERATE; return r;
    }

    float shortEdgeV[32];
    for (uint32_t L = 0; L < 32; ++L) shortEdgeV[L] = (lD[L ^ 4u] < lD[L]) ? eV[L ^ 4u] : eV[L];
    float d1[32]; for (uint32_t L = 0; L < 32; ++L) d1[L] = zV[L ^ 4u] - zV[L];
    float mV[32]; crossW(d1, shortEdgeV, cp0I, cp1I, mV);
    float pS[32]; dotW(eV, cV, pS);
    float cme[32]; crossW(mV, eV, cp0I, cp1I, cme);
    float dcme[32]; dotW(cme, cV, dcme);
    bool uS[32]; for (uint32_t L = 0; L < 32; ++L) uS[L] = dcme[L] <= 0.0f;

    if (!(uS[0] || uS[4]))
    {
        float dmc[32]; dotW(mV, cV, dmc);
        const float projD = dmc[0];
        float dmm[32]; dotW(mV, mV, dmm);
        float sxdmm[32]; splatXW(dmm, sxdmm);
        float tmp[32]; for (uint32_t L = 0; L < 32; ++L) tmp[L] = mV[L] * projD / sxdmm[L];
        splatVW(0u, tmp, r.cp);
        const bool flip = projD < 0.0f;
        for (uint32_t L = 0; L < 32; ++L) if ((L & 3u) == 3u) mV[L] = projD;
        const uint32_t index = flip ? 1u : 0u;
        for (uint32_t L = 0; L < 32; ++L)
        {
            const uint32_t src = ((index << 2) | (L & 3u)) & 31u;
            zV[L] = (L < 16u || L > 19u) ? zV[L] : mV[src];
        }
        for (uint32_t L = 0; L < 32; ++L) r.zOut[L] = zV[L];
        r.ret = flip ? 0x0123u : 0x1023u; return r;
    }

    uint32_t emask = 0u;
    if (uS[0] && (pS[0] > 0.0f)) emask |= 1u;
    if (uS[4] && (pS[4] > 0.0f)) emask |= 2u;
    uint32_t eid = ffs(emask);
    if (eid != 0u)
    {
        eid = eid - 1u;
        float sxpS[32]; splatXW(pS, sxpS);
        float tmp[32]; for (uint32_t L = 0; L < 32; ++L) tmp[L] = cV[L] - eV[L] * sxpS[L] / lD[L];
        splatVW(eid, tmp, r.cp);
        for (uint32_t L = 0; L < 32; ++L) r.zOut[L] = zV[L];
        r.ret = (eid << 8) | 0x22u; return r;
    }

    for (uint32_t L = 0; L < 32; ++L) { r.cp[L] = cV[L]; r.zOut[L] = zV[L]; }
    r.ret = 0x0021u; return r;
}

PXG_TEST(npGjkClosestTri_closest)
{
    std::mt19937 rng(1220122);
    std::vector<float> Z, CV;
    std::vector<CtRef> ref;
    size_t nFace = 0, nEdge = 0, nApex = 0, nDegen = 0;

    auto iv = [&]() -> V3 { return { float(int(rng() % 13) - 6), float(int(rng() % 13) - 6), float(int(rng() % 13) - 6) }; };

    auto emit = [&](V3 a, V3 b, V3 apex)
    {
        float z[32], c[32];
        for (uint32_t L = 0; L < 32; ++L) { z[L] = float(100 + L); c[L] = 0.0f; } // distinct sentinels
        // slot 0 = a, slot 1 = b (verts); slots 2-7 zeroed for zV
        z[0] = a.x; z[1] = a.y; z[2] = a.z; z[3] = 0;
        z[4] = b.x; z[5] = b.y; z[6] = b.z; z[7] = 0;
        for (uint32_t s = 2; s < 8; ++s) { z[s * 4] = 0; z[s * 4 + 1] = 0; z[s * 4 + 2] = 0; z[s * 4 + 3] = 0; }
        // apex replicated across all 8 vector slots (component 3 = pad 0)
        for (uint32_t s = 0; s < 8; ++s) { c[s * 4] = apex.x; c[s * 4 + 1] = apex.y; c[s * 4 + 2] = apex.z; c[s * 4 + 3] = 0; }
        for (uint32_t L = 0; L < 32; ++L) { Z.push_back(z[L]); CV.push_back(c[L]); }
        CtRef r = closestTriRef(z, c);
        ref.push_back(r);
        if (r.ret == DEGENERATE) ++nDegen;
        else if (r.ret == 0x0021u) ++nApex;
        else if ((r.ret & 0xffu) == 0x22u) ++nEdge;
        else ++nFace;
    };

    // family A: random integer triangles around the origin (all regions)
    for (uint32_t iter = 0; iter < 1500; ++iter) emit(iv(), iv(), iv());
    // family B: apex coincident with a base vert (degenerate)
    for (uint32_t iter = 0; iter < 150; ++iter)
    {
        V3 a = iv(), b = iv();
        emit(a, b, (rng() % 2) ? a : b);
    }
    // family C: apex Voronoi region — apex P near the origin with both base verts
    // beyond it (a,b ~ 2P + noise) so the origin's closest feature is the apex
    for (uint32_t iter = 0; iter < 400; ++iter)
    {
        V3 p; do { p = { float(int(rng() % 9) - 4), float(int(rng() % 9) - 4), float(int(rng() % 9) - 4) }; }
        while (p.x == 0 && p.y == 0 && p.z == 0);
        V3 n1 = { float(int(rng() % 5) - 2), float(int(rng() % 5) - 2), float(int(rng() % 5) - 2) };
        V3 n2 = { float(int(rng() % 5) - 2), float(int(rng() % 5) - 2), float(int(rng() % 5) - 2) };
        V3 a = { 2 * p.x + n1.x, 2 * p.y + n1.y, 2 * p.z + n1.z };
        V3 b = { 2 * p.x + n2.x, 2 * p.y + n2.y, 2 * p.z + n2.z };
        emit(a, b, p);
    }

    const uint32_t N = (uint32_t)ref.size();
    PxgSlangComputeDevice* dev = testDevice();
    PxgBufferHandle bZ = createBufferFrom(dev, Z);
    PxgBufferHandle bCV = createBufferFrom(dev, CV);
    std::vector<float> cpInit(N * 32, -777.0f), zoInit(N * 32, -777.0f);
    std::vector<uint32_t> retInit(N, 0xEEEEEEEEu);
    PxgBufferHandle bCP = createBufferFrom(dev, cpInit);
    PxgBufferHandle bZO = createBufferFrom(dev, zoInit);
    PxgBufferHandle bRet = createBufferFrom(dev, retInit);

    CtParams prm = { N, 0, 0, 0 };
    Kernel k("npGjkClosestTri.spv", "ctClosestPointTri", { 4, 4, 4, 4, 4 }, sizeof(CtParams));
    k.dispatch((N + 63) / 64, { bZ, bCV, bCP, bZO, bRet }, prm);
    dev->flush();

    const auto gCP = downloadBuffer<float>(dev, bCP, N * 32);
    const auto gZO = downloadBuffer<float>(dev, bZO, N * 32);
    const auto gRet = downloadBuffer<uint32_t>(dev, bRet, N);

    size_t mm = 0;
    for (uint32_t t = 0; t < N; ++t)
    {
        const CtRef& r = ref[t];
        if (gRet[t] != r.ret) ++mm;
        for (uint32_t L = 0; L < 32; ++L)
        {
            if (asU32(gCP[t * 32 + L]) != asU32(r.cp[L])) ++mm;
            if (asU32(gZO[t * 32 + L]) != asU32(r.zOut[L])) ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "gjkClosestTri: %zu mismatches", mm);
    PXG_CHECK_MSG(nFace > 100 && nEdge > 100 && nApex > 30 && nDegen > 100,
                  "gjkClosestTri coverage thin (face=%zu edge=%zu apex=%zu degen=%zu)",
                  nFace, nEdge, nApex, nDegen);

    for (PxgBufferHandle h : { bZ, bCV, bCP, bZO, bRet }) dev->destroyBuffer(h);
}

} // namespace
