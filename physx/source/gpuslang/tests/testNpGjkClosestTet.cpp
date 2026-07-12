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

// Verification of npGjkClosestTet.slang: ctClosestPointTet — the GJK 3-simplex
// closest-point core (gjk.cuh :113-169).  One thread per test simulates the
// 32-lane warp; the return feature code, the 32-lane closest point, and the
// 32-lane simplex (with a possibly-set normal) are compared exactly against a CPU
// transcription that serialises the identical warp ops.  Integer simplex
// coordinates keep every dot/cross exact and the divisions correctly-rounded, so
// the whole output is bit-for-bit reproducible.  Fixtures place a triangle
// simplex (3 verts) + apex in the 8-vec4 layout and sweep integer tetrahedra to
// hit all regions (inside / apex / edge / face / the several degeneracies).

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
static CtRef closestTetRef(const float zVin[32], const float dV[32])
{
    CtRef r;
    float zV[32]; for (uint32_t L = 0; L < 32; ++L) zV[L] = zVin[L];
    for (uint32_t L = 0; L < 32; ++L) r.cp[L] = 0.0f;

    float n0V[32]; splatVW(4u, zV, n0V);
    float dn0d[32]; dotW(n0V, dV, dn0d);
    float dn0z[32]; dotW(n0V, zV, dn0z);
    if (dn0d[0] > dn0z[0]) { for (uint32_t L = 0; L < 32; ++L) r.zOut[L] = zV[L]; r.ret = DEGENERATE; return r; }

    uint32_t cp0I[32], cp1I[32], vNextI[32], vPrevI[32];
    for (uint32_t L = 0; L < 32; ++L)
    {
        cp0I[L] = ((L & 28u) | next3(L & 3u)) & 31u; cp1I[L] = ((L & 28u) | prev3(L & 3u)) & 31u;
        vNextI[L] = (L > 7u) ? (L - 8u) : (L + 4u); vPrevI[L] = (L < 4u) ? (L + 8u) : (L - 4u);
    }

    float eV[32]; for (uint32_t L = 0; L < 32; ++L) eV[L] = dV[L] - zV[L];
    float dee[32]; dotW(eV, eV, dee);
    float lD[32]; splatXW(dee, lD);
    if ((lD[0] == 0.0f) || (lD[4] == 0.0f) || (lD[8] == 0.0f)) { for (uint32_t L = 0; L < 32; ++L) r.zOut[L] = zV[L]; r.ret = DEGENERATE; return r; }

    float shortEdgeV[32];
    for (uint32_t L = 0; L < 32; ++L) shortEdgeV[L] = (lD[vNextI[L]] < lD[L]) ? eV[vNextI[L]] : eV[L];
    float d2[32]; for (uint32_t L = 0; L < 32; ++L) d2[L] = zV[vNextI[L]] - zV[L];
    float nV[32]; crossW(shortEdgeV, d2, cp0I, cp1I, nV);
    float aS[32]; dotW(nV, dV, aS);
    float zvp[32]; for (uint32_t L = 0; L < 32; ++L) zvp[L] = zV[vPrevI[L]];
    float gS[32]; dotW(zvp, nV, gS);
    if ((aS[0] <= gS[0]) || (aS[4] <= gS[4]) || (aS[8] <= gS[8])) { for (uint32_t L = 0; L < 32; ++L) r.zOut[L] = zV[L]; r.ret = DEGENERATE; return r; }

    for (uint32_t L = 0; L < 32; ++L) if ((L & 3u) == 3u) nV[L] = aS[L - 3u];

    if ((aS[0] > 0.0f) && (aS[4] > 0.0f) && (aS[8] > 0.0f))
    {
        for (uint32_t L = 0; L < 32; ++L) if (L >= 20u) zV[L] = nV[L - 20u];
        for (uint32_t L = 0; L < 32; ++L) r.zOut[L] = zV[L];
        r.ret = 0x2104u; return r;
    }

    float pS[32]; dotW(eV, dV, pS);
    if ((pS[0] <= 0.0f) && (pS[4] <= 0.0f) && (pS[8] <= 0.0f))
    {
        for (uint32_t L = 0; L < 32; ++L) { r.cp[L] = dV[L]; r.zOut[L] = zV[L]; }
        r.ret = 0x2131u; return r;
    }

    float cen[32]; crossW(eV, nV, cp0I, cp1I, cen);
    float dcen[32]; dotW(cen, dV, dcen);
    bool uiS[32]; for (uint32_t L = 0; L < 32; ++L) uiS[L] = dcen[L] <= 0.0f;
    float evn[32]; for (uint32_t L = 0; L < 32; ++L) evn[L] = eV[vNextI[L]];
    float cevn[32]; crossW(evn, nV, cp0I, cp1I, cevn);
    float dcevn[32]; dotW(cevn, dV, dcevn);
    bool ujS[32]; for (uint32_t L = 0; L < 32; ++L) ujS[L] = dcevn[L] > 0.0f;

    uint32_t emask = 0u;
    if (uiS[0] && ujS[vPrevI[0]] && (pS[0] >= 0.0f)) emask |= 1u;
    if (uiS[4] && ujS[vPrevI[4]] && (pS[4] >= 0.0f)) emask |= 2u;
    if (uiS[8] && ujS[vPrevI[8]] && (pS[8] >= 0.0f)) emask |= 4u;
    uint32_t eid = ffs(emask);
    if (eid != 0u)
    {
        eid = eid - 1u;
        float pl[32]; for (uint32_t L = 0; L < 32; ++L) pl[L] = pS[L] / lD[L];
        float spl[32]; splatXW(pl, spl);
        float tmp[32]; for (uint32_t L = 0; L < 32; ++L) tmp[L] = dV[L] - eV[L] * spl[L];
        splatVW(eid, tmp, r.cp);
        for (uint32_t L = 0; L < 32; ++L) r.zOut[L] = zV[L];
        r.ret = (eid << 8) | 0x32u; return r;
    }

    uint32_t fmask = 0u;
    if (!uiS[0] && !ujS[0] && (aS[0] <= 0.0f)) fmask |= 1u;
    if (!uiS[4] && !ujS[4] && (aS[4] <= 0.0f)) fmask |= 2u;
    if (!uiS[8] && !ujS[8] && (aS[8] <= 0.0f)) fmask |= 4u;
    uint32_t fid = ffs(fmask);
    if (fid != 0u)
    {
        fid = fid - 1u;
        float dnn[32]; dotW(nV, nV, dnn);
        float ad[32]; for (uint32_t L = 0; L < 32; ++L) ad[L] = aS[L] / dnn[L];
        float sad[32]; splatXW(ad, sad);
        float tmp[32]; for (uint32_t L = 0; L < 32; ++L) tmp[L] = nV[L] * sad[L];
        splatVW(fid, tmp, r.cp);
        for (uint32_t L = 0; L < 32; ++L)
        {
            const uint32_t src = ((fid << 2) | (L & 3u)) & 31u;
            zV[L] = (L < 16u || L > 19u) ? zV[L] : -nV[src];
        }
        for (uint32_t L = 0; L < 32; ++L) r.zOut[L] = zV[L];
        r.ret = (next3(fid) << 12) | (fid << 8) | 0x33u; return r;
    }

    for (uint32_t L = 0; L < 32; ++L) r.zOut[L] = zV[L];
    r.ret = DEGENERATE; return r;
}

PXG_TEST(npGjkClosestTet_closest)
{
    std::mt19937 rng(1230123);
    std::vector<float> Z, DV;
    std::vector<CtRef> ref;
    size_t nInside = 0, nApex = 0, nEdge = 0, nFace = 0, nDegen = 0;

    auto iv = [&](int L) -> V3 { return { float(int(rng() % (2 * L + 1)) - L), float(int(rng() % (2 * L + 1)) - L), float(int(rng() % (2 * L + 1)) - L) }; };

    auto emit = [&](V3 a, V3 b, V3 c, V3 apex, V3 n0)
    {
        float z[32], d[32];
        for (uint32_t L = 0; L < 32; ++L) { z[L] = float(500 + L); d[L] = 0.0f; }
        z[0] = a.x; z[1] = a.y; z[2] = a.z; z[3] = 0;
        z[4] = b.x; z[5] = b.y; z[6] = b.z; z[7] = 0;
        z[8] = c.x; z[9] = c.y; z[10] = c.z; z[11] = 0;
        z[12] = 0; z[13] = 0; z[14] = 0; z[15] = 0;              // slot 3 (unused base)
        z[16] = n0.x; z[17] = n0.y; z[18] = n0.z; z[19] = 0;      // normal slot 0 = n0V
        // lanes 20-31 keep the 500+L sentinels (normal slots 1-3, verify setNormal123)
        for (uint32_t s = 0; s < 8; ++s) { d[s * 4] = apex.x; d[s * 4 + 1] = apex.y; d[s * 4 + 2] = apex.z; d[s * 4 + 3] = 0; }
        for (uint32_t L = 0; L < 32; ++L) { Z.push_back(z[L]); DV.push_back(d[L]); }
        CtRef r = closestTetRef(z, d);
        ref.push_back(r);
        if (r.ret == DEGENERATE) ++nDegen;
        else if (r.ret == 0x2104u) ++nInside;
        else if (r.ret == 0x2131u) ++nApex;
        else if ((r.ret & 0xffu) == 0x32u) ++nEdge;
        else if ((r.ret & 0xffu) == 0x33u) ++nFace;
    };

    // family A: random integer tetrahedra around origin, n0V = 0 (skip that check)
    for (uint32_t iter = 0; iter < 2000; ++iter) emit(iv(6), iv(6), iv(6), iv(6), V3{ 0, 0, 0 });
    // family B: apex coincident with a base vert (edge-length-0 degeneracy)
    for (uint32_t iter = 0; iter < 150; ++iter)
    {
        V3 a = iv(6), b = iv(6), c = iv(6);
        const V3 vs[3] = { a, b, c };
        emit(a, b, c, vs[rng() % 3], V3{ 0, 0, 0 });
    }
    // family C: apex Voronoi — apex near origin, base verts beyond (2*apex + noise)
    for (uint32_t iter = 0; iter < 400; ++iter)
    {
        V3 p; do { p = iv(4); } while (p.x == 0 && p.y == 0 && p.z == 0);
        auto beyond = [&]() -> V3 { V3 n = iv(2); return { 2 * p.x + n.x, 2 * p.y + n.y, 2 * p.z + n.z }; };
        emit(beyond(), beyond(), beyond(), p, V3{ 0, 0, 0 });
    }
    // family D: n0V degeneracy — nonzero normal slot 0, apex beyond the old face
    for (uint32_t iter = 0; iter < 300; ++iter)
        emit(iv(5), iv(5), iv(5), iv(8), iv(4));

    const uint32_t N = (uint32_t)ref.size();
    PxgSlangComputeDevice* dev = testDevice();
    PxgBufferHandle bZ = createBufferFrom(dev, Z);
    PxgBufferHandle bDV = createBufferFrom(dev, DV);
    std::vector<float> cpInit(N * 32, -777.0f), zoInit(N * 32, -777.0f);
    std::vector<uint32_t> retInit(N, 0xEEEEEEEEu);
    PxgBufferHandle bCP = createBufferFrom(dev, cpInit);
    PxgBufferHandle bZO = createBufferFrom(dev, zoInit);
    PxgBufferHandle bRet = createBufferFrom(dev, retInit);

    CtParams prm = { N, 0, 0, 0 };
    Kernel k("npGjkClosestTet.spv", "ctClosestPointTet", { 4, 4, 4, 4, 4 }, sizeof(CtParams));
    k.dispatch((N + 63) / 64, { bZ, bDV, bCP, bZO, bRet }, prm);
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
    PXG_CHECK_MSG(mm == 0, "gjkClosestTet: %zu mismatches", mm);
    PXG_CHECK_MSG(nInside > 20 && nApex > 30 && nEdge > 100 && nFace > 100 && nDegen > 200,
                  "gjkClosestTet coverage thin (inside=%zu apex=%zu edge=%zu face=%zu degen=%zu)",
                  nInside, nApex, nEdge, nFace, nDegen);

    for (PxgBufferHandle h : { bZ, bDV, bCP, bZO, bRet }) dev->destroyBuffer(h);
}

} // namespace
