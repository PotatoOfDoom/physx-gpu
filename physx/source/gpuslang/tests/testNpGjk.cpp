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

// Verification of npGjk.slang: ctGjk — the full cold-start gjk() driver
// (gjk.cuh :235-418).  One thread per test runs the whole GJK do-while over a
// pair of integer convex point clouds; the result enum, bounds, degenerate flag,
// stored direction and the three witness points are compared bit-exactly against
// a CPU transcription that mirrors the shader line-for-line.  Integer clouds keep
// every dot/cross exact, magnitude a correctly-rounded sqrt and every division a
// correctly-rounded OpFDiv, so the entire iteration is bit-for-bit reproducible.

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
struct GjkParams { uint32_t numTests, maxV, pad0, pad1; };
struct GjkIn { float initialDir[4]; uint32_t cfg[4]; float params[4]; };
struct GjkOut
{
    uint32_t ret, degenerate, size, pad;
    float lowerBound, upperBound, p1, p2;
    float direction[4], closestPointA[4], closestPointB[4], closestPointDir[4];
};

static const uint32_t DEGENERATE = 0xffffu;
static const float PX_MAX_F32 = 3.4028234663852886e38f;
static const uint32_t GJK_DISTANT = 0u, GJK_CLOSE = 1u, GJK_OVERLAP = 2u;

static uint32_t asU32(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float asF32(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
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
static void magW(const float v[32], float cm[32])
{
    float dv[32]; dotW(v, v, dv);
    for (uint32_t L = 0; L < 32; ++L) cm[L] = std::sqrt(dv[L & 28u]);
}
static uint32_t encodeIndices(uint32_t a, uint32_t b) { return 0x3f800000u | ((b & 0xffu) << 8) | (a & 0xffu); }

static V3 crossX(V3 a, V3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static V3 absX(V3 a) { return { std::fabs(a.x), std::fabs(a.y), std::fabs(a.z) }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 sub3(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 getNormalized(V3 v)
{
    const float m = dot3(v, v);
    if (m > 0.0f) { const float r = 1.0f / std::sqrt(m); return { v.x * r, v.y * r, v.z * r }; }
    return { 0, 0, 0 };
}

static uint32_t closestPointTriW(float zV[32], const float cV[32], float closestPoint[32])
{
    uint32_t cp0I[32], cp1I[32];
    for (uint32_t L = 0; L < 32; ++L) { cp0I[L] = ((L & 28u) | next3(L & 3u)) & 31u; cp1I[L] = ((L & 28u) | prev3(L & 3u)) & 31u; }
    for (uint32_t L = 0; L < 32; ++L) closestPoint[L] = 0.0f;
    float eV[32]; for (uint32_t L = 0; L < 32; ++L) eV[L] = cV[L] - zV[L];
    float dee[32]; dotW(eV, eV, dee);
    float lD[32]; splatXW(dee, lD);
    if ((lD[0] == 0.0f) || (lD[4] == 0.0f)) return DEGENERATE;
    float shortEdgeV[32]; for (uint32_t L = 0; L < 32; ++L) shortEdgeV[L] = (lD[L ^ 4u] < lD[L]) ? eV[L ^ 4u] : eV[L];
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
        splatVW(0u, tmp, closestPoint);
        const bool flip = projD < 0.0f;
        for (uint32_t L = 0; L < 32; ++L) if ((L & 3u) == 3u) mV[L] = projD;
        const uint32_t index = flip ? 1u : 0u;
        for (uint32_t L = 0; L < 32; ++L) { const uint32_t src = ((index << 2) | (L & 3u)) & 31u; zV[L] = (L < 16u || L > 19u) ? zV[L] : mV[src]; }
        return flip ? 0x0123u : 0x1023u;
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
        splatVW(eid, tmp, closestPoint);
        return (eid << 8) | 0x22u;
    }
    for (uint32_t L = 0; L < 32; ++L) closestPoint[L] = cV[L];
    return 0x0021u;
}

static uint32_t closestPointTetW(float zV[32], const float dV[32], float closestPoint[32])
{
    for (uint32_t L = 0; L < 32; ++L) closestPoint[L] = 0.0f;
    float n0V[32]; splatVW(4u, zV, n0V);
    float dn0d[32]; dotW(n0V, dV, dn0d);
    float dn0z[32]; dotW(n0V, zV, dn0z);
    if (dn0d[0] > dn0z[0]) return DEGENERATE;
    uint32_t cp0I[32], cp1I[32], vNextI[32], vPrevI[32];
    for (uint32_t L = 0; L < 32; ++L)
    {
        cp0I[L] = ((L & 28u) | next3(L & 3u)) & 31u; cp1I[L] = ((L & 28u) | prev3(L & 3u)) & 31u;
        vNextI[L] = (L > 7u) ? (L - 8u) : (L + 4u); vPrevI[L] = (L < 4u) ? (L + 8u) : (L - 4u);
    }
    float eV[32]; for (uint32_t L = 0; L < 32; ++L) eV[L] = dV[L] - zV[L];
    float dee[32]; dotW(eV, eV, dee);
    float lD[32]; splatXW(dee, lD);
    if ((lD[0] == 0.0f) || (lD[4] == 0.0f) || (lD[8] == 0.0f)) return DEGENERATE;
    float shortEdgeV[32]; for (uint32_t L = 0; L < 32; ++L) shortEdgeV[L] = (lD[vNextI[L]] < lD[L]) ? eV[vNextI[L]] : eV[L];
    float d2[32]; for (uint32_t L = 0; L < 32; ++L) d2[L] = zV[vNextI[L]] - zV[L];
    float nV[32]; crossW(shortEdgeV, d2, cp0I, cp1I, nV);
    float aS[32]; dotW(nV, dV, aS);
    float zvp[32]; for (uint32_t L = 0; L < 32; ++L) zvp[L] = zV[vPrevI[L]];
    float gS[32]; dotW(zvp, nV, gS);
    if ((aS[0] <= gS[0]) || (aS[4] <= gS[4]) || (aS[8] <= gS[8])) return DEGENERATE;
    for (uint32_t L = 0; L < 32; ++L) if ((L & 3u) == 3u) nV[L] = aS[L - 3u];
    if ((aS[0] > 0.0f) && (aS[4] > 0.0f) && (aS[8] > 0.0f))
    {
        for (uint32_t L = 0; L < 32; ++L) if (L >= 20u) zV[L] = nV[L - 20u];
        return 0x2104u;
    }
    float pS[32]; dotW(eV, dV, pS);
    if ((pS[0] <= 0.0f) && (pS[4] <= 0.0f) && (pS[8] <= 0.0f))
    {
        for (uint32_t L = 0; L < 32; ++L) closestPoint[L] = dV[L];
        return 0x2131u;
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
        splatVW(eid, tmp, closestPoint);
        return (eid << 8) | 0x32u;
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
        splatVW(fid, tmp, closestPoint);
        for (uint32_t L = 0; L < 32; ++L) { const uint32_t src = ((fid << 2) | (L & 3u)) & 31u; zV[L] = (L < 16u || L > 19u) ? zV[L] : -nV[src]; }
        return (next3(fid) << 12) | (fid << 8) | 0x33u;
    }
    return DEGENERATE;
}

static uint32_t supportIdx(V3 dir, const std::vector<float>& V, uint32_t vbase, uint32_t nb)
{
    uint32_t bestN = 0xFFFFFFFFu; float bestDist = -PX_MAX_F32;
    for (uint32_t i = 0; i < nb; i += 32)
    {
        float d[32];
        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            const uint32_t n = i + lane;
            V3 v = { V[(vbase + n) * 4], V[(vbase + n) * 4 + 1], V[(vbase + n) * 4 + 2] };
            d[lane] = (n < nb) ? dot3(dir, v) : -PX_MAX_F32;
        }
        float chunkMax = -PX_MAX_F32; for (uint32_t lane = 0; lane < 32; ++lane) chunkMax = std::fmax(chunkMax, d[lane]);
        if (chunkMax > bestDist) { uint32_t ll = 0; for (uint32_t lane = 0; lane < 32; ++lane) if (d[lane] == chunkMax) { ll = lane; break; } bestDist = chunkMax; bestN = i + ll; }
    }
    return bestN;
}

static GjkOut gjkRef(const GjkIn& e, const std::vector<float>& VA, const std::vector<float>& VB, uint32_t vbase)
{
    const uint32_t nbA = e.cfg[0], nbB = e.cfg[1];
    const float minSep = e.params[0], convergenceRatio = e.params[1];
    const V3 initialUnitDir = { e.initialDir[0], e.initialDir[1], e.initialDir[2] };

    float simplexV[32]; for (uint32_t L = 0; L < 32; ++L) simplexV[L] = 0.0f;
    float closestPointV[32];
    for (uint32_t L = 0; L < 32; ++L)
    {
        const uint32_t c = L & 3u;
        closestPointV[L] = (c == 0) ? initialUnitDir.x : (c == 1) ? initialUnitDir.y : (c == 2) ? initialUnitDir.z : PX_MAX_F32;
    }
    float upperBound = PX_MAX_F32, lowerBound = -PX_MAX_F32;
    float cmA[32]; magW(closestPointV, cmA);
    bool degenerate = false;
    int size = 0, backupSize = 0;
    float backupSimplexV[32]; for (uint32_t L = 0; L < 32; ++L) backupSimplexV[L] = 0.0f;
    float backupClosestV[32]; for (uint32_t L = 0; L < 32; ++L) backupClosestV[L] = 0.0f;
    V3 directionOut = { 0, 0, 0 };

    bool run = true;
    while (run)
    {
        float dV[32]; for (uint32_t L = 0; L < 32; ++L) dV[L] = closestPointV[L] / cmA[L];
        const V3 dvVec = { dV[0], dV[1], dV[2] };
        const uint32_t indexA = supportIdx({ -dvVec.x, -dvVec.y, -dvVec.z }, VA, vbase, nbA);
        const uint32_t indexB = supportIdx(dvVec, VB, vbase, nbB);
        const V3 vA = { VA[(vbase + indexA) * 4], VA[(vbase + indexA) * 4 + 1], VA[(vbase + indexA) * 4 + 2] };
        const V3 vB = { VB[(vbase + indexB) * 4], VB[(vbase + indexB) * 4 + 1], VB[(vbase + indexB) * 4 + 2] };
        const V3 sv = { vA.x - vB.x, vA.y - vB.y, vA.z - vB.z };
        float supportVertexV[32];
        for (uint32_t L = 0; L < 32; ++L) { const uint32_t c = L & 3u; supportVertexV[L] = (c == 0) ? sv.x : (c == 1) ? sv.y : (c == 2) ? sv.z : 0.0f; }
        const float encoded = asF32(encodeIndices(indexA, indexB));
        for (uint32_t L = 0; L < 32; ++L) if ((L >> 2) == (uint32_t)size) simplexV[L] = ((L & 3u) < 3u) ? supportVertexV[L] : encoded;

        float dds[32]; dotW(dV, supportVertexV, dds);
        const float lb = dds[0];
        if (lb > lowerBound)
        {
            directionOut = { -dV[0], -dV[1], -dV[2] };
            lowerBound = lb;
            if (lowerBound > minSep || lowerBound >= upperBound * convergenceRatio) { run = false; break; }
        }

        uint32_t f = 0u;
        float pointV[32];
        if (size == 0) { f = 0x2101u; for (uint32_t L = 0; L < 32; ++L) pointV[L] = supportVertexV[L]; }
        else if (size == 1)
        {
            float aV[32]; splatVW(0u, simplexV, aV);
            float lV[32]; for (uint32_t L = 0; L < 32; ++L) lV[L] = supportVertexV[L] - aV[L];
            float dll[32]; dotW(lV, lV, dll);
            float mm[32]; splatXW(dll, mm);
            float dal[32]; dotW(aV, lV, dal);
            float xx[32]; splatXW(dal, xx); for (uint32_t L = 0; L < 32; ++L) xx[L] = -xx[L];
            if (mm[0] == 0.0f) { f = DEGENERATE; for (uint32_t L = 0; L < 32; ++L) pointV[L] = 0.0f; }
            else if (xx[0] < mm[0]) { f = 0x2102u; for (uint32_t L = 0; L < 32; ++L) pointV[L] = aV[L] + lV[L] * xx[L] / mm[L]; }
            else { f = 0x2111u; for (uint32_t L = 0; L < 32; ++L) pointV[L] = supportVertexV[L]; }
        }
        else if (size == 2) { f = closestPointTriW(simplexV, supportVertexV, pointV); }
        else { f = closestPointTetW(simplexV, supportVertexV, pointV); }

        magW(pointV, cmA);
        const int newSize = (int)(f & 0xfu);
        degenerate = (f == DEGENERATE) || (cmA[0] >= upperBound && newSize <= size);
        if (degenerate) { run = false; break; }

        backupSize = (cmA[0] >= upperBound) ? size : 0;
        if (backupSize != 0) { for (uint32_t L = 0; L < 32; ++L) { backupSimplexV[L] = simplexV[L]; backupClosestV[L] = closestPointV[L]; } }
        else upperBound = cmA[0];

        for (uint32_t L = 0; L < 32; ++L) closestPointV[L] = pointV[L];
        size = newSize;

        float tmpP[32];
        for (uint32_t L = 0; L < 32; ++L) { const uint32_t sourceIndex = (f >> ((L & 28u) + 4u)) & 0xfu; tmpP[L] = simplexV[((sourceIndex << 2) | (L & 3u)) & 31u]; }
        for (uint32_t L = 0; L < 32; ++L) if (L < 12u) simplexV[L] = tmpP[L];

        if (!(upperBound > 0.0f && lowerBound < upperBound * convergenceRatio)) run = false;
    }

    if (backupSize > 0) { for (uint32_t L = 0; L < 32; ++L) { simplexV[L] = backupSimplexV[L]; closestPointV[L] = backupClosestV[L]; } size = backupSize; }

    GjkOut o; std::memset(&o, 0, sizeof(o));
    o.degenerate = degenerate ? 1u : 0u; o.lowerBound = lowerBound; o.upperBound = upperBound; o.size = (uint32_t)size;
    o.direction[0] = directionOut.x; o.direction[1] = directionOut.y; o.direction[2] = directionOut.z;

    if (upperBound == 0.0f) { o.ret = GJK_OVERLAP; return o; }
    if (lowerBound > minSep) { o.ret = GJK_DISTANT; return o; }

    const V3 cp = { closestPointV[0], closestPointV[1], closestPointV[2] };
    const V3 gn = getNormalized(cp);
    const V3 dir = { -gn.x, -gn.y, -gn.z };
    o.closestPointDir[0] = dir.x; o.closestPointDir[1] = dir.y; o.closestPointDir[2] = dir.z;

    const uint32_t a0 = asU32(simplexV[3]) & 0xffu, a1 = asU32(simplexV[7]) & 0xffu, a2 = asU32(simplexV[11]) & 0xffu;
    auto va = [&](uint32_t i) -> V3 { return { VA[(vbase + i) * 4], VA[(vbase + i) * 4 + 1], VA[(vbase + i) * 4 + 2] }; };
    V3 cpA;
    if (size == 3)
    {
        const V3 A = { simplexV[0], simplexV[1], simplexV[2] }, B = { simplexV[4], simplexV[5], simplexV[6] }, C = { simplexV[8], simplexV[9], simplexV[10] };
        const V3 m = crossX(sub3(B, A), sub3(C, A));
        const V3 ma = absX(m);
        const V3 s1 = crossX(sub3(B, cp), sub3(C, cp));
        const V3 s2 = crossX(sub3(C, cp), sub3(A, cp));
        float d, u, v;
        if (ma.x >= ma.y && ma.x >= ma.z) { d = 1.0f / m.x; u = s1.x * d; v = s2.x * d; }
        else if (ma.y >= ma.z)            { d = 1.0f / m.y; u = s1.y * d; v = s2.y * d; }
        else                              { d = 1.0f / m.z; u = s1.z * d; v = s2.z * d; }
        const float w = 1.0f - u - v;
        const V3 p0 = va(a0), p1 = va(a1), p2 = va(a2);
        cpA = { u * p0.x + v * p1.x + w * p2.x, u * p0.y + v * p1.y + w * p2.y, u * p0.z + v * p1.z + w * p2.z };
    }
    else if (size == 2)
    {
        const V3 V0 = { simplexV[0], simplexV[1], simplexV[2] }, V1 = { simplexV[4], simplexV[5], simplexV[6] };
        const V3 dv = sub3(V1, V0);
        const float q = -dot3(V0, dv) / dot3(dv, dv);
        const V3 p0 = va(a0), p1 = va(a1);
        cpA = { (1.0f - q) * p0.x + q * p1.x, (1.0f - q) * p0.y + q * p1.y, (1.0f - q) * p0.z + q * p1.z };
    }
    else cpA = va(a0);

    o.closestPointA[0] = cpA.x; o.closestPointA[1] = cpA.y; o.closestPointA[2] = cpA.z;
    o.closestPointB[0] = cpA.x + upperBound * dir.x; o.closestPointB[1] = cpA.y + upperBound * dir.y; o.closestPointB[2] = cpA.z + upperBound * dir.z;
    o.ret = GJK_CLOSE;
    return o;
}

PXG_TEST(npGjk_driver)
{
    std::mt19937 rng(1290129);
    const uint32_t MAXV = 32;
    std::vector<float> VA, VB;
    std::vector<GjkIn> in;
    std::vector<GjkOut> ref;
    size_t nOverlap = 0, nDistant = 0, nClose = 0, nDegen = 0;

    auto iv = [&](int r) -> V3 { return { float(int(rng() % (2 * r + 1)) - r), float(int(rng() % (2 * r + 1)) - r), float(int(rng() % (2 * r + 1)) - r) }; };

    auto emit = [&](uint32_t nbA, uint32_t nbB, V3 shiftB, V3 dir0, float minSep)
    {
        const uint32_t vbase = (uint32_t)(VA.size() / 4);
        for (uint32_t k = 0; k < MAXV; ++k)
        {
            V3 a = (k < nbA) ? iv(6) : V3{ 0, 0, 0 };
            V3 b = (k < nbB) ? iv(6) : V3{ 0, 0, 0 };
            b = { b.x + shiftB.x, b.y + shiftB.y, b.z + shiftB.z };
            VA.push_back(a.x); VA.push_back(a.y); VA.push_back(a.z); VA.push_back(0);
            VB.push_back(b.x); VB.push_back(b.y); VB.push_back(b.z); VB.push_back(0);
        }
        GjkIn e; std::memset(&e, 0, sizeof(e));
        e.initialDir[0] = dir0.x; e.initialDir[1] = dir0.y; e.initialDir[2] = dir0.z;
        e.cfg[0] = nbA; e.cfg[1] = nbB;
        e.params[0] = minSep; e.params[1] = 0.98f;
        in.push_back(e);
        GjkOut r = gjkRef(e, VA, VB, vbase);
        ref.push_back(r);
        if (r.degenerate) ++nDegen;
        if (r.ret == GJK_OVERLAP) ++nOverlap; else if (r.ret == GJK_DISTANT) ++nDistant; else ++nClose;
    };

    auto dir0 = [&]() -> V3 { V3 d; do { d = iv(4); } while (d.x == 0 && d.y == 0 && d.z == 0); return d; };

    // overlapping: B shifted little -> Minkowski diff contains origin (upperBound -> 0)
    for (uint32_t it = 0; it < 500; ++it) emit(6 + rng() % 20, 6 + rng() % 20, iv(2), dir0(), 0.25f);
    // close: B shifted moderately with a LARGE minSep so the true distance (< minSep)
    // yields eCLOSE -> exercises the witness-point output (barycentrics/getNormalized)
    // across vertex/edge/face closest features (simplex size 1/2/3).
    for (uint32_t it = 0; it < 600; ++it) emit(6 + rng() % 20, 6 + rng() % 20, { float(15 + int(rng() % 10)), float(int(rng() % 9) - 4), float(int(rng() % 9) - 4) }, dir0(), 1000.0f);
    // distant: B shifted far, minSep small -> lowerBound > minSep -> eDISTANT
    for (uint32_t it = 0; it < 400; ++it) emit(6 + rng() % 20, 6 + rng() % 20, { float(40 + int(rng() % 20)), float(int(rng() % 9) - 4), float(int(rng() % 9) - 4) }, dir0(), 0.25f);

    const uint32_t N = (uint32_t)ref.size();
    PxgSlangComputeDevice* dev = testDevice();
    PxgBufferHandle bVA = createBufferFrom(dev, VA);
    PxgBufferHandle bVB = createBufferFrom(dev, VB);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    std::vector<GjkOut> outInit(N);
    std::memset(outInit.data(), 0xEE, outInit.size() * sizeof(GjkOut));
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    GjkParams prm = { N, MAXV, 0, 0 };
    Kernel k("npGjk.spv", "ctGjk", { 16, 16, (uint32_t)sizeof(GjkIn), (uint32_t)sizeof(GjkOut) }, sizeof(GjkParams));
    k.dispatch((N + 63) / 64, { bVA, bVB, bIn, bOut }, prm);
    dev->flush();

    const auto gOut = downloadBuffer<GjkOut>(dev, bOut, N);

    size_t mm = 0;
    auto ckv = [&](const float g[4], const float r[4]) { return asU32(g[0]) == asU32(r[0]) && asU32(g[1]) == asU32(r[1]) && asU32(g[2]) == asU32(r[2]); };
    for (uint32_t t = 0; t < N; ++t)
    {
        const GjkOut& r = ref[t]; const GjkOut& g = gOut[t];
        if (g.ret != r.ret) { ++mm; continue; }
        if (g.degenerate != r.degenerate) ++mm;
        if (g.size != r.size) ++mm;
        if (asU32(g.lowerBound) != asU32(r.lowerBound)) ++mm;
        if (asU32(g.upperBound) != asU32(r.upperBound)) ++mm;
        if (r.ret == GJK_CLOSE)
        {
            if (!ckv(g.closestPointDir, r.closestPointDir)) ++mm;
            if (!ckv(g.closestPointA, r.closestPointA)) ++mm;
            if (!ckv(g.closestPointB, r.closestPointB)) ++mm;
        }
        if (r.ret != GJK_OVERLAP) if (!ckv(g.direction, r.direction)) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "gjk driver: %zu mismatches", mm);
    PXG_CHECK_MSG(nOverlap > 50 && nDistant > 50 && nClose > 50,
                  "gjk driver coverage thin (overlap=%zu distant=%zu close=%zu degen=%zu)", nOverlap, nDistant, nClose, nDegen);

    for (PxgBufferHandle h : { bVA, bVB, bIn, bOut }) dev->destroyBuffer(h);
}

// ===========================================================================
// P8.6 — tetPrimitivesCollide2 (pxgSlangTetCollide), the GJK-then-EPA driver.
//
// ctGjk and ctEpa pin the arithmetic underneath this; what is left to pin is the driver's
// POLICY, and it is checked by OUTCOME CLASS rather than against a CPU reference: porting
// GJK+EPA to C++ a second time to check a thirty-line router would be the very duplication
// the extraction just removed.  The three cases below have unambiguous answers by
// construction -- a point far outside a tetrahedron, one deep inside it, and one just
// outside within the contact distance -- so the routing is observable without one.
struct TcIn  { float searchDir[4]; float spheres[4]; uint32_t counts[4]; };
struct TcOut
{
    uint32_t result, degenerate, size, pad;
    float lowerBound, upperBound, innerSphere, pad2;
    float direction[4], closestPointA[4], closestPointB[4], tetCenter[4];
};

PXG_TEST(npTetCollide_driver)
{
    const uint32_t MAXV = 64u;
    const uint32_t N = 3u;
    std::vector<float> VA(N * MAXV * 4, 0.0f), VB(N * MAXV * 4, 0.0f);
    std::vector<TcIn> in(N);
    std::memset(in.data(), 0, in.size() * sizeof(TcIn));

    // The tetrahedron is the unit tet at the origin in every case; only the rigid POINT
    // moves.  Shape A is a single vertex, which is what a sphere reduces to (its radius is
    // folded into the contact distance by the caller, exactly as sbPrimitiveCollision does).
    const float tet[4][3] = { {0,0,0}, {1,0,0}, {0,1,0}, {0,0,1} };
    const float pts[N][3] = {
        { 10.0f, 10.0f, 10.0f },   // 0: far outside            -> DISTANT, EPA never runs
        {  0.2f,  0.2f,  0.2f },   // 1: deep inside            -> OVERLAP via EPA
        { -0.05f, 0.25f, 0.25f },  // 2: just outside the x face -> CLOSE, EPA never runs
    };
    const float cDist[N] = { 0.5f, 0.5f, 0.5f };
    for (uint32_t t = 0; t < N; ++t)
    {
        const uint32_t vb = t * MAXV * 4;
        for (int c = 0; c < 3; ++c) VA[vb + c] = pts[t][c];
        for (int v = 0; v < 4; ++v) for (int c = 0; c < 3; ++c) VB[vb + v * 4 + c] = tet[v][c];
        in[t].counts[0] = 1u; in[t].counts[1] = 4u;
        in[t].searchDir[0] = pts[t][0]; in[t].searchDir[1] = pts[t][1]; in[t].searchDir[2] = pts[t][2];
        in[t].searchDir[3] = cDist[t];
        in[t].spheres[0] = 0.0f;          // a point has no inner sphere
        in[t].spheres[1] = 0.14433756f;   // the unit tet's, checked below
    }

    PxgSlangComputeDevice* dev = testDevice();
    PxgBufferHandle bVA = createBufferFrom(dev, VA);
    PxgBufferHandle bVB = createBufferFrom(dev, VB);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    std::vector<TcOut> outInit(N);
    std::memset(outInit.data(), 0xEE, outInit.size() * sizeof(TcOut));
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct TcParams { uint32_t numTests, p0, p1, p2; } prm = { N, 0, 0, 0 };
    Kernel k("npTetCollide.spv", "ctTetCollide",
             { 16, 16, (uint32_t)sizeof(TcIn), (uint32_t)sizeof(TcOut) }, sizeof(TcParams));
    k.dispatch(1u, { bVA, bVB, bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<TcOut>(dev, bOut, N);
    for (uint32_t t = 0; t < N; ++t)
        std::printf("  case %u: result=%u degenerate=%u lower=%.6f upper=%.6f dir=(%.4f %.4f %.4f) "
                    "innerSphere=%.7f center=(%.4f %.4f %.4f)\n",
                    t, g[t].result, g[t].degenerate, g[t].lowerBound, g[t].upperBound,
                    g[t].direction[0], g[t].direction[1], g[t].direction[2],
                    g[t].innerSphere, g[t].tetCenter[0], g[t].tetCenter[1], g[t].tetCenter[2]);

    // computeInnerSphere: the CENTROID (0.25, 0.25, 0.25) and the smallest distance from it
    // to the four faces.  Three coordinate planes are 0.25 away; the slant plane
    // x + y + z = 1 is (1 - 0.75)/sqrt(3) = 0.1443376 away, so THAT is the answer.  An
    // incentre formula -- which is what "inner sphere" suggests and what a re-derivation
    // produces -- gives a different number, and this one feeds EPA's epsilon.
    for (uint32_t t = 0; t < N; ++t)
    {
        PXG_CHECK_MSG(std::fabs(g[t].innerSphere - 0.14433756f) < 1e-6f,
                      "case %u: the unit tet's inner sphere is the centroid's distance to the "
                      "SLANT face, 0.1443376, got %.7f", t, g[t].innerSphere);
        PXG_CHECK(std::fabs(g[t].tetCenter[0] - 0.25f) < 1e-6f &&
                  std::fabs(g[t].tetCenter[1] - 0.25f) < 1e-6f &&
                  std::fabs(g[t].tetCenter[2] - 0.25f) < 1e-6f);
    }

    // (0) far outside: GJK alone answers DISTANT and the driver returns immediately.  EPA
    // must not have run -- if it had, it would have written a direction and witness points.
    PXG_CHECK_MSG(g[0].result == 0u /*GJK_DISTANT*/,
                  "a point 17 units from the tet with a 0.5 contact distance is DISTANT "
                  "(got result %u)", g[0].result);
    PXG_CHECK_MSG(g[0].upperBound > 1.0e30f,
                  "and EPA must NOT have run: the upper bound is still GJK's FLT_MAX initial "
                  "value (got %g).  A driver that fell through to EPA on DISTANT would have "
                  "overwritten it", g[0].upperBound);

    // (1) deep inside: GJK cannot answer, so the driver MUST enter EPA, and EPA reports the
    // penetration depth.
    //
    // NOTE THE RESULT CODE.  EPA returns eCLOSE, not eOVERLAP, when it measured a real
    // penetration -- `result = (lowerBound < 0) ? eCLOSE : eOVERLAP` (epa.cuh).  The caller
    // treats BOTH as intersecting (`intersect = result == eOVERLAP || result == eCLOSE`,
    // softbodyPrimitives.cu:561), so the code alone does not say whether EPA ran.  What says
    // it is the NEGATIVE lower bound: GJK's lowerBound is a separation and is never negative,
    // so only EPA can produce one.
    //
    // The value is checkable.  (0.2,0.2,0.2) inside the unit tet is 0.2/1 from each
    // coordinate plane and (1 - 0.6)/sqrt(3) = 0.2309401 from the slant face, so the
    // shallowest exit is through the SLANT face and the depth is that number.
    PXG_CHECK_MSG(std::fabs(g[1].lowerBound + 0.2309401f) < 1e-5f,
                  "EPA must report the penetration depth to the nearest face, -0.2309401 "
                  "(the slant face), got %.7f -- a positive value means EPA never ran",
                  g[1].lowerBound);
    PXG_CHECK_MSG(g[1].upperBound == 0.0f,
                  "and the overlap's upper bound is 0 (got %.7f)", g[1].upperBound);
    {
        const float dl = std::sqrt(g[1].direction[0] * g[1].direction[0] +
                                   g[1].direction[1] * g[1].direction[1] +
                                   g[1].direction[2] * g[1].direction[2]);
        PXG_CHECK_MSG(std::fabs(dl - 1.0f) < 1e-3f,
                      "EPA's penetration direction must be a unit vector (|d| = %.6f) -- a "
                      "driver that returned GJK's direction here would hand back whatever the "
                      "overlap simplex left", dl);
        // ...and it points along the slant face's normal, (-1,-1,-1)/sqrt(3).
        PXG_CHECK_MSG(std::fabs(g[1].direction[0] + 0.5773503f) < 1e-4f &&
                      std::fabs(g[1].direction[1] + 0.5773503f) < 1e-4f &&
                      std::fabs(g[1].direction[2] + 0.5773503f) < 1e-4f,
                      "and it is the slant face's normal (got %.5f %.5f %.5f)",
                      g[1].direction[0], g[1].direction[1], g[1].direction[2]);
    }

    // (2) just outside, within the contact distance: GJK answers CLOSE with no anomaly, so
    // the driver returns WITHOUT entering EPA and the witness points are GJK's.  Those
    // satisfy closestPointB = closestPointA + upperBound * direction, which EPA's do not.
    PXG_CHECK_MSG(g[2].result == 1u /*GJK_CLOSE*/,
                  "a point 0.05 outside the x face with a 0.5 contact distance is CLOSE "
                  "(got %u)", g[2].result);
    {
        float mx = 0.0f;
        for (int c = 0; c < 3; ++c)
            mx = std::fmax(mx, std::fabs(g[2].closestPointB[c] -
                                         (g[2].closestPointA[c] + g[2].upperBound * g[2].direction[c])));
        PXG_CHECK_MSG(mx < 1e-5f,
                      "a CLOSE result's witness points come from GJK, so B = A + upperBound * dir "
                      "(max deviation %.8f).  Entering EPA on a clean CLOSE would break this",
                      mx);
        PXG_CHECK_MSG(std::fabs(g[2].upperBound - 0.05f) < 1e-4f,
                      "and the separation is the 0.05 gap to the x face (got %.6f)", g[2].upperBound);
    }

    for (PxgBufferHandle h : { bVA, bVB, bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
