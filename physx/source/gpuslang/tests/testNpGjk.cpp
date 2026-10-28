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
    // penetration depth.  (0.2,0.2,0.2) inside the unit tet is 0.2 from each of the three
    // coordinate planes and 0.2309401 from the slant face, so the shallowest exit -- which is
    // what EPA finds -- is 0.2, through one of the three tied coordinate planes.  Which one it
    // picks is not specified, so the direction is only checked for being a unit axis.
    PXG_CHECK_MSG(g[1].result == 2u /*GJK_OVERLAP*/,
                  "a point inside the tet comes back OVERLAP: EPA's lowerBound is a positive "
                  "penetration depth, and `result = (lowerBound < 0) ? eCLOSE : eOVERLAP` "
                  "(epa.cuh) therefore says OVERLAP (got %u)", g[1].result);
    PXG_CHECK_MSG(std::fabs(g[1].lowerBound - 0.2f) < 1e-5f &&
                  std::fabs(g[1].upperBound - 0.2f) < 1e-5f,
                  "and the depth is the shallowest exit, 0.2 (got lower %.7f upper %.7f)",
                  g[1].lowerBound, g[1].upperBound);
    {
        const float dl = std::sqrt(g[1].direction[0] * g[1].direction[0] +
                                   g[1].direction[1] * g[1].direction[1] +
                                   g[1].direction[2] * g[1].direction[2]);
        PXG_CHECK_MSG(std::fabs(dl - 1.0f) < 1e-3f,
                      "EPA's penetration direction must be a unit vector (|d| = %.6f)", dl);
        float mx = 0.0f;
        for (int c = 0; c < 3; ++c) mx = std::fmax(mx, std::fabs(g[1].direction[c]));
        PXG_CHECK_MSG(std::fabs(mx - 1.0f) < 1e-4f,
                      "and it is one of the three tied coordinate-plane normals, i.e. an AXIS "
                      "(got %.5f %.5f %.5f)", g[1].direction[0], g[1].direction[1], g[1].direction[2]);
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

// ===========================================================================
// P9.89 -- computeInnerSphere's running minimum is the CUDA's SELECT chain.
//
// softbodyPrimitives.cu:140/:147/:154 fold the four face distances with
// `innerSphere = PxMin(PxAbs(d), innerSphere)`, and PxMin(float, float) is
// selectMin(a, b) = a < b ? a : b (PxMath.h:98, PxUnixMathIntrinsics.h:116).
// That is not fmin: a NaN running value b FAILS the compare, so it is kept, and
// the chain stays NaN.  The port used to spell it `min(...)`, i.e. GLSL.std.450
// FMin, and llvmpipe's FMin returns the non-NaN operand, so the chain recovered
// at the first finite face.  Ties and signed zeros cannot tell the two apart
// (abs() never yields -0, and equal values have equal bits); only a NaN can.
//
// The running value is NaN exactly when the FIRST face's |dot| is.  With finite
// vertices that needs the face normal's cross product to OVERFLOW in one
// component only: getNormalized then sees m = inf, scales by 1/sqrt(inf) = 0,
// and inf * 0 = NaN.  The "first-face overflow" family below builds exactly
// that: the edges v2 - v0 and v1 - v0 are axis-aligned and ~2e19 long, so
// their cross product's x overflows, while v3 stays close to v0, so the faces
// (0,1,3) and (0,3,2) keep finite normals.  The CUDA returns NaN there; the
// old port returned the smaller of those two finite distances.  Face (1,2,3)
// is NaN in that family as well, so a "far apex" family (v0 ~2e38 out, the
// other three near the origin) overflows the first face alone and leaves face
// (1,2,3) finite: each of the three fold sites is then observable on its own.
//
// Two control families pin everything else bit for bit: ordinary tetrahedra
// with non-round coordinates (the PxVec3 dots are (x + y) + z, which the leaf
// now gets from the NoContraction stamp exactly like every engine importer),
// and a "later-face overflow" family whose NaN appears on face (1,2,3), where
// both spellings ignore it.
//
// The kernel is ctInnerSphere (npTetCollide.slang), computeInnerSphere alone.
// The mirror is the CUDA transcribed; the twin is the old FMin chain and must
// differ on every first-face-overflow tetrahedron.
namespace p89inner
{
static V3 addV(V3 a, V3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 mulV(V3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
// PxMin(float, float) = selectMin (PxUnixMathIntrinsics.h:116).
static float selectMin(float a, float b) { return a < b ? a : b; }
// llvmpipe's GLSL.std.450 FMin: the non-NaN operand wins.
static float fminOld(float a, float b) { return std::isnan(a) ? b : std::isnan(b) ? a : (a < b ? a : b); }

// softbodyPrimitives.cu:123-160, with the fold as a parameter.
template <typename Fold>
static float innerSphere(V3 v0, V3 v1, V3 v2, V3 v3, V3& center, Fold fold)
{
    center = mulV(addV(addV(addV(v0, v1), v2), v3), 0.25f);
    V3 n = getNormalized(crossX(sub3(v2, v0), sub3(v1, v0)));          // 0 2 1
    float inner = std::fabs(dot3(sub3(center, v0), n));
    n = getNormalized(crossX(sub3(v2, v1), sub3(v3, v1)));               // 1 2 3
    inner = fold(std::fabs(dot3(sub3(center, v1), n)), inner);
    n = getNormalized(crossX(sub3(v1, v0), sub3(v3, v0)));               // 0 1 3
    inner = fold(std::fabs(dot3(sub3(center, v0), n)), inner);
    n = getNormalized(crossX(sub3(v3, v0), sub3(v2, v0)));               // 0 3 2
    inner = fold(std::fabs(dot3(sub3(center, v0), n)), inner);
    return inner;
}
} // namespace p89inner

PXG_TEST(npTetCollide_p89InnerSphereSelect)
{
    using namespace p89inner;
    uint64_t s = 0x7e7ac011ull;
    auto u = [&]() { s = s * 6364136223846793005ull + 1442695040888963407ull; return float(uint32_t(s >> 40)) / 16777216.0f; };
    auto sym = [&](float r) { return (u() * 2.0f - 1.0f) * r; };

    enum { ORDINARY = 0, FIRST_FACE = 1, FAR_APEX = 2, LATER_FACE = 3 };
    std::vector<float> VB;
    std::vector<int> kind;
    auto push = [&](const V3 (&v)[4], int k)
    {
        for (int i = 0; i < 4; ++i) { VB.push_back(v[i].x); VB.push_back(v[i].y); VB.push_back(v[i].z); VB.push_back(0.0f); }
        kind.push_back(k);
    };

    // Ordinary tetrahedra: non-round, of every aspect from regular to sliver.
    for (int t = 0; t < 448; ++t)
    {
        const float sc = 0.05f + 3.0f * u();
        const V3 o = { sym(4.0f), sym(4.0f), sym(4.0f) };
        V3 v[4];
        for (int i = 0; i < 4; ++i) v[i] = { o.x + sym(sc), o.y + sym(sc), o.z + sym(sc) };
        if (t % 7 == 3) v[3] = { (v[0].x + v[1].x + v[2].x) * 0.3333f + sym(0.01f * sc),
                                 (v[0].y + v[1].y + v[2].y) * 0.3333f + sym(0.01f * sc),
                                 (v[0].z + v[1].z + v[2].z) * 0.3333f + sym(0.01f * sc) };   // sliver
        push(v, ORDINARY);
    }
    // First-face overflow: (v2 - v0) x (v1 - v0) overflows in x ONLY.
    for (int t = 0; t < 96; ++t)
    {
        const V3 v0 = { sym(3.0f), sym(3.0f), sym(3.0f) };
        const float L1 = 1.9e19f + 1.0e19f * u(), L2 = 1.9e19f + 1.0e19f * u();
        const float sgn1 = (t & 1) ? -1.0f : 1.0f, sgn2 = (t & 2) ? -1.0f : 1.0f;
        V3 v[4];
        v[0] = v0;
        v[1] = { v0.x + sym(2.0f), v0.y + sym(2.0f), v0.z + sgn1 * L1 };
        v[2] = { v0.x + sym(2.0f), v0.y + sgn2 * L2, v0.z + sym(2.0f) };
        v[3] = { v0.x + sym(0.4f), v0.y + sym(0.4f), v0.z + sym(0.4f) };
        push(v, FIRST_FACE);
    }
    // Far apex: v0 sits ~2e38 out along -x and v1, v2, v3 near the origin, so the first
    // face overflows in y alone (|v2.z - v0.z| >= 1.75) while face (1,2,3) is small and
    // FINITE.  The first family's face (1,2,3) is NaN too, so without this one a revert
    // of that site alone would still keep the NaN.
    for (int t = 0; t < 48; ++t)
    {
        const float X = -(2.0e38f + 0.4e38f * u());
        const float y0 = sym(0.3f), z0 = sym(0.3f);
        V3 v[4];
        v[0] = { X, y0, z0 };
        v[1] = { sym(1.0f), y0 + sym(1.2f), z0 + sym(1.2f) };
        v[2] = { sym(1.0f), y0 + sym(1.2f), z0 + ((t & 1) ? -1.0f : 1.0f) * (1.75f + 0.75f * u()) };
        v[3] = { sym(1.0f), sym(1.2f), sym(1.2f) };
        push(v, FAR_APEX);
    }
    // Later-face overflow: the first face stays finite and a LATER face's distance is
    // NaN, which both folds must ignore.  Even t: faces (1,2,3) and (0,1,3) overflow;
    // odd t: faces (1,2,3) and (0,3,2).  Together they put a NaN new distance against
    // a finite running value at each of the three fold sites.
    for (int t = 0; t < 48; ++t)
    {
        const V3 v0 = { sym(3.0f), sym(3.0f), sym(3.0f) };
        const float L1 = 1.9e19f + 1.0e19f * u(), L2 = 1.9e19f + 1.0e19f * u();
        V3 v[4];
        v[0] = v0;
        if ((t & 1) == 0)
        {
            v[1] = { v0.x + sym(2.0f), v0.y + sym(2.0f), v0.z + L1 };
            v[2] = { v0.x + sym(0.4f), v0.y + sym(0.4f), v0.z + sym(0.4f) };
            v[3] = { v0.x + sym(2.0f), v0.y + L2, v0.z + sym(2.0f) };
        }
        else
        {
            v[1] = { v0.x + sym(0.3f), v0.y + sym(0.3f), v0.z + sym(0.3f) };
            v[2] = { v0.x + sym(2.0f), v0.y + L2, v0.z + sym(2.0f) };
            v[3] = { v0.x + sym(2.0f), v0.y + sym(2.0f), v0.z + L1 };
        }
        push(v, LATER_FACE);
    }

    const uint32_t N = uint32_t(kind.size());
    const uint32_t laterStart = uint32_t(std::find(kind.begin(), kind.end(), int(LATER_FACE)) - kind.begin());
    std::vector<float> ref(N), twin(N);
    std::vector<V3> refC(N);
    uint32_t nNaN = 0, nTwinDiff = 0, nFirstNaN = 0, nApexNaN = 0, nLaterNaNFace = 0;
    for (uint32_t t = 0; t < N; ++t)
    {
        const V3 v0 = { VB[t * 16 + 0], VB[t * 16 + 1], VB[t * 16 + 2] };
        const V3 v1 = { VB[t * 16 + 4], VB[t * 16 + 5], VB[t * 16 + 6] };
        const V3 v2 = { VB[t * 16 + 8], VB[t * 16 + 9], VB[t * 16 + 10] };
        const V3 v3 = { VB[t * 16 + 12], VB[t * 16 + 13], VB[t * 16 + 14] };
        V3 c2;
        ref[t] = innerSphere(v0, v1, v2, v3, refC[t], selectMin);
        twin[t] = innerSphere(v0, v1, v2, v3, c2, fminOld);
        if (std::isnan(ref[t])) ++nNaN;
        if (asU32(ref[t]) != asU32(twin[t])) ++nTwinDiff;
        if (kind[t] == FIRST_FACE && std::isnan(ref[t]) && std::isfinite(twin[t])) ++nFirstNaN;
        if (kind[t] == FAR_APEX && std::isnan(ref[t]) && std::isfinite(twin[t]))
        {
            // ... and face (1,2,3) really is finite there
            const V3 n = getNormalized(crossX(sub3(v2, v1), sub3(v3, v1)));
            if (std::isfinite(dot3(sub3(refC[t], v1), n))) ++nApexNaN;
        }
        if (kind[t] == LATER_FACE)
        {
            // the NaNs really are on the planned later faces, and both folds ignore them
            const bool nan123 = std::isnan(dot3(sub3(refC[t], v1), getNormalized(crossX(sub3(v2, v1), sub3(v3, v1)))));
            const bool nan013 = std::isnan(dot3(sub3(refC[t], v0), getNormalized(crossX(sub3(v1, v0), sub3(v3, v0)))));
            const bool nan032 = std::isnan(dot3(sub3(refC[t], v0), getNormalized(crossX(sub3(v3, v0), sub3(v2, v0)))));
            const bool planned = ((t - laterStart) & 1u) == 0u ? (nan123 && nan013 && !nan032) : (nan123 && !nan013 && nan032);
            if (planned && std::isfinite(ref[t]) && asU32(ref[t]) == asU32(twin[t]))
                ++nLaterNaNFace;
        }
    }
    const uint32_t nFirst = 96u, nApex = 48u, nLater = 48u;
    PXG_CHECK_MSG(nFirstNaN == nFirst,
                  "p89 innerSphere: only %u of %u first-face-overflow tetrahedra give the CUDA NaN "
                  "against a finite FMin chain -- the family does not plant the case", nFirstNaN, nFirst);
    PXG_CHECK_MSG(nApexNaN == nApex,
                  "p89 innerSphere: only %u of %u far-apex tetrahedra give the CUDA NaN against a finite "
                  "FMin chain with a finite face (1,2,3)", nApexNaN, nApex);
    PXG_CHECK_MSG(nTwinDiff == nFirst + nApex && nNaN == nFirst + nApex,
                  "p89 innerSphere: the old FMin chain differs from the CUDA select chain on %u tetrahedra "
                  "and the CUDA gives NaN on %u; both should be exactly the %u first-face-NaN ones",
                  nTwinDiff, nNaN, nFirst + nApex);
    PXG_CHECK_MSG(nLaterNaNFace == nLater,
                  "p89 innerSphere: %u of %u later-face tetrahedra carry a NaN face that both folds ignore",
                  nLaterNaNFace, nLater);

    PxgSlangComputeDevice* dev = testDevice();
    PxgBufferHandle bVA = createBufferFrom(dev, std::vector<float>(16, 0.0f));
    PxgBufferHandle bVB = createBufferFrom(dev, VB);
    PxgBufferHandle bIn = createBufferFrom(dev, std::vector<TcIn>(1));
    std::vector<TcOut> outInit(N);
    std::memset(outInit.data(), 0xEE, outInit.size() * sizeof(TcOut));
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);
    struct TcParams { uint32_t numTests, p0, p1, p2; } prm = { N, 0, 0, 0 };
    Kernel k("npTetCollide.spv", "ctInnerSphere",
             { 16, 16, (uint32_t)sizeof(TcIn), (uint32_t)sizeof(TcOut) }, sizeof(TcParams));
    k.dispatch((N + 63u) / 64u, { bVA, bVB, bIn, bOut }, prm);
    dev->flush();
    const auto g = downloadBuffer<TcOut>(dev, bOut, N);

    uint32_t bad[4] = { 0, 0, 0, 0 }, asTwin = 0, badC = 0;
    for (uint32_t t = 0; t < N; ++t)
    {
        const bool cOk = asU32(g[t].tetCenter[0]) == asU32(refC[t].x) && asU32(g[t].tetCenter[1]) == asU32(refC[t].y) &&
                         asU32(g[t].tetCenter[2]) == asU32(refC[t].z);
        if (!cOk) ++badC;
        if (asU32(g[t].innerSphere) != asU32(ref[t]))
        {
            if (bad[0] + bad[1] + bad[2] + bad[3] < 4)
                std::printf("    p89 innerSphere tet %u (kind %d): gpu %08x (%g) cuda %08x (%g) old-FMin twin %08x (%g)\n",
                            t, kind[t], asU32(g[t].innerSphere), g[t].innerSphere, asU32(ref[t]), ref[t],
                            asU32(twin[t]), twin[t]);
            ++bad[kind[t]];
            if (asU32(g[t].innerSphere) == asU32(twin[t])) ++asTwin;
        }
    }
    std::printf("    [p89 innerSphere] %u tetrahedra, %u CUDA NaN (first face overflows), twin differs on %u\n",
                N, nNaN, nTwinDiff);
    PXG_CHECK_MSG(badC == 0u, "p89 innerSphere: %u tetCenters differ from ((v0 + v1) + v2 + v3) * 0.25", badC);
    PXG_CHECK_MSG(bad[0] + bad[1] + bad[2] + bad[3] == 0u,
                  "p89 innerSphere: %u ordinary, %u first-face-overflow, %u far-apex and %u later-face-overflow "
                  "tetrahedra differ from the CUDA selectMin chain (%u of them equal the old FMin chain)",
                  bad[0], bad[1], bad[2], bad[3], asTwin);

    for (PxgBufferHandle h : { bVA, bVB, bIn, bOut }) dev->destroyBuffer(h);
}

// ===========================================================================
// npTetCollide_warpDifferential -- the WARP-COOPERATIVE driver (pxgSlangGjkWarp, the
// CUDA's shape: 32 threads, lane L holding CUDA lane L's registers, shuffles as
// groupshared exchanges) against the SERIAL one (pxgSlangTetCollide over pxgSlangGjk /
// pxgSlangEpa, one thread simulating the warp), bit for bit on every output field.
//
// The serial driver is the reference: its GJK and EPA are pinned against CPU
// transcriptions (npGjk_driver, npEpa_*), and sb_primitiveContactGenLaunch ran on it until
// the cooperative rework.  So one fixture certifies the new driver against everything the
// old one was certified against, without a third GJK.
//
// Families, all through the driver's whole policy (GJK, the anomaly gate, EPA from a
// tetrahedron or from a 1-3 vertex simplex, the degenerate-EPA retry from +y):
//   point / segment / hull (4..64 vertices) vs a tetrahedron, integer and non-integer
//   coordinates, separated, close, touching and overlapping; FLAT hulls (one coordinate
//   constant), whose overlap leaves GJK a triangle so EPA seeds from epaStart123;
//   FLAT or collapsed tetrahedra; and general clouds of up to 64 vertices on BOTH sides
//   (two 32-vertex chunks per support).  The warp entry reports which branches ran in
//   `pad`; the fixture requires every one of them to be reached.
// ===========================================================================
namespace warpdiff
{
// GW_PATH_* (pxgSlangGjkWarp), reported in TcOut::pad by ctTetCollideWarp
static const uint32_t PATH_EPA = 1u, PATH_ANOMALY = 2u, PATH_RETRY = 4u, PATH_START4 = 8u;
}

PXG_TEST(npTetCollide_warpDifferential)
{
    using namespace warpdiff;
    const uint32_t MAXV = 64u;
    std::mt19937 rng(20261028u);
    std::vector<float> VA, VB;
    std::vector<TcIn> in;

    auto uni = [&](float lo, float hi) { return lo + (hi - lo) * float(rng() % 1000001u) / 1000000.0f; };
    auto ivr = [&](int r) { return float(int(rng() % uint32_t(2 * r + 1)) - r); };
    // one test: A has nA points, B has nB points; `flatA` zeroes A's z, `flatB` B's
    auto emit = [&](uint32_t nA, uint32_t nB, bool integer, float spread, V3 shiftB, bool flatA, bool flatB,
                    float cDist)
    {
        V3 a[64], b[64];
        for (uint32_t k = 0; k < MAXV; ++k)
        {
            a[k] = integer ? V3{ ivr(int(spread)), ivr(int(spread)), ivr(int(spread)) }
                           : V3{ uni(-spread, spread), uni(-spread, spread), uni(-spread, spread) };
            b[k] = integer ? V3{ ivr(int(spread)), ivr(int(spread)), ivr(int(spread)) }
                           : V3{ uni(-spread, spread), uni(-spread, spread), uni(-spread, spread) };
            if (flatA) a[k].z = 0.0f;
            if (flatB) b[k].z = 0.0f;
            b[k] = { b[k].x + shiftB.x, b[k].y + shiftB.y, b[k].z + shiftB.z };
            if (k >= nA) a[k] = { 0, 0, 0 };
            if (k >= nB) b[k] = { 0, 0, 0 };
        }
        for (uint32_t k = 0; k < MAXV; ++k)
        {
            VA.push_back(a[k].x); VA.push_back(a[k].y); VA.push_back(a[k].z); VA.push_back(0);
            VB.push_back(b[k].x); VB.push_back(b[k].y); VB.push_back(b[k].z); VB.push_back(0);
        }
        // the kernel's search direction: rigid centre minus tet centroid
        V3 ca = { 0, 0, 0 }, cb = { 0, 0, 0 };
        for (uint32_t k = 0; k < nA; ++k) { ca.x += a[k].x; ca.y += a[k].y; ca.z += a[k].z; }
        for (uint32_t k = 0; k < nB; ++k) { cb.x += b[k].x; cb.y += b[k].y; cb.z += b[k].z; }
        TcIn e;
        std::memset(&e, 0, sizeof(e));
        e.searchDir[0] = ca.x / float(nA) - cb.x / float(nB);
        e.searchDir[1] = ca.y / float(nA) - cb.y / float(nB);
        e.searchDir[2] = ca.z / float(nA) - cb.z / float(nB);
        if (rng() % 16u == 0u) { e.searchDir[0] = 0; e.searchDir[1] = 0; e.searchDir[2] = 0; }  // the +x fallback
        e.searchDir[3] = cDist;
        e.spheres[0] = (rng() % 4u == 0u) ? 0.0f : uni(0.0f, 0.5f);
        e.spheres[1] = uni(0.01f, 0.5f);
        e.counts[0] = nA; e.counts[1] = nB;
        in.push_back(e);
    };
    auto shift = [&](float lo, float hi) { return V3{ uni(lo, hi) * ((rng() & 1u) ? 1.0f : -1.0f), uni(-hi, hi), uni(-hi, hi) }; };

    for (uint32_t it = 0; it < 300; ++it) emit(1, 4, it & 1, 3.0f, shift(0.0f, 3.0f), false, false, uni(0.0f, 1.0f));     // sphere
    for (uint32_t it = 0; it < 300; ++it) emit(2, 4, it & 1, 3.0f, shift(0.0f, 3.0f), false, false, uni(0.0f, 1.0f));     // capsule
    for (uint32_t it = 0; it < 600; ++it) emit(4 + rng() % 61u, 4, it & 1, 4.0f, shift(0.0f, 6.0f), false, false, uni(0.0f, 1.5f));
    for (uint32_t it = 0; it < 300; ++it) emit(8, 4, true, 2.0f, shift(0.0f, 1.0f), false, false, uni(0.0f, 0.5f));   // deep overlap
    for (uint32_t it = 0; it < 300; ++it) emit(3 + rng() % 30u, 4, it & 1, 3.0f, shift(0.0f, 2.0f), true, false, uni(0.0f, 1.0f));  // flat hull
    for (uint32_t it = 0; it < 200; ++it) emit(4 + rng() % 20u, 4, it & 1, 3.0f, shift(0.0f, 2.0f), false, true, uni(0.0f, 1.0f));  // flat tet
    for (uint32_t it = 0; it < 200; ++it) emit(1 + rng() % 64u, 1 + rng() % 64u, it & 1, 5.0f, shift(0.0f, 8.0f), false, false, uni(0.0f, 2.0f));
    for (uint32_t it = 0; it < 200; ++it) emit(33 + rng() % 32u, 33 + rng() % 32u, true, 4.0f, shift(0.0f, 2.0f), false, false, uni(0.0f, 1.0f));
    // both clouds in the plane z = 0: the Minkowski difference is planar, so an overlap
    // ends GJK on a triangle or a segment and EPA seeds from epaStart123
    for (uint32_t it = 0; it < 300; ++it) emit(3 + rng() % 20u, 3 + rng() % 20u, true, 3.0f, V3{ ivr(1), ivr(1), 0.0f }, true, true, uni(0.0f, 1.0f));
    // touching: A at z <= 0, B at z >= 0 sharing the plane z = 0 (integer, so exact)
    for (uint32_t it = 0; it < 200; ++it)
    {
        emit(4 + rng() % 12u, 4, true, 2.0f, V3{ ivr(1), ivr(1), 0.0f }, false, false, uni(0.0f, 0.5f));
        const size_t vb = (in.size() - 1) * MAXV * 4;
        for (uint32_t k = 0; k < in.back().counts[0]; ++k) VA[vb + k * 4 + 2] = -std::fabs(VA[vb + k * 4 + 2]);
        for (uint32_t k = 0; k < in.back().counts[1]; ++k) VB[vb + k * 4 + 2] = std::fabs(VB[vb + k * 4 + 2]);
    }
    // a hull a few ULP across against a point near it: GJK stops early on a degenerate
    // simplex about once in a hundred, which is the only way the anomaly gate fires
    // (testPsConvexContactGen's P9.89 rig shrinks its hull the same way)
    for (uint32_t it = 0; it < 1500; ++it)
    {
        const float kScale = 1.3e-6f;
        const V3 c = { 0.37f, -0.21f, 0.52f };
        TcIn e;
        std::memset(&e, 0, sizeof(e));
        const uint32_t nB = 12u;
        V3 b[12];
        for (uint32_t k = 0; k < nB; ++k)
        {
            V3 d = { uni(-1.0f, 1.0f), uni(-1.0f, 1.0f), uni(-1.0f, 1.0f) };
            const float m = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z) + 1e-3f;
            const float r = (0.55f + 0.25f * uni(0.0f, 1.0f)) * kScale;
            b[k] = { c.x + d.x / m * r * 1.3f, c.y + d.y / m * r * 0.9f, c.z + d.z / m * r };
        }
        V3 d = { uni(-1.0f, 1.0f), uni(-1.0f, 1.0f), uni(-1.0f, 1.0f) };
        const float m = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z) + 1e-3f;
        const float r = kScale * (0.3f + 2.0f * uni(0.0f, 1.0f));
        const V3 pt = { c.x + d.x / m * r, c.y + d.y / m * r, c.z + d.z / m * r };
        for (uint32_t k = 0; k < MAXV; ++k)
        {
            const V3 a = (k == 0) ? pt : V3{ 0, 0, 0 };
            const V3 bb = (k < nB) ? b[k] : V3{ 0, 0, 0 };
            VA.push_back(a.x); VA.push_back(a.y); VA.push_back(a.z); VA.push_back(0);
            VB.push_back(bb.x); VB.push_back(bb.y); VB.push_back(bb.z); VB.push_back(0);
        }
        e.searchDir[0] = pt.x - c.x; e.searchDir[1] = pt.y - c.y; e.searchDir[2] = pt.z - c.z;
        e.searchDir[3] = 0.01f;
        e.spheres[0] = 0.0f;
        e.spheres[1] = 0.4f * kScale;
        e.counts[0] = 1u; e.counts[1] = nB;
        in.push_back(e);
    }
    // collapsed tets: two coincident corners (integer, so they really coincide)
    for (uint32_t it = 0; it < 200; ++it)
    {
        emit(4 + rng() % 12u, 4, true, 2.0f, shift(0.0f, 1.5f), false, false, uni(0.0f, 0.5f));
        const size_t vb = (in.size() - 1) * MAXV * 4;
        for (int c = 0; c < 3; ++c) VB[vb + 4 + c] = VB[vb + c];
    }

    const uint32_t N = (uint32_t)in.size();
    PxgSlangComputeDevice* dev = testDevice();
    PxgBufferHandle bVA = createBufferFrom(dev, VA);
    PxgBufferHandle bVB = createBufferFrom(dev, VB);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    std::vector<TcOut> outInit(N);
    std::memset(outInit.data(), 0xEE, outInit.size() * sizeof(TcOut));
    PxgBufferHandle bOutS = createBufferFrom(dev, outInit);
    PxgBufferHandle bOutW = createBufferFrom(dev, outInit);

    struct TcParams { uint32_t numTests, p0, p1, p2; } prm = { N, 0, 0, 0 };
    {
        Kernel k("npTetCollide.spv", "ctTetCollide",
                 { 16, 16, (uint32_t)sizeof(TcIn), (uint32_t)sizeof(TcOut) }, sizeof(TcParams));
        k.dispatch((N + 63u) / 64u, { bVA, bVB, bIn, bOutS }, prm);
        dev->flush();
    }
    {
        Kernel k("npTetCollide.spv", "ctTetCollideWarp",
                 { 16, 16, (uint32_t)sizeof(TcIn), (uint32_t)sizeof(TcOut) }, sizeof(TcParams));
        k.dispatch(N, { bVA, bVB, bIn, bOutW }, prm);
        dev->flush();
    }
    const auto gS = downloadBuffer<TcOut>(dev, bOutS, N);
    const auto gW = downloadBuffer<TcOut>(dev, bOutW, N);

    size_t mm = 0, nRes[3] = { 0, 0, 0 }, nEpa = 0, nStart4 = 0, nStart123 = 0, nAnomaly = 0, nRetry = 0,
           nDegen = 0;
    for (uint32_t t = 0; t < N; ++t)
    {
        const TcOut& s = gS[t];
        const TcOut& w = gW[t];
        TcOut sc = s, wc = w;
        sc.pad = 0; wc.pad = 0;
        if (std::memcmp(&sc, &wc, sizeof(TcOut)) != 0)
        {
            if (mm < 8)
                std::printf("  test %u (nA %u nB %u path %x): serial r%u d%u s%u lb %08x ub %08x dir %08x %08x %08x A %08x %08x %08x"
                            " | warp r%u d%u s%u lb %08x ub %08x dir %08x %08x %08x A %08x %08x %08x\n",
                            t, in[t].counts[0], in[t].counts[1], w.pad, s.result, s.degenerate, s.size,
                            asU32(s.lowerBound), asU32(s.upperBound), asU32(s.direction[0]), asU32(s.direction[1]),
                            asU32(s.direction[2]), asU32(s.closestPointA[0]), asU32(s.closestPointA[1]),
                            asU32(s.closestPointA[2]), w.result, w.degenerate, w.size, asU32(w.lowerBound),
                            asU32(w.upperBound), asU32(w.direction[0]), asU32(w.direction[1]), asU32(w.direction[2]),
                            asU32(w.closestPointA[0]), asU32(w.closestPointA[1]), asU32(w.closestPointA[2]));
            ++mm;
        }
        if (s.result < 3) ++nRes[s.result];
        if (w.pad & PATH_EPA) { ++nEpa; if (w.pad & PATH_START4) ++nStart4; else ++nStart123; }
        if (w.pad & PATH_ANOMALY) ++nAnomaly;
        if (w.pad & PATH_RETRY) ++nRetry;
        if (s.degenerate) ++nDegen;
    }
    std::printf("  %u queries: distant %zu close %zu overlap %zu; EPA %zu (from a tet %zu, from 1-3 vertices %zu), "
                "anomaly %zu, degenerate-EPA retry %zu, degenerate out %zu; %zu mismatches\n",
                N, nRes[0], nRes[1], nRes[2], nEpa, nStart4, nStart123, nAnomaly, nRetry, nDegen, mm);
    PXG_CHECK_MSG(mm == 0, "warp-cooperative tetPrimitivesCollide2: %zu of %u queries differ from the serial driver",
                  mm, N);
    PXG_CHECK_MSG(nRes[0] > 100 && nRes[1] > 100 && nRes[2] > 100 && nStart4 > 50 && nStart123 > 20 &&
                  nAnomaly > 0 && nRetry > 0,
                  "warp differential coverage thin (distant %zu close %zu overlap %zu start4 %zu start123 %zu "
                  "anomaly %zu retry %zu)", nRes[0], nRes[1], nRes[2], nStart4, nStart123, nAnomaly, nRetry);

    for (PxgBufferHandle h : { bVA, bVB, bIn, bOutS, bOutW }) dev->destroyBuffer(h);
}

} // namespace
