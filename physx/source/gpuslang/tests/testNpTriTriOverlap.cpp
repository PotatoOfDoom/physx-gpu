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

// Verification of npTriTriOverlap.slang: triangleTriangleOverlaps
// (convexMeshMidphase.cu L1386-1528).  The prep stage (root BV32 bounds
// reduced over the masked lanes, the 8 AABB corners with the set-bit-selects-
// MIN pattern, vertex2Shape scaling, the aToB move of the A side, the
// initialDir fallback) is transcribed exactly; the GJK over the two 8-point
// clouds reuses the tranche-129 cold-start reference verbatim (copied from
// testNpGjk.cpp).  overlaps = (ret != GJK_DISTANT).  The GJK-internal
// comparisons are bit-reproducible by construction (established in tranche
// 129); the prep floats are compared to tight tolerance and near-DISTANT
// boundary cases are margin-skipped symmetrically on lowerBound vs minSep.

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


struct F4 { float x, y, z, w; };
struct TtoIn
{
    F4 t0q, t0p, t1q, t1p;
    F4 scaleS0, scaleR0, scaleS1, scaleR1;
    uint32_t counts[4];
    F4 misc;
    F4 nMin0[32], nMax0[32], nMin1[32], nMax1[32];
};

static V3 qRot(const F4& q, V3 v)
{
    const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z;
    const float w2 = q.w * q.w - 0.5f, d2 = q.x * vx + q.y * vy + q.z * vz;
    return V3{ vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d2,
               vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d2,
               vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d2 };
}
static V3 qRotInv(const F4& q, V3 v)
{
    const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z;
    const float w2 = q.w * q.w - 0.5f, d2 = q.x * vx + q.y * vy + q.z * vz;
    return V3{ vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * d2,
               vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * d2,
               vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * d2 };
}
static F4 qMul(const F4& a, const F4& b) // PxQuat::operator* (this = a)
{
    return F4{ a.w * b.x + b.w * a.x + a.y * b.z - a.z * b.y,
               a.w * b.y + b.w * a.y + a.z * b.x - a.x * b.z,
               a.w * b.z + b.w * a.z + a.x * b.y - a.y * b.x,
               a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z }; // sequential like PxQuat.h:390
}
static V3 v2s(V3 v, V3 sc, const F4& rot) // vertex2Shape, multiply transform
{
    const V3 r = qRot(rot, v);
    return qRotInv(rot, V3{ r.x * sc.x, r.y * sc.y, r.z * sc.z });
}

// prep transcription (cu L1447-1514): fills VA/VB (8 verts each) and GjkIn
static void refPrep(const TtoIn& s, float* VA, float* VB, GjkIn& gin)
{
    const F4 conj1{ -s.t1q.x, -s.t1q.y, -s.t1q.z, s.t1q.w };
    const F4 aToBq = qMul(conj1, s.t0q);
    const V3 aToBp = qRotInv(s.t1q, V3{ s.t0p.x - s.t1p.x, s.t0p.y - s.t1p.y, s.t0p.z - s.t1p.z });

    V3 mn0{ s.nMin0[0].x, s.nMin0[0].y, s.nMin0[0].z }, mx0{ s.nMax0[0].x, s.nMax0[0].y, s.nMax0[0].z };
    for (uint32_t L = 1; L < s.counts[0]; ++L)
    {
        mn0.x = std::fmin(mn0.x, s.nMin0[L].x); mn0.y = std::fmin(mn0.y, s.nMin0[L].y); mn0.z = std::fmin(mn0.z, s.nMin0[L].z);
        mx0.x = std::fmax(mx0.x, s.nMax0[L].x); mx0.y = std::fmax(mx0.y, s.nMax0[L].y); mx0.z = std::fmax(mx0.z, s.nMax0[L].z);
    }
    V3 mn1{ s.nMin1[0].x, s.nMin1[0].y, s.nMin1[0].z }, mx1{ s.nMax1[0].x, s.nMax1[0].y, s.nMax1[0].z };
    for (uint32_t L = 1; L < s.counts[1]; ++L)
    {
        mn1.x = std::fmin(mn1.x, s.nMin1[L].x); mn1.y = std::fmin(mn1.y, s.nMin1[L].y); mn1.z = std::fmin(mn1.z, s.nMin1[L].z);
        mx1.x = std::fmax(mx1.x, s.nMax1[L].x); mx1.y = std::fmax(mx1.y, s.nMax1[L].y); mx1.z = std::fmax(mx1.z, s.nMax1[L].z);
    }

    for (uint32_t i = 0; i < 8; ++i)
    {
        V3 p0{ (i & 1) ? mn0.x : mx0.x, (i & 2) ? mn0.y : mx0.y, (i & 4) ? mn0.z : mx0.z };
        V3 v0 = v2s(p0, V3{ s.scaleS0.x, s.scaleS0.y, s.scaleS0.z }, s.scaleR0);
        const V3 r0 = qRot(aToBq, v0);
        VA[i * 4 + 0] = r0.x + aToBp.x; VA[i * 4 + 1] = r0.y + aToBp.y; VA[i * 4 + 2] = r0.z + aToBp.z; VA[i * 4 + 3] = 0;

        V3 p1{ (i & 1) ? mn1.x : mx1.x, (i & 2) ? mn1.y : mx1.y, (i & 4) ? mn1.z : mx1.z };
        const V3 v1 = v2s(p1, V3{ s.scaleS1.x, s.scaleS1.y, s.scaleS1.z }, s.scaleR1);
        VB[i * 4 + 0] = v1.x; VB[i * 4 + 1] = v1.y; VB[i * 4 + 2] = v1.z; VB[i * 4 + 3] = 0;
    }

    const float sqDist = dot3(aToBp, aToBp);
    const V3 dir = sqDist > 0.f ? getNormalized(aToBp) : V3{ 0, 1, 0 };
    gin.initialDir[0] = dir.x; gin.initialDir[1] = dir.y; gin.initialDir[2] = dir.z; gin.initialDir[3] = 0;
    gin.cfg[0] = 8; gin.cfg[1] = 8; gin.cfg[2] = 0; gin.cfg[3] = 0;
    gin.params[0] = s.misc.x; gin.params[1] = 1.f - 0.000225f; gin.params[2] = 0; gin.params[3] = 0;
}

static bool eqF(float a, float b) { return almostEqual(a, b, 2e-4f, 2e-5f); }

PXG_TEST(npTriTriOverlap_overlaps)
{
    std::mt19937 rng(159001);
    const uint32_t N = 4000, MAXV = 8;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto randQuat = [&]() { float a[4]; float m2 = 0; for (int k = 0; k < 4; ++k) { a[k] = rf(-1, 1); m2 += a[k] * a[k]; } const float i = 1.f / std::sqrt(m2 > 0 ? m2 : 1.f); return F4{ a[0] * i, a[1] * i, a[2] * i, a[3] * i }; };

    std::vector<TtoIn> in(N);
    std::vector<GjkOut> ref(N);
    std::vector<float> refVA(N * MAXV * 4, 0.f), refVB(N * MAXV * 4, 0.f);
    std::vector<uint8_t> skip(N, 0);
    size_t nSkip = 0, cOverlap = 0, cDistant = 0, cClose = 0, cIdent = 0, cFallback = 0, cMulti = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        TtoIn s{};
        // separations from near-touching to clearly apart/overlapping
        const float off = rf(-1.5f, 6.f);
        s.t0q = randQuat(); s.t0p = F4{ off, rf(-1, 1), rf(-1, 1), 0 };
        s.t1q = randQuat(); s.t1p = F4{ rf(-1, 1), rf(-1, 1), rf(-1, 1), 0 };
        if (i % 19 == 4) { s.t0q = s.t1q; s.t0p = s.t1p; ++cFallback; } // identical transforms: aToB.p = 0 -> (0,1,0) fallback
        s.scaleS0 = F4{ rf(0.5f, 2), rf(0.5f, 2), rf(0.5f, 2), 0 }; s.scaleR0 = randQuat();
        s.scaleS1 = F4{ rf(0.5f, 2), rf(0.5f, 2), rf(0.5f, 2), 0 }; s.scaleR1 = randQuat();
        const uint32_t n0 = 1 + rng() % 32, n1 = 1 + rng() % 32;
        s.counts[0] = n0; s.counts[1] = n1;
        if (n0 > 1 || n1 > 1) ++cMulti;
        s.misc = F4{ rf(0.01f, 0.5f), 0, 0, 0 };
        if (i % 4 == 1)
        {
            // near band: separated surfaces within a wide contact distance
            s.misc.x = rf(0.5f, 2.5f);
            s.t0p.x = rf(3.f, 6.f);
        }
        for (int L = 0; L < 32; ++L)
        {
            // node boxes clustered around the origin; min <= max per component
            for (int c = 0; c < 3; ++c)
            {
                const float a = rf(-1.5f, 1.5f), b = rf(-1.5f, 1.5f);
                (&s.nMin0[L].x)[c] = std::fmin(a, b);
                (&s.nMax0[L].x)[c] = std::fmax(a, b);
                const float a1 = rf(-1.5f, 1.5f), b1 = rf(-1.5f, 1.5f);
                (&s.nMin1[L].x)[c] = std::fmin(a1, b1);
                (&s.nMax1[L].x)[c] = std::fmax(a1, b1);
            }
            s.nMin0[L].w = s.nMax0[L].w = s.nMin1[L].w = s.nMax1[L].w = 0;
        }
        in[i] = s;

        GjkIn gin{};
        refPrep(s, &refVA[i * MAXV * 4], &refVB[i * MAXV * 4], gin);
        ref[i] = gjkRef(gin, refVA, refVB, i * MAXV);

        // symmetric margin skip on the DISTANT decision boundary
        if (ref[i].ret != GJK_OVERLAP && std::fabs(ref[i].lowerBound - gin.params[0]) < 2e-4f) { skip[i] = 1; ++nSkip; continue; }
        if (ref[i].ret == GJK_OVERLAP) ++cOverlap;
        else if (ref[i].ret == GJK_DISTANT) ++cDistant;
        else ++cClose;
        if (i % 19 == 4) ++cIdent;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<float> vaInit(N * MAXV * 4, 0.f), vbInit(N * MAXV * 4, 0.f);
    std::vector<GjkOut> outInit(N);
    PxgBufferHandle bVA = createBufferFrom(dev, vaInit);
    PxgBufferHandle bVB = createBufferFrom(dev, vbInit);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    GjkParams prm = { N, MAXV, 0, 0 };
    Kernel k("npTriTriOverlap.spv", "triTriOverlap",
             { 16u, 16u, (uint32_t)sizeof(TtoIn), (uint32_t)sizeof(GjkOut) }, sizeof(GjkParams));
    k.dispatch(GROUPS, { bVA, bVB, bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<GjkOut>(dev, bOut, N);
    const auto gva = downloadBuffer<float>(dev, bVA, N * MAXV * 4);
    const auto gvb = downloadBuffer<float>(dev, bVB, N * MAXV * 4);
    size_t mmO = 0, mmV = 0, mmR = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        for (uint32_t j = 0; j < MAXV * 4; ++j)
        {
            if ((j & 3) == 3) continue;
            if (asU32(refVA[i * MAXV * 4 + j]) != asU32(gva[i * MAXV * 4 + j]) ||
                asU32(refVB[i * MAXV * 4 + j]) != asU32(gvb[i * MAXV * 4 + j])) { ++mmV; break; }
        }
        if (skip[i]) continue;
        const bool refOv = ref[i].ret != GJK_DISTANT, gOv = g[i].ret != GJK_DISTANT;
        if (refOv != gOv) ++mmO;
        if (g[i].ret != ref[i].ret || g[i].degenerate != ref[i].degenerate) ++mmR;
        // the iteration is bit-reproducible: the bounds must match EXACTLY
        else if (asU32(g[i].lowerBound) != asU32(ref[i].lowerBound) ||
                 asU32(g[i].upperBound) != asU32(ref[i].upperBound)) ++mmR;
    }
    PXG_CHECK_MSG(mmO == 0, "triTriOverlap: %zu overlap-flag mismatches", mmO);
    PXG_CHECK_MSG(mmR == 0, "triTriOverlap: %zu gjk-result mismatches", mmR);
    PXG_CHECK_MSG(mmV == 0, "triTriOverlap: %zu prep-point mismatches", mmV);
    PXG_CHECK_MSG(nSkip < N / 4, "triTriOverlap: too many marginal (%zu/%u)", nSkip, N);
    PXG_CHECK_MSG(cOverlap > N / 10 && cDistant > N / 10 && cClose > N / 20 && cIdent > N / 30 && cMulti > N / 2,
                  "triTriOverlap: coverage thin (ovl %zu dist %zu close %zu ident %zu multi %zu)",
                  cOverlap, cDistant, cClose, cIdent, cMulti);

    for (PxgBufferHandle h : { bVA, bVB, bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
