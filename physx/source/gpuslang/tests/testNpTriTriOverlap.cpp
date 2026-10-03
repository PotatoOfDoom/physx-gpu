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

// P9.83: the ENGINE-ABI half of this file builds real cooked-mesh blobs, and the
// BV32 node record is taken from the ENGINE HEADER rather than typed out.  The
// same reason testCvxTriSetup.cpp gives: this fixture would otherwise be the
// oracle for a layout it also defines, and a mutation sweep cannot see that.
#include "GuBV32.h"
#include "geometry/PxGeometry.h"   // PxGeometryType::eTRIANGLEMESH, never a literal (P9.57)

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
// RENAMED IN P9.83.  This was PX_MAX_F32, which foundation/PxSimpleTypes.h
// defines as a MACRO -- and GuBV32.h (included above for the engine's BV32 node
// layout) drags it in.  The macro and this constant hold the same value; the
// clash is purely lexical, and the local name is the one that gives way.
static const float TTO_MAX_F32 = 3.4028234663852886e38f;
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
    uint32_t bestN = 0xFFFFFFFFu; float bestDist = -TTO_MAX_F32;
    for (uint32_t i = 0; i < nb; i += 32)
    {
        float d[32];
        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            const uint32_t n = i + lane;
            V3 v = { V[(vbase + n) * 4], V[(vbase + n) * 4 + 1], V[(vbase + n) * 4 + 2] };
            d[lane] = (n < nb) ? dot3(dir, v) : -TTO_MAX_F32;
        }
        float chunkMax = -TTO_MAX_F32; for (uint32_t lane = 0; lane < 32; ++lane) chunkMax = std::fmax(chunkMax, d[lane]);
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
        closestPointV[L] = (c == 0) ? initialUnitDir.x : (c == 1) ? initialUnitDir.y : (c == 2) ? initialUnitDir.z : TTO_MAX_F32;
    }
    float upperBound = TTO_MAX_F32, lowerBound = -TTO_MAX_F32;
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

// P9.94d: aToB = transform1.transformInv(transform0) (cu L1447) is spelled as
// PxTransform::transformInv(PxTransform) (PxTransform.h:162-169): qinv =
// q.getConjugate(); (qinv.rotate(src.p - p), qinv * src.q).  The twin flag selects
// the pre-P9.94d qRotInv(q1, .) -- the same value, but an exact-zero component can
// take the other sign.  Used only by the vacuity probe in npTriTriOverlap_p94dZeroSign.
static bool gP94dRotInvTwin = false;

// prep transcription (cu L1447-1514): fills VA/VB (8 verts each) and GjkIn
static void refPrep(const TtoIn& s, float* VA, float* VB, GjkIn& gin)
{
    const F4 conj1{ -s.t1q.x, -s.t1q.y, -s.t1q.z, s.t1q.w };
    const F4 aToBq = qMul(conj1, s.t0q);
    const V3 dP{ s.t0p.x - s.t1p.x, s.t0p.y - s.t1p.y, s.t0p.z - s.t1p.z };
    const V3 aToBp = gP94dRotInvTwin ? qRotInv(s.t1q, dP) : qRot(conj1, dP);

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

// P9.94d: the aToB.p zero sign through the prep clouds, BITWISE.  q0 = (1,0,0,0)
// (180 deg about x), q1 = identity, p1 = 0, mesh 0's box min.y = +0, so its corners
// with y = +0 rotate to y = -0 and keep that sign only when aToB.p.y is -0 too.
//   case 0: p0 = (0,-0,-0): v = (+0,-0,-0); conj(q1).rotate(v) = (+0,-0,+0) where the
//           old q1.rotateInv(v) gave (+0,+0,+0) -> cloudA[3].y is -0 in the CUDA, +0
//           in the old port (the rotateInv twin must differ here);
//   case 1: p0 = (0,0,0): the spellings agree, but a "-rotate(qinv, p1 - p0)"
//           rewrite does not (rotate of (+0,+0,+0) is not the negation of rotate of
//           (+0,-0,-0)), so the sign-swapped forms are pinned too.
// Every VA/VB word is compared bitwise against the literal oracle.
PXG_TEST(npTriTriOverlap_p94dZeroSign)
{
    const uint32_t N = 2, MAXV = 8;
    std::vector<TtoIn> in(N);
    for (uint32_t c = 0; c < N; ++c)
    {
        TtoIn& s = in[c];
        s = TtoIn{};
        s.t0q = F4{ 1.0f, 0.0f, 0.0f, 0.0f };
        s.t0p = c == 0 ? F4{ 0.0f, -0.0f, -0.0f, 0 } : F4{ 0.0f, 0.0f, 0.0f, 0 };
        s.t1q = F4{ 0.0f, 0.0f, 0.0f, 1.0f }; s.t1p = F4{ 0.0f, 0.0f, 0.0f, 0 };
        s.scaleS0 = F4{ 1, 1, 1, 0 }; s.scaleR0 = F4{ 0, 0, 0, 1 };
        s.scaleS1 = F4{ 1, 1, 1, 0 }; s.scaleR1 = F4{ 0, 0, 0, 1 };
        s.counts[0] = 1; s.counts[1] = 1;
        s.misc = F4{ 0.1f, 0, 0, 0 };
        s.nMin0[0] = F4{ -1.0f, 0.0f, -1.0f, 0 }; s.nMax0[0] = F4{ 1.0f, 1.0f, 1.0f, 0 };
        s.nMin1[0] = F4{ -0.5f, -0.5f, -0.5f, 0 }; s.nMax1[0] = F4{ 0.5f, 0.5f, 0.5f, 0 };
    }

    std::vector<float> refVA(N * MAXV * 4, 0.f), refVB(N * MAXV * 4, 0.f);
    std::vector<float> twVA(N * MAXV * 4, 0.f), twVB(N * MAXV * 4, 0.f);
    for (uint32_t c = 0; c < N; ++c)
    {
        GjkIn gin{}, gtw{};
        refPrep(in[c], &refVA[c * MAXV * 4], &refVB[c * MAXV * 4], gin);
        gP94dRotInvTwin = true;
        refPrep(in[c], &twVA[c * MAXV * 4], &twVB[c * MAXV * 4], gtw);
        gP94dRotInvTwin = false;
    }
    size_t twinWords = 0;
    for (uint32_t j = 0; j < MAXV * 4; ++j)
        if ((j & 3) != 3 && asU32(refVA[j]) != asU32(twVA[j])) ++twinWords;
    std::printf("  [p94d] CUDA cloudA[3] %08x %08x %08x | rotateInv twin %08x %08x %08x (twin differs in %zu words)\n",
                asU32(refVA[12]), asU32(refVA[13]), asU32(refVA[14]), asU32(twVA[12]), asU32(twVA[13]), asU32(twVA[14]), twinWords);
    PXG_CHECK_MSG(asU32(refVA[13]) == 0x80000000u && asU32(twVA[13]) == 0u,
                  "p94d: hand-derived cloudA[3].y bits (CUDA -0, rotateInv +0) not reproduced");
    PXG_CHECK_MSG(twinWords > 0, "p94d: the rotateInv twin equals the literal oracle -- case not zero-sign sensitive");

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<float> vaInit(N * MAXV * 4, 7.f), vbInit(N * MAXV * 4, 7.f);
    std::vector<GjkOut> outInit(N);
    PxgBufferHandle bVA = createBufferFrom(dev, vaInit);
    PxgBufferHandle bVB = createBufferFrom(dev, vbInit);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);
    GjkParams prm = { N, MAXV, 0, 0 };
    Kernel k("npTriTriOverlap.spv", "triTriOverlap",
             { 16u, 16u, (uint32_t)sizeof(TtoIn), (uint32_t)sizeof(GjkOut) }, sizeof(GjkParams));
    k.dispatch(1, { bVA, bVB, bIn, bOut }, prm);
    dev->flush();
    const auto gva = downloadBuffer<float>(dev, bVA, N * MAXV * 4);
    const auto gvb = downloadBuffer<float>(dev, bVB, N * MAXV * 4);
    size_t mm = 0, eqTwin = 0;
    for (uint32_t j = 0; j < N * MAXV * 4; ++j)
    {
        if ((j & 3) == 3) continue;
        if (asU32(refVA[j]) != asU32(gva[j]) || asU32(refVB[j]) != asU32(gvb[j])) ++mm;
        if (asU32(twVA[j]) != asU32(refVA[j]) && asU32(twVA[j]) == asU32(gva[j])) ++eqTwin;
    }
    std::printf("  [p94d] GPU cloudA[3] %08x %08x %08x\n", asU32(gva[12]), asU32(gva[13]), asU32(gva[14]));
    PXG_CHECK_MSG(mm == 0, "p94d: %zu cloud words differ bitwise from the literal oracle (%zu take the rotateInv twin's bits)",
                  mm, eqTwin);
    for (PxgBufferHandle h : { bVA, bVB, bIn, bOut }) dev->destroyBuffer(h);
}

// ===========================================================================
// P9.83: triTriOverlapBda -- triangleTriangleOverlaps on the ENGINE ABI
// ===========================================================================
// The test above certifies the DECISION (pxgSlangTriTriOverlap + pxgSlangGjk).
// This one certifies the GLUE the engine kernel adds on top of it, which is the
// only thing in that file the module does not already cover:
//
//   1. the contact-manager load and the NO-SWAP operand order;
//   2. the cooked-blob walk to each mesh's BV32 ROOT node -- node 0, at +16;
//   3. the mNbNodes MASK, i.e. that slots past the root's child count are NOT
//      folded into the box;
//   4. contactDistance[ref0] + contactDistance[ref1], both halves;
//   5. the byte-lane store into an un-zeroed PxU8 array that FOUR pairs share.
//
// Every trap below exists because the corresponding bug is silent:
//
//   THE POISON SLOTS.  Child slots [mNbNodes, 32) of every root node carry
//   +/-1e6.  A kernel that folds all 32 (or that reads mNbNodes from the wrong
//   offset and gets something >= 32) produces a box a million units wide, and
//   EVERY pair then overlaps -- which looks exactly like "the physics works"
//   until something is supposed to be distant.  The expected pattern below has
//   five distant pairs for that reason.
//
//   THE SECOND NODE.  Every blob carries nbPackedNodes = 2, and node 1 is all
//   poison with mNbNodes = 32.  Reading node 1 instead of node 0 -- an off-by-one
//   in the +16 header skip, or BV32_STRIDE used as the header size -- lands there.
//
//   THE SLOT SPLIT.  The live slots tile the box: slot 0 holds the low corner,
//   slot n-1 the high one, and the union is the whole box.  A fold that stops
//   after slot 0 gets the right minimum and a far too small maximum, so it is
//   wrong in a direction a symmetric test would not notice.
//
//   THE UN-ZEROED ARRAY.  `overlaps` is uploaded as 0xFF everywhere, which is
//   what mIntermStackAlloc.allocateAligned hands the CUDA: a fresh, unwritten
//   allocation.  A store that ORs without clearing leaves every false flag
//   reading as true, and a store that does a plain read-modify-write loses three
//   of every four flags to the other warps sharing the word.
//
//   THE TAIL BYTES.  Eleven pairs occupy bytes 0..10 of a sixteen-byte buffer.
//   Bytes 11..15 must come back STILL 0xFF: a kernel that writes whole words
//   clobbers byte 11, and one whose `pairIdx >= numWorkItems` guard is wrong
//   writes past the work.
//
//   THE EXPECTED PATTERN.  T,F,F,T | F,T,T,F | T,F,T -- every one of the four
//   byte lanes carries both a true and a false somewhere.  With an alternating
//   pattern a shift computed as `(pairIdx & 1) * 8` would still pass.
namespace p983tto
{
struct GpuShape
{
    float sx, sy, sz;
    float rx, ry, rz, rw;
    uint32_t materialIndex;
    uint32_t hullLo, hullHi;
    uint32_t type;
    uint32_t particleOrSoftbodyId;
};
static_assert(sizeof(GpuShape) == 48, "PxgShape mirror");

struct CmInput { uint32_t shapeRef0, shapeRef1, tcRef0, tcRef1; };

// PxgNarrowphaseCore.cpp:1393-1401 -- ABI row { 4,8,8,8,8,8,8 }
struct TtoPush
{
    uint32_t numWorkItems;
    uint32_t pad4;
    uint64_t cmInputs, transformCache, contactDistance, shapes, cmOutputs, overlaps;
};
static_assert(sizeof(TtoPush) == 56, "triTriOverlapBda push block / ABI row { 4,8,8,8,8,8,8 }");

static const uint32_t POISON_SLOT_LO = 0xFFu;   // the byte the CUDA's allocation holds

// One cooked mesh: the uint4 header, then nbPackedNodes BV32 records.  The
// kernel never reads past the tree, so the vert/index/adjacency/remap sections
// are represented by trailing zeros only.
struct MeshBlob
{
    std::vector<uint8_t> bytes;
    F4 mins[32], maxs[32];     // what the ROOT node actually carries
    uint32_t nbNodes = 0;
};

static void buildMeshBlob(MeshBlob& m, uint32_t nbNodes, V3 lo, V3 hi)
{
    m.nbNodes = nbNodes;

    physx::Gu::BV32DataPacked root;
    physx::Gu::BV32DataPacked decoy;
    for (uint32_t s = 0; s < 32; ++s)
    {
        if (s < nbNodes)
        {
            const float t0 = float(s) / float(nbNodes), t1 = float(s + 1) / float(nbNodes);
            const V3 a{ lo.x + (hi.x - lo.x) * t0, lo.y + (hi.y - lo.y) * t0, lo.z + (hi.z - lo.z) * t0 };
            const V3 b{ lo.x + (hi.x - lo.x) * t1, lo.y + (hi.y - lo.y) * t1, lo.z + (hi.z - lo.z) * t1 };
            m.mins[s] = F4{ a.x, a.y, a.z, 0.0f };
            m.maxs[s] = F4{ b.x, b.y, b.z, 0.0f };
        }
        else
        {
            m.mins[s] = F4{ -1.0e6f, -1.0e6f, -1.0e6f, 0.0f };
            m.maxs[s] = F4{ 1.0e6f, 1.0e6f, 1.0e6f, 0.0f };
        }
        root.mMin[s] = physx::PxVec4(m.mins[s].x, m.mins[s].y, m.mins[s].z, m.mins[s].w);
        root.mMax[s] = physx::PxVec4(m.maxs[s].x, m.maxs[s].y, m.maxs[s].z, m.maxs[s].w);
        root.mData[s] = 0xBAD00000u | s;
        decoy.mMin[s] = physx::PxVec4(-1.0e6f, -1.0e6f, -1.0e6f, 0.0f);
        decoy.mMax[s] = physx::PxVec4(1.0e6f, 1.0e6f, 1.0e6f, 0.0f);
        decoy.mData[s] = 0xDEAD0000u | s;
    }
    root.mNbNodes = nbNodes;
    root.mDepth = 1;
    decoy.mNbNodes = 32;      // the trap: folding this box makes everything overlap
    decoy.mDepth = 2;

    auto u32 = [&](uint32_t v) { for (int i = 0; i < 4; ++i) m.bytes.push_back(uint8_t(v >> (8 * i))); };
    auto raw = [&](const void* p, size_t n)
    { const uint8_t* b = static_cast<const uint8_t*>(p); for (size_t i = 0; i < n; ++i) m.bytes.push_back(b[i]); };

    u32(12u);            // numVerts    -- never read by this kernel
    u32(4u);             // numTris     -- never read by this kernel
    u32(0xA5A5A5A5u);    // .z is NOT a node count (see pxgSlangCvxTriSetup)
    u32(2u);             // nbPackedNodes: node 0 is the root, node 1 is the decoy
    raw(&root, sizeof(root));
    raw(&decoy, sizeof(decoy));
    for (int i = 0; i < 64; ++i) u32(0);   // the sections past the tree
}
} // namespace p983tto

PXG_TEST(triTriOverlapEngineAbi)
{
    using namespace p983tto;
    PxgSlangComputeDevice* dev = testDevice();

    // ---- four meshes, four DIFFERENT root child counts ---------------------
    // 1 exercises the single-slot fold (where the seed IS the answer), 32 the
    // full one (where no slot is poison and the mask can only be wrong low), and
    // 3 / 7 the ordinary case with live poison above them.
    //
    // MESHES 4 AND 5 EXIST BECAUSE FOUR MUTANTS SURVIVED THE FIRST SWEEP.
    // `shape1 = shapes[shapeRef0]`, `b1 = meshRootBox(shape0)` and swapping the
    // two transforms all passed, because meshes 0..3 are four boxes of roughly
    // the same size, symmetric about their origins, and every pair was either a
    // deep intersection or fourteen units apart.  Under those conditions a
    // per-pair BOOLEAN cannot tell the two operands apart at all.
    //
    // Mesh 4 is a near-point and mesh 5 is a slab that extends only in +x from
    // its origin.  Pair 11 puts the point at x = 1.9 and the slab's origin at
    // x = 0, and the asymmetry does the work:
    //   correct        the point sits inside the slab's [0, 4] reach   -> TRUE
    //   both operands the point twice, 1.9 apart                       -> FALSE
    //   both meshes    same                                            -> FALSE
    //   swapped poses  point at 0, slab now reaching [1.9, 5.9]        -> FALSE
    // One pair, three mutants, and none of them needs a tolerance.
    MeshBlob mesh[6];
    buildMeshBlob(mesh[0], 1, V3{ -1.0f, -0.8f, -1.2f }, V3{ 1.0f, 0.9f, 1.1f });
    buildMeshBlob(mesh[1], 3, V3{ -0.6f, -1.4f, -0.5f }, V3{ 1.3f, 0.7f, 1.5f });
    buildMeshBlob(mesh[2], 7, V3{ -1.1f, -0.9f, -1.0f }, V3{ 0.8f, 1.2f, 0.6f });
    buildMeshBlob(mesh[3], 32, V3{ -0.7f, -0.7f, -1.3f }, V3{ 1.4f, 1.1f, 0.9f });
    buildMeshBlob(mesh[4], 2, V3{ -0.05f, -0.05f, -0.05f }, V3{ 0.05f, 0.05f, 0.05f });
    buildMeshBlob(mesh[5], 4, V3{ 0.0f, -0.05f, -0.05f }, V3{ 4.0f, 0.05f, 0.05f });

    PxgBufferHandle bMesh[6];
    uint64_t meshAddr[6];
    for (int i = 0; i < 6; ++i)
    {
        bMesh[i] = createBufferFrom(dev, mesh[i].bytes);
        meshAddr[i] = dev->getBufferDeviceAddress(bMesh[i]);
    }

    // ---- shapes: eight, all eTRIANGLEMESH, with NON-IDENTITY scale rotations
    // With every scale rotation the identity, vertex2Shape collapses to a
    // componentwise multiply and both quaternion rotations inside it vanish.
    auto axisQuat = [](int axis, float a) {
        F4 q{ 0, 0, 0, std::cos(a * 0.5f) };
        (&q.x)[axis] = std::sin(a * 0.5f);
        return q;
    };
    std::vector<GpuShape> shapes(10);
    struct ShapeDef { int mesh; V3 scale; int axis; float ang; };
    const ShapeDef sdef[10] = {
        { 0, { 1.0f, 1.0f, 1.0f }, 0, 0.0f },      // identity scale rotation: the trivial case, kept once
        { 1, { 1.4f, 0.7f, 1.1f }, 0, 0.85f },
        { 2, { 0.6f, 1.3f, 0.9f }, 1, 1.20f },
        { 3, { 1.1f, 1.1f, 0.5f }, 2, 0.45f },
        { 0, { 0.8f, 1.5f, 1.2f }, 1, 0.70f },
        { 1, { 1.2f, 0.9f, 1.4f }, 2, 1.05f },
        { 2, { 1.0f, 0.6f, 0.8f }, 0, 0.30f },
        { 3, { 0.9f, 1.2f, 1.0f }, 1, 0.95f },
        // 8 and 9 are the asymmetry pair: unit scale and an identity scale
        // rotation, so their whole difference is the MESH behind them.
        { 4, { 1.0f, 1.0f, 1.0f }, 0, 0.0f },   // the near-point
        { 5, { 1.0f, 1.0f, 1.0f }, 0, 0.0f },   // the +x slab
    };
    for (int i = 0; i < 10; ++i)
    {
        const ShapeDef& d = sdef[i];
        const F4 q = axisQuat(d.axis, d.ang);
        shapes[i].sx = d.scale.x; shapes[i].sy = d.scale.y; shapes[i].sz = d.scale.z;
        shapes[i].rx = q.x; shapes[i].ry = q.y; shapes[i].rz = q.z; shapes[i].rw = q.w;
        shapes[i].materialIndex = 0;
        shapes[i].hullLo = uint32_t(meshAddr[d.mesh] & 0xffffffffu);
        shapes[i].hullHi = uint32_t(meshAddr[d.mesh] >> 32);
        shapes[i].type = uint32_t(physx::PxGeometryType::eTRIANGLEMESH);
        shapes[i].particleOrSoftbodyId = 0;
    }

    // ---- the eleven pairs, and the separation that decides each flag -------
    // Target pattern T,F,F,T | F,T,T,F | T,F,T -- see the header.  `sep` is the
    // x-offset given to the FIRST body of the pair; the boxes are at most ~3
    // units across after scaling, so 0.3 overlaps and 14 is far outside any
    // contact distance.
    //
    // PAIR 5 IS THE `CLOSE` PAIR, AND IT IS THERE BECAUSE A MUTANT SURVIVED.
    // The CUDA stores `overlaps[i] = (result != eDISTANT)`, so GJK_CLOSE -- the
    // boxes do NOT intersect but sit within contactDistance of each other --
    // counts as an overlap.  The first sweep replaced that with
    // `result == GJK_OVERLAP` and nothing noticed: every true pair here was a
    // deep intersection and every false one was far outside any contact
    // distance, so the CLOSE band, which is the whole reason contactDistance is
    // passed to the GJK at all, was never entered.  Pair 5 enters it with a wide
    // contact distance and a real gap, and the mirror's verdict is asserted to
    // be exactly GJK_CLOSE below -- an observable that is also true under the
    // null hypothesis measures nothing.
    //
    // PAIRS 12 AND 13 EXIST FOR THE SAME REASON AS 11.  `cd[ref0] + cd[ref0]`
    // survived the first sweep because every contact distance in the fixture was
    // a few hundredths against gaps of zero or fourteen.  Each of the two now
    // carries ONE large half and one tiny one, with a gap that sits between the
    // doubled small half and the true sum -- 12 catches a kernel that reads only
    // slot 0, 13 one that reads only slot 1.  The pair has to exist twice
    // because one asymmetry only constrains one direction.
    //
    // `cd0`/`cd1` of zero mean "use the per-index default".  `identityPose` gives
    // a pair the identity rotation on both bodies, which pair 11 needs: its whole
    // argument is about WHERE each box sits, and a rotation would smear that.
    struct PairDef
    {
        uint32_t s0, s1, tc0, tc1;
        float sep;
        bool wantOverlap;
        bool wantClose;
        bool identityPose;
        float cd0, cd1;
    };
    const PairDef pdef[15] = {
        { 0, 1,  0,  1,  0.30f, true,  false, false, 0.0f,  0.0f  },
        { 1, 2,  2,  3, 14.00f, false, false, false, 0.0f,  0.0f  },
        { 2, 3,  4,  5, 15.50f, false, false, false, 0.0f,  0.0f  },
        { 3, 4,  6,  7,  0.10f, true,  false, false, 0.0f,  0.0f  },
        { 4, 5,  8,  9, 16.25f, false, false, false, 0.0f,  0.0f  },
        { 5, 6, 10, 11,  2.60f, true,  true,  false, 0.90f, 0.70f },  // the CLOSE band
        { 6, 7, 12, 13,  0.20f, true,  false, false, 0.0f,  0.0f  },
        { 7, 0, 14, 15, 17.00f, false, false, false, 0.0f,  0.0f  },
        { 0, 3, 16, 17,  0.05f, true,  false, false, 0.0f,  0.0f  },
        { 3, 6, 18, 19, 18.50f, false, false, false, 0.0f,  0.0f  },
        { 6, 1, 20, 21,  0.35f, true,  false, false, 0.0f,  0.0f  },
        { 8, 9, 22, 23,  1.90f, true,  false, true,  0.0f,  0.0f  },  // operand asymmetry
        { 0, 1, 24, 25,  3.00f, true,  true,  false, 0.05f, 1.20f },  // cd asymmetry, slot 0 small
        { 0, 1, 26, 27,  3.00f, true,  true,  false, 1.20f, 0.05f },  // cd asymmetry, slot 1 small
        { 9, 8, 28, 29, -1.90f, true,  false, true,  0.0f,  0.0f  },  // asymmetry, operands reversed
    };
    // FIFTEEN, AND THE ODD COUNT IS LOAD-BEARING.  The host grid is
    // ceil(N / 2) groups of two warps, so an EVEN N fills every warp and no
    // invocation ever carries `pairIdx == numWorkItems` -- which is the only
    // index at which `>=` and `>` differ.  At N = 14 the work-guard mutant
    // survived for exactly that reason, having been killed at N = 11.  Fifteen
    // pairs means eight groups, sixteen warps, and one spare that must write
    // nothing.
    const uint32_t N = 15;

    // ---- transforms: 22 entries, every one of them ROTATED ----------------
    // aToB is conj(q1)*q0 with rotInv(q1, p0 - p1); with q1 the identity the
    // conjugate is a no-op and the translation needs no rotating, so three
    // separate mistakes would become invisible.
    std::vector<float> xf(30 * 8, 0.0f);
    std::vector<float> cdist(30, 0.0f);
    for (uint32_t i = 0; i < N; ++i)
    {
        const PairDef& d = pdef[i];
        const float a0 = 0.23f + 0.31f * float(i), a1 = -0.41f + 0.17f * float(i);
        const F4 q0 = d.identityPose ? F4{ 0, 0, 0, 1 } : axisQuat(int(i % 3), a0);
        const F4 q1 = d.identityPose ? F4{ 0, 0, 0, 1 } : axisQuat(int((i + 1) % 3), a1);
        float* r0 = xf.data() + size_t(d.tc0) * 8;
        float* r1 = xf.data() + size_t(d.tc1) * 8;
        r0[0] = q0.x; r0[1] = q0.y; r0[2] = q0.z; r0[3] = q0.w;
        // The incidental y/z offsets vary per pair so no two transforms are the
        // same, but they stay SMALL beside the boxes: with 0.12f/i they reached
        // 1.2 by pair 10 and pushed a pair meant to overlap apart on their own,
        // which the intent check below caught.  `sep` is the only thing that is
        // supposed to decide a flag.
        const float jitter = d.identityPose ? 0.0f : 1.0f;
        r0[4] = d.sep; r0[5] = jitter * 0.04f * float(i); r0[6] = jitter * -0.03f * float(i);
        r1[0] = q1.x; r1[1] = q1.y; r1[2] = q1.z; r1[3] = q1.w;
        r1[4] = 0.0f; r1[5] = jitter * -0.03f * float(i); r1[6] = jitter * 0.02f * float(i);
        // BOTH halves of the sum, and deliberately unequal: a kernel reading only
        // cd[ref0] or only cd[ref1] gets a different minSep.  They are small
        // beside the separations above, so they never flip a decision by
        // themselves -- they are checked by the bit-exact lowerBound comparison.
        cdist[d.tc0] = d.cd0 > 0.0f ? d.cd0 : (0.010f + 0.003f * float(i));
        cdist[d.tc1] = d.cd1 > 0.0f ? d.cd1 : (0.070f + 0.005f * float(i));
    }

    std::vector<CmInput> cms(N);
    for (uint32_t i = 0; i < N; ++i)
        cms[i] = CmInput{ pdef[i].s0, pdef[i].s1, pdef[i].tc0, pdef[i].tc1 };

    // ---- the host mirror: build each pair's TtoIn the way the kernel's blob
    //      walk must, then run the SAME refPrep/gjkRef the test above uses ----
    std::vector<uint8_t> want(N, 0);
    std::vector<uint32_t> gotRet(N, 0);
    std::vector<float> gotLower(N, 0.f), gotMinSep(N, 0.f);
    std::vector<float> refVA(size_t(N) * 8 * 4, 0.f), refVB(size_t(N) * 8 * 4, 0.f);
    size_t nMarginal = 0, nTrue = 0, nFalse = 0, nClose = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const PairDef& d = pdef[i];
        const GpuShape& sh0 = shapes[d.s0];
        const GpuShape& sh1 = shapes[d.s1];
        const MeshBlob& m0 = mesh[sdef[d.s0].mesh];
        const MeshBlob& m1 = mesh[sdef[d.s1].mesh];
        const float* r0 = xf.data() + size_t(d.tc0) * 8;
        const float* r1 = xf.data() + size_t(d.tc1) * 8;

        TtoIn s{};
        s.t0q = F4{ r0[0], r0[1], r0[2], r0[3] }; s.t0p = F4{ r0[4], r0[5], r0[6], 0 };
        s.t1q = F4{ r1[0], r1[1], r1[2], r1[3] }; s.t1p = F4{ r1[4], r1[5], r1[6], 0 };
        s.scaleS0 = F4{ sh0.sx, sh0.sy, sh0.sz, 0 }; s.scaleR0 = F4{ sh0.rx, sh0.ry, sh0.rz, sh0.rw };
        s.scaleS1 = F4{ sh1.sx, sh1.sy, sh1.sz, 0 }; s.scaleR1 = F4{ sh1.rx, sh1.ry, sh1.rz, sh1.rw };
        s.counts[0] = m0.nbNodes; s.counts[1] = m1.nbNodes;
        s.misc = F4{ cdist[d.tc0] + cdist[d.tc1], 0, 0, 0 };
        for (int L = 0; L < 32; ++L)
        {
            s.nMin0[L] = m0.mins[L]; s.nMax0[L] = m0.maxs[L];
            s.nMin1[L] = m1.mins[L]; s.nMax1[L] = m1.maxs[L];
        }

        GjkIn gin{};
        refPrep(s, &refVA[size_t(i) * 32], &refVB[size_t(i) * 32], gin);
        const GjkOut o = gjkRef(gin, refVA, refVB, i * 8);
        want[i] = (o.ret != GJK_DISTANT) ? 1u : 0u;
        gotRet[i] = o.ret;
        gotLower[i] = o.lowerBound;
        gotMinSep[i] = gin.params[0];
        if (o.ret != GJK_OVERLAP && std::fabs(o.lowerBound - gin.params[0]) < 1e-3f) ++nMarginal;
        if (want[i]) ++nTrue; else ++nFalse;
        if (o.ret == GJK_CLOSE) ++nClose;
    }
    // THE PATTERN IS PART OF THE FIXTURE, NOT AN OBSERVATION OF IT.  If the
    // hand-picked separations stop producing it -- because a constant moved, or
    // because the geometry drifted -- the byte-lane traps silently weaken.  So
    // the mirror's own verdict is checked against the intent first.
    for (uint32_t i = 0; i < N; ++i)
        PXG_CHECK_MSG(bool(want[i]) == pdef[i].wantOverlap,
                      "triTriOverlapEngineAbi: pair %u was meant to be %s and the CPU mirror says %s -- "
                      "the separation no longer produces the designed byte pattern",
                      i, pdef[i].wantOverlap ? "overlapping" : "distant", want[i] ? "overlapping" : "distant");
    for (uint32_t i = 0; i < N; ++i)
        PXG_CHECK_MSG(!pdef[i].wantClose || gotRet[i] == GJK_CLOSE,
                      "triTriOverlapEngineAbi: pair %u is the CLOSE-band observable and the CPU mirror "
                      "returned %u (0 distant, 1 close, 2 overlap) with lowerBound %.6f against minSep "
                      "%.6f -- without a CLOSE pair the != eDISTANT test is indistinguishable from "
                      "== eOVERLAP",
                      i, gotRet[i], double(gotLower[i]), double(gotMinSep[i]));
    PXG_CHECK_MSG(nClose >= 1, "triTriOverlapEngineAbi: no pair reaches GJK_CLOSE");
    // The cd-asymmetry pairs only measure anything while the gap sits STRICTLY
    // between the doubled small half and the true sum.  Outside that window both
    // spellings agree and the pair silently stops testing the sum.
    for (uint32_t i = 0; i < N; ++i)
    {
        if (!(pdef[i].cd0 > 0.0f && pdef[i].cd1 > 0.0f) || pdef[i].identityPose) continue;
        const float small = std::fmin(pdef[i].cd0, pdef[i].cd1);
        const float sum = pdef[i].cd0 + pdef[i].cd1;
        if (!(small * 2.0f < sum * 0.5f)) continue;   // pair 5 is the CLOSE pair, not an asymmetry one
        PXG_CHECK_MSG(gotLower[i] > small * 2.0f && gotLower[i] < sum,
                      "triTriOverlapEngineAbi: pair %u is a contact-distance asymmetry pair and its gap "
                      "%.6f is outside (%.6f, %.6f) -- a kernel reading only one half of the sum would "
                      "still get the same flag",
                      i, double(gotLower[i]), double(small * 2.0f), double(sum));
    }
    PXG_CHECK_MSG(nMarginal == 0, "triTriOverlapEngineAbi: %zu pair(s) sit on the DISTANT boundary", nMarginal);
    PXG_CHECK_MSG(nTrue >= 4 && nFalse >= 4,
                  "triTriOverlapEngineAbi: coverage thin (%zu overlapping, %zu distant)", nTrue, nFalse);

    // ---- run ---------------------------------------------------------------
    // Sixteen bytes for eleven pairs: bytes 11..15 are the tail trap.  0xFF is
    // what a fresh mIntermStackAlloc allocation hands the CUDA.
    // Twenty-four bytes for fifteen pairs.  Bytes 15..23 are the tail trap, and
    // byte 15 does double duty: it shares word 3 with pairs 12..14, so a store
    // that writes whole words clobbers it, and it is also the byte the spare
    // sixteenth warp would reach if the work guard were `>` instead of `>=`.
    std::vector<uint32_t> overlapInit(6, 0xFFFFFFFFu);
    std::vector<uint32_t> cmOutputs(N * 12, 0x5A5A5A5Au);   // pushed, never read

    PxgBufferHandle bShapes = createBufferFrom(dev, shapes);
    PxgBufferHandle bXf = createBufferFrom(dev, xf);
    PxgBufferHandle bCd = createBufferFrom(dev, cdist);
    PxgBufferHandle bCm = createBufferFrom(dev, cms);
    PxgBufferHandle bOv = createBufferFrom(dev, overlapInit);
    PxgBufferHandle bCmOut = createBufferFrom(dev, cmOutputs);

    TtoPush push{};
    push.numWorkItems = N;
    push.cmInputs = dev->getBufferDeviceAddress(bCm);
    push.transformCache = dev->getBufferDeviceAddress(bXf);
    push.contactDistance = dev->getBufferDeviceAddress(bCd);
    push.shapes = dev->getBufferDeviceAddress(bShapes);
    push.cmOutputs = dev->getBufferDeviceAddress(bCmOut);
    push.overlaps = dev->getBufferDeviceAddress(bOv);

    {
        // THE HOST'S OWN GRID, verbatim (PxgNarrowphaseCore.cpp:1407):
        // numBlocks = ceil(numTests / NP_TRIMESH_WARPS_PER_BLOCK).  With N = 11
        // and warpsPerBlock = 2 that is 6 groups covering 12 warps, so the last
        // group's second warp is past the work and must write nothing.
        const uint32_t WARPS_PER_BLOCK = 2;
        const uint32_t groups = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        Kernel k("triTriOverlapBda.spv", "triangleTriangleOverlaps", {}, sizeof(TtoPush));
        k.dispatch(groups, {}, push, 1u, 1u);
        dev->flush();
    }

    const auto gotWords = downloadBuffer<uint32_t>(dev, bOv, 6);
    const uint8_t* got = reinterpret_cast<const uint8_t*>(gotWords.data());

    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        // The CUDA stores `overlaps[i] = (result != eDISTANT)`, a bool, so the
        // byte is exactly 0 or 1 -- not "non-zero".  A stale 0xFF would read as
        // true to the consumer but is still a defect and is reported as one.
        const uint8_t expect = want[i] ? 1u : 0u;
        if (got[i] != expect)
        {
            ++mm;
            std::printf("  pair %2u: got 0x%02x want 0x%02x%s\n", i, got[i], expect,
                        got[i] == POISON_SLOT_LO ? "   (still the uploaded poison -- nothing was stored)" : "");
        }
    }
    PXG_CHECK_MSG(mm == 0, "triTriOverlapEngineAbi: %zu overlap byte(s) wrong", mm);

    size_t tail = 0;
    for (uint32_t b = N; b < 24; ++b)
        if (got[b] != POISON_SLOT_LO) { ++tail; std::printf("  tail byte %u: 0x%02x\n", b, got[b]); }
    PXG_CHECK_MSG(tail == 0,
                  "triTriOverlapEngineAbi: %zu byte(s) past numWorkItems were overwritten -- "
                  "the store is not byte-scoped or the work guard is wrong", tail);

    for (PxgBufferHandle h : { bShapes, bXf, bCd, bCm, bOv, bCmOut }) dev->destroyBuffer(h);
    for (int i = 0; i < 4; ++i) dev->destroyBuffer(bMesh[i]);
}

} // namespace
