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

// Verification of npGjkWarm.slang: ctGjkWarm — the full gjk() driver with the
// warm-start cache-reuse path (gjk.cuh :243-418).  Each test runs a bit-exact
// CPU transcription of the driver first COLD (cacheSize 0) to obtain a genuine
// GjkCachedData (simplex + size), then feeds that cache back WARM to exercise the
// warm-start candidate fold + restart path.  The shader is compared bit-exactly
// against the same CPU transcription for both the cold and warm calls.  Integer
// hulls keep the whole run bit-for-bit reproducible.

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
static const float MAXF = 3.4028234663852886e38f;
static const uint32_t DEG16 = 0xffffu;
static uint32_t asU32(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float asF32(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

static void dotW(const float a[32], const float b[32], float o[32]) { float x[32]; for (uint32_t L = 0; L < 32; ++L) x[L] = a[L] * b[L]; for (uint32_t L = 0; L < 32; ++L) { const float s1 = (L + 1 < 32) ? x[L + 1] : x[L]; const float s2 = (L + 2 < 32) ? x[L + 2] : x[L]; o[L] = x[L] + s1 + s2; } }
static void crossW(const float a[32], const float b[32], const uint32_t c0[32], const uint32_t c1[32], float o[32]) { for (uint32_t L = 0; L < 32; ++L) o[L] = a[c0[L]] * b[c1[L]] - a[c1[L]] * b[c0[L]]; }
static void splatXW(const float a[32], float o[32]) { for (uint32_t L = 0; L < 32; ++L) o[L] = a[L & 28u]; }
static void splatVW(uint32_t index, const float a[32], float o[32]) { for (uint32_t L = 0; L < 32; ++L) o[L] = a[((index << 2) | (L & 3u)) & 31u]; }
static void magW(const float v[32], float cm[32]) { float dv[32]; dotW(v, v, dv); for (uint32_t L = 0; L < 32; ++L) cm[L] = std::sqrt(dv[L & 28u]); }
static uint32_t next3(uint32_t x) { return x == 2u ? 0u : x + 1u; }
static uint32_t prev3(uint32_t x) { return x == 0u ? 2u : x - 1u; }
static uint32_t ffs(uint32_t m) { if (!m) return 0; uint32_t i = 0; while (!((m >> i) & 1u)) ++i; return i + 1; }
static uint32_t fbl(uint32_t m) { if (!m) return 0xffffffffu; uint32_t i = 0; while (!((m >> i) & 1u)) ++i; return i; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 crossX(V3 a, V3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static V3 absX(V3 a) { return { std::fabs(a.x), std::fabs(a.y), std::fabs(a.z) }; }
static V3 sub3(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 getNormalized(V3 v) { const float m = dot3(v, v); if (m > 0.0f) { const float r = 1.0f / std::sqrt(m); return { v.x * r, v.y * r, v.z * r }; } return { 0, 0, 0 }; }

static uint32_t closestPointTriW(float zV[32], const float cV[32], float cp[32])
{
    uint32_t c0[32], c1[32]; for (uint32_t L = 0; L < 32; ++L) { c0[L] = ((L & 28u) | next3(L & 3u)) & 31u; c1[L] = ((L & 28u) | prev3(L & 3u)) & 31u; }
    for (uint32_t L = 0; L < 32; ++L) cp[L] = 0.0f;
    float eV[32]; for (uint32_t L = 0; L < 32; ++L) eV[L] = cV[L] - zV[L];
    float dee[32]; dotW(eV, eV, dee); float lD[32]; splatXW(dee, lD);
    if ((lD[0] == 0.0f) || (lD[4] == 0.0f)) return DEG16;
    float se[32]; for (uint32_t L = 0; L < 32; ++L) se[L] = (lD[L ^ 4u] < lD[L]) ? eV[L ^ 4u] : eV[L];
    float d1[32]; for (uint32_t L = 0; L < 32; ++L) d1[L] = zV[L ^ 4u] - zV[L];
    float mV[32]; crossW(d1, se, c0, c1, mV);
    float pS[32]; dotW(eV, cV, pS);
    float cme[32]; crossW(mV, eV, c0, c1, cme); float dcme[32]; dotW(cme, cV, dcme);
    bool uS[32]; for (uint32_t L = 0; L < 32; ++L) uS[L] = dcme[L] <= 0.0f;
    if (!(uS[0] || uS[4])) { float dmc[32]; dotW(mV, cV, dmc); const float projD = dmc[0]; float dmm[32]; dotW(mV, mV, dmm); float sx[32]; splatXW(dmm, sx); float tmp[32]; for (uint32_t L = 0; L < 32; ++L) tmp[L] = mV[L] * projD / sx[L]; splatVW(0u, tmp, cp); const bool flip = projD < 0.0f; for (uint32_t L = 0; L < 32; ++L) if ((L & 3u) == 3u) mV[L] = projD; const uint32_t idx = flip ? 1u : 0u; for (uint32_t L = 0; L < 32; ++L) { const uint32_t src = ((idx << 2) | (L & 3u)) & 31u; zV[L] = (L < 16u || L > 19u) ? zV[L] : mV[src]; } return flip ? 0x0123u : 0x1023u; }
    uint32_t em = 0u; if (uS[0] && pS[0] > 0.0f) em |= 1u; if (uS[4] && pS[4] > 0.0f) em |= 2u; uint32_t eid = ffs(em);
    if (eid != 0u) { eid -= 1u; float sp[32]; splatXW(pS, sp); float tmp[32]; for (uint32_t L = 0; L < 32; ++L) tmp[L] = cV[L] - eV[L] * sp[L] / lD[L]; splatVW(eid, tmp, cp); return (eid << 8) | 0x22u; }
    for (uint32_t L = 0; L < 32; ++L) cp[L] = cV[L]; return 0x0021u;
}
static uint32_t closestPointTetW(float zV[32], const float dV[32], float cp[32])
{
    for (uint32_t L = 0; L < 32; ++L) cp[L] = 0.0f;
    float n0[32]; splatVW(4u, zV, n0); float dn0d[32]; dotW(n0, dV, dn0d); float dn0z[32]; dotW(n0, zV, dn0z);
    if (dn0d[0] > dn0z[0]) return DEG16;
    uint32_t c0[32], c1[32], vN[32], vP[32]; for (uint32_t L = 0; L < 32; ++L) { c0[L] = ((L & 28u) | next3(L & 3u)) & 31u; c1[L] = ((L & 28u) | prev3(L & 3u)) & 31u; vN[L] = (L > 7u) ? (L - 8u) : (L + 4u); vP[L] = (L < 4u) ? (L + 8u) : (L - 4u); }
    float eV[32]; for (uint32_t L = 0; L < 32; ++L) eV[L] = dV[L] - zV[L]; float dee[32]; dotW(eV, eV, dee); float lD[32]; splatXW(dee, lD);
    if ((lD[0] == 0.0f) || (lD[4] == 0.0f) || (lD[8] == 0.0f)) return DEG16;
    float se[32]; for (uint32_t L = 0; L < 32; ++L) se[L] = (lD[vN[L]] < lD[L]) ? eV[vN[L]] : eV[L];
    float d2[32]; for (uint32_t L = 0; L < 32; ++L) d2[L] = zV[vN[L]] - zV[L]; float nV[32]; crossW(se, d2, c0, c1, nV);
    float aS[32]; dotW(nV, dV, aS); float zvp[32]; for (uint32_t L = 0; L < 32; ++L) zvp[L] = zV[vP[L]]; float gS[32]; dotW(zvp, nV, gS);
    if ((aS[0] <= gS[0]) || (aS[4] <= gS[4]) || (aS[8] <= gS[8])) return DEG16;
    for (uint32_t L = 0; L < 32; ++L) if ((L & 3u) == 3u) nV[L] = aS[L - 3u];
    if (aS[0] > 0.0f && aS[4] > 0.0f && aS[8] > 0.0f) { for (uint32_t L = 0; L < 32; ++L) if (L >= 20u) zV[L] = nV[L - 20u]; return 0x2104u; }
    float pS[32]; dotW(eV, dV, pS);
    if (pS[0] <= 0.0f && pS[4] <= 0.0f && pS[8] <= 0.0f) { for (uint32_t L = 0; L < 32; ++L) cp[L] = dV[L]; return 0x2131u; }
    float cen[32]; crossW(eV, nV, c0, c1, cen); float dcen[32]; dotW(cen, dV, dcen); bool uiS[32]; for (uint32_t L = 0; L < 32; ++L) uiS[L] = dcen[L] <= 0.0f;
    float evn[32]; for (uint32_t L = 0; L < 32; ++L) evn[L] = eV[vN[L]]; float cevn[32]; crossW(evn, nV, c0, c1, cevn); float dcevn[32]; dotW(cevn, dV, dcevn); bool ujS[32]; for (uint32_t L = 0; L < 32; ++L) ujS[L] = dcevn[L] > 0.0f;
    uint32_t em = 0u; if (uiS[0] && ujS[vP[0]] && pS[0] >= 0.0f) em |= 1u; if (uiS[4] && ujS[vP[4]] && pS[4] >= 0.0f) em |= 2u; if (uiS[8] && ujS[vP[8]] && pS[8] >= 0.0f) em |= 4u; uint32_t eid = ffs(em);
    if (eid != 0u) { eid -= 1u; float pl[32]; for (uint32_t L = 0; L < 32; ++L) pl[L] = pS[L] / lD[L]; float sp[32]; splatXW(pl, sp); float tmp[32]; for (uint32_t L = 0; L < 32; ++L) tmp[L] = dV[L] - eV[L] * sp[L]; splatVW(eid, tmp, cp); return (eid << 8) | 0x32u; }
    uint32_t fm = 0u; if (!uiS[0] && !ujS[0] && aS[0] <= 0.0f) fm |= 1u; if (!uiS[4] && !ujS[4] && aS[4] <= 0.0f) fm |= 2u; if (!uiS[8] && !ujS[8] && aS[8] <= 0.0f) fm |= 4u; uint32_t fid = ffs(fm);
    if (fid != 0u) { fid -= 1u; float dnn[32]; dotW(nV, nV, dnn); float ad[32]; for (uint32_t L = 0; L < 32; ++L) ad[L] = aS[L] / dnn[L]; float sad[32]; splatXW(ad, sad); float tmp[32]; for (uint32_t L = 0; L < 32; ++L) tmp[L] = nV[L] * sad[L]; splatVW(fid, tmp, cp); for (uint32_t L = 0; L < 32; ++L) { const uint32_t src = ((fid << 2) | (L & 3u)) & 31u; zV[L] = (L < 16u || L > 19u) ? zV[L] : -nV[src]; } return (next3(fid) << 12) | (fid << 8) | 0x33u; }
    return DEG16;
}
static uint32_t gSupport(V3 dir, const std::vector<float>& V, uint32_t vbase, uint32_t nb)
{
    uint32_t bestN = 0xFFFFFFFFu; float bestDist = -MAXF;
    for (uint32_t i = 0; i < nb; i += 32) { float d[32]; for (uint32_t l = 0; l < 32; ++l) { const uint32_t n = i + l; V3 v = { V[(vbase + n) * 4], V[(vbase + n) * 4 + 1], V[(vbase + n) * 4 + 2] }; d[l] = (n < nb) ? dot3(dir, v) : -MAXF; } float cm = -MAXF; for (uint32_t l = 0; l < 32; ++l) cm = std::fmax(cm, d[l]); if (cm > bestDist) { uint32_t ll = 0; for (uint32_t l = 0; l < 32; ++l) if (d[l] == cm) { ll = l; break; } bestDist = cm; bestN = i + ll; } }
    return bestN;
}
struct GjkOut { uint32_t ret, degenerate, size, pad; float lowerBound, upperBound, p1, p2; float direction[4], closestPointA[4], closestPointB[4], closestPointDir[4]; };

// full gjk() with warm-start (mirrors npGjkWarm.slang); returns output + final simplex + size
static GjkOut gjkWarmRef(V3 initialDir, const std::vector<float>& VA, const std::vector<float>& VB, uint32_t vbase, uint32_t nbA, uint32_t nbB, float minSep, float convRatio, uint32_t cacheSize, const float cacheIn[32], float simplexOut[32])
{
    uint32_t cacheMask = (1u << cacheSize) - 1u;
    float simplexV[32]; for (uint32_t L = 0; L < 32; ++L) simplexV[L] = cacheIn[L];
    float closestPointV[32]; float cmA[32]; float upperBound = MAXF, lowerBound = -MAXF; int size = 0, backupSize = 0; bool degenerate = false;
    float bkS[32], bkC[32]; V3 directionOut = { 0, 0, 0 };
    bool restart = true;
    while (restart)
    {
        restart = false; bool warmStarted = false; size = 0; backupSize = 0;
        for (uint32_t L = 0; L < 32; ++L) { bkS[L] = 0.0f; bkC[L] = 0.0f; }
        if (cacheMask != 0u) splatVW(0u, simplexV, closestPointV);
        else for (uint32_t L = 0; L < 32; ++L) { const uint32_t c = L & 3u; closestPointV[L] = (c == 0) ? initialDir.x : (c == 1) ? initialDir.y : (c == 2) ? initialDir.z : MAXF; }
        upperBound = MAXF; lowerBound = -MAXF; magW(closestPointV, cmA);
        bool run = true;
        while (run)
        {
            float dV[32]; for (uint32_t L = 0; L < 32; ++L) dV[L] = closestPointV[L] / cmA[L];
            uint32_t candidates = 0u;
            if (cacheMask != 0u) { float dd[32]; dotW(dV, simplexV, dd); for (uint32_t L = 0; L < 32; ++L) if (dd[(L << 2) & 31u] < upperBound) candidates |= (1u << L); candidates &= cacheMask; }
            float sup[32];
            if (candidates != 0u)
            {
                warmStarted = true; const uint32_t lowBit = candidates & (0u - candidates); cacheMask &= ~(lowBit * 2u - 1u); const uint32_t lowIdx = fbl(candidates);
                splatVW(lowIdx, simplexV, sup);
                for (uint32_t L = 0; L < 32; ++L) if ((L >> 2) == (uint32_t)size) simplexV[L] = sup[L];
            }
            else
            {
                cacheMask = 0u;
                V3 dv = { dV[0], dV[1], dV[2] };
                const uint32_t iA = gSupport({ -dv.x, -dv.y, -dv.z }, VA, vbase, nbA), iB = gSupport(dv, VB, vbase, nbB);
                V3 vA = { VA[(vbase + iA) * 4], VA[(vbase + iA) * 4 + 1], VA[(vbase + iA) * 4 + 2] }, vB = { VB[(vbase + iB) * 4], VB[(vbase + iB) * 4 + 1], VB[(vbase + iB) * 4 + 2] };
                V3 sv = sub3(vA, vB);
                for (uint32_t L = 0; L < 32; ++L) { const uint32_t c = L & 3u; sup[L] = (c == 0) ? sv.x : (c == 1) ? sv.y : (c == 2) ? sv.z : 0.0f; }
                const float enc = asF32(0x3f800000u | ((iB & 0xffu) << 8) | (iA & 0xffu));
                for (uint32_t L = 0; L < 32; ++L) if ((L >> 2) == (uint32_t)size) simplexV[L] = ((L & 3u) < 3u) ? sup[L] : enc;
                float dds[32]; dotW(dV, sup, dds); const float lb = dds[0];
                if (lb > lowerBound) { directionOut = { -dV[0], -dV[1], -dV[2] }; lowerBound = lb; if (lowerBound > minSep || lowerBound >= upperBound * convRatio) { run = false; break; } }
            }
            uint32_t f = 0u; float pv[32];
            if (size == 0) { f = 0x2101u; for (uint32_t L = 0; L < 32; ++L) pv[L] = sup[L]; }
            else if (size == 1) { float aV[32]; splatVW(0u, simplexV, aV); float lV[32]; for (uint32_t L = 0; L < 32; ++L) lV[L] = sup[L] - aV[L]; float dll[32]; dotW(lV, lV, dll); float mm[32]; splatXW(dll, mm); float dal[32]; dotW(aV, lV, dal); float xx[32]; splatXW(dal, xx); for (uint32_t L = 0; L < 32; ++L) xx[L] = -xx[L]; if (mm[0] == 0.0f) { f = DEG16; for (uint32_t L = 0; L < 32; ++L) pv[L] = 0.0f; } else if (xx[0] < mm[0]) { f = 0x2102u; for (uint32_t L = 0; L < 32; ++L) pv[L] = aV[L] + lV[L] * xx[L] / mm[L]; } else { f = 0x2111u; for (uint32_t L = 0; L < 32; ++L) pv[L] = sup[L]; } }
            else if (size == 2) f = closestPointTriW(simplexV, sup, pv);
            else f = closestPointTetW(simplexV, sup, pv);
            magW(pv, cmA); const int newSize = (int)(f & 0xfu);
            degenerate = (f == DEG16) || (cmA[0] >= upperBound && newSize <= size);
            if (degenerate) { if (!warmStarted) { run = false; break; } else { cacheMask = 0u; restart = true; run = false; break; } }
            backupSize = (cmA[0] >= upperBound) ? size : 0;
            if (backupSize != 0) { for (uint32_t L = 0; L < 32; ++L) { bkS[L] = simplexV[L]; bkC[L] = closestPointV[L]; } } else upperBound = cmA[0];
            for (uint32_t L = 0; L < 32; ++L) closestPointV[L] = pv[L]; size = newSize;
            float tp[32]; for (uint32_t L = 0; L < 32; ++L) { const uint32_t si = (f >> ((L & 28u) + 4u)) & 0xfu; tp[L] = simplexV[((si << 2) | (L & 3u)) & 31u]; }
            for (uint32_t L = 0; L < 32; ++L) if (L < 12u) simplexV[L] = tp[L];
            if (!(upperBound > 0.0f && lowerBound < upperBound * convRatio)) run = false;
        }
    }
    if (backupSize > 0) { for (uint32_t L = 0; L < 32; ++L) { simplexV[L] = bkS[L]; closestPointV[L] = bkC[L]; } size = backupSize; }
    for (uint32_t L = 0; L < 32; ++L) simplexOut[L] = simplexV[L];
    GjkOut o; std::memset(&o, 0, sizeof(o));
    o.degenerate = degenerate ? 1u : 0u; o.lowerBound = lowerBound; o.upperBound = upperBound; o.size = (uint32_t)size;
    o.direction[0] = directionOut.x; o.direction[1] = directionOut.y; o.direction[2] = directionOut.z;
    if (upperBound == 0.0f) { o.ret = 2u; return o; }
    if (lowerBound > minSep) { o.ret = 0u; return o; }
    V3 cp = { closestPointV[0], closestPointV[1], closestPointV[2] }; V3 gn = getNormalized(cp); V3 dir = { -gn.x, -gn.y, -gn.z };
    o.closestPointDir[0] = dir.x; o.closestPointDir[1] = dir.y; o.closestPointDir[2] = dir.z;
    const uint32_t a0 = asU32(simplexV[3]) & 0xffu, a1 = asU32(simplexV[7]) & 0xffu, a2 = asU32(simplexV[11]) & 0xffu;
    auto va = [&](uint32_t i) -> V3 { return { VA[(vbase + i) * 4], VA[(vbase + i) * 4 + 1], VA[(vbase + i) * 4 + 2] }; };
    V3 cpA;
    if (size == 3) { V3 A = { simplexV[0], simplexV[1], simplexV[2] }, B = { simplexV[4], simplexV[5], simplexV[6] }, C = { simplexV[8], simplexV[9], simplexV[10] }; V3 m = crossX(sub3(B, A), sub3(C, A)); V3 ma = absX(m); V3 s1 = crossX(sub3(B, cp), sub3(C, cp)); V3 s2 = crossX(sub3(C, cp), sub3(A, cp)); float d, u, v; if (ma.x >= ma.y && ma.x >= ma.z) { d = 1.0f / m.x; u = s1.x * d; v = s2.x * d; } else if (ma.y >= ma.z) { d = 1.0f / m.y; u = s1.y * d; v = s2.y * d; } else { d = 1.0f / m.z; u = s1.z * d; v = s2.z * d; } const float w = 1.0f - u - v; V3 p0 = va(a0), p1 = va(a1), p2 = va(a2); cpA = { u * p0.x + v * p1.x + w * p2.x, u * p0.y + v * p1.y + w * p2.y, u * p0.z + v * p1.z + w * p2.z }; }
    else if (size == 2) { V3 V0 = { simplexV[0], simplexV[1], simplexV[2] }, V1 = { simplexV[4], simplexV[5], simplexV[6] }; V3 dv = sub3(V1, V0); const float q = -dot3(V0, dv) / dot3(dv, dv); V3 p0 = va(a0), p1 = va(a1); cpA = { (1.0f - q) * p0.x + q * p1.x, (1.0f - q) * p0.y + q * p1.y, (1.0f - q) * p0.z + q * p1.z }; }
    else cpA = va(a0);
    o.closestPointA[0] = cpA.x; o.closestPointA[1] = cpA.y; o.closestPointA[2] = cpA.z;
    o.closestPointB[0] = cpA.x + upperBound * dir.x; o.closestPointB[1] = cpA.y + upperBound * dir.y; o.closestPointB[2] = cpA.z + upperBound * dir.z;
    o.ret = 1u; return o;
}

struct GjkParams { uint32_t numTests, maxV, pad0, pad1; };
struct GjkIn { float initialDir[4]; uint32_t cfg[4]; float params[4]; };

PXG_TEST(npGjkWarm_driver)
{
    std::mt19937 rng(1330133);
    const uint32_t MAXV = 32;
    std::vector<float> VA, VB, Cache;
    std::vector<GjkIn> in;
    std::vector<GjkOut> ref;
    size_t nCold = 0, nWarm = 0, nWarmOverlap = 0;

    auto ic = [&](int r) -> int { return int(rng() % (2 * r + 1)) - r; };

    auto pushOne = [&](const std::vector<float>& tA, const std::vector<float>& tB, uint32_t nbA, uint32_t nbB, V3 d0, float minSep, uint32_t cacheSize, const float cache[32])
    {
        const uint32_t vbase = (uint32_t)(VA.size() / 4);
        for (uint32_t k = 0; k < MAXV; ++k) { VA.push_back(tA[k * 4]); VA.push_back(tA[k * 4 + 1]); VA.push_back(tA[k * 4 + 2]); VA.push_back(0); VB.push_back(tB[k * 4]); VB.push_back(tB[k * 4 + 1]); VB.push_back(tB[k * 4 + 2]); VB.push_back(0); }
        for (uint32_t L = 0; L < 32; ++L) Cache.push_back(cache[L]);
        GjkIn e; std::memset(&e, 0, sizeof(e)); e.initialDir[0] = d0.x; e.initialDir[1] = d0.y; e.initialDir[2] = d0.z; e.cfg[0] = nbA; e.cfg[1] = nbB; e.cfg[2] = cacheSize; e.params[0] = minSep; e.params[1] = 0.98f;
        in.push_back(e);
        float dummy[32];
        ref.push_back(gjkWarmRef(d0, VA, VB, vbase, nbA, nbB, minSep, 0.98f, cacheSize, cache, dummy));
    };

    uint32_t attempts = 0;
    while ((nCold < 900 || nWarm < 600) && attempts < 30000)
    {
        ++attempts;
        std::vector<float> tA(MAXV * 4, 0.0f), tB(MAXV * 4, 0.0f);
        const uint32_t nbA = 5 + rng() % 20, nbB = 5 + rng() % 20;
        for (uint32_t k = 0; k < nbA; ++k) { tA[k * 4] = float(ic(6)); tA[k * 4 + 1] = float(ic(6)); tA[k * 4 + 2] = float(ic(6)); }
        const int shift = (int)(rng() % 3); // 0 overlap, 1 close, 2 distant mix
        const int sh = (shift == 0) ? ic(2) : (shift == 1) ? (8 + int(rng() % 6)) : (25 + int(rng() % 15));
        for (uint32_t k = 0; k < nbB; ++k) { tB[k * 4] = float(ic(6) + sh); tB[k * 4 + 1] = float(ic(6)); tB[k * 4 + 2] = float(ic(6)); }
        V3 d0; do { d0 = { float(ic(4)), float(ic(4)), float(ic(4)) }; } while (d0.x == 0 && d0.y == 0 && d0.z == 0);
        const float minSep = (shift == 2) ? 0.25f : (shift == 1) ? 1000.0f : 0.25f;
        // run the reference cold to get a genuine cache
        float coldZero[32]; for (uint32_t L = 0; L < 32; ++L) coldZero[L] = 0.0f;
        float coldSimplex[32]; const GjkOut cold = gjkWarmRef(d0, tA, tB, 0, nbA, nbB, minSep, 0.98f, 0u, coldZero, coldSimplex);
        if (nCold < 900) { pushOne(tA, tB, nbA, nbB, d0, minSep, 0u, coldZero); ++nCold; }
        // feed the cache back warm
        if (nWarm < 600 && cold.size >= 1 && cold.size <= 4)
        {
            pushOne(tA, tB, nbA, nbB, d0, minSep, cold.size, coldSimplex); ++nWarm;
            if (cold.ret == 2u) ++nWarmOverlap;
        }
    }

    const uint32_t N = (uint32_t)ref.size();
    PxgSlangComputeDevice* dev = testDevice();
    PxgBufferHandle bVA = createBufferFrom(dev, VA);
    PxgBufferHandle bVB = createBufferFrom(dev, VB);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bCache = createBufferFrom(dev, Cache);
    std::vector<GjkOut> outInit(N); std::memset(outInit.data(), 0xEE, outInit.size() * sizeof(GjkOut));
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    GjkParams prm = { N, MAXV, 0, 0 };
    Kernel k("npGjkWarm.spv", "ctGjkWarm", { 16, 16, (uint32_t)sizeof(GjkIn), 4, (uint32_t)sizeof(GjkOut) }, sizeof(GjkParams));
    k.dispatch((N + 63) / 64, { bVA, bVB, bIn, bCache, bOut }, prm);
    dev->flush();

    const auto gOut = downloadBuffer<GjkOut>(dev, bOut, N);
    auto ckv = [&](const float g[4], const float r[4]) { return asU32(g[0]) == asU32(r[0]) && asU32(g[1]) == asU32(r[1]) && asU32(g[2]) == asU32(r[2]); };
    size_t mm = 0;
    for (uint32_t t = 0; t < N; ++t)
    {
        const GjkOut& r = ref[t]; const GjkOut& g = gOut[t];
        if (g.ret != r.ret) { ++mm; continue; }
        if (g.degenerate != r.degenerate) ++mm;
        if (g.size != r.size) ++mm;
        if (asU32(g.lowerBound) != asU32(r.lowerBound)) ++mm;
        if (asU32(g.upperBound) != asU32(r.upperBound)) ++mm;
        if (r.ret == 1u) { if (!ckv(g.closestPointDir, r.closestPointDir)) ++mm; if (!ckv(g.closestPointA, r.closestPointA)) ++mm; if (!ckv(g.closestPointB, r.closestPointB)) ++mm; }
        if (r.ret != 2u) if (!ckv(g.direction, r.direction)) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "gjkWarm: %zu mismatches (N=%u)", mm, N);
    PXG_CHECK_MSG(nCold > 400 && nWarm > 300 && nWarmOverlap > 30, "gjkWarm coverage thin (cold=%zu warm=%zu warmOverlap=%zu)", nCold, nWarm, nWarmOverlap);

    for (PxgBufferHandle h : { bVA, bVB, bIn, bCache, bOut }) dev->destroyBuffer(h);
}

} // namespace
