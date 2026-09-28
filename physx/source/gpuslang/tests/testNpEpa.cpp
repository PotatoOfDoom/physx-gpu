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

// Verification of npEpa.slang: ctEpa — the full EPA driver (epa.cuh :388-526).
// EPA runs on the overlap simplex GJK produces, so each test first runs a
// bit-exact CPU transcription of the GJK driver (tranche 129) on a pair of
// overlapping integer convex hulls to obtain a genuine overlap GjkCachedData
// (simplex + size), then feeds that to the EPA shader.  The result enum, bounds,
// degenerate flag, direction and witness points are compared bit-exactly against
// an EPA CPU transcription that mirrors the shader line-for-line.  Integer hulls
// keep every dot/cross exact and magnitude/getNormalized/divisions correctly-
// rounded, so the whole EPA run is bit-for-bit reproducible.

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
static uint32_t asU32(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float asF32(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static const float MAXF = 3.4028234663852886e38f;
static const uint32_t DEG16 = 0xffffu;

static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static V3 sub3(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 absX(V3 a) { return { std::fabs(a.x), std::fabs(a.y), std::fabs(a.z) }; }
static V3 getNormalized(V3 v) { const float m = dot3(v, v); if (m > 0.0f) { const float r = 1.0f / std::sqrt(m); return { v.x * r, v.y * r, v.z * r }; } return { 0, 0, 0 }; }
static float mag3(V3 v) { return std::sqrt(dot3(v, v)); }
static float magSq3(V3 v) { return dot3(v, v); }
static uint32_t mk3(uint32_t a, uint32_t b, uint32_t c) { return (a & 0xffu) | ((b & 0xffu) << 8) | ((c & 0xffu) << 16); }
static uint32_t g3(uint32_t p, uint32_t i) { return (p >> (i * 8u)) & 0xffu; }
static uint32_t s3(uint32_t p, uint32_t i, uint32_t v) { return (p & ~(0xffu << (i * 8u))) | ((v & 0xffu) << (i * 8u)); }
static uint32_t fbl(uint32_t m) { if (!m) return 0xffffffffu; uint32_t i = 0; while (!((m >> i) & 1u)) ++i; return i; }
static uint32_t fbh(uint32_t m) { if (!m) return 0xffffffffu; int i = 31; while (!((m >> i) & 1u)) --i; return (uint32_t)i; }
static uint32_t popc(uint32_t m) { uint32_t c = 0; while (m) { c += m & 1u; m >>= 1; } return c; }

// ======================= GJK reference (tranche 129) =======================
static void gdotW(const float a[32], const float b[32], float o[32])
{
    float x[32]; for (uint32_t L = 0; L < 32; ++L) x[L] = a[L] * b[L];
    for (uint32_t L = 0; L < 32; ++L) { const float s1 = (L + 1 < 32) ? x[L + 1] : x[L]; const float s2 = (L + 2 < 32) ? x[L + 2] : x[L]; o[L] = x[L] + s1 + s2; }
}
static void gcrossW(const float a[32], const float b[32], const uint32_t c0[32], const uint32_t c1[32], float o[32]) { for (uint32_t L = 0; L < 32; ++L) o[L] = a[c0[L]] * b[c1[L]] - a[c1[L]] * b[c0[L]]; }
static void gsplatXW(const float a[32], float o[32]) { for (uint32_t L = 0; L < 32; ++L) o[L] = a[L & 28u]; }
static void gsplatVW(uint32_t index, const float a[32], float o[32]) { for (uint32_t L = 0; L < 32; ++L) o[L] = a[((index << 2) | (L & 3u)) & 31u]; }
static void gmagW(const float v[32], float cm[32]) { float dv[32]; gdotW(v, v, dv); for (uint32_t L = 0; L < 32; ++L) cm[L] = std::sqrt(dv[L & 28u]); }
static uint32_t gnext3(uint32_t x) { return x == 2u ? 0u : x + 1u; }
static uint32_t gprev3(uint32_t x) { return x == 0u ? 2u : x - 1u; }
static uint32_t gffs(uint32_t m) { if (!m) return 0; uint32_t i = 0; while (!((m >> i) & 1u)) ++i; return i + 1; }

static uint32_t closestPointTriW(float zV[32], const float cV[32], float cp[32])
{
    uint32_t c0[32], c1[32]; for (uint32_t L = 0; L < 32; ++L) { c0[L] = ((L & 28u) | gnext3(L & 3u)) & 31u; c1[L] = ((L & 28u) | gprev3(L & 3u)) & 31u; }
    for (uint32_t L = 0; L < 32; ++L) cp[L] = 0.0f;
    float eV[32]; for (uint32_t L = 0; L < 32; ++L) eV[L] = cV[L] - zV[L];
    float dee[32]; gdotW(eV, eV, dee); float lD[32]; gsplatXW(dee, lD);
    if ((lD[0] == 0.0f) || (lD[4] == 0.0f)) return DEG16;
    float se[32]; for (uint32_t L = 0; L < 32; ++L) se[L] = (lD[L ^ 4u] < lD[L]) ? eV[L ^ 4u] : eV[L];
    float d1[32]; for (uint32_t L = 0; L < 32; ++L) d1[L] = zV[L ^ 4u] - zV[L];
    float mV[32]; gcrossW(d1, se, c0, c1, mV);
    float pS[32]; gdotW(eV, cV, pS);
    float cme[32]; gcrossW(mV, eV, c0, c1, cme); float dcme[32]; gdotW(cme, cV, dcme);
    bool uS[32]; for (uint32_t L = 0; L < 32; ++L) uS[L] = dcme[L] <= 0.0f;
    if (!(uS[0] || uS[4]))
    {
        float dmc[32]; gdotW(mV, cV, dmc); const float projD = dmc[0];
        float dmm[32]; gdotW(mV, mV, dmm); float sx[32]; gsplatXW(dmm, sx);
        float tmp[32]; for (uint32_t L = 0; L < 32; ++L) tmp[L] = mV[L] * projD / sx[L]; gsplatVW(0u, tmp, cp);
        const bool flip = projD < 0.0f; for (uint32_t L = 0; L < 32; ++L) if ((L & 3u) == 3u) mV[L] = projD;
        const uint32_t idx = flip ? 1u : 0u; for (uint32_t L = 0; L < 32; ++L) { const uint32_t src = ((idx << 2) | (L & 3u)) & 31u; zV[L] = (L < 16u || L > 19u) ? zV[L] : mV[src]; }
        return flip ? 0x0123u : 0x1023u;
    }
    uint32_t em = 0u; if (uS[0] && pS[0] > 0.0f) em |= 1u; if (uS[4] && pS[4] > 0.0f) em |= 2u;
    uint32_t eid = gffs(em);
    if (eid != 0u) { eid -= 1u; float sp[32]; gsplatXW(pS, sp); float tmp[32]; for (uint32_t L = 0; L < 32; ++L) tmp[L] = cV[L] - eV[L] * sp[L] / lD[L]; gsplatVW(eid, tmp, cp); return (eid << 8) | 0x22u; }
    for (uint32_t L = 0; L < 32; ++L) cp[L] = cV[L]; return 0x0021u;
}
static uint32_t closestPointTetW(float zV[32], const float dV[32], float cp[32])
{
    for (uint32_t L = 0; L < 32; ++L) cp[L] = 0.0f;
    float n0[32]; gsplatVW(4u, zV, n0); float dn0d[32]; gdotW(n0, dV, dn0d); float dn0z[32]; gdotW(n0, zV, dn0z);
    if (dn0d[0] > dn0z[0]) return DEG16;
    uint32_t c0[32], c1[32], vN[32], vP[32];
    for (uint32_t L = 0; L < 32; ++L) { c0[L] = ((L & 28u) | gnext3(L & 3u)) & 31u; c1[L] = ((L & 28u) | gprev3(L & 3u)) & 31u; vN[L] = (L > 7u) ? (L - 8u) : (L + 4u); vP[L] = (L < 4u) ? (L + 8u) : (L - 4u); }
    float eV[32]; for (uint32_t L = 0; L < 32; ++L) eV[L] = dV[L] - zV[L];
    float dee[32]; gdotW(eV, eV, dee); float lD[32]; gsplatXW(dee, lD);
    if ((lD[0] == 0.0f) || (lD[4] == 0.0f) || (lD[8] == 0.0f)) return DEG16;
    float se[32]; for (uint32_t L = 0; L < 32; ++L) se[L] = (lD[vN[L]] < lD[L]) ? eV[vN[L]] : eV[L];
    float d2[32]; for (uint32_t L = 0; L < 32; ++L) d2[L] = zV[vN[L]] - zV[L];
    float nV[32]; gcrossW(se, d2, c0, c1, nV);
    float aS[32]; gdotW(nV, dV, aS); float zvp[32]; for (uint32_t L = 0; L < 32; ++L) zvp[L] = zV[vP[L]]; float gS[32]; gdotW(zvp, nV, gS);
    if ((aS[0] <= gS[0]) || (aS[4] <= gS[4]) || (aS[8] <= gS[8])) return DEG16;
    for (uint32_t L = 0; L < 32; ++L) if ((L & 3u) == 3u) nV[L] = aS[L - 3u];
    if (aS[0] > 0.0f && aS[4] > 0.0f && aS[8] > 0.0f) { for (uint32_t L = 0; L < 32; ++L) if (L >= 20u) zV[L] = nV[L - 20u]; return 0x2104u; }
    float pS[32]; gdotW(eV, dV, pS);
    if (pS[0] <= 0.0f && pS[4] <= 0.0f && pS[8] <= 0.0f) { for (uint32_t L = 0; L < 32; ++L) cp[L] = dV[L]; return 0x2131u; }
    float cen[32]; gcrossW(eV, nV, c0, c1, cen); float dcen[32]; gdotW(cen, dV, dcen); bool uiS[32]; for (uint32_t L = 0; L < 32; ++L) uiS[L] = dcen[L] <= 0.0f;
    float evn[32]; for (uint32_t L = 0; L < 32; ++L) evn[L] = eV[vN[L]]; float cevn[32]; gcrossW(evn, nV, c0, c1, cevn); float dcevn[32]; gdotW(cevn, dV, dcevn); bool ujS[32]; for (uint32_t L = 0; L < 32; ++L) ujS[L] = dcevn[L] > 0.0f;
    uint32_t em = 0u;
    if (uiS[0] && ujS[vP[0]] && pS[0] >= 0.0f) em |= 1u;
    if (uiS[4] && ujS[vP[4]] && pS[4] >= 0.0f) em |= 2u;
    if (uiS[8] && ujS[vP[8]] && pS[8] >= 0.0f) em |= 4u;
    uint32_t eid = gffs(em);
    if (eid != 0u) { eid -= 1u; float pl[32]; for (uint32_t L = 0; L < 32; ++L) pl[L] = pS[L] / lD[L]; float sp[32]; gsplatXW(pl, sp); float tmp[32]; for (uint32_t L = 0; L < 32; ++L) tmp[L] = dV[L] - eV[L] * sp[L]; gsplatVW(eid, tmp, cp); return (eid << 8) | 0x32u; }
    uint32_t fm = 0u;
    if (!uiS[0] && !ujS[0] && aS[0] <= 0.0f) fm |= 1u;
    if (!uiS[4] && !ujS[4] && aS[4] <= 0.0f) fm |= 2u;
    if (!uiS[8] && !ujS[8] && aS[8] <= 0.0f) fm |= 4u;
    uint32_t fid = gffs(fm);
    if (fid != 0u) { fid -= 1u; float dnn[32]; gdotW(nV, nV, dnn); float ad[32]; for (uint32_t L = 0; L < 32; ++L) ad[L] = aS[L] / dnn[L]; float sad[32]; gsplatXW(ad, sad); float tmp[32]; for (uint32_t L = 0; L < 32; ++L) tmp[L] = nV[L] * sad[L]; gsplatVW(fid, tmp, cp); for (uint32_t L = 0; L < 32; ++L) { const uint32_t src = ((fid << 2) | (L & 3u)) & 31u; zV[L] = (L < 16u || L > 19u) ? zV[L] : -nV[src]; } return (gnext3(fid) << 12) | (fid << 8) | 0x33u; }
    return DEG16;
}
static uint32_t gSupport(V3 dir, const std::vector<float>& V, uint32_t vbase, uint32_t nb)
{
    uint32_t bestN = 0xFFFFFFFFu; float bestDist = -MAXF;
    for (uint32_t i = 0; i < nb; i += 32) { float d[32]; for (uint32_t l = 0; l < 32; ++l) { const uint32_t n = i + l; V3 v = { V[(vbase + n) * 4], V[(vbase + n) * 4 + 1], V[(vbase + n) * 4 + 2] }; d[l] = (n < nb) ? dot3(dir, v) : -MAXF; } float cm = -MAXF; for (uint32_t l = 0; l < 32; ++l) cm = std::fmax(cm, d[l]); if (cm > bestDist) { uint32_t ll = 0; for (uint32_t l = 0; l < 32; ++l) if (d[l] == cm) { ll = l; break; } bestDist = cm; bestN = i + ll; } }
    return bestN;
}
// Runs the GJK cold-start driver; returns the result enum and fills the final simplex + size.
static uint32_t gjkRun(V3 initialDir, const std::vector<float>& VA, const std::vector<float>& VB, uint32_t vbase, uint32_t nbA, uint32_t nbB, float minSep, float convRatio, float simplexOut[32], int& sizeOut)
{
    float simplexV[32]; for (uint32_t L = 0; L < 32; ++L) simplexV[L] = 0.0f;
    float cpv[32]; for (uint32_t L = 0; L < 32; ++L) { const uint32_t c = L & 3u; cpv[L] = (c == 0) ? initialDir.x : (c == 1) ? initialDir.y : (c == 2) ? initialDir.z : MAXF; }
    float upperBound = MAXF, lowerBound = -MAXF; float cmA[32]; gmagW(cpv, cmA);
    int size = 0, backupSize = 0; float bkS[32], bkC[32]; for (uint32_t L = 0; L < 32; ++L) { bkS[L] = 0.0f; bkC[L] = 0.0f; }
    bool run = true;
    while (run)
    {
        float dV[32]; for (uint32_t L = 0; L < 32; ++L) dV[L] = cpv[L] / cmA[L];
        V3 dv = { dV[0], dV[1], dV[2] };
        const uint32_t iA = gSupport({ -dv.x, -dv.y, -dv.z }, VA, vbase, nbA), iB = gSupport(dv, VB, vbase, nbB);
        V3 vA = { VA[(vbase + iA) * 4], VA[(vbase + iA) * 4 + 1], VA[(vbase + iA) * 4 + 2] };
        V3 vB = { VB[(vbase + iB) * 4], VB[(vbase + iB) * 4 + 1], VB[(vbase + iB) * 4 + 2] };
        V3 sv = sub3(vA, vB);
        float sup[32]; for (uint32_t L = 0; L < 32; ++L) { const uint32_t c = L & 3u; sup[L] = (c == 0) ? sv.x : (c == 1) ? sv.y : (c == 2) ? sv.z : 0.0f; }
        const float enc = asF32(0x3f800000u | ((iB & 0xffu) << 8) | (iA & 0xffu));
        for (uint32_t L = 0; L < 32; ++L) if ((L >> 2) == (uint32_t)size) simplexV[L] = ((L & 3u) < 3u) ? sup[L] : enc;
        float dds[32]; gdotW(dV, sup, dds); const float lb = dds[0];
        if (lb > lowerBound) { lowerBound = lb; if (lowerBound > minSep || lowerBound >= upperBound * convRatio) { run = false; break; } }
        uint32_t f = 0u; float pv[32];
        if (size == 0) { f = 0x2101u; for (uint32_t L = 0; L < 32; ++L) pv[L] = sup[L]; }
        else if (size == 1) { float aV[32]; gsplatVW(0u, simplexV, aV); float lV[32]; for (uint32_t L = 0; L < 32; ++L) lV[L] = sup[L] - aV[L]; float dll[32]; gdotW(lV, lV, dll); float mm[32]; gsplatXW(dll, mm); float dal[32]; gdotW(aV, lV, dal); float xx[32]; gsplatXW(dal, xx); for (uint32_t L = 0; L < 32; ++L) xx[L] = -xx[L]; if (mm[0] == 0.0f) { f = DEG16; for (uint32_t L = 0; L < 32; ++L) pv[L] = 0.0f; } else if (xx[0] < mm[0]) { f = 0x2102u; for (uint32_t L = 0; L < 32; ++L) pv[L] = aV[L] + lV[L] * xx[L] / mm[L]; } else { f = 0x2111u; for (uint32_t L = 0; L < 32; ++L) pv[L] = sup[L]; } }
        else if (size == 2) f = closestPointTriW(simplexV, sup, pv);
        else f = closestPointTetW(simplexV, sup, pv);
        gmagW(pv, cmA); const int newSize = (int)(f & 0xfu);
        const bool degenerate = (f == DEG16) || (cmA[0] >= upperBound && newSize <= size);
        if (degenerate) { run = false; break; }
        backupSize = (cmA[0] >= upperBound) ? size : 0;
        if (backupSize != 0) { for (uint32_t L = 0; L < 32; ++L) { bkS[L] = simplexV[L]; bkC[L] = cpv[L]; } } else upperBound = cmA[0];
        for (uint32_t L = 0; L < 32; ++L) cpv[L] = pv[L]; size = newSize;
        float tp[32]; for (uint32_t L = 0; L < 32; ++L) { const uint32_t si = (f >> ((L & 28u) + 4u)) & 0xfu; tp[L] = simplexV[((si << 2) | (L & 3u)) & 31u]; }
        for (uint32_t L = 0; L < 32; ++L) if (L < 12u) simplexV[L] = tp[L];
        if (!(upperBound > 0.0f && lowerBound < upperBound * convRatio)) run = false;
    }
    if (backupSize > 0) { for (uint32_t L = 0; L < 32; ++L) { simplexV[L] = bkS[L]; cpv[L] = bkC[L]; } size = backupSize; }
    for (uint32_t L = 0; L < 32; ++L) simplexOut[L] = simplexV[L];
    sizeOut = size;
    if (upperBound == 0.0f) return 2u; // OVERLAP
    if (lowerBound > minSep) return 0u; // DISTANT
    return 1u; // CLOSE
}

// ======================= EPA reference (mirrors npEpa.slang) =======================
struct Scratch { V3 verts[18]; uint32_t idx[18], nxt[18], edg[18], lp[18]; V3 bestDir; };
struct Lanes { V3 normal[32]; float dist[32]; uint32_t vv[32], aa[32]; };

static uint32_t supA(V3 dir, const std::vector<float>& V, uint32_t vbase, uint32_t nb) { return gSupport(dir, V, vbase, nb); }
static void newPointR(V3 dir, V3 ref, const std::vector<float>& VA, const std::vector<float>& VB, uint32_t vbase, uint32_t nbA, uint32_t nbB, V3& vertex, uint32_t& indices)
{
    auto va = [&](uint32_t i) -> V3 { return { VA[(vbase + i) * 4], VA[(vbase + i) * 4 + 1], VA[(vbase + i) * 4 + 2] }; };
    auto vb = [&](uint32_t i) -> V3 { return { VB[(vbase + i) * 4], VB[(vbase + i) * 4 + 1], VB[(vbase + i) * 4 + 2] }; };
    V3 nd = { -dir.x, -dir.y, -dir.z };
    const uint32_t iAp = supA(dir, VA, vbase, nbA), iBp = supA(nd, VB, vbase, nbB), iAn = supA(nd, VA, vbase, nbA), iBn = supA(dir, VB, vbase, nbB);
    const V3 vp = sub3(va(iAp), vb(iBp)), vn = sub3(va(iAn), vb(iBn));
    if (std::fabs(dot3(sub3(vp, ref), dir)) > std::fabs(dot3(sub3(vn, ref), dir))) { vertex = vp; indices = ((iBp << 8) | (iAp & 0xffu)) & 0xffffu; }
    else { vertex = vn; indices = ((iBn << 8) | (iAn & 0xffu)) & 0xffffu; }
}
static void barycentricsR(V3 a, V3 b, V3 c, V3 p, float& u, float& v, float& w)
{
    const V3 m = cross3(sub3(b, a), sub3(c, a)); const V3 ma = absX(m);
    const V3 a1 = cross3(sub3(b, p), sub3(c, p)), a2 = cross3(sub3(c, p), sub3(a, p));
    float d;
    if (ma.x >= ma.y && ma.x >= ma.z) { d = 1.0f / m.x; u = a1.x * d; v = a2.x * d; }
    else if (ma.y >= ma.z) { d = 1.0f / m.y; u = a1.y * d; v = a2.y * d; }
    else { d = 1.0f / m.z; u = a1.z * d; v = a2.z * d; }
    w = 1.0f - u - v;
}
static void epaStart4R(const float input[32], Scratch& S, Lanes& L)
{
    for (uint32_t tI = 0; tI < 16; ++tI) { if ((tI & 3u) < 3u) { const uint32_t fi = tI - tI / 4u; const uint32_t vk = fi / 3u, cc = fi % 3u; if (cc == 0) S.verts[vk].x = input[tI]; else if (cc == 1) S.verts[vk].y = input[tI]; else S.verts[vk].z = input[tI]; } else S.idx[tI >> 2] = asU32(input[tI]) & 0xffffu; }
    float d[32]; { float x[32]; for (uint32_t l = 0; l < 32; ++l) x[l] = input[l] * input[l]; for (uint32_t l = 0; l < 32; ++l) { const float s1 = (l + 1 < 32) ? x[l + 1] : x[l]; const float s2 = (l + 2 < 32) ? x[l + 2] : x[l]; d[l] = x[l] + s1 + s2; } }
    for (uint32_t tI = 0; tI < 4; ++tI) { const float m = 1.0f / std::sqrt(d[tI * 4 + 16]); L.normal[tI] = { input[tI * 4 + 16] * m, input[tI * 4 + 17] * m, input[tI * 4 + 18] * m }; L.dist[tI] = input[tI * 4 + 19] * m; }
    L.aa[0] = 0x010302u; L.aa[1] = 0x020300u; L.aa[2] = 0x030100u; L.aa[3] = 0x010200u;
    L.vv[0] = 0x020100u; L.vv[1] = 0x000103u; L.vv[2] = 0x010203u; L.vv[3] = 0x020003u;
}
static void epaStart123R(const float input[32], uint32_t size, const std::vector<float>& VA, const std::vector<float>& VB, uint32_t vbase, uint32_t nbA, uint32_t nbB, Scratch& S, Lanes& L)
{
    float verts[12]; uint32_t indices[4];
    for (uint32_t tI = 0; tI < 16; ++tI) { if ((tI & 3u) < 3u) verts[tI - tI / 4u] = input[tI]; else indices[tI >> 2] = asU32(input[tI]) & 0xffffu; }
    V3 v0 = { verts[0], verts[1], verts[2] }, v1 = { verts[3], verts[4], verts[5] };
    if (size < 2u) { V3 nv; uint32_t ni; newPointR({ 1, 0, 0 }, v0, VA, VB, vbase, nbA, nbB, nv, ni); v1 = nv; verts[3] = nv.x; verts[4] = nv.y; verts[5] = nv.z; indices[1] = ni; }
    V3 v2 = { verts[6], verts[7], verts[8] };
    if (size < 3u) { V3 dir = sub3(v1, v0); V3 axis = (std::fabs(dir.x) < std::fabs(dir.y)) ? V3{ 1, 0, 0 } : V3{ 0, 1, 0 }; V3 nv; uint32_t ni; newPointR(cross3(axis, dir), v0, VA, VB, vbase, nbA, nbB, nv, ni); v2 = nv; verts[6] = nv.x; verts[7] = nv.y; verts[8] = nv.z; indices[2] = ni; }
    S.verts[0] = { verts[0], verts[1], verts[2] }; S.verts[1] = { verts[3], verts[4], verts[5] }; S.verts[2] = { verts[6], verts[7], verts[8] };
    S.idx[0] = indices[0]; S.idx[1] = indices[1]; S.idx[2] = indices[2];
    V3 e01 = sub3(v1, v0), e12 = sub3(v2, v1), e02 = sub3(v2, v0);
    V3 normal = (mag3(e01) <= mag3(e02)) ? getNormalized(cross3(e01, e12)) : getNormalized(cross3(e02, e12));
    const float distance = dot3(normal, v0);
    L.normal[0] = normal; L.dist[0] = distance; L.vv[0] = 0x020100u; L.aa[0] = 0x010101u;
    L.normal[1] = { -normal.x, -normal.y, -normal.z }; L.dist[1] = -distance; L.vv[1] = 0x020001u; L.aa[1] = 0x000000u;
}
static bool replaceTriangleR(Scratch& S, Lanes& L, int nbVertices, int triIndex, float lowerBound, uint32_t& liveB, float releps)
{
    const V3 apex = S.verts[nbVertices - 1];
    bool threatened[32]; uint32_t threatenedB = 0u;
    for (uint32_t tI = 0; tI < 32; ++tI) { const bool th = ((liveB & (1u << tI)) != 0u) && (dot3(L.normal[tI], apex) - L.dist[tI] >= releps); threatened[tI] = th; if (th) threatenedB |= (1u << tI); }
    if (threatenedB == 0u || threatenedB == liveB || ((threatenedB & (1u << (uint32_t)triIndex)) == 0u)) return false;
    uint32_t b0[32], b1[32], b2[32], nbr[32];
    for (uint32_t tI = 0; tI < 32; ++tI) { b0[tI] = 1u << g3(L.aa[tI], 0u); b1[tI] = 1u << g3(L.aa[tI], 1u); b2[tI] = 1u << g3(L.aa[tI], 2u); nbr[tI] = (b0[tI] | b1[tI] | b2[tI]) & (threatened[tI] ? threatenedB : 0u); }
    uint32_t killB = 1u << (uint32_t)triIndex;
    for (uint32_t pB = 0u; pB != killB;) { pB = killB; uint32_t bal = 0u; for (uint32_t tI = 0; tI < 32; ++tI) if ((nbr[tI] & killB) != 0u) bal |= (1u << tI); killB |= bal; }
    const uint32_t deadB = (~liveB) | killB;
    bool d0[32], d1[32], d2[32]; uint32_t v0a[32], v1a[32], v2a[32], zz[32];
    for (uint32_t tI = 0; tI < 32; ++tI) { const uint32_t laneB = 1u << tI; const bool alive = (laneB & ~deadB) != 0u; d0[tI] = alive && ((b0[tI] & deadB) != 0u); d1[tI] = alive && ((b1[tI] & deadB) != 0u); d2[tI] = alive && ((b2[tI] & deadB) != 0u); v0a[tI] = g3(L.vv[tI], 0u); v1a[tI] = g3(L.vv[tI], 1u); v2a[tI] = g3(L.vv[tI], 2u); zz[tI] = 0u; }
    for (uint32_t tI = 0; tI < 32; ++tI) { if (d0[tI]) { zz[tI] = v2a[tI]; S.nxt[v2a[tI]] = v1a[tI]; S.edg[v2a[tI]] = tI; } if (d1[tI]) { zz[tI] = v0a[tI]; S.nxt[v0a[tI]] = v2a[tI]; S.edg[v0a[tI]] = tI; } if (d2[tI]) { zz[tI] = v1a[tI]; S.nxt[v1a[tI]] = v0a[tI]; S.edg[v1a[tI]] = tI; } }
    uint32_t dB = 0u; for (uint32_t tI = 0; tI < 32; ++tI) if (d0[tI] || d1[tI] || d2[tI]) dB |= (1u << tI);
    if (dB == 0u) return false;
    const uint32_t loopStart = zz[fbl(dB)];
    int nbLoop = 0; S.lp[nbLoop++] = loopStart;
    { uint32_t vertex = S.nxt[loopStart]; while (vertex != loopStart) { if (vertex >= 18u) { nbLoop = -1; break; } if (nbLoop >= 17) { nbLoop = -1; break; } S.lp[nbLoop++] = vertex; vertex = S.nxt[vertex]; } }
    uint32_t bd0 = 0u, bd1 = 0u, bd2 = 0u; for (uint32_t tI = 0; tI < 32; ++tI) { if (d0[tI]) bd0 |= (1u << tI); if (d1[tI]) bd1 |= (1u << tI); if (d2[tI]) bd2 |= (1u << tI); }
    if (nbLoop != (int)(popc(bd0) + popc(bd1) + popc(bd2))) return false;
    uint32_t coneIndex[32]; uint32_t coneB = 0u;
    for (uint32_t tI = 0; tI < 32; ++tI) { const uint32_t laneB = 1u << tI; coneIndex[tI] = popc(deadB & (laneB - 1u)); if (((deadB & laneB) != 0u) && ((int)coneIndex[tI] < nbLoop)) coneB |= (1u << tI); }
    bool fail[32]; for (uint32_t tI = 0; tI < 32; ++tI) fail[tI] = false;
    for (uint32_t tI = 0; tI < 32; ++tI)
    {
        const uint32_t laneB = 1u << tI;
        if ((coneB & laneB) != 0u)
        {
            const uint32_t v0c = (uint32_t)(nbVertices - 1), v1c = S.lp[coneIndex[tI]], v2c = S.lp[(coneIndex[tI] + 1u == (uint32_t)nbLoop) ? 0u : coneIndex[tI] + 1u];
            L.vv[tI] = mk3(v0c, v1c, v2c);
            L.aa[tI] = s3(L.aa[tI], 0u, S.edg[v1c]);
            const uint32_t mask1 = (coneIndex[tI] + 1u == (uint32_t)nbLoop) ? 0u : (2u * laneB - 1u);
            L.aa[tI] = s3(L.aa[tI], 1u, fbl(coneB & ~mask1));
            const uint32_t mask2 = ((coneIndex[tI] == 0u) ? 0u : laneB) - 1u;
            L.aa[tI] = s3(L.aa[tI], 2u, fbh(coneB & mask2));
            S.edg[v1c] = tI;
            const V3 p0 = S.verts[v0c], p1 = S.verts[v1c], p2 = S.verts[v2c];
            const V3 e01 = sub3(p1, p0), e02 = sub3(p2, p0);
            V3 n = (magSq3(e01) < magSq3(e02)) ? cross3(e01, sub3(p2, p1)) : cross3(e02, sub3(p2, p1));
            const float nm = mag3(n); const float inv = 1.0f / nm;
            L.normal[tI] = { n.x * inv, n.y * inv, n.z * inv };
            L.dist[tI] = dot3(L.normal[tI], p0);
            fail[tI] = (nm == 0.0f) || ((lowerBound > 0.0f) && ((L.dist[tI] - lowerBound) < -1e-4f));
        }
    }
    for (uint32_t tI = 0; tI < 32; ++tI) { if (d0[tI]) L.aa[tI] = s3(L.aa[tI], 0u, S.edg[g3(L.vv[tI], 2u)]); if (d1[tI]) L.aa[tI] = s3(L.aa[tI], 1u, S.edg[g3(L.vv[tI], 0u)]); if (d2[tI]) L.aa[tI] = s3(L.aa[tI], 2u, S.edg[g3(L.vv[tI], 1u)]); }
    liveB = (liveB & ~deadB) | coneB;
    uint32_t failB = 0u; for (uint32_t tI = 0; tI < 32; ++tI) if (fail[tI]) failB |= (1u << tI);
    return failB == 0u;
}
// CUDA epa.cuh:427-433 reduces the lane distances with an fminf butterfly.  NVIDIA fminf is
// minNum with -0 < +0: NaN-ignoring, and a +-0 tie gives -0 in either operand order (std::fmin
// leaves the zero sign to libm; glibc x86-64 lets the second operand win).  gEpaMinVariant
// selects mutant twins for npEpa_driverZeroTie: 1 = second operand wins a tie (x86 minps(e,tv)),
// 2 = first operand wins a tie (GLSL FMin(e,tv) read literally).
static float nvFminf(float a, float b)
{
    if (std::isnan(a)) return b;
    if (std::isnan(b)) return a;
    if (a == b) return std::signbit(a) ? a : b;
    return (a < b) ? a : b;
}
static int gEpaMinVariant = 0;
static float epaMinStep(float e, float tv)
{
    switch (gEpaMinVariant)
    {
    case 1: return (e < tv) ? e : tv;
    case 2: return (tv < e) ? tv : e;
    default: return nvFminf(e, tv);
    }
}
struct EpaOut { uint32_t result, degenerate, p0, p1; float lowerBound, upperBound, q0, q1; float direction[4], closestPointA[4], closestPointB[4]; };
static EpaOut epaRef(const float input[32], uint32_t size, const std::vector<float>& VA, const std::vector<float>& VB, uint32_t vbase, uint32_t nbA, uint32_t nbB, float convRatio, float relEpsBase)
{
    const float releps = relEpsBase * (1.0f - convRatio);
    Scratch S; for (uint32_t i = 0; i < 18; ++i) { S.verts[i] = { 0, 0, 0 }; S.idx[i] = S.nxt[i] = S.edg[i] = S.lp[i] = 0u; } S.bestDir = { 0, 0, 0 };
    Lanes L; for (uint32_t i = 0; i < 32; ++i) { L.normal[i] = { 0, 0, 0 }; L.dist[i] = 0.0f; L.vv[i] = L.aa[i] = 0u; }
    float lowerBound = -MAXF, upperBound = MAXF; uint32_t bestVerts = 0u; int nbVertices; uint32_t liveTriangles;
    if (size == 4u) { epaStart4R(input, S, L); nbVertices = 4; liveTriangles = 0xfu; }
    else { epaStart123R(input, size, VA, VB, vbase, nbA, nbB, S, L); nbVertices = 3; liveTriangles = 0x3u; }
    bool nondegenerate = true;
    while (nondegenerate && (uint32_t)nbVertices < 18u)
    {
        float e = MAXF; for (uint32_t tI = 0; tI < 32; ++tI) { const float tv = ((liveTriangles & (1u << tI)) != 0u) ? L.dist[tI] : MAXF; e = epaMinStep(e, tv); }
        uint32_t f = 0u; for (uint32_t tI = 0; tI < 32; ++tI) { const float tv = ((liveTriangles & (1u << tI)) != 0u) ? L.dist[tI] : MAXF; if (e == tv) f |= (1u << tI); }
        const uint32_t triIndex = fbl(f); const V3 d = L.normal[triIndex];
        const uint32_t iA = supA(d, VA, vbase, nbA), iB = supA({ -d.x, -d.y, -d.z }, VB, vbase, nbB);
        const V3 supportVertex = sub3({ VA[(vbase + iA) * 4], VA[(vbase + iA) * 4 + 1], VA[(vbase + iA) * 4 + 2] }, { VB[(vbase + iB) * 4], VB[(vbase + iB) * 4 + 1], VB[(vbase + iB) * 4 + 2] });
        const float thisFrameUb = dot3(supportVertex, d);
        if (std::fabs(thisFrameUb - e) <= releps) { upperBound = thisFrameUb; lowerBound = e; bestVerts = L.vv[triIndex]; S.bestDir = d; break; }
        if (e > upperBound) break;
        const float ub = std::fmin(upperBound, thisFrameUb);
        if (ub < upperBound) { upperBound = ub; lowerBound = e; bestVerts = L.vv[triIndex]; S.bestDir = d; }
        S.verts[nbVertices] = supportVertex; S.idx[nbVertices] = ((iB << 8) | (iA & 0xffu)) & 0xffffu; nbVertices += 1;
        nondegenerate = replaceTriangleR(S, L, nbVertices, (int)triIndex, e, liveTriangles, releps);
    }
    EpaOut o; std::memset(&o, 0, sizeof(o));
    o.degenerate = nondegenerate ? 0u : 1u; o.lowerBound = lowerBound; o.upperBound = upperBound;
    const V3 bestDir = S.bestDir; const V3 closestPoint = { bestDir.x * lowerBound, bestDir.y * lowerBound, bestDir.z * lowerBound };
    o.direction[0] = bestDir.x; o.direction[1] = bestDir.y; o.direction[2] = bestDir.z;
    const uint32_t v0 = g3(bestVerts, 0u), v1 = g3(bestVerts, 1u), v2 = g3(bestVerts, 2u);
    float u, vv, ww; barycentricsR(S.verts[v0], S.verts[v1], S.verts[v2], closestPoint, u, vv, ww);
    auto va = [&](uint32_t i) -> V3 { return { VA[(vbase + i) * 4], VA[(vbase + i) * 4 + 1], VA[(vbase + i) * 4 + 2] }; };
    const V3 p0 = va(S.idx[v0] & 0xffu), p1 = va(S.idx[v1] & 0xffu), p2 = va(S.idx[v2] & 0xffu);
    const V3 cpA = { p0.x * u + p1.x * vv + p2.x * ww, p0.y * u + p1.y * vv + p2.y * ww, p0.z * u + p1.z * vv + p2.z * ww };
    o.closestPointA[0] = cpA.x; o.closestPointA[1] = cpA.y; o.closestPointA[2] = cpA.z;
    o.closestPointB[0] = cpA.x - closestPoint.x; o.closestPointB[1] = cpA.y - closestPoint.y; o.closestPointB[2] = cpA.z - closestPoint.z;
    o.result = (lowerBound < 0.0f) ? 1u : 2u;
    return o;
}

struct EpaParams { uint32_t numTests, maxV, pad0, pad1; };
struct EpaScalars { uint32_t size, nbA, nbB, pad; float convergenceRatio, relEpsBase, p0, p1; };

PXG_TEST(npEpa_driver)
{
    std::mt19937 rng(1320132);
    const uint32_t MAXV = 32;
    std::vector<float> Data, VA, VB;
    std::vector<EpaScalars> S;
    std::vector<EpaOut> ref;
    size_t nOverlap = 0, nSize4 = 0, nSize3 = 0;

    auto ic = [&](int r) -> int { return int(rng() % (2 * r + 1)) - r; };

    uint32_t attempts = 0;
    while (nOverlap < 800 && attempts < 20000)
    {
        ++attempts;
        // build a fresh overlapping pair in a scratch cloud
        std::vector<float> tA(MAXV * 4, 0.0f), tB(MAXV * 4, 0.0f);
        const uint32_t nbA = 6 + rng() % 20, nbB = 6 + rng() % 20;
        for (uint32_t k = 0; k < nbA; ++k) { tA[k * 4] = float(ic(6)); tA[k * 4 + 1] = float(ic(6)); tA[k * 4 + 2] = float(ic(6)); }
        for (uint32_t k = 0; k < nbB; ++k) { tB[k * 4] = float(ic(6) + ic(2)); tB[k * 4 + 1] = float(ic(6) + ic(2)); tB[k * 4 + 2] = float(ic(6) + ic(2)); }
        V3 d0; do { d0 = { float(ic(4)), float(ic(4)), float(ic(4)) }; } while (d0.x == 0 && d0.y == 0 && d0.z == 0);
        float simplex[32]; int size;
        const uint32_t r = gjkRun(d0, tA, tB, 0, nbA, nbB, 0.25f, 0.98f, simplex, size);
        if (r != 2u) continue; // only OVERLAP produces a valid EPA seed
        // commit the clouds + simplex into the aggregate buffers
        const uint32_t vbase = (uint32_t)(VA.size() / 4);
        for (uint32_t k = 0; k < MAXV; ++k) { VA.push_back(tA[k * 4]); VA.push_back(tA[k * 4 + 1]); VA.push_back(tA[k * 4 + 2]); VA.push_back(0); VB.push_back(tB[k * 4]); VB.push_back(tB[k * 4 + 1]); VB.push_back(tB[k * 4 + 2]); VB.push_back(0); }
        for (uint32_t L = 0; L < 32; ++L) Data.push_back(simplex[L]);
        EpaScalars sc{}; sc.size = (uint32_t)size; sc.nbA = nbA; sc.nbB = nbB; sc.convergenceRatio = 0.98f; sc.relEpsBase = 4.0f;
        S.push_back(sc);
        ref.push_back(epaRef(simplex, (uint32_t)size, VA, VB, vbase, nbA, nbB, 0.98f, 4.0f));
        ++nOverlap; if (size == 4) ++nSize4; else if (size == 3) ++nSize3;
    }

    const uint32_t N = (uint32_t)ref.size();
    PxgSlangComputeDevice* dev = testDevice();
    PxgBufferHandle bData = createBufferFrom(dev, Data);
    PxgBufferHandle bVA = createBufferFrom(dev, VA);
    PxgBufferHandle bVB = createBufferFrom(dev, VB);
    PxgBufferHandle bS = createBufferFrom(dev, S);
    std::vector<EpaOut> outInit(N); std::memset(outInit.data(), 0xEE, outInit.size() * sizeof(EpaOut));
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    EpaParams prm = { N, MAXV, 0, 0 };
    Kernel k("npEpa.spv", "ctEpa", { 4, 16, 16, (uint32_t)sizeof(EpaScalars), (uint32_t)sizeof(EpaOut) }, sizeof(EpaParams));
    k.dispatch((N + 63) / 64, { bData, bVA, bVB, bS, bOut }, prm);
    dev->flush();

    const auto gOut = downloadBuffer<EpaOut>(dev, bOut, N);
    auto ckv = [&](const float g[4], const float r[4]) { return asU32(g[0]) == asU32(r[0]) && asU32(g[1]) == asU32(r[1]) && asU32(g[2]) == asU32(r[2]); };
    size_t mm = 0;
    for (uint32_t t = 0; t < N; ++t)
    {
        const EpaOut& r = ref[t]; const EpaOut& g = gOut[t];
        if (g.result != r.result) ++mm;
        if (g.degenerate != r.degenerate) ++mm;
        if (asU32(g.lowerBound) != asU32(r.lowerBound)) ++mm;
        if (asU32(g.upperBound) != asU32(r.upperBound)) ++mm;
        if (!ckv(g.direction, r.direction)) ++mm;
        if (!ckv(g.closestPointA, r.closestPointA)) ++mm;
        if (!ckv(g.closestPointB, r.closestPointB)) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "epa driver: %zu mismatches (N=%u)", mm, N);
    PXG_CHECK_MSG(nOverlap > 300 && nSize4 > 100, "epa driver coverage thin (overlap=%zu size4=%zu size3=%zu)", nOverlap, nSize4, nSize3);

    for (PxgBufferHandle h : { bData, bVA, bVB, bS, bOut }) dev->destroyBuffer(h);
}


// P9.86c: signed-zero tie at the EPA distance reduction (pxgSlangEpa.slang epaRun, CUDA
// epa.cuh:427-433 fminf butterfly).  Two integer cubes touching corner to corner at the origin
// (zero penetration): A = [0,2]^3, B = [-2,0]^3, A0 = (s,0,s).  GJK returns OVERLAP with a size-1
// simplex A0-B0 = (s,0,s); epaStart123 builds the double-sided start triangle whose lanes hold
// dist = n.v0 and -dist, i.e. (-0,+0) for s = -0 (a -0 from negating a zero) and (+0,-0) for
// s = +0.  The CUDA's fminf gives e = -0 in both cases and the first iteration terminates with
// lowerBound = e, so o.lowerBound is 0x80000000.  Twins 1/2 (second/first operand wins a tie)
// must each change exactly one case, otherwise the input no longer reaches the tie.
PXG_TEST(npEpa_driverZeroTie)
{
    const uint32_t MAXV = 32;
    std::vector<float> Data, VA, VB;
    std::vector<EpaScalars> S;
    std::vector<EpaOut> ref;
    size_t tieNegFirst = 0, tiePosFirst = 0, twinOther = 0, badSetup = 0;
    for (int k = 0; k < 2; ++k)
    {
        const float s = (k == 0) ? -0.0f : 0.0f;
        const float bv[8][3] = { { 0, 0, 0 }, { -2, 0, 0 }, { 0, -2, 0 }, { -2, -2, 0 }, { 0, 0, -2 }, { -2, 0, -2 }, { 0, -2, -2 }, { -2, -2, -2 } };
        const float av[8][3] = { { s, 0, s }, { 2, 0, 0 }, { 0, 2, 0 }, { 2, 2, 0 }, { 0, 0, 2 }, { 2, 0, 2 }, { 0, 2, 2 }, { 2, 2, 2 } };
        std::vector<float> tA(MAXV * 4, 0.0f), tB(MAXV * 4, 0.0f);
        for (uint32_t v = 0; v < 8; ++v) for (uint32_t c = 0; c < 3; ++c) { tA[v * 4 + c] = av[v][c]; tB[v * 4 + c] = bv[v][c]; }
        float simplex[32]; int size;
        const uint32_t r = gjkRun(V3{ 0, 0, 1 }, tA, tB, 0, 8, 8, 0.25f, 0.98f, simplex, size);
        const uint32_t sx = (k == 0) ? 0x80000000u : 0u;
        if (r != 2u || size != 1 || asU32(simplex[0]) != sx || asU32(simplex[1]) != 0u || asU32(simplex[2]) != sx) ++badSetup;
        const uint32_t vbase = (uint32_t)(VA.size() / 4);
        for (uint32_t v = 0; v < MAXV; ++v) { for (uint32_t c = 0; c < 4; ++c) { VA.push_back(tA[v * 4 + c]); VB.push_back(tB[v * 4 + c]); } }
        for (uint32_t L = 0; L < 32; ++L) Data.push_back(simplex[L]);
        EpaScalars sc{}; sc.size = (uint32_t)size; sc.nbA = 8; sc.nbB = 8; sc.convergenceRatio = 0.98f; sc.relEpsBase = 4.0f;
        S.push_back(sc);
        gEpaMinVariant = 0;
        const EpaOut base = epaRef(simplex, (uint32_t)size, VA, VB, vbase, 8, 8, 0.98f, 4.0f);
        ref.push_back(base);
        if (asU32(base.lowerBound) != 0x80000000u) ++badSetup;
        for (int v = 1; v <= 2; ++v)
        {
            gEpaMinVariant = v;
            const EpaOut twin = epaRef(simplex, (uint32_t)size, VA, VB, vbase, 8, 8, 0.98f, 4.0f);
            gEpaMinVariant = 0;
            const bool changed = std::memcmp(&twin, &base, sizeof(EpaOut)) != 0;
            if (changed && v == 1 && k == 0) ++tieNegFirst;
            else if (changed && v == 2 && k == 1) ++tiePosFirst;
            else if (changed) ++twinOther;
        }
    }
    PXG_CHECK_MSG(badSetup == 0, "epa zero tie: setup drifted (%zu)", badSetup);
    PXG_CHECK_MSG(tieNegFirst == 1 && twinOther == 0, "epa zero tie: (-0 lane0, +0 lane1) tie not observed (%zu, other %zu)", tieNegFirst, twinOther);
    PXG_CHECK_MSG(tiePosFirst == 1 && twinOther == 0, "epa zero tie: (+0 lane0, -0 lane1) tie not observed (%zu, other %zu)", tiePosFirst, twinOther);

    const uint32_t N = (uint32_t)ref.size();
    PxgSlangComputeDevice* dev = testDevice();
    PxgBufferHandle bData = createBufferFrom(dev, Data);
    PxgBufferHandle bVA = createBufferFrom(dev, VA);
    PxgBufferHandle bVB = createBufferFrom(dev, VB);
    PxgBufferHandle bS = createBufferFrom(dev, S);
    std::vector<EpaOut> outInit(N); std::memset(outInit.data(), 0xEE, outInit.size() * sizeof(EpaOut));
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);
    EpaParams prm = { N, MAXV, 0, 0 };
    Kernel kr("npEpa.spv", "ctEpa", { 4, 16, 16, (uint32_t)sizeof(EpaScalars), (uint32_t)sizeof(EpaOut) }, sizeof(EpaParams));
    kr.dispatch(1, { bData, bVA, bVB, bS, bOut }, prm);
    dev->flush();
    const auto gOut = downloadBuffer<EpaOut>(dev, bOut, N);
    auto ckv = [&](const float g[4], const float r[4]) { return asU32(g[0]) == asU32(r[0]) && asU32(g[1]) == asU32(r[1]) && asU32(g[2]) == asU32(r[2]); };
    size_t mm = 0;
    for (uint32_t t = 0; t < N; ++t)
    {
        const EpaOut& r = ref[t]; const EpaOut& g = gOut[t];
        if (g.result != r.result) ++mm;
        if (g.degenerate != r.degenerate) ++mm;
        if (asU32(g.lowerBound) != asU32(r.lowerBound)) ++mm;
        if (asU32(g.upperBound) != asU32(r.upperBound)) ++mm;
        if (!ckv(g.direction, r.direction)) ++mm;
        if (!ckv(g.closestPointA, r.closestPointA)) ++mm;
        if (!ckv(g.closestPointB, r.closestPointB)) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "epa zero tie: %zu mismatches (N=%u, lb0 %08x lb1 %08x)", mm, N, N > 0 ? asU32(gOut[0].lowerBound) : 0u, N > 1 ? asU32(gOut[1].lowerBound) : 0u);
    for (PxgBufferHandle h : { bData, bVA, bVB, bS, bOut }) dev->destroyBuffer(h);
}

} // namespace
