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

// Verification of npBoxIncident.slang: getIncidentPolygon4 (width-4 broadcast
// __shfl_sync incident-face selection + per-lane corner transform) and
// calculateBounds4 (width-4 __shfl_xor_sync butterfly AABB reduction).  Both are
// warp-cooperative over groups of 4 lanes; the references transcribe cudaBox.cu
// and are evaluated per group.  For incident, random rigid transforms exercise
// all three face-axis branches; for bounds, every lane is compared bit for bit
// against a lane-exact transcription of the butterfly.  npBoxIncident_bounds-
// SelectTies (P9.89) drives the butterfly with +-0 ties and NaN lanes, so the
// selectMin/selectMax winner of every combine is observable, against an
// FMin/FMax, a swapped-select and a swapped-distance twin.

#include "pxgTestFramework.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct V3 { float x, y, z; };
struct IncidentIn { F4 col0, col1, col2, col3, axis, ext; };

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static bool f4Same(const F4& a, const F4& b) { return std::memcmp(&a, &b, sizeof(F4)) == 0; }

// PxMin/PxMax(float) = intrinsics::selectMin/selectMax (PxMath.h:82-101,
// PxUnixMathIntrinsics.h:116-125).
static float selMin(float a, float b) { return a < b ? a : b; }
static float selMax(float a, float b) { return a > b ? a : b; }

// llvmpipe GLSL.std.450 FMin/FMax under SignedZeroInfNanPreserve (P9.86 F4) -- the
// pre-P9.89 port's Slang min()/max(): zero-ordered, NaN-ignoring.  Mutant twin only.
static float lpFMin(float a, float b)
{
    if (std::isnan(a)) return b;
    if (std::isnan(b)) return a;
    if (a == b) return std::signbit(a) ? a : b;
    return a < b ? a : b;
}
static float lpFMax(float a, float b)
{
    if (std::isnan(a)) return b;
    if (std::isnan(b)) return a;
    if (a == b) return std::signbit(a) ? b : a;
    return a > b ? a : b;
}

// ---------------------------------------------------------------------------------------------
// cudaBox.cu:199-219 calculateBounds4, transcribed lane by lane for one group of 4:
//   bounds.minimum = bounds.maximum = pt;                                   (L203)
//   val = __shfl_xor_sync(pt, 1, 4); bounds.include(val);                   (L205-208)
//   bounds.minimum = bounds.minimum.minimum(__shfl_xor_sync(minimum, 2, 4)); (L210-212)
//   bounds.maximum = bounds.maximum.maximum(__shfl_xor_sync(maximum, 2, 4)); (L214-216)
// include / minimum / maximum are componentwise selectMin/selectMax(own, partner), so on a
// +-0 tie a lane takes its PARTNER's zero, a NaN partner propagates and a NaN own value is
// replaced: the 4 lanes of a group need not end with the same bits.
//
// Twins (mutants of the port, never the CUDA): combine position pos = stage * 3 + component,
// stage 0 = include min (xor 1), 1 = include max (xor 1), 2 = minimum (xor 2), 3 = maximum
// (xor 2).  fminMask bit pos replaces that combine by lpFMin/lpFMax, swapMask bit pos by the
// operand-swapped select partner < own ? partner : own; swapDist runs the distances 2 then 1.
// ---------------------------------------------------------------------------------------------
static void refBounds4(const F4* p, F4* mnOut, F4* mxOut, uint32_t fminMask = 0u, uint32_t swapMask = 0u,
                       bool swapDist = false)
{
    const uint32_t d1 = swapDist ? 2u : 1u, d2 = swapDist ? 1u : 2u;
    auto comb = [&](uint32_t pos, bool isMax, float own, float partner) -> float {
        if (fminMask & (1u << pos)) return isMax ? lpFMax(own, partner) : lpFMin(own, partner);
        if (swapMask & (1u << pos)) return isMax ? selMax(partner, own) : selMin(partner, own);
        return isMax ? selMax(own, partner) : selMin(own, partner);
    };
    float pt[4][3], mn1[4][3], mx1[4][3];
    for (uint32_t l = 0; l < 4; ++l) { pt[l][0] = p[l].x; pt[l][1] = p[l].y; pt[l][2] = p[l].z; }
    for (uint32_t l = 0; l < 4; ++l)
        for (uint32_t k = 0; k < 3; ++k)
        {
            const float val = pt[l ^ d1][k];                        // __shfl_xor_sync(pt, 1, 4)
            mn1[l][k] = comb(0u * 3u + k, false, pt[l][k], val);    // include: minimum.minimum(val)
            mx1[l][k] = comb(1u * 3u + k, true, pt[l][k], val);     // include: maximum.maximum(val)
        }
    for (uint32_t l = 0; l < 4; ++l)
    {
        float mn[3], mx[3];
        for (uint32_t k = 0; k < 3; ++k)
        {
            mn[k] = comb(2u * 3u + k, false, mn1[l][k], mn1[l ^ d2][k]);
            mx[k] = comb(3u * 3u + k, true, mx1[l][k], mx1[l ^ d2][k]);
        }
        mnOut[l] = F4{ mn[0], mn[1], mn[2], 0.f };
        mxOut[l] = F4{ mx[0], mx[1], mx[2], 0.f };
    }
}

// The pre-P9.89 fixture's reference: a serial std::min/std::max fold over lanes 0..3,
// broadcast to all 4 lanes (not the CUDA: it has one winner per group).  Twin only.
static void refSerialBroadcast(const F4* p, F4* mnOut, F4* mxOut)
{
    F4 mn = p[0], mx = p[0];
    for (uint32_t l = 1; l < 4; ++l)
    {
        mn.x = std::min(mn.x, p[l].x); mn.y = std::min(mn.y, p[l].y); mn.z = std::min(mn.z, p[l].z);
        mx.x = std::max(mx.x, p[l].x); mx.y = std::max(mx.y, p[l].y); mx.z = std::max(mx.z, p[l].z);
    }
    for (uint32_t l = 0; l < 4; ++l)
    {
        mnOut[l] = F4{ mn.x, mn.y, mn.z, 0.f };
        mxOut[l] = F4{ mx.x, mx.y, mx.z, 0.f };
    }
}

// Runs boxCalculateBounds4 over pts (4 lanes per group) and returns the per-lane min / max.
static void runBounds4(const std::vector<F4>& pts, std::vector<F4>& gMin, std::vector<F4>& gMax)
{
    const uint32_t N = uint32_t(pts.size());
    const uint32_t NG = N / 4u;
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<IncidentIn> dummyIn(16);
    std::vector<F4> dummyV(16), dummyN(16), minInit(N), maxInit(N);
    std::memset(minInit.data(), 0xcd, N * sizeof(F4));
    std::memset(maxInit.data(), 0xcd, N * sizeof(F4));
    PxgBufferHandle bIn = createBufferFrom(dev, dummyIn);
    PxgBufferHandle bV = createBufferFrom(dev, dummyV);
    PxgBufferHandle bN = createBufferFrom(dev, dummyN);
    PxgBufferHandle bPts = createBufferFrom(dev, pts);
    PxgBufferHandle bMin = createBufferFrom(dev, minInit);
    PxgBufferHandle bMax = createBufferFrom(dev, maxInit);

    struct BoxIncParams { uint32_t numGroups; } p = { NG };
    Kernel k("npBoxIncident.spv", "boxCalculateBounds4", { 96, 16, 16, 16, 16, 16 }, sizeof(BoxIncParams));
    k.dispatch(GROUPS, { bIn, bV, bN, bPts, bMin, bMax }, p);
    dev->flush();

    gMin = downloadBuffer<F4>(dev, bMin, N);
    gMax = downloadBuffer<F4>(dev, bMax, N);
    for (PxgBufferHandle h : { bIn, bV, bN, bPts, bMin, bMax }) dev->destroyBuffer(h);
}

static inline V3 mulAdd(const F4& c0, const F4& c1, const F4& c2, const F4& c3, const V3& v)
{
    return V3{ c0.x * v.x + c1.x * v.y + c2.x * v.z + c3.x,
               c0.y * v.x + c1.y * v.y + c2.y * v.z + c3.y,
               c0.z * v.x + c1.z * v.y + c2.z * v.z + c3.z };
}
static inline float dot3(const F4& a, const F4& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// mirrors cudaBox.cu getIncidentPolygon4(): returns the 4 face-corner verts
// (one per lane) and the group-uniform incident-face normal.  Returns the
// selected face axis (0/1/2) so the test can confirm branch coverage.
uint32_t refIncident(const IncidentIn& in, V3 outVerts[4], V3& faceNormal, bool& nearTie)
{
    const float d0 = dot3(in.col0, in.axis);
    const float d1 = dot3(in.col1, in.axis);
    const float d2 = dot3(in.col2, in.axis);
    const float a0 = std::fabs(d0), a1 = std::fabs(d1), a2 = std::fabs(d2);

    uint32_t axisSel;
    if (a0 >= a1 && a0 >= a2) axisSel = 0;
    else if (a1 >= a2)        axisSel = 1;
    else                      axisSel = 2;

    // The face-axis branch turns on the ordering of |d0|,|d1|,|d2|.  If the top
    // two magnitudes are within FMA-contraction noise, the GPU dot product could
    // pick a different face than this CPU reference — a whole-vertex flip that no
    // tolerance can absorb.  Flag such (measure-zero) groups so the caller can
    // skip them; a genuine branch bug still shows on the ~all non-tie groups.
    const float top1 = std::max(a0, std::max(a1, a2));
    const float mid = (a0 + a1 + a2) - top1 - std::min(a0, std::min(a1, a2));
    nearTie = (top1 - mid) <= 1e-4f * (top1 + 1e-6f);

    for (uint32_t lane = 0; lane < 4; ++lane)
    {
        const float s0 = (lane == 0 || lane == 3) ? -1.f : 1.f;
        const float s1 = (lane & 2u) ? 1.f : -1.f;
        V3 sgn;
        if (axisSel == 0)
        {
            sgn.x = (d0 > 0.f) ? -1.f : 1.f; sgn.y = s0; sgn.z = s1;
            faceNormal = V3{ in.col0.x * sgn.x, in.col0.y * sgn.x, in.col0.z * sgn.x };
        }
        else if (axisSel == 1)
        {
            sgn.y = (d1 > 0.f) ? -1.f : 1.f; sgn.x = s0; sgn.z = s1;
            faceNormal = V3{ in.col1.x * sgn.y, in.col1.y * sgn.y, in.col1.z * sgn.y };
        }
        else
        {
            sgn.z = (d2 > 0.f) ? -1.f : 1.f; sgn.x = s0; sgn.y = s1;
            faceNormal = V3{ in.col2.x * sgn.z, in.col2.y * sgn.z, in.col2.z * sgn.z };
        }
        const V3 vert{ in.ext.x * sgn.x, in.ext.y * sgn.y, in.ext.z * sgn.z };
        outVerts[lane] = mulAdd(in.col0, in.col1, in.col2, in.col3, vert);
    }
    return axisSel;
}

PXG_TEST(npBoxIncident_incidentPolygon4)
{
    std::mt19937 rng(48010);
    const uint32_t NG = 3000;               // groups of 4
    const uint32_t N = NG * 4;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 100000) / 100000.f); };

    std::vector<IncidentIn> in(NG);
    size_t branchCount[3] = { 0, 0, 0 };
    std::vector<V3> refV(N);
    std::vector<V3> refN(NG);
    std::vector<uint8_t> tie(NG, 0);
    size_t tieCount = 0;
    for (uint32_t g = 0; g < NG; ++g)
    {
        // random rigid transform (rotation columns from Euler XYZ + translation)
        const float ax = rf(-3.1f, 3.1f), ay = rf(-3.1f, 3.1f), az = rf(-3.1f, 3.1f);
        const float cx = std::cos(ax), sx = std::sin(ax);
        const float cy = std::cos(ay), sy = std::sin(ay);
        const float cz = std::cos(az), sz = std::sin(az);
        IncidentIn s{};
        s.col0 = { cy * cz, cy * sz, -sy, 0 };
        s.col1 = { sx * sy * cz - cx * sz, sx * sy * sz + cx * cz, sx * cy, 0 };
        s.col2 = { cx * sy * cz + sx * sz, cx * sy * sz - sx * cz, cx * cy, 0 };
        s.col3 = { rf(-5, 5), rf(-5, 5), rf(-5, 5), 0 };
        // random axis (need not be unit; sign selection uses |col.dot(axis)|)
        s.axis = { rf(-1, 1), rf(-1, 1), rf(-1, 1), 0 };
        s.ext = { rf(0.3f, 2.5f), rf(0.3f, 2.5f), rf(0.3f, 2.5f), 0 };
        in[g] = s;

        V3 v[4]; V3 fn; bool nt = false;
        branchCount[refIncident(s, v, fn, nt)]++;
        for (uint32_t l = 0; l < 4; ++l) refV[g * 4 + l] = v[l];
        refN[g] = fn;
        tie[g] = nt ? 1u : 0u;
        if (nt) ++tieCount;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<F4> vInit(N, F4{ 9, 9, 9, 9 }), nInit(NG, F4{ 9, 9, 9, 9 });
    std::vector<F4> dummyPts(16, F4{ 0, 0, 0, 0 }), dummyMin(16), dummyMax(16);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bV = createBufferFrom(dev, vInit);
    PxgBufferHandle bN = createBufferFrom(dev, nInit);
    PxgBufferHandle bPts = createBufferFrom(dev, dummyPts);
    PxgBufferHandle bMin = createBufferFrom(dev, dummyMin);
    PxgBufferHandle bMax = createBufferFrom(dev, dummyMax);

    struct BoxIncParams { uint32_t numGroups; } p = { NG };
    Kernel k("npBoxIncident.spv", "boxIncidentPolygon4", { 96, 16, 16, 16, 16, 16 }, sizeof(BoxIncParams));
    k.dispatch(GROUPS, { bIn, bV, bN, bPts, bMin, bMax }, p);
    dev->flush();

    const auto gV = downloadBuffer<F4>(dev, bV, N);
    const auto gN = downloadBuffer<F4>(dev, bN, NG);
    size_t mmV = 0, mmN = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (tie[i >> 2]) continue; // measure-zero face-axis tie: branch may differ legitimately
        const V3& r = refV[i];
        if (!almostEqual(gV[i].x, r.x, 1e-4f, 1e-5f) || !almostEqual(gV[i].y, r.y, 1e-4f, 1e-5f) ||
            !almostEqual(gV[i].z, r.z, 1e-4f, 1e-5f)) ++mmV;
    }
    for (uint32_t g = 0; g < NG; ++g)
    {
        if (tie[g]) continue;
        const V3& r = refN[g];
        if (!almostEqual(gN[g].x, r.x, 1e-4f, 1e-5f) || !almostEqual(gN[g].y, r.y, 1e-4f, 1e-5f) ||
            !almostEqual(gN[g].z, r.z, 1e-4f, 1e-5f)) ++mmN;
    }
    PXG_CHECK_MSG(mmV == 0, "boxIncidentPolygon4: %zu vertex mismatches", mmV);
    PXG_CHECK_MSG(mmN == 0, "boxIncidentPolygon4: %zu faceNormal mismatches", mmN);
    // near-ties are measure-zero; if many appear the skip would gut the test
    PXG_CHECK_MSG(tieCount < NG / 100, "incident: too many face-axis near-ties (%zu)", tieCount);
    PXG_CHECK_MSG(branchCount[0] > NG / 8 && branchCount[1] > NG / 8 && branchCount[2] > NG / 8,
                  "incident: face-axis branches under-exercised (%zu/%zu/%zu)",
                  branchCount[0], branchCount[1], branchCount[2]);

    for (PxgBufferHandle h : { bIn, bV, bN, bPts, bMin, bMax }) dev->destroyBuffer(h);
}

PXG_TEST(npBoxIncident_calculateBounds4)
{
    std::mt19937 rng(48020);
    const uint32_t NG = 4000;
    const uint32_t N = NG * 4;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 100000) / 100000.f); };

    std::vector<F4> pts(N);
    for (uint32_t i = 0; i < N; ++i)
        pts[i] = F4{ rf(-10, 10), rf(-10, 10), rf(-10, 10), 0 };

    // reference: the lane-exact butterfly transcription (refBounds4), per group of 4
    std::vector<F4> refMin(N), refMax(N);
    for (uint32_t g = 0; g < NG; ++g)
        refBounds4(&pts[g * 4], &refMin[g * 4], &refMax[g * 4]);

    std::vector<F4> gMin, gMax;
    runBounds4(pts, gMin, gMax);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
        if (!f4Same(gMin[i], refMin[i]) || !f4Same(gMax[i], refMax[i])) ++mm;   // bit for bit
    PXG_CHECK_MSG(mm == 0, "boxCalculateBounds4: %zu mismatches", mm);
}

// ---------------------------------------------------------------------------------------------
// P9.89 (npBoxIncident_boundsSelectTies): the tie and NaN winner of every butterfly combine.
// Every combine of calculateBounds4 is selectMin/selectMax(own, partner) (see refBounds4); the
// pre-P9.89 port spelled them as Slang min()/max(), i.e. GLSL.std.450 FMin/FMax, which the
// SZINP-stamped llvmpipe build runs zero-ordered (a +-0 tie gives -0 for min and +0 for max in
// either order, so every lane of a group got the same zero) and NaN-ignoring.  The old fixture
// could not see it: its rf(-10, 10) grid never yields -0 or NaN, its reference was a serial
// std::min/std::max broadcast, and it compared with float != (+0 == -0).
//
// Group kinds (components rotate through the variants): a +-0 tie inside one xor-1 pair with
// the other pair strictly beyond it (the level-1 include winner reaches every lane), a +-0 tie
// between the two pairs' level-1 results (the xor-2 winner), three or four zeros of mixed sign
// (both levels tie; also separates the distance order), 1/2/3 NaN lanes with distinct payloads
// and random finite partners, a mixed pool of zeros, NaNs and finite values, and a base kind
// of distinct non-round finite values where every spelling agrees.
// ---------------------------------------------------------------------------------------------
PXG_TEST(npBoxIncident_boundsSelectTies)
{
    std::mt19937 rng(890041);
    const uint32_t NG = 4096;   // 16384 lanes = 64 workgroups of 256
    const uint32_t N = NG * 4;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    const float PZ = 0.0f, NZ = u2f(0x80000000u);
    // distinct quiet-NaN payload per (lane, component), both signs
    auto nanOf = [](uint32_t lane, uint32_t k) {
        return u2f(((lane & 1u) ? 0xffc00000u : 0x7fc00000u) | (0x1000u * (lane + 1u)) | (0x10u * (k + 1u)) | 0x1u);
    };
    static const uint32_t kPairs[6][2] = { { 0, 1 }, { 2, 3 }, { 0, 2 }, { 1, 3 }, { 0, 3 }, { 1, 2 } };

    enum { K_L1_MIN, K_L1_MAX, K_L2_MIN, K_L2_MAX, K_TRI_MIN, K_TRI_MAX, K_ZERO4, K_NAN1, K_NAN2, K_NAN3,
           K_MIXED, K_BASE, K_COUNT };
    uint32_t kindGroups[K_COUNT] = {};

    std::vector<F4> pts(N);
    for (uint32_t g = 0; g < NG; ++g)
    {
        const uint32_t kind = g % K_COUNT;
        const uint32_t rot = g / K_COUNT;
        ++kindGroups[kind];
        float v[4][3];
        for (uint32_t k = 0; k < 3; ++k)
        {
            const uint32_t j = rot + k;
            for (uint32_t l = 0; l < 4; ++l) v[l][k] = rf(-5.f, 5.f);
            switch (kind)
            {
            case K_L1_MIN:   // +-0 tie in one xor-1 pair, the other pair positive: include-min winner
            case K_L1_MAX:   // ... other pair negative: include-max winner
            {
                const uint32_t a = (j & 1u) * 2u;
                const bool negFirst = ((j >> 1) & 1u) != 0u;
                v[a][k] = negFirst ? NZ : PZ;
                v[a ^ 1u][k] = negFirst ? PZ : NZ;
                const float lo = (kind == K_L1_MIN) ? 0.1f : -5.f, hi = (kind == K_L1_MIN) ? 5.f : -0.1f;
                v[a ^ 2u][k] = rf(lo, hi);
                v[a ^ 3u][k] = rf(lo, hi);
                break;
            }
            case K_L2_MIN:   // one zero per xor-1 pair, opposite signs, rest positive: xor-2 min winner
            case K_L2_MAX:   // ... rest negative: xor-2 max winner
            {
                const float lo = (kind == K_L2_MIN) ? 0.1f : -5.f, hi = (kind == K_L2_MIN) ? 5.f : -0.1f;
                for (uint32_t l = 0; l < 4; ++l) v[l][k] = rf(lo, hi);
                const uint32_t zA = j & 1u, zB = 2u + ((j >> 1) & 1u);
                const bool negA = ((j >> 2) & 1u) != 0u;
                v[zA][k] = negA ? NZ : PZ;
                v[zB][k] = negA ? PZ : NZ;
                break;
            }
            case K_TRI_MIN:  // three zeros of mixed sign + one positive lane
            case K_TRI_MAX:  // ... + one negative lane
            {
                const uint32_t f = j & 3u;
                const uint32_t sbits = 1u + ((j >> 2) + 2u * k) % 6u;   // 3-bit patterns 1..6: mixed signs
                uint32_t zi = 0;
                for (uint32_t l = 0; l < 4; ++l)
                {
                    if (l == f) v[l][k] = (kind == K_TRI_MIN) ? rf(0.1f, 5.f) : rf(-5.f, -0.1f);
                    else v[l][k] = ((sbits >> zi++) & 1u) ? NZ : PZ;
                }
                break;
            }
            case K_ZERO4:    // four zeros, mixed signs: min and max tie at both levels
            {
                const uint32_t sbits = 1u + (rot * 3u + k * 5u) % 14u;   // 4-bit patterns 1..14
                for (uint32_t l = 0; l < 4; ++l) v[l][k] = ((sbits >> l) & 1u) ? NZ : PZ;
                break;
            }
            case K_NAN1:
                v[j & 3u][k] = nanOf(j & 3u, k);
                break;
            case K_NAN2:
            {
                const uint32_t* pr = kPairs[j % 6u];
                v[pr[0]][k] = nanOf(pr[0], k);
                v[pr[1]][k] = nanOf(pr[1], k);
                break;
            }
            case K_NAN3:
                for (uint32_t l = 0; l < 4; ++l)
                    if (l != (j & 3u)) v[l][k] = nanOf(l, k);
                break;
            case K_MIXED:    // zeros of both signs, two finite values (shared by lanes) and NaNs
            {
                const float fa = rf(-5.f, 5.f), fb = rf(-5.f, 5.f);
                for (uint32_t l = 0; l < 4; ++l)
                {
                    const uint32_t c = rng() % 5u;
                    v[l][k] = (c == 0u) ? PZ : (c == 1u) ? NZ : (c == 2u) ? fa : (c == 3u) ? fb : nanOf(l, k);
                }
                break;
            }
            default: break;  // K_BASE: distinct finite non-round values
            }
        }
        for (uint32_t l = 0; l < 4; ++l) pts[g * 4 + l] = F4{ v[l][0], v[l][1], v[l][2], 0.f };
    }

    std::vector<F4> refMin(N), refMax(N);
    uint32_t nonUniformGroups = 0;
    for (uint32_t g = 0; g < NG; ++g)
    {
        refBounds4(&pts[g * 4], &refMin[g * 4], &refMax[g * 4]);
        for (uint32_t l = 1; l < 4; ++l)
            if (!f4Same(refMin[g * 4 + l], refMin[g * 4]) || !f4Same(refMax[g * 4 + l], refMax[g * 4]))
            {
                ++nonUniformGroups;
                break;
            }
    }

    // Twin replay over all lanes: returns how many lanes differ from the CUDA transcription.
    auto twinDiff = [&](uint32_t fminMask, uint32_t swapMask, bool swapDist, uint32_t* dMin, uint32_t* dMax,
                        std::vector<F4>* keepMin, std::vector<F4>* keepMax) {
        uint32_t d = 0, dn = 0, dx = 0;
        for (uint32_t g = 0; g < NG; ++g)
        {
            F4 tMin[4], tMax[4];
            refBounds4(&pts[g * 4], tMin, tMax, fminMask, swapMask, swapDist);
            for (uint32_t l = 0; l < 4; ++l)
            {
                const bool bn = !f4Same(tMin[l], refMin[g * 4 + l]), bx = !f4Same(tMax[l], refMax[g * 4 + l]);
                dn += bn ? 1u : 0u;
                dx += bx ? 1u : 0u;
                d += (bn || bx) ? 1u : 0u;
                if (keepMin) (*keepMin)[g * 4 + l] = tMin[l];
                if (keepMax) (*keepMax)[g * 4 + l] = tMax[l];
            }
        }
        if (dMin) *dMin = dn;
        if (dMax) *dMax = dx;
        return d;
    };

    // Coverage: each of the 12 combine positions must be observable for both replacements
    // (FMin/FMax and the operand-swapped select) on many lanes; so must the swapped distance
    // order, the whole old port and the old fixture's serial broadcast.
    const char* stageName[4] = { "include.min", "include.max", "xor2.min", "xor2.max" };
    uint32_t minFmin = ~0u, minSwap = ~0u;
    for (uint32_t pos = 0; pos < 12; ++pos)
    {
        const uint32_t dF = twinDiff(1u << pos, 0u, false, nullptr, nullptr, nullptr, nullptr);
        const uint32_t dS = twinDiff(0u, 1u << pos, false, nullptr, nullptr, nullptr, nullptr);
        std::printf("[ info ] boundsSelectTies %s.%c: FMin/FMax twin differs on %u lanes, swapped select on %u\n",
                    stageName[pos / 3], "xyz"[pos % 3], dF, dS);
        minFmin = std::min(minFmin, dF);
        minSwap = std::min(minSwap, dS);
    }
    const uint32_t dDist = twinDiff(0u, 0u, true, nullptr, nullptr, nullptr, nullptr);
    std::vector<F4> oldMin(N), oldMax(N);
    uint32_t oldDiffMin = 0, oldDiffMax = 0;
    const uint32_t oldDiff = twinDiff(0xfffu, 0u, false, &oldDiffMin, &oldDiffMax, &oldMin, &oldMax);
    uint32_t serialDiff = 0;
    for (uint32_t g = 0; g < NG; ++g)
    {
        F4 sMin[4], sMax[4];
        refSerialBroadcast(&pts[g * 4], sMin, sMax);
        for (uint32_t l = 0; l < 4; ++l)
            if (!f4Same(sMin[l], refMin[g * 4 + l]) || !f4Same(sMax[l], refMax[g * 4 + l])) ++serialDiff;
    }
    std::printf("[ info ] boundsSelectTies: %u lanes (%u groups per kind), %u groups with non-uniform lanes;"
                " old-port FMin/FMax twin differs on %u lanes (%u min, %u max); per-position minimum %u (FMin/FMax)"
                " / %u (swapped); distances 2,1 differ on %u lanes; serial broadcast on %u lanes\n",
                N, kindGroups[0], nonUniformGroups, oldDiff, oldDiffMin, oldDiffMax, minFmin, minSwap, dDist,
                serialDiff);
    PXG_CHECK_MSG(minFmin >= 32u && minSwap >= 32u && dDist >= 32u && oldDiff >= N / 3u && oldDiffMin >= N / 8u &&
                      oldDiffMax >= N / 8u && serialDiff >= N / 3u && nonUniformGroups >= NG / 4u,
                  "P9.89: the select-tie lanes lost coverage (min FMin %u, min swapped %u, distances %u, old port %u"
                  " lanes (%u min, %u max), serial %u, non-uniform groups %u)",
                  minFmin, minSwap, dDist, oldDiff, oldDiffMin, oldDiffMax, serialDiff, nonUniformGroups);

    std::vector<F4> gMin, gMax;
    runBounds4(pts, gMin, gMax);
    size_t mm = 0, asOldPort = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (f4Same(gMin[i], refMin[i]) && f4Same(gMax[i], refMax[i]))
            continue;
        if (f4Same(gMin[i], oldMin[i]) && f4Same(gMax[i], oldMax[i]))
            ++asOldPort;
        if (mm < 6)
            std::printf("MM lane %u kind %u min R %08x %08x %08x G %08x %08x %08x | max R %08x %08x %08x G %08x %08x %08x\n",
                        i, (i / 4u) % K_COUNT, f2u(refMin[i].x), f2u(refMin[i].y), f2u(refMin[i].z), f2u(gMin[i].x),
                        f2u(gMin[i].y), f2u(gMin[i].z), f2u(refMax[i].x), f2u(refMax[i].y), f2u(refMax[i].z),
                        f2u(gMax[i].x), f2u(gMax[i].y), f2u(gMax[i].z));
        ++mm;
    }
    if (asOldPort)
        std::printf("MM boundsSelectTies: %zu lanes reproduce the old port's FMin/FMax twin (Slang min()/max())\n",
                    asOldPort);
    PXG_CHECK_MSG(mm == 0, "P9.89: boxCalculateBounds4 differs from the CUDA selectMin/selectMax butterfly on %zu lanes"
                  " (%zu as the FMin/FMax twin)", mm, asOldPort);
}

} // namespace
