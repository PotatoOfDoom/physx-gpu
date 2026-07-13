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

// Verification of npCvxClipEmit.slang: the polyClip contact-emission tail of
// generateConvexContacts.  Given clip points + seps and the clip-plane normal,
// it reduces to <=4 (schlock::reduce, only when nbContacts>4), derives the paired
// manifold point, and compacts into the manifold.  The reference transcribes the
// CUDA; numManifold is exact, the manifold slots to tolerance, with reduce
// argmin/argmax near-ties skipped.

#include "pxgTestFramework.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct V3 { float x, y, z; };
struct ClipEmitIn { F4 clipPt[32]; F4 clipN, aToBq, aToBp; };
struct ClipEmitOut { F4 mA[4], mB[4], mNp[4]; uint32_t numManifold, pad0, pad1, pad2; };

const float FLTMAX = 3.4028235e38f;
const float TIE = 2e-4f;

static inline V3 add(const V3& a, const V3& b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static inline V3 sub(const V3& a, const V3& b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static inline V3 mul(const V3& a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; }
static inline float dot3(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 crossp(const V3& a, const V3& b) { return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static inline V3 nrm(const V3& a) { float l = std::sqrt(dot3(a, a)); return l > 0 ? mul(a, 1.f / l) : a; }
static uint32_t popc(uint32_t m) { uint32_t c = 0; while (m) { c += m & 1u; m >>= 1; } return c; }
static V3 qRotInv(const F4& q, const V3& v) { const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z, w2 = q.w * q.w - 0.5f, d2 = q.x * vx + q.y * vy + q.z * vz; return V3{ vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * d2, vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * d2, vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * d2 }; }

static uint32_t minIdx(const float v[32], uint32_t mask, bool& marg)
{
    float mn = FLTMAX;
    for (uint32_t L = 0; L < 32; ++L) if (mask & (1u << L)) mn = std::fmin(mn, v[L]);
    uint32_t win = 0; for (uint32_t L = 0; L < 32; ++L) if ((mask & (1u << L)) && v[L] == mn) { win = L; break; }
    for (uint32_t L = 0; L < 32; ++L) if ((mask & (1u << L)) && v[L] != mn && (v[L] - mn) < TIE) marg = true;
    return win;
}
static uint32_t maxIdx(const float v[32], uint32_t mask, bool& marg)
{
    float mx = -FLTMAX;
    for (uint32_t L = 0; L < 32; ++L) if (mask & (1u << L)) mx = std::fmax(mx, v[L]);
    uint32_t win = 0; for (uint32_t L = 0; L < 32; ++L) if ((mask & (1u << L)) && v[L] == mx) { win = L; break; }
    for (uint32_t L = 0; L < 32; ++L) if ((mask & (1u << L)) && v[L] != mx && (mx - v[L]) < TIE) marg = true;
    return win;
}

static void refClipEmit(const ClipEmitIn& s, ClipEmitOut& o, bool& marg)
{
    const uint32_t nbContacts = (uint32_t)s.clipN.w;
    const V3 clipN{ s.clipN.x, s.clipN.y, s.clipN.z };
    const V3 aToBp{ s.aToBp.x, s.aToBp.y, s.aToBp.z };
    const bool refIsReference = s.aToBp.w != 0.f;

    V3 pt[32]; float sep[32];
    for (uint32_t L = 0; L < 32; ++L) { pt[L] = V3{ s.clipPt[L].x, s.clipPt[L].y, s.clipPt[L].z }; sep[L] = s.clipPt[L].w; }

    const uint32_t allMask = (nbContacts >= 32) ? 0xffffffffu : ((1u << nbContacts) - 1);
    uint32_t mask;
    if (nbContacts > 4)
    {
        const uint32_t i0 = minIdx(sep, allMask, marg);
        mask = (1u << i0);
        float d1[32]; for (uint32_t L = 0; L < 32; ++L) { const V3 df = sub(pt[i0], pt[L]); d1[L] = dot3(df, df); }
        const uint32_t i1 = maxIdx(d1, allMask & ~mask, marg);
        mask |= (1u << i1);
        const V3 dir = crossp(clipN, sub(pt[i1], pt[i0]));
        float d[32]; for (uint32_t L = 0; L < 32; ++L) d[L] = dot3(dir, sub(pt[L], pt[i0]));
        const uint32_t f = maxIdx(d, allMask & ~mask, marg);
        mask |= (1u << f);
        uint32_t g = minIdx(d, allMask & ~mask, marg);
        const float prod = d[f] * d[g];
        if (std::fabs(prod) < TIE) marg = true;
        if (prod > 0.f) g = maxIdx(d, allMask & ~mask, marg);
        mask |= (1u << g);
    }
    else mask = allMask;

    V3 mA[4], mB[4], mN[4]; float mS[4];
    for (uint32_t j = 0; j < 4; ++j) { mA[j] = V3{ 0, 0, 0 }; mB[j] = V3{ 0, 0, 0 }; mN[j] = V3{ 0, 0, 0 }; mS[j] = 0; }
    for (uint32_t L = 0; L < 32; ++L)
        if (mask & (1u << L))
        {
            const uint32_t idx = popc(mask & ((1u << L) - 1));
            if (idx < 4)
            {
                const V3 clipPt = pt[L]; const float sepL = sep[L];
                const V3 otherPt = add(clipPt, mul(clipN, sepL));
                V3 pa, pb, n;
                if (refIsReference) { pa = otherPt; pb = clipPt; n = clipN; }
                else { pa = clipPt; pb = otherPt; n = mul(clipN, -1.f); }
                mA[idx] = qRotInv(s.aToBq, sub(pa, aToBp));
                mB[idx] = pb; mN[idx] = n; mS[idx] = sepL;
            }
        }

    for (uint32_t j = 0; j < 4; ++j) { o.mA[j] = F4{ mA[j].x, mA[j].y, mA[j].z, 0 }; o.mB[j] = F4{ mB[j].x, mB[j].y, mB[j].z, 0 }; o.mNp[j] = F4{ mN[j].x, mN[j].y, mN[j].z, mS[j] }; }
    o.numManifold = popc(mask);
}

static bool eqP(const F4& r, const F4& v) { return almostEqual(v.x, r.x, 2e-4f, 2e-5f) && almostEqual(v.y, r.y, 2e-4f, 2e-5f) && almostEqual(v.z, r.z, 2e-4f, 2e-5f) && almostEqual(v.w, r.w, 2e-4f, 2e-4f); }

PXG_TEST(npCvxClipEmit_emit)
{
    std::mt19937 rng(146001);
    const uint32_t N = 12000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto randQuat = [&]() { V3 ax = nrm(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) }); const float a = rf(-3.1f, 3.1f); return F4{ ax.x * std::sin(a * .5f), ax.y * std::sin(a * .5f), ax.z * std::sin(a * .5f), std::cos(a * .5f) }; };

    std::vector<ClipEmitIn> in(N);
    std::vector<uint8_t> skip(N, 0);
    std::vector<ClipEmitOut> ref(N);
    size_t nSkip = 0, red = 0, noRed = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        ClipEmitIn s{};
        const uint32_t nbC = 1 + (rng() % 20); // 1..20
        const V3 clipN = nrm(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) });
        for (uint32_t L = 0; L < 32; ++L)
        {
            const V3 p = (L < nbC) ? V3{ rf(-2, 2), rf(-2, 2), rf(-2, 2) } : V3{ 0, 0, 0 };
            s.clipPt[L] = F4{ p.x, p.y, p.z, (L < nbC) ? rf(-1.5f, 0.2f) : 0.f };
        }
        s.clipN = { clipN.x, clipN.y, clipN.z, float(nbC) };
        s.aToBq = randQuat();
        const float refIsRef = (i % 2 == 0) ? 1.f : 0.f;
        s.aToBp = { rf(-2, 2), rf(-2, 2), rf(-2, 2), refIsRef };
        in[i] = s;

        bool marg = false;
        refClipEmit(s, ref[i], marg);
        skip[i] = marg ? 1u : 0u;
        if (marg) { ++nSkip; continue; }
        if (nbC > 4) ++red; else ++noRed;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<ClipEmitOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct ClipEmitParams { uint32_t numTests; } prm = { N };
    Kernel k("npCvxClipEmit.spv", "cvxClipEmit", { (uint32_t)sizeof(ClipEmitIn), (uint32_t)sizeof(ClipEmitOut) }, sizeof(ClipEmitParams));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<ClipEmitOut>(dev, bOut, N);
    size_t mmN = 0, mmSlot = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (skip[i]) continue;
        if (g[i].numManifold != ref[i].numManifold) { ++mmN; continue; }
        for (uint32_t j = 0; j < ref[i].numManifold; ++j)
            if (!eqP(ref[i].mA[j], g[i].mA[j]) || !eqP(ref[i].mB[j], g[i].mB[j]) || !eqP(ref[i].mNp[j], g[i].mNp[j])) { ++mmSlot; break; }
    }
    PXG_CHECK_MSG(mmN == 0, "clipEmit: %zu numManifold mismatches", mmN);
    PXG_CHECK_MSG(mmSlot == 0, "clipEmit: %zu manifold-slot mismatches", mmSlot);
    PXG_CHECK_MSG(nSkip < N / 2, "clipEmit: too many marginal (%zu/%u)", nSkip, N);
    PXG_CHECK_MSG(red > N / 20 && noRed > N / 20, "clipEmit: reduce/no-reduce coverage thin (%zu/%zu)", red, noRed);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
