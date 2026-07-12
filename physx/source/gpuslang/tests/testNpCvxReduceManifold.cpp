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

// Verification of npCvxReduceManifold.slang: schlock::reduce<false,false,4>
// manifold contact reduction from cudaGJKEPA.cu.  Given nbContacts (5..32)
// candidate contacts (point + separation + normal per lane) it selects the four
// extreme lanes (deepest, furthest, two perpendicular extremes).  The C++
// reference transcribes the CUDA (butterfly-min/max argmin/argmax with
// lowest-lane tie-break); the result 4-bit lane mask is compared exactly, with
// groups whose argmin/argmax has a near-tie competitor (or the d[f]*d[g] sign
// gate sits near zero) skipped as measure-zero GPU-vs-CPU flips.

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
struct ReduceIn { F4 pt[32]; F4 nrm[32]; F4 sep[32]; uint32_t nbContacts, pad0, pad1, pad2; };
struct ReduceOut { uint32_t mask, count, pad0, pad1; };

const float FLTMAX = 3.4028235e38f;
const float TIE = 2e-4f;   // near-tie margin (well above FMA-contraction error)

static inline V3 sub(const V3& a, const V3& b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static inline float dot3(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 crossp(const V3& a, const V3& b) { return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }

// argmin/argmax over masked lanes with lowest-lane tie-break; flags a near-tie.
static uint32_t minIdx(const float v[32], uint32_t mask, bool& marg)
{
    float mn = FLTMAX;
    for (uint32_t L = 0; L < 32; ++L) if (mask & (1u << L)) mn = std::fmin(mn, v[L]);
    uint32_t win = 0xffffffffu;
    for (uint32_t L = 0; L < 32; ++L)
        if ((mask & (1u << L)) && v[L] == mn) { win = L; break; }
    // near-tie: another masked lane within TIE of the min but not exactly equal
    for (uint32_t L = 0; L < 32; ++L)
        if ((mask & (1u << L)) && v[L] != mn && (v[L] - mn) < TIE) marg = true;
    return win;
}
static uint32_t maxIdx(const float v[32], uint32_t mask, bool& marg)
{
    float mx = -FLTMAX;
    for (uint32_t L = 0; L < 32; ++L) if (mask & (1u << L)) mx = std::fmax(mx, v[L]);
    uint32_t win = 0xffffffffu;
    for (uint32_t L = 0; L < 32; ++L)
        if ((mask & (1u << L)) && v[L] == mx) { win = L; break; }
    for (uint32_t L = 0; L < 32; ++L)
        if ((mask & (1u << L)) && v[L] != mx && (mx - v[L]) < TIE) marg = true;
    return win;
}

static uint32_t refReduce(const ReduceIn& s, bool& marg)
{
    V3 pt[32], nrm[32];
    float sep[32];
    for (uint32_t L = 0; L < 32; ++L)
    {
        pt[L] = V3{ s.pt[L].x, s.pt[L].y, s.pt[L].z };
        nrm[L] = V3{ s.nrm[L].x, s.nrm[L].y, s.nrm[L].z };
        sep[L] = s.sep[L].x;
    }
    const uint32_t nb = s.nbContacts;
    const uint32_t allMask = (nb >= 32) ? 0xffffffffu : ((1u << nb) - 1u);

    const uint32_t i0 = minIdx(sep, allMask, marg);
    uint32_t mask = (1u << i0);

    float d1[32];
    for (uint32_t L = 0; L < 32; ++L) { const V3 df = sub(pt[i0], pt[L]); d1[L] = dot3(df, df); }
    const uint32_t i1 = maxIdx(d1, allMask & ~mask, marg);
    mask |= (1u << i1);

    float d[32];
    for (uint32_t L = 0; L < 32; ++L)
    {
        const V3 dir = crossp(nrm[L], sub(pt[i1], pt[i0]));
        d[L] = dot3(dir, sub(pt[L], pt[i0]));
    }
    const uint32_t f = maxIdx(d, allMask & ~mask, marg);
    mask |= (1u << f);

    uint32_t g = minIdx(d, allMask & ~mask, marg);
    const float prod = d[f] * d[g];
    if (std::fabs(prod) < TIE) marg = true;
    if (prod > 0.f) g = maxIdx(d, allMask & ~mask, marg);
    mask |= (1u << g);

    return mask;
}

PXG_TEST(npCvxReduceManifold_reduce)
{
    std::mt19937 rng(137001);
    const uint32_t N = 14000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<ReduceIn> in(N);
    std::vector<uint8_t> skip(N, 0);
    std::vector<uint32_t> ref(N);
    size_t nSkip = 0, nbLo = 0, nbHi = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        ReduceIn s{};
        const uint32_t nb = 5 + (rng() % 28); // 5..32
        s.nbContacts = nb;
        // shared-ish normal with mild per-lane variation (reduce uses per-lane normal)
        V3 n0{ rf(-1, 1), rf(-1, 1), rf(-1, 1) };
        for (uint32_t L = 0; L < 32; ++L)
        {
            V3 p{ rf(-3, 3), rf(-3, 3), rf(-3, 3) };
            V3 n{ n0.x + rf(-0.15f, 0.15f), n0.y + rf(-0.15f, 0.15f), n0.z + rf(-0.15f, 0.15f) };
            s.pt[L] = { p.x, p.y, p.z, 0 };
            s.nrm[L] = { n.x, n.y, n.z, 0 };
            s.sep[L] = { rf(-1.5f, 0.2f), 0, 0, 0 };
        }
        if (i % 6 == 0)
        {
            // Exact tie at the minimum separation: two distinct lanes share the
            // bit-identical smallest value.  Bit-identical inputs are handled
            // deterministically (lowest-lane tie-break) by both CPU and GPU, so
            // these are NOT near-ties -> not margin-skipped, and they exercise
            // the lowestSetIndex tie-break in i0 selection.
            const uint32_t a = rng() % nb;
            uint32_t b = rng() % nb;
            if (b == a) b = (a + 1) % nb;
            s.sep[a].x = -2.0f;
            s.sep[b].x = -2.0f; // below the [-1.5,0.2] range -> unambiguous min pair
        }
        in[i] = s;
        bool marg = false;
        ref[i] = refReduce(s, marg);
        skip[i] = marg ? 1u : 0u;
        if (marg) { ++nSkip; continue; }
        if (nb <= 12) ++nbLo; else ++nbHi;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<ReduceOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct ReduceParams { uint32_t numTests; } prm = { N };
    Kernel k("npCvxReduceManifold.spv", "cvxReduceManifold",
             { (uint32_t)sizeof(ReduceIn), (uint32_t)sizeof(ReduceOut) }, sizeof(ReduceParams));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<ReduceOut>(dev, bOut, N);
    size_t mm = 0, mmc = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (skip[i]) continue;
        if (g[i].mask != ref[i]) ++mm;
        // every result must select exactly 4 distinct extreme contacts
        if (g[i].count != 4u) ++mmc;
    }
    PXG_CHECK_MSG(mm == 0, "reduce: %zu mask mismatches", mm);
    PXG_CHECK_MSG(mmc == 0, "reduce: %zu results not selecting 4 contacts", mmc);
    PXG_CHECK_MSG(nSkip < N / 2, "reduce: too many marginal (%zu/%u)", nSkip, N);
    PXG_CHECK_MSG(nbLo > N / 20 && nbHi > N / 20, "reduce: nbContacts coverage thin (%zu/%zu)", nbLo, nbHi);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
