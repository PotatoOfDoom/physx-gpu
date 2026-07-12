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

// Verification of npCvxAddManifold.slang: addManifoldPoint() from cudaGJKEPA.cu.
// Insert a new contact into a persistent manifold: replace a matching slot,
// append when not full, else reduce the 5 candidates to the best 4.  The output
// manifold slot values are bit-exact copies of the inputs (the reduce only
// selects), so the reference compares the used slots + numContacts exactly, with
// groups whose match test or reduce argmin/argmax sits near a boundary skipped.

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
struct AddIn { F4 mA[4], mB[4], mNp[4], newA, newB, newNp; uint32_t numContacts; float rbt; uint32_t pad0, pad1; };
struct AddOut { F4 mA[4], mB[4], mNp[4]; uint32_t numContacts, pad0, pad1, pad2; };

const float FLTMAX = 3.4028235e38f;
const float TIE = 2e-4f;

static inline V3 v3(const F4& f) { return V3{ f.x, f.y, f.z }; }
static inline V3 sub(const V3& a, const V3& b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static inline float dot3(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 crossp(const V3& a, const V3& b) { return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static uint32_t lowestSet(uint32_t m) { uint32_t i = 0; while (!(m & (1u << i))) ++i; return i; }
static uint32_t popc(uint32_t m) { uint32_t c = 0; while (m) { c += m & 1u; m >>= 1; } return c; }

static uint32_t minIdx5(const float v[5], uint32_t mask, bool& marg)
{
    float mn = FLTMAX;
    for (uint32_t L = 0; L < 5; ++L) if (mask & (1u << L)) mn = std::fmin(mn, v[L]);
    uint32_t win = 0;
    for (uint32_t L = 0; L < 5; ++L) if ((mask & (1u << L)) && v[L] == mn) { win = L; break; }
    for (uint32_t L = 0; L < 5; ++L) if ((mask & (1u << L)) && v[L] != mn && (v[L] - mn) < TIE) marg = true;
    return win;
}
static uint32_t maxIdx5(const float v[5], uint32_t mask, bool& marg)
{
    float mx = -FLTMAX;
    for (uint32_t L = 0; L < 5; ++L) if (mask & (1u << L)) mx = std::fmax(mx, v[L]);
    uint32_t win = 0;
    for (uint32_t L = 0; L < 5; ++L) if ((mask & (1u << L)) && v[L] == mx) { win = L; break; }
    for (uint32_t L = 0; L < 5; ++L) if ((mask & (1u << L)) && v[L] != mx && (mx - v[L]) < TIE) marg = true;
    return win;
}

static void refAdd(const AddIn& s, AddOut& o, bool& marg)
{
    V3 A[4], B[4], N[4]; float S[4];
    for (uint32_t j = 0; j < 4; ++j) { A[j] = v3(s.mA[j]); B[j] = v3(s.mB[j]); N[j] = v3(s.mNp[j]); S[j] = s.mNp[j].w; }
    const V3 newA = v3(s.newA), newB = v3(s.newB), newN = v3(s.newNp);
    const float newS = s.newNp.w;
    uint32_t numContacts = s.numContacts;
    const float sqRbt = s.rbt * s.rbt;

    uint32_t matches = 0;
    for (uint32_t j = 0; j < numContacts; ++j)
    {
        const V3 db = sub(B[j], newB), da = sub(A[j], newA);
        const float ddb = dot3(db, db), dda = dot3(da, da);
        if (std::fabs(ddb - sqRbt) < TIE || std::fabs(dda - sqRbt) < TIE) marg = true;
        if (dda < sqRbt || ddb < sqRbt) matches |= (1u << j);
    }

    if (matches != 0 || numContacts < 4)
    {
        uint32_t index;
        if (matches != 0) index = lowestSet(matches);
        else { index = numContacts; numContacts += 1; }
        A[index] = newA; B[index] = newB; N[index] = newN; S[index] = newS;
    }
    else
    {
        V3 cA[5], cB[5], cN[5]; float cS[5];
        for (uint32_t j = 0; j < 4; ++j) { cA[j] = A[j]; cB[j] = B[j]; cN[j] = N[j]; cS[j] = S[j]; }
        cA[4] = newA; cB[4] = newB; cN[4] = newN; cS[4] = newS;
        const uint32_t allMask = 0x1fu;

        const uint32_t i0 = minIdx5(cS, allMask, marg);
        uint32_t mask = (1u << i0);
        float d1[5];
        for (uint32_t L = 0; L < 5; ++L) { const V3 df = sub(cB[i0], cB[L]); d1[L] = dot3(df, df); }
        const uint32_t i1 = maxIdx5(d1, allMask & ~mask, marg);
        mask |= (1u << i1);
        float d[5];
        for (uint32_t L = 0; L < 5; ++L) { const V3 dir = crossp(cN[L], sub(cB[i1], cB[i0])); d[L] = dot3(dir, sub(cB[L], cB[i0])); }
        const uint32_t f = maxIdx5(d, allMask & ~mask, marg);
        mask |= (1u << f);
        uint32_t g = minIdx5(d, allMask & ~mask, marg);
        const float prod = d[f] * d[g];
        if (std::fabs(prod) < TIE) marg = true;
        if (prod > 0.f) g = maxIdx5(d, allMask & ~mask, marg);
        mask |= (1u << g);

        for (uint32_t L = 0; L < 5; ++L)
            if (mask & (1u << L))
            {
                const uint32_t idx = popc(mask & ((1u << L) - 1u));
                A[idx] = cA[L]; B[idx] = cB[L]; N[idx] = cN[L]; S[idx] = cS[L];
            }
        numContacts = popc(mask);
    }

    for (uint32_t j = 0; j < 4; ++j)
    {
        o.mA[j] = F4{ A[j].x, A[j].y, A[j].z, 0 };
        o.mB[j] = F4{ B[j].x, B[j].y, B[j].z, 0 };
        o.mNp[j] = F4{ N[j].x, N[j].y, N[j].z, S[j] };
    }
    o.numContacts = numContacts;
}

static bool eqF4(const F4& a, const F4& b) { return almostEqual(a.x, b.x, 1e-4f, 1e-5f) && almostEqual(a.y, b.y, 1e-4f, 1e-5f) && almostEqual(a.z, b.z, 1e-4f, 1e-5f) && almostEqual(a.w, b.w, 1e-4f, 1e-4f); }

PXG_TEST(npCvxAddManifold_insert)
{
    std::mt19937 rng(140001);
    const uint32_t N = 14000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<AddIn> in(N);
    std::vector<uint8_t> skip(N, 0);
    std::vector<AddOut> ref(N);
    size_t nSkip = 0, pMatch = 0, pAppend = 0, pReduce = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        AddIn s{};
        const float rbt = rf(0.05f, 0.3f);
        s.rbt = rbt;
        // choose a scenario
        uint32_t nc;
        const uint32_t scen = i % 3; // 0 append, 1 match, 2 reduce
        if (scen == 2) nc = 4; else nc = (scen == 1) ? (1 + rng() % 4) : (rng() % 4);
        s.numContacts = nc;
        for (uint32_t j = 0; j < 4; ++j)
        {
            V3 a{ rf(-2, 2), rf(-2, 2), rf(-2, 2) };
            V3 b{ rf(-2, 2), rf(-2, 2), rf(-2, 2) };
            V3 n{ rf(-1, 1), rf(-1, 1), rf(-1, 1) };
            s.mA[j] = { a.x, a.y, a.z, 0 };
            s.mB[j] = { b.x, b.y, b.z, 0 };
            s.mNp[j] = { n.x, n.y, n.z, rf(-1.5f, 0.1f) };
        }
        V3 nA, nB;
        if (scen == 1 && nc > 0)
        {
            // new point matches an existing slot (well inside rbt on both A and B)
            const uint32_t j = rng() % nc;
            const float off = rbt * 0.3f;
            nA = V3{ s.mA[j].x + rf(-off, off), s.mA[j].y + rf(-off, off), s.mA[j].z + rf(-off, off) };
            nB = V3{ s.mB[j].x + rf(-off, off), s.mB[j].y + rf(-off, off), s.mB[j].z + rf(-off, off) };
        }
        else
        {
            // far from all existing (no match): sample until clear of every slot by >2*rbt
            for (int tries = 0; tries < 40; ++tries)
            {
                nA = V3{ rf(-2.5f, 2.5f), rf(-2.5f, 2.5f), rf(-2.5f, 2.5f) };
                nB = V3{ rf(-2.5f, 2.5f), rf(-2.5f, 2.5f), rf(-2.5f, 2.5f) };
                bool clear = true;
                const float lim = (2.0f * rbt) * (2.0f * rbt);
                for (uint32_t j = 0; j < nc; ++j)
                {
                    const V3 da = sub(nA, v3(s.mA[j])), db = sub(nB, v3(s.mB[j]));
                    if (dot3(da, da) < lim || dot3(db, db) < lim) { clear = false; break; }
                }
                if (clear) break;
            }
        }
        s.newA = { nA.x, nA.y, nA.z, 0 };
        s.newB = { nB.x, nB.y, nB.z, 0 };
        V3 nn{ rf(-1, 1), rf(-1, 1), rf(-1, 1) };
        s.newNp = { nn.x, nn.y, nn.z, rf(-1.5f, 0.1f) };
        in[i] = s;

        bool marg = false;
        refAdd(s, ref[i], marg);
        skip[i] = marg ? 1u : 0u;
        if (marg) { ++nSkip; continue; }
        if (scen == 1 && nc > 0) ++pMatch; else if (nc < 4) ++pAppend; else ++pReduce;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<AddOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct AddParams { uint32_t numTests; } prm = { N };
    Kernel k("npCvxAddManifold.spv", "cvxAddManifold",
             { (uint32_t)sizeof(AddIn), (uint32_t)sizeof(AddOut) }, sizeof(AddParams));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<AddOut>(dev, bOut, N);
    size_t mmN = 0, mmSlot = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (skip[i]) continue;
        if (g[i].numContacts != ref[i].numContacts) { ++mmN; continue; }
        for (uint32_t j = 0; j < ref[i].numContacts; ++j)
        {
            if (!eqF4(ref[i].mA[j], g[i].mA[j]) || !eqF4(ref[i].mB[j], g[i].mB[j]) || !eqF4(ref[i].mNp[j], g[i].mNp[j])) { ++mmSlot; break; }
        }
    }
    PXG_CHECK_MSG(mmN == 0, "addManifold: %zu numContacts mismatches", mmN);
    PXG_CHECK_MSG(mmSlot == 0, "addManifold: %zu manifold-slot mismatches", mmSlot);
    PXG_CHECK_MSG(nSkip < N / 2, "addManifold: too many marginal (%zu/%u)", nSkip, N);
    PXG_CHECK_MSG(pMatch > N / 20 && pAppend > N / 20 && pReduce > N / 20,
                  "addManifold: path coverage thin (match %zu / append %zu / reduce %zu)", pMatch, pAppend, pReduce);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
