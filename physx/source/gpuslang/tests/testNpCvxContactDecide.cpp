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

// Verification of npCvxContactDecide.slang: the front half of
// generateConvexContacts (cudaGJKEPA.cu L858-887).  Computes localNor over the
// pre-add manifold, adds the GJK/EPA witness contact via addManifoldPoint, then
// derives fullContactGen = (initialNbContacts > 1) || (0.707 > localNor.dot(-normal))
// || (numContacts<4) (the first term is 5.12's PCM jitter fix, cudaGJKEPA.cu:864).
// The reference transcribes the CUDA; manifold slots + numContacts are exact,
// localNor to tolerance, fullContactGen exact, with reduce/match near-ties and
// the 0.707 boundary margin-skipped.

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
struct DecIn { F4 mA[4], mB[4], mNp[4], witnessA, witnessB, normal, aToBq, aToBp; uint32_t numContacts, initialNbContacts, pad1, pad2; };
struct DecOut { F4 mA[4], mB[4], mNp[4], localNor; uint32_t numContacts, fullContactGen, pad0, pad1; };

const float FLTMAX = 3.4028235e38f;
const float TIE = 2e-4f;

static inline V3 v3(const F4& f) { return V3{ f.x, f.y, f.z }; }
static inline V3 add(const V3& a, const V3& b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static inline V3 sub(const V3& a, const V3& b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static inline V3 mul(const V3& a, float s) { return V3{ a.x * s, a.y * s, a.z * s }; }
static inline float dot3(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 crossp(const V3& a, const V3& b) { return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static inline V3 normz(const V3& a) { const float m = dot3(a, a); return m > 0.f ? mul(a, 1.f / std::sqrt(m)) : V3{ 0, 0, 0 }; }
static inline V3 unit(const V3& a) { float l = std::sqrt(dot3(a, a)); return l > 0 ? mul(a, 1.f / l) : a; }
static uint32_t lowestSet(uint32_t m) { uint32_t i = 0; while (!(m & (1u << i))) ++i; return i; }
static uint32_t popc(uint32_t m) { uint32_t c = 0; while (m) { c += m & 1u; m >>= 1; } return c; }
static V3 qRotInv(const F4& q, const V3& v) { const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z, w2 = q.w * q.w - 0.5f, d2 = q.x * vx + q.y * vy + q.z * vz; return V3{ vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * d2, vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * d2, vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * d2 }; }

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

static void refDecide(const DecIn& s, DecOut& o, bool& marg)
{
    V3 A[4], B[4], N[4]; float S[4];
    for (uint32_t j = 0; j < 4; ++j) { A[j] = v3(s.mA[j]); B[j] = v3(s.mB[j]); N[j] = v3(s.mNp[j]); S[j] = s.mNp[j].w; }
    uint32_t numContacts = s.numContacts;

    const V3 normal = v3(s.normal);
    const V3 witnessA = v3(s.witnessA), witnessB = v3(s.witnessB);
    const V3 aToBp = v3(s.aToBp);
    const float rbt = s.aToBp.w;

    // 1. localNor pre-add (pairwise sum + getNormalized), only when numContacts>0
    V3 localNor{ 0, 0, 0 };
    if (numContacts != 0)
    {
        V3 n0 = (0u < numContacts) ? N[0] : V3{ 0, 0, 0 };
        V3 n1 = (1u < numContacts) ? N[1] : V3{ 0, 0, 0 };
        V3 n2 = (2u < numContacts) ? N[2] : V3{ 0, 0, 0 };
        V3 n3 = (3u < numContacts) ? N[3] : V3{ 0, 0, 0 };
        localNor = normz(add(add(n0, n1), add(n2, n3)));
    }

    // 2. addManifoldPoint(aToB.transformInv(witnessA), witnessB, -normal, sep, ...)
    const V3 newA = qRotInv(s.aToBq, sub(witnessA, aToBp));
    const V3 newB = witnessB;
    const V3 newN = mul(normal, -1.f);
    const float newS = dot3(sub(witnessB, witnessA), normal);
    const float sqRbt = rbt * rbt;

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

    // 3. fullContactGen (post-add count)
    const float align = dot3(localNor, mul(normal, -1.f));
    if (std::fabs(0.707106781f - align) < TIE) marg = true; // 0.707 boundary near-tie
    // 5.12 cudaGJKEPA.cu:864-865: `ss_scratch.initialNbContacts > 1` in front (PCM jitter fix)
    const bool fullContactGen = (s.initialNbContacts > 1) || (0.707106781f > align) || (numContacts < 4);

    for (uint32_t j = 0; j < 4; ++j)
    {
        o.mA[j] = F4{ A[j].x, A[j].y, A[j].z, 0 };
        o.mB[j] = F4{ B[j].x, B[j].y, B[j].z, 0 };
        o.mNp[j] = F4{ N[j].x, N[j].y, N[j].z, S[j] };
    }
    o.localNor = F4{ localNor.x, localNor.y, localNor.z, 0 };
    o.numContacts = numContacts;
    o.fullContactGen = fullContactGen ? 1u : 0u;
}

static bool eqF4(const F4& a, const F4& b) { return almostEqual(a.x, b.x, 1e-4f, 1e-5f) && almostEqual(a.y, b.y, 1e-4f, 1e-5f) && almostEqual(a.z, b.z, 1e-4f, 1e-5f) && almostEqual(a.w, b.w, 1e-4f, 1e-4f); }
static bool eqN(const F4& a, const F4& b) { return almostEqual(a.x, b.x, 3e-4f, 3e-5f) && almostEqual(a.y, b.y, 3e-4f, 3e-5f) && almostEqual(a.z, b.z, 3e-4f, 3e-5f); }

PXG_TEST(npCvxContactDecide_decide)
{
    std::mt19937 rng(147001);
    const uint32_t N = 16000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto randQuat = [&]() { V3 ax = unit(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) }); const float a = rf(-3.1f, 3.1f); return F4{ ax.x * std::sin(a * .5f), ax.y * std::sin(a * .5f), ax.z * std::sin(a * .5f), std::cos(a * .5f) }; };

    std::vector<DecIn> in(N);
    std::vector<uint8_t> skip(N, 0);
    std::vector<DecOut> ref(N);
    size_t nSkip = 0, pMatch = 0, pAppend = 0, pReduce = 0, fcT = 0, fcF = 0, fcJitter = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        DecIn s{};
        const float rbt = rf(0.05f, 0.3f);
        uint32_t nc;
        const uint32_t scen = i % 3; // 0 append, 1 match, 2 reduce
        if (scen == 2) nc = 4; else nc = (scen == 1) ? (1 + rng() % 4) : (rng() % 4);
        s.numContacts = nc;
        // stage 1's pre-refresh count: 0 or 1 mostly (the 5.6.1 predicate decides),
        // 2..4 in every fourth case (the 5.12 term forces the full gen)
        s.initialNbContacts = (i % 4 == 3) ? 2u + rng() % 3u : rng() % 2u;

        // half the time make manifold normals coherent about a common axis and
        // set the separating normal ~ -axis, so localNor.dot(-normal) ~ 1 (>0.707)
        const bool coherent = (i % 2 == 0);
        V3 axis = unit(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) });
        for (uint32_t j = 0; j < 4; ++j)
        {
            V3 a{ rf(-2, 2), rf(-2, 2), rf(-2, 2) };
            V3 b{ rf(-2, 2), rf(-2, 2), rf(-2, 2) };
            V3 n = coherent ? unit(add(axis, mul(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) }, 0.1f)))
                            : V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) };
            s.mA[j] = { a.x, a.y, a.z, 0 };
            s.mB[j] = { b.x, b.y, b.z, 0 };
            s.mNp[j] = { n.x, n.y, n.z, rf(-1.5f, 0.1f) };
        }
        V3 nrm = coherent ? unit(add(mul(axis, -1.f), mul(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) }, 0.15f)))
                          : unit(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) });
        s.normal = { nrm.x, nrm.y, nrm.z, 0 };

        // witness A/B: newB = witnessB is the point matched/appended; drive the
        // match vs append/reduce scenario via witnessB proximity to a slot's B.
        s.aToBq = randQuat();
        V3 aToBp{ rf(-1.5f, 1.5f), rf(-1.5f, 1.5f), rf(-1.5f, 1.5f) };
        s.aToBp = { aToBp.x, aToBp.y, aToBp.z, rbt };

        V3 wB, wA;
        if (scen == 1 && nc > 0)
        {
            // witnessB near an existing B (and put witnessA so transformInv(wA)
            // lands near that slot's A too, guaranteeing a match)
            const uint32_t j = rng() % nc;
            const float off = rbt * 0.3f;
            wB = V3{ s.mB[j].x + rf(-off, off), s.mB[j].y + rf(-off, off), s.mB[j].z + rf(-off, off) };
            // newA target = mA[j]; invert transformInv: witnessA = aToB.transform(target)=? use rotate then +p
            // simpler: pick witnessA random; a B-match alone already flags the slot.
            wA = V3{ rf(-2, 2), rf(-2, 2), rf(-2, 2) };
        }
        else
        {
            for (int tries = 0; tries < 40; ++tries)
            {
                wB = V3{ rf(-2.5f, 2.5f), rf(-2.5f, 2.5f), rf(-2.5f, 2.5f) };
                wA = V3{ rf(-2.5f, 2.5f), rf(-2.5f, 2.5f), rf(-2.5f, 2.5f) };
                const V3 nA = qRotInv(s.aToBq, sub(wA, aToBp));
                bool clear = true;
                const float lim = (2.0f * rbt) * (2.0f * rbt);
                for (uint32_t j = 0; j < nc; ++j)
                {
                    const V3 da = sub(nA, v3(s.mA[j])), db = sub(wB, v3(s.mB[j]));
                    if (dot3(da, da) < lim || dot3(db, db) < lim) { clear = false; break; }
                }
                if (clear) break;
            }
        }
        s.witnessA = { wA.x, wA.y, wA.z, 0 };
        s.witnessB = { wB.x, wB.y, wB.z, 0 };
        in[i] = s;

        bool marg = false;
        refDecide(s, ref[i], marg);
        skip[i] = marg ? 1u : 0u;
        if (marg) { ++nSkip; continue; }
        // path bookkeeping uses the reference's realized behaviour
        if (ref[i].fullContactGen) ++fcT; else ++fcF;
        {
            // the cases where only the 5.12 term turns the full gen on
            const F4 n = s.normal; const F4 l = ref[i].localNor;
            const float align = l.x * -n.x + l.y * -n.y + l.z * -n.z;
            if (s.initialNbContacts > 1 && !(0.707106781f > align) && !(ref[i].numContacts < 4)) ++fcJitter;
        }
        if (scen == 1 && nc > 0) ++pMatch; else if (nc < 4) ++pAppend; else ++pReduce;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<DecOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct DecParams { uint32_t numTests; } prm = { N };
    Kernel k("npCvxContactDecide.spv", "cvxContactDecide",
             { (uint32_t)sizeof(DecIn), (uint32_t)sizeof(DecOut) }, sizeof(DecParams));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<DecOut>(dev, bOut, N);
    size_t mmN = 0, mmSlot = 0, mmNor = 0, mmFc = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (skip[i]) continue;
        if (g[i].numContacts != ref[i].numContacts) { ++mmN; continue; }
        if (g[i].fullContactGen != ref[i].fullContactGen) { ++mmFc; }
        if (!eqN(ref[i].localNor, g[i].localNor)) { ++mmNor; }
        for (uint32_t j = 0; j < ref[i].numContacts; ++j)
            if (!eqF4(ref[i].mA[j], g[i].mA[j]) || !eqF4(ref[i].mB[j], g[i].mB[j]) || !eqF4(ref[i].mNp[j], g[i].mNp[j])) { ++mmSlot; break; }
    }
    PXG_CHECK_MSG(mmN == 0, "contactDecide: %zu numContacts mismatches", mmN);
    PXG_CHECK_MSG(mmSlot == 0, "contactDecide: %zu manifold-slot mismatches", mmSlot);
    PXG_CHECK_MSG(mmNor == 0, "contactDecide: %zu localNor mismatches", mmNor);
    PXG_CHECK_MSG(mmFc == 0, "contactDecide: %zu fullContactGen mismatches", mmFc);
    PXG_CHECK_MSG(nSkip < N / 2, "contactDecide: too many marginal (%zu/%u)", nSkip, N);
    PXG_CHECK_MSG(pMatch > N / 20 && pAppend > N / 20 && pReduce > N / 20,
                  "contactDecide: path coverage thin (match %zu / append %zu / reduce %zu)", pMatch, pAppend, pReduce);
    PXG_CHECK_MSG(fcT > N / 20 && fcF > N / 20, "contactDecide: fullContactGen coverage thin (T %zu / F %zu)", fcT, fcF);
    PXG_CHECK_MSG(fcJitter > N / 100, "contactDecide: the 5.12 initialNbContacts > 1 term never decides (%zu)", fcJitter);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
