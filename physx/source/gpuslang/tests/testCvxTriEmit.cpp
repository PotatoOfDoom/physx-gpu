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

// Verification of pxgSlangCvxTriEmit — the tail of convexTriangleContactGen
// (convexTriangle.cuh:1769-1881).
//
// THE REDUCTION ORACLE HERE IS WRITTEN FRESH rather than shared with
// testContactReduce.cpp, and that is on purpose: this one is the
// <false, true, 5, true> specialisation transcribed straight from
// contactReduction.cuh, so the two are independent derivations of the same
// algorithm.  Agreeing with both is a stronger statement than agreeing with one.
//
// THE ATOMICS ARE THE REASON THIS FILE IS SHAPED THE WAY IT IS.  Every pair
// reserves its slice of the temp contact stream through one shared counter, so the
// ORDER of reservations is nondeterministic across groups and no per-pair index
// can be predicted.  What IS deterministic, and what the test asserts:
//   * the final cursor equals the preset base plus the sum of every pair's
//     reserved count -- including the pairs whose contacts were then dropped,
//     because the CUDA never rolls a reservation back;
//   * each pair's payload appears at the index the GPU itself reported, so the
//     slices are checked wherever they landed;
//   * the slices tile the reserved range exactly -- no gap, no overlap;
//   * the second-pass list is compared as a MULTISET.
// The three overflow cases need a predictable startIndex, so they are dispatched
// one pair at a time against a preset cursor.

#include "pxgTestFramework.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;

struct V3 { float x, y, z; };
struct F4 { float x, y, z, w; };

static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b)
{ return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static V3 muls(V3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }

static const float MAXF = 3.4028234663852886e+38f;

static const uint32_t CVX_TRI_MAX_CONTACTS = 5;
static const uint32_t DEFERRED_MASK = 0x80000000u;
static const uint32_t NB_CONTACTS_SHIFT = 27;
static const uint32_t VS_V0 = 1u << 29, VS_V1 = 2u << 29, VS_V2 = 3u << 29;
static const uint32_t VS_E01 = 4u << 29, VS_E02 = 5u << 29, VS_E12 = 6u << 29;

// ---------------------------------------------------------------------------
// contactReduceShared<false, true, 5, true> (contactReduction.cuh:55-165),
// transcribed directly.  maxIndex/minIndex are nputils.cuh:137-163: the winner is
// the lowest lane IN THE SET holding the set's extreme.
// ---------------------------------------------------------------------------
static uint32_t argMaxIn(const float* v, uint32_t mask)
{
    float best = -MAXF; uint32_t idx = 0;
    for (uint32_t j = 0; j < 32; ++j)
        if (((mask >> j) & 1) && v[j] > best) { best = v[j]; idx = j; }
    return idx;
}
static uint32_t argMinIn(const float* v, uint32_t mask)
{
    float best = MAXF; uint32_t idx = 0;
    for (uint32_t j = 0; j < 32; ++j)
        if (((mask >> j) & 1) && v[j] < best) { best = v[j]; idx = j; }
    return idx;
}

static uint32_t refReduce5(const V3* pointRaw, const float* sep, V3 normal, uint32_t allMask)
{
    // dupe cull
    {
        uint32_t ballot = 0;
        for (uint32_t j = 0; j < 32; ++j)
        {
            bool dupe = false;
            for (uint32_t i = 0; i < 32; ++i)
            {
                if (!((allMask >> i) & 1)) continue;
                const V3 dv = sub(pointRaw[i], pointRaw[j]);
                const float d = dot3(dv, dv);
                if (d < 1e-8f && (sep[i] < sep[j] || (sep[i] == sep[j] && i > j)))
                    dupe = true;
            }
            if (dupe) ballot |= 1u << j;
        }
        allMask &= ~ballot;
    }

    if (__builtin_popcount(allMask) <= int(CVX_TRI_MAX_CONTACTS))
        return allMask;

    V3 point[32];
    for (uint32_t j = 0; j < 32; ++j)
        point[j] = sub(pointRaw[j], muls(normal, dot3(pointRaw[j], normal)));

    const uint32_t i0 = argMaxIn(sep, allMask);
    uint32_t mask = 1u << i0;

    float dist[32];
    for (uint32_t j = 0; j < 32; ++j)
    { const V3 dv = sub(point[i0], point[j]); dist[j] = std::sqrt(dot3(dv, dv)); }
    const uint32_t i1 = argMaxIn(dist, allMask & ~mask);
    mask |= 1u << i1;

    const V3 dir = cross3(normal, sub(point[i1], point[i0]));
    float d[32];
    for (uint32_t j = 0; j < 32; ++j) d[j] = dot3(dir, sub(point[j], point[i0]));

    const uint32_t f = argMaxIn(d, allMask & ~mask); mask |= 1u << f;
    const uint32_t g = argMinIn(d, allMask & ~mask); mask |= 1u << g;

    if (__builtin_popcount(mask) == 4)
        mask |= 1u << argMinIn(sep, allMask & ~mask);

    for (int i = __builtin_popcount(mask); i < int(CVX_TRI_MAX_CONTACTS); ++i)
        mask |= 1u << argMinIn(sep, allMask & ~mask);

    return mask;
}

// convexTriangle.cuh:1538 / :1562
static uint32_t refActiveMask(V3 p, V3 a, V3 b, V3 c)
{
    const V3 n = cross3(sub(b, a), sub(c, a));
    const V3 ap = sub(a, p), bp = sub(b, p), cp = sub(c, p);
    const float va = dot3(n, cross3(bp, cp));
    const float vb = dot3(n, cross3(cp, ap));
    const float vc = dot3(n, cross3(ap, bp));
    const float total = va + vb + vc;
    const float denom = total == 0.0f ? 0.0f : 1.0f / total;
    const float v = vb * denom, w = vc * denom;

    const float ub = 0.97f;
    const float u = 1.0f - v - w;
    if (v > ub) return VS_V1;
    if (w > ub) return VS_V2;
    if (u > ub) return VS_V0;
    if ((v + u) > ub) return VS_E01;
    if ((u + w) > ub) return VS_E02;
    if ((w + v) > ub) return VS_E12;
    return 0;
}

// ---------------------------------------------------------------------------
struct EmitCase
{
    bool sat = true;
    bool delay = false;
    uint32_t nbGen = 0;
    F4 staged[32]{};
    V3 minNormal{ 0, 0, 1 };
    V3 t0{}, t1{}, t2{};
    uint32_t pairOffset = 0, paddedOffset = 0, remapIdx = 0, triIdx = 0, warpIdx = 0;
    uint32_t tempContactSize = 1u << 20;
    const char* name = "";
};

struct EmitRef
{
    uint32_t mask = 0;
    uint32_t nbOut = 0;
    float maxSep = MAXF;
    bool delayFinal = false;
    uint32_t activeMask = 0;
    bool wroteIndex = false;
    std::vector<F4> payload;   // in scan order
};

static EmitRef refEmit(const EmitCase& c)
{
    EmitRef r;
    if (c.sat && c.nbGen != 0)
    {
        V3 pa[32]; float sep[32];
        for (uint32_t j = 0; j < 32; ++j)
        {
            const bool live = j < c.nbGen;
            pa[j] = live ? V3{ c.staged[j].x, c.staged[j].y, c.staged[j].z } : V3{ 0, 0, 0 };
            sep[j] = live ? c.staged[j].w : 0.0f;
        }

        uint32_t mask = (c.nbGen >= 32) ? 0xffffffffu : ((1u << c.nbGen) - 1u);
        if (c.nbGen > CVX_TRI_MAX_CONTACTS)
            mask = refReduce5(pa, sep, c.minNormal, mask);
        r.mask = mask;

        float m = MAXF;
        for (uint32_t j = 0; j < 32; ++j)
            if ((mask >> j) & 1) m = std::fmin(m, sep[j]);
        r.maxSep = m;

        r.nbOut = uint32_t(__builtin_popcount(mask));
        r.wroteIndex = true;
        for (uint32_t j = 0; j < 32; ++j)
            if ((mask >> j) & 1)
                r.payload.push_back(F4{ pa[j].x, pa[j].y, pa[j].z, sep[j] });
    }
    r.delayFinal = c.sat && c.delay && (c.nbGen != 0);
    if (r.delayFinal)
        r.activeMask = refActiveMask(V3{ c.staged[0].x, c.staged[0].y, c.staged[0].z },
                                     c.t0, c.t1, c.t2);
    return r;
}

struct EmitPush
{
    uint64_t input, cvxTriNI, cvxTriContacts, cvxTriMaxDepth, cvxTriInterm;
    uint64_t orderedCvxTriInterm, convTriSecondPassPairs, nbSecondPassPairs;
    uint64_t tempConvexTriContacts, pTempContactIndex;
    uint32_t numPairs, pad;
};

static void packCase(float* r, const EmitCase& c)
{
    uint32_t* u = reinterpret_cast<uint32_t*>(r);
    u[0] = c.sat ? 1u : 0u; u[1] = c.delay ? 1u : 0u; u[2] = c.nbGen; u[3] = c.tempContactSize;
    u[4] = c.pairOffset; u[5] = c.paddedOffset; u[6] = c.remapIdx; u[7] = c.triIdx;
    u[8] = c.warpIdx;
    r[12] = c.minNormal.x; r[13] = c.minNormal.y; r[14] = c.minNormal.z;
    r[16] = c.t0.x; r[17] = c.t0.y; r[18] = c.t0.z;
    r[20] = c.t1.x; r[21] = c.t1.y; r[22] = c.t1.z;
    r[24] = c.t2.x; r[25] = c.t2.y; r[26] = c.t2.z;
    for (uint32_t j = 0; j < 32; ++j)
    {
        float* q = r + 32 + 4 * j;
        q[0] = c.staged[j].x; q[1] = c.staged[j].y; q[2] = c.staged[j].z; q[3] = c.staged[j].w;
    }
}

// A triangle far from degenerate, and a point placed at a chosen barycentric so
// the Voronoi classification is unambiguous.
static const V3 TRI0 = { 0.0f, 0.0f, 0.0f };
static const V3 TRI1 = { 2.0f, 0.0f, 0.0f };
static const V3 TRI2 = { 0.0f, 2.0f, 0.0f };
static V3 baryPoint(float v, float w)
{
    const float u = 1.0f - v - w;
    return { u * TRI0.x + v * TRI1.x + w * TRI2.x,
             u * TRI0.y + v * TRI1.y + w * TRI2.y,
             u * TRI0.z + v * TRI1.z + w * TRI2.z };
}

// Coverage bookkeeping, all reference-side and therefore deterministic.  Without
// these a fixture in which every case took the same path would still pass.
struct Cover
{
    size_t shift32 = 0, reduced = 0, overflowDropped = 0, slot0Discarded = 0;
    size_t satFail = 0, deferredDemoted = 0, deferredLive = 0, indexNotWritten = 0;
    size_t region[7] = { 0, 0, 0, 0, 0, 0, 0 };   // V0 V1 V2 E01 E02 E12 interior
};
static Cover gCover;

static void note(const EmitCase& c, const EmitRef& e, bool fits)
{
    if (c.sat && c.nbGen == 32u) ++gCover.shift32;
    if (c.sat && c.nbGen > CVX_TRI_MAX_CONTACTS) ++gCover.reduced;
    if (e.wroteIndex && !fits) ++gCover.overflowDropped;
    if (e.wroteIndex && c.nbGen > CVX_TRI_MAX_CONTACTS && ((e.mask & 1u) == 0u))
        ++gCover.slot0Discarded;
    if (!c.sat) ++gCover.satFail;
    if (c.delay && !e.delayFinal) ++gCover.deferredDemoted;
    if (e.delayFinal) ++gCover.deferredLive;
    if (!e.wroteIndex) ++gCover.indexNotWritten;
    if (e.delayFinal)
    {
        switch (e.activeMask)
        {
        case VS_V0: ++gCover.region[0]; break;
        case VS_V1: ++gCover.region[1]; break;
        case VS_V2: ++gCover.region[2]; break;
        case VS_E01: ++gCover.region[3]; break;
        case VS_E02: ++gCover.region[4]; break;
        case VS_E12: ++gCover.region[5]; break;
        default: ++gCover.region[6]; break;
        }
    }
}

// Runs one batch of cases and checks every output.  `base` is the value the temp
// cursor is preset to.
static void runBatch(PxgSlangComputeDevice* dev, const std::vector<EmitCase>& cases,
                     uint32_t base, const char* what, size_t& bad)
{
    const uint32_t n = uint32_t(cases.size());

    uint32_t maxPair = 0, maxPadded = 0;
    for (const EmitCase& c : cases)
    { maxPair = std::max(maxPair, c.pairOffset); maxPadded = std::max(maxPadded, c.paddedOffset); }

    std::vector<float> input(size_t(n) * (640 / 4), 0.0f);
    for (uint32_t i = 0; i < n; ++i)
        packCase(input.data() + size_t(i) * (640 / 4), cases[i]);

    const uint32_t NI_WORDS = 4 * (maxPair + 1);
    const uint32_t TEMP_SLOTS = 4096;

    std::vector<uint32_t> niInit(NI_WORDS, 0xdeadbeefu);
    std::vector<uint32_t> contactsInit(maxPair + 1, 0xdeadbeefu);
    std::vector<uint32_t> depthInit(maxPair + 1, 0xdeadbeefu);
    std::vector<uint32_t> intermInit(maxPair + 1, 0xdeadbeefu);
    std::vector<uint32_t> orderedInit(maxPadded + 1, 0xdeadbeefu);
    std::vector<uint32_t> secondInit(n + 4, 0xdeadbeefu);
    std::vector<uint32_t> nbSecondInit(1, 0);
    std::vector<F4> tempInit(TEMP_SLOTS, F4{ -1e30f, -1e30f, -1e30f, -1e30f });
    std::vector<uint32_t> cursorInit(1, base);

    PxgBufferHandle bIn = createBufferFrom(dev, input);
    PxgBufferHandle bNI = createBufferFrom(dev, niInit);
    PxgBufferHandle bCt = createBufferFrom(dev, contactsInit);
    PxgBufferHandle bDp = createBufferFrom(dev, depthInit);
    PxgBufferHandle bIt = createBufferFrom(dev, intermInit);
    PxgBufferHandle bOr = createBufferFrom(dev, orderedInit);
    PxgBufferHandle bSp = createBufferFrom(dev, secondInit);
    PxgBufferHandle bNs = createBufferFrom(dev, nbSecondInit);
    PxgBufferHandle bTc = createBufferFrom(dev, tempInit);
    PxgBufferHandle bCu = createBufferFrom(dev, cursorInit);

    EmitPush push{};
    push.input = dev->getBufferDeviceAddress(bIn);
    push.cvxTriNI = dev->getBufferDeviceAddress(bNI);
    push.cvxTriContacts = dev->getBufferDeviceAddress(bCt);
    push.cvxTriMaxDepth = dev->getBufferDeviceAddress(bDp);
    push.cvxTriInterm = dev->getBufferDeviceAddress(bIt);
    push.orderedCvxTriInterm = dev->getBufferDeviceAddress(bOr);
    push.convTriSecondPassPairs = dev->getBufferDeviceAddress(bSp);
    push.nbSecondPassPairs = dev->getBufferDeviceAddress(bNs);
    push.tempConvexTriContacts = dev->getBufferDeviceAddress(bTc);
    push.pTempContactIndex = dev->getBufferDeviceAddress(bCu);
    push.numPairs = n;

    {
        Kernel k("cvxTriEmitBda.spv", "cvxTriEmitLaunch", {}, sizeof(EmitPush));
        k.dispatch(n, {}, push, 1u, 1u);
        dev->flush();
    }

    const auto gNI = downloadBuffer<uint32_t>(dev, bNI, NI_WORDS);
    const auto gCt = downloadBuffer<uint32_t>(dev, bCt, maxPair + 1);
    const auto gDp = downloadBuffer<uint32_t>(dev, bDp, maxPair + 1);
    const auto gIt = downloadBuffer<uint32_t>(dev, bIt, maxPair + 1);
    const auto gOr = downloadBuffer<uint32_t>(dev, bOr, maxPadded + 1);
    const auto gSp = downloadBuffer<uint32_t>(dev, bSp, n + 4);
    const auto gNs = downloadBuffer<uint32_t>(dev, bNs, 1);
    const auto gTc = downloadBuffer<F4>(dev, bTc, TEMP_SLOTS);
    const auto gCu = downloadBuffer<uint32_t>(dev, bCu, 1);

    auto sameF = [](float a, float b)
    { uint32_t x, y; std::memcpy(&x, &a, 4); std::memcpy(&y, &b, 4); return x == y; };
    auto fail = [&](const char* nm, const char* w, uint32_t g, uint32_t e)
    {
        ++bad;
        if (bad <= 12) std::printf("  [%s/%s] %s: got=%08x exp=%08x\n", what, nm, w, g, e);
    };

    uint32_t reservedTotal = 0;
    std::vector<uint32_t> refSecond;
    std::vector<std::pair<uint32_t, uint32_t>> slices;   // (index, count) of what was written

    for (uint32_t i = 0; i < n; ++i)
    {
        const EmitCase& c = cases[i];
        const EmitRef e = refEmit(c);
        reservedTotal += e.nbOut > 0 || e.wroteIndex ? uint32_t(e.payload.size()) : 0u;

        // maxDepth
        float gd; std::memcpy(&gd, &gDp[c.pairOffset], 4);
        if (!sameF(gd, e.maxSep)) fail(c.name, "maxDepth", gDp[c.pairOffset], 0);

        // intermediate
        if (gIt[c.pairOffset] != c.triIdx) fail(c.name, "interm", gIt[c.pairOffset], c.triIdx);

        // THE EXPECTED CONTACT COUNT IS DERIVED, NOT READ BACK.  An earlier version
        // recovered it from the GPU's own NI word by masking and shifting, which is
        // circular -- and it breaks on exactly the case that matters: the NI word is
        // an ADDITION, so a remap index at or above 2^27 CARRIES into the count
        // field and the read-back count is one too high.  The overflow verdict is
        // still taken from the index the GPU reported, because the reservation
        // order across groups is not predictable, but everything else is computed
        // from the reference.
        const uint32_t reportedIdx = gCt[c.pairOffset];
        const bool fitsHere = e.wroteIndex
                              && (reportedIdx + e.payload.size()) < c.tempContactSize;
        const uint32_t expNb = fitsHere ? e.nbOut : 0u;

        const uint32_t expFlag = (expNb != 0 && !e.delayFinal) ? (1u << 31) : 0u;
        if (gOr[c.paddedOffset] != (expFlag | c.triIdx))
            fail(c.name, "ordered", gOr[c.paddedOffset], expFlag | c.triIdx);

        // normal + index record
        float nx, ny, nz;
        std::memcpy(&nx, &gNI[4 * c.pairOffset + 0], 4);
        std::memcpy(&ny, &gNI[4 * c.pairOffset + 1], 4);
        std::memcpy(&nz, &gNI[4 * c.pairOffset + 2], 4);
        if (!sameF(nx, c.minNormal.x) || !sameF(ny, c.minNormal.y) || !sameF(nz, c.minNormal.z))
            fail(c.name, "NI.normal", 0, 0);
        const uint32_t expIndex = (e.delayFinal ? DEFERRED_MASK : 0u)
                                  + (expNb << NB_CONTACTS_SHIFT) + c.remapIdx;
        if (gNI[4 * c.pairOffset + 3] != expIndex)
            fail(c.name, "NI.index", gNI[4 * c.pairOffset + 3], expIndex);

        if (e.delayFinal)
            refSecond.push_back(c.warpIdx | e.activeMask);

        // the payload, wherever the GPU said it put it
        if (!e.wroteIndex)
        {
            note(c, e, true);
            if (gCt[c.pairOffset] != 0xdeadbeefu)
                fail(c.name, "index written when it should not be", gCt[c.pairOffset], 0xdeadbeefu);
        }
        else
        {
            const uint32_t idx = reportedIdx;
            slices.push_back({ idx, uint32_t(e.payload.size()) });
            note(c, e, fitsHere);
            if (fitsHere)
            {
                for (size_t k = 0; k < e.payload.size(); ++k)
                {
                    const F4 g = gTc[idx + k], x = e.payload[k];
                    if (!sameF(g.x, x.x) || !sameF(g.y, x.y) || !sameF(g.z, x.z) || !sameF(g.w, x.w))
                        fail(c.name, "payload", uint32_t(k), 0);
                }
            }
        }
    }

    // the cursor advanced by every reservation, including the dropped ones
    if (gCu[0] != base + reservedTotal)
        fail("batch", "cursor", gCu[0], base + reservedTotal);

    // the slices tile the reserved range exactly
    std::sort(slices.begin(), slices.end());
    uint32_t cur = base;
    for (const auto& s : slices)
    {
        if (s.first != cur) fail("batch", "slice start", s.first, cur);
        cur += s.second;
    }

    // second-pass list as a multiset
    if (gNs[0] != refSecond.size()) fail("batch", "nbSecondPass", gNs[0], uint32_t(refSecond.size()));
    else
    {
        std::vector<uint32_t> got(gSp.begin(), gSp.begin() + gNs[0]);
        std::sort(got.begin(), got.end());
        std::sort(refSecond.begin(), refSecond.end());
        for (size_t k = 0; k < got.size(); ++k)
            if (got[k] != refSecond[k]) fail("batch", "secondPass", got[k], refSecond[k]);
    }

    dev->destroyBuffer(bIn); dev->destroyBuffer(bNI); dev->destroyBuffer(bCt);
    dev->destroyBuffer(bDp); dev->destroyBuffer(bIt); dev->destroyBuffer(bOr);
    dev->destroyBuffer(bSp); dev->destroyBuffer(bNs); dev->destroyBuffer(bTc);
    dev->destroyBuffer(bCu);
}

PXG_TEST(cvxTriEmitTail)
{
    PxgSlangComputeDevice* dev = testDevice();
    size_t bad = 0;

    // A spread of contacts on the triangle's plane, distinct and far from ties.
    auto fill = [](EmitCase& c, uint32_t n) {
        for (uint32_t j = 0; j < 32; ++j)
        {
            const float a = 0.19634954f * float(j);
            c.staged[j] = F4{ 0.6f + 0.4f * std::cos(a), 0.6f + 0.4f * std::sin(a),
                              0.01f * float(j), -0.02f - 0.013f * float((j * 7u) % 32u) };
        }
        c.nbGen = n;
        c.minNormal = V3{ 0.0f, 0.0f, 1.0f };
        c.t0 = TRI0; c.t1 = TRI1; c.t2 = TRI2;
    };

    std::vector<EmitCase> cases;
    uint32_t pair = 0;
    auto add = [&](EmitCase c, const char* nm) {
        c.pairOffset = pair;
        // The padded offset is DELIBERATELY not the pair offset: the two index
        // different arrays in the CUDA and a port that confuses them still looks
        // right when they are equal.
        c.paddedOffset = pair + 3;
        c.remapIdx = 1000 + pair;
        c.triIdx = 500 + pair;
        c.warpIdx = 70000 + 64 * pair;
        // A DISTINCT, NON-AXIS-ALIGNED NORMAL PER CASE.  With (0,0,1) everywhere
        // the x and y words of the ConvexTriNormalAndIndex record are both zero,
        // so permuting them in the four-lane cooperative write is invisible.
        {
            const float a = 0.7f + 0.11f * float(pair);
            V3 nn = { std::cos(a), 0.35f * std::sin(a), 0.6f + 0.05f * float(pair % 5) };
            const float inv = 1.0f / std::sqrt(dot3(nn, nn));
            c.minNormal = muls(nn, inv);
        }
        c.name = nm;
        cases.push_back(c);
        ++pair;
    };

    // ---- contact counts, including both sides of the reduction gate ----------
    for (uint32_t n : { 0u, 1u, 5u, 6u, 31u, 32u })
    {
        EmitCase c; fill(c, n);
        static const char* names[] = { "n=0", "n=1", "n=5", "n=6", "n=31", "n=32" };
        static const uint32_t idx[] = { 0, 1, 5, 6, 31, 32 };
        const char* nm = "n";
        for (int k = 0; k < 6; ++k) if (idx[k] == n) nm = names[k];
        add(c, nm);
    }

    // ---- the SAT failure: no contacts, but the records are still written -----
    {
        EmitCase c; fill(c, 12u); c.sat = false; c.delay = true;
        add(c, "SAT failed");
    }

    // ---- delayContacts, all three ways --------------------------------------
    {   // deferred with contacts: appends to the second-pass list
        EmitCase c; fill(c, 9u); c.delay = true;
        c.staged[0] = F4{ baryPoint(0.99f, 0.005f).x, baryPoint(0.99f, 0.005f).y, 0.0f, -0.3f };
        add(c, "deferred, v1 corner");
    }
    {   // deferred but the clipper produced nothing: :1823 demotes it
        EmitCase c; fill(c, 0u); c.delay = true;
        add(c, "deferred but n=0");
    }

    // ---- the Voronoi classification, one case per region + the interior -----
    struct { float v, w; const char* nm; } regions[] = {
        { 0.99f, 0.005f, "V1 corner" },
        { 0.005f, 0.99f, "V2 corner" },
        { 0.005f, 0.005f, "V0 corner" },
        { 0.50f, 0.005f, "E01 edge" },
        { 0.005f, 0.50f, "E02 edge" },
        { 0.50f, 0.49f, "E12 edge" },
        { 0.33f, 0.33f, "interior, mask 0" },
    };
    for (const auto& reg : regions)
    {
        EmitCase c; fill(c, 7u); c.delay = true;
        const V3 p = baryPoint(reg.v, reg.w);
        c.staged[0] = F4{ p.x, p.y, 0.0f, -0.4f };
        add(c, reg.nm);
    }

    // ---- the reduction discards raw slot 0 ----------------------------------
    // Slot 0 must be UNREMARKABLE, not extreme.  The first attempt made it the
    // shallowest contact, which makes it p0 -- the deepest-anchor pick is an
    // argMAX over the separation -- so it was the first thing KEPT.  Here it sits
    // at the centre of the ring, so it is no geometric extreme, and at a middling
    // separation, so neither the PCM pick nor the deepest-first fill reaches it.
    // Twenty contacts reduce to five, and slot 0 is none of them -- yet the
    // second-pass classifier must still read it.
    {
        EmitCase c; fill(c, 20u); c.delay = true;
        c.staged[0] = F4{ 0.6f, 0.6f, 0.15f, -0.2f };
        add(c, "reduction discards slot 0");
    }

    // ---- five contacts INCLUDING A DUPLICATE ---------------------------------
    // The reduction gate is `> CVX_TRI_MAX_CONTACTS`, and at exactly five an
    // `>=` spelling would still enter and let the dupe cull drop one.  With no
    // duplicate present the two spellings agree, so the duplicate is the point.
    {
        EmitCase c; fill(c, 5u);
        c.staged[3] = c.staged[1];
        c.staged[3].w = c.staged[1].w - 0.05f;    // deeper, so lane 1 is the one culled
        add(c, "five contacts with a duplicate");
    }

    // ---- the deepest contact on a HIGH lane ----------------------------------
    // The depth reduction is an XOR butterfly, and its widest stage is the only
    // one that crosses between the low and high halves of the warp.  With the
    // deepest contact always below lane 16 -- which the default separation ramp
    // happens to guarantee -- dropping that stage changes nothing.
    {
        EmitCase c; fill(c, 30u);
        for (uint32_t j = 0; j < 30; ++j) c.staged[j].w = -0.05f - 0.002f * float(j);
        c.staged[27].w = -9.0f;                   // by far the deepest, high lane
        add(c, "deepest contact on lane 27");
    }

    // ---- separations ALL POSITIVE --------------------------------------------
    // maxSeparation is FLT_MAX on the lanes the reduction did not select, and the
    // lanes past the contact count carry a zeroed separation.  While every
    // separation is negative the unselected lanes cannot win the minimum, so
    // taking the min over ALL lanes agrees with taking it over the selected ones.
    // Positive separations break that agreement.
    {
        EmitCase c; fill(c, 12u);
        for (uint32_t j = 0; j < 12; ++j) c.staged[j].w = 0.5f + 0.03f * float(j);
        add(c, "all separations positive");
    }

    // ---- a DEGENERATE triangle ------------------------------------------------
    // barycentricCoordinates guards a zero total area by setting the reciprocal
    // to zero rather than dividing.  Without a zero-area triangle the guard is
    // never taken and removing it changes nothing.
    {
        EmitCase c; fill(c, 7u); c.delay = true;
        c.t1 = c.t0;                              // zero area
        add(c, "degenerate triangle");
    }

    // ---- a triangle index that reaches the count field ------------------------
    // The NI word is an ADDITION of three fields, not an OR, so a remap index at
    // or above 2^27 CARRIES into the four-bit contact count.  That is the
    // documented behaviour of the original and the only case that distinguishes
    // the two spellings.
    {
        EmitCase c; fill(c, 9u); c.delay = true;
        add(c, "remap index carries into the count");
        cases.back().remapIdx = (1u << 27) + 12345u;
    }

    runBatch(dev, cases, 64u, "main", bad);

    // ---- the temp-stream boundary, one pair per dispatch so startIndex is
    //      predictable ---------------------------------------------------------
    {   // startIndex + nb == tempContactSize: STRICTLY `<` fails, nothing written
        EmitCase c; fill(c, 9u);
        c.pairOffset = 0; c.paddedOffset = 0; c.remapIdx = 7; c.triIdx = 11; c.warpIdx = 99;
        c.tempContactSize = 100u + CVX_TRI_MAX_CONTACTS;   // base 100 + 5 == size
        c.name = "overflow at exactly the size";
        runBatch(dev, { c }, 100u, "boundary-eq", bad);
    }
    {   // one slot lower: it fits
        EmitCase c; fill(c, 9u);
        c.pairOffset = 0; c.paddedOffset = 0; c.remapIdx = 7; c.triIdx = 11; c.warpIdx = 99;
        c.tempContactSize = 100u + CVX_TRI_MAX_CONTACTS + 1u;
        c.name = "one below the boundary";
        runBatch(dev, { c }, 100u, "boundary-lt", bad);
    }
    {   // the cursor is already past the end: the reservation is NOT rolled back
        //  and .index keeps the out-of-range value
        EmitCase c; fill(c, 9u); c.delay = true;
        c.pairOffset = 0; c.paddedOffset = 0; c.remapIdx = 7; c.triIdx = 11; c.warpIdx = 99;
        c.tempContactSize = 8u;
        c.name = "cursor already past the end";
        runBatch(dev, { c }, 500u, "past-end", bad);
    }

    PXG_CHECK_MSG(bad == 0, "cvxTriEmit: %zu mismatches", bad);

    PXG_CHECK_MSG(gCover.shift32 > 0, "no case has 32 contacts (the 1<<32 shift is untested)");
    PXG_CHECK_MSG(gCover.reduced > 0, "no case runs the reduction");
    PXG_CHECK_MSG(gCover.overflowDropped > 0, "no case overflows the temp stream");
    PXG_CHECK_MSG(gCover.slot0Discarded > 0,
                  "no case has the reduction discard raw slot 0 (the classifier's input)");
    PXG_CHECK_MSG(gCover.satFail > 0, "no case fails the SAT");
    PXG_CHECK_MSG(gCover.deferredDemoted > 0, "no deferred pair is demoted by the raw count");
    PXG_CHECK_MSG(gCover.deferredLive > 0, "no pair reaches the second-pass list");
    PXG_CHECK_MSG(gCover.indexNotWritten > 0, "no case leaves .index unwritten");
    for (int k = 0; k < 7; ++k)
        PXG_CHECK_MSG(gCover.region[k] > 0, "Voronoi region %d never classified", k);
}

} // namespace
