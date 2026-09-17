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

// Verification of ttpReduceContactsSerial (pxgSlangTriTriPatch.slang) against a
// CPU mirror of reduceContactsSingleWarp (convexMeshMidphase.cu:440-503).
//
// THE MIRROR IS WRITTEN FROM THE CUDA.  In particular:
//   :457  readMask = ~currentContactMask            -- FULL 32-bit complement
//   :459  readIndex = c + popcount(readMask below lane)
//   :461  readIndex < totalContacts                 -- STRICT
//   :469  c += popcount(readMask)                   -- LANES OFFERED, not taken
//   :471  ballot(hasContact), which is sticky within a pass
//   :472  contactReduce2<true, true, 6, false>      -- WRAPPER 2
//   :478  minIndex over the survivor mask, FLT_MAX outside
//   :492  the compaction scan is over currentContactMask, NOT readMask
//
// AND IT ASSERTS THE LOOP ACTUALLY LOOPS.  The reduction itself is already
// certified by gpuslang.npCrsWrapper2; what is new here is the refill discipline
// around it.  A case set whose contacts all fit in one pass would exercise none
// of it and would still pass, so the fixture counts passes and fails if the
// average is not well above one.

#include "pxgTestFramework.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct V3 { float x, y, z; };
struct V4 { float x, y, z, w; };

static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b)
{
    return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}
static V3 scale3(V3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
static float mag3(V3 a) { return std::sqrt(dot3(a, a)); }

static const float FLT_MAX_LOCAL = 3.402823466e38f;
static uint32_t popcount32(uint32_t v) { uint32_t c = 0; while (v) { c += v & 1u; v >>= 1; } return c; }

static uint32_t argMax(const float* vals, uint32_t mask, float& best)
{
    best = -FLT_MAX_LOCAL;
    uint32_t idx = 0;
    for (uint32_t j = 0; j < 32; ++j)
        if (((mask >> j) & 1) != 0 && vals[j] > best) { best = vals[j]; idx = j; }
    return idx;
}
static uint32_t argMin(const float* vals, uint32_t mask, float& best)
{
    best = FLT_MAX_LOCAL;
    uint32_t idx = 0;
    for (uint32_t j = 0; j < 32; ++j)
        if (((mask >> j) & 1) != 0 && vals[j] < best) { best = vals[j]; idx = j; }
    return idx;
}

// contactReduceShared<true, true, 6, false> (contactReduction.cuh:55-165), the
// clustering arm only -- this instantiation always clusters.
static uint32_t reduceShared(const V3* ptRaw, const float* sep, V3 normal,
                             uint32_t allMask, float clusterBias,
                             const float* adj, const float* initial, int maxPoints)
{
    V3 point[32];
    for (uint32_t j = 0; j < 32; ++j)
        point[j] = sub(ptRaw[j], scale3(normal, dot3(ptRaw[j], normal)));

    if (popcount32(allMask) <= uint32_t(maxPoints))
        return allMask;

    float v, vals[32];
    for (uint32_t j = 0; j < 32; ++j) vals[j] = initial[j] + adj[j];       // :91
    const uint32_t i0 = argMax(vals, allMask, v);
    uint32_t mask = 1u << i0;

    for (uint32_t j = 0; j < 32; ++j)                                      // :93-94
        vals[j] = mag3(sub(point[i0], point[j])) + adj[j];
    const uint32_t i1 = argMax(vals, allMask & ~mask, v);
    mask |= 1u << i1;

    const V3 dir = cross3(normal, sub(point[i1], point[i0]));              // :105
    float dval[32];
    for (uint32_t j = 0; j < 32; ++j) dval[j] = dot3(dir, sub(point[j], point[i0]));

    for (uint32_t j = 0; j < 32; ++j) vals[j] = dval[j] + adj[j];          // :107
    const uint32_t f = argMax(vals, allMask & ~mask, v);
    mask |= 1u << f;

    for (uint32_t j = 0; j < 32; ++j) vals[j] = dval[j] - adj[j];          // :110 SIGN FLIP
    const uint32_t g = argMin(vals, allMask & ~mask, v);
    mask |= 1u << g;

    if (popcount32(mask) == 4 && maxPoints > 4)                            // :121-128
    {
        const uint32_t i4 = argMin(sep, allMask & ~mask, v);
        mask |= 1u << i4;
    }

    {                                                                      // :135-158
        float sepB[32];
        for (uint32_t j = 0; j < 32; ++j)
            sepB[j] = sep[j] - (((mask >> j) & 1) != 0 ? clusterBias : 0.0f);

        int label[32];
        int nbClusters = 0;
        {
            uint32_t m = mask;
            float tbest[32];
            for (uint32_t j = 0; j < 32; ++j) { tbest[j] = FLT_MAX_LOCAL; label[j] = -1; }
            for (; m != 0; ++nbClusters, m &= (m - 1))
            {
                uint32_t a = 0;
                while (((m >> a) & 1u) == 0u) ++a;
                for (uint32_t j = 0; j < 32; ++j)
                {
                    const V3 dv = sub(point[j], point[a]);
                    const float d = dot3(dv, dv);
                    if (d < tbest[j]) { tbest[j] = d; label[j] = nbClusters; }
                }
            }
        }

        mask = 0;
        for (int i = 0; i < nbClusters; ++i)
        {
            uint32_t cl = 0;
            for (uint32_t j = 0; j < 32; ++j) if (label[j] == i) cl |= 1u << j;
            cl &= allMask;
            if (cl) mask |= 1u << argMin(sepB, cl, v);
        }
        for (int i = nbClusters; i < maxPoints; ++i)
            mask |= 1u << argMin(sepB, allMask & ~mask, v);
    }
    return mask;
}

// contactReduce2 (contactReduction.cuh:180-188): adj = -separation,
// initial = |pointRaw projected onto the normal plane|.
static uint32_t contactReduce2(const V3* pt, const float* sep, V3 n, uint32_t allMask, float bias)
{
    float adj[32], initial[32];
    for (uint32_t j = 0; j < 32; ++j)
    {
        const V3 p = sub(pt[j], scale3(n, dot3(pt[j], n)));
        adj[j] = -sep[j];
        initial[j] = mag3(p);
    }
    return reduceShared(pt, sep, n, allMask, bias, adj, initial, 6);
}

struct RefResult
{
    V4 keepPoints[6];
    uint32_t pointCount;
    float deepest;
    uint32_t survivorMask;
    uint32_t passes;
};

// reduceContactsSingleWarp (convexMeshMidphase.cu:440-503), serialised.
static RefResult refReduce(const V4* tempPoints, uint32_t totalContacts, V3 nor, float clusterBias)
{
    V3 pA[32];
    float sep[32];
    for (uint32_t j = 0; j < 32; ++j) { pA[j] = { 0, 0, 0 }; sep[j] = FLT_MAX_LOCAL; }

    uint32_t currentContactMask = 0, hasMask = 0;
    const uint32_t maxCounter = 100;
    uint32_t counter2 = 0;

    for (uint32_t c = 0; c < totalContacts && counter2 < maxCounter; )
    {
        ++counter2;                                        // :455
        const uint32_t readMask = ~currentContactMask;     // :457
        for (uint32_t l = 0; l < 32; ++l)
        {
            const bool needsContact = ((readMask >> l) & 1u) != 0u;   // :458, both arms
            const uint32_t readIndex = c + popcount32(readMask & ((1u << l) - 1u));  // :459
            if (needsContact && readIndex < totalContacts)  // :461
            {
                pA[l] = { tempPoints[readIndex].x, tempPoints[readIndex].y, tempPoints[readIndex].z };
                sep[l] = tempPoints[readIndex].w;
                hasMask |= 1u << l;                        // :466
            }
        }
        c += popcount32(readMask);                         // :469, LANES OFFERED
        currentContactMask = hasMask;                      // :471
        currentContactMask = contactReduce2(pA, sep, nor, currentContactMask, clusterBias); // :472
        hasMask = currentContactMask;                      // :473
    }

    RefResult r{};
    r.passes = counter2;

    float deepest = FLT_MAX_LOCAL;                         // :477-478
    for (uint32_t j = 0; j < 32; ++j)
        if (((currentContactMask >> j) & 1u) != 0u && sep[j] < deepest) deepest = sep[j];
    r.deepest = deepest;

    r.pointCount = popcount32(currentContactMask);         // :481
    r.survivorMask = currentContactMask;
    for (uint32_t k = 0; k < 6; ++k) r.keepPoints[k] = { 0, 0, 0, 0 };

    for (uint32_t l = 0; l < 32; ++l)                      // :491-495
    {
        if (((currentContactMask >> l) & 1u) == 0u) continue;
        const uint32_t off = popcount32(currentContactMask & ((1u << l) - 1u));
        if (off < 6) r.keepPoints[off] = { pA[l].x, pA[l].y, pA[l].z, sep[l] };
    }
    return r;
}

struct TtrParams { uint32_t numCases, pad0, pad1, pad2; };
struct TtrCase
{
    V4 pts[198];
    V4 normal;      // xyz = normal, w = clusterBias
    uint32_t misc[4];
};

PXG_TEST(npTriTriReduce_stream)
{
    std::mt19937 rng(5512099);
    const uint32_t N = 1200;
    auto rf = [&](float lo, float hi)
    { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<TtrCase> in(N);
    std::vector<RefResult> ref(N);
    size_t totalPasses = 0, multiPass = 0, fullBuffer = 0, shortTail = 0, withDupes = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        TtrCase c{};

        V3 n{ rf(-1, 1), rf(-1, 1), rf(-1, 1) };
        const float nl = mag3(n);
        n = nl > 1e-4f ? scale3(n, 1.0f / nl) : V3{ 0.371391f, 0.557086f, 0.742781f };
        const float bias = rf(1e-5f, 5e-3f);
        c.normal = { n.x, n.y, n.z, bias };

        // THE COUNT DRIVES THE PASS COUNT, so it is spread across the whole range
        // rather than sampled uniformly: 198 is the buffer bound the CUDA asserts,
        // and a SHORT tail (a count that is not a multiple of the lanes offered) is
        // the only input that exercises `c += popcount(readMask)` overshooting.
        uint32_t total;
        const uint32_t pick = i % 6u;
        if (pick == 0u) { total = 198u; ++fullBuffer; }
        else if (pick == 1u) { total = 3u + (rng() % 5u); ++shortTail; }   // fits in one pass
        else if (pick == 2u) { total = 33u + (rng() % 3u); ++shortTail; }  // just past one refill
        else total = 7u + (rng() % 192u);

        c.misc[0] = total;

        // An off-centre cloud: |projected point| is a distance from the ORIGIN and
        // it is wrapper 2's initial criterion, so a cloud centred on the origin
        // makes every lane's criterion nearly equal -- the Class D blind spot
        // npCrsWrapper2 documents.
        const V3 centre{ rf(-6, 6), rf(-6, 6), rf(-6, 6) };
        std::vector<V4> pts(198);
        for (uint32_t j = 0; j < 198u; ++j)
        {
            pts[j] = { centre.x + rf(-2, 2), centre.y + rf(-2, 2), centre.z + rf(-2, 2),
                       rf(-0.4f, 0.25f) };
        }

        // COINCIDENT STAGING POINTS, because the real buffer has them and a random
        // cloud never does.  tryToAssignContactToExistingPatch copies the existing
        // patch's KeepPoints into tempPointsS[0..PointCount) (:832-833) and then
        // appends this round's contacts after them (:839) -- so a contact that has
        // barely moved since the previous round lands in the same buffer as its own
        // kept copy, at a distance of zero.
        //
        // The sweep found this: `reduce-cfg-dupetest`, which flips the config from
        // TDoDupeTest=false to true, survived.  The CUDA passes false at :472 and
        // the difference is only visible on coincident input, so without planted
        // duplicates this column could not tell the two apart.
        if (total >= 12u)
        {
            const uint32_t a = rng() % (total / 2u);
            const uint32_t b = total / 2u + (rng() % (total - total / 2u));
            pts[b] = pts[a];                              // exact duplicate
            const uint32_t d = rng() % (total / 2u);
            const uint32_t e = total / 2u + (rng() % (total - total / 2u));
            pts[e] = pts[d];
            pts[e].w = pts[d].w + 0.01f;                  // coincident, shallower
            ++withDupes;
        }

        for (uint32_t j = 0; j < 198u; ++j)
            c.pts[j] = pts[j];

        in[i] = c;
        ref[i] = refReduce(pts.data(), total, n, bias);
        totalPasses += ref[i].passes;
        if (ref[i].passes > 1) ++multiPass;
    }

    // THE GUARD THAT MAKES THE REST MEAN ANYTHING.  If every case finished in one
    // pass, this fixture would be a second copy of gpuslang.npCrsWrapper2 and the
    // refill discipline -- the only thing it adds -- would be untested.
    PXG_CHECK_MSG(multiPass > N / 2,
                  "npTriTriReduce_stream: only %zu of %u cases needed more than one "
                  "refill pass -- the streaming loop is what this fixture exists to "
                  "test, and it is not being exercised", multiPass, N);
    PXG_CHECK_MSG(withDupes > N / 2,
                  "npTriTriReduce_stream: only %zu of %u cases carry coincident staging "
                  "points -- without them TDoDupeTest=false and =true are "
                  "indistinguishable, and the CUDA passes false deliberately",
                  withDupes, N);
    PXG_CHECK_MSG(fullBuffer > N / 10 && shortTail > N / 10,
                  "npTriTriReduce_stream: %zu full-buffer and %zu short-tail cases; "
                  "both shapes are needed (the short tail is the only input where "
                  "`c += popcount(readMask)` overshoots totalContacts)",
                  fullBuffer, shortTail);

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<float> outInit(size_t(N) * 8 * 4, -1.0f);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    TtrParams prm{ N, 0, 0, 0 };
    Kernel k("npTriTriReduce.spv", "ttrReduce",
             { (uint32_t)sizeof(TtrCase), 16u }, sizeof(TtrParams));
    k.dispatch((N + 63) / 64, { bIn, bOut }, prm);
    dev->flush();

    const auto got = downloadBuffer<float>(dev, bOut, size_t(N) * 8 * 4);

    size_t mmCount = 0, mmMask = 0, mmDeep = 0, mmPts = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* g = &got[size_t(i) * 32];
        const uint32_t gotCount = (uint32_t)g[24];
        const float gotDeep = g[25];
        uint32_t gotMask; std::memcpy(&gotMask, &g[26], 4);

        if (gotCount != ref[i].pointCount)
        {
            if (mmCount < 4)
                std::printf("  case %u: pointCount got %u want %u\n", i, gotCount, ref[i].pointCount);
            ++mmCount;
        }
        if (gotMask != ref[i].survivorMask)
        {
            if (mmMask < 4)
                std::printf("  case %u: mask got 0x%08x want 0x%08x\n", i, gotMask, ref[i].survivorMask);
            ++mmMask;
        }
        if (gotDeep != ref[i].deepest)
        {
            if (mmDeep < 4)
                std::printf("  case %u: deepest got %.9g want %.9g\n", i, gotDeep, ref[i].deepest);
            ++mmDeep;
        }
        for (uint32_t kk = 0; kk < 6; ++kk)
        {
            const V4& w = ref[i].keepPoints[kk];
            if (g[kk * 4 + 0] != w.x || g[kk * 4 + 1] != w.y ||
                g[kk * 4 + 2] != w.z || g[kk * 4 + 3] != w.w)
            {
                if (mmPts < 4)
                    std::printf("  case %u keepPoint %u: got (%.9g %.9g %.9g %.9g) "
                                "want (%.9g %.9g %.9g %.9g)\n", i, kk,
                                g[kk * 4 + 0], g[kk * 4 + 1], g[kk * 4 + 2], g[kk * 4 + 3],
                                w.x, w.y, w.z, w.w);
                ++mmPts;
            }
        }
    }

    PXG_CHECK_MSG(mmCount == 0, "npTriTriReduce_stream: %zu pointCount mismatches", mmCount);
    PXG_CHECK_MSG(mmMask == 0, "npTriTriReduce_stream: %zu survivor-mask mismatches", mmMask);
    PXG_CHECK_MSG(mmDeep == 0, "npTriTriReduce_stream: %zu deepest mismatches", mmDeep);
    PXG_CHECK_MSG(mmPts == 0, "npTriTriReduce_stream: %zu keepPoint mismatches", mmPts);

    std::printf("  npTriTriReduce_stream: %u cases, %.2f refill passes on average, "
                "%zu needed more than one; %zu full-buffer, %zu short-tail, "
                "%zu with coincident staging points\n",
                N, double(totalPasses) / double(N), multiPass, fullBuffer, shortTail,
                withDupes);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
