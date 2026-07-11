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

// Verification of npCvxTriFind.slang: ctCvxTriFind — binarySearch (reduction.cuh
// :729) + findTriangle / findFaceContactTriangle / findEdgeContactTriangle
// (convexMeshPostProcess.cu :154-179).  Pure unsigned-integer arithmetic, so the
// shader / CPU reference / CUDA agree bit-exactly by construction; one thread per
// independent lookup.  Fixtures drive the full branch matrix of the search and
// the equality confirm: key present (interior hit), absent but in range, below
// data[0] (the left==0 clamp), above the tail, duplicate keys (the `<=` step
// returns the LAST occurrence), and count 0 / 1.  findTriangle (which=0) and
// findFaceContactTriangle (which=1) share the flag 0x80000000 and are asserted
// identical; findEdgeContactTriangle (which=2) uses 0xc0000000.

#include "pxgTestFramework.h"

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct FindIn { uint32_t count, target, which, pad; };
struct FindOut { uint32_t result, searchIndex; };
struct FindParams { uint32_t numTests, maxLen, pad0, pad1; };

static const uint32_t MAXLEN = 64u;
static const uint32_t FACE = 0x80000000u;
static const uint32_t EDGE = 0xc0000000u;
static uint32_t flagOf(uint32_t which) { return which == 2u ? EDGE : FACE; }

// exact transcription of reduction.cuh :729
static uint32_t binSearchRef(const uint32_t* d, uint32_t n, uint32_t v)
{
    uint32_t left = 0, right = n;
    while (left < right)
    {
        const uint32_t pos = (left + right) / 2u;
        if (d[pos] <= v) left = pos + 1u;
        else right = pos;
    }
    return left ? left - 1u : 0u;
}

PXG_TEST(npCvxTriFind_find)
{
    std::mt19937 rng(1130113);

    std::vector<uint32_t> dataFlat;
    std::vector<FindIn> in;
    std::vector<FindOut> ref;

    size_t nPresent = 0, nAbsent = 0, nBelow = 0, nAbove = 0, nDup = 0, nCount0 = 0, nCount1 = 0;

    auto emit = [&](const std::vector<uint32_t>& keys, uint32_t target, uint32_t which)
    {
        const uint32_t count = (uint32_t)keys.size();
        for (uint32_t i = 0; i < MAXLEN; ++i) dataFlat.push_back(i < count ? keys[i] : 0u);
        FindIn e; e.count = count; e.target = target; e.which = which; e.pad = 0;
        in.push_back(e);

        const uint32_t flagged = target | flagOf(which);
        const uint32_t idx = count ? binSearchRef(keys.data(), count, flagged) : 0u;
        const uint32_t result = (idx < count && keys[idx] == flagged) ? 1u : 0u;
        FindOut o; o.result = result; o.searchIndex = idx;
        ref.push_back(o);

        // branch bookkeeping (reference-side, deterministic)
        bool hasDup = false;
        for (uint32_t i = 1; i < count; ++i) if (keys[i] == keys[i - 1]) hasDup = true;
        if (count == 0) ++nCount0;
        else if (count == 1) ++nCount1;
        if (result) { ++nPresent; if (hasDup) ++nDup; }
        else if (count)
        {
            if (flagged < keys[0]) ++nBelow;
            else if (flagged > keys[count - 1]) ++nAbove;
            else ++nAbsent;
        }
    };

    // raw emitter: writes an EXPLICIT MAXLEN key block (so a test can seed the
    // buffer past `count`, e.g. a count==0 lookup whose slot 0 holds the target
    // key — the `index < count` OOB guard must still report not-found)
    auto emitRaw = [&](const std::vector<uint32_t>& block, uint32_t count, uint32_t target, uint32_t which)
    {
        for (uint32_t i = 0; i < MAXLEN; ++i) dataFlat.push_back(i < block.size() ? block[i] : 0u);
        FindIn e; e.count = count; e.target = target; e.which = which; e.pad = 0;
        in.push_back(e);
        const uint32_t flagged = target | flagOf(which);
        const uint32_t idx = count ? binSearchRef(block.data(), count, flagged) : 0u;
        const uint32_t result = (idx < count && idx < (uint32_t)block.size() && block[idx] == flagged) ? 1u : 0u;
        FindOut o; o.result = result; o.searchIndex = idx;
        ref.push_back(o);
        if (count == 0) ++nCount0; else if (count == 1) ++nCount1;
    };

    // build a sorted array of `count` distinct flagged keys from low indices in
    // [lo, hi); returns the low-index set for target selection
    auto buildKeys = [&](uint32_t which, uint32_t count, uint32_t lo, uint32_t hi,
                         std::vector<uint32_t>& lowOut) -> std::vector<uint32_t>
    {
        const uint32_t flag = flagOf(which);
        std::vector<uint32_t> low;
        while (low.size() < count)
        {
            uint32_t x = lo + rng() % (hi - lo);
            if (std::find(low.begin(), low.end(), x) == low.end()) low.push_back(x);
        }
        std::sort(low.begin(), low.end());
        lowOut = low;
        std::vector<uint32_t> keys;
        for (uint32_t v : low) keys.push_back(flag | v);
        return keys;
    };

    // ---- family A: present (interior hit) -----------------------------------
    for (uint32_t iter = 0; iter < 500; ++iter)
    {
        const uint32_t which = rng() % 3;
        const uint32_t count = 2u + rng() % 38u;
        std::vector<uint32_t> low, keys = buildKeys(which, count, 1u, 400u, low);
        const uint32_t j = rng() % count;
        emit(keys, low[j], which); // target = low[j] -> flagged hits keys[j]
    }

    // ---- family B: absent but within [data[0], data[count-1]] ---------------
    for (uint32_t iter = 0; iter < 400; ++iter)
    {
        const uint32_t which = rng() % 3;
        const uint32_t count = 3u + rng() % 37u;
        std::vector<uint32_t> low, keys = buildKeys(which, count, 2u, 400u, low);
        // pick a low value strictly between low.front() and low.back() not present
        uint32_t target = 0; bool ok = false;
        for (uint32_t tries = 0; tries < 64 && !ok; ++tries)
        {
            uint32_t x = low.front() + 1u + rng() % (low.back() - low.front());
            if (std::find(low.begin(), low.end(), x) == low.end()) { target = x; ok = true; }
        }
        if (ok) emit(keys, target, which);
    }

    // ---- family C: below data[0] (the left==0 clamp) ------------------------
    for (uint32_t iter = 0; iter < 250; ++iter)
    {
        const uint32_t which = rng() % 3;
        const uint32_t count = 2u + rng() % 38u;
        std::vector<uint32_t> low, keys = buildKeys(which, count, 5u, 400u, low);
        emit(keys, rng() % low.front(), which); // target < low.front() -> flagged < keys[0]
    }

    // ---- family D: above data[count-1] --------------------------------------
    for (uint32_t iter = 0; iter < 250; ++iter)
    {
        const uint32_t which = rng() % 3;
        const uint32_t count = 2u + rng() % 38u;
        std::vector<uint32_t> low, keys = buildKeys(which, count, 1u, 400u, low);
        emit(keys, low.back() + 1u + rng() % 50u, which); // target > low.back()
    }

    // ---- family E: duplicate keys (the `<=` step lands on the LAST dup) ------
    for (uint32_t iter = 0; iter < 300; ++iter)
    {
        const uint32_t which = rng() % 3;
        const uint32_t flag = flagOf(which);
        const uint32_t distinct = 3u + rng() % 15u;
        std::vector<uint32_t> low, keysD = buildKeys(which, distinct, 1u, 200u, low);
        // duplicate a few entries, then re-sort
        std::vector<uint32_t> keys = keysD;
        for (int k = 0; k < 3; ++k) keys.push_back(keysD[rng() % distinct]);
        std::sort(keys.begin(), keys.end());
        // half hit a (possibly duplicated) key, half probe between
        if (rng() % 2) emit(keys, low[rng() % distinct], which);
        else emit(keys, (rng() % 250u), which);
        (void)flag;
    }

    // ---- family F: count 0 and count 1 --------------------------------------
    for (uint32_t iter = 0; iter < 120; ++iter)
    {
        const uint32_t which = rng() % 3;
        if (iter % 3 == 0) { emit({}, rng() % 100, which); } // count 0
        else
        {
            const uint32_t flag = flagOf(which);
            const uint32_t x = 1u + rng() % 200u;
            std::vector<uint32_t> keys = { flag | x };
            emit(keys, (rng() % 2) ? x : (x + 1u + rng() % 20u), which); // hit or miss
        }
    }

    // ---- family G: mixed FACE/EDGE keys, random target (breadth) ------------
    for (uint32_t iter = 0; iter < 400; ++iter)
    {
        const uint32_t count = 2u + rng() % 40u;
        std::vector<uint32_t> keys;
        for (uint32_t i = 0; i < count; ++i)
        {
            const uint32_t flag = (rng() % 2) ? FACE : EDGE;
            keys.push_back(flag | (rng() % 400u));
        }
        std::sort(keys.begin(), keys.end());
        emit(keys, rng() % 400u, rng() % 3);
    }

    // ---- family H: findTriangle (0) vs findFaceContactTriangle (1) identity --
    // same (array, target) under which=0 and which=1 must yield identical results
    for (uint32_t iter = 0; iter < 150; ++iter)
    {
        const uint32_t count = 2u + rng() % 30u;
        std::vector<uint32_t> low, keys = buildKeys(0u, count, 1u, 400u, low);
        const uint32_t target = (rng() % 2) ? low[rng() % count] : (rng() % 420u);
        const size_t i0 = ref.size(); emit(keys, target, 0u);
        const size_t i1 = ref.size(); emit(keys, target, 1u);
        PXG_CHECK_MSG(ref[i0].result == ref[i1].result && ref[i0].searchIndex == ref[i1].searchIndex,
                      "findTriangle vs findFaceContactTriangle diverge in reference");
    }

    // ---- family I: count==0 with slot 0 seeded to the target key -----------
    // pins the `index < count` guard: binarySearch returns 0, and without the
    // guard the code would read slot 0 (== flagged target) and wrongly report
    // found.  The correct result is ALWAYS not-found when count==0.
    for (uint32_t iter = 0; iter < 60; ++iter)
    {
        const uint32_t which = rng() % 3;
        const uint32_t low = 1u + rng() % 300u;
        std::vector<uint32_t> block = { low | flagOf(which) }; // slot 0 = the flagged target
        emitRaw(block, 0u, low, which); // count == 0 -> must be not-found
    }

    const uint32_t N = (uint32_t)in.size();
    PxgSlangComputeDevice* dev = testDevice();
    std::vector<FindOut> outInit(N, { 0xEEEEEEEEu, 0xEEEEEEEEu });

    PxgBufferHandle bD = createBufferFrom(dev, dataFlat);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    FindParams prm = { N, MAXLEN, 0, 0 };
    Kernel k("npCvxTriFind.spv", "ctCvxTriFind",
             { 4, (uint32_t)sizeof(FindIn), (uint32_t)sizeof(FindOut) }, sizeof(FindParams));
    k.dispatch((N + 63) / 64, { bD, bIn, bOut }, prm);
    dev->flush();

    const auto gOut = downloadBuffer<FindOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t t = 0; t < N; ++t)
        if (gOut[t].result != ref[t].result || gOut[t].searchIndex != ref[t].searchIndex) ++mm;

    PXG_CHECK_MSG(mm == 0, "cvxTriFind: %zu mismatches", mm);
    PXG_CHECK_MSG(nPresent > 200 && nAbsent > 100 && nBelow > 100 && nAbove > 100 &&
                  nDup > 50 && nCount0 > 10 && nCount1 > 20,
                  "cvxTriFind coverage thin (present=%zu absent=%zu below=%zu above=%zu dup=%zu c0=%zu c1=%zu)",
                  nPresent, nAbsent, nBelow, nAbove, nDup, nCount0, nCount1);

    for (PxgBufferHandle h : { bD, bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
