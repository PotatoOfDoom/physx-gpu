// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

// Verification of sortTriIndicesBda — `sortTriangleIndices`
// (convexMeshPostProcess.cu:623) on the engine BDA ABI.
//
// The sort itself is already pinned by testRadixWarp; what this fixture pins is
// the WRAPPER, and the wrapper is almost entirely address arithmetic:
//
//   * `count` lives at @32 and `roundedStartIndex` at @40 in ConvexMeshPair.  The
//     declaration in convexNpCommon.h carries running END offsets in its trailing
//     comments, so reading those as field offsets lands on `cmIndex` (@36) and
//     `pad` (@44).  Every pair here gives @28/@36/@44 values that are plausible
//     but wrong, and points the wrong ones at scratch regions, so a misread is a
//     mismatch rather than an out-of-bounds write.
//   * the key run starts at `*orderedCvxTriIntermPtr + roundedStartIndex` in
//     PxU32 units (byte offset x4) and its temp half starts one nbUint4s-sized
//     uint4 run later (byte offset x16).  Both scales are exercised by regions
//     whose word offsets are not their byte offsets.
//   * `count == 0` and `count == 0xFFFFFFFF` are early-outs — the run must come
//     back BIT-IDENTICAL, not merely sorted-and-equal.
//   * the grid is ceil(nbPairs / 2) workgroups each covering two warps, so an odd
//     nbPairs leaves the last workgroup's second warp out of range.  That warp
//     must touch nothing.
//
// Regions are laid out in an order that is NOT the pair order, so a kernel that
// derived the run's base from the pair index instead of reading @40 would fail.

#include "pxgTestFramework.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct SortTriPush
{
    uint64_t cvxTrimeshPair;
    uint64_t orderedCvxTriIntermPtr;
    uint32_t nbPairs;
    uint32_t pad;
};

const uint32_t PAD_WORD = 0xCDCDCDCDu;   // the rounding tail sanitizeKeys overwrites
const uint32_t TEMP_WORD = 0xEEEEEEEEu;  // the ping-pong half
const uint32_t GUARD_WORD = 0x7E577E57u; // between regions: must never move
const uint32_t SCRATCH_WORD = 0x5C5A7C40u;

// `groupsFor`: the dispatch for a pair count.  The engine launches ceil(nbPairs / 2) blocks of
// two warps (PxgNarrowphaseCore.cpp); the port's launch site (PXG_NO_CUDA_TOOLKIT) launches one
// 32-thread group per CUDA warp, twice that, and the shader's group-stride loop covers any
// smaller grid -- the CUDA's own (a group takes two pairs one grid-width apart) and one group.
static void runSortTriIndices(const char* gridName, uint32_t (*groupsFor)(uint32_t))
{
    PxgSlangComputeDevice* dev = testDevice();

    std::mt19937 rng(90210);

    struct Pair
    {
        uint32_t count;     // the value written to @32
        uint32_t nominal;   // the region is sized for this many keys
        const char* name;
        bool active;
        std::vector<uint32_t> keys;
        uint32_t rounded;
        uint32_t regionStart;  // in PxU32 words -- the @40 value
    };

    // count == 0xFFFFFFFF is "cached from last frame"; count == 0 is "no
    // triangles".  Both must leave the run alone.  The active counts straddle a
    // partial uint4 (5), an exact one (12), a single key (1), one that needs more
    // than one strided pass through the warp (130), and one just over a warp of
    // keys (33).
    std::vector<Pair> pairs = {
        { 5u,          5u, "five keys, partial uint4", true, {}, 0u, 0u },
        { 0u,          8u, "count == 0 (early out)",   false, {}, 0u, 0u },
        { 130u,      130u, "130 keys, multi-pass",     true, {}, 0u, 0u },
        { 0xFFFFFFFFu, 8u, "count == ~0 (cached)",     false, {}, 0u, 0u },
        { 1u,          1u, "a single key",             true, {}, 0u, 0u },
        { 12u,        12u, "exactly three uint4s",     true, {}, 0u, 0u },
        { 33u,        33u, "just over a warp",         true, {}, 0u, 0u },
    };
    const uint32_t nbPairs = uint32_t(pairs.size());  // odd on purpose

    for (Pair& p : pairs)
    {
        const uint32_t n = p.active ? p.count : p.nominal;
        p.rounded = ((n + 3u) / 4u) * 4u;
        if (p.active)
            for (uint32_t i = 0; i < p.count; ++i)
                p.keys.push_back((rng() & 0x07ffffffu) | ((rng() & 1u) << 31));  // the face flag
    }

    // Region order is deliberately not pair order.
    const uint32_t order[] = { 4u, 0u, 6u, 2u, 5u, 1u, 3u };

    std::vector<uint32_t> ordered;
    auto guard = [&]() { for (int i = 0; i < 4; ++i) ordered.push_back(GUARD_WORD); };

    guard();
    for (uint32_t oi = 0; oi < nbPairs; ++oi)
    {
        Pair& p = pairs[order[oi]];
        p.regionStart = uint32_t(ordered.size());
        if (p.active)
        {
            for (uint32_t i = 0; i < p.rounded; ++i)
                ordered.push_back(i < p.count ? p.keys[i] : PAD_WORD);
            for (uint32_t i = 0; i < p.rounded; ++i)
                ordered.push_back(TEMP_WORD);
        }
        else
        {
            // A skipped run is filled with a per-pair marker so "untouched" is a
            // bit-exact claim and not just "still looks unsorted".
            for (uint32_t i = 0; i < p.rounded * 2u; ++i)
                ordered.push_back(0xA5A50000u | ((order[oi] << 8) & 0xff00u) | (i & 0xffu));
        }
        guard();
    }

    // Two scratch regions, each large enough for the biggest run plus its temp
    // half, so a wrong-offset kernel writes here instead of off the end.
    uint32_t biggest = 0;
    for (const Pair& p : pairs) biggest = std::max(biggest, p.rounded);
    // The fill DESCENDS: a constant fill would come back bit-identical from a
    // stray sort and the "scratch never moved" check would prove nothing.
    const uint32_t scratchA = uint32_t(ordered.size());
    for (uint32_t i = 0; i < biggest * 2u; ++i) ordered.push_back(SCRATCH_WORD - i);
    guard();
    const uint32_t scratchB = uint32_t(ordered.size());
    for (uint32_t i = 0; i < biggest * 2u; ++i) ordered.push_back(SCRATCH_WORD - i);
    guard();

    // ConvexMeshPair, 56 bytes = 14 words:
    //   @0 aToB(28) @28 startIndex @32 count @36 cmIndex @40 roundedStartIndex
    //   @44 pad @48 materialIndices(8)
    //
    // The array is allocated with ONE ENTRY MORE than nbPairs and that entry is a
    // live-looking decoy pointing at scratch.  The real pair array is sized by
    // contact-manager capacity, not by the live count, so stale entries past
    // nbPairs are exactly what the engine's memory holds — and a kernel that
    // dropped the `globalWarpIndex >= nbPairs` guard would sort the decoy, because
    // ceil(7/2) blocks of two warps reach index 7.
    std::vector<uint32_t> pairBuf((nbPairs + 1u) * 14u, 0u);
    {
        uint32_t* d = &pairBuf[nbPairs * 14u];
        d[7] = scratchB;    // @28
        d[8] = 8u;          // @32 count -- a plausible live run
        d[9] = 8u;          // @36
        d[10] = scratchA;   // @40 roundedStartIndex -- into scratch, which must not move
        d[11] = scratchA;   // @44
    }
    for (uint32_t i = 0; i < nbPairs; ++i)
    {
        uint32_t* w = &pairBuf[i * 14u];
        for (uint32_t t = 0; t < 7u; ++t) w[t] = 0xDEAD0000u | t;  // the transform
        w[7] = scratchA;             // @28 startIndex -- plausible, wrong, harmless
        w[8] = pairs[i].count;       // @32 count
        // @36 cmIndex: a value a misread would happily use as a count.  Kept at or
        // below the real count so the wrong sort still fits its region; zero for
        // the single-key pair (a misread then skips, which is just as visible).
        w[9] = pairs[i].active ? (pairs[i].count / 2u) : 7u;
        w[10] = pairs[i].regionStart;  // @40 roundedStartIndex
        w[11] = scratchB;              // @44 pad
        w[12] = 0x11110000u | i;       // @48 materialIndices.x
        w[13] = 0x22220000u | i;       // @52 materialIndices.y
    }

    std::vector<uint32_t> before = ordered;

    PxgBufferHandle bOrdered = createBufferFrom(dev, ordered);
    std::vector<uint64_t> ptrBuf(1, dev->getBufferDeviceAddress(bOrdered));
    PxgBufferHandle bPtr = createBufferFrom(dev, ptrBuf);
    PxgBufferHandle bPairs = createBufferFrom(dev, pairBuf);

    SortTriPush push{};
    push.cvxTrimeshPair = dev->getBufferDeviceAddress(bPairs);
    push.orderedCvxTriIntermPtr = dev->getBufferDeviceAddress(bPtr);
    push.nbPairs = nbPairs;

    {
        Kernel k("sortTriIndicesBda.spv", "sortTriangleIndices", {}, sizeof(SortTriPush));
        k.dispatch(groupsFor(nbPairs), {}, push, 1u, 1u);
        dev->flush();
    }

    const auto got = downloadBuffer<uint32_t>(dev, bOrdered, ordered.size());

    size_t bad = 0;
    uint32_t sawPartial = 0, sawMultiPass = 0, sawFlagged = 0, sawSkipped = 0;

    for (uint32_t i = 0; i < nbPairs; ++i)
    {
        const Pair& p = pairs[i];
        if (!p.active)
        {
            ++sawSkipped;
            for (uint32_t j = 0; j < p.rounded * 2u; ++j)
            {
                const uint32_t g = got[p.regionStart + j];
                if (g != before[p.regionStart + j])
                {
                    ++bad;
                    if (bad <= 12)
                        std::printf("  [%s] word %u touched: got=%08x was=%08x\n",
                                    p.name, j, g, before[p.regionStart + j]);
                }
            }
            continue;
        }

        std::vector<uint32_t> exp(p.keys);
        for (uint32_t j = p.count; j < p.rounded; ++j) exp.push_back(0xFFFFFFFFu);
        std::sort(exp.begin(), exp.end());

        if (p.rounded > p.count) ++sawPartial;
        if (p.rounded / 4u > 32u) ++sawMultiPass;
        for (uint32_t k : p.keys) if (k & 0x80000000u) { ++sawFlagged; break; }

        for (uint32_t j = 0; j < p.rounded; ++j)
        {
            const uint32_t g = got[p.regionStart + j];
            if (g != exp[j])
            {
                ++bad;
                if (bad <= 12)
                    std::printf("  [%s] slot %u: got=%08x exp=%08x\n", p.name, j, g, exp[j]);
            }
        }
    }

    // Nothing outside a live key run may move: not the guards, not the scratch
    // regions the wrong offsets point at.
    size_t guardBad = 0, scratchBad = 0;
    std::vector<bool> live(ordered.size(), false);
    for (const Pair& p : pairs)
        if (p.active)
            for (uint32_t j = 0; j < p.rounded * 2u; ++j) live[p.regionStart + j] = true;

    for (size_t j = 0; j < ordered.size(); ++j)
    {
        if (live[j] || got[j] == before[j]) continue;
        if (before[j] == GUARD_WORD) ++guardBad;
        else if (j >= scratchA) ++scratchBad;
    }

    dev->destroyBuffer(bOrdered);
    dev->destroyBuffer(bPtr);
    dev->destroyBuffer(bPairs);

    PXG_CHECK_MSG(bad == 0, "sortTriIndices (%s grid, %u groups): %zu mismatches", gridName, groupsFor(nbPairs), bad);
    PXG_CHECK_MSG(guardBad == 0, "sortTriIndices: %zu guard words moved (a run overran its region)",
                  guardBad);
    PXG_CHECK_MSG(scratchBad == 0,
                  "sortTriIndices: %zu scratch words moved (an offset was read from the wrong field)",
                  scratchBad);
    PXG_CHECK_MSG(sawPartial > 0, "no active pair has a partial final uint4");
    PXG_CHECK_MSG(sawMultiPass > 0, "no active pair needs more than one strided pass");
    PXG_CHECK_MSG(sawFlagged > 0, "no active pair sets bit 31");
    PXG_CHECK_MSG(sawSkipped == 2, "both early-out counts must be present");
}

PXG_TEST(sortTriIndices)
{
    runSortTriIndices("port", [](uint32_t n) { return 2u * ((n + 1u) / 2u); });
}

PXG_TEST(sortTriIndicesCudaGrid)
{
    runSortTriIndices("CUDA", [](uint32_t n) { return (n + 1u) / 2u; });
    runSortTriIndices("one-group", [](uint32_t) { return 1u; });
}

} // namespace
