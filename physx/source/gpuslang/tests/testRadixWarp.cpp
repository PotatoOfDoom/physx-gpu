// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

// Verification of pxgSlangRadixWarp — radixSortSingleWarpKeysOnly
// (RadixSort.cuh:579-720) on a 32-lane virtual warp.
//
// The oracle is std::sort: eight four-bit passes least-significant-first are a
// stable ascending sort of the whole 32-bit key, so the expected output is simply
// the input sorted — INCLUDING the tail that sanitizeKeys forced to 0xFFFFFFFF,
// which must land at the end.
//
// The keys are shaped like the array this actually sorts: `faceFlag | triangleIdx`
// words, so bit 31 is set on some of them and the sort has to treat the key as
// UNSIGNED.  A signed comparison would put the flagged entries first.

#include "pxgTestFramework.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct RadixPush { uint64_t keys, cases; uint32_t numCases, pad; };
struct CaseDesc { uint32_t count, numUint4, slotBase, pad; };

PXG_TEST(radixWarpSort)
{
    PxgSlangComputeDevice* dev = testDevice();

    std::mt19937 rng(31337);

    struct Case { std::vector<uint32_t> keys; uint32_t count; const char* name; };
    std::vector<Case> cases;

    auto addCase = [&](uint32_t count, const char* name, int mode) {
        Case c; c.count = count; c.name = name;
        for (uint32_t i = 0; i < count; ++i)
        {
            uint32_t k;
            switch (mode)
            {
            case 0: k = rng() & 0x07ffffffu; break;                  // plain indices
            case 1: k = (rng() & 0x07ffffffu) | ((rng() & 1u) << 31); break;  // with the face flag
            case 2: k = count - 1 - i; break;                        // exactly reversed
            case 3: k = 7u; break;                                   // all equal
            default: k = i; break;                                   // already sorted
            }
            c.keys.push_back(k);
        }
        cases.push_back(c);
    };

    // The counts straddle every boundary the pass loop has: below one uint4, an
    // exact uint4, one past it, exactly one warp's worth of uint4s (128 keys),
    // and past it so the strided loops iterate more than once.
    addCase(1, "one key", 0);
    addCase(3, "under one uint4", 1);
    addCase(4, "exactly one uint4", 1);
    addCase(5, "one past a uint4", 1);
    addCase(7, "reversed", 2);
    addCase(16, "all equal", 3);
    addCase(31, "already sorted", 4);
    addCase(32, "a full warp of keys", 1);
    addCase(128, "one warp of uint4s", 1);
    addCase(129, "one past a warp of uint4s", 1);
    addCase(200, "two strided passes", 1);
    addCase(511, "large, flagged", 1);

    // Pack every case into one buffer: a rounded key run followed by its temp
    // half, which is what sortTriangleIndices hands the sort.
    std::vector<uint32_t> keyBuf;
    std::vector<CaseDesc> descs;
    std::vector<std::vector<uint32_t>> expected;
    for (const Case& c : cases)
    {
        const uint32_t numUint4 = (c.count + 3) / 4;
        const uint32_t rounded = numUint4 * 4;
        CaseDesc d{ c.count, numUint4, uint32_t(keyBuf.size()), 0 };
        descs.push_back(d);

        for (uint32_t i = 0; i < rounded; ++i)
            keyBuf.push_back(i < c.count ? c.keys[i] : 0xCDCDCDCDu);  // padding: NOT the sentinel
        for (uint32_t i = 0; i < rounded; ++i)
            keyBuf.push_back(0xEEEEEEEEu);                            // the temp half

        // The oracle: the real keys plus the sanitised tail, ascending unsigned.
        std::vector<uint32_t> e(c.keys);
        for (uint32_t i = c.count; i < rounded; ++i) e.push_back(0xFFFFFFFFu);
        std::sort(e.begin(), e.end());
        expected.push_back(e);
    }

    const uint32_t n = uint32_t(cases.size());
    PxgBufferHandle bKeys = createBufferFrom(dev, keyBuf);
    PxgBufferHandle bCases = createBufferFrom(dev, descs);

    RadixPush push{};
    push.keys = dev->getBufferDeviceAddress(bKeys);
    push.cases = dev->getBufferDeviceAddress(bCases);
    push.numCases = n;

    {
        Kernel k("radixWarpBda.spv", "radixWarpLaunch", {}, sizeof(RadixPush));
        k.dispatch(n, {}, push, 1u, 1u);
        dev->flush();
    }

    const auto got = downloadBuffer<uint32_t>(dev, bKeys, keyBuf.size());

    size_t bad = 0;
    size_t sawFlagged = 0, sawPadding = 0, sawMultiPass = 0;
    for (uint32_t ci = 0; ci < n; ++ci)
    {
        const CaseDesc& d = descs[ci];
        const std::vector<uint32_t>& e = expected[ci];
        const uint32_t rounded = d.numUint4 * 4;

        if (rounded > d.count) ++sawPadding;
        if (d.numUint4 > 32) ++sawMultiPass;
        for (uint32_t k : cases[ci].keys) if (k & 0x80000000u) { ++sawFlagged; break; }

        for (uint32_t i = 0; i < rounded; ++i)
        {
            const uint32_t g = got[d.slotBase + i];
            if (g != e[i])
            {
                ++bad;
                if (bad <= 10)
                    std::printf("  [%s] slot %u: got=%08x exp=%08x\n", cases[ci].name, i, g, e[i]);
            }
        }
    }

    dev->destroyBuffer(bKeys);
    dev->destroyBuffer(bCases);

    PXG_CHECK_MSG(bad == 0, "radixWarp: %zu mismatches", bad);
    PXG_CHECK_MSG(sawFlagged > 0, "no case sets bit 31 (the unsigned comparison is untested)");
    PXG_CHECK_MSG(sawPadding > 0, "no case has a partial final uint4 (sanitizeKeys is untested)");
    PXG_CHECK_MSG(sawMultiPass > 0, "no case needs more than one strided pass");
}

} // namespace
