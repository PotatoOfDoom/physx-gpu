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

// Verification of thresholdForceChange.slang: the cross-frame force-change event
// pipeline (setThresholdElementsMask / setPersistentForceElementMask /
// computeThresholdElementMaskIndices / outputThresholdPairsMaskIndices /
// createForceChangeThresholdElements).  The five kernels are run in sequence on
// synthetic previous/current exceeded-force sets (with persistent matches, lost
// pairs, found pairs and equal-key runs that exercise the binary-search backward
// walk), and the compacted force-change stream + count are compared bit-exactly
// against a C++ reference that replicates the kernels' logic.

#include "pxgTestFramework.h"

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct ThresholdStreamElement
{
    uint64_t shapeInteraction;
    float normalForce;
    float threshold;
    uint64_t nodeIndexA;
    uint64_t nodeIndexB;
    float accumulatedForce;
    uint32_t pad;
};
static_assert(sizeof(ThresholdStreamElement) == 40, "threshold element");

struct ForceChangeParams { uint32_t nbExceeded, nbPrev; };

// Dy::ThresholdStreamElement::operator<= (nodeIndexA then nodeIndexB, 64-bit)
bool elemLe(const ThresholdStreamElement& a, const ThresholdStreamElement& b)
{
    return (a.nodeIndexA < b.nodeIndexA) ||
           (a.nodeIndexA == b.nodeIndexA && a.nodeIndexB <= b.nodeIndexB);
}

// gpucommon reduction.cuh binarySearch(): largest idx s.t. prev[idx] <= value, else 0
uint32_t binarySearch(const std::vector<ThresholdStreamElement>& prev, uint32_t n,
                      const ThresholdStreamElement& value)
{
    uint32_t left = 0, right = n;
    while (left < right)
    {
        const uint32_t pos = (left + right) / 2;
        if (elemLe(prev[pos], value)) left = pos + 1;
        else right = pos;
    }
    return left ? left - 1 : 0;
}

PXG_TEST(thresholdForceChange_pipeline)
{
    std::mt19937 rng(34010);
    const uint32_t nbPrev = 4000;
    const uint32_t nbExceeded = 5000;
    const uint32_t total = nbExceeded + nbPrev * 2;
    const uint32_t GRID = 32;

    auto rnd64 = [&]() { return (uint64_t(rng()) << 32) | rng(); };
    // Node indices with both words populated so the 64-bit compare is exercised,
    // but a bounded key pool so keys repeat (equal-key runs + persistent hits).
    const uint32_t numKeys = 2600;
    std::vector<std::pair<uint64_t, uint64_t>> keys(numKeys);
    for (uint32_t i = 0; i < numKeys; ++i)
        keys[i] = { (uint64_t(rng() % 5000) << 32) | (rng() % 100000), (uint64_t(rng() % 5000) << 32) | (rng() % 100000) };

    auto makeElem = [&](std::pair<uint64_t, uint64_t> key, uint64_t shape) {
        ThresholdStreamElement e{};
        e.nodeIndexA = key.first;
        e.nodeIndexB = key.second;
        e.shapeInteraction = shape;
        e.normalForce = float(rng() % 100000) * 0.001f;
        e.threshold = float(rng() % 100000) * 0.001f;
        e.accumulatedForce = float(rng() % 100000) * 0.01f;
        e.pad = 0;
        return e;
    };

    // previous exceeded elements: random keys/shapes, some keys reused (runs)
    std::vector<ThresholdStreamElement> prev(nbPrev);
    for (uint32_t i = 0; i < nbPrev; ++i)
        prev[i] = makeElem(keys[rng() % numKeys], rnd64());
    std::stable_sort(prev.begin(), prev.end(),
                     [](const ThresholdStreamElement& a, const ThresholdStreamElement& b) {
                         return a.nodeIndexA != b.nodeIndexA ? a.nodeIndexA < b.nodeIndexA
                                                             : a.nodeIndexB < b.nodeIndexB;
                     });

    // current exceeded elements: ~half exact copies of a random previous element
    // (persistent), the rest fresh keys/shapes (found)
    std::vector<ThresholdStreamElement> cur(nbExceeded);
    for (uint32_t i = 0; i < nbExceeded; ++i)
    {
        if (rng() % 2 == 0)
        {
            ThresholdStreamElement p = prev[rng() % nbPrev]; // same key+shape -> persistent
            p.normalForce = float(rng() % 100000) * 0.001f;
            p.threshold = float(rng() % 100000) * 0.001f;
            p.accumulatedForce = float(rng() % 100000) * 0.01f;
            cur[i] = p;
        }
        else
        {
            cur[i] = makeElem(keys[rng() % numKeys], rnd64()); // fresh -> found
        }
    }
    std::stable_sort(cur.begin(), cur.end(),
                     [](const ThresholdStreamElement& a, const ThresholdStreamElement& b) {
                         return a.nodeIndexA != b.nodeIndexA ? a.nodeIndexA < b.nodeIndexA
                                                             : a.nodeIndexB < b.nodeIndexB;
                     });

    // ---- C++ reference (replicates the five kernels exactly) ----
    std::vector<uint32_t> mask(total, 1u), writeable(total, 0u), scanned(total, 0u);
    // 1) setThresholdElementsMask
    for (uint32_t w = 0; w < nbExceeded; ++w)
    {
        const ThresholdStreamElement& el = cur[w];
        uint32_t pos = binarySearch(prev, nbPrev, el);
        bool done = false;
        while (!done)
        {
            done = true;
            const ThresholdStreamElement& pp = prev[pos];
            if (pp.nodeIndexA == el.nodeIndexA && pp.nodeIndexB == el.nodeIndexB)
            {
                if (pp.shapeInteraction == el.shapeInteraction)
                {
                    mask[pos] = 0;
                    mask[nbPrev + w] = 0;
                }
                else if (pos > 1) { pos = pos - 1; done = false; }
            }
        }
    }
    // 2) setPersistentForceElementMask
    const uint32_t persistentStart = nbPrev + nbExceeded;
    for (uint32_t i = 0; i < nbPrev; ++i)
        mask[persistentStart + i] = mask[i] != 0 ? 0u : 1u;
    // 3+4) exclusive prefix sum, writeable = original mask
    {
        uint32_t run = 0;
        for (uint32_t w = 0; w < total; ++w)
        {
            writeable[w] = mask[w] != 0 ? 1u : 0u;
            scanned[w] = run;
            run += mask[w];
        }
    }
    // 5) createForceChangeThresholdElements
    std::vector<ThresholdStreamElement> refFC(total);
    std::memset(refFC.data(), 0, refFC.size() * sizeof(ThresholdStreamElement));
    uint32_t refCount = 0;
    for (uint32_t w = 0; w < total; ++w)
    {
        const bool has = writeable[w] != 0;
        const uint32_t wi = scanned[w];
        if (has)
        {
            const bool lost = w < nbPrev;
            const bool found = (w < persistentStart) && !lost;
            ThresholdStreamElement src = lost ? prev[w] : (found ? cur[w - nbPrev] : prev[w - persistentStart]);
            ThresholdStreamElement t{};
            t.shapeInteraction = src.shapeInteraction;
            t.nodeIndexA = src.nodeIndexA;
            t.nodeIndexB = src.nodeIndexB;
            t.normalForce = src.normalForce;
            t.accumulatedForce = lost ? 0.f : src.accumulatedForce;
            t.threshold = src.threshold;
            t.pad = 0;
            refFC[wi] = t;
        }
        if (w == total - 1) refCount = has ? (wi + 1) : wi;
    }

    // ---- GPU ----
    PxgSlangComputeDevice* dev = testDevice();
    const std::vector<uint32_t> strides = { 40, 40, 40, 4, 4, 4, 4 };
    ForceChangeParams p = { nbExceeded, nbPrev };

    std::vector<ThresholdStreamElement> fcInit(total);
    std::memset(fcInit.data(), 0, fcInit.size() * sizeof(ThresholdStreamElement));
    std::vector<uint32_t> maskInit(total, 1u), writeableInit(total, 0u),
        betweenInit(GRID, 0u), nbInit(1, 0u);

    PxgBufferHandle bExceeded = createBufferFrom(dev, cur);
    PxgBufferHandle bPrev = createBufferFrom(dev, prev);
    PxgBufferHandle bFC = createBufferFrom(dev, fcInit);
    PxgBufferHandle bMask = createBufferFrom(dev, maskInit);
    PxgBufferHandle bBetween = createBufferFrom(dev, betweenInit);
    PxgBufferHandle bWriteable = createBufferFrom(dev, writeableInit);
    PxgBufferHandle bNb = createBufferFrom(dev, nbInit);

    const std::vector<PxgBufferHandle> binds = { bExceeded, bPrev, bFC, bMask, bBetween, bWriteable, bNb };

    Kernel k1("thresholdForceChange.spv", "setThresholdElementsMask", strides, sizeof(ForceChangeParams));
    k1.dispatch(GRID, binds, p); dev->flush();
    Kernel k2("thresholdForceChange.spv", "setPersistentForceElementMask", strides, sizeof(ForceChangeParams));
    k2.dispatch(GRID, binds, p); dev->flush();
    Kernel k3("thresholdForceChange.spv", "computeThresholdElementMaskIndices", strides, sizeof(ForceChangeParams));
    k3.dispatch(GRID, binds, p); dev->flush();
    Kernel k4("thresholdForceChange.spv", "outputThresholdPairsMaskIndices", strides, sizeof(ForceChangeParams));
    k4.dispatch(GRID, binds, p); dev->flush();
    Kernel k5("thresholdForceChange.spv", "createForceChangeThresholdElements", strides, sizeof(ForceChangeParams));
    k5.dispatch(GRID, binds, p); dev->flush();

    const auto gNb = downloadBuffer<uint32_t>(dev, bNb, 1);
    const auto gFC = downloadBuffer<ThresholdStreamElement>(dev, bFC, total);

    PXG_CHECK_MSG(gNb[0] == refCount, "nbForceChange: gpu %u vs ref %u", gNb[0], refCount);

    size_t mm = 0;
    for (uint32_t w = 0; w < refCount; ++w)
    {
        const ThresholdStreamElement& a = gFC[w];
        const ThresholdStreamElement& b = refFC[w];
        if (a.shapeInteraction != b.shapeInteraction || a.nodeIndexA != b.nodeIndexA ||
            a.nodeIndexB != b.nodeIndexB || a.normalForce != b.normalForce ||
            a.threshold != b.threshold || a.accumulatedForce != b.accumulatedForce)
            ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "forceChange elements: %zu / %u mismatches", mm, refCount);

    for (PxgBufferHandle h : binds) dev->destroyBuffer(h);
}

} // namespace
