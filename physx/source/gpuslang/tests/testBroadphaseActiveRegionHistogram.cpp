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

// Verification of broadphaseActiveRegionHistogram.slang against a C++
// transcription of the three-kernel active-region pipeline.  A consistent
// synthetic region layout is built as createRegionsKernel + the region
// prefix-sum scan would produce it (region spans in [0,3], startRegionAccum =
// exclusive prefix of per-box yz-cell counts).  All three kernels run on the
// GPU in sequence; the resulting offset tables (start/active histograms) and
// the compacted ordered start/active handle arrays are compared bit-exactly
// against the transcription.

#include "pxgTestFramework.h"

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

const uint32_t TOTAL_REGION_SIZE = 64;
const uint32_t SENT = 0xdeadbeefu;

struct ActiveRegionParams
{
    uint32_t numHandles;
    uint32_t totalComparison;
    uint32_t gridDimX;
};

uint32_t getHandle(uint32_t sh) { return sh >> 3; }
uint32_t isNew(uint32_t sh) { return (sh >> 1) & 1u; }

uint32_t binarySearch(const std::vector<uint32_t>& data, uint32_t n, uint32_t value)
{
    uint32_t left = 0, right = n;
    while (left < right)
    {
        uint32_t pos = (left + right) / 2;
        if (data[pos] <= value) left = pos + 1;
        else right = pos;
    }
    return left ? left - 1 : 0;
}

void runActiveRegionCase(const char* tag, uint32_t seed, uint32_t N)
{
    std::mt19937 rng(seed);
    const uint32_t nbProj = N * 2;

    // region spans per box (indices in [0,3], s<=e), + X-axis SAP arrangement
    std::vector<uint32_t> regionRange(N * 8, 0);
    std::vector<uint32_t> sapBoxX(N * 2);
    std::vector<uint32_t> handlesX(nbProj);
    std::vector<uint32_t> yz(N);
    {
        std::vector<uint32_t> perm(nbProj);
        std::iota(perm.begin(), perm.end(), 0u);
        std::shuffle(perm.begin(), perm.end(), rng);
        for (uint32_t b = 0; b < N; ++b)
        {
            auto span = [&](uint32_t& s, uint32_t& e) { s = rng() % 4; e = s + rng() % (4 - s); };
            uint32_t sX, eX, sY, eY, sZ, eZ;
            span(sX, eX);
            span(sY, eY);
            span(sZ, eZ);
            regionRange[b * 8 + 0] = sX;
            regionRange[b * 8 + 1] = sY;
            regionRange[b * 8 + 2] = sZ;
            regionRange[b * 8 + 3] = 0;
            regionRange[b * 8 + 4] = eX;
            regionRange[b * 8 + 5] = eY;
            regionRange[b * 8 + 6] = eZ;
            regionRange[b * 8 + 7] = 0;
            yz[b] = (eY - sY + 1) * (eZ - sZ + 1);

            uint32_t mn = std::min(perm[2 * b], perm[2 * b + 1]);
            uint32_t mx = std::max(perm[2 * b], perm[2 * b + 1]);
            sapBoxX[b * 2 + 0] = mn;
            sapBoxX[b * 2 + 1] = mx;
            const uint32_t newBit = (rng() % 100 < 40) ? 1u : 0u; // ~40% new
            handlesX[mn] = (b << 3) | 1u | (newBit << 1); // start projection at min
            handlesX[mx] = (b << 3) | 0u;                 // end projection at max
        }
    }

    // startRegionAccum = exclusive prefix of per-position counts (yz at each
    // box's min position, 0 at its max position); total = grand sum
    std::vector<uint32_t> startRegionAccum(nbProj);
    uint32_t acc = 0;
    for (uint32_t pos = 0; pos < nbProj; ++pos)
    {
        startRegionAccum[pos] = acc;
        const uint32_t sh = handlesX[pos];
        if ((sh & 1u) != 0u)
            acc += yz[getHandle(sh)];
    }
    const uint32_t total = acc;

    // ---- reference: transcribe the 3 kernels ----
    std::vector<uint32_t> rStartHist(TOTAL_REGION_SIZE * nbProj, 0), rActiveHist(TOTAL_REGION_SIZE * nbProj, 0);
    std::vector<uint32_t> rOrdStart(TOTAL_REGION_SIZE * nbProj, SENT), rOrdActive(TOTAL_REGION_SIZE * nbProj, SENT);

    auto decodeRegionBase = [&](uint32_t workIndex, uint32_t& pos, uint32_t& handle, uint32_t& sh, uint32_t& sZ,
                                uint32_t& eZ) {
        pos = binarySearch(startRegionAccum, nbProj, workIndex);
        sh = handlesX[pos];
        handle = getHandle(sh);
        const uint32_t offset = workIndex - startRegionAccum[pos];
        const uint32_t sX = regionRange[handle * 8 + 0], sY = regionRange[handle * 8 + 1];
        sZ = regionRange[handle * 8 + 2];
        eZ = regionRange[handle * 8 + 6];
        const uint32_t nbZ = (eZ - sZ) + 1;
        const uint32_t zR = (offset % nbZ) + sZ;
        const uint32_t yR = (offset / nbZ) + sY;
        const uint32_t regionIndex = sX + yR * 4 + zR * 16;
        return regionIndex * nbProj;
    };

    // kernel 1: flags
    for (uint32_t wi = 0; wi < total; ++wi)
    {
        uint32_t pos, handle, sh, sZ, eZ;
        const uint32_t rb = decodeRegionBase(wi, pos, handle, sh, sZ, eZ);
        rStartHist[rb + pos] = 1;
        rActiveHist[rb + pos] = isNew(sh);
    }
    // kernel 2: per-region exclusive prefix in place
    for (uint32_t a = 0; a < TOTAL_REGION_SIZE; ++a)
    {
        uint32_t sAcc = 0, aAcc = 0;
        const uint32_t rb = a * nbProj;
        for (uint32_t i = 0; i < nbProj; ++i)
        {
            const uint32_t s = rStartHist[rb + i], ac = rActiveHist[rb + i];
            rStartHist[rb + i] = sAcc;
            rActiveHist[rb + i] = aAcc;
            sAcc += s;
            aAcc += ac;
        }
    }
    // kernel 3: scatter
    for (uint32_t wi = 0; wi < total; ++wi)
    {
        uint32_t pos, handle, sh, sZ, eZ;
        const uint32_t rb = decodeRegionBase(wi, pos, handle, sh, sZ, eZ);
        const uint32_t startProjIndex = sapBoxX[handle * 2 + 0];
        const uint32_t startIndex = rStartHist[rb + startProjIndex];
        if (eZ >= sZ)
        {
            rOrdStart[rb + startIndex] = handle;
            if (isNew(sh))
            {
                const uint32_t activeIndex = rActiveHist[rb + startProjIndex];
                rOrdActive[rb + activeIndex] = handle;
            }
        }
    }

    // ---- device ----
    PxgSlangComputeDevice* dev = testDevice();
    std::vector<uint32_t> histInit(TOTAL_REGION_SIZE * nbProj, 0);
    std::vector<uint32_t> ordInit(TOTAL_REGION_SIZE * nbProj, SENT);

    PxgBufferHandle bAccum = createBufferFrom(dev, startRegionAccum);
    PxgBufferHandle bHand = createBufferFrom(dev, handlesX);
    PxgBufferHandle bRange = createBufferFrom(dev, regionRange);
    PxgBufferHandle bStartH = createBufferFrom(dev, histInit);
    PxgBufferHandle bActiveH = createBufferFrom(dev, histInit);
    PxgBufferHandle bSap = createBufferFrom(dev, sapBoxX);
    PxgBufferHandle bOrdS = createBufferFrom(dev, ordInit);
    PxgBufferHandle bOrdA = createBufferFrom(dev, ordInit);

    const std::vector<PxgBufferHandle> buffers = { bAccum, bHand, bRange, bStartH, bActiveH, bSap, bOrdS, bOrdA };
    const std::vector<uint32_t> strides(8, 4);

    ActiveRegionParams p32 = { N, total, 32 };
    ActiveRegionParams p64 = { N, total, 64 };

    Kernel k1("broadphaseActiveRegionHistogram.spv", "writeOutStartAndActiveRegionHistogram", strides,
              sizeof(ActiveRegionParams));
    k1.dispatch(32, buffers, p32);
    dev->barrier();
    Kernel k2("broadphaseActiveRegionHistogram.spv", "computeStartAndActiveRegionHistogram", strides,
              sizeof(ActiveRegionParams));
    k2.dispatch(64, buffers, p64);
    dev->barrier();
    Kernel k3("broadphaseActiveRegionHistogram.spv", "outputOrderedActiveRegionHistogram", strides,
              sizeof(ActiveRegionParams));
    k3.dispatch(64, buffers, p64);
    dev->flush();

    const auto gStartH = downloadBuffer<uint32_t>(dev, bStartH, TOTAL_REGION_SIZE * nbProj);
    const auto gActiveH = downloadBuffer<uint32_t>(dev, bActiveH, TOTAL_REGION_SIZE * nbProj);
    const auto gOrdS = downloadBuffer<uint32_t>(dev, bOrdS, TOTAL_REGION_SIZE * nbProj);
    const auto gOrdA = downloadBuffer<uint32_t>(dev, bOrdA, TOTAL_REGION_SIZE * nbProj);

    size_t shMism = 0, ahMism = 0, osMism = 0, oaMism = 0;
    for (uint32_t i = 0; i < TOTAL_REGION_SIZE * nbProj; ++i)
    {
        if (gStartH[i] != rStartHist[i]) ++shMism;
        if (gActiveH[i] != rActiveHist[i]) ++ahMism;
        if (gOrdS[i] != rOrdStart[i]) ++osMism;
        if (gOrdA[i] != rOrdActive[i]) ++oaMism;
    }

    PXG_CHECK_MSG(shMism == 0, "[%s N=%u] %zu startHist(offset) mismatches", tag, N, shMism);
    PXG_CHECK_MSG(ahMism == 0, "[%s N=%u] %zu activeHist(offset) mismatches", tag, N, ahMism);
    PXG_CHECK_MSG(osMism == 0, "[%s N=%u] %zu orderedStart mismatches", tag, N, osMism);
    PXG_CHECK_MSG(oaMism == 0, "[%s N=%u] %zu orderedActive mismatches", tag, N, oaMism);
    PXG_CHECK_MSG(total > 0, "[%s N=%u] expected nonzero comparisons", tag, N);

    for (PxgBufferHandle h : { bAccum, bHand, bRange, bStartH, bActiveH, bSap, bOrdS, bOrdA })
        dev->destroyBuffer(h);
}

PXG_TEST(broadphaseActiveRegionHistogram_pipeline)
{
    runActiveRegionCase("small-350", 19010, 350);          // nbProj=700 -> k2 2 iterations
    runActiveRegionCase("multi-iter-3000", 19011, 3000);   // nbProj=6000 -> k2 12 iters; k1/k3 multi-iter
}

} // namespace
