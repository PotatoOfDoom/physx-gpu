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

// Verification of broadphaseOverlapCheckHistogram.slang.  Pass 1 (the
// per-comparison overlap-check count + handle/region record) is checked
// elementwise against a C++ transcription over a consistent synthetic region
// layout.  Passes 2+3 (the 64-bit device-wide exclusive prefix scan) are
// checked end-to-end against a uint64 exclusive scan whose total deliberately
// exceeds 2^32, proving the 64-bit accumulation path (the exact overflow PhysX
// types these buffers PxU64 for).

#include "pxgTestFramework.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

const uint32_t TOTAL_REGION_SIZE = 64;

struct OverlapCheckParams
{
    uint32_t numHandles;
    uint32_t totalComparison;
    uint32_t gridDimX;
};

uint32_t getHandle(uint32_t sh) { return sh >> 3; }
uint32_t isNewP(uint32_t sh) { return (sh >> 1) & 1u; }

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

// ---- pass 1: elementwise overlap-check counts + (handle, region) records ----
PXG_TEST(broadphaseOverlapCheckHistogram_writeout)
{
    std::mt19937 rng(20010);
    const uint32_t N = 1500;
    const uint32_t nbProj = N * 2;

    std::vector<uint32_t> regionRange(N * 8, 0), sapBoxX(N * 2), handlesX(nbProj), fullSpan(N);
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
            regionRange[b * 8 + 4] = eX;
            regionRange[b * 8 + 5] = eY;
            regionRange[b * 8 + 6] = eZ;
            fullSpan[b] = (eX - sX + 1) * (eY - sY + 1) * (eZ - sZ + 1);
            uint32_t mn = std::min(perm[2 * b], perm[2 * b + 1]);
            uint32_t mx = std::max(perm[2 * b], perm[2 * b + 1]);
            sapBoxX[b * 2 + 0] = mn;
            sapBoxX[b * 2 + 1] = mx;
            const uint32_t newBit = (rng() % 100 < 50) ? 1u : 0u;
            handlesX[mn] = (b << 3) | 1u | (newBit << 1);
            handlesX[mx] = (b << 3) | 0u;
        }
    }

    // regionAccum = exclusive prefix of full-3D-span counts (createRegions + scan)
    std::vector<uint32_t> regionAccum(nbProj);
    uint32_t acc = 0;
    for (uint32_t pos = 0; pos < nbProj; ++pos)
    {
        regionAccum[pos] = acc;
        const uint32_t sh = handlesX[pos];
        if ((sh & 1u) != 0u) acc += fullSpan[getHandle(sh)];
    }
    const uint32_t total = acc;

    // startHist / activeHist offset tables: non-decreasing per region
    std::vector<uint32_t> startHist(TOTAL_REGION_SIZE * nbProj), activeHist(TOTAL_REGION_SIZE * nbProj);
    for (uint32_t r = 0; r < TOTAL_REGION_SIZE; ++r)
    {
        uint32_t sa = 0, aa = 0;
        for (uint32_t i = 0; i < nbProj; ++i)
        {
            startHist[r * nbProj + i] = sa;
            activeHist[r * nbProj + i] = aa;
            sa += rng() & 1u;
            aa += rng() & 1u;
        }
    }

    // reference
    std::vector<uint64_t> refOverlap(total);
    std::vector<uint32_t> refHR(total * 2);
    for (uint32_t wi = 0; wi < total; ++wi)
    {
        const uint32_t bs = binarySearch(regionAccum, nbProj, wi);
        const uint32_t offset = wi - regionAccum[bs];
        const uint32_t sh = handlesX[bs];
        const uint32_t handle = getHandle(sh);
        const uint32_t isNew = isNewP(sh);
        const uint32_t be = sapBoxX[handle * 2 + 1];
        const uint32_t sX = regionRange[handle * 8 + 0], sY = regionRange[handle * 8 + 1],
                       sZ = regionRange[handle * 8 + 2];
        const uint32_t eY = regionRange[handle * 8 + 5], eZ = regionRange[handle * 8 + 6];
        const uint32_t nbZ = (eZ - sZ) + 1, nbY = (eY - sY) + 1;
        const uint32_t zR = (offset % nbZ) + sZ;
        const uint32_t yR = ((offset / nbZ) % nbY) + sY;
        const uint32_t xR = (offset / (nbZ * nbY)) + sX;
        const uint32_t regionIndex = (xR + yR * 4 + zR * 16) * nbProj;
        const std::vector<uint32_t>& tbl = isNew ? startHist : activeHist;
        const uint32_t rStart = tbl[regionIndex + bs + 1];
        const uint32_t rEnd = tbl[regionIndex + be];
        refHR[wi * 2 + 0] = handle;
        refHR[wi * 2 + 1] = regionIndex;
        refOverlap[wi] = uint64_t(rEnd - rStart);
    }

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<uint64_t> overlapInit(total, 0xdeadbeefdeadbeefull);
    std::vector<uint32_t> hrInit(total * 2, 0xcdcdcdcdu);
    std::vector<uint64_t> blkInit(TOTAL_REGION_SIZE, 0), totInit(1, 0);
    std::vector<uint32_t> dummy(16, 0);

    PxgBufferHandle bAcc = createBufferFrom(dev, regionAccum);
    PxgBufferHandle bH = createBufferFrom(dev, handlesX);
    PxgBufferHandle bSap = createBufferFrom(dev, sapBoxX);
    PxgBufferHandle bRange = createBufferFrom(dev, regionRange);
    PxgBufferHandle bSH = createBufferFrom(dev, startHist);
    PxgBufferHandle bAH = createBufferFrom(dev, activeHist);
    PxgBufferHandle bOv = createBufferFrom(dev, overlapInit);
    PxgBufferHandle bHR = createBufferFrom(dev, hrInit);
    PxgBufferHandle bBlk = createBufferFrom(dev, blkInit);
    PxgBufferHandle bTot = createBufferFrom(dev, totInit);

    OverlapCheckParams p = { N, total, 32 };
    const std::vector<uint32_t> strides = { 4, 4, 4, 4, 4, 4, 8, 4, 8, 8 };
    Kernel k("broadphaseOverlapCheckHistogram.spv", "writeOutOverlapChecksForInsertedBoundsRegionsHistogram", strides,
             sizeof(OverlapCheckParams));
    k.dispatch(32, { bAcc, bH, bSap, bRange, bSH, bAH, bOv, bHR, bBlk, bTot }, p);
    dev->flush();

    const auto gOverlap = downloadBuffer<uint64_t>(dev, bOv, total);
    const auto gHR = downloadBuffer<uint32_t>(dev, bHR, total * 2);

    size_t ovMism = 0, hrMism = 0;
    for (uint32_t i = 0; i < total; ++i)
    {
        if (gOverlap[i] != refOverlap[i]) ++ovMism;
        if (gHR[i * 2] != refHR[i * 2] || gHR[i * 2 + 1] != refHR[i * 2 + 1]) ++hrMism;
    }
    PXG_CHECK_MSG(ovMism == 0, "%zu overlap-count mismatches (total=%u)", ovMism, total);
    PXG_CHECK_MSG(hrMism == 0, "%zu handleRegion mismatches", hrMism);
    PXG_CHECK_MSG(total > 0, "expected nonzero comparisons");

    for (PxgBufferHandle h : { bAcc, bH, bSap, bRange, bSH, bAH, bOv, bHR, bBlk, bTot })
        dev->destroyBuffer(h);
}

// ---- passes 2+3: 64-bit device-wide exclusive prefix scan (sum > 2^32) ----
PXG_TEST(broadphaseOverlapCheckHistogram_scan64)
{
    std::mt19937 rng(20020);
    const uint32_t n = 100000; // regionAccumTotal
    std::vector<uint64_t> counts(n);
    for (uint32_t i = 0; i < n; ++i)
        counts[i] = 50000 + rng() % 10001; // in [50000,60000] -> sum in [5.0e9, 6.0e9] > 2^32

    // reference: 64-bit exclusive prefix + grand total
    std::vector<uint64_t> ref(n);
    uint64_t a = 0;
    for (uint32_t i = 0; i < n; ++i) { ref[i] = a; a += counts[i]; }
    const uint64_t refTotal = a;
    PXG_CHECK_MSG(refTotal > (uint64_t(1) << 32), "test setup: total %llu should exceed 2^32",
                  (unsigned long long) refTotal);

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<uint64_t> blkInit(TOTAL_REGION_SIZE, 0), totInit(1, 0);
    std::vector<uint32_t> dummy(16, 0);

    PxgBufferHandle bOv = createBufferFrom(dev, counts);
    PxgBufferHandle bBlk = createBufferFrom(dev, blkInit);
    PxgBufferHandle bTot = createBufferFrom(dev, totInit);
    PxgBufferHandle bDummy = createBufferFrom(dev, dummy);

    OverlapCheckParams p = { 0, n, 32 };
    const std::vector<uint32_t> strides = { 4, 4, 4, 4, 4, 4, 8, 4, 8, 8 };
    // bindings 0-5,7 unused by passes 2/3 -> bind a dummy; 6=overlap, 8=block, 9=total
    const std::vector<PxgBufferHandle> buffers = { bDummy, bDummy, bDummy, bDummy, bDummy, bDummy,
                                                   bOv, bDummy, bBlk, bTot };

    Kernel k2("broadphaseOverlapCheckHistogram.spv", "computeOverlapChecksForRegionsHistogram", strides,
              sizeof(OverlapCheckParams));
    k2.dispatch(32, buffers, p);
    dev->barrier();
    Kernel k3("broadphaseOverlapCheckHistogram.spv", "outputOverlapChecksForRegionHistogram", strides,
              sizeof(OverlapCheckParams));
    k3.dispatch(32, buffers, p);
    dev->flush();

    const auto gOverlap = downloadBuffer<uint64_t>(dev, bOv, n);
    const uint64_t gTotal = downloadBuffer<uint64_t>(dev, bTot, 1)[0];

    size_t mism = 0;
    uint32_t firstBad = 0xffffffffu;
    for (uint32_t i = 0; i < n; ++i)
        if (gOverlap[i] != ref[i]) { if (firstBad == 0xffffffffu) firstBad = i; ++mism; }
    if (mism)
        PXG_CHECK_MSG(mism == 0, "%zu prefix mismatches (first at %u: gpu=%llu ref=%llu)", mism, firstBad,
                      (unsigned long long) gOverlap[firstBad], (unsigned long long) ref[firstBad]);
    else
        PXG_CHECK_MSG(mism == 0, "%zu prefix mismatches", mism);
    PXG_CHECK_MSG(gTotal == refTotal, "total: gpu=%llu ref=%llu", (unsigned long long) gTotal,
                  (unsigned long long) refTotal);

    for (PxgBufferHandle h : { bOv, bBlk, bTot, bDummy })
        dev->destroyBuffer(h);
}

} // namespace
