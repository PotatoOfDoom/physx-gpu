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

// Verification of npPairRemoveSwap.slang: the sparse-remove Stage5 data swap
// (removeContactManagers_Stage5 and _CvxTri).  A run-sum buffer (the exclusive
// prefix sum of the keep/drop marks, produced by npPairManagement.slang) drives a
// compaction that moves kept elements from the removed tail into the holes in the
// kept region.  The GPU payload + manifold arrays are compared bit-exactly against
// a C++ reference replicating getNbSwapsRequired/getSwapIndices, and a semantic
// set-check confirms the compacted region [0, N-K) holds exactly the kept indices.

#include "pxgTestFramework.h"

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct RemoveSwapParams { uint32_t totalCount, nbToRemove, nbFloat4, copyManifold, numGroups; };
struct F4 { float x, y, z, w; };

// gpucommon reduction.cuh binarySearch(): largest idx s.t. data[idx] <= value, else 0
uint32_t binarySearch(const std::vector<uint32_t>& data, uint32_t n, uint32_t value)
{
    uint32_t left = 0, right = n;
    while (left < right)
    {
        const uint32_t pos = (left + right) / 2;
        if (data[pos] <= value) left = pos + 1;
        else right = pos;
    }
    return left ? left - 1 : 0;
}

// build the exclusive-prefix run-sum for N pairs with K removed (distinct indices)
static void buildRunSum(uint32_t N, uint32_t K, const std::vector<uint32_t>& removeIndices,
                        std::vector<uint32_t>& runSum)
{
    const uint32_t newArraySize = N - K;
    std::vector<uint32_t> marks(N);
    for (uint32_t i = 0; i < N; ++i) marks[i] = (i < newArraySize) ? 0u : 1u;
    for (uint32_t j = 0; j < K; ++j)
    {
        const uint32_t idx = removeIndices[j];
        marks[idx] = (idx < newArraySize) ? 1u : 0u;
    }
    runSum.resize(N);
    uint32_t run = 0;
    for (uint32_t i = 0; i < N; ++i) { runSum[i] = run; run += marks[i]; }
}

static void runVariant(const char* entry, uint32_t granularity)
{
    std::mt19937 rng(40010 + granularity);
    const uint32_t N = 8000;
    const uint32_t K = 1500;
    const uint32_t newArraySize = N - K;
    const uint32_t nbFloat4 = (granularity == 16) ? 20 : 48;

    std::vector<uint32_t> perm(N);
    for (uint32_t i = 0; i < N; ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<uint32_t> removeIndices(perm.begin(), perm.begin() + K);
    std::vector<bool> removed(N, false);
    for (uint32_t j = 0; j < K; ++j) removed[removeIndices[j]] = true;

    std::vector<uint32_t> runSum;
    buildRunSum(N, K, removeIndices, runSum);
    const uint32_t nbToSwap = runSum[newArraySize];

    // payload[i] = i (identity, so we can track); manifold block = (i, f)
    std::vector<uint32_t> payload(N);
    for (uint32_t i = 0; i < N; ++i) payload[i] = i;
    std::vector<F4> manifolds(N * nbFloat4);
    for (uint32_t i = 0; i < N; ++i)
        for (uint32_t f = 0; f < nbFloat4; ++f)
            manifolds[i * nbFloat4 + f] = { float(i), float(f), float(i * 7 + f), -1.f };

    // ---- C++ reference: replicate the swaps ----
    std::vector<uint32_t> refPayload = payload;
    std::vector<F4> refManifolds = manifolds;
    for (uint32_t i = 0; i < nbToSwap; ++i)
    {
        const uint32_t dst = binarySearch(runSum, N, i);
        const uint32_t src = binarySearch(runSum, N, i + nbToSwap);
        refPayload[dst] = payload[src];
        for (uint32_t f = 0; f < nbFloat4; ++f)
            refManifolds[dst * nbFloat4 + f] = manifolds[src * nbFloat4 + f];
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t numGroups = 256;

    PxgBufferHandle bRun = createBufferFrom(dev, runSum);
    PxgBufferHandle bPay = createBufferFrom(dev, payload);
    PxgBufferHandle bMan = createBufferFrom(dev, manifolds);

    Kernel k("npPairRemoveSwap.spv", entry, { 4, 4, 16 }, sizeof(RemoveSwapParams));
    RemoveSwapParams p = { N, K, nbFloat4, 1u, numGroups };
    k.dispatch(numGroups, { bRun, bPay, bMan }, p);
    dev->flush();

    const auto gPay = downloadBuffer<uint32_t>(dev, bPay, N);
    const auto gMan = downloadBuffer<F4>(dev, bMan, N * nbFloat4);

    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i) if (gPay[i] != refPayload[i]) ++mm;
    PXG_CHECK_MSG(mm == 0, "%s payload: %zu mismatches", entry, mm);

    size_t mmm = 0;
    for (uint32_t e = 0; e < N * nbFloat4; ++e)
    {
        const F4& a = gMan[e];
        const F4& r = refManifolds[e];
        if (a.x != r.x || a.y != r.y || a.z != r.z || a.w != r.w) ++mmm;
    }
    PXG_CHECK_MSG(mmm == 0, "%s manifold: %zu mismatches", entry, mmm);

    // semantic set-check: compacted region [0, N-K) holds exactly the kept indices
    std::vector<uint32_t> keptExpected, keptGot;
    for (uint32_t i = 0; i < N; ++i) if (!removed[i]) keptExpected.push_back(i);
    for (uint32_t i = 0; i < newArraySize; ++i) keptGot.push_back(gPay[i]);
    std::sort(keptExpected.begin(), keptExpected.end());
    std::sort(keptGot.begin(), keptGot.end());
    const bool setOk = (keptExpected == keptGot);
    PXG_CHECK_MSG(setOk, "%s compacted region is not exactly the kept set", entry);

    dev->destroyBuffer(bRun); dev->destroyBuffer(bPay); dev->destroyBuffer(bMan);
}

PXG_TEST(npPairRemoveSwap_stage5)      { runVariant("removeContactManagers_Stage5", 16); }
PXG_TEST(npPairRemoveSwap_stage5CvxTri){ runVariant("removeContactManagers_Stage5_CvxTri", 32); }

} // namespace
