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

// P9.42 -- verification of frozenHistogram.slang, the three-kernel frozen/unfrozen
// shape compaction of updateTransformAndBoundArray.cu:245-428, on the ENGINE ABI
// (one PxgSimulationCoreDesc pointer, grids 32 / 32 / 64 of 256).
//
// The reference is a transcription of the CUDA, stage by stage, and every stage is
// compared separately because the chain's contract lives BETWEEN them:
//   1. after pass 1 the flag arrays hold BLOCK-LOCAL exclusive prefixes over the
//      .cu:283 partition (contiguous chunks of nbIterationsPerBlock * 256 per block,
//      not grid-stride) and BlockAndRes[0..32) holds the 32 block totals -- ALL 32,
//      zero for idle blocks;
//   2. after pass 2 the prefixes are device-wide and the two grand totals sit in
//      the descriptor's own mTotalFrozenShapes / mTotalUnfrozenShapes (bytes
//      144 / 148), which is where the host reads them from (PxgSimulationCore.cpp:
//      2224, :2306-2307) -- a port that wrote them anywhere else would report zero
//      frozen shapes forever;
//   3. after pass 3 BlockAndRes[0..total) holds the compacted shape indices via
//      reduction.cuh:728-752 binarySearch, and BlockAndRes[total..) is UNTOUCHED
//      (the block totals for i < 32, the sentinel above).
//
// The lane-level detail the reference keeps as written: the ballot tests `v != 0`
// (.cu:295) while the warp total adds the RAW value (.cu:300).  The engine only
// ever writes 0/1, where both agree; one case here plants a 2 to pin the
// transcription rather than a normalised re-derivation.
//
// n == 0 runs with NULL flag pointers: the .cu's zero-trip loops never touch them,
// and llvmpipe faults on a speculated null BDA load, so a hoisted load fails here
// by crashing rather than by mismatching.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "PxNodeIndex.h" // completes PxNodeIndex for the engine desc header
#include "PxgSimulationCoreDesc.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

static std::vector<uint8_t> fhShimLoader(const char* file) { return loadShaderBlob(file); }

static const uint32_t kBlock = 256;
static const uint32_t kNumBlocks = 32;
static const uint32_t kCreateGrid = 64; // PxgSimulationCoreKernelGridDim::CREATE_FROZEN_UNFROZEN_ARRAY
static const uint32_t kSentinel = 0xdeadbeefu;

static uint32_t popc(uint32_t v) { return uint32_t(__builtin_popcount(v)); }

// .cu:245-341 -- the partition, the per-lane ballot count, the raw-value warp
// total, the previous-tiles accumulator read before the barrier, the exclusive
// scan of the 8 warp totals, the in-place overwrite, the 32 block totals.
static void refPass1(const std::vector<uint32_t>& in, std::vector<uint32_t>& out, std::vector<uint32_t>& blockTotals)
{
    const uint32_t n = uint32_t(in.size());
    const uint32_t nbBlocksRequired = (n + kBlock - 1) / kBlock;
    const uint32_t nbIterationsPerBlock = (nbBlocksRequired + kNumBlocks - 1) / kNumBlocks;
    out.assign(n, 0u);
    blockTotals.assign(kNumBlocks, 0u);

    for (uint32_t b = 0; b < kNumBlocks; ++b)
    {
        uint32_t sBlockAccumulator = 0;
        for (uint32_t it = 0; it < nbIterationsPerBlock; ++it)
        {
            uint32_t vals[kBlock], laneAccum[kBlock], warpTotals[kBlock / 32];
            for (uint32_t t = 0; t < kBlock; ++t)
            {
                const uint32_t workIndex = it * kBlock + t + nbIterationsPerBlock * b * kBlock; // .cu:283
                vals[t] = workIndex < n ? in[workIndex] : 0u;                                  // .cu:286-291
            }
            for (uint32_t warp = 0; warp < kBlock / 32; ++warp)
            {
                uint32_t ballot = 0;
                for (uint32_t l = 0; l < 32; ++l)
                    if (vals[warp * 32 + l] != 0u) ballot |= 1u << l;                          // __ballot_sync(FULL_MASK, v)
                for (uint32_t l = 0; l < 32; ++l)
                    laneAccum[warp * 32 + l] = popc(ballot & ((1u << l) - 1u));                // .cu:293-296
                warpTotals[warp] = laneAccum[warp * 32 + 31] + vals[warp * 32 + 31];           // .cu:298-302 (RAW value)
            }
            const uint32_t prevBlockAccumulator = sBlockAccumulator;                            // .cu:304-305
            uint32_t warpExclusive[kBlock / 32], acc = 0;
            for (uint32_t w = 0; w < kBlock / 32; ++w)                                          // .cu:314-317
            {
                warpExclusive[w] = acc;
                acc += warpTotals[w];
            }
            sBlockAccumulator += acc;                                                          // .cu:321 (output + value at lane 7)
            for (uint32_t t = 0; t < kBlock; ++t)
            {
                const uint32_t workIndex = it * kBlock + t + nbIterationsPerBlock * b * kBlock;
                if (workIndex < n)
                    out[workIndex] = laneAccum[t] + prevBlockAccumulator + warpExclusive[t / 32]; // .cu:331-332
            }
        }
        blockTotals[b] = sBlockAccumulator; // .cu:336-340
    }
}

// .cu:343-403 -- exclusive scan of the 32 block totals, grand total, fold.
static void refPass2(std::vector<uint32_t>& pfx, const std::vector<uint32_t>& blockTotals, uint32_t& total)
{
    const uint32_t n = uint32_t(pfx.size());
    uint32_t blockExclusive[kNumBlocks], acc = 0;
    for (uint32_t b = 0; b < kNumBlocks; ++b)      // .cu:369-375
    {
        blockExclusive[b] = acc;
        acc += blockTotals[b];
    }
    total = acc;                                    // .cu:380-381 (exclusive at 31 + own)
    const uint32_t totalBlockRequired = (n + kBlock - 1) / kBlock;
    const uint32_t numIterationPerBlock = (totalBlockRequired + kNumBlocks - 1) / kNumBlocks; // constant 32
    for (uint32_t b = 0; b < kNumBlocks; ++b)
        for (uint32_t it = 0; it < numIterationPerBlock; ++it)
            for (uint32_t t = 0; t < kBlock; ++t)
            {
                const uint32_t workIndex = it * kBlock + t + numIterationPerBlock * b * kBlock; // .cu:395
                if (workIndex < n) pfx[workIndex] = pfx[workIndex] + blockExclusive[b];         // .cu:399-400
            }
}

// reduction.cuh:728-752
static uint32_t refBinarySearch(const std::vector<uint32_t>& data, uint32_t numElements, uint32_t value)
{
    uint32_t left = 0, right = numElements;
    while (left < right)
    {
        const uint32_t pos = (left + right) / 2;
        const uint32_t element = data[pos];
        if (element <= value) left = pos + 1;
        else right = pos;
    }
    return left ? left - 1 : 0;
}

// .cu:405-428
static void refPass3(const std::vector<uint32_t>& pfx, uint32_t total, std::vector<uint32_t>& res)
{
    res.assign(total, 0u);
    for (uint32_t i = 0; i < total; ++i)
        res[i] = refBinarySearch(pfx, uint32_t(pfx.size()), i);
}

struct Expect
{
    std::vector<uint32_t> pass1, blockTotals, pass2, res;
    uint32_t total = 0;
};

static void reference(const std::vector<uint32_t>& in, Expect& e)
{
    refPass1(in, e.pass1, e.blockTotals);
    e.pass2 = e.pass1;
    refPass2(e.pass2, e.blockTotals, e.total);
    refPass3(e.pass2, e.total, e.res);
}

static size_t countMismatch(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, uint32_t& first)
{
    size_t m = 0;
    first = 0xffffffffu;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) { if (first == 0xffffffffu) first = uint32_t(i); ++m; }
    return m;
}

// frozenPct / unfrozenPct: flag densities; rawValue: plant a 2 in every 7th flagged
// frozen entry (transcription pin, see the header).
void runFrozenCase(const char* tag, uint32_t seed, uint32_t n, uint32_t frozenPct, uint32_t unfrozenPct, bool rawValue)
{
    using namespace physx;
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    std::mt19937 rng(seed);
    std::vector<uint32_t> frozen(n), unfrozen(n);
    uint32_t planted = 0;
    for (uint32_t i = 0; i < n; ++i)
    {
        frozen[i] = (rng() % 100u) < frozenPct ? 1u : 0u;
        unfrozen[i] = (rng() % 100u) < unfrozenPct ? 1u : 0u;
        if (rawValue && frozen[i] && (++planted % 7u) == 0u) frozen[i] = 2u;
    }

    Expect eF, eU;
    reference(frozen, eF);
    reference(unfrozen, eU);

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "frozenHistogram.spv", "computeFrozenAndUnfrozenHistogramLaunch", 8u, 0 },
        { "frozenHistogram.spv", "outputFrozenAndUnfrozenHistogram", 8u, 0 },
        { "frozenHistogram.spv", "createFrozenAndUnfrozenArray", 8u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 3, &fhShimLoader);

    // BlockAndRes: PxMax(32, n) entries (PxgSimulationCore.cpp:2150-2151), sentinel-filled
    const uint32_t nbRes = n > kNumBlocks ? n : kNumBlocks;
    const std::vector<uint32_t> resInit(nbRes, kSentinel);

    PxgDevicePtr dFrozen = 0, dUnfrozen = 0, dResF = 0, dResU = 0, dDesc = 0;
    if (n)
    {
        shim.memAlloc(&dFrozen, n * 4);
        shim.memAlloc(&dUnfrozen, n * 4);
    }
    shim.memAlloc(&dResF, nbRes * 4);
    shim.memAlloc(&dResU, nbRes * 4);
    shim.memAlloc(&dDesc, sizeof(PxgSimulationCoreDesc));
    if ((n && (!dFrozen || !dUnfrozen)) || !dResF || !dResU || !dDesc)
    {
        std::printf("  NOTE: shim has no device-address allocations (no BDA) -> skip\n");
        return;
    }
    if (n)
    {
        shim.memcpyHtoD(dFrozen, frozen.data(), n * 4);
        shim.memcpyHtoD(dUnfrozen, unfrozen.data(), n * 4);
    }
    shim.memcpyHtoD(dResF, resInit.data(), nbRes * 4);
    shim.memcpyHtoD(dResU, resInit.data(), nbRes * 4);

    // the REAL engine desc: the four arrays, the count, and sentinels in the two
    // totals the chain must overwrite (n == 0 leaves the flag pointers NULL)
    PxgSimulationCoreDesc desc;
    std::memset(&desc, 0, sizeof(desc));
    desc.mFrozen = reinterpret_cast<PxU32*>(dFrozen);
    desc.mUnfrozen = reinterpret_cast<PxU32*>(dUnfrozen);
    desc.mFrozenBlockAndRes = reinterpret_cast<PxU32*>(dResF);
    desc.mUnfrozenBlockAndRes = reinterpret_cast<PxU32*>(dResU);
    desc.mNbTotalShapes = n;
    desc.mTotalFrozenShapes = kSentinel;
    desc.mTotalUnfrozenShapes = kSentinel;
    shim.memcpyHtoD(dDesc, &desc, sizeof(desc));

    std::vector<uint8_t> blob(8);
    std::memcpy(blob.data(), &dDesc, 8); // the engine's one-pointer argument list, {8}

    std::vector<uint32_t> gF(n), gU(n), gResF(nbRes), gResU(nbRes);
    PxgSimulationCoreDesc back;
    uint32_t first = 0;
    size_t m = 0;

    // ---- pass 1: block-local prefixes + 32 block totals; totals in the desc untouched
    PXG_CHECK_MSG(shim.launchKernel(0, kNumBlocks, 1, 1, blob.data(), 8u) == PxgResult::Success,
                  "[%s] launch computeFrozenAndUnfrozenHistogramLaunch", tag);
    shim.streamSynchronize();
    if (n)
    {
        shim.memcpyDtoH(gF.data(), dFrozen, n * 4);
        shim.memcpyDtoH(gU.data(), dUnfrozen, n * 4);
    }
    shim.memcpyDtoH(gResF.data(), dResF, nbRes * 4);
    shim.memcpyDtoH(gResU.data(), dResU, nbRes * 4);
    m = countMismatch(gF, eF.pass1, first);
    PXG_CHECK_MSG(m == 0, "[%s n=%u] pass1 frozen: %zu block-local prefix mismatches (first at %u: gpu=%u ref=%u)",
                  tag, n, m, first, m ? gF[first] : 0u, m ? eF.pass1[first] : 0u);
    m = countMismatch(gU, eU.pass1, first);
    PXG_CHECK_MSG(m == 0, "[%s n=%u] pass1 unfrozen: %zu block-local prefix mismatches (first at %u: gpu=%u ref=%u)",
                  tag, n, m, first, m ? gU[first] : 0u, m ? eU.pass1[first] : 0u);
    for (uint32_t b = 0; b < kNumBlocks; ++b)
    {
        PXG_CHECK_MSG(gResF[b] == eF.blockTotals[b], "[%s n=%u] pass1 frozen block total[%u]: gpu=%u ref=%u", tag, n, b,
                      gResF[b], eF.blockTotals[b]);
        PXG_CHECK_MSG(gResU[b] == eU.blockTotals[b], "[%s n=%u] pass1 unfrozen block total[%u]: gpu=%u ref=%u", tag, n,
                      b, gResU[b], eU.blockTotals[b]);
    }
    for (uint32_t i = kNumBlocks; i < nbRes; ++i)
    {
        PXG_CHECK_MSG(gResF[i] == kSentinel && gResU[i] == kSentinel,
                      "[%s n=%u] pass1 wrote BlockAndRes[%u] beyond the 32 block totals", tag, n, i);
        if (gResF[i] != kSentinel || gResU[i] != kSentinel) break;
    }
    std::memset(&back, 0, sizeof(back));
    shim.memcpyDtoH(&back, dDesc, sizeof(back));
    PXG_CHECK_MSG(back.mTotalFrozenShapes == kSentinel && back.mTotalUnfrozenShapes == kSentinel,
                  "[%s n=%u] pass1 must not write the desc totals (got %u / %u)", tag, n, back.mTotalFrozenShapes,
                  back.mTotalUnfrozenShapes);

    // ---- pass 2: device-wide prefixes; grand totals in the DESCRIPTOR
    PXG_CHECK_MSG(shim.launchKernel(1, kNumBlocks, 1, 1, blob.data(), 8u) == PxgResult::Success,
                  "[%s] launch outputFrozenAndUnfrozenHistogram", tag);
    shim.streamSynchronize();
    if (n)
    {
        shim.memcpyDtoH(gF.data(), dFrozen, n * 4);
        shim.memcpyDtoH(gU.data(), dUnfrozen, n * 4);
    }
    m = countMismatch(gF, eF.pass2, first);
    PXG_CHECK_MSG(m == 0, "[%s n=%u] pass2 frozen: %zu device-wide prefix mismatches (first at %u: gpu=%u ref=%u)", tag,
                  n, m, first, m ? gF[first] : 0u, m ? eF.pass2[first] : 0u);
    m = countMismatch(gU, eU.pass2, first);
    PXG_CHECK_MSG(m == 0, "[%s n=%u] pass2 unfrozen: %zu device-wide prefix mismatches (first at %u: gpu=%u ref=%u)",
                  tag, n, m, first, m ? gU[first] : 0u, m ? eU.pass2[first] : 0u);
    std::memset(&back, 0, sizeof(back));
    shim.memcpyDtoH(&back, dDesc, sizeof(back));
    PXG_CHECK_MSG(back.mTotalFrozenShapes == eF.total, "[%s n=%u] desc.mTotalFrozenShapes: gpu=%u ref=%u", tag, n,
                  back.mTotalFrozenShapes, eF.total);
    PXG_CHECK_MSG(back.mTotalUnfrozenShapes == eU.total, "[%s n=%u] desc.mTotalUnfrozenShapes: gpu=%u ref=%u", tag, n,
                  back.mTotalUnfrozenShapes, eU.total);
    // the block totals are still in place for pass 3 to be indifferent to
    shim.memcpyDtoH(gResF.data(), dResF, nbRes * 4);
    for (uint32_t b = 0; b < kNumBlocks; ++b)
    {
        PXG_CHECK_MSG(gResF[b] == eF.blockTotals[b], "[%s n=%u] pass2 disturbed block total[%u]", tag, n, b);
        if (gResF[b] != eF.blockTotals[b]) break;
    }

    // ---- pass 3: compacted indices over the BlockAndRes buffers; the tail untouched
    PXG_CHECK_MSG(shim.launchKernel(2, kCreateGrid, 1, 1, blob.data(), 8u) == PxgResult::Success,
                  "[%s] launch createFrozenAndUnfrozenArray", tag);
    shim.streamSynchronize();
    shim.memcpyDtoH(gResF.data(), dResF, nbRes * 4);
    shim.memcpyDtoH(gResU.data(), dResU, nbRes * 4);
    std::vector<uint32_t> expF(nbRes), expU(nbRes);
    for (uint32_t i = 0; i < nbRes; ++i)
    {
        expF[i] = i < eF.total ? eF.res[i] : (i < kNumBlocks ? eF.blockTotals[i] : kSentinel);
        expU[i] = i < eU.total ? eU.res[i] : (i < kNumBlocks ? eU.blockTotals[i] : kSentinel);
    }
    m = countMismatch(gResF, expF, first);
    PXG_CHECK_MSG(m == 0, "[%s n=%u total=%u] pass3 frozen indices: %zu mismatches (first at %u: gpu=%u ref=%u)", tag,
                  n, eF.total, m, first, m ? gResF[first] : 0u, m ? expF[first] : 0u);
    m = countMismatch(gResU, expU, first);
    PXG_CHECK_MSG(m == 0, "[%s n=%u total=%u] pass3 unfrozen indices: %zu mismatches (first at %u: gpu=%u ref=%u)",
                  tag, n, eU.total, m, first, m ? gResU[first] : 0u, m ? expU[first] : 0u);

    // the compaction's meaning, checked against the INPUT rather than the reference:
    // res[k] is the index of the k-th flagged shape (0/1 inputs only)
    if (!rawValue)
    {
        uint32_t k = 0;
        bool ok = true;
        for (uint32_t i = 0; i < n && ok; ++i)
            if (frozen[i]) { ok = (k < eF.total) && gResF[k] == i; ++k; }
        PXG_CHECK_MSG(ok && k == eF.total, "[%s n=%u] the frozen list is the ordered list of flagged shapes (%u of %u)",
                      tag, n, k, eF.total);
    }

    for (PxgDevicePtr p : { dFrozen, dUnfrozen, dResF, dResU, dDesc })
        if (p) shim.memFree(p);
}

PXG_TEST(frozenHistogram_chain)
{
    // sizes straddling lane (32), tile (256) and partition (256 * 32 = 8192)
    // boundaries; 20000 makes nbIterationsPerBlock = 3 with five idle blocks.
    runFrozenCase("empty-null-ptrs", 4201, 0, 30, 30, false);
    runFrozenCase("one", 4202, 1, 100, 0, false);
    runFrozenCase("five", 4203, 5, 60, 40, false);
    runFrozenCase("lane-31", 4204, 31, 30, 30, false);
    runFrozenCase("lane-32", 4205, 32, 50, 50, false);
    runFrozenCase("lane-33", 4206, 33, 30, 70, false);
    runFrozenCase("tile-255", 4207, 255, 30, 30, false);
    runFrozenCase("tile-256", 4208, 256, 30, 30, false);
    runFrozenCase("tile-257", 4209, 257, 30, 30, false);
    runFrozenCase("all-zero-300", 4210, 300, 0, 0, false);
    runFrozenCase("all-one-300", 4211, 300, 100, 100, false);
    runFrozenCase("partition-8191", 4212, 8191, 25, 25, false);
    runFrozenCase("partition-8192", 4213, 8192, 25, 25, false);
    runFrozenCase("partition-8193", 4214, 8193, 25, 25, false);
    runFrozenCase("multi-iter-20000", 4215, 20000, 20, 35, false);
    // the raw-value pin: ballot counts `!= 0`, the warp total adds the value itself
    runFrozenCase("raw-values-3000", 4216, 3000, 30, 30, true);
}

} // namespace
