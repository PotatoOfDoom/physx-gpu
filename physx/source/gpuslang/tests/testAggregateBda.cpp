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

// P9.92: aggregateBda.slang -- the GPU broadphase's aggregate path (aggregate.cu) on the
// ENGINE launch ABI.
//
// Every kernel is driven with the push blob PxgAABBManager builds (the void** launch form)
// and the engine's grid, over device memory laid out as the real PxgAggregate (136 bytes),
// PxgAggregateDesc (136), PxgAggregatePair (12), Bp::VolumeData (16) and the broadphase
// desc (pxgSlangBpDesc.slang, pinned by testBpDescLayout).  The oracle is a literal C++
// transcription of each CUDA kernel run on the same buffers, and every buffer is compared
// BITWISE afterwards, bytes the CUDA leaves alone included (poisoned on both sides first).
//
// Two things are compared as MULTISETS rather than word by word, and only those: the
// found/lost pair reports.  Their slots come from a global atomicAdd per warp, so the
// order of the warps' blocks in the report is the order in which the warps reached the
// counter -- unordered in the CUDA as well.  Their COUNTS are exact.  The overflow test
// runs a single aggregate, where the order is fixed, and compares those reports word by
// word.
//
//   aggregateBda_dirtyBounds  updateDirtyAggregate (new and existing records), clearDirty,
//                             markAggregateBoundsUpdatedBitmap, updateAggregateBounds (the
//                             select folds and the xor butterfly on +-0 ties and NaNs),
//                             markAddedAndDeletedAggregatedBounds
//   aggregateBda_sortSelf     two frames of sortAndUpdateAggregateProjections (the eight-pass
//                             warp radix sort with its padding lanes and temp halves) and
//                             doSelfCollision (found, then lost pairs), 34 aggregates of 1..70
//                             members, plus a one-aggregate report overflow
//   aggregateBda_pairs        the aggregate-pair pool: addAggPairsStage1/2 into free and new
//                             slots, doAggPairCollisions over aggregate-aggregate and
//                             aggregate-actor pairs (found and lost; a removed handle; a dead
//                             pair), removeAggPairsStage1/2/3 over two waves (40000 pairs),
//                             aggCopyReports, and every overflow flag

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

static std::vector<uint8_t> aggLoader(const char* file) { return loadShaderBlob(file); }

static const uint32_t POISON = 0xA5A5A5A5u;
static const uint32_t WARP = 32;

enum : uint32_t
{
    K_UPD_DIRTY, K_CLEAR_DIRTY, K_MARK_BITMAP, K_UPD_BOUNDS, K_MARK_ADD_DEL, K_SORT, K_SELF, K_PAIR,
    K_COPY, K_ADD1, K_ADD2, K_REM1, K_REM2, K_REM3, K_COUNT
};

PxgCudaShim* makeShim(PxgSlangComputeDevice* dev)
{
    static const PxgKernelDef manifest[K_COUNT] = {
        { "aggregateBda.spv", "updateDirtyAggregate", 56u, 0 },
        { "aggregateBda.spv", "clearDirtyAggregates", 32u, 0 },
        { "aggregateBda.spv", "markAggregateBoundsUpdatedBitmap", 16u, 0 },
        { "aggregateBda.spv", "updateAggregateBounds", 24u, 0 },
        { "aggregateBda.spv", "markAddedAndDeletedAggregatedBounds", 48u, 0 },
        { "aggregateBda.spv", "sortAndUpdateAggregateProjections", 16u, 0 },
        { "aggregateBda.spv", "doSelfCollision", 16u, 0 },
        { "aggregateBda.spv", "doAggPairCollisions", 16u, 0 },
        { "aggregateBda.spv", "aggCopyReports", 8u, 0 },
        { "aggregateBda.spv", "addAggPairsStage1", 16u, 0 },
        { "aggregateBda.spv", "addAggPairsStage2", 16u, 0 },
        { "aggregateBda.spv", "removeAggPairsStage1", 8u, 0 },
        { "aggregateBda.spv", "removeAggPairsStage2", 8u, 0 },
        { "aggregateBda.spv", "removeAggPairsStage3", 8u, 0 },
    };
    return new PxgCudaShim(dev, manifest, K_COUNT, &aggLoader);
}

// ===========================================================================
// a flat "device memory" model: the oracle works on the same bytes the GPU sees.  Every
// buffer is a Region with a host image; device addresses map back to (region, offset).
// ===========================================================================
struct Region
{
    PxgDevicePtr dev = 0;
    std::vector<uint8_t> bytes;
};

struct Mem
{
    PxgCudaShim* shim = nullptr;
    std::vector<Region> regions;

    uint64_t alloc(size_t size, uint8_t fill = 0xA5)
    {
        Region r;
        r.bytes.assign((size + 15) & ~size_t(15), fill);
        shim->memAlloc(&r.dev, r.bytes.size());
        regions.push_back(std::move(r));
        return regions.back().dev;
    }
    Region* find(uint64_t addr, size_t& off)
    {
        for (Region& r : regions)
            if (addr >= r.dev && addr < r.dev + r.bytes.size()) { off = size_t(addr - r.dev); return &r; }
        return nullptr;
    }
    uint32_t& u32(uint64_t addr)
    {
        size_t off = 0;
        Region* r = find(addr, off);
        if (!r) { std::printf("  oracle: stray address %llx\n", (unsigned long long)addr); static uint32_t junk; return junk; }
        return *reinterpret_cast<uint32_t*>(r->bytes.data() + off);
    }
    uint64_t& u64(uint64_t addr) { return *reinterpret_cast<uint64_t*>(&u32(addr)); }
    float& f32(uint64_t addr) { return *reinterpret_cast<float*>(&u32(addr)); }
    uint8_t& u8(uint64_t addr) { return *(reinterpret_cast<uint8_t*>(&u32(addr & ~uint64_t(3))) + (addr & 3)); }
    void upload()
    {
        for (Region& r : regions) shim->memcpyHtoD(r.dev, r.bytes.data(), r.bytes.size());
    }
    // compare every region except the ones listed; returns the number of differing words
    size_t compare(const std::vector<uint64_t>& skip, const char* stage)
    {
        size_t bad = 0;
        for (Region& r : regions)
        {
            if (std::find(skip.begin(), skip.end(), r.dev) != skip.end()) continue;
            std::vector<uint8_t> g(r.bytes.size());
            shim->memcpyDtoH(g.data(), r.dev, g.size());
            for (size_t i = 0; i < g.size(); i += 4)
            {
                uint32_t a, b;
                std::memcpy(&a, g.data() + i, 4);
                std::memcpy(&b, r.bytes.data() + i, 4);
                if (a != b)
                {
                    if (bad < 6) std::printf("  MM %s region %llx word %zu: gpu %08x ref %08x\n", stage,
                                             (unsigned long long)r.dev, i / 4, a, b);
                    ++bad;
                }
            }
        }
        return bad;
    }
    std::vector<uint32_t> downloadU32(uint64_t addr, size_t n)
    {
        std::vector<uint32_t> v(n);
        if (n) shim->memcpyDtoH(v.data(), addr, n * 4);
        return v;
    }
};

// ---- the struct offsets ----
enum : uint32_t
{
    A_UPD = 0, A_BI0 = 8, A_BI1 = 16, A_SP0 = 24, A_SP1 = 32, A_SH0 = 40, A_SH1 = 48, A_SAP0 = 56, A_SAP1 = 64,
    A_SM0 = 72, A_SM1 = 80, A_CMP0 = 88, A_CMP1 = 96, A_PREVCMP = 104, A_PREVSIZE = 108, A_MINDEX = 112,
    A_SIZE = 116, A_FILTER = 120, A_ENV = 124, A_ISNEW = 128, AGG_SIZE = 136,

    D_FOUND = 0, D_LOST = 8, D_FOUNDMAP = 16, D_LOSTMAP = 24, D_SHFOUND = 32, D_SHLOST = 36, D_AGGS = 40,
    D_NUMAGG = 48, D_FREELIST = 56, D_FREEIDX = 64, D_REMBITMAP = 72, D_REMHISTO = 80, D_NBREMOVED = 88,
    D_AGGPAIRS = 96, D_AGGPAIRCOUNT = 104, D_MAXFL = 112, D_MAXAGG = 116, D_AGGOVF = 120, D_CANDOVF = 124,
    D_FLAGS = 128, DESC_SIZE = 136,

    // PxgBroadPhaseDesc (pxgSlangBpDesc.slang)
    BP_REMOVEDMAP = 72, BP_NEWIB = 104, BP_GROUPS = 112, BP_OLDIB = 128, BP_SHFOUNDAGG = 268,
    BP_FOUNDAGGREPORT = 296, BP_VOLUMEDATA = 344, BP_SIZE = 944,

    PAIR_SIZE = 12, VD_SIZE = 16, IAABB_SIZE = 24
};

// ===========================================================================
// the CUDA transcriptions
// ===========================================================================
uint32_t popc(uint32_t v) { return uint32_t(__builtin_popcount(v)); }
uint32_t maskUpTo(uint32_t b) { return uint32_t((uint64_t(2) << b) - 1u); }   // (2 << b) - 1 in PTX
uint32_t createHandle(uint32_t h, bool isStart, bool isNew) { return (h << 3) | uint32_t(isStart) | (uint32_t(isNew) << 1); }
uint32_t getHandle(uint32_t s) { return s >> 3; }
bool isStartProjection(uint32_t s) { return (s & 1u) != 0; }
float selMin(float a, float b) { return a < b ? a : b; }
float selMax(float a, float b) { return a > b ? a : b; }
float bitsF(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

bool aggValid(Mem& m, uint64_t a) { return m.u32(a + A_SIZE) > 0 && m.u32(a + A_MINDEX) != 0xffffffffu; }

bool iaabbIntersects(Mem& m, uint64_t base, uint32_t i, uint32_t j)
{
    const uint64_t a = base + uint64_t(i) * IAABB_SIZE, b = base + uint64_t(j) * IAABB_SIZE;
    return !(m.u32(b + 0) > m.u32(a + 12) || m.u32(a + 0) > m.u32(b + 12) ||
             m.u32(b + 4) > m.u32(a + 16) || m.u32(a + 4) > m.u32(b + 16) ||
             m.u32(b + 8) > m.u32(a + 20) || m.u32(a + 8) > m.u32(b + 20));
}

uint32_t binarySearchU(Mem& m, uint64_t data, uint32_t n, uint32_t value)
{
    uint32_t left = 0, right = n;
    while (left < right)
    {
        const uint32_t pos = (left + right) / 2;
        if (m.u32(data + pos * 4ull) <= value) left = pos + 1; else right = pos;
    }
    return left ? left - 1 : 0;
}

uint32_t getIthBit(uint32_t mask, uint32_t i)
{
    uint32_t lower = i, upper = 32;
    const uint32_t bitTarget = i + 1;
    while ((upper - lower) > 1)
    {
        const uint32_t mid = (lower + upper) / 2;
        const uint32_t nbBits = popc(mask & maskUpTo(mid));
        const uint32_t active = mask & (1u << mid);
        if (nbBits > bitTarget || (!active && nbBits == bitTarget)) upper = mid; else lower = mid;
    }
    return lower;
}

// a "side": device words or the single actor's one-box structure
struct OSide
{
    bool shared = false;
    uint64_t proj = 0, handles = 0, masks = 0, bi = 0;
    uint32_t sproj[2] = {}, sbi[2] = {};
    uint32_t nbObj = 0;
};
uint32_t sProj(Mem& m, const OSide& s, uint32_t i) { return s.shared ? s.sproj[i] : m.u32(s.proj + i * 4ull); }
uint32_t sHandle(Mem& m, const OSide& s, uint32_t i) { return s.shared ? createHandle(0, i == 1, false) : m.u32(s.handles + i * 4ull); }
uint32_t sMask(Mem& m, const OSide& s, uint32_t w) { return s.shared ? 1u : m.u32(s.masks + w * 4ull); }
uint32_t sBi(Mem& m, const OSide& s, uint32_t i) { return s.shared ? s.sbi[i] : m.u32(s.bi + i * 4ull); }

uint32_t getIndexSide(Mem& m, const OSide& s, uint32_t startIdx, uint32_t targetIndex, uint32_t baseIndex)
{
    const uint32_t mask = ~((1u << (startIdx & 31)) - 1u);
    uint32_t count = 0, target = targetIndex - baseIndex, w = startIdx / 32;
    uint32_t maskedResult = sMask(m, s, w) & mask;
    for (;;)
    {
        const uint32_t newCount = count + popc(maskedResult);
        if (newCount > target) break;
        count = newCount;
        maskedResult = sMask(m, s, w + 1);
        ++w;
    }
    return w * 32 + getIthBit(maskedResult, target - count);
}

struct Pair2 { uint32_t a, b; bool operator<(const Pair2& o) const { return a < o.a || (a == o.a && b < o.b); } bool operator==(const Pair2& o) const { return a == o.a && b == o.b; } };

// the report appends: in the oracle a report is the counter plus the list of pairs in
// warp order (the device order across warps is not asserted)
struct Report { uint64_t pairs = 0; uint64_t counterAddr = 0; uint32_t maxPairs = 0; };

void oAppend(Mem& m, const Report& r, const std::vector<Pair2>& warpPairs)
{
    uint32_t& counter = m.u32(r.counterAddr);
    const uint32_t out = counter;
    counter += uint32_t(warpPairs.size());
    for (uint32_t k = 0; k < warpPairs.size(); ++k)
        if (out + k < r.maxPairs)
        {
            m.u32(r.pairs + (out + k) * 8ull) = warpPairs[k].a;
            m.u32(r.pairs + (out + k) * 8ull + 4) = warpPairs[k].b;
        }
}

// radixSortSingleWarp (RadixSort.cuh:702) over one aggregate's arrays: eight stable
// four-bit passes over all 4 * numUint4 lanes, padding lanes carrying key 0xffffffff and
// their raw rank word, ping-ponging into the temp halves
void oRadixSortWarp(Mem& m, uint64_t keys, uint64_t ranks, uint32_t n, uint32_t numUint4, uint64_t tk, uint64_t tr)
{
    uint64_t k0 = keys, k1 = tk, r0 = ranks, r1 = tr;
    const uint32_t total = numUint4 * 4;
    for (uint32_t p = 0; p < 8; ++p)
    {
        std::vector<uint32_t> kk(total), rr(total);
        for (uint32_t i = 0; i < total; ++i)
        {
            kk[i] = i < n ? m.u32(k0 + i * 4ull) : 0xffffffffu;
            rr[i] = m.u32(r0 + i * 4ull);
        }
        std::vector<uint32_t> order(total);
        std::iota(order.begin(), order.end(), 0u);
        const uint32_t sb = p * 4;
        std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            return ((kk[a] >> sb) & 0xF) < ((kk[b] >> sb) & 0xF); });
        for (uint32_t i = 0; i < total; ++i)
        {
            m.u32(k1 + i * 4ull) = kk[order[i]];
            m.u32(r1 + i * 4ull) = rr[order[i]];
        }
        std::swap(k0, k1);
        std::swap(r0, r1);
    }
}

// aggregate.cu:231
void oSort(Mem& m, uint64_t bpDesc, uint64_t desc)
{
    const uint32_t nb = m.u32(desc + D_NUMAGG);
    const uint64_t aggs = m.u64(desc + D_AGGS);
    const uint64_t newIB = m.u64(bpDesc + BP_NEWIB);
    for (uint32_t g = 0; g < nb; ++g)
    {
        const uint64_t a = aggs + uint64_t(g) * AGG_SIZE;
        if (!aggValid(m, a)) continue;
        const uint32_t numBounds = m.u32(a + A_SIZE);
        const uint64_t boundIndices = m.u64(a + A_BI1), projections = m.u64(a + A_SP1), handles = m.u64(a + A_SH1),
                       sap = m.u64(a + A_SAP1), masks = m.u64(a + A_SM1);
        m.u64(a + A_SP1) = m.u64(a + A_SP0); m.u64(a + A_SP0) = projections;
        m.u64(a + A_SH1) = m.u64(a + A_SH0); m.u64(a + A_SH0) = handles;
        m.u64(a + A_SAP1) = m.u64(a + A_SAP0); m.u64(a + A_SAP0) = sap;
        m.u64(a + A_SM1) = m.u64(a + A_SM0); m.u64(a + A_SM0) = masks;
        const uint64_t comp = m.u64(a + A_CMP1); m.u64(a + A_CMP1) = m.u64(a + A_CMP0); m.u64(a + A_CMP0) = comp;
        m.u64(a + A_BI1) = m.u64(a + A_BI0); m.u64(a + A_BI0) = boundIndices;

        const uint64_t upd = m.u64(a + A_UPD);
        const uint32_t nbProj = numBounds * 2;
        for (uint32_t i = 0; i < nbProj; ++i)
        {
            const uint32_t lane = i & 31, odd = lane & 1, id = i / 2;
            const uint32_t handle = m.u32(upd + id * 4ull);
            m.u32(projections + i * 4ull) = m.u32(newIB + uint64_t(handle) * IAABB_SIZE + odd * 12);
            m.u32(handles + i * 4ull) = createHandle(id, odd == 0, false);
            if ((i % 2) == 0) m.u32(boundIndices + id * 4ull) = m.u32(upd + id * 4ull);
        }
        const uint32_t nbU4 = (nbProj + 3) / 4;
        oRadixSortWarp(m, projections, handles, nbProj, nbU4, projections + nbU4 * 16ull, handles + nbU4 * 16ull);
        for (uint32_t i = 0; i < nbProj; i += 32)
        {
            uint32_t mask = 0;
            for (uint32_t l = 0; l < 32; ++l)
            {
                const uint32_t idx = i + l;
                if (idx >= nbProj) continue;
                const uint32_t sh = m.u32(handles + idx * 4ull);
                const bool st = isStartProjection(sh);
                m.u32(sap + (getHandle(sh) * 2 + (st ? 0 : 1)) * 4ull) = idx;
                if (st) mask |= 1u << l;
            }
            m.u32(masks + (i / 32) * 4ull) = mask;
        }
    }
}

// aggregate.cu:467-554
void oGenerateAllPairs(Mem& m, uint64_t a, uint32_t totalComparisons, uint32_t nbBounds, const Report& rep,
                       uint64_t bounds, uint64_t prevBounds, uint64_t groups, uint32_t isLost)
{
    const uint64_t cmp = m.u64(a + (isLost ? A_CMP1 : A_CMP0)), sap = m.u64(a + (isLost ? A_SAP1 : A_SAP0)),
                   masks = m.u64(a + (isLost ? A_SM1 : A_SM0)), sh = m.u64(a + (isLost ? A_SH1 : A_SH0)),
                   bi = m.u64(a + (isLost ? A_BI1 : A_BI0));
    OSide s;
    s.masks = masks;
    for (uint32_t i = 0; i < totalComparisons; i += 32)
    {
        std::vector<Pair2> warp;
        for (uint32_t l = 0; l < 32; ++l)
        {
            const uint32_t idx = i + l;
            if (idx >= totalComparisons) continue;
            const uint32_t k = binarySearchU(m, cmp, nbBounds, idx);
            const uint32_t startIdx = m.u32(sap + k * 8ull);
            const uint32_t boundsIdx = m.u32(bi + k * 4ull);
            const uint32_t index = getIndexSide(m, s, startIdx + 1, idx, m.u32(cmp + k * 4ull));
            const uint32_t otherBoundsIdx = m.u32(bi + getHandle(m.u32(sh + index * 4ull)) * 4ull);
            if (m.u32(groups + otherBoundsIdx * 4ull) != m.u32(groups + boundsIdx * 4ull) &&
                iaabbIntersects(m, bounds, otherBoundsIdx, boundsIdx) && !iaabbIntersects(m, prevBounds, otherBoundsIdx, boundsIdx))
                warp.push_back({ std::min(boundsIdx, otherBoundsIdx), std::max(boundsIdx, otherBoundsIdx) });
        }
        oAppend(m, rep, warp);
    }
}

// aggregate.cu:557
void oSelf(Mem& m, uint64_t bpDesc, uint64_t desc)
{
    const uint32_t nb = m.u32(desc + D_NUMAGG);
    const uint64_t aggs = m.u64(desc + D_AGGS);
    const uint32_t maxPairs = m.u32(desc + D_MAXFL);
    for (uint32_t g = 0; g < nb; ++g)
    {
        const uint64_t a = aggs + uint64_t(g) * AGG_SIZE;
        uint32_t totalCount = 0;
        if (aggValid(m, a) && (m.u32(a + A_FILTER) & 1u))
        {
            const uint32_t numBounds = m.u32(a + A_SIZE);
            const uint64_t sap = m.u64(a + A_SAP0), masks = m.u64(a + A_SM0), cmp = m.u64(a + A_CMP0);
            for (uint32_t i = 0; i < numBounds; ++i)
            {
                const uint32_t startIdx = m.u32(sap + i * 8ull), endIdx = m.u32(sap + i * 8ull + 4);
                uint32_t mask = ~maskUpTo(startIdx & 31), count = 0;
                for (uint32_t w = startIdx / 32; w < endIdx / 32; ++w)
                {
                    count += popc(m.u32(masks + w * 4ull) & mask);
                    mask = 0xffffffffu;
                }
                mask &= maskUpTo(endIdx & 31);
                count += popc(m.u32(masks + (endIdx / 32) * 4ull) & mask);
                m.u32(cmp + i * 4ull) = count;
            }
            for (uint32_t i = 0; i < numBounds; ++i)
            {
                const uint32_t c = m.u32(cmp + i * 4ull);
                m.u32(cmp + i * 4ull) = totalCount;
                totalCount += c;
            }
            Report lost{ m.u64(desc + D_LOST), desc + D_SHLOST, maxPairs };
            Report found{ m.u64(desc + D_FOUND), desc + D_SHFOUND, maxPairs };
            if (!m.u8(a + A_ISNEW))
                oGenerateAllPairs(m, a, m.u32(a + A_PREVCMP), m.u32(a + A_PREVSIZE), lost,
                                  m.u64(bpDesc + BP_OLDIB), m.u64(bpDesc + BP_NEWIB), m.u64(bpDesc + BP_GROUPS), 1);
            oGenerateAllPairs(m, a, totalCount, m.u32(a + A_SIZE), found,
                              m.u64(bpDesc + BP_NEWIB), m.u64(bpDesc + BP_OLDIB), m.u64(bpDesc + BP_GROUPS), 0);
        }
        m.u32(a + A_PREVCMP) = totalCount;
    }
}

// aggregate.cu:737-904
uint32_t oCountComparisons(Mem& m, uint64_t bounds, uint32_t handle, const OSide& other, uint32_t nbProj, uint32_t& outStart)
{
    const uint32_t v0 = m.u32(bounds + uint64_t(handle) * IAABB_SIZE), v3 = m.u32(bounds + uint64_t(handle) * IAABB_SIZE + 12);
    uint32_t left = 0, right = nbProj;
    while (left < right) { const uint32_t pos = (left + right) / 2; if (sProj(m, other, pos) < v0) left = pos + 1; else right = pos; }
    const uint32_t startIdx = left;
    left = 0; right = nbProj;
    while (left < right) { const uint32_t pos = (left + right) / 2; if (sProj(m, other, pos) <= v3) left = pos + 1; else right = pos; }
    const uint32_t endIdx = left ? left - 1 : 0;
    uint32_t w = sMask(m, other, startIdx / 32) & ~((1u << (startIdx & 31)) - 1u);
    uint32_t total = 0;
    for (uint32_t i = startIdx / 32; i < endIdx / 32;)
    {
        total += popc(w);
        ++i;
        w = sMask(m, other, i);
    }
    w &= maskUpTo(endIdx & 31);
    total += popc(w);
    outStart = startIdx;
    return total;
}

void oBoxPruning(Mem& m, uint64_t bounds, uint64_t oldBounds, const OSide& objs, const OSide& other, uint64_t groups,
                 const Report& rep, uint32_t isFirst)
{
    const uint32_t nbProj = other.nbObj * 2;
    for (uint32_t i = 0; i < objs.nbObj; i += 32)
    {
        uint32_t cmp[32], start[32], total = 0;
        for (uint32_t l = 0; l < 32; ++l)
        {
            uint32_t c = 0, st = 0;
            if (i + l < objs.nbObj)
                c = oCountComparisons(m, bounds, sBi(m, objs, i + l), other, nbProj, st);
            cmp[l] = total;
            start[l] = st;
            total += c;
        }
        for (uint32_t b = 0; b < total; b += 32)
        {
            std::vector<Pair2> warp;
            for (uint32_t l = 0; l < 32; ++l)
            {
                const uint32_t cmpId = b + l;
                if (cmpId >= total) continue;
                uint32_t left = 0, right = 32;
                while (left < right) { const uint32_t pos = (left + right) / 2; if (cmp[pos] <= cmpId) left = pos + 1; else right = pos; }
                const uint32_t idx = left ? left - 1 : 0;
                const uint32_t handle = sBi(m, objs, i + idx);
                const uint32_t index = getIndexSide(m, other, start[idx], cmpId, cmp[idx]);
                const uint32_t otherHandle = sBi(m, other, getHandle(sHandle(m, other, index)));
                if (m.u32(groups + handle * 4ull) != m.u32(groups + otherHandle * 4ull) &&
                    iaabbIntersects(m, bounds, handle, otherHandle) && !iaabbIntersects(m, oldBounds, handle, otherHandle) &&
                    (isFirst || m.u32(bounds + uint64_t(handle) * IAABB_SIZE) != m.u32(bounds + uint64_t(otherHandle) * IAABB_SIZE)))
                    warp.push_back({ std::min(handle, otherHandle), std::max(handle, otherHandle) });
            }
            oAppend(m, rep, warp);
        }
    }
}

// aggregate.cu:928
void oPairs(Mem& m, uint64_t bpDesc, uint64_t desc)
{
    const uint64_t pairs = m.u64(desc + D_AGGPAIRS);
    const uint32_t nbPairs = std::min(m.u32(m.u64(desc + D_AGGPAIRCOUNT)), m.u32(desc + D_MAXAGG));
    const uint32_t maxFL = m.u32(desc + D_MAXFL);
    const uint64_t removed = m.u64(bpDesc + BP_REMOVEDMAP), bitmap = m.u64(desc + D_REMBITMAP);
    const uint64_t newIB = m.u64(bpDesc + BP_NEWIB), oldIB = m.u64(bpDesc + BP_OLDIB), groups = m.u64(bpDesc + BP_GROUPS);
    const uint64_t vd = m.u64(bpDesc + BP_VOLUMEDATA), aggs = m.u64(desc + D_AGGS);
    Report found{ m.u64(desc + D_FOUND), desc + D_SHFOUND, maxFL };
    Report lost{ m.u64(desc + D_LOST), desc + D_SHLOST, maxFL };
    for (uint32_t i = 0; i < nbPairs; ++i)
    {
        const uint64_t p = pairs + uint64_t(i) * PAIR_SIZE;
        const uint32_t h0 = m.u32(p), h1 = m.u32(p + 4);
        const bool isNew = m.u8(p + 8) != 0, isDead = m.u8(p + 9) != 0;
        bool shouldRemove = false;
        if (!isDead)
        {
            auto removedBit = [&](uint32_t h) { return (m.u32(removed + (h / 32) * 4ull) & (1u << (h & 31))) != 0; };
            if (removedBit(h0) || removedBit(h1))
            {
                m.u32(bitmap + i * 4ull) = 1;
                continue;
            }
            shouldRemove = !iaabbIntersects(m, newIB, h0, h1);
            const uint32_t va0 = m.u32(vd + uint64_t(h0) * VD_SIZE + 8), va1 = m.u32(vd + uint64_t(h1) * VD_SIZE + 8);
            const bool isAgg0 = va0 != 0xffffffffu && (va0 & 1u), isAgg1 = va1 != 0xffffffffu && (va1 & 1u);
            OSide s[2][2];   // [side][frame]
            const uint32_t actor = !isAgg1 ? h1 : h0;
            for (uint32_t side = 0; side < 2; ++side)
            {
                const bool isAgg = side ? isAgg1 : isAgg0;
                const uint64_t a = aggs + uint64_t(((side ? va1 : va0) >> 1)) * AGG_SIZE;
                for (uint32_t f = 0; f < 2; ++f)
                {
                    OSide& o = s[side][f];
                    o.shared = !isAgg;
                    if (isAgg)
                    {
                        o.proj = m.u64(a + (f ? A_SP1 : A_SP0));
                        o.handles = m.u64(a + (f ? A_SH1 : A_SH0));
                        o.masks = m.u64(a + (f ? A_SM1 : A_SM0));
                        o.bi = m.u64(a + (f ? A_BI1 : A_BI0));
                        o.nbObj = m.u32(a + (f ? A_PREVSIZE : A_SIZE));
                    }
                    else
                    {
                        const uint64_t ib = f ? oldIB : newIB;
                        o.sproj[0] = m.u32(ib + uint64_t(actor) * IAABB_SIZE);
                        o.sproj[1] = m.u32(ib + uint64_t(actor) * IAABB_SIZE + 12);
                        o.sbi[0] = o.sbi[1] = actor;
                        o.nbObj = 1;
                    }
                }
            }
            oBoxPruning(m, newIB, oldIB, s[0][0], s[1][0], groups, found, 1);
            oBoxPruning(m, newIB, oldIB, s[1][0], s[0][0], groups, found, 0);
            if (!isNew)
            {
                oBoxPruning(m, oldIB, newIB, s[0][1], s[1][1], groups, lost, 1);
                oBoxPruning(m, oldIB, newIB, s[1][1], s[0][1], groups, lost, 0);
            }
            m.u8(p + 8) = 0;
        }
        m.u32(bitmap + i * 4ull) = shouldRemove ? 1 : 0;
    }
}

// the pool stages (:1133-1425)
void oCopyReports(Mem& m, uint64_t desc)
{
    const uint32_t nbC = m.u32(desc + D_SHFOUND), nbL = m.u32(desc + D_SHLOST), mx = m.u32(desc + D_MAXFL);
    const bool fo = nbC > mx, lo = nbL > mx;
    m.u8(desc + D_FLAGS) = (m.u8(desc + D_FLAGS) != 0 || fo) ? 1 : 0;
    m.u8(desc + D_FLAGS + 1) = lo ? 1 : 0;
    const uint32_t tc = (fo ? mx : nbC) * 2, tl = (lo ? mx : nbL) * 2;
    for (uint32_t i = 0; i < tc; ++i) m.u32(m.u64(desc + D_FOUNDMAP) + i * 4ull) = m.u32(m.u64(desc + D_FOUND) + i * 4ull);
    for (uint32_t i = 0; i < tl; ++i) m.u32(m.u64(desc + D_LOSTMAP) + i * 4ull) = m.u32(m.u64(desc + D_LOST) + i * 4ull);
}

void oAdd(Mem& m, uint64_t desc, uint64_t bpDesc)
{
    const uint32_t nbFound = m.u32(bpDesc + BP_SHFOUNDAGG);
    const uint32_t safe = std::min(nbFound, m.u32(desc + D_MAXFL));
    const uint64_t fl = m.u64(desc + D_FREELIST);
    const uint32_t numFree = m.u32(fl), maxIndex = m.u32(fl + 4), maxAgg = m.u32(desc + D_MAXAGG);
    const uint64_t pairs = m.u64(desc + D_AGGPAIRS), report = m.u64(bpDesc + BP_FOUNDAGGREPORT);
    for (uint32_t i = 0; i < safe; ++i)
    {
        const uint32_t index = i < numFree ? m.u32(m.u64(desc + D_FREEIDX) + (numFree - i - 1) * 4ull) : maxIndex + (i - numFree);
        if (index < maxAgg)
        {
            m.u32(pairs + index * uint64_t(PAIR_SIZE)) = m.u32(report + i * 8ull);
            m.u32(pairs + index * uint64_t(PAIR_SIZE) + 4) = m.u32(report + i * 8ull + 4);
            m.u32(pairs + index * uint64_t(PAIR_SIZE) + 8) = 1u;
        }
    }
    // stage 2
    uint32_t newFree = 0, inc = 0;
    if (numFree >= safe) newFree = numFree - safe; else inc = safe - numFree;
    m.u32(fl) = newFree;
    uint32_t mi = m.u32(fl + 4) + inc;
    m.u32(desc + D_AGGOVF) = mi;
    if (mi > maxAgg) { m.u8(desc + D_FLAGS + 2) = 1; mi = maxAgg; }
    m.u32(fl + 4) = mi;
    m.u32(m.u64(desc + D_AGGPAIRCOUNT)) = mi;
    if (nbFound > m.u32(desc + D_MAXFL)) { m.u8(desc + D_FLAGS) = 1; m.u32(desc + D_CANDOVF) = nbFound; }
}

void oRemove(Mem& m, uint64_t desc)
{
    const uint32_t n = m.u32(m.u64(desc + D_AGGPAIRCOUNT));
    const uint64_t bitmap = m.u64(desc + D_REMBITMAP), histo = m.u64(desc + D_REMHISTO), fl = m.u64(desc + D_FREELIST);
    const uint32_t perWave = 32 * 32, nbWarps = (n + 31) / 32, nbIter = (nbWarps + perWave - 1) / perWave;
    // stage 1
    for (uint32_t it = 0; it < nbIter; ++it)
        for (uint32_t b = 0; b < 32; ++b)
        {
            uint32_t c = 0;
            for (uint32_t w = 0; w < 32; ++w)
                for (uint32_t l = 0; l < 32; ++l)
                {
                    const uint32_t th = (it * perWave + b * 32 + w) * 32 + l;
                    if (th < n && m.u32(bitmap + th * 4ull)) ++c;
                }
            m.u32(histo + (b + it * 32) * 4ull) = c;
        }
    // stage 2: slots in bitmap order within each wave (block, warp, lane order == index order)
    const uint32_t numFree = m.u32(fl);
    uint32_t accum = 0;
    for (uint32_t it = 0; it < nbIter; ++it)
    {
        uint32_t waveCount = 0;
        for (uint32_t b = 0; b < 32; ++b) waveCount += m.u32(histo + (b + it * 32) * 4ull);
        uint32_t k = 0;
        for (uint32_t th = it * perWave * 32; th < (it + 1) * perWave * 32; ++th)
            if (th < n && m.u32(bitmap + th * 4ull))
            {
                m.u8(m.u64(desc + D_AGGPAIRS) + th * uint64_t(PAIR_SIZE) + 9) = 1;
                m.u32(m.u64(desc + D_FREEIDX) + (numFree + accum + k) * 4ull) = th;
                ++k;
            }
        accum += waveCount;
    }
    m.u32(desc + D_NBREMOVED) = accum;
    // stage 3
    m.u32(fl) = m.u32(fl) + accum;
}

// ===========================================================================
// device launches
// ===========================================================================
bool L(PxgCudaShim* shim, uint32_t id, uint32_t grid, const void* blob, uint32_t size)
{
    return shim->launchKernel(id, grid, 1, 1, blob, size) == PxgResult::Success;
}
struct Two { uint64_t a, b; };
struct One { uint64_t a; };

std::vector<Pair2> readReport(Mem& m, uint64_t addr, uint32_t n)
{
    std::vector<uint32_t> w = m.downloadU32(addr, size_t(n) * 2);
    std::vector<Pair2> v(n);
    for (uint32_t i = 0; i < n; ++i) v[i] = { w[2 * i], w[2 * i + 1] };
    return v;
}
std::vector<Pair2> hostReport(Mem& m, uint64_t addr, uint32_t n)
{
    std::vector<Pair2> v(n);
    for (uint32_t i = 0; i < n; ++i) v[i] = { m.u32(addr + i * 8ull), m.u32(addr + i * 8ull + 4) };
    return v;
}
// the report's first `n` entries as a multiset, the rest of the region word by word
size_t compareReport(Mem& m, uint64_t addr, uint32_t n, const char* what)
{
    std::vector<Pair2> g = readReport(m, addr, n), r = hostReport(m, addr, n);
    std::sort(g.begin(), g.end());
    std::sort(r.begin(), r.end());
    if (!(g == r)) { std::printf("  MM %s: the %u reported pairs differ as a multiset\n", what, n); return 1; }
    // copy the device order into the host image so the word compare sees only the rest
    for (uint32_t i = 0; i < n; ++i)
    {
        std::vector<uint32_t> w = m.downloadU32(addr + i * 8ull, 2);
        m.u32(addr + i * 8ull) = w[0];
        m.u32(addr + i * 8ull + 4) = w[1];
    }
    return 0;
}

// ---------------------------------------------------------------------------
// the shared scenario builders
// ---------------------------------------------------------------------------
struct World
{
    Mem m;
    uint64_t bpDesc = 0, desc = 0, aggs = 0, newIB = 0, oldIB = 0, groups = 0, vd = 0, removedMap = 0;
    uint64_t found = 0, lost = 0, foundMap = 0, lostMap = 0, freeList = 0, freeIdx = 0, remBitmap = 0, remHisto = 0,
             aggPairs = 0, aggPairCount = 0, foundAggReport = 0;
    uint32_t nbAgg = 0, nbBounds = 0;
};

// an integer bound from a float box, encoded like PxgIntegerAABB::encode (monotone map)
uint32_t encodeMin(float f) { uint32_t u; std::memcpy(&u, &f, 4); u = (u & 0x80000000u) ? ~u : (u | 0x80000000u); return (u + 1) & ~1u; }
uint32_t encodeMax(float f) { uint32_t u; std::memcpy(&u, &f, 4); u = (u & 0x80000000u) ? ~u : (u | 0x80000000u); return u | 1u; }
void setIB(Mem& m, uint64_t base, uint32_t i, float x0, float y0, float z0, float x1, float y1, float z1)
{
    const uint64_t a = base + uint64_t(i) * IAABB_SIZE;
    m.u32(a) = encodeMin(x0); m.u32(a + 4) = encodeMin(y0); m.u32(a + 8) = encodeMin(z0);
    m.u32(a + 12) = encodeMax(x1); m.u32(a + 16) = encodeMax(y1); m.u32(a + 20) = encodeMax(z1);
}

void buildWorld(World& w, PxgCudaShim* shim, uint32_t nbBounds, uint32_t nbAgg, uint32_t maxFL, uint32_t maxAgg)
{
    w.m.shim = shim;
    w.nbAgg = nbAgg;
    w.nbBounds = nbBounds;
    w.bpDesc = w.m.alloc(BP_SIZE, 0xEE);
    w.desc = w.m.alloc(DESC_SIZE, 0xEE);
    w.aggs = w.m.alloc(size_t(std::max(nbAgg, 1u)) * AGG_SIZE);
    w.newIB = w.m.alloc(size_t(nbBounds) * IAABB_SIZE);
    w.oldIB = w.m.alloc(size_t(nbBounds) * IAABB_SIZE);
    w.groups = w.m.alloc(size_t(nbBounds) * 4);
    w.vd = w.m.alloc(size_t(nbBounds) * VD_SIZE);
    w.removedMap = w.m.alloc(size_t((nbBounds + 31) / 32) * 4 + 4, 0);
    w.found = w.m.alloc(size_t(maxFL) * 8 + 64);
    w.lost = w.m.alloc(size_t(maxFL) * 8 + 64);
    w.foundMap = w.m.alloc(size_t(maxFL) * 8 + 64);
    w.lostMap = w.m.alloc(size_t(maxFL) * 8 + 64);
    w.freeList = w.m.alloc(8, 0);
    w.freeIdx = w.m.alloc(size_t(maxAgg) * 4 + 64);
    w.remBitmap = w.m.alloc(size_t(maxAgg) * 4 + 64);
    w.remHisto = w.m.alloc(4096 * 4);
    w.aggPairs = w.m.alloc(size_t(maxAgg) * PAIR_SIZE + 64);
    w.aggPairCount = w.m.alloc(4, 0);
    w.foundAggReport = w.m.alloc(size_t(maxFL) * 8 + 64);
    Mem& m = w.m;
    m.u64(w.bpDesc + BP_REMOVEDMAP) = w.removedMap;
    m.u64(w.bpDesc + BP_NEWIB) = w.newIB;
    m.u64(w.bpDesc + BP_OLDIB) = w.oldIB;
    m.u64(w.bpDesc + BP_GROUPS) = w.groups;
    m.u64(w.bpDesc + BP_VOLUMEDATA) = w.vd;
    m.u64(w.bpDesc + BP_FOUNDAGGREPORT) = w.foundAggReport;
    m.u32(w.bpDesc + BP_SHFOUNDAGG) = 0;
    m.u64(w.desc + D_FOUND) = w.found; m.u64(w.desc + D_LOST) = w.lost;
    m.u64(w.desc + D_FOUNDMAP) = w.foundMap; m.u64(w.desc + D_LOSTMAP) = w.lostMap;
    m.u32(w.desc + D_SHFOUND) = 0; m.u32(w.desc + D_SHLOST) = 0;
    m.u64(w.desc + D_AGGS) = w.aggs; m.u32(w.desc + D_NUMAGG) = nbAgg;
    m.u64(w.desc + D_FREELIST) = w.freeList; m.u64(w.desc + D_FREEIDX) = w.freeIdx;
    m.u64(w.desc + D_REMBITMAP) = w.remBitmap; m.u64(w.desc + D_REMHISTO) = w.remHisto;
    m.u64(w.desc + D_AGGPAIRS) = w.aggPairs; m.u64(w.desc + D_AGGPAIRCOUNT) = w.aggPairCount;
    m.u32(w.desc + D_MAXFL) = maxFL; m.u32(w.desc + D_MAXAGG) = maxAgg;
    m.u32(w.desc + D_FLAGS) = 0;   // the three bools, and the pad byte
    for (uint32_t i = 0; i < nbBounds; ++i)
    {
        m.u64(w.vd + uint64_t(i) * VD_SIZE) = 0x1000ull * (i + 1);
        m.u32(w.vd + uint64_t(i) * VD_SIZE + 8) = 0xffffffffu;   // single actor
    }
}

// an aggregate record whose arrays all live in fresh (poisoned) regions; the sort arrays
// have room for the radix temp half (2 * 4 * numUint4 words)
void makeAggregate(World& w, uint32_t g, const std::vector<uint32_t>& members, uint32_t mIndex, bool self, bool isNew,
                   uint32_t capacity = 0)
{
    Mem& m = w.m;
    const uint64_t a = w.aggs + uint64_t(g) * AGG_SIZE;
    const uint32_t cap = std::max<uint32_t>(std::max<uint32_t>(uint32_t(members.size()), capacity), 1u);
    const uint32_t projWords = ((cap * 2 + 3) / 4) * 8 + 8;
    const uint64_t upd = m.alloc(cap * 4);
    for (uint32_t i = 0; i < members.size(); ++i) m.u32(upd + i * 4ull) = members[i];
    m.u64(a + A_UPD) = upd;
    const uint32_t offs[6] = { A_BI0, A_BI1, A_CMP0, A_CMP1, 0, 0 };
    for (uint32_t k = 0; k < 4; ++k) m.u64(a + offs[k]) = m.alloc(cap * 4 + 16);
    m.u64(a + A_SP0) = m.alloc(projWords * 4); m.u64(a + A_SP1) = m.alloc(projWords * 4);
    m.u64(a + A_SH0) = m.alloc(projWords * 4); m.u64(a + A_SH1) = m.alloc(projWords * 4);
    m.u64(a + A_SAP0) = m.alloc(cap * 8 + 16); m.u64(a + A_SAP1) = m.alloc(cap * 8 + 16);
    m.u64(a + A_SM0) = m.alloc(((cap * 2 + 31) / 32) * 4 + 16, 0);
    m.u64(a + A_SM1) = m.alloc(((cap * 2 + 31) / 32) * 4 + 16, 0);
    m.u32(a + A_PREVCMP) = 0;
    m.u32(a + A_PREVSIZE) = uint32_t(members.size());
    m.u32(a + A_MINDEX) = mIndex;
    m.u32(a + A_SIZE) = uint32_t(members.size());
    m.u32(a + A_FILTER) = (4u << 1) | (self ? 1u : 0u);
    m.u32(a + A_ENV) = 0xffffffffu;
    m.u32(a + A_ISNEW) = isNew ? 0x77777701u : 0x77777700u;   // the pad bytes must survive
    m.u32(a + A_ISNEW + 4) = 0x66666666u;
    for (uint32_t id : members) m.u32(w.vd + uint64_t(id) * VD_SIZE + 8) = g << 1;       // aggregated
    m.u32(w.vd + uint64_t(mIndex) * VD_SIZE + 8) = (g << 1) | 1u;                        // the aggregate's bound
}

// ---------------------------------------------------------------------------
PXG_TEST(aggregateBda_dirtyBounds)
{
    PxgSlangComputeDevice* dev = testDevice();
    PxgCudaShim* shim = makeShim(dev);
    std::mt19937 rng(0x9291);
    World w;
    const uint32_t NB = 600, NAGG = 40;
    buildWorld(w, shim, NB, NAGG, 256, 256);
    Mem& m = w.m;
    if (!w.bpDesc) { std::printf("  NOTE: no BDA -> skip\n"); delete shim; return; }

    // the persistent records: half new, half existing, slot 7 empty (mIndex invalid)
    uint32_t next = 0;
    std::vector<std::vector<uint32_t>> members(NAGG);
    for (uint32_t g = 0; g < NAGG; ++g)
    {
        const uint32_t sz = (g % 9 == 0) ? 45 : 1 + rng() % 12;
        for (uint32_t k = 0; k < sz && next < NB - NAGG; ++k) members[g].push_back(next++);
        makeAggregate(w, g, members[g], NB - 1 - g, (g & 1) != 0, (g % 3) == 0);
        // a stale prevSize: updateDirtyAggregate must replace it with the old size
        m.u32(w.aggs + uint64_t(g) * AGG_SIZE + A_PREVSIZE) = uint32_t(members[g].size()) + 5;
    }
    m.u32(w.aggs + 7ull * AGG_SIZE + A_MINDEX) = 0xffffffffu;

    // the dirty staging records: 21 of the 40, in a scrambled order, with new sizes
    std::vector<uint32_t> dirty(NAGG);
    std::iota(dirty.begin(), dirty.end(), 0u);
    std::shuffle(dirty.begin(), dirty.end(), rng);
    dirty.resize(21);
    const uint64_t dIdx = m.alloc(dirty.size() * 4), dAggs = m.alloc(dirty.size() * AGG_SIZE),
                   dBounds = m.alloc(4096 * 4), dStart = m.alloc(dirty.size() * 4);
    uint32_t pos = 0;
    for (uint32_t k = 0; k < dirty.size(); ++k)
    {
        const uint32_t g = dirty[k];
        m.u32(dIdx + k * 4ull) = g;
        const uint64_t src = w.aggs + uint64_t(g) * AGG_SIZE, dst = dAggs + uint64_t(k) * AGG_SIZE;
        for (uint32_t b = 0; b < AGG_SIZE; b += 4) m.u32(dst + b) = m.u32(src + b);
        // the new member list: some records shrink, one moves its own bound (mIndex)
        const uint32_t sz = uint32_t(members[g].size()) - ((k % 3 == 1 && members[g].size() > 3) ? 2u : 0u);
        m.u64(dst + A_UPD) = m.alloc(std::max(sz, 1u) * 4);   // a fresh bound-index list
        m.u32(dst + A_SIZE) = sz;
        m.u32(dst + A_MINDEX) = (k % 4 == 2) ? NB - NAGG - 1 - g : m.u32(src + A_MINDEX);
        m.u32(dst + A_FILTER) = m.u32(src + A_FILTER) ^ 1u;
        m.u32(dst + A_PREVCMP) = 0x12340000u + k;              // copied only when the record is new
        m.u32(dst + A_ISNEW) = 0x55555501u;
        m.u32(dst + A_ISNEW + 4) = 0x44444444u;
        m.u32(dStart + k * 4ull) = pos;
        for (uint32_t i = 0; i < sz; ++i) m.u32(dBounds + (pos + i) * 4ull) = members[g][(i * 7) % members[g].size()];
        pos += sz;
    }

    // float bounds for updateAggregateBounds: members overlap, some coordinates are +-0 and
    // a few are NaN, so the select folds' tie and NaN rules decide bits
    const uint64_t fb = m.alloc(NB * 24), cd = m.alloc(NB * 4);
    std::uniform_real_distribution<float> uf(-10.f, 10.f);
    for (uint32_t i = 0; i < NB; ++i)
    {
        for (uint32_t c = 0; c < 3; ++c)
        {
            float lo = uf(rng), hi = lo + std::fabs(uf(rng));
            if (rng() % 7 == 0) lo = (rng() & 1) ? -0.0f : 0.0f;
            if (rng() % 7 == 0) hi = (rng() & 1) ? -0.0f : 0.0f;
            if (rng() % 97 == 0) lo = std::nanf("");
            m.f32(fb + i * 24ull + c * 4) = lo;
            m.f32(fb + i * 24ull + 12 + c * 4) = hi;
        }
        m.f32(cd + i * 4ull) = (rng() % 5 == 0) ? 0.0f : std::fabs(uf(rng)) * 0.01f;
    }
    // directed members of the first two non-dirty aggregates: aggregate A's maxima are all
    // +-0 or negative with a -0 inflation (the max fold decides +0 against -0 across lanes,
    // lane 0 holding -0) and one member's max is NaN; aggregate B's members all have min +inf
    // and max -inf, so the empty bound's constant itself is what gets stored
    {
        std::vector<uint32_t> clean;
        for (uint32_t g = 0; g < NAGG && clean.size() < 2; ++g)
            if (g != 7 && members[g].size() >= 3 && std::find(dirty.begin(), dirty.end(), g) == dirty.end())
                clean.push_back(g);
        PXG_CHECK_MSG(clean.size() == 2, "no clean aggregates for the directed float members");
        const std::vector<uint32_t>& ma = members[clean[0]];
        for (uint32_t k = 0; k < ma.size(); ++k)
        {
            const uint32_t b = ma[k];
            m.f32(cd + b * 4ull) = -0.0f;   // max + (-0) keeps a -0 max (+0 would turn it into +0)
            for (uint32_t c = 0; c < 3; ++c)
            {
                m.f32(fb + b * 24ull + c * 4) = -5.0f;
                m.f32(fb + b * 24ull + 12 + c * 4) = (k == 0) ? -0.0f : ((k % 2) ? 0.0f : -1.0f);
            }
        }
        m.f32(fb + ma[ma.size() - 1] * 24ull + 12 + 4) = std::nanf("");
        for (uint32_t b0 : members[clean[1]])
            for (uint32_t c = 0; c < 3; ++c)
            {
                m.f32(fb + b0 * 24ull + c * 4) = INFINITY;
                m.f32(fb + b0 * 24ull + 12 + c * 4) = -INFINITY;
            }
    }
    const uint64_t changed = m.alloc(((NB + 31) / 32) * 4);
    for (uint32_t i = 0; i < (NB + 31) / 32; ++i) m.u32(changed + i * 4ull) = rng();

    const uint32_t nbDel = 37, nbAdd = 150;
    const uint64_t delB = m.alloc(nbDel * 4), addB = m.alloc(nbAdd * 4);
    for (uint32_t i = 0; i < nbDel; ++i) m.u32(delB + i * 4ull) = rng() % NB;
    for (uint32_t i = 0; i < nbAdd; ++i) m.u32(addB + i * 4ull) = rng() % NB;
    for (uint32_t i = 0; i < NB; ++i) { setIB(m, w.newIB, i, 0, 0, 0, 1, 1, 1); setIB(m, w.oldIB, i, 2, 2, 2, 3, 3, 3); }

    m.upload();
    struct UD { uint64_t aggs; uint32_t numAggs, pad0; uint64_t dirtyIdx, dirtyAggs; uint32_t numDirty, pad1; uint64_t dirtyBounds, dirtyStart; }
        ud = { w.aggs, NAGG, 0, dIdx, dAggs, uint32_t(dirty.size()), 0, dBounds, dStart };
    struct CD { uint64_t aggs; uint32_t numAggs, pad0; uint64_t dirtyIdx; uint32_t numDirty, pad1; }
        cdp = { w.aggs, NAGG, 0, dIdx, uint32_t(dirty.size()), 0 };
    Two mark = { w.desc, changed };
    struct Three { uint64_t a, b, c; } ub = { w.desc, fb, cd };
    struct MAD { uint64_t bp, desc, del; uint32_t nbDel, pad0; uint64_t add; uint32_t nbAdd, pad1; }
        mad = { w.bpDesc, w.desc, delB, nbDel, 0, addB, nbAdd, 0 };
    bool ok = true;
    ok &= L(shim, K_UPD_DIRTY, (uint32_t(dirty.size()) + 15) / 16, &ud, sizeof(ud));
    shim->streamSynchronize();
    // ---- oracle: updateDirtyAggregate, compared on its own (clearDirty rewrites prevSize) ----
    for (uint32_t k = 0; k < dirty.size(); ++k)
    {
        const uint64_t a = w.aggs + uint64_t(dirty[k]) * AGG_SIZE, d = dAggs + uint64_t(k) * AGG_SIZE;
        const uint32_t sz = m.u32(d + A_SIZE), st = m.u32(dStart + k * 4ull);
        for (uint32_t i = 0; i < sz; ++i) m.u32(m.u64(d + A_UPD) + i * 4ull) = m.u32(dBounds + (st + i) * 4ull);
        if (m.u8(a + A_ISNEW))
            for (uint32_t b = 0; b < AGG_SIZE; b += 4) m.u32(a + b) = m.u32(d + b);
        else
        {
            m.u32(a + A_PREVSIZE) = m.u32(a + A_SIZE);
            m.u64(a + A_UPD) = m.u64(d + A_UPD);
            m.u32(a + A_MINDEX) = m.u32(d + A_MINDEX);
            m.u32(a + A_SIZE) = m.u32(d + A_SIZE);
            m.u32(a + A_FILTER) = m.u32(d + A_FILTER);
        }
    }
    size_t badDirty = m.compare({}, "updateDirtyAggregate");
    ok &= L(shim, K_MARK_BITMAP, (NAGG + 7) / 8, &mark, sizeof(mark));
    ok &= L(shim, K_UPD_BOUNDS, (NAGG + 15) / 16, &ub, sizeof(ub));
    ok &= L(shim, K_CLEAR_DIRTY, (uint32_t(dirty.size()) + 15) / 16, &cdp, sizeof(cdp));
    // the host's grid: (max + (127) / 128) -- a precedence slip that launches max blocks
    ok &= L(shim, K_MARK_ADD_DEL, std::max(nbDel, nbAdd), &mad, sizeof(mad));
    shim->streamSynchronize();
    PXG_CHECK_MSG(ok, "a launch was refused");

    // ---- oracle, the rest ----
    uint32_t validAggs = 0;
    for (uint32_t g = 0; g < NAGG; ++g)   // markAggregateBoundsUpdatedBitmap
    {
        const uint64_t a = w.aggs + uint64_t(g) * AGG_SIZE;
        if (!aggValid(m, a)) continue;
        ++validAggs;
        const uint32_t mi = m.u32(a + A_MINDEX);
        m.u32(changed + (mi / 32) * 4ull) |= 1u << (mi % 32);
        for (uint32_t i = 0; i < m.u32(a + A_SIZE); ++i)
        {
            const uint32_t idx = m.u32(m.u64(a + A_UPD) + i * 4ull);
            m.u32(changed + (idx / 32) * 4ull) &= ~(1u << (idx % 32));
        }
    }
    const float ext = bitsF(0x7e7fffffu);
    for (uint32_t g = 0; g < NAGG; ++g)   // updateAggregateBounds
    {
        const uint64_t a = w.aggs + uint64_t(g) * AGG_SIZE;
        if (!aggValid(m, a)) continue;
        float mn[32][3], mx[32][3];
        for (uint32_t l = 0; l < 32; ++l)
            for (uint32_t c = 0; c < 3; ++c) { mn[l][c] = ext; mx[l][c] = -ext; }
        for (uint32_t i = 0; i < m.u32(a + A_SIZE); ++i)
        {
            const uint32_t idx = m.u32(m.u64(a + A_UPD) + i * 4ull), l = i % 32;
            const float inflate = m.f32(cd + idx * 4ull);
            for (uint32_t c = 0; c < 3; ++c)
            {
                mn[l][c] = selMin(mn[l][c], m.f32(fb + idx * 24ull + c * 4) - inflate);
                mx[l][c] = selMax(mx[l][c], m.f32(fb + idx * 24ull + 12 + c * 4) + inflate);
            }
        }
        for (uint32_t r = 1; r < 32; r *= 2)
        {
            float pmn[32][3], pmx[32][3];
            std::memcpy(pmn, mn, sizeof(mn));
            std::memcpy(pmx, mx, sizeof(mx));
            for (uint32_t l = 0; l < 32; ++l)
                for (uint32_t c = 0; c < 3; ++c)
                {
                    mn[l][c] = selMin(pmn[l ^ r][c], pmn[l][c]);
                    mx[l][c] = selMax(pmx[l ^ r][c], pmx[l][c]);
                }
        }
        const uint32_t mi = m.u32(a + A_MINDEX);
        for (uint32_t c = 0; c < 3; ++c) { m.f32(fb + mi * 24ull + c * 4) = mn[0][c]; m.f32(fb + mi * 24ull + 12 + c * 4) = mx[0][c]; }
    }
    for (uint32_t k = 0; k < dirty.size(); ++k)   // clearDirtyAggregates
    {
        const uint64_t a = w.aggs + uint64_t(dirty[k]) * AGG_SIZE;
        m.u8(a + A_ISNEW) = 0;
        m.u32(a + A_PREVSIZE) = m.u32(a + A_SIZE);
    }
    auto empty = [&](uint64_t base, uint32_t h) {
        for (uint32_t c = 0; c < 3; ++c) { m.u32(base + h * 24ull + c * 4) = 0xff7fffffu; m.u32(base + h * 24ull + 12 + c * 4) = 0x00800000u; } };
    for (uint32_t i = 0; i < nbDel; ++i) { empty(w.newIB, m.u32(delB + i * 4ull)); empty(w.oldIB, m.u32(delB + i * 4ull)); }
    for (uint32_t i = 0; i < nbAdd; ++i) empty(w.oldIB, m.u32(addB + i * 4ull));

    std::printf("[ info ] aggregateBda dirty/bounds: %u aggregates (%u valid), %zu dirty, %u bounds\n",
                NAGG, validAggs, dirty.size(), NB);
    const size_t bad = badDirty + m.compare({}, "dirty/bounds");
    PXG_CHECK_MSG(bad == 0, "%zu words differ from the CUDA transcription", bad);
    delete shim;
}

// ---------------------------------------------------------------------------
// two frames of sort + self-collision
void runSortSelf(World& w, PxgCudaShim* shim, bool& ok)
{
    Two ts = { w.bpDesc, w.desc };
    ok &= L(shim, K_SORT, (w.nbAgg + 1) / 2, &ts, sizeof(ts));
    ok &= L(shim, K_SELF, (w.nbAgg + 15) / 16, &ts, sizeof(ts));
    shim->streamSynchronize();
    oSort(w.m, w.bpDesc, w.desc);
    oSelf(w.m, w.bpDesc, w.desc);
}

PXG_TEST(aggregateBda_sortSelf)
{
    PxgSlangComputeDevice* dev = testDevice();
    PxgCudaShim* shim = makeShim(dev);
    std::mt19937 rng(0x9292);
    World w;
    const uint32_t NB = 1400, NAGG = 34;
    buildWorld(w, shim, NB, NAGG, 20000, 64);
    Mem& m = w.m;
    if (!w.bpDesc) { std::printf("  NOTE: no BDA -> skip\n"); delete shim; return; }

    uint32_t next = 0;
    for (uint32_t g = 0; g < NAGG; ++g)
    {
        const uint32_t sz = (g == 3) ? 70 : (g == 4 ? 33 : 1 + (g * 5) % 23);
        std::vector<uint32_t> mem;
        for (uint32_t k = 0; k < sz; ++k) mem.push_back(next++);
        std::shuffle(mem.begin(), mem.end(), rng);
        makeAggregate(w, g, mem, NB - 1 - g, g != 5, true, sz + 8);
        m.u32(w.aggs + uint64_t(g) * AGG_SIZE + A_ISNEW) = 0x77777701u;
    }
    m.u32(w.aggs + 9ull * AGG_SIZE + A_SIZE) = 0;   // an empty (invalid) slot
    // dense, quantized bounds: members overlap often, and equal projection values occur,
    // which is what the radix sort's stability decides
    std::uniform_int_distribution<int> q(0, 12);
    auto frame = [&](uint64_t ib) {
        for (uint32_t i = 0; i < NB; ++i)
        {
            const float x = float(q(rng)) * 0.25f, y = float(q(rng)) * 0.25f, z = float(q(rng)) * 0.25f;
            setIB(m, ib, i, x, y, z, x + 1.0f + float(q(rng) % 4) * 0.25f, y + 1.0f, z + 1.0f);
        }
    };
    frame(w.newIB);
    for (uint32_t i = 0; i < NB; ++i) { setIB(m, w.oldIB, i, 100, 100, 100, 99, 99, 99); }   // empty-ish old
    for (uint32_t i = 0; i < NB; ++i) m.u32(w.groups + i * 4ull) = (rng() % 13 == 0) ? 7u : i + 1;
    m.upload();

    bool ok = true;
    runSortSelf(w, shim, ok);
    size_t bad = compareReport(m, w.found, std::min(m.u32(w.desc + D_SHFOUND), 20000u), "frame 1 found");
    bad += m.compare({}, "frame 1");
    const uint32_t found1 = m.u32(w.desc + D_SHFOUND);

    // frame 2: not new any more, bounds move, old = the previous new; the counters restart
    for (uint32_t g = 0; g < NAGG; ++g)
        m.u32(w.aggs + uint64_t(g) * AGG_SIZE + A_ISNEW) = 0x77777700u;
    // aggregate 3 (70 members) loses 9, aggregate 6 gains 6 spare bounds: what
    // updateDirtyAggregate leaves -- prevSize = the old size, size = the new one, a new list
    {
        const uint64_t a3 = w.aggs + 3ull * AGG_SIZE, a6 = w.aggs + 6ull * AGG_SIZE;
        m.u32(a3 + A_PREVSIZE) = m.u32(a3 + A_SIZE);
        m.u32(a3 + A_SIZE) = m.u32(a3 + A_SIZE) - 9;
        m.u32(a6 + A_PREVSIZE) = m.u32(a6 + A_SIZE);
        const uint64_t upd6 = m.u64(a6 + A_UPD);
        for (uint32_t k = 0; k < 6; ++k) m.u32(upd6 + (m.u32(a6 + A_SIZE) + k) * 4ull) = next + k;
        m.u32(a6 + A_SIZE) = m.u32(a6 + A_SIZE) + 6;
    }
    for (uint32_t i = 0; i < NB * IAABB_SIZE / 4; ++i) m.u32(w.oldIB + i * 4ull) = m.u32(w.newIB + i * 4ull);
    frame(w.newIB);
    m.u32(w.desc + D_SHFOUND) = 0;
    m.u32(w.desc + D_SHLOST) = 0;
    m.upload();
    runSortSelf(w, shim, ok);
    bad += compareReport(m, w.found, std::min(m.u32(w.desc + D_SHFOUND), 20000u), "frame 2 found");
    bad += compareReport(m, w.lost, std::min(m.u32(w.desc + D_SHLOST), 20000u), "frame 2 lost");
    bad += m.compare({}, "frame 2");
    std::printf("[ info ] aggregateBda sort/self: frame 1 found %u, frame 2 found %u lost %u\n", found1,
                m.u32(w.desc + D_SHFOUND), m.u32(w.desc + D_SHLOST));
    PXG_CHECK_MSG(ok, "a launch was refused");
    PXG_CHECK_MSG(found1 > 100 && m.u32(w.desc + D_SHLOST) > 50, "too few pairs to measure");
    PXG_CHECK_MSG(bad == 0, "%zu words differ from the CUDA transcription", bad);

    // overflow: one aggregate, one warp, so the report order is fixed and compared word by word
    World o;
    buildWorld(o, shim, 80, 1, 17, 64);
    std::vector<uint32_t> mem(60);
    std::iota(mem.begin(), mem.end(), 0u);
    makeAggregate(o, 0, mem, 79, true, true);
    for (uint32_t i = 0; i < 80; ++i) { setIB(o.m, o.newIB, i, 0, 0, 0, 1, 1, 1); setIB(o.m, o.oldIB, i, 5, 5, 5, 4, 4, 4); }
    for (uint32_t i = 0; i < 80; ++i) o.m.u32(o.groups + i * 4ull) = i + 1;
    o.m.upload();
    bool ok2 = true;
    runSortSelf(o, shim, ok2);
    const size_t bad2 = o.m.compare({}, "overflow");
    std::printf("[ info ] aggregateBda self overflow: %u found for max 17\n", o.m.u32(o.desc + D_SHFOUND));
    PXG_CHECK_MSG(ok2 && o.m.u32(o.desc + D_SHFOUND) > 17, "the overflow case does not overflow");
    PXG_CHECK_MSG(bad2 == 0, "overflow: %zu words differ", bad2);
    delete shim;
}

// ---------------------------------------------------------------------------
PXG_TEST(aggregateBda_pairs)
{
    PxgSlangComputeDevice* dev = testDevice();
    PxgCudaShim* shim = makeShim(dev);
    std::mt19937 rng(0x9293);
    World w;
    // MAXFL well above the pair counts: past it, WHICH pairs land in the report depends on the
    // order the warps reach the counter (unordered in the CUDA too); overflow is tested apart
    const uint32_t NB = 900, NAGG = 12, MAXAGG = 40000, MAXFL = 60000;
    buildWorld(w, shim, NB, NAGG, MAXFL, MAXAGG);
    Mem& m = w.m;
    if (!w.bpDesc) { std::printf("  NOTE: no BDA -> skip\n"); delete shim; return; }

    // twelve aggregates in two clusters, and 300 single actors among them
    uint32_t next = 0;
    for (uint32_t g = 0; g < NAGG; ++g)
    {
        const uint32_t sz = 5 + (g * 7) % 30;
        std::vector<uint32_t> mem;
        for (uint32_t k = 0; k < sz; ++k) mem.push_back(next++);
        makeAggregate(w, g, mem, 600 + g, true, false);
    }
    std::uniform_int_distribution<int> q(0, 30);
    auto frame = [&](uint64_t ib) {
        for (uint32_t i = 0; i < NB; ++i)
        {
            const float x = float(q(rng)) * 0.25f, y = float(q(rng) % 8) * 0.25f, z = float(q(rng) % 8) * 0.25f;
            setIB(m, ib, i, x, y, z, x + 1.5f, y + 1.0f, z + 1.0f);
        }
        // the aggregates' own bounds cover everything
        for (uint32_t g = 0; g < NAGG; ++g) setIB(m, ib, 600 + g, -1, -1, -1, 20, 20, 20);
    };
    frame(w.oldIB);
    frame(w.newIB);
    // one in eleven bounds shares a group (7): a pair of those is filtered out
    for (uint32_t i = 0; i < NB; ++i) m.u32(w.groups + i * 4ull) = (i % 11 == 3) ? 7u : i + 1;
    m.upload();
    // two frames of structures (sort twice), so both [0] and [1] exist; between them
    // aggregate 2 loses three members, so its previous frame has prevSize = the old size
    bool ok = true;
    Two ts = { w.bpDesc, w.desc };
    ok &= L(shim, K_SORT, (NAGG + 1) / 2, &ts, sizeof(ts));
    shim->streamSynchronize();
    oSort(m, w.bpDesc, w.desc);
    size_t badSort = m.compare({}, "pairs: first sort");
    {
        const uint64_t a2 = w.aggs + 2ull * AGG_SIZE;
        m.u32(a2 + A_PREVSIZE) = m.u32(a2 + A_SIZE);
        m.u32(a2 + A_SIZE) = m.u32(a2 + A_SIZE) - 3;
    }
    m.upload();
    ok &= L(shim, K_SORT, (NAGG + 1) / 2, &ts, sizeof(ts));
    shim->streamSynchronize();
    oSort(m, w.bpDesc, w.desc);

    // the pool: 20000 slots in use, 150 free (scattered), plus 12 found pairs from the BP
    const uint32_t INUSE = 20000;
    std::vector<uint32_t> freeSlots;
    for (uint32_t i = 0; i < INUSE; ++i)
    {
        const uint64_t p = w.aggPairs + uint64_t(i) * PAIR_SIZE;
        const bool dead = (rng() % 133 == 0);
        if (dead && freeSlots.size() < 150) freeSlots.push_back(i);
        const uint32_t ga = rng() % NAGG, gb = (ga + 1 + rng() % (NAGG - 1)) % NAGG;
        const uint32_t h0 = 600 + ga;
        // agg-agg, or agg-actor with the actor's handle below (side 0) or above (side 1)
        const uint32_t h1 = (rng() % 3 == 0) ? (600 + gb) : ((rng() & 1) ? 300 + rng() % 100 : 700 + rng() % 150);
        m.u32(p) = std::min(h0, h1) == h0 ? h0 : h1;
        m.u32(p + 4) = std::min(h0, h1) == h0 ? h1 : h0;
        // live pairs are mostly not new, a few new; slots beyond 16384 exercise the grid stride
        m.u32(p + 8) = (dead ? 0x0100u : 0u) | ((rng() % 9 == 0) ? 1u : 0u) | 0xBEEF0000u;
        if (!dead && i > 64 && rng() % 23 != 0) m.u32(p + 8) = 0x0100u | 0xBEEF0000u;   // keep the work bounded
    }
    // a removed handle
    m.u32(w.removedMap + (777 / 32) * 4ull) |= 1u << (777 % 32);
    m.u32(w.aggPairs + 5ull * PAIR_SIZE) = 600; m.u32(w.aggPairs + 5ull * PAIR_SIZE + 4) = 777;
    m.u32(w.aggPairs + 5ull * PAIR_SIZE + 8) = 0xBEEF0000u;
    // a pair whose second handle is an AGGREGATED member (VolumeData low bit clear): it is
    // not an aggregate, so it takes the one-box path
    m.u32(w.aggPairs + 6ull * PAIR_SIZE) = 601; m.u32(w.aggPairs + 6ull * PAIR_SIZE + 4) = 3;
    m.u32(w.aggPairs + 6ull * PAIR_SIZE + 8) = 0xBEEF0000u;
    m.u32(w.freeList) = uint32_t(freeSlots.size());
    m.u32(w.freeList + 4) = INUSE;
    for (uint32_t i = 0; i < freeSlots.size(); ++i) m.u32(w.freeIdx + i * 4ull) = freeSlots[i];
    const uint32_t NFOUNDAGG = 100;   // fewer than the 150 free slots: the free list shrinks
    m.u32(w.bpDesc + BP_SHFOUNDAGG) = NFOUNDAGG;
    for (uint32_t i = 0; i < NFOUNDAGG; ++i)
    {
        m.u32(w.foundAggReport + i * 8ull) = 600 + (i % NAGG);
        m.u32(w.foundAggReport + i * 8ull + 4) = 700 + (i * 3) % 150;
    }
    m.u32(w.aggPairCount) = INUSE;
    m.upload();

    Two ab = { w.desc, w.bpDesc };
    One od = { w.desc };
    ok &= L(shim, K_ADD1, 64, &ab, sizeof(ab));
    ok &= L(shim, K_ADD2, 64, &ab, sizeof(ab));
    ok &= L(shim, K_PAIR, 8192, &ts, sizeof(ts));
    shim->streamSynchronize();
    oAdd(m, w.desc, w.bpDesc);
    oPairs(m, w.bpDesc, w.desc);
    size_t bad = badSort + compareReport(m, w.found, std::min(m.u32(w.desc + D_SHFOUND), MAXFL), "pairs found");
    bad += compareReport(m, w.lost, std::min(m.u32(w.desc + D_SHLOST), MAXFL), "pairs lost");
    bad += m.compare({}, "add + pair collisions");
    const uint32_t f = m.u32(w.desc + D_SHFOUND), l = m.u32(w.desc + D_SHLOST);

    // the removal waves: 40000 pairs, so two waves of 32768
    const uint32_t N2 = MAXAGG;
    for (uint32_t i = m.u32(w.aggPairCount); i < N2; ++i) m.u32(w.remBitmap + i * 4ull) = (rng() % 3 == 0) ? 1u : 0u;
    m.u32(w.aggPairCount) = N2;
    m.upload();
    ok &= L(shim, K_REM1, 32, &od, sizeof(od));
    ok &= L(shim, K_REM2, 32, &od, sizeof(od));
    ok &= L(shim, K_REM3, 1, &od, sizeof(od));
    ok &= L(shim, K_COPY, 64, &od, sizeof(od));
    shim->streamSynchronize();
    oRemove(m, w.desc);
    oCopyReports(m, w.desc);
    bad += m.compare({}, "remove + copy");

    // every overflow path: too many found agg pairs, the pool past max, the reports past max
    World o;
    buildWorld(o, shim, 64, 1, 4, 10);
    makeAggregate(o, 0, { 1, 2, 3 }, 63, true, true);
    o.m.u32(o.freeList) = 2; o.m.u32(o.freeList + 4) = 9;   // 9 + 2 > 10: the pool overflows
    o.m.u32(o.freeIdx) = 1; o.m.u32(o.freeIdx + 4) = 4;
    o.m.u32(o.bpDesc + BP_SHFOUNDAGG) = 9;
    for (uint32_t i = 0; i < 9; ++i) { o.m.u32(o.foundAggReport + i * 8ull) = 63; o.m.u32(o.foundAggReport + i * 8ull + 4) = 10 + i; }
    o.m.u32(o.desc + D_SHFOUND) = 3; o.m.u32(o.desc + D_SHLOST) = 5;   // found flag comes from stage 2 alone
    o.m.u32(o.desc + D_FLAGS) = 0x00000000u;
    for (uint32_t i = 0; i < 8; ++i) { o.m.u32(o.found + i * 4ull) = 0x100 + i; o.m.u32(o.lost + i * 4ull) = 0x200 + i; }
    o.m.upload();
    Two oab = { o.desc, o.bpDesc };
    One ood = { o.desc };
    bool ok2 = true;
    ok2 &= L(shim, K_ADD1, 64, &oab, sizeof(oab));
    ok2 &= L(shim, K_ADD2, 64, &oab, sizeof(oab));
    ok2 &= L(shim, K_COPY, 64, &ood, sizeof(ood));
    shim->streamSynchronize();
    oAdd(o.m, o.desc, o.bpDesc);
    oCopyReports(o.m, o.desc);
    const uint32_t flags = o.m.u32(o.desc + D_FLAGS);
    bad += o.m.compare({}, "overflow flags");

    std::printf("[ info ] aggregateBda pairs: found %u lost %u; removal over %u pairs, %u removed; overflow flags %06x\n",
                f, l, N2, m.u32(w.desc + D_NBREMOVED), flags & 0xffffffu);
    PXG_CHECK_MSG(ok && ok2, "a launch was refused");
    PXG_CHECK_MSG(f > 20 && l > 20, "too few found/lost pairs (%u/%u) to measure", f, l);
    PXG_CHECK_MSG((flags & 0xffffffu) == 0x010101u, "not every overflow flag is raised (%06x)", flags & 0xffffffu);
    PXG_CHECK_MSG(bad == 0, "%zu words differ from the CUDA transcription", bad);
    delete shim;
}

} // namespace
