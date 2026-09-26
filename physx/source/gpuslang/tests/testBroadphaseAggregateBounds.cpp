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

// Verification of broadphaseAggregateBounds.slang against C++ transcriptions of
// the two aggregate bounds/bitmap refresh kernels from aggregate.cu:
// markAggregateBoundsUpdatedBitmap (atomic OR the aggregate's own handle bit +
// atomic AND ~bit for each member shape) and updateAggregateBounds (union of
// contact-distance-inflated member bounds written back at the aggregate slot).
// The per-aggregate pointer lists are flattened into a shared pool indexed by a
// base offset; results are compared bit-exactly.

#include "pxgTestFramework.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

const uint32_t INVALID = 0xffffffffu;

struct AggBoundsParams
{
    uint32_t numAggregates;
};

// PxBounds3::setEmpty() sentinel: PX_MAX_BOUNDS_EXTENTS = FLT_MAX * 0.25f.
const float MAX_BOUNDS_EXTENTS = std::numeric_limits<float>::max() * 0.25f;

// Build a synthetic aggregate set: nbShapes member-shape bound slots [0,nbShapes)
// and nbAgg aggregate slots [nbShapes, nbShapes+nbAgg) (disjoint).  Each aggregate
// gets a distinct mIndex in its own slot, a member count, and a contiguous run in
// the shared bound-index pool.  ~5% are made invalid (half size 0, half mIndex
// INVALID) to exercise the isValid() guard.
struct AggSet
{
    uint32_t nbShapes, nbAgg, nbSlots;
    std::vector<uint32_t> mIndex, size, boundBase, boundIndices;
};

AggSet genAggSet(std::mt19937& rng, uint32_t nbAgg, uint32_t nbShapes)
{
    AggSet s;
    s.nbShapes = nbShapes;
    s.nbAgg = nbAgg;
    s.nbSlots = nbShapes + nbAgg;
    s.mIndex.resize(nbAgg);
    s.size.resize(nbAgg);
    s.boundBase.resize(nbAgg);
    uint32_t base = 0;
    for (uint32_t a = 0; a < nbAgg; ++a)
    {
        uint32_t sz = 1 + rng() % 40; // up to 40 members -> exercises >32 strided loop + reduction
        uint32_t mi = nbShapes + a;   // distinct, disjoint from member shape slots
        const uint32_t roll = rng() % 100;
        if (roll < 3) sz = 0;             // invalid via size 0
        else if (roll < 5) mi = INVALID; // invalid via mIndex
        s.size[a] = sz;
        s.mIndex[a] = mi;
        s.boundBase[a] = base;
        const uint32_t emit = (sz == 0) ? (1 + rng() % 8) : sz; // still reserve a run for size-0 aggs
        for (uint32_t i = 0; i < emit; ++i)
            s.boundIndices.push_back(rng() % nbShapes);
        base += emit;
    }
    return s;
}

bool valid(uint32_t size, uint32_t mIndex) { return size > 0 && mIndex != INVALID; }

// ---------------------------------------------------------------------------
// CUDA-exact mirror of updateAggregateBounds (aggregate.cu:193-225), P9.86c.
//
// Per lane: aggBounds.setEmpty(), then for id = lane, lane+32, ... the member bound is
// folded with PxVec3::minimum/maximum = PxMin/PxMax<float> = intrinsics::selectMin(acc,v)
// = acc < v ? acc : v and selectMax(acc,v) = acc > v ? acc : v (PxVec3.h:339-358,
// PxMath.h:82-100, PxUnixMathIntrinsics.h:116-124): the NEW value survives a +-0 tie.
// Then a warp butterfly for r = 1,2,4,8,16: v = PxMin(__shfl_xor_sync(v, r), v) --
// the lane keeps its OWN value on a tie -- and lane 0 stores (aggregate.cu:209-225).
//
// The old mirror was a serial std::min/std::max fold, which keeps the ACCUMULATOR on a
// tie and ignores the lane/butterfly order: neither CUDA rule.
//
// gAggVariant selects mutant twins of the shader's two P9.86 fixes, so the directed
// test below can prove that each of its cases actually separates the CUDA from the
// pre-fix spellings (llvmpipe's stamped FMin/FMax order the zeros in either operand
// order -- min -> -0, max -> +0 -- and return the other operand on NaN; P9.86 F4):
//   0 = CUDA;
//   1 = per-lane fold with FMin/FMax (revert of the fold fix), CUDA butterfly;
//   2 = pre-fix lane+s tree (s = 16..1) with FMin/FMax (revert of the tree fix);
//   3 = lane+s tree with keep-own selects (topology-only half revert);
//   4 = adjacent tree with FMin/FMax (operator-only half revert).
// ---------------------------------------------------------------------------
int gAggVariant = 0;
struct AggCov
{
    uint32_t foldMinNegThenPos;  // lane fold: acc = -0 meets v = +0 on the min side
    uint32_t foldMaxPosThenNeg;  // lane fold: acc = +0 meets v = -0 on the max side
    uint32_t treeMinTie;         // butterfly at lane 0: mixed +-0 pair on the min side
    uint32_t treeMaxTie;         // butterfly at lane 0: mixed +-0 pair on the max side
};
AggCov gAggCov = {};

inline float pxMinF(float a, float b) { return a < b ? a : b; } // intrinsics::selectMin
inline float pxMaxF(float a, float b) { return a > b ? a : b; } // intrinsics::selectMax
inline bool mixedZeros(float a, float b)
{
    return a == 0.0f && b == 0.0f && std::signbit(a) != std::signbit(b);
}
// llvmpipe GLSL.std.450 FMin/FMax under SignedZeroInfNanPreserve (P9.86 F4).
inline float lpFMin(float a, float b)
{
    if (std::isnan(a)) return b;
    if (std::isnan(b)) return a;
    if (a == b) return std::signbit(a) ? a : b;
    return a < b ? a : b;
}
inline float lpFMax(float a, float b)
{
    if (std::isnan(a)) return b;
    if (std::isnan(b)) return a;
    if (a == b) return std::signbit(a) ? b : a;
    return a > b ? a : b;
}

void aggBoundsRef(const uint32_t* inds, uint32_t size, const std::vector<float>& bounds,
                  const std::vector<float>& cd, float out[6])
{
    const bool count = (gAggVariant == 0);
    float v[32][6];
    for (uint32_t l = 0; l < 32; ++l)
    {
        for (uint32_t c = 0; c < 3; ++c) { v[l][c] = MAX_BOUNDS_EXTENTS; v[l][3 + c] = -MAX_BOUNDS_EXTENTS; }
        for (uint32_t id = l; id < size; id += 32)
        {
            const uint32_t idx = inds[id];
            const float infl = cd[idx];
            for (uint32_t c = 0; c < 3; ++c)
            {
                const float vn = bounds[idx * 6 + c] - infl;
                const float vx = bounds[idx * 6 + 3 + c] + infl;
                if (count && mixedZeros(v[l][c], vn) && std::signbit(v[l][c])) ++gAggCov.foldMinNegThenPos;
                if (count && mixedZeros(v[l][3 + c], vx) && !std::signbit(v[l][3 + c])) ++gAggCov.foldMaxPosThenNeg;
                v[l][c] = (gAggVariant == 1) ? lpFMin(v[l][c], vn) : pxMinF(v[l][c], vn);
                v[l][3 + c] = (gAggVariant == 1) ? lpFMax(v[l][3 + c], vx) : pxMaxF(v[l][3 + c], vx);
            }
        }
    }

    if (gAggVariant == 0 || gAggVariant == 1)
    {
        // aggregate.cu:209-216, every lane of the xor butterfly.
        for (uint32_t r = 1; r < 32; r <<= 1)
        {
            float t[32][6];
            for (uint32_t l = 0; l < 32; ++l)
            {
                const uint32_t q = l ^ r;
                for (uint32_t c = 0; c < 3; ++c)
                {
                    if (count && l == 0 && mixedZeros(v[q][c], v[l][c])) ++gAggCov.treeMinTie;
                    if (count && l == 0 && mixedZeros(v[q][3 + c], v[l][3 + c])) ++gAggCov.treeMaxTie;
                    t[l][c] = pxMinF(v[q][c], v[l][c]);
                    t[l][3 + c] = pxMaxF(v[q][3 + c], v[l][3 + c]);
                }
            }
            std::memcpy(v, t, sizeof(v));
        }
    }
    else if (gAggVariant == 2 || gAggVariant == 3)
    {
        for (uint32_t s = 16; s > 0; s >>= 1)
            for (uint32_t l = 0; l < s; ++l)
                for (uint32_t c = 0; c < 3; ++c)
                {
                    if (gAggVariant == 2)
                    {
                        v[l][c] = lpFMin(v[l][c], v[l + s][c]);
                        v[l][3 + c] = lpFMax(v[l][3 + c], v[l + s][3 + c]);
                    }
                    else
                    {
                        v[l][c] = pxMinF(v[l + s][c], v[l][c]);
                        v[l][3 + c] = pxMaxF(v[l + s][3 + c], v[l][3 + c]);
                    }
                }
    }
    else
    {
        for (uint32_t r = 1; r < 32; r <<= 1)
            for (uint32_t l = 0; l < 32; l += 2 * r)
                for (uint32_t c = 0; c < 3; ++c)
                {
                    v[l][c] = lpFMin(v[l][c], v[l + r][c]);
                    v[l][3 + c] = lpFMax(v[l][3 + c], v[l + r][3 + c]);
                }
    }
    for (uint32_t k = 0; k < 6; ++k) out[k] = v[0][k];
}

inline uint32_t fbits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

PXG_TEST(broadphaseAggregateBounds_bitmap)
{
    std::mt19937 rng(24010);
    const uint32_t nbAgg = 500, nbShapes = 20000;
    AggSet s = genAggSet(rng, nbAgg, nbShapes);
    const uint32_t words = (s.nbSlots + 31) / 32;

    std::vector<uint32_t> changed(words);
    for (auto& w : changed) w = rng();

    // reference (order-independent: OR-set and AND-clear sets are disjoint)
    std::vector<uint32_t> rChanged = changed;
    for (uint32_t a = 0; a < nbAgg; ++a)
    {
        if (!valid(s.size[a], s.mIndex[a])) continue;
        const uint32_t mi = s.mIndex[a];
        rChanged[mi / 32] |= (1u << (mi % 32));
        for (uint32_t i = 0; i < s.size[a]; ++i)
        {
            const uint32_t idx = s.boundIndices[s.boundBase[a] + i];
            rChanged[idx / 32] &= ~(1u << (idx % 32));
        }
    }

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<float> fdummy(16, 0.f);
    PxgBufferHandle bMI = createBufferFrom(dev, s.mIndex);
    PxgBufferHandle bSz = createBufferFrom(dev, s.size);
    PxgBufferHandle bBase = createBufferFrom(dev, s.boundBase);
    PxgBufferHandle bIdx = createBufferFrom(dev, s.boundIndices);
    PxgBufferHandle bChg = createBufferFrom(dev, changed);
    PxgBufferHandle bFDummy = createBufferFrom(dev, fdummy);

    AggBoundsParams p = { nbAgg };
    const std::vector<uint32_t> strides(7, 4);
    Kernel k("broadphaseAggregateBounds.spv", "markAggregateBoundsUpdatedBitmap", strides, sizeof(AggBoundsParams));
    k.dispatch(nbAgg, { bMI, bSz, bBase, bIdx, bChg, bFDummy, bFDummy }, p);
    dev->flush();

    const auto gChanged = downloadBuffer<uint32_t>(dev, bChg, words);
    size_t mism = 0;
    for (uint32_t w = 0; w < words; ++w) if (gChanged[w] != rChanged[w]) ++mism;
    PXG_CHECK_MSG(mism == 0, "%zu bitmap-word mismatches (words=%u)", mism, words);
    for (PxgBufferHandle h : { bMI, bSz, bBase, bIdx, bChg, bFDummy }) dev->destroyBuffer(h);
}

PXG_TEST(broadphaseAggregateBounds_bounds)
{
    std::mt19937 rng(24020);
    const uint32_t nbAgg = 500, nbShapes = 20000;
    AggSet s = genAggSet(rng, nbAgg, nbShapes);

    // flat PxBounds3 pool: shape slots [0,nbShapes) carry real bounds; aggregate
    // slots [nbShapes,nbSlots) start as a sentinel and are overwritten by valid aggs.
    std::vector<float> bounds(s.nbSlots * 6, 0.f);
    std::vector<float> contactDist(s.nbSlots, 0.f);
    std::uniform_real_distribution<float> pos(-500.f, 500.f), ext(0.01f, 50.f), cd(0.f, 2.f);
    for (uint32_t i = 0; i < nbShapes; ++i)
    {
        float mnx = pos(rng), mny = pos(rng), mnz = pos(rng);
        bounds[i * 6 + 0] = mnx; bounds[i * 6 + 1] = mny; bounds[i * 6 + 2] = mnz;
        bounds[i * 6 + 3] = mnx + ext(rng); bounds[i * 6 + 4] = mny + ext(rng); bounds[i * 6 + 5] = mnz + ext(rng);
        contactDist[i] = cd(rng);
    }
    for (uint32_t i = nbShapes; i < s.nbSlots; ++i) // sentinel garbage in the aggregate slots
    {
        for (uint32_t c = 0; c < 6; ++c) bounds[i * 6 + c] = 12345.678f + float(i) + float(c);
    }

    // reference: the CUDA-exact lane fold + xor butterfly (aggregate.cu:193-225).
    std::vector<float> rBounds = bounds;
    for (uint32_t a = 0; a < nbAgg; ++a)
    {
        if (!valid(s.size[a], s.mIndex[a])) continue;
        float out[6];
        aggBoundsRef(&s.boundIndices[s.boundBase[a]], s.size[a], bounds, contactDist, out);
        const uint32_t o = s.mIndex[a] * 6;
        for (uint32_t k = 0; k < 6; ++k) rBounds[o + k] = out[k];
    }

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<uint32_t> udummy(16, 0);
    PxgBufferHandle bMI = createBufferFrom(dev, s.mIndex);
    PxgBufferHandle bSz = createBufferFrom(dev, s.size);
    PxgBufferHandle bBase = createBufferFrom(dev, s.boundBase);
    PxgBufferHandle bIdx = createBufferFrom(dev, s.boundIndices);
    PxgBufferHandle bUDummy = createBufferFrom(dev, udummy);
    PxgBufferHandle bBounds = createBufferFrom(dev, bounds);
    PxgBufferHandle bCD = createBufferFrom(dev, contactDist);

    AggBoundsParams p = { nbAgg };
    const std::vector<uint32_t> strides(7, 4);
    Kernel k("broadphaseAggregateBounds.spv", "updateAggregateBounds", strides, sizeof(AggBoundsParams));
    k.dispatch(nbAgg, { bMI, bSz, bBase, bIdx, bUDummy, bBounds, bCD }, p);
    dev->flush();

    const auto gBounds = downloadBuffer<float>(dev, bBounds, s.nbSlots * 6);
    size_t mism = 0; float maxDiff = 0.f;
    for (uint32_t i = 0; i < s.nbSlots * 6; ++i)
    {
        if (std::memcmp(&gBounds[i], &rBounds[i], sizeof(float)) != 0)
        {
            ++mism;
            maxDiff = std::max(maxDiff, std::fabs(gBounds[i] - rBounds[i]));
        }
    }
    PXG_CHECK_MSG(mism == 0, "%zu aggregate-bounds mismatches (maxAbsDiff=%g)", mism, maxDiff);
    for (PxgBufferHandle h : { bMI, bSz, bBase, bIdx, bUDummy, bBounds, bCD }) dev->destroyBuffer(h);
}

// P9.86c: the sign of a zero aggregate bound.  The random test above never produces an
// exact +-0 member value, so neither P9.86 fix of updateAggregateBounds (the per-lane
// PxMin/PxMax select fold, and the CUDA butterfly tree with keep-own selects) was
// observable there.  Every member value below is exact; the zeros come from
// (-0) - (+0) = -0, 0.5 - 0.5 = +0, (-0.5) + 0.5 = +0 and (-0) + (-0) = -0, i.e. member
// bounds with a -0 coordinate (a flat shape at a -0 plane) and contactOffset +0 or -0
// (both pass NpShape.cpp:595-596 with a negative restOffset).
//
//   A = [Zmn, P x31, Zmp]  lane 0 folds min -0 then +0: PxMin(-0,+0) = +0 (FMin: -0)
//   B = [ZMp, N x31, ZMn]  lane 0 folds max +0 then -0: PxMax(+0,-0) = -0 (FMax: +0)
//   C = [Zmp, Zmn]         lanes 0/1 min +0/-0: butterfly keeps lane 0's +0
//   D = [ZMn, ZMp]         lanes 0/1 max -0/+0: butterfly keeps lane 0's -0
//   E = [P, Zmn, Zmp]      min 0.75/-0/+0: adjacent tree -0 (lane+s tree with selects: +0)
//   F = [Zmp, P, Zmn]      min +0/0.75/-0: adjacent tree +0 (lane+s tree with FMin: -0)
PXG_TEST(broadphaseAggregateBounds_boundsZeroSign)
{
    struct Kind { float mn, mx, cd; };
    const Kind P = { 1.0f, 2.0f, 0.25f };      // vn  0.75, vx  2.25
    const Kind N = { -2.0f, -1.0f, 0.25f };    // vn -2.25, vx -0.75
    const Kind Zmn = { -0.0f, 1.0f, 0.0f };    // vn -0,    vx  1
    const Kind Zmp = { 0.5f, 1.0f, 0.5f };     // vn +0,    vx  1.5
    const Kind ZMp = { -1.0f, -0.5f, 0.5f };   // vn -1.5,  vx +0
    const Kind ZMn = { -1.0f, -0.0f, -0.0f };  // vn -1,    vx -0

    std::vector<std::vector<Kind>> aggs(6);
    aggs[0].push_back(Zmn); for (int i = 0; i < 31; ++i) aggs[0].push_back(P); aggs[0].push_back(Zmp);
    aggs[1].push_back(ZMp); for (int i = 0; i < 31; ++i) aggs[1].push_back(N); aggs[1].push_back(ZMn);
    aggs[2] = { Zmp, Zmn };
    aggs[3] = { ZMn, ZMp };
    aggs[4] = { P, Zmn, Zmp };
    aggs[5] = { Zmp, P, Zmn };

    const uint32_t nbAgg = uint32_t(aggs.size());
    AggSet s;
    s.nbAgg = nbAgg;
    std::vector<Kind> members;
    for (uint32_t a = 0; a < nbAgg; ++a)
    {
        s.boundBase.push_back(uint32_t(members.size()));
        s.size.push_back(uint32_t(aggs[a].size()));
        for (const Kind& k : aggs[a])
        {
            s.boundIndices.push_back(uint32_t(members.size()));
            members.push_back(k);
        }
    }
    s.nbShapes = uint32_t(members.size());   // 76 member slots, then 6 aggregate slots
    s.nbSlots = s.nbShapes + nbAgg;
    for (uint32_t a = 0; a < nbAgg; ++a) s.mIndex.push_back(s.nbShapes + a);

    std::vector<float> bounds(s.nbSlots * 6, 0.f);
    std::vector<float> contactDist(s.nbSlots, 0.f);
    for (uint32_t i = 0; i < s.nbShapes; ++i)
    {
        for (uint32_t c = 0; c < 3; ++c) { bounds[i * 6 + c] = members[i].mn; bounds[i * 6 + 3 + c] = members[i].mx; }
        contactDist[i] = members[i].cd;
    }
    for (uint32_t i = s.nbShapes; i < s.nbSlots; ++i)
        for (uint32_t c = 0; c < 6; ++c) bounds[i * 6 + c] = 12345.678f + float(i) + float(c);

    auto reference = [&](int variant) {
        std::vector<float> r = bounds;
        gAggVariant = variant;
        for (uint32_t a = 0; a < nbAgg; ++a)
        {
            float out[6];
            aggBoundsRef(&s.boundIndices[s.boundBase[a]], s.size[a], bounds, contactDist, out);
            for (uint32_t k = 0; k < 6; ++k) r[s.mIndex[a] * 6 + k] = out[k];
        }
        gAggVariant = 0;
        return r;
    };
    gAggCov = {};
    const std::vector<float> want = reference(0);

    // coverage: the inputs must keep producing the ties the fixes are about
    PXG_CHECK_MSG(gAggCov.foldMinNegThenPos == 3 && gAggCov.foldMaxPosThenNeg == 3,
                  "lane-fold ties: min -0->+0 %u (want 3), max +0->-0 %u (want 3)",
                  gAggCov.foldMinNegThenPos, gAggCov.foldMaxPosThenNeg);
    PXG_CHECK_MSG(gAggCov.treeMinTie == 9 && gAggCov.treeMaxTie == 3,
                  "butterfly lane-0 ties: min %u (want 9), max %u (want 3)",
                  gAggCov.treeMinTie, gAggCov.treeMaxTie);

    // the CUDA answers, spelled out (x = y = z)
    struct Expect { uint32_t agg, comp; uint32_t bits; };
    const Expect ex[] = {
        { 0, 0, 0x00000000u }, { 1, 3, 0x80000000u }, { 2, 0, 0x00000000u },
        { 3, 3, 0x80000000u }, { 4, 0, 0x80000000u }, { 5, 0, 0x00000000u },
    };
    for (const Expect& e : ex)
        for (uint32_t c = 0; c < 3; ++c)
        {
            const uint32_t b = fbits(want[s.mIndex[e.agg] * 6 + e.comp + c]);
            PXG_CHECK_MSG(b == e.bits, "mirror: aggregate %u component %u bits %08x, CUDA %08x",
                          e.agg, e.comp + c, b, e.bits);
        }

    // each mutant twin must change the image in exactly the floats it is built to change
    const size_t wantDiff[5] = { 0, 6, 9, 3, 9 };
    for (int variant = 1; variant <= 4; ++variant)
    {
        const std::vector<float> m = reference(variant);
        size_t diff = 0;
        for (size_t i = 0; i < m.size(); ++i) if (fbits(m[i]) != fbits(want[i])) ++diff;
        PXG_CHECK_MSG(diff == wantDiff[variant], "mutant twin %d changes %zu floats (want %zu)",
                      variant, diff, wantDiff[variant]);
    }

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<uint32_t> udummy(16, 0);
    PxgBufferHandle bMI = createBufferFrom(dev, s.mIndex);
    PxgBufferHandle bSz = createBufferFrom(dev, s.size);
    PxgBufferHandle bBase = createBufferFrom(dev, s.boundBase);
    PxgBufferHandle bIdx = createBufferFrom(dev, s.boundIndices);
    PxgBufferHandle bUDummy = createBufferFrom(dev, udummy);
    PxgBufferHandle bBounds = createBufferFrom(dev, bounds);
    PxgBufferHandle bCD = createBufferFrom(dev, contactDist);

    AggBoundsParams p = { nbAgg };
    const std::vector<uint32_t> strides(7, 4);
    Kernel k("broadphaseAggregateBounds.spv", "updateAggregateBounds", strides, sizeof(AggBoundsParams));
    k.dispatch(nbAgg, { bMI, bSz, bBase, bIdx, bUDummy, bBounds, bCD }, p);
    dev->flush();

    const auto got = downloadBuffer<float>(dev, bBounds, s.nbSlots * 6);
    size_t mism = 0;
    for (uint32_t i = 0; i < s.nbSlots * 6; ++i)
    {
        if (fbits(got[i]) != fbits(want[i]))
        {
            if (mism < 8)
                PXG_CHECK_MSG(false, "slot %u comp %u: bits %08x, CUDA %08x", i / 6, i % 6,
                              fbits(got[i]), fbits(want[i]));
            ++mism;
        }
    }
    PXG_CHECK_MSG(mism == 0, "%zu aggregate-bounds bit mismatches", mism);
    for (PxgBufferHandle h : { bMI, bSz, bBase, bIdx, bUDummy, bBounds, bCD }) dev->destroyBuffer(h);
}

} // namespace
