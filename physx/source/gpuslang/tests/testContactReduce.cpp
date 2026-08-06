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

// Verification of contactReduce.slang (contactReduceShared +
// contactReduce/contactReduce2 from gpucommon/src/CUDA/contactReduction.cuh).
// The result is a discrete lane mask; the separations are RAW uploaded floats
// (their comparisons are bit-exact on both sides, including the dupe
// tie-break and every deepest-fill pick), so slack tracking is needed only
// for COMPUTED quantities: the 1e-8 dupe distance threshold, the projected
// distance / cross-axis argmax/argmin near-ties (and the reduce2 initial
// criterion) and the cluster-label distance ties.  Cases with any tracked
// comparison within its margin are skipped; all others must match the CUDA
// transcription's mask bit-exactly.  All 16 flag combinations, both
// TMaxPoints values, early-outs, dupe culls, the PCM extra pick and the
// cluster/fill paths are exercised.

#include "pxgTestFramework.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };
struct CrCfg { U4 cfg; F4 normal_bias; };
struct CrParams { uint32_t numTests, pad0, pad1, pad2; };

struct V3 { float x, y, z; };
static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static V3 muls(V3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }

static const float MARG = 1e-4f;      // computed-value near-tie margin
static const float MARG_DUPE = 1e-9f; // around the 1e-8 squared-distance gate
static const float MAXF = 3.4028235e38f;

struct Slack { bool near = false; };

// lowest-index argmax/argmin with near-tie tracking (tracked only when the
// values are computed floats; exact==false disables tracking)
static uint32_t argMax(const float* vals, uint32_t mask, Slack& sl, bool tracked)
{
    float best = -MAXF, second = -MAXF;
    uint32_t idx = 0;
    for (uint32_t j = 0; j < 32; ++j)
        if ((mask >> j) & 1)
        {
            if (vals[j] > best) { second = best; best = vals[j]; idx = j; }
            else if (vals[j] > second) second = vals[j];
        }
    if (tracked && second > -MAXF && best - second < MARG) sl.near = true;
    return idx;
}
static uint32_t argMin(const float* vals, uint32_t mask, Slack& sl, bool tracked)
{
    float best = MAXF, second = MAXF;
    uint32_t idx = 0;
    for (uint32_t j = 0; j < 32; ++j)
        if ((mask >> j) & 1)
        {
            if (vals[j] < best) { second = best; best = vals[j]; idx = j; }
            else if (vals[j] < second) second = vals[j];
        }
    if (tracked && second < MAXF && second - best < MARG) sl.near = true;
    return idx;
}

static uint32_t refReduce(const V3* ptRaw, const float* sep, V3 normal, uint32_t allMask,
                          float clusterBias, bool clustering, bool keepDeepest, int maxPoints,
                          bool dupeTest, bool useReduce2, Slack& sl)
{
    V3 point[32];
    float distAdj[32], initCrit[32];
    for (uint32_t j = 0; j < 32; ++j)
    {
        point[j] = sub(ptRaw[j], muls(normal, dot3(ptRaw[j], normal)));
        if (useReduce2) { distAdj[j] = -sep[j]; initCrit[j] = std::sqrt(dot3(point[j], point[j])); }
        else { distAdj[j] = 0.0f; initCrit[j] = sep[j]; }
    }

    if (dupeTest)
    {
        uint32_t ballot = 0;
        for (uint32_t j = 0; j < 32; ++j)
        {
            if (!((allMask >> j) & 1)) continue;
            bool dupe = false;
            for (uint32_t i = 0; i < 32; ++i)
            {
                if (!((allMask >> i) & 1)) continue;
                const V3 dv = sub(ptRaw[i], ptRaw[j]);
                const float d = dot3(dv, dv);
                if (std::fabs(d - 1e-8f) < MARG_DUPE) sl.near = true;
                if (d < 1e-8f && (sep[i] < sep[j] || (sep[i] == sep[j] && i > j)))
                    dupe = true;
            }
            if (dupe) ballot |= 1u << j;
        }
        allMask &= ~ballot;
    }

    if (__builtin_popcount(allMask) <= maxPoints)
        return allMask;

    float vals[32];
    for (uint32_t j = 0; j < 32; ++j) vals[j] = initCrit[j] + distAdj[j];
    const uint32_t i0 = argMax(vals, allMask, sl, useReduce2); // raw seps when !useReduce2
    uint32_t mask = 1u << i0;

    for (uint32_t j = 0; j < 32; ++j)
    {
        const V3 dv = sub(point[i0], point[j]);
        vals[j] = std::sqrt(dot3(dv, dv)) + distAdj[j];
    }
    const uint32_t i1 = argMax(vals, allMask & ~mask, sl, true);
    mask |= 1u << i1;

    const V3 dir = cross3(normal, sub(point[i1], point[i0]));
    float dval[32];
    for (uint32_t j = 0; j < 32; ++j) dval[j] = dot3(dir, sub(point[j], point[i0]));

    for (uint32_t j = 0; j < 32; ++j) vals[j] = dval[j] + distAdj[j];
    const uint32_t f = argMax(vals, allMask & ~mask, sl, true);
    mask |= 1u << f;
    for (uint32_t j = 0; j < 32; ++j) vals[j] = dval[j] - distAdj[j];
    const uint32_t g = argMin(vals, allMask & ~mask, sl, true);
    mask |= 1u << g;

    if (keepDeepest && __builtin_popcount(mask) == 4 && maxPoints > 4)
    {
        const uint32_t i4 = argMin(sep, allMask & ~mask, sl, false); // raw seps: exact
        mask |= 1u << i4;
    }

    if (clustering)
    {
        float sepB[32];
        for (uint32_t j = 0; j < 32; ++j)
            sepB[j] = sep[j] - (((mask >> j) & 1) ? clusterBias : 0.0f); // same subtract both sides: exact

        int label[32];
        float tbest[32];
        for (uint32_t j = 0; j < 32; ++j) { tbest[j] = MAXF; label[j] = -1; }
        int nbClusters = 0;
        for (uint32_t m = mask; m; ++nbClusters, m &= (m - 1))
        {
            const uint32_t anchor = (uint32_t)__builtin_ctz(m);
            for (uint32_t j = 0; j < 32; ++j)
            {
                const V3 dv = sub(point[j], point[anchor]);
                const float d = dot3(dv, dv);
                if ((allMask >> j) & 1)
                {
                    const float slack = std::fabs(d - tbest[j]);
                    if (tbest[j] < MAXF && slack < MARG) sl.near = true;
                }
                if (d < tbest[j]) { tbest[j] = d; label[j] = nbClusters; }
            }
        }

        mask = 0;
        float dummy;
        (void)dummy;
        for (int i = 0; i < nbClusters; ++i)
        {
            uint32_t cluster = 0;
            for (uint32_t j = 0; j < 32; ++j)
                if (label[j] == i) cluster |= 1u << j;
            cluster &= allMask;
            if (cluster)
                mask |= 1u << argMin(sepB, cluster, sl, false); // exact
        }
        for (int i = nbClusters; i < maxPoints; ++i)
            mask |= 1u << argMin(sepB, allMask & ~mask, sl, false); // exact
    }
    else
    {
        const int count = __builtin_popcount(mask);
        for (int i = count; i < maxPoints; ++i)
            mask |= 1u << argMin(sep, allMask & ~mask, sl, false); // exact
    }
    return mask;
}

PXG_TEST(contactReduce_all)
{
    std::mt19937 rng(82019);
    const uint32_t N = 4096;
    std::vector<F4> pts(N * 32);
    std::vector<CrCfg> cfg(N);
    std::vector<uint32_t> refMask(N);
    std::vector<uint8_t> skip(N, 0);
    size_t earlyOut = 0, dupeCulled = 0, pcmFired = 0, clusteringRan = 0, plainRan = 0, skipped = 0;

    std::uniform_real_distribution<float> uv(-1, 1);
    std::uniform_real_distribution<float> us(-0.5f, 0.1f);
    std::normal_distribution<float> gn(0, 1);

    for (uint32_t t = 0; t < N; ++t)
    {
        // random unit normal
        V3 n = { gn(rng), gn(rng), gn(rng) };
        const float nl = 1.0f / std::sqrt(dot3(n, n));
        n = muls(n, nl);

        // 4 cluster centers; points scattered around them
        V3 centers[4];
        for (int k = 0; k < 4; ++k) centers[k] = { uv(rng), uv(rng), uv(rng) };
        V3 p[32];
        float s[32];
        for (uint32_t j = 0; j < 32; ++j)
        {
            const V3 c = centers[rng() % 4];
            p[j] = { c.x + 0.25f * uv(rng), c.y + 0.25f * uv(rng), c.z + 0.25f * uv(rng) };
            s[j] = us(rng);
        }
        // ~1/3 cases seed exact/near duplicates (distinct separations)
        if (t % 3 == 0)
        {
            for (int k = 0; k < 3; ++k)
            {
                const uint32_t a = rng() % 32, b = rng() % 32;
                if (a == b) continue;
                p[b] = p[a];
                if (rng() % 2) { p[b].x += 2e-5f * uv(rng); }
                // half the dupes share the EXACT separation so the
                // sep[i] == sep[j] lane-index tie-break is exercised
                if (rng() % 2) s[b] = s[a];
                else s[b] = s[a] + 0.01f + 0.05f * (rng() % 100) / 100.0f;
            }
        }

        // mask with popcount 2..32 (uniform-ish)
        const uint32_t pop = 2 + rng() % 31;
        uint32_t lanes[32];
        for (uint32_t j = 0; j < 32; ++j) lanes[j] = j;
        for (uint32_t j = 0; j < pop; ++j) std::swap(lanes[j], lanes[j + rng() % (32 - j)]);
        uint32_t allMask = 0;
        for (uint32_t j = 0; j < pop; ++j) allMask |= 1u << lanes[j];

        const uint32_t flags = t % 16;
        const int maxPoints = (t % 32 < 16) ? 6 : 4;
        const float bias = 0.01f;

        for (uint32_t j = 0; j < 32; ++j)
            pts[t * 32 + j] = { p[j].x, p[j].y, p[j].z, s[j] };
        cfg[t].cfg = { allMask, flags, (uint32_t)maxPoints, 0 };
        cfg[t].normal_bias = { n.x, n.y, n.z, bias };

        Slack sl;
        const bool clustering = flags & 1, keepDeepest = flags & 2, dupeTest = flags & 4, useReduce2 = flags & 8;
        refMask[t] = refReduce(p, s, n, allMask, bias, clustering, keepDeepest, maxPoints, dupeTest, useReduce2, sl);
        skip[t] = sl.near ? 1 : 0;

        // coverage bookkeeping (reference-side, deterministic)
        uint32_t cullMask = allMask;
        if (dupeTest)
        {
            for (uint32_t j = 0; j < 32; ++j)
                if (((allMask >> j) & 1))
                    for (uint32_t i = 0; i < 32; ++i)
                        if (((allMask >> i) & 1) && i != j)
                        {
                            const V3 dv = sub(p[i], p[j]);
                            if (dot3(dv, dv) < 1e-8f && (s[i] < s[j] || (s[i] == s[j] && i > j)))
                                cullMask &= ~(1u << j);
                        }
            if (cullMask != allMask) ++dupeCulled;
        }
        if ((int)__builtin_popcount(cullMask) <= maxPoints) ++earlyOut;
        else if (clustering) ++clusteringRan;
        else ++plainRan;
        if (keepDeepest && maxPoints > 4 && (int)__builtin_popcount(cullMask) > maxPoints) ++pcmFired;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<uint32_t> outInit(N, 0xdeadbeefu);
    PxgBufferHandle bP = createBufferFrom(dev, pts);
    PxgBufferHandle bC = createBufferFrom(dev, cfg);
    PxgBufferHandle bO = createBufferFrom(dev, outInit);

    CrParams prm = { N, 0, 0, 0 };
    Kernel k("contactReduce.spv", "ctContactReduce",
             { 16, (uint32_t)sizeof(CrCfg), 4 }, sizeof(CrParams));
    k.dispatch(GROUPS, { bP, bC, bO }, prm);
    dev->flush();

    const auto g = downloadBuffer<uint32_t>(dev, bO, N);
    size_t mm = 0, checked = 0;
    for (uint32_t t = 0; t < N; ++t)
    {
        if (skip[t]) { ++skipped; continue; }
        ++checked;
        if (g[t] != refMask[t]) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "contactReduce: %zu mask mismatches", mm);
    PXG_CHECK_MSG(checked > N * 3 / 4, "contactReduce: too many skipped (%zu)", skipped);
    PXG_CHECK_MSG(earlyOut > 200 && dupeCulled > 300 && pcmFired > 100 && clusteringRan > 800 && plainRan > 800,
                  "contactReduce: coverage thin (early=%zu dupe=%zu pcm=%zu clus=%zu plain=%zu)",
                  earlyOut, dupeCulled, pcmFired, clusteringRan, plainRan);

    dev->destroyBuffer(bP); dev->destroyBuffer(bC); dev->destroyBuffer(bO);
}

// ---------------------------------------------------------------------------
// The 32-LANE VIRTUAL WARP port (pxgSlangContactReduce), at the one
// instantiation the convex-vs-triangle emit tail reaches:
// contactReduce<false, true, 5, true> (convexTriangle.cuh:1789).
//
// It is driven against `refReduce` above -- the SAME oracle the scalar port
// uses -- which is the point: the scalar shader proves the algorithm, so a
// divergence here is attributable to the warp emulation and nothing else.
//
// The random cases come from the same generator, but random data alone cannot
// reach the places a warp version can differ from a scalar one, so the list
// opens with constructed cases that each pin exactly one of them.
// ---------------------------------------------------------------------------

struct CrWarpCase
{
    V3 p[32];
    float s[32];
    V3 normal;
    uint32_t allMask;
    const char* name;
};

struct CrWarpPush { uint64_t input, output; uint32_t numCases, pad; };

PXG_TEST(contactReduceWarp)
{
    PxgSlangComputeDevice* dev = testDevice();

    const int MAXP = 5;
    const bool CLUSTERING = false, KEEPDEEPEST = true, DUPETEST = true, REDUCE2 = false;
    const float BIAS = 0.0f;

    std::vector<CrWarpCase> cases;

    // A spread of points on the z=0 plane, deliberately far from any tie, used
    // as the neutral background for the constructed cases.
    auto spread = [](CrWarpCase& c) {
        for (uint32_t j = 0; j < 32; ++j)
        {
            const float a = 0.19634954f * float(j);          // 2*pi/32
            c.p[j] = { std::cos(a) * (1.0f + 0.03f * float(j)),
                       std::sin(a) * (1.0f + 0.03f * float(j)),
                       0.011f * float(j) };
            c.s[j] = -0.02f - 0.011f * float(j);             // strictly decreasing, distinct
        }
        c.normal = { 0.0f, 0.0f, 1.0f };
        c.allMask = 0xFFFFFFFFu;
    };

    {   // 1: THE EARLY-OUT.  popc(allMask) <= TMaxPoints, so the function returns
        //    allMask before a single point is projected.  Nothing downstream runs.
        CrWarpCase c; spread(c);
        c.allMask = (1u << 3) | (1u << 9) | (1u << 17) | (1u << 30);
        c.name = "early-out, popc 4";
        cases.push_back(c);
    }
    {   // 2: ALL 32 LANES LIVE, so every stage of the butterfly carries a real
        //    value and the ballot spans the whole word.
        CrWarpCase c; spread(c);
        c.name = "all 32 lanes";
        cases.push_back(c);
    }
    {   // 3: AN OUT-OF-MASK LANE HOLDS THE WINNING VALUE.  This is the single case
        //    that separates a correct maxIndex from one that ballots over the set
        //    mask instead of ANDing with it: lane 0 is NOT in allMask but carries
        //    the deepest separation, so a port that forgets the `& mask` picks it
        //    as p0 and every anchor after it moves.
        CrWarpCase c; spread(c);
        c.allMask = 0xFFFFFFFEu;                 // lane 0 excluded
        c.s[0] = c.s[5];                         // ... but equal to an in-set value
        c.name = "out-of-mask lane ties the winner";
        cases.push_back(c);
    }
    {   // 4: EXACT-TIE SEPARATIONS ON TWO LANES.  The lowest-lane rule decides p0,
        //    and nothing else in the algorithm does.
        CrWarpCase c; spread(c);
        c.s[7] = -0.005f;
        c.s[21] = -0.005f;                       // the same deepest value, twice
        c.name = "exact tie on the deepest separation";
        cases.push_back(c);
    }
    {   // 5: COINCIDENT POINTS, EQUAL SEPARATIONS, LOW LANE FIRST.  The dupe rule
        //    is `sep < separation || (sep == separation && i > lane)`, so of two
        //    identical contacts the HIGHER lane is culled.
        CrWarpCase c; spread(c);
        c.p[11] = c.p[4]; c.s[11] = c.s[4];
        c.name = "coincident pair, equal separation";
        cases.push_back(c);
    }
    {   // 6: COINCIDENT POINTS, THE DEEPER ONE ON THE HIGHER LANE.  Now the
        //    separation comparison decides rather than the lane index, and it must
        //    cull the other one -- the opposite lane from case 5.
        CrWarpCase c; spread(c);
        c.p[11] = c.p[4]; c.s[11] = c.s[4] - 0.05f;
        c.name = "coincident pair, deeper on the higher lane";
        cases.push_back(c);
    }
    {   // 7: A DUPE CULL THAT DROPS THE SET BELOW TMaxPoints, so the early-out is
        //    reached through the cull rather than through the input mask.
        CrWarpCase c; spread(c);
        c.allMask = 0u;
        const uint32_t live[7] = { 1, 2, 3, 4, 5, 6, 7 };
        for (uint32_t k = 0; k < 7; ++k) c.allMask |= 1u << live[k];
        c.p[5] = c.p[1]; c.s[5] = c.s[1] - 0.01f;   // 5 culled
        c.p[6] = c.p[2]; c.s[6] = c.s[2] - 0.01f;   // 6 culled
        c.name = "cull drops the set to the early-out";
        cases.push_back(c);
    }
    {   // 8: A FLAT SET.  Every point lies on one line through the origin in the
        //    normal plane, so the cross-axis distance d is 0 for every lane and the
        //    f/g picks are decided purely by the lowest-lane rule.
        CrWarpCase c; spread(c);
        for (uint32_t j = 0; j < 32; ++j)
        {
            c.p[j] = { 0.05f * float(j) - 0.8f, 0.0f, 0.0f };
            c.s[j] = -0.02f - 0.007f * float((j * 13u) % 32u);
        }
        c.name = "collinear points, zero cross-axis";
        cases.push_back(c);
    }

    {   // 9: EVERY POINT PROJECTS TO THE SAME PLACE.  The contacts differ only
        //     ALONG the normal, so `point` is identical on every lane while
        //     `pointRaw` is not -- they are not duplicates and none is culled, but
        //     the distance from p0 is 0 on every lane.  maxIndex then falls back
        //     to the lowest-lane rule, and because p0 is itself lane 0, a p1 pick
        //     that forgets to exclude the anchors already chosen re-picks p0 and
        //     the manifold comes back one contact short.  This is a stack of
        //     contacts along the normal, which is an ordinary thing for a clipper
        //     to produce, and no amount of random data lands on it.
        //     THE SEPARATIONS MUST NOT FOLLOW THE LANE ORDER.  With every distance
        //     zero the anchors fall to the lowest lanes, and the fill loop then
        //     picks the DEEPEST remaining -- so if depth and lane index agree, the
        //     fill silently restores exactly the lane the bad p1 lost and the
        //     mutant survives.  Lane 3 is therefore made shallow and the two
        //     deepest contacts parked out at lanes 10 and 20.
        CrWarpCase c;
        for (uint32_t j = 0; j < 32; ++j)
        {
            c.p[j] = { 0.3f, -0.2f, 0.05f * float(j) };     // same x,y for all
            c.s[j] = -0.1f - 0.001f * float(j);
        }
        c.s[0] = 0.05f;      // the largest separation -> p0 is lane 0
        c.s[3] = 0.0f;       // shallow, so the fill never reaches for it
        c.s[10] = -0.9f;     // the two deepest, far from the low lanes
        c.s[20] = -0.8f;
        c.normal = { 0.0f, 0.0f, 1.0f };
        c.allMask = 0xFFFFFFFFu;
        c.name = "all points project onto one place";
        cases.push_back(c);
    }

    const uint32_t nConstructed = uint32_t(cases.size());

    // Random cases, same generator as the scalar test but pinned to this
    // instantiation.
    std::mt19937 rng(90210);
    std::uniform_real_distribution<float> uv(-1, 1);
    std::uniform_real_distribution<float> us(-0.5f, 0.1f);
    std::normal_distribution<float> gn(0, 1);
    const uint32_t NRAND = 640;
    for (uint32_t t = 0; t < NRAND; ++t)
    {
        CrWarpCase c;
        V3 n = { gn(rng), gn(rng), gn(rng) };
        const float nl = 1.0f / std::sqrt(dot3(n, n));
        c.normal = muls(n, nl);

        V3 centers[4];
        for (int k = 0; k < 4; ++k) centers[k] = { uv(rng), uv(rng), uv(rng) };
        for (uint32_t j = 0; j < 32; ++j)
        {
            const V3 ctr = centers[rng() % 4];
            c.p[j] = { ctr.x + 0.25f * uv(rng), ctr.y + 0.25f * uv(rng), ctr.z + 0.25f * uv(rng) };
            c.s[j] = us(rng);
        }
        if (t % 3 == 0)
            for (int k = 0; k < 3; ++k)
            {
                const uint32_t a = rng() % 32, b = rng() % 32;
                if (a == b) continue;
                c.p[b] = c.p[a];
                if (rng() % 2) c.p[b].x += 2e-5f * uv(rng);
                if (rng() % 2) c.s[b] = c.s[a];
                else c.s[b] = c.s[a] + 0.01f + 0.05f * (rng() % 100) / 100.0f;
            }

        const uint32_t pop = 2 + rng() % 31;
        uint32_t lanes[32];
        for (uint32_t j = 0; j < 32; ++j) lanes[j] = j;
        for (uint32_t j = 0; j < pop; ++j) std::swap(lanes[j], lanes[j + rng() % (32 - j)]);
        c.allMask = 0;
        for (uint32_t j = 0; j < pop; ++j) c.allMask |= 1u << lanes[j];
        c.name = "random";
        cases.push_back(c);
    }

    const uint32_t N = uint32_t(cases.size());

    // ---- reference + coverage ---------------------------------------------
    std::vector<uint32_t> refMask(N);
    std::vector<uint8_t> skip(N, 0);
    size_t earlyOut = 0, dupeCulled = 0, pcmFired = 0, fillRan = 0, skipped = 0;
    for (uint32_t t = 0; t < N; ++t)
    {
        const CrWarpCase& c = cases[t];
        Slack sl;
        refMask[t] = refReduce(c.p, c.s, c.normal, c.allMask, BIAS, CLUSTERING, KEEPDEEPEST,
                               MAXP, DUPETEST, REDUCE2, sl);
        // The constructed cases are constructed to sit ON ties, so the slack
        // tracker must not be allowed to skip them -- they would then assert
        // nothing at all.  Their ties are all in RAW uploaded separations, which
        // compare bit-exactly on both sides.
        skip[t] = (t >= nConstructed && sl.near) ? 1 : 0;

        uint32_t cullMask = c.allMask;
        for (uint32_t j = 0; j < 32; ++j)
            if ((c.allMask >> j) & 1)
                for (uint32_t i = 0; i < 32; ++i)
                    if (((c.allMask >> i) & 1) && i != j)
                    {
                        const V3 dv = sub(c.p[i], c.p[j]);
                        if (dot3(dv, dv) < 1e-8f && (c.s[i] < c.s[j] || (c.s[i] == c.s[j] && i > j)))
                            cullMask &= ~(1u << j);
                    }
        if (cullMask != c.allMask) ++dupeCulled;
        if (__builtin_popcount(cullMask) <= MAXP) ++earlyOut;
        else
        {
            ++pcmFired;
            if (__builtin_popcount(refMask[t]) == MAXP) ++fillRan;
        }
    }

    // ---- upload ------------------------------------------------------------
    // 544-byte record: float4 normal | uint4 allMask | float4[32] point.xyz, sep
    std::vector<float> input(size_t(N) * (544 / 4), 0.0f);
    for (uint32_t t = 0; t < N; ++t)
    {
        float* r = input.data() + size_t(t) * (544 / 4);
        const CrWarpCase& c = cases[t];
        r[0] = c.normal.x; r[1] = c.normal.y; r[2] = c.normal.z;
        std::memcpy(r + 4, &c.allMask, 4);
        for (uint32_t j = 0; j < 32; ++j)
        {
            float* q = r + 8 + 4 * j;
            q[0] = c.p[j].x; q[1] = c.p[j].y; q[2] = c.p[j].z; q[3] = c.s[j];
        }
    }

    PxgBufferHandle bIn = createBufferFrom(dev, input);
    std::vector<uint32_t> outInit(N, 0xdeadbeefu);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    CrWarpPush push{};
    push.input = dev->getBufferDeviceAddress(bIn);
    push.output = dev->getBufferDeviceAddress(bOut);
    push.numCases = N;

    {
        Kernel k("contactReduceWarpBda.spv", "contactReduceWarpLaunch", {}, sizeof(CrWarpPush));
        k.dispatch(N, {}, push, 1u, 1u);
        dev->flush();
    }

    const auto got = downloadBuffer<uint32_t>(dev, bOut, N);

    size_t mm = 0, checked = 0;
    for (uint32_t t = 0; t < N; ++t)
    {
        if (skip[t]) { ++skipped; continue; }
        ++checked;
        if (got[t] != refMask[t])
        {
            ++mm;
            if (mm <= 8)
                std::printf("  [%s] case %u: got=%08x exp=%08x allMask=%08x\n",
                            cases[t].name, t, got[t], refMask[t], cases[t].allMask);
        }
    }

    dev->destroyBuffer(bIn);
    dev->destroyBuffer(bOut);

    PXG_CHECK_MSG(mm == 0, "contactReduceWarp: %zu mask mismatches", mm);
    PXG_CHECK_MSG(checked > N * 3 / 4, "contactReduceWarp: too many skipped (%zu)", skipped);
    PXG_CHECK_MSG(earlyOut > 20, "contactReduceWarp: early-out thin (%zu)", earlyOut);
    PXG_CHECK_MSG(dupeCulled > 100, "contactReduceWarp: dupe cull thin (%zu)", dupeCulled);
    PXG_CHECK_MSG(pcmFired > 200, "contactReduceWarp: full path thin (%zu)", pcmFired);
    PXG_CHECK_MSG(fillRan > 200, "contactReduceWarp: five-point results thin (%zu)", fillRan);
}


// ---------------------------------------------------------------------------
// The SECOND warp instantiation: contactReduce<true, false, 6, false>
// (convexMeshCorrelate.cu:277), which convexTrimeshCorrelate reaches.
//
// Driven against the SAME refReduce oracle, with clustering on, the PCM step off,
// six slots and no dupe test -- so a divergence is attributable to the clustering
// branch and the warp emulation, not to the algorithm.
//
// TWO STRUCTURAL FACTS shape the constructed cases, and both were derived from the
// CUDA rather than assumed:
//
//   * popc(mask) after the four anchors is ALWAYS 4 at this instantiation.  Each
//     anchor is picked from `allMask & ~mask`, so they land on four distinct
//     lanes, and the set cannot be exhausted because popc(allMask) > 6.  So
//     nbClusters is always 4 and the fill loop always runs exactly twice.
//   * a cluster is normally non-empty, because an anchor is at distance zero from
//     itself and the label test is a strict `<`.  The ONE way to empty one is two
//     anchors at the SAME POSITION: the later anchor is zero away from the earlier
//     one, the earlier one wins the strict test, and nothing is left to label
//     itself into the later cluster.  That cluster then costs a slot that the fill
//     loop -- which counts from nbClusters, not from the points actually selected
//     -- never gives back, and the result has FIVE points instead of six.
// ---------------------------------------------------------------------------

PXG_TEST(contactReduceWarpCluster6)
{
    PxgSlangComputeDevice* dev = testDevice();

    const int MAXP = 6;
    const bool CLUSTERING = true, KEEPDEEPEST = false, DUPETEST = false, REDUCE2 = false;
    const float BIAS = 0.05f;

    std::vector<CrWarpCase> cases;

    auto spread = [](CrWarpCase& c) {
        for (uint32_t j = 0; j < 32; ++j)
        {
            const float a = 0.19634954f * float(j);          // 2*pi/32
            c.p[j] = { std::cos(a) * (1.0f + 0.03f * float(j)),
                       std::sin(a) * (1.0f + 0.03f * float(j)),
                       0.011f * float(j) };
            c.s[j] = -0.02f - 0.011f * float(j);             // strictly decreasing, distinct
        }
        c.normal = { 0.0f, 0.0f, 1.0f };
        c.allMask = 0xFFFFFFFFu;
    };

    {   // 1: THE EARLY-OUT at exactly six lanes.  Six is the boundary: `<=` returns
        //    allMask untouched, so nothing downstream runs.
        CrWarpCase c; spread(c);
        c.allMask = (1u << 1) | (1u << 4) | (1u << 9) | (1u << 17) | (1u << 24) | (1u << 30);
        c.name = "early-out, popc 6";
        cases.push_back(c);
    }
    {   // 2: SEVEN LANES -- one past the early-out, the smallest input that reaches
        //    the clustering branch at all.
        CrWarpCase c; spread(c);
        c.allMask = (1u << 1) | (1u << 4) | (1u << 9) | (1u << 17) | (1u << 24) | (1u << 30) | (1u << 31);
        c.name = "seven lanes, the narrowest full path";
        cases.push_back(c);
    }
    {   // 3: ALL 32 LANES LIVE.
        CrWarpCase c; spread(c);
        c.name = "all 32 lanes";
        cases.push_back(c);
    }
    {   // 4: THE CLUSTER BIAS DECIDES A REPRESENTATIVE.  Lane 3 is deeper than
        //    lane 7 by less than the bias.  Lane 7 is arranged to be an anchor
        //    (the deepest overall, hence p0) and lane 3 sits right next to it, so
        //    both land in the same cluster: with the bias the anchor keeps the
        //    slot, without it lane 3 takes it.
        CrWarpCase c; spread(c);
        c.s[7] = -0.001f;                     // the deepest -> p0
        c.s[3] = -0.001f - BIAS * 0.5f;       // deeper than the biased anchor? no: less deep
        c.p[3] = { c.p[7].x + 0.002f, c.p[7].y + 0.002f, c.p[7].z };
        c.name = "cluster bias keeps the anchor";
        cases.push_back(c);
    }
    {   // 5: TWO ANCHORS AT THE SAME POSITION -> an empty cluster -> five points.
        //
        //    Getting two anchors to coincide takes a specific geometry, because
        //    p1 is the FURTHEST point from p0 and p2/p3 are the extremes of the
        //    signed distance to the p0-p1 line -- a coincident pair sits at
        //    distance 0 from p0 and on the line, so it is normally neither.
        //
        //    The construction that works: put every other point strictly on ONE
        //    side of the p0-p1 line.  Then the signed distance is >= 0 everywhere
        //    and its MINIMUM is zero, attained only on the line itself -- so p3
        //    (the argmin) is forced onto p0's coincident twin.
        //
        //      lane 0  (0,0,0)   separation -0.001, the largest -> p0
        //      lane 1  (0,0,0)   p0's twin, much deeper so it cannot be p0
        //      lane 2  (20,0,0)  by far the furthest from p0        -> p1
        //      3..31             y > 0, so d > 0                    -> p2 = max y
        //                                                              p3 = lane 1
        //
        //    Anchors walk in lane order 0, 1, 2, p2.  Anchor 1 is zero away from
        //    anchor 0, the label test is a strict `<`, so nothing -- not even
        //    lane 1 itself -- lands in cluster 1.  Three representatives plus two
        //    fills is FIVE points, and the missing sixth is the quirk.
        CrWarpCase c; spread(c);
        c.p[0] = { 0.0f, 0.0f, 0.0f };
        c.p[1] = { 0.0f, 0.0f, 0.0f };
        c.p[2] = { 20.0f, 0.0f, 0.0f };
        c.s[0] = -0.001f;
        c.s[1] = -0.5f;
        c.s[2] = -0.3f;
        for (uint32_t j = 3; j < 32; ++j)
        {
            // The sweep must stay strictly inside (0, pi): at lane 31 an angle of
            // pi/30 per lane would be PAST pi, sin would go negative, and that
            // lane -- not lane 1 -- would win the argmin.
            const float a = 0.09f * float(j);                // max 2.79 < pi
            c.p[j] = { std::cos(a) * (1.0f + 0.02f * float(j)),
                       std::sin(a) * (0.5f + 0.03f * float(j)) + 0.05f,
                       0.0f };
            c.s[j] = -0.02f - 0.011f * float(j);
        }
        // Lane 31 is held OUT of the set.  A port that dropped the empty-cluster
        // guard asks crMinIndex to reduce over nothing, which hands back
        // crLowestSetIndex(0) == -1 and shifts by it -- and a shift by 31 is what
        // that lands on.  With lane 31 excluded, any bit 31 in the answer is
        // provably wrong rather than coincidentally right.
        c.allMask = 0x7FFFFFFFu;
        c.name = "coincident anchors leave a cluster empty";
        cases.push_back(c);
    }
    {   // 6: A LANE EQUIDISTANT FROM TWO ANCHORS.  The label test is a strict `<`,
        //    so the EARLIER anchor keeps it -- the only thing that decides.
        CrWarpCase c; spread(c);
        c.p[10] = { -12.0f, 0.0f, 0.0f };
        c.p[20] = { 12.0f, 0.0f, 0.0f };
        c.p[15] = { 0.0f, 0.0f, 0.0f };        // exactly between them
        c.name = "equidistant lane goes to the earlier anchor";
        cases.push_back(c);
    }
    {   // 7: AN EXACT TIE IN THE BIASED SEPARATION inside one cluster.  The lowest
        //    lane wins, and nothing else in the algorithm breaks that tie.
        CrWarpCase c; spread(c);
        c.p[12] = { 0.30f, 0.05f, 0.0f };
        c.p[13] = { 0.31f, 0.05f, 0.0f };       // neighbours -> same cluster
        c.s[12] = -0.007f;
        c.s[13] = -0.007f;                      // bit-identical separations
        c.name = "exact separation tie inside a cluster";
        cases.push_back(c);
    }
    {   // 8: AN OUT-OF-MASK LANE HOLDS THE DEEPEST SEPARATION.  Labels are assigned
        //    to every lane, but the representative pick is restricted to allMask --
        //    a port that dropped the `& allMask` would elect a dead lane.
        CrWarpCase c; spread(c);
        c.allMask = 0xFFFFFFFEu;                // lane 0 excluded
        c.s[0] = -0.0001f;                      // ... and it is the deepest of all
        c.name = "out-of-mask lane is the deepest";
        cases.push_back(c);
    }
    {   // 9: ZERO BIAS.  The bias must be a real term, not folded away: with it at
        //    zero the anchors get no preference and the representatives are simply
        //    the deepest per cluster.
        CrWarpCase c; spread(c);
        c.name = "zero bias (driven separately below)";
        cases.push_back(c);
    }
    {   // 10: A LANE EXACTLY EQUIDISTANT FROM TWO ANCHORS, arranged so the tie
        //     actually changes the answer.  The four anchors are pinned by
        //     construction -- lane 0 has the largest separation (p0), lane 1 is
        //     twenty units away (p1), and lanes 2 and 3 are the extreme +y and -y
        //     (p2, p3).  Lane 4 sits at (10, 0), the same distance from lane 2 as
        //     from lane 3, and is deeper than either even after their anchor bias,
        //     so it takes whichever cluster the tie awards it -- and the OTHER
        //     cluster's representative changes with it.
        //
        //     The loser must NOT come back through the fill loop, or the two
        //     answers converge again.  Lanes 2 and 3 are therefore made SHALLOW,
        //     and three genuinely deep lanes are parked at the end of the warp so
        //     the two fill slots go to them instead.
        CrWarpCase c;
        c.normal = { 0.0f, 0.0f, 1.0f };
        c.allMask = 0xFFFFFFFFu;
        c.p[0] = { 0.0f, 0.0f, 0.0f };      c.s[0] = -0.001f;   // p0
        c.p[1] = { 20.0f, 0.0f, 0.0f };     c.s[1] = -0.05f;    // p1
        c.p[2] = { 10.0f, 5.0f, 0.0f };     c.s[2] = -0.02f;    // p2, shallow
        c.p[3] = { 10.0f, -5.0f, 0.0f };    c.s[3] = -0.02f;    // p3, shallow
        c.p[4] = { 10.0f, 0.0f, 0.0f };     c.s[4] = -0.20f;    // equidistant, deep
        for (uint32_t j = 5; j < 32; ++j)
        {
            c.p[j] = { 1.0f + 0.05f * float(j), 0.5f + 0.05f * float(j), 0.0f };
            c.s[j] = -0.01f - 0.001f * float(j);
        }
        c.s[29] = -0.40f; c.s[30] = -0.41f; c.s[31] = -0.42f;   // the fill's targets
        c.name = "equidistant lane, and the loser cannot be refilled";
        cases.push_back(c);
    }
    {   // 11: SIX LANES WITH TWO COINCIDENT ANCHORS.  popc(allMask) is exactly the
        //     boundary, so the early-out returns all six untouched -- while the
        //     full path would lose a slot to the empty cluster and return five.
        //     That difference is the only thing separating `<=` from `<`.
        CrWarpCase c;
        c.normal = { 0.0f, 0.0f, 1.0f };
        c.allMask = 0x3Fu;
        for (uint32_t j = 0; j < 32; ++j) { c.p[j] = { 0.0f, 0.0f, 0.0f }; c.s[j] = -1.0f; }
        c.p[0] = { 0.0f, 0.0f, 0.0f };   c.s[0] = -0.001f;
        c.p[1] = { 0.0f, 0.0f, 0.0f };   c.s[1] = -0.5f;
        c.p[2] = { 20.0f, 0.0f, 0.0f };  c.s[2] = -0.3f;
        c.p[3] = { 1.0f, 0.4f, 0.0f };   c.s[3] = -0.1f;
        c.p[4] = { 2.0f, 0.9f, 0.0f };   c.s[4] = -0.12f;
        c.p[5] = { 1.5f, 1.4f, 0.0f };   c.s[5] = -0.15f;
        c.name = "six lanes, early-out vs a five-point full path";
        cases.push_back(c);
    }
    {   // 12: EVERY CONTACT AT THE SAME POINT.  Degenerate but reachable, and the
        //     only input that can make p1 collide with p0: the distance from p0 is
        //     zero for every lane, so the argmax has nothing to prefer and falls to
        //     the lowest candidate.  Lane 0 is given the largest separation so it
        //     is p0 AND the lowest lane -- a port that searched `allMask` instead
        //     of `allMask & ~mask` would hand p1 straight back to p0.
        CrWarpCase c;
        c.normal = { 0.0f, 0.0f, 1.0f };
        c.allMask = 0xFFFFFFFFu;
        for (uint32_t j = 0; j < 32; ++j)
        {
            c.p[j] = { 0.25f, -0.75f, 0.0f };
            c.s[j] = -0.05f - 0.003f * float(j);
        }
        c.s[0] = -0.001f;                       // the largest -> p0, and lane 0
        c.name = "all 32 contacts coincide";
        cases.push_back(c);
    }
    const uint32_t nConstructed = uint32_t(cases.size());

    std::mt19937 rng(577215);
    for (uint32_t t = 0; t < 512; ++t)
    {
        CrWarpCase c;
        for (uint32_t j = 0; j < 32; ++j)
        {
            c.p[j] = { -1.5f + 3.0f * float(rng() % 1000) / 1000.0f,
                       -1.5f + 3.0f * float(rng() % 1000) / 1000.0f,
                       -0.2f + 0.4f * float(rng() % 1000) / 1000.0f };
            c.s[j] = -0.4f * float(rng() % 1000) / 1000.0f;
        }
        c.normal = { 0.0f, 0.0f, 1.0f };
        // Between 7 and 32 live lanes: below 7 every case would early out.
        const uint32_t pop = 7 + rng() % 26;
        uint32_t lanes[32];
        for (uint32_t j = 0; j < 32; ++j) lanes[j] = j;
        for (uint32_t j = 0; j < pop; ++j) std::swap(lanes[j], lanes[j + rng() % (32 - j)]);
        c.allMask = 0;
        for (uint32_t j = 0; j < pop; ++j) c.allMask |= 1u << lanes[j];
        c.name = "random";
        cases.push_back(c);
    }

    const uint32_t N = uint32_t(cases.size());

    // Case 9 runs with a zero bias; everything else with BIAS.
    std::vector<float> biasOf(N, BIAS);
    biasOf[8] = 0.0f;

    std::vector<uint32_t> refMask(N);
    std::vector<uint8_t> skip(N, 0);
    size_t earlyOut = 0, fullPath = 0, sixPoints = 0, shortPatch = 0, skipped = 0;
    for (uint32_t t = 0; t < N; ++t)
    {
        const CrWarpCase& c = cases[t];
        Slack sl;
        refMask[t] = refReduce(c.p, c.s, c.normal, c.allMask, biasOf[t], CLUSTERING, KEEPDEEPEST,
                               MAXP, DUPETEST, REDUCE2, sl);
        skip[t] = (t >= nConstructed && sl.near) ? 1 : 0;

        if (__builtin_popcount(c.allMask) <= MAXP) ++earlyOut;
        else
        {
            ++fullPath;
            const int pc = __builtin_popcount(refMask[t]);
            if (pc == MAXP) ++sixPoints;
            else if (pc < MAXP) ++shortPatch;   // an empty cluster cost a slot
        }
    }

    std::vector<float> input(size_t(N) * (544 / 4), 0.0f);
    for (uint32_t t = 0; t < N; ++t)
    {
        float* r = input.data() + size_t(t) * (544 / 4);
        const CrWarpCase& c = cases[t];
        r[0] = c.normal.x; r[1] = c.normal.y; r[2] = c.normal.z;
        r[3] = biasOf[t];                       // the bias rides in normal.w
        std::memcpy(r + 4, &c.allMask, 4);
        for (uint32_t j = 0; j < 32; ++j)
        {
            float* q = r + 8 + 4 * j;
            q[0] = c.p[j].x; q[1] = c.p[j].y; q[2] = c.p[j].z; q[3] = c.s[j];
        }
    }

    PxgBufferHandle bIn = createBufferFrom(dev, input);
    std::vector<uint32_t> outInit(N, 0xdeadbeefu);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    CrWarpPush push{};
    push.input = dev->getBufferDeviceAddress(bIn);
    push.output = dev->getBufferDeviceAddress(bOut);
    push.numCases = N;

    {
        Kernel k("contactReduceWarpBda.spv", "contactReduceWarpCluster6Launch", {}, sizeof(CrWarpPush));
        k.dispatch(N, {}, push, 1u, 1u);
        dev->flush();
    }

    const auto got = downloadBuffer<uint32_t>(dev, bOut, N);

    size_t mm = 0, checked = 0;
    for (uint32_t t = 0; t < N; ++t)
    {
        if (skip[t]) { ++skipped; continue; }
        ++checked;
        if (got[t] != refMask[t])
        {
            ++mm;
            if (mm <= 8)
                std::printf("  [%s] case %u: got=%08x exp=%08x allMask=%08x bias=%g\n",
                            cases[t].name, t, got[t], refMask[t], cases[t].allMask, biasOf[t]);
        }
    }

    dev->destroyBuffer(bIn);
    dev->destroyBuffer(bOut);

    PXG_CHECK_MSG(mm == 0, "contactReduceWarpCluster6: %zu mask mismatches", mm);
    PXG_CHECK_MSG(checked > N * 3 / 4, "contactReduceWarpCluster6: too many skipped (%zu)", skipped);
    PXG_CHECK_MSG(earlyOut > 0, "contactReduceWarpCluster6: the early-out never fires");
    PXG_CHECK_MSG(fullPath > 300, "contactReduceWarpCluster6: full path thin (%zu)", fullPath);
    PXG_CHECK_MSG(sixPoints > 200, "contactReduceWarpCluster6: six-point results thin (%zu)", sixPoints);
    // Without this the empty-cluster quirk is untested, and it is the one place
    // the fill loop's `i = nbClusters` start differs from `i = popc(mask)`.
    PXG_CHECK_MSG(shortPatch > 0,
                  "contactReduceWarpCluster6: no case produced fewer than six points, so the "
                  "empty-cluster slot loss is unexercised");
}

} // namespace
