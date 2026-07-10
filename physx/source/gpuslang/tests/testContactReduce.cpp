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

} // namespace
