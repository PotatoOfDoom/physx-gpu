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

// Verification of npConvexReduce.slang: ctReducePoints (reducePoints) — the
// 4-point contact-manifold reduction.  The output is a bitmask of the four kept
// candidate indices (deepest / farthest / two side extrema), compared BIT-EXACTLY.
// Each of the four argmin/argmax selections is gated on a float comparison, so a
// case is skipped if any of the four extrema has a near-tie (top-two values within
// MARG) — which also covers the degenerate near-collinear layout where the side
// axis vanishes and every side distance collapses to ~0.

#include "pxgTestFramework.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct V3 { float x, y, z; };
struct F4 { float x, y, z, w; };
struct CrParams { uint32_t numTests, maxPts, pad0, pad1; };

static const uint32_t MAX_PTS = 32; // full warp width, matching the real reducePoints
static const float MARG = 2e-3f;

static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 cross3(V3 a, V3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static float len(V3 a) { return std::sqrt(dot3(a, a)); }

// track the extremum index (lowest on ties) and the gap to the runner-up
struct Sel { float best, second; uint32_t idx; bool isMax; void init(bool mx) { isMax = mx; best = mx ? -FLT_MAX : FLT_MAX; second = mx ? -FLT_MAX : FLT_MAX; idx = 0; }
    void add(float v, uint32_t i) {
        if (isMax) { if (v > best) { second = best; best = v; idx = i; } else if (v > second) second = v; }
        else       { if (v < best) { second = best; best = v; idx = i; } else if (v < second) second = v; }
    }
    bool nearTie() const { return std::fabs(best - second) < MARG; } };

static uint32_t refReduce(const std::vector<F4>& pts, uint32_t base, uint32_t maxPts, uint32_t mask, V3 n, bool& skip)
{
    Sel s0; s0.init(false);
    for (uint32_t i = 0; i < maxPts; ++i) if (mask & (1u << i)) s0.add(pts[base + i].w, i);
    const uint32_t pi0 = s0.idx;
    const V3 p0 = { pts[base + pi0].x, pts[base + pi0].y, pts[base + pi0].z };

    Sel s1; s1.init(true);
    for (uint32_t i = 0; i < maxPts; ++i) if (mask & (1u << i)) { const V3 p = { pts[base + i].x, pts[base + i].y, pts[base + i].z }; s1.add(len(sub(p0, p)), i); }
    const uint32_t pi1 = s1.idx;
    const V3 p1 = { pts[base + pi1].x, pts[base + pi1].y, pts[base + pi1].z };

    const V3 sideAxis = cross3(n, sub(p1, p0));
    Sel s2; s2.init(false);
    Sel s3; s3.init(true);
    for (uint32_t i = 0; i < maxPts; ++i) if (mask & (1u << i)) { const V3 p = { pts[base + i].x, pts[base + i].y, pts[base + i].z }; const float sd = dot3(sideAxis, sub(p, p0)); s2.add(sd, i); s3.add(sd, i); }
    const uint32_t pi2 = s2.idx, pi3 = s3.idx;

    skip = s0.nearTie() || s1.nearTie() || s2.nearTie() || s3.nearTie();
    return (1u << pi0) | (1u << pi1) | (1u << pi2) | (1u << pi3);
}

PXG_TEST(npConvexReduce_reducePoints)
{
    std::mt19937 rng(70011);
    const uint32_t N = 16000;
    std::vector<F4> pts(N * MAX_PTS);
    std::vector<F4> norms(N);
    std::vector<uint32_t> masks(N);
    std::vector<uint32_t> ref(N);
    std::vector<uint8_t> skip(N, 0);
    std::uniform_real_distribution<float> uc(-2.0f, 2.0f);
    std::uniform_real_distribution<float> ud(-1.0f, 0.0f);
    std::uniform_real_distribution<float> u11(-1.0f, 1.0f);
    size_t full4 = 0, fewer = 0, skipped = 0;

    for (uint32_t t = 0; t < N; ++t)
    {
        for (uint32_t i = 0; i < MAX_PTS; ++i) pts[t * MAX_PTS + i] = { uc(rng), uc(rng), uc(rng), ud(rng) };
        V3 n; float m; do { n = { u11(rng), u11(rng), u11(rng) }; m = len(n); } while (m < 1e-2f);
        n = { n.x / m, n.y / m, n.z / m };
        norms[t] = { n.x, n.y, n.z, 0.0f };
        // mask: pick the popcount uniformly in [1,MAX_PTS], then set that many
        // random bits.  Small subsets force the four picks to coincide, giving
        // <4-bit output masks (exercising the OR-dedup path).
        const uint32_t k = 1u + rng() % MAX_PTS;
        uint32_t idx[MAX_PTS];
        for (uint32_t j = 0; j < MAX_PTS; ++j) idx[j] = j;
        for (uint32_t j = 0; j < k; ++j) { const uint32_t r = j + rng() % (MAX_PTS - j); std::swap(idx[j], idx[r]); }
        uint32_t mask = 0u;
        for (uint32_t j = 0; j < k; ++j) mask |= (1u << idx[j]);
        masks[t] = mask;
        bool sk; ref[t] = refReduce(pts, t * MAX_PTS, MAX_PTS, mask, n, sk);
        skip[t] = sk ? 1 : 0;
        if (sk) { ++skipped; continue; }
        if (__builtin_popcount(ref[t]) == 4) ++full4; else ++fewer;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 255) / 256;
    std::vector<uint32_t> outInit(N, 0xffffffffu);
    PxgBufferHandle bP = createBufferFrom(dev, pts);
    PxgBufferHandle bN = createBufferFrom(dev, norms);
    PxgBufferHandle bM = createBufferFrom(dev, masks);
    PxgBufferHandle bO = createBufferFrom(dev, outInit);

    CrParams prm = { N, MAX_PTS, 0, 0 };
    Kernel k("npConvexReduce.spv", "ctReducePoints",
             { (uint32_t)sizeof(F4), (uint32_t)sizeof(F4), 4, 4 }, sizeof(CrParams));
    k.dispatch(GROUPS, { bP, bN, bM, bO }, prm);
    dev->flush();

    const auto g = downloadBuffer<uint32_t>(dev, bO, N);
    size_t mm = 0;
    for (uint32_t t = 0; t < N; ++t) { if (skip[t]) continue; if (g[t] != ref[t]) ++mm; }
    PXG_CHECK_MSG(mm == 0, "ctReducePoints: %zu bitmask mismatches", mm);
    PXG_CHECK_MSG(full4 > N / 4 && fewer > 200, "ctReducePoints: coverage thin (full4=%zu fewer=%zu skip=%zu)", full4, fewer, skipped);

    dev->destroyBuffer(bP); dev->destroyBuffer(bN); dev->destroyBuffer(bM); dev->destroyBuffer(bO);
}

} // namespace
