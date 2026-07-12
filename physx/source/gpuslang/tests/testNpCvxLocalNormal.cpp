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

// Verification of npCvxLocalNormal.slang: calculateLocalNormal() from
// cudaGJKEPA.cu.  It normalizes the sum of the first numContacts (<=4) manifold
// normals, summed in the width-4 tree (pairwise (n0+n1)+(n2+n3)) association.
// The C++ reference transcribes the same pairwise sum + getNormalized; the
// normalized result is compared to tolerance, with groups whose summed vector is
// near-degenerate (tiny magnitude -> the normalize is ill-conditioned or hits
// the m>0 branch boundary) skipped.

#include "pxgTestFramework.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct V3 { float x, y, z; };
struct LocalNormalIn { F4 np[4]; uint32_t numContacts, pad0, pad1, pad2; };
struct LocalNormalOut { F4 normal; };

static inline V3 add(const V3& a, const V3& b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static inline float dot3(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 nrm(const V3& a) { float l = std::sqrt(dot3(a, a)); return l > 0 ? V3{ a.x / l, a.y / l, a.z / l } : a; }

static V3 getNormalized(const V3& v) { float m = dot3(v, v); return m > 0 ? V3{ v.x / std::sqrt(m), v.y / std::sqrt(m), v.z / std::sqrt(m) } : V3{ 0, 0, 0 }; }

static V3 refLocalNormal(const LocalNormalIn& s, bool& marg)
{
    const V3 z{ 0, 0, 0 };
    const V3 n0 = (0u < s.numContacts) ? V3{ s.np[0].x, s.np[0].y, s.np[0].z } : z;
    const V3 n1 = (1u < s.numContacts) ? V3{ s.np[1].x, s.np[1].y, s.np[1].z } : z;
    const V3 n2 = (2u < s.numContacts) ? V3{ s.np[2].x, s.np[2].y, s.np[2].z } : z;
    const V3 n3 = (3u < s.numContacts) ? V3{ s.np[3].x, s.np[3].y, s.np[3].z } : z;
    const V3 sum = add(add(n0, n1), add(n2, n3)); // pairwise (n0+n1)+(n2+n3)
    if (dot3(sum, sum) < 1e-4f) marg = true;      // near-degenerate sum
    return getNormalized(sum);
}

static bool eqN(const V3& r, const F4& v) { return almostEqual(v.x, r.x, 2e-4f, 2e-5f) && almostEqual(v.y, r.y, 2e-4f, 2e-5f) && almostEqual(v.z, r.z, 2e-4f, 2e-5f); }

PXG_TEST(npCvxLocalNormal_average)
{
    std::mt19937 rng(139001);
    const uint32_t N = 12000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<LocalNormalIn> in(N);
    std::vector<uint8_t> skip(N, 0);
    std::vector<V3> ref(N);
    size_t nSkip = 0;
    uint32_t cnt[5] = { 0, 0, 0, 0, 0 };
    for (uint32_t i = 0; i < N; ++i)
    {
        LocalNormalIn s{};
        const uint32_t nc = 1 + (rng() % 4); // 1..4
        s.numContacts = nc;
        // clustered normals (a shared direction + jitter) so the sum is non-degenerate,
        // matching real manifolds; a few fully-random to vary magnitude.
        V3 base = nrm(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) });
        for (uint32_t L = 0; L < 4; ++L)
        {
            V3 n;
            if (i % 7 == 0) n = nrm(V3{ rf(-1, 1), rf(-1, 1), rf(-1, 1) });
            else n = nrm(add(base, V3{ rf(-0.3f, 0.3f), rf(-0.3f, 0.3f), rf(-0.3f, 0.3f) }));
            s.np[L] = { n.x, n.y, n.z, rf(-1, 0) };
        }
        in[i] = s;
        bool marg = false;
        ref[i] = refLocalNormal(s, marg);
        skip[i] = marg ? 1u : 0u;
        if (marg) { ++nSkip; continue; }
        ++cnt[nc];
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<LocalNormalOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct LocalNormalParams { uint32_t numTests; } prm = { N };
    Kernel k("npCvxLocalNormal.spv", "cvxLocalNormal",
             { (uint32_t)sizeof(LocalNormalIn), (uint32_t)sizeof(LocalNormalOut) }, sizeof(LocalNormalParams));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<LocalNormalOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (skip[i]) continue;
        if (!eqN(ref[i], g[i].normal)) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "localNormal: %zu mismatches", mm);
    PXG_CHECK_MSG(nSkip < N / 3, "localNormal: too many marginal (%zu/%u)", nSkip, N);
    PXG_CHECK_MSG(cnt[1] > N / 20 && cnt[2] > N / 20 && cnt[3] > N / 20 && cnt[4] > N / 20,
                  "localNormal: numContacts coverage thin (%u/%u/%u/%u)", cnt[1], cnt[2], cnt[3], cnt[4]);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
