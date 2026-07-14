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

// Verification of npCvxCoreContact.slang: the Gu::Contact patch accumulator
// (addPoint / addPatch with the at-capacity replacement / addPatchPoint with
// the depth bubble and keep1st reducePolygon).  Exact CPU transcription;
// every output compares BITWISE (dot/compare-only arithmetic).

#include "pxgTestFramework.h"

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };
struct V3 { float x, y, z; };
struct AccIn
{
    U4 meta;
    F4 pts[24];
    F4 nrm[24];
};
struct AccOut
{
    U4 counts;
    F4 patchN[16];
    F4 pool[64];
};

const uint32_t NOIDX = 0xFFFFFFFFu;
const uint32_t MAXP = 16, MAXPP = 4;

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b) { return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static V3 sub(V3 a, V3 b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static float magSq(V3 v) { return v.x * v.x + v.y * v.y + v.z * v.z; }

static uint32_t reduceP(const F4 pts[5], uint32_t count, V3 normal, uint32_t inds[4])
{
    inds[0] = inds[1] = inds[2] = inds[3] = NOIDX;
    uint32_t pi0 = 0, pi1 = NOIDX;
    {
        float maxDist = -FLT_MAX;
        const V3 p0{ pts[0].x, pts[0].y, pts[0].z };
        for (uint32_t j = 0; j < count; ++j)
        {
            const float dist = magSq(sub(V3{ pts[j].x, pts[j].y, pts[j].z }, p0));
            if (dist > maxDist + FLT_EPSILON) { maxDist = dist; pi1 = j; }
        }
    }
    if (pi0 == NOIDX || pi1 == NOIDX) return 0;
    if (pi0 == pi1) { inds[0] = pi0; return 1; }

    uint32_t pi2 = NOIDX, pi3 = NOIDX;
    {
        const V3 p0{ pts[pi0].x, pts[pi0].y, pts[pi0].z };
        const V3 p1{ pts[pi1].x, pts[pi1].y, pts[pi1].z };
        const V3 n01 = cross3(normal, sub(p1, p0));
        float minDist = 0, maxDist = 0;
        for (uint32_t i = 0; i < count; ++i)
        {
            const float d = dot3(n01, sub(V3{ pts[i].x, pts[i].y, pts[i].z }, p0));
            if (d < minDist - FLT_EPSILON) { pi2 = i; minDist = d; }
            if (d > maxDist + FLT_EPSILON) { pi3 = i; maxDist = d; }
        }
    }
    uint32_t n = 0;
    inds[n++] = pi0;
    inds[n++] = pi1;
    if (pi2 != NOIDX && pi2 != pi0 && pi2 != pi1) inds[n++] = pi2;
    if (pi3 != NOIDX && pi3 != pi0 && pi3 != pi1 && pi3 != pi2) inds[n++] = pi3;
    return n;
}

static void refAcc(const AccIn& s, AccOut& o)
{
    std::memset(&o, 0, sizeof(o));
    const float SAME_NORMAL = 0.999f;

    uint32_t numPatches = 0, numPoints = 0;
    V3 patchNormal[MAXP];
    uint32_t patchCount[MAXP] = {};
    F4 pool[MAXP * MAXPP] = {};

    const uint32_t numAdds = std::min(s.meta.x, 24u);
    for (uint32_t a = 0; a < numAdds; ++a)
    {
        const V3 position{ s.pts[a].x, s.pts[a].y, s.pts[a].z };
        const float depth = s.pts[a].w;
        const V3 normal{ s.nrm[a].x, s.nrm[a].y, s.nrm[a].z };

        uint32_t target = NOIDX;
        for (uint32_t i = 0; i < numPatches && target == NOIDX; ++i)
            if (dot3(patchNormal[i], normal) >= SAME_NORMAL)
                target = i;

        if (target != NOIDX)
        {
            F4 points[5] = {};
            points[0] = F4{ position.x, position.y, position.z, depth };
            for (uint32_t i = 0; i < patchCount[target]; ++i)
                points[i + 1] = pool[target * MAXPP + i];
            for (uint32_t i = 0; i < patchCount[target]; ++i)
            {
                if (points[i].w > points[i + 1].w) std::swap(points[i], points[i + 1]);
                else break;
            }
            uint32_t inds[4];
            const uint32_t np = reduceP(points, patchCount[target] + 1, patchNormal[target], inds);
            for (uint32_t i = 0; i < np; ++i)
                pool[target * MAXPP + i] = points[inds[i]];
            numPoints -= patchCount[target];
            patchCount[target] = np;
            numPoints += np;
        }
        else if (numPatches < MAXP)
        {
            const uint32_t pi = numPatches++;
            patchNormal[pi] = normal;
            patchCount[pi] = 1;
            pool[pi * MAXPP] = F4{ position.x, position.y, position.z, depth };
            ++numPoints;
        }
        else
        {
            float maxDot0 = -FLT_MAX;
            for (uint32_t i = 0; i < numPatches; ++i)
                maxDot0 = std::fmax(maxDot0, dot3(normal, patchNormal[i]));

            float maxDot1 = -FLT_MAX;
            uint32_t pi0 = NOIDX, pi1 = NOIDX;
            for (uint32_t i = 0; i < numPatches; ++i)
                for (uint32_t j = i + 1; j < numPatches; ++j)
                {
                    const float d = dot3(patchNormal[i], patchNormal[j]);
                    if (d > maxDot1) { maxDot1 = d; pi0 = i; pi1 = j; }
                }

            if (maxDot0 > maxDot1)
                continue;

            float depth0 = pool[pi0 * MAXPP].w;
            for (uint32_t i = 1; i < patchCount[pi0]; ++i)
                depth0 = std::fmin(depth0, pool[pi0 * MAXPP + i].w);
            float depth1 = pool[pi1 * MAXPP].w;
            for (uint32_t i = 1; i < patchCount[pi1]; ++i)
                depth1 = std::fmin(depth1, pool[pi1 * MAXPP + i].w);

            const uint32_t pi = depth0 > depth1 ? pi0 : pi1;
            numPoints -= patchCount[pi];
            patchNormal[pi] = normal;
            patchCount[pi] = 1;
            pool[pi * MAXPP] = F4{ position.x, position.y, position.z, depth };
            ++numPoints;
        }
    }

    o.counts = U4{ numPatches, numPoints, 0, 0 };
    for (uint32_t i = 0; i < MAXP; ++i)
    {
        if (i < numPatches)
            o.patchN[i] = F4{ patchNormal[i].x, patchNormal[i].y, patchNormal[i].z, u2f(patchCount[i]) };
    }
    for (uint32_t i = 0; i < MAXP * MAXPP; ++i)
    {
        const uint32_t pi = i / MAXPP;
        if (pi < numPatches && (i % MAXPP) < patchCount[pi])
            o.pool[i] = pool[i];
    }
}

static bool eqF4(const F4& a, const F4& b)
{ return f2u(a.x) == f2u(b.x) && f2u(a.y) == f2u(b.y) && f2u(a.z) == f2u(b.z) && f2u(a.w) == f2u(b.w); }

PXG_TEST(npCvxCoreContact_addPoints)
{
    std::mt19937 rng(168001);
    const uint32_t N = 6000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<AccIn> in(N);
    std::vector<AccOut> ref(N);
    size_t cFull = 0, cReduce = 0, cDrop = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        AccIn s{};
        const uint32_t numAdds = (i % 3 == 2) ? 20 + uint32_t(rng() % 5)  // many-direction cases need many adds to fill 16 patches
                              : 4 + uint32_t(rng() % 21);
        s.meta = U4{ numAdds, 0, 0, 0 };

        // direction pool: few clusters (patch growth + reduce) vs many
        // distinct directions (capacity replacement / drop)
        const uint32_t numDirs = (i % 3 == 0) ? 2 + uint32_t(rng() % 3)
                              : (i % 3 == 1) ? 6 + uint32_t(rng() % 6)
                              : 18 + uint32_t(rng() % 5);
        V3 dirs[23];
        for (uint32_t d = 0; d < numDirs; ++d)
        {
            V3 v{ rf(-1, 1), rf(-1, 1), rf(-1, 1) };
            const float m = std::sqrt(magSq(v));
            dirs[d] = m > 0 ? V3{ v.x / m, v.y / m, v.z / m } : V3{ 1, 0, 0 };
        }

        for (uint32_t a = 0; a < numAdds; ++a)
        {
            const V3 base = dirs[rng() % numDirs];
            // jitter within / around the 0.999 same-normal cone
            const float j = (rng() % 2) ? rf(0.f, 0.02f) : rf(0.f, 0.002f);
            V3 n{ base.x + rf(-j, j), base.y + rf(-j, j), base.z + rf(-j, j) };
            const float m = std::sqrt(magSq(n));
            n = V3{ n.x / m, n.y / m, n.z / m };
            s.nrm[a] = F4{ n.x, n.y, n.z, 0 };
            s.pts[a] = F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), rf(-0.5f, 0.1f) };
        }

        if (i % 37 == 5 && numAdds >= 2)
        {
            // land the same-normal dot EXACTLY on 0.999 (the comparison is
            // >=): patch normal (1,0,0), query normal x = 0.999f -- the
            // accumulator never renormalizes its inputs
            s.nrm[0] = F4{ 1.f, 0.f, 0.f, 0 };
            s.nrm[1] = F4{ 0.999f, 0.0447101779f, 0.f, 0 };
        }
        in[i] = s;
        refAcc(s, ref[i]);
        if (ref[i].counts.x == MAXP) ++cFull;
        bool reduced = false, multi = false;
        for (uint32_t p = 0; p < ref[i].counts.x; ++p)
        {
            const uint32_t c = f2u(ref[i].patchN[p].w);
            if (c >= 4) reduced = true;
            if (c >= 2) multi = true;
        }
        if (reduced) ++cReduce;
        if (ref[i].counts.x == MAXP && multi) ++cDrop;
    }

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 63) / 64;
    std::vector<AccOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npCvxCoreContact.spv", "cctAddPoints",
             { (uint32_t)sizeof(AccIn), (uint32_t)sizeof(AccOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<AccOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const AccOut& r = ref[i]; const AccOut& q = g[i];
        bool ok = r.counts.x == q.counts.x && r.counts.y == q.counts.y;
        for (int j = 0; j < 16 && ok; ++j) ok = eqF4(r.patchN[j], q.patchN[j]);
        for (int j = 0; j < 64 && ok; ++j) ok = eqF4(r.pool[j], q.pool[j]);
        if (!ok) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "cctAddPoints: %zu mismatches", mm);
    PXG_CHECK_MSG(cFull > N / 32 && cReduce > N / 8,
                  "cctAddPoints: coverage thin (full %zu reduce %zu drop %zu)", cFull, cReduce, cDrop);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
