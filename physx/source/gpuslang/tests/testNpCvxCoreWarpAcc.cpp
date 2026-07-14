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

// Verification of npCvxCoreWarpAcc.slang: Gu::Contact32, the full-warp
// contact-patch accumulator of convexCoreCollision.cu (L686-917).  Exact
// CPU transcription of the deterministic warp emulation: the inclusive
// warpScan prefix layout of copyPoints (per-lane rotated normals), the
// addPoints round loop with lowest-lane-wins minIndex/maxIndex ballots,
// the two-stage warp reducePoints (deepest / farthest / two side extremes
// with duplicate-collapsing pick masks and popcount write offsets), and
// findCreatePatch (strict > SAME_NORMAL match over 32 lanes with zero
// normals past mNumPatches, append below capacity, and the closest-pair
// drop/replace decision at capacity keyed on slot-0 dists).
// ALL outputs compare bitwise.

#include "pxgTestFramework.h"

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };
struct V3 { float x, y, z; };

const uint32_t WARP_SIZE = 32, MAX_PATCHES = 16, MAX_PATCH_POINTS = 4, MAX_CC = 4;
const uint32_t NOIDX = 0xFFFFFFFFu;
const float SAME_NORMAL = 0.999f;

struct WIn
{
    U4 meta;
    F4 transformQ, transformP;
    U4 nums[2][8];
    F4 pts[2][128];
    F4 nrm[2][32];
};
struct WOut
{
    U4 counts;
    F4 patchN[MAX_PATCHES];
    F4 pool[MAX_PATCHES * MAX_PATCH_POINTS];
};

static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static V3 vadd(V3 a, V3 b) { return V3{ a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 vsub(V3 a, V3 b) { return V3{ a.x - b.x, a.y - b.y, a.z - b.z }; }
static float vdot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 vcross(V3 a, V3 b)
{ return V3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static float vmag(V3 v) { return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }
static V3 qRot(const F4& q, V3 v)
{
    const float vx = 2 * v.x, vy = 2 * v.y, vz = 2 * v.z;
    const float w2 = q.w * q.w - 0.5f, d2 = q.x * vx + q.y * vy + q.z * vz;
    return V3{ vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d2,
               vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d2,
               vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d2 };
}
static uint32_t popc(uint32_t v) { uint32_t c = 0; while (v) { v &= v - 1; ++c; } return c; }

static uint32_t minIdx(const float v[WARP_SIZE], uint32_t mask, float& best)
{
    best = FLT_MAX; uint32_t idx = NOIDX;
    for (uint32_t l = 0; l < WARP_SIZE; ++l)
        if ((mask & (1u << l)) && v[l] < best) { best = v[l]; idx = l; }
    return idx;
}
static uint32_t maxIdx(const float v[WARP_SIZE], uint32_t mask, float& best)
{
    best = -FLT_MAX; uint32_t idx = NOIDX;
    for (uint32_t l = 0; l < WARP_SIZE; ++l)
        if ((mask & (1u << l)) && v[l] > best) { best = v[l]; idx = l; }
    return idx;
}

static void refWarpAcc(const WIn& s, WOut& o)
{
    std::memset(&o, 0, sizeof(o));

    uint32_t mNumPatches = 0, mNumPoints = 0;
    V3 patchNormal[MAX_PATCHES];
    uint32_t patchCount[MAX_PATCHES];
    F4 mPoints[MAX_PATCHES * MAX_PATCH_POINTS];
    for (uint32_t i = 0; i < MAX_PATCHES; ++i) { patchNormal[i] = V3{ 0, 0, 0 }; patchCount[i] = 0; }
    std::memset(mPoints, 0, sizeof(mPoints));

    const uint32_t numCalls = s.meta.x < 2 ? s.meta.x : 2;
    for (uint32_t call = 0; call < numCalls; ++call)
    {
        uint32_t laneNum[WARP_SIZE];
        for (uint32_t l = 0; l < WARP_SIZE; ++l)
        {
            const uint32_t n = (&s.nums[call][l / 4].x)[l & 3];
            laneNum[l] = n < MAX_CC ? n : MAX_CC;
        }

        F4 newPts[WARP_SIZE * MAX_CC];
        V3 newNrm[WARP_SIZE * MAX_CC];
        for (uint32_t i = 0; i < WARP_SIZE * MAX_CC; ++i)
        { newPts[i] = F4{ 0, 0, 0, FLT_MAX }; newNrm[i] = V3{ 0, 0, 0 }; }

        uint32_t prefix = 0, totalPoints = 0;
        for (uint32_t l = 0; l < WARP_SIZE; ++l)
        {
            prefix += laneNum[l];
            const uint32_t off = prefix - laneNum[l];
            const V3 worldNormal = qRot(s.transformQ, V3{ s.nrm[call][l].x, s.nrm[call][l].y, s.nrm[call][l].z });
            for (uint32_t j = 0; j < laneNum[l]; ++j)
            {
                const F4 lp = s.pts[call][l * MAX_CC + j];
                const V3 wp = vadd(qRot(s.transformQ, V3{ lp.x, lp.y, lp.z }),
                                   V3{ s.transformP.x, s.transformP.y, s.transformP.z });
                newPts[off + j] = F4{ wp.x, wp.y, wp.z, lp.w };
                newNrm[off + j] = worldNormal;
            }
            if (l == WARP_SIZE - 1) totalPoints = prefix;
        }

        const uint32_t numFullWarps = (totalPoints + WARP_SIZE - 1) / WARP_SIZE;
        for (uint32_t round = 0; round < numFullWarps; ++round)
        {
            F4 lp[WARP_SIZE]; V3 ln[WARP_SIZE];
            uint32_t allMask = 0;
            for (uint32_t l = 0; l < WARP_SIZE; ++l)
            {
                const uint32_t i = round * WARP_SIZE + l;
                if (i < totalPoints) { allMask |= (1u << l); lp[l] = newPts[i]; ln[l] = newNrm[i]; }
                else { lp[l] = F4{ 0, 0, 0, FLT_MAX }; ln[l] = V3{ 0, 0, 0 }; }
            }

            uint32_t guard = 0;
            while (allMask != 0 && guard < 40)
            {
                ++guard;
                float dists[WARP_SIZE];
                for (uint32_t l = 0; l < WARP_SIZE; ++l) dists[l] = lp[l].w;
                float minDist;
                const uint32_t minDistThread = minIdx(dists, allMask, minDist);
                const V3 minDistNormal = ln[minDistThread];

                uint32_t sameNormalMask = 0;
                for (uint32_t l = 0; l < WARP_SIZE; ++l)
                    if (vdot(ln[l], minDistNormal) > SAME_NORMAL) sameNormalMask |= (1u << l);
                sameNormalMask &= allMask;

                float tmpv[WARP_SIZE]; float md;
                for (uint32_t l = 0; l < WARP_SIZE; ++l) tmpv[l] = lp[l].w;
                const uint32_t pi0 = minIdx(tmpv, sameNormalMask, md);
                const V3 p0{ lp[pi0].x, lp[pi0].y, lp[pi0].z };
                for (uint32_t l = 0; l < WARP_SIZE; ++l) tmpv[l] = vmag(vsub(p0, V3{ lp[l].x, lp[l].y, lp[l].z }));
                const uint32_t pi1 = maxIdx(tmpv, sameNormalMask, md);
                const V3 p1{ lp[pi1].x, lp[pi1].y, lp[pi1].z };
                const V3 sideN = vcross(minDistNormal, vsub(p1, p0));
                for (uint32_t l = 0; l < WARP_SIZE; ++l) tmpv[l] = vdot(sideN, vsub(V3{ lp[l].x, lp[l].y, lp[l].z }, p0));
                const uint32_t pi2 = minIdx(tmpv, sameNormalMask, md);
                const uint32_t pi3 = maxIdx(tmpv, sameNormalMask, md);
                uint32_t bestPointsMask = (1u << pi0) | (1u << pi1) | (1u << pi2) | (1u << pi3);

                F4 tmpBuffer[8];
                std::memset(tmpBuffer, 0, sizeof(tmpBuffer));
                uint32_t count = popc(bestPointsMask);
                for (uint32_t l = 0; l < WARP_SIZE; ++l)
                    if (bestPointsMask & (1u << l))
                        tmpBuffer[popc(bestPointsMask & ((1u << l) - 1))] = lp[l];

                allMask &= ~sameNormalMask;

                uint32_t patchIndex = NOIDX;
                {
                    float dots[WARP_SIZE];
                    for (uint32_t l = 0; l < WARP_SIZE; ++l)
                    {
                        const V3 pn = l < mNumPatches ? patchNormal[l] : V3{ 0, 0, 0 };
                        dots[l] = vdot(minDistNormal, pn);
                    }
                    float maxDot;
                    const uint32_t maxDotThread = maxIdx(dots, 0xFFFFFFFFu, maxDot);

                    if (maxDot > SAME_NORMAL)
                        patchIndex = maxDotThread;
                    else if (mNumPatches < MAX_PATCHES)
                    {
                        patchNormal[mNumPatches] = minDistNormal;
                        patchCount[mNumPatches] = 0;
                        ++mNumPatches;
                        patchIndex = mNumPatches - 1;
                    }
                    else
                    {
                        float pMax[WARP_SIZE]; uint32_t pClosest[WARP_SIZE];
                        for (uint32_t l = 0; l < WARP_SIZE; ++l)
                        {
                            const V3 pn = l < mNumPatches ? patchNormal[l] : V3{ 0, 0, 0 };
                            float patchMaxDot = -FLT_MAX;
                            uint32_t closestPatchThread = NOIDX;
                            for (uint32_t i = l + 1; i < MAX_PATCHES; ++i)
                            {
                                const float d = vdot(pn, patchNormal[i]);
                                if (patchMaxDot < d) { patchMaxDot = d; closestPatchThread = i; }
                            }
                            pMax[l] = patchMaxDot; pClosest[l] = closestPatchThread;
                        }
                        float patchMaxDot;
                        const uint32_t patchMaxDotThread = maxIdx(pMax, 0xFFFFFFFFu, patchMaxDot);

                        if (patchMaxDot > maxDot)
                        {
                            const uint32_t closestPatchThread = pClosest[patchMaxDotThread];
                            const float dist0 = mPoints[patchMaxDotThread * MAX_PATCH_POINTS].w;
                            const float dist1 = mPoints[closestPatchThread * MAX_PATCH_POINTS].w;
                            const uint32_t pi = dist0 > dist1 ? patchMaxDotThread : closestPatchThread;
                            mNumPoints -= patchCount[pi];
                            patchCount[pi] = 0;
                            patchNormal[pi] = minDistNormal;
                            patchIndex = pi;
                        }
                    }
                }

                if (patchIndex == NOIDX)
                    continue;

                for (uint32_t j = 0; j < patchCount[patchIndex]; ++j)
                    tmpBuffer[count + j] = mPoints[patchIndex * MAX_PATCH_POINTS + j];
                count += patchCount[patchIndex];

                const uint32_t patchPointMask = count >= 32 ? 0xFFFFFFFFu : ((1u << count) - 1);
                F4 pp[WARP_SIZE];
                for (uint32_t l = 0; l < WARP_SIZE; ++l)
                    pp[l] = l < count ? tmpBuffer[l] : F4{ 0, 0, 0, 0 };
                for (uint32_t l = 0; l < WARP_SIZE; ++l) tmpv[l] = pp[l].w;
                float md2;
                const uint32_t qi0 = minIdx(tmpv, patchPointMask, md2);
                const uint32_t deepest2 = qi0;
                const V3 q0{ pp[qi0].x, pp[qi0].y, pp[qi0].z };
                for (uint32_t l = 0; l < WARP_SIZE; ++l) tmpv[l] = vmag(vsub(q0, V3{ pp[l].x, pp[l].y, pp[l].z }));
                const uint32_t qi1 = maxIdx(tmpv, patchPointMask, md2);
                const V3 q1{ pp[qi1].x, pp[qi1].y, pp[qi1].z };
                const V3 sideN2 = vcross(minDistNormal, vsub(q1, q0));
                for (uint32_t l = 0; l < WARP_SIZE; ++l) tmpv[l] = vdot(sideN2, vsub(V3{ pp[l].x, pp[l].y, pp[l].z }, q0));
                const uint32_t qi2 = minIdx(tmpv, patchPointMask, md2);
                const uint32_t qi3 = maxIdx(tmpv, patchPointMask, md2);
                bestPointsMask = (1u << qi0) | (1u << qi1) | (1u << qi2) | (1u << qi3);

                const uint32_t newCount = popc(bestPointsMask);
                for (uint32_t l = 0; l < WARP_SIZE; ++l)
                    if (bestPointsMask & (1u << l))
                        mPoints[patchIndex * MAX_PATCH_POINTS + popc(bestPointsMask & ((1u << l) - 1))] = pp[l];

                const uint32_t dOff = popc(bestPointsMask & ((1u << deepest2) - 1));
                const F4 tv = mPoints[patchIndex * MAX_PATCH_POINTS];
                mPoints[patchIndex * MAX_PATCH_POINTS] = mPoints[patchIndex * MAX_PATCH_POINTS + dOff];
                mPoints[patchIndex * MAX_PATCH_POINTS + dOff] = tv;

                mNumPoints -= patchCount[patchIndex];
                patchCount[patchIndex] = newCount;
                mNumPoints += patchCount[patchIndex];
            }
        }
    }

    o.counts = U4{ mNumPatches, mNumPoints, 0, 0 };
    for (uint32_t i = 0; i < MAX_PATCHES; ++i)
    {
        const V3 n = i < mNumPatches ? patchNormal[i] : V3{ 0, 0, 0 };
        uint32_t c = i < mNumPatches ? patchCount[i] : 0;
        float cf; std::memcpy(&cf, &c, 4);
        o.patchN[i] = F4{ n.x, n.y, n.z, cf };
    }
    for (uint32_t i = 0; i < MAX_PATCHES * MAX_PATCH_POINTS; ++i)
    {
        const uint32_t pi = i / MAX_PATCH_POINTS;
        const bool live = pi < mNumPatches && (i % MAX_PATCH_POINTS) < patchCount[pi];
        o.pool[i] = live ? mPoints[i] : F4{ 0, 0, 0, 0 };
    }
}

PXG_TEST(npCvxCoreWarpAcc_addPoints)
{
    std::mt19937 rng(173001);
    const uint32_t N = 3000;
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };
    auto randQuat = [&]() { float a[4]; float m2 = 0; for (int k = 0; k < 4; ++k) { a[k] = rf(-1, 1); m2 += a[k] * a[k]; } const float i = 1.f / std::sqrt(m2 > 0 ? m2 : 1.f); return F4{ a[0] * i, a[1] * i, a[2] * i, a[3] * i }; };

    std::vector<WIn> in(N);
    std::vector<WOut> ref(N);
    size_t cFull = 0, cMulti = 0, cTwoCalls = 0, cSparse = 0, cTie = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        WIn s{};
        const uint32_t numCalls = (i % 3 == 2) ? 2 : 1;
        if (numCalls == 2) ++cTwoCalls;
        s.meta = U4{ numCalls, 0, 0, 0 };
        s.transformQ = (i % 6 == 1) ? F4{ 0, 0, 0, 1 } : randQuat();
        s.transformP = F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), 0 };

        // direction pool: few clusters (patch growth) vs many distinct
        // directions (capacity replacement / drop)
        const uint32_t numDirs = (i % 3 == 0) ? 2 + uint32_t(rng() % 3)
                              : (i % 3 == 1) ? 6 + uint32_t(rng() % 6)
                              : 18 + uint32_t(rng() % 5);
        V3 dirs[23];
        for (uint32_t d = 0; d < numDirs; ++d)
        {
            V3 v{ rf(-1, 1), rf(-1, 1), rf(-1, 1) };
            const float m = std::sqrt(vdot(v, v));
            dirs[d] = m > 0 ? V3{ v.x / m, v.y / m, v.z / m } : V3{ 1, 0, 0 };
        }

        bool sparse = (i % 5 == 3);
        if (sparse) ++cSparse;
        for (uint32_t call = 0; call < numCalls; ++call)
        {
            for (uint32_t l = 0; l < WARP_SIZE; ++l)
            {
                const uint32_t n = sparse ? ((rng() % 4 == 0) ? 1 + rng() % 4 : 0)
                                          : rng() % 5;
                (&s.nums[call][l / 4].x)[l & 3] = n;
                const V3 base = dirs[rng() % numDirs];
                const float j = (rng() % 2) ? rf(0.f, 0.02f) : rf(0.f, 0.002f);
                V3 nv{ base.x + rf(-j, j), base.y + rf(-j, j), base.z + rf(-j, j) };
                const float m = std::sqrt(vdot(nv, nv));
                nv = V3{ nv.x / m, nv.y / m, nv.z / m };
                s.nrm[call][l] = F4{ nv.x, nv.y, nv.z, 0 };
                for (uint32_t p = 0; p < 4; ++p)
                    s.pts[call][l * 4 + p] = F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), rf(-0.5f, 0.1f) };
            }
        }
        if (i % 37 == 11)
        {
            // review-pattern exact farthest-distance tie (9+16 == 25+0 in
            // float): lanes 2/3 share a normal with lane 1's deepest point
            // at the origin; two DIFFERENT points at bitwise-equal distance
            // 5 make maxIndex's lowest-lane tie pick observable through the
            // side-axis choice
            const F4 n{ 0, 0, 1, 0 };
            for (uint32_t l = 1; l <= 3; ++l)
            {
                (&s.nums[0][l / 4].x)[l & 3] = 1;
                s.nrm[0][l] = n;
            }
            s.transformQ = F4{ 0, 0, 0, 1 };
            s.transformP = F4{ 0, 0, 0, 0 };
            for (uint32_t l = 4; l <= 5; ++l)
            {
                (&s.nums[0][l / 4].x)[l & 3] = 1;
                s.nrm[0][l] = n;
            }
            s.pts[0][1 * 4 + 0] = F4{ 0, 0, 0, -1.0f };   // deepest, unique
            s.pts[0][2 * 4 + 0] = F4{ 3, 4, 0, 0.0f };    // dist 5 (9+16)
            s.pts[0][3 * 4 + 0] = F4{ 5, 0, 0, 0.1f };    // dist 5 (25+0)
            // side candidates whose pick depends on WHICH tie lane wins:
            // p1 = (3,4,0) -> extremes are (5,0,0) and (1,3,0);
            // p1 = (5,0,0) -> extremes are (0,-2,0) and (3,4,0)
            s.pts[0][4 * 4 + 0] = F4{ 0, -2, 0, 0.2f };
            s.pts[0][5 * 4 + 0] = F4{ 1, 3, 0, 0.3f };
        }
        if (i % 23 == 7)
        {
            // exact depth ties across lanes: the lowest-lane minIndex pick
            // is observable through the deepest-first slot order
            for (uint32_t l = 4; l < 12; ++l)
                for (uint32_t p = 0; p < 4; ++p)
                    s.pts[0][l * 4 + p].w = -0.25f;
            ++cTie;
        }

        in[i] = s;
        refWarpAcc(s, ref[i]);
        if (ref[i].counts.x == MAX_PATCHES) ++cFull;
        if (ref[i].counts.x >= 2) ++cMulti;
    }

    std::printf("[ info ] warpacc cov: full %zu multi %zu twoCalls %zu sparse %zu tie %zu\n",
                cFull, cMulti, cTwoCalls, cSparse, cTie);

    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t GROUPS = (N + 31) / 32;
    std::vector<WOut> outInit(N);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    struct P { uint32_t numTests; } prm = { N };
    Kernel k("npCvxCoreWarpAcc.spv", "cwaAddPoints",
             { (uint32_t)sizeof(WIn), (uint32_t)sizeof(WOut) }, sizeof(P));
    k.dispatch(GROUPS, { bIn, bOut }, prm);
    dev->flush();

    const auto g = downloadBuffer<WOut>(dev, bOut, N);
    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const bool ok = std::memcmp(&ref[i], &g[i], sizeof(WOut)) == 0;
        if (!ok && mm < 5)
            std::printf("MM %u npR %u npG %u ptsR %u ptsG %u\n",
                        i, ref[i].counts.x, g[i].counts.x, ref[i].counts.y, g[i].counts.y);
        if (!ok) ++mm;
    }
    PXG_CHECK_MSG(mm == 0, "cwaAddPoints: %zu mismatches", mm);
    PXG_CHECK_MSG(cFull > 100 && cMulti > N / 2 && cTwoCalls > N / 4 && cSparse > N / 8 && cTie > 50,
                  "cwaAddPoints: coverage thin (full %zu multi %zu)", cFull, cMulti);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
