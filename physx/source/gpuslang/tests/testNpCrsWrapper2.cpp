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

// Verification of pxgSlangContactReduceSerial's TWO wrappers over one body
// (contactReduction.cuh:171-177 and :180-188 over contactReduceShared :55-165).
//
// P9.83b-2 folded four transcriptions of contactReduceShared into one and added
// the contactReduce2 wrapper that triangleTriangleCollision needs.  The wrapper 1
// arm is already certified by npTriPlaneNp's fixtures, which re-ran bit-exactly
// across the move; THIS file exists for the wrapper 2 arm, which nothing covered.
//
// THE CPU MIRROR IS WRITTEN FROM THE CUDA, not from the Slang.  The four sites
// distanceAdjustment enters, and the sign flip at the fourth, are transcribed from
// contactReduction.cuh:91, :94, :107 and :110 directly.
//
// AND THE TEST CHECKS THAT THE TWO WRAPPERS ACTUALLY DISAGREE.  A fixture that
// only compared wrapper 2 against its own mirror would pass with both wrappers
// implemented identically -- which is the shape all four pre-existing
// transcriptions had, since they collapse f and g onto one array (correct only
// while the adjustment is zero).  So `nDiffer` is asserted to be a healthy
// fraction of the cases: an observable that is true under the null hypothesis
// measures nothing.
//
// A THIRD COLUMN covers TDoDupeTest, the one template argument that differs
// between the four serial instantiations in a way no other fixture aims at.
// npTriHfNp is the only caller that passes true and its own fixture reaches the
// cull, but only incidentally: its inputs are mesh vertices against a plane, so
// coincident points arrive at no chosen distance and never at exactly equal
// separations.  Here they are PLANTED -- an exact duplicate at an exact
// separation tie, one pair just inside the 1e-8 threshold and one just outside --
// and `nDupeDiffer` guards that column the same way `nDiffer` guards the other
// two.

#include "pxgTestFramework.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

struct V3 { float x, y, z; };
struct V4 { float x, y, z, w; };

static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 add(V3 a, V3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b)
{
    return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}
static V3 scale3(V3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
// PxVec3::magnitude() is PxSqrt(x*x + y*y + z*z) -- sqrt(dot), never a length
// intrinsic.  tools/check_length_intrinsic.py exists for that distinction and the
// mirror has to make the same choice or the comparison measures the choice.
static float mag3(V3 a) { return std::sqrt(dot3(a, a)); }

static const float FLT_MAX_LOCAL = 3.402823466e38f;

static uint32_t argMax(const float* vals, uint32_t mask, float& best)
{
    best = -FLT_MAX_LOCAL;
    uint32_t idx = 0;
    for (uint32_t j = 0; j < 32; ++j)
        if (((mask >> j) & 1) != 0 && vals[j] > best) { best = vals[j]; idx = j; }
    return idx;
}
static uint32_t argMin(const float* vals, uint32_t mask, float& best)
{
    best = FLT_MAX_LOCAL;
    uint32_t idx = 0;
    for (uint32_t j = 0; j < 32; ++j)
        if (((mask >> j) & 1) != 0 && vals[j] < best) { best = vals[j]; idx = j; }
    return idx;
}

// contactReduceShared<TClustering=true, TKeepAnotherDeepestForPCM=true,
//                     TMaxPoints=6, TDoDupeTest=dupeTest>  (contactReduction.cuh:55-165)
static uint32_t reduceShared(const V3* ptRaw, const float* sep, V3 normal,
                             uint32_t allMask, float clusterBias,
                             const float* adj, const float* initial, int maxPoints,
                             bool dupeTest)
{
    // contactReduction.cuh:65-79, transcribed from the CUDA and not from the
    // Slang.  In CUDA every lane runs this against every set bit of allMask and
    // the verdicts are collected with __ballot_sync, so the cull is simultaneous:
    // the mask a lane tests against is the PRE-cull mask, not one thinned by
    // earlier verdicts.  That is why the verdicts are accumulated here and applied
    // only afterwards -- culling in place would let a lane be saved by the death
    // of the lane that killed it.
    if (dupeTest)
    {
        uint32_t dupes = 0;
        for (uint32_t l = 0; l < 32; ++l)                       // threadIndexInWarp
        {
            if (((allMask >> l) & 1) == 0) continue;
            bool imADupe = false;
            for (uint32_t m = allMask; m; m &= (m - 1))
            {
                uint32_t i = 0;
                while (((m >> i) & 1u) == 0u) ++i;              // lowestSetIndex
                const V3 dv = sub(ptRaw[i], ptRaw[l]);          // RAW points, :71
                const float d = dot3(dv, dv);                   // magnitudeSquared
                if (d < 1e-8f && (sep[i] < sep[l] || (sep[i] == sep[l] && i > l)))
                    imADupe = true;
            }
            if (imADupe) dupes |= 1u << l;
        }
        allMask &= ~dupes;                                      // :78
    }

    V3 point[32];
    for (uint32_t j = 0; j < 32; ++j)
        point[j] = sub(ptRaw[j], scale3(normal, dot3(ptRaw[j], normal)));

    uint32_t newCount = 0;
    for (uint32_t j = 0; j < 32; ++j) newCount += (allMask >> j) & 1u;
    if (newCount <= uint32_t(maxPoints))
        return allMask;

    float v;
    float vals[32];

    for (uint32_t j = 0; j < 32; ++j) vals[j] = initial[j] + adj[j];      // :91
    const uint32_t i0 = argMax(vals, allMask, v);
    uint32_t mask = 1u << i0;

    for (uint32_t j = 0; j < 32; ++j)                                     // :93-94
        vals[j] = mag3(sub(point[i0], point[j])) + adj[j];
    const uint32_t i1 = argMax(vals, allMask & ~mask, v);
    mask |= 1u << i1;

    const V3 dir = cross3(normal, sub(point[i1], point[i0]));             // :105
    float dval[32];
    for (uint32_t j = 0; j < 32; ++j) dval[j] = dot3(dir, sub(point[j], point[i0]));

    for (uint32_t j = 0; j < 32; ++j) vals[j] = dval[j] + adj[j];         // :107
    const uint32_t f = argMax(vals, allMask & ~mask, v);
    mask |= 1u << f;

    for (uint32_t j = 0; j < 32; ++j) vals[j] = dval[j] - adj[j];         // :110 SIGN FLIP
    const uint32_t g = argMin(vals, allMask & ~mask, v);
    mask |= 1u << g;

    uint32_t bits = 0;
    for (uint32_t j = 0; j < 32; ++j) bits += (mask >> j) & 1u;
    if (bits == 4 && maxPoints > 4)                                       // :121-128
    {
        const uint32_t i4 = argMin(sep, allMask & ~mask, v);              // RAW sep
        mask |= 1u << i4;
    }

    {                                                                     // :135-165
        float sepB[32];
        for (uint32_t j = 0; j < 32; ++j)
            sepB[j] = sep[j] - (((mask >> j) & 1) != 0 ? clusterBias : 0.0f);

        int label[32];
        int nbClusters = 0;
        {
            uint32_t m = mask;
            float tbest[32];
            for (uint32_t j = 0; j < 32; ++j) { tbest[j] = FLT_MAX_LOCAL; label[j] = -1; }
            for (; m != 0; ++nbClusters, m &= (m - 1))
            {
                uint32_t anchor = 0;
                while (((m >> anchor) & 1u) == 0u) ++anchor;
                for (uint32_t j = 0; j < 32; ++j)
                {
                    const V3 dv = sub(point[j], point[anchor]);
                    const float d = dot3(dv, dv);
                    if (d < tbest[j]) { tbest[j] = d; label[j] = nbClusters; }
                }
            }
        }

        mask = 0;
        for (int i = 0; i < nbClusters; ++i)
        {
            uint32_t cluster = 0;
            for (uint32_t j = 0; j < 32; ++j)
                if (label[j] == i) cluster |= 1u << j;
            cluster &= allMask;
            if (cluster != 0)
                mask |= 1u << argMin(sepB, cluster, v);
        }
        for (int i = nbClusters; i < maxPoints; ++i)
            mask |= 1u << argMin(sepB, allMask & ~mask, v);
    }
    return mask;
}

// wrapper 1 (contactReduction.cuh:171-177): adj = 0, initial = separation
static uint32_t refWrapper1(const V3* pt, const float* sep, V3 n, uint32_t allMask, float bias)
{
    float adj[32], initial[32];
    for (uint32_t j = 0; j < 32; ++j) { adj[j] = 0.0f; initial[j] = sep[j]; }
    return reduceShared(pt, sep, n, allMask, bias, adj, initial, 6, false);
}

// wrapper 1 with the DEFAULT TDoDupeTest = true -- npTriHfNp's instantiation.
static uint32_t refWrapper1Dupe(const V3* pt, const float* sep, V3 n, uint32_t allMask, float bias)
{
    float adj[32], initial[32];
    for (uint32_t j = 0; j < 32; ++j) { adj[j] = 0.0f; initial[j] = sep[j]; }
    return reduceShared(pt, sep, n, allMask, bias, adj, initial, 6, true);
}

// wrapper 2 (contactReduction.cuh:180-188): adj = -separation,
// initial = |pointRaw projected onto the normal plane| (:185 point.magnitude())
static uint32_t refWrapper2(const V3* pt, const float* sep, V3 n, uint32_t allMask, float bias)
{
    float adj[32], initial[32];
    for (uint32_t j = 0; j < 32; ++j)
    {
        const V3 p = sub(pt[j], scale3(n, dot3(pt[j], n)));
        adj[j] = -sep[j];
        initial[j] = mag3(p);
    }
    return reduceShared(pt, sep, n, allMask, bias, adj, initial, 6, false);
}

struct CrsParams { uint32_t numCases, pad0, pad1, pad2; };
struct CrsCase
{
    V4 pt[32];
    V4 sep8[8];
    V4 normal;
    uint32_t misc[4];
    V4 bias;
};

// THE TEST NAME MUST CONTAIN THE SUITE NAME.  ctest runs
// `gpuslang-tests <suite>` and main.cpp's filter is a SUBSTRING match, so a test
// called `crsWrapperPair` under a suite called `npCrsWrapper2` runs nothing --
// which main.cpp correctly treats as a failure (`ran > 0` is part of its exit
// condition).  The convention is the one npTriTriOverlap/npTriTriOverlap_overlaps
// already uses: prefix the test with its suite.
PXG_TEST(npCrsWrapper2_pair)
{
    std::mt19937 rng(783021);
    const uint32_t N = 3000;
    auto rf = [&](float lo, float hi)
    { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<CrsCase> in(N);
    std::vector<uint32_t> ref1(N), ref2(N), ref3(N);
    size_t nDiffer = 0, nBoundary = 0, nBoundaryWrong = 0;
    size_t nDupeDiffer = 0, nPlanted = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        CrsCase c{};
        // A normal that is NOT an axis: with n = +Y the projection drops one
        // component and |projected point| degenerates towards the raw radius,
        // which would make wrapper 2's criterion look more like wrapper 1's.
        V3 n{ rf(-1, 1), rf(-1, 1), rf(-1, 1) };
        const float nl = mag3(n);
        n = scale3(n, nl > 1e-4f ? 1.0f / nl : 1.0f);
        if (nl <= 1e-4f) n = V3{ 0.371391f, 0.557086f, 0.742781f };
        c.normal = V4{ n.x, n.y, n.z, 0 };

        // THE POINT CLOUD IS DELIBERATELY OFF-CENTRE.  |projected point| is a
        // distance from the ORIGIN, so a cloud centred on the origin makes every
        // lane's criterion nearly equal and the wrapper difference vanishes --
        // a Class D blind spot.  The cloud is offset by up to 6 units.
        const V3 centre{ rf(-6, 6), rf(-6, 6), rf(-6, 6) };
        V3 pts[32];
        float seps[32];
        for (uint32_t j = 0; j < 32; ++j)
        {
            pts[j] = V3{ centre.x + rf(-2, 2), centre.y + rf(-2, 2), centre.z + rf(-2, 2) };
            // Separations straddle zero -- penetrating and separated both occur,
            // which matters because adj = -sep flips sign with them.
            seps[j] = rf(-0.4f, 0.25f);
        }
        // c.pt / c.sep8 are filled AFTER the mask is drawn, because the planted
        // coincidences below need to know which lanes are live.

        // THE LIVE-LANE COUNT, AND THE BOUNDARY THE FIRST DRAFT EXCLUDED.
        // The body returns allMask unchanged when `newCount <= maxPoints`
        // (contactReduction.cuh:85).  The first draft insisted on bits > 6 for
        // EVERY case, reasoning that a trivial return reduces nothing -- and in
        // doing so it excluded the only inputs at which `<=` and `<` differ.  The
        // mutation sweep said so: `earlyout-strict` survived.
        //
        // So one case in seven now has EXACTLY six live lanes, and those cases
        // assert the trivial return explicitly.  A boundary no input touches is a
        // boundary no fixture is testing.
        const bool atBoundary = (i % 7u) == 3u;
        uint32_t mask = 0;
        while (true)
        {
            mask = rng();
            uint32_t bits = 0;
            for (uint32_t j = 0; j < 32; ++j) bits += (mask >> j) & 1u;
            if (atBoundary)
            {
                while (bits > 6)                    // trim to exactly six
                {
                    uint32_t top = 31;
                    while (((mask >> top) & 1u) == 0u) --top;
                    mask &= ~(1u << top);
                    --bits;
                }
                if (bits == 6) break;
            }
            else if (bits > 6) break;
        }
        c.misc[0] = mask;
        c.bias = V4{ rf(1e-5f, 5e-3f), 0, 0, 0 };

        // ---- THE PLANTED COINCIDENCES (for the third column) ----------------
        // The cull has three decisions and a random cloud aims at none of them:
        // two of these points land within 1e-4 of each other with probability
        // ~0, and at EXACTLY equal separation with probability 0.  So three
        // pairs are planted among the live lanes, each pointed at one decision:
        //
        //   A  exact duplicate, EXACTLY equal separation -- neither depth test
        //      decides, so only the index tie-break does, and the HIGHER index
        //      is the survivor.  Without this pair, `i > l` could be `i < l` or
        //      absent and nothing would notice.
        //   B  6e-5 apart (squared 3.6e-9: inside 1e-8, outside 1e-10) with the
        //      second lane strictly shallower -- dies by depth, and pins the
        //      threshold from BELOW.
        //   C  4e-4 apart (squared 1.6e-7: outside 1e-8, inside 1e-6) with
        //      different separations -- must NOT be culled, pinning the
        //      threshold from ABOVE.
        //
        // The two offsets are far larger than the ~3e-6 rounding of a
        // coordinate of this magnitude, so every one of these verdicts is
        // decided identically on both sides; and far enough from 1e-8 that no
        // reasonable contraction of dot() can move one across it.
        uint32_t live[32], nLive = 0;
        for (uint32_t j = 0; j < 32; ++j) if ((mask >> j) & 1u) live[nLive++] = j;
        for (uint32_t j = nLive; j > 1; --j)                 // Fisher-Yates
        {
            const uint32_t k = rng() % j;
            const uint32_t t = live[j - 1]; live[j - 1] = live[k]; live[k] = t;
        }
        size_t planted = 0;
        if (nLive >= 6)
        {
            auto unitDir = [&]() {
                V3 d{ rf(-1, 1), rf(-1, 1), rf(-1, 1) };
                const float l = mag3(d);
                return l > 1e-3f ? scale3(d, 1.0f / l) : V3{ 0.5773503f, 0.5773503f, 0.5773503f };
            };
            const uint32_t a = live[0], b = live[1];
            pts[b] = pts[a];                                  // exact duplicate
            seps[b] = seps[a];                                // exact tie
            const uint32_t cc = live[2], d = live[3];
            pts[d] = add(pts[cc], scale3(unitDir(), 6e-5f));
            seps[d] = seps[cc] + 0.01f;                       // d is shallower
            const uint32_t e = live[4], f = live[5];
            pts[f] = add(pts[e], scale3(unitDir(), 4e-4f));
            seps[f] = seps[e] + 0.01f;                        // shallower, but far
            planted = 3;
        }

        for (uint32_t j = 0; j < 32; ++j)
        {
            c.pt[j] = V4{ pts[j].x, pts[j].y, pts[j].z, 0 };
            (&c.sep8[j >> 2].x)[j & 3] = seps[j];
        }
        if (planted) ++nPlanted;

        in[i] = c;
        ref1[i] = refWrapper1(pts, seps, n, mask, c.bias.x);
        ref2[i] = refWrapper2(pts, seps, n, mask, c.bias.x);
        ref3[i] = refWrapper1Dupe(pts, seps, n, mask, c.bias.x);
        if (ref1[i] != ref2[i]) ++nDiffer;
        if (ref1[i] != ref3[i]) ++nDupeDiffer;
        uint32_t bits = 0;
        for (uint32_t j = 0; j < 32; ++j) bits += (mask >> j) & 1u;
        if (bits <= 6)
        {
            ++nBoundary;
            // At or below maxPoints the body must hand allMask straight back --
            // for BOTH wrappers, since the early-out precedes every criterion.
            if (ref1[i] != mask || ref2[i] != mask) ++nBoundaryWrong;
        }
    }

    PXG_CHECK_MSG(nBoundaryWrong == 0,
                  "npCrsWrapper2_pair: %zu boundary case(s) did not return allMask unchanged "
                  "-- the CPU mirror disagrees with contactReduction.cuh:85", nBoundaryWrong);
    PXG_CHECK_MSG(nBoundary > N / 10,
                  "npCrsWrapper2_pair: only %zu of %u cases sit at the maxPoints boundary, "
                  "which is the only place `newCount <= maxPoints` and `<` differ",
                  nBoundary, N);
    // THE GUARD THAT MAKES THE REST MEAN ANYTHING.  If the two wrappers agreed on
    // every case, this fixture would pass against an implementation in which
    // wrapper 2 simply IS wrapper 1 -- which is exactly the shape the four
    // pre-existing transcriptions had.
    PXG_CHECK_MSG(nDiffer > N / 4,
                  "npCrsWrapper2_pair: the two wrappers differ on only %zu of %u cases -- "
                  "the inputs cannot distinguish adj=0 from adj=-sep, so this fixture "
                  "would pass with both wrappers implemented identically",
                  nDiffer, N);
    // THE SAME GUARD FOR THE CULL.  Column 3 is column 1 with TDoDupeTest = true,
    // so if the planted coincidences never changed the answer the column would
    // pass against a body that ignores the flag entirely -- the exact silent
    // failure the flag was withheld for one commit to prevent.
    PXG_CHECK_MSG(nPlanted > N - N / 10,
                  "npCrsWrapper2_pair: only %zu of %u cases could take the planted "
                  "coincidences (they need six live lanes)", nPlanted, N);
    PXG_CHECK_MSG(nDupeDiffer > N / 4,
                  "npCrsWrapper2_pair: the dupe cull changes the mask on only %zu of %u "
                  "cases -- the inputs cannot distinguish TDoDupeTest true from false, "
                  "so this column would pass with the cull not implemented",
                  nDupeDiffer, N);

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<uint32_t> outInit(size_t(N) * 4, 0xDEADBEEFu);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    CrsParams prm{ N, 0, 0, 0 };
    Kernel k("npCrsWrapper2.spv", "crsWrapperPair",
             { (uint32_t)sizeof(CrsCase), 16u }, sizeof(CrsParams));
    k.dispatch((N + 63) / 64, { bIn, bOut }, prm);
    dev->flush();

    const auto got = downloadBuffer<uint32_t>(dev, bOut, size_t(N) * 4);

    size_t mm1 = 0, mm2 = 0, mm3 = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (got[i * 4 + 0] != ref1[i])
        {
            if (mm1 < 4)
                std::printf("  wrapper1 case %u: got 0x%08x want 0x%08x\n",
                            i, got[i * 4 + 0], ref1[i]);
            ++mm1;
        }
        if (got[i * 4 + 1] != ref2[i])
        {
            if (mm2 < 4)
                std::printf("  wrapper2 case %u: got 0x%08x want 0x%08x\n",
                            i, got[i * 4 + 1], ref2[i]);
            ++mm2;
        }
        if (got[i * 4 + 2] != ref3[i])
        {
            if (mm3 < 4)
                std::printf("  wrapper1+dupe case %u: got 0x%08x want 0x%08x\n",
                            i, got[i * 4 + 2], ref3[i]);
            ++mm3;
        }
    }
    PXG_CHECK_MSG(mm1 == 0, "npCrsWrapper2_pair: %zu wrapper-1 mask mismatches", mm1);
    PXG_CHECK_MSG(mm2 == 0, "npCrsWrapper2_pair: %zu wrapper-2 mask mismatches", mm2);
    PXG_CHECK_MSG(mm3 == 0, "npCrsWrapper2_pair: %zu wrapper-1+dupeTest mask mismatches", mm3);
    // Printed on SUCCESS, not only on failure: the margin above N/4 is the
    // fixture's own strength, and a number that only appears when the test breaks
    // is a number nobody ever reads.
    std::printf("  npCrsWrapper2_pair: %u cases, the two wrappers disagree on %zu (%.1f%%), "
                "%zu at the maxPoints boundary; the dupe cull changes the mask on "
                "%zu (%.1f%%)\n",
                N, nDiffer, 100.0 * double(nDiffer) / double(N), nBoundary,
                nDupeDiffer, 100.0 * double(nDupeDiffer) / double(N));

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
