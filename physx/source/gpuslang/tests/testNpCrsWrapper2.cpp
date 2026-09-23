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
#include "pxgCrsRef.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

// V3/V4, the vector helpers, reduceShared and the four wrappers live in
// pxgCrsRef.h since P9.83b-5 (14/n) -- moved there verbatim from this file.
using namespace pxgtest::crsref;

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
    std::vector<uint32_t> ref1(N), ref2(N), ref3(N), ref4(N);
    std::vector<uint8_t> isChain(N, 0);
    size_t nDiffer = 0, nBoundary = 0, nBoundaryWrong = 0;
    size_t nDupeDiffer = 0, nPlanted = 0, nChain = 0, nNoClusterDiffer = 0;

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

        // ---- THE COINCIDENCE CHAIN -----------------------------------------
        // COINCIDENCE IS NOT TRANSITIVE.  a~b and b~c does not give a~c, and
        // that is the whole difference between CUDA's cull and a sequential
        // one.  With sep[a] < sep[b] < sep[c] along a chain:
        //   simultaneous (CUDA): b dies to a, c dies to b, a lives   -> {a}
        //   sequential:          b dies and is REMOVED, so c has no
        //                        coincident partner left and lives   -> {a,c}
        // Two mutants survived the sweep on exactly this blind spot
        // (dupe-applied-in-place, dupe-scan-post-cull).  Neither is equivalent;
        // the fixture simply had no chain in it.  Three collinear lanes one
        // STEP apart give d(a,b)^2 = d(b,c)^2 = 4.9e-9 (inside 1e-8) and
        // d(a,c)^2 = 1.96e-8 (outside) -- both margins a factor of two, and
        // ~90x the rounding of a coordinate of this magnitude.
        // ---- THE COINCIDENCE CHAIN, ON ITS OWN THIRD OF THE CASES --------
        // A chain case is only an input for the columns that CULL, and that is
        // why it gets its own slice rather than being added to every case.
        //
        // Columns 1 and 2 pass dupeTest=false, so three lanes 7e-5 apart reach
        // their clustering intact, where `d < tbest[j]` compares two squared
        // distances of ~5e-9 built by cancellation from coordinates of
        // magnitude ~8.  At ~1.4% error on those, an FMA'd dot() and the
        // mirror's -ffp-contract=off dot() legitimately disagree: 7 of 3000
        // cases did, and moving the chain to |p| ~ 0.02 to shorten the
        // cancellation made it 30, because an isolated tight triple is MORE
        // likely to supply several anchors, not fewer.  The answer those two
        // columns give on a chain case is ill-conditioned, full stop.
        //
        // Stamping this entry NoContraction would hide that, and would be
        // wrong for an unrelated reason: none of the module's other callers is
        // stamped (npTriPlaneNp, npTriHfNp, npCvxCorrelate and triPlaneNpBda
        // are all absent from PXG_NOCONTRACT_SHADERS), so it would certify a
        // build of the module the engine never runs.  The note this file used
        // to carry, claiming they were "already stamped through their engine
        // callers", was simply wrong.
        //
        // So: a third of the cases carry a chain and assert the two culling
        // columns; the rest carry no chain and assert all four.  Every column
        // is asserted on more than a thousand cases either way, and no case is
        // asked a question its instantiation cannot answer.
        const bool chainCase = (i % 3u) == 1u && nLive >= 9;
        if (chainCase)
        {
            const float STEP = 7e-5f;
            V3 u{ rf(-1, 1), rf(-1, 1), rf(-1, 1) };
            const float ul = mag3(u);
            u = ul > 1e-3f ? scale3(u, 1.0f / ul) : V3{ 0.5773503f, 0.5773503f, 0.5773503f };
            const uint32_t ca = live[6], cb = live[7], cc2 = live[8];
            pts[cb]  = add(pts[ca], scale3(u, STEP));
            pts[cc2] = add(pts[ca], scale3(u, 2.0f * STEP));
            // sep[a] < sep[b] < sep[c] along the chain: simultaneously b dies to
            // a and c dies to b, leaving {a}; sequentially b is removed first and
            // c then has no coincident partner left, leaving {a, c}.
            seps[ca] = -0.30f; seps[cb] = -0.20f; seps[cc2] = -0.10f;
            ++nChain;
        }
        isChain[i] = chainCase;

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
        ref4[i] = refNoCluster5(pts, seps, n, mask, c.bias.x);
        if (!chainCase && ref1[i] != ref2[i]) ++nDiffer;
        if (ref1[i] != ref3[i]) ++nDupeDiffer;
        if (ref3[i] != ref4[i]) ++nNoClusterDiffer;
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
    PXG_CHECK_MSG(nDiffer > N / 5,
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
    PXG_CHECK_MSG(nChain > N / 5,
                  "npCrsWrapper2_pair: only %zu of %u cases carry a coincidence CHAIN "
                  "(they need nine live lanes) -- without one, a sequential cull is "
                  "indistinguishable from CUDA's simultaneous one", nChain, N);
    // And the same guard for the fourth column: it differs from the third only
    // in the TClustering tail and in maxPoints, so if the two never disagreed it
    // would pass against a module with no else branch at all -- which is exactly
    // what the module had until this commit.
    PXG_CHECK_MSG(nNoClusterDiffer > N / 4,
                  "npCrsWrapper2_pair: the clustering and non-clustering tails agree on "
                  "all but %zu of %u cases -- this column cannot see a missing "
                  "TClustering=false branch", nNoClusterDiffer, N);

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

    size_t mm1 = 0, mm2 = 0, mm3 = 0, mm4 = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (!isChain[i] && got[i * 4 + 0] != ref1[i])
        {
            if (mm1 < 4)
                std::printf("  wrapper1 case %u: got 0x%08x want 0x%08x\n",
                            i, got[i * 4 + 0], ref1[i]);
            ++mm1;
        }
        if (!isChain[i] && got[i * 4 + 1] != ref2[i])
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
        if (got[i * 4 + 3] != ref4[i])
        {
            if (mm4 < 4)
                std::printf("  noCluster5 case %u: got 0x%08x want 0x%08x\n",
                            i, got[i * 4 + 3], ref4[i]);
            ++mm4;
        }
    }
    PXG_CHECK_MSG(mm1 == 0, "npCrsWrapper2_pair: %zu wrapper-1 mask mismatches", mm1);
    PXG_CHECK_MSG(mm2 == 0, "npCrsWrapper2_pair: %zu wrapper-2 mask mismatches", mm2);
    PXG_CHECK_MSG(mm3 == 0, "npCrsWrapper2_pair: %zu wrapper-1+dupeTest mask mismatches", mm3);
    PXG_CHECK_MSG(mm4 == 0, "npCrsWrapper2_pair: %zu <false,true,5,true> mask mismatches", mm4);
    // Printed on SUCCESS, not only on failure: the margin above N/4 is the
    // fixture's own strength, and a number that only appears when the test breaks
    // is a number nobody ever reads.
    std::printf("  npCrsWrapper2_pair: %u cases, the two wrappers disagree on %zu (%.1f%%), "
                "%zu at the maxPoints boundary; the dupe cull changes the mask on "
                "%zu (%.1f%%); %zu carry a coincidence chain; the non-clustering "
                "tail differs on %zu (%.1f%%)\n",
                N, nDiffer, 100.0 * double(nDiffer) / double(N), nBoundary,
                nDupeDiffer, 100.0 * double(nDupeDiffer) / double(N), nChain,
                nNoClusterDiffer, 100.0 * double(nNoClusterDiffer) / double(N));

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

} // namespace
