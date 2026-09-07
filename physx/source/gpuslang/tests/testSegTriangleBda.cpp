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

// Verification of pxgSlangSegTriangle -- distanceSegmentSegmentSquared
// (distanceSegmentSegment.cuh:33-88) and distanceSegmentTriangleSquared
// (capsuleTriangle.cuh:117-327).  P9.52b-2.
//
// Both mirrors are written from the CUDA, including PxClamp
// (PxMath.h:75-94, :142-146) spelled as the comparisons it actually is rather
// than as a library clamp -- see the module header for why that is not the same
// function.
//
// ===========================================================================
// WHAT THE CASES HAVE TO REACH
// ===========================================================================
//
// distanceSegmentSegmentSquared has four decisions and each needs its own input:
// the parallel test (`eps > denom`), the two guarded reciprocals (a and e below
// eps), and the three clamps, each at both ends.  Its two outputs come from
// DIFFERENT stages of the one-pass refinement -- t from the first recomputation
// and s from the second -- so a case where the clamps do nothing cannot tell the
// stages apart.
//
// distanceSegmentTriangleSquared has four ways out, and the fixture requires all
// of them plus both sub-branches of each:
//   1. the segment crosses the plane inside the triangle -> exactly 0.0f;
//   2. both endpoint projections land inside -> the nearer endpoint;
//   3. one lands inside -> the better of that endpoint and the best edge;
//   4. neither -> the best edge, and each of the three edges must be the winner
//      at least once, because the three carry DIFFERENT barycentric formulas.

#include "pxgTestFramework.h"
#include "pxgCapsuleTriRef.h"
#include "pxgNpRefMath.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;

// The mirrors this fixture verifies against live in shared headers -- see
// pxgNpRefMath.h and pxgCapsuleTriRef.h for why, and for the diff that had to come
// first.  This file keeps only what is its own: the probe ABI and the cases.
using namespace pxgtest::npref;
using namespace pxgtest::captriref;

struct StPush { uint64_t cases, output; uint32_t numCases, pad; };
struct SegTriCaseH { float p[4], q[4], a[4], b[4], c[4]; };
struct SegSegCaseH { float p1[4], d1[4], p2[4], d2[4]; };
static_assert(sizeof(SegTriCaseH) == 80, "SegTriCase mirror");
static_assert(sizeof(SegSegCaseH) == 64, "SegSegCase mirror");

static void put(float* dst, V3 v) { dst[0] = v.x; dst[1] = v.y; dst[2] = v.z; dst[3] = 0.0f; }

static bool sameBits(float g, float e)
{ uint32_t x, y; std::memcpy(&x, &g, 4); std::memcpy(&y, &e, 4); return x == y; }

PXG_TEST(segTriangleBda)
{
    PxgSlangComputeDevice* dev = testDevice();

    // ---- distanceSegmentSegmentSquared -----------------------------------
    struct SS { V3 p1, d1, p2, d2; const char* why; };
    const std::vector<SS> ss = {
        { {0,0,0}, {1,0,0}, {0,1,0}, {0,0,1}, "general, skew" },
        { {0,0,0}, {1,0,0}, {0,1,0}, {1,0,0}, "PARALLEL -> denom below eps, sTmp = 0.5" },
        { {0,0,0}, {1,0,0}, {0,1,0}, {2,0,0}, "parallel with a different length" },
        { {0,0,0}, {0,0,0}, {0,1,0}, {0,0,1}, "first direction DEGENERATE -> aRecip 0" },
        { {0,0,0}, {1,0,0}, {0,1,0}, {0,0,0}, "second direction DEGENERATE -> eRecip 0" },
        { {0,0,0}, {0,0,0}, {0,1,0}, {0,0,0}, "both degenerate" },
        { {5,0,0}, {1,0,0}, {0,1,0}, {0,0,1}, "s clamps HIGH" },
        { {-5,0,0}, {1,0,0}, {0,1,0}, {0,0,1}, "s clamps LOW" },
        { {0,0,5}, {1,0,0}, {0,1,0}, {0,0,1}, "t clamps HIGH" },
        { {0,0,-5}, {1,0,0}, {0,1,0}, {0,0,1}, "t clamps LOW" },
        { {0.25f,0.5f,0.25f}, {1,1,0}, {0,0,0}, {0,1,1}, "interior on both" },
        { {1,2,3}, {-2,1,0.5f}, {-1,0,2}, {0.5f,-1,1}, "general, no clamp" },
        // --- the epsilon band.  The first sweep moved `eps` from 1e-6 to 1e-3 and
        //     nothing changed, because every squared length was either 0 or >= 1.
        //     These sit BETWEEN the two, so the two epsilons disagree.
        { {0.3f,0.2f,0.1f}, {0.02f,0,0}, {0,1,0}, {0,0,1}, "a = 4e-4, inside the epsilon band" },
        { {0.3f,0.2f,0.1f}, {1,0,0}, {0,1,0}, {0,0,0.02f}, "e = 4e-4, inside the epsilon band" },
        // --- a direction that is BELOW eps but NOT zero.  With d1 exactly zero,
        //     b and c are zero too and `comp` is zero whatever aRecip is, so the
        //     fallback value is unobservable; a tiny non-zero direction makes the
        //     numerator non-zero and the fallback visible.
        { {0.3f,0.2f,0.1f}, {0.0005f,0,0}, {0,1,0}, {0.3f,0,1}, "a below eps but NOT zero" },
        { {0.3f,0.2f,0.1f}, {1,0,0.3f}, {0,1,0}, {0.0005f,0,0}, "e below eps but NOT zero" },
        // ... and with the numerator POSITIVE.  The case above still could not tell
        // `aRecip = 0` from `aRecip = 1`, because `b * t2 - c` came out negative and
        // the clamp flattened both answers to zero.  Here c is negative, so the
        // mutant's product is positive and lands strictly inside [0, 1].
        { {0.0f,0.2f,0.1f}, {0.0005f,0,0}, {1.0f,1.0f,0.0f}, {0.3f,0,1.0f},
          "a below eps with a POSITIVE numerator" },
    };

    // ---- distanceSegmentTriangleSquared -----------------------------------
    // The triangle is the unit right triangle in the y = 0 plane unless a case
    // says otherwise: a = (0,0,0), b = (1,0,0), c = (0,0,1).
    const V3 A = { 0, 0, 0 }, B = { 1, 0, 0 }, C = { 0, 0, 1 };
    struct ST { V3 p, q, a, b, c; const char* why; };
    const std::vector<ST> st = {
        // 1. crosses the plane inside the triangle
        { {0.25f,  1.0f, 0.25f}, {0.25f, -1.0f, 0.25f}, A, B, C, "crosses INSIDE -> 0" },
        { {0.1f,   0.5f, 0.1f },  {0.2f,  -0.5f, 0.2f }, A, B, C, "crosses inside, oblique" },
        // an endpoint EXACTLY on the plane: dMul == 0, and the gate is non-strict
        { {0.25f,  0.0f, 0.25f}, {0.25f, -1.0f, 0.25f}, A, B, C, "endpoint ON the plane" },
        // crosses the plane OUTSIDE the triangle -> falls through
        { {2.0f,   1.0f, 2.0f },  {2.0f,  -1.0f, 2.0f }, A, B, C, "crosses OUTSIDE" },
        // 2. both projections inside, p nearer / q nearer
        { {0.25f,  1.0f, 0.25f}, {0.3f,   2.0f, 0.3f }, A, B, C, "both inside, p nearer" },
        { {0.25f,  2.0f, 0.25f}, {0.3f,   1.0f, 0.3f }, A, B, C, "both inside, q nearer" },
        // 3. one projection inside
        { {0.25f,  1.0f, 0.25f}, {3.0f,   1.0f, 3.0f }, A, B, C, "con0 only" },
        { {3.0f,   1.0f, 3.0f },  {0.25f,  1.0f, 0.25f}, A, B, C, "con1 only" },
        // con0 with the EDGE winning: p is inside but far above, the other end is
        // close to an edge.
        { {0.25f, 4.0f, 0.25f},  {0.5f,  0.02f, -0.1f}, A, B, C, "con0, edge wins" },
        { {0.5f,  0.02f, -0.1f}, {0.25f, 4.0f, 0.25f},  A, B, C, "con1, edge wins" },
        // ... and with the ENDPOINT winning, which needs the opposite shape: p sits
        // JUST above the face while the rest of the segment climbs steeply away, so
        // by the time it passes over an edge it is already far from it.  Without
        // this the `sqDistPE > sqDist3` comparison is only ever answered one way.
        { {0.2f, 0.05f, 0.2f}, {3.0f, 6.0f, 3.0f}, A, B, C, "con0, endpoint wins" },
        { {3.0f, 6.0f, 3.0f}, {0.2f, 0.05f, 0.2f}, A, B, C, "con1, endpoint wins" },
        // 4. neither inside -- one case per winning edge
        { {0.5f, 0.3f, -1.0f},  {0.5f,  0.3f, -0.5f}, A, B, C, "edge AB wins" },
        { {1.0f, 0.3f,  1.0f},  {1.5f,  0.3f,  1.5f}, A, B, C, "edge BC wins" },
        { {-1.0f, 0.3f, 0.5f},  {-0.5f, 0.3f,  0.5f}, A, B, C, "edge AC wins" },
        // a DEGENERATE triangle: b == a, so tDenom is zero and bdenom is PX_MAX_F32
        { {0.25f, 1.0f, 0.25f}, {0.25f, -1.0f, 0.25f}, A, A, C, "degenerate triangle" },
        // a triangle that is NOT axis-aligned, so the plane normal has three
        // non-zero components and the projections are not exact.
        { {0.3f, 0.4f, 0.2f}, {0.9f, -0.6f, 0.4f},
          {0.0f, 0.0f, 0.0f}, {1.0f, 0.25f, 0.1f}, {0.2f, 0.5f, 1.0f}, "tilted triangle" },
        { {2.0f, 0.4f, 2.0f}, {2.5f, -0.6f, 2.4f},
          {0.0f, 0.0f, 0.0f}, {1.0f, 0.25f, 0.1f}, {0.2f, 0.5f, 1.0f}, "tilted, outside" },
        // --- a plane crossing on the TILTED triangle.  The axis-aligned cases all
        //     have `n.dot(pq)` a power of two, where `nom * (1/d)` and `nom / d`
        //     agree exactly; this one does not, so the reciprocal spelling is
        //     pinned.
        { {0.3f, 2.0f, 0.25f}, {0.34f, -2.0f, 0.28f},
          {0.0f, 0.0f, 0.0f}, {1.0f, 0.25f, 0.1f}, {0.2f, 0.5f, 1.0f}, "tilted, crosses inside" },
        // --- a COLLINEAR triangle: ab and ac are both non-zero and parallel, so
        //     tDenom is zero while the barycentric NUMERATORS are not.  The
        //     degenerate case above has ab == 0, which makes the numerators zero
        //     too and hides which fallback bdenom uses.
        { {0.5f, 1.0f, 0.5f}, {0.5f, -1.0f, 0.5f},
          {0,0,0}, {1,0,0}, {2,0,0}, "COLLINEAR triangle" },
        // A collinear triangle turns out NOT to separate the two bdenom fallbacks:
        // when ac is parallel to ab the barycentric NUMERATORS vanish identically
        // (d11*(p.ab) - d01*(p.ac) is |ac|^2(p.ab) - (ab.ac)(p.ac), which is zero
        // for every p), so 0 * PX_MAX_F32 and 0 * 0 agree.  By Cauchy-Schwarz
        // tDenom is non-negative for ANY triangle, so the fallback is reachable
        // only where ROUNDING drives it to zero: here d11 is 4 + 1e-10, which is
        // 4.0 in float, so tDenom is exactly 4 - 4 while pD21 keeps the 1e-5 the
        // numerator needs.
        { {0.5f, 0.5f, 1.0f}, {0.5f, 0.5f, 2.0f},
          {0,0,0}, {1,0,0}, {2.0f, 1e-5f, 0.0f}, "tDenom ROUNDS to zero, numerator does not" },
        // A COPLANAR segment outside the triangle.  Two things at once: it is the
        // input that makes the plane branch compute `+0 * -INF` = NaN and then
        // discard it (the barycentric test rejects NaN because every one of its
        // comparisons is false), and because every closest-point difference has
        // y == 0, the distances to edge AB and edge AC come out BIT-IDENTICAL --
        // the only kind of input on which the strict `>` in the edge selection
        // differs from `>=`.
        { {-1.0f, 0.0f, -1.0f}, {-2.0f, 0.0f, -2.0f}, A, B, C,
          "COPLANAR and outside: a NaN discarded, and an exact edge TIE" },
        // The tie above does not SEPARATE the two edges: both closest points collapse
        // onto vertex a, so edge AB reports (t10, 0) = (0, 0) and edge AC (0, t12) =
        // (0, 0) and the strictness of the comparison is unobservable.  This one
        // separates them.  The triangle lies in the z = 0 plane and is symmetric
        // under x <-> y; the segment lies in the same plane and maps to ITSELF
        // REVERSED under that mirror, so the two edge computations are the same
        // arithmetic with the two non-zero dot-product terms exchanged -- and a
        // two-term sum is commutative, so they agree BIT FOR BIT.  The segment
        // crosses both edges, so both distances are exactly zero while the
        // parameters are (0.3, 0.4, 0) against (0.7, 0, 0.4).
        { {0.7f, -0.3f, 0.0f}, {-0.3f, 0.7f, 0.0f},
          {0,0,0}, {1,0,0}, {0,1,0}, "coplanar TIE that separates the two edges" },
        // --- a symmetric approach over vertex a, where the distances to edge AB
        //     and edge AC come out equal and the strict `>` decides the tie.
        { {-1.0f, 1.0f, -1.0f}, {-0.5f, 1.0f, -0.5f}, A, B, C, "symmetric corner: an edge TIE" },
        // --- barycentrics ON the validity boundary.  For this triangle the
        //     projection is exact (n is (0,-1,0) and closestP31 keeps x and z), so
        //     v0 == p.x and w0 == p.z to the bit.
        { {-1e-6f, 1.0f, 0.25f}, {-1e-6f, 2.0f, 0.25f}, A, B, C,
          "v EXACTLY at -eps: the non-strict lower bound" },
        { {0.0f, 1.0f, 1.0f + 5e-7f}, {0.0f, 2.0f, 1.0f + 5e-7f}, A, B, C,
          "w just above 1 but inside the tolerance" },
        { {0.5f, 1.0f, 1.000001f - 0.5f}, {0.5f, 2.0f, 1.000001f - 0.5f}, A, B, C,
          "v + w EXACTLY at 1 + eps: the non-strict sum bound" },
    };

    // ---- run the segment-segment probe ------------------------------------
    size_t bad = 0;
    size_t ssParallel = 0, ssADegen = 0, ssEDegen = 0;
    size_t ssSLo = 0, ssSHi = 0, ssTLo = 0, ssTHi = 0;
    {
        std::vector<SegSegCaseH> cs(ss.size());
        for (size_t i = 0; i < ss.size(); ++i)
        {
            put(cs[i].p1, ss[i].p1); put(cs[i].d1, ss[i].d1);
            put(cs[i].p2, ss[i].p2); put(cs[i].d2, ss[i].d2);
        }
        const uint32_t n = uint32_t(cs.size());
        PxgBufferHandle bCases = createBufferFrom(dev, cs);
        std::vector<float> outInit(size_t(n) * 4, -12345.0f);
        PxgBufferHandle bOut = createBufferFrom(dev, outInit);

        StPush push{};
        push.cases = dev->getBufferDeviceAddress(bCases);
        push.output = dev->getBufferDeviceAddress(bOut);
        push.numCases = n;
        {
            Kernel k("segTriangleBda.spv", "segSegLaunch", {}, sizeof(StPush));
            k.dispatch((n + 63u) / 64u, {}, push, 1u, 1u);
            dev->flush();
        }
        const auto got = downloadBuffer<float>(dev, bOut, outInit.size());

        for (uint32_t i = 0; i < n; ++i)
        {
            float s, t; SegSegTrace tr{};
            segSegSquared(ss[i].p1, ss[i].d1, ss[i].p2, ss[i].d2, s, t, tr);
            if (!sameBits(got[i * 4 + 0], s) || !sameBits(got[i * 4 + 1], t))
            {
                ++bad;
                std::printf("  [segSeg %u %s] got (%.9g, %.9g) exp (%.9g, %.9g)\n",
                            i, ss[i].why, double(got[i * 4 + 0]), double(got[i * 4 + 1]),
                            double(s), double(t));
            }
            if (tr.parallelArm) ++ssParallel;
            if (tr.aDegenerate) ++ssADegen;
            if (tr.eDegenerate) ++ssEDegen;
            if (tr.sClampedLo) ++ssSLo;
            if (tr.sClampedHi) ++ssSHi;
            if (tr.tClampedLo) ++ssTLo;
            if (tr.tClampedHi) ++ssTHi;
        }
        dev->destroyBuffer(bCases); dev->destroyBuffer(bOut);
    }

    // ---- run the segment-triangle probe -----------------------------------
    size_t wayCount[5] = { 0, 0, 0, 0, 0 };
    size_t edgeCount[4] = { 0, 0, 0, 0 };
    size_t stEndpointWon = 0, stEdgeWon = 0, stTookQ = 0, stDegenerate = 0, stDMulZero = 0;
    {
        std::vector<SegTriCaseH> cs(st.size());
        for (size_t i = 0; i < st.size(); ++i)
        {
            put(cs[i].p, st[i].p); put(cs[i].q, st[i].q);
            put(cs[i].a, st[i].a); put(cs[i].b, st[i].b); put(cs[i].c, st[i].c);
        }
        const uint32_t n = uint32_t(cs.size());
        PxgBufferHandle bCases = createBufferFrom(dev, cs);
        std::vector<float> outInit(size_t(n) * 4, -12345.0f);
        PxgBufferHandle bOut = createBufferFrom(dev, outInit);

        StPush push{};
        push.cases = dev->getBufferDeviceAddress(bCases);
        push.output = dev->getBufferDeviceAddress(bOut);
        push.numCases = n;
        {
            Kernel k("segTriangleBda.spv", "segTriLaunch", {}, sizeof(StPush));
            k.dispatch((n + 63u) / 64u, {}, push, 1u, 1u);
            dev->flush();
        }
        const auto got = downloadBuffer<float>(dev, bOut, outInit.size());

        for (uint32_t i = 0; i < n; ++i)
        {
            float t, u, v; StTrace tr{};
            const float sq = segTriSquared(st[i].p, st[i].q, st[i].a, st[i].b, st[i].c,
                                           t, u, v, tr);
            const float* r = got.data() + size_t(i) * 4;
            const float exp4[4] = { t, u, v, sq };
            static const char* nm[4] = { "t", "u", "v", "sqDist" };
            for (uint32_t k = 0; k < 4; ++k)
                if (!sameBits(r[k], exp4[k]))
                {
                    ++bad;
                    std::printf("  [segTri %u %s] %s: got=%.9g exp=%.9g\n",
                                i, st[i].why, nm[k], double(r[k]), double(exp4[k]));
                }
            ++wayCount[tr.way];
            ++edgeCount[tr.edge];
            if (tr.endpointWon) ++stEndpointWon;
            else if (tr.way == WAY_CON0 || tr.way == WAY_CON1) ++stEdgeWon;
            if (tr.tookQ) ++stTookQ;
            if (tr.degenerateTri) ++stDegenerate;
            if (tr.dMulZero) ++stDMulZero;
        }
        dev->destroyBuffer(bCases); dev->destroyBuffer(bOut);
    }

    PXG_CHECK_MSG(bad == 0, "segTriangleBda: %zu mismatches", bad);

    PXG_CHECK_MSG(ssParallel > 0, "no segment-segment case is PARALLEL (the sTmp = 0.5 arm)");
    PXG_CHECK_MSG(ssADegen > 0, "no case has a degenerate FIRST direction (aRecip = 0)");
    PXG_CHECK_MSG(ssEDegen > 0, "no case has a degenerate SECOND direction (eRecip = 0)");
    PXG_CHECK_MSG(ssSLo > 0 && ssSHi > 0, "the s clamp is not driven at both ends");
    PXG_CHECK_MSG(ssTLo > 0 && ssTHi > 0, "the t clamp is not driven at both ends");

    PXG_CHECK_MSG(wayCount[WAY_PLANE] > 0, "no case crosses the plane inside the triangle");
    PXG_CHECK_MSG(wayCount[WAY_BOTH_INSIDE] > 0, "no case has BOTH projections inside");
    PXG_CHECK_MSG(wayCount[WAY_CON0] > 0, "no case has only p's projection inside");
    PXG_CHECK_MSG(wayCount[WAY_CON1] > 0, "no case has only q's projection inside");
    PXG_CHECK_MSG(wayCount[WAY_EDGE_ONLY] > 0, "no case has NEITHER projection inside");
    PXG_CHECK_MSG(edgeCount[EDGE_AB] > 0, "edge AB never wins -- its barycentric pair (t10, 0) "
                                          "is then untested");
    PXG_CHECK_MSG(edgeCount[EDGE_BC] > 0, "edge BC never wins -- its barycentric pair "
                                          "(1 - t11, t11) is then untested");
    PXG_CHECK_MSG(edgeCount[EDGE_AC] > 0, "edge AC never wins -- its barycentric pair (0, t12) "
                                          "is then untested");
    PXG_CHECK_MSG(stEndpointWon > 0, "in the one-projection-inside arm the ENDPOINT never wins");
    PXG_CHECK_MSG(stEdgeWon > 0, "in the one-projection-inside arm the EDGE never wins");
    PXG_CHECK_MSG(stTookQ > 0, "in the both-inside arm q is never the nearer endpoint");
    PXG_CHECK_MSG(stDegenerate > 0, "no case has a DEGENERATE triangle (the PX_MAX_F32 fallback)");
    PXG_CHECK_MSG(stDMulZero > 0, "no case puts an endpoint EXACTLY on the plane -- the "
                                  "non-strict `dMul <= 0` gate is then untested");
}

} // namespace
