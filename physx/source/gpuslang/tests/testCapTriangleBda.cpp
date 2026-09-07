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

// Verification of pxgSlangCapTriangle -- generateContacts
// (capsuleTriangle.cuh:329-406), generateEE (:408-461) and generateEEContacts
// (:463-474).  P9.52b-3.
//
// The mirror is written from the CUDA.  Note in particular that generateContacts
// emits the SEGMENT ENDPOINT and not the projected point, and that its radius test
// is UNSQUARED while generateEE's is SQUARED.
//
// On the edge ORDER: generateEEContacts calls (a,b), (b,c), (a,c).  An earlier
// version of this header said reversing an edge moves the contact; it does not, in
// exact arithmetic, and the module header carries the derivation.  What the sweep
// then showed is that it is not bit-exact either: `d` is `dot(n, firstVertex)` and
// the two candidate first vertices differ by the edge vector n is orthogonal to,
// so the two differ by that term's rounding residue -- one to two ulp on the
// tilted-normal case below.
//
// ===========================================================================
// THE GEOMETRY THE CASES ARE BUILT ON
// ===========================================================================
//
// The triangle is a = (0,0,0), b = (1,0,0), c = (0,0,1) in the y = 0 plane, and
// the plane normal handed in is (0,1,0).  With the patch normal also (0,1,0) the
// projection is exact: ipt is just p.y, closestP31 is (p.x, 0, p.z), and the
// barycentrics are p.x and p.z to the bit.  That makes it possible to place a case
// ON a decision boundary rather than near it.

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
// pxgNpRefMath.h and pxgCapsuleTriRef.h.  This file keeps only the probe ABI and
// the cases.
using namespace pxgtest::npref;
using namespace pxgtest::captriref;

// ---------------------------------------------------------------------------
struct CtPush { uint64_t cases, output; uint32_t numCases, pad; };
struct CaseH
{
    float a[4], b[4], c[4], planeNormal[4], normal[4], p[4], q[4], cfg[4];
};
static_assert(sizeof(CaseH) == 128, "CapTriCase mirror");

static void put(float* dst, V3 v) { dst[0] = v.x; dst[1] = v.y; dst[2] = v.z; dst[3] = 0.0f; }
static bool sameBits(float g, float e)
{ uint32_t x, y; std::memcpy(&x, &g, 4); std::memcpy(&y, &e, 4); return x == y; }

struct Spec
{
    V3 a, b, c, planeNormal, normal, p, q;
    float inflatedRadius, sqInflatedRadius;
    uint32_t startN;
    const char* why;
};

PXG_TEST(capTriangleBda)
{
    PxgSlangComputeDevice* dev = testDevice();

    const V3 A = { 0, 0, 0 }, B = { 1, 0, 0 }, C = { 0, 0, 1 };
    const V3 UP = { 0, 1, 0 }, DOWN = { 0, -1, 0 };

    const std::vector<Spec> specs = {
        // ---- generateContacts ------------------------------------------------
        { A,B,C, UP, UP, {0.25f, 0.1f, 0.25f}, {0.3f, 0.15f, 0.3f}, 0.5f, 0.25f, 0,
          "both endpoints over the face, both within the radius -> 2" },
        { A,B,C, UP, UP, {0.25f, 0.1f, 0.25f}, {3.0f, 0.15f, 3.0f}, 0.5f, 0.25f, 0,
          "only p over the face -> 1" },
        { A,B,C, UP, UP, {3.0f, 0.1f, 3.0f}, {0.25f, 0.15f, 0.25f}, 0.5f, 0.25f, 0,
          "only q over the face -> 1" },
        { A,B,C, UP, UP, {3.0f, 0.1f, 3.0f}, {4.0f, 0.15f, 4.0f}, 0.5f, 0.25f, 0,
          "neither over the face -> 0" },
        { A,B,C, UP, UP, {0.25f, 0.9f, 0.25f}, {0.3f, 0.95f, 0.3f}, 0.5f, 0.25f, 0,
          "over the face but OUTSIDE the (unsquared) radius -> 0" },
        { A,B,C, UP, UP, {0.25f, -0.2f, 0.25f}, {0.3f, -0.25f, 0.3f}, 0.5f, 0.25f, 0,
          "BELOW the face: ipt negative, still inside the radius -> 2" },
        { A,B,C, UP, DOWN, {0.25f, 0.1f, 0.25f}, {0.3f, 0.15f, 0.3f}, 0.5f, 0.25f, 0,
          "patch normal opposed: ideom <= 0, the PX_MAX_F32 fallback" },
        // The radius test is `inflatedRadius > ipt` STRICT, and with this geometry
        // ipt IS p.y exactly, so a height equal to the radius sits on the boundary.
        { A,B,C, UP, UP, {0.25f, 0.5f, 0.25f}, {0.3f, 0.25f, 0.3f}, 0.5f, 0.25f, 0,
          "p EXACTLY at the radius (strict test drops it), q inside -> 1" },
        // barycentrics exactly on the tolerance, as in the segment-triangle fixture
        { A,B,C, UP, UP, {-1e-6f, 0.1f, 0.25f}, {5.0f, 0.1f, 5.0f}, 0.5f, 0.25f, 0,
          "v EXACTLY at -eps" },
        { A,B,C, UP, UP, {0.5f, 0.1f, 1.000001f - 0.5f}, {5.0f, 0.1f, 5.0f}, 0.5f, 0.25f, 0,
          "v + w EXACTLY at 1 + eps" },
        // a degenerate triangle, so the bdenom fallback is reached
        { A,A,C, UP, UP, {0.25f, 0.1f, 0.25f}, {0.3f, 0.15f, 0.3f}, 0.5f, 0.25f, 0,
          "degenerate triangle (b == a)" },
        // A TILTED patch normal.  With the patch normal equal to (0,1,0) the
        // projection travels along a direction orthogonal to both ab and ac, so the
        // barycentrics cannot see it at all and the SIGN of `p - normal * ipt` is
        // invisible.  The core reaches this function with the patch normal set to
        // the segment-to-triangle direction whenever the closest feature is an edge
        // (convexHeightfield.cu:735), so a tilted normal is the ordinary case, not
        // an exotic one.
        { A,B,C, UP, {0.28223372f, 0.94077903f, 0.18815581f}, {0.25f, 0.1f, 0.25f},
          {0.3f, 0.15f, 0.3f}, 0.5f, 0.25f, 0, "TILTED patch normal" },

        // ---- generateEE ------------------------------------------------------
        // straddling edge AB (the z = 0 plane), crossing inside the edge
        { A,B,C, UP, UP, {0.5f, 0.1f, -1.0f}, {0.5f, 0.1f, 1.0f}, 0.5f, 0.25f, 0,
          "edge AB: straddles and emits" },
        // same but too far from the edge line
        { A,B,C, UP, UP, {0.5f, 0.9f, -1.0f}, {0.5f, 0.9f, 1.0f}, 0.5f, 0.25f, 0,
          "edge AB: straddles but outside the SQUARED radius" },
        // both endpoints on one side -> the temp > 0 early return on every edge
        { A,B,C, UP, UP, {0.5f, 0.1f, 2.0f}, {0.5f, 0.1f, 3.0f}, 0.5f, 0.25f, 0,
          "all three edges: both endpoints on one side" },
        // crossing the edge LINE outside the edge's own extent -> tValue out of range
        { A,B,C, UP, UP, {-2.0f, 0.1f, -1.0f}, {-2.0f, 0.1f, 1.0f}, 0.5f, 0.25f, 0,
          "edge AB: crosses the line at tValue < 0" },
        { A,B,C, UP, UP, {3.0f, 0.1f, -1.0f}, {3.0f, 0.1f, 1.0f}, 0.5f, 0.25f, 0,
          "edge AB: crosses the line at tValue > 1" },
        // a segment PARALLEL to the edge plane -> npq == 0 exactly
        { A,B,C, UP, UP, {-1.0f, 0.1f, 0.0f}, {2.0f, 0.1f, 0.0f}, 0.5f, 0.25f, 0,
          "edge AB: npq EXACTLY zero (parallel to the edge plane)" },
        // reaching edge BC and edge AC
        { A,B,C, UP, UP, {0.6f, 0.1f, 0.6f}, {0.2f, 0.1f, 0.2f}, 0.5f, 0.25f, 0,
          "edge BC: straddles the hypotenuse" },
        { A,B,C, UP, UP, {-1.0f, 0.1f, 0.5f}, {1.0f, 0.1f, 0.5f}, 0.5f, 0.25f, 0,
          "edge AC: straddles the x = 0 plane" },
        // ---- the cases the first sweep proved missing -------------------------
        // A triangle whose tDenom ROUNDS to zero while the barycentric numerators
        // do not: d11 is 4 + 1e-10, which is 4.0 in float.  With b == a the
        // numerators vanish identically and the two bdenom fallbacks agree, which
        // is why the degenerate case above cannot see them (the same trap as in
        // pxgSlangSegTriangle).
        { A,B,{2.0f, 0.0f, 1e-5f}, UP, UP, {0.25f, 0.1f, 0.25f}, {5.0f, 5.0f, 5.0f},
          0.5f, 0.25f, 0, "tDenom ROUNDS to zero, numerator does not" },
        // A STRONGLY tilted patch normal with p just outside the triangle in x.
        // The projection along `normal` lands INSIDE and the projection along
        // `planeNormal` -- or along -normal -- lands OUTSIDE, which is the only
        // shape of input that can tell those three apart.  Its ipt is 0.384, inside
        // [radius^2, radius) = [0.25, 0.5), so it also separates the unsquared
        // radius test from a squared one.
        { A,B,C, UP, {0.62469505f, 0.78086882f, 0.0f}, {1.1f, 0.3f, 0.05f},
          {5.0f, 5.0f, 5.0f}, 0.5f, 0.25f, 0, "tilted normal, p outside in x" },
        // An endpoint EXACTLY on edge AB's plane, so `temp` is exactly zero and the
        // strict `temp > 0` return differs from a non-strict one.
        { A,B,C, UP, UP, {0.5f, 0.1f, 0.0f}, {0.5f, 0.1f, 1.0f}, 0.5f, 0.25f, 0,
          "edge AB: an endpoint EXACTLY on the edge plane" },
        // sqDist EXACTLY equal to sqInflatedRadius: v is (0, 0.5, 0) and the radius
        // squared is 0.25, so the strict `>` drops what a non-strict one would keep.
        { A,B,C, UP, UP, {0.5f, 0.5f, -1.0f}, {0.5f, 0.5f, 1.0f}, 0.5f, 0.25f, 0,
          "edge AB: sqDist EXACTLY at the squared radius" },
        // sqDist = 0.09, which is below 0.25 while its square ROOT, 0.3, is above:
        // the only band in which comparing against the unsquared distance differs.
        { A,B,C, UP, UP, {0.5f, 0.3f, -1.0f}, {0.5f, 0.3f, 1.0f}, 0.5f, 0.25f, 0,
          "edge AB: sqDist inside the band that separates squared from unsquared" },
        // the numContacts < 2 guards
        { A,B,C, UP, UP, {0.5f, 0.1f, -1.0f}, {0.5f, 0.1f, 1.0f}, 0.5f, 0.25f, 1,
          "starting count 1: one more edge contact fits" },
        { A,B,C, UP, UP, {0.5f, 0.1f, -1.0f}, {0.5f, 0.1f, 1.0f}, 0.5f, 0.25f, 2,
          "starting count 2: every edge call is skipped" },
    };

    const uint32_t n = uint32_t(specs.size());
    std::vector<CaseH> cs(n);
    for (uint32_t i = 0; i < n; ++i)
    {
        put(cs[i].a, specs[i].a); put(cs[i].b, specs[i].b); put(cs[i].c, specs[i].c);
        put(cs[i].planeNormal, specs[i].planeNormal); put(cs[i].normal, specs[i].normal);
        put(cs[i].p, specs[i].p); put(cs[i].q, specs[i].q);
        cs[i].cfg[0] = specs[i].inflatedRadius;
        cs[i].cfg[1] = specs[i].sqInflatedRadius;
        cs[i].cfg[2] = float(specs[i].startN);
        cs[i].cfg[3] = 0.0f;
    }

    PxgBufferHandle bCases = createBufferFrom(dev, cs);
    std::vector<uint32_t> outInit(size_t(n) * 12, 0xdeadbeefu);

    auto run = [&](const char* entry, std::vector<uint32_t>& got) {
        PxgBufferHandle bOut = createBufferFrom(dev, outInit);
        CtPush push{};
        push.cases = dev->getBufferDeviceAddress(bCases);
        push.output = dev->getBufferDeviceAddress(bOut);
        push.numCases = n;
        {
            Kernel k("capTriangleBda.spv", entry, {}, sizeof(CtPush));
            k.dispatch((n + 63u) / 64u, {}, push, 1u, 1u);
            dev->flush();
        }
        got = downloadBuffer<uint32_t>(dev, bOut, outInit.size());
        dev->destroyBuffer(bOut);
    };

    std::vector<uint32_t> gotFace, gotEE, gotBoth;
    run("capFaceLaunch", gotFace);
    run("capEELaunch", gotEE);
    run("capBothLaunch", gotBoth);

    size_t bad = 0;
    size_t faceN[3] = { 0, 0, 0 };
    size_t sawIdeomGuard = 0, sawCon0False = 0, sawCon1False = 0, sawRadiusRejected = 0;
    size_t sawNegativeIpt = 0;
    size_t eeExits[6] = { 0, 0, 0, 0, 0, 0 };
    size_t sawEdgeEmit[3] = { 0, 0, 0 };
    size_t sawStart1 = 0, sawStart2 = 0, sawGuardSkipped = 0;
    size_t sawTripleProductEqual = 0, sawTripleProductRun = 0;

    auto check = [&](const char* what, uint32_t i, const std::vector<uint32_t>& got,
                     const Contacts& e) {
        const uint32_t* r = got.data() + size_t(i) * 12;
        if (r[0] != e.n)
        {
            ++bad;
            std::printf("  [%s case %u %s] n: got=%u exp=%u\n", what, i, specs[i].why,
                        r[0], e.n);
            return;
        }
        for (uint32_t k = 0; k < 2; ++k)
        {
            const float* f = reinterpret_cast<const float*>(r + 4 + k * 4);
            const float exp4[4] = { e.pt[k].x, e.pt[k].y, e.pt[k].z, e.sep[k] };
            static const char* nm[4] = { "x", "y", "z", "sep" };
            for (uint32_t j = 0; j < 4; ++j)
                if (!sameBits(f[j], exp4[j]))
                {
                    ++bad;
                    if (bad <= 24)
                        std::printf("  [%s case %u %s] slot %u %s: got=%.9g exp=%.9g\n",
                                    what, i, specs[i].why, k, nm[j],
                                    double(f[j]), double(exp4[j]));
                }
        }
    };

    for (uint32_t i = 0; i < n; ++i)
    {
        const Spec& s = specs[i];

        FaceTrace ft{};
        const Contacts face = genContacts(s.a, s.b, s.c, s.planeNormal, s.normal,
                                          s.p, s.q, s.inflatedRadius, ft);
        check("face", i, gotFace, face);

        Contacts start = contactsInit();
        start.n = s.startN;
        start.pt[0] = V3{ -7, -7, -7 }; start.sep[0] = -7.0f;
        start.pt[1] = V3{ -9, -9, -9 }; start.sep[1] = -9.0f;
        EeTrace et{};
        const Contacts ee = genEEContacts(s.a, s.b, s.c, s.normal, s.p, s.q,
                                          s.sqInflatedRadius, start, et);
        check("ee", i, gotEE, ee);

        EeTrace bt{};
        const Contacts both = genEEContacts(s.a, s.b, s.c, s.normal, s.p, s.q,
                                            s.sqInflatedRadius, face, bt);
        check("both", i, gotBoth, both);

        // ---- coverage -------------------------------------------------------
        faceN[face.n]++;
        if (ft.ideomGuard) ++sawIdeomGuard;
        if (!ft.con0) ++sawCon0False;
        if (!ft.con1) ++sawCon1False;
        if ((ft.con0 && !ft.took0) || (ft.con1 && !ft.took1)) ++sawRadiusRejected;
        if (ft.took0 && face.sep[0] < 0.0f) ++sawNegativeIpt;

        for (uint32_t k = 0; k < et.nRun; ++k)
        {
            eeExits[et.exits[k]]++;
            if (et.exits[k] == EE_EMIT) sawEdgeEmit[k]++;
            // npq and denom are the same triple product, computed two ways -- see
            // the module header.  Where both were reached, require them to agree to
            // within a few ulp, which is what makes the unguarded divide safe.
            if (et.exits[k] != EE_SAME_SIDE && et.exits[k] != EE_PARALLEL)
            {
                ++sawTripleProductRun;
                const float d = std::fabs(et.npq[k] - et.denom[k]);
                const float m = std::fabs(et.npq[k]) + std::fabs(et.denom[k]);
                if (d <= 1e-5f * (m > 1.0f ? m : 1.0f)) ++sawTripleProductEqual;
            }
        }
        if (s.startN == 1) ++sawStart1;
        if (s.startN == 2) { ++sawStart2; if (et.nRun == 0) ++sawGuardSkipped; }
    }

    dev->destroyBuffer(bCases);

    PXG_CHECK_MSG(bad == 0, "capTriangleBda: %zu mismatches", bad);

    PXG_CHECK_MSG(faceN[0] > 0 && faceN[1] > 0 && faceN[2] > 0,
                  "generateContacts never emitted each of 0, 1 and 2 contacts");
    PXG_CHECK_MSG(sawIdeomGuard > 0,
                  "no case has ideom <= 0 -- the PX_MAX_F32 fallback is untested");
    PXG_CHECK_MSG(sawCon0False > 0 && sawCon1False > 0,
                  "one of the two barycentric tests is never answered false");
    PXG_CHECK_MSG(sawRadiusRejected > 0,
                  "no case passes the barycentric test and FAILS the radius test -- the two "
                  "conditions of the emit are then indistinguishable");
    PXG_CHECK_MSG(sawNegativeIpt > 0,
                  "no emitted face contact has a NEGATIVE separation -- the test is "
                  "`inflatedRadius > ipt` on a SIGNED value, not on a distance");
    PXG_CHECK_MSG(eeExits[EE_SAME_SIDE] > 0, "generateEE never takes its same-side return");
    PXG_CHECK_MSG(eeExits[EE_PARALLEL] > 0,
                  "generateEE never takes its `npq == 0` return -- the exact comparison "
                  "against zero is then untested");
    PXG_CHECK_MSG(eeExits[EE_T_HIGH] > 0, "generateEE never rejects tValue > 1");
    PXG_CHECK_MSG(eeExits[EE_T_LOW] > 0, "generateEE never rejects tValue < 0");
    PXG_CHECK_MSG(eeExits[EE_TOO_FAR] > 0,
                  "generateEE never rejects on the SQUARED radius");
    PXG_CHECK_MSG(eeExits[EE_EMIT] > 0, "generateEE never emits");
    PXG_CHECK_MSG(sawEdgeEmit[0] > 0 && sawEdgeEmit[1] > 0 && sawEdgeEmit[2] > 0,
                  "the three edges do not each emit at least once -- the (a,b), (b,c), "
                  "(a,c) call order is then untested");
    PXG_CHECK_MSG(sawStart1 > 0, "no case starts generateEEContacts at 1");
    PXG_CHECK_MSG(sawGuardSkipped > 0,
                  "no case starts at 2 with every edge call skipped -- the `numContacts < 2` "
                  "guards are then untested");
    PXG_CHECK_MSG(sawTripleProductRun > 0 && sawTripleProductEqual == sawTripleProductRun,
                  "npq and denom disagreed: they are the same triple product computed two "
                  "ways, and that identity is what makes the unguarded divide safe "
                  "(%zu of %zu agreed)", sawTripleProductEqual, sawTripleProductRun);
}

} // namespace
