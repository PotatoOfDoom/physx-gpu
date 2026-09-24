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

// Verification of ttpReduceContactsSerial (pxgSlangTriTriPatch.slang) against a
// CPU mirror of reduceContactsSingleWarp (convexMeshMidphase.cu:440-496).
//
// THE MIRROR IS WRITTEN FROM THE CUDA.  In particular:
//   :457  readMask = ~currentContactMask            -- FULL 32-bit complement
//   :459  readIndex = c + popcount(readMask below lane)
//   :461  readIndex < totalContacts                 -- STRICT
//   :469  c += popcount(readMask)                   -- LANES OFFERED, not taken
//   :471  ballot(hasContact), which is sticky within a pass
//   :472  contactReduce2<true, true, 6, false>      -- WRAPPER 2
//   :478  minIndex over the survivor mask, FLT_MAX outside
//   :492  the compaction scan is over currentContactMask, NOT readMask
//
// AND IT ASSERTS THE LOOP ACTUALLY LOOPS.  The reduction itself is already
// certified by gpuslang.npCrsWrapper2; what is new here is the refill discipline
// around it.  A case set whose contacts all fit in one pass would exercise none
// of it and would still pass, so the fixture counts passes and fails if the
// average is not well above one.

#include "pxgTestFramework.h"
#include "pxgCrsRef.h"
#include "pxgTriTriRef.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

// V3/V4, the vector helpers and contactReduce2 (as refWrapper2) live in
// pxgCrsRef.h since P9.83b-5 (14/n).  This file carried a reduced copy of
// reduceShared -- the <clustering, !dupe, 6> path only -- which was compared
// against the full one expression by expression before it was deleted.
using namespace pxgtest::crsref;
using namespace pxgtest::tritriref;   // refReduce, refMinIndex, the EPS constants -- 15/n

// RefResult and refReduce (reduceContactsSingleWarp, serialised) moved VERBATIM to
// tests/pxgTriTriRef.h in P9.83b-5 (15/n), where the block mirror uses them too.

struct TtrParams { uint32_t numCases, pad0, pad1, pad2; };
struct TtrCase
{
    V4 pts[198];
    V4 normal;      // xyz = normal, w = clusterBias
    uint32_t misc[4];
};

PXG_TEST(npTriTriReduce_stream)
{
    std::mt19937 rng(5512099);
    const uint32_t N = 1200;
    auto rf = [&](float lo, float hi)
    { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<TtrCase> in(N);
    std::vector<RefResult> ref(N);
    size_t totalPasses = 0, multiPass = 0, fullBuffer = 0, shortTail = 0, withDupes = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        TtrCase c{};

        V3 n{ rf(-1, 1), rf(-1, 1), rf(-1, 1) };
        const float nl = mag3(n);
        n = nl > 1e-4f ? scale3(n, 1.0f / nl) : V3{ 0.371391f, 0.557086f, 0.742781f };
        const float bias = rf(1e-5f, 5e-3f);
        c.normal = { n.x, n.y, n.z, bias };

        // THE COUNT DRIVES THE PASS COUNT, so it is spread across the whole range
        // rather than sampled uniformly: 198 is the buffer bound the CUDA asserts,
        // and a SHORT tail (a count that is not a multiple of the lanes offered) is
        // the only input that exercises `c += popcount(readMask)` overshooting.
        uint32_t total;
        const uint32_t pick = i % 6u;
        if (pick == 0u) { total = 198u; ++fullBuffer; }
        else if (pick == 1u) { total = 3u + (rng() % 5u); ++shortTail; }   // fits in one pass
        else if (pick == 2u) { total = 33u + (rng() % 3u); ++shortTail; }  // just past one refill
        else total = 7u + (rng() % 192u);

        c.misc[0] = total;

        // An off-centre cloud: |projected point| is a distance from the ORIGIN and
        // it is wrapper 2's initial criterion, so a cloud centred on the origin
        // makes every lane's criterion nearly equal -- the Class D blind spot
        // npCrsWrapper2 documents.
        const V3 centre{ rf(-6, 6), rf(-6, 6), rf(-6, 6) };
        std::vector<V4> pts(198);
        for (uint32_t j = 0; j < 198u; ++j)
        {
            pts[j] = { centre.x + rf(-2, 2), centre.y + rf(-2, 2), centre.z + rf(-2, 2),
                       rf(-0.4f, 0.25f) };
        }

        // COINCIDENT STAGING POINTS, because the real buffer has them and a random
        // cloud never does.  tryToAssignContactToExistingPatch copies the existing
        // patch's KeepPoints into tempPointsS[0..PointCount) (:832-833) and then
        // appends this round's contacts after them (:839) -- so a contact that has
        // barely moved since the previous round lands in the same buffer as its own
        // kept copy, at a distance of zero.
        //
        // The sweep found this: `reduce-cfg-dupetest`, which flips the config from
        // TDoDupeTest=false to true, survived.  The CUDA passes false at :472 and
        // the difference is only visible on coincident input, so without planted
        // duplicates this column could not tell the two apart.
        if (total >= 12u)
        {
            const uint32_t a = rng() % (total / 2u);
            const uint32_t b = total / 2u + (rng() % (total - total / 2u));
            pts[b] = pts[a];                              // exact duplicate
            const uint32_t d = rng() % (total / 2u);
            const uint32_t e = total / 2u + (rng() % (total - total / 2u));
            pts[e] = pts[d];
            pts[e].w = pts[d].w + 0.01f;                  // coincident, shallower
            ++withDupes;
        }

        for (uint32_t j = 0; j < 198u; ++j)
            c.pts[j] = pts[j];

        in[i] = c;
        ref[i] = refReduce(pts.data(), total, n, bias);
        totalPasses += ref[i].passes;
        if (ref[i].passes > 1) ++multiPass;
    }

    // THE GUARD THAT MAKES THE REST MEAN ANYTHING.  If every case finished in one
    // pass, this fixture would be a second copy of gpuslang.npCrsWrapper2 and the
    // refill discipline -- the only thing it adds -- would be untested.
    PXG_CHECK_MSG(multiPass > N / 2,
                  "npTriTriReduce_stream: only %zu of %u cases needed more than one "
                  "refill pass -- the streaming loop is what this fixture exists to "
                  "test, and it is not being exercised", multiPass, N);
    PXG_CHECK_MSG(withDupes > N / 2,
                  "npTriTriReduce_stream: only %zu of %u cases carry coincident staging "
                  "points -- without them TDoDupeTest=false and =true are "
                  "indistinguishable, and the CUDA passes false deliberately",
                  withDupes, N);
    PXG_CHECK_MSG(fullBuffer > N / 10 && shortTail > N / 10,
                  "npTriTriReduce_stream: %zu full-buffer and %zu short-tail cases; "
                  "both shapes are needed (the short tail is the only input where "
                  "`c += popcount(readMask)` overshoots totalContacts)",
                  fullBuffer, shortTail);

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<float> outInit(size_t(N) * 8 * 4, -1.0f);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    TtrParams prm{ N, 0, 0, 0 };
    Kernel k("npTriTriReduce.spv", "ttrReduce",
             { (uint32_t)sizeof(TtrCase), 16u }, sizeof(TtrParams));
    k.dispatch((N + 63) / 64, { bIn, bOut }, prm);
    dev->flush();

    const auto got = downloadBuffer<float>(dev, bOut, size_t(N) * 8 * 4);

    size_t mmCount = 0, mmMask = 0, mmDeep = 0, mmPts = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const float* g = &got[size_t(i) * 32];
        const uint32_t gotCount = (uint32_t)g[24];
        const float gotDeep = g[25];
        uint32_t gotMask; std::memcpy(&gotMask, &g[26], 4);

        if (gotCount != ref[i].pointCount)
        {
            if (mmCount < 4)
                std::printf("  case %u: pointCount got %u want %u\n", i, gotCount, ref[i].pointCount);
            ++mmCount;
        }
        if (gotMask != ref[i].survivorMask)
        {
            if (mmMask < 4)
                std::printf("  case %u: mask got 0x%08x want 0x%08x\n", i, gotMask, ref[i].survivorMask);
            ++mmMask;
        }
        if (gotDeep != ref[i].deepest)
        {
            if (mmDeep < 4)
                std::printf("  case %u: deepest got %.9g want %.9g\n", i, gotDeep, ref[i].deepest);
            ++mmDeep;
        }
        for (uint32_t kk = 0; kk < 6; ++kk)
        {
            const V4& w = ref[i].keepPoints[kk];
            if (g[kk * 4 + 0] != w.x || g[kk * 4 + 1] != w.y ||
                g[kk * 4 + 2] != w.z || g[kk * 4 + 3] != w.w)
            {
                if (mmPts < 4)
                    std::printf("  case %u keepPoint %u: got (%.9g %.9g %.9g %.9g) "
                                "want (%.9g %.9g %.9g %.9g)\n", i, kk,
                                g[kk * 4 + 0], g[kk * 4 + 1], g[kk * 4 + 2], g[kk * 4 + 3],
                                w.x, w.y, w.z, w.w);
                ++mmPts;
            }
        }
    }

    PXG_CHECK_MSG(mmCount == 0, "npTriTriReduce_stream: %zu pointCount mismatches", mmCount);
    PXG_CHECK_MSG(mmMask == 0, "npTriTriReduce_stream: %zu survivor-mask mismatches", mmMask);
    PXG_CHECK_MSG(mmDeep == 0, "npTriTriReduce_stream: %zu deepest mismatches", mmDeep);
    PXG_CHECK_MSG(mmPts == 0, "npTriTriReduce_stream: %zu keepPoint mismatches", mmPts);

    std::printf("  npTriTriReduce_stream: %u cases, %.2f refill passes on average, "
                "%zu needed more than one; %zu full-buffer, %zu short-tail, "
                "%zu with coincident staging points\n",
                N, double(totalPasses) / double(N), multiPass, fullBuffer, shortTail,
                withDupes);

    for (PxgBufferHandle h : { bIn, bOut }) dev->destroyBuffer(h);
}

// ---------------------------------------------------------------------------
// The acceptance and seed tests (convexMeshMidphase.cu:779-783, :879-885,
// :921-925), mirrored from the CUDA.
//
// THE TWO ACCEPTANCE FORMS DIFFER, and the fixture asserts they do.  If they
// agreed on every case, this test would pass against a port that used the
// tryToAssign form at both sites -- which is exactly the shape the two tests
// invite, since they look like the same three lines.
// ---------------------------------------------------------------------------

// PATCH_ACCEPTANCE_EPS2/3 and refMinIndex (nputils.cuh:151-164) moved to
// tests/pxgTriTriRef.h in P9.83b-5 (15/n).

static void refAcceptExisting(const V3* cn, V3 pn, uint32_t& acceptMask, uint32_t& accept2Mask)
{
    acceptMask = 0; accept2Mask = 0;
    for (uint32_t l = 0; l < 32; ++l)                       // :779-783, UNGATED
    {
        const float proj = dot3(cn[l], pn);
        if (proj > PATCH_ACCEPTANCE_EPS2) accept2Mask |= 1u << l;
        if (proj > PATCH_ACCEPTANCE_EPS3) acceptMask |= 1u << l;
    }
}

static void refAcceptNew(const V3* cn, uint32_t candidateMask, V3 pn, uint32_t seedLane,
                         uint32_t& acceptMask, uint32_t& accept2Mask)
{
    acceptMask = 0; accept2Mask = 0;
    for (uint32_t l = 0; l < 32; ++l)
    {
        if (((candidateMask >> l) & 1u) == 0u) continue;    // :921, the gate
        const float proj = dot3(cn[l], pn);
        if (proj > PATCH_ACCEPTANCE_EPS2) accept2Mask |= 1u << l;
        if (proj > PATCH_ACCEPTANCE_EPS3 || l == seedLane) acceptMask |= 1u << l;  // :925
    }
}

// THE ASSOCIATION OF THE PROJECTION (P9.83b-8 T1).  PxVec3::dot is
// (x*x' + y*y') + z*z'; the port spells it dot(), i.e. OpDot, whose order is the
// driver's.  This finds a lane normal whose projection on `pn` falls on OPPOSITE
// sides of `eps` in the CUDA's order and in the reversed one, x*x' + (y*y' + z*z'):
// jump the component with the largest |pn| analytically close to the crossing,
// then step it one ulp at a time through it.  Deterministic; returns false if a
// trial budget runs out, which the caller counts.
static bool plantAssociation(std::mt19937& rng, V3 pn, float eps, V3& out, bool& cudaAbove)
{
    auto lr = [](V3 a, V3 b) { return (a.x * b.x + a.y * b.y) + a.z * b.z; };
    auto rl = [](V3 a, V3 b) { return a.x * b.x + (a.y * b.y + a.z * b.z); };
    auto noise = [&]() { return (float(rng() % 2000001u) / 1000000.f - 1.f) * 1e-3f; };
    float* comp[3];
    for (uint32_t trial = 0; trial < 20000u; ++trial)
    {
        V3 v{ pn.x * eps + noise(), pn.y * eps + noise(), pn.z * eps + noise() };
        comp[0] = &v.x; comp[1] = &v.y; comp[2] = &v.z;
        const float pc[3] = { pn.x, pn.y, pn.z };
        uint32_t k = 0;
        for (uint32_t j = 1; j < 3; ++j) if (std::fabs(pc[j]) > std::fabs(pc[k])) k = j;
        *comp[k] += (eps - lr(v, pn)) / pc[k];
        const bool up = (lr(v, pn) <= eps) == (pc[k] > 0.f);   // the direction that raises lr
        for (uint32_t s = 0; s < 64u; ++s)
        {
            const bool a = lr(v, pn) > eps, b = rl(v, pn) > eps;
            if (a != b) { out = v; cudaAbove = a; return true; }
            *comp[k] = std::nextafter(*comp[k], up ? 1e30f : -1e30f);
        }
    }
    return false;
}

struct TtaParams { uint32_t numCases, pad0, pad1, pad2; };
struct TtaCase
{
    V4 cn[32];
    V4 sep8[8];
    V4 patchNormal;
    uint32_t misc[4];
};

PXG_TEST(npTriTriReduce_accept)
{
    std::mt19937 rng(220417);
    const uint32_t N = 2000;
    auto rf = [&](float lo, float hi)
    { return lo + (hi - lo) * (float(rng() % 1000000) / 1000000.f); };

    std::vector<TtaCase> in(N);
    std::vector<uint32_t> ra1(N), ra2(N), rb1(N), rb2(N), rseed(N);
    std::vector<float> rmin(N);
    size_t formsDiffer = 0, bandCases = 0, seedRescued = 0, emptyCandidate = 0;
    size_t tieCases = 0, unmaskedTie = 0, exactEps = 0, straddle = 0;
    size_t assocEps3 = 0, assocEps2 = 0, assocCudaAbove = 0, assocMissed = 0;

    for (uint32_t i = 0; i < N; ++i)
    {
        TtaCase c{};

        V3 pn{ rf(-1, 1), rf(-1, 1), rf(-1, 1) };
        const float pl = mag3(pn);
        pn = pl > 1e-4f ? scale3(pn, 1.0f / pl) : V3{ 0.371391f, 0.557086f, 0.742781f };
        c.patchNormal = { pn.x, pn.y, pn.z, 0 };

        // THE NORMALS MUST STRADDLE BOTH EPSILONS.  Random directions in 3D are
        // almost never within 0.9995 of a given one (that is a 1.8-degree cone),
        // so a uniform sample would put every lane below both thresholds and the
        // two masks would be zero -- an observable true under the null hypothesis.
        // So each lane is a small perturbation of the patch normal, with the
        // perturbation scale chosen to land across the interesting band.
        V3 cn[32];
        float sep[32];
        for (uint32_t l = 0; l < 32; ++l)
        {
            const uint32_t bucket = l % 5u;
            //  0: inside 0.9995          1: in the (0.85, 0.9995] band
            //  2: STRADDLING 0.9995      3: a retired contact -- the ZERO normal
            //  4: below 0.85
            float ang;
            if (bucket == 0u) ang = rf(0.0f, 0.025f);        // cos > 0.99968
            else if (bucket == 1u) ang = rf(0.05f, 0.5f);    // cos in (0.877, 0.99875)
            else if (bucket == 2u) ang = rf(0.025f, 0.05f);  // cos in (0.99875, 0.99968)
            else ang = rf(0.6f, 2.0f);                       // cos < 0.825
            // BUCKET 2 IS THE ONE THE SWEEP ASKED FOR.  The first draft had only
            // buckets 0 and 1, whose cosines are 0.99968-and-up and 0.99875-and-
            // down -- so NOTHING landed between them, and `eps3-value` (0.9995 ->
            // 0.999) survived because both thresholds classified every lane the
            // same way.  A threshold with no input near it is a threshold no
            // fixture is testing.
            V3 perp{ rf(-1, 1), rf(-1, 1), rf(-1, 1) };
            const float d = dot3(perp, pn);
            perp = sub(perp, scale3(pn, d));
            const float pm = mag3(perp);
            perp = pm > 1e-5f ? scale3(perp, 1.0f / pm) : V3{ 0, 0, 0 };
            if (bucket == 3u) cn[l] = { 0, 0, 0 };           // markAsProcessed
            else cn[l] = { pn.x * std::cos(ang) + perp.x * std::sin(ang),
                           pn.y * std::cos(ang) + perp.y * std::sin(ang),
                           pn.z * std::cos(ang) + perp.z * std::sin(ang) };
            c.cn[l] = { cn[l].x, cn[l].y, cn[l].z, 0 };
            sep[l] = rf(-0.4f, 0.25f);
        }
        uint32_t candidateMask = rng();
        if ((i % 17u) == 5u) { candidateMask = 0; ++emptyCandidate; }  // the 0xFFFFFFFF return
        c.misc[0] = candidateMask;

        // ---- THREE PLANTED EXACT COINCIDENCES, ALL ASKED FOR BY THE SWEEP ----
        // Random floats never produce an exact tie, so three mutants survived the
        // first run on inputs that cannot occur by chance:
        //
        //  eps-nonstrict-existing  `>` vs `>=` differs only at proj EXACTLY
        //      0.9995f.  Constructible: make the patch normal an exact axis and
        //      set a lane's normal to (EPS3, 0, 0), so the dot product is EPS3
        //      bit-for-bit on both sides.
        //  minindex-highest-tie    firstbitlow vs firstbithigh differs only when
        //      TWO masked lanes share the minimum.
        //  minindex-ballot-unmasked  dropping `ballot &= mask` matters only when
        //      an UNMASKED lane at a LOWER index holds a value equal to the masked
        //      minimum.
        if ((i % 6u) == 1u && candidateMask != 0u)
        {
            uint32_t a = 0; while (((candidateMask >> a) & 1u) == 0u) ++a;
            uint32_t b = a + 1u;
            while (b < 32u && ((candidateMask >> b) & 1u) == 0u) ++b;
            if (b < 32u) { sep[a] = -0.39f; sep[b] = -0.39f; ++tieCases; }   // exact tie
        }
        if ((i % 6u) == 3u && candidateMask != 0u && (candidateMask & 1u) == 0u)
        {
            float mn = FLT_MAX_LOCAL;
            for (uint32_t l = 0; l < 32; ++l)
                if (((candidateMask >> l) & 1u) != 0u && sep[l] < mn) mn = sep[l];
            sep[0] = mn;                      // lane 0 is OUTSIDE the mask
            ++unmaskedTie;
        }
        if ((i % 6u) == 5u)
        {
            pn = V3{ 1.0f, 0.0f, 0.0f };      // an EXACT axis
            c.patchNormal = { 1.0f, 0.0f, 0.0f, 0 };
            const uint32_t l = 7u;
            cn[l] = { PATCH_ACCEPTANCE_EPS3, 0.0f, 0.0f };   // dot == EPS3 exactly
            c.cn[l] = { cn[l].x, cn[l].y, cn[l].z, 0 };
            ++exactEps;
        }

        for (uint32_t l = 0; l < 32; ++l) (&c.sep8[l >> 2].x)[l & 3] = sep[l];

        // The seed lane: usually a real candidate, sometimes 0xFFFFFFFF (this warp
        // did not donate the normal).  A DEGENERATE seed normal is planted in one
        // case in nine, because `|| myNormal` only matters when the seed's own
        // projection fails -- and the seed normal is contact.normal copied verbatim,
        // so proj == |normal|^2, which is below 0.9995 exactly when the normal is
        // short.  Contact normals are normalised only under `if (m > 0.0f)`.
        uint32_t seedLane = 0xFFFFFFFFu;
        if (candidateMask != 0u && (i % 3u) != 0u)
        {
            uint32_t k = rng() % 32u;
            while (((candidateMask >> k) & 1u) == 0u) k = (k + 1u) % 32u;
            seedLane = k;
            if ((i % 9u) == 2u)
            {
                const float shrink = 0.9f;                  // |n|^2 = 0.81 < 0.85
                cn[k] = scale3(pn, shrink);
                c.cn[k] = { cn[k].x, cn[k].y, cn[k].z, 0 };
            }
        }
        c.misc[1] = seedLane;

        // ---- THE ASSOCIATION, ON BOTH THRESHOLDS ----
        // The exact-epsilon lane above has ONE nonzero product, so the order of the
        // three is invisible there, and the T1 sweep's `accept-existing-dot-rl`
        // survived.  One case in eight plants a lane whose projection straddles
        // EPS3 between the two orders, and another that straddles EPS2; both lanes
        // are made candidates so the NEW form sees them too.  Because the port calls
        // dot(), these lanes also MEASURE the order this driver gives OpDot: the
        // mirror uses the CUDA's, so a reversed OpDot fails here.
        if ((i % 8u) == 6u && candidateMask != 0u)
        {
            const uint32_t l3 = (seedLane == 11u) ? 12u : 11u;
            const uint32_t l2 = (seedLane == 13u) ? 14u : 13u;
            bool above = false;
            if (plantAssociation(rng, pn, PATCH_ACCEPTANCE_EPS3, cn[l3], above))
            { ++assocEps3; if (above) ++assocCudaAbove; }
            else ++assocMissed;
            if (plantAssociation(rng, pn, PATCH_ACCEPTANCE_EPS2, cn[l2], above))
            { ++assocEps2; if (above) ++assocCudaAbove; }
            else ++assocMissed;
            c.cn[l3] = { cn[l3].x, cn[l3].y, cn[l3].z, 0 };
            c.cn[l2] = { cn[l2].x, cn[l2].y, cn[l2].z, 0 };
            candidateMask |= (1u << l3) | (1u << l2);
            c.misc[0] = candidateMask;
        }

        in[i] = c;
        refAcceptExisting(cn, pn, ra1[i], ra2[i]);
        refAcceptNew(cn, candidateMask, pn, seedLane, rb1[i], rb2[i]);
        rseed[i] = refMinIndex(sep, candidateMask, rmin[i]);

        // NOT `ra1 != rb1`: that is true on nearly every case merely because the
        // new form is gated on candidateContact, so it would be an observable true
        // under the null hypothesis.  `rb1 & ~ra1` is a lane the NEW form accepts
        // that the EXISTING form rejects, and with the same normals and the same
        // epsilons the ONLY thing that can produce one is the `|| myNormal`
        // disjunct at :925.
        if ((rb1[i] & ~ra1[i]) != 0u) ++formsDiffer;
        if ((ra2[i] & ~ra1[i]) != 0u) ++bandCases;
        for (uint32_t l = 0; l < 32; ++l)
        {
            const float pj = dot3(cn[l], pn);
            if (pj > 0.999f && pj < 0.99968f) { ++straddle; break; }
        }
        if (seedLane != 0xFFFFFFFFu && ((rb1[i] >> seedLane) & 1u) != 0u
            && dot3(cn[seedLane], pn) <= PATCH_ACCEPTANCE_EPS3) ++seedRescued;
    }

    PXG_CHECK_MSG(formsDiffer > N / 20,
                  "npTriTriReduce_accept: only %zu of %u cases have a lane the NEW "
                  "acceptance form takes and the EXISTING one does not -- that "
                  "difference is `|| myNormal` and nothing else, and without it this "
                  "fixture would pass against a port that dropped the disjunct",
                  formsDiffer, N);
    PXG_CHECK_MSG(bandCases > N / 2,
                  "npTriTriReduce_accept: only %zu of %u cases have a contact in the "
                  "(0.85, 0.9995] band -- that band is the whole point of the "
                  "accept/accept2 split (consumed but not reduced)", bandCases, N);
    PXG_CHECK_MSG(seedRescued > N / 20,
                  "npTriTriReduce_accept: only %zu of %u cases have a seed rescued by "
                  "`|| myNormal` -- without those, dropping that disjunct is invisible",
                  seedRescued, N);
    PXG_CHECK_MSG(straddle > N / 4,
                  "npTriTriReduce_accept: only %zu of %u cases have a contact whose "
                  "projection lands between 0.999 and 0.99968 -- without one, moving "
                  "PATCH_ACCEPTANCE_EPS3 off 0.9995 is invisible", straddle, N);
    PXG_CHECK_MSG(tieCases > N / 20 && unmaskedTie > N / 40 && exactEps > N / 10,
                  "npTriTriReduce_accept: planted coincidences are thin (%zu masked "
                  "ties, %zu unmasked ties, %zu exact-epsilon lanes) -- each is the "
                  "only input that distinguishes one of the minIndex/epsilon mutants",
                  tieCases, unmaskedTie, exactEps);
    PXG_CHECK_MSG(assocEps3 > N / 10 && assocEps2 > N / 10 && assocMissed == 0 &&
                  assocCudaAbove > 0 && assocCudaAbove < assocEps3 + assocEps2,
                  "npTriTriReduce_accept: %zu EPS3 and %zu EPS2 lanes straddle the threshold between "
                  "the CUDA's (x + y) + z and the reversed x + (y + z), %zu with the CUDA order above, "
                  "%zu searches failed -- without them the association of the projection is invisible",
                  assocEps3, assocEps2, assocCudaAbove, assocMissed);
    PXG_CHECK_MSG(emptyCandidate > N / 40,
                  "npTriTriReduce_accept: only %zu of %u cases have an EMPTY candidate "
                  "mask, which is the only input where minIndex returns 0xFFFFFFFF",
                  emptyCandidate, N);

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<uint32_t> outA(size_t(N) * 4, 0xDEADBEEFu);
    std::vector<uint32_t> outS(size_t(N) * 4, 0xDEADBEEFu);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bA = createBufferFrom(dev, outA);
    PxgBufferHandle bS = createBufferFrom(dev, outS);

    TtaParams prm{ N, 0, 0, 0 };
    Kernel k("npTriTriAccept.spv", "ttrAccept",
             { (uint32_t)sizeof(TtaCase), 16u, 16u }, sizeof(TtaParams));
    k.dispatch((N + 63) / 64, { bIn, bA, bS }, prm);
    dev->flush();

    const auto gotA = downloadBuffer<uint32_t>(dev, bA, size_t(N) * 4);
    const auto gotS = downloadBuffer<uint32_t>(dev, bS, size_t(N) * 4);

    size_t mmA = 0, mmSeed = 0, mmMin = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        if (gotA[i * 4 + 0] != ra1[i] || gotA[i * 4 + 1] != ra2[i] ||
            gotA[i * 4 + 2] != rb1[i] || gotA[i * 4 + 3] != rb2[i])
        {
            if (mmA < 4)
                std::printf("  case %u: got (%08x %08x %08x %08x) want (%08x %08x %08x %08x)\n",
                            i, gotA[i*4+0], gotA[i*4+1], gotA[i*4+2], gotA[i*4+3],
                            ra1[i], ra2[i], rb1[i], rb2[i]);
            ++mmA;
        }
        if (gotS[i * 4 + 0] != rseed[i])
        {
            if (mmSeed < 4)
                std::printf("  case %u: seed got 0x%08x want 0x%08x\n", i, gotS[i*4+0], rseed[i]);
            ++mmSeed;
        }
        float gm; std::memcpy(&gm, &gotS[i * 4 + 1], 4);
        if (gm != rmin[i]) ++mmMin;
    }

    PXG_CHECK_MSG(mmA == 0, "npTriTriReduce_accept: %zu acceptance-mask mismatches", mmA);
    PXG_CHECK_MSG(mmSeed == 0, "npTriTriReduce_accept: %zu seed-lane mismatches", mmSeed);
    PXG_CHECK_MSG(mmMin == 0, "npTriTriReduce_accept: %zu minSep mismatches", mmMin);

    std::printf("  npTriTriReduce_accept: %u cases, the new form takes a lane the "
                "existing one rejects on %zu, "
                "%zu have a contact in the (0.85, 0.9995] band, %zu have a seed "
                "rescued by `|| myNormal`, %zu have an empty candidate mask; "
                "planted: %zu masked ties, %zu unmasked ties, %zu exact-epsilon, "
                "%zu straddling 0.9995; association straddles %zu EPS3 / %zu EPS2 (%zu CUDA-above)\n",
                N, formsDiffer, bandCases, seedRescued, emptyCandidate,
                tieCases, unmaskedTie, exactEps, straddle, assocEps3, assocEps2, assocCudaAbove);

    for (PxgBufferHandle h : { bIn, bA, bS }) dev->destroyBuffer(h);
}

// createPatchWriterFullThreadBlock's decisions (:504-641): refReserve and the
// flag-byte helpers live in pxgTriTriRef.h since P9.83b-8 T2, where the block-tail
// fixture composes them with the writer mirror.

PXG_TEST(npTriTriReduce_reserve)
{
    std::mt19937 rng(771903);
    const uint32_t N = 3000;

    std::vector<TtaCase> in(N);
    std::vector<RefRes> ref(N);
    size_t zeroContact = 0, ovfPatch = 0, ovfContact = 0, ovfForce = 0;
    size_t forceOnlyByDoubling = 0, sentinelNotNull = 0, exactLimit = 0;
    size_t byte1NeByte3 = 0, noPatchChange = 0, ovfWithPrev = 0;
    size_t wrap32[3] = { 0, 0, 0 };

    for (uint32_t i = 0; i < N; ++i)
    {
        TtaCase c{};
        uint32_t oldAllFlags = rng();
        uint32_t nP, nC;

        const uint32_t shape = i % 8u;
        if (shape == 0u) { nP = 0; nC = 0; ++zeroContact; }          // the sentinel path
        else if (shape == 1u) { nP = 1 + rng() % 128u; nC = 0; ++zeroContact; }
        else { nP = 1 + rng() % 128u; nC = 1 + rng() % 768u; }

        // BYTE 1 IS THE PREVIOUS COUNT (:547).  A random word almost never has byte
        // 1 equal to the new count, so patchChange would be 1 everywhere and a
        // reading of any other byte would be caught only through the stored word.
        // One case in four puts byte 1 ON the new count (patchChange = 0 unless the
        // reservation overflows; half of those are kept inside the limits below),
        // one in eight puts byte 3 there instead (so a byte-3 reading gives
        // patchChange = 0 where the CUDA gives 1).
        if (i % 4u == 2u) oldAllFlags = (oldAllFlags & ~0x0000FF00u) | ((nP & 0xFFu) << 8);
        else if (i % 8u == 3u) oldAllFlags = (oldAllFlags & ~0xFF000000u) | ((nP & 0xFFu) << 24);

        uint32_t patchAdd = rng() % 400000u;
        uint32_t contactAdd = rng() % 400000u;
        uint32_t forceAdd = rng() % 400000u;
        // RESERVATIONS PAST 2^32 (P9.83b-8 T2).  The checks are 64-bit (:560, :567,
        // :573) and every draw above keeps offset + reservation far below 2^32, so a
        // 32-bit check -- the natural Slang spelling -- agreed on every case and the
        // trap the header names was never exercised.  One case in sixteen per stream
        // puts that stream's offset within one reservation of 2^32 (no extra draws,
        // so every other case is unchanged): the 64-bit sum exceeds every 32-bit
        // limit, while the wrapped 32-bit sum is small and usually passes.  Not
        // always -- a force limit placed between N and 2N wraps too (to 6N - 4k) and
        // fails both spellings -- so the guard below counts the cases where the two
        // spellings DISAGREE, not the reservations that reach past 2^32.
        if (i % 16u == 5u) patchAdd = 0u - 64u * (1u + i % 7u);
        if (i % 16u == 7u) contactAdd = 0u - 16u * (1u + i % 7u);
        if (i % 16u == 13u) forceAdd = 0u - 4u * (1u + i % 7u);

        // EVERY LIMIT IS PLACED RELATIVE TO ITS OWN RESERVATION, not at an absolute
        // address.  The first draft used absolute limits near the top of the offset
        // range and the coverage guard caught it: 27 patch and 31 contact overflows
        // in 3000 cases, because overflow then needs the random offset to land in a
        // narrow band.  Each limit now sits just below or just above `offset +
        // reservation`, so roughly half of each kind overflows by construction.
        // ONE DRAW IN FOUR LANDS EXACTLY ON `need`.  The first version always
        // offset by at least 1, so `offset + reservation == limit` never occurred
        // and `patch-check-nonstrict` (`>` vs `>=`) survived the sweep: the two
        // spellings differ on that single point and nowhere else.  The same
        // exactness now covers all three checks' strictness.
        auto straddle = [&](uint64_t need) -> uint32_t {
            const uint32_t pick = rng() % 4u;
            uint64_t lim;
            if (pick == 0u) lim = need;                         // the boundary itself
            else
            {
                const uint64_t slack = 1u + (rng() % 4096u);
                lim = (pick == 1u) ? (need + slack) : (need > slack ? need - slack : 0u);
            }
            return (uint32_t)(lim > 0xFFFFFFFFull ? 0xFFFFFFFFull : lim);
        };
        uint32_t patchLimit = straddle((uint64_t)patchAdd + (uint64_t)TTP_PATCH_SIZE * nP);
        uint32_t contactLimit = straddle((uint64_t)contactAdd + (uint64_t)TTP_CONTACT_SIZE * nC);
        // THE FORCE LIMIT ALSO STRADDLES THE DOUBLING.  A limit that clears 2N or
        // fails N alike would never distinguish the CUDA's `* 2` from its absence,
        // so one case in three is placed strictly between N and 2N.
        uint32_t forceLimit = ((i % 3u) == 0u)
            ? (uint32_t)((uint64_t)forceAdd + TTP_FORCE_SIZE * nC + (TTP_FORCE_SIZE * (uint64_t)nC) / 2u)
            : straddle((uint64_t)forceAdd + (uint64_t)TTP_FORCE_SIZE * nC * 2);
        // HALF THE BYTE-1 CASES ARE KEPT INSIDE EVERY LIMIT.  An overflow zeroes the
        // count (:580-586), so patchChange is 1 again; with three straddled checks
        // only about one case in eight fits, and the first run counted just 65
        // unchanged counts in 3000.  Raising each limit to its own need (never
        // lowering one) keeps the draws, and the boundary hits, as they were.
        if (i % 8u == 2u)
        {
            auto fit = [](uint32_t lim, uint64_t need) -> uint32_t {
                return (uint32_t)((uint64_t)lim < need ? need : (uint64_t)lim);
            };
            patchLimit = fit(patchLimit, (uint64_t)patchAdd + (uint64_t)TTP_PATCH_SIZE * nP);
            contactLimit = fit(contactLimit, (uint64_t)contactAdd + (uint64_t)TTP_CONTACT_SIZE * nC);
            forceLimit = fit(forceLimit, (uint64_t)forceAdd + (uint64_t)TTP_FORCE_SIZE * nC * 2);
        }

        // On the FINAL limits: a need past 2^32 is clamped off it, and fit() puts
        // others on it.
        {
            const uint64_t need[3] = { (uint64_t)patchAdd + (uint64_t)TTP_PATCH_SIZE * nP,
                                       (uint64_t)contactAdd + (uint64_t)TTP_CONTACT_SIZE * nC,
                                       (uint64_t)forceAdd + (uint64_t)TTP_FORCE_SIZE * nC * 2 };
            const uint32_t lim[3] = { patchLimit, contactLimit, forceLimit };
            for (uint32_t s = 0; s < 3u; ++s)
                if (need[s] <= 0xFFFFFFFFull && (uint64_t)lim[s] == need[s]) ++exactLimit;
        }

        c.cn[0] = { 0, 0, 0, 0 };
        std::memcpy(&c.cn[0].x, &oldAllFlags, 4);
        std::memcpy(&c.cn[0].y, &nP, 4);
        std::memcpy(&c.cn[0].z, &nC, 4);
        std::memcpy(&c.cn[0].w, &patchAdd, 4);
        std::memcpy(&c.cn[1].x, &contactAdd, 4);
        std::memcpy(&c.cn[1].y, &forceAdd, 4);
        std::memcpy(&c.cn[1].z, &patchLimit, 4);
        std::memcpy(&c.cn[1].w, &contactLimit, 4);
        std::memcpy(&c.cn[2].x, &forceLimit, 4);

        in[i] = c;
        ref[i] = refReserve(oldAllFlags, nP, nC, patchAdd, contactAdd, forceAdd,
                            patchLimit, contactLimit, forceLimit);

        if (ref[i].ovfBits & OVF_PATCH) ++ovfPatch;
        if (ref[i].ovfBits & OVF_CONTACT) ++ovfContact;
        if (ref[i].ovfBits & OVF_FORCE) ++ovfForce;
        // Cases where the force check fails ONLY because of the *2.
        if (nC != 0 &&
            (uint64_t)forceAdd + (uint64_t)TTP_FORCE_SIZE * nC * 2 > (uint64_t)forceLimit &&
            (uint64_t)forceAdd + (uint64_t)TTP_FORCE_SIZE * nC <= (uint64_t)forceLimit)
            ++forceOnlyByDoubling;
        // The zero-contact path: sentinels held, but NOT flagged as overflow --
        // which is what makes the CUDA publish base + 0xFFFFFFFF.
        if (nC == 0 && !ref[i].overflow && ref[i].patchOfs == 0xFFFFFFFFu) ++sentinelNotNull;
        if (((oldAllFlags >> 8) & 0xFFu) != ((oldAllFlags >> 24) & 0xFFu)) ++byte1NeByte3;
        if (nC != 0)
        {
            // The 64-bit check against its wrapped 32-bit respelling: count only where
            // they disagree.
            auto disagree = [](uint32_t add, uint64_t res, uint32_t lim) {
                return ((uint64_t)add + res > (uint64_t)lim) != ((uint32_t)(add + (uint32_t)res) > lim);
            };
            if (disagree(patchAdd, uint64_t(TTP_PATCH_SIZE) * nP, patchLimit)) ++wrap32[0];
            if (disagree(contactAdd, uint64_t(TTP_CONTACT_SIZE) * nC, contactLimit)) ++wrap32[1];
            if (disagree(forceAdd, uint64_t(TTP_FORCE_SIZE) * nC * 2, forceLimit)) ++wrap32[2];
        }
        if (!ref[i].patchChange) ++noPatchChange;
        if (ref[i].overflow && ((oldAllFlags >> 8) & 0xFFu) != 0u) ++ovfWithPrev;
    }
    PXG_CHECK_MSG(byte1NeByte3 > N / 2 && noPatchChange > N / 10 && ovfWithPrev > N / 20,
                  "npTriTriReduce_reserve: the prevPatches byte is under-covered (%zu cases with byte 1 != "
                  "byte 3, %zu with no patch change, %zu overflows with a previous count)",
                  byte1NeByte3, noPatchChange, ovfWithPrev);

    PXG_CHECK_MSG(wrap32[0] > N / 40 && wrap32[1] > N / 40 && wrap32[2] > N / 40,
                  "npTriTriReduce_reserve: only %zu / %zu / %zu cases where a wrapped 32-bit check "
                  "disagrees with the CUDA's 64-bit one -- without them the respelling is invisible",
                  wrap32[0], wrap32[1], wrap32[2]);
    PXG_CHECK_MSG(ovfPatch > N / 20 && ovfContact > N / 20 && ovfForce > N / 20,
                  "npTriTriReduce_reserve: overflow coverage is thin (%zu patch, %zu "
                  "contact, %zu force) -- a limit always above the offset makes the "
                  "checks unobservable", ovfPatch, ovfContact, ovfForce);
    PXG_CHECK_MSG(forceOnlyByDoubling > N / 40,
                  "npTriTriReduce_reserve: only %zu of %u cases fail the force check "
                  "SOLELY because of the `* 2` at :573 -- without those, dropping the "
                  "doubling is invisible", forceOnlyByDoubling, N);
    PXG_CHECK_MSG(exactLimit > N / 10,
                  "npTriTriReduce_reserve: only %zu of %u limits land EXACTLY on "
                  "`offset + reservation` -- that single point is the only place "
                  "`>` and `>=` differ in all three overflow checks", exactLimit, N);
    PXG_CHECK_MSG(sentinelNotNull > N / 20,
                  "npTriTriReduce_reserve: only %zu of %u cases take the zero-contact "
                  "path that leaves the sentinel WITHOUT setting overflow -- that "
                  "combination is the base+0xFFFFFFFF publish", sentinelNotNull, N);

    PxgSlangComputeDevice* dev = testDevice();
    std::vector<uint32_t> outA(size_t(N) * 4, 0xDEADBEEFu);
    std::vector<uint32_t> outS(size_t(N) * 4, 0xDEADBEEFu);
    PxgBufferHandle bIn = createBufferFrom(dev, in);
    PxgBufferHandle bA = createBufferFrom(dev, outA);
    PxgBufferHandle bS = createBufferFrom(dev, outS);

    TtaParams prm{ N, 0, 0, 0 };
    Kernel k("npTriTriAccept.spv", "ttrReserve",
             { (uint32_t)sizeof(TtaCase), 16u, 16u }, sizeof(TtaParams));
    k.dispatch((N + 63) / 64, { bIn, bA, bS }, prm);
    dev->flush();

    const auto gA = downloadBuffer<uint32_t>(dev, bA, size_t(N) * 4);
    const auto gS = downloadBuffer<uint32_t>(dev, bS, size_t(N) * 4);

    size_t mm = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const RefRes& w = ref[i];
        const uint32_t counts = w.nPatches | (w.nContacts << 16);
        const uint32_t bits = (w.touchChange ? 1u : 0u) | (w.patchChange ? 2u : 0u)
                            | (w.overflow ? 4u : 0u) | (w.wPatchNull ? 8u : 0u)
                            | (w.wContactNull ? 16u : 0u);
        if (gA[i*4+0] != w.patchOfs || gA[i*4+1] != w.contactOfs ||
            gA[i*4+2] != w.forceOfs || gA[i*4+3] != w.allFlags ||
            gS[i*4+0] != counts || gS[i*4+1] != bits ||
            gS[i*4+2] != w.ovfBits || gS[i*4+3] != w.statusFlags)
        {
            if (mm < 4)
                std::printf("  case %u: got (%08x %08x %08x %08x | %08x %08x %08x %08x) "
                            "want (%08x %08x %08x %08x | %08x %08x %08x %08x)\n", i,
                            gA[i*4+0], gA[i*4+1], gA[i*4+2], gA[i*4+3],
                            gS[i*4+0], gS[i*4+1], gS[i*4+2], gS[i*4+3],
                            w.patchOfs, w.contactOfs, w.forceOfs, w.allFlags,
                            counts, bits, w.ovfBits, w.statusFlags);
            ++mm;
        }
    }
    PXG_CHECK_MSG(mm == 0, "npTriTriReduce_reserve: %zu reservation mismatches", mm);

    std::printf("  npTriTriReduce_reserve: %u cases; %zu zero-contact, %zu patch / "
                "%zu contact / %zu force overflows, %zu fail the force check only "
                "because of the *2, %zu publish base+0xFFFFFFFF, %zu limits sit "
                "exactly on the boundary; prevPatches: %zu byte1!=byte3, %zu without a patch change, "
                "%zu overflows with a previous count; 32-bit respelling told apart %zu/%zu/%zu\n",
                N, zeroContact, ovfPatch, ovfContact, ovfForce,
                forceOnlyByDoubling, sentinelNotNull, exactLimit, byte1NeByte3, noPatchChange, ovfWithPrev,
                wrap32[0], wrap32[1], wrap32[2]);

    for (PxgBufferHandle h : { bIn, bA, bS }) dev->destroyBuffer(h);
}

} // namespace
