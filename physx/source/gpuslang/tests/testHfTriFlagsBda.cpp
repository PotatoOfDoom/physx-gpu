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

// Verification of pxgSlangHfTriFlags -- computeTriangleFlags
// (convexHeightfield.cu:565-614) reading the sample grid, and selectNormal
// (capsuleTriangle.cuh:38-98), which is its only consumer.  P9.52b-1.
//
// The mirror is written from those two functions plus getTriangle
// (convexHeightfield.cu:83-116).  It drives the WHOLE path -- fetch, normal,
// classify, select -- because a fixture that fed the classifier a pre-decoded
// triangle would let the fetch and the classifier agree about a triangle neither
// obtained correctly.
//
// ===========================================================================
// THE FOUR EDGE OUTCOMES, AND HOW A CASE REACHES EACH
// ===========================================================================
//
//   BOUNDARY      adjacency == 0xFFFFFFFF, the edge is skipped entirely and
//                 contributes no bit.  Rim triangles.
//   ACTIVE-FOLD   projD < 0 (a convex ridge) AND proj < 0.997 (steep enough).
//                 The steep ridge below, whose height step is 16 units at
//                 heightScale 1/32 -- a slope of 0.5.
//   FOLD-IGNORED  projD < 0 but proj >= 0.997.  The shallow ridge, height step 1
//                 at the same scale -- a slope of 1/32, cos about 0.9995.
//   ACTIVE-BOUND  projD >= 0, which reaches the `boundaryCollisions` arm and marks
//                 the edge active.
//
// NOTE WHAT THE LAST ONE MEANS ON FLAT TERRAIN.  Two coplanar triangles give
// projD EXACTLY zero, and `0.0f < 0.0f` is false -- so on a perfectly flat
// heightfield EVERY interior edge is marked active, and `selectNormal` then
// refuses the face normal for every vertex and edge feature.  That is upstream's
// behaviour, not a port artefact, and the flat block below is here to pin it.
//
// ===========================================================================
// WHAT CANNOT BE REACHED, AND IS SAID RATHER THAN HIDDEN
// ===========================================================================
//
// The silhouette bits (0..2) are unreachable twice over: their branch is `else`
// to `boundaryCollisions`, which is a constant `true` (upstream candidate 48), and
// `selectNormal` reads only bits 3..5 anyway.  No case here sets them and none
// can; the module says so at the site.
//
// The `m2 > 0` guard before the normalize is likewise unreachable-when-false: it
// sits inside `projD < 0`, and a zero adjNormal would make projD zero.

#include "pxgTestFramework.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;

struct V3 { float x, y, z; };

static V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross3(V3 a, V3 b)
{ return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
// PxVec3::getNormalized (PxVec3.h:286)
static V3 normalized(V3 v)
{
    const float m = dot3(v, v);
    if (!(m > 0.0f)) return V3{ 0, 0, 0 };
    const float r = 1.0f / std::sqrt(m);
    return V3{ v.x * r, v.y * r, v.z * r };
}

static const uint32_t HF_BOUNDARY = 0xFFFFFFFFu;
static const uint32_t ETD_CONVEX_EDGE_01 = 1u << 3;
static const uint32_t ETD_CONVEX_EDGE_12 = 1u << 4;
static const uint32_t ETD_CONVEX_EDGE_20 = 1u << 5;

// ---------------------------------------------------------------------------
// the heightfield blob and its decoders (heightfieldUtil.cuh)
// ---------------------------------------------------------------------------
struct Hf
{
    uint32_t rows = 0, cols = 0;
    std::vector<uint32_t> samples;
    std::vector<uint32_t> blob;
};

static uint32_t sampleWord(int16_t height, uint8_t mat0, bool tessellationFlag, uint8_t mat1)
{
    return uint32_t(uint16_t(height))
         | (uint32_t(mat0 & 0x7f) << 16)
         | (tessellationFlag ? (1u << 23) : 0u)
         | (uint32_t(mat1 & 0x7f) << 24);
}

static void buildHf(Hf& h)
{
    h.blob.clear();
    h.blob.push_back(h.rows);
    h.blob.push_back(h.cols);
    for (uint32_t s : h.samples) h.blob.push_back(s);
}

static bool zerothVertexShared(const Hf& h, uint32_t cell)
{ return (h.samples[cell] & 0x800000u) != 0u; }

static float hfHeight(const Hf& h, uint32_t vi)
{ return float(int16_t(uint16_t(h.samples[vi] & 0xffffu))); }

static V3 hfGetVertex(const Hf& h, uint32_t vi)
{ return V3{ float(vi / h.cols), hfHeight(h, vi), float(vi % h.cols) }; }

static void triVertexIndices(const Hf& h, uint32_t t, uint32_t& i0, uint32_t& i1, uint32_t& i2)
{
    const uint32_t cell = t >> 1;
    const bool firstTri = (t & 1u) == 0u;
    const uint32_t c = h.cols;
    if (zerothVertexShared(h, cell))
    {
        if (firstTri) { i0 = cell + c; i1 = cell; i2 = cell + c + 1u; }
        else          { i0 = cell + 1u; i1 = cell + c + 1u; i2 = cell; }
    }
    else
    {
        if (firstTri) { i0 = cell; i1 = cell + 1u; i2 = cell + c; }
        else          { i0 = cell + c + 1u; i1 = cell + c; i2 = cell + 1u; }
    }
}

static void triAdjacency(const Hf& h, uint32_t t, uint32_t& a0, uint32_t& a1, uint32_t& a2)
{
    const uint32_t cell = t >> 1;
    const bool firstTri = (t & 1u) == 0u;
    const uint32_t c = h.cols, r = h.rows;
    a0 = HF_BOUNDARY;
    a2 = HF_BOUNDARY;
    if (zerothVertexShared(h, cell))
    {
        if (firstTri)
        {
            a1 = t + 1u;
            if (cell % c != 0u)        a0 = t - 1u;
            if (cell / c != r - 2u)    a2 = ((cell + c) * 2u) + 1u;
        }
        else
        {
            a1 = t - 1u;
            if (cell % c < c - 2u)     a0 = t + 1u;
            if (cell >= c - 1u)        a2 = (cell - c) * 2u;
        }
    }
    else
    {
        if (firstTri)
        {
            a1 = t + 1u;
            if (cell >= c - 1u)        a0 = ((cell - c) * 2u) + 1u;
            if (cell % c != 0u)        a2 = t - 1u;
        }
        else
        {
            a1 = t - 1u;
            if (cell / c != r - 2u)    a0 = (cell + c) * 2u;
            if (cell % c < c - 2u)     a2 = t + 1u;
        }
    }
}

// getTriangle (convexHeightfield.cu:83-116): the vertex permutation swaps slots
// 1/2, the adjacency permutation swaps slots 0/2 -- two different permutations.
static void getTriangle(const Hf& h, uint32_t t, V3 scale,
                        V3& v0, V3& v1, V3& v2, uint32_t adj[3])
{
    bool wrongHanded = false;
    if (scale.z < 0.0f) wrongHanded = !wrongHanded;
    if (scale.x < 0.0f) wrongHanded = !wrongHanded;

    uint32_t i0, i1, i2;
    triVertexIndices(h, t, i0, i1, i2);
    uint32_t a0, a1, a2;
    triAdjacency(h, t, a0, a1, a2);
    adj[0] = wrongHanded ? a2 : a0;
    adj[1] = a1;
    adj[2] = wrongHanded ? a0 : a2;

    const uint32_t tv[3] = { i0, wrongHanded ? i2 : i1, wrongHanded ? i1 : i2 };
    V3* out[3] = { &v0, &v1, &v2 };
    for (uint32_t k = 0; k < 3; ++k)
    {
        const V3 g = hfGetVertex(h, tv[k]);
        *out[k] = V3{ g.x * scale.x, g.y * scale.y, g.z * scale.z };
    }
}

// ---------------------------------------------------------------------------
// computeTriangleFlags (convexHeightfield.cu:565-614), with boundaryCollisions
// nailed to true -- see the module header (upstream candidate 48).
// ---------------------------------------------------------------------------
enum EdgeOutcome { EO_BOUNDARY = 0, EO_ACTIVE_FOLD, EO_FOLD_IGNORED, EO_ACTIVE_BOUND };

static uint32_t computeTriangleFlags(const Hf& h, const uint32_t adj[3],
                                     V3 cur0, V3 cur1, V3 cur2, V3 triNormal, V3 scale,
                                     EdgeOutcome outcome[3])
{
    const uint32_t nextInd[3] = { 2u, 0u, 1u };
    uint32_t triFlags = 0u;

    for (uint32_t a = 0; a < 3; ++a)
    {
        if (adj[a] == HF_BOUNDARY) { outcome[a] = EO_BOUNDARY; continue; }

        V3 av0, av1, av2;
        uint32_t ignored[3];
        getTriangle(h, adj[a], scale, av0, av1, av2, ignored);

        V3 adjNormal = cross3(sub(av1, av0), sub(av2, av0));   // denormalized
        const uint32_t otherIndex = nextInd[a];
        const V3 other = (otherIndex == 0u) ? cur0 : ((otherIndex == 1u) ? cur1 : cur2);
        const float projD = dot3(adjNormal, sub(other, av0));

        if (projD < 0.0f)
        {
            // PxVec3::normalize -- reciprocal then multiply, zero left alone.
            const float m2 = dot3(adjNormal, adjNormal);
            if (m2 > 0.0f)
            {
                const float r = 1.0f / std::sqrt(m2);
                adjNormal = V3{ adjNormal.x * r, adjNormal.y * r, adjNormal.z * r };
            }
            const float proj = dot3(adjNormal, triNormal);
            if (proj < 0.997f) { triFlags |= (1u << (a + 3u)); outcome[a] = EO_ACTIVE_FOLD; }
            else               { outcome[a] = EO_FOLD_IGNORED; }
        }
        else
        {
            // boundaryCollisions is always true, so the silhouette arm is dead.
            triFlags |= (1u << (a + 3u));
            outcome[a] = EO_ACTIVE_BOUND;
        }
    }
    return triFlags;
}

// selectNormal (capsuleTriangle.cuh:38-98).  `region` records which arm was taken
// so the fixture can require every one of them.
enum SnRegion { SN_V0 = 0, SN_V2, SN_E02, SN_V1, SN_U_HIGH_FALLTHROUGH,
                SN_E01, SN_E12, SN_FACE };

static bool selectNormal(float u, float v, uint32_t triFlags, SnRegion& region)
{
    const float zero = 1e-6f;
    const float one = 0.999999f;

    if (zero > u)
    {
        if (zero > v)
        {
            region = SN_V0;
            if ((triFlags & (ETD_CONVEX_EDGE_01 | ETD_CONVEX_EDGE_20)) == 0u) return true;
        }
        else if (v > one)
        {
            region = SN_V2;
            if ((triFlags & (ETD_CONVEX_EDGE_12 | ETD_CONVEX_EDGE_20)) == 0u) return true;
        }
        else
        {
            region = SN_E02;
            if ((triFlags & ETD_CONVEX_EDGE_20) == 0u) return true;
        }
    }
    else if (u > one)
    {
        if (zero > v)
        {
            region = SN_V1;
            if ((triFlags & (ETD_CONVEX_EDGE_01 | ETD_CONVEX_EDGE_12)) == 0u) return true;
        }
        else
        {
            // NO `else` UPSTREAM: this arm just falls out of the cascade.
            region = SN_U_HIGH_FALLTHROUGH;
        }
    }
    else
    {
        if (zero > v)
        {
            region = SN_E01;
            if ((triFlags & ETD_CONVEX_EDGE_01) == 0u) return true;
        }
        else
        {
            const float threshold = 0.9999f;
            if ((u + v) >= threshold)
            {
                region = SN_E12;
                if ((triFlags & ETD_CONVEX_EDGE_12) == 0u) return true;
            }
            else
            {
                region = SN_FACE;
                return true;
            }
        }
    }
    return false;
}

struct HtfPush
{
    uint64_t samples, cases, output;
    uint32_t numCases, rows, columns, pad;
};

struct HtfCase
{
    float sx, sy, sz, sw;
    float u, v, uw0, uw1;
    uint32_t tri, c1, c2, c3;
};
static_assert(sizeof(HtfCase) == 48, "HfTriFlagsCase mirror");

PXG_TEST(hfTriFlagsBda)
{
    PxgSlangComputeDevice* dev = testDevice();

    // 6 rows x 5 columns.  Heights are integers and the height scale is a power of
    // two, so every vertex coordinate is exact.
    //
    //   CELL 0 carries a SHALLOW RIDGE along its own diagonal: vertices 1 and 5 are
    //   raised by one unit, and for an unflagged cell the two triangles are
    //   (0,1,5) and (6,5,1), so 1-5 IS the shared edge.  A one-unit step at
    //   heightScale 1/32 is a slope of 1/32 and cos about 0.9995, which is >= 0.997
    //   -- the fold is real but too shallow to mark.  Cell 0 is also a corner, so
    //   its other two adjacencies are BOUNDARY: it is the one triangle in the grid
    //   that comes out with NO active edge, and the fixture requires it, because
    //   otherwise `selectNormal` is never asked a question it can answer `true` to
    //   on a vertex or edge feature.
    //   ROW 3 is a STEEP ridge (step 16 -> slope 0.5, cos about 0.894 < 0.997) and
    //   ROW 4 the matching VALLEY, whose folds reach the projD >= 0 arm.
    //   EVERYWHERE ELSE is flat, where projD is EXACTLY zero and `0 < 0` is false,
    //   so every one of those interior edges is marked ACTIVE.
    Hf hf;
    hf.rows = 6; hf.cols = 5;
    {
        const int16_t hts[30] = {  0,  1,  0,  0,  0,
                                   1,  0,  0,  0,  0,
                                   0,  0,  0,  0,  0,
                                   0, 16, 16,  0,  0,
                                   0,-16,-16,  0,  0,
                                   0,  0,  0,  0,  0 };
        for (uint32_t i = 0; i < 30; ++i)
            hf.samples.push_back(sampleWord(hts[i], uint8_t(1 + (i % 5)), (i % 4u) == 1u,
                                            uint8_t(2 + (i % 7))));
    }
    buildHf(hf);

    PxgBufferHandle bHf = createBufferFrom(dev, hf.blob);
    const uint64_t addrHf = dev->getBufferDeviceAddress(bHf);

    // THREE DIFFERENT SCALE COMPONENTS so a hf2shapep that puts the wrong factor on
    // the wrong axis is visible, and a NEGATIVE column scale so the handedness
    // permutation runs (the classification indexes the PERMUTED adjacency, so a
    // port that permuted one and not the other lands on the wrong neighbour).
    const V3 scaleP = { 1.0f, 0.03125f, 1.5f };
    const V3 scaleN = { 1.0f, 0.03125f, -1.5f };

    // Every triangle of the grid under both scales, and a spread of barycentric
    // pairs: the seven regions selectNormal distinguishes plus its fall-through.
    // The eight arms, plus five pairs that sit ON the constants rather than far
    // from them.  The first sweep of this module killed 24 of 35 mutants and left
    // eleven; five of those were this list being too coarse -- with u and v only
    // ever 0, 0.25, 0.5 or 1, the epsilon could be moved from 1e-6 to 1e-1 and
    // `one` from 0.999999 to 0.99 without changing a single classification.
    struct UV { float u, v; const char* why; };
    const std::vector<UV> uvs = {
        { 0.0f,       0.0f,      "vertex 0" },
        { 0.0f,       1.0f,      "vertex 2" },
        { 0.0f,       0.5f,      "edge 0-2" },
        { 1.0f,       0.0f,      "vertex 1" },
        { 1.0f,       0.5f,      "u > one with v >= zero -- the arm with no else" },
        { 0.5f,       0.0f,      "edge 0-1" },
        { 0.5f,       0.5f,      "edge 1-2 (u+v >= 0.9999)" },
        { 0.25f,      0.25f,     "face" },
        // u strictly between 1e-6 and 1e-1: the only kind of input that tells the
        // epsilon apart from any larger one.
        { 0.01f,      0.0f,      "u just above the epsilon" },
        // u strictly between 0.99 and 0.999999: the same for `one`.
        { 0.9995f,    0.0f,      "u just below one" },
        // u EXACTLY at the epsilon, where `zero > u` and `zero >= u` disagree.
        { 1e-6f,      0.0f,      "u exactly at the epsilon" },
        // u + v strictly between 0.9 and 0.9999, which pins the edge-1-2 threshold
        // against any coarser value.
        { 0.48f,      0.47f,     "u+v between 0.9 and the threshold" },
        // u + v EXACTLY 0.9999f, where `>=` and `>` disagree.  Exact by
        // construction: 0.9999f and 0.9999f - 0.25f share a binade, so the
        // subtraction is exact and adding 0.25f back reproduces it bit for bit.
        { 0.9999f - 0.25f, 0.25f, "u+v exactly at the threshold" },
    };

    // THE FULL CROSS PRODUCT, and deliberately so.  Pairing each triangle with one
    // rotating (u, v) left the two vertex arms untested against the flag patterns
    // that distinguish their masks -- `B4-v0-mask-uses-edge-12` and
    // `B7-v1-mask-uses-edge-20` both survived the first sweep, because the V0 and
    // V1 queries never landed on a triangle whose edge-12 and edge-20 bits differ.
    // Every triangle against every pair costs a thousand cases and settles it.
    std::vector<HtfCase> cases;
    const uint32_t nTri = (hf.rows - 1u) * (hf.cols - 1u) * 2u;
    for (uint32_t pass = 0; pass < 2; ++pass)
    {
        const V3 sc = (pass == 0) ? scaleP : scaleN;
        for (uint32_t t = 0; t < nTri; ++t)
        {
            for (const UV& uv : uvs)
            {
                HtfCase c{};
                c.sx = sc.x; c.sy = sc.y; c.sz = sc.z;
                c.u = uv.u; c.v = uv.v;
                c.tri = t;
                cases.push_back(c);
            }
        }
    }

    const uint32_t n = uint32_t(cases.size());

    PxgBufferHandle bCases = createBufferFrom(dev, cases);
    std::vector<uint32_t> outInit(size_t(n) * 8, 0xdeadbeefu);
    PxgBufferHandle bOut = createBufferFrom(dev, outInit);

    HtfPush push{};
    push.samples = addrHf + 8ull;       // the samples begin after rows and columns
    push.cases = dev->getBufferDeviceAddress(bCases);
    push.output = dev->getBufferDeviceAddress(bOut);
    push.numCases = n;
    push.rows = hf.rows;
    push.columns = hf.cols;

    {
        Kernel k("hfTriFlagsBda.spv", "hfTriFlagsLaunch", {}, sizeof(HtfPush));
        k.dispatch((n + 63u) / 64u, {}, push, 1u, 1u);
        dev->flush();
    }

    const auto got = downloadBuffer<uint32_t>(dev, bOut, outInit.size());

    size_t bad = 0;
    size_t sawBoundary = 0, sawActiveFold = 0, sawFoldIgnored = 0, sawActiveBound = 0;
    size_t sawWrongHanded = 0, sawFlagsMixed = 0, sawFlagsZero = 0;
    size_t sawRegion[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    size_t sawSelTrue = 0, sawSelFalse = 0;
    size_t sawV0MaskSplit = 0, sawV1MaskSplit = 0;

    for (uint32_t i = 0; i < n; ++i)
    {
        const HtfCase& c = cases[i];
        const V3 sc = { c.sx, c.sy, c.sz };

        V3 v0, v1, v2; uint32_t adj[3];
        getTriangle(hf, c.tri, sc, v0, v1, v2, adj);
        const V3 triNormal = normalized(cross3(sub(v1, v0), sub(v2, v0)));

        EdgeOutcome outcome[3] = { EO_BOUNDARY, EO_BOUNDARY, EO_BOUNDARY };
        const uint32_t flags = computeTriangleFlags(hf, adj, v0, v1, v2, triNormal, sc, outcome);
        SnRegion region = SN_FACE;
        const bool sel = selectNormal(c.u, c.v, flags, region);

        const uint32_t* r = got.data() + size_t(i) * 8;
        auto chk = [&](const char* w, uint32_t g, uint32_t e) {
            if (g != e) { ++bad; if (bad <= 20)
                std::printf("  [case %u tri %u scaleZ %.4g uv (%.4g, %.4g)] %s: got=0x%x exp=0x%x\n",
                            i, c.tri, double(c.sz), double(c.u), double(c.v), w, g, e); } };
        chk("triFlags", r[0], flags);
        chk("selectNormal", r[1], sel ? 1u : 0u);
        chk("triangleIdx", r[2], c.tri);
        chk("adj.x", r[4], adj[0]);
        chk("adj.y", r[5], adj[1]);
        chk("adj.z", r[6], adj[2]);

        for (uint32_t a = 0; a < 3; ++a)
        {
            if (outcome[a] == EO_BOUNDARY) ++sawBoundary;
            else if (outcome[a] == EO_ACTIVE_FOLD) ++sawActiveFold;
            else if (outcome[a] == EO_FOLD_IGNORED) ++sawFoldIgnored;
            else ++sawActiveBound;
        }
        if (sc.z < 0.0f || sc.x < 0.0f) ++sawWrongHanded;
        if (flags == 0u) ++sawFlagsZero;
        else if (flags != (ETD_CONVEX_EDGE_01 | ETD_CONVEX_EDGE_12 | ETD_CONVEX_EDGE_20))
            ++sawFlagsMixed;
        ++sawRegion[region];
        {
            // The V0 arm masks (01 | 20) and the V1 arm (01 | 12).  A port that
            // swapped 12 for 20 in either is invisible unless those two bits
            // DIFFER on a triangle one of those arms is asked about.
            const bool b12 = (flags & ETD_CONVEX_EDGE_12) != 0u;
            const bool b20 = (flags & ETD_CONVEX_EDGE_20) != 0u;
            if (b12 != b20 && region == SN_V0) ++sawV0MaskSplit;
            if (b12 != b20 && region == SN_V1) ++sawV1MaskSplit;
        }
        if (sel) ++sawSelTrue; else ++sawSelFalse;
        // The silhouette bits cannot be set; if one ever is, the mirror and the
        // module have both changed and the claim in the headers is stale.
        if ((flags & 0x7u) != 0u)
        { ++bad; std::printf("  [case %u] silhouette bit set -- see the header\n", i); }
    }

    dev->destroyBuffer(bHf); dev->destroyBuffer(bCases); dev->destroyBuffer(bOut);

    PXG_CHECK_MSG(bad == 0, "hfTriFlagsBda: %zu mismatches", bad);

    PXG_CHECK_MSG(sawBoundary > 0, "no edge has a BOUNDARY adjacency");
    PXG_CHECK_MSG(sawActiveFold > 0,
                  "no edge is a convex fold steep enough to mark active (proj < 0.997)");
    PXG_CHECK_MSG(sawFoldIgnored > 0,
                  "no edge is a convex fold too shallow to mark (proj >= 0.997) -- the 0.997 "
                  "threshold is then unpinned");
    PXG_CHECK_MSG(sawActiveBound > 0,
                  "no edge reaches the projD >= 0 arm -- on flat terrain that is EVERY interior "
                  "edge, so a fixture without it is not looking at flat ground");
    PXG_CHECK_MSG(sawWrongHanded > 0,
                  "no case has a negative scale -- the classification indexes the PERMUTED "
                  "adjacency, so a half-permuted port would land on the wrong neighbour");
    PXG_CHECK_MSG(sawFlagsZero > 0, "no triangle comes out with NO active edge");
    PXG_CHECK_MSG(sawFlagsMixed > 0,
                  "no triangle comes out with a MIXTURE of active and inactive edges -- with "
                  "all-or-nothing flags, which bit belongs to which edge is untested");
    PXG_CHECK_MSG(sawSelTrue > 0 && sawSelFalse > 0,
                  "selectNormal never answered both ways");
    PXG_CHECK_MSG(sawV0MaskSplit > 0,
                  "the vertex-0 arm is never asked about a triangle whose edge-12 and edge-20 "
                  "bits differ -- its mask is then untested");
    PXG_CHECK_MSG(sawV1MaskSplit > 0,
                  "the vertex-1 arm is never asked about a triangle whose edge-12 and edge-20 "
                  "bits differ -- its mask is then untested");
    static const char* regionName[8] = { "vertex 0", "vertex 2", "edge 0-2", "vertex 1",
                                         "u > one fall-through", "edge 0-1", "edge 1-2",
                                         "face" };
    for (uint32_t k = 0; k < 8; ++k)
        PXG_CHECK_MSG(sawRegion[k] > 0, "selectNormal never reached its %s arm", regionName[k]);
}

} // namespace
