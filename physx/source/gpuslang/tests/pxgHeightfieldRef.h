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

// pxgHeightfieldRef.h -- the heightfield BLOB, its triangle decoders, the triangle
// EDGE-FLAG classifier and `selectNormal`, transcribed from the ENGINE.
//
// Sources, cited per function below: heightfieldUtil.cuh for the blob layout, the
// vertex indices and the adjacency; convexHeightfield.cu:83-116 for `getTriangle`
// and its wrongHanded winding swap; convexHeightfield.cu:565-614 for
// `computeTriangleFlags`; capsuleTriangle.cuh:38-98 for `selectNormal`.
//
// WHY IT IS A HEADER (P9.52b-5).  Three fixtures carried a copy -- testSphereHfCore,
// testHfTriFlagsBda and testCapsuleHfCore.  All three were diffed function by
// function before the move; the only differences were formatting, identifier names
// and the trace out-parameters below.  The bodies are moved VERBATIM.
//
// THE TRACES ARE THE POINT OF TWO OF THESE SIGNATURES.  `computeTriangleFlags` and
// `selectNormal` take an out-parameter naming which arm ran, because the probe
// fixtures assert COVERAGE of the classifier, not merely its answer -- and in the
// engine both are nearly unobservable from the outside: `triFlags` reaches an
// output only through `selectNormal`, and `selectNormal`'s answer only when the
// segment/triangle distance is exactly zero, since its other two callers both
// answer `triNormal` regardless.  Callers that do not want the trace use the
// overloads at the bottom, which pass a local sink.

#ifndef PXG_HEIGHTFIELD_REF_H
#define PXG_HEIGHTFIELD_REF_H

#include "pxgNpRefMath.h"

#include <cstdint>
#include <vector>

namespace pxgtest
{
namespace hfref
{
using npref::V3;
using npref::sub;
using npref::dot3;
using npref::cross3;
using npref::normalized;

static const uint32_t HF_BOUNDARY = 0xFFFFFFFFu;
// triangle.cuh: bit 31 of an adjacency word is set by the cooker when it found the
// shared edge non-convex.  Only the mesh path reads it; the heightfield decoders
// above never set it, which is exactly why isEdgeNonconvex is worth having here --
// it documents that the two adjacency encodings share a word and not a meaning.
static const uint32_t NONCONVEX_FLAG = 0x80000000u;
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

inline uint32_t sampleWord(int16_t height, uint8_t mat0, bool tessellationFlag, uint8_t mat1)
{
    return uint32_t(uint16_t(height))
         | (uint32_t(mat0 & 0x7f) << 16)
         | (tessellationFlag ? (1u << 23) : 0u)
         | (uint32_t(mat1 & 0x7f) << 24);
}

inline void buildHf(Hf& h)
{
    h.blob.clear();
    h.blob.push_back(h.rows);
    h.blob.push_back(h.cols);
    for (uint32_t s : h.samples) h.blob.push_back(s);
}

inline bool zerothVertexShared(const Hf& h, uint32_t cell)
{ return (h.samples[cell] & 0x800000u) != 0u; }

inline float hfHeight(const Hf& h, uint32_t vi)
{ return float(int16_t(uint16_t(h.samples[vi] & 0xffffu))); }

inline V3 hfGetVertex(const Hf& h, uint32_t vi)
{ return V3{ float(vi / h.cols), hfHeight(h, vi), float(vi % h.cols) }; }

inline void triVertexIndices(const Hf& h, uint32_t t, uint32_t& i0, uint32_t& i1, uint32_t& i2)
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

inline void triAdjacency(const Hf& h, uint32_t t, uint32_t& a0, uint32_t& a1, uint32_t& a2)
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
inline void getTriangle(const Hf& h, uint32_t t, V3 scale,
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

// The three-vertex overload the switch calls with a NULL adjacency (:470).
inline void getTriangleNoAdj(const Hf& h, uint32_t t, V3 scale, V3& v0, V3& v1, V3& v2)
{
    uint32_t ignored[3];
    getTriangle(h, t, scale, v0, v1, v2, ignored);
}

// ---------------------------------------------------------------------------
// distancePointTriangleSquared (sphereTriangle.cuh:41-140)
//
// STRICT comparisons in the three vertex regions and in `vc < 0`, the branch order
// V0, V1, V2 then E01, E02, E12, and a face region that SUMS the two scaled edge
// vectors and THEN divides -- PxVec3::operator/ being one reciprocal and three
// multiplies (PxVec3.h:218).  `edgeTol` is computed in DOUBLE because `1e-5` is an
// unsuffixed double literal (P9.55); it is dead on this path, because heightfield
// adjacency indices never carry bit 31 except as the BOUNDARY sentinel, which
// isEdgeNonconvex excludes -- so the `||` short-circuits before it is reached.
// ---------------------------------------------------------------------------
inline bool isEdgeNonconvex(uint32_t i)
{ return (i != HF_BOUNDARY) && ((i & NONCONVEX_FLAG) != 0u); }



// ---------------------------------------------------------------------------
// computeTriangleFlags (convexHeightfield.cu:565-614), with boundaryCollisions
// nailed to true -- see the module header (upstream candidate 48).
// ---------------------------------------------------------------------------
enum EdgeOutcome { EO_BOUNDARY = 0, EO_ACTIVE_FOLD, EO_FOLD_IGNORED, EO_ACTIVE_BOUND };

inline uint32_t computeTriangleFlags(const Hf& h, const uint32_t adj[3],
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

inline bool selectNormal(float u, float v, uint32_t triFlags, SnRegion& region)
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

// ---------------------------------------------------------------------------
// Trace-free overloads, for the fixtures that only want the answer.
// ---------------------------------------------------------------------------
inline uint32_t computeTriangleFlags(const Hf& h, const uint32_t adj[3],
                                     V3 cur0, V3 cur1, V3 cur2, V3 triNormal, V3 scale)
{
    EdgeOutcome sink[3];
    return computeTriangleFlags(h, adj, cur0, cur1, cur2, triNormal, scale, sink);
}

inline bool selectNormal(float u, float v, uint32_t triFlags)
{
    SnRegion sink;
    return selectNormal(u, v, triFlags, sink);
}

}  // namespace hfref
}  // namespace pxgtest

#endif  // PXG_HEIGHTFIELD_REF_H
