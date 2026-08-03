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
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS \'\'AS IS\'\' AND ANY
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
//
// ===========================================================================
// pxgSlangTriFeature — the triangle-feature classifier of the SECOND PASS.
//
// Step 3 of the decomposition P8.28 recommended, and the part of the
// convex-vs-trimesh chain with NO precedent anywhere else in this port.
//
// WHAT THE SECOND PASS IS.  `convexTriangleContactGen`
// (convexTriangle.cuh:1849) does not always emit its contacts immediately.  When
// `delayContacts` is set it instead appends the pair to a deferral list —
//
//     convTriSecondPairPass[atomicAdd(nbSecondPassPairs, 1)] = globalWarpIndex | mask;
//
// — where `mask` records WHICH FEATURE of the triangle the first contact landed
// on.  A contact sitting on a shared edge or a shared vertex cannot be accepted
// or rejected by looking at one triangle: the neighbour decides.  So the mask is
// the question the later pass answers, and this module is the classifier that
// asks it.
//
// The mask occupies the TOP THREE BITS (`n << 29`, convexNpCommon.h:121) and is
// OR-ed into a warp index, so the two share one 32-bit word.  That is why the
// enum starts at 1 rather than 0: zero means "no feature, an interior hit", and
// the word is then just the warp index.
//
// THE CLASSIFIER, verbatim (convexTriangle.cuh:1562).  A single threshold, 0.97,
// and a strict `>` throughout, tested in a FIXED ORDER:
//
//     u = 1 - v - w
//     v > 0.97          -> V1
//     w > 0.97          -> V2
//     u > 0.97          -> V0
//     (v + u) > 0.97    -> E01
//     (u + w) > 0.97    -> E02
//     (w + v) > 0.97    -> E12
//     otherwise            0
//
// TWO THINGS THE ORDER HIDES, both of which a "cleaner" rewrite loses:
//
//   1. THE VERTEX TESTS COME FIRST AND WIN.  At v = 0.98 both `v > 0.97` and
//      `(v + u) > 0.97` are true — a point that close to a vertex is also close
//      to both edges meeting there.  The order is what makes it a VERTEX.
//      Reordering the chain silently reclassifies every near-vertex contact as
//      an edge one, and the second pass then asks the wrong neighbour.
//
//   2. THE PAIRINGS ARE NOT THE OBVIOUS ONES.  Edge 0-1 is `v + u`, edge 0-2 is
//      `u + w`, edge 1-2 is `w + v`.  Reading `v` as "the weight of vertex 1"
//      and `w` as "the weight of vertex 2" makes each of those the sum of the
//      two vertices the edge JOINS, which is right — but the naming invites
//      `v + w` for edge 0-1, which is the OPPOSITE edge.
//
// Nothing here is clamped or normalised: the classifier takes whatever
// barycentrics it is handed, including ones outside the triangle.
// ===========================================================================

module pxgSlangTriFeature;

// ConvexTriIntermediateData::VoronoiState (convexNpCommon.h:121)
public static const uint TF_NONE = 0u;
public static const uint TF_V0  = 1u << 29;
public static const uint TF_V1  = 2u << 29;
public static const uint TF_V2  = 3u << 29;
public static const uint TF_E01 = 4u << 29;
public static const uint TF_E02 = 5u << 29;
public static const uint TF_E12 = 6u << 29;

public static const float TF_UPPER_BOUND = 0.97f;

// mirrors getTriangleActiveMask(v, w) (convexTriangle.cuh:1562).
public uint tfActiveMaskFromBary(float v, float w)
{
    const float u = 1.0f - v - w;

    if (v > TF_UPPER_BOUND)       return TF_V1;
    else if (w > TF_UPPER_BOUND)  return TF_V2;
    else if (u > TF_UPPER_BOUND)  return TF_V0;
    else if ((v + u) > TF_UPPER_BOUND) return TF_E01;
    else if ((u + w) > TF_UPPER_BOUND) return TF_E02;
    else if ((w + v) > TF_UPPER_BOUND) return TF_E12;
    return TF_NONE;
}

// mirrors barycentricCoordinates (convexTriangle.cuh:1538).  Signed-area form,
// with the degenerate case pinned to ZERO rather than to a NaN: upstream tests
// `totalArea == 0.f` exactly and substitutes a zero reciprocal, so a degenerate
// triangle yields v = w = 0 and the classifier reports V0.
public void tfBarycentric(float3 p, float3 a, float3 b, float3 c, out float v, out float w)
{
    const float3 ab = b - a;
    const float3 ac = c - a;
    const float3 n = cross(ab, ac);

    const float3 ap = a - p;
    const float3 bp = b - p;
    const float3 cp = c - p;

    const float va = dot(n, cross(bp, cp));
    const float vb = dot(n, cross(cp, ap));
    const float vc = dot(n, cross(ap, bp));
    const float totalArea = va + vb + vc;
    const float denom = totalArea == 0.0f ? 0.0f : 1.0f / totalArea;
    v = vb * denom;
    w = vc * denom;
}

// mirrors getTriangleActiveMask(pa, v0, v1, v2) (convexTriangle.cuh:1632).
public uint tfActiveMask(float3 pa, float3 v0, float3 v1, float3 v2)
{
    float v, w;
    tfBarycentric(pa, v0, v1, v2, v, w);
    return tfActiveMaskFromBary(v, w);
}
