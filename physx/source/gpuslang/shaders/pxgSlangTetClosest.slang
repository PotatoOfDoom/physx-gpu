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


// ===========================================================================
// pxgSlangTetClosest — the point-vs-tetrahedron geometry the deformable narrowphase shares.
//
// Extracted from softBodySelfCollisionBda when the softbody-vs-cloth VERTEX contact generator
// (P8.8 step 4) turned out to need the same four functions.  Nothing here changed in the
// move; the self-collision fixture re-verifies the module for free, which is the point of
// extracting rather than copying.
//
//   getNormalized            — rsqrt normalise, zero for a zero vector
//   isValidTet               — signed volume > 1e-9
//   isPointOutsideOfPlane4   — the four per-face sign products; ALL >= 0 means the point is
//                              INSIDE.  The CUDA name reads the other way round and is
//                              transcribed rather than renamed.
//   closestPtTetrahedron     — the closest point on the SURFACE faces named by a 4-bit hint,
//                              with the outward normal of the winning face
//
// NoContraction applies at the importer: these are dot-of-cross chains throughout, and the
// only rsqrt picks a direction rather than a decision.
// ===========================================================================
module pxgSlangTetClosest;

public float3 getNormalized(float3 v)
{
    const float m = dot(v, v);
    return m > 0.0f ? v * rsqrt(m) : float3(0, 0, 0);
}

public bool isValidTet(float3 v0, float3 v1, float3 v2, float3 v3)
{
    return dot(v1 - v0, cross(v2 - v0, v3 - v0)) > 1e-9f;
}

// PointOutsideOfPlane4: per-face product of the point's and the opposite vertex's signed
// distances; all >= 0 means the point is INSIDE (the CUDA name reads the other way round —
// it is transcribed, not renamed).
public bool isPointOutsideOfPlane4(float3 p, float3 a, float3 b, float3 c, float3 d)
{
    const float3 ap = p - a;
    const float3 ab = b - a;
    const float3 ac = c - a;
    const float3 ad = d - a;

    const float3 v0 = cross(ab, ac);
    const float signa0 = dot(v0, ap);
    const float signd0 = dot(v0, ad);

    const float3 v1 = cross(ac, ad);
    const float signa1 = dot(v1, ap);
    const float signd1 = dot(v1, ab);

    const float3 v2 = cross(ad, ab);
    const float signa2 = dot(v2, ap);
    const float signd2 = dot(v2, ac);

    const float3 bd = d - b;
    const float3 bc = c - b;
    const float3 v3 = cross(bd, bc);
    const float signd3 = dot(v3, p - b);
    const float signa3 = dot(v3, a - b);

    return (signa0 * signd0 >= 0.0f) && (signa1 * signd1 >= 0.0f) &&
           (signa2 * signd2 >= 0.0f) && (signa3 * signd3 >= 0.0f);
}

// P9.30 made this PUBLIC.  It is Gu::closestPtPointTriangle2
// (GuDistancePointTriangle.h:44) verbatim, including the final association
// `(a + ab*v) + ac*w`, and femClothPsContactGenBda needs exactly it.  Four
// copies of this function already exist in the corpus; three are in Phase-1
// descriptor-set shaders and stay there by the recorded rule, and the fourth is
// in psMeshMidphaseBda, a BDA shader that should move onto this one the next
// time it is touched -- noted rather than changed inside a tranche about
// cloth-vs-particle.
public float3 closestPtPointTriangle2(float3 p, float3 a, float3 b, float3 c, float3 ab, float3 ac)
{
    const float3 ap = p - a;
    const float d1 = dot(ab, ap);
    const float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;
    const float3 bp = p - b;
    const float d3 = dot(ab, bp);
    const float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) { const float v = d1 / (d1 - d3); return a + v * ab; }
    const float3 cp = p - c;
    const float d5 = dot(ab, cp);
    const float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) { const float w = d2 / (d2 - d6); return a + w * ac; }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
    { const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6)); return b + w * (c - b); }
    const float denom = 1.0f / (va + vb + vc);
    return a + ab * (vb * denom) + ac * (vc * denom);
}

public void closestPtTetrahedron(float3 point, uint hint, float3 a, float3 b, float3 c, float3 d,
                          inout float3 normal, inout float3 closestP, inout float sqDist)
{
    const float3 ac = c - a;
    const float3 ab = b - a;
    const float3 ad = d - a;

    if ((hint & 1u) != 0u)
    {
        const float3 tPoint = closestPtPointTriangle2(point, a, b, c, ab, ac);
        const float3 v = tPoint - point;
        const float tSqDist = dot(v, v);
        if (tSqDist < sqDist)
        {
            normal = getNormalized(cross(ac, ab));
            if (dot(ad, normal) < 0.0f) normal = -normal;
            closestP = tPoint;
            sqDist = tSqDist;
        }
    }
    if ((hint & 2u) != 0u)
    {
        const float3 tPoint = closestPtPointTriangle2(point, a, b, d, ab, ad);
        const float3 v = tPoint - point;
        const float tSqDist = dot(v, v);
        if (tSqDist < sqDist)
        {
            normal = getNormalized(cross(ab, ad));
            if (dot(ac, normal) < 0.0f) normal = -normal;
            closestP = tPoint;
            sqDist = tSqDist;
        }
    }
    if ((hint & 4u) != 0u)
    {
        const float3 tPoint = closestPtPointTriangle2(point, a, c, d, ac, ad);
        const float3 v = tPoint - point;
        const float tSqDist = dot(v, v);
        if (tSqDist < sqDist)
        {
            normal = getNormalized(cross(ac, ad));
            if (dot(ab, normal) < 0.0f) normal = -normal;
            closestP = tPoint;
            sqDist = tSqDist;
        }
    }
    if ((hint & 8u) != 0u)
    {
        const float3 bd = d - b;
        const float3 bc = c - b;
        const float3 tPoint = closestPtPointTriangle2(point, b, c, d, bc, bd);
        const float3 v = tPoint - point;
        const float tSqDist = dot(v, v);
        if (tSqDist < sqDist)
        {
            normal = getNormalized(cross(bc, bd));
            if (dot(ab, normal) > 0.0f) normal = -normal;
            closestP = tPoint;
            sqDist = tSqDist;
        }
    }
}

// ---------------------------------------------------------------------------
// Gu::closestPtPointBaryTriangle (GuDistancePointTriangle.h:123) — P9.34a.
//
// The closest point on the triangle (1,0,0), (0,1,0), (0,0,1) to `c`, i.e. the
// projection of an arbitrary barycentric triple back onto the valid simplex.  The
// header calls it "a specialization of Gu::closestPtPointTriangle2 for this
// specific triangle", which is why it belongs beside that function rather than in
// the SDF module that happens to be its only caller today
// (doTriangleSDFCollision, sdfCollision.cuh:473).
//
// UNLIKE almost everything else in this corpus, this function is PX_CUDA_CALLABLE
// and takes no texture, so a fixture can call THE ENGINE'S OWN COPY on the host
// and compare bit for bit.  It is an oracle, not a transcription checked against a
// hand-written mirror, and the fixture uses it as one.
//
// THE TRAPS, all of them about which branch is taken rather than about arithmetic:
//
//  B1. The projection onto the plane x+y+z=1 comes FIRST and is subtracted as a
//      SCALAR broadcast: `c -= PxVec3(third * (c.x + c.y + c.z - 1.0f))`.  The sum
//      is formed, then 1 subtracted, then scaled by 1/3 — not `(sum - 1) / 3`, and
//      not a per-component adjustment.
//
//  B2. The three two-negative tests come BEFORE the three one-negative ones, and
//      they test the OTHER two components: `c.y < 0 && c.z < 0` returns the X
//      vertex.  Reordering them changes the answer on a doubly-negative input,
//      because the one-negative branches below would fire on the first negative
//      they find and project onto an edge instead of snapping to a vertex.
//
//  B3. Each one-negative branch spreads HALF the negative component into the other
//      two (`d = c.x * 0.5f`, then `y = c.y + d`), and the result keeps the
//      ZEROED component, not the original.  `d` is negative, so this SUBTRACTS.
//
//  B4. The `> 1.0f` escapes inside each one-negative branch snap to a vertex, and
//      they are checked in the branch's own order — y before z, x before z, x
//      before y.  Both can be true only for a badly out-of-simplex input, and then
//      the first one wins.
//
//  B5. The fall-through returns the PROJECTED c, not the original argument.
// ---------------------------------------------------------------------------
public float3 closestPtPointBaryTriangle(float3 c)
{
    const float third = 1.0f / 3.0f;
    c -= float3(third * (c.x + c.y + c.z - 1.0f));      // B1

    // B2: two negative -> the positive vertex.
    if (c.y < 0.0f && c.z < 0.0f)
        return float3(1.0f, 0.0f, 0.0f);
    if (c.x < 0.0f && c.z < 0.0f)
        return float3(0.0f, 1.0f, 0.0f);
    if (c.x < 0.0f && c.y < 0.0f)
        return float3(0.0f, 0.0f, 1.0f);

    // B3/B4: one negative -> the edge projection, or the larger vertex.
    if (c.x < 0.0f)
    {
        const float d = c.x * 0.5f;
        const float y = c.y + d;
        const float z = c.z + d;
        if (y > 1.0f) return float3(0.0f, 1.0f, 0.0f);
        if (z > 1.0f) return float3(0.0f, 0.0f, 1.0f);
        return float3(0.0f, y, z);
    }
    if (c.y < 0.0f)
    {
        const float d = c.y * 0.5f;
        const float x = c.x + d;
        const float z = c.z + d;
        if (x > 1.0f) return float3(1.0f, 0.0f, 0.0f);
        if (z > 1.0f) return float3(0.0f, 0.0f, 1.0f);
        return float3(x, 0.0f, z);
    }
    if (c.z < 0.0f)
    {
        const float d = c.z * 0.5f;
        const float x = c.x + d;
        const float y = c.y + d;
        if (x > 1.0f) return float3(1.0f, 0.0f, 0.0f);
        if (y > 1.0f) return float3(0.0f, 1.0f, 0.0f);
        return float3(x, y, 0.0f);
    }
    return c;                                            // B5
}

// ---------------------------------------------------------------------------
// Gu::closestPtPointTetrahedron, the FIVE-ARGUMENT overload
// (GuDistancePointTetrahedron.h:77) — P9.35a.
//
// Like closestPtPointBaryTriangle above, this one is PX_INLINE PX_CUDA_CALLABLE
// and takes no texture, so the fixture calls THE ENGINE'S OWN COPY and compares
// bit for bit.  A second oracle in the same module.
//
// IT IS NOT `closestPtTetrahedron` ABOVE, and collapsing the two is the failure
// mode this header exists to prevent.  Every one of these differs:
//
//   G1. THE FACES ARE DIFFERENT, in order AND in winding:
//         this one    (a,b,c)  (a,c,d)  (a,d,b)  (b,d,c)
//         the other   (a,b,c)  (a,b,d)  (a,c,d)  (b,c,d)
//       Only the first face agrees.  The third here is (a,d,b) — b and d
//       swapped relative to any ascending spelling — and the fourth is (b,d,c).
//       Since closestPtPointTriangle2 takes its two edge vectors as separate
//       arguments, a wrong winding is not a sign flip that cancels: the edges
//       passed are ab/ac, ac/ad, ad/ab and bd/bc, each paired with ITS OWN face.
//
//   G2. THERE IS NO HINT MASK.  All four faces are always evaluated.
//
//   G3. IT SEEDS FROM FACE 0 rather than from PX_MAX_F32 and an empty point.
//       The other one takes `sqDist` in as an inout and may leave it untouched;
//       this one always returns one of the four faces' points.
//
//   G4. THE DISTANCE IS TO `p`, not the point's own norm.  `diff = closestPt - p`
//       and `sqDist = diff.dot(diff)` — the six-argument overload in the .cpp
//       spells the same idea as `bestClosestPt.dot(bestClosestPt)`, which is only
//       equal when p is the origin.  This overload is the one the SDF descent
//       calls, and it is called with a NON-origin p.
//
//   G5. TIES GO TO THE EARLIER FACE: the comparison is a strict `<`, so a point
//       equidistant from two faces keeps the first one evaluated.
//
// The SDF descent calls it with the canonical simplex — a = (1,0,0), b = (0,1,0),
// c = (0,0,1), d = (0,0,0) — so it clamps a barycentric triple back into the
// standard 3-simplex.  Nothing here assumes that; the fixture feeds it general
// tetrahedra as well, because the engine copy it is checked against does too.
// ---------------------------------------------------------------------------
public float3 guClosestPtPointTetrahedron(float3 p, float3 a, float3 b, float3 c, float3 d)
{
    const float3 ab = b - a;
    const float3 ac = c - a;
    const float3 ad = d - a;
    const float3 bc = c - b;
    const float3 bd = d - b;

    // G3: face 0 is the seed, not a candidate.
    float3 bestClosestPt = closestPtPointTriangle2(p, a, b, c, ab, ac);
    float3 diff = bestClosestPt - p;                       // G4
    float bestSqDist = dot(diff, diff);

    // G1: (a,c,d) with ac/ad.
    float3 closestPt = closestPtPointTriangle2(p, a, c, d, ac, ad);
    diff = closestPt - p;
    float sqDist = dot(diff, diff);
    if (sqDist < bestSqDist)                               // G5
    {
        bestClosestPt = closestPt;
        bestSqDist = sqDist;
    }

    // G1: (a,d,b) with ad/ab — NOT (a,b,d).
    closestPt = closestPtPointTriangle2(p, a, d, b, ad, ab);
    diff = closestPt - p;
    sqDist = dot(diff, diff);
    if (sqDist < bestSqDist)
    {
        bestClosestPt = closestPt;
        bestSqDist = sqDist;
    }

    // G1: (b,d,c) with bd/bc — NOT (b,c,d).
    closestPt = closestPtPointTriangle2(p, b, d, c, bd, bc);
    diff = closestPt - p;
    sqDist = dot(diff, diff);
    if (sqDist < bestSqDist)
    {
        bestClosestPt = closestPt;
        bestSqDist = sqDist;
    }

    return bestClosestPt;
}
