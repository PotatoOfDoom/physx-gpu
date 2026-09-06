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

// pxgSlangCapTriangle.slang -- `generateContacts` (capsuleTriangle.cuh:329-406),
// `generateEE` (:408-461) and `generateEEContacts` (:463-474).  P9.52b-3.
//
// The two contact generators the capsule-vs-triangle cores call back to back, on
// the heightfield path (convexHeightfield.cu:770-775) and on the trimesh path
// (convexMesh.cu). `generateContacts` produces the FACE contacts -- the capsule's
// two endpoints projected down the patch normal -- and `generateEEContacts` the
// EDGE-EDGE ones, and between them they may emit at most two.
//
// ===========================================================================
// THE CAP IS STRUCTURAL, AND IT IS WHY A TWO-SLOT ARRAY IS ENOUGH
// ===========================================================================
//
// `generateContacts` has exactly two `if` blocks and each appends at most one, so
// it returns 0, 1 or 2.  `generateEEContacts` guards EVERY one of its three calls
// with `if (numContacts < 2)`, so it can only raise the count to 2.  Nothing in
// either function can write index 2.  The CUDA relies on the same argument -- its
// caller hands it `shContacts[threadIdx.x * 2]`, a two-slot window -- so the bound
// is upstream's, not this port's, and it is written down because an array overrun
// in Slang is undefined rather than merely wrong.
//
// ===========================================================================
// THE TWO RADIUS TESTS ARE NOT THE SAME TEST
// ===========================================================================
//
// `generateContacts` compares `inflatedRadius > ipt` -- UNSQUARED, against a
// SIGNED penetration along the patch normal, which may be negative.
// `generateEE` compares `sqInflatedRadius > sqDist` -- SQUARED, against a squared
// distance.  Passing the squared radius to the first or the plain radius to the
// second compiles, runs, and is wrong by a squaring; the core is what keeps them
// apart, by taking both values and handing each the right one
// (convexHeightfield.cu:770 and :774).
//
// ===========================================================================
// WHAT `generateContacts` ACTUALLY EMITS
// ===========================================================================
//
// NOT the projected point.  It computes `closestP31 = p - normal * ipt` purely to
// get the barycentrics, and then writes the SEGMENT ENDPOINT `p` as the contact
// (:370) with `ipt` as the separation.  Writing the projected point instead would
// look more principled and would move every capsule contact by the penetration
// depth.
//
// The `ipt` guard is `ideom > 0.f` STRICT with a PX_MAX_F32 fallback, and the
// negation sits OUTSIDE the divide: `-(inomp / ideom)`.  Its trailing comment
// claims `FSel(FIsGrtr(ideom, FZero()), FDiv(inomp, ideom), FZero())` -- no
// negation and a zero fallback -- and is wrong twice, the same way the bdenom
// comment in `distanceSegmentTriangleSquared` is wrong (see pxgSlangSegTriangle,
// upstream candidate 51).  Following the comments in this file is a reliable way
// to write a different function.
//
// ===========================================================================
// `generateEE` DIVIDES BY `denom` UNGUARDED -- AND THE EARLIER GUARD IS THE GUARD
// ===========================================================================
//
// `tValue = nom / denom` is an unguarded divide, and the line above it in the .cu
// is the COMMENTED-OUT clamped version (`FClamp(FDiv(nom, denom), zero, one)`).
// The live test is only `if (tValue > 1.f || tValue < 0.f) return;`, which a NaN
// fails BOTH halves of -- so a NaN would walk straight through it.
//
// It cannot arrive, and the reason is worth stating because it is not visible from
// the two lines involved.  The two quantities are THE SAME TRIPLE PRODUCT:
//
//     npq   = n . pq         where n = ab x normal
//           = (ab x normal) . pq              = det(ab, normal, pq)
//     denom = perNormal . ab  where perNormal = normal x pq
//           = (normal x pq) . ab              = det(normal, pq, ab)
//
// and a determinant is invariant under a CYCLIC permutation of its rows, so
// det(ab, normal, pq) == det(normal, pq, ab).  `npq` and `denom` are one number
// computed two ways.  The `npq == 0.f` return three lines earlier therefore
// already excludes the zero denominator, and the divide is safe -- not because
// anyone guarded it, but because the guard that looks unrelated is the same test.
//
// In float the two routes round differently, so they are equal only to within an
// ulp or so; a `denom` that rounds to exactly zero while `npq` does not is
// possible in principle.  Then `nom / 0` is +-INF for a non-zero numerator, which
// the `> 1` test catches, and NaN only if `nom` rounds to zero at the same time.
// The port reproduces the whole path either way, NaN included: with a NaN
// `tValue` the guard passes, `v` and `sqDist` become NaN, and
// `sqInflatedRadius > NaN` is FALSE, so the contact is dropped at the last
// comparison rather than at the guard that appears to handle it.

module pxgSlangCapTriangle;

import pxgSlangPointTriangle;   // the corpus's one explicit dot/cross (P9.56)
import pxgSlangSegTriangle;     // stValidBarycentric, shared with the distance core

static const float CT_MAX_F32 = 3.4028234663852886e+38f;   // PX_MAX_F32

// The two-slot window the CUDA's caller provides, carried as a value so the two
// generators can be composed without pointers.
public struct CapContacts
{
    public float3 pt[2];
    public float sep[2];
    public uint n;
}

public CapContacts capContactsInit()
{
    CapContacts o;
    o.pt[0] = float3(0.0f, 0.0f, 0.0f);
    o.pt[1] = float3(0.0f, 0.0f, 0.0f);
    o.sep[0] = 0.0f;
    o.sep[1] = 0.0f;
    o.n = 0u;
    return o;
}

// generateContacts (capsuleTriangle.cuh:329-406).
//
// `planeNormal` is the TRIANGLE's normal and `normal` the patch normal the core
// selected; they are different vectors and the function uses each exactly once in
// a way the other would not survive -- `ideom` is their dot product, and the
// projection travels along `normal`.
//
// IT STARTS FROM ZERO, and takes no incoming count.  The .cu declares
// `PxU32 numContacts = 0;` inside the function and RETURNS it (:368, :405); only
// `generateEEContacts` takes a running count by reference.  Threading an incoming
// count through here would compile and would be a different function the first
// time a caller passed a non-zero one.
public CapContacts capGenerateContacts(float3 a, float3 b, float3 c,
                                       float3 planeNormal, float3 normal,
                                       float3 p, float3 q, float inflatedRadius)
{
    CapContacts o = capContactsInit();

    const float3 ab = b - a;
    const float3 ac = c - a;
    const float3 ap = p - a;
    const float3 aq = q - a;

    const float d00 = ptDot(ab, ab);
    const float d01 = ptDot(ab, ac);
    const float d11 = ptDot(ac, ac);
    const float tdenom = (d00 * d11 - d01 * d01);
    const float bdenom = (tdenom > 0.0f) ? (1.0f / tdenom) : CT_MAX_F32;

    // `planeNormal.dot(-ap)`, with the negation on the VECTOR: each product is
    // negated before the sum.  That is the same value as negating the sum (IEEE
    // negation is exact and addition commutes with it), and it is spelled as the
    // .cu spells it.
    const float inomp = ptDot(planeNormal, -ap);
    const float ideom = ptDot(planeNormal, normal);

    const float ipt = ideom > 0.0f ? -(inomp / ideom) : CT_MAX_F32;

    const float3 closestP31 = p - normal * ipt;

    const float3 pV20 = closestP31 - a;
    const float pD20 = ptDot(pV20, ab);
    const float pD21 = ptDot(pV20, ac);
    const float v0 = (d11 * pD20 - d01 * pD21) * bdenom;
    const float w0 = (d00 * pD21 - d01 * pD20) * bdenom;

    const bool con0 = stValidBarycentric(v0, w0);

    if (con0 && (inflatedRadius > ipt))
    {
        // THE SEGMENT ENDPOINT, not closestP31 -- see the header.
        o.pt[o.n] = p;
        o.sep[o.n] = ipt;
        o.n = o.n + 1u;
    }

    const float inomq = ptDot(planeNormal, -aq);

    const float iqt = ideom > 0.0f ? -(inomq / ideom) : CT_MAX_F32;

    const float3 closestP41 = q - normal * iqt;

    const float3 qV20 = closestP41 - a;
    const float qD20 = ptDot(qV20, ab);
    const float qD21 = ptDot(qV20, ac);
    const float v1 = (d11 * qD20 - d01 * qD21) * bdenom;
    const float w1 = (d00 * qD21 - d01 * qD20) * bdenom;

    const bool con1 = stValidBarycentric(v1, w1);

    if (con1 && (inflatedRadius > iqt))
    {
        o.pt[o.n] = q;
        o.sep[o.n] = iqt;
        o.n = o.n + 1u;
    }

    return o;
}

// generateEE (capsuleTriangle.cuh:408-461) -- one triangle edge against the
// capsule segment.
//
// The early returns are spelled as a guard chain because Slang cannot return a
// value-carrying struct from the middle of a function that also has to fall
// through; the ARITHMETIC order is unchanged, and every quantity is still computed
// exactly where the .cu computes it (nothing below a `return` in the .cu is
// evaluated here either).
public CapContacts capGenerateEE(float3 p, float3 q, float sqInflatedRadius,
                                 float3 normal, float3 a, float3 b,
                                 CapContacts inOut)
{
    CapContacts o = inOut;

    const float3 ab = b - a;
    const float3 n = ptCross(ab, normal);
    const float d = ptDot(n, a);
    const float np = ptDot(n, p);
    const float nq = ptDot(n, q);
    const float signP = np - d;
    const float signQ = nq - d;
    const float temp = signP * signQ;
    if (temp > 0.0f)
        return o;   // both endpoints on one side of the edge plane

    const float3 pq = q - p;
    const float npq = ptDot(n, pq);
    // EXACT equality against zero, not an epsilon.  A segment very nearly parallel
    // to the edge plane passes this test and then divides by a very small number.
    if (npq == 0.0f)
        return o;

    const float segTValue = (d - np) / npq;
    const float3 localPointA = p + pq * segTValue;

    const float3 perNormal = ptCross(normal, pq);
    const float3 ap = localPointA - a;
    const float nom = ptDot(perNormal, ap);
    const float denom = ptDot(perNormal, ab);

    // No clamp -- see the header.  A NaN here fails BOTH comparisons and walks
    // through this guard.
    const float tValue = nom / denom;
    if (tValue > 1.0f || tValue < 0.0f)
        return o;

    const float3 v = ap - ab * tValue;
    const float sqDist = ptDot(v, v);

    if (sqInflatedRadius > sqDist)
    {
        // The separation is SIGNED and taken along the PATCH normal, so it can be
        // positive: this is a separation, not a penetration depth.
        const float signedDist = ptDot(v, normal);
        o.pt[o.n] = localPointA;
        o.sep[o.n] = signedDist;
        o.n = o.n + 1u;
    }

    return o;
}

// generateEEContacts (capsuleTriangle.cuh:463-474).
//
// The three calls are (a,b), (b,c), (a,c), so the last one traverses the closing
// edge from a rather than from c.
//
// WHETHER THE DIRECTION MATTERS TOOK TWO CORRECTIONS TO GET RIGHT, so here is the
// measured answer rather than the plausible one.
//
// In EXACT arithmetic, reversing an edge changes nothing.  Every quantity derived
// from it flips or cancels:
//   n' = (a - b) x normal = -n, so np' and nq' are the exact negations of theirs
//        (negating a vector negates its dot products exactly in IEEE);
//   d' = n'.b = -(n.b), and n.b = n.a + n.(b - a) where the second term is
//        IDENTICALLY ZERO because n is orthogonal to the edge by construction --
//        so d' = -d;
//   temp' = (-signP)(-signQ) = temp, so the same-side return is unchanged;
//   segTValue' = (-(d - np)) / (-npq) = segTValue, so `localPointA` -- the point
//        that gets EMITTED -- is the same point;
//   perNormal is untouched, denom' = -denom and nom' = nom - denom, so
//        tValue' = 1 - tValue, which lies in [0, 1] exactly when tValue does;
//   v' = (L - b) - (a - b)(1 - tValue) = L - (a + (b - a) tValue) = v,
//        so sqDist, the radius test and the signed separation all agree.
//
// IN FLOAT IT IS NOT BIT-EXACT, and the one term above that is only zero
// mathematically is why: `d` is computed as `dot(n, firstVertex)` directly, and
// the two first vertices differ by exactly the edge vector n is orthogonal to, so
// the two `d` values differ by the ROUNDING RESIDUE of `n . (b - a)`.
//
// The sweep measured that.  `G2-second-edge-reversed` is KILLED -- on the
// tilted-patch-normal case the reported separation moves from 0.490612626 to
// 0.490612596, one to two ulp -- while `G1-third-edge-reversed` SURVIVES, because
// no case in the fixture is sharp enough on that edge.  That difference is a
// statement about the inputs, not about the code.
//
// So: mathematically symmetric, numerically not quite.  The (a,c) spelling is
// transcribed because the .cu writes it, and because at one ulp the distinction is
// exactly the kind this port exists to preserve.
public CapContacts capGenerateEEContacts(float3 a, float3 b, float3 c, float3 normal,
                                         float3 p, float3 q, float sqInflatedRadius,
                                         CapContacts inOut)
{
    CapContacts o = inOut;

    if (o.n < 2u) o = capGenerateEE(p, q, sqInflatedRadius, normal, a, b, o);
    if (o.n < 2u) o = capGenerateEE(p, q, sqInflatedRadius, normal, b, c, o);
    if (o.n < 2u) o = capGenerateEE(p, q, sqInflatedRadius, normal, a, c, o);

    return o;
}
