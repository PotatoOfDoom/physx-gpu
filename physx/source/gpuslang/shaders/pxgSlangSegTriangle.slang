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

// pxgSlangSegTriangle.slang -- `distanceSegmentSegmentSquared`
// (distanceSegmentSegment.cuh:33-88), `isValidTriangleBarycentricCoord` and
// `distanceSegmentTriangleSquared` (capsuleTriangle.cuh:102-327).  P9.52b-2.
//
// This is the geometric core of every capsule-vs-triangle path in the engine: the
// heightfield capsule arm (convexHeightfield.cu:700), its trimesh twin
// (convexMesh.cu:492) and the cloth primitive path (femClothPrimitives.cu:430) all
// call the same function, and all three want it on the BDA ABI eventually.  There
// is a verified Phase-1 port (`npSegTriangle`); this is the shared module the
// engine side will import, written from the CUDA again rather than copied from it.
//
// ===========================================================================
// PxClamp IS NOT `clamp`, AND THE DIFFERENCE IS PINNED HERE RATHER THAN ARGUED
// ===========================================================================
//
// `PxClamp(v, lo, hi)` is `PxMin(hi, PxMax(lo, v))` (PxMath.h:142-146), and both
// halves are written as a comparison-and-select: for float, `PxMax(a, b)` is
// selectMax `a > b ? a : b` and `PxMin(a, b)` is selectMin `a < b ? a : b`
// (PxMath.h:81-101, PxUnixMathIntrinsics.h:116-125).  That pins the NaN answer:
// with v = NaN, `lo > NaN` is false, so PxMax yields NaN, and PxClamp yields NaN
// (P9.86: corrected; the generic template `a < b ? b : a` is not the float path).
//
// Slang's `clamp` lowers to GLSL.std.450 FClamp/NClamp, whose result the Vulkan
// environment spec leaves UNDEFINED when an operand is NaN, and whose zero sign
// on a +0/-0 tie is implementation-defined (P9.86: PxClamp(-0, 0, 1) is -0).  So
// the two spellings agree on every non-zero, non-NaN input and are not the same
// function.
//
// At the three sites below a NaN provably cannot arrive:
//   * `s0` is computed from `temp / denom` and is DISCARDED whenever
//     `eps > denom`, which covers denom == 0 -- so the only path that consumes s0
//     has denom >= 1e-6 and a finite quotient;
//   * `tTmp` and `comp` are built from `eRecip` / `aRecip`, which are exactly 0
//     when their denominator is below eps, so neither can be an indeterminate
//     form.
// The Phase-1 port relies on that argument without stating it.  This module does
// not rely on it at all: it spells PxClamp, which costs one select and removes the
// question.  (The wider question -- 132 `clamp(` sites across the corpus against
// 43 PxClamp calls in the narrowphase CUDA -- is an audit of its own, not this
// tranche.)

// ===========================================================================
// THE CPU SIBLING OF THIS FUNCTION DISAGREES WITH IT -- UPSTREAM CANDIDATE 51
// ===========================================================================
//
// PhysX carries a second implementation of the same function for the CPU, in
// physx/source/geomutils/src/distance/GuDistanceSegmentTriangle.cpp, and the two
// differ in TWO decisions, both of which change the returned point:
//
//   GPU  capsuleTriangle.cuh:136   bdenom = tDenom > 0 ? 1/tDenom : PX_MAX_F32
//   CPU  GuDistanceSegmentTriangle.cpp:401
//                                  bdenom = FSel(FIsGrtr(tDenom, zero),
//                                                FRecip(tDenom), zero)
//
//   GPU  capsuleTriangle.cuh:151   if (dMul <= 0.f)          NON-STRICT
//   CPU  GuDistanceSegmentTriangle.cpp:413
//                                  con = FIsGrtr(zero, dMul)  i.e. dMul < 0, STRICT
//
// The first is not cosmetic.  For a degenerate triangle (ab parallel to ac, so
// tDenom is zero) the CPU's zero fallback makes v0 = w0 = 0, `isValid(0, 0)` is
// TRUE, and the function returns VERTEX A at distance sqDist3; the GPU's
// PX_MAX_F32 sends the barycentrics to +-INF, the validity test fails, and control
// falls through to the three EDGE distances.  Different point, different distance,
// same input.  Note also that the GPU line's own trailing comment documents the
// CPU's spelling -- it is a stale copy of the SIMD original, and following it
// would have silently imported the CPU behaviour.
//
// This module ports the GPU source, because that is what the GPU pipeline runs and
// what this port replaces.  DOCUMENTED, NOT REPORTED, like the other 50.
//
// ===========================================================================
// THE MUTATION SWEEP: 44 MUTANTS, 41 KILLED, THREE EQUIVALENCES
// ===========================================================================
//
// The first sweep left THIRTEEN survivors and eight of them were the fixture:
//
//   * every squared length in the segment-segment cases was either 0 or >= 1, so
//     `eps` could move from 1e-6 to 1e-3 unnoticed.  Cases now sit INSIDE that
//     band;
//   * the degenerate direction was exactly zero, which makes b and c zero too and
//     the guarded reciprocal's FALLBACK VALUE unobservable -- and a first attempt
//     at fixing that still failed, because the numerator came out negative and the
//     clamp flattened both answers to zero.  The case that works has a below eps,
//     non-zero, AND a positive numerator;
//   * `bdenom`'s fallback could not be seen at all with a collinear triangle,
//     because when ac is parallel to ab the barycentric numerators vanish
//     IDENTICALLY.  By Cauchy-Schwarz tDenom is non-negative for every triangle,
//     so the fallback is reachable only where ROUNDING drives it to zero -- the
//     case now uses d11 = 4 + 1e-10, which is 4.0 in float;
//   * the barycentric tolerances were only ever tested far from their boundaries.
//     Cases now put v EXACTLY at -1e-6 and v + w EXACTLY at 1 + 1e-6, which is
//     constructible bit-exactly on an axis-aligned triangle because the projection
//     preserves the two in-plane coordinates;
//   * the plane-crossing cases all had `n.dot(pq)` a power of two, where
//     `nom * (1/d)` and `nom / d` agree exactly.  A tilted triangle pins the
//     reciprocal spelling;
//   * the edge selection's strict `>` needs an exact TIE, and the first tie
//     constructed did not separate the two tied edges (both closest points
//     collapsed onto vertex a, so both reported (0, 0)).  The tie that works puts
//     the triangle and the segment in one plane where every dot product is a
//     TWO-term sum -- and a two-term sum is commutative, so the mirrored edge
//     computations agree bit for bit while their parameters do not.
//
// The three survivors are equivalences:
//
//   S14-clamp-is-builtin        PxClamp against the library clamp.  They differ
//       on NaN, which the header above proves cannot reach any of the three
//       sites, and (P9.86) on the sign of a zero result, which can: `comp` is -0
//       when aRecip == 0.  This survivor predates the -0 fixture cases and is no
//       longer an equivalence under a sign-of-zero comparison -- the module keeps
//       the explicit spelling for that reason.
//   T10-closestP31-sign         `p - n*dist3` against `p + n*dist3`.
//   T11-closestP41-uses-dist3   `q - n*dist4` against `q - n*dist3`.
//       Both displace the projected point ALONG n, and the only things read from
//       it are its dot products with ab and ac -- both of which are orthogonal to
//       n in exact arithmetic, so a displacement along n is invisible to them.
//       The same family as B4 and C8 in the sphere core's sweep.

module pxgSlangSegTriangle;

import pxgSlangPointTriangle;   // the corpus's one explicit dot/cross (P9.56)

static const float ST_MAX_F32 = 3.4028234663852886e+38f;   // PX_MAX_F32

// PxMath.h:75-94 and :142-146, transcribed as comparisons rather than delegated
// to an intrinsic -- see the header.
// P9.86: PxMax<float> is the specialisation intrinsics::selectMax (PxMath.h:81-85 ->
// PxUnixMathIntrinsics.h:122), `a > b ? a : b`, NOT the generic `a < b ? b : a`:
// PxMax(+0, -0) = -0 and a NaN returns b, so PxClamp(-0,0,1) = -0, PxClamp(NaN,0,1) = NaN
// (distanceSegmentSegment.cuh:63/76/81).
public float stMax(float a, float b) { return a > b ? a : b; }
public float stMin(float a, float b) { return a < b ? a : b; }
public float stClamp(float v, float lo, float hi) { return stMin(hi, stMax(lo, v)); }

// isValidTriangleBarycentricCoord (capsuleTriangle.cuh:102-110).
//
// The tolerance is applied on BOTH sides: `zero` is -1e-6 and `one` is 1 + 1e-6,
// so a coordinate a hair outside the triangle still counts as inside.  All three
// comparisons are NON-STRICT, and the third is on the SUM only -- there is no
// `(v + w) >= zero` test, because v >= zero and w >= zero already imply it to
// within the tolerance the function grants.
public bool stValidBarycentric(float v, float w)
{
    const float eps = 1e-6f;
    const float zero = -eps;
    const float one = 1.0f + eps;

    return (v >= zero && v <= one) && (w >= zero && w <= one) && ((v + w) <= one);
}

// distanceSegmentSegmentSquared (distanceSegmentSegment.cuh:33-88).
//
// Note what the parameters are: `p1, d1` and `p2, d2` are a POINT AND A DIRECTION,
// not two endpoints.  The outputs are the two segment parameters in [0, 1].
//
// THE STRUCTURE IS ONE PASS, NOT AN ITERATION.  It clamps s, recomputes t from the
// clamped s, clamps t, and recomputes s from the clamped t -- and then stops.  The
// final `s` therefore comes from the SECOND recomputation and the final `t` from
// the first, which is why they are assigned from `s2` and `t2` and not from `s0`
// and `tTmp`.
//
// `aRecip` and `eRecip` are GUARDED reciprocals (zero when the squared length is
// below eps), but `temp / denom` on the next line is a PLAIN DIVIDE with no guard
// at all: when the segments are parallel `denom` is zero and that divide produces
// an infinity or a NaN.  It is harmless because `sTmp` discards it in exactly that
// case -- `eps > denom` is true for every denom below 1e-6, negatives included --
// but the divide still happens, and a port that "helpfully" guarded it would be
// computing something the .cu does not.
public void stSegmentSegmentSquared(float3 p1, float3 d1, float3 p2, float3 d2,
                                    out float s, out float t)
{
    const float eps = 1e-6f;
    const float3 r = p1 - p2;
    const float a = ptDot(d1, d1);
    const float e = ptDot(d2, d2);
    const float b = ptDot(d1, d2);
    const float c = ptDot(d1, r);
    const float aRecip = a > eps ? (1.0f / a) : 0.0f;
    const float eRecip = e > eps ? (1.0f / e) : 0.0f;
    const float f = ptDot(d2, r);

    const float denom = a * e - b * b;
    const float temp = b * f - c * e;
    const float s0 = stClamp(temp / denom, 0.0f, 1.0f);

    const float sTmp = eps > denom ? 0.5f : s0;

    const float tTmp = (b * sTmp + f) * eRecip;
    const float t2 = stClamp(tTmp, 0.0f, 1.0f);

    const float comp = (b * t2 - c) * aRecip;
    const float s2 = stClamp(comp, 0.0f, 1.0f);

    s = s2;
    t = t2;
}

// distanceSegmentTriangleSquared (capsuleTriangle.cuh:117-327).
//
// `t` is the parameter along the segment p->q; `u` and `v` are the triangle's
// barycentric pair, with the third weight implied as `1 - (u + v)`.
//
// ===========================================================================
// THE FOUR WAYS OUT, IN THE ORDER THE FUNCTION TAKES THEM
// ===========================================================================
//
//  1. THE SEGMENT CROSSES THE PLANE INSIDE THE TRIANGLE (`dMul <= 0` and the
//     intersection's barycentrics are valid) -- returns EXACTLY 0.0f.  Note the
//     gate is NON-STRICT, so a segment endpoint lying exactly on the plane takes
//     this branch.
//  2. BOTH PROJECTIONS LAND INSIDE (`con0 && con1`) -- the closer endpoint wins,
//     and `t` is 0 or 1 exactly.
//  3. ONE PROJECTION LANDS INSIDE -- the better of that endpoint and the best
//     edge.
//  4. NEITHER -- the best edge.
//
// `bdenom` is `1/tDenom` when tDenom > 0 and PX_MAX_F32 otherwise, so a degenerate
// triangle does not divide by zero; it produces enormous barycentrics instead,
// which `stValidBarycentric` then rejects.  PX_MAX_F32, not zero -- the comment
// beside it in the .cu says FRecip and is simply wrong about the fallback.
//
// THE PLANE-INTERSECTION PARAMETER IS A RECIPROCAL TIMES A NUMERATOR
// (`denom = 1.f / n.dot(pq); t0 = nom * denom;`), NOT a divide.  The three
// edge distances below, by contrast, come from stSegmentSegmentSquared, which
// divides.  Two different spellings inside one function, both transcribed.
//
// THE EDGE CANDIDATES ARE COMPARED WITH STRICT `>`, so ties go to the LATER edge:
// `sqDist1 > sqDist0 && sqDist2 > sqDist0` picks ab, `else if (sqDist2 > sqDist1)`
// picks bc, and everything else picks ac.  Reversing either comparison changes
// which edge a tie selects, and a capsule lying symmetrically across a corner is
// exactly such a tie.
public float stSegmentTriangleSquared(float3 p, float3 q,
                                      float3 a, float3 b, float3 c,
                                      out float t, out float u, out float v)
{
    const float3 pq = q - p;
    const float3 ab = b - a;
    const float3 ac = c - a;
    const float3 bc = c - b;
    const float3 ap = p - a;
    const float3 aq = q - a;

    const float d00 = ptDot(ab, ab);
    const float d01 = ptDot(ab, ac);
    const float d11 = ptDot(ac, ac);
    const float tDenom = d00 * d11 - d01 * d01;

    const float bdenom = tDenom > 0.0f ? 1.0f / tDenom : ST_MAX_F32;

    // getNormalized: v * PxRecipSqrt(magnitudeSquared()), zero for a zero input.
    const float3 n = ptNormalized(ptCross(ab, ac));

    const float dist3 = ptDot(ap, n);
    const float sqDist3 = dist3 * dist3;

    const float dist4 = ptDot(aq, n);
    const float sqDist4 = dist4 * dist4;
    const float dMul = dist3 * dist4;

    // Slang has no early `return` out of the middle of a function that also has to
    // set out-parameters on the other paths, so the .cu's early return is spelled
    // as a flag.  The ARITHMETIC below is unconditional either way -- the .cu also
    // computes closestP31 and closestP41 after the branch, so nothing is skipped
    // that upstream would have executed.
    bool done = false;
    float rt = 0.0f, ru = 0.0f, rv = 0.0f, rsq = 0.0f;

    if (dMul <= 0.0f)
    {
        const float nom = -ptDot(n, ap);
        const float denom = 1.0f / ptDot(n, pq);
        const float t0 = nom * denom;
        const float3 ip = p + pq * t0;
        const float3 v2 = ip - a;
        const float d20 = ptDot(v2, ab);
        const float d21 = ptDot(v2, ac);
        const float v00 = (d11 * d20 - d01 * d21) * bdenom;
        const float w00 = (d00 * d21 - d01 * d20) * bdenom;
        if (stValidBarycentric(v00, w00))
        {
            rt = t0; ru = v00; rv = w00; rsq = 0.0f;
            done = true;
        }
    }

    if (!done)
    {
        const float3 closestP31 = p - n * dist3;

        const float3 pV20 = closestP31 - a;
        const float pD20 = ptDot(pV20, ab);
        const float pD21 = ptDot(pV20, ac);
        const float v0 = (d11 * pD20 - d01 * pD21) * bdenom;
        const float w0 = (d00 * pD21 - d01 * pD20) * bdenom;

        const bool con0 = stValidBarycentric(v0, w0);

        const float3 closestP41 = q - n * dist4;

        const float3 qV20 = closestP41 - a;
        const float qD20 = ptDot(qV20, ab);
        const float qD21 = ptDot(qV20, ac);
        const float v1 = (d11 * qD20 - d01 * qD21) * bdenom;
        const float w1 = (d00 * qD21 - d01 * qD20) * bdenom;

        const bool con1 = stValidBarycentric(v1, w1);

        if (con0 && con1)
        {
            if (sqDist4 > sqDist3) { rt = 0.0f; ru = v0; rv = w0; rsq = sqDist3; }
            else                   { rt = 1.0f; ru = v1; rv = w1; rsq = sqDist4; }
        }
        else
        {
            // THE ARGUMENT ORDER MATTERS AND IS NOT SYMMETRIC: the segment is
            // always (p, pq) and the edge is (a, ab), (b, bc) or (a, ac).  The
            // first output is the SEGMENT parameter, the second the EDGE
            // parameter, which is why the closest points below pair t00 with pq
            // and t10 with ab -- and NOT t00 with ab.
            float t00, t01, t02;
            float t10, t11, t12;

            stSegmentSegmentSquared(p, pq, a, ab, t00, t10);
            stSegmentSegmentSquared(p, pq, b, bc, t01, t11);
            stSegmentSegmentSquared(p, pq, a, ac, t02, t12);

            const float3 closestP00 = p + pq * t00;
            const float3 closestP01 = a + ab * t10;
            float3 tempV = closestP00 - closestP01;
            const float sqDist0 = ptDot(tempV, tempV);

            const float3 closestP10 = p + pq * t01;
            const float3 closestP11 = b + bc * t11;
            tempV = closestP10 - closestP11;
            const float sqDist1 = ptDot(tempV, tempV);

            const float3 closestP20 = p + pq * t02;
            const float3 closestP21 = a + ac * t12;
            tempV = closestP20 - closestP21;
            const float sqDist2 = ptDot(tempV, tempV);

            // edge ab: the edge parameter IS u, and v is zero.
            const float u10 = t10;
            const float v10 = 0.0f;
            // edge bc: the parameter runs b -> c, so u falls as v rises.
            const float u11 = 1.0f - t11;
            const float v11 = t11;
            // edge ac: u is zero and the edge parameter IS v.
            const float u12 = 0.0f;
            const float v12 = t12;

            float sqDistPE, uEdge, vEdge, tSeg;
            if (sqDist1 > sqDist0 && sqDist2 > sqDist0)
            {
                sqDistPE = sqDist0; uEdge = u10; vEdge = v10; tSeg = t00;
            }
            else if (sqDist2 > sqDist1)
            {
                sqDistPE = sqDist1; uEdge = u11; vEdge = v11; tSeg = t01;
            }
            else
            {
                sqDistPE = sqDist2; uEdge = u12; vEdge = v12; tSeg = t02;
            }

            if (con0)
            {
                if (sqDistPE > sqDist3) { rt = 0.0f; ru = v0; rv = w0; rsq = sqDist3; }
                else                    { rt = tSeg; ru = uEdge; rv = vEdge; rsq = sqDistPE; }
            }
            else if (con1)
            {
                if (sqDistPE > sqDist4) { rt = 1.0f; ru = v1; rv = w1; rsq = sqDist4; }
                else                    { rt = tSeg; ru = uEdge; rv = vEdge; rsq = sqDistPE; }
            }
            else
            {
                rt = tSeg; ru = uEdge; rv = vEdge; rsq = sqDistPE;
            }
        }
    }

    t = rt;
    u = ru;
    v = rv;
    return rsq;
}
