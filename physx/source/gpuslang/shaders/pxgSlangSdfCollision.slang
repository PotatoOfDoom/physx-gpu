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

// pxgSlangSdfCollision.slang -- P9.12d, the four functions over SparseSDFTexture.
//
// Ports sdfCollision.cuh:156-240 -- PxSdfDistance, PxVolumeGrad,
// PxSdfSampleWithGrad and doVertexSDFCollision -- onto the P9.12c texture layer.
// Everything ps_sdfMeshCollisonLaunch does per particle is one call to
// doVertexSDFCollision plus the contact write, so this is the whole of that
// kernel's numerics; what is left after it is index chains and buffer writes.
//
// ---------------------------------------------------------------------------
// THE ORACLE, and how much of one it is
// ---------------------------------------------------------------------------
// The four functions are __device__ templates, so none of them is callable from
// the host and this is the P8.87 arrangement: CONTROL FLOW TRANSCRIBED, every
// primitive underneath it CALLED from the engine's own header.  That is a
// stronger claim here than it was there, because the primitives are nearly the
// whole content -- PxVec3::maximum / minimum / magnitude / getNormalized /
// normalize / dot and PxAbs are all PX_CUDA_CALLABLE and the fixture calls every
// one of them on the host, so the only transcribed part is the ORDER.
//
// The one thing the fixture cannot call is Sample(), which lives on the GPU.  So
// the fixture never predicts a sample value: it makes every function report the
// sample values IT used (the *Ex variants below) and checks the composition
// against those, exactly.  Sample() itself is P9.12c's claim and was measured
// there against the engine's own Gu::SDF::decodeSparse.
//
// The *Ex variants exist ONLY for that.  Each plain function is a one-line
// forwarder, so the traced and untraced paths cannot drift apart.
//
// ---------------------------------------------------------------------------
// THE TRAPS
// ---------------------------------------------------------------------------
//  C1. `maximum` THEN `minimum`, in that order (:160).  With a well-formed box
//      the order does not matter; with an inverted one (higher < lower) it
//      decides which end wins, and nothing in the engine validates the box.
//      Transcribed in the engine's order.
//  C2. PxVec3::maximum/minimum are PxMax/PxMin, which are TERNARIES --
//      `a > b ? a : b` and `a < b ? a : b` (float specialisations ->
//      intrinsics::selectMax/selectMin, P9.86) -- not min()/max()
//      intrinsics.  They differ on NaN (both selects return the SECOND operand
//      whenever a compare involves NaN: PxMax(NaN, b) = b, PxMax(a, NaN) = NaN)
//      and on signed zero (on a +0/-0 tie the second operand wins).
//      npSdfBoxDist.slang made the same call in Phase 1.
//  C3. PxSdfDistance ADDS the outside distance: `Sample(f) + diffMag`, with
//      diffMag = |localPos - clampedGridPt| >= 0 always.  Outside the box the
//      reported distance GROWS.  Subtracting, or dropping the term, is right
//      for every in-box point and wrong for every other one, which is why the
//      fixture's point list is half outside.
//  C4. `f = (clampedGridPt - sdfBoxLower) * invSdfDx` -- a MULTIPLY by the
//      stored reciprocal, not a divide by sdfDx.  The two round differently and
//      the fixture compares f bit-for-bit.
//  C5. PxVolumeGrad has TWO paths and they are not the same arithmetic.  The
//      interior one differences Sample() at f +- 1 in TEXEL space; the fall-back
//      differences PxSdfDistance at localPos +- sdfDx in WORLD space -- which
//      clamps, and therefore picks up C3's diffMag.  Well inside the box the two
//      agree exactly (the offset points stay in the box, so diffMag is 0 and the
//      texel coordinates coincide); at and beyond the border they do not, and
//      that is the whole reason the guard exists.
//  C6. NEITHER path divides by the step size.  What comes back is 2*dx times the
//      gradient, and every caller only ever normalises it -- but a port that
//      "fixes" the scale changes doVertexSDFCollision's zero test (below) from
//      exact to approximate, because dividing a zero difference by 2*dx is still
//      zero only while the division is exact.
//  C7. The guard is `f >= 1 && f <= sdfDims - 2` on ALL THREE axes, ANDed.  The
//      bounds are INCLUSIVE and the dimensions are the FINE ones (sdfDims), not
//      coarseSize -- on a sparse SDF those differ by the subgrid size.
//      `sdfDims.x - 2.0f` promotes the unsigned to float first, so a dimension
//      of 1 gives -1.0f and not a wrapped 4294967295.
//  C8. PxSdfSampleWithGrad's `dist` is Sample(f) RAW -- it does NOT add diffMag,
//      unlike PxSdfDistance three functions above it.  The outside distance
//      enters through `dir.dot(diff)` at the end instead, which is a different
//      number.  A port that reuses PxSdfDistance here is wrong for every point
//      outside the box.
//  C9. `dir` is an OUT-BY-REFERENCE that the reject path does NOT touch
//      (:224 returns without assigning).  The caller's value survives.  Ported
//      as `inout`, and the fixture seeds it with a sentinel and requires the
//      sentinel back.
// C10. The gradient is normalised AGAIN inside PxSdfSampleWithGrad
//      (`sdfGradient.getNormalized()`, :220) even though doVertexSDFCollision
//      already normalised it.  Idempotent to within an ulp, not to the bit, and
//      transcribed rather than elided.
// C11. `getNormalized()` is `m2 > 0 ? v * PxRecipSqrt(m2) : PxVec3(0)`
//      (PxVec3.h:285) and PxRecipSqrt is `1.0f / sqrtf(a)` on this platform
//      (PxUnixMathIntrinsics.h:92) -- NOT an rsqrt approximation.  `rsqrt()` in
//      Slang lowers to an approximate instruction on some drivers, so the
//      reciprocal is spelled out.  Same reasoning as P8.69's rotateAndNormalize.
// C12. `normalize()` (PxVec3.h:295) is a DIFFERENT function: it normalises IN
//      PLACE and returns the MAGNITUDE, and it only divides when that magnitude
//      is > 0.  doVertexSDFCollision's `if (sdfGradient.normalize() == 0.0f)`
//      therefore does two things at once, and the substitution fires on the
//      value it just failed to divide by.
// C13. The discontinuity substitute is `PxVec3(0.5448762f, 0.672269f,
//      0.398825f).getNormalized()` -- and the getNormalized is LOAD-BEARING:
//      that vector is 0.95284 long, so dropping it hands PxSdfSampleWithGrad a
//      gradient of the wrong length.  (It is re-normalised there by C10, which
//      is exactly why the fixture checks the SUBSTITUTED VECTOR rather than the
//      final dir -- through dir alone the mutant is invisible.)
// C14. `mag > 0` and `mag == 0` are not complementary: a NaN gradient satisfies
//      neither, so it is neither divided nor substituted and flows through
//      untouched.  That is the CUDA's behaviour and it is transcribed, not
//      tidied into an else.
//
// A DELIBERATE DIVERGENCE, recorded rather than hidden: the CUDA spells
// `localPos.maximum(lower).minimum(higher)` and `(clamped - lower) * invSdfDx`
// three times over, once in each of the first three functions.  Here they are
// two shared helpers.  The expressions are identical; factoring them is what
// lets the fixture check them ONCE, against the engine's own PxVec3, and have
// that check cover all three call sites.

import pxgSlangSdfTexture;
import pxgSlangCommon;   // quatRotate / quatRotateInv for the two helpers below
import pxgSlangTetClosest;   // Gu::closestPtPointBaryTriangle (P9.34a)

// PX_MAX_F32, PxSimpleTypes.h:94.
static const float SDFC_MAX_F32 = 3.4028234663852885981170418348452e+38f;

// C13: the arbitrary unit direction the CUDA falls back on at a discontinuity.
static const float3 SDFC_DISCONTINUITY_DIR = float3(0.5448762f, 0.672269f, 0.398825f);

// C2: PxMin / PxMax as PxMath.h spells them.
float sdfcMin(float a, float b) { return a < b ? a : b; }
float sdfcMax(float a, float b) { return a > b ? a : b; }   // P9.86: PxMax<float> is intrinsics::selectMax (PxUnixMathIntrinsics.h:122) -- b wins ties (either zero sign) and any NaN compare

// PxVec3::magnitudeSquared / magnitude (PxVec3.h:171 / :179), spelled out so the
// summation order is the engine's and NoContraction can forbid the fma.
float sdfcMagSq(float3 v) { return v.x * v.x + v.y * v.y + v.z * v.z; }
float sdfcMagnitude(float3 v) { return sqrt(sdfcMagSq(v)); }

// C11: PxVec3::getNormalized (PxVec3.h:285).
float3 sdfcGetNormalized(float3 v)
{
    const float m = sdfcMagSq(v);
    return (m > 0.0f) ? (v * (1.0f / sqrt(m))) : float3(0.0f, 0.0f, 0.0f);
}

// PxVec3::dot (PxVec3.h:272).
float sdfcDot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// C1 + C2: localPos.maximum(sdfBoxLower).minimum(sdfBoxHigher).
float3 sdfcClampToBox(SparseSdfTexture t, float3 localPos)
{
    const float3 hi = float3(sdfcMax(localPos.x, t.sdfBoxLower.x),
                             sdfcMax(localPos.y, t.sdfBoxLower.y),
                             sdfcMax(localPos.z, t.sdfBoxLower.z));
    return float3(sdfcMin(hi.x, t.sdfBoxHigher.x),
                  sdfcMin(hi.y, t.sdfBoxHigher.y),
                  sdfcMin(hi.z, t.sdfBoxHigher.z));
}

// C4: a multiply by the stored reciprocal, not a divide by sdfDx.
float3 sdfcTexCoord(SparseSdfTexture t, float3 clampedGridPt)
{
    return (clampedGridPt - t.sdfBoxLower) * t.invSdfDx;
}

// ---------------------------------------------------------------------------
// The two shape-space helpers every caller of this layer needs.  They live here
// rather than beside their first user because P9.12f gave them a second one:
// pxgSlangSdfFind (the block gather) and pxgSlangSdfParticle (the contact body)
// both transform a world point into the SDF's vertex space, and a second copy
// would be a second thing to keep in step with cudaNpCommon.h.
// ---------------------------------------------------------------------------

// shape2Vertex, cudaNpCommon.h:284-296.  A per-component DIVIDE by the scale
// between rotate and rotateInv -- not a multiply by a reciprocal, and the two
// round differently.  vertex2ShapeNormalVector (:299) is THE SAME FUNCTION:
// a covariant vector transforms with the inverse scale, so the engine defines
// it as a call to this one.
float3 sdfShape2Vertex(float3 v, float3 scale, float4 rotation)
{
    return quatRotateInv(rotation, quatRotate(rotation, v) / scale);
}

// vertex2Shape (cudaNpCommon.h:270) -- the INVERSE of the above, and P9.34b's
// triangle gather needs it because the cloth's vertices arrive in vertex space and
// have to reach shape space before the a-to-b transform.  Note it is a component-wise
// MULTIPLY between rotate and rotateInv where shape2Vertex divides; the two are not
// each other's spelling with a reciprocal, and they do not round alike.
float3 sdfVertex2Shape(float3 v, float3 scale, float4 rotation)
{
    const float3 t = quatRotate(rotation, v);
    return quatRotateInv(rotation, float3(t.x * scale.x, t.y * scale.y, t.z * scale.z));
}

// PxTransform::transform(p) = q.rotate(p) + t.p.
float3 sdfTransformPoint(float3 tp, float4 tq, float3 v)
{
    return quatRotate(tq, v) + tp;
}

// ---------------------------------------------------------------------------
// PxSdfDistance, sdfCollision.cuh:156-166.
// ---------------------------------------------------------------------------
float sdfDistanceEx(SparseSdfTexture t, float3 localPos, out float3 clampedGridPt,
                    out float3 f, out float diffMag, out float sampleValue)
{
    clampedGridPt = sdfcClampToBox(t, localPos);
    diffMag = sdfcMagnitude(localPos - clampedGridPt);
    f = sdfcTexCoord(t, clampedGridPt);
    sampleValue = sdfTexSample(t, f);
    return sampleValue + diffMag;      // C3: ADDED
}

float sdfDistance(SparseSdfTexture t, float3 localPos)
{
    float3 clampedGridPt, f;
    float diffMag, sampleValue;
    return sdfDistanceEx(t, localPos, clampedGridPt, f, diffMag, sampleValue);
}

// ---------------------------------------------------------------------------
// PxVolumeGrad, sdfCollision.cuh:168-198.
// ---------------------------------------------------------------------------
// The six taps, reported so the fixture can check the composition against the
// values the shader actually read rather than against a second sampler.  On the
// INTERIOR path `coord` is the texel-space f handed to Sample(); on the
// fall-back path it is the world-space point handed to PxSdfDistance.  The two
// spaces are wildly different numbers, which is what makes C7's guard
// observable at a point where both paths would produce the same gradient.
struct SdfGradTrace
{
    float3 coord[6];
    float value[6];
    float3 f;          // the texel coordinate the guard was tested on
    uint interior;     // 1 == the central-difference path was taken
};

float3 sdfVolumeGradEx(SparseSdfTexture t, float3 localPos, out SdfGradTrace tr)
{
    float3 grad;

    const float3 clampedGridPt = sdfcClampToBox(t, localPos);
    const float3 f = sdfcTexCoord(t, clampedGridPt);
    tr.f = f;

    // C7: inclusive on both ends, ANDed over all three axes, against the FINE
    // dimensions.  The unsigned dimension promotes to float before the subtract.
    if (f.x >= 1.0f && f.x <= float(t.sdfDims.x) - 2.0f &&
        f.y >= 1.0f && f.y <= float(t.sdfDims.y) - 2.0f &&
        f.z >= 1.0f && f.z <= float(t.sdfDims.z) - 2.0f)
    {
        tr.interior = 1u;
        tr.coord[0] = float3(f.x + 1.0f, f.y, f.z);
        tr.coord[1] = float3(f.x - 1.0f, f.y, f.z);
        tr.coord[2] = float3(f.x, f.y + 1.0f, f.z);
        tr.coord[3] = float3(f.x, f.y - 1.0f, f.z);
        tr.coord[4] = float3(f.x, f.y, f.z + 1.0f);
        tr.coord[5] = float3(f.x, f.y, f.z - 1.0f);
        [ForceUnroll]
        for (uint i = 0; i < 6; ++i)
            tr.value[i] = sdfTexSample(t, tr.coord[i]);
        // C6: no division by the step.
        grad.x = tr.value[0] - tr.value[1];
        grad.y = tr.value[2] - tr.value[3];
        grad.z = tr.value[4] - tr.value[5];
        return grad;
    }

    tr.interior = 0u;
    tr.coord[0] = localPos + float3(t.sdfDx, 0.0f, 0.0f);
    tr.coord[1] = localPos - float3(t.sdfDx, 0.0f, 0.0f);
    tr.coord[2] = localPos + float3(0.0f, t.sdfDx, 0.0f);
    tr.coord[3] = localPos - float3(0.0f, t.sdfDx, 0.0f);
    tr.coord[4] = localPos + float3(0.0f, 0.0f, t.sdfDx);
    tr.coord[5] = localPos - float3(0.0f, 0.0f, t.sdfDx);
    [ForceUnroll]
    for (uint i = 0; i < 6; ++i)
        tr.value[i] = sdfDistance(t, tr.coord[i]);   // C5: the CLAMPING form
    grad.x = tr.value[0] - tr.value[1];
    grad.y = tr.value[2] - tr.value[3];
    grad.z = tr.value[4] - tr.value[5];
    return grad;
}

float3 sdfVolumeGrad(SparseSdfTexture t, float3 localPos)
{
    SdfGradTrace tr;
    return sdfVolumeGradEx(t, localPos, tr);
}

// ---------------------------------------------------------------------------
// PxSdfSampleWithGrad, sdfCollision.cuh:200-225.
// ---------------------------------------------------------------------------
// The CUDA's commented-out early return
//     const PxReal diffMag = diff.magnitudeSquared();
//     if (diffMag > tolerance*tolerance) return PX_MAX_F32;
// (:207-210) is NOT active and is NOT ported.  It is noted because it is the
// obvious "the tolerance must gate the distance to the box" reading, and the
// live code gates the SAMPLE VALUE instead.
float sdfSampleWithGradEx(SparseSdfTexture t, float3 localPos, float3 sdfGradient,
                          inout float3 dir, float tolerance,
                          out float dist, out float3 diff)
{
    const float3 clampedGridPt = sdfcClampToBox(t, localPos);
    diff = localPos - clampedGridPt;
    const float3 f = sdfcTexCoord(t, clampedGridPt);

    dist = sdfTexSample(t, f);           // C8: RAW, no diffMag

    if (dist < tolerance)
    {
        // C10: normalised again here.
        dir = sdfcGetNormalized(sdfcGetNormalized(sdfGradient) * abs(dist) + diff);
        return dist + sdfcDot(dir, diff);
    }

    // C9: `dir` is deliberately left alone on this path.
    return SDFC_MAX_F32;
}

float sdfSampleWithGrad(SparseSdfTexture t, float3 localPos, float3 sdfGradient,
                        inout float3 dir, float tolerance)
{
    float dist;
    float3 diff;
    return sdfSampleWithGradEx(t, localPos, sdfGradient, dir, tolerance, dist, diff);
}

// ---------------------------------------------------------------------------
// doVertexSDFCollision, sdfCollision.cuh:227-240.
// ---------------------------------------------------------------------------
float sdfDoVertexCollisionEx(SparseSdfTexture t, float3 v, inout float3 dir, float tolerance,
                             out SdfGradTrace tr, out float3 rawGrad, out float gradMag,
                             out float3 usedGrad, out float dist, out float3 diff)
{
    rawGrad = sdfVolumeGradEx(t, v, tr);

    // C12: PxVec3::normalize() -- in place, returning the MAGNITUDE, dividing
    // only when that magnitude is positive.
    float3 sdfGradient = rawGrad;
    gradMag = sdfcMagnitude(sdfGradient);
    if (gradMag > 0.0f)
        sdfGradient = sdfGradient * (1.0f / gradMag);

    // C14: not an else -- a NaN magnitude satisfies neither test.
    if (gradMag == 0.0f)
    {
        // We ran into a discontinuity e.g. the exact center of a cube.  C13: the
        // getNormalized is load-bearing -- the constant is 0.95284 long.
        sdfGradient = sdfcGetNormalized(SDFC_DISCONTINUITY_DIR);
    }

    usedGrad = sdfGradient;
    return sdfSampleWithGradEx(t, v, sdfGradient, dir, tolerance, dist, diff);
}

float sdfDoVertexCollision(SparseSdfTexture t, float3 v, inout float3 dir, float tolerance)
{
    SdfGradTrace tr;
    float3 rawGrad, usedGrad, diff;
    float gradMag, dist;
    return sdfDoVertexCollisionEx(t, v, dir, tolerance, tr, rawGrad, gradMag, usedGrad, dist, diff);
}


// ===========================================================================
// doTriangleSDFCollision, sdfCollision.cuh:402-494 — P9.34a.
//
// The whole numeric content of cloth_sdfMeshContactGenLaunch's per-triangle work:
// a 16-iteration PROJECTED GRADIENT DESCENT over a triangle's barycentric
// coordinates, looking for the point of the triangle that is deepest inside the
// SDF.  Everything it calls underneath — PxSdfDistance, PxVolumeGrad,
// PxSdfSampleWithGrad — is already here from P9.12d, and
// Gu::closestPtPointBaryTriangle came with P9.34a into pxgSlangTetClosest.
//
// THE ORACLE, and its limit.  Same arrangement as doVertexSDFCollision above: the
// primitives are GPU-only because they Sample() a texture, so no host mirror can
// predict them.  The Ex variant therefore TRACES the descent — every gradient it
// received, every uvw it produced, every step it used — and the fixture REPLAYS
// the iteration on the host from those traced gradients.  What that checks is the
// composition: the branch cascade, the step schedule, the clamp, the convergence
// test and where the loop stops.  What it does not check is the gradient values
// themselves, and it does not have to: those are P9.12c/d's claim, measured there.
//
// ---------------------------------------------------------------------------
// THE TRAPS
// ---------------------------------------------------------------------------
//  D1. THE STARTING ITERATE IS AN ASYMMETRIC CASCADE, and it is not "pick the
//      smallest".  Written out (:420-433):
//          d0 < d1 && d0 < d2 && d0 < dist -> v0
//          else if d1 < d2 && d1 < dist    -> v1
//          else if d2 < dist               -> v2
//          else                            -> the centroid, uvw = (1/3,1/3,1/3)
//      The second test does NOT compare against d0.  So with d0 < d1 but
//      d0 >= d2, the first arm fails and the second can still choose v1 even
//      though v0 is nearer.  Transcribed as written; a port that "fixes" it into a
//      three-way minimum changes the starting point and therefore the answer.
//
//  D2. `p` IS NOT ASSIGNED IN THE else ARM.  It keeps the centroid it was given
//      before the cascade, and only `uvw` is set there.  The three vertex arms set
//      both.
//
//  D3. `difference` IS A NESTED MAX OF SQUARED LENGTHS WITH ONE sqrt AT THE END
//      (:437) — not three lengths maximised.  `step = 1/(2*difference)`.
//
//  D4. `step` DECAYS EVERY ITERATION, and the decay happens BETWEEN the uvw update
//      and the clamp (:470).  A port that decays at the top of the loop, or after
//      the clamp, runs a different schedule from iteration two onwards.
//
//  D5. THE CONVERGENCE TEST USES THE CLAMPED newUVW, and `break` happens BEFORE
//      `uvw = newUVW` (:476-483).  So on the iteration that converges, `uvw` stays
//      stale — but `p` has already been recomputed from the clamped value, and `p`
//      is what leaves the function.  Copying the assignment above the test changes
//      nothing observable; moving the `p` update below it changes everything.
//
//  D6. THE DISCONTINUITY VECTOR IS A DIFFERENT ONE from doVertexSDFCollision's.
//      Here it is (0.571846586, 0.705545099, 0.418566116) (:455), a truncation of
//      the full-precision constant in GuContactMeshMesh.h:138; the vertex path uses
//      (0.5448762, 0.672269, 0.398825).  And unlike that one it is NOT normalised
//      before use — it is already unit to within a float, and the CUDA assigns it
//      raw.  Normalising it here would be a port-introduced change.
//
//  D7. THE GRADIENT IS RECOMPUTED AFTER THE LOOP (:487) rather than reusing the
//      last one the loop produced, and the `dist` computed from the centroid at the
//      top (:409) is DISCARDED — the returned distance comes only from the final
//      PxSdfSampleWithGrad.
//
//  D8. `normalize()` here is PxVec3::normalize (PxVec3.h:295), which returns the
//      MAGNITUDE and divides in place through operator/= — and operator/=
//      (PxVec3.h:260) is itself `f = 1.0/f; x *= f;`, a reciprocal multiply.  So it
//      is the same expression as getNormalized, and the guard is the only
//      difference: `m > 0` on the magnitude rather than on its square.
// ===========================================================================

static const float3 SDFC_TRI_DISCONTINUITY_DIR =
    float3(0.571846586f, 0.705545099f, 0.418566116f);   // D6

static const uint SDF_TRI_MAX_ITERS = 16u;

// One iteration of the descent, as the shader ran it.
struct SdfTriIterTrace
{
    float3 gradRaw;     // PxVolumeGrad before normalise
    float gradMag;      // what normalize() returned
    float3 gradUsed;    // after normalise / discontinuity substitution
    float3 newUvw;      // after closestPtPointBaryTriangle
    float stepIn;       // the step this iteration multiplied by
}

struct SdfTriTrace
{
    float d0, d1, d2, dCenter;
    uint startArm;      // 0..2 = that vertex, 3 = the centroid
    float3 uvw0;
    float3 p0;
    float difference;
    uint iters;         // iterations actually run (1..16)
    uint converged;     // 1 == the loop broke on the tolerance
    SdfTriIterTrace it[16];
    float3 pFinal;
    float3 gradFinal;   // the RECOMPUTED gradient of D7
}

float sdfDoTriangleCollisionEx(SparseSdfTexture t, float3 v0, float3 v1, float3 v2,
                                      out float3 point, inout float3 dir, float tolerance,
                                      out SdfTriTrace tr)
{
    const float third = 1.0f / 3.0f;
    const float3 center = (v0 + v1 + v2) * third;
    float3 p = center;

    const float dist = sdfDistance(t, p);
    const float d0 = sdfDistance(t, v0);
    const float d1 = sdfDistance(t, v1);
    const float d2 = sdfDistance(t, v2);
    tr.d0 = d0; tr.d1 = d1; tr.d2 = d2; tr.dCenter = dist;

    float3 uvw = float3(0.0f, 0.0f, 0.0f);

    // D1: the cascade, exactly as written.  D2: `p` keeps the centroid in the else.
    if (d0 < d1 && d0 < d2 && d0 < dist)      { p = v0; uvw.x = 1.0f; tr.startArm = 0u; }
    else if (d1 < d2 && d1 < dist)            { p = v1; uvw.y = 1.0f; tr.startArm = 1u; }
    else if (d2 < dist)                       { p = v2; uvw.z = 1.0f; tr.startArm = 2u; }
    else                                      { uvw = float3(third, third, third); tr.startArm = 3u; }
    tr.uvw0 = uvw;
    tr.p0 = p;

    // D3: nested max of SQUARED lengths, one sqrt.
    const float3 e0 = v0 - p, e1 = v1 - p, e2 = v2 - p;
    const float difference = sqrt(max(dot(e0, e0), max(dot(e1, e1), dot(e2, e2))));
    tr.difference = difference;

    const float toleranceSq = 1e-3f * 1e-3f;

    float3 sdfGradient = float3(0.0f, 0.0f, 0.0f);
    float step = 1.0f / (2.0f * difference);

    uint iters = 0u, converged = 0u;
    for (uint iter = 0u; iter < SDF_TRI_MAX_ITERS; ++iter)
    {
        SdfTriIterTrace itr;
        itr.stepIn = step;

        sdfGradient = sdfVolumeGrad(t, p);
        itr.gradRaw = sdfGradient;

        // D8: PxVec3::normalize() -- returns the magnitude, divides only when positive.
        const float gm = sdfcMagnitude(sdfGradient);
        itr.gradMag = gm;
        if (gm > 0.0f)
            sdfGradient = sdfGradient * (1.0f / gm);
        if (gm == 0.0f)
            sdfGradient = SDFC_TRI_DISCONTINUITY_DIR;   // D6: used RAW, not normalised
        itr.gradUsed = sdfGradient;

        const float dfdu = sdfcDot(sdfGradient, v0 - p);
        const float dfdv = sdfcDot(sdfGradient, v1 - p);
        const float dfdw = sdfcDot(sdfGradient, v2 - p);

        float3 newUVW = uvw;
        newUVW.x -= step * dfdu;
        newUVW.y -= step * dfdv;
        newUVW.z -= step * dfdw;

        step = step * 0.8f;                              // D4: between the update and the clamp

        newUVW = closestPtPointBaryTriangle(newUVW);
        itr.newUvw = newUVW;

        p = v0 * newUVW.x + v1 * newUVW.y + v2 * newUVW.z;

        tr.it[iter] = itr;
        ++iters;

        // D5: the test uses the CLAMPED newUVW, and the break precedes the assignment.
        const float3 duvw = uvw - newUVW;
        if (sdfcDot(duvw, duvw) < toleranceSq)
        {
            converged = 1u;
            break;
        }

        uvw = newUVW;
    }
    tr.iters = iters;
    tr.converged = converged;

    // D7: recomputed, and the centroid distance from the top is discarded.
    sdfGradient = sdfVolumeGrad(t, p);
    tr.gradFinal = sdfGradient;

    point = p;
    tr.pFinal = p;

    return sdfSampleWithGrad(t, p, sdfGradient, dir, tolerance);
}

float sdfDoTriangleCollision(SparseSdfTexture t, float3 v0, float3 v1, float3 v2,
                                    out float3 point, inout float3 dir, float tolerance)
{
    SdfTriTrace tr;
    return sdfDoTriangleCollisionEx(t, v0, v1, v2, point, dir, tolerance, tr);
}

// ---------------------------------------------------------------------------
// doTetrahedronSDFCollision, sdfCollision.cuh:286-393 -- P9.35a.
//
// The four-vertex sibling of doTriangleSDFCollision above, and the two are close
// enough that the DIFFERENCES are the whole content of this header.  Same trace
// arrangement, same reason: the descent samples a texture, so there is nothing on
// the host to call; the shader logs each iteration and the fixture replays it.
//
// WHAT MATCHES the triangle version: the asymmetric start cascade, `difference` as
// a nested max of SQUARES with one sqrt, `step = 1/(2*difference)` decaying by 0.8
// BETWEEN the uvw update and the clamp, the convergence test on the post-clamp
// value with the break before the assignment, and the gradient resampled after the
// loop and passed RAW to PxSdfSampleWithGrad.
//
// WHAT DIFFERS, and every one of these is a way to get it wrong:
//
//  E1. THE CASCADE IS FOUR-WAY and asymmetric in the same style: arm 2 compares
//      `d1` against d2, d3 and dist but NOT against d0; arm 3 compares d2 against
//      d3 and dist only; arm 4 compares d3 against dist only.  A port that turns
//      this into a four-way minimum starts somewhere else.
//
//  E2. THE ELSE ARM SETS uvw EXPLICITLY to PxVec4(0.25f).  The triangle version
//      leaves `p` at the centroid and fills uvw with thirds; here the assignment is
//      written out, and `p` is likewise left at the centroid.
//
//  E3. THE CLAMP IS CONDITIONAL.  The triangle version calls its projection every
//      iteration; this one only calls closestPtPointTetrahedron when some component
//      escapes [-eps, 1+eps] -- and the escape test is `<= -eps || >= 1+eps`, a
//      four-way disjunction with NON-STRICT comparisons on both sides.  Making it
//      unconditional is a different kernel, and so is making either end strict.
//
//  E4. `newUVW.w` IS RECOVERED, not clamped.  closestPtPointTetrahedron returns a
//      three-vector; the fourth barycentric is rebuilt as `1 - cp.x - cp.y - cp.z`,
//      in that association.  Upstream even flags it: "TODO: Veryfy this logic".
//
//  E5. `p` IS REBUILT FROM newUVW WHETHER OR NOT THE CLAMP RAN, so on an iteration
//      that stays inside the simplex the point comes from the raw gradient step.
//
//  E6. THE CONVERGENCE TEST IS A FOUR-COMPONENT MAGNITUDE.  `(uvw - newUVW)` is a
//      PxVec4 and its magnitudeSquared includes w; dropping w makes the loop stop
//      earlier on inputs whose w is the only thing still moving.
//
//  E7. THE DISCONTINUITY VECTOR IS THE **VERTEX** PATH'S, AND IT IS NORMALISED.
//      doTriangleSDFCollision uses SDFC_TRI_DISCONTINUITY_DIR raw, because that
//      constant is exactly unit in float32; this one uses SDFC_DISCONTINUITY_DIR
//      through getNormalized, and there the normalise is load-bearing -- P9.34a
//      measured that constant at 0.90790 squared, i.e. 0.95284 long.  Two
//      constants, two roles, and now both roles appear in the same module.
//
//  E8. THE CLAMP IS THE FIVE-ARGUMENT Gu::closestPtPointTetrahedron, applied to
//      the CANONICAL SIMPLEX (1,0,0), (0,1,0), (0,0,1), (0,0,0) -- not the actual
//      tetrahedron.  It projects the barycentric triple, not the point.
//
// ---------------------------------------------------------------------------
// TWO THINGS THE SWEEP LEFT STANDING, and what each of them turned out to be.
//
//  * `max(max(a,b), max(c,d))` against a left fold is an EQUIVALENT, not a gap.
//    `max` on floats is a SELECTION -- it returns one of its arguments unchanged,
//    so no rounding enters -- and it is associative and commutative on non-NaN
//    inputs.  Both spellings therefore return the same element, bit for bit.  NaN
//    cannot arise here: the four arguments are `dot(e, e)` of finite differences,
//    and the degenerate tetrahedron (all four corners equal) gives four exact
//    zeros rather than NaN.  Transcribed in upstream's shape anyway.
//
//  * `<=` / `>=` against `<` / `>` IN E3's ESCAPE TEST is a NAMED GAP, and the
//    condition is written out here so the next pass can build it rather than
//    rediscover it.  The two differ only when a component is EXACTLY -1e-4f or
//    EXACTLY 1.0f + 1e-4f, and the fixture cannot manufacture that.
//    It manufactures the neighbouring boundary and does so on purpose -- the
//    perpendicular tetrahedra put a component at exactly 1.0, which is what makes
//    the SIGN of eps observable -- but exactly ±eps is out of reach, and the
//    reason is specific rather than general.  The two boundaries that P9.34b and
//    P9.34c DID manufacture were each reachable because a free host-side
//    parameter mapped monotonically onto the compared value: a gather tolerance
//    there, a contact distance here.  This comparison has no such handle.  The
//    compared value is `uvw - step * (grad . (v_i - p))`, in which `grad` is a
//    GPU-only texture sample; the only host-controllable factors are the
//    vertices, and they move `step` and the dot product together.  A bisection
//    over the direction converges to where the product crosses 1e-4f but is not
//    guaranteed to LAND on that float.  So: a real difference, on a measure-zero
//    input, unreached -- recorded, not excused.
// ---------------------------------------------------------------------------
static const uint SDF_TET_MAX_ITERS = 16u;

struct SdfTetIterTrace
{
    float3 gradRaw;     // PxVolumeGrad before normalise
    float gradMag;      // what normalize() returned
    float3 gradUsed;    // after normalise / discontinuity substitution
    float4 newUvw;      // after the CONDITIONAL clamp
    float stepIn;       // the step this iteration multiplied by
    uint clamped;       // 1 == the escape test fired and the projection ran
}

struct SdfTetTrace
{
    float d0, d1, d2, d3, dCenter;
    uint startArm;      // 0..3 = that vertex, 4 = the centroid
    float4 uvw0;
    float3 p0;
    float difference;
    uint iters;         // iterations actually run (1..16)
    uint converged;     // 1 == the loop broke on the tolerance
    SdfTetIterTrace it[16];
    float3 pFinal;
    float3 gradFinal;   // the RECOMPUTED gradient
}

float sdfDoTetrahedronCollisionEx(SparseSdfTexture t, float3 v0, float3 v1, float3 v2, float3 v3,
                                  out float3 point, inout float3 dir, float tolerance,
                                  out SdfTetTrace tr)
{
    const float3 center = (v0 + v1 + v2 + v3) * 0.25f;
    float3 p = center;

    const float distC = sdfDistance(t, p);
    const float d0 = sdfDistance(t, v0);
    const float d1 = sdfDistance(t, v1);
    const float d2 = sdfDistance(t, v2);
    const float d3 = sdfDistance(t, v3);
    tr.d0 = d0; tr.d1 = d1; tr.d2 = d2; tr.d3 = d3; tr.dCenter = distC;

    float4 uvw = float4(0.0f, 0.0f, 0.0f, 0.0f);

    // E1 / E2.  `dist` is shadowed upstream by the winning value; nothing after the
    // cascade reads it, so only `p` and `uvw` are carried.
    if (d0 < d1 && d0 < d2 && d0 < d3 && d0 < distC) { p = v0; uvw.x = 1.0f; tr.startArm = 0u; }
    else if (d1 < d2 && d1 < d3 && d1 < distC)       { p = v1; uvw.y = 1.0f; tr.startArm = 1u; }
    else if (d2 < d3 && d2 < distC)                  { p = v2; uvw.z = 1.0f; tr.startArm = 2u; }
    else if (d3 < distC)                             { p = v3; uvw.w = 1.0f; tr.startArm = 3u; }
    else                                             { uvw = float4(0.25f, 0.25f, 0.25f, 0.25f);
                                                       tr.startArm = 4u; }
    tr.uvw0 = uvw;
    tr.p0 = p;

    // Nested max of SQUARED lengths, one sqrt -- and the nesting is
    // max(max(e0,e1), max(e2,e3)), not a left fold.
    const float3 e0 = v0 - p, e1 = v1 - p, e2 = v2 - p, e3 = v3 - p;
    const float difference = sqrt(max(max(dot(e0, e0), dot(e1, e1)),
                                      max(dot(e2, e2), dot(e3, e3))));
    tr.difference = difference;

    const float eps = 1e-4f;
    const float toleranceSq = 1e-3f * 1e-3f;

    float3 sdfGradient = float3(0.0f, 0.0f, 0.0f);
    float step = 1.0f / (2.0f * difference);

    uint iters = 0u, converged = 0u;
    for (uint iter = 0u; iter < SDF_TET_MAX_ITERS; ++iter)
    {
        SdfTetIterTrace itr;
        itr.stepIn = step;

        sdfGradient = sdfVolumeGrad(t, p);
        itr.gradRaw = sdfGradient;

        // PxVec3::normalize() -- returns the magnitude, divides only when positive.
        const float gm = sdfcMagnitude(sdfGradient);
        itr.gradMag = gm;
        if (gm > 0.0f)
            sdfGradient = sdfGradient * (1.0f / gm);
        if (gm == 0.0f)
            sdfGradient = sdfcGetNormalized(SDFC_DISCONTINUITY_DIR);   // E7
        itr.gradUsed = sdfGradient;

        const float dfdu = sdfcDot(sdfGradient, v0 - p);
        const float dfdv = sdfcDot(sdfGradient, v1 - p);
        const float dfdw = sdfcDot(sdfGradient, v2 - p);
        const float dfd4 = sdfcDot(sdfGradient, v3 - p);

        float4 newUVW = uvw;
        newUVW.x -= step * dfdu;
        newUVW.y -= step * dfdv;
        newUVW.z -= step * dfdw;
        newUVW.w -= step * dfd4;

        step = step * 0.8f;                              // between the update and the clamp

        // E3: CONDITIONAL, and non-strict at both ends.
        uint clamped = 0u;
        if ((newUVW.x <= -eps || newUVW.x >= 1.0f + eps) ||
            (newUVW.y <= -eps || newUVW.y >= 1.0f + eps) ||
            (newUVW.z <= -eps || newUVW.z >= 1.0f + eps) ||
            (newUVW.w <= -eps || newUVW.w >= 1.0f + eps))
        {
            // E8: the CANONICAL simplex, not the real tetrahedron.
            const float3 cp = guClosestPtPointTetrahedron(newUVW.xyz,
                                                          float3(1.0f, 0.0f, 0.0f),
                                                          float3(0.0f, 1.0f, 0.0f),
                                                          float3(0.0f, 0.0f, 1.0f),
                                                          float3(0.0f, 0.0f, 0.0f));
            newUVW.x = cp.x;
            newUVW.y = cp.y;
            newUVW.z = cp.z;
            newUVW.w = 1.0f - cp.x - cp.y - cp.z;         // E4, in that association
            clamped = 1u;
        }
        itr.newUvw = newUVW;
        itr.clamped = clamped;

        // E5: rebuilt either way.
        p = v0 * newUVW.x + v1 * newUVW.y + v2 * newUVW.z + v3 * newUVW.w;

        tr.it[iter] = itr;
        ++iters;

        // E6: FOUR components, and the break precedes the assignment.
        const float4 duvw = uvw - newUVW;
        const float dsq = duvw.x * duvw.x + duvw.y * duvw.y +
                          duvw.z * duvw.z + duvw.w * duvw.w;
        if (dsq < toleranceSq)
        {
            converged = 1u;
            break;
        }

        uvw = newUVW;
    }
    tr.iters = iters;
    tr.converged = converged;

    // Recomputed, and passed RAW.
    sdfGradient = sdfVolumeGrad(t, p);
    tr.gradFinal = sdfGradient;

    point = p;
    tr.pFinal = p;

    return sdfSampleWithGrad(t, p, sdfGradient, dir, tolerance);
}

float sdfDoTetrahedronCollision(SparseSdfTexture t, float3 v0, float3 v1, float3 v2, float3 v3,
                                out float3 point, inout float3 dir, float tolerance)
{
    SdfTetTrace tr;
    return sdfDoTetrahedronCollisionEx(t, v0, v1, v2, v3, point, dir, tolerance, tr);
}
