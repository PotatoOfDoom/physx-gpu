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
//      `a < b ? b : a` and `a < b ? a : b` (PxMath.h) -- not min()/max()
//      intrinsics.  They differ on NaN (the ternary returns the SECOND operand
//      for PxMax(NaN, b) ... i.e. b, since NaN < b is false gives a) and on
//      signed zero.  npSdfBoxDist.slang made the same call in Phase 1.
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

// PX_MAX_F32, PxSimpleTypes.h:94.
static const float SDFC_MAX_F32 = 3.4028234663852885981170418348452e+38f;

// C13: the arbitrary unit direction the CUDA falls back on at a discontinuity.
static const float3 SDFC_DISCONTINUITY_DIR = float3(0.5448762f, 0.672269f, 0.398825f);

// C2: PxMin / PxMax as PxMath.h spells them.
float sdfcMin(float a, float b) { return a < b ? a : b; }
float sdfcMax(float a, float b) { return a < b ? b : a; }

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
