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

// pxgSlangTriTriGeom -- P9.83b-8 (1/n).
//
//   The geometry helpers doTriangleTriangleCollision (convexMeshMidphase.cu) stands
//   on and no module had yet:
//
//     the recursive triangle subdivision   GuTriangleRefinement.h:43-117
//     getTriangleVertices WITH a sub-index sdfCollision.cuh:522-550
//     triangleRadiusSquared                sdfCollision.cuh:729-734
//     sdfRadiusSquared                     sdfCollision.cuh:736-739
//     PxTransform::transformInv / getInverse  PxTransform.h:114-118, :134-138
//
// The SDF modules already have vertex2Shape and PxTransform::transform
// (sdfVertex2Shape, sdfTransformPoint in pxgSlangSdfCollision); this file uses
// them rather than growing a third copy.  The gather in pxgSlangSdfFind calls
// subdivision dead ("T3") -- true for the cloth, whose call site compiles
// SUPPORT_SUBDIVISION to 0 (femClothPrimitives.cu:2306), and FALSE for this
// kernel, where a mesh without an SDF sets needsRefinement and the next gather
// must read the refined sub-triangles back.  Everything here is what that path
// needs.
//
// getVertex (sdfCollision.cuh:553-556) is NOT here, although it sits between the
// two functions this file does transcribe.  Its only caller is the
// `if (useVertices)` arm of findInterestingTrianglesA (:655), and every
// instantiation -- cu:1072, cu:1077, femClothPrimitives.cu:2434 -- takes the
// default useVertices = false (:619).  No engine path executes it, so it is not
// ported (the first draft of this file carried it, with a fixture column; the
// P9.83b-8 review found the column certified a function nothing calls).
//
// ---------------------------------------------------------------------------
// THE SUBDIVISION ENCODING, AND WHAT ITS COMMENT GETS WRONG
// ---------------------------------------------------------------------------
// encodeSubdivisionId puts the level at `<< (30 - 4)` = << 26 and says "4 bits for
// subdivision level, the rest for the index".  The shift gives the level SIX bits
// (26..31) and the index 26.  Nothing depends on the comment; the transcription
// follows the shift.
//
// The index at level L uses 2L bits, so levels past 13 overflow into the level
// field, and getSubTriangle shifts by 2*(L-1): past L = 16 that is a shift of 32
// or more, undefined in C++ and in SPIR-V alike.  Both are unreachable here --
// addToRefinementBuffer refuses to refine at `subdivisionLevel >= maxRefinementLevel`
// (sdfCollision.cuh:747) and the kernel sets maxRefinementLevel = 8
// (convexMeshMidphase.cu:1059), so an elevated id never exceeds level 8 and a
// 16-bit index.  The fixture stays within levels 0..13, where every bit is
// defined, and says so.
//
// `if (encodedSubIndex > 0)` in getTriangleVertices (:529, :548) is kept although
// it changes nothing: decode(0) is level 0, and getSubTriangle runs zero steps.
// A level-1 id with index 0 encodes to 1 << 26, not 0, so no refined triangle is
// ever mistaken for an unrefined one.
//
// ---------------------------------------------------------------------------
// ORDER OF OPERATIONS, which is the whole difficulty
// ---------------------------------------------------------------------------
// getTriangleVertices applies vertex2Shape, THEN the a-to-b transform, THEN the
// subdivision: the midpoints are taken in the OTHER mesh's shape space, not in
// the triangle's own vertex space.  Subdividing first and transforming after
// gives the same triangle only in exact arithmetic.  And the midpoint is
// `0.5f * (a + b)` -- add first, then halve (GuTriangleRefinement.h:46-48).  The
// distributed spelling 0.5f*a + 0.5f*b is the same number for every input whose
// sum does not overflow (halving a normal float is exact), so the ONLY witness is
// a + b past FLT_MAX: the engine gives 0.5f * inf = inf, the distributed form a
// finite value.  The fixture's +-3e38 corners exist for that and nothing else.
//
// triangleRadiusSquared is not what its name says.  It returns the largest
// squared EDGE length, nested as PxMax(PxMax(|v1-v0|^2, |v2-v1|^2), |v0-v2|^2), and
// the `center` it computes first is never used.  PxMax(float) is
// intrinsics::selectMax, `a > b ? a : b` (PxUnixMathIntrinsics.h:122-125) -- NOT
// GLSL FMax, which is undefined for NaN; the ternary returns b when a is NaN.
// Of the ways to misspell that nesting, the OPERAND ORDER is observable (a NaN
// edge is swallowed or propagated depending on its slot) and the RE-ASSOCIATION
// PxMax(e01, PxMax(e12, e20)) is not: selectMax only fails to associate when the
// middle operand is NaN, and a NaN e12 forces e01 NaN, e20 NaN, or e01 = e20 = inf,
// on each of which both spellings agree.  Nor is swapping the INNER pair,
// PxMax(e12, e01): the two orders differ only when exactly one is NaN, and a NaN
// e01 beside a finite e12 forces e20 NaN or e12 = e20 = inf (a NaN e12 beside a
// finite e01, e20 NaN or e01 = e20 = inf) -- again both spellings agree.  What
// the fixture CAN see is which edge takes the OUTER slot.  Both equivalences are
// asserted per case, and were brute-forced over 10^6 two-component triangles.
//
// magnitudeSquared is spelled x*x + y*y + z*z, not dot(v, v).  SPIR-V leaves
// OpDot's summation order to the driver; the spelled-out sum pins the engine's
// left-to-right order on any driver.  On llvmpipe the two give the same bits: the
// stamp decorates OpDot too (spv_nocontract.py:35, since P9.1d), and the stamped
// OpDot is summed left to right without a fused step -- the `mag-dot` mutant
// survived every case of the fixture.  That makes it an equivalent HERE, a
// property of this driver and not of SPIR-V.  It does not overturn the
// RUNTIME_STATUS dot()/cross() rule, which was measured on another module; it
// says only that a stamped dot(v, v) was not fused on this build.  cross() is
// GLSL.std.450 Cross, an OpExtInst the pass does not decorate at all.

import pxgSlangCommon;         // quatRotate / quatRotateInv / quatConjugate
import pxgSlangSdfCollision;   // sdfVertex2Shape, sdfTransformPoint

// ---------------------------------------------------------------------------
// GuTriangleRefinement.h
// ---------------------------------------------------------------------------
public static const uint TTG_SUBDIVISION_SHIFT = 30u - 4u;   // :88, :94

// getSubTriangle4 (:43-72).  id is always `local & 3` here, so the CUDA's
// `default: PX_ASSERT(false)` is unreachable and has no transcription.
public void ttgGetSubTriangle4(uint id, inout float3 baryA, inout float3 baryB, inout float3 baryC)
{
    const float3 ab = 0.5f * (baryA + baryB);
    const float3 bc = 0.5f * (baryB + baryC);
    const float3 ca = 0.5f * (baryC + baryA);

    if (id == 0u)
    {
        baryB = ab;
        baryC = ca;
    }
    else if (id == 1u)
    {
        baryA = ab;
        baryC = bc;
    }
    else if (id == 2u)
    {
        baryA = ca;
        baryB = bc;
    }
    else
    {
        baryA = ab;
        baryB = bc;
        baryC = ca;
    }
}

// getSubTriangle (:74-83): the most significant 2-bit digit first.
public void ttgGetSubTriangle(inout float3 a, inout float3 b, inout float3 c, uint id, uint numSubdivisionSteps)
{
    for (uint i = 0u; i < numSubdivisionSteps; ++i)
    {
        const uint j = numSubdivisionSteps - i - 1u;
        uint local = id >> (2u * j);
        local = local & 3u;
        ttgGetSubTriangle4(local, a, b, c);
    }
}

// encodeSubdivisionId (:85-90)
public uint ttgEncodeSubdivisionId(uint subdivisionLevel, uint subTriangleIndex)
{
    return (subdivisionLevel << TTG_SUBDIVISION_SHIFT) | subTriangleIndex;
}

// decodeSubdivisionId (:92-97)
public void ttgDecodeSubdivisionId(uint encodedId, out uint subdivisionLevel, out uint subTriangleIndex)
{
    subdivisionLevel = encodedId >> TTG_SUBDIVISION_SHIFT;
    subTriangleIndex = encodedId & ((1u << TTG_SUBDIVISION_SHIFT) - 1u);
}

// elevateSubdivisionId (:99-109): one level deeper, child `subTriangle` (0..3).
public uint ttgElevateSubdivisionId(uint encodedId, uint subTriangle)
{
    uint subdivisionLevel;
    uint subTriangleIndex;
    ttgDecodeSubdivisionId(encodedId, subdivisionLevel, subTriangleIndex);
    return ttgEncodeSubdivisionId(subdivisionLevel + 1u, 4u * subTriangleIndex + subTriangle);
}

// getSubTriangleEncoded (:111-117)
public void ttgGetSubTriangleEncoded(inout float3 a, inout float3 b, inout float3 c, uint encodedSubIndex)
{
    uint subdivisionLevel;
    uint subTriangleIndex;
    ttgDecodeSubdivisionId(encodedSubIndex, subdivisionLevel, subTriangleIndex);
    ttgGetSubTriangle(a, b, c, subTriangleIndex, subdivisionLevel);
}

// ---------------------------------------------------------------------------
// sdfCollision.cuh
// ---------------------------------------------------------------------------

// getTriangleVertices (:522-531, and the mesh-struct form :540-550, which is the
// same body).  `meshInd` is the uint4-per-triangle index array, `meshVerts` the
// float4-per-vertex array; the kernel reaches both through the mesh blob.
public void ttgGetTriangleVertices(uint64_t meshInd, uint64_t meshVerts, float3 scale, float4 scaleRot,
                                   float3 tp, float4 tq, uint triangleIndex, uint encodedSubIndex,
                                   out float3 a, out float3 b, out float3 c)
{
    const uint4 inds = Ptr<uint4>(meshInd)[triangleIndex];
    Ptr<float4> v = Ptr<float4>(meshVerts);
    a = sdfTransformPoint(tp, tq, sdfVertex2Shape(v[inds.x].xyz, scale, scaleRot));
    b = sdfTransformPoint(tp, tq, sdfVertex2Shape(v[inds.y].xyz, scale, scaleRot));
    c = sdfTransformPoint(tp, tq, sdfVertex2Shape(v[inds.z].xyz, scale, scaleRot));

    if (encodedSubIndex > 0u)
        ttgGetSubTriangleEncoded(a, b, c, encodedSubIndex);
}

public float ttgMagnitudeSquared(float3 v) { return v.x * v.x + v.y * v.y + v.z * v.z; }

// PxMax(float) = intrinsics::selectMax (PxUnixMathIntrinsics.h:122-125)
public float ttgPxMax(float a, float b) { return a > b ? a : b; }

// triangleRadiusSquared (:729-734).  The dead `center` is not transcribed: it has
// no effect and nothing reads it.
public float ttgTriangleRadiusSquared(float3 v0, float3 v1, float3 v2)
{
    return ttgPxMax(ttgPxMax(ttgMagnitudeSquared(v1 - v0), ttgMagnitudeSquared(v2 - v1)),
                    ttgMagnitudeSquared(v0 - v2));
}

// sdfRadiusSquared (:736-739): the squared DIAGONAL of the SDF box, not a radius.
public float ttgSdfRadiusSquared(float3 sdfBoxLower, float3 sdfBoxHigher)
{
    return ttgMagnitudeSquared(sdfBoxHigher - sdfBoxLower);
}

// ---------------------------------------------------------------------------
// PxTransform.h -- the two inverse spellings the kernel uses, kept apart
// ---------------------------------------------------------------------------
// The pass-1 gather is handed a materialised aToB.getInverse() (cu:1078), and the
// pass-1 contact body calls aToB.transformInv(v) (cu:1165-1167).  They are not the
// same arithmetic: getInverse().transform(v) = q*.rotate(v) + q.rotateInv(-p),
// transformInv(v) = q.rotateInv(v - p).  A port that folded one into the other
// would be exact only in exact arithmetic.

// transformInv (:134-138)
public float3 ttgTransformInv(float3 tp, float4 tq, float3 v)
{
    return quatRotateInv(tq, v - tp);
}

// getInverse (:114-118): PxTransformT(q.rotateInv(-p), q.getConjugate()).
public void ttgGetInverse(float3 tp, float4 tq, out float3 invP, out float4 invQ)
{
    invP = quatRotateInv(tq, -tp);
    invQ = quatConjugate(tq);
}
