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

// pxgSlangPointTriangle.slang -- the NINE-ARGUMENT
// `distancePointTriangleSquared`, sphereTriangle.cuh:41-140, in ONE place.
// P9.56.
//
// ===========================================================================
// WHY THIS MODULE EXISTS: THREE COPIES, AND THREE DEFECTS THAT LANDED IN ALL
// THREE
// ===========================================================================
//
// This function was transcribed three times -- into npTriangle.slang,
// npCvxMeshNp.slang and npCvxHfNp.slang -- and every defect found in it since
// has had to be fixed three times:
//
//   P9.36  changed the face region's ASSOCIATION in all three, to the tail of a
//          DIFFERENT upstream function (Gu::closestPtPointTriangle2), on the
//          warrant that "eighteen of the twenty-one copies in the corpus already
//          spell it this way".  Two of the three were correct before that change.
//   P9.53  withdrew it, in all three, and rewrote all three host mirrors from the
//          .cu.  `gpuslang.npCvxHfNp` went red on 123 cases in the process -- the
//          same 123 P9.36 had recorded and answered by changing the mirror.
//   P9.55  found `edgeTol` computed in DOUBLE upstream (sphereTriangle.cuh:96's
//          `1e-5` carries no suffix) and float in the ports -- again all three,
//          again three fixes and three mirror fixes.
//
// AND THE THREE HAD ALREADY DRIFTED APART ON THEIR OWN.  npCvxHfNp spells the dot
// product explicitly as `a.x*b.x + a.y*b.y + a.z*b.z`, with a comment (tranche 54)
// saying it is written that way "so the FMA contraction matches the CPU reference
// bit-for-bit".  npCvxMeshNp and npTriangle call the builtin `dot()`, which is
// `OpDot` -- a single SPIR-V instruction whose internal summation order and fusion
// the spec does not pin, and which `NoContraction` does not pin either (that
// decoration forbids fusing an op with its NEIGHBOURS, not the reduction inside
// one).  So the tranche-54 rule was written down once and applied in ONE file out
// of three -- measured, not guessed: moving both onto this module removed 13
// OpDots from each (80 -> 67 in npCvxMeshNp, 16 -> 3 in npTriangle).
//
// Both fixtures stayed bit-exact and green across that change, so on llvmpipe
// `OpDot` and the explicit form agree on these inputs today.  That is a platform
// observation, not a guarantee, and it is exactly what the tranche-54 comment was
// written to avoid relying on.
//
// P9.21 consolidated nine copies of `find()` for the same reason and set the
// precedent.  This is the same shape with a worse record: nine copies that never
// diverged, against three that diverged three times in three tranches.
//
// ===========================================================================
// WHAT THIS IS *NOT*: THE OTHER, SIMILAR-LOOKING FUNCTION
// ===========================================================================
//
// There are TWO closest-point-on-triangle implementations in this engine and
// they are NOT interchangeable:
//
//   (A) THIS ONE.  sphereTriangle.cuh:41-140, nine arguments, hand-written for
//       the GPU narrowphase.  STRICT comparisons (`d1 < 0 && d2 < 0`,
//       `d3 > 0 && d4 < d3`, `vc < 0.0f`), a uint4 adjacency, a Voronoi mask,
//       `generateContact`/`faceContact`, the `edgeTol` concavity relaxation, and
//       a face region that SUMS the two scaled edge vectors and scales the sum
//       ONCE.  It tests V0, V1, V2 and only then E01, E02, E12, because
//       `edgeTol` needs va+vb+vc up front.
//
//   (B) `Gu::closestPtPointTriangle2`, GuDistancePointTriangle.h:44-97, which the
//       5-argument `distancePointTriangleSquared` in deformableCollision.cuh:223
//       delegates to.  NON-STRICT comparisons, no adjacency, no mask, no
//       `generateContact`, no `edgeTol`, an interleaved branch order, and a face
//       region that scales the two SCALARS first.
//
// Eighteen other Slang copies in this corpus port (B) and are right to spell it
// (B)'s way.  Anything importing THIS module is porting (A).  If you find
// yourself reaching for a majority spelling to settle a question here, that is
// the mistake P9.36 made: a majority spelling is not evidence about a different
// function.

module pxgSlangPointTriangle;

// triangle.cuh:35
public static const uint PT_BOUNDARY = 0xffffffffu;
// GuCookingGrbTriangleMesh.h:56 -- set by the MESH cooker only.
public static const uint PT_NONCONVEX_FLAG = 0x80000000u;

// ConvexTriIntermediateData::VoronoiState (convexNpCommon.h:118-127).  NOTE that
// the enum declares eE02 BEFORE eE12 -- 5<<29 and 6<<29 -- which is not the order
// a reader expects.
public static const uint PT_EV0  = 1u << 29;
public static const uint PT_EV1  = 2u << 29;
public static const uint PT_EV2  = 3u << 29;
public static const uint PT_EE01 = 4u << 29;
public static const uint PT_EE02 = 5u << 29;
public static const uint PT_EE12 = 6u << 29;

// NOT `[ForceInline]`.  The originals were plain functions and Slang decided for
// itself; force-inlining them replicated the arithmetic at every call site and
// moved 42 OpFMul and 28 OpFAdd in the compiled module.  That is value-preserving
// under NoContraction -- inlining straight-line IEEE arithmetic changes nothing --
// but it makes the consolidation impossible to CHECK as inert, and the point of
// this module is that it changes nothing.
//
// EXPLICIT dot and cross, never the builtins.  `dot()` lowers to `OpDot`, whose
// internal summation order and fusion SPIR-V leaves to the implementation, and
// which `NoContraction` does not constrain -- that decoration forbids fusing an
// operation with its neighbours, not the reduction inside a single one.  Written
// out, the three multiplies and two adds are ordinary ops that the NoContraction
// stamp does pin.  (Tranche 54 established this; P9.56 is where it stops being
// true in two files out of three.)
public float ptDot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

public float3 ptCross(float3 a, float3 b)
{
    return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

// PxVec3::getNormalized (PxVec3.h:286-290) -- the ZERO VECTOR for a zero-magnitude
// input, and ONE reciprocal with three multiplies, never a per-component divide.
public float3 ptNormalized(float3 v)
{
    const float m = ptDot(v, v);
    return m > 0.0f ? v * (1.0f / sqrt(m)) : float3(0, 0, 0);
}

// triangle.cuh: an adjacency word is BOUNDARY, or a triangle index whose bit 31
// the mesh cooker sets when it found the shared edge non-convex.
public bool ptIsEdgeNonconvex(uint adjEdgeIndex)
{
    return (adjEdgeIndex != PT_BOUNDARY) && ((adjEdgeIndex & PT_NONCONVEX_FLAG) != 0u);
}

// sphereTriangle.cuh:41-140, transcribed.
//
// `adj` is (x = edge 01, y = edge 12, z = edge 02) and ONLY .xyz are read -- the
// callers build it as an uninitialised `uint4` whose .w getTriangle never writes
// (convexHeightfield.cu:406 against heightfieldUtil.cuh:297-299, which assigns
// x, y and z only).  A port that reads .w reads a garbage register.
public float ptDistancePointTriangleSquared(float3 p, float3 a, float3 b, float3 c, uint4 adj,
                                            out float3 closestP, out bool generateContact,
                                            out bool faceContact, out uint mask)
{
    faceContact = false;
    const float3 ab = b - a, ac = c - a, ap = p - a;
    const float d1 = ptDot(ab, ap), d2 = ptDot(ac, ap);
    const float3 bp = p - b;
    const float d3 = ptDot(ab, bp), d4 = ptDot(ac, bp);
    const float3 cp = p - c;
    const float d5 = ptDot(ab, cp), d6 = ptDot(ac, cp);
    const float vc = d1 * d4 - d3 * d2;
    const float vb = d5 * d2 - d1 * d6;
    const float va = d3 * d6 - d5 * d4;

    // sphereTriangle.cuh:96 is `const float edgeTol = (va + vb + vc) * 1e-5;` --
    // the literal carries NO SUFFIX, so in C++ it is a DOUBLE.  The multiply
    // therefore happens in DOUBLE precision and only the assignment narrows to
    // float.  `1e-5f` computes the whole thing in float, which lands about 0.4 ULP
    // away: float(1e-5) is 0x3727c5ac = 9.999999747e-06, the double is
    // 0x3ee4f8b588e368f1.
    //
    // `double(1.0f) / double(100000.0f)` reproduces the C++ literal BIT FOR BIT
    // (both 0x3ee4f8b588e368f1, checked): 100000 is exact in float and in double
    // and the division is correctly rounded.  Same idiom, and the same reason, as
    // npSbExtractRotAPD.slang:125-126.
    //
    // This is the ONLY site in the whole GPU CUDA corpus where an unsuffixed
    // literal meets float arithmetic -- searched, not assumed.  NO FIXTURE
    // SEPARATES THE TWO SPELLINGS (P9.55 measured that): on the heightfield path
    // `edgeTol` is dead, because a heightfield adjacency is a plain index or
    // BOUNDARY and `ptIsEdgeNonconvex` is then always false, so the `||`
    // short-circuits; elsewhere separating them needs `vc` within ~2.5e-8 relative
    // of `-edgeTol`.  It is spelled the source's way on the strength of the
    // source, not of a red test.
    const double D1E_5 = double(1.0f) / double(100000.0f);
    const float edgeTol = float(double(va + vb + vc) * D1E_5);

    if (d1 < 0.0f && d2 < 0.0f)
    {
        mask = PT_EV0;
        generateContact = (!ptIsEdgeNonconvex(adj.x)) || (!ptIsEdgeNonconvex(adj.z));
        closestP = a; return ptDot(ap, ap);
    }
    if (d3 > 0.0f && d4 < d3)
    {
        mask = PT_EV1;
        generateContact = (!ptIsEdgeNonconvex(adj.x)) || (!ptIsEdgeNonconvex(adj.y));
        closestP = b; return ptDot(bp, bp);
    }
    if (d6 > 0.0f && d5 < d6)
    {
        mask = PT_EV2;
        generateContact = (!ptIsEdgeNonconvex(adj.y)) || (!ptIsEdgeNonconvex(adj.z));
        closestP = c; return ptDot(cp, cp);
    }
    if (vc < 0.0f && d1 >= 0.0f && d3 <= 0.0f)
    {
        mask = PT_EE01;
        generateContact = !ptIsEdgeNonconvex(adj.x) || vc > -edgeTol;
        const float v = d1 / (d1 - d3);
        closestP = a + v * ab; return ptDot(closestP - p, closestP - p);
    }
    if (vb < 0.0f && d2 >= 0.0f && d6 <= 0.0f)
    {
        mask = PT_EE02;
        generateContact = !ptIsEdgeNonconvex(adj.z) || vb > -edgeTol;
        const float w = d2 / (d2 - d6);
        closestP = a + w * ac; return ptDot(closestP - p, closestP - p);
    }
    if (va < 0.0f && d4 >= d3 && d5 >= d6)
    {
        mask = PT_EE12;
        generateContact = !ptIsEdgeNonconvex(adj.y) || va > -edgeTol;
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        closestP = b + w * (c - b); return ptDot(closestP - p, closestP - p);
    }

    generateContact = true;
    faceContact = true;
    mask = 0u;
    // THE FACE REGION SUMS FIRST AND SCALES ONCE.  sphereTriangle.cuh:138 is
    //     closestP = a + (ab * vb + ac * vc) / (va + vb + vc);
    // and `PxVec3::operator/(Type f)` (PxVec3.h:218-223) is `f = 1/f; return
    // (x*f, y*f, z*f)` -- ONE vector sum, ONE reciprocal, three multiplies, and
    // ONE add onto `a`.  Not two scaled scalars, and not two adds.  See the
    // header for what P9.36 did here and why P9.53 withdrew it.
    const float denom = 1.0f / (va + vb + vc);
    closestP = a + (ab * vb + ac * vc) * denom;
    return ptDot(closestP - p, closestP - p);
}
