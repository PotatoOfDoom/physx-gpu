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

// pxgSlangCvxCoreBvh.slang -- the TrimeshBvh scalar glue and the TestBvh callback
// prologue (convexCoreCollision.cu:549-682), lifted out of npCvxBvhGlue.slang by
// P9.70c so the engine kernel and the fixture that verifies it BITWISE compile the
// same text rather than two copies of it.
//
// npCvxBvhGlue was written first, as a Phase-1 fixture, and it stays the arbiter:
// `gpuslang.npCvxBvhGlue` drives scaleBounds, the callback margin, intersectBox,
// createConvex and (since P9.94d) getTriBarycentric against a CPU mirror and
// compares every output word.  After the lift it drives THESE functions, which is the whole point of lifting this way round
// (the same move P9.67b made for the shape layer and P9.69a for GJK/EPA).
//
// An #include and not a module, for the reason MEASURED in pxgSlangCvxCoreCvx's
// header.  The includer must `import pxgSlangCvxCoreShape;` first.
//
// ===========================================================================
// createConvex BUILDS THE TRIANGLE ON THE STACK, AND THAT DECIDED PtsRef
// ===========================================================================
//
//     float4 verts[_Bvh::PRIM_VERT_COUNT];   // three, for a trimesh
//     bvh.createConvex(primIndex, margin, verts, prim);
//     ...
//     core.points = verts;
//
// (:583-590, :632-655).  A stack array has no device address in SPIR-V, so the
// shape layer's pool parameter became a PtsRef -- see pxgSlangCvxCoreShape.slang.
// `cvxBvhCreateConvex` returns the ShapeRef and the PtsRef together for that
// reason: they are one object and handing them over separately invites a caller
// to pair a triangle's ShapeRef with somebody else's points.
//
// THE POINTS-POINTER MARKER.  supportLocalQ's CORE_POINTS arm uses core[0].xy as a
// 64-bit NULL guard on `core.points`.  In the CUDA that field is a live stack
// address -- never null, and its VALUE is never observable, only its non-nullness.
// So a non-zero stand-in goes in that word pair.  0x30 is npCvxBvhGlue's, kept
// because the fixture's CPU mirror names it and the move is meant to be a move.

static const uint CVXBVH_PTS_MARKER = 0x30u;

// columns of PxMat33(quat) (matches PxMat33T(const PxQuatT&))
void cvxBvhQuatToMat33(float4 q, out float3 c0, out float3 c1, out float3 c2)
{
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    const float x2 = x + x, y2 = y + y, z2 = z + z;
    const float xx = x2 * x, yy = y2 * y, zz = z2 * z;
    const float xy = x2 * y, xz = x2 * z, xw = x2 * w;
    const float yz = y2 * z, yw = y2 * w, zw = z2 * w;
    c0 = float3(1.0f - yy - zz, xy + zw, xz - yw);
    c1 = float3(xy - zw, 1.0f - xx - zz, yz + xw);
    c2 = float3(xz + yw, yz - xw, 1.0f - xx - yy);
}

// columns of PxMeshScale::toMat33() = (transpose(R) with columns scaled) * R
void cvxBvhMeshScaleMat(float3 scale, float4 rot, out float3 m0, out float3 m1, out float3 m2)
{
    float3 r0, r1, r2;
    cvxBvhQuatToMat33(rot, r0, r1, r2);
    // transpose of R, columns scaled by scale.x/y/z
    const float3 t0 = float3(r0.x, r1.x, r2.x) * scale.x;
    const float3 t1 = float3(r0.y, r1.y, r2.y) * scale.y;
    const float3 t2 = float3(r0.z, r1.z, r2.z) * scale.z;
    // M = trans_scaled * R : column_j = trans_scaled.transform(R.column_j)
    m0 = t0 * r0.x + t1 * r0.y + t2 * r0.z;
    m1 = t0 * r1.x + t1 * r1.y + t2 * r1.z;
    m2 = t0 * r2.x + t1 * r2.y + t2 * r2.z;
}

float3 cvxBvhMat33Transform(float3 m0, float3 m1, float3 m2, float3 v)
{
    return m0 * v.x + m1 * v.y + m2 * v.z;
}

// TrimeshBvh::scaleBounds (:620-623):
//   PxBounds3::transformFast(scale.getInverse().toMat33(), bounds)
// = basisExtent(M * center, M, extents) with the abs-sum row extents
// (PxBounds3.h L301-314, L460-464).  getInverse() inverts the scale COMPONENTS and
// keeps the rotation, which is why the same meshScaleMat serves both directions.
void cvxBvhScaleBounds(float3 scale, float4 scaleRot, float3 bmin, float3 bmax,
                       out float3 smin, out float3 smax)
{
    const float3 invScale = float3(1.0f / scale.x, 1.0f / scale.y, 1.0f / scale.z);
    float3 im0, im1, im2;
    cvxBvhMeshScaleMat(invScale, scaleRot, im0, im1, im2);
    const float3 center = (bmin + bmax) * 0.5f;
    const float3 extents = (bmax - bmin) * 0.5f;
    const float3 mc = cvxBvhMat33Transform(im0, im1, im2, center);
    // basisExtent: abs-sum of the scaled basis vectors, per component
    const float3 c0 = im0 * extents.x;
    const float3 c1 = im1 * extents.y;
    const float3 c2 = im2 * extents.z;
    const float3 w = float3(abs(c0.x) + abs(c1.x) + abs(c2.x),
                            abs(c0.y) + abs(c1.y) + abs(c2.y),
                            abs(c0.z) + abs(c1.z) + abs(c2.z));
    smin = mc - w;
    smax = mc + w;
}

// _Callback ctor (:570-573): the margin is taken from the SCALED bounds, because
// the ctor's `_bounds` argument is `bvh.scaleBounds(bounds)` (:592), not the
// caller's world bounds.  Getting that wrong scales every triangle's margin by the
// mesh scale and is silent.
float cvxBvhCallbackMargin(float3 smin, float3 smax)
{
    const float3 dims = smax - smin;
    return min(dims.x, min(dims.y, dims.z)) * 0.0001f;
}

// _Callback::intersectBoxFullWarp (:575-581) -> PxBounds3::intersects, six
// comparisons.  `hasBox` false returns false and is the caller's business.
bool cvxBvhIntersectBox(float3 pmin, float3 pmax, float3 smin, float3 smax)
{
    return !(pmin.x > smax.x || smin.x > pmax.x ||
             pmin.y > smax.y || smin.y > pmax.y ||
             pmin.z > smax.z || smin.z > pmax.z);
}

// TrimeshBvh::createConvex (:632-655).  The three vertices are transformed by the
// FORWARD mesh-scale matrix and packed as an ePOINTS ConvexShape: identity pose,
// PointsCore with numPoints = 3, stride = sizeof(float4), S = (1,1,1), R = identity,
// and the callback's margin.
void cvxBvhCreateConvex(float3 scale, float4 scaleRot, float3 v0, float3 v1, float3 v2,
                        float margin, out ShapeRef sh, out PtsRef pts, out float outMargin)
{
    float3 m0, m1, m2;
    cvxBvhMeshScaleMat(scale, scaleRot, m0, m1, m2);
    const float3 tv0 = cvxBvhMat33Transform(m0, m1, m2, v0);
    const float3 tv1 = cvxBvhMat33Transform(m0, m1, m2, v1);
    const float3 tv2 = cvxBvhMat33Transform(m0, m1, m2, v2);

    sh.type = CORE_POINTS;
    // PointsCore image: words 0/1 the pointer, 2..4 S, 5..8 R, 9 numPoints|stride<<8
    sh.core[0] = uint4(CVXBVH_PTS_MARKER, 0u, asuint(1.0f), asuint(1.0f));
    sh.core[1] = uint4(asuint(1.0f), asuint(0.0f), asuint(0.0f), asuint(0.0f));
    sh.core[2] = uint4(asuint(1.0f), 3u | (16u << 8u), 0u, 0u);
    sh.poseQ = float4(0, 0, 0, 1);   // PxTransform(PxIdentity)
    sh.poseP = float3(0, 0, 0);
    pts = ptsFromTri(tv0, tv1, tv2);
    outMargin = margin;
}

// ---------------------------------------------------------------------------
// TetmeshBvh (convexCoreCollision.cu:1042-1093).  P9.72.
//
// It is the SIBLING of TrimeshBvh and it is much simpler, in three ways that all
// have to be reproduced rather than assumed away:
//
//   scaleBounds IS THE IDENTITY (:1060-1063, `return bounds;`).  A softbody's tet
//   mesh carries no PxMeshScale at all, so the query box goes to the traversal in
//   the frame it was computed in.  Calling cvxBvhScaleBounds here with an identity
//   PxMeshScale would give the same numbers -- and would also silently start
//   working differently the day someone gives a softbody a scale.
//
//   createConvex TAKES THE VERTICES RAW (:1075-1079).  No mesh-scale matrix, no
//   transform of any kind: `verts[0] = tetmeshVerts[inds.x];` straight out of the
//   softbody's mPosition_InvMass.  Those positions are already in world space,
//   which is also why the kernel's transform0in1 puts the CONVEX into that frame.
//
//   THERE ARE FOUR OF THEM.  PRIM_VERT_COUNT = 4 (:1071) and core.numPoints = 4
//   (:1086).  That is what forced PtsRef's fourth inline vertex; before P9.72 the
//   fourth corner had nowhere to go and ptsAt(p, 3, ...) returned the third.
//
// tetmeshSurfaceHint is read into the struct (:1046, :1057) and never used by any
// of its methods.  Transcribed as absent rather than as a field nobody reads.
void cvxBvhCreateConvexTet(float3 v0, float3 v1, float3 v2, float3 v3,
                           float margin, out ShapeRef sh, out PtsRef pts, out float outMargin)
{
    sh.type = CORE_POINTS;
    // The same PointsCore image as the triangle arm, with numPoints = 4.  The count
    // lives in the low byte of word 9 and the stride in the next byte; supportLocalQ
    // and contactFaceQ both read it from there, so a wrong count here does not fail
    // loudly -- it silently drops a corner.
    sh.core[0] = uint4(CVXBVH_PTS_MARKER, 0u, asuint(1.0f), asuint(1.0f));
    sh.core[1] = uint4(asuint(1.0f), asuint(0.0f), asuint(0.0f), asuint(0.0f));
    sh.core[2] = uint4(asuint(1.0f), 4u | (16u << 8u), 0u, 0u);
    sh.poseQ = float4(0, 0, 0, 1);   // PxTransform(PxIdentity)
    sh.poseP = float3(0, 0, 0);
    pts = ptsFromTet(v0, v1, v2, v3);
    outMargin = margin;
}

// TrimeshBvh::getTriNormal (:657-665): the SAME scale transform, then
// (v1-v0).cross(v2-v0).getNormalized().  getNormalized is the guarded form --
// magSq > 0 before the reciprocal -- which getNormalizedQ is.
float3 cvxBvhTriNormal(float3 scale, float4 scaleRot, float3 v0, float3 v1, float3 v2)
{
    float3 m0, m1, m2;
    cvxBvhMeshScaleMat(scale, scaleRot, m0, m1, m2);
    const float3 tv0 = cvxBvhMat33Transform(m0, m1, m2, v0);
    const float3 tv1 = cvxBvhMat33Transform(m0, m1, m2, v1);
    const float3 tv2 = cvxBvhMat33Transform(m0, m1, m2, v2);
    return getNormalizedQ(crossQ(tv1 - tv0, tv2 - tv0));
}

// PxComputeBarycentric, the TRIANGLE overload (PxMathUtils.h:390-409).  Moved here
// from cvxCoreClothNpBda by P9.94d so the fixture drives it.
//
// TRANSCRIBED FROM PxMathUtils.h, NOT from the corpus's `computeBarycentric`
// (pxgSlangClothTriContact.slang, from femClothPrimitives.cu).  Those are femClothPrimitives.cu's
// function: the same Cramer arithmetic plus a [-1e-4, 1+1e-4] acceptance window whose
// bool result gates a contact.  This call site has no gate and wants the raw
// coordinates, so lifting one of those and ignoring its return would have imported a
// window that does not belong here.
//
// `denom` is NOT guarded upstream -- a degenerate triangle divides by zero and
// produces inf/NaN.  Transcribed as written; the engine relies on the midphase not
// emitting degenerate triangles.
float4 cvxBvhComputeBarycentric(float3 a, float3 b, float3 c, float3 p)
{
    const float3 v0 = b - a;
    const float3 v1 = c - a;
    const float3 v2 = p - a;

    const float d00 = dot3Q(v0, v0);
    const float d01 = dot3Q(v0, v1);
    const float d11 = dot3Q(v1, v1);
    const float d20 = dot3Q(v2, v0);
    const float d21 = dot3Q(v2, v1);

    const float denom = d00 * d11 - d01 * d01;
    const float v = (d11 * d20 - d01 * d21) / denom;
    const float w = (d00 * d21 - d01 * d20) / denom;
    const float u = 1.0f - v - w;

    // bary.w = 0 upstream -- the fourth slot exists for the TETRAHEDRON overload
    // (:358-377) and is deliberately zeroed on the triangle path.
    return float4(u, v, w, 0.0f);
}

// TrimeshBvh::getTriBarycentric (convexCoreCollision.cu:672-680): the SAME forward
// scale transform as createConvex and getTriNormal, THEN PxComputeBarycentric of the
// query point against the transformed triangle.  P9.94d: the cloth kernel used to
// pass the raw vertices on the grounds that the cloth's scale is identity -- but an
// identity PxMat33 transform is not a bit-level no-op: `column0*x + column1*y +
// column2*z` with +0 off-diagonals turns a -0 coordinate into +0 wherever a +0
// product joins the sum, and that changes the zero sign of v or w on triangles with
// signed-zero coordinates (npCvxBvhGlue_p94dTriBaryZeroSign).
float4 cvxBvhTriBarycentric(float3 scale, float4 scaleRot, float3 v0, float3 v1, float3 v2,
                            float3 p)
{
    float3 m0, m1, m2;
    cvxBvhMeshScaleMat(scale, scaleRot, m0, m1, m2);
    const float3 tv0 = cvxBvhMat33Transform(m0, m1, m2, v0);
    const float3 tv1 = cvxBvhMat33Transform(m0, m1, m2, v1);
    const float3 tv2 = cvxBvhMat33Transform(m0, m1, m2, v2);
    return cvxBvhComputeBarycentric(tv0, tv1, tv2, p);
}
