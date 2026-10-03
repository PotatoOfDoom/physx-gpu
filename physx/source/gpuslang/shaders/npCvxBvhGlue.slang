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

// npCvxBvhGlue.slang
//
// P9.70c LIFTED THE BODY OUT.  Everything this fixture verifies now lives in
// pxgSlangCvxCoreBvh.slang, included below, and is compiled from that one text by
// both this fixture and cvxCoreTriNpBda's engine kernel.  What is left here is the
// I/O record and the entry point -- the same shape the P9.67b/P9.69a lifts left
// behind, and for the same reason: a green fixture after the move is evidence
// about the SHARED code rather than about a copy of it.
//
// The description below is kept because it is the reading that produced the port.
//
// Slang port of the TrimeshBvh scalar glue from
// gpunarrowphase/src/CUDA/convexCoreCollision.cu (L597-682) and the TestBvh
// callback prologue (L548-595):
//   - scaleBounds: PxBounds3::transformFast(scale.getInverse().toMat33(), b)
//     = basisExtent(M * center, M, extents) with the abs-sum row extents
//     (PxBounds3.h L301-314, L460-464); getInverse() inverts the scale
//     components, keeping the rotation;
//   - the traversal callback margin: bounds.getDimensions().minElement()
//     * 0.0001f;
//   - intersectBox: PxBounds3::intersects' six-comparison form;
//   - createConvex: the three mesh-scale-transformed vertices (the same
//     PxMeshScale::toMat33 = R^T * diag(s) * R form ported in npTriMesh)
//     packed as an ePOINTS ConvexShape (PointsCore: numPoints = 3, stride =
//     sizeof(float4), S = 1, R = identity, identity pose, the callback
//     margin) -- emitted here as the tranche-163 coreData image.
// getTriNormal / getTriBarycentric were ported in npTriMesh.slang
// (tranche 71); the convex-core getTriBarycentric (cvxBvhTriBarycentric, the
// cloth kernel's) is driven here too since P9.94d; bv32TreeTraversal in npBv32.slang (tranche 76); the
// Contact32 warp accumulator is tranche 173.  ALL outputs verify bitwise.

struct BvhIn
{
    float4 scale;     // PxMeshScale scale.xyz
    float4 scaleRot;  // PxMeshScale rotation quat
    float4 bmin, bmax;   // query bounds (world/shape space)
    float4 pmin, pmax;   // node box to intersect (scaled space)
    float4 v0, v1, v2;   // unscaled triangle verts
    float4 v3;           // the FOURTH vertex, for the tetrahedron arm (P9.73)
    float4 dir;          // support direction for the tet support query (P9.73)
    float4 p;            // query point for getTriBarycentric (P9.94d)
}

struct BvhOut
{
    float4 smin, smax;   // scaleBounds output
    float4 marginOut;    // x = callback margin, y = intersects ? 1 : 0
    float4 verts[3];     // createConvex verts (w = 1)
    uint4 core[3];       // PointsCore image (tranche-163 packing, ptr = 0x30)
    float4 prim;         // x = margin, y = asfloat(coreType), zw = pose id marker
    // ---- the TETRAHEDRON arm (P9.73) ----
    float4 tetVerts[4];  // read back THROUGH ptsAt, so the storage itself is checked
    uint4 tetCore[3];    // the tet PointsCore image (numPoints = 4)
    float4 tetSup;       // supportLocalQ over the 4-point inline pool.  THE decisive one.
    // ---- getTriBarycentric (P9.94d) ----
    float4 bary;         // cvxBvhTriBarycentric(scale, scaleRot, v0, v1, v2, p)
}

import pxgSlangCvxCoreShape;             // ShapeRef / PtsRef / ptsFromTri / CORE_POINTS
#include "pxgSlangCvxCoreBvh.slang"      // the glue itself (P9.70c)

struct BvhParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<BvhParams> gParams;

RWStructuredBuffer<BvhIn> gIn;    // binding 0
RWStructuredBuffer<BvhOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void cbgGlue(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const BvhIn s = gIn[t];

    const float3 bmin = s.bmin.xyz, bmax = s.bmax.xyz;
    const float3 scale = s.scale.xyz;

    float3 smin, smax;
    cvxBvhScaleBounds(scale, s.scaleRot, bmin, bmax, smin, smax);
    const float margin = cvxBvhCallbackMargin(smin, smax);
    const bool isect = cvxBvhIntersectBox(s.pmin.xyz, s.pmax.xyz, smin, smax);

    ShapeRef sh;
    PtsRef pts;
    float outMargin;
    cvxBvhCreateConvex(scale, s.scaleRot, s.v0.xyz, s.v1.xyz, s.v2.xyz, margin,
                       sh, pts, outMargin);

    BvhOut r;
    r.smin = float4(smin, 0);
    r.smax = float4(smax, 0);
    r.marginOut = float4(margin, isect ? 1.0f : 0.0f, 0, 0);
    // The three vertices come back through the PtsRef's inline arm, which is where
    // the kernel reads them from too -- so the fixture checks the ACTUAL storage the
    // engine path uses, not a second copy computed alongside it.
    r.verts[0] = float4(ptsAt(pts, 0u, 16u), 1.0f);
    r.verts[1] = float4(ptsAt(pts, 1u, 16u), 1.0f);
    r.verts[2] = float4(ptsAt(pts, 2u, 16u), 1.0f);
    r.core[0] = sh.core[0];
    r.core[1] = sh.core[1];
    r.core[2] = sh.core[2];
    r.prim = float4(outMargin, asfloat(sh.type), 0, 0); // coreType = ePOINTS, identity pose

    // ---- the TETRAHEDRON arm (P9.73) --------------------------------------
    //
    // WHY THIS EXISTS.  P9.72's mutation sweep measured that the observing scene
    // (PXG_SCENE=softbody PXG_SB_FLOOR=convexcore) cannot see a tetrahedron's fourth
    // vertex at all: both `ptsref-v3-dropped` (ptsAt(p,3,..) -> v2) and
    // `tet-numpoints-3` (numPoints 4 -> 3) SURVIVED.  A cube resting flat on a planar
    // floor touches it with its SURFACE tetrahedra, whose fourth corner points inward,
    // away from the contact, so it never wins the support query.
    //
    // npCvxCoreSupport has an ePOINTS arm and cannot help either: it is a Phase-1
    // fixture with its OWN copy of supportLocalQ and never compiles
    // pxgSlangCvxCoreShape at all.  This fixture already imports that module and
    // includes pxgSlangCvxCoreBvh, so it is the one place that drives the real code.
    //
    // THE SUPPORT QUERY IS THE DECISIVE OUTPUT, not the vertices.  Reading the four
    // verts back would catch ptsref-v3-dropped but NOT tet-numpoints-3, which changes
    // only the count in the core image.  supportLocalQ reads that count, so a 3 there
    // silently drops v3 from the argmax -- and the returned point changes exactly when
    // v3 was the extreme one.  The host mirror counts how often that happens, because
    // a query where v3 never wins proves nothing.
    ShapeRef tetSh;
    PtsRef tetPts;
    float tetMargin;
    // createConvexTet takes the vertices RAW -- no mesh scale anywhere (cu:1075-1079).
    cvxBvhCreateConvexTet(s.v0.xyz, s.v1.xyz, s.v2.xyz, s.v3.xyz, margin,
                          tetSh, tetPts, tetMargin);
    r.tetVerts[0] = float4(ptsAt(tetPts, 0u, 16u), 1.0f);
    r.tetVerts[1] = float4(ptsAt(tetPts, 1u, 16u), 1.0f);
    r.tetVerts[2] = float4(ptsAt(tetPts, 2u, 16u), 1.0f);
    r.tetVerts[3] = float4(ptsAt(tetPts, 3u, 16u), 1.0f);
    r.tetCore[0] = tetSh.core[0];
    r.tetCore[1] = tetSh.core[1];
    r.tetCore[2] = tetSh.core[2];
    r.tetSup = float4(supportLocalQ(tetSh, tetPts, s.dir.xyz), 0.0f);

    // ---- TrimeshBvh::getTriBarycentric (P9.94d) ----------------------------
    // The cloth kernel's barycentric (convexCoreCollision.cu:1270 -> :672-680): the
    // vertices go through the SAME forward mesh-scale matrix as createConvex before
    // PxComputeBarycentric.  Driven here so the engine text is checked bitwise,
    // including the signed-zero case the identity matrix changes.
    r.bary = cvxBvhTriBarycentric(scale, s.scaleRot, s.v0.xyz, s.v1.xyz, s.v2.xyz, s.p.xyz);
    gOut[t] = r;
}
