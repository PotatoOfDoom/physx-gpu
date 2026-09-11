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
// (tranche 71); bv32TreeTraversal in npBv32.slang (tranche 76); the
// Contact32 warp accumulator is tranche 173.  ALL outputs verify bitwise.

struct BvhIn
{
    float4 scale;     // PxMeshScale scale.xyz
    float4 scaleRot;  // PxMeshScale rotation quat
    float4 bmin, bmax;   // query bounds (world/shape space)
    float4 pmin, pmax;   // node box to intersect (scaled space)
    float4 v0, v1, v2;   // unscaled triangle verts
}

struct BvhOut
{
    float4 smin, smax;   // scaleBounds output
    float4 marginOut;    // x = callback margin, y = intersects ? 1 : 0
    float4 verts[3];     // createConvex verts (w = 1)
    uint4 core[3];       // PointsCore image (tranche-163 packing, ptr = 0x30)
    float4 prim;         // x = margin, y = asfloat(coreType), zw = pose id marker
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
    gOut[t] = r;
}
