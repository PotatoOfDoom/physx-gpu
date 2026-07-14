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

struct BvhParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<BvhParams> gParams;

RWStructuredBuffer<BvhIn> gIn;    // binding 0
RWStructuredBuffer<BvhOut> gOut;  // binding 1

float dot3Q(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// columns of PxMat33(quat) (matches PxMat33T(const PxQuatT&))
void quatToMat33(float4 q, out float3 c0, out float3 c1, out float3 c2)
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
void meshScaleMat(float3 scale, float4 rot, out float3 m0, out float3 m1, out float3 m2)
{
    float3 r0, r1, r2;
    quatToMat33(rot, r0, r1, r2);
    // transpose of R, columns scaled by scale.x/y/z
    const float3 t0 = float3(r0.x, r1.x, r2.x) * scale.x;
    const float3 t1 = float3(r0.y, r1.y, r2.y) * scale.y;
    const float3 t2 = float3(r0.z, r1.z, r2.z) * scale.z;
    // M = trans_scaled * R : column_j = trans_scaled.transform(R.column_j)
    m0 = t0 * r0.x + t1 * r0.y + t2 * r0.z;
    m1 = t0 * r1.x + t1 * r1.y + t2 * r1.z;
    m2 = t0 * r2.x + t1 * r2.y + t2 * r2.z;
}


float3 mat33Transform(float3 m0, float3 m1, float3 m2, float3 v)
{
    return m0 * v.x + m1 * v.y + m2 * v.z;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cbgGlue(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const BvhIn s = gIn[t];

    const float3 bmin = s.bmin.xyz, bmax = s.bmax.xyz;

    // scaleBounds: transformFast(scale.getInverse().toMat33(), bounds)
    const float3 invScale = float3(1.0f / s.scale.x, 1.0f / s.scale.y, 1.0f / s.scale.z);
    float3 im0, im1, im2;
    meshScaleMat(invScale, s.scaleRot, im0, im1, im2);
    const float3 center = (bmin + bmax) * 0.5f;
    const float3 extents = (bmax - bmin) * 0.5f;
    const float3 mc = mat33Transform(im0, im1, im2, center);
    // basisExtent: abs-sum of the scaled basis vectors, per component
    const float3 c0 = im0 * extents.x;
    const float3 c1 = im1 * extents.y;
    const float3 c2 = im2 * extents.z;
    const float3 w = float3(abs(c0.x) + abs(c1.x) + abs(c2.x),
                            abs(c0.y) + abs(c1.y) + abs(c2.y),
                            abs(c0.z) + abs(c1.z) + abs(c2.z));
    const float3 smin = mc - w;
    const float3 smax = mc + w;

    // _Callback ctor margin: scaled-bounds dimensions min element * 0.0001f
    const float3 dims = smax - smin;
    const float margin = min(dims.x, min(dims.y, dims.z)) * 0.0001f;

    // intersectBox: PxBounds3::intersects six-comparison form vs the node box
    const float3 pmin = s.pmin.xyz, pmax = s.pmax.xyz;
    const bool isect = !(pmin.x > smax.x || smin.x > pmax.x || pmin.y > smax.y || smin.y > pmax.y ||
                         pmin.z > smax.z || smin.z > pmax.z);

    // createConvex: forward scale matrix on the three verts + PointsCore image
    float3 m0, m1, m2;
    meshScaleMat(s.scale.xyz, s.scaleRot, m0, m1, m2);
    const float3 tv0 = mat33Transform(m0, m1, m2, s.v0.xyz);
    const float3 tv1 = mat33Transform(m0, m1, m2, s.v1.xyz);
    const float3 tv2 = mat33Transform(m0, m1, m2, s.v2.xyz);

    BvhOut r;
    r.smin = float4(smin, 0);
    r.smax = float4(smax, 0);
    r.marginOut = float4(margin, isect ? 1.0f : 0.0f, 0, 0);
    r.verts[0] = float4(tv0, 1.0f);
    r.verts[1] = float4(tv1, 1.0f);
    r.verts[2] = float4(tv2, 1.0f);
    // PointsCore image (tranche-163 packing): ptr lo/hi, S, R, numPoints|stride<<8
    r.core[0] = uint4(0x30u, 0u, asuint(1.0f), asuint(1.0f));
    r.core[1] = uint4(asuint(1.0f), asuint(0.0f), asuint(0.0f), asuint(0.0f));
    r.core[2] = uint4(asuint(1.0f), 3u | (16u << 8u), 0u, 0u);
    r.prim = float4(margin, asfloat(6u), 0, 0); // coreType = ePOINTS, identity pose
    gOut[t] = r;
}
