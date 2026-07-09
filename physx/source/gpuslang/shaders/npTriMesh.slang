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

// npTriMesh.slang
//
// Slang port of the scaled-triangle accessors from
// gpunarrowphase/src/CUDA/convexCoreCollision.cu: getTriNormal (transform the
// three triangle vertices by the mesh scale and return the normalized face
// normal) and getTriBarycentric (transform the vertices then compute the
// barycentric coordinates of a query point via PxComputeBarycentric).  The mesh
// scale is PxMeshScale, whose toMat33() is R^T * diag(scale) * R for the scale
// rotation R (a symmetric non-uniform-scale matrix).
//
// PxComputeBarycentric here is the Cramer's-rule triangle variant from
// PxMathUtils.h (u,v,w via the Gram matrix), distinct from the signed-area
// barycentricCoordinates ported in npTriMask.slang.

struct TmParams { uint numTests; uint pad0, pad1, pad2; }

struct MeshScale { float4 scale; float4 rot; } // scale.xyz, rot = quaternion
struct TriIn { float4 v0, v1, v2, p; }         // unscaled verts + query point (scaled space)

[[vk::push_constant]]
ConstantBuffer<TmParams> gParams;

RWStructuredBuffer<TriIn> gTri;      // binding 0
RWStructuredBuffer<MeshScale> gScale; // binding 1
RWStructuredBuffer<float4> gNormalOut; // binding 2  (xyz normal)
RWStructuredBuffer<float4> gBaryOut;   // binding 3  (u, v, w, 0)

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

float3 matTransform(float3 m0, float3 m1, float3 m2, float3 v) { return m0 * v.x + m1 * v.y + m2 * v.z; }

// PxComputeBarycentric (triangle, Cramer's rule) -> (u, v, w, 0)
float4 computeBarycentric(float3 a, float3 b, float3 c, float3 p)
{
    const float3 v0 = b - a, v1 = c - a, v2 = p - a;
    const float d00 = dot(v0, v0), d01 = dot(v0, v1), d11 = dot(v1, v1);
    const float d20 = dot(v2, v0), d21 = dot(v2, v1);
    const float denom = d00 * d11 - d01 * d01;
    const float v = (d11 * d20 - d01 * d21) / denom;
    const float w = (d00 * d21 - d01 * d20) / denom;
    const float u = 1.0f - v - w;
    return float4(u, v, w, 0.0f);
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void ctTriNormal(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numTests) return;
    const TriIn e = gTri[i];
    const MeshScale s = gScale[i];
    float3 m0, m1, m2;
    meshScaleMat(s.scale.xyz, s.rot, m0, m1, m2);
    const float3 v0 = matTransform(m0, m1, m2, e.v0.xyz);
    const float3 v1 = matTransform(m0, m1, m2, e.v1.xyz);
    const float3 v2 = matTransform(m0, m1, m2, e.v2.xyz);
    const float3 nrm = normalize(cross(v1 - v0, v2 - v0));
    gNormalOut[i] = float4(nrm, 0.0f);
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void ctTriBarycentric(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numTests) return;
    const TriIn e = gTri[i];
    const MeshScale s = gScale[i];
    float3 m0, m1, m2;
    meshScaleMat(s.scale.xyz, s.rot, m0, m1, m2);
    const float3 v0 = matTransform(m0, m1, m2, e.v0.xyz);
    const float3 v1 = matTransform(m0, m1, m2, e.v1.xyz);
    const float3 v2 = matTransform(m0, m1, m2, e.v2.xyz);
    gBaryOut[i] = computeBarycentric(v0, v1, v2, e.p.xyz);
}
