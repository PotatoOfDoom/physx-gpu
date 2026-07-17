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

// npIsoVecNormalize.slang
//
// Slang port of three isosurface vector helpers (isosurfaceExtraction.cu):
//   areaTimes2(v1, v2) (L895-898): |v1 x v2| = magnitude of the cross product
//   normalize(PxVec3 v) (L956-962): m = |v|; if m>0 return v/m (DIVISION) else v
//   normalize(PxVec4 v) (L997-1001):
//     s = 1 / max(1e-6, sqrt(v.x^2+v.y^2+v.z^2)); return (s*v.x, s*v.y, s*v.z, v.w)
//     (RECIPROCAL-MULTIPLY, eps-clamped, preserves .w)
//
// The two normalize forms differ deliberately: the vec3 form divides by the
// magnitude and passes the vector through unchanged when m==0; the vec4 form
// multiplies by the reciprocal of an eps-clamped magnitude (never divides by
// < 1e-6) and keeps .w.  Float mul/div/sqrt NoContraction-stamped; +-0 eqFZ.

float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float3 cross3(float3 a, float3 b) { return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
float pmax(float a, float b) { return (a > b) ? a : b; }

struct NvIn
{
    float4 v1;   // .xyz : areaTimes2 arg 1 AND the vec3 to normalize
    float4 v2;   // .xyz : areaTimes2 arg 2
    float4 v4;   // the vec4 to normalize
}

struct NvOut
{
    float4 normVec3;   // normalize(v1.xyz) in .xyz
    float4 normVec4;   // normalize(v4)
    float4 scalars;    // areaTimes2(v1, v2), _, _, _
}

struct NvParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<NvParams> gParams;

RWStructuredBuffer<NvIn> gIn;    // binding 0
RWStructuredBuffer<NvOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void isoVecNormalize(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const NvIn s = gIn[tid];

    const float3 v1 = s.v1.xyz;
    const float3 v2 = s.v2.xyz;
    const float4 v4 = s.v4;

    // areaTimes2 = |v1 x v2|
    const float3 cr = cross3(v1, v2);
    const float area = sqrt(dot3(cr, cr));

    // normalize(PxVec3): m = |v|; if m>0: v/m else v
    float3 nv3 = v1;
    const float m = sqrt(dot3(v1, v1));
    if (m > 0.0f)
        nv3 = float3(v1.x / m, v1.y / m, v1.z / m);

    // normalize(PxVec4): reciprocal of eps-clamped xyz magnitude, keep .w
    const float sc = 1.0f / pmax(1.0e-6f, sqrt(v4.x * v4.x + v4.y * v4.y + v4.z * v4.z));
    const float4 nv4 = float4(sc * v4.x, sc * v4.y, sc * v4.z, v4.w);

    NvOut o;
    o.normVec3 = float4(nv3, 0.0f);
    o.normVec4 = nv4;
    o.scalars = float4(area, 0.0f, 0.0f, 0.0f);
    gOut[tid] = o;
}
