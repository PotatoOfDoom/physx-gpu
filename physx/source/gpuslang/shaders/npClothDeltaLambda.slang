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

// npClothDeltaLambda.slang
//
// Slang port of the FEM-cloth XPBD delta-lambda family from FEMClothUtil.cuh:
// queryDeltaLambda 3-vertex no-damping (L211-223), 3-vertex damping (L229-243),
// 4-vertex no-damping (L249-262) and 4-vertex damping (L268-...).
//   no-damping:  denom = (Sum mi*|dCdx|^2) + alphaTilde;
//                if denom < THRESHOLD return 0; return (-C - alphaTilde*lambda)/denom
//   damping:     denom = (1 + damping*dtInv)*(Sum mi*|dCdx|^2) + alphaTilde;
//                if denom < THRESHOLD return 0; return -(C + alphaTilde*lambda + damping*dCdT)/denom
//   |dCdx|^2 is PxVec2 magnitudeSquared (2D) or PxVec3 (3D/4-vertex).
//
// NoContraction-stamped (mi*magSq sums are the FMA surface); +-0 eqFZ.

static const float FEMCLOTH_THRESHOLD = 1.0e-14f;

float magSq2(float2 v) { return v.x * v.x + v.y * v.y; }
float magSq3(float3 v) { return v.x * v.x + v.y * v.y + v.z * v.z; }

// 3-vertex, 2D (PxVec2), no damping
float qdl3_2d(float C, float2 d0, float2 d1, float2 d2, float alphaTilde, float lambda, float mi0, float mi1, float mi2)
{
    const float denom = (mi0 * magSq2(d0) + mi1 * magSq2(d1) + mi2 * magSq2(d2)) + alphaTilde;
    if (denom < FEMCLOTH_THRESHOLD) return 0.0f;
    return (-C - alphaTilde * lambda) / denom;
}

// 3-vertex, 3D, damping
float qdl3_3d_damp(float C, float3 d0, float3 d1, float3 d2, float alphaTilde, float lambda,
                   float mi0, float mi1, float mi2, float damping, float dtInv, float dCdT)
{
    const float denom = (1.0f + damping * dtInv) * (mi0 * magSq3(d0) + mi1 * magSq3(d1) + mi2 * magSq3(d2)) + alphaTilde;
    if (denom < FEMCLOTH_THRESHOLD) return 0.0f;
    return -(C + alphaTilde * lambda + damping * dCdT) / denom;
}

// 4-vertex, 3D, no damping
float qdl4(float C, float3 d0, float3 d1, float3 d2, float3 d3, float alphaTilde, float lambda,
           float mi0, float mi1, float mi2, float mi3)
{
    const float denom = (mi0 * magSq3(d0) + mi1 * magSq3(d1) + mi2 * magSq3(d2) + mi3 * magSq3(d3)) + alphaTilde;
    if (denom < FEMCLOTH_THRESHOLD) return 0.0f;
    return (-C - alphaTilde * lambda) / denom;
}

// 4-vertex, 3D, damping
float qdl4_damp(float C, float3 d0, float3 d1, float3 d2, float3 d3, float alphaTilde, float lambda,
                float mi0, float mi1, float mi2, float mi3, float damping, float dtInv, float dCdT)
{
    const float denom = (1.0f + damping * dtInv) * (mi0 * magSq3(d0) + mi1 * magSq3(d1) + mi2 * magSq3(d2) + mi3 * magSq3(d3)) + alphaTilde;
    if (denom < FEMCLOTH_THRESHOLD) return 0.0f;
    return -(C + alphaTilde * lambda + damping * dCdT) / denom;
}

struct DlIn
{
    float4 d0, d1, d2, d3;   // dCdx0..3 (.xyz used; .xy for 2D)
    float4 mi;               // massInv0..3
    float4 scal;             // C, alphaTilde, lambda, _
    float4 damp;             // damping, dtInv, dCdT, _
}

struct DlOut
{
    float4 res;   // qdl3_2d, qdl3_3d_damp, qdl4, qdl4_damp
}

struct DlParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<DlParams> gParams;

RWStructuredBuffer<DlIn> gIn;    // binding 0
RWStructuredBuffer<DlOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void clothDeltaLambda(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const DlIn s = gIn[tid];

    const float C = s.scal.x, at = s.scal.y, lam = s.scal.z;
    const float dmp = s.damp.x, dtInv = s.damp.y, dCdT = s.damp.z;
    const float mi0 = s.mi.x, mi1 = s.mi.y, mi2 = s.mi.z, mi3 = s.mi.w;

    const float r0 = qdl3_2d(C, s.d0.xy, s.d1.xy, s.d2.xy, at, lam, mi0, mi1, mi2);
    const float r1 = qdl3_3d_damp(C, s.d0.xyz, s.d1.xyz, s.d2.xyz, at, lam, mi0, mi1, mi2, dmp, dtInv, dCdT);
    const float r2 = qdl4(C, s.d0.xyz, s.d1.xyz, s.d2.xyz, s.d3.xyz, at, lam, mi0, mi1, mi2, mi3);
    const float r3 = qdl4_damp(C, s.d0.xyz, s.d1.xyz, s.d2.xyz, s.d3.xyz, at, lam, mi0, mi1, mi2, mi3, dmp, dtInv, dCdT);

    DlOut o;
    o.res = float4(r0, r1, r2, r3);
    gOut[tid] = o;
}
