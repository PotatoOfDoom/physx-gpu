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

// npIsoAvgVerts.slang
//
// Slang port of the isosurface averageVerts blend (isosurfaceExtraction.cu
// L934-948).  Per vertex, given the accumulator v = vertices[vNr] (whose w holds
// the summed weight) and the current smoothed output x = output[vNr]:
//
//   if (v.w > 0):
//     output[vNr] = PxVec4( (1 - blendWeight) * x + blendWeight * (v.xyz / v.w), 0 )
//   vertices[vNr] = PxVec4(0)     // always cleared for the next smoothing pass
//
// CRITICAL: `v.getXYZ() / v.w` is PxVec3::operator/(Type), which computes
// `inv = 1/v.w` once then multiplies each component (two roundings), NOT
// per-component division.  `(1-bw)*x` and `bw*(...)` are scalar*vector direct
// per-component multiplies; the two scaled vectors are then added.  When
// v.w<=0 the output is left unchanged (original xyz and w).  The vertices
// accumulator is unconditionally zeroed.  All float mul/add/sub
// NoContraction-stamped; +-0 compared with eqFZ.

struct AvIn
{
    float4 vtx;      // vertices[vNr] = v : accumulated position xyz, weight w
    float4 outp;     // output[vNr] current value : smoothed position xyz, w
    float4 params;   // blendWeight, _, _, _
}

struct AvOut
{
    float4 outNew;   // new output[vNr]
    float4 vtxNew;   // cleared vertices[vNr] (0,0,0,0)
}

struct AvParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<AvParams> gParams;

RWStructuredBuffer<AvIn> gIn;    // binding 0
RWStructuredBuffer<AvOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void isoAvgVerts(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const AvIn s = gIn[tid];

    const float3 v = s.vtx.xyz;
    const float vw = s.vtx.w;
    const float3 x = s.outp.xyz;
    const float blendWeight = s.params.x;

    float4 outNew = s.outp;   // v.w <= 0 : leave output unchanged
    if (vw > 0.0f)
    {
        const float inv = 1.0f / vw;                               // v.xyz / v.w (reciprocal-multiply)
        const float3 r = float3(v.x * inv, v.y * inv, v.z * inv);
        const float ombw = 1.0f - blendWeight;
        const float3 blended = float3(ombw * x.x + blendWeight * r.x,
                                      ombw * x.y + blendWeight * r.y,
                                      ombw * x.z + blendWeight * r.z);
        outNew = float4(blended, 0.0f);
    }

    AvOut o;
    o.outNew = outNew;
    o.vtxNew = float4(0.0f, 0.0f, 0.0f, 0.0f);
    gOut[tid] = o;
}
