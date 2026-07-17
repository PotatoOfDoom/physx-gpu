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

// npDefMaterial.slang
//
// Slang port of the deformable material / geometry helpers from
// deformableUtils.cuh: projectVectorOntoPlane (L118-121) and lameParameters
// (L124-130).
//   projectVectorOntoPlane(v, n) = v - (n.dot(v) / n.magnitudeSquared()) * n
//   lameParameters(Young, Poisson):
//     lambda = Young*Poisson / ((1 + Poisson) * (1 - 2*Poisson))
//     mu     = Young / (2 * (1 + Poisson))
//
// Pure scalar / PxVec3 arithmetic (division).  NoContraction-stamped; +-0 eqFZ.

float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

struct DmIn
{
    float4 v;         // v.xyz for projection
    float4 n;         // planeNormal.xyz
    float4 misc;      // Young, Poisson, _, _
}

struct DmOut
{
    float4 proj;      // projected vector .xyz
    float4 lame;      // lambda, mu, _, _
}

struct DmParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<DmParams> gParams;

RWStructuredBuffer<DmIn> gIn;    // binding 0
RWStructuredBuffer<DmOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void defMaterial(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const DmIn s = gIn[tid];

    const float3 v = s.v.xyz;
    const float3 n = s.n.xyz;
    // v - (n.dot(v) / n.magnitudeSquared()) * n
    const float scale = dot3(n, v) / dot3(n, n);
    const float3 proj = v - scale * n;

    const float Young = s.misc.x;
    const float Poisson = s.misc.y;
    const float lambda = Young * Poisson / ((1.0f + Poisson) * (1.0f - 2.0f * Poisson));
    const float mu = Young / (2.0f * (1.0f + Poisson));

    DmOut o;
    o.proj = float4(proj, 0.0f);
    o.lame = float4(lambda, mu, 0.0f, 0.0f);
    gOut[tid] = o;
}
