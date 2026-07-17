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

// npArtiPropagateImpDof.slang
//
// Slang port of propagateImpulseDofAligned (internalConstraints2.cu L53-62):
// propagate a single-dof zero-acceleration impulse from a child link into its
// parent's spatial frame.
//   stZ  = sa.innerProduct(Z)                 // spatial motion/force duality
//   temp = isInvD * stZ
//   qstZ = -stZ
//   return translateSpatialVector(childToParent, Z - temp)
//          = ((Z-temp).top, (Z-temp).bottom + childToParent x (Z-temp).top)
//
// innerProduct is the duality product a.bottom.b.top + a.top.b.bottom;
// operator*(scalar) scales top+bottom; operator- is componentwise;
// translateSpatialVector keeps top, adds offset x top to bottom.
// All float arith NoContraction-stamped; +-0 eqFZ.

float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float3 cross3(float3 a, float3 b) { return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
// spatial duality inner product a.innerProduct(b) = a.bottom.b.top + a.top.b.bottom
float inner(float3 aTop, float3 aBot, float3 bTop, float3 bBot) { return dot3(aBot, bTop) + dot3(aTop, bBot); }

struct PiIn
{
    float4 isInvDTop, isInvDBot;   // isInvD
    float4 saTop, saBot;           // sa (motion vector)
    float4 zTop, zBot;             // Z
    float4 c2p;                    // childToParent (xyz), _
}

struct PiOut
{
    float4 resTop, resBot;         // propagated impulse in parent space
    float4 scalars;                // qstZ, _, _, _
}

struct PiParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<PiParams> gParams;

RWStructuredBuffer<PiIn> gIn;    // binding 0
RWStructuredBuffer<PiOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void propImpDof(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const PiIn s = gIn[tid];

    const float3 isInvDTop = s.isInvDTop.xyz, isInvDBot = s.isInvDBot.xyz;
    const float3 zTop = s.zTop.xyz, zBot = s.zBot.xyz;
    const float3 c2p = s.c2p.xyz;

    const float stZ = inner(s.saTop.xyz, s.saBot.xyz, zTop, zBot);

    const float3 tempTop = isInvDTop * stZ;   // temp = isInvD * stZ
    const float3 tempBot = isInvDBot * stZ;

    const float qstZ = -stZ;

    const float3 xTop = zTop - tempTop;        // X = Z - temp
    const float3 xBot = zBot - tempBot;

    // translateSpatialVector(childToParent, X)
    const float3 resTop = xTop;
    const float3 resBot = xBot + cross3(c2p, xTop);

    PiOut o;
    o.resTop = float4(resTop, 0.0f);
    o.resBot = float4(resBot, 0.0f);
    o.scalars = float4(qstZ, 0.0f, 0.0f, 0.0f);
    gOut[tid] = o;
}
