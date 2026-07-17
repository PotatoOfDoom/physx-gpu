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

// npArtiPropagateZA.slang
//
// Slang port of the prismatic/revolute (single-dof) branch of
// computePropagateSpatialInertia_ZA_ZIc (forwardDynamic2.cu L1420-1448): the
// articulated-inertia + zero-acceleration-force propagation term for a 1-dof
// joint.
//   stIs   = sa.innerProduct(Is) + armature
//   iStIs  = (stIs > 0) ? 1/stIs : 0
//   isInvD = Is * iStIs
//   stI    = UnAlignedSpatialVector(Is.bottom, Is.top)   // top/bottom SWAPPED
//   stZ    = sa.innerProduct(Z);  stZInt = sa.innerProduct(ZIcInt)
//   qstZ   = -stZ;  qstZIcInternal = (jF ? jF0 : 0) - stZInt
//   ZA    += isInvD * qstZ;  ZAInt += isInvD * qstZIcInternal
//   return constructSpatialMatrix(isInvD, stI)
//
// innerProduct is the spatial motion/force DUALITY product
// (a.bottom.dot(b.top) + a.top.dot(b.bottom)); operator*(scalar) and += are
// componentwise; constructSpatialMatrix builds the block outer product
// (topLeft=isInvD.top(x)stI.top, topRight=isInvD.top(x)stI.bottom,
// bottomLeft=isInvD.bottom(x)stI.top).  All float arith NoContraction-stamped;
// +-0 eqFZ.

static const uint JF_NULL = 1u;

float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
// spatial duality inner product a.innerProduct(b) = a.bottom.b.top + a.top.b.bottom
float inner(float3 aTop, float3 aBot, float3 bTop, float3 bBot) { return dot3(aBot, bTop) + dot3(aTop, bBot); }

struct PpIn
{
    float4 saTop, saBot;     // sa (motionMatrix)
    float4 isTop, isBot;     // Is
    float4 zTop, zBot;       // Z
    float4 zicTop, zicBot;   // ZIcInt
    float4 zaTop, zaBot;     // ZA initial
    float4 zaiTop, zaiBot;   // ZAInt initial
    float4 misc;             // armature, jF0, jFnull(1=null), 0
}

struct PpOut
{
    float4 scalars;              // iStIs, qstZ, qstZIcInternal, 0
    float4 isInvDTop, isInvDBot; // isInvD
    float4 zaTop, zaBot;         // ZA updated
    float4 zaiTop, zaiBot;       // ZAInt updated
    float4 tlC0, tlC1, tlC2;     // SM topLeft
    float4 trC0, trC1, trC2;     // SM topRight
    float4 blC0, blC1, blC2;     // SM bottomLeft
}

struct PpParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<PpParams> gParams;

RWStructuredBuffer<PpIn> gIn;    // binding 0
RWStructuredBuffer<PpOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void propZA(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const PpIn s = gIn[tid];

    const float3 saT = s.saTop.xyz, saB = s.saBot.xyz;
    const float3 isT = s.isTop.xyz, isB = s.isBot.xyz;
    const float armature = s.misc.x;
    const float jF0 = s.misc.y;
    const bool jFnull = asuint(s.misc.z) == JF_NULL;

    const float stIs = inner(saT, saB, isT, isB) + armature;
    const float iStIs = (stIs > 0.0f) ? (1.0f / stIs) : 0.0f;

    const float3 isInvDTop = isT * iStIs;   // isInvD = Is * iStIs
    const float3 isInvDBot = isB * iStIs;

    // stI = (Is.bottom, Is.top)  -> stI.top = Is.bottom, stI.bottom = Is.top
    const float3 stiT = isB;
    const float3 stiB = isT;

    const float stZ = inner(saT, saB, s.zTop.xyz, s.zBot.xyz);
    const float stZInt = inner(saT, saB, s.zicTop.xyz, s.zicBot.xyz);

    const float qstZ = -stZ;
    const float qstZIcInternal = (jFnull ? 0.0f : jF0) - stZInt;

    const float3 zaTop = s.zaTop.xyz + isInvDTop * qstZ;   // ZA += isInvD * qstZ
    const float3 zaBot = s.zaBot.xyz + isInvDBot * qstZ;
    const float3 zaiTop = s.zaiTop.xyz + isInvDTop * qstZIcInternal;
    const float3 zaiBot = s.zaiBot.xyz + isInvDBot * qstZIcInternal;

    PpOut o;
    o.scalars = float4(iStIs, qstZ, qstZIcInternal, 0.0f);
    o.isInvDTop = float4(isInvDTop, 0.0f);
    o.isInvDBot = float4(isInvDBot, 0.0f);
    o.zaTop = float4(zaTop, 0.0f);
    o.zaBot = float4(zaBot, 0.0f);
    o.zaiTop = float4(zaiTop, 0.0f);
    o.zaiBot = float4(zaiBot, 0.0f);

    // constructSpatialMatrix(isInvD, stI): block outer product
    o.tlC0 = float4(isInvDTop * stiT.x, 0.0f);
    o.tlC1 = float4(isInvDTop * stiT.y, 0.0f);
    o.tlC2 = float4(isInvDTop * stiT.z, 0.0f);
    o.trC0 = float4(isInvDTop * stiB.x, 0.0f);
    o.trC1 = float4(isInvDTop * stiB.y, 0.0f);
    o.trC2 = float4(isInvDTop * stiB.z, 0.0f);
    o.blC0 = float4(isInvDBot * stiT.x, 0.0f);
    o.blC1 = float4(isInvDBot * stiT.y, 0.0f);
    o.blC2 = float4(isInvDBot * stiT.z, 0.0f);
    gOut[tid] = o;
}
