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

// npArtiConstructSM.slang
//
// Slang port of Dy::SpatialMatrix::constructSpatialMatrix(Is, stI)
// (DyFeatherstoneArticulationUtils.h L364-386): builds a SpatialMatrix from two
// spatial vectors as the block outer product.  Each block column is one spatial
// vector scaled by a scalar component of the other:
//   topLeft    cols = Is.top    * stI.top.{x,y,z}
//   topRight   cols = Is.top    * stI.bottom.{x,y,z}
//   bottomLeft cols = Is.bottom * stI.top.{x,y,z}
// (top=angular, bottom=linear; the implicit bottom-right block is not stored.)
//
// Only isolated vector*scalar products (no mul-add), so no fma-contraction is
// possible and the shader is NOT in PXG_NOCONTRACT_SHADERS; +-0 tolerated via
// eqFZ.

struct CsIn
{
    float4 isTop, isBot;    // Is.top / Is.bottom
    float4 stiTop, stiBot;  // stI.top / stI.bottom
}

struct CsOut
{
    float4 tlC0, tlC1, tlC2;   // topLeft columns
    float4 trC0, trC1, trC2;   // topRight columns
    float4 blC0, blC1, blC2;   // bottomLeft columns
}

struct CsParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<CsParams> gParams;

RWStructuredBuffer<CsIn> gIn;    // binding 0
RWStructuredBuffer<CsOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void constructSM(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const CsIn s = gIn[tid];

    const float3 isT = s.isTop.xyz;
    const float3 isB = s.isBot.xyz;
    const float3 stT = s.stiTop.xyz;
    const float3 stB = s.stiBot.xyz;

    CsOut o;
    // topLeft = Is.top (x) stI.top
    o.tlC0 = float4(isT * stT.x, 0.0f);
    o.tlC1 = float4(isT * stT.y, 0.0f);
    o.tlC2 = float4(isT * stT.z, 0.0f);
    // topRight = Is.top (x) stI.bottom
    o.trC0 = float4(isT * stB.x, 0.0f);
    o.trC1 = float4(isT * stB.y, 0.0f);
    o.trC2 = float4(isT * stB.z, 0.0f);
    // bottomLeft = Is.bottom (x) stI.top
    o.blC0 = float4(isB * stT.x, 0.0f);
    o.blC1 = float4(isB * stT.y, 0.0f);
    o.blC2 = float4(isB * stT.z, 0.0f);
    gOut[tid] = o;
}
