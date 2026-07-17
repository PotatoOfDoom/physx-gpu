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

// npArtiAccumLinkImp.slang
//
// Slang port of accumulateLinkImpulsesAndLinkVelocities (internalConstraints2.cu
// L1554-1575): fold a scalar constraint impulse deltaF into the parent/child
// link velocity-delta, velocity, and impulse accumulators.
//   pDelta = deltaVA * -deltaF;  cDelta = deltaVB * -deltaF;
//   parentDeltaV += pDelta;  deltaV += cDelta;
//   parentV += pDelta;       childV += cDelta;
//   parentImp += row0 * deltaF;  impulse -= row1 * deltaF;
//
// operator*(scalar) scales top+bottom; += / -= are componentwise.
// All float arith NoContraction-stamped; +-0 eqFZ.

struct AiIn
{
    float4 deltaVATop, deltaVABot;   // deltaVA
    float4 deltaVBTop, deltaVBBot;   // deltaVB
    float4 row0Top, row0Bot;         // row0
    float4 row1Top, row1Bot;         // row1
    float4 parentDeltaVTop, parentDeltaVBot;
    float4 deltaVTop, deltaVBot;
    float4 parentVTop, parentVBot;
    float4 childVTop, childVBot;
    float4 parentImpTop, parentImpBot;
    float4 impulseTop, impulseBot;
    float4 misc;                     // deltaF, _, _, _
}

struct AiOut
{
    float4 parentDeltaVTop, parentDeltaVBot;
    float4 deltaVTop, deltaVBot;
    float4 parentVTop, parentVBot;
    float4 childVTop, childVBot;
    float4 parentImpTop, parentImpBot;
    float4 impulseTop, impulseBot;
}

struct AiParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<AiParams> gParams;

RWStructuredBuffer<AiIn> gIn;    // binding 0
RWStructuredBuffer<AiOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void accumLinkImp(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const AiIn s = gIn[tid];

    const float deltaF = s.misc.x;

    // pDelta = deltaVA * -deltaF;  cDelta = deltaVB * -deltaF
    const float3 pDeltaTop = s.deltaVATop.xyz * -deltaF;
    const float3 pDeltaBot = s.deltaVABot.xyz * -deltaF;
    const float3 cDeltaTop = s.deltaVBTop.xyz * -deltaF;
    const float3 cDeltaBot = s.deltaVBBot.xyz * -deltaF;

    AiOut o;
    o.parentDeltaVTop = float4(s.parentDeltaVTop.xyz + pDeltaTop, 0.0f);
    o.parentDeltaVBot = float4(s.parentDeltaVBot.xyz + pDeltaBot, 0.0f);
    o.deltaVTop = float4(s.deltaVTop.xyz + cDeltaTop, 0.0f);
    o.deltaVBot = float4(s.deltaVBot.xyz + cDeltaBot, 0.0f);
    o.parentVTop = float4(s.parentVTop.xyz + pDeltaTop, 0.0f);
    o.parentVBot = float4(s.parentVBot.xyz + pDeltaBot, 0.0f);
    o.childVTop = float4(s.childVTop.xyz + cDeltaTop, 0.0f);
    o.childVBot = float4(s.childVBot.xyz + cDeltaBot, 0.0f);

    // parentImp += row0 * deltaF;  impulse -= row1 * deltaF
    o.parentImpTop = float4(s.parentImpTop.xyz + s.row0Top.xyz * deltaF, 0.0f);
    o.parentImpBot = float4(s.parentImpBot.xyz + s.row0Bot.xyz * deltaF, 0.0f);
    o.impulseTop = float4(s.impulseTop.xyz - s.row1Top.xyz * deltaF, 0.0f);
    o.impulseBot = float4(s.impulseBot.xyz - s.row1Bot.xyz * deltaF, 0.0f);
    gOut[tid] = o;
}
