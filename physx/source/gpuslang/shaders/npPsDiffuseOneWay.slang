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

// npPsDiffuseOneWay.slang
//
// Slang port of the diffuse-particle one-way (static) collision position
// correction in ps_diffuseParticleOneWayCollision (diffuseParticles.cu
// L143-159).  Per diffuse particle it walks its static-contact list and pushes
// the position out along each contact normal by that contact's penetration:
//
//   contactCount = PxMin(MaxStaticContactsPerParticle, contactCounts[pi])  // Max = 12
//   if (contactCount):
//     posCorr = PxLoad3(newPos)
//     for each contact c:
//       deltaP = -surfaceNormal * penetration
//       posCorr += deltaP
//     newPos = (posCorr.xyz, newPos.w)
//
// Modeled as a fixed-order K=8 accumulation (K < MaxStaticContactsPerParticle=12,
// so the PxMin clamp never binds for counts in [0,8]).  deltaP = (-normal)*pen
// per component; posCorr += deltaP is a strict left-to-right fold (each += one
// rounding), so contact order is significant.  When contactCount==0 the loop is
// empty and posCorr stays the original position (CUDA leaves newPos unwritten --
// output-identical).  The w channel passes through unchanged.  All float
// mul/add NoContraction-stamped; +-0 compared with eqFZ.

static const uint K = 8u;

struct CcIn
{
    float4 pos;          // newPos.xyz, w (pass-through)
    float4 cnt;          // asfloat(contactCount), _, _, _
    float4 contacts[8];  // per contact: surfaceNormal.xyz, penetration (w)
}

struct CcOut
{
    float4 pos;          // posCorr.xyz, w
}

struct CcParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<CcParams> gParams;

RWStructuredBuffer<CcIn> gIn;    // binding 0
RWStructuredBuffer<CcOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void psDiffuseOneWay(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const CcIn s = gIn[tid];

    const uint rawCount = asuint(s.cnt.x);
    const uint contactCount = (rawCount < K) ? rawCount : K;   // PxMin(K, count), K < Max=12

    float3 posCorr = s.pos.xyz;
    for (uint c = 0u; c < contactCount; ++c)
    {
        const float3 surfaceNormal = s.contacts[c].xyz;
        const float penetration = s.contacts[c].w;
        const float3 deltaP = -surfaceNormal * penetration;
        posCorr += deltaP;
    }

    CcOut o;
    o.pos = float4(posCorr, s.pos.w);
    gOut[tid] = o;
}
