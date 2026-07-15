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

// npArtiPropagateAccel.slang
//
// Slang port of propagateAccelerationW_everythingPreLoaded (articulationDynamic.
// cuh L206-242): the dual of propagateImpulseW (tranche 191).  Propagates an
// acceleration (or velocity) from a PARENT link DOWN to a child link through
// the inbound joint (Mirtich p121).
//
// pDeltaV = translateSpatialVector(-parentToChild, parentLinkAccelerationW).
// Pass 1 (uses the ORIGINAL translated pDeltaV): tJointDelta[i] = QMinusSTZ[i]
//   - IsW[i].innerProduct(pDeltaV).
// Pass 2 (accumulates): jDelta = invStISW[i] . tJointDelta (full 3-lane dot),
//   pDeltaV += motionMatrix[i] * jDelta.
//
// ** Traps **:
//  - the translate offset is NEGATED: -parentToChild.
//  - innerProduct is the spatial motion/force DUALITY product (bottom.top +
//    top.bottom), NOT the Euclidean dot.  Same as tranche 191.
//  - two-pass: every tJointDelta is formed from the pre-accumulation pDeltaV;
//    only pass 2 mutates pDeltaV.  tJointDelta lanes >= dofCount stay 0, so the
//    jDelta 3-dot is effectively masked.
//
// top=angular, bottom=linear.  All float arith NoContraction-stamped; +-0 eqFZ.

static const uint MAX_DOF = 3u;

struct AaIn
{
    float4 paTop;      // parentLinkAcceleration.top (angular)
    float4 paBot;      // parentLinkAcceleration.bottom (linear)
    float4 p2c;        // parentToChild.xyz
    uint4 cfg;         // dofCount
    float4 mmTop[3];   // motionMatrix[i].top
    float4 mmBot[3];   // motionMatrix[i].bottom
    float4 isTop[3];   // IsW[i].top
    float4 isBot[3];   // IsW[i].bottom
    float4 invS[3];    // invStISW[i].xyz (row of inverse joint-space inertia)
    float4 qmstz;      // xyz = QMinusSTZ[0..2]
}

struct AaOut
{
    float4 pdTop;      // pDeltaV.top
    float4 pdBot;      // pDeltaV.bottom
}

struct AaParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<AaParams> gParams;

RWStructuredBuffer<AaIn> gIn;    // binding 0
RWStructuredBuffer<AaOut> gOut;  // binding 1

float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float3 cross3(float3 a, float3 b)  // PxVec3::cross(a, b) = a x b
{
    return float3(a.y * b.z - a.z * b.y,
                  a.z * b.x - a.x * b.z,
                  a.x * b.y - a.y * b.x);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void apAccel(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const AaIn s = gIn[tid];

    const uint dofCount = min(s.cfg.x, MAX_DOF);

    // pDeltaV = translateSpatialVector(-parentToChild, parentAccel):
    //   top unchanged; bottom += (-parentToChild) x top
    const float3 paT = s.paTop.xyz;
    const float3 negP2c = -s.p2c.xyz;
    float3 pdT = paT;
    float3 pdB = s.paBot.xyz + cross3(negP2c, paT);

    const float qm[3] = { s.qmstz.x, s.qmstz.y, s.qmstz.z };

    // pass 1: tJointDelta[i] = QMinusSTZ[i] - IsW[i].innerProduct(pDeltaV)
    float tjd[3] = { 0.0f, 0.0f, 0.0f };
    for (uint ind = 0u; ind < MAX_DOF; ++ind)
    {
        if (ind < dofCount)
        {
            const float temp = dot3(s.isBot[ind].xyz, pdT) + dot3(s.isTop[ind].xyz, pdB);
            tjd[ind] = qm[ind] - temp;
        }
    }

    // pass 2: jDelta = invStISW[i] . tJointDelta; pDeltaV += motionMatrix[i] * jDelta
    for (uint ind = 0u; ind < MAX_DOF; ++ind)
    {
        if (ind < dofCount)
        {
            const float3 iS = s.invS[ind].xyz;
            const float jDelta = iS.x * tjd[0] + iS.y * tjd[1] + iS.z * tjd[2];
            pdT += s.mmTop[ind].xyz * jDelta;
            pdB += s.mmBot[ind].xyz * jDelta;
        }
    }

    AaOut o;
    o.pdTop = float4(pdT, 0.0f);
    o.pdBot = float4(pdB, 0.0f);
    gOut[tid] = o;
}
