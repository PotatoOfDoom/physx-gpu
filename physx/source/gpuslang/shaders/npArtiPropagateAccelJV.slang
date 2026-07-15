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

// npArtiPropagateAccelJV.slang
//
// Slang port of propagateAccelerationW_nothingpreloaded_updateJointDofVels
// (articulationDynamic.cuh L329-386): the same parent->child acceleration
// propagation as apAccel (tranche 192), but it additionally WRITES BACK the
// updated joint dof velocities.  (In CUDA the inputs come from block data
// loads; here they are modelled as plain per-dof arrays — the loads carry no
// computation.)
//
// pDeltaV = translateSpatialVector(-parentToChild, parentAccel).
// tJointDelta starts at mDeferredQstZ, then tJointDelta[i] -= IsW[i].innerProduct
//   (pDeltaV)  (== QMinusSTZ[i] - temp of tranche 192).
// jDelta = invStIsT[i] . tJointDelta (full 3-lane dot).
// mJointVelocities[i] = jVel[i] + jDelta   (*** the extra output ***).
// pDeltaV += motionMatrix[i] * jDelta.
//
// Same traps as 192: NEGATED translate offset, the spatial-duality innerProduct
// (bottom.top + top.bottom), and the two-pass ordering.  All float arith
// NoContraction-stamped; +-0 eqFZ.

static const uint MAX_DOF = 3u;

struct JvIn
{
    float4 paTop;      // parentAccel.top
    float4 paBot;      // parentAccel.bottom
    float4 p2c;        // parentToChild.xyz
    uint4 cfg;         // dofCount
    float4 qstz;       // xyz = mDeferredQstZ[0..2]
    float4 jvel;       // xyz = joint velocities initial
    float4 mmTop[3];   // motionMatrix[i].top
    float4 mmBot[3];   // motionMatrix[i].bottom
    float4 isTop[3];   // IsW[i].top
    float4 isBot[3];   // IsW[i].bottom
    float4 invS[3];    // invStIsT[i].xyz
}

struct JvOut
{
    float4 pdTop;      // pDeltaV.top
    float4 pdBot;      // pDeltaV.bottom
    float4 jvout;      // xyz = joint velocities final
}

struct JvParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<JvParams> gParams;

RWStructuredBuffer<JvIn> gIn;    // binding 0
RWStructuredBuffer<JvOut> gOut;  // binding 1

float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float3 cross3(float3 a, float3 b)  // PxVec3::cross(a, b) = a x b
{
    return float3(a.y * b.z - a.z * b.y,
                  a.z * b.x - a.x * b.z,
                  a.x * b.y - a.y * b.x);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void apAccelJV(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const JvIn s = gIn[tid];

    const uint dofCount = min(s.cfg.x, MAX_DOF);

    // pDeltaV = translateSpatialVector(-parentToChild, parentAccel)
    const float3 paT = s.paTop.xyz;
    const float3 negP2c = -s.p2c.xyz;
    float3 pdT = paT;
    float3 pdB = s.paBot.xyz + cross3(negP2c, paT);

    // tJointDelta starts at mDeferredQstZ, joint velocities start at jVel
    float tjd[3] = { s.qstz.x, s.qstz.y, s.qstz.z };
    const float jv[3] = { s.jvel.x, s.jvel.y, s.jvel.z };
    float jvo[3] = { jv[0], jv[1], jv[2] };

    // pass 1: tJointDelta[i] -= IsW[i].innerProduct(pDeltaV)
    for (uint ind = 0u; ind < MAX_DOF; ++ind)
    {
        if (ind < dofCount)
        {
            tjd[ind] -= dot3(s.isBot[ind].xyz, pdT) + dot3(s.isTop[ind].xyz, pdB);
        }
    }

    // pass 2: jDelta = invStIsT[i] . tJointDelta; write joint vel; accumulate pDeltaV
    for (uint ind = 0u; ind < MAX_DOF; ++ind)
    {
        if (ind < dofCount)
        {
            const float3 iS = s.invS[ind].xyz;
            const float jDelta = iS.x * tjd[0] + iS.y * tjd[1] + iS.z * tjd[2];
            jvo[ind] = jv[ind] + jDelta;   // mJointVelocities[i] = jVel[i] + jDelta
            pdT += s.mmTop[ind].xyz * jDelta;
            pdB += s.mmBot[ind].xyz * jDelta;
        }
    }

    JvOut o;
    o.pdTop = float4(pdT, 0.0f);
    o.pdBot = float4(pdB, 0.0f);
    o.jvout = float4(jvo[0], jvo[1], jvo[2], 0.0f);
    gOut[tid] = o;
}
