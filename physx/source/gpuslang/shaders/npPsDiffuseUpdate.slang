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

// npPsDiffuseUpdate.slang
//
// Slang port of the diffuse-particle velocity/position integration in
// ps_diffuseParticleCompact (diffuseParticles.cu L332-363).  Per diffuse
// particle it reconstructs the velocity from the position delta, classifies the
// particle by neighbor count (spray < 4, foam < 8, bubble otherwise), applies
// the corresponding drag/buoyancy update, clamps the speed to mMaxVelocity, and
// integrates the position.
//
//   vel = (pos - oldPos) * (1/dt)                 // reciprocal scalar multiply
//   if (classW < 4):   newVel = vel * (1 - airDrag*dt)                      // spray
//   elif (classW < 8): newVel = velAvg                                      // foam
//   else:              newVel = vel - (1+buoyancy)*gravity*dt               // bubble
//                                   + bubbleDrag*(velAvg - vel)
//   if (newVel.magnitudeSquared() > 0):
//     newVel = PxMin(newVel.magnitude(), maxVel) * newVel.getNormalized()
//   newPosCorr = pos + (newVel - vel) * dt
//   lifeTime   = max(life - dt, 0)                // lifeDelta = dt
//
// getNormalized(v) = magnitudeSquared()>0 ? v * PxRecipSqrt(magSq) : 0, and
// PxRecipSqrt(m) = 1.0/sqrtf(m) (IEEE, NOT the fast __frsqrt_rn/rsqrtf paths).
// Since m = magSq and mag = sqrt(magSq), 1.0/mag is bit-identical to
// 1.0/sqrtf(magSq).  PxMin(a,b) = selectMin = a<b?a:b.  All float arith
// NoContraction-stamped; +-0 compared with eqFZ.

float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float pmin(float a, float b) { return (a < b) ? a : b; }

struct DpIn
{
    float4 pos;      // pos.xyz, life
    float4 oldPos;   // oldPos.xyz, _
    float4 velAvg;   // velAvg.xyz, classW (neighbor-count class indicator)
    float4 gravity;  // gravity.xyz, _
    float4 params;   // airDrag, buoyancy, bubbleDrag, dt
    float4 params2;  // maxVel, _, _, _
}

struct DpOut
{
    float4 newPos;   // newPosCorr.xyz, lifeTime
    float4 newVel;   // newVelCorr.xyz, _
}

struct DpParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<DpParams> gParams;

RWStructuredBuffer<DpIn> gIn;    // binding 0
RWStructuredBuffer<DpOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void psDiffuseUpdate(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const DpIn s = gIn[tid];

    const float3 pos = s.pos.xyz;
    const float life = s.pos.w;
    const float3 oldPos = s.oldPos.xyz;
    const float3 velAvg = s.velAvg.xyz;
    const float classW = s.velAvg.w;
    const float3 gravity = s.gravity.xyz;
    const float airDrag = s.params.x, buoyancy = s.params.y, bubbleDrag = s.params.z, dt = s.params.w;
    const float maxVel = s.params2.x;

    const float3 vel = (pos - oldPos) * (1.0f / dt);

    float3 newVel;
    if (classW < 4.0f)
        newVel = vel * (1.0f - airDrag * dt);                              // spray
    else if (classW < 8.0f)
        newVel = velAvg;                                                   // foam
    else
        newVel = vel - (1.0f + buoyancy) * gravity * dt                    // bubble
                     + bubbleDrag * (velAvg - vel);

    const float magSq = dot3(newVel, newVel);
    if (magSq > 0.0f)
    {
        const float mag = sqrt(magSq);
        const float3 nrm = newVel * (1.0f / mag);   // getNormalized: v * (1/sqrt(magSq))
        newVel = pmin(mag, maxVel) * nrm;
    }

    const float3 newPosCorr = pos + (newVel - vel) * dt;
    const float lifeTime = ((life - dt) > 0.0f) ? (life - dt) : 0.0f;      // max(life - dt, 0)

    DpOut o;
    o.newPos = float4(newPosCorr, lifeTime);
    o.newVel = float4(newVel, 0.0f);
    gOut[tid] = o;
}
