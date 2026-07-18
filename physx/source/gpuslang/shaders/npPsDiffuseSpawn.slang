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

// npPsDiffuseSpawn.slang
//
// Slang port of the diffuse-particle spawn decision + spawn-state computation in
// ps_diffuseParticleCreate (diffuseParticles.cu L456-495).  Per fluid particle it
// forms a spawn "intensity" from the divergence/pressure potentials and kinetic
// energy, decides whether to emit a diffuse particle (random-weighted threshold
// test), and computes the emitted particle's lifetime and jittered spawn position.
//
//   kineticEnergy = dot3(vi, vi) * kineticEnergyWeight
//   divergence    = divergenceWeight * ptnts.x
//   pressure      = pressureWeight   * ptnts.y
//   intensity     = pressure - divergence + kineticEnergy      // (p - d) + ke
//   spawn         = (r0 * intensity > threshold)
//   lifeScale     = fminf(intensity / threshold, 1) * r1
//   lifetime      = 1 + lifeScale * (lifetimeMax - 1)          // lifeMin = 1
//   q             = xi - r2*vi*dt + PxVec3(r1,r2,r3)*restOffset*0.25
//
// dot3(a,b) = a.x*b.x + a.y*b.y + a.z*b.z (mathsExtensions.h).  fminf(a,1) is the
// IEEE min, transcribed as a ternary.  `intensity/threshold` is a plain
// single-rounding division.  q per component: (xi.i - (r2*vi.i)*dt) +
// ((r_i*restOffset)*0.25) with (r_x,r_y,r_z)=(r1,r2,r3).  All float arith
// NoContraction-stamped; +-0 compared with eqFZ.

float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

struct SpIn
{
    float4 vel;      // vi.xyz, _
    float4 pos;      // xi.xyz, _
    float4 ptn;      // ptnts.x (divergence potential), ptnts.y (pressure potential), _, _
    float4 wts;      // kineticEnergyWeight, divergenceWeight, pressureWeight, threshold
    float4 rnd;      // r0, r1, r2, r3
    float4 params;   // restOffset, dt, lifetimeMax, _
}

struct SpOut
{
    float4 q;        // spawn position q.xyz, lifetime
    float4 misc;     // intensity, spawn (1.0/0.0), _, _
}

struct SpParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<SpParams> gParams;

RWStructuredBuffer<SpIn> gIn;    // binding 0
RWStructuredBuffer<SpOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void psDiffuseSpawn(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const SpIn s = gIn[tid];

    const float3 vi = s.vel.xyz;
    const float3 xi = s.pos.xyz;
    const float divergencePot = s.ptn.x, pressurePot = s.ptn.y;
    const float kineticEnergyWeight = s.wts.x, divergenceWeight = s.wts.y, pressureWeight = s.wts.z, threshold = s.wts.w;
    const float r0 = s.rnd.x, r1 = s.rnd.y, r2 = s.rnd.z, r3 = s.rnd.w;
    const float restOffset = s.params.x, dt = s.params.y, lifetimeMax = s.params.z;

    const float kineticEnergy = dot3(vi, vi) * kineticEnergyWeight;
    const float divergence = divergenceWeight * divergencePot;
    const float pressure = pressureWeight * pressurePot;
    const float intensity = pressure - divergence + kineticEnergy;

    const float spawn = (r0 * intensity > threshold) ? 1.0f : 0.0f;

    const float lifeMin = 1.0f;
    const float lifeMax = lifetimeMax;
    const float ratio = intensity / threshold;
    const float lifeScale = ((ratio < 1.0f) ? ratio : 1.0f) * r1;   // fminf(intensity/threshold, 1) * r1
    const float lifetime = lifeMin + lifeScale * (lifeMax - lifeMin);

    const float3 jitter = float3(r1, r2, r3) * restOffset * 0.25f;
    const float3 q = xi - r2 * vi * dt + jitter;

    SpOut o;
    o.q = float4(q, lifetime);
    o.misc = float4(intensity, spawn, 0.0f, 0.0f);
    gOut[tid] = o;
}
