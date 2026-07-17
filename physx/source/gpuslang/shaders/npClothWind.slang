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

// npClothWind.slang
//
// Slang port of disneyWindModelEffect (particleSystem.cuh L71-96): the "Frozen"
// aerodynamic wind force applied to a cloth triangle (drag + lift), using the
// current deformed triangle area (option 1).
//   x01 = x1-x0 ; x02 = x2-x0 ; n = x01 x x02
//   v    = (vel0+vel1+vel2) * 0.3333
//   vrel = wind - v
//   if vrel.dot(n) < 0: n *= -1                    // face the wind
//   coef = 0.25 * airDensity * dt * inverseMass
//   return coef * ((drag-lift)*vrel.dot(n)*vrel + lift*|vrel|^2*n)
//
// The vrel.dot(n) in the return uses the possibly-flipped n.  Constant 0.3333f
// is the exact source literal (not 1/3).  All float arith NoContraction-stamped;
// +-0 eqFZ.

float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float3 cross3(float3 a, float3 b) { return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }

struct WdIn
{
    float4 x0, x1, x2;         // .xyz
    float4 vel0, vel1, vel2;   // .xyz
    float4 wind;               // .xyz
    float4 params;             // inverseMass, drag, lift, dt
    float4 params2;            // airDensity, _, _, _
}

struct WdOut
{
    float4 force;              // .xyz aerodynamic force
}

struct WdParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<WdParams> gParams;

RWStructuredBuffer<WdIn> gIn;    // binding 0
RWStructuredBuffer<WdOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void clothWind(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const WdIn s = gIn[tid];

    const float3 x0 = s.x0.xyz, x1 = s.x1.xyz, x2 = s.x2.xyz;
    const float3 vel0 = s.vel0.xyz, vel1 = s.vel1.xyz, vel2 = s.vel2.xyz;
    const float3 wind = s.wind.xyz;
    const float inverseMass = s.params.x, drag = s.params.y, lift = s.params.z, dt = s.params.w;
    const float airDensity = s.params2.x;

    const float3 x01 = x1 - x0;
    const float3 x02 = x2 - x0;
    float3 n = cross3(x01, x02);

    const float3 v = (vel0 + vel1 + vel2) * 0.3333f;
    const float3 vrel = wind - v;

    if (dot3(vrel, n) < 0.0f)
    {
        n = n * -1.0f;
    }

    const float coef = 0.25f * airDensity * dt * inverseMass;

    const float3 force = coef * ((drag - lift) * dot3(vrel, n) * vrel + lift * dot3(vrel, vrel) * n);

    WdOut o;
    o.force = float4(force, 0.0f);
    gOut[tid] = o;
}
