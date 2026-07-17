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

// npPsSolvePC.slang
//
// Slang port of solvePCOutputDeltaV (particlesystem.cu L1421-1471): the PBD
// particle-contact constraint solve output.  Given a contact normal/error and a
// position delta, it computes the adhesion-clamped normal impulse increment
// deltaF (accumulating appliedForce.x), then distributes the Coulomb friction
// budget across the two tangent directions (updating appliedForce.y) to produce
// deltaFr0/deltaFr1.
//
//   separation = error - normal.dot(delta)
//   clamp = -appliedForce.x
//   if (adhesionRadiusScale > 0 && adhesion > 0):
//     adhesionRadius = restOffset*adhesionRadiusScale - restOffset
//     if (separation < adhesionRadiusScale):
//       clamp = PxMin(clamp, -PxMax(0, adhesion*quart(1 - separation/adhesionRadius)
//                                        * PxMax(0, separation) * velMultiplier))
//   deltaF = PxMax(clamp, (-separation - normalVel*dt) * velMultiplier)
//   appliedForce.x = PxMax(0, deltaF + appliedForce.x)
//   friction = appliedForce.x * frictionCoefficient
//   requiredFiForce = (-(frici.dot(delta)) + tanVeli*dt) * vmFi
//   requiredForce = sqrt(rF0*rF0 + rF1*rF1)
//   recip = requiredForce >= 1e-16 ? 1/requiredForce : 0
//   ratioi = requiredFiForce * recip
//   deltaFr = PxMin(requiredForce + appliedForce.y, friction) - appliedForce.y
//   appliedForce.y += deltaFr;  deltaFri = deltaFr * ratioi
//
// PxMin/PxMax for float = selectMin/selectMax = a<b?a:b / a>b?a:b (transcribed as
// ternaries).  quart(x)=x*x*x*x.  `separation/adhesionRadius` and the guarded
// `1/requiredForce` are plain single-rounding divisions.  All float
// mul/add/sub/div NoContraction-stamped; +-0 compared with eqFZ.

float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float quart(float x) { return x * x * x * x; }
float pmin(float a, float b) { return (a < b) ? a : b; }   // PxMin = selectMin
float pmax(float a, float b) { return (a > b) ? a : b; }   // PxMax = selectMax

struct PcIn
{
    float4 normal;   // normal.xyz, error
    float4 delta;    // delta.xyz, normalVel
    float4 fric0;    // fric0.xyz, tanVel0
    float4 fric1;    // fric1.xyz, tanVel1
    float4 vm;       // velMultiplier, vmF0, vmF1, frictionCoefficient
    float4 adh;      // restOffset, adhesion, adhesionRadiusScale, dt
    float4 applied;  // appliedForce.x, appliedForce.y, _, _
}

struct PcOut
{
    float4 f;        // deltaF, appliedForce.x (new), appliedForce.y (new), _
    float4 fr;       // deltaFr0, deltaFr1, _, _
}

struct PcParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<PcParams> gParams;

RWStructuredBuffer<PcIn> gIn;    // binding 0
RWStructuredBuffer<PcOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void psSolvePC(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const PcIn s = gIn[tid];

    const float3 normal = s.normal.xyz;
    const float error = s.normal.w;
    const float3 delta = s.delta.xyz;
    const float normalVel = s.delta.w;
    const float3 fric0 = s.fric0.xyz;
    const float tanVel0 = s.fric0.w;
    const float3 fric1 = s.fric1.xyz;
    const float tanVel1 = s.fric1.w;
    const float velMultiplier = s.vm.x, vmF0 = s.vm.y, vmF1 = s.vm.z, frictionCoefficient = s.vm.w;
    const float restOffset = s.adh.x, adhesion = s.adh.y, adhesionRadiusScale = s.adh.z, dt = s.adh.w;
    float appliedForceX = s.applied.x, appliedForceY = s.applied.y;

    const float separation = (error - dot3(normal, delta));
    float clamp = -appliedForceX;

    if (adhesionRadiusScale > 0.0f && adhesion > 0.0f)
    {
        const float adhesionRadius = restOffset * adhesionRadiusScale - restOffset;
        if (separation < adhesionRadiusScale)
        {
            clamp = pmin(clamp, -pmax(0.0f, adhesion * quart(1.0f - separation / adhesionRadius) * pmax(0.0f, separation) * velMultiplier));
        }
    }

    const float deltaF = pmax(clamp, (-separation - normalVel * dt) * velMultiplier);
    appliedForceX = pmax(0.0f, deltaF + appliedForceX);

    const float friction = appliedForceX * frictionCoefficient;

    const float biasedF0 = -(dot3(fric0, delta));
    const float biasedF1 = -(dot3(fric1, delta));
    const float requiredF0Force = (biasedF0 + tanVel0 * dt) * vmF0;
    const float requiredF1Force = (biasedF1 + tanVel1 * dt) * vmF1;

    const float requiredForce = sqrt(requiredF0Force * requiredF0Force + requiredF1Force * requiredF1Force);
    const float recipRequiredForce = (requiredForce >= 1e-16f) ? (1.0f / requiredForce) : 0.0f;

    const float ratio0 = requiredF0Force * recipRequiredForce;
    const float ratio1 = requiredF1Force * recipRequiredForce;

    const float deltaFr = pmin(requiredForce + appliedForceY, friction) - appliedForceY;
    appliedForceY = appliedForceY + deltaFr;

    const float deltaFr0 = deltaFr * ratio0;
    const float deltaFr1 = deltaFr * ratio1;

    PcOut o;
    o.f = float4(deltaF, appliedForceX, appliedForceY, 0.0f);
    o.fr = float4(deltaFr0, deltaFr1, 0.0f, 0.0f);
    gOut[tid] = o;
}
