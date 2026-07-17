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

// npPsSolvePCTGS.slang
//
// Slang port of solvePCOutputDeltaVTGS (particlesystem.cu L1473-1527): the TGS
// variant of the PBD particle-contact constraint solve output.  Differs from the
// PGS variant (npPsSolvePC) by folding position deltas (normalDelta) and the
// velocity-iteration switch into the separation, using PxAbs on the normal force
// for the friction budget, and biasing the tangential terms by tanDelta.
//
//   projectionDt = isVelocityIteration ? 0 : dt
//   separation = (error - normal.dot(delta) + normalDelta) + normalVel*projectionDt
//   clamp = -appliedForce.x
//   if (adhesionRadiusScale > 0 && adhesion > 0):
//     adhesionRadius = restOffset*adhesionRadiusScale - restOffset
//     if (separation < adhesionRadiusScale):
//       clamp = PxMin(clamp, -PxMax(0, adhesion*quart(1 - separation/adhesionRadius)
//                                        * PxMax(0, separation) * velMultiplier))
//   deltaF = PxMax(clamp, -separation * velMultiplier)
//   appliedForce.x = PxMax(0, appliedForce.x + deltaF)
//   maxFriction = PxAbs(appliedForce.x) * frictionCoefficient
//   biasedFi = -(frici.dot(delta) - tanDeltai)
//   requiredFiForce = (biasedFi + tanVeli*projectionDt) * vmFi
//   requiredForce = sqrt(rF0*rF0 + rF1*rF1)
//   recip = requiredForce >= 1e-16 ? 1/requiredForce : 0
//   ratioi = requiredFiForce * recip
//   deltaFr = PxMin(requiredForce + appliedForce.y, maxFriction) - appliedForce.y
//   appliedForce.y += deltaFr;  deltaFri = deltaFr * ratioi
//
// PxMin/PxMax = selectMin/selectMax = a<b?a:b / a>b?a:b (ternaries).  PxAbs=abs.
// quart(x)=x*x*x*x.  `separation/adhesionRadius` and guarded `1/requiredForce`
// are plain single-rounding divisions.  All float arith NoContraction-stamped;
// +-0 compared with eqFZ.

float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float quart(float x) { return x * x * x * x; }
float pmin(float a, float b) { return (a < b) ? a : b; }   // PxMin = selectMin
float pmax(float a, float b) { return (a > b) ? a : b; }   // PxMax = selectMax

struct PcIn
{
    float4 normal;    // normal.xyz, error
    float4 delta;     // delta.xyz, normalVel
    float4 fric0;     // fric0.xyz, tanVel0
    float4 fric1;     // fric1.xyz, tanVel1
    float4 tan;       // tanDelta0, tanDelta1, normalDelta, velMultiplier
    float4 vm;        // vmF0, vmF1, frictionCoefficient, restOffset
    float4 adh;       // adhesion, adhesionRadiusScale, dt, asfloat(isVelocityIteration)
    float4 applied;   // appliedForce.x, appliedForce.y, _, _
}

struct PcOut
{
    float4 f;         // deltaF, appliedForce.x (new), appliedForce.y (new), _
    float4 fr;        // deltaFr0, deltaFr1, _, _
}

struct PcParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<PcParams> gParams;

RWStructuredBuffer<PcIn> gIn;    // binding 0
RWStructuredBuffer<PcOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void psSolvePCTGS(uint3 dtid : SV_DispatchThreadID)
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
    const float tanDelta0 = s.tan.x, tanDelta1 = s.tan.y, normalDelta = s.tan.z, velMultiplier = s.tan.w;
    const float vmF0 = s.vm.x, vmF1 = s.vm.y, frictionCoefficient = s.vm.z, restOffset = s.vm.w;
    const float adhesion = s.adh.x, adhesionRadiusScale = s.adh.y, dt = s.adh.z;
    const bool isVelocityIteration = asuint(s.adh.w) == 1u;
    float appliedForceX = s.applied.x, appliedForceY = s.applied.y;

    const float projectionDt = isVelocityIteration ? 0.0f : dt;
    const float separation = (error - dot3(normal, delta) + normalDelta) + normalVel * projectionDt;
    float clamp = -appliedForceX;

    if (adhesionRadiusScale > 0.0f && adhesion > 0.0f)
    {
        const float adhesionRadius = restOffset * adhesionRadiusScale - restOffset;
        if (separation < adhesionRadiusScale)
        {
            clamp = pmin(clamp, -pmax(0.0f, adhesion * quart(1.0f - separation / adhesionRadius) * pmax(0.0f, separation) * velMultiplier));
        }
    }

    const float deltaF = pmax(clamp, -separation * velMultiplier);
    appliedForceX = pmax(0.0f, appliedForceX + deltaF);

    const float maxFriction = abs(appliedForceX) * frictionCoefficient;

    const float biasedF0 = -(dot3(fric0, delta) - tanDelta0);
    const float biasedF1 = -(dot3(fric1, delta) - tanDelta1);
    const float requiredF0Force = (biasedF0 + tanVel0 * projectionDt) * vmF0;
    const float requiredF1Force = (biasedF1 + tanVel1 * projectionDt) * vmF1;

    const float requiredForce = sqrt(requiredF0Force * requiredF0Force + requiredF1Force * requiredF1Force);
    const float recipRequiredForce = (requiredForce >= 1e-16f) ? (1.0f / requiredForce) : 0.0f;

    const float ratio0 = requiredF0Force * recipRequiredForce;
    const float ratio1 = requiredF1Force * recipRequiredForce;

    const float deltaFr = pmin(requiredForce + appliedForceY, maxFriction) - appliedForceY;
    appliedForceY = appliedForceY + deltaFr;

    const float deltaFr0 = deltaFr * ratio0;
    const float deltaFr1 = deltaFr * ratio1;

    PcOut o;
    o.f = float4(deltaF, appliedForceX, appliedForceY, 0.0f);
    o.fr = float4(deltaFr0, deltaFr1, 0.0f, 0.0f);
    gOut[tid] = o;
}
