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

// npArtiDriveParams.slang
//
// Slang port of the articulation implicit-drive scalar family from
// DyCpuGpuArticulation.h: computeImplicitDriveParamsForceDrive (L74-90),
// computeImplicitDriveParamsAccelerationDrive (L93-109), computeDriveImpulse
// (L184-195) and computeFrictionImpulse (L165-173).  The eFORCE/eACCELERATION
// dispatch of computeImplicitDriveParams (L127) is the isAccel select below.
//
// ArticulationImplicitDriveDesc = (targetVelPlusInitialBias, biasCoefficient,
// velMultiplier, impulseMultiplier, targetPosBias).  All pure scalar arithmetic
// (a few divides + branches on isTGSSolver / unitResponse>0 / friction abs).
// NoContraction-stamped; +-0 eqFZ.

struct DriveDesc { float tvBias, biasCoef, velMul, impMul, posBias; }

DriveDesc forceDrive(float stiffness, float damping, float dt, float simDt,
                     float unitResponse, float geomError, float targetVelocity, bool isTGS)
{
    const float a = dt * (dt * stiffness + damping);
    const float b = dt * (damping * targetVelocity);
    const float x = unitResponse > 0.0f ? 1.0f / (1.0f + a * unitResponse) : 0.0f;
    const float initialBias = geomError - (simDt - dt) * targetVelocity;
    const float biasCoefficient = stiffness * x * dt;
    DriveDesc d;
    d.tvBias = (x * b) + biasCoefficient * initialBias;
    d.velMul = -x * a;
    d.biasCoef = biasCoefficient;
    d.impMul = isTGS ? 1.0f : (1.0f - x);
    d.posBias = isTGS ? (biasCoefficient * targetVelocity) : 0.0f;
    return d;
}

DriveDesc accelDrive(float stiffness, float damping, float dt, float simDt,
                     float recipUnitResponse, float geomError, float targetVelocity, bool isTGS)
{
    const float a = dt * (dt * stiffness + damping);
    const float b = dt * (damping * targetVelocity);
    const float x = 1.0f / (1.0f + a);
    const float initialBias = geomError - (simDt - dt) * targetVelocity;
    const float biasCoefficient = stiffness * x * recipUnitResponse * dt;
    DriveDesc d;
    d.tvBias = (x * b * recipUnitResponse) + biasCoefficient * initialBias;
    d.velMul = -x * a * recipUnitResponse;
    d.biasCoef = biasCoefficient;
    d.impMul = isTGS ? 1.0f : (1.0f - x);
    d.posBias = isTGS ? (biasCoefficient * targetVelocity) : 0.0f;
    return d;
}

float driveImpulse(float accum, float jointVel, float jointDeltaPos, float elapsed, DriveDesc d)
{
    return accum * d.impMul
         + jointVel * d.velMul
         + d.tvBias
         - jointDeltaPos * d.biasCoef
         + elapsed * d.posBias;
}

float frictionImpulse(float fImp, float staticF, float dynamicF, float viscous, float jointVel)
{
    if (abs(fImp) > staticF)
    {
        const float lo = -dynamicF - viscous * abs(jointVel);
        const float hi = dynamicF + viscous * abs(jointVel);
        // PxClamp(v, lo, hi) = PxMin(hi, PxMax(lo, v)); PxMax(a,b)=(a>b)?a:b, PxMin(a,b)=(a<b)?a:b
        const float mx = (lo > fImp) ? lo : fImp;
        fImp = (hi < mx) ? hi : mx;
    }
    return fImp;
}

struct DpIn
{
    float4 p0;    // stiffness, damping, dt, simDt
    float4 p1;    // unitResponse, recipUnitResponse, geomError, targetVelocity
    float4 di;    // accum, jointVel, jointDeltaPos, elapsed
    float4 fr;    // frictionImpulse, staticF, dynamicF, viscous
    float4 misc;  // frictionJointVel, isTGS(1/0 bit), isAccel(1/0 bit), 0
}

struct DpOut
{
    float4 forceDesc0;  // force: tvBias, biasCoef, velMul, impMul
    float4 forceDesc1;  // force: posBias, _, _, _
    float4 accelDesc0;  // accel: tvBias, biasCoef, velMul, impMul
    float4 accelDesc1;  // accel: posBias, _, _, _
    float4 outs;        // driveImpulse(selected), frictionImpulse, _, _
}

struct DpParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<DpParams> gParams;

RWStructuredBuffer<DpIn> gIn;    // binding 0
RWStructuredBuffer<DpOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void driveParams(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const DpIn s = gIn[tid];

    const float stiffness = s.p0.x, damping = s.p0.y, dt = s.p0.z, simDt = s.p0.w;
    const float unitResp = s.p1.x, recipUnitResp = s.p1.y, geomError = s.p1.z, targetVel = s.p1.w;
    const bool isTGS = asuint(s.misc.y) == 1u;
    const bool isAccel = asuint(s.misc.z) == 1u;

    const DriveDesc fd = forceDrive(stiffness, damping, dt, simDt, unitResp, geomError, targetVel, isTGS);
    const DriveDesc ad = accelDrive(stiffness, damping, dt, simDt, recipUnitResp, geomError, targetVel, isTGS);
    const DriveDesc sel = isAccel ? ad : fd;

    const float drimp = driveImpulse(s.di.x, s.di.y, s.di.z, s.di.w, sel);
    const float frimp = frictionImpulse(s.fr.x, s.fr.y, s.fr.z, s.fr.w, s.misc.x);

    DpOut o;
    o.forceDesc0 = float4(fd.tvBias, fd.biasCoef, fd.velMul, fd.impMul);
    o.forceDesc1 = float4(fd.posBias, 0.0f, 0.0f, 0.0f);
    o.accelDesc0 = float4(ad.tvBias, ad.biasCoef, ad.velMul, ad.impMul);
    o.accelDesc1 = float4(ad.posBias, 0.0f, 0.0f, 0.0f);
    o.outs = float4(drimp, frimp, 0.0f, 0.0f);
    gOut[tid] = o;
}
