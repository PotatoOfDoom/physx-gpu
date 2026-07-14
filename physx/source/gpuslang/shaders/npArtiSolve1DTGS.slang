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

// npArtiSolve1DTGS.slang
//
// Slang port of solveExt1DBlockTGS (solverBlockTGS.cuh L932-1089): the TGS
// articulation ext-body 1D-constraint solve.  The TGS analogue of the PGS
// as1dPGS (185), adding the position-delta bias machinery.  Consumes the
// header stores from ajhTGS (184) -- rAWorld_invMass0D0,
// rowCounts_breakable_orthoAxisCount -- and the reduced coefficients from
// a1dTGS (183), from which the runtime compute1dConstraintSolverConstantsTGS
// recovers biasScale/velMultiplier/initBias/targetVel each sub-step.
//
// Three TGS helpers transcribed locally (from the battery-verified rigid
// solverTGS.slang / DyCpuGpu1dConstraint.h; not public): computeMinBiasTGS,
// computeResolvedGeometricErrorTGS, compute1dConstraintSolverConstantsTGS
// (runtime from-coeffs variant).  computeRecipUnitResponse is public.
// Impulse store uses the same top/bottom cross as as1dPGS.  All arith
// NoContraction-stamped; +-0 tolerated.

import pxgSlangCommon;
import pxgSlangPrepCommon;

static const float DY_ARTI_MIN_RESPONSE_Q = 1e-12f;
static const float PX_MAX_F32_Q = 3.402823466e+38f;
static const uint MAX_ROWS_T = 20u;

float dot3T(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// PxQuat::rotate (PxQuat.h)
float3 quatRotT(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = q.x * vx + q.y * vy + q.z * vz;
    return float3(vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * dot2,
                  vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * dot2,
                  vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * dot2);
}

// mirrors Dy::computeMinBiasTGS()
float computeMinBiasTGS(uint flags, float maxBias)
{
    return -((flags & DY_SC_FLAG_INEQUALITY) != 0u ? PX_MAX_F32_Q : maxBias);
}

// mirrors Dy::computeResolvedGeometricErrorTGS()
float computeResolvedGeometricErrorTGS(float3 deltaLin0, float3 deltaLin1, float3 cLinVel0,
                                       float3 cLinVel1, float3 deltaAngInertia0,
                                       float3 deltaAngInertia1, float3 cAngVelInertia0,
                                       float3 cAngVelInertia1, float angularErrorScale,
                                       bool isSpringConstraint, float targetVel, float elapsedTime)
{
    const float deltaAng =
        (dot3T(cAngVelInertia0, deltaAngInertia0) - dot3T(cAngVelInertia1, deltaAngInertia1)) *
        angularErrorScale;
    const float deltaLin = dot3T(cLinVel0, deltaLin0) - dot3T(cLinVel1, deltaLin1);
    const float motion = deltaLin + deltaAng;
    return isSpringConstraint ? motion : motion - targetVel * elapsedTime;
}

// mirrors Dy::compute1dConstraintSolverConstantsTGS (runtime from-coeffs
// variant, DyCpuGpu1dConstraint.h L544-597); recovers biasScale/velMultiplier/
// error(initBias)/targetVel from the reduced coeff0..3.
void compute1dConstraintSolverConstantsTGS(bool isSpring, bool isAccelerationSpring,
                                           float geometricError, float unitResponse,
                                           float recipUnitResponse, float coeff0, float coeff1,
                                           float coeff2, float coeff3, out float biasScale,
                                           out float velMultiplier, out float error,
                                           out float targetVel)
{
    const float initJointSpeed = coeff0;
    biasScale = coeff1;

    if (isSpring)
    {
        const float a = coeff2;
        const float b = coeff3;

        if (isAccelerationSpring)
        {
            biasScale *= recipUnitResponse;
            error = biasScale * geometricError;

            const float x = 1.0f / (1.0f + a);
            velMultiplier = -x * a * recipUnitResponse;
            targetVel = x * b * recipUnitResponse - velMultiplier * initJointSpeed;
        }
        else
        {
            const float x = 1.0f / (1.0f + a * unitResponse);
            biasScale *= x;
            error = biasScale * geometricError;
            velMultiplier = -x * a;
            targetVel = x * b - velMultiplier * initJointSpeed;
        }
    }
    else
    {
        const float bounceVel = coeff2;
        const float velocityTarget = coeff3;

        if (bounceVel != 0.0f)
        {
            biasScale = 0.0f;
            error = 0.0f;
            velMultiplier = -1.0f;
            targetVel = bounceVel - velMultiplier * initJointSpeed;
        }
        else
        {
            error = geometricError * biasScale;
            velMultiplier = -1.0f;
            targetVel = velocityTarget - velMultiplier * initJointSpeed;
        }
    }
}

struct TRow
{
    float4 lin0_c0;   // clinVel0.xyz, coeff0
    float4 lin1_c1;   // clinVel1.xyz, coeff1
    float4 ang0_c2;   // cangVel0.xyz, coeff2
    float4 ang1_c3;   // cangVel1.xyz, coeff3
    float4 misc;      // geometricError, maxBias, angularErrorScale, appliedForce
    float4 misc2;     // resp0, resp1, maxImpulse, minImpulse
    uint4 flagsU;     // flags
    float4 dRA, dRA2, dRB;  // artiResponse (deltaR* packed)
}

struct At1In
{
    uint4 cfg;        // rowCount, residualEnabled, 0, 0
    float4 hdr;       // invInertiaScale0, invInertiaScale1, cfm, elapsedTime
    float4 refs;      // ref0, ref1, 0, 0
    float4 rAWorld;   // raPrev.xyz, invMass0D0
    float4 rBWorld;   // rbPrev.xyz, invMass1D1
    float4 deltaQ0;   // quat (x,y,z,w)
    float4 deltaQ1;
    float4 motion0Top, motion0Bot;  // spatial position delta (top=ang, bottom=lin)
    float4 motion1Top, motion1Bot;
    float4 vel0Top, vel0Bot, vel1Top, vel1Bot;
    TRow rows[20];
}

struct At1Out
{
    float4 vel0Top, vel0Bot, vel1Top, vel1Bot;
    float4 imp0, imp0b, imp1, imp1b;
    float4 appliedForce[5];
    float4 residual[5];
}

struct At1Params { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<At1Params> gParams;

RWStructuredBuffer<At1In> gIn;    // binding 0
RWStructuredBuffer<At1Out> gOut;  // binding 1

float calculateResidualT(float deltaF, float velocityMultiplier)
{
    return velocityMultiplier == 0.0f ? 0.0f : deltaF / velocityMultiplier;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void as1dTGS(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const At1In s = gIn[tid];

    At1Out o;
    o.imp0 = float4(0, 0, 0, 0); o.imp0b = float4(0, 0, 0, 0);
    o.imp1 = float4(0, 0, 0, 0); o.imp1b = float4(0, 0, 0, 0);
    for (uint z = 0u; z < 5u; ++z) { o.appliedForce[z] = float4(0, 0, 0, 0); o.residual[z] = float4(0, 0, 0, 0); }

    const uint rowCount = min(s.cfg.x, MAX_ROWS_T);
    const bool residualReportingEnabled = s.cfg.y != 0u;
    const float ref0 = s.refs.x, ref1 = s.refs.y;
    const float cfm = s.hdr.z;
    const float elapsedTime = s.hdr.w;

    const float invMass0 = ref0 * s.rAWorld.w;
    const float invMass1 = ref1 * s.rBWorld.w;
    const float invInertiaScale0 = ref0 * s.hdr.x;
    const float invInertiaScale1 = ref1 * s.hdr.y;

    const float3 raPrev = s.rAWorld.xyz;
    const float3 rbPrev = s.rBWorld.xyz;
    const float3 ra = quatRotT(s.deltaQ0, raPrev);
    const float3 rb = quatRotT(s.deltaQ1, rbPrev);
    const float3 raMotion = (ra + s.motion0Bot.xyz) - raPrev;
    const float3 rbMotion = (rb + s.motion1Bot.xyz) - rbPrev;

    float3 linVel0 = s.vel0Bot.xyz, linVel1 = s.vel1Bot.xyz;
    float3 angVel0 = s.vel0Top.xyz, angVel1 = s.vel1Top.xyz;

    float3 li0 = float3(0, 0, 0), li1 = float3(0, 0, 0);
    float3 ai0 = float3(0, 0, 0), ai1 = float3(0, 0, 0);

    for (uint i = 0u; i < rowCount; ++i)
    {
        const TRow r = s.rows[i];

        const float3 clinVel0 = r.lin0_c0.xyz;
        const float3 clinVel1 = r.lin1_c1.xyz;
        const float3 cangVel0 = r.ang0_c2.xyz;
        const float3 cangVel1 = r.ang1_c3.xyz;

        const uint flags = r.flagsU.x;
        const bool isSpringConstraint = (flags & DY_SC_FLAG_SPRING) != 0u;
        const bool isAccelerationSpring = (flags & DY_SC_FLAG_ACCELERATION_SPRING) != 0u;

        const float resp0 = ref0 * r.misc2.x;
        const float resp1 = ref1 * r.misc2.y;
        float unitResponse = resp0 + resp1;
        unitResponse += cfm;

        const float recipResponse = computeRecipUnitResponse(unitResponse, DY_ARTI_MIN_RESPONSE_Q);

        const float coeff0 = r.lin0_c0.w;
        const float coeff1 = r.lin1_c1.w;
        const float coeff2 = r.ang0_c2.w;
        const float coeff3 = r.ang1_c3.w;
        const float geometricError = r.misc.x;

        float biasScale, velMultiplier, initBias, targetVel;
        compute1dConstraintSolverConstantsTGS(isSpringConstraint, isAccelerationSpring,
                                              geometricError, unitResponse, recipResponse, coeff0,
                                              coeff1, coeff2, coeff3, biasScale, velMultiplier,
                                              initBias, targetVel);

        initBias = (biasScale == 0.0f) ? 0.0f : initBias;

        const float angularErrorScale = r.misc.z;
        const float errorDelta = computeResolvedGeometricErrorTGS(
            raMotion, rbMotion, clinVel0, clinVel1, s.motion0Top.xyz, s.motion1Top.xyz, cangVel0,
            cangVel1, angularErrorScale, isSpringConstraint, targetVel, elapsedTime);

        const float maxBias = r.misc.y;
        const float vMul = isSpringConstraint ? velMultiplier : recipResponse * velMultiplier;
        const float appliedForce = r.misc.w;

        const float unclampedBias = initBias + errorDelta * biasScale;
        const float minBias = computeMinBiasTGS(flags, maxBias);
        const float bias = clamp(unclampedBias, minBias, maxBias);

        const float constant =
            isSpringConstraint ? (bias + targetVel) : recipResponse * (bias + targetVel);

        const float maxImpulse = r.misc2.z;
        const float minImpulse = r.misc2.w;

        const float v0 = dot3T(linVel0, clinVel0) + dot3T(angVel0, cangVel0);
        const float v1 = dot3T(linVel1, clinVel1) + dot3T(angVel1, cangVel1);
        const float normalVel = v0 - v1;

        const float unclampedForce = appliedForce + (vMul * normalVel + constant);
        const float clampedForce = min(maxImpulse, max(minImpulse, unclampedForce));  // fminf/fmaxf
        const float deltaF = clampedForce - appliedForce;

        o.appliedForce[i / 4u][i & 3u] = clampedForce;
        if (residualReportingEnabled)
            o.residual[i / 4u][i & 3u] = calculateResidualT(deltaF, vMul);

        li0 = clinVel0 * deltaF + li0;
        ai0 = cangVel0 * deltaF + ai0;
        li1 = clinVel1 * deltaF + li1;
        ai1 = cangVel1 * deltaF + ai1;

        const float3 linVa = ref0 * r.dRA.xyz;
        const float3 angVa = ref0 * float3(r.dRA.w, r.dRA2.x, r.dRA2.y);
        const float3 linVb = ref1 * float3(r.dRA2.z, r.dRA2.w, r.dRB.x);
        const float3 angVb = ref1 * r.dRB.yzw;

        linVel0 = linVa * deltaF + linVel0;
        angVel0 = angVa * deltaF + angVel0;
        linVel1 = linVb * deltaF + linVel1;
        angVel1 = angVb * deltaF + angVel1;
    }

    o.vel0Top = float4(angVel0, 0.0f);
    o.vel0Bot = float4(linVel0, 0.0f);
    o.vel1Top = float4(angVel1, 0.0f);
    o.vel1Bot = float4(linVel1, 0.0f);

    o.imp0 = float4(li0 * invMass0, 0.0f);
    o.imp0b = float4(ai0 * invInertiaScale0, 0.0f);
    o.imp1 = float4(li1 * invMass1, 0.0f);
    o.imp1b = float4(ai1 * invInertiaScale1, 0.0f);

    gOut[tid] = o;
}
