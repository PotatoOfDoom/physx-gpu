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

// npArtiSolve1D.slang
//
// Slang port of solveExt1DBlock (solverBlock.cuh L208-326): the PGS
// articulation ext-body 1D-constraint solve core.  Per batch, iterates the
// constraint rows, applies the clamped impulse update to the two bodies'
// spatial velocities (top = angular, bottom = linear) via the articulation
// impulse response (deltaR*, exactly what a1dPGS packArti wrote in prep),
// and accumulates the applied spatial impulses.
//
// Reuses the solve-side helpers already ported for the rigid solve1DBlock:
// compute1dConstraintSolverConstantsPGS, computeRecipUnitResponse,
// calculateResidual, DY_MIN_RESPONSE, DY_SC_FLAG_SPRING/ACCELERATION_SPRING.
// All arith NoContraction-stamped; +-0 tolerated.
//
// Impulse store convention (cu L324-325, ported verbatim -- note the
// top/bottom cross): impulse.top = (linear accum) * invMass,
// impulse.bottom = (angular accum) * invInertiaScale.

import pxgSlangCommon;
import pxgSlangPrepCommon;

static const float DY_MIN_RESPONSE_Q = 1e-12f;
static const uint MAX_ROWS_S = 20u;

// one constraint row in the solve's PxgBlockSolverConstraint1DCon/Mod image
struct SRow
{
    float4 lin0XYZ_minImpulse;   // clinVel0, minImpulse
    float4 lin1XYZ_maxImpulse;   // clinVel1, maxImpulse
    float4 ang0XYZ_resp0;        // cangVel0, resp0
    float4 ang1XYZ_resp1;        // cangVel1, resp1
    float4 misc;                 // initJointSpeed, coeff0, coeff1, appliedForce
    uint4 flagsU;                // flags, 0, 0, 0
    float4 dRA;                  // deltaRALin.xyz, deltaRAAng.x  (artiResponse)
    float4 dRA2;                 // deltaRAAng.yz, deltaRBLin.xy
    float4 dRB;                  // deltaRBLin.z, deltaRBAng.xyz
}

struct As1In
{
    uint4 cfg;                   // rowCounts, residualEnabled, 0, 0
    float4 hdr;                  // invMass0D0, invMass1D1, invInertiaScale0, invInertiaScale1
    float4 hdr2;                 // cfm, ref0, ref1, 0
    float4 vel0Top;              // angVel0
    float4 vel0Bot;              // linVel0
    float4 vel1Top;              // angVel1
    float4 vel1Bot;              // linVel1
    float4 imp0Top;              // ai0 seed (impulse0.top pre)
    float4 imp0Bot;              // li0 seed... see note
    float4 imp1Top;
    float4 imp1Bot;
    SRow rows[20];
}

struct As1Out
{
    float4 vel0Top;              // angVel0 final
    float4 vel0Bot;              // linVel0 final
    float4 vel1Top;
    float4 vel1Bot;
    float4 imp0;                 // impulse0.top.xyz, top-not-stored-w=0 ; packed below
    float4 imp0b;                // impulse0.bottom.xyz, 0
    float4 imp1;
    float4 imp1b;
    float4 appliedForce[5];      // per-row appliedForce (20 lanes)
    float4 residual[5];          // per-row residual (20 lanes)
}

struct As1Params { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<As1Params> gParams;

RWStructuredBuffer<As1In> gIn;    // binding 0
RWStructuredBuffer<As1Out> gOut;  // binding 1

float dot3S(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// mirrors Dy::compute1dConstraintSolverConstantsPGS (reduced-coefficient
// variant); coeff2 carries initJointSpeed.  Transcribed verbatim from the
// rigid solve1DBlock port (solverPGS.slang) -- not public there, so kept
// local; the arti call path re-pins it via the tranche-185 battery.
void compute1dConstraintSolverConstantsPGS(bool isSpring, bool isAccelerationSpring, float coeff0,
                                           float coeff1, float coeff2, float unitResponse,
                                           float recipUnitResponse, out float constant,
                                           out float unbiasedConstant, out float velMultiplier,
                                           out float impulseMultiplier)
{
    if (isSpring)
    {
        const float a = coeff0;
        const float b = coeff1;

        if (isAccelerationSpring)
        {
            const float x = 1.0f / (1.0f + a);
            constant = x * recipUnitResponse * b;
            unbiasedConstant = constant;
            velMultiplier = -x * recipUnitResponse * a;
            impulseMultiplier = 1.0f - x;
        }
        else
        {
            const float x = 1.0f / (1.0f + a * unitResponse);
            constant = x * b;
            unbiasedConstant = constant;
            velMultiplier = -x * a;
            impulseMultiplier = 1.0f - x;
        }
    }
    else
    {
        velMultiplier = -recipUnitResponse;
        impulseMultiplier = 1.0f;
        constant = coeff0 * recipUnitResponse;
        unbiasedConstant = coeff1 * recipUnitResponse;
    }

    const float velBias = coeff2 * velMultiplier;
    constant += velBias;
    unbiasedConstant += velBias;
}

// mirrors PxgErrorAccumulator::calculateResidual()
float calculateResidual(float deltaF, float velocityMultiplier)
{
    return velocityMultiplier == 0.0f ? 0.0f : deltaF / velocityMultiplier;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void as1dPGS(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const As1In s = gIn[tid];

    As1Out o;
    o.imp0 = float4(0, 0, 0, 0); o.imp0b = float4(0, 0, 0, 0);
    o.imp1 = float4(0, 0, 0, 0); o.imp1b = float4(0, 0, 0, 0);
    for (uint z = 0u; z < 5u; ++z) { o.appliedForce[z] = float4(0, 0, 0, 0); o.residual[z] = float4(0, 0, 0, 0); }

    const uint numRows = min(s.cfg.x, MAX_ROWS_S);
    const bool residualReportingEnabled = s.cfg.y != 0u;
    const float ref0 = s.hdr2.y;
    const float ref1 = s.hdr2.z;

    float3 linVel0 = s.vel0Bot.xyz;
    float3 linVel1 = s.vel1Bot.xyz;
    float3 angVel0 = s.vel0Top.xyz;
    float3 angVel1 = s.vel1Top.xyz;

    // impulse seeds: impulse.top = li (linear accum), impulse.bottom = ai (ang)
    float3 li0 = s.imp0Bot.xyz;
    float3 li1 = s.imp1Bot.xyz;
    float3 ai0 = s.imp0Top.xyz;
    float3 ai1 = s.imp1Top.xyz;

    const float invMass0 = ref0 * s.hdr.x;
    const float invMass1 = ref1 * s.hdr.y;
    const float invInertiaScale0 = ref0 * s.hdr.z;
    const float invInertiaScale1 = ref1 * s.hdr.w;

    const float cfm = s.hdr2.x;

    for (uint i = 0u; i < numRows; ++i)
    {
        const SRow r = s.rows[i];

        const float3 clinVel0 = r.lin0XYZ_minImpulse.xyz;
        const float3 clinVel1 = r.lin1XYZ_maxImpulse.xyz;
        const float3 cangVel0 = r.ang0XYZ_resp0.xyz;
        const float3 cangVel1 = r.ang1XYZ_resp1.xyz;

        const float resp0 = ref0 * r.ang0XYZ_resp0.w;
        const float resp1 = ref1 * r.ang1XYZ_resp1.w;
        const float initJointSpeed = r.misc.x;
        const float coeff0 = r.misc.y;
        const float coeff1 = r.misc.z;
        const uint flags = r.flagsU.x;

        const float unitResponse = resp0 + resp1 + cfm;
        const float recipResponse = computeRecipUnitResponse(unitResponse, DY_MIN_RESPONSE_Q);

        float constant, unbiasedConstant, vMul, iMul;
        const bool isSpring = (flags & DY_SC_FLAG_SPRING) != 0u;
        const bool isAccelerationSpring = (flags & DY_SC_FLAG_ACCELERATION_SPRING) != 0u;
        compute1dConstraintSolverConstantsPGS(isSpring, isAccelerationSpring, coeff0, coeff1,
                                              initJointSpeed, unitResponse, recipResponse, constant,
                                              unbiasedConstant, vMul, iMul);

        const float appliedForce = r.misc.w;
        const float maxImpulse = r.lin1XYZ_maxImpulse.w;
        const float minImpulse = r.lin0XYZ_minImpulse.w;

        const float v0 = dot3S(linVel0, clinVel0) + dot3S(angVel0, cangVel0);
        const float v1 = dot3S(linVel1, clinVel1) + dot3S(angVel1, cangVel1);
        const float normalVel = v0 - v1;

        const float unclampedForce = iMul * appliedForce + (vMul * normalVel + constant);
        const float clampedForce = min(maxImpulse, max(minImpulse, unclampedForce));  // fminf/fmaxf
        const float deltaF = clampedForce - appliedForce;

        o.appliedForce[i / 4u][i & 3u] = clampedForce;
        if (residualReportingEnabled)
            o.residual[i / 4u][i & 3u] = calculateResidual(deltaF, vMul);

        li0 = clinVel0 * deltaF + li0;
        ai0 = cangVel0 * deltaF + ai0;
        li1 = clinVel1 * deltaF + li1;
        ai1 = cangVel1 * deltaF + ai1;

        // artiResponse deltaV (deltaR* unpacked from the packArti layout)
        const float3 dRALin = r.dRA.xyz;
        const float3 dRAAng = float3(r.dRA.w, r.dRA2.x, r.dRA2.y);
        const float3 dRBLin = float3(r.dRA2.z, r.dRA2.w, r.dRB.x);
        const float3 dRBAng = r.dRB.yzw;

        const float3 linVa = ref0 * dRALin;
        const float3 angVa = ref0 * dRAAng;
        const float3 linVb = ref1 * dRBLin;
        const float3 angVb = ref1 * dRBAng;

        linVel0 = linVa * deltaF + linVel0;
        angVel0 = angVa * deltaF + angVel0;
        linVel1 = linVb * deltaF + linVel1;
        angVel1 = angVb * deltaF + angVel1;
    }

    o.vel0Top = float4(angVel0, 0.0f);
    o.vel0Bot = float4(linVel0, 0.0f);
    o.vel1Top = float4(angVel1, 0.0f);
    o.vel1Bot = float4(linVel1, 0.0f);

    // impulse.top = li * invMass; impulse.bottom = ai * invInertiaScale
    o.imp0 = float4(li0 * invMass0, 0.0f);
    o.imp0b = float4(ai0 * invInertiaScale0, 0.0f);
    o.imp1 = float4(li1 * invMass1, 0.0f);
    o.imp1b = float4(ai1 * invInertiaScale1, 0.0f);

    gOut[tid] = o;
}
