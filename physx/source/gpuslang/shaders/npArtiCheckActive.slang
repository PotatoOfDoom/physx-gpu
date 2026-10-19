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

// npArtiCheckActive.slang
//
// Slang port of the two "light" articulation active-contact predicates:
//   ceacPGS <- checkExtActiveContactBlock    (solver.cuh L322-426)
//   ceacTGS <- checkExtActiveContactBlockTGS  (solver.cuh L737-831)
// Each mirrors the NORMAL-constraint force update of its full ext contact
// solve (asCtPGS/asCtTGS) but returns a BOOL — whether any contact is active
// (|deltaF| > 1.0e-8f) — instead of mutating velocity/impulse.
//
// Three structural facts, all faithfully preserved:
//   1. NO ref-scaling.  Neither predicate takes ref0/ref1; resp0/resp1 are
//      used UNSCALED (unitResponse = resp0 + resp1).
//   2. Velocity is NOT propagated between contacts (v0/v1 read-only), so each
//      contact's deltaF is an INDEPENDENT function of the shared initial
//      velocity.  The CUDA early-returns at the first active contact; because
//      the deltaF are independent, this port computes every contact's deltaF
//      (exposed as a diagnostic for strong bit-exact verification) and derives
//      the faithful early-return result as their disjunction — identical bool.
//   3. The CUDA arti response loads (deltaRA*/deltaRB*) are a dead memory-
//      schedule (never consumed, since velocity is not propagated) and are
//      omitted; they cannot affect the bool.
// No friction is evaluated in either predicate.  Active threshold PxAbs(deltaF)
// > 1.0e-8f.  computeContactCoefficients (PGS) / computeCompliantContact-
// CoefficientsTGS (TGS) transcribed locally, as in asCtPGS/asCtTGS.  All arith
// NoContraction-stamped; floats +-0 tolerated (eqFZ).

import pxgSlangCommon;
import pxgSlangPrepCommon;

static const uint MAX_NC = 6u;
static const float ACTIVE_EPS = 1.0e-8f;
static const uint PXG_CONTACT_FLAG_COMPLIANT_ACCELERATION_SPRING = 1u << 4; // 5.12 PxgSolverFlags.h (5.6.1: 1 << 3)

float dot3C(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float pxMaxQ(float a, float b) { return (a > b) ? a : b; }  // PxMax float = selectMax
float pxMinQ(float a, float b) { return (a < b) ? a : b; }  // PxMin float = selectMin

// mirrors computeContactCoefficients() (constraintPrepShared.cuh; PGS path),
// same local transcription as asCtPGS
void computeContactCoefficients(uint flags, float restitution, float unitResponse,
                                float recipResponse, float targetVelocity, float coeff0,
                                float coeff1, out float velMultiplier, out float impulseMultiplier,
                                out float unbiasedErr, out float biasedErr)
{
    if (restitution < 0.0f)
    {
        const bool accelSpring = (flags & PXG_CONTACT_FLAG_COMPLIANT_ACCELERATION_SPRING) != 0u;
        const float massIfAccelElseOne = accelSpring ? recipResponse : 1.0f;
        const float oneIfAccelElseR = accelSpring ? 1.0f : unitResponse;
        const float a = coeff0;
        const float b = coeff1 * massIfAccelElseOne;
        const float x = 1.0f / (a * oneIfAccelElseR + 1.0f);
        const float scaledBias = x * b;
        velMultiplier = x * a * massIfAccelElseOne;
        impulseMultiplier = 1.0f - x;
        unbiasedErr = biasedErr = targetVelocity * velMultiplier - scaledBias;
    }
    else
    {
        velMultiplier = recipResponse;
        biasedErr = coeff0 * velMultiplier;
        unbiasedErr = coeff1 * velMultiplier;
        impulseMultiplier = 1.0f;
    }
}

// runtime a-coefficient overload of computeCompliantContactCoefficientsTGS
// (constraintPrepShared.cuh L365-376), same local transcription as asCtTGS
void computeCompliantContactCoefficientsTGS(uint flags, float nrdt, float unitResponse,
                                            float recipResponse, float a,
                                            inout float velMultiplier, inout float scaledBias)
{
    const bool accelSpring = (flags & PXG_CONTACT_FLAG_COMPLIANT_ACCELERATION_SPRING) != 0u;
    const float massIfAccelElseOne = accelSpring ? recipResponse : 1.0f;
    const float oneIfAccelElseR = accelSpring ? 1.0f : unitResponse;
    const float x = 1.0f / (a * oneIfAccelElseR + 1.0f);
    velMultiplier = x * a * massIfAccelElseOne;
    scaledBias = nrdt * x * oneIfAccelElseR;
}

// ---- PGS ceacPGS ----

struct CPRow
{
    float4 raXn_tv;    // raXn.xyz, targetVelocity
    float4 rbXn_maxI;  // rbXn.xyz, maxImpulse
    float4 crm;        // appliedForce, resp0, resp1, coeff0
    float4 crm2;       // coeff1, 0, 0, 0
}

struct CeacPgsIn
{
    uint4 cfg;         // numNormalConstr, flags, 0, 0
    float4 hdr;        // restitution, cfm, 0, 0
    float4 normalSF;   // normal.xyz, 0
    float4 vel0Top, vel0Bot, vel1Top, vel1Bot;
    CPRow contacts[6];
}

struct CeacOut
{
    float4 dF[2];      // per-contact deltaF (8 lanes, 6 used) diagnostic
    uint4 active;      // .x = any contact active (0/1)
}

struct CeacParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<CeacParams> gParams;

RWStructuredBuffer<CeacPgsIn> gInP;    // binding 0
RWStructuredBuffer<CeacOut> gOutP;     // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void ceacPGS(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const CeacPgsIn s = gInP[tid];

    CeacOut o;
    o.dF[0] = float4(0, 0, 0, 0); o.dF[1] = float4(0, 0, 0, 0);
    o.active = uint4(0, 0, 0, 0);

    const uint numNormalConstr = min(s.cfg.x, MAX_NC);
    const uint flags = s.cfg.y;
    const float restitution = s.hdr.x;
    const float cfm = s.hdr.y;
    const float3 normal = s.normalSF.xyz;

    const float3 v0Top = s.vel0Top.xyz, v0Bot = s.vel0Bot.xyz;
    const float3 v1Top = s.vel1Top.xyz, v1Bot = s.vel1Bot.xyz;

    uint active = 0u;
    for (uint i = 0u; i < numNormalConstr; ++i)
    {
        const CPRow c = s.contacts[i];
        const float3 raXn = c.raXn_tv.xyz;
        const float targetVelocity = c.raXn_tv.w;
        const float3 rbXn = c.rbXn_maxI.xyz;
        const float maxImpulse = c.rbXn_maxI.w;
        const float appliedForce = c.crm.x;
        const float resp0 = c.crm.y;
        const float resp1 = c.crm.z;
        const float coeff0 = c.crm.w;
        const float coeff1 = c.crm2.x;

        const float unitResponse = resp0 + resp1;
        const float recipResponse = (unitResponse > 0.0f) ? (1.0f / (unitResponse + cfm)) : 0.0f;
        float velMultiplier, impulseMul, unbiasedError, biasedErr;
        computeContactCoefficients(flags, restitution, unitResponse, recipResponse, targetVelocity,
                                   coeff0, coeff1, velMultiplier, impulseMul, unbiasedError, biasedErr);

        const float v0_ = dot3C(v0Bot, normal) + dot3C(v0Top, raXn);
        const float v1_ = dot3C(v1Bot, normal) + dot3C(v1Top, rbXn);
        const float normalVel = v0_ - v1_;

        const float tempDeltaF = biasedErr - normalVel * velMultiplier;
        const float _deltaF = max(tempDeltaF, -appliedForce);   // fmaxf
        const float _newForce = appliedForce * impulseMul + _deltaF;
        const float newForce = min(_newForce, maxImpulse);      // fminf
        const float deltaF = newForce - appliedForce;

        o.dF[i / 4u][i & 3u] = deltaF;
        if (abs(deltaF) > ACTIVE_EPS)
            active = 1u;
    }

    o.active.x = active;
    gOutP[tid] = o;
}

// ---- TGS ceacTGS ----

struct CTRow
{
    float4 raXn_extra;  // raXn.xyz, compliantContactCoef
    float4 rbXn_tv;     // rbXn.xyz, targetVel
    float4 misc;        // appliedForce, separation, maxImpulse, biasCoefficient
    float4 resp;        // resp0, resp1, 0, 0
}

struct CeacTgsIn
{
    uint4 cfg;          // numNormalConstr, flags, 0, 0
    float4 hdr;         // restitutionXdt, cfm, maxPenBias, minPen
    float4 hdr2;        // elapsedTime, 0, 0, 0
    float4 normalSF;    // normal.xyz, 0
    float4 d0Top, d0Bot, d1Top, d1Bot;
    float4 vel0Top, vel0Bot, vel1Top, vel1Bot;
    CTRow contacts[6];
}

RWStructuredBuffer<CeacTgsIn> gInT;    // binding 2 (gInP/gOutP occupy 0/1)
RWStructuredBuffer<CeacOut> gOutT;     // binding 3

[shader("compute")]
[numthreads(64, 1, 1)]
void ceacTGS(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const CeacTgsIn s = gInT[tid];

    CeacOut o;
    o.dF[0] = float4(0, 0, 0, 0); o.dF[1] = float4(0, 0, 0, 0);
    o.active = uint4(0, 0, 0, 0);

    const uint numNormalConstr = min(s.cfg.x, MAX_NC);
    const uint flags = s.cfg.y;
    const float restitutionXdt = s.hdr.x;
    const float cfm = s.hdr.y;
    const float maxPenBias = s.hdr.z;
    const float minPen = s.hdr.w;
    const float elapsedTime = s.hdr2.x;
    const float3 normal = s.normalSF.xyz;

    const float3 v0Top = s.vel0Top.xyz, v0Bot = s.vel0Bot.xyz;
    const float3 v1Top = s.vel1Top.xyz, v1Bot = s.vel1Bot.xyz;
    const float3 d0Top = s.d0Top.xyz, d0Bot = s.d0Bot.xyz;
    const float3 d1Top = s.d1Top.xyz, d1Bot = s.d1Bot.xyz;

    const float3 relMotion = d0Bot - d1Bot;
    const float deltaV = dot3C(normal, relMotion);

    uint active = 0u;
    for (uint i = 0u; i < numNormalConstr; ++i)
    {
        const CTRow c = s.contacts[i];
        const float3 raXn = c.raXn_extra.xyz;
        const float compliantContactCoef = c.raXn_extra.w;
        const float3 rbXn = c.rbXn_tv.xyz;
        const float targetVel = c.rbXn_tv.w;
        const float appliedForce = c.misc.x;
        const float separation = c.misc.y;
        const float maxImpulse = c.misc.z;
        float biasCoefficient = c.misc.w;
        const float resp0 = c.resp.x;
        const float resp1 = c.resp.y;

        const float unitResponse = resp0 + resp1;
        const float recipResponse = (unitResponse > 0.0f) ? (1.0f / (unitResponse + cfm)) : 0.0f;
        float velMultiplier = recipResponse;
        if (restitutionXdt < 0.0f)
        {
            computeCompliantContactCoefficientsTGS(flags, restitutionXdt, unitResponse, recipResponse,
                                                   compliantContactCoef, velMultiplier, biasCoefficient);
        }

        const float v0 = dot3C(v0Top, raXn) + dot3C(v0Bot, normal);
        const float v1 = dot3C(v1Top, rbXn) + dot3C(v1Bot, normal);
        const float normalVel = v0 - v1;

        const float deltaBias =
            deltaV + dot3C(d0Top, raXn) - dot3C(d1Top, rbXn) - targetVel * elapsedTime;
        const float sep = pxMaxQ(minPen, separation + deltaBias);
        const float biased = pxMinQ(-maxPenBias, biasCoefficient * sep);
        const float tVelBias = recipResponse * biased;

        const float tempDeltaF = tVelBias - (normalVel - targetVel) * velMultiplier;
        const float _deltaF = max(tempDeltaF, -appliedForce);   // fmaxf
        const float _newForce = appliedForce + _deltaF;
        const float newForce = min(_newForce, maxImpulse);      // fminf
        const float deltaF = newForce - appliedForce;

        o.dF[i / 4u][i & 3u] = deltaF;
        if (abs(deltaF) > ACTIVE_EPS)
            active = 1u;
    }

    o.active.x = active;
    gOutT[tid] = o;
}
