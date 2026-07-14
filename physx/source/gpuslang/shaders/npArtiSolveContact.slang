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

// npArtiSolveContact.slang
//
// Slang port of solveExtContactsBlock (solver.cuh L54-320): the PGS
// articulation ext-body CONTACT solve.  A normal-constraint loop followed by
// a friction loop, mutating the two bodies' spatial velocities (top=angular,
// bottom=linear) via the articulation impulse response and accumulating
// impulse0/impulse1 (which START at zero -- no impulse seed, unlike the 1D
// solve).
//
// The CUDA software-pipelines the loads (nextRaxn_extraCoeff etc.); that is a
// memory-latency schedule, not a computation -- the loaded values are
// identical to a direct per-iteration load, so the port loads serially.  The
// articulation responses are consumed sequentially: contact i uses
// artiResponse[i], friction j uses artiResponse[numNormal + j] (packed as
// respC[i] / respF[j] here).
//
// computeContactCoefficients is transcribed locally (verbatim from the
// battery-verified rigid solveContactBlock port in solverPGS.slang; not
// public there).  The arti call path re-pins it via the tranche-186 battery.
// pxMaxQ = (a>b)?a:b matches PxMax(float) = intrinsics::selectMax (used for
// the minNormalForce clamp).  All arith NoContraction-stamped; +-0 tolerated.

import pxgSlangCommon;
import pxgSlangPrepCommon;

static const uint MAX_NC = 6u;
static const uint MAX_NF = 6u;
static const float P8_Q = 0.8f;
static const float PX_EPS_REAL_Q = 1.192092896e-07f;
static const uint PXG_CONTACT_FLAG_COMPLIANT_ACCELERATION_SPRING = 1u << 3;

struct CRow
{
    float4 raXn_tv;      // raXn.xyz, targetVelocity
    float4 rbXn_maxI;    // rbXn.xyz, maxImpulse
    float4 crm;          // appliedForce, resp0, resp1, coeff0
    float4 crm2;         // coeff1, 0, 0, 0
    float4 dRA, dRA2, dRB;  // artiResponse (deltaRALin/Ang, deltaRBLin/Ang) packed
}

struct FRow
{
    float4 raXn_bias;    // raXn.xyz, bias
    float4 rbXn_tvW;     // rbXn.xyz, targetVel
    float4 frm;          // appliedForce, resp0, resp1, 0
    float4 dRA, dRA2, dRB;
}

struct AscIn
{
    uint4 cfg;           // numNormalConstr, numFrictionConstr, doFriction, flags
    float4 invMassAngDom;// invMass0, invMass1, angDom0, angDom1
    float4 normalSF;     // normal.xyz, staticCof
    float4 hdr;          // restitution, cfm, minNormalForce, dynamicFriction
    float4 fricN[2];     // frictionNormals[0], frictionNormals[1]
    float4 refs;         // ref0, ref1, 0, 0
    float4 vel0Top, vel0Bot, vel1Top, vel1Bot;
    CRow contacts[6];
    FRow frictions[6];
}

struct AscOut
{
    float4 vel0Top, vel0Bot, vel1Top, vel1Bot;
    float4 imp0, imp0b;  // impulse0.top, impulse0.bottom
    float4 imp1, imp1b;  // impulse1.top, impulse1.bottom
    float4 cApplied[2];  // per-contact appliedForce (8 lanes; 6 used)
    float4 fApplied[2];  // per-friction appliedForce
    uint4 broken;        // broken flag (x)
}

struct AscParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<AscParams> gParams;

RWStructuredBuffer<AscIn> gIn;    // binding 0
RWStructuredBuffer<AscOut> gOut;  // binding 1

float dot3C(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float pxMaxQ(float a, float b) { return (a > b) ? a : b; }  // PxMax float = intrinsics::selectMax

// mirrors computeContactCoefficients() in constraintPrepShared.cuh
// (verbatim from solverPGS.slang; local because not public there)
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

[shader("compute")]
[numthreads(64, 1, 1)]
void asCtPGS(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const AscIn s = gIn[tid];

    AscOut o;
    o.imp0 = float4(0, 0, 0, 0); o.imp0b = float4(0, 0, 0, 0);
    o.imp1 = float4(0, 0, 0, 0); o.imp1b = float4(0, 0, 0, 0);
    o.cApplied[0] = float4(0, 0, 0, 0); o.cApplied[1] = float4(0, 0, 0, 0);
    o.fApplied[0] = float4(0, 0, 0, 0); o.fApplied[1] = float4(0, 0, 0, 0);
    o.broken = uint4(0, 0, 0, 0);

    const uint numNormalConstr = min(s.cfg.x, MAX_NC);
    const uint numFrictionConstr = min(s.cfg.y, MAX_NF);
    const bool doFriction = s.cfg.z != 0u;
    const uint flags = s.cfg.w;

    const float ref0 = s.refs.x;
    const float ref1 = s.refs.y;

    const float invMass0 = s.invMassAngDom.x;
    const float invMass1 = s.invMassAngDom.y;
    const float angDom0 = s.invMassAngDom.z;
    const float angDom1 = s.invMassAngDom.w;

    const float3 normal = s.normalSF.xyz;
    const float staticCof = s.normalSF.w;
    const float restitution = s.hdr.x;
    const float cfm = s.hdr.y;
    const float p8 = P8_Q;

    // imp0/imp1 start at zero
    float3 imp0Top = float3(0, 0, 0), imp0Bot = float3(0, 0, 0);
    float3 imp1Top = float3(0, 0, 0), imp1Bot = float3(0, 0, 0);

    // v0/v1 spatial velocities (top=ang, bottom=lin)
    float3 v0Top = s.vel0Top.xyz, v0Bot = s.vel0Bot.xyz;
    float3 v1Top = s.vel1Top.xyz, v1Bot = s.vel1Bot.xyz;

    float accumulatedNormalImpulse = 0.0f;

    // ---- normal-constraint loop (cu L108-204) ----
    for (uint i = 0u; i < numNormalConstr; ++i)
    {
        const CRow c = s.contacts[i];

        const float3 raXn = c.raXn_tv.xyz;
        const float targetVelocity = c.raXn_tv.w;
        const float3 rbXn = c.rbXn_maxI.xyz;
        const float maxImpulse = c.rbXn_maxI.w;
        const float appliedForce = c.crm.x;
        const float resp0 = c.crm.y;
        const float resp1 = c.crm.z;
        const float coeff0 = c.crm.w;
        const float coeff1 = c.crm2.x;

        float unitResponse = ref0 * resp0 + ref1 * resp1;
        float recipResponse = (unitResponse > 0.0f) ? (1.0f / (unitResponse + cfm)) : 0.0f;
        float velMultiplier = recipResponse;
        float impulseMul = 1.0f;
        float unbiasedError, biasedErr;
        computeContactCoefficients(flags, restitution, unitResponse, recipResponse, targetVelocity,
                                   coeff0, coeff1, velMultiplier, impulseMul, unbiasedError,
                                   biasedErr);

        float3 dvATop, dvABot, dvBTop, dvBBot;
        // deltaVA = (ref0*deltaRAAng, ref0*deltaRALin)
        dvATop = ref0 * float3(c.dRA.w, c.dRA2.x, c.dRA2.y);
        dvABot = ref0 * c.dRA.xyz;
        // deltaVB = (ref1*deltaRBAng, ref1*deltaRBLin)
        dvBTop = ref1 * c.dRB.yzw;
        dvBBot = ref1 * float3(c.dRA2.z, c.dRA2.w, c.dRB.x);

        const float v0_ = dot3C(v0Bot, normal) + dot3C(v0Top, raXn);
        const float v1_ = dot3C(v1Bot, normal) + dot3C(v1Top, rbXn);
        const float normalVel = v0_ - v1_;

        const float tempDeltaF = biasedErr - normalVel * velMultiplier;
        const float _deltaF = max(tempDeltaF, -appliedForce);  // fmaxf
        const float _newForce = appliedForce * impulseMul + _deltaF;
        const float newForce = min(_newForce, maxImpulse);  // fminf
        const float deltaF = newForce - appliedForce;

        o.cApplied[i / 4u][i & 3u] = newForce;

        imp0Bot -= raXn * deltaF;
        imp0Top -= normal * deltaF;
        imp1Bot += rbXn * deltaF;
        imp1Top += normal * deltaF;

        v0Top += dvATop * deltaF;
        v0Bot += dvABot * deltaF;
        v1Top += dvBTop * deltaF;
        v1Bot += dvBBot * deltaF;

        accumulatedNormalImpulse = accumulatedNormalImpulse + newForce;
    }

    // minimum normal force for friction (cu L209) -- PxMax float = selectMax
    accumulatedNormalImpulse = pxMaxQ(accumulatedNormalImpulse, s.hdr.z);

    // ---- friction loop (cu L211-313) ----
    if (numFrictionConstr != 0u && doFriction)
    {
        const float dynamicFrictionCof = s.hdr.w;
        const float maxFrictionImpulse = staticCof * accumulatedNormalImpulse;
        const float maxDynFrictionImpulse = dynamicFrictionCof * accumulatedNormalImpulse;

        uint broken = 0u;

        for (uint i = 0u; i < numFrictionConstr; ++i)
        {
            const FRow f = s.frictions[i];

            const float4 frictionNormal = s.fricN[i & 1u];

            const float3 raXn = f.raXn_bias.xyz;
            const float bias = f.raXn_bias.w;
            const float3 rbXn = f.rbXn_tvW.xyz;
            const float targetVel = f.rbXn_tvW.w;
            const float appliedForce = f.frm.x;
            const float resp0 = f.frm.y;
            const float resp1 = f.frm.z;

            const float respF = ref0 * resp0 + ref1 * resp1;
            const float velMultiplier = (respF > PX_EPS_REAL_Q) ? (p8 / respF) : 0.0f;

            float3 dvATop, dvABot, dvBTop, dvBBot;
            dvATop = ref0 * float3(f.dRA.w, f.dRA2.x, f.dRA2.y);
            dvABot = ref0 * f.dRA.xyz;
            dvBTop = ref1 * f.dRB.yzw;
            dvBBot = ref1 * float3(f.dRA2.z, f.dRA2.w, f.dRB.x);

            const float3 fnormal = frictionNormal.xyz;

            const float v0_ = dot3C(v0Top, raXn) + dot3C(v0Bot, fnormal);
            const float v1_ = dot3C(v1Top, rbXn) + dot3C(v1Bot, fnormal);
            const float normalVel = v0_ - v1_;

            const float tmp1 = appliedForce - (bias - targetVel) * velMultiplier;
            const float totalImpulse = tmp1 - normalVel * velMultiplier;

            const bool clamp = abs(totalImpulse) > maxFrictionImpulse;  // fabsf
            const float totalClamped =
                min(maxDynFrictionImpulse, max(-maxDynFrictionImpulse, totalImpulse));
            const float newAppliedForce = clamp ? totalClamped : totalImpulse;
            const float deltaF = newAppliedForce - appliedForce;

            v0Top += dvATop * deltaF;
            v0Bot += dvABot * deltaF;
            v1Top += dvBTop * deltaF;
            v1Bot += dvBBot * deltaF;

            imp0Bot -= raXn * deltaF;
            imp0Top -= fnormal * deltaF;
            imp1Bot += rbXn * deltaF;
            imp1Top += fnormal * deltaF;

            o.fApplied[i / 4u][i & 3u] = newAppliedForce;
            broken = broken | (clamp ? 1u : 0u);
        }
        o.broken.x = broken;
    }

    // final impulse stores (cu L315-316): scale(l,a) = (top*l, bottom*a).
    // ASYMMETRIC lane pairing: body0 (angDom0, invMass0), body1 (invMass1, angDom1).
    o.imp0 = float4(imp0Top * (ref0 * angDom0), 0.0f);
    o.imp0b = float4(imp0Bot * (ref0 * invMass0), 0.0f);
    o.imp1 = float4(imp1Top * (ref1 * invMass1), 0.0f);
    o.imp1b = float4(imp1Bot * (ref1 * angDom1), 0.0f);

    o.vel0Top = float4(v0Top, 0.0f);
    o.vel0Bot = float4(v0Bot, 0.0f);
    o.vel1Top = float4(v1Top, 0.0f);
    o.vel1Bot = float4(v1Bot, 0.0f);

    gOut[tid] = o;
}
