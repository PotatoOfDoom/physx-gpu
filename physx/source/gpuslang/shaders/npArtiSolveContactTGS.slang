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

// npArtiSolveContactTGS.slang
//
// Slang port of solveExtContactBlockTGS (solver.cuh L430-735): the TGS
// articulation ext-body CONTACT solve.  The TGS analogue of asCtPGS (tranche
// 186) with position-delta bias machinery.  A normal-constraint loop, a
// PAIRED-friction loop (two rows per iteration with a combined-magnitude
// clamp), and an optional torsional-friction tail, mutating the two bodies'
// spatial velocities (top=angular, bottom=linear) via the articulation impulse
// response and accumulating impulse0/impulse1 (which START at zero).
//
// TGS differs from PGS in three load-bearing ways:
//   1. The final impulse scale is SYMMETRIC -- BOTH bodies scale (top*invMass,
//      bottom*angDom).  cu L732-733:
//        impulse0 = imp0.scale(invMass0*ref0, angDom0*ref0)
//        impulse1 = imp1.scale(invMass1*ref1, angDom1*ref1)
//      The PGS asCtPGS was ASYMMETRIC (body0 top*angDom0/bottom*invMass0).
//   2. The normal force update carries NO impulseMultiplier: _newForce =
//      appliedForce + _deltaF (PGS used appliedForce*impulseMul + _deltaF).
//   3. The friction velMultiplier divides by (resp + cfm) -- PGS divided by
//      resp alone.
// The position-delta bias adds deltaBias/sep/biased/tVelBias to the normal
// loop and an error-delta term to the friction rows, driven by delta0/delta1
// (accumulated position deltas, top=angular bottom=linear) and elapsedTime.
//
// computeCompliantContactCoefficientsTGS is transcribed locally (the runtime
// `a`-coefficient overload, constraintPrepShared.cuh L365-376; not public
// here).  The arti call path re-pins it via the tranche-188 battery.
// pxMaxQ/pxMinQ = selectMax/selectMin match PxMax(float)/PxMin(float).  All
// arith NoContraction-stamped; +-0 tolerated.

import pxgSlangCommon;
import pxgSlangPrepCommon;

static const uint MAX_NC = 6u;
static const uint MAX_NF = 6u;
static const float PX_EPS_REAL_Q = 1.192092896e-07f;
static const uint PXG_CONTACT_FLAG_COMPLIANT_ACCELERATION_SPRING = 1u << 4; // 5.12 PxgSolverFlags.h (5.6.1: 1 << 3)

struct CRow
{
    float4 raXn_extra;   // raXn.xyz, compliantContactCoef
    float4 rbXn_tv;      // rbXn.xyz, targetVel
    float4 misc;         // appliedForce, separation, maxImpulse, biasCoefficient
    float4 resp;         // resp0, resp1, 0, 0
    float4 dRA, dRA2, dRB;  // artiResponse packed (deltaRALin/Ang, deltaRBLin/Ang)
}

struct FRow
{
    float4 raXn_err;     // raXn.xyz, initialError
    float4 rbXn_tv;      // rbXn.xyz, targetVel
    float4 frm;          // appliedForce, resp0, resp1, 0
    float4 dRA, dRA2, dRB;
}

struct AtcIn
{
    uint4 cfg;           // numNormalConstr, totalFrictionConstr, 0, flags
    float4 invMassAngDom;// invMass0, invMass1, angDom0, angDom1
    float4 normalSF;     // normal.xyz, staticFriction
    float4 hdr;          // restitutionXdt, p8, cfm, maxPenBias
    float4 hdr2;         // minNormalForce, minPen, elapsedTime, dynamicFriction
    float4 fricHdr;      // friction biasCoefficient, torsionalFrictionScale, 0, 0
    float4 fricN[2];     // frictionNormals[0], frictionNormals[1]
    float4 refs;         // ref0, ref1, 0, 0
    float4 d0Top, d0Bot, d1Top, d1Bot;   // delta0/delta1 (top=ang, bottom=lin)
    float4 vel0Top, vel0Bot, vel1Top, vel1Bot;
    CRow contacts[6];
    FRow frictions[6];
}

struct AtcOut
{
    float4 vel0Top, vel0Bot, vel1Top, vel1Bot;
    float4 imp0, imp0b;  // impulse0.top, impulse0.bottom
    float4 imp1, imp1b;  // impulse1.top, impulse1.bottom
    float4 cApplied[2];  // per-contact appliedForce (8 lanes; 6 used)
    float4 fApplied[2];  // per-friction appliedForce
    uint4 broken;        // broken flag (x)
}

struct AtcParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<AtcParams> gParams;

RWStructuredBuffer<AtcIn> gIn;    // binding 0
RWStructuredBuffer<AtcOut> gOut;  // binding 1

float dot3C(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float pxMaxQ(float a, float b) { return (a > b) ? a : b; }  // PxMax float = selectMax
float pxMinQ(float a, float b) { return (a < b) ? a : b; }  // PxMin float = selectMin

// mirrors the runtime `a`-coefficient overload of
// computeCompliantContactCoefficientsTGS (constraintPrepShared.cuh L365-376);
// overwrites velMultiplier and scaledBias (the biasCoefficient) in place.
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

// P9.86: the impulse accumulators start at a literal +0 and only ever += / -= a term;
// an IEEE chain from +0 never holds -0 (exact cancellation rounds to +0).  llvmpipe
// unrolls the <= 6-trip loops, forwards the literal seed and rewrites the first
// `0 - t` to -t and `0 + t` to t, even under SignedZeroInfNanPreserve (measured).
// The fold only ever changes the sign of a zero, so the finished accumulator is
// canonicalised (-0 -> +0) before the final scale: bit for bit the engine's value.
float3 atcPlusZero3(float3 v)
{
    return float3((v.x == 0.0f) ? 0.0f : v.x, (v.y == 0.0f) ? 0.0f : v.y, (v.z == 0.0f) ? 0.0f : v.z);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void asCtTGS(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const AtcIn s = gIn[tid];

    AtcOut o;
    o.imp0 = float4(0, 0, 0, 0); o.imp0b = float4(0, 0, 0, 0);
    o.imp1 = float4(0, 0, 0, 0); o.imp1b = float4(0, 0, 0, 0);
    o.cApplied[0] = float4(0, 0, 0, 0); o.cApplied[1] = float4(0, 0, 0, 0);
    o.fApplied[0] = float4(0, 0, 0, 0); o.fApplied[1] = float4(0, 0, 0, 0);
    o.broken = uint4(0, 0, 0, 0);

    const uint numNormalConstr = min(s.cfg.x, MAX_NC);
    const uint totalFrictionConstr = min(s.cfg.y, MAX_NF);
    const uint numFrictionConstr = totalFrictionConstr & (~1u);
    const uint flags = s.cfg.w;

    const float ref0 = s.refs.x;
    const float ref1 = s.refs.y;

    const float invMass0 = s.invMassAngDom.x;
    const float invMass1 = s.invMassAngDom.y;
    const float angDom0 = s.invMassAngDom.z;
    const float angDom1 = s.invMassAngDom.w;

    const float3 normal = s.normalSF.xyz;
    const float staticFrictionCof = s.normalSF.w;

    const float restitutionXdt = s.hdr.x;
    const float p8 = s.hdr.y;
    const float cfm = s.hdr.z;
    const float maxPenBias = s.hdr.w;

    const float minNormalForce = s.hdr2.x;
    const float minPen = s.hdr2.y;
    const float elapsedTime = s.hdr2.z;
    const float dynamicFrictionCof = s.hdr2.w;

    // imp0/imp1 start at zero
    float3 imp0Top = float3(0, 0, 0), imp0Bot = float3(0, 0, 0);
    float3 imp1Top = float3(0, 0, 0), imp1Bot = float3(0, 0, 0);

    // v0/v1 spatial velocities (top=ang, bottom=lin)
    float3 v0Top = s.vel0Top.xyz, v0Bot = s.vel0Bot.xyz;
    float3 v1Top = s.vel1Top.xyz, v1Bot = s.vel1Bot.xyz;

    // delta0/delta1 accumulated position deltas (top=ang, bottom=lin)
    const float3 d0Top = s.d0Top.xyz, d0Bot = s.d0Bot.xyz;
    const float3 d1Top = s.d1Top.xyz, d1Bot = s.d1Bot.xyz;

    const float3 relMotion = d0Bot - d1Bot;
    const float deltaV = dot3C(normal, relMotion);

    float accumulatedNormalImpulse = 0.0f;

    // ---- normal-constraint loop (cu L477-556) ----
    for (uint i = 0u; i < numNormalConstr; ++i)
    {
        const CRow c = s.contacts[i];

        const float3 raXn = c.raXn_extra.xyz;
        const float compliantContactCoef = c.raXn_extra.w;
        const float3 rbXn = c.rbXn_tv.xyz;
        const float targetVel = c.rbXn_tv.w;
        const float appliedForce = c.misc.x;
        const float separation = c.misc.y;
        const float maxImpulse = c.misc.z;
        float biasCoefficient = c.misc.w;
        const float resp0 = ref0 * c.resp.x;
        const float resp1 = ref1 * c.resp.y;

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
        const float _deltaF = max(tempDeltaF, -appliedForce);  // fmaxf
        const float _newForce = appliedForce + _deltaF;
        const float newForce = min(_newForce, maxImpulse);     // fminf
        const float deltaF = newForce - appliedForce;

        o.cApplied[i / 4u][i & 3u] = newForce;

        // arti response (ref-scaled), DIRECT velocity update
        const float3 deltaRALin = ref0 * c.dRA.xyz;
        const float3 deltaRAAng = ref0 * float3(c.dRA.w, c.dRA2.x, c.dRA2.y);
        const float3 deltaRBLin = ref1 * float3(c.dRA2.z, c.dRA2.w, c.dRB.x);
        const float3 deltaRBAng = ref1 * c.dRB.yzw;
        v0Bot += deltaRALin * deltaF;
        v1Bot += deltaRBLin * deltaF;
        v0Top += deltaRAAng * deltaF;
        v1Top += deltaRBAng * deltaF;

        imp0Top -= normal * deltaF;
        imp0Bot -= raXn * deltaF;
        imp1Top += normal * deltaF;
        imp1Bot += rbXn * deltaF;

        accumulatedNormalImpulse = accumulatedNormalImpulse + newForce;
    }

    accumulatedNormalImpulse = pxMaxQ(accumulatedNormalImpulse, minNormalForce);  // selectMax

    // ---- friction (cu L558-725), gated on numFrictionConstr ----
    if (numFrictionConstr != 0u)
    {
        const float biasCoefficient = s.fricHdr.x;
        const float maxFrictionImpulse = staticFrictionCof * accumulatedNormalImpulse;
        const float maxDynFrictionImpulse = dynamicFrictionCof * accumulatedNormalImpulse;

        uint broken = 0u;

        const float3 normal0 = s.fricN[0].xyz;
        const float3 normal1 = s.fricN[1].xyz;
        const float deltaMotion0 = dot3C(normal0, relMotion);
        const float deltaMotion1 = dot3C(normal1, relMotion);

        // paired-friction loop (cu L577-670), two rows per iteration
        for (uint i = 0u; i < numFrictionConstr; i += 2u)
        {
            const FRow f0 = s.frictions[i];
            const FRow f1 = s.frictions[i + 1u];

            const float3 raXn0 = f0.raXn_err.xyz;
            const float initialError0 = f0.raXn_err.w;
            const float3 rbXn0 = f0.rbXn_tv.xyz;
            const float targetVel0 = f0.rbXn_tv.w;
            const float appliedForce0 = f0.frm.x;

            const float3 raXn1 = f1.raXn_err.xyz;
            const float initialError1 = f1.raXn_err.w;
            const float3 rbXn1 = f1.rbXn_tv.xyz;
            const float targetVel1 = f1.rbXn_tv.w;
            const float appliedForce1 = f1.frm.x;

            const float resp0 = ref0 * f0.frm.y + ref1 * f0.frm.z;
            const float velMultiplier0 = (resp0 > PX_EPS_REAL_Q) ? (p8 / (resp0 + cfm)) : 0.0f;
            const float resp1 = ref0 * f1.frm.y + ref1 * f1.frm.z;
            const float velMultiplier1 = (resp1 > PX_EPS_REAL_Q) ? (p8 / (resp1 + cfm)) : 0.0f;

            const float v00 = dot3C(v0Top, raXn0) + dot3C(v0Bot, normal0);
            const float v10 = dot3C(v1Top, rbXn0) + dot3C(v1Bot, normal0);
            const float normalVel0 = v00 - v10;

            const float v01 = dot3C(v0Top, raXn1) + dot3C(v0Bot, normal1);
            const float v11 = dot3C(v1Top, rbXn1) + dot3C(v1Bot, normal1);
            const float normalVel1 = v01 - v11;

            const float error0 = initialError0 - targetVel0 * elapsedTime +
                                 (dot3C(raXn0, d0Top) - dot3C(rbXn0, d1Top) + deltaMotion0);
            const float bias0 = error0 * biasCoefficient;
            const float tmp10 = appliedForce0 - (bias0 - targetVel0) * velMultiplier0;
            const float totalImpulse0 = tmp10 - normalVel0 * velMultiplier0;

            const float error1 = initialError1 - targetVel1 * elapsedTime +
                                 (dot3C(raXn1, d0Top) - dot3C(rbXn1, d1Top) + deltaMotion1);
            const float bias1 = error1 * biasCoefficient;
            const float tmp11 = appliedForce1 - (bias1 - targetVel1) * velMultiplier1;
            const float totalImpulse1 = tmp11 - normalVel1 * velMultiplier1;

            const float totalImpulse =
                sqrt(totalImpulse0 * totalImpulse0 + totalImpulse1 * totalImpulse1);

            const bool clamp = totalImpulse > maxFrictionImpulse;
            const float ratio = clamp ? (min(maxDynFrictionImpulse, totalImpulse) / totalImpulse) : 1.0f;

            const float newAppliedForce0 = totalImpulse0 * ratio;
            const float newAppliedForce1 = totalImpulse1 * ratio;
            const float deltaF0 = newAppliedForce0 - appliedForce0;
            const float deltaF1 = newAppliedForce1 - appliedForce1;

            // row-0 response (deltaF0)
            v0Bot += ref0 * f0.dRA.xyz * deltaF0;
            v1Bot += ref1 * float3(f0.dRA2.z, f0.dRA2.w, f0.dRB.x) * deltaF0;
            v0Top += ref0 * float3(f0.dRA.w, f0.dRA2.x, f0.dRA2.y) * deltaF0;
            v1Top += ref1 * f0.dRB.yzw * deltaF0;

            // row-1 response (deltaF1)
            v0Bot += ref0 * f1.dRA.xyz * deltaF1;
            v1Bot += ref1 * float3(f1.dRA2.z, f1.dRA2.w, f1.dRB.x) * deltaF1;
            v0Top += ref0 * float3(f1.dRA.w, f1.dRA2.x, f1.dRA2.y) * deltaF1;
            v1Top += ref1 * f1.dRB.yzw * deltaF1;

            o.fApplied[i / 4u][i & 3u] = newAppliedForce0;
            o.fApplied[(i + 1u) / 4u][(i + 1u) & 3u] = newAppliedForce1;
            broken = broken | (clamp ? 1u : 0u);

            imp0Top -= normal0 * deltaF0;
            imp0Bot -= raXn0 * deltaF0;
            imp1Top += normal0 * deltaF0;
            imp1Bot += rbXn0 * deltaF0;

            imp0Top -= normal1 * deltaF1;
            imp0Bot -= raXn1 * deltaF1;
            imp1Top += normal1 * deltaF1;
            imp1Bot += rbXn1 * deltaF1;
        }

        // torsional-friction tail (cu L672-722), if an odd row remains
        if (numFrictionConstr < totalFrictionConstr)
        {
            const float frictionScale = s.fricHdr.y;

            const FRow f0 = s.frictions[numFrictionConstr];
            const float3 raXn0 = f0.raXn_err.xyz;
            const float3 rbXn0 = f0.rbXn_tv.xyz;
            const float appliedForce0 = f0.frm.x;
            const float targetVel0 = f0.rbXn_tv.w;

            const float resp0 = ref0 * f0.frm.y + ref1 * f0.frm.z;
            const float velMultiplier0 = (resp0 > 0.0f) ? (p8 / (resp0 + cfm)) : 0.0f;

            const float v00 = dot3C(v0Top, raXn0);   // angular only
            const float v10 = dot3C(v1Top, rbXn0);
            const float normalVel0 = v00 - v10;

            const float tmp10 = appliedForce0 - (-targetVel0) * velMultiplier0;
            const float totalImpulse = tmp10 - normalVel0 * velMultiplier0;

            const bool clamp = abs(totalImpulse) > (maxFrictionImpulse * frictionScale);
            const float lo = -maxDynFrictionImpulse * frictionScale;
            const float hi = maxDynFrictionImpulse * frictionScale;
            // PxClamp(v,lo,hi) = PxMin(hi, PxMax(lo, v)) -- selectMin/selectMax
            const float totalClamped = pxMinQ(hi, pxMaxQ(lo, totalImpulse));
            const float newAppliedForce = clamp ? totalClamped : totalImpulse;
            const float deltaF0 = newAppliedForce - appliedForce0;

            v0Bot += ref0 * f0.dRA.xyz * deltaF0;
            v1Bot += ref1 * float3(f0.dRA2.z, f0.dRA2.w, f0.dRB.x) * deltaF0;
            v0Top += ref0 * float3(f0.dRA.w, f0.dRA2.x, f0.dRA2.y) * deltaF0;
            v1Top += ref1 * f0.dRB.yzw * deltaF0;

            o.fApplied[numFrictionConstr / 4u][numFrictionConstr & 3u] = newAppliedForce;
            broken = broken | (clamp ? 1u : 0u);

            imp0Bot -= raXn0 * deltaF0;   // bottom only (no top)
            imp1Bot += rbXn0 * deltaF0;
        }

        o.broken.x = broken;
    }

    // final impulse stores (cu L732-733): scale(l,a) = (top*l, bottom*a).
    // SYMMETRIC lane pairing: BOTH bodies (top*invMass, bottom*angDom).
    o.imp0 = float4(atcPlusZero3(imp0Top) * (invMass0 * ref0), 0.0f);
    o.imp0b = float4(atcPlusZero3(imp0Bot) * (angDom0 * ref0), 0.0f);
    o.imp1 = float4(atcPlusZero3(imp1Top) * (invMass1 * ref1), 0.0f);
    o.imp1b = float4(atcPlusZero3(imp1Bot) * (angDom1 * ref1), 0.0f);

    o.vel0Top = float4(v0Top, 0.0f);
    o.vel0Bot = float4(v0Bot, 0.0f);
    o.vel1Top = float4(v1Top, 0.0f);
    o.vel1Bot = float4(v1Bot, 0.0f);

    gOut[tid] = o;
}
