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


// npArtiSolveExtPGS.slang
//
// P8.122a: the PGS articulation EXT-body solve pair on the ENGINE BLOCK
// LAYOUT — `solveExtContactsBlock` (solver.cuh:54-320) and `solveExt1DBlock`
// (solverBlock.cuh:208-326), the two bodies artiSolveBlockPartition (P8.122b)
// dispatches on through batch.constraintType.  The Phase-1 batteries
// (npArtiSolveContact / npArtiSolve1D) pinned this MATH on abstracted flat
// inputs; this module is the same math over the real PxgBlockSolver* SoA
// blocks (file-local SvpPgs* mirrors, the P8.120/121 pattern) plus the
// PxgArticulationBlockResponse rows.  The TGS twin is npArtiSolveExtTGS
// (P8.99a).
//
// ===========================================================================
// TRAPS (each carries a mutant unless marked)
// ===========================================================================
//
//  1. **The CUDA software-pipelines its loads** (nextRaxn... prefetch with a
//     running `resp++` cursor); the port un-pipelines into direct loads with
//     the SAME indices: contact i reads response base+i, friction row i reads
//     base+numNormalConstr+i — the friction rows CONTINUE where the contacts
//     stopped.  (The pipelined form would read contacts[0]'s stale prefetch
//     into friction row 0 if numNormalConstr were 0 with frictions present —
//     unreachable: the prep's zero-contact early-out also zeroes
//     numFrictionConstr, so both forms agree on all reachable states.)
//  2. **p8 is a FIXED 0.8 constant** — the TGS twin reads the header's p8
//     slot; the PGS contact header has no such field.
//  3. **The normal update carries the IMPULSE MULTIPLIER**:
//     _newForce = appliedForce * impulseMul + _deltaF (the TGS twin has no
//     multiplier), with velMultiplier/impulseMul/biasedErr from
//     computeContactCoefficients over the header's RAW restitution and the
//     row's coeff0/coeff1.
//  4. **The friction velMultiplier divides by resp ALONE** —
//     (resp > PX_EPS_REAL) ? p8/resp : 0 — the TGS twin divides by
//     (resp + cfm).
//  5. **Friction rows clamp INDEPENDENTLY per row** (|totalImpulse| against
//     maxFrictionImpulse, dyn clamp via min/max) — the TGS twin clamps the
//     COMBINED pair magnitude; the tangent alternates as
//     frictionNormals[i & 1].
//  6. **NO torsional tail and no motion terms**: the PGS friction bias comes
//     precomputed in raXn_bias.w (bias - targetVel enters tmp1), there is no
//     elapsedTime/delta/deltaQ input anywhere.
//  7. **TENTH UPSTREAM-BUG CANDIDATE (transcribed VERBATIM)**: the final
//     contact impulse scale is ASYMMETRIC between the bodies —
//     impulse0 = imp0.scale(ref0*angDom0, ref0*lin0)   [solver.cu... :315]
//     impulse1 = imp1.scale(ref1*lin1,    ref1*ang1)   [:316]
//     although imp0.top and imp1.top accumulate the SAME normal-direction
//     sums.  Body A's two factors are swapped relative to body B AND to the
//     TGS twin (solver.cuh:732: scale(lin0, angDom0)).  The "fixing" mutant
//     (x, z) must DIE — the port reproduces the engine bit-for-bit.
//  8. **The 1D accumulators SEED from the inout impulse arguments**
//     (li0 = impulse0.bottom, ai0 = impulse0.top — note the crossed halves);
//     the TGS twin zeroes internally.  The partition caller passes zeros.
//  9. **The 1D row force update multiplies appliedForce by iMul** from
//     compute1dConstraintSolverConstantsPGS (spring/accel-spring/hard arms).
// 10. **The 1D unitResponse ADDS the header cfm** (resp0 + resp1 + cfm),
//     then recip via DY_MIN_RESPONSE = 1e-12.
// 11. The 1D final packs LINEAR in top (li*invMass) and angular in bottom
//     (ai*invInertiaScale) — shared with the TGS twin, swept again here.
// 12. ref0/ref1 scale the responses, the invMasses/inertiaScales AND the
//     response rows; the header rowCounts is a packed PxU16 pair per word.
// ===========================================================================

import pxgSlangCommon;
import pxgSlangSpatialMatrix;
import pxgSlangSolverBlocks;   // PxgBlockConstraintBatch + PxgArticulationBlockResponse

static const float SVP_PX_MAX_F32 = 3.4028234663852885981170418348452e+38f;
static const float SVP_PX_EPS_REAL = 1.192092896e-07f;
static const uint SVP_DY_SC_FLAG_SPRING = 1u << 5;
static const uint SVP_DY_SC_FLAG_ACCELERATION_SPRING = 1u << 7;
static const float SVP_DY_MIN_RESPONSE = 1e-12f;
static const uint SVP_COMPLIANT_ACCEL_SPRING = 1u << 3;   // contact header flag

// ---------------------------------------------------------------------------
// the PGS solver block mirrors (PxgConstraintBlock.h /
// PxgSolverConstraintBlock1D.h), file-local as in P8.120/121
// ---------------------------------------------------------------------------

// mirrors PxgBlockSolverContactHeader (1920 bytes)
struct SvpPgsContactHeader
{
    float4 invMass0_1_angDom0_1[32];  // 0
    float4 normal_staticFriction[32]; // 512
    float accumNormalForce[32];       // 1024 (not touched by this solve)
    float minNormalForce[32];         // 1152
    uint flags[32];                   // 1280
    uint numNormalConstr[32];         // 1408
    uint forceWritebackOffset[32];    // 1536
    float restitution[32];            // 1664
    float cfm[32];                    // 1792
}                                     // = 1920

// mirrors PxgBlockSolverFrictionHeader (1408 bytes)
struct SvpPgsFrictionHeader
{
    float4 frictionNormals0[32];  // 0
    float4 frictionNormals1[32];  // 512
    uint numFrictionConstr[32];   // 1024
    float dynamicFriction[32];    // 1152
    uint broken[32];              // 1280
}                                 // = 1408

// mirrors PxgBlockSolverContactPoint (1664 bytes)
struct SvpPgsContactPoint
{
    float4 raXn_targetVelocity[32]; // 0
    float4 rbXn_maxImpulse[32];     // 512
    float appliedForce[32];         // 1024
    float resp0[32];                // 1152
    float resp1[32];                // 1280
    float coeff0[32];               // 1408
    float coeff1[32];               // 1536
}                                   // = 1664

// mirrors PxgBlockSolverContactFriction (1408 bytes)
struct SvpPgsContactFriction
{
    float4 raXn_bias[32];       // 0
    float4 rbXn_targetVelW[32]; // 512
    float appliedForce[32];     // 1024
    float resp0[32];            // 1152
    float resp1[32];            // 1280
}                               // = 1408

// mirrors PxgBlockSolverConstraint1DHeader (1664 bytes; PxU16 rowCounts /
// breakable are packed two lanes per word)
struct SvpPgs1DHeader
{
    uint rowCounts[16];    // 0
    uint pad0[16];
    uint breakable[16];    // 128
    uint pad1[16];
    float angBreakImpulse[32];                   // 256
    float4 body0WorldOffset_linBreakImpulse[32]; // 384
    float invMass0D0[32];                        // 896
    float invMass1D1[32];                        // 1024
    float invInertiaScale0[32];                  // 1152
    float invInertiaScale1[32];                  // 1280
    uint writeBackOffset[32];                    // 1408
    float cfm[32];                               // 1536
}                                                // = 1664

// mirrors PxgBlockSolverConstraint1DCon (2304 bytes)
struct SvpPgs1DCon
{
    float4 lin0XYZ_minImpulse[32]; // 0
    float4 lin1XYZ_maxImpulse[32]; // 512
    float4 ang0XYZ_resp0[32];      // 1024
    float4 ang1XYZ_resp1[32];      // 1536
    float initJointSpeed[32];      // 2048
    uint pad0[32];                 // 2176
}                                  // = 2304

// mirrors PxgBlockSolverConstraint1DMod (1024 bytes)
struct SvpPgs1DMod
{
    float ang0Writeback[96]; // 0: PxVec3[32], tightly packed
    float appliedForce[32];  // 384
    uint flags[32];          // 512
    float residual[32];      // 640
    float coeff0[32];        // 768
    float coeff1[32];        // 896
}                            // = 1024

// ---------------------------------------------------------------------------
// helpers (the proven solverPGS spellings)
// ---------------------------------------------------------------------------

public struct SvpError
{
    public float sumOfSquares;
    public int counter;
    public float maxError;
}

// PxgErrorAccumulator::accumulateErrorLocal
public void svpAccumulateError(inout SvpError e, float deltaF, float velocityMultiplier)
{
    const float r = velocityMultiplier == 0.0f ? 0.0f : deltaF / velocityMultiplier;
    e.sumOfSquares += r * r;
    ++e.counter;
    e.maxError = max(e.maxError, abs(r));
}

// PxgErrorAccumulator::calculateResidual
float svpCalculateResidual(float deltaF, float velocityMultiplier)
{
    return velocityMultiplier == 0.0f ? 0.0f : deltaF / velocityMultiplier;
}

// Dy::computeRecipUnitResponse
float svpRecipUnitResponse(float unitResponse, float minRowResponse)
{
    return unitResponse <= minRowResponse ? 0.0f : 1.0f / unitResponse;
}

// computeContactCoefficients (constraintPrepShared.cuh:314) -- the solve-side
// form over the prep's coeff0/coeff1
void svpComputeContactCoefficients(uint flags, float restitution, float unitResponse,
                                   float recipResponse, float targetVelocity,
                                   float coeff0, float coeff1,
                                   inout float velMultiplier, inout float impulseMultiplier,
                                   out float unbiasedErr, out float biasedErr)
{
    if (restitution < 0.0f)
    {
        const bool accelSpring = (flags & SVP_COMPLIANT_ACCEL_SPRING) != 0u;
        const float massIfAccelElseOne = accelSpring ? recipResponse : 1.0f;
        const float oneIfAccelElseR = accelSpring ? 1.0f : unitResponse;

        const float a = coeff0;
        const float b = coeff1 * massIfAccelElseOne;

        const float x = 1.0f / (a * oneIfAccelElseR + 1.0f);
        const float scaledBias = x * b;

        velMultiplier = x * a * massIfAccelElseOne;
        impulseMultiplier = 1.0f - x;
        biasedErr = targetVelocity * velMultiplier - scaledBias;
        unbiasedErr = biasedErr;
    }
    else
    {
        velMultiplier = recipResponse;
        biasedErr = coeff0 * velMultiplier;
        unbiasedErr = coeff1 * velMultiplier;
        impulseMultiplier = 1.0f;
    }
}

// Dy::compute1dConstraintSolverConstantsPGS (reduced-coefficient overload,
// DyCpuGpu1dConstraint.h:262); coeff2 carries initJointSpeed
void svp1dConstantsPGS(bool isSpring, bool isAccelerationSpring, float coeff0,
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

// ===========================================================================
// solveExtContactsBlock (solver.cuh:54-320)
// ===========================================================================

void svpSolveExtContactsBlock(
    Ptr<PxgBlockConstraintBatch> batch,
    inout AbSpatialVector vel0, inout AbSpatialVector vel1,
    bool doFriction,
    uint threadIndex,
    Ptr<SvpPgsContactHeader> contactHeaders,
    Ptr<SvpPgsFrictionHeader> frictionHeaders,
    Ptr<SvpPgsContactPoint> contactPoints,
    Ptr<SvpPgsContactFriction> frictionPoints,
    Ptr<PxgArticulationBlockResponse> responses,
    out AbSpatialVector impulse0, out AbSpatialVector impulse1,
    inout SvpError error, bool accumulateError,
    float ref0, float ref1)
{
    AbSpatialVector imp0 = asmSVec(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f));
    AbSpatialVector imp1 = imp0;

    const uint batchIndex = batch[0].mConstraintBatchIndex;

    const float4 invMass0_1_angDom0_1 = contactHeaders[batchIndex].invMass0_1_angDom0_1[threadIndex];

    const uint numNormalConstr = contactHeaders[batchIndex].numNormalConstr[threadIndex];
    const uint numFrictionConstr = frictionHeaders[batchIndex].numFrictionConstr[threadIndex];

    // trap 3: the RAW restitution feeds the coefficient recompute
    const float restitution = contactHeaders[batchIndex].restitution[threadIndex];
    const float p8 = 0.8f;   // trap 2: FIXED (the PGS header has no p8 slot)
    const float cfm = contactHeaders[batchIndex].cfm[threadIndex];
    const uint flags = contactHeaders[batchIndex].flags[threadIndex];

    float accumulatedNormalImpulse = 0.0f;

    float3 linVel0 = vel0.bottom;
    float3 linVel1 = vel1.bottom;
    float3 angVel0 = vel0.top;
    float3 angVel1 = vel1.top;

    const float4 normal_staticFriction = contactHeaders[batchIndex].normal_staticFriction[threadIndex];
    const float3 normal = normal_staticFriction.xyz;
    const float staticCof = normal_staticFriction.w;

    // trap 1: the un-pipelined running response cursor
    uint respIndex = batch[0].mArticulationResponseIndex;

    for (uint i = 0u; i < numNormalConstr; ++i)
    {
        const uint c = batch[0].startConstraintIndex + i;
        const uint r = respIndex;
        respIndex++;

        const float4 raXn_targetVelocity = contactPoints[c].raXn_targetVelocity[threadIndex];
        const float4 rbXn_maxImpulse = contactPoints[c].rbXn_maxImpulse[threadIndex];
        const float appliedForce = contactPoints[c].appliedForce[threadIndex];
        const float resp0 = contactPoints[c].resp0[threadIndex];
        const float resp1 = contactPoints[c].resp1[threadIndex];
        const float coeff0 = contactPoints[c].coeff0[threadIndex];
        const float coeff1 = contactPoints[c].coeff1[threadIndex];

        const float3 deltaRALin = float3(responses[r].deltaRALin_x[threadIndex],
                                         responses[r].deltaRALin_y[threadIndex],
                                         responses[r].deltaRALin_z[threadIndex]);
        const float3 deltaRAAng = float3(responses[r].deltaRAAng_x[threadIndex],
                                         responses[r].deltaRAAng_y[threadIndex],
                                         responses[r].deltaRAAng_z[threadIndex]);
        const float3 deltaRBLin = float3(responses[r].deltaRBLin_x[threadIndex],
                                         responses[r].deltaRBLin_y[threadIndex],
                                         responses[r].deltaRBLin_z[threadIndex]);
        const float3 deltaRBAng = float3(responses[r].deltaRBAng_x[threadIndex],
                                         responses[r].deltaRBAng_y[threadIndex],
                                         responses[r].deltaRBAng_z[threadIndex]);

        const float3 raXn = raXn_targetVelocity.xyz;
        const float3 rbXn = rbXn_maxImpulse.xyz;
        const float targetVelocity = raXn_targetVelocity.w;
        const float maxImpulse = rbXn_maxImpulse.w;

        const float unitResponse = ref0 * resp0 + ref1 * resp1;
        const float recipResponse = (unitResponse > 0.0f) ? 1.0f / (unitResponse + cfm) : 0.0f;
        float velMultiplier = recipResponse;
        float impulseMul = 1.0f;
        float unbiasedError;
        float biasedErr;

        svpComputeContactCoefficients(flags, restitution, unitResponse, recipResponse,
                                      targetVelocity, coeff0, coeff1,
                                      velMultiplier, impulseMul, unbiasedError, biasedErr);

        // trap 12: ref-scaled response deltas, top = angular
        const float3 dVATop = ref0 * deltaRAAng;
        const float3 dVABot = ref0 * deltaRALin;
        const float3 dVBTop = ref1 * deltaRBAng;
        const float3 dVBBot = ref1 * deltaRBLin;

        const float v0_ = asmDot(linVel0, normal) + asmDot(angVel0, raXn);
        const float v1_ = asmDot(linVel1, normal) + asmDot(angVel1, rbXn);
        const float normalVel = v0_ - v1_;

        // trap 3: the impulse multiplier on appliedForce
        const float tempDeltaF = biasedErr - normalVel * velMultiplier;
        const float _deltaF = max(tempDeltaF, -appliedForce);
        const float _newForce = appliedForce * impulseMul + _deltaF;
        const float newForce = min(_newForce, maxImpulse);
        const float deltaF = newForce - appliedForce;

        if (accumulateError)
            svpAccumulateError(error, deltaF, velMultiplier);
        contactPoints[c].appliedForce[threadIndex] = newForce;

        imp0.bottom -= raXn * deltaF;
        imp0.top -= normal * deltaF;
        imp1.bottom += rbXn * deltaF;
        imp1.top += normal * deltaF;

        angVel0 += dVATop * deltaF;
        linVel0 += dVABot * deltaF;
        angVel1 += dVBTop * deltaF;
        linVel1 += dVBBot * deltaF;

        accumulatedNormalImpulse = accumulatedNormalImpulse + newForce;
    }

    // the minNormalForce floor happens unconditionally after the loop
    accumulatedNormalImpulse = max(accumulatedNormalImpulse,
                                   contactHeaders[batchIndex].minNormalForce[threadIndex]);

    if (numFrictionConstr != 0u && doFriction)
    {
        const float dynamicFrictionCof = frictionHeaders[batchIndex].dynamicFriction[threadIndex];
        const float maxFrictionImpulse = staticCof * accumulatedNormalImpulse;
        const float maxDynFrictionImpulse = dynamicFrictionCof * accumulatedNormalImpulse;

        uint broken = 0u;

        for (uint i = 0u; i < numFrictionConstr; ++i)
        {
            const uint f = batch[0].startFrictionIndex + i;
            const uint r = respIndex;
            respIndex++;

            // trap 5: the alternating tangent
            const float4 frictionNormal = (i & 1u) != 0u
                ? frictionHeaders[batchIndex].frictionNormals1[threadIndex]
                : frictionHeaders[batchIndex].frictionNormals0[threadIndex];

            const float4 raXn_extraCoeff = frictionPoints[f].raXn_bias[threadIndex];
            const float4 rbXn_targetVelW = frictionPoints[f].rbXn_targetVelW[threadIndex];
            const float resp0 = frictionPoints[f].resp0[threadIndex];
            const float resp1 = frictionPoints[f].resp1[threadIndex];
            const float appliedForce = frictionPoints[f].appliedForce[threadIndex];

            const float3 deltaRALin = float3(responses[r].deltaRALin_x[threadIndex],
                                             responses[r].deltaRALin_y[threadIndex],
                                             responses[r].deltaRALin_z[threadIndex]);
            const float3 deltaRAAng = float3(responses[r].deltaRAAng_x[threadIndex],
                                             responses[r].deltaRAAng_y[threadIndex],
                                             responses[r].deltaRAAng_z[threadIndex]);
            const float3 deltaRBLin = float3(responses[r].deltaRBLin_x[threadIndex],
                                             responses[r].deltaRBLin_y[threadIndex],
                                             responses[r].deltaRBLin_z[threadIndex]);
            const float3 deltaRBAng = float3(responses[r].deltaRBAng_x[threadIndex],
                                             responses[r].deltaRBAng_y[threadIndex],
                                             responses[r].deltaRBAng_z[threadIndex]);

            const float3 raXn = raXn_extraCoeff.xyz;
            const float3 rbXn = rbXn_targetVelW.xyz;

            // trap 4: resp alone in the divisor
            const float resp = ref0 * resp0 + ref1 * resp1;
            const float velMultiplier = (resp > SVP_PX_EPS_REAL) ? (p8 / resp) : 0.0f;
            const float bias = raXn_extraCoeff.w;
            const float targetVel = rbXn_targetVelW.w;

            const float3 dVATop = ref0 * deltaRAAng;
            const float3 dVABot = ref0 * deltaRALin;
            const float3 dVBTop = ref1 * deltaRBAng;
            const float3 dVBBot = ref1 * deltaRBLin;

            const float3 fNormal = frictionNormal.xyz;

            const float v0_ = asmDot(angVel0, raXn) + asmDot(linVel0, fNormal);
            const float v1_ = asmDot(angVel1, rbXn) + asmDot(linVel1, fNormal);
            const float normalVel = v0_ - v1_;

            const float tmp1 = appliedForce - (bias - targetVel) * velMultiplier;
            const float totalImpulse = tmp1 - normalVel * velMultiplier;

            // trap 5: independent per-row clamp
            const bool doClamp = abs(totalImpulse) > maxFrictionImpulse;
            const float totalClamped = min(maxDynFrictionImpulse,
                                           max(-maxDynFrictionImpulse, totalImpulse));
            const float newAppliedForce = doClamp ? totalClamped : totalImpulse;
            const float deltaF = newAppliedForce - appliedForce;

            if (accumulateError)
                svpAccumulateError(error, deltaF, velMultiplier);

            angVel0 += dVATop * deltaF;
            linVel0 += dVABot * deltaF;
            angVel1 += dVBTop * deltaF;
            linVel1 += dVBBot * deltaF;

            imp0.bottom -= raXn * deltaF;
            imp0.top -= fNormal * deltaF;
            imp1.bottom += rbXn * deltaF;
            imp1.top += fNormal * deltaF;

            frictionPoints[f].appliedForce[threadIndex] = newAppliedForce;
            broken = broken | (doClamp ? 1u : 0u);
        }
        frictionHeaders[batchIndex].broken[threadIndex] = broken;
    }

    // trap 7: the TENTH upstream-bug candidate, verbatim -- body A's factors
    // are swapped ((angDom0, lin0)) relative to body B ((lin1, ang1)) and to
    // the TGS twin ((lin0, angDom0)); UnAlignedSpatialVector::scale scales
    // top by the first factor, bottom by the second
    impulse0 = asmSVec(imp0.top * (ref0 * invMass0_1_angDom0_1.z),
                       imp0.bottom * (ref0 * invMass0_1_angDom0_1.x));
    impulse1 = asmSVec(imp1.top * (ref1 * invMass0_1_angDom0_1.y),
                       imp1.bottom * (ref1 * invMass0_1_angDom0_1.w));

    vel0.bottom = linVel0;
    vel0.top = angVel0;
    vel1.bottom = linVel1;
    vel1.top = angVel1;
}

// ===========================================================================
// solveExt1DBlock (solverBlock.cuh:208-326)
// ===========================================================================

void svpSolveExt1DBlock(
    Ptr<PxgBlockConstraintBatch> batch,
    inout AbSpatialVector vel0, inout AbSpatialVector vel1,
    uint threadIndex,
    Ptr<SvpPgs1DHeader> headers,
    Ptr<SvpPgs1DCon> rowsCon,
    Ptr<SvpPgs1DMod> rowsMod,
    Ptr<PxgArticulationBlockResponse> artiResponse,
    inout AbSpatialVector impulse0, inout AbSpatialVector impulse1,
    bool residualReportingEnabled,
    float ref0, float ref1)
{
    const uint batchIndex = batch[0].mConstraintBatchIndex;
    const uint startRow = batch[0].startConstraintIndex;

    float3 linVel0 = vel0.bottom;
    float3 linVel1 = vel1.bottom;
    float3 angVel0 = vel0.top;
    float3 angVel1 = vel1.top;

    // trap 8: the accumulators SEED from the inout impulses, crossed halves
    float3 li0 = impulse0.bottom;
    float3 li1 = impulse1.bottom;
    float3 ai0 = impulse0.top;
    float3 ai1 = impulse1.top;

    const float invMass0 = ref0 * headers[batchIndex].invMass0D0[threadIndex];
    const float invMass1 = ref1 * headers[batchIndex].invMass1D1[threadIndex];

    const float invInertiaScale0 = ref0 * headers[batchIndex].invInertiaScale0[threadIndex];
    const float invInertiaScale1 = ref1 * headers[batchIndex].invInertiaScale1[threadIndex];

    // trap 10
    const float cfm = headers[batchIndex].cfm[threadIndex];

    // trap 12: PxU16 rowCounts, two lanes per word
    const uint numRows = (headers[batchIndex].rowCounts[threadIndex >> 1]
                          >> ((threadIndex & 1u) * 16u)) & 0xFFFFu;

    for (uint i = 0u; i < numRows; ++i)
    {
        const uint c = startRow + i;

        const float4 _clinVel0_minImpulse = rowsCon[c].lin0XYZ_minImpulse[threadIndex];
        const float4 _clinVel1_maxImpulse = rowsCon[c].lin1XYZ_maxImpulse[threadIndex];
        const float4 _cangVel0_resp0 = rowsCon[c].ang0XYZ_resp0[threadIndex];
        const float4 _cangVel1_resp1 = rowsCon[c].ang1XYZ_resp1[threadIndex];

        const float resp0 = ref0 * _cangVel0_resp0.w;
        const float resp1 = ref1 * _cangVel1_resp1.w;
        const float initJointSpeed = rowsCon[c].initJointSpeed[threadIndex];

        const float3 clinVel0 = _clinVel0_minImpulse.xyz;
        const float3 clinVel1 = _clinVel1_maxImpulse.xyz;
        const float3 cangVel0 = _cangVel0_resp0.xyz;
        const float3 cangVel1 = _cangVel1_resp1.xyz;

        const float coeff0 = rowsMod[c].coeff0[threadIndex];
        const float coeff1 = rowsMod[c].coeff1[threadIndex];
        const uint flags = rowsMod[c].flags[threadIndex];

        const float unitResponse = resp0 + resp1 + cfm;   // trap 10

        const float recipResponse = svpRecipUnitResponse(unitResponse, SVP_DY_MIN_RESPONSE);

        const bool isSpring = (flags & SVP_DY_SC_FLAG_SPRING) != 0u;
        const bool isAccelerationSpring = (flags & SVP_DY_SC_FLAG_ACCELERATION_SPRING) != 0u;

        float constant;
        float unbiasedConstant;
        float vMul;
        float iMul;
        svp1dConstantsPGS(isSpring, isAccelerationSpring, coeff0, coeff1, initJointSpeed,
                          unitResponse, recipResponse, constant, unbiasedConstant, vMul, iMul);

        const float appliedForce = rowsMod[c].appliedForce[threadIndex];
        const float maxImpulse = _clinVel1_maxImpulse.w;
        const float minImpulse = _clinVel0_minImpulse.w;

        const float v0 = asmDot(linVel0, clinVel0) + asmDot(angVel0, cangVel0);
        const float v1 = asmDot(linVel1, clinVel1) + asmDot(angVel1, cangVel1);
        const float normalVel = v0 - v1;

        // trap 9: iMul on appliedForce
        const float unclampedForce = iMul * appliedForce + (vMul * normalVel + constant);
        const float clampedForce = min(maxImpulse, max(minImpulse, unclampedForce));
        const float deltaF = clampedForce - appliedForce;

        rowsMod[c].appliedForce[threadIndex] = clampedForce;
        if (residualReportingEnabled)
            rowsMod[c].residual[threadIndex] = svpCalculateResidual(deltaF, vMul);

        li0 = clinVel0 * deltaF + li0;
        ai0 = cangVel0 * deltaF + ai0;
        li1 = clinVel1 * deltaF + li1;
        ai1 = cangVel1 * deltaF + ai1;

        // the response base is the CALLER's &responses[responseIndex]; row i
        // reads entry i of that base (trap 12: ref-scaled)
        const float3 linVa = ref0 * float3(artiResponse[i].deltaRALin_x[threadIndex],
                                           artiResponse[i].deltaRALin_y[threadIndex],
                                           artiResponse[i].deltaRALin_z[threadIndex]);
        const float3 angVa = ref0 * float3(artiResponse[i].deltaRAAng_x[threadIndex],
                                           artiResponse[i].deltaRAAng_y[threadIndex],
                                           artiResponse[i].deltaRAAng_z[threadIndex]);
        const float3 linVb = ref1 * float3(artiResponse[i].deltaRBLin_x[threadIndex],
                                           artiResponse[i].deltaRBLin_y[threadIndex],
                                           artiResponse[i].deltaRBLin_z[threadIndex]);
        const float3 angVb = ref1 * float3(artiResponse[i].deltaRBAng_x[threadIndex],
                                           artiResponse[i].deltaRBAng_y[threadIndex],
                                           artiResponse[i].deltaRBAng_z[threadIndex]);

        linVel0 = linVa * deltaF + linVel0;
        angVel0 = angVa * deltaF + angVel0;
        linVel1 = linVb * deltaF + linVel1;
        angVel1 = angVb * deltaF + angVel1;
    }

    vel0.top = angVel0;
    vel0.bottom = linVel0;
    vel1.top = angVel1;
    vel1.bottom = linVel1;

    // trap 11: LINEAR in top by invMass, angular in bottom by inertia scale
    impulse0 = asmSVec(li0 * invMass0, ai0 * invInertiaScale0);
    impulse1 = asmSVec(li1 * invMass1, ai1 * invInertiaScale1);
}

// ---------------------------------------------------------------------------
// fixture entry points: one batch, 32 lanes = 32 independent constraint slots
// (the npArtiSolveExtTGS arrangement; the io layout keeps the same 32-float
// stride -- PGS has no delta/deltaQ inputs, those slots stay unread)
// ---------------------------------------------------------------------------

struct SvpParams
{
    uint64_t batch;    // @0   PxgBlockConstraintBatch*
    uint64_t h0;       // @8   contact: contactHeaders / 1D: headers
    uint64_t h1;       // @16  contact: frictionHeaders / 1D: rowsCon
    uint64_t p0;       // @24  contact: contactPoints / 1D: rowsMod
    uint64_t p1;       // @32  contact: frictions / 1D: unused
    uint64_t responses;// @40
    uint64_t io;       // @48
    float ref0;        // @56
    float ref1;        // @60
    uint nbLanes;      // @64
    uint flagsWord;    // @68: bit0 accumulateError/residualReporting, bit1 doFriction
}

[[vk::push_constant]]
ConstantBuffer<SvpParams> gSvp;

static const uint SVP_IO_IN_STRIDE = 32u;
static const uint SVP_IO_OUT_BASE = 32u * 32u;

void svpLoadIo(uint lane, out AbSpatialVector vel0, out AbSpatialVector vel1)
{
    Ptr<float> f = Ptr<float>(gSvp.io + uint64_t(lane) * SVP_IO_IN_STRIDE * 4u);
    vel0 = asmSVec(float3(f[0], f[1], f[2]), float3(f[3], f[4], f[5]));
    vel1 = asmSVec(float3(f[6], f[7], f[8]), float3(f[9], f[10], f[11]));
}

void svpStoreIo(uint lane, AbSpatialVector vel0, AbSpatialVector vel1,
                AbSpatialVector imp0, AbSpatialVector imp1, SvpError err)
{
    Ptr<float> f = Ptr<float>(gSvp.io + uint64_t(SVP_IO_OUT_BASE + lane * SVP_IO_IN_STRIDE) * 4u);
    f[0] = vel0.top.x; f[1] = vel0.top.y; f[2] = vel0.top.z;
    f[3] = vel0.bottom.x; f[4] = vel0.bottom.y; f[5] = vel0.bottom.z;
    f[6] = vel1.top.x; f[7] = vel1.top.y; f[8] = vel1.top.z;
    f[9] = vel1.bottom.x; f[10] = vel1.bottom.y; f[11] = vel1.bottom.z;
    f[12] = imp0.top.x; f[13] = imp0.top.y; f[14] = imp0.top.z;
    f[15] = imp0.bottom.x; f[16] = imp0.bottom.y; f[17] = imp0.bottom.z;
    f[18] = imp1.top.x; f[19] = imp1.top.y; f[20] = imp1.top.z;
    f[21] = imp1.bottom.x; f[22] = imp1.bottom.y; f[23] = imp1.bottom.z;
    f[24] = err.sumOfSquares; f[25] = float(err.counter); f[26] = err.maxError;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void svpContactTest(uint3 groupThreadId : SV_GroupThreadID)
{
    const uint lane = groupThreadId.x;
    if (lane >= gSvp.nbLanes)
        return;

    AbSpatialVector vel0, vel1;
    svpLoadIo(lane, vel0, vel1);

    AbSpatialVector imp0, imp1;
    SvpError err = { 0.0f, 0, 0.0f };
    svpSolveExtContactsBlock(Ptr<PxgBlockConstraintBatch>(gSvp.batch), vel0, vel1,
                             (gSvp.flagsWord & 2u) != 0u,
                             lane,
                             Ptr<SvpPgsContactHeader>(gSvp.h0),
                             Ptr<SvpPgsFrictionHeader>(gSvp.h1),
                             Ptr<SvpPgsContactPoint>(gSvp.p0),
                             Ptr<SvpPgsContactFriction>(gSvp.p1),
                             Ptr<PxgArticulationBlockResponse>(gSvp.responses),
                             imp0, imp1,
                             err, (gSvp.flagsWord & 1u) != 0u, gSvp.ref0, gSvp.ref1);

    svpStoreIo(lane, vel0, vel1, imp0, imp1, err);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void svp1DTest(uint3 groupThreadId : SV_GroupThreadID)
{
    const uint lane = groupThreadId.x;
    if (lane >= gSvp.nbLanes)
        return;

    AbSpatialVector vel0, vel1;
    svpLoadIo(lane, vel0, vel1);

    // the partition caller zeroes the impulses before the 1D call; the test
    // does the same (the trap-8 seeding is pinned via the row math instead)
    AbSpatialVector imp0 = asmSVec(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f));
    AbSpatialVector imp1 = imp0;
    SvpError err = { 0.0f, 0, 0.0f };

    // the response base is offset by mArticulationResponseIndex at the CALL
    // SITE, exactly as the kernel does (&responses[responseIndex])
    const uint respIdx = (Ptr<PxgBlockConstraintBatch>(gSvp.batch))[0].mArticulationResponseIndex;
    Ptr<PxgArticulationBlockResponse> respBase = Ptr<PxgArticulationBlockResponse>(
        gSvp.responses + uint64_t(respIdx) * 1536u);

    svpSolveExt1DBlock(Ptr<PxgBlockConstraintBatch>(gSvp.batch), vel0, vel1,
                       lane,
                       Ptr<SvpPgs1DHeader>(gSvp.h0),
                       Ptr<SvpPgs1DCon>(gSvp.h1),
                       Ptr<SvpPgs1DMod>(gSvp.p0),
                       respBase,
                       imp0, imp1,
                       (gSvp.flagsWord & 1u) != 0u, gSvp.ref0, gSvp.ref1);

    svpStoreIo(lane, vel0, vel1, imp0, imp1, err);
}
