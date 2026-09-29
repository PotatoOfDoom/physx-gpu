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
import pxgSlangSolverBlocks;        // PxgBlockConstraintBatch + PxgArticulationBlockResponse
import pxgSlangSolverDesc;          // PxgSolverCoreDescGpu + PGS shared desc + countActiveSlabs
import pxgSlangArticulationBlocks;  // abStoreU8 + PXG_AB_* (the P8.122b entry)
import pxgSlangArticulationDesc;    // artiPtrField / PXG_ARTI_CORE_*

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
public struct SvpPgsContactHeader
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
public struct SvpPgsFrictionHeader
{
    float4 frictionNormals0[32];  // 0
    float4 frictionNormals1[32];  // 512
    uint numFrictionConstr[32];   // 1024
    float dynamicFriction[32];    // 1152
    uint broken[32];              // 1280
}                                 // = 1408

// mirrors PxgBlockSolverContactPoint (1664 bytes)
public struct SvpPgsContactPoint
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
public struct SvpPgsContactFriction
{
    float4 raXn_bias[32];       // 0
    float4 rbXn_targetVelW[32]; // 512
    float appliedForce[32];     // 1024
    float resp0[32];            // 1152
    float resp1[32];            // 1280
}                               // = 1408

// mirrors PxgBlockSolverConstraint1DHeader (1664 bytes; PxU16 rowCounts /
// breakable are packed two lanes per word)
public struct SvpPgs1DHeader
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
public struct SvpPgs1DCon
{
    float4 lin0XYZ_minImpulse[32]; // 0
    float4 lin1XYZ_maxImpulse[32]; // 512
    float4 ang0XYZ_resp0[32];      // 1024
    float4 ang1XYZ_resp1[32];      // 1536
    float initJointSpeed[32];      // 2048
    uint pad0[32];                 // 2176
}                                  // = 2304

// mirrors PxgBlockSolverConstraint1DMod (1024 bytes)
public struct SvpPgs1DMod
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
    // P9.89: maxError = PxMax(maxError, PxAbs(e)) (solverResidual.cuh:57) is selectMax(a, b) =
    // a > b ? a : b (PxUnixMathIntrinsics.h:122), a serial select chain over the rows: a NaN
    // residual replaces the running maximum and the next row's |e| replaces the NaN.  GLSL max()
    // (FMax, NaN-ignoring on stamped llvmpipe) kept the old maximum across the NaN row.  Feeds
    // artiSolveBlockPartition and the PGS static contacts (npArtiSolveInternal, npArtiSolveSelf).
    const float ar = abs(r);
    e.maxError = (e.maxError > ar) ? e.maxError : ar;
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

public void svpSolveExtContactsBlock(
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
    impulse0 = asmSVec(asepPlusZero3(imp0.top) * (ref0 * invMass0_1_angDom0_1.z),
                       asepPlusZero3(imp0.bottom) * (ref0 * invMass0_1_angDom0_1.x));
    impulse1 = asmSVec(asepPlusZero3(imp1.top) * (ref1 * invMass0_1_angDom0_1.y),
                       asepPlusZero3(imp1.bottom) * (ref1 * invMass0_1_angDom0_1.w));

    vel0.bottom = linVel0;
    vel0.top = angVel0;
    vel1.bottom = linVel1;
    vel1.top = angVel1;
}

// ===========================================================================
// solveExt1DBlock (solverBlock.cuh:208-326)
// ===========================================================================

public void svpSolveExt1DBlock(
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

// P9.86: the impulse accumulators start at a literal +0 and only ever += / -= a term;
// an IEEE chain from +0 never holds -0 (exact cancellation rounds to +0).  llvmpipe
// unrolls the <= 6-trip loops, forwards the literal seed and rewrites the first
// `0 - t` to -t and `0 + t` to t, even under SignedZeroInfNanPreserve (measured).
// The fold only ever changes the sign of a zero, so the finished accumulator is
// canonicalised (-0 -> +0) before the final scale: bit for bit the engine's value.
float3 asepPlusZero3(float3 v)
{
    return float3((v.x == 0.0f) ? 0.0f : v.x, (v.y == 0.0f) ? 0.0f : v.y, (v.z == 0.0f) ? 0.0f : v.z);
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

// ===========================================================================
// P8.122b: the ENGINE entry -- artiSolveBlockPartition (solver.cu:47-266),
// the PGS twin of P8.104's artiSolveBlockPartitionTGSInternal.  Launch:
// grid nbArtiBlocks x block (32, 2, 1) (COMPUTE_UNCONSTRAINED_VELOCITES/32
// warps per block, PxgCudaSolverCore.cpp:1194).  Params (8,8,4,4,1,8).
//
// TRAPS, continuing the module numbering (each a sweep mutant):
// 13. **startIndex is partition-0-special** (island.mArtiBatchStartIndex vs
//     artiConstraintsPerPartition[partitionIndex + startPartitionIndex - 1]);
//     endIndex adds islandContextPool[0].mBatchCount, startIndex does NOT
//     (it enters k separately).
// 14. **mBatchCount / mArticulationCount / mBodyStartIndex / mBodyCount read
//     islandContextPool[0]** (the CUDA derefs the POINTER), while
//     mStartPartitionIndex/mArtiBatchStartIndex read pool[islandIndex].
// 15. **The 2-slot velocity staging** at readIndex = k*128 + lane: slots
//     0/32 = body A lin/ang, 64/96 = body B -- the PGS packing carries the
//     halves whole ((ang.xyz) top, (lin.xyz) bottom), unlike the TGS
//     3-slot split (trap 4 of P8.104).
// 16. curRef counts active slabs for articulations at remappedBodyIndex +
//     numDynamicBodies + bodyOffset, for rigids only when solverBodyId >=
//     bodyOffset; both default to 1.  (Shared with P8.104, swept again.)
// 17. **The rigid writeback goes to remappedBodyA/BIndex** in the 2-slot
//     packing (bottom, 0) / (top, 0); the arti writeback scatters the
//     IMPULSE to deferredZ[computeDeltaVIndex(...)], writes isSlabDirty
//     [artiId + slabId*nbArticulations].x = linkIndexA (.y for body B) and
//     sets mStateDirty = eHAS_IMPULSES | eVEL_DIRTY through the sub-word
//     atomic pair.  (The CUDA stores body B's deferredZ TWICE -- a direct
//     assignment at :249 immediately followed by storeSpatialVector on the
//     same 24 bytes; the port stores once.)
// 18. constraintType == eARTICULATION_CONTACT (3) selects the contact solve,
//     everything else the 1D solve (the CUDA asserts 1D); the 1D response
//     base is &responses[batch.mArticulationResponseIndex].
// 19. **The lane guard must not skip the error reduction**: the CUDA returns
//     for lanes >= mDescStride before the accumulate (zero contribution);
//     the port guards the body so the groupshared butterfly stays in
//     uniform control flow (the P8.104 trap-3 discipline).
// 20. doFriction reaches ONLY the contact solve; there is no elapsedTime /
//     minPen / deltaQ anywhere (the PGS solve needs no motion terms).
// ===========================================================================

static const uint SVP_ARTI_CONTACT = 3u;   // PxgSolverConstraintDesc::eARTICULATION_CONTACT
static const uint SVP_DIRTY = 3u;          // eHAS_IMPULSES | eVEL_DIRTY

// computeDeltaVIndex (PxgArticulation.h:967)
uint svpDeltaVIndex(uint maxArticulations, uint maxLinks, uint articulationID,
                    uint linkID, uint slabID)
{
    return articulationID + linkID * maxArticulations + slabID * maxArticulations * maxLinks;
}

struct SvpPartParams
{
    uint64_t solverCoreDesc;   // @0
    uint64_t sharedDesc;       // @8
    uint islandIndex;          // @16
    uint partitionIndex;       // @20
    uint doFrictionWord;       // @24 (bool byte, read masked)
    uint pad0;                 // @28
    uint64_t artiDesc;         // @32 -> 40
}

[[vk::push_constant]]
ConstantBuffer<SvpPartParams> gSvpPart;

// per-warp residual-reduction scratch: block = (32,2) -> 2 warps
groupshared float shSvpErrSum[64];
groupshared float shSvpErrMax[64];
groupshared uint  shSvpErrCount[64];

[shader("compute")]
[numthreads(32, 2, 1)]
void artiSolveBlockPartition(uint3 groupThreadId : SV_GroupThreadID,
                             uint3 groupId : SV_GroupID)
{
    const uint lane = groupThreadId.x;
    const uint warpIndex = groupThreadId.y;
    const uint globalWarpIndex = groupId.x * 2u + warpIndex;
    const uint threadInBlock = warpIndex * 32u + lane;

    Ptr<PxgSolverCoreDescGpu> solverDesc = Ptr<PxgSolverCoreDescGpu>(gSvpPart.solverCoreDesc);
    Ptr<PxgSolverSharedDescPGSGpu> sharedDesc =
        Ptr<PxgSolverSharedDescPGSGpu>(gSvpPart.sharedDesc);
    Ptr<PxgIslandContextGpu> islandPool =
        Ptr<PxgIslandContextGpu>(solverDesc[0].islandContextPool);

    const bool accumulateError = solverDesc[0].contactErrorAccumulator_mCounter >= 0;

    const uint64_t d = gSvpPart.artiDesc;
    const uint64_t gArticulations = artiPtrField(d, PXG_ARTI_CORE_ARTICULATION_BLOCKS);
    const uint maxLinks = artiU32Field(d, PXG_ARTI_CORE_MAX_LINKS);
    const uint64_t deferredZ = sharedDesc[0].articulationDeferredZ;
    const uint64_t isSlabDirty = artiPtrField(d, PXG_ARTI_CORE_SLAB_HAS_CHANGES);

    const uint startPartitionIndex = islandPool[gSvpPart.islandIndex].mStartPartitionIndex;
    Ptr<uint> artiConstraintsPerPartition = Ptr<uint>(solverDesc[0].artiConstraintsPerPartition);

    // trap 13 (with the P8.96c index clamp: the [idx-1] load is speculated on
    // the partition-0 arm too, where it would underflow)
    const uint prevIdx = (gSvpPart.partitionIndex == 0u)
        ? gSvpPart.partitionIndex + startPartitionIndex
        : gSvpPart.partitionIndex + startPartitionIndex - 1u;
    const uint prevCpp = artiConstraintsPerPartition[prevIdx];
    const uint startIndex = (gSvpPart.partitionIndex == 0u)
        ? islandPool[gSvpPart.islandIndex].mArtiBatchStartIndex
        : prevCpp;

    // trap 14: the POINTER deref -- element 0
    const uint articulationBatchOffset = islandPool[0].mBatchCount;
    const uint endIndex =
        artiConstraintsPerPartition[gSvpPart.partitionIndex + startPartitionIndex]
        + articulationBatchOffset;

    const uint k = startIndex + globalWarpIndex + articulationBatchOffset;

    SvpError error = { 0.0f, 0, 0.0f };

    if (k < endIndex)
    {
        const bool doFriction = (gSvpPart.doFrictionWord & 0xFFu) != 0u;

        Ptr<PxgBlockConstraintBatch> batches =
            Ptr<PxgBlockConstraintBatch>(sharedDesc[0].iter_blockConstraintBatch);
        Ptr<PxgBlockConstraintBatch> batch = batches + k;

        const uint descStride = batch[0].descStride_constraintType & 0xFFFFu;
        const uint constraintType = batch[0].descStride_constraintType >> 16u;

        // trap 19: guard, not return -- the butterfly below needs every lane
        if (lane < descStride)
        {
            const uint nbArticulations = islandPool[0].mArticulationCount;

            const uint2 igNodeIndexA = batch[0].bodyANodeIndex[lane];
            const uint2 igNodeIndexB = batch[0].bodyBNodeIndex[lane];
            const uint slabId = batch[0].slabId[lane];
            const uint solverBodyId0 = batch[0].bodyAIndex[lane];
            const uint solverBodyId1 = batch[0].bodyBIndex[lane];
            const uint linkIndexA = nodeIndexArticulationLinkId(igNodeIndexA);
            const uint linkIndexB = nodeIndexArticulationLinkId(igNodeIndexB);
            const bool isArtiA = nodeIndexIsArticulation(igNodeIndexA);
            const bool isArtiB = nodeIndexIsArticulation(igNodeIndexB);

            const uint readIndex = k * 128u + lane;

            Ptr<float4> bodyVelocities = Ptr<float4>(sharedDesc[0].iter_solverBodyVelPool);

            // trap 15: the 2-slot staging
            AbSpatialVector vel0, vel1;
            {
                const float4 lin = bodyVelocities[readIndex];
                const float4 ang = bodyVelocities[readIndex + 32u];
                vel0 = asmSVec(ang.xyz, lin.xyz);
            }
            {
                const float4 lin = bodyVelocities[readIndex + 64u];
                const float4 ang = bodyVelocities[readIndex + 96u];
                vel1 = asmSVec(ang.xyz, lin.xyz);
            }

            // trap 16: reference counts
            float curRef0 = 1.0f;
            float curRef1 = 1.0f;
            {
                const uint bodyOffset = islandPool[0].mBodyStartIndex;
                const uint numDynamicBodies = islandPool[0].mBodyCount;
                Ptr<uint> encodedReferenceCount =
                    Ptr<uint>(sharedDesc[0].iter_solverEncodedReferenceCount);
                const uint numTotalBodies = bodyOffset + numDynamicBodies + nbArticulations;
                const uint numSlabs = solverDesc[0].numSlabs;

                if (isArtiA)
                {
                    const uint globalBodyIdA =
                        batch[0].remappedBodyAIndex[lane] + numDynamicBodies + bodyOffset;
                    curRef0 = float(pxgCountActiveSlabsBda(globalBodyIdA, numSlabs,
                                                           numTotalBodies, encodedReferenceCount));
                }
                else if (solverBodyId0 >= bodyOffset)
                {
                    curRef0 = float(pxgCountActiveSlabsBda(solverBodyId0, numSlabs,
                                                           numTotalBodies, encodedReferenceCount));
                }

                if (isArtiB)
                {
                    const uint globalBodyIdB =
                        batch[0].remappedBodyBIndex[lane] + numDynamicBodies + bodyOffset;
                    curRef1 = float(pxgCountActiveSlabsBda(globalBodyIdB, numSlabs,
                                                           numTotalBodies, encodedReferenceCount));
                }
                else if (solverBodyId1 >= bodyOffset)
                {
                    curRef1 = float(pxgCountActiveSlabsBda(solverBodyId1, numSlabs,
                                                           numTotalBodies, encodedReferenceCount));
                }
            }

            AbSpatialVector impulse0 = asmSVec(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f));
            AbSpatialVector impulse1 = impulse0;

            // trap 18/20
            if (constraintType == SVP_ARTI_CONTACT)
            {
                svpSolveExtContactsBlock(batch, vel0, vel1, doFriction, lane,
                                         Ptr<SvpPgsContactHeader>(sharedDesc[0].iter_blockContactHeaders),
                                         Ptr<SvpPgsFrictionHeader>(sharedDesc[0].iter_blockFrictionHeaders),
                                         Ptr<SvpPgsContactPoint>(sharedDesc[0].iter_blockContactPoints),
                                         Ptr<SvpPgsContactFriction>(sharedDesc[0].iter_blockFrictions),
                                         Ptr<PxgArticulationBlockResponse>(sharedDesc[0].iter_artiResponse),
                                         impulse0, impulse1,
                                         error, accumulateError, curRef0, curRef1);
            }
            else
            {
                const uint responseIndex = batch[0].mArticulationResponseIndex;
                Ptr<PxgArticulationBlockResponse> respBase = Ptr<PxgArticulationBlockResponse>(
                    sharedDesc[0].iter_artiResponse + uint64_t(responseIndex) * 1536u);
                svpSolveExt1DBlock(batch, vel0, vel1, lane,
                                   Ptr<SvpPgs1DHeader>(sharedDesc[0].iter_blockJointConstraintHeaders),
                                   Ptr<SvpPgs1DCon>(sharedDesc[0].iter_blockJointConstraintRowsCon),
                                   Ptr<SvpPgs1DMod>(sharedDesc[0].iter_blockJointConstraintRowsMod),
                                   respBase,
                                   impulse0, impulse1,
                                   accumulateError, curRef0, curRef1);
            }

            // trap 17: the writebacks
            if (!isArtiA)
            {
                const uint outIndex = batch[0].remappedBodyAIndex[lane];
                bodyVelocities[outIndex] = float4(vel0.bottom, 0.0f);
                bodyVelocities[outIndex + 32u] = float4(vel0.top, 0.0f);
            }
            else
            {
                const uint index =
                    svpDeltaVIndex(nbArticulations, maxLinks, solverBodyId0, linkIndexA, slabId);
                Ptr<float> z = Ptr<float>(deferredZ + uint64_t(index) * 24u);
                z[0] = impulse0.top.x; z[1] = impulse0.top.y; z[2] = impulse0.top.z;
                z[3] = impulse0.bottom.x; z[4] = impulse0.bottom.y; z[5] = impulse0.bottom.z;

                (Ptr<uint>(isSlabDirty))[(solverBodyId0 + slabId * nbArticulations) * 2u] =
                    linkIndexA;

                const uint articBlockId = solverBodyId0 / 32u;
                abStoreU8(gArticulations + uint64_t(articBlockId) * PXG_AB_BLOCK_SIZE,
                          PXG_AB_mStateDirty, solverBodyId0 & 31u, SVP_DIRTY);
            }

            if (!isArtiB)
            {
                const uint outIndex = batch[0].remappedBodyBIndex[lane];
                bodyVelocities[outIndex] = float4(vel1.bottom, 0.0f);
                bodyVelocities[outIndex + 32u] = float4(vel1.top, 0.0f);
            }
            else
            {
                // (the CUDA's duplicated deferredZ store collapses to one)
                const uint index =
                    svpDeltaVIndex(nbArticulations, maxLinks, solverBodyId1, linkIndexB, slabId);
                Ptr<float> z = Ptr<float>(deferredZ + uint64_t(index) * 24u);
                z[0] = impulse1.top.x; z[1] = impulse1.top.y; z[2] = impulse1.top.z;
                z[3] = impulse1.bottom.x; z[4] = impulse1.bottom.y; z[5] = impulse1.bottom.z;

                (Ptr<uint>(isSlabDirty))[(solverBodyId1 + slabId * nbArticulations) * 2u + 1u] =
                    linkIndexB;

                const uint articBlockId = solverBodyId1 / 32u;
                abStoreU8(gArticulations + uint64_t(articBlockId) * PXG_AB_BLOCK_SIZE,
                          PXG_AB_mStateDirty, solverBodyId1 & 31u, SVP_DIRTY);
            }
        }
    }

    // PxgErrorAccumulator::accumulateErrorGlobalFullWarp: per-warp butterfly,
    // then lane 0 of each warp accumulates atomically (the solverPGS
    // solveBlockPartition arrangement; uniform control flow around barriers)
    GroupMemoryBarrierWithGroupSync();
    shSvpErrSum[threadInBlock] = error.sumOfSquares;
    shSvpErrMax[threadInBlock] = error.maxError;
    shSvpErrCount[threadInBlock] = uint(error.counter);
    GroupMemoryBarrierWithGroupSync();

    for (uint radius = 16u; radius > 0u; radius >>= 1)
    {
        const uint partner = threadInBlock ^ radius;   // stays inside the warp
        const float s = shSvpErrSum[partner];
        const float m = shSvpErrMax[partner];
        const uint c = shSvpErrCount[partner];
        GroupMemoryBarrierWithGroupSync();
        shSvpErrSum[threadInBlock] += s;
        shSvpErrMax[threadInBlock] = max(shSvpErrMax[threadInBlock], m);
        shSvpErrCount[threadInBlock] += c;
        GroupMemoryBarrierWithGroupSync();
    }

    if (accumulateError && lane == 0u)
    {
        const float sumSq = shSvpErrSum[threadInBlock];
        const uint count = shSvpErrCount[threadInBlock];
        const float maxErr = shSvpErrMax[threadInBlock];

        uint assumed = solverDesc[0].contactErrorAccumulator_mErrorSumOfSquares;
        for (;;)
        {
            const uint newBits = asuint(asfloat(assumed) + sumSq);
            uint old;
            InterlockedCompareExchange(solverDesc[0].contactErrorAccumulator_mErrorSumOfSquares,
                                       assumed, newBits, old);
            if (old == assumed)
                break;
            assumed = old;
        }

        int oldCount;
        InterlockedAdd(solverDesc[0].contactErrorAccumulator_mCounter, int(count), oldCount);

        if (maxErr > asfloat(solverDesc[0].contactErrorAccumulator_mMaxError))
        {
            assumed = solverDesc[0].contactErrorAccumulator_mMaxError;
            for (;;)
            {
                const uint newBits = asuint(max(maxErr, asfloat(assumed)));
                uint old;
                InterlockedCompareExchange(solverDesc[0].contactErrorAccumulator_mMaxError,
                                           assumed, newBits, old);
                if (old == assumed)
                    break;
                assumed = old;
            }
        }
    }
}
