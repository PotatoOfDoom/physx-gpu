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


// npArtiSolveExtTGS.slang
//
// P8.99a: the TGS articulation EXT-body solve pair on the ENGINE BLOCK
// LAYOUT — `solveExtContactBlockTGS` (solver.cuh:430-735) and
// `solveExt1DBlockTGS` (solverBlockTGS.cuh:932-1089), the two bodies the
// self-constraint solve kernel (P8.99b) and the partition solve dispatch on
// through batch.constraintType.  The Phase-1 batteries
// (npArtiSolveContactTGS / npArtiSolve1DTGS) pinned this MATH on abstracted
// flat inputs; this module is the same math over the real
// PxgTGSBlockSolver* SoA blocks (pxgSlangSolverBlocks) plus the
// PxgArticulationBlockResponse rows.
//
// ===========================================================================
// TRAPS (each carries a mutant)
// ===========================================================================
//
//  1. **The response cursor is a RUNNING pointer**: it starts at
//     responses[batch.mArticulationResponseIndex] and advances by ONE row per
//     normal contact, TWO per friction pair, ONE for the torsional tail — the
//     friction rows continue where the contacts stopped.
//  2. **The contact impulse halves are asymmetric accumulations**:
//     imp0.top -= normal*deltaF / imp0.bottom -= raXn*deltaF (both MINUS) and
//     imp1 both PLUS — and the FINAL scale is symmetric
//     (top*invMass, bottom*angDom) for both bodies, unlike PGS.
//  3. **The friction pair clamp is a combined magnitude**:
//     sqrt(t0^2 + t1^2) against staticFriction*accumulatedNormalImpulse, with
//     the ratio min(maxDyn, total)/total — and accumulatedNormalImpulse is
//     floored by minNormalForce BEFORE the friction loop.
//  4. **The torsional tail** exists when totalFrictionConstr is ODD; its clamp
//     is PxClamp against +-maxDyn*frictionScale, its velocities use ONLY the
//     angular parts, and its impulse skips the .top halves.
//  5. **The normal-force update carries NO impulse multiplier** (TGS):
//     _newForce = appliedForce + _deltaF, clamped by maxImpulse, and the
//     WRITTEN-BACK force is newForce while deltaF = newForce - appliedForce.
//  6. **The 1D row coefficients are mode-dependent**: the four .w fields are
//     coeff0..3 fed to compute1dConstraintSolverConstantsTGS
//     (DyCpuGpu1dConstraint.h:545), whose spring/acceleration-spring/bounce
//     branches this module transcribes EXACTLY — including
//     `initBias = biasScale == 0 ? 0 : initBias` after it.
//  7. **The 1D final impulse is li*invMass in .top and ai*invInertiaScale in
//     .bottom** — the 1D pair packs LINEAR in top, unlike the contact path.
//  8. **vMul and constant are recipResponse-scaled only for NON-spring rows.**
//  9. ref0/ref1 scale the responses, the invMasses AND the response rows —
//     seven sites each; the self-constraint caller passes 1.
// 10. The residual (error accumulator) divides deltaF by velMultiplier
//     exactly (the CUDA's __fdividef is fast-math; exact division here, the
//     established solverTGS spelling).

import pxgSlangCommon;
import pxgSlangSpatialMatrix;
import pxgSlangSolverBlocks;

static const float SVX_PX_MAX_F32 = 3.4028234663852885981170418348452e+38f;
static const float SVX_PX_EPS_REAL = 1.192092896e-07f;
static const uint SVX_DY_SC_FLAG_SPRING = 1u << 5;
static const uint SVX_DY_SC_FLAG_INEQUALITY = 1u << 6;
static const uint SVX_DY_SC_FLAG_ACCELERATION_SPRING = 1u << 7;
static const float SVX_DY_ARTICULATION_MIN_RESPONSE = 1e-12f;
static const uint SVX_COMPLIANT_ACCEL_SPRING = 1u << 3;  // contact header flag

public struct SvxError
{
    public float sumOfSquares;
    public int counter;
    public float maxError;
}

public void svxAccumulateError(inout SvxError e, float deltaF, float velocityMultiplier)
{
    const float r = velocityMultiplier == 0.0f ? 0.0f : deltaF / velocityMultiplier;
    e.sumOfSquares += r * r;
    ++e.counter;
    e.maxError = max(e.maxError, abs(r));
}

// computeCompliantContactCoefficientsTGS (constraintPrepShared.cuh, the
// runtime-`a` overload) — the established solverTGS spelling
void svxCompliantCoeffsTGS(uint flags, float nrdt, float unitResponse, float recipResponse,
                           float a, inout float velMultiplier, inout float biasCoefficient)
{
    const bool accelSpring = (flags & SVX_COMPLIANT_ACCEL_SPRING) != 0u;
    const float massIfAccelElseOne = accelSpring ? recipResponse : 1.0f;
    const float oneIfAccelElseR = accelSpring ? 1.0f : unitResponse;
    const float x = 1.0f / (a * oneIfAccelElseR + 1.0f);

    velMultiplier = x * a * massIfAccelElseOne;
    biasCoefficient = nrdt * x * oneIfAccelElseR;
}

// Dy::computeRecipUnitResponse (DyCpuGpu1dConstraint.h:50)
float svxRecipUnitResponse(float unitResponse, float minRowResponse)
{
    return unitResponse <= minRowResponse ? 0.0f : 1.0f / unitResponse;
}

// Dy::computeMinBiasTGS
float svxMinBiasTGS(uint flags, float maxBias)
{
    return -((flags & SVX_DY_SC_FLAG_INEQUALITY) != 0u ? SVX_PX_MAX_F32 : maxBias);
}

// Dy::computeResolvedGeometricErrorTGS
float svxResolvedGeometricErrorTGS(float3 raMotion, float3 rbMotion, float3 cLinVel0,
                                   float3 cLinVel1, float3 deltaAng0, float3 deltaAng1,
                                   float3 cAngVel0, float3 cAngVel1,
                                   float angularErrorScale, bool isSpringConstraint,
                                   float targetVel, float elapsedTime)
{
    const float deltaAng = (asmDot(cAngVel0, deltaAng0) - asmDot(cAngVel1, deltaAng1))
                         * angularErrorScale;
    const float deltaLin = asmDot(cLinVel0, raMotion) - asmDot(cLinVel1, rbMotion);
    const float motion = deltaLin + deltaAng;
    return isSpringConstraint ? motion : motion - targetVel * elapsedTime;
}

// Dy::compute1dConstraintSolverConstantsTGS (DyCpuGpu1dConstraint.h:545) —
// the reduced-coefficient overload the ext 1D rows carry
void svx1dConstantsTGS(bool isSpring, bool isAccelerationSpring, float geometricError,
                       float unitResponse, float recipUnitResponse,
                       float coeff0, float coeff1, float coeff2, float coeff3,
                       out float biasScale, out float velMultiplier, out float error,
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

// ===========================================================================
// solveExtContactBlockTGS (solver.cuh:430-735)
// ===========================================================================

public void svxSolveExtContactBlockTGS(
    Ptr<PxgBlockConstraintBatch> batch,
    inout AbSpatialVector vel0, inout AbSpatialVector vel1,
    AbSpatialVector delta0, AbSpatialVector delta1,
    uint threadIndex,
    Ptr<PxgTGSBlockSolverContactHeader> contactHeaders,
    Ptr<PxgTGSBlockSolverFrictionHeader> frictionHeaders,
    Ptr<PxgTGSBlockSolverContactPoint> contactPoints,
    Ptr<PxgTGSBlockSolverContactFriction> frictionPoints,
    Ptr<PxgArticulationBlockResponse> responses,
    float elapsedTime, float minPen,
    out AbSpatialVector impulse0, out AbSpatialVector impulse1,
    inout SvxError error, bool accumulateError,
    float ref0, float ref1)
{
    float3 linVel0 = vel0.bottom;
    float3 linVel1 = vel1.bottom;
    float3 angVel0 = vel0.top;
    float3 angVel1 = vel1.top;

    float accumulatedNormalImpulse = 0.0f;

    AbSpatialVector imp0 = asmSVec(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f));
    AbSpatialVector imp1 = imp0;

    impulse0 = imp0;
    impulse1 = imp0;

    {
        const uint batchIndex = batch[0].mConstraintBatchIndex;

        const uint numNormalConstr = contactHeaders[batchIndex].numNormalConstr[threadIndex];
        const uint totalFrictionConstr = frictionHeaders[batchIndex].numFrictionConstr[threadIndex];
        const uint numFrictionConstr = totalFrictionConstr & (~0x1u);

        const float maxPenBias = contactHeaders[batchIndex].maxPenBias[threadIndex];

        // trap 1: the running response cursor
        uint respIndex = batch[0].mArticulationResponseIndex;

        const float4 invMass0_1_angDom0_1 = contactHeaders[batchIndex].invMass0_1_angDom0_1[threadIndex];
        const float4 normal_staticFriction = contactHeaders[batchIndex].normal_staticFriction[threadIndex];

        const float restitutionXdt = contactHeaders[batchIndex].restitutionXdt[threadIndex];
        const float p8 = contactHeaders[batchIndex].p8[threadIndex];
        const float cfm = contactHeaders[batchIndex].cfm[threadIndex];
        const uint flags = contactHeaders[batchIndex].flags[threadIndex];

        const float3 normal = normal_staticFriction.xyz;
        const float staticFrictionCof = normal_staticFriction.w;

        const float3 relMotion = delta0.bottom - delta1.bottom;
        const float deltaV = asmDot(normal, relMotion);

        for (uint i = 0u; i < numNormalConstr; ++i)
        {
            const uint c = batch[0].startConstraintIndex + i;
            const uint r = respIndex;
            respIndex++;

            const float4 raXn_extraCoeff = contactPoints[c].raXn_extraCoeff[threadIndex];
            const float3 raXn = raXn_extraCoeff.xyz;
            const float compliantContactCoef = raXn_extraCoeff.w;

            const float4 rbXn_targetVelW = contactPoints[c].rbXn_targetVelW[threadIndex];
            const float appliedForce = contactPoints[c].appliedForce[threadIndex];
            const float separation = contactPoints[c].separation[threadIndex];
            const float maxImpulse = contactPoints[c].maxImpulse[threadIndex];

            const float3 rbXn = rbXn_targetVelW.xyz;
            const float targetVel = rbXn_targetVelW.w;

            float biasCoefficient = contactPoints[c].biasCoefficient[threadIndex];

            const float resp0 = ref0 * contactPoints[c].resp0[threadIndex];
            const float resp1 = ref1 * contactPoints[c].resp1[threadIndex];

            const float unitResponse = resp0 + resp1;
            const float recipResponse = (unitResponse > 0.0f) ? (1.0f / (unitResponse + cfm))
                                                              : 0.0f;

            float velMultiplier = recipResponse;

            if (restitutionXdt < 0.0f)
            {
                svxCompliantCoeffsTGS(flags, restitutionXdt, unitResponse, recipResponse,
                                      compliantContactCoef, velMultiplier, biasCoefficient);
            }

            const float v0 = asmDot(angVel0, raXn) + asmDot(linVel0, normal);
            const float v1 = asmDot(angVel1, rbXn) + asmDot(linVel1, normal);
            const float normalVel = v0 - v1;

            const float deltaBias = deltaV + asmDot(delta0.top, raXn) - asmDot(delta1.top, rbXn)
                                  - targetVel * elapsedTime;

            const float sep = max(minPen, separation + deltaBias);

            const float biased = min(-maxPenBias, biasCoefficient * sep);
            const float tVelBias = recipResponse * biased;

            // trap 5: NO impulse multiplier in TGS
            const float tempDeltaF = tVelBias - (normalVel - targetVel) * velMultiplier;
            const float _deltaF = max(tempDeltaF, -appliedForce);
            const float _newForce = appliedForce + _deltaF;
            const float newForce = min(_newForce, maxImpulse);
            const float deltaF = newForce - appliedForce;

            const float3 deltaRALin = ref0 * float3(responses[r].deltaRALin_x[threadIndex],
                                                    responses[r].deltaRALin_y[threadIndex],
                                                    responses[r].deltaRALin_z[threadIndex]);
            const float3 deltaRAAng = ref0 * float3(responses[r].deltaRAAng_x[threadIndex],
                                                    responses[r].deltaRAAng_y[threadIndex],
                                                    responses[r].deltaRAAng_z[threadIndex]);
            const float3 deltaRBLin = ref1 * float3(responses[r].deltaRBLin_x[threadIndex],
                                                    responses[r].deltaRBLin_y[threadIndex],
                                                    responses[r].deltaRBLin_z[threadIndex]);
            const float3 deltaRBAng = ref1 * float3(responses[r].deltaRBAng_x[threadIndex],
                                                    responses[r].deltaRBAng_y[threadIndex],
                                                    responses[r].deltaRBAng_z[threadIndex]);
            linVel0 += deltaRALin * deltaF;
            linVel1 += deltaRBLin * deltaF;
            angVel0 += deltaRAAng * deltaF;
            angVel1 += deltaRBAng * deltaF;

            // trap 2: minus into imp0, plus into imp1
            imp0.top -= normal * deltaF;
            imp0.bottom -= raXn * deltaF;
            imp1.top += normal * deltaF;
            imp1.bottom += rbXn * deltaF;

            if (accumulateError)
                svxAccumulateError(error, deltaF, velMultiplier);

            contactPoints[c].appliedForce[threadIndex] = newForce;

            accumulatedNormalImpulse = accumulatedNormalImpulse + newForce;
        }

        // trap 3: the floor happens BEFORE the friction loop
        accumulatedNormalImpulse = max(accumulatedNormalImpulse,
                                       contactHeaders[batchIndex].minNormalForce[threadIndex]);

        if (numFrictionConstr != 0u)
        {
            const float biasCoefficient = frictionHeaders[batchIndex].biasCoefficient[threadIndex];
            const float dynamicFrictionCof = frictionHeaders[batchIndex].dynamicFriction[threadIndex];
            const float maxFrictionImpulse = staticFrictionCof * accumulatedNormalImpulse;
            const float maxDynFrictionImpulse = dynamicFrictionCof * accumulatedNormalImpulse;

            uint broken = 0u;

            const float4 frictionNormal0 = frictionHeaders[batchIndex].frictionNormals0[threadIndex];
            const float4 frictionNormal1 = frictionHeaders[batchIndex].frictionNormals1[threadIndex];
            const float3 normal0 = frictionNormal0.xyz;
            const float3 normal1 = frictionNormal1.xyz;

            const float deltaMotion0 = asmDot(normal0, relMotion);
            const float deltaMotion1 = asmDot(normal1, relMotion);

            for (uint i = 0u; i < numFrictionConstr; i += 2u)
            {
                const uint f0 = batch[0].startFrictionIndex + i;
                const uint f1 = f0 + 1u;
                const uint r0 = respIndex;
                respIndex++;
                const uint r1 = respIndex;
                respIndex++;

                const float4 raXn_error0 = frictionPoints[f0].raXn_error[threadIndex];
                const float4 rbXn_targetVelW0 = frictionPoints[f0].rbXn_targetVelW[threadIndex];
                const float initialError0 = raXn_error0.w;
                const float appliedForce0 = frictionPoints[f0].appliedForce[threadIndex];
                const float targetVel0 = rbXn_targetVelW0.w;

                const float4 raXn_error1 = frictionPoints[f1].raXn_error[threadIndex];
                const float4 rbXn_targetVelW1 = frictionPoints[f1].rbXn_targetVelW[threadIndex];
                const float initialError1 = raXn_error1.w;
                const float appliedForce1 = frictionPoints[f1].appliedForce[threadIndex];
                const float targetVel1 = rbXn_targetVelW1.w;

                const float3 raXn0 = raXn_error0.xyz;
                const float3 rbXn0 = rbXn_targetVelW0.xyz;
                const float3 raXn1 = raXn_error1.xyz;
                const float3 rbXn1 = rbXn_targetVelW1.xyz;

                const float resp0_0 = ref0 * frictionPoints[f0].resp0[threadIndex];
                const float resp0_1 = ref1 * frictionPoints[f0].resp1[threadIndex];
                const float resp0 = resp0_0 + resp0_1;
                const float velMultiplier0 = (resp0 > SVX_PX_EPS_REAL) ? (p8 / (resp0 + cfm))
                                                                       : 0.0f;

                const float resp1_0 = ref0 * frictionPoints[f1].resp0[threadIndex];
                const float resp1_1 = ref1 * frictionPoints[f1].resp1[threadIndex];
                const float resp1 = resp1_0 + resp1_1;
                const float velMultiplier1 = (resp1 > SVX_PX_EPS_REAL) ? (p8 / (resp1 + cfm))
                                                                       : 0.0f;

                const float v00 = asmDot(angVel0, raXn0) + asmDot(linVel0, normal0);
                const float v10 = asmDot(angVel1, rbXn0) + asmDot(linVel1, normal0);
                const float normalVel0 = v00 - v10;

                const float v01 = asmDot(angVel0, raXn1) + asmDot(linVel0, normal1);
                const float v11 = asmDot(angVel1, rbXn1) + asmDot(linVel1, normal1);
                const float normalVel1 = v01 - v11;

                const float error0 = initialError0 - targetVel0 * elapsedTime
                                   + (asmDot(raXn0, delta0.top) - asmDot(rbXn0, delta1.top)
                                      + deltaMotion0);
                const float bias0 = error0 * biasCoefficient;
                const float tmp10 = appliedForce0 - (bias0 - targetVel0) * velMultiplier0;
                const float totalImpulse0 = tmp10 - normalVel0 * velMultiplier0;

                const float error1 = initialError1 - targetVel1 * elapsedTime
                                   + (asmDot(raXn1, delta0.top) - asmDot(rbXn1, delta1.top)
                                      + deltaMotion1);
                const float bias1 = error1 * biasCoefficient;
                const float tmp11 = appliedForce1 - (bias1 - targetVel1) * velMultiplier1;
                const float totalImpulse1 = tmp11 - normalVel1 * velMultiplier1;

                const float totalImpulse = sqrt(totalImpulse0 * totalImpulse0
                                                + totalImpulse1 * totalImpulse1);

                const bool doClamp = totalImpulse > maxFrictionImpulse;

                const float ratio = doClamp
                    ? min(maxDynFrictionImpulse, totalImpulse) / totalImpulse : 1.0f;

                const float newAppliedForce0 = totalImpulse0 * ratio;
                const float newAppliedForce1 = totalImpulse1 * ratio;

                float deltaF0 = newAppliedForce0 - appliedForce0;
                float deltaF1 = newAppliedForce1 - appliedForce1;

                if (accumulateError)
                {
                    svxAccumulateError(error, deltaF0, velMultiplier0);
                    svxAccumulateError(error, deltaF1, velMultiplier1);
                }

                linVel0 += ref0 * float3(responses[r0].deltaRALin_x[threadIndex],
                                         responses[r0].deltaRALin_y[threadIndex],
                                         responses[r0].deltaRALin_z[threadIndex]) * deltaF0;
                linVel1 += ref1 * float3(responses[r0].deltaRBLin_x[threadIndex],
                                         responses[r0].deltaRBLin_y[threadIndex],
                                         responses[r0].deltaRBLin_z[threadIndex]) * deltaF0;
                angVel0 += ref0 * float3(responses[r0].deltaRAAng_x[threadIndex],
                                         responses[r0].deltaRAAng_y[threadIndex],
                                         responses[r0].deltaRAAng_z[threadIndex]) * deltaF0;
                angVel1 += ref1 * float3(responses[r0].deltaRBAng_x[threadIndex],
                                         responses[r0].deltaRBAng_y[threadIndex],
                                         responses[r0].deltaRBAng_z[threadIndex]) * deltaF0;

                linVel0 += ref0 * float3(responses[r1].deltaRALin_x[threadIndex],
                                         responses[r1].deltaRALin_y[threadIndex],
                                         responses[r1].deltaRALin_z[threadIndex]) * deltaF1;
                linVel1 += ref1 * float3(responses[r1].deltaRBLin_x[threadIndex],
                                         responses[r1].deltaRBLin_y[threadIndex],
                                         responses[r1].deltaRBLin_z[threadIndex]) * deltaF1;
                angVel0 += ref0 * float3(responses[r1].deltaRAAng_x[threadIndex],
                                         responses[r1].deltaRAAng_y[threadIndex],
                                         responses[r1].deltaRAAng_z[threadIndex]) * deltaF1;
                angVel1 += ref1 * float3(responses[r1].deltaRBAng_x[threadIndex],
                                         responses[r1].deltaRBAng_y[threadIndex],
                                         responses[r1].deltaRBAng_z[threadIndex]) * deltaF1;

                frictionPoints[f0].appliedForce[threadIndex] = newAppliedForce0;
                frictionPoints[f1].appliedForce[threadIndex] = newAppliedForce1;
                broken = broken | (doClamp ? 1u : 0u);

                imp0.top -= normal0 * deltaF0;
                imp0.bottom -= raXn0 * deltaF0;
                imp1.top += normal0 * deltaF0;
                imp1.bottom += rbXn0 * deltaF0;

                imp0.top -= normal1 * deltaF1;
                imp0.bottom -= raXn1 * deltaF1;
                imp1.top += normal1 * deltaF1;
                imp1.bottom += rbXn1 * deltaF1;
            }

            // trap 4: the torsional tail on an odd friction count
            if (numFrictionConstr < totalFrictionConstr)
            {
                const float frictionScale = frictionHeaders[batchIndex].torsionalFrictionScale[threadIndex];

                const uint f0 = batch[0].startFrictionIndex + numFrictionConstr;
                const uint r0 = respIndex;
                respIndex++;

                const float4 raXn_error0 = frictionPoints[f0].raXn_error[threadIndex];
                const float4 rbXn_targetVelW0 = frictionPoints[f0].rbXn_targetVelW[threadIndex];
                const float appliedForce0 = frictionPoints[f0].appliedForce[threadIndex];
                const float targetVel0 = rbXn_targetVelW0.w;

                const float3 raXn0 = raXn_error0.xyz;
                const float3 rbXn0 = rbXn_targetVelW0.xyz;

                const float resp0_0 = ref0 * frictionPoints[f0].resp0[threadIndex];
                const float resp0_1 = ref1 * frictionPoints[f0].resp1[threadIndex];
                const float resp0 = resp0_0 + resp0_1;
                const float velMultiplier0 = (resp0 > 0.0f) ? (p8 / (resp0 + cfm)) : 0.0f;

                const float v00 = asmDot(angVel0, raXn0);
                const float v10 = asmDot(angVel1, rbXn0);
                const float normalVel0 = v00 - v10;

                const float tmp10 = appliedForce0 - (-targetVel0) * velMultiplier0;
                const float totalImpulse = tmp10 - normalVel0 * velMultiplier0;

                const bool doClamp = abs(totalImpulse) > (maxFrictionImpulse * frictionScale);

                // P9.86: PxClamp(v,lo,hi) = PxMin(hi, PxMax(lo, v)) -- selectMin/selectMax (solver.cuh:704).
                // A zero normal impulse (or torsional scale) makes the bounds exactly (-0, +0);
                // FClamp's fmin(-0, +0) tie is driver-defined, the CUDA's selects keep -0 for v < 0.
                const float lo = -maxDynFrictionImpulse * frictionScale;
                const float hi = maxDynFrictionImpulse * frictionScale;
                const float pxMaxLoV = (lo > totalImpulse) ? lo : totalImpulse;   // selectMax(lo, v)
                const float totalClamped = (hi < pxMaxLoV) ? hi : pxMaxLoV;      // selectMin(hi, m)

                const float newAppliedForce = doClamp ? totalClamped : totalImpulse;

                const float deltaF0 = newAppliedForce - appliedForce0;
                if (accumulateError)
                    svxAccumulateError(error, deltaF0, velMultiplier0);

                linVel0 += ref0 * float3(responses[r0].deltaRALin_x[threadIndex],
                                         responses[r0].deltaRALin_y[threadIndex],
                                         responses[r0].deltaRALin_z[threadIndex]) * deltaF0;
                linVel1 += ref1 * float3(responses[r0].deltaRBLin_x[threadIndex],
                                         responses[r0].deltaRBLin_y[threadIndex],
                                         responses[r0].deltaRBLin_z[threadIndex]) * deltaF0;
                angVel0 += ref0 * float3(responses[r0].deltaRAAng_x[threadIndex],
                                         responses[r0].deltaRAAng_y[threadIndex],
                                         responses[r0].deltaRAAng_z[threadIndex]) * deltaF0;
                angVel1 += ref1 * float3(responses[r0].deltaRBAng_x[threadIndex],
                                         responses[r0].deltaRBAng_y[threadIndex],
                                         responses[r0].deltaRBAng_z[threadIndex]) * deltaF0;

                frictionPoints[f0].appliedForce[threadIndex] = newAppliedForce;
                broken = broken | (doClamp ? 1u : 0u);

                imp0.bottom -= raXn0 * deltaF0;
                imp1.bottom += rbXn0 * deltaF0;
            }

            frictionHeaders[batchIndex].broken[threadIndex] = broken;
        }

        vel0.bottom = linVel0;
        vel0.top = angVel0;
        vel1.bottom = linVel1;
        vel1.top = angVel1;

        // trap 2: the SYMMETRIC final scale (UnAlignedSpatialVector::scale
        // scales top by the first factor, bottom by the second)
        impulse0 = asmSVec(asetPlusZero3(imp0.top) * (invMass0_1_angDom0_1.x * ref0),
                           asetPlusZero3(imp0.bottom) * (invMass0_1_angDom0_1.z * ref0));
        impulse1 = asmSVec(asetPlusZero3(imp1.top) * (invMass0_1_angDom0_1.y * ref1),
                           asetPlusZero3(imp1.bottom) * (invMass0_1_angDom0_1.w * ref1));
    }
}

// ===========================================================================
// solveExt1DBlockTGS (solverBlockTGS.cuh:932-1089)
// ===========================================================================

public void svxSolveExt1DBlockTGS(
    Ptr<PxgBlockConstraintBatch> batch,
    inout AbSpatialVector vel0, inout AbSpatialVector vel1,
    AbSpatialVector motion0, AbSpatialVector motion1,
    uint threadIndex,
    Ptr<PxgTGSBlockSolverConstraint1DHeader> headers,
    Ptr<PxgTGSBlockSolverConstraint1DCon> rowsCon,
    Ptr<PxgArticulationBlockResponse> artiResponse,
    float4 deltaQ0, float4 deltaQ1,
    float elapsedTime,
    out AbSpatialVector impulse0, out AbSpatialVector impulse1,
    bool residualReportingEnabled,
    float ref0, float ref1)
{
    const uint batchIndex = batch[0].mConstraintBatchIndex;
    const uint startRow = batch[0].startConstraintIndex;
    const uint startResp = batch[0].mArticulationResponseIndex;

    float3 linVel0 = vel0.bottom;
    float3 linVel1 = vel1.bottom;
    float3 angVel0 = vel0.top;
    float3 angVel1 = vel1.top;

    const float4 raInvMass0 = headers[batchIndex].rAWorld_invMass0D0[threadIndex];
    const float4 rbInvMass1 = headers[batchIndex].rBWorld_invMass1D1[threadIndex];

    const float invMass0 = ref0 * raInvMass0.w;
    const float invMass1 = ref1 * rbInvMass1.w;

    const float invInertiaScale0 = ref0 * headers[batchIndex].invInertiaScale0[threadIndex];
    const float invInertiaScale1 = ref1 * headers[batchIndex].invInertiaScale1[threadIndex];

    const float3 raPrev = raInvMass0.xyz;
    const float3 rbPrev = rbInvMass1.xyz;

    const float3 ra = quatRotate(deltaQ0, raPrev);
    const float3 rb = quatRotate(deltaQ1, rbPrev);

    const float3 raMotion = (ra + motion0.bottom) - raPrev;
    const float3 rbMotion = (rb + motion1.bottom) - rbPrev;

    // uchar4 packed in a uint: .x is the low byte
    const uint rowCount = headers[batchIndex].rowCounts_breakable_orthoAxisCount[threadIndex]
                        & 0xFFu;

    const float cfmHdr = headers[batchIndex].cfm[threadIndex];

    float3 li0 = float3(0.0f, 0.0f, 0.0f);
    float3 li1 = float3(0.0f, 0.0f, 0.0f);
    float3 ai0 = float3(0.0f, 0.0f, 0.0f);
    float3 ai1 = float3(0.0f, 0.0f, 0.0f);

    for (uint i = 0u; i < rowCount; ++i)
    {
        const uint r = startRow + i;
        const uint rr = startResp + i;

        const float4 _clinVel0_coeff0W = rowsCon[r].lin0XYZ_initBiasOrCoeff0[threadIndex];
        const float4 _clinVel1_coeff1W = rowsCon[r].lin1XYZ_biasScaleOrCoeff1[threadIndex];
        const float4 _cangVel0_coeff2W = rowsCon[r].ang0XYZ_velMultiplierOrCoeff2[threadIndex];
        const float4 _cangVel1_coeff3W = rowsCon[r].ang1XYZ_velTargetOrCoeff3[threadIndex];

        const float3 clinVel0 = _clinVel0_coeff0W.xyz;
        const float3 clinVel1 = _clinVel1_coeff1W.xyz;
        const float3 cangVel0 = _cangVel0_coeff2W.xyz;
        const float3 cangVel1 = _cangVel1_coeff3W.xyz;

        const uint flags = rowsCon[r].flags[threadIndex];
        const bool isSpringConstraint = (flags & SVX_DY_SC_FLAG_SPRING) != 0u;

        const float resp0 = ref0 * rowsCon[r].resp0[threadIndex];
        const float resp1 = ref1 * rowsCon[r].resp1[threadIndex];

        float unitResponse = resp0 + resp1;
        unitResponse += cfmHdr;

        const float recipResponse = svxRecipUnitResponse(unitResponse,
                                                         SVX_DY_ARTICULATION_MIN_RESPONSE);

        const bool isAccelerationSpring = (flags & SVX_DY_SC_FLAG_ACCELERATION_SPRING) != 0u;

        const float coeff0 = _clinVel0_coeff0W.w;
        const float coeff1 = _clinVel1_coeff1W.w;
        const float coeff2 = _cangVel0_coeff2W.w;
        const float coeff3 = _cangVel1_coeff3W.w;
        const float geometricError = rowsCon[r].geometricError[threadIndex];

        float biasScale, velMultiplier, initBias, targetVel;
        svx1dConstantsTGS(isSpringConstraint, isAccelerationSpring, geometricError,
                          unitResponse, recipResponse, coeff0, coeff1, coeff2, coeff3,
                          biasScale, velMultiplier, initBias, targetVel);

        // trap 6: velocity iterations force initBias back to zero
        initBias = (biasScale == 0.0f) ? 0.0f : initBias;

        const float errorDelta = svxResolvedGeometricErrorTGS(
            raMotion, rbMotion, clinVel0, clinVel1, motion0.top, motion1.top,
            cangVel0, cangVel1, rowsCon[r].angularErrorScale[threadIndex],
            isSpringConstraint, targetVel, elapsedTime);

        const float maxBias = rowsCon[r].maxBias[threadIndex];

        // trap 8
        const float vMul = isSpringConstraint ? velMultiplier : recipResponse * velMultiplier;
        const float appliedForce = rowsCon[r].appliedForce[threadIndex];

        const float unclampedBias = initBias + errorDelta * biasScale;
        const float minBias = svxMinBiasTGS(flags, maxBias);
        const float bias = clamp(unclampedBias, minBias, maxBias);

        const float constant = isSpringConstraint ? (bias + targetVel)
                                                  : recipResponse * (bias + targetVel);

        const float maxImpulse = rowsCon[r].maxImpulse[threadIndex];
        const float minImpulse = rowsCon[r].minImpulse[threadIndex];

        const float v0 = asmDot(linVel0, clinVel0) + asmDot(angVel0, cangVel0);
        const float v1 = asmDot(linVel1, clinVel1) + asmDot(angVel1, cangVel1);

        const float normalVel = v0 - v1;

        const float unclampedForce = appliedForce + (vMul * normalVel + constant);
        // P9.86: the CUDA's fminf/fmaxf (solverBlockTGS.cuh:1060) are PTX min/max, which order
        // the zeros (+0 > -0): fmaxf(+0, -0) = +0 and fminf(-0, +0) = -0 in either operand order.
        // llvmpipe's FMax/FMin are x86 maxps/minps and return the SECOND operand on a zero tie
        // (e.g. a drive with forceLimit -0.0f: minImpulse +0, maxImpulse -0).  Only a pair of
        // zeros is re-spelled, as a select; every other input, inf and NaN included, still goes
        // through max/min unchanged.
        const float loForce = (minImpulse == 0.0f && unclampedForce == 0.0f)
                            ? ((asuint(minImpulse) == 0u) ? minImpulse : unclampedForce)
                            : max(minImpulse, unclampedForce);
        const float clampedForce = (maxImpulse == 0.0f && loForce == 0.0f)
                                 ? ((asuint(maxImpulse) != 0u) ? maxImpulse : loForce)
                                 : min(maxImpulse, loForce);
        const float deltaF = clampedForce - appliedForce;

        rowsCon[r].appliedForce[threadIndex] = clampedForce;
        if (residualReportingEnabled)
        {
            const float res = vMul == 0.0f ? 0.0f : deltaF / vMul;
            rowsCon[r].residual[threadIndex] = res;
        }

        li0 = clinVel0 * deltaF + li0;
        ai0 = cangVel0 * deltaF + ai0;
        li1 = clinVel1 * deltaF + li1;
        ai1 = cangVel1 * deltaF + ai1;

        const float3 linVa = ref0 * float3(artiResponse[rr].deltaRALin_x[threadIndex],
                                           artiResponse[rr].deltaRALin_y[threadIndex],
                                           artiResponse[rr].deltaRALin_z[threadIndex]);
        const float3 angVa = ref0 * float3(artiResponse[rr].deltaRAAng_x[threadIndex],
                                           artiResponse[rr].deltaRAAng_y[threadIndex],
                                           artiResponse[rr].deltaRAAng_z[threadIndex]);
        const float3 linVb = ref1 * float3(artiResponse[rr].deltaRBLin_x[threadIndex],
                                           artiResponse[rr].deltaRBLin_y[threadIndex],
                                           artiResponse[rr].deltaRBLin_z[threadIndex]);
        const float3 angVb = ref1 * float3(artiResponse[rr].deltaRBAng_x[threadIndex],
                                           artiResponse[rr].deltaRBAng_y[threadIndex],
                                           artiResponse[rr].deltaRBAng_z[threadIndex]);

        linVel0 = linVa * deltaF + linVel0;
        angVel0 = angVa * deltaF + angVel0;
        linVel1 = linVb * deltaF + linVel1;
        angVel1 = angVb * deltaF + angVel1;
    }

    vel0.top = angVel0;
    vel0.bottom = linVel0;
    vel1.top = angVel1;
    vel1.bottom = linVel1;

    // trap 7: LINEAR in top, scaled by invMass; angular in bottom by inertia scale
    impulse0 = asmSVec(li0 * invMass0, ai0 * invInertiaScale0);
    impulse1 = asmSVec(li1 * invMass1, ai1 * invInertiaScale1);
}

// ---------------------------------------------------------------------------
// fixture entry points: one batch, 32 lanes = 32 independent constraint slots
// ---------------------------------------------------------------------------

struct SvxParams
{
    uint64_t batch;           // @0   PxgBlockConstraintBatch*
    uint64_t contactHeaders;  // @8   (1D: 1DHeaders)
    uint64_t frictionHeaders; // @16  (1D: rowsCon)
    uint64_t contactPoints;   // @24  (1D: unused)
    uint64_t frictionPoints;  // @32  (1D: unused)
    uint64_t responses;       // @40
    uint64_t io;              // @48  in/out: vel0, vel1, delta0/motion0, delta1/motion1,
                              //      deltaQ0, deltaQ1 per lane; then out impulses + error
    float elapsedTime;        // @56
    float minPen;             // @60
    float ref0;               // @64
    float ref1;               // @68
    uint nbLanes;             // @72
    uint accumulateError;     // @76
}

[[vk::push_constant]]
ConstantBuffer<SvxParams> gSvx;

// io layout per lane (floats): vel0 top/bottom (6), vel1 (6), d0 (6), d1 (6),
// dq0 (4), dq1 (4) = 32 in; out: vel0 (6), vel1 (6), imp0 (6), imp1 (6),
// err (3) = 27, padded to 32.  in at lane*64, out at (32*64) + lane*64... the
// fixture packs in the first 32*32 floats and outputs in the second.
static const uint SVX_IO_IN_STRIDE = 32u;
static const uint SVX_IO_OUT_BASE = 32u * 32u;

void svxLoadIo(uint lane, out AbSpatialVector vel0, out AbSpatialVector vel1,
               out AbSpatialVector d0, out AbSpatialVector d1,
               out float4 dq0, out float4 dq1)
{
    Ptr<float> f = Ptr<float>(gSvx.io + uint64_t(lane) * SVX_IO_IN_STRIDE * 4u);
    vel0 = asmSVec(float3(f[0], f[1], f[2]), float3(f[3], f[4], f[5]));
    vel1 = asmSVec(float3(f[6], f[7], f[8]), float3(f[9], f[10], f[11]));
    d0 = asmSVec(float3(f[12], f[13], f[14]), float3(f[15], f[16], f[17]));
    d1 = asmSVec(float3(f[18], f[19], f[20]), float3(f[21], f[22], f[23]));
    dq0 = float4(f[24], f[25], f[26], f[27]);
    dq1 = float4(f[28], f[29], f[30], f[31]);
}

void svxStoreIo(uint lane, AbSpatialVector vel0, AbSpatialVector vel1,
                AbSpatialVector imp0, AbSpatialVector imp1, SvxError err)
{
    Ptr<float> f = Ptr<float>(gSvx.io + uint64_t(SVX_IO_OUT_BASE + lane * SVX_IO_IN_STRIDE) * 4u);
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
float3 asetPlusZero3(float3 v)
{
    return float3((v.x == 0.0f) ? 0.0f : v.x, (v.y == 0.0f) ? 0.0f : v.y, (v.z == 0.0f) ? 0.0f : v.z);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void svxContactTest(uint3 groupThreadId : SV_GroupThreadID)
{
    const uint lane = groupThreadId.x;
    if (lane >= gSvx.nbLanes)
        return;

    AbSpatialVector vel0, vel1, d0, d1;
    float4 dq0, dq1;
    svxLoadIo(lane, vel0, vel1, d0, d1, dq0, dq1);

    AbSpatialVector imp0, imp1;
    SvxError err = { 0.0f, 0, 0.0f };
    svxSolveExtContactBlockTGS(Ptr<PxgBlockConstraintBatch>(gSvx.batch), vel0, vel1, d0, d1,
                               lane,
                               Ptr<PxgTGSBlockSolverContactHeader>(gSvx.contactHeaders),
                               Ptr<PxgTGSBlockSolverFrictionHeader>(gSvx.frictionHeaders),
                               Ptr<PxgTGSBlockSolverContactPoint>(gSvx.contactPoints),
                               Ptr<PxgTGSBlockSolverContactFriction>(gSvx.frictionPoints),
                               Ptr<PxgArticulationBlockResponse>(gSvx.responses),
                               gSvx.elapsedTime, gSvx.minPen, imp0, imp1,
                               err, gSvx.accumulateError != 0u, gSvx.ref0, gSvx.ref1);

    svxStoreIo(lane, vel0, vel1, imp0, imp1, err);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void svx1DTest(uint3 groupThreadId : SV_GroupThreadID)
{
    const uint lane = groupThreadId.x;
    if (lane >= gSvx.nbLanes)
        return;

    AbSpatialVector vel0, vel1, m0, m1;
    float4 dq0, dq1;
    svxLoadIo(lane, vel0, vel1, m0, m1, dq0, dq1);

    AbSpatialVector imp0, imp1;
    SvxError err = { 0.0f, 0, 0.0f };
    svxSolveExt1DBlockTGS(Ptr<PxgBlockConstraintBatch>(gSvx.batch), vel0, vel1, m0, m1,
                          lane,
                          Ptr<PxgTGSBlockSolverConstraint1DHeader>(gSvx.contactHeaders),
                          Ptr<PxgTGSBlockSolverConstraint1DCon>(gSvx.frictionHeaders),
                          Ptr<PxgArticulationBlockResponse>(gSvx.responses),
                          dq0, dq1, gSvx.elapsedTime, imp0, imp1,
                          gSvx.accumulateError != 0u, gSvx.ref0, gSvx.ref1);

    svxStoreIo(lane, vel0, vel1, imp0, imp1, err);
}
