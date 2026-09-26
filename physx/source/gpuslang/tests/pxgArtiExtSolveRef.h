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

// pxgArtiExtSolveRef.h
//
// The TGS ext-body solve mirrors (P8.99a): solveExtContactBlockTGS
// (solver.cuh:430-735) and solveExt1DBlockTGS (solverBlockTGS.cuh:932-1089)
// over the ENGINE's own block structs, plus the three __device__-only scalar
// helpers they need transcribed (the compliant-coefficient reduced overload,
// computeResolvedGeometricErrorTGS, computeMinBiasTGS) and the
// PxgErrorAccumulator mirror.  Hoisted VERBATIM from testArtiSolveExtTGS.cpp
// (static -> inline is the only edit) so the self-constraint solve fixture
// (P8.99b) is a USER, not a second transcription.
//
// The 1D solver constants come from the ENGINE's host-callable
// Dy::compute1dConstraintSolverConstantsTGS / Dy::computeRecipUnitResponse;
// only the structure around them is transcribed.
//
// Include-order caveat (measured in P8.99a): DyCpuGpu1dConstraint.h needs
// DySolverConstraintTypes.h and DyCpuGpuArticulation.h included BEFORE it.

#ifndef PXG_ARTI_EXT_SOLVE_REF_H
#define PXG_ARTI_EXT_SOLVE_REF_H

#include "PxgCudaShim.h"

#include "foundation/PxQuat.h"
#include "foundation/PxVec3.h"
#include "CmSpatialVector.h"
#include "DySolverConstraintTypes.h"
#include "DyCpuGpuArticulation.h"
#include "DyCpuGpu1dConstraint.h"
#include "PxgSolverConstraintDesc.h"
#include "PxgConstraintBlock.h"
#include "PxgSolverConstraintBlock1D.h"

#include <cstdint>
#include <cmath>

namespace pxgtest
{
using namespace physx;
using physx::Cm::UnAlignedSpatialVector;

const float SVX_EPS_REAL = 1.192092896e-07f;
const uint32_t DY_FLAG_SPRING = 1u << 5, DY_FLAG_INEQ = 1u << 6, DY_FLAG_ACCEL = 1u << 7;
const uint32_t COMPLIANT_ACCEL = 1u << 3;

// the __device__-only reduced compliant-coefficient overload, transcribed
inline void refCompliantCoeffsTGS(uint32_t flags, float nrdt, float unitResponse,
                                  float recipResponse, float a,
                                  float& velMultiplier, float& biasCoefficient)
{
    const bool accelSpring = (flags & COMPLIANT_ACCEL) != 0;
    const float massIfAccelElseOne = accelSpring ? recipResponse : 1.0f;
    const float oneIfAccelElseR = accelSpring ? 1.0f : unitResponse;
    const float x = 1.0f / (a * oneIfAccelElseR + 1.0f);
    velMultiplier = x * a * massIfAccelElseOne;
    biasCoefficient = nrdt * x * oneIfAccelElseR;
}

// the scalar computeResolvedGeometricErrorTGS / computeMinBiasTGS overloads are
// __device__-only; transcribed here (the SIMD twins are host-callable but not
// bit-comparable)
inline float refResolvedGeometricErrorTGS(const PxVec3& deltaLin0, const PxVec3& deltaLin1,
                                          const PxVec3& cLinVel0, const PxVec3& cLinVel1,
                                          const PxVec3& deltaAng0, const PxVec3& deltaAng1,
                                          const PxVec3& cAngVel0, const PxVec3& cAngVel1,
                                          float angularErrorScale, bool isSpringConstraint,
                                          float targetVel, float elapsedTime)
{
    const float deltaAng = (cAngVel0.dot(deltaAng0) - cAngVel1.dot(deltaAng1)) * angularErrorScale;
    const float deltaLin = cLinVel0.dot(deltaLin0) - cLinVel1.dot(deltaLin1);
    const float motion = deltaLin + deltaAng;
    return isSpringConstraint ? motion : motion - targetVel * elapsedTime;
}

inline float refMinBiasTGS(uint32_t flags, float maxBias)
{
    return -((flags & DY_FLAG_INEQ) != 0 ? PX_MAX_F32 : maxBias);
}

// P9.86c: the CUDA's fminf/fmaxf -- IEEE minNum/maxNum (NaN-ignoring) with NVIDIA's zero rule:
// the zeros are ORDERED (-0 < +0) whatever the operand order, fminf(+0, -0) = fminf(-0, +0) = -0
// and fmaxf(+0, -0) = fmaxf(-0, +0) = +0.  This is also llvmpipe's GLSL FMin/FMax under the
// SignedZeroInfNanPreserve stamp (P9.86 F4).  The host's glibc fminf/fmaxf are NOT: they keep the
// FIRST operand on a +-0 tie (measured with glibc 2.39: fmin(+0, -0) = +0, fmax(-0, +0) = -0).
inline float refCudaFminf(float a, float b)
{
    if (std::isnan(a)) return b;
    if (std::isnan(b)) return a;
    if (a == b) return std::signbit(a) ? a : b;
    return a < b ? a : b;
}
inline float refCudaFmaxf(float a, float b)
{
    if (std::isnan(a)) return b;
    if (std::isnan(b)) return a;
    if (a == b) return std::signbit(a) ? b : a;
    return a > b ? a : b;
}

// P9.86c twins of the npArtiSolveExtTGS audit fixes (0 = the CUDA semantics, always the default).
// gExtTorsClampVariant (the torsional PxClamp, solver.cuh:704): 1 = the reverted Slang clamp(),
// GLSL FClamp = FMin(FMax(v, lo), hi) with llvmpipe's zero-ordered FMin/FMax.
inline int gExtTorsClampVariant = 0;
// gExt1DClampVariant (the 1D fminf(maxImpulse, fmaxf(minImpulse, u)), solverBlockTGS.cuh:1060):
// 1 = PxMin/PxMax selects (a zero tie keeps the SECOND operand); 2 = the host's glibc
// std::fmin/std::fmax (the FIRST operand; this oracle's spelling before P9.86c).  Neither is the
// reverted FMin/FMax: under the stamp that one orders the zeros exactly like the CUDA.
inline int gExt1DClampVariant = 0;

struct RefError
{
    float sumOfSquares = 0.f;
    int counter = 0;
    float maxError = 0.f;
    void add(float deltaF, float velMul)
    {
        const float r = velMul == 0.f ? 0.f : deltaF / velMul;
        sumOfSquares += r * r;
        ++counter;
        maxError = PxMax(maxError, PxAbs(r));
    }
};

// solveExtContactBlockTGS (solver.cuh:430-735), transcribed over the engine structs
inline void refSolveExtContactTGS(const PxgBlockConstraintBatch& batch,
                                  UnAlignedSpatialVector& vel0, UnAlignedSpatialVector& vel1,
                                  const UnAlignedSpatialVector& delta0,
                                  const UnAlignedSpatialVector& delta1, PxU32 ti,
                                  PxgTGSBlockSolverContactHeader* contactHeaders,
                                  PxgTGSBlockSolverFrictionHeader* frictionHeaders,
                                  PxgTGSBlockSolverContactPoint* contactPoints,
                                  PxgTGSBlockSolverContactFriction* frictionPoints,
                                  PxgArticulationBlockResponse* responses,
                                  float elapsedTime, float minPen,
                                  UnAlignedSpatialVector& impulse0,
                                  UnAlignedSpatialVector& impulse1,
                                  RefError* error, float ref0, float ref1)
{
    PxVec3 linVel0 = vel0.bottom, linVel1 = vel1.bottom;
    PxVec3 angVel0 = vel0.top, angVel1 = vel1.top;

    float accumulatedNormalImpulse = 0.f;
    UnAlignedSpatialVector imp0(PxVec3(0.f), PxVec3(0.f)), imp1(PxVec3(0.f), PxVec3(0.f));

    PxgTGSBlockSolverContactHeader* contactHeader = &contactHeaders[batch.mConstraintBatchIndex];
    PxgTGSBlockSolverFrictionHeader* frictionHeader = &frictionHeaders[batch.mConstraintBatchIndex];

    const PxU32 numNormalConstr = contactHeader->numNormalConstr[ti];
    const PxU32 totalFrictionConstr = frictionHeader->numFrictionConstr[ti];
    const PxU32 numFrictionConstr = totalFrictionConstr & (~0x1u);

    const float maxPenBias = contactHeader->maxPenBias[ti];
    PxgTGSBlockSolverContactPoint* contacts = &contactPoints[batch.startConstraintIndex];
    PxgArticulationBlockResponse* resp = &responses[batch.mArticulationResponseIndex];

    const float4 invMass0_1_angDom0_1 = contactHeader->invMass0_1_angDom0_1[ti];
    const float4 normal_staticFriction = contactHeader->normal_staticFriction[ti];
    const float restitutionXdt = contactHeader->restitutionXdt[ti];
    const float p8 = contactHeader->p8[ti];
    const float cfm = contactHeader->cfm[ti];
    const PxU8 flags = PxU8(contactHeader->flags[ti]);

    const PxVec3 normal(normal_staticFriction.x, normal_staticFriction.y, normal_staticFriction.z);
    const float staticFrictionCof = normal_staticFriction.w;

    const PxVec3 relMotion = delta0.bottom - delta1.bottom;
    const float deltaV = normal.dot(relMotion);

    for (PxU32 i = 0; i < numNormalConstr; ++i)
    {
        PxgTGSBlockSolverContactPoint& c = contacts[i];
        PxgArticulationBlockResponse& r = *resp;
        resp++;

        const float4 raXn_extraCoeff = c.raXn_extraCoeff[ti];
        const PxVec3 raXn(raXn_extraCoeff.x, raXn_extraCoeff.y, raXn_extraCoeff.z);
        const float compliantContactCoef = raXn_extraCoeff.w;

        const float4 rbXn_targetVelW = c.rbXn_targetVelW[ti];
        const float appliedForce = c.appliedForce[ti];
        const float separation = c.separation[ti];
        const float maxImpulse = c.maxImpulse[ti];

        const PxVec3 rbXn(rbXn_targetVelW.x, rbXn_targetVelW.y, rbXn_targetVelW.z);
        const float targetVel = rbXn_targetVelW.w;

        float biasCoefficient = c.biasCoefficient[ti];

        const float resp0 = ref0 * c.resp0[ti];
        const float resp1 = ref1 * c.resp1[ti];
        const float unitResponse = resp0 + resp1;
        const float recipResponse = (unitResponse > 0.f) ? (1.f / (unitResponse + cfm)) : 0.f;

        float velMultiplier = recipResponse;
        if (restitutionXdt < 0.f)
            refCompliantCoeffsTGS(flags, restitutionXdt, unitResponse, recipResponse,
                                  compliantContactCoef, velMultiplier, biasCoefficient);

        const float v0 = angVel0.dot(raXn) + linVel0.dot(normal);
        const float v1 = angVel1.dot(rbXn) + linVel1.dot(normal);
        const float normalVel = v0 - v1;

        const float deltaBias = deltaV + delta0.top.dot(raXn) - delta1.top.dot(rbXn)
                              - targetVel * elapsedTime;

        const float sep = PxMax(minPen, separation + deltaBias);
        const float biased = PxMin(-maxPenBias, biasCoefficient * sep);
        const float tVelBias = recipResponse * biased;

        const float tempDeltaF = tVelBias - (normalVel - targetVel) * velMultiplier;
        const float _deltaF = PxMax(tempDeltaF, -appliedForce);
        const float _newForce = appliedForce + _deltaF;
        const float newForce = PxMin(_newForce, maxImpulse);
        const float deltaF = newForce - appliedForce;

        const PxVec3 deltaRALin = ref0 * PxVec3(r.deltaRALin_x[ti], r.deltaRALin_y[ti], r.deltaRALin_z[ti]);
        const PxVec3 deltaRAAng = ref0 * PxVec3(r.deltaRAAng_x[ti], r.deltaRAAng_y[ti], r.deltaRAAng_z[ti]);
        const PxVec3 deltaRBLin = ref1 * PxVec3(r.deltaRBLin_x[ti], r.deltaRBLin_y[ti], r.deltaRBLin_z[ti]);
        const PxVec3 deltaRBAng = ref1 * PxVec3(r.deltaRBAng_x[ti], r.deltaRBAng_y[ti], r.deltaRBAng_z[ti]);
        linVel0 += deltaRALin * deltaF;
        linVel1 += deltaRBLin * deltaF;
        angVel0 += deltaRAAng * deltaF;
        angVel1 += deltaRBAng * deltaF;

        imp0.top -= normal * deltaF;
        imp0.bottom -= raXn * deltaF;
        imp1.top += normal * deltaF;
        imp1.bottom += rbXn * deltaF;

        if (error)
            error->add(deltaF, velMultiplier);

        c.appliedForce[ti] = newForce;
        accumulatedNormalImpulse = accumulatedNormalImpulse + newForce;
    }

    accumulatedNormalImpulse = PxMax(accumulatedNormalImpulse, contactHeader->minNormalForce[ti]);

    if (numFrictionConstr)
    {
        PxgTGSBlockSolverContactFriction* frictions = &frictionPoints[batch.startFrictionIndex];
        const float biasCoefficient = frictionHeader->biasCoefficient[ti];
        const float dynamicFrictionCof = frictionHeader->dynamicFriction[ti];
        const float maxFrictionImpulse = staticFrictionCof * accumulatedNormalImpulse;
        const float maxDynFrictionImpulse = dynamicFrictionCof * accumulatedNormalImpulse;

        PxU32 broken = 0;

        const float4 frictionNormal0 = frictionHeader->frictionNormals[0][ti];
        const float4 frictionNormal1 = frictionHeader->frictionNormals[1][ti];
        const PxVec3 normal0(frictionNormal0.x, frictionNormal0.y, frictionNormal0.z);
        const PxVec3 normal1(frictionNormal1.x, frictionNormal1.y, frictionNormal1.z);

        const float deltaMotion0 = normal0.dot(relMotion);
        const float deltaMotion1 = normal1.dot(relMotion);

        for (PxU32 i = 0; i < numFrictionConstr; i += 2)
        {
            PxgTGSBlockSolverContactFriction& f0 = frictions[i];
            PxgArticulationBlockResponse& r0 = *resp;
            resp++;
            PxgTGSBlockSolverContactFriction& f1 = frictions[i + 1];
            PxgArticulationBlockResponse& r1 = *resp;
            resp++;

            const float4 raXn_error0 = f0.raXn_error[ti];
            const float4 rbXn_targetVelW0 = f0.rbXn_targetVelW[ti];
            const float initialError0 = raXn_error0.w;
            const float appliedForce0 = f0.appliedForce[ti];
            const float targetVel0 = rbXn_targetVelW0.w;

            const float4 raXn_error1 = f1.raXn_error[ti];
            const float4 rbXn_targetVelW1 = f1.rbXn_targetVelW[ti];
            const float initialError1 = raXn_error1.w;
            const float appliedForce1 = f1.appliedForce[ti];
            const float targetVel1 = rbXn_targetVelW1.w;

            const PxVec3 raXn0(raXn_error0.x, raXn_error0.y, raXn_error0.z);
            const PxVec3 rbXn0(rbXn_targetVelW0.x, rbXn_targetVelW0.y, rbXn_targetVelW0.z);
            const PxVec3 raXn1(raXn_error1.x, raXn_error1.y, raXn_error1.z);
            const PxVec3 rbXn1(rbXn_targetVelW1.x, rbXn_targetVelW1.y, rbXn_targetVelW1.z);

            const float resp0_0 = ref0 * f0.resp0[ti];
            const float resp0_1 = ref1 * f0.resp1[ti];
            const float resp0 = resp0_0 + resp0_1;
            const float velMultiplier0 = (resp0 > SVX_EPS_REAL) ? (p8 / (resp0 + cfm)) : 0.f;

            const float resp1_0 = ref0 * f1.resp0[ti];
            const float resp1_1 = ref1 * f1.resp1[ti];
            const float resp1 = resp1_0 + resp1_1;
            const float velMultiplier1 = (resp1 > SVX_EPS_REAL) ? (p8 / (resp1 + cfm)) : 0.f;

            const float v00 = angVel0.dot(raXn0) + linVel0.dot(normal0);
            const float v10 = angVel1.dot(rbXn0) + linVel1.dot(normal0);
            const float normalVel0 = v00 - v10;

            const float v01 = angVel0.dot(raXn1) + linVel0.dot(normal1);
            const float v11 = angVel1.dot(rbXn1) + linVel1.dot(normal1);
            const float normalVel1 = v01 - v11;

            const float error0 = initialError0 - targetVel0 * elapsedTime
                               + (raXn0.dot(delta0.top) - rbXn0.dot(delta1.top) + deltaMotion0);
            const float bias0 = error0 * biasCoefficient;
            const float tmp10 = appliedForce0 - (bias0 - targetVel0) * velMultiplier0;
            const float totalImpulse0 = tmp10 - normalVel0 * velMultiplier0;

            const float error1 = initialError1 - targetVel1 * elapsedTime
                               + (raXn1.dot(delta0.top) - rbXn1.dot(delta1.top) + deltaMotion1);
            const float bias1 = error1 * biasCoefficient;
            const float tmp11 = appliedForce1 - (bias1 - targetVel1) * velMultiplier1;
            const float totalImpulse1 = tmp11 - normalVel1 * velMultiplier1;

            const float totalImpulse = PxSqrt(totalImpulse0 * totalImpulse0
                                              + totalImpulse1 * totalImpulse1);

            const bool clampIt = totalImpulse > maxFrictionImpulse;
            const float ratio = clampIt ? std::fmin(maxDynFrictionImpulse, totalImpulse) / totalImpulse
                                        : 1.f;

            const float newAppliedForce0 = totalImpulse0 * ratio;
            const float newAppliedForce1 = totalImpulse1 * ratio;

            const float deltaF0 = newAppliedForce0 - appliedForce0;
            const float deltaF1 = newAppliedForce1 - appliedForce1;

            if (error)
            {
                error->add(deltaF0, velMultiplier0);
                error->add(deltaF1, velMultiplier1);
            }

            linVel0 += ref0 * PxVec3(r0.deltaRALin_x[ti], r0.deltaRALin_y[ti], r0.deltaRALin_z[ti]) * deltaF0;
            linVel1 += ref1 * PxVec3(r0.deltaRBLin_x[ti], r0.deltaRBLin_y[ti], r0.deltaRBLin_z[ti]) * deltaF0;
            angVel0 += ref0 * PxVec3(r0.deltaRAAng_x[ti], r0.deltaRAAng_y[ti], r0.deltaRAAng_z[ti]) * deltaF0;
            angVel1 += ref1 * PxVec3(r0.deltaRBAng_x[ti], r0.deltaRBAng_y[ti], r0.deltaRBAng_z[ti]) * deltaF0;

            linVel0 += ref0 * PxVec3(r1.deltaRALin_x[ti], r1.deltaRALin_y[ti], r1.deltaRALin_z[ti]) * deltaF1;
            linVel1 += ref1 * PxVec3(r1.deltaRBLin_x[ti], r1.deltaRBLin_y[ti], r1.deltaRBLin_z[ti]) * deltaF1;
            angVel0 += ref0 * PxVec3(r1.deltaRAAng_x[ti], r1.deltaRAAng_y[ti], r1.deltaRAAng_z[ti]) * deltaF1;
            angVel1 += ref1 * PxVec3(r1.deltaRBAng_x[ti], r1.deltaRBAng_y[ti], r1.deltaRBAng_z[ti]) * deltaF1;

            f0.appliedForce[ti] = newAppliedForce0;
            f1.appliedForce[ti] = newAppliedForce1;
            broken = broken | (clampIt ? 1u : 0u);

            imp0.top -= normal0 * deltaF0;
            imp0.bottom -= raXn0 * deltaF0;
            imp1.top += normal0 * deltaF0;
            imp1.bottom += rbXn0 * deltaF0;

            imp0.top -= normal1 * deltaF1;
            imp0.bottom -= raXn1 * deltaF1;
            imp1.top += normal1 * deltaF1;
            imp1.bottom += rbXn1 * deltaF1;
        }

        if (numFrictionConstr < totalFrictionConstr)
        {
            const float frictionScale = frictionHeader->torsionalFrictionScale[ti];
            PxgTGSBlockSolverContactFriction& f0 = frictions[numFrictionConstr];
            PxgArticulationBlockResponse& r0 = *resp;
            resp++;

            const float4 raXn_error0 = f0.raXn_error[ti];
            const float4 rbXn_targetVelW0 = f0.rbXn_targetVelW[ti];
            const float appliedForce0 = f0.appliedForce[ti];
            const float targetVel0 = rbXn_targetVelW0.w;

            const PxVec3 raXn0(raXn_error0.x, raXn_error0.y, raXn_error0.z);
            const PxVec3 rbXn0(rbXn_targetVelW0.x, rbXn_targetVelW0.y, rbXn_targetVelW0.z);

            const float resp0_0 = ref0 * f0.resp0[ti];
            const float resp0_1 = ref1 * f0.resp1[ti];
            const float resp0 = resp0_0 + resp0_1;
            const float velMultiplier0 = (resp0 > 0.f) ? (p8 / (resp0 + cfm)) : 0.f;

            const float v00 = angVel0.dot(raXn0);
            const float v10 = angVel1.dot(rbXn0);
            const float normalVel0 = v00 - v10;

            const float tmp10 = appliedForce0 - (-targetVel0) * velMultiplier0;
            const float totalImpulse = tmp10 - normalVel0 * velMultiplier0;

            const bool clampIt = PxAbs(totalImpulse) > (maxFrictionImpulse * frictionScale);
            const float torsLo = -maxDynFrictionImpulse * frictionScale;
            const float torsHi = maxDynFrictionImpulse * frictionScale;
            const float totalClamped = (gExtTorsClampVariant == 1)
                                     ? refCudaFminf(refCudaFmaxf(totalImpulse, torsLo), torsHi)
                                     : PxClamp(totalImpulse, torsLo, torsHi);
            const float newAppliedForce = clampIt ? totalClamped : totalImpulse;
            const float deltaF0 = newAppliedForce - appliedForce0;
            if (error)
                error->add(deltaF0, velMultiplier0);

            linVel0 += ref0 * PxVec3(r0.deltaRALin_x[ti], r0.deltaRALin_y[ti], r0.deltaRALin_z[ti]) * deltaF0;
            linVel1 += ref1 * PxVec3(r0.deltaRBLin_x[ti], r0.deltaRBLin_y[ti], r0.deltaRBLin_z[ti]) * deltaF0;
            angVel0 += ref0 * PxVec3(r0.deltaRAAng_x[ti], r0.deltaRAAng_y[ti], r0.deltaRAAng_z[ti]) * deltaF0;
            angVel1 += ref1 * PxVec3(r0.deltaRBAng_x[ti], r0.deltaRBAng_y[ti], r0.deltaRBAng_z[ti]) * deltaF0;

            f0.appliedForce[ti] = newAppliedForce;
            broken = broken | (clampIt ? 1u : 0u);

            imp0.bottom -= raXn0 * deltaF0;
            imp1.bottom += rbXn0 * deltaF0;
        }

        frictionHeader->broken[ti] = broken;
    }

    vel0.bottom = linVel0;
    vel0.top = angVel0;
    vel1.bottom = linVel1;
    vel1.top = angVel1;

    impulse0 = imp0.scale(invMass0_1_angDom0_1.x * ref0, invMass0_1_angDom0_1.z * ref0);
    impulse1 = imp1.scale(invMass0_1_angDom0_1.y * ref1, invMass0_1_angDom0_1.w * ref1);
}

// solveExt1DBlockTGS (solverBlockTGS.cuh:932-1089), transcribed; the 1D solver
// constants come from the ENGINE's Dy::compute1dConstraintSolverConstantsTGS
inline void refSolveExt1DTGS(const PxgBlockConstraintBatch& batch,
                             UnAlignedSpatialVector& vel0, UnAlignedSpatialVector& vel1,
                             const UnAlignedSpatialVector& motion0,
                             const UnAlignedSpatialVector& motion1, PxU32 ti,
                             const PxgTGSBlockSolverConstraint1DHeader* headers,
                             PxgTGSBlockSolverConstraint1DCon* rowsCon,
                             PxgArticulationBlockResponse* artiResponse,
                             const PxQuat& deltaQ0, const PxQuat& deltaQ1,
                             float elapsedTime,
                             UnAlignedSpatialVector& impulse0, UnAlignedSpatialVector& impulse1,
                             bool residualReportingEnabled, float ref0, float ref1)
{
    const PxgTGSBlockSolverConstraint1DHeader* header = &headers[batch.mConstraintBatchIndex];
    PxgTGSBlockSolverConstraint1DCon* baseCon = &rowsCon[batch.startConstraintIndex];
    PxgArticulationBlockResponse* responses = &artiResponse[batch.mArticulationResponseIndex];

    PxVec3 linVel0 = vel0.bottom, linVel1 = vel1.bottom;
    PxVec3 angVel0 = vel0.top, angVel1 = vel1.top;

    const float4 raInvMass0 = header->rAWorld_invMass0D0[ti];
    const float4 rbInvMass1 = header->rBWorld_invMass1D1[ti];

    const float invMass0 = ref0 * raInvMass0.w;
    const float invMass1 = ref1 * rbInvMass1.w;
    const float invInertiaScale0 = ref0 * header->invInertiaScale0[ti];
    const float invInertiaScale1 = ref1 * header->invInertiaScale1[ti];

    const PxVec3 raPrev(raInvMass0.x, raInvMass0.y, raInvMass0.z);
    const PxVec3 rbPrev(rbInvMass1.x, rbInvMass1.y, rbInvMass1.z);
    const PxVec3 ra = deltaQ0.rotate(raPrev);
    const PxVec3 rb = deltaQ1.rotate(rbPrev);
    const PxVec3 raMotion = (ra + motion0.bottom) - raPrev;
    const PxVec3 rbMotion = (rb + motion1.bottom) - rbPrev;

    const uchar4 rowCounts = header->rowCounts_breakable_orthoAxisCount[ti];
    const PxU32 rowCount = rowCounts.x;
    const PxReal cfm = header->cfm[ti];

    PxVec3 li0(0.f), li1(0.f), ai0(0.f), ai1(0.f);

    for (PxU32 i = 0; i < rowCount; ++i)
    {
        PxgTGSBlockSolverConstraint1DCon& ccon = baseCon[i];
        const PxgArticulationBlockResponse& response = responses[i];

        const float4 _c0 = ccon.lin0XYZ_initBiasOrCoeff0[ti];
        const float4 _c1 = ccon.lin1XYZ_biasScaleOrCoeff1[ti];
        const float4 _c2 = ccon.ang0XYZ_velMultiplierOrCoeff2[ti];
        const float4 _c3 = ccon.ang1XYZ_velTargetOrCoeff3[ti];

        const PxVec3 clinVel0(_c0.x, _c0.y, _c0.z);
        const PxVec3 clinVel1(_c1.x, _c1.y, _c1.z);
        const PxVec3 cangVel0(_c2.x, _c2.y, _c2.z);
        const PxVec3 cangVel1(_c3.x, _c3.y, _c3.z);

        const PxU32 flags = ccon.flags[ti];
        const bool isSpringConstraint = (flags & DY_FLAG_SPRING) != 0;

        const PxReal resp0 = ref0 * ccon.resp0[ti];
        const PxReal resp1 = ref1 * ccon.resp1[ti];
        float unitResponse = resp0 + resp1;
        unitResponse += cfm;

        const PxReal recipResponse = Dy::computeRecipUnitResponse(unitResponse,
                                                                  DY_ARTICULATION_MIN_RESPONSE);

        const bool isAccelerationSpring = (flags & DY_FLAG_ACCEL) != 0;
        const PxReal geometricError = ccon.geometricError[ti];

        PxReal biasScale, velMultiplier, initBias, targetVel;
        Dy::compute1dConstraintSolverConstantsTGS(isSpringConstraint, isAccelerationSpring,
                                                  geometricError, unitResponse, recipResponse,
                                                  _c0.w, _c1.w, _c2.w, _c3.w, biasScale,
                                                  velMultiplier, initBias, targetVel);

        initBias = (biasScale == 0.f) ? 0.f : initBias;

        const PxReal errorDelta = refResolvedGeometricErrorTGS(
            raMotion, rbMotion, clinVel0, clinVel1, motion0.top, motion1.top, cangVel0,
            cangVel1, ccon.angularErrorScale[ti], isSpringConstraint, targetVel, elapsedTime);

        const PxReal maxBias = ccon.maxBias[ti];
        const PxReal vMul = isSpringConstraint ? velMultiplier : recipResponse * velMultiplier;
        const float appliedForce = ccon.appliedForce[ti];

        const PxReal unclampedBias = initBias + errorDelta * biasScale;
        const PxReal minBias = refMinBiasTGS(flags, maxBias);
        const PxReal bias = PxClamp(unclampedBias, minBias, maxBias);

        const PxReal constant = isSpringConstraint ? (bias + targetVel)
                                                   : recipResponse * (bias + targetVel);

        const float maxImpulse = ccon.maxImpulse[ti];
        const float minImpulse = ccon.minImpulse[ti];

        const float v0 = linVel0.dot(clinVel0) + angVel0.dot(cangVel0);
        const float v1 = linVel1.dot(clinVel1) + angVel1.dot(cangVel1);
        const float normalVel = v0 - v1;

        const float unclampedForce = appliedForce + (vMul * normalVel + constant);
        const float clampedForce =
            (gExt1DClampVariant == 1) ? PxMin(maxImpulse, PxMax(minImpulse, unclampedForce))
          : (gExt1DClampVariant == 2) ? std::fmin(maxImpulse, std::fmax(minImpulse, unclampedForce))
                                      : refCudaFminf(maxImpulse, refCudaFmaxf(minImpulse, unclampedForce));
        const float deltaF = clampedForce - appliedForce;

        ccon.appliedForce[ti] = clampedForce;
        if (residualReportingEnabled)
            ccon.residual[ti] = vMul == 0.f ? 0.f : deltaF / vMul;

        li0 = clinVel0 * deltaF + li0;
        ai0 = cangVel0 * deltaF + ai0;
        li1 = clinVel1 * deltaF + li1;
        ai1 = cangVel1 * deltaF + ai1;

        const PxVec3 linVa = ref0 * PxVec3(response.deltaRALin_x[ti], response.deltaRALin_y[ti], response.deltaRALin_z[ti]);
        const PxVec3 angVa = ref0 * PxVec3(response.deltaRAAng_x[ti], response.deltaRAAng_y[ti], response.deltaRAAng_z[ti]);
        const PxVec3 linVb = ref1 * PxVec3(response.deltaRBLin_x[ti], response.deltaRBLin_y[ti], response.deltaRBLin_z[ti]);
        const PxVec3 angVb = ref1 * PxVec3(response.deltaRBAng_x[ti], response.deltaRBAng_y[ti], response.deltaRBAng_z[ti]);

        linVel0 = linVa * deltaF + linVel0;
        angVel0 = angVa * deltaF + angVel0;
        linVel1 = linVb * deltaF + linVel1;
        angVel1 = angVb * deltaF + angVel1;
    }

    vel0.top = angVel0; vel0.bottom = linVel0;
    vel1.top = angVel1; vel1.bottom = linVel1;

    impulse0.top = li0 * invMass0; impulse0.bottom = ai0 * invInertiaScale0;
    impulse1.top = li1 * invMass1; impulse1.bottom = ai1 * invInertiaScale1;
}

// ===========================================================================
// P8.122a: the PGS twins -- solveExtContactsBlock (solver.cuh:54-320) and
// solveExt1DBlock (solverBlock.cuh:208-326) over the engine structs.  The 1D
// constants come from the ENGINE's host-callable
// Dy::compute1dConstraintSolverConstantsPGS / Dy::computeRecipUnitResponse;
// the contact-side computeContactCoefficients (constraintPrepShared.cuh:314)
// is __device__-only and transcribed here.  The CUDA's software-pipelined
// prefetch is un-pipelined into direct loads with the same indices (see the
// module's trap 1).  The body-A final-scale factor swap at solver.cuh:315
// (the TENTH upstream-bug candidate) is transcribed verbatim.
// ===========================================================================

inline void refComputeContactCoefficientsPGS(uint32_t flags, float restitution,
                                             float unitResponse, float recipResponse,
                                             float targetVelocity, float coeff0, float coeff1,
                                             float& velMultiplier, float& impulseMultiplier,
                                             float& unbiasedErr, float& biasedErr)
{
    if (restitution < 0.f)
    {
        const bool accelSpring = (flags & COMPLIANT_ACCEL) != 0;
        const float massIfAccelElseOne = accelSpring ? recipResponse : 1.0f;
        const float oneIfAccelElseR = accelSpring ? 1.0f : unitResponse;

        const float a = coeff0;
        const float b = coeff1 * massIfAccelElseOne;

        const float x = 1.f / (a * oneIfAccelElseR + 1.f);
        const float scaledBias = x * b;

        velMultiplier = x * a * massIfAccelElseOne;
        impulseMultiplier = 1.f - x;
        unbiasedErr = biasedErr = targetVelocity * velMultiplier - scaledBias;
    }
    else
    {
        velMultiplier = recipResponse;
        biasedErr = coeff0 * velMultiplier;
        unbiasedErr = coeff1 * velMultiplier;
        impulseMultiplier = 1.f;
    }
}

// solveExtContactsBlock (solver.cuh:54-320), transcribed over the engine structs
inline void refSolveExtContactPGS(const PxgBlockConstraintBatch& batch,
                                  UnAlignedSpatialVector& vel0, UnAlignedSpatialVector& vel1,
                                  bool doFriction, PxU32 ti,
                                  PxgBlockSolverContactHeader* contactHeaders,
                                  PxgBlockSolverFrictionHeader* frictionHeaders,
                                  PxgBlockSolverContactPoint* contactPoints,
                                  PxgBlockSolverContactFriction* frictionPoints,
                                  PxgArticulationBlockResponse* responses,
                                  UnAlignedSpatialVector& impulse0,
                                  UnAlignedSpatialVector& impulse1,
                                  RefError* error, float ref0, float ref1)
{
    UnAlignedSpatialVector imp0(PxVec3(0.f), PxVec3(0.f)), imp1(PxVec3(0.f), PxVec3(0.f));

    PxgBlockSolverContactHeader& contactHeader = contactHeaders[batch.mConstraintBatchIndex];
    PxgBlockSolverFrictionHeader& frictionHeader = frictionHeaders[batch.mConstraintBatchIndex];

    const float4 invMass0_1_angDom0_1 = contactHeader.invMass0_1_angDom0_1[ti];

    const PxU32 numNormalConstr = contactHeader.numNormalConstr[ti];
    const PxU32 numFrictionConstr = frictionHeader.numFrictionConstr[ti];

    const float restitution = contactHeader.restitution[ti];
    const float p8 = 0.8f;   // FIXED (module trap 2)
    const float cfm = contactHeader.cfm[ti];
    const PxU8 flags = PxU8(contactHeader.flags[ti]);

    PxgBlockSolverContactPoint* contacts = &contactPoints[batch.startConstraintIndex];

    PxReal accumulatedNormalImpulse = 0.f;

    UnAlignedSpatialVector v0 = vel0;
    UnAlignedSpatialVector v1 = vel1;

    const float4 normal_staticFriction = contactHeader.normal_staticFriction[ti];
    const PxVec3 normal(normal_staticFriction.x, normal_staticFriction.y,
                        normal_staticFriction.z);
    const PxReal staticCof = normal_staticFriction.w;

    PxU32 respIndex = batch.mArticulationResponseIndex;

    PxgBlockSolverContactFriction* frictions = &frictionPoints[batch.startFrictionIndex];

    for (PxU32 i = 0; i < numNormalConstr; ++i)
    {
        PxgBlockSolverContactPoint& c = contacts[i];
        const PxgArticulationBlockResponse& resp = responses[respIndex];
        respIndex++;

        const float4 raXn_targetVelocity = c.raXn_targetVelocity[ti];
        const float4 rbXn_maxImpulse = c.rbXn_maxImpulse[ti];
        const float appliedForce = c.appliedForce[ti];
        const float resp0 = c.resp0[ti];
        const float resp1 = c.resp1[ti];
        const float coeff0 = c.coeff0[ti];
        const float coeff1 = c.coeff1[ti];

        const PxVec3 deltaRALin(resp.deltaRALin_x[ti], resp.deltaRALin_y[ti], resp.deltaRALin_z[ti]);
        const PxVec3 deltaRAAng(resp.deltaRAAng_x[ti], resp.deltaRAAng_y[ti], resp.deltaRAAng_z[ti]);
        const PxVec3 deltaRBLin(resp.deltaRBLin_x[ti], resp.deltaRBLin_y[ti], resp.deltaRBLin_z[ti]);
        const PxVec3 deltaRBAng(resp.deltaRBAng_x[ti], resp.deltaRBAng_y[ti], resp.deltaRBAng_z[ti]);

        const PxVec3 raXn(raXn_targetVelocity.x, raXn_targetVelocity.y, raXn_targetVelocity.z);
        const PxVec3 rbXn(rbXn_maxImpulse.x, rbXn_maxImpulse.y, rbXn_maxImpulse.z);
        const float targetVelocity = raXn_targetVelocity.w;
        const float maxImpulse = rbXn_maxImpulse.w;

        const float unitResponse = ref0 * resp0 + ref1 * resp1;
        const float recipResponse = (unitResponse > 0.0f) ? 1.0f / (unitResponse + cfm) : 0.0f;
        float velMultiplier = recipResponse;
        float impulseMul = 1.0f;
        float unbiasedError, biasedErr;

        refComputeContactCoefficientsPGS(flags, restitution, unitResponse, recipResponse,
                                         targetVelocity, coeff0, coeff1,
                                         velMultiplier, impulseMul, unbiasedError, biasedErr);

        const UnAlignedSpatialVector deltaVA(ref0 * deltaRAAng, ref0 * deltaRALin);
        const UnAlignedSpatialVector deltaVB(ref1 * deltaRBAng, ref1 * deltaRBLin);

        const float v0_ = v0.bottom.dot(normal) + v0.top.dot(raXn);
        const float v1_ = v1.bottom.dot(normal) + v1.top.dot(rbXn);
        const float normalVel = v0_ - v1_;

        const float tempDeltaF = biasedErr - normalVel * velMultiplier;
        const float _deltaF = fmaxf(tempDeltaF, -appliedForce);
        const float _newForce = appliedForce * impulseMul + _deltaF;
        const float newForce = fminf(_newForce, maxImpulse);
        const float deltaF = newForce - appliedForce;

        if (error)
            error->add(deltaF, velMultiplier);
        c.appliedForce[ti] = newForce;

        imp0.bottom -= raXn * deltaF;
        imp0.top -= normal * deltaF;
        imp1.bottom += rbXn * deltaF;
        imp1.top += normal * deltaF;

        v0 += deltaVA * deltaF;
        v1 += deltaVB * deltaF;

        accumulatedNormalImpulse = accumulatedNormalImpulse + newForce;
    }

    accumulatedNormalImpulse = PxMax(accumulatedNormalImpulse, contactHeader.minNormalForce[ti]);

    if (numFrictionConstr && doFriction)
    {
        const float dynamicFrictionCof = frictionHeader.dynamicFriction[ti];
        const float maxFrictionImpulse = staticCof * accumulatedNormalImpulse;
        const float maxDynFrictionImpulse = dynamicFrictionCof * accumulatedNormalImpulse;

        PxU32 broken = 0;

        for (PxU32 i = 0; i < numFrictionConstr; ++i)
        {
            PxgBlockSolverContactFriction& f = frictions[i];
            const PxgArticulationBlockResponse& resp = responses[respIndex];
            respIndex++;

            const float4 frictionNormal = frictionHeader.frictionNormals[i & 1][ti];

            const float4 raXn_extraCoeff = f.raXn_bias[ti];
            const float4 rbXn_targetVelW = f.rbXn_targetVelW[ti];
            const float resp0 = f.resp0[ti];
            const float resp1 = f.resp1[ti];
            const float appliedForce = f.appliedForce[ti];

            const PxVec3 deltaRALin(resp.deltaRALin_x[ti], resp.deltaRALin_y[ti], resp.deltaRALin_z[ti]);
            const PxVec3 deltaRAAng(resp.deltaRAAng_x[ti], resp.deltaRAAng_y[ti], resp.deltaRAAng_z[ti]);
            const PxVec3 deltaRBLin(resp.deltaRBLin_x[ti], resp.deltaRBLin_y[ti], resp.deltaRBLin_z[ti]);
            const PxVec3 deltaRBAng(resp.deltaRBAng_x[ti], resp.deltaRBAng_y[ti], resp.deltaRBAng_z[ti]);

            const PxVec3 raXn(raXn_extraCoeff.x, raXn_extraCoeff.y, raXn_extraCoeff.z);
            const PxVec3 rbXn(rbXn_targetVelW.x, rbXn_targetVelW.y, rbXn_targetVelW.z);

            const float resp_ = ref0 * resp0 + ref1 * resp1;
            const float velMultiplier = (resp_ > SVX_EPS_REAL) ? (p8 / resp_) : 0.f;
            const float bias = raXn_extraCoeff.w;
            const float targetVel = rbXn_targetVelW.w;

            const UnAlignedSpatialVector deltaVA(ref0 * deltaRAAng, ref0 * deltaRALin);
            const UnAlignedSpatialVector deltaVB(ref1 * deltaRBAng, ref1 * deltaRBLin);

            const PxVec3 fNormal(frictionNormal.x, frictionNormal.y, frictionNormal.z);

            const float v0_ = v0.top.dot(raXn) + v0.bottom.dot(fNormal);
            const float v1_ = v1.top.dot(rbXn) + v1.bottom.dot(fNormal);
            const float normalVel = v0_ - v1_;

            const float tmp1 = appliedForce - (bias - targetVel) * velMultiplier;
            const float totalImpulse = tmp1 - normalVel * velMultiplier;

            const bool doClamp = fabsf(totalImpulse) > maxFrictionImpulse;
            const float totalClamped =
                fminf(maxDynFrictionImpulse, fmaxf(-maxDynFrictionImpulse, totalImpulse));
            const float newAppliedForce = doClamp ? totalClamped : totalImpulse;
            const float deltaF = newAppliedForce - appliedForce;

            if (error)
                error->add(deltaF, velMultiplier);

            v0 += deltaVA * deltaF;
            v1 += deltaVB * deltaF;

            imp0.bottom -= raXn * deltaF;
            imp0.top -= fNormal * deltaF;
            imp1.bottom += rbXn * deltaF;
            imp1.top += fNormal * deltaF;

            f.appliedForce[ti] = newAppliedForce;
            broken = broken | (doClamp ? 1u : 0u);
        }
        frictionHeader.broken[ti] = broken;
    }

    // the TENTH upstream-bug candidate, verbatim (body A's factors swapped)
    impulse0 = imp0.scale(ref0 * invMass0_1_angDom0_1.z, ref0 * invMass0_1_angDom0_1.x);
    impulse1 = imp1.scale(ref1 * invMass0_1_angDom0_1.y, ref1 * invMass0_1_angDom0_1.w);

    vel0 = v0;
    vel1 = v1;
}

// solveExt1DBlock (solverBlock.cuh:208-326); the constants come from the
// ENGINE's Dy::compute1dConstraintSolverConstantsPGS
inline void refSolveExt1DPGS(const PxgBlockConstraintBatch& batch,
                             UnAlignedSpatialVector& vel0, UnAlignedSpatialVector& vel1,
                             PxU32 ti,
                             const PxgBlockSolverConstraint1DHeader* headers,
                             PxgBlockSolverConstraint1DCon* rowsCon,
                             PxgBlockSolverConstraint1DMod* rowsMod,
                             PxgArticulationBlockResponse* artiResponse,
                             UnAlignedSpatialVector& impulse0,
                             UnAlignedSpatialVector& impulse1,
                             bool residualReportingEnabled, float ref0, float ref1)
{
    const PxgBlockSolverConstraint1DHeader* header = &headers[batch.mConstraintBatchIndex];
    PxgBlockSolverConstraint1DCon* baseCon = &rowsCon[batch.startConstraintIndex];
    PxgBlockSolverConstraint1DMod* baseMod = &rowsMod[batch.startConstraintIndex];

    PxVec3 linVel0 = vel0.bottom;
    PxVec3 linVel1 = vel1.bottom;
    PxVec3 angVel0 = vel0.top;
    PxVec3 angVel1 = vel1.top;

    // the accumulators SEED from the inout impulses (module trap 8)
    PxVec3 li0 = impulse0.bottom;
    PxVec3 li1 = impulse1.bottom;
    PxVec3 ai0 = impulse0.top;
    PxVec3 ai1 = impulse1.top;

    const float invMass0 = ref0 * header->invMass0D0[ti];
    const float invMass1 = ref1 * header->invMass1D1[ti];

    const float invInertiaScale0 = ref0 * header->invInertiaScale0[ti];
    const float invInertiaScale1 = ref1 * header->invInertiaScale1[ti];

    const float cfm = header->cfm[ti];

    const PxU32 numRows = header->rowCounts[ti];
    for (PxU32 i = 0; i < numRows; ++i)
    {
        PxgBlockSolverConstraint1DCon& ccon = baseCon[i];
        PxgBlockSolverConstraint1DMod& cmod = baseMod[i];
        const PxgArticulationBlockResponse& response = artiResponse[i];

        const float4 _clinVel0_minImpulse = ccon.lin0XYZ_minImpulse[ti];
        const float4 _clinVel1_maxImpulse = ccon.lin1XYZ_maxImpulse[ti];
        const float4 _cangVel0_resp0 = ccon.ang0XYZ_resp0[ti];
        const float4 _cangVel1_resp1 = ccon.ang1XYZ_resp1[ti];

        const PxReal resp0 = ref0 * _cangVel0_resp0.w;
        const PxReal resp1 = ref1 * _cangVel1_resp1.w;
        const PxReal initJointSpeed = ccon.initJointSpeed[ti];

        const PxVec3 clinVel0(_clinVel0_minImpulse.x, _clinVel0_minImpulse.y, _clinVel0_minImpulse.z);
        const PxVec3 clinVel1(_clinVel1_maxImpulse.x, _clinVel1_maxImpulse.y, _clinVel1_maxImpulse.z);
        const PxVec3 cangVel0(_cangVel0_resp0.x, _cangVel0_resp0.y, _cangVel0_resp0.z);
        const PxVec3 cangVel1(_cangVel1_resp1.x, _cangVel1_resp1.y, _cangVel1_resp1.z);

        const PxReal coeff0 = cmod.coeff0[ti];
        const PxReal coeff1 = cmod.coeff1[ti];
        const PxU32 flags = cmod.flags[ti];

        const PxReal unitResponse = resp0 + resp1 + cfm;   // module trap 10

        const PxReal recipResponse = Dy::computeRecipUnitResponse(unitResponse, DY_MIN_RESPONSE);

        PxReal constant, unbiasedConstant, vMul, iMul;

        const bool isSpring = (flags & DY_FLAG_SPRING) != 0;
        const bool isAccelerationSpring = (flags & DY_FLAG_ACCEL) != 0;

        Dy::compute1dConstraintSolverConstantsPGS(isSpring, isAccelerationSpring, coeff0, coeff1,
                                                  initJointSpeed, unitResponse, recipResponse,
                                                  constant, unbiasedConstant, vMul, iMul);

        const float appliedForce = cmod.appliedForce[ti];
        const float maxImpulse = _clinVel1_maxImpulse.w;
        const float minImpulse = _clinVel0_minImpulse.w;

        const float v0 = linVel0.dot(clinVel0) + angVel0.dot(cangVel0);
        const float v1 = linVel1.dot(clinVel1) + angVel1.dot(cangVel1);
        const float normalVel = v0 - v1;

        const float unclampedForce = iMul * appliedForce + (vMul * normalVel + constant);
        const float clampedForce = fminf(maxImpulse, fmaxf(minImpulse, unclampedForce));
        const float deltaF = clampedForce - appliedForce;

        cmod.appliedForce[ti] = clampedForce;
        if (residualReportingEnabled)
            cmod.residual[ti] = (vMul == 0.f) ? 0.f : deltaF / vMul;

        li0 = clinVel0 * deltaF + li0;
        ai0 = cangVel0 * deltaF + ai0;
        li1 = clinVel1 * deltaF + li1;
        ai1 = cangVel1 * deltaF + ai1;

        const PxVec3 linVa = ref0 * PxVec3(response.deltaRALin_x[ti], response.deltaRALin_y[ti],
                                           response.deltaRALin_z[ti]);
        const PxVec3 angVa = ref0 * PxVec3(response.deltaRAAng_x[ti], response.deltaRAAng_y[ti],
                                           response.deltaRAAng_z[ti]);
        const PxVec3 linVb = ref1 * PxVec3(response.deltaRBLin_x[ti], response.deltaRBLin_y[ti],
                                           response.deltaRBLin_z[ti]);
        const PxVec3 angVb = ref1 * PxVec3(response.deltaRBAng_x[ti], response.deltaRBAng_y[ti],
                                           response.deltaRBAng_z[ti]);

        linVel0 = linVa * deltaF + linVel0;
        angVel0 = angVa * deltaF + angVel0;
        linVel1 = linVb * deltaF + linVel1;
        angVel1 = angVb * deltaF + angVel1;
    }

    vel0.top = angVel0; vel0.bottom = linVel0;
    vel1.top = angVel1; vel1.bottom = linVel1;

    impulse0.top = li0 * invMass0; impulse0.bottom = ai0 * invInertiaScale0;
    impulse1.top = li1 * invMass1; impulse1.bottom = ai1 * invInertiaScale1;
}

}  // namespace pxgtest

#endif  // PXG_ARTI_EXT_SOLVE_REF_H
