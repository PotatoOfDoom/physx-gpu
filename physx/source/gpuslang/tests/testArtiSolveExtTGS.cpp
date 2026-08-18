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

// P8.99a: `solveExtContactBlockTGS` (solver.cuh:430-735) and
// `solveExt1DBlockTGS` (solverBlockTGS.cuh:932-1089) on the ENGINE BLOCK
// LAYOUT — the two bodies the articulation self-constraint and partition
// solves dispatch through batch.constraintType.
//
// THE ORACLE: transcriptions over the ENGINE's own host-includable block
// structs (PxgConstraintBlock.h / PxgSolverConstraintBlock1D.h /
// PxgSolverConstraintDesc.h), with the 1D constants computed by the ENGINE's
// PX_CUDA_CALLABLE Dy::compute1dConstraintSolverConstantsTGS /
// Dy::computeRecipUnitResponse — the strongest available oracle for the 1D
// path.  The compliant-contact coefficient pair is transcribed (its reduced
// overload is __device__-only), pinned previously by the tranche-188 battery.
//
// WHAT THE FIXTURE IS SHAPED TO SEE
//  * ONE batch per family, 32 lanes as 32 independent constraint slots, with
//    per-lane counts covering: numNormalConstr 0..6, totalFrictionConstr
//    0..5 INCLUDING odd counts (the torsional tail), 1D rowCount 0..6;
//  * per-lane flag mixes: compliant contacts (restitutionXdt < 0) with and
//    without the acceleration-spring bit; 1D rows as hard / inequality /
//    spring / acceleration-spring / bounce (coeff2 != 0 in non-spring mode);
//  * non-zero startConstraintIndex / startFrictionIndex /
//    mArticulationResponseIndex so the three index bases are pinned;
//  * friction clamping both ways (the combined-magnitude sqrt clamp), the
//    minNormalForce floor, maxImpulse clamping;
//  * two rounds: refs (1, 1) and refs (0.75, 1.25) — every ref site scales;
//  * the mutated block fields (appliedForce, broken, residual) and the io
//    outputs (velocities, impulses, error accumulator) compared byte for
//    byte; untouched block bytes must come back at the seed.

#include "pxgTestFramework.h"
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
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

namespace
{
using namespace pxgtest;
using namespace physx;
using Cm::UnAlignedSpatialVector;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

static std::vector<uint8_t> svxLoader(const char* file) { return loadShaderBlob(file); }

const uint32_t SVX_LANES = 32;
const uint32_t SVX_MAX_CONTACTS = 6, SVX_MAX_FRICTIONS = 6, SVX_MAX_ROWS = 6;
const uint32_t SVX_START_CONSTRAINT = 2, SVX_START_FRICTION = 3, SVX_START_RESP = 5;
const uint32_t SVX_NB_RESP = SVX_START_RESP + SVX_MAX_CONTACTS + SVX_MAX_FRICTIONS + 1 + 2;
const float SVX_ELAPSED = 0.011f, SVX_MINPEN = -0.02f;
const float SVX_EPS_REAL = 1.192092896e-07f;
const uint32_t DY_FLAG_SPRING = 1u << 5, DY_FLAG_INEQ = 1u << 6, DY_FLAG_ACCEL = 1u << 7;
const uint32_t COMPLIANT_ACCEL = 1u << 3;

static float fv(uint32_t a, uint32_t l, uint32_t d, uint32_t salt)
{
    uint32_t h = a * 2654435761u + l * 40503u + d * 9973u + salt * 733u + 19u;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    return -1.0f + 2.0f * float(h & 0xFFFFFFu) / float(0x1000000);
}

// per-lane spatial io (the shader's io buffer layout)
struct LaneIo
{
    UnAlignedSpatialVector vel0, vel1, d0, d1;
    PxQuat dq0, dq1;
};

static LaneIo svxLaneIo(uint32_t lane, uint32_t salt)
{
    LaneIo io;
    io.vel0 = UnAlignedSpatialVector(
        PxVec3(fv(lane, 0, 0, salt), fv(lane, 0, 1, salt), fv(lane, 0, 2, salt)),
        PxVec3(fv(lane, 0, 3, salt), fv(lane, 0, 4, salt), fv(lane, 0, 5, salt)));
    io.vel1 = UnAlignedSpatialVector(
        PxVec3(fv(lane, 1, 0, salt), fv(lane, 1, 1, salt), fv(lane, 1, 2, salt)),
        PxVec3(fv(lane, 1, 3, salt), fv(lane, 1, 4, salt), fv(lane, 1, 5, salt)));
    io.d0 = UnAlignedSpatialVector(
        PxVec3(0.1f * fv(lane, 2, 0, salt), 0.1f * fv(lane, 2, 1, salt), 0.1f * fv(lane, 2, 2, salt)),
        PxVec3(0.1f * fv(lane, 2, 3, salt), 0.1f * fv(lane, 2, 4, salt), 0.1f * fv(lane, 2, 5, salt)));
    io.d1 = UnAlignedSpatialVector(
        PxVec3(0.1f * fv(lane, 3, 0, salt), 0.1f * fv(lane, 3, 1, salt), 0.1f * fv(lane, 3, 2, salt)),
        PxVec3(0.1f * fv(lane, 3, 3, salt), 0.1f * fv(lane, 3, 4, salt), 0.1f * fv(lane, 3, 5, salt)));
    PxQuat q0(fv(lane, 4, 0, salt), fv(lane, 4, 1, salt), fv(lane, 4, 2, salt),
              1.0f + 0.3f * fv(lane, 4, 3, salt));
    PxQuat q1(fv(lane, 5, 0, salt), fv(lane, 5, 1, salt), fv(lane, 5, 2, salt),
              1.0f + 0.3f * fv(lane, 5, 3, salt));
    io.dq0 = q0.getNormalized();
    io.dq1 = q1.getNormalized();
    return io;
}

// the __device__-only reduced compliant-coefficient overload, transcribed
static void refCompliantCoeffsTGS(uint32_t flags, float nrdt, float unitResponse,
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
static float refResolvedGeometricErrorTGS(const PxVec3& deltaLin0, const PxVec3& deltaLin1,
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

static float refMinBiasTGS(uint32_t flags, float maxBias)
{
    return -((flags & DY_FLAG_INEQ) != 0 ? PX_MAX_F32 : maxBias);
}

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
static void refSolveExtContactTGS(const PxgBlockConstraintBatch& batch,
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
            const float totalClamped = PxClamp(totalImpulse, -maxDynFrictionImpulse * frictionScale,
                                               maxDynFrictionImpulse * frictionScale);
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
static void refSolveExt1DTGS(const PxgBlockConstraintBatch& batch,
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
        const float clampedForce = std::fmin(maxImpulse, std::fmax(minImpulse, unclampedForce));
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

// ---- seeds ----

static void svxSeedResponses(std::vector<PxgArticulationBlockResponse>& rsp, uint32_t salt)
{
    rsp.resize(SVX_NB_RESP);
    std::memset(rsp.data(), 0xAB, rsp.size() * sizeof(rsp[0]));
    for (uint32_t r = 0; r < SVX_NB_RESP; ++r)
        for (uint32_t l = 0; l < 32; ++l)
        {
            rsp[r].deltaRALin_x[l] = 0.2f * fv(r, l, 0, salt);
            rsp[r].deltaRALin_y[l] = 0.2f * fv(r, l, 1, salt);
            rsp[r].deltaRALin_z[l] = 0.2f * fv(r, l, 2, salt);
            rsp[r].deltaRAAng_x[l] = 0.2f * fv(r, l, 3, salt);
            rsp[r].deltaRAAng_y[l] = 0.2f * fv(r, l, 4, salt);
            rsp[r].deltaRAAng_z[l] = 0.2f * fv(r, l, 5, salt);
            rsp[r].deltaRBLin_x[l] = 0.2f * fv(r, l, 6, salt);
            rsp[r].deltaRBLin_y[l] = 0.2f * fv(r, l, 7, salt);
            rsp[r].deltaRBLin_z[l] = 0.2f * fv(r, l, 8, salt);
            rsp[r].deltaRBAng_x[l] = 0.2f * fv(r, l, 9, salt);
            rsp[r].deltaRBAng_y[l] = 0.2f * fv(r, l, 10, salt);
            rsp[r].deltaRBAng_z[l] = 0.2f * fv(r, l, 11, salt);
        }
}

static void svxSeedBatch(PxgBlockConstraintBatch& b)
{
    std::memset(&b, 0xAB, sizeof(b));
    b.mConstraintBatchIndex = 0;
    b.startConstraintIndex = SVX_START_CONSTRAINT;
    b.startFrictionIndex = SVX_START_FRICTION;
    b.mArticulationResponseIndex = SVX_START_RESP;
}

struct SvxIoBuf
{
    std::vector<float> data;   // 2 * 32 lanes * 32 floats
    void init(uint32_t salt)
    {
        data.assign(2 * 32 * 32, -555.f);
        for (uint32_t l = 0; l < SVX_LANES; ++l)
        {
            const LaneIo io = svxLaneIo(l, salt);
            float* f = &data[l * 32];
            const UnAlignedSpatialVector* vs[4] = { &io.vel0, &io.vel1, &io.d0, &io.d1 };
            for (int v = 0; v < 4; ++v)
            {
                f[v * 6 + 0] = vs[v]->top.x; f[v * 6 + 1] = vs[v]->top.y; f[v * 6 + 2] = vs[v]->top.z;
                f[v * 6 + 3] = vs[v]->bottom.x; f[v * 6 + 4] = vs[v]->bottom.y; f[v * 6 + 5] = vs[v]->bottom.z;
            }
            f[24] = io.dq0.x; f[25] = io.dq0.y; f[26] = io.dq0.z; f[27] = io.dq0.w;
            f[28] = io.dq1.x; f[29] = io.dq1.y; f[30] = io.dq1.z; f[31] = io.dq1.w;
        }
    }
};

struct SvxParamsHost
{
    uint64_t batch, h0, h1, p0, p1, responses, io;
    float elapsedTime, minPen, ref0, ref1;
    uint32_t nbLanes, accumulateError;
};

static size_t svxRunContact(PxgCudaShim& shim, float ref0, float ref1)
{
    PxgBlockConstraintBatch batch;
    svxSeedBatch(batch);

    std::vector<PxgTGSBlockSolverContactHeader> ch(1);
    std::vector<PxgTGSBlockSolverFrictionHeader> fh(1);
    std::vector<PxgTGSBlockSolverContactPoint> cp(SVX_START_CONSTRAINT + SVX_MAX_CONTACTS);
    std::vector<PxgTGSBlockSolverContactFriction> fp(SVX_START_FRICTION + SVX_MAX_FRICTIONS + 1);
    std::vector<PxgArticulationBlockResponse> rsp;
    std::memset(ch.data(), 0xAB, sizeof(ch[0]));
    std::memset(fh.data(), 0xAB, sizeof(fh[0]));
    std::memset(cp.data(), 0xAB, cp.size() * sizeof(cp[0]));
    std::memset(fp.data(), 0xAB, fp.size() * sizeof(fp[0]));
    svxSeedResponses(rsp, 700);

    uint32_t nTors = 0, nCompliant = 0, nZeroContacts = 0;
    for (uint32_t l = 0; l < 32; ++l)
    {
        const uint32_t nc = l % (SVX_MAX_CONTACTS + 1);            // 0..6
        const uint32_t tf = (l * 7 + 3) % (SVX_MAX_FRICTIONS);     // 0..5 incl odd
        ch[0].numNormalConstr[l] = nc;
        fh[0].numFrictionConstr[l] = tf;
        if (tf & 1) ++nTors;
        if (nc == 0) ++nZeroContacts;
        ch[0].invMass0_1_angDom0_1[l] = make_float4(0.5f + 0.4f * fv(l, 0, 0, 71),
                                                    0.5f + 0.4f * fv(l, 0, 1, 71),
                                                    0.5f + 0.4f * fv(l, 0, 2, 71),
                                                    0.5f + 0.4f * fv(l, 0, 3, 71));
        PxVec3 n(fv(l, 1, 0, 72), fv(l, 1, 1, 72), fv(l, 1, 2, 72));
        n = n.getNormalized();
        ch[0].normal_staticFriction[l] = make_float4(n.x, n.y, n.z, 0.6f + 0.3f * std::fabs(fv(l, 1, 3, 72)));
        ch[0].minNormalForce[l] = 0.05f * std::fabs(fv(l, 2, 0, 73));
        ch[0].maxPenBias[l] = -0.2f * std::fabs(fv(l, 2, 1, 73));
        ch[0].flags[l] = (l % 3u == 0u) ? COMPLIANT_ACCEL : 0u;
        // every fourth lane compliant (restitutionXdt < 0)
        const bool compliant = (l % 4u) == 1u;
        if (compliant) ++nCompliant;
        ch[0].restitutionXdt[l] = compliant ? -0.5f - std::fabs(fv(l, 2, 2, 73)) : 0.3f * std::fabs(fv(l, 2, 2, 73));
        ch[0].cfm[l] = 0.01f * std::fabs(fv(l, 2, 3, 73));
        ch[0].p8[l] = 0.8f + 0.1f * fv(l, 3, 0, 74);
        fh[0].dynamicFriction[l] = 0.4f + 0.3f * std::fabs(fv(l, 3, 1, 74));
        fh[0].biasCoefficient[l] = 0.5f * fv(l, 3, 2, 74);
        fh[0].torsionalFrictionScale[l] = 0.7f + 0.2f * std::fabs(fv(l, 3, 3, 74));
        PxVec3 f0(fv(l, 4, 0, 75), fv(l, 4, 1, 75), fv(l, 4, 2, 75));
        PxVec3 f1(fv(l, 5, 0, 75), fv(l, 5, 1, 75), fv(l, 5, 2, 75));
        f0 = f0.getNormalized(); f1 = f1.getNormalized();
        fh[0].frictionNormals[0][l] = make_float4(f0.x, f0.y, f0.z, 0.f);
        fh[0].frictionNormals[1][l] = make_float4(f1.x, f1.y, f1.z, 0.f);
        fh[0].broken[l] = 0;

        for (uint32_t i = 0; i < SVX_MAX_CONTACTS; ++i)
        {
            PxgTGSBlockSolverContactPoint& c = cp[SVX_START_CONSTRAINT + i];
            c.raXn_extraCoeff[l] = make_float4(fv(l, 10 + i, 0, 76), fv(l, 10 + i, 1, 76),
                                               fv(l, 10 + i, 2, 76), 0.5f + 0.4f * std::fabs(fv(l, 10 + i, 3, 76)));
            c.rbXn_targetVelW[l] = make_float4(fv(l, 20 + i, 0, 77), fv(l, 20 + i, 1, 77),
                                               fv(l, 20 + i, 2, 77), 0.2f * fv(l, 20 + i, 3, 77));
            c.separation[l] = 0.05f * fv(l, 30 + i, 0, 78);
            c.maxImpulse[l] = 0.4f + 0.3f * std::fabs(fv(l, 30 + i, 1, 78));
            c.appliedForce[l] = 0.2f * std::fabs(fv(l, 30 + i, 2, 78));
            c.biasCoefficient[l] = 0.4f * fv(l, 30 + i, 3, 78);
            c.resp0[l] = 0.4f + 0.3f * std::fabs(fv(l, 40 + i, 0, 79));
            c.resp1[l] = 0.4f + 0.3f * std::fabs(fv(l, 40 + i, 1, 79));
        }
        for (uint32_t i = 0; i < SVX_MAX_FRICTIONS + 1; ++i)
        {
            PxgTGSBlockSolverContactFriction& f = fp[SVX_START_FRICTION + i];
            f.raXn_error[l] = make_float4(fv(l, 50 + i, 0, 80), fv(l, 50 + i, 1, 80),
                                          fv(l, 50 + i, 2, 80), 0.05f * fv(l, 50 + i, 3, 80));
            f.rbXn_targetVelW[l] = make_float4(fv(l, 60 + i, 0, 81), fv(l, 60 + i, 1, 81),
                                               fv(l, 60 + i, 2, 81), 0.1f * fv(l, 60 + i, 3, 81));
            f.appliedForce[l] = 0.15f * fv(l, 70 + i, 0, 82);
            f.resp0[l] = 0.4f + 0.3f * std::fabs(fv(l, 70 + i, 1, 82));
            f.resp1[l] = 0.4f + 0.3f * std::fabs(fv(l, 70 + i, 2, 82));
        }
    }
    std::printf("[ info ] svx contact(ref %g/%g): torsional %u, compliant %u, zero-contact %u\n",
                double(ref0), double(ref1), nTors, nCompliant, nZeroContacts);
    PXG_CHECK_MSG(nTors > 5 && nCompliant > 5 && nZeroContacts > 0, "contact coverage hole");

    SvxIoBuf io;
    io.init(90);

    // mirror
    std::vector<PxgTGSBlockSolverContactHeader> wCh = ch;
    std::vector<PxgTGSBlockSolverFrictionHeader> wFh = fh;
    std::vector<PxgTGSBlockSolverContactPoint> wCp = cp;
    std::vector<PxgTGSBlockSolverContactFriction> wFp = fp;
    std::vector<float> wantIo = io.data;
    for (uint32_t l = 0; l < SVX_LANES; ++l)
    {
        LaneIo lio = svxLaneIo(l, 90);
        UnAlignedSpatialVector imp0(PxVec3(0.f), PxVec3(0.f)), imp1(PxVec3(0.f), PxVec3(0.f));
        RefError err;
        refSolveExtContactTGS(batch, lio.vel0, lio.vel1, lio.d0, lio.d1, l, wCh.data(),
                              wFh.data(), wCp.data(), wFp.data(), rsp.data(), SVX_ELAPSED,
                              SVX_MINPEN, imp0, imp1, &err, ref0, ref1);
        float* o = &wantIo[32 * 32 + l * 32];
        const UnAlignedSpatialVector* vs[4] = { &lio.vel0, &lio.vel1, &imp0, &imp1 };
        for (int v = 0; v < 4; ++v)
        {
            o[v * 6 + 0] = vs[v]->top.x; o[v * 6 + 1] = vs[v]->top.y; o[v * 6 + 2] = vs[v]->top.z;
            o[v * 6 + 3] = vs[v]->bottom.x; o[v * 6 + 4] = vs[v]->bottom.y; o[v * 6 + 5] = vs[v]->bottom.z;
        }
        o[24] = err.sumOfSquares; o[25] = float(err.counter); o[26] = err.maxError;
    }

    // device
    std::vector<PxgDevicePtr> bufs;
    auto up = [&](const void* p, size_t n) {
        PxgDevicePtr dp = 0;
        shim.memAlloc(&dp, n);
        if (dp && p) shim.memcpyHtoD(dp, p, n);
        bufs.push_back(dp);
        return dp;
    };
    PxgDevicePtr dB = up(&batch, sizeof(batch));
    PxgDevicePtr dCh = up(ch.data(), sizeof(ch[0]));
    PxgDevicePtr dFh = up(fh.data(), sizeof(fh[0]));
    PxgDevicePtr dCp = up(cp.data(), cp.size() * sizeof(cp[0]));
    PxgDevicePtr dFp = up(fp.data(), fp.size() * sizeof(fp[0]));
    PxgDevicePtr dR = up(rsp.data(), rsp.size() * sizeof(rsp[0]));
    PxgDevicePtr dIo = up(io.data.data(), io.data.size() * 4);
    if (!dB || !dCh || !dFh || !dCp || !dFp || !dR || !dIo)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
        return 0;
    }

    SvxParamsHost prm = { uint64_t(dB), uint64_t(dCh), uint64_t(dFh), uint64_t(dCp),
                          uint64_t(dFp), uint64_t(dR), uint64_t(dIo),
                          SVX_ELAPSED, SVX_MINPEN, ref0, ref1, SVX_LANES, 1u };
    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch svxContactTest");
    shim.streamSynchronize();

    size_t bad = 0;
    auto cmp = [&](PxgDevicePtr dp, const void* host, size_t n, const char* name) {
        std::vector<uint8_t> g(n);
        shim.memcpyDtoH(g.data(), dp, n);
        if (std::memcmp(g.data(), host, n) != 0)
        {
            size_t mm = 0, first = size_t(-1);
            const uint8_t* h = reinterpret_cast<const uint8_t*>(host);
            for (size_t i = 0; i < n; ++i)
                if (g[i] != h[i]) { ++mm; if (first == size_t(-1)) first = i; }
            std::printf("MM contact/%s: %zu/%zu bytes differ (first at %zu)\n", name, mm, n,
                        first);
            bad += mm;
        }
    };
    cmp(dCh, wCh.data(), sizeof(wCh[0]), "contactHeader");
    cmp(dFh, wFh.data(), sizeof(wFh[0]), "frictionHeader");
    cmp(dCp, wCp.data(), wCp.size() * sizeof(wCp[0]), "contactPoints");
    cmp(dFp, wFp.data(), wFp.size() * sizeof(wFp[0]), "frictionPoints");
    cmp(dIo, wantIo.data(), wantIo.size() * 4, "io");

    for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
    return bad;
}

static size_t svxRun1D(PxgCudaShim& shim, float ref0, float ref1)
{
    PxgBlockConstraintBatch batch;
    svxSeedBatch(batch);

    std::vector<PxgTGSBlockSolverConstraint1DHeader> hd(1);
    std::vector<PxgTGSBlockSolverConstraint1DCon> rows(SVX_START_CONSTRAINT + SVX_MAX_ROWS);
    std::vector<PxgArticulationBlockResponse> rsp;
    std::memset(hd.data(), 0xAB, sizeof(hd[0]));
    std::memset(rows.data(), 0xAB, rows.size() * sizeof(rows[0]));
    svxSeedResponses(rsp, 701);

    uint32_t nSpring = 0, nAccel = 0, nBounce = 0, nIneq = 0, nHard = 0;
    for (uint32_t l = 0; l < 32; ++l)
    {
        const uint32_t rc = (l * 5 + 1) % (SVX_MAX_ROWS + 1);   // 0..6
        uchar4 counts;
        counts.x = PxU8(rc); counts.y = 0; counts.z = 0; counts.w = 0;
        hd[0].rowCounts_breakable_orthoAxisCount[l] = counts;
        hd[0].rAWorld_invMass0D0[l] = make_float4(fv(l, 0, 0, 83), fv(l, 0, 1, 83),
                                                  fv(l, 0, 2, 83), 0.5f + 0.4f * std::fabs(fv(l, 0, 3, 83)));
        hd[0].rBWorld_invMass1D1[l] = make_float4(fv(l, 1, 0, 83), fv(l, 1, 1, 83),
                                                  fv(l, 1, 2, 83), 0.5f + 0.4f * std::fabs(fv(l, 1, 3, 83)));
        hd[0].invInertiaScale0[l] = 0.6f + 0.3f * fv(l, 2, 0, 84);
        hd[0].invInertiaScale1[l] = 0.6f + 0.3f * fv(l, 2, 1, 84);
        hd[0].cfm[l] = 0.01f * std::fabs(fv(l, 2, 2, 84));

        for (uint32_t i = 0; i < SVX_MAX_ROWS; ++i)
        {
            PxgTGSBlockSolverConstraint1DCon& c = rows[SVX_START_CONSTRAINT + i];
            // row mode: hard / inequality / spring / accel-spring / bounce
            const uint32_t mode = (l + i) % 5u;
            uint32_t flags = 0;
            float coeff2 = 0.f;   // hard: bounceVel 0
            if (mode == 1) { flags = DY_FLAG_INEQ; ++nIneq; }
            else if (mode == 2) { flags = DY_FLAG_SPRING; coeff2 = 0.4f + 0.3f * std::fabs(fv(l, i, 0, 85)); ++nSpring; }
            else if (mode == 3) { flags = DY_FLAG_SPRING | DY_FLAG_ACCEL; coeff2 = 0.4f + 0.3f * std::fabs(fv(l, i, 1, 85)); ++nAccel; }
            else if (mode == 4) { coeff2 = 0.5f * fv(l, i, 2, 85); if (coeff2 == 0.f) coeff2 = 0.25f; ++nBounce; }
            else ++nHard;

            c.lin0XYZ_initBiasOrCoeff0[l] = make_float4(fv(l, 10 + i, 0, 86), fv(l, 10 + i, 1, 86),
                                                        fv(l, 10 + i, 2, 86), 0.1f * fv(l, 10 + i, 3, 86));
            c.lin1XYZ_biasScaleOrCoeff1[l] = make_float4(fv(l, 20 + i, 0, 86), fv(l, 20 + i, 1, 86),
                                                         fv(l, 20 + i, 2, 86), -0.4f * std::fabs(fv(l, 20 + i, 3, 86)));
            c.ang0XYZ_velMultiplierOrCoeff2[l] = make_float4(fv(l, 30 + i, 0, 86), fv(l, 30 + i, 1, 86),
                                                             fv(l, 30 + i, 2, 86), coeff2);
            c.ang1XYZ_velTargetOrCoeff3[l] = make_float4(fv(l, 40 + i, 0, 86), fv(l, 40 + i, 1, 86),
                                                         fv(l, 40 + i, 2, 86), 0.2f * fv(l, 40 + i, 3, 86));
            c.flags[l] = flags;
            c.resp0[l] = 0.4f + 0.3f * std::fabs(fv(l, 50 + i, 0, 87));
            c.resp1[l] = 0.4f + 0.3f * std::fabs(fv(l, 50 + i, 1, 87));
            c.geometricError[l] = 0.1f * fv(l, 50 + i, 2, 87);
            c.minImpulse[l] = -0.6f - 0.3f * std::fabs(fv(l, 60 + i, 0, 88));
            c.maxImpulse[l] = 0.6f + 0.3f * std::fabs(fv(l, 60 + i, 1, 88));
            c.maxBias[l] = 0.3f + 0.2f * std::fabs(fv(l, 60 + i, 2, 88));
            c.angularErrorScale[l] = (l + i) % 2 ? 1.f : 0.f;
            c.appliedForce[l] = 0.2f * fv(l, 60 + i, 3, 88);
            c.residual[l] = 0.f;
        }
    }
    std::printf("[ info ] svx 1d(ref %g/%g): hard %u, ineq %u, spring %u, accel %u, bounce %u\n",
                double(ref0), double(ref1), nHard, nIneq, nSpring, nAccel, nBounce);
    PXG_CHECK_MSG(nHard > 5 && nIneq > 5 && nSpring > 5 && nAccel > 5 && nBounce > 5,
                  "1d row-mode coverage hole");

    SvxIoBuf io;
    io.init(91);

    // mirror
    std::vector<PxgTGSBlockSolverConstraint1DCon> wRows = rows;
    std::vector<float> wantIo = io.data;
    for (uint32_t l = 0; l < SVX_LANES; ++l)
    {
        LaneIo lio = svxLaneIo(l, 91);
        UnAlignedSpatialVector imp0(PxVec3(0.f), PxVec3(0.f)), imp1(PxVec3(0.f), PxVec3(0.f));
        refSolveExt1DTGS(batch, lio.vel0, lio.vel1, lio.d0, lio.d1, l, hd.data(),
                         wRows.data(), rsp.data(), lio.dq0, lio.dq1, SVX_ELAPSED, imp0, imp1,
                         true, ref0, ref1);
        float* o = &wantIo[32 * 32 + l * 32];
        const UnAlignedSpatialVector* vs[4] = { &lio.vel0, &lio.vel1, &imp0, &imp1 };
        for (int v = 0; v < 4; ++v)
        {
            o[v * 6 + 0] = vs[v]->top.x; o[v * 6 + 1] = vs[v]->top.y; o[v * 6 + 2] = vs[v]->top.z;
            o[v * 6 + 3] = vs[v]->bottom.x; o[v * 6 + 4] = vs[v]->bottom.y; o[v * 6 + 5] = vs[v]->bottom.z;
        }
        o[24] = 0.f; o[25] = 0.f; o[26] = 0.f;   // the 1D entry writes a zero error
    }

    std::vector<PxgDevicePtr> bufs;
    auto up = [&](const void* p, size_t n) {
        PxgDevicePtr dp = 0;
        shim.memAlloc(&dp, n);
        if (dp && p) shim.memcpyHtoD(dp, p, n);
        bufs.push_back(dp);
        return dp;
    };
    PxgDevicePtr dB = up(&batch, sizeof(batch));
    PxgDevicePtr dHd = up(hd.data(), sizeof(hd[0]));
    PxgDevicePtr dRows = up(rows.data(), rows.size() * sizeof(rows[0]));
    PxgDevicePtr dR = up(rsp.data(), rsp.size() * sizeof(rsp[0]));
    PxgDevicePtr dIo = up(io.data.data(), io.data.size() * 4);
    if (!dB || !dHd || !dRows || !dR || !dIo)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
        return 0;
    }

    SvxParamsHost prm = { uint64_t(dB), uint64_t(dHd), uint64_t(dRows), 0, 0,
                          uint64_t(dR), uint64_t(dIo),
                          SVX_ELAPSED, SVX_MINPEN, ref0, ref1, SVX_LANES, 1u };
    PXG_CHECK_MSG(shim.launchKernel(1, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch svx1DTest");
    shim.streamSynchronize();

    size_t bad = 0;
    auto cmp = [&](PxgDevicePtr dp, const void* host, size_t n, const char* name) {
        std::vector<uint8_t> g(n);
        shim.memcpyDtoH(g.data(), dp, n);
        if (std::memcmp(g.data(), host, n) != 0)
        {
            size_t mm = 0, first = size_t(-1);
            const uint8_t* h = reinterpret_cast<const uint8_t*>(host);
            for (size_t i = 0; i < n; ++i)
                if (g[i] != h[i]) { ++mm; if (first == size_t(-1)) first = i; }
            std::printf("MM 1d/%s: %zu/%zu bytes differ (first at %zu)\n", name, mm, n, first);
            bad += mm;
        }
    };
    cmp(dHd, hd.data(), sizeof(hd[0]), "header");         // header is read-only
    cmp(dRows, wRows.data(), wRows.size() * sizeof(wRows[0]), "rows");
    cmp(dIo, wantIo.data(), wantIo.size() * 4, "io");

    for (PxgDevicePtr p : bufs) if (p) shim.memFree(p);
    return bad;
}

} // namespace

PXG_TEST(artiSolveExtTGS)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiSolveExtTGS.spv", "svxContactTest", 80u, 0 },
        { "npArtiSolveExtTGS.spv", "svx1DTest", 80u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 2, &svxLoader);

    size_t bad = 0;
    bad += svxRunContact(shim, 1.0f, 1.0f);
    bad += svxRunContact(shim, 0.75f, 1.25f);
    bad += svxRun1D(shim, 1.0f, 1.0f);
    bad += svxRun1D(shim, 0.75f, 1.25f);
    PXG_CHECK_MSG(bad == 0, "%zu mismatches", bad);
}
