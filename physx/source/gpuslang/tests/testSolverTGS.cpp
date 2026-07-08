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

// Verification of solverTGS.slang (solveBlockPartitionTGS) against a C++
// transcription of solverBlockTGS.cuh: TGS contact solve with accumulated
// deltas, paired friction with torsional anchors, and the 1D joint solve
// with delta-transform re-anchoring and ortho-axis projection.

#include "pxgTestFramework.h"
#include "pxgGpuStructs.h"
#include "pxgRefMath.h"

#include <numeric>
#include <random>

namespace
{

using namespace pxgtest;

const uint32_t DY_SC_FLAG_OUTPUT_FORCE = 1u << 1;
const uint32_t DY_SC_FLAG_KEEP_BIAS = 1u << 2;
const uint32_t DY_SC_FLAG_ORTHO_TARGET = 1u << 4;
const uint32_t DY_SC_FLAG_SPRING = 1u << 5;
const uint32_t DY_SC_FLAG_INEQUALITY = 1u << 6;
const uint32_t COMPLIANT_ACCEL_SPRING = 1u << 3;
const float PX_MAX_F32V = 3.4028234663852885981170418348452e+38f;

// ---- TGS block mirrors ----

struct BatchTGS
{
    uint16_t descStride;
    uint16_t constraintType;
    uint32_t scalars[7];
    uint32_t pad0[24];
    uint64_t nodeA[32];
    uint64_t nodeB[32];
    uint32_t bodyAIndex[32];
    uint32_t bodyBIndex[32];
    uint32_t remappedBodyAIndex[32];
    uint32_t remappedBodyBIndex[32];
    uint32_t slabId[32];
    uint64_t shapeInteraction[32];
};
static_assert(sizeof(BatchTGS) == 1536, "batch mirror");

struct TGSContactHeader
{
    Float4 invMass0_1_angDom0_1[32];
    Float4 normal_staticFriction[32];
    float minNormalForce[32];
    float maxPenBias[32];
    uint32_t flags[32];
    uint32_t numNormalConstr[32];
    uint32_t forceWritebackOffset[32];
    float restitutionXdt[32];
    float cfm[32];
    float p8[32];
};
static_assert(sizeof(TGSContactHeader) == 2048, "TGS contact header mirror");

struct TGSFrictionHeader
{
    Float4 frictionNormals0[32];
    Float4 frictionNormals1[32];
    uint32_t numFrictionConstr[32];
    float dynamicFriction[32];
    uint32_t broken[32];
    float biasCoefficient[32];
    float torsionalFrictionScale[32];
};
static_assert(sizeof(TGSFrictionHeader) == 1664, "TGS friction header mirror");

struct TGSContactPoint
{
    Float4 raXn_extraCoeff[32];
    Float4 rbXn_targetVelW[32];
    float separation[32];
    float maxImpulse[32];
    float appliedForce[32];
    float biasCoefficient[32];
    float resp0[32];
    float resp1[32];
};
static_assert(sizeof(TGSContactPoint) == 1792, "TGS contact point mirror");

struct TGSFriction
{
    Float4 raXn_error[32];
    Float4 rbXn_targetVelW[32];
    float appliedForce[32];
    float resp0[32];
    float resp1[32];
};
static_assert(sizeof(TGSFriction) == 1408, "TGS friction mirror");

struct TGSJointHeader
{
    uint32_t rowCounts_breakable_orthoAxisCount[32]; // uchar4
    Float4 rAWorld_invMass0D0[32];
    Float4 rBWorld_invMass1D1[32];
    float invInertiaScale0[32];
    float invInertiaScale1[32];
    uint32_t writeBackOffset[32];
    Float4 angOrthoAxis0_recipResponseW[3][32];
    Float4 angOrthoAxis1_ErrorW[3][32];
    float linBreakImpulse[32];
    float angBreakImpulse[32];
    float cfm[32];
};
static_assert(sizeof(TGSJointHeader) == 4992, "TGS joint header mirror");

struct TGSJointRowCon
{
    Float4 lin0XYZ_initBiasOrCoeff0[32];
    Float4 lin1XYZ_biasScaleOrCoeff1[32];
    Float4 ang0XYZ_velMultiplierOrCoeff2[32];
    Float4 ang1XYZ_velTargetOrCoeff3[32];
    float resp0[32];
    float resp1[32];
    float geometricError[32];
    float minImpulse[32];
    float maxImpulse[32];
    float maxBias[32];
    float angularErrorScale[32];
    uint32_t flags[32];
    float appliedForce[32];
    float residual[32];
};
static_assert(sizeof(TGSJointRowCon) == 3328, "TGS joint row mirror");

struct SolverTGSParams
{
    uint32_t startIndex;
    uint32_t endIndex;
    uint32_t residualReportingEnabled;
    uint32_t numSlabs;
    uint32_t numTotalBodies;
    uint32_t bodyOffset;
    float elapsedTime;
    float minPen;
    // conclude/writeback/averaging kernels:
    uint32_t totalBodyCount;
    uint32_t deltaVOffset;
    uint32_t averageOutputOffsets;
    uint32_t isVelocityIteration;
    float invDt;
    float stepDt;
};

struct ThresholdElement
{
    uint64_t shapeInteraction;
    float normalForce;
    float threshold;
    uint64_t nodeIndexA;
    uint64_t nodeIndexB;
    float accumulatedForce;
    uint32_t pad;
};
static_assert(sizeof(ThresholdElement) == 40, "threshold element mirror");

struct ConstraintWriteback
{
    Float4 linearImpulse_broken;
    Float4 angularImpulse_residual;
};

struct FrictionPatchGPU
{
    Float3 points[2];
    Float3 impulses[2];
    uint32_t anchors;
};
static_assert(sizeof(FrictionPatchGPU) == 52, "friction patch mirror");

struct BlockFrictionPatch
{
    Float4 body0Normal[32];
    Float4 body1Normal[32];
    uint32_t anchorCount[32];
    uint32_t broken[32];
    uint32_t contactID[2][32];
    Float4 anchorPoints[2][32];
    uint32_t patchIndex[32];
    uint32_t pad0[32];
};
static_assert(sizeof(BlockFrictionPatch) == 2816, "block friction patch mirror");

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------

struct BodyStateTGS
{
    Vec3 lv0, av0, ld0, ad0;
    Vec3 lv1, av1, ld1, ad1;
};

void refCompliantTGS(uint32_t flags, float nrdt, float unitResponse, float recipResponse, float a,
                     float& velMultiplier, float& scaledBias)
{
    const bool accelSpring = (flags & COMPLIANT_ACCEL_SPRING) != 0;
    const float massIfAccelElseOne = accelSpring ? recipResponse : 1.0f;
    const float oneIfAccelElseR = accelSpring ? 1.0f : unitResponse;
    const float x = 1.f / (a * oneIfAccelElseR + 1.f);
    velMultiplier = x * a * massIfAccelElseOne;
    scaledBias = nrdt * x * oneIfAccelElseR;
}

void refSolveContactTGS(const BatchTGS& b, BodyStateTGS& s, uint32_t lane,
                        std::vector<TGSContactHeader>& chs, std::vector<TGSFrictionHeader>& fhs,
                        std::vector<TGSContactPoint>& cps, std::vector<TGSFriction>& frs,
                        float elapsedTime, float minPen, float ref0, float ref1)
{
    const uint32_t bi = b.scalars[1]; // constraintBatchIndex
    const uint32_t sc = b.scalars[2]; // startConstraintIndex
    const uint32_t sf = b.scalars[3]; // startFrictionIndex

    TGSContactHeader& ch = chs[bi];
    TGSFrictionHeader& fh = fhs[bi];

    const uint32_t numNormal = ch.numNormalConstr[lane];
    const uint32_t numFricTotal = fh.numFrictionConstr[lane];
    const uint32_t numFric = numFricTotal & ~1u;

    float accumulatedNormalImpulse = 0.f;
    float accumDeltaF = 0.f;

    if (!numNormal)
        return;

    const float maxPenBias = ch.maxPenBias[lane];
    const Float4 im = ch.invMass0_1_angDom0_1[lane];
    const float invMassA = ref0 * im.x, invMassB = ref1 * im.y;
    const float angDom0 = ref0 * im.z, angDom1 = ref1 * im.w;
    const float sumInvMass = invMassA + invMassB;

    const Float4 nsf = ch.normal_staticFriction[lane];
    const Vec3 normal(nsf.x, nsf.y, nsf.z);
    const float staticFrictionCof = nsf.w;
    const float restitutionXdt = ch.restitutionXdt[lane];
    const float p8 = ch.p8[lane];
    const uint32_t flags = ch.flags[lane];

    const Vec3 delLinVel0 = normal * invMassA;
    const Vec3 delLinVel1 = normal * invMassB;

    const Vec3 relMotion = s.ld0 - s.ld1;
    const float deltaV = normal.dot(relMotion);
    float relVel1 = (s.lv0 - s.lv1).dot(normal);

    for (uint32_t i = 0; i < numNormal; ++i)
    {
        TGSContactPoint& c = cps[sc + i];
        const Float4 ra = c.raXn_extraCoeff[lane];
        const Float4 rb = c.rbXn_targetVelW[lane];
        const float appliedForce = c.appliedForce[lane];
        const float separation = c.separation[lane];
        const float maxImpulse = c.maxImpulse[lane];

        const float resp0 = ref0 * c.resp0[lane], resp1 = ref1 * c.resp1[lane];
        const float unitResponse = resp0 + resp1;
        const float recipResponse = unitResponse > 0.f ? 1.f / unitResponse : 0.f;

        float velMultiplier = recipResponse;
        float biasCoefficient = c.biasCoefficient[lane];
        if (restitutionXdt < 0.f)
            refCompliantTGS(flags, restitutionXdt, unitResponse, recipResponse, ra.w, velMultiplier,
                            biasCoefficient);

        const Vec3 raXn(ra.x, ra.y, ra.z), rbXn(rb.x, rb.y, rb.z);
        const float targetVel = rb.w;

        const float v0 = s.av0.dot(raXn), v1 = s.av1.dot(rbXn);
        const float normalVel = relVel1 + (v0 - v1);

        const float sep = std::fmax(minPen, separation + deltaV + s.ad0.dot(raXn) - s.ad1.dot(rbXn));
        const float tVelErr = targetVel * elapsedTime;
        const float biasedErr = recipResponse * std::fmin(-maxPenBias, biasCoefficient * (sep - tVelErr));

        const float tempDeltaF = biasedErr - (normalVel - targetVel) * velMultiplier;
        const float _deltaF = std::fmax(tempDeltaF, -appliedForce);
        const float newForce = std::fmin(appliedForce + _deltaF, maxImpulse);
        const float deltaF = newForce - appliedForce;

        accumDeltaF += deltaF;
        relVel1 += sumInvMass * deltaF;

        s.av0 += raXn * (deltaF * angDom0);
        s.av1 = s.av1 - rbXn * (deltaF * angDom1);

        c.appliedForce[lane] = newForce;
        accumulatedNormalImpulse += newForce;
    }

    s.lv0 += delLinVel0 * accumDeltaF;
    s.lv1 = s.lv1 - delLinVel1 * accumDeltaF;

    if (numFricTotal)
    {
        const float biasCoefficient = fh.biasCoefficient[lane];
        const float dynCof = fh.dynamicFriction[lane];
        const float maxFrictionImpulse = staticFrictionCof * accumulatedNormalImpulse;
        const float maxDynFrictionImpulse = dynCof * accumulatedNormalImpulse;
        uint32_t broken = 0;

        const Float4 fn0 = fh.frictionNormals0[lane];
        const Float4 fn1 = fh.frictionNormals1[lane];
        const Vec3 normal0(fn0.x, fn0.y, fn0.z), normal1(fn1.x, fn1.y, fn1.z);

        const float relDelta0 = relMotion.dot(normal0);
        const float relDelta1 = relMotion.dot(normal1);

        for (uint32_t i = 0; i < numFric; i += 2)
        {
            TGSFriction& f0 = frs[sf + i];
            TGSFriction& f1 = frs[sf + i + 1];

            const Float4 ra0 = f0.raXn_error[lane], rb0 = f0.rbXn_targetVelW[lane];
            const Float4 ra1 = f1.raXn_error[lane], rb1 = f1.rbXn_targetVelW[lane];
            const float af0 = f0.appliedForce[lane], af1 = f1.appliedForce[lane];
            const float tv0 = rb0.w, tv1 = rb1.w;

            const float r0 = ref0 * f0.resp0[lane] + ref1 * f0.resp1[lane];
            const float vm0 = r0 > 0.f ? p8 / r0 : 0.f;
            const float r1 = ref0 * f1.resp0[lane] + ref1 * f1.resp1[lane];
            const float vm1 = r1 > 0.f ? p8 / r1 : 0.f;

            const Vec3 raXn0(ra0.x, ra0.y, ra0.z), rbXn0(rb0.x, rb0.y, rb0.z);
            const float nv0 = (s.av0.dot(raXn0) + s.lv0.dot(normal0)) - (s.av1.dot(rbXn0) + s.lv1.dot(normal0));
            const float err0 = ra0.w - tv0 * elapsedTime + raXn0.dot(s.ad0) - rbXn0.dot(s.ad1) + relDelta0;
            const float ti0 = (af0 - (err0 * biasCoefficient - tv0) * vm0) - nv0 * vm0;

            const Vec3 raXn1(ra1.x, ra1.y, ra1.z), rbXn1(rb1.x, rb1.y, rb1.z);
            const float nv1 = (s.av0.dot(raXn1) + s.lv0.dot(normal1)) - (s.av1.dot(rbXn1) + s.lv1.dot(normal1));
            const float err1 = ra1.w - tv1 * elapsedTime + raXn1.dot(s.ad0) - rbXn1.dot(s.ad1) + relDelta1;
            const float ti1 = (af1 - (err1 * biasCoefficient - tv1) * vm1) - nv1 * vm1;

            const float total = std::sqrt(ti0 * ti0 + ti1 * ti1);
            const bool clampI = total > maxFrictionImpulse;
            const float ratio = clampI ? std::fmin(maxDynFrictionImpulse, total) / total : 1.f;

            const float na0 = ti0 * ratio, na1 = ti1 * ratio;
            const float d0 = na0 - af0, d1 = na1 - af1;

            s.lv0 += normal0 * invMassA * d0;
            s.lv1 = s.lv1 - normal0 * invMassB * d0;
            s.av0 += raXn0 * (d0 * angDom0);
            s.av1 = s.av1 - rbXn0 * (d0 * angDom1);

            s.lv0 += normal1 * invMassA * d1;
            s.lv1 = s.lv1 - normal1 * invMassB * d1;
            s.av0 += raXn1 * (d1 * angDom0);
            s.av1 = s.av1 - rbXn1 * (d1 * angDom1);

            f0.appliedForce[lane] = na0;
            f1.appliedForce[lane] = na1;
            broken |= clampI ? 1u : 0u;
        }

        if (numFric < numFricTotal)
        {
            // torsional anchor
            const float fscale = fh.torsionalFrictionScale[lane];
            TGSFriction& f0 = frs[sf + numFric];
            const Float4 ra0 = f0.raXn_error[lane], rb0 = f0.rbXn_targetVelW[lane];
            const float af0 = f0.appliedForce[lane];
            const float tv0 = rb0.w;
            const float r0 = ref0 * f0.resp0[lane] + ref1 * f0.resp1[lane];
            const float vm0 = r0 > 0.f ? p8 / r0 : 0.f;

            const Vec3 raXn0(ra0.x, ra0.y, ra0.z), rbXn0(rb0.x, rb0.y, rb0.z);
            const float nv0 = s.av0.dot(raXn0) - s.av1.dot(rbXn0);
            const float total = (af0 - (-tv0) * vm0) - nv0 * vm0;

            const bool clampI = std::fabs(total) > maxFrictionImpulse * fscale;
            const float clamped = std::fmin(maxDynFrictionImpulse * fscale,
                                            std::fmax(-maxDynFrictionImpulse * fscale, total));
            const float na = clampI ? clamped : total;
            const float d = na - af0;

            s.av0 += raXn0 * (d * angDom0);
            s.av1 = s.av1 - rbXn0 * (d * angDom1);
            f0.appliedForce[lane] = na;
            broken |= clampI ? 1u : 0u;
        }

        fh.broken[lane] = broken;
    }
}

void refSolve1DTGS(const BatchTGS& b, BodyStateTGS& s, uint32_t lane,
                   std::vector<TGSJointHeader>& jhs, std::vector<TGSJointRowCon>& rows,
                   const SolverTxIData& iData0, const SolverTxIData& iData1, float elapsedTime,
                   bool residual, float ref0, float ref1)
{
    const uint32_t bi = b.scalars[1];
    const uint32_t sc = b.scalars[2];
    TGSJointHeader& h = jhs[bi];

    const Float4 raIm = h.rAWorld_invMass0D0[lane];
    const Float4 rbIm = h.rBWorld_invMass1D1[lane];
    const float invMass0 = ref0 * raIm.w, invMass1 = ref1 * rbIm.w;

    const Vec3 raPrev(raIm.x, raIm.y, raIm.z), rbPrev(rbIm.x, rbIm.y, rbIm.z);
    const Vec3 ra = QuatV(iData0.deltaBody2World.q).rotate(raPrev);
    const Vec3 rb = QuatV(iData1.deltaBody2World.q).rotate(rbPrev);
    const Vec3 raMotion = (ra + s.ld0) - raPrev;
    const Vec3 rbMotion = (rb + s.ld1) - rbPrev;

    const float invInertiaScale0 = ref0 * h.invInertiaScale0[lane];
    const float invInertiaScale1 = ref1 * h.invInertiaScale1[lane];

    const uint32_t rowCount = h.rowCounts_breakable_orthoAxisCount[lane] & 0xFFu;

    Vec3 a0o[3], a1o[3];
    float rr[3], er[3];
    for (int k = 0; k < 3; ++k)
    {
        const Float4 v0 = h.angOrthoAxis0_recipResponseW[k][lane];
        const Float4 v1 = h.angOrthoAxis1_ErrorW[k][lane];
        a0o[k] = Vec3(v0.x, v0.y, v0.z);
        a1o[k] = Vec3(v1.x, v1.y, v1.z);
        rr[k] = v0.w;
        er[k] = v1.w + (a0o[k].dot(s.ad0) - a1o[k].dot(s.ad1));
    }

    const Mat33 sqrtInvInertia0 = Mat33::load(iData0.sqrtInvInertia);
    const Mat33 sqrtInvInertia1 = Mat33::load(iData1.sqrtInvInertia);

    for (uint32_t i = 0; i < rowCount; ++i)
    {
        TGSJointRowCon& c = rows[sc + i];

        const Float4 l0 = c.lin0XYZ_initBiasOrCoeff0[lane];
        const Float4 l1 = c.lin1XYZ_biasScaleOrCoeff1[lane];
        const Float4 a0 = c.ang0XYZ_velMultiplierOrCoeff2[lane];
        const Float4 a1 = c.ang1XYZ_velTargetOrCoeff3[lane];

        const Vec3 clin0(l0.x, l0.y, l0.z), clin1(l1.x, l1.y, l1.z);
        float initBias = l0.w;
        const float biasScale = l1.w;
        const float velMultiplier = a0.w;
        const float targetVel = a1.w;

        const Vec3 cang0 = Vec3(a0.x, a0.y, a0.z) + ra.cross(clin0);
        const Vec3 cang1 = Vec3(a1.x, a1.y, a1.z) + rb.cross(clin1);

        const uint32_t flags = c.flags[lane];
        const float maxBias = c.maxBias[lane];
        const float minBias = -((flags & DY_SC_FLAG_INEQUALITY) ? PX_MAX_F32V : maxBias);

        Vec3 raXnI = sqrtInvInertia0.transform(cang0);
        Vec3 rbXnI = sqrtInvInertia1.transform(cang1);

        if (flags & DY_SC_FLAG_ORTHO_TARGET)
        {
            const float p0 = (raXnI.dot(a0o[0]) + rbXnI.dot(a1o[0])) * rr[0];
            const float p1 = (raXnI.dot(a0o[1]) + rbXnI.dot(a1o[1])) * rr[1];
            const float p2 = (raXnI.dot(a0o[2]) + rbXnI.dot(a1o[2])) * rr[2];
            raXnI = raXnI - (a0o[0] * p0 + a0o[1] * p1 + a0o[2] * p2);
            rbXnI = rbXnI - (a1o[0] * p0 + a1o[1] * p1 + a1o[2] * p2);
            initBias -= biasScale * (er[0] * p0 + er[1] * p1 + er[2] * p2);
        }

        initBias = (biasScale == 0.f) ? 0.f : initBias;

        const bool isSpring = (flags & DY_SC_FLAG_SPRING) != 0;

        const float deltaAng = (raXnI.dot(s.ad0) - rbXnI.dot(s.ad1)) * c.angularErrorScale[lane];
        const float deltaLin = clin0.dot(raMotion) - clin1.dot(rbMotion);
        const float motion = deltaLin + deltaAng;
        const float errorDelta = isSpring ? motion : motion - targetVel * elapsedTime;

        const float resp0 = invMass0 * clin0.dot(clin0) + raXnI.dot(raXnI) * invInertiaScale0;
        const float resp1 = invMass1 * clin1.dot(clin1) + rbXnI.dot(rbXnI) * invInertiaScale1;
        const float response = resp0 + resp1;
        const float recipResponse = response > 0.f ? 1.f / response : 0.f;

        const float vMul = isSpring ? velMultiplier : recipResponse * velMultiplier;
        const float bias = std::fmin(maxBias, std::fmax(minBias, initBias + errorDelta * biasScale));
        const float constant = isSpring ? bias + targetVel : recipResponse * (bias + targetVel);

        const float appliedForce = c.appliedForce[lane];
        const float v0 = s.lv0.dot(clin0) + s.av0.dot(raXnI);
        const float v1 = s.lv1.dot(clin1) + s.av1.dot(rbXnI);
        const float normalVel = v0 - v1;

        const float unclampedForce = appliedForce + (vMul * normalVel + constant);
        const float clampedForce = std::fmin(c.maxImpulse[lane], std::fmax(c.minImpulse[lane], unclampedForce));
        const float deltaF = clampedForce - appliedForce;

        c.appliedForce[lane] = clampedForce;
        if (residual)
            c.residual[lane] = vMul == 0.f ? 0.f : deltaF / vMul;

        s.lv0 += clin0 * (deltaF * invMass0);
        s.lv1 = s.lv1 - clin1 * (deltaF * invMass1);
        s.av0 += raXnI * deltaF * invInertiaScale0;
        s.av1 = s.av1 - rbXnI * deltaF * invInertiaScale1;
    }
}

// ---------------------------------------------------------------------------
// test
// ---------------------------------------------------------------------------

Vec3 randUnitV(std::mt19937& rng)
{
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    Vec3 v(d(rng), d(rng), d(rng));
    return v.magnitude() > 1e-3f ? v.getNormalized() : Vec3(1, 0, 0);
}

QuatV randomQuatTGS(std::mt19937& rng)
{
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    QuatV q(dist(rng) * 0.1f, dist(rng) * 0.1f, dist(rng) * 0.1f, 1.f);
    return q.getNormalized();
}

PXG_TEST(solverTGS_contactAndJointBatches)
{
    std::mt19937 rng(3300);
    std::uniform_real_distribution<float> unit(0.f, 1.f);
    std::uniform_real_distribution<float> sym(-2.f, 2.f);

    const uint32_t numBatches = 20;
    const uint32_t rowsPerBatch = 4;
    const uint32_t numSlabs = 12;
    const uint32_t numTotalBodies = 2048;
    const uint32_t bodyOffset = 16;
    const float elapsedTime = 0.005f;
    const float minPen = -0.02f;

    std::vector<BatchTGS> batches(numBatches);
    std::vector<TGSContactHeader> chs(numBatches);
    std::vector<TGSFrictionHeader> fhs(numBatches);
    std::vector<TGSContactPoint> cps(numBatches * rowsPerBatch);
    std::vector<TGSFriction> frs(numBatches * (rowsPerBatch + 1));
    std::vector<TGSJointHeader> jhs(numBatches);
    std::vector<TGSJointRowCon> jrows(numBatches * rowsPerBatch);
    std::vector<SolverTxIData> txi(numTotalBodies);
    std::memset(batches.data(), 0, batches.size() * sizeof(BatchTGS));
    std::memset(chs.data(), 0, chs.size() * sizeof(TGSContactHeader));
    std::memset(fhs.data(), 0, fhs.size() * sizeof(TGSFrictionHeader));
    std::memset(cps.data(), 0, cps.size() * sizeof(TGSContactPoint));
    std::memset(frs.data(), 0, frs.size() * sizeof(TGSFriction));
    std::memset(jhs.data(), 0, jhs.size() * sizeof(TGSJointHeader));
    std::memset(jrows.data(), 0, jrows.size() * sizeof(TGSJointRowCon));
    std::memset(txi.data(), 0, txi.size() * sizeof(SolverTxIData));

    for (SolverTxIData& t : txi)
    {
        t.deltaBody2World.q = randomQuatTGS(rng).toQuat();
        t.deltaBody2World.p = Float3{ sym(rng) * 0.01f, sym(rng) * 0.01f, sym(rng) * 0.01f };
        // random symmetric sqrt inv inertia
        const Mat33 rot(randomQuatTGS(rng));
        const Vec3 d(unit(rng), unit(rng), unit(rng));
        const Mat33 m = rot * Mat33(Vec3(d.x, 0, 0), Vec3(0, d.y, 0), Vec3(0, 0, d.z)) * rot.getTranspose();
        m.store(t.sqrtInvInertia);
    }

    std::vector<Float4> velPool(numBatches * 192);
    for (Float4& v : velPool)
        v = Float4{ sym(rng), sym(rng), sym(rng), sym(rng) };

    std::vector<uint32_t> encoded(numTotalBodies);
    for (uint32_t& e : encoded)
        e = rng();

    for (uint32_t k = 0; k < numBatches; ++k)
    {
        BatchTGS& b = batches[k];
        b.descStride = uint16_t(1 + rng() % 32);
        b.constraintType = uint16_t(k % 2 == 0 ? 0 : 1);
        b.scalars[1] = k;                       // constraintBatchIndex
        b.scalars[2] = k * rowsPerBatch;        // startConstraintIndex
        b.scalars[3] = k * (rowsPerBatch + 1);  // startFrictionIndex

        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            b.bodyAIndex[lane] = rng() % numTotalBodies;
            b.bodyBIndex[lane] = rng() % numTotalBodies;
            b.remappedBodyAIndex[lane] = k * 192 + lane;
            b.remappedBodyBIndex[lane] = k * 192 + lane + 96;

            if (b.constraintType == 0)
            {
                TGSContactHeader& h = chs[k];
                h.invMass0_1_angDom0_1[lane] = Float4{ unit(rng), unit(rng), unit(rng), unit(rng) };
                const Vec3 n = randUnitV(rng);
                h.normal_staticFriction[lane] = Float4{ n.x, n.y, n.z, unit(rng) };
                h.maxPenBias[lane] = -unit(rng) * 5.f;
                h.flags[lane] = unit(rng) < 0.5f ? COMPLIANT_ACCEL_SPRING : 0;
                h.numNormalConstr[lane] = rng() % (rowsPerBatch + 1);
                h.restitutionXdt[lane] = unit(rng) < 0.5f ? 0.f : -unit(rng);
                h.p8[lane] = 0.8f;

                TGSFrictionHeader& fh = fhs[k];
                // 0, 2, 4 or 2+torsional / 4+torsional friction rows
                const uint32_t fr = h.numNormalConstr[lane] ? (rng() % 3) * 2 + (unit(rng) < 0.3f ? 1 : 0) : 0;
                fh.numFrictionConstr[lane] = std::min(fr, rowsPerBatch + 1);
                fh.dynamicFriction[lane] = unit(rng);
                fh.biasCoefficient[lane] = unit(rng) * 10.f;
                fh.torsionalFrictionScale[lane] = 0.5f + unit(rng);
                const Vec3 f0 = randUnitV(rng), f1 = randUnitV(rng);
                fh.frictionNormals0[lane] = Float4{ f0.x, f0.y, f0.z, 0 };
                fh.frictionNormals1[lane] = Float4{ f1.x, f1.y, f1.z, 0 };

                for (uint32_t i = 0; i < rowsPerBatch; ++i)
                {
                    TGSContactPoint& c = cps[b.scalars[2] + i];
                    const Vec3 ra = Vec3(sym(rng), sym(rng), sym(rng)) * 0.3f;
                    const Vec3 rb = Vec3(sym(rng), sym(rng), sym(rng)) * 0.3f;
                    c.raXn_extraCoeff[lane] = Float4{ ra.x, ra.y, ra.z, unit(rng) * 0.3f };
                    c.rbXn_targetVelW[lane] = Float4{ rb.x, rb.y, rb.z, sym(rng) * 0.05f };
                    c.separation[lane] = sym(rng) * 0.05f;
                    c.maxImpulse[lane] = 10.f + unit(rng) * 100.f;
                    c.appliedForce[lane] = unit(rng);
                    c.biasCoefficient[lane] = unit(rng) * 20.f;
                    c.resp0[lane] = 0.1f + unit(rng);
                    c.resp1[lane] = 0.1f + unit(rng);
                }
                for (uint32_t i = 0; i < rowsPerBatch + 1; ++i)
                {
                    TGSFriction& f = frs[b.scalars[3] + i];
                    const Vec3 ra = Vec3(sym(rng), sym(rng), sym(rng)) * 0.3f;
                    const Vec3 rb = Vec3(sym(rng), sym(rng), sym(rng)) * 0.3f;
                    f.raXn_error[lane] = Float4{ ra.x, ra.y, ra.z, sym(rng) * 0.02f };
                    f.rbXn_targetVelW[lane] = Float4{ rb.x, rb.y, rb.z, sym(rng) * 0.05f };
                    f.appliedForce[lane] = sym(rng) * 0.2f;
                    f.resp0[lane] = 0.1f + unit(rng);
                    f.resp1[lane] = 0.1f + unit(rng);
                }
            }
            else
            {
                TGSJointHeader& h = jhs[k];
                h.rowCounts_breakable_orthoAxisCount[lane] = rng() % (rowsPerBatch + 1); // uchar4.x
                const Vec3 raw = Vec3(sym(rng), sym(rng), sym(rng)) * 0.3f;
                const Vec3 rbw = Vec3(sym(rng), sym(rng), sym(rng)) * 0.3f;
                h.rAWorld_invMass0D0[lane] = Float4{ raw.x, raw.y, raw.z, unit(rng) };
                h.rBWorld_invMass1D1[lane] = Float4{ rbw.x, rbw.y, rbw.z, unit(rng) };
                h.invInertiaScale0[lane] = unit(rng);
                h.invInertiaScale1[lane] = unit(rng);
                for (int kk = 0; kk < 3; ++kk)
                {
                    const Vec3 o0 = randUnitV(rng), o1 = randUnitV(rng);
                    h.angOrthoAxis0_recipResponseW[kk][lane] = Float4{ o0.x, o0.y, o0.z, unit(rng) * 0.5f };
                    h.angOrthoAxis1_ErrorW[kk][lane] = Float4{ o1.x, o1.y, o1.z, sym(rng) * 0.02f };
                }

                for (uint32_t i = 0; i < rowsPerBatch; ++i)
                {
                    TGSJointRowCon& c = jrows[b.scalars[2] + i];
                    const Vec3 l0 = randUnitV(rng), l1 = randUnitV(rng);
                    const Vec3 a0 = Vec3(sym(rng), sym(rng), sym(rng)) * 0.4f;
                    const Vec3 a1 = Vec3(sym(rng), sym(rng), sym(rng)) * 0.4f;
                    c.lin0XYZ_initBiasOrCoeff0[lane] = Float4{ l0.x, l0.y, l0.z, sym(rng) * 0.05f };
                    c.lin1XYZ_biasScaleOrCoeff1[lane] =
                        Float4{ l1.x, l1.y, l1.z, unit(rng) < 0.2f ? 0.f : unit(rng) * 5.f };
                    c.ang0XYZ_velMultiplierOrCoeff2[lane] = Float4{ a0.x, a0.y, a0.z, 0.5f + unit(rng) * 0.5f };
                    c.ang1XYZ_velTargetOrCoeff3[lane] = Float4{ a1.x, a1.y, a1.z, sym(rng) * 0.05f };
                    c.minImpulse[lane] = -50.f - unit(rng) * 50.f;
                    c.maxImpulse[lane] = 50.f + unit(rng) * 50.f;
                    c.maxBias[lane] = unit(rng) * 10.f;
                    c.angularErrorScale[lane] = unit(rng);
                    uint32_t fl = 0;
                    if (unit(rng) < 0.4f)
                        fl |= DY_SC_FLAG_SPRING;
                    if (unit(rng) < 0.4f)
                        fl |= DY_SC_FLAG_ORTHO_TARGET;
                    if (unit(rng) < 0.3f)
                        fl |= DY_SC_FLAG_INEQUALITY;
                    c.flags[lane] = fl;
                    c.appliedForce[lane] = sym(rng) * 0.2f;
                }
            }
        }
    }

    // ---- GPU ----
    PxgSlangComputeDevice* dev = testDevice();

    Kernel kernel("solverTGS.spv", "solveBlockPartitionTGS",
                  { sizeof(BatchTGS), sizeof(TGSContactHeader), sizeof(TGSFrictionHeader),
                    sizeof(TGSContactPoint), sizeof(TGSFriction), sizeof(TGSJointHeader),
                    sizeof(TGSJointRowCon), sizeof(SolverTxIData), 16, 4 },
                  sizeof(SolverTGSParams));

    PxgBufferHandle batchBuf = createBufferFrom(dev, batches);
    PxgBufferHandle chBuf = createBufferFrom(dev, chs);
    PxgBufferHandle fhBuf = createBufferFrom(dev, fhs);
    PxgBufferHandle cpBuf = createBufferFrom(dev, cps);
    PxgBufferHandle frBuf = createBufferFrom(dev, frs);
    PxgBufferHandle jhBuf = createBufferFrom(dev, jhs);
    PxgBufferHandle jrBuf = createBufferFrom(dev, jrows);
    PxgBufferHandle txBuf = createBufferFrom(dev, txi);
    PxgBufferHandle velBuf = createBufferFrom(dev, velPool);
    PxgBufferHandle encBuf = createBufferFrom(dev, encoded);

    const SolverTGSParams params = { 0, numBatches, 1u, numSlabs, numTotalBodies, bodyOffset,
                                     elapsedTime, minPen };
    kernel.dispatch(numBatches, { batchBuf, chBuf, fhBuf, cpBuf, frBuf, jhBuf, jrBuf, txBuf, velBuf, encBuf },
                    params);
    dev->flush();

    // ---- reference ----
    std::vector<Float4> refVel = velPool;
    auto popcount = [](uint32_t v) { uint32_t c = 0; while (v) { c += v & 1; v >>= 1; } return c; };

    for (uint32_t k = 0; k < numBatches; ++k)
    {
        const BatchTGS& b = batches[k];
        for (uint32_t lane = 0; lane < b.descStride; ++lane)
        {
            const uint32_t ri = k * 192 + lane;
            const Float4* v = &refVel[0];
            BodyStateTGS s;
            s.lv0 = Vec3(v[ri].x, v[ri].y, v[ri].z);
            s.av0 = Vec3(v[ri].w, v[ri + 32].x, v[ri + 32].y);
            s.ld0 = Vec3(v[ri + 32].z, v[ri + 32].w, v[ri + 64].x);
            s.ad0 = Vec3(v[ri + 64].y, v[ri + 64].z, v[ri + 64].w);
            s.lv1 = Vec3(v[ri + 96].x, v[ri + 96].y, v[ri + 96].z);
            s.av1 = Vec3(v[ri + 96].w, v[ri + 128].x, v[ri + 128].y);
            s.ld1 = Vec3(v[ri + 128].z, v[ri + 128].w, v[ri + 160].x);
            s.ad1 = Vec3(v[ri + 160].y, v[ri + 160].z, v[ri + 160].w);

            float ref0 = 1.f, ref1 = 1.f;
            if (b.bodyAIndex[lane] >= bodyOffset)
                ref0 = float(std::max(1u, popcount(encoded[b.bodyAIndex[lane]])));
            if (b.bodyBIndex[lane] >= bodyOffset)
                ref1 = float(std::max(1u, popcount(encoded[b.bodyBIndex[lane]])));

            if (b.constraintType == 0)
                refSolveContactTGS(b, s, lane, chs, fhs, cps, frs, elapsedTime, minPen, ref0, ref1);
            else
                refSolve1DTGS(b, s, lane, jhs, jrows, txi[b.bodyAIndex[lane]], txi[b.bodyBIndex[lane]],
                              elapsedTime, true, ref0, ref1);

            const uint32_t oa = b.remappedBodyAIndex[lane];
            const uint32_t ob = b.remappedBodyBIndex[lane];
            refVel[oa] = Float4{ s.lv0.x, s.lv0.y, s.lv0.z, s.av0.x };
            refVel[oa + 32] = Float4{ s.av0.y, s.av0.z, s.ld0.x, s.ld0.y };
            refVel[oa + 64] = Float4{ s.ld0.z, s.ad0.x, s.ad0.y, s.ad0.z };
            refVel[ob] = Float4{ s.lv1.x, s.lv1.y, s.lv1.z, s.av1.x };
            refVel[ob + 32] = Float4{ s.av1.y, s.av1.z, s.ld1.x, s.ld1.y };
            refVel[ob + 64] = Float4{ s.ld1.z, s.ad1.x, s.ad1.y, s.ad1.z };
        }
    }

    // ---- compare ----
    const auto gpuVel = downloadBuffer<Float4>(dev, velBuf, velPool.size());
    const auto gpuCps = downloadBuffer<TGSContactPoint>(dev, cpBuf, cps.size());
    const auto gpuFrs = downloadBuffer<TGSFriction>(dev, frBuf, frs.size());
    const auto gpuFhs = downloadBuffer<TGSFrictionHeader>(dev, fhBuf, fhs.size());
    const auto gpuJrows = downloadBuffer<TGSJointRowCon>(dev, jrBuf, jrows.size());

    size_t mismatches = 0;
    for (size_t i = 0; i < refVel.size(); ++i)
    {
        const float* g = &gpuVel[i].x;
        const float* r = &refVel[i].x;
        for (int c = 0; c < 4; ++c)
            if (!almostEqual(g[c], r[c], 2e-4f, 2e-4f))
            {
                if (mismatches < 5)
                    PXG_CHECK_MSG(false, "vel[%zu].%d: gpu=%g ref=%g", i, c, g[c], r[c]);
                ++mismatches;
            }
    }
    PXG_CHECK_MSG(mismatches == 0, "%zu velocity-pool mismatches", mismatches);

    mismatches = 0;
    for (size_t i = 0; i < cps.size(); ++i)
        for (int l = 0; l < 32; ++l)
            if (!almostEqual(gpuCps[i].appliedForce[l], cps[i].appliedForce[l], 2e-4f, 2e-4f))
                ++mismatches;
    for (size_t i = 0; i < frs.size(); ++i)
        for (int l = 0; l < 32; ++l)
            if (!almostEqual(gpuFrs[i].appliedForce[l], frs[i].appliedForce[l], 2e-4f, 2e-4f))
                ++mismatches;
    for (size_t i = 0; i < jrows.size(); ++i)
        for (int l = 0; l < 32; ++l)
        {
            if (!almostEqual(gpuJrows[i].appliedForce[l], jrows[i].appliedForce[l], 2e-4f, 2e-4f))
                ++mismatches;
            if (!almostEqual(gpuJrows[i].residual[l], jrows[i].residual[l], 1e-3f, 1e-3f))
                ++mismatches;
        }
    for (size_t i = 0; i < fhs.size(); ++i)
        for (int l = 0; l < 32; ++l)
            if (gpuFhs[i].broken[l] != fhs[i].broken[l])
                ++mismatches;
    PXG_CHECK_MSG(mismatches == 0, "%zu applied-force/residual/broken mismatches", mismatches);

    for (PxgBufferHandle h : { batchBuf, chBuf, fhBuf, cpBuf, frBuf, jhBuf, jrBuf, txBuf, velBuf, encBuf })
        dev->destroyBuffer(h);
}

// ---------------------------------------------------------------------------
// concludeBlocksTGS + writebackBlocksTGS
// ---------------------------------------------------------------------------
PXG_TEST(solverTGS_concludeAndWriteback)
{
    std::mt19937 rng(3400);
    std::uniform_real_distribution<float> unit(0.f, 1.f);
    std::uniform_real_distribution<float> sym(-2.f, 2.f);

    const uint32_t numBatches = 12;
    const uint32_t rowsPerBatch = 4;
    const uint32_t numBodies = 64;

    std::vector<BatchTGS> batches(numBatches);
    std::vector<TGSContactHeader> chs(numBatches);
    std::vector<TGSFrictionHeader> fhs(numBatches);
    std::vector<TGSContactPoint> cps(numBatches * rowsPerBatch);
    std::vector<TGSFriction> frs(numBatches * (rowsPerBatch + 1));
    std::vector<TGSJointHeader> jhs(numBatches);
    std::vector<TGSJointRowCon> jrows(numBatches * rowsPerBatch);
    std::vector<BlockFrictionPatch> patchBlocks(numBatches);
    std::memset(batches.data(), 0, batches.size() * sizeof(BatchTGS));
    std::memset(chs.data(), 0, chs.size() * sizeof(TGSContactHeader));
    std::memset(fhs.data(), 0, fhs.size() * sizeof(TGSFrictionHeader));
    std::memset(cps.data(), 0, cps.size() * sizeof(TGSContactPoint));
    std::memset(frs.data(), 0, frs.size() * sizeof(TGSFriction));
    std::memset(jhs.data(), 0, jhs.size() * sizeof(TGSJointHeader));
    std::memset(jrows.data(), 0, jrows.size() * sizeof(TGSJointRowCon));
    std::memset(patchBlocks.data(), 0, patchBlocks.size() * sizeof(BlockFrictionPatch));

    std::vector<SolverBodyData> bodyData(numBodies);
    std::memset(bodyData.data(), 0, bodyData.size() * sizeof(SolverBodyData));
    for (uint32_t b = 0; b < numBodies; ++b)
    {
        bodyData[b].reportThreshold = unit(rng) < 0.5f ? unit(rng) * 5.f : PX_MAX_F32V;
        bodyData[b].islandNodeIndex = makeNodeIndex(rng() % 1000);
    }

    const uint32_t numPatches = numBatches * 32;
    std::vector<FrictionPatchGPU> patches(numPatches);
    std::memset(patches.data(), 0, patches.size() * sizeof(FrictionPatchGPU));

    uint32_t forceBufCursor = 0;
    uint32_t writebackCursor = 0;

    for (uint32_t k = 0; k < numBatches; ++k)
    {
        BatchTGS& b = batches[k];
        b.descStride = uint16_t(1 + rng() % 32);
        b.constraintType = uint16_t(k % 2);
        b.scalars[1] = k;
        b.scalars[2] = k * rowsPerBatch;
        b.scalars[3] = k * (rowsPerBatch + 1);

        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            b.bodyAIndex[lane] = rng() % numBodies;
            b.bodyBIndex[lane] = rng() % numBodies;
            b.shapeInteraction[lane] = (uint64_t(rng()) << 32) | rng();

            if (b.constraintType == 0)
            {
                chs[k].numNormalConstr[lane] = rng() % (rowsPerBatch + 1);
                chs[k].flags[lane] = unit(rng) < 0.6f ? 1u /*eHAS_FORCE_THRESHOLDS*/ : 0u;
                chs[k].forceWritebackOffset[lane] =
                    unit(rng) < 0.8f ? (forceBufCursor += rowsPerBatch) - rowsPerBatch : 0xFFFFFFFFu;
                // 0, 2, 4 rows, optionally + torsional anchor
                const uint32_t fr =
                    chs[k].numNormalConstr[lane] ? (rng() % 3) * 2 + (unit(rng) < 0.3f ? 1 : 0) : 0;
                fhs[k].numFrictionConstr[lane] = std::min(fr, rowsPerBatch + 1);
                fhs[k].broken[lane] = rng() & 1;
                const Vec3 f0 = randUnitV(rng), f1 = randUnitV(rng);
                fhs[k].frictionNormals0[lane] = Float4{ f0.x, f0.y, f0.z, 0 };
                fhs[k].frictionNormals1[lane] = Float4{ f1.x, f1.y, f1.z, 0 };
                patchBlocks[k].patchIndex[lane] = unit(rng) < 0.7f ? k * 32 + lane : 0xFFFFFFFFu;
                patchBlocks[k].anchorPoints[0][lane] = Float4{ sym(rng), sym(rng), sym(rng), 0 };
                patchBlocks[k].anchorPoints[1][lane] = Float4{ sym(rng), sym(rng), sym(rng), 0 };

                for (uint32_t i = 0; i < rowsPerBatch; ++i)
                    cps[b.scalars[2] + i].appliedForce[lane] = unit(rng) * 3.f;
                for (uint32_t i = 0; i < rowsPerBatch + 1; ++i)
                    frs[b.scalars[3] + i].appliedForce[lane] = sym(rng);
            }
            else
            {
                const uint32_t rows = rng() % (rowsPerBatch + 1);
                const uint32_t breakable = rng() & 1;
                jhs[k].rowCounts_breakable_orthoAxisCount[lane] = rows | (breakable << 8);
                jhs[k].linBreakImpulse[lane] = unit(rng) * 2.f;
                jhs[k].angBreakImpulse[lane] = unit(rng) * 2.f;
                jhs[k].rAWorld_invMass0D0[lane] =
                    Float4{ sym(rng) * 0.2f, sym(rng) * 0.2f, sym(rng) * 0.2f, unit(rng) };
                jhs[k].writeBackOffset[lane] = unit(rng) < 0.8f ? writebackCursor++ : 0xFFFFFFFFu;

                for (uint32_t i = 0; i < rowsPerBatch; ++i)
                {
                    TGSJointRowCon& c = jrows[b.scalars[2] + i];
                    const Vec3 l0 = randUnitV(rng);
                    c.lin0XYZ_initBiasOrCoeff0[lane] = Float4{ l0.x, l0.y, l0.z, sym(rng) };
                    c.lin1XYZ_biasScaleOrCoeff1[lane] = Float4{ sym(rng), sym(rng), sym(rng), sym(rng) };
                    c.ang0XYZ_velMultiplierOrCoeff2[lane] =
                        Float4{ sym(rng) * 0.3f, sym(rng) * 0.3f, sym(rng) * 0.3f, sym(rng) };
                    c.ang1XYZ_velTargetOrCoeff3[lane] = Float4{ sym(rng), sym(rng), sym(rng), sym(rng) };
                    c.appliedForce[lane] = sym(rng);
                    c.residual[lane] = sym(rng) * 0.05f;
                    c.flags[lane] = (unit(rng) < 0.7f ? DY_SC_FLAG_OUTPUT_FORCE : 0u) |
                                    (unit(rng) < 0.4f ? DY_SC_FLAG_SPRING : 0u) |
                                    (unit(rng) < 0.4f ? DY_SC_FLAG_KEEP_BIAS : 0u);
                }
            }
        }
    }

    PxgSlangComputeDevice* dev = testDevice();

    const std::vector<uint32_t> strides = {
        sizeof(BatchTGS), sizeof(TGSContactHeader), sizeof(TGSFrictionHeader), sizeof(TGSContactPoint),
        sizeof(TGSFriction), sizeof(TGSJointHeader), sizeof(TGSJointRowCon), sizeof(SolverTxIData), 16, 4,
        sizeof(SolverBodyData), sizeof(ConstraintWriteback), 4, 40, 4, sizeof(BlockFrictionPatch),
        sizeof(FrictionPatchGPU)
    };

    Kernel conclude("solverTGS.spv", "concludeBlocksTGS",
                    { strides[0], strides[1], strides[2], strides[3], strides[4], strides[5], strides[6] },
                    sizeof(SolverTGSParams));
    Kernel writeback("solverTGS.spv", "writebackBlocksTGS", strides, sizeof(SolverTGSParams));

    PxgBufferHandle batchBuf = createBufferFrom(dev, batches);
    PxgBufferHandle chBuf = createBufferFrom(dev, chs);
    PxgBufferHandle fhBuf = createBufferFrom(dev, fhs);
    PxgBufferHandle cpBuf = createBufferFrom(dev, cps);
    PxgBufferHandle frBuf = createBufferFrom(dev, frs);
    PxgBufferHandle jhBuf = createBufferFrom(dev, jhs);
    PxgBufferHandle jrBuf = createBufferFrom(dev, jrows);
    PxgBufferHandle bdBuf = createBufferFrom(dev, bodyData);
    std::vector<ConstraintWriteback> wbInit(std::max(writebackCursor, 1u));
    std::memset(wbInit.data(), 0, wbInit.size() * sizeof(ConstraintWriteback));
    PxgBufferHandle wbBuf = createBufferFrom(dev, wbInit);
    std::vector<float> forceInit(std::max(forceBufCursor, 1u), 0.f);
    PxgBufferHandle forceBuf = createBufferFrom(dev, forceInit);
    PxgBufferHandle tsBuf = dev->createBuffer(numBatches * 32 * sizeof(ThresholdElement));
    std::vector<uint32_t> zero1 = { 0 };
    PxgBufferHandle tsIdxBuf = createBufferFrom(dev, zero1);
    PxgBufferHandle fpbBuf = createBufferFrom(dev, patchBlocks);
    PxgBufferHandle fpBuf = createBufferFrom(dev, patches);

    SolverTGSParams params = {};
    params.startIndex = 0;
    params.endIndex = numBatches;

    conclude.dispatch(numBatches, { batchBuf, chBuf, fhBuf, cpBuf, frBuf, jhBuf, jrBuf }, params);
    dev->barrier();
    writeback.dispatch(numBatches,
                       { batchBuf, chBuf, fhBuf, cpBuf, frBuf, jhBuf, jrBuf, bdBuf /*dummy at 7*/,
                         bdBuf /*dummy at 8*/, bdBuf /*dummy at 9*/, bdBuf, wbBuf, forceBuf, tsBuf,
                         tsIdxBuf, fpbBuf, fpBuf },
                       params);
    dev->flush();

    // ---- reference ----
    auto refRows = jrows;
    auto refPatchBlocks = patchBlocks;
    std::vector<float> refForce = forceInit;
    std::vector<ConstraintWriteback> refWb = wbInit;
    std::vector<FrictionPatchGPU> refPatches = patches;
    std::vector<ThresholdElement> refStream;

    for (uint32_t k = 0; k < numBatches; ++k)
    {
        const BatchTGS& b = batches[k];
        for (uint32_t lane = 0; lane < b.descStride; ++lane)
        {
            if (b.constraintType == 0)
            {
                // concludeContactBlockTGS is "#if 0" in CUDA: no contact changes.
                // writeBackContactBlockTGS
                const uint32_t nc = chs[k].numNormalConstr[lane];
                const uint32_t nf = fhs[k].numFrictionConstr[lane];
                float normalForce = 0.f;
                const uint32_t fwo = chs[k].forceWritebackOffset[lane];
                if (fwo != 0xFFFFFFFFu)
                    for (uint32_t i = 0; i < nc; ++i)
                    {
                        const float af = cps[b.scalars[2] + i].appliedForce[lane];
                        refForce[fwo + i] = af;
                        normalForce += af;
                    }

                const uint32_t patchIndex = refPatchBlocks[k].patchIndex[lane];
                if (patchIndex != 0xFFFFFFFFu)
                {
                    FrictionPatchGPU& fp = refPatches[patchIndex];
                    const Float4 a0 = fhs[k].frictionNormals0[lane];
                    const Float4 a1 = fhs[k].frictionNormals1[lane];
                    fp.anchors = nf / 2;
                    if (nf >= 2)
                    {
                        const Float4 an = refPatchBlocks[k].anchorPoints[0][lane];
                        fp.points[0] = Float3{ an.x, an.y, an.z };
                        const float i0 = frs[b.scalars[3]].appliedForce[lane];
                        const float i1 = frs[b.scalars[3] + 1].appliedForce[lane];
                        fp.impulses[0] = Float3{ a0.x * i0 + a1.x * i1, a0.y * i0 + a1.y * i1,
                                                 a0.z * i0 + a1.z * i1 };
                    }
                    if (nf >= 4)
                    {
                        const Float4 an = refPatchBlocks[k].anchorPoints[1][lane];
                        fp.points[1] = Float3{ an.x, an.y, an.z };
                        const float i0 = frs[b.scalars[3] + 2].appliedForce[lane];
                        const float i1 = frs[b.scalars[3] + 3].appliedForce[lane];
                        fp.impulses[1] = Float3{ a0.x * i0 + a1.x * i1, a0.y * i0 + a1.y * i1,
                                                 a0.z * i0 + a1.z * i1 };
                    }
                }
                if (nf && fhs[k].broken[lane])
                    refPatchBlocks[k].broken[lane] = 1;

                const float rt0 = bodyData[b.bodyAIndex[lane]].reportThreshold;
                const float rt1 = bodyData[b.bodyBIndex[lane]].reportThreshold;
                const bool forceThreshold = (chs[k].flags[lane] & 1u) != 0;
                if (forceThreshold && normalForce != 0.f && (rt0 < PX_MAX_F32V || rt1 < PX_MAX_F32V))
                {
                    ThresholdElement e = {};
                    e.normalForce = normalForce;
                    e.threshold = std::fmin(rt0, rt1);
                    e.nodeIndexA = bodyData[b.bodyAIndex[lane]].islandNodeIndex;
                    e.nodeIndexB = bodyData[b.bodyBIndex[lane]].islandNodeIndex;
                    if (e.nodeIndexB < e.nodeIndexA)
                        std::swap(e.nodeIndexA, e.nodeIndexB);
                    e.shapeInteraction = b.shapeInteraction[lane];
                    refStream.push_back(e);
                }
            }
            else
            {
                // conclude1DBlockTGS
                const uint32_t rows = jhs[k].rowCounts_breakable_orthoAxisCount[lane] & 0xFFu;
                for (uint32_t i = 0; i < rows; ++i)
                {
                    TGSJointRowCon& c = refRows[b.scalars[2] + i];
                    if (!(c.flags[lane] & DY_SC_FLAG_KEEP_BIAS))
                        c.lin1XYZ_biasScaleOrCoeff1[lane].w = 0.f;
                    if (c.flags[lane] & DY_SC_FLAG_SPRING)
                    {
                        c.lin0XYZ_initBiasOrCoeff0[lane].w = 0.f;
                        c.lin1XYZ_biasScaleOrCoeff1[lane].w = 0.f;
                        c.ang0XYZ_velMultiplierOrCoeff2[lane].w = 0.f;
                        c.ang1XYZ_velTargetOrCoeff3[lane].w = 0.f;
                    }
                }

                // writeBack1DBlockTGS
                const uint32_t fwo = jhs[k].writeBackOffset[lane];
                const uint32_t breakable = (jhs[k].rowCounts_breakable_orthoAxisCount[lane] >> 8) & 0xFFu;
                if (fwo != 0xFFFFFFFFu)
                {
                    Vec3 linVel(0.f), angVel(0.f);
                    float errSq = 0.f;
                    for (uint32_t i = 0; i < rows; ++i)
                    {
                        const TGSJointRowCon& c = refRows[b.scalars[2] + i];
                        if (c.flags[lane] & DY_SC_FLAG_OUTPUT_FORCE)
                        {
                            const Float4 l0 = c.lin0XYZ_initBiasOrCoeff0[lane];
                            const Float4 a0 = c.ang0XYZ_velMultiplierOrCoeff2[lane];
                            linVel += Vec3(l0.x, l0.y, l0.z) * c.appliedForce[lane];
                            angVel += Vec3(a0.x, a0.y, a0.z) * c.appliedForce[lane];
                        }
                        errSq += c.residual[lane] * c.residual[lane];
                    }
                    const Float4 rAW = jhs[k].rAWorld_invMass0D0[lane];
                    angVel = angVel - Vec3(rAW.x, rAW.y, rAW.z).cross(linVel);
                    const uint32_t broken = breakable
                        ? uint32_t(linVel.magnitude() > jhs[k].linBreakImpulse[lane] ||
                                   angVel.magnitude() > jhs[k].angBreakImpulse[lane])
                        : 0;
                    refWb[fwo].angularImpulse_residual = Float4{ angVel.x, angVel.y, angVel.z, errSq };
                    refWb[fwo].linearImpulse_broken =
                        Float4{ linVel.x, linVel.y, linVel.z, broken ? -0.0f : 0.0f };
                }
            }
        }
    }

    // ---- compare ----
    const auto gpuRows = downloadBuffer<TGSJointRowCon>(dev, jrBuf, jrows.size());
    const auto gpuCps = downloadBuffer<TGSContactPoint>(dev, cpBuf, cps.size());
    const auto gpuForce = downloadBuffer<float>(dev, forceBuf, refForce.size());
    const auto gpuWb = downloadBuffer<ConstraintWriteback>(dev, wbBuf, refWb.size());
    const auto gpuPatches = downloadBuffer<FrictionPatchGPU>(dev, fpBuf, patches.size());
    const auto gpuPatchBlocks = downloadBuffer<BlockFrictionPatch>(dev, fpbBuf, patchBlocks.size());
    const auto gpuTsCount = downloadBuffer<uint32_t>(dev, tsIdxBuf, 1)[0];
    auto gpuStream = downloadBuffer<ThresholdElement>(dev, tsBuf, gpuTsCount);

    size_t mismatches = 0;
    for (size_t i = 0; i < jrows.size(); ++i)
        for (int l = 0; l < 32; ++l)
        {
            const Float4* g[] = { &gpuRows[i].lin0XYZ_initBiasOrCoeff0[l],
                                  &gpuRows[i].lin1XYZ_biasScaleOrCoeff1[l],
                                  &gpuRows[i].ang0XYZ_velMultiplierOrCoeff2[l],
                                  &gpuRows[i].ang1XYZ_velTargetOrCoeff3[l] };
            const Float4* r[] = { &refRows[i].lin0XYZ_initBiasOrCoeff0[l],
                                  &refRows[i].lin1XYZ_biasScaleOrCoeff1[l],
                                  &refRows[i].ang0XYZ_velMultiplierOrCoeff2[l],
                                  &refRows[i].ang1XYZ_velTargetOrCoeff3[l] };
            for (int c = 0; c < 4; ++c)
                if (std::memcmp(g[c], r[c], sizeof(Float4)) != 0)
                    ++mismatches;
        }
    // contact conclude is a no-op: the contact coefficients must be untouched
    for (size_t i = 0; i < cps.size(); ++i)
        for (int l = 0; l < 32; ++l)
            if (gpuCps[i].biasCoefficient[l] != cps[i].biasCoefficient[l] ||
                gpuCps[i].appliedForce[l] != cps[i].appliedForce[l])
                ++mismatches;
    PXG_CHECK_MSG(mismatches == 0, "conclude: %zu mismatches", mismatches);

    mismatches = 0;
    for (size_t i = 0; i < refForce.size(); ++i)
        if (gpuForce[i] != refForce[i])
            ++mismatches;
    for (size_t i = 0; i < refWb.size(); ++i)
        if (std::memcmp(&gpuWb[i], &refWb[i], sizeof(ConstraintWriteback)) != 0)
            ++mismatches;
    for (size_t i = 0; i < patches.size(); ++i)
        if (std::memcmp(&gpuPatches[i], &refPatches[i], sizeof(FrictionPatchGPU)) != 0)
            ++mismatches;
    for (size_t i = 0; i < patchBlocks.size(); ++i)
        for (int l = 0; l < 32; ++l)
            if (gpuPatchBlocks[i].broken[l] != refPatchBlocks[i].broken[l])
                ++mismatches;
    PXG_CHECK_MSG(mismatches == 0, "writeback: %zu mismatches", mismatches);

    // threshold stream: order is nondeterministic -> sort both sides
    PXG_CHECK_MSG(gpuTsCount == refStream.size(), "threshold stream count: gpu=%u ref=%zu", gpuTsCount,
                  refStream.size());
    auto tsLess = [](const ThresholdElement& a, const ThresholdElement& b) {
        return std::memcmp(&a, &b, sizeof(ThresholdElement)) < 0;
    };
    std::sort(gpuStream.begin(), gpuStream.end(), tsLess);
    std::sort(refStream.begin(), refStream.end(), tsLess);
    mismatches = 0;
    for (size_t i = 0; i < refStream.size() && i < gpuStream.size(); ++i)
        if (std::memcmp(&gpuStream[i], &refStream[i], sizeof(ThresholdElement)) != 0)
            ++mismatches;
    PXG_CHECK_MSG(mismatches == 0, "threshold stream: %zu mismatches", mismatches);

    for (PxgBufferHandle h : { batchBuf, chBuf, fhBuf, cpBuf, frBuf, jhBuf, jrBuf, bdBuf, wbBuf, forceBuf,
                               tsBuf, tsIdxBuf, fpbBuf, fpBuf })
        dev->destroyBuffer(h);
}

// ---------------------------------------------------------------------------
// computeAverageSolverBodyVelocityTGS + propagateAverageSolverBodyVelocityTGS
// + writeBackBodiesTGS
// ---------------------------------------------------------------------------
PXG_TEST(solverTGS_velocityAveragingAndPropagate)
{
    std::mt19937 rng(3500);
    std::uniform_real_distribution<float> unit(0.f, 1.f);
    std::uniform_real_distribution<float> sym(-3.f, 3.f);

    for (uint32_t isVelocityIteration : { 0u, 1u })
    for (uint32_t numSlabs : { 4u, 12u, 40u })
    {
        const uint32_t totalBodyCount = 70; // dynamic bodies
        const uint32_t bodyOffset = 8;      // world body + kinematics
        const uint32_t T = totalBodyCount + bodyOffset;
        const uint32_t numTotalBodies = T;  // no articulations
        const uint32_t numBatches = 6;
        const float stepDt = 1.f / 240.f;
        const float invDt = 60.f;

        const uint32_t averageOutputOffsets = numBatches * 32 * 3 * 2;
        // three float4 per slab slot, grouped in clusters of 32
        const uint32_t slabRegion = ((totalBodyCount * numSlabs + 31) / 32) * 96 + 96;
        const uint32_t deltaVOffset = averageOutputOffsets + slabRegion;
        const uint32_t propBase = deltaVOffset + 3 * T;
        const uint32_t poolSize = propBase + slabRegion;

        std::vector<Float4> velPool(poolSize);
        for (Float4& v : velPool)
            v = Float4{ sym(rng), sym(rng), sym(rng), sym(rng) };

        // remapped write slots: unique (r, r+32, r+64) triples per slab slot in
        // the cluster-of-32 pattern, in a dedicated pool region (partitions
        // keep these race-free in the real solver)
        std::vector<uint32_t> references(totalBodyCount * numSlabs, 0xFFFFFFFFu);
        for (uint32_t i = 0; i < references.size(); ++i)
            if ((rng() & 3) != 0) // ~75% of slab slots active
                references[i] = propBase + 3 * (i & ~31u) + (i & 31u);

        const uint32_t num32Slabs = (numSlabs + 31) / 32;
        std::vector<uint32_t> encoded(numTotalBodies * num32Slabs);
        for (uint32_t& e : encoded)
            e = rng();

        // per-body delta poses + sqrt inertia, node indices and lock flags
        std::vector<SolverTxIData> txi(T);
        std::memset(txi.data(), 0, txi.size() * sizeof(SolverTxIData));
        for (SolverTxIData& t : txi)
        {
            t.deltaBody2World.q = randomQuatTGS(rng).toQuat();
            t.deltaBody2World.p = Float3{ sym(rng) * 0.01f, sym(rng) * 0.01f, sym(rng) * 0.01f };
            const Mat33 rot(randomQuatTGS(rng));
            const Vec3 d(unit(rng), unit(rng), unit(rng));
            const Mat33 m = rot * Mat33(Vec3(d.x, 0, 0), Vec3(0, d.y, 0), Vec3(0, 0, d.z)) * rot.getTranspose();
            m.store(t.sqrtInvInertia);
        }

        const uint32_t numBodySims = 128;
        std::vector<BodySim> bodySims(numBodySims);
        std::memset(bodySims.data(), 0, bodySims.size() * sizeof(BodySim));
        for (BodySim& s : bodySims)
            s.lockFlags = uint16_t(unit(rng) < 0.3f ? rng() % 64 : 0);

        std::vector<SolverBodyData> bodyData(T);
        std::memset(bodyData.data(), 0, bodyData.size() * sizeof(SolverBodyData));
        for (SolverBodyData& d : bodyData)
            d.islandNodeIndex = makeNodeIndex(rng() % numBodySims);

        std::vector<Float4> motionVel(2 * T);
        for (Float4& v : motionVel)
            v = Float4{ sym(rng), sym(rng), sym(rng), sym(rng) };

        PxgSlangComputeDevice* dev = testDevice();

        const std::vector<uint32_t> strides = { 1536, 2048, 1664, 1792, 1408, 4992, 3328,
                                                sizeof(SolverTxIData), 16, 4, sizeof(SolverBodyData), 32,
                                                4, 40, 4, 2816, 52, 16, 4, sizeof(BodySim) };

        Kernel average("solverTGS.spv", "computeAverageSolverBodyVelocityTGS",
                       std::vector<uint32_t>(strides.begin(), strides.begin() + 19),
                       sizeof(SolverTGSParams));
        Kernel propagate("solverTGS.spv", "propagateAverageSolverBodyVelocityTGS", strides,
                         sizeof(SolverTGSParams));
        Kernel writeBackBodies("solverTGS.spv", "writeBackBodiesTGS",
                               std::vector<uint32_t>(strides.begin(), strides.begin() + 18),
                               sizeof(SolverTGSParams));

        std::vector<uint32_t> dummy(16, 0);
        PxgBufferHandle dummyBuf = createBufferFrom(dev, dummy);
        PxgBufferHandle velBuf = createBufferFrom(dev, velPool);
        PxgBufferHandle encBuf = createBufferFrom(dev, encoded);
        PxgBufferHandle refBuf = createBufferFrom(dev, references);
        PxgBufferHandle txBuf = createBufferFrom(dev, txi);
        PxgBufferHandle bdBuf = createBufferFrom(dev, bodyData);
        PxgBufferHandle bsBuf = createBufferFrom(dev, bodySims);
        PxgBufferHandle mvBuf = createBufferFrom(dev, motionVel);

        SolverTGSParams params = {};
        params.numSlabs = numSlabs;
        params.numTotalBodies = numTotalBodies;
        params.bodyOffset = bodyOffset;
        params.totalBodyCount = totalBodyCount;
        params.deltaVOffset = deltaVOffset;
        params.averageOutputOffsets = averageOutputOffsets;
        params.isVelocityIteration = isVelocityIteration;
        params.invDt = invDt;
        params.stepDt = stepDt;

        const uint32_t pow2 = [&] {
            uint32_t x = numSlabs;
            if ((x & (x - 1)) == 0)
                return x;
            x--; x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16;
            return x + 1;
        }();
        const uint32_t threadsPerBody = std::min(pow2, 32u);
        const uint32_t bodiesPerWarp = 32 / threadsPerBody;
        const uint32_t avgGroups = (totalBodyCount + bodiesPerWarp - 1) / bodiesPerWarp;
        const uint32_t propGroups = (T + bodiesPerWarp - 1) / bodiesPerWarp;

        std::vector<PxgBufferHandle> bufs = { dummyBuf, dummyBuf, dummyBuf, dummyBuf, dummyBuf, dummyBuf,
                                              dummyBuf, txBuf, velBuf, encBuf, bdBuf, dummyBuf, dummyBuf,
                                              dummyBuf, dummyBuf, dummyBuf, dummyBuf, mvBuf, refBuf, bsBuf };
        average.dispatch(avgGroups, std::vector<PxgBufferHandle>(bufs.begin(), bufs.begin() + 19), params);
        dev->barrier();
        propagate.dispatch(propGroups, bufs, params);
        dev->barrier();
        writeBackBodies.dispatch((totalBodyCount + 127) / 128,
                                 std::vector<PxgBufferHandle>(bufs.begin(), bufs.begin() + 18), params);
        dev->flush();

        // ---- reference ----
        std::vector<Float4> refVel = velPool;
        std::vector<uint32_t> refEnc = encoded;
        std::vector<SolverTxIData> refTxi = txi;
        std::vector<Float4> refMotion = motionVel;
        auto popcount = [](uint32_t v) { uint32_t c = 0; while (v) { c += v & 1; v >>= 1; } return c; };

        // average pass (dynamic bodies only)
        for (uint32_t bodyId = 0; bodyId < totalBodyCount; ++bodyId)
        {
            const Float4 vel0 = velPool[bodyOffset + deltaVOffset + bodyId];
            const Float4 vel1 = velPool[bodyOffset + deltaVOffset + bodyId + T];
            float acc[6] = {};
            bool hasRef = false;
            for (uint32_t b = 0; b < numSlabs; ++b)
            {
                const uint32_t bodyIndex = bodyId * numSlabs + b;
                if (references[bodyIndex] != 0xFFFFFFFFu)
                {
                    const uint32_t velIndex = averageOutputOffsets + 3 * (bodyIndex & ~31u) + (bodyIndex & 31u);
                    acc[0] += velPool[velIndex].x - vel0.x;
                    acc[1] += velPool[velIndex].y - vel0.y;
                    acc[2] += velPool[velIndex].z - vel0.z;
                    acc[3] += velPool[velIndex].w - vel0.w;
                    acc[4] += velPool[velIndex + 32].x - vel1.x;
                    acc[5] += velPool[velIndex + 32].y - vel1.y;
                    hasRef = true;
                }
            }
            if (hasRef)
            {
                uint32_t refCount = 0;
                for (uint32_t i = 0; i < num32Slabs; ++i)
                    refCount += popcount(encoded[bodyOffset + bodyId + i * numTotalBodies]);
                refCount = std::max(1u, refCount);
                const float recip = 1.f / float(refCount);
                for (float& a : acc)
                    a *= recip;
                for (uint32_t i = 0; i < num32Slabs; ++i)
                    refEnc[bodyOffset + bodyId + i * numTotalBodies] = 0;
                refVel[bodyOffset + deltaVOffset + bodyId] =
                    Float4{ vel0.x + acc[0], vel0.y + acc[1], vel0.z + acc[2], vel0.w + acc[3] };
                refVel[bodyOffset + deltaVOffset + bodyId + T] =
                    Float4{ vel1.x + acc[4], vel1.y + acc[5], vel1.z, vel1.w };
            }
        }

        // propagate pass (all bodies incl. kinematics; separate pass mirrors
        // the barrier between the dispatches)
        for (uint32_t bodyId = 0; bodyId < T; ++bodyId)
        {
            const Float4 curVel0 = refVel[deltaVOffset + bodyId];
            Float4 curVel1 = refVel[deltaVOffset + bodyId + T];
            Float4 curVel2 = refVel[deltaVOffset + bodyId + 2 * T];

            if (!isVelocityIteration && bodyId > 0)
            {
                const uint32_t nodeIndex = uint32_t(bodyData[bodyId].islandNodeIndex & 0xFFFFFFFFu);
                const uint32_t lockFlags = bodySims[nodeIndex].lockFlags;

                Vec3 linVel(curVel0.x, curVel0.y, curVel0.z);
                const Vec3 angVel(curVel0.w, curVel1.x, curVel1.y);

                Vec3 amv = angVel;
                if (bodyId >= bodyOffset)
                    amv = Mat33::load(refTxi[bodyId].sqrtInvInertia).transform(angVel);

                if (lockFlags & (1u << 3)) amv.x = 0.f;
                if (lockFlags & (1u << 4)) amv.y = 0.f;
                if (lockFlags & (1u << 5)) amv.z = 0.f;
                if (lockFlags & (1u << 0)) linVel.x = 0.f;
                if (lockFlags & (1u << 1)) linVel.y = 0.f;
                if (lockFlags & (1u << 2)) linVel.z = 0.f;

                const Vec3 deltaP = linVel * stepDt;
                curVel1.z += deltaP.x;
                curVel1.w += deltaP.y;
                curVel2.x += deltaP.z;
                curVel2.y += angVel.x * stepDt;
                curVel2.z += angVel.y * stepDt;
                curVel2.w += angVel.z * stepDt;

                QuatV q(refTxi[bodyId].deltaBody2World.q);
                const float w2 = amv.dot(amv);
                if (w2 != 0.f)
                {
                    const float w = std::sqrt(w2);
                    const float v = w * 0.5f * stepDt;
                    const float s = std::sin(v) / w;
                    const float c = std::cos(v);
                    const Vec3 pqr = amv * s;
                    const QuatV quatVel(pqr.x, pqr.y, pqr.z, 0.f);
                    q = (quatVel * q + q * c).getNormalized();
                }
                refTxi[bodyId].deltaBody2World.q = q.toQuat();
                refTxi[bodyId].deltaBody2World.p.x += deltaP.x;
                refTxi[bodyId].deltaBody2World.p.y += deltaP.y;
                refTxi[bodyId].deltaBody2World.p.z += deltaP.z;
            }

            if (bodyId >= bodyOffset)
            {
                for (uint32_t b = 0; b < numSlabs; ++b)
                {
                    const uint32_t bodyIndex = (bodyId - bodyOffset) * numSlabs + b;
                    const uint32_t remapped = references[bodyIndex];
                    if (remapped != 0xFFFFFFFFu)
                    {
                        refVel[remapped] = curVel0;
                        refVel[remapped + 32] = curVel1;
                        refVel[remapped + 64] = curVel2;
                    }
                }
            }

            if (!isVelocityIteration)
            {
                refVel[deltaVOffset + bodyId + T] = curVel1;
                refVel[deltaVOffset + bodyId + 2 * T] = curVel2;
            }
        }

        // writeBackBodiesTGS
        for (uint32_t a = 0; a < totalBodyCount; ++a)
        {
            const Float4 vel1 = refVel[bodyOffset + deltaVOffset + a + T];
            const Float4 vel2 = refVel[bodyOffset + deltaVOffset + a + 2 * T];
            refMotion[a + bodyOffset] = Float4{ vel1.z * invDt, vel1.w * invDt, vel2.x * invDt, 0.f };
            refMotion[a + bodyOffset + T] = Float4{ vel2.y * invDt, vel2.z * invDt, vel2.w * invDt, 0.f };
        }

        // ---- compare ----
        const auto gpuVel = downloadBuffer<Float4>(dev, velBuf, poolSize);
        const auto gpuEnc = downloadBuffer<uint32_t>(dev, encBuf, encoded.size());
        const auto gpuTxi = downloadBuffer<SolverTxIData>(dev, txBuf, txi.size());
        const auto gpuMotion = downloadBuffer<Float4>(dev, mvBuf, motionVel.size());

        size_t mismatches = 0;
        for (size_t i = 0; i < poolSize; ++i)
        {
            const float* g = &gpuVel[i].x;
            const float* r = &refVel[i].x;
            for (int c = 0; c < 4; ++c)
                if (!almostEqual(g[c], r[c], 1e-4f, 1e-4f))
                {
                    if (mismatches < 3)
                        PXG_CHECK_MSG(false, "slabs=%u velIt=%u vel[%zu].%d: gpu=%g ref=%g", numSlabs,
                                      isVelocityIteration, i, c, g[c], r[c]);
                    ++mismatches;
                }
        }
        PXG_CHECK_MSG(mismatches == 0, "slabs=%u velIt=%u: %zu velocity mismatches", numSlabs,
                      isVelocityIteration, mismatches);

        mismatches = 0;
        for (size_t i = 0; i < encoded.size(); ++i)
            if (gpuEnc[i] != refEnc[i])
                ++mismatches;
        PXG_CHECK_MSG(mismatches == 0, "slabs=%u: %zu encoded-count mismatches", numSlabs, mismatches);

        mismatches = 0;
        for (size_t i = 0; i < txi.size(); ++i)
        {
            const float* g = &gpuTxi[i].deltaBody2World.q.x;
            const float* r = &refTxi[i].deltaBody2World.q.x;
            for (int c = 0; c < 7; ++c)
                if (!almostEqual(g[c], r[c], 1e-4f, 1e-4f))
                {
                    if (mismatches < 3)
                        PXG_CHECK_MSG(false, "slabs=%u velIt=%u txi[%zu].%d: gpu=%g ref=%g", numSlabs,
                                      isVelocityIteration, i, c, g[c], r[c]);
                    ++mismatches;
                }
        }
        PXG_CHECK_MSG(mismatches == 0, "slabs=%u velIt=%u: %zu delta-pose mismatches", numSlabs,
                      isVelocityIteration, mismatches);

        mismatches = 0;
        for (size_t i = 0; i < motionVel.size(); ++i)
        {
            const float* g = &gpuMotion[i].x;
            const float* r = &refMotion[i].x;
            for (int c = 0; c < 4; ++c)
                if (!almostEqual(g[c], r[c], 1e-4f, 1e-4f))
                    ++mismatches;
        }
        PXG_CHECK_MSG(mismatches == 0, "slabs=%u velIt=%u: %zu motion-velocity mismatches", numSlabs,
                      isVelocityIteration, mismatches);

        for (PxgBufferHandle h : { dummyBuf, velBuf, encBuf, refBuf, txBuf, bdBuf, bsBuf, mvBuf })
            dev->destroyBuffer(h);
    }
}

} // namespace
