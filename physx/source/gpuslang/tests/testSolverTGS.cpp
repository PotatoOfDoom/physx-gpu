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
#include "PxgCudaShim.h"

#include "PxNodeIndex.h" // completes PxNodeIndex for the PxgNarrowphaseCore.h chain
#include "PxgSolverCoreDesc.h"
#include "PxgIslandContext.h"

#include <cmath>
#include <cstring>
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

// ---- P9.86c: CUDA min/max semantics for the mirrors ----
inline uint32_t tgsBits(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}
// PxMin<float>/PxMax<float> = intrinsics::selectMin/selectMax (PxUnixMathIntrinsics.h): a<b?a:b /
// a>b?a:b -- on a +-0 tie (or NaN) the SECOND operand survives.
inline float pxSelectMin(float a, float b) { return a < b ? a : b; }
inline float pxSelectMax(float a, float b) { return a > b ? a : b; }
// CUDA fminf/fmaxf: IEEE minNum/maxNum with NVIDIA's ordered zeros (fminf(+-0) = -0 unless both
// +0, fmaxf(+-0) = +0 unless both -0).  glibc's fminf/fmaxf keep an operand on a zero tie instead.
inline float tgsCudaFminf(float a, float b)
{
    if (std::isnan(a)) return b;
    if (std::isnan(b)) return a;
    if (a == b) return std::signbit(a) ? a : b;
    return a < b ? a : b;
}
inline float tgsCudaFmaxf(float a, float b)
{
    if (std::isnan(a)) return b;
    if (std::isnan(b)) return a;
    if (a == b) return std::signbit(a) ? b : a;
    return a > b ? a : b;
}
// The reverted port's clamp(v, lo, hi) = GLSL FClamp -> fmin(fmax(v, lo), hi) with ordered zeros
// under the SignedZeroInfNanPreserve stamp (P9.86 F4) -- used only to count directed inputs on
// which the CUDA PxClamp and the old port spelling store different bits.
inline float tgsRevertFClamp(float v, float lo, float hi) { return tgsCudaFminf(tgsCudaFmaxf(v, lo), hi); }
uint32_t gTorsClampRevertDiffers = 0; // torsional rows where PxClamp != FClamp bitwise (clamp taken)

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

// PxgErrorAccumulator reference (solverResidual.cuh): per-thread contact
// residual statistics, accumulated by solveWholeIslandTGS only.
struct RefErrAcc
{
    float sumSq = 0.f;
    uint32_t counter = 0;
    float maxErr = 0.f;

    void add(float deltaF, float velMultiplier)
    {
        const float r = velMultiplier == 0.f ? 0.f : deltaF / velMultiplier;
        sumSq += r * r;
        ++counter;
        maxErr = std::fmax(maxErr, std::fabs(r));
    }
};

void refSolveContactTGS(const BatchTGS& b, BodyStateTGS& s, uint32_t lane,
                        std::vector<TGSContactHeader>& chs, std::vector<TGSFrictionHeader>& fhs,
                        std::vector<TGSContactPoint>& cps, std::vector<TGSFriction>& frs,
                        float elapsedTime, float minPen, float ref0, float ref1,
                        RefErrAcc* err = nullptr)
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

        if (err)
            err->add(deltaF, velMultiplier);

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

            if (err)
            {
                err->add(d0, vm0);
                err->add(d1, vm1);
            }

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
            // PxClamp(v, lo, hi) = PxMin(hi, PxMax(lo, v)) = selectMin(hi, selectMax(lo, v))
            // (solverBlockTGS.cuh:473): with (lo, hi) = (+0, -0) and v < 0 this stores +0.
            const float lo = -maxDynFrictionImpulse * fscale;
            const float hi = maxDynFrictionImpulse * fscale;
            const float clamped = pxSelectMin(hi, pxSelectMax(lo, total));
            if (clampI && tgsBits(tgsRevertFClamp(total, lo, hi)) != tgsBits(clamped))
                ++gTorsClampRevertDiffers;
            const float na = clampI ? clamped : total;
            const float d = na - af0;

            if (err)
                err->add(d, vm0);

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
        // CUDA fminf(maxImpulse, fmaxf(minImpulse, u)) (solverBlockTGS.cuh:907)
        const float clampedForce =
            tgsCudaFminf(c.maxImpulse[lane], tgsCudaFmaxf(c.minImpulse[lane], unclampedForce));
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

// P9.86c directed contact lane: a torsional anchor with torsionalFrictionScale = -0.  The CUDA prep
// writes exactly that for maxPenetration == +0 with minTorsionalPatchRadius = 0
// (contactConstraintBlockPrep.cuh:1128: selectMax(0, -0) = -0, sqrtf(-0) = -0).  Axis-aligned frame
// (contact normal +y), one normal row on the line of centres, a y-free friction pair, torsional
// raXn = rbXn = (0, 0.5, 0), body A spinning about the normal (angVel0.y = 1), body B not.  Then
// acc >= +0, (lo, hi) = (-maxDyn * -0, maxDyn * -0) = (+0, -0), |total| > maxFric * -0 and
// total = +0 - 0.5 * vm < 0: CUDA PxClamp stores +0, the old FClamp spelling -0.
// Returns the index of the torsional friction row.
uint32_t injectTorsNegZeroScaleLane(BatchTGS& b, uint32_t lane, std::vector<TGSContactHeader>& chs,
                                    std::vector<TGSFrictionHeader>& fhs, std::vector<TGSContactPoint>& cps,
                                    std::vector<TGSFriction>& frs, std::vector<Float4>& velPool, uint32_t k)
{
    const uint32_t bi = b.scalars[1];
    TGSContactHeader& h = chs[bi];
    h.normal_staticFriction[lane] = Float4{ 0.f, 1.f, 0.f, 0.6f };
    h.numNormalConstr[lane] = 1;

    TGSContactPoint& c = cps[b.scalars[2]];
    c.raXn_extraCoeff[lane].x = c.raXn_extraCoeff[lane].y = c.raXn_extraCoeff[lane].z = 0.f;
    c.rbXn_targetVelW[lane].x = c.rbXn_targetVelW[lane].y = c.rbXn_targetVelW[lane].z = 0.f;
    c.appliedForce[lane] = 1.0f;
    c.separation[lane] = 0.f;

    TGSFrictionHeader& fh = fhs[bi];
    fh.numFrictionConstr[lane] = 3;
    fh.dynamicFriction[lane] = 0.5f;
    fh.torsionalFrictionScale[lane] = -0.0f;
    fh.frictionNormals0[lane] = Float4{ 1.f, 0.f, 0.f, 0.f };
    fh.frictionNormals1[lane] = Float4{ 0.f, 0.f, 1.f, 0.f };

    const uint32_t sf = b.scalars[3];
    const float pr[2][6] = { { 0.f, 0.f, 0.3f, 0.f, 0.f, -0.3f }, { -0.3f, 0.f, 0.f, 0.3f, 0.f, 0.f } };
    for (uint32_t i = 0; i < 2; ++i)
    {
        Float4& ra = frs[sf + i].raXn_error[lane];
        Float4& rb = frs[sf + i].rbXn_targetVelW[lane];
        ra.x = pr[i][0], ra.y = pr[i][1], ra.z = pr[i][2];
        rb.x = pr[i][3], rb.y = pr[i][4], rb.z = pr[i][5];
    }
    TGSFriction& t = frs[sf + 2];
    t.raXn_error[lane] = Float4{ 0.f, 0.5f, 0.f, 0.f };
    t.rbXn_targetVelW[lane] = Float4{ 0.f, 0.5f, 0.f, 0.f };
    t.appliedForce[lane] = 0.f;

    velPool[k * 192 + lane + 32].x = 1.0f;  // angVel0.y
    velPool[k * 192 + lane + 128].x = 0.0f; // angVel1.y
    return sf + 2;
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
        // engine PxgConstraintType encoding: even index = contact = eCONTACT(1),
        // odd = 1D joint = eCONSTRAINT_1D(0) (matches solveBlockPartitionTGS /
        // pxgIsContactType, flipped from the legacy PXG_CONSTRAINT_CONTACT=0).
        b.constraintType = uint16_t(k % 2 == 0 ? 1u : 0u);
        b.scalars[1] = k;                       // constraintBatchIndex
        b.scalars[2] = k * rowsPerBatch;        // startConstraintIndex
        b.scalars[3] = k * (rowsPerBatch + 1);  // startFrictionIndex

        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            b.bodyAIndex[lane] = rng() % numTotalBodies;
            b.bodyBIndex[lane] = rng() % numTotalBodies;
            b.remappedBodyAIndex[lane] = k * 192 + lane;
            b.remappedBodyBIndex[lane] = k * 192 + lane + 96;

            if (b.constraintType == 1) // eCONTACT
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

    // P9.86c directed lane (batch 0 = eCONTACT, lane 0 < descStride): torsional scale -0.
    const uint32_t torsRow = injectTorsNegZeroScaleLane(batches[0], 0, chs, fhs, cps, frs, velPool, 0);

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

    gTorsClampRevertDiffers = 0;
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

            if (b.constraintType == 1) // eCONTACT
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

    // P9.86c: the directed torsional lane, compared BITWISE (almostEqual accepts -0 == +0).
    PXG_CHECK_MSG(gTorsClampRevertDiffers > 0,
                  "directed torsional lane: (lo,hi)=(+0,-0), v<0 not reached (PxClamp vs FClamp untested)");
    PXG_CHECK_MSG(tgsBits(frs[torsRow].appliedForce[0]) == 0u, "CUDA PxClamp stores +0 (ref %08x)",
                  tgsBits(frs[torsRow].appliedForce[0]));
    PXG_CHECK_MSG(tgsBits(gpuFrs[torsRow].appliedForce[0]) == tgsBits(frs[torsRow].appliedForce[0]),
                  "torsional scale -0: gpu %08x ref %08x", tgsBits(gpuFrs[torsRow].appliedForce[0]),
                  tgsBits(frs[torsRow].appliedForce[0]));

    for (PxgBufferHandle h : { batchBuf, chBuf, fhBuf, cpBuf, frBuf, jhBuf, jrBuf, txBuf, velBuf, encBuf })
        dev->destroyBuffer(h);
}

// Shared engine-ABI test helpers (used by every PxgCudaShim-driven test below).
static std::vector<uint8_t> solverTgsAbiLoader(const char* file) { return loadShaderBlob(file); }
static uint32_t f2uTGS(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

// Mirrors the bridge's marshalling: place each argument at the natural
// alignment inferred from its size, then round the blob to the largest
// alignment seen (a compiled struct's trailing padding).
struct EngineBlob
{
    std::vector<uint8_t> bytes;
    size_t maxAlign = 1;

    static size_t alignFor(size_t size)
    {
        if (size >= 16) return 16;
        if (size >= 8)  return 8;
        if (size >= 4)  return 4;
        if (size >= 2)  return 2;
        return 1;
    }
    template <typename T>
    void add(const T& v)
    {
        const size_t a = alignFor(sizeof(T));
        if (a > maxAlign) maxAlign = a;
        bytes.resize((bytes.size() + (a - 1)) & ~(a - 1));
        const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
        bytes.insert(bytes.end(), p, p + sizeof(T));
    }
    void finish() { bytes.resize((bytes.size() + (maxAlign - 1)) & ~(maxAlign - 1)); }
};

// ---------------------------------------------------------------------------
// writebackBlocksTGS (engine ABI, P5 Step 4 BDA)
// ---------------------------------------------------------------------------
// ENGINE ABI: the writeback-block kernel takes the engine's
// (PxgSolverCoreDesc*, PxgSolverSharedDesc<IterativeSolveDataTGS>*, PxU32
// islandIndex) pack -> {8,8,4} = 24-byte blob.  The test fills the REAL engine
// descs (+ a 2-element PxgIslandContext array; element [0] is a decoy so
// islandIndex = 1 pins the island stride) with live device addresses and
// launches through PxgCudaShim with the engine's grid arithmetic (blockDim
// WRITEBACK_BLOCKS = 256 = 8 warps; one warp per batch).  constraintType uses
// the engine encoding (contact == eCONTACT(1) / eARTICULATION_CONTACT(3); 1D
// joints are 0).  The batchStartIndex-offset arithmetic of the shared island
// derivation is additionally covered by solverTGS_concludeBlocksAbi.
//
// conclude runs before writeback in production but only zeroes repacked .w
// coefficients that the writeback path never reads, so the writeback outputs
// are independent of it; no conclude emulation is needed here.
PXG_TEST(solverTGS_writebackBlocksAbi)
{
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    // engine PxgConstraintType: eCONTACT == 1, eARTICULATION_CONTACT == 3
    auto isEngineContact = [](uint16_t t) { return t == 1u || t == 3u; };

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
        // engine encoding: even batches are contacts (eCONTACT=1), odd are 1D joints (0)
        b.constraintType = uint16_t(k % 2 == 0 ? 1u : 0u);
        b.scalars[1] = k;
        b.scalars[2] = k * rowsPerBatch;
        b.scalars[3] = k * (rowsPerBatch + 1);

        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            b.bodyAIndex[lane] = rng() % numBodies;
            b.bodyBIndex[lane] = rng() % numBodies;
            b.shapeInteraction[lane] = (uint64_t(rng()) << 32) | rng();

            if (isEngineContact(b.constraintType))
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

    // P9.86c directed pair: a contact report threshold of -0.0f is legal data
    // (NpRigidDynamic::setContactReportThreshold keeps -0; copied bit for bit into
    // PxgSolverBodyData.reportThreshold).  Body A = -0, body B = +0: the CUDA PxMin<float>
    // (selectMin, solverBlockTGS.cuh:722) stores +0, GLSL FMin would store -0.
    {
        bodyData[0].reportThreshold = -0.0f;
        bodyData[0].islandNodeIndex = makeNodeIndex(1001);
        bodyData[1].reportThreshold = 0.0f;
        bodyData[1].islandNodeIndex = makeNodeIndex(1002);
        BatchTGS& tb = batches[0]; // eCONTACT, lane 0 < descStride
        tb.bodyAIndex[0] = 0;
        tb.bodyBIndex[0] = 1;
        chs[0].flags[0] = 1u; // eHAS_FORCE_THRESHOLDS
        chs[0].numNormalConstr[0] = 1;
        if (chs[0].forceWritebackOffset[0] == 0xFFFFFFFFu)
        {
            chs[0].forceWritebackOffset[0] = forceBufCursor;
            forceBufCursor += rowsPerBatch;
        }
        cps[tb.scalars[2]].appliedForce[0] = 1.5f;
    }

    PxgSlangComputeDevice* dev = testDevice();

    std::vector<ConstraintWriteback> wbInit(std::max(writebackCursor, 1u));
    std::memset(wbInit.data(), 0, wbInit.size() * sizeof(ConstraintWriteback));
    std::vector<float> forceInit(std::max(forceBufCursor, 1u), 0.f);
    const uint32_t tsCapacity = numBatches * 32;
    std::vector<ThresholdElement> tsInit(tsCapacity);
    std::memset(tsInit.data(), 0, tsInit.size() * sizeof(ThresholdElement));

    const PxgKernelDef manifest[] = {
        { "solverTGS.spv", "writebackBlocksTGS", 24u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &solverTgsAbiLoader);

    // ---- upload the block pools + writeback targets ----
    PxgDevicePtr dBatch = 0, dCh = 0, dFh = 0, dCp = 0, dFr = 0, dJh = 0, dJr = 0, dBody = 0;
    PxgDevicePtr dPatchBlocks = 0, dPatches = 0, dForce = 0, dWb = 0, dTs = 0, dIsl = 0, dDesc = 0, dShared = 0;
    shim.memAlloc(&dBatch, batches.size() * sizeof(BatchTGS));
    shim.memAlloc(&dCh, chs.size() * sizeof(TGSContactHeader));
    shim.memAlloc(&dFh, fhs.size() * sizeof(TGSFrictionHeader));
    shim.memAlloc(&dCp, cps.size() * sizeof(TGSContactPoint));
    shim.memAlloc(&dFr, frs.size() * sizeof(TGSFriction));
    shim.memAlloc(&dJh, jhs.size() * sizeof(TGSJointHeader));
    shim.memAlloc(&dJr, jrows.size() * sizeof(TGSJointRowCon));
    shim.memAlloc(&dBody, bodyData.size() * sizeof(SolverBodyData));
    shim.memAlloc(&dPatchBlocks, patchBlocks.size() * sizeof(BlockFrictionPatch));
    shim.memAlloc(&dPatches, patches.size() * sizeof(FrictionPatchGPU));
    shim.memAlloc(&dForce, forceInit.size() * sizeof(float));
    shim.memAlloc(&dWb, wbInit.size() * sizeof(ConstraintWriteback));
    shim.memAlloc(&dTs, tsInit.size() * sizeof(ThresholdElement));
    if (!dBatch || !dCh || !dJr || !dBody || !dForce || !dWb || !dTs)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    shim.memcpyHtoD(dBatch, batches.data(), batches.size() * sizeof(BatchTGS));
    shim.memcpyHtoD(dCh, chs.data(), chs.size() * sizeof(TGSContactHeader));
    shim.memcpyHtoD(dFh, fhs.data(), fhs.size() * sizeof(TGSFrictionHeader));
    shim.memcpyHtoD(dCp, cps.data(), cps.size() * sizeof(TGSContactPoint));
    shim.memcpyHtoD(dFr, frs.data(), frs.size() * sizeof(TGSFriction));
    shim.memcpyHtoD(dJh, jhs.data(), jhs.size() * sizeof(TGSJointHeader));
    shim.memcpyHtoD(dJr, jrows.data(), jrows.size() * sizeof(TGSJointRowCon));
    shim.memcpyHtoD(dBody, bodyData.data(), bodyData.size() * sizeof(SolverBodyData));
    shim.memcpyHtoD(dPatchBlocks, patchBlocks.data(), patchBlocks.size() * sizeof(BlockFrictionPatch));
    shim.memcpyHtoD(dPatches, patches.data(), patches.size() * sizeof(FrictionPatchGPU));
    shim.memcpyHtoD(dForce, forceInit.data(), forceInit.size() * sizeof(float));
    shim.memcpyHtoD(dWb, wbInit.data(), wbInit.size() * sizeof(ConstraintWriteback));
    shim.memcpyHtoD(dTs, tsInit.data(), tsInit.size() * sizeof(ThresholdElement));

    // island[0] is a decoy so islandIndex = 1 exercises the PxgIslandContext stride
    physx::PxgIslandContext islands[2];
    std::memset(islands, 0, sizeof(islands));
    islands[0].mBodyCount = 3u; // decoy
    islands[1].mBatchStartIndex = 0u;
    islands[1].mBatchCount = numBatches; // endIndex = numBatches + startIndex
    shim.memAlloc(&dIsl, sizeof(islands));
    shim.memcpyHtoD(dIsl, islands, sizeof(islands));

    physx::PxgSolverCoreDesc desc;
    std::memset(&desc, 0, sizeof(desc));
    desc.islandContextPool = reinterpret_cast<physx::PxgIslandContext*>(dIsl);
    desc.solverBodyDataPool = reinterpret_cast<physx::PxgSolverBodyData*>(dBody);
    desc.forceBuffer = reinterpret_cast<physx::PxF32*>(dForce);
    desc.constraintWriteBack = reinterpret_cast<physx::PxgConstraintWriteback*>(dWb);
    desc.frictionPatches = reinterpret_cast<physx::PxFrictionPatch*>(dPatches);
    desc.thresholdStream = reinterpret_cast<physx::Dy::ThresholdStreamElement*>(dTs);
    desc.sharedThresholdStreamIndex = 0; // in-struct atomic counter (memset already zeroed it)

    physx::PxgSolverSharedDesc<physx::IterativeSolveDataTGS> shared;
    std::memset(&shared, 0, sizeof(shared));
    shared.blockCurrentFrictionPatches = reinterpret_cast<physx::PxgBlockFrictionPatch*>(dPatchBlocks);
    shared.iterativeData.blockConstraintBatch = reinterpret_cast<physx::PxgBlockConstraintBatch*>(dBatch);
    shared.iterativeData.blockJointConstraintHeaders =
        reinterpret_cast<physx::PxgTGSBlockSolverConstraint1DHeader*>(dJh);
    shared.iterativeData.blockJointConstraintRowsCon =
        reinterpret_cast<physx::PxgTGSBlockSolverConstraint1DCon*>(dJr);
    shared.iterativeData.blockContactHeaders =
        reinterpret_cast<physx::PxgTGSBlockSolverContactHeader*>(dCh);
    shared.iterativeData.blockFrictionHeaders =
        reinterpret_cast<physx::PxgTGSBlockSolverFrictionHeader*>(dFh);
    shared.iterativeData.blockContactPoints =
        reinterpret_cast<physx::PxgTGSBlockSolverContactPoint*>(dCp);
    shared.iterativeData.blockFrictions =
        reinterpret_cast<physx::PxgTGSBlockSolverContactFriction*>(dFr);

    shim.memAlloc(&dDesc, sizeof(desc));
    shim.memcpyHtoD(dDesc, &desc, sizeof(desc));
    shim.memAlloc(&dShared, sizeof(shared));
    shim.memcpyHtoD(dShared, &shared, sizeof(shared));

    EngineBlob blob;
    blob.add<uint64_t>(dDesc);
    blob.add<uint64_t>(dShared);
    blob.add<uint32_t>(1u); // islandIndex
    blob.finish();
    PXG_CHECK_MSG(blob.bytes.size() == 24, "writeback blob is 24 bytes (got %zu)", blob.bytes.size());

    const uint32_t gridX = (numBatches + 7) / 8; // 8 warps per 256-thread block
    PXG_CHECK_MSG(shim.launchKernel(0, gridX, 1, 1, blob.bytes.data(), uint32_t(blob.bytes.size())) ==
                      PxgResult::Success,
                  "launch writebackBlocksTGS");
    shim.streamSynchronize();

    // ---- reference (writebackBlocksTGS; engine constraintType encoding) ----
    auto refPatchBlocks = patchBlocks;
    std::vector<float> refForce = forceInit;
    std::vector<ConstraintWriteback> refWb = wbInit;
    std::vector<FrictionPatchGPU> refPatches = patches;
    std::vector<ThresholdElement> refStream;
    size_t refMutations = 0;    // guard against an all-no-op fixture
    size_t refBrokenJoints = 0; // guard against a fixture that never breaks a joint
    size_t thrNegPosZero = 0;   // P9.86c: threshold elements with (rtA, rtB) = (-0, +0)

    for (uint32_t k = 0; k < numBatches; ++k)
    {
        const BatchTGS& b = batches[k];
        for (uint32_t lane = 0; lane < b.descStride; ++lane)
        {
            if (isEngineContact(b.constraintType))
            {
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
                        ++refMutations;
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
                    // PxMin<float> = selectMin (solverBlockTGS.cuh:722): (-0, +0) -> +0
                    e.threshold = pxSelectMin(rt0, rt1);
                    if (tgsBits(rt0) == 0x80000000u && tgsBits(rt1) == 0u)
                        ++thrNegPosZero;
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
                // writeBack1DBlockTGS — reads jrows directly (writeback never
                // mutates the joint rows; conclude, which does, is a separate pass)
                const uint32_t rows = jhs[k].rowCounts_breakable_orthoAxisCount[lane] & 0xFFu;
                const uint32_t fwo = jhs[k].writeBackOffset[lane];
                const uint32_t breakable = (jhs[k].rowCounts_breakable_orthoAxisCount[lane] >> 8) & 0xFFu;
                if (fwo != 0xFFFFFFFFu)
                {
                    Vec3 linVel(0.f), angVel(0.f);
                    float errSq = 0.f;
                    for (uint32_t i = 0; i < rows; ++i)
                    {
                        const TGSJointRowCon& c = jrows[b.scalars[2] + i];
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
                    ++refMutations;
                    if (broken)
                        ++refBrokenJoints;
                }
            }
        }
    }
    PXG_CHECK_MSG(refMutations > 0, "writeback fixture mutated nothing (too weak / no-op?)");
    // COVERAGE: the broken flag is the SIGN BIT of a zero.  A platform that drops
    // signed zeros writes a record that is bit-perfect everywhere else and never
    // reports a broken joint, so the memcmp below can only see it if the fixture
    // actually breaks one.
    PXG_CHECK_MSG(refBrokenJoints > 0, "writeback fixture breaks no joint (the sign-bit path is untested)");
    PXG_CHECK_MSG(thrNegPosZero > 0,
                  "threshold fixture has no (bodyA -0, bodyB +0) element: selectMin vs FMin untested");

    // ---- download + compare ----
    std::vector<TGSJointRowCon> gpuRows(jrows.size());
    std::vector<TGSContactPoint> gpuCps(cps.size());
    std::vector<float> gpuForce(refForce.size());
    std::vector<ConstraintWriteback> gpuWb(refWb.size());
    std::vector<FrictionPatchGPU> gpuPatches(patches.size());
    std::vector<BlockFrictionPatch> gpuPatchBlocks(patchBlocks.size());
    shim.memcpyDtoH(gpuRows.data(), dJr, gpuRows.size() * sizeof(TGSJointRowCon));
    shim.memcpyDtoH(gpuCps.data(), dCp, gpuCps.size() * sizeof(TGSContactPoint));
    shim.memcpyDtoH(gpuForce.data(), dForce, gpuForce.size() * sizeof(float));
    shim.memcpyDtoH(gpuWb.data(), dWb, gpuWb.size() * sizeof(ConstraintWriteback));
    shim.memcpyDtoH(gpuPatches.data(), dPatches, gpuPatches.size() * sizeof(FrictionPatchGPU));
    shim.memcpyDtoH(gpuPatchBlocks.data(), dPatchBlocks, gpuPatchBlocks.size() * sizeof(BlockFrictionPatch));
    // the threshold-stream count lives in the desc's in-struct PxI32 counter
    physx::PxgSolverCoreDesc descBack;
    shim.memcpyDtoH(&descBack, dDesc, sizeof(descBack));
    const uint32_t gpuTsCount = uint32_t(descBack.sharedThresholdStreamIndex);
    std::vector<ThresholdElement> gpuStream(gpuTsCount);
    if (gpuTsCount)
        shim.memcpyDtoH(gpuStream.data(), dTs, gpuTsCount * sizeof(ThresholdElement));

    // writeback never touches the joint rows or contact points: they must be
    // byte-identical to the upload (canary).
    size_t mismatches = 0;
    for (size_t i = 0; i < jrows.size(); ++i)
        if (std::memcmp(&gpuRows[i], &jrows[i], sizeof(TGSJointRowCon)) != 0)
            ++mismatches;
    for (size_t i = 0; i < cps.size(); ++i)
        if (std::memcmp(&gpuCps[i], &cps[i], sizeof(TGSContactPoint)) != 0)
            ++mismatches;
    PXG_CHECK_MSG(mismatches == 0, "writeback touched read-only inputs: %zu", mismatches);

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

    for (PxgDevicePtr pfree : { dBatch, dCh, dFh, dCp, dFr, dJh, dJr, dBody, dPatchBlocks, dPatches,
                                dForce, dWb, dTs, dIsl, dDesc, dShared })
        shim.memFree(pfree);
}

// ---------------------------------------------------------------------------
// computeAverageSolverBodyVelocityTGS + propagateAverageSolverBodyVelocityTGS
// + writeBackBodiesTGS
// ---------------------------------------------------------------------------
// ENGINE ABI (P5 Step 4): the averaging/propagate/writeback-bodies kernels
// take the engine's (PxgSolverCoreDesc*, PxgSolverSharedDesc*) packs; the
// test fills the REAL engine descs (+ PxgIslandContext) with live device
// addresses and launches through PxgCudaShim with the engine's grid
// arithmetic (warp-structured blockDim (32,8,1) -> one-warp workgroups x8;
// WRITE_BACK_BODIES blockDim 256).  writeBackBodiesTGS is launched with
// islandIndex = 1 against a 2-element island array whose element [0] is a
// decoy, pinning the PxgIslandContext stride on the real ABI.
// solverTgsAbiLoader + EngineBlob are defined above (before the first
// engine-ABI test).

PXG_TEST(solverTGS_velocityAveragingAndPropagate)
{
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

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
        // P9.86f DIRECTED -0 delta-position lanes.  The CUDA's curVel1 is make_float4(0.f) with only .x/.y
        // accumulated, so `curVel1 += vel1` (solverMultiBlockTGS.cu:348/410) turns a -0 vel1.z / vel1.w
        // into +0 for every body with a reference.  Seeded AFTER the random draws (rng untouched).
        for (uint32_t bodyId = 1; bodyId < totalBodyCount; bodyId += 2)
        {
            Float4& v1 = velPool[bodyOffset + deltaVOffset + bodyId + T];
            if (bodyId % 4 != 3) v1.z = -0.0f;
            if (bodyId % 4 != 1) v1.w = -0.0f;
            if (bodyId % 8 == 5) { v1.z = -0.0f; v1.w = -0.0f; }
        }

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

        // engine ABI: {desc, shared} = 16; propagate {desc, shared, bool, bool,
        // float} = 24; writeBack {desc, shared, u32} = 24
        const PxgKernelDef manifest[] = {
            { "solverTGS.spv", "computeAverageSolverBodyVelocityTGS", 16u, 0 },
            { "solverTGS.spv", "propagateAverageSolverBodyVelocityTGS", 24u, 0 },
            { "solverTGS.spv", "writeBackBodiesTGS", 24u, 0 },
        };
        PxgCudaShim shim(dev, manifest, 3, &solverTgsAbiLoader);

        PxgDevicePtr dVel = 0, dEnc = 0, dRef = 0, dTx = 0, dBd = 0, dBs = 0, dMv = 0;
        shim.memAlloc(&dVel, velPool.size() * sizeof(Float4));
        shim.memAlloc(&dEnc, encoded.size() * 4);
        shim.memAlloc(&dRef, references.size() * 4);
        shim.memAlloc(&dTx, txi.size() * sizeof(SolverTxIData));
        shim.memAlloc(&dBd, bodyData.size() * sizeof(SolverBodyData));
        shim.memAlloc(&dBs, bodySims.size() * sizeof(BodySim));
        shim.memAlloc(&dMv, motionVel.size() * sizeof(Float4));
        if (!dVel || !dEnc || !dRef || !dTx || !dBd || !dBs || !dMv)
        {
            std::printf("  NOTE: no BDA -> skip\n");
            return;
        }
        shim.memcpyHtoD(dVel, velPool.data(), velPool.size() * sizeof(Float4));
        shim.memcpyHtoD(dEnc, encoded.data(), encoded.size() * 4);
        shim.memcpyHtoD(dRef, references.data(), references.size() * 4);
        shim.memcpyHtoD(dTx, txi.data(), txi.size() * sizeof(SolverTxIData));
        shim.memcpyHtoD(dBd, bodyData.data(), bodyData.size() * sizeof(SolverBodyData));
        shim.memcpyHtoD(dBs, bodySims.data(), bodySims.size() * sizeof(BodySim));
        shim.memcpyHtoD(dMv, motionVel.data(), motionVel.size() * sizeof(Float4));

        // island array: [0] is a DECOY, [1] is the real island — average/
        // propagate read island [0] through descA, writeBackBodiesTGS reads
        // islandIndex = 1 through descB, pinning the PxgIslandContext stride.
        physx::PxgIslandContext islands[2];
        std::memset(islands, 0, sizeof(islands));
        islands[0].mBodyStartIndex = bodyOffset;
        islands[0].mBodyCount = totalBodyCount;
        islands[0].mArticulationCount = 0;
        islands[1] = islands[0];
        physx::PxgIslandContext islandsB[2];
        std::memset(islandsB, 0, sizeof(islandsB));
        islandsB[0].mBodyStartIndex = 999u; // decoy: wrong stride/index fails
        islandsB[0].mBodyCount = 3u;
        islandsB[1].mBodyStartIndex = bodyOffset;
        islandsB[1].mBodyCount = totalBodyCount;
        PxgDevicePtr dIslA = 0, dIslB = 0;
        shim.memAlloc(&dIslA, sizeof(islands));
        shim.memcpyHtoD(dIslA, islands, sizeof(islands));
        shim.memAlloc(&dIslB, sizeof(islandsB));
        shim.memcpyHtoD(dIslB, islandsB, sizeof(islandsB));

        // the REAL engine descs with live device addresses
        physx::PxgSolverCoreDesc desc;
        std::memset(&desc, 0, sizeof(desc));
        desc.islandContextPool = reinterpret_cast<physx::PxgIslandContext*>(dIslA);
        desc.solverBodyReferences = reinterpret_cast<physx::PxgSolverReferences*>(dRef);
        desc.solverBodyDataPool = reinterpret_cast<physx::PxgSolverBodyData*>(dBd);
        desc.solverBodyTxIDataPool = reinterpret_cast<physx::PxgSolverTxIData*>(dTx);
        desc.mBodySimBufferDeviceData = reinterpret_cast<physx::PxgBodySim*>(dBs);
        desc.motionVelocityArray = reinterpret_cast<float4*>(dMv);
        desc.numSlabs = numSlabs;
        desc.numBatches = numBatches;
        desc.numArticBatches = 0;
        desc.numSolverBodies = T;
        desc.accumulatedBodyDeltaVOffset = deltaVOffset;
        physx::PxgSolverCoreDesc descB = desc;
        descB.islandContextPool = reinterpret_cast<physx::PxgIslandContext*>(dIslB);

        physx::PxgSolverSharedDesc<physx::IterativeSolveDataTGS> shared;
        std::memset(&shared, 0, sizeof(shared));
        shared.stepDt = stepDt;
        shared.invDtF32 = invDt;
        shared.iterativeData.solverBodyVelPool = reinterpret_cast<float4*>(dVel);
        shared.iterativeData.solverEncodedReferenceCount = reinterpret_cast<physx::PxU32*>(dEnc);

        PxgDevicePtr dDescA = 0, dDescB = 0, dShared = 0;
        shim.memAlloc(&dDescA, sizeof(desc));
        shim.memcpyHtoD(dDescA, &desc, sizeof(desc));
        shim.memAlloc(&dDescB, sizeof(descB));
        shim.memcpyHtoD(dDescB, &descB, sizeof(descB));
        shim.memAlloc(&dShared, sizeof(shared));
        shim.memcpyHtoD(dShared, &shared, sizeof(shared));

        // engine grid arithmetic: COMPUTE_BODIES_AVERAGE_VELOCITY = 256,
        // blockDim (32,8,1) -> dispatch nbBlocks*8 one-warp workgroups.
        // average (solvePartitions): nbThreads = 32 * mBodyCount;
        // propagate (pos-iter site): nbThreads = 32 * (mBodyCount + mBodyStartIndex).
        // (The velocity-iteration engine site under-dispatches via
        // 32*((mBodyCount+9)/10); the kernel is guard-correct under any grid,
        // so the test keeps the full-coverage grid for both variants to match
        // the all-bodies reference.)
        const uint32_t avgBlocks = (32 * totalBodyCount + 255) / 256;
        const uint32_t propBlocks = (32 * (totalBodyCount + bodyOffset) + 255) / 256;

        struct { uint64_t desc, shared; } avgPrm = { dDescA, dShared };
        PXG_CHECK_MSG(shim.launchKernel(0, avgBlocks * 8, 1, 1, &avgPrm, sizeof(avgPrm)) == PxgResult::Success,
                      "launch computeAverageSolverBodyVelocityTGS");

        // (descs..., bool isVelocityIteration, bool isLastPosIteration, PxReal
        // biasCoefficient) marshalled by the bridge's natural-alignment rule
        EngineBlob propBlob;
        propBlob.add<uint64_t>(dDescA);
        propBlob.add<uint64_t>(dShared);
        propBlob.add<uint8_t>(uint8_t(isVelocityIteration));
        propBlob.add<uint8_t>(0); // isLastPosIteration (unused by the kernel)
        propBlob.add<float>(0.7f); // biasCoefficient (unused by the kernel)
        propBlob.finish();
        PXG_CHECK_MSG(propBlob.bytes.size() == 24, "propagate blob is 24 bytes (got %zu)",
                      propBlob.bytes.size());
        PXG_CHECK_MSG(shim.launchKernel(1, propBlocks * 8, 1, 1, propBlob.bytes.data(),
                                        uint32_t(propBlob.bytes.size())) == PxgResult::Success,
                      "launch propagateAverageSolverBodyVelocityTGS");

        // WRITE_BACK_BODIES = 256, 1 thread/body; islandIndex = 1 (decoy at [0])
        EngineBlob wbBlob;
        wbBlob.add<uint64_t>(dDescB);
        wbBlob.add<uint64_t>(dShared);
        wbBlob.add<uint32_t>(1u); // islandIndex
        wbBlob.finish();
        PXG_CHECK_MSG(wbBlob.bytes.size() == 24, "writeback blob is 24 bytes (got %zu)",
                      wbBlob.bytes.size());
        PXG_CHECK_MSG(shim.launchKernel(2, (totalBodyCount + 255) / 256, 1, 1, wbBlob.bytes.data(),
                                        uint32_t(wbBlob.bytes.size())) == PxgResult::Success,
                      "launch writeBackBodiesTGS");
        shim.streamSynchronize();

        // ---- reference ----
        std::vector<Float4> refVel = velPool;
        std::vector<uint32_t> refEnc = encoded;
        std::vector<SolverTxIData> refTxi = txi;
        std::vector<Float4> refMotion = motionVel;
        auto popcount = [](uint32_t v) { uint32_t c = 0; while (v) { c += v & 1; v >>= 1; } return c; };

        // average pass (dynamic bodies only)
        size_t cNegZ = 0, cNegW = 0;   // P9.86f: bodies whose -0 z / w lane the (+0)+ add turns +0
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
                // (+0) + vel1.z / (+0) + vel1.w, the CUDA's untouched curVel1 lanes (P9.86f).
                // avgZwVariant replays the pre-P9.86f port (lanes passed through): 1 = z, 2 = w.
                auto plus0 = [](float x) { return (x == 0.f) ? 0.f : x; };
                refVel[bodyOffset + deltaVOffset + bodyId + T] =
                    Float4{ vel1.x + acc[4], vel1.y + acc[5], plus0(vel1.z), plus0(vel1.w) };
                if (std::signbit(vel1.z) && vel1.z == 0.f) ++cNegZ;
                if (std::signbit(vel1.w) && vel1.w == 0.f) ++cNegW;
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
        std::vector<Float4> gpuVel(poolSize);
        std::vector<uint32_t> gpuEnc(encoded.size());
        std::vector<SolverTxIData> gpuTxi(txi.size());
        std::vector<Float4> gpuMotion(motionVel.size());
        shim.memcpyDtoH(gpuVel.data(), dVel, gpuVel.size() * sizeof(Float4));
        shim.memcpyDtoH(gpuEnc.data(), dEnc, gpuEnc.size() * 4);
        shim.memcpyDtoH(gpuTxi.data(), dTx, gpuTxi.size() * sizeof(SolverTxIData));
        shim.memcpyDtoH(gpuMotion.data(), dMv, gpuMotion.size() * sizeof(Float4));

        // P9.86f: the averaged deltaV slot's z/w lanes (delta position x/y) are copied, not computed, so
        // they are compared BITWISE -- in the velocity iteration only: the position iteration's
        // propagate adds deltaP to them, which washes a zero's sign on both sides.  The pass-through
        // twin (pre-P9.86f port) must differ on every counted -0 lane; motion velocity = lane * invDt
        // keeps the sign through writeBackBodiesTGS.
        PXG_CHECK_MSG(cNegZ >= 4 && cNegW >= 4, "slabs=%u: -0 z/w coverage thin (%zu / %zu)", numSlabs, cNegZ, cNegW);
        if (isVelocityIteration)
        {
            size_t zwMm = 0, zwMotionMm = 0, zwTwinZ = 0, zwTwinW = 0;
            for (uint32_t bodyId = 0; bodyId < totalBodyCount; ++bodyId)
            {
                const size_t i = size_t(bodyOffset) + deltaVOffset + bodyId + T;
                if (f2uTGS(gpuVel[i].z) != f2uTGS(refVel[i].z) || f2uTGS(gpuVel[i].w) != f2uTGS(refVel[i].w)) ++zwMm;
                const Float4& m = gpuMotion[bodyOffset + bodyId];
                const Float4& rm = refMotion[bodyOffset + bodyId];
                if (f2uTGS(m.x) != f2uTGS(rm.x) || f2uTGS(m.y) != f2uTGS(rm.y)) ++zwMotionMm;
                // twin: the port without the select would keep the input lane
                const Float4& in1 = velPool[i];
                if (f2uTGS(in1.z) != f2uTGS(refVel[i].z)) ++zwTwinZ;
                if (f2uTGS(in1.w) != f2uTGS(refVel[i].w)) ++zwTwinW;
            }
            PXG_CHECK_MSG(zwMm == 0, "slabs=%u: %zu averaged deltaV z/w lanes differ bitwise from the CUDA's (+0)+v", numSlabs, zwMm);
            PXG_CHECK_MSG(zwMotionMm == 0, "slabs=%u: %zu motion-velocity x/y lanes differ bitwise", numSlabs, zwMotionMm);
            PXG_CHECK_MSG(zwTwinZ == cNegZ && zwTwinW == cNegW,
                          "slabs=%u: pass-through twin changes %zu/%zu z/w lanes, want %zu/%zu", numSlabs, zwTwinZ, zwTwinW, cNegZ, cNegW);
        }

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

        for (PxgDevicePtr pfree : { dVel, dEnc, dRef, dTx, dBd, dBs, dMv, dIslA, dIslB, dDescA, dDescB, dShared })
            shim.memFree(pfree);
    }
}

// ---------------------------------------------------------------------------
// solveWholeIslandTGS (ENGINE ABI, P5 Step 4)
// ---------------------------------------------------------------------------
// (PxgSolverCoreDesc*, PxgSolverSharedDesc<IterativeSolveDataTGS>*, PxU32
// islandIndex, bool doFriction, PxReal elapsedTime, PxReal minPen):
// argSizes {8,8,4,1,4,4} -> 32-byte blob (bool byte at 20, floats at 24/28).
// Launch shape is fixed: grid (1,1,1), block (256,1,1).

// Free-fall island: one dynamic body behind a kinematic offset and ZERO
// batches (two partitions whose cumulative batch counts stay at the island's
// mBatchStartIndex).  The kernel must pass the staged velocities through from
// the deltaV region to averageOutputOffsets + ComputeAverageBodyOutputIndex
// (+32/+64 rows) bit-exactly, honour the solverBodyReferences 0xFFFFFFFF
// gate, and leave the (enabled) contactErrorAccumulator bit-identical
// (every warp atomically adds an all-zero partial).  islandIndex = 1 against
// a decoy island [0] pins the PxgIslandContext stride on this ABI row too.
PXG_TEST(solverTGS_wholeIslandFreeFall)
{
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    std::mt19937 rng(3600);
    std::uniform_real_distribution<float> sym(-3.f, 3.f);

    const uint32_t bodyOffset = 2;
    const uint32_t numDynamicBodies = 1;
    const uint32_t T = numDynamicBodies + bodyOffset; // 3
    const uint32_t numSlabs = 2;
    const uint32_t numBatches = 1;         // desc.numBatches (offset arithmetic only)
    const uint32_t averageOutputOffsets = numBatches * 32 * 3 * 2; // 192
    const uint32_t deltaVOffset = 300;
    const uint32_t poolSize = deltaVOffset + 3 * T;

    std::vector<Float4> velPool(poolSize);
    for (Float4& v : velPool)
        v = Float4{ sym(rng), sym(rng), sym(rng), sym(rng) };

    // slab slot 0 active, slot 1 culled by the 0xFFFFFFFF gate
    std::vector<uint32_t> references = { 7u, 0xFFFFFFFFu };

    // cumulative per-partition batch END indices; index 0 is a decoy
    // (mStartPartitionIndex = 1), both partitions stay empty at
    // mBatchStartIndex = 0.
    std::vector<uint32_t> cpp = { 999u, 0u, 0u };

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "solverTGS.spv", "solveWholeIslandTGS", 32u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &solverTgsAbiLoader);

    PxgDevicePtr dVel = 0, dRef = 0, dCpp = 0, dIsl = 0, dDesc = 0, dShared = 0;
    shim.memAlloc(&dVel, velPool.size() * sizeof(Float4));
    shim.memAlloc(&dRef, references.size() * 4);
    shim.memAlloc(&dCpp, cpp.size() * 4);
    if (!dVel || !dRef || !dCpp)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    shim.memcpyHtoD(dVel, velPool.data(), velPool.size() * sizeof(Float4));
    shim.memcpyHtoD(dRef, references.data(), references.size() * 4);
    shim.memcpyHtoD(dCpp, cpp.data(), cpp.size() * 4);

    // island array: [0] is a DECOY, [1] is the real island (islandIndex = 1)
    physx::PxgIslandContext islands[2];
    std::memset(islands, 0, sizeof(islands));
    islands[0].mBodyStartIndex = 77u;
    islands[0].mBodyCount = 13u;
    islands[0].mNumPartitions = 0u;
    islands[1].mBodyStartIndex = bodyOffset;
    islands[1].mBodyCount = numDynamicBodies;
    islands[1].mArticulationCount = 0;
    islands[1].mStartPartitionIndex = 1;
    islands[1].mNumPartitions = 2;
    islands[1].mBatchStartIndex = 0;
    shim.memAlloc(&dIsl, sizeof(islands));
    shim.memcpyHtoD(dIsl, islands, sizeof(islands));

    physx::PxgSolverCoreDesc desc;
    std::memset(&desc, 0, sizeof(desc));
    desc.islandContextPool = reinterpret_cast<physx::PxgIslandContext*>(dIsl);
    desc.constraintsPerPartition = reinterpret_cast<physx::PxU32*>(dCpp);
    desc.solverBodyReferences = reinterpret_cast<physx::PxgSolverReferences*>(dRef);
    desc.numSlabs = numSlabs;
    desc.numBatches = numBatches;
    desc.numArticBatches = 0;
    desc.accumulatedBodyDeltaVOffset = deltaVOffset;
    desc.contactErrorAccumulator.mErrorSumOfSquares = 0.5f;
    desc.contactErrorAccumulator.mCounter = 5; // >= 0: residual accumulation ON
    desc.contactErrorAccumulator.mMaxError = 0.125f;

    physx::PxgSolverSharedDesc<physx::IterativeSolveDataTGS> shared;
    std::memset(&shared, 0, sizeof(shared));
    shared.iterativeData.solverBodyVelPool = reinterpret_cast<float4*>(dVel);

    shim.memAlloc(&dDesc, sizeof(desc));
    shim.memcpyHtoD(dDesc, &desc, sizeof(desc));
    shim.memAlloc(&dShared, sizeof(shared));
    shim.memcpyHtoD(dShared, &shared, sizeof(shared));

    // engine blob: {8,8,4,1,4,4} -> 32 bytes (bytes 21..23 zero-filled)
    EngineBlob blob;
    blob.add<uint64_t>(dDesc);
    blob.add<uint64_t>(dShared);
    blob.add<uint32_t>(1u);      // islandIndex
    blob.add<uint8_t>(1u);       // doFriction (unused by the kernel body)
    blob.add<float>(0.004f);     // elapsedTime
    blob.add<float>(-0.02f);     // minPen
    blob.finish();
    PXG_CHECK_MSG(blob.bytes.size() == 32, "whole-island blob is 32 bytes (got %zu)", blob.bytes.size());

    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, blob.bytes.data(), uint32_t(blob.bytes.size())) ==
                      PxgResult::Success,
                  "launch solveWholeIslandTGS");
    shim.streamSynchronize();

    // ---- reference: pure passthrough for the single active slab slot ----
    std::vector<Float4> refVel = velPool;
    {
        // bodyIndex 0 (bodyId 0, slabId 0): readIndex = bodyOffset + 0*T = 2,
        // staged from inputBody = deltaVOffset + (2 % T) = deltaVOffset + 2.
        const uint32_t velIndex = averageOutputOffsets + 0;
        refVel[velIndex] = velPool[deltaVOffset + 2];
        refVel[velIndex + 32] = velPool[deltaVOffset + 2 + T];
        refVel[velIndex + 64] = velPool[deltaVOffset + 2 + 2 * T];
        // bodyIndex 1 (slabId 1) is gated off by reference 0xFFFFFFFF.
    }

    std::vector<Float4> gpuVel(poolSize);
    shim.memcpyDtoH(gpuVel.data(), dVel, gpuVel.size() * sizeof(Float4));

    size_t mismatches = 0;
    for (size_t i = 0; i < poolSize; ++i)
        if (std::memcmp(&gpuVel[i], &refVel[i], sizeof(Float4)) != 0)
        {
            if (mismatches < 5)
                PXG_CHECK_MSG(false, "vel[%zu]: gpu=(%g %g %g %g) ref=(%g %g %g %g)", i, gpuVel[i].x,
                              gpuVel[i].y, gpuVel[i].z, gpuVel[i].w, refVel[i].x, refVel[i].y, refVel[i].z,
                              refVel[i].w);
            ++mismatches;
        }
    PXG_CHECK_MSG(mismatches == 0, "free-fall island: %zu velocity-pool mismatches", mismatches);

    // the enabled accumulator absorbed 8 all-zero warp partials: bit-identical
    physx::PxgSolverCoreDesc gpuDesc;
    shim.memcpyDtoH(&gpuDesc, dDesc, sizeof(gpuDesc));
    PXG_CHECK_MSG(std::memcmp(&gpuDesc.contactErrorAccumulator, &desc.contactErrorAccumulator,
                              sizeof(desc.contactErrorAccumulator)) == 0,
                  "free-fall island: contactErrorAccumulator unchanged (sum=%g count=%d max=%g)",
                  gpuDesc.contactErrorAccumulator.mErrorSumOfSquares,
                  gpuDesc.contactErrorAccumulator.mCounter, gpuDesc.contactErrorAccumulator.mMaxError);

    for (PxgDevicePtr pfree : { dVel, dRef, dCpp, dIsl, dDesc, dShared })
        shim.memFree(pfree);
}

// Contact + 1D-joint island: 4 live batches (plus a decoy at index 0 pinning
// mBatchStartIndex) across 2 partitions, mixed contact/joint, partial
// descStrides, kinematic bodies below bodyOffset, and the contact-residual
// statistics accumulated into the REAL engine desc's contactErrorAccumulator.
// The CPU reference is the same solverBlockTGS.cuh transcription the
// bound-buffer test uses, staged through model shared-memory arrays in the
// whole-island kernel's partition order; every (body, slab) pair is unique
// within a partition, so batch order inside a partition cannot matter.
PXG_TEST(solverTGS_wholeIslandSolve)
{
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    std::mt19937 rng(3700);
    std::uniform_real_distribution<float> unit(0.f, 1.f);
    std::uniform_real_distribution<float> sym(-2.f, 2.f);

    const uint32_t bodyOffset = 4;
    const uint32_t numDynamicBodies = 36;
    const uint32_t T = numDynamicBodies + bodyOffset; // 40 (also numTotalBodies: no articulations)
    const uint32_t numSlabs = 4;
    const uint32_t num32Slabs = 1;
    const uint32_t rowsPerBatch = 4;
    const uint32_t numAllBatches = 5;      // [0] = decoy, 1..4 live
    const uint32_t batchStartIndex = 1;
    const float elapsedTime = 0.005f;
    const float minPen = -0.02f;

    const uint32_t averageOutputOffsets = numAllBatches * 32 * 3 * 2; // 960
    const uint32_t deltaVOffset = 1600;
    const uint32_t poolSize = deltaVOffset + 3 * T;

    // batches 1,2 -> partition 0 (warp 0/1), batches 3,4 -> partition 1
    std::vector<uint32_t> cpp = { 123u, 3u, 5u }; // index 0 is a decoy (mStartPartitionIndex = 1)
    // engine PxgConstraintType encoding: eCONTACT=1, eCONSTRAINT_1D=0
    const uint16_t types[5] = { 1, 1, 0, 0, 1 };  // k: 1=contact 2=1D 3=1D 4=contact
    const uint16_t strides[5] = { 0, 32, 27, 32, 20 };

    std::vector<BatchTGS> batches(numAllBatches);
    std::vector<TGSContactHeader> chs(numAllBatches - 1);
    std::vector<TGSFrictionHeader> fhs(numAllBatches - 1);
    std::vector<TGSContactPoint> cps((numAllBatches - 1) * rowsPerBatch);
    std::vector<TGSFriction> frs((numAllBatches - 1) * (rowsPerBatch + 1));
    std::vector<TGSJointHeader> jhs(numAllBatches - 1);
    std::vector<TGSJointRowCon> jrows((numAllBatches - 1) * rowsPerBatch);
    std::vector<SolverTxIData> txi(T);
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
        const Mat33 rot(randomQuatTGS(rng));
        const Vec3 d(unit(rng), unit(rng), unit(rng));
        const Mat33 m = rot * Mat33(Vec3(d.x, 0, 0), Vec3(0, d.y, 0), Vec3(0, 0, d.z)) * rot.getTranspose();
        m.store(t.sqrtInvInertia);
    }

    std::vector<Float4> velPool(poolSize);
    for (Float4& v : velPool)
        v = Float4{ sym(rng), sym(rng), sym(rng), sym(rng) };

    std::vector<uint32_t> encoded(T * num32Slabs);
    for (uint32_t& e : encoded)
        e = rng();

    std::vector<uint32_t> references(numDynamicBodies * numSlabs, 0xFFFFFFFFu);
    for (uint32_t i = 0; i < references.size(); ++i)
        if ((rng() & 3) != 0) // ~75% of slab slots active; the value itself is unused here
            references[i] = i;

    for (uint32_t k = 1; k < numAllBatches; ++k)
    {
        BatchTGS& b = batches[k];
        b.descStride = strides[k];
        b.constraintType = types[k];
        b.scalars[1] = k - 1;                       // constraintBatchIndex (decoupled from k)
        b.scalars[2] = (k - 1) * rowsPerBatch;      // startConstraintIndex
        b.scalars[3] = (k - 1) * (rowsPerBatch + 1); // startFrictionIndex

        // batch j (0/1) within its partition; every (body, slab) pair unique
        // inside a partition: pairId = j*32 + lane -> slab = pairId % numSlabs,
        // bodies 2*(pairId/numSlabs), +1 (0..31 < T; 0..3 are "kinematic")
        const uint32_t j = (k - 1) & 1u;

        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            if (lane >= b.descStride)
                continue; // tail lanes stay zero: the kernel must not use them

            const uint32_t pairId = j * 32 + lane;
            b.slabId[lane] = pairId % numSlabs;
            b.bodyAIndex[lane] = 2 * (pairId / numSlabs);
            b.bodyBIndex[lane] = 2 * (pairId / numSlabs) + 1;

            if (b.constraintType == 1) // eCONTACT
            {
                TGSContactHeader& h = chs[b.scalars[1]];
                h.invMass0_1_angDom0_1[lane] = Float4{ unit(rng), unit(rng), unit(rng), unit(rng) };
                const Vec3 n = randUnitV(rng);
                h.normal_staticFriction[lane] = Float4{ n.x, n.y, n.z, unit(rng) };
                h.maxPenBias[lane] = -unit(rng) * 5.f;
                h.flags[lane] = unit(rng) < 0.5f ? COMPLIANT_ACCEL_SPRING : 0;
                h.numNormalConstr[lane] = rng() % (rowsPerBatch + 1);
                h.restitutionXdt[lane] = unit(rng) < 0.5f ? 0.f : -unit(rng);
                h.p8[lane] = 0.8f;

                TGSFrictionHeader& fh = fhs[b.scalars[1]];
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
                TGSJointHeader& h = jhs[b.scalars[1]];
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

    // ---- GPU (engine ABI through the shim) ----
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "solverTGS.spv", "solveWholeIslandTGS", 32u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &solverTgsAbiLoader);

    PxgDevicePtr dVel = 0, dEnc = 0, dRef = 0, dTx = 0, dCpp = 0, dIsl = 0;
    PxgDevicePtr dBatch = 0, dCh = 0, dFh = 0, dCp = 0, dFr = 0, dJh = 0, dJr = 0;
    PxgDevicePtr dDesc = 0, dShared = 0;
    shim.memAlloc(&dVel, velPool.size() * sizeof(Float4));
    shim.memAlloc(&dEnc, encoded.size() * 4);
    shim.memAlloc(&dRef, references.size() * 4);
    shim.memAlloc(&dTx, txi.size() * sizeof(SolverTxIData));
    shim.memAlloc(&dCpp, cpp.size() * 4);
    shim.memAlloc(&dBatch, batches.size() * sizeof(BatchTGS));
    shim.memAlloc(&dCh, chs.size() * sizeof(TGSContactHeader));
    shim.memAlloc(&dFh, fhs.size() * sizeof(TGSFrictionHeader));
    shim.memAlloc(&dCp, cps.size() * sizeof(TGSContactPoint));
    shim.memAlloc(&dFr, frs.size() * sizeof(TGSFriction));
    shim.memAlloc(&dJh, jhs.size() * sizeof(TGSJointHeader));
    shim.memAlloc(&dJr, jrows.size() * sizeof(TGSJointRowCon));
    if (!dVel || !dEnc || !dRef || !dTx || !dCpp || !dBatch || !dCh || !dFh || !dCp || !dFr || !dJh || !dJr)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    shim.memcpyHtoD(dVel, velPool.data(), velPool.size() * sizeof(Float4));
    shim.memcpyHtoD(dEnc, encoded.data(), encoded.size() * 4);
    shim.memcpyHtoD(dRef, references.data(), references.size() * 4);
    shim.memcpyHtoD(dTx, txi.data(), txi.size() * sizeof(SolverTxIData));
    shim.memcpyHtoD(dCpp, cpp.data(), cpp.size() * 4);
    shim.memcpyHtoD(dBatch, batches.data(), batches.size() * sizeof(BatchTGS));
    shim.memcpyHtoD(dCh, chs.data(), chs.size() * sizeof(TGSContactHeader));
    shim.memcpyHtoD(dFh, fhs.data(), fhs.size() * sizeof(TGSFrictionHeader));
    shim.memcpyHtoD(dCp, cps.data(), cps.size() * sizeof(TGSContactPoint));
    shim.memcpyHtoD(dFr, frs.data(), frs.size() * sizeof(TGSFriction));
    shim.memcpyHtoD(dJh, jhs.data(), jhs.size() * sizeof(TGSJointHeader));
    shim.memcpyHtoD(dJr, jrows.data(), jrows.size() * sizeof(TGSJointRowCon));

    physx::PxgIslandContext islands[2];
    std::memset(islands, 0, sizeof(islands));
    islands[0].mBodyStartIndex = 88u; // decoy
    islands[0].mBodyCount = 9u;
    islands[1].mBodyStartIndex = bodyOffset;
    islands[1].mBodyCount = numDynamicBodies;
    islands[1].mArticulationCount = 0;
    islands[1].mStartPartitionIndex = 1;
    islands[1].mNumPartitions = 2;
    islands[1].mBatchStartIndex = batchStartIndex;
    shim.memAlloc(&dIsl, sizeof(islands));
    shim.memcpyHtoD(dIsl, islands, sizeof(islands));

    physx::PxgSolverCoreDesc desc;
    std::memset(&desc, 0, sizeof(desc));
    desc.islandContextPool = reinterpret_cast<physx::PxgIslandContext*>(dIsl);
    desc.constraintsPerPartition = reinterpret_cast<physx::PxU32*>(dCpp);
    desc.solverBodyReferences = reinterpret_cast<physx::PxgSolverReferences*>(dRef);
    desc.solverBodyTxIDataPool = reinterpret_cast<physx::PxgSolverTxIData*>(dTx);
    desc.numSlabs = numSlabs;
    desc.numBatches = numAllBatches;
    desc.numArticBatches = 0;
    desc.accumulatedBodyDeltaVOffset = deltaVOffset;
    desc.contactErrorAccumulator.mErrorSumOfSquares = 0.5f;
    desc.contactErrorAccumulator.mCounter = 0; // >= 0: residual accumulation ON
    desc.contactErrorAccumulator.mMaxError = 0.0625f;

    physx::PxgSolverSharedDesc<physx::IterativeSolveDataTGS> shared;
    std::memset(&shared, 0, sizeof(shared));
    shared.iterativeData.solverBodyVelPool = reinterpret_cast<float4*>(dVel);
    shared.iterativeData.solverEncodedReferenceCount = reinterpret_cast<physx::PxU32*>(dEnc);
    shared.iterativeData.blockConstraintBatch = reinterpret_cast<physx::PxgBlockConstraintBatch*>(dBatch);
    shared.iterativeData.blockContactHeaders = reinterpret_cast<physx::PxgTGSBlockSolverContactHeader*>(dCh);
    shared.iterativeData.blockFrictionHeaders = reinterpret_cast<physx::PxgTGSBlockSolverFrictionHeader*>(dFh);
    shared.iterativeData.blockContactPoints = reinterpret_cast<physx::PxgTGSBlockSolverContactPoint*>(dCp);
    shared.iterativeData.blockFrictions = reinterpret_cast<physx::PxgTGSBlockSolverContactFriction*>(dFr);
    shared.iterativeData.blockJointConstraintHeaders =
        reinterpret_cast<physx::PxgTGSBlockSolverConstraint1DHeader*>(dJh);
    shared.iterativeData.blockJointConstraintRowsCon =
        reinterpret_cast<physx::PxgTGSBlockSolverConstraint1DCon*>(dJr);

    shim.memAlloc(&dDesc, sizeof(desc));
    shim.memcpyHtoD(dDesc, &desc, sizeof(desc));
    shim.memAlloc(&dShared, sizeof(shared));
    shim.memcpyHtoD(dShared, &shared, sizeof(shared));

    EngineBlob blob;
    blob.add<uint64_t>(dDesc);
    blob.add<uint64_t>(dShared);
    blob.add<uint32_t>(1u);   // islandIndex
    blob.add<uint8_t>(0u);    // doFriction (unused by the kernel body)
    blob.add<float>(elapsedTime);
    blob.add<float>(minPen);
    blob.finish();
    PXG_CHECK_MSG(blob.bytes.size() == 32, "whole-island blob is 32 bytes (got %zu)", blob.bytes.size());

    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, blob.bytes.data(), uint32_t(blob.bytes.size())) ==
                      PxgResult::Success,
                  "launch solveWholeIslandTGS");
    shim.streamSynchronize();

    // ---- reference (kernel transcription over model shared arrays) ----
    const uint32_t numStaged = T * numSlabs;
    std::vector<Vec3> shLin(numStaged), shAng(numStaged), shLinD(numStaged), shAngD(numStaged);
    for (uint32_t j = 0; j < numStaged; ++j)
    {
        const Float4 v0 = velPool[deltaVOffset + (j % T)];
        const Float4 v1 = velPool[deltaVOffset + (j % T) + T];
        const Float4 v2 = velPool[deltaVOffset + (j % T) + 2 * T];
        shLin[j] = Vec3(v0.x, v0.y, v0.z);
        shAng[j] = Vec3(v0.w, v1.x, v1.y);
        shLinD[j] = Vec3(v1.z, v1.w, v2.x);
        shAngD[j] = Vec3(v2.y, v2.z, v2.w);
    }

    auto popcount = [](uint32_t v) { uint32_t c = 0; while (v) { c += v & 1; v >>= 1; } return c; };

    RefErrAcc errAcc[8][32]; // per-thread accumulators (warp x lane)

    uint32_t startIndex = batchStartIndex;
    for (uint32_t p = 0; p < 2; ++p)
    {
        const uint32_t endIndex = cpp[p + 1];
        for (uint32_t k = startIndex; k < endIndex; ++k)
        {
            const uint32_t warp = (k - startIndex) % 8; // k = startIndex + warpIndex (+ n*numWarps)
            const BatchTGS& b = batches[k];
            for (uint32_t lane = 0; lane < b.descStride; ++lane)
            {
                const uint32_t rA = b.bodyAIndex[lane] + b.slabId[lane] * T;
                const uint32_t rB = b.bodyBIndex[lane] + b.slabId[lane] * T;

                BodyStateTGS s;
                s.lv0 = shLin[rA]; s.av0 = shAng[rA]; s.ld0 = shLinD[rA]; s.ad0 = shAngD[rA];
                s.lv1 = shLin[rB]; s.av1 = shAng[rB]; s.ld1 = shLinD[rB]; s.ad1 = shAngD[rB];

                float ref0 = 1.f, ref1 = 1.f;
                if (b.bodyAIndex[lane] >= bodyOffset)
                    ref0 = float(std::max(1u, popcount(encoded[b.bodyAIndex[lane]])));
                if (b.bodyBIndex[lane] >= bodyOffset)
                    ref1 = float(std::max(1u, popcount(encoded[b.bodyBIndex[lane]])));

                if (b.constraintType == 1) // eCONTACT
                    refSolveContactTGS(b, s, lane, chs, fhs, cps, frs, elapsedTime, minPen, ref0, ref1,
                                       &errAcc[warp][lane]);
                else
                    refSolve1DTGS(b, s, lane, jhs, jrows, txi[b.bodyAIndex[lane]], txi[b.bodyBIndex[lane]],
                                  elapsedTime, true, ref0, ref1);

                shLin[rA] = s.lv0; shAng[rA] = s.av0;
                shLin[rB] = s.lv1; shAng[rB] = s.av1;
            }
        }
        startIndex = endIndex;
    }

    std::vector<Float4> refVel = velPool;
    for (uint32_t idx = 0; idx < numDynamicBodies * numSlabs; ++idx)
    {
        if (references[idx] == 0xFFFFFFFFu)
            continue;
        const uint32_t velIndex = averageOutputOffsets + 3 * (idx & ~31u) + (idx & 31u);
        const uint32_t readIndex = idx / numSlabs + bodyOffset + (idx % numSlabs) * T;
        const Vec3 lv = shLin[readIndex], av = shAng[readIndex];
        const Vec3 ld = shLinD[readIndex], ad = shAngD[readIndex];
        refVel[velIndex] = Float4{ lv.x, lv.y, lv.z, av.x };
        refVel[velIndex + 32] = Float4{ av.y, av.z, ld.x, ld.y };
        refVel[velIndex + 64] = Float4{ ld.z, ad.x, ad.y, ad.z };
    }

    // accumulateErrorGlobalFullWarp reference: the __shfl_xor butterfly
    // (radius 16..1) per warp, then the per-warp partials into the
    // accumulator (the sum's warp order is nondeterministic -> tolerance)
    float expectSum = 0.5f;
    uint32_t expectCount = 0;
    float expectMax = 0.0625f;
    for (uint32_t w = 0; w < 8; ++w)
    {
        float sums[32], maxs[32];
        uint32_t cnts[32];
        for (uint32_t l = 0; l < 32; ++l)
        {
            sums[l] = errAcc[w][l].sumSq;
            maxs[l] = errAcc[w][l].maxErr;
            cnts[l] = errAcc[w][l].counter;
        }
        for (uint32_t radius = 16; radius > 0; radius >>= 1)
        {
            float ns[32], nm[32];
            uint32_t nc[32];
            for (uint32_t l = 0; l < 32; ++l)
            {
                ns[l] = sums[l] + sums[l ^ radius];
                nm[l] = std::fmax(maxs[l], maxs[l ^ radius]);
                nc[l] = cnts[l] + cnts[l ^ radius];
            }
            std::memcpy(sums, ns, sizeof(ns));
            std::memcpy(maxs, nm, sizeof(nm));
            std::memcpy(cnts, nc, sizeof(nc));
        }
        expectSum += sums[0];
        expectCount += cnts[0];
        expectMax = std::fmax(expectMax, maxs[0]);
    }
    PXG_CHECK_MSG(expectCount > 0, "fixture sanity: the residual statistics path was exercised");

    // ---- compare ----
    std::vector<Float4> gpuVel(poolSize);
    shim.memcpyDtoH(gpuVel.data(), dVel, gpuVel.size() * sizeof(Float4));
    std::vector<TGSContactPoint> gpuCps(cps.size());
    shim.memcpyDtoH(gpuCps.data(), dCp, gpuCps.size() * sizeof(TGSContactPoint));
    std::vector<TGSFriction> gpuFrs(frs.size());
    shim.memcpyDtoH(gpuFrs.data(), dFr, gpuFrs.size() * sizeof(TGSFriction));
    std::vector<TGSFrictionHeader> gpuFhs(fhs.size());
    shim.memcpyDtoH(gpuFhs.data(), dFh, gpuFhs.size() * sizeof(TGSFrictionHeader));
    std::vector<TGSJointRowCon> gpuJrows(jrows.size());
    shim.memcpyDtoH(gpuJrows.data(), dJr, gpuJrows.size() * sizeof(TGSJointRowCon));

    size_t mismatches = 0;
    for (size_t i = 0; i < poolSize; ++i)
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
    PXG_CHECK_MSG(mismatches == 0, "whole island: %zu velocity-pool mismatches", mismatches);

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
    PXG_CHECK_MSG(mismatches == 0, "whole island: %zu applied-force/residual/broken mismatches", mismatches);

    physx::PxgSolverCoreDesc gpuDesc;
    shim.memcpyDtoH(&gpuDesc, dDesc, sizeof(gpuDesc));
    PXG_CHECK_MSG(almostEqual(gpuDesc.contactErrorAccumulator.mErrorSumOfSquares, expectSum, 2e-4f, 2e-4f),
                  "residual sum-of-squares: gpu=%g ref=%g", gpuDesc.contactErrorAccumulator.mErrorSumOfSquares,
                  expectSum);
    PXG_CHECK_MSG(gpuDesc.contactErrorAccumulator.mCounter == int32_t(expectCount),
                  "residual counter: gpu=%d ref=%u", gpuDesc.contactErrorAccumulator.mCounter, expectCount);
    PXG_CHECK_MSG(almostEqual(gpuDesc.contactErrorAccumulator.mMaxError, expectMax, 2e-4f, 2e-4f),
                  "residual max: gpu=%g ref=%g", gpuDesc.contactErrorAccumulator.mMaxError, expectMax);

    for (PxgDevicePtr pfree : { dVel, dEnc, dRef, dTx, dCpp, dIsl, dBatch, dCh, dFh, dCp, dFr, dJh, dJr,
                                dDesc, dShared })
        shim.memFree(pfree);
}

// ---------------------------------------------------------------------------
// solveBlockUnified (ENGINE ABI, P5 Step 5 tranche A BDA) — the per-partition
// dynamic-vs-dynamic rigid contact/joint velocity solve.
// ---------------------------------------------------------------------------
// The engine launches solveBlockUnified ONCE PER PARTITION with the current
// partition index in the push blob; each launch solves that one partition's
// rigid batches out of the per-partition solverBodyVelPool (readIndex =
// k*192+lane, writeback to remappedBodyA/BIndex — the same staging as the
// bound-buffer solveBlockPartitionTGS, NOT the deltaV/slab staging of
// solveWholeIslandTGS).  This test fills the REAL engine descs (+ a 2-element
// PxgIslandContext array; element [0] is a decoy so islandIndex = 1 pins the
// stride), one partition (mStartPartitionIndex = 1) holding ONE dynamic-dynamic
// CONTACT batch (eCONTACT=1) and ONE 1D-joint batch (eCONSTRAINT_1D=0), and
// launches through PxgCudaShim with the 48-byte blob (islandIndex=1,
// partitionIndex=0, doFriction=1).  numthreads = (32,2,1); grid (1,1,1) => 2
// warps => warp 0 solves the contact batch, warp 1 the joint batch.  Body
// velocities, appliedForce/residual/broken, and the contact-residual
// accumulator are all checked BIT-EXACT (2e-4) against the proven
// refSolveContactTGS / refSolve1DTGS transcriptions.
PXG_TEST(solverTGS_solveBlockUnified)
{
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    std::mt19937 rng(3900);
    std::uniform_real_distribution<float> unit(0.f, 1.f);
    std::uniform_real_distribution<float> sym(-2.f, 2.f);

    const uint32_t bodyOffset = 8;
    const uint32_t numDynamicBodies = 400;
    const uint32_t numTotalBodies = bodyOffset + numDynamicBodies; // no articulations
    const uint32_t numSlabs = 12;
    const uint32_t rowsPerBatch = 4;
    const float elapsedTime = 0.005f;
    const float minPen = -0.02f;

    // batches: [0] = decoy (never solved), [1] = contact, [2] = 1D joint.
    // mBatchStartIndex = 1, mStartPartitionIndex = 1, partition 0 spans
    // batches [1, cpp[1]) = [1, 3) = {1, 2}.
    const uint32_t numAllBatches = 3;
    const uint32_t batchStartIndex = 1;
    const uint16_t types[3] = { 0, 1, 0 };     // [1]=contact(eCONTACT=1), [2]=1D joint(0)
    const uint16_t strides[3] = { 0, 32, 27 }; // partial stride on the joint batch
    std::vector<uint32_t> cpp = { 999u, 3u };  // index 0 decoy; partition-0 endIndex = cpp[1] = 3

    std::vector<BatchTGS> batches(numAllBatches);
    std::vector<TGSContactHeader> chs(numAllBatches);
    std::vector<TGSFrictionHeader> fhs(numAllBatches);
    std::vector<TGSContactPoint> cps(numAllBatches * rowsPerBatch);
    std::vector<TGSFriction> frs(numAllBatches * (rowsPerBatch + 1));
    std::vector<TGSJointHeader> jhs(numAllBatches);
    std::vector<TGSJointRowCon> jrows(numAllBatches * rowsPerBatch);
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
        const Mat33 rot(randomQuatTGS(rng));
        const Vec3 d(unit(rng), unit(rng), unit(rng));
        const Mat33 m = rot * Mat33(Vec3(d.x, 0, 0), Vec3(0, d.y, 0), Vec3(0, 0, d.z)) * rot.getTranspose();
        m.store(t.sqrtInvInertia);
    }

    const uint32_t poolSize = numAllBatches * 192; // k*192 per-partition layout
    std::vector<Float4> velPool(poolSize);
    for (Float4& v : velPool)
        v = Float4{ sym(rng), sym(rng), sym(rng), sym(rng) };

    std::vector<uint32_t> encoded(numTotalBodies);
    for (uint32_t& e : encoded)
        e = rng();

    for (uint32_t k = batchStartIndex; k < numAllBatches; ++k)
    {
        BatchTGS& b = batches[k];
        b.descStride = strides[k];
        b.constraintType = types[k];
        b.scalars[1] = k;                       // constraintBatchIndex
        b.scalars[2] = k * rowsPerBatch;        // startConstraintIndex
        b.scalars[3] = k * (rowsPerBatch + 1);  // startFrictionIndex

        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            b.bodyAIndex[lane] = rng() % numTotalBodies;
            b.bodyBIndex[lane] = rng() % numTotalBodies;
            b.remappedBodyAIndex[lane] = k * 192 + lane;
            b.remappedBodyBIndex[lane] = k * 192 + lane + 96;

            if (b.constraintType == 1) // eCONTACT
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

    // P9.86c directed lane (batch 1 = eCONTACT, stride 32, lane 0): torsional scale -0.
    const uint32_t torsRow = injectTorsNegZeroScaleLane(batches[1], 0, chs, fhs, cps, frs, velPool, 1);

    // ---- GPU (engine ABI through the shim) ----
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "solverTGS.spv", "solveBlockUnified", 48u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &solverTgsAbiLoader);

    PxgDevicePtr dVel = 0, dEnc = 0, dTx = 0, dCpp = 0, dIsl = 0;
    PxgDevicePtr dBatch = 0, dCh = 0, dFh = 0, dCp = 0, dFr = 0, dJh = 0, dJr = 0;
    PxgDevicePtr dDesc = 0, dShared = 0;
    shim.memAlloc(&dVel, velPool.size() * sizeof(Float4));
    shim.memAlloc(&dEnc, encoded.size() * 4);
    shim.memAlloc(&dTx, txi.size() * sizeof(SolverTxIData));
    shim.memAlloc(&dCpp, cpp.size() * 4);
    shim.memAlloc(&dBatch, batches.size() * sizeof(BatchTGS));
    shim.memAlloc(&dCh, chs.size() * sizeof(TGSContactHeader));
    shim.memAlloc(&dFh, fhs.size() * sizeof(TGSFrictionHeader));
    shim.memAlloc(&dCp, cps.size() * sizeof(TGSContactPoint));
    shim.memAlloc(&dFr, frs.size() * sizeof(TGSFriction));
    shim.memAlloc(&dJh, jhs.size() * sizeof(TGSJointHeader));
    shim.memAlloc(&dJr, jrows.size() * sizeof(TGSJointRowCon));
    if (!dVel || !dEnc || !dTx || !dCpp || !dBatch || !dCh || !dFh || !dCp || !dFr || !dJh || !dJr)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    shim.memcpyHtoD(dVel, velPool.data(), velPool.size() * sizeof(Float4));
    shim.memcpyHtoD(dEnc, encoded.data(), encoded.size() * 4);
    shim.memcpyHtoD(dTx, txi.data(), txi.size() * sizeof(SolverTxIData));
    shim.memcpyHtoD(dCpp, cpp.data(), cpp.size() * 4);
    shim.memcpyHtoD(dBatch, batches.data(), batches.size() * sizeof(BatchTGS));
    shim.memcpyHtoD(dCh, chs.data(), chs.size() * sizeof(TGSContactHeader));
    shim.memcpyHtoD(dFh, fhs.data(), fhs.size() * sizeof(TGSFrictionHeader));
    shim.memcpyHtoD(dCp, cps.data(), cps.size() * sizeof(TGSContactPoint));
    shim.memcpyHtoD(dFr, frs.data(), frs.size() * sizeof(TGSFriction));
    shim.memcpyHtoD(dJh, jhs.data(), jhs.size() * sizeof(TGSJointHeader));
    shim.memcpyHtoD(dJr, jrows.data(), jrows.size() * sizeof(TGSJointRowCon));

    physx::PxgIslandContext islands[2];
    std::memset(islands, 0, sizeof(islands));
    islands[0].mBodyStartIndex = 55u; // decoy
    islands[0].mBodyCount = 7u;
    islands[1].mBodyStartIndex = bodyOffset;
    islands[1].mBodyCount = numDynamicBodies;
    islands[1].mArticulationCount = 0;
    islands[1].mStartPartitionIndex = 1;
    islands[1].mNumPartitions = 1;
    islands[1].mBatchStartIndex = batchStartIndex;
    shim.memAlloc(&dIsl, sizeof(islands));
    shim.memcpyHtoD(dIsl, islands, sizeof(islands));

    physx::PxgSolverCoreDesc desc;
    std::memset(&desc, 0, sizeof(desc));
    desc.islandContextPool = reinterpret_cast<physx::PxgIslandContext*>(dIsl);
    desc.constraintsPerPartition = reinterpret_cast<physx::PxU32*>(dCpp);
    desc.solverBodyTxIDataPool = reinterpret_cast<physx::PxgSolverTxIData*>(dTx);
    desc.numSlabs = numSlabs;
    desc.accumulatedBodyDeltaVOffset = 0;
    desc.contactErrorAccumulator.mErrorSumOfSquares = 0.5f;
    desc.contactErrorAccumulator.mCounter = 0; // >= 0: residual accumulation ON
    desc.contactErrorAccumulator.mMaxError = 0.0625f;

    physx::PxgSolverSharedDesc<physx::IterativeSolveDataTGS> shared;
    std::memset(&shared, 0, sizeof(shared));
    shared.iterativeData.solverBodyVelPool = reinterpret_cast<float4*>(dVel);
    shared.iterativeData.solverEncodedReferenceCount = reinterpret_cast<physx::PxU32*>(dEnc);
    shared.iterativeData.blockConstraintBatch = reinterpret_cast<physx::PxgBlockConstraintBatch*>(dBatch);
    shared.iterativeData.blockContactHeaders = reinterpret_cast<physx::PxgTGSBlockSolverContactHeader*>(dCh);
    shared.iterativeData.blockFrictionHeaders = reinterpret_cast<physx::PxgTGSBlockSolverFrictionHeader*>(dFh);
    shared.iterativeData.blockContactPoints = reinterpret_cast<physx::PxgTGSBlockSolverContactPoint*>(dCp);
    shared.iterativeData.blockFrictions = reinterpret_cast<physx::PxgTGSBlockSolverContactFriction*>(dFr);
    shared.iterativeData.blockJointConstraintHeaders =
        reinterpret_cast<physx::PxgTGSBlockSolverConstraint1DHeader*>(dJh);
    shared.iterativeData.blockJointConstraintRowsCon =
        reinterpret_cast<physx::PxgTGSBlockSolverConstraint1DCon*>(dJr);

    shim.memAlloc(&dDesc, sizeof(desc));
    shim.memcpyHtoD(dDesc, &desc, sizeof(desc));
    shim.memAlloc(&dShared, sizeof(shared));
    shim.memcpyHtoD(dShared, &shared, sizeof(shared));

    EngineBlob blob;
    blob.add<uint64_t>(dDesc);     // @0  solverCoreDesc
    blob.add<uint64_t>(dShared);   // @8  sharedDesc
    blob.add<uint32_t>(1u);        // @16 islandIndex
    blob.add<uint32_t>(0u);        // @20 partitionIndex
    blob.add<uint8_t>(1u);         // @24 doFriction (unused by the rigid body)
    blob.add<float>(elapsedTime);  // @28
    blob.add<float>(minPen);       // @32
    blob.add<uint64_t>(0u);        // @40 artiDesc (deferred; unused on the rigid path)
    blob.finish();
    PXG_CHECK_MSG(blob.bytes.size() == 48, "solveBlockUnified blob is 48 bytes (got %zu)", blob.bytes.size());

    // grid (maxBlocks, 1, 1); block (32, 2, 1) -> 2 warps.  maxBlocks =
    // ceil(batchesInPartition / numWarpsPerBlock) = ceil(2/2) = 1.
    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, blob.bytes.data(), uint32_t(blob.bytes.size())) ==
                      PxgResult::Success,
                  "launch solveBlockUnified");
    shim.streamSynchronize();

    // ---- reference (the proven solverBlockTGS.cuh transcription over the
    // per-partition k*192 pool; partition 0 = batches [batchStartIndex, cpp[1]))
    std::vector<Float4> refVel = velPool;
    auto popcount = [](uint32_t v) { uint32_t c = 0; while (v) { c += v & 1; v >>= 1; } return c; };

    const uint32_t numWarpsInGrid = 1u * 2u; // WorkgroupCount().x(1) * numthreads.y(2)
    RefErrAcc errAcc[2][32];                  // per-thread contact accumulators (warp x lane)

    const uint32_t startIndex = batchStartIndex;
    const uint32_t endIndex = cpp[1];
    gTorsClampRevertDiffers = 0;
    for (uint32_t k = startIndex; k < endIndex; ++k)
    {
        const uint32_t warp = (k - startIndex) % numWarpsInGrid;
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

            if (b.constraintType == 1) // eCONTACT
                refSolveContactTGS(b, s, lane, chs, fhs, cps, frs, elapsedTime, minPen, ref0, ref1,
                                   &errAcc[warp][lane]);
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

    // accumulateErrorGlobalFullWarp reference: __shfl_xor butterfly (radius
    // 16..1) per warp, then per-warp partials into the accumulator (order-
    // independent under tolerance).
    float expectSum = 0.5f;
    uint32_t expectCount = 0;
    float expectMax = 0.0625f;
    for (uint32_t w = 0; w < numWarpsInGrid; ++w)
    {
        float sums[32], maxs[32];
        uint32_t cnts[32];
        for (uint32_t l = 0; l < 32; ++l)
        {
            sums[l] = errAcc[w][l].sumSq;
            maxs[l] = errAcc[w][l].maxErr;
            cnts[l] = errAcc[w][l].counter;
        }
        for (uint32_t radius = 16; radius > 0; radius >>= 1)
        {
            float ns[32], nm[32];
            uint32_t nc[32];
            for (uint32_t l = 0; l < 32; ++l)
            {
                ns[l] = sums[l] + sums[l ^ radius];
                nm[l] = std::fmax(maxs[l], maxs[l ^ radius]);
                nc[l] = cnts[l] + cnts[l ^ radius];
            }
            std::memcpy(sums, ns, sizeof(ns));
            std::memcpy(maxs, nm, sizeof(nm));
            std::memcpy(cnts, nc, sizeof(nc));
        }
        expectSum += sums[0];
        expectCount += cnts[0];
        expectMax = std::fmax(expectMax, maxs[0]);
    }
    PXG_CHECK_MSG(expectCount > 0, "fixture sanity: the contact-residual path was exercised");

    // ---- compare ----
    std::vector<Float4> gpuVel(poolSize);
    shim.memcpyDtoH(gpuVel.data(), dVel, gpuVel.size() * sizeof(Float4));
    std::vector<TGSContactPoint> gpuCps(cps.size());
    shim.memcpyDtoH(gpuCps.data(), dCp, gpuCps.size() * sizeof(TGSContactPoint));
    std::vector<TGSFriction> gpuFrs(frs.size());
    shim.memcpyDtoH(gpuFrs.data(), dFr, gpuFrs.size() * sizeof(TGSFriction));
    std::vector<TGSFrictionHeader> gpuFhs(fhs.size());
    shim.memcpyDtoH(gpuFhs.data(), dFh, gpuFhs.size() * sizeof(TGSFrictionHeader));
    std::vector<TGSJointRowCon> gpuJrows(jrows.size());
    shim.memcpyDtoH(gpuJrows.data(), dJr, gpuJrows.size() * sizeof(TGSJointRowCon));

    size_t mismatches = 0;
    for (size_t i = 0; i < poolSize; ++i)
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
    PXG_CHECK_MSG(mismatches == 0, "solveBlockUnified: %zu velocity-pool mismatches", mismatches);

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
    PXG_CHECK_MSG(mismatches == 0, "solveBlockUnified: %zu applied-force/residual/broken mismatches",
                  mismatches);

    // P9.86c: the directed torsional lane, compared BITWISE (almostEqual accepts -0 == +0).
    PXG_CHECK_MSG(gTorsClampRevertDiffers > 0,
                  "solveBlockUnified: directed torsional lane (lo,hi)=(+0,-0), v<0 not reached");
    PXG_CHECK_MSG(tgsBits(frs[torsRow].appliedForce[0]) == 0u, "solveBlockUnified: CUDA PxClamp stores +0 (ref %08x)",
                  tgsBits(frs[torsRow].appliedForce[0]));
    PXG_CHECK_MSG(tgsBits(gpuFrs[torsRow].appliedForce[0]) == tgsBits(frs[torsRow].appliedForce[0]),
                  "solveBlockUnified torsional scale -0: gpu %08x ref %08x",
                  tgsBits(gpuFrs[torsRow].appliedForce[0]), tgsBits(frs[torsRow].appliedForce[0]));

    physx::PxgSolverCoreDesc gpuDesc;
    shim.memcpyDtoH(&gpuDesc, dDesc, sizeof(gpuDesc));
    PXG_CHECK_MSG(almostEqual(gpuDesc.contactErrorAccumulator.mErrorSumOfSquares, expectSum, 2e-4f, 2e-4f),
                  "residual sum-of-squares: gpu=%g ref=%g",
                  gpuDesc.contactErrorAccumulator.mErrorSumOfSquares, expectSum);
    PXG_CHECK_MSG(gpuDesc.contactErrorAccumulator.mCounter == int32_t(expectCount),
                  "residual counter: gpu=%d ref=%u", gpuDesc.contactErrorAccumulator.mCounter, expectCount);
    PXG_CHECK_MSG(almostEqual(gpuDesc.contactErrorAccumulator.mMaxError, expectMax, 2e-4f, 2e-4f),
                  "residual max: gpu=%g ref=%g", gpuDesc.contactErrorAccumulator.mMaxError, expectMax);

    for (PxgDevicePtr pfree : { dVel, dEnc, dTx, dCpp, dIsl, dBatch, dCh, dFh, dCp, dFr, dJh, dJr,
                                dDesc, dShared })
        shim.memFree(pfree);
}

// ---------------------------------------------------------------------------
// solveStaticBlockTGS + propagateStaticSolverBodyVelocitiesTGS (ENGINE ABI)
// ---------------------------------------------------------------------------
// The dynamic-vs-static rigid contact/joint solve.  Two dynamic bodies (lanes
// 0 and 1 of shared static batches, exercising warpScanExclusive), each with 3
// static contact batches split across 2 slabs (maxStaticPartitions = 2) plus a
// shared static 1D joint batch.  solveStaticBlockTGS writes per-slab body
// velocities to tempStaticBodyOutputs; propagateStaticSolverBodyVelocitiesTGS
// averages them back into the deltaV pool.  Both are checked bit-exact (2e-4)
// against the proven refSolveContactTGS / refSolve1DTGS transcriptions, and the
// contact-residual accumulator is checked (mutation + value).
PXG_TEST(solverTGS_staticContactSolve)
{
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    std::mt19937 rng(3900);
    std::uniform_real_distribution<float> unit(0.f, 1.f);
    std::uniform_real_distribution<float> sym(-2.f, 2.f);

    const uint32_t numDynamicBodies = 2;
    const uint32_t bodyOffset = 1;
    const uint32_t T = numDynamicBodies + bodyOffset; // 3
    const uint32_t nbStaticSlabs = 2;
    const uint32_t maxStaticPartitions = 2;
    const uint32_t numDynamicBodiesRounded = 32; // (2 + 31) & ~31
    const uint32_t rowsPerBatch = 4;
    const float elapsedTime = 0.005f;
    const float minPen = -0.02f;
    const uint32_t deltaVOffset = 100;
    const uint32_t poolSize = 200;

    // batch 0 = decoy; 1,2,3 = contacts (eCONTACT=1); 4 = joint (eCONSTRAINT_1D=0)
    const uint32_t numAllBatches = 5;
    const uint32_t contactStart = 1; // shared by both bodies
    const uint32_t jointStart = 4;

    std::vector<BatchTGS> batches(numAllBatches);
    std::vector<TGSContactHeader> chs(3);
    std::vector<TGSFrictionHeader> fhs(3);
    std::vector<TGSContactPoint> cps(3 * rowsPerBatch);
    std::vector<TGSFriction> frs(3 * (rowsPerBatch + 1));
    std::vector<TGSJointHeader> jhs(1);
    std::vector<TGSJointRowCon> jrows(rowsPerBatch);
    std::vector<SolverTxIData> txi(T);
    std::memset(batches.data(), 0, batches.size() * sizeof(BatchTGS));
    std::memset(chs.data(), 0, chs.size() * sizeof(TGSContactHeader));
    std::memset(fhs.data(), 0, fhs.size() * sizeof(TGSFrictionHeader));
    std::memset(cps.data(), 0, cps.size() * sizeof(TGSContactPoint));
    std::memset(frs.data(), 0, frs.size() * sizeof(TGSFriction));
    std::memset(jhs.data(), 0, jhs.size() * sizeof(TGSJointHeader));
    std::memset(jrows.data(), 0, jrows.size() * sizeof(TGSJointRowCon));
    std::memset(txi.data(), 0, txi.size() * sizeof(SolverTxIData));

    for (uint32_t b = bodyOffset; b < T; ++b)
    {
        SolverTxIData& t = txi[b];
        t.deltaBody2World.q = randomQuatTGS(rng).toQuat();
        t.deltaBody2World.p = Float3{ sym(rng) * 0.01f, sym(rng) * 0.01f, sym(rng) * 0.01f };
        const Mat33 rot(randomQuatTGS(rng));
        const Vec3 d(unit(rng), unit(rng), unit(rng));
        const Mat33 m = rot * Mat33(Vec3(d.x, 0, 0), Vec3(0, d.y, 0), Vec3(0, 0, d.z)) * rot.getTranspose();
        m.store(t.sqrtInvInertia);
    }

    std::vector<Float4> velPool(poolSize);
    for (Float4& v : velPool)
        v = Float4{ sym(rng), sym(rng), sym(rng), sym(rng) };

    // ---- contact / joint batch data (lanes 0 and 1 = bodies 0 and 1) ----
    auto fillContactLane = [&](uint32_t header, uint32_t lane) {
        TGSContactHeader& h = chs[header];
        h.invMass0_1_angDom0_1[lane] = Float4{ unit(rng), 0.f, unit(rng), 0.f }; // static B: invMassB=angDom1=0
        const Vec3 n = randUnitV(rng);
        h.normal_staticFriction[lane] = Float4{ n.x, n.y, n.z, unit(rng) };
        h.maxPenBias[lane] = -unit(rng) * 5.f;
        h.flags[lane] = unit(rng) < 0.5f ? COMPLIANT_ACCEL_SPRING : 0;
        h.numNormalConstr[lane] = 1 + rng() % rowsPerBatch; // >=1 so the solve does work
        h.restitutionXdt[lane] = unit(rng) < 0.5f ? 0.f : -unit(rng);
        h.p8[lane] = 0.8f;

        TGSFrictionHeader& fh = fhs[header];
        const uint32_t fr = (rng() % 3) * 2 + (unit(rng) < 0.3f ? 1 : 0);
        fh.numFrictionConstr[lane] = std::min(fr, rowsPerBatch + 1);
        fh.dynamicFriction[lane] = unit(rng);
        fh.biasCoefficient[lane] = unit(rng) * 10.f;
        fh.torsionalFrictionScale[lane] = 0.5f + unit(rng);
        const Vec3 f0 = randUnitV(rng), f1 = randUnitV(rng);
        fh.frictionNormals0[lane] = Float4{ f0.x, f0.y, f0.z, 0 };
        fh.frictionNormals1[lane] = Float4{ f1.x, f1.y, f1.z, 0 };

        for (uint32_t i = 0; i < rowsPerBatch; ++i)
        {
            TGSContactPoint& c = cps[header * rowsPerBatch + i];
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
            TGSFriction& f = frs[header * (rowsPerBatch + 1) + i];
            const Vec3 ra = Vec3(sym(rng), sym(rng), sym(rng)) * 0.3f;
            const Vec3 rb = Vec3(sym(rng), sym(rng), sym(rng)) * 0.3f;
            f.raXn_error[lane] = Float4{ ra.x, ra.y, ra.z, sym(rng) * 0.02f };
            f.rbXn_targetVelW[lane] = Float4{ rb.x, rb.y, rb.z, sym(rng) * 0.05f };
            f.appliedForce[lane] = sym(rng) * 0.2f;
            f.resp0[lane] = 0.1f + unit(rng);
            f.resp1[lane] = 0.1f + unit(rng);
        }
    };
    auto fillJointLane = [&](uint32_t lane) {
        TGSJointHeader& h = jhs[0];
        h.rowCounts_breakable_orthoAxisCount[lane] = 1 + rng() % rowsPerBatch;
        const Vec3 raw = Vec3(sym(rng), sym(rng), sym(rng)) * 0.3f;
        const Vec3 rbw = Vec3(sym(rng), sym(rng), sym(rng)) * 0.3f;
        h.rAWorld_invMass0D0[lane] = Float4{ raw.x, raw.y, raw.z, unit(rng) };
        h.rBWorld_invMass1D1[lane] = Float4{ rbw.x, rbw.y, rbw.z, 0.f }; // static B invMass1=0
        h.invInertiaScale0[lane] = unit(rng);
        h.invInertiaScale1[lane] = 0.f; // static B
        for (int kk = 0; kk < 3; ++kk)
        {
            const Vec3 o0 = randUnitV(rng), o1 = randUnitV(rng);
            h.angOrthoAxis0_recipResponseW[kk][lane] = Float4{ o0.x, o0.y, o0.z, unit(rng) * 0.5f };
            h.angOrthoAxis1_ErrorW[kk][lane] = Float4{ o1.x, o1.y, o1.z, sym(rng) * 0.02f };
        }
        for (uint32_t i = 0; i < rowsPerBatch; ++i)
        {
            TGSJointRowCon& c = jrows[i];
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
            if (unit(rng) < 0.4f) fl |= DY_SC_FLAG_SPRING;
            if (unit(rng) < 0.4f) fl |= DY_SC_FLAG_ORTHO_TARGET;
            if (unit(rng) < 0.3f) fl |= DY_SC_FLAG_INEQUALITY;
            c.flags[lane] = fl;
            c.appliedForce[lane] = sym(rng) * 0.2f;
        }
    };

    for (uint32_t k = 1; k <= 3; ++k) // contact batches
    {
        BatchTGS& b = batches[k];
        b.descStride = 2;
        b.constraintType = 1; // eCONTACT
        b.scalars[1] = k - 1;                        // mConstraintBatchIndex -> chs/fhs index
        b.scalars[2] = (k - 1) * rowsPerBatch;       // startConstraintIndex -> cps
        b.scalars[3] = (k - 1) * (rowsPerBatch + 1); // startFrictionIndex   -> frs
        b.scalars[6] = 0x3u;                         // mask: lanes 0,1 active
        fillContactLane(k - 1, 0);
        fillContactLane(k - 1, 1);
    }
    {
        BatchTGS& b = batches[4]; // joint batch
        b.descStride = 2;
        b.constraintType = 0; // eCONSTRAINT_1D
        b.scalars[1] = 0;     // jhs index
        b.scalars[2] = 0;     // jrows start
        b.scalars[6] = 0x3u;  // mask
        fillJointLane(0);
        fillJointLane(1);
    }

    std::vector<uint32_t> contactCounts = { 3, 3 };
    std::vector<uint32_t> jointCounts = { 1, 1 };
    std::vector<uint32_t> contactStarts = { contactStart, contactStart };
    std::vector<uint32_t> jointStarts = { jointStart, jointStart };

    // ---- GPU (engine ABI through the shim) ----
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "solverTGS.spv", "solveStaticBlockTGS", 40u, 0 },
        { "solverTGS.spv", "propagateStaticSolverBodyVelocitiesTGS", 40u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 2, &solverTgsAbiLoader);

    const uint32_t bodyOutCount = nbStaticSlabs * T * 2 + T; // slabIdx*T*2 + bodyIndex + T max
    std::vector<Float4> bodyOutInit(bodyOutCount, Float4{ 9.f, 9.f, 9.f, 9.f }); // sentinel

    PxgDevicePtr dVel = 0, dBodyOut = 0, dTx = 0, dIsl = 0, dCC = 0, dJC = 0, dCS = 0, dJS = 0;
    PxgDevicePtr dBatch = 0, dCh = 0, dFh = 0, dCp = 0, dFr = 0, dJh = 0, dJr = 0, dDesc = 0, dShared = 0;
    shim.memAlloc(&dVel, velPool.size() * sizeof(Float4));
    shim.memAlloc(&dBodyOut, bodyOutInit.size() * sizeof(Float4));
    shim.memAlloc(&dTx, txi.size() * sizeof(SolverTxIData));
    shim.memAlloc(&dCC, contactCounts.size() * 4);
    shim.memAlloc(&dJC, jointCounts.size() * 4);
    shim.memAlloc(&dCS, contactStarts.size() * 4);
    shim.memAlloc(&dJS, jointStarts.size() * 4);
    shim.memAlloc(&dBatch, batches.size() * sizeof(BatchTGS));
    shim.memAlloc(&dCh, chs.size() * sizeof(TGSContactHeader));
    shim.memAlloc(&dFh, fhs.size() * sizeof(TGSFrictionHeader));
    shim.memAlloc(&dCp, cps.size() * sizeof(TGSContactPoint));
    shim.memAlloc(&dFr, frs.size() * sizeof(TGSFriction));
    shim.memAlloc(&dJh, jhs.size() * sizeof(TGSJointHeader));
    shim.memAlloc(&dJr, jrows.size() * sizeof(TGSJointRowCon));
    if (!dVel || !dBodyOut || !dTx || !dCC || !dJC || !dCS || !dJS || !dBatch || !dCh || !dFh || !dCp ||
        !dFr || !dJh || !dJr)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    shim.memcpyHtoD(dVel, velPool.data(), velPool.size() * sizeof(Float4));
    shim.memcpyHtoD(dBodyOut, bodyOutInit.data(), bodyOutInit.size() * sizeof(Float4));
    shim.memcpyHtoD(dTx, txi.data(), txi.size() * sizeof(SolverTxIData));
    shim.memcpyHtoD(dCC, contactCounts.data(), contactCounts.size() * 4);
    shim.memcpyHtoD(dJC, jointCounts.data(), jointCounts.size() * 4);
    shim.memcpyHtoD(dCS, contactStarts.data(), contactStarts.size() * 4);
    shim.memcpyHtoD(dJS, jointStarts.data(), jointStarts.size() * 4);
    shim.memcpyHtoD(dBatch, batches.data(), batches.size() * sizeof(BatchTGS));
    shim.memcpyHtoD(dCh, chs.data(), chs.size() * sizeof(TGSContactHeader));
    shim.memcpyHtoD(dFh, fhs.data(), fhs.size() * sizeof(TGSFrictionHeader));
    shim.memcpyHtoD(dCp, cps.data(), cps.size() * sizeof(TGSContactPoint));
    shim.memcpyHtoD(dFr, frs.data(), frs.size() * sizeof(TGSFriction));
    shim.memcpyHtoD(dJh, jhs.data(), jhs.size() * sizeof(TGSJointHeader));
    shim.memcpyHtoD(dJr, jrows.data(), jrows.size() * sizeof(TGSJointRowCon));

    physx::PxgIslandContext islands[2];
    std::memset(islands, 0, sizeof(islands));
    islands[0].mBodyStartIndex = 77u; // decoy
    islands[0].mBodyCount = 5u;
    islands[1].mBodyStartIndex = bodyOffset;
    islands[1].mBodyCount = numDynamicBodies;
    shim.memAlloc(&dIsl, sizeof(islands));
    shim.memcpyHtoD(dIsl, islands, sizeof(islands));

    physx::PxgSolverCoreDesc desc;
    std::memset(&desc, 0, sizeof(desc));
    desc.islandContextPool = reinterpret_cast<physx::PxgIslandContext*>(dIsl);
    desc.solverBodyTxIDataPool = reinterpret_cast<physx::PxgSolverTxIData*>(dTx);
    desc.accumulatedBodyDeltaVOffset = deltaVOffset;
    desc.mRigidStaticContactCounts = reinterpret_cast<physx::PxU32*>(dCC);
    desc.mRigidStaticJointCounts = reinterpret_cast<physx::PxU32*>(dJC);
    desc.mRigidStaticContactStartIndices = reinterpret_cast<physx::PxU32*>(dCS);
    desc.mRigidStaticJointStartIndices = reinterpret_cast<physx::PxU32*>(dJS);
    desc.contactErrorAccumulator.mErrorSumOfSquares = 0.5f;
    desc.contactErrorAccumulator.mCounter = 0; // >= 0: residual accumulation ON
    desc.contactErrorAccumulator.mMaxError = 0.0625f;

    physx::PxgSolverSharedDesc<physx::IterativeSolveDataTGS> shared;
    std::memset(&shared, 0, sizeof(shared));
    shared.iterativeData.solverBodyVelPool = reinterpret_cast<float4*>(dVel);
    shared.iterativeData.tempStaticBodyOutputs = reinterpret_cast<float4*>(dBodyOut);
    shared.iterativeData.blockConstraintBatch = reinterpret_cast<physx::PxgBlockConstraintBatch*>(dBatch);
    shared.iterativeData.blockContactHeaders = reinterpret_cast<physx::PxgTGSBlockSolverContactHeader*>(dCh);
    shared.iterativeData.blockFrictionHeaders = reinterpret_cast<physx::PxgTGSBlockSolverFrictionHeader*>(dFh);
    shared.iterativeData.blockContactPoints = reinterpret_cast<physx::PxgTGSBlockSolverContactPoint*>(dCp);
    shared.iterativeData.blockFrictions = reinterpret_cast<physx::PxgTGSBlockSolverContactFriction*>(dFr);
    shared.iterativeData.blockJointConstraintHeaders =
        reinterpret_cast<physx::PxgTGSBlockSolverConstraint1DHeader*>(dJh);
    shared.iterativeData.blockJointConstraintRowsCon =
        reinterpret_cast<physx::PxgTGSBlockSolverConstraint1DCon*>(dJr);

    shim.memAlloc(&dDesc, sizeof(desc));
    shim.memcpyHtoD(dDesc, &desc, sizeof(desc));
    shim.memAlloc(&dShared, sizeof(shared));
    shim.memcpyHtoD(dShared, &shared, sizeof(shared));

    auto makeBlob = [&]() {
        EngineBlob blob;
        blob.add<uint64_t>(dDesc);
        blob.add<uint64_t>(dShared);
        blob.add<uint32_t>(1u);                   // islandIndex
        blob.add<uint32_t>(nbStaticSlabs);
        blob.add<uint32_t>(maxStaticPartitions);
        blob.add<float>(elapsedTime);
        blob.add<float>(minPen);
        blob.finish();
        return blob;
    };
    EngineBlob solveBlob = makeBlob();
    PXG_CHECK_MSG(solveBlob.bytes.size() == 40, "static-solve blob is 40 bytes (got %zu)",
                  solveBlob.bytes.size());

    const uint32_t nbBlocksRequired = ((numDynamicBodies + 63) / 64) * nbStaticSlabs;
    PXG_CHECK_MSG(shim.launchKernel(0, nbBlocksRequired, 1, 1, solveBlob.bytes.data(),
                                    uint32_t(solveBlob.bytes.size())) == PxgResult::Success,
                  "launch solveStaticBlockTGS");
    shim.streamSynchronize();

    EngineBlob propBlob = makeBlob();
    PXG_CHECK_MSG(shim.launchKernel(1, nbBlocksRequired, 1, 1, propBlob.bytes.data(),
                                    uint32_t(propBlob.bytes.size())) == PxgResult::Success,
                  "launch propagateStaticSolverBodyVelocitiesTGS");
    shim.streamSynchronize();

    // ---- reference ----
    SolverTxIData idTx;
    std::memset(&idTx, 0, sizeof(idTx));
    idTx.deltaBody2World.q = Quat{ 0.f, 0.f, 0.f, 1.f };

    RefErrAcc errAcc; // single accumulator: final = sum over all threads (order-independent under tol)
    std::vector<Float4> refBodyOut = bodyOutInit;

    auto solveOneBodySlab = [&](uint32_t bodyIndex, uint32_t slabIdx) {
        const uint32_t idx = bodyIndex; // lane 0 or 1 via warpScanExclusive(0b11, tIW=bodyIndex)
        const uint32_t startIndex = slabIdx * maxStaticPartitions;
        const uint32_t rawC = contactCounts[bodyIndex], rawJ = jointCounts[bodyIndex];
        const uint32_t contactCount = std::min(maxStaticPartitions, std::max(startIndex, rawC) - startIndex);
        const uint32_t jointCount = std::min(maxStaticPartitions, std::max(startIndex, rawJ) - startIndex);
        if (contactCount == 0 && jointCount == 0)
            return;

        const uint32_t inputBody = deltaVOffset + bodyIndex + bodyOffset;
        const Float4 v0 = velPool[inputBody], v1 = velPool[inputBody + T], v2 = velPool[inputBody + 2 * T];
        BodyStateTGS s;
        s.lv0 = Vec3(v0.x, v0.y, v0.z);
        s.av0 = Vec3(v0.w, v1.x, v1.y);
        s.ld0 = Vec3(v1.z, v1.w, v2.x);
        s.ad0 = Vec3(v2.y, v2.z, v2.w);
        s.lv1 = s.av1 = s.ld1 = s.ad1 = Vec3(0.f);

        const uint32_t startContactIndex = contactStarts[bodyIndex] + startIndex;
        const uint32_t startJointIndex = jointStarts[bodyIndex] + startIndex;
        const SolverTxIData iData0 = jointCount ? txi[bodyIndex + bodyOffset] : idTx;

        for (uint32_t i = 0; i < jointCount; ++i)
            refSolve1DTGS(batches[startJointIndex + i], s, idx, jhs, jrows, iData0, idTx, elapsedTime, true,
                          1.f, 1.f);
        for (uint32_t i = 0; i < contactCount; ++i)
            refSolveContactTGS(batches[startContactIndex + i], s, idx, chs, fhs, cps, frs, elapsedTime,
                               minPen, 1.f, 1.f, &errAcc);

        const uint32_t outputBody = slabIdx * T * 2;
        refBodyOut[bodyIndex + outputBody] = Float4{ s.lv0.x, s.lv0.y, s.lv0.z, s.av0.x };
        refBodyOut[bodyIndex + outputBody + T] = Float4{ s.av0.y, s.av0.z, v1.z, v1.w };
    };

    for (uint32_t slab = 0; slab < nbStaticSlabs; ++slab)
        for (uint32_t body = 0; body < numDynamicBodies; ++body)
            solveOneBodySlab(body, slab);

    // propagate reference
    std::vector<Float4> refVel = velPool;
    for (uint32_t body = 0; body < numDynamicBodies; ++body)
    {
        const uint32_t maxOutputs = std::max(contactCounts[body], jointCounts[body]);
        Float4 acc0{ 0, 0, 0, 0 }, acc1{ 0, 0, 0, 0 };
        float scale = 0.f;
        uint32_t index = body;
        for (uint32_t i = 0; i < maxOutputs; i += maxStaticPartitions)
        {
            const Float4 a = refBodyOut[index], b = refBodyOut[index + T];
            acc0 = Float4{ acc0.x + a.x, acc0.y + a.y, acc0.z + a.z, acc0.w + a.w };
            acc1 = Float4{ acc1.x + b.x, acc1.y + b.y, acc1.z + b.z, acc1.w + b.w };
            index += T * 2;
            scale += 1.f;
        }
        const float inv = 1.f / scale;
        const uint32_t outputBody = deltaVOffset + body + bodyOffset;
        refVel[outputBody] = Float4{ acc0.x * inv, acc0.y * inv, acc0.z * inv, acc0.w * inv };
        refVel[outputBody + T] = Float4{ acc1.x * inv, acc1.y * inv, acc1.z * inv, acc1.w * inv };
    }

    // ---- compare ----
    std::vector<Float4> gpuBodyOut(bodyOutCount);
    shim.memcpyDtoH(gpuBodyOut.data(), dBodyOut, gpuBodyOut.size() * sizeof(Float4));
    std::vector<Float4> gpuVel(poolSize);
    shim.memcpyDtoH(gpuVel.data(), dVel, gpuVel.size() * sizeof(Float4));
    std::vector<TGSContactPoint> gpuCps(cps.size());
    shim.memcpyDtoH(gpuCps.data(), dCp, gpuCps.size() * sizeof(TGSContactPoint));
    std::vector<TGSFriction> gpuFrs(frs.size());
    shim.memcpyDtoH(gpuFrs.data(), dFr, gpuFrs.size() * sizeof(TGSFriction));
    std::vector<TGSJointRowCon> gpuJrows(jrows.size());
    shim.memcpyDtoH(gpuJrows.data(), dJr, gpuJrows.size() * sizeof(TGSJointRowCon));

    size_t mismatches = 0;
    bool anyWritten = false;
    for (size_t i = 0; i < bodyOutCount; ++i)
    {
        // a slab output that was written differs from the 9.f sentinel
        if (gpuBodyOut[i].x != 9.f || gpuBodyOut[i].y != 9.f || gpuBodyOut[i].z != 9.f)
            anyWritten = true;
        const float* g = &gpuBodyOut[i].x;
        const float* r = &refBodyOut[i].x;
        for (int c = 0; c < 4; ++c)
            if (!almostEqual(g[c], r[c], 2e-4f, 2e-4f))
            {
                if (mismatches < 5)
                    PXG_CHECK_MSG(false, "tempStaticBodyOutputs[%zu].%d: gpu=%g ref=%g", i, c, g[c], r[c]);
                ++mismatches;
            }
    }
    PXG_CHECK_MSG(anyWritten, "solveStaticBlockTGS wrote no outputs (no-op / unbound launch?)");
    PXG_CHECK_MSG(mismatches == 0, "static solve: %zu tempStaticBodyOutputs mismatches", mismatches);

    mismatches = 0;
    for (uint32_t body = 0; body < numDynamicBodies; ++body)
        for (uint32_t half = 0; half < 2; ++half)
        {
            const uint32_t i = deltaVOffset + body + bodyOffset + half * T;
            const float* g = &gpuVel[i].x;
            const float* r = &refVel[i].x;
            for (int c = 0; c < 4; ++c)
                if (!almostEqual(g[c], r[c], 2e-4f, 2e-4f))
                {
                    if (mismatches < 5)
                        PXG_CHECK_MSG(false, "propagate deltaV[%u].%d: gpu=%g ref=%g", i, c, g[c], r[c]);
                    ++mismatches;
                }
        }
    PXG_CHECK_MSG(mismatches == 0, "propagate: %zu deltaV mismatches", mismatches);

    mismatches = 0;
    for (size_t i = 0; i < cps.size(); ++i)
        for (int l = 0; l < 2; ++l)
            if (!almostEqual(gpuCps[i].appliedForce[l], cps[i].appliedForce[l], 2e-4f, 2e-4f))
                ++mismatches;
    for (size_t i = 0; i < frs.size(); ++i)
        for (int l = 0; l < 2; ++l)
            if (!almostEqual(gpuFrs[i].appliedForce[l], frs[i].appliedForce[l], 2e-4f, 2e-4f))
                ++mismatches;
    for (size_t i = 0; i < jrows.size(); ++i)
        for (int l = 0; l < 2; ++l)
        {
            if (!almostEqual(gpuJrows[i].appliedForce[l], jrows[i].appliedForce[l], 2e-4f, 2e-4f))
                ++mismatches;
            if (!almostEqual(gpuJrows[i].residual[l], jrows[i].residual[l], 1e-3f, 1e-3f))
                ++mismatches;
        }
    PXG_CHECK_MSG(mismatches == 0, "static solve: %zu applied-force/residual mismatches", mismatches);

    physx::PxgSolverCoreDesc gpuDesc;
    shim.memcpyDtoH(&gpuDesc, dDesc, sizeof(gpuDesc));
    const float expectSum = 0.5f + errAcc.sumSq;
    const float expectMax = std::fmax(0.0625f, errAcc.maxErr);
    PXG_CHECK_MSG(gpuDesc.contactErrorAccumulator.mCounter == int32_t(errAcc.counter),
                  "residual counter: gpu=%d ref=%u", gpuDesc.contactErrorAccumulator.mCounter,
                  errAcc.counter);
    PXG_CHECK_MSG(errAcc.counter > 0, "fixture sanity: contact residual path exercised");
    PXG_CHECK_MSG(almostEqual(gpuDesc.contactErrorAccumulator.mErrorSumOfSquares, expectSum, 2e-4f, 2e-4f),
                  "residual sum-of-squares: gpu=%g ref=%g",
                  gpuDesc.contactErrorAccumulator.mErrorSumOfSquares, expectSum);
    PXG_CHECK_MSG(almostEqual(gpuDesc.contactErrorAccumulator.mMaxError, expectMax, 2e-4f, 2e-4f),
                  "residual max: gpu=%g ref=%g", gpuDesc.contactErrorAccumulator.mMaxError, expectMax);

    for (PxgDevicePtr pfree : { dVel, dBodyOut, dTx, dIsl, dCC, dJC, dCS, dJS, dBatch, dCh, dFh, dCp, dFr,
                                dJh, dJr, dDesc, dShared })
        shim.memFree(pfree);
}

// ---------------------------------------------------------------------------
// concludeBlocksTGS (ENGINE ABI)
// ---------------------------------------------------------------------------
// Derives the batch range from the island context and runs conclude1DBlockTGS
// on joint batches (eCONSTRAINT_1D=0) while no-op'ing contact batches
// (eCONTACT=1), matching solverMultiBlockTGS.cu.  Verifies joint rows are
// repacked (biasScale/spring .w zeroed per flags) and a canary joint row
// referenced only by a contact batch is left untouched.
PXG_TEST(solverTGS_concludeBlocksAbi)
{
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    std::mt19937 rng(4100);
    std::uniform_real_distribution<float> unit(0.f, 1.f);
    std::uniform_real_distribution<float> sym(-2.f, 2.f);

    const uint32_t rowsPerBatch = 4;
    const uint32_t batchStartIndex = 1; // island.mBatchStartIndex; batch 0 is a decoy
    // batches 1..4: joint, contact, joint, contact
    const uint16_t types[5] = { 0, 0, 1, 0, 1 };
    const uint32_t numAllBatches = 5;
    const uint32_t numLiveBatches = 4; // endIndex = mBatchCount + startIndex

    // jrows: joint batches use [0,4) and [4,8); the canary at [8,12) is
    // referenced only by the contact batches and must stay pristine.
    std::vector<BatchTGS> batches(numAllBatches);
    std::vector<TGSJointHeader> jhs(numAllBatches);
    std::vector<TGSJointRowCon> jrows(numAllBatches * rowsPerBatch);
    std::memset(batches.data(), 0, batches.size() * sizeof(BatchTGS));
    std::memset(jhs.data(), 0, jhs.size() * sizeof(TGSJointHeader));
    std::memset(jrows.data(), 0, jrows.size() * sizeof(TGSJointRowCon));

    for (uint32_t k = 1; k <= numLiveBatches; ++k)
    {
        BatchTGS& b = batches[k];
        b.descStride = uint16_t(1 + rng() % 32);
        b.constraintType = types[k];
        b.scalars[1] = k; // mConstraintBatchIndex -> jhs index
        b.scalars[2] = k * rowsPerBatch; // startConstraintIndex -> jrows
        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            const uint32_t rows = (lane < b.descStride) ? (1 + rng() % rowsPerBatch) : 0;
            jhs[k].rowCounts_breakable_orthoAxisCount[lane] = rows;
            for (uint32_t i = 0; i < rowsPerBatch; ++i)
            {
                TGSJointRowCon& c = jrows[b.scalars[2] + i];
                c.lin0XYZ_initBiasOrCoeff0[lane] = Float4{ sym(rng), sym(rng), sym(rng), sym(rng) };
                c.lin1XYZ_biasScaleOrCoeff1[lane] = Float4{ sym(rng), sym(rng), sym(rng), sym(rng) };
                c.ang0XYZ_velMultiplierOrCoeff2[lane] = Float4{ sym(rng), sym(rng), sym(rng), sym(rng) };
                c.ang1XYZ_velTargetOrCoeff3[lane] = Float4{ sym(rng), sym(rng), sym(rng), sym(rng) };
                c.flags[lane] = (unit(rng) < 0.5f ? DY_SC_FLAG_KEEP_BIAS : 0u) |
                                (unit(rng) < 0.5f ? DY_SC_FLAG_SPRING : 0u);
            }
        }
    }

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "solverTGS.spv", "concludeBlocksTGS", 24u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &solverTgsAbiLoader);

    PxgDevicePtr dBatch = 0, dJh = 0, dJr = 0, dIsl = 0, dDesc = 0, dShared = 0;
    shim.memAlloc(&dBatch, batches.size() * sizeof(BatchTGS));
    shim.memAlloc(&dJh, jhs.size() * sizeof(TGSJointHeader));
    shim.memAlloc(&dJr, jrows.size() * sizeof(TGSJointRowCon));
    if (!dBatch || !dJh || !dJr)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    shim.memcpyHtoD(dBatch, batches.data(), batches.size() * sizeof(BatchTGS));
    shim.memcpyHtoD(dJh, jhs.data(), jhs.size() * sizeof(TGSJointHeader));
    shim.memcpyHtoD(dJr, jrows.data(), jrows.size() * sizeof(TGSJointRowCon));

    physx::PxgIslandContext islands[2];
    std::memset(islands, 0, sizeof(islands));
    islands[0].mBodyCount = 3u; // decoy
    islands[1].mBatchStartIndex = batchStartIndex;
    islands[1].mBatchCount = numLiveBatches; // endIndex = numLiveBatches + startIndex
    shim.memAlloc(&dIsl, sizeof(islands));
    shim.memcpyHtoD(dIsl, islands, sizeof(islands));

    physx::PxgSolverCoreDesc desc;
    std::memset(&desc, 0, sizeof(desc));
    desc.islandContextPool = reinterpret_cast<physx::PxgIslandContext*>(dIsl);

    physx::PxgSolverSharedDesc<physx::IterativeSolveDataTGS> shared;
    std::memset(&shared, 0, sizeof(shared));
    shared.iterativeData.blockConstraintBatch = reinterpret_cast<physx::PxgBlockConstraintBatch*>(dBatch);
    shared.iterativeData.blockJointConstraintHeaders =
        reinterpret_cast<physx::PxgTGSBlockSolverConstraint1DHeader*>(dJh);
    shared.iterativeData.blockJointConstraintRowsCon =
        reinterpret_cast<physx::PxgTGSBlockSolverConstraint1DCon*>(dJr);

    shim.memAlloc(&dDesc, sizeof(desc));
    shim.memcpyHtoD(dDesc, &desc, sizeof(desc));
    shim.memAlloc(&dShared, sizeof(shared));
    shim.memcpyHtoD(dShared, &shared, sizeof(shared));

    EngineBlob blob;
    blob.add<uint64_t>(dDesc);
    blob.add<uint64_t>(dShared);
    blob.add<uint32_t>(1u); // islandIndex
    blob.finish();
    PXG_CHECK_MSG(blob.bytes.size() == 24, "conclude blob is 24 bytes (got %zu)", blob.bytes.size());

    const uint32_t gridX = (numLiveBatches + 7) / 8; // 8 warps per 256-thread block
    PXG_CHECK_MSG(shim.launchKernel(0, gridX, 1, 1, blob.bytes.data(), uint32_t(blob.bytes.size())) ==
                      PxgResult::Success,
                  "launch concludeBlocksTGS");
    shim.streamSynchronize();

    // ---- reference (conclude1DBlockTGS on joint batches only) ----
    auto refRows = jrows;
    for (uint32_t k = 1; k <= numLiveBatches; ++k)
    {
        const BatchTGS& b = batches[k];
        if (b.constraintType != 0)
            continue; // contact / articulation-contact: no-op
        for (uint32_t lane = 0; lane < b.descStride; ++lane)
        {
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
        }
    }

    std::vector<TGSJointRowCon> gpuRows(jrows.size());
    shim.memcpyDtoH(gpuRows.data(), dJr, gpuRows.size() * sizeof(TGSJointRowCon));

    size_t mismatches = 0;
    size_t mutated = 0;
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
            const Float4* o[] = { &jrows[i].lin0XYZ_initBiasOrCoeff0[l],
                                  &jrows[i].lin1XYZ_biasScaleOrCoeff1[l],
                                  &jrows[i].ang0XYZ_velMultiplierOrCoeff2[l],
                                  &jrows[i].ang1XYZ_velTargetOrCoeff3[l] };
            for (int c = 0; c < 4; ++c)
            {
                if (std::memcmp(g[c], r[c], sizeof(Float4)) != 0)
                    ++mismatches;
                if (std::memcmp(r[c], o[c], sizeof(Float4)) != 0)
                    ++mutated; // reference actually changed something here
            }
        }
    PXG_CHECK_MSG(mutated > 0, "concludeBlocksTGS reference mutated nothing (fixture too weak / no-op?)");
    PXG_CHECK_MSG(mismatches == 0, "conclude (engine ABI): %zu joint-row mismatches", mismatches);

    // canary: jrows[16..20) (batch index 4 * rowsPerBatch, referenced only by a
    // contact batch) must be byte-identical to the upload.
    for (uint32_t i = 4 * rowsPerBatch; i < 5 * rowsPerBatch; ++i)
        PXG_CHECK_MSG(std::memcmp(&gpuRows[i], &jrows[i], sizeof(TGSJointRowCon)) == 0,
                      "conclude touched a contact-only joint row (canary %u)", i);

    for (PxgDevicePtr pfree : { dBatch, dJh, dJr, dIsl, dDesc, dShared })
        shim.memFree(pfree);
}

// ---------------------------------------------------------------------------
// markActiveSlabTGS (ENGINE ABI, P5 Step 5 tranche 6 BDA) — per-partition
// active-slab marking for the mass-splitting reference counts.
// ---------------------------------------------------------------------------
// The read-only predicate that decides whether a contact still pushes.  Mirror
// of checkActiveContactBlockTGS() in solverBlockTGS.cuh (and of the Slang
// checkActiveContactBlockTGSBda): a stripped contact solve that (a) applies NO
// reference-count scaling — this is what COMPUTES the reference counts — (b)
// never mutates the input velocities between contact points, and (c) returns
// true on the first |deltaF| > 1e-8.  No friction.
bool refCheckActiveContactTGS(const BatchTGS& b, const BodyStateTGS& s, uint32_t lane,
                              std::vector<TGSContactHeader>& chs, std::vector<TGSContactPoint>& cps,
                              float elapsedTime, float minPen)
{
    const uint32_t bi = b.scalars[1]; // constraintBatchIndex
    const uint32_t sc = b.scalars[2]; // startConstraintIndex
    TGSContactHeader& ch = chs[bi];

    const uint32_t numNormal = ch.numNormalConstr[lane];
    if (!numNormal)
        return false;

    const float maxPenBias = ch.maxPenBias[lane];
    const Float4 nsf = ch.normal_staticFriction[lane];
    const Vec3 normal(nsf.x, nsf.y, nsf.z);
    const float restitutionXdt = ch.restitutionXdt[lane];
    const uint32_t flags = ch.flags[lane];

    const Vec3 relMotion = s.ld0 - s.ld1;
    const float deltaV = normal.dot(relMotion);
    const float relVel1 = (s.lv0 - s.lv1).dot(normal);

    for (uint32_t i = 0; i < numNormal; ++i)
    {
        const TGSContactPoint& c = cps[sc + i];
        const Float4 ra = c.raXn_extraCoeff[lane];
        const Float4 rb = c.rbXn_targetVelW[lane];
        const float appliedForce = c.appliedForce[lane];
        const float separation = c.separation[lane];
        const float maxImpulse = c.maxImpulse[lane];

        const float resp0 = c.resp0[lane], resp1 = c.resp1[lane]; // NO ref scaling
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

        if (std::fabs(deltaF) > 1.0e-8f)
            return true;
    }
    return false;
}

// The engine launches markActiveSlabTGS ONCE per island with the LAST partition
// index; it marks every batch in [mBatchStartIndex, cpp[lastPartition +
// mStartPartitionIndex]).  For each batch lane < descStride it decides
// isActiveSlab (joint -> always active; contact -> checkActiveContactBlockTGS)
// and atomicOr's (1 << (slabId % 32)) into solverEncodedReferenceCount[(slabId /
// 32) * numTotalBodies + bodyId] for bodyA and bodyB, only when that body index
// is a real dynamic body (>= bodyOffset).  Body velocities come from the
// accumulated deltaV staging (3 float4 per body at accumulatedBodyDeltaVOffset +
// bodyId + k*totalBodiesIncKinematics), NOT the k*192 per-partition pool of
// solveBlockUnified.  This fixture fills the REAL engine descs (+ a 2-element
// PxgIslandContext array; islandIndex = 1 pins the stride while element [0]
// carries decoy partition/batch fields but the SAME body fields — the .cu reads
// mBodyCount / mArticulationCount from islandContextPool[0]), one partition
// holding a mix of contact and joint batches (one contact deliberately has
// numNormalConstr == 0 so the inactive early-return is exercised), pre-zeroes
// the encoded buffer on device, launches through PxgCudaShim with the 40-byte
// blob, and compares the final encoded words BIT-EXACT (popcount-of-OR is
// order-independent) against refCheckActiveContactTGS.
PXG_TEST(solverTGS_markActiveSlab)
{
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    std::mt19937 rng(5106);
    std::uniform_real_distribution<float> unit(0.f, 1.f);
    std::uniform_real_distribution<float> sym(-2.f, 2.f);

    const uint32_t bodyOffset = 8;
    const uint32_t numDynamicBodies = 200;
    const uint32_t numTotalBodies = bodyOffset + numDynamicBodies; // no articulations
    const uint32_t totalBodiesIncKinematics = numDynamicBodies + bodyOffset;
    const uint32_t numSlabs = 40;                   // > 32 -> 2-word slabIndexOffset
    const uint32_t num32Slabs = (numSlabs + 31) / 32;
    const uint32_t rowsPerBatch = 4;
    const float elapsedTime = 0.005f;
    const float minPen = -0.02f;
    const uint32_t outputOffset = 0; // accumulatedBodyDeltaVOffset

    // batches: [0] decoy (never marked), then a mix over one partition.
    // mBatchStartIndex = 1, mStartPartitionIndex = 1, lastPartition = 0 =>
    // endIndex = cpp[0 + 1] = cpp[1] = 5, so partition spans batches [1, 5).
    const uint32_t numAllBatches = 5;
    const uint32_t batchStartIndex = 1;
    const uint16_t types[5] = { 0, 1, 0, 1, 1 };     // [1]contact [2]joint [3]contact(empty) [4]contact
    const uint16_t strides[5] = { 0, 32, 27, 32, 20 }; // partial strides gate tail lanes
    std::vector<uint32_t> cpp = { 777u, 5u };        // index 0 decoy; endIndex = cpp[1] = 5

    std::vector<BatchTGS> batches(numAllBatches);
    std::vector<TGSContactHeader> chs(numAllBatches);
    std::vector<TGSContactPoint> cps(numAllBatches * rowsPerBatch);
    std::memset(batches.data(), 0, batches.size() * sizeof(BatchTGS));
    std::memset(chs.data(), 0, chs.size() * sizeof(TGSContactHeader));
    std::memset(cps.data(), 0, cps.size() * sizeof(TGSContactPoint));

    const uint32_t poolSize = 3 * totalBodiesIncKinematics; // 3 float4 stripes per body
    std::vector<Float4> velPool(poolSize);
    for (Float4& v : velPool)
        v = Float4{ sym(rng), sym(rng), sym(rng), sym(rng) };

    for (uint32_t k = batchStartIndex; k < numAllBatches; ++k)
    {
        BatchTGS& b = batches[k];
        b.descStride = strides[k];
        b.constraintType = types[k];
        b.scalars[1] = k;                // constraintBatchIndex
        b.scalars[2] = k * rowsPerBatch; // startConstraintIndex

        // batch [3] is a contact with NO normal constraints -> always inactive.
        const bool emptyContact = (k == 3);

        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            b.slabId[lane] = rng() % numSlabs;
            b.bodyAIndex[lane] = rng() % numTotalBodies; // some < bodyOffset (kinematic)
            b.bodyBIndex[lane] = rng() % numTotalBodies;

            if (b.constraintType == 1) // eCONTACT
            {
                TGSContactHeader& h = chs[k];
                h.invMass0_1_angDom0_1[lane] = Float4{ unit(rng), unit(rng), unit(rng), unit(rng) };
                const Vec3 n = randUnitV(rng);
                h.normal_staticFriction[lane] = Float4{ n.x, n.y, n.z, unit(rng) };
                h.maxPenBias[lane] = -unit(rng) * 5.f;
                h.flags[lane] = unit(rng) < 0.5f ? COMPLIANT_ACCEL_SPRING : 0;
                h.numNormalConstr[lane] = emptyContact ? 0u : (1u + rng() % rowsPerBatch);
                h.restitutionXdt[lane] = unit(rng) < 0.5f ? 0.f : -unit(rng);
                h.p8[lane] = 0.8f;

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
            }
            // joint batches need no header data: markActiveSlab marks joints
            // active unconditionally.
        }
    }

    // ---- reference: OR from a zero-initialised encoded buffer ----
    auto unpackBody = [&](uint32_t bodyId, Vec3& lv, Vec3& av, Vec3& ld, Vec3& ad) {
        const uint32_t fi = outputOffset + bodyId;
        const Float4 v0 = velPool[fi];
        const Float4 v1 = velPool[fi + totalBodiesIncKinematics];
        const Float4 v2 = velPool[fi + 2u * totalBodiesIncKinematics];
        lv = Vec3(v0.x, v0.y, v0.z);
        av = Vec3(v0.w, v1.x, v1.y);
        ld = Vec3(v1.z, v1.w, v2.x);
        ad = Vec3(v2.y, v2.z, v2.w);
    };

    const uint32_t encSize = num32Slabs * numTotalBodies;
    std::vector<uint32_t> refEnc(encSize, 0);
    uint32_t contactActive = 0, contactInactive = 0, jointMarks = 0;
    for (uint32_t k = batchStartIndex; k < cpp[1]; ++k)
    {
        const BatchTGS& b = batches[k];
        for (uint32_t lane = 0; lane < b.descStride; ++lane)
        {
            const uint32_t slabId = b.slabId[lane];
            const uint32_t bodyIdA = b.bodyAIndex[lane];
            const uint32_t bodyIdB = b.bodyBIndex[lane];
            const uint32_t slabIndexOffset = (slabId / 32u) * numTotalBodies;
            const uint32_t encodedSlabIndex = (1u << (slabId % 32u));

            bool active;
            if (b.constraintType != 1) // joint (eCONSTRAINT_1D) -> always active
            {
                active = true;
                ++jointMarks;
            }
            else
            {
                BodyStateTGS s;
                unpackBody(bodyIdA, s.lv0, s.av0, s.ld0, s.ad0);
                unpackBody(bodyIdB, s.lv1, s.av1, s.ld1, s.ad1);
                active = refCheckActiveContactTGS(b, s, lane, chs, cps, elapsedTime, minPen);
                if (active)
                    ++contactActive;
                else
                    ++contactInactive;
            }

            if (active && bodyIdA >= bodyOffset)
                refEnc[slabIndexOffset + bodyIdA] |= encodedSlabIndex;
            if (active && bodyIdB >= bodyOffset)
                refEnc[slabIndexOffset + bodyIdB] |= encodedSlabIndex;
        }
    }

    // fixture sanity: every code path was genuinely exercised.
    PXG_CHECK_MSG(contactActive > 0, "fixture: at least one contact marks its slab active");
    PXG_CHECK_MSG(contactInactive > 0, "fixture: at least one contact leaves its slab inactive");
    PXG_CHECK_MSG(jointMarks > 0, "fixture: at least one joint batch (unconditionally active)");
    uint32_t markedWords = 0;
    for (uint32_t e : refEnc)
        if (e)
            ++markedWords;
    PXG_CHECK_MSG(markedWords > 0 && markedWords < encSize,
                  "fixture: encoded buffer is a mix of marked/unmarked (%u/%u)", markedWords, encSize);

    // ---- GPU (engine ABI through the shim) ----
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "solverTGS.spv", "markActiveSlabTGS", 40u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &solverTgsAbiLoader);

    PxgDevicePtr dVel = 0, dEnc = 0, dCpp = 0, dIsl = 0, dBatch = 0, dCh = 0, dCp = 0;
    PxgDevicePtr dDesc = 0, dShared = 0;
    shim.memAlloc(&dVel, velPool.size() * sizeof(Float4));
    shim.memAlloc(&dEnc, encSize * 4);
    shim.memAlloc(&dCpp, cpp.size() * 4);
    shim.memAlloc(&dBatch, batches.size() * sizeof(BatchTGS));
    shim.memAlloc(&dCh, chs.size() * sizeof(TGSContactHeader));
    shim.memAlloc(&dCp, cps.size() * sizeof(TGSContactPoint));
    if (!dVel || !dEnc || !dCpp || !dBatch || !dCh || !dCp)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    std::vector<uint32_t> encInit(encSize, 0); // engine memsets this to 0 first
    shim.memcpyHtoD(dVel, velPool.data(), velPool.size() * sizeof(Float4));
    shim.memcpyHtoD(dEnc, encInit.data(), encSize * 4);
    shim.memcpyHtoD(dCpp, cpp.data(), cpp.size() * 4);
    shim.memcpyHtoD(dBatch, batches.data(), batches.size() * sizeof(BatchTGS));
    shim.memcpyHtoD(dCh, chs.data(), chs.size() * sizeof(TGSContactHeader));
    shim.memcpyHtoD(dCp, cps.data(), cps.size() * sizeof(TGSContactPoint));

    physx::PxgIslandContext islands[2];
    std::memset(islands, 0, sizeof(islands));
    // element [0]: SAME body fields (the .cu reads mBodyCount / mArticulationCount
    // from islandContextPool[0]) but DECOY partition/batch fields — a bug that
    // read them via islandIndex would corrupt the batch range and mismatch.
    islands[0].mBodyStartIndex = bodyOffset;
    islands[0].mBodyCount = numDynamicBodies;
    islands[0].mArticulationCount = 0;
    islands[0].mStartPartitionIndex = 5u;  // decoy
    islands[0].mNumPartitions = 9u;        // decoy
    islands[0].mBatchStartIndex = 42u;     // decoy
    // element [1] = islandIndex: the real island-indexed fields.
    islands[1].mBodyStartIndex = bodyOffset;
    islands[1].mBodyCount = 999u;          // decoy: kernel reads mBodyCount from [0]
    islands[1].mArticulationCount = 7u;    // decoy: kernel reads it from [0]
    islands[1].mStartPartitionIndex = 1;
    islands[1].mNumPartitions = 1;
    islands[1].mBatchStartIndex = batchStartIndex;
    shim.memAlloc(&dIsl, sizeof(islands));
    shim.memcpyHtoD(dIsl, islands, sizeof(islands));

    physx::PxgSolverCoreDesc desc;
    std::memset(&desc, 0, sizeof(desc));
    desc.islandContextPool = reinterpret_cast<physx::PxgIslandContext*>(dIsl);
    desc.constraintsPerPartition = reinterpret_cast<physx::PxU32*>(dCpp);
    desc.numSlabs = numSlabs;
    desc.accumulatedBodyDeltaVOffset = outputOffset;

    physx::PxgSolverSharedDesc<physx::IterativeSolveDataTGS> shared;
    std::memset(&shared, 0, sizeof(shared));
    shared.iterativeData.solverBodyVelPool = reinterpret_cast<float4*>(dVel);
    shared.iterativeData.solverEncodedReferenceCount = reinterpret_cast<physx::PxU32*>(dEnc);
    shared.iterativeData.blockConstraintBatch = reinterpret_cast<physx::PxgBlockConstraintBatch*>(dBatch);
    shared.iterativeData.blockContactHeaders = reinterpret_cast<physx::PxgTGSBlockSolverContactHeader*>(dCh);
    shared.iterativeData.blockContactPoints = reinterpret_cast<physx::PxgTGSBlockSolverContactPoint*>(dCp);

    shim.memAlloc(&dDesc, sizeof(desc));
    shim.memcpyHtoD(dDesc, &desc, sizeof(desc));
    shim.memAlloc(&dShared, sizeof(shared));
    shim.memcpyHtoD(dShared, &shared, sizeof(shared));

    EngineBlob blob;
    blob.add<uint64_t>(dDesc);     // @0  solverCoreDesc
    blob.add<uint64_t>(dShared);   // @8  sharedDesc
    blob.add<uint32_t>(1u);        // @16 islandIndex
    blob.add<uint32_t>(0u);        // @20 lastPartition
    blob.add<float>(minPen);       // @24 (note: minPen precedes elapsedTime)
    blob.add<float>(elapsedTime);  // @28
    blob.add<uint64_t>(0u);        // @32 artiDesc (deferred; unused on the rigid path)
    blob.finish();
    PXG_CHECK_MSG(blob.bytes.size() == 40, "markActiveSlabTGS blob is 40 bytes (got %zu)", blob.bytes.size());

    // grid (maxBlocks, 1, 1); block (32, 2, 1) -> 2 warps/block.  maxBlocks =
    // ceil(batchesInPartition / numWarpsPerBlock) = ceil(4/2) = 2.
    PXG_CHECK_MSG(shim.launchKernel(0, 2, 1, 1, blob.bytes.data(), uint32_t(blob.bytes.size())) ==
                      PxgResult::Success,
                  "launch markActiveSlabTGS");
    shim.streamSynchronize();

    // ---- compare ----
    std::vector<uint32_t> gpuEnc(encSize);
    shim.memcpyDtoH(gpuEnc.data(), dEnc, encSize * 4);

    size_t mismatches = 0;
    for (uint32_t i = 0; i < encSize; ++i)
        if (gpuEnc[i] != refEnc[i])
        {
            if (mismatches < 5)
                PXG_CHECK_MSG(false, "encoded[%u]: gpu=0x%08x ref=0x%08x", i, gpuEnc[i], refEnc[i]);
            ++mismatches;
        }
    PXG_CHECK_MSG(mismatches == 0, "markActiveSlabTGS: %zu encoded-word mismatches", mismatches);

    for (PxgDevicePtr pfree : { dVel, dEnc, dCpp, dBatch, dCh, dCp, dIsl, dDesc, dShared })
        shim.memFree(pfree);
}

} // namespace
