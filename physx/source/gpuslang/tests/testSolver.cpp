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

// Verification of solverPGS.slang (solveBlockPartition: contact + 1D joint
// constraint solve) against a C++ transcription of solverBlock.cuh /
// constraintPrepShared.cuh / DyCpuGpu1dConstraint.h.

#include "pxgTestFramework.h"
#include "pxgGpuStructs.h"
#include "pxgRefMath.h"

#include <numeric>
#include <random>

namespace
{

using namespace pxgtest;

const float DY_MIN_RESPONSE = 1e-12f;
const uint32_t DY_SC_FLAG_SPRING = 1u << 5;
const uint32_t DY_SC_FLAG_ACCELERATION_SPRING = 1u << 7;
const uint32_t COMPLIANT_ACCEL_SPRING = 1u << 3;

// ---- block SoA mirrors (sizes pinned to the CUDA layouts) ----

struct BlockConstraintBatch
{
    uint16_t descStride;
    uint16_t constraintType; // 0 = contact, 1 = joint
    uint32_t blockContactIndex;
    uint32_t constraintBatchIndex;
    uint32_t startConstraintIndex;
    uint32_t startFrictionIndex;
    uint32_t startPartitionIndex;
    uint32_t articulationResponseIndex;
    uint32_t mask;
    uint32_t pad0[24];
    uint64_t bodyANodeIndex[32];
    uint64_t bodyBNodeIndex[32];
    uint32_t bodyAIndex[32];
    uint32_t bodyBIndex[32];
    uint32_t remappedBodyAIndex[32];
    uint32_t remappedBodyBIndex[32];
    uint32_t slabId[32];
    uint64_t shapeInteraction[32];
};
static_assert(sizeof(BlockConstraintBatch) == 1536, "PxgBlockConstraintBatch mirror");

struct BlockContactHeader
{
    Float4 invMass0_1_angDom0_1[32];
    Float4 normal_staticFriction[32];
    float accumNormalForce[32];
    float minNormalForce[32];
    uint32_t flags[32];
    uint32_t numNormalConstr[32];
    uint32_t forceWritebackOffset[32];
    float restitution[32];
    float cfm[32];
};
static_assert(sizeof(BlockContactHeader) == 1920, "PxgBlockSolverContactHeader mirror");

struct BlockFrictionHeader
{
    Float4 frictionNormals0[32];
    Float4 frictionNormals1[32];
    uint32_t numFrictionConstr[32];
    float dynamicFriction[32];
    uint32_t broken[32];
};
static_assert(sizeof(BlockFrictionHeader) == 1408, "PxgBlockSolverFrictionHeader mirror");

struct BlockContactPoint
{
    Float4 raXn_targetVelocity[32];
    Float4 rbXn_maxImpulse[32];
    float appliedForce[32];
    float resp0[32];
    float resp1[32];
    float coeff0[32];
    float coeff1[32];
};
static_assert(sizeof(BlockContactPoint) == 1664, "PxgBlockSolverContactPoint mirror");

struct BlockContactFriction
{
    Float4 raXn_bias[32];
    Float4 rbXn_targetVelW[32];
    float appliedForce[32];
    float resp0[32];
    float resp1[32];
};
static_assert(sizeof(BlockContactFriction) == 1408, "PxgBlockSolverContactFriction mirror");

struct BlockJointHeader
{
    uint16_t rowCounts[32];
    uint16_t pad0[32];
    uint16_t breakable[32];
    uint16_t pad1[32];
    float angBreakImpulse[32];
    Float4 body0WorldOffset_linBreakImpulse[32];
    float invMass0D0[32];
    float invMass1D1[32];
    float invInertiaScale0[32];
    float invInertiaScale1[32];
    uint32_t writeBackOffset[32];
    float cfm[32];
};
static_assert(sizeof(BlockJointHeader) == 1664, "PxgBlockSolverConstraint1DHeader mirror");

struct BlockJointRowCon
{
    Float4 lin0XYZ_minImpulse[32];
    Float4 lin1XYZ_maxImpulse[32];
    Float4 ang0XYZ_resp0[32];
    Float4 ang1XYZ_resp1[32];
    float initJointSpeed[32];
    uint32_t pad0[32];
};
static_assert(sizeof(BlockJointRowCon) == 2304, "PxgBlockSolverConstraint1DCon mirror");

struct BlockJointRowMod
{
    float ang0Writeback[96];
    float appliedForce[32];
    uint32_t flags[32];
    float residual[32];
    float coeff0[32];
    float coeff1[32];
};
static_assert(sizeof(BlockJointRowMod) == 1024, "PxgBlockSolverConstraint1DMod mirror");

struct SolverParams
{
    uint32_t startIndex;
    uint32_t endIndex;
    uint32_t doFriction;
    uint32_t residualReportingEnabled;
    uint32_t numSlabs;
    uint32_t numTotalBodies;
    uint32_t bodyOffset;
};

// ---------------------------------------------------------------------------
// CPU reference (transcriptions)
// ---------------------------------------------------------------------------

void refContactCoefficients(uint32_t flags, float restitution, float unitResponse, float recipResponse,
                            float targetVelocity, float coeff0, float coeff1, float& velMultiplier,
                            float& impulseMultiplier, float& biasedErr)
{
    if (restitution < 0.f)
    {
        const bool accelSpring = (flags & COMPLIANT_ACCEL_SPRING) != 0;
        const float massIfAccelElseOne = accelSpring ? recipResponse : 1.0f;
        const float oneIfAccelElseR = accelSpring ? 1.0f : unitResponse;
        const float a = coeff0;
        const float b = coeff1 * massIfAccelElseOne;
        const float x = 1.f / (a * oneIfAccelElseR + 1.f);
        const float scaledBias = x * b;
        velMultiplier = x * a * massIfAccelElseOne;
        impulseMultiplier = 1.f - x;
        biasedErr = targetVelocity * velMultiplier - scaledBias;
    }
    else
    {
        velMultiplier = recipResponse;
        biasedErr = coeff0 * velMultiplier;
        impulseMultiplier = 1.f;
    }
}

void refSolveContact(const BlockConstraintBatch& batch, Vec3& lv0, Vec3& av0, Vec3& lv1, Vec3& av1,
                     bool doFriction, uint32_t lane, std::vector<BlockContactHeader>& contactHeaders,
                     std::vector<BlockFrictionHeader>& frictionHeaders,
                     std::vector<BlockContactPoint>& contacts, std::vector<BlockContactFriction>& frictions,
                     float ref0, float ref1)
{
    BlockContactHeader& hdr = contactHeaders[batch.constraintBatchIndex];
    BlockFrictionHeader& fhdr = frictionHeaders[batch.constraintBatchIndex];

    const uint32_t numNormal = hdr.numNormalConstr[lane];
    const uint32_t numFriction = fhdr.numFrictionConstr[lane];

    float accumulatedNormalImpulse = 0.f;

    const Float4 im = hdr.invMass0_1_angDom0_1[lane];
    const float invMassA = ref0 * im.x, invMassB = ref1 * im.y;
    const float angDom0 = ref0 * im.z, angDom1 = ref1 * im.w;

    const Float4 nsf = hdr.normal_staticFriction[lane];
    const Vec3 normal(nsf.x, nsf.y, nsf.z);
    const float staticFrictionCof = nsf.w;
    const float restitution = hdr.restitution[lane];
    const uint32_t flags = hdr.flags[lane];

    const Vec3 delLinVel0 = normal * invMassA;
    const Vec3 delLinVel1 = normal * invMassB;

    for (uint32_t i = 0; i < numNormal; ++i)
    {
        BlockContactPoint& c = contacts[batch.startConstraintIndex + i];
        const Float4 ra = c.raXn_targetVelocity[lane];
        const Float4 rb = c.rbXn_maxImpulse[lane];
        const float appliedForce = c.appliedForce[lane];
        const Vec3 raXn(ra.x, ra.y, ra.z), rbXn(rb.x, rb.y, rb.z);

        const float unitResponse = ref0 * c.resp0[lane] + ref1 * c.resp1[lane];
        const float recipResponse = unitResponse > 0.f ? 1.f / unitResponse : 0.f;

        float velMultiplier, impulseMul, biasedErr;
        refContactCoefficients(flags, restitution, unitResponse, recipResponse, ra.w, c.coeff0[lane],
                               c.coeff1[lane], velMultiplier, impulseMul, biasedErr);

        const float v0 = lv0.dot(normal) + av0.dot(raXn);
        const float v1 = lv1.dot(normal) + av1.dot(rbXn);
        const float normalVel = v0 - v1;

        const float tempDeltaF = biasedErr - normalVel * velMultiplier;
        const float _deltaF = std::fmax(tempDeltaF, -appliedForce);
        const float _newForce = appliedForce * impulseMul + _deltaF;
        const float newForce = std::fmin(_newForce, rb.w);
        const float deltaF = newForce - appliedForce;

        lv0 += delLinVel0 * deltaF;
        lv1 = lv1 - delLinVel1 * deltaF;
        av0 += raXn * (deltaF * angDom0);
        av1 = av1 - rbXn * (deltaF * angDom1);

        c.appliedForce[lane] = newForce;
        accumulatedNormalImpulse += newForce;
    }

    if (numFriction && doFriction)
    {
        const float dynCof = fhdr.dynamicFriction[lane];
        const float maxFrictionImpulse = staticFrictionCof * accumulatedNormalImpulse;
        const float maxDynFrictionImpulse = dynCof * accumulatedNormalImpulse;
        uint32_t broken = 0;

        for (uint32_t i = 0; i < numFriction; ++i)
        {
            BlockContactFriction& f = frictions[batch.startFrictionIndex + i];
            const Float4 fn = (i & 1) ? fhdr.frictionNormals1[lane] : fhdr.frictionNormals0[lane];
            const Float4 ra = f.raXn_bias[lane];
            const Float4 rb = f.rbXn_targetVelW[lane];
            const float appliedForce = f.appliedForce[lane];

            const Vec3 fnormal(fn.x, fn.y, fn.z), raXn(ra.x, ra.y, ra.z), rbXn(rb.x, rb.y, rb.z);

            const float resp = ref0 * f.resp0[lane] + ref1 * f.resp1[lane];
            const float velMultiplier = resp > 0.f ? 0.8f / resp : 0.f;

            const Vec3 fdel0 = fnormal * invMassA, fdel1 = fnormal * invMassB;

            const float v0 = av0.dot(raXn) + lv0.dot(fnormal);
            const float v1 = av1.dot(rbXn) + lv1.dot(fnormal);
            const float normalVel = v0 - v1;

            const float tmp1 = appliedForce - (ra.w - rb.w) * velMultiplier;
            const float totalImpulse = tmp1 - normalVel * velMultiplier;
            const bool clamp = std::fabs(totalImpulse) > maxFrictionImpulse;
            const float totalClamped = std::fmin(maxDynFrictionImpulse, std::fmax(-maxDynFrictionImpulse, totalImpulse));
            const float newAppliedForce = clamp ? totalClamped : totalImpulse;
            const float deltaF = newAppliedForce - appliedForce;

            lv0 += fdel0 * deltaF;
            lv1 = lv1 - fdel1 * deltaF;
            av0 += raXn * (deltaF * angDom0);
            av1 = av1 - rbXn * (deltaF * angDom1);

            f.appliedForce[lane] = newAppliedForce;
            broken |= clamp ? 1u : 0u;
        }
        fhdr.broken[lane] = broken;
    }
}

void refSolve1D(const BlockConstraintBatch& batch, Vec3& lv0, Vec3& av0, Vec3& lv1, Vec3& av1,
                uint32_t lane, bool residual, std::vector<BlockJointHeader>& headers,
                std::vector<BlockJointRowCon>& rowsCon, std::vector<BlockJointRowMod>& rowsMod,
                float ref0, float ref1)
{
    BlockJointHeader& hdr = headers[batch.constraintBatchIndex];

    const float invMass0 = ref0 * hdr.invMass0D0[lane];
    const float invMass1 = ref1 * hdr.invMass1D1[lane];
    const float invInertiaScale0 = ref0 * hdr.invInertiaScale0[lane];
    const float invInertiaScale1 = ref1 * hdr.invInertiaScale1[lane];

    for (uint32_t i = 0; i < hdr.rowCounts[lane]; ++i)
    {
        BlockJointRowCon& con = rowsCon[batch.startConstraintIndex + i];
        BlockJointRowMod& mod = rowsMod[batch.startConstraintIndex + i];

        const Float4 l0 = con.lin0XYZ_minImpulse[lane];
        const Float4 l1 = con.lin1XYZ_maxImpulse[lane];
        const Float4 a0 = con.ang0XYZ_resp0[lane];
        const Float4 a1 = con.ang1XYZ_resp1[lane];

        const Vec3 clin0(l0.x, l0.y, l0.z), clin1(l1.x, l1.y, l1.z);
        const Vec3 cang0(a0.x, a0.y, a0.z), cang1(a1.x, a1.y, a1.z);

        const float unitResponse = ref0 * a0.w + ref1 * a1.w;
        const float recipResponse = unitResponse <= DY_MIN_RESPONSE ? 0.f : 1.f / unitResponse;

        const float coeff0 = mod.coeff0[lane], coeff1 = mod.coeff1[lane];
        const uint32_t flags = mod.flags[lane];
        const float initJointSpeed = con.initJointSpeed[lane];

        float constant, unbiasedConstant, vMul, iMul;
        const bool isSpring = (flags & DY_SC_FLAG_SPRING) != 0;
        const bool isAccel = (flags & DY_SC_FLAG_ACCELERATION_SPRING) != 0;
        if (isSpring)
        {
            const float a = coeff0, b = coeff1;
            if (isAccel)
            {
                const float x = 1.f / (1.f + a);
                constant = unbiasedConstant = x * recipResponse * b;
                vMul = -x * recipResponse * a;
                iMul = 1.f - x;
            }
            else
            {
                const float x = 1.f / (1.f + a * unitResponse);
                constant = unbiasedConstant = x * b;
                vMul = -x * a;
                iMul = 1.f - x;
            }
        }
        else
        {
            vMul = -recipResponse;
            iMul = 1.f;
            constant = coeff0 * recipResponse;
            unbiasedConstant = coeff1 * recipResponse;
        }
        const float velBias = initJointSpeed * vMul;
        constant += velBias;
        unbiasedConstant += velBias;

        const float appliedForce = mod.appliedForce[lane];
        const float v0 = lv0.dot(clin0) + av0.dot(cang0);
        const float v1 = lv1.dot(clin1) + av1.dot(cang1);
        const float normalVel = v0 - v1;

        const float unclampedForce = iMul * appliedForce + (vMul * normalVel + constant);
        const float clampedForce = std::fmin(l1.w, std::fmax(l0.w, unclampedForce));
        const float deltaF = clampedForce - appliedForce;

        mod.appliedForce[lane] = clampedForce;
        if (residual)
            mod.residual[lane] = vMul == 0.f ? 0.f : deltaF / vMul;

        lv0 += clin0 * (deltaF * invMass0);
        lv1 = lv1 - clin1 * (deltaF * invMass1);
        av0 += cang0 * deltaF * invInertiaScale0;
        av1 = av1 - cang1 * deltaF * invInertiaScale1;
    }
}

uint32_t refCountActiveSlabs(uint32_t index, uint32_t numSlabs, uint32_t numBodies,
                             const std::vector<uint32_t>& encoded)
{
    uint32_t count = 0;
    for (uint32_t i = 0; i < (numSlabs + 31) / 32; ++i)
    {
        uint32_t v = encoded[index + i * numBodies];
        while (v) { count += v & 1; v >>= 1; }
    }
    return std::max(1u, count);
}

// ---------------------------------------------------------------------------
// test
// ---------------------------------------------------------------------------

Vec3 randUnit(std::mt19937& rng)
{
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    Vec3 v(d(rng), d(rng), d(rng));
    return v.magnitude() > 1e-3f ? v.getNormalized() : Vec3(1, 0, 0);
}

PXG_TEST(solverPGS_contactAndJointBatches)
{
    std::mt19937 rng(3000);
    std::uniform_real_distribution<float> unit(0.f, 1.f);
    std::uniform_real_distribution<float> sym(-3.f, 3.f);

    const uint32_t numBatches = 24;
    const uint32_t maxRowsPerBatch = 4;
    const uint32_t numSlabs = 40; // > 32 exercises the multi-word reference counts
    const uint32_t numTotalBodies = 4096;
    const uint32_t bodyOffset = 16;

    std::vector<BlockConstraintBatch> batches(numBatches);
    std::memset(batches.data(), 0, batches.size() * sizeof(BlockConstraintBatch));
    std::vector<BlockContactHeader> contactHeaders(numBatches);
    std::vector<BlockFrictionHeader> frictionHeaders(numBatches);
    std::vector<BlockContactPoint> contactPoints(numBatches * maxRowsPerBatch);
    std::vector<BlockContactFriction> frictions(numBatches * 2);
    std::vector<BlockJointHeader> jointHeaders(numBatches);
    std::vector<BlockJointRowCon> jointRowsCon(numBatches * maxRowsPerBatch);
    std::vector<BlockJointRowMod> jointRowsMod(numBatches * maxRowsPerBatch);
    std::memset(contactHeaders.data(), 0, contactHeaders.size() * sizeof(BlockContactHeader));
    std::memset(frictionHeaders.data(), 0, frictionHeaders.size() * sizeof(BlockFrictionHeader));
    std::memset(contactPoints.data(), 0, contactPoints.size() * sizeof(BlockContactPoint));
    std::memset(frictions.data(), 0, frictions.size() * sizeof(BlockContactFriction));
    std::memset(jointHeaders.data(), 0, jointHeaders.size() * sizeof(BlockJointHeader));
    std::memset(jointRowsCon.data(), 0, jointRowsCon.size() * sizeof(BlockJointRowCon));
    std::memset(jointRowsMod.data(), 0, jointRowsMod.size() * sizeof(BlockJointRowMod));

    std::vector<Float4> velPool(numBatches * 128);
    for (Float4& v : velPool)
        v = Float4{ sym(rng), sym(rng), sym(rng), 0.f };

    std::vector<uint32_t> encoded(numTotalBodies * 2);
    for (uint32_t& e : encoded)
        e = rng();

    for (uint32_t k = 0; k < numBatches; ++k)
    {
        BlockConstraintBatch& b = batches[k];
        b.descStride = uint16_t(1 + rng() % 32);
        b.constraintType = uint16_t(k % 2); // alternate contact / joint
        b.constraintBatchIndex = k;
        b.startConstraintIndex = k * maxRowsPerBatch;
        b.startFrictionIndex = k * 2;

        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            b.bodyAIndex[lane] = rng() % numTotalBodies;
            b.bodyBIndex[lane] = rng() % numTotalBodies;
            // unique writeback slots: write back where we read
            b.remappedBodyAIndex[lane] = k * 128 + lane;
            b.remappedBodyBIndex[lane] = k * 128 + lane + 64;

            if (b.constraintType == 0)
            {
                BlockContactHeader& h = contactHeaders[k];
                h.invMass0_1_angDom0_1[lane] = Float4{ unit(rng), unit(rng), unit(rng), unit(rng) };
                const Vec3 n = randUnit(rng);
                h.normal_staticFriction[lane] = Float4{ n.x, n.y, n.z, unit(rng) };
                h.flags[lane] = unit(rng) < 0.5f ? COMPLIANT_ACCEL_SPRING : 0;
                h.numNormalConstr[lane] = rng() % (maxRowsPerBatch + 1);
                // mix rigid (>= 0) and compliant (< 0) restitution
                h.restitution[lane] = unit(rng) < 0.5f ? unit(rng) * 0.8f : -unit(rng) * 100.f;

                BlockFrictionHeader& fh = frictionHeaders[k];
                fh.numFrictionConstr[lane] = h.numNormalConstr[lane] ? (rng() % 3) : 0;
                fh.dynamicFriction[lane] = unit(rng);
                const Vec3 f0 = randUnit(rng), f1 = randUnit(rng);
                fh.frictionNormals0[lane] = Float4{ f0.x, f0.y, f0.z, 0 };
                fh.frictionNormals1[lane] = Float4{ f1.x, f1.y, f1.z, 0 };

                for (uint32_t i = 0; i < maxRowsPerBatch; ++i)
                {
                    BlockContactPoint& c = contactPoints[b.startConstraintIndex + i];
                    const Vec3 ra = Vec3(sym(rng), sym(rng), sym(rng)) * 0.3f;
                    const Vec3 rb = Vec3(sym(rng), sym(rng), sym(rng)) * 0.3f;
                    c.raXn_targetVelocity[lane] = Float4{ ra.x, ra.y, ra.z, sym(rng) * 0.1f };
                    c.rbXn_maxImpulse[lane] = Float4{ rb.x, rb.y, rb.z, 10.f + unit(rng) * 100.f };
                    c.appliedForce[lane] = unit(rng);
                    c.resp0[lane] = 0.1f + unit(rng);
                    c.resp1[lane] = 0.1f + unit(rng);
                    c.coeff0[lane] = unit(rng) * 0.2f;
                    c.coeff1[lane] = unit(rng) * 0.2f;
                }
                for (uint32_t i = 0; i < 2; ++i)
                {
                    BlockContactFriction& f = frictions[b.startFrictionIndex + i];
                    const Vec3 ra = Vec3(sym(rng), sym(rng), sym(rng)) * 0.3f;
                    const Vec3 rb = Vec3(sym(rng), sym(rng), sym(rng)) * 0.3f;
                    f.raXn_bias[lane] = Float4{ ra.x, ra.y, ra.z, sym(rng) * 0.05f };
                    f.rbXn_targetVelW[lane] = Float4{ rb.x, rb.y, rb.z, sym(rng) * 0.05f };
                    f.appliedForce[lane] = sym(rng) * 0.2f;
                    f.resp0[lane] = 0.1f + unit(rng);
                    f.resp1[lane] = 0.1f + unit(rng);
                }
            }
            else
            {
                BlockJointHeader& h = jointHeaders[k];
                h.rowCounts[lane] = uint16_t(rng() % (maxRowsPerBatch + 1));
                h.invMass0D0[lane] = unit(rng);
                h.invMass1D1[lane] = unit(rng);
                h.invInertiaScale0[lane] = unit(rng);
                h.invInertiaScale1[lane] = unit(rng);

                for (uint32_t i = 0; i < maxRowsPerBatch; ++i)
                {
                    BlockJointRowCon& c = jointRowsCon[b.startConstraintIndex + i];
                    BlockJointRowMod& m = jointRowsMod[b.startConstraintIndex + i];
                    const Vec3 l0 = randUnit(rng), l1 = randUnit(rng);
                    const Vec3 a0 = Vec3(sym(rng), sym(rng), sym(rng)) * 0.4f;
                    const Vec3 a1 = Vec3(sym(rng), sym(rng), sym(rng)) * 0.4f;
                    c.lin0XYZ_minImpulse[lane] = Float4{ l0.x, l0.y, l0.z, -50.f - unit(rng) * 50.f };
                    c.lin1XYZ_maxImpulse[lane] = Float4{ l1.x, l1.y, l1.z, 50.f + unit(rng) * 50.f };
                    c.ang0XYZ_resp0[lane] = Float4{ a0.x, a0.y, a0.z, 0.1f + unit(rng) };
                    c.ang1XYZ_resp1[lane] = Float4{ a1.x, a1.y, a1.z, 0.1f + unit(rng) };
                    c.initJointSpeed[lane] = sym(rng) * 0.1f;
                    m.appliedForce[lane] = sym(rng) * 0.2f;
                    uint32_t fl = 0;
                    const float r = unit(rng);
                    if (r < 0.4f)
                        fl |= DY_SC_FLAG_SPRING;
                    if (r < 0.2f)
                        fl |= DY_SC_FLAG_ACCELERATION_SPRING;
                    m.flags[lane] = fl;
                    m.coeff0[lane] = unit(rng) * 0.3f;
                    m.coeff1[lane] = unit(rng) * 0.3f;
                }
            }
        }
    }

    // ---- GPU ----
    PxgSlangComputeDevice* dev = testDevice();

    Kernel kernel("solverPGS.spv", "solveBlockPartition",
                  { sizeof(BlockConstraintBatch), sizeof(BlockContactHeader), sizeof(BlockFrictionHeader),
                    sizeof(BlockContactPoint), sizeof(BlockContactFriction), sizeof(BlockJointHeader),
                    sizeof(BlockJointRowCon), sizeof(BlockJointRowMod), 16, 4 },
                  sizeof(SolverParams));

    PxgBufferHandle batchBuf = createBufferFrom(dev, batches);
    PxgBufferHandle chBuf = createBufferFrom(dev, contactHeaders);
    PxgBufferHandle fhBuf = createBufferFrom(dev, frictionHeaders);
    PxgBufferHandle cpBuf = createBufferFrom(dev, contactPoints);
    PxgBufferHandle frBuf = createBufferFrom(dev, frictions);
    PxgBufferHandle jhBuf = createBufferFrom(dev, jointHeaders);
    PxgBufferHandle jcBuf = createBufferFrom(dev, jointRowsCon);
    PxgBufferHandle jmBuf = createBufferFrom(dev, jointRowsMod);
    PxgBufferHandle velBuf = createBufferFrom(dev, velPool);
    PxgBufferHandle encBuf = createBufferFrom(dev, encoded);

    const SolverParams params = { 0, numBatches, 1u, 1u, numSlabs, numTotalBodies, bodyOffset };
    kernel.dispatch(numBatches, { batchBuf, chBuf, fhBuf, cpBuf, frBuf, jhBuf, jcBuf, jmBuf, velBuf, encBuf },
                    params);
    dev->flush();

    // ---- reference ----
    std::vector<Float4> refVel = velPool;
    for (uint32_t k = 0; k < numBatches; ++k)
    {
        const BlockConstraintBatch& b = batches[k];
        for (uint32_t lane = 0; lane < b.descStride; ++lane)
        {
            const uint32_t readIndex = k * 128 + lane;
            Vec3 lv0(refVel[readIndex].x, refVel[readIndex].y, refVel[readIndex].z);
            Vec3 av0(refVel[readIndex + 32].x, refVel[readIndex + 32].y, refVel[readIndex + 32].z);
            Vec3 lv1(refVel[readIndex + 64].x, refVel[readIndex + 64].y, refVel[readIndex + 64].z);
            Vec3 av1(refVel[readIndex + 96].x, refVel[readIndex + 96].y, refVel[readIndex + 96].z);

            float ref0 = 1.f, ref1 = 1.f;
            if (b.bodyAIndex[lane] >= bodyOffset)
                ref0 = float(refCountActiveSlabs(b.bodyAIndex[lane], numSlabs, numTotalBodies, encoded));
            if (b.bodyBIndex[lane] >= bodyOffset)
                ref1 = float(refCountActiveSlabs(b.bodyBIndex[lane], numSlabs, numTotalBodies, encoded));

            if (b.constraintType == 0)
                refSolveContact(b, lv0, av0, lv1, av1, true, lane, contactHeaders, frictionHeaders,
                                contactPoints, frictions, ref0, ref1);
            else
                refSolve1D(b, lv0, av0, lv1, av1, lane, true, jointHeaders, jointRowsCon, jointRowsMod,
                           ref0, ref1);

            refVel[b.remappedBodyAIndex[lane]] = Float4{ lv0.x, lv0.y, lv0.z, 0.f };
            refVel[b.remappedBodyAIndex[lane] + 32] = Float4{ av0.x, av0.y, av0.z, 0.f };
            refVel[b.remappedBodyBIndex[lane]] = Float4{ lv1.x, lv1.y, lv1.z, 0.f };
            refVel[b.remappedBodyBIndex[lane] + 32] = Float4{ av1.x, av1.y, av1.z, 0.f };
        }
    }

    // ---- compare ----
    const std::vector<Float4> gpuVel = downloadBuffer<Float4>(dev, velBuf, velPool.size());
    const std::vector<BlockContactPoint> gpuContacts = downloadBuffer<BlockContactPoint>(dev, cpBuf, contactPoints.size());
    const std::vector<BlockContactFriction> gpuFrictions = downloadBuffer<BlockContactFriction>(dev, frBuf, frictions.size());
    const std::vector<BlockFrictionHeader> gpuFrictionHeaders = downloadBuffer<BlockFrictionHeader>(dev, fhBuf, frictionHeaders.size());
    const std::vector<BlockJointRowMod> gpuJointMods = downloadBuffer<BlockJointRowMod>(dev, jmBuf, jointRowsMod.size());

    size_t velMismatches = 0;
    for (size_t i = 0; i < refVel.size(); ++i)
    {
        if (!almostEqual(gpuVel[i].x, refVel[i].x, 1e-4f, 1e-4f) ||
            !almostEqual(gpuVel[i].y, refVel[i].y, 1e-4f, 1e-4f) ||
            !almostEqual(gpuVel[i].z, refVel[i].z, 1e-4f, 1e-4f))
        {
            if (velMismatches < 5)
                PXG_CHECK_MSG(false, "vel[%zu]: gpu=(%g,%g,%g) ref=(%g,%g,%g)", i, gpuVel[i].x,
                              gpuVel[i].y, gpuVel[i].z, refVel[i].x, refVel[i].y, refVel[i].z);
            ++velMismatches;
        }
    }
    PXG_CHECK_MSG(velMismatches == 0, "%zu velocity mismatches", velMismatches);

    size_t forceMismatches = 0;
    for (size_t c = 0; c < contactPoints.size(); ++c)
        for (uint32_t lane = 0; lane < 32; ++lane)
            if (!almostEqual(gpuContacts[c].appliedForce[lane], contactPoints[c].appliedForce[lane], 1e-4f, 1e-4f))
                ++forceMismatches;
    for (size_t f = 0; f < frictions.size(); ++f)
        for (uint32_t lane = 0; lane < 32; ++lane)
            if (!almostEqual(gpuFrictions[f].appliedForce[lane], frictions[f].appliedForce[lane], 1e-4f, 1e-4f))
                ++forceMismatches;
    for (size_t j = 0; j < jointRowsMod.size(); ++j)
        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            if (!almostEqual(gpuJointMods[j].appliedForce[lane], jointRowsMod[j].appliedForce[lane], 1e-4f, 1e-4f))
                ++forceMismatches;
            if (!almostEqual(gpuJointMods[j].residual[lane], jointRowsMod[j].residual[lane], 1e-3f, 1e-3f))
                ++forceMismatches;
        }
    PXG_CHECK_MSG(forceMismatches == 0, "%zu applied-force/residual mismatches", forceMismatches);

    size_t brokenMismatches = 0;
    for (size_t h = 0; h < frictionHeaders.size(); ++h)
        for (uint32_t lane = 0; lane < 32; ++lane)
            if (gpuFrictionHeaders[h].broken[lane] != frictionHeaders[h].broken[lane])
                ++brokenMismatches;
    PXG_CHECK_MSG(brokenMismatches == 0, "%zu friction-broken mismatches", brokenMismatches);

    dev->destroyBuffer(batchBuf);
    dev->destroyBuffer(chBuf);
    dev->destroyBuffer(fhBuf);
    dev->destroyBuffer(cpBuf);
    dev->destroyBuffer(frBuf);
    dev->destroyBuffer(jhBuf);
    dev->destroyBuffer(jcBuf);
    dev->destroyBuffer(jmBuf);
    dev->destroyBuffer(velBuf);
    dev->destroyBuffer(encBuf);
}

} // namespace
