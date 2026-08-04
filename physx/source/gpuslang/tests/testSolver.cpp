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

// Verification of solverPGS.slang on the BDA ENGINE ABI (P5 Step 5): every
// kernel is launched through PxgCudaShim with the engine's exact CUDA
// parameter pack — REAL PxgSolverCoreDesc / PxgSolverSharedDesc<
// IterativeSolveData> / PxgIslandContext structs filled with live device
// addresses — and compared against a C++ transcription of solverMultiBlock.cu
// / solverBlock.cuh / solverBlockCommon.cuh / constraintPrepShared.cuh /
// DyCpuGpu1dConstraint.h.

#include "pxgTestFramework.h"
#include "pxgGpuStructs.h"
#include "pxgRefMath.h"
#include "PxgCudaShim.h"

#include "PxNodeIndex.h" // completes PxNodeIndex for the PxgNarrowphaseCore.h chain
#include "PxgSolverCoreDesc.h"
#include "PxgIslandContext.h"

#include <cstring>
#include <numeric>
#include <random>

namespace
{

using namespace pxgtest;

const float DY_MIN_RESPONSE = 1e-12f;
const uint32_t DY_SC_FLAG_SPRING = 1u << 5;
const uint32_t DY_SC_FLAG_ACCELERATION_SPRING = 1u << 7;
const uint32_t COMPLIANT_ACCEL_SPRING = 1u << 3;

// The PGS shared desc PxgSolverSharedDesc<IterativeSolveData> must have the
// same byte layout as the TGS instantiation (the pxgSlangSolverDesc mirror
// relies on it, and the TGS layout is empirically pinned by
// testSolverDescLayout): 16 pointers, base + 128.
static_assert(sizeof(physx::IterativeSolveData) == 16 * sizeof(void*),
              "IterativeSolveData is 16 device pointers");
static_assert(sizeof(physx::PxgSolverSharedDesc<physx::IterativeSolveData>) ==
                  sizeof(physx::PxgSolverSharedDesc<physx::IterativeSolveDataTGS>),
              "PGS/TGS shared desc instantiations must have identical layout");

// ---- block SoA mirrors (sizes pinned to the CUDA layouts) ----

struct BlockConstraintBatch
{
    uint16_t descStride;
    uint16_t constraintType; // engine PxgConstraintType: 0 = 1D joint, 1 = contact
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
static_assert(sizeof(ThresholdElement) == 40, "ThresholdStreamElement mirror");

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
static_assert(sizeof(FrictionPatchGPU) == 52, "PxgFrictionPatchGPU mirror");

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
static_assert(sizeof(BlockFrictionPatch) == 2816, "PxgBlockFrictionPatch mirror");

// ---------------------------------------------------------------------------
// CPU reference (transcriptions)
// ---------------------------------------------------------------------------

// PxgErrorAccumulator reference (solverResidual.cuh): per-thread contact
// residual statistics, accumulated by the contact solve.
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

// solveContactBlock() incl. the PxgErrorAccumulator statistics (err may be null)
void refSolveContact(const BlockConstraintBatch& batch, Vec3& lv0, Vec3& av0, Vec3& lv1, Vec3& av1,
                     bool doFriction, uint32_t lane, std::vector<BlockContactHeader>& contactHeaders,
                     std::vector<BlockFrictionHeader>& frictionHeaders,
                     std::vector<BlockContactPoint>& contacts, std::vector<BlockContactFriction>& frictions,
                     float ref0, float ref1, RefErrAcc* err)
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

        if (err)
            err->add(deltaF, velMultiplier);

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

            if (err)
                err->add(deltaF, velMultiplier);

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

uint32_t refPopcount(uint32_t v)
{
    uint32_t c = 0;
    while (v) { c += v & 1; v >>= 1; }
    return c;
}

uint32_t refCountActiveSlabs(uint32_t index, uint32_t numSlabs, uint32_t numBodies,
                             const std::vector<uint32_t>& encoded)
{
    uint32_t count = 0;
    for (uint32_t i = 0; i < (numSlabs + 31) / 32; ++i)
        count += refPopcount(encoded[index + i * numBodies]);
    return std::max(1u, count);
}

Vec3 randUnit(std::mt19937& rng)
{
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    Vec3 v(d(rng), d(rng), d(rng));
    return v.magnitude() > 1e-3f ? v.getNormalized() : Vec3(1, 0, 0);
}

// Shared engine-ABI test helpers (used by every PxgCudaShim-driven test below).
static std::vector<uint8_t> solverPgsAbiLoader(const char* file) { return loadShaderBlob(file); }

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
// solveBlockPartition (ENGINE ABI): contact + 1D joint constraint solve over
// TWO partitions (two launches — the engine launches once per partition),
// exercising both start-index derivations (mBatchStartIndex for partition 0,
// constraintsPerPartition[p + startPartitionIndex - 1] for partition 1), the
// mass-splitting reference counts (numSlabs > 32 -> multi-word popcounts) and
// the contact-residual accumulator (accumulateErrorGlobalFullWarp).
// islandIndex = 1 pins the PxgIslandContext stride; element [0] carries the
// REAL body fields (the .cu reads mBodyStartIndex/mBodyCount/mArticulationCount
// from islandContextPool[0]) but DECOY partition/batch fields, while element
// [1] carries the real partition/batch fields but decoy body counts.
// ---------------------------------------------------------------------------
PXG_TEST(solverPGS_contactAndJointBatches)
{
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    std::mt19937 rng(3000);
    std::uniform_real_distribution<float> unit(0.f, 1.f);
    std::uniform_real_distribution<float> sym(-3.f, 3.f);

    const uint32_t maxRowsPerBatch = 4;
    const uint32_t numSlabs = 40; // > 32 exercises the multi-word reference counts
    const uint32_t bodyOffset = 16;
    const uint32_t numDynamicBodies = 4080;
    const uint32_t numTotalBodies = bodyOffset + numDynamicBodies; // no articulations
    const uint32_t num32Slabs = (numSlabs + 31) / 32;

    // batches: [0] decoy (never solved); partition 0 = [1, 13), partition 1 =
    // [13, 25).  mBatchStartIndex = 1, mStartPartitionIndex = 1 =>
    // partition 0 end = cpp[1] = 13, partition 1 start = cpp[1], end = cpp[2].
    const uint32_t numAllBatches = 25;
    const uint32_t batchStartIndex = 1;
    std::vector<uint32_t> cpp = { 777u, 13u, 25u }; // index 0 decoy

    std::vector<BlockConstraintBatch> batches(numAllBatches);
    std::memset(batches.data(), 0, batches.size() * sizeof(BlockConstraintBatch));
    std::vector<BlockContactHeader> contactHeaders(numAllBatches);
    std::vector<BlockFrictionHeader> frictionHeaders(numAllBatches);
    std::vector<BlockContactPoint> contactPoints(numAllBatches * maxRowsPerBatch);
    std::vector<BlockContactFriction> frictions(numAllBatches * 2);
    std::vector<BlockJointHeader> jointHeaders(numAllBatches);
    std::vector<BlockJointRowCon> jointRowsCon(numAllBatches * maxRowsPerBatch);
    std::vector<BlockJointRowMod> jointRowsMod(numAllBatches * maxRowsPerBatch);
    std::memset(contactHeaders.data(), 0, contactHeaders.size() * sizeof(BlockContactHeader));
    std::memset(frictionHeaders.data(), 0, frictionHeaders.size() * sizeof(BlockFrictionHeader));
    std::memset(contactPoints.data(), 0, contactPoints.size() * sizeof(BlockContactPoint));
    std::memset(frictions.data(), 0, frictions.size() * sizeof(BlockContactFriction));
    std::memset(jointHeaders.data(), 0, jointHeaders.size() * sizeof(BlockJointHeader));
    std::memset(jointRowsCon.data(), 0, jointRowsCon.size() * sizeof(BlockJointRowCon));
    std::memset(jointRowsMod.data(), 0, jointRowsMod.size() * sizeof(BlockJointRowMod));

    // per-partition PGS velocity pool: 4 float4 stripes per batch lane
    // (readIndex = k * 128 + lane)
    std::vector<Float4> velPool(numAllBatches * 128);
    for (Float4& v : velPool)
        v = Float4{ sym(rng), sym(rng), sym(rng), 0.f };

    std::vector<uint32_t> encoded(numTotalBodies * num32Slabs);
    for (uint32_t& e : encoded)
        e = rng();

    for (uint32_t k = batchStartIndex; k < numAllBatches; ++k)
    {
        BlockConstraintBatch& b = batches[k];
        b.descStride = uint16_t(1 + rng() % 32);
        b.constraintType = uint16_t(k % 2); // alternate joint (0) / contact (1)
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

            if (b.constraintType == 1) // engine encoding: eCONTACT = 1
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

    // ---- GPU (engine ABI through the shim) ----
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "solverPGS.spv", "solveBlockPartition", 32u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &solverPgsAbiLoader);

    PxgDevicePtr dVel = 0, dEnc = 0, dCpp = 0, dIsl = 0;
    PxgDevicePtr dBatch = 0, dCh = 0, dFh = 0, dCp = 0, dFr = 0, dJh = 0, dJc = 0, dJm = 0;
    PxgDevicePtr dDesc = 0, dShared = 0;
    shim.memAlloc(&dVel, velPool.size() * sizeof(Float4));
    shim.memAlloc(&dEnc, encoded.size() * 4);
    shim.memAlloc(&dCpp, cpp.size() * 4);
    shim.memAlloc(&dBatch, batches.size() * sizeof(BlockConstraintBatch));
    shim.memAlloc(&dCh, contactHeaders.size() * sizeof(BlockContactHeader));
    shim.memAlloc(&dFh, frictionHeaders.size() * sizeof(BlockFrictionHeader));
    shim.memAlloc(&dCp, contactPoints.size() * sizeof(BlockContactPoint));
    shim.memAlloc(&dFr, frictions.size() * sizeof(BlockContactFriction));
    shim.memAlloc(&dJh, jointHeaders.size() * sizeof(BlockJointHeader));
    shim.memAlloc(&dJc, jointRowsCon.size() * sizeof(BlockJointRowCon));
    shim.memAlloc(&dJm, jointRowsMod.size() * sizeof(BlockJointRowMod));
    if (!dVel || !dEnc || !dCpp || !dBatch || !dCh || !dFh || !dCp || !dFr || !dJh || !dJc || !dJm)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    shim.memcpyHtoD(dVel, velPool.data(), velPool.size() * sizeof(Float4));
    shim.memcpyHtoD(dEnc, encoded.data(), encoded.size() * 4);
    shim.memcpyHtoD(dCpp, cpp.data(), cpp.size() * 4);
    shim.memcpyHtoD(dBatch, batches.data(), batches.size() * sizeof(BlockConstraintBatch));
    shim.memcpyHtoD(dCh, contactHeaders.data(), contactHeaders.size() * sizeof(BlockContactHeader));
    shim.memcpyHtoD(dFh, frictionHeaders.data(), frictionHeaders.size() * sizeof(BlockFrictionHeader));
    shim.memcpyHtoD(dCp, contactPoints.data(), contactPoints.size() * sizeof(BlockContactPoint));
    shim.memcpyHtoD(dFr, frictions.data(), frictions.size() * sizeof(BlockContactFriction));
    shim.memcpyHtoD(dJh, jointHeaders.data(), jointHeaders.size() * sizeof(BlockJointHeader));
    shim.memcpyHtoD(dJc, jointRowsCon.data(), jointRowsCon.size() * sizeof(BlockJointRowCon));
    shim.memcpyHtoD(dJm, jointRowsMod.data(), jointRowsMod.size() * sizeof(BlockJointRowMod));

    physx::PxgIslandContext islands[2];
    std::memset(islands, 0, sizeof(islands));
    // element [0]: REAL body fields (the .cu reads them via islandContextPool->),
    // DECOY partition/batch fields.
    islands[0].mBodyStartIndex = bodyOffset;
    islands[0].mBodyCount = numDynamicBodies;
    islands[0].mArticulationCount = 0;
    islands[0].mStartPartitionIndex = 4u; // decoy
    islands[0].mNumPartitions = 9u;       // decoy
    islands[0].mBatchStartIndex = 42u;    // decoy
    // element [1] = islandIndex: the real island-indexed fields.
    islands[1].mBodyStartIndex = bodyOffset;
    islands[1].mBodyCount = 999u;         // decoy: kernel reads mBodyCount from [0]
    islands[1].mArticulationCount = 7u;   // decoy
    islands[1].mStartPartitionIndex = 1;
    islands[1].mNumPartitions = 2;
    islands[1].mBatchStartIndex = batchStartIndex;
    shim.memAlloc(&dIsl, sizeof(islands));
    shim.memcpyHtoD(dIsl, islands, sizeof(islands));

    physx::PxgSolverCoreDesc desc;
    std::memset(&desc, 0, sizeof(desc));
    desc.islandContextPool = reinterpret_cast<physx::PxgIslandContext*>(dIsl);
    desc.constraintsPerPartition = reinterpret_cast<physx::PxU32*>(dCpp);
    desc.numSlabs = numSlabs;
    desc.contactErrorAccumulator.mErrorSumOfSquares = 0.5f;
    desc.contactErrorAccumulator.mCounter = 0; // >= 0: residual accumulation ON
    desc.contactErrorAccumulator.mMaxError = 0.0625f;

    physx::PxgSolverSharedDesc<physx::IterativeSolveData> shared;
    std::memset(&shared, 0, sizeof(shared));
    shared.iterativeData.solverBodyVelPool = reinterpret_cast<float4*>(dVel);
    shared.iterativeData.solverEncodedReferenceCount = reinterpret_cast<physx::PxU32*>(dEnc);
    shared.iterativeData.blockConstraintBatch = reinterpret_cast<physx::PxgBlockConstraintBatch*>(dBatch);
    shared.iterativeData.blockContactHeaders = reinterpret_cast<physx::PxgBlockSolverContactHeader*>(dCh);
    shared.iterativeData.blockFrictionHeaders = reinterpret_cast<physx::PxgBlockSolverFrictionHeader*>(dFh);
    shared.iterativeData.blockContactPoints = reinterpret_cast<physx::PxgBlockSolverContactPoint*>(dCp);
    shared.iterativeData.blockFrictions = reinterpret_cast<physx::PxgBlockSolverContactFriction*>(dFr);
    shared.iterativeData.blockJointConstraintHeaders =
        reinterpret_cast<physx::PxgBlockSolverConstraint1DHeader*>(dJh);
    shared.iterativeData.blockJointConstraintRowsCon =
        reinterpret_cast<physx::PxgBlockSolverConstraint1DCon*>(dJc);
    shared.iterativeData.blockJointConstraintRowsMod =
        reinterpret_cast<physx::PxgBlockSolverConstraint1DMod*>(dJm);

    shim.memAlloc(&dDesc, sizeof(desc));
    shim.memcpyHtoD(dDesc, &desc, sizeof(desc));
    shim.memAlloc(&dShared, sizeof(shared));
    shim.memcpyHtoD(dShared, &shared, sizeof(shared));

    // engine launch: once per partition; nbBlocks = ceil(batchesInPartition *
    // 32 / SOLVE_BLOCK_PARTITION(64)) = ceil(12/2) = 6 -> 12 warps.
    const uint32_t nbBlocks = 6;
    for (uint32_t partition = 0; partition < 2; ++partition)
    {
        EngineBlob blob;
        blob.add<uint64_t>(dDesc);       // @0  solverCoreDesc
        blob.add<uint64_t>(dShared);     // @8  sharedDesc
        blob.add<uint32_t>(1u);          // @16 islandIndex
        blob.add<uint32_t>(partition);   // @20 partitionIndex
        blob.add<uint8_t>(1u);           // @24 doFriction
        blob.finish();
        PXG_CHECK_MSG(blob.bytes.size() == 32, "solveBlockPartition blob is 32 bytes (got %zu)",
                      blob.bytes.size());
        PXG_CHECK_MSG(shim.launchKernel(0, nbBlocks, 1, 1, blob.bytes.data(),
                                        uint32_t(blob.bytes.size())) == PxgResult::Success,
                      "launch solveBlockPartition (partition %u)", partition);
    }
    shim.streamSynchronize();

    // ---- reference ----
    std::vector<Float4> refVel = velPool;
    const uint32_t numWarpsInGrid = nbBlocks * 2;
    // per-launch, per-warp, per-lane contact accumulators
    std::vector<RefErrAcc> errAcc(2 * numWarpsInGrid * 32);

    for (uint32_t partition = 0; partition < 2; ++partition)
    {
        const uint32_t startIndex = partition == 0 ? batchStartIndex : cpp[1];
        const uint32_t endIndex = cpp[1 + partition];
        for (uint32_t k = startIndex; k < endIndex; ++k)
        {
            const uint32_t warp = (k - startIndex) % numWarpsInGrid;
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

                if (b.constraintType == 1) // engine encoding: eCONTACT = 1
                    refSolveContact(b, lv0, av0, lv1, av1, true, lane, contactHeaders, frictionHeaders,
                                    contactPoints, frictions, ref0, ref1,
                                    &errAcc[(partition * numWarpsInGrid + warp) * 32 + lane]);
                else
                    refSolve1D(b, lv0, av0, lv1, av1, lane, true, jointHeaders, jointRowsCon, jointRowsMod,
                               ref0, ref1);

                refVel[b.remappedBodyAIndex[lane]] = Float4{ lv0.x, lv0.y, lv0.z, 0.f };
                refVel[b.remappedBodyAIndex[lane] + 32] = Float4{ av0.x, av0.y, av0.z, 0.f };
                refVel[b.remappedBodyBIndex[lane]] = Float4{ lv1.x, lv1.y, lv1.z, 0.f };
                refVel[b.remappedBodyBIndex[lane] + 32] = Float4{ av1.x, av1.y, av1.z, 0.f };
            }
        }
    }

    // accumulateErrorGlobalFullWarp reference: __shfl_xor butterfly (radius
    // 16..1) per warp, then per-warp partials into the accumulator (order-
    // independent under tolerance).
    float expectSum = 0.5f;
    uint32_t expectCount = 0;
    float expectMax = 0.0625f;
    for (uint32_t w = 0; w < 2 * numWarpsInGrid; ++w)
    {
        float sums[32], maxs[32];
        uint32_t cnts[32];
        for (uint32_t l = 0; l < 32; ++l)
        {
            sums[l] = errAcc[w * 32 + l].sumSq;
            maxs[l] = errAcc[w * 32 + l].maxErr;
            cnts[l] = errAcc[w * 32 + l].counter;
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
    std::vector<Float4> gpuVel(velPool.size());
    shim.memcpyDtoH(gpuVel.data(), dVel, gpuVel.size() * sizeof(Float4));
    std::vector<BlockContactPoint> gpuContacts(contactPoints.size());
    shim.memcpyDtoH(gpuContacts.data(), dCp, gpuContacts.size() * sizeof(BlockContactPoint));
    std::vector<BlockContactFriction> gpuFrictions(frictions.size());
    shim.memcpyDtoH(gpuFrictions.data(), dFr, gpuFrictions.size() * sizeof(BlockContactFriction));
    std::vector<BlockFrictionHeader> gpuFrictionHeaders(frictionHeaders.size());
    shim.memcpyDtoH(gpuFrictionHeaders.data(), dFh, gpuFrictionHeaders.size() * sizeof(BlockFrictionHeader));
    std::vector<BlockJointRowMod> gpuJointMods(jointRowsMod.size());
    shim.memcpyDtoH(gpuJointMods.data(), dJm, gpuJointMods.size() * sizeof(BlockJointRowMod));

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

    physx::PxgSolverCoreDesc gpuDesc;
    shim.memcpyDtoH(&gpuDesc, dDesc, sizeof(gpuDesc));
    PXG_CHECK_MSG(almostEqual(gpuDesc.contactErrorAccumulator.mErrorSumOfSquares, expectSum, 2e-4f, 2e-4f),
                  "residual sum-of-squares: gpu=%g ref=%g",
                  gpuDesc.contactErrorAccumulator.mErrorSumOfSquares, expectSum);
    PXG_CHECK_MSG(gpuDesc.contactErrorAccumulator.mCounter == int32_t(expectCount),
                  "residual counter: gpu=%d ref=%u", gpuDesc.contactErrorAccumulator.mCounter, expectCount);
    PXG_CHECK_MSG(almostEqual(gpuDesc.contactErrorAccumulator.mMaxError, expectMax, 2e-4f, 2e-4f),
                  "residual max: gpu=%g ref=%g", gpuDesc.contactErrorAccumulator.mMaxError, expectMax);

    for (PxgDevicePtr pfree : { dVel, dEnc, dCpp, dIsl, dBatch, dCh, dFh, dCp, dFr, dJh, dJc, dJm,
                                dDesc, dShared })
        shim.memFree(pfree);
}

// ---------------------------------------------------------------------------
// concludeBlocks + writebackBlocks (ENGINE ABI): the batch range is derived
// from the island context (mBatchStartIndex + the FIVE-count sum — the counts
// are split across mBatchCount/mArtiBatchCount/mStaticArtiBatchCount/
// mStaticRigidBatchCount to pin the full expression); batch [0] is a decoy.
// The threshold stream is appended through the desc's own
// sharedThresholdStreamIndex field (atomic).
// ---------------------------------------------------------------------------
PXG_TEST(solverPGS_concludeAndWriteback)
{
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    std::mt19937 rng(3100);
    std::uniform_real_distribution<float> unit(0.f, 1.f);
    std::uniform_real_distribution<float> sym(-2.f, 2.f);

    const uint32_t numRealBatches = 12;
    const uint32_t numAllBatches = numRealBatches + 1; // [0] decoy
    const uint32_t batchStartIndex = 1;
    const uint32_t rowsPerBatch = 4;
    const uint32_t numBodies = 64;
    const float PX_MAX_REAL_F = 3.4028234663852885981170418348452e+38f;

    std::vector<BlockConstraintBatch> batches(numAllBatches);
    std::vector<BlockContactHeader> contactHeaders(numAllBatches);
    std::vector<BlockFrictionHeader> frictionHeaders(numAllBatches);
    std::vector<BlockContactPoint> contactPoints(numAllBatches * rowsPerBatch);
    std::vector<BlockContactFriction> frictions(numAllBatches * rowsPerBatch);
    std::vector<BlockJointHeader> jointHeaders(numAllBatches);
    std::vector<BlockJointRowCon> jointRowsCon(numAllBatches * rowsPerBatch);
    std::vector<BlockJointRowMod> jointRowsMod(numAllBatches * rowsPerBatch);
    std::vector<BlockFrictionPatch> frictionPatchBlocks(numAllBatches);
    std::memset(batches.data(), 0, batches.size() * sizeof(BlockConstraintBatch));
    std::memset(contactHeaders.data(), 0, contactHeaders.size() * sizeof(BlockContactHeader));
    std::memset(frictionHeaders.data(), 0, frictionHeaders.size() * sizeof(BlockFrictionHeader));
    std::memset(contactPoints.data(), 0, contactPoints.size() * sizeof(BlockContactPoint));
    std::memset(frictions.data(), 0, frictions.size() * sizeof(BlockContactFriction));
    std::memset(jointHeaders.data(), 0, jointHeaders.size() * sizeof(BlockJointHeader));
    std::memset(jointRowsCon.data(), 0, jointRowsCon.size() * sizeof(BlockJointRowCon));
    std::memset(jointRowsMod.data(), 0, jointRowsMod.size() * sizeof(BlockJointRowMod));
    std::memset(frictionPatchBlocks.data(), 0, frictionPatchBlocks.size() * sizeof(BlockFrictionPatch));

    std::vector<SolverBodyData> bodyData(numBodies);
    std::memset(bodyData.data(), 0, bodyData.size() * sizeof(SolverBodyData));
    for (uint32_t b = 0; b < numBodies; ++b)
    {
        bodyData[b].reportThreshold = unit(rng) < 0.5f ? unit(rng) * 5.f : PX_MAX_REAL_F;
        bodyData[b].islandNodeIndex = makeNodeIndex(rng() % 1000);
    }

    const uint32_t numPatches = numAllBatches * 32;
    std::vector<FrictionPatchGPU> patches(numPatches);
    std::memset(patches.data(), 0, patches.size() * sizeof(FrictionPatchGPU));

    uint32_t forceBufCursor = 0;
    uint32_t writebackCursor = 0;

    for (uint32_t k = batchStartIndex; k < numAllBatches; ++k)
    {
        BlockConstraintBatch& b = batches[k];
        b.descStride = uint16_t(1 + rng() % 32);
        b.constraintType = uint16_t(k % 2);
        b.constraintBatchIndex = k;
        b.startConstraintIndex = k * rowsPerBatch;
        b.startFrictionIndex = k * rowsPerBatch;

        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            b.bodyAIndex[lane] = rng() % numBodies;
            b.bodyBIndex[lane] = rng() % numBodies;
            b.shapeInteraction[lane] = (uint64_t(rng()) << 32) | rng();

            if (b.constraintType == 1) // engine encoding: eCONTACT = 1
            {
                contactHeaders[k].numNormalConstr[lane] = rng() % (rowsPerBatch + 1);
                contactHeaders[k].restitution[lane] = unit(rng) < 0.5f ? unit(rng) : -unit(rng) * 50.f;
                contactHeaders[k].flags[lane] = unit(rng) < 0.6f ? 1u /*eHAS_FORCE_THRESHOLDS*/ : 0u;
                contactHeaders[k].forceWritebackOffset[lane] =
                    unit(rng) < 0.8f ? (forceBufCursor += rowsPerBatch) - rowsPerBatch : 0xFFFFFFFFu;
                frictionHeaders[k].numFrictionConstr[lane] =
                    contactHeaders[k].numNormalConstr[lane] ? (rng() % 3) * 2 : 0; // 0, 2 or 4
                frictionHeaders[k].broken[lane] = rng() & 1;
                const Vec3 f0 = randUnit(rng), f1 = randUnit(rng);
                frictionHeaders[k].frictionNormals0[lane] = Float4{ f0.x, f0.y, f0.z, 0 };
                frictionHeaders[k].frictionNormals1[lane] = Float4{ f1.x, f1.y, f1.z, 0 };
                frictionPatchBlocks[k].patchIndex[lane] = unit(rng) < 0.7f ? k * 32 + lane : 0xFFFFFFFFu;
                frictionPatchBlocks[k].anchorPoints[0][lane] = Float4{ sym(rng), sym(rng), sym(rng), 0 };
                frictionPatchBlocks[k].anchorPoints[1][lane] = Float4{ sym(rng), sym(rng), sym(rng), 0 };

                for (uint32_t i = 0; i < rowsPerBatch; ++i)
                {
                    BlockContactPoint& c = contactPoints[b.startConstraintIndex + i];
                    c.appliedForce[lane] = unit(rng) * 3.f;
                    c.coeff0[lane] = unit(rng);
                    c.coeff1[lane] = unit(rng);
                    BlockContactFriction& f = frictions[b.startFrictionIndex + i];
                    f.appliedForce[lane] = sym(rng);
                    f.raXn_bias[lane] = Float4{ sym(rng), sym(rng), sym(rng), sym(rng) };
                }
            }
            else
            {
                jointHeaders[k].rowCounts[lane] = uint16_t(rng() % (rowsPerBatch + 1));
                jointHeaders[k].breakable[lane] = uint16_t(rng() & 1);
                jointHeaders[k].angBreakImpulse[lane] = unit(rng) * 2.f;
                jointHeaders[k].body0WorldOffset_linBreakImpulse[lane] =
                    Float4{ sym(rng) * 0.2f, sym(rng) * 0.2f, sym(rng) * 0.2f, unit(rng) * 2.f };
                jointHeaders[k].writeBackOffset[lane] =
                    unit(rng) < 0.8f ? writebackCursor++ : 0xFFFFFFFFu;

                for (uint32_t i = 0; i < rowsPerBatch; ++i)
                {
                    BlockJointRowCon& c = jointRowsCon[b.startConstraintIndex + i];
                    BlockJointRowMod& m = jointRowsMod[b.startConstraintIndex + i];
                    const Vec3 l0 = randUnit(rng);
                    c.lin0XYZ_minImpulse[lane] = Float4{ l0.x, l0.y, l0.z, -10.f };
                    m.ang0Writeback[lane * 3 + 0] = sym(rng) * 0.3f;
                    m.ang0Writeback[lane * 3 + 1] = sym(rng) * 0.3f;
                    m.ang0Writeback[lane * 3 + 2] = sym(rng) * 0.3f;
                    m.appliedForce[lane] = sym(rng);
                    m.residual[lane] = sym(rng) * 0.05f;
                    m.flags[lane] = (unit(rng) < 0.7f ? 2u /*OUTPUT_FORCE*/ : 0u) |
                                    (unit(rng) < 0.4f ? DY_SC_FLAG_SPRING : 0u);
                    m.coeff0[lane] = unit(rng);
                    m.coeff1[lane] = unit(rng);
                }
            }
        }
    }

    // ---- GPU (engine ABI through the shim) ----
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "solverPGS.spv", "concludeBlocks", 24u, 0 },
        { "solverPGS.spv", "writebackBlocks", 24u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 2, &solverPgsAbiLoader);

    PxgDevicePtr dBatch = 0, dCh = 0, dFh = 0, dCp = 0, dFr = 0, dJh = 0, dJc = 0, dJm = 0;
    PxgDevicePtr dBd = 0, dWb = 0, dForce = 0, dTs = 0, dFpb = 0, dFp = 0, dIsl = 0;
    PxgDevicePtr dDesc = 0, dShared = 0;
    shim.memAlloc(&dBatch, batches.size() * sizeof(BlockConstraintBatch));
    shim.memAlloc(&dCh, contactHeaders.size() * sizeof(BlockContactHeader));
    shim.memAlloc(&dFh, frictionHeaders.size() * sizeof(BlockFrictionHeader));
    shim.memAlloc(&dCp, contactPoints.size() * sizeof(BlockContactPoint));
    shim.memAlloc(&dFr, frictions.size() * sizeof(BlockContactFriction));
    shim.memAlloc(&dJh, jointHeaders.size() * sizeof(BlockJointHeader));
    shim.memAlloc(&dJc, jointRowsCon.size() * sizeof(BlockJointRowCon));
    shim.memAlloc(&dJm, jointRowsMod.size() * sizeof(BlockJointRowMod));
    shim.memAlloc(&dBd, bodyData.size() * sizeof(SolverBodyData));
    shim.memAlloc(&dFpb, frictionPatchBlocks.size() * sizeof(BlockFrictionPatch));
    shim.memAlloc(&dFp, patches.size() * sizeof(FrictionPatchGPU));
    shim.memAlloc(&dTs, numAllBatches * 32 * sizeof(ThresholdElement));
    std::vector<ConstraintWriteback> wbInit(std::max(writebackCursor, 1u));
    std::memset(wbInit.data(), 0, wbInit.size() * sizeof(ConstraintWriteback));
    shim.memAlloc(&dWb, wbInit.size() * sizeof(ConstraintWriteback));
    std::vector<float> forceInit(std::max(forceBufCursor, 1u), 0.f);
    shim.memAlloc(&dForce, forceInit.size() * 4);
    if (!dBatch || !dCh || !dFh || !dCp || !dFr || !dJh || !dJc || !dJm || !dBd || !dFpb || !dFp ||
        !dTs || !dWb || !dForce)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    shim.memcpyHtoD(dBatch, batches.data(), batches.size() * sizeof(BlockConstraintBatch));
    shim.memcpyHtoD(dCh, contactHeaders.data(), contactHeaders.size() * sizeof(BlockContactHeader));
    shim.memcpyHtoD(dFh, frictionHeaders.data(), frictionHeaders.size() * sizeof(BlockFrictionHeader));
    shim.memcpyHtoD(dCp, contactPoints.data(), contactPoints.size() * sizeof(BlockContactPoint));
    shim.memcpyHtoD(dFr, frictions.data(), frictions.size() * sizeof(BlockContactFriction));
    shim.memcpyHtoD(dJh, jointHeaders.data(), jointHeaders.size() * sizeof(BlockJointHeader));
    shim.memcpyHtoD(dJc, jointRowsCon.data(), jointRowsCon.size() * sizeof(BlockJointRowCon));
    shim.memcpyHtoD(dJm, jointRowsMod.data(), jointRowsMod.size() * sizeof(BlockJointRowMod));
    shim.memcpyHtoD(dBd, bodyData.data(), bodyData.size() * sizeof(SolverBodyData));
    shim.memcpyHtoD(dFpb, frictionPatchBlocks.data(), frictionPatchBlocks.size() * sizeof(BlockFrictionPatch));
    shim.memcpyHtoD(dFp, patches.data(), patches.size() * sizeof(FrictionPatchGPU));
    shim.memcpyHtoD(dWb, wbInit.data(), wbInit.size() * sizeof(ConstraintWriteback));
    shim.memcpyHtoD(dForce, forceInit.data(), forceInit.size() * 4);

    physx::PxgIslandContext islands[2];
    std::memset(islands, 0, sizeof(islands));
    islands[0].mBatchStartIndex = 55u; // decoy
    islands[0].mBatchCount = 7u;       // decoy
    // the real island: batch range [1, 13) via the five-count sum 8+2+1+0+1 = 12
    islands[1].mBatchStartIndex = batchStartIndex;
    islands[1].mBatchCount = 8;
    islands[1].mArtiBatchCount = 2;
    islands[1].mStaticArtiBatchCount = 1;
    islands[1].mSelfArtiBatchCount = 0;
    islands[1].mStaticRigidBatchCount = 1;
    shim.memAlloc(&dIsl, sizeof(islands));
    shim.memcpyHtoD(dIsl, islands, sizeof(islands));

    physx::PxgSolverCoreDesc desc;
    std::memset(&desc, 0, sizeof(desc));
    desc.islandContextPool = reinterpret_cast<physx::PxgIslandContext*>(dIsl);
    desc.solverBodyDataPool = reinterpret_cast<physx::PxgSolverBodyData*>(dBd);
    desc.constraintWriteBack = reinterpret_cast<physx::PxgConstraintWriteback*>(dWb);
    desc.forceBuffer = reinterpret_cast<physx::PxF32*>(dForce);
    desc.frictionPatches = reinterpret_cast<physx::PxFrictionPatch*>(dFp);
    desc.thresholdStream = reinterpret_cast<physx::Dy::ThresholdStreamElement*>(dTs);
    desc.sharedThresholdStreamIndex = 0;

    physx::PxgSolverSharedDesc<physx::IterativeSolveData> shared;
    std::memset(&shared, 0, sizeof(shared));
    shared.blockCurrentFrictionPatches = reinterpret_cast<physx::PxgBlockFrictionPatch*>(dFpb);
    shared.iterativeData.blockConstraintBatch = reinterpret_cast<physx::PxgBlockConstraintBatch*>(dBatch);
    shared.iterativeData.blockContactHeaders = reinterpret_cast<physx::PxgBlockSolverContactHeader*>(dCh);
    shared.iterativeData.blockFrictionHeaders = reinterpret_cast<physx::PxgBlockSolverFrictionHeader*>(dFh);
    shared.iterativeData.blockContactPoints = reinterpret_cast<physx::PxgBlockSolverContactPoint*>(dCp);
    shared.iterativeData.blockFrictions = reinterpret_cast<physx::PxgBlockSolverContactFriction*>(dFr);
    shared.iterativeData.blockJointConstraintHeaders =
        reinterpret_cast<physx::PxgBlockSolverConstraint1DHeader*>(dJh);
    shared.iterativeData.blockJointConstraintRowsCon =
        reinterpret_cast<physx::PxgBlockSolverConstraint1DCon*>(dJc);
    shared.iterativeData.blockJointConstraintRowsMod =
        reinterpret_cast<physx::PxgBlockSolverConstraint1DMod*>(dJm);

    shim.memAlloc(&dDesc, sizeof(desc));
    shim.memcpyHtoD(dDesc, &desc, sizeof(desc));
    shim.memAlloc(&dShared, sizeof(shared));
    shim.memcpyHtoD(dShared, &shared, sizeof(shared));

    EngineBlob blob;
    blob.add<uint64_t>(dDesc);   // @0  solverCoreDesc
    blob.add<uint64_t>(dShared); // @8  sharedDesc
    blob.add<uint32_t>(1u);      // @16 islandIndex
    blob.finish();
    PXG_CHECK_MSG(blob.bytes.size() == 24, "conclude/writeback blob is 24 bytes (got %zu)",
                  blob.bytes.size());

    // engine grid arithmetic: nbBlocksRequired = ceil(12 * 32 / 256) = 2
    const uint32_t nbBlocks = (numRealBatches * 32 + 255) / 256;
    PXG_CHECK_MSG(shim.launchKernel(0, nbBlocks, 1, 1, blob.bytes.data(), uint32_t(blob.bytes.size())) ==
                      PxgResult::Success,
                  "launch concludeBlocks");
    PXG_CHECK_MSG(shim.launchKernel(1, nbBlocks, 1, 1, blob.bytes.data(), uint32_t(blob.bytes.size())) ==
                      PxgResult::Success,
                  "launch writebackBlocks");
    shim.streamSynchronize();

    // ---- reference ----
    auto refContacts = contactPoints;
    auto refFrictions = frictions;
    auto refMods = jointRowsMod;
    auto refPatchBlocks = frictionPatchBlocks;
    std::vector<float> refForce = forceInit;
    std::vector<ConstraintWriteback> refWb = wbInit;
    size_t refBrokenJoints = 0; // guard against a fixture that never breaks a joint
    std::vector<FrictionPatchGPU> refPatches = patches;
    std::vector<ThresholdElement> refStream;

    for (uint32_t k = batchStartIndex; k < numAllBatches; ++k)
    {
        const BlockConstraintBatch& b = batches[k];
        for (uint32_t lane = 0; lane < b.descStride; ++lane)
        {
            if (b.constraintType == 1) // engine encoding: eCONTACT = 1
            {
                // concludeContactBlock
                const uint32_t nc = contactHeaders[k].numNormalConstr[lane];
                const uint32_t nf = frictionHeaders[k].numFrictionConstr[lane];
                if (nc && contactHeaders[k].restitution[lane] >= 0.f)
                    for (uint32_t i = 0; i < nc; ++i)
                        refContacts[b.startConstraintIndex + i].coeff0[lane] =
                            refContacts[b.startConstraintIndex + i].coeff1[lane];
                for (uint32_t i = 0; i < nf; ++i)
                    refFrictions[b.startFrictionIndex + i].raXn_bias[lane].w = 0.f;

                // writeBackContactBlock
                float normalForce = 0.f;
                const uint32_t fwo = contactHeaders[k].forceWritebackOffset[lane];
                if (fwo != 0xFFFFFFFFu)
                    for (uint32_t i = 0; i < nc; ++i)
                    {
                        const float af = refContacts[b.startConstraintIndex + i].appliedForce[lane];
                        refForce[fwo + i] = af;
                        normalForce += af;
                    }

                const uint32_t patchIndex = refPatchBlocks[k].patchIndex[lane];
                if (patchIndex != 0xFFFFFFFFu)
                {
                    FrictionPatchGPU& fp = refPatches[patchIndex];
                    const Float4 a0 = frictionHeaders[k].frictionNormals0[lane];
                    const Float4 a1 = frictionHeaders[k].frictionNormals1[lane];
                    fp.anchors = nf / 2;
                    if (nf >= 2)
                    {
                        const Float4 an = refPatchBlocks[k].anchorPoints[0][lane];
                        fp.points[0] = Float3{ an.x, an.y, an.z };
                        const float i0 = refFrictions[b.startFrictionIndex].appliedForce[lane];
                        const float i1 = refFrictions[b.startFrictionIndex + 1].appliedForce[lane];
                        fp.impulses[0] = Float3{ a0.x * i0 + a1.x * i1, a0.y * i0 + a1.y * i1,
                                                 a0.z * i0 + a1.z * i1 };
                    }
                    if (nf >= 4)
                    {
                        const Float4 an = refPatchBlocks[k].anchorPoints[1][lane];
                        fp.points[1] = Float3{ an.x, an.y, an.z };
                        const float i0 = refFrictions[b.startFrictionIndex + 2].appliedForce[lane];
                        const float i1 = refFrictions[b.startFrictionIndex + 3].appliedForce[lane];
                        fp.impulses[1] = Float3{ a0.x * i0 + a1.x * i1, a0.y * i0 + a1.y * i1,
                                                 a0.z * i0 + a1.z * i1 };
                    }
                }
                if (nf && frictionHeaders[k].broken[lane])
                    refPatchBlocks[k].broken[lane] = 1;

                const float rt0 = bodyData[b.bodyAIndex[lane]].reportThreshold;
                const float rt1 = bodyData[b.bodyBIndex[lane]].reportThreshold;
                const bool forceThreshold = (contactHeaders[k].flags[lane] & 1u) != 0;
                if (forceThreshold && normalForce != 0.f && (rt0 < PX_MAX_REAL_F || rt1 < PX_MAX_REAL_F))
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
                // conclude1DBlock
                const uint32_t rows = jointHeaders[k].rowCounts[lane];
                for (uint32_t i = 0; i < rows; ++i)
                    if (!(refMods[b.startConstraintIndex + i].flags[lane] & DY_SC_FLAG_SPRING))
                        refMods[b.startConstraintIndex + i].coeff0[lane] =
                            refMods[b.startConstraintIndex + i].coeff1[lane];

                // writeBack1DBlock
                const uint32_t fwo = jointHeaders[k].writeBackOffset[lane];
                if (fwo != 0xFFFFFFFFu)
                {
                    Vec3 linVel(0.f), angVel(0.f);
                    float errSq = 0.f;
                    for (uint32_t i = 0; i < rows; ++i)
                    {
                        const BlockJointRowMod& m = refMods[b.startConstraintIndex + i];
                        if (m.flags[lane] & 2u)
                        {
                            const Float4 l0 = jointRowsCon[b.startConstraintIndex + i].lin0XYZ_minImpulse[lane];
                            linVel += Vec3(l0.x, l0.y, l0.z) * m.appliedForce[lane];
                            angVel += Vec3(m.ang0Writeback[lane * 3], m.ang0Writeback[lane * 3 + 1],
                                           m.ang0Writeback[lane * 3 + 2]) * m.appliedForce[lane];
                        }
                        errSq += m.residual[lane] * m.residual[lane];
                    }
                    const Float4 bwo = jointHeaders[k].body0WorldOffset_linBreakImpulse[lane];
                    angVel = angVel - Vec3(bwo.x, bwo.y, bwo.z).cross(linVel);
                    const uint32_t broken = jointHeaders[k].breakable[lane]
                        ? uint32_t(linVel.magnitude() > bwo.w ||
                                   angVel.magnitude() > jointHeaders[k].angBreakImpulse[lane])
                        : 0;
                    refWb[fwo].angularImpulse_residual = Float4{ angVel.x, angVel.y, angVel.z, errSq };
                    refWb[fwo].linearImpulse_broken = Float4{ linVel.x, linVel.y, linVel.z, broken ? -0.0f : 0.0f };
                    if (broken)
                        ++refBrokenJoints;
                }
            }
        }
    }

    // ---- compare ----
    std::vector<BlockContactPoint> gpuContacts(contactPoints.size());
    shim.memcpyDtoH(gpuContacts.data(), dCp, gpuContacts.size() * sizeof(BlockContactPoint));
    std::vector<BlockContactFriction> gpuFrictions(frictions.size());
    shim.memcpyDtoH(gpuFrictions.data(), dFr, gpuFrictions.size() * sizeof(BlockContactFriction));
    std::vector<BlockJointRowMod> gpuMods(jointRowsMod.size());
    shim.memcpyDtoH(gpuMods.data(), dJm, gpuMods.size() * sizeof(BlockJointRowMod));
    std::vector<float> gpuForce(refForce.size());
    shim.memcpyDtoH(gpuForce.data(), dForce, gpuForce.size() * 4);
    std::vector<ConstraintWriteback> gpuWb(refWb.size());
    shim.memcpyDtoH(gpuWb.data(), dWb, gpuWb.size() * sizeof(ConstraintWriteback));
    std::vector<FrictionPatchGPU> gpuPatches(patches.size());
    shim.memcpyDtoH(gpuPatches.data(), dFp, gpuPatches.size() * sizeof(FrictionPatchGPU));
    std::vector<BlockFrictionPatch> gpuPatchBlocks(frictionPatchBlocks.size());
    shim.memcpyDtoH(gpuPatchBlocks.data(), dFpb, gpuPatchBlocks.size() * sizeof(BlockFrictionPatch));
    physx::PxgSolverCoreDesc gpuDesc;
    shim.memcpyDtoH(&gpuDesc, dDesc, sizeof(gpuDesc));
    const uint32_t gpuTsCount = uint32_t(gpuDesc.sharedThresholdStreamIndex);
    std::vector<ThresholdElement> gpuStream(gpuTsCount);
    if (gpuTsCount)
        shim.memcpyDtoH(gpuStream.data(), dTs, gpuTsCount * sizeof(ThresholdElement));

    // COVERAGE: the broken flag is the SIGN BIT of a zero.  A platform that drops
    // signed zeros writes a record that is bit-perfect everywhere else and never
    // reports a broken joint, so the memcmp below can only see it if the fixture
    // actually breaks one.
    PXG_CHECK_MSG(refBrokenJoints > 0, "writeback fixture breaks no joint (the sign-bit path is untested)");

    size_t mismatches = 0;
    for (size_t i = 0; i < contactPoints.size(); ++i)
        for (int l = 0; l < 32; ++l)
            if (gpuContacts[i].coeff0[l] != refContacts[i].coeff0[l])
                ++mismatches;
    for (size_t i = 0; i < frictions.size(); ++i)
        for (int l = 0; l < 32; ++l)
            if (gpuFrictions[i].raXn_bias[l].w != refFrictions[i].raXn_bias[l].w)
                ++mismatches;
    for (size_t i = 0; i < jointRowsMod.size(); ++i)
        for (int l = 0; l < 32; ++l)
            if (gpuMods[i].coeff0[l] != refMods[i].coeff0[l])
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
    for (size_t i = 0; i < frictionPatchBlocks.size(); ++i)
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

    for (PxgDevicePtr pfree : { dBatch, dCh, dFh, dCp, dFr, dJh, dJc, dJm, dBd, dWb, dForce, dTs,
                                dFpb, dFp, dIsl, dDesc, dShared })
        shim.memFree(pfree);
}

// ---------------------------------------------------------------------------
// computeAverageSolverBodyVelocity + propagateSolverBodyVelocity +
// writeBackBodies (ENGINE ABI): the PGS 2-float4-stripe averaging over
// numSlabs in {4, 12, 40} (40 > 32 -> multi-word popcount + reset), including
// the CUDA quirk that only the .xyz deltas are butterfly-reduced while lane 0's
// LOCAL .w accumulation survives into the averaged store.  average/propagate
// read island [0] through descA; writeBackBodies reads islandIndex = 1 through
// descB (decoy at [0]) to pin the PxgIslandContext stride.
// ---------------------------------------------------------------------------
PXG_TEST(solverPGS_velocityAveraging)
{
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    std::mt19937 rng(3200);
    std::uniform_real_distribution<float> sym(-3.f, 3.f);

    for (uint32_t numSlabs : { 4u, 12u, 40u })
    {
        const uint32_t totalBodyCount = 70; // dynamic bodies
        const uint32_t bodyOffset = 8;
        const uint32_t T = totalBodyCount + bodyOffset;
        const uint32_t numTotalBodies = T; // no articulations
        const uint32_t numBatches = 6;
        const uint32_t averageOutputOffsets = numBatches * 32 * 2 * 2;
        const uint32_t slabRegion = ((totalBodyCount * numSlabs + 31) / 32) * 64 + 64;
        const uint32_t deltaVOffset = averageOutputOffsets + slabRegion;
        const uint32_t propBase = deltaVOffset + 2 * T;
        const uint32_t poolSize = propBase + slabRegion;

        std::vector<Float4> velPool(poolSize);
        for (Float4& v : velPool)
            v = Float4{ sym(rng), sym(rng), sym(rng), sym(rng) }; // random .w exercises the lane-0 .w path

        // remapped write slots: unique (r, r+32) pairs per slab slot in the
        // cluster-of-32 pattern, in a dedicated pool region (partitions keep
        // these race-free in the real solver)
        std::vector<uint32_t> references(totalBodyCount * numSlabs, 0xFFFFFFFFu);
        for (uint32_t i = 0; i < references.size(); ++i)
            if ((rng() & 3) != 0) // ~75% of slab slots active
                references[i] = propBase + 2 * (i & ~31u) + (i & 31u);

        const uint32_t num32Slabs = (numSlabs + 31) / 32;
        std::vector<uint32_t> encoded(numTotalBodies * num32Slabs);
        for (uint32_t& e : encoded)
            e = rng();

        std::vector<Float4> motionVel(2 * T);
        for (Float4& v : motionVel)
            v = Float4{ sym(rng), sym(rng), sym(rng), sym(rng) };

        PxgSlangComputeDevice* dev = testDevice();
        const PxgKernelDef manifest[] = {
            { "solverPGS.spv", "computeAverageSolverBodyVelocity", 16u, 0 },
            { "solverPGS.spv", "propagateSolverBodyVelocity", 16u, 0 },
            { "solverPGS.spv", "writeBackBodies", 24u, 0 },
        };
        PxgCudaShim shim(dev, manifest, 3, &solverPgsAbiLoader);

        PxgDevicePtr dVel = 0, dEnc = 0, dRef = 0, dMv = 0, dIslA = 0, dIslB = 0;
        PxgDevicePtr dDescA = 0, dDescB = 0, dShared = 0;
        shim.memAlloc(&dVel, velPool.size() * sizeof(Float4));
        shim.memAlloc(&dEnc, encoded.size() * 4);
        shim.memAlloc(&dRef, references.size() * 4);
        shim.memAlloc(&dMv, motionVel.size() * sizeof(Float4));
        if (!dVel || !dEnc || !dRef || !dMv)
        {
            std::printf("  NOTE: no BDA -> skip\n");
            return;
        }
        shim.memcpyHtoD(dVel, velPool.data(), velPool.size() * sizeof(Float4));
        shim.memcpyHtoD(dEnc, encoded.data(), encoded.size() * 4);
        shim.memcpyHtoD(dRef, references.data(), references.size() * 4);
        shim.memcpyHtoD(dMv, motionVel.data(), motionVel.size() * sizeof(Float4));

        // island arrays: average/propagate read island [0]; writeBackBodies
        // reads islandIndex = 1 (decoy at [0]) through descB.
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
        shim.memAlloc(&dIslA, sizeof(islands));
        shim.memcpyHtoD(dIslA, islands, sizeof(islands));
        shim.memAlloc(&dIslB, sizeof(islandsB));
        shim.memcpyHtoD(dIslB, islandsB, sizeof(islandsB));

        physx::PxgSolverCoreDesc desc;
        std::memset(&desc, 0, sizeof(desc));
        desc.islandContextPool = reinterpret_cast<physx::PxgIslandContext*>(dIslA);
        desc.solverBodyReferences = reinterpret_cast<physx::PxgSolverReferences*>(dRef);
        desc.motionVelocityArray = reinterpret_cast<float4*>(dMv);
        desc.numSlabs = numSlabs;
        desc.numBatches = numBatches;
        desc.numArticBatches = 0;
        desc.numSolverBodies = T;
        desc.accumulatedBodyDeltaVOffset = deltaVOffset;
        physx::PxgSolverCoreDesc descB = desc;
        descB.islandContextPool = reinterpret_cast<physx::PxgIslandContext*>(dIslB);

        physx::PxgSolverSharedDesc<physx::IterativeSolveData> shared;
        std::memset(&shared, 0, sizeof(shared));
        shared.deltaOutOffset = deltaVOffset; // writeBackBodies source offset
        shared.iterativeData.solverBodyVelPool = reinterpret_cast<float4*>(dVel);
        shared.iterativeData.solverEncodedReferenceCount = reinterpret_cast<physx::PxU32*>(dEnc);

        shim.memAlloc(&dDescA, sizeof(desc));
        shim.memcpyHtoD(dDescA, &desc, sizeof(desc));
        shim.memAlloc(&dDescB, sizeof(descB));
        shim.memcpyHtoD(dDescB, &descB, sizeof(descB));
        shim.memAlloc(&dShared, sizeof(shared));
        shim.memcpyHtoD(dShared, &shared, sizeof(shared));

        // engine grid arithmetic (COMPUTE_BODIES_AVERAGE_VELOCITY = 256):
        // average (solvePartitions): nbThreads = 32 * mBodyCount, block (32,8,1);
        // propagate (pos-iter site): nbThreads = 32 * mBodyCount, block 256 flat.
        const uint32_t nbBlocks = (32 * totalBodyCount + 255) / 256;

        struct { uint64_t desc, shared; } prm = { dDescA, dShared };
        PXG_CHECK_MSG(shim.launchKernel(0, nbBlocks, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                      "launch computeAverageSolverBodyVelocity");
        PXG_CHECK_MSG(shim.launchKernel(1, nbBlocks, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                      "launch propagateSolverBodyVelocity");

        // WRITE_BACK_BODIES = 256, 1 thread/body over [bodyOffset, T);
        // islandIndex = 1 (decoy at [0])
        EngineBlob wbBlob;
        wbBlob.add<uint64_t>(dDescB);
        wbBlob.add<uint64_t>(dShared);
        wbBlob.add<uint32_t>(1u); // islandIndex
        wbBlob.finish();
        PXG_CHECK_MSG(wbBlob.bytes.size() == 24, "writeBackBodies blob is 24 bytes (got %zu)",
                      wbBlob.bytes.size());
        PXG_CHECK_MSG(shim.launchKernel(2, (totalBodyCount + 255) / 256, 1, 1, wbBlob.bytes.data(),
                                        uint32_t(wbBlob.bytes.size())) == PxgResult::Success,
                      "launch writeBackBodies");
        shim.streamSynchronize();

        // ---- reference ----
        const uint32_t pow2 = [&] {
            uint32_t x = numSlabs;
            if ((x & (x - 1)) == 0)
                return x;
            x--; x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16;
            return x + 1;
        }();
        const uint32_t threadsPerBody = std::min(pow2, 32u);

        std::vector<Float4> refVel = velPool;
        std::vector<uint32_t> refEnc = encoded;

        for (uint32_t bodyId = 0; bodyId < totalBodyCount; ++bodyId)
        {
            const uint32_t outputBody = deltaVOffset + bodyId + bodyOffset;
            const Float4 lastLin = velPool[outputBody];
            const Float4 lastAng = velPool[outputBody + totalBodyCount + bodyOffset];
            Vec3 linDelta(0.f), angDelta(0.f);
            // lane 0 of the work unit covers slabs {0, tpb, 2*tpb, ...}; only
            // ITS local .w accumulation survives (the butterfly reduces xyz only)
            float wLin = 0.f, wAng = 0.f;
            bool hasRef = false;
            for (uint32_t b = 0; b < numSlabs; ++b)
            {
                const uint32_t bodyIndex = bodyId * numSlabs + b;
                if (references[bodyIndex] != 0xFFFFFFFFu)
                {
                    const uint32_t velIndex = averageOutputOffsets + 2 * (bodyIndex & ~31u) + (bodyIndex & 31u);
                    linDelta += Vec3(velPool[velIndex].x - lastLin.x, velPool[velIndex].y - lastLin.y,
                                     velPool[velIndex].z - lastLin.z);
                    angDelta += Vec3(velPool[velIndex + 32].x - lastAng.x, velPool[velIndex + 32].y - lastAng.y,
                                     velPool[velIndex + 32].z - lastAng.z);
                    if (b % threadsPerBody == 0) // lane 0's slab subset
                    {
                        wLin += velPool[velIndex].w - lastLin.w;
                        wAng += velPool[velIndex + 32].w - lastAng.w;
                    }
                    hasRef = true;
                }
            }
            if (hasRef)
            {
                uint32_t refCount = 0;
                for (uint32_t i = 0; i < num32Slabs; ++i)
                    refCount += refPopcount(encoded[bodyOffset + bodyId + i * numTotalBodies]);
                refCount = std::max(1u, refCount);
                const float recip = 1.f / float(refCount);
                linDelta *= recip;
                angDelta *= recip;
                wLin *= recip;
                wAng *= recip;
                for (uint32_t i = 0; i < num32Slabs; ++i)
                    refEnc[bodyOffset + bodyId + i * numTotalBodies] = 0;
                refVel[outputBody] = Float4{ lastLin.x + linDelta.x, lastLin.y + linDelta.y,
                                             lastLin.z + linDelta.z, lastLin.w + wLin };
                refVel[outputBody + totalBodyCount + bodyOffset] =
                    Float4{ lastAng.x + angDelta.x, lastAng.y + angDelta.y, lastAng.z + angDelta.z,
                            lastAng.w + wAng };
            }
        }

        // propagate pass (separate launch, after the averaging)
        for (uint32_t bodyId = 0; bodyId < totalBodyCount; ++bodyId)
        {
            const uint32_t outputBody = deltaVOffset + bodyId + bodyOffset;
            Float4 linVel = refVel[outputBody];
            const Float4 angVel = refVel[outputBody + totalBodyCount + bodyOffset];
            linVel.w = 0.f;
            for (uint32_t b = 0; b < numSlabs; ++b)
            {
                const uint32_t bodyIndex = bodyId * numSlabs + b;
                const uint32_t remapped = references[bodyIndex];
                if (remapped != 0xFFFFFFFFu)
                {
                    refVel[remapped] = linVel;
                    refVel[remapped + 32] = angVel;
                }
            }
        }

        // writeBackBodies: motionVel[a] / motionVel[a + T] for a in [bodyOffset, T)
        std::vector<Float4> refMotion = motionVel;
        for (uint32_t a = bodyOffset; a < T; ++a)
        {
            refMotion[a] = refVel[a + deltaVOffset];
            refMotion[a + T] = refVel[a + deltaVOffset + T];
        }

        // ---- compare ----
        std::vector<Float4> gpuVel(poolSize);
        shim.memcpyDtoH(gpuVel.data(), dVel, gpuVel.size() * sizeof(Float4));
        std::vector<uint32_t> gpuEnc(encoded.size());
        shim.memcpyDtoH(gpuEnc.data(), dEnc, gpuEnc.size() * 4);
        std::vector<Float4> gpuMotion(motionVel.size());
        shim.memcpyDtoH(gpuMotion.data(), dMv, gpuMotion.size() * sizeof(Float4));

        size_t mismatches = 0;
        for (size_t i = 0; i < poolSize; ++i)
            if (!almostEqual(gpuVel[i].x, refVel[i].x, 1e-4f, 1e-4f) ||
                !almostEqual(gpuVel[i].y, refVel[i].y, 1e-4f, 1e-4f) ||
                !almostEqual(gpuVel[i].z, refVel[i].z, 1e-4f, 1e-4f) ||
                !almostEqual(gpuVel[i].w, refVel[i].w, 1e-4f, 1e-4f))
            {
                if (mismatches < 3)
                    PXG_CHECK_MSG(false, "slabs=%u vel[%zu]: gpu=(%g,%g,%g,%g) ref=(%g,%g,%g,%g)", numSlabs,
                                  i, gpuVel[i].x, gpuVel[i].y, gpuVel[i].z, gpuVel[i].w, refVel[i].x,
                                  refVel[i].y, refVel[i].z, refVel[i].w);
                ++mismatches;
            }
        PXG_CHECK_MSG(mismatches == 0, "slabs=%u: %zu velocity mismatches", numSlabs, mismatches);

        mismatches = 0;
        for (size_t i = 0; i < encoded.size(); ++i)
            if (gpuEnc[i] != refEnc[i])
                ++mismatches;
        PXG_CHECK_MSG(mismatches == 0, "slabs=%u: %zu encoded-count mismatches", numSlabs, mismatches);

        mismatches = 0;
        for (size_t i = 0; i < motionVel.size(); ++i)
            if (!almostEqual(gpuMotion[i].x, refMotion[i].x, 1e-4f, 1e-4f) ||
                !almostEqual(gpuMotion[i].y, refMotion[i].y, 1e-4f, 1e-4f) ||
                !almostEqual(gpuMotion[i].z, refMotion[i].z, 1e-4f, 1e-4f) ||
                !almostEqual(gpuMotion[i].w, refMotion[i].w, 1e-4f, 1e-4f))
                ++mismatches;
        PXG_CHECK_MSG(mismatches == 0, "slabs=%u: %zu motion-velocity mismatches", numSlabs, mismatches);

        for (PxgDevicePtr pfree : { dVel, dEnc, dRef, dMv, dIslA, dIslB, dDescA, dDescB, dShared })
            shim.memFree(pfree);
    }
}

// ---------------------------------------------------------------------------
// markActiveSlabPGS (ENGINE ABI, P5 Step 5) — per-island active-slab marking
// for the mass-splitting reference counts.
// ---------------------------------------------------------------------------
// The read-only predicate that decides whether a contact still pushes.  Mirror
// of checkActiveContactBlock() in solverBlock.cuh (and of the Slang
// checkActiveContactBlockBda): a stripped PGS contact solve that (a) applies NO
// reference-count scaling — this is what COMPUTES the reference counts — (b)
// never mutates the input velocities between contact points, and (c) returns
// true on the first |deltaF| > 1e-8.  No friction; incl. the full
// computeContactCoefficients math (compliant-spring restitution < 0 branch).
bool refCheckActiveContact(const BlockConstraintBatch& b, const Vec3& lv0, const Vec3& av0,
                           const Vec3& lv1, const Vec3& av1, uint32_t lane,
                           std::vector<BlockContactHeader>& chs, std::vector<BlockContactPoint>& cps)
{
    BlockContactHeader& ch = chs[b.constraintBatchIndex];

    const uint32_t numNormal = ch.numNormalConstr[lane];

    const Float4 nsf = ch.normal_staticFriction[lane];
    const Vec3 normal(nsf.x, nsf.y, nsf.z);
    const float restitution = ch.restitution[lane];
    const uint32_t flags = ch.flags[lane];

    for (uint32_t i = 0; i < numNormal; ++i)
    {
        const BlockContactPoint& c = cps[b.startConstraintIndex + i];
        const Float4 ra = c.raXn_targetVelocity[lane];
        const Float4 rb = c.rbXn_maxImpulse[lane];
        const float appliedForce = c.appliedForce[lane];

        const float resp0 = c.resp0[lane], resp1 = c.resp1[lane]; // NO ref scaling
        const float unitResponse = resp0 + resp1;
        const float recipResponse = unitResponse > 0.f ? 1.f / unitResponse : 0.f;

        float velMultiplier, impulseMul, biasedErr;
        refContactCoefficients(flags, restitution, unitResponse, recipResponse, ra.w, c.coeff0[lane],
                               c.coeff1[lane], velMultiplier, impulseMul, biasedErr);

        const Vec3 raXn(ra.x, ra.y, ra.z), rbXn(rb.x, rb.y, rb.z);

        const float v0 = lv0.dot(normal) + av0.dot(raXn);
        const float v1 = lv1.dot(normal) + av1.dot(rbXn);
        const float normalVel = v0 - v1;

        const float tempDeltaF = biasedErr - normalVel * velMultiplier;
        const float _deltaF = std::fmax(tempDeltaF, -appliedForce);
        const float _newForce = appliedForce * impulseMul + _deltaF;
        const float newForce = std::fmin(_newForce, rb.w);
        const float deltaF = newForce - appliedForce;

        if (std::fabs(deltaF) > 1.0e-8f)
            return true;
    }
    return false;
}

// The engine launches markActiveSlabPGS ONCE per island with the LAST partition
// index; it marks every batch in [mBatchStartIndex, cpp[lastPartition +
// mStartPartitionIndex]).  For each batch lane < descStride it decides
// isActiveSlab (joint -> always active; contact -> checkActiveContactBlock) and
// atomicOr's (1 << (slabId % 32)) into solverEncodedReferenceCount[(slabId /
// 32) * numTotalBodies + bodyId] for bodyA and bodyB, only when that body index
// is a real dynamic body (>= bodyOffset).  Body velocities come from the PGS
// TWO-float4-stripe accumulated layout (lin @ outputOffset + bodyId, ang @
// + totalBodiesIncKinematics) with a NONZERO accumulatedBodyDeltaVOffset to pin
// the offset add.  The fixture fills the REAL engine descs (+ a 2-element
// PxgIslandContext array; islandIndex = 1 pins the stride while element [0]
// carries decoy partition/batch fields but the SAME body fields — the .cu reads
// mBodyCount / mArticulationCount from islandContextPool[0]), one partition
// holding a mix of contact and joint batches (one contact deliberately has
// numNormalConstr == 0 so the inactive early-return AND the .cu's unconditional
// contacts[0] prefetch are exercised), pre-zeroes the encoded buffer on device,
// launches through PxgCudaShim with the 32-byte blob AND gridY = 2 (the
// articulation groups are a documented no-op), and compares the final encoded
// words BIT-EXACT (popcount-of-OR is order-independent) against
// refCheckActiveContact.
PXG_TEST(solverPGS_markActiveSlab)
{
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    std::mt19937 rng(5200);
    std::uniform_real_distribution<float> unit(0.f, 1.f);
    std::uniform_real_distribution<float> sym(-2.f, 2.f);

    const uint32_t bodyOffset = 8;
    const uint32_t numDynamicBodies = 200;
    const uint32_t numTotalBodies = bodyOffset + numDynamicBodies; // no articulations
    const uint32_t totalBodiesIncKinematics = numDynamicBodies + bodyOffset;
    const uint32_t numSlabs = 40;                   // > 32 -> 2-word slabIndexOffset
    const uint32_t num32Slabs = (numSlabs + 31) / 32;
    const uint32_t rowsPerBatch = 4;
    const uint32_t outputOffset = 16; // accumulatedBodyDeltaVOffset (nonzero pins the add)

    // batches: [0] decoy (never marked), then a mix over one partition.
    // mBatchStartIndex = 1, mStartPartitionIndex = 1, lastPartition = 0 =>
    // endIndex = cpp[0 + 1] = cpp[1] = 5, so the launch spans batches [1, 5).
    const uint32_t numAllBatches = 5;
    const uint32_t batchStartIndex = 1;
    const uint16_t types[5] = { 0, 1, 0, 1, 1 };     // [1]contact [2]joint [3]contact(empty) [4]contact
    const uint16_t strides[5] = { 0, 32, 27, 32, 20 }; // partial strides gate tail lanes
    std::vector<uint32_t> cpp = { 777u, 5u };        // index 0 decoy; endIndex = cpp[1] = 5

    std::vector<BlockConstraintBatch> batches(numAllBatches);
    std::vector<BlockContactHeader> chs(numAllBatches);
    std::vector<BlockContactPoint> cps(numAllBatches * rowsPerBatch);
    std::memset(batches.data(), 0, batches.size() * sizeof(BlockConstraintBatch));
    std::memset(chs.data(), 0, chs.size() * sizeof(BlockContactHeader));
    std::memset(cps.data(), 0, cps.size() * sizeof(BlockContactPoint));

    // PGS 2-float4 stripes per body, offset by outputOffset
    const uint32_t poolSize = outputOffset + 2 * totalBodiesIncKinematics;
    std::vector<Float4> velPool(poolSize);
    for (Float4& v : velPool)
        v = Float4{ sym(rng), sym(rng), sym(rng), sym(rng) };

    for (uint32_t k = batchStartIndex; k < numAllBatches; ++k)
    {
        BlockConstraintBatch& b = batches[k];
        b.descStride = strides[k];
        b.constraintType = types[k];
        b.constraintBatchIndex = k;
        b.startConstraintIndex = k * rowsPerBatch;

        // batch [3] is a contact with NO normal constraints -> always inactive
        // (and still prefetches contacts[startConstraintIndex], like the .cu).
        const bool emptyContact = (k == 3);

        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            b.slabId[lane] = rng() % numSlabs;
            b.bodyAIndex[lane] = rng() % numTotalBodies; // some < bodyOffset (kinematic)
            b.bodyBIndex[lane] = rng() % numTotalBodies;

            if (b.constraintType == 1) // eCONTACT
            {
                BlockContactHeader& h = chs[k];
                h.invMass0_1_angDom0_1[lane] = Float4{ unit(rng), unit(rng), unit(rng), unit(rng) };
                const Vec3 n = randUnit(rng);
                h.normal_staticFriction[lane] = Float4{ n.x, n.y, n.z, unit(rng) };
                h.flags[lane] = unit(rng) < 0.5f ? COMPLIANT_ACCEL_SPRING : 0;
                h.numNormalConstr[lane] = emptyContact ? 0u : (1u + rng() % rowsPerBatch);
                // mix rigid (>= 0) and compliant (< 0) restitution
                h.restitution[lane] = unit(rng) < 0.5f ? unit(rng) * 0.8f : -unit(rng) * 100.f;

                for (uint32_t i = 0; i < rowsPerBatch; ++i)
                {
                    BlockContactPoint& c = cps[b.startConstraintIndex + i];
                    const Vec3 ra = Vec3(sym(rng), sym(rng), sym(rng)) * 0.3f;
                    const Vec3 rb = Vec3(sym(rng), sym(rng), sym(rng)) * 0.3f;
                    c.raXn_targetVelocity[lane] = Float4{ ra.x, ra.y, ra.z, sym(rng) * 0.1f };
                    c.rbXn_maxImpulse[lane] = Float4{ rb.x, rb.y, rb.z, 10.f + unit(rng) * 100.f };
                    // some zero applied forces widen the naturally-inactive band
                    c.appliedForce[lane] = unit(rng) < 0.3f ? 0.f : unit(rng);
                    c.resp0[lane] = 0.1f + unit(rng);
                    c.resp1[lane] = 0.1f + unit(rng);
                    c.coeff0[lane] = unit(rng) * 0.2f;
                    c.coeff1[lane] = unit(rng) * 0.2f;
                }
            }
            // joint batches need no header data: markActiveSlab marks joints
            // active unconditionally.
        }
    }

    // ---- reference: OR from a zero-initialised encoded buffer ----
    auto bodyVel = [&](uint32_t bodyId, Vec3& lv, Vec3& av) {
        const uint32_t fi = outputOffset + bodyId;
        const Float4 l = velPool[fi];
        const Float4 a = velPool[fi + totalBodiesIncKinematics];
        lv = Vec3(l.x, l.y, l.z);
        av = Vec3(a.x, a.y, a.z);
    };

    const uint32_t encSize = num32Slabs * numTotalBodies;
    std::vector<uint32_t> refEnc(encSize, 0);
    uint32_t contactActive = 0, contactInactive = 0, jointMarks = 0;
    for (uint32_t k = batchStartIndex; k < cpp[1]; ++k)
    {
        const BlockConstraintBatch& b = batches[k];
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
                Vec3 lv0, av0, lv1, av1;
                bodyVel(bodyIdA, lv0, av0);
                bodyVel(bodyIdB, lv1, av1);
                active = refCheckActiveContact(b, lv0, av0, lv1, av1, lane, chs, cps);
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
        { "solverPGS.spv", "markActiveSlabPGS", 32u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &solverPgsAbiLoader);

    PxgDevicePtr dVel = 0, dEnc = 0, dCpp = 0, dIsl = 0, dBatch = 0, dCh = 0, dCp = 0;
    PxgDevicePtr dDesc = 0, dShared = 0;
    shim.memAlloc(&dVel, velPool.size() * sizeof(Float4));
    shim.memAlloc(&dEnc, encSize * 4);
    shim.memAlloc(&dCpp, cpp.size() * 4);
    shim.memAlloc(&dBatch, batches.size() * sizeof(BlockConstraintBatch));
    shim.memAlloc(&dCh, chs.size() * sizeof(BlockContactHeader));
    shim.memAlloc(&dCp, cps.size() * sizeof(BlockContactPoint));
    if (!dVel || !dEnc || !dCpp || !dBatch || !dCh || !dCp)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    std::vector<uint32_t> encInit(encSize, 0); // engine memsets this to 0 first
    shim.memcpyHtoD(dVel, velPool.data(), velPool.size() * sizeof(Float4));
    shim.memcpyHtoD(dEnc, encInit.data(), encSize * 4);
    shim.memcpyHtoD(dCpp, cpp.data(), cpp.size() * 4);
    shim.memcpyHtoD(dBatch, batches.data(), batches.size() * sizeof(BlockConstraintBatch));
    shim.memcpyHtoD(dCh, chs.data(), chs.size() * sizeof(BlockContactHeader));
    shim.memcpyHtoD(dCp, cps.data(), cps.size() * sizeof(BlockContactPoint));

    physx::PxgIslandContext islands[2];
    std::memset(islands, 0, sizeof(islands));
    // element [0]: SAME body-count fields (the .cu reads mBodyCount /
    // mArticulationCount from islandContextPool[0]) but DECOY partition/batch
    // fields — a bug that read them via islandIndex would corrupt the batch
    // range and mismatch.
    islands[0].mBodyStartIndex = bodyOffset;
    islands[0].mBodyCount = numDynamicBodies;
    islands[0].mArticulationCount = 0;
    islands[0].mStartPartitionIndex = 5u;  // decoy
    islands[0].mNumPartitions = 9u;        // decoy
    islands[0].mBatchStartIndex = 42u;     // decoy
    // element [1] = islandIndex: the real island-indexed fields (incl.
    // mBodyStartIndex — the .cu reads bodyOffset from island[islandIndex]).
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

    physx::PxgSolverSharedDesc<physx::IterativeSolveData> shared;
    std::memset(&shared, 0, sizeof(shared));
    shared.iterativeData.solverBodyVelPool = reinterpret_cast<float4*>(dVel);
    shared.iterativeData.solverEncodedReferenceCount = reinterpret_cast<physx::PxU32*>(dEnc);
    shared.iterativeData.blockConstraintBatch = reinterpret_cast<physx::PxgBlockConstraintBatch*>(dBatch);
    shared.iterativeData.blockContactHeaders = reinterpret_cast<physx::PxgBlockSolverContactHeader*>(dCh);
    shared.iterativeData.blockContactPoints = reinterpret_cast<physx::PxgBlockSolverContactPoint*>(dCp);

    shim.memAlloc(&dDesc, sizeof(desc));
    shim.memcpyHtoD(dDesc, &desc, sizeof(desc));
    shim.memAlloc(&dShared, sizeof(shared));
    shim.memcpyHtoD(dShared, &shared, sizeof(shared));

    EngineBlob blob;
    blob.add<uint64_t>(dDesc);   // @0  solverCoreDesc
    blob.add<uint64_t>(dShared); // @8  sharedDesc
    blob.add<uint32_t>(1u);      // @16 islandIndex
    blob.add<uint32_t>(0u);      // @20 lastPartition
    blob.add<uint64_t>(0u);      // @24 artiDesc (deferred; unused on the rigid path)
    blob.finish();
    PXG_CHECK_MSG(blob.bytes.size() == 32, "markActiveSlabPGS blob is 32 bytes (got %zu)",
                  blob.bytes.size());

    // grid (maxBlocks, blockY, 1); block (32, 2, 1) -> 2 warps/block.
    // maxBlocks = ceil(batchesInPartition / numWarpsPerBlock) = ceil(4/2) = 2.
    // gridY = 2 exercises the DEFERRED articulation groups (groupId.y != 0),
    // which must be a clean no-op — a marking bug there would corrupt the
    // bit-exact comparison below.
    PXG_CHECK_MSG(shim.launchKernel(0, 2, 2, 1, blob.bytes.data(), uint32_t(blob.bytes.size())) ==
                      PxgResult::Success,
                  "launch markActiveSlabPGS");
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
    PXG_CHECK_MSG(mismatches == 0, "markActiveSlabPGS: %zu encoded-word mismatches", mismatches);

    for (PxgDevicePtr pfree : { dVel, dEnc, dCpp, dBatch, dCh, dCp, dIsl, dDesc, dShared })
        shim.memFree(pfree);
}

// ---------------------------------------------------------------------------
// solveStaticBlock + propagateStaticSolverBodyVelocities (ENGINE ABI)
// ---------------------------------------------------------------------------
// The dynamic-vs-static rigid contact/joint solve, mirroring the proven TGS
// twin test (solverTGS_staticContactSolve) on the PGS layouts.  Two dynamic
// bodies (lanes 0 and 1 of shared static batches, exercising
// warpScanExclusive), each with 3 static contact batches split across 2 slabs
// (maxStaticPartitions = 2) plus a shared static 1D joint batch.
// solveStaticBlock walks the batches with ref0 = ref1 = 1 (no mass splitting
// against statics) and writes per-slab body velocities (PGS TWO-float4-stripe
// layout, .w preserved from the loaded value) to tempStaticBodyOutputPool;
// propagateStaticSolverBodyVelocities averages them back into the deltaV
// pool.  Both are checked against the proven refSolveContact / refSolve1D
// transcriptions, and the contact-residual accumulator is checked.
PXG_TEST(solverPGS_staticContactSolve)
{
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    std::mt19937 rng(3950);
    std::uniform_real_distribution<float> unit(0.f, 1.f);
    std::uniform_real_distribution<float> sym(-2.f, 2.f);

    const uint32_t numDynamicBodies = 2;
    const uint32_t bodyOffset = 1;
    const uint32_t T = numDynamicBodies + bodyOffset; // totalBodiesIncKinematics = 3
    const uint32_t nbStaticSlabs = 2;
    const uint32_t maxStaticPartitions = 2;
    const uint32_t rowsPerBatch = 4;
    const uint32_t deltaVOffset = 100;
    const uint32_t poolSize = 200;

    // batch 0 = decoy; 1,2,3 = contacts (eCONTACT=1); 4 = joint (eCONSTRAINT_1D=0)
    const uint32_t numAllBatches = 5;
    const uint32_t contactStart = 1; // shared by both bodies
    const uint32_t jointStart = 4;

    std::vector<BlockConstraintBatch> batches(numAllBatches);
    std::vector<BlockContactHeader> chs(3);
    std::vector<BlockFrictionHeader> fhs(3);
    std::vector<BlockContactPoint> cps(3 * rowsPerBatch);
    std::vector<BlockContactFriction> frs(3 * 2);
    std::vector<BlockJointHeader> jhs(1);
    std::vector<BlockJointRowCon> jcons(rowsPerBatch);
    std::vector<BlockJointRowMod> jmods(rowsPerBatch);
    std::memset(batches.data(), 0, batches.size() * sizeof(BlockConstraintBatch));
    std::memset(chs.data(), 0, chs.size() * sizeof(BlockContactHeader));
    std::memset(fhs.data(), 0, fhs.size() * sizeof(BlockFrictionHeader));
    std::memset(cps.data(), 0, cps.size() * sizeof(BlockContactPoint));
    std::memset(frs.data(), 0, frs.size() * sizeof(BlockContactFriction));
    std::memset(jhs.data(), 0, jhs.size() * sizeof(BlockJointHeader));
    std::memset(jcons.data(), 0, jcons.size() * sizeof(BlockJointRowCon));
    std::memset(jmods.data(), 0, jmods.size() * sizeof(BlockJointRowMod));

    std::vector<Float4> velPool(poolSize);
    for (Float4& v : velPool)
        v = Float4{ sym(rng), sym(rng), sym(rng), sym(rng) };

    // ---- contact / joint batch data (lanes 0 and 1 = bodies 0 and 1;
    // body B is the static: invMassB = angDom1 = resp1 = 0) ----
    auto fillContactLane = [&](uint32_t header, uint32_t lane) {
        BlockContactHeader& h = chs[header];
        h.invMass0_1_angDom0_1[lane] = Float4{ unit(rng), 0.f, unit(rng), 0.f };
        const Vec3 n = randUnit(rng);
        h.normal_staticFriction[lane] = Float4{ n.x, n.y, n.z, unit(rng) };
        h.flags[lane] = unit(rng) < 0.5f ? COMPLIANT_ACCEL_SPRING : 0;
        h.numNormalConstr[lane] = 1 + rng() % rowsPerBatch; // >=1 so the solve does work
        // mix rigid (>= 0) and compliant (< 0) restitution
        h.restitution[lane] = unit(rng) < 0.5f ? unit(rng) * 0.8f : -unit(rng) * 100.f;

        BlockFrictionHeader& fh = fhs[header];
        fh.numFrictionConstr[lane] = rng() % 3; // 0..2
        fh.dynamicFriction[lane] = unit(rng);
        const Vec3 f0 = randUnit(rng), f1 = randUnit(rng);
        fh.frictionNormals0[lane] = Float4{ f0.x, f0.y, f0.z, 0 };
        fh.frictionNormals1[lane] = Float4{ f1.x, f1.y, f1.z, 0 };

        for (uint32_t i = 0; i < rowsPerBatch; ++i)
        {
            BlockContactPoint& c = cps[header * rowsPerBatch + i];
            const Vec3 ra = Vec3(sym(rng), sym(rng), sym(rng)) * 0.3f;
            const Vec3 rb = Vec3(sym(rng), sym(rng), sym(rng)) * 0.3f;
            c.raXn_targetVelocity[lane] = Float4{ ra.x, ra.y, ra.z, sym(rng) * 0.1f };
            c.rbXn_maxImpulse[lane] = Float4{ rb.x, rb.y, rb.z, 10.f + unit(rng) * 100.f };
            c.appliedForce[lane] = unit(rng);
            c.resp0[lane] = 0.1f + unit(rng);
            c.resp1[lane] = 0.f; // static B
            c.coeff0[lane] = unit(rng) * 0.2f;
            c.coeff1[lane] = unit(rng) * 0.2f;
        }
        for (uint32_t i = 0; i < 2; ++i)
        {
            BlockContactFriction& f = frs[header * 2 + i];
            const Vec3 ra = Vec3(sym(rng), sym(rng), sym(rng)) * 0.3f;
            const Vec3 rb = Vec3(sym(rng), sym(rng), sym(rng)) * 0.3f;
            f.raXn_bias[lane] = Float4{ ra.x, ra.y, ra.z, sym(rng) * 0.05f };
            f.rbXn_targetVelW[lane] = Float4{ rb.x, rb.y, rb.z, sym(rng) * 0.05f };
            f.appliedForce[lane] = sym(rng) * 0.2f;
            f.resp0[lane] = 0.1f + unit(rng);
            f.resp1[lane] = 0.f; // static B
        }
    };
    auto fillJointLane = [&](uint32_t lane) {
        BlockJointHeader& h = jhs[0];
        h.rowCounts[lane] = uint16_t(1 + rng() % rowsPerBatch);
        h.invMass0D0[lane] = unit(rng);
        h.invMass1D1[lane] = 0.f; // static B
        h.invInertiaScale0[lane] = unit(rng);
        h.invInertiaScale1[lane] = 0.f; // static B
        for (uint32_t i = 0; i < rowsPerBatch; ++i)
        {
            BlockJointRowCon& c = jcons[i];
            BlockJointRowMod& m = jmods[i];
            const Vec3 l0 = randUnit(rng), l1 = randUnit(rng);
            const Vec3 a0 = Vec3(sym(rng), sym(rng), sym(rng)) * 0.4f;
            const Vec3 a1 = Vec3(sym(rng), sym(rng), sym(rng)) * 0.4f;
            c.lin0XYZ_minImpulse[lane] = Float4{ l0.x, l0.y, l0.z, -50.f - unit(rng) * 50.f };
            c.lin1XYZ_maxImpulse[lane] = Float4{ l1.x, l1.y, l1.z, 50.f + unit(rng) * 50.f };
            c.ang0XYZ_resp0[lane] = Float4{ a0.x, a0.y, a0.z, 0.1f + unit(rng) };
            c.ang1XYZ_resp1[lane] = Float4{ a1.x, a1.y, a1.z, 0.f }; // static B resp1 = 0
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
    };

    for (uint32_t k = 1; k <= 3; ++k) // contact batches
    {
        BlockConstraintBatch& b = batches[k];
        b.descStride = 2;
        b.constraintType = 1; // eCONTACT
        b.constraintBatchIndex = k - 1;         // -> chs/fhs index
        b.startConstraintIndex = (k - 1) * rowsPerBatch; // -> cps
        b.startFrictionIndex = (k - 1) * 2;              // -> frs
        b.mask = 0x3u; // lanes 0,1 active
        fillContactLane(k - 1, 0);
        fillContactLane(k - 1, 1);
    }
    {
        BlockConstraintBatch& b = batches[4]; // joint batch
        b.descStride = 2;
        b.constraintType = 0; // eCONSTRAINT_1D
        b.constraintBatchIndex = 0; // jhs index
        b.startConstraintIndex = 0; // jcons/jmods start
        b.mask = 0x3u;
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
        { "solverPGS.spv", "solveStaticBlock", 32u, 0 },
        { "solverPGS.spv", "propagateStaticSolverBodyVelocities", 32u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 2, &solverPgsAbiLoader);

    const uint32_t bodyOutCount = nbStaticSlabs * T * 2 + T; // slabIdx*T*2 + bodyIndex + T max
    std::vector<Float4> bodyOutInit(bodyOutCount, Float4{ 9.f, 9.f, 9.f, 9.f }); // sentinel

    PxgDevicePtr dVel = 0, dBodyOut = 0, dIsl = 0, dCC = 0, dJC = 0, dCS = 0, dJS = 0;
    PxgDevicePtr dBatch = 0, dCh = 0, dFh = 0, dCp = 0, dFr = 0, dJh = 0, dJc = 0, dJm = 0;
    PxgDevicePtr dDesc = 0, dShared = 0;
    shim.memAlloc(&dVel, velPool.size() * sizeof(Float4));
    shim.memAlloc(&dBodyOut, bodyOutInit.size() * sizeof(Float4));
    shim.memAlloc(&dCC, contactCounts.size() * 4);
    shim.memAlloc(&dJC, jointCounts.size() * 4);
    shim.memAlloc(&dCS, contactStarts.size() * 4);
    shim.memAlloc(&dJS, jointStarts.size() * 4);
    shim.memAlloc(&dBatch, batches.size() * sizeof(BlockConstraintBatch));
    shim.memAlloc(&dCh, chs.size() * sizeof(BlockContactHeader));
    shim.memAlloc(&dFh, fhs.size() * sizeof(BlockFrictionHeader));
    shim.memAlloc(&dCp, cps.size() * sizeof(BlockContactPoint));
    shim.memAlloc(&dFr, frs.size() * sizeof(BlockContactFriction));
    shim.memAlloc(&dJh, jhs.size() * sizeof(BlockJointHeader));
    shim.memAlloc(&dJc, jcons.size() * sizeof(BlockJointRowCon));
    shim.memAlloc(&dJm, jmods.size() * sizeof(BlockJointRowMod));
    if (!dVel || !dBodyOut || !dCC || !dJC || !dCS || !dJS || !dBatch || !dCh || !dFh || !dCp || !dFr ||
        !dJh || !dJc || !dJm)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    shim.memcpyHtoD(dVel, velPool.data(), velPool.size() * sizeof(Float4));
    shim.memcpyHtoD(dBodyOut, bodyOutInit.data(), bodyOutInit.size() * sizeof(Float4));
    shim.memcpyHtoD(dCC, contactCounts.data(), contactCounts.size() * 4);
    shim.memcpyHtoD(dJC, jointCounts.data(), jointCounts.size() * 4);
    shim.memcpyHtoD(dCS, contactStarts.data(), contactStarts.size() * 4);
    shim.memcpyHtoD(dJS, jointStarts.data(), jointStarts.size() * 4);
    shim.memcpyHtoD(dBatch, batches.data(), batches.size() * sizeof(BlockConstraintBatch));
    shim.memcpyHtoD(dCh, chs.data(), chs.size() * sizeof(BlockContactHeader));
    shim.memcpyHtoD(dFh, fhs.data(), fhs.size() * sizeof(BlockFrictionHeader));
    shim.memcpyHtoD(dCp, cps.data(), cps.size() * sizeof(BlockContactPoint));
    shim.memcpyHtoD(dFr, frs.data(), frs.size() * sizeof(BlockContactFriction));
    shim.memcpyHtoD(dJh, jhs.data(), jhs.size() * sizeof(BlockJointHeader));
    shim.memcpyHtoD(dJc, jcons.data(), jcons.size() * sizeof(BlockJointRowCon));
    shim.memcpyHtoD(dJm, jmods.data(), jmods.size() * sizeof(BlockJointRowMod));

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
    desc.accumulatedBodyDeltaVOffset = deltaVOffset;
    desc.mRigidStaticContactCounts = reinterpret_cast<physx::PxU32*>(dCC);
    desc.mRigidStaticJointCounts = reinterpret_cast<physx::PxU32*>(dJC);
    desc.mRigidStaticContactStartIndices = reinterpret_cast<physx::PxU32*>(dCS);
    desc.mRigidStaticJointStartIndices = reinterpret_cast<physx::PxU32*>(dJS);
    desc.contactErrorAccumulator.mErrorSumOfSquares = 0.5f;
    desc.contactErrorAccumulator.mCounter = 0; // >= 0: residual accumulation ON
    desc.contactErrorAccumulator.mMaxError = 0.0625f;

    physx::PxgSolverSharedDesc<physx::IterativeSolveData> shared;
    std::memset(&shared, 0, sizeof(shared));
    shared.iterativeData.solverBodyVelPool = reinterpret_cast<float4*>(dVel);
    shared.iterativeData.tempStaticBodyOutputPool = reinterpret_cast<float4*>(dBodyOut);
    shared.iterativeData.blockConstraintBatch = reinterpret_cast<physx::PxgBlockConstraintBatch*>(dBatch);
    shared.iterativeData.blockContactHeaders = reinterpret_cast<physx::PxgBlockSolverContactHeader*>(dCh);
    shared.iterativeData.blockFrictionHeaders = reinterpret_cast<physx::PxgBlockSolverFrictionHeader*>(dFh);
    shared.iterativeData.blockContactPoints = reinterpret_cast<physx::PxgBlockSolverContactPoint*>(dCp);
    shared.iterativeData.blockFrictions = reinterpret_cast<physx::PxgBlockSolverContactFriction*>(dFr);
    shared.iterativeData.blockJointConstraintHeaders =
        reinterpret_cast<physx::PxgBlockSolverConstraint1DHeader*>(dJh);
    shared.iterativeData.blockJointConstraintRowsCon =
        reinterpret_cast<physx::PxgBlockSolverConstraint1DCon*>(dJc);
    shared.iterativeData.blockJointConstraintRowsMod =
        reinterpret_cast<physx::PxgBlockSolverConstraint1DMod*>(dJm);

    shim.memAlloc(&dDesc, sizeof(desc));
    shim.memcpyHtoD(dDesc, &desc, sizeof(desc));
    shim.memAlloc(&dShared, sizeof(shared));
    shim.memcpyHtoD(dShared, &shared, sizeof(shared));

    // the engine's exact 6-entry staticKernelParams pack, used verbatim for
    // BOTH kernels (the propagate kernel's 5-arg signature treats the trailing
    // doFriction byte as tail padding): 32-byte blob.
    auto makeBlob = [&]() {
        EngineBlob blob;
        blob.add<uint64_t>(dDesc);
        blob.add<uint64_t>(dShared);
        blob.add<uint32_t>(1u);                   // islandIndex
        blob.add<uint32_t>(nbStaticSlabs);
        blob.add<uint32_t>(maxStaticPartitions);
        blob.add<uint8_t>(1u);                    // doFriction
        blob.finish();
        return blob;
    };
    EngineBlob solveBlob = makeBlob();
    PXG_CHECK_MSG(solveBlob.bytes.size() == 32, "static-solve blob is 32 bytes (got %zu)",
                  solveBlob.bytes.size());

    // nbBlocksRequired = ceil(bodyCount / SOLVE_BLOCK_PARTITION = 64) * nbStaticSlabs
    const uint32_t nbBlocksRequired = ((numDynamicBodies + 63) / 64) * nbStaticSlabs;
    PXG_CHECK_MSG(shim.launchKernel(0, nbBlocksRequired, 1, 1, solveBlob.bytes.data(),
                                    uint32_t(solveBlob.bytes.size())) == PxgResult::Success,
                  "launch solveStaticBlock");
    shim.streamSynchronize();

    EngineBlob propBlob = makeBlob();
    PXG_CHECK_MSG(shim.launchKernel(1, nbBlocksRequired, 1, 1, propBlob.bytes.data(),
                                    uint32_t(propBlob.bytes.size())) == PxgResult::Success,
                  "launch propagateStaticSolverBodyVelocities");
    shim.streamSynchronize();

    // ---- reference ----
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
        // PGS 2-stripe layout: linVel @ inputBody, angVel @ + T
        const Float4 vLin = velPool[inputBody], vAng = velPool[inputBody + T];
        Vec3 lv0(vLin.x, vLin.y, vLin.z), av0(vAng.x, vAng.y, vAng.z);
        Vec3 lv1(0.f), av1(0.f);

        const uint32_t startContactIndex = contactStarts[bodyIndex] + startIndex;
        const uint32_t startJointIndex = jointStarts[bodyIndex] + startIndex;

        for (uint32_t i = 0; i < jointCount; ++i)
            refSolve1D(batches[startJointIndex + i], lv0, av0, lv1, av1, idx, true, jhs, jcons, jmods,
                       1.f, 1.f);
        for (uint32_t i = 0; i < contactCount; ++i)
            refSolveContact(batches[startContactIndex + i], lv0, av0, lv1, av1, true, idx, chs, fhs, cps,
                            frs, 1.f, 1.f, &errAcc);

        const uint32_t outputBody = slabIdx * T * 2;
        // xyz overwritten, .w preserved from the loaded value (mirrors the .cu)
        refBodyOut[bodyIndex + outputBody] = Float4{ lv0.x, lv0.y, lv0.z, vLin.w };
        refBodyOut[bodyIndex + outputBody + T] = Float4{ av0.x, av0.y, av0.z, vAng.w };
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
    std::vector<BlockContactPoint> gpuCps(cps.size());
    shim.memcpyDtoH(gpuCps.data(), dCp, gpuCps.size() * sizeof(BlockContactPoint));
    std::vector<BlockContactFriction> gpuFrs(frs.size());
    shim.memcpyDtoH(gpuFrs.data(), dFr, gpuFrs.size() * sizeof(BlockContactFriction));
    std::vector<BlockJointRowMod> gpuJmods(jmods.size());
    shim.memcpyDtoH(gpuJmods.data(), dJm, gpuJmods.size() * sizeof(BlockJointRowMod));

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
                    PXG_CHECK_MSG(false, "tempStaticBodyOutputPool[%zu].%d: gpu=%g ref=%g", i, c, g[c], r[c]);
                ++mismatches;
            }
    }
    PXG_CHECK_MSG(anyWritten, "solveStaticBlock wrote no outputs (no-op / unbound launch?)");
    PXG_CHECK_MSG(mismatches == 0, "static solve: %zu tempStaticBodyOutputPool mismatches", mismatches);

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
    for (size_t i = 0; i < jmods.size(); ++i)
        for (int l = 0; l < 2; ++l)
        {
            if (!almostEqual(gpuJmods[i].appliedForce[l], jmods[i].appliedForce[l], 2e-4f, 2e-4f))
                ++mismatches;
            if (!almostEqual(gpuJmods[i].residual[l], jmods[i].residual[l], 1e-3f, 1e-3f))
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

    for (PxgDevicePtr pfree : { dVel, dBodyOut, dIsl, dCC, dJC, dCS, dJS, dBatch, dCh, dFh, dCp, dFr,
                                dJh, dJc, dJm, dDesc, dShared })
        shim.memFree(pfree);
}

} // namespace
