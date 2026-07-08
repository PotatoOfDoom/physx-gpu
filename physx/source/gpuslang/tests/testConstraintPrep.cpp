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

// Verification of constraintPrepPGS.slang / constraintPrepTGS.slang against C++
// transcriptions of the CUDA constraint-prep kernels
// (contactConstraintBlockPrep.cuh, jointConstraintBlockPrep(TGS).cuh,
// constraintPrepShared.cuh, DyCpuGpu1dConstraint.h).

#include "pxgTestFramework.h"
#include "pxgGpuStructs.h"
#include "pxgRefMath.h"

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;

const float PX_MAX_F32 = 3.4028234663852885981170418348452e+38f;

// flag values (mirror the shader constants)
const uint32_t PX1D_SPRING = 1u << 0;
const uint32_t PX1D_ACCELERATION_SPRING = 1u << 1;
const uint32_t PX1D_RESTITUTION = 1u << 2;
const uint32_t PX1D_KEEPBIAS = 1u << 3;
const uint32_t PX1D_OUTPUT_FORCE = 1u << 4;
const uint32_t PX1D_HAS_DRIVE_LIMIT = 1u << 5;
const uint32_t PX1D_ANGULAR_CONSTRAINT = 1u << 6;

const uint32_t HINT_ROT_EQ = 1024;
const uint32_t HINT_ROT_INEQ = 1025;
const uint32_t HINT_EQ = 2048;
const uint32_t HINT_INEQ = 2049;

const uint32_t CFLAG_DRIVE_LIMITS_ARE_FORCES = 1u << 5;
const uint32_t CFLAG_DISABLE_PREPROCESSING = 1u << 6;

const uint32_t RB_KINEMATIC = 1u << 0;
const uint32_t RB_SPEC_CCD = 1u << 4;

const uint32_t SC_HAS_FORCE_THRESHOLDS = 1u << 0;
const uint32_t SC_PER_POINT_FRICTION = 1u << 1;
const uint32_t SC_DISABLE_FRICTION = 1u << 2;
const uint32_t SC_COMPLIANT_ACCEL_SPRING = 1u << 3;

const uint32_t DY_SC_OUTPUT_FORCE = 1u << 1;
const uint32_t DY_SC_KEEP_BIAS = 1u << 2;
const uint32_t DY_SC_ROT_EQ = 1u << 3;
const uint32_t DY_SC_ORTHO_TARGET = 1u << 4;
const uint32_t DY_SC_SPRING = 1u << 5;
const uint32_t DY_SC_INEQUALITY = 1u << 6;
const uint32_t DY_SC_ACCELERATION_SPRING = 1u << 7;

const float PXC_SAME_NORMAL = 0.999f;
const float DY_MIN_RESPONSE = 1e-12f;
const uint32_t MAX_CONSTRAINT_ROWS = 20;

struct Float2
{
    float x, y;
};

// ---------------------------------------------------------------------------
// byte-exact struct mirrors
// ---------------------------------------------------------------------------

struct Batch
{
    uint16_t descStride;
    uint16_t constraintType;
    uint32_t blockContactIndex;
    uint32_t mConstraintBatchIndex;
    uint32_t startConstraintIndex;
    uint32_t startFrictionIndex;
    uint32_t mStartPartitionIndex;
    uint32_t mArticulationResponseIndex;
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
static_assert(sizeof(Batch) == 1536, "batch");

struct BlockContactData
{
    Float4 mInvMassScale[32];
    Float4 normal_restitutionW[32];
    Float4 contactData[32];
    float damping[32];
};
static_assert(sizeof(BlockContactData) == 1664, "contact data");

struct BlockContactPoint
{
    Float4 point_separationW[32];
    Float4 targetVel_maxImpulseW[32];
};
static_assert(sizeof(BlockContactPoint) == 1024, "contact point");

struct BlockFrictionAnchorPatch
{
    Float4 body0Anchors0[32];
    Float4 body0Anchors1[32];
    Float4 body1Anchors0[32];
    Float4 body1Anchors1[32];
};
static_assert(sizeof(BlockFrictionAnchorPatch) == 2048, "anchor patch");

struct BlockFrictionPatch
{
    Float4 body0Normal[32];
    Float4 body1Normal[32];
    uint32_t anchorCount[32];
    uint32_t broken[32];
    uint32_t contactID0[32];
    uint32_t contactID1[32];
    Float4 anchorPoints0[32];
    Float4 anchorPoints1[32];
    uint32_t patchIndex[32];
    uint32_t pad0[32];
};
static_assert(sizeof(BlockFrictionPatch) == 2816, "friction patch");

struct BlockConstraint1DData
{
    Float4 mInvMassScale[32];
    Float4 mRAWorld_linBreakForce[32];
    Float4 mRBWorld_AngBreakForce[32];
    uint32_t mNumRows[32];
    uint32_t mFlags[32];
};
static_assert(sizeof(BlockConstraint1DData) == 1792, "1d data");

struct BlockConstraint1DVelocities
{
    Float4 linear0XYZ_geometricErrorW[32];
    Float4 angular0XYZ_velocityTargetW[32];
    Float4 linear1XYZ_minImpulseW[32];
    Float4 angular1XYZ_maxImpulseW[32];
};
static_assert(sizeof(BlockConstraint1DVelocities) == 2048, "1d vel");

struct BlockConstraint1DParameters
{
    float springStiffness_restitution[32];
    float springDamping_velocityThreshold[32];
    uint32_t flags[32];
    uint32_t solveHint[32];
};
static_assert(sizeof(BlockConstraint1DParameters) == 512, "1d par");

struct BlockWorkUnit
{
    uint32_t mWriteback[32];
    float mRestDistance[32];
    uint32_t mEdgeIndex[32];
    uint32_t mFlags[32];
    uint32_t mPatchIndex[32];
    uint32_t mFrictionPatchIndex[32];
    Float2 mTorsionalFrictionData[32];
};
static_assert(sizeof(BlockWorkUnit) == 1024, "work unit");

// ---- PGS output ----
struct BlockSolverContactHeader
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
static_assert(sizeof(BlockSolverContactHeader) == 1920, "pgs cheader");

struct BlockSolverFrictionHeader
{
    Float4 frictionNormals0[32];
    Float4 frictionNormals1[32];
    uint32_t numFrictionConstr[32];
    float dynamicFriction[32];
    uint32_t broken[32];
};
static_assert(sizeof(BlockSolverFrictionHeader) == 1408, "pgs fheader");

struct BlockSolverContactPoint
{
    Float4 raXn_targetVelocity[32];
    Float4 rbXn_maxImpulse[32];
    float appliedForce[32];
    float resp0[32];
    float resp1[32];
    float coeff0[32];
    float coeff1[32];
};
static_assert(sizeof(BlockSolverContactPoint) == 1664, "pgs cpoint");

struct BlockSolverContactFriction
{
    Float4 raXn_bias[32];
    Float4 rbXn_targetVelW[32];
    float appliedForce[32];
    float resp0[32];
    float resp1[32];
};
static_assert(sizeof(BlockSolverContactFriction) == 1408, "pgs friction");

struct BlockSolverConstraint1DHeader
{
    uint32_t rowCounts[16];
    uint32_t pad0[16];
    uint32_t breakable[16];
    uint32_t pad1[16];
    float angBreakImpulse[32];
    Float4 body0WorldOffset_linBreakImpulse[32];
    float invMass0D0[32];
    float invMass1D1[32];
    float invInertiaScale0[32];
    float invInertiaScale1[32];
    uint32_t writeBackOffset[32];
    float cfm[32];
};
static_assert(sizeof(BlockSolverConstraint1DHeader) == 1664, "pgs jheader");

struct BlockSolverConstraint1DCon
{
    Float4 lin0XYZ_minImpulse[32];
    Float4 lin1XYZ_maxImpulse[32];
    Float4 ang0XYZ_resp0[32];
    Float4 ang1XYZ_resp1[32];
    float initJointSpeed[32];
    uint32_t pad0[32];
};
static_assert(sizeof(BlockSolverConstraint1DCon) == 2304, "pgs jcon");

struct BlockSolverConstraint1DMod
{
    float ang0Writeback[96];
    float appliedForce[32];
    uint32_t flags[32];
    float residual[32];
    float coeff0[32];
    float coeff1[32];
};
static_assert(sizeof(BlockSolverConstraint1DMod) == 1024, "pgs jmod");

// ---- TGS output ----
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
static_assert(sizeof(TGSContactHeader) == 2048, "tgs cheader");

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
static_assert(sizeof(TGSFrictionHeader) == 1664, "tgs fheader");

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
static_assert(sizeof(TGSContactPoint) == 1792, "tgs cpoint");

struct TGSFriction
{
    Float4 raXn_error[32];
    Float4 rbXn_targetVelW[32];
    float appliedForce[32];
    float resp0[32];
    float resp1[32];
};
static_assert(sizeof(TGSFriction) == 1408, "tgs friction");

struct TGSJointHeader
{
    uint32_t rowCounts_breakable_orthoAxisCount[32];
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
static_assert(sizeof(TGSJointHeader) == 4992, "tgs jheader");

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
static_assert(sizeof(TGSJointRowCon) == 3328, "tgs jcon");

struct PrepParams
{
    uint32_t numBatches;
    uint32_t totalPreviousEdges;
    uint32_t totalCurrentEdges;
    uint32_t hasPrevFrictionIndices;
    float dtOrStepDt;
    float invDtOrStepInvDt;
    float totalDt;
    float invTotalDt;
    float bounceThreshold;
    float frictionOffsetThreshold;
    float correlationDistance;
    float ccdMaxSeparation;
    float lengthScale;
    float biasCoefficient;
};

// ---------------------------------------------------------------------------
// reference helpers (verbatim transcriptions of the CUDA source)
// ---------------------------------------------------------------------------

// PxAlignedQuat rotate is identical to QuatV::rotate; body frames stored as
// (QuatV q, Vec3 p) in the reference
struct AFrame
{
    QuatV q;
    Vec3 p;
};

Vec3 aRotate(const QuatV& q, const Vec3& v) { return q.rotate(v); }
Vec3 aRotateInv(const QuatV& q, const Vec3& v) { return q.rotateInv(v); }
Vec3 aTransform(const AFrame& t, const Vec3& v) { return t.q.rotate(v) + t.p; }
Vec3 aTransformInv(const AFrame& t, const Vec3& v) { return t.q.rotateInv(v - t.p); }

// body1ToBody0 = bodyFrame0.transformInv(bodyFrame1)
AFrame aTransformInvFrame(const AFrame& a, const AFrame& b)
{
    AFrame r;
    r.q = a.q.getConjugate() * b.q;
    r.p = a.q.rotateInv(b.p - a.p);
    return r;
}

float dot3f4(const Float4& a, const Float4& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

bool pointsAreClose(const AFrame& body1ToBody0, const Float4& la0, const Float4& la1, const Float4& axis,
                    float correlDist)
{
    const Vec3 p1 = aTransform(body1ToBody0, Vec3(la1.x, la1.y, la1.z));
    const Vec3 d(la0.x - p1.x, la0.y - p1.y, la0.z - p1.z);
    return std::fabs(d.x * axis.x + d.y * axis.y + d.z * axis.z) < correlDist;
}

float computeCompliantDamping(bool isSeparated, bool collidingWithVrel, float damping)
{
    return (isSeparated && !collidingWithVrel) ? 0.0f : damping;
}

void queryReducedCompliantContactCoefficients(float dt, uint32_t /*flags*/, float restitution, float damping,
                                              float penetration, float /*targetVelocity*/, bool isSeparated,
                                              bool collidingWithVrel, float& coeff0, float& coeff1)
{
    const float nrdt = dt * restitution;
    const float dampingIfEnabled = computeCompliantDamping(isSeparated, collidingWithVrel, damping);
    coeff0 = dt * (dampingIfEnabled - nrdt);
    coeff1 = -nrdt * penetration;
}

float computeCompliantContactCoefficientsTGS(float stepDt, uint32_t flags, float restitution, float damping,
                                             float unitResponse, float recipResponse, bool isSeparated,
                                             bool collidingWithVrel, float& velMultiplier, float& scaledBias)
{
    const bool accelSpring = (flags & SC_COMPLIANT_ACCEL_SPRING) != 0;
    const float massIfAccelElseOne = accelSpring ? recipResponse : 1.0f;
    const float oneIfAccelElseR = accelSpring ? 1.0f : unitResponse;
    const float dampingIfEnabled = computeCompliantDamping(isSeparated, collidingWithVrel, damping);
    const float nrdt = stepDt * restitution;
    const float a = stepDt * (dampingIfEnabled - nrdt);
    const float x = 1.0f / (a * oneIfAccelElseR + 1.0f);
    velMultiplier = x * a * massIfAccelElseOne;
    scaledBias = nrdt * x * oneIfAccelElseR;
    return a;
}

float computeRecipUnitResponse(float unitResponse, float minRowResponse)
{
    return unitResponse <= minRowResponse ? 0.0f : 1.0f / unitResponse;
}

void computeMinMaxImpulseOrForceAsImpulse(float minS, float maxS, bool hasDriveLimit, bool driveLimitsAreForces,
                                          float simDt, float& minImp, float& maxImp)
{
    const float driveScale = (hasDriveLimit && driveLimitsAreForces) ? simDt : 1.0f;
    minImp = minS * driveScale;
    maxImp = maxS * driveScale;
}

float computeBounceVelocity(uint32_t flags, float jointSpeed, float bounceThreshold, float restitution,
                            float geometricError)
{
    const float bounceVel = jointSpeed * (-restitution);
    if ((flags & PX1D_RESTITUTION) != 0 && (-jointSpeed > bounceThreshold) && ((bounceVel * geometricError) <= 0.0f))
        return bounceVel;
    return 0.0f;
}

void queryReduced1dPGS(uint32_t flags, float stiffness, float damping, float restitution, float bounceThreshold,
                       float geometricError, float velocityTarget, float jointSpeed, float erp, float simDt,
                       float recipSimDt, float& coeff0, float& coeff1)
{
    if ((flags & PX1D_SPRING) != 0)
    {
        coeff0 = simDt * simDt * stiffness + simDt * damping;
        coeff1 = simDt * (damping * velocityTarget - stiffness * geometricError);
    }
    else
    {
        const float bounceVel = computeBounceVelocity(flags, jointSpeed, bounceThreshold, restitution, geometricError);
        if (bounceVel != 0.0f)
        {
            coeff0 = bounceVel;
            coeff1 = bounceVel;
        }
        else
        {
            const float geomError = geometricError * erp;
            coeff0 = velocityTarget - geomError * recipSimDt;
            coeff1 = ((flags & PX1D_KEEPBIAS) == 0) ? velocityTarget : coeff0;
        }
    }
}

float computeMaxBiasVelocityTGS(uint32_t flags, float jointSpeed, float bounceThreshold, float restitution,
                                float geometricError, bool isExtended, float lengthScale, float recipSimDt)
{
    if ((flags & PX1D_SPRING) != 0)
        return PX_MAX_F32;
    const float bounceVel = computeBounceVelocity(flags, jointSpeed, bounceThreshold, restitution, geometricError);
    if (bounceVel != 0.0f)
        return 0.0f;
    if ((flags & PX1D_ANGULAR_CONSTRAINT) != 0)
        return recipSimDt * 0.75f;
    return isExtended ? recipSimDt * 1.5f * lengthScale : recipSimDt * 15.0f * lengthScale;
}

void computeJointSpeedTGS(float normalVel0, bool isKin0, float normalVel1, bool isKin1, float& jointSpeed,
                          float& initJointSpeed)
{
    initJointSpeed = 0.0f;
    if (isKin0)
        initJointSpeed -= normalVel0;
    if (isKin1)
        initJointSpeed += normalVel1;
    jointSpeed = normalVel0 - normalVel1;
}

void compute1dConstraintSolverConstantsTGS(uint32_t flags, float stiffness, float damping, float restitution,
                                           float bounceThreshold, float geometricError, float velocityTarget,
                                           float jointSpeed, float initJointSpeed, float unitResponse,
                                           float recipUnitResponse, float erp, float stepDt, float recipStepDt,
                                           float& biasScale, float& error, float& velMultiplier, float& targetVel)
{
    biasScale = error = velMultiplier = targetVel = 0.0f;
    if ((flags & PX1D_SPRING) != 0)
    {
        const float a = stepDt * (stepDt * stiffness + damping);
        const float b = stepDt * (damping * velocityTarget);
        if ((flags & PX1D_ACCELERATION_SPRING) != 0)
        {
            const float x = 1.0f / (1.0f + a);
            biasScale = -x * stiffness * stepDt * recipUnitResponse;
            velMultiplier = -x * a * recipUnitResponse;
            error = biasScale * geometricError;
            targetVel = x * b * recipUnitResponse - velMultiplier * initJointSpeed;
        }
        else
        {
            const float x = 1.0f / (1.0f + a * unitResponse);
            biasScale = -x * stiffness * stepDt;
            velMultiplier = -x * a;
            error = biasScale * geometricError;
            targetVel = x * b - velMultiplier * initJointSpeed;
        }
    }
    else
    {
        const float bounceVel = computeBounceVelocity(flags, jointSpeed, bounceThreshold, restitution, geometricError);
        if (bounceVel != 0.0f)
        {
            velMultiplier = -1.0f;
            biasScale = 0.0f;
            error = 0.0f;
            targetVel = bounceVel - velMultiplier * initJointSpeed;
        }
        else
        {
            velMultiplier = -1.0f;
            biasScale = -recipStepDt * erp;
            error = geometricError * biasScale;
            targetVel = velocityTarget - velMultiplier * initJointSpeed;
        }
    }
}

void raiseInternalFlagsTGS(uint32_t externalFlags, uint32_t externalSolveHint, uint32_t& internalFlags)
{
    if ((externalFlags & PX1D_SPRING) != 0)
        internalFlags |= DY_SC_SPRING;
    if ((externalFlags & PX1D_ACCELERATION_SPRING) != 0)
        internalFlags |= DY_SC_ACCELERATION_SPRING;
    if ((externalFlags & PX1D_OUTPUT_FORCE) != 0)
        internalFlags |= DY_SC_OUTPUT_FORCE;
    if ((externalFlags & PX1D_KEEPBIAS) != 0)
        internalFlags |= DY_SC_KEEP_BIAS;
    if ((externalSolveHint & 1u) != 0)
        internalFlags |= DY_SC_INEQUALITY;
}
} // namespace

// The tests (contact + joint, PGS + TGS) live in testConstraintPrep2.inc to
// keep this translation unit readable; it is #included below.
#include "testConstraintPrep2.inc"
