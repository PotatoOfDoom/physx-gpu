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

// P8.67: layout proof for the articulation BLOCK-LAYOUT address map
// (pxgSlangArticulationBlocks.slang).
//
// PxgArticulationBlockData and PxgArticulationBlockDofData hold thirty-two
// articulations interleaved as structure-of-arrays, so every field access is
// three numbers -- struct stride, field offset, lane stride -- and each is
// independently part of the ABI.  This test proves all three, twice over:
//
//   * COMPILE TIME: every constant in the Slang module is re-derived here from
//     host-side sizeof/offsetof of the real engine structs.  The list below was
//     GENERATED from those headers, not transcribed, for the same reason the
//     Slang constants were: fifty hand-copied offsets out of a 29 KB SoA header
//     is how P8.49a's BV32 stride bug happened, where shader and fixture
//     carried the same wrong number and agreed with each other.
//
//   * RUN TIME: the artiBlockProbe kernel reads element [1] of a two-element
//     array at lane 3 and the values come back for a bitwise compare.  Element
//     [0] is filled with a DIFFERENT sentinel set rather than zeros, so a wrong
//     struct stride reads plausible-looking wrong data instead of an obvious
//     zero; and lane 3 with distinct per-lane values means a wrong element
//     stride lands on another articulation's slot.
//
// No kernel consumes this map yet.  It is landed on its own because it is the
// prerequisite for the whole unconstrained-dynamics chain, and because this
// port already treats a layout proof as a deliverable (testSimCoreDescLayout,
// testSolverDescLayout, testPrepDescLayout, testNpAbiLayout) -- those are what
// made the kernel tranches that followed them safe.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "PxArticulationTendonData.h"
#include "PxgArticulation.h"
#include "PxgArticulationLink.h"

#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

static std::vector<uint8_t> artiBlockLoader(const char* file) { return loadShaderBlob(file); }

// ---------------------------------------------------------------------------
// Compile-time pins -- GENERATED from the engine headers, one per Slang constant
// ---------------------------------------------------------------------------
namespace pin
{
using namespace physx;
static_assert(sizeof(PxgSpatialVectorBlock) == 768, "PXG_AB_SPATIAL_VECTOR_BLOCK_SIZE");
static_assert(offsetof(PxgSpatialVectorBlock, mTopxyz_bx) == 0, "PXG_AB_SVB_TOPXYZ_BX");
static_assert(offsetof(PxgSpatialVectorBlock, mbyz) == 512, "PXG_AB_SVB_BYZ");
static_assert(sizeof(PxgMat33Block) == 1152, "PXG_AB_MAT33_BLOCK_SIZE");
static_assert(offsetof(PxgMat33Block, mCol0) == 0, "PXG_AB_MAT33_COL0");
static_assert(offsetof(PxgMat33Block, mCol1) == 512, "PXG_AB_MAT33_COL1");
static_assert(offsetof(PxgMat33Block, mCol2) == 1024, "PXG_AB_MAT33_COL2");
static_assert(sizeof(PxgSpatialMatrixBlock) == 3584, "PXG_AB_SPATIAL_MATRIX_BLOCK_SIZE");
static_assert(sizeof(PxgSpatialTransformBlock) == 1024, "PXG_AB_SPATIAL_TRANSFORM_BLOCK_SIZE");
static_assert(offsetof(PxgSpatialTransformBlock, q) == 0, "PXG_AB_STB_Q");
static_assert(offsetof(PxgSpatialTransformBlock, p) == 512, "PXG_AB_STB_P");

static_assert(sizeof(PxgArticulationBlockData) == 7168, "PXG_AB_BLOCK_SIZE");
static_assert(offsetof(PxgArticulationBlockData, mInvSpatialArticulatedInertia) == 0, "PXG_AB_mInvSpatialArticulatedInertia");
static_assert(offsetof(PxgArticulationBlockData, mFlags) == 3584, "PXG_AB_mFlags");
static_assert(offsetof(PxgArticulationBlockData, mNumLinks) == 3712, "PXG_AB_mNumLinks");
static_assert(offsetof(PxgArticulationBlockData, mNumSpatialTendons) == 3840, "PXG_AB_mNumSpatialTendons");
static_assert(offsetof(PxgArticulationBlockData, mNumFixedTendons) == 3968, "PXG_AB_mNumFixedTendons");
static_assert(offsetof(PxgArticulationBlockData, mNumMimicJoints) == 4096, "PXG_AB_mNumMimicJoints");
static_assert(offsetof(PxgArticulationBlockData, mTotalDofs) == 4224, "PXG_AB_mTotalDofs");
static_assert(offsetof(PxgArticulationBlockData, mArticulationIndex) == 4352, "PXG_AB_mArticulationIndex");
static_assert(offsetof(PxgArticulationBlockData, mSleepThreshold) == 4480, "PXG_AB_mSleepThreshold");
static_assert(offsetof(PxgArticulationBlockData, mCommonLinkDeltaVelocity) == 4608, "PXG_AB_mCommonLinkDeltaVelocity");
static_assert(offsetof(PxgArticulationBlockData, mLinkWithDeferredImpulse) == 5376, "PXG_AB_mLinkWithDeferredImpulse");
static_assert(offsetof(PxgArticulationBlockData, mMotionVelocitiesPtr) == 5504, "PXG_AB_mMotionVelocitiesPtr");
static_assert(offsetof(PxgArticulationBlockData, mRootDeferredZ) == 5760, "PXG_AB_mRootDeferredZ");
static_assert(offsetof(PxgArticulationBlockData, mCOM_TotalInvMassW) == 6528, "PXG_AB_mCOM_TotalInvMassW");
static_assert(offsetof(PxgArticulationBlockData, mStateDirty) == 7040, "PXG_AB_mStateDirty");
static_assert(offsetof(PxgArticulationBlockData, mTotalSelfConstraintCount) == 7072, "PXG_AB_mTotalSelfConstraintCount");
static_assert(offsetof(PxgArticulationBlockData, mSelfConstraintOffset) == 7076, "PXG_AB_mSelfConstraintOffset");

static_assert(sizeof(PxgArticulationBlockDofData) == 12800, "PXG_AB_DOF_SIZE");
static_assert(offsetof(PxgArticulationBlockDofData, mJointPositions) == 0, "PXG_AB_DOF_mJointPositions");
static_assert(offsetof(PxgArticulationBlockDofData, mJointVelocities) == 128, "PXG_AB_DOF_mJointVelocities");
static_assert(offsetof(PxgArticulationBlockDofData, mJointUnconstrainedVelocities) == 256, "PXG_AB_DOF_mJointUnconstrainedVelocities");
static_assert(offsetof(PxgArticulationBlockDofData, mPosJointVelocities) == 384, "PXG_AB_DOF_mPosJointVelocities");
static_assert(offsetof(PxgArticulationBlockDofData, mJointAccel) == 512, "PXG_AB_DOF_mJointAccel");
static_assert(offsetof(PxgArticulationBlockDofData, mQstZ) == 640, "PXG_AB_DOF_mQstZ");
static_assert(offsetof(PxgArticulationBlockDofData, mQstZIcInternal) == 768, "PXG_AB_DOF_mQstZIcInternal");
static_assert(offsetof(PxgArticulationBlockDofData, mDeferredQstZ) == 896, "PXG_AB_DOF_mDeferredQstZ");
static_assert(offsetof(PxgArticulationBlockDofData, mTmpQstZ) == 1024, "PXG_AB_DOF_mTmpQstZ");
static_assert(offsetof(PxgArticulationBlockDofData, mWorldMotionMatrix) == 1152, "PXG_AB_DOF_mWorldMotionMatrix");
static_assert(offsetof(PxgArticulationBlockDofData, mLocalMotionMatrix) == 1920, "PXG_AB_DOF_mLocalMotionMatrix");
static_assert(offsetof(PxgArticulationBlockDofData, mIsInvDW) == 2688, "PXG_AB_DOF_mIsInvDW");
static_assert(offsetof(PxgArticulationBlockDofData, mIsW) == 3456, "PXG_AB_DOF_mIsW");
static_assert(offsetof(PxgArticulationBlockDofData, mJointAxis) == 4224, "PXG_AB_DOF_mJointAxis");
static_assert(offsetof(PxgArticulationBlockDofData, mInvStIsT_x) == 4992, "PXG_AB_DOF_mInvStIsT_x");
static_assert(offsetof(PxgArticulationBlockDofData, mInvStIsT_y) == 5120, "PXG_AB_DOF_mInvStIsT_y");
static_assert(offsetof(PxgArticulationBlockDofData, mInvStIsT_z) == 5248, "PXG_AB_DOF_mInvStIsT_z");
static_assert(offsetof(PxgArticulationBlockDofData, mDofIds) == 5376, "PXG_AB_DOF_mDofIds");
static_assert(offsetof(PxgArticulationBlockDofData, mMotion) == 5408, "PXG_AB_DOF_mMotion");
static_assert(offsetof(PxgArticulationBlockDofData, mConstraintData) == 5440, "PXG_AB_DOF_mConstraintData");
static_assert(sizeof(PxgArticulationBlockLinkData) == 29184, "PXG_AB_LINK_SIZE");
static_assert(sizeof(PxgSpatialMatrix6x6Block) == 4608, "PXG_AB_SPATIAL_MATRIX_6X6_BLOCK_SIZE");
static_assert(offsetof(PxgArticulationBlockLinkData, mSpatialArticulatedInertia) == 0, "PXG_AB_LINK_mSpatialArticulatedInertia");
static_assert(offsetof(PxgArticulationBlockLinkData, mSpatialResponseMatrix) == 3584, "PXG_AB_LINK_mSpatialResponseMatrix");
static_assert(offsetof(PxgArticulationBlockLinkData, mIsolatedInertia) == 8192, "PXG_AB_LINK_mIsolatedInertia");
static_assert(offsetof(PxgArticulationBlockLinkData, mMass) == 9344, "PXG_AB_LINK_mMass");
static_assert(offsetof(PxgArticulationBlockLinkData, mMotionVelocity) == 9472, "PXG_AB_LINK_mMotionVelocity");
static_assert(offsetof(PxgArticulationBlockLinkData, mPosMotionVelocity) == 10240, "PXG_AB_LINK_mPosMotionVelocity");
static_assert(offsetof(PxgArticulationBlockLinkData, mDeltaMotion) == 11008, "PXG_AB_LINK_mDeltaMotion");
static_assert(offsetof(PxgArticulationBlockLinkData, mScratchImpulse) == 11776, "PXG_AB_LINK_mScratchImpulse");
static_assert(offsetof(PxgArticulationBlockLinkData, mScratchDeltaV) == 12544, "PXG_AB_LINK_mScratchDeltaV");
static_assert(offsetof(PxgArticulationBlockLinkData, mSolverSpatialDeltaVel) == 13312, "PXG_AB_LINK_mSolverSpatialDeltaVel");
static_assert(offsetof(PxgArticulationBlockLinkData, mSolverSpatialImpulse) == 14080, "PXG_AB_LINK_mSolverSpatialImpulse");
static_assert(offsetof(PxgArticulationBlockLinkData, mSolverSpatialInternalConstraintImpulse) == 14848, "PXG_AB_LINK_mSolverSpatialInternalConstraintImpulse");
static_assert(offsetof(PxgArticulationBlockLinkData, mZAVector) == 15616, "PXG_AB_LINK_mZAVector");
static_assert(offsetof(PxgArticulationBlockLinkData, mZAIntVector) == 16384, "PXG_AB_LINK_mZAIntVector");
static_assert(offsetof(PxgArticulationBlockLinkData, mCoriolis) == 17152, "PXG_AB_LINK_mCoriolis");
static_assert(offsetof(PxgArticulationBlockLinkData, mConstraintForces) == 17920, "PXG_AB_LINK_mConstraintForces");
static_assert(offsetof(PxgArticulationBlockLinkData, mMotionAcceleration) == 18688, "PXG_AB_LINK_mMotionAcceleration");
static_assert(offsetof(PxgArticulationBlockLinkData, mMotionAccelerationInternal) == 19456, "PXG_AB_LINK_mMotionAccelerationInternal");
static_assert(offsetof(PxgArticulationBlockLinkData, mBiasForce) == 20224, "PXG_AB_LINK_mBiasForce");
static_assert(offsetof(PxgArticulationBlockLinkData, mDeltaScale) == 20992, "PXG_AB_LINK_mDeltaScale");
static_assert(offsetof(PxgArticulationBlockLinkData, mRw_x) == 21120, "PXG_AB_LINK_mRw_x");
static_assert(offsetof(PxgArticulationBlockLinkData, mRw_y) == 21248, "PXG_AB_LINK_mRw_y");
static_assert(offsetof(PxgArticulationBlockLinkData, mRw_z) == 21376, "PXG_AB_LINK_mRw_z");
static_assert(offsetof(PxgArticulationBlockLinkData, mRelativeQuat) == 21504, "PXG_AB_LINK_mRelativeQuat");
static_assert(offsetof(PxgArticulationBlockLinkData, mNbStaticContacts) == 22016, "PXG_AB_LINK_mNbStaticContacts");
static_assert(offsetof(PxgArticulationBlockLinkData, mStaticContactStartIndex) == 22144, "PXG_AB_LINK_mStaticContactStartIndex");
static_assert(offsetof(PxgArticulationBlockLinkData, mNbStaticJoints) == 22272, "PXG_AB_LINK_mNbStaticJoints");
static_assert(offsetof(PxgArticulationBlockLinkData, mStaticJointStartIndex) == 22400, "PXG_AB_LINK_mStaticJointStartIndex");
static_assert(offsetof(PxgArticulationBlockLinkData, mPreTransform) == 22528, "PXG_AB_LINK_mPreTransform");
static_assert(offsetof(PxgArticulationBlockLinkData, mAccumulatedPose) == 23552, "PXG_AB_LINK_mAccumulatedPose");
static_assert(offsetof(PxgArticulationBlockLinkData, mChildPose) == 24576, "PXG_AB_LINK_mChildPose");
static_assert(offsetof(PxgArticulationBlockLinkData, mParentPose) == 25600, "PXG_AB_LINK_mParentPose");
static_assert(offsetof(PxgArticulationBlockLinkData, mDeltaQ) == 26624, "PXG_AB_LINK_mDeltaQ");
static_assert(offsetof(PxgArticulationBlockLinkData, mLinDampingX_AngDampingY_maxLinVelSqZ_maxAngVelSqW) == 27136, "PXG_AB_LINK_mLinDampingX_AngDampingY_maxLinVelSqZ_maxAngVelSqW");
static_assert(offsetof(PxgArticulationBlockLinkData, mInvInertiaXYZ_invMassW) == 27648, "PXG_AB_LINK_mInvInertiaXYZ_invMassW");
static_assert(offsetof(PxgArticulationBlockLinkData, mCfm) == 28160, "PXG_AB_LINK_mCfm");
static_assert(offsetof(PxgArticulationBlockLinkData, mChildrenOffset) == 28288, "PXG_AB_LINK_mChildrenOffset");
static_assert(offsetof(PxgArticulationBlockLinkData, mNumChildren) == 28416, "PXG_AB_LINK_mNumChildren");
static_assert(offsetof(PxgArticulationBlockLinkData, mJointOffset) == 28544, "PXG_AB_LINK_mJointOffset");
static_assert(offsetof(PxgArticulationBlockLinkData, mParents) == 28672, "PXG_AB_LINK_mParents");
static_assert(offsetof(PxgArticulationBlockLinkData, mDofs) == 28800, "PXG_AB_LINK_mDofs");
static_assert(offsetof(PxgArticulationBlockLinkData, mJointType) == 28832, "PXG_AB_LINK_mJointType");
static_assert(offsetof(PxgArticulationBlockLinkData, mInvDofIds) == 28864, "PXG_AB_LINK_mInvDofIds");
static_assert(offsetof(PxgArticulationBlockLinkData, mDisableGravity) == 29056, "PXG_AB_LINK_mDisableGravity");
static_assert(offsetof(PxgArticulationBlockLinkData, mRetainsAcceleration) == 29088, "PXG_AB_LINK_mRetainsAcceleration");
static_assert(sizeof(PxgArticulationLink) == 80, "PXG_AR_LINK_SIZE");
static_assert(offsetof(PxgArticulationLink, initialAngVel) == 0, "PXG_AR_LINK_initialAngVel");
static_assert(offsetof(PxgArticulationLink, penBiasClamp) == 12, "PXG_AR_LINK_penBiasClamp");
static_assert(offsetof(PxgArticulationLink, initialLinVel) == 16, "PXG_AR_LINK_initialLinVel");
static_assert(offsetof(PxgArticulationLink, invMass) == 28, "PXG_AR_LINK_invMass");
static_assert(offsetof(PxgArticulationLink, offsetSlop) == 60, "PXG_AR_LINK_offsetSlop");
static_assert(offsetof(PxgArticulationLink, pathToRoot) == 64, "PXG_AR_LINK_pathToRoot");
static_assert(offsetof(PxgArticulationLink, childrenOffset) == 52, "PXG_AR_LINK_childrenOffset");
static_assert(offsetof(PxgArticulationLink, numChildren) == 58, "PXG_AR_LINK_numChildren");
static_assert(offsetof(PxgArticulationLink, pathToRootOffset) == 48, "PXG_AR_LINK_pathToRootOffset");
static_assert(offsetof(PxgArticulationLink, numPathToRoot) == 56, "PXG_AR_LINK_numPathToRoot");
static_assert(offsetof(PxgArticulationLink, linearDamping) == 40, "PXG_AR_LINK_linearDamping");
static_assert(offsetof(PxgArticulationLink, angularDamping) == 44, "PXG_AR_LINK_angularDamping");
static_assert(offsetof(PxgArticulationLink, maxLinearVelocitySq) == 36, "PXG_AR_LINK_maxLinearVelocitySq");
static_assert(offsetof(PxgArticulationLink, maxAngularVelocitySq) == 32, "PXG_AR_LINK_maxAngularVelocitySq");
static_assert(offsetof(PxgArticulationLink, disableGravity) == 76, "PXG_AR_LINK_disableGravity");
static_assert(offsetof(PxgArticulationLink, retainsAccelerations) == 77, "PXG_AR_LINK_retainsAccelerations");
static_assert(offsetof(PxgArticulationLink, cfmScale) == 72, "PXG_AR_LINK_cfmScale");

static_assert(sizeof(Dy::ArticulationJointCore) == 544, "PXG_AR_JOINT_CORE_SIZE");
static_assert(offsetof(Dy::ArticulationJointCore, parentPose) == 0, "PXG_AR_JOINT_parentPose");
static_assert(offsetof(Dy::ArticulationJointCore, childPose) == 28, "PXG_AR_JOINT_childPose");
static_assert(offsetof(Dy::ArticulationJointCore, limits) == 56, "PXG_AR_JOINT_limits");
static_assert(offsetof(Dy::ArticulationJointCore, drives) == 104, "PXG_AR_JOINT_drives");
static_assert(offsetof(Dy::ArticulationJointCore, targetP) == 296, "PXG_AR_JOINT_targetP");
static_assert(offsetof(Dy::ArticulationJointCore, targetV) == 320, "PXG_AR_JOINT_targetV");
static_assert(offsetof(Dy::ArticulationJointCore, armature) == 344, "PXG_AR_JOINT_armature");
static_assert(offsetof(Dy::ArticulationJointCore, frictionParams) == 420, "PXG_AR_JOINT_frictionParams");
static_assert(offsetof(Dy::ArticulationJointCore, maxJointVelocity) == 492, "PXG_AR_JOINT_maxJointVelocity");
static_assert(offsetof(Dy::ArticulationJointCore, jointPos) == 368, "PXG_AR_JOINT_jointPos");
static_assert(offsetof(Dy::ArticulationJointCore, jointVel) == 392, "PXG_AR_JOINT_jointVel");
static_assert(offsetof(Dy::ArticulationJointCore, dofIds) == 520, "PXG_AR_JOINT_dofIds");
static_assert(offsetof(Dy::ArticulationJointCore, invDofIds) == 532, "PXG_AR_JOINT_invDofIds");
static_assert(offsetof(Dy::ArticulationJointCore, motion) == 526, "PXG_AR_JOINT_motion");
static_assert(offsetof(Dy::ArticulationJointCore, frictionCoefficient) == 416, "PXG_AR_JOINT_frictionCoefficient");
static_assert(offsetof(Dy::ArticulationJointCore, jointOffset) == 516, "PXG_AR_JOINT_jointOffset");
static_assert(offsetof(Dy::ArticulationJointCore, jCalcUpdateFrames) == 538, "PXG_AR_JOINT_jCalcUpdateFrames");
static_assert(offsetof(Dy::ArticulationJointCore, jointType) == 539, "PXG_AR_JOINT_jointType");
static_assert(sizeof(PxArticulationDrive) == 32, "PXG_AR_DRIVE_SIZE");
static_assert(sizeof(PxArticulationLimit) == 8, "PXG_AR_LIMIT_SIZE");
static_assert(sizeof(PxJointFrictionParams) == 12, "PXG_AR_FRICTION_SIZE");

static_assert(sizeof(Dy::ArticulationJointCoreData) == 8, "PXG_AR_JOINT_CORE_DATA_SIZE");
static_assert(sizeof(Dy::ArticulationJointCoreData::nbDof) == 1, "nbDof is a byte");
static_assert(offsetof(Dy::ArticulationJointCoreData, nbDof) == 4, "PXG_AR_JCD_nbDof");
static_assert(offsetof(Dy::ArticulationJointCoreData, jointOffset) == 0, "PXG_AR_JCD_jointOffset");
static_assert(sizeof(PxgArticulationInternalConstraintData) == 7296, "PXG_AB_IC_SIZE");
static_assert(offsetof(PxgArticulationInternalConstraintData, mRow0) == 0, "PXG_AB_IC_mRow0");
static_assert(offsetof(PxgArticulationInternalConstraintData, mRow1) == 768, "PXG_AB_IC_mRow1");
static_assert(offsetof(PxgArticulationInternalConstraintData, mDeltaVA) == 1536, "PXG_AB_IC_mDeltaVA");
static_assert(offsetof(PxgArticulationInternalConstraintData, mDeltaVB) == 2304, "PXG_AB_IC_mDeltaVB");
static_assert(offsetof(PxgArticulationInternalConstraintData, mRecipResponse) == 3072, "PXG_AB_IC_mRecipResponse");
static_assert(offsetof(PxgArticulationInternalConstraintData, mResponse) == 3200, "PXG_AB_IC_mResponse");
static_assert(offsetof(PxgArticulationInternalConstraintData, mLimits_LowLimitX_highLimitY) == 3328, "PXG_AB_IC_mLimits_LowLimitX_highLimitY");
static_assert(offsetof(PxgArticulationInternalConstraintData, mLimitError_LowX_highY) == 3584, "PXG_AB_IC_mLimitError_LowX_highY");
static_assert(offsetof(PxgArticulationInternalConstraintData, mHighImpulse) == 3840, "PXG_AB_IC_mHighImpulse");
static_assert(offsetof(PxgArticulationInternalConstraintData, mLowImpulse) == 3968, "PXG_AB_IC_mLowImpulse");
static_assert(offsetof(PxgArticulationInternalConstraintData, mMaxJointVelocity) == 4096, "PXG_AB_IC_mMaxJointVelocity");
static_assert(offsetof(PxgArticulationInternalConstraintData, mDriveImpulse) == 4224, "PXG_AB_IC_mDriveImpulse");
static_assert(offsetof(PxgArticulationInternalConstraintData, mConstraintMaxImpulse) == 4352, "PXG_AB_IC_mConstraintMaxImpulse");
static_assert(offsetof(PxgArticulationInternalConstraintData, mMaxForce) == 4480, "PXG_AB_IC_mMaxForce");
static_assert(offsetof(PxgArticulationInternalConstraintData, mDriveType) == 4608, "PXG_AB_IC_mDriveType");
static_assert(offsetof(PxgArticulationInternalConstraintData, mMaxEffort) == 4736, "PXG_AB_IC_mMaxEffort");
static_assert(offsetof(PxgArticulationInternalConstraintData, mMaxActuatorVelocity) == 4864, "PXG_AB_IC_mMaxActuatorVelocity");
static_assert(offsetof(PxgArticulationInternalConstraintData, mVelocityDependentResistance) == 4992, "PXG_AB_IC_mVelocityDependentResistance");
static_assert(offsetof(PxgArticulationInternalConstraintData, mSpeedEffortGradient) == 5120, "PXG_AB_IC_mSpeedEffortGradient");
static_assert(offsetof(PxgArticulationInternalConstraintData, mDriveTargetPos) == 5248, "PXG_AB_IC_mDriveTargetPos");
static_assert(offsetof(PxgArticulationInternalConstraintData, mArmature) == 5376, "PXG_AB_IC_mArmature");
static_assert(offsetof(PxgArticulationInternalConstraintData, mTargetVelPlusInitialBiasX_DriveBiasCoefficientY_VelMultiplierZ_ImpulseMultiplierW) == 5504, "PXG_AB_IC_mTargetVelPlusInitialBiasX_DriveBiasCoefficientY_VelMultiplierZ_ImpulseMultiplierW");
static_assert(offsetof(PxgArticulationInternalConstraintData, mDriveStiffness) == 6016, "PXG_AB_IC_mDriveStiffness");
static_assert(offsetof(PxgArticulationInternalConstraintData, mDamping) == 6144, "PXG_AB_IC_mDamping");
static_assert(offsetof(PxgArticulationInternalConstraintData, mDriveTargetVel) == 6272, "PXG_AB_IC_mDriveTargetVel");
static_assert(offsetof(PxgArticulationInternalConstraintData, mTargetPosBias) == 6400, "PXG_AB_IC_mTargetPosBias");
static_assert(offsetof(PxgArticulationInternalConstraintData, mAccumulatedFrictionImpulse) == 6528, "PXG_AB_IC_mAccumulatedFrictionImpulse");
static_assert(offsetof(PxgArticulationInternalConstraintData, mMaxFrictionForce) == 6656, "PXG_AB_IC_mMaxFrictionForce");
static_assert(offsetof(PxgArticulationInternalConstraintData, mFrictionCoefficient) == 6784, "PXG_AB_IC_mFrictionCoefficient");
static_assert(offsetof(PxgArticulationInternalConstraintData, mStaticFrictionEffort) == 6912, "PXG_AB_IC_mStaticFrictionEffort");
static_assert(offsetof(PxgArticulationInternalConstraintData, mDynamicFrictionEffort) == 7040, "PXG_AB_IC_mDynamicFrictionEffort");
static_assert(offsetof(PxgArticulationInternalConstraintData, mViscousFrictionCoefficient) == 7168, "PXG_AB_IC_mViscousFrictionCoefficient");

static_assert(sizeof(Dy::SpatialSubspaceMatrix) == 80, "PXG_AR_SUBSPACE_SIZE");
static_assert(offsetof(Dy::SpatialSubspaceMatrix, columns) == 0, "PXG_AR_SUBSPACE_COLUMNS");
static_assert(sizeof(Cm::UnAlignedSpatialVector) == 24, "PXG_AR_SPATIAL_VECTOR_SIZE");
static_assert(offsetof(PxgArticulation, jointAxis) == 232, "PXG_ARTI_jointAxis");
static_assert(offsetof(PxgArticulation, motionMatrix) == 216, "PXG_ARTI_motionMatrix");
static_assert(offsetof(PxgArticulation, relativeQuat) == 280, "PXG_ARTI_relativeQuat");
static_assert(offsetof(PxgArticulation, jointOffsets) == 160, "PXG_ARTI_jointOffsets");
static_assert(offsetof(PxgArticulation, cfmScale) == 344, "PXG_ARTI_cfmScale");
static_assert(offsetof(PxgArticulation, cfms) == 336, "PXG_ARTI_cfms");
static_assert(offsetof(PxgArticulationData, flags) == 26, "PXG_ARTI_DATA_FLAGS");
static_assert(offsetof(PxgArticulationData, confiDirty) == 27, "PXG_ARTI_DATA_CONFI_DIRTY");
static_assert(offsetof(PxgArticulationData, updateDirty) == 12, "PXG_ARTI_DATA_UPDATE_DIRTY");
static_assert(offsetof(PxgArticulationLinkProp, invMass) == 12, "PXG_ARTI_LINK_PROP_INV_MASS");
static_assert(offsetof(PxgArticulationLink, cfmScale) == 72, "PXG_ARTI_LINK_CFM_SCALE");
static_assert(uint32_t(PxArticulationFlag::eFIX_BASE) == 1, "PXG_ARTI_FLAG_FIX_BASE");
static_assert(sizeof(Dy::SpatialSubspaceMatrix) == 80, "PXG_ARTI_SUBSPACE_STRIDE");
static_assert(sizeof(PxQuat) == 16, "PXG_ARTI_QUAT_SIZE");
static_assert(sizeof(PxgArticulationData::flags) == 1, "flags is a byte");
static_assert(sizeof(PxgArticulationData::confiDirty) == 1, "confiDirty is a byte");
static_assert(sizeof(PxgArticulationData::updateDirty) == 4, "updateDirty is a word");
} // namespace pin

// distinct sentinels: element index and lane both enter every value, so no two
// slots in the whole array can collide
static uint32_t sentU32(uint32_t elem, uint32_t lane, uint32_t field)
{
    return 0x51000000u + elem * 0x00410000u + lane * 0x00000700u + field;
}
static float sentF32(uint32_t elem, uint32_t lane, uint32_t field)
{
    return 1.5f + float(elem) * 64.0f + float(lane) * 2.0f + float(field) * 0.03125f;
}
static uint64_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

PXG_TEST(artiBlockLayout)
{
    using namespace physx;

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiBlockProbe.spv", "artiBlockProbe", 56u, 0 }, // { u64 x7 }
    };
    PxgCudaShim shim(dev, manifest, 1, &artiBlockLoader);

    const uint32_t ELEM = 1;   // the element the probe reads
    const uint32_t LANE = 3;   // the lane the probe reads

    // ---- fill the REAL engine structs, both elements, every lane ----
    std::vector<PxgArticulationBlockData> blocks(2);
    std::memset(blocks.data(), 0xEE, blocks.size() * sizeof(PxgArticulationBlockData));
    std::vector<PxgArticulationBlockDofData> dofs(2);
    std::memset(dofs.data(), 0xEE, dofs.size() * sizeof(PxgArticulationBlockDofData));
    // standalone sub-blocks: PxgMat33Block and PxgSpatialTransformBlock live
    // inside PxgArticulationBlockLinkData, which is not mapped yet, so they are
    // probed on their own rather than left unread (see the probe's note)
    std::vector<PxgMat33Block> mat33(2);
    std::memset(mat33.data(), 0xEE, mat33.size() * sizeof(PxgMat33Block));
    std::vector<PxgSpatialTransformBlock> xform(2);
    std::memset(xform.data(), 0xEE, xform.size() * sizeof(PxgSpatialTransformBlock));
    // the link block: the largest of the three, filled field by field
    std::vector<PxgArticulationBlockLinkData> links(2);
    std::memset(links.data(), 0xEE, links.size() * sizeof(PxgArticulationBlockLinkData));
    for (uint32_t e = 0; e < 2; ++e)
        for (uint32_t l = 0; l < 32; ++l)
        {
            PxgArticulationBlockLinkData& K = links[e];
            uint32_t f = 300;
            for (uint32_t c = 0; c < 7; ++c)
                K.mSpatialArticulatedInertia.columns[c][l] =
                    make_float4(sentF32(e, l, f), sentF32(e, l, f + 1),
                                sentF32(e, l, f + 2), sentF32(e, l, f + 3)), f += 4;
            for (uint32_t c = 0; c < 9; ++c)
                K.mSpatialResponseMatrix.columns[c][l] =
                    make_float4(sentF32(e, l, f), sentF32(e, l, f + 1),
                                sentF32(e, l, f + 2), sentF32(e, l, f + 3)), f += 4;
            K.mIsolatedInertia.mCol0[l] = make_float4(sentF32(e, l, f), sentF32(e, l, f + 1),
                                                      sentF32(e, l, f + 2), sentF32(e, l, f + 3)); f += 4;
            K.mIsolatedInertia.mCol1[l] = make_float4(sentF32(e, l, f), sentF32(e, l, f + 1),
                                                      sentF32(e, l, f + 2), sentF32(e, l, f + 3)); f += 4;
            K.mIsolatedInertia.mCol2[l] = sentF32(e, l, f++);
            K.mMass[l] = sentF32(e, l, f++);
            PxgSpatialVectorBlock* sv[15] = {
                &K.mMotionVelocity, &K.mPosMotionVelocity, &K.mDeltaMotion, &K.mScratchImpulse,
                &K.mScratchDeltaV, &K.mSolverSpatialDeltaVel, &K.mSolverSpatialImpulse,
                &K.mSolverSpatialInternalConstraintImpulse, &K.mZAVector, &K.mZAIntVector,
                &K.mCoriolis, &K.mConstraintForces, &K.mMotionAcceleration,
                &K.mMotionAccelerationInternal, &K.mBiasForce };
            for (uint32_t v = 0; v < 15; ++v)
            {
                sv[v]->mTopxyz_bx[l] = make_float4(sentF32(e, l, f), sentF32(e, l, f + 1),
                                                   sentF32(e, l, f + 2), sentF32(e, l, f + 3));
                sv[v]->mbyz[l] = make_float2(sentF32(e, l, f + 4), sentF32(e, l, f + 5));
                f += 6;
            }
            K.mDeltaScale[l] = sentF32(e, l, f++);
            K.mRw_x[l] = sentF32(e, l, f++);
            K.mRw_y[l] = sentF32(e, l, f++);
            K.mRw_z[l] = sentF32(e, l, f++);
            K.mRelativeQuat[l] = make_float4(sentF32(e, l, f), sentF32(e, l, f + 1),
                                             sentF32(e, l, f + 2), sentF32(e, l, f + 3)); f += 4;
            K.mNbStaticContacts[l] = sentU32(e, l, f++);
            K.mStaticContactStartIndex[l] = sentU32(e, l, f++);
            K.mNbStaticJoints[l] = sentU32(e, l, f++);
            K.mStaticJointStartIndex[l] = sentU32(e, l, f++);
            PxgSpatialTransformBlock* tb[4] = { &K.mPreTransform, &K.mAccumulatedPose,
                                                &K.mChildPose, &K.mParentPose };
            for (uint32_t v = 0; v < 4; ++v)
            {
                tb[v]->q[l] = make_float4(sentF32(e, l, f), sentF32(e, l, f + 1),
                                          sentF32(e, l, f + 2), sentF32(e, l, f + 3));
                tb[v]->p[l] = make_float4(sentF32(e, l, f + 4), sentF32(e, l, f + 5),
                                          sentF32(e, l, f + 6), sentF32(e, l, f + 7));
                f += 8;
            }
            K.mDeltaQ[l] = make_float4(sentF32(e, l, f), sentF32(e, l, f + 1),
                                       sentF32(e, l, f + 2), sentF32(e, l, f + 3)); f += 4;
            K.mLinDampingX_AngDampingY_maxLinVelSqZ_maxAngVelSqW[l] =
                make_float4(sentF32(e, l, f), sentF32(e, l, f + 1),
                            sentF32(e, l, f + 2), sentF32(e, l, f + 3)); f += 4;
            K.mInvInertiaXYZ_invMassW[l] = make_float4(sentF32(e, l, f), sentF32(e, l, f + 1),
                                                       sentF32(e, l, f + 2), sentF32(e, l, f + 3)); f += 4;
            K.mCfm[l] = sentF32(e, l, f++);
            K.mChildrenOffset[l] = sentU32(e, l, f++);
            K.mNumChildren[l] = sentU32(e, l, f++);
            K.mJointOffset[l] = sentU32(e, l, f++);
            K.mParents[l] = sentU32(e, l, f++);
            K.mDofs[l] = PxU8(0x11u + e * 4u + l % 4u);
            K.mJointType[l] = PxU8(0x22u + e * 4u + l % 4u);
            K.mDisableGravity[l] = PxU8(0x33u + e * 4u + l % 4u);
            K.mRetainsAcceleration[l] = PxU8(0x44u + e * 4u + l % 4u);
            // axis-major: a distinct value per (axis, lane) so a probe that
            // ignores the row stride returns axis 0 three times
            for (uint32_t axis = 0; axis < 6; ++axis)
                K.mInvDofIds[axis][l] = PxU8(0x50u + axis * 16u + e * 4u + l % 4u);
        }

    for (uint32_t e = 0; e < 2; ++e)
        for (uint32_t l = 0; l < 32; ++l)
        {
            mat33[e].mCol0[l] = make_float4(sentF32(e, l, 200), sentF32(e, l, 201),
                                            sentF32(e, l, 202), sentF32(e, l, 203));
            mat33[e].mCol1[l] = make_float4(sentF32(e, l, 204), sentF32(e, l, 205),
                                            sentF32(e, l, 206), sentF32(e, l, 207));
            mat33[e].mCol2[l] = sentF32(e, l, 208);
            xform[e].q[l] = make_float4(sentF32(e, l, 210), sentF32(e, l, 211),
                                        sentF32(e, l, 212), sentF32(e, l, 213));
            xform[e].p[l] = make_float4(sentF32(e, l, 214), sentF32(e, l, 215),
                                        sentF32(e, l, 216), sentF32(e, l, 217));
        }

    for (uint32_t e = 0; e < 2; ++e)
    {
        PxgArticulationBlockData& B = blocks[e];
        for (uint32_t l = 0; l < 32; ++l)
        {
            uint32_t f = 0;
            B.mFlags[l] = sentU32(e, l, f++);
            B.mNumLinks[l] = sentU32(e, l, f++);
            B.mNumSpatialTendons[l] = sentU32(e, l, f++);
            B.mNumFixedTendons[l] = sentU32(e, l, f++);
            B.mNumMimicJoints[l] = sentU32(e, l, f++);
            B.mTotalDofs[l] = sentU32(e, l, f++);
            B.mArticulationIndex[l] = sentU32(e, l, f++);
            B.mSleepThreshold[l] = sentF32(e, l, f++);
            B.mLinkWithDeferredImpulse[l] = sentU32(e, l, f++);
            B.mMotionVelocitiesPtr[l] = reinterpret_cast<Cm::UnAlignedSpatialVector*>(
                0xC000000000000000ull + uint64_t(e) * 0x1000000ull + uint64_t(l) * 0x10000ull);
            B.mStateDirty[l] = PxU8(0x40u + e * 16u + l);
            B.mCOM_TotalInvMassW[l] = make_float4(sentF32(e, l, 20), sentF32(e, l, 21),
                                                  sentF32(e, l, 22), sentF32(e, l, 23));
            for (uint32_t c = 0; c < 7; ++c)
                B.mInvSpatialArticulatedInertia.columns[c][l] =
                    make_float4(sentF32(e, l, 30 + c * 4), sentF32(e, l, 31 + c * 4),
                                sentF32(e, l, 32 + c * 4), sentF32(e, l, 33 + c * 4));
            B.mCommonLinkDeltaVelocity.mTopxyz_bx[l] =
                make_float4(sentF32(e, l, 70), sentF32(e, l, 71), sentF32(e, l, 72), sentF32(e, l, 73));
            B.mCommonLinkDeltaVelocity.mbyz[l] = make_float2(sentF32(e, l, 74), sentF32(e, l, 75));
            B.mRootDeferredZ.mTopxyz_bx[l] =
                make_float4(sentF32(e, l, 80), sentF32(e, l, 81), sentF32(e, l, 82), sentF32(e, l, 83));
            B.mRootDeferredZ.mbyz[l] = make_float2(sentF32(e, l, 84), sentF32(e, l, 85));
        }
        B.mTotalSelfConstraintCount = sentU32(e, 0, 90);
        B.mSelfConstraintOffset = sentU32(e, 0, 91);

        PxgArticulationBlockDofData& D = dofs[e];
        for (uint32_t l = 0; l < 32; ++l)
        {
            uint32_t f = 100;
            D.mJointPositions[l] = sentF32(e, l, f++);
            D.mJointVelocities[l] = sentF32(e, l, f++);
            D.mJointUnconstrainedVelocities[l] = sentF32(e, l, f++);
            D.mPosJointVelocities[l] = sentF32(e, l, f++);
            D.mJointAccel[l] = sentF32(e, l, f++);
            D.mQstZ[l] = sentF32(e, l, f++);
            D.mQstZIcInternal[l] = sentF32(e, l, f++);
            D.mDeferredQstZ[l] = sentF32(e, l, f++);
            D.mTmpQstZ[l] = sentF32(e, l, f++);
            D.mInvStIsT_x[l] = sentF32(e, l, f++);
            D.mInvStIsT_y[l] = sentF32(e, l, f++);
            D.mInvStIsT_z[l] = sentF32(e, l, f++);
            D.mDofIds[l] = PxU8(0x10u + e * 8u + l % 8u);
            D.mMotion[l] = PxU8(0x90u + e * 8u + l % 8u);
            PxgSpatialVectorBlock* svb[5] = { &D.mWorldMotionMatrix, &D.mLocalMotionMatrix,
                                              &D.mIsInvDW, &D.mIsW, &D.mJointAxis };
            for (uint32_t k = 0; k < 5; ++k)
            {
                svb[k]->mTopxyz_bx[l] = make_float4(sentF32(e, l, 130 + k * 6), sentF32(e, l, 131 + k * 6),
                                                    sentF32(e, l, 132 + k * 6), sentF32(e, l, 133 + k * 6));
                svb[k]->mbyz[l] = make_float2(sentF32(e, l, 134 + k * 6), sentF32(e, l, 135 + k * 6));
            }
            PxgArticulationInternalConstraintData& C = D.mConstraintData;
            uint32_t g = 500;
            PxgSpatialVectorBlock* rows[4] = { &C.mRow0, &C.mRow1, &C.mDeltaVA, &C.mDeltaVB };
            for (uint32_t v = 0; v < 4; ++v)
            {
                rows[v]->mTopxyz_bx[l] = make_float4(sentF32(e, l, g), sentF32(e, l, g + 1),
                                                     sentF32(e, l, g + 2), sentF32(e, l, g + 3));
                rows[v]->mbyz[l] = make_float2(sentF32(e, l, g + 4), sentF32(e, l, g + 5));
                g += 6;
            }
            C.mRecipResponse[l] = sentF32(e, l, g++);
            C.mResponse[l] = sentF32(e, l, g++);
            C.mLimits_LowLimitX_highLimitY[l] = make_float2(sentF32(e, l, g), sentF32(e, l, g + 1)); g += 2;
            C.mLimitError_LowX_highY[l] = make_float2(sentF32(e, l, g), sentF32(e, l, g + 1)); g += 2;
            C.mHighImpulse[l] = sentF32(e, l, g++);
            C.mLowImpulse[l] = sentF32(e, l, g++);
            C.mMaxJointVelocity[l] = sentF32(e, l, g++);
            C.mDriveImpulse[l] = sentF32(e, l, g++);
            C.mConstraintMaxImpulse[l] = sentF32(e, l, g++);
            C.mMaxForce[l] = sentF32(e, l, g++);
            C.mDriveType[l] = sentU32(e, l, g++);   // the one PxU32 in the run
            C.mMaxEffort[l] = sentF32(e, l, g++);
            C.mMaxActuatorVelocity[l] = sentF32(e, l, g++);
            C.mVelocityDependentResistance[l] = sentF32(e, l, g++);
            C.mSpeedEffortGradient[l] = sentF32(e, l, g++);
            C.mDriveTargetPos[l] = sentF32(e, l, g++);
            C.mArmature[l] = sentF32(e, l, g++);
            C.mTargetVelPlusInitialBiasX_DriveBiasCoefficientY_VelMultiplierZ_ImpulseMultiplierW[l] =
                make_float4(sentF32(e, l, g), sentF32(e, l, g + 1),
                            sentF32(e, l, g + 2), sentF32(e, l, g + 3)); g += 4;
            C.mDriveStiffness[l] = sentF32(e, l, g++);
            C.mDamping[l] = sentF32(e, l, g++);
            C.mDriveTargetVel[l] = sentF32(e, l, g++);
            C.mTargetPosBias[l] = sentF32(e, l, g++);
            C.mAccumulatedFrictionImpulse[l] = sentF32(e, l, g++);
            C.mMaxFrictionForce[l] = sentF32(e, l, g++);
            C.mFrictionCoefficient[l] = sentF32(e, l, g++);
            C.mStaticFrictionEffort[l] = sentF32(e, l, g++);
            C.mDynamicFrictionEffort[l] = sentF32(e, l, g++);
            C.mViscousFrictionCoefficient[l] = sentF32(e, l, g++);
        }
    }

    // ---- the RECORD structs: PxgArticulationLink[2], ArticulationJointCore[2],
    //      ArticulationJointCoreData[2], packed back to back in one buffer ----
    const size_t RECBYTES = 2 * sizeof(PxgArticulationLink) +
                            2 * sizeof(Dy::ArticulationJointCore) +
                            2 * sizeof(Dy::ArticulationJointCoreData);
    std::vector<uint8_t> rec(RECBYTES, 0xEE);
    PxgArticulationLink* recLinks = reinterpret_cast<PxgArticulationLink*>(rec.data());
    Dy::ArticulationJointCore* recJoints = reinterpret_cast<Dy::ArticulationJointCore*>(
        rec.data() + 2 * sizeof(PxgArticulationLink));
    Dy::ArticulationJointCoreData* recJointData =
        reinterpret_cast<Dy::ArticulationJointCoreData*>(
            rec.data() + 2 * sizeof(PxgArticulationLink) + 2 * sizeof(Dy::ArticulationJointCore));
    for (uint32_t e = 0; e < 2; ++e)
    {
        PxgArticulationLink& L = recLinks[e];
        L.penBiasClamp = sentF32(e, 0, 400);
        L.invMass = sentF32(e, 0, 401);
        L.maxAngularVelocitySq = sentF32(e, 0, 402);
        L.maxLinearVelocitySq = sentF32(e, 0, 403);
        L.linearDamping = sentF32(e, 0, 404);
        L.angularDamping = sentF32(e, 0, 405);
        L.pathToRootOffset = sentU32(e, 0, 406);
        L.childrenOffset = sentU32(e, 0, 407);
        L.numPathToRoot = PxU16(0x1100u + e);
        L.numChildren = PxU16(0x2200u + e);
        L.cfmScale = sentF32(e, 0, 408);
        L.disableGravity = (e == 0);
        L.retainsAccelerations = (e == 1);

        Dy::ArticulationJointCore& J = recJoints[e];
        std::memset(&J, 0xEE, sizeof(J));
        J.parentPose.q.x = sentF32(e, 0, 410);
        J.childPose.q.x = sentF32(e, 0, 411);
        for (uint32_t a = 0; a < PxArticulationAxis::eCOUNT; ++a)
        {
            J.limits[a].low = sentF32(e, a, 420);
            J.drives[a].stiffness = sentF32(e, a, 430);
            J.armature[a] = sentF32(e, a, 440);
            J.frictionParams[a].staticFrictionEffort = sentF32(e, a, 450);
            J.maxJointVelocity[a] = sentF32(e, a, 460);
            J.jointPos[a] = sentF32(e, a, 470);
            J.jointVel[a] = sentF32(e, a, 480);
            J.dofIds[a] = PxU8(0x30u + a * 2u + e);
            J.motion[a] = PxU8(0x60u + a * 2u + e);
            J.invDofIds[a] = PxU8(0x90u + a * 2u + e);
        }
        J.frictionCoefficient = sentF32(e, 0, 490);
        J.jointOffset = sentU32(e, 0, 491);
        J.jointType = PxU8(2u + e);

        recJointData[e].jointOffset = sentU32(e, 0, 492);
        recJointData[e].nbDof = PxU8(0xA0u + e);   // PxU8, not PxU32
    }

    // ---- the expected values, in the probe's emit order ----
    std::vector<uint64_t> expect;
    expect.push_back(sizeof(PxgSpatialVectorBlock));
    expect.push_back(sizeof(PxgMat33Block));
    expect.push_back(sizeof(PxgSpatialMatrixBlock));
    expect.push_back(sizeof(PxgSpatialTransformBlock));
    expect.push_back(sizeof(PxgArticulationBlockData));
    expect.push_back(sizeof(PxgArticulationBlockDofData));

    const PxgArticulationBlockData& B = blocks[ELEM];
    expect.push_back(B.mFlags[LANE]);
    expect.push_back(B.mNumLinks[LANE]);
    expect.push_back(B.mNumSpatialTendons[LANE]);
    expect.push_back(B.mNumFixedTendons[LANE]);
    expect.push_back(B.mNumMimicJoints[LANE]);
    expect.push_back(B.mTotalDofs[LANE]);
    expect.push_back(B.mArticulationIndex[LANE]);
    expect.push_back(bits(B.mSleepThreshold[LANE]));
    expect.push_back(B.mLinkWithDeferredImpulse[LANE]);
    expect.push_back(uint64_t(B.mMotionVelocitiesPtr[LANE]));
    expect.push_back(B.mStateDirty[LANE]);
    expect.push_back(B.mTotalSelfConstraintCount);
    expect.push_back(B.mSelfConstraintOffset);
    expect.push_back(bits(B.mCOM_TotalInvMassW[LANE].x));
    expect.push_back(bits(B.mCOM_TotalInvMassW[LANE].y));
    expect.push_back(bits(B.mCOM_TotalInvMassW[LANE].z));
    expect.push_back(bits(B.mCOM_TotalInvMassW[LANE].w));
    expect.push_back(bits(B.mInvSpatialArticulatedInertia.columns[5][LANE].x));
    expect.push_back(bits(B.mInvSpatialArticulatedInertia.columns[5][LANE].w));
    expect.push_back(bits(B.mCommonLinkDeltaVelocity.mTopxyz_bx[LANE].x));
    expect.push_back(bits(B.mCommonLinkDeltaVelocity.mTopxyz_bx[LANE].z));
    expect.push_back(bits(B.mCommonLinkDeltaVelocity.mTopxyz_bx[LANE].w));
    expect.push_back(bits(B.mCommonLinkDeltaVelocity.mbyz[LANE].y));
    expect.push_back(bits(B.mRootDeferredZ.mTopxyz_bx[LANE].y));
    expect.push_back(bits(B.mRootDeferredZ.mbyz[LANE].x));

    const PxgArticulationBlockDofData& D = dofs[ELEM];
    expect.push_back(bits(D.mJointPositions[LANE]));
    expect.push_back(bits(D.mJointVelocities[LANE]));
    expect.push_back(bits(D.mJointUnconstrainedVelocities[LANE]));
    expect.push_back(bits(D.mPosJointVelocities[LANE]));
    expect.push_back(bits(D.mJointAccel[LANE]));
    expect.push_back(bits(D.mQstZ[LANE]));
    expect.push_back(bits(D.mQstZIcInternal[LANE]));
    expect.push_back(bits(D.mDeferredQstZ[LANE]));
    expect.push_back(bits(D.mTmpQstZ[LANE]));
    expect.push_back(bits(D.mInvStIsT_x[LANE]));
    expect.push_back(bits(D.mInvStIsT_y[LANE]));
    expect.push_back(bits(D.mInvStIsT_z[LANE]));
    expect.push_back(D.mDofIds[LANE]);
    expect.push_back(D.mMotion[LANE]);
    expect.push_back(bits(D.mWorldMotionMatrix.mTopxyz_bx[LANE].x));
    expect.push_back(bits(D.mLocalMotionMatrix.mTopxyz_bx[LANE].x));
    expect.push_back(bits(D.mIsInvDW.mTopxyz_bx[LANE].x));
    expect.push_back(bits(D.mIsW.mTopxyz_bx[LANE].x));
    expect.push_back(bits(D.mJointAxis.mTopxyz_bx[LANE].x));
    expect.push_back(bits(D.mJointAxis.mbyz[LANE].y));
    {
        const PxgArticulationInternalConstraintData& C = D.mConstraintData;
        expect.push_back(sizeof(PxgArticulationInternalConstraintData));
        expect.push_back(bits(C.mRow0.mTopxyz_bx[LANE].x));
        expect.push_back(bits(C.mRow1.mTopxyz_bx[LANE].y));
        expect.push_back(bits(C.mDeltaVA.mTopxyz_bx[LANE].w));   // bottom.x
        expect.push_back(bits(C.mDeltaVB.mbyz[LANE].y));         // bottom.z
        expect.push_back(bits(C.mRecipResponse[LANE]));
        expect.push_back(bits(C.mResponse[LANE]));
        expect.push_back(bits(C.mLimits_LowLimitX_highLimitY[LANE].x));
        expect.push_back(bits(C.mLimits_LowLimitX_highLimitY[LANE].y));
        expect.push_back(bits(C.mLimitError_LowX_highY[LANE].x));
        expect.push_back(bits(C.mLimitError_LowX_highY[LANE].y));
        expect.push_back(bits(C.mHighImpulse[LANE]));
        expect.push_back(bits(C.mLowImpulse[LANE]));
        expect.push_back(bits(C.mMaxJointVelocity[LANE]));
        expect.push_back(bits(C.mDriveImpulse[LANE]));
        expect.push_back(bits(C.mConstraintMaxImpulse[LANE]));
        expect.push_back(bits(C.mMaxForce[LANE]));
        expect.push_back(C.mDriveType[LANE]);
        expect.push_back(bits(C.mMaxEffort[LANE]));
        expect.push_back(bits(C.mMaxActuatorVelocity[LANE]));
        expect.push_back(bits(C.mVelocityDependentResistance[LANE]));
        expect.push_back(bits(C.mSpeedEffortGradient[LANE]));
        expect.push_back(bits(C.mDriveTargetPos[LANE]));
        expect.push_back(bits(C.mArmature[LANE]));
        expect.push_back(bits(C.mTargetVelPlusInitialBiasX_DriveBiasCoefficientY_VelMultiplierZ_ImpulseMultiplierW[LANE].x));
        expect.push_back(bits(C.mTargetVelPlusInitialBiasX_DriveBiasCoefficientY_VelMultiplierZ_ImpulseMultiplierW[LANE].w));
        expect.push_back(bits(C.mDriveStiffness[LANE]));
        expect.push_back(bits(C.mDamping[LANE]));
        expect.push_back(bits(C.mDriveTargetVel[LANE]));
        expect.push_back(bits(C.mTargetPosBias[LANE]));
        expect.push_back(bits(C.mAccumulatedFrictionImpulse[LANE]));
        expect.push_back(bits(C.mMaxFrictionForce[LANE]));
        expect.push_back(bits(C.mFrictionCoefficient[LANE]));
        expect.push_back(bits(C.mStaticFrictionEffort[LANE]));
        expect.push_back(bits(C.mDynamicFrictionEffort[LANE]));
        expect.push_back(bits(C.mViscousFrictionCoefficient[LANE]));
        expect.push_back(sizeof(Dy::SpatialSubspaceMatrix));
        expect.push_back(sizeof(Cm::UnAlignedSpatialVector));
    }

    expect.push_back(bits(mat33[ELEM].mCol0[LANE].x));
    expect.push_back(bits(mat33[ELEM].mCol1[LANE].y));
    expect.push_back(bits(mat33[ELEM].mCol2[LANE]));
    expect.push_back(bits(xform[ELEM].q[LANE].z));
    expect.push_back(bits(xform[ELEM].p[LANE].x));

    const PxgArticulationBlockLinkData& K = links[ELEM];
    expect.push_back(sizeof(PxgArticulationBlockLinkData));
    expect.push_back(sizeof(PxgSpatialMatrix6x6Block));
    expect.push_back(bits(K.mSpatialArticulatedInertia.columns[6][LANE].y));
    expect.push_back(bits(K.mSpatialResponseMatrix.columns[8][LANE].z));
    expect.push_back(bits(K.mIsolatedInertia.mCol1[LANE].w));
    expect.push_back(bits(K.mMass[LANE]));
    {
        const PxgSpatialVectorBlock* sv[15] = {
            &K.mMotionVelocity, &K.mPosMotionVelocity, &K.mDeltaMotion, &K.mScratchImpulse,
            &K.mScratchDeltaV, &K.mSolverSpatialDeltaVel, &K.mSolverSpatialImpulse,
            &K.mSolverSpatialInternalConstraintImpulse, &K.mZAVector, &K.mZAIntVector,
            &K.mCoriolis, &K.mConstraintForces, &K.mMotionAcceleration,
            &K.mMotionAccelerationInternal, &K.mBiasForce };
        // the probe cycles top.x/y/z then bottom.x/y/z; bottom is (w, byz.x, byz.y)
        const int comp[15] = { 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2 };
        for (uint32_t v = 0; v < 15; ++v)
        {
            const float4 t = sv[v]->mTopxyz_bx[LANE];
            const float2 b = sv[v]->mbyz[LANE];
            const float vals[6] = { t.x, t.y, t.z, t.w, b.x, b.y };
            expect.push_back(bits(vals[comp[v]]));
        }
    }
    expect.push_back(bits(K.mDeltaScale[LANE]));
    expect.push_back(bits(K.mRw_x[LANE]));
    expect.push_back(bits(K.mRw_y[LANE]));
    expect.push_back(bits(K.mRw_z[LANE]));
    expect.push_back(bits(K.mRelativeQuat[LANE].w));
    expect.push_back(K.mNbStaticContacts[LANE]);
    expect.push_back(K.mStaticContactStartIndex[LANE]);
    expect.push_back(K.mNbStaticJoints[LANE]);
    expect.push_back(K.mStaticJointStartIndex[LANE]);
    expect.push_back(bits(K.mPreTransform.q[LANE].x));
    expect.push_back(bits(K.mAccumulatedPose.p[LANE].y));
    expect.push_back(bits(K.mChildPose.q[LANE].z));
    expect.push_back(bits(K.mParentPose.p[LANE].w));
    expect.push_back(bits(K.mDeltaQ[LANE].x));
    expect.push_back(bits(K.mLinDampingX_AngDampingY_maxLinVelSqZ_maxAngVelSqW[LANE].z));
    expect.push_back(bits(K.mInvInertiaXYZ_invMassW[LANE].w));
    expect.push_back(bits(K.mCfm[LANE]));
    expect.push_back(K.mChildrenOffset[LANE]);
    expect.push_back(K.mNumChildren[LANE]);
    expect.push_back(K.mJointOffset[LANE]);
    expect.push_back(K.mParents[LANE]);
    expect.push_back(K.mDofs[LANE]);
    expect.push_back(K.mJointType[LANE]);
    expect.push_back(K.mDisableGravity[LANE]);
    expect.push_back(K.mRetainsAcceleration[LANE]);
    expect.push_back(K.mInvDofIds[0][LANE]);
    expect.push_back(K.mInvDofIds[2][LANE]);
    expect.push_back(K.mInvDofIds[5][LANE]);

    {
        const PxgArticulationLink& L = recLinks[ELEM];
        expect.push_back(sizeof(PxgArticulationLink));
        expect.push_back(bits(L.penBiasClamp));
        expect.push_back(bits(L.invMass));
        expect.push_back(bits(L.maxAngularVelocitySq));
        expect.push_back(bits(L.maxLinearVelocitySq));
        expect.push_back(bits(L.linearDamping));
        expect.push_back(bits(L.angularDamping));
        expect.push_back(L.pathToRootOffset);
        expect.push_back(L.childrenOffset);
        expect.push_back(L.numPathToRoot);
        expect.push_back(L.numChildren);
        expect.push_back(bits(L.cfmScale));
        expect.push_back(L.disableGravity ? 1u : 0u);
        expect.push_back(L.retainsAccelerations ? 1u : 0u);

        const Dy::ArticulationJointCore& J = recJoints[ELEM];
        expect.push_back(sizeof(Dy::ArticulationJointCore));
        expect.push_back(bits(J.parentPose.q.x));
        expect.push_back(bits(J.childPose.q.x));
        expect.push_back(bits(J.limits[4].low));
        expect.push_back(bits(J.drives[4].stiffness));
        expect.push_back(bits(J.armature[4]));
        expect.push_back(bits(J.frictionParams[4].staticFrictionEffort));
        expect.push_back(bits(J.maxJointVelocity[4]));
        expect.push_back(bits(J.jointPos[4]));
        expect.push_back(bits(J.jointVel[4]));
        expect.push_back(bits(J.frictionCoefficient));
        expect.push_back(J.jointOffset);
        expect.push_back(J.dofIds[4]);
        expect.push_back(J.motion[4]);
        expect.push_back(J.invDofIds[4]);
        expect.push_back(J.jointType);

        expect.push_back(sizeof(Dy::ArticulationJointCoreData));
        expect.push_back(recJointData[ELEM].jointOffset);
        expect.push_back(recJointData[ELEM].nbDof);
    }

    // ---- upload, dispatch, compare ----
    PxgDevicePtr dBlocks = 0, dDofs = 0, dMat33 = 0, dXform = 0, dLinks = 0, dRec = 0, dOut = 0;
    shim.memAlloc(&dBlocks, blocks.size() * sizeof(PxgArticulationBlockData));
    shim.memAlloc(&dDofs, dofs.size() * sizeof(PxgArticulationBlockDofData));
    shim.memAlloc(&dMat33, mat33.size() * sizeof(PxgMat33Block));
    shim.memAlloc(&dXform, xform.size() * sizeof(PxgSpatialTransformBlock));
    shim.memAlloc(&dLinks, links.size() * sizeof(PxgArticulationBlockLinkData));
    shim.memAlloc(&dRec, rec.size());
    shim.memAlloc(&dOut, expect.size() * 8);
    if (!dBlocks || !dDofs || !dMat33 || !dXform || !dLinks || !dRec || !dOut)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    shim.memcpyHtoD(dBlocks, blocks.data(), blocks.size() * sizeof(PxgArticulationBlockData));
    shim.memcpyHtoD(dDofs, dofs.data(), dofs.size() * sizeof(PxgArticulationBlockDofData));
    shim.memcpyHtoD(dMat33, mat33.data(), mat33.size() * sizeof(PxgMat33Block));
    shim.memcpyHtoD(dXform, xform.data(), xform.size() * sizeof(PxgSpatialTransformBlock));
    shim.memcpyHtoD(dLinks, links.data(), links.size() * sizeof(PxgArticulationBlockLinkData));
    shim.memcpyHtoD(dRec, rec.data(), rec.size());
    shim.memsetD32(dOut, 0, uint32_t(expect.size() * 2));

    struct Params { uint64_t blockData, dofData, mat33, transform, linkData, records, out; } prm = {
        uint64_t(dBlocks), uint64_t(dDofs), uint64_t(dMat33), uint64_t(dXform),
        uint64_t(dLinks), uint64_t(dRec), uint64_t(dOut)
    };
    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch artiBlockProbe");
    shim.streamSynchronize();

    std::vector<uint64_t> got(expect.size(), 0);
    shim.memcpyDtoH(got.data(), dOut, got.size() * 8);

    size_t bad = 0;
    for (size_t i = 0; i < expect.size(); ++i)
        if (got[i] != expect[i])
        {
            if (bad < 12)
                std::printf("  value %zu: got 0x%016llX want 0x%016llX\n", i,
                            (unsigned long long)got[i], (unsigned long long)expect[i]);
            ++bad;
        }

    PXG_CHECK_MSG(bad == 0, "%zu of %zu probed values wrong", bad, expect.size());
    std::printf("  articulation block layout: %zu values, element [%u] lane %u, "
                "struct stride + field offset + lane stride all pinned\n",
                expect.size(), ELEM, LANE);
}

} // namespace
