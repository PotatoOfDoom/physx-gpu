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

// pxgSlangArticulationBlocks.slang
//
// The BLOCK-LAYOUT address map: PxgArticulationBlockData and
// PxgArticulationBlockDofData (gpusimulationcontroller/include/PxgArticulation.h)
// and the three sub-blocks they are built from.
//
// These are not ordinary structs.  Each one holds the state of THIRTY-TWO
// articulations interleaved as structure-of-arrays: every field is a `[32]`
// array indexed by the lane, and each field array is padded so the whole struct
// is a multiple of 128 bytes for coalesced loads.  So a field access is
//
//     block[warpIndex] . field [lane]
//     ^ struct stride    ^ field offset  ^ element stride
//
// — three numbers, every one of them part of the ABI, and none of them
// inferable from the others.  `PxgArticulationBlockLinkData` alone is 29184
// bytes of such fields.
//
// EVERY CONSTANT BELOW WAS GENERATED from host-side `sizeof`/`offsetof` of the
// real engine structs rather than transcribed, and each is pinned again by
// static_assert in tests/testArtiBlockLayout.cpp.  Hand-copying fifty offsets
// out of a 29 KB SoA header is exactly the kind of work that produced the BV32
// stride bug in P8.49a, where shader and fixture carried the same wrong number
// and agreed with each other instead of with the engine.
//
// The layout is proven functionally as well: the artiBlockProbe kernel reads
// element [1] of a two-element array at lane 3 — so the struct stride, the
// field offset and the lane stride are each independently wrong-able — and the
// test compares every value against the real engine struct filled with
// sentinels on the host.
module pxgSlangArticulationBlocks;

// ---------------------------------------------------------------------------
// Sub-blocks
// ---------------------------------------------------------------------------
public static const uint PXG_AB_SPATIAL_VECTOR_BLOCK_SIZE               =    768;
public static const uint PXG_AB_SVB_TOPXYZ_BX                           =      0;
public static const uint PXG_AB_SVB_BYZ                                 =    512;
public static const uint PXG_AB_MAT33_BLOCK_SIZE                        =   1152;
public static const uint PXG_AB_MAT33_COL0                              =      0;
public static const uint PXG_AB_MAT33_COL1                              =    512;
public static const uint PXG_AB_MAT33_COL2                              =   1024;
public static const uint PXG_AB_SPATIAL_MATRIX_BLOCK_SIZE               =   3584;
public static const uint PXG_AB_SPATIAL_TRANSFORM_BLOCK_SIZE            =   1024;
public static const uint PXG_AB_STB_Q                                   =      0;
public static const uint PXG_AB_STB_P                                   =    512;
// PxgArticulationBitFieldStackData is a lone `PxU64 bitField[32]`, so the field
// offset is 0 and the only number that matters is the element stride.
public static const uint PXG_AB_BITFIELD_STACK_SIZE                      =    256;

public static const uint PXG_AB_BLOCK_SIZE                              =   7168;
public static const uint PXG_AB_mInvSpatialArticulatedInertia           =      0;
public static const uint PXG_AB_mFlags                                  =   3584;
public static const uint PXG_AB_mNumLinks                               =   3712;
public static const uint PXG_AB_mNumSpatialTendons                      =   3840;
public static const uint PXG_AB_mNumFixedTendons                        =   3968;
public static const uint PXG_AB_mNumMimicJoints                         =   4096;
public static const uint PXG_AB_mTotalDofs                              =   4224;
public static const uint PXG_AB_mArticulationIndex                      =   4352;
public static const uint PXG_AB_mSleepThreshold                         =   4480;
public static const uint PXG_AB_mCommonLinkDeltaVelocity                =   4608;
public static const uint PXG_AB_mLinkWithDeferredImpulse                =   5376;
public static const uint PXG_AB_mMotionVelocitiesPtr                    =   5504;
public static const uint PXG_AB_mRootDeferredZ                          =   5760;
public static const uint PXG_AB_mCOM_TotalInvMassW                      =   6528;
public static const uint PXG_AB_mStateDirty                             =   7040;
public static const uint PXG_AB_mTotalSelfConstraintCount               =   7072;
public static const uint PXG_AB_mSelfConstraintOffset                   =   7076;

public static const uint PXG_AB_DOF_SIZE                                =  12800;
public static const uint PXG_AB_DOF_mJointPositions                     =      0;
public static const uint PXG_AB_DOF_mJointVelocities                    =    128;
public static const uint PXG_AB_DOF_mJointUnconstrainedVelocities       =    256;
public static const uint PXG_AB_DOF_mPosJointVelocities                 =    384;
public static const uint PXG_AB_DOF_mJointAccel                         =    512;
public static const uint PXG_AB_DOF_mQstZ                               =    640;
public static const uint PXG_AB_DOF_mQstZIcInternal                     =    768;
public static const uint PXG_AB_DOF_mDeferredQstZ                       =    896;
public static const uint PXG_AB_DOF_mTmpQstZ                            =   1024;
public static const uint PXG_AB_DOF_mWorldMotionMatrix                  =   1152;
public static const uint PXG_AB_DOF_mLocalMotionMatrix                  =   1920;
public static const uint PXG_AB_DOF_mIsInvDW                            =   2688;
public static const uint PXG_AB_DOF_mIsW                                =   3456;
public static const uint PXG_AB_DOF_mJointAxis                          =   4224;
public static const uint PXG_AB_DOF_mInvStIsT_x                         =   4992;
public static const uint PXG_AB_DOF_mInvStIsT_y                         =   5120;
public static const uint PXG_AB_DOF_mInvStIsT_z                         =   5248;
public static const uint PXG_AB_DOF_mDofIds                             =   5376;
public static const uint PXG_AB_DOF_mMotion                             =   5408;
public static const uint PXG_AB_DOF_mConstraintData                     =   5440;

// ---------------------------------------------------------------------------
// PxgArticulationBlockLinkData (29184 bytes) — the largest of the three, and
// the one the link loops walk.  Deferred out of the first block tranche so the
// map's infrastructure and its biggest consumer could be reviewed separately.
//
// Note the two irregular members among the otherwise uniform [32] arrays:
//   * mInvDofIds is PxU8[6][32] — a 6 x 32 table, so its element stride is 32
//     bytes per AXIS and the lane indexes within a row.  Reading it like the
//     neighbouring PxU8[32] fields silently lands on axis 0 only.
//   * mSpatialResponseMatrix is a 6x6 block (9 float4 columns, 4608 bytes), not
//     the 7-column 3584-byte PxgSpatialMatrixBlock immediately before it.
// ---------------------------------------------------------------------------
public static const uint PXG_AB_LINK_SIZE                                     =  29184;
public static const uint PXG_AB_SPATIAL_MATRIX_6X6_BLOCK_SIZE                 =   4608;
public static const uint PXG_AB_LINK_mSpatialArticulatedInertia               =      0;
public static const uint PXG_AB_LINK_mSpatialResponseMatrix                   =   3584;
public static const uint PXG_AB_LINK_mIsolatedInertia                         =   8192;
public static const uint PXG_AB_LINK_mMass                                    =   9344;
public static const uint PXG_AB_LINK_mMotionVelocity                          =   9472;
public static const uint PXG_AB_LINK_mPosMotionVelocity                       =  10240;
public static const uint PXG_AB_LINK_mDeltaMotion                             =  11008;
public static const uint PXG_AB_LINK_mScratchImpulse                          =  11776;
public static const uint PXG_AB_LINK_mScratchDeltaV                           =  12544;
public static const uint PXG_AB_LINK_mSolverSpatialDeltaVel                   =  13312;
public static const uint PXG_AB_LINK_mSolverSpatialImpulse                    =  14080;
public static const uint PXG_AB_LINK_mSolverSpatialInternalConstraintImpulse  =  14848;
public static const uint PXG_AB_LINK_mZAVector                                =  15616;
public static const uint PXG_AB_LINK_mZAIntVector                             =  16384;
public static const uint PXG_AB_LINK_mCoriolis                                =  17152;
public static const uint PXG_AB_LINK_mConstraintForces                        =  17920;
public static const uint PXG_AB_LINK_mMotionAcceleration                      =  18688;
public static const uint PXG_AB_LINK_mMotionAccelerationInternal              =  19456;
public static const uint PXG_AB_LINK_mBiasForce                               =  20224;
public static const uint PXG_AB_LINK_mDeltaScale                              =  20992;
public static const uint PXG_AB_LINK_mRw_x                                    =  21120;
public static const uint PXG_AB_LINK_mRw_y                                    =  21248;
public static const uint PXG_AB_LINK_mRw_z                                    =  21376;
public static const uint PXG_AB_LINK_mRelativeQuat                            =  21504;
public static const uint PXG_AB_LINK_mNbStaticContacts                        =  22016;
public static const uint PXG_AB_LINK_mStaticContactStartIndex                 =  22144;
public static const uint PXG_AB_LINK_mNbStaticJoints                          =  22272;
public static const uint PXG_AB_LINK_mStaticJointStartIndex                   =  22400;
public static const uint PXG_AB_LINK_mPreTransform                            =  22528;
public static const uint PXG_AB_LINK_mAccumulatedPose                         =  23552;
public static const uint PXG_AB_LINK_mChildPose                               =  24576;
public static const uint PXG_AB_LINK_mParentPose                              =  25600;
public static const uint PXG_AB_LINK_mDeltaQ                                  =  26624;
public static const uint PXG_AB_LINK_mLinDampingX_AngDampingY_maxLinVelSqZ_maxAngVelSqW =  27136;
public static const uint PXG_AB_LINK_mInvInertiaXYZ_invMassW                  =  27648;
public static const uint PXG_AB_LINK_mCfm                                     =  28160;
public static const uint PXG_AB_LINK_mChildrenOffset                          =  28288;
public static const uint PXG_AB_LINK_mNumChildren                             =  28416;
public static const uint PXG_AB_LINK_mJointOffset                             =  28544;
public static const uint PXG_AB_LINK_mParents                                 =  28672;
public static const uint PXG_AB_LINK_mDofs                                    =  28800;
public static const uint PXG_AB_LINK_mJointType                               =  28832;
public static const uint PXG_AB_LINK_mInvDofIds                               =  28864;
public static const uint PXG_AB_LINK_mDisableGravity                          =  29056;
public static const uint PXG_AB_LINK_mRetainsAcceleration                     =  29088;

// ---------------------------------------------------------------------------
// The RECORD structs the block fill reads FROM: PxgArticulationLink,
// Dy::ArticulationJointCore and Dy::ArticulationJointCoreData.  Plain
// arrays-of-structs, not SoA -- one record per link or per joint, indexed by
// link id, so only a stride and a field offset apply.
//
// PxgArticulationLink is declared TWICE, behind `#if !PX_CUDA_COMPILER`
// (PxgArticulationLink.h:54-63): the host sees `PxVec3 initialAngVel; PxReal
// penBiasClamp; PxVec3 initialLinVel; PxReal invMass;` and the device sees
// `float4 initialAngVelXYZ_penBiasClamp; float4 initialLinVelXYZ_invMassW;`
// over the SAME bytes.  The offsets below are generated from the host spelling
// because that is what `offsetof` can see; the kernels read the device one.
// Both agree at 0/12/16/28, which is why the two declarations can coexist -- and
// why a generator is safer here than reading either declaration by eye.
//
// Note also that the small integer fields are NOT all 32-bit: numPathToRoot and
// numChildren are PxU16 at 56 and 58, and disableGravity / retainsAccelerations
// are single bytes at 76 and 77.  Reading any of them as a word picks up its
// neighbour.
//
// The element STRIDE is deliberately NOT restated here: `PXG_ARTI_LINK_SIZE`
// in pxgSlangArticulationDesc is the same 80 bytes, and P8.82 collapsed the
// two after P8.74's sweep produced a tautological mutant -- swapping one
// constant for the other changed nothing, which is what a duplicate generated
// independently into two modules always looks like right up until the day one
// of them is regenerated and the other is not.  Only field offsets below.
public static const uint PXG_AR_LINK_initialAngVel                          =      0;
public static const uint PXG_AR_LINK_penBiasClamp                           =     12;
public static const uint PXG_AR_LINK_initialLinVel                          =     16;
public static const uint PXG_AR_LINK_invMass                                =     28;
public static const uint PXG_AR_LINK_offsetSlop                             =     60;
public static const uint PXG_AR_LINK_pathToRoot                             =     64;
public static const uint PXG_AR_LINK_childrenOffset                         =     52;
public static const uint PXG_AR_LINK_numChildren                            =     58;
public static const uint PXG_AR_LINK_pathToRootOffset                       =     48;
public static const uint PXG_AR_LINK_numPathToRoot                          =     56;
public static const uint PXG_AR_LINK_linearDamping                          =     40;
public static const uint PXG_AR_LINK_angularDamping                         =     44;
public static const uint PXG_AR_LINK_maxLinearVelocitySq                    =     36;
public static const uint PXG_AR_LINK_maxAngularVelocitySq                   =     32;
public static const uint PXG_AR_LINK_disableGravity                         =     76;
public static const uint PXG_AR_LINK_retainsAccelerations                   =     77;
public static const uint PXG_AR_LINK_cfmScale                               =     72;

public static const uint PXG_AR_JOINT_CORE_SIZE                             =    544;
public static const uint PXG_AR_JOINT_parentPose                            =      0;
public static const uint PXG_AR_JOINT_childPose                             =     28;
public static const uint PXG_AR_JOINT_limits                                =     56;
public static const uint PXG_AR_JOINT_drives                                =    104;
public static const uint PXG_AR_JOINT_targetP                               =    296;
public static const uint PXG_AR_JOINT_targetV                               =    320;
public static const uint PXG_AR_JOINT_armature                              =    344;
public static const uint PXG_AR_JOINT_frictionParams                        =    420;
public static const uint PXG_AR_JOINT_maxJointVelocity                      =    492;
public static const uint PXG_AR_JOINT_jointPos                              =    368;
public static const uint PXG_AR_JOINT_jointVel                              =    392;
public static const uint PXG_AR_JOINT_dofIds                                =    520;
public static const uint PXG_AR_JOINT_invDofIds                             =    532;
public static const uint PXG_AR_JOINT_motion                                =    526;
public static const uint PXG_AR_JOINT_frictionCoefficient                   =    416;
public static const uint PXG_AR_JOINT_jointOffset                           =    516;
public static const uint PXG_AR_JOINT_jCalcUpdateFrames                     =    538;
public static const uint PXG_AR_JOINT_jointType                             =    539;
public static const uint PXG_AR_DRIVE_SIZE                                  =     32;
public static const uint PXG_AR_LIMIT_SIZE                                  =      8;
public static const uint PXG_AR_FRICTION_SIZE                               =     12;

// Sub-fields of the three per-axis parameter structs the per-dof fill reads
// (P8.75).  PxPerformanceEnvelope is nested INSIDE PxArticulationDrive at +12,
// so an envelope field is drive[dofId] + PXG_AR_DRIVE_envelope + PXG_AR_ENV_*.
public static const uint PXG_AR_DRIVE_stiffness                             =      0;
public static const uint PXG_AR_DRIVE_damping                               =      4;
public static const uint PXG_AR_DRIVE_maxForce                              =      8;
public static const uint PXG_AR_DRIVE_envelope                              =     12;
public static const uint PXG_AR_DRIVE_driveType                             =     28;
public static const uint PXG_AR_ENV_SIZE                                    =     16;
public static const uint PXG_AR_ENV_maxEffort                               =      0;
public static const uint PXG_AR_ENV_maxActuatorVelocity                     =      4;
public static const uint PXG_AR_ENV_velocityDependentResistance             =      8;
public static const uint PXG_AR_ENV_speedEffortGradient                     =     12;
public static const uint PXG_AR_LIMIT_low                                   =      0;
public static const uint PXG_AR_LIMIT_high                                  =      4;
public static const uint PXG_AR_FRICTION_staticFrictionEffort               =      0;
public static const uint PXG_AR_FRICTION_dynamicFrictionEffort              =      4;
public static const uint PXG_AR_FRICTION_viscousFrictionCoefficient         =      8;

public static const uint PXG_AR_JOINT_CORE_DATA_SIZE                        =      8;
public static const uint PXG_AR_JCD_nbDof                                   =      4;
public static const uint PXG_AR_JCD_jointOffset                             =      0;

// ---------------------------------------------------------------------------
// PxgArticulationInternalConstraintData (7296 bytes), embedded in every
// PxgArticulationBlockDofData at offset 5440 -- the LAST unmapped layer of the
// articulation solver.  Four PxgSpatialVectorBlocks then thirty scalar [32]
// arrays: the drive, limit, friction and envelope parameters jcalc's per-dof
// loop fills and the solver reads back.
//
// Two things to notice before using it.  `mDriveType` is the only PxU32 array
// in a run of PxReals; `abF32` is `asfloat(abU32(...))`, so reading it the wrong
// way is only harmful once a CONSUMER treats the value as a float -- a probe
// that reads it as a float and converts straight back cannot tell, and the
// sweep mutant that tried was a tautology rather than a finding.  And
// `mLimits_LowLimitX_highLimitY` and
// `mLimitError_LowX_highY` are float2[32], so their element stride is 8 bytes,
// not 4, and they sit adjacent: reading either at a 4-byte stride lands inside
// the other.
// ---------------------------------------------------------------------------
public static const uint PXG_AB_IC_SIZE                                           =   7296;
public static const uint PXG_AB_IC_mRow0                                          =      0;
public static const uint PXG_AB_IC_mRow1                                          =    768;
public static const uint PXG_AB_IC_mDeltaVA                                       =   1536;
public static const uint PXG_AB_IC_mDeltaVB                                       =   2304;
public static const uint PXG_AB_IC_mRecipResponse                                 =   3072;
public static const uint PXG_AB_IC_mResponse                                      =   3200;
public static const uint PXG_AB_IC_mLimits_LowLimitX_highLimitY                   =   3328;
public static const uint PXG_AB_IC_mLimitError_LowX_highY                         =   3584;
public static const uint PXG_AB_IC_mHighImpulse                                   =   3840;
public static const uint PXG_AB_IC_mLowImpulse                                    =   3968;
public static const uint PXG_AB_IC_mMaxJointVelocity                              =   4096;
public static const uint PXG_AB_IC_mDriveImpulse                                  =   4224;
public static const uint PXG_AB_IC_mConstraintMaxImpulse                          =   4352;
public static const uint PXG_AB_IC_mMaxForce                                      =   4480;
public static const uint PXG_AB_IC_mDriveType                                     =   4608;
public static const uint PXG_AB_IC_mMaxEffort                                     =   4736;
public static const uint PXG_AB_IC_mMaxActuatorVelocity                           =   4864;
public static const uint PXG_AB_IC_mVelocityDependentResistance                   =   4992;
public static const uint PXG_AB_IC_mSpeedEffortGradient                           =   5120;
public static const uint PXG_AB_IC_mDriveTargetPos                                =   5248;
public static const uint PXG_AB_IC_mArmature                                      =   5376;
public static const uint PXG_AB_IC_mTargetVelPlusInitialBiasX_DriveBiasCoefficientY_VelMultiplierZ_ImpulseMultiplierW =   5504;
public static const uint PXG_AB_IC_mDriveStiffness                                =   6016;
public static const uint PXG_AB_IC_mDamping                                       =   6144;
public static const uint PXG_AB_IC_mDriveTargetVel                                =   6272;
public static const uint PXG_AB_IC_mTargetPosBias                                 =   6400;
public static const uint PXG_AB_IC_mAccumulatedFrictionImpulse                    =   6528;
public static const uint PXG_AB_IC_mMaxFrictionForce                              =   6656;
public static const uint PXG_AB_IC_mFrictionCoefficient                           =   6784;
public static const uint PXG_AB_IC_mStaticFrictionEffort                          =   6912;
public static const uint PXG_AB_IC_mDynamicFrictionEffort                         =   7040;
public static const uint PXG_AB_IC_mViscousFrictionCoefficient                    =   7168;

public static const uint PXG_AR_SUBSPACE_SIZE                                     =     80;
public static const uint PXG_AR_SUBSPACE_COLUMNS                                  =      0;
public static const uint PXG_AR_SPATIAL_VECTOR_SIZE                               =     24;

// ---------------------------------------------------------------------------
// P8.77: the TENDON and MIMIC-JOINT block structs, and the records they copy
// from.  Generated like everything else above; pinned by static_assert in
// tests/testArtiBlockLayout.cpp and read back functionally by artiBlockProbe.
//
// Two element strides appear here that no earlier articulation block used, and
// both are traps a uniform-looking `[32]` array hides:
//
//   * `PxgArticulationBlockAttachmentData::mRelativeOffset` is a `PxVec3[32]` --
//     384 bytes, so the element stride is TWELVE, not sixteen.  Every other
//     three-component field in these blocks is a float4 lane or a packed spatial
//     vector; this one is bare PxVec3s butted together.
//   * `mLinkIndex` is a `PxU16[32]` -- 64 bytes, stride two, so TWO LANES SHARE A
//     32-BIT WORD.  Same hazard as the PxU8[32] fields, one size up.
//
// One ordering trap in the records: `Dy::ArticulationMimicJointCore` is laid out
// linkA, axisA, linkB, axisB, but the CUDA copies it in the order linkA, linkB,
// axisA, axisB.  Reading it in declaration-of-the-copy order swaps two fields.
// ---------------------------------------------------------------------------
public static const uint PXG_AB_ST_SIZE                                               =    768;
public static const uint PXG_AB_ST_mNumAttachments                                    =      0;
public static const uint PXG_AB_ST_mNumConstraints                                    =    128;
public static const uint PXG_AB_ST_mStiffness                                         =    256;
public static const uint PXG_AB_ST_mDamping                                           =    384;
public static const uint PXG_AB_ST_mLimitStiffness                                    =    512;
public static const uint PXG_AB_ST_mOffset                                            =    640;
public static const uint PXG_AB_FT_SIZE                                               =   1152;
public static const uint PXG_AB_FT_mNumTendonJoints                                   =      0;
public static const uint PXG_AB_FT_mNumConstraints                                    =    128;
public static const uint PXG_AB_FT_mStiffness                                         =    256;
public static const uint PXG_AB_FT_mDamping                                           =    384;
public static const uint PXG_AB_FT_mLimitStiffness                                    =    512;
public static const uint PXG_AB_FT_mOffset                                            =    640;
public static const uint PXG_AB_FT_mLowLimit                                          =    768;
public static const uint PXG_AB_FT_mHighLimit                                         =    896;
public static const uint PXG_AB_FT_mRestLength                                        =   1024;
public static const uint PXG_AB_AT_SIZE                                               =   1344;
public static const uint PXG_AB_AT_mRelativeOffset                                    =      0;
public static const uint PXG_AB_AT_mRestDistance                                      =    384;
public static const uint PXG_AB_AT_mCoefficient                                       =    512;
public static const uint PXG_AB_AT_mChildrens                                         =    640;
public static const uint PXG_AB_AT_mLinkIndex                                         =    896;
public static const uint PXG_AB_AT_mParents                                           =    960;
public static const uint PXG_AB_AT_mLowLimit                                          =   1088;
public static const uint PXG_AB_AT_mHighLimit                                         =   1216;
public static const uint PXG_AB_TJ_SIZE                                               =    960;
public static const uint PXG_AB_TJ_mAxis                                              =      0;
public static const uint PXG_AB_TJ_mCoefficient                                       =    128;
public static const uint PXG_AB_TJ_mRecipCoefficient                                  =    256;
public static const uint PXG_AB_TJ_mChildrens                                         =    384;
public static const uint PXG_AB_TJ_mLinkIndex                                         =    640;
public static const uint PXG_AB_TJ_mParents                                           =    704;
public static const uint PXG_AB_TJ_mConstraintId                                      =    832;
public static const uint PXG_AB_MJ_SIZE                                               =   1408;
public static const uint PXG_AB_MJ_mLinkA                                             =      0;
public static const uint PXG_AB_MJ_mAxisA                                             =    128;
public static const uint PXG_AB_MJ_mLinkB                                             =    256;
public static const uint PXG_AB_MJ_mAxisB                                             =    384;
public static const uint PXG_AB_MJ_mGearRatio                                         =    512;
public static const uint PXG_AB_MJ_mOffset                                            =    640;
public static const uint PXG_AB_MJ_mNaturalFrequency                                  =    768;
public static const uint PXG_AB_MJ_mDampingRatio                                      =    896;
public static const uint PXG_AB_MJ_mInternalData                                      =   1024;
public static const uint PXG_AB_IMJ_SIZE                                              =    384;
public static const uint PXG_AB_IMJ_mDofA                                             =      0;
public static const uint PXG_AB_IMJ_mDofB                                             =    128;
public static const uint PXG_AB_IMJ_recipEffectiveInertia                             =    256;

// PxgArticulationInternalTendonConstraintData (PxgArticulation.h:545) -- the
// OUTPUT block the tendon setups fill and the tendon solver reads.  Generated
// P8.97b (gen97b), pinned in testArtiBlockLayout.cpp.
public static const uint PXG_AB_ITC_SIZE                                              =   4224;
public static const uint PXG_AB_ITC_mRow0                                             =      0;
public static const uint PXG_AB_ITC_mRow1                                             =    768;
public static const uint PXG_AB_ITC_mDeltaVB                                          =   1536;
public static const uint PXG_AB_ITC_mRecipResponse                                    =   2304;
public static const uint PXG_AB_ITC_mAccumulatedLength                                =   2432;
public static const uint PXG_AB_ITC_mBiasCoefficient                                  =   2560;
public static const uint PXG_AB_ITC_mVelMultiplier                                    =   2688;
public static const uint PXG_AB_ITC_mImpulseMultiplier                                =   2816;
public static const uint PXG_AB_ITC_mAppliedForce                                     =   2944;
public static const uint PXG_AB_ITC_mLimitBiasCoefficient                             =   3072;
public static const uint PXG_AB_ITC_mLimitImpulseMultiplier                           =   3200;
public static const uint PXG_AB_ITC_mLimitAppliedForce                                =   3328;
public static const uint PXG_AB_ITC_mLink0                                            =   3456;
public static const uint PXG_AB_ITC_mLink1                                            =   3584;
public static const uint PXG_AB_ITC_mDeltaVA                                          =   3712;
public static const uint PXG_AB_ITC_mRestDistance                                     =   3840;
public static const uint PXG_AB_ITC_mLowLimit                                         =   3968;
public static const uint PXG_AB_ITC_mHighLimit                                        =   4096;

// DyVArticulation.h:63 / DyCpuGpuArticulation.h:39 -- the tendon traversal
// bound and the response floor the tendon setups use.
public static const uint PXG_DY_ARTICULATION_TENDON_MAX_SIZE                          =     64;
public static const float PXG_DY_ARTICULATION_MIN_RESPONSE                            = 1e-12f;

// The element STRIDES of these records already live in pxgSlangArticulationDesc
// (PXG_ARTI_TENDON_SIZE, PXG_ARTI_TENDON_FIXED_ELEM_SIZE,
// PXG_ARTI_SPATIAL_TENDON_DATA_SIZE, PXG_ARTI_FIXED_TENDON_DATA_SIZE,
// PXG_ARTI_ATTACHMENT_MOD_SIZE, PXG_ARTI_TENDON_JOINT_COEFF_SIZE,
// PXG_ARTI_MIMIC_JOINT_CORE_SIZE), as do PxgArticulationTendon's three field
// offsets (PXG_TENDON_FIXED_ELEMENTS / _MOD_ELEMENTS / _NB_ELEMENTS).  Those are
// NOT restated here: a size generated independently into two modules is exactly
// the PXG_ARTI_LINK_SIZE / PXG_AR_LINK_SIZE duplicate this port carried until
// P8.82 collapsed it, and one of those is enough.  Only genuinely new field
// offsets appear below.
public static const uint PXG_AR_TEF_children                                          =      0;
public static const uint PXG_AR_TEF_parent                                            =      8;
public static const uint PXG_AR_TEF_linkInd                                           =     12;
public static const uint PXG_AR_STD_stiffness                                         =      0;
public static const uint PXG_AR_STD_damping                                           =      4;
public static const uint PXG_AR_STD_limitStiffness                                    =      8;
public static const uint PXG_AR_STD_offset                                            =     12;
public static const uint PXG_AR_FTD_stiffness                                         =      0;
public static const uint PXG_AR_FTD_damping                                           =      4;
public static const uint PXG_AR_FTD_limitStiffness                                    =      8;
public static const uint PXG_AR_FTD_offset                                            =     12;
public static const uint PXG_AR_FTD_lowLimit                                          =     16;
public static const uint PXG_AR_FTD_highLimit                                         =     20;
public static const uint PXG_AR_FTD_restLength                                        =     24;
public static const uint PXG_AR_TAD_relativeOffset                                    =      0;
public static const uint PXG_AR_TAD_restLength                                        =     12;
public static const uint PXG_AR_TAD_coefficient                                       =     16;
public static const uint PXG_AR_TAD_lowLimit                                          =     20;
public static const uint PXG_AR_TAD_highLimit                                         =     24;
public static const uint PXG_AR_TJC_coefficient                                       =      0;
public static const uint PXG_AR_TJC_recipCoefficient                                  =      4;
public static const uint PXG_AR_TJC_axis                                              =      8;
public static const uint PXG_AR_MJC_linkA                                             =      0;
public static const uint PXG_AR_MJC_linkB                                             =      8;
public static const uint PXG_AR_MJC_axisA                                             =      4;
public static const uint PXG_AR_MJC_axisB                                             =     12;
public static const uint PXG_AR_MJC_gearRatio                                         =     16;
public static const uint PXG_AR_MJC_offset                                            =     20;
public static const uint PXG_AR_MJC_naturalFrequency                                  =     24;
public static const uint PXG_AR_MJC_dampingRatio                                      =     28;

// ---------------------------------------------------------------------------
// Lane accessors.  `base` is the address of the block element (i.e. the array
// base plus warpIndex * stride); `fieldOffset` is one of the constants above.
// ---------------------------------------------------------------------------
public uint abU32(uint64_t base, uint fieldOffset, uint lane)
{
    return (Ptr<uint>(base + uint64_t(fieldOffset)))[lane];
}

public float abF32(uint64_t base, uint fieldOffset, uint lane)
{
    return asfloat(abU32(base, fieldOffset, lane));
}

public uint64_t abU64(uint64_t base, uint fieldOffset, uint lane)
{
    return (Ptr<uint64_t>(base + uint64_t(fieldOffset)))[lane];
}

// PxU8[32] fields: a byte array, read as a word plus a byte extract.  SPIR-V
// without the Int8 capability cannot load a byte directly, and requiring that
// capability for a bookkeeping flag would be a portability cost for nothing.
public uint abU8(uint64_t base, uint fieldOffset, uint lane)
{
    const uint word = (Ptr<uint>(base + uint64_t(fieldOffset)))[lane >> 2];
    return (word >> ((lane & 3u) * 8u)) & 0xFFu;
}

public float4 abF4(uint64_t base, uint fieldOffset, uint lane)
{
    return (Ptr<float4>(base + uint64_t(fieldOffset)))[lane];
}

public float2 abF2(uint64_t base, uint fieldOffset, uint lane)
{
    return (Ptr<float2>(base + uint64_t(fieldOffset)))[lane];
}

// PxVec3[32] (attachment mRelativeOffset): element stride TWELVE.  Slang's
// float3 is 16-byte aligned, so `Ptr<float3>[lane]` would stride by 16 and walk
// off the end of a 384-byte field -- the three components are read separately.
public float3 abF3(uint64_t base, uint fieldOffset, uint lane)
{
    Ptr<float> f = Ptr<float>(base + uint64_t(fieldOffset) + uint64_t(lane) * 12);
    return float3(f[0], f[1], f[2]);
}

// PxU16[32]: TWO lanes per 32-bit word
public uint abU16(uint64_t base, uint fieldOffset, uint lane)
{
    const uint word = (Ptr<uint>(base + uint64_t(fieldOffset)))[lane >> 1];
    return (word >> ((lane & 1u) * 16u)) & 0xFFFFu;
}

// a PxU32 that is NOT a [32] array -- mTotalSelfConstraintCount and
// mSelfConstraintOffset are per-BLOCK scalars sitting in the tail padding, so
// they take no lane index.  Reading one with a lane index would silently walk
// into the padding.
public uint abBlockScalar(uint64_t base, uint fieldOffset)
{
    return (Ptr<uint>(base + uint64_t(fieldOffset)))[0];
}

public void abStoreBlockScalar(uint64_t base, uint fieldOffset, uint value)
{
    (Ptr<uint>(base + uint64_t(fieldOffset)))[0] = value;
}

// PxgSpatialVectorBlock is stored as float4 mTopxyz_bx[32] + float2 mbyz[32]:
// the top three components plus the bottom X in one float4, the bottom Y and Z
// in a float2.  Returns (top.xyz, bottom.xyz) as two float3s.
public void abLoadSpatialVector(uint64_t base, uint fieldOffset, uint lane,
                                out float3 top, out float3 bottom)
{
    const float4 t = abF4(base, fieldOffset + PXG_AB_SVB_TOPXYZ_BX, lane);
    const float2 b = abF2(base, fieldOffset + PXG_AB_SVB_BYZ, lane);
    top = t.xyz;
    bottom = float3(t.w, b.x, b.y);
}

// ---------------------------------------------------------------------------
// Lane STORES (P8.73).  Mirror images of the loads above, with one exception
// that is not cosmetic -- see abStoreU8.
// ---------------------------------------------------------------------------
public void abStoreU32(uint64_t base, uint fieldOffset, uint lane, uint value)
{
    (Ptr<uint>(base + uint64_t(fieldOffset)))[lane] = value;
}

public void abStoreF32(uint64_t base, uint fieldOffset, uint lane, float value)
{
    abStoreU32(base, fieldOffset, lane, asuint(value));
}

public void abStoreU64(uint64_t base, uint fieldOffset, uint lane, uint64_t value)
{
    (Ptr<uint64_t>(base + uint64_t(fieldOffset)))[lane] = value;
}

public void abStoreF4(uint64_t base, uint fieldOffset, uint lane, float4 value)
{
    (Ptr<float4>(base + uint64_t(fieldOffset)))[lane] = value;
}

// float2[32] fields (the limit pairs) -- element stride 8, not 4
public void abStoreF2(uint64_t base, uint fieldOffset, uint lane, float2 value)
{
    (Ptr<float2>(base + uint64_t(fieldOffset)))[lane] = value;
}

// A PxU8[32] field: FOUR LANES SHARE ONE 32-BIT WORD, and they run concurrently.
// CUDA issues a native byte store here and the lanes never meet; SPIR-V without
// the Int8 capability cannot address a byte, so a naive read-modify-write would
// have three of every four lanes lose their value to whichever neighbour wrote
// last.  The and/or pair below is atomic per operation and each lane touches
// disjoint bits, so any interleaving lands the same result -- the same fix
// storeHeaderU16 (constraintPrepPGS.slang) uses for packed PxU16 header fields.
//
// NOTE this is NOT interchangeable with artiStoreU8Field in
// pxgSlangArticulationDesc.slang: that one is a plain RMW documented as
// single-writer-only, which is correct for the record structs (one thread per
// articulation record) and wrong for every block field.
public void abStoreU8(uint64_t base, uint fieldOffset, uint lane, uint value)
{
    Ptr<uint> w = Ptr<uint>(base + uint64_t(fieldOffset));
    const uint shift = (lane & 3u) * 8u;
    uint prev;
    InterlockedAnd(w[lane >> 2], ~(0xFFu << shift), prev);
    InterlockedOr(w[lane >> 2], (value & 0xFFu) << shift, prev);
}

// PxVec3[32] (attachment mRelativeOffset), element stride TWELVE -- see abF3
public void abStoreF3(uint64_t base, uint fieldOffset, uint lane, float3 value)
{
    Ptr<float> f = Ptr<float>(base + uint64_t(fieldOffset) + uint64_t(lane) * 12);
    f[0] = value.x;
    f[1] = value.y;
    f[2] = value.z;
}

// PxU16[32]: TWO lanes share a 32-bit word, so this is abStoreU8's atomic
// and/or pair with a 16-bit mask -- storeHeaderU16's shape
// (constraintPrepPGS.slang:1028) applied to a block field.
public void abStoreU16(uint64_t base, uint fieldOffset, uint lane, uint value)
{
    Ptr<uint> w = Ptr<uint>(base + uint64_t(fieldOffset));
    const uint shift = (lane & 1u) * 16u;
    uint prev;
    InterlockedAnd(w[lane >> 1], ~(0xFFFFu << shift), prev);
    InterlockedOr(w[lane >> 1], (value & 0xFFFFu) << shift, prev);
}

public void abStoreSpatialVector(uint64_t base, uint fieldOffset, uint lane,
                                 float3 top, float3 bottom)
{
    (Ptr<float4>(base + uint64_t(fieldOffset + PXG_AB_SVB_TOPXYZ_BX)))[lane] =
        float4(top.x, top.y, top.z, bottom.x);
    (Ptr<float2>(base + uint64_t(fieldOffset + PXG_AB_SVB_BYZ)))[lane] =
        float2(bottom.y, bottom.z);
}

// ---------------------------------------------------------------------------
// P8.84: the two matrix blocks.
//
// Both are packed formats, not plain SoA, and both are transcribed from the
// engine's own PX_CUDA_CALLABLE writers rather than inferred from the sizes:
// storeSpatialMatrix / storePxMat33 (PxgArticulation.h:130 / 282).  The fixture
// pins them by calling those same functions on the host, so the packing here is
// checked rather than asserted.
//
// PxgSpatialMatrixBlock is `float4 columns[7][32]` -- 27 floats of a
// Dy::SpatialMatrix (topLeft, topRight, bottomLeft, all row-major 3x3) packed
// four at a time with a single 0 in the last slot.  The BOTTOM RIGHT block is
// not stored at all: it is topLeft transposed (SpatialMatrix::getBottomRight),
// which is the whole reason this is 7 columns and not 9.
// A PxMat33 as the engine spells it: THREE COLUMNS.  `PxMat33::operator[](i)`
// returns column i, so the engine's `mat.topLeft[1][2]` is column 1's z -- not
// row 1's z.  Carrying the columns explicitly rather than a float3x3 keeps that
// convention from having to be remembered at every call site, and Slang's
// float3x3 indexes rows first, which is exactly the confusion to avoid.
public struct AbMat33
{
    public float3 c0;
    public float3 c1;
    public float3 c2;
};

public AbMat33 abMat33Zero()
{
    AbMat33 m;
    m.c0 = float3(0.0f, 0.0f, 0.0f);
    m.c1 = float3(0.0f, 0.0f, 0.0f);
    m.c2 = float3(0.0f, 0.0f, 0.0f);
    return m;
}

public void abStoreSpatialMatrix(uint64_t base, uint fieldOffset, uint lane,
                                 AbMat33 topLeft, AbMat33 topRight, AbMat33 bottomLeft)
{
    Ptr<float4> c = Ptr<float4>(base + uint64_t(fieldOffset));
    c[0u * 32u + lane] = float4(topLeft.c0.x, topLeft.c0.y, topLeft.c0.z, topLeft.c1.x);
    c[1u * 32u + lane] = float4(topLeft.c1.y, topLeft.c1.z, topLeft.c2.x, topLeft.c2.y);
    c[2u * 32u + lane] = float4(topLeft.c2.z, topRight.c0.x, topRight.c0.y, topRight.c0.z);
    c[3u * 32u + lane] = float4(topRight.c1.x, topRight.c1.y, topRight.c1.z, topRight.c2.x);
    c[4u * 32u + lane] = float4(topRight.c2.y, topRight.c2.z, bottomLeft.c0.x, bottomLeft.c0.y);
    c[5u * 32u + lane] = float4(bottomLeft.c0.z, bottomLeft.c1.x, bottomLeft.c1.y, bottomLeft.c1.z);
    c[6u * 32u + lane] = float4(bottomLeft.c2.x, bottomLeft.c2.y, bottomLeft.c2.z, 0.0f);
}

// P8.86: the load half of the pair above, mirroring loadSpatialMatrix
// (PxgArticulation.h:90).  Note it reads SEVEN float4s and reconstructs NINE
// columns' worth of matrix -- the bottom-right block is not in the block at all,
// and a caller that wants it takes topLeft's transpose (asmBottomRight).
//
// The engine writes `mat.topLeft[0][1]`, which is column 0's y: PxMat33's
// operator[] returns a COLUMN.  Hence c0.y here and not c1.x.
public void abLoadSpatialMatrix(uint64_t base, uint fieldOffset, uint lane,
                                out AbMat33 topLeft, out AbMat33 topRight, out AbMat33 bottomLeft)
{
    Ptr<float4> c = Ptr<float4>(base + uint64_t(fieldOffset));
    const float4 v0 = c[0u * 32u + lane];
    const float4 v1 = c[1u * 32u + lane];
    const float4 v2 = c[2u * 32u + lane];
    const float4 v3 = c[3u * 32u + lane];
    const float4 v4 = c[4u * 32u + lane];
    const float4 v5 = c[5u * 32u + lane];
    const float4 v6 = c[6u * 32u + lane];

    topLeft.c0 = float3(v0.x, v0.y, v0.z);
    topLeft.c1 = float3(v0.w, v1.x, v1.y);
    topLeft.c2 = float3(v1.z, v1.w, v2.x);

    topRight.c0 = float3(v2.y, v2.z, v2.w);
    topRight.c1 = float3(v3.x, v3.y, v3.z);
    topRight.c2 = float3(v3.w, v4.x, v4.y);

    bottomLeft.c0 = float3(v4.z, v4.w, v5.x);
    bottomLeft.c1 = float3(v5.y, v5.z, v5.w);
    bottomLeft.c2 = float3(v6.x, v6.y, v6.z);
}

// PxgMat33Block is `float4 mCol0[32]; float4 mCol1[32]; float mCol2[32]` -- nine
// floats packed as 4 + 4 + 1, so the THIRD array has element stride four while
// the first two have sixteen.
public static const uint PXG_AB_M33_COL0 = 0;
public static const uint PXG_AB_M33_COL1 = 512;
public static const uint PXG_AB_M33_COL2 = 1024;

public void abStorePxMat33(uint64_t base, uint fieldOffset, uint lane, AbMat33 m)
{
    (Ptr<float4>(base + uint64_t(fieldOffset + PXG_AB_M33_COL0)))[lane] =
        float4(m.c0.x, m.c0.y, m.c0.z, m.c1.x);
    (Ptr<float4>(base + uint64_t(fieldOffset + PXG_AB_M33_COL1)))[lane] =
        float4(m.c1.y, m.c1.z, m.c2.x, m.c2.y);
    (Ptr<float>(base + uint64_t(fieldOffset + PXG_AB_M33_COL2)))[lane] = m.c2.z;
}

// ---------------------------------------------------------------------------
// P8.89: the FULL 6x6, PxSpatialMatrix
// ---------------------------------------------------------------------------
//
// A different animal from the seven-block Dy::SpatialMatrix above.  This one has
// no implied block: `PxSpatialMatrix` is a plain `PxReal column[6][6]`
// (PxSpatialMatrix.h:53) and `PxgSpatialMatrix6x6Block` is `float4 columns[9][32]`
// -- 36 floats chunked four at a time, at the usual lane stride of 32.  The
// packing is therefore FLAT, not the interleaved one the seven-column form uses:
// `columns[k]` holds flat indices `4k .. 4k+3`, and flat `f` is
// `column[f / 6][f % 6]`.  Nine columns, 4608 bytes, no bottom-right trick.
//
// Held as a flat 36 rather than six float3x3s or a nested array because that is
// what the block is, and because the impulse-response code writes it a COLUMN at
// a time (`response.column[i][0..5]`) while reading it as a matrix-vector
// product -- see PxgArticulation.h:141 / 163 / 176 for the three originals the
// fixture pins these against.
public struct AbMatrix6x6
{
    public float e[36];   // e[i * 6 + j] == PxSpatialMatrix::column[i][j]
};

public float abM6Get(AbMatrix6x6 m, uint i, uint j) { return m.e[i * 6u + j]; }

public AbMatrix6x6 abMatrix6x6Zero()
{
    AbMatrix6x6 m;
    for (uint k = 0u; k < 36u; ++k)
        m.e[k] = 0.0f;
    return m;
}

public AbMatrix6x6 abLoadSpatialMatrix6x6(uint64_t base, uint fieldOffset, uint lane)
{
    Ptr<float4> c = Ptr<float4>(base + uint64_t(fieldOffset));
    AbMatrix6x6 m;
    for (uint k = 0u; k < 9u; ++k)
    {
        const float4 v = c[k * 32u + lane];
        m.e[k * 4u + 0u] = v.x;
        m.e[k * 4u + 1u] = v.y;
        m.e[k * 4u + 2u] = v.z;
        m.e[k * 4u + 3u] = v.w;
    }
    return m;
}

public void abStoreSpatialMatrix6x6(uint64_t base, uint fieldOffset, uint lane, AbMatrix6x6 m)
{
    Ptr<float4> c = Ptr<float4>(base + uint64_t(fieldOffset));
    for (uint k = 0u; k < 9u; ++k)
        c[k * 32u + lane] =
            float4(m.e[k * 4u + 0u], m.e[k * 4u + 1u], m.e[k * 4u + 2u], m.e[k * 4u + 3u]);
}

// zeroSpatialMatrix (PxgArticulation.h:176).  The engine writes nine explicit
// zeroed float4s rather than memsetting, and this lane's nine slots are the only
// ones it touches -- the other 31 lanes' are somebody else's articulation.
public void abZeroSpatialMatrix6x6(uint64_t base, uint fieldOffset, uint lane)
{
    Ptr<float4> c = Ptr<float4>(base + uint64_t(fieldOffset));
    for (uint k = 0u; k < 9u; ++k)
        c[k * 32u + lane] = float4(0.0f, 0.0f, 0.0f, 0.0f);
}

// loadPxMat33 (PxgArticulation.h:271), the mirror of the store above: the third
// array is a plain float[32], so its element stride is FOUR while the first two
// are sixteen.
public AbMat33 abLoadPxMat33(uint64_t base, uint fieldOffset, uint lane)
{
    const float4 v0 = (Ptr<float4>(base + uint64_t(fieldOffset + PXG_AB_M33_COL0)))[lane];
    const float4 v1 = (Ptr<float4>(base + uint64_t(fieldOffset + PXG_AB_M33_COL1)))[lane];
    const float v2 = (Ptr<float>(base + uint64_t(fieldOffset + PXG_AB_M33_COL2)))[lane];

    AbMat33 m;
    m.c0 = float3(v0.x, v0.y, v0.z);
    m.c1 = float3(v0.w, v1.x, v1.y);
    m.c2 = float3(v1.z, v1.w, v2);
    return m;
}

// PxgSpatialTransformBlock is float4 q[32] + float4 p[32]; the engine writes an
// explicit 0 into p.w rather than leaving it, so the port does too.
public void abStoreSpatialTransform(uint64_t base, uint fieldOffset, uint lane,
                                    float4 q, float3 p)
{
    (Ptr<float4>(base + uint64_t(fieldOffset + PXG_AB_STB_Q)))[lane] = q;
    (Ptr<float4>(base + uint64_t(fieldOffset + PXG_AB_STB_P)))[lane] =
        float4(p.x, p.y, p.z, 0.0f);
}

// ---------------------------------------------------------------------------
// P8.100c: PxgArticulationTraversalStackData (PxgArticulation.h:340) — the
// per-link DFS stack artiSolveInternalConstraints walks: PxU32 indices[32]
// at 0, then two PxgSpatialVectorBlock frames (768 bytes each: float4
// mTopxyz_bx[32] + float2 mbyz[32], the layout every abLoad/StoreSpatialVector
// already speaks).  Pinned by static_assert in testArtiSolveDFS.cpp.
// ---------------------------------------------------------------------------
public static const uint PXG_AB_TRAVERSAL_STACK_SIZE          = 1664;
public static const uint PXG_AB_TRAVERSAL_indices             =    0;
public static const uint PXG_AB_TRAVERSAL_impulseStack        =  128;
public static const uint PXG_AB_TRAVERSAL_deltaVStack         =  896;
