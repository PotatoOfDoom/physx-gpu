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

// pxgSlangPrepDesc — P5 Step 4: BDA mirrors of the engine's contact-
// constraint-PREP descriptor aggregates (gpusolver/include/PxgSolverCoreDesc.h)
// and of every structure the prep kernels INDEX through those descs (whose
// array STRIDE is therefore part of the ABI):
//
//   - PxgPrePrepDescGpu:            FULL 760-byte mirror.  rigidSum2 reads the
//     very last fields (mTempConstraintBlockHeader @752) and WRITES
//     numStaticBatches (@12); the pre-prep kernel atomically bumps the four
//     shared counters (@256..272).  No truncation is possible.
//   - PxgConstraintPrepareDescGpu:  FULL 320-byte mirror.  The prep kernels
//     read the last field (5.12: Dy::BiasCoefficientCollection
//     biasCoefficients @296, 20 bytes; 5.6.1 had one PxReal biasCoefficient
//     there and the struct was 304 bytes); rigidSum2 writes
//     numStatic*Batches (@216/@220) and numStaticBatches (@256).
//   - PxgSolverSharedDescIterGpu:   PxgSolverSharedDesc<IterativeSolveData>,
//     TRUNCATED after the PxgSolverSharedDescBase (96 bytes).
//     constraintContactBlockPrePrepLaunch dereferences the desc only as
//     desc[0] and reads only mBodySimBufferDeviceData (@32),
//     articulationSlabMask (@64) and deltaOutOffset (@72); the trailing
//     IterativeSolveData (14 pointers, @96..208) is untouched by the
//     converted kernels, so the mirror stops at 96.  (The TGS variant lives
//     in pxgSlangSolverDesc.slang / PxgSolverSharedDescTGSGpu and is owned
//     by the solver conversion.)
//   - Indexed element mirrors:      PxgConstraintBatchHeaderGpu (16),
//     PartitionIndexDataGpu (8), PartitionNodeDataGpu (24),
//     PxgSolverConstraintManagerConstantsGpu (8), PxsContactManagerOutputGpu
//     (48), PxContactPatchGpu (64), PxgBlockConstraintBatchGpu (1536).
//     PxContact is a bare float4 (16); PxsTorsionalFrictionData a float2 (8);
//     PxgSolverReferences / solverBodyIndices / friction-patch-count arrays
//     are bare uints; Sc::ShapeInteraction* arrays are bare uint64_ts.
//     Block structures (PxgBlockContactData, PxgBlockContactPoint,
//     PxgBlockWorkUnit, ...) come from pxgSlangPrepCommon; PxgBodySim /
//     PxgSolverBodyData / PxgSolverTxIData from pxgSlangCommon.
//
// Pointers -> uint64_t; PxU16/PxU8 packs -> masked words; nested structs
// flattened.  The layout is verified EMPIRICALLY by
// tests/testPrepDescLayout.cpp + npPrepDescProbe.slang: the REAL engine
// structs are filled with sentinels on the host and every mirrored field is
// read back through Ptr<T> (elements [1] of the indexed mirrors pin the
// strides).

module pxgSlangPrepDesc;

import pxgSlangSolverDesc; // BiasCoefficientCollectionGpu

// GPU_BUCKET_ID::eCount (gpunarrowphase/include/PxgNarrowphaseCore.h)
public static const uint PXG_GPU_BUCKET_COUNT = 38;

// mirrors PxgPrePrepDesc (PxgSolverCoreDesc.h), FULL, 760 bytes
public struct PxgPrePrepDescGpu
{
    public uint64_t blockBatches;                    // @0   PxgBlockConstraintBatch*
    public uint numBatches;                          // @8
    public uint numStaticBatches;                    // @12  (WRITTEN by rigidSum2)
    public uint numArtiBatches;                      // @16
    public uint numArtiStaticBatches;                // @20
    public uint numArtiSelfBatches;                  // @24
    public uint nbD6RigidJoints;                     // @28
    public uint nbD6ArtiJoints;                      // @32
    public uint nbTotalArtiJoints;                   // @36
    public uint numTotalContacts;                    // @40
    public uint numTotalConstraints;                 // @44
    public uint numTotalStaticContacts;              // @48
    public uint numTotalStaticConstraints;           // @52
    public uint numTotalArtiContacts;                // @56
    public uint numTotalArtiConstraints;             // @60
    public uint numTotalStaticArtiContacts;          // @64
    public uint numTotalStaticArtiConstraints;       // @68
    public uint numTotalSelfArtiContacts;            // @72
    public uint numTotalSelfArtiConstraints;         // @76
    public uint artiStaticConstraintBatchOffset;     // @80
    public uint artiStaticContactBatchOffset;        // @84
    public uint64_t blockWorkUnit;                   // @88  PxgBlockWorkUnit*
    public uint64_t blockContactData;                // @96  PxgBlockContactData*
    public uint64_t blockContactPoints;              // @104 PxgBlockContactPoint*
    public uint64_t compressedContacts;              // @112 PxContact*
    public uint64_t compressedPatches;               // @120 PxContactPatch*
    public uint64_t forceBuffer;                     // @128 PxU8*
    public uint64_t cpuCompressedContactsBase;       // @136 PxContact*
    public uint64_t cpuCompressedPatchesBase;        // @144 PxContactPatch*
    public uint64_t cpuForceBufferBase;              // @152 PxReal*
    public uint64_t blockPrepData;                   // @160 PxgBlockConstraint1DData*
    public uint64_t blockPrepVelocityData;           // @168 PxgBlockConstraint1DVelocities*
    public uint64_t blockPrepParameterData;          // @176 PxgBlockConstraint1DParameters*
    public uint64_t constraintData;                  // @184 PxgConstraintData*
    public uint64_t constraintRows;                  // @192 Px1DConstraint*
    public uint64_t artiConstraintData;              // @200 PxgConstraintData*
    public uint64_t artiConstraintRows;              // @208 Px1DConstraint*
    public uint64_t rigidJointData;                  // @216 const PxgD6JointData*
    public uint64_t rigidConstraintPrePrep;          // @224 const PxgConstraintPrePrep*
    public uint64_t artiJointData;                   // @232 const PxgD6JointData*
    public uint64_t artiConstraintPrePrep;           // @240 const PxgConstraintPrePrep*
    public uint64_t contactManagerOutputBase;        // @248 PxsContactManagerOutput*
    public uint sharedJointRowIndex;                 // @256 (atomic)
    public uint sharedFrictionConstraintIndex;       // @260 (atomic)
    public uint sharedContactConstraintIndex;        // @264 (atomic)
    public uint sharedArticulationResponseIndex;     // @268 (atomic)
    public uint64_t solverBodyIndices;               // @272 PxU32*
    public uint64_t mPartitionIndices;               // @280 PartitionIndexData*
    public uint64_t mPartitionstartBatchIndices;     // @288 PxU32*
    public uint64_t mPartitionArtiStartBatchIndices; // @296 PxU32*
    public uint64_t mPartitionJointCounts;           // @304 PxU32*
    public uint64_t mPartitionArtiJointCounts;       // @312 PxU32*
    public uint64_t prevFrictionPatchCount;          // @320 PxU32*
    public uint64_t currFrictionPatchCount;          // @328 PxU32*
    public uint64_t mNpOutputIndices;                // @336 PxU32*
    public uint64_t mSolverBodyData;                 // @344 PxgSolverBodyData*
    public uint mCmOutputOffsets[PXG_GPU_BUCKET_COUNT]; // @352..504 (38 x u32)
    public uint64_t mPartitionNodeData;              // @504 PartitionNodeData*
    public uint64_t mContactConstantData;            // @512 PxgSolverConstraintManagerConstants*
    public uint64_t mBatchHeaders;                   // @520 PxgConstraintBatchHeader*
    public uint64_t mContactUniqueIndices;           // @528 PxU32*
    public uint64_t mConstraintUniqueIndices;        // @536 PxU32*
    public uint64_t mArtiConstraintUniqueIndices;    // @544 PxU32*
    public uint64_t mArtiContactUniqueIndices;       // @552 PxU32*
    public uint64_t mSolverBodyReferences;           // @560 PxgSolverReferences*
    public uint mMaxConstraintPartitions;            // @568
    public uint mTotalSlabs;                         // @572
    public uint mTotalActiveBodies;                  // @576
    public uint mTotalActiveArticulations;           // @580
    public uint mActiveBodyStartOffset;              // @584
    public uint nbElementsPerBody;                   // @588
    public uint64_t mShapeInteractions;              // @592 Sc::ShapeInteraction**
    public uint64_t mRestDistances;                  // @600 PxReal*
    public uint64_t mTorsionalFrictionData;          // @608 PxsTorsionalFrictionData*
    public uint64_t mArtiStaticContactIndices;       // @616 PxU32*
    public uint64_t mArtiStaticConstraintIndices;    // @624 PxU32*
    public uint64_t mArtiStaticContactCounts;        // @632 PxU32*
    public uint64_t mArtiStaticConstraintCounts;     // @640 PxU32*
    public uint64_t mArtiSelfContactIndices;         // @648 PxU32*
    public uint64_t mArtiSelfConstraintIndices;      // @656 PxU32*
    public uint64_t mArtiSelfContactCounts;          // @664 PxU32*
    public uint64_t mArtiSelfConstraintCounts;       // @672 PxU32*
    public uint64_t mRigidStaticContactIndices;      // @680 PxU32*
    public uint64_t mRigidStaticConstraintIndices;   // @688 PxU32*
    public uint64_t mRigidStaticContactCounts;       // @696 PxU32*
    public uint64_t mRigidStaticConstraintCounts;    // @704 PxU32*
    public uint64_t mRigidStaticContactStartIndices; // @712 PxU32*
    public uint64_t mRigidStaticConstraintStartIndices; // @720 PxU32*
    public uint64_t mTempContactUniqueIndices;       // @728 PxU32*
    public uint64_t mTempConstraintUniqueIndices;    // @736 PxU32*
    public uint64_t mTempContactBlockHeader;         // @744 PxU32*
    public uint64_t mTempConstraintBlockHeader;      // @752 PxU32*
}                                                    // = 760

// mirrors PxgConstraintPrepareDesc (PxgSolverCoreDesc.h L289), FULL, 304 bytes
public struct PxgConstraintPrepareDescGpu
{
    public uint64_t jointConstraintBatchIndices;     // @0   PxU32*
    public uint64_t contactConstraintBatchIndices;   // @8   PxU32*
    public uint64_t artiJointConstraintBatchIndices; // @16  PxU32*
    public uint64_t artiContactConstraintBatchIndices; // @24 PxU32*
    public uint64_t solverConstantData;              // @32  PxgSolverConstraintManagerConstants*
    public uint64_t blockJointPrepPool;              // @40  PxgBlockConstraint1DData*
    public uint64_t blockJointPrepPool0;             // @48  PxgBlockConstraint1DVelocities*
    public uint64_t blockJointPrepPool1;             // @56  PxgBlockConstraint1DParameters*
    public uint64_t solverBodyDataPool;              // @64  PxgSolverBodyData*
    public uint64_t solverBodyTxIDataPool;           // @72  PxgSolverTxIData*
    public uint64_t blockWorkUnit;                   // @80  PxgBlockWorkUnit*
    public uint64_t blockCurrentFrictionIndices;     // @88  PxgBlockFrictionIndex*
    public uint64_t blockPreviousFrictionIndices;    // @96  PxgBlockFrictionIndex*
    public uint64_t blockContactCurrentPrepPool;     // @104 PxgBlockContactData*
    public uint64_t blockContactPoints;              // @112 PxgBlockContactPoint*
    public uint64_t blockCurrentAnchorPatches;       // @120 PxgBlockFrictionAnchorPatch*
    public uint64_t blockPreviousAnchorPatches;      // @128 PxgBlockFrictionAnchorPatch*
    public uint64_t currentFrictionPatches;          // @136 PxgFrictionPatch*
    public uint64_t previousFrictionPatches;         // @144 PxgFrictionPatch*
    public uint64_t currentAnchorPatches;            // @152 PxgFrictionAnchorPatch*
    public uint64_t previousAnchorPatches;           // @160 PxgFrictionAnchorPatch*
    public uint64_t body2WorldPool;                  // @168 PxAlignedTransform*
    public uint64_t contactManagerOutputBase;        // @176 PxsContactManagerOutput*
    public uint64_t constraintUniqueIndices;         // @184 PxU32*
    public uint64_t artiConstraintUniqueIndices;     // @192 PxU32*
    public uint64_t artiContactUniqueIndices;        // @200 PxU32*
    public uint num1dConstraintBatches;              // @208
    public uint numContactBatches;                   // @212
    public uint numStatic1dConstraintBatches;        // @216 (WRITTEN by rigidSum2)
    public uint numStaticContactBatches;             // @220 (WRITTEN by rigidSum2)
    public uint numArti1dConstraintBatches;          // @224
    public uint numArtiStatic1dConstraintBatches;    // @228
    public uint numArtiSelf1dConstraintBatches;      // @232
    public uint numArtiContactBatches;               // @236
    public uint numArtiStaticContactBatches;         // @240
    public uint numArtiSelfContactBatches;           // @244
    public uint totalBodyCount;                      // @248
    public uint numBatches;                          // @252
    public uint numStaticBatches;                    // @256 (WRITTEN by rigidSum2)
    public float bounceThresholdF32;                 // @260
    public float frictionOffsetThreshold;            // @264
    public float correlationDistance;                // @268
    public float ccdMaxSeparation;                   // @272
    public uint totalPreviousEdges;                  // @276
    public uint totalCurrentEdges;                   // @280
    public uint articContactIndex;                   // @284
    public uint articJointIndex;                     // @288
    public uint nbElementsPerBody;                   // @292
    public BiasCoefficientCollectionGpu biasCoefficients; // @296 (rigidContact @296, joint @300,
                                                     //  articulation @304, particle @308,
                                                     //  deformableRigidAttachment @312)
    public uint tailPad;                             // @316 (sizeof rounds to 320 on host: 8-align)
}                                                    // = 320

// mirrors PxgSolverSharedDesc<IterativeSolveData> — TRUNCATED at the end of
// PxgSolverSharedDescBase (96 bytes): the converted kernels dereference the
// desc only as desc[0] and never touch iterativeData (see file header).
public struct PxgSolverSharedDescIterGpu
{
    public uint64_t blockCurrentFrictionPatches;     // @0  PxgBlockFrictionPatch*
    public uint64_t blockPreviousFrictionPatches;    // @8
    public uint64_t currentFrictionPatches;          // @16 PxgFrictionPatch*
    public uint64_t previousFrictionPatches;         // @24
    public uint64_t mBodySimBufferDeviceData;        // @32 PxgBodySim*
    public uint64_t articulations;                   // @40 PxgArticulation*
    public uint64_t articulationDeferredZ;           // @48 Cm::UnAlignedSpatialVector*
    public uint64_t articulationDirty;               // @56 PxU32*
    public uint64_t articulationSlabMask;            // @64 uint4*
    public uint  deltaOutOffset;                     // @72
    public float dt;                                 // @76
    public float stepDt;                             // @80
    public float invDtF32;                           // @84
    public float stepInvDtF32;                       // @88
    public float lengthScale;                        // @92
    // TRUNCATED @96 — iterativeData (IterativeSolveData) not read on this path.
}

// mirrors PxgConstraintBatchHeader (PxgSolverConstraintDesc.h), 16 bytes.
// mDescStride (PxU16) | constraintType (PxU16) packed into word 0.
public struct PxgConstraintBatchHeaderGpu
{
    public uint mDescStride_constraintType;          // @0
    public uint mConstraintBatchIndex;               // @4
    public uint mStartPartitionIndex;                // @8
    public uint mask;                                // @12
}                                                    // = 16 (array stride)

// mirrors PartitionIndexData (PxgPartitionNode.h), 8 bytes.
// word0 = mPartitionIndex (PxU16) | mPatchIndex (PxU8) << 16 | mCType (PxU8) << 24
public struct PartitionIndexDataGpu
{
    public uint partition_patch_ctype;               // @0
    public uint mPartitionEntryIndex;                // @4
}                                                    // = 8 (array stride)

// mirrors PartitionNodeData (PxgPartitionNode.h), 24 bytes
public struct PartitionNodeDataGpu
{
    public uint2 mNodeIndex0;                        // @0  PxNodeIndex (x = mID, y = mLinkID)
    public uint2 mNodeIndex1;                        // @8
    public uint mNextIndex0;                         // @16
    public uint mNextIndex1;                         // @20
}                                                    // = 24 (array stride)

// mirrors PxgSolverConstraintManagerConstants (PxgSolverConstraintDesc.h)
public struct PxgSolverConstraintManagerConstantsGpu
{
    public uint mEdgeIndex;                          // @0
    public uint mConstraintWriteBackIndex;           // @4
}                                                    // = 8 (array stride)

// mirrors PxsContactManagerOutput (PxsContactManagerState.h), PX_ALIGN(16), 48 bytes.
// allflagsStart/nbPatches/statusFlag/prevPatches (4x PxU8) pack into
// patchWord; nbContacts/flags (2x PxU16) pack into contactWord.
public struct PxsContactManagerOutputGpu
{
    public uint64_t contactPatches;                  // @0  PxU8*
    public uint64_t contactPoints;                   // @8  PxU8*
    public uint64_t contactForces;                   // @16 PxReal*
    public uint64_t frictionPatches;                 // @24 PxU8*
    public uint patchWord;                           // @32 allflagsStart | nbPatches<<8 | statusFlag<<16 | prevPatches<<24
    public uint contactWord;                         // @36 nbContacts | flags<<16
    public uint2 pad;                                // @40 PxU8 pad[8]
}                                                    // = 48 (array stride)

// mirrors PxContactPatch (PxContact.h; = the .cu's shared-memory ContactPatch
// image), 64 bytes.  startContactIndex (PxU16) | nbContacts (PxU8) << 16 |
// materialFlags (PxU8) << 24 pack into cnt0; internalFlags (PxU16) |
// materialIndex0 (PxU16) << 16 into cnt1.
public struct PxContactPatchGpu
{
    public float4 mMassModification;                 // @0  invMassScale0, invInertiaScale0, invMassScale1, invInertiaScale1
    public float4 normal_restitution;                // @16 normal.xyz, restitution
    public float dynamicFriction;                    // @32
    public float staticFriction;                     // @36
    public float damping;                            // @40
    public uint cnt0;                                // @44 startContactIndex | nbContacts<<16 | materialFlags<<24
    public uint cnt1;                                // @48 internalFlags | materialIndex0<<16
    public uint cnt2;                                // @52 materialIndex1 | pad[0]<<16
    public uint2 pad;                                // @56 pad[1..4]
}                                                    // = 64 (array stride)

// mirrors PxgBlockConstraintBatch (PxgSolverConstraintDesc.h), 1536 bytes.
// Word 0 packs mDescStride (PxU16) | constraintType (PxU16); the scalar head
// is padded to the PX_ALIGN(128) of bodyANodeIndex.
public struct PxgBlockConstraintBatchGpu
{
    public uint descStride_constraintType;           // @0
    public uint blockContactIndex;                   // @4
    public uint mConstraintBatchIndex;               // @8
    public uint startConstraintIndex;                // @12
    public uint startFrictionIndex;                  // @16
    public uint mStartPartitionIndex;                // @20
    public uint mArticulationResponseIndex;          // @24
    public uint mask;                                // @28
    public uint pad0[24];                            // @32..128
    public uint2 bodyANodeIndex[32];                 // @128  PxNodeIndex
    public uint2 bodyBNodeIndex[32];                 // @384
    public uint bodyAIndex[32];                      // @640
    public uint bodyBIndex[32];                      // @768
    public uint remappedBodyAIndex[32];              // @896
    public uint remappedBodyBIndex[32];              // @1024
    public uint slabId[32];                          // @1152 (ends @1280, already 128-aligned)
    public uint64_t shapeInteraction[32];            // @1280 Sc::ShapeInteraction*
}                                                    // = 1536 (array stride)
