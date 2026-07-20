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

// P5 Step 4: empirical layout proof for the constraint-prep descriptor
// mirrors (pxgSlangPrepDesc.slang).  Fills the REAL engine PxgPrePrepDesc,
// PxgConstraintPrepareDesc and PxgSolverSharedDesc<IterativeSolveData> (the
// exact structs the engine DMAs to the device) with distinct sentinels, plus
// [2]-element arrays of every structure the converted prep kernels INDEX
// through those descs (PxgConstraintBatchHeader, PartitionIndexData,
// PartitionNodeData, PxgSolverConstraintManagerConstants,
// PxsContactManagerOutput, PxContactPatch, PxgBlockConstraintBatch) so their
// array STRIDES are pinned by probing element [1].  The npPrepDescProbe
// kernel reads every mirrored field through Ptr<T>; the host compares
// bitwise.  The rigidSumInternalContactAndJointBatches1/2,
// constraintContactBlockPrePrepLaunch and
// contactConstraintBlockPrepareParallelLaunchTGS conversions rest on this
// proof.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "PxNodeIndex.h" // completes PxNodeIndex for the PxgNarrowphaseCore.h chain
#include "PxgSolverCoreDesc.h"
#include "PxgSolverConstraintDesc.h"
#include "PxgPartitionNode.h"
#include "PxsContactManagerState.h"
#include "PxContact.h"

#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

static std::vector<uint8_t> prepDescLoader(const char* file) { return loadShaderBlob(file); }

// Distinct 64-bit sentinel per probed pointer field.
static uint64_t sent(uint32_t idx) { return 0xC100000000000400ull + uint64_t(idx) * 0x700013ull; }
// Distinct 32-bit sentinel per probed uint field.
static uint32_t sent32(uint32_t idx) { return 0x51000047u + idx * 0x3001Du; }
// Distinct float sentinel (compared by bit pattern).
static float sentF(uint32_t idx) { return 2.25f + float(idx) * 0.015625f; }
static uint64_t bitsOf(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

PXG_TEST(prepDescLayout)
{
    using namespace physx;

    // Host-side proof that the engine structs are what the mirrors assume.
    static_assert(sizeof(PxgPrePrepDesc) == 760, "PxgPrePrepDesc drifted");
    static_assert(sizeof(PxgConstraintPrepareDesc) == 304, "PxgConstraintPrepareDesc drifted");
    static_assert(sizeof(PxgSolverSharedDesc<IterativeSolveData>) == 224, "PGS shared desc drifted");
    static_assert(sizeof(PxgConstraintBatchHeader) == 16, "PxgConstraintBatchHeader drifted");
    static_assert(sizeof(PartitionIndexData) == 8, "PartitionIndexData drifted");
    static_assert(sizeof(PartitionNodeData) == 24, "PartitionNodeData drifted");
    static_assert(sizeof(PxgSolverConstraintManagerConstants) == 8, "constants drifted");
    static_assert(sizeof(PxsContactManagerOutput) == 48, "PxsContactManagerOutput drifted");
    static_assert(sizeof(PxContactPatch) == 64, "PxContactPatch drifted");
    static_assert(sizeof(PxgBlockConstraintBatch) == 1536, "PxgBlockConstraintBatch drifted");

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npPrepDescProbe.spv", "prepDescProbe", 40u, 0 }, // { u64 x5 }
    };
    PxgCudaShim shim(dev, manifest, 1, &prepDescLoader);

    uint32_t p = 0, u = 0, f = 0;
    std::vector<uint64_t> expect;

    // ---- the REAL engine PxgPrePrepDesc ----
    PxgPrePrepDesc d;
    std::memset(&d, 0xEE, sizeof(d));
    d.blockBatches = reinterpret_cast<PxgBlockConstraintBatch*>(sent(p++));
    d.numBatches = sent32(u++);
    d.numStaticBatches = sent32(u++);
    d.numArtiBatches = sent32(u++);
    d.numArtiStaticBatches = sent32(u++);
    d.numArtiSelfBatches = sent32(u++);
    d.nbD6RigidJoints = sent32(u++);
    d.nbD6ArtiJoints = sent32(u++);
    d.nbTotalArtiJoints = sent32(u++);
    d.numTotalContacts = sent32(u++);
    d.numTotalConstraints = sent32(u++);
    d.numTotalStaticContacts = sent32(u++);
    d.numTotalStaticConstraints = sent32(u++);
    d.numTotalArtiContacts = sent32(u++);
    d.numTotalArtiConstraints = sent32(u++);
    d.numTotalStaticArtiContacts = sent32(u++);
    d.numTotalStaticArtiConstraints = sent32(u++);
    d.numTotalSelfArtiContacts = sent32(u++);
    d.numTotalSelfArtiConstraints = sent32(u++);
    d.artiStaticConstraintBatchOffset = sent32(u++);
    d.artiStaticContactBatchOffset = sent32(u++);
    d.blockWorkUnit = reinterpret_cast<PxgBlockWorkUnit*>(sent(p++));
    d.blockContactData = reinterpret_cast<PxgBlockContactData*>(sent(p++));
    d.blockContactPoints = reinterpret_cast<PxgBlockContactPoint*>(sent(p++));
    d.compressedContacts = reinterpret_cast<PxContact*>(sent(p++));
    d.compressedPatches = reinterpret_cast<PxContactPatch*>(sent(p++));
    d.forceBuffer = reinterpret_cast<PxU8*>(sent(p++));
    d.cpuCompressedContactsBase = reinterpret_cast<PxContact*>(sent(p++));
    d.cpuCompressedPatchesBase = reinterpret_cast<PxContactPatch*>(sent(p++));
    d.cpuForceBufferBase = reinterpret_cast<PxReal*>(sent(p++));
    d.blockPrepData = reinterpret_cast<PxgBlockConstraint1DData*>(sent(p++));
    d.blockPrepVelocityData = reinterpret_cast<PxgBlockConstraint1DVelocities*>(sent(p++));
    d.blockPrepParameterData = reinterpret_cast<PxgBlockConstraint1DParameters*>(sent(p++));
    d.constraintData = reinterpret_cast<PxgConstraintData*>(sent(p++));
    d.constraintRows = reinterpret_cast<Px1DConstraint*>(sent(p++));
    d.artiConstraintData = reinterpret_cast<PxgConstraintData*>(sent(p++));
    d.artiConstraintRows = reinterpret_cast<Px1DConstraint*>(sent(p++));
    d.rigidJointData = reinterpret_cast<const PxgD6JointData*>(sent(p++));
    d.rigidConstraintPrePrep = reinterpret_cast<const PxgConstraintPrePrep*>(sent(p++));
    d.artiJointData = reinterpret_cast<const PxgD6JointData*>(sent(p++));
    d.artiConstraintPrePrep = reinterpret_cast<const PxgConstraintPrePrep*>(sent(p++));
    d.contactManagerOutputBase = reinterpret_cast<PxsContactManagerOutput*>(sent(p++));
    d.sharedJointRowIndex = sent32(u++);
    d.sharedFrictionConstraintIndex = sent32(u++);
    d.sharedContactConstraintIndex = sent32(u++);
    d.sharedArticulationResponseIndex = sent32(u++);
    d.solverBodyIndices = reinterpret_cast<PxU32*>(sent(p++));
    d.mPartitionIndices = reinterpret_cast<PartitionIndexData*>(sent(p++));
    d.mPartitionstartBatchIndices = reinterpret_cast<PxU32*>(sent(p++));
    d.mPartitionArtiStartBatchIndices = reinterpret_cast<PxU32*>(sent(p++));
    d.mPartitionJointCounts = reinterpret_cast<PxU32*>(sent(p++));
    d.mPartitionArtiJointCounts = reinterpret_cast<PxU32*>(sent(p++));
    d.prevFrictionPatchCount = reinterpret_cast<PxU32*>(sent(p++));
    d.currFrictionPatchCount = reinterpret_cast<PxU32*>(sent(p++));
    d.mNpOutputIndices = reinterpret_cast<PxU32*>(sent(p++));
    d.mSolverBodyData = reinterpret_cast<PxgSolverBodyData*>(sent(p++));
    for (uint32_t k = 0; k < GPU_BUCKET_ID::eCount; ++k)
        d.mCmOutputOffsets[k] = sent32(u + k);
    const uint32_t cmBase = u;
    u += GPU_BUCKET_ID::eCount;
    d.mPartitionNodeData = reinterpret_cast<PartitionNodeData*>(sent(p++));
    d.mContactConstantData = reinterpret_cast<PxgSolverConstraintManagerConstants*>(sent(p++));
    d.mBatchHeaders = reinterpret_cast<PxgConstraintBatchHeader*>(sent(p++));
    d.mContactUniqueIndices = reinterpret_cast<PxU32*>(sent(p++));
    d.mConstraintUniqueIndices = reinterpret_cast<PxU32*>(sent(p++));
    d.mArtiConstraintUniqueIndices = reinterpret_cast<PxU32*>(sent(p++));
    d.mArtiContactUniqueIndices = reinterpret_cast<PxU32*>(sent(p++));
    d.mSolverBodyReferences = reinterpret_cast<PxgSolverReferences*>(sent(p++));
    d.mMaxConstraintPartitions = sent32(u++);
    d.mTotalSlabs = sent32(u++);
    d.mTotalActiveBodies = sent32(u++);
    d.mTotalActiveArticulations = sent32(u++);
    d.mActiveBodyStartOffset = sent32(u++);
    d.nbElementsPerBody = sent32(u++);
    d.mShapeInteractions = reinterpret_cast<Sc::ShapeInteraction**>(sent(p++));
    d.mRestDistances = reinterpret_cast<PxReal*>(sent(p++));
    d.mTorsionalFrictionData = reinterpret_cast<PxsTorsionalFrictionData*>(sent(p++));
    d.mArtiStaticContactIndices = reinterpret_cast<PxU32*>(sent(p++));
    d.mArtiStaticConstraintIndices = reinterpret_cast<PxU32*>(sent(p++));
    d.mArtiStaticContactCounts = reinterpret_cast<PxU32*>(sent(p++));
    d.mArtiStaticConstraintCounts = reinterpret_cast<PxU32*>(sent(p++));
    d.mArtiSelfContactIndices = reinterpret_cast<PxU32*>(sent(p++));
    d.mArtiSelfConstraintIndices = reinterpret_cast<PxU32*>(sent(p++));
    d.mArtiSelfContactCounts = reinterpret_cast<PxU32*>(sent(p++));
    d.mArtiSelfConstraintCounts = reinterpret_cast<PxU32*>(sent(p++));
    d.mRigidStaticContactIndices = reinterpret_cast<PxU32*>(sent(p++));
    d.mRigidStaticConstraintIndices = reinterpret_cast<PxU32*>(sent(p++));
    d.mRigidStaticContactCounts = reinterpret_cast<PxU32*>(sent(p++));
    d.mRigidStaticConstraintCounts = reinterpret_cast<PxU32*>(sent(p++));
    d.mRigidStaticContactStartIndices = reinterpret_cast<PxU32*>(sent(p++));
    d.mRigidStaticConstraintStartIndices = reinterpret_cast<PxU32*>(sent(p++));
    d.mTempContactUniqueIndices = reinterpret_cast<PxU32*>(sent(p++));
    d.mTempConstraintUniqueIndices = reinterpret_cast<PxU32*>(sent(p++));
    d.mTempContactBlockHeader = reinterpret_cast<PxU32*>(sent(p++));
    d.mTempConstraintBlockHeader = reinterpret_cast<PxU32*>(sent(p++));

    expect.push_back(uint64_t(d.blockBatches));
    expect.push_back(d.numBatches);
    expect.push_back(d.numStaticBatches);
    expect.push_back(d.numArtiBatches);
    expect.push_back(d.numArtiStaticBatches);
    expect.push_back(d.numArtiSelfBatches);
    expect.push_back(d.nbD6RigidJoints);
    expect.push_back(d.nbD6ArtiJoints);
    expect.push_back(d.nbTotalArtiJoints);
    expect.push_back(d.numTotalContacts);
    expect.push_back(d.numTotalConstraints);
    expect.push_back(d.numTotalStaticContacts);
    expect.push_back(d.numTotalStaticConstraints);
    expect.push_back(d.numTotalArtiContacts);
    expect.push_back(d.numTotalArtiConstraints);
    expect.push_back(d.numTotalStaticArtiContacts);
    expect.push_back(d.numTotalStaticArtiConstraints);
    expect.push_back(d.numTotalSelfArtiContacts);
    expect.push_back(d.numTotalSelfArtiConstraints);
    expect.push_back(d.artiStaticConstraintBatchOffset);
    expect.push_back(d.artiStaticContactBatchOffset);
    expect.push_back(uint64_t(d.blockWorkUnit));
    expect.push_back(uint64_t(d.blockContactData));
    expect.push_back(uint64_t(d.blockContactPoints));
    expect.push_back(uint64_t(d.compressedContacts));
    expect.push_back(uint64_t(d.compressedPatches));
    expect.push_back(uint64_t(d.forceBuffer));
    expect.push_back(uint64_t(d.cpuCompressedContactsBase));
    expect.push_back(uint64_t(d.cpuCompressedPatchesBase));
    expect.push_back(uint64_t(d.cpuForceBufferBase));
    expect.push_back(uint64_t(d.blockPrepData));
    expect.push_back(uint64_t(d.blockPrepVelocityData));
    expect.push_back(uint64_t(d.blockPrepParameterData));
    expect.push_back(uint64_t(d.constraintData));
    expect.push_back(uint64_t(d.constraintRows));
    expect.push_back(uint64_t(d.artiConstraintData));
    expect.push_back(uint64_t(d.artiConstraintRows));
    expect.push_back(uint64_t(d.rigidJointData));
    expect.push_back(uint64_t(d.rigidConstraintPrePrep));
    expect.push_back(uint64_t(d.artiJointData));
    expect.push_back(uint64_t(d.artiConstraintPrePrep));
    expect.push_back(uint64_t(d.contactManagerOutputBase));
    expect.push_back(d.sharedJointRowIndex);
    expect.push_back(d.sharedFrictionConstraintIndex);
    expect.push_back(d.sharedContactConstraintIndex);
    expect.push_back(d.sharedArticulationResponseIndex);
    expect.push_back(uint64_t(d.solverBodyIndices));
    expect.push_back(uint64_t(d.mPartitionIndices));
    expect.push_back(uint64_t(d.mPartitionstartBatchIndices));
    expect.push_back(uint64_t(d.mPartitionArtiStartBatchIndices));
    expect.push_back(uint64_t(d.mPartitionJointCounts));
    expect.push_back(uint64_t(d.mPartitionArtiJointCounts));
    expect.push_back(uint64_t(d.prevFrictionPatchCount));
    expect.push_back(uint64_t(d.currFrictionPatchCount));
    expect.push_back(uint64_t(d.mNpOutputIndices));
    expect.push_back(uint64_t(d.mSolverBodyData));
    expect.push_back(sent32(cmBase + 0));
    expect.push_back(sent32(cmBase + 1));
    expect.push_back(sent32(cmBase + 19));
    expect.push_back(sent32(cmBase + 37));
    expect.push_back(uint64_t(d.mPartitionNodeData));
    expect.push_back(uint64_t(d.mContactConstantData));
    expect.push_back(uint64_t(d.mBatchHeaders));
    expect.push_back(uint64_t(d.mContactUniqueIndices));
    expect.push_back(uint64_t(d.mConstraintUniqueIndices));
    expect.push_back(uint64_t(d.mArtiConstraintUniqueIndices));
    expect.push_back(uint64_t(d.mArtiContactUniqueIndices));
    expect.push_back(uint64_t(d.mSolverBodyReferences));
    expect.push_back(d.mMaxConstraintPartitions);
    expect.push_back(d.mTotalSlabs);
    expect.push_back(d.mTotalActiveBodies);
    expect.push_back(d.mTotalActiveArticulations);
    expect.push_back(d.mActiveBodyStartOffset);
    expect.push_back(d.nbElementsPerBody);
    expect.push_back(uint64_t(d.mShapeInteractions));
    expect.push_back(uint64_t(d.mRestDistances));
    expect.push_back(uint64_t(d.mTorsionalFrictionData));
    expect.push_back(uint64_t(d.mArtiStaticContactIndices));
    expect.push_back(uint64_t(d.mArtiStaticConstraintIndices));
    expect.push_back(uint64_t(d.mArtiStaticContactCounts));
    expect.push_back(uint64_t(d.mArtiStaticConstraintCounts));
    expect.push_back(uint64_t(d.mArtiSelfContactIndices));
    expect.push_back(uint64_t(d.mArtiSelfConstraintIndices));
    expect.push_back(uint64_t(d.mArtiSelfContactCounts));
    expect.push_back(uint64_t(d.mArtiSelfConstraintCounts));
    expect.push_back(uint64_t(d.mRigidStaticContactIndices));
    expect.push_back(uint64_t(d.mRigidStaticConstraintIndices));
    expect.push_back(uint64_t(d.mRigidStaticContactCounts));
    expect.push_back(uint64_t(d.mRigidStaticConstraintCounts));
    expect.push_back(uint64_t(d.mRigidStaticContactStartIndices));
    expect.push_back(uint64_t(d.mRigidStaticConstraintStartIndices));
    expect.push_back(uint64_t(d.mTempContactUniqueIndices));
    expect.push_back(uint64_t(d.mTempConstraintUniqueIndices));
    expect.push_back(uint64_t(d.mTempContactBlockHeader));
    expect.push_back(uint64_t(d.mTempConstraintBlockHeader));

    // ---- the REAL engine PxgConstraintPrepareDesc ----
    PxgConstraintPrepareDesc c;
    std::memset(&c, 0xEE, sizeof(c));
    c.jointConstraintBatchIndices = reinterpret_cast<PxU32*>(sent(p++));
    c.contactConstraintBatchIndices = reinterpret_cast<PxU32*>(sent(p++));
    c.artiJointConstraintBatchIndices = reinterpret_cast<PxU32*>(sent(p++));
    c.artiContactConstraintBatchIndices = reinterpret_cast<PxU32*>(sent(p++));
    c.solverConstantData = reinterpret_cast<PxgSolverConstraintManagerConstants*>(sent(p++));
    c.blockJointPrepPool = reinterpret_cast<PxgBlockConstraint1DData*>(sent(p++));
    c.blockJointPrepPool0 = reinterpret_cast<PxgBlockConstraint1DVelocities*>(sent(p++));
    c.blockJointPrepPool1 = reinterpret_cast<PxgBlockConstraint1DParameters*>(sent(p++));
    c.solverBodyDataPool = reinterpret_cast<PxgSolverBodyData*>(sent(p++));
    c.solverBodyTxIDataPool = reinterpret_cast<PxgSolverTxIData*>(sent(p++));
    c.blockWorkUnit = reinterpret_cast<PxgBlockWorkUnit*>(sent(p++));
    c.blockCurrentFrictionIndices = reinterpret_cast<PxgBlockFrictionIndex*>(sent(p++));
    c.blockPreviousFrictionIndices = reinterpret_cast<PxgBlockFrictionIndex*>(sent(p++));
    c.blockContactCurrentPrepPool = reinterpret_cast<PxgBlockContactData*>(sent(p++));
    c.blockContactPoints = reinterpret_cast<PxgBlockContactPoint*>(sent(p++));
    c.blockCurrentAnchorPatches = reinterpret_cast<PxgBlockFrictionAnchorPatch*>(sent(p++));
    c.blockPreviousAnchorPatches = reinterpret_cast<PxgBlockFrictionAnchorPatch*>(sent(p++));
    c.currentFrictionPatches = reinterpret_cast<PxgFrictionPatch*>(sent(p++));
    c.previousFrictionPatches = reinterpret_cast<PxgFrictionPatch*>(sent(p++));
    c.currentAnchorPatches = reinterpret_cast<PxgFrictionAnchorPatch*>(sent(p++));
    c.previousAnchorPatches = reinterpret_cast<PxgFrictionAnchorPatch*>(sent(p++));
    c.body2WorldPool = reinterpret_cast<PxAlignedTransform*>(sent(p++));
    c.contactManagerOutputBase = reinterpret_cast<PxsContactManagerOutput*>(sent(p++));
    c.constraintUniqueIndices = reinterpret_cast<PxU32*>(sent(p++));
    c.artiConstraintUniqueIndices = reinterpret_cast<PxU32*>(sent(p++));
    c.artiContactUniqueIndices = reinterpret_cast<PxU32*>(sent(p++));
    c.num1dConstraintBatches = sent32(u++);
    c.numContactBatches = sent32(u++);
    c.numStatic1dConstraintBatches = sent32(u++);
    c.numStaticContactBatches = sent32(u++);
    c.numArti1dConstraintBatches = sent32(u++);
    c.numArtiStatic1dConstraintBatches = sent32(u++);
    c.numArtiSelf1dConstraintBatches = sent32(u++);
    c.numArtiContactBatches = sent32(u++);
    c.numArtiStaticContactBatches = sent32(u++);
    c.numArtiSelfContactBatches = sent32(u++);
    c.totalBodyCount = sent32(u++);
    c.numBatches = sent32(u++);
    c.numStaticBatches = sent32(u++);
    c.bounceThresholdF32 = sentF(f++);
    c.frictionOffsetThreshold = sentF(f++);
    c.correlationDistance = sentF(f++);
    c.ccdMaxSeparation = sentF(f++);
    c.totalPreviousEdges = sent32(u++);
    c.totalCurrentEdges = sent32(u++);
    c.articContactIndex = sent32(u++);
    c.articJointIndex = sent32(u++);
    c.nbElementsPerBody = sent32(u++);
    c.biasCoefficient = sentF(f++);

    expect.push_back(uint64_t(c.jointConstraintBatchIndices));
    expect.push_back(uint64_t(c.contactConstraintBatchIndices));
    expect.push_back(uint64_t(c.artiJointConstraintBatchIndices));
    expect.push_back(uint64_t(c.artiContactConstraintBatchIndices));
    expect.push_back(uint64_t(c.solverConstantData));
    expect.push_back(uint64_t(c.blockJointPrepPool));
    expect.push_back(uint64_t(c.blockJointPrepPool0));
    expect.push_back(uint64_t(c.blockJointPrepPool1));
    expect.push_back(uint64_t(c.solverBodyDataPool));
    expect.push_back(uint64_t(c.solverBodyTxIDataPool));
    expect.push_back(uint64_t(c.blockWorkUnit));
    expect.push_back(uint64_t(c.blockCurrentFrictionIndices));
    expect.push_back(uint64_t(c.blockPreviousFrictionIndices));
    expect.push_back(uint64_t(c.blockContactCurrentPrepPool));
    expect.push_back(uint64_t(c.blockContactPoints));
    expect.push_back(uint64_t(c.blockCurrentAnchorPatches));
    expect.push_back(uint64_t(c.blockPreviousAnchorPatches));
    expect.push_back(uint64_t(c.currentFrictionPatches));
    expect.push_back(uint64_t(c.previousFrictionPatches));
    expect.push_back(uint64_t(c.currentAnchorPatches));
    expect.push_back(uint64_t(c.previousAnchorPatches));
    expect.push_back(uint64_t(c.body2WorldPool));
    expect.push_back(uint64_t(c.contactManagerOutputBase));
    expect.push_back(uint64_t(c.constraintUniqueIndices));
    expect.push_back(uint64_t(c.artiConstraintUniqueIndices));
    expect.push_back(uint64_t(c.artiContactUniqueIndices));
    expect.push_back(c.num1dConstraintBatches);
    expect.push_back(c.numContactBatches);
    expect.push_back(c.numStatic1dConstraintBatches);
    expect.push_back(c.numStaticContactBatches);
    expect.push_back(c.numArti1dConstraintBatches);
    expect.push_back(c.numArtiStatic1dConstraintBatches);
    expect.push_back(c.numArtiSelf1dConstraintBatches);
    expect.push_back(c.numArtiContactBatches);
    expect.push_back(c.numArtiStaticContactBatches);
    expect.push_back(c.numArtiSelfContactBatches);
    expect.push_back(c.totalBodyCount);
    expect.push_back(c.numBatches);
    expect.push_back(c.numStaticBatches);
    expect.push_back(bitsOf(c.bounceThresholdF32));
    expect.push_back(bitsOf(c.frictionOffsetThreshold));
    expect.push_back(bitsOf(c.correlationDistance));
    expect.push_back(bitsOf(c.ccdMaxSeparation));
    expect.push_back(c.totalPreviousEdges);
    expect.push_back(c.totalCurrentEdges);
    expect.push_back(c.articContactIndex);
    expect.push_back(c.articJointIndex);
    expect.push_back(c.nbElementsPerBody);
    expect.push_back(bitsOf(c.biasCoefficient));

    // ---- the REAL engine PxgSolverSharedDesc<IterativeSolveData> ----
    PxgSolverSharedDesc<IterativeSolveData> s;
    std::memset(&s, 0xEE, sizeof(s));
    s.blockCurrentFrictionPatches = reinterpret_cast<PxgBlockFrictionPatch*>(sent(p++));
    s.blockPreviousFrictionPatches = reinterpret_cast<PxgBlockFrictionPatch*>(sent(p++));
    s.currentFrictionPatches = reinterpret_cast<PxgFrictionPatch*>(sent(p++));
    s.previousFrictionPatches = reinterpret_cast<PxgFrictionPatch*>(sent(p++));
    s.mBodySimBufferDeviceData = reinterpret_cast<PxgBodySim*>(sent(p++));
    s.articulations = reinterpret_cast<PxgArticulation*>(sent(p++));
    s.articulationDeferredZ = reinterpret_cast<Cm::UnAlignedSpatialVector*>(sent(p++));
    s.articulationDirty = reinterpret_cast<PxU32*>(sent(p++));
    s.articulationSlabMask = reinterpret_cast<uint4*>(sent(p++));
    s.deltaOutOffset = sent32(u++);
    s.dt = sentF(f++);
    s.stepDt = sentF(f++);
    s.invDtF32 = sentF(f++);
    s.stepInvDtF32 = sentF(f++);
    s.lengthScale = sentF(f++);

    expect.push_back(uint64_t(s.blockCurrentFrictionPatches));
    expect.push_back(uint64_t(s.blockPreviousFrictionPatches));
    expect.push_back(uint64_t(s.currentFrictionPatches));
    expect.push_back(uint64_t(s.previousFrictionPatches));
    expect.push_back(uint64_t(s.mBodySimBufferDeviceData));
    expect.push_back(uint64_t(s.articulations));
    expect.push_back(uint64_t(s.articulationDeferredZ));
    expect.push_back(uint64_t(s.articulationDirty));
    expect.push_back(uint64_t(s.articulationSlabMask));
    expect.push_back(s.deltaOutOffset);
    expect.push_back(bitsOf(s.dt));
    expect.push_back(bitsOf(s.stepDt));
    expect.push_back(bitsOf(s.invDtF32));
    expect.push_back(bitsOf(s.stepInvDtF32));
    expect.push_back(bitsOf(s.lengthScale));

    // ---- indexed element arrays (strides pinned via element [1]) ----
    std::vector<uint8_t> elems(384 + 2 * sizeof(PxgBlockConstraintBatch), 0xEE);

    PxgConstraintBatchHeader bh[2];
    std::memset(bh, 0xEE, sizeof(bh));
    bh[0].mDescStride = PxU16(sent32(u++) & 0xFFFF);
    bh[0].constraintType = PxU16(sent32(u++) & 0xFFFF);
    bh[0].mConstraintBatchIndex = sent32(u++);
    bh[0].mStartPartitionIndex = sent32(u++);
    bh[0].mask = sent32(u++);
    bh[1].mDescStride = PxU16(sent32(u++) & 0xFFFF);
    bh[1].constraintType = PxU16(sent32(u++) & 0xFFFF);
    bh[1].mask = sent32(u++);
    std::memcpy(elems.data() + 0, bh, sizeof(bh));
    expect.push_back(uint64_t(bh[0].mDescStride) | (uint64_t(bh[0].constraintType) << 16));
    expect.push_back(bh[0].mConstraintBatchIndex);
    expect.push_back(bh[0].mStartPartitionIndex);
    expect.push_back(bh[0].mask);
    expect.push_back(uint64_t(bh[1].mDescStride) | (uint64_t(bh[1].constraintType) << 16));
    expect.push_back(bh[1].mask);

    PartitionIndexData pid[2];
    std::memset(pid, 0xEE, sizeof(pid));
    pid[0].mPartitionIndex = PxU16(sent32(u++) & 0xFFFF);
    pid[0].mPatchIndex = PxU8(sent32(u++) & 0xFF);
    pid[0].mCType = PxU8(sent32(u++) & 0xFF);
    pid[0].mPartitionEntryIndex = sent32(u++);
    pid[1].mPartitionIndex = PxU16(sent32(u++) & 0xFFFF);
    pid[1].mPatchIndex = PxU8(sent32(u++) & 0xFF);
    pid[1].mCType = PxU8(sent32(u++) & 0xFF);
    pid[1].mPartitionEntryIndex = sent32(u++);
    std::memcpy(elems.data() + 32, pid, sizeof(pid));
    expect.push_back(uint64_t(pid[0].mPartitionIndex) | (uint64_t(pid[0].mPatchIndex) << 16) |
                     (uint64_t(pid[0].mCType) << 24));
    expect.push_back(pid[0].mPartitionEntryIndex);
    expect.push_back(uint64_t(pid[1].mPartitionIndex) | (uint64_t(pid[1].mPatchIndex) << 16) |
                     (uint64_t(pid[1].mCType) << 24));
    expect.push_back(pid[1].mPartitionEntryIndex);

    PartitionNodeData pnd[2];
    std::memset(pnd, 0xEE, sizeof(pnd));
    pnd[0].mNodeIndex0 = PxNodeIndex(PxU64(sent(p++)));
    pnd[0].mNodeIndex1 = PxNodeIndex(PxU64(sent(p++)));
    pnd[0].mNextIndex[0] = sent32(u++);
    pnd[0].mNextIndex[1] = sent32(u++);
    pnd[1].mNodeIndex0 = PxNodeIndex(PxU64(sent(p++)));
    pnd[1].mNextIndex[1] = sent32(u++);
    std::memcpy(elems.data() + 48, pnd, sizeof(pnd));
    expect.push_back(pnd[0].mNodeIndex0.getInd());
    expect.push_back(pnd[0].mNodeIndex1.getInd());
    expect.push_back(pnd[0].mNextIndex[0]);
    expect.push_back(pnd[0].mNextIndex[1]);
    expect.push_back(pnd[1].mNodeIndex0.getInd());
    expect.push_back(pnd[1].mNextIndex[1]);

    PxgSolverConstraintManagerConstants cdc[2];
    std::memset(cdc, 0xEE, sizeof(cdc));
    cdc[0].mEdgeIndex = sent32(u++);
    cdc[0].mConstraintWriteBackIndex = sent32(u++);
    cdc[1].mEdgeIndex = sent32(u++);
    cdc[1].mConstraintWriteBackIndex = sent32(u++);
    std::memcpy(elems.data() + 96, cdc, sizeof(cdc));
    expect.push_back(cdc[0].mEdgeIndex);
    expect.push_back(cdc[0].mConstraintWriteBackIndex);
    expect.push_back(cdc[1].mEdgeIndex);
    expect.push_back(cdc[1].mConstraintWriteBackIndex);

    PxsContactManagerOutput cmo[2];
    std::memset(cmo, 0xEE, sizeof(cmo));
    cmo[0].contactPatches = reinterpret_cast<PxU8*>(sent(p++));
    cmo[0].contactPoints = reinterpret_cast<PxU8*>(sent(p++));
    cmo[0].contactForces = reinterpret_cast<PxReal*>(sent(p++));
    cmo[0].frictionPatches = reinterpret_cast<PxU8*>(sent(p++));
    cmo[0].allflagsStart = PxU8(0x11);
    cmo[0].nbPatches = PxU8(0x22);
    cmo[0].statusFlag = PxU8(0x33);
    cmo[0].prevPatches = PxU8(0x44);
    cmo[0].nbContacts = PxU16(sent32(u++) & 0xFFFF);
    cmo[0].flags = PxU16(sent32(u++) & 0xFFFF);
    cmo[1].contactPatches = reinterpret_cast<PxU8*>(sent(p++));
    cmo[1].allflagsStart = PxU8(0x55);
    cmo[1].nbPatches = PxU8(0x66);
    cmo[1].statusFlag = PxU8(0x77);
    cmo[1].prevPatches = PxU8(0x88);
    cmo[1].nbContacts = PxU16(sent32(u++) & 0xFFFF);
    cmo[1].flags = PxU16(sent32(u++) & 0xFFFF);
    std::memcpy(elems.data() + 112, cmo, sizeof(cmo));
    expect.push_back(uint64_t(cmo[0].contactPatches));
    expect.push_back(uint64_t(cmo[0].contactPoints));
    expect.push_back(uint64_t(cmo[0].contactForces));
    expect.push_back(uint64_t(cmo[0].frictionPatches));
    expect.push_back(uint64_t(0x11) | (uint64_t(0x22) << 8) | (uint64_t(0x33) << 16) | (uint64_t(0x44) << 24));
    expect.push_back(uint64_t(cmo[0].nbContacts) | (uint64_t(cmo[0].flags) << 16));
    expect.push_back(uint64_t(cmo[1].contactPatches));
    expect.push_back(uint64_t(0x55) | (uint64_t(0x66) << 8) | (uint64_t(0x77) << 16) | (uint64_t(0x88) << 24));
    expect.push_back(uint64_t(cmo[1].nbContacts) | (uint64_t(cmo[1].flags) << 16));

    PxContactPatch cpp[2];
    std::memset(cpp, 0xEE, sizeof(cpp));
    cpp[0].mMassModification.linear0 = sentF(f++);
    cpp[0].mMassModification.angular0 = sentF(f++);
    cpp[0].mMassModification.linear1 = sentF(f++);
    cpp[0].mMassModification.angular1 = sentF(f++);
    cpp[0].normal = PxVec3(sentF(f), sentF(f + 1), sentF(f + 2));
    f += 3;
    cpp[0].restitution = sentF(f++);
    cpp[0].dynamicFriction = sentF(f++);
    cpp[0].staticFriction = sentF(f++);
    cpp[0].damping = sentF(f++);
    cpp[0].startContactIndex = PxU16(sent32(u++) & 0xFFFF);
    cpp[0].nbContacts = PxU8(0x2A);
    cpp[0].materialFlags = PxU8(0x15);
    cpp[0].internalFlags = PxU16(sent32(u++) & 0xFFFF);
    cpp[0].materialIndex0 = PxU16(sent32(u++) & 0xFFFF);
    cpp[1].mMassModification.linear0 = sentF(f++);
    cpp[1].startContactIndex = PxU16(sent32(u++) & 0xFFFF);
    cpp[1].nbContacts = PxU8(0x07);
    cpp[1].materialFlags = PxU8(0x30);
    cpp[1].internalFlags = PxU16(sent32(u++) & 0xFFFF);
    cpp[1].materialIndex0 = PxU16(sent32(u++) & 0xFFFF);
    std::memcpy(elems.data() + 208, cpp, sizeof(cpp));
    expect.push_back(bitsOf(cpp[0].mMassModification.linear0) |
                     (bitsOf(cpp[0].mMassModification.angular1) << 32));
    expect.push_back(bitsOf(cpp[0].normal.x) | (bitsOf(cpp[0].restitution) << 32));
    expect.push_back(bitsOf(cpp[0].dynamicFriction));
    expect.push_back(bitsOf(cpp[0].staticFriction));
    expect.push_back(bitsOf(cpp[0].damping));
    expect.push_back(uint64_t(cpp[0].startContactIndex) | (uint64_t(0x2A) << 16) | (uint64_t(0x15) << 24));
    expect.push_back(uint64_t(cpp[0].internalFlags) | (uint64_t(cpp[0].materialIndex0) << 16));
    expect.push_back(bitsOf(cpp[1].mMassModification.linear0));
    expect.push_back(uint64_t(cpp[1].startContactIndex) | (uint64_t(0x07) << 16) | (uint64_t(0x30) << 24));
    expect.push_back(uint64_t(cpp[1].internalFlags) | (uint64_t(cpp[1].materialIndex0) << 16));

    std::vector<PxgBlockConstraintBatch> bb(2);
    std::memset(bb.data(), 0xEE, 2 * sizeof(PxgBlockConstraintBatch));
    bb[0].mDescStride = PxU16(sent32(u++) & 0xFFFF);
    bb[0].constraintType = PxU16(sent32(u++) & 0xFFFF);
    bb[0].blockContactIndex = sent32(u++);
    bb[0].mConstraintBatchIndex = sent32(u++);
    bb[0].startConstraintIndex = sent32(u++);
    bb[0].startFrictionIndex = sent32(u++);
    bb[0].mStartPartitionIndex = sent32(u++);
    bb[0].mArticulationResponseIndex = sent32(u++);
    bb[0].mask = sent32(u++);
    bb[0].bodyANodeIndex[0] = PxNodeIndex(PxU64(sent(p++)));
    bb[0].bodyANodeIndex[31] = PxNodeIndex(PxU64(sent(p++)));
    bb[0].bodyBNodeIndex[7] = PxNodeIndex(PxU64(sent(p++)));
    bb[0].bodyAIndex[0] = sent32(u++);
    bb[0].bodyAIndex[31] = sent32(u++);
    bb[0].bodyBIndex[13] = sent32(u++);
    bb[0].remappedBodyAIndex[5] = sent32(u++);
    bb[0].remappedBodyBIndex[29] = sent32(u++);
    bb[0].slabId[17] = sent32(u++);
    bb[0].shapeInteraction[0] = reinterpret_cast<Sc::ShapeInteraction*>(sent(p++));
    bb[0].shapeInteraction[31] = reinterpret_cast<Sc::ShapeInteraction*>(sent(p++));
    bb[1].mDescStride = PxU16(sent32(u++) & 0xFFFF);
    bb[1].constraintType = PxU16(sent32(u++) & 0xFFFF);
    bb[1].bodyAIndex[0] = sent32(u++);
    bb[1].shapeInteraction[31] = reinterpret_cast<Sc::ShapeInteraction*>(sent(p++));
    std::memcpy(elems.data() + 384, bb.data(), 2 * sizeof(PxgBlockConstraintBatch));
    expect.push_back(uint64_t(bb[0].mDescStride) | (uint64_t(bb[0].constraintType) << 16));
    expect.push_back(bb[0].blockContactIndex);
    expect.push_back(bb[0].mConstraintBatchIndex);
    expect.push_back(bb[0].startConstraintIndex);
    expect.push_back(bb[0].startFrictionIndex);
    expect.push_back(bb[0].mStartPartitionIndex);
    expect.push_back(bb[0].mArticulationResponseIndex);
    expect.push_back(bb[0].mask);
    expect.push_back(bb[0].bodyANodeIndex[0].getInd());
    expect.push_back(bb[0].bodyANodeIndex[31].getInd());
    expect.push_back(bb[0].bodyBNodeIndex[7].getInd());
    expect.push_back(bb[0].bodyAIndex[0]);
    expect.push_back(bb[0].bodyAIndex[31]);
    expect.push_back(bb[0].bodyBIndex[13]);
    expect.push_back(bb[0].remappedBodyAIndex[5]);
    expect.push_back(bb[0].remappedBodyBIndex[29]);
    expect.push_back(bb[0].slabId[17]);
    expect.push_back(uint64_t(bb[0].shapeInteraction[0]));
    expect.push_back(uint64_t(bb[0].shapeInteraction[31]));
    expect.push_back(uint64_t(bb[1].mDescStride) | (uint64_t(bb[1].constraintType) << 16));
    expect.push_back(bb[1].bodyAIndex[0]);
    expect.push_back(uint64_t(bb[1].shapeInteraction[31]));

    // ---- upload + probe ----
    PxgDevicePtr dPre = 0, dPrep = 0, dShared = 0, dElems = 0, dOut = 0;
    PXG_CHECK_MSG(shim.memAlloc(&dPre, sizeof(d)) == PxgResult::Success, "memAlloc preprep");
    PXG_CHECK_MSG(shim.memAlloc(&dPrep, sizeof(c)) == PxgResult::Success, "memAlloc prep");
    PXG_CHECK_MSG(shim.memAlloc(&dShared, sizeof(s)) == PxgResult::Success, "memAlloc shared");
    PXG_CHECK_MSG(shim.memAlloc(&dElems, elems.size()) == PxgResult::Success, "memAlloc elems");
    PXG_CHECK_MSG(shim.memAlloc(&dOut, expect.size() * 8) == PxgResult::Success, "memAlloc out");
    if (!dPre || !dPrep || !dShared || !dElems || !dOut)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    shim.memcpyHtoD(dPre, &d, sizeof(d));
    shim.memcpyHtoD(dPrep, &c, sizeof(c));
    shim.memcpyHtoD(dShared, &s, sizeof(s));
    shim.memcpyHtoD(dElems, elems.data(), elems.size());
    shim.memsetD32(dOut, 0, expect.size() * 2);

    struct ProbeParams { uint64_t pre, prep, shared, elems, out; } prm = { dPre, dPrep, dShared, dElems, dOut };
    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success, "launch probe");
    shim.streamSynchronize();

    std::vector<uint64_t> got(expect.size(), 0);
    shim.memcpyDtoH(got.data(), dOut, got.size() * 8);

    size_t bad = 0;
    for (size_t i = 0; i < expect.size(); ++i)
        if (got[i] != expect[i])
        {
            if (bad < 10)
                PXG_CHECK_MSG(false, "probe[%zu]: gpu=%016llx expected=%016llx", i,
                              (unsigned long long)got[i], (unsigned long long)expect[i]);
            ++bad;
        }
    PXG_CHECK_MSG(bad == 0, "prep desc mirrors are byte-exact (%zu bad of %zu)", bad, expect.size());

    for (PxgDevicePtr pf : { dPre, dPrep, dShared, dElems, dOut })
        shim.memFree(pf);
}

} // namespace
