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

// npPrepDescProbe — P5 Step 4: the empirical layout proof for the constraint-
// prep descriptor mirrors (pxgSlangPrepDesc.slang).  The host test fills the
// REAL engine PxgPrePrepDesc / PxgConstraintPrepareDesc /
// PxgSolverSharedDesc<IterativeSolveData> plus [2]-arrays of every structure
// the prep kernels INDEX through those descs, all with distinct sentinels;
// this kernel reads every mirrored field through Ptr<T> into an output array
// the host compares bitwise.  Elements [1] of the indexed mirrors pin their
// array STRIDES.  A single wrong offset — missing padding, a drifted engine
// struct, a wrong stride — mismatches.

import pxgSlangPrepDesc;

struct PrepProbeParams
{
    uint64_t prePrepDesc;  // engine PxgPrePrepDesc
    uint64_t prepDesc;     // engine PxgConstraintPrepareDesc
    uint64_t sharedDesc;   // engine PxgSolverSharedDesc<IterativeSolveData>
    uint64_t elems;        // packed element arrays (see host test for layout)
    uint64_t outBuf;       // uint64_t[PROBE_COUNT]
}

[[vk::push_constant]]
ConstantBuffer<PrepProbeParams> gProbe;

[shader("compute")]
[numthreads(1, 1, 1)]
void prepDescProbe(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x != 0)
        return;
    PxgPrePrepDescGpu d = (Ptr<PxgPrePrepDescGpu>(gProbe.prePrepDesc))[0];
    PxgConstraintPrepareDescGpu c = (Ptr<PxgConstraintPrepareDescGpu>(gProbe.prepDesc))[0];
    PxgSolverSharedDescIterGpu s = (Ptr<PxgSolverSharedDescIterGpu>(gProbe.sharedDesc))[0];
    Ptr<uint64_t> o = Ptr<uint64_t>(gProbe.outBuf);
    uint i = 0;

    // ---- PxgPrePrepDesc: every field in declaration order ----
    o[i++] = d.blockBatches;
    o[i++] = uint64_t(d.numBatches);
    o[i++] = uint64_t(d.numStaticBatches);
    o[i++] = uint64_t(d.numArtiBatches);
    o[i++] = uint64_t(d.numArtiStaticBatches);
    o[i++] = uint64_t(d.numArtiSelfBatches);
    o[i++] = uint64_t(d.nbD6RigidJoints);
    o[i++] = uint64_t(d.nbD6ArtiJoints);
    o[i++] = uint64_t(d.nbTotalArtiJoints);
    o[i++] = uint64_t(d.numTotalContacts);
    o[i++] = uint64_t(d.numTotalConstraints);
    o[i++] = uint64_t(d.numTotalStaticContacts);
    o[i++] = uint64_t(d.numTotalStaticConstraints);
    o[i++] = uint64_t(d.numTotalArtiContacts);
    o[i++] = uint64_t(d.numTotalArtiConstraints);
    o[i++] = uint64_t(d.numTotalStaticArtiContacts);
    o[i++] = uint64_t(d.numTotalStaticArtiConstraints);
    o[i++] = uint64_t(d.numTotalSelfArtiContacts);
    o[i++] = uint64_t(d.numTotalSelfArtiConstraints);
    o[i++] = uint64_t(d.artiStaticConstraintBatchOffset);
    o[i++] = uint64_t(d.artiStaticContactBatchOffset);
    o[i++] = d.blockWorkUnit;
    o[i++] = d.blockContactData;
    o[i++] = d.blockContactPoints;
    o[i++] = d.compressedContacts;
    o[i++] = d.compressedPatches;
    o[i++] = d.forceBuffer;
    o[i++] = d.cpuCompressedContactsBase;
    o[i++] = d.cpuCompressedPatchesBase;
    o[i++] = d.cpuForceBufferBase;
    o[i++] = d.blockPrepData;
    o[i++] = d.blockPrepVelocityData;
    o[i++] = d.blockPrepParameterData;
    o[i++] = d.constraintData;
    o[i++] = d.constraintRows;
    o[i++] = d.artiConstraintData;
    o[i++] = d.artiConstraintRows;
    o[i++] = d.rigidJointData;
    o[i++] = d.rigidConstraintPrePrep;
    o[i++] = d.artiJointData;
    o[i++] = d.artiConstraintPrePrep;
    o[i++] = d.contactManagerOutputBase;
    o[i++] = uint64_t(d.sharedJointRowIndex);
    o[i++] = uint64_t(d.sharedFrictionConstraintIndex);
    o[i++] = uint64_t(d.sharedContactConstraintIndex);
    o[i++] = uint64_t(d.sharedArticulationResponseIndex);
    o[i++] = d.solverBodyIndices;
    o[i++] = d.mPartitionIndices;
    o[i++] = d.mPartitionstartBatchIndices;
    o[i++] = d.mPartitionArtiStartBatchIndices;
    o[i++] = d.mPartitionJointCounts;
    o[i++] = d.mPartitionArtiJointCounts;
    o[i++] = d.prevFrictionPatchCount;
    o[i++] = d.currFrictionPatchCount;
    o[i++] = d.mNpOutputIndices;
    o[i++] = d.mSolverBodyData;
    o[i++] = uint64_t(d.mCmOutputOffsets[0]);
    o[i++] = uint64_t(d.mCmOutputOffsets[1]);
    o[i++] = uint64_t(d.mCmOutputOffsets[19]);
    o[i++] = uint64_t(d.mCmOutputOffsets[37]);
    o[i++] = d.mPartitionNodeData;
    o[i++] = d.mContactConstantData;
    o[i++] = d.mBatchHeaders;
    o[i++] = d.mContactUniqueIndices;
    o[i++] = d.mConstraintUniqueIndices;
    o[i++] = d.mArtiConstraintUniqueIndices;
    o[i++] = d.mArtiContactUniqueIndices;
    o[i++] = d.mSolverBodyReferences;
    o[i++] = uint64_t(d.mMaxConstraintPartitions);
    o[i++] = uint64_t(d.mTotalSlabs);
    o[i++] = uint64_t(d.mTotalActiveBodies);
    o[i++] = uint64_t(d.mTotalActiveArticulations);
    o[i++] = uint64_t(d.mActiveBodyStartOffset);
    o[i++] = uint64_t(d.nbElementsPerBody);
    o[i++] = d.mShapeInteractions;
    o[i++] = d.mRestDistances;
    o[i++] = d.mTorsionalFrictionData;
    o[i++] = d.mArtiStaticContactIndices;
    o[i++] = d.mArtiStaticConstraintIndices;
    o[i++] = d.mArtiStaticContactCounts;
    o[i++] = d.mArtiStaticConstraintCounts;
    o[i++] = d.mArtiSelfContactIndices;
    o[i++] = d.mArtiSelfConstraintIndices;
    o[i++] = d.mArtiSelfContactCounts;
    o[i++] = d.mArtiSelfConstraintCounts;
    o[i++] = d.mRigidStaticContactIndices;
    o[i++] = d.mRigidStaticConstraintIndices;
    o[i++] = d.mRigidStaticContactCounts;
    o[i++] = d.mRigidStaticConstraintCounts;
    o[i++] = d.mRigidStaticContactStartIndices;
    o[i++] = d.mRigidStaticConstraintStartIndices;
    o[i++] = d.mTempContactUniqueIndices;
    o[i++] = d.mTempConstraintUniqueIndices;
    o[i++] = d.mTempContactBlockHeader;
    o[i++] = d.mTempConstraintBlockHeader;

    // ---- PxgConstraintPrepareDesc: every field in declaration order ----
    o[i++] = c.jointConstraintBatchIndices;
    o[i++] = c.contactConstraintBatchIndices;
    o[i++] = c.artiJointConstraintBatchIndices;
    o[i++] = c.artiContactConstraintBatchIndices;
    o[i++] = c.solverConstantData;
    o[i++] = c.blockJointPrepPool;
    o[i++] = c.blockJointPrepPool0;
    o[i++] = c.blockJointPrepPool1;
    o[i++] = c.solverBodyDataPool;
    o[i++] = c.solverBodyTxIDataPool;
    o[i++] = c.blockWorkUnit;
    o[i++] = c.blockCurrentFrictionIndices;
    o[i++] = c.blockPreviousFrictionIndices;
    o[i++] = c.blockContactCurrentPrepPool;
    o[i++] = c.blockContactPoints;
    o[i++] = c.blockCurrentAnchorPatches;
    o[i++] = c.blockPreviousAnchorPatches;
    o[i++] = c.currentFrictionPatches;
    o[i++] = c.previousFrictionPatches;
    o[i++] = c.currentAnchorPatches;
    o[i++] = c.previousAnchorPatches;
    o[i++] = c.body2WorldPool;
    o[i++] = c.contactManagerOutputBase;
    o[i++] = c.constraintUniqueIndices;
    o[i++] = c.artiConstraintUniqueIndices;
    o[i++] = c.artiContactUniqueIndices;
    o[i++] = uint64_t(c.num1dConstraintBatches);
    o[i++] = uint64_t(c.numContactBatches);
    o[i++] = uint64_t(c.numStatic1dConstraintBatches);
    o[i++] = uint64_t(c.numStaticContactBatches);
    o[i++] = uint64_t(c.numArti1dConstraintBatches);
    o[i++] = uint64_t(c.numArtiStatic1dConstraintBatches);
    o[i++] = uint64_t(c.numArtiSelf1dConstraintBatches);
    o[i++] = uint64_t(c.numArtiContactBatches);
    o[i++] = uint64_t(c.numArtiStaticContactBatches);
    o[i++] = uint64_t(c.numArtiSelfContactBatches);
    o[i++] = uint64_t(c.totalBodyCount);
    o[i++] = uint64_t(c.numBatches);
    o[i++] = uint64_t(c.numStaticBatches);
    o[i++] = uint64_t(asuint(c.bounceThresholdF32));
    o[i++] = uint64_t(asuint(c.frictionOffsetThreshold));
    o[i++] = uint64_t(asuint(c.correlationDistance));
    o[i++] = uint64_t(asuint(c.ccdMaxSeparation));
    o[i++] = uint64_t(c.totalPreviousEdges);
    o[i++] = uint64_t(c.totalCurrentEdges);
    o[i++] = uint64_t(c.articContactIndex);
    o[i++] = uint64_t(c.articJointIndex);
    o[i++] = uint64_t(c.nbElementsPerBody);
    // 5.12: Dy::BiasCoefficientCollection biasCoefficients (20 bytes @296)
    o[i++] = uint64_t(asuint(c.biasCoefficients.rigidContact));
    o[i++] = uint64_t(asuint(c.biasCoefficients.joint));
    o[i++] = uint64_t(asuint(c.biasCoefficients.articulation));
    o[i++] = uint64_t(asuint(c.biasCoefficients.particle));
    o[i++] = uint64_t(asuint(c.biasCoefficients.deformableRigidAttachment));

    // ---- PxgSolverSharedDesc<IterativeSolveData>: the truncated base ----
    o[i++] = s.blockCurrentFrictionPatches;
    o[i++] = s.blockPreviousFrictionPatches;
    o[i++] = s.currentFrictionPatches;
    o[i++] = s.previousFrictionPatches;
    o[i++] = s.mBodySimBufferDeviceData;
    o[i++] = s.articulations;
    o[i++] = s.articulationDeferredZ;
    o[i++] = s.articulationDirty;
    o[i++] = s.articulationSlabMask;
    o[i++] = uint64_t(s.deltaOutOffset);
    o[i++] = uint64_t(asuint(s.dt));
    o[i++] = uint64_t(asuint(s.stepDt));
    o[i++] = uint64_t(asuint(s.invDtF32));
    o[i++] = uint64_t(asuint(s.stepInvDtF32));
    o[i++] = uint64_t(asuint(s.lengthScale));

    // ---- indexed element mirrors: [0] fields + [1] stride pins ----
    // The elems buffer packs, in order (offsets fixed by the host test):
    //   @0    PxgConstraintBatchHeader[2]              (32)
    //   @32   PartitionIndexData[2]                    (16)
    //   @48   PartitionNodeData[2]                     (48)
    //   @96   PxgSolverConstraintManagerConstants[2]   (16)
    //   @112  PxsContactManagerOutput[2]               (96)
    //   @208  PxContactPatch[2]                        (128)
    //   @336  pad to 384 (16-align safety)
    //   @384  PxgBlockConstraintBatch[2]               (3072)
    const uint64_t e = gProbe.elems;

    Ptr<PxgConstraintBatchHeaderGpu> bh = Ptr<PxgConstraintBatchHeaderGpu>(e);
    o[i++] = uint64_t(bh[0].mDescStride_constraintType);
    o[i++] = uint64_t(bh[0].mConstraintBatchIndex);
    o[i++] = uint64_t(bh[0].mStartPartitionIndex);
    o[i++] = uint64_t(bh[0].mask);
    o[i++] = uint64_t(bh[1].mDescStride_constraintType);
    o[i++] = uint64_t(bh[1].mask);

    Ptr<PartitionIndexDataGpu> pid = Ptr<PartitionIndexDataGpu>(e + 32);
    o[i++] = uint64_t(pid[0].partition_patch_ctype);
    o[i++] = uint64_t(pid[0].mPartitionEntryIndex);
    o[i++] = uint64_t(pid[1].partition_patch_ctype);
    o[i++] = uint64_t(pid[1].mPartitionEntryIndex);

    Ptr<PartitionNodeDataGpu> pnd = Ptr<PartitionNodeDataGpu>(e + 48);
    o[i++] = uint64_t(pnd[0].mNodeIndex0.x) | (uint64_t(pnd[0].mNodeIndex0.y) << 32);
    o[i++] = uint64_t(pnd[0].mNodeIndex1.x) | (uint64_t(pnd[0].mNodeIndex1.y) << 32);
    o[i++] = uint64_t(pnd[0].mNextIndex0);
    o[i++] = uint64_t(pnd[0].mNextIndex1);
    o[i++] = uint64_t(pnd[1].mNodeIndex0.x) | (uint64_t(pnd[1].mNodeIndex0.y) << 32);
    o[i++] = uint64_t(pnd[1].mNextIndex1);

    Ptr<PxgSolverConstraintManagerConstantsGpu> cd = Ptr<PxgSolverConstraintManagerConstantsGpu>(e + 96);
    o[i++] = uint64_t(cd[0].mEdgeIndex);
    o[i++] = uint64_t(cd[0].mConstraintWriteBackIndex);
    o[i++] = uint64_t(cd[1].mEdgeIndex);
    o[i++] = uint64_t(cd[1].mConstraintWriteBackIndex);

    Ptr<PxsContactManagerOutputGpu> cmo = Ptr<PxsContactManagerOutputGpu>(e + 112);
    o[i++] = cmo[0].contactPatches;
    o[i++] = cmo[0].contactPoints;
    o[i++] = cmo[0].contactForces;
    o[i++] = cmo[0].frictionPatches;
    o[i++] = uint64_t(cmo[0].patchWord);
    o[i++] = uint64_t(cmo[0].contactWord);
    o[i++] = cmo[1].contactPatches;
    o[i++] = uint64_t(cmo[1].patchWord);
    o[i++] = uint64_t(cmo[1].contactWord);

    Ptr<PxContactPatchGpu> cp = Ptr<PxContactPatchGpu>(e + 208);
    o[i++] = uint64_t(asuint(cp[0].mMassModification.x)) | (uint64_t(asuint(cp[0].mMassModification.w)) << 32);
    o[i++] = uint64_t(asuint(cp[0].normal_restitution.x)) | (uint64_t(asuint(cp[0].normal_restitution.w)) << 32);
    o[i++] = uint64_t(asuint(cp[0].dynamicFriction));
    o[i++] = uint64_t(asuint(cp[0].staticFriction));
    o[i++] = uint64_t(asuint(cp[0].damping));
    o[i++] = uint64_t(cp[0].cnt0);
    o[i++] = uint64_t(cp[0].cnt1);
    o[i++] = uint64_t(asuint(cp[1].mMassModification.x));
    o[i++] = uint64_t(cp[1].cnt0);
    o[i++] = uint64_t(cp[1].cnt1);

    Ptr<PxgBlockConstraintBatchGpu> bb = Ptr<PxgBlockConstraintBatchGpu>(e + 384);
    o[i++] = uint64_t(bb[0].descStride_constraintType);
    o[i++] = uint64_t(bb[0].blockContactIndex);
    o[i++] = uint64_t(bb[0].mConstraintBatchIndex);
    o[i++] = uint64_t(bb[0].startConstraintIndex);
    o[i++] = uint64_t(bb[0].startFrictionIndex);
    o[i++] = uint64_t(bb[0].mStartPartitionIndex);
    o[i++] = uint64_t(bb[0].mArticulationResponseIndex);
    o[i++] = uint64_t(bb[0].mask);
    o[i++] = uint64_t(bb[0].bodyANodeIndex[0].x) | (uint64_t(bb[0].bodyANodeIndex[0].y) << 32);
    o[i++] = uint64_t(bb[0].bodyANodeIndex[31].x) | (uint64_t(bb[0].bodyANodeIndex[31].y) << 32);
    o[i++] = uint64_t(bb[0].bodyBNodeIndex[7].x) | (uint64_t(bb[0].bodyBNodeIndex[7].y) << 32);
    o[i++] = uint64_t(bb[0].bodyAIndex[0]);
    o[i++] = uint64_t(bb[0].bodyAIndex[31]);
    o[i++] = uint64_t(bb[0].bodyBIndex[13]);
    o[i++] = uint64_t(bb[0].remappedBodyAIndex[5]);
    o[i++] = uint64_t(bb[0].remappedBodyBIndex[29]);
    o[i++] = uint64_t(bb[0].slabId[17]);
    o[i++] = bb[0].shapeInteraction[0];
    o[i++] = bb[0].shapeInteraction[31];
    o[i++] = uint64_t(bb[1].descStride_constraintType);
    o[i++] = uint64_t(bb[1].bodyAIndex[0]);
    o[i++] = bb[1].shapeInteraction[31];
}
