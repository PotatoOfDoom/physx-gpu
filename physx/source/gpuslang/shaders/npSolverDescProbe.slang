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

// npSolverDescProbe — P5 Step 4: the empirical layout proof for the solver
// descriptor mirrors (pxgSlangSolverDesc.slang).  The host test fills REAL
// engine-side PxgSolverCoreDesc / PxgSolverSharedDesc<IterativeSolveDataTGS> /
// PxgIslandContext[2] structs with distinct sentinels, uploads them, and this
// kernel reads fields spanning all three mirrors into an output array the
// host compares bitwise.  A single wrong offset (missing padding, wrong
// truncation point, drifted engine struct, wrong island stride) mismatches.

import pxgSlangSolverDesc;

struct SolverProbeParams
{
    uint64_t coreDesc;   // device address of the engine PxgSolverCoreDesc
    uint64_t sharedDesc; // device address of the engine PxgSolverSharedDesc<IterativeSolveDataTGS>
    uint64_t islandCtx;  // device address of an engine PxgIslandContext[2]
    uint64_t out;        // device address of uint64_t[PROBE_COUNT]
}

[[vk::push_constant]]
ConstantBuffer<SolverProbeParams> gProbe;

[shader("compute")]
[numthreads(1, 1, 1)]
void solverDescProbe(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x != 0)
        return;
    PxgSolverCoreDescGpu d = (Ptr<PxgSolverCoreDescGpu>(gProbe.coreDesc))[0];
    PxgSolverSharedDescTGSGpu s = (Ptr<PxgSolverSharedDescTGSGpu>(gProbe.sharedDesc))[0];
    Ptr<PxgIslandContextGpu> isl = Ptr<PxgIslandContextGpu>(gProbe.islandCtx);
    Ptr<uint64_t> o = Ptr<uint64_t>(gProbe.out);
    uint i = 0;

    // ---- PxgSolverCoreDesc: all 32 pointers in order ----
    o[i++] = d.outSolverVelocity;
    o[i++] = d.outBody2World;
    o[i++] = d.solverBodyDataPool;
    o[i++] = d.solverBodyTxIDataPool;
    o[i++] = d.solverBodySleepDataPool;
    o[i++] = d.outArtiVelocity;
    o[i++] = d.islandContextPool;
    o[i++] = d.motionVelocityArray;
    o[i++] = d.constraintsPerPartition;
    o[i++] = d.artiConstraintsPerPartition;
    o[i++] = d.thresholdStream;
    o[i++] = d.tmpThresholdStream;
    o[i++] = d.exceededForceElements;
    o[i++] = d.prevExceededForceElements;
    o[i++] = d.forceChangeThresholdElements;
    o[i++] = d.thresholdStreamAccumulatedForce;
    o[i++] = d.thresholdStreamAccumulatedForceBetweenBlocks;
    o[i++] = d.thresholdStreamWriteIndex;
    o[i++] = d.thresholdStreamWriteIndexBetweenBlocks;
    o[i++] = d.thresholdStreamWriteable;
    o[i++] = d.accumulatedForceObjectPairs;
    o[i++] = d.constraintWriteBack;
    o[i++] = d.forceBuffer;
    o[i++] = d.frictionPatches;
    o[i++] = d.mRigidStaticContactCounts;
    o[i++] = d.mRigidStaticContactStartIndices;
    o[i++] = d.mRigidStaticJointCounts;
    o[i++] = d.mRigidStaticJointStartIndices;
    o[i++] = d.solverBodyReferences;
    o[i++] = d.contactManagerOutputBase;
    o[i++] = d.mBodySimBufferDeviceData;
    o[i++] = d.mBodySimPrevVelocitiesBufferDeviceData;
    // ---- the scalar tail up to the truncation point ----
    o[i++] = uint64_t(d.numIslands);
    o[i++] = uint64_t(d.numBatches);
    o[i++] = uint64_t(d.numArticBatches);
    o[i++] = uint64_t(d.numSolverBodies);
    o[i++] = uint64_t(d.numSlabs);
    o[i++] = uint64_t(d.accumulatedBodyDeltaVOffset);
    o[i++] = uint64_t(uint(d.sharedThresholdStreamIndex));
    o[i++] = uint64_t(d.enableStabilizationWord & 0xffu); // the trailing bool byte

    // ---- PxgSolverSharedDesc<IterativeSolveDataTGS> ----
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
    o[i++] = s.iter_blockConstraintBatch;
    o[i++] = s.iter_blockJointConstraintHeaders;
    o[i++] = s.iter_blockJointConstraintRowsCon;
    o[i++] = s.iter_blockJointConstraintRowsMod;
    o[i++] = s.iter_blockContactHeaders;
    o[i++] = s.iter_blockFrictionHeaders;
    o[i++] = s.iter_blockContactPoints;
    o[i++] = s.iter_blockFrictions;
    o[i++] = s.iter_solverBodyVelPool;
    o[i++] = s.iter_tempStaticBodyOutputs;
    o[i++] = s.iter_solverEncodedReferenceCount;
    o[i++] = s.iter_contactHeaders;
    o[i++] = s.iter_frictionHeaders;
    o[i++] = s.iter_contactPoints;
    o[i++] = s.iter_frictions;
    o[i++] = s.iter_artiResponse;

    // ---- PxgIslandContext[2]: fields + the 68-byte array stride ----
    o[i++] = uint64_t(isl[0].mBodyStartIndex);
    o[i++] = uint64_t(isl[0].mBodyCount);
    o[i++] = uint64_t(isl[0].mArticulationCount);
    o[i++] = uint64_t(uint(isl[0].mNumVelocityIterations));
    o[i++] = uint64_t(isl[0].mStaticRigidBatchCount);
    o[i++] = uint64_t(asuint(isl[0].mBiasCoefficient));
    o[i++] = uint64_t(isl[1].mBodyStartIndex);
    o[i++] = uint64_t(isl[1].mBodyCount);
    o[i++] = uint64_t(asuint(isl[1].mBiasCoefficient));
}
