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

// P5 Step 4: empirical layout proof for the solver descriptor mirrors
// (pxgSlangSolverDesc.slang).  Fills the REAL engine PxgSolverCoreDesc,
// PxgSolverSharedDesc<IterativeSolveDataTGS> and a PxgIslandContext[2] array
// (the exact structs the engine DMAs to the device) with distinct sentinels,
// uploads the raw bytes, and the solverDescProbe kernel reads 87 fields
// spanning all three structs through the Slang mirrors.  A single wrong
// offset — missing padding, engine struct drift, a wrong PxgIslandContext
// array stride — mismatches bitwise.  Every solver
// desc-walking kernel conversion (ZeroBodiesTGS, initializeSolverVelocitiesTGS,
// computeAverage/propagateAverageSolverBodyVelocityTGS, writeBackBodiesTGS,
// integrateCoreParallelLaunchTGS) rests on this proof.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "PxNodeIndex.h" // completes PxNodeIndex for the PxgNarrowphaseCore.h chain
#include "PxgSolverCoreDesc.h"
#include "PxgIslandContext.h"

#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

static std::vector<uint8_t> solverDescLoader(const char* file) { return loadShaderBlob(file); }

// Distinct 64-bit sentinel per probed pointer field.
static uint64_t sent(uint32_t idx) { return 0xB000000000000200ull + uint64_t(idx) * 0x500009ull; }
// Distinct 32-bit sentinel per probed uint field.
static uint32_t sent32(uint32_t idx) { return 0x60000031u + idx * 0x2000Bu; }
// Distinct float sentinel (compared by bit pattern).
static float sentF(uint32_t idx)
{
    return 1.5f + float(idx) * 0.03125f;
}
static uint64_t bitsOf(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

PXG_TEST(solverDescLayout)
{
    using namespace physx;

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npSolverDescProbe.spv", "solverDescProbe", 32u, 0 }, // { u64 x4 }
    };
    PxgCudaShim shim(dev, manifest, 1, &solverDescLoader);

    uint32_t p = 0, u = 0, f = 0;

    // ---- the REAL engine PxgSolverCoreDesc (0xEE background catches any
    // field the fill misses) ----
    PxgSolverCoreDesc d;
    std::memset(&d, 0xEE, sizeof(d));
    d.outSolverVelocity = reinterpret_cast<float4*>(sent(p++));
    d.outBody2World = reinterpret_cast<PxAlignedTransform*>(sent(p++));
    d.solverBodyDataPool = reinterpret_cast<PxgSolverBodyData*>(sent(p++));
    d.solverBodyTxIDataPool = reinterpret_cast<PxgSolverTxIData*>(sent(p++));
    d.solverBodySleepDataPool = reinterpret_cast<PxgSolverBodySleepData*>(sent(p++));
    d.outArtiVelocity = reinterpret_cast<float4*>(sent(p++));
    d.islandContextPool = reinterpret_cast<PxgIslandContext*>(sent(p++));
    d.motionVelocityArray = reinterpret_cast<float4*>(sent(p++));
    d.constraintsPerPartition = reinterpret_cast<PxU32*>(sent(p++));
    d.artiConstraintsPerPartition = reinterpret_cast<PxU32*>(sent(p++));
    d.thresholdStream = reinterpret_cast<Dy::ThresholdStreamElement*>(sent(p++));
    d.tmpThresholdStream = reinterpret_cast<Dy::ThresholdStreamElement*>(sent(p++));
    d.exceededForceElements = reinterpret_cast<Dy::ThresholdStreamElement*>(sent(p++));
    d.prevExceededForceElements = reinterpret_cast<Dy::ThresholdStreamElement*>(sent(p++));
    d.forceChangeThresholdElements = reinterpret_cast<Dy::ThresholdStreamElement*>(sent(p++));
    d.thresholdStreamAccumulatedForce = reinterpret_cast<PxReal*>(sent(p++));
    d.thresholdStreamAccumulatedForceBetweenBlocks = reinterpret_cast<PxReal*>(sent(p++));
    d.thresholdStreamWriteIndex = reinterpret_cast<PxU32*>(sent(p++));
    d.thresholdStreamWriteIndexBetweenBlocks = reinterpret_cast<PxU32*>(sent(p++));
    d.thresholdStreamWriteable = reinterpret_cast<bool*>(sent(p++));
    d.accumulatedForceObjectPairs = reinterpret_cast<PxReal*>(sent(p++));
    d.constraintWriteBack = reinterpret_cast<PxgConstraintWriteback*>(sent(p++));
    d.forceBuffer = reinterpret_cast<PxF32*>(sent(p++));
    d.frictionPatches = reinterpret_cast<PxFrictionPatch*>(sent(p++));
    d.mRigidStaticContactCounts = reinterpret_cast<PxU32*>(sent(p++));
    d.mRigidStaticContactStartIndices = reinterpret_cast<PxU32*>(sent(p++));
    d.mRigidStaticJointCounts = reinterpret_cast<PxU32*>(sent(p++));
    d.mRigidStaticJointStartIndices = reinterpret_cast<PxU32*>(sent(p++));
    d.solverBodyReferences = reinterpret_cast<PxgSolverReferences*>(sent(p++));
    d.contactManagerOutputBase = reinterpret_cast<PxsContactManagerOutput*>(sent(p++));
    d.mBodySimBufferDeviceData = reinterpret_cast<PxgBodySim*>(sent(p++));
    d.mBodySimPrevVelocitiesBufferDeviceData = reinterpret_cast<PxgBodySimVelocities*>(sent(p++));
    d.numIslands = sent32(u++);
    d.numBatches = sent32(u++);
    d.numArticBatches = sent32(u++);
    d.numSolverBodies = sent32(u++);
    d.numSlabs = sent32(u++);
    d.accumulatedBodyDeltaVOffset = sent32(u++);
    d.sharedThresholdStreamIndex = -1234567; // PxI32, negative on purpose
    d.enableStabilization = true;
    d.nbExceededThresholdElements = sent32(u++);
    d.nbPrevExceededThresholdElements = sent32(u++);
    d.nbForceChangeElements = sent32(u++);
    d.maxLinksPerArticulation = sent32(u++);
    d.contactErrorAccumulator.mErrorSumOfSquares = sentF(f++);
    d.contactErrorAccumulator.mCounter = -55667788; // PxI32, negative on purpose
    d.contactErrorAccumulator.mMaxError = sentF(f++);

    // ---- the REAL engine PxgSolverSharedDesc<IterativeSolveDataTGS> ----
    PxgSolverSharedDesc<IterativeSolveDataTGS> s;
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
    s.iterativeData.blockConstraintBatch = reinterpret_cast<PxgBlockConstraintBatch*>(sent(p++));
    s.iterativeData.blockJointConstraintHeaders = reinterpret_cast<PxgTGSBlockSolverConstraint1DHeader*>(sent(p++));
    s.iterativeData.blockJointConstraintRowsCon = reinterpret_cast<PxgTGSBlockSolverConstraint1DCon*>(sent(p++));
    s.iterativeData.blockJointConstraintRowsMod = reinterpret_cast<PxgBlockSolverConstraint1DMod*>(sent(p++));
    s.iterativeData.blockContactHeaders = reinterpret_cast<PxgTGSBlockSolverContactHeader*>(sent(p++));
    s.iterativeData.blockFrictionHeaders = reinterpret_cast<PxgTGSBlockSolverFrictionHeader*>(sent(p++));
    s.iterativeData.blockContactPoints = reinterpret_cast<PxgTGSBlockSolverContactPoint*>(sent(p++));
    s.iterativeData.blockFrictions = reinterpret_cast<PxgTGSBlockSolverContactFriction*>(sent(p++));
    s.iterativeData.solverBodyVelPool = reinterpret_cast<float4*>(sent(p++));
    s.iterativeData.tempStaticBodyOutputs = reinterpret_cast<float4*>(sent(p++));
    s.iterativeData.solverEncodedReferenceCount = reinterpret_cast<PxU32*>(sent(p++));
    s.iterativeData.contactHeaders = reinterpret_cast<PxgTGSSolverContactHeader*>(sent(p++));
    s.iterativeData.frictionHeaders = reinterpret_cast<PxgSolverFrictionHeader*>(sent(p++));
    s.iterativeData.contactPoints = reinterpret_cast<PxgTGSSolverContactPointExt*>(sent(p++));
    s.iterativeData.frictions = reinterpret_cast<PxgTGSSolverFrictionExt*>(sent(p++));
    s.iterativeData.artiResponse = reinterpret_cast<PxgArticulationBlockResponse*>(sent(p++));

    // ---- a REAL engine PxgIslandContext[2] (the probe reads element [1] to
    // pin the 68-byte array stride) ----
    PxgIslandContext isl[2];
    std::memset(isl, 0xEE, sizeof(isl));
    isl[0].mBodyStartIndex = sent32(u++);
    isl[0].mBodyCount = sent32(u++);
    isl[0].mArticulationCount = sent32(u++);
    isl[0].mNumVelocityIterations = -424242;
    isl[0].mStaticRigidBatchCount = sent32(u++);
    isl[0].mBiasCoefficient = sentF(f++);
    isl[1].mBodyStartIndex = sent32(u++);
    isl[1].mBodyCount = sent32(u++);
    isl[1].mBiasCoefficient = sentF(f++);

    // Expected values in EXACTLY the probe kernel's write order.
    std::vector<uint64_t> expect;
    expect.push_back(uint64_t(d.outSolverVelocity));
    expect.push_back(uint64_t(d.outBody2World));
    expect.push_back(uint64_t(d.solverBodyDataPool));
    expect.push_back(uint64_t(d.solverBodyTxIDataPool));
    expect.push_back(uint64_t(d.solverBodySleepDataPool));
    expect.push_back(uint64_t(d.outArtiVelocity));
    expect.push_back(uint64_t(d.islandContextPool));
    expect.push_back(uint64_t(d.motionVelocityArray));
    expect.push_back(uint64_t(d.constraintsPerPartition));
    expect.push_back(uint64_t(d.artiConstraintsPerPartition));
    expect.push_back(uint64_t(d.thresholdStream));
    expect.push_back(uint64_t(d.tmpThresholdStream));
    expect.push_back(uint64_t(d.exceededForceElements));
    expect.push_back(uint64_t(d.prevExceededForceElements));
    expect.push_back(uint64_t(d.forceChangeThresholdElements));
    expect.push_back(uint64_t(d.thresholdStreamAccumulatedForce));
    expect.push_back(uint64_t(d.thresholdStreamAccumulatedForceBetweenBlocks));
    expect.push_back(uint64_t(d.thresholdStreamWriteIndex));
    expect.push_back(uint64_t(d.thresholdStreamWriteIndexBetweenBlocks));
    expect.push_back(uint64_t(d.thresholdStreamWriteable));
    expect.push_back(uint64_t(d.accumulatedForceObjectPairs));
    expect.push_back(uint64_t(d.constraintWriteBack));
    expect.push_back(uint64_t(d.forceBuffer));
    expect.push_back(uint64_t(d.frictionPatches));
    expect.push_back(uint64_t(d.mRigidStaticContactCounts));
    expect.push_back(uint64_t(d.mRigidStaticContactStartIndices));
    expect.push_back(uint64_t(d.mRigidStaticJointCounts));
    expect.push_back(uint64_t(d.mRigidStaticJointStartIndices));
    expect.push_back(uint64_t(d.solverBodyReferences));
    expect.push_back(uint64_t(d.contactManagerOutputBase));
    expect.push_back(uint64_t(d.mBodySimBufferDeviceData));
    expect.push_back(uint64_t(d.mBodySimPrevVelocitiesBufferDeviceData));
    expect.push_back(d.numIslands);
    expect.push_back(d.numBatches);
    expect.push_back(d.numArticBatches);
    expect.push_back(d.numSolverBodies);
    expect.push_back(d.numSlabs);
    expect.push_back(d.accumulatedBodyDeltaVOffset);
    expect.push_back(uint64_t(uint32_t(d.sharedThresholdStreamIndex)));
    expect.push_back(1u); // enableStabilization byte
    expect.push_back(d.nbExceededThresholdElements);
    expect.push_back(d.nbPrevExceededThresholdElements);
    expect.push_back(d.nbForceChangeElements);
    expect.push_back(d.maxLinksPerArticulation);
    expect.push_back(bitsOf(d.contactErrorAccumulator.mErrorSumOfSquares));
    expect.push_back(uint64_t(uint32_t(d.contactErrorAccumulator.mCounter)));
    expect.push_back(bitsOf(d.contactErrorAccumulator.mMaxError));

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
    expect.push_back(uint64_t(s.iterativeData.blockConstraintBatch));
    expect.push_back(uint64_t(s.iterativeData.blockJointConstraintHeaders));
    expect.push_back(uint64_t(s.iterativeData.blockJointConstraintRowsCon));
    expect.push_back(uint64_t(s.iterativeData.blockJointConstraintRowsMod));
    expect.push_back(uint64_t(s.iterativeData.blockContactHeaders));
    expect.push_back(uint64_t(s.iterativeData.blockFrictionHeaders));
    expect.push_back(uint64_t(s.iterativeData.blockContactPoints));
    expect.push_back(uint64_t(s.iterativeData.blockFrictions));
    expect.push_back(uint64_t(s.iterativeData.solverBodyVelPool));
    expect.push_back(uint64_t(s.iterativeData.tempStaticBodyOutputs));
    expect.push_back(uint64_t(s.iterativeData.solverEncodedReferenceCount));
    expect.push_back(uint64_t(s.iterativeData.contactHeaders));
    expect.push_back(uint64_t(s.iterativeData.frictionHeaders));
    expect.push_back(uint64_t(s.iterativeData.contactPoints));
    expect.push_back(uint64_t(s.iterativeData.frictions));
    expect.push_back(uint64_t(s.iterativeData.artiResponse));

    expect.push_back(isl[0].mBodyStartIndex);
    expect.push_back(isl[0].mBodyCount);
    expect.push_back(isl[0].mArticulationCount);
    expect.push_back(uint64_t(uint32_t(isl[0].mNumVelocityIterations)));
    expect.push_back(isl[0].mStaticRigidBatchCount);
    expect.push_back(bitsOf(isl[0].mBiasCoefficient));
    expect.push_back(isl[1].mBodyStartIndex);
    expect.push_back(isl[1].mBodyCount);
    expect.push_back(bitsOf(isl[1].mBiasCoefficient));

    PxgDevicePtr dCore = 0, dShared = 0, dIsl = 0, dOut = 0;
    PXG_CHECK_MSG(shim.memAlloc(&dCore, sizeof(d)) == PxgResult::Success, "memAlloc core");
    PXG_CHECK_MSG(shim.memAlloc(&dShared, sizeof(s)) == PxgResult::Success, "memAlloc shared");
    PXG_CHECK_MSG(shim.memAlloc(&dIsl, sizeof(isl)) == PxgResult::Success, "memAlloc islands");
    PXG_CHECK_MSG(shim.memAlloc(&dOut, expect.size() * 8) == PxgResult::Success, "memAlloc out");
    if (!dCore || !dShared || !dIsl || !dOut)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    shim.memcpyHtoD(dCore, &d, sizeof(d));
    shim.memcpyHtoD(dShared, &s, sizeof(s));
    shim.memcpyHtoD(dIsl, isl, sizeof(isl));
    shim.memsetD32(dOut, 0, expect.size() * 2);

    struct ProbeParams { uint64_t core, shared, islands, out; } prm = { dCore, dShared, dIsl, dOut };
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
    PXG_CHECK_MSG(bad == 0, "solver desc mirrors are byte-exact (%zu bad of %zu)", bad, expect.size());

    shim.memFree(dCore);
    shim.memFree(dShared);
    shim.memFree(dIsl);
    shim.memFree(dOut);
}

} // namespace
