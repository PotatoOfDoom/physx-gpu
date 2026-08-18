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

// pxgSlangSolverDesc — RUNTIME_PLAN.md P5 Step 4: BDA mirrors of the engine's
// solver descriptor aggregates (gpusolver/include/PxgSolverCoreDesc.h +
// PxgIslandContext.h).  The per-frame TGS solver kernels receive TWO device
// pointers — PxgSolverCoreDesc* and PxgSolverSharedDesc<IterativeSolveDataTGS>*
// — and walk their nested device pointers (including the PxgIslandContext
// array reached through islandContextPool), so the Slang side needs byte-exact
// mirrors.  Pointers -> uint64_t, PxU32/PxI32/float -> 4-byte scalars, the
// trailing `bool enableStabilization` -> masked word read (its 3 padding bytes
// are uninitialized host memory).
//
// PxgSolverCoreDescGpu now mirrors the FULL engine struct: the scalar tail
// past enableStabilization (nbExceededThresholdElements ..
// maxLinksPerArticulation, then the Dy::ErrorAccumulator
// contactErrorAccumulator at 304) is required by solveWholeIslandTGS, which
// tests contactErrorAccumulator.mCounter >= 0 and atomically accumulates the
// per-launch contact residual statistics into the accumulator.  The two float
// members of the accumulator are declared as raw uint BITS so the kernel can
// CAS-emulate CUDA's float atomicAdd / AtomicMax on them
// (InterlockedCompareExchange has no float overload); mCounter keeps its int
// type for the plain InterlockedAdd and the signed >= 0 enable test.  The
// engine struct is 316 bytes rounded up to 320 by its pointer alignment; the
// desc is only ever dereferenced as desc[0] (never indexed), so the mirror's
// own 8-byte size rounding is inert.
//
// The layout is verified EMPIRICALLY, not by inspection:
// tests/testSolverDescLayout.cpp fills the REAL engine structs with sentinels
// and the solverDescProbe kernel reads fields across all three mirrors
// (bitwise compare).  Any drift between compiler and mirror fails that test.
module pxgSlangSolverDesc;

public struct PxgSolverCoreDescGpu
{
    public uint64_t outSolverVelocity;                          // @0   float4*
    public uint64_t outBody2World;                              // @8   PxAlignedTransform*
    public uint64_t solverBodyDataPool;                         // @16  PxgSolverBodyData*
    public uint64_t solverBodyTxIDataPool;                      // @24  PxgSolverTxIData*
    public uint64_t solverBodySleepDataPool;                    // @32  PxgSolverBodySleepData*
    public uint64_t outArtiVelocity;                            // @40  float4*
    public uint64_t islandContextPool;                          // @48  PxgIslandContext*
    public uint64_t motionVelocityArray;                        // @56  float4*
    public uint64_t constraintsPerPartition;                    // @64  PxU32*
    public uint64_t artiConstraintsPerPartition;                // @72  PxU32*
    public uint64_t thresholdStream;                            // @80  Dy::ThresholdStreamElement*
    public uint64_t tmpThresholdStream;                         // @88
    public uint64_t exceededForceElements;                      // @96
    public uint64_t prevExceededForceElements;                  // @104
    public uint64_t forceChangeThresholdElements;               // @112
    public uint64_t thresholdStreamAccumulatedForce;            // @120 PxReal*
    public uint64_t thresholdStreamAccumulatedForceBetweenBlocks; // @128
    public uint64_t thresholdStreamWriteIndex;                  // @136 PxU32*
    public uint64_t thresholdStreamWriteIndexBetweenBlocks;     // @144
    public uint64_t thresholdStreamWriteable;                   // @152 bool*
    public uint64_t accumulatedForceObjectPairs;                // @160 PxReal*
    public uint64_t constraintWriteBack;                        // @168 PxgConstraintWriteback*
    public uint64_t forceBuffer;                                // @176 PxF32*
    public uint64_t frictionPatches;                            // @184 PxFrictionPatch*
    public uint64_t mRigidStaticContactCounts;                  // @192 PxU32*
    public uint64_t mRigidStaticContactStartIndices;            // @200
    public uint64_t mRigidStaticJointCounts;                    // @208
    public uint64_t mRigidStaticJointStartIndices;              // @216
    public uint64_t solverBodyReferences;                       // @224 PxgSolverReferences*
    public uint64_t contactManagerOutputBase;                   // @232 PxsContactManagerOutput*
    public uint64_t mBodySimBufferDeviceData;                   // @240 PxgBodySim*
    public uint64_t mBodySimPrevVelocitiesBufferDeviceData;     // @248 PxgBodySimVelocities*
    public uint numIslands;                                     // @256
    public uint numBatches;                                     // @260
    public uint numArticBatches;                                // @264
    public uint numSolverBodies;                                // @268
    public uint numSlabs;                                       // @272
    public uint accumulatedBodyDeltaVOffset;                    // @276
    public int  sharedThresholdStreamIndex;                     // @280 PxI32
    public uint enableStabilizationWord;                        // @284 bool in byte 0 (mask & 0xff)
    public uint nbExceededThresholdElements;                    // @288
    public uint nbPrevExceededThresholdElements;                // @292
    public uint nbForceChangeElements;                          // @296
    public uint maxLinksPerArticulation;                        // @300
    // ---- Dy::ErrorAccumulator contactErrorAccumulator @304 ----
    // float members held as raw bits (uint) for the CAS-loop float atomics;
    // see the file header.
    public uint contactErrorAccumulator_mErrorSumOfSquares;     // @304 float bits
    public int  contactErrorAccumulator_mCounter;               // @308 PxI32
    public uint contactErrorAccumulator_mMaxError;              // @312 float bits
}

// PxgSolverSharedDesc<IterativeSolveDataTGS> = PxgSolverSharedDescBase (96
// bytes) + IterativeSolveDataTGS (16 pointers, 128 bytes) = 224 bytes total.
// The iterativeData member is flattened with the iter_ prefix.
public struct PxgSolverSharedDescTGSGpu
{
    public uint64_t blockCurrentFrictionPatches;    // @0   PxgBlockFrictionPatch*
    public uint64_t blockPreviousFrictionPatches;   // @8
    public uint64_t currentFrictionPatches;         // @16  PxgFrictionPatch*
    public uint64_t previousFrictionPatches;        // @24
    public uint64_t mBodySimBufferDeviceData;       // @32  PxgBodySim*
    public uint64_t articulations;                  // @40  PxgArticulation*
    public uint64_t articulationDeferredZ;          // @48  Cm::UnAlignedSpatialVector*
    public uint64_t articulationDirty;              // @56  PxU32*
    public uint64_t articulationSlabMask;           // @64  uint4*
    public uint  deltaOutOffset;                    // @72
    public float dt;                                // @76
    public float stepDt;                            // @80
    public float invDtF32;                          // @84
    public float stepInvDtF32;                      // @88
    public float lengthScale;                       // @92
    // ---- iterativeData (IterativeSolveDataTGS) @96 ----
    public uint64_t iter_blockConstraintBatch;         // @96  PxgBlockConstraintBatch*
    public uint64_t iter_blockJointConstraintHeaders;  // @104 PxgTGSBlockSolverConstraint1DHeader*
    public uint64_t iter_blockJointConstraintRowsCon;  // @112 PxgTGSBlockSolverConstraint1DCon*
    public uint64_t iter_blockJointConstraintRowsMod;  // @120 PxgBlockSolverConstraint1DMod*
    public uint64_t iter_blockContactHeaders;          // @128 PxgTGSBlockSolverContactHeader*
    public uint64_t iter_blockFrictionHeaders;         // @136 PxgTGSBlockSolverFrictionHeader*
    public uint64_t iter_blockContactPoints;           // @144 PxgTGSBlockSolverContactPoint*
    public uint64_t iter_blockFrictions;               // @152 PxgTGSBlockSolverContactFriction*
    public uint64_t iter_solverBodyVelPool;            // @160 float4*
    public uint64_t iter_tempStaticBodyOutputs;        // @168 float4*
    public uint64_t iter_solverEncodedReferenceCount;  // @176 PxU32*
    public uint64_t iter_contactHeaders;               // @184 PxgTGSSolverContactHeader*
    public uint64_t iter_frictionHeaders;              // @192 PxgSolverFrictionHeader*
    public uint64_t iter_contactPoints;                // @200 PxgTGSSolverContactPointExt*
    public uint64_t iter_frictions;                    // @208 PxgTGSSolverFrictionExt*
    public uint64_t iter_artiResponse;                 // @216 PxgArticulationBlockResponse*
}                                                      // = 224

// PxgSolverSharedDesc<IterativeSolveData> — the PGS variant.  IterativeSolveData
// (PxgSolverCoreDesc.h:141) has EXACTLY the same shape as IterativeSolveDataTGS:
// 16 device pointers in the same order (only the pointee block types differ —
// PxgBlockSolverContactHeader/Point/... instead of the PxgTGSBlock* types — and
// slot 9 is named tempStaticBodyOutputPool instead of tempStaticBodyOutputs), so
// the byte layout is identical to PxgSolverSharedDescTGSGpu: base (96) +
// iterativeData (128) = 224 bytes.  Kept as a separate struct so the PGS kernels
// document which engine aggregate they walk; testSolver.cpp static_asserts the
// host-side sizeof/offsetof equality of the two instantiations, and the
// TGS-variant probe (testSolverDescLayout) pins the shared byte layout.
public struct PxgSolverSharedDescPGSGpu
{
    public uint64_t blockCurrentFrictionPatches;    // @0   PxgBlockFrictionPatch*
    public uint64_t blockPreviousFrictionPatches;   // @8
    public uint64_t currentFrictionPatches;         // @16  PxgFrictionPatch*
    public uint64_t previousFrictionPatches;        // @24
    public uint64_t mBodySimBufferDeviceData;       // @32  PxgBodySim*
    public uint64_t articulations;                  // @40  PxgArticulation*
    public uint64_t articulationDeferredZ;          // @48  Cm::UnAlignedSpatialVector*
    public uint64_t articulationDirty;              // @56  PxU32*
    public uint64_t articulationSlabMask;           // @64  uint4*
    public uint  deltaOutOffset;                    // @72
    public float dt;                                // @76
    public float stepDt;                            // @80
    public float invDtF32;                          // @84
    public float stepInvDtF32;                      // @88
    public float lengthScale;                       // @92
    // ---- iterativeData (IterativeSolveData) @96 ----
    public uint64_t iter_blockConstraintBatch;         // @96  PxgBlockConstraintBatch*
    public uint64_t iter_blockJointConstraintHeaders;  // @104 PxgBlockSolverConstraint1DHeader*
    public uint64_t iter_blockJointConstraintRowsCon;  // @112 PxgBlockSolverConstraint1DCon*
    public uint64_t iter_blockJointConstraintRowsMod;  // @120 PxgBlockSolverConstraint1DMod*
    public uint64_t iter_blockContactHeaders;          // @128 PxgBlockSolverContactHeader*
    public uint64_t iter_blockFrictionHeaders;         // @136 PxgBlockSolverFrictionHeader*
    public uint64_t iter_blockContactPoints;           // @144 PxgBlockSolverContactPoint*
    public uint64_t iter_blockFrictions;               // @152 PxgBlockSolverContactFriction*
    public uint64_t iter_solverBodyVelPool;            // @160 float4*
    public uint64_t iter_tempStaticBodyOutputPool;     // @168 float4*
    public uint64_t iter_solverEncodedReferenceCount;  // @176 PxU32*
    public uint64_t iter_contactHeaders;               // @184 PxgSolverContactHeader*
    public uint64_t iter_frictionHeaders;              // @192 PxgSolverFrictionHeader*
    public uint64_t iter_contactPoints;                // @200 PxgSolverContactPointExt*
    public uint64_t iter_frictions;                    // @208 PxgSolverContactFrictionExt*
    public uint64_t iter_artiResponse;                 // @216 PxgArticulationBlockResponse*
}                                                      // = 224

// PxgIslandContext (gpusolver/include/PxgIslandContext.h), 68 bytes.  The
// kernels index islandContextPool[islandIndex], so this mirror's SIZE (array
// stride) matters, not only the field offsets — the layout test probes
// element [1] to pin the stride.
public struct PxgIslandContextGpu
{
    public uint mBodyStartIndex;        // @0
    public uint mBodyCount;             // @4
    public uint mArticulationCount;     // @8
    public uint mDescStartIndex;        // @12
    public uint mDescCount;             // @16
    public int  mNumPositionIterations; // @20
    public int  mNumVelocityIterations; // @24
    public uint mStartPartitionIndex;   // @28
    public uint mNumPartitions;         // @32
    public uint mBatchStartIndex;       // @36
    public uint mBatchCount;            // @40
    public uint mArtiBatchStartIndex;   // @44
    public uint mArtiBatchCount;        // @48
    public uint mStaticArtiBatchCount;  // @52
    public uint mSelfArtiBatchCount;    // @56
    public uint mStaticRigidBatchCount; // @60
    public float mBiasCoefficient;      // @64
}                                       // = 68 (4-aligned; the array stride)

// ---------------------------------------------------------------------------
// P8.99c: countActiveSlabs / resetSlabCount (solverBlock.cuh:84/:100) as
// PUBLIC helpers, hoisted VERBATIM from the file-local copies the two solver
// entry modules carry, so the articulation self-solve wrapper composes them
// rather than adding a fourth transcription.  solverPGS/solverTGS still hold
// their own file-local twins (unifying them means recompiling and re-verifying
// two launched solver modules -- a follow-up cleanup, tracked in
// RUNTIME_STATUS, not a P8.99c side effect).
// ---------------------------------------------------------------------------

public uint pxgCountActiveSlabsBda(uint index, uint numSlabs, uint numSolverBodies,
                                   Ptr<uint> encodedReferenceCount)
{
    uint referenceCount = 0;
    const uint num32Slabs = (numSlabs + 31) / 32;
    for (uint i = 0; i < num32Slabs; ++i)
        referenceCount += countbits(encodedReferenceCount[index + i * numSolverBodies]);
    return max(1u, referenceCount);
}

public void pxgResetSlabCountBda(uint index, uint numSlabs, uint numSolverBodies,
                                 Ptr<uint> encodedReferenceCount)
{
    const uint num32Slabs = (numSlabs + 31) / 32;
    for (uint i = 0; i < num32Slabs; ++i)
        encodedReferenceCount[index + i * numSolverBodies] = 0u;
}
