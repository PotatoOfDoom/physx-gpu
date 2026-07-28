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

// See PxgVulkanKernelAbi.h.  One row per kernel brought up on the engine
// launch path (P5 Step 4); argument sizes are transcribed from the CUDA
// __global__ signature verbatim.  Pointer = 8, PxU32/float/enum = 4,
// PxU16 = 2, PxU8/bool = 1; aggregates passed BY VALUE get their sizeof
// (PhysX passes large aggregates by device pointer, so by-value structs
// stay small).

#include "PxgVulkanKernelAbi.h"

#include <cstring>

namespace physx
{
namespace gpuslang
{
namespace
{

// gpucommon/src/CUDA/MemCopyBalanced.cu:
//   MemCopyBalanced(CopyDesc* descs, PxU32 count)
static const uint8_t kMemCopyBalanced[] = { 8, 4 };

// gpubroadphase/src/CUDA/broadphase.cu:
//   translateAABBsLaunch(const PxBounds3* in, PxgIntegerAABB* out,
//                        const PxReal* contactDistances, const PxU32* envIds,
//                        PxU32 boxesCapacity, PxU8 x6 shift/envID bits)
static const uint8_t kTranslateAABBs[] = { 8, 8, 8, 8, 4, 1, 1, 1, 1, 1, 1 };

// broadphase.cu: clearNewFlagLaunch(const PxgBroadPhaseDesc*) / bpSignalComplete(PxU32*)
static const uint8_t kOnePtr[] = { 8 };

// broadphase.cu: initializeSapBox1DLaunch(const PxgBroadPhaseDesc*, PxU32 numHandles, bool isNew)
static const uint8_t kSapBoxInit[] = { 8, 4, 1 };

// radixSortImpl.cu: (PxgRadixSortDesc* desc, PxU32 numKeys, PxU32 startBit)
static const uint8_t kRadixDesc[] = { 8, 4, 4 };

// broadphase.cu: (const PxgBroadPhaseDesc* bpDesc, bool isIncremental)
static const uint8_t kBpDescBool[] = { 8, 1 };

// gpusolver/src/CUDA/preIntegration.cu:
//   initStaticKinematics(PxU32 nbStaticKinematics, PxU32 nbSolverBodies,
//                        PxgSolverBodyData*, PxgSolverTxIData*, PxAlignedTransform*,
//                        float4* outVelocityPool, PxU32* activeNodeIndices, PxU32* solverBodyIndices)
static const uint8_t kInitStaticKin[] = { 4, 4, 8, 8, 8, 8, 8, 8 };

// gpusolver TGS kernels taking (PxgSolverCoreDesc*, PxgSolverSharedDesc<...>*)
static const uint8_t kTwoPtr[] = { 8, 8 };

// solverTGS: propagateAverageSolverBodyVelocityTGS(desc, sharedDesc,
//   bool isVelocityIteration, bool isLastPosIteration, PxReal biasCoefficient)
static const uint8_t kPropagateAvg[] = { 8, 8, 1, 1, 4 };

// solverTGS: writeBackBodiesTGS(desc, sharedDesc, PxU32 islandIndex)
static const uint8_t kWriteBackTGS[] = { 8, 8, 4 };

// solverTGS: solveWholeIslandTGS(desc, sharedDesc, PxU32 islandIndex,
//   bool doFriction, PxReal elapsedTime, PxReal minPen)
static const uint8_t kSolveWholeIslandTGS[] = { 8, 8, 4, 1, 4, 4 };

// solverTGS: concludeBlocksTGS(desc, sharedDesc, PxU32 islandIndex) -> 24-byte blob
static const uint8_t kConcludeBlocksTGS[] = { 8, 8, 4 };

// solverTGS: solveStaticBlockTGS / propagateStaticSolverBodyVelocitiesTGS
//   (desc, sharedDesc, PxU32 islandIndex, PxU32 nbStaticSlabs,
//    PxU32 maxStaticPartitions, PxReal elapsedTime, PxReal minPen) -> 40-byte blob
static const uint8_t kSolveStaticBlockTGS[] = { 8, 8, 4, 4, 4, 4, 4 };

// solverTGS: solveBlockUnified(desc, sharedDesc, PxU32 islandIndex,
//   PxU32 partitionIndex, bool doFriction, PxReal elapsedTime, PxReal minPen,
//   PxgArticulationCoreDesc*) -> 48-byte blob (bool byte @24, arti ptr @40)
static const uint8_t kSolveBlockUnified[] = { 8, 8, 4, 4, 1, 4, 4, 8 };

// solverTGS: markActiveSlabTGS(desc, sharedDesc, PxU32 islandIndex,
//   PxU32 lastPartition, PxReal minPen, PxReal elapsedTime,
//   PxgArticulationCoreDesc*) -> 40-byte blob (arti ptr @32; note minPen
//   precedes elapsedTime, unlike solveBlockUnified)
static const uint8_t kMarkActiveSlabTGS[] = { 8, 8, 4, 4, 4, 4, 8 };

// gpusolver PGS kernels (solverMultiBlock.cu, P5 Step 5 BDA):
// solveBlockPartition(PxgSolverCoreDesc*, const PxgSolverSharedDesc<IterativeSolveData>*,
//   PxU32 islandIndex, PxU32 partitionIndex, bool doFriction) -> 32-byte blob
//   (bool byte @24, bytes 25..31 padding)
static const uint8_t kSolveBlockPartitionPGS[] = { 8, 8, 4, 4, 1 };

// concludeBlocks / writebackBlocks / writeBackBodies
//   (desc, sharedDesc, PxU32 islandIndex) -> 24-byte blob
static const uint8_t kDescsIslandPGS[] = { 8, 8, 4 };

// computeAverageSolverBodyVelocity / propagateSolverBodyVelocity
//   (const PxgSolverCoreDesc*, const PxgSolverSharedDesc<IterativeSolveData>*)
//   -> kTwoPtr (16 bytes)

// markActiveSlabPGS(const PxgSolverCoreDesc*, const
//   PxgSolverSharedDesc<IterativeSolveData>*, PxU32 islandIndex,
//   PxU32 lastPartition, PxgArticulationCoreDesc*) -> 32-byte blob (arti ptr
//   @24; NOTE: unlike markActiveSlabTGS there is no minPen/elapsedTime)
static const uint8_t kMarkActiveSlabPGS[] = { 8, 8, 4, 4, 8 };

// integration.cu: integrateCoreParallelLaunchTGS(PxU32 offset, desc, sharedDesc,
//   PxU32* islandIds, PxU32* islandStaticTouchCounts, PxU32* numCountedInteractions).
//   The PGS twin integrateCoreParallelLaunch (integration.cu) has the IDENTICAL
//   6-arg pack (only the shared-desc instantiation differs), so it shares this
//   row's argSizes.
static const uint8_t kIntegrateTGS[] = { 4, 8, 8, 8, 8, 8 };

// preIntegration.cu: preIntegrationLaunch(PxU32 offset, PxU32 nbSolverBodies,
//   PxReal dt, PxVec3 gravity, PxgSolverBodyData*, PxgSolverBodySleepData*,
//   PxgSolverTxIData*, const PxgBodySim*, const PxNodeIndex*,
//   PxAlignedTransform*, float4* outVel, PxU32* solverBodyIndices).  Same as
//   the TGS twin (kPreIntegrateTGS below) WITHOUT the trailing bool.
//   PxVec3 by value = 12 bytes -> 88-byte blob.
static const uint8_t kPreIntegratePGS[] = { 4, 4, 4, 12, 8, 8, 8, 8, 8, 8, 8, 8 };

// solverMultiBlock.cu: solveStaticBlock(PxgSolverCoreDesc*,
//   const PxgSolverSharedDesc<IterativeSolveData>*, PxU32 islandIndex,
//   PxU32 nbStaticSlabs, PxU32 maxStaticPartitions, bool doFriction)
//   -> 32-byte blob (bool byte @28, bytes 29..31 padding).  NOTE: unlike the
//   TGS twin there is no elapsedTime/minPen.
static const uint8_t kSolveStaticBlockPGS[] = { 8, 8, 4, 4, 4, 1 };

// solverMultiBlock.cu: propagateStaticSolverBodyVelocities(const
//   PxgSolverCoreDesc*, const PxgSolverSharedDesc<IterativeSolveData>*,
//   PxU32 islandIndex, PxU32 nbStaticSlabs, PxU32 maxStaticPartitions)
//   -> 32-byte blob (28 rounded to the 8-alignment).  The engine's launch
//   site reuses solveStaticBlock's 6-entry param array; the stray doFriction
//   byte lands in this blob's tail padding, so both packs produce the same
//   32 bytes (see the .slang push struct).
static const uint8_t kPropagateStaticPGS[] = { 8, 8, 4, 4, 4 };

// solverMultiBlock.cu: dmaConstraintResidual(const PxgConstraintWriteback*,
//   PxReal* residuals, PxU32 count) — also the shape of
//   clearFrictionPatchCounts(PxU32* frictionPatchCount,
//   PxU32* edgeIndicesToClear, PxU32 nbToClear) in constraintBlockPrePrep.cu.
static const uint8_t kTwoPtrU32[] = { 8, 8, 4 };

// gpusimulationcontroller updateBodiesAndShapes.cu:
//   updateShapesLaunch(const PxgNewShapeSim*, PxgShapeSim*, PxU32 nbNewShapes)
static const uint8_t kUpdateShapes[] = { 8, 8, 4 };

// gpunarrowphase cudaBox.cu: boxBoxNphase_Kernel(PxU32 numTests, 14 device
//   pointers, PxU32 patchBytesLimit, contactBytesLimit, forceBytesLimit,
//   PxReal toleranceLength) — the box-vs-box contact generator.
static const uint8_t kBoxBoxNphase[] = { 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4 };

// gpunarrowphase cudaSphere.cu: sphereNphase_Kernel(PxU32 numTests,
//   PxReal toleranceLength, 14 device pointers, PxU32 patchBytesLimit,
//   contactBytesLimit, forceBytesLimit) — the sphere/capsule/plane/box primitive
//   contact generator.  Note toleranceLength is the 2nd arg (unlike box-box's last).
static const uint8_t kSphereNphase[] = { 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 4, 4 };
// gpunarrowphase cudaGJKEPA.cu: convexPlaneNphase_Kernel(PxU32 numTests, 15
//   device pointers (cmInputs, cmOutputs, shapes, transformCache,
//   contactDistance, materials, contactStream, patchStream, cmManifolds,
//   patchAndContactCounters, touchChangeFlags, patchChangeFlags,
//   startContactPatches, startContactPoints, startContactForces), PxU32
//   patchBytesLimit, contactBytesLimit, forceBytesLimit, PxReal
//   toleranceLength) — the convex/box-vs-plane PCM contact generator (144-byte
//   push blob).
static const uint8_t kConvexPlaneNphase[] = { 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4 };

// gpunarrowphase convex-convex core (cudaGJKEPA.cu):
//   convexConvexNphase_stage1Kernel(PxU32 numTests, PxReal toleranceLength,
//     cmInputs*, transformCache*, shapes*, cmManifolds*, tempKeepPairArray*)
static const uint8_t kCvxStage1[] = { 4, 4, 8, 8, 8, 8, 8 };
//   convexConvexNphase_stage2Kernel(PxU32 numTests, PxReal toleranceLength,
//     tempKeepPairsArray*, cmInputs*, cmManifolds*, shapes*, transformCache*, contactDistance*)
static const uint8_t kCvxStage2[] = { 4, 4, 8, 8, 8, 8, 8, 8 };
//   finishContactsKernel(PxU32 numTests, 9 ptrs, bool insertAveragePoint, 6 ptrs, 3 PxU32)
static const uint8_t kFinishContacts[] = { 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1, 8, 8, 8, 8, 8, 8, 4, 4, 4 };
//   compressContactStage1(ptr, PxU32, ptr) / compressContactStage2(2 ptr, u32, 12 ptr, u32)
static const uint8_t kCompress1[] = { 8, 4, 8 };
static const uint8_t kCompress2[] = { 8, 8, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4 };
//   updateFrictionPatches(PxU32 pairCount, const PxU8* startContactPatches,
//     PxU8* startFrictionPatches, PxsContactManagerOutput* outputs) -> 32-byte blob
static const uint8_t kFrictionPatch[] = { 4, 8, 8, 8 };
//   prepareLostFoundPairs_Stage1(const PxU32* inputFlags, PxU32* tempRunsum,
//     PxU32* blockAccumulation, PxU32 numPairs) -> 32-byte blob
static const uint8_t kLostFound1[] = { 8, 8, 8, 4 };
//   prepareLostFoundPairs_Stage2(inputFlags*, tempRunsum*, allOutputManagers*,
//     compactedOutputManagers*, outputManagersPtrs*, lostAndTotalReportedPairsCount*,
//     blockAccumulation*, PxU32 numPairs, contactManagerArray*) -> 72-byte blob
static const uint8_t kLostFound2[] = { 8, 8, 8, 8, 8, 8, 8, 4, 8 };

// gpusolver: rigidSumInternalContactAndJointBatches1(PxgPrePrepDesc*, PxU32 nbBodies)
static const uint8_t kPtrU32[] = { 8, 4 };
// rigidSumInternalContactAndJointBatches2(PxgPrePrepDesc*, PxgSolverCoreDesc*,
//   PxgConstraintPrepareDesc*, PxU32 nbBodies)
static const uint8_t kThreePtrU32[] = { 8, 8, 8, 4 };

// gpusolver preIntegrationTGS.cu: preIntegrationLaunchTGS(PxU32 offset,
//   PxU32 nbSolverBodies, PxReal dt, PxVec3 gravity, PxgSolverBodyData*,
//   PxgSolverBodySleepData*, PxgSolverTxIData*, const PxgBodySim*,
//   const PxNodeIndex*, PxAlignedTransform*, float4* outVel, PxU32* solverBodyIndices,
//   bool skipGravityApplication).  PxVec3 by value = 12 bytes.
static const uint8_t kPreIntegrateTGS[] = { 4, 4, 4, 12, 8, 8, 8, 8, 8, 8, 8, 8, 1 };

// gpusimulationcontroller particlesystem.cu (P6.0 PBD particle BDA):
//   ps_preIntegrateLaunch(const PxgParticleSystem* particleSystems,
//     const PxU32* activeParticleSystems, PxVec3 gravity, PxReal dt,
//     bool isTGS, bool externalForcesEveryTgsIterationEnabled)
//   PxVec3 by value = 12 bytes -> 40-byte blob (two 1-byte bools at 32/33).
static const uint8_t kPsPreIntegrate[] = { 8, 8, 12, 4, 1, 1 };
//   ps_integrateLaunch(PxgParticleSystem* particleSystems,
//     const PxU32* activeParticleSystems, PxReal dt, PxReal epsilonSq)
//     -> 24-byte blob.
static const uint8_t kPsIntegrate[] = { 8, 8, 4, 4 };
//   ps_updateUserBufferLaunch(const PxgParticleSystem* particleSystems,
//     const PxU32* activeParticleSystems) — the __global__ takes 2 pointers, but
//     the launch site (copyUnsortedArrayToUserBuffer) passes a 3rd PxU32 `count`
//     (unused by the kernel), so the marshalled pack is {8,8,4} -> 24-byte blob.
static const uint8_t kPsUpdateUserBuffer[] = { 8, 8, 4 };
//   ps_updateUnsortedArrayLaunch(const PxgParticleSystem* particleSystems,
//     const PxU32* activeParticleSystems) — user-buffer -> unsorted ingest.
//     Two device pointers -> 16-byte blob (same shape as kTwoPtr).
static const uint8_t kPsUpdateUnsorted[] = { 8, 8 };

// gpusimulationcontroller FEMCloth.cu (P7.0 FEMCloth free-fall BDA):
//   cloth_preIntegrateLaunch(PxgFEMCloth* femCloths, const PxU32* activeIds,
//     PxVec3 gravity, PxReal dt, bool isTGS, PxReal* speculativeCCDContactOffset,
//     bool externalForcesEveryTgsIterationEnabled, bool adaptiveCollisionPairUpdate,
//     PxU8* updateContactPairs, PxU32* VTContactCounts, PxU32* EEContactCounts)
//     PxVec3 by value = 12 -> natural-alignment blob = 80 bytes.
static const uint8_t kClothPreIntegrate[] = { 8, 8, 12, 4, 1, 8, 1, 1, 8, 8, 8 };
//   cloth_finalizeVelocitiesLaunch(PxgFEMCloth* femCloths, const PxU32* activeId,
//     PxReal invTotalDt, PxReal dt, bool alwaysRunVelocityAveraging) -> 32-byte blob.
static const uint8_t kClothFinalizeVel[] = { 8, 8, 4, 4, 1 };
//   cloth_sleeping(PxgFEMCloth* femCloths, PxU32 numActiveCloths, const PxU32*
//     activeId, PxReal dt, PxReal resetCounter, PxReal* wakeCounters,
//     PxU32* stateChangedMask) -> 48-byte blob (activeId 8-aligned after the PxU32).
static const uint8_t kClothSleeping[] = { 8, 4, 8, 4, 4, 8, 8 };

// gpusimulationcontroller FEMCloth.cu (P7.1 FEMCloth in-plane membrane energy solve):
//   cloth_solveNonSharedTriangleEnergyLaunch(PxgFEMCloth* femCloths, const PxU32*
//     activeFEMCloths, const PxsDeformableSurfaceMaterialData* clothMaterials,
//     PxReal dt, PxU32 partitionId, bool isTGS)  -- launch site
//     PxgFEMClothCore.cpp:1556; the cluster twin
//     cloth_solveNonSharedTriangleEnergyClusterLaunch (:1573) shares the pack
//     (clusterStartId in place of partitionId).  Three device pointers + PxReal +
//     PxU32 + bool -> natural-alignment blob = 40 bytes.
static const uint8_t kClothSolveTriEnergy[] = { 8, 8, 8, 4, 4, 1 };

// gpusimulationcontroller FEMCloth.cu (P7.1 tranche 2 FEMCloth triangle-pair
// shared-energy + bending solve):
//   cloth_solveTrianglePairEnergyLaunch(PxgFEMCloth* gFEMCloths, const PxU32*
//     activeFEMCloths, const PxsDeformableSurfaceMaterialData* clothMaterials,
//     PxReal dt, PxU32 partitionId, bool isSharedTrianglePair, bool isTGS)  --
//     launch site PxgFEMClothCore.cpp:1617; the cluster twin
//     cloth_solveTrianglePairEnergyClusterLaunch (:1637) shares the pack
//     (firstClusterId in place of partitionId).  Three device pointers + PxReal +
//     PxU32 + two bools (@32/@33) -> natural-alignment blob = 40 bytes.
static const uint8_t kClothSolveTriPairEnergy[] = { 8, 8, 8, 4, 4, 1, 1 };
//   cloth_averageTrianglePairVertsLaunch(PxgFEMCloth* gFEMCloths, const PxU32*
//     activeFEMCloths, PxReal invDt, bool isSharedPartition)  -- launch site
//     PxgFEMClothCore.cpp:1668.  Two device pointers + PxReal + bool -> 24-byte
//     blob (bool @20, rounded to the 8-alignment).
static const uint8_t kClothAvgPairVerts[] = { 8, 8, 4, 1 };

// gpusimulationcontroller FEMCloth rigid-attachment kernels (P7.1 tranche 3a BDA):
//   cloth_rigidAttachmentPrepareLaunch(PxgFEMCloth* clothes,
//     PxgFEMRigidAttachment* rigidAttachments, PxU32* activeRigidAttachments,
//     PxNodeIndex* rigidAttachmentIds, PxU32 numRigidAttachments,
//     PxgFEMRigidAttachmentConstraint* attachmentConstraints, PxgPrePrepDesc* preDesc,
//     PxgConstraintPrepareDesc* prepareDesc, PxgSolverSharedDescBase* sharedDesc,
//     float4* rigidDeltaVel)  -- launch site PxgFEMClothCore.cpp:226.  Four device
//   pointers + PxU32 (@32) + five device pointers -> natural-alignment blob = 80
//   bytes (numRigidAttachments's 4-byte tail-pad realigns the @40 pointer).
static const uint8_t kClothRigidAttachPrep[] = { 8, 8, 8, 8, 4, 8, 8, 8, 8, 8 };
//   cloth_solveRigidClothAttachmentLaunch / ...TGSLaunch(PxgFEMCloth* clothes,
//     PxgFEMRigidAttachmentConstraint* attachments, PxU32 numAttachments,
//     PxgPrePrepDesc* prePrepDesc, PxgSolverCoreDesc* solverCoreDesc,
//     PxgArticulationCoreDesc* artiCoreDesc, PxgSolverSharedDesc* sharedDesc,
//     PxReal dt, float4* rigidDeltaVel)  -- launch site PxgFEMClothCore.cpp:1858
//   (both PGS and TGS share the pack).  Two device pointers + PxU32 (@16) + four
//   device pointers + PxReal (@56) + device pointer -> natural-alignment blob = 72
//   bytes.
static const uint8_t kClothRigidAttachSolve[] = { 8, 8, 4, 8, 8, 8, 8, 4, 8 };

// gpusimulationcontroller FEMCloth.cu apply-external-deltas (P7.1 tranche 3c):
//   cloth_applyExternalDeltasLaunch(PxgFEMCloth* femClothes, const PxU32*
//     activeClothes, PxReal dt)  -- launch PxgFEMClothCore.cpp:2113.  Two device
//   pointers + PxReal -> natural-alignment blob = 24 bytes (20 rounded to 8).
static const uint8_t kClothApplyExtDeltas[] = { 8, 8, 4 };

// gpunarrowphase femClothMidPhase.cu (P7.1 tranche 4a cloth-vs-primitive contact
// GENERATION — the broadphase->midphase pass):
//   cloth_midphaseGenerateVertexPairsLaunch(PxU32 numWorkItems,
//     PxReal tolerenceLength, const PxgContactManagerInput* cmInputs,
//     const PxsCachedTransform* transformCache, const PxBounds3* bounds,
//     const PxReal* contactDistance, const PxgShape* gpuShapes,
//     const PxgFEMCloth* femClothes, uint4* stackPtr, PxU32* midphasePairsNum)
//   -- launch site PxgNarrowphaseCore.cpp:6317.  PxU32 + PxReal (@0/@4) + eight
//   device pointers -> natural-alignment blob = 72 bytes.
static const uint8_t kClothMidphaseVertPairs[] = { 4, 4, 8, 8, 8, 8, 8, 8, 8, 8 };
//   cloth_planeVertContactGenLaunch / cloth_SphereVertexContactGenLaunch share the
//   arg pack (femClothPrimitives.cu:853 / :700, launch PxgNarrowphaseCore.cpp:5853 /
//   :5776): (PxU32 numWorkItems, const PxgContactManagerInput* cmInputs,
//   const PxsCachedTransform* transformCache, const PxReal* contactDistance,
//   const PxReal* restDistance, const PxgShape* gpuShapes,
//   const PxgFEMCloth* femClothes, const PxNodeIndex* shapeToRigidRemapTable,
//   PxgRigidFilterPair* filterPairs, PxU32 nbFilterPairs, PxgFEMContactWriter writer).
//   The writer is passed BY VALUE (nine device pointers + PxU32 maxNumContacts +
//   4B tail pad = 80 bytes, sizeof(PxgFEMContactWriter)).  Eight device pointers +
//   PxU32 nbFilterPairs (@72) + the 80-byte writer (@80) -> 160-byte blob.
static const uint8_t kClothVertContactGen[] = { 4, 8, 8, 8, 8, 8, 8, 8, 8, 4, 80 };

// gpunarrowphase softbodyPrimitives.cu:1655 (P7.1 tranche 4b-1 cloth-vs-rigid
// contact SORT / reorder):
//   fem_reorderRigidContactsLaunch(const float4* contacts, const float4* normalpens,
//     const float4* barycentrics, const PxgFemOtherContactInfo* contactInfos,
//     const PxU32* numContacts, const PxU32* remapByRigid, float4* sortedContacts,
//     float4* sortedNormalPens, float4* sortedBarycentrics,
//     PxgFemOtherContactInfo* sortedContactInfos, PxU64* sortedRigidIds)
//   -> 11 device pointers = {8,8,8,8,8,8,8,8,8,8,8} = 88-byte blob.  Launch site
//   PxgFEMCore::reorderRigidContacts() PxgFEMCore.cpp:232.
static const uint8_t kFemReorderRSContacts[] = { 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8 };

// gpusimulationcontroller FEMCloth-vs-rigid contact PREPARE + SOLVE (P7.1 tranche 4b-2/3):
//   cloth_rigidContactPrepareLaunch(PxgFEMCloth* femClothes, float4* contacts_restW,
//     float4* normalPens, float4* barycentrics, const PxgFemOtherContactInfo* contactInfos,
//     PxU32* numContacts, PxgFemRigidConstraintBlock* primitiveConstraints,
//     PxgPrePrepDesc* preDesc, PxgConstraintPrepareDesc* prepareDesc, PxReal* rigidLambdaNs,
//     PxReal invDt, PxgSolverSharedDescBase* sharedDesc, bool isTGS)  -- launch site
//   PxgFEMClothCore.cpp:328.  Ten device pointers + PxReal (@80) + device pointer (@88) +
//   bool (@96) -> natural-alignment blob = 104 bytes.
static const uint8_t kClothRigidContactPrep[] = { 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 8, 1 };
//   cloth_queryRigidClothContactReferenceCount{,TGS}Launch(PxgFEMCloth* femClothes,
//     PxgFemOtherContactInfo* contactInfos, PxgFemRigidConstraintBlock* constraints,
//     PxU32* numContacts, PxgPrePrepDesc* prePrepDesc, PxgSolverCoreDesc* solverCoreDesc,
//     PxgArticulationCoreDesc* artiCoreDesc, PxgSolverSharedDesc* sharedDesc, PxReal dt,
//     PxReal* appliedForces, PxU32* rigidBodyReferenceCounts)  -- launch site
//   PxgFEMClothCore.cpp:1741 (both PGS and TGS share the pack).  Eight device pointers +
//   PxReal (@64) + two device pointers (@72/@80) -> natural-alignment blob = 88 bytes.
static const uint8_t kClothRigidContactQuery[] = { 8, 8, 8, 8, 8, 8, 8, 8, 4, 8, 8 };
//   cloth_solveRigidClothCollision{,TGS}Launch(PxgFEMCloth* femClothes,
//     PxgFemOtherContactInfo* contactInfos, PxgFemRigidConstraintBlock* constraints,
//     PxU32* numContacts, PxgPrePrepDesc* prePrepDesc, PxgSolverCoreDesc* solverCoreDesc,
//     PxgArticulationCoreDesc* artiCoreDesc, PxgSolverSharedDesc* sharedDesc,
//     float4* rigidDeltaVel, PxReal* appliedForces, PxU32* rigidBodyReferenceCounts,
//     PxReal dt, PxsDeformableSurfaceMaterialData* materials,
//     const PxsMaterialData* rigidBodyMaterials)  -- launch site PxgFEMClothCore.cpp:1797
//   (both PGS and TGS share the pack).  Eleven device pointers + PxReal (@88) + two device
//   pointers (@96/@104) -> natural-alignment blob = 112 bytes.
static const uint8_t kClothRigidContactSolve[] = { 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 8, 8 };

// gpunarrowphase femClothClothMidPhase.cu (P7.2 tranche 1 cloth SELF-collision midphase
// + contact GENERATION):
//   cloth_selfCollisionMidphase{VT,EE}Launch(PxU32 maxContacts, const PxU32*
//     activeClothes, const PxReal* contactDistance, const PxgFEMCloth* clothes,
//     PxgNonRigidFilterPair* clothClothPairs, PxU32 numClothClothPairs,
//     const PxU8* updateContactPairs, PxgFemFemContactInfo* outContactInfo,
//     PxU32* totalContactCount)  -- launch site PxgFEMClothCore.cpp:698 (VT and EE
//   share the pack).  PxU32 (@0) + four device pointers + PxU32 (@40) + three device
//   pointers -> natural-alignment blob = 72 bytes (the two PxU32 4-byte tail-pads
//   realign the following pointers).
static const uint8_t kClothSelfCollisionMidphase[] = { 4, 8, 8, 8, 8, 4, 8, 8, 8 };

// gpusimulationcontroller FEMCloth self-contact SOLVE (P7.2 tranche 2 cloth-vs-cloth
// self-contact prepare + count + solve + validity):
//   cloth_clothContactPrepareLaunch(PxgFEMCloth* clothes,
//     PxgFemFemContactInfo* contactInfos, PxU32* numContacts, PxU32 maxContacts,
//     PxsDeformableSurfaceMaterialData* clothMaterials, const PxU8* updateContactPairs)
//   -- launch site PxgFEMClothCore.cpp:505.  Three device pointers + PxU32 (@24) + two
//   device pointers -> natural-alignment blob = 48 bytes (the PxU32 4-byte tail-pad
//   realigns the following pointer).
static const uint8_t kClothClothContactPrep[] = { 8, 8, 8, 4, 8, 8 };
//   cloth_queryClothClothContact{VT,EE}CountLaunch(PxgFEMCloth* cloths,
//     PxgFemFemContactInfo* contactInfos, PxU32* numContacts)  -- launch site
//   PxgFEMClothCore.cpp:1983 (VT and EE share the pack).  Three device pointers = 24-byte blob.
static const uint8_t kClothClothContactQuery[] = { 8, 8, 8 };
//   cloth_solveClothClothDelta{VT,EE}Launch(PxgFEMCloth* cloths,
//     PxgFemFemContactInfo* contactInfos, PxU32* numContacts, PxReal dt)  -- launch site
//   PxgFEMClothCore.cpp:2005 (VT and EE share the pack).  Three device pointers + PxReal
//   (@24) -> 32-byte blob.
static const uint8_t kClothClothContactSolve[] = { 8, 8, 8, 4 };
//   cloth_updateClothContactValidityLaunch(PxgFEMCloth* femCloths, const PxU32* activeIds,
//     bool adaptiveCollisionPairUpdate, bool forceUpdateClothContactPairs,
//     const PxU8* updateContactPairs, PxU32* VTContactCount, PxU32* EEContactCount,
//     PxReal dt)  -- launch site PxgFEMClothCore.cpp:652.  Two device pointers + two bools
//   (@16/@17) + three device pointers + PxReal (@48) -> natural-alignment blob = 56 bytes.
static const uint8_t kClothUpdateContactValidity[] = { 8, 8, 1, 1, 8, 8, 8, 4 };

// gpusimulationcontroller FEMCloth.cu:338 (P7.2 tranche 3a cloth BVH bottom-up refit):
//   cloth_refitBoundLaunch(PxgFEMCloth* gFemClothes, const PxU32* activeFemClothes,
//     const PxU32 nbActiveFemClothes, PxReal* contactDists,
//     PxReal* speculativeCCDContactOffset, PxBounds3* boundArray, const PxU32* elemIndex,
//     const PxReal nbCollisionPairUpdatesPerTimestep, const PxU8* updateContactPairs)
//   -- launch site PxgFEMClothCore.cpp:563.  Two device pointers + PxU32 (@16, its
//   4-byte tail-pad realigning the next pointer to @24) + four device pointers +
//   PxReal (@56, likewise tail-padded) + device pointer (@64) -> natural-alignment
//   blob = 72 bytes.
static const uint8_t kClothRefitBound[] = { 8, 8, 4, 8, 8, 8, 8, 4, 8 };

// gpusimulationcontroller softBodyGM.cu:72 (P8.0 SOFTBODY / FEM-volume free-fall BDA):
//   sb_gm_preIntegrateLaunch(PxgSoftBody* softbodies, const PxU32* activeId,
//     PxVec3 gravity, PxReal dt, bool isTGS, PxReal* speculativeCCDContactOffset,
//     bool externalForcesEveryTgsIterationEnabled)  -- launch site
//   PxgSoftBodyCore.cpp:165.  Two device pointers + PxVec3 by value (12) + PxReal
//   (@28) + bool isTGS (@32) + device pointer (@40, after the isTGS 4-byte tail-pad
//   realigns to 8) + bool externalForces (@48) -> natural-alignment blob = 56 bytes.
static const uint8_t kSbPreIntegrate[] = { 8, 8, 12, 4, 1, 8, 1 };

// gpusimulationcontroller softBody.cu:99 (P8.0 tranche 2 softbody BVH bottom-up refit):
//   sb_refitBoundLaunch(PxgSoftBody* gSoftbodies, const PxU32* activeSoftbodies,
//     const PxU32 nbActiveSoftBodies, PxReal* bpContactDistance, PxReal* npContactDistance,
//     PxReal* speculativeCCDContactOffset, PxBounds3* boundArray, const PxU32* elemIndex)
//   -- launch site PxgSoftBodyCore.cpp:284.  Two device pointers + PxU32 (@16, its 4-byte
//   tail-pad realigning the next pointer to @24) + five device pointers -> natural-alignment
//   blob = 64 bytes.
static const uint8_t kSbRefitBound[] = { 8, 8, 4, 8, 8, 8, 8, 8 };

// gpusimulationcontroller softBodyGM.cu (P8.0 tranche 3 — the rest of the softbody
// free-fall chain):
//   sb_gm_stepSoftbodyLaunch(const PxgSoftBody* softbodies, const PxU32* activeId,
//     const PxReal dt, const PxVec3 gravity, const bool
//     externalForcesEveryTgsIterationEnabled)  -- launch site PxgSoftBodyCore.cpp:2637.
//   Two device pointers + PxReal (@16) + PxVec3 by value (12, 4-byte aligned -> @20..32)
//   + bool (@32) -> 33, rounded up to the pointer alignment -> 40-byte blob.
static const uint8_t kSbStepSoftbody[] = { 8, 8, 4, 12, 1 };
//   sb_gm_finalizeVelocitiesLaunch(PxgSoftBody* gSoftbodies, const PxU32* activeSoftbodies,
//     const PxReal invDt, const PxReal velocityScale, const PxReal dt,
//     const bool alwaysRunVelocityAveraging)  -- launch site PxgSoftBodyCore.cpp:922.
//   Two device pointers + three PxReal (@16/@20/@24) + bool (@28) -> 29, rounded up to
//   the pointer alignment -> 32-byte blob.
static const uint8_t kSbFinalizeVelocities[] = { 8, 8, 4, 4, 4, 1 };
//   sb_sleeping(PxgSoftBody* softBodies, const PxU32 numActiveSoftBodies,
//     const PxU32* activeId, const PxReal dt, const PxReal resetCounter,
//     PxReal* wakeCounters, PxU32* stateChangedMask)  -- launch site
//   PxgSoftBodyCore.cpp:955.  Device pointer + PxU32 (@8, its 4-byte tail-pad realigning
//   the next pointer to @16) + device pointer + two PxReal (@24/@28) + two device
//   pointers (@32/@40) -> 48-byte blob.  Same shape as the cloth kClothSleeping row.
static const uint8_t kSbSleeping[] = { 8, 4, 8, 4, 4, 8, 8 };

// gpusimulationcontroller softBodyGM.cu (P8.0 tranche 4 — the softbody FEM-volume solve
// bracket):
//   sb_gm_zeroTetMultipliers(PxgSoftBody* gSoftbodies, const PxU32* activeSoftbodies)
//   -- launch site PxgSoftBodyCore.cpp:198.  Two device pointers -> 16-byte blob.
static const uint8_t kSbZeroTetMultipliers[] = { 8, 8 };
//   sb_gm_applyExternalDeltasLaunch(PxgSoftBody* gSoftbodies,
//     const PxU32* activeSoftbodies, const PxReal invDt)  -- launch site
//   PxgSoftBodyCore.cpp:874.  Two device pointers + PxReal (@16) -> 20, rounded up to the
//   pointer alignment -> 24-byte blob.
static const uint8_t kSbApplyExternalDeltas[] = { 8, 8, 4 };
//   sb_gm_cp_solveTetrahedronsPartitionLaunch(PxgSoftBody* gSoftbodies,
//     const PxU32* activeSoftbodies, const PxU32 numActiveSoftbodies, const PxReal invDt,
//     const PxU32 partitionId, bool isTGS, bool isFirstIteration,
//     PxsDeformableVolumeMaterialData* materials)  -- launch site PxgSoftBodyCore.cpp:1202.
//   Two device pointers + three 4-byte scalars (@16/@20/@24) + two bools (@28/@29) + a
//   device pointer realigned to @32 -> 40-byte blob (P8.0 tranche 6).
static const uint8_t kSbSolveTetraPartition[] = { 8, 8, 4, 4, 4, 1, 1, 8 };
//   sb_gm_updateTetrahedraRotationsLaunch / sb_updateTetrahedraRotationsLaunch(
//     PxgSoftBody* gSoftbodies, const PxU32* activeSoftbodies)  -- launch sites
//   PxgSoftBodyCore.cpp:747 and :790.  Same two-device-pointer pack as
//   sb_gm_zeroTetMultipliers -> 16-byte blob (P8.0 tranche 5).

// gpusimulationcontroller particlesystem.cu (P6.2 PBD grid build + reorder):
//   ps_calculateHashLaunch(const PxgParticleSystem* particleSystems,
//     const PxU32* activeParticleSystems) — grid-cell hash + identity seed.
//   Launch site PxgParticleSystemCore.cpp:636 packs two device pointers
//   { particleSystemsd, activeParticleSystemsd } -> 16-byte blob.
static const uint8_t kPsCalcHash[] = { 8, 8 };
//   ps_reorderDataAndFindCellStartLaunch(const PxgParticleSystem* particleSystems,
//     const PxU32 id) — gather + cell start/end.  Launch site
//   PxgParticleSystemCore.cpp:681 packs { particleSystemsd, id } (pointer +
//   PxU32 by value) -> {8,4} = 12, rounded to the 8-alignment -> 16-byte blob.
static const uint8_t kPsReorder[] = { 8, 4 };

// gpusimulationcontroller particlesystem.cu (P6.3 PBD self-collision + solve):
//   ps_selfCollisionLaunch(const PxgParticleSystem* particleSystems,
//     const PxU32 id) — 3x3x3 grid neighbour search.  Launch site
//   PxgParticleSystemCore.cpp:1043 packs { particleSystemsd, id } (pointer +
//   PxU32 by value) -> {8,4}; reuses kPsReorder (identical shape).
//   ps_solveOneWayContactDeltaVLaunch(PxgParticleSystem* particleSystems,
//     const PxU32* activeParticleSystems, PxReal invDt, PxReal dt,
//     PxReal biasCoefficient, bool isVelocityIteration) — one-way contact solve.
//   Launch site PxgParticleSystemCore.cpp:1359 -> {8,8,4,4,4,1} = 32-byte blob
//   (bool at @28).
static const uint8_t kPsSolveOneWay[] = { 8, 8, 4, 4, 4, 1 };
//   ps_applyDeltaLaunch(PxgParticleSystem* particleSystems,
//     const PxU32* activeParticleSystems, PxReal dt) — SOR-average mDelta into
//   mAccumDeltaP.  Launch site PxgParticleSystemCore.cpp:1526 -> {8,8,4};
//   reuses kPsUpdateUserBuffer (identical shape).
//   ps_updateParticleLaunch(const PxgParticleSystem* particleSystems,
//     const PxU32* activeParticleSystems, PxReal invDt,
//     bool skipNewPositionAdjustment) — apply accumDeltaP + recompute velocity.
//   Launch site PxgParticleSystemCore.cpp:1568 -> {8,8,4,1} = 24-byte blob
//   (bool at @20).
static const uint8_t kPsUpdateParticle[] = { 8, 8, 4, 1 };

// gpusimulationcontroller particlesystem.cu (P6.4a-1 PBD bound bookkeeping):
//   ps_updateBoundFirstPassLaunch(const PxgParticleSystem* particleSystems,
//     const PxU32 id, const bool isTGS, PxBounds3* bounds) — per-block world AABB
//   reduce.  Launch site PxgParticleSystemCore.cpp:856 packs
//   { particleSystems, particleSystemId, isTGS, tempBuf } (ptr + PxU32 + bool +
//   ptr) -> {8,4,1,8} = 24-byte blob (bool @12, bounds @16).
static const uint8_t kPsUpdateBound1[] = { 8, 4, 1, 8 };
//   ps_updateBoundSecondPassLaunch(PxgParticleSystem* particleSystems,
//     const PxU32 id, const PxBounds3* bounds, const PxU32 numBounds,
//     PxBounds3* boundArray, const PxReal* contactDists, const PxU32 elemIndex)
//   — global AABB + fatten + centre.  Launch site
//   PxgParticleSystemCore.cpp:907 packs { particleSystems, particleSystemId,
//   tempBuf, numBlocks, boundArray, contactDists, elemId }
//   -> {8,4,8,4,8,8,4} = 56-byte blob.
static const uint8_t kPsUpdateBound2[] = { 8, 4, 8, 4, 8, 8, 4 };

// gpunarrowphase cudaParticleSystem.cu (P6.4a-1 PBD primitive cell-range):
//   ps_primitivesBoundFirstPassLaunch(const PxU32 numTests,
//     const PxgContactManagerInput* cmInputs, const PxgShape* shapes,
//     const PxsCachedTransform* transformCache, const PxBounds3* bounds,
//     const PxReal* contactDistance, const PxgParticleSystem* particleSystems,
//     PxU32* blockOffsets, PxU32* offsets) — per-pair cell-range count + scan.
//   Launch site PxgNarrowphaseCore.cpp:3151 -> {4,8,8,8,8,8,8,8,8} = 72-byte blob.
static const uint8_t kPsPrimBound1[] = { 4, 8, 8, 8, 8, 8, 8, 8, 8 };
//   ps_primitivesBoundSecondPassLaunch(const PxU32 numTests,
//     const PxU32* blockOffsets, PxU32* offsets, PxU32* totalNumPairs) —
//   block-total scan + fold.  Launch site PxgNarrowphaseCore.cpp:3190
//   -> {4,8,8,8} = 32-byte blob.
static const uint8_t kPsPrimBound2[] = { 4, 8, 8, 8 };

// gpunarrowphase cudaParticleSystem.cu:1189 (P6.4a-2 particle contact generator):
//   ps_primitivesCollisionLaunch(const bool isTGS, const PxU32 numTests,
//     const PxReal toleranceLength, const PxgContactManagerInput* cmInputs,
//     PxgShape* shapes, const PxsCachedTransform* transformCache,
//     const PxBounds3* bounds, const PxReal* contactDistance,
//     const PxReal* restDistances, const PxsMaterialData* materials,
//     const PxU32* startIndices, const PxU32* totalNumPairs,
//     PxgParticleSystem* particleSystems, const PxNodeIndex* shapeToRigidRemapTable,
//     PxgParticleContactWriter writer) — the writer is passed BY VALUE (8 device
//   pointers + PxU32 maxContacts + 4B tail pad = 72 bytes, sizeof =
//   sizeof(PxgParticleContactWriter)).  Launch site PxgNarrowphaseCore.cpp:3225.
//   Natural-alignment blob (bool @0, u32 @4, f32 @8, first ptr @16, writer @104)
//   -> {1,4,4,8x11,72} = 176-byte blob.
static const uint8_t kPsPrimCollide[] = { 1, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 72 };

// gpucommon/src/CUDA/radixSortImpl.cu (P6.2 grid radix sort):
//   radixSortMultiBlockLaunchWithoutCount(PxgRadixSortDesc* desc, PxU32 startBit)
//   / radixSortMultiCalculateRanksLaunchWithoutCount(same ABI) — the numKeys
//   push param the WithCount twins carry is dropped (sourced from desc.count),
//   leaving a device pointer + PxU32 -> {8,4} = 16-byte blob.
static const uint8_t kRadixDescNoCount[] = { 8, 4 };

// gpucommon/src/CUDA/radixSortImpl.cu:39 / :53 (P6.4b-1 plain radix twins the PBD
// particle contact-sort loop drives, PxgParticleSystemCore.cpp:1141):
//   radixSortMultiBlockLaunch(PxgRadixSortBlockDesc* desc, const PxU32 gStartBit)
//   / radixSortMultiCalculateRanksLaunch(same ABI) — SAME {desc, startBit} = {8,4}
//   push pack as the WithoutCount twins; the numKeys difference (deref'd from the
//   PxgRadixSortBlockDesc::numKeys DEVICE POINTER @48, not the inline count) is
//   entirely on-device, so this row is shared.  Launched grid (RADIX_SORT, 2, 1).
static const uint8_t kRadixDescPlain[] = { 8, 4 };

// gpunarrowphase cudaParticleSystem.cu:1687 (P6.4b-1 contact reorder):
//   ps_reorderPrimitiveContactsLaunch(const PxgParticlePrimitiveContact* particleContacts,
//     const PxI32* numParticleContacts, const PxU32* remapByRigid, const PxU32* remapByParticle,
//     PxgParticlePrimitiveContact* sortedContactsByRigid,
//     PxgParticlePrimitiveContact* sortedContactsByParticle)
//   -> 6 device pointers = {8,8,8,8,8,8} = 48-byte blob.  Launch PxgParticleSystemCore.cpp:1302.
static const uint8_t kPsReorderContacts[] = { 8, 8, 8, 8, 8, 8 };

// gpusimulationcontroller particlesystem.cu:2277 (P6.4b-1 range-find pass 1):
//   ps_findStartEndParticleFirstLaunch(PxgParticleSystem* particleSystems,
//     PxgParticlePrimitiveContact* sortedContacts, PxU32* numContacts,
//     PxU32* blockOffsets, PxU32* offsets)
//   -> 5 device pointers = {8,8,8,8,8} = 40-byte blob (particleSystems ABI-only,
//   never dereferenced).  Launch PxgParticleSystemCore.cpp:1648.
static const uint8_t kPsFindStartEnd1[] = { 8, 8, 8, 8, 8 };

// gpusimulationcontroller particlesystem.cu:2402 (P6.4b-1 range-find pass 2):
//   ps_findStartEndParticleSecondLaunch(PxgParticlePrimitiveContact* sortedContacts,
//     PxU32* numContacts, PxU32* blockOffsets, PxU32* offsets, PxU32* totalNumPairs,
//     PxU32* rangeStart, PxU32* rangeEnd)
//   -> 7 device pointers = {8,8,8,8,8,8,8} = 56-byte blob.  Launch PxgParticleSystemCore.cpp:1690.
static const uint8_t kPsFindStartEnd2[] = { 8, 8, 8, 8, 8, 8, 8 };

// gpusimulationcontroller particlesystem.cu:1238 (P6.4b-2 contact constraint prep):
//   ps_contactPrepareLaunch(PxgParticleSystem* particleSystems,
//     PxgParticlePrimitiveContact* sortedContacts, PxU32* numContacts,
//     PxgParticlePrimitiveConstraintBlock* primitiveConstraints,
//     PxgPrePrepDesc* preDesc, PxgConstraintPrepareDesc* prepareDesc,
//     float2* appliedForces, PxReal dt, bool isTGS, float4* deltaVel,
//     PxgSolverSharedDescBase* sharedDesc)
//   -> {8,8,8,8,8,8,8,4,1,8,8} = 80-byte blob (dt@56, isTGS byte@60, deltaVel@64
//   after the 8-align).  Launch PxgParticleSystemCore.cpp:1609.
static const uint8_t kPsContactPrepare[] = { 8, 8, 8, 8, 8, 8, 8, 4, 1, 8, 8 };

// gpusimulationcontroller particlesystem.cu:1531 (P6.4b-2 particle-side PC solve,
// the NON-TGS variant):
//   ps_solvePCOutputParticleDeltaVLaunch(PxgParticleSystem* particleSystems,
//     PxgParticlePrimitiveContact* sortedContacts,
//     PxgParticlePrimitiveConstraintBlock* primitiveConstraints, PxU32* numContacts,
//     PxgPrePrepDesc* prePrepDesc, PxgSolverCoreDesc* solverCoreDesc,
//     PxgSolverSharedDesc<IterativeSolveData>* sharedDesc, float4* deltaPos,
//     float2* appliedForces, PxReal dt, PxReal relaxationCoefficient,
//     bool isVelocityIteration, PxgArticulationCoreDesc* artiCoreDesc)
//   -> {8,8,8,8,8,8,8,8,8,4,4,1,8} = 96-byte blob (dt@72, coeff@76, bool byte@80,
//   artiCoreDesc@88).  Launch PxgParticleSystemCore.cpp:1948.
static const uint8_t kPsSolvePCParticle[] = { 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 4, 1, 8 };

// gpusimulationcontroller particlesystem.cu:2828 (P6.4b-2 per-particle deltaV accum):
//   ps_accumulateDeltaVParticleLaunch(PxgParticleSystem* particleSystems,
//     PxgParticlePrimitiveContact* sortedContacts, PxU32* totalNumPairs,
//     PxU32* rangeStart, PxU32* rangeEnd, float4* deltaPos)
//   -> 6 device pointers = {8,8,8,8,8,8} = 48-byte blob.  Launch
//   PxgParticleSystemCore.cpp:2005.
static const uint8_t kPsAccumulateDeltaVParticle[] = { 8, 8, 8, 8, 8, 8 };

// gpusimulationcontroller particlesystem.cu:120 (P6.6 TGS substep predictor):
//   ps_stepParticlesLaunch(const PxgParticleSystem* particleSystems,
//     const PxU32* activeParticleSystems, const PxReal dt, const PxReal invTotalDt,
//     bool isFirstIteration, const bool isTGS,
//     const bool externalForcesEveryTgsIterationEnabled, const PxVec3 gravity)
//   PxVec3 gravity by value = 12 bytes -> {8,8,4,4,1,1,1,12} = 40-byte blob (three
//   bool bytes @24/25/26, gravity @28 after the 4-align).  Launch
//   PxgParticleSystemCore.cpp:2460.
static const uint8_t kPsStepParticles[] = { 8, 8, 4, 4, 1, 1, 1, 12 };

// gpusimulationcontroller particlesystem.cu:1690 (P6.6 TGS twin of the particle-
// side PC solve).  The launch site packs the IDENTICAL array as the non-TGS solve
// (PxgParticleSystemCore.cpp:1947, the isTGS ? PS_SOLVE_PC_PARTICLE_TGS branch;
// only the CUfunction differs), so it reuses kPsSolvePCParticle:
//   ps_solvePCOutputParticleDeltaVTGSLaunch(PxgParticleSystem* particleSystems,
//     PxgParticlePrimitiveContact* sortedContacts,
//     PxgParticlePrimitiveConstraintBlock* primitiveConstraints, PxU32* numContacts,
//     PxgPrePrepDesc* prePrepDesc, PxgSolverCoreDesc* solverCoreDesc,
//     PxgSolverSharedDesc<IterativeSolveData>* sharedDesc, float4* deltaPos,
//     float2* appliedForces, PxReal dt, PxReal relaxationCoefficient,
//     bool isVelocityIteration, PxgArticulationCoreDesc* artiCoreDesc)
//   -> {8,8,8,8,8,8,8,8,8,4,4,1,8} = 96-byte blob (kPsSolvePCParticle).

// gpusimulationcontroller particlesystem.cu:5731 (P6.6 TGS velocity-iter finalize):
//   ps_finalizeParticlesLaunch(const PxgParticleSystem* particleSystems,
//     const PxU32 id, const PxReal invTotalDt, PxReal velocityScale)
//   -> {8,4,4,4} = 20 data bytes, rounded to the 8-alignment (maxAlign 8) -> 24-byte
//   blob.  Launch PxgPBDParticleSystemCore.cpp:498.
static const uint8_t kPsFinalizeParticles[] = { 8, 4, 4, 4 };

// gpucommon/src/CUDA/radixSortImpl.cu (P6.1 particle contact-sort BDA):
//   radixSortDoubleCopy(const PxU64* inValue0, PxU64* outValue0, PxU32* rank0,
//     const PxU64* inValue1, PxU64* outValue1, PxU32* rank1, const PxU32* numKeys)
//     (radixSortImpl.cu:187) and radixSortDoubleCopyHigh32Bits (radixSortImpl.cu:139),
//   which has the IDENTICAL 7-device-pointer pack (only the output width differs,
//   in the kernel body), so both share this row.  numKeys is a const PxU32*
//   DEVICE POINTER (deref'd on-device), not a host scalar -> 56-byte blob.
static const uint8_t kRadixDoubleCopy[] = { 8, 8, 8, 8, 8, 8, 8 };
//   clampMaxValue(PxU32* value, const PxU32 maxValue) (MemCopyBalanced.cu:88):
//   pointer@0 + PxU32 by value@8 -> 16-byte blob.  Reuses kPtrU32 = {8,4}.

static const PxgKernelAbiRow kRows[] = {
    { "MemCopyBalanced", kMemCopyBalanced, uint32_t(sizeof(kMemCopyBalanced)) },
    { "translateAABBsLaunch", kTranslateAABBs, uint32_t(sizeof(kTranslateAABBs)) },
    { "clearNewFlagLaunch", kOnePtr, 1u },
    { "bpSignalComplete", kOnePtr, 1u },
    { "markCreatedPairsLaunch", kOnePtr, 1u },
    // Incremental-SAP path (moved shapes): all take a single PxgBroadPhaseDesc*
    // device pointer (BDA), so kOnePtr = {8} matches the void** engine ABI.
    { "markUpdatedPairsLaunch", kOnePtr, 1u },
    { "markUpdatedPairsLaunch2", kOnePtr, 1u },
    { "markRemovedPairsLaunch", kOnePtr, 1u },
    { "markRemovedPairsProjectionsLaunch", kOnePtr, 1u },
    { "computeIncrementalComparisonHistograms_Stage1", kOnePtr, 1u },
    { "computeIncrementalComparisonHistograms_Stage2", kOnePtr, 1u },
    { "performIncrementalSAP", kOnePtr, 1u },
    { "initializeSapBox1DLaunch", kSapBoxInit, uint32_t(sizeof(kSapBoxInit)) },
    { "initializeRadixRanks", kOnePtr, 1u },
    { "updateHandles", kOnePtr, 1u },
    { "radixSortMultiBlockLaunchWithCount", kRadixDesc, uint32_t(sizeof(kRadixDesc)) },
    { "radixSortMultiCalculateRanksLaunchWithCount", kRadixDesc, uint32_t(sizeof(kRadixDesc)) },
    { "createRegionsKernel", kOnePtr, 1u },
    { "computeEndPtsHistogram", kBpDescBool, uint32_t(sizeof(kBpDescBool)) },
    { "outputEndPtsHistogram", kBpDescBool, uint32_t(sizeof(kBpDescBool)) },
    { "computeRegionsHistogram", kOnePtr, 1u },
    { "outputRegionsHistogram", kOnePtr, 1u },
    { "writeOutOverlapChecksForInsertedBoundsRegionsHistogram", kOnePtr, 1u },
    { "computeOverlapChecksForRegionsHistogram", kOnePtr, 1u },
    { "outputOverlapChecksForRegionHistogram", kOnePtr, 1u },
    { "generateFoundPairsForNewBoundsRegion", kOnePtr, 1u },
    { "accumulateReportsStage_1", kOnePtr, 1u },
    { "accumulateReportsStage_2", kOnePtr, 1u },
    { "copyReports", kOnePtr, 1u },
    { "computeStartAndActiveRegionHistogram", kOnePtr, 1u },
    { "writeOutStartAndActiveRegionHistogram", kOnePtr, 1u },
    { "outputOrderedActiveRegionHistogram", kOnePtr, 1u },
    { "computeStartRegionsHistogram", kOnePtr, 1u },
    { "outputStartRegionsHistogram", kOnePtr, 1u },
    { "initStaticKinematics", kInitStaticKin, uint32_t(sizeof(kInitStaticKin)) },
    { "ZeroBodiesTGS", kTwoPtr, uint32_t(sizeof(kTwoPtr)) },
    { "initializeSolverVelocitiesTGS", kTwoPtr, uint32_t(sizeof(kTwoPtr)) },
    { "computeAverageSolverBodyVelocityTGS", kTwoPtr, uint32_t(sizeof(kTwoPtr)) },
    { "propagateAverageSolverBodyVelocityTGS", kPropagateAvg, uint32_t(sizeof(kPropagateAvg)) },
    { "writeBackBodiesTGS", kWriteBackTGS, uint32_t(sizeof(kWriteBackTGS)) },
    { "solveWholeIslandTGS", kSolveWholeIslandTGS, uint32_t(sizeof(kSolveWholeIslandTGS)) },
    { "concludeBlocksTGS", kConcludeBlocksTGS, uint32_t(sizeof(kConcludeBlocksTGS)) },
    { "writebackBlocksTGS", kConcludeBlocksTGS, uint32_t(sizeof(kConcludeBlocksTGS)) },
    { "solveStaticBlockTGS", kSolveStaticBlockTGS, uint32_t(sizeof(kSolveStaticBlockTGS)) },
    { "propagateStaticSolverBodyVelocitiesTGS", kSolveStaticBlockTGS, uint32_t(sizeof(kSolveStaticBlockTGS)) },
    { "solveBlockUnified", kSolveBlockUnified, uint32_t(sizeof(kSolveBlockUnified)) },
    { "markActiveSlabTGS", kMarkActiveSlabTGS, uint32_t(sizeof(kMarkActiveSlabTGS)) },
    // gpusolver PGS solver kernels (P5 Step 5 BDA)
    { "ZeroBodies", kTwoPtr, uint32_t(sizeof(kTwoPtr)) },
    { "solveBlockPartition", kSolveBlockPartitionPGS, uint32_t(sizeof(kSolveBlockPartitionPGS)) },
    { "concludeBlocks", kDescsIslandPGS, uint32_t(sizeof(kDescsIslandPGS)) },
    { "writebackBlocks", kDescsIslandPGS, uint32_t(sizeof(kDescsIslandPGS)) },
    { "writeBackBodies", kDescsIslandPGS, uint32_t(sizeof(kDescsIslandPGS)) },
    { "computeAverageSolverBodyVelocity", kTwoPtr, uint32_t(sizeof(kTwoPtr)) },
    { "propagateSolverBodyVelocity", kTwoPtr, uint32_t(sizeof(kTwoPtr)) },
    { "markActiveSlabPGS", kMarkActiveSlabPGS, uint32_t(sizeof(kMarkActiveSlabPGS)) },
    { "solveStaticBlock", kSolveStaticBlockPGS, uint32_t(sizeof(kSolveStaticBlockPGS)) },
    { "propagateStaticSolverBodyVelocities", kPropagateStaticPGS, uint32_t(sizeof(kPropagateStaticPGS)) },
    { "integrateCoreParallelLaunchTGS", kIntegrateTGS, uint32_t(sizeof(kIntegrateTGS)) },
    { "integrateCoreParallelLaunch", kIntegrateTGS, uint32_t(sizeof(kIntegrateTGS)) },
    // solverMultiBlock.cu DMA-back glue (P5 Step 5 BDA)
    { "dmaBackChangedElems", kTwoPtr, uint32_t(sizeof(kTwoPtr)) },
    { "dmaConstraintResidual", kTwoPtrU32, uint32_t(sizeof(kTwoPtrU32)) },
    // constraintBlockPrePrep.cu: clearFrictionPatchCounts (P5 Step 5 BDA)
    { "clearFrictionPatchCounts", kTwoPtrU32, uint32_t(sizeof(kTwoPtrU32)) },
    // gpusimulationcontroller: updateTransformAndBoundArray.cu / updateBodiesAndShapes.cu
    { "updateTransformCacheAndBoundArrayLaunch", kOnePtr, 1u },
    { "updateBodiesLaunch", kOnePtr, 1u },
    { "updateChangedAABBMgrHandlesLaunch", kOnePtr, 1u },
    { "updateShapesLaunch", kUpdateShapes, uint32_t(sizeof(kUpdateShapes)) },
    // updateBodiesAndShapes.cu: updateJointsLaunch(const PxgUpdatedJointsDesc*)
    // -> single device pointer {8} (P5 Step 5 tranche 12: the joint CPU->GPU scatter)
    { "updateJointsLaunch", kOnePtr, 1u },
    { "preIntegrationLaunchTGS", kPreIntegrateTGS, uint32_t(sizeof(kPreIntegrateTGS)) },
    { "preIntegrationLaunch", kPreIntegratePGS, uint32_t(sizeof(kPreIntegratePGS)) },
    // gpusolver contact/joint constraint prep (P5 Step 4 BDA; PGS twins P5 Step 5)
    { "rigidSumInternalContactAndJointBatches1", kPtrU32, uint32_t(sizeof(kPtrU32)) },
    { "rigidSumInternalContactAndJointBatches2", kThreePtrU32, uint32_t(sizeof(kThreePtrU32)) },
    { "constraintContactBlockPrePrepLaunch", kTwoPtr, uint32_t(sizeof(kTwoPtr)) },
    { "constraint1DBlockPrePrepLaunch", kTwoPtr, uint32_t(sizeof(kTwoPtr)) },
    { "constraint1DPrePrepLaunch", kTwoPtr, uint32_t(sizeof(kTwoPtr)) },
    { "contactConstraintBlockPrepareParallelLaunchTGS", kTwoPtr, uint32_t(sizeof(kTwoPtr)) },
    { "contactConstraintBlockPrepareParallelLaunch", kTwoPtr, uint32_t(sizeof(kTwoPtr)) },
    { "jointConstraintBlockPrepareParallelLaunch", kTwoPtr, uint32_t(sizeof(kTwoPtr)) },
    // jointConstraintBlockPrepareParallelLaunchTGS (P5 Step 5 tranche 13 BDA):
    // now converted to the engine ABI, taking the same
    // (PxgConstraintPrepareDesc*, PxgSolverSharedDesc<IterativeSolveDataTGS>*)
    // pointer pack as its PGS twin and the TGS contact prep
    // (PxgTGSCudaSolverCore::jointConstraintPrepareParallel L933-949), so it
    // gets the {8,8} kTwoPtr row.  This unblocks the TGS jointed scene.
    { "jointConstraintBlockPrepareParallelLaunchTGS", kTwoPtr, uint32_t(sizeof(kTwoPtr)) },
    // gpunarrowphase convex-convex core + contact-stream compress (P5 Step 4 BDA)
    { "convexConvexNphase_stage1Kernel", kCvxStage1, uint32_t(sizeof(kCvxStage1)) },
    { "convexConvexNphase_stage2Kernel", kCvxStage2, uint32_t(sizeof(kCvxStage2)) },
    { "finishContactsKernel", kFinishContacts, uint32_t(sizeof(kFinishContacts)) },
    { "compressContactStage1", kCompress1, uint32_t(sizeof(kCompress1)) },
    { "compressContactStage2", kCompress2, uint32_t(sizeof(kCompress2)) },
    { "updateFrictionPatches", kFrictionPatch, uint32_t(sizeof(kFrictionPatch)) },
    { "prepareLostFoundPairs_Stage1", kLostFound1, uint32_t(sizeof(kLostFound1)) },
    { "prepareLostFoundPairs_Stage2", kLostFound2, uint32_t(sizeof(kLostFound2)) },
    { "boxBoxNphase_Kernel", kBoxBoxNphase, uint32_t(sizeof(kBoxBoxNphase)) },
    { "sphereNphase_Kernel", kSphereNphase, uint32_t(sizeof(kSphereNphase)) },
    { "convexPlaneNphase_Kernel", kConvexPlaneNphase, uint32_t(sizeof(kConvexPlaneNphase)) },
    // gpusimulationcontroller PBD particle kernels (P6.0 BDA)
    { "ps_updateUnsortedArrayLaunch", kPsUpdateUnsorted, uint32_t(sizeof(kPsUpdateUnsorted)) },
    { "ps_preIntegrateLaunch", kPsPreIntegrate, uint32_t(sizeof(kPsPreIntegrate)) },
    { "ps_integrateLaunch", kPsIntegrate, uint32_t(sizeof(kPsIntegrate)) },
    { "ps_updateUserBufferLaunch", kPsUpdateUserBuffer, uint32_t(sizeof(kPsUpdateUserBuffer)) },
    // gpusimulationcontroller FEMCloth kernels (P7.0 FEMCloth free-fall BDA)
    { "cloth_preIntegrateLaunch", kClothPreIntegrate, uint32_t(sizeof(kClothPreIntegrate)) },
    { "cloth_finalizeVelocitiesLaunch", kClothFinalizeVel, uint32_t(sizeof(kClothFinalizeVel)) },
    { "cloth_sleeping", kClothSleeping, uint32_t(sizeof(kClothSleeping)) },
    // gpusimulationcontroller FEMCloth in-plane membrane energy solve (P7.1 BDA)
    { "cloth_solveNonSharedTriangleEnergyLaunch", kClothSolveTriEnergy, uint32_t(sizeof(kClothSolveTriEnergy)) },
    { "cloth_solveNonSharedTriangleEnergyClusterLaunch", kClothSolveTriEnergy, uint32_t(sizeof(kClothSolveTriEnergy)) },
    // gpusimulationcontroller FEMCloth triangle-pair shared-energy + bending solve (P7.1 tranche 2 BDA)
    { "cloth_solveTrianglePairEnergyLaunch", kClothSolveTriPairEnergy, uint32_t(sizeof(kClothSolveTriPairEnergy)) },
    { "cloth_solveTrianglePairEnergyClusterLaunch", kClothSolveTriPairEnergy, uint32_t(sizeof(kClothSolveTriPairEnergy)) },
    { "cloth_averageTrianglePairVertsLaunch", kClothAvgPairVerts, uint32_t(sizeof(kClothAvgPairVerts)) },
    // gpusimulationcontroller FEMCloth rigid-attachment prep + PGS/TGS solve (P7.1 tranche 3a BDA)
    { "cloth_rigidAttachmentPrepareLaunch", kClothRigidAttachPrep, uint32_t(sizeof(kClothRigidAttachPrep)) },
    { "cloth_solveRigidClothAttachmentLaunch", kClothRigidAttachSolve, uint32_t(sizeof(kClothRigidAttachSolve)) },
    { "cloth_solveRigidClothAttachmentTGSLaunch", kClothRigidAttachSolve, uint32_t(sizeof(kClothRigidAttachSolve)) },
    { "cloth_applyExternalDeltasLaunch", kClothApplyExtDeltas, uint32_t(sizeof(kClothApplyExtDeltas)) },
    // gpunarrowphase FEMCloth-vs-primitive contact GENERATION (P7.1 tranche 4a BDA)
    { "cloth_midphaseGenerateVertexPairsLaunch", kClothMidphaseVertPairs, uint32_t(sizeof(kClothMidphaseVertPairs)) },
    { "cloth_planeVertContactGenLaunch", kClothVertContactGen, uint32_t(sizeof(kClothVertContactGen)) },
    { "cloth_SphereVertexContactGenLaunch", kClothVertContactGen, uint32_t(sizeof(kClothVertContactGen)) },
    // gpunarrowphase FEMCloth-vs-rigid contact SORT / reorder (P7.1 tranche 4b-1 BDA)
    { "fem_reorderRigidContactsLaunch", kFemReorderRSContacts, uint32_t(sizeof(kFemReorderRSContacts)) },
    // gpusimulationcontroller FEMCloth-vs-rigid contact PREPARE + query/solve (P7.1 tranche 4b-2/3 BDA)
    { "cloth_rigidContactPrepareLaunch", kClothRigidContactPrep, uint32_t(sizeof(kClothRigidContactPrep)) },
    { "cloth_queryRigidClothContactReferenceCountLaunch", kClothRigidContactQuery, uint32_t(sizeof(kClothRigidContactQuery)) },
    { "cloth_queryRigidClothContactReferenceCountTGSLaunch", kClothRigidContactQuery, uint32_t(sizeof(kClothRigidContactQuery)) },
    { "cloth_solveRigidClothCollisionLaunch", kClothRigidContactSolve, uint32_t(sizeof(kClothRigidContactSolve)) },
    { "cloth_solveRigidClothCollisionTGSLaunch", kClothRigidContactSolve, uint32_t(sizeof(kClothRigidContactSolve)) },
    // gpunarrowphase FEMCloth SELF-collision midphase + contact GENERATION (P7.2 tranche 1 BDA)
    { "cloth_selfCollisionMidphaseVTLaunch", kClothSelfCollisionMidphase, uint32_t(sizeof(kClothSelfCollisionMidphase)) },
    { "cloth_selfCollisionMidphaseEELaunch", kClothSelfCollisionMidphase, uint32_t(sizeof(kClothSelfCollisionMidphase)) },
    // gpusimulationcontroller FEMCloth self-contact prepare + count + solve + validity (P7.2 tranche 2 BDA)
    { "cloth_clothContactPrepareLaunch", kClothClothContactPrep, uint32_t(sizeof(kClothClothContactPrep)) },
    { "cloth_queryClothClothContactVTCountLaunch", kClothClothContactQuery, uint32_t(sizeof(kClothClothContactQuery)) },
    { "cloth_queryClothClothContactEECountLaunch", kClothClothContactQuery, uint32_t(sizeof(kClothClothContactQuery)) },
    { "cloth_solveClothClothDeltaVTLaunch", kClothClothContactSolve, uint32_t(sizeof(kClothClothContactSolve)) },
    { "cloth_solveClothClothDeltaEELaunch", kClothClothContactSolve, uint32_t(sizeof(kClothClothContactSolve)) },
    { "cloth_updateClothContactValidityLaunch", kClothUpdateContactValidity, uint32_t(sizeof(kClothUpdateContactValidity)) },
    // gpusimulationcontroller FEMCloth BVH bottom-up bounds refit (P7.2 tranche 3a BDA)
    { "cloth_refitBoundLaunch", kClothRefitBound, uint32_t(sizeof(kClothRefitBound)) },
    // gpusimulationcontroller softbody (FEM-volume) free-fall predict/integrate (P8.0 BDA)
    { "sb_gm_preIntegrateLaunch", kSbPreIntegrate, uint32_t(sizeof(kSbPreIntegrate)) },
    { "sb_refitBoundLaunch", kSbRefitBound, uint32_t(sizeof(kSbRefitBound)) },
    // gpusimulationcontroller softbody free-fall chain: substep integrate + finalize + sleep (P8.0 tranche 3)
    { "sb_gm_stepSoftbodyLaunch", kSbStepSoftbody, uint32_t(sizeof(kSbStepSoftbody)) },
    { "sb_gm_finalizeVelocitiesLaunch", kSbFinalizeVelocities, uint32_t(sizeof(kSbFinalizeVelocities)) },
    { "sb_sleeping", kSbSleeping, uint32_t(sizeof(kSbSleeping)) },
    // gpusimulationcontroller softbody FEM-volume solve bracket (P8.0 tranche 4)
    { "sb_gm_zeroTetMultipliers", kSbZeroTetMultipliers, uint32_t(sizeof(kSbZeroTetMultipliers)) },
    { "sb_gm_applyExternalDeltasLaunch", kSbApplyExternalDeltas, uint32_t(sizeof(kSbApplyExternalDeltas)) },
    // gpusimulationcontroller softbody tetrahedron rotation extractors (P8.0 tranche 5)
    { "sb_gm_updateTetrahedraRotationsLaunch", kSbZeroTetMultipliers, uint32_t(sizeof(kSbZeroTetMultipliers)) },
    { "sb_updateTetrahedraRotationsLaunch", kSbZeroTetMultipliers, uint32_t(sizeof(kSbZeroTetMultipliers)) },
    // gpusimulationcontroller softbody FEM-volume constraint solve (P8.0 tranche 6)
    { "sb_gm_cp_solveTetrahedronsPartitionLaunch", kSbSolveTetraPartition, uint32_t(sizeof(kSbSolveTetraPartition)) },
    // gpusimulationcontroller PBD particle grid build + reorder (P6.2 BDA)
    { "ps_calculateHashLaunch", kPsCalcHash, uint32_t(sizeof(kPsCalcHash)) },
    { "ps_reorderDataAndFindCellStartLaunch", kPsReorder, uint32_t(sizeof(kPsReorder)) },
    // gpusimulationcontroller PBD particle self-collision + solve (P6.3 BDA)
    // ps_selfCollisionLaunch(const PxgParticleSystem*, PxU32 id) -> {8,4}
    { "ps_selfCollisionLaunch", kPsReorder, uint32_t(sizeof(kPsReorder)) },
    // ps_solveOneWayContactDeltaVLaunch(PxgParticleSystem*, const PxU32*, PxReal invDt,
    //   PxReal dt, PxReal biasCoefficient, bool isVelocityIteration) -> {8,8,4,4,4,1}
    { "ps_solveOneWayContactDeltaVLaunch", kPsSolveOneWay, uint32_t(sizeof(kPsSolveOneWay)) },
    // ps_applyDeltaLaunch(PxgParticleSystem*, const PxU32*, PxReal dt) -> {8,8,4}
    { "ps_applyDeltaLaunch", kPsUpdateUserBuffer, uint32_t(sizeof(kPsUpdateUserBuffer)) },
    // ps_updateParticleLaunch(const PxgParticleSystem*, const PxU32*, PxReal invDt,
    //   bool skipNewPositionAdjustment) -> {8,8,4,1}
    { "ps_updateParticleLaunch", kPsUpdateParticle, uint32_t(sizeof(kPsUpdateParticle)) },
    // gpusimulationcontroller/gpunarrowphase (P6.4a-1 PBD bound / cell-range)
    { "ps_updateBoundFirstPassLaunch", kPsUpdateBound1, uint32_t(sizeof(kPsUpdateBound1)) },
    { "ps_updateBoundSecondPassLaunch", kPsUpdateBound2, uint32_t(sizeof(kPsUpdateBound2)) },
    { "ps_primitivesBoundFirstPassLaunch", kPsPrimBound1, uint32_t(sizeof(kPsPrimBound1)) },
    { "ps_primitivesBoundSecondPassLaunch", kPsPrimBound2, uint32_t(sizeof(kPsPrimBound2)) },
    // gpunarrowphase (P6.4a-2 PBD particle-vs-primitive contact generator)
    { "ps_primitivesCollisionLaunch", kPsPrimCollide, uint32_t(sizeof(kPsPrimCollide)) },
    // gpucommon radixSortImpl.cu:39 / :53 — the PLAIN radix twins (P6.4b-1 BDA);
    // same {8,4} pack as the WithoutCount twins (numKeys deref'd on-device).
    { "radixSortMultiBlockLaunch", kRadixDescPlain, uint32_t(sizeof(kRadixDescPlain)) },
    { "radixSortMultiCalculateRanksLaunch", kRadixDescPlain, uint32_t(sizeof(kRadixDescPlain)) },
    // gpusimulationcontroller/gpunarrowphase (P6.4b-1 PBD contact sort/organize)
    { "ps_reorderPrimitiveContactsLaunch", kPsReorderContacts, uint32_t(sizeof(kPsReorderContacts)) },
    { "ps_findStartEndParticleFirstLaunch", kPsFindStartEnd1, uint32_t(sizeof(kPsFindStartEnd1)) },
    { "ps_findStartEndParticleSecondLaunch", kPsFindStartEnd2, uint32_t(sizeof(kPsFindStartEnd2)) },
    // gpusimulationcontroller (P6.4b-2 PBD particle-side contact solve)
    { "ps_contactPrepareLaunch", kPsContactPrepare, uint32_t(sizeof(kPsContactPrepare)) },
    { "ps_solvePCOutputParticleDeltaVLaunch", kPsSolvePCParticle, uint32_t(sizeof(kPsSolvePCParticle)) },
    { "ps_accumulateDeltaVParticleLaunch", kPsAccumulateDeltaVParticle, uint32_t(sizeof(kPsAccumulateDeltaVParticle)) },
    // gpusimulationcontroller (P6.6 PBD particle-side TGS contact-solve loop)
    { "ps_stepParticlesLaunch", kPsStepParticles, uint32_t(sizeof(kPsStepParticles)) },
    // ps_solvePCOutputParticleDeltaVTGSLaunch shares the non-TGS solve's 96-byte pack.
    { "ps_solvePCOutputParticleDeltaVTGSLaunch", kPsSolvePCParticle, uint32_t(sizeof(kPsSolvePCParticle)) },
    { "ps_finalizeParticlesLaunch", kPsFinalizeParticles, uint32_t(sizeof(kPsFinalizeParticles)) },
    // gpucommon radixSortImpl.cu:69 / :81 — the WithoutCount radix twins (P6.2 BDA)
    { "radixSortMultiBlockLaunchWithoutCount", kRadixDescNoCount, uint32_t(sizeof(kRadixDescNoCount)) },
    { "radixSortMultiCalculateRanksLaunchWithoutCount", kRadixDescNoCount, uint32_t(sizeof(kRadixDescNoCount)) },
    // gpucommon particle contact-sort kernels (P6.1 BDA)
    // radixSortImpl.cu:187 / :139 — the twin rank-indexed gathers share the pack.
    { "radixSortDoubleCopy", kRadixDoubleCopy, uint32_t(sizeof(kRadixDoubleCopy)) },
    { "radixSortDoubleCopyHigh32Bits", kRadixDoubleCopy, uint32_t(sizeof(kRadixDoubleCopy)) },
    // MemCopyBalanced.cu:88 — clampMaxValue(PxU32* value, const PxU32 maxValue) -> {8,4}
    { "clampMaxValue", kPtrU32, uint32_t(sizeof(kPtrU32)) },
};

inline size_t abiAlign(size_t size)
{
    // Largest power of two that divides the size (capped at 16) — the natural
    // alignment of a tightly-packed scalar aggregate; matches paramAlign() in
    // PxgVulkanCudaContext.cpp so the void** blob layout equals the sized one.
    if (size == 0) return 1;
    if (size % 16 == 0) return 16;
    if (size % 8 == 0)  return 8;
    if (size % 4 == 0)  return 4;
    if (size % 2 == 0)  return 2;
    return 1;
}

} // anonymous namespace

const PxgKernelAbiRow* pxgFindKernelAbi(const char* cudaName)
{
    if (!cudaName)
        return nullptr;
    for (const PxgKernelAbiRow& row : kRows)
        if (std::strcmp(row.cudaName, cudaName) == 0)
            return &row;
    return nullptr;
}

uint32_t pxgKernelAbiPushSize(const PxgKernelAbiRow& row)
{
    size_t offset = 0;
    size_t maxAlign = 1;
    for (uint32_t i = 0; i < row.argCount; ++i)
    {
        const size_t a = abiAlign(row.argSizes[i]);
        if (a > maxAlign)
            maxAlign = a;
        offset = (offset + (a - 1)) & ~(a - 1);
        offset += row.argSizes[i];
    }
    offset = (offset + (maxAlign - 1)) & ~(maxAlign - 1);
    return uint32_t(offset);
}

} // namespace gpuslang
} // namespace physx
