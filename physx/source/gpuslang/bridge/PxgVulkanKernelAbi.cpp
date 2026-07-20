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

// integration.cu: integrateCoreParallelLaunchTGS(PxU32 offset, desc, sharedDesc,
//   PxU32* islandIds, PxU32* islandStaticTouchCounts, PxU32* numCountedInteractions)
static const uint8_t kIntegrateTGS[] = { 4, 8, 8, 8, 8, 8 };

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
    { "integrateCoreParallelLaunchTGS", kIntegrateTGS, uint32_t(sizeof(kIntegrateTGS)) },
    // gpusimulationcontroller: updateTransformAndBoundArray.cu / updateBodiesAndShapes.cu
    { "updateTransformCacheAndBoundArrayLaunch", kOnePtr, 1u },
    { "updateBodiesLaunch", kOnePtr, 1u },
    { "updateChangedAABBMgrHandlesLaunch", kOnePtr, 1u },
    { "updateShapesLaunch", kUpdateShapes, uint32_t(sizeof(kUpdateShapes)) },
    { "preIntegrationLaunchTGS", kPreIntegrateTGS, uint32_t(sizeof(kPreIntegrateTGS)) },
    // gpusolver contact/joint constraint prep (P5 Step 4 BDA)
    { "rigidSumInternalContactAndJointBatches1", kPtrU32, uint32_t(sizeof(kPtrU32)) },
    { "rigidSumInternalContactAndJointBatches2", kThreePtrU32, uint32_t(sizeof(kThreePtrU32)) },
    { "constraintContactBlockPrePrepLaunch", kTwoPtr, uint32_t(sizeof(kTwoPtr)) },
    { "contactConstraintBlockPrepareParallelLaunchTGS", kTwoPtr, uint32_t(sizeof(kTwoPtr)) },
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
