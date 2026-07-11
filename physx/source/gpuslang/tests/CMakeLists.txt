## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions
## are met:
##  * Redistributions of source code must retain the above copyright
##    notice, this list of conditions and the following disclaimer.
##  * Redistributions in binary form must reproduce the above copyright
##    notice, this list of conditions and the following disclaimer in the
##    documentation and/or other materials provided with the distribution.
##  * Neither the name of NVIDIA CORPORATION nor the names of its
##    contributors may be used to endorse or promote products derived
##    from this software without specific prior written permission.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
## EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
## IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
## PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
## CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
## EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
## PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
## PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
## OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
## (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
## OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
##
## Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

# Standalone verification suite for the Slang port of the PhysX GPU kernels.
#
#   cmake -S physx/source/gpuslang/tests -B build-gpuslang
#   cmake --build build-gpuslang -j
#   ctest --test-dir build-gpuslang --output-on-failure
#
# Requirements: a C++17 compiler, the Vulkan SDK (loader + headers), a Vulkan
# ICD (any conformant GPU driver, or CPU implementations such as Mesa
# llvmpipe/lavapipe for GPU-less CI machines) and the slangc compiler
# (https://github.com/shader-slang/slang; also shipped inside the slangtorch
# python wheel as slangtorch/bin/slangc).

cmake_minimum_required(VERSION 3.16)
project(PhysXGpuSlangTests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)

find_program(PXG_SLANGC slangc DOC "Path to the slangc compiler")
if(NOT PXG_SLANGC)
    message(FATAL_ERROR "slangc not found; pass -DPXG_SLANGC=/path/to/slangc")
endif()

# -O0 keeps slangc from invoking the external spirv-opt pass (which needs the
# optional slang-glslang shared library).  The SPIR-V still passes spirv-val
# and drivers run their own optimizers; bump this when a full slang
# distribution is available.
# -fvk-use-entrypoint-name keeps the source entry-point names in the SPIR-V
# (slangc otherwise renames a single entry point to "main").
set(PXG_SLANG_FLAGS -O0 -fvk-use-entrypoint-name CACHE STRING "Extra flags passed to slangc")

set(GPUSLANG_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(SHADER_DIR ${GPUSLANG_ROOT}/shaders)
set(SHADER_OUT_DIR ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_OUT_DIR})

set(SHADER_COMMON_DEPS
    ${SHADER_DIR}/pxgSlangCommon.slang
    ${SHADER_DIR}/pxgSlangScan.slang
    ${SHADER_DIR}/pxgSlangPrepCommon.slang)

# pxg_compile_shader(<file-stem> <entry> [<entry> ...])
function(pxg_compile_shader stem)
    set(entries "")
    foreach(entry ${ARGN})
        list(APPEND entries -entry ${entry})
    endforeach()
    add_custom_command(
        OUTPUT ${SHADER_OUT_DIR}/${stem}.spv
        COMMAND ${PXG_SLANGC} ${SHADER_DIR}/${stem}.slang -target spirv ${entries}
                ${PXG_SLANG_FLAGS} -o ${SHADER_OUT_DIR}/${stem}.spv
        DEPENDS ${SHADER_DIR}/${stem}.slang ${SHADER_COMMON_DEPS}
        COMMENT "slangc ${stem}.slang -> ${stem}.spv"
        VERBATIM)
    list(APPEND PXG_SHADER_OUTPUTS ${SHADER_OUT_DIR}/${stem}.spv)
    set(PXG_SHADER_OUTPUTS ${PXG_SHADER_OUTPUTS} PARENT_SCOPE)
endfunction()

pxg_compile_shader(scan scanPerBlockKernel addBlockSumsKernel)
pxg_compile_shader(radixSort radixSortHistogram radixSortCalculateRanks)
pxg_compile_shader(radixSort64 radixSortCopyHigh32Bits)
pxg_compile_shader(radixSortGather radixSortCopy radixSortCopy2 radixSortCopyBits2 radixSortDoubleCopy radixSortDoubleCopyHigh32Bits)
pxg_compile_shader(preIntegration preIntegrationLaunch initStaticKinematics)
pxg_compile_shader(integration integrateCoreParallelLaunch integrateCoreParallelLaunchTGS)
pxg_compile_shader(updateBounds updateTransformCacheAndBoundArrayLaunch setRigidDynamicGlobalPose)
pxg_compile_shader(bodyUpdate updateBodiesLaunch updateBodiesLaunchDirectAPI updateShapesLaunch
                   updateBodyExternalVelocitiesLaunch)
pxg_compile_shader(directApi getRigidDynamicGlobalPose getRigidDynamicLinearVelocity
                   getRigidDynamicAngularVelocity getRigidDynamicLinearAcceleration
                   getRigidDynamicAngularAcceleration setRigidDynamicLinearVelocity
                   setRigidDynamicAngularVelocity setRigidDynamicForce setRigidDynamicTorque)
pxg_compile_shader(scan4x4 scanPerBlockKernel4x4 addBlockSumsKernel4x4)
pxg_compile_shader(radixFourBit radixFourBitCountPerBlockKernel radixFourBitReorderKernel)
pxg_compile_shader(reorderData reorderKernel)
pxg_compile_shader(solverPGS solveBlockPartition concludeBlocks writebackBlocks writeBackBodies
                   computeAverageSolverBodyVelocity propagateSolverBodyVelocity)
pxg_compile_shader(solverTGS solveBlockPartitionTGS concludeBlocksTGS writebackBlocksTGS
                   writeBackBodiesTGS computeAverageSolverBodyVelocityTGS
                   propagateAverageSolverBodyVelocityTGS)
pxg_compile_shader(constraintPrepPGS contactConstraintBlockPrepareParallelLaunch
                   jointConstraintBlockPrepareParallelLaunch)
pxg_compile_shader(constraintPrepTGS contactConstraintBlockPrepareParallelLaunchTGS
                   jointConstraintBlockPrepareParallelLaunchTGS)
pxg_compile_shader(thresholdStream initialRanksAndBodyIndexB initialRanksAndBodyIndexA
                   reorganizeThresholdElements computeAccumulateThresholdStream
                   outputAccumulateThresholdStream writeoutAccumulatedForcePerObject
                   computeExceededForceThresholdElementIndice outputExceededForceThresholdElementIndice)
pxg_compile_shader(thresholdForceChange setThresholdElementsMask setPersistentForceElementMask
                   computeThresholdElementMaskIndices outputThresholdPairsMaskIndices
                   createForceChangeThresholdElements)
pxg_compile_shader(solverZeroBodies ZeroBodies ZeroBodiesTGS)
pxg_compile_shader(solverInitVelocitiesTGS initializeSolverVelocitiesTGS)
pxg_compile_shader(solverApplyGravityTGS applyTGSSubstepGravity)
pxg_compile_shader(solverDmaBack dmaBackChangedElems dmaConstraintResidual)
pxg_compile_shader(npPairManagement initializeKeepDropBuffer markKeepDropBuff processKeepDropBuff
                   accumulateKeepDrop initializeManifolds)
pxg_compile_shader(npPairRemoveSwap removeContactManagers_Stage5 removeContactManagers_Stage5_CvxTri)
pxg_compile_shader(npSphereCollision sphereNphase_sphereSphere sphereNphase_spherePlane
                   sphereNphase_sphereCapsule sphereNphase_sphereBox)
pxg_compile_shader(npCapsuleCollision planeCapsule capsuleCapsule)
pxg_compile_shader(npMaterialCombine combineMaterials)
pxg_compile_shader(npContactPatch setContactPointAndForcePointers registerContactPatch)
pxg_compile_shader(npInsertPatch insertIntoPatchStream)
pxg_compile_shader(npCompressContacts compressContactStage1 updateFrictionPatches)
pxg_compile_shader(npCompressContacts2 compressContactStage2)
pxg_compile_shader(npTriangle triPointDistance triLineIntersect)
pxg_compile_shader(npSegTriangle segTriDistance)
pxg_compile_shader(npCapTriangle capTriGenContacts capTriGenEE)
pxg_compile_shader(npTriTri triTriSat)
pxg_compile_shader(npHeightfield hfGetTriangle hfGetAdjacency hfGetTriangleScaled)
pxg_compile_shader(npHfMidphase hfmComputePairs hfmOutputPairs)
pxg_compile_shader(npHfPointQuery hfGridRange hfTestLocalPoint)
pxg_compile_shader(npConvexTri ctMinkowskiFace ctEdgePairPassed)
pxg_compile_shader(npConvexSat ctSupportConvex ctSupportTriangle ctSatTriangleNormal)
pxg_compile_shader(npConvexFaceSat ctSatConvexFaceNormals)
pxg_compile_shader(npConvexEdgeSat ctEdgeCrossSat)
pxg_compile_shader(npConvexFeature ctSelectFeature)
pxg_compile_shader(npConvexTriOut ctConvexTriOut)
pxg_compile_shader(npConvexPolyClip ctPolyClip)
pxg_compile_shader(npCvxTriEmit ctCvxTriEmit)
pxg_compile_shader(npCvxTriFind ctCvxTriFind)
pxg_compile_shader(npCvxTriPurge ctCvxTriPurge)
pxg_compile_shader(npManifold mfRefresh mfInvalidate)
pxg_compile_shader(npSegDist sdPointSegment sdSegmentSegment)
pxg_compile_shader(npTriMask ctBarycentric ctActiveMaskVW ctActiveMaskFromNormal ctActiveMaskPoint)
pxg_compile_shader(npNpUtils nuNext3Prev3 nuEdgeFlags nuDelayContact)
pxg_compile_shader(npConvexReduce ctReducePoints)
pxg_compile_shader(npTriMesh ctTriNormal ctTriBarycentric)
pxg_compile_shader(npClosest ctClosestPtTriangle ctClosestPtEdge)
pxg_compile_shader(npTetra ctTriBBox ctTetBBox ctTetBary)
pxg_compile_shader(npDefFilter ctFindParticleRigid ctFindRigid ctFindNonRigid ctFindRange ctEdgeAuth)
pxg_compile_shader(npBv32 ctBv32Query)
pxg_compile_shader(npMidphaseCvx ctMidphaseCvx)
pxg_compile_shader(npMidphaseDeform ctClothTri ctSoftbodyTet ctSoftbodyBoxTet)
pxg_compile_shader(npMidphaseAlloc ctMidphaseAlloc ctCvxTriAccessors)
pxg_compile_shader(npClothCloth ctClothClothQuery)
pxg_compile_shader(gridCal ctGridCal)
pxg_compile_shader(contactReduce ctContactReduce)
pxg_compile_shader(npSbTet ctSbTet)
pxg_compile_shader(npSbSelf ctSbSelfQuery)
pxg_compile_shader(npSbSb ctSbSbQuery)
pxg_compile_shader(npTetTriMid ctTetTriMid)
pxg_compile_shader(psPrimContact ctPsPrimContact)
pxg_compile_shader(psBounds ctPsBoundFirstPass ctPsBoundSecondPass ctPsReorderContacts ctPsGridHash)
pxg_compile_shader(psPrimCollide ctPsPrimCollide)
pxg_compile_shader(psDiffuseCollide ctPsDiffuseCollide)
pxg_compile_shader(fcPrim ctFcPrim)
pxg_compile_shader(psMeshMid ctPsMeshMid)
pxg_compile_shader(psHfCollide ctPsHfCollide)
pxg_compile_shader(fcPrim2 ctFcPrim2)
pxg_compile_shader(npPolyClip4 ctPolyClip4)
pxg_compile_shader(fcConvexTri ctFcConvexTri)
pxg_compile_shader(fcVertexConvex ctFcVertexConvex)
pxg_compile_shader(fcStaticTriCloth fcStaticTriClothKernel)
pxg_compile_shader(fcMidphase fmVertexAabb fmSphereTree fmHfBound sbHfBound)
pxg_compile_shader(npSbPrim sbTetPlane sbEmit sbParticle sbClothVert)
pxg_compile_shader(npSbRemap rmOther rmFem rmReorder)
pxg_compile_shader(npCvxHfMid chMidphase sbShrink)
pxg_compile_shader(npTriPlaneNp tpNarrowphase)
pxg_compile_shader(npTriHfNp thNarrowphase)
pxg_compile_shader(npCvxMeshNp cmSphereTri cmCapsuleTri)
pxg_compile_shader(npCvxHfNp chSphereTri)
pxg_compile_shader(npEdgeEdge ctLineLine ctEdgeEdge)
pxg_compile_shader(npBoxClip boxContains boxIntersectSegmentAABB)
pxg_compile_shader(npBoxIncident boxIncidentPolygon4 boxCalculateBounds4)
pxg_compile_shader(npBoxContacts boxCalculateContacts)
pxg_compile_shader(npBoxSat boxSat)
pxg_compile_shader(npBoxFeature boxFeatureSetup)
pxg_compile_shader(npBoxReduce boxCompactContacts)
pxg_compile_shader(broadphaseSetup translateAABBsLaunch markCreatedPairsLaunch initializeSapBox1DLaunch
                   initializeRadixRanks updateHandles)
pxg_compile_shader(broadphaseReports accumulateReportsStage_1 accumulateReportsStage_2 copyReports)
pxg_compile_shader(broadphaseIncrementalHistogram computeIncrementalComparisonHistograms_Stage1
                   computeIncrementalComparisonHistograms_Stage2)
pxg_compile_shader(broadphaseEndPtHistogram computeEndPtsHistogram outputEndPtsHistogram)
pxg_compile_shader(broadphaseIncrementalSap performIncrementalSAP)
pxg_compile_shader(broadphaseRegionHistogram computeRegionsHistogram outputRegionsHistogram)
pxg_compile_shader(broadphaseCreateRegions createRegionsKernel)
pxg_compile_shader(broadphaseActiveRegionHistogram writeOutStartAndActiveRegionHistogram
                   computeStartAndActiveRegionHistogram outputOrderedActiveRegionHistogram)
pxg_compile_shader(broadphaseOverlapCheckHistogram writeOutOverlapChecksForInsertedBoundsRegionsHistogram
                   computeOverlapChecksForRegionsHistogram outputOverlapChecksForRegionHistogram)
pxg_compile_shader(broadphaseNewBoundsPairs generateFoundPairsForNewBoundsRegion)
pxg_compile_shader(broadphaseMarkPairs markRemovedPairsLaunch markRemovedPairsProjectionsLaunch
                   markUpdatedPairsLaunch markUpdatedPairsLaunch2)
pxg_compile_shader(broadphaseClearNewFlag clearNewFlagLaunch bpSignalComplete)
pxg_compile_shader(broadphaseAggregateBounds markAggregateBoundsUpdatedBitmap updateAggregateBounds)
pxg_compile_shader(broadphaseAggregateDirty updateDirtyAggregate clearDirtyAggregates)
pxg_compile_shader(broadphaseAggregatePairsAdd addAggPairsStage1 addAggPairsStage2)
pxg_compile_shader(broadphaseAggregateFinalize aggCopyReports removeAggPairsStage3
                   markAddedAndDeletedAggregatedBounds)
pxg_compile_shader(broadphaseAggregatePairsRemove removeAggPairsStage1 removeAggPairsStage2)
pxg_compile_shader(broadphaseAggregateBoxPruning aggregateBipartiteBoxPruning)
pxg_compile_shader(broadphaseAggPairCollision doAggPairCollisions)
pxg_compile_shader(broadphaseAggSelfCollision doSelfCollision)
pxg_compile_shader(broadphaseAggSortProjections sortAndUpdateAggregateProjections)
pxg_compile_shader(memCopy MemCopyBalanced)
pxg_compile_shader(clampValues clampMaxValue clampMaxValues)
pxg_compile_shader(skinning interleaveBuffers zeroNormals normalVectorsAreaWeighted
                   normalizeNormals interpolateSkinnedClothVertices interpolateSkinnedSoftBodyVertices)

add_custom_target(gpuslang-shaders ALL DEPENDS ${PXG_SHADER_OUTPUTS})

# ---- backend library ----
set(BACKEND_SOURCES ${GPUSLANG_ROOT}/src/vulkan/PxgVulkanComputeDevice.cpp)
if(WIN32)
    list(APPEND BACKEND_SOURCES ${GPUSLANG_ROOT}/src/dx12/PxgDx12ComputeDevice.cpp)
endif()

add_library(gpuslang-backend STATIC ${BACKEND_SOURCES})
target_include_directories(gpuslang-backend PUBLIC ${GPUSLANG_ROOT}/include)
target_link_libraries(gpuslang-backend PUBLIC Vulkan::Vulkan)
if(WIN32)
    target_link_libraries(gpuslang-backend PUBLIC d3d12 dxgi)
endif()

# ---- test executable ----
add_executable(gpuslang-tests
    main.cpp
    testScan.cpp
    testRadixSort.cpp
    testRadixSortGather.cpp
    testPreIntegration.cpp
    testIntegration.cpp
    testUpdateBounds.cpp
    testUtilities.cpp
    testAlgorithms.cpp
    testBodyUpdate.cpp
    testSolver.cpp
    testSolverTGS.cpp
    testConstraintPrep.cpp
    testThresholdStream.cpp
    testThresholdForceChange.cpp
    testSolverZeroBodies.cpp
    testSolverInitVelocitiesTGS.cpp
    testSolverApplyGravityTGS.cpp
    testSolverDmaBack.cpp
    testNpPairManagement.cpp
    testNpPairRemoveSwap.cpp
    testNpSphereCollision.cpp
    testNpCapsuleCollision.cpp
    testNpMaterialCombine.cpp
    testNpContactPatch.cpp
    testNpInsertPatch.cpp
    testNpCompressContacts.cpp
    testNpCompressContacts2.cpp
    testNpTriangle.cpp
    testNpSegTriangle.cpp
    testNpCapTriangle.cpp
    testNpTriTri.cpp
    testNpHeightfield.cpp
    testNpHfMidphase.cpp
    testNpHfPointQuery.cpp
    testNpConvexTri.cpp
    testNpConvexSat.cpp
    testNpConvexEdgeSat.cpp
    testNpConvexFeature.cpp
    testNpConvexTriOut.cpp
    testNpConvexPolyClip.cpp
    testNpCvxTriEmit.cpp
    testNpCvxTriFind.cpp
    testNpCvxTriPurge.cpp
    testNpConvexFaceSat.cpp
    testNpManifold.cpp
    testNpSegDist.cpp
    testNpTriMask.cpp
    testNpNpUtils.cpp
    testNpConvexReduce.cpp
    testNpTriMesh.cpp
    testNpClosest.cpp
    testNpTetra.cpp
    testNpEdgeEdge.cpp
    testNpDefFilter.cpp
    testNpBv32.cpp
    testNpMidphaseCvx.cpp
    testNpMidphaseDeform.cpp
    testNpMidphaseAlloc.cpp
    testNpClothCloth.cpp
    testGridCal.cpp
    testContactReduce.cpp
    testNpSbTet.cpp
    testNpSbSelf.cpp
    testNpSbSb.cpp
    testNpTetTriMid.cpp
    testPsPrimContact.cpp
    testPsBounds.cpp
    testPsPrimCollide.cpp
    testPsDiffuseCollide.cpp
    testFcPrim.cpp
    testPsMeshMid.cpp
    testPsHfCollide.cpp
    testFcPrim2.cpp
    testNpPolyClip4.cpp
    testFcConvexTri.cpp
    testFcVertexConvex.cpp
    testFcStaticTriCloth.cpp
    testFcMidphase.cpp
    testNpSbPrim.cpp
    testNpSbRemap.cpp
    testNpCvxHfMid.cpp
    testNpTriPlaneNp.cpp
    testNpTriHfNp.cpp
    testNpCvxMeshNp.cpp
    testNpCvxHfNp.cpp
    testNpBoxClip.cpp
    testNpBoxIncident.cpp
    testNpBoxContacts.cpp
    testNpBoxSat.cpp
    testNpBoxFeature.cpp
    testNpBoxReduce.cpp
    testBroadphaseSetup.cpp
    testBroadphaseReports.cpp
    testBroadphaseIncrementalHistogram.cpp
    testBroadphaseEndPtHistogram.cpp
    testBroadphaseIncrementalSap.cpp
    testBroadphaseRegionHistogram.cpp
    testBroadphaseCreateRegions.cpp
    testBroadphaseActiveRegionHistogram.cpp
    testBroadphaseOverlapCheckHistogram.cpp
    testBroadphaseNewBoundsPairs.cpp
    testBroadphaseMarkPairs.cpp
    testBroadphaseClearNewFlag.cpp
    testBroadphaseAggregateBounds.cpp
    testBroadphaseAggregateDirty.cpp
    testBroadphaseAggregatePairsAdd.cpp
    testBroadphaseAggregateFinalize.cpp
    testBroadphaseAggregatePairsRemove.cpp
    testBroadphaseAggregateBoxPruning.cpp
    testBroadphaseAggPairCollision.cpp
    testBroadphaseAggSelfCollision.cpp
    testBroadphaseAggSortProjections.cpp)
target_link_libraries(gpuslang-tests PRIVATE gpuslang-backend)
add_dependencies(gpuslang-tests gpuslang-shaders)

enable_testing()
foreach(suite scan radixSort radixSortGather preIntegration integration updateBounds memCopy clamp skinning
        scan4x4 gpuRadixSort reorderKernel bodyUpdate directApi setRigidDynamicGlobalPose solverPGS solverTGS
        constraintPrepPGS constraintPrepTGS thresholdStream thresholdForceChange solverZeroBodies solverInitVelocitiesTGS solverApplyGravityTGS solverDmaBack npPairManagement npPairRemoveSwap npSphereCollision npCapsuleCollision npMaterialCombine npContactPatch npInsertPatch npCompressContacts npCompressContacts2 npTriangle npSegTriangle npCapTriangle npTriTri npHeightfield npHfMidphase npHfPointQuery npConvexTri npConvexSat npConvexFaceSat npConvexEdgeSat npConvexFeature npConvexTriOut npConvexPolyClip npCvxTriEmit npCvxTriFind npCvxTriPurge npManifold npSegDist npTriMask npNpUtils npConvexReduce npTriMesh npClosest npTetra npEdgeEdge npDefFilter npBv32 npMidphaseCvx npMidphaseDeform npMidphaseAlloc npClothCloth gridCal contactReduce npSbTet npSbSelf npSbSb npTetTriMid psPrimContact psBounds psPrimCollide psDiffuseCollide fcPrim psMeshMid psHfCollide fcPrim2 npPolyClip4 fcConvexTri fcVertexConvex fcStaticTriCloth fcMidphase npSbPrim npSbRemap npCvxHfMid npTriPlaneNp npTriHfNp npCvxMeshNp npCvxHfNp npBoxClip npBoxIncident npBoxContacts npBoxSat npBoxFeature npBoxReduce broadphaseSetup broadphaseReports
        broadphaseIncrementalHistogram broadphaseEndPtHistogram broadphaseIncrementalSap
        broadphaseRegionHistogram broadphaseCreateRegions broadphaseActiveRegionHistogram
        broadphaseOverlapCheckHistogram broadphaseNewBoundsPairs broadphaseMarkPairs
        broadphaseClearNewFlag broadphaseAggregateBounds broadphaseAggregateDirty
        broadphaseAggregatePairsAdd broadphaseAggregateFinalize broadphaseAggregatePairsRemove
        broadphaseAggregateBoxPruning broadphaseAggPairCollision broadphaseAggSelfCollision broadphaseAggSortProjections)
    add_test(NAME gpuslang.${suite} COMMAND gpuslang-tests ${suite})
    set_tests_properties(gpuslang.${suite} PROPERTIES
        ENVIRONMENT "PXG_SHADER_DIR=${SHADER_OUT_DIR}")
endforeach()

# Cross-compilation check for the D3D12 backend: every kernel must also
# compile to HLSL (SM 6.0).  Actual DXIL codegen and execution on WARP/real
# adapters requires a Windows host; this keeps the DX12 shader path honest on
# every platform.
set(HLSL_CHECK_DIR ${CMAKE_BINARY_DIR}/hlsl-check)
file(MAKE_DIRECTORY ${HLSL_CHECK_DIR})
set(PXG_HLSL_ENTRIES
    "scan:scanPerBlockKernel" "scan:addBlockSumsKernel"
    "radixSort:radixSortHistogram" "radixSort:radixSortCalculateRanks"
    "radixSort64:radixSortCopyHigh32Bits"
    "radixSortGather:radixSortCopy" "radixSortGather:radixSortCopy2" "radixSortGather:radixSortCopyBits2"
    "radixSortGather:radixSortDoubleCopy" "radixSortGather:radixSortDoubleCopyHigh32Bits"
    "preIntegration:preIntegrationLaunch" "preIntegration:initStaticKinematics"
    "integration:integrateCoreParallelLaunch" "integration:integrateCoreParallelLaunchTGS"
    "updateBounds:updateTransformCacheAndBoundArrayLaunch" "updateBounds:setRigidDynamicGlobalPose"
    "bodyUpdate:updateBodiesLaunch" "bodyUpdate:updateBodiesLaunchDirectAPI"
    "bodyUpdate:updateShapesLaunch" "bodyUpdate:updateBodyExternalVelocitiesLaunch"
    "directApi:getRigidDynamicGlobalPose" "directApi:getRigidDynamicLinearAcceleration"
    "directApi:setRigidDynamicLinearVelocity" "directApi:setRigidDynamicTorque"
    "solverPGS:solveBlockPartition" "solverPGS:concludeBlocks" "solverPGS:writebackBlocks"
    "solverTGS:solveBlockPartitionTGS" "solverTGS:concludeBlocksTGS" "solverTGS:writebackBlocksTGS"
    "solverTGS:writeBackBodiesTGS" "solverTGS:computeAverageSolverBodyVelocityTGS"
    "solverTGS:propagateAverageSolverBodyVelocityTGS"
    "constraintPrepPGS:contactConstraintBlockPrepareParallelLaunch"
    "constraintPrepPGS:jointConstraintBlockPrepareParallelLaunch"
    "constraintPrepTGS:contactConstraintBlockPrepareParallelLaunchTGS"
    "constraintPrepTGS:jointConstraintBlockPrepareParallelLaunchTGS"
    "thresholdStream:initialRanksAndBodyIndexB" "thresholdStream:reorganizeThresholdElements"
    "thresholdStream:computeAccumulateThresholdStream" "thresholdStream:outputAccumulateThresholdStream"
    "thresholdStream:computeExceededForceThresholdElementIndice"
    "thresholdStream:outputExceededForceThresholdElementIndice"
    "thresholdForceChange:setThresholdElementsMask"
    "thresholdForceChange:setPersistentForceElementMask"
    "thresholdForceChange:computeThresholdElementMaskIndices"
    "thresholdForceChange:outputThresholdPairsMaskIndices"
    "thresholdForceChange:createForceChangeThresholdElements"
    "solverZeroBodies:ZeroBodies" "solverZeroBodies:ZeroBodiesTGS"
    "solverInitVelocitiesTGS:initializeSolverVelocitiesTGS"
    "solverApplyGravityTGS:applyTGSSubstepGravity"
    "solverDmaBack:dmaBackChangedElems" "solverDmaBack:dmaConstraintResidual"
    "npPairManagement:initializeKeepDropBuffer" "npPairManagement:markKeepDropBuff"
    "npPairManagement:processKeepDropBuff" "npPairManagement:accumulateKeepDrop"
    "npPairManagement:initializeManifolds"
    "npPairRemoveSwap:removeContactManagers_Stage5"
    "npPairRemoveSwap:removeContactManagers_Stage5_CvxTri"
    "npSphereCollision:sphereNphase_sphereSphere" "npSphereCollision:sphereNphase_spherePlane"
    "npSphereCollision:sphereNphase_sphereCapsule" "npSphereCollision:sphereNphase_sphereBox"
    "npCapsuleCollision:planeCapsule" "npCapsuleCollision:capsuleCapsule"
    "npMaterialCombine:combineMaterials"
    "npContactPatch:setContactPointAndForcePointers" "npContactPatch:registerContactPatch"
    "npInsertPatch:insertIntoPatchStream"
    "npCompressContacts:compressContactStage1" "npCompressContacts:updateFrictionPatches"
    "npCompressContacts2:compressContactStage2"
    "npTriangle:triPointDistance" "npTriangle:triLineIntersect"
    "npSegTriangle:segTriDistance"
    "npCapTriangle:capTriGenContacts" "npCapTriangle:capTriGenEE"
    "npTriTri:triTriSat"
    "npHeightfield:hfGetTriangle" "npHeightfield:hfGetAdjacency" "npHeightfield:hfGetTriangleScaled"
    "npHfMidphase:hfmComputePairs" "npHfMidphase:hfmOutputPairs"
    "npHfPointQuery:hfGridRange" "npHfPointQuery:hfTestLocalPoint"
    "npConvexTri:ctMinkowskiFace" "npConvexTri:ctEdgePairPassed"
    "npConvexSat:ctSupportConvex" "npConvexSat:ctSupportTriangle" "npConvexSat:ctSatTriangleNormal"
    "npConvexFaceSat:ctSatConvexFaceNormals"
    "npConvexEdgeSat:ctEdgeCrossSat"
    "npConvexFeature:ctSelectFeature"
    "npConvexTriOut:ctConvexTriOut"
    "npConvexPolyClip:ctPolyClip"
    "npCvxTriEmit:ctCvxTriEmit"
    "npCvxTriFind:ctCvxTriFind"
    "npCvxTriPurge:ctCvxTriPurge"
    "npManifold:mfRefresh" "npManifold:mfInvalidate"
    "npSegDist:sdPointSegment" "npSegDist:sdSegmentSegment"
    "npTriMask:ctBarycentric" "npTriMask:ctActiveMaskVW" "npTriMask:ctActiveMaskFromNormal" "npTriMask:ctActiveMaskPoint"
    "npNpUtils:nuNext3Prev3" "npNpUtils:nuEdgeFlags" "npNpUtils:nuDelayContact"
    "npConvexReduce:ctReducePoints"
    "npTriMesh:ctTriNormal" "npTriMesh:ctTriBarycentric"
    "npClosest:ctClosestPtTriangle" "npClosest:ctClosestPtEdge"
    "npTetra:ctTriBBox" "npTetra:ctTetBBox" "npTetra:ctTetBary"
    "npEdgeEdge:ctLineLine" "npEdgeEdge:ctEdgeEdge"
    "npDefFilter:ctFindParticleRigid" "npDefFilter:ctFindRigid" "npDefFilter:ctFindNonRigid" "npDefFilter:ctFindRange" "npDefFilter:ctEdgeAuth"
    "npBv32:ctBv32Query"
    "npMidphaseCvx:ctMidphaseCvx"
    "npMidphaseDeform:ctClothTri" "npMidphaseDeform:ctSoftbodyTet" "npMidphaseDeform:ctSoftbodyBoxTet"
    "npMidphaseAlloc:ctMidphaseAlloc" "npMidphaseAlloc:ctCvxTriAccessors"
    "npClothCloth:ctClothClothQuery"
    "gridCal:ctGridCal"
    "contactReduce:ctContactReduce"
    "npSbTet:ctSbTet"
    "npSbSelf:ctSbSelfQuery"
    "npSbSb:ctSbSbQuery"
    "npTetTriMid:ctTetTriMid"
    "psPrimContact:ctPsPrimContact"
    "psBounds:ctPsBoundFirstPass" "psBounds:ctPsBoundSecondPass" "psBounds:ctPsReorderContacts" "psBounds:ctPsGridHash"
    "psPrimCollide:ctPsPrimCollide"
    "psDiffuseCollide:ctPsDiffuseCollide"
    "fcPrim:ctFcPrim"
    "psMeshMid:ctPsMeshMid"
    "psHfCollide:ctPsHfCollide"
    "fcPrim2:ctFcPrim2"
    "npPolyClip4:ctPolyClip4"
    "fcConvexTri:ctFcConvexTri"
    "fcVertexConvex:ctFcVertexConvex"
    "fcStaticTriCloth:fcStaticTriClothKernel"
    "fcMidphase:fmVertexAabb"
    "fcMidphase:fmSphereTree"
    "fcMidphase:fmHfBound"
    "fcMidphase:sbHfBound"
    "npSbPrim:sbTetPlane"
    "npSbPrim:sbEmit"
    "npSbPrim:sbParticle"
    "npSbPrim:sbClothVert"
    "npSbRemap:rmOther"
    "npSbRemap:rmFem"
    "npSbRemap:rmReorder"
    "npCvxHfMid:chMidphase"
    "npCvxHfMid:sbShrink"
    "npTriPlaneNp:tpNarrowphase"
    "npTriHfNp:thNarrowphase"
    "npCvxMeshNp:cmSphereTri"
    "npCvxMeshNp:cmCapsuleTri"
    "npCvxHfNp:chSphereTri"
    "npBoxClip:boxContains" "npBoxClip:boxIntersectSegmentAABB"
    "npBoxIncident:boxIncidentPolygon4" "npBoxIncident:boxCalculateBounds4"
    "npBoxContacts:boxCalculateContacts"
    "npBoxSat:boxSat"
    "npBoxFeature:boxFeatureSetup"
    "npBoxReduce:boxCompactContacts"
    "broadphaseSetup:translateAABBsLaunch" "broadphaseSetup:markCreatedPairsLaunch"
    "broadphaseSetup:initializeSapBox1DLaunch" "broadphaseSetup:initializeRadixRanks"
    "broadphaseSetup:updateHandles"
    "broadphaseReports:accumulateReportsStage_1" "broadphaseReports:accumulateReportsStage_2"
    "broadphaseReports:copyReports"
    "broadphaseIncrementalHistogram:computeIncrementalComparisonHistograms_Stage1"
    "broadphaseIncrementalHistogram:computeIncrementalComparisonHistograms_Stage2"
    "broadphaseEndPtHistogram:computeEndPtsHistogram"
    "broadphaseEndPtHistogram:outputEndPtsHistogram"
    "broadphaseIncrementalSap:performIncrementalSAP"
    "broadphaseRegionHistogram:computeRegionsHistogram"
    "broadphaseRegionHistogram:outputRegionsHistogram"
    "broadphaseCreateRegions:createRegionsKernel"
    "broadphaseActiveRegionHistogram:writeOutStartAndActiveRegionHistogram"
    "broadphaseActiveRegionHistogram:computeStartAndActiveRegionHistogram"
    "broadphaseActiveRegionHistogram:outputOrderedActiveRegionHistogram"
    "broadphaseOverlapCheckHistogram:writeOutOverlapChecksForInsertedBoundsRegionsHistogram"
    "broadphaseOverlapCheckHistogram:computeOverlapChecksForRegionsHistogram"
    "broadphaseOverlapCheckHistogram:outputOverlapChecksForRegionHistogram"
    "broadphaseNewBoundsPairs:generateFoundPairsForNewBoundsRegion"
    "broadphaseMarkPairs:markRemovedPairsLaunch"
    "broadphaseMarkPairs:markRemovedPairsProjectionsLaunch"
    "broadphaseMarkPairs:markUpdatedPairsLaunch"
    "broadphaseMarkPairs:markUpdatedPairsLaunch2"
    "broadphaseClearNewFlag:clearNewFlagLaunch"
    "broadphaseClearNewFlag:bpSignalComplete"
    "broadphaseAggregateBounds:markAggregateBoundsUpdatedBitmap"
    "broadphaseAggregateBounds:updateAggregateBounds"
    "broadphaseAggregateDirty:updateDirtyAggregate"
    "broadphaseAggregateDirty:clearDirtyAggregates"
    "broadphaseAggregatePairsAdd:addAggPairsStage1"
    "broadphaseAggregatePairsAdd:addAggPairsStage2"
    "broadphaseAggregateFinalize:aggCopyReports"
    "broadphaseAggregateFinalize:removeAggPairsStage3"
    "broadphaseAggregateFinalize:markAddedAndDeletedAggregatedBounds"
    "broadphaseAggregatePairsRemove:removeAggPairsStage1"
    "broadphaseAggregatePairsRemove:removeAggPairsStage2"
    "broadphaseAggregateBoxPruning:aggregateBipartiteBoxPruning"
    "broadphaseAggPairCollision:doAggPairCollisions"
    "broadphaseAggSelfCollision:doSelfCollision"
    "broadphaseAggSortProjections:sortAndUpdateAggregateProjections"
    "solverPGS:computeAverageSolverBodyVelocity" "solverPGS:propagateSolverBodyVelocity"
    "scan4x4:scanPerBlockKernel4x4" "scan4x4:addBlockSumsKernel4x4"
    "radixFourBit:radixFourBitCountPerBlockKernel" "radixFourBit:radixFourBitReorderKernel"
    "reorderData:reorderKernel"
    "memCopy:MemCopyBalanced"
    "clampValues:clampMaxValue" "clampValues:clampMaxValues"
    "skinning:interleaveBuffers" "skinning:zeroNormals" "skinning:normalVectorsAreaWeighted"
    "skinning:normalizeNormals" "skinning:interpolateSkinnedClothVertices"
    "skinning:interpolateSkinnedSoftBodyVertices")
foreach(spec ${PXG_HLSL_ENTRIES})
    string(REPLACE ":" ";" parts ${spec})
    list(GET parts 0 stem)
    list(GET parts 1 entry)
    add_test(NAME gpuslang.hlsl.${entry}
        COMMAND ${PXG_SLANGC} ${SHADER_DIR}/${stem}.slang -target hlsl -profile cs_6_0
                -entry ${entry} -o ${HLSL_CHECK_DIR}/${stem}-${entry}.hlsl)
endforeach()
