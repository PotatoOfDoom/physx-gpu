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
    testBroadphaseAggregateDirty.cpp)
target_link_libraries(gpuslang-tests PRIVATE gpuslang-backend)
add_dependencies(gpuslang-tests gpuslang-shaders)

enable_testing()
foreach(suite scan radixSort preIntegration integration updateBounds memCopy clamp skinning
        scan4x4 gpuRadixSort reorderKernel bodyUpdate directApi setRigidDynamicGlobalPose solverPGS solverTGS
        constraintPrepPGS constraintPrepTGS thresholdStream broadphaseSetup broadphaseReports
        broadphaseIncrementalHistogram broadphaseEndPtHistogram broadphaseIncrementalSap
        broadphaseRegionHistogram broadphaseCreateRegions broadphaseActiveRegionHistogram
        broadphaseOverlapCheckHistogram broadphaseNewBoundsPairs broadphaseMarkPairs
        broadphaseClearNewFlag broadphaseAggregateBounds broadphaseAggregateDirty)
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
