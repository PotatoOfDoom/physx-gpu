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

# The slangc discovery, the compile flags, the NoContraction/SignedZero stamping
# lists and pxg_compile_shader live in ../cmake/PxgShaders.cmake; every
# pxg_compile_shader call lives in ../cmake/PxgShaderList.cmake.  The official
# Vulkan PhysXGpu build (source/compiler/cmakegpuvulkan) includes the SAME two
# files, so the SPIR-V it ships is compiled by exactly the commands this suite
# validates.
set(GPUSLANG_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(SHADER_OUT_DIR ${CMAKE_BINARY_DIR}/shaders)
include(${GPUSLANG_ROOT}/cmake/PxgShaders.cmake)
include(${GPUSLANG_ROOT}/cmake/PxgShaderList.cmake)

add_custom_target(gpuslang-shaders ALL DEPENDS ${PXG_SHADER_OUTPUTS})

# ---- PhysXFoundation (RUNTIME_PLAN.md P4: the engine base) ----
# The keystone for linking real PhysX engine code (gpucommon heap allocator /
# kernel wrangler, and eventually the gpu subsystems) against the Vulkan bridge.
# Built from the SDK's authoritative foundation source list — plain C++, no
# packman, no CUDA.  Static + PIC so it links into both the test exe and the
# PhysXGpuVulkan drop-in.  PX_PHYSX_STATIC_LIB keeps it PRIVATE so it does not
# alter the export visibility of the PhysXGpuVulkan C-ABI symbols.
set(PXFND_ROOT ${GPUSLANG_ROOT}/../foundation)
add_library(physx-foundation STATIC
    ${PXFND_ROOT}/FdAllocator.cpp
    ${PXFND_ROOT}/FdString.cpp
    ${PXFND_ROOT}/FdTempAllocator.cpp
    ${PXFND_ROOT}/FdAssert.cpp
    ${PXFND_ROOT}/FdMathUtils.cpp
    ${PXFND_ROOT}/FdFoundation.cpp
    ${PXFND_ROOT}/unix/FdUnixAtomic.cpp
    ${PXFND_ROOT}/unix/FdUnixMutex.cpp
    ${PXFND_ROOT}/unix/FdUnixSync.cpp
    ${PXFND_ROOT}/unix/FdUnixThread.cpp
    ${PXFND_ROOT}/unix/FdUnixPrintString.cpp
    ${PXFND_ROOT}/unix/FdUnixSList.cpp
    ${PXFND_ROOT}/unix/FdUnixSocket.cpp
    ${PXFND_ROOT}/unix/FdUnixTime.cpp
    ${PXFND_ROOT}/unix/FdUnixFPU.cpp)
set_target_properties(physx-foundation PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_STANDARD 17)
target_compile_definitions(physx-foundation PRIVATE PX_PHYSX_STATIC_LIB)
target_include_directories(physx-foundation PUBLIC ${GPUSLANG_ROOT}/../../include
    PRIVATE ${PXFND_ROOT} ${PXFND_ROOT}/include)
find_package(Threads REQUIRED)
target_link_libraries(physx-foundation PUBLIC Threads::Threads ${CMAKE_DL_LIBS})

# ---- backend library ----
set(BACKEND_SOURCES ${GPUSLANG_ROOT}/src/vulkan/PxgVulkanComputeDevice.cpp
    ${GPUSLANG_ROOT}/src/PxgCudaShim.cpp)
if(WIN32)
    list(APPEND BACKEND_SOURCES ${GPUSLANG_ROOT}/src/dx12/PxgDx12ComputeDevice.cpp)
endif()

add_library(gpuslang-backend STATIC ${BACKEND_SOURCES})
# Position-independent so the static backend can be linked into the shared
# PhysXGpuVulkan drop-in as well as the test executable.
set_target_properties(gpuslang-backend PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(gpuslang-backend PUBLIC ${GPUSLANG_ROOT}/include)
target_link_libraries(gpuslang-backend PUBLIC Vulkan::Vulkan)
if(WIN32)
    target_link_libraries(gpuslang-backend PUBLIC d3d12 dxgi)
endif()

# ---- test executable ----
set(PXG_TEST_SOURCES
    main.cpp
    testPipelineRefusal.cpp
    testScan.cpp
    testRadixSort.cpp
    testRadixSortGather.cpp
    testRadixSortDesc.cpp
    testPreIntegration.cpp
    testIntegration.cpp
    testUpdateBounds.cpp
    testFrozenHistogram.cpp
    testUtilities.cpp
    testAlgorithms.cpp
    testBodyUpdate.cpp
    testSolver.cpp
    testSolverTGS.cpp
    testConstraintPrep.cpp
    testThresholdStream.cpp
    testThresholdStreamBda.cpp
    testAggregateBda.cpp
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
    testNpAbiLayout.cpp
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
    testNpCvxTriWorldNormal.cpp
    testNpGjkSupport.cpp
    testNpGjkNewPoint.cpp
    testNpGjkBarycentric.cpp
    testNpHfTriFlags.cpp
    testNpWarpVec.cpp
    testNpGjkSimplex.cpp
    testNpGjkClosestTri.cpp
    testNpGjkClosestTet.cpp
    testNpGjkClosestSeg.cpp
    testNpGjkOutput.cpp
    testNpGjkPermute.cpp
    testNpGjkSupportStep.cpp
    testNpGjkWarmStart.cpp
    testNpGjk.cpp
    testNpEpaStart4.cpp
    testNpEpaStart123.cpp
    testNpEpa.cpp
    testNpGjkWarm.cpp
    testNpCvxCvxEE.cpp
    testNpCvxRayPoly.cpp
    testNpCvxFaceContacts.cpp
    testNpCvxReduceManifold.cpp
    testNpCvxWitnessPoly.cpp
    testNpCvxLocalNormal.cpp
    testNpCvxAddManifold.cpp
    testNpCvxBlob.cpp
    testNpCvxBlobRead.cpp
    testNpCvxEEDriver.cpp
    testNpCvxRefFrame.cpp
    testNpCvxPolyClip.cpp
    testNpCvxClipEmit.cpp
    testNpCvxContactDecide.cpp
    testNpCvxCollidePrep.cpp
    testNpCvxCollideDecide.cpp
    testNpCvxLostFound.cpp
    testNpCvxWriteStream.cpp
    testNpCvxFinishContacts.cpp
    testNpCvxStage1Refresh.cpp
    testNpCvxStage2Setup.cpp
    testCvxCollide.cpp
    testNpCvxPlanePolyIndex.cpp
    testNpCvxPlaneRefresh.cpp
    testNpCvxPlaneGen.cpp
    testNpCvxPlaneOut.cpp
    testConvexPlaneNphase.cpp
    testNpCvxCorrelate.cpp
    testNpCvxPostProcess.cpp
    testNpCvxTrimeshOut.cpp
    testNpTriTriOverlap.cpp
    testNpCrsWrapper2.cpp
    testNpTriTriReduce.cpp
    testNpTriTriWrite.cpp
    testNpTriTriBlock.cpp
    testNpTriTriGeom.cpp
    testNpTriTriBody.cpp
    testNpTriTriFind.cpp
    testNpTriTriTail.cpp
    testNpTriTriKernel.cpp
    testNpCvxTriNpSetup.cpp
    testNpCvxHfNpSetup.cpp
    testNpCvxCoreOut.cpp
    testCvxCoreTailBda.cpp
    testP86SignedZero.cpp
    testNpCvxCoreSetup.cpp
    testNpCvxCoreSupport.cpp
    testNpCvxCoreFace.cpp
    testNpCvxCorePtsFace.cpp
    testNpCvxCoreClip.cpp
    testNpCvxCoreContact.cpp
    testNpCvxCoreGjk.cpp
    testNpCvxCoreEpa.cpp
    testNpCvxCorePlaneGen.cpp
    testNpCvxCoreCvxGen.cpp
    testNpCvxBvhGlue.cpp
    testNpCvxCoreWarpAcc.cpp
    testNpD6JointPrep.cpp
    testNpRemapIndex.cpp
    testNpContactPrePrep.cpp
    testNpRigidSumBatches.cpp
    testNpArtiExtBody.cpp
    testNpArti1DPrep.cpp
    testNpArtiJointHeader.cpp
    testNpArtiSolve1D.cpp
    testNpArtiSolveContact.cpp
    testNpArtiSolve1DTGS.cpp
    testNpArtiSolveContactTGS.cpp
    testNpArtiCheckActive.cpp
    testNpSlabRefCount.cpp
    testNpArtiPropagate.cpp
    testNpArtiPropagateAccel.cpp
    testNpArtiPropagateAccelJV.cpp
    testNpArtiSpatialMatVec.cpp
    testNpArtiBitfield.cpp
    testNpArtiSpatialMatMat.cpp
    testNpArtiConstructSM.cpp
    testNpArtiTranslateInertia.cpp
    testNpArtiTranslateSInertia.cpp
    testNpArtiPropagateZA.cpp
    testNpArtiPropagateZASph.cpp
    testNpArtiTranslateInertiaMass.cpp
    testNpArtiInvertInertia.cpp
    testNpArtiPropagateImpDof.cpp
    testNpArtiAccumLinkImp.cpp
    testNpArtiDriveParams.cpp
    testNpArtiClampDriveImp.cpp
    testNpClothDefGrad2D.cpp
    testNpSbDefGrad.cpp
    testNpDefMaterial.cpp
    testNpGridIndex.cpp
    testNpJacobiRotate.cpp
    testNpClothAtan.cpp
    testNpAnisoHelpers.cpp
    testNpSparseGridHash.cpp
    testNpClothDeltaLambda.cpp
    testNpClothExtractRot2D.cpp
    testNpClothVelClamp.cpp
    testNpClothArap2D.cpp
    testNpClothArea2D.cpp
    testNpClothWind.cpp
    testNpSbBaryInvMass.cpp
    testNpSbVolumeSolve.cpp
    testNpIsoConcaveCorner.cpp
    testNpSphSpikyKernel.cpp
    testNpIsoAcosDecode.cpp
    testNpSparseGridDomain.cpp
    testNpDenseGridCell.cpp
    testNpSbBaryProject.cpp
    testNpIsoVecNormalize.cpp
    testNpSbCompliance.cpp
    testNpSbDeltaLambdaGradC.cpp
    testNpSbTetConstraint.cpp
    testNpPsSpecVolume.cpp
    testNpSbComputeLambda.cpp
    testNpGridLocation.cpp
    testNpAnisoOutput.cpp
    testNpSmoothPos.cpp
    testNpIsoAvgVerts.cpp
    testNpPsWeightKernels.cpp
    testNpPsDiffuseUpdate.cpp
    testNpPsDiffuseSpawn.cpp
    testNpPsDiffuseAvg.cpp
    testNpPsDiffuseOneWay.cpp
    testNpSdfBoxDist.cpp
    testNpSparseGridLookup.cpp
    testNpSparseGridCellFromPos.cpp
    testNpIsoGridFilter.cpp
    testNpSbExtractRotAPD.cpp
    testNpSbSolveNeoHookean.cpp
    testNpSbSolveCorot.cpp
    testNpSbSolveGM.cpp
    testNpClothMembrane.cpp
    testNpClothBending.cpp
    testNpClothSharedEnergy.cpp
    testNpSbCollisionConstraint.cpp
    testNpClothCollisionConstraint.cpp
    testNpSdfBvhMorton.cpp
    testNpSbStress.cpp
    testNpSdfHoles.cpp
    testNpClothTriContact.cpp
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
    testDeformableRef512.cpp
    testNpBv32.cpp
    testNpMidphaseCvx.cpp
    testNpMidphaseDeform.cpp
    testNpMidphaseAlloc.cpp
    testNpClothCloth.cpp
    testGridCal.cpp
    testParticleSim.cpp
    testFemClothSim.cpp
    testSoftBodySim.cpp
    testSoftBodyPsContactGen.cpp
    testSoftBodyPsPrepare.cpp
    testSoftBodySolve512.cpp
    testFemClothSolve512.cpp
    testParticleContactSort.cpp
    testFemContactSort.cpp
    testPcmManifold.cpp
    testConvexTrimeshMidphase.cpp
    testConvexHfMidphase.cpp
    testMidphaseAlloc.cpp
    testTriFeature.cpp
    testDeferContact.cpp
    testInitManifolds.cpp
    testPairManagement.cpp
    testPolyClip.cpp
    testConvexHull.cpp
    testConvexSat.cpp
    testParticleGrid.cpp
    testParticleSelfCollision.cpp
    testPsMaterials.cpp
    testPsFluidDensity.cpp
    testPsSolveDensity.cpp
    testPsFluidVelocity.cpp
    testPsBoundaryDensity.cpp
    testPsMeshMidphase.cpp
    testPsHfMidphase.cpp
    testContactReduce.cpp
    testCvxTriEmit.cpp
    testCvxTriSetup.cpp
    testSphereTriNp.cpp
    testCvxHfSetup.cpp
    testCvxHfPostProcess.cpp
    testSphereHfCore.cpp
    testHfTriFlagsBda.cpp
    testSegTriangleBda.cpp
    testCapTriangleBda.cpp
    testCapsuleHfCore.cpp
    testRadixWarp.cpp
    testSortTriIndices.cpp
    testCvxTriPostProcess.cpp
    testNewArticulations.cpp
    testUpdateArticulations.cpp
    testArtiBlockLayout.cpp
    testArtiCoreDescLayout.cpp
    testArtiJointFrame.cpp
    testArtiJcalcPhaseA.cpp
    testArtiJcalcPhaseB1.cpp
    testArtiJcalcPhaseB2.cpp
    testArtiJcalcPhaseB3.cpp
    testArtiJcalcPhaseC.cpp
    testArtiJcalcTendons.cpp
    testArtiJcalcGpuDirty.cpp
    testArtiJcalcInit.cpp
    testArtiJcalcLinkVel.cpp
    testArtiUnconstrainedVel.cpp
    testArtiSpatialInertia.cpp
    testArtiSpatialMatrix.cpp
    testArtiPropagateInertia.cpp
    testArtiDynamic.cpp
    testArtiMassMatrix.cpp
    testArtiLinkAccel.cpp
    testArtiDmaBack.cpp
    testArtiStep.cpp
    testArtiFlushVel.cpp
    testArtiAvgImpulse.cpp
    testArtiStepKernel.cpp
    testArtiSleep.cpp
    testArtiUpdateBodies.cpp
    testArtiUpdateBodies3.cpp
    testArtiUpdateBodies1T.cpp
    testArtiSumBatches.cpp
    testArtiSumBatches2.cpp
    testArtiSumSelfBatches.cpp
    testArtiSetupInternal.cpp
    testArtiSetupTendons.cpp
    testArtiSetupMimic.cpp
    testArtiSetupICLaunch.cpp
    testArtiSolveVelImp.cpp
    testArtiSolveBodies.cpp
    testArtiSolveTMLaunch.cpp
    testArtiSolveExtTGS.cpp
    testArtiSolveSelf.cpp
    testArtiSolveSelfLaunch.cpp
    testArtiSolveSelfPgs.cpp
    testArtiSolveSelfLaunchPgs.cpp
    testArtiSolveStatic.cpp
    testArtiSolveStaticPgs.cpp
    testArtiSolveDofPgs.cpp
    testArtiSolveDFSPgs.cpp
    testArtiSolveInternalLaunchPgs.cpp
    testArtiSolveDof.cpp
    testArtiSolveDFS.cpp
    testArtiSolveInternalLaunch.cpp
    testArtiAvgImpulse2.cpp
    testArtiPropagate2.cpp
    testArtiConstraintPrepBody.cpp
    testArtiConstraintPrepSetup.cpp
    testArtiConstraintPrepWrap.cpp
    testArtiJointPrepWrap.cpp
    testArtiSolvePartitionWrap.cpp
    testArtiOutputVelocity.cpp
    testArtiPushImpulse.cpp
    testArtiApplyTgsSubstepForces.cpp
    testArtiSaveVelocity.cpp
    testArtiComputeDeps.cpp
    testArtiComputeCOM.cpp
    testArtiInverseDyn2.cpp
    testArtiGravityForces.cpp
    testArtiCentrifugal.cpp
    testArtiMassMatrices.cpp
    testArtiUpdateKinematic.cpp
    testArtiGetSet.cpp
    testArtiJointPrepPgs.cpp
    testArtiConstraintPrepPgs.cpp
    testArtiSolveExtPgs.cpp
    testArtiSolvePartitionPgs.cpp
    testNpSbTet.cpp
    testNpSbSelf.cpp
    testNpSbSb.cpp
    testNpTetTriMid.cpp
    testPsPrimContact.cpp
    testPsBounds.cpp
    testPsContactGen.cpp
    testPsContactSort.cpp
    testPsContactSolve.cpp
    testPsContactSolveTgs.cpp
    testPsContactSolveRigid.cpp
    testDirectApiBda.cpp
    testDirectApiBdaWrite.cpp
    testRigidDeltaAccum.cpp
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
    testBoxBoxNphase.cpp
    testSphereNphase.cpp
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
    testBroadphaseAggSortProjections.cpp
    testP0Subgroup.cpp
    testP0Bda.cpp
    testRobustProbe.cpp
    testP1DeviceV2.cpp
    testP4ModulePath.cpp
    ${GPUSLANG_ROOT}/bridge/PxgModulePath.cpp
    testP1HeapAlloc.cpp
    testNpWarpLib.cpp
    testNpAtomicLib.cpp
    testNpFloatIntrin.cpp
    testNpTex3d.cpp
    testNpTex3dBindless.cpp
    testP3ShimTexture.cpp
    testShimBindless.cpp
    testShimOrdering.cpp
    testSdfTexture.cpp
    testSdfCollision.cpp
    testSdfQuery.cpp
    testSdfFind.cpp
    testSdfParticle.cpp
    testPsSdfMidphase.cpp
    testFemClothSdfCg.cpp
    testSoftBodySdfMid.cpp
    testPsDiffuseBuffers.cpp
    testPsDiffuseCreate.cpp
    testPsDiffuseAdvect.cpp
    testPsDiffuseOneWay.cpp
    testPsDiffuseContactGen.cpp
    testPsConvexContactGen.cpp
    testWarpNative.cpp
    testP1PipelineCache.cpp
    testP1Timestamps.cpp
    testP6Indirect.cpp
    testP3Shim.cpp
    testP3CudaContext.cpp
    testP3UnmappedLaunch.cpp
    testP4Manifest.cpp
    testP3CudaContextManager.cpp
    testP3KernelWrangler.cpp
    testP3ContextHardening.cpp
    testP4Loader.cpp
    testP4Foundation.cpp
    testP4Heap.cpp
    testP4Wrangler.cpp
    testP4Broadphase.cpp
    testBpDescLayout.cpp
    testSolverDescLayout.cpp
    testSimCoreDescLayout.cpp
    testPrepDescLayout.cpp
    testP4Aabb.cpp
    ${GPUSLANG_ROOT}/bridge/PxgVulkanCudaContext.cpp
    ${GPUSLANG_ROOT}/bridge/PxgVulkanKernelAbi.cpp
    # P4: the REAL gpucommon buddy allocator, reused over gpucommon's own memory
    # manager on the Vulkan context — compiles cuda-free against real headers.
    ${GPUSLANG_ROOT}/../gpucommon/src/PxgHeapMemoryAllocator.cpp
    # P4 wrangler seam: the REAL engine KernelWrangler + its manager, resolving
    # kernel ids to our encoding via the live manifest-driven function table.
    ${GPUSLANG_ROOT}/../gpucommon/src/PxgKernelWrangler.cpp
    ${GPUSLANG_ROOT}/../cudamanager/src/CudaKernelWrangler.cpp
    ${GPUSLANG_ROOT}/bridge/PxgVulkanKernelTable.cpp
    # P4.4 broadphase subsystem seam: the REAL concrete PxgCudaBroadPhaseSap + its
    # gpucommon buffer/pinned-allocator support, plus the no-toolkit driver shim
    # (cuCtxGetStreamPriorityRange) and the empty kernel-registration anchors.
    ${GPUSLANG_ROOT}/../gpucommon/src/PxgCudaBuffer.cpp
    ${GPUSLANG_ROOT}/../gpucommon/src/PxgCudaMemoryAllocator.cpp
    ${GPUSLANG_ROOT}/../gpubroadphase/src/PxgBroadPhase.cpp
    ${GPUSLANG_ROOT}/../gpubroadphase/src/PxgCudaBroadPhaseSap.cpp
    ${GPUSLANG_ROOT}/bridge/PxgVulkanDriverShim.cpp
    ${GPUSLANG_ROOT}/bridge/PxgVulkanKernelInit.cpp
    # P4.5 AABB-manager subsystem seam: the real PxgAABBManager (wrapping the
    # broadphase) + its lowlevelaabb CPU base (AABBManagerBase + BpFiltering).
    ${GPUSLANG_ROOT}/../gpubroadphase/src/PxgAABBManager.cpp
    ${GPUSLANG_ROOT}/../lowlevelaabb/src/BpAABBManagerBase.cpp
    ${GPUSLANG_ROOT}/../lowlevelaabb/src/BpFiltering.cpp
    # PxgBoundsArray::updateBounds (via the bounds-array vtable) needs the CPU
    # geometry bounds path: Gu::computeBounds + its convex-shape helpers.
    ${GPUSLANG_ROOT}/../geomutils/src/GuBounds.cpp
    ${GPUSLANG_ROOT}/../geomutils/src/convex/GuConvexUtilsInternal.cpp
    ${GPUSLANG_ROOT}/../geomutils/src/GuConvexGeometry.cpp
    # 5.12: gpucommon's PxgMemoryManager (createPxgMemoryManager), which the heap
    # allocator manager now takes by reference; it allocates through the
    # PxCudaContext and therefore through the Vulkan context.
    ${GPUSLANG_ROOT}/../gpucommon/src/PxgMemoryManager.cpp)
add_executable(gpuslang-tests ${PXG_TEST_SOURCES})
target_link_libraries(gpuslang-tests PRIVATE gpuslang-backend physx-foundation ${CMAKE_DL_LIBS})
# PXG_NO_CUDA_TOOLKIT lets the real CudaKernelWrangler.cpp compile without <cuda.h>
# and without the CUDART registration shims (only the ctor is needed here).
target_compile_definitions(gpuslang-tests PRIVATE PXG_NO_CUDA_TOOLKIT)
# The P3.1 CUDA-context bridge builds against the real PhysX headers
# (foundation + cudamanager) with PX_SUPPORT_GPU_PHYSX; no CUDA toolkit needed.
# physxgpu/include adds the PxPhysXGpu interface the P3.5 loader test dlsym's;
# the foundation/common/lowlevel include dirs let the real gpucommon engine
# source (heap allocator) compile in the harness (P4).
target_include_directories(gpuslang-tests PRIVATE ${GPUSLANG_ROOT}/../../include ${GPUSLANG_ROOT}/bridge
    ${GPUSLANG_ROOT}/../gpucommon/include ${GPUSLANG_ROOT}/../physxgpu/include
    ${GPUSLANG_ROOT}/../foundation/include ${GPUSLANG_ROOT}/../common/include
    ${GPUSLANG_ROOT}/../lowlevel/software/include ${GPUSLANG_ROOT}/../lowlevelaabb/include
    ${GPUSLANG_ROOT}/../cudamanager/include
    # P4.4: no-toolkit CUDA header stand-ins (vector_types/vector_functions/host_defines)
    # + the gpu-module + lowlevel/geomutils include dirs the real broadphase pulls.
    ${GPUSLANG_ROOT}/nocuda
    ${GPUSLANG_ROOT}/../gpubroadphase/include ${GPUSLANG_ROOT}/../gpunarrowphase/include
    ${GPUSLANG_ROOT}/../gpusimulationcontroller/include ${GPUSLANG_ROOT}/../gpusolver/include
    ${GPUSLANG_ROOT}/../gpuarticulation/include
    ${GPUSLANG_ROOT}/../common/src
    ${GPUSLANG_ROOT}/../lowlevel/api/include
    ${GPUSLANG_ROOT}/../lowlevel/common/include/pipeline ${GPUSLANG_ROOT}/../lowlevel/common/include/utils
    ${GPUSLANG_ROOT}/../lowleveldynamics/include ${GPUSLANG_ROOT}/../lowleveldynamics/shared ${GPUSLANG_ROOT}/../lowleveldynamics/src
    ${GPUSLANG_ROOT}/../geomutils/include ${GPUSLANG_ROOT}/../geomutils/src
    ${GPUSLANG_ROOT}/../geomutils/src/contact ${GPUSLANG_ROOT}/../geomutils/src/pcm
    ${GPUSLANG_ROOT}/../geomutils/src/convex ${GPUSLANG_ROOT}/../geomutils/src/common
    ${GPUSLANG_ROOT}/../geomutils/src/mesh ${GPUSLANG_ROOT}/../geomutils/src/distance
    ${GPUSLANG_ROOT}/../geomutils/src/intersection ${GPUSLANG_ROOT}/../geomutils/src/gjk
    ${GPUSLANG_ROOT}/../geomutils/src/sweep ${GPUSLANG_ROOT}/../geomutils/src/ccd ${GPUSLANG_ROOT}/../geomutils/src/hf
    ${GPUSLANG_ROOT}/../lowlevelaabb/src)
# The P3.5 loader test dlopen's the built PhysXGpuVulkan drop-in by its real path.
add_dependencies(gpuslang-tests PhysXGpuVulkan)
target_compile_definitions(gpuslang-tests PRIVATE PXG_PHYSXGPUVULKAN_SO="$<TARGET_FILE:PhysXGpuVulkan>")

# pxg-caps: standalone device-profile probe (RUNTIME_PLAN.md P0)
add_executable(pxg-caps ${GPUSLANG_ROOT}/tools/pxgCaps.cpp)
target_link_libraries(pxg-caps PRIVATE Vulkan::Vulkan)
set_target_properties(pxg-caps PROPERTIES CXX_STANDARD 17)

# PhysXGpuVulkan: the C-ABI GPU-module export library (RUNTIME_PLAN.md P3.5).
# A SHARED drop-in exporting the unmangled C symbols the PhysX module loader
# dlsym's (PxCreatePhysXGpu / PxCreateCudaContextManager /
# PxGetSuggestedCudaDeviceOrdinal + the profiler/foundation setters + CUDART
# table shims).  Built against the REAL PhysX headers — the same no-CUDA recipe
# as the bridge, plus physxgpu/include for the PxPhysXGpu interface.  OUTPUT_NAME
# is deliberately NOT libPhysXGpu_64 so it never clobbers a real CUDA module; the
# engine selects it via PxSetPhysXGpuLoadHook.  Serves as the compile-check that
# the whole export surface builds against the real ABI with no CUDA toolkit.
add_library(PhysXGpuVulkan SHARED
    ${GPUSLANG_ROOT}/bridge/PhysXGpuVulkan.cpp
    ${GPUSLANG_ROOT}/bridge/PxgModulePath.cpp
    ${GPUSLANG_ROOT}/bridge/PxgVulkanCudaContext.cpp
    ${GPUSLANG_ROOT}/bridge/PxgVulkanKernelAbi.cpp
    # P4: the real gpucommon buddy allocator, backing createGpuHeapMemoryAllocatorManager,
    # and its memory manager, backing createGpuMemoryManager (5.12).
    ${GPUSLANG_ROOT}/../gpucommon/src/PxgHeapMemoryAllocator.cpp
    ${GPUSLANG_ROOT}/../gpucommon/src/PxgMemoryManager.cpp
    # P4 wrangler seam: the real engine KernelWrangler + manager + the live table.
    ${GPUSLANG_ROOT}/../gpucommon/src/PxgKernelWrangler.cpp
    ${GPUSLANG_ROOT}/../cudamanager/src/CudaKernelWrangler.cpp
    ${GPUSLANG_ROOT}/bridge/PxgVulkanKernelTable.cpp
    # P4.4 broadphase subsystem seam: the real concrete broadphase (backing
    # createGpuBroadPhase) + gpucommon buffer/allocator support + driver/init shims.
    ${GPUSLANG_ROOT}/../gpucommon/src/PxgCudaBuffer.cpp
    ${GPUSLANG_ROOT}/../gpucommon/src/PxgCudaMemoryAllocator.cpp
    ${GPUSLANG_ROOT}/../gpubroadphase/src/PxgBroadPhase.cpp
    ${GPUSLANG_ROOT}/../gpubroadphase/src/PxgCudaBroadPhaseSap.cpp
    ${GPUSLANG_ROOT}/bridge/PxgVulkanDriverShim.cpp
    ${GPUSLANG_ROOT}/bridge/PxgVulkanKernelInit.cpp
    # P4.5 AABB-manager subsystem seam (backing createGpuAABBManager).
    ${GPUSLANG_ROOT}/../gpubroadphase/src/PxgAABBManager.cpp
    ${GPUSLANG_ROOT}/../lowlevelaabb/src/BpAABBManagerBase.cpp
    ${GPUSLANG_ROOT}/../lowlevelaabb/src/BpFiltering.cpp
    # CPU geometry bounds path for PxgBoundsArray::updateBounds.
    ${GPUSLANG_ROOT}/../geomutils/src/GuBounds.cpp
    ${GPUSLANG_ROOT}/../geomutils/src/convex/GuConvexUtilsInternal.cpp
    ${GPUSLANG_ROOT}/../geomutils/src/GuConvexGeometry.cpp)
target_link_libraries(PhysXGpuVulkan PRIVATE gpuslang-backend physx-foundation)
target_include_directories(PhysXGpuVulkan PRIVATE
    ${GPUSLANG_ROOT}/../../include ${GPUSLANG_ROOT}/include ${GPUSLANG_ROOT}/bridge
    ${GPUSLANG_ROOT}/../gpucommon/include ${GPUSLANG_ROOT}/../physxgpu/include
    ${GPUSLANG_ROOT}/../foundation/include ${GPUSLANG_ROOT}/../common/include
    ${GPUSLANG_ROOT}/../lowlevel/software/include ${GPUSLANG_ROOT}/../lowlevelaabb/include
    ${GPUSLANG_ROOT}/../cudamanager/include
    # P4.4: no-toolkit CUDA header stand-ins + gpu-module/lowlevel/geomutils dirs.
    ${GPUSLANG_ROOT}/nocuda
    ${GPUSLANG_ROOT}/../gpubroadphase/include ${GPUSLANG_ROOT}/../gpunarrowphase/include
    ${GPUSLANG_ROOT}/../gpusimulationcontroller/include ${GPUSLANG_ROOT}/../gpusolver/include
    ${GPUSLANG_ROOT}/../gpuarticulation/include
    ${GPUSLANG_ROOT}/../common/src
    ${GPUSLANG_ROOT}/../lowlevel/api/include
    ${GPUSLANG_ROOT}/../lowlevel/common/include/pipeline ${GPUSLANG_ROOT}/../lowlevel/common/include/utils
    ${GPUSLANG_ROOT}/../lowleveldynamics/include ${GPUSLANG_ROOT}/../lowleveldynamics/shared ${GPUSLANG_ROOT}/../lowleveldynamics/src
    ${GPUSLANG_ROOT}/../geomutils/include ${GPUSLANG_ROOT}/../geomutils/src
    ${GPUSLANG_ROOT}/../geomutils/src/contact ${GPUSLANG_ROOT}/../geomutils/src/pcm
    ${GPUSLANG_ROOT}/../geomutils/src/convex ${GPUSLANG_ROOT}/../geomutils/src/common
    ${GPUSLANG_ROOT}/../geomutils/src/mesh ${GPUSLANG_ROOT}/../geomutils/src/distance
    ${GPUSLANG_ROOT}/../geomutils/src/intersection ${GPUSLANG_ROOT}/../geomutils/src/gjk
    ${GPUSLANG_ROOT}/../geomutils/src/sweep ${GPUSLANG_ROOT}/../geomutils/src/ccd ${GPUSLANG_ROOT}/../geomutils/src/hf
    ${GPUSLANG_ROOT}/../lowlevelaabb/src)
target_compile_definitions(PhysXGpuVulkan PRIVATE PX_PHYSX_GPU_EXPORTS PXG_NO_CUDA_TOOLKIT)
set_target_properties(PhysXGpuVulkan PROPERTIES OUTPUT_NAME PhysXGpuVulkan_64 CXX_STANDARD 17)
# The CPU references are the bitwise ground truth: forbid the compiler from
# contracting their separately-rounded mul+add chains into FMA (GCC defaults
# to -ffp-contract=fast; harmless on non-FMA baselines but load-bearing on
# any -march with FMA).  Mirrors the NoContraction stamping on the shaders.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(gpuslang-tests PRIVATE -ffp-contract=off)
endif()
add_dependencies(gpuslang-tests gpuslang-shaders)

enable_testing()
foreach(suite scan radixSort radixSortDesc aggregateBda radixSortGather preIntegration integration updateBounds frozenHistogram memCopy clamp skinning
        scan4x4 gpuRadixSort reorderKernel bodyUpdate directApi setRigidDynamicGlobalPose solverPGS solverTGS npCrsWrapper2 npTriTriReduce npTriTriWrite npTriTriBlock npTriTriGeom npTriTriBody npTriTriFind npTriTriTail npTriTriKernel
        constraintPrepPGS constraintPrepTGS thresholdStream thresholdForceChange solverZeroBodies solverInitVelocitiesTGS solverApplyGravityTGS solverDmaBack npPairManagement npPairRemoveSwap npSphereCollision npCapsuleCollision npMaterialCombine npContactPatch npInsertPatch npCompressContacts npCompressContacts2 npTriangle npSegTriangle npCapTriangle npTriTri npHeightfield npHfMidphase npHfPointQuery npConvexTri npConvexSat npConvexFaceSat npConvexEdgeSat npConvexFeature npConvexTriOut npConvexPolyClip npCvxTriEmit npCvxTriFind npCvxTriPurge npCvxTriWorldNormal npGjkSupport npGjkNewPoint npGjkBarycentric npHfTriFlags npWarpVec npGjkSimplex npGjkClosestTri npGjkClosestTet npGjkClosestSeg npGjkOutput npGjkPermute npGjkSupportStep npGjkWarmStart npGjk npEpaStart4 npEpaStart123 npEpa npGjkWarm npCvxCvxEE npCvxRayPoly npCvxFaceContacts npCvxReduceManifold npCvxWitnessPoly npCvxLocalNormal npCvxAddManifold npCvxBlob npCvxBlobRead npCvxEEDriver npCvxRefFrame npCvxPolyClip npCvxClipEmit npCvxContactDecide npCvxCollidePrep npCvxCollideDecide npCvxLostFound npCvxWriteStream npCvxFinishContacts npCvxStage1Refresh npCvxStage2Setup cvxCollide npCvxPlanePolyIndex npCvxPlaneRefresh npCvxPlaneGen npCvxPlaneOut npCvxCorrelate npCvxPostProcess npCvxTrimeshOut npTriTriOverlap npCvxTriNpSetup npCvxHfNpSetup npCvxCoreOut npCvxCoreSetup npCvxCoreSupport npCvxCoreFace npCvxCorePtsFace npCvxCoreClip npCvxCoreContact npCvxCoreGjk npCvxCoreEpa npCvxCorePlaneGen npCvxCoreCvxGen npCvxBvhGlue npCvxCoreWarpAcc cvxCoreTailBda p86SignedZero npD6JointPrep npRemapIndex npContactPrePrep npRigidSumBatches npArtiExtBody npArti1DPrep npArtiJointHeader npArtiSolve1D npArtiSolveContact npArtiSolve1DTGS npArtiSolveContactTGS npArtiCheckActive npSlabRefCount npArtiPropagate npArtiPropagateAccel npArtiPropagateAccelJV npArtiSpatialMatVec npArtiBitfield npArtiSpatialMatMat npArtiConstructSM npArtiTranslateInertia npArtiTranslateSInertia npArtiPropagateZA npArtiPropagateZASph npArtiTranslateInertiaMass npArtiInvertInertia npArtiPropagateImpDof npArtiAccumLinkImp npArtiDriveParams npArtiClampDriveImp npClothDefGrad2D npSbDefGrad npDefMaterial npGridIndex npJacobiRotate npClothAtan npAnisoHelpers npSparseGridHash npClothDeltaLambda npClothExtractRot2D npClothVelClamp npClothArap2D npClothArea2D npClothWind npSbBaryInvMass npSbVolumeSolve npIsoConcaveCorner npSphSpikyKernel npIsoAcosDecode npSparseGridDomain npDenseGridCell npSbBaryProject npIsoVecNormalize npSbCompliance npSbDeltaLambdaGradC npSbTetConstraint npPsSpecVolume npSbComputeLambda npGridLocation npAnisoOutput npSmoothPos npIsoAvgVerts npPsWeightKernels npPsDiffuseUpdate npPsDiffuseSpawn npPsDiffuseAvg npPsDiffuseOneWay npSdfBoxDist npSparseGridLookup npSparseGridCellFromPos npIsoGridFilter npSbExtractRotAPD npSbSolveNeoHookean npSbSolveCorot npSbSolveGM npClothMembrane npClothBending npClothSharedEnergy npSbCollisionConstraint npClothCollisionConstraint npSdfBvhMorton npSbStress npSdfHoles npClothTriContact npManifold npSegDist npTriMask npNpUtils npConvexReduce npTriMesh npClosest npTetra npEdgeEdge deformableRef512 softBodyPsContactGen softBodyPsPrepare npBv32 npMidphaseCvx npMidphaseDeform npMidphaseAlloc npClothCloth gridCal particleSim femClothSim cvxCoreClothNp softBodySim softBodySolve512 femClothSolve512 particleContactSort femContactSort pcmManifold convexTrimeshMidphase convexHfMidphase midphaseAlloc triFeature deferContact initManifolds pairMgmt polyClip convexHull convexSat convexSelect particleGrid contactReduce cvxTriEmit cvxTriSetup sphereTriNp cvxHfSetup cvxHfPostProcess sphereHfCore hfTriFlagsBda segTriangleBda capTriangleBda capsuleHfCore radixWarp sortTriIndices cvxTriPostProcess cvxTriCorrelate cvxTriFinish npSbTet npSbSelf npSbSb npTetTriMid psPrimContact psBounds psContactGen psContactSort psContactSolve psContactSolveTgs psContactSolveRigid directApiRigidGet directApiRigidAccelGet directApiRigidGetForceTorque directApiRigidSet directApiD6JointGet directApiRadixCopy2 directApiMergeAabb directApiUpdateBodies rigidDeltaAccum psPrimCollide psDiffuseCollide fcPrim psMeshMid psHfCollide fcPrim2 npPolyClip4 fcConvexTri fcVertexConvex fcStaticTriCloth fcMidphase npSbPrim npSbRemap npCvxHfMid npTriPlaneNp npTriHfNp npCvxMeshNp npCvxHfNp npBoxClip npBoxIncident npBoxContacts npBoxSat npBoxFeature npBoxReduce boxBoxNphase sphereNphase convexPlaneNphase broadphaseSetup broadphaseReports
        broadphaseIncrementalHistogram broadphaseEndPtHistogram broadphaseIncrementalSap
        broadphaseRegionHistogram broadphaseCreateRegions broadphaseActiveRegionHistogram
        broadphaseOverlapCheckHistogram broadphaseNewBoundsPairs broadphaseMarkPairs
        broadphaseClearNewFlag broadphaseAggregateBounds broadphaseAggregateDirty
        broadphaseAggregatePairsAdd broadphaseAggregateFinalize broadphaseAggregatePairsRemove
        broadphaseAggregateBoxPruning broadphaseAggPairCollision broadphaseAggSelfCollision broadphaseAggSortProjections
        p0Subgroup p0Bda robustProbe p1DeviceV2 p4ModulePath p1HeapAlloc npWarpLib npAtomicLib npFloatIntrinRecipSqrt npTex3d npTex3dBindless p3ShimTexture3d shimBindless shimOrdering sdfTexture sdfCollision sdfTriangleCollision sdfTetrahedronCollision sdfQuery sdfFind sdfFindTriangles sdfFindTets sdfParticle psSdfMidphase psSdfMidphaseDiffuse femClothSdfContactGen femClothSdfContactGenExactGate femClothSdfContactGenNegScale softBodySdfMidphase softBodySdfMidphaseExactGate softBodySdfMidphaseNegScale warpNative p1PipelineCache p1Timestamps p6Indirect p3Shim p3CudaContext p3UnmappedLaunch p3CudaContextManager p3KernelWrangler p3ContextHardening p4Manifest p4Loader p4Foundation p4Heap p4Wrangler p4Broadphase p4Aabb solverDescLayout simCoreDescLayout prepDescLayout npAbiLayout particleSelfCollision newArticulations artiBlockLayout artiCoreDescLayout artiJointFrame artiJcalcPhaseA artiJcalcPhaseB1 artiJcalcPhaseB2 artiJcalcPhaseB3 artiJcalcPhaseC artiJcalcTendons artiJcalcGpuDirty artiJcalcInit artiJcalcLinkVel artiUnconstrainedVel artiSpatialInertia artiSpatialInertiaLaunch artiSpatialMatrix artiSpatialMatrixBlockIo artiPropagateInertia artiArticulatedInertia artiArticulatedInertiaLaunch artiDynamic artiMassMatrixWrite artiMassMatrixZeroWalk artiMassMatrixBody artiMassMatrixLaunch artiTranslateInertiaMass artiLinkAcceleration artiAccelerationLaunch artiDmaBack artiStepAlgebra artiStepQuatProbe artiStepRotVecProbe artiStepSphGuardProbe artiFlushVelocity artiAvgImpulse artiStepKernel artiSleepCheck artiUpdateBodies2 artiUpdateBodies3 artiUpdateBodies1T artiSumBatches1 artiSumBatches2 artiSumSelfBatches artiSetupInternal artiSetupTendons artiSetupMimic artiSetupICLaunch artiSolveVelImp artiSolveBodies artiSolveTMLaunch artiSolveExtTGS artiSolveSelf artiSolveSelfLaunch artiSolveStatic artiSolveDof artiSolveDFS artiSolveInternalLaunch artiAvgImpulse2 artiPropagate2 artiConstraintPrepBody artiConstraintPrepSetup artiConstraintPrepWrap artiJointPrepWrap artiSolvePartitionWrap artiOutputVelocity artiPushImpulse artiApplyTgsSubstepForces artiSaveVelocityPGS artiComputeDeps artiComputeCOM artiInverseDyn2 artiGravityForces artiCentrifugal artiMassMatrices artiUpdateKinematic artiGetSet artiJointPrepPgs artiConstraintPrepPgs artiSolveExtPgs artiSolvePartitionPgs artiSolveSelfPgs artiSolveSelfLaunchPgs artiSolveStaticPgs artiSolveDofPgs artiSolveDFSPgs artiSolveInternalLaunchPgs updateArticulations psMaterials psFluidDensity psSolveDensity psFluidVelocity psBoundaryDensity psMeshMidphaseLayout psMeshMidphase psMeshMidphaseDiffuse psMeshMidphaseNoFilter psMeshMidphaseWorkIndex psHfMidphaseLayout psHfMidphase psHfMidphaseDiffuse psHfMidphaseClobber psHfMidphaseAccum psDiffuseBuffers psDiffuseCreate psDiffuseAdvect psDiffuseOneWay psDiffuseContactGen psConvexContactGen psConvexDiffuseContactGen bpDescLayout npTetCollide
        pipelineRefusal)
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
    # updateBounds:updateTransformCacheAndBoundArrayLaunch, bodyUpdate:updateBodiesLaunch and
    # bodyUpdate:updateShapesLaunch retired: converted to the BDA engine ABI (P5 Step 4) and
    # grid-stride via WorkgroupCount(), which has no HLSL builtin (P0 policy).
    # P9.63 retires the last two bodyUpdate entries for the same reason: both were
    # one-pass ports that stopped at the engine's 64 x 256 = 16384 threads, and
    # restoring the CUDA's grid-stride loop needs WorkgroupCount().
    "updateBounds:setRigidDynamicGlobalPose"
    "directApi:getRigidDynamicGlobalPose" "directApi:getRigidDynamicLinearAcceleration"
    "directApi:setRigidDynamicLinearVelocity" "directApi:setRigidDynamicTorque"
    # solverPGS:solveBlockPartition retired: converted to the BDA engine ABI
    # (P5 Step 5) and grid-strides via WorkgroupCount(), which has no HLSL
    # builtin (P0 policy).  The remaining solverPGS entries lower to cs_6_0.
    "solverPGS:concludeBlocks" "solverPGS:writebackBlocks"
    # solverTGS:writebackBlocksTGS retired: converted to the BDA engine ABI
    # (P5 Step 4), reads its descriptors through Ptr<> and does not lower to
    # cs_6_0, so it is no longer an HLSL bit-exactness entry.
    "solverTGS:solveBlockPartitionTGS"
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
    # solverZeroBodies:ZeroBodies retired: converted to the BDA engine ABI
    # (P5 Step 5) and grid-strides via WorkgroupCount(), which has no HLSL
    # builtin (P0 policy) — same as its TGS twin.
    "solverInitVelocitiesTGS:initializeSolverVelocitiesTGS"
    "solverApplyGravityTGS:applyTGSSubstepGravity"
    # solverDmaBack:dmaBackChangedElems retired: converted to the BDA engine ABI
    # (P5 Step 5) and grid-strides via WorkgroupCount(), which has no HLSL
    # builtin (P0 policy).  Its sibling dmaConstraintResidual was removed upstream
    # (5.12 has no residual reporting), and its HLSL entry with it.
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
    "npCvxTriWorldNormal:ctCvxTriWorldNormal"
    "npGjkSupport:ctGjkSupportIndex"
    "npGjkNewPoint:ctGjkNewPoint"
    "npGjkBarycentric:ctGjkBarycentric"
    "npHfTriFlags:ctHfTriFlags"
    "npWarpVec:ctWarpDot"
    "npWarpVec:ctWarpCross"
    "npWarpVec:ctWarpSplatV"
    "npWarpVec:ctWarpSplatX"
    "npWarpVec:ctWarpPackX"
    "npGjkSimplex:ctGjkSetNormal0"
    "npGjkSimplex:ctGjkSetNormal123"
    "npGjkClosestTri:ctClosestPointTri"
    "npGjkClosestTet:ctClosestPointTet"
    "npGjkClosestSeg:ctClosestPointSeg"
    "npGjkOutput:ctGjkOutput"
    "npGjkPermute:ctGjkPermute"
    "npGjkSupportStep:ctGjkSupportStep"
    "npGjkWarmStart:ctGjkWarmStart"
    "npGjk:ctGjk"
    "npEpaStart4:ctEpaStart4"
    "npEpaStart123:ctEpaStart123"
    "npEpa:ctEpa"
    "npGjkWarm:ctGjkWarm"
    "npCvxCvxEE:cvxCvxGenEE"
    "npCvxRayPoly:cvxRayPoly"
    "npCvxFaceContacts:cvxFaceContacts"
    "npCvxReduceManifold:cvxReduceManifold"
    "npCvxWitnessPoly:cvxWitnessPoly"
    "npCvxLocalNormal:cvxLocalNormal"
    "npCvxAddManifold:cvxAddManifold"
    "npCvxBlob:cvxBlob"
    "npCvxBlobRead:cvxBlobRead"
    "npCvxEEDriver:cvxEEDriver"
    "npCvxRefFrame:cvxRefFrame"
    "npCvxPolyClip:cvxPolyClip"
    "npCvxClipEmit:cvxClipEmit"
    "npCvxContactDecide:cvxContactDecide"
    "npCvxCollidePrep:cvxCollidePrep"
    "npCvxCollideDecide:cvxCollideDecide"
    "npCvxWriteStream:cvxWriteStream"
    "npCvxPlanePolyIndex:cvxPlanePolyIndex"
    "npCvxPlaneRefresh:cvxPlaneRefresh"
    "npCvxPlaneGen:cvxPlaneGen"
    "npCvxPlaneOut:cvxPlaneOut"
    "npCvxCorrelate:cvxCorrelate"
    "npCvxPostProcess:cvxPostProcess"
    "npCvxTrimeshOut:cvxTrimeshOut"
    "npTriTriOverlap:triTriOverlap"
    "npCvxTriNpSetup:cvxTriNpSetup"
    "npCvxHfNpSetup:cvxHfNpSetup"
    "npCvxCoreOut:ccoWriteContacts" "npCvxCoreOut:ccoWriteContact"
    "npCvxCoreSetup:ccsPlanePrep" "npCvxCoreSetup:ccsConvexPrep"
    "npCvxCoreSupport:ccsSupportAll"
    "npCvxCoreFace:ccfContactFace"
    "npCvxCorePtsFace:ccpReduce" "npCvxCorePtsFace:ccpPointsFace"
    "npCvxCoreClip:cccClip"
    "npCvxCoreContact:cctAddPoints"
    "npCvxCoreGjk:cgjDistance"
    "npCvxCoreEpa:cepDepth"
    "npCvxCorePlaneGen:cpgContacts"
    "npCvxCoreCvxGen:cvgContacts"
    "npCvxBvhGlue:cbgGlue"
    "npCvxCoreWarpAcc:cwaAddPoints"
    "npD6JointPrep:d6jPrep"
    "npD6JointPrep:c1dPrePrep"
    "npRemapIndex:remapIndex"
    # npRigidSumBatches + npContactPrePrep:constraintContactBlockPrePrepLaunch:
    # BDA engine-ABI ports — Vulkan-only (P0 policy), removed from the HLSL set.
    "npContactPrePrep:cppPack"
    "npArtiExtBody:aebBody"
    "npArtiExtBody:aebFinalizePGS"
    "npArtiExtBody:aebFinalizeTGS"
    "npArti1DPrep:a1dPGS"
    "npArti1DPrep:a1dTGS"
    "npArtiJointHeader:ajhPGS"
    "npArtiJointHeader:ajhTGS"
    "npArtiSolve1D:as1dPGS"
    "npArtiSolveContact:asCtPGS"
    "npArtiSolve1DTGS:as1dTGS"
    "npArtiSolveContactTGS:asCtTGS"
    "npArtiCheckActive:ceacPGS" "npArtiCheckActive:ceacTGS"
    "npSlabRefCount:casCount"
    "npArtiPropagate:apImpulse"
    "npArtiPropagateAccel:apAccel"
    "npArtiPropagateAccelJV:apAccelJV"
    "npArtiSpatialMatVec:smMulVec"
    "npArtiBitfield:bitScan"
    "npArtiSpatialMatMat:smMulMat"
    "npArtiConstructSM:constructSM"
    "npArtiTranslateInertia:transInertia"
    "npArtiTranslateSInertia:transSInertia"
    "npArtiPropagateZA:propZA"
    "npArtiPropagateZASph:propZASph"
    "npArtiTranslateInertiaMass:transInertiaMass"
    "npArtiInvertInertia:invertInertia"
    "npArtiPropagateImpDof:propImpDof"
    "npArtiAccumLinkImp:accumLinkImp"
    "npArtiDriveParams:driveParams"
    "npArtiClampDriveImp:clampDriveImp"
    "npClothDefGrad2D:clothDefGrad2D"
    "npSbDefGrad:sbDefGrad"
    "npDefMaterial:defMaterial"
    "npGridIndex:gridIndex"
    "npJacobiRotate:jacobiRotate"
    "npClothAtan:clothAtan"
    "npAnisoHelpers:anisoHelpers"
    "npSparseGridHash:sparseGridHash"
    "npClothDeltaLambda:clothDeltaLambda"
    "npClothExtractRot2D:clothExtractRot2D"
    "npClothVelClamp:clothVelClamp"
    "npClothArap2D:clothArap2D"
    "npClothArea2D:clothArea2D"
    "npClothWind:clothWind"
    "npSbBaryInvMass:sbBaryInvMass"
    "npSbVolumeSolve:sbVolumeSolve"
    "npIsoConcaveCorner:isoConcaveCorner"
    "npSphSpikyKernel:sphSpikyKernel"
    "npIsoAcosDecode:isoAcosDecode"
    "npSparseGridDomain:sparseGridDomain"
    "npDenseGridCell:denseGridCell"
    "npSbBaryProject:sbBaryProject"
    "npIsoVecNormalize:isoVecNormalize"
    "npSbCompliance:sbCompliance"
    "npSbDeltaLambdaGradC:sbDeltaLambdaGradC"
    "npSbTetConstraint:sbTetConstraint"
    "npPsSpecVolume:psSpecVolume"
    "npSbComputeLambda:sbComputeLambda"
    "npGridLocation:gridLocation"
    "npAnisoOutput:anisoOutput"
    "npSmoothPos:smoothPos"
    "npIsoAvgVerts:isoAvgVerts"
    "npPsWeightKernels:psWeightKernels"
    "npPsDiffuseUpdate:psDiffuseUpdate"
    "npPsDiffuseSpawn:psDiffuseSpawn"
    "npPsDiffuseAvg:psDiffuseAvg"
    "npPsDiffuseOneWay:psDiffuseOneWay"
    "npSdfBoxDist:sdfBoxDist"
    "npSparseGridLookup:sparseGridLookup"
    "npSparseGridCellFromPos:sparseGridCellFromPos"
    "npIsoGridFilter:isoGridFilter"
    "npSbExtractRotAPD:sbExtractRotAPD"
    "npSbSolveNeoHookean:sbSolveNeoHookean"
    "npSbSolveCorot:sbSolveCorot"
    "npSbSolveGM:sbSolveGM"
    "npClothMembrane:clothMembrane"
    "npClothBending:clothBending"
    "npClothSharedEnergy:clothSharedEnergy"
    "npSbCollisionConstraint:sbCollisionConstraint"
    "npClothCollisionConstraint:clothCollisionConstraint"
    "npSdfBvhMorton:sdfBvhMorton"
    "npSbStress:sbStress"
    "npSdfHoles:sdfHoles"
    "npClothTriContact:clothTriContact"
    "npManifold:mfRefresh" "npManifold:mfInvalidate"
    "npSegDist:sdPointSegment" "npSegDist:sdSegmentSegment"
    "npTriMask:ctBarycentric" "npTriMask:ctActiveMaskVW" "npTriMask:ctActiveMaskFromNormal" "npTriMask:ctActiveMaskPoint"
    "npNpUtils:nuNext3Prev3" "npNpUtils:nuEdgeFlags" "npNpUtils:nuDelayContact"
    "npConvexReduce:ctReducePoints"
    "npTriMesh:ctTriNormal" "npTriMesh:ctTriBarycentric"
    "npClosest:ctClosestPtTriangle" "npClosest:ctClosestPtEdge"
    "npTetra:ctTriBBox" "npTetra:ctTetBBox" "npTetra:ctTetBary"
    "npEdgeEdge:ctLineLine" "npEdgeEdge:ctEdgeEdge"
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
    # markCreatedPairs/initializeRadixRanks/updateHandles: BDA + WorkgroupCount — Vulkan-only (P0 policy).
    "broadphaseSetup:translateAABBsLaunch"
    "broadphaseSetup:initializeSapBox1DLaunch"

    # broadphaseIncrementalHistogram:computeIncrementalComparisonHistograms_Stage1/_Stage2,
    # broadphaseIncrementalSap:performIncrementalSAP and broadphaseMarkPairs:markRemovedPairsLaunch/
    # markRemovedPairsProjectionsLaunch/markUpdatedPairsLaunch/markUpdatedPairsLaunch2 retired:
    # converted to the BDA engine ABI (P5 Step 4); BDA does not lower to HLSL cs_6_0 (and the
    # SAP/mark kernels also grid-stride via WorkgroupCount()), so they are Vulkan-only (P0 policy).
    "broadphaseEndPtHistogram:computeEndPtsHistogram"
    "broadphaseEndPtHistogram:outputEndPtsHistogram"
    "broadphaseRegionHistogram:outputRegionsHistogram"
    # clearNewFlagLaunch: BDA + WorkgroupCount (gl_NumWorkGroups) — Vulkan-only,
    # no cs_6_0 lowering (same policy as the P0 BDA spikes).
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
    # P9.62 removed "broadphaseAggPairCollision:doAggPairCollisions": it now carries
    # the CUDA's covering loop, whose stride is WorkgroupCount().x, and that has no
    # cs_6_0 lowering.  Same trade as the skinning entries in P9.61 -- a real
    # coverage defect is not worth keeping for a compile target this port ships
    # nothing against.
    "broadphaseAggSelfCollision:doSelfCollision"
    "broadphaseAggSortProjections:sortAndUpdateAggregateProjections"
    "solverPGS:computeAverageSolverBodyVelocity" "solverPGS:propagateSolverBodyVelocity"
    "scan4x4:scanPerBlockKernel4x4" "scan4x4:addBlockSumsKernel4x4"
    "radixFourBit:radixFourBitCountPerBlockKernel" "radixFourBit:radixFourBitReorderKernel"
    "reorderData:reorderKernel"
    "memCopy:MemCopyBalanced"
    "clampValues:clampMaxValue" "clampValues:clampMaxValues"
    # P9.61: only `interleaveBuffers` remains.  The other five skinning entries now
    # carry the CUDA's covering loop, and its stride is `WorkgroupCount().x *
    # BLOCK_SIZE` -- the thread count actually dispatched.  `WorkgroupCount()` has no
    # cs_6_0 lowering (slangc: "unavailable features in entry point"), the same reason
    # radixSortDesc, particleSim, particleContactSortBda and solveBlockUnified are
    # absent.  Dropping the loop to keep the HLSL twin would trade a real defect --
    # zeroNormals and normalizeNormals reached a QUARTER of the guide vertices on
    # every call, see skinning.slang -- for a compile target nothing in this port
    # ships against.
    "skinning:interleaveBuffers")
foreach(spec ${PXG_HLSL_ENTRIES})
    string(REPLACE ":" ";" parts ${spec})
    list(GET parts 0 stem)
    list(GET parts 1 entry)
    add_test(NAME gpuslang.hlsl.${entry}
        COMMAND ${PXG_SLANGC} ${SHADER_DIR}/${stem}.slang -target hlsl -profile cs_6_0
                -entry ${entry} -o ${HLSL_CHECK_DIR}/${stem}-${entry}.hlsl)
endforeach()

# ---------------------------------------------------------------------------
# Barrier uniformity (P9.16).
#
# Our warp-intrinsic emulation is groupshared + GroupMemoryBarrierWithGroupSync,
# i.e. OpControlBarrier at Workgroup scope, which SPIR-V requires to be reached
# in control flow uniform across the WHOLE BLOCK -- while the CUDA it replaces
# only ever needed warp-uniform control flow.  A transcription that is correct
# in CUDA can therefore be undefined here, and NOTHING ELSE in this suite can
# see it: the emitted output is identical (the per-lane guard alone decides who
# writes), spirv-val does not check it (the property is dynamic), and llvmpipe
# does not enforce it.  So it gets a static check.
#
# The script runs its own negative control first -- it re-injects the original
# P9.15a bug and fails if it can no longer see it -- so a green run means the
# checker is awake, not merely quiet.
add_test(NAME gpuslang.barrierUniformity
    COMMAND ${Python3_EXECUTABLE} ${GPUSLANG_ROOT}/tools/check_barrier_uniformity.py)

# ---------------------------------------------------------------------------
# CUDA twin-kernel equivalence (P9.20).
#
# femClothContactGenBda.slang gives ONE body two entry points --
# cloth_boxTriangleContactGenLaunch and cloth_convexContactGenLaunch -- because
# the two engine kernels are the same kernel written twice.  Nothing else in
# this build would notice if that stopped being true: their SIGNATURES are
# checked (shared ABI row), their BODIES are not.  This is that signal.
#
# Like the barrier checker, it runs its own negative control first: a single
# changed operand in one body must make the pair diverge, or the normalisation
# has grown too generous to discriminate.
add_test(NAME gpuslang.cudaTwinKernels
    COMMAND ${Python3_EXECUTABLE} ${GPUSLANG_ROOT}/tools/check_cuda_twin_kernels.py)

# P9.23: BDA reads whose only protection is a per-lane guard.  This platform
# turns such a guard into a mask and issues the load anyway, so the out-of-range
# address is still formed and dereferenced -- which is what P9.22 measured as a
# crash.  No fixture can see it (the value is discarded), no mutant can (same
# reason), and spirv-val cannot (the SPIR-V is valid; the speculation happens in
# the driver's back end).  This check is the only thing in the build that can.
add_test(NAME gpuslang.speculatedLoads
    COMMAND ${Python3_EXECUTABLE} ${GPUSLANG_ROOT}/tools/check_speculated_loads.py)

# ---------------------------------------------------------------------------
# Disabled engine kernels (P9.26).
#
# cloth_meshVertexContactGenLaunch computes a full contact and discards it: the
# only assignment to its `hasContacts` flag is commented out upstream, and
# PxgOutputIndex skips its atomicAdd on an empty ballot, so the kernel emits
# neither contacts nor a count.  femClothMeshVertContactGenBda.slang therefore
# implements the observable behaviour -- an empty walk and one atomicMax -- and
# deliberately does NOT transcribe two hundred lines of dead closest-point
# geometry that no fixture here could distinguish from a wrong transcription.
#
# That decision is correct only while the line stays commented, and NOTHING else
# in this build would notice if it stopped being: the ABI row, the launch site
# and the fixture all stay green while the engine starts emitting contacts the
# port does not produce.  Same device as the twin-kernel check -- a fact about
# someone else's source that this port depends on gets a check, not a comment.
#
# Its controls are drawn from real source: the live triangle siblings in the
# same file must read as live, or a green run proves nothing.
add_test(NAME gpuslang.disabledKernels
    COMMAND ${Python3_EXECUTABLE} ${GPUSLANG_ROOT}/tools/check_disabled_kernels.py)

# ---------------------------------------------------------------------------
# Vector divided by a splat scalar (P9.36).
#
# PxVec3 has no three-divide form.  operator/ (PxVec3.h:219), operator/= (:260)
# and normalize() (:295) all form the reciprocal ONCE and multiply, and so do the
# CUDA float3/float4 operators the GPU kernels actually use (cutil_math.h:342,
# :352, :485, :495).  Written in Slang, `v / m` is instead three correctly-rounded
# componentwise divides -- a different function, disagreeing with the engine in at
# least one component for 53.4595% of random vectors (measured, 3e6 samples).
#
# No fixture here can see that, and the reason is the same one that put ~60 of
# these in the corpus: the host mirrors are written from the Slang, by the same
# hand at the same time, so they divide too.  A green bit-exact suite then proves
# only that shader and mirror agree with each other.  Four fixtures went red the
# moment the first six shaders were corrected while their mirrors still divided --
# that is what this check is standing in for.  A mutant cannot see it either: a
# fixture blind to the original is blind to the mutant.
#
# It was not a scattering of slips.  femClothClothContactBda.slang's header stated
# the WRONG rule as deliberate policy ("a PxVec3/float scalar divide is transcribed
# as Slang `float3 / float` (per-component OpFDiv) ... mirrored bit-for-bit by the
# CPU reference"), and testFemClothSim.cpp carried a `divv` helper written to match
# it.  Meanwhile five other files carried the CORRECT rule in their comments.  Two
# doctrines coexisted for tranches; comments could not adjudicate between them and
# a check can.
#
# Its controls are its own: a selftest synthesises a splat divisor, a genuine
# vector divisor and a scalar-result divide, and fails the run if the rule cannot
# separate them -- a checker that flagged everything or nothing would otherwise
# pass silently.  A vector divided by a real VECTOR is faithful and common
# (shape2Vertex divides each rotated component by its own scale component, and
# cutil_math.h:337 float3/float3 is componentwise), so only the splat is flagged.
add_test(NAME gpuslang.vectorDivide
    COMMAND ${Python3_EXECUTABLE} ${GPUSLANG_ROOT}/tools/check_vector_divide.py)

# ---------------------------------------------------------------------------
# The same defect, written one component at a time (P9.45).  SIXTH checker.
#
# gpuslang.vectorDivide keys on an OpFDiv whose RESULT TYPE is a vector, which is
# what `v / m` lowers to.  The identical defect written as
# `float3(v.x / m, v.y / m, v.z / m)` -- or as three assignments -- lowers to
# SCALAR divides and slipped past it entirely.  Three sites survived P9.36 that
# way, all transcriptions of PxVec3::normalize() (which is one reciprocal and
# three multiplies, PxVec3.h:260/:293): computeBasisVectors in
# pxgSlangFemRigidContact, normalize3 in psContactSolveBda, and asNormalizeInPlace
# in npArtiStep.
#
# Neither a fixture nor a mutant found them, and P9.45 measured why twice over: the
# cloth rigid-contact fixture carried ONE contact normal for eight tranches, and
# for that normal the two spellings agree in every component.  Widening it to six
# normals exposed the first branch immediately; the SECOND branch needed a normal
# searched for by enumeration before its mutant would die.
#
# The rule flags two or more scalar OpFDivs in one function that share a divisor
# and take their dividends from OpCompositeExtract with DISTINCT component
# indices -- a vector taken apart and divided.  A shared denominator over
# independent dividends (femClothPrimitives.cu:129, softBody.cu:558) is not an
# extract and is not flagged.  Its selftest synthesises all four shapes and fails
# the run if the rule cannot separate them.
#
# The allowlist is NOT empty here, and that is a finding: the engine writes
# componentwise division out by hand in two places, both on raw floats rather than
# through PxVec3 -- calcGridPos (gridCal.cuh:38-40, transcribed in seven shaders)
# and the cone limit's PxTan(clamped.y/4) pair (CmConeLimitHelper.h:173).
add_test(NAME gpuslang.componentDivide
    COMMAND ${Python3_EXECUTABLE} ${GPUSLANG_ROOT}/tools/check_component_divide.py)

# ---------------------------------------------------------------------------
# A precision-LOOSE GLSL.std.450 instruction must not stand in for an exact
# engine expression (P9.47).  The SEVENTH static checker.
#
# The Vulkan environment spec gives Length, Distance and Normalize a precision
# allowance and lets an implementation lower them as it pleases.  The engine is
# exact on both sides of every one of them: PxVec3::magnitude() is
# PxSqrt(magnitudeSquared()) (PxVec3.h:179), cutil_math's length() is
# sqrtf(dot(v,v)), PxVec3::getNormalized() is v * PxRecipSqrt(m) where
# PxRecipSqrt is 1.0f/sqrtf, PxVec3::normalize() is one reciprocal and three
# multiplies through operator/= (PxVec3.h:260), and PxQuat::getNormalized() is
# 1/magnitude() and four multiplies.  None of those is a Length or a Normalize.
#
# The audit that produced this rule found FIFTY-EIGHT such calls, and every
# single one of them mapped to one of those exact engine expressions -- not one
# site in the corpus had a CUDA original that called an intrinsic with the same
# latitude.  Hence the empty allowlist, which is the finding: unlike
# gpuslang.componentDivide there is no faithful case to carve out.
#
# InverseSqrt is deliberately NOT in this rule: P9.33a measured it against
# 1.0f/sqrtf over 258442 inputs and gpuslang.npFloatIntrinRecipSqrt keeps that
# measurement live.  Sqrt is not in it either -- it is correctly rounded by the
# same spec that leaves Length loose, so it IS the engine's operation.
add_test(NAME gpuslang.lengthIntrinsic
    COMMAND ${Python3_EXECUTABLE} ${GPUSLANG_ROOT}/tools/check_length_intrinsic.py)

# ---------------------------------------------------------------------------
# A sub-word store must be atomic unless one invocation owns the whole word
# (P9.83a).  The SEVENTEENTH static checker, and the first written because a
# MUTANT SURVIVED rather than because a defect shipped.
#
# This device enables no 8-bit storage, so every PxU8 the engine's buffers hold
# is written through the 32-bit word containing it, and four PxU8 lanes share a
# word.  A plain read-modify-write is then a lost update whenever two invocations
# own two different bytes -- which is the NORMAL case, not a corner one.
#
# P9.83a's sweep replaced triTriOverlapBda's InterlockedAnd/InterlockedOr pair
# with the plain form and the mutant survived, on merit: the race needs two
# groups inside the same word at the same instant, llvmpipe may serialize them,
# and a fixture that fails only when a race happens to manifest is a flaky test
# rather than a measurement.  Every other survivor in that sweep was answered by
# strengthening the fixture; this one could not be, because the defect is not a
# function of the input.  Its SHAPE, though, is decidable by reading it.
#
# Both rules were verified by mutation before the checker was registered: the
# plain read-modify-write is reported, and so is an InterlockedAnd whose matching
# InterlockedOr has been deleted.
add_test(NAME gpuslang.subwordStore
    COMMAND ${Python3_EXECUTABLE} ${GPUSLANG_ROOT}/tools/check_subword_store.py)

# ---------------------------------------------------------------------------
# No entry point may declare more Workgroup storage than the device allows
# (P9.83b-1).  The EIGHTEENTH static checker, and the THIRD written because its
# defect had already happened: P5 had to cap solveWholeIslandTGS's shared staging
# 944 -> 640 entries after a pipeline-creation failure, found by hand and late.
#
# llvmpipe reports maxComputeSharedMemorySize = 32768 (measured with vulkaninfo,
# not assumed).  Past it, vkCreateComputePipelines fails, the kernel is never
# dispatched, and on this port an undispatched kernel is SILENT -- the same
# failure mode the [pxg-unmapped] guards exist for.
#
# THE NUMBER COMES OUT OF THE COMPILED .spv, not out of arithmetic on the source,
# because two effects the source cannot show move it: SPIR-V pads Workgroup
# structs to their members' alignment, and an import brings its module's
# groupshared along.  P9.83b's projection for triangleTriangleCollision was done
# both ways by hand and came out 29.2 KB one time and 38.9 KB the other.
#
# It found one entry point in 171 above the 75% watermark -- solveWholeIslandTGS,
# at 31744 of 32768, which is 1024 bytes from a hard failure.  That margin was
# unwatched until this existed.  Verified by mutation before registering: a
# deliberately bloated groupshared array in triTriOverlapBda was reported at 33600
# with the overage named.
add_test(NAME gpuslang.groupshared
    COMMAND ${Python3_EXECUTABLE} ${GPUSLANG_ROOT}/tools/check_groupshared.py ${SHADER_OUT_DIR})

# ---------------------------------------------------------------------------
# A mapped kernel must have a launch-ABI row, or the engine cannot run it
# (P9.48d).  The EIGHTH static checker, and the only one written because the
# defect had already happened.
#
# pushConstantSizeFor() (bridge/PhysXGpuVulkan.cpp:178) derives a kernel's push-
# constant size from bridge/PxgVulkanKernelAbi.cpp and returns 0 for a kernel
# that is not listed there.  The shim's blob check is EXACT
# (PxgCudaShim.cpp:232), so an unlisted kernel has every launch REFUSED and
# never runs -- while its fixtures pass, the manifest test passes, the suite
# passes and the scene sweep passes, because a fixture dispatches the shader
# directly and never goes through the bridge.
#
# P9.48a and P9.48b mapped four kernels without their ABI rows.  All four were
# refused on every launch for two whole tranches, and the only thing that
# noticed was a scene whose particles fell through a softbody.
#
# The rule is a RATCHET: the set of mapped-but-unwired kernels may shrink, never
# grow, and a kernel wired since the last edit must come off the list.  54 sit
# there today, on paths the engine does not drive in this port; claiming each is
# fine would be inventing certainty the check does not have.
add_test(NAME gpuslang.launchAbi
    COMMAND ${Python3_EXECUTABLE} ${GPUSLANG_ROOT}/tools/check_launch_abi.py)

# ---------------------------------------------------------------------------
# Replicated engine helpers must agree (P9.37).
#
# This corpus writes the same engine helper out by hand over and over -- 69 copies
# of PxQuat::rotate, 62 of rotateInv, 19 of closestPtPointTriangle2.  Copies are the
# risk, and the two preceding tranches both ended on one: P9.36 found an engine
# block ported FOUR times with one copy dividing where the other three multiplied,
# and P9.37 found rotateInv written 63 times with ONE copy carrying an
# algebraically-equivalent but bitwise-different form.  That copy fed the softbody
# plane path, where the value both gates contact existence and is written out as the
# penetration the solver consumes -- and the file's own comment had already said
# that a last-bit difference matters, while handing the different form over anyway.
#
# No fixture saw it, for the reason P9.36 measured in detail: the fixture's geometry
# is exact by construction (a 45-degree quaternion, an axis-aligned plane), so the
# two forms agree there and the mutant is unreachable.  Source-level review did not
# see it either -- clustering the 340 copies by source tokens produced 48 shapes of
# which every one examined was a false split (statement decomposition, `1` vs `1u`,
# helper renaming), which buries a real difference in noise.
#
# So the check reads what the GPU actually runs: it clusters same-named functions in
# the built SPIR-V by the expression DAG they compute.  The compiler has already
# inlined single-use temporaries, so naming and decomposition disappear, and hashing
# each value's subtree rather than the instruction stream removes scheduling too.
# What survives is association -- which is precisely the property at issue.
#
# Its control is its own: the selftest asserts that reordering two independent
# instructions does NOT change the hash and that re-associating a product DOES.  A
# screen failing either would be silently useless, and an earlier draft of this file
# failed the first.
add_test(NAME gpuslang.helperUniformity
    COMMAND ${Python3_EXECUTABLE} ${GPUSLANG_ROOT}/tools/check_helper_uniformity.py)
# The gate reads the BUILT SPIR-V, so it needs the shader directory the functional
# suites get; without it the script refuses ("set PXG_SHADER_DIR") and ctest
# reported a failure that said nothing about the shaders.
set_tests_properties(gpuslang.helperUniformity PROPERTIES
    ENVIRONMENT "PXG_SHADER_DIR=${SHADER_OUT_DIR}")

# ---------------------------------------------------------------------------
# A constant that names a PxGeometryType member must carry that member's value
# (P9.57).  The TENTH static checker, and the second written because the defect
# had already happened.
#
# `PxGeometryType` grew eCONVEXCORE at position 4, which shifted eCONVEXMESH to 5,
# eTRIANGLEMESH to 8 and eHEIGHTFIELD to 9.  A Slang shader cannot include the
# header, so the corpus carries fifty-odd hand-written copies of those numbers --
# and two of them were still on the old numbering, both on LIVE MAPPED kernels:
#
#   pxgSlangCvxTriSetup  CT_GEOM_TRIANGLEMESH = 4   -> the convex-vs-mesh operand
#                                                      swap never fired, so a
#                                                      mesh-first pair would have
#                                                      read the MESH blob as a hull
#   softBodyContactGen   GEOM_CONVEXMESH      = 4   -> a rigid convex mesh failed
#                                                      the GJK dispatch and its
#                                                      softbody contact was dropped
#                                                      in silence
#
# No fixture could see either, because THE FIXTURES CARRIED THE SAME NUMBER:
# testCvxTriSetup declared `GEOM_TRIANGLEMESH = 4`, set a shape's type to it and
# asserted the swap, so shader and oracle agreed exactly on a value the engine
# never produces.  That is the failure mode pxgSlangCvxTriSetup's own header
# describes for the BV32 stride, six lines above the defect.  The fixtures now take
# the values from `PxGeometry.h`, so the compiler is the oracle on the C++ side;
# this rule is the same oracle for the Slang side, where no include can reach.
add_test(NAME gpuslang.geomTypeConstants
    COMMAND ${Python3_EXECUTABLE} ${GPUSLANG_ROOT}/tools/check_geom_type_constants.py)

# P9.59: the ELEVENTH static checker, and the THIRD written because its defect had
# already happened twice.  A mapped kernel whose CUDA covers its work with a
# grid-stride loop must have that loop in its Slang port; a port that makes one pass
# processes only `gridDim.x * groupSize` items and leaves the rest untouched.  That is
# not graceful: P9.52b-6 measured a SIGSEGV in convexTrimeshCorrelate reading records
# the uncovered pairs never wrote, and P9.58 measured a box falling straight through a
# heightfield.  Neither a fixture nor a mutation sweep can see it -- a fixture chooses
# its own grid, so the ceiling is never approached, and a mutant that removes the loop
# SURVIVES and reads as an equivalence.  A scene can, but only one whose work exceeds
# that kernel's own ceiling, and the ceilings differ by four orders of magnitude.
# The known-omission list is a RATCHET: it may shrink, never grow, and an entry left
# behind after its port is fixed fails the check too.
add_test(NAME gpuslang.gridStride
    COMMAND ${Python3_EXECUTABLE} ${GPUSLANG_ROOT}/tools/check_grid_stride.py)

# TWELFTH static checker (P9.64), and the other half of the sentence above: "a
# fixture chooses its own grid, so the ceiling is never approached".  gridStride
# checks that the SHADER loops; this checks that the FIXTURE gives it the engine's
# grid, without which the loop is never needed and neither its absence nor a wrong
# stride constant is visible.  Written because the blind spot had already hidden a
# real defect three times (P9.52b-6, P9.61, P9.63) -- the P9.57 precedent.  It found
# a fourth on its first run: thresholdForceChange.slang carried one stride constant
# for a file the engine launches with two different grids.
# Entries whose shader strides by WorkgroupCount() are EXEMPT: those are correct at
# any grid, so the fixture's is a free choice.  The known list is a RATCHET and is
# currently empty.
add_test(NAME gpuslang.fixtureGrid
    COMMAND ${Python3_EXECUTABLE} ${GPUSLANG_ROOT}/tools/check_fixture_grid.py)

# THIRTEENTH static checker (P9.65).  Not every unmapped kernel is remaining work.
# The isosurface extractors, SDF builder, anisotropy and smoothed-position generators
# and the particle neighbourhood provider are created only through PxGetPhysicsGpu(),
# whose factory (physxgpu/src/PxgPhysXGpu.cpp) is not compiled into this build and
# whose bridge replacement returns NULL -- 58 of the 98 unmapped kernels cannot be
# reached here at all, and 11 more are launched by nothing even upstream.  The real
# remaining work is the 29 the simulation pipeline launches, ratcheted in
# tools/engine_reachable_unmapped.txt.  The checker re-derives the unreachability
# basis every run rather than trusting a hardcoded list, so wiring one of these
# features into the engine fails the build instead of silently invalidating the plan.
add_test(NAME gpuslang.unmappedReachability
    COMMAND ${Python3_EXECUTABLE} ${GPUSLANG_ROOT}/tools/check_unmapped_reachability.py)

# FOURTEENTH static checker (P9.68), and the class gridStride structurally cannot see.
# gridStride fires only when the CUDA ITSELF carries a grid-stride loop.  This one is
# about the gap between the two BLOCK sizes: the bridge forwards gridDim and DROPS
# blockDim, so on a launch site that writes `numBlocks = ceil(n / numThreadsPerBlock)`
# the Slang [numthreads] IS the block, and a narrower group reaches
# slangThreads/cudaBlock of the work -- a fixed FRACTION, on every call, at every
# scene size, with no CUDA loop missing anywhere.
# It has happened twice: P9.61's zeroNormals/normalizeNormals (a quarter of the guide
# vertices, forever) and P9.67c's convexCorePlaneNphase_Kernel (one sixteenth of the
# pairs), both found by hand and by accident.  P9.68 measured the second by widening
# PXG_SCENE=convexcoreplane to 80 sleepless cylinders: exactly 16 fell through the
# plane, 32 failed checks, 0 after the loop.
# Four separate PASS conditions exist because the check without each of them reported
# ports that DO cover their work (35 -> 14 -> 7 -> 0 across four readings): a grid not
# sized from the block, a covering stride, and a serial partition of the block are all
# legitimate.  The known list is a RATCHET and is currently EMPTY.
add_test(NAME gpuslang.blockCoverage
    COMMAND ${Python3_EXECUTABLE} ${GPUSLANG_ROOT}/tools/check_block_coverage.py)

# P9.77, the FIFTEENTH checker, and the THIRD written because its defect had already
# happened.  gpuslang.launchAbi (P9.48d) asks whether a mapped kernel HAS an ABI row;
# it cannot ask whether the row MATCHES.  P9.76b is the difference: the softbody
# attachment TGS solve carried the 72-byte PGS row while its launch site pushes 80,
# so the shim's exact blobSize == pushConstantSize gate refused every launch and the
# kernel had never run on Vulkan -- with fixtures, the suite and all 48 scenes green,
# because a fixture dispatches the shader directly and never passes the gate.
# This one derives the blob from the PX_CUDA_KERNEL_PARAM list at the launch site and
# compares.
# P9.97b: PER ARGUMENT, not totals (P9.97 measured that the totals-only version and
# check_push_block together let five of nine row mutants through, two runtime-fatal).
# All 415 rows: I1 row integrity (array[:argCount], duplicates, unparsed arrays FAIL),
# I4 hostPtrMask by size, the host leg (410 decided per argument over 497 launch sites
# in both launch forms, 1 by named tail waiver, 0 undecided, 5 named no-site rows), and
# the CUDA-signature leg (407 equal, 8 host-shaped rows by name, 0 unresolved).  A
# selftest on scratch copies of the ABI text with deliberately wrong rows, and on literal
# resolver snippets, runs first.  About 4 s (was 17 s).
add_test(NAME gpuslang.launchAbiPack
    COMMAND ${Python3_EXECUTABLE} ${GPUSLANG_ROOT}/tools/check_launch_abi_pack.py)

# P9.78: the SIXTEENTH checker.  pushConstantSizeFor() derives the pushed blob
# size from the ABI ROW and never from the SPIR-V, so a shader whose push block
# disagrees with its row is not refused -- it reads bytes that were never pushed,
# silently.  This walks each entry point's call graph and compares the highest
# push-constant byte it can reach against its row.  It needs the built .spv, so
# it depends on the shader target like the other SPIR-V readers.
# P9.97b: plus the LAYOUT JOIN -- every reached leaf member laid against the row's
# arguments as the bridge places them (paramAlign), by-value aggregates flattened into
# their fields: PAD-READ / STRADDLE / CUT / F-SUBWORD, never ratcheted.  It caught
# sb_sdfMeshMidphaseGeneratePairsLaunch reading its 80-byte writer at @88 where the
# bridge pushes it at @96; 415 kernels joined, 0 findings since that shader's fix.
# P9.99: + R3 RUN-ON (a member over a field of >= 4 bytes running on into padding) and the
# NAME leg -- every reached member against the CUDA names of the fields it covers, so a
# swap of equal-size members (two pointers, or a word over a bool and a PxU32) that no
# offset can see is NAME-SWAP; 0 findings.  Swaps neither the join nor the names can see
# are simulated, in the shader and (check round) in the CUDA declaration: 324 of 337
# kernels with an equal-size reached pair decidable, 9 of 9728 CUDA-side pairs blind;
# 17 kernels on tools/push_name_known.txt.  Selftest plus an end-to-end red run of the
# whole checker on a scratch dir with two member names swapped; about 12 s.
add_test(NAME gpuslang.pushBlock
    COMMAND ${Python3_EXECUTABLE} ${GPUSLANG_ROOT}/tools/check_push_block.py ${SHADER_OUT_DIR})

# P9.99: the premise behind createPipelineEx's refusals (no fallback): no engine host path
# calls createPipelineEx (src/, bridge/, include/: declarations only, comments and strings
# stripped), and no engine module carries an explicit subgroup dependency -- capability,
# Subgroup* built-in, OpGroupNonUniform* / OpSubgroup*KHR, or a barrier / atomic with
# Scope Subgroup.  The allowlist of probe modules that use subgroups is exact over the
# whole shader directory.  A selftest injects each kind into an engine module first.  No
# GPU, under a second.
add_test(NAME gpuslang.subgroupOps
    COMMAND ${Python3_EXECUTABLE} ${GPUSLANG_ROOT}/tools/check_subgroup_ops.py ${SHADER_OUT_DIR})

# P9.97: the mutation-sweep runners derive their per-run bound from a timed baseline
# (tools/sweep_timeout.sh) instead of a fixed 300 s, which refused the 435-527 s
# triTriCollisionBda suite and, near the bound, would read a merely slower mutant as
# KILLED-HANG.  This pins the POLICY: the derivation table as literals, the knob
# validation (0, suffixes, leading zeros, factor < 2), the warning boundary, the
# 137 -> hang mapping, the Python twin (derive_timeout, knob, the process-group kill of
# run_bounded), and that no runner calls timeout(1) on its own.  No GPU, about 20 s.
add_test(NAME gpuslang.sweepTimeoutPolicy
    COMMAND sh ${GPUSLANG_ROOT}/tools/test_sweep_timeout.sh)

# P9.97: the runners THEMSELVES, end to end, on a mirror tree with fake suite and scene
# binaries (no GPU, no slangc, about 80 s): the refusal runs the suite once, the derived
# bound is printed and used, a SIGTERM-ignoring hang and a hang followed by a crash are
# hangs (one attempt), a closing-baseline timeout says so, sweep_common.sh refuses a
# baseline that times out or reads "10 failed", and the host runner refuses an empty
# regex before building, sizes its outer bound by the number of tests, and does not
# re-run a red baseline.  Every row also checks that source and .spv are restored.
add_test(NAME gpuslang.sweepRunners
    COMMAND sh ${GPUSLANG_ROOT}/tools/test_sweep_runners.sh)

# The Windows guard for the Vulkan PhysXGpu (PX_GPU_BACKEND=VULKAN): scans exactly the
# sources cmake/PxgVulkanGpuSources.cmake hands the official build, plus the gpuslang
# headers, for what MSVC rejects and gcc/clang accept -- RTTI under /GR-, GNU
# attributes/builtins/keywords, POSIX headers and calls outside a non-Windows branch, a
# raw <windows.h>, a non-literal __declspec(align()).  Runs its self-test first (every
# rule must fire on a planted line).  No GPU, no slangc, well under a second.
include(${GPUSLANG_ROOT}/cmake/PxgVulkanGpuSources.cmake)
pxg_vulkan_gpu_sources(${GPUSLANG_ROOT}/..)
add_test(NAME gpuslang.msvcPortability
    COMMAND ${Python3_EXECUTABLE} ${GPUSLANG_ROOT}/tools/check_msvc_portability.py
            ${PXG_VK_HOST_SOURCES} ${PXG_VK_BRIDGE_SOURCES} ${PXG_VK_BACKEND_SOURCES})

# P9.98 check round 1: tools/vvl_triage.py against the checked-in waivers, on tiny
# inline logs: a device / shim / bridge refusal fails outside the fixtures that provoke it
# (it used to pass), a widened push range fails unless its entry is listed, the F5 waivers
# hold only under their allocation bound.  No GPU, no layer, about 2 s.
add_test(NAME gpuslang.vvlTriage
    COMMAND sh ${GPUSLANG_ROOT}/tools/test_vvl_triage.sh)

# P9.98: a small validation-layer run -- tools/run_validation.sh subset: the
# pipelineRefusal fixtures plus the suites whose modules the layer flagged in the P9.98
# scope (sdfFind, sdfParticle, psSdfMidphase, psDiffuseContactGen, solverTGS) and the
# sync-control suites (scan, p6Indirect), in both memory modes, failing on any message
# tools/validation_waivers.txt does not cover.  SKIPPED (77) when no
# VK_LAYER_KHRONOS_validation is found -- set PXG_VVL_DIR to an extracted
# vulkan-validationlayers package to arm it.  The full run is
# `sh tools/run_validation.sh suite` (about 9 min per memory mode).
add_test(NAME gpuslang.validationSubset
    COMMAND sh ${GPUSLANG_ROOT}/tools/run_validation.sh subset)
set_tests_properties(gpuslang.validationSubset PROPERTIES
    ENVIRONMENT "PXG_SHADER_DIR=${SHADER_OUT_DIR};PXG_BUILD_TESTS=${CMAKE_BINARY_DIR}"
    SKIP_RETURN_CODE 77)
