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
    ${SHADER_DIR}/pxgSlangPrepCommon.slang
    ${SHADER_DIR}/pxgSlangWarpInc.slang
    ${SHADER_DIR}/pxgSlangBpDesc.slang)

# Shaders whose SPIR-V gets NoContraction decorations stamped onto every
# float arithmetic op (tools/spv_nocontract.py).  slangc emits none and its
# -fp-mode precise flag is a no-op for SPIR-V, while Vulkan permits drivers
# to contract undecorated expressions: llvmpipe fuses npCvxCoreEpa's
# makePlane triple-cross sums into fma chains (1-ulp vs strict IEEE), which
# breaks the bitwise CPU-vs-GPU doctrine.  The decoration forbids that.
set(PXG_NOCONTRACT_SHADERS npCvxCoreEpa npCvxCorePlaneGen npCvxCoreCvxGen npD6JointPrep npArtiExtBody npArti1DPrep npArtiJointHeader npArtiSolve1D npArtiSolveContact npArtiSolve1DTGS npArtiSolveContactTGS npArtiCheckActive npArtiPropagate npArtiPropagateAccel npArtiPropagateAccelJV npArtiSpatialMatVec npArtiSpatialMatMat npArtiTranslateInertia npArtiTranslateSInertia npArtiPropagateZA npArtiPropagateZASph npArtiTranslateInertiaMass npArtiInvertInertia npArtiPropagateImpDof npArtiAccumLinkImp npArtiDriveParams npArtiClampDriveImp npClothDefGrad2D npSbDefGrad npDefMaterial npJacobiRotate npClothAtan npAnisoHelpers npClothDeltaLambda npClothExtractRot2D npClothVelClamp npClothArap2D npClothArea2D npClothWind npSbBaryInvMass npSbVolumeSolve npIsoConcaveCorner npSphSpikyKernel npIsoAcosDecode npSparseGridDomain npDenseGridCell npSbBaryProject npIsoVecNormalize npSbCompliance npSbDeltaLambdaGradC npSbTetConstraint npPsSpecVolume npSbComputeLambda npGridLocation npAnisoOutput npSmoothPos npIsoAvgVerts npPsSolvePC npPsSolvePCTGS npPsWeightKernels npPsDiffuseUpdate npPsDiffuseSpawn npPsDiffuseAvg npPsDiffuseOneWay npSdfBoxDist npSparseGridCellFromPos npIsoGridFilter npSbExtractRotAPD npSbSolveNeoHookean npSbSolveCorot npSbSolveGM npClothMembrane npClothBending npClothSharedEnergy npSbCollisionConstraint npClothCollisionConstraint npSdfBvhMorton npSbStress npSdfHoles npClothTriContact)

# pxg_compile_shader(<file-stem> <entry> [<entry> ...])
function(pxg_compile_shader stem)
    set(entries "")
    foreach(entry ${ARGN})
        list(APPEND entries -entry ${entry})
    endforeach()
    set(postprocess "")
    if(stem IN_LIST PXG_NOCONTRACT_SHADERS)
        set(postprocess COMMAND python3 ${GPUSLANG_ROOT}/tools/spv_nocontract.py ${SHADER_OUT_DIR}/${stem}.spv)
    endif()
    add_custom_command(
        OUTPUT ${SHADER_OUT_DIR}/${stem}.spv
        COMMAND ${PXG_SLANGC} ${SHADER_DIR}/${stem}.slang -target spirv ${entries}
                ${PXG_SLANG_FLAGS} -o ${SHADER_OUT_DIR}/${stem}.spv
        ${postprocess}
        DEPENDS ${SHADER_DIR}/${stem}.slang ${SHADER_COMMON_DEPS} ${GPUSLANG_ROOT}/tools/spv_nocontract.py
        COMMENT "slangc ${stem}.slang -> ${stem}.spv"
        VERBATIM)
    list(APPEND PXG_SHADER_OUTPUTS ${SHADER_OUT_DIR}/${stem}.spv)
    set(PXG_SHADER_OUTPUTS ${PXG_SHADER_OUTPUTS} PARENT_SCOPE)
endfunction()

pxg_compile_shader(scan scanPerBlockKernel addBlockSumsKernel)
pxg_compile_shader(radixSort radixSortHistogram radixSortCalculateRanks)
pxg_compile_shader(radixSort64 radixSortCopyHigh32Bits)
# P5 Step 4: descriptor-based multi-launch radix sort (BDA engine ABI).
pxg_compile_shader(radixSortDesc radixSortMultiBlockLaunchWithCount radixSortMultiCalculateRanksLaunchWithCount)
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
pxg_compile_shader(npCvxTriWorldNormal ctCvxTriWorldNormal)
pxg_compile_shader(npGjkSupport ctGjkSupportIndex)
pxg_compile_shader(npGjkNewPoint ctGjkNewPoint)
pxg_compile_shader(npGjkBarycentric ctGjkBarycentric)
pxg_compile_shader(npHfTriFlags ctHfTriFlags)
pxg_compile_shader(npWarpVec ctWarpDot ctWarpCross ctWarpSplatV ctWarpSplatX ctWarpPackX)
pxg_compile_shader(npGjkSimplex ctGjkSetNormal0 ctGjkSetNormal123)
pxg_compile_shader(npGjkClosestTri ctClosestPointTri)
pxg_compile_shader(npGjkClosestTet ctClosestPointTet)
pxg_compile_shader(npGjkClosestSeg ctClosestPointSeg)
pxg_compile_shader(npGjkOutput ctGjkOutput)
pxg_compile_shader(npGjkPermute ctGjkPermute)
pxg_compile_shader(npGjkSupportStep ctGjkSupportStep)
pxg_compile_shader(npGjkWarmStart ctGjkWarmStart)
pxg_compile_shader(npGjk ctGjk)
pxg_compile_shader(npEpaStart4 ctEpaStart4)
pxg_compile_shader(npEpaStart123 ctEpaStart123)
pxg_compile_shader(npEpa ctEpa)
pxg_compile_shader(npGjkWarm ctGjkWarm)
pxg_compile_shader(npCvxCvxEE cvxCvxGenEE)
pxg_compile_shader(npCvxRayPoly cvxRayPoly)
pxg_compile_shader(npCvxFaceContacts cvxFaceContacts)
pxg_compile_shader(npCvxReduceManifold cvxReduceManifold)
pxg_compile_shader(npCvxWitnessPoly cvxWitnessPoly)
pxg_compile_shader(npCvxLocalNormal cvxLocalNormal)
pxg_compile_shader(npCvxAddManifold cvxAddManifold)
pxg_compile_shader(npCvxBlob cvxBlob)
pxg_compile_shader(npCvxBlobRead cvxBlobRead)
pxg_compile_shader(npCvxEEDriver cvxEEDriver)
pxg_compile_shader(npCvxRefFrame cvxRefFrame)
pxg_compile_shader(npCvxPolyClip cvxPolyClip)
pxg_compile_shader(npCvxClipEmit cvxClipEmit)
pxg_compile_shader(npCvxContactDecide cvxContactDecide)
pxg_compile_shader(npCvxCollidePrep cvxCollidePrep)
pxg_compile_shader(npCvxCollideDecide cvxCollideDecide)
pxg_compile_shader(npCvxLostFound lostFoundStage1 lostFoundStage2)
pxg_compile_shader(npCvxWriteStream cvxWriteStream)
pxg_compile_shader(npCvxFinishContacts cvxFinishContacts)
pxg_compile_shader(npCvxStage1Refresh cvxStage1Refresh)
pxg_compile_shader(npCvxStage2Setup cvxStage2Setup)
pxg_compile_shader(npCvxPlanePolyIndex cvxPlanePolyIndex)
pxg_compile_shader(npCvxPlaneRefresh cvxPlaneRefresh)
pxg_compile_shader(npCvxPlaneGen cvxPlaneGen)
pxg_compile_shader(npCvxPlaneOut cvxPlaneOut)
pxg_compile_shader(npCvxCorrelate cvxCorrelate)
pxg_compile_shader(npCvxPostProcess cvxPostProcess)
pxg_compile_shader(npCvxTrimeshOut cvxTrimeshOut)
pxg_compile_shader(npTriTriOverlap triTriOverlap)
pxg_compile_shader(npCvxTriNpSetup cvxTriNpSetup)
pxg_compile_shader(npCvxHfNpSetup cvxHfNpSetup)
pxg_compile_shader(npCvxCoreOut ccoWriteContacts ccoWriteContact)
pxg_compile_shader(npCvxCoreSetup ccsPlanePrep ccsConvexPrep)
pxg_compile_shader(npCvxCoreSupport ccsSupportAll)
pxg_compile_shader(npCvxCoreFace ccfContactFace)
pxg_compile_shader(npCvxCorePtsFace ccpReduce ccpPointsFace)
pxg_compile_shader(npCvxCoreClip cccClip)
pxg_compile_shader(npCvxCoreContact cctAddPoints)
pxg_compile_shader(npCvxCoreGjk cgjDistance)
pxg_compile_shader(npCvxCoreEpa cepDepth)
pxg_compile_shader(npCvxCorePlaneGen cpgContacts)
pxg_compile_shader(npCvxCoreCvxGen cvgContacts)
pxg_compile_shader(npCvxBvhGlue cbgGlue)
pxg_compile_shader(npCvxCoreWarpAcc cwaAddPoints)
pxg_compile_shader(npD6JointPrep d6jPrep c1dPrePrep)
pxg_compile_shader(npRemapIndex remapIndex)
pxg_compile_shader(npContactPrePrep cppPack)
pxg_compile_shader(npRigidSumBatches rsBatches1 rsBatches2)
pxg_compile_shader(npArtiExtBody aebBody aebFinalizePGS aebFinalizeTGS)
pxg_compile_shader(npArti1DPrep a1dPGS a1dTGS)
pxg_compile_shader(npArtiJointHeader ajhPGS ajhTGS)
pxg_compile_shader(npArtiSolve1D as1dPGS)
pxg_compile_shader(npArtiSolveContact asCtPGS)
pxg_compile_shader(npArtiSolve1DTGS as1dTGS)
pxg_compile_shader(npArtiSolveContactTGS asCtTGS)
pxg_compile_shader(npArtiCheckActive ceacPGS ceacTGS)
pxg_compile_shader(npSlabRefCount casCount)
pxg_compile_shader(npArtiPropagate apImpulse)
pxg_compile_shader(npArtiPropagateAccel apAccel)
pxg_compile_shader(npArtiPropagateAccelJV apAccelJV)
pxg_compile_shader(npArtiSpatialMatVec smMulVec)
pxg_compile_shader(npArtiBitfield bitScan)
pxg_compile_shader(npArtiSpatialMatMat smMulMat)
pxg_compile_shader(npArtiConstructSM constructSM)
pxg_compile_shader(npArtiTranslateInertia transInertia)
pxg_compile_shader(npArtiTranslateSInertia transSInertia)
pxg_compile_shader(npArtiPropagateZA propZA)
pxg_compile_shader(npArtiPropagateZASph propZASph)
pxg_compile_shader(npArtiTranslateInertiaMass transInertiaMass)
pxg_compile_shader(npArtiInvertInertia invertInertia)
pxg_compile_shader(npArtiPropagateImpDof propImpDof)
pxg_compile_shader(npArtiAccumLinkImp accumLinkImp)
pxg_compile_shader(npArtiDriveParams driveParams)
pxg_compile_shader(npArtiClampDriveImp clampDriveImp)
pxg_compile_shader(npClothDefGrad2D clothDefGrad2D)
pxg_compile_shader(npSbDefGrad sbDefGrad)
pxg_compile_shader(npDefMaterial defMaterial)
pxg_compile_shader(npGridIndex gridIndex)
pxg_compile_shader(npJacobiRotate jacobiRotate)
pxg_compile_shader(npClothAtan clothAtan)
pxg_compile_shader(npAnisoHelpers anisoHelpers)
pxg_compile_shader(npSparseGridHash sparseGridHash)
pxg_compile_shader(npClothDeltaLambda clothDeltaLambda)
pxg_compile_shader(npClothExtractRot2D clothExtractRot2D)
pxg_compile_shader(npClothVelClamp clothVelClamp)
pxg_compile_shader(npClothArap2D clothArap2D)
pxg_compile_shader(npClothArea2D clothArea2D)
pxg_compile_shader(npClothWind clothWind)
pxg_compile_shader(npSbBaryInvMass sbBaryInvMass)
pxg_compile_shader(npSbVolumeSolve sbVolumeSolve)
pxg_compile_shader(npIsoConcaveCorner isoConcaveCorner)
pxg_compile_shader(npSphSpikyKernel sphSpikyKernel)
pxg_compile_shader(npIsoAcosDecode isoAcosDecode)
pxg_compile_shader(npSparseGridDomain sparseGridDomain)
pxg_compile_shader(npDenseGridCell denseGridCell)
pxg_compile_shader(npSbBaryProject sbBaryProject)
pxg_compile_shader(npIsoVecNormalize isoVecNormalize)
pxg_compile_shader(npSbCompliance sbCompliance)
pxg_compile_shader(npSbDeltaLambdaGradC sbDeltaLambdaGradC)
pxg_compile_shader(npSbTetConstraint sbTetConstraint)
pxg_compile_shader(npPsSpecVolume psSpecVolume)
pxg_compile_shader(npSbComputeLambda sbComputeLambda)
pxg_compile_shader(npGridLocation gridLocation)
pxg_compile_shader(npAnisoOutput anisoOutput)
pxg_compile_shader(npSmoothPos smoothPos)
pxg_compile_shader(npIsoAvgVerts isoAvgVerts)
pxg_compile_shader(npPsSolvePC psSolvePC)
pxg_compile_shader(npPsSolvePCTGS psSolvePCTGS)
pxg_compile_shader(npPsWeightKernels psWeightKernels)
pxg_compile_shader(npPsDiffuseUpdate psDiffuseUpdate)
pxg_compile_shader(npPsDiffuseSpawn psDiffuseSpawn)
pxg_compile_shader(npPsDiffuseAvg psDiffuseAvg)
pxg_compile_shader(npPsDiffuseOneWay psDiffuseOneWay)
pxg_compile_shader(npSdfBoxDist sdfBoxDist)
pxg_compile_shader(npSparseGridLookup sparseGridLookup)
pxg_compile_shader(npSparseGridCellFromPos sparseGridCellFromPos)
pxg_compile_shader(npIsoGridFilter isoGridFilter)
pxg_compile_shader(npSbExtractRotAPD sbExtractRotAPD)
pxg_compile_shader(npSbSolveNeoHookean sbSolveNeoHookean)
pxg_compile_shader(npSbSolveCorot sbSolveCorot)
pxg_compile_shader(npSbSolveGM sbSolveGM)
pxg_compile_shader(npClothMembrane clothMembrane)
pxg_compile_shader(npClothBending clothBending)
pxg_compile_shader(npClothSharedEnergy clothSharedEnergy)
pxg_compile_shader(npSbCollisionConstraint sbCollisionConstraint)
pxg_compile_shader(npClothCollisionConstraint clothCollisionConstraint)
pxg_compile_shader(npSdfBvhMorton sdfBvhMorton)
pxg_compile_shader(npSbStress sbStress)
pxg_compile_shader(npSdfHoles sdfHoles)
pxg_compile_shader(npClothTriContact clothTriContact)
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
# RUNTIME_PLAN.md P0 spikes (Vulkan-only: BDA does not lower to HLSL cs_6_0,
# so deliberately absent from PXG_HLSL_ENTRIES; not bit-exactness kernels, so
# absent from PXG_NOCONTRACT_SHADERS).
pxg_compile_shader(npSpikeSubgroup spikeSubgroup)
pxg_compile_shader(npSpikeBda spikeBda)
# RUNTIME_PLAN.md P2.1: warp-32 primitive library battery (emulation path).
pxg_compile_shader(npWarpLib wlShuffle wlBallot)
# P2.1 native-path compile/lowering validation (WaveShuffle/WaveActiveBallot).
pxg_compile_shader(npWarpNative warpNativeProbe)
# RUNTIME_PLAN.md P6: indirect dispatch plumbing.
pxg_compile_shader(npIndirect writeArgs fillGrid)
pxg_compile_shader(npAtomicLib alAtomics)
# RUNTIME_PLAN.md P2: tex3D replacement (Vulkan-only sampled-image path).
pxg_compile_shader(npTex3d tex3dSample)
# RUNTIME_PLAN.md P3 shim proof kernel (nested BDA params).
pxg_compile_shader(npShimAxpy shimAxpy)
# P5 Step 4: PxgBroadPhaseDesc mirror layout proof (BDA).
pxg_compile_shader(npBpDescProbe bpDescProbe)
# RUNTIME_PLAN.md P3 launch-marshalling proof kernel (multi-arg natural alignment).
pxg_compile_shader(npShimMisalign shimMisAdd)

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
add_executable(gpuslang-tests
    main.cpp
    testScan.cpp
    testRadixSort.cpp
    testRadixSortGather.cpp
    testRadixSortDesc.cpp
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
    testNpCvxPlanePolyIndex.cpp
    testNpCvxPlaneRefresh.cpp
    testNpCvxPlaneGen.cpp
    testNpCvxPlaneOut.cpp
    testNpCvxCorrelate.cpp
    testNpCvxPostProcess.cpp
    testNpCvxTrimeshOut.cpp
    testNpTriTriOverlap.cpp
    testNpCvxTriNpSetup.cpp
    testNpCvxHfNpSetup.cpp
    testNpCvxCoreOut.cpp
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
    testNpPsSolvePC.cpp
    testNpPsSolvePCTGS.cpp
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
    testBroadphaseAggSortProjections.cpp
    testP0Subgroup.cpp
    testP0Bda.cpp
    testP1DeviceV2.cpp
    testP1HeapAlloc.cpp
    testNpWarpLib.cpp
    testNpAtomicLib.cpp
    testNpTex3d.cpp
    testWarpNative.cpp
    testP1PipelineCache.cpp
    testP1Timestamps.cpp
    testP6Indirect.cpp
    testP3Shim.cpp
    testP3CudaContext.cpp
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
    testP4Aabb.cpp
    ${GPUSLANG_ROOT}/bridge/PxgVulkanCudaContext.cpp
    ${GPUSLANG_ROOT}/bridge/PxgVulkanKernelAbi.cpp
    # P4: the REAL gpucommon buddy allocator, reused over the Vulkan memory
    # manager (PxgVulkanMemoryManager) — compiles cuda-free against real headers.
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
    ${GPUSLANG_ROOT}/../geomutils/src/GuConvexGeometry.cpp)
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
    ${GPUSLANG_ROOT}/../common/src
    ${GPUSLANG_ROOT}/../lowlevel/api/include
    ${GPUSLANG_ROOT}/../lowlevel/common/include/pipeline ${GPUSLANG_ROOT}/../lowlevel/common/include/utils
    ${GPUSLANG_ROOT}/../lowleveldynamics/include ${GPUSLANG_ROOT}/../lowleveldynamics/shared
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
    ${GPUSLANG_ROOT}/bridge/PxgVulkanCudaContext.cpp
    ${GPUSLANG_ROOT}/bridge/PxgVulkanKernelAbi.cpp
    # P4: the real gpucommon buddy allocator, backing createGpuHeapMemoryAllocatorManager.
    ${GPUSLANG_ROOT}/../gpucommon/src/PxgHeapMemoryAllocator.cpp
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
    ${GPUSLANG_ROOT}/../common/src
    ${GPUSLANG_ROOT}/../lowlevel/api/include
    ${GPUSLANG_ROOT}/../lowlevel/common/include/pipeline ${GPUSLANG_ROOT}/../lowlevel/common/include/utils
    ${GPUSLANG_ROOT}/../lowleveldynamics/include ${GPUSLANG_ROOT}/../lowleveldynamics/shared
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
foreach(suite scan radixSort radixSortDesc radixSortGather preIntegration integration updateBounds memCopy clamp skinning
        scan4x4 gpuRadixSort reorderKernel bodyUpdate directApi setRigidDynamicGlobalPose solverPGS solverTGS
        constraintPrepPGS constraintPrepTGS thresholdStream thresholdForceChange solverZeroBodies solverInitVelocitiesTGS solverApplyGravityTGS solverDmaBack npPairManagement npPairRemoveSwap npSphereCollision npCapsuleCollision npMaterialCombine npContactPatch npInsertPatch npCompressContacts npCompressContacts2 npTriangle npSegTriangle npCapTriangle npTriTri npHeightfield npHfMidphase npHfPointQuery npConvexTri npConvexSat npConvexFaceSat npConvexEdgeSat npConvexFeature npConvexTriOut npConvexPolyClip npCvxTriEmit npCvxTriFind npCvxTriPurge npCvxTriWorldNormal npGjkSupport npGjkNewPoint npGjkBarycentric npHfTriFlags npWarpVec npGjkSimplex npGjkClosestTri npGjkClosestTet npGjkClosestSeg npGjkOutput npGjkPermute npGjkSupportStep npGjkWarmStart npGjk npEpaStart4 npEpaStart123 npEpa npGjkWarm npCvxCvxEE npCvxRayPoly npCvxFaceContacts npCvxReduceManifold npCvxWitnessPoly npCvxLocalNormal npCvxAddManifold npCvxBlob npCvxBlobRead npCvxEEDriver npCvxRefFrame npCvxPolyClip npCvxClipEmit npCvxContactDecide npCvxCollidePrep npCvxCollideDecide npCvxLostFound npCvxWriteStream npCvxFinishContacts npCvxStage1Refresh npCvxStage2Setup npCvxPlanePolyIndex npCvxPlaneRefresh npCvxPlaneGen npCvxPlaneOut npCvxCorrelate npCvxPostProcess npCvxTrimeshOut npTriTriOverlap npCvxTriNpSetup npCvxHfNpSetup npCvxCoreOut npCvxCoreSetup npCvxCoreSupport npCvxCoreFace npCvxCorePtsFace npCvxCoreClip npCvxCoreContact npCvxCoreGjk npCvxCoreEpa npCvxCorePlaneGen npCvxCoreCvxGen npCvxBvhGlue npCvxCoreWarpAcc npD6JointPrep npRemapIndex npContactPrePrep npRigidSumBatches npArtiExtBody npArti1DPrep npArtiJointHeader npArtiSolve1D npArtiSolveContact npArtiSolve1DTGS npArtiSolveContactTGS npArtiCheckActive npSlabRefCount npArtiPropagate npArtiPropagateAccel npArtiPropagateAccelJV npArtiSpatialMatVec npArtiBitfield npArtiSpatialMatMat npArtiConstructSM npArtiTranslateInertia npArtiTranslateSInertia npArtiPropagateZA npArtiPropagateZASph npArtiTranslateInertiaMass npArtiInvertInertia npArtiPropagateImpDof npArtiAccumLinkImp npArtiDriveParams npArtiClampDriveImp npClothDefGrad2D npSbDefGrad npDefMaterial npGridIndex npJacobiRotate npClothAtan npAnisoHelpers npSparseGridHash npClothDeltaLambda npClothExtractRot2D npClothVelClamp npClothArap2D npClothArea2D npClothWind npSbBaryInvMass npSbVolumeSolve npIsoConcaveCorner npSphSpikyKernel npIsoAcosDecode npSparseGridDomain npDenseGridCell npSbBaryProject npIsoVecNormalize npSbCompliance npSbDeltaLambdaGradC npSbTetConstraint npPsSpecVolume npSbComputeLambda npGridLocation npAnisoOutput npSmoothPos npIsoAvgVerts npPsSolvePC npPsSolvePCTGS npPsWeightKernels npPsDiffuseUpdate npPsDiffuseSpawn npPsDiffuseAvg npPsDiffuseOneWay npSdfBoxDist npSparseGridLookup npSparseGridCellFromPos npIsoGridFilter npSbExtractRotAPD npSbSolveNeoHookean npSbSolveCorot npSbSolveGM npClothMembrane npClothBending npClothSharedEnergy npSbCollisionConstraint npClothCollisionConstraint npSdfBvhMorton npSbStress npSdfHoles npClothTriContact npManifold npSegDist npTriMask npNpUtils npConvexReduce npTriMesh npClosest npTetra npEdgeEdge npDefFilter npBv32 npMidphaseCvx npMidphaseDeform npMidphaseAlloc npClothCloth gridCal contactReduce npSbTet npSbSelf npSbSb npTetTriMid psPrimContact psBounds psPrimCollide psDiffuseCollide fcPrim psMeshMid psHfCollide fcPrim2 npPolyClip4 fcConvexTri fcVertexConvex fcStaticTriCloth fcMidphase npSbPrim npSbRemap npCvxHfMid npTriPlaneNp npTriHfNp npCvxMeshNp npCvxHfNp npBoxClip npBoxIncident npBoxContacts npBoxSat npBoxFeature npBoxReduce broadphaseSetup broadphaseReports
        broadphaseIncrementalHistogram broadphaseEndPtHistogram broadphaseIncrementalSap
        broadphaseRegionHistogram broadphaseCreateRegions broadphaseActiveRegionHistogram
        broadphaseOverlapCheckHistogram broadphaseNewBoundsPairs broadphaseMarkPairs
        broadphaseClearNewFlag broadphaseAggregateBounds broadphaseAggregateDirty
        broadphaseAggregatePairsAdd broadphaseAggregateFinalize broadphaseAggregatePairsRemove
        broadphaseAggregateBoxPruning broadphaseAggPairCollision broadphaseAggSelfCollision broadphaseAggSortProjections
        p0Subgroup p0Bda p1DeviceV2 p1HeapAlloc npWarpLib npAtomicLib npTex3d warpNative p1PipelineCache p1Timestamps p6Indirect p3Shim p3CudaContext p3CudaContextManager p3KernelWrangler p3ContextHardening p4Manifest p4Loader p4Foundation p4Heap p4Wrangler p4Broadphase p4Aabb)
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
    "npCvxLostFound:lostFoundStage1"
    "npCvxWriteStream:cvxWriteStream"
    "npCvxFinishContacts:cvxFinishContacts"
    "npCvxStage1Refresh:cvxStage1Refresh"
    "npCvxStage2Setup:cvxStage2Setup"
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
    "npContactPrePrep:cppPack"
    "npRigidSumBatches:rsBatches1"
    "npRigidSumBatches:rsBatches2"
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
    "npPsSolvePC:psSolvePC"
    "npPsSolvePCTGS:psSolvePCTGS"
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
    # markCreatedPairs/initializeRadixRanks/updateHandles: BDA + WorkgroupCount — Vulkan-only (P0 policy).
    "broadphaseSetup:translateAABBsLaunch"
    "broadphaseSetup:initializeSapBox1DLaunch" 
    
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
