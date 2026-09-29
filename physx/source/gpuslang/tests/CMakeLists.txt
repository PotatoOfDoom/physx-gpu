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

# Every shader depends on every module: slangc resolves `import` by file name out
# of SHADER_DIR, so a module edit must rebuild each .spv that imports it.
#
# P9.21 GLOBS the pxgSlang* modules instead of naming them.  The hand-written
# list had drifted to TWENTY missing entries -- pxgSlangBv32, pxgSlangSphereBox,
# pxgSlangConvexTriCloth, pxgSlangGjk, pxgSlangEpa, the articulation trio and the
# rest -- so editing any of them left every importing .spv stale until something
# else forced a recompile.  That is not a cosmetic gap for this tranche in
# particular: P9.21 adds a differential that compares two shaders which compile
# the SAME lifted module, and a stale half would make it compare an old core
# against a new one and call the difference a glue bug.
#
# CONFIGURE_DEPENDS re-globs on build, so a NEW module is picked up without a
# manual cmake run.  The three non-module files below are named individually
# because they are shaders that other shaders import, which no naming
# convention marks.
file(GLOB SHADER_MODULE_DEPS CONFIGURE_DEPENDS ${SHADER_DIR}/pxgSlang*.slang)
set(SHADER_COMMON_DEPS
    ${SHADER_MODULE_DEPS}
    ${SHADER_DIR}/npCvxCollide.slang
    ${SHADER_DIR}/npCvxStage2Core.slang
    ${SHADER_DIR}/npSpherePrimitives.slang)

# Shaders whose SPIR-V gets NoContraction decorations stamped onto every
# float arithmetic op (tools/spv_nocontract.py).  slangc emits none and its
# -fp-mode precise flag is a no-op for SPIR-V, while Vulkan permits drivers
# to contract undecorated expressions: llvmpipe fuses npCvxCoreEpa's
# makePlane triple-cross sums into fma chains (1-ulp vs strict IEEE), which
# breaks the bitwise CPU-vs-GPU doctrine.  The decoration forbids that.
set(PXG_NOCONTRACT_SHADERS softBodySdfMidphaseBda femClothSdfMeshCgBda npCvxCoreEpa npCvxCorePlaneGen npCvxCoreCvxGen npD6JointPrep npArtiExtBody npArti1DPrep npArtiJointHeader npArtiSolve1D npArtiSolveContact npArtiSolve1DTGS npArtiSolveContactTGS npArtiCheckActive npArtiPropagate npArtiPropagateAccel npArtiPropagateAccelJV npArtiSpatialMatVec npArtiSpatialMatMat npArtiTranslateInertia npArtiTranslateSInertia npArtiPropagateZA npArtiPropagateZASph npArtiTranslateInertiaMass npArtiInvertInertia npArtiPropagateImpDof npArtiAccumLinkImp npArtiDriveParams npArtiClampDriveImp npClothDefGrad2D npSbDefGrad npDefMaterial npJacobiRotate npClothAtan npAnisoHelpers npClothDeltaLambda npClothExtractRot2D npClothVelClamp npClothArap2D npClothArea2D npClothWind npSbBaryInvMass npSbVolumeSolve npIsoConcaveCorner npSphSpikyKernel npIsoAcosDecode npSparseGridDomain npDenseGridCell npSbBaryProject npIsoVecNormalize npSbCompliance npSbDeltaLambdaGradC npSbTetConstraint npPsSpecVolume npSbComputeLambda npGridLocation npAnisoOutput npSmoothPos npIsoAvgVerts npPsSolvePC npPsSolvePCTGS npPsWeightKernels npPsDiffuseUpdate npPsDiffuseSpawn npPsDiffuseAvg npPsDiffuseOneWay npSdfBoxDist npSparseGridCellFromPos npIsoGridFilter npSbExtractRotAPD npSbSolveNeoHookean npSbSolveCorot npSbSolveGM npClothMembrane npClothBending npClothSharedEnergy npSbCollisionConstraint npClothCollisionConstraint npSdfBvhMorton npSbStress npSdfHoles npClothTriContact particleSim particleSelfCollisionBda psContactSolveTgsBda psContactSolveRigidBda rigidDeltaAccumBda femClothSimBda femClothSolveBda femClothSolvePairBda femClothDampingBda femClothRigidAttachBda femClothContactGenBda femClothRigidContactBda femClothSelfCollisionBda femClothClothContactBda femClothRefitBda softBodySimBda softBodyRefitBda softBodySolveBda softBodyContactGenBda softBodyContactRemapBda softBodyRigidContactBda softBodyRigidAttachBda psRigidAttachBda directApiBda psShapeMatchBda softBodySelfCollisionBda softBodySelfSolveBda softBodyClothMidphaseBda softBodyClothContactGenBda softBodyClothVertMidphaseBda softBodyClothVertContactGenBda softBodyClothPrepareBda softBodyClothSolveBda softBodySoftBodyMidphaseBda softBodyMeshMidphaseBda softBodyMeshContactGenBda softBodyHfContactGenBda softBodyHfMidphaseBda femClothStepBda femClothDifferentClothBda femClothMeshMidphaseBda femClothMeshContactGenBda femClothMeshVertMidphaseBda femClothPsMidphaseBda softBodyPsMidphaseBda
    softBodyPsContactGenBda softBodyPsPrepareBda softBodyPsSolveBda npDefFilterBda femClothPsContactGenBda femClothPsPrepareBda femClothPsSolveBda femClothAttachBda femClothHfMidphaseBda femClothHfContactGenBda softBodyFemAttachBda softBodyClothAttachBda pcmManifoldBda convexTrimeshMidphaseBda triFeatureBda deferContactBda polyClipBda convexSatBda contactReduceWarpBda cvxTriEmitBda cvxTriSetupBda convexTrimeshCoreBda convexHfMidphaseBda convexHfCoreBda cvxHfSetupBda sphereHfCoreBda hfTriFlagsBda segTriangleBda capTriangleBda capsuleHfCoreBda sphereHfNpBda sphereTriNpBda triPlaneNpBda cvxCorePlaneNpBda cvxCoreCvxNpBda cvxCoreTriNpBda cvxCoreTailBda p86SignedZero cvxCoreTetNpBda cvxCoreClothNpBda cvxTriPostProcessBda cvxTriCorrelateBda cvxTriFinishBda npArtiJointFrame npArtiSpatialInertia npArtiSpatialMatrix npArtiPropagateInertia npArtiDynamic npArtiMassMatrix npArtiLinkAccel npArtiStep npArtiFlushVel npArtiAvgImpulse npArtiStepKernel npArtiSleep npArtiUpdateBodies npArtiUpdateBodies1T npArtiSumBatches npArtiSumBatches2 npArtiSumSelfBatches npArtiSetupInternal npArtiSetupTendons npArtiSetupMimic npArtiSetupICKernel npArtiSolveTendons npArtiSolveTMKernel npArtiSolveExtTGS npArtiSolveExtPGS npArtiSolveSelf npArtiSolveInternal npArtiPropagate2 npArtiConstraintPrep npArtiJointPrep updateArticulationsBda solverTGS npArtiInverseDyn npArtiUpdateKin npArtiGetSet npPsMaterials npPsFluidDensity npPsFluidVelocity npPsSprings npPsAerodynamics npPsInflatable npPsBoundaryDensity psMeshMid psMeshMidphaseBda psHfCollide psHfMidphaseBda psSdfMidphaseBda npSdfTexture npSdfCollision npSdfFind npSdfParticle psDiffuseBuffers psDiffuseCreate psDiffuseAdvect psDiffuseOneWay
    # P9.15d: fcConvexTri joins the list because it and femClothContactGenBda now
    # compile the SAME lifted module (pxgSlangConvexTriCloth).  With only one of the
    # two stamped NoContraction the shared body was contracted differently in each,
    # and the P9.15d differential -- which compares the two kernels bit for bit --
    # measured that instead of the glue it is meant to measure.  The Phase-1 battery
    # compares against a CPU reference built with -ffp-contract=off, so stamping this
    # shader moves it TOWARD that reference, not away.
    fcConvexTri
    # P9.21: fcVertexConvex joins for exactly the same reason -- it and
    # femClothContactGenBda now compile the SAME pxgSlangVertexConvexCloth, and
    # the P9.21 differential compares their contact streams bit for bit.
    fcVertexConvex
    # P9.83: triTriOverlapBda and npTriTriOverlap join TOGETHER, and the pairing is
    # the point.  They compile the same two modules (pxgSlangTriTriOverlap and
    # pxgSlangGjk) since P9.83 lifted the decision out of the Phase-1 shader, so
    # stamping only one would contract the SHARED body differently in each -- the
    # exact fcConvexTri/femClothContactGenBda situation five entries up.  The
    # Phase-1 fixture compares against a CPU mirror built with -ffp-contract=off,
    # so the stamp moves it toward its reference, not away.
    triTriOverlapBda npTriTriOverlap
    # P9.83b-8 (1/n): the geometry fixture.  Its oracle is engine code compiled
    # with -ffp-contract=off and the comparison is bitwise, so quatRotate's
    # mul+add chains must not fuse on the GPU side either.  The mesh-mesh kernel
    # that imports the same module joins this list when it is wired, so the stamp
    # does not certify a build the engine never runs.
    npTriTriGeom
    # P9.83b-8 (2/n): the contact-body fixture, for the same reason.
    npTriTriBody
    # P9.83b-8 (3/n): the gather fixture -- its plane-mode cull is compared bit for bit.
    npTriTriFind
    # P9.83b-8 T1: the five fixtures that certify the patch, block and serial-reduce
    # modules join, TOGETHER, because the builds the engine runs of those modules are
    # stamped.  Their notes said the opposite -- "none of the module's other callers
    # is stamped, so stamping only the fixture certifies a build the engine never
    # runs" -- and that was stale everywhere it appeared: triPlaneNpBda (stamped,
    # above) imports pxgSlangTriPlane, which calls crsContactReduce from
    # pxgSlangContactReduceSerial; the stamped cloth kernels reach the same module
    # through pxgSlangConvexTriCloth; and the mesh-mesh kernel that imports
    # pxgSlangTriTriPatch/Block will be stamped like the Geom, Body and Find
    # fixtures.  Every ENGINE importer is stamped, so an unstamped fixture was the
    # one certifying a build nobody runs.
    npTriTriBlock npTriTriReduce npTriTriAccept npTriTriWrite npCrsWrapper2
    # P9.83b-8 T2: the block-tail fixture -- the header normal and the points are
    # quatRotate chains compared bit for bit.
    npTriTriTail
    # P9.83b-8: the mesh-mesh kernel itself, stamped like every fixture that
    # certifies one of its modules (the promise npTriTriGeom's entry made above).
    triTriCollisionBda
    # P9.89: the particle contact generators.  llvmpipe (Mesa lower_fdot) evaluates an
    # UNDECORATED OpDot as x + (y + z) and a decorated one as (x + y) + z -- measured,
    # 0/65536 misses each way.  Every float dot in this .spv transcribes a PxVec3
    # dot/magnitudeSquared, `x*v.x + y*v.y + z*v.z` = (x + y) + z (PxVec3.h:173/:274),
    # or the three-lane shuffle sum shfl0 + shfl1 + shfl2 of cudaParticleSystem.cu:964
    # and :1117.  Unstamped, the plane/box/sphere/capsule distances, getSpecVolume's
    # magnitude and both convex penetrations came out re-associated.  The importers of
    # the same pxgSlangGjk/Epa/TetCollide modules on the softbody path were already
    # stamped.  testPsContactGen / testPsDiffuseContactGen / testPsConvexContactGen
    # (the p89 cases) compare against -ffp-contract=off mirrors in the CUDA order and
    # fail without this entry.
    psContactGenBda
    # P9.89: the same getSpecVolume in ps_updateBoundFirstPassLaunch
    # (particlesystem.cu:485/:505 -> gridCal.cuh:139) is psBoundsBda's only float OpDot.
    # It sized the CCD particle AABBs as sqrt(x^2 + (y^2 + z^2)).  testPsBounds
    # psBounds_p89SpecVolumeOrder plants order-sensitive block extremes and fails
    # without this entry.
    psBoundsBda
    # P9.89: the leaf of pxgSlangTetCollide, stamped like every engine importer of that
    # module (softBody{,Cloth,Mesh,Hf}ContactGenBda, psContactGenBda), so the leaf
    # certifies the module as the engine compiles it.  computeInnerSphere's face
    # distances are PxVec3 dots, (x + y) + z (softbodyPrimitives.cu:131-154); unstamped
    # they came out x + (y + z).  testNpGjk npTetCollide_p89InnerSphereSelect compares
    # them bit for bit and fails without this entry.
    npTetCollide
    # P9.90: the PGS contact/joint prep.  Every row it writes is PxVec3 dots,
    # (x + y) + z (PxVec3.h:173/:274): cTargetVel, norVel, angNorVel, resp0/1 =
    # angDom * |I^-1/2 (r x n)|^2 + invMass * |n|^2, the friction targetVel and
    # bias (contactConstraintBlockPrep.cuh:414-470/:545-600).  Unstamped, llvmpipe
    # summed them x + (y + z).  The other engine importers of pxgSlangPrepCommon
    # (npArtiConstraintPrep and the npArti* leaves) were already stamped.
    # testConstraintPrep constraintPrepPGS_p90DotOrder compares the rows bitwise
    # and fails without this entry.
    constraintPrepPGS
    # P9.90: the TGS contact/joint prep, the same PxVec3 dots in
    # setupFinalizeSolverConstraintsBlockTGS (contactConstraintBlockPrep.cuh
    # :859-969 normal rows, :1004-1150 friction rows): normalLenSq, norVel0/1,
    # cTargetVel, angNorVel0/1, resp0/1, the kinematic targetVel terms and the friction
    # error/targetVel.  Unstamped, llvmpipe summed them x + (y + z).  It imports
    # pxgSlangPrepCommon like constraintPrepPGS (stamped above), so both now compile
    # the shared module the same way.  testConstraintPrep constraintPrepTGS_p90DotOrder
    # compares the rows bitwise and fails without this entry.
    constraintPrepTGS
    # P9.90: the engine sphere/capsule/plane/box narrow phase.  Every distance it
    # emits is a PxVec3 dot, (x + y) + z (PxVec3.h:173/:274): spheresphere's
    # distanceSq, distancePointSegmentSquared's nom/denom/v.v, the capsule-capsule
    # segment dots and sqDist (sphereCollision.cuh:45/:96-102/:668-686).  Unstamped,
    # llvmpipe summed them x + (y + z).  The harness entries npSphereCollision /
    # npCapsuleCollision compile the same npSpherePrimitives but are compared with a
    # tolerance against their own references, never against this kernel.
    # testSphereNphase sphereNphase_p90DotOrder compares the contact penetrations
    # bitwise and fails without this entry.
    npSphereNphase
    # P9.90: the engine box-box narrow phase.  transform0.transformTranspose(transform1)
    # and newTransformV.transformTranspose(...) are PxVec3 column dots, (x + y) + z
    # (PxVec3.h:173, cudaBox.cu:444/:793-794), and so are the SAT radii and signs
    # (:486/:507/:509), the incident-axis dots (:77), the corner-in-polygon nom (:351)
    # and the reduction's plane/distance/perpendicular dots (:872-905).  Unstamped,
    # llvmpipe summed them x + (y + z).  It imports only pxgSlangCommon; the npBox*
    # primitive shaders transcribe the stages separately and are never compared against
    # this kernel.  testBoxBoxNphase boxBoxNphase_p90DotOrder compares the contact points
    # and penetrations bitwise and fails without this entry.
    boxBoxNphase
    # P9.90: the rigid PGS solve.  Every row projects the body velocities with PxVec3
    # dots, v = lin.dot(n) + ang.dot(raXn), (x + y) + z (PxVec3.h:173,
    # solverBlock.cuh:178/:291/:465/:537/:659); unstamped, llvmpipe summed them
    # x + (y + z).  Its imports (pxgSlangCommon, pxgSlangSolverDesc) carry no shared
    # solve math and no other kernel is compared against it; solverTGS was already
    # stamped.  testSolver solverPGS_p90DotOrder compares the solved velocities, applied
    # forces and residuals bitwise and fails without this entry.
    solverPGS
    # P9.90: the rigid PGS/TGS integration.  The sleep energies
    # sleepAngVelAcc.multiply(..).dot(inertia) + sleepLinVelAcc.magnitudeSquared()
    # are PxVec3 dots, (x + y) + z (integration.cuh:81-82/:164-165/:222-223), and they
    # reach the wake counter through min(energy / threshold, 2); getNormalized of the
    # output quaternion is the 4-lane ((x + y) + z) + w (PxQuat.h:200,
    # AlignedQuat.h:163; integration.cuh:375/:431).  Unstamped, llvmpipe summed them
    # x + (y + z) and x + (y + (z + w)).  Its imports (pxgSlangCommon,
    # pxgSlangSolverDesc) carry no math another kernel is compared against it with.
    # testIntegration integration_p90DotOrder compares the outputs bitwise and fails
    # without this entry.
    integration
    # P9.90: the convex-convex stage-2 kernel.  collide()'s float OpDots transcribe
    # PxVec3 dots, (x + y) + z (PxVec3.h:173): the capsule face-ray
    # intersectRayPolyhedron denominator/distToPlane (cudaGJKEPA.cu:1051-1052), the
    # witness-polygon selection (:647/:682) and the EE fallback (:1150-1185).
    # Unstamped, llvmpipe summed them x + (y + z).  npCvxStage2SetupDbg compiles the
    # SAME npCvxStage2Core/npCvxCollide modules, so it joins too.  testCvxCollide
    # cvxCollide_p90RayDotOrder compares the face-ray tEnter and contact points bitwise
    # and fails without this entry.
    npCvxStage2Setup npCvxStage2SetupDbg
    # P9.90: the PGS particle/rigid contact prep + solve.  The prep's response
    # terms raXnSqrtInertia.dot(raXnSqrtInertia) (particlesystem.cu:1386/:1391-1392)
    # and the solve's normal/tangent velocities linVel0.dot(n) + angVel0.dot(raXn)
    # (:1654-1656) and separation/biasedF dots (:1438/:1453-1454) are float OpDots,
    # (x + y) + z in the CUDA; unstamped, llvmpipe summed them x + (y + z).  Its
    # imports are descriptor layouts only (no shared float code), and the TGS solve
    # that consumes this prep (psContactSolveTgsBda) is already stamped.
    # testPsContactSolve psContactSolve_p90DotOrder compares the constraint block,
    # deltaPos, appliedForces and mAccumDeltaP bitwise and fails without this entry.
    psContactSolveBda
    # P9.90: the engine convex/box-vs-plane narrow phase.  Its float OpDots transcribe
    # PxVec3 dots, (x + y) + z (PxVec3.h:173/:274): the working normal
    # bToA.rotateInv(localNormal).getNormalized() (cudaGJKEPA.cu:2863), the refresh
    # distance v.dot(n) and projectedDifference.magnitudeSquared() (:2905-2909), the
    # face/edge selection and the reduction dots.  Unstamped, llvmpipe summed them
    # x + (y + z).  Its other import, pxgSlangNpPatchTail, carries no dot and its
    # engine importer triPlaneNpBda is already stamped; the npCvxPlane{Refresh,Gen,Out}
    # harness ports are compared against their own references, never this kernel.
    # testConvexPlaneNphase convexPlaneNphase_p90DotOrder compares the manifold normals
    # and the reloaded separations bitwise and fails without this entry.
    npCvxPlaneNphase
    # P9.90: the deformable skinning kernels.  normalizeSafe's magnitude
    # (utility.cu:159/:191/:97) and the Phong projection scales (q - a).dot(nA)
    # (utility.cu:77/:82/:87) are PxVec3 dots, (x + y) + z (PxVec3.h:173/:274);
    # unstamped, llvmpipe summed them x + (y + z).  skinning.slang imports no module.
    # testUtilities skinning_p90DotOrder compares the normals and skinned vertices bitwise.
    skinning)

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
    # P9.86: every shader demands IEEE signed zeros (OpExecutionMode
    # SignedZeroInfNanPreserve), as the CUDA computes -- except the probe that
    # measures what the mode changes.
    if(NOT stem STREQUAL "p86SignedZero")
        list(APPEND postprocess COMMAND python3 ${GPUSLANG_ROOT}/tools/spv_szinp.py ${SHADER_OUT_DIR}/${stem}.spv)
    endif()
    add_custom_command(
        OUTPUT ${SHADER_OUT_DIR}/${stem}.spv
        COMMAND ${PXG_SLANGC} ${SHADER_DIR}/${stem}.slang -target spirv ${entries}
                ${PXG_SLANG_FLAGS} -o ${SHADER_OUT_DIR}/${stem}.spv
        ${postprocess}
        DEPENDS ${SHADER_DIR}/${stem}.slang ${SHADER_COMMON_DEPS} ${GPUSLANG_ROOT}/tools/spv_nocontract.py ${GPUSLANG_ROOT}/tools/spv_szinp.py
        COMMENT "slangc ${stem}.slang -> ${stem}.spv"
        VERBATIM)
    list(APPEND PXG_SHADER_OUTPUTS ${SHADER_OUT_DIR}/${stem}.spv)
    set(PXG_SHADER_OUTPUTS ${PXG_SHADER_OUTPUTS} PARENT_SCOPE)
endfunction()

pxg_compile_shader(scan scanPerBlockKernel addBlockSumsKernel)
pxg_compile_shader(radixSort radixSortHistogram radixSortCalculateRanks)
pxg_compile_shader(radixSort64 radixSortCopyHigh32Bits)
# P5 Step 4: descriptor-based multi-launch radix sort (BDA engine ABI).
# P6.2 adds the WithoutCount twins (numKeys from desc.count) for the grid path.
# P6.4b-1 adds the PLAIN twins (numKeys deref'd from *desc.numKeys) for the PBD
# particle contact-sort loop.  radixSortDesc is BDA-only (reads its descriptor
# through Ptr<>), so it is deliberately absent from PXG_HLSL_ENTRIES (same
# precedent as its WithCount twins / particleSim).
pxg_compile_shader(radixSortDesc radixSortMultiBlockLaunchWithCount radixSortMultiCalculateRanksLaunchWithCount
                   radixSortMultiBlockLaunchWithoutCount radixSortMultiCalculateRanksLaunchWithoutCount
                   radixSortMultiBlockLaunch radixSortMultiCalculateRanksLaunch)
pxg_compile_shader(radixSortGather radixSortCopy radixSortCopy2 radixSortCopyBits2 radixSortDoubleCopy radixSortDoubleCopyHigh32Bits)
pxg_compile_shader(preIntegration preIntegrationLaunch initStaticKinematics preIntegrationLaunchTGS)
pxg_compile_shader(integration integrateCoreParallelLaunch integrateCoreParallelLaunchTGS)
pxg_compile_shader(updateBounds updateTransformCacheAndBoundArrayLaunch setRigidDynamicGlobalPose updateChangedAABBMgrHandlesLaunch
                   mergeChangedAABBMgrHandlesLaunch)
# P9.79: the DIRECT-GPU API on the engine launch ABI.  directApi.slang keeps the
# Phase-1 bound-buffer ports and testDirectApi; these are what the engine gets.
# P9.80: the PBD particle-VOLUME bounds -- the other half of the PxParticleRigidBuffer
# that PXG_SCENE=particlerigid is the first scene to create.
pxg_compile_shader(psVolumeBoundBda ps_update_volume_bound)
pxg_compile_shader(psShapeMatchBda ps_solveShapes)
pxg_compile_shader(directApiBda getRigidDynamicGlobalPose getRigidDynamicLinearVelocity
                   getRigidDynamicAngularVelocity getD6JointForces getD6JointTorques
                   radixSortCopy2 radixSortCopyBits2 updateBodiesLaunchDirectAPI)
# P9.42: the frozen/unfrozen shape compaction (updateTransformAndBoundArray.cu:245-428)
pxg_compile_shader(frozenHistogram computeFrozenAndUnfrozenHistogramLaunch outputFrozenAndUnfrozenHistogram
                   createFrozenAndUnfrozenArray)
pxg_compile_shader(bodyUpdate updateBodiesLaunch updateBodiesLaunchDirectAPI updateShapesLaunch
                   updateBodyExternalVelocitiesLaunch updateJointsLaunch)
pxg_compile_shader(directApi getRigidDynamicGlobalPose getRigidDynamicLinearVelocity
                   getRigidDynamicAngularVelocity getRigidDynamicLinearAcceleration
                   getRigidDynamicAngularAcceleration setRigidDynamicLinearVelocity
                   setRigidDynamicAngularVelocity setRigidDynamicForce setRigidDynamicTorque)
pxg_compile_shader(scan4x4 scanPerBlockKernel4x4 addBlockSumsKernel4x4)
pxg_compile_shader(radixFourBit radixFourBitCountPerBlockKernel radixFourBitReorderKernel)
pxg_compile_shader(reorderData reorderKernel)
pxg_compile_shader(solverPGS solveBlockPartition concludeBlocks writebackBlocks writeBackBodies
                   computeAverageSolverBodyVelocity propagateSolverBodyVelocity markActiveSlabPGS
                   solveStaticBlock propagateStaticSolverBodyVelocities)
pxg_compile_shader(solverTGS solveBlockPartitionTGS concludeBlocksTGS writebackBlocksTGS
                   writeBackBodiesTGS computeAverageSolverBodyVelocityTGS
                   propagateAverageSolverBodyVelocityTGS solveWholeIslandTGS
                   solveStaticBlockTGS propagateStaticSolverBodyVelocitiesTGS
                   solveBlockUnified markActiveSlabTGS)
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
# P8.6: the GJK-then-EPA driver (tetPrimitivesCollide2) that sits on top of both, plus
# computeInnerSphere.  Its leaf test pins the driver's POLICY -- which result
# short-circuits, when EPA is entered, and the degenerate re-run -- while ctGjk and ctEpa
# pin the arithmetic underneath it.
# P9.89: ctInnerSphere, computeInnerSphere alone (npTetCollide_p89InnerSphereSelect).
pxg_compile_shader(npTetCollide ctTetCollide ctInnerSphere)
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
pxg_compile_shader(npCvxLostFound prepareLostFoundPairs_Stage1 prepareLostFoundPairs_Stage2)
pxg_compile_shader(npCvxWriteStream cvxWriteStream)
pxg_compile_shader(npCvxFinishContacts finishContactsKernel)
pxg_compile_shader(npCvxStage1Refresh convexConvexNphase_stage1Kernel)
pxg_compile_shader(npCvxStage2Setup convexConvexNphase_stage2Kernel)
pxg_compile_shader(npCvxStage2SetupDbg convexConvexNphase_stage2KernelDebug)
pxg_compile_shader(npCvxPlanePolyIndex cvxPlanePolyIndex)
pxg_compile_shader(npCvxPlaneRefresh cvxPlaneRefresh)
pxg_compile_shader(npCvxPlaneGen cvxPlaneGen)
pxg_compile_shader(npCvxPlaneOut cvxPlaneOut)
pxg_compile_shader(npCvxCorrelate cvxCorrelate)
pxg_compile_shader(npCvxPostProcess cvxPostProcess)
pxg_compile_shader(npCvxTrimeshOut cvxTrimeshOut)
pxg_compile_shader(npTriTriOverlap triTriOverlap)
# P9.83b-2: the fixture entry for pxgSlangContactReduceSerial's TWO wrappers over
# one body.  It runs both on the same lane arrays so the test can assert they
# DISAGREE -- a fixture that only checked wrapper 2 against its own mirror would
# pass with both implemented identically, which is what the four pre-existing
# transcriptions of contactReduceShared all were.
pxg_compile_shader(npCrsWrapper2 crsWrapperPair)
pxg_compile_shader(npTriTriReduce ttrReduce)
pxg_compile_shader(npTriTriAccept ttrAccept ttrReserve)
pxg_compile_shader(npTriTriWrite ttrScan ttrWrite)
pxg_compile_shader(npTriTriBlock ttbProbe ttbReduceProbe ttbAssignProbe)
# P9.83b-8 (1/n): pxgSlangTriTriGeom -- subdivision, getTriangleVertices with a
# sub-index, the two radii, PxMax, and the two PxTransform inverses.  Serial lanes;
# the host oracle is the engine's own GuTriangleRefinement.h / PxTransform.h.
pxg_compile_shader(npTriTriGeom ttgSubdivProbe ttgVertsProbe ttgMathProbe)
# P9.83b-8 (2/n): pxgSlangTriTriBody -- the per-thread contact body of both passes,
# split around the (already certified) SDF call into texture-free halves.
pxg_compile_shader(npTriTriBody ttcVertsProbe ttcPostProbe)
# P9.83b-8 (3/n): pxgSlangTriTriFind -- the subdivision-aware gather, the refinement
# buffer, and the kernel's inner-loop control, one 1024-thread workgroup per case.
pxg_compile_shader(npTriTriFind ttfRounds ttfCullProbe)
# P9.83b-8 T2: pxgSlangTriTriTail -- writePatchesAndContacts and the block-wide
# reservation as one block function, against real counters, records and streams.
pxg_compile_shader(npTriTriTail tttTail)
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
# P9.87: the two engine emission tails (pxgSlangCvxCoreOut / Out32) run on raw,
# poisoned streams -- the engine's own text, #included; BDA-only (not in the HLSL list).
pxg_compile_shader(cvxCoreTailBda cctSingle cctMulti)
# P9.86: the sign-of-zero probe, built twice -- as-is and with SignedZeroInfNanPreserve.
pxg_compile_shader(p86SignedZero szProbe tbProbe fcProbe)
add_custom_command(
    OUTPUT ${SHADER_OUT_DIR}/p86SignedZeroSz.spv
    COMMAND python3 ${GPUSLANG_ROOT}/tools/spv_szinp.py ${SHADER_OUT_DIR}/p86SignedZero.spv ${SHADER_OUT_DIR}/p86SignedZeroSz.spv
    DEPENDS ${SHADER_OUT_DIR}/p86SignedZero.spv ${GPUSLANG_ROOT}/tools/spv_szinp.py
    COMMENT "spv_szinp p86SignedZero.spv -> p86SignedZeroSz.spv"
    VERBATIM)
list(APPEND PXG_SHADER_OUTPUTS ${SHADER_OUT_DIR}/p86SignedZeroSz.spv)
pxg_compile_shader(npD6JointPrep d6jPrep c1dPrePrep constraint1DPrePrepLaunch)
pxg_compile_shader(npRemapIndex remapIndex)
pxg_compile_shader(npContactPrePrep cppPack constraintContactBlockPrePrepLaunch
                   constraint1DBlockPrePrepLaunch clearFrictionPatchCounts)
pxg_compile_shader(npRigidSumBatches rigidSumInternalContactAndJointBatches1
                   rigidSumInternalContactAndJointBatches2)
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
# P6.0: PBD particle free-fall bring-up — the first three particle orchestration
# kernels on the engine ABI (BDA).  Vulkan-only (BDA reads the PxgParticleSystem
# descriptor array through Ptr<>, which does not lower to HLSL cs_6_0), so
# deliberately absent from PXG_HLSL_ENTRIES (same precedent as solveBlockUnified).
pxg_compile_shader(particleSim ps_updateUnsortedArrayLaunch ps_preIntegrateLaunch ps_integrateLaunch ps_updateUserBufferLaunch ps_preDiffuseIntegrateLaunch)
# P6.1: particle contact-sort bring-up — the three contact-sort orchestration
# kernels on the engine ABI (BDA), re-pointed off the Phase-1 harness
# (radixSortGather / clampValues).  The radix double-copies grid-stride via
# WorkgroupCount().x, which has no HLSL cs_6_0 lowering, so this shader is
# deliberately absent from PXG_HLSL_ENTRIES (same precedent as particleSim /
# solveBlockUnified).
pxg_compile_shader(particleContactSortBda radixSortDoubleCopy radixSortDoubleCopyHigh32Bits clampMaxValue)
# P8.20: the SINGLE-gather siblings of the three above — the contact-sort kernels
# PxgSoftBodyCore::sortContacts() and PxgFEMClothCore::sortContacts() launch, also
# re-pointed off the Phase-1 harness (radixSort64 / radixSortGather / clampValues),
# which the engine-ABI dispatcher cannot drive.  Absent from PXG_HLSL_ENTRIES for
# the same reason as particleContactSortBda: the grid-stride reads WorkgroupCount().x,
# which has no HLSL cs_6_0 lowering, and every buffer comes through Ptr<>.  Absent
# from PXG_NOCONTRACT_SHADERS as well — a rank-indexed load/store gather and three
# integer compare-and-stores contain NO float arithmetic, so there is no contraction
# for the stamp to forbid (same reasoning as fem_reorderRigidContactsLaunch).
pxg_compile_shader(femContactSortBda radixSortCopyHigh32Bits radixSortCopy clampMaxValues)
# P8.23: the PCM manifold GATE of midphaseGeneratePairs (convexMeshMidphase.cu:250-263),
# extracted ahead of the traversal half because it is the only STATEFUL part of that
# kernel and the only part a single-dispatch fixture cannot certify.  Not an engine
# kernel and deliberately absent from the manifest -- convexTrimeshMidphaseBda will
# import pxgSlangPcmManifold and inline the same sequence.  Absent from
# PXG_HLSL_ENTRIES (Ptr<> everywhere, groupshared ballot).  IS in
# PXG_NOCONTRACT_SHADERS: refreshManifolds is dot-of-difference chains
# (v.dot(normal), projectedDifference.magnitudeSquared()) compared against a squared
# threshold, exactly the mul-feeding-add shape contraction would move.
pxg_compile_shader(pcmManifoldBda pcmInvalidateLaunch pcmRefreshLaunch)
# P8.26: midphaseGeneratePairs (convexMeshMidphase.cu:378) on the engine ABI --
# the first of the six kernels behind rigid-vs-cooked-triangle-mesh collision.
# Absent from PXG_HLSL_ENTRIES (Ptr<> everywhere, groupshared ballot).  IS in
# PXG_NOCONTRACT_SHADERS, as every other mesh midphase in this port is: the SAT's
# AXISTESTs and the plane distance are mul-feeding-add chains.
pxg_compile_shader(convexTrimeshMidphaseBda midphaseGeneratePairs)
# P9.51a: convexHeightFieldMidphase (convexHFMidphase.cu:180-330).  IS in
# PXG_NOCONTRACT_SHADERS -- ADDED IN P9.51b, and its absence until then was a
# defect, not a scope decision.  This shader's own header names the hazard: it
# converts a world bound into heightfield units through quaternion rotations and
# scaled basis columns, then feeds the result to floor()/ceil() to pick the
# row/column rectangle.  Those are mul-feeding-add chains, so fusing one to an FMA
# moves a bound by an ULP -- which is enough to move a floor() and therefore to
# change the emitted pair SET, not merely its arithmetic.  pxgSlangHeightfield's
# header states the rule ("every importer of this module belongs in
# PXG_NOCONTRACT_SHADERS"); P9.51a wired the shader without applying it.
pxg_compile_shader(convexHfMidphaseBda convexHeightFieldMidphase)
# P8.29: midphaseAllocate's layout arithmetic (midphaseAllocate.cuh:129), the
# sub-buffer carve-up every kernel of the convex-trimesh chain re-derives.  Test
# scaffolding, not an engine kernel, and absent from the manifest.  Pure integer
# address arithmetic, so deliberately absent from PXG_NOCONTRACT_SHADERS too.
pxg_compile_shader(midphaseAllocBda midphaseAllocLaunch)
# P8.30: the second pass's triangle-feature classifier (convexTriangle.cuh:1562).
# Test scaffolding, absent from the manifest.  IS in PXG_NOCONTRACT_SHADERS: the
# barycentric form is a dot-of-cross chain and the classifier compares sums of
# those against a fixed threshold, so contraction moves the classification of
# near-boundary contacts.
pxg_compile_shader(triFeatureBda triFeatureLaunch)
# P8.32: delayGenerateContact (convexTriangle.cuh:1402), the predicate deciding
# whether a convex-triangle pair is emitted now or deferred to the second pass.
# Test scaffolding, absent from the manifest.  IS in PXG_NOCONTRACT_SHADERS: the
# predicate is a dot product compared against a fixed cosine, exactly where
# contraction moves a near-threshold pair across the bar.
pxg_compile_shader(deferContactBda deferContactLaunch)
# P8.35: initializeManifolds (pairManagement.cu:150) on the engine ABI -- one of
# the two launches still refused in EVERY scene after P8.20.  Absent from
# PXG_HLSL_ENTRIES (Ptr<> plus WorkgroupCount()).  Absent from
# PXG_NOCONTRACT_SHADERS: a replicate-copy of float4s does no arithmetic on them.
pxg_compile_shader(initManifoldsBda initializeManifolds)
# P8.37: the whole removeContactManagers chain (pairManagement.cu:45-145) on the
# engine ABI -- the last refused launch plus the four unmapped stages that build
# the run-sum it reads.  Absent from PXG_HLSL_ENTRIES (Ptr<> everywhere plus
# WorkgroupCount()).  Absent from PXG_NOCONTRACT_SHADERS: the chain is an integer
# prefix scan, a binary search and byte copies -- there is no float arithmetic for
# the stamp to constrain (the PxReal rest distances move as raw words).
# P8.38: polyClip + addContacts (convexTriangle.cuh:918-1058), the per-triangle
# core of rigid convex-vs-mesh contact generation, as a 32-lane virtual warp in the
# pxgSlangPolyClip module.  Test scaffolding, absent from the manifest.  Absent
# from PXG_HLSL_ENTRIES (Ptr<> plus groupshared ballots).  IS in
# PXG_NOCONTRACT_SHADERS: every classification in the clipper is the SIGN of a dot
# or a cross-then-dot, which is exactly where contraction moves a vertex across an
# edge and changes the contact set.
pxg_compile_shader(polyClipBda polyClipLaunch)
# P8.39: the cooked convex-hull blob reader (pxgSlangConvexHull), the address
# arithmetic every remaining kernel of the convex-vs-mesh chain re-derives.  Test
# scaffolding, absent from the manifest.  Absent from PXG_HLSL_ENTRIES (Ptr<>
# everywhere).  Absent from PXG_NOCONTRACT_SHADERS: it is integer address
# arithmetic and raw loads, with no float arithmetic to contract.
pxg_compile_shader(convexHullProbeBda convexHullProbeLaunch)
# P8.40: the first two convex-vs-triangle SAT stages.  Test scaffolding, absent
# from the manifest.  Absent from PXG_HLSL_ENTRIES.  IS in PXG_NOCONTRACT_SHADERS:
# every SAT decision is the sign of a projection difference.  P9.89 adds
# convexSupportLaunch, the stage-3 support folds read out per lane.
pxg_compile_shader(convexSatBda convexSatLaunch convexSelectLaunch convexSupportLaunch)
# P8.44: contactReduceShared (contactReduction.cuh:55-165) as a 32-lane virtual
# warp, at the <false,true,5,true> instantiation the convex-vs-triangle emit tail
# reaches.  Test scaffolding, absent from the manifest.  Absent from
# PXG_HLSL_ENTRIES (Ptr<> plus groupshared shuffles).  IS in
# PXG_NOCONTRACT_SHADERS: every anchor pick is an argmax over a dot or a
# cross-then-dot, which is exactly where contraction moves the winner one lane.
# Its test lives in testContactReduce.cpp beside the scalar port's, so both run
# under the existing gpuslang.contactReduce suite against the same oracle.
pxg_compile_shader(contactReduceWarpBda contactReduceWarpLaunch contactReduceWarpCluster6Launch)
# P8.45: the TAIL of convexTriangleContactGen (convexTriangle.cuh:1769-1881) --
# the reduction, the temp-stream reservation and the five per-pair records the
# rest of the convex-vs-trimesh chain reads.  Test scaffolding, absent from the
# manifest.  Absent from PXG_HLSL_ENTRIES (Ptr<> plus groupshared shuffles and
# global atomics).  IS in PXG_NOCONTRACT_SHADERS: the second-pass classifier is a
# chain of cross-then-dot signed areas compared against 0.97, which is exactly
# where contraction moves a contact from one Voronoi region to the next.
pxg_compile_shader(cvxTriEmitBda cvxTriEmitLaunch)
# P8.46 prerequisite: an atomic read-modify-write through a buffer device address
# that points into PINNED HOST memory.  Test scaffolding, absent from the manifest.
# Absent from PXG_HLSL_ENTRIES (Ptr<>) and from PXG_NOCONTRACT_SHADERS (integer
# only, nothing to contract).
pxg_compile_shader(pinnedAtomicBda pinnedAtomicLaunch)
# P8.46b: convexTrimeshNarrowphaseCore's prologue (convexMesh.cu:99-210) -- the
# shape/transform resolve, the cooked-mesh walk down to the adjacency and
# face-remap sections, and the triangle in convex-local space.  Test scaffolding,
# absent from the manifest.  Absent from PXG_HLSL_ENTRIES (Ptr<> everywhere).  IS
# in PXG_NOCONTRACT_SHADERS: the triangle normal is a cross-then-normalize of
# transformed differences, and the winding it fixes decides every contact sign.
pxg_compile_shader(cvxTriSetupBda cvxTriSetupLaunch)
# P8.46c: convexTrimeshNarrowphase (convexMesh.cu:621-759) on the engine BDA ABI,
# composed from the six modules the preceding tranches verified one at a time.
# Absent from PXG_HLSL_ENTRIES (Ptr<>, groupshared warp emulation, global
# atomics).  IS in PXG_NOCONTRACT_SHADERS: it inherits every SAT projection, every
# clipper classification and the barycentric region tests.
pxg_compile_shader(convexTrimeshCoreBda convexTrimeshNarrowphase)
# P9.51b: convexHeightfieldNarrowphase (convexHeightfield.cu:271-357) on the same
# engine BDA ABI -- the trimesh core with the triangle FETCH replaced (a sample
# grid instead of a cooked index buffer), the faceRemap dropped (a heightfield has
# none, so the .cu passes triangleIdx twice at :266) and the slot swap ABSENT (:211
# and :214 read slot 0 as the convex unconditionally).  Shares kCvxTriCore: the two
# kernels declare the same 21 parameters in the same order, so the push-constant
# blob is byte-identical.  Absent from PXG_HLSL_ENTRIES for the same reasons as its
# sibling.  IS in PXG_NOCONTRACT_SHADERS: it inherits every SAT projection, every
# clipper classification and the barycentric region tests.
pxg_compile_shader(convexHfCoreBda convexHeightfieldNarrowphase)
# P9.51b: test scaffolding for pxgSlangCvxHfSetup -- the heightfield twin of
# cvxTriSetupBda, and the ONLY place the heightfield triangle fetch is driven
# against a host mirror.  convexHfCoreBda has no unit fixture (nor does its trimesh
# sibling: both are compositions of separately verified modules), so the fetch was
# put in a module and this probe drives it, rather than inlined in the core where
# a fixture could only have tested a copy of it -- the P9.15d lesson.  Absent from
# the manifest and from PXG_HLSL_ENTRIES (Ptr<> everywhere).  IS in
# PXG_NOCONTRACT_SHADERS: the triangle normal is a cross-then-normalize of
# transformed differences, exactly as its trimesh twin.
pxg_compile_shader(cvxHfSetupBda cvxHfSetupLaunch)
# P9.51c: convexHeightfieldPostProcess (convexMeshPostProcess.cu:585-620) on the
# engine BDA ABI -- the last kernel of the rigid-vs-heightfield chain, and the one
# that stops duplicate contacts along the grid's shared edges and vertices.  Shares
# kCvxTriPostProcess: the same eight pointers in the same order as its trimesh
# sibling.  The engine launches this SAME kernel id for the sphere-vs-heightfield
# path too (PxgNarrowphaseCore.cpp:2367 and :2975), so mapping it serves both.
# Absent from PXG_HLSL_ENTRIES (Ptr<> everywhere).  Absent from
# PXG_NOCONTRACT_SHADERS, and deliberately: the whole kernel is integer index
# arithmetic, bit masks and a binary search -- the only floats it touches are the
# three scale components, and only their SIGNS are read.  There is no contraction
# for the stamp to forbid (the same reasoning as femContactSortBda).
pxg_compile_shader(convexHfPostProcessBda convexHeightfieldPostProcess)
# P9.52a: test scaffolding for pxgSlangSphereHfCore -- the SPHERE arm of
# sphereHeightfieldNarrowphase (convexHeightfield.cu:359-562) on the engine BDA
# ABI.  ONE THREAD PER PAIR, which is the shape the CUDA core has: unlike the
# convex path there is no SAT and no clipper to spread over 32 lanes.  The four
# record arrays are real buffers at their engine strides, so the fixture checks
# WHERE each write landed as well as what it computed.  The kernel that will call
# this module stays absent from the manifest until the CAPSULE arm
# (capsuleHeightfieldNarrowphaseCore, :636-789) lands too -- a kernel that handles
# spheres and drops capsules in silence is the P9.27/P9.40 defect.  Absent from
# PXG_HLSL_ENTRIES (Ptr<> everywhere).  IS in PXG_NOCONTRACT_SHADERS: the
# barycentric region tests, the two normalizations and the plane-side dot products
# are exactly the arithmetic the stamp exists for.
pxg_compile_shader(sphereHfCoreBda sphereHfCoreLaunch)
# P9.52b-1: test scaffolding for pxgSlangHfTriFlags -- computeTriangleFlags
# (convexHeightfield.cu:565-614) reading the SAMPLE GRID rather than pre-decoded
# neighbours, plus selectNormal (capsuleTriangle.cuh:38-98), its only consumer.
# The Phase-1 npHfTriFlags verifies the classification against supplied triangles;
# what changes on the engine ABI is exactly the part it does not cover, so this
# probe drives fetch, normal, classification and selection as one path.  Absent
# from the manifest and from PXG_HLSL_ENTRIES (Ptr<> everywhere).  IS in
# PXG_NOCONTRACT_SHADERS: the fold test is a normalize and two dot products whose
# sign and 0.997 comparison decide a branch.
pxg_compile_shader(hfTriFlagsBda hfTriFlagsLaunch)
# P9.52b-2: test scaffolding for pxgSlangSegTriangle -- distanceSegmentSegmentSquared
# (distanceSegmentSegment.cuh:33-88) and distanceSegmentTriangleSquared
# (capsuleTriangle.cuh:117-327), the geometric core every capsule-vs-triangle path
# in the engine calls.  TWO entry points, because the inner function's parallel and
# degenerate arms are reachable only by accident through the outer one.  Absent from
# the manifest and from PXG_HLSL_ENTRIES (Ptr<>).  IS in PXG_NOCONTRACT_SHADERS:
# the whole module is barycentric arithmetic feeding strict comparisons.
pxg_compile_shader(segTriangleBda segTriLaunch segSegLaunch)
# P9.52b-3: test scaffolding for pxgSlangCapTriangle -- generateContacts
# (capsuleTriangle.cuh:329-406), generateEE (:408-461) and generateEEContacts
# (:463-474), the two contact generators the capsule cores call back to back.
# THREE entry points: each generator alone, plus the composition the cores run, so
# the `numContacts < 2` guards are reachable without first arranging a face
# contact.  Absent from the manifest and from PXG_HLSL_ENTRIES (Ptr<>).  IS in
# PXG_NOCONTRACT_SHADERS: every emit decision is a comparison on a dot product.
pxg_compile_shader(capTriangleBda capFaceLaunch capEELaunch capBothLaunch)
# P9.52b-4: test scaffolding for pxgSlangCapsuleHfCore -- the capsule arm of
# capsuleHeightfieldNarrowphaseCore (convexHeightfield.cu:636-789), the composition
# of the heightfield decoders, the triangle-flag classifier, the segment/triangle
# distance core and the two contact generators on the engine's BDA ABI.  ONE thread
# per pair, the shape the CUDA core has, so the fixture checks WHERE the tail's
# writes land (pairOffset vs pairOffsetPadded) as well as what they hold.  Absent
# from the manifest and from PXG_HLSL_ENTRIES (Ptr<>).  IS in
# PXG_NOCONTRACT_SHADERS: the whole chain is dot products feeding strict
# comparisons, and the capsule axis is a getBasisVector0 whose x component differs
# term-for-term from the generic rotate.
pxg_compile_shader(capsuleHfCoreBda capsuleHfCoreLaunch)
# P9.52b-6: sphereHeightfieldNarrowphase (convexHeightfield.cu:791-996) on the
# engine BDA ABI -- the kernel that makes P9.52a..b-4 reachable from a scene.  It
# dispatches on slot 0's geometry type into the sphere core or the capsule core and
# then runs the warp output stage both arms share.  ONE THREAD per pair, so the
# workgroup is exactly one warp: pxgSlangWarpInc's emulation path routes every
# primitive through groupshared memory and therefore needs WORKGROUP-uniform call
# sites, and the output stage's guards are only warp-uniform.  Shares kCvxTriCore:
# the same 21 parameters in the same order as both convex cores.  Absent from
# PXG_HLSL_ENTRIES (Ptr<> everywhere).  IS in PXG_NOCONTRACT_SHADERS: it inherits
# every comparison in both cores.
pxg_compile_shader(sphereHfNpBda sphereHeightfieldNarrowphase)
# P9.66b: the trimesh twin.  Same 21-argument ABI, same one-warp group, but the
# sphere arm REMAPS the triangle index and the capsule arm does not.
pxg_compile_shader(sphereTriNpBda sphereTrimeshNarrowphase)
# P9.67c: convexCorePlaneNphase_Kernel (convexCoreCollision.cu:449-494) on the
# engine BDA ABI.  One thread per test, no grid-stride loop (the .cu has none).
# Its own 136-byte ABI row -- NOT the convex-plane 144, which carries a manifold
# pointer and a tolerance length this kernel does not take.
# P9.82: trimeshPlaneNarrowphase on the engine ABI -- ASSEMBLY of the modules
# pxgSlangTriPlane (the Phase-1 decision core), pxgSlangPcmManifold,
# pxgSlangNpPatchTail and pxgSlangBv32.  IS in PXG_NOCONTRACT_SHADERS: the
# plane-space projection and the reduction's anchor picks are mul-feeding-add
# chains whose SIGN decides which contacts survive.
pxg_compile_shader(triPlaneNpBda trimeshPlaneNarrowphase)
# P9.83: triangleTriangleOverlaps on the engine ABI -- ASSEMBLY of
# pxgSlangTriTriOverlap (the Phase-1 decision, lifted) and pxgSlangGjk (the
# cold-start driver every deformable contact gen already ran).  One WARP per pair
# with the grid sized from the block, so [numthreads] is the host's (32,2,1)
# verbatim -- see the P9.68 note in the shader.  IS in PXG_NOCONTRACT_SHADERS.
pxg_compile_shader(triTriOverlapBda triangleTriangleOverlaps)
# P9.83b-8: triangleTriangleCollision -- the SDF mesh-mesh contact generator, one
# 1024-thread workgroup per pair, ASSEMBLY of the certified tri-tri modules
# (Geom, Body, Find, Block, Patch, Tail) around the SDF descent.  Bindless (set 1)
# through sdfDistance / sdfDoTriangleCollision.  IS in PXG_NOCONTRACT_SHADERS.
pxg_compile_shader(triTriCollisionBda triangleTriangleCollision)
pxg_compile_shader(cvxCorePlaneNpBda convexCorePlaneNphase_Kernel)
pxg_compile_shader(cvxCoreCvxNpBda convexCoreConvexNphase_Kernel)
pxg_compile_shader(cvxCoreTriNpBda convexCoreTrimeshNphase_Kernel32)
pxg_compile_shader(cvxCoreTetNpBda convexCoreTetmeshNphase_Kernel32)
pxg_compile_shader(cvxCoreClothNpBda convexCoreClothmeshNphase_Kernel32)
# P8.47: radixSortSingleWarpKeysOnly (RadixSort.cuh:579-720) as a 32-lane virtual
# warp -- the one piece of the convex-vs-trimesh chain with no Phase-1 Slang to
# lean on.  Test scaffolding, absent from the manifest.  Absent from
# PXG_HLSL_ENTRIES (Ptr<> plus groupshared warp emulation) and from
# PXG_NOCONTRACT_SHADERS (integer only -- there is no float arithmetic to
# constrain).
pxg_compile_shader(radixWarpBda radixWarpLaunch)
# P8.48: sortTriangleIndices (convexMeshPostProcess.cu:623) on the engine BDA ABI
# -- a thin wrapper over the P8.47 radix sort.  Absent from PXG_HLSL_ENTRIES
# (Ptr<> plus groupshared warp emulation) and from PXG_NOCONTRACT_SHADERS
# (integer only).
pxg_compile_shader(sortTriIndicesBda sortTriangleIndices)
pxg_compile_shader(cvxTriPostProcessBda convexTrimeshPostProcess)
pxg_compile_shader(cvxTriCorrelateBda convexTrimeshCorrelate)
pxg_compile_shader(cvxTriFinishBda convexTrimeshFinishContacts)
# P8.65: newArticulationsLaunch (updateBodiesAndShapes.cu:228) on the engine BDA
# ABI -- the kernel that fills mArticulationPool, without which every consumer of
# the pool reads uninitialised device pointers (P8.64).  Ptr<> throughout, so
# absent from PXG_HLSL_ENTRIES; integer/copy only, so absent from
# PXG_NOCONTRACT_SHADERS.
pxg_compile_shader(newArticulationsBda newArticulationsLaunch)
pxg_compile_shader(updateArticulationsBda updateArticulationsLaunch)
# P8.66: dmaArticulationResidual (forwardDynamic2.cu:3766) -- one thread per
# articulation, and the kernel that forces the PxgArticulationCoreDesc address
# map the rest of the articulation block needs.  Ptr<> throughout (no HLSL
# lowering); integer/copy only (no float contraction to pin).
pxg_compile_shader(dmaArtiResidualBda dmaArticulationResidual artiCoreDescProbe)
# P8.67: the articulation BLOCK-LAYOUT address map, proven functionally.  The
# probe reads element [1] of a two-element array at lane 3, so struct stride,
# field offset and lane stride are each independently wrong-able.
pxg_compile_shader(npArtiBlockProbe artiBlockProbe)
# P8.69: the joint-frame math at the head of jcalc.  Bit-exact float work, so it
# MUST be in PXG_NOCONTRACT_SHADERS below -- a fused multiply-add anywhere in
# rotateAndNormalize puts the result off the engine's bit pattern.
pxg_compile_shader(npArtiJointFrame artiJointFrame artiJcalcPhaseA artiJcalcPhaseB1 artiJcalcPhaseB2 artiJcalcPhaseB3 artiJcalcPhaseC artiJcalcTendons artiJcalcGpuDirty artiJcalcInit artiJcalcLinkVel computeUnconstrainedVelocities1TLaunch)
# P8.84: computeSpatialInertiaW (forwardDynamic2.cu:1114) -- the per-link spatial
# inertia, Coriolis term and bias force.  MUST be in PXG_NOCONTRACT_SHADERS: it is
# compared bit-for-bit against the engine's own transformInertiaTensor and the
# PxMat33/PxQuat arithmetic on the host, and a fused multiply-add anywhere in it
# moves the result off that bit pattern.
pxg_compile_shader(npArtiSpatialInertia artiSpatialInertiaW computeUnconstrainedSpatialInertiaLaunchPartial1T)
# P8.86: the spatial-matrix algebra layer (pxgSlangSpatialMatrix.slang), driven
# by a table of opcodes.  MUST be in PXG_NOCONTRACT_SHADERS: every operation is
# compared bit-for-bit against the engine's own PxMat33 / Dy::SpatialMatrix
# arithmetic on the host.
pxg_compile_shader(npArtiSpatialMatrix artiSpatialMatrixOps artiSpatialMatrixBlockIo)
# P8.87: computeIs + computePropagateSpatialInertia_ZA_ZIc (forwardDynamic2.cu:1345
# -1544) -- the per-dof algebra of the articulated-inertia propagation.  MUST be in
# PXG_NOCONTRACT_SHADERS: dense float work compared bit-for-bit against a mirror
# built out of the engine's own SpatialMatrix / PxMat33 operators.
pxg_compile_shader(npArtiPropagateInertia artiPropagateInertia computeUnconstrainedSpatialInertiaLaunch1T artiArticulatedInertia)
# P8.90: the impulse-response pair (pxgSlangArticulationDynamic.slang), LIFTED from
# the Phase-1 npArtiPropagate / npArtiPropagateAccel bodies rather than re-derived.
# MUST be in PXG_NOCONTRACT_SHADERS: compared bit-for-bit against the engine's own
# translateSpatialVector / innerProduct on the host.
pxg_compile_shader(npArtiDynamic artiDynamicOps)
# P8.91b/c: computeMassMatrix1T and the warp-cooperative machinery it is built
# out of -- writeMatrix's staged scatter, warpCopy and the set-bit walk.  MUST be
# in PXG_NOCONTRACT_SHADERS as of P8.91c: the six-axis impulse response is dense
# float work compared bit-for-bit against the engine's own SpatialMatrix /
# PxSpatialMatrix operators on the host.  Absent from PXG_HLSL_ENTRIES (Ptr<>
# plus the groupshared warp emulation).
pxg_compile_shader(npArtiMassMatrix artiMassMatrixWrite artiMassMatrixZeroWalk computeMassMatrix1T artiMassMatrixBody)
# P8.92a: the bias-force sweep and the parallel-axis theorem -- the two pieces of
# the acceleration kernel that stand on their own.  MUST be in
# PXG_NOCONTRACT_SHADERS: both are compared bit-for-bit against the engine's own
# PxQuat / PxMat33 arithmetic on the host.
pxg_compile_shader(npArtiLinkAccel artiJointFriction artiTranslateInertiaMass artiLinkAcceleration computeUnconstrainedAccelerationsLaunch1T dmaBackArticulationDataLaunch)
# P8.94a: the position-integration algebra -- updateRootBody, propagateLink and
# the two quaternion helpers.  MUST be in PXG_NOCONTRACT_SHADERS: every result is
# compared bit-for-bit against the engine's own PxQuat / PxExp arithmetic.
pxg_compile_shader(npArtiStep artiStepAlgebra artiStepQuatProbe artiStepRotVecProbe artiStepSphGuardProbe)
pxg_compile_shader(npArtiFlushVel artiFlushVelocity artiOutputVelocity)
pxg_compile_shader(npArtiAvgImpulse artiAvgImpulse aiAvg2Test artiPushImpulse artiApplyTgsSubstepForces)
pxg_compile_shader(npArtiPropagate2 artiPropagateImpulses2TGS artiPropagateVelocityTGS artiPropagateImpulses2PGS artiPropagateVelocity artiSaveVelocity1TPGS artiComputeDependencies)
pxg_compile_shader(npArtiInverseDyn computeArtiCOM computeArtiDenseJacobians computeArtiCentroidalMomentumMatrices computeArtiGravityForces computeArtiCentrifugalForces computeArtiMassMatrices)
pxg_compile_shader(npArtiUpdateKin artiUpdateKinematic)
pxg_compile_shader(npArtiGetSet getArtiDofStates getArtiTransformStates getArtiVelocityStates getArtiSpatialForceStates setArtiDofStates setArtiRootGlobalPoseState setArtiRootVelocityState setArtiLinkForceState setArtiLinkTorqueState getArtiTendonState setArtiTendonState getArtiSpatialTendonAttachmentState setArtiSpatialTendonAttachmentState getArtiFixedTendonJointState setArtiFixedTendonJointState)
pxg_compile_shader(npArtiConstraintPrep acpBodyTest acpSetupTest artiTGSContactConstraintBlockPrepareLaunch artiContactConstraintBlockPrepareLaunch)
pxg_compile_shader(npArtiJointPrep artiTGSJointConstraintBlockPrepareParallelLaunch artiJointConstraintBlockPrepareParallelLaunch)
pxg_compile_shader(npArtiStepKernel stepArticulation1TTGS)
pxg_compile_shader(npArtiSleep artiSleepCheck)
pxg_compile_shader(npArtiUpdateBodies updateBodiesLaunch_Part2 updateBodiesLaunch_Part3)
# P9.43b: the first body-update stage (forwardDynamic2.cu:3047), joint-state writeback + momentum
pxg_compile_shader(npArtiUpdateBodies1T updateBodiesLaunch1T)
pxg_compile_shader(npArtiSumBatches artiSumInternalContactAndJointBatches1Launch)
pxg_compile_shader(npArtiSumBatches2 artiSumInternalContactAndJointBatches2Launch)
pxg_compile_shader(npArtiSumSelfBatches artiSumSelfContactAndJointBatches)
pxg_compile_shader(npArtiSetupInternal siSetupInternalTest)
pxg_compile_shader(npArtiSetupTendons stSetupSpatialTest stSetupFixedTest)
pxg_compile_shader(npArtiSetupMimic mjSetupMimicTest)
pxg_compile_shader(npArtiSetupICKernel setupInternalConstraintLaunch1T)
pxg_compile_shader(npArtiSolveTendons svGetVelTest svApplyImpTest svSolveSpatialTest svSolveFixedTest svSolveMimicTest)
pxg_compile_shader(npArtiSolveTMKernel artiSolveInternalTendonAndMimicJointConstraints1T)
pxg_compile_shader(npArtiSolveExtTGS svxContactTest svx1DTest)
pxg_compile_shader(npArtiSolveExtPGS svpContactTest svp1DTest artiSolveBlockPartition)
pxg_compile_shader(npArtiSolveSelf ssSolveSelfTest artiPropagateRigidImpulsesAndSolveSelfConstraintsTGS1T ssSolveSelfPgsTest artiPropagateRigidImpulsesAndSolveSelfConstraints1T)
pxg_compile_shader(npArtiSolveInternal siStaticTest siDofTest siDfsTest siStaticPgsTest siDofPgsTest siDfsPgsTest artiSolveInternalConstraintsTGS1T artiSolveInternalConstraints1T)
pxg_compile_shader(npPairMgmtBda removeContactManagers_Stage1 removeContactManagers_Stage2 removeContactManagers_Stage3 removeContactManagers_Stage4 removeContactManagers_Stage5 removeContactManagers_Stage5_CvxTri)
# P6.2: PBD particle spatial-hash grid build + reorder on the engine ABI (BDA).
# Both kernels read the PxgParticleSystem descriptor array through Ptr<> and
# grid-stride, which has no HLSL cs_6_0 lowering, so this shader is deliberately
# absent from PXG_HLSL_ENTRIES (same precedent as particleSim /
# particleContactSortBda / solveBlockUnified).
pxg_compile_shader(particleGridBda ps_calculateHashLaunch ps_reorderDataAndFindCellStartLaunch ps_calculateHashForDiffuseParticlesLaunch)
# P6.3: PBD particle self-collision neighbour search + one-way contact solve +
# delta-apply + velocity recompute on the engine ABI (BDA).  All four kernels read
# the PxgParticleSystem descriptor array through Ptr<>, which has no HLSL cs_6_0
# lowering, so this shader is deliberately absent from PXG_HLSL_ENTRIES (same
# precedent as particleSim / particleGridBda / particleContactSortBda).
pxg_compile_shader(particleSelfCollisionBda ps_selfCollisionLaunch ps_solveOneWayContactDeltaVLaunch ps_applyDeltaLaunch ps_updateParticleLaunch)

# P9.1a: the per-phase-group derived-material precompute (particlesystem.cu:5780),
# the only writer of mDerivedPBDMaterialData and therefore a hard prerequisite of
# the PBD density kernels.  IS in PXG_NOCONTRACT_SHADERS: the four derived
# coefficients are dense float arithmetic the density solve consumes bit for bit.
pxg_compile_shader(npPsMaterials ps_updateMaterials)

# P9.1b: the PBD SPH density / surface-normal / diffuse-potential accumulation
# (particlesystem.cu:3222).  IS in PXG_NOCONTRACT_SHADERS: dense SPH float work
# whose bit pattern ps_solveDensityLaunch consumes directly.
pxg_compile_shader(npPsFluidDensity ps_calculateDensityAndPotentialLaunch ps_solveDensityLaunch)

# P9.1d: the PBD velocity pair -- the curl of the velocity field and the
# vorticity-confinement impulse it drives (particlesystem.cu:3757 and :3891).
# IS in PXG_NOCONTRACT_SHADERS: dense SPH float work.
pxg_compile_shader(npPsFluidVelocity ps_vorticityConfinementLaunch ps_solveVelocityLaunch)

# P9.3a: the first two kernels of the PBD cloth spring chain (particlesystem.cu
# :4111 and :4043).  A 3D grid -- system on Z, cloth buffer on Y.
pxg_compile_shader(npPsSprings ps_initializeSpringsLaunch ps_updateRemapVertsLaunch ps_solveSpringsLaunch ps_averageVertsLaunch)
pxg_compile_shader(npPsAerodynamics ps_solveAerodynamics1Launch ps_solveAerodynamics2Launch)
pxg_compile_shader(npPsInflatable ps_calculateInflatableVolume ps_solveInflatableVolume)
# P9.8: the PBD fluid BOUNDARY density pair (particlesystem.cu :2894 and :2944).
# The two are CHAINED -- the rigid kernel seeds its accumulator from what the
# static one wrote into mStaticDensity.  IS in PXG_NOCONTRACT_SHADERS: W() is
# dense float work.
pxg_compile_shader(npPsBoundaryDensity ps_accumulateStaticDensity ps_accumulateRigidDensity)

# P9.10 -- the particle-vs-trimesh midphase + contact gen on the engine BDA ABI.
# IS in PXG_NOCONTRACT_SHADERS: the triangle tests are dense float work and FMA
# contraction moves the results off psMeshMid.slang's bit pattern, which is what
# the differential fixture compares against.
pxg_compile_shader(psMeshMidphaseBda ps_meshCollisonLaunch psMeshMidphaseWorkIndex)

# P9.11 -- the particle-vs-heightfield midphase + contact gen on the engine BDA ABI.
# IS in PXG_NOCONTRACT_SHADERS, and so is psHfCollide as of this tranche.  THE RULE,
# learned the hard way twice: when a tranche lifts a Phase-1 module onto BDA for a
# GPU-differential fixture, the PHASE-1 MODULE MUST JOIN THE LIST IN THE SAME
# TRANCHE -- otherwise the two programs compute the same arithmetic under different
# contraction rules and the differential silently degrades to a tolerance test.
# P9.10 found psMeshMid missing after the fact (14 of 97 particles disagreed at the
# ULP with the counts matching everywhere); P9.11 checked first.
# ABSENT from PXG_HLSL_ENTRIES: Ptr<> plus WorkgroupCount(), which has no cs_6_0
# lowering -- the same reason particleContactSortBda is excluded.
pxg_compile_shader(psHfMidphaseBda ps_heightfieldCollisonLaunch)
# P9.12h -- the particle-vs-SDF-mesh midphase + contact gen on the engine BDA ABI.
# IS in PXG_NOCONTRACT_SHADERS: it inherits every float decision of
# pxgSlangSdfCollision and pxgSlangSdfParticle, which the P9.12d/f fixtures hold
# to bit-exactness against the engine's own PxVec3 arithmetic.
pxg_compile_shader(psSdfMidphaseBda ps_sdfMeshCollisonLaunch)
# P9.13c: the three per-BUFFER bookkeeping kernels of the diffuse chain.  In
# PXG_NOCONTRACT_SHADERS with the rest of the particle corpus -- there is no float
# arithmetic in any of the three, so contraction is moot, but the flag is what the
# other particle shaders carry and a divergent build flag is its own trap.
pxg_compile_shader(psDiffuseBuffers ps_updateUnsortedDiffuseArrayLaunch ps_diffuseParticleCopy ps_diffuseParticleSum)
# P9.13d: ps_diffuseParticleCreate -- the only kernel in the chain that makes a
# diffuse particle.  NOCONTRACT with the rest of the particle corpus: the spawn
# position is a three-term float expression and contraction would move it off the
# engine's bit pattern.
pxg_compile_shader(psDiffuseCreate ps_diffuseParticleCreate)
# P9.13e: the two kernels that advect and age diffuse particles.  NOCONTRACT --
# the neighbourhood weighted average and the three integrations are dense float
# work and contraction moves them off the engine's bit pattern.
pxg_compile_shader(psDiffuseAdvect ps_diffuseParticleUpdatePBF ps_diffuseParticleCompact)
pxg_compile_shader(psDiffuseOneWay ps_diffuseParticleOneWayCollision)
# P7.0: FEMCloth free-fall bring-up — the three cloth orchestration kernels on the
# engine ABI (BDA).  All three read the PxgFEMCloth descriptor array through Ptr<>
# (no HLSL cs_6_0 lowering) and the finalize/sleeping kernels use groupshared
# InterlockedOr ballots, so this shader is deliberately absent from
# PXG_HLSL_ENTRIES (same precedent as particleSim / particleSelfCollisionBda).
pxg_compile_shader(femClothSimBda cloth_preIntegrateLaunch cloth_finalizeVelocitiesLaunch cloth_sleeping)
# P7.1 tranche 1: FEMCloth in-plane (membrane / ARAP + area) energy solve on the
# engine ABI (BDA).  Both kernels read the PxgFEMCloth descriptor array + the
# PxsDeformableSurfaceMaterialData array through Ptr<> (no HLSL cs_6_0 lowering)
# and the cluster kernel loops partitions with a groupshared barrier, so this
# shader is deliberately absent from PXG_HLSL_ENTRIES (same precedent as
# femClothSimBda / particleSelfCollisionBda).  The in-plane math is imported
# verbatim from the Phase-1-verified pxgSlangClothMembrane module.
pxg_compile_shader(femClothSolveBda cloth_solveNonSharedTriangleEnergyLaunch
                   cloth_solveNonSharedTriangleEnergyClusterLaunch)
# P7.1 tranche 2: FEMCloth triangle-pair (shared shell-energy + Discrete-Shells
# bending) solve on the engine ABI (BDA).  All three kernels read the PxgFEMCloth
# descriptor array + the PxsDeformableSurfaceMaterialData array through Ptr<> (no
# HLSL cs_6_0 lowering) and the cluster kernel loops partitions with a groupshared
# barrier, so this shader is deliberately absent from PXG_HLSL_ENTRIES (same
# precedent as femClothSolveBda / femClothSimBda).  The in-plane math is imported
# verbatim from pxgSlangClothMembrane; the bending math from pxgSlangClothBending
# (a DOCUMENTED-REPLACEMENT / tolerance leaf — it evaluates atan2 on-device).
pxg_compile_shader(femClothSolvePairBda cloth_solveTrianglePairEnergyLaunch
                   cloth_solveTrianglePairEnergyClusterLaunch
                   cloth_averageTrianglePairVertsLaunch)
# P9.19: the cloth DAMPING trio.  One file because PxgFEMClothCore::applyDamping
# launches all three back to back and the applier is meaningless without the two
# accumulators that fill mAccumulatedDeltaVel.
pxg_compile_shader(femClothDampingBda cloth_accumulateInPlaneDampingDeltaVelocity
                   cloth_accumulateBendingDampingDeltaVelocity
                   cloth_applyAccumulatedDeltaVelocity)
# P7.1 tranche 3a: FEMCloth RIGID-ATTACHMENT constraint prep + PGS/TGS solve on the
# engine ABI (BDA) — the machinery to PIN a cloth vertex to a rigid / world anchor
# (toward a hanging cloth).  All three kernels read the PxgFEMCloth descriptor array,
# the block-of-32 PxgFEMRigidAttachmentConstraint, the PxgFEMRigidAttachment array
# and the prep / solver-core / solver-shared desc mirrors through Ptr<> (no HLSL
# cs_6_0 lowering) and grid-stride via nbIterationsPerBlock, so this shader is
# deliberately absent from PXG_HLSL_ENTRIES (same precedent as femClothSolveBda /
# psContactSolveTgsBda).  The articulation anchor branch + dynamic-rigid reaction +
# cone-limited attachment path are deferred (rigid / world anchor scope).
pxg_compile_shader(femClothRigidAttachBda cloth_rigidAttachmentPrepareLaunch
                   cloth_solveRigidClothAttachmentLaunch
                   cloth_solveRigidClothAttachmentTGSLaunch
                   cloth_applyExternalDeltasLaunch)
# P7.1 tranche 4a: cloth-vertex-vs-PRIMITIVE contact GENERATION on the engine ABI
# (BDA) — the first half of the cloth-drape-on-a-floor arc.  The midphase vertex-
# pair pass + the plane / sphere vertex contact generators read the PxgFEMCloth
# descriptor array through Ptr<> and write the by-value PxgFEMContactWriter's device
# pointers, and lower the warp ballot/shuffle contact-compaction to portable
# groupshared tile scans (no HLSL cs_6_0 lowering), so this shader is deliberately
# absent from PXG_HLSL_ENTRIES (same precedent as psContactGenBda / femClothSolveBda).
pxg_compile_shader(femClothContactGenBda cloth_midphaseGenerateVertexPairsLaunch
                   cloth_planeVertContactGenLaunch
                   cloth_SphereVertexContactGenLaunch
                   cloth_boxVertexContactGenLaunch
                   cloth_midphaseGeneratePairsLaunch
                   cloth_boxTriangleContactGenLaunch
                   cloth_convexContactGenLaunch
                   cloth_convexVertexContactGenLaunch
                   # P9.29: the sphere/capsule TRIANGLE path -- the last unmapped
                   # primitive contact generator on the triangle side.
                   cloth_SphereContactGenLaunch)
# P7.1 tranche 4b-1: cloth-vs-rigid contact SORT / reorder on the engine ABI (BDA)
# — the second half of the cloth-drape-on-a-floor arc.  The single reorder kernel
# gathers the 4a-generated cloth-vs-rigid contacts into the sorted buffers through
# the by-rigid radix ranks; it reads every buffer through Ptr<> (BDA, no HLSL cs_6_0
# lowering) so it is deliberately absent from PXG_HLSL_ENTRIES (same precedent as
# psContactSortBda / femClothContactGenBda).  A PURE index-remapped load/store
# gather — no float arithmetic — so deliberately absent from PXG_NOCONTRACT_SHADERS
# too (same reasoning as ps_reorderPrimitiveContactsLaunch).
pxg_compile_shader(femClothContactSortBda fem_reorderRigidContactsLaunch)
# P7.1 tranche 4b-2/3: cloth-vs-rigid contact PREPARE + SOLVE on the engine ABI (BDA) —
# the final half of the cloth-drape-on-a-floor arc.  The prepare builds the block-of-32
# PxgFemRigidConstraintBlock from the sorted contacts; the query/solve (PGS + TGS)
# accumulate the per-vertex reference count and scatter the cloth-side push into mDeltaPos.
# Reads every descriptor / SoA block through Ptr<> (BDA, no HLSL cs_6_0 lowering) so it is
# deliberately absent from PXG_HLSL_ENTRIES (same precedent as femClothRigidAttachBda /
# psContactSolveTgsBda), and IS in PXG_NOCONTRACT_SHADERS (the dot/normalize-heavy contact
# bias math needs -ffp-contract=off parity, same as psContactSolveTgsBda).  Articulation
# branch + dynamic-rigid reaction + material friction are deferred (static-rigid scope).
pxg_compile_shader(femClothRigidContactBda cloth_rigidContactPrepareLaunch
                   cloth_queryRigidClothContactReferenceCountLaunch
                   cloth_queryRigidClothContactReferenceCountTGSLaunch
                   cloth_solveRigidClothCollisionLaunch
                   cloth_solveRigidClothCollisionTGSLaunch)
# P7.2 tranche 1: cloth SELF-collision midphase + contact GENERATION on the engine ABI
# (BDA) — the first tranche of the cloth-fold-on-itself arc.  Both kernels walk the
# cloth's own BV32 triangle BVH (reused verbatim from the proven npBv32 / npClothCloth
# ports) reading the PxgFEMCloth descriptor array + the BV32DataPacked nodes through
# Ptr<> (BDA, no HLSL cs_6_0 lowering) and lower the warp ballot/shuffle contact-
# compaction to portable groupshared tile scans, so this shader is deliberately absent
# from PXG_HLSL_ENTRIES (same precedent as femClothContactGenBda / psContactGenBda).  IS
# in PXG_NOCONTRACT_SHADERS (the dot/cross-heavy VT/EE closest-point distances feed the
# compared penetration tests, needing -ffp-contract=off parity).  The different-cloth
# siblings (cloth_differentClothCollision{VT,EE}Launch — a separate arg pack) are the
# next scope.
pxg_compile_shader(femClothSelfCollisionBda cloth_selfCollisionMidphaseVTLaunch
                   cloth_selfCollisionMidphaseEELaunch)
# P7.2 tranche 2: cloth-vs-CLOTH self-contact SOLVE on the engine ABI (BDA) — completes the
# cloth-fold-on-itself pipeline the tranche-1 midphase started.  The prepare recomputes the
# exact rest-pose distance filter, the query counts each cloth vertex's self-contact
# references into mDeltaPos.w, the VT/EE solves scatter BOTH endpoints' corrections into
# mDeltaPos (folded by the already-live cloth_applyExternalDeltasLaunch), and the validity
# pass snapshots prevPositionInContactOffset.  Reads every descriptor / contact record
# through Ptr<> (BDA, no HLSL cs_6_0 lowering) so it is deliberately absent from
# PXG_HLSL_ENTRIES (same precedent as femClothRigidContactBda / femClothSelfCollisionBda),
# and IS in PXG_NOCONTRACT_SHADERS (the dot/cross/normalize-heavy closest-point + constraint
# math needs -ffp-contract=off parity).  Material friction is the deferred null-material
# (frictionless) path; the different-cloth pairs flow through the identical generic
# arithmetic.
pxg_compile_shader(femClothClothContactBda cloth_clothContactPrepareLaunch
                   cloth_queryClothClothContactVTCountLaunch
                   cloth_queryClothClothContactEECountLaunch
                   cloth_solveClothClothDeltaVTLaunch
                   cloth_solveClothClothDeltaEELaunch
                   cloth_updateClothContactValidityLaunch)
# P7.2 tranche 3a: the cloth BVH bottom-up BOUNDS REFIT (cloth_refitBoundLaunch) — the
# missing link that makes the tranche-1 self-collision midphase query a LIVE tree instead
# of the rest-pose one it was cooked with (without it the midphase finds zero contacts).
# Walks the Gu::BV32DataPacked nodes + BV32DataDepthInfo levels through Ptr<> (BDA, no
# HLSL cs_6_0 lowering) and lowers the CUDA __ballot_sync / warpShuffleMin/Max butterflies
# to groupshared collectives, so this shader is deliberately absent from PXG_HLSL_ENTRIES
# (same precedent as femClothSelfCollisionBda / psContactGenBda).  IS in
# PXG_NOCONTRACT_SHADERS: the refit itself is pure min/max (no rounding freedom), but the
# broad-phase tail's `PxMax(nbUpdates,1) * contactDist` feeds an add/sub that would fuse
# to an FMA without -ffp-contract=off parity.
pxg_compile_shader(femClothRefitBda cloth_refitBoundLaunch)
# P8.0: SOFTBODY (FEM-volume) free-fall bring-up — the first softbody orchestration
# kernel on the engine ABI (BDA).  sb_gm_preIntegrateLaunch reads the PxgSoftBody
# descriptor array through Ptr<> (no HLSL cs_6_0 lowering), so this shader is
# deliberately absent from PXG_HLSL_ENTRIES (same precedent as femClothSimBda /
# particleSim).  IS in PXG_NOCONTRACT_SHADERS (the gravity/damping mul+add predict
# needs -ffp-contract=off parity vs the CPU reference).  The kernel's ONLY warp op
# (the speculative-CCD AtomicMax reduction) is omitted as dead on the free-fall
# scene, so there is no groupshared / barrier and no subgroup lowering (see the
# shader header).
pxg_compile_shader(softBodySimBda sb_gm_preIntegrateLaunch
                   sb_gm_stepSoftbodyLaunch
                   sb_gm_finalizeVelocitiesLaunch
                   sb_sleeping)
# P8.0 tranche 2: the SOFTBODY BVH bounds refit (sb_refitBoundLaunch) — second kernel of
# the free-fall chain.  Direct sibling of femClothRefitBda (same algorithm over
# tetrahedra instead of triangles), same BDA Ptr<> reads and same groupshared warp
# lowering, so likewise absent from PXG_HLSL_ENTRIES.  IS in PXG_NOCONTRACT_SHADERS: the
# kernel is pure min/max plus one addition and cannot contract today, but the stamp keeps
# the bit-exactness guarantee independent of future edits.
pxg_compile_shader(softBodyRefitBda sb_refitBoundLaunch)
# P8.0 tranche 4: the first two kernels of the softbody FEM-volume SOLVE loop, which
# bracket it — sb_gm_zeroTetMultipliers clears the per-tet Lagrange multipliers at the
# top, sb_gm_applyExternalDeltasLaunch folds the accumulated per-vertex correction back
# into position/velocity and clears the delta buffer at the bottom.  Reads the descriptor
# through Ptr<> (BDA, no HLSL cs_6_0 lowering) so it is deliberately absent from
# PXG_HLSL_ENTRIES.  IS in PXG_NOCONTRACT_SHADERS (`vel + delta * invDt` is a mul feeding
# an add).  No transcendental and no warp op in either kernel.
pxg_compile_shader(softBodySolveBda sb_gm_zeroTetMultipliers
                   sb_gm_applyExternalDeltasLaunch
                   sb_gm_updateTetrahedraRotationsLaunch
                   sb_updateTetrahedraRotationsLaunch
                   sb_gm_cp_solveTetrahedronsPartitionLaunch
# P8.0 tranche 7: the three kernels that CLOSE the softbody step —
# sb_gm_cp_averageVertsLaunch merges the per-partition vertex copies,
# sb_gm_applyDeformationDeltasLaunch is the Jacobi twin of applyExternalDeltas, and
# sb_gm_updateTetModelVertsLaunch skins the simulation mesh onto the collision mesh (the
# step's last kernel, and the one that makes the motion observable outside the solver).
                   sb_gm_cp_averageVertsLaunch
                   sb_gm_applyDeformationDeltasLaunch
                   sb_gm_updateTetModelVertsLaunch
# P8.5: sb_calculateStressLaunch, the co-rotated Cauchy stress field.  It lives here because
# it shares the flat-PxMat33 rest-pose read and the descriptor walk with the rotation
# extractors above; its output is what the SELF-COLLISION stress gate reads.
                   sb_calculateStressLaunch)
# P8.2 tranche a: the SOFTBODY-vs-RIGID midphase -- one warp per contact-manager pair
# walks the softbody's own BV32 tetrahedron BVH against the rigid's fattened world AABB and
# appends uint4(cmIdx, tetIdx, 0, 0) to the shared collision stack.  Reads everything through
# Ptr<> (BDA, no HLSL cs_6_0 lowering) so it is deliberately absent from PXG_HLSL_ENTRIES.
# IS in PXG_NOCONTRACT_SHADERS: isValidTet's signed volume is a dot-of-cross that would
# otherwise fuse to FMAs and break parity with the -ffp-contract=off CPU reference.
pxg_compile_shader(softBodyContactGenBda sb_midphaseGeneratePairsLaunch
                   sb_primitiveContactGenLaunch)
# P8.2 tranche c: the collision-mesh -> simulation-mesh contact remap and the sorted-order
# gather.  Both are LIFTS of npSbRemap.slang's unit-tested descriptor-set ports onto the
# engine BDA ABI; the geometry helpers are copied from it verbatim.
# P8.4 tranche b adds sb_fem_contact_remap_to_simLaunch to this file: the
# DEFORMABLE-vs-DEFORMABLE twin of the remap above, run twice over the same contact array
# (gridDim.y == 2 selects which half of the pair is remapped).  It shares every helper and
# both llvmpipe workarounds with its sibling, which is why it lives here rather than in its
# own shader.
pxg_compile_shader(softBodyContactRemapBda sb_other_contact_remap_to_simLaunch
                   sb_reorderPSContactsLaunch
                   sb_fem_contact_remap_to_simLaunch)
# P8.4 tranche b: the SOFTBODY-vs-SOFTBODY constraint prepare and solve -- chain kernels 2
# and 3 of the self-collision arc.  Symmetric twins of the rigid-vs-softbody pair from P8.2
# tranche d/e: both sides are softbody tets, so every step happens twice with the sign
# flipped.  In PXG_NOCONTRACT_SHADERS: the barycentric blends, the relative-position dot and
# the friction terms are mul-feeding-add chains compared bit-exactly against a CPU reference.
pxg_compile_shader(softBodySelfSolveBda sb_softbodyContactPrepareLaunch
                   sb_solveOutputSSDeltaVLaunch
                   sb_solveOutputSSDeltaVLaunchTGS)
# P8.2 tranche d: chain kernel 5, the softbody-vs-rigid CONSTRAINT prepare.  A near-clone of
# cloth_rigidContactPrepareLaunch -- same ABI row, same 104-byte blob -- sharing its
# 78-line constraint-block writer through the new pxgSlangFemRigidContact module rather than
# copying it.  In PXG_NOCONTRACT_SHADERS: the response terms are dot-of-cross chains through
# mat33MulVec that would otherwise fuse to FMAs and break parity with the CPU reference.
# P8.3: the softbody-vs-rigid ATTACHMENT path -- pin a softbody to a rigid or to the world.
# Same near-clone relationship to femClothRigidAttachBda as every other softbody tranche:
# identical ABI rows, with the constraint/attachment byte layouts shared through
# pxgSlangFemRigidAttach.  In PXG_NOCONTRACT_SHADERS: the responses are dot-of-cross chains
# and the quaternion rotate is a mul/add chain, both compared bit-exactly against a CPU
# reference.
# P9.33c: the softbody-vs-CLOTH attachment solve -- the THIRD and last consumer of
# PxgFEMFEMAttachmentConstraint, after the softbody-softbody pair (P8.19) and the cloth-cloth
# one (P9.32).  A tetrahedron of one body pinned to a triangle of the other, so it walks TWO
# descriptor arrays with different strides and scatters through updateTriPositionDelta on the
# cloth side (three corners, each skippable) and updateTetPositionDelta on the softbody side.
# The cone axis is `pos1 - projectP` over the SOFTBODY's corners -- the negation of the
# sibling's shape, which is the one difference here a derived port carries over intact.
# IS in PXG_NOCONTRACT_SHADERS: the barycentric blends, the mass sums and the cone projection
# are mul-feeding-add chains a fixture compares bit-exactly against a CPU reference.
pxg_compile_shader(softBodyClothAttachBda sb_solveOutputClothAttachmentDeltaVLaunch)
# P9.48b: the FOURTH consumer of PxgFEMFEMAttachmentConstraint -- a softbody
# tetrahedron pinned to a single PARTICLE.  No cone limit, an asymmetric wTot0, a
# buffer id hiding in barycentric0.x, and a three-indirection particle index.
pxg_compile_shader(softBodyParticleAttachBda sb_solveOutputParticleAttachmentDeltaVLaunch)
pxg_compile_shader(softBodyRigidAttachBda sb_rigidAttachmentPrepareLaunch
                   sb_solveRigidSoftAttachmentLaunch
                   sb_solveRigidSoftAttachmentLaunchTGS)
# P9.76: the PARTICLE-vs-rigid attachment trio, the third and last attachment
# family.  Its two solve entry points have DIFFERENT CUDA signatures (8 and 10
# parameters) and ONE shared host pack, so both declare the same 72-byte push
# block -- transcribing the signature instead of the pack would refuse the PGS
# entry on every launch, which is P9.76b's failure in mirror image.
pxg_compile_shader(psRigidAttachBda ps_rigidAttachmentPrepareLaunch
                   ps_solveRigidAttachmentsLaunch
                   ps_solveRigidAttachmentsTGSLaunch)
# P8.4 tranche a: the softbody SELF-collision midphase -- one warp per SURFACE vertex walks
# its own softbody's BV32 tetrahedron BVH and keeps the single deepest surviving
# vertex-vs-tetrahedron contact.  The traverser body is a LIFT of npSbSelf.slang's
# unit-tested descriptor-set port onto the engine BDA ABI.  Reads everything through Ptr<>
# so it is deliberately absent from PXG_HLSL_ENTRIES.  IS in PXG_NOCONTRACT_SHADERS:
# isValidTet, isPointOutsideOfPlane4 and closestPtPointTriangle2 are dot-of-cross and
# mul-feeding-add chains compared bit-exactly against a CPU reference.
pxg_compile_shader(softBodySelfCollisionBda sb_selfCollisionMidphaseGeneratePairsLaunch)
pxg_compile_shader(softBodySoftBodyMidphaseBda sb_sbMidphaseGeneratePairsLaunch)
pxg_compile_shader(softBodyMeshMidphaseBda sb_meshMidphaseGeneratePairsLaunch)
pxg_compile_shader(softBodyMeshContactGenBda sb_meshContactGenLaunch)
# P9.81: the softbody-vs-HEIGHTFIELD contact gen -- step 2 of the pair whose step 1
# (softBodyHfMidphaseBda) has been mapped since P8.11.  Its own 176-byte ABI row, NOT
# the mesh sibling's 192: this signature has no filterPairs/nbFilterPairs.
pxg_compile_shader(softBodyHfContactGenBda sb_heightfieldContactGenLaunch)
pxg_compile_shader(softBodyHfMidphaseBda sb_heightfieldMidphaseGeneratePairsLaunch)
pxg_compile_shader(femClothStepBda cloth_stepLaunch)
# P8.16: cloth-vs-DIFFERENT-cloth midphase, the sibling femClothSelfCollisionBda's header
# deferred as "a SEPARATE arg pack / ABI".  Both entry points share the 88-byte
# narrowphase-driven blob (cmInputs/gpuShapes instead of an active-cloth list) and import
# pxgSlangClothMidphase, the module P8.15 extracted for exactly this second consumer.  Reads
# every descriptor through Ptr<> and lowers the warp ballot/scan to groupshared collectives,
# so it is deliberately absent from PXG_HLSL_ENTRIES (same precedent as
# femClothSelfCollisionBda).  IS in PXG_NOCONTRACT_SHADERS: the VT/EE closest-point distances
# feed the compared penetration tests and need -ffp-contract=off parity.
pxg_compile_shader(femClothDifferentClothBda cloth_differentClothCollisionVTLaunch
                   cloth_differentClothCollisionEELaunch)
# P8.17 step 1: the cloth-vs-COOKED-TRIANGLE-MESH midphase.  softBodyMeshMidphaseBda's sibling —
# same cooked blob walk, same top-4 extraction, same grid — with a cloth TRIANGLE query, the
# triangle-vs-triangle satIntersect (nine edge-cross axes on top of the two plane terms), a plain
# AABB overlap against a PRE-FATTENED bound, and no scale division on the contact distance.  Reads
# the blob and the PxgFEMCloth descriptor through Ptr<> and lowers the warp ballot / minIndex to
# groupshared collectives, so it is deliberately absent from PXG_HLSL_ENTRIES.  IS in
# PXG_NOCONTRACT_SHADERS: every plane product and SAT projection is a mul-feeding-add chain.
# P9.89: cloth_meshMidphaseSatProbe, a test-only entry that reads the triangle-triangle SAT's
# selectMin/selectMax folds out bit for bit (femClothSimMeshMidphase_p89SatProbe).
pxg_compile_shader(femClothMeshMidphaseBda cloth_meshMidphaseGeneratePairsLaunch
                   cloth_meshMidphaseSatProbe)
# P8.17 step 2: the cloth-vs-COOKED-TRIANGLE-MESH contact generation, the consumer step 1's
# midphase had none.  The clipping core is a LIFT of fcStaticTriCloth.slang's mode 0 — already
# fixture-verified against a CPU transcription for the descriptor-set engine — with new BDA
# plumbing (the 192-byte blob, the cooked-blob walk, the PxgFEMCloth reads, the pair-stack
# iteration) that fcStaticTriCloth explicitly leaves out as engine glue.  Reads everything
# through Ptr<> and lowers the warp ballot to a groupshared collective, so it is deliberately
# absent from PXG_HLSL_ENTRIES.  IS in PXG_NOCONTRACT_SHADERS: the plane distances, the trim
# lerps and the barycentric solve are mul-feeding-add chains.
pxg_compile_shader(femClothMeshContactGenBda cloth_meshContactGenLaunch)
# P9.26: the VERTEX half of the same cloth-vs-cooked-mesh block.  Steps 1 and 2 above are the
# TRIANGLE half; the engine runs both, one after the other, off the same bounds and the same
# intermediate stack (PxgNarrowphaseCore.cpp:6581/:6671 then :6724/:6814).
#
# The midphase is softBodyClothVertMidphaseBda's mirror image -- there a cloth VERTEX queries a
# softbody tet tree, here it queries a cooked TRIANGLE tree -- and it reuses the shared BV32
# reader (P8.25) rather than a fourth copy of the walk.  Reads the blob and the PxgFEMCloth
# descriptor through Ptr<> and lowers the warp ballot to a groupshared collective, so it is
# deliberately absent from PXG_HLSL_ENTRIES.  IS in PXG_NOCONTRACT_SHADERS: the vertex-space
# round trip and the sphere bound are mul-feeding-add chains that a fixture compares bit-exactly.
pxg_compile_shader(femClothMeshVertMidphaseBda cloth_midphaseVertexMeshLaunch)
# P9.30: the cloth-vs-PARTICLE midphase.  Same BV32 point query as the sibling above,
# but the query point is a particle and the tree is the CLOTH's, both already in world
# space -- so no vertex-space round trip here, which is the one thing that differs.
pxg_compile_shader(femClothPsMidphaseBda cloth_psMidphaseGeneratePairsLaunch)

# P9.39a -- sb_psMidphaseGeneratePairsLaunch, the TET twin of the line above.
pxg_compile_shader(softBodyPsMidphaseBda sb_psMidphaseGeneratePairsLaunch)
pxg_compile_shader(softBodyPsContactGenBda sb_psContactGenLaunch)
pxg_compile_shader(softBodyPsPrepareBda sb_particleContactPrepareLaunch)
pxg_compile_shader(softBodyPsSolveBda sb_solveOutputSPDeltaVLaunch sb_solveOutputParticleDeltaVLaunch)

# P9.39b -- the BDA fixture shader for pxgSlangDefFilter's engine-path searches.
pxg_compile_shader(npDefFilterBda dfbRun)
# P9.30: the contact gen it feeds.  Per THREAD, at most one contact per pair, and
# no warp aggregation -- every contact takes its own atomicAdd of 1.
pxg_compile_shader(femClothPsContactGenBda cloth_psContactGenLaunch)
# P9.31: the constraint PREPARE that turns those contacts into solver constraints.
pxg_compile_shader(femClothPsPrepareBda cloth_particleContactPrepareLaunch)
# P9.31: the two delta-V outputs.  ONE body, two entry points -- and unlike the
# box/convex pair these are NOT twins: the two differ in the sign of the impulse,
# in an extra invMass0 factor, and in where the result lands.
pxg_compile_shader(femClothPsSolveBda cloth_solveCPOutputClothDeltaVLaunch
                   cloth_solveCPOutputParticleDeltaVLaunch)
# P9.32: the cloth-vs-CLOTH ATTACHMENT solve -- pin a point inside a triangle of one cloth
# to a point inside a triangle of another (or of the same one).  The constraint block is
# written by femAttachmentPrepareLaunch in softBodyFemAttachBda, which makes
# PxgFEMFEMAttachmentConstraint an ABI with three consumers; its byte layout therefore moved
# out of that file into the new pxgSlangFemFemAttach module rather than being copied here.
# The two SOLVE bodies stay apart -- they differ in corner count, in the cone-limit branch,
# in what divides the impulse and in whether the scatter skips zero weights.
# IS in PXG_NOCONTRACT_SHADERS: the barycentric blend and the scatter are mul-feeding-add
# chains a fixture compares bit-exactly against a CPU reference.
pxg_compile_shader(femClothAttachBda cloth_solveOutputAttachmentClothClothDeltaVLaunch)
# P9.34c: the SDF cloth contact gen -- the last unmapped cloth contact generator, deferred
# as "large" since P9.27 and reached here as COMPOSITION rather than transcription.  Both
# halves that carry the work landed in the two tranches before it: P9.34a's
# doTriangleSDFCollision (the descent, in pxgSlangSdfCollision) and P9.34b's
# findInterestingTrianglesA (the resumable block-wide gather, in pxgSlangSdfFind), each
# swept on its own.  What is new is the glue -- the cloth/trimesh side sort, the three
# different distance-indexing conventions, the unscaled cloth side, the sep rescale between
# the two gates, and the block-wide scan every thread must reach.
#
# `#define SUPPORT_SUBDIVISION 0` sits directly above the CUDA kernel, so the entire
# on-the-fly refinement apparatus is compiled away upstream and is not transcribed.
# IS in PXG_NOCONTRACT_SHADERS: the vertex-space round trips, the barycentric solve and the
# normalise-and-rescale are mul-feeding-add chains a fixture compares bit-exactly.
pxg_compile_shader(femClothSdfMeshCgBda cloth_sdfMeshContactGenLaunch)
# P9.35c: the SOFTBODY SDF midphase, the direct sibling of the cloth one above and the
# fastest way to see what each of them says.  Both halves it stands on were ported and
# swept on their own first: P9.35a's doTetrahedronSDFCollision and P9.35b's
# findInterestingTets.  It INVERTS the cloth kernel's cullScale trap -- this one takes
# the absolute value -- and disagrees with the gather it itself calls, which does not.
# IS in PXG_NOCONTRACT_SHADERS: the vertex-space round trips and the normalise-and-
# rescale are mul-feeding-add chains a fixture compares bit-exactly.
pxg_compile_shader(softBodySdfMidphaseBda sb_sdfMeshMidphaseGeneratePairsLaunch)
# The contact gen this midphase feeds emits NOTHING, and that is upstream's doing: the only
# assignment to its `hasContacts` flag is commented out (femClothPrimitives.cu:3056), and
# PxgOutputIndex skips its atomicAdd on an empty ballot, so neither a contact nor the count
# moves.  Its one observable output is an atomicMax that sizes next frame's stack, and that is
# what this shader implements; the dead closest-point geometry is deliberately NOT transcribed,
# because no fixture here could tell a right transcription from a wrong one.
#
# NOT in PXG_NOCONTRACT_SHADERS, and not by oversight: the shader contains no floating-point
# arithmetic at all, so there is no mul-feeding-add for -ffp-contract to fuse.  Add it the day
# the geometry arrives.  gpuslang.disabledKernels is what will tell you that day has come.
pxg_compile_shader(femClothMeshVertContactGenBda cloth_meshVertexContactGenLaunch)
# P9.27: the cloth-vs-HEIGHTFIELD midphase pair.  ONE body with two entry points, and
# unlike P9.20's pair that is upstream's own structure -- the CUDA declares one
# __device__ clothHeightfieldMidphaseCore and calls it from both kernels, so no
# twin-equivalence check is needed to keep the sharing honest.  The two differ in the
# element bound (a fattened cloth TRIANGLE bound against a SPHERE bound around a cloth
# VERTEX) and in the count the element index is compared against, and this shader
# parameterises exactly those two.
#
# Reads the heightfield blob and the PxgFEMCloth descriptor through Ptr<> and lowers the
# warp scan to a groupshared prefix sum, so it is deliberately absent from
# PXG_HLSL_ENTRIES.  IS in PXG_NOCONTRACT_SHADERS: the basisExtent columns feeding the
# local bound are mul-then-add chains, and an ULP there changes a floor()/ceil() and
# therefore the emitted pair SET.
pxg_compile_shader(femClothHfMidphaseBda cloth_heightfieldMidphaseGeneratePairsLaunch
                   cloth_midphaseVertexHeightfieldLaunch)
# P9.27b: the two consumers of that midphase.  cloth_heightfieldContactGenLaunch is the
# mesh contact gen's sibling -- same pair-stack walk, same staticTriangleVsClothCollision
# (shared via pxgSlangClothTriContact) -- differing in four things the header names: the
# shapes are read POSITIONALLY rather than routed by body id, the triangle comes out of
# the samples already scaled (so no vertex2Shape), there are no filter pairs, and the
# transform is hfToCloth.
#
# cloth_heightfieldVertexContactGenLaunch emits NOTHING: its only `hasContacts`
# assignment is commented out upstream (femClothPrimitives.cu:2737), exactly as its
# cloth-vs-cooked-mesh counterpart's is.  Its one observable output is an atomicMax, and
# that is what this shader implements; tools/check_disabled_kernels.py now GATES that
# reading rather than merely noting it.
#
# Reads everything through Ptr<> and lowers the warp ballot to a groupshared collective,
# so deliberately absent from PXG_HLSL_ENTRIES.  IS in PXG_NOCONTRACT_SHADERS: the plane
# distances, the clipping lerps and the barycentric solve are mul-feeding-add chains.
pxg_compile_shader(femClothHfContactGenBda cloth_heightfieldContactGenLaunch
                   cloth_heightfieldVertexContactGenLaunch)
# P8.19: the FEM-vs-FEM ATTACHMENT pair -- the prepare that scatters each attachment into the
# warp-strided PxgFEMFEMAttachmentConstraint block, and the solve that projects the two anchors
# together with a barycentric-weighted inverse-mass split.  Simpler than its P8.3 rigid sibling:
# both sides are position-based, so there is no velocity reader, no articulation branch and no
# rigid response term.  Neither kernel has a cross-lane operation, so both keep the CUDA
# [numthreads(256,1,1)] block verbatim.  Reads everything through Ptr<> so it is deliberately
# absent from PXG_HLSL_ENTRIES.  IS in PXG_NOCONTRACT_SHADERS: the barycentric interpolation and
# the inverse-mass weighting are mul-feeding-add chains.
pxg_compile_shader(softBodyFemAttachBda femAttachmentPrepareLaunch
                   sb_solveOutputSoftBodyAttachmentDeltaVLaunch)

# softBodyClothMidphaseBda — sb_clothMidphaseGeneratePairsLaunch (P8.8).  The BV32 walk over
# the CLOTH tree with a tet-bound query; the leaf test is the same TriangleLeafBoundMinMax
# traverser npTetTriMid ports for the descriptor set.  IS in PXG_NOCONTRACT_SHADERS: satIntersect
# is dot-of-cross throughout and the plane distances are mul-feeding-add chains.  Pure BDA, so
# deliberately absent from PXG_HLSL_ENTRIES.
# P9.89: sb_clothMidphaseSatProbe, a test-only entry that reads the tet-vs-triangle SAT's
# selectMin/selectMax folds and tetBoundingBox out bit for bit
# (softBodySimClothMidphase_p89SatProbe).
pxg_compile_shader(softBodyClothMidphaseBda sb_clothMidphaseGeneratePairsLaunch
                   sb_clothMidphaseSatProbe)

# softBodyClothContactGenBda — sb_clothContactGenLaunch (P8.8 step 2).  Tet-vs-triangle GJK/EPA
# through the shared driver, both barycentrics, the surface-face rejection and the two-level
# filter.  IS in PXG_NOCONTRACT_SHADERS (the barycentric determinants are dot-of-cross chains and
# the penetration is a mul-feeding-add).  Pure BDA, so absent from PXG_HLSL_ENTRIES.
pxg_compile_shader(softBodyClothContactGenBda sb_clothContactGenLaunch)

# softBodyClothVertMidphaseBda — sb_clothVertMidphaseGeneratePairsLaunch (P8.8 step 3).  The
# other half of the midphase: a cloth VERTEX queries the softbody tet tree, AABB against AABB.
# IS in PXG_NOCONTRACT_SHADERS (isValidTet is a dot-of-cross and the box deflation is a
# mul-feeding-add).  Pure BDA, so absent from PXG_HLSL_ENTRIES.
pxg_compile_shader(softBodyClothVertMidphaseBda sb_clothVertMidphaseGeneratePairsLaunch)

# softBodyClothVertContactGenBda — sb_clothVertContactGenLaunch (P8.8 step 4).  One thread per
# pair, point-vs-tetrahedron closest point, no GJK.  IS in PXG_NOCONTRACT_SHADERS (the closest
# point and the barycentric determinants are dot-of-cross chains).  Pure BDA, so absent from
# PXG_HLSL_ENTRIES.
pxg_compile_shader(softBodyClothVertContactGenBda sb_clothVertContactGenLaunch)

# softBodyClothPrepareBda — sb_clothContactPrepareLaunch (P8.8 step 5).  Turns the contacts of
# steps 2 and 4 into solver constraints; clothBC.w discriminates the triangle case from the
# vertex one.  IS in PXG_NOCONTRACT_SHADERS (the effective-mass sums and the penetration
# correction are mul-feeding-add chains).  Pure BDA, so absent from PXG_HLSL_ENTRIES.
pxg_compile_shader(softBodyClothPrepareBda sb_clothContactPrepareLaunch)

# softBodyClothSolveBda — sb_solveOutputSCDeltaVLaunch (P8.8 step 6, the last of the chain).
# Solves the constraints and scatters opposite-signed corrections onto both bodies through real
# float atomics.  IS in PXG_NOCONTRACT_SHADERS (the constraint residual and the friction
# projection are dot-feeding-mul chains).  Pure BDA, so absent from PXG_HLSL_ENTRIES.
pxg_compile_shader(softBodyClothSolveBda sb_solveOutputSCDeltaVLaunch)
pxg_compile_shader(softBodyRigidContactBda sb_rigidContactPrepareLaunch
                   sb_queryRigidSoftContactReferenceCountLaunch
                   sb_queryRigidSoftContactReferenceCountLaunchTGS
                   sb_solveRigidSoftCollisionLaunch
                   sb_solveRigidSoftCollisionLaunchTGS)
pxg_compile_shader(contactReduce ctContactReduce)
pxg_compile_shader(npSbTet ctSbTet)
pxg_compile_shader(npSbSelf ctSbSelfQuery)
pxg_compile_shader(npSbSb ctSbSbQuery)
# P9.89: ctTetTriSatProbe, a test-only entry that reads the tet-vs-triangle SAT's
# selectMin/selectMax folds out bit for bit (npTetTriMid_p89SatProbe), and ctTetTriBoundProbe,
# the same for tetBoundingBox (npTetTriMid_p89BoundProbe).
pxg_compile_shader(npTetTriMid ctTetTriMid ctTetTriSatProbe ctTetTriBoundProbe)
pxg_compile_shader(psPrimContact ctPsPrimContact)
pxg_compile_shader(psBounds ctPsBoundFirstPass ctPsBoundSecondPass ctPsReorderContacts ctPsGridHash)
# P6.4a-1: PBD particle bound / cell-range bookkeeping on the engine ABI (BDA).
# All four kernels read the PxgParticleSystem descriptor array (and the rigid
# narrowphase cmInputs/shapes/bounds mirrors) through Ptr<>, which has no HLSL
# cs_6_0 lowering, so this shader is deliberately absent from PXG_HLSL_ENTRIES
# (same precedent as particleSim / particleGridBda / particleSelfCollisionBda).
pxg_compile_shader(psBoundsBda ps_updateBoundFirstPassLaunch ps_updateBoundSecondPassLaunch
                   ps_primitivesBoundFirstPassLaunch ps_primitivesBoundSecondPassLaunch)
# P6.4a-2: the PBD particle-vs-primitive contact GENERATOR on the engine ABI (BDA).
# Reads the PxgParticleSystem descriptor array + rigid narrowphase mirrors through
# Ptr<> and writes the by-value PxgParticleContactWriter's device pointers, which
# has no HLSL cs_6_0 lowering, so this shader is deliberately absent from
# PXG_HLSL_ENTRIES (same precedent as particleSim / psBoundsBda).
pxg_compile_shader(psContactGenBda ps_primitivesCollisionLaunch ps_primitivesDiffuseCollisionLaunch ps_convexCollisionLaunch ps_convexDiffuseCollisionLaunch)
# P6.4b-1: the PBD particle contact ORGANIZATION half on the engine ABI (BDA) —
# reorder + 2-pass range-find.  Reads the contacts and remaps through Ptr<> and
# uses portable groupshared scans in place of warp ballots/shuffles, which have no
# HLSL cs_6_0 lowering, so this shader is deliberately absent from PXG_HLSL_ENTRIES
# (same precedent as particleSim / psBoundsBda / radixSortDesc).
# P9.48a adds the FEM twins of the two range-find passes, over
# PxgFemOtherContactInfo instead of PxgParticlePrimitiveContact.
pxg_compile_shader(psContactSortBda ps_reorderPrimitiveContactsLaunch
                   ps_findStartEndParticleFirstLaunch ps_findStartEndParticleSecondLaunch
                   ps_findStartEndFEMFirstLaunch ps_findStartEndFEMSecondLaunch)
# P6.4b-2: the PBD particle contact SOLVE half on the engine ABI (BDA) — contact
# constraint prepare + per-contact particle solve + per-particle deltaV accumulate.
# All three grid-stride via nbIterationsPerBlock (the launch grid is fixed at 1024
# PS_COLLISION / 32 ACCUMULATE_DELTA, no WorkgroupCount()) and read every buffer
# through Ptr<> (BDA), which has no HLSL cs_6_0 lowering, so this shader is
# deliberately absent from PXG_HLSL_ENTRIES (same precedent as particleSim /
# psContactSortBda / particleSelfCollisionBda).
# P9.48a adds ps_accumulateFEMParticleDeltaVLaunch, the FEM twin of the
# per-particle accumulate -- the consumer that finally MOVES a particle a cloth
# or softbody pushed.  Its reciprocal is fcount>0?1/fcount:0, NOT the particle
# twin's 1/max(1,fcount).
pxg_compile_shader(psContactSolveBda ps_contactPrepareLaunch
                   ps_solvePCOutputParticleDeltaVLaunch ps_accumulateDeltaVParticleLaunch
                   ps_accumulateFEMParticleDeltaVLaunch)
# P6.6: the three TGS-only kernels that finish the PBD particle contact SOLVE loop
# under the TGS solver — the TGS substep predictor (ps_stepParticlesLaunch), the
# TGS twin of the particle-side PC solve (ps_solvePCOutputParticleDeltaVTGSLaunch),
# and the TGS velocity-iteration finalize (ps_finalizeParticlesLaunch).  All three
# read the PxgParticleSystem descriptor array (and, for the solve, the solver
# core/shared/prep desc mirrors) through Ptr<> (BDA) and grid-stride via
# nbIterationsPerBlock (the solve) / SV_GroupID (step + finalize, fixed launch
# grid, no WorkgroupCount()), which has no HLSL cs_6_0 lowering, so this shader is
# deliberately absent from PXG_HLSL_ENTRIES (same precedent as particleSim /
# psContactSolveBda / particleSelfCollisionBda).
pxg_compile_shader(psContactSolveTgsBda ps_stepParticlesLaunch
                   ps_solvePCOutputParticleDeltaVTGSLaunch ps_finalizeParticlesLaunch)
# P9.44: the RIGID side of the particle contact solve (particlesystem.cu:1943/2123) and the
# deformable/particle -> rigid impulse accumulation (rigidDeltaAccum.cu, all five kernels).
# Both in PXG_NOCONTRACT_SHADERS: the reaction products and the scanned sums are compared
# word-for-word against host mirrors.
pxg_compile_shader(psContactSolveRigidBda ps_solvePCOutputRigidDeltaVLaunch ps_solvePCOutputRigidDeltaVTGSLaunch)
pxg_compile_shader(rigidDeltaAccumBda accumulateDeltaVRigidFirstLaunch accumulateDeltaVRigidSecondLaunch
                   clearDeltaVRigidSecondLaunchMulti accumulateDeltaVRigidSecondLaunchMultiStage1
                   accumulateDeltaVRigidSecondLaunchMultiStage2)
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
# P5 Step 4: full box-box narrow-phase entry on the engine ABI (BDA) — assembles
# the box-box primitives (npBox*) + the contact-output tail into boxBoxNphase_Kernel.
pxg_compile_shader(boxBoxNphase boxBoxNphase_Kernel)
# P5 Step 5 tranche B: sphere/capsule primitive narrow-phase dispatcher on the engine
# ABI (BDA) — dispatches into the shared npSpherePrimitives contact routines + the
# contact-output tail.  BDA-only, so deliberately absent from PXG_HLSL_ENTRIES.
pxg_compile_shader(npSphereNphase sphereNphase_Kernel)
# P5 Step 5 tranche C: full convex/box-vs-plane PCM narrow-phase entry on the
# engine ABI (BDA) — refresh + gen + out consolidated from the npCvxPlane*
# harness math into convexPlaneNphase_Kernel.  Vulkan-only (BDA + warp->lane-0
# serialization do not lower to HLSL cs_6_0), so absent from PXG_HLSL_ENTRIES.
pxg_compile_shader(npCvxPlaneNphase convexPlaneNphase_Kernel)
pxg_compile_shader(broadphaseSetup translateAABBsLaunch markCreatedPairsLaunch initializeSapBox1DLaunch
                   initializeRadixRanks updateHandles)
pxg_compile_shader(broadphaseReports accumulateReportsStage_1 accumulateReportsStage_2 copyReports)
pxg_compile_shader(broadphaseIncrementalHistogram computeIncrementalComparisonHistograms_Stage1
                   computeIncrementalComparisonHistograms_Stage2)
pxg_compile_shader(broadphaseEndPtHistogram computeEndPtsHistogram outputEndPtsHistogram)
pxg_compile_shader(broadphaseIncrementalSap performIncrementalSAP)
pxg_compile_shader(broadphaseRegionHistogram computeRegionsHistogram outputRegionsHistogram
                   computeStartRegionsHistogram outputStartRegionsHistogram)
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
pxg_compile_shader(npRobustProbe robustProbeRead)
# RUNTIME_PLAN.md P2.1: warp-32 primitive library battery (emulation path).
pxg_compile_shader(npWarpLib wlShuffle wlBallot)
# P2.1 native-path compile/lowering validation (WaveShuffle/WaveActiveBallot).
pxg_compile_shader(npWarpNative warpNativeProbe)
# RUNTIME_PLAN.md P6: indirect dispatch plumbing.
pxg_compile_shader(npIndirect writeArgs fillGrid)
# P9.89: alAtomicSelect drives atomic.cuh's select rule (resident +-0, NaN cells).
pxg_compile_shader(npAtomicLib alAtomics alAtomicSelect)
# P9.33a: a PLATFORM measurement rather than a kernel port.  About forty shaders here spell
# PxVec3::getNormalized as `v * rsqrt(m)`, but the engine computes PxRecipSqrt = 1.0f/sqrtf
# -- an exactly-specified pair of IEEE operations -- while SPIR-V's InverseSqrt is allowed
# 2 ULP of slack.  Every bit-exact fixture running through one of those sites has been
# resting on an unmeasured property of the driver; this shader computes both spellings side
# by side so the fixture can compare them, and compare the division form against the host's.
# NOT in PXG_NOCONTRACT_SHADERS, and not by oversight: the whole point is to observe what
# the compiler and driver do to these two expressions, so nothing here may be constrained.
pxg_compile_shader(npFloatIntrin floatIntrinRecipSqrtLaunch)
# RUNTIME_PLAN.md P2: tex3D replacement (Vulkan-only sampled-image path).
pxg_compile_shader(npTex3d tex3dSample)
pxg_compile_shader(npTex3dBindless tex3dBindlessSample)
pxg_compile_shader(npSdfTexture sdfTexProbe sdfTexSampleAt)
# P9.34a adds the two entries the cloth-vs-SDF contact gen needs: the barycentric clamp
# (Gu::closestPtPointBaryTriangle, the one function on this page with a real host oracle)
# and doTriangleSDFCollision, the 16-iteration projected-gradient descent that is the
# whole numeric content of cloth_sdfMeshContactGenLaunch's per-triangle work.
pxg_compile_shader(npSdfCollision sdfCollDistance sdfCollGrad sdfCollSampleGrad sdfCollVertex
                   sdfCollBaryClamp sdfCollTriangle
                   # P9.35a: the TETRAHEDRON pair.  sdfCollTetClamp is a second ORACLE --
                   # Gu::closestPtPointTetrahedron is PX_CUDA_CALLABLE and texture-free, so
                   # the fixture calls the engine itself.  sdfCollTetrahedron traces the
                   # four-vertex descent, with one field the triangle trace has no use for:
                   # its projection is CONDITIONAL, so whether it ran is part of the claim.
                   sdfCollTetClamp sdfCollTetrahedron)
# P9.34b adds the TRIANGLE gather beside the vertex one, so the two resumable streams
# can be driven identically and compared round for round -- the triangle form has a
# count guard the vertex form does not, which is the difference under test.
pxg_compile_shader(npSdfFind sdfFindRounds sdfFindTriangleRounds sdfFindTetRounds)
pxg_compile_shader(npSdfParticle sdfParticleContactAt)
# RUNTIME_PLAN.md P3 shim proof kernel (nested BDA params).
pxg_compile_shader(npShimAxpy shimAxpy)
# P5 Step 4: PxgBroadPhaseDesc mirror layout proof (BDA).
pxg_compile_shader(npBpDescProbe bpDescProbe)
# P5 Step 4: narrowphase element-struct mirror layout proof (BDA).
pxg_compile_shader(npNpAbiProbe npAbiProbe)
# P5 Step 4: PxgSolverCoreDesc / PxgSolverSharedDesc<IterativeSolveDataTGS> /
# PxgIslandContext mirror layout proof (BDA).
pxg_compile_shader(npSolverDescProbe solverDescProbe)
# P5 Step 4: empirical layout proof for the simulation-controller desc mirrors.
pxg_compile_shader(npSimCoreDescProbe simCoreDescProbe)
# P5 Step 4: PxgPrePrepDesc / PxgConstraintPrepareDesc /
# PxgSolverSharedDesc<IterativeSolveData> + indexed-element mirror layout proof (BDA).
pxg_compile_shader(npPrepDescProbe prepDescProbe)
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
    testFrozenHistogram.cpp
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
    testNpDefFilterBda.cpp
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
    testPsSprings.cpp
    testPsAerodynamics.cpp
    testPsInflatable.cpp
    testPsBoundaryDensity.cpp
    testPsMeshMidphase.cpp
    testPsHfMidphase.cpp
    testPsSolveSprings.cpp
    testPsAverageVerts.cpp
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
    testDmaArtiResidual.cpp
    testArtiBlockLayout.cpp
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
    testPsRigidAttach.cpp
    testDirectApiBda.cpp
    testPsShapeMatch.cpp
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
foreach(suite scan radixSort radixSortDesc radixSortGather preIntegration integration updateBounds frozenHistogram memCopy clamp skinning
        scan4x4 gpuRadixSort reorderKernel bodyUpdate directApi setRigidDynamicGlobalPose solverPGS solverTGS npCrsWrapper2 npTriTriReduce npTriTriWrite npTriTriBlock npTriTriGeom npTriTriBody npTriTriFind npTriTriTail npTriTriKernel
        constraintPrepPGS constraintPrepTGS thresholdStream thresholdForceChange solverZeroBodies solverInitVelocitiesTGS solverApplyGravityTGS solverDmaBack npPairManagement npPairRemoveSwap npSphereCollision npCapsuleCollision npMaterialCombine npContactPatch npInsertPatch npCompressContacts npCompressContacts2 npTriangle npSegTriangle npCapTriangle npTriTri npHeightfield npHfMidphase npHfPointQuery npConvexTri npConvexSat npConvexFaceSat npConvexEdgeSat npConvexFeature npConvexTriOut npConvexPolyClip npCvxTriEmit npCvxTriFind npCvxTriPurge npCvxTriWorldNormal npGjkSupport npGjkNewPoint npGjkBarycentric npHfTriFlags npWarpVec npGjkSimplex npGjkClosestTri npGjkClosestTet npGjkClosestSeg npGjkOutput npGjkPermute npGjkSupportStep npGjkWarmStart npGjk npEpaStart4 npEpaStart123 npEpa npGjkWarm npCvxCvxEE npCvxRayPoly npCvxFaceContacts npCvxReduceManifold npCvxWitnessPoly npCvxLocalNormal npCvxAddManifold npCvxBlob npCvxBlobRead npCvxEEDriver npCvxRefFrame npCvxPolyClip npCvxClipEmit npCvxContactDecide npCvxCollidePrep npCvxCollideDecide npCvxLostFound npCvxWriteStream npCvxFinishContacts npCvxStage1Refresh npCvxStage2Setup cvxCollide npCvxPlanePolyIndex npCvxPlaneRefresh npCvxPlaneGen npCvxPlaneOut npCvxCorrelate npCvxPostProcess npCvxTrimeshOut npTriTriOverlap npCvxTriNpSetup npCvxHfNpSetup npCvxCoreOut npCvxCoreSetup npCvxCoreSupport npCvxCoreFace npCvxCorePtsFace npCvxCoreClip npCvxCoreContact npCvxCoreGjk npCvxCoreEpa npCvxCorePlaneGen npCvxCoreCvxGen npCvxBvhGlue npCvxCoreWarpAcc cvxCoreTailBda p86SignedZero npD6JointPrep npRemapIndex npContactPrePrep npRigidSumBatches npArtiExtBody npArti1DPrep npArtiJointHeader npArtiSolve1D npArtiSolveContact npArtiSolve1DTGS npArtiSolveContactTGS npArtiCheckActive npSlabRefCount npArtiPropagate npArtiPropagateAccel npArtiPropagateAccelJV npArtiSpatialMatVec npArtiBitfield npArtiSpatialMatMat npArtiConstructSM npArtiTranslateInertia npArtiTranslateSInertia npArtiPropagateZA npArtiPropagateZASph npArtiTranslateInertiaMass npArtiInvertInertia npArtiPropagateImpDof npArtiAccumLinkImp npArtiDriveParams npArtiClampDriveImp npClothDefGrad2D npSbDefGrad npDefMaterial npGridIndex npJacobiRotate npClothAtan npAnisoHelpers npSparseGridHash npClothDeltaLambda npClothExtractRot2D npClothVelClamp npClothArap2D npClothArea2D npClothWind npSbBaryInvMass npSbVolumeSolve npIsoConcaveCorner npSphSpikyKernel npIsoAcosDecode npSparseGridDomain npDenseGridCell npSbBaryProject npIsoVecNormalize npSbCompliance npSbDeltaLambdaGradC npSbTetConstraint npPsSpecVolume npSbComputeLambda npGridLocation npAnisoOutput npSmoothPos npIsoAvgVerts npPsSolvePC npPsSolvePCTGS npPsWeightKernels npPsDiffuseUpdate npPsDiffuseSpawn npPsDiffuseAvg npPsDiffuseOneWay npSdfBoxDist npSparseGridLookup npSparseGridCellFromPos npIsoGridFilter npSbExtractRotAPD npSbSolveNeoHookean npSbSolveCorot npSbSolveGM npClothMembrane npClothBending npClothSharedEnergy npSbCollisionConstraint npClothCollisionConstraint npSdfBvhMorton npSbStress npSdfHoles npClothTriContact npManifold npSegDist npTriMask npNpUtils npConvexReduce npTriMesh npClosest npTetra npEdgeEdge npDefFilter npDefFilterBda softBodyPsContactGen softBodyPsPrepare npBv32 npMidphaseCvx npMidphaseDeform npMidphaseAlloc npClothCloth gridCal particleSim femClothSim cvxCoreClothNp softBodySim particleContactSort femContactSort pcmManifold convexTrimeshMidphase convexHfMidphase midphaseAlloc triFeature deferContact initManifolds pairMgmt polyClip convexHull convexSat convexSelect particleGrid contactReduce cvxTriEmit cvxTriSetup sphereTriNp cvxHfSetup cvxHfPostProcess sphereHfCore hfTriFlagsBda segTriangleBda capTriangleBda capsuleHfCore radixWarp sortTriIndices cvxTriPostProcess cvxTriCorrelate cvxTriFinish npSbTet npSbSelf npSbSb npTetTriMid psPrimContact psBounds psContactGen psContactSort psContactSolve psContactSolveTgs psContactSolveRigid psRigidAttachPrepare psRigidAttachSolve directApiRigidGet directApiD6JointGet directApiRadixCopy2 directApiMergeAabb directApiUpdateBodies psVolumeBound psShapeMatch rigidDeltaAccum psPrimCollide psDiffuseCollide fcPrim psMeshMid psHfCollide fcPrim2 npPolyClip4 fcConvexTri fcVertexConvex fcStaticTriCloth fcMidphase npSbPrim npSbRemap npCvxHfMid npTriPlaneNp npTriHfNp npCvxMeshNp npCvxHfNp npBoxClip npBoxIncident npBoxContacts npBoxSat npBoxFeature npBoxReduce boxBoxNphase sphereNphase convexPlaneNphase broadphaseSetup broadphaseReports
        broadphaseIncrementalHistogram broadphaseEndPtHistogram broadphaseIncrementalSap
        broadphaseRegionHistogram broadphaseCreateRegions broadphaseActiveRegionHistogram
        broadphaseOverlapCheckHistogram broadphaseNewBoundsPairs broadphaseMarkPairs
        broadphaseClearNewFlag broadphaseAggregateBounds broadphaseAggregateDirty
        broadphaseAggregatePairsAdd broadphaseAggregateFinalize broadphaseAggregatePairsRemove
        broadphaseAggregateBoxPruning broadphaseAggPairCollision broadphaseAggSelfCollision broadphaseAggSortProjections
        p0Subgroup p0Bda robustProbe p1DeviceV2 p1HeapAlloc npWarpLib npAtomicLib npFloatIntrinRecipSqrt npTex3d npTex3dBindless p3ShimTexture3d shimBindless shimOrdering sdfTexture sdfCollision sdfTriangleCollision sdfTetrahedronCollision sdfFind sdfFindTriangles sdfFindTets sdfParticle psSdfMidphase psSdfMidphaseDiffuse femClothSdfContactGen femClothSdfContactGenExactGate femClothSdfContactGenNegScale softBodySdfMidphase softBodySdfMidphaseExactGate softBodySdfMidphaseNegScale warpNative p1PipelineCache p1Timestamps p6Indirect p3Shim p3CudaContext p3UnmappedLaunch p3CudaContextManager p3KernelWrangler p3ContextHardening p4Manifest p4Loader p4Foundation p4Heap p4Wrangler p4Broadphase p4Aabb solverDescLayout simCoreDescLayout prepDescLayout npAbiLayout particleSelfCollision newArticulations dmaArtiResidual artiBlockLayout artiJointFrame artiJcalcPhaseA artiJcalcPhaseB1 artiJcalcPhaseB2 artiJcalcPhaseB3 artiJcalcPhaseC artiJcalcTendons artiJcalcGpuDirty artiJcalcInit artiJcalcLinkVel artiUnconstrainedVel artiSpatialInertia artiSpatialInertiaLaunch artiSpatialMatrix artiSpatialMatrixBlockIo artiPropagateInertia artiArticulatedInertia artiArticulatedInertiaLaunch artiDynamic artiMassMatrixWrite artiMassMatrixZeroWalk artiMassMatrixBody artiMassMatrixLaunch artiJointFriction artiTranslateInertiaMass artiLinkAcceleration artiAccelerationLaunch artiDmaBack artiStepAlgebra artiStepQuatProbe artiStepRotVecProbe artiStepSphGuardProbe artiFlushVelocity artiAvgImpulse artiStepKernel artiSleepCheck artiUpdateBodies2 artiUpdateBodies3 artiUpdateBodies1T artiSumBatches1 artiSumBatches2 artiSumSelfBatches artiSetupInternal artiSetupTendons artiSetupMimic artiSetupICLaunch artiSolveVelImp artiSolveBodies artiSolveTMLaunch artiSolveExtTGS artiSolveSelf artiSolveSelfLaunch artiSolveStatic artiSolveDof artiSolveDFS artiSolveInternalLaunch artiAvgImpulse2 artiPropagate2 artiConstraintPrepBody artiConstraintPrepSetup artiConstraintPrepWrap artiJointPrepWrap artiSolvePartitionWrap artiOutputVelocity artiPushImpulse artiApplyTgsSubstepForces artiSaveVelocityPGS artiComputeDeps artiComputeCOM artiInverseDyn2 artiGravityForces artiCentrifugal artiMassMatrices artiUpdateKinematic artiGetSet artiJointPrepPgs artiConstraintPrepPgs artiSolveExtPgs artiSolvePartitionPgs artiSolveSelfPgs artiSolveSelfLaunchPgs artiSolveStaticPgs artiSolveDofPgs artiSolveDFSPgs artiSolveInternalLaunchPgs updateArticulations psMaterials psFluidDensity psSolveDensity psFluidVelocity psSprings psSolveSprings psAverageVerts psAerodynamics psInflatable psBoundaryDensity psMeshMidphaseLayout psMeshMidphase psMeshMidphaseDiffuse psMeshMidphaseFilter psMeshMidphaseWorkIndex psHfMidphaseLayout psHfMidphase psHfMidphaseDiffuse psHfMidphaseClobber psHfMidphaseAccum psDiffuseBuffers psDiffuseCreate psDiffuseAdvect psDiffuseOneWay psDiffuseContactGen psConvexContactGen psConvexDiffuseContactGen bpDescLayout npTetCollide)
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
    # builtin (P0 policy).  dmaConstraintResidual still lowers to cs_6_0.
    "solverDmaBack:dmaConstraintResidual"
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
    COMMAND python3 ${GPUSLANG_ROOT}/tools/check_barrier_uniformity.py)

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
    COMMAND python3 ${GPUSLANG_ROOT}/tools/check_cuda_twin_kernels.py)

# P9.23: BDA reads whose only protection is a per-lane guard.  This platform
# turns such a guard into a mask and issues the load anyway, so the out-of-range
# address is still formed and dereferenced -- which is what P9.22 measured as a
# crash.  No fixture can see it (the value is discarded), no mutant can (same
# reason), and spirv-val cannot (the SPIR-V is valid; the speculation happens in
# the driver's back end).  This check is the only thing in the build that can.
add_test(NAME gpuslang.speculatedLoads
    COMMAND python3 ${GPUSLANG_ROOT}/tools/check_speculated_loads.py)

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
    COMMAND python3 ${GPUSLANG_ROOT}/tools/check_disabled_kernels.py)

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
    COMMAND python3 ${GPUSLANG_ROOT}/tools/check_vector_divide.py)

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
    COMMAND python3 ${GPUSLANG_ROOT}/tools/check_component_divide.py)

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
    COMMAND python3 ${GPUSLANG_ROOT}/tools/check_length_intrinsic.py)

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
    COMMAND python3 ${GPUSLANG_ROOT}/tools/check_subword_store.py)

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
    COMMAND python3 ${GPUSLANG_ROOT}/tools/check_groupshared.py ${SHADER_OUT_DIR})

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
    COMMAND python3 ${GPUSLANG_ROOT}/tools/check_launch_abi.py)

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
    COMMAND python3 ${GPUSLANG_ROOT}/tools/check_helper_uniformity.py)

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
    COMMAND python3 ${GPUSLANG_ROOT}/tools/check_geom_type_constants.py)

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
    COMMAND python3 ${GPUSLANG_ROOT}/tools/check_grid_stride.py)

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
    COMMAND python3 ${GPUSLANG_ROOT}/tools/check_fixture_grid.py)

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
    COMMAND python3 ${GPUSLANG_ROOT}/tools/check_unmapped_reachability.py)

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
    COMMAND python3 ${GPUSLANG_ROOT}/tools/check_block_coverage.py)

# P9.77, the FIFTEENTH checker, and the THIRD written because its defect had already
# happened.  gpuslang.launchAbi (P9.48d) asks whether a mapped kernel HAS an ABI row;
# it cannot ask whether the row MATCHES.  P9.76b is the difference: the softbody
# attachment TGS solve carried the 72-byte PGS row while its launch site pushes 80,
# so the shim's exact blobSize == pushConstantSize gate refused every launch and the
# kernel had never run on Vulkan -- with fixtures, the suite and all 48 scenes green,
# because a fixture dispatches the shader directly and never passes the gate.
# This one derives the blob from the PX_CUDA_KERNEL_PARAM list at the launch site and
# compares.  141 kernels decided, 99 in a ratcheted undecided set (mostly by-value
# structs whose size wants the struct read).
add_test(NAME gpuslang.launchAbiPack
    COMMAND python3 ${GPUSLANG_ROOT}/tools/check_launch_abi_pack.py)

# P9.78: the SIXTEENTH checker.  pushConstantSizeFor() derives the pushed blob
# size from the ABI ROW and never from the SPIR-V, so a shader whose push block
# disagrees with its row is not refused -- it reads bytes that were never pushed,
# silently.  This walks each entry point's call graph and compares the highest
# push-constant byte it can reach against its row.  It needs the built .spv, so
# it depends on the shader target like the other SPIR-V readers.
add_test(NAME gpuslang.pushBlock
    COMMAND python3 ${GPUSLANG_ROOT}/tools/check_push_block.py ${SHADER_OUT_DIR})
