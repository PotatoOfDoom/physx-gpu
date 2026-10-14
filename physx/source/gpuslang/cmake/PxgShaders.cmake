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

# SPIR-V compilation for the Slang port, shared by the verification suite
# (gpuslang/tests) and the official Vulkan PhysXGpu build
# (source/compiler/cmakegpuvulkan).  One definition of the slangc command, its
# flags and the SPIR-V post-processing, so the shaders the SDK ships are compiled
# exactly as the ones the suite validates.
#
# The caller sets, before include():
#   GPUSLANG_ROOT    physx/source/gpuslang
#   SHADER_OUT_DIR   where the .spv files are written
#   PXG_SHADER_ONLY  (optional) the stems to build; pxg_compile_shader ignores any
#                    other stem.  Unset = build every shader (the suite).
#
# After include(): pxg_compile_shader(<stem> <entry>...) appends each output to
# PXG_SHADER_OUTPUTS and records the stem's entry list in the global property
# PXG_SHADER_ENTRIES_<stem>.

cmake_policy(SET CMP0057 NEW) # IN_LIST

# ---- Tools ----
# slangc: -DPXG_SLANGC=<path> wins; otherwise $SLANG_DIR/bin and $SLANG_DIR (a Slang
# release), then PATH, then the Vulkan SDK (which bundles slangc in its Bin dir).
set(_pxgSlangHints "")
if(DEFINED ENV{SLANG_DIR} AND NOT "$ENV{SLANG_DIR}" STREQUAL "")
    set(_pxgSlangHints "$ENV{SLANG_DIR}/bin" "$ENV{SLANG_DIR}")
endif()
set(_pxgSlangPaths "")
if(DEFINED ENV{VULKAN_SDK} AND NOT "$ENV{VULKAN_SDK}" STREQUAL "")
    set(_pxgSlangPaths "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
endif()
find_program(PXG_SLANGC slangc
    HINTS ${_pxgSlangHints}
    PATHS ${_pxgSlangPaths}
    DOC "Path to the slangc compiler")
# A cached path is not searched again, so check it exists -- a stale or mistyped
# -DPXG_SLANGC would otherwise configure fine and fail on the first shader.
if(NOT PXG_SLANGC OR NOT EXISTS "${PXG_SLANGC}")
    message(FATAL_ERROR
        "slangc not found (PXG_SLANGC='${PXG_SLANGC}').\n"
        "The Slang/Vulkan GPU build compiles its shaders with slangc. Install the LunarG\n"
        "Vulkan SDK (it ships slangc in %VULKAN_SDK%\\Bin on Windows, $VULKAN_SDK/bin on Linux) or a Slang\n"
        "release from https://github.com/shader-slang/slang/releases, then put slangc on\n"
        "PATH, set SLANG_DIR to the release directory, or pass -DPXG_SLANGC=/path/to/slangc.")
endif()
if(NOT DEFINED PXG_SLANGC_VERSION_CHECKED OR NOT PXG_SLANGC_VERSION_CHECKED STREQUAL PXG_SLANGC)
    execute_process(COMMAND "${PXG_SLANGC}" -v
        RESULT_VARIABLE _pxgSlangcResult
        OUTPUT_VARIABLE _pxgSlangcVersion ERROR_VARIABLE _pxgSlangcVersionErr
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE)
    if(NOT _pxgSlangcResult EQUAL 0)
        message(FATAL_ERROR "'${PXG_SLANGC} -v' failed (${_pxgSlangcResult}): ${_pxgSlangcVersionErr}")
    endif()
    if(NOT _pxgSlangcVersion)
        set(_pxgSlangcVersion "${_pxgSlangcVersionErr}")
    endif()
    # The port is developed and validated with this slangc; another version may
    # emit different (still valid) SPIR-V, which the suite has not checked.
    set(PXG_SLANGC_EXPECTED_VERSION "2026.14.1")
    if(NOT _pxgSlangcVersion STREQUAL PXG_SLANGC_EXPECTED_VERSION)
        message(WARNING "slangc ${_pxgSlangcVersion} (${PXG_SLANGC}); the Slang port is validated with "
            "${PXG_SLANGC_EXPECTED_VERSION}. Other versions are expected to work but are untested.")
    else()
        message(STATUS "slangc ${_pxgSlangcVersion}: ${PXG_SLANGC}")
    endif()
    set(PXG_SLANGC_VERSION_CHECKED "${PXG_SLANGC}" CACHE INTERNAL "slangc whose version was checked")
endif()

# The SPIR-V post-processors (tools/spv_*.py) are stdlib-only Python 3.
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# -O0 keeps slangc from invoking the external spirv-opt pass (which needs the
# optional slang-glslang shared library).  The SPIR-V still passes spirv-val
# and drivers run their own optimizers; bump this when a full slang
# distribution is available.
# -fvk-use-entrypoint-name keeps the source entry-point names in the SPIR-V
# (slangc otherwise renames a single entry point to "main").
set(PXG_SLANG_FLAGS -O0 -fvk-use-entrypoint-name CACHE STRING "Extra flags passed to slangc")

set(SHADER_DIR ${GPUSLANG_ROOT}/shaders)
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
    # P9.96: the pxgSlangTex3d fixture joins.  pxgTex3D is now an explicit linear
    # filter -- eight texel products and seven adds, r = r + w*t -- compared BIT FOR
    # BIT against tests/pxgTex3dRef.h (built -ffp-contract=off), and every ENGINE
    # importer of the module (sdfQueryBda, psSdfMidphaseBda, softBodySdfMidphaseBda,
    # femClothSdfMeshCgBda, triTriCollisionBda, and the npSdf* fixtures) is already
    # stamped above.  Unstamped, llvmpipe may fuse the chain into fmas, and the
    # fixture would certify a build no engine kernel runs.
    npTex3dBindless
    # P9.83b-8: the mesh-mesh kernel itself, stamped like every fixture that
    # certifies one of its modules (the promise npTriTriGeom's entry made above).
    triTriCollisionBda
    # P9.95: evaluatePointDistancesSDFBatch.  It composes pxgSlangSdfCollision, which
    # every other importer (npSdfCollision, psSdfMidphaseBda, femClothSdfMeshCgBda,
    # softBodySdfMidphaseBda) compiles stamped, and its one new expression --
    # normalizeSafe's sqrt(x*x + y*y + z*z) -- must not fuse either: testSdfQuery
    # compares the normalised gradient bitwise against the engine's PxVec3.
    sdfQueryBda
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
    skinning
    # P9.90: the rigid pre-integration (PGS and TGS entries).  The velocity-clamp
    # magnitudes angVelSq / linVelSq (preIntegration.cuh:117/:123) and the gyroscopic
    # newMom/origMom magnitudes (:315-316) are PxVec3 dots, (x + y) + z (PxVec3.h:173/:274);
    # unstamped, llvmpipe summed them x + (y + z).  Its only import, pxgSlangCommon,
    # contributes explicit mul+add chains (quatRotate, mat33FromQuat,
    # transformInertiaTensor), and no other kernel is compared against this one.
    # testPreIntegration preIntegration_p90DotOrder compares every output word bitwise
    # and fails without this entry.
    preIntegration
    # P9.90: the convex-convex stage-1 manifold refresh.  The refreshed separation
    # dist = v.dot(localNormal), stored as mLocalNormal_PenW[j].w, and the keep test's
    # projectedDifference.magnitudeSquared() (cudaGJKEPA.cu:2278/:2286) are PxVec3 dots,
    # (x + y) + z (PxVec3.h:173/:274); unstamped, llvmpipe summed them x + (y + z).
    # Its only import, pxgSlangNpAbi, contributes explicit mul+add chains (quatRotate,
    # quatMul) and no other kernel is compared against this one.
    # testNpCvxStage1Refresh npCvxStage1Refresh_p90DotOrder compares the refreshed
    # manifold bitwise and fails without this entry.
    npCvxStage1Refresh
    # P9.90: the convex-convex finishContactsKernel.  The group-average normal's
    # magnitudeSquared (cudaGJKEPA.cu writeContactsToStream4threads, PxVec3.h:173) is a
    # PxVec3 dot, (x + y) + z; unstamped, llvmpipe summed it x + (y + z) and the patch
    # normal (PxContactPatch words 4..6) came out re-associated.  Its only import,
    # pxgSlangNpAbi, contributes explicit mul+add chains (quatRotate), and no other
    # kernel is compared against this one.  testNpCvxFinishContacts
    # npCvxFinishContacts_p90DotOrder compares the patch normals bitwise and fails
    # without this entry.
    npCvxFinishContacts
    # P9.90: the TGS substep gravity kernel.  applyAcceleration's sqrtInvInertia.getInverse()
    # (solverMultiBlockTGS.cu:1592) takes the determinant column0.dot(column1.cross(column2))
    # (PxMat33.h:247), a PxVec3 dot (x + y) + z; unstamped, llvmpipe summed the port's OpDot
    # x + (y + z) and 1/det shifted every angular-acceleration lane.  No other kernel is
    # compared against it.  testSolverApplyGravityTGS solverApplyGravityTGS_p90DotOrder
    # compares the velocity buffer bitwise and fails without this entry.  P9.95: rewritten
    # on the engine ABI; the shader now imports pxgSlangCommon / pxgSlangSolverDesc (layout
    # mirrors and an integer node-index helper, no float math), spells the determinant dot
    # out explicitly as (x + y) + z, and every fixture in the file is bitwise.
    solverApplyGravityTGS
    # P9.90: the CPU->GPU body mirror, stamped for consistency.  Its only float math is
    # updateBodiesLaunchDirectAPI's body2World recompute (PxAlignedQuat rotate/rotateInv
    # and the Hamilton product, AlignedQuat.h:257-366), scalar mul+add chains spelled
    # left to right with no OpDot, so llvmpipe's OpDot association never applies and it
    # did not contract the chains either: the unstamped .spv already matched the
    # -ffp-contract=off mirror on all 4096 poses.  The stamp pins that order against a
    # driver that would fuse.  It imports only pxgSlangCommon/pxgSlangSimCoreDesc and no
    # other kernel is compared against it.  testBodyUpdate
    # bodyUpdate_p90PoseRecomputeOrder compares the recomputed body2World bitwise (its
    # old-order twin must differ); it passes unstamped too, so the entry is unobserved.
    bodyUpdate
    # P9.90: the Phase-1 Direct-GPU-API getters/setters, stamped for consistency with
    # their engine twin directApiBda (stamped above; the two are never compared against
    # each other).  The float math is getRigidDynamicGlobalPose's rotateInv/rotate and
    # Hamilton product (AlignedQuat.h:257-366) and setRigidDynamicTorque's
    # transformInertiaTensor + PxMat33::transform (CmUtils.h:57-70, PxMat33.h:344),
    # scalar/vector mul+add chains spelled left to right with no OpDot, so llvmpipe's
    # OpDot association never applies and it did not contract the chains: the
    # unstamped .spv already matched the -ffp-contract=off mirror on all 4096 rows.
    # The stamp pins that order against a driver that would fuse.  testBodyUpdate
    # directApi_p90PoseTorqueOrder compares pose and angular acceleration bitwise (its
    # old-order twins must differ); it passes unstamped too, so the entry is unobserved.
    directApi
    # P9.90: the transform-cache/bounds kernels, stamped for consistency.  The float
    # math is getAbsPose (PxTransform::transformInv/transform, PxQuat rotate/rotateInv
    # and the Hamilton product), PxMat33(q), basisExtent, PxMeshScale::toMat33, the
    # capsule extents and transformFast (updateCacheAndBound.cuh), all scalar/vector
    # mul+add chains spelled left to right with no OpDot, so llvmpipe's OpDot
    # association never applies and it did not contract the chains: the unstamped .spv
    # already matched the -ffp-contract=off mirror bitwise.  The other importers of
    # pxgSlangCacheBound -- directApiBda, npArtiUpdateKin and (P9.94c)
    # convexTrimeshMidphaseBda -- are stamped too.  testUpdateBounds
    # setRigidDynamicGlobalPose_p90PoseOrder compares cache, bounds and body2World
    # bitwise (its old-order twin must differ); it passes unstamped too, so the entry is
    # unobserved.
    updateBounds
    # P9.90: the softbody-particle attachment solve, stamped for consistency.  Its float
    # math (softBodyGM.cu:2580-2604) is the barycentric interpolation pos1 = sum sp_i*b_i,
    # wTot1 = sum b1w_i*b_i, error*(1/wTot) and the scatters' delta*weight products, all
    # scalar/vector mul+add chains spelled left to right with no OpDot (the cone-limit
    # dots of pxgSlangFemFemAttach are never called: E1), so llvmpipe's OpDot association
    # never applies and it did not contract the chains: the unstamped .spv already matched
    # the -ffp-contract=off mirror bitwise.  The stamp pins that order against a driver
    # that would fuse.  The other engine importers of pxgSlangFemFemAttach
    # (softBodyFemAttachBda, softBodyClothAttachBda, femClothAttachBda) were already
    # stamped, and no kernel is compared against this one.  testSoftBodySim
    # softBodySimParticleAttachmentSolve_p90BaryOrder compares both delta buffers bitwise
    # (its reassociated twin must differ); it passes unstamped too, so the entry is
    # unobserved.
    softBodyParticleAttachBda
    # P9.90: the force-threshold accumulation (accumulateThresholdStream.cu), stamped for
    # consistency.  Its float math is additions only (the scanned force prefixes, spelled
    # in the CUDA's order since P9.86f) and one product, threshold * dt (:440), which feeds
    # only the `accumulatedForce >` compare and never an add.  There is no OpDot and no
    # mul feeding an add, so neither llvmpipe's OpDot association nor contraction can
    # reach an output: the stamped and unstamped .spv give the same testThresholdStream
    # results, and no fixture can observe the entry.  No other shader compiles this module.
    thresholdStream
    # P9.91: its engine-ABI twin, the file the manifest maps -- the same bodies and the same
    # argument: additions in the CUDA's order and one threshold * dt that feeds a compare.
    thresholdStreamBda
    # P9.92: the aggregate broadphase.  Its only float arithmetic is updateAggregateBounds'
    # `bound -+ contactDistance` feeding PxMin/PxMax selects -- no OpDot, no mul, so nothing
    # a stamp could change; stamped for consistency with every other float-computing shader.
    aggregateBda)


# pxg_compile_shader(<file-stem> <entry> [<entry> ...])
function(pxg_compile_shader stem)
    if(DEFINED PXG_SHADER_ONLY AND NOT stem IN_LIST PXG_SHADER_ONLY)
        return()
    endif()
    set_property(GLOBAL PROPERTY PXG_SHADER_ENTRIES_${stem} ${ARGN})
    set(entries "")
    foreach(entry ${ARGN})
        list(APPEND entries -entry ${entry})
    endforeach()
    set(postprocess "")
    if(stem IN_LIST PXG_NOCONTRACT_SHADERS)
        set(postprocess COMMAND ${Python3_EXECUTABLE} ${GPUSLANG_ROOT}/tools/spv_nocontract.py ${SHADER_OUT_DIR}/${stem}.spv)
    endif()
    # P9.86: every shader demands IEEE signed zeros (OpExecutionMode
    # SignedZeroInfNanPreserve), as the CUDA computes -- except the probe that
    # measures what the mode changes.
    if(NOT stem STREQUAL "p86SignedZero")
        list(APPEND postprocess COMMAND ${Python3_EXECUTABLE} ${GPUSLANG_ROOT}/tools/spv_szinp.py ${SHADER_OUT_DIR}/${stem}.spv)
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
