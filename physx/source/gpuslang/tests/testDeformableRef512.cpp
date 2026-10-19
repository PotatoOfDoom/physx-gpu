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

// testDeformableRef512 -- the PhysX 5.12 deformable / particle shared Slang modules,
// measured against the REAL 5.12 headers (gpusimulationcontroller/include) on the host:
//
//   deformableRef512Layout   every field of the descriptor mirrors (pxgSlangFEMClothDesc,
//                            pxgSlangSoftBodyDesc, pxgSlangParticleDesc), read on the
//                            device from element [1] of a two-element array of the real
//                            engine struct filled with a position-dependent byte pattern,
//                            so the field offsets AND the array stride are measured; plus
//                            sizeof / offsetof of every block, writer and record the
//                            shared modules spell as byte constants.
//   deformableRef512Encoding pxgSlangDeformableRef against PxgEncode* / PxgGet* /
//                            PxgDeformablePairInfo accessors / PxgComparePairInfo.
//   deformableRef512Writers  pxgSlangDeformableWriters against the real writers run on
//                            host memory, byte for byte, both PxU16 halves of one word.
//   deformableRef512Filter   pxgSlangDefFilter against a transcription of 5.12
//                            deformableElementFilter.cuh over real encoders.

#include "pxgTestFramework.h"

#include "host_defines.h"
#include "vector_functions.h"
#include "PxgFEMCloth.h"
#include "PxgSoftBody.h"
#include "PxgParticleSystem.h"
#include "PxgSimulationCoreDesc.h"
#include "PxgDeformableConstraints.h"
#include "PxgDeformableContactWriters.h"
#include "PxsDeformableSurfaceMaterialCore.h"
#include "PxsDeformableVolumeMaterialCore.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Winvalid-offsetof"
#endif

namespace
{
using namespace pxgtest;
using namespace physx;

struct DevBuf { PxgBufferHandle h = 0; uint64_t addr = 0; };
template <class T> static DevBuf mk(PxgSlangComputeDevice* dev, const std::vector<T>& v)
{
    DevBuf b;
    b.h = createBufferFrom(dev, v);
    b.addr = dev->getBufferDeviceAddress(b.h);
    return b;
}

// ===========================================================================
// Layout
// ===========================================================================
struct LayoutRow
{
    const char* name;
    uint32_t mirrorOff;   // the @offset the mirror's comment claims
    uint32_t size;        // the mirror member's size
    size_t realOff;       // offsetof in the real engine struct
};

// PxgFEMCloth mirror (PxgFEMClothGpu in pxgSlangFEMClothDesc.slang): one row per mirrored field.
static const LayoutRow kPxgFEMClothRows[] = {
    { "mTriMeshData", 0, 8, offsetof(PxgFEMCloth, mTriMeshData) },
    { "mVelocity_InvMass", 8, 8, offsetof(PxgFEMCloth, mVelocity_InvMass) },
    { "mPosition_InvMass", 16, 8, offsetof(PxgFEMCloth, mPosition_InvMass) },
    { "mRestPosition", 24, 8, offsetof(PxgFEMCloth, mRestPosition) },
    { "mPrevPositionInContactOffset", 32, 8, offsetof(PxgFEMCloth, mPrevPositionInContactOffset) },
    { "mPrevPositionInRestOffset", 40, 8, offsetof(PxgFEMCloth, mPrevPositionInRestOffset) },
    { "mDynamicFrictions", 48, 8, offsetof(PxgFEMCloth, mDynamicFrictions) },
    { "mMaterialIndices", 56, 8, offsetof(PxgFEMCloth, mMaterialIndices) },
    { "mTrianglesWithOwnedEdges", 64, 8, offsetof(PxgFEMCloth, mTrianglesWithOwnedEdges) },
    { "mTriangleVertexIndices_ownedEdges", 72, 8, offsetof(PxgFEMCloth, mTriangleVertexIndices_ownedEdges) },
    { "mOrderedNonSharedTriangleVertexIndices_triIndex", 80, 8, offsetof(PxgFEMCloth, mOrderedNonSharedTriangleVertexIndices_triIndex) },
    { "mOrderedSharedTriangleLambdas", 88, 8, offsetof(PxgFEMCloth, mOrderedSharedTriangleLambdas) },
    { "mOrderedNonSharedTriangleLambdas", 96, 8, offsetof(PxgFEMCloth, mOrderedNonSharedTriangleLambdas) },
    { "mOrderedNonSharedTriangleRestPoseInv", 104, 8, offsetof(PxgFEMCloth, mOrderedNonSharedTriangleRestPoseInv) },
    { "mOrderedSharedTrianglePairVertexIndices", 112, 8, offsetof(PxgFEMCloth, mOrderedSharedTrianglePairVertexIndices) },
    { "mOrderedNonSharedTrianglePairVertexIndices", 120, 8, offsetof(PxgFEMCloth, mOrderedNonSharedTrianglePairVertexIndices) },
    { "mOrderedSharedRestBendingAngle_flexuralStiffness_damping", 128, 8, offsetof(PxgFEMCloth, mOrderedSharedRestBendingAngle_flexuralStiffness_damping) },
    { "mOrderedNonSharedRestBendingAngle_flexuralStiffness_damping", 136, 8, offsetof(PxgFEMCloth, mOrderedNonSharedRestBendingAngle_flexuralStiffness_damping) },
    { "mSharedBendingLambdas", 144, 8, offsetof(PxgFEMCloth, mSharedBendingLambdas) },
    { "mNonSharedBendingLambdas", 152, 8, offsetof(PxgFEMCloth, mNonSharedBendingLambdas) },
    { "mNonSharedTriPair_hasActiveBendingWord", 160, 4, offsetof(PxgFEMCloth, mNonSharedTriPair_hasActiveBending) },
    { "mOrderedSharedRestEdge0_edge1", 168, 8, offsetof(PxgFEMCloth, mOrderedSharedRestEdge0_edge1) },
    { "mOrderedSharedRestEdgeLength_material0_material1", 176, 8, offsetof(PxgFEMCloth, mOrderedSharedRestEdgeLength_material0_material1) },
    { "mPosition_InvMassCP", 184, 8, offsetof(PxgFEMCloth, mPosition_InvMassCP) },
    { "mNonSharedTriAccumulatedPartitionsCP", 192, 8, offsetof(PxgFEMCloth, mNonSharedTriAccumulatedPartitionsCP) },
    { "mSharedTriPairRemapOutputCP", 200, 8, offsetof(PxgFEMCloth, mSharedTriPairRemapOutputCP) },
    { "mNonSharedTriPairRemapOutputCP", 208, 8, offsetof(PxgFEMCloth, mNonSharedTriPairRemapOutputCP) },
    { "mSharedTriPairAccumulatedCopiesCP", 216, 8, offsetof(PxgFEMCloth, mSharedTriPairAccumulatedCopiesCP) },
    { "mNonSharedTriPairAccumulatedCopiesCP", 224, 8, offsetof(PxgFEMCloth, mNonSharedTriPairAccumulatedCopiesCP) },
    { "mSharedTriPairAccumulatedPartitionsCP", 232, 8, offsetof(PxgFEMCloth, mSharedTriPairAccumulatedPartitionsCP) },
    { "mNonSharedTriPairAccumulatedPartitionsCP", 240, 8, offsetof(PxgFEMCloth, mNonSharedTriPairAccumulatedPartitionsCP) },
    { "mDeltaPos", 248, 8, offsetof(PxgFEMCloth, mDeltaPos) },
    { "mAccumulatedDeltaPos", 256, 8, offsetof(PxgFEMCloth, mAccumulatedDeltaPos) },
    { "mAccumulatedDeltaVel", 264, 8, offsetof(PxgFEMCloth, mAccumulatedDeltaVel) },
    { "mPackedNodeBounds", 272, 8, offsetof(PxgFEMCloth, mPackedNodeBounds) },
    { "mRigidSurfaceFilterPairs", 280, 8, offsetof(PxgFEMCloth, mRigidSurfaceFilterPairs) },
    { "mSurfaceSurfaceFilterPairs", 288, 8, offsetof(PxgFEMCloth, mSurfaceSurfaceFilterPairs) },
    { "mNumRigidSurfaceFilterPairs", 296, 4, offsetof(PxgFEMCloth, mNumRigidSurfaceFilterPairs) },
    { "mNumSurfaceSurfaceFilterPairs", 300, 4, offsetof(PxgFEMCloth, mNumSurfaceSurfaceFilterPairs) },
    { "mLinearDamping", 304, 4, offsetof(PxgFEMCloth, mLinearDamping) },
    { "mMaxLinearVelocity", 308, 4, offsetof(PxgFEMCloth, mMaxLinearVelocity) },
    { "mPenBiasClamp", 312, 4, offsetof(PxgFEMCloth, mPenBiasClamp) },
    { "mSettlingThreshold", 316, 4, offsetof(PxgFEMCloth, mSettlingThreshold) },
    { "mSleepThreshold", 320, 4, offsetof(PxgFEMCloth, mSleepThreshold) },
    { "mSettlingDamping", 324, 4, offsetof(PxgFEMCloth, mSettlingDamping) },
    { "mSelfCollisionFilterDistance", 328, 4, offsetof(PxgFEMCloth, mSelfCollisionFilterDistance) },
    { "mNbVerts", 332, 4, offsetof(PxgFEMCloth, mNbVerts) },
    { "mNbTriangles", 336, 4, offsetof(PxgFEMCloth, mNbTriangles) },
    { "mNbNonSharedTriangles", 340, 4, offsetof(PxgFEMCloth, mNbNonSharedTriangles) },
    { "mNbTrianglesWithOwnedEdges", 344, 4, offsetof(PxgFEMCloth, mNbTrianglesWithOwnedEdges) },
    { "mNbTrianglePairs", 348, 4, offsetof(PxgFEMCloth, mNbTrianglePairs) },
    { "mNbSharedTrianglePairs", 352, 4, offsetof(PxgFEMCloth, mNbSharedTrianglePairs) },
    { "mNbNonSharedTrianglePairs", 356, 4, offsetof(PxgFEMCloth, mNbNonSharedTrianglePairs) },
    { "mNbNonSharedTriPartitions", 360, 4, offsetof(PxgFEMCloth, mNbNonSharedTriPartitions) },
    { "mNbSharedTriPairPartitions", 364, 4, offsetof(PxgFEMCloth, mNbSharedTriPairPartitions) },
    { "mNbNonSharedTriPairPartitions", 368, 4, offsetof(PxgFEMCloth, mNbNonSharedTriPairPartitions) },
    { "mNonSharedTriClusterId", 372, 4, offsetof(PxgFEMCloth, mNonSharedTriClusterId) },
    { "mSharedTriPairClusterId", 376, 4, offsetof(PxgFEMCloth, mSharedTriPairClusterId) },
    { "mNonSharedTriPairClusterId", 380, 4, offsetof(PxgFEMCloth, mNonSharedTriPairClusterId) },
    { "mElementIndex", 384, 4, offsetof(PxgFEMCloth, mElementIndex) },
    { "mGpuRemapIndex", 388, 4, offsetof(PxgFEMCloth, mGpuRemapIndex) },
    { "mFlagsWord", 392, 4, offsetof(PxgFEMCloth, mActorFlags) },
    { "mIsActive", 396, 4, offsetof(PxgFEMCloth, mIsActive) },
    { "mRestOffset", 400, 4, offsetof(PxgFEMCloth, mRestOffset) },
    { "mOriginalContactOffset", 404, 4, offsetof(PxgFEMCloth, mOriginalContactOffset) },
    { "mNbCollisionPairUpdatesPerTimestep", 408, 4, offsetof(PxgFEMCloth, mNbCollisionPairUpdatesPerTimestep) },
    { "mNbCollisionSubsteps", 412, 4, offsetof(PxgFEMCloth, mNbCollisionSubsteps) },
};

// PxgSoftBody mirror (PxgSoftBodyGpu in pxgSlangSoftBodyDesc.slang): one row per mirrored field.
static const LayoutRow kPxgSoftBodyRows[] = {
    { "mTetMeshData", 0, 8, offsetof(PxgSoftBody, mTetMeshData) },
    { "mTetMeshSurfaceHint", 8, 8, offsetof(PxgSoftBody, mTetMeshSurfaceHint) },
    { "mTetIndices", 16, 8, offsetof(PxgSoftBody, mTetIndices) },
    { "mTetIndicesRemapTable", 24, 8, offsetof(PxgSoftBody, mTetIndicesRemapTable) },
    { "mTetraStresses", 32, 8, offsetof(PxgSoftBody, mTetraStresses) },
    { "mTetraRestPoses", 40, 8, offsetof(PxgSoftBody, mTetraRestPoses) },
    { "mTetraStressCoefficient", 48, 8, offsetof(PxgSoftBody, mTetraStressCoefficient) },
    { "mTetraRotations", 56, 8, offsetof(PxgSoftBody, mTetraRotations) },
    { "mPosition_InvMass", 64, 8, offsetof(PxgSoftBody, mPosition_InvMass) },
    { "mRestPosition", 72, 8, offsetof(PxgSoftBody, mRestPosition) },
    { "mSimTetIndices", 80, 8, offsetof(PxgSoftBody, mSimTetIndices) },
    { "mSimVelocity_InvMass", 88, 8, offsetof(PxgSoftBody, mSimVelocity_InvMass) },
    { "mSimPosition_InvMass", 96, 8, offsetof(PxgSoftBody, mSimPosition_InvMass) },
    { "mSimKinematicTarget", 104, 8, offsetof(PxgSoftBody, mSimKinematicTarget) },
    { "mVertsAreDeformed", 112, 8, offsetof(PxgSoftBody, mVertsAreDeformed) },
    { "mVertsCanNotDeform", 120, 8, offsetof(PxgSoftBody, mVertsCanNotDeform) },
    { "mSimDeltaPos", 128, 8, offsetof(PxgSoftBody, mSimDeltaPos) },
    { "mSimTetraRestPoses", 136, 8, offsetof(PxgSoftBody, mSimTetraRestPoses) },
    { "mOrigQinv", 144, 8, offsetof(PxgSoftBody, mOrigQinv) },
    { "mSimOrderedTetrahedrons", 152, 8, offsetof(PxgSoftBody, mSimOrderedTetrahedrons) },
    { "mSimJacobiVertIndices", 160, 8, offsetof(PxgSoftBody, mSimJacobiVertIndices) },
    { "mSimTetraRotations", 168, 8, offsetof(PxgSoftBody, mSimTetraRotations) },
    { "mSimVelocity_invMassCP", 176, 8, offsetof(PxgSoftBody, mSimVelocity_invMassCP) },
    { "mSimPosition_InvMassCP", 184, 8, offsetof(PxgSoftBody, mSimPosition_InvMassCP) },
    { "mSimRemapOutputCP", 192, 8, offsetof(PxgSoftBody, mSimRemapOutputCP) },
    { "mSimAccumulatedCopiesCP", 200, 8, offsetof(PxgSoftBody, mSimAccumulatedCopiesCP) },
    { "mSimAccumulatedPartitionsCP", 208, 8, offsetof(PxgSoftBody, mSimAccumulatedPartitionsCP) },
    { "mVertsBarycentricInGridModel", 216, 8, offsetof(PxgSoftBody, mVertsBarycentricInGridModel) },
    { "mVertsRemapInGridModel", 224, 8, offsetof(PxgSoftBody, mVertsRemapInGridModel) },
    { "mTetsRemapColToSim", 232, 8, offsetof(PxgSoftBody, mTetsRemapColToSim) },
    { "mTetsAccumulatedRemapColToSim", 240, 8, offsetof(PxgSoftBody, mTetsAccumulatedRemapColToSim) },
    { "mSurfaceVertsHint", 248, 8, offsetof(PxgSoftBody, mSurfaceVertsHint) },
    { "mSurfaceVertToTetRemap", 256, 8, offsetof(PxgSoftBody, mSurfaceVertToTetRemap) },
    { "mSimTetraMultipliers", 264, 8, offsetof(PxgSoftBody, mSimTetraMultipliers) },
    { "mSimDelta", 272, 8, offsetof(PxgSoftBody, mSimDelta) },
    { "mPackedNodeBounds", 280, 8, offsetof(PxgSoftBody, mPackedNodeBounds) },
    { "mOrderedMaterialIndices", 288, 8, offsetof(PxgSoftBody, mOrderedMaterialIndices) },
    { "mMaterialIndices", 296, 8, offsetof(PxgSoftBody, mMaterialIndices) },
    { "mRigidVolumeFilterPairs", 304, 8, offsetof(PxgSoftBody, mRigidVolumeFilterPairs) },
    { "mVolumeVolumeFilterPairs", 312, 8, offsetof(PxgSoftBody, mVolumeVolumeFilterPairs) },
    { "mVolumeSurfaceFilterPairs", 320, 8, offsetof(PxgSoftBody, mVolumeSurfaceFilterPairs) },
    { "mNumRigidVolumeFilterPairs", 328, 4, offsetof(PxgSoftBody, mNumRigidVolumeFilterPairs) },
    { "mNumVolumeVolumeFilterPairs", 332, 4, offsetof(PxgSoftBody, mNumVolumeVolumeFilterPairs) },
    { "mNumVolumeSurfaceFilterPairs", 336, 4, offsetof(PxgSoftBody, mNumVolumeSurfaceFilterPairs) },
    { "mLinearDamping", 340, 4, offsetof(PxgSoftBody, mLinearDamping) },
    { "mMaxLinearVelocity", 344, 4, offsetof(PxgSoftBody, mMaxLinearVelocity) },
    { "mPenBiasClamp", 348, 4, offsetof(PxgSoftBody, mPenBiasClamp) },
    { "mSettlingThreshold", 352, 4, offsetof(PxgSoftBody, mSettlingThreshold) },
    { "mSleepThreshold", 356, 4, offsetof(PxgSoftBody, mSleepThreshold) },
    { "mSettlingDamping", 360, 4, offsetof(PxgSoftBody, mSettlingDamping) },
    { "mSelfCollisionFilterDistance", 364, 4, offsetof(PxgSoftBody, mSelfCollisionFilterDistance) },
    { "mSelfCollisionStressTolerance", 368, 4, offsetof(PxgSoftBody, mSelfCollisionStressTolerance) },
    { "mInitialRotationX", 372, 4, offsetof(PxgSoftBody, mInitialRotation) + 0 },
    { "mInitialRotationY", 376, 4, offsetof(PxgSoftBody, mInitialRotation) + 4 },
    { "mInitialRotationZ", 380, 4, offsetof(PxgSoftBody, mInitialRotation) + 8 },
    { "mInitialRotationW", 384, 4, offsetof(PxgSoftBody, mInitialRotation) + 12 },
    { "mNumVerts", 388, 4, offsetof(PxgSoftBody, mNumVerts) },
    { "mNumTets", 392, 4, offsetof(PxgSoftBody, mNumTets) },
    { "mNumVertsGM", 396, 4, offsetof(PxgSoftBody, mNumVertsGM) },
    { "mNumTetsGM", 400, 4, offsetof(PxgSoftBody, mNumTetsGM) },
    { "mNumPartitionsGM", 404, 4, offsetof(PxgSoftBody, mNumPartitionsGM) },
    { "mElementIndex", 408, 4, offsetof(PxgSoftBody, mElementIndex) },
    { "mGpuRemapIndex", 412, 4, offsetof(PxgSoftBody, mGpuRemapIndex) },
    { "mFlagsWord", 416, 4, offsetof(PxgSoftBody, mActorFlags) },
    { "mSimPullIndices", 424, 8, offsetof(PxgSoftBody, mSimPullIndices) },
    { "mAwake", 432, 4, offsetof(PxgSoftBody, mAwake) },
    { "mNumTetsPerElement", 436, 4, offsetof(PxgSoftBody, mNumTetsPerElement) },
    { "mNumJacobiVertices", 440, 4, offsetof(PxgSoftBody, mNumJacobiVertices) },
    { "mRestOffset", 444, 4, offsetof(PxgSoftBody, mRestOffset) },
    { "mOriginalContactOffset", 448, 4, offsetof(PxgSoftBody, mOriginalContactOffset) },
    { "mJacobiScale", 452, 4, offsetof(PxgSoftBody, mJacobiScale) },
};

// PxgParticleSystem mirror (PxgParticleSystemGpu in pxgSlangParticleDesc.slang): one row per mirrored field.
static const LayoutRow kPxgParticleSystemRows[] = {
    { "mUnsortedPositions_InvMass", 0, 8, offsetof(PxgParticleSystem, mUnsortedPositions_InvMass) },
    { "mUnsortedVelocities", 8, 8, offsetof(PxgParticleSystem, mUnsortedVelocities) },
    { "mUnsortedPhaseArray", 16, 8, offsetof(PxgParticleSystem, mUnsortedPhaseArray) },
    { "mSortedPositions_InvMass", 24, 8, offsetof(PxgParticleSystem, mSortedPositions_InvMass) },
    { "mSortedVelocities", 32, 8, offsetof(PxgParticleSystem, mSortedVelocities) },
    { "mSortedPhaseArray", 40, 8, offsetof(PxgParticleSystem, mSortedPhaseArray) },
    { "mUnsortedToSortedMapping", 48, 8, offsetof(PxgParticleSystem, mUnsortedToSortedMapping) },
    { "mSortedToUnsortedMapping", 56, 8, offsetof(PxgParticleSystem, mSortedToUnsortedMapping) },
    { "mParticleSelfCollisionCount", 64, 8, offsetof(PxgParticleSystem, mParticleSelfCollisionCount) },
    { "mCollisionIndex", 72, 8, offsetof(PxgParticleSystem, mCollisionIndex) },
    { "mParticleMaterials", 80, 8, offsetof(PxgParticleSystem, mParticleMaterials) },
    { "mCommonData_GridCellWidth", 88, 4, offsetof(PxgParticleSystem, mCommonData) + offsetof(PxGpuParticleData, mGridCellWidth) },
    { "mCommonData_GridSizeX", 92, 4, offsetof(PxgParticleSystem, mCommonData) + offsetof(PxGpuParticleData, mGridSizeX) },
    { "mCommonData_GridSizeY", 96, 4, offsetof(PxgParticleSystem, mCommonData) + offsetof(PxGpuParticleData, mGridSizeY) },
    { "mCommonData_GridSizeZ", 100, 4, offsetof(PxgParticleSystem, mCommonData) + offsetof(PxGpuParticleData, mGridSizeZ) },
    { "mCommonData_ParticleContactDistance", 104, 4, offsetof(PxgParticleSystem, mCommonData) + offsetof(PxGpuParticleData, mParticleContactDistance) },
    { "mCommonData_ParticleContactDistanceInv", 108, 4, offsetof(PxgParticleSystem, mCommonData) + offsetof(PxGpuParticleData, mParticleContactDistanceInv) },
    { "mCommonData_ParticleContactDistanceSq", 112, 4, offsetof(PxgParticleSystem, mCommonData) + offsetof(PxGpuParticleData, mParticleContactDistanceSq) },
    { "mCommonData_NumParticles", 116, 4, offsetof(PxgParticleSystem, mCommonData) + offsetof(PxGpuParticleData, mNumParticles) },
    { "mCommonData_MaxParticles", 120, 4, offsetof(PxgParticleSystem, mCommonData) + offsetof(PxGpuParticleData, mMaxParticles) },
    { "mCommonData_MaxNeighborhood", 124, 4, offsetof(PxgParticleSystem, mCommonData) + offsetof(PxGpuParticleData, mMaxNeighborhood) },
    { "mCommonData_MaxDiffuseParticles", 128, 4, offsetof(PxgParticleSystem, mCommonData) + offsetof(PxGpuParticleData, mMaxDiffuseParticles) },
    { "mCommonData_NumParticleBuffers", 132, 4, offsetof(PxgParticleSystem, mCommonData) + offsetof(PxGpuParticleData, mNumParticleBuffers) },
    { "mOriginPos_InvMass", 136, 8, offsetof(PxgParticleSystem, mOriginPos_InvMass) },
    { "mGridParticleHash", 144, 8, offsetof(PxgParticleSystem, mGridParticleHash) },
    { "mAccumDeltaP", 152, 8, offsetof(PxgParticleSystem, mAccumDeltaP) },
    { "mSortedDeltaP", 160, 8, offsetof(PxgParticleSystem, mSortedDeltaP) },
    { "mCellStart", 168, 8, offsetof(PxgParticleSystem, mCellStart) },
    { "mCellEnd", 176, 8, offsetof(PxgParticleSystem, mCellEnd) },
    { "mCollisionHeaders", 184, 8, offsetof(PxgParticleSystem, mCollisionHeaders) },
    { "mDensityCollisionImpulses", 192, 8, offsetof(PxgParticleSystem, mDensityCollisionImpulses) },
    { "mOneWayContactInfos", 200, 8, offsetof(PxgParticleSystem, mOneWayContactInfos) },
    { "mOneWayForces", 208, 8, offsetof(PxgParticleSystem, mOneWayForces) },
    { "mOneWayNodeIndex", 216, 8, offsetof(PxgParticleSystem, mOneWayNodeIndex) },
    { "mOneWaySurfaceVelocity", 224, 8, offsetof(PxgParticleSystem, mOneWaySurfaceVelocity) },
    { "mOneWayContactCount", 232, 8, offsetof(PxgParticleSystem, mOneWayContactCount) },
    { "mRestArray", 240, 8, offsetof(PxgParticleSystem, mRestArray) },
    { "mData_BoundCenterX", 248, 4, offsetof(PxgParticleSystem, mData) + offsetof(PxgParticleSystemData, mBoundCenter) + 0 },
    { "mData_BoundCenterY", 252, 4, offsetof(PxgParticleSystem, mData) + offsetof(PxgParticleSystemData, mBoundCenter) + 4 },
    { "mData_BoundCenterZ", 256, 4, offsetof(PxgParticleSystem, mData) + offsetof(PxgParticleSystemData, mBoundCenter) + 8 },
    { "mData_ElementIndex", 260, 4, offsetof(PxgParticleSystem, mData) + offsetof(PxgParticleSystemData, mElementIndex) },
    { "mData_RemapIndex", 264, 4, offsetof(PxgParticleSystem, mData) + offsetof(PxgParticleSystemData, mRemapIndex) },
    { "mData_RestOffset", 268, 4, offsetof(PxgParticleSystem, mData) + offsetof(PxgParticleSystemData, mRestOffset) },
    { "mData_SolidRestOffset", 272, 4, offsetof(PxgParticleSystem, mData) + offsetof(PxgParticleSystemData, mSolidRestOffset) },
    { "mData_Spiky1", 276, 4, offsetof(PxgParticleSystem, mData) + offsetof(PxgParticleSystemData, mSpiky1) },
    { "mData_Spiky2", 280, 4, offsetof(PxgParticleSystem, mData) + offsetof(PxgParticleSystemData, mSpiky2) },
    { "mData_RestDensity", 284, 4, offsetof(PxgParticleSystem, mData) + offsetof(PxgParticleSystemData, mRestDensity) },
    { "mData_RestDensityBoundary", 288, 4, offsetof(PxgParticleSystem, mData) + offsetof(PxgParticleSystemData, mRestDensityBoundary) },
    { "mData_FluidSurfaceConstraintScale", 292, 4, offsetof(PxgParticleSystem, mData) + offsetof(PxgParticleSystemData, mFluidSurfaceConstraintScale) },
    { "mData_LambdaScale", 296, 4, offsetof(PxgParticleSystem, mData) + offsetof(PxgParticleSystemData, mLambdaScale) },
    { "mData_RelaxationFactor", 300, 4, offsetof(PxgParticleSystem, mData) + offsetof(PxgParticleSystemData, mRelaxationFactor) },
    { "mData_FluidRestOffset", 304, 4, offsetof(PxgParticleSystem, mData) + offsetof(PxgParticleSystemData, mFluidRestOffset) },
    { "mData_InvRestDensity", 308, 4, offsetof(PxgParticleSystem, mData) + offsetof(PxgParticleSystemData, mInvRestDensity) },
    { "mData_Flags", 312, 4, offsetof(PxgParticleSystem, mData) + offsetof(PxgParticleSystemData, mFlags) },
    { "mData_MaxVelocity", 316, 4, offsetof(PxgParticleSystem, mData) + offsetof(PxgParticleSystemData, mMaxVelocity) },
    { "mData_PenBiasClamp", 320, 4, offsetof(PxgParticleSystem, mData) + offsetof(PxgParticleSystemData, mPenBiasClamp) },
    { "mData_LockFlagsWord", 324, 4, offsetof(PxgParticleSystem, mData) + offsetof(PxgParticleSystemData, mLockFlags) },
    { "mData_NumPhaseToMaterials", 328, 4, offsetof(PxgParticleSystem, mData) + offsetof(PxgParticleSystemData, mNumPhaseToMaterials) },
    { "mData_WindX", 332, 4, offsetof(PxgParticleSystem, mData) + offsetof(PxgParticleSystemData, mWind) + 0 },
    { "mData_WindY", 336, 4, offsetof(PxgParticleSystem, mData) + offsetof(PxgParticleSystemData, mWind) + 4 },
    { "mData_WindZ", 340, 4, offsetof(PxgParticleSystem, mData) + offsetof(PxgParticleSystemData, mWind) + 8 },
    { "mDensity", 344, 8, offsetof(PxgParticleSystem, mDensity) },
    { "mStaticDensity", 352, 8, offsetof(PxgParticleSystem, mStaticDensity) },
    { "mSurfaceNormal", 360, 8, offsetof(PxgParticleSystem, mSurfaceNormal) },
    { "mDelta", 368, 8, offsetof(PxgParticleSystem, mDelta) },
    { "mCurl", 376, 8, offsetof(PxgParticleSystem, mCurl) },
    { "mNormalArray", 384, 8, offsetof(PxgParticleSystem, mNormalArray) },
    { "mSortedOriginPos_InvMass", 392, 8, offsetof(PxgParticleSystem, mSortedOriginPos_InvMass) },
    { "mParticleSimBuffers", 400, 8, offsetof(PxgParticleSystem, mParticleSimBuffers) },
    { "mParticleBufferRunsum", 408, 8, offsetof(PxgParticleSystem, mParticleBufferRunsum) },
    { "mParticleBufferSortedUniqueIds", 416, 8, offsetof(PxgParticleSystem, mParticleBufferSortedUniqueIds) },
    { "mParticleBufferSortedUniqueIdsOriginalIndex", 424, 8, offsetof(PxgParticleSystem, mParticleBufferSortedUniqueIdsOriginalIndex) },
    { "mDiffuseSimBuffers", 432, 8, offsetof(PxgParticleSystem, mDiffuseSimBuffers) },
    { "mNumDiffuseBuffers", 440, 4, offsetof(PxgParticleSystem, mNumDiffuseBuffers) },
    { "mNumDiffuseParticles", 448, 8, offsetof(PxgParticleSystem, mNumDiffuseParticles) },
    { "mDiffusePosition_LifeTime", 456, 8, offsetof(PxgParticleSystem, mDiffusePosition_LifeTime) },
    { "mDiffuseVelocity", 464, 8, offsetof(PxgParticleSystem, mDiffuseVelocity) },
    { "mDiffusePotentials", 472, 8, offsetof(PxgParticleSystem, mDiffusePotentials) },
    { "mDiffuseCellStart", 480, 8, offsetof(PxgParticleSystem, mDiffuseCellStart) },
    { "mDiffuseCellEnd", 488, 8, offsetof(PxgParticleSystem, mDiffuseCellEnd) },
    { "mDiffuseGridParticleHash", 496, 8, offsetof(PxgParticleSystem, mDiffuseGridParticleHash) },
    { "mDiffuseOriginPos_LifeTime", 504, 8, offsetof(PxgParticleSystem, mDiffuseOriginPos_LifeTime) },
    { "mDiffuseSortedPos_LifeTime", 512, 8, offsetof(PxgParticleSystem, mDiffuseSortedPos_LifeTime) },
    { "mDiffuseSortedOriginPos_LifeTime", 520, 8, offsetof(PxgParticleSystem, mDiffuseSortedOriginPos_LifeTime) },
    { "mDiffuseSortedVel", 528, 8, offsetof(PxgParticleSystem, mDiffuseSortedVel) },
    { "mDiffuseSortedToUnsortedMapping", 536, 8, offsetof(PxgParticleSystem, mDiffuseSortedToUnsortedMapping) },
    { "mDiffuseUnsortedToSortedMapping", 544, 8, offsetof(PxgParticleSystem, mDiffuseUnsortedToSortedMapping) },
    { "mDiffuseOneWayContactInfos", 552, 8, offsetof(PxgParticleSystem, mDiffuseOneWayContactInfos) },
    { "mDiffuseOneWayForces", 560, 8, offsetof(PxgParticleSystem, mDiffuseOneWayForces) },
    { "mDiffuseOneWayNodeIndex", 568, 8, offsetof(PxgParticleSystem, mDiffuseOneWayNodeIndex) },
    { "mDiffuseOneWayContactCount", 576, 8, offsetof(PxgParticleSystem, mDiffuseOneWayContactCount) },
    { "mDerivedPBDMaterialData", 584, 8, offsetof(PxgParticleSystem, mDerivedPBDMaterialData) },
    { "mParticleMaterialStride", 592, 4, offsetof(PxgParticleSystem, mParticleMaterialStride) },
    { "mPhaseGroupToMaterialHandle", 600, 8, offsetof(PxgParticleSystem, mPhaseGroupToMaterialHandle) },
};

// PxgParticleSimBuffer mirror (PxgParticleSimBufferGpu in pxgSlangParticleDesc.slang): one row per mirrored field.
static const LayoutRow kPxgParticleSimBufferRows[] = {
    { "mPositionInvMasses", 0, 8, offsetof(PxgParticleSimBuffer, mPositionInvMasses) },
    { "mVelocities", 8, 8, offsetof(PxgParticleSimBuffer, mVelocities) },
    { "mRestPositions", 16, 8, offsetof(PxgParticleSimBuffer, mRestPositions) },
    { "mPhases", 24, 8, offsetof(PxgParticleSimBuffer, mPhases) },
    { "mNumActiveParticles", 32, 4, offsetof(PxgParticleSimBuffer, mNumActiveParticles) },
    { "mFlags", 36, 4, offsetof(PxgParticleSimBuffer, mFlags) },
    { "mDiffuseParticleBufferIndex", 40, 4, offsetof(PxgParticleSimBuffer, mDiffuseParticleBufferIndex) },
    { "mUniqueId", 44, 4, offsetof(PxgParticleSimBuffer, mUniqueId) },
};

// PxgParticleDiffuseSimBuffer mirror (PxgParticleDiffuseSimBufferGpu in pxgSlangParticleDesc.slang): one row per mirrored field.
static const LayoutRow kPxgParticleDiffuseSimBufferRows[] = {
    { "mParams_threshold", 0, 4, offsetof(PxgParticleDiffuseSimBuffer, mParams) + offsetof(PxDiffuseParticleParams, threshold) },
    { "mParams_lifetime", 4, 4, offsetof(PxgParticleDiffuseSimBuffer, mParams) + offsetof(PxDiffuseParticleParams, lifetime) },
    { "mParams_airDrag", 8, 4, offsetof(PxgParticleDiffuseSimBuffer, mParams) + offsetof(PxDiffuseParticleParams, airDrag) },
    { "mParams_bubbleDrag", 12, 4, offsetof(PxgParticleDiffuseSimBuffer, mParams) + offsetof(PxDiffuseParticleParams, bubbleDrag) },
    { "mParams_buoyancy", 16, 4, offsetof(PxgParticleDiffuseSimBuffer, mParams) + offsetof(PxDiffuseParticleParams, buoyancy) },
    { "mParams_kineticEnergyWeight", 20, 4, offsetof(PxgParticleDiffuseSimBuffer, mParams) + offsetof(PxDiffuseParticleParams, kineticEnergyWeight) },
    { "mParams_pressureWeight", 24, 4, offsetof(PxgParticleDiffuseSimBuffer, mParams) + offsetof(PxDiffuseParticleParams, pressureWeight) },
    { "mParams_divergenceWeight", 28, 4, offsetof(PxgParticleDiffuseSimBuffer, mParams) + offsetof(PxDiffuseParticleParams, divergenceWeight) },
    { "mParams_collisionDecay", 32, 4, offsetof(PxgParticleDiffuseSimBuffer, mParams) + offsetof(PxDiffuseParticleParams, collisionDecay) },
    { "mParams_useAccurateVelocity", 36, 4, offsetof(PxgParticleDiffuseSimBuffer, mParams) + offsetof(PxDiffuseParticleParams, useAccurateVelocity) },
    { "mDiffusePositions_LifeTime", 40, 8, offsetof(PxgParticleDiffuseSimBuffer, mDiffusePositions_LifeTime) },
    { "mDiffuseVelocities", 48, 8, offsetof(PxgParticleDiffuseSimBuffer, mDiffuseVelocities) },
    { "mMaxNumParticles", 56, 4, offsetof(PxgParticleDiffuseSimBuffer, mMaxNumParticles) },
    { "mNumDiffuseParticles", 64, 8, offsetof(PxgParticleDiffuseSimBuffer, mNumDiffuseParticles) },
    { "mNumActiveDiffuseParticles", 72, 8, offsetof(PxgParticleDiffuseSimBuffer, mNumActiveDiffuseParticles) },
    { "mStartIndex", 80, 4, offsetof(PxgParticleDiffuseSimBuffer, mStartIndex) },
    { "mFlags", 84, 4, offsetof(PxgParticleDiffuseSimBuffer, mFlags) },
};

// PxsDeformableSurfaceMaterialData mirror (PxsDeformableSurfaceMaterialGpu in pxgSlangFEMClothDesc.slang): one row per mirrored field.
static const LayoutRow kPxsDeformableSurfaceMaterialDataRows[] = {
    { "youngs", 0, 4, offsetof(PxsDeformableSurfaceMaterialData, youngs) },
    { "poissons", 4, 4, offsetof(PxsDeformableSurfaceMaterialData, poissons) },
    { "dynamicFriction", 8, 4, offsetof(PxsDeformableSurfaceMaterialData, dynamicFriction) },
    { "thickness", 12, 4, offsetof(PxsDeformableSurfaceMaterialData, thickness) },
    { "bendingStiffness", 16, 4, offsetof(PxsDeformableSurfaceMaterialData, bendingStiffness) },
    { "elasticityDamping", 20, 4, offsetof(PxsDeformableSurfaceMaterialData, elasticityDamping) },
    { "bendingDamping", 24, 4, offsetof(PxsDeformableSurfaceMaterialData, bendingDamping) },
    { "padding0", 28, 4, offsetof(PxsDeformableSurfaceMaterialData, padding) },
};

// Block, writer and record layouts the shared modules spell as byte constants
// (pxgSlangDeformableRef / pxgSlangDeformableWriters).
static_assert(sizeof(PxgDeformablePairInfo) == 16 && alignof(PxgDeformablePairInfo) == 16, "PxgDeformablePairInfo");
static_assert(offsetof(PxgDeformablePairInfo, ref1) == 8, "PXG_PAIRINFO_ref1");
static_assert(sizeof(PxgDeformableAttachment) == 48, "PXG_DATT_STRIDE");
static_assert(offsetof(PxgDeformableAttachment, barycentricOrPosition0) == 16, "PXG_DATT_barycentricOrPosition0");
static_assert(offsetof(PxgDeformableAttachment, barycentric1) == 32, "PXG_DATT_barycentric1");

static_assert(sizeof(PxgRigidDbContactBlock) == 2688, "PXG_RDBC_STRIDE");
static_assert(offsetof(PxgRigidDbContactBlock, raXnF0_resp) == 512, "PXG_RDBC_raXnF0_resp");
static_assert(offsetof(PxgRigidDbContactBlock, raXnF1_resp) == 1024, "PXG_RDBC_raXnF1_resp");
static_assert(offsetof(PxgRigidDbContactBlock, normal_errorW) == 1536, "PXG_RDBC_normal_errorW");
static_assert(offsetof(PxgRigidDbContactBlock, fricTan0_invMass0) == 2048, "PXG_RDBC_fricTan0_invMass0");
static_assert(offsetof(PxgRigidDbContactBlock, maxPenBiasClamp) == 2560, "PXG_RDBC_maxPenBiasClamp");

static_assert(sizeof(PxgRigidDbAttachmentBlock) == 3200, "PXG_RDBA_STRIDE");
static_assert(offsetof(PxgRigidDbAttachmentBlock, raXn0_biasW) == 512, "PXG_RDBA_raXn0_biasW");
static_assert(offsetof(PxgRigidDbAttachmentBlock, raXn1_biasW) == 1024, "PXG_RDBA_raXn1_biasW");
static_assert(offsetof(PxgRigidDbAttachmentBlock, raXn2_biasW) == 1536, "PXG_RDBA_raXn2_biasW");
static_assert(offsetof(PxgRigidDbAttachmentBlock, velMultiplierXYZ_invMassW) == 2048, "PXG_RDBA_velMultiplierXYZ_invMassW");
static_assert(offsetof(PxgRigidDbAttachmentBlock, deformableRef) == 2560, "PXG_RDBA_deformableRef");
static_assert(offsetof(PxgRigidDbAttachmentBlock, rigidId) == 2816, "PXG_RDBA_rigidId");
static_assert(offsetof(PxgRigidDbAttachmentBlock, rigidBodyRefCount) == 3072, "PXG_RDBA_rigidBodyRefCount");

static_assert(sizeof(PxgDbDbContactBlock) == 640, "PXG_DBDBC_STRIDE");
static_assert(offsetof(PxgDbDbContactBlock, maxPenBiasClamp) == 512, "PXG_DBDBC_maxPenBiasClamp");

static_assert(sizeof(PxgDbDbAttachmentBlock) == 1536, "PXG_DBDBA_STRIDE");
static_assert(offsetof(PxgDbDbAttachmentBlock, barycentric1) == 512, "PXG_DBDBA_barycentric1");
static_assert(offsetof(PxgDbDbAttachmentBlock, deformableRef0) == 1024, "PXG_DBDBA_deformableRef0");
static_assert(offsetof(PxgDbDbAttachmentBlock, deformableRef1) == 1280, "PXG_DBDBA_deformableRef1");

static_assert(sizeof(PxgParticleDbContactBlock) == 1152, "PXG_PDBC_STRIDE");
static_assert(offsetof(PxgParticleDbContactBlock, barycentric) == 512, "PXG_PDBC_barycentric");
static_assert(offsetof(PxgParticleDbContactBlock, maxPenBiasClamp) == 1024, "PXG_PDBC_maxPenBiasClamp");

static_assert(sizeof(PxgRigidParticleContactBlock) == 2688, "PXG_RPC_STRIDE");
static_assert(offsetof(PxgRigidParticleContactBlock, raXnF0_resp) == 512, "PXG_RPC_raXnF0_resp");
static_assert(offsetof(PxgRigidParticleContactBlock, raXnF1_resp) == 1024, "PXG_RPC_raXnF1_resp");
static_assert(offsetof(PxgRigidParticleContactBlock, normal_errorW) == 1536, "PXG_RPC_normal_errorW");
static_assert(offsetof(PxgRigidParticleContactBlock, fricTan0_invMass0) == 2048, "PXG_RPC_fricTan0_invMass0");
static_assert(offsetof(PxgRigidParticleContactBlock, penBiasClampRigid) == 2560, "PXG_RPC_penBiasClampRigid");

// The by-value writers the kernels' push blocks inline (pxgSlangDeformableWriters header).
static_assert(sizeof(PxgDbDbContactWriter) == 56, "PxgDbDbContactWriter");
static_assert(offsetof(PxgDbDbContactWriter, outContactInfo) == 32, "PxgDbDbContactWriter::outContactInfo");
static_assert(offsetof(PxgDbDbContactWriter, maxContacts) == 48, "PxgDbDbContactWriter::maxContacts");
static_assert(sizeof(PxgRigidDbContactWriter) == 80, "PxgRigidDbContactWriter");
static_assert(offsetof(PxgRigidDbContactWriter, outRigidMaterialIndex) == 32, "PxgRigidDbContactWriter::outRigidMaterialIndex");
static_assert(offsetof(PxgRigidDbContactWriter, maxNumContacts) == 72, "PxgRigidDbContactWriter::maxNumContacts");
static_assert(sizeof(PxgParticleDbContactWriter) == 72, "PxgParticleDbContactWriter");
static_assert(offsetof(PxgParticleDbContactWriter, maxNumContacts) == 64, "PxgParticleDbContactWriter::maxNumContacts");
static_assert(sizeof(PxgRigidParticleContactWriter) == 80, "PxgRigidParticleContactWriter");
static_assert(offsetof(PxgRigidParticleContactWriter, contactByRigid) == 48, "PxgRigidParticleContactWriter::contactByRigid");
static_assert(offsetof(PxgRigidParticleContactWriter, maxContacts) == 72, "PxgRigidParticleContactWriter::maxContacts");

// The descriptor sizes the mirrors' array strides must equal.
static_assert(sizeof(PxgFEMCloth) == 416, "PxgFEMCloth (pxgSlangFEMClothDesc)");
static_assert(sizeof(PxgSoftBody) == 464, "PxgSoftBody (pxgSlangSoftBodyDesc)");
static_assert(sizeof(PxgParticleSystem) == 608, "PxgParticleSystem (PXG_PARTICLE_SYSTEM_STRIDE)");
static_assert(sizeof(PxgParticleSystemData) == 96, "PxgParticleSystemData");
static_assert(offsetof(PxgParticleSystemData, mPenBiasClamp) == 72, "PxgParticleSystemData::mPenBiasClamp");
static_assert(sizeof(PxgParticleSimBuffer) == 48, "PxgParticleSimBuffer");

// PxsDeformableVolumeMaterialData (5.12): materialModel moved to @16 (softBodySolveBda's sbLoadMaterial).
static_assert(sizeof(PxsDeformableVolumeMaterialData) == 32, "PxsDeformableVolumeMaterialData");
static_assert(offsetof(PxsDeformableVolumeMaterialData, dynamicFriction) == 8, "PxsDeformableVolumeMaterialData::dynamicFriction");
static_assert(offsetof(PxsDeformableVolumeMaterialData, elasticityDamping) == 12, "PxsDeformableVolumeMaterialData::elasticityDamping");
static_assert(offsetof(PxsDeformableVolumeMaterialData, materialModel) == 16, "PxsDeformableVolumeMaterialData::materialModel");
static_assert(offsetof(PxsDeformableVolumeMaterialData, deformThreshold) == 20, "PxsDeformableVolumeMaterialData::deformThreshold");

struct LayoutParams { uint64_t src, dst; };

static uint8_t layoutByte(size_t i) { return uint8_t((i * 131u + 7u) ^ (i >> 8) * 29u); }

static void runLayout(const char* entry, size_t structSize, const LayoutRow* rows, size_t nRows, const char* what)
{
    PxgSlangComputeDevice* dev = testDevice();
    std::vector<uint8_t> src(2 * structSize);
    for (size_t i = 0; i < src.size(); ++i)
        src[i] = layoutByte(i);
    DevBuf bSrc = mk(dev, src);
    if (!bSrc.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }
    std::vector<uint64_t> dstInit(nRows, 0xababababababababull);
    DevBuf bDst = mk(dev, dstInit);

    LayoutParams p = { bSrc.addr, bDst.addr };
    Kernel k("npDeformableRef512.spv", entry, {}, sizeof(LayoutParams));
    k.dispatch(1u, {}, p, 1u, 1u);
    dev->flush();
    const std::vector<uint64_t> got = downloadBuffer<uint64_t>(dev, bDst.h, nRows);

    uint32_t bad = 0;
    for (size_t r = 0; r < nRows; ++r)
    {
        const LayoutRow& row = rows[r];
        PXG_CHECK_MSG(row.mirrorOff == row.realOff, "%s::%s: the mirror says @%u, offsetof is %zu",
                      what, row.name, row.mirrorOff, row.realOff);
        uint64_t want = 0;
        std::memcpy(&want, &src[structSize + row.realOff], row.size);
        if (got[r] != want)
        {
            ++bad;
            std::printf("  %s::%s: device read 0x%016llx, the engine struct holds 0x%016llx @%zu\n", what,
                        row.name, (unsigned long long)got[r], (unsigned long long)want, row.realOff);
        }
    }
    PXG_CHECK_MSG(bad == 0u, "%u of %zu %s mirror fields read wrong bytes", bad, nRows, what);
}

PXG_TEST(deformableRef512Layout)
{
#define PXG_RUN_LAYOUT(T) runLayout("layout" #T, sizeof(T), k##T##Rows, sizeof(k##T##Rows) / sizeof(LayoutRow), #T)
    PXG_RUN_LAYOUT(PxgFEMCloth);
    PXG_RUN_LAYOUT(PxgSoftBody);
    PXG_RUN_LAYOUT(PxgParticleSystem);
    PXG_RUN_LAYOUT(PxgParticleSimBuffer);
    PXG_RUN_LAYOUT(PxgParticleDiffuseSimBuffer);
    PXG_RUN_LAYOUT(PxsDeformableSurfaceMaterialData);
#undef PXG_RUN_LAYOUT
}

// ===========================================================================
// Encoding
// ===========================================================================
struct RunParams
{
    uint32_t numTests, pad4;
    uint64_t queries, results, aux0, aux1;
    uint32_t auxCount, pad44;
};
static_assert(sizeof(RunParams) == 48, "npDeformableRef512 RunParams");

struct RefQuery { uint32_t op, a, b, c; uint64_t x, y, z, w; };
static_assert(sizeof(RefQuery) == 48, "RefQuery");

static uint64_t refOracle(const RefQuery& q)
{
    const PxgDeformablePairInfo p0 = { q.x, q.y };
    const PxgDeformablePairInfo p1 = { q.z, q.w };
    switch (q.op)
    {
    case 0: return PxgEncodeDeformableRef(q.a, PxgDeformableRef::Enum(q.b), q.c);
    case 1: return PxgEncodeInvalidDeformableRef(q.a);
    case 2: return PxgEncodeDeformableVertex(q.a, q.b);
    case 3: return PxgEncodeDeformableTriangle(q.a, q.b);
    case 4: return PxgEncodeDeformableTetrahedron(q.a, q.b);
    case 5: return PxgEncodeDeformableTriangleEdge(q.a, q.b, q.c);
    case 6: return PxgGetDeformableId(q.x);
    case 7: return uint64_t(PxgGetDeformableFeatureType(q.x));
    case 8: return PxgGetDeformableFeatureId(q.x);
    case 9: return PxgGetDeformableEdgeTriangle(q.x);
    case 10: return PxgGetDeformableEdgeLocalIndex(q.x);
    case 11: return PxgEncodeParticleRef(q.a, q.b);
    case 12: return PxgGetParticleIndex(q.x);
    case 13: return PxgGetParticleSystemId(q.x);
    case 14: return PxgGetContactSortKey(q.x);
    case 15: return PxgEncodeStaticShapeRef(q.a);
    case 16: return PxgEncodeRigidFilterRef(PxNodeIndex(q.x), q.a);
    case 17: return PxgIsValidPairInfo(p0) ? 1u : 0u;
    case 18: { PxgDeformablePairInfo p = p0; PxgInvalidatePairInfo(p); return p.ref1; }
    case 19: return PxgIsActivePairInfo(p0) ? 1u : 0u;
    case 20: { PxgDeformablePairInfo p = p0; PxgSetActivePairInfo(p, q.a != 0u); return p.ref1; }
    case 21: return uint64_t(uint32_t(PxgComparePairInfo(p0, p1)));
    case 22: return (q.a & (1u << q.b)) != 0u ? 1u : 0u;   // ownsEdge (deformableCollision.cuh)
    case 23: return PxNodeIndex(q.x).isStaticBody() ? 1u : 0u;
    case 24: return PxNodeIndex(q.x).index();
    case 25: return PxNodeIndex(q.x).isArticulation() ? 1u : 0u;
    case 26: return PxNodeIndex(q.x).articulationLinkId();
    case 27: return PxgEncodeRigidNode(PxNodeIndex(q.x));
    }
    return 0xdeadbeefu;
}

PXG_TEST(deformableRef512Encoding)
{
    PxgSlangComputeDevice* dev = testDevice();
    std::vector<RefQuery> qs;
    auto add = [&](uint32_t op, uint32_t a, uint32_t b, uint32_t c, uint64_t x = 0, uint64_t y = 0,
                   uint64_t z = 0, uint64_t w = 0) { qs.push_back(RefQuery{ op, a, b, c, x, y, z, w }); };

    const uint32_t ids[] = { 0u, 1u, 7u, 0x5a5u, PxgDeformableRef::MAX_NB_DEFORMABLES - 1u };
    const uint32_t feats[] = { 0u, 1u, 0x12345u, PxgDeformableRef::MAX_NB_FEATURES - 1u, PxgDeformableRef::ANY_FEATURE };
    std::vector<uint64_t> refs;
    for (uint32_t id : ids)
        for (uint32_t f : feats)
        {
            for (uint32_t t = 0; t < PxgDeformableRef::eLAST; ++t)
                add(0u, id, t, f);
            add(2u, id, f, 0u);
            add(3u, id, f, 0u);
            add(4u, id, f, 0u);
            for (uint32_t e = 0; e < 4u; ++e)
            {
                add(5u, id, f, e);
                refs.push_back(PxgEncodeDeformableTriangleEdge(id, f, e));
            }
            refs.push_back(PxgEncodeDeformableTetrahedron(id, f));
        }
    for (uint32_t id : ids)
    {
        add(1u, id, 0u, 0u);
        refs.push_back(PxgEncodeInvalidDeformableRef(id));
    }
    // Decoders over every encoded ref, also with each flag set (they must be ignored).
    for (uint64_t r : refs)
        for (uint64_t flags : { uint64_t(0), PxgDeformableRef::INVALID_FLAG, PxgDeformableRef::ACTIVE_FLAG,
                                PxgDeformableRef::INVALID_FLAG | PxgDeformableRef::ACTIVE_FLAG })
            for (uint32_t op = 6u; op <= 10u; ++op)
                add(op, 0u, 0u, 0u, r | flags);

    // Particles: the system id above the flags, the full 32-bit index below them.
    const uint32_t sys[] = { 0u, 3u, 0x3fffffffu };
    const uint32_t pidx[] = { 0u, 1u, 0x7fffffffu, 0xfffffffeu };
    for (uint32_t s : sys)
        for (uint32_t i : pidx)
        {
            add(11u, s, i, 0u);
            const uint64_t pr = PxgEncodeParticleRef(s, i);
            add(12u, 0u, 0u, 0u, pr);
            add(13u, 0u, 0u, 0u, pr);
            add(14u, 0u, 0u, 0u, pr);
        }

    // Rigids: a body, a link (bit 32 = isArticulation, link id above it), the static node.
    const uint64_t body = 1234u;
    const uint64_t link = (uint64_t(5u << 1 | 1u) << 32) | 77u;
    const uint64_t staticNode = uint64_t(PX_INVALID_NODE);
    const uint64_t worldAttach = PxNodeIndex().getInd();
    for (uint64_t n : { body, link, staticNode, worldAttach })
    {
        for (uint32_t cache : { 0u, 1u, 99u, 0xfffffffeu })
            add(16u, cache, 0u, 0u, n);
        for (uint32_t op = 23u; op <= 27u; ++op)
            add(op, 0u, 0u, 0u, n);
        add(14u, 0u, 0u, 0u, n);
    }
    for (uint32_t cache : { 0u, 1u, 99u, 0xfffffffeu })
        add(15u, cache, 0u, 0u);

    // Pair accessors: ref1 with and without the flags; ref0 holding an articulation
    // link, whose bit 32 must NOT read as invalid.
    const uint64_t tri = PxgEncodeDeformableTriangle(9u, 40u);
    for (uint64_t r1 : { tri, tri | PxgDeformableRef::INVALID_FLAG, tri | PxgDeformableRef::ACTIVE_FLAG,
                         tri | PxgDeformableRef::INVALID_FLAG | PxgDeformableRef::ACTIVE_FLAG })
        for (uint64_t r0 : { body, link })
        {
            add(17u, 0u, 0u, 0u, r0, r1);
            add(18u, 0u, 0u, 0u, r0, r1);
            add(19u, 0u, 0u, 0u, r0, r1);
            add(20u, 1u, 0u, 0u, r0, r1);
            add(20u, 0u, 0u, 0u, r0, r1);
        }
    // PxgComparePairInfo: equal, ref0 decides, ref1 decides, a difference in the top bit
    // only (the comparison must be an unsigned 64-bit one), and in the low word only.
    const uint64_t hi = uint64_t(1) << 63;
    const uint64_t cmp[][4] = {
        { 5, 6, 5, 6 }, { 5, 6, 4, 9 }, { 4, 9, 5, 6 }, { 5, 6, 5, 7 }, { 5, 7, 5, 6 },
        { hi, 0, 1, 0 }, { 1, 0, hi, 0 }, { 0, hi, 0, 1 }, { 0, 1, 0, hi },
        { uint64_t(1) << 32, 0, 0xffffffffu, 0 }, { 0xffffffffu, 0, uint64_t(1) << 32, 0 },
    };
    for (const auto& c : cmp)
        add(21u, 0u, 0u, 0u, c[0], c[1], c[2], c[3]);
    for (uint32_t o = 0; o < 8u; ++o)
        for (uint32_t e = 0; e < 3u; ++e)
            add(22u, o, e, 0u);

    const uint32_t N = uint32_t(qs.size());
    DevBuf bQ = mk(dev, qs);
    if (!bQ.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }
    DevBuf bOut = mk(dev, std::vector<uint64_t>(N, 0xababababababababull));
    RunParams p = {};
    p.numTests = N; p.queries = bQ.addr; p.results = bOut.addr;
    Kernel k("npDeformableRef512.spv", "refRun", {}, sizeof(RunParams));
    k.dispatch((N + 63u) / 64u, {}, p, 1u, 1u);
    dev->flush();
    const std::vector<uint64_t> got = downloadBuffer<uint64_t>(dev, bOut.h, N);

    uint32_t bad = 0, perOp[32] = {};
    for (uint32_t i = 0; i < N; ++i)
    {
        const uint64_t want = refOracle(qs[i]);
        ++perOp[qs[i].op];
        if (got[i] != want)
        {
            if (bad < 20u)
                std::printf("  op %u (a %u b %u c %u x 0x%016llx y 0x%016llx): got 0x%016llx, want 0x%016llx\n",
                            qs[i].op, qs[i].a, qs[i].b, qs[i].c, (unsigned long long)qs[i].x,
                            (unsigned long long)qs[i].y, (unsigned long long)got[i], (unsigned long long)want);
            ++bad;
        }
    }
    PXG_CHECK_MSG(bad == 0u, "%u of %u encoding queries disagree with the 5.12 header", bad, N);
    for (uint32_t op = 0; op <= 27u; ++op)
        PXG_CHECK_MSG(perOp[op] > 0u, "op %u is never asked", op);
}

// ===========================================================================
// Writers
// ===========================================================================
struct F4 { float x, y, z, w; };
struct WriterQuery
{
    uint32_t kind, index, matIdx, pad;
    F4 contact, normalPen, bary;
    uint64_t ref0, ref1;
};
static_assert(sizeof(WriterQuery) == 80, "WriterQuery");

// The Slang writer structs as Slang lays them out in a device buffer (no padding
// between u64 members; each struct ends on its uint rounded to 8).
struct SDbDb { uint64_t outPoint, outNormalPen, outBary0, outBary1, outInfo, total; uint32_t maxContacts, pad; };
struct SRigidDb { uint64_t outPoint, outNormalPen, outBary, outInfo, outMat, total, byRigid, tempByRigid, idxByRigid; uint32_t maxNumContacts, pad; };
struct SParticleDb { uint64_t outPoint, outNormalPen, outBary, outInfo, total, byParticle, tempByParticle, idxByParticle; uint32_t maxNumContacts, pad; };
struct SRigidParticle { uint64_t infos, normalPens, total, sortedByParticle, tempByParticle, idxByParticle, byRigid, tempByRigid, idxByRigid; uint32_t maxContacts, pad; };
struct SWriterSet { SDbDb dbdb; SRigidDb rigidDb; SParticleDb particleDb; SRigidParticle rigidParticle; };

static const uint32_t W_CAP = 16u;   // every stream's capacity; maxContacts below it

template <class T> struct Stream
{
    std::vector<T> host;
    DevBuf dev;
};

PXG_TEST(deformableRef512Writers)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint32_t MAXC = 12u;   // indices >= 12 must be refused

    auto f4s = [](uint32_t salt) {
        std::vector<float4> v(W_CAP);
        for (uint32_t i = 0; i < W_CAP; ++i) v[i] = make_float4(-1.0f - i - salt, -2.0f, -3.0f, -4.0f);
        return v;
    };
    // Host streams the real writers write; the device streams start from the same bytes.
    std::vector<float4> hPoint[4], hNormalPen[4], hBary[4], hBary1;
    std::vector<PxgDeformablePairInfo> hInfo[4];
    for (int s = 0; s < 4; ++s)
    {
        hPoint[s] = f4s(10u * s); hNormalPen[s] = f4s(10u * s + 1u); hBary[s] = f4s(10u * s + 2u);
        hInfo[s].assign(W_CAP, PxgDeformablePairInfo{ 0x1111111111111111ull, 0x2222222222222222ull });
    }
    hBary1 = f4s(99u);
    std::vector<uint16_t> hMat(W_CAP, 0xabcdu);
    std::vector<uint64_t> hByRigid(W_CAP, 0x3333ull), hByParticle(W_CAP, 0x4444ull), hRpByParticle(W_CAP, 0x5555ull), hRpByRigid(W_CAP, 0x6666ull);
    std::vector<uint32_t> hTempR(W_CAP, 7u), hIdxR(W_CAP, 8u), hTempP(W_CAP, 9u), hIdxP(W_CAP, 10u),
        hRpTempP(W_CAP, 11u), hRpIdxP(W_CAP, 12u), hRpTempR(W_CAP, 13u), hRpIdxR(W_CAP, 14u);
    std::vector<uint32_t> hCount(4, 0u);

    // Device copies.
    DevBuf dPoint[4], dNormalPen[4], dBary[4], dInfo[4];
    for (int s = 0; s < 4; ++s)
    {
        dPoint[s] = mk(dev, hPoint[s]); dNormalPen[s] = mk(dev, hNormalPen[s]);
        dBary[s] = mk(dev, hBary[s]); dInfo[s] = mk(dev, hInfo[s]);
    }
    if (!dPoint[0].addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }
    DevBuf dBary1 = mk(dev, hBary1), dMat = mk(dev, hMat);
    DevBuf dByRigid = mk(dev, hByRigid), dByParticle = mk(dev, hByParticle), dRpByParticle = mk(dev, hRpByParticle),
           dRpByRigid = mk(dev, hRpByRigid);
    DevBuf dTempR = mk(dev, hTempR), dIdxR = mk(dev, hIdxR), dTempP = mk(dev, hTempP), dIdxP = mk(dev, hIdxP),
           dRpTempP = mk(dev, hRpTempP), dRpIdxP = mk(dev, hRpIdxP), dRpTempR = mk(dev, hRpTempR), dRpIdxR = mk(dev, hRpIdxR);
    DevBuf dCount = mk(dev, hCount);

    // The real writers over host memory.
    PxgDbDbContactWriter wDbDb = { hPoint[0].data(), hNormalPen[0].data(), hBary[0].data(), hBary1.data(),
                                   hInfo[0].data(), &hCount[0], MAXC };
    PxgRigidDbContactWriter wRigidDb = { hPoint[1].data(), hNormalPen[1].data(), hBary[1].data(), hInfo[1].data(),
                                         hMat.data(), &hCount[1], hByRigid.data(), hTempR.data(), hIdxR.data(), MAXC };
    PxgParticleDbContactWriter wParticleDb = { hPoint[2].data(), hNormalPen[2].data(), hBary[2].data(), hInfo[2].data(),
                                               &hCount[2], hByParticle.data(), hTempP.data(), hIdxP.data(), MAXC };
    PxgRigidParticleContactWriter wRp = { hInfo[3].data(), hNormalPen[3].data(), &hCount[3], hRpByParticle.data(),
                                          hRpTempP.data(), hRpIdxP.data(), reinterpret_cast<PxNodeIndex*>(hRpByRigid.data()),
                                          hRpTempR.data(), hRpIdxR.data(), MAXC };

    SWriterSet ws = {};
    ws.dbdb = { dPoint[0].addr, dNormalPen[0].addr, dBary[0].addr, dBary1.addr, dInfo[0].addr, dCount.addr, MAXC, 0u };
    ws.rigidDb = { dPoint[1].addr, dNormalPen[1].addr, dBary[1].addr, dInfo[1].addr, dMat.addr, dCount.addr + 4u,
                   dByRigid.addr, dTempR.addr, dIdxR.addr, MAXC, 0u };
    ws.particleDb = { dPoint[2].addr, dNormalPen[2].addr, dBary[2].addr, dInfo[2].addr, dCount.addr + 8u,
                      dByParticle.addr, dTempP.addr, dIdxP.addr, MAXC, 0u };
    ws.rigidParticle = { dInfo[3].addr, dNormalPen[3].addr, dCount.addr + 12u, dRpByParticle.addr, dRpTempP.addr,
                         dRpIdxP.addr, dRpByRigid.addr, dRpTempR.addr, dRpIdxR.addr, MAXC, 0u };
    DevBuf dWs = mk(dev, std::vector<SWriterSet>{ ws });

    // The queries: every write method, on distinct slots; material indices 0..3 share two
    // words (both halves of each), and one refused index per writer.
    std::vector<WriterQuery> qs;
    uint32_t salt = 0;
    auto add = [&](uint32_t kind, uint32_t index, uint64_t r0, uint64_t r1, uint32_t mat = 0u) {
        ++salt;
        WriterQuery q = {};
        q.kind = kind; q.index = index; q.matIdx = mat;
        q.contact = { 1.0f * salt, 2.0f, 3.0f, 0.25f };
        q.normalPen = { 0.0f, 1.0f, 0.0f, -0.5f * salt };
        q.bary = { 0.25f, 0.5f, 0.125f, 0.125f + salt };
        q.ref0 = r0; q.ref1 = r1;
        qs.push_back(q);
    };
    const uint64_t vtx = PxgEncodeDeformableVertex(3u, 17u), tri = PxgEncodeDeformableTriangle(4u, 18u),
                   tet = PxgEncodeDeformableTetrahedron(5u, 19u);
    const uint64_t rigid = (uint64_t(9u << 1 | 1u) << 32) | 41u, part = PxgEncodeParticleRef(2u, 0x80000001u);
    add(0u, 0u, vtx, tet); add(1u, 1u, tri, tet); add(2u, 2u, vtx, tet); add(0u, MAXC, vtx, tet);
    add(3u, 0u, rigid, tet, 0x1234u); add(4u, 1u, rigid, tri, 0xfffeu); add(5u, 2u, rigid, vtx, 0u);
    add(3u, 3u, uint64_t(PX_INVALID_NODE), tet, 0x10001u);   // the PxU16 narrowing
    add(4u, MAXC + 1u, rigid, tri, 7u);
    add(6u, 0u, part, tet); add(7u, 1u, part, tri); add(6u, MAXC, part, tet);
    add(8u, 0u, rigid, part); add(8u, 5u, uint64_t(PX_INVALID_NODE), part); add(8u, MAXC, rigid, part);

    std::vector<uint64_t> wantOk;
    for (const WriterQuery& q : qs)
    {
        const float4 c = make_float4(q.contact.x, q.contact.y, q.contact.z, q.contact.w);
        const float4 n = make_float4(q.normalPen.x, q.normalPen.y, q.normalPen.z, q.normalPen.w);
        const float4 b = make_float4(q.bary.x, q.bary.y, q.bary.z, q.bary.w);
        bool ok = true;
        switch (q.kind)
        {
        case 0: ok = wDbDb.writeVolumeVolumeContact(q.index, c, n, q.ref0, q.ref1); break;
        case 1: ok = wDbDb.writeSurfaceVolumeContact(q.index, c, n, q.ref0, b, q.ref1); break;
        case 2: ok = wDbDb.writeSurfaceVolumeContact(q.index, c, n, q.ref0, q.ref1); break;
        case 3: ok = wRigidDb.writeVolumeContact(q.index, c, n, q.ref0, q.matIdx, q.ref1); break;
        case 4: ok = wRigidDb.writeSurfaceContact(q.index, c, n, q.ref0, q.matIdx, q.ref1, b); break;
        case 5: ok = wRigidDb.writeSurfaceContact(q.index, c, n, q.ref0, q.matIdx, q.ref1); break;
        case 6: ok = wParticleDb.writeVolumeContact(q.index, c, n, q.ref0, q.ref1); break;
        case 7: ok = wParticleDb.writeSurfaceContact(q.index, c, n, q.ref0, q.ref1, b); break;
        case 8: wRp.writeContact(q.index, PxVec4(n.x, n.y, n.z, n.w), PxNodeIndex(q.ref0), q.ref1); break;
        }
        wantOk.push_back(ok ? 1u : 0u);
    }

    const uint32_t N = uint32_t(qs.size());
    DevBuf bQ = mk(dev, qs);
    DevBuf bOut = mk(dev, std::vector<uint64_t>(N, 0xababababababababull));
    RunParams p = {};
    p.numTests = N; p.queries = bQ.addr; p.results = bOut.addr; p.aux0 = dWs.addr;
    Kernel k("npDeformableRef512.spv", "writerRun", {}, sizeof(RunParams));
    k.dispatch((N + 63u) / 64u, {}, p, 1u, 1u);
    dev->flush();

    PXG_CHECK(downloadBuffer<uint64_t>(dev, bOut.h, N) == wantOk);
    auto same = [&](const char* what, PxgBufferHandle h, const void* host, size_t bytes) {
        const std::vector<uint8_t> got = downloadBuffer<uint8_t>(dev, h, bytes);
        PXG_CHECK_MSG(std::memcmp(got.data(), host, bytes) == 0, "%s differs from the real writer's bytes", what);
    };
    const char* sn[4] = { "DbDb", "RigidDb", "ParticleDb", "RigidParticle" };
    char name[64];
    for (int s = 0; s < 4; ++s)
    {
        std::snprintf(name, sizeof(name), "%s outPoint", sn[s]); same(name, dPoint[s].h, hPoint[s].data(), W_CAP * 16);
        std::snprintf(name, sizeof(name), "%s outNormalPen", sn[s]); same(name, dNormalPen[s].h, hNormalPen[s].data(), W_CAP * 16);
        std::snprintf(name, sizeof(name), "%s outBarycentric", sn[s]); same(name, dBary[s].h, hBary[s].data(), W_CAP * 16);
        std::snprintf(name, sizeof(name), "%s outContactInfo", sn[s]); same(name, dInfo[s].h, hInfo[s].data(), W_CAP * 16);
    }
    same("DbDb outBarycentric1", dBary1.h, hBary1.data(), W_CAP * 16);
    same("RigidDb outRigidMaterialIndex", dMat.h, hMat.data(), W_CAP * 2);
    same("RigidDb contactByRigid", dByRigid.h, hByRigid.data(), W_CAP * 8);
    same("RigidDb tempContactByRigid", dTempR.h, hTempR.data(), W_CAP * 4);
    same("RigidDb contactIndexSortedByRigid", dIdxR.h, hIdxR.data(), W_CAP * 4);
    same("ParticleDb contactByParticle", dByParticle.h, hByParticle.data(), W_CAP * 8);
    same("ParticleDb tempContactByParticle", dTempP.h, hTempP.data(), W_CAP * 4);
    same("ParticleDb contactIndexSortedByParticle", dIdxP.h, hIdxP.data(), W_CAP * 4);
    same("RigidParticle contactSortedByParticle", dRpByParticle.h, hRpByParticle.data(), W_CAP * 8);
    same("RigidParticle tempContactByParticle", dRpTempP.h, hRpTempP.data(), W_CAP * 4);
    same("RigidParticle contactIndexSortedByParticle", dRpIdxP.h, hRpIdxP.data(), W_CAP * 4);
    same("RigidParticle contactByRigid", dRpByRigid.h, hRpByRigid.data(), W_CAP * 8);
    same("RigidParticle tempContactByRigid", dRpTempR.h, hRpTempR.data(), W_CAP * 4);
    same("RigidParticle contactIndexSortedByRigid", dRpIdxR.h, hRpIdxR.data(), W_CAP * 4);
    // The writers never touch the counters; the kernels own them.
    same("totalContactCount", dCount.h, hCount.data(), 16);
    // Coverage: the first material word's two halves were written by two invocations.
    PXG_CHECK(hMat[0] == 0x1234u && hMat[1] == 0xfffeu && hMat[2] == 0u && hMat[3] == 0x0001u);
}

// ===========================================================================
// Filter
// ===========================================================================
struct FilterQuery { uint32_t op, a, b, c, d, e, f, g; uint64_t x, y; };
static_assert(sizeof(FilterQuery) == 48, "FilterQuery");

// find() of deformableElementFilter.cuh (5.12), over the real PxgComparePairInfo.
static bool findRef(const std::vector<PxgDeformablePairInfo>& pairs, uint32_t nbPairs, const PxgDeformablePairInfo& pair)
{
    PxU32 l = 0, r = nbPairs;
    while (l < r)
    {
        PxU32 mid = (l + r) / 2;
        PxI32 cmp = PxgComparePairInfo(pair, pairs[mid]);
        if (cmp == 0)
            return true;
        if (cmp < 0)
            r = mid;
        else
            l = mid + 1;
    }
    return false;
}

static bool filterOracle(const std::vector<PxgDeformablePairInfo>& pairs, const FilterQuery& q)
{
    const uint32_t n = q.op >= 100u ? 0u : uint32_t(pairs.size());
    PxgDeformablePairInfo pr = {};
    switch (q.op % 100u)
    {
    case 0: pr = { q.x, q.y }; break;
    case 1: pr = { PxgEncodeDeformableVertex(q.a, q.b), PxgEncodeDeformableTetrahedron(q.c, q.d) }; break;
    case 2: pr = { PxgEncodeRigidFilterRef(PxNodeIndex(q.x), q.a), PxgEncodeDeformableTetrahedron(q.b, q.c) }; break;
    case 3: pr = { PxgEncodeDeformableTetrahedron(q.a, q.b), PxgEncodeDeformableVertex(q.c, q.d) }; break;
    case 4: pr = { PxgEncodeDeformableTetrahedron(q.a, q.b), PxgEncodeDeformableTriangle(q.c, q.d) }; break;
    case 5: pr = { PxgEncodeRigidFilterRef(PxNodeIndex(q.x), q.a), PxgEncodeDeformableVertex(q.b, q.c) }; break;
    case 6: pr = { PxgEncodeRigidFilterRef(PxNodeIndex(q.x), q.a), PxgEncodeDeformableTriangle(q.b, q.c) }; break;
    case 7: pr = { PxgEncodeDeformableVertex(q.a, q.b), PxgEncodeDeformableTriangle(q.c, q.d) }; break;
    case 8: pr = { PxgEncodeDeformableTriangleEdge(q.a, q.b, q.c), PxgEncodeDeformableTriangleEdge(q.d, q.e, q.f) }; break;
    }
    return findRef(pairs, n, pr);
}

PXG_TEST(deformableRef512Filter)
{
    PxgSlangComputeDevice* dev = testDevice();
    const uint64_t body = 1234u, link = (uint64_t(3u << 1 | 1u) << 32) | 50u, stat = uint64_t(PX_INVALID_NODE);
    const uint32_t ANY = PxgDeformableRef::ANY_FEATURE;

    // One array holding every pair type the five per-actor arrays can carry; sorted with
    // the real comparison.  Several entries share ref0 and differ in ref1 (and vice
    // versa) so both halves of the comparison choose between entries.
    std::vector<PxgDeformablePairInfo> pairs = {
        { PxgEncodeDeformableVertex(1, 10), PxgEncodeDeformableTetrahedron(2, 20) },
        { PxgEncodeDeformableVertex(1, 10), PxgEncodeDeformableTetrahedron(2, 21) },
        { PxgEncodeDeformableVertex(1, ANY), PxgEncodeDeformableTetrahedron(2, 22) },
        { PxgEncodeRigidFilterRef(PxNodeIndex(body), 7), PxgEncodeDeformableTetrahedron(2, 23) },
        { PxgEncodeRigidFilterRef(PxNodeIndex(stat), 7), PxgEncodeDeformableTetrahedron(2, 24) },
        { PxgEncodeRigidFilterRef(PxNodeIndex(link), 8), PxgEncodeDeformableTetrahedron(2, ANY) },
        { PxgEncodeDeformableTetrahedron(2, 30), PxgEncodeDeformableVertex(3, 31) },
        { PxgEncodeDeformableTetrahedron(2, 30), PxgEncodeDeformableTriangle(3, 31) },
        { PxgEncodeDeformableTetrahedron(2, ANY), PxgEncodeDeformableTriangle(3, 32) },
        { PxgEncodeRigidFilterRef(PxNodeIndex(body), 7), PxgEncodeDeformableVertex(3, 40) },
        { PxgEncodeRigidFilterRef(PxNodeIndex(stat), 9), PxgEncodeDeformableTriangle(3, 41) },
        { PxgEncodeDeformableVertex(3, 50), PxgEncodeDeformableTriangle(4, 51) },
        { PxgEncodeDeformableTriangleEdge(3, 60, 1), PxgEncodeDeformableTriangleEdge(4, 61, 2) },
        { PxgEncodeDeformableTriangleEdge(3, 60, 1), PxgEncodeDeformableTriangleEdge(4, ANY, 0) },
    };
    std::sort(pairs.begin(), pairs.end(), [](const PxgDeformablePairInfo& a, const PxgDeformablePairInfo& b) {
        return PxgComparePairInfo(a, b) < 0;
    });
    DevBuf bPairs = mk(dev, pairs);
    if (!bPairs.addr) { std::printf("  NOTE: no BDA -> skip\n"); return; }
    DevBuf bSafe = mk(dev, std::vector<uint64_t>(4, 0ull));

    std::vector<FilterQuery> qs;
    auto add = [&](uint32_t op, uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t e = 0, uint32_t f = 0,
                   uint64_t x = 0, uint64_t y = 0) {
        qs.push_back(FilterQuery{ op, a, b, c, d, e, f, 0u, x, y });
        qs.push_back(FilterQuery{ op + 100u, a, b, c, d, e, f, 0u, x, y });   // the empty array
    };
    for (const PxgDeformablePairInfo& pr : pairs)
    {
        add(0u, 0, 0, 0, 0, 0, 0, pr.ref0, pr.ref1);
        add(0u, 0, 0, 0, 0, 0, 0, pr.ref0, pr.ref1 + 1u);
        add(0u, 0, 0, 0, 0, 0, 0, pr.ref0 - 1u, pr.ref1);
        add(0u, 0, 0, 0, 0, 0, 0, pr.ref0, pr.ref1 | PxgDeformableRef::ACTIVE_FLAG);
    }
    // Volume-volume, including the wildcard probe the softbody midphase issues.
    add(1u, 1, 10, 2, 20); add(1u, 1, 10, 2, 21); add(1u, 1, 11, 2, 20); add(1u, 1, ANY, 2, 22); add(1u, 1, 10, 2, 22);
    // Rigid-volume: a body, a static keyed on its shape's cache id, a link with a wildcard.
    add(2u, 7, 2, 23, 0, 0, 0, body); add(2u, 8, 2, 23, 0, 0, 0, body);   // a body ignores the cache id
    add(2u, 7, 2, 24, 0, 0, 0, stat); add(2u, 9, 2, 24, 0, 0, 0, stat);   // a static does not
    add(2u, 8, 2, ANY, 0, 0, 0, link); add(2u, 8, 2, 5, 0, 0, 0, link);
    // Volume-surface, both overloads over one array: a vertex and a triangle of the same id.
    add(3u, 2, 30, 3, 31); add(4u, 2, 30, 3, 31); add(3u, 2, 30, 3, 32); add(4u, 2, ANY, 3, 32); add(4u, 2, 30, 3, 32);
    // Rigid-surface.
    add(5u, 7, 3, 40, 0, 0, 0, body); add(6u, 7, 3, 40, 0, 0, 0, body); add(6u, 9, 3, 41, 0, 0, 0, stat);
    add(6u, 7, 3, 41, 0, 0, 0, stat);
    // Surface-surface vertex-triangle and edge-edge (the local edge is part of the key).
    add(7u, 3, 50, 4, 51); add(7u, 3, 51, 4, 51);
    add(8u, 3, 60, 1, 4, 61, 2); add(8u, 3, 60, 2, 4, 61, 2); add(8u, 3, 60, 1, 4, 61, 1); add(8u, 3, 60, 1, 4, ANY, 0);

    const uint32_t N = uint32_t(qs.size());
    DevBuf bQ = mk(dev, qs);
    DevBuf bOut = mk(dev, std::vector<uint64_t>(N, 0xababababababababull));
    RunParams p = {};
    p.numTests = N; p.queries = bQ.addr; p.results = bOut.addr;
    p.aux0 = bPairs.addr; p.aux1 = bSafe.addr; p.auxCount = uint32_t(pairs.size());
    Kernel k("npDeformableRef512.spv", "filterRun", {}, sizeof(RunParams));
    k.dispatch((N + 63u) / 64u, {}, p, 1u, 1u);
    dev->flush();
    const std::vector<uint64_t> got = downloadBuffer<uint64_t>(dev, bOut.h, N);

    uint32_t bad = 0, hits = 0, misses = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        const uint64_t want = filterOracle(pairs, qs[i]) ? 1u : 0u;
        (want ? hits : misses) += (qs[i].op < 100u) ? 1u : 0u;
        if (got[i] != want)
        {
            ++bad;
            std::printf("  filter query %u (op %u): got %llu, want %llu\n", i, qs[i].op,
                        (unsigned long long)got[i], (unsigned long long)want);
        }
    }
    PXG_CHECK_MSG(bad == 0u, "%u of %u filter queries disagree with deformableElementFilter.cuh", bad, N);
    PXG_CHECK_MSG(hits >= 20u && misses >= 20u, "too few hits (%u) or misses (%u) to tell the search apart", hits, misses);
}

} // namespace
