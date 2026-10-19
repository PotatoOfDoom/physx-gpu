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

// npDeformableRef512.slang -- fixture for the PhysX 5.12 deformable shared modules.
//
// Four groups of entry points, driven by testDeformableRef512.cpp against the REAL
// 5.12 headers on the host:
//   layout*      read every field of a descriptor mirror (pxgSlangFEMClothDesc,
//                pxgSlangSoftBodyDesc, pxgSlangParticleDesc) from element [1] of a
//                two-element array of the real engine struct, so both the field offsets
//                and the array stride are measured, not asserted.  The bodies are one
//                line per mirrored field.
//   refRun       every encode / decode / pair accessor of pxgSlangDeformableRef.
//   writerRun    every write method of the four pxgSlangDeformableWriters writers,
//                including both halves of one PxU16 material word.
//   filterRun    dfFind and the eight typed find helpers of pxgSlangDefFilter, plus a
//                zero-length NULL filter array behind dfFilterArray's safe address.
import pxgSlangDeformableRef;
import pxgSlangDeformableWriters;
import pxgSlangDefFilter;
import pxgSlangFEMClothDesc;
import pxgSlangSoftBodyDesc;
import pxgSlangParticleDesc;

struct LayoutParams
{
    uint64_t src;   // @0  two elements of the real engine struct
    uint64_t dst;   // @8  uint64_t per mirrored field
}
[[vk::push_constant]] ConstantBuffer<LayoutParams> gLay;

[shader("compute")]
[numthreads(1, 1, 1)]
void layoutPxgFEMCloth(uint3 tid : SV_DispatchThreadID)
{
    Ptr<PxgFEMClothGpu> s = Ptr<PxgFEMClothGpu>(gLay.src);
    Ptr<uint64_t> o = Ptr<uint64_t>(gLay.dst);
    o[0] = s[1].mTriMeshData;
    o[1] = s[1].mVelocity_InvMass;
    o[2] = s[1].mPosition_InvMass;
    o[3] = s[1].mRestPosition;
    o[4] = s[1].mPrevPositionInContactOffset;
    o[5] = s[1].mPrevPositionInRestOffset;
    o[6] = s[1].mDynamicFrictions;
    o[7] = s[1].mMaterialIndices;
    o[8] = s[1].mTrianglesWithOwnedEdges;
    o[9] = s[1].mTriangleVertexIndices_ownedEdges;
    o[10] = s[1].mOrderedNonSharedTriangleVertexIndices_triIndex;
    o[11] = s[1].mOrderedSharedTriangleLambdas;
    o[12] = s[1].mOrderedNonSharedTriangleLambdas;
    o[13] = s[1].mOrderedNonSharedTriangleRestPoseInv;
    o[14] = s[1].mOrderedSharedTrianglePairVertexIndices;
    o[15] = s[1].mOrderedNonSharedTrianglePairVertexIndices;
    o[16] = s[1].mOrderedSharedRestBendingAngle_flexuralStiffness_damping;
    o[17] = s[1].mOrderedNonSharedRestBendingAngle_flexuralStiffness_damping;
    o[18] = s[1].mSharedBendingLambdas;
    o[19] = s[1].mNonSharedBendingLambdas;
    o[20] = uint64_t(s[1].mNonSharedTriPair_hasActiveBendingWord);
    o[21] = s[1].mOrderedSharedRestEdge0_edge1;
    o[22] = s[1].mOrderedSharedRestEdgeLength_material0_material1;
    o[23] = s[1].mPosition_InvMassCP;
    o[24] = s[1].mNonSharedTriAccumulatedPartitionsCP;
    o[25] = s[1].mSharedTriPairRemapOutputCP;
    o[26] = s[1].mNonSharedTriPairRemapOutputCP;
    o[27] = s[1].mSharedTriPairAccumulatedCopiesCP;
    o[28] = s[1].mNonSharedTriPairAccumulatedCopiesCP;
    o[29] = s[1].mSharedTriPairAccumulatedPartitionsCP;
    o[30] = s[1].mNonSharedTriPairAccumulatedPartitionsCP;
    o[31] = s[1].mDeltaPos;
    o[32] = s[1].mAccumulatedDeltaPos;
    o[33] = s[1].mAccumulatedDeltaVel;
    o[34] = s[1].mPackedNodeBounds;
    o[35] = s[1].mRigidSurfaceFilterPairs;
    o[36] = s[1].mSurfaceSurfaceFilterPairs;
    o[37] = uint64_t(s[1].mNumRigidSurfaceFilterPairs);
    o[38] = uint64_t(s[1].mNumSurfaceSurfaceFilterPairs);
    o[39] = uint64_t(asuint(s[1].mLinearDamping));
    o[40] = uint64_t(asuint(s[1].mMaxLinearVelocity));
    o[41] = uint64_t(asuint(s[1].mPenBiasClamp));
    o[42] = uint64_t(asuint(s[1].mSettlingThreshold));
    o[43] = uint64_t(asuint(s[1].mSleepThreshold));
    o[44] = uint64_t(asuint(s[1].mSettlingDamping));
    o[45] = uint64_t(asuint(s[1].mSelfCollisionFilterDistance));
    o[46] = uint64_t(s[1].mNbVerts);
    o[47] = uint64_t(s[1].mNbTriangles);
    o[48] = uint64_t(s[1].mNbNonSharedTriangles);
    o[49] = uint64_t(s[1].mNbTrianglesWithOwnedEdges);
    o[50] = uint64_t(s[1].mNbTrianglePairs);
    o[51] = uint64_t(s[1].mNbSharedTrianglePairs);
    o[52] = uint64_t(s[1].mNbNonSharedTrianglePairs);
    o[53] = uint64_t(s[1].mNbNonSharedTriPartitions);
    o[54] = uint64_t(s[1].mNbSharedTriPairPartitions);
    o[55] = uint64_t(s[1].mNbNonSharedTriPairPartitions);
    o[56] = uint64_t(s[1].mNonSharedTriClusterId);
    o[57] = uint64_t(s[1].mSharedTriPairClusterId);
    o[58] = uint64_t(s[1].mNonSharedTriPairClusterId);
    o[59] = uint64_t(s[1].mElementIndex);
    o[60] = uint64_t(s[1].mGpuRemapIndex);
    o[61] = uint64_t(s[1].mFlagsWord);
    o[62] = uint64_t(s[1].mIsActive);
    o[63] = uint64_t(asuint(s[1].mRestOffset));
    o[64] = uint64_t(asuint(s[1].mOriginalContactOffset));
    o[65] = uint64_t(s[1].mNbCollisionPairUpdatesPerTimestep);
    o[66] = uint64_t(s[1].mNbCollisionSubsteps);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void layoutPxgSoftBody(uint3 tid : SV_DispatchThreadID)
{
    Ptr<PxgSoftBodyGpu> s = Ptr<PxgSoftBodyGpu>(gLay.src);
    Ptr<uint64_t> o = Ptr<uint64_t>(gLay.dst);
    o[0] = s[1].mTetMeshData;
    o[1] = s[1].mTetMeshSurfaceHint;
    o[2] = s[1].mTetIndices;
    o[3] = s[1].mTetIndicesRemapTable;
    o[4] = s[1].mTetraStresses;
    o[5] = s[1].mTetraRestPoses;
    o[6] = s[1].mTetraStressCoefficient;
    o[7] = s[1].mTetraRotations;
    o[8] = s[1].mPosition_InvMass;
    o[9] = s[1].mRestPosition;
    o[10] = s[1].mSimTetIndices;
    o[11] = s[1].mSimVelocity_InvMass;
    o[12] = s[1].mSimPosition_InvMass;
    o[13] = s[1].mSimKinematicTarget;
    o[14] = s[1].mVertsAreDeformed;
    o[15] = s[1].mVertsCanNotDeform;
    o[16] = s[1].mSimDeltaPos;
    o[17] = s[1].mSimTetraRestPoses;
    o[18] = s[1].mOrigQinv;
    o[19] = s[1].mSimOrderedTetrahedrons;
    o[20] = s[1].mSimJacobiVertIndices;
    o[21] = s[1].mSimTetraRotations;
    o[22] = s[1].mSimVelocity_invMassCP;
    o[23] = s[1].mSimPosition_InvMassCP;
    o[24] = s[1].mSimRemapOutputCP;
    o[25] = s[1].mSimAccumulatedCopiesCP;
    o[26] = s[1].mSimAccumulatedPartitionsCP;
    o[27] = s[1].mVertsBarycentricInGridModel;
    o[28] = s[1].mVertsRemapInGridModel;
    o[29] = s[1].mTetsRemapColToSim;
    o[30] = s[1].mTetsAccumulatedRemapColToSim;
    o[31] = s[1].mSurfaceVertsHint;
    o[32] = s[1].mSurfaceVertToTetRemap;
    o[33] = s[1].mSimTetraMultipliers;
    o[34] = s[1].mSimDelta;
    o[35] = s[1].mPackedNodeBounds;
    o[36] = s[1].mOrderedMaterialIndices;
    o[37] = s[1].mMaterialIndices;
    o[38] = s[1].mRigidVolumeFilterPairs;
    o[39] = s[1].mVolumeVolumeFilterPairs;
    o[40] = s[1].mVolumeSurfaceFilterPairs;
    o[41] = uint64_t(s[1].mNumRigidVolumeFilterPairs);
    o[42] = uint64_t(s[1].mNumVolumeVolumeFilterPairs);
    o[43] = uint64_t(s[1].mNumVolumeSurfaceFilterPairs);
    o[44] = uint64_t(asuint(s[1].mLinearDamping));
    o[45] = uint64_t(asuint(s[1].mMaxLinearVelocity));
    o[46] = uint64_t(asuint(s[1].mPenBiasClamp));
    o[47] = uint64_t(asuint(s[1].mSettlingThreshold));
    o[48] = uint64_t(asuint(s[1].mSleepThreshold));
    o[49] = uint64_t(asuint(s[1].mSettlingDamping));
    o[50] = uint64_t(asuint(s[1].mSelfCollisionFilterDistance));
    o[51] = uint64_t(asuint(s[1].mSelfCollisionStressTolerance));
    o[52] = uint64_t(asuint(s[1].mInitialRotationX));
    o[53] = uint64_t(asuint(s[1].mInitialRotationY));
    o[54] = uint64_t(asuint(s[1].mInitialRotationZ));
    o[55] = uint64_t(asuint(s[1].mInitialRotationW));
    o[56] = uint64_t(s[1].mNumVerts);
    o[57] = uint64_t(s[1].mNumTets);
    o[58] = uint64_t(s[1].mNumVertsGM);
    o[59] = uint64_t(s[1].mNumTetsGM);
    o[60] = uint64_t(s[1].mNumPartitionsGM);
    o[61] = uint64_t(s[1].mElementIndex);
    o[62] = uint64_t(s[1].mGpuRemapIndex);
    o[63] = uint64_t(s[1].mFlagsWord);
    o[64] = s[1].mSimPullIndices;
    o[65] = uint64_t(s[1].mAwake);
    o[66] = uint64_t(s[1].mNumTetsPerElement);
    o[67] = uint64_t(s[1].mNumJacobiVertices);
    o[68] = uint64_t(asuint(s[1].mRestOffset));
    o[69] = uint64_t(asuint(s[1].mOriginalContactOffset));
    o[70] = uint64_t(asuint(s[1].mJacobiScale));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void layoutPxgParticleSystem(uint3 tid : SV_DispatchThreadID)
{
    Ptr<PxgParticleSystemGpu> s = Ptr<PxgParticleSystemGpu>(gLay.src);
    Ptr<uint64_t> o = Ptr<uint64_t>(gLay.dst);
    o[0] = s[1].mUnsortedPositions_InvMass;
    o[1] = s[1].mUnsortedVelocities;
    o[2] = s[1].mUnsortedPhaseArray;
    o[3] = s[1].mSortedPositions_InvMass;
    o[4] = s[1].mSortedVelocities;
    o[5] = s[1].mSortedPhaseArray;
    o[6] = s[1].mUnsortedToSortedMapping;
    o[7] = s[1].mSortedToUnsortedMapping;
    o[8] = s[1].mParticleSelfCollisionCount;
    o[9] = s[1].mCollisionIndex;
    o[10] = s[1].mParticleMaterials;
    o[11] = uint64_t(asuint(s[1].mCommonData_GridCellWidth));
    o[12] = uint64_t(s[1].mCommonData_GridSizeX);
    o[13] = uint64_t(s[1].mCommonData_GridSizeY);
    o[14] = uint64_t(s[1].mCommonData_GridSizeZ);
    o[15] = uint64_t(asuint(s[1].mCommonData_ParticleContactDistance));
    o[16] = uint64_t(asuint(s[1].mCommonData_ParticleContactDistanceInv));
    o[17] = uint64_t(asuint(s[1].mCommonData_ParticleContactDistanceSq));
    o[18] = uint64_t(s[1].mCommonData_NumParticles);
    o[19] = uint64_t(s[1].mCommonData_MaxParticles);
    o[20] = uint64_t(s[1].mCommonData_MaxNeighborhood);
    o[21] = uint64_t(s[1].mCommonData_MaxDiffuseParticles);
    o[22] = uint64_t(s[1].mCommonData_NumParticleBuffers);
    o[23] = s[1].mOriginPos_InvMass;
    o[24] = s[1].mGridParticleHash;
    o[25] = s[1].mAccumDeltaP;
    o[26] = s[1].mSortedDeltaP;
    o[27] = s[1].mCellStart;
    o[28] = s[1].mCellEnd;
    o[29] = s[1].mCollisionHeaders;
    o[30] = s[1].mDensityCollisionImpulses;
    o[31] = s[1].mOneWayContactInfos;
    o[32] = s[1].mOneWayForces;
    o[33] = s[1].mOneWayNodeIndex;
    o[34] = s[1].mOneWaySurfaceVelocity;
    o[35] = s[1].mOneWayContactCount;
    o[36] = s[1].mRestArray;
    o[37] = uint64_t(asuint(s[1].mData_BoundCenterX));
    o[38] = uint64_t(asuint(s[1].mData_BoundCenterY));
    o[39] = uint64_t(asuint(s[1].mData_BoundCenterZ));
    o[40] = uint64_t(s[1].mData_ElementIndex);
    o[41] = uint64_t(s[1].mData_RemapIndex);
    o[42] = uint64_t(asuint(s[1].mData_RestOffset));
    o[43] = uint64_t(asuint(s[1].mData_SolidRestOffset));
    o[44] = uint64_t(asuint(s[1].mData_Spiky1));
    o[45] = uint64_t(asuint(s[1].mData_Spiky2));
    o[46] = uint64_t(asuint(s[1].mData_RestDensity));
    o[47] = uint64_t(asuint(s[1].mData_RestDensityBoundary));
    o[48] = uint64_t(asuint(s[1].mData_FluidSurfaceConstraintScale));
    o[49] = uint64_t(asuint(s[1].mData_LambdaScale));
    o[50] = uint64_t(asuint(s[1].mData_RelaxationFactor));
    o[51] = uint64_t(asuint(s[1].mData_FluidRestOffset));
    o[52] = uint64_t(asuint(s[1].mData_InvRestDensity));
    o[53] = uint64_t(s[1].mData_Flags);
    o[54] = uint64_t(asuint(s[1].mData_MaxVelocity));
    o[55] = uint64_t(asuint(s[1].mData_PenBiasClamp));
    o[56] = uint64_t(s[1].mData_LockFlagsWord);
    o[57] = uint64_t(s[1].mData_NumPhaseToMaterials);
    o[58] = uint64_t(asuint(s[1].mData_WindX));
    o[59] = uint64_t(asuint(s[1].mData_WindY));
    o[60] = uint64_t(asuint(s[1].mData_WindZ));
    o[61] = s[1].mDensity;
    o[62] = s[1].mStaticDensity;
    o[63] = s[1].mSurfaceNormal;
    o[64] = s[1].mDelta;
    o[65] = s[1].mCurl;
    o[66] = s[1].mNormalArray;
    o[67] = s[1].mSortedOriginPos_InvMass;
    o[68] = s[1].mParticleSimBuffers;
    o[69] = s[1].mParticleBufferRunsum;
    o[70] = s[1].mParticleBufferSortedUniqueIds;
    o[71] = s[1].mParticleBufferSortedUniqueIdsOriginalIndex;
    o[72] = s[1].mDiffuseSimBuffers;
    o[73] = uint64_t(s[1].mNumDiffuseBuffers);
    o[74] = s[1].mNumDiffuseParticles;
    o[75] = s[1].mDiffusePosition_LifeTime;
    o[76] = s[1].mDiffuseVelocity;
    o[77] = s[1].mDiffusePotentials;
    o[78] = s[1].mDiffuseCellStart;
    o[79] = s[1].mDiffuseCellEnd;
    o[80] = s[1].mDiffuseGridParticleHash;
    o[81] = s[1].mDiffuseOriginPos_LifeTime;
    o[82] = s[1].mDiffuseSortedPos_LifeTime;
    o[83] = s[1].mDiffuseSortedOriginPos_LifeTime;
    o[84] = s[1].mDiffuseSortedVel;
    o[85] = s[1].mDiffuseSortedToUnsortedMapping;
    o[86] = s[1].mDiffuseUnsortedToSortedMapping;
    o[87] = s[1].mDiffuseOneWayContactInfos;
    o[88] = s[1].mDiffuseOneWayForces;
    o[89] = s[1].mDiffuseOneWayNodeIndex;
    o[90] = s[1].mDiffuseOneWayContactCount;
    o[91] = s[1].mDerivedPBDMaterialData;
    o[92] = uint64_t(s[1].mParticleMaterialStride);
    o[93] = s[1].mPhaseGroupToMaterialHandle;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void layoutPxgParticleSimBuffer(uint3 tid : SV_DispatchThreadID)
{
    Ptr<PxgParticleSimBufferGpu> s = Ptr<PxgParticleSimBufferGpu>(gLay.src);
    Ptr<uint64_t> o = Ptr<uint64_t>(gLay.dst);
    o[0] = s[1].mPositionInvMasses;
    o[1] = s[1].mVelocities;
    o[2] = s[1].mRestPositions;
    o[3] = s[1].mPhases;
    o[4] = uint64_t(s[1].mNumActiveParticles);
    o[5] = uint64_t(s[1].mFlags);
    o[6] = uint64_t(s[1].mDiffuseParticleBufferIndex);
    o[7] = uint64_t(s[1].mUniqueId);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void layoutPxgParticleDiffuseSimBuffer(uint3 tid : SV_DispatchThreadID)
{
    Ptr<PxgParticleDiffuseSimBufferGpu> s = Ptr<PxgParticleDiffuseSimBufferGpu>(gLay.src);
    Ptr<uint64_t> o = Ptr<uint64_t>(gLay.dst);
    o[0] = uint64_t(asuint(s[1].mParams_threshold));
    o[1] = uint64_t(asuint(s[1].mParams_lifetime));
    o[2] = uint64_t(asuint(s[1].mParams_airDrag));
    o[3] = uint64_t(asuint(s[1].mParams_bubbleDrag));
    o[4] = uint64_t(asuint(s[1].mParams_buoyancy));
    o[5] = uint64_t(asuint(s[1].mParams_kineticEnergyWeight));
    o[6] = uint64_t(asuint(s[1].mParams_pressureWeight));
    o[7] = uint64_t(asuint(s[1].mParams_divergenceWeight));
    o[8] = uint64_t(asuint(s[1].mParams_collisionDecay));
    o[9] = uint64_t(s[1].mParams_useAccurateVelocity);
    o[10] = s[1].mDiffusePositions_LifeTime;
    o[11] = s[1].mDiffuseVelocities;
    o[12] = uint64_t(s[1].mMaxNumParticles);
    o[13] = s[1].mNumDiffuseParticles;
    o[14] = s[1].mNumActiveDiffuseParticles;
    o[15] = uint64_t(s[1].mStartIndex);
    o[16] = uint64_t(s[1].mFlags);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void layoutPxsDeformableSurfaceMaterialData(uint3 tid : SV_DispatchThreadID)
{
    Ptr<PxsDeformableSurfaceMaterialGpu> s = Ptr<PxsDeformableSurfaceMaterialGpu>(gLay.src);
    Ptr<uint64_t> o = Ptr<uint64_t>(gLay.dst);
    o[0] = uint64_t(asuint(s[1].youngs));
    o[1] = uint64_t(asuint(s[1].poissons));
    o[2] = uint64_t(asuint(s[1].dynamicFriction));
    o[3] = uint64_t(asuint(s[1].thickness));
    o[4] = uint64_t(asuint(s[1].bendingStiffness));
    o[5] = uint64_t(asuint(s[1].elasticityDamping));
    o[6] = uint64_t(asuint(s[1].bendingDamping));
    o[7] = uint64_t(asuint(s[1].padding0));
}

// ---------------------------------------------------------------------------
// refRun
// ---------------------------------------------------------------------------
struct RefQuery
{
    uint op, a, b, c;   // @0
    uint64_t x, y;      // @16
    uint64_t z, w;      // @32 -> 48
}

struct RunParams
{
    uint numTests;      // @0
    uint pad4;          // @4
    uint64_t queries;   // @8
    uint64_t results;   // @16
    uint64_t aux0;      // @24  writer set / filter pairs
    uint64_t aux1;      // @32  filter: safe address
    uint auxCount;      // @40  filter: number of pairs
    uint pad44;         // @44 -> 48
}
[[vk::push_constant]] ConstantBuffer<RunParams> gRun;

[shader("compute")]
[numthreads(64, 1, 1)]
void refRun(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gRun.numTests)
        return;
    const RefQuery q = Ptr<RefQuery>(gRun.queries)[i];
    uint64_t r = uint64_t(0xdeadbeefu);
    const PxgDeformablePairInfoGpu p0 = PxgMakePairInfo(q.x, q.y);
    const PxgDeformablePairInfoGpu p1 = PxgMakePairInfo(q.z, q.w);
    switch (q.op)
    {
    case 0: r = PxgEncodeDeformableRef(q.a, q.b, q.c); break;
    case 1: r = PxgEncodeInvalidDeformableRef(q.a); break;
    case 2: r = PxgEncodeDeformableVertex(q.a, q.b); break;
    case 3: r = PxgEncodeDeformableTriangle(q.a, q.b); break;
    case 4: r = PxgEncodeDeformableTetrahedron(q.a, q.b); break;
    case 5: r = PxgEncodeDeformableTriangleEdge(q.a, q.b, q.c); break;
    case 6: r = PxgGetDeformableId(q.x); break;
    case 7: r = PxgGetDeformableFeatureType(q.x); break;
    case 8: r = PxgGetDeformableFeatureId(q.x); break;
    case 9: r = PxgGetDeformableEdgeTriangle(q.x); break;
    case 10: r = PxgGetDeformableEdgeLocalIndex(q.x); break;
    case 11: r = PxgEncodeParticleRef(q.a, q.b); break;
    case 12: r = PxgGetParticleIndex(q.x); break;
    case 13: r = PxgGetParticleSystemId(q.x); break;
    case 14: r = PxgGetContactSortKey(q.x); break;
    case 15: r = PxgEncodeStaticShapeRef(q.a); break;
    case 16: r = PxgEncodeRigidFilterRef(q.x, q.a); break;
    case 17: r = PxgIsValidPairInfo(p0) ? 1u : 0u; break;
    case 18: r = PxgInvalidatePairInfo(p0).ref1; break;
    case 19: r = PxgIsActivePairInfo(p0) ? 1u : 0u; break;
    case 20: r = PxgSetActiveRef1(q.y, q.a != 0u); break;
    case 21: r = uint64_t(uint(PxgComparePairInfo(p0, p1))); break;
    case 22: r = ownsEdge(q.a, q.b) ? 1u : 0u; break;
    case 23: r = PxgNodeIsStaticBody(q.x) ? 1u : 0u; break;
    case 24: r = PxgNodeIndexOf(q.x); break;
    case 25: r = PxgNodeIsArticulation(q.x) ? 1u : 0u; break;
    case 26: r = PxgNodeArticulationLinkId(q.x); break;
    case 27: r = PxgEncodeRigidNode(q.x); break;
    default: break;
    }
    Ptr<uint64_t>(gRun.results)[i] = r;
}

// ---------------------------------------------------------------------------
// writerRun: one write per query, through the writer structs held in a device
// buffer (gRun.aux0) in the order DbDb, RigidDb, ParticleDb, RigidParticle.
// ---------------------------------------------------------------------------
struct WriterQuery
{
    uint kind, index, matIdx, pad;   // @0
    float4 contact;                  // @16
    float4 normalPen;                // @32
    float4 bary;                     // @48
    uint64_t ref0, ref1;             // @64 -> 80
}

struct WriterSet
{
    PxgDbDbContactWriterGpu dbdb;
    PxgRigidDbContactWriterGpu rigidDb;
    PxgParticleDbContactWriterGpu particleDb;
    PxgRigidParticleContactWriterGpu rigidParticle;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void writerRun(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gRun.numTests)
        return;
    const WriterQuery q = Ptr<WriterQuery>(gRun.queries)[i];
    const WriterSet ws = Ptr<WriterSet>(gRun.aux0)[0];
    bool ok = true;
    switch (q.kind)
    {
    case 0: ok = ws.dbdb.writeVolumeVolumeContact(q.index, q.contact, q.normalPen, q.ref0, q.ref1); break;
    case 1: ok = ws.dbdb.writeSurfaceVolumeContact(q.index, q.contact, q.normalPen, q.ref0, q.bary, q.ref1); break;
    case 2: ok = ws.dbdb.writeSurfaceVolumeContactVertex(q.index, q.contact, q.normalPen, q.ref0, q.ref1); break;
    case 3: ok = ws.rigidDb.writeVolumeContact(q.index, q.contact, q.normalPen, q.ref0, q.matIdx, q.ref1); break;
    case 4: ok = ws.rigidDb.writeSurfaceContact(q.index, q.contact, q.normalPen, q.ref0, q.matIdx, q.ref1, q.bary); break;
    case 5: ok = ws.rigidDb.writeSurfaceContactVertex(q.index, q.contact, q.normalPen, q.ref0, q.matIdx, q.ref1); break;
    case 6: ok = ws.particleDb.writeVolumeContact(q.index, q.contact, q.normalPen, q.ref0, q.ref1); break;
    case 7: ok = ws.particleDb.writeSurfaceContact(q.index, q.contact, q.normalPen, q.ref0, q.ref1, q.bary); break;
    case 8: ws.rigidParticle.writeContact(q.index, q.normalPen, q.ref0, q.ref1); break;
    default: break;
    }
    Ptr<uint64_t>(gRun.results)[i] = ok ? 1u : 0u;
}

// ---------------------------------------------------------------------------
// filterRun
// ---------------------------------------------------------------------------
struct FilterQuery
{
    uint op, a, b, c;   // @0
    uint d, e, f, g;    // @16
    uint64_t x, y;      // @32 -> 48
}

[shader("compute")]
[numthreads(64, 1, 1)]
void filterRun(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gRun.numTests)
        return;
    const FilterQuery q = Ptr<FilterQuery>(gRun.queries)[i];
    // op >= 100: the same search against a zero-length NULL array.
    const bool empty = q.op >= 100u;
    const DfFilterArray arr = empty ? dfFilterArray(uint64_t(0), 0u, gRun.aux1)
                                    : dfFilterArray(gRun.aux0, gRun.auxCount, gRun.aux1);
    bool r = false;
    switch (q.op % 100u)
    {
    case 0: r = dfFind(arr, q.x, q.y); break;
    case 1: r = findVolumeVolumeFilter(arr, q.a, q.b, q.c, q.d); break;
    case 2: r = findRigidVolumeFilter(arr, q.x, q.a, q.b, q.c); break;
    case 3: r = findVolumeSurfaceVertexFilter(arr, q.a, q.b, q.c, q.d); break;
    case 4: r = findVolumeSurfaceTriangleFilter(arr, q.a, q.b, q.c, q.d); break;
    case 5: r = findRigidSurfaceVertexFilter(arr, q.x, q.a, q.b, q.c); break;
    case 6: r = findRigidSurfaceTriangleFilter(arr, q.x, q.a, q.b, q.c); break;
    case 7: r = findSurfaceSurfaceVertexTriangleFilter(arr, q.a, q.b, q.c, q.d); break;
    case 8: r = findSurfaceSurfaceEdgeEdgeFilter(arr, q.a, q.b, q.c, q.d, q.e, q.f); break;
    default: break;
    }
    Ptr<uint64_t>(gRun.results)[i] = r ? 1u : 0u;
}
