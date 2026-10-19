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

// pxgSlangFEMClothDesc — P7.0: the first BDA mirror of the engine's per-cloth
// descriptor (gpusimulationcontroller/include/PxgFEMCloth.h:69, class PxgFEMCloth).
// The FEMCloth orchestration kernels receive a device pointer to an ARRAY of
// PxgFEMCloth structs plus an active-index array, so the Slang side needs a
// BYTE-EXACT mirror (offsets AND the array stride) to walk the nested device
// pointers, exactly as pxgSlangParticleDesc did for the particle path.
//
// PxgFEMCloth is a flat pointer+scalar block (PX_ALIGN_PREFIX(16)), NO nested
// value structs — simpler than the particle mirror.  sizeof = 416, alignof = 16
// (PhysX 5.12: mRigidConstraints became the two element filter arrays + counts,
// +16 B; mTriangleVertexIndices became mTriangleVertexIndices_ownedEdges, whose .w
// holds the 3-bit edge ownership; mRestDistance became mRestOffset).  The layout is verified
// byte-exact against the engine header with g++ offsetof in testFemClothSim.cpp
// (which pushes a REAL PxgFEMCloth filled with sentinels and reads it back through
// this mirror — any compiler/mirror drift fails that test).
//
// Pointers -> uint64_t (device addresses read through Ptr<>); PxU32/PxReal/int ->
// 4-byte scalars.  Two byte-packing folds reproduce the C++ layout without
// requiring 8/16-bit storage on llvmpipe:
//   - @160 bool mNonSharedTriPair_hasActiveBending (1 byte + 7 pad) -> a uint word
//     @160 + explicit _pad164 so the next pointer lands at @168;
//   - @392 PxU8 mActorFlags / PxU8 mBodyFlags / PxU16 mSurfaceFlags fold into one
//     uint mFlagsWord (low byte = mActorFlags; test mActorFlags & eDISABLE_GRAVITY).
module pxgSlangFEMClothDesc;

// PxActorFlag::eDISABLE_GRAVITY (PxActor.h:73) = (1<<1).  Tested against the low
// byte of mFlagsWord (mActorFlags) in the pre-integrate gravity predict.
public static const uint PXG_FEMCLOTH_ACTORFLAG_DISABLE_GRAVITY = 2u;
// sizeof(PxgFEMCloth), 5.12: the array stride of the femClothes argument.
public static const uint64_t PXG_FEMCLOTH_STRIDE = 416;

// P7.1: byte-exact mirror of PxsDeformableSurfaceMaterialData
// (PxsDeformableSurfaceMaterialCore.h:38, PX_ALIGN(16), sizeof == 32).  The
// per-triangle in-plane energy solve reads this array through a device pointer
// (mMaterialIndices[triIndex] -> clothMaterials[globalMaterialIndex]); the eight
// PxReal fields lay out at their natural f32 offsets and 32 is already a multiple
// of 16, so the Slang array stride (32) equals the C++ one.  Offsets are pinned
// with g++ offsetof static_asserts in testFemClothSim.cpp.
public struct PxsDeformableSurfaceMaterialGpu
{
    public float youngs;             // @0
    public float poissons;           // @4
    public float dynamicFriction;    // @8
    public float thickness;          // @12
    public float bendingStiffness;   // @16
    public float elasticityDamping;  // @20
    public float bendingDamping;     // @24
    public float padding0;           // @28  -> 32
}

// Byte-exact mirror of class PxgFEMCloth (PxgFEMCloth.h, 5.12).  Every pointer is
// 8 bytes; the block is a run of device pointers (0..295), then the PxReal/PxU32
// scalar tail (296..415).  sizeof = 416.
public struct PxgFEMClothGpu
{
    public uint64_t mTriMeshData;                                          // @0    void*
    public uint64_t mVelocity_InvMass;                                     // @8    float4*
    public uint64_t mPosition_InvMass;                                     // @16   float4* (the observable)
    public uint64_t mRestPosition;                                         // @24   float4*
    public uint64_t mPrevPositionInContactOffset;                          // @32   float4*
    public uint64_t mPrevPositionInRestOffset;                             // @40   float4*
    public uint64_t mDynamicFrictions;                                     // @48   float*
    public uint64_t mMaterialIndices;                                      // @56   PxU16*
    public uint64_t mTrianglesWithOwnedEdges;                              // @64   PxU32*
    public uint64_t mTriangleVertexIndices_ownedEdges;                     // @72   uint4* (.w = 3-bit edge ownership)
    public uint64_t mOrderedNonSharedTriangleVertexIndices_triIndex;       // @80   uint4*
    public uint64_t mOrderedSharedTriangleLambdas;                         // @88   float2*
    public uint64_t mOrderedNonSharedTriangleLambdas;                      // @96   float2*
    public uint64_t mOrderedNonSharedTriangleRestPoseInv;                  // @104  float4*
    public uint64_t mOrderedSharedTrianglePairVertexIndices;              // @112  uint4*
    public uint64_t mOrderedNonSharedTrianglePairVertexIndices;          // @120  uint4*
    public uint64_t mOrderedSharedRestBendingAngle_flexuralStiffness_damping;    // @128 float4*
    public uint64_t mOrderedNonSharedRestBendingAngle_flexuralStiffness_damping; // @136 float4*
    public uint64_t mSharedBendingLambdas;                                 // @144  float*
    public uint64_t mNonSharedBendingLambdas;                             // @152  float*
    public uint mNonSharedTriPair_hasActiveBendingWord;                    // @160  bool (low byte) + 3 pad
    public uint _pad164;                                                   // @164  (7 tail-pad bytes of the bool -> next ptr @168)
    public uint64_t mOrderedSharedRestEdge0_edge1;                         // @168  float4*
    public uint64_t mOrderedSharedRestEdgeLength_material0_material1;      // @176  float4*
    public uint64_t mPosition_InvMassCP;                                   // @184  float4*
    public uint64_t mNonSharedTriAccumulatedPartitionsCP;                  // @192  PxU32*
    public uint64_t mSharedTriPairRemapOutputCP;                          // @200  PxU32*
    public uint64_t mNonSharedTriPairRemapOutputCP;                       // @208  PxU32*
    public uint64_t mSharedTriPairAccumulatedCopiesCP;                    // @216  PxU32*
    public uint64_t mNonSharedTriPairAccumulatedCopiesCP;                 // @224  PxU32*
    public uint64_t mSharedTriPairAccumulatedPartitionsCP;               // @232  PxU32*
    public uint64_t mNonSharedTriPairAccumulatedPartitionsCP;            // @240  PxU32*
    public uint64_t mDeltaPos;                                             // @248  float4*
    public uint64_t mAccumulatedDeltaPos;                                  // @256  float4* (posDelta)
    public uint64_t mAccumulatedDeltaVel;                                  // @264  float4*
    public uint64_t mPackedNodeBounds;                                     // @272  PxBounds3*
    public uint64_t mRigidSurfaceFilterPairs;                              // @280  PxgDeformablePairInfo*
    public uint64_t mSurfaceSurfaceFilterPairs;                            // @288  PxgDeformablePairInfo*
    public uint mNumRigidSurfaceFilterPairs;                               // @296
    public uint mNumSurfaceSurfaceFilterPairs;                             // @300
    public float mLinearDamping;                                           // @304
    public float mMaxLinearVelocity;                                       // @308
    public float mPenBiasClamp;                                            // @312
    public float mSettlingThreshold;                                       // @316
    public float mSleepThreshold;                                          // @320
    public float mSettlingDamping;                                         // @324
    public float mSelfCollisionFilterDistance;                             // @328
    public uint mNbVerts;                                                  // @332
    public uint mNbTriangles;                                              // @336
    public uint mNbNonSharedTriangles;                                     // @340
    public uint mNbTrianglesWithOwnedEdges;                                // @344
    public uint mNbTrianglePairs;                                          // @348
    public uint mNbSharedTrianglePairs;                                    // @352
    public uint mNbNonSharedTrianglePairs;                                 // @356
    public uint mNbNonSharedTriPartitions;                                 // @360
    public uint mNbSharedTriPairPartitions;                                // @364
    public uint mNbNonSharedTriPairPartitions;                             // @368
    public uint mNonSharedTriClusterId;                                    // @372
    public uint mSharedTriPairClusterId;                                   // @376
    public uint mNonSharedTriPairClusterId;                                // @380
    public uint mElementIndex;                                             // @384
    public uint mGpuRemapIndex;                                            // @388
    public uint mFlagsWord;                                                // @392  PxU8 mActorFlags|PxU8 mBodyFlags|PxU16 mSurfaceFlags
    public uint mIsActive;                                                 // @396  PxU32 (atomicOr target)
    public float mRestOffset;                                              // @400  (5.6.1 mRestDistance; now the rest offset)
    public float mOriginalContactOffset;                                   // @404
    public uint mNbCollisionPairUpdatesPerTimestep;                        // @408
    public uint mNbCollisionSubsteps;                                      // @412  -> 416
}

// &clothes[id]: the descriptor of cloth `id` in the array at `clothes`.
public Ptr<PxgFEMClothGpu> clothAt(uint64_t clothes, uint id)
{
    return Ptr<PxgFEMClothGpu>(clothes + uint64_t(id) * PXG_FEMCLOTH_STRIDE);
}
