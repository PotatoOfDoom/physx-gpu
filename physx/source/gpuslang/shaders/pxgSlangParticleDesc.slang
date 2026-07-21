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

// pxgSlangParticleDesc — P6.0: the FIRST-ever BDA mirror of the engine's
// per-system particle descriptor (gpusimulationcontroller/include/
// PxgParticleSystem.h).  The PBD particle kernels receive a device pointer to an
// ARRAY of PxgParticleSystem structs plus an active-index array, so the Slang
// side needs a BYTE-EXACT mirror (offsets AND the array stride) to walk the
// nested device pointers, exactly as PxgBodySim / PxgSolverCoreDesc did for the
// rigid path.
//
// PxgParticleSystem : public PxGpuParticleSystem (base [0..136)) + derived
// members [136..656).  The base embeds PxGpuParticleData mCommonData at @88; the
// derived class embeds PxgParticleSystemData mData at @248.  Both nested value
// structs are flattened here (mCommonData_/mData_ prefixes).  sizeof = 656,
// alignof = 16 (PX_ALIGN_PREFIX(16)); 656 is a multiple of 16, so the mirror's
// own 8-byte size rounding lands on the same stride.  The layout was verified
// against the engine header with g++ offsetof (see testParticleSim.cpp, which
// pushes a REAL PxgParticleSystem filled with sentinels and reads it back
// through this mirror — any compiler/mirror drift fails that test).
//
// Pointers -> uint64_t (device addresses read through Ptr<>); PxU32/PxReal/int
// -> 4-byte scalars; PxU16 mData.mLockFlags folds into mData_lockFlagsWord (its
// 2 tail-pad bytes are uninitialized host memory).  Explicit _padNNN members
// reproduce the C++ inter-member padding at 322/340/484/636 and the 8-byte tail.
module pxgSlangParticleDesc;

// PxParticleSystemFlag.h:89 — group is bits [0,19] of the phase.
public static const uint PXG_PARTICLE_PHASE_GROUP_MASK = 0x000fffffu;
// PxPBDParticleSystem.h:200-202
public static const uint PXG_PARTICLE_LOCK_X = 1u;
public static const uint PXG_PARTICLE_LOCK_Y = 2u;
public static const uint PXG_PARTICLE_LOCK_Z = 4u;

// PxsParticleMaterialData (PxParticleGpu.h:47-54) — 5 PxReal.  The engine
// dereferences it with a RUNTIME stride (sizeof(PxsPBDMaterialData), a derived
// type), so the kernels index it by raw byte offset rather than through this
// struct; it is kept for documentation and the layout test.
//   friction @0, damping @4, adhesion @8, gravityScale @12, adhesionRadiusScale @16
public static const uint PXG_PARTICLE_MAT_DAMPING_OFFSET_FLOATS = 1u;      // byte 4
public static const uint PXG_PARTICLE_MAT_GRAVITYSCALE_OFFSET_FLOATS = 3u; // byte 12

// Byte-exact mirror of PxgParticleSimBuffer (PxgParticleSystem.h:138-156).
// 7 device pointers (56) + 7 PxU32 (28) = 84, rounded to the pointer alignment
// -> 88.  The array stride the ps_updateUserBufferLaunch kernel indexes.
public struct PxgParticleSimBufferGpu
{
    public uint64_t mPositionInvMasses;   // @0
    public uint64_t mVelocities;          // @8
    public uint64_t mRestPositions;       // @16
    public uint64_t mPhases;              // @24
    public uint64_t mVolumes;             // @32
    public uint64_t mFilterPairs;         // @40
    public uint64_t mRigidAttachments;    // @48
    public uint mNumActiveParticles;      // @56
    public uint mNumVolumes;              // @60
    public uint mNumFilterPairs;          // @64
    public uint mNumRigidAttachments;     // @68
    public uint mFlags;                   // @72
    public uint mDiffuseParticleBufferIndex; // @76
    public uint mUniqueId;                // @80
    public uint _pad84;                   // @84 -> 88
}

// Byte-exact mirror of PxgParticleSystem (PxgParticleSystem.h:405-488) with the
// base PxGpuParticleSystem members inlined first (offsets 0..135) then the
// derived members (136..655).  sizeof = 656.
public struct PxgParticleSystemGpu
{
    // ---- base PxGpuParticleSystem [0..136) ----
    public uint64_t mUnsortedPositions_InvMass;   // @0   float4*
    public uint64_t mUnsortedVelocities;          // @8   float4*
    public uint64_t mUnsortedPhaseArray;          // @16  PxU32*
    public uint64_t mSortedPositions_InvMass;     // @24  float4*
    public uint64_t mSortedVelocities;            // @32  float4*
    public uint64_t mSortedPhaseArray;            // @40  PxU32*
    public uint64_t mUnsortedToSortedMapping;     // @48  PxU32*
    public uint64_t mSortedToUnsortedMapping;     // @56  PxU32*
    public uint64_t mParticleSelfCollisionCount;  // @64  PxU32*
    public uint64_t mCollisionIndex;              // @72  PxU32*
    public uint64_t mParticleMaterials;           // @80  PxsParticleMaterialData*
    // ---- mCommonData (PxGpuParticleData, 48 bytes) @88 ----
    public float mCommonData_GridCellWidth;               // @88
    public uint  mCommonData_GridSizeX;                   // @92
    public uint  mCommonData_GridSizeY;                   // @96
    public uint  mCommonData_GridSizeZ;                   // @100
    public float mCommonData_ParticleContactDistance;     // @104
    public float mCommonData_ParticleContactDistanceInv;  // @108
    public float mCommonData_ParticleContactDistanceSq;   // @112
    public uint  mCommonData_NumParticles;                // @116
    public uint  mCommonData_MaxParticles;                // @120
    public uint  mCommonData_MaxNeighborhood;             // @124
    public uint  mCommonData_MaxDiffuseParticles;         // @128
    public uint  mCommonData_NumParticleBuffers;          // @132
    // ---- derived PxgParticleSystem members ----
    public uint64_t mOriginPos_InvMass;           // @136 float4*
    public uint64_t mGridParticleHash;            // @144 PxU32*
    public uint64_t mAccumDeltaP;                 // @152 float4*
    public uint64_t mSortedDeltaP;                // @160 float4*
    public uint64_t mCellStart;                   // @168 PxU32*
    public uint64_t mCellEnd;                     // @176 PxU32*
    public uint64_t mCollisionHeaders;            // @184
    public uint64_t mDensityCollisionImpulses;    // @192 float2*
    public uint64_t mOneWayContactInfos;          // @200
    public uint64_t mOneWayForces;                // @208 float2*
    public uint64_t mOneWayNodeIndex;             // @216
    public uint64_t mOneWaySurfaceVelocity;       // @224 float4*
    public uint64_t mOneWayContactCount;          // @232 PxU32*
    public uint64_t mRestArray;                   // @240 float4*
    // ---- mData (PxgParticleSystemData, 92 bytes) @248 ----
    public float mData_BoundCenterX;              // @248
    public float mData_BoundCenterY;              // @252
    public float mData_BoundCenterZ;              // @256
    public uint  mData_ElementIndex;              // @260
    public uint  mData_RemapIndex;                // @264
    public float mData_RestOffset;                // @268
    public float mData_SolidRestOffset;           // @272
    public float mData_Spiky1;                    // @276
    public float mData_Spiky2;                    // @280
    public float mData_RestDensity;               // @284
    public float mData_RestDensityBoundary;       // @288
    public float mData_FluidSurfaceConstraintScale; // @292
    public float mData_LambdaScale;               // @296
    public float mData_RelaxationFactor;          // @300
    public float mData_FluidRestOffset;           // @304
    public float mData_InvRestDensity;            // @308
    public uint  mData_Flags;                     // @312
    public float mData_MaxVelocity;               // @316
    public uint  mData_LockFlagsWord;             // @320 (low 16 bits = PxU16 mLockFlags; hi 16 = pad @322)
    public uint  mData_NumPhaseToMaterials;       // @324
    public float mData_WindX;                     // @328
    public float mData_WindY;                     // @332
    public float mData_WindZ;                     // @336
    public uint  _pad340;                         // @340 (mData ends at 340; next member 8-aligned)
    public uint64_t mDensity;                     // @344 PxReal*
    public uint64_t mStaticDensity;               // @352 PxReal*
    public uint64_t mSurfaceNormal;               // @360 float4*
    public uint64_t mDelta;                       // @368 float4*
    public uint64_t mCurl;                        // @376 float4*
    public uint64_t mNormalArray;                 // @384 float4*
    public uint64_t mSortedOriginPos_InvMass;     // @392 float4*
    public uint64_t mParticleSimBuffers;          // @400 PxgParticleSimBuffer*
    public uint64_t mParticleBufferRunsum;        // @408 PxU32*
    public uint64_t mParticleBufferSortedUniqueIds;              // @416
    public uint64_t mParticleBufferSortedUniqueIdsOriginalIndex; // @424
    public uint64_t mClothSimBuffers;             // @432
    public uint64_t mRigidSimBuffers;             // @440
    public uint64_t mDiffuseSimBuffers;           // @448
    public uint64_t mAttachmentRunsum;            // @456
    public uint  mNumClothBuffers;                // @464
    public uint  mNumRigidBuffers;                // @468
    public uint  mNumDiffuseBuffers;              // @472
    public uint  mNumRigidAttachments;            // @476
    public uint  mRigidAttachmentOffset;          // @480
    public uint  _pad484;                         // @484 (next member 8-aligned)
    public uint64_t mNumDiffuseParticles;         // @488 int*
    public uint64_t mDiffusePosition_LifeTime;    // @496
    public uint64_t mDiffuseVelocity;             // @504
    public uint64_t mDiffusePotentials;           // @512
    public uint64_t mDiffuseCellStart;            // @520
    public uint64_t mDiffuseCellEnd;              // @528
    public uint64_t mDiffuseGridParticleHash;     // @536
    public uint64_t mDiffuseOriginPos_LifeTime;   // @544
    public uint64_t mDiffuseSortedPos_LifeTime;   // @552
    public uint64_t mDiffuseSortedOriginPos_LifeTime; // @560
    public uint64_t mDiffuseSortedVel;            // @568
    public uint64_t mDiffuseSortedToUnsortedMapping;  // @576
    public uint64_t mDiffuseUnsortedToSortedMapping;  // @584
    public uint64_t mDiffuseOneWayContactInfos;   // @592
    public uint64_t mDiffuseOneWayForces;         // @600
    public uint64_t mDiffuseOneWayNodeIndex;      // @608
    public uint64_t mDiffuseOneWayContactCount;   // @616
    public uint64_t mDerivedPBDMaterialData;      // @624
    public uint  mParticleMaterialStride;         // @632
    public uint  _pad636;                         // @636 (next member 8-aligned)
    public uint64_t mPhaseGroupToMaterialHandle;  // @640 PxU16*
    public uint64_t _pad648;                      // @648 tail pad to align-16 -> 656
}
