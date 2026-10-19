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

// pxgSlangSoftBodyDesc — P8.0: the first BDA mirror of the engine's per-softbody
// (FEM-volume) descriptor (gpusimulationcontroller/include/PxgSoftBody.h:79, class
// PxgSoftBody).  The softbody orchestration kernels receive a device pointer to an
// ARRAY of PxgSoftBody structs plus an active-index array, so the Slang side needs
// a BYTE-EXACT mirror (offsets AND the array stride) to walk the nested device
// pointers — exactly as pxgSlangFEMClothDesc did for the cloth path.
//
// PxgSoftBody is PX_ALIGN_PREFIX(16): a long run of device pointers (0..311), then
// a PxU32/PxReal scalar tail with a PxQuat (mInitialRotation, 16 bytes) and a
// PxU8/PxU8/PxU16 flag triple folded into one word.  sizeof = 464, alignof = 16
// (PhysX 5.12: mFilteringPairs + mNumFilterPairs became three typed element filter
// arrays + three counts, shifting the tail by 24 B; mRestDistance became
// mRestOffset; the struct ends on 8 B of alignment padding, mirrored as _pad456/460).  The layout is verified
// byte-exact against the engine header with g++ offsetof in testSoftBodySim.cpp
// (which pushes a REAL PxgSoftBody filled with sentinels and reads it back through
// this mirror — any compiler/mirror drift fails that test).
//
// Pointers -> uint64_t (device addresses read through Ptr<>); PxU32/PxReal -> the
// 4-byte scalars.  Two byte-packing folds reproduce the C++ layout without
// requiring 8/16-bit storage on llvmpipe:
//   - @416 PxU8 mActorFlags / PxU8 mBodyFlags / PxU16 mVolumeFlags fold into one
//     uint mFlagsWord (byte0 = mActorFlags, byte1 = mBodyFlags, bytes2..3 =
//     mVolumeFlags), plus explicit _pad420 so the following pointer lands at @424;
//   - the two `bool*` arrays (mVertsAreDeformed / mVertsCanNotDeform) are plain
//     8-byte device pointers in the struct; the per-vertex byte STORES the kernel
//     makes into them are elided in softBodySimBda (see that file's header) — the
//     pointer fields are mirrored here only to keep every downstream offset exact.
module pxgSlangSoftBodyDesc;

// PxActorFlag::eDISABLE_GRAVITY (PxActor.h:73) = (1<<1).  Tested against the low
// byte of mFlagsWord (mActorFlags) in the pre-integrate gravity predict.
public static const uint PXG_SOFTBODY_ACTORFLAG_DISABLE_GRAVITY = 2u;
// PxDeformableBodyFlag (PxDeformableBodyFlag.h:47-51): eENABLE_SPECULATIVE_CCD =
// (1<<1), eKINEMATIC = (1<<2).  Tested against byte1 of mFlagsWord (mBodyFlags).
public static const uint PXG_SOFTBODY_BODYFLAG_DISABLE_SELF_COLLISION = 1u;
public static const uint PXG_SOFTBODY_BODYFLAG_ENABLE_SPECULATIVE_CCD = 2u;
public static const uint PXG_SOFTBODY_BODYFLAG_KINEMATIC = 4u;
// PxDeformableVolumeFlag::ePARTIALLY_KINEMATIC (PxDeformableVolumeFlag.h:48) =
// (1<<1).  Tested against bytes2..3 of mFlagsWord (mVolumeFlags).
// PxDeformableVolumeFlag::eCOMPUTE_STRESS_TENSOR (PxDeformableVolumeFlag.h:47) = (1<<0).
public static const uint PXG_SOFTBODY_VOLUMEFLAG_COMPUTE_STRESS_TENSOR = 1u;
public static const uint PXG_SOFTBODY_VOLUMEFLAG_PARTIALLY_KINEMATIC = 2u;

// sizeof(PxgSoftBody), 5.12: the array stride of the softbodies argument.
public static const uint64_t PXG_SOFTBODY_STRIDE = 464;

// Byte-exact mirror of class PxgSoftBody (PxgSoftBody.h, 5.12).  Every pointer is 8
// bytes; the block is a run of device pointers (0..327), then the scalar tail
// (328..463).  sizeof = 464.  Every field the engine header carries is mirrored so
// the shader "reads like a sibling" of it — the ones sb_gm_preIntegrateLaunch
// actually dereferences are flagged (***).
public struct PxgSoftBodyGpu
{
    public uint64_t mTetMeshData;                    // @0    void*
    public uint64_t mTetMeshSurfaceHint;             // @8    PxU8*
    public uint64_t mTetIndices;                     // @16   uint4*
    public uint64_t mTetIndicesRemapTable;           // @24   PxU32*
    public uint64_t mTetraStresses;                  // @32   PxMat33*
    public uint64_t mTetraRestPoses;                 // @40   PxMat33*
    public uint64_t mTetraStressCoefficient;         // @48   PxReal*
    public uint64_t mTetraRotations;                 // @56   float4*
    public uint64_t mPosition_InvMass;               // @64   float4* (collision mesh)
    public uint64_t mRestPosition;                   // @72   float4*
    public uint64_t mSimTetIndices;                  // @80   uint4*
    public uint64_t mSimVelocity_InvMass;            // @88   float4* ***
    public uint64_t mSimPosition_InvMass;            // @96   float4* *** (the observable)
    public uint64_t mSimKinematicTarget;             // @104  const float4* ***
    public uint64_t mVertsAreDeformed;               // @112  bool* *** (byte stores elided)
    public uint64_t mVertsCanNotDeform;              // @120  bool* *** (byte stores elided)
    public uint64_t mSimDeltaPos;                    // @128  float4* *** (posDelta)
    public uint64_t mSimTetraRestPoses;              // @136  PxgMat33Block*
    public uint64_t mOrigQinv;                       // @144  PxgMat33Block*
    public uint64_t mSimOrderedTetrahedrons;         // @152  PxU32*
    public uint64_t mSimJacobiVertIndices;           // @160  PxU32*
    public uint64_t mSimTetraRotations;              // @168  float4*
    public uint64_t mSimVelocity_invMassCP;          // @176  float4*
    public uint64_t mSimPosition_InvMassCP;          // @184  float4*
    public uint64_t mSimRemapOutputCP;               // @192  uint4*
    public uint64_t mSimAccumulatedCopiesCP;         // @200  PxU32*
    public uint64_t mSimAccumulatedPartitionsCP;     // @208  PxU32*
    public uint64_t mVertsBarycentricInGridModel;    // @216  float4*
    public uint64_t mVertsRemapInGridModel;          // @224  PxU32*
    public uint64_t mTetsRemapColToSim;              // @232  PxU32*
    public uint64_t mTetsAccumulatedRemapColToSim;   // @240  PxU32*
    public uint64_t mSurfaceVertsHint;               // @248  PxU8*
    public uint64_t mSurfaceVertToTetRemap;          // @256  PxU32*
    public uint64_t mSimTetraMultipliers;            // @264  PxgSpatialVectorBlock*
    public uint64_t mSimDelta;                       // @272  float4*
    public uint64_t mPackedNodeBounds;               // @280  PxBounds3*
    public uint64_t mOrderedMaterialIndices;         // @288  PxU16*
    public uint64_t mMaterialIndices;                // @296  PxU16*
    public uint64_t mRigidVolumeFilterPairs;         // @304  PxgDeformablePairInfo* (rigid, tet)
    public uint64_t mVolumeVolumeFilterPairs;        // @312  PxgDeformablePairInfo* (vertex, tet)
    public uint64_t mVolumeSurfaceFilterPairs;       // @320  PxgDeformablePairInfo* (tet, vertex|triangle)
    public uint mNumRigidVolumeFilterPairs;          // @328  PxU32
    public uint mNumVolumeVolumeFilterPairs;         // @332  PxU32
    public uint mNumVolumeSurfaceFilterPairs;        // @336  PxU32
    public float mLinearDamping;                     // @340  *** PxReal
    public float mMaxLinearVelocity;                 // @344  *** PxReal
    public float mPenBiasClamp;                      // @348  PxReal (not read by pre-integrate)
    public float mSettlingThreshold;                 // @352  PxReal
    public float mSleepThreshold;                    // @356  PxReal
    public float mSettlingDamping;                   // @360  PxReal
    public float mSelfCollisionFilterDistance;       // @364  PxReal
    public float mSelfCollisionStressTolerance;      // @368  PxReal
    public float mInitialRotationX;                  // @372  PxQuat.x
    public float mInitialRotationY;                  // @376  PxQuat.y
    public float mInitialRotationZ;                  // @380  PxQuat.z
    public float mInitialRotationW;                  // @384  PxQuat.w
    public uint mNumVerts;                           // @388  PxU32
    public uint mNumTets;                            // @392  PxU32
    public uint mNumVertsGM;                         // @396  *** PxU32
    public uint mNumTetsGM;                          // @400  PxU32
    public uint mNumPartitionsGM;                    // @404  PxU32
    public uint mElementIndex;                       // @408  PxU32
    public uint mGpuRemapIndex;                      // @412  PxU32
    public uint mFlagsWord;                          // @416  PxU8 mActorFlags|PxU8 mBodyFlags|PxU16 mVolumeFlags ***
    public uint _pad420;                             // @420  (mVolumeFlags's tail-pad -> next ptr @424)
    public uint64_t mSimPullIndices;                 // @424  uint4*
    public uint mAwake;                              // @432  *** PxU32 (reset each frame)
    public uint mNumTetsPerElement;                  // @436  PxU32
    public uint mNumJacobiVertices;                  // @440  PxU32
    public float mRestOffset;                        // @444  PxReal (5.6.1 mRestDistance; now the rest offset)
    public float mOriginalContactOffset;             // @448  PxReal
    public float mJacobiScale;                       // @452  PxReal
    public uint _pad456;                             // @456  (align-16 tail pad)
    public uint _pad460;                             // @460  -> 464
}

// &softbodies[id]: the descriptor of softbody `id` in the array at `softbodies`.
public Ptr<PxgSoftBodyGpu> sbAt(uint64_t softbodies, uint id)
{
    return Ptr<PxgSoftBodyGpu>(softbodies + uint64_t(id) * PXG_SOFTBODY_STRIDE);
}
