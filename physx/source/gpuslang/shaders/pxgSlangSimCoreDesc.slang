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

// pxgSlangSimCoreDesc — P5 Step 4: BDA mirrors of the engine's simulation-
// controller descriptor aggregates (gpusimulationcontroller/include/
// PxgSimulationCoreDesc.h) and the element structs their kernels index
// through embedded device pointers.
//
// The converted kernels receive ONE device pointer each:
//   - updateTransformCacheAndBoundArrayLaunch(const PxgSimulationCoreDesc*)
//   - updateBodiesLaunch(const PxgNewBodiesDesc*)
// (updateShapesLaunch takes raw element pointers, no desc), and walk nested
// device pointers out of the descs, so the Slang side needs byte-exact
// mirrors: pointers -> uint64_t, PxU32 -> 4-byte scalars.
//
// INDEXED pointers make each element STRIDE part of the ABI.  Cheap structs
// are mirrored fully below (PxBounds3Gpu, PxgShapeGpu, PxgNewShapeSimGpu) or
// reused from pxgSlangCommon (PxgShapeSim 72 B, PxsCachedTransform 32 B,
// PxgSolverBodySleepData 8 B, PxgBodySim 224 B — all scalar-packed, so their
// physical-storage layout equals the C++ layout).  PxgArticulation (496 B,
// ~40 device pointers) is NOT mirrored: the articulation branch reads exactly
// two pointer fields, addressed through the raw stride/offset constants at
// the bottom of this file.
//
// The layout is verified EMPIRICALLY, not by inspection:
// tests/testSimCoreDescLayout.cpp fills the REAL engine structs with
// sentinels, uploads the raw bytes, and the simCoreDescProbe kernel reads
// every mirror field and every raw stride/offset constant back for a bitwise
// host compare (the constants against host-side sizeof/offsetof).  Any drift
// between compiler and mirror fails that test.
module pxgSlangSimCoreDesc;

import pxgSlangCommon;

// mirrors PxgSimulationCoreDesc (152 bytes; full struct — it is only ever
// dereferenced as desc[0], but the mirror is cheap so nothing is truncated)
public struct PxgSimulationCoreDescGpu
{
    public uint64_t mChangedAABBMgrHandles;     // @0   PxU32*
    public uint64_t mFrozen;                    // @8   PxU32*
    public uint64_t mUnfrozen;                  // @16  PxU32*
    public uint64_t mFrozenBlockAndRes;         // @24  PxU32*
    public uint64_t mUnfrozenBlockAndRes;       // @32  PxU32*
    public uint64_t mUpdated;                   // @40  PxU32*
    public uint64_t mActivate;                  // @48  PxU32*
    public uint64_t mDeactivate;                // @56  PxU32*
    public uint64_t mBodySimBufferDeviceData;   // @64  PxgBodySim*
    public uint64_t mShapeSimsBufferDeviceData; // @72  const PxgShapeSim*
    public uint64_t mArticulationPool;          // @80  PxgArticulation*
    public uint64_t mArticulationSleepDataPool; // @88  PxgSolverBodySleepData*
    public uint64_t mTransformCache;            // @96  PxsCachedTransform*
    public uint64_t mBounds;                    // @104 PxBounds3*
    public uint64_t mBodyDataIndices;           // @112 PxU32*
    public uint64_t mSleepData;                 // @120 PxgSolverBodySleepData*
    public uint64_t mShapes;                    // @128 PxgShape*
    public uint mNbTotalShapes;                 // @136
    public uint mBitMapWordCounts;              // @140
    public uint mTotalFrozenShapes;             // @144
    public uint mTotalUnfrozenShapes;           // @148
}                                               // = 152

// mirrors PxgUpdatedBodiesDesc (PxgSimulationCoreDesc.h:316, 24 bytes incl. 4
// tail-padding bytes; P9.93, for updateBodyExternalVelocitiesLaunch).  NOTE the
// pointer order is the REVERSE of PxgNewBodiesDesc: the pool comes FIRST here.
// Pinned by static_asserts in tests/testBodyUpdate.cpp.
public struct PxgUpdatedBodiesDescGpu
{
    public uint64_t mBodySimBufferDeviceData; // @0  PxgBodySim*
    public uint64_t mUpdatedBodySim;          // @8  const PxgBodySimVelocityUpdate*
    public uint mNbUpdatedBodies;             // @16 (+4 bytes pad -> 24)
}

// mirrors PxgNewBodiesDesc (32 bytes incl. 4 tail-padding bytes; only ever
// dereferenced as desc[0]).  5.12 inserted mPrevVelocitiesBuffer at @16, which
// moved mNbNewBodies from @16 to @24 (5.6.1: 24 bytes).
public struct PxgNewBodiesDescGpu
{
    public uint64_t mNewBodySim;              // @0  const PxgBodySim*
    public uint64_t mBodySimBufferDeviceData; // @8  PxgBodySim*
    public uint64_t mPrevVelocitiesBuffer;    // @16 PxgBodySimVelocities* (may be null)
    public uint mNbNewBodies;                 // @24 (+4 bytes pad -> 32)
}

// mirrors PxgRigidBodyAcceleration (PxgSimulationCore.h, new in 5.12, 32 bytes):
// written by computeRigidBodyAccelerations, gathered by the direct-GPU-API
// get{Linear,Angular}Acceleration kernels.
public struct PxgRigidBodyAccelerationGpu
{
    public float3 linear;   // @0  PxVec3
    public float pad0;      // @12
    public float3 angular;  // @16 PxVec3
    public float pad1;      // @28
}

// mirrors PxgUpdateActorDataDesc (80 bytes incl. 4 bytes of padding after
// mBitMapWordCounts; only ever dereferenced as desc[0]).  This is the DIRECT-GPU
// API's twin of PxgSimulationCoreDesc: the same pipeline stage, a different and
// much smaller descriptor, uploaded from PxgSimulationCore::gpuDmaUpdateData().
//
// P9.79 added it for mergeChangedAABBMgrHandlesLaunch, which is launched from
// ScPipeline.cpp:355 under `if (isDirectGPUAPIInitialized())` and nowhere else.
// Its offsets are pinned against the real struct by static_asserts in
// tests/testDirectApiBdaWrite.cpp (every field and the size, P9.93) -- NOT by
// testSimCoreDescLayout, as this comment said until P9.93.
public struct PxgUpdateActorDataDescGpu
{
    public uint64_t mBodySimBufferDeviceData;   // @0  PxgBodySim*
    public uint64_t mShapes;                    // @8  PxgShape*
    public uint64_t mTransformCache;            // @16 PxsCachedTransform*
    public uint64_t mBounds;                    // @24 PxBounds3*
    public uint64_t mRigidNodeIndices;          // @32 PxNodeIndex*
    public uint64_t mShapeIndices;              // @40 PxU32*
    public uint64_t mUpdated;                   // @48 PxU32*  (direct-API changed handles)
    public uint64_t mChangedAABBMgrHandles;     // @56 PxU32*  (CPU-API changed handles)
    public uint mBitMapWordCounts;              // @64 (+4 bytes pad)
    public uint _pad0;                          // @68
    public uint64_t mShapeSimsBufferDeviceData; // @72 const PxgShapeSim*
}                                               // = 80

// mirrors PxBounds3 (24 bytes, two tightly packed PxVec3)
public struct PxBounds3Gpu
{
    public float minX, minY, minZ; // @0  minimum
    public float maxX, maxY, maxZ; // @12 maximum
}                                  // = 24 (4-aligned; the array stride)

// mirrors PxgShape (gpunarrowphase/include/PxgConvexConvexShape.h, 48 bytes)
// with hullOrMeshPtr as what it is under the engine ABI: a raw 8-byte device
// address of the cooked hull blob (the bound-buffer ports repurposed it as a
// buffer offset; the BDA path dereferences it directly like the CUDA does).
public struct PxgShapeGpu
{
    public float scaleX, scaleY, scaleZ;                     // @0  PxMeshScale::scale
    public float scaleRotX, scaleRotY, scaleRotZ, scaleRotW; // @12 PxMeshScale::rotation
    public uint materialIndex;                               // @28
    public uint64_t hullOrMeshPtr;                           // @32 size_t (device address)
    public uint type;                                        // @40
    public uint particleOrSoftbodyId;                        // @44
}                                                            // = 48 (8-aligned stride)

// mirrors PxgNewShapeSim (PxgShapeSim + PxU32 mElementIndex, 80 bytes; the
// PxU64 PxNodeIndex member gives the struct 8-byte alignment, hence the tail
// pad).  The leading 72 bytes reuse the proven PxgShapeSim mirror from
// pxgSlangCommon.
public struct PxgNewShapeSimGpu
{
    public PxgShapeSim base;   // @0  (72 bytes)
    public uint mElementIndex; // @72
    public uint pad1;          // @76 (tail padding to the 8-byte alignment)
}                              // = 80 (the array stride)

// ---------------------------------------------------------------------------
// PxgBodySim uint4 addressing (updateBodiesLaunch)
// ---------------------------------------------------------------------------
// The CUDA kernel derives these from sizeof/offsetof at compile time
// (updateBodiesAndShapes.cu):
//   PXG_BODY_SIM_UINT4_SIZE        = sizeof(PxgBodySim) / sizeof(uint4)
//   PXG_BODY_SIM_BODYSIM_INDEX_IND = offsetof(PxgBodySim,
//       freezeThresholdX_wakeCounterY_sleepThresholdZ_bodySimIndex) / sizeof(uint4)
// The layout test checks both against host-side sizeof/offsetof of the real
// engine struct.
public static const uint PXG_BODY_SIM_UINT4_SIZE = 14;        // 224 / 16
public static const uint PXG_BODY_SIM_BODYSIM_INDEX_IND = 6;  // 96 / 16

// ---------------------------------------------------------------------------
// PxgArticulation raw addressing (updateTransformCacheAndBoundArrayLaunch)
// ---------------------------------------------------------------------------
// The articulation branch indexes scDesc->mArticulationPool[articulationId]
// and reads only linkBody2Worlds / linkBody2Actors (PxTransform* fields), so
// instead of mirroring the whole 496-byte class the pool is addressed with a
// byte stride + field byte offsets.  All three constants are checked against
// host-side sizeof/offsetof by the layout test.  (5.12 values; 5.6.1: 544 /
// 184 / 200.)
public static const uint PXG_ARTICULATION_STRIDE = 496;           // sizeof(PxgArticulation)
public static const uint PXG_ARTI_LINK_BODY2WORLDS_OFFSET = 176;  // offsetof(..., linkBody2Worlds)
public static const uint PXG_ARTI_LINK_BODY2ACTORS_OFFSET = 192;  // offsetof(..., linkBody2Actors)

// loads the PxTransform* linkBody2Worlds / linkBody2Actors array pointer of
// articulation [articulationId] out of the raw PxgArticulation pool
public uint64_t artiLinkTransformArray(uint64_t articulationPool, uint articulationId, uint fieldByteOffset)
{
    const uint64_t artiBase = articulationPool + uint64_t(articulationId) * PXG_ARTICULATION_STRIDE;
    return (Ptr<uint64_t>(artiBase + fieldByteOffset))[0];
}
