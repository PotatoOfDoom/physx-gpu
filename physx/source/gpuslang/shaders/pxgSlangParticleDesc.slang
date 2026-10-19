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
// members [136..608).  The base embeds PxGpuParticleData mCommonData at @88; the
// derived class embeds PxgParticleSystemData mData at @248.  Both nested value
// structs are flattened here (mCommonData_/mData_ prefixes).  sizeof = 608,
// alignof = 16 (PX_ALIGN_PREFIX(16)); 608 is a multiple of 16, so the mirror's
// own 8-byte size rounding lands on the same stride.  PhysX 5.12: mData gained
// mPenBiasClamp (+4 B), and the particle cloth / rigid / attachment members are gone.  The layout was verified
// against the engine header with g++ offsetof (see testParticleSim.cpp, which
// pushes a REAL PxgParticleSystem filled with sentinels and reads it back
// through this mirror — any compiler/mirror drift fails that test).
//
// Pointers -> uint64_t (device addresses read through Ptr<>); PxU32/PxReal/int
// -> 4-byte scalars; PxU16 mData.mLockFlags folds into mData_lockFlagsWord (its
// 2 tail-pad bytes are uninitialized host memory).  Explicit _padNNN members
// reproduce the C++ inter-member padding at 444/596.
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

// Byte-exact mirror of PxgParticleSimBuffer (PxgParticleSystem.h, 5.12): 4 device
// pointers (32) + 4 PxU32 (16) = 48.  PhysX 5.9 removed mVolumes, mFilterPairs,
// mRigidAttachments and their counts with particle rigids / particle-rigid filtering.
public struct PxgParticleSimBufferGpu
{
    public uint64_t mPositionInvMasses;   // @0
    public uint64_t mVelocities;          // @8
    public uint64_t mRestPositions;       // @16
    public uint64_t mPhases;              // @24
    public uint mNumActiveParticles;      // @32
    public uint mFlags;                   // @36
    public uint mDiffuseParticleBufferIndex; // @40
    public uint mUniqueId;                // @44 -> 48
}

// The PHASE bit test, PxGetFluid (PxParticleGpu.h:65) over
// PxParticlePhaseFlag::eParticlePhaseFluid (PxParticleSystemFlag.h:94).
//
// P9.13d lifted this here rather than making a THIRD copy: npPsFluidDensity and
// npPsFluidVelocity each carried their own constant and their own one-line
// predicate, and ps_diffuseParticleCreate needs the same test.  Both existing
// files already import this module, so the move is verified by their own suites
// the moment it lands.
public static const uint PS_PHASE_FLUID = 0x00400000u;   // 1<<22
public bool psGetFluid(uint phase) { return (phase & PS_PHASE_FLUID) != 0u; }

// P9.13c -- the DIFFUSE per-buffer record, PxgParticleDiffuseSimBuffer
// (PxgParticleSystem.h:196-210).  Generated from the engine headers and pinned
// by static_assert in testPsDiffuseBuffers.cpp.
//
// 88 bytes: a 40-byte PxDiffuseParticleParams by VALUE at offset 0, then two
// device pointers, the capacity, two more pointers and two PxU32.
//
// TWO THINGS THE LAYOUT ITSELF TRAPS.
//
//   * mParams is EMBEDDED, not a pointer.  PxDiffuseParticleParams is nine
//     PxReal and a bool -- 36 bytes of float plus one byte, padded to 40 -- so
//     every pointer in this struct sits 40 bytes further along than a reader
//     that expected a pointer would put it.
//   * mNumDiffuseParticles is an `int*` to TWO ints, not one.  Slot [0] is the
//     published count and slot [1] is the staging counter the create kernel
//     atomically appends to; ps_diffuseParticleCopy is what moves [1] into [0]
//     and zeroes [1] again.  A reader that treats it as a single int silently
//     works until it needs the staging half.
public struct PxgParticleDiffuseSimBufferGpu
{
    // PxDiffuseParticleParams, PxFiltering-adjacent (PxParticleBuffer.h:235-273).
    public float mParams_threshold;             // @0
    public float mParams_lifetime;              // @4
    public float mParams_airDrag;               // @8
    public float mParams_bubbleDrag;            // @12
    public float mParams_buoyancy;              // @16
    public float mParams_kineticEnergyWeight;   // @20
    public float mParams_pressureWeight;        // @24
    public float mParams_divergenceWeight;      // @28
    public float mParams_collisionDecay;        // @32
    public uint  mParams_useAccurateVelocity;   // @36  a BYTE bool + 3 pad
    public uint64_t mDiffusePositions_LifeTime; // @40  float4*
    public uint64_t mDiffuseVelocities;         // @48  float4*
    public uint  mMaxNumParticles;              // @56
    public uint  _pad60;                        // @60
    public uint64_t mNumDiffuseParticles;       // @64  int*  -- TWO ints, see above
    public uint64_t mNumActiveDiffuseParticles; // @72  int*  (pinned host memory)
    public uint  mStartIndex;                   // @80
    public uint  mFlags;                        // @84 -> 88
}

// Byte-exact mirror of PxgParticleSystem (PxgParticleSystem.h, 5.12) with the
// base PxGpuParticleSystem members inlined first (offsets 0..135) then the
// derived members (136..607).  sizeof = 608.
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
    // ---- mData (PxgParticleSystemData, 96 bytes) @248 ----
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
    public float mData_PenBiasClamp;              // @320 (5.10: = -maxDepenetrationVelocity)
    public uint  mData_LockFlagsWord;             // @324 (low 16 bits = PxU16 mLockFlags; hi 16 = pad @326)
    public uint  mData_NumPhaseToMaterials;       // @328
    public float mData_WindX;                     // @332
    public float mData_WindY;                     // @336
    public float mData_WindZ;                     // @340 (mData ends at 344)
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
    public uint64_t mDiffuseSimBuffers;           // @432
    public uint  mNumDiffuseBuffers;              // @440
    public uint  _pad444;                         // @444 (next member 8-aligned)
    public uint64_t mNumDiffuseParticles;         // @448 int*
    public uint64_t mDiffusePosition_LifeTime;    // @456
    public uint64_t mDiffuseVelocity;             // @464
    public uint64_t mDiffusePotentials;           // @472
    public uint64_t mDiffuseCellStart;            // @480
    public uint64_t mDiffuseCellEnd;              // @488
    public uint64_t mDiffuseGridParticleHash;     // @496
    public uint64_t mDiffuseOriginPos_LifeTime;   // @504
    public uint64_t mDiffuseSortedPos_LifeTime;   // @512
    public uint64_t mDiffuseSortedOriginPos_LifeTime; // @520
    public uint64_t mDiffuseSortedVel;            // @528
    public uint64_t mDiffuseSortedToUnsortedMapping;  // @536
    public uint64_t mDiffuseUnsortedToSortedMapping;  // @544
    public uint64_t mDiffuseOneWayContactInfos;   // @552
    public uint64_t mDiffuseOneWayForces;         // @560
    public uint64_t mDiffuseOneWayNodeIndex;      // @568
    public uint64_t mDiffuseOneWayContactCount;   // @576
    public uint64_t mDerivedPBDMaterialData;      // @584
    public uint  mParticleMaterialStride;         // @592
    public uint  _pad596;                         // @596 (next member 8-aligned)
    public uint64_t mPhaseGroupToMaterialHandle;  // @600 PxU16* -> 608
}

// P9.39c.  Indexing `Ptr<PxgParticleSystemGpu>(base)[id]` already depends on this
// number; the constant exists because the softbody-vs-particle contact gen needs the
// ADDRESS of one element to hand to pxgSlangDefFilter, and an open-coded 608 at that
// call site would be a second place for the layout to drift.  It must equal
// sizeof(PxgParticleSystem) (PxgParticleSystem.h) -- gpuslang.npAbiLayout pins the
// field offsets that make up the total.
public static const uint64_t PXG_PARTICLE_SYSTEM_STRIDE = 608UL;

// ---------------------------------------------------------------------------
// P9.31: two accessors that were file-local in psContactSolveBda and are needed
// again by the cloth-vs-particle solve.  They are moved here rather than copied,
// by the recorded rule that the engine BDA path gets the modules; the copy in
// psContactSolveBda should move onto these the next time that file is touched,
// which is noted rather than done inside a tranche about the cloth chain.
// ---------------------------------------------------------------------------

// phaseToMat[group], where mPhaseGroupToMaterialHandle is a PxU16 array read
// through a 32-bit load and a shift/mask.
public uint pxgLoadPhaseToMaterial(uint64_t handleBase, uint group)
{
    let word = (Ptr<uint>(handleBase))[group >> 1];
    return (word >> ((group & 1u) * 16u)) & 0xffffu;
}

// getParticleMaterial (PxgParticleSystem.h:64) -- a raw byte-offset dereference
// with the RUNTIME material stride, then a float by index.  Float layout
// (PxParticleGpu.h:47): friction@0, damping@1, adhesion@2, gravityScale@3,
// adhesionRadiusScale@4.
public float pxgLoadParticleMaterialFloat(uint64_t materialsBase, uint mi, uint stride, uint floatIndex)
{
    let elem = materialsBase + uint64_t(mi) * uint64_t(stride);
    return (Ptr<float>(elem))[floatIndex];
}

// ---------------------------------------------------------------------------
// P9.95: findBufferIndexFromUniqueId (particleSystem.cuh:99-117), lifted here from
// softBodyParticleAttachBda (its E4) when applyParticleBufferDataDEPRECATED became the
// second user, by the same rule P9.13d applied to psGetFluid: one transcription, not two.
//
// It is NOT dfFindRange.  It is an UPPER-BOUND binary search over
// mParticleBufferSortedUniqueIds (`values[m] > id` moves r down, everything else moves
// l up), then a lookup of r - 1 in the ORIGINAL-INDEX table.  The length-0 early return
// is what keeps r - 1 from wrapping.  There is NO validation, upstream and kept: an id
// below every registered id leaves r == 0 and reads OriginalIndex[0xFFFFFFFF], and an
// unregistered larger id resolves to its predecessor's buffer.
// ---------------------------------------------------------------------------
public uint psFindBufferIndexFromUniqueId(Ptr<PxgParticleSystemGpu> systems, uint systemId, uint uniqueBufferId)
{
    const uint length = systems[systemId].mCommonData_NumParticleBuffers;
    if (length == 0u)                                                    // FB:len0
        return 0u;

    Ptr<uint> values = Ptr<uint>(systems[systemId].mParticleBufferSortedUniqueIds);

    uint l = 0u, r = length;
    while (l < r)
    {
        const uint m = (l + r) / 2u;                                     // FB:mid
        if (values[m] > uniqueBufferId)                                  // FB:cmp
            r = m;
        else
            l = m + 1u;
    }
    return Ptr<uint>(systems[systemId].mParticleBufferSortedUniqueIdsOriginalIndex)[r - 1u];   // FB:ret
}
