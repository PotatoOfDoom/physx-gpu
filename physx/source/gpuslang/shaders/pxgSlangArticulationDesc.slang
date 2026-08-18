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

// pxgSlangArticulationDesc.slang
//
// Byte-exact mirrors for the articulation setup ABI: the descriptor
// PxgUpdateArticulationDesc (gpusimulationcontroller/include/
// PxgSimulationCoreDesc.h), the per-articulation index record
// PxgArticulationSimUpdate, and the FIELD BYTE OFFSETS of PxgArticulation
// (gpusimulationcontroller/include/PxgArticulation.h).
//
// Why offsets and not a struct mirror for PxgArticulation: the class is 544
// bytes of ~40 device pointers plus an embedded PxgArticulationData header.
// pxgSlangSimCoreDesc already addresses two of those pointers raw, for the
// same reason — a full mirror is 40 fields that must each stay in step with a
// header nobody in this port controls, and every one of them is a silent
// wrong-address bug if it drifts.  A named offset constant is one number, and
// every number below is pinned to host-side `sizeof`/`offsetof` of the REAL
// engine struct by static_assert in tests/testNewArticulations.cpp.  A header
// change breaks the build instead of corrupting memory at runtime.
//
// The offsets are NOT in declaration order in the C++ class — they are
// grouped here the way the kernel uses them, with the true offset on each
// line.  Do not infer one from another.
module pxgSlangArticulationDesc;

// ---------------------------------------------------------------------------
// PxgArticulation: strides and field byte offsets
// ---------------------------------------------------------------------------
public static const uint PXG_ARTI_STRIDE = 544;   // sizeof(PxgArticulation)

// PxgArticulationData, embedded at PxgArticulation @0 (32 bytes).  The three
// 16-bit counts are packed, so each needs a halfword extract, not a load.
public static const uint PXG_ARTI_DATA_INDEX = 0;               // PxU32
public static const uint PXG_ARTI_DATA_NUM_LINKS = 16;          // PxU16
public static const uint PXG_ARTI_DATA_NUM_JOINT_DOFS = 18;     // PxU16
public static const uint PXG_ARTI_DATA_NUM_SPATIAL_TENDONS = 20; // PxU16
public static const uint PXG_ARTI_DATA_NUM_FIXED_TENDONS = 22;  // PxU16
public static const uint PXG_ARTI_DATA_NUM_MIMIC_JOINTS = 24;   // PxU16
public static const uint PXG_ARTI_DATA_NUM_PATH_TO_ROOTS = 28;  // PxU32

// the device-pointer fields this kernel writes through
public static const uint PXG_ARTI_LINKS = 32;
public static const uint PXG_ARTI_JOINTS = 40;
public static const uint PXG_ARTI_JOINT_DATA = 48;
public static const uint PXG_ARTI_MOTION_VELOCITIES = 56;
public static const uint PXG_ARTI_MOTION_ACCELERATIONS = 64;
// P8.95a: the incoming-joint-force output array and the two per-link-record
// initial-velocity float4s updateBodiesLaunch_Part2 rewrites.  NOTE THE ORDER
// in the record: the ANGULAR half sits at 0 and the LINEAR at 16 -- the CUDA's
// field names (initialAngVelXYZ_penBiasClamp, initialLinVelXYZ_invMassW) make
// that look arbitrary; the host declaration order is what pins it.
public static const uint PXG_ARTI_LINK_INCOMING_JOINT_FORCES = 72;  // Cm::UnAlignedSpatialVector*
public static const uint PXG_ARTI_LINKREC_INITIAL_ANGVEL_PENBIAS = 0;   // float4 in PxgArticulationLink
public static const uint PXG_ARTI_LINKREC_INITIAL_LINVEL_INVMASS = 16;  // float4 in PxgArticulationLink
public static const uint PXG_ARTI_SLEEPDATA_LINVEL = 0;   // float4 in PxgArticulationLinkSleepData
public static const uint PXG_ARTI_SLEEPDATA_ANGVEL = 16;  // float4
public static const uint PXG_ARTI_CORIOLISE_VECTORS = 80;
// P8.84.  Despite the name this is not only an inverse-dynamics buffer: when
// isExternalForcesEveryTgsIterationEnabled is set, computeSpatialInertiaW parks
// the ISOLATED external Z force per link here, because externalAccelerations
// (96) is only refreshed on demand and must not be overwritten.
public static const uint PXG_ARTI_ZA_FORCES = 88;
public static const uint PXG_ARTI_EXTERNAL_ACCELERATIONS = 96;
public static const uint PXG_ARTI_ROOT_PRE_MOTION_VELOCITY = 104;  // a SINGLE
                                                                  // Cm::UnAlignedSpatialVector,
                                                                  // not an array
public static const uint PXG_ARTI_JOINT_POSITIONS = 112;
public static const uint PXG_ARTI_JOINT_VELOCITIES = 120;
public static const uint PXG_ARTI_JOINT_FORCE = 136;
// PxSpatialMatrix* -- the per-link 6x6 impulse-response matrices, one array per
// articulation.  P8.89: GENERATED, not guessed.  The plan expected it between
// pathToRoot (272) and spatialTendonParams (304); it is nowhere near, which is
// exactly the P8.49a failure mode (shader and fixture sharing one wrong number)
// that generating it closes.
public static const uint PXG_ARTI_SPATIAL_RESPONSE_MATRIX_W = 176;
// sizeof(PxSpatialMatrix) -- a plain PxReal column[6][6]
public static const uint PXG_ARTI_SPATIAL_MATRIX_6X6_SIZE = 144;
public static const uint PXG_ARTI_JOINT_TARGET_POSITIONS = 144;
public static const uint PXG_ARTI_JOINT_TARGET_VELOCITIES = 152;
public static const uint PXG_ARTI_LINK_BODY2WORLDS = 184;
public static const uint PXG_ARTI_LINK_BODY2ACTORS = 200;
public static const uint PXG_ARTI_PARENTS = 208;
public static const uint PXG_ARTI_LINK_WAKE_COUNTERS = 240;
public static const uint PXG_ARTI_LINK_SLEEP_DATA = 248;
public static const uint PXG_ARTI_LINK_PROPS = 256;
public static const uint PXG_ARTI_CHILDREN = 264;
public static const uint PXG_ARTI_PATH_TO_ROOT = 272;
public static const uint PXG_ARTI_SPATIAL_TENDON_PARAMS = 304;
public static const uint PXG_ARTI_SPATIAL_TENDONS = 312;
public static const uint PXG_ARTI_FIXED_TENDON_PARAMS = 320;
public static const uint PXG_ARTI_FIXED_TENDONS = 328;
public static const uint PXG_ARTI_MIMIC_JOINT_CORES = 352;

// ---------------------------------------------------------------------------
// Element strides of the arrays copied through those pointers
// ---------------------------------------------------------------------------
public static const uint PXG_ARTI_LINK_SIZE = 80;             // PxgArticulationLink
public static const uint PXG_ARTI_LINK_PROP_SIZE = 16;        // PxgArticulationLinkProp
public static const uint PXG_ARTI_LINK_SLEEP_DATA_SIZE = 32;  // PxgArticulationLinkSleepData
public static const uint PXG_ARTI_BITFIELD_SIZE = 8;          // Dy::ArticulationBitField (PxU64)
public static const uint PXG_ARTI_TRANSFORM_SIZE = 28;        // PxTransform
public static const uint PXG_ARTI_JOINT_CORE_SIZE = 544;      // Dy::ArticulationJointCore
public static const uint PXG_ARTI_JOINT_CORE_DATA_SIZE = 8;   // Dy::ArticulationJointCoreData
public static const uint PXG_ARTI_SPATIAL_VECTOR_SIZE = 24;   // Cm::UnAlignedSpatialVector
public static const uint PXG_ARTI_MIMIC_JOINT_CORE_SIZE = 32; // Dy::ArticulationMimicJointCore
public static const uint PXG_ARTI_SLEEP_DATA_SIZE = 8;        // PxgSolverBodySleepData

// tendon element strides
public static const uint PXG_ARTI_TENDON_SIZE = 24;            // PxgArticulationTendon
public static const uint PXG_ARTI_TENDON_FIXED_ELEM_SIZE = 16; // PxgArticulationTendonElementFixedData
public static const uint PXG_ARTI_ATTACHMENT_MOD_SIZE = 32;    // PxGpuTendonAttachmentData
public static const uint PXG_ARTI_TENDON_JOINT_COEFF_SIZE = 16; // PxGpuTendonJointCoefficientData
public static const uint PXG_ARTI_SPATIAL_TENDON_DATA_SIZE = 16; // PxGpuSpatialTendonData
public static const uint PXG_ARTI_FIXED_TENDON_DATA_SIZE = 32;  // PxGpuFixedTendonData

// PxgArticulationTendon field offsets (24 bytes)
public static const uint PXG_TENDON_FIXED_ELEMENTS = 0;  // PxgArticulationTendonElementFixedData*
public static const uint PXG_TENDON_MOD_ELEMENTS = 8;    // void*
public static const uint PXG_TENDON_NB_ELEMENTS = 16;    // PxU32

// ---------------------------------------------------------------------------
// PxgArticulationSimUpdate (48 bytes) — one record per new/updated
// articulation, indexing the flattened new-link / new-dof / new-tendon pools.
// Mirrored in full: it is cheap, scalar, and indexed as an array so the whole
// stride is part of the ABI.  Offsets 20 and 28 are the attachment /
// tendon-joint start indices, which this kernel does not read (it takes those
// from the per-tendon remap arrays instead).
// ---------------------------------------------------------------------------
public struct PxgArticulationSimUpdateGpu
{
    public uint dirtyFlags;              // @0
    public uint articulationIndex;       // @4
    public uint linkStartIndex;          // @8
    public uint dofDataStartIndex;       // @12
    public uint spatialTendonStartIndex; // @16
    public uint attachmentStartIndex;    // @20 (unused here)
    public uint fixedTendonStartIndex;   // @24
    public uint tendonJointStartIndex;   // @28 (unused here)
    public uint mimicJointStartIndex;    // @32
    public uint pathToRootIndex;         // @36
    public uint pad0;                    // @40
    public uint pad1;                    // @44
}                                        // = 48

// ---------------------------------------------------------------------------
// PxgUpdateArticulationDesc (216 bytes) — the single descriptor argument of
// newArticulationsLaunch / updateArticulationsLaunch.  Only ever dereferenced
// as desc[0], but mirrored whole because it is cheap and the tail counts sit
// after the pointer block.
// ---------------------------------------------------------------------------
public struct PxgUpdateArticulationDescGpu
{
    public uint64_t mNewArticulations;              // @0   PxgArticulation*
    public uint64_t mNewLinkWakeCounters;           // @8   PxReal*
    public uint64_t mNewLinks;                      // @16  PxgArticulationLink*
    public uint64_t mNewLinkProps;                  // @24  PxgArticulationLinkProp*
    public uint64_t mNewLinkParents;                // @32  PxU32*
    public uint64_t mNewLinkChildren;               // @40  ArticulationBitField*
    public uint64_t mNewLinkBody2Worlds;            // @48  PxTransform*
    public uint64_t mNewLinkBody2Actors;            // @56  PxTransform*
    public uint64_t mNewLinkExtAccels;              // @64  Cm::UnAlignedSpatialVector*
    public uint64_t mNewJointCores;                 // @72  Dy::ArticulationJointCore*
    public uint64_t mNewJointData;                  // @80  Dy::ArticulationJointCoreData*
    public uint64_t mIndicesOffset;                 // @88  PxgArticulationSimUpdate*
    public uint64_t mArticulationPool;              // @96  PxgArticulation*
    public uint64_t mArticulationSleepDataPool;     // @104 PxgSolverBodySleepData*
    public uint64_t mNewSpatialTendonParamsPool;    // @112 PxGpuSpatialTendonData*
    public uint64_t mNewSpatialTendonPool;          // @120 PxgArticulationTendon*
    public uint64_t mNewAttachmentFixedPool;        // @128 PxgArticulationTendonElementFixedData*
    public uint64_t mNewAttachmentModPool;          // @136 PxGpuTendonAttachmentData*
    public uint64_t mNewTendonAttachmentRemapPool;  // @144 PxU32*
    public uint64_t mNewFixedTendonParamsPool;      // @152 PxGpuFixedTendonData*
    public uint64_t mNewFixedTendonPool;            // @160 PxgArticulationTendon*
    public uint64_t mNewTendonJointFixedPool;       // @168 PxgArticulationTendonElementFixedData*
    public uint64_t mNewTendonJointCoefficientPool; // @176 PxGpuTendonJointCoefficientData*
    public uint64_t mNewTendonTendonJointRemapPool; // @184 PxU32*
    public uint64_t mNewArticulationMimicJointPool; // @192 Dy::ArticulationMimicJointCore*
    public uint64_t mNewPathToRootPool;             // @200 PxU32*
    public uint mNbNewArticulations;                // @208
    public uint pad0;                               // @212 (tail pad to 8-byte alignment)
}                                                   // = 216

// ---------------------------------------------------------------------------
// Dy::ArticulationDirtyFlag bits newArticulationsLaunch tests
// ---------------------------------------------------------------------------
public static const uint PXG_ARTI_DIRTY_POSITIONS = 2;
public static const uint PXG_ARTI_DIRTY_VELOCITIES = 4;
public static const uint PXG_ARTI_DIRTY_FORCES = 8;
public static const uint PXG_ARTI_DIRTY_JOINT_TARGET_VEL = 4096;
public static const uint PXG_ARTI_DIRTY_JOINT_TARGET_POS = 8192;
public static const uint PXG_ARTI_DIRTY_EXT_ACCEL = 512;
public static const uint PXG_ARTI_DIRTY_ROOT_TRANSFORM = 16;                // 1 << 4
public static const uint PXG_ARTI_DIRTY_ROOT_VELOCITIES = 32;               // 1 << 5
// the tendon and mimic-joint bits jcalc phase C masks OUT before deciding whether
// the block refresh has anything left to do (P8.76)
public static const uint PXG_ARTI_DIRTY_SPATIAL_TENDON = 16384;             // 1 << 14
public static const uint PXG_ARTI_DIRTY_SPATIAL_TENDON_ATTACHMENT = 32768;  // 1 << 15
public static const uint PXG_ARTI_DIRTY_FIXED_TENDON = 65536;               // 1 << 16
public static const uint PXG_ARTI_DIRTY_FIXED_TENDON_JOINT = 131072;        // 1 << 17
public static const uint PXG_ARTI_DIRTY_MIMIC_JOINT = 262144;               // 1 << 18

// ---------------------------------------------------------------------------
// PxgArticulationCoreDesc (gpuarticulation/include/PxgArticulationCoreDesc.h,
// 360 bytes) — the descriptor the WHOLE articulation solver block shares.
//
// Addressed by named byte offset for the same reason PxgArticulation is: it
// carries ~30 device pointers into block-layout scratch (link blocks, dof
// blocks, tendon blocks, bitfield stacks) and each converted kernel touches a
// handful.  A full mirror would be thirty fields that must stay in step with a
// header this port does not own, every one a silent wrong-address bug on drift.
// Constants are added as kernels need them, each pinned by static_assert.
// ---------------------------------------------------------------------------
public static const uint PXG_ARTI_CORE_BODY_SIM_POOL = 0;      // PxgBodySim*
public static const uint PXG_ARTI_CORE_SLEEP_DATA = 8;         // PxgSolverBodySleepData*
public static const uint PXG_ARTI_CORE_ISLAND_NODE_INDICES = 16; // PxNodeIndex*
public static const uint PXG_ARTI_CORE_SOLVER_BODY_INDICES = 24; // PxU32*
public static const uint PXG_ARTI_CORE_ARTICULATIONS = 32;     // PxgArticulation*
public static const uint PXG_ARTI_CORE_ARTICULATION_OFFSET = 40; // PxU32
public static const uint PXG_ARTI_CORE_NB_ARTICULATIONS = 44;  // PxU32

// P8.82: the rest of what computeUnconstrainedVelocities1TLaunch addresses --
// the nine block-array base pointers, the seven per-warp pitches and
// invLengthScale.  All generated by `offsetof` against the real header and
// pinned in tests/testDmaArtiResidual.cpp; NONE of them are in declaration
// order in the sense a reader would guess, which is the whole reason they are
// generated:
//
//   * `mPathToRootBitFieldBlocks` (176) sits BEFORE `mArticulationDofBlocks`
//     (184) and every tendon block pointer, because the header declares it in
//     the middle of the temp-bitfield group, not with the block pointers the
//     wrapper uses it alongside;
//   * the pitches are NOT in the order the wrapper reads them --
//     `mMaxMimicJointsPerArticulation` (256) sits between `mMaxDofs` (252) and
//     `mMaxSpatialTendons` (260), so a port that lays them out in the order the
//     launch code names them is off by one slot for four of the seven;
//   * `invLengthScale` (64) follows `gravity`, a PxVec3 -- it is exactly the
//     field a wrong sizeof(PxVec3) assumption lands on, and the only way to
//     know it is 64 rather than 60 or 68 is to ask the compiler.
//
// The gaps are real and accounted for: 48..63 are dt (48) + gravity (52), 68 is
// the external-forces bool, 72..127 the impulse/slab/partition group, 144..175
// the traversal stack (144) and the three TEMP bitfield pointers (152/160/168),
// and 200 / 224 the two tendon CONSTRAINT block pointers -- which are NOT the
// tendon block pointers mapped above and must not be confused with them.
// Unmapped on purpose --
// this port adds a constant when a kernel needs it, never speculatively.
// P8.84: the three step parameters computeSpatialInertiaW needs.  `gravity` is
// the PxVec3 at 52 that invLengthScale at 64 sits behind, which is why a wrong
// sizeof(PxVec3) lands on invLengthScale and not on gravity itself; and
// isExternalForcesEveryTgsIterationEnabled is a `bool`, i.e. ONE BYTE at 68 --
// reading it as a word picks up the four padding bytes before `impulses` at 72.
// P8.94d: the three fields stepArticulation1TTGS hands to
// averageLinkImpulsesAndPropagate.  Generated by offsetof against the real
// header (the prose comment below saying "impulses at 72" predates them and was
// NOT the source).  `nbPartitions` at 84 sits between nbSlabs and
// slabHasChanges and stays unmapped; `slabDirtyMasks` at 96 is a DIFFERENT
// array with different indexing and must not be confused with slabHasChanges.
public static const uint PXG_ARTI_CORE_IMPULSES = 72;           // Cm::UnAlignedSpatialVector*
public static const uint PXG_ARTI_CORE_NB_SLABS = 80;           // PxU32
public static const uint PXG_ARTI_CORE_SLAB_HAS_CHANGES = 88;   // uint2*
// P8.101b: the per-partition propagate pointers, bracketed by the pinned
// neighbors SLAB_HAS_CHANGES (88) and ARTICULATION_BLOCKS (128)
public static const uint PXG_ARTI_CORE_SLAB_DIRTY_MASKS = 96;              // uint4*
public static const uint PXG_ARTI_CORE_PATH_TO_ROOTS_PER_PARTITION = 104;  // PxgArticulationBitFieldStackData*
public static const uint PXG_ARTI_CORE_IMPULSE_HOLDING_LINK = 112;         // PxU32*
public static const uint PXG_ARTI_CORE_PARTITION_AVERAGE_SCALE = 120;      // PxReal*
public static const uint PXG_ARTI_CORE_DT = 48;                 // PxReal
public static const uint PXG_ARTI_CORE_GRAVITY = 52;            // PxVec3
public static const uint PXG_ARTI_CORE_EXTERNAL_FORCES_EVERY_TGS_ITER = 68;  // bool
public static const uint PXG_ARTI_CORE_INV_LENGTH_SCALE = 64;   // PxReal
public static const uint PXG_ARTI_CORE_ARTICULATION_BLOCKS = 128;        // PxgArticulationBlockData*
public static const uint PXG_ARTI_CORE_LINK_BLOCKS = 136;                // PxgArticulationBlockLinkData*
// P8.100c: the DFS traversal stack of artiSolveInternalConstraints
public static const uint PXG_ARTI_CORE_TRAVERSAL_STACK = 144;   // PxgArticulationTraversalStackData*
// P8.99b: the three per-batch scratch bitfield buffers the self-constraint
// solve stages its link-path masks into (mTempPathToRootBitFieldBlocks /
// mTempSharedBitFieldBlocks / mTempRootBitFieldBlocks -- header lines 87-89,
// between mTraversalStack at 144 and mPathToRootBitFieldBlocks at 176).
public static const uint PXG_ARTI_CORE_TEMP_PATH_TO_ROOT_BITFIELD_BLOCKS = 152; // PxgArticulationBitFieldStackData*
public static const uint PXG_ARTI_CORE_TEMP_SHARED_BITFIELD_BLOCKS = 160;       // PxgArticulationBitFieldStackData*
public static const uint PXG_ARTI_CORE_TEMP_ROOT_BITFIELD_BLOCKS = 168;         // PxgArticulationBitFieldStackData*
public static const uint PXG_ARTI_CORE_PATH_TO_ROOT_BITFIELD_BLOCKS = 176; // PxgArticulationBitFieldData*
public static const uint PXG_ARTI_CORE_DOF_BLOCKS = 184;                 // PxgArticulationBlockDofData*
public static const uint PXG_ARTI_CORE_SPATIAL_TENDON_BLOCKS = 192;      // PxgArticulationBlockSpatialTendonData*
public static const uint PXG_ARTI_CORE_SPATIAL_TENDON_CONSTRAINT_BLOCKS = 200; // PxgArticulationInternalTendonConstraintData* (P8.97b)
public static const uint PXG_ARTI_CORE_FIXED_TENDON_CONSTRAINT_BLOCKS = 224;   // PxgArticulationInternalTendonConstraintData* (P8.97b)
public static const uint PXG_ARTI_CORE_ATTACHMENT_BLOCKS = 208;          // PxgArticulationBlockAttachmentData*
public static const uint PXG_ARTI_CORE_FIXED_TENDON_BLOCKS = 216;        // PxgArticulationBlockFixedTendonData*
public static const uint PXG_ARTI_CORE_TENDON_JOINT_BLOCKS = 232;        // PxgArticulationBlockTendonJointData*
public static const uint PXG_ARTI_CORE_MIMIC_JOINT_BLOCKS = 240;         // PxgArticulationBlockMimicJointData*
// P8.96a: the eight per-block scratch arrays artiSumInternalContactAndJoint-
// Batches1 reduces into (contiguous run 280..343, generated and pinned).  The
// UNIQUE-INDICES arrays carry plain per-block SUMS; the HEADER arrays carry
// sums of PER-WARP-CHUNK MAXES -- same convention as the rigid twin's four.
public static const uint PXG_ARTI_CORE_TEMP_CONTACT_UNIQUE = 280;        // PxU32*
public static const uint PXG_ARTI_CORE_TEMP_CONSTRAINT_UNIQUE = 288;     // PxU32*
public static const uint PXG_ARTI_CORE_TEMP_CONTACT_HEADER = 296;        // PxU32*
public static const uint PXG_ARTI_CORE_TEMP_CONSTRAINT_HEADER = 304;     // PxU32*
public static const uint PXG_ARTI_CORE_TEMP_SELF_CONTACT_UNIQUE = 312;   // PxU32*
public static const uint PXG_ARTI_CORE_TEMP_SELF_CONSTRAINT_UNIQUE = 320; // PxU32*
public static const uint PXG_ARTI_CORE_TEMP_SELF_CONTACT_HEADER = 328;   // PxU32*
public static const uint PXG_ARTI_CORE_TEMP_SELF_CONSTRAINT_HEADER = 336; // PxU32*
public static const uint PXG_ARTI_CORE_MAX_LINKS = 248;                  // PxU32
public static const uint PXG_ARTI_CORE_MAX_DOFS = 252;                   // PxU32
public static const uint PXG_ARTI_CORE_MAX_MIMIC_JOINTS = 256;           // PxU32
public static const uint PXG_ARTI_CORE_MAX_SPATIAL_TENDONS = 260;        // PxU32
public static const uint PXG_ARTI_CORE_MAX_ATTACHMENTS = 264;            // PxU32
public static const uint PXG_ARTI_CORE_MAX_FIXED_TENDONS = 268;          // PxU32
public static const uint PXG_ARTI_CORE_MAX_TENDON_JOINTS = 272;          // PxU32

// PxgArticulation's two residual accumulators, and the stride of one.
// Dy::ErrorAccumulator is 12 bytes of plain scalars, so the whole-struct
// assignment the CUDA writes is a 12-byte copy at any granularity.
public static const uint PXG_ARTI_INTERNAL_RESIDUAL = 512;
public static const uint PXG_ARTI_CONTACT_RESIDUAL = 524;
public static const uint PXG_ARTI_ERROR_ACCUMULATOR_SIZE = 12;

// ---------------------------------------------------------------------------
// P8.93: PxgArticulationOutputDesc (32 bytes) and what feeds it
// ---------------------------------------------------------------------------
//
// The scene-wide staging struct `dmaBackArticulationDataLaunch` writes into.
// Generated from the engine headers and pinned by static_assert in
// tests/testArtiDmaBack.cpp -- the P8.49a failure mode (shader and fixture
// sharing one wrong number) is closed only by generation.
public static const uint PXG_ARTI_OUT_STATE_DATA = 0;    // PxU8*  linkAndJointAndRootStateData
public static const uint PXG_ARTI_OUT_SLEEP_DATA = 8;    // PxgSolverBodySleepData*
public static const uint PXG_ARTI_OUT_ERROR_ACC = 16;    // Dy::ErrorAccumulator*   per articulation
public static const uint PXG_ARTI_OUT_CONTACT_ACC = 24;  // Dy::ErrorAccumulator*   one for the scene
public static const uint PXG_ARTI_OUT_DESC_SIZE = 32;

// PxgArticulationCoreDesc::mContactErrorAccumulator -- a VALUE, not a pointer,
// and it sits at the very end of the 360-byte descriptor.
public static const uint PXG_ARTI_CORE_CONTACT_ERROR_ACC = 344;

// PxgArticulation::linkJointRootStateDataBuffer -- the per-articulation source
// the scene-wide staging buffer is filled FROM.  Every one of
// `linkBody2Worlds`, `motionVelocities`, `motionAccelerations`,
// `linkIncomingJointForces`, `jointPositions`, `jointVelocities`,
// `jointAccelerations` and `rootPreMotionVelocity` is a sub-range of it
// (PxgSimulationCore.cpp:893), which is why copying this one buffer copies the
// whole articulation state.
public static const uint PXG_ARTI_LINK_JOINT_ROOT_STATE = 192;

// PxgArticulationLinkJointRootStateData's arithmetic (PxgArticulation.h:768).
// sizeof(PxTransform) is 28, NOT 32 -- it is a quat and a vec3 with no padding,
// and a port that rounds it up gets every link's stride wrong.
public static const uint PXG_ARTI_STATE_TRANSFORM_SIZE = 28;
public static const uint PXG_ARTI_STATE_SVEC_SIZE = 24;   // Cm::UnAlignedSpatialVector

// (sizeof(PxTransform) + 3 * sizeof(UnAlignedSpatialVector)) * links
//   + 3 * sizeof(PxReal) * dofs + sizeof(UnAlignedSpatialVector)
public static uint artiStateBytesRaw(uint maxLinks, uint maxDofs)
{
    return (PXG_ARTI_STATE_TRANSFORM_SIZE + 3u * PXG_ARTI_STATE_SVEC_SIZE) * maxLinks
         + 3u * 4u * maxDofs + PXG_ARTI_STATE_SVEC_SIZE;
}

// The per-articulation stride, rounded UP to 16 -- and note the engine multiplies
// the ALIGNED per-articulation size by the count, so the rounding happens once
// per articulation rather than once for the whole ensemble.
public static uint artiStateBytesAligned16(uint maxLinks, uint maxDofs, uint nbArticulations)
{
    const uint raw = artiStateBytesRaw(maxLinks, maxDofs);
    return ((raw + 15u) & ~15u) * nbArticulations;
}

// PxNodeIndex is a PxU64 whose low word is the index and whose high word
// carries the articulation flag + link id (pxgSlangCommon::nodeIndexIndex).
public static const uint PXG_NODE_INDEX_SIZE = 8;

// PxgBodySim::articulationRemapId (pxgSlangCommon mirrors the struct at 224
// bytes; this is the byte offset the raw-addressed path needs).
public static const uint PXG_BODY_SIM_SIZE = 224;
public static const uint PXG_BODY_SIM_ARTICULATION_REMAP_ID = 176;
// P8.83: the block header's mSleepThreshold comes from `.z` of the float4
// `freezeThresholdX_wakeCounterY_sleepThresholdZ_bodySimIndex` at 96 -- so 104,
// which is a DERIVED offset and therefore pinned in its own right rather than
// left as "96 + 8" at the call site.  Reading it as the float4's `.x` or `.y`
// picks up the freeze threshold or the wake counter, both plausible reals.
public static const uint PXG_BODY_SIM_SLEEP_THRESHOLD = 104;

// ---------------------------------------------------------------------------
// The fields jcalc's phase A recomputes (P8.72).  Note the two SINGLE BYTES
// sharing one word: `flags` at 26 and `confiDirty` at 27.  Writing confiDirty
// therefore means a read-modify-write of that word, which is safe here only
// because phase A runs one thread per articulation -- no other thread touches
// this record concurrently.  `updateDirty` is a PxU32 at 12 and needs no such
// care.  (Widths measured, not inferred: the ArticulationJointCoreData::nbDof
// byte in P8.70 is why.)
// ---------------------------------------------------------------------------
public static const uint PXG_ARTI_JOINT_AXIS                             =    232;
public static const uint PXG_ARTI_MOTION_MATRIX                          =    216;
public static const uint PXG_ARTI_RELATIVE_QUAT                          =    280;
public static const uint PXG_ARTI_JOINT_OFFSETS                          =    160;
public static const uint PXG_ARTI_CFM_SCALE                              =    344;
public static const uint PXG_ARTI_CFMS                                   =    336;
public static const uint PXG_ARTI_DATA_FLAGS                            =     26;
public static const uint PXG_ARTI_DATA_CONFI_DIRTY                      =     27;
public static const uint PXG_ARTI_DATA_UPDATE_DIRTY                     =     12;
public static const uint PXG_ARTI_DATA_gpuDirtyFlag                      =      8;
public static const uint PXG_ARTI_LINK_PROP_INV_MASS                    =     12;
public static const uint PXG_ARTI_LINK_CFM_SCALE                        =     72;
public static const uint PXG_ARTI_FLAG_FIX_BASE                         =      1;
// PxgArtiStateDirtyFlag (P8.94d): the block's mStateDirty is a MASK over these
// two bits, not a boolean -- callers set eVEL_DIRTY | eHAS_IMPULSES, so 3 is a
// reachable value.  NOT the same flag set as PXG_ARTI_DATA_* above, which is
// the record-side gpuDirtyFlag family.
public static const uint PXG_ARTI_STATE_VEL_DIRTY                       =      1;
public static const uint PXG_ARTI_STATE_HAS_IMPULSES                    =      2;
public static const uint PXG_ARTI_SUBSPACE_STRIDE                       =     80;
public static const uint PXG_ARTI_QUAT_SIZE                             =     16;

// reads a single byte out of a record, via the containing word
public uint artiU8Field(uint64_t rec, uint fieldByteOffset)
{
    const uint word = (Ptr<uint>(rec + uint64_t(fieldByteOffset & ~3u)))[0];
    return (word >> ((fieldByteOffset & 3u) * 8u)) & 0xFFu;
}

// writes a single byte, preserving the other three in its word.  Single-writer
// only -- see the note above.
public void artiStoreU8Field(uint64_t rec, uint fieldByteOffset, uint value)
{
    Ptr<uint> w = Ptr<uint>(rec + uint64_t(fieldByteOffset & ~3u));
    const uint shift = (fieldByteOffset & 3u) * 8u;
    w[0] = (w[0] & ~(0xFFu << shift)) | ((value & 0xFFu) << shift);
}

// ---------------------------------------------------------------------------
// Field accessors
// ---------------------------------------------------------------------------

// loads a device-pointer field out of a raw PxgArticulation record
public uint64_t artiPtrField(uint64_t artiRecord, uint fieldByteOffset)
{
    return (Ptr<uint64_t>(artiRecord + uint64_t(fieldByteOffset)))[0];
}

// loads a packed PxU16 count out of a raw PxgArticulationData header.  The
// offsets are even but not multiples of four, so this is a word load plus a
// halfword extract — NOT a 16-bit load, which SPIR-V would need an extra
// capability for.
public uint artiU16Field(uint64_t artiRecord, uint fieldByteOffset)
{
    const uint word = (Ptr<uint>(artiRecord + uint64_t(fieldByteOffset & ~3u)))[0];
    const uint shift = (fieldByteOffset & 2u) != 0u ? 16u : 0u;
    return (word >> shift) & 0xFFFFu;
}

// loads a PxU32 field out of a raw record
public uint artiU32Field(uint64_t artiRecord, uint fieldByteOffset)
{
    return (Ptr<uint>(artiRecord + uint64_t(fieldByteOffset)))[0];
}
