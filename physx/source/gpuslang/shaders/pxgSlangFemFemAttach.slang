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

// pxgSlangFemFemAttach — the deformable-vs-DEFORMABLE attachment ABI, shared by the
// softbody-vs-softbody and cloth-vs-cloth attachment solves.
//
// WHY THIS EXISTS.  `PxgFEMFEMAttachmentConstraint` (PxgFEMCore.h:204) is ONE struct
// written by `femAttachmentPrepareLaunch` (softBodyGM.cu:2036) and read back by BOTH
// solves — `sb_solveOutputSoftBodyAttachmentDeltaVLaunch` (softBodyGM.cu:2330) and
// `cloth_solveOutputAttachmentClothClothDeltaVLaunch` (FEMClothExternalSolve.cu:1493).
// Its byte layout is therefore an ABI with three consumers, and the sibling module
// pxgSlangFemRigidAttach already states the rule this follows: two copies of an ABI drift
// silently.  Up to P9.32 this layout sat file-local in softBodyFemAttachBda.slang; the
// cloth solve would have been the second copy, so the layout moved here instead.
//
// WHAT DELIBERATELY DID NOT MOVE.  The two solve BODIES stay apart, and not merely out of
// habit — they are NOT twins, and the differences are load-bearing:
//   - the softbody solve interpolates over FOUR tetrahedron corners, the cloth solve over
//     THREE triangle corners;
//   - the softbody solve carries the cone-limit branch (isConeLimitedEnabled /
//     computeConeLimitedError) and therefore reads low_high_angles, attachmentBarycentric
//     and constraintOffset.  The cloth solve reads NONE of those three fields;
//   - the softbody solve divides by the barycentric-weighted mass sum
//     `sum_i bary[i]^2 * invMass[i]` and gates on it being positive.  The cloth solve
//     divides by `invMass0 + invMass1`, where each is the barycentrically INTERPOLATED
//     `pos.w`, and applies a fixed 0.5 coefficient;
//   - the softbody scatter skips a corner whose weighted inverse mass is not strictly
//     positive; the cloth scatter has NO such skip and writes all three corners
//     unconditionally.
// Folding them together would have to reintroduce every one of those as a branch, which
// is a port-introduced structure that upstream does not have.
//
// The two solves therefore share exactly what upstream shares: the bytes.
module pxgSlangFemFemAttach;

// ---------------------------------------------------------------------------
// Byte layout of PxgFEMFEMAttachmentConstraint (PxgFEMCore.h:204) — a WARP-STRIDED block
// of 32-element arrays.  Every access is
//     base + (workIndex / 32) * FAC_STRIDE + <field> + (workIndex & 31) * elemSize.
//
//   low_high_angles[32]        float4 @0      (union: low_high_limits)
//   attachmentBarycentric[32]  float4 @512    (union: axis_angle)
//   barycentric0[32]           float4 @1024
//   barycentric1[32]           float4 @1536
//   elemId0[32]                PxU64  @2048
//   elemId1[32]                PxU64  @2304
//   constraintOffset[32]       float  @2560 -> 2688
//
// The two leading fields are UNIONS.  Both prepare/solve families use the SECOND name of
// each pair; the first names belong to the RIGID attachment path and never appear here.
//
// elemId0/elemId1 are declared PxU64 but every reader in the engine narrows them to PxU32
// on load ("can be triangleId(cloth) or tetrahedron index").  The port narrows the same
// way — reading the LOW word at the PxU64 stride — because widening it would change which
// bits reach the id/element split.
// ---------------------------------------------------------------------------
public static const uint64_t FAC_STRIDE          = 2688UL;
public static const uint64_t FAC_LOW_HIGH_ANGLES = 0UL;
public static const uint64_t FAC_ATT_BARYCENTRIC = 512UL;
public static const uint64_t FAC_BARYCENTRIC0    = 1024UL;
public static const uint64_t FAC_BARYCENTRIC1    = 1536UL;
public static const uint64_t FAC_ELEMID0         = 2048UL;
public static const uint64_t FAC_ELEMID1         = 2304UL;
public static const uint64_t FAC_CONSTR_OFFSET   = 2560UL;

// Byte address of one field slot within the block that holds `workIndex`.
public uint64_t facSlot(uint64_t base, uint workIndex, uint64_t field, uint64_t elemSize)
{
    return base + uint64_t(workIndex / 32u) * FAC_STRIDE + field + uint64_t(workIndex & 31u) * elemSize;
}
