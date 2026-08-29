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

// ---------------------------------------------------------------------------
// THE SHARED HELPERS.  Each is ONE CUDA definition with two consumers -- the
// softbody-vs-softbody solve (softBodyGM.cu:2330) and the softbody-vs-cloth solve
// (softBodyGM.cu:2424) -- which is the case pxgSlangFemRigidAttach's header says to
// share: "one CUDA function used by both -> share it, or the copies rot".  They sat
// file-local in softBodyFemAttachBda.slang until P9.33 made that file's twin exist.
// ---------------------------------------------------------------------------
import pxgSlangAtomicFloat;   // pxgAtomicAddF4XYZ / pxgAtomicAddF4W

public static const float FAC_MAX_F32 = 3.4028235e38f;

// PxVec3::getNormalized (PxVec3.h:286) = `m > 0 ? v * PxRecipSqrt(m) : 0`, and
// PxRecipSqrt(float) (PxMath.h:161) is intrinsics::recipSqrt, which on unix is
// `1.0f / ::sqrtf(a)` (PxUnixMathIntrinsics.h:93).  BOTH operations are exactly
// specified by IEEE 754, so this expression has one right answer and the CPU
// mirrors compute it.
//
// This is spelled as a reciprocal and a multiply, NOT as `v / sqrt(m)`: those
// differ, because a divide rounds once per component while a reciprocal rounds
// once and then each multiply rounds again.  It is also NOT spelled `rsqrt(m)`,
// which is what this file's predecessor used: SPIR-V's InverseSqrt is allowed
// 2 ULP by Vulkan and is therefore a different function on paper.  On llvmpipe
// the two agree bit for bit over 258442 inputs -- gpuslang.npFloatIntrinRecipSqrt
// (P9.33a) measures exactly that, which is what lets the corpus's other ~40 rsqrt
// sites stand.  Here the literal transcription costs nothing, so it is used.
public float3 facGetNormalized(float3 v)
{
    const float m = dot(v, v);
    if (m > 0.0f)
    {
        const float rs = 1.0f / sqrt(m);
        return float3(v.x * rs, v.y * rs, v.z * rs);
    }
    return float3(0.0f, 0.0f, 0.0f);
}

// attachments.cuh:36.
public bool isConeLimitedEnabled(float maxAngle, float minDist, float maxDist)
{
    return maxAngle >= 0.0f || minDist >= 0.0f || maxDist >= 0.0f;
}

// attachments.cuh:48 -- the minimum translation from relPos into the allowed cone
// volume.  The sin/cos pair is the only transcendental seam in the attachment path.
public float3 computeConeLimitedError(float maxAngle, float minDist, float maxDist,
                                      float3 coneAxis, float3 relPos)
{
    float len = length(relPos);

    float3 dir;
    if (maxAngle == 0.0f)
    {
        dir = coneAxis;
    }
    else if (maxAngle > 0.0f)
    {
        dir = (len > 1.0e-6f) ? (relPos / len) : coneAxis;
        const float cosAngle = dot(dir, coneAxis);
        const float sinMaxAngle = sin(maxAngle);
        const float cosMaxAngle = cos(maxAngle);
        if (cosAngle < cosMaxAngle)   // theta > maxAngle
        {
            const float3 t1 = cross(dir, coneAxis);
            const float3 b1 = facGetNormalized(cross(coneAxis, t1));
            dir = cosMaxAngle * coneAxis + sinMaxAngle * b1;
        }
    }
    else
    {
        dir = (len > 1.0e-6f) ? (relPos / len) : coneAxis;
    }

    len = clamp(len, minDist, maxDist >= 0.0f ? maxDist : FAC_MAX_F32);

    return relPos - len * dir;   // the ideal relPos is len * dir
}

// updateTetPositionDelta (softBody.cuh:105) and updateTriPositionDelta (:130).
// FOUR corners against THREE, and that is their only difference -- both skip a
// corner whose weighted inverse mass is not strictly positive, and both add
// `constraintWeight` to the .w lane of every corner they do touch.
//
// The weight is a PARAMETER here even though all five CUDA call sites pass 1.f
// (softBodyGM.cu:2416, :2417, :2509, :2510, :2599).  A shared helper that hardcodes
// what its callers happen to pass is a helper that stops matching the day one of
// them passes something else, and nothing would notice.
public void updateTetPositionDelta(uint64_t out, uint4 tetVertIndices, float3 deltaPosition,
                                   float4 invMassBary, float constraintWeight)
{
    if (invMassBary.x > 0.0f)
    {
        pxgAtomicAddF4XYZ(out, tetVertIndices.x, deltaPosition * invMassBary.x);
        pxgAtomicAddF4W(out, tetVertIndices.x, constraintWeight);
    }
    if (invMassBary.y > 0.0f)
    {
        pxgAtomicAddF4XYZ(out, tetVertIndices.y, deltaPosition * invMassBary.y);
        pxgAtomicAddF4W(out, tetVertIndices.y, constraintWeight);
    }
    if (invMassBary.z > 0.0f)
    {
        pxgAtomicAddF4XYZ(out, tetVertIndices.z, deltaPosition * invMassBary.z);
        pxgAtomicAddF4W(out, tetVertIndices.z, constraintWeight);
    }
    if (invMassBary.w > 0.0f)
    {
        pxgAtomicAddF4XYZ(out, tetVertIndices.w, deltaPosition * invMassBary.w);
        pxgAtomicAddF4W(out, tetVertIndices.w, constraintWeight);
    }
}

// The triangle form.  It reads `invMassBary.w` NEVER -- not even to test it -- so a
// caller may leave that slot holding anything.  The one caller in this port passes a
// literal 0.0f there because the CUDA does (softBodyGM.cu:2500); see the trap note in
// softBodyClothAttachBda.slang for why that zero is load-bearing elsewhere.
public void updateTriPositionDelta(uint64_t out, uint4 triVertIndices, float3 deltaPosition,
                                   float4 invMassBary, float constraintWeight)
{
    if (invMassBary.x > 0.0f)
    {
        pxgAtomicAddF4XYZ(out, triVertIndices.x, deltaPosition * invMassBary.x);
        pxgAtomicAddF4W(out, triVertIndices.x, constraintWeight);
    }
    if (invMassBary.y > 0.0f)
    {
        pxgAtomicAddF4XYZ(out, triVertIndices.y, deltaPosition * invMassBary.y);
        pxgAtomicAddF4W(out, triVertIndices.y, constraintWeight);
    }
    if (invMassBary.z > 0.0f)
    {
        pxgAtomicAddF4XYZ(out, triVertIndices.z, deltaPosition * invMassBary.z);
        pxgAtomicAddF4W(out, triVertIndices.z, constraintWeight);
    }
}
