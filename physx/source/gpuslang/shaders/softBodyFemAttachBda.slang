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
//
// ===========================================================================
// softBodyFemAttachBda — the FEM-vs-FEM ATTACHMENT pair, on the engine BDA ABI:
//   - femAttachmentPrepareLaunch (softBodyGM.cu:2036, launch PxgSoftBodyCore.cpp:2100,
//     FEM_ATTACHMENT_CONSTRAINT_PREP): scatters the per-attachment descriptor into the
//     warp-strided PxgFEMFEMAttachmentConstraint block the solve reads;
//   - sb_solveOutputSoftBodyAttachmentDeltaVLaunch (softBodyGM.cu:2330, launch
//     PxgSoftBodyCore.cpp:1921, SB_SOLVE_SOFTBODY_ATTACHMENT_DELTA): one mass-weighted
//     position correction per attachment, scattered into BOTH softbodies' mSimDelta.
//
// This is the last piece that lets one FEM volume HANG FROM another with no contact in the
// loop.  The rigid-attachment siblings (sb_rigidAttachmentPrepareLaunch /
// sb_solveRigidSoftAttachmentLaunch) landed at P8.3; this is the same shape with a second
// deformable in place of the rigid, which makes it SIMPLER, not harder: there is no velocity
// reader, no articulation branch and no rigid response term — both sides are position-based.
//
// THE CONSTRAINT IS PURE POSITION PROJECTION.  For attachment k the solve interpolates an
// anchor in each softbody's simulation tetrahedron by its barycentric coordinates, takes the
// error `pos1 - pos0`, and splits it between the two by the barycentric-weighted inverse
// masses:
//
//     wTot0 = sum_i bary0[i]^2 * invMass0[i]          (b0w[i] = bary0[i] * invMass0[i])
//     wTot1 = sum_i bary1[i]^2 * invMass1[i]
//     delta = error / (wTot0 + wTot1)
//     softbody1 -= delta * b1w      softbody0 += delta * b0w
//
// `wTot > 0.0f` is the only gate: two fully pinned tetrahedra (all invMass 0) contribute
// nothing and are skipped rather than dividing by zero.
//
// THE TWO STRUCT LAYOUTS ARE THE WHOLE RISK, so they are spelled out as named offsets below
// and pinned by static_asserts in the fixture.  Note in particular:
//   - PxgFEMFEMAttachmentConstraint is a WARP-STRIDED block: every field is an array of 32,
//     indexed by `workIndex & 31`, and the block index is `workIndex / 32`.  Its two leading
//     fields are UNIONS (`low_high_limits`/`low_high_angles` and `axis_angle`/
//     `attachmentBarycentric`) — this pair of kernels uses the SECOND name of each;
//   - `elemId1` is declared PxU64 in the constraint block and assigned from a PxU32
//     (`attachment.index1`), then read back as a PxU32.  The port reads the LOW WORD, which is
//     what the CUDA narrowing conversion does; reading the full 64 bits would be wrong the
//     moment the upper half held anything.
//
// CONE LIMITS ARE PORTED BUT DORMANT ON EVERY SCENE HERE.  `isConeLimitedEnabled(maxAngle,
// minDist, maxDist)` is `maxAngle >= 0 || minDist >= 0 || maxDist >= 0` (attachments.cuh:36),
// and an attachment created without a PxConeLimitedConstraint carries -1 in all three.  The
// branch is transcribed anyway — including its PxSinCos, the ONLY transcendental in this
// shader and therefore the only tolerance seam — and the fixture drives it directly.
//
// *** THE GRID. ***  Both kernels launch grid (SB_UPDATEROTATION = 1024, 1, 1) with block
// (SB_UPDATEROTATION = 256, 1, 1) and use the engine's standard strided form:
//
//     nbBlocksRequired      = ceil(numAttachments / blockDim.x)
//     nbIterationsPerBlock  = ceil(nbBlocksRequired / gridDim.x)
//     workIndex = i * blockDim.x + threadIdx.x + nbIterationsPerBlock * blockIdx.x * blockDim.x
//
// Neither kernel has any cross-lane operation — no ballot, no shuffle, no shared memory — so
// unlike the midphase ports this one keeps the CUDA block shape verbatim as
// [numthreads(256,1,1)] and the arithmetic is transcribed unchanged.  The `sub`-loop lowering
// the warp-cooperative kernels need would buy nothing here.
//
// DEVIATIONS from CUDA: the only one is `AtomicAdd(float4&, PxVec3, 1.f)` ->
// pxgAtomicAddF4XYZ + pxgAtomicAddF4W (the module that turns VK_EXT_shader_atomic_float into
// the same read-modify-write the CUDA helper performs), and `return` inside the strided loop
// becomes `break` because Slang's entry point has cleanup after it.  Both preserve the emitted
// deltas exactly.
//
// NoContraction-stamped (PXG_NOCONTRACT_SHADERS): the barycentric interpolations, the
// inverse-mass weighting and the cone-limit projection are mul-feeding-add chains compared
// bit-exactly against the -ffp-contract=off CPU reference.
// ===========================================================================
module softBodyFemAttachBda;

import pxgSlangSoftBodyDesc;   // PxgSoftBodyGpu (mSimTetIndices@80, mSimPosition_InvMass@96, mSimDelta@272)
import pxgSlangAtomicFloat;    // pxgAtomicAddF4XYZ / pxgAtomicAddF4W

// PxgSoftBodyKernelBlockDim/GridDim::SB_UPDATEROTATION (PxgSoftBodyCoreKernelIndices.h:42/59).
static const uint SB_BLOCK = 256u;
static const uint SB_GRID = 1024u;

// PxEncodeSoftBodyIndex(id, elem) = (id << 20) | elem; PX_MAX_NB_DEFORMABLE_VOLUME_TET is the
// element mask (PxgSimulationCoreDesc.h).
static const uint PXG_SB_ID_SHIFT = 20u;
static const uint PXG_SB_ELEM_MASK = 0x000fffffu;

static const float PX_MAX_F32 = 3.4028235e38f;

// ---------------------------------------------------------------------------
// PxgFEMFEMAttachment (PxgSimulationCoreDesc.h:265, PX_ALIGN(16), sizeof 80).
//   coneLimitParams.low_high_angle  float4 @0     (union with low_high_limits)
//   coneLimitParams.barycentric     float4 @16    (union with axis_angle)
//   barycentricCoordinates0         float4 @32
//   barycentricCoordinates1         float4 @48
//   index0                          PxU64  @64
//   index1                          PxU32  @72
//   constraintOffset                PxReal @76  -> 80
// ---------------------------------------------------------------------------
static const uint64_t FA_STRIDE          = 80UL;
static const uint64_t FA_LOW_HIGH_ANGLE  = 0UL;
static const uint64_t FA_BARYCENTRIC     = 16UL;
static const uint64_t FA_BARY0           = 32UL;
static const uint64_t FA_BARY1           = 48UL;
static const uint64_t FA_INDEX0          = 64UL;
static const uint64_t FA_INDEX1          = 72UL;
static const uint64_t FA_CONSTR_OFFSET   = 76UL;

// ---------------------------------------------------------------------------
// PxgFEMFEMAttachmentConstraint (PxgFEMCore.h:204) — a WARP-STRIDED block of 32-element
// arrays.  Every access is `base + block * FAC_STRIDE + field + offset * elemSize`.
//   low_high_angles[32]        float4 @0      (union: low_high_limits)
//   attachmentBarycentric[32]  float4 @512    (union: axis_angle)
//   barycentric0[32]           float4 @1024
//   barycentric1[32]           float4 @1536
//   elemId0[32]                PxU64  @2048
//   elemId1[32]                PxU64  @2304
//   constraintOffset[32]       float  @2560 -> 2688
// ---------------------------------------------------------------------------
static const uint64_t FAC_STRIDE          = 2688UL;
static const uint64_t FAC_LOW_HIGH_ANGLES = 0UL;
static const uint64_t FAC_ATT_BARYCENTRIC = 512UL;
static const uint64_t FAC_BARYCENTRIC0    = 1024UL;
static const uint64_t FAC_BARYCENTRIC1    = 1536UL;
static const uint64_t FAC_ELEMID0         = 2048UL;
static const uint64_t FAC_ELEMID1         = 2304UL;
static const uint64_t FAC_CONSTR_OFFSET   = 2560UL;

// ---------------------------------------------------------------------------
// Push-constant blobs.
// ---------------------------------------------------------------------------
// femAttachmentPrepareLaunch(PxgFEMFEMAttachment* femAttachments, PxU32* activeFemAttachments,
//   PxU32 numFemAttachments, PxgFEMFEMAttachmentConstraint* attachmentConstraints)
//   -> 8 @0, 8 @8, 4 @16, a pointer realigned to @24 -> 32-byte blob.  ABI row {8,8,4,8}.
struct FemAttachPrepParams
{
    uint64_t femAttachments;        // @0
    uint64_t activeFemAttachments;  // @8
    uint numFemAttachments;         // @16
    uint _pad20;                    // @20
    uint64_t attachmentConstraints; // @24 -> 32
}

// sb_solveOutputSoftBodyAttachmentDeltaVLaunch(PxgSoftBody* softbodies,
//   PxgFEMFEMAttachmentConstraint* attachments, PxU32 numAttachments)
//   -> 8 @0, 8 @8, 4 @16 -> 24-byte blob (the trailing PxU32 padded to the 8-byte alignment the
//   pack already carries).  ABI row {8,8,4}.
struct FemAttachSolveParams
{
    uint64_t softbodies;            // @0
    uint64_t attachments;           // @8
    uint numAttachments;            // @16
    uint _pad20;                    // @20 -> 24
}

[[vk::push_constant]] ConstantBuffer<FemAttachPrepParams> gAp;

// ---------------------------------------------------------------------------
// Cone-limit helpers (attachments.cuh:36 / :48), transcribed literally.
// ---------------------------------------------------------------------------
bool isConeLimitedEnabled(float maxAngle, float minDist, float maxDist)
{
    return maxAngle >= 0.0f || minDist >= 0.0f || maxDist >= 0.0f;
}

float3 faGetNormalized(float3 v)
{
    const float m = dot(v, v);
    return m > 0.0f ? v * rsqrt(m) : float3(0, 0, 0);
}

// The minimum translation from relPos into the allowed cone volume.  PxSinCos is the only
// transcendental in this shader.
float3 computeConeLimitedError(float maxAngle, float minDist, float maxDist,
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
            const float3 b1 = faGetNormalized(cross(coneAxis, t1));
            dir = cosMaxAngle * coneAxis + sinMaxAngle * b1;
        }
    }
    else
    {
        dir = (len > 1.0e-6f) ? (relPos / len) : coneAxis;
    }

    len = clamp(len, minDist, maxDist >= 0.0f ? maxDist : PX_MAX_F32);

    return relPos - len * dir;   // the ideal relPos is len * dir
}

// updateTetPositionDelta (softBody.cuh:105): AtomicAdd(dst, delta * invMassBary_k, 1.f) for
// each corner whose weight is strictly positive — xyz takes the scaled delta, .w takes ONE.
void updateTetPositionDelta(uint64_t out, uint4 tetVertIndices, float3 deltaPosition,
                            float4 invMassBary)
{
    if (invMassBary.x > 0.0f)
    {
        pxgAtomicAddF4XYZ(out, tetVertIndices.x, deltaPosition * invMassBary.x);
        pxgAtomicAddF4W(out, tetVertIndices.x, 1.0f);
    }
    if (invMassBary.y > 0.0f)
    {
        pxgAtomicAddF4XYZ(out, tetVertIndices.y, deltaPosition * invMassBary.y);
        pxgAtomicAddF4W(out, tetVertIndices.y, 1.0f);
    }
    if (invMassBary.z > 0.0f)
    {
        pxgAtomicAddF4XYZ(out, tetVertIndices.z, deltaPosition * invMassBary.z);
        pxgAtomicAddF4W(out, tetVertIndices.z, 1.0f);
    }
    if (invMassBary.w > 0.0f)
    {
        pxgAtomicAddF4XYZ(out, tetVertIndices.w, deltaPosition * invMassBary.w);
        pxgAtomicAddF4W(out, tetVertIndices.w, 1.0f);
    }
}

// ===========================================================================
// femAttachmentPrepareLaunch — scatter the descriptor into the warp-strided block.
// ===========================================================================
[shader("compute")]
[numthreads(256, 1, 1)]   // SB_UPDATEROTATION
void femAttachmentPrepareLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint numFemAttachments = gAp.numFemAttachments;
    const uint nbBlocksRequired = (numFemAttachments + SB_BLOCK - 1u) / SB_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + SB_GRID - 1u) / SB_GRID;
    const uint idx = gtid.x;

    for (uint i = 0u; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * SB_BLOCK + idx + nbIterationsPerBlock * gid.x * SB_BLOCK;
        if (workIndex >= numFemAttachments)
            break;   // CUDA `return`; nothing follows the loop here either

        const uint blockIdx32 = workIndex / 32u;
        const uint offset = workIndex & 31u;

        const uint attachmentIndex = Ptr<uint>(gAp.activeFemAttachments)[workIndex];
        const uint64_t att = gAp.femAttachments + uint64_t(attachmentIndex) * FA_STRIDE;
        const uint64_t con = gAp.attachmentConstraints + uint64_t(blockIdx32) * FAC_STRIDE;

        const float4 lowHighAngle = Ptr<float4>(att + FA_LOW_HIGH_ANGLE)[0];
        const float4 barycentric = Ptr<float4>(att + FA_BARYCENTRIC)[0];

        Ptr<float4>(con + FAC_BARYCENTRIC0 + uint64_t(offset) * 16UL)[0] =
            Ptr<float4>(att + FA_BARY0)[0];
        Ptr<float4>(con + FAC_BARYCENTRIC1 + uint64_t(offset) * 16UL)[0] =
            Ptr<float4>(att + FA_BARY1)[0];

        // elemId0 is a PxU64 copied whole; elemId1 is a PxU32 WIDENED into a PxU64 slot, so the
        // upper word must be cleared rather than left as whatever was there.
        Ptr<uint2>(con + FAC_ELEMID0 + uint64_t(offset) * 8UL)[0] =
            Ptr<uint2>(att + FA_INDEX0)[0];
        Ptr<uint2>(con + FAC_ELEMID1 + uint64_t(offset) * 8UL)[0] =
            uint2(Ptr<uint>(att + FA_INDEX1)[0], 0u);

        Ptr<float>(con + FAC_CONSTR_OFFSET + uint64_t(offset) * 4UL)[0] =
            Ptr<float>(att + FA_CONSTR_OFFSET)[0];

        // make_float4(low.x, low.y, low.z, 0.f) — the .w of the source is DROPPED.
        Ptr<float4>(con + FAC_LOW_HIGH_ANGLES + uint64_t(offset) * 16UL)[0] =
            float4(lowHighAngle.x, lowHighAngle.y, lowHighAngle.z, 0.0f);
        Ptr<float4>(con + FAC_ATT_BARYCENTRIC + uint64_t(offset) * 16UL)[0] = barycentric;
    }
}

// ===========================================================================
// sb_solveOutputSoftBodyAttachmentDeltaVLaunch — the mass-split position projection.
// ===========================================================================
[[vk::push_constant]] ConstantBuffer<FemAttachSolveParams> gAs;

[shader("compute")]
[numthreads(256, 1, 1)]   // SB_UPDATEROTATION
void sb_solveOutputSoftBodyAttachmentDeltaVLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint numAttachments = gAs.numAttachments;
    const uint nbBlocksRequired = (numAttachments + SB_BLOCK - 1u) / SB_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + SB_GRID - 1u) / SB_GRID;
    const uint idx = gtid.x;

    Ptr<PxgSoftBodyGpu> softbodies = Ptr<PxgSoftBodyGpu>(gAs.softbodies);

    for (uint i = 0u; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * SB_BLOCK + idx + nbIterationsPerBlock * gid.x * SB_BLOCK;
        if (workIndex >= numAttachments)
            break;

        const uint blockIdx32 = workIndex / 32u;
        const uint offset = workIndex & 31u;
        const uint64_t con = gAs.attachments + uint64_t(blockIdx32) * FAC_STRIDE;

        // ---- softbody 1: the anchor interpolated in its simulation tetrahedron -------------
        // The CUDA reads a PxU64 slot into a PxU32, so only the low word is meaningful.
        const uint elemId1 = Ptr<uint>(con + FAC_ELEMID1 + uint64_t(offset) * 8UL)[0];
        const uint softBodyId1 = elemId1 >> PXG_SB_ID_SHIFT;
        const uint tetrahedronIdx1 = elemId1 & PXG_SB_ELEM_MASK;
        Ptr<float4> simPos1 = Ptr<float4>(softbodies[softBodyId1].mSimPosition_InvMass);
        const uint4 tetInd1 = Ptr<uint4>(softbodies[softBodyId1].mSimTetIndices)[tetrahedronIdx1];
        const float4 bary1 = Ptr<float4>(con + FAC_BARYCENTRIC1 + uint64_t(offset) * 16UL)[0];

        const float4 p10 = simPos1[tetInd1.x];
        const float4 p11 = simPos1[tetInd1.y];
        const float4 p12 = simPos1[tetInd1.z];
        const float4 p13 = simPos1[tetInd1.w];
        const float4 pos1 = p10 * bary1.x + p11 * bary1.y + p12 * bary1.z + p13 * bary1.w;

        // ---- softbody 0 -------------------------------------------------------------------
        const uint elemInd0 = Ptr<uint>(con + FAC_ELEMID0 + uint64_t(offset) * 8UL)[0];
        const uint softBodyId0 = elemInd0 >> PXG_SB_ID_SHIFT;
        const uint tetrahedronIdx0 = elemInd0 & PXG_SB_ELEM_MASK;
        const float4 bary0 = Ptr<float4>(con + FAC_BARYCENTRIC0 + uint64_t(offset) * 16UL)[0];
        Ptr<float4> simPos0 = Ptr<float4>(softbodies[softBodyId0].mSimPosition_InvMass);
        const uint4 tetInd0 = Ptr<uint4>(softbodies[softBodyId0].mSimTetIndices)[tetrahedronIdx0];

        const float4 p00 = simPos0[tetInd0.x];
        const float4 p01 = simPos0[tetInd0.y];
        const float4 p02 = simPos0[tetInd0.z];
        const float4 p03 = simPos0[tetInd0.w];
        const float4 pos0 = p00 * bary0.x + p01 * bary0.y + p02 * bary0.z + p03 * bary0.w;

        const uint64_t accumDeltaV1 = softbodies[softBodyId1].mSimDelta;
        const uint64_t accumDeltaV0 = softbodies[softBodyId0].mSimDelta;

        const float4 relPos1 = pos1 - pos0;

        const float4 lowHighAngle = Ptr<float4>(con + FAC_LOW_HIGH_ANGLES + uint64_t(offset) * 16UL)[0];

        float3 error;
        if (isConeLimitedEnabled(lowHighAngle.z, lowHighAngle.x, lowHighAngle.y))
        {
            const float4 attachmentBary =
                Ptr<float4>(con + FAC_ATT_BARYCENTRIC + uint64_t(offset) * 16UL)[0];
            const float constraintOffset =
                Ptr<float>(con + FAC_CONSTR_OFFSET + uint64_t(offset) * 4UL)[0];
            const float4 projectP = p00 * attachmentBary.x + p01 * attachmentBary.y +
                                    p02 * attachmentBary.z + p03 * attachmentBary.w;
            const float4 axis = projectP - pos0;
            const float3 worldAxis = faGetNormalized(axis.xyz);

            error = computeConeLimitedError(lowHighAngle.z, lowHighAngle.x, lowHighAngle.y,
                                            worldAxis, relPos1.xyz - worldAxis * constraintOffset);
        }
        else
        {
            error = relPos1.xyz;
        }

        // The barycentric-weighted inverse masses.  b*w[i] = bary[i] * invMass[i], and
        // wTot = sum_i bary[i] * b*w[i] = sum_i bary[i]^2 * invMass[i].
        const float4 b0w = float4(bary0.x * p00.w, bary0.y * p01.w, bary0.z * p02.w, bary0.w * p03.w);
        const float4 b1w = float4(bary1.x * p10.w, bary1.y * p11.w, bary1.z * p12.w, bary1.w * p13.w);

        const float wTot0 = b0w.x * bary0.x + b0w.y * bary0.y + b0w.z * bary0.z + b0w.w * bary0.w;
        const float wTot1 = b1w.x * bary1.x + b1w.y * bary1.y + b1w.z * bary1.z + b1w.w * bary1.w;
        const float wTot = wTot0 + wTot1;

        if (wTot > 0.0f)
        {
            const float3 delta = error * (1.0f / wTot);
            updateTetPositionDelta(accumDeltaV1, tetInd1, -delta, b1w);
            updateTetPositionDelta(accumDeltaV0, tetInd0, delta, b0w);
        }
    }
}
