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

// femClothRigidAttachBda — P7.1 tranche 3a: the FEM-cloth RIGID-ATTACHMENT
// constraint kernels on the Vulkan engine ABI (BDA).  The machinery that PINS a
// cloth vertex to a rigid / world anchor (toward a hanging / draping cloth),
// ported line-for-line from
//   gpusimulationcontroller/src/CUDA/FEMClothConstraintPrep.cu (the prepare)
//   gpusimulationcontroller/src/CUDA/FEMClothExternalSolve.cu   (the solves)
// with the attachment-impulse leaf from src/CUDA/attachments.cuh:
//   - cloth_rigidAttachmentPrepareLaunch (FEMClothConstraintPrep.cu:387, launch
//     PxgFEMClothCore.cpp:226): builds the block-of-32 PxgFEMRigidAttachment-
//     Constraint from each PxgFEMRigidAttachment — resolves the cloth attach point,
//     the rigid anchor frame (ra = body2World.rotate(localPose0)), the three
//     axis responses (raXn x sqrtInvInertia) and the per-axis velocity multipliers.
//   - cloth_solveRigidClothAttachmentLaunch (PGS) / ...TGSLaunch (TGS)
//     (FEMClothExternalSolve.cu:1371 / :1571, launch PxgFEMClothCore.cpp:1850):
//     read the rigid anchor velocity, compute the attachment delta impulse
//     (calculateAttachmentDeltaImpulsePGS/TGS, attachments.cuh:83 / :147) and
//     scatter the cloth-side position correction into mDeltaPos.
//
// SCOPE — rigid / WORLD anchor (the hanging-cloth pin).  For a STATIC / world
// anchor the rigid side has zero velocity + infinite mass, so the reaction path
// collapses: the same precedent as the particle-vs-static-floor tranches, we DEFER
// the dynamic-rigid reaction (see the "DEVIATIONS" list).  The cloth-side pull is
// exact (bit-exact vs the CPU transcription in testFemClothSim.cpp).
//
// DEVIATIONS from the CUDA original (recipe policy; each preserves the constraint
// semantics for a rigid / world anchor and is exercised by testFemClothSim.cpp):
//   - the descriptors arrive as device pointers read through Ptr<> (BDA), reusing
//     the byte-exact pxgSlang* mirrors (FEMClothDesc / PrepDesc / SolverDesc /
//     Common).  The block-of-32 constraint and the 96-byte PxgFEMRigidAttachment
//     are addressed by explicit byte offset through Ptr<> (no descriptor block
//     copy), exactly as the other BDA kernels read their SoA blocks;
//   - the ARTICULATION anchor branch is DEFERRED (a rigid / world anchor is never
//     an articulation: rigidId.isArticulation()==false).  Both the prepare's
//     articulation constraint build (spatialResponseMatrixW path) and the solve's
//     articulation link-velocity read are omitted — same precedent as
//     psContactSolveTgsBda, which sinks the outArtiVelocity read behind the
//     isArticulation guard so a rigid / static contact never dereferences it;
//   - the DYNAMIC-rigid REACTION is DEFERRED.  For a static anchor the CUDA guards
//     the rigidDeltaVel writeback behind !rigidId.isStaticBody() (never taken), so
//     nothing is lost; for a dynamic anchor the reaction accumulation is a
//     follow-on tranche.  The prepare still zeroes rigidDeltaVel[workIndex] and
//     [workIndex+numRigidAttachments] when the buffer is bound (the CUDA does this
//     unconditionally in the rigid branch), and the solve leaves the buffer
//     untouched for a static anchor (matching the guarded CUDA);
//   - the cone-LIMITED attachment path (isConeLimitedEnabled ->
//     computeConeLimitedError, a PxSinCos transcendental seam, attachments.cuh:48)
//     is DEFERRED.  For a plain pin (worldAxis_angle.w < 0 && low/high limits < 0,
//     the world-anchor case) isConeLimitedEnabled() is false and the branch is
//     skipped verbatim, so the plain-attachment velError / tgsError below is the
//     exact CUDA false-path;
//   - cloth.mDeltaPos accumulation: the CUDA AtomicAdd(float4&, PxVec3, 1.f)
//     (atomic.cuh:46) is emitted as a plain load-add-store.  Attachments are
//     solved one work-item per constraint; for the single-attachment unit test
//     (and any attachment set with no two constraints sharing a cloth vertex) there
//     is no cross-thread contention, so the read-modify-write is bit-identical to
//     the atomic.  The general multi-attachment averaging is a follow-on scene
//     tranche — same precedent as the particle one-way solve, whose per-contact
//     deltaP scatter was replaced by a contention-free gather;
//   - getNormalized()'s PxRecipSqrt lowers to `v * (1.0/sqrt(m))` (the unix
//     intrinsic recipSqrt is literally 1.0f/::sqrtf, PxUnixMathIntrinsics.h:93) —
//     IEEE-correctly-rounded, bit-exact across backends; the CPU reference uses the
//     identical expression, so worldAxis (dead in the plain-attachment solve) is
//     bit-exact too.
//
// All float arithmetic is NoContraction-stamped (PXG_NOCONTRACT_SHADERS): the raXn
// cross / sqrtInvInertia dot responses feed the compared velMultiplier, and the
// velError / tgsError sums feed the compared deltaImpulse — either would fuse to
// FMAs otherwise, breaking bitwise parity with the -ffp-contract=off reference.

import pxgSlangFEMClothDesc;   // PxgFEMClothGpu
import pxgSlangPrepDesc;       // PxgPrePrepDescGpu, PxgConstraintPrepareDescGpu
import pxgSlangSolverDesc;     // PxgSolverCoreDescGpu, PxgSolverSharedDescPGSGpu, PxgSolverSharedDescTGSGpu
import pxgSlangCommon;         // PxgSolverBodyData, PxgSolverTxIData, PxgAlignedTransform

// PxgSoftBodyKernelBlockDim/GridDim::SB_UPDATEROTATION (PxgSoftBodyCoreKernelIndices.h).
static const uint RA_BLOCK = 256u;
static const uint RA_GRID = 1024u;

// FEMCLOTH_BIAS_COEFFICIENT (FEMClothExternalSolve.cu:53).
static const float FEMCLOTH_BIAS_COEFFICIENT = 0.7f;

// PX_MAX_NB_DEFORMABLE_SURFACE_TRI (PxDeformableSurface.h) / PXG_BITSHIFT_ELEMENT_ID
// (PxgSimulationCoreDesc.h:41) — the cloth-id / element-index packing.
static const uint PXG_CLOTH_ELEMENT_MASK = 0x000fffffu;
static const uint PXG_CLOTH_ID_SHIFT = 20u;

// ---------------------------------------------------------------------------
// Byte layout of the block-of-32 PxgFEMRigidAttachmentConstraint
// (PxgFEMCore.h:191).  32 constraints per block (SoA), indexed [offset].
// ---------------------------------------------------------------------------
static const uint RAC_baryOrType      = 0u;      // float4[32]
static const uint RAC_raXn0_biasW     = 512u;    // float4[32]
static const uint RAC_raXn1_biasW     = 1024u;   // float4[32]
static const uint RAC_raXn2_biasW     = 1536u;   // float4[32]
static const uint RAC_velMult_invMass = 2048u;   // float4[32]
static const uint RAC_low_high_limits = 2560u;   // float4[32]
static const uint RAC_axis_angle      = 3072u;   // float4[32]
static const uint RAC_elemId          = 3584u;   // PxU32[32]
static const uint RAC_rigidId         = 3712u;   // PxU64[32]
static const uint RAC_STRIDE          = 3968u;

// Byte layout of PxgFEMRigidAttachment (PxgSimulationCoreDesc.h:208, extends
// PxgRigidFilterPair @0).  96 bytes.
static const uint RA_index0     = 0u;    // PxU64  (rigid node index)
static const uint RA_index1     = 8u;    // PxU64  (elemId in low 32 bits)
static const uint RA_localPose0 = 32u;   // float4 (anchor local pose = ra4)
static const uint RA_lowHigh    = 48u;   // float4 (coneLimitParams.low_high_limits)
static const uint RA_axisAngle  = 64u;   // float4 (coneLimitParams.axis_angle)
static const uint RA_baryOrType1 = 80u;  // float4
static const uint RA_STRIDE_ATT = 96u;

// PxGetIsVertexType (PxgSimulationCoreDesc.h:145): all four components exactly 0.
bool clothIsVertexType(float4 baryOrType)
{
    return baryOrType.x == 0.0f && baryOrType.y == 0.0f && baryOrType.z == 0.0f && baryOrType.w == 0.0f;
}

// PxQuat::rotate (PxQuat.h:286).
float3 quatRotateVec(float4 q, float3 v)
{
    const float vx = 2.0f * v.x;
    const float vy = 2.0f * v.y;
    const float vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = (q.x * vx + q.y * vy + q.z * vz);
    return float3(vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * dot2,
                  vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * dot2,
                  vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * dot2);
}

// PxVec3::getNormalized (PxVec3.h:286) with recipSqrt = 1.0/sqrt (unix intrinsic).
float3 vecGetNormalized(float3 v)
{
    const float m = v.x * v.x + v.y * v.y + v.z * v.z;
    if (m > 0.0f)
    {
        const float rs = 1.0f / sqrt(m);
        return float3(v.x * rs, v.y * rs, v.z * rs);
    }
    return float3(0.0f, 0.0f, 0.0f);
}

float3 crossVec(float3 a, float3 b)
{
    return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

float dotVec(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// PxMat33 (column-major sqrtInvInertia[9]) * PxVec3.
float3 mat33MulVec(float m[9], float3 v)
{
    return float3(m[0] * v.x + m[3] * v.y + m[6] * v.z,
                  m[1] * v.x + m[4] * v.y + m[7] * v.z,
                  m[2] * v.x + m[5] * v.y + m[8] * v.z);
}

// PxNodeIndex (PxNodeIndex.h:49) packed in the low/high 32 bits of a PxU64.
bool nodeIsStaticBody(uint64_t ind) { return uint(ind & 0xffffffffUL) == 0xffffffffu; }
uint nodeIndex(uint64_t ind) { return uint(ind & 0xffffffffUL); }
bool nodeIsArticulation(uint64_t ind) { return (uint(ind >> 32) & 1u) != 0u; }

// ---------------------------------------------------------------------------
// cloth_rigidAttachmentPrepareLaunch push-constant blob (FEMClothConstraintPrep.cu:387).
// CUDA arg pack (PxgFEMClothCore.cpp:226): { PxgFEMCloth* clothes,
// PxgFEMRigidAttachment* rigidAttachments, PxU32* activeRigidAttachments,
// PxNodeIndex* rigidAttachmentIds, PxU32 numRigidAttachments,
// PxgFEMRigidAttachmentConstraint* attachmentConstraints, PxgPrePrepDesc* preDesc,
// PxgConstraintPrepareDesc* prepareDesc, PxgSolverSharedDescBase* sharedDesc,
// float4* rigidDeltaVel } -> natural-alignment blob = 80 bytes.
// (ABI row {8,8,8,8,4,8,8,8,8,8}.)
struct PrepAttachParams
{
    uint64_t clothes;                 // @0
    uint64_t rigidAttachments;        // @8
    uint64_t activeRigidAttachments;  // @16
    uint64_t rigidAttachmentIds;      // @24
    uint numRigidAttachments;         // @32
    uint _pad36;                      // @36
    uint64_t constraints;             // @40
    uint64_t prePrepDesc;             // @48
    uint64_t prepDesc;                // @56
    uint64_t sharedDesc;              // @64
    uint64_t rigidDeltaVel;           // @72  -> 80
}

// cloth_solveRigidClothAttachment{,TGS}Launch push-constant blob
// (FEMClothExternalSolve.cu:1371 / :1571).  CUDA arg pack (PxgFEMClothCore.cpp:1858):
// { PxgFEMCloth* clothes, PxgFEMRigidAttachmentConstraint* attachments,
// PxU32 numAttachments, PxgPrePrepDesc* prePrepDesc, PxgSolverCoreDesc* solverCoreDesc,
// PxgArticulationCoreDesc* artiCoreDesc, PxgSolverSharedDesc* sharedDesc, PxReal dt,
// float4* rigidDeltaVel } -> natural-alignment blob = 72 bytes.
// (ABI row {8,8,4,8,8,8,8,4,8}.)
struct SolveAttachParams
{
    uint64_t clothes;                 // @0
    uint64_t attachments;             // @8
    uint numAttachments;              // @16
    uint _pad20;                      // @20
    uint64_t prePrepDesc;             // @24
    uint64_t solverCoreDesc;          // @32
    uint64_t artiCoreDesc;            // @40
    uint64_t sharedDesc;              // @48
    float dt;                         // @56
    uint _pad60;                      // @60
    uint64_t rigidDeltaVel;           // @64  -> 72
}

// cloth_applyExternalDeltasLaunch push-constant blob (FEMClothExternalSolve.cu:1053).
// CUDA arg pack (PxgFEMClothCore.cpp:2113): { PxgFEMCloth* femClothes,
// const PxU32* activeClothes, PxReal dt } -> natural-alignment blob = 24 bytes
// (8+8+4 rounded to 8-alignment).  (ABI row {8,8,4}.)
struct ApplyExtDeltasParams
{
    uint64_t clothes;                 // @0
    uint64_t activeClothes;           // @8
    float dt;                         // @16
    uint _pad20;                      // @20  -> 24
}

[[vk::push_constant]] ConstantBuffer<PrepAttachParams> gPrep;
[[vk::push_constant]] ConstantBuffer<SolveAttachParams> gSolve;
[[vk::push_constant]] ConstantBuffer<ApplyExtDeltasParams> gApply;

// ===========================================================================
// cloth_rigidAttachmentPrepareLaunch (FEMClothConstraintPrep.cu:387)
// ===========================================================================
[shader("compute")]
[numthreads(256, 1, 1)] // SB_UPDATEROTATION
void cloth_rigidAttachmentPrepareLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint numRigidAttachments = gPrep.numRigidAttachments;
    const uint nbBlocksRequired = (numRigidAttachments + RA_BLOCK - 1u) / RA_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + RA_GRID - 1u) / RA_GRID;
    const uint idx = groupThreadId.x;

    Ptr<PxgFEMClothGpu> clothes = Ptr<PxgFEMClothGpu>(gPrep.clothes);
    Ptr<uint> activeRigidAttachments = Ptr<uint>(gPrep.activeRigidAttachments);
    Ptr<uint64_t> rigidAttachmentIds = Ptr<uint64_t>(gPrep.rigidAttachmentIds);
    Ptr<PxgConstraintPrepareDescGpu> prepDesc = Ptr<PxgConstraintPrepareDescGpu>(gPrep.prepDesc);

    // body2WorldPool / solverBodyDataPool / solverBodyTxIDataPool bases.
    const uint64_t bodyFramesBase = prepDesc[0].body2WorldPool;
    const uint64_t solverBodyDataBase = prepDesc[0].solverBodyDataPool;
    const uint64_t solverTxIDataBase = prepDesc[0].solverBodyTxIDataPool;

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * RA_BLOCK + idx + nbIterationsPerBlock * groupId.x * RA_BLOCK;
        if (workIndex >= numRigidAttachments)
            return;

        const uint index = workIndex / 32u;
        const uint offset = workIndex & 31u;

        const uint attachmentId = activeRigidAttachments[workIndex];
        const uint64_t attBase = gPrep.rigidAttachments + uint64_t(attachmentId) * RA_STRIDE_ATT;
        const uint64_t cBase = gPrep.constraints + uint64_t(index) * RAC_STRIDE;

        const uint64_t attIndex0 = Ptr<uint64_t>(attBase + RA_index0)[0];
        const uint64_t attIndex1 = Ptr<uint64_t>(attBase + RA_index1)[0];
        const float4 localPose0 = Ptr<float4>(attBase + RA_localPose0)[0];
        const float4 low_high_limits = Ptr<float4>(attBase + RA_lowHigh)[0];
        const float4 axis_angle = Ptr<float4>(attBase + RA_axisAngle)[0];
        const float4 baryOrType1 = Ptr<float4>(attBase + RA_baryOrType1)[0];

        const uint elemId = uint(attIndex1);
        const uint clothId = elemId >> PXG_CLOTH_ID_SHIFT;
        const uint elemIdx = elemId & PXG_CLOTH_ELEMENT_MASK;
        const bool elemIsVertex = clothIsVertexType(baryOrType1);

        Ptr<float4> pos_invMass = Ptr<float4>(clothes[clothId].mPosition_InvMass);

        float4 attachmentPose;
        if (elemIsVertex)
        {
            attachmentPose = pos_invMass[elemIdx];
        }
        else
        {
            const float4 barycentric = baryOrType1;
            const uint4 triVertInd = Ptr<uint4>(clothes[clothId].mTriangleVertexIndices)[elemIdx];
            const float4 p0 = pos_invMass[triVertInd.x];
            const float4 p1 = pos_invMass[triVertInd.y];
            const float4 p2 = pos_invMass[triVertInd.z];
            attachmentPose = p0 * barycentric.x + p1 * barycentric.y + p2 * barycentric.z;
        }

        const float invMass1 = attachmentPose.w;
        const float3 point = float3(attachmentPose.x, attachmentPose.y, attachmentPose.z);
        const float3 axis = float3(axis_angle.x, axis_angle.y, axis_angle.z);
        const float3 ra4 = float3(localPose0.x, localPose0.y, localPose0.z);

        const uint64_t rigidId = attIndex0;
        const bool isStatic = nodeIsStaticBody(rigidId);

        rigidAttachmentIds[workIndex] = rigidId;

        const float3 normal0 = float3(1.0f, 0.0f, 0.0f);
        const float3 normal1 = float3(0.0f, 1.0f, 0.0f);
        const float3 normal2 = float3(0.0f, 0.0f, 1.0f);

        // ARTICULATION anchor branch DEFERRED (see header): a rigid / world anchor
        // is never an articulation, so the else (rigid / static) path is taken.
        // solverBodyIdx is 0 for a static body (solverBodyIndices untouched).
        uint solverBodyIdx = 0u;
        if (!isStatic)
            solverBodyIdx = Ptr<uint>(Ptr<PxgPrePrepDescGpu>(gPrep.prePrepDesc)[0].solverBodyIndices)[nodeIndex(rigidId)];

        const float4 linVel_invMass0 =
            Ptr<PxgSolverBodyData>(solverBodyDataBase)[solverBodyIdx].initialLinVelXYZ_invMassW;
        const float invMass0 = linVel_invMass0.w;

        float invSqrtInertia[9];
        float inertiaScale = 1.0f;
        if (invMass0 == 0.0f && !isStatic)
        {
            // PxMat33(PxIdentity), column-major.
            invSqrtInertia[0] = 1.0f; invSqrtInertia[1] = 0.0f; invSqrtInertia[2] = 0.0f;
            invSqrtInertia[3] = 0.0f; invSqrtInertia[4] = 1.0f; invSqrtInertia[5] = 0.0f;
            invSqrtInertia[6] = 0.0f; invSqrtInertia[7] = 0.0f; invSqrtInertia[8] = 1.0f;
            inertiaScale = 0.0f;
        }
        else
        {
            PxgSolverTxIData txI = Ptr<PxgSolverTxIData>(solverTxIDataBase)[solverBodyIdx];
            for (uint k = 0; k < 9u; ++k)
                invSqrtInertia[k] = txI.sqrtInvInertia[k];
        }

        const float4 bfQ = Ptr<float4>(bodyFramesBase + uint64_t(solverBodyIdx) * 32u + 0u)[0];
        const float4 bfP = Ptr<float4>(bodyFramesBase + uint64_t(solverBodyIdx) * 32u + 16u)[0];
        const float3 bodyFrame0p = float3(bfP.x, bfP.y, bfP.z);

        const float3 ra = quatRotateVec(bfQ, ra4);
        const float3 error = float3(ra.x + bodyFrame0p.x - point.x,
                                    ra.y + bodyFrame0p.y - point.y,
                                    ra.z + bodyFrame0p.z - point.z);
        const float3 worldAxis = vecGetNormalized(quatRotateVec(bfQ, axis));

        const float3 raXn0 = crossVec(ra, normal0);
        const float3 raXn1 = crossVec(ra, normal1);
        const float3 raXn2 = crossVec(ra, normal2);

        const float3 raXnSqrtInertia0 = mat33MulVec(invSqrtInertia, raXn0);
        const float3 raXnSqrtInertia1 = mat33MulVec(invSqrtInertia, raXn1);
        const float3 raXnSqrtInertia2 = mat33MulVec(invSqrtInertia, raXn2);

        const float resp0 = dotVec(raXnSqrtInertia0, raXnSqrtInertia0) * inertiaScale + invMass0 + invMass1;
        const float resp1 = dotVec(raXnSqrtInertia1, raXnSqrtInertia1) * inertiaScale + invMass0 + invMass1;
        const float resp2 = dotVec(raXnSqrtInertia2, raXnSqrtInertia2) * inertiaScale + invMass0 + invMass1;

        const float velMultiplier0 = (resp0 > 0.0f) ? (1.0f / resp0) : 0.0f;
        const float velMultiplier1 = (resp1 > 0.0f) ? (1.0f / resp1) : 0.0f;
        const float velMultiplier2 = (resp2 > 0.0f) ? (1.0f / resp2) : 0.0f;

        const float biasedErr0 = dotVec(error, normal0);
        const float biasedErr1 = dotVec(error, normal1);
        const float biasedErr2 = dotVec(error, normal2);

        Ptr<float4>(cBase + RAC_raXn0_biasW)[offset] = float4(raXnSqrtInertia0.x, raXnSqrtInertia0.y, raXnSqrtInertia0.z, biasedErr0);
        Ptr<float4>(cBase + RAC_raXn1_biasW)[offset] = float4(raXnSqrtInertia1.x, raXnSqrtInertia1.y, raXnSqrtInertia1.z, biasedErr1);
        Ptr<float4>(cBase + RAC_raXn2_biasW)[offset] = float4(raXnSqrtInertia2.x, raXnSqrtInertia2.y, raXnSqrtInertia2.z, biasedErr2);
        Ptr<float4>(cBase + RAC_velMult_invMass)[offset] = float4(velMultiplier0, velMultiplier1, velMultiplier2, invMass0);
        Ptr<uint>(cBase + RAC_elemId)[offset] = elemId;
        Ptr<uint64_t>(cBase + RAC_rigidId)[offset] = rigidId;
        Ptr<float4>(cBase + RAC_baryOrType)[offset] = baryOrType1;
        Ptr<float4>(cBase + RAC_low_high_limits)[offset] = low_high_limits;
        Ptr<float4>(cBase + RAC_axis_angle)[offset] = float4(worldAxis.x, worldAxis.y, worldAxis.z, axis_angle.w);

        // DYNAMIC-rigid reaction DEFERRED: the CUDA zeroes rigidDeltaVel here in the
        // rigid branch when the buffer is bound; the accumulation itself is a
        // follow-on tranche.  For a static anchor this write is inert.
        if (gPrep.rigidDeltaVel != 0UL)
        {
            Ptr<float4> rigidDeltaVel = Ptr<float4>(gPrep.rigidDeltaVel);
            rigidDeltaVel[workIndex] = float4(0.0f, 0.0f, 0.0f, 0.0f);
            rigidDeltaVel[workIndex + numRigidAttachments] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
}

// ---------------------------------------------------------------------------
// The cloth-side attachment scatter (both solves): the CUDA AtomicAdd emitted as a
// plain read-modify-write (see header).  Vertex attachment only (the world-anchor
// pin); the triangle-barycentric scatter follows the same shape.
// ---------------------------------------------------------------------------
void applyClothDelta(uint64_t deltaPosBase, uint elemIdx, float3 deltaPos, float w)
{
    Ptr<float4> deltaPosBuf = Ptr<float4>(deltaPosBase);
    float4 dp = deltaPosBuf[elemIdx];
    dp.x += deltaPos.x * w;
    dp.y += deltaPos.y * w;
    dp.z += deltaPos.z * w;
    dp.w += 1.0f;
    deltaPosBuf[elemIdx] = dp;
}

// ===========================================================================
// cloth_solveRigidClothAttachmentLaunch — PGS (FEMClothExternalSolve.cu:1371)
// ===========================================================================
[shader("compute")]
[numthreads(256, 1, 1)] // SB_UPDATEROTATION
void cloth_solveRigidClothAttachmentLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint numAttachments = gSolve.numAttachments;
    const float dt = gSolve.dt;
    const float invDt = 1.0f / dt;

    Ptr<PxgFEMClothGpu> clothes = Ptr<PxgFEMClothGpu>(gSolve.clothes);
    Ptr<PxgSolverCoreDescGpu> solverCoreDesc = Ptr<PxgSolverCoreDescGpu>(gSolve.solverCoreDesc);
    Ptr<PxgSolverSharedDescPGSGpu> sharedDesc = Ptr<PxgSolverSharedDescPGSGpu>(gSolve.sharedDesc);
    Ptr<PxgPrePrepDescGpu> prePrepDesc = Ptr<PxgPrePrepDescGpu>(gSolve.prePrepDesc);

    const uint numSolverBodies = solverCoreDesc[0].numSolverBodies;
    const uint accOffset = solverCoreDesc[0].accumulatedBodyDeltaVOffset;
    Ptr<float4> initialVel = Ptr<float4>(solverCoreDesc[0].outSolverVelocity);
    Ptr<float4> solverBodyVelPool = Ptr<float4>(sharedDesc[0].iter_solverBodyVelPool);

    const uint nbBlocksRequired = (numAttachments + RA_BLOCK - 1u) / RA_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + RA_GRID - 1u) / RA_GRID;
    const uint idx = groupThreadId.x;

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * RA_BLOCK + idx + nbIterationsPerBlock * groupId.x * RA_BLOCK;
        if (workIndex >= numAttachments)
            return;

        const uint index = workIndex / 32u;
        const uint offset = workIndex & 31u;
        const uint64_t cBase = gSolve.attachments + uint64_t(index) * RAC_STRIDE;

        const uint elemId = Ptr<uint>(cBase + RAC_elemId)[offset];
        const uint clothId = elemId >> PXG_CLOTH_ID_SHIFT;
        const uint elemIdx = elemId & PXG_CLOTH_ELEMENT_MASK;

        Ptr<float4> clothVelocities = Ptr<float4>(clothes[clothId].mVelocity_InvMass);

        // VERTEX attachment (world-anchor pin; constraint.baryOrType == 0).  The
        // triangle-barycentric read is the same shape (deferred — see header).
        const float4 clothVel = clothVelocities[elemIdx];
        const float3 linVel1 = float3(clothVel.x, clothVel.y, clothVel.z);

        const uint64_t rigidId = Ptr<uint64_t>(cBase + RAC_rigidId)[offset];
        const bool isStatic = nodeIsStaticBody(rigidId);
        if (isStatic && clothVel.w == 0.0f)
            continue;

        // PxgVelocityReader::readVelocitiesPGS (dataReadWriteHelper.cuh:101).
        // ARTICULATION branch DEFERRED (see header).
        uint solverBodyIdx = 0u;
        if (!isStatic)
            solverBodyIdx = Ptr<uint>(prePrepDesc[0].solverBodyIndices)[nodeIndex(rigidId)];

        const float4 iv0 = initialVel[solverBodyIdx];
        const float4 iv1 = initialVel[solverBodyIdx + numSolverBodies];
        const float4 dv0 = solverBodyVelPool[accOffset + solverBodyIdx];
        const float4 dv1 = solverBodyVelPool[accOffset + solverBodyIdx + numSolverBodies];
        const float3 rigidLinVel = float3(iv0.x + dv0.x, iv0.y + dv0.y, iv0.z + dv0.z);
        const float3 rigidAngVel = float3(iv1.x + dv1.x, iv1.y + dv1.y, iv1.z + dv1.z);

        // calculateAttachmentDeltaImpulsePGS (attachments.cuh:83).  NOTE the CUDA
        // solve passes constraint.raXn0_biasW THREE TIMES (FEMClothExternalSolve.cu:
        // 1451) — a faithful port of that upstream quirk, so raXn0/1/2 AND all three
        // position-error components come from raXn0_biasW.
        const float4 raXn0_biasW = Ptr<float4>(cBase + RAC_raXn0_biasW)[offset];
        const float4 velMult = Ptr<float4>(cBase + RAC_velMult_invMass)[offset];

        const float3 raXn0 = float3(raXn0_biasW.x, raXn0_biasW.y, raXn0_biasW.z);
        const float velOfRigidX = rigidLinVel.x + dotVec(rigidAngVel, raXn0);
        const float velOfRigidY = rigidLinVel.y + dotVec(rigidAngVel, raXn0);
        const float velOfRigidZ = rigidLinVel.z + dotVec(rigidAngVel, raXn0);

        const float positionError = raXn0_biasW.w;
        const float biasFactor = 0.5f;
        // Cone-LIMITED branch DEFERRED (see header): for a plain pin the guard is
        // false and velError is the CUDA false-path.
        const float3 velError = float3(linVel1.x - positionError * biasFactor * invDt,
                                       linVel1.y - positionError * biasFactor * invDt,
                                       linVel1.z - positionError * biasFactor * invDt);

        const float deltaF0 = (velError.x - velOfRigidX) * velMult.x;
        const float deltaF1 = (velError.y - velOfRigidY) * velMult.y;
        const float deltaF2 = (velError.z - velOfRigidZ) * velMult.z;
        const float3 deltaImpulse = float3(deltaF0, deltaF1, deltaF2);

        // DYNAMIC-rigid reaction DEFERRED: static anchor never writes rigidDeltaVel.

        if (deltaImpulse.x != 0.0f || deltaImpulse.y != 0.0f || deltaImpulse.z != 0.0f)
        {
            const float3 deltaPos = float3(-deltaImpulse.x * dt, -deltaImpulse.y * dt, -deltaImpulse.z * dt);
            applyClothDelta(clothes[clothId].mDeltaPos, elemIdx, deltaPos, clothVel.w);
        }
    }
}

// ===========================================================================
// cloth_solveRigidClothAttachmentTGSLaunch — TGS (FEMClothExternalSolve.cu:1571)
// ===========================================================================
[shader("compute")]
[numthreads(256, 1, 1)] // SB_UPDATEROTATION
void cloth_solveRigidClothAttachmentTGSLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint numAttachments = gSolve.numAttachments;
    const float dt = gSolve.dt;

    Ptr<PxgFEMClothGpu> clothes = Ptr<PxgFEMClothGpu>(gSolve.clothes);
    Ptr<PxgSolverCoreDescGpu> solverCoreDesc = Ptr<PxgSolverCoreDescGpu>(gSolve.solverCoreDesc);
    Ptr<PxgSolverSharedDescTGSGpu> sharedDesc = Ptr<PxgSolverSharedDescTGSGpu>(gSolve.sharedDesc);
    Ptr<PxgPrePrepDescGpu> prePrepDesc = Ptr<PxgPrePrepDescGpu>(gSolve.prePrepDesc);

    const uint numSolverBodies = solverCoreDesc[0].numSolverBodies;
    const uint accOffset = solverCoreDesc[0].accumulatedBodyDeltaVOffset;
    Ptr<float4> solverBodyVelPool = Ptr<float4>(sharedDesc[0].iter_solverBodyVelPool);

    const uint nbBlocksRequired = (numAttachments + RA_BLOCK - 1u) / RA_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + RA_GRID - 1u) / RA_GRID;
    const uint idx = groupThreadId.x;

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * RA_BLOCK + idx + nbIterationsPerBlock * groupId.x * RA_BLOCK;
        if (workIndex >= numAttachments)
            return;

        const uint index = workIndex / 32u;
        const uint offset = workIndex & 31u;
        const uint64_t cBase = gSolve.attachments + uint64_t(index) * RAC_STRIDE;

        const uint elemId = Ptr<uint>(cBase + RAC_elemId)[offset];
        const uint clothId = elemId >> PXG_CLOTH_ID_SHIFT;
        const uint elemIdx = elemId & PXG_CLOTH_ELEMENT_MASK;

        Ptr<float4> clothAccumDeltas = Ptr<float4>(clothes[clothId].mAccumulatedDeltaPos);

        // VERTEX attachment (world-anchor pin; constraint.baryOrType == 0).
        const float4 clothAccumDelta = clothAccumDeltas[elemIdx];
        const float3 linDelta1 = float3(clothAccumDelta.x, clothAccumDelta.y, clothAccumDelta.z);

        const uint64_t rigidId = Ptr<uint64_t>(cBase + RAC_rigidId)[offset];
        const bool isStatic = nodeIsStaticBody(rigidId);
        if (isStatic && clothAccumDelta.w == 0.0f)
            continue;

        // PxgVelocityReader::readVelocitiesTGS (dataReadWriteHelper.cuh:122).
        // ARTICULATION branch DEFERRED (see header).
        uint solverBodyIdx = 0u;
        if (!isStatic)
            solverBodyIdx = Ptr<uint>(prePrepDesc[0].solverBodyIndices)[nodeIndex(rigidId)];

        const float4 v0 = solverBodyVelPool[accOffset + solverBodyIdx];
        const float4 v1 = solverBodyVelPool[accOffset + solverBodyIdx + numSolverBodies];
        const float4 v2 = solverBodyVelPool[accOffset + solverBodyIdx + numSolverBodies + numSolverBodies];
        const float3 rigidLinVel = float3(v0.x, v0.y, v0.z);
        const float3 rigidAngVel = float3(v0.w, v1.x, v1.y);
        const float3 rigidLinDelta = float3(v1.z, v1.w, v2.x);
        const float3 rigidAngDelta = float3(v2.y, v2.z, v2.w);

        // calculateAttachmentDeltaImpulseTGS (attachments.cuh:147) — the offset
        // overload reads raXn0/1/2 correctly (distinct rows).
        const float4 raXn0_biasW = Ptr<float4>(cBase + RAC_raXn0_biasW)[offset];
        const float4 raXn1_biasW = Ptr<float4>(cBase + RAC_raXn1_biasW)[offset];
        const float4 raXn2_biasW = Ptr<float4>(cBase + RAC_raXn2_biasW)[offset];
        const float4 velMult = Ptr<float4>(cBase + RAC_velMult_invMass)[offset];

        const float3 raXn0 = float3(raXn0_biasW.x, raXn0_biasW.y, raXn0_biasW.z);
        const float3 raXn1 = float3(raXn1_biasW.x, raXn1_biasW.y, raXn1_biasW.z);
        const float3 raXn2 = float3(raXn2_biasW.x, raXn2_biasW.y, raXn2_biasW.z);

        const float velOfRigid0 = rigidLinVel.x + dotVec(rigidAngVel, raXn0);
        const float velOfRigid1 = rigidLinVel.y + dotVec(rigidAngVel, raXn1);
        const float velOfRigid2 = rigidLinVel.z + dotVec(rigidAngVel, raXn2);

        const float3 linDelta = float3(linDelta1.x - rigidLinDelta.x, linDelta1.y - rigidLinDelta.y, linDelta1.z - rigidLinDelta.z);

        const float positionErrorX = raXn0_biasW.w;
        const float positionErrorY = raXn1_biasW.w;
        const float positionErrorZ = raXn2_biasW.w;

        // Cone-LIMITED branch DEFERRED (see header): plain pin -> guard is false.
        const float3 tgsError = float3(linDelta.x - positionErrorX - dotVec(rigidAngDelta, raXn0),
                                       linDelta.y - positionErrorY - dotVec(rigidAngDelta, raXn1),
                                       linDelta.z - positionErrorZ - dotVec(rigidAngDelta, raXn2));

        const float velDt = dt; // isVelocityIteration == false
        const float deltaF0 = (tgsError.x - velOfRigid0 * velDt) * velMult.x * FEMCLOTH_BIAS_COEFFICIENT;
        const float deltaF1 = (tgsError.y - velOfRigid1 * velDt) * velMult.y * FEMCLOTH_BIAS_COEFFICIENT;
        const float deltaF2 = (tgsError.z - velOfRigid2 * velDt) * velMult.z * FEMCLOTH_BIAS_COEFFICIENT;
        const float3 deltaImpulse = float3(deltaF0, deltaF1, deltaF2);

        // DYNAMIC-rigid reaction DEFERRED: static anchor never writes rigidDeltaVel.

        if (deltaImpulse.x != 0.0f || deltaImpulse.y != 0.0f || deltaImpulse.z != 0.0f)
        {
            const float3 deltaPos = float3(-deltaImpulse.x, -deltaImpulse.y, -deltaImpulse.z);
            applyClothDelta(clothes[clothId].mDeltaPos, elemIdx, deltaPos, clothAccumDelta.w);
        }
    }
}

// ===========================================================================
// cloth_applyExternalDeltasLaunch (FEMClothExternalSolve.cu:1053) — the per-vertex
// pass that folds the accumulated external correction (mDeltaPos, written by the
// attachment / contact solves; .w = the running per-vertex count) into the live
// position/velocity/accumulated-delta state, then clears it.  Without this the
// attachment (and cloth-cloth / contact) corrections are dead-ended, so nothing
// pinned actually moves.  Per-vertex, block dim SB_SOLVETETRA (64), grid
// (numBlocks, nbActiveClothes, 1) — vertIdx = tid + blockIdx.x*64, guard < nbVerts.
// ===========================================================================
[shader("compute")]
[numthreads(64, 1, 1)] // SB_SOLVETETRA
void cloth_applyExternalDeltasLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgFEMClothGpu> femClothes = Ptr<PxgFEMClothGpu>(gApply.clothes);
    const uint id = Ptr<uint>(gApply.activeClothes)[groupId.y];
    const float dt = gApply.dt;

    const uint nbVerts = femClothes[id].mNbVerts;
    const uint vertIdx = groupThreadId.x + groupId.x * 64u;

    if (vertIdx < nbVerts)
    {
        Ptr<float4> mDeltaPos = Ptr<float4>(femClothes[id].mDeltaPos);
        float4 delta = mDeltaPos[vertIdx];

        if (delta.w != 0.0f)
        {
            Ptr<float4> positions = Ptr<float4>(femClothes[id].mPosition_InvMass);
            Ptr<float4> velocities = Ptr<float4>(femClothes[id].mVelocity_InvMass);
            Ptr<float4> accumulatedDeltaPos = Ptr<float4>(femClothes[id].mAccumulatedDeltaPos);

            float4 accumDelta = accumulatedDeltaPos[vertIdx];
            const float invDt = 1.0f / dt;
            float4 pos = positions[vertIdx];
            float4 vel = velocities[vertIdx];
            const float scale = 1.0f / delta.w;

            delta.x *= scale;
            delta.y *= scale;
            delta.z *= scale;
            delta.w = 0.0f;

            vel += delta * invDt;
            pos += delta;
            accumDelta += delta;

            positions[vertIdx] = pos;
            velocities[vertIdx] = vel;
            accumulatedDeltaPos[vertIdx] = accumDelta;

            mDeltaPos[vertIdx] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
}
