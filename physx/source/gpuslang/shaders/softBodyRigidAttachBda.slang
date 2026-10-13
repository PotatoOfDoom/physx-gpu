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

// softBodyRigidAttachBda.slang
//
// P8.3 — the softbody-vs-rigid ATTACHMENT path: pinning a softbody vertex (or a point
// inside a tetrahedron) to a rigid body or to the world.
//
//   sb_rigidAttachmentPrepareLaunch (softBodyGM.cu:1822, launch PxgSoftBodyCore.cpp:2060
//   via prepRigidAttachmentConstraints, SB_RIGID_ATTACHMENT_CONSTRAINT_PREP) — turns each
//   active attachment into a three-axis constraint block: the response terms about the
//   world axes, the position error, and the velocity multipliers.
//
//   sb_solveRigidSoftAttachmentLaunch{,TGS} (softBodyGM.cu:2085 / :2209, launch :1809 /
//   :1866, SB_SOLVE_RIGID_SOFT_ATTACHMENT{,_TGS}) — solves it and drags the attached
//   point back onto its anchor.  The PGS arm is a three-axis VELOCITY constraint; the
//   TGS arm is a three-axis POSITION constraint over accumulated deltas, and the two
//   share only their loop, their element blend and their atomic scatter (P9.76c).
//
// The prepare is a near-clone of cloth_rigidAttachmentPrepareLaunch, argument for
// argument apart from argument 0 (PxgSoftBody* vs PxgFEMCloth*), so its ABI row is the
// cloth row -- 80 bytes -- already proven in the bridge.
//
// THE SOLVES ARE NOT.  P9.76b: cloth builds ONE 9-argument pack and picks the entry point
// with a ternary (PxgFEMClothCore.cpp:1845); softbody builds TWO -- 9 arguments / 72 bytes
// for PGS (PxgSoftBodyCore.cpp:1810) and 11 / 80 for TGS (:1866).  Giving both the cloth
// row left the TGS entry refused on every launch, silently, for as long as it existed.
//
// WHAT ACTUALLY DIFFERS from the cloth pair, in full:
//   - PxgSoftBodyGpu for PxgFEMClothGpu, and the softbody id/element packing;
//   - mSimPosition_InvMass (@96) / mSimVelocity_InvMass (@88) / mSimDeltaPos (@128) /
//     mSimDelta (@272) for the cloth position / velocity / accumulated-delta / delta
//     buffers, and mSimTetIndices (@80) for mTriangleVertexIndices;
//   - the element blend is over FOUR tet vertices, not three triangle vertices;
//   - the inverse mass of a TET attachment is `getSoftBodyInvMass` (softBody.cuh:38):
//     `baryMass * sqrt(bx^2+by^2+bz^2+bw^2)`, NOT the blended `.w` the cloth path uses.
//     That square root is the one genuinely new piece of arithmetic in this tranche, and
//     dropping it silently gives every tet attachment the wrong stiffness.
//
// The rigid-side half of the prepare -- the identity-inertia special case, ra, the error,
// the three responses and velocity multipliers, the constraint writes -- is LINE-FOR-LINE
// what the cloth prepare does, and it is transcribed here rather than shared.  That is
// deliberate: CUDA itself carries the two copies (FEMClothConstraintPrep.cu:387 and
// softBodyGM.cu:1822), so two copies here track upstream, whereas one shared function would
// be a port-introduced divergence.  See pxgSlangFemRigidAttach's header for the rule, and
// for what IS shared: the constraint / attachment byte layouts, which are one ABI and would
// drift silently if copied.
//
// DEFERRED, exactly as in the cloth port, and for the same reasons:
//   - the ARTICULATION anchor branch (softBodyGM.cu:1912).  It reads
//     sharedDesc->mBodySimBufferDeviceData and the articulation link transforms; a rigid or
//     world anchor never takes it;
//   - the CONE-LIMIT branch of the solve.  A plain pin has the guard false;
//   - the DYNAMIC-rigid reaction.  The prepare zeroes rigidDeltaVel (which this port does)
//     and the solve accumulates into it (which it does not, for a static anchor).  A
//     softbody pinned to a MOVING rigid needs that half; pinned to the world it does not.
//
// THE TWO llvmpipe LANDMINES from tranche c are avoided by construction: `rigidDeltaVel` is
// the only optionally-null pointer and it is guarded by a POINTER comparison that never
// feeds a load (the writes are inside the guard, and a store is not speculated), and no
// block redefines the strided loop's bound.
//
// NoContraction-stamped (PXG_NOCONTRACT_SHADERS): the responses are dot-of-cross chains
// through mat33MulVec, the quaternion rotate is a mul/add chain, and the constraint block is
// compared bit-exactly against a CPU transcription.

import pxgSlangCommon;            // PxgSolverBodyData, PxgSolverTxIData
import pxgSlangPrepDesc;          // PxgPrePrepDescGpu, PxgConstraintPrepareDescGpu
import pxgSlangSolverDesc;        // PxgSolverCoreDescGpu, PxgSolverSharedDescPGSGpu
import pxgSlangSoftBodyDesc;      // PxgSoftBodyGpu
import pxgSlangAtomicFloat;       // pxgAtomicAddF4XYZ / pxgAtomicAddF4W (P8.1)
import pxgSlangFemRigidAttach;    // RAC_*/RA_* layouts, femIsVertexType, quatRotateVec
import pxgSlangFemRigidContact;   // crossVec, dotVec, mat33MulVec, nodeIsStaticBody, nodeIndex

// PxgSoftBodyKernelGridDim/BlockDim::SB_UPDATEROTATION — the engine's fixed launch
// (PxgSoftBodyCore.cpp:2068).  RA_BLOCK MUST equal [numthreads] below: the bridge forwards
// only gridDim, so the stride is reconstructed from these constants, not from blockDim.
static const uint RA_BLOCK = 256u;
static const uint RA_GRID = 1024u;

static const uint PX_MAX_NB_DEFORMABLE_VOLUME_TET = 0x000fffffu;
static const uint PXG_BITSHIFT_TET_ID = 20u;

// getSoftBodyInvMass (softBody.cuh:38) — the TET attachment's inverse mass.  The blended
// `.w` is scaled by the LENGTH of the barycentric vector, which the cloth path has no
// counterpart for.
float getSoftBodyInvMass(float baryMass, float4 bary)
{
    const float scale = sqrt(bary.x * bary.x + bary.y * bary.y + bary.z * bary.z + bary.w * bary.w);
    return baryMass * scale;
}

// ---------------------------------------------------------------------------
// sb_rigidAttachmentPrepareLaunch push blob.
// CUDA arg pack (PxgSoftBodyCore.cpp:2056): { PxgSoftBody* softbodies,
// PxgFEMRigidAttachment* rigidAttachments, PxU32* activeRigidAttachments,
// PxNodeIndex* rigidAttachmentIds, PxU32 numRigidAttachments,
// PxgFEMRigidAttachmentConstraint* attachmentConstraints, PxgPrePrepDesc* preDesc,
// PxgConstraintPrepareDesc* prepareDesc, PxgSolverSharedDescBase* sharedDesc,
// float4* rigidDeltaVel } -> 80-byte blob (numRigidAttachments's 4-byte tail-pad realigns
// the @40 pointer).  (ABI row {8,8,8,8,4,8,8,8,8,8} — the cloth row.)
// ---------------------------------------------------------------------------
struct SbPrepAttachParams
{
    uint64_t softbodies;             // @0
    uint64_t rigidAttachments;       // @8
    uint64_t activeRigidAttachments; // @16
    uint64_t rigidAttachmentIds;     // @24  output
    uint numRigidAttachments;        // @32
    uint _pad36;                     // @36
    uint64_t constraints;            // @40  output
    uint64_t prePrepDesc;            // @48
    uint64_t prepareDesc;            // @56
    uint64_t sharedDesc;             // @64  (DEFERRED: articulation-only)
    uint64_t rigidDeltaVel;          // @72  output, OPTIONAL -> 80
}
[[vk::push_constant]]
ConstantBuffer<SbPrepAttachParams> gSbPrepAtt;

// ---------------------------------------------------------------------------
// sb_solveRigidSoftAttachmentLaunch{,TGS} push blob.
// CUDA arg pack (PxgSoftBodyCore.cpp:1810): { PxgSoftBody* softbodies,
// PxgFEMRigidAttachmentConstraint* attachments, PxU32 numAttachments,
// PxgPrePrepDesc* prePrepDesc, PxgSolverCoreDesc* solverCoreDesc,
// PxgArticulationCoreDesc* artiCoreDesc, PxgSolverSharedDesc* sharedDesc, PxReal dt,
// float4* rigidDeltaVel } -> 72-byte blob.  (ABI row {8,8,4,8,8,8,8,4,8} — the cloth row.)
// ---------------------------------------------------------------------------
struct SbSolveAttachParams
{
    uint64_t softbodies;      // @0
    uint64_t attachments;     // @8
    uint numAttachments;      // @16
    uint _pad20;              // @20
    uint64_t prePrepDesc;     // @24
    uint64_t solverCoreDesc;  // @32
    uint64_t artiCoreDesc;    // @40  (DEFERRED)
    uint64_t sharedDesc;      // @48
    float dt;                 // @56
    uint _pad60;              // @60
    uint64_t rigidDeltaVel;   // @64  output -> 72
}
[[vk::push_constant]]
ConstantBuffer<SbSolveAttachParams> gSbSolveAtt;

// ---------------------------------------------------------------------------
// sb_solveRigidSoftAttachmentLaunchTGS push blob -- ITS OWN, AND THAT IS P9.76b.
//
// Both entry points used to share the 72-byte block above, and the ABI table gave
// them one row, under a comment reading "both PGS and TGS share the pack".  That
// is true of CLOTH -- PxgFEMClothCore.cpp:1845 builds one 9-argument pack and
// picks the function with a ternary -- and FALSE of softbody, which has TWO packs:
//   PGS  PxgSoftBodyCore.cpp:1810   9 arguments -> 72 bytes
//   TGS  PxgSoftBodyCore.cpp:1866  11 arguments -> 80 bytes
// The claim was carried over from the cloth row, exactly as P9.76a carried over a
// cloth-only constraint quirk into this same file's solve.
//
// The cost was total and silent: the bridge builds the blob from the HOST pack and
// the shim gates on blobSize == pushConstantSize (PxgVulkanCudaContext.cpp:459), so
// the TGS kernel was refused on EVERY launch and has never run on Vulkan -- while
// every fixture stayed green, because a fixture dispatches the shader directly and
// never passes that gate.
//
// Two arguments the PGS pack does not have, in the HOST's order, which is not the
// order one would guess from reading the kernel: biasCoefficient comes after dt,
// and isVelocityIteration comes after the OUTPUT pointer.
// ---------------------------------------------------------------------------
struct SbSolveAttachParamsTGS
{
    uint64_t softbodies;         // @0
    uint64_t attachments;        // @8
    uint numAttachments;         // @16
    uint _pad20;                 // @20
    uint64_t prePrepDesc;        // @24
    uint64_t solverCoreDesc;     // @32
    uint64_t artiCoreDesc;       // @40  (DEFERRED, as in the PGS block)
    uint64_t sharedDesc;         // @48
    float dt;                    // @56
    float biasCoefficient;       // @60
    uint64_t rigidDeltaVel;      // @64  output
    uint isVelocityIteration;    // @72  bool, 1 byte + 3 tail pad
    uint _pad76;                 // @76 -> 80
}
[[vk::push_constant]]
ConstantBuffer<SbSolveAttachParamsTGS> gSbSolveAttTGS;

// The values the solve needs, so ONE body can serve two differently-sized push
// blocks.  Each entry point fills this from its own block; nothing else in the
// function reads a push constant.  The last two are TGS-only: the PGS entry leaves
// biasCoefficient at 0 and isVelocityIteration at false, and the PGS arm of the
// body reads neither.
struct SbSolveAttArgs
{
    uint numAttachments;
    float dt;
    uint64_t softbodies;
    uint64_t solverCoreDesc;
    uint64_t prePrepDesc;
    uint64_t attachments;
    uint64_t sharedDesc;
    float biasCoefficient;
    bool isVelocityIteration;
}

[shader("compute")]
[numthreads(256, 1, 1)]   // == RA_BLOCK; SB_UPDATEROTATION
void sb_rigidAttachmentPrepareLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint numRigidAttachments = gSbPrepAtt.numRigidAttachments;
    const uint nbBlocksRequired = (numRigidAttachments + RA_BLOCK - 1u) / RA_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + RA_GRID - 1u) / RA_GRID;
    const uint idx = groupThreadId.x;

    Ptr<PxgSoftBodyGpu> softbodies = Ptr<PxgSoftBodyGpu>(gSbPrepAtt.softbodies);
    Ptr<uint> activeRigidAttachments = Ptr<uint>(gSbPrepAtt.activeRigidAttachments);
    Ptr<uint64_t> rigidAttachmentIds = Ptr<uint64_t>(gSbPrepAtt.rigidAttachmentIds);
    Ptr<PxgConstraintPrepareDescGpu> prepDesc = Ptr<PxgConstraintPrepareDescGpu>(gSbPrepAtt.prepareDesc);

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
        const uint64_t attBase = gSbPrepAtt.rigidAttachments + uint64_t(attachmentId) * RA_STRIDE_ATT;
        const uint64_t cBase = gSbPrepAtt.constraints + uint64_t(index) * RAC_STRIDE;

        const uint64_t attIndex0 = Ptr<uint64_t>(attBase + RA_index0)[0];
        const uint64_t attIndex1 = Ptr<uint64_t>(attBase + RA_index1)[0];
        const float4 localPose0 = Ptr<float4>(attBase + RA_localPose0)[0];
        const float4 low_high_limits = Ptr<float4>(attBase + RA_lowHigh)[0];
        const float4 axis_angle = Ptr<float4>(attBase + RA_axisAngle)[0];
        const float4 baryOrType1 = Ptr<float4>(attBase + RA_baryOrType1)[0];

        const uint elemId = uint(attIndex1);
        const uint softBodyId = elemId >> PXG_BITSHIFT_TET_ID;
        const uint elemIdx = elemId & PX_MAX_NB_DEFORMABLE_VOLUME_TET;
        const bool elemIsVertex = femIsVertexType(baryOrType1);

        Ptr<float4> pos_invMassGM = Ptr<float4>(softbodies[softBodyId].mSimPosition_InvMass);

        float4 gridPos;
        float invMass1;
        if (elemIsVertex)
        {
            gridPos = pos_invMassGM[elemIdx];
            invMass1 = gridPos.w;
        }
        else
        {
            const float4 barycentric = baryOrType1;
            const uint4 tetInd = Ptr<uint4>(softbodies[softBodyId].mSimTetIndices)[elemIdx];
            const float4 p0 = pos_invMassGM[tetInd.x];
            const float4 p1 = pos_invMassGM[tetInd.y];
            const float4 p2 = pos_invMassGM[tetInd.z];
            const float4 p3 = pos_invMassGM[tetInd.w];
            gridPos = p0 * barycentric.x + p1 * barycentric.y + p2 * barycentric.z + p3 * barycentric.w;
            // NOT the blended .w -- see the header.
            invMass1 = getSoftBodyInvMass(gridPos.w, barycentric);
        }

        const float3 point = float3(gridPos.x, gridPos.y, gridPos.z);
        const float3 axis = float3(axis_angle.x, axis_angle.y, axis_angle.z);
        const float3 ra4 = float3(localPose0.x, localPose0.y, localPose0.z);

        const uint64_t rigidId = attIndex0;
        const bool isStatic = nodeIsStaticBody(rigidId);

        rigidAttachmentIds[workIndex] = rigidId;

        const float3 normal0 = float3(1.0f, 0.0f, 0.0f);
        const float3 normal1 = float3(0.0f, 1.0f, 0.0f);
        const float3 normal2 = float3(0.0f, 0.0f, 1.0f);

        // ARTICULATION anchor branch DEFERRED (see header).  solverBodyIdx is 0 for a
        // static body (solverBodyIndices untouched).
        uint solverBodyIdx = 0u;
        if (!isStatic)
            solverBodyIdx = Ptr<uint>(Ptr<PxgPrePrepDescGpu>(gSbPrepAtt.prePrepDesc)[0].solverBodyIndices)[nodeIndex(rigidId)];

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

        // CUDA zeroes rigidDeltaVel here when the buffer is bound (softBodyGM.cu:2026).
        // The pointer is compared, never loaded through, so there is nothing to speculate.
        if (gSbPrepAtt.rigidDeltaVel != 0UL)
        {
            Ptr<float4> rigidDeltaVel = Ptr<float4>(gSbPrepAtt.rigidDeltaVel);
            rigidDeltaVel[workIndex] = float4(0.0f, 0.0f, 0.0f, 0.0f);
            rigidDeltaVel[workIndex + numRigidAttachments] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
}

void solveRigidSoftAttachment(SbSolveAttArgs a, bool isTGS, uint idx, uint blockX)
{
    const uint numAttachments = a.numAttachments;
    const float dt = a.dt;
    const float invDt = 1.0f / dt;

    Ptr<PxgSoftBodyGpu> softbodies = Ptr<PxgSoftBodyGpu>(a.softbodies);
    Ptr<PxgSolverCoreDescGpu> solverCoreDesc = Ptr<PxgSolverCoreDescGpu>(a.solverCoreDesc);
    Ptr<PxgPrePrepDescGpu> prePrepDesc = Ptr<PxgPrePrepDescGpu>(a.prePrepDesc);

    const uint numSolverBodies = solverCoreDesc[0].numSolverBodies;
    const uint accOffset = solverCoreDesc[0].accumulatedBodyDeltaVOffset;
    Ptr<float4> initialVel = Ptr<float4>(solverCoreDesc[0].outSolverVelocity);

    const uint nbBlocksRequired = (numAttachments + RA_BLOCK - 1u) / RA_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + RA_GRID - 1u) / RA_GRID;

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * RA_BLOCK + idx + nbIterationsPerBlock * blockX * RA_BLOCK;
        if (workIndex >= numAttachments)
            return;

        const uint index = workIndex / 32u;
        const uint offset = workIndex & 31u;
        const uint64_t cBase = a.attachments + uint64_t(index) * RAC_STRIDE;

        const uint elemId = Ptr<uint>(cBase + RAC_elemId)[offset];
        const uint softBodyId = elemId >> PXG_BITSHIFT_TET_ID;
        const uint elemIdx = elemId & PX_MAX_NB_DEFORMABLE_VOLUME_TET;
        const float4 baryOrType = Ptr<float4>(cBase + RAC_baryOrType)[offset];
        const bool elemIsVertex = femIsVertexType(baryOrType);

        // P9.76c: THE TWO ARMS READ DIFFERENT BUFFERS.  PGS blends the VELOCITY
        // (mSimVelocity_InvMass, softBodyGM.cu:2125); TGS blends the accumulated
        // POSITION DELTA (mSimDeltaPos, :2251) -- and everything downstream follows
        // from that, because a TGS "error" is a position error where a PGS one is a
        // velocity error.  The blend itself, the .w inverse masses and the atomic
        // scatter tail are identical in both, which is why one body still serves both.
        Ptr<float4> srcGM = Ptr<float4>(isTGS ? softbodies[softBodyId].mSimDeltaPos
                                              : softbodies[softBodyId].mSimVelocity_InvMass);

        float4 softbodyState;
        uint4 tetInd = uint4(0u, 0u, 0u, 0u);
        float4 tetInvMass = float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (elemIsVertex)
        {
            softbodyState = srcGM[elemIdx];
        }
        else
        {
            tetInd = Ptr<uint4>(softbodies[softBodyId].mSimTetIndices)[elemIdx];
            const float4 v0 = srcGM[tetInd.x];
            const float4 v1 = srcGM[tetInd.y];
            const float4 v2 = srcGM[tetInd.z];
            const float4 v3 = srcGM[tetInd.w];
            softbodyState = v0 * baryOrType.x + v1 * baryOrType.y + v2 * baryOrType.z + v3 * baryOrType.w;
            tetInvMass = float4(v0.w, v1.w, v2.w, v3.w);
        }

        // CUDA's linVel1 (PGS) / linDelta1 (TGS) -- the same three floats, read from
        // whichever buffer this arm blended.
        const float3 lin1 = float3(softbodyState.x, softbodyState.y, softbodyState.z);

        const uint64_t rigidId = Ptr<uint64_t>(cBase + RAC_rigidId)[offset];
        const bool isStatic = nodeIsStaticBody(rigidId);
        // A static anchor holding a zero-inverse-mass (pinned) vertex has nothing to solve.
        if (isStatic && softbodyState.w == 0.0f)
            continue;

        // PxgVelocityReader::readVelocitiesPGS (dataReadWriteHelper.cuh:101).
        // ARTICULATION branch DEFERRED (see header).
        uint solverBodyIdx = 0u;
        if (!isStatic)
            solverBodyIdx = Ptr<uint>(prePrepDesc[0].solverBodyIndices)[nodeIndex(rigidId)];

        float3 rigidLinVel;
        float3 rigidAngVel;
        // TGS-only, and zero on the PGS arm: readVelocitiesPGS has no counterpart.
        float3 rigidLinDelta = float3(0.0f, 0.0f, 0.0f);
        float3 rigidAngDelta = float3(0.0f, 0.0f, 0.0f);
        if (isTGS)
        {
            // readVelocitiesTGS (dataReadWriteHelper.cuh:144-151) reads THREE float4
            // rows, not two, and the four vectors straddle the row boundaries.  P9.76c:
            // rows two and three were not read at all before this tranche, so linDelta
            // and angDelta were silently (0,0,0) -- invisible in softbodyhang, whose
            // anchor is static and whose pool rows are zero, and invisible to any
            // fixture that had not yet been written.
            Ptr<float4> pool = Ptr<float4>(Ptr<PxgSolverSharedDescTGSGpu>(a.sharedDesc)[0].iter_solverBodyVelPool);
            const float4 v0 = pool[accOffset + solverBodyIdx];
            const float4 v1 = pool[accOffset + solverBodyIdx + numSolverBodies];
            const float4 v2 = pool[accOffset + solverBodyIdx + numSolverBodies + numSolverBodies];
            rigidLinVel = float3(v0.x, v0.y, v0.z);
            rigidAngVel = float3(v0.w, v1.x, v1.y);
            rigidLinDelta = float3(v1.z, v1.w, v2.x);
            rigidAngDelta = float3(v2.y, v2.z, v2.w);
        }
        else
        {
            Ptr<float4> pool = Ptr<float4>(Ptr<PxgSolverSharedDescPGSGpu>(a.sharedDesc)[0].iter_solverBodyVelPool);
            const float4 iv0 = initialVel[solverBodyIdx];
            const float4 iv1 = initialVel[solverBodyIdx + numSolverBodies];
            const float4 dv0 = pool[accOffset + solverBodyIdx];
            const float4 dv1 = pool[accOffset + solverBodyIdx + numSolverBodies];
            rigidLinVel = float3(iv0.x + dv0.x, iv0.y + dv0.y, iv0.z + dv0.z);
            rigidAngVel = float3(iv1.x + dv1.x, iv1.y + dv1.y, iv1.z + dv1.z);
        }

        // calculateAttachmentDeltaImpulsePGS.  P9.76a CORRECTED THIS, AND THE CORRECTION
        // IS THE WHOLE TRANCHE.
        //
        // What stood here was:
        //     // NOTE the CUDA solve passes constraint.raXn0_biasW THREE TIMES, so
        //     // raXn0/1/2 AND all three position-error components come from
        //     // raXn0_biasW -- a faithful port of that upstream quirk, IDENTICAL TO
        //     // THE CLOTH PATH.
        // The last four words are the error, and they are the reason the error was
        // made.  The quirk is real -- but it belongs to the CLOTH call site
        // (FEMClothExternalSolve.cu:1450-1452), which passes constraint.raXn0_biasW
        // three times to the NON-template overload.  THIS path's call site is
        // softBodyGM.cu:2167, the TEMPLATE overload (attachments.cuh:138-145), whose
        // body forwards raXn0_biasW, raXn1_biasW AND raXn2_biasW -- three DISTINCT
        // rows -- into attachments.cuh:83, which reads one position error from each
        // (:105-107) and dots the angular velocity against a different raXn per axis
        // (:100-102).  femClothRigidAttachBda.slang gets this right in both
        // directions: raXn0 thrice in its PGS arm (:436) and three distinct rows in
        // its TGS arm (:531-533), because the cloth has BOTH kinds of call site.
        //
        // The prepare has always written all three rows correctly -- above at :298-300,
        // and softBodyGM.cu:2010-2012 -- with .w carrying error.x, error.y and error.z
        // respectively.  So the data was there the whole time; only this read was wrong.
        //
        // WHAT IT COST.  For a world anchor the sqrtInvInertia is zero, so the three
        // .xyz rows collapse to (0,0,0) and that half was invisible.  The .w half was
        // not: a volume pinned to the world and sagging under gravity has
        // error ~ (0, sag, 0), so positionError = error.x ~ 0 and the attachment
        // became a pure velocity damper with NO positional recovery.  For a KINEMATIC
        // anchor the .xyz half is live too -- softBodyGM.cu:1973-1979 gives
        // invMass0 == 0 && !isStatic the IDENTITY inertia, making the three rows
        // ra x n0/n1/n2, three distinct non-zero vectors.
        const float4 raXn0_biasW = Ptr<float4>(cBase + RAC_raXn0_biasW)[offset];
        const float4 raXn1_biasW = Ptr<float4>(cBase + RAC_raXn1_biasW)[offset];
        const float4 raXn2_biasW = Ptr<float4>(cBase + RAC_raXn2_biasW)[offset];
        const float4 velMult = Ptr<float4>(cBase + RAC_velMult_invMass)[offset];

        const float3 raXn0 = float3(raXn0_biasW.x, raXn0_biasW.y, raXn0_biasW.z);
        const float3 raXn1 = float3(raXn1_biasW.x, raXn1_biasW.y, raXn1_biasW.z);
        const float3 raXn2 = float3(raXn2_biasW.x, raXn2_biasW.y, raXn2_biasW.z);
        const float velOfRigidX = rigidLinVel.x + dotVec(rigidAngVel, raXn0);
        const float velOfRigidY = rigidLinVel.y + dotVec(rigidAngVel, raXn1);
        const float velOfRigidZ = rigidLinVel.z + dotVec(rigidAngVel, raXn2);

        const float positionErrorX = raXn0_biasW.w;
        const float positionErrorY = raXn1_biasW.w;
        const float positionErrorZ = raXn2_biasW.w;

        // P9.76c: THE TWO FORMULAS.  Until this tranche both entry points ran the PGS
        // one -- which could not even be noticed before P9.76b, because the TGS entry
        // was refused on every engine launch and no fixture covered it.  The moment the
        // ABI row was corrected and it began to dispatch, softbodyhang went to NaN:
        // a velocity formula fed accumulated position deltas, with 1/dt on the error.
        //
        // calculateAttachmentDeltaImpulseTGS (attachments.cuh:147-200) differs from its
        // PGS twin in four places, and the port had none of them:
        //   - the error is a POSITION error: linDelta - positionError - angDelta.raXn,
        //     where linDelta = lin1 - rigidLinDelta.  No invDt anywhere;
        //   - the rigid velocity enters only as `- velOfRigid * velDt`;
        //   - velDt is 0 on a velocity iteration, dt otherwise;
        //   - the whole product is scaled by biasCoefficient, which the host passes per
        //     substep (and which is documented upstream as already multiplied by dt).
        float3 deltaImpulse;
        if (isTGS)
        {
            const float3 linDelta = float3(lin1.x - rigidLinDelta.x,
                                           lin1.y - rigidLinDelta.y,
                                           lin1.z - rigidLinDelta.z);
            // Cone-LIMITED branch DEFERRED (see header) -- as in the PGS arm.
            const float3 tgsError = float3(linDelta.x - positionErrorX - dotVec(rigidAngDelta, raXn0),
                                           linDelta.y - positionErrorY - dotVec(rigidAngDelta, raXn1),
                                           linDelta.z - positionErrorZ - dotVec(rigidAngDelta, raXn2));
            const float velDt = a.isVelocityIteration ? 0.0f : dt;
            const float bc = a.biasCoefficient;
            deltaImpulse = float3((tgsError.x - velOfRigidX * velDt) * velMult.x * bc,
                                  (tgsError.y - velOfRigidY * velDt) * velMult.y * bc,
                                  (tgsError.z - velOfRigidZ * velDt) * velMult.z * bc);
        }
        else
        {
            const float biasFactor = 0.5f;
            // Cone-LIMITED branch DEFERRED (see header).
            const float3 velError = float3(lin1.x - positionErrorX * biasFactor * invDt,
                                           lin1.y - positionErrorY * biasFactor * invDt,
                                           lin1.z - positionErrorZ * biasFactor * invDt);

            deltaImpulse = float3((velError.x - velOfRigidX) * velMult.x,
                                  (velError.y - velOfRigidY) * velMult.y,
                                  (velError.z - velOfRigidZ) * velMult.z);
        }

        // DYNAMIC-rigid reaction DEFERRED: a static anchor never writes rigidDeltaVel.

        if (deltaImpulse.x != 0.0f || deltaImpulse.y != 0.0f || deltaImpulse.z != 0.0f)
        {
            // BOTH softbody arms scale by dt here (softBodyGM.cu:2199 and :2312).  The
            // cloth TGS twin does NOT -- FEMClothExternalSolve.cu:1668 is
            // `deltaPos = -deltaImpulse` with no dt, out of the same
            // calculateAttachmentDeltaImpulseTGS.  UPSTREAM ASYMMETRY, documented and
            // not reconciled: each port follows its own CUDA.
            const float3 deltaPos = float3(-deltaImpulse.x * dt, -deltaImpulse.y * dt, -deltaImpulse.z * dt);
            const uint64_t simDelta = softbodies[softBodyId].mSimDelta;
            // AtomicAdd(mSimDelta[v], deltaPos * invMass * bary, 1.f): the xyz accumulate
            // and the w carries a reference count of 1 per touched vertex.
            if (elemIsVertex)
            {
                pxgAtomicAddF4XYZ(simDelta, elemIdx, deltaPos * softbodyState.w);
                pxgAtomicAddF4W(simDelta, elemIdx, 1.0f);
            }
            else
            {
                // ASSOCIATION, the P9.37 class.  The two arms multiply the same three
                // factors in DIFFERENT orders upstream, and C's left-to-right grouping
                // makes that observable under NoContraction:
                //   PGS  deltaPos * tetInvMass.k * barycentric.k   (softBodyGM.cu:2195-2198)
                //   TGS  deltaPos * barycentric.k * tetInvMass.k   (softBodyGM.cu:2318-2321)
                // Grouping (invMass * bary) first, as this port did, matches NEITHER.
                const float4 f0 = isTGS ? baryOrType : tetInvMass;
                const float4 f1 = isTGS ? tetInvMass : baryOrType;
                pxgAtomicAddF4XYZ(simDelta, tetInd.x, (deltaPos * f0.x) * f1.x);
                pxgAtomicAddF4W(simDelta, tetInd.x, 1.0f);
                pxgAtomicAddF4XYZ(simDelta, tetInd.y, (deltaPos * f0.y) * f1.y);
                pxgAtomicAddF4W(simDelta, tetInd.y, 1.0f);
                pxgAtomicAddF4XYZ(simDelta, tetInd.z, (deltaPos * f0.z) * f1.z);
                pxgAtomicAddF4W(simDelta, tetInd.z, 1.0f);
                pxgAtomicAddF4XYZ(simDelta, tetInd.w, (deltaPos * f0.w) * f1.w);
                pxgAtomicAddF4W(simDelta, tetInd.w, 1.0f);
            }
        }
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]   // == RA_BLOCK; SB_UPDATEROTATION
void sb_solveRigidSoftAttachmentLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    SbSolveAttArgs a;
    a.numAttachments = gSbSolveAtt.numAttachments;
    a.dt = gSbSolveAtt.dt;
    a.softbodies = gSbSolveAtt.softbodies;
    a.solverCoreDesc = gSbSolveAtt.solverCoreDesc;
    a.prePrepDesc = gSbSolveAtt.prePrepDesc;
    a.attachments = gSbSolveAtt.attachments;
    a.sharedDesc = gSbSolveAtt.sharedDesc;
    a.biasCoefficient = 0.0f;        // TGS-only; the PGS arm reads neither
    a.isVelocityIteration = false;
    solveRigidSoftAttachment(a, false, groupThreadId.x, groupId.x);
}

[shader("compute")]
[numthreads(256, 1, 1)]   // == RA_BLOCK; SB_UPDATEROTATION
void sb_solveRigidSoftAttachmentLaunchTGS(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    // Its OWN 80-byte block (see SbSolveAttachParamsTGS).  P9.76c gave the body a real
    // TGS arm: until then this entry point ran the PGS formula, and both of its extra
    // arguments were read into nothing.
    SbSolveAttArgs a;
    a.numAttachments = gSbSolveAttTGS.numAttachments;
    a.dt = gSbSolveAttTGS.dt;
    a.softbodies = gSbSolveAttTGS.softbodies;
    a.solverCoreDesc = gSbSolveAttTGS.solverCoreDesc;
    a.prePrepDesc = gSbSolveAttTGS.prePrepDesc;
    a.attachments = gSbSolveAttTGS.attachments;
    a.sharedDesc = gSbSolveAttTGS.sharedDesc;
    a.biasCoefficient = gSbSolveAttTGS.biasCoefficient;
    a.isVelocityIteration = gSbSolveAttTGS.isVelocityIteration != 0u;
    solveRigidSoftAttachment(a, true, groupThreadId.x, groupId.x);
}
