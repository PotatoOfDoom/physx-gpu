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
// cloth vertex or triangle point to a world anchor, a dynamic rigid body or an
// articulation link, ported line-for-line from
//   gpusimulationcontroller/src/CUDA/FEMClothConstraintPrep.cu (the prepare)
//   gpusimulationcontroller/src/CUDA/FEMClothExternalSolve.cu   (the solves)
// with the attachment-impulse leaf from src/CUDA/attachments.cuh:
//   - cloth_rigidAttachmentPrepareLaunch (FEMClothConstraintPrep.cu:387, launch
//     PxgFEMClothCore.cpp:226): builds the block-of-32 PxgFEMRigidAttachment-
//     Constraint from each PxgFEMRigidAttachment — resolves the cloth attach point
//     (vertex or barycentric triangle point), the anchor frame, the three axis
//     responses and the per-axis velocity multipliers.  BOTH arms: the
//     ARTICULATION link (:478-527) and the rigid / static body (:528-587).
//   - cloth_solveRigidClothAttachmentLaunch (PGS) / ...TGSLaunch (TGS)
//     (FEMClothExternalSolve.cu:1371 / :1571, launch PxgFEMClothCore.cpp:1850):
//     read the anchor velocity (rigid body or link), compute the attachment delta
//     impulse (calculateAttachmentDeltaImpulsePGS/TGS, attachments.cuh:83 / :147,
//     cone-limited branch included), write the anchor's reaction into rigidDeltaVel
//     and scatter the cloth-side position correction into mDeltaPos (one vertex, or
//     the three corners of a triangle, barycentric * invMass weighted).
//
// TRANCHE B of task 35 ported what had been deferred here:
//   - the prepare's ARTICULATION arm (FEMClothConstraintPrep.cu:478-527): the link
//     pose is articulations[bodySims[node].articulationRemapId].linkBody2Worlds
//     [linkId] and the response spatialResponseMatrixW[linkId] (the same two
//     indices, and the same asmLoad6x6Record, as prepareFEMContacts in
//     pxgSlangFemRigidContact).  The CUDA quirks are copied as written: resp1 and
//     resp2 dot deltaV0 (not deltaV1 / deltaV2, which the CUDA computes and never
//     reads -- not computed here), velMultiplierXYZ_invMassW.w = 1, raXn stored
//     WITHOUT a sqrtInvInertia factor, and this arm does NOT zero rigidDeltaVel;
//   - the solves' ARTICULATION velocity read (dataReadWriteHelper.cuh:105-113 /
//     :126-141) -- readRigidVelocity of pxgSlangFemRigidContact, which carries both
//     arms (and sinks the artiCoreDesc reads into the link arm, its documented
//     deviation);
//   - the DYNAMIC-rigid / link REACTION (FEMClothExternalSolve.cu:1455-1467 PGS /
//     :1651-1664 TGS): deltaLinVel = deltaImpulse * invMass0, deltaAngVel from the
//     SAME raXn rows as the impulse (attachments.cuh:129-132 / :195-196).  The PGS
//     call site passes raXn0_biasW THREE times (:1451), so PGS takes raXn0 for all
//     three rows of the velocity, of the position error AND of deltaAngVel -- an
//     upstream quirk, reproduced.  The rows are consumed by accumulateDeltaVRigid*
//     (rigidDeltaAccumBda, PxgFEMClothCore.cpp:1888);
//   - TRIANGLE attachments in the solves (:1417-1431 + :1474-1483 PGS, :1616-1630
//     + :1671-1680 TGS).  Until then both solves read and scattered vertex #elemIdx
//     for a triangle attachment -- an unrelated vertex, or past the end of the
//     vertex buffers once the triangle index reached nbVerts;
//   - the cone-LIMITED branch (attachments.cuh:112-120 / :176-180).  It is NOT a
//     corner case: every public-API cloth attachment takes it (a NULL cone leaves
//     low_high_limits = (0,0), angle = -1, PxgSimulationController.cpp:3062-3063,
//     and isConeLimitedEnabled tests `>= 0`).  isConeLimitedEnabled /
//     computeConeLimitedError are the shared ones of pxgSlangFemFemAttach.  With a
//     zero-length cone (0,0) the branch returns relPos - 0 * dir; in TGS that is the
//     input up to the sign of a zero, in PGS the velError -> posError -> velError
//     round trip moves it by ~1 ulp.
//
// DEVIATIONS from the CUDA original (recipe policy):
//   - the descriptors arrive as device pointers read through Ptr<> (BDA), reusing
//     the byte-exact pxgSlang* mirrors (FEMClothDesc / PrepDesc / SolverDesc /
//     Common).  The block-of-32 constraint and the 96-byte PxgFEMRigidAttachment
//     are addressed by explicit byte offset through Ptr<> (no descriptor block
//     copy), exactly as the other BDA kernels read their SoA blocks;
//   - the artiCoreDesc reads are inside the link arm of readRigidVelocity (see
//     pxgSlangFemRigidContact's header); no value changes;
//   - getNormalized()'s PxRecipSqrt lowers to `v * (1.0/sqrt(m))` (the unix
//     intrinsic recipSqrt is literally 1.0f/::sqrtf, PxUnixMathIntrinsics.h:93) —
//     IEEE-correctly-rounded, bit-exact across backends; the CPU reference uses the
//     identical expression.
//   - computeConeLimitedError's PxSinCos is sin()/cos() (the module's own seam); it
//     is only reached for a cone with angle > 0, which no fixture or scene here uses.
//
// All float arithmetic is NoContraction-stamped (PXG_NOCONTRACT_SHADERS): the raXn
// cross / sqrtInvInertia dot responses feed the compared velMultiplier, and the
// velError / tgsError sums feed the compared deltaImpulse — either would fuse to
// FMAs otherwise, breaking bitwise parity with the -ffp-contract=off reference.

import pxgSlangAtomicFloat;   // atomicAdd(float*) parity — see that module
import pxgSlangFEMClothDesc;   // PxgFEMClothGpu
import pxgSlangPrepDesc;       // PxgPrePrepDescGpu, PxgConstraintPrepareDescGpu
import pxgSlangSolverDesc;     // PxgSolverCoreDescGpu, PxgSolverSharedDescPGSGpu, PxgSolverSharedDescTGSGpu
import pxgSlangCommon;         // PxgSolverBodyData, PxgSolverTxIData, PxgAlignedTransform
import pxgSlangFemRigidAttach; // the PxgFEMRigidAttachmentConstraint / PxgFEMRigidAttachment
                               // byte layouts and the quat/normalize helpers, shared with
                               // the softbody attachment path (see that module's header)
import pxgSlangFemRigidContact; // crossVec / dotVec / mat33MulVec, the PxNodeIndex accessors
                                // and readRigidVelocity (PxgVelocityReader, both arms)
import pxgSlangFemFemAttach;   // isConeLimitedEnabled / computeConeLimitedError (attachments.cuh:36/:48)
import pxgSlangArticulationDesc;   // PXG_ARTI_*, artiPtrField
import pxgSlangArticulationBlocks; // AbMatrix6x6
import pxgSlangSpatialMatrix;      // AbSpatialVector, asmSVec, asmM6MulVec, asmLoad6x6Record

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
    uint64_t prepareDesc;             // @56
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
    Ptr<PxgConstraintPrepareDescGpu> prepDesc = Ptr<PxgConstraintPrepareDescGpu>(gPrep.prepareDesc);

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
        const bool elemIsVertex = femIsVertexType(baryOrType1);

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

        if (nodeIsArticulation(rigidId))
        {
            // FEMClothConstraintPrep.cu:478-527.  Two indices from one PxNodeIndex: the
            // ARTICULATION through the node's body sim (articulationRemapId), the LINK
            // through articulationLinkId() -- as prepareFEMContacts' link arm.
            Ptr<PxgSolverSharedDescPGSGpu> sharedDesc = Ptr<PxgSolverSharedDescPGSGpu>(gPrep.sharedDesc);
            Ptr<PxgBodySim> bodySims = Ptr<PxgBodySim>(sharedDesc[0].mBodySimBufferDeviceData);
            const uint nodeIndexA = nodeIndex(rigidId);
            const uint artiId = bodySims[nodeIndexA].articulationRemapId;
            const uint64_t articulation = sharedDesc[0].articulations + uint64_t(artiId) * PXG_ARTI_STRIDE;

            const uint linkID = nodeArticulationLinkId(rigidId);
            // linkBody2Worlds: PxTransform (28 B) -- q at 0, p at 16.
            Ptr<float> body2World = Ptr<float>(artiPtrField(articulation, PXG_ARTI_LINK_BODY2WORLDS)
                                               + uint64_t(linkID) * PXG_ARTI_TRANSFORM_SIZE);
            const float4 b2wQ = float4(body2World[0], body2World[1], body2World[2], body2World[3]);
            const float3 bodyFrame0p = float3(body2World[4], body2World[5], body2World[6]);

            const float3 worldAxis = vecGetNormalized(quatRotateVec(b2wQ, axis));

            const float3 ra = quatRotateVec(b2wQ, ra4);
            const float3 error = float3(ra.x + bodyFrame0p.x - point.x,
                                        ra.y + bodyFrame0p.y - point.y,
                                        ra.z + bodyFrame0p.z - point.z);

            const float3 raXn0 = crossVec(ra, normal0);
            const float3 raXn1 = crossVec(ra, normal1);
            const float3 raXn2 = crossVec(ra, normal2);

            const AbMatrix6x6 spatialResponse =
                asmLoad6x6Record(artiPtrField(articulation, PXG_ARTI_SPATIAL_RESPONSE_MATRIX_W), linkID);
            // spatialResponse * Cm::UnAlignedSpatialVector(normal0, raXn0).  The CUDA also
            // forms deltaV1 / deltaV2 (:504-505) and never reads them: resp1 and resp2 below
            // dot deltaV0 -- the upstream quirk, copied.
            const AbSpatialVector deltaV0 = asmM6MulVec(spatialResponse, asmSVec(normal0, raXn0));

            const float resp0 = dotVec(deltaV0.top, raXn0) + dotVec(deltaV0.bottom, normal0) + invMass1;
            const float resp1 = dotVec(deltaV0.top, raXn1) + dotVec(deltaV0.bottom, normal1) + invMass1;
            const float resp2 = dotVec(deltaV0.top, raXn2) + dotVec(deltaV0.bottom, normal2) + invMass1;

            const float velMultiplier0 = (resp0 > 0.0f) ? (1.0f / resp0) : 0.0f;
            const float velMultiplier1 = (resp1 > 0.0f) ? (1.0f / resp1) : 0.0f;
            const float velMultiplier2 = (resp2 > 0.0f) ? (1.0f / resp2) : 0.0f;

            const float biasedErr0 = dotVec(error, normal0);
            const float biasedErr1 = dotVec(error, normal1);
            const float biasedErr2 = dotVec(error, normal2);

            // Raw raXn rows (no sqrtInvInertia factor), and invMass0 = 1 "so that the impulse
            // scaling for the linear impulse to convert it to a velocity change remains an
            // impulse" (:519-520).  No rigidDeltaVel zeroing in this arm (the CUDA has none).
            Ptr<float4>(cBase + RAC_raXn0_biasW)[offset] = float4(raXn0.x, raXn0.y, raXn0.z, biasedErr0);
            Ptr<float4>(cBase + RAC_raXn1_biasW)[offset] = float4(raXn1.x, raXn1.y, raXn1.z, biasedErr1);
            Ptr<float4>(cBase + RAC_raXn2_biasW)[offset] = float4(raXn2.x, raXn2.y, raXn2.z, biasedErr2);
            Ptr<float4>(cBase + RAC_velMult_invMass)[offset] = float4(velMultiplier0, velMultiplier1, velMultiplier2, 1.0f);
            Ptr<uint>(cBase + RAC_elemId)[offset] = elemId;
            Ptr<uint64_t>(cBase + RAC_rigidId)[offset] = rigidId;
            Ptr<float4>(cBase + RAC_baryOrType)[offset] = baryOrType1;
            Ptr<float4>(cBase + RAC_low_high_limits)[offset] = low_high_limits;
            Ptr<float4>(cBase + RAC_axis_angle)[offset] = float4(worldAxis.x, worldAxis.y, worldAxis.z, axis_angle.w);
            continue;
        }

        // :528 -- the rigid / static arm.  solverBodyIdx is 0 for a static body
        // (solverBodyIndices untouched).
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

        // :582-586 -- the rigid arm zeroes both reaction rows when the buffer is bound.
        if (gPrep.rigidDeltaVel != 0UL)
        {
            Ptr<float4> rigidDeltaVel = Ptr<float4>(gPrep.rigidDeltaVel);
            rigidDeltaVel[workIndex] = float4(0.0f, 0.0f, 0.0f, 0.0f);
            rigidDeltaVel[workIndex + numRigidAttachments] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
}

// ---------------------------------------------------------------------------
// The cloth-side attachment scatter (both solves): the CUDA
// AtomicAdd(float4&, PxVec3 v, 1.f) (atomic.cuh:46) as a REAL atomic
// (pxgSlangAtomicFloat).  `deltaPos * w` is the CUDA's last product: for a vertex
// `deltaPos * invMass`, for a triangle corner `(deltaPos * bary_i) * invMass_i`, the
// caller passing `deltaPos * bary_i` (FEMClothExternalSolve.cu:1476-1482 / :1673-1679).
// ---------------------------------------------------------------------------
void applyClothDelta(uint64_t deltaPosBase, uint vertIdx, float3 deltaPos, float w)
{
    pxgAtomicAddF4XYZ(deltaPosBase, vertIdx, deltaPos * w);
    pxgAtomicAddF4W(deltaPosBase, vertIdx, 1.0f);
}

// The anchor's reaction rows (FEMClothExternalSolve.cu:1455-1467 / :1651-1664), consumed
// by accumulateDeltaVRigidFirst/Second (rigidDeltaAccumBda, PxgFEMClothCore.cpp:1888).
void writeRigidReaction(uint64_t rigidDeltaVelBase, uint workIndex, uint numAttachments,
                        float invMass0, float3 deltaLinVel, float3 deltaAngVel)
{
    Ptr<float4> rigidDeltaVel = Ptr<float4>(rigidDeltaVelBase);
    if (invMass0 == 0.0f)
    {
        rigidDeltaVel[workIndex] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        rigidDeltaVel[workIndex + numAttachments] = float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    else
    {
        rigidDeltaVel[workIndex] = float4(deltaLinVel.x, deltaLinVel.y, deltaLinVel.z, 1.0f);
        rigidDeltaVel[workIndex + numAttachments] = float4(deltaAngVel.x, deltaAngVel.y, deltaAngVel.z, 0.0f);
    }
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

        // Cloth info (:1406-1431)
        const uint elemId = Ptr<uint>(cBase + RAC_elemId)[offset];
        const uint clothId = elemId >> PXG_CLOTH_ID_SHIFT;
        const uint elemIdx = elemId & PXG_CLOTH_ELEMENT_MASK;
        const float4 baryOrType = Ptr<float4>(cBase + RAC_baryOrType)[offset];
        const bool elemIsVertex = femIsVertexType(baryOrType);

        Ptr<float4> clothVelocities = Ptr<float4>(clothes[clothId].mVelocity_InvMass);

        float4 clothVel;
        uint4 triVertIdx = uint4(0u, 0u, 0u, 0u);
        float3 triInvMass = float3(0.0f, 0.0f, 0.0f);
        float4 barycentric = float4(0.0f, 0.0f, 0.0f, 0.0f);

        if (elemIsVertex)
        {
            clothVel = clothVelocities[elemIdx];
        }
        else
        {
            barycentric = baryOrType;
            triVertIdx = Ptr<uint4>(clothes[clothId].mTriangleVertexIndices)[elemIdx];
            const float4 v0 = clothVelocities[triVertIdx.x];
            const float4 v1 = clothVelocities[triVertIdx.y];
            const float4 v2 = clothVelocities[triVertIdx.z];
            clothVel = v0 * barycentric.x + v1 * barycentric.y + v2 * barycentric.z;

            triInvMass.x = v0.w;
            triInvMass.y = v1.w;
            triInvMass.z = v2.w;
        }

        const float3 linVel1 = float3(clothVel.x, clothVel.y, clothVel.z);

        // Rigid body info (:1435-1444)
        const uint64_t rigidId = Ptr<uint64_t>(cBase + RAC_rigidId)[offset];
        const bool isStatic = nodeIsStaticBody(rigidId);
        if (isStatic && clothVel.w == 0.0f)
            continue;

        // PxgVelocityReader::readVelocitiesPGS (dataReadWriteHelper.cuh:101) -- both arms.
        const RigidState vel0 = readRigidVelocity(rigidId, /*isTGS*/ false, gSolve.solverCoreDesc,
                                                  gSolve.sharedDesc, gSolve.prePrepDesc, gSolve.artiCoreDesc);

        // calculateAttachmentDeltaImpulsePGS (attachments.cuh:83).  NOTE the CUDA
        // solve passes constraint.raXn0_biasW THREE TIMES (FEMClothExternalSolve.cu:
        // 1451) — a faithful port of that upstream quirk, so raXn0/1/2, all three
        // position-error components AND the three deltaAngVel terms come from raXn0_biasW.
        const float4 raXn0_biasW = Ptr<float4>(cBase + RAC_raXn0_biasW)[offset];
        const float4 velMultiplierXYZ_invMassW = Ptr<float4>(cBase + RAC_velMult_invMass)[offset];
        const float4 low_high_limits = Ptr<float4>(cBase + RAC_low_high_limits)[offset];
        const float4 worldAxis_angle = Ptr<float4>(cBase + RAC_axis_angle)[offset];

        const float3 raXn0 = float3(raXn0_biasW.x, raXn0_biasW.y, raXn0_biasW.z);
        const float3 raXn1 = raXn0;
        const float3 raXn2 = raXn0;

        const float velOfRigidX = vel0.linVel.x + dotVec(vel0.angVel, raXn0);
        const float velOfRigidY = vel0.linVel.y + dotVec(vel0.angVel, raXn1);
        const float velOfRigidZ = vel0.linVel.z + dotVec(vel0.angVel, raXn2);

        const float positionErrorX = raXn0_biasW.w;
        const float positionErrorY = raXn0_biasW.w;
        const float positionErrorZ = raXn0_biasW.w;
        const float biasFactor = 0.5f;
        float3 velError = float3(linVel1.x - positionErrorX * biasFactor * invDt,
                                 linVel1.y - positionErrorY * biasFactor * invDt,
                                 linVel1.z - positionErrorZ * biasFactor * invDt);

        // attachments.cuh:112-120 -- taken for every public-API attachment (see header).
        if (isConeLimitedEnabled(worldAxis_angle.w, low_high_limits.x, low_high_limits.y))
        {
            // "we don't understand why we need to scale by biasFactor in order to get the
            // right error offset for the cone." (attachments.cuh:114)
            const float weirdScale = invDt * biasFactor;
            float3 posError = velError * (1.0f / weirdScale);
            const float3 worldAxis = float3(worldAxis_angle.x, worldAxis_angle.y, worldAxis_angle.z);
            posError = computeConeLimitedError(worldAxis_angle.w, low_high_limits.x, low_high_limits.y, worldAxis, posError);
            velError = posError * weirdScale;
        }

        const float deltaF0 = (velError.x - velOfRigidX) * velMultiplierXYZ_invMassW.x;
        const float deltaF1 = (velError.y - velOfRigidY) * velMultiplierXYZ_invMassW.y;
        const float deltaF2 = (velError.z - velOfRigidZ) * velMultiplierXYZ_invMassW.z;
        const float3 deltaImpulse = float3(deltaF0, deltaF1, deltaF2);

        // attachments.cuh:129-132
        const float invMass0 = velMultiplierXYZ_invMassW.w;
        const float3 deltaLinVel = deltaImpulse * invMass0;
        const float3 deltaAngVel = raXn0 * deltaF0 + raXn1 * deltaF1 + raXn2 * deltaF2;

        // Update rigid body (:1455-1467)
        if (!isStatic)
            writeRigidReaction(gSolve.rigidDeltaVel, workIndex, numAttachments, invMass0, deltaLinVel, deltaAngVel);

        // Update cloth (:1470-1484)
        if (deltaImpulse.x != 0.0f || deltaImpulse.y != 0.0f || deltaImpulse.z != 0.0f)
        {
            const float3 deltaPos = float3(-deltaImpulse.x * dt, -deltaImpulse.y * dt, -deltaImpulse.z * dt);
            const uint64_t deltaPosBase = clothes[clothId].mDeltaPos;

            if (elemIsVertex)
            {
                applyClothDelta(deltaPosBase, elemIdx, deltaPos, clothVel.w);
            }
            else
            {
                applyClothDelta(deltaPosBase, triVertIdx.x, deltaPos * barycentric.x, triInvMass.x);
                applyClothDelta(deltaPosBase, triVertIdx.y, deltaPos * barycentric.y, triInvMass.y);
                applyClothDelta(deltaPosBase, triVertIdx.z, deltaPos * barycentric.z, triInvMass.z);
            }
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

        // Cloth info (:1605-1630)
        const uint elemId = Ptr<uint>(cBase + RAC_elemId)[offset];
        const uint clothId = elemId >> PXG_CLOTH_ID_SHIFT;
        const uint elemIdx = elemId & PXG_CLOTH_ELEMENT_MASK;
        const float4 baryOrType = Ptr<float4>(cBase + RAC_baryOrType)[offset];
        const bool elemIsVertex = femIsVertexType(baryOrType);

        Ptr<float4> clothAccumDeltas = Ptr<float4>(clothes[clothId].mAccumulatedDeltaPos);

        float4 clothAccumDelta;
        uint4 triVertIdx = uint4(0u, 0u, 0u, 0u);
        float3 triInvMass = float3(0.0f, 0.0f, 0.0f);
        float4 barycentric = float4(0.0f, 0.0f, 0.0f, 0.0f);

        if (elemIsVertex)
        {
            clothAccumDelta = clothAccumDeltas[elemIdx];
        }
        else
        {
            barycentric = baryOrType;
            triVertIdx = Ptr<uint4>(clothes[clothId].mTriangleVertexIndices)[elemIdx];
            const float4 d0 = clothAccumDeltas[triVertIdx.x];
            const float4 d1 = clothAccumDeltas[triVertIdx.y];
            const float4 d2 = clothAccumDeltas[triVertIdx.z];
            clothAccumDelta = d0 * barycentric.x + d1 * barycentric.y + d2 * barycentric.z;

            triInvMass.x = d0.w;
            triInvMass.y = d1.w;
            triInvMass.z = d2.w;
        }
        const float3 linDelta1 = float3(clothAccumDelta.x, clothAccumDelta.y, clothAccumDelta.z);

        // Rigid body info (:1633-1641)
        const uint64_t rigidId = Ptr<uint64_t>(cBase + RAC_rigidId)[offset];
        const bool isStatic = nodeIsStaticBody(rigidId);
        if (isStatic && clothAccumDelta.w == 0.0f)
            continue;

        // PxgVelocityReader::readVelocitiesTGS (dataReadWriteHelper.cuh:122) -- both arms.
        const RigidState vel0 = readRigidVelocity(rigidId, /*isTGS*/ true, gSolve.solverCoreDesc,
                                                  gSolve.sharedDesc, gSolve.prePrepDesc, gSolve.artiCoreDesc);

        // calculateAttachmentDeltaImpulseTGS (attachments.cuh:147) — the offset
        // overload reads raXn0/1/2 correctly (distinct rows).
        const float4 raXn0_biasW = Ptr<float4>(cBase + RAC_raXn0_biasW)[offset];
        const float4 raXn1_biasW = Ptr<float4>(cBase + RAC_raXn1_biasW)[offset];
        const float4 raXn2_biasW = Ptr<float4>(cBase + RAC_raXn2_biasW)[offset];
        const float4 velMultiplierXYZ_invMassW = Ptr<float4>(cBase + RAC_velMult_invMass)[offset];
        const float4 low_high_limits = Ptr<float4>(cBase + RAC_low_high_limits)[offset];
        const float4 worldAxis_angle = Ptr<float4>(cBase + RAC_axis_angle)[offset];

        const float3 raXn0 = float3(raXn0_biasW.x, raXn0_biasW.y, raXn0_biasW.z);
        const float3 raXn1 = float3(raXn1_biasW.x, raXn1_biasW.y, raXn1_biasW.z);
        const float3 raXn2 = float3(raXn2_biasW.x, raXn2_biasW.y, raXn2_biasW.z);

        const float velOfRigid0 = vel0.linVel.x + dotVec(vel0.angVel, raXn0);
        const float velOfRigid1 = vel0.linVel.y + dotVec(vel0.angVel, raXn1);
        const float velOfRigid2 = vel0.linVel.z + dotVec(vel0.angVel, raXn2);

        const float3 linDelta = float3(linDelta1.x - vel0.linDelta.x, linDelta1.y - vel0.linDelta.y,
                                       linDelta1.z - vel0.linDelta.z);

        const float positionErrorX = raXn0_biasW.w;
        const float positionErrorY = raXn1_biasW.w;
        const float positionErrorZ = raXn2_biasW.w;

        float3 tgsError = float3(linDelta.x - positionErrorX - dotVec(vel0.angDelta, raXn0),
                                 linDelta.y - positionErrorY - dotVec(vel0.angDelta, raXn1),
                                 linDelta.z - positionErrorZ - dotVec(vel0.angDelta, raXn2));

        // attachments.cuh:176-180 -- taken for every public-API attachment (see header).
        if (isConeLimitedEnabled(worldAxis_angle.w, low_high_limits.x, low_high_limits.y))
        {
            const float3 worldAxis = float3(worldAxis_angle.x, worldAxis_angle.y, worldAxis_angle.z);
            tgsError = computeConeLimitedError(worldAxis_angle.w, low_high_limits.x, low_high_limits.y, worldAxis, tgsError);
        }

        const float velDt = dt; // isVelocityIteration == false (:1645)
        const float deltaF0 = (tgsError.x - velOfRigid0 * velDt) * velMultiplierXYZ_invMassW.x * FEMCLOTH_BIAS_COEFFICIENT;
        const float deltaF1 = (tgsError.y - velOfRigid1 * velDt) * velMultiplierXYZ_invMassW.y * FEMCLOTH_BIAS_COEFFICIENT;
        const float deltaF2 = (tgsError.z - velOfRigid2 * velDt) * velMultiplierXYZ_invMassW.z * FEMCLOTH_BIAS_COEFFICIENT;
        const float3 deltaImpulse = float3(deltaF0, deltaF1, deltaF2);

        // attachments.cuh:195-196
        const float3 deltaLinVel = deltaImpulse * velMultiplierXYZ_invMassW.w;
        const float3 deltaAngVel = raXn0 * deltaF0 + raXn1 * deltaF1 + raXn2 * deltaF2;

        // Update rigid body (:1651-1664)
        if (!isStatic)
            writeRigidReaction(gSolve.rigidDeltaVel, workIndex, numAttachments, velMultiplierXYZ_invMassW.w,
                               deltaLinVel, deltaAngVel);

        // Update cloth (:1667-1681)
        if (deltaImpulse.x != 0.0f || deltaImpulse.y != 0.0f || deltaImpulse.z != 0.0f)
        {
            const float3 deltaPos = float3(-deltaImpulse.x, -deltaImpulse.y, -deltaImpulse.z);
            const uint64_t deltaPosBase = clothes[clothId].mDeltaPos;

            if (elemIsVertex)
            {
                applyClothDelta(deltaPosBase, elemIdx, deltaPos, clothAccumDelta.w);
            }
            else
            {
                applyClothDelta(deltaPosBase, triVertIdx.x, deltaPos * barycentric.x, triInvMass.x);
                applyClothDelta(deltaPosBase, triVertIdx.y, deltaPos * barycentric.y, triInvMass.y);
                applyClothDelta(deltaPosBase, triVertIdx.z, deltaPos * barycentric.z, triInvMass.z);
            }
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
            // P9.86f: FEMClothExternalSolve.cu:1101-1102 are IEEE `pos.w + 0.f` / `accumDelta.w + 0.f`
            // (delta.w = 0.f above), which turn a -0 w (the inverse mass) into +0; llvmpipe folds the
            // forwarded literal-zero w lane of these float4 adds, so that lane's `x + 0` is a select
            // (as softBodySolveBda's apply-deltas).  vel.w gets `0 * invDt`, a computed zero: IEEE as is.
            pos += delta;
            pos.w = (pos.w == 0.0f) ? 0.0f : pos.w;
            accumDelta += delta;
            accumDelta.w = (accumDelta.w == 0.0f) ? 0.0f : accumDelta.w;

            positions[vertIdx] = pos;
            velocities[vertIdx] = vel;
            accumulatedDeltaPos[vertIdx] = accumDelta;

            mDeltaPos[vertIdx] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
}
