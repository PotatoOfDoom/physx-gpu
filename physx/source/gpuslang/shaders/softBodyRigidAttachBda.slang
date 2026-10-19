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

// softBodyRigidAttachBda.slang -- the softbody-vs-rigid ATTACHMENT chain of softBodyGM.cu,
// PhysX 5.12: pin a softbody vertex, or a point inside a simulation tetrahedron, to a rigid
// body, an articulation link or the world.
//
//   sb_rigidAttachmentPrepareLaunch                   one PxgRigidDbAttachmentBlock lane
//       per active PxgDeformableAttachment (prepareRigidDbAttachment): the three per-axis
//       response rows, the position error, the velocity multipliers, the deformable ref,
//       the rigid node and the host-counted rigid reference count; rigidAttachmentIds
//       receives the rigid node.  The deformable inverse mass is the vertex's .w, or for a
//       tet the SQUARED-barycentric sum bx^2 w0 + by^2 w1 + bz^2 w2 + bw^2 w3 (5.6.1 used
//       the blended .w scaled by |bary|).
//   sb_queryRigidSoftAttachmentReferenceCountLaunch   NEW in 5.12: bumps mSimDelta.w once
//       per deformable vertex the attachment writes (a vertex ref: that vertex; a tet:
//       every corner with |bc| > 1e-3).
//   sb_solveRigidSoftAttachmentLaunch                 solveRigidDbAttachment for PGS and
//       TGS (isTGS; the TGS twin is gone), position and velocity iterations, the rigid
//       writeback (writeAttachmentDeltas) and the deformable scatter of -deltaImpulse * dt.
//       A static rigid attached to an all-kinematic point is skipped.
//
// 5.12 removed the cone limits (and with them the per-attachment limit/axis rows) and the
// separate attachment record layout; PxgDeformableAttachment is {pair, barycentricOrPosition0,
// barycentric1}, 48 bytes.
//
// DEVIATIONS (value-neutral): the velocity reader reads artiCoreDesc inside its articulation
// arm only; `rigidDeltaVel` is optional in the prepare and compared, never loaded through.
//
// In PXG_NOCONTRACT_SHADERS: the responses are dot-of-cross chains, the quaternion rotate a
// mul/add chain, all compared bit-exactly against a CPU transcription of the 5.12 CUDA.

import pxgSlangPrepDesc;                    // PxgPrePrepDescGpu
import pxgSlangSolverDesc;                  // PxgSolverCoreDescGpu
import pxgSlangSoftBodyDesc;                // PxgSoftBodyGpu, sbAt
import pxgSlangDeformableRef;               // PXG_DATT_*, PXG_RDBA_*, PxgGetDeformable*
import pxgSlangVelocityReader;              // PxgMakeVelocityReader
import pxgSlangDeformableUtils;             // prepareRigidDbAttachment
import pxgSlangDeformableAndParticleUtils;  // PxgRigidPart, PxgDeformablePart, solveRigidDbAttachment

// PxgSoftBodyKernelGridDim / BlockDim::SB_UPDATEROTATION.
static const uint RA_BLOCK = 256u;
static const uint RA_GRID = 1024u;

// ===========================================================================
// sb_rigidAttachmentPrepareLaunch
//   (PxgSoftBody* softbodies, PxgDeformableAttachment* rigidAttachments,
//    PxU32* activeRigidAttachments, const PxU32* rigidAttachmentRigidRefCounts,
//    PxNodeIndex* rigidAttachmentIds, PxU32 numRigidAttachments,
//    PxgRigidDbAttachmentBlock* attachmentBlocks, PxgPrePrepDesc* preDesc,
//    PxgConstraintPrepareDesc* prepareDesc, PxgSolverSharedDescBase* sharedDesc,
//    float4* rigidDeltaVel)
//   -> {8, 8, 8, 8, 8, 4, 8, 8, 8, 8, 8} = 88-byte blob.
// ===========================================================================
struct SbPrepAttachParams
{
    uint64_t softbodies;                     // @0
    uint64_t rigidAttachments;               // @8
    uint64_t activeRigidAttachments;         // @16
    uint64_t rigidAttachmentRigidRefCounts;  // @24
    uint64_t rigidAttachmentIds;             // @32  output
    uint numRigidAttachments;                // @40
    uint _pad44;                             // @44
    uint64_t attachmentBlocks;               // @48  output
    uint64_t preDesc;                        // @56
    uint64_t prepareDesc;                    // @64
    uint64_t sharedDesc;                     // @72  articulation arm only
    uint64_t rigidDeltaVel;                  // @80  output, OPTIONAL -> 88
}
[[vk::push_constant]]
ConstantBuffer<SbPrepAttachParams> gSbPrepAtt;

[shader("compute")]
[numthreads(256, 1, 1)]   // == RA_BLOCK
void sb_rigidAttachmentPrepareLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint numRigidAttachments = gSbPrepAtt.numRigidAttachments;
    const uint nbBlocksRequired = (numRigidAttachments + RA_BLOCK - 1u) / RA_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + RA_GRID - 1u) / RA_GRID;
    const uint idx = groupThreadId.x;

    Ptr<uint> activeRigidAttachments = Ptr<uint>(gSbPrepAtt.activeRigidAttachments);
    Ptr<uint64_t> rigidAttachmentIds = Ptr<uint64_t>(gSbPrepAtt.rigidAttachmentIds);
    const uint64_t solverBodyIndices = Ptr<PxgPrePrepDescGpu>(gSbPrepAtt.preDesc)[0].solverBodyIndices;

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * RA_BLOCK + idx + nbIterationsPerBlock * groupId.x * RA_BLOCK;
        if (workIndex >= numRigidAttachments)
            return;

        const uint index = workIndex / 32u;

        const uint attachmentIndex = activeRigidAttachments[workIndex];
        const uint64_t attachment = gSbPrepAtt.rigidAttachments + uint64_t(attachmentIndex) * PXG_DATT_STRIDE;
        const uint64_t block = gSbPrepAtt.attachmentBlocks + uint64_t(index) * PXG_RDBA_STRIDE;

        const uint64_t deformableRef = Ptr<uint64_t>(attachment + PXG_DATT_pair + PXG_PAIRINFO_ref1)[0];
        const uint softBodyId = PxgGetDeformableId(deformableRef);
        const uint elemIdx = PxgGetDeformableFeatureId(deformableRef);
        const bool elemIsVertex = PxgGetDeformableFeatureType(deformableRef) == PXG_DREF_eVERTEX;

        Ptr<PxgSoftBodyGpu> softbody = sbAt(gSbPrepAtt.softbodies, softBodyId);
        Ptr<float4> pos_invMassGM = Ptr<float4>(softbody[0].mSimPosition_InvMass);

        const float4 barycentric1 = Ptr<float4>(attachment + PXG_DATT_barycentric1)[0];

        float4 gridPos;
        float invMass1;
        if (elemIsVertex)
        {
            gridPos = pos_invMassGM[elemIdx];
            invMass1 = gridPos.w;
        }
        else
        {
            const float4 barycentric = barycentric1;
            const uint4 tetInd = Ptr<uint4>(softbody[0].mSimTetIndices)[elemIdx];
            const float4 pos_iMass0 = pos_invMassGM[tetInd.x];
            const float4 pos_iMass1 = pos_invMassGM[tetInd.y];
            const float4 pos_iMass2 = pos_invMassGM[tetInd.z];
            const float4 pos_iMass3 = pos_invMassGM[tetInd.w];
            gridPos = pos_iMass0 * barycentric.x + pos_iMass1 * barycentric.y + pos_iMass2 * barycentric.z
                    + pos_iMass3 * barycentric.w;
            // Squared-bary invMass for the constraint Jacobian-Mass-Jacobian denominator.
            invMass1 = barycentric.x * barycentric.x * pos_iMass0.w
                     + barycentric.y * barycentric.y * pos_iMass1.w
                     + barycentric.z * barycentric.z * pos_iMass2.w
                     + barycentric.w * barycentric.w * pos_iMass3.w;
        }

        const float3 point = float3(gridPos.x, gridPos.y, gridPos.z);

        const uint64_t rigidId = Ptr<uint64_t>(attachment + PXG_DATT_pair + PXG_PAIRINFO_ref0)[0];   // PxgGetRigidNode
        rigidAttachmentIds[workIndex] = rigidId;

        const float4 barycentricOrPosition0 = Ptr<float4>(attachment + PXG_DATT_barycentricOrPosition0)[0];

        prepareRigidDbAttachment(block, workIndex & 31u, point, invMass1,
                                 float3(barycentricOrPosition0.x, barycentricOrPosition0.y, barycentricOrPosition0.z),
                                 rigidId, deformableRef, barycentric1,
                                 Ptr<uint>(gSbPrepAtt.rigidAttachmentRigidRefCounts)[workIndex],
                                 gSbPrepAtt.prepareDesc, solverBodyIndices, gSbPrepAtt.sharedDesc,
                                 gSbPrepAtt.rigidDeltaVel, workIndex, numRigidAttachments);
    }
}

// ===========================================================================
// sb_queryRigidSoftAttachmentReferenceCountLaunch (NEW in 5.12)
//   (PxgSoftBody* softbodies, PxgRigidDbAttachmentBlock* attachmentBlocks,
//    const PxU32 numAttachments) -> {8, 8, 4} = 24-byte blob.
// ===========================================================================
struct SbQueryAttachParams
{
    uint64_t softbodies;        // @0
    uint64_t attachmentBlocks;  // @8
    uint numAttachments;        // @16
    uint _pad20;                // @20 -> 24
}
[[vk::push_constant]]
ConstantBuffer<SbQueryAttachParams> gSbQueryAtt;

[shader("compute")]
[numthreads(256, 1, 1)]   // == RA_BLOCK
void sb_queryRigidSoftAttachmentReferenceCountLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint numAttachments = gSbQueryAtt.numAttachments;
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

        const uint64_t block = gSbQueryAtt.attachmentBlocks + uint64_t(index) * PXG_RDBA_STRIDE;

        const uint64_t softBodyRef = Ptr<uint64_t>(block + PXG_RDBA_deformableRef)[offset];
        Ptr<PxgSoftBodyGpu> softbody = sbAt(gSbQueryAtt.softbodies, PxgGetDeformableId(softBodyRef));

        bumpSoftBodyRefCountBary(softbody, softBodyRef, block + PXG_RDBA_barycentric, offset);
    }
}

// ===========================================================================
// sb_solveRigidSoftAttachmentLaunch
//   (PxgSoftBody* softbodies, PxgRigidDbAttachmentBlock* attachmentBlocks,
//    const PxU32 numAttachments, PxgPrePrepDesc* prePrepDesc, PxgSolverCoreDesc* solverCoreDesc,
//    PxgArticulationCoreDesc* artiCoreDesc, float4* solverBodyVelPool, const PxReal dt,
//    const PxReal biasCoefficient, float4* rigidDeltaVel, bool isVelocityIteration, bool isTGS)
//   -> {8, 8, 4, 8, 8, 8, 8, 4, 4, 8, 1, 1} = 80-byte blob (isVelocityIteration byte @72,
//   isTGS byte @73).
// ===========================================================================
struct SbSolveAttachParams
{
    uint64_t softbodies;                 // @0
    uint64_t attachmentBlocks;           // @8
    uint numAttachments;                 // @16
    uint _pad20;                         // @20
    uint64_t prePrepDesc;                // @24
    uint64_t solverCoreDesc;             // @32
    uint64_t artiCoreDesc;               // @40
    uint64_t solverBodyVelPool;          // @48
    float dt;                            // @56
    float biasCoefficient;               // @60
    uint64_t rigidDeltaVel;              // @64  output
    uint isVelocityIteration_isTGS;      // @72  byte0 = isVelocityIteration, byte1 = isTGS
    uint _pad76;                         // @76 -> 80
}
[[vk::push_constant]]
ConstantBuffer<SbSolveAttachParams> gSbSolveAtt;

[shader("compute")]
[numthreads(256, 1, 1)]   // == RA_BLOCK
void sb_solveRigidSoftAttachmentLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint numSolverBodies = Ptr<PxgSolverCoreDescGpu>(gSbSolveAtt.solverCoreDesc)[0].numSolverBodies;
    const uint numAttachments = gSbSolveAtt.numAttachments;
    const uint nbBlocksRequired = (numAttachments + RA_BLOCK - 1u) / RA_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + RA_GRID - 1u) / RA_GRID;
    const uint idx = groupThreadId.x;

    const float dt = gSbSolveAtt.dt;
    const float biasCoefficient = gSbSolveAtt.biasCoefficient;
    const bool isVelocityIteration = (gSbSolveAtt.isVelocityIteration_isTGS & 0xffu) != 0u;
    const bool isTGS = ((gSbSolveAtt.isVelocityIteration_isTGS >> 8u) & 0xffu) != 0u;

    const PxgVelocityReaderGpu velocityReader = PxgMakeVelocityReader(gSbSolveAtt.prePrepDesc, gSbSolveAtt.solverCoreDesc,
                                                                      gSbSolveAtt.artiCoreDesc,
                                                                      gSbSolveAtt.solverBodyVelPool, numSolverBodies);

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * RA_BLOCK + idx + nbIterationsPerBlock * groupId.x * RA_BLOCK;
        if (workIndex >= numAttachments)
            return;

        const uint index = workIndex / 32u;
        const uint offset = workIndex & 31u;

        const uint64_t block = gSbSolveAtt.attachmentBlocks + uint64_t(index) * PXG_RDBA_STRIDE;

        // A vertex attachment has no barycentric; the feature type says so without reading one.
        const uint64_t softBodyRef = Ptr<uint64_t>(block + PXG_RDBA_deformableRef)[offset];
        Ptr<PxgSoftBodyGpu> softbody = sbAt(gSbSolveAtt.softbodies, PxgGetDeformableId(softBodyRef));

        // Deformable side: PGS reads mSimVelocity_InvMass, TGS mSimDeltaPos; vertexInvMasses
        // come refCount-inflated, attachPointInvMass carries the raw bc^2 * invM.
        PxgDeformablePart db = PxgDeformablePart();
        db.readSoftBodyAttachment(softbody, softBodyRef, block + PXG_RDBA_barycentric, offset, isTGS, dt,
                                  isVelocityIteration);

        // Both-static early exit: rigid static AND every contributing vertex kinematic.
        const uint64_t rigidId = Ptr<uint64_t>(block + PXG_RDBA_rigidId)[offset];
        if (PxgNodeIsStaticBody(rigidId) && db.attachPointInvMass == 0.0f)
            continue;

        // Rigid side: readVelocity zeroes linDelta / angDelta on PGS.
        PxgRigidPart rigid = PxgRigidPart();
        rigid.readAttachmentPrep(block, offset);
        rigid.readVelocity(velocityReader, rigidId, isTGS);

        float3 deltaLinVel, deltaAngVel;
        const float3 deltaImpulse = solveRigidDbAttachment(rigid, db, dt, biasCoefficient, isTGS, isVelocityIteration,
                                                           deltaLinVel, deltaAngVel);

        // Rigid writeback.
        rigid.writeAttachmentDeltas(gSbSolveAtt.rigidDeltaVel, rigidId, deltaLinVel, deltaAngVel, workIndex,
                                    workIndex + numAttachments);

        // Scatter .xyz weighted by the refCount-inflated vertexInvMasses; the pre-count owns .w.
        if (!(deltaImpulse.x == 0.0f && deltaImpulse.y == 0.0f && deltaImpulse.z == 0.0f))
        {
            // A position delta in position iterations, velocity * dt in velocity iterations;
            // sb_gm_applyExternalDeltasLaunch applies it.
            const float3 delta = float3((-deltaImpulse.x) * dt, (-deltaImpulse.y) * dt, (-deltaImpulse.z) * dt);
            db.writeSoftBody(softbody, delta);
        }
    }
}
