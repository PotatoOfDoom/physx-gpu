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

// femClothRigidAttachBda.slang -- the cloth-vs-rigid ATTACHMENT chain and the per-vertex
// delta fold, PhysX 5.12: pin a cloth vertex, or a point on a cloth triangle, to a rigid
// body, an articulation link or the world.
//
//   cloth_rigidAttachmentPrepareLaunch (FEMClothConstraintPrep.cu)   one
//       PxgRigidDbAttachmentBlock lane per active PxgDeformableAttachment
//       (prepareRigidDbAttachment): the three per-axis response rows, the position error,
//       the velocity multipliers, the deformable ref, the rigid node and the host-counted
//       rigid reference count; rigidAttachmentIds receives the rigid node.  The cloth
//       inverse mass is the vertex's .w, or for a triangle the SQUARED-barycentric sum
//       bx^2 w0 + by^2 w1 + bz^2 w2 (5.6.1 used the blended .w).
//   cloth_queryRigidClothAttachmentReferenceCountLaunch (FEMClothExternalSolve.cu)  NEW in
//       5.12: bumps mDeltaPos.w once per cloth vertex the attachment writes (a vertex ref:
//       that vertex; a triangle: every corner with |bc| > 1e-3).
//   cloth_solveRigidClothAttachmentLaunch (FEMClothExternalSolve.cu)  solveRigidDbAttachment
//       for PGS and TGS (isTGS; the TGS twin is gone), position and velocity iterations, the
//       rigid writeback (writeAttachmentDeltas) and the cloth scatter of -deltaImpulse * dt.
//       A static rigid attached to an all-kinematic point is skipped.
//   cloth_applyExternalDeltasLaunch (FEMClothExternalSolve.cu)  folds mDeltaPos (the
//       contact / attachment scatters; .w = the pre-counted reference count) into the
//       vertex state and clears it.  5.12 adds isVelocityIteration: the velocity always
//       takes delta * invDt, the position and the accumulated delta only in position
//       iterations.
//
// 5.12 removed the cone limits (and with them PxgFEMRigidAttachmentConstraint and the
// attachments.cuh helpers) and the 5.6.1 attachment record; PxgDeformableAttachment is
// {pair, barycentricOrPosition0, barycentric1}, 48 bytes.  It also fixes the PGS solve,
// which corrected the X axis only in 5.6.1.
//
// DEVIATIONS (value-neutral): the velocity reader reads artiCoreDesc inside its articulation
// arm only; `rigidDeltaVel` is optional in the prepare and compared, never loaded through.
//
// In PXG_NOCONTRACT_SHADERS: the responses are dot-of-cross chains, the quaternion rotate a
// mul/add chain, all compared bit-exactly against a CPU transcription of the 5.12 CUDA.

import pxgSlangPrepDesc;                    // PxgPrePrepDescGpu
import pxgSlangSolverDesc;                  // PxgSolverCoreDescGpu
import pxgSlangFEMClothDesc;                // PxgFEMClothGpu, clothAt
import pxgSlangDeformableRef;               // PXG_DATT_*, PXG_RDBA_*, PxgGetDeformable*
import pxgSlangVelocityReader;              // PxgMakeVelocityReader
import pxgSlangDeformableUtils;             // prepareRigidDbAttachment
import pxgSlangDeformableAndParticleUtils;  // PxgRigidPart, PxgDeformablePart, solveRigidDbAttachment

// PxgSoftBodyKernelGridDim / BlockDim::SB_UPDATEROTATION (the host launch dims of the three
// attachment kernels).
static const uint RA_BLOCK = 256u;
static const uint RA_GRID = 1024u;

// ===========================================================================
// cloth_rigidAttachmentPrepareLaunch
//   (PxgFEMCloth* clothes, PxgDeformableAttachment* rigidAttachments,
//    PxU32* activeRigidAttachments, const PxU32* rigidAttachmentRigidRefCounts,
//    PxNodeIndex* rigidAttachmentIds, PxU32 numRigidAttachments,
//    PxgRigidDbAttachmentBlock* attachmentBlocks, const PxgPrePrepDesc* preDesc,
//    const PxgConstraintPrepareDesc* prepareDesc, const PxgSolverSharedDescBase* sharedDesc,
//    float4* rigidDeltaVel)
//   -> {8, 8, 8, 8, 8, 4, 8, 8, 8, 8, 8} = 88-byte blob.
// ===========================================================================
struct PrepAttachParams
{
    uint64_t clothes;                        // @0
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
ConstantBuffer<PrepAttachParams> gPrepAtt;

[shader("compute")]
[numthreads(256, 1, 1)]   // == RA_BLOCK
void cloth_rigidAttachmentPrepareLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint numRigidAttachments = gPrepAtt.numRigidAttachments;
    const uint nbBlocksRequired = (numRigidAttachments + RA_BLOCK - 1u) / RA_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + RA_GRID - 1u) / RA_GRID;
    const uint idx = groupThreadId.x;

    Ptr<uint> activeRigidAttachments = Ptr<uint>(gPrepAtt.activeRigidAttachments);
    Ptr<uint64_t> rigidAttachmentIds = Ptr<uint64_t>(gPrepAtt.rigidAttachmentIds);
    const uint64_t solverBodyIndices = Ptr<PxgPrePrepDescGpu>(gPrepAtt.preDesc)[0].solverBodyIndices;

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * RA_BLOCK + idx + nbIterationsPerBlock * groupId.x * RA_BLOCK;
        if (workIndex >= numRigidAttachments)
            return;

        const uint index = workIndex / 32u;

        const uint attachmentId = activeRigidAttachments[workIndex];
        const uint64_t attachment = gPrepAtt.rigidAttachments + uint64_t(attachmentId) * PXG_DATT_STRIDE;
        const uint64_t block = gPrepAtt.attachmentBlocks + uint64_t(index) * PXG_RDBA_STRIDE;

        const uint64_t clothRef = Ptr<uint64_t>(attachment + PXG_DATT_pair + PXG_PAIRINFO_ref1)[0];
        const uint clothId = PxgGetDeformableId(clothRef);
        const uint elemIdx = PxgGetDeformableFeatureId(clothRef);
        const bool elemIsVertex = PxgGetDeformableFeatureType(clothRef) == PXG_DREF_eVERTEX;

        Ptr<PxgFEMClothGpu> cloth = clothAt(gPrepAtt.clothes, clothId);
        Ptr<float4> pos_invMass = Ptr<float4>(cloth[0].mPosition_InvMass);

        const float4 barycentric1 = Ptr<float4>(attachment + PXG_DATT_barycentric1)[0];

        float4 attachmentPose;
        float invMass1;
        if (elemIsVertex)
        {
            attachmentPose = pos_invMass[elemIdx];
            invMass1 = attachmentPose.w;
        }
        else
        {
            const float4 barycentric = barycentric1;
            const uint4 triVertInd = Ptr<uint4>(cloth[0].mTriangleVertexIndices_ownedEdges)[elemIdx];
            const float4 pos_iMass0 = pos_invMass[triVertInd.x];
            const float4 pos_iMass1 = pos_invMass[triVertInd.y];
            const float4 pos_iMass2 = pos_invMass[triVertInd.z];
            attachmentPose = pos_iMass0 * barycentric.x + pos_iMass1 * barycentric.y + pos_iMass2 * barycentric.z;
            // Squared-bary invMass for the constraint Jacobian-Mass-Jacobian denominator.
            invMass1 = barycentric.x * barycentric.x * pos_iMass0.w
                     + barycentric.y * barycentric.y * pos_iMass1.w
                     + barycentric.z * barycentric.z * pos_iMass2.w;
        }
        const float3 point = float3(attachmentPose.x, attachmentPose.y, attachmentPose.z);

        const uint64_t rigidId = Ptr<uint64_t>(attachment + PXG_DATT_pair + PXG_PAIRINFO_ref0)[0];   // PxgGetRigidNode
        rigidAttachmentIds[workIndex] = rigidId;

        const float4 barycentricOrPosition0 = Ptr<float4>(attachment + PXG_DATT_barycentricOrPosition0)[0];

        prepareRigidDbAttachment(block, workIndex & 31u, point, invMass1,
                                 float3(barycentricOrPosition0.x, barycentricOrPosition0.y, barycentricOrPosition0.z),
                                 rigidId, clothRef, barycentric1,
                                 Ptr<uint>(gPrepAtt.rigidAttachmentRigidRefCounts)[workIndex],
                                 gPrepAtt.prepareDesc, solverBodyIndices, gPrepAtt.sharedDesc,
                                 gPrepAtt.rigidDeltaVel, workIndex, numRigidAttachments);
    }
}

// ===========================================================================
// cloth_queryRigidClothAttachmentReferenceCountLaunch (NEW in 5.12)
//   (PxgFEMCloth* clothes, PxgRigidDbAttachmentBlock* attachmentBlocks,
//    const PxU32 numAttachments) -> {8, 8, 4} = 24-byte blob.
// ===========================================================================
struct QueryAttachParams
{
    uint64_t clothes;           // @0
    uint64_t attachmentBlocks;  // @8
    uint numAttachments;        // @16
    uint _pad20;                // @20 -> 24
}
[[vk::push_constant]]
ConstantBuffer<QueryAttachParams> gQueryAtt;

[shader("compute")]
[numthreads(256, 1, 1)]   // == RA_BLOCK
void cloth_queryRigidClothAttachmentReferenceCountLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint numAttachments = gQueryAtt.numAttachments;
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

        const uint64_t block = gQueryAtt.attachmentBlocks + uint64_t(index) * PXG_RDBA_STRIDE;

        const uint64_t clothRef = Ptr<uint64_t>(block + PXG_RDBA_deformableRef)[offset];
        Ptr<PxgFEMClothGpu> cloth = clothAt(gQueryAtt.clothes, PxgGetDeformableId(clothRef));

        bumpClothRefCountBary(cloth, clothRef, block + PXG_RDBA_barycentric, offset);
    }
}

// ===========================================================================
// cloth_solveRigidClothAttachmentLaunch
//   (PxgFEMCloth* clothes, PxgRigidDbAttachmentBlock* attachmentBlocks,
//    const PxU32 numAttachments, PxgPrePrepDesc* prePrepDesc, PxgSolverCoreDesc* solverCoreDesc,
//    PxgArticulationCoreDesc* artiCoreDesc, float4* solverBodyVelPool, const PxReal dt,
//    const PxReal biasCoefficient, float4* rigidDeltaVel, bool isVelocityIteration, bool isTGS)
//   -> {8, 8, 4, 8, 8, 8, 8, 4, 4, 8, 1, 1} = 80-byte blob (isVelocityIteration byte @72,
//   isTGS byte @73).
// ===========================================================================
struct SolveAttachParams
{
    uint64_t clothes;                    // @0
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
ConstantBuffer<SolveAttachParams> gSolveAtt;

[shader("compute")]
[numthreads(256, 1, 1)]   // == RA_BLOCK
void cloth_solveRigidClothAttachmentLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint numSolverBodies = Ptr<PxgSolverCoreDescGpu>(gSolveAtt.solverCoreDesc)[0].numSolverBodies;
    const uint numAttachments = gSolveAtt.numAttachments;
    const uint nbBlocksRequired = (numAttachments + RA_BLOCK - 1u) / RA_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + RA_GRID - 1u) / RA_GRID;
    const uint idx = groupThreadId.x;

    const float dt = gSolveAtt.dt;
    const float biasCoefficient = gSolveAtt.biasCoefficient;
    const bool isVelocityIteration = (gSolveAtt.isVelocityIteration_isTGS & 0xffu) != 0u;
    const bool isTGS = ((gSolveAtt.isVelocityIteration_isTGS >> 8u) & 0xffu) != 0u;

    const PxgVelocityReaderGpu velocityReader = PxgMakeVelocityReader(gSolveAtt.prePrepDesc, gSolveAtt.solverCoreDesc,
                                                                      gSolveAtt.artiCoreDesc,
                                                                      gSolveAtt.solverBodyVelPool, numSolverBodies);

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * RA_BLOCK + idx + nbIterationsPerBlock * groupId.x * RA_BLOCK;
        if (workIndex >= numAttachments)
            return;

        const uint index = workIndex / 32u;
        const uint offset = workIndex & 31u;

        const uint64_t block = gSolveAtt.attachmentBlocks + uint64_t(index) * PXG_RDBA_STRIDE;

        // A vertex attachment has no barycentric; the feature type says so without reading one.
        const uint64_t clothRef = Ptr<uint64_t>(block + PXG_RDBA_deformableRef)[offset];
        Ptr<PxgFEMClothGpu> cloth = clothAt(gSolveAtt.clothes, PxgGetDeformableId(clothRef));

        // Deformable side: PGS reads mVelocity_InvMass, TGS mAccumulatedDeltaPos;
        // vertexInvMasses come refCount-inflated, attachPointInvMass carries the raw
        // bc^2 * invM for the solver's denomBias.
        PxgDeformablePart db = PxgDeformablePart();
        db.readClothAttachment(cloth, clothRef, block + PXG_RDBA_barycentric, offset, isTGS, dt, isVelocityIteration);

        // Both-static early exit: rigid static AND every contributing cloth vertex kinematic.
        const uint64_t rigidId = Ptr<uint64_t>(block + PXG_RDBA_rigidId)[offset];
        if (PxgNodeIsStaticBody(rigidId) && db.attachPointInvMass == 0.0f)
            continue;

        // Rigid side: readVelocity zeroes linDelta / angDelta on PGS; on TGS they carry the
        // rigid's accumulated motion, which the solve zeroes in velocity iterations.
        PxgRigidPart rigid = PxgRigidPart();
        rigid.readAttachmentPrep(block, offset);
        rigid.readVelocity(velocityReader, rigidId, isTGS);

        float3 deltaLinVel, deltaAngVel;
        const float3 deltaImpulse = solveRigidDbAttachment(rigid, db, dt, biasCoefficient, isTGS, isVelocityIteration,
                                                           deltaLinVel, deltaAngVel);

        // Rigid writeback.
        rigid.writeAttachmentDeltas(gSolveAtt.rigidDeltaVel, rigidId, deltaLinVel, deltaAngVel, workIndex,
                                    workIndex + numAttachments);

        // Scatter .xyz weighted by the refCount-inflated vertexInvMasses; the pre-count owns .w.
        if (!(deltaImpulse.x == 0.0f && deltaImpulse.y == 0.0f && deltaImpulse.z == 0.0f))
        {
            // A position delta in position iterations, velocity * dt in velocity iterations;
            // cloth_applyExternalDeltasLaunch applies it.
            const float3 delta = float3((-deltaImpulse.x) * dt, (-deltaImpulse.y) * dt, (-deltaImpulse.z) * dt);
            db.writeCloth(cloth, delta);
        }
    }
}

// ===========================================================================
// cloth_applyExternalDeltasLaunch
//   (PxgFEMCloth* femClothes, const PxU32* activeClothes, const PxReal dt,
//    bool isVelocityIteration) -> {8, 8, 4, 1} = 24-byte blob (isVelocityIteration byte @20).
// Per vertex, block SB_SOLVETETRA (64), grid (numBlocks, nbActiveClothes, 1).  The CUDA
// block-copies the PxgFEMCloth into shared memory first; the fields are read through Ptr<>
// here (no value changes).
// ===========================================================================
struct ApplyExtDeltasParams
{
    uint64_t clothes;                 // @0
    uint64_t activeClothes;           // @8
    float dt;                         // @16
    uint isVelocityIteration;         // @20  bool byte0 -> 24
}
[[vk::push_constant]]
ConstantBuffer<ApplyExtDeltasParams> gApply;

[shader("compute")]
[numthreads(64, 1, 1)] // SB_SOLVETETRA
void cloth_applyExternalDeltasLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint id = Ptr<uint>(gApply.activeClothes)[groupId.y];
    Ptr<PxgFEMClothGpu> cloth = clothAt(gApply.clothes, id);
    const float dt = gApply.dt;
    const bool isVelocityIteration = (gApply.isVelocityIteration & 0xffu) != 0u;

    const uint nbVerts = cloth[0].mNbVerts;
    const uint vertIdx = groupThreadId.x + groupId.x * 64u;

    if (vertIdx < nbVerts)
    {
        Ptr<float4> mDeltaPos = Ptr<float4>(cloth[0].mDeltaPos);
        float4 delta = mDeltaPos[vertIdx];

        if (delta.w != 0.0f)
        {
            Ptr<float4> positions = Ptr<float4>(cloth[0].mPosition_InvMass);
            Ptr<float4> velocities = Ptr<float4>(cloth[0].mVelocity_InvMass);
            Ptr<float4> accumulatedDeltaPos = Ptr<float4>(cloth[0].mAccumulatedDeltaPos);

            float4 accumDelta = accumulatedDeltaPos[vertIdx];
            const float invDt = 1.0f / dt;
            float4 pos = positions[vertIdx];
            float4 vel = velocities[vertIdx];
            const float scale = 1.0f / delta.w;

            delta.x *= scale;
            delta.y *= scale;
            delta.z *= scale;
            delta.w = 0.0f;

            // Velocity iteration writes only the velocity correction, skipping the position
            // and accumulated-delta updates (Muller SCA 2020 S3.6).  vel.w gets
            // `0 * invDt`, a computed zero: IEEE as is.
            vel += delta * invDt;
            velocities[vertIdx] = vel;

            if (!isVelocityIteration)
            {
                // P9.86f: `pos.w + 0.f` / `accumDelta.w + 0.f` (delta.w = 0.f above) turn a
                // -0 w (the inverse mass) into +0; llvmpipe folds the forwarded literal-zero
                // w lane of these float4 adds, so that lane's `x + 0` is a select.
                pos += delta;
                pos.w = (pos.w == 0.0f) ? 0.0f : pos.w;
                accumDelta += delta;
                accumDelta.w = (accumDelta.w == 0.0f) ? 0.0f : accumDelta.w;
                positions[vertIdx] = pos;
                accumulatedDeltaPos[vertIdx] = accumDelta;
            }

            // Clear delta.
            mDeltaPos[vertIdx] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
}
