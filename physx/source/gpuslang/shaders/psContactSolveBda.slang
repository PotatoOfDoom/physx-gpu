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

// psContactSolveBda -- the particle-SIDE half of the PBD particle/rigid contact solve on
// the BDA engine ABI, PhysX 5.12 (gpusimulationcontroller/src/CUDA/particlesystem.cu):
//
//   ps_contactPrepareLaunch               prepares each rigid-particle contact into the
//                                         block-of-32 PxgRigidParticleContactBlock
//                                         (raXn_resp, raXnF0_resp, raXnF1_resp,
//                                         normal_errorW, fricTan0_invMass0,
//                                         penBiasClampRigid).  Launched twice: by rigid
//                                         (with the rigid deltaVel) and by particle
//                                         (deltaVel NULL).
//   ps_queryPCContactReferenceCountLaunch NEW in 5.12: the mass-splitting pre-count.  One
//                                         thread per contact bumps rigidRefCount at the
//                                         rigid's getGlobalRigidBodyId slot.
//   ps_solvePCOutputParticleDeltaVLaunch  the particle half of solveRigidDbContact (with
//                                         adhesion); 5.12 folds the TGS twin into isTGS.
//   ps_accumulateDeltaVParticleLaunch     per-particle reduction of the per-contact deltas.
//   ps_accumulateFEMParticleDeltaVLaunch  its FEM twin; 5.12 also resets the mass-split
//                                         count (mAccumDeltaP.w) afterwards.
//
// THE 5.12 DATA.  The contacts arrive as the SoA PxgDeformablePairInfo stream
// {ref0 = rigid PxNodeIndex, ref1 = PxgParticleRef} plus a float4 normalPen stream
// (pxgSlangDeformableRef); the constraint block no longer carries the ids, and the solves
// read them from the pair stream.  The velMultipliers of 5.6.1 are gone: the block keeps
// the rigid unit responses, and solveRigidDbContact builds the denominator from them, the
// particle invMass and the rigid's PC contact count (pxgSlangDeformableAndParticleUtils).
// The 5.6.1 solvePCOutputDeltaV / solvePCOutputDeltaVTGS helpers are deleted upstream.
//
// THE ARTICULATION ARM of the prepare is ported (5.6.1's port deferred it): the link pose,
// the three spatialResponseMatrixW products and the link's penBiasClamp, invMass0 = 1.
// UPSTREAM, kept: respF0 and respF1 are formed from deltaV0 (the NORMAL response) rather
// than from deltaV1 / deltaV2, which are computed and never read.  The same lines are in
// 5.6.1.  DOCUMENTED, NOT FIXED.
//
// DEVIATIONS (value-neutral):
//   - artiCoreDesc->mMaxLinksPerArticulation is read inside the articulation arm of
//     getGlobalRigidBodyId, and the velocity reader reads artiCoreDesc lazily
//     (pxgSlangVelocityReader), where the CUDA reads both at the kernel top.
//   - The PxU16 phase -> material handle is read through its 32-bit word.
//   - Every BDA load sits behind the uniform guard `workIndex < tNumContacts` (an early
//     return; no group barriers in any of these kernels).

import pxgSlangParticleDesc;               // PxgParticleSystemGpu, PXG_PARTICLE_PHASE_GROUP_MASK, pxgLoadPhaseToMaterial
import pxgSlangCommon;                     // PxgSolverBodyData, PxgSolverTxIData, PxgAlignedTransform, PxgBodySim
import pxgSlangPrepDesc;                   // PxgPrePrepDescGpu, PxgConstraintPrepareDescGpu
import pxgSlangSolverDesc;                 // PxgSolverCoreDescGpu, PxgSolverSharedDescPGSGpu
import pxgSlangArticulationDesc;           // PXG_ARTI_*, artiPtrField
import pxgSlangArticulationBlocks;         // AbMatrix6x6
import pxgSlangSpatialMatrix;              // AbSpatialVector, asmSVec, asmM6MulVec, asmLoad6x6Record
import pxgSlangDeformableRef;              // PxgDeformablePairInfoGpu, PXG_RPC_*, PxgGetParticle*
import pxgSlangVelocityReader;             // PxgMakeVelocityReader
import pxgSlangDeformableAndParticleUtils; // PxgRigidPart, PxgDeformablePart, solveRigidDbContact

// PxgParticleSystemKernelBlockDim / GridDim (PxgParticleSystemCoreKernelIndices.h)
static const uint PC_BLOCK = 256u;   // PS_COLLISION block   (prepare, query, solve)
static const uint PC_GRID = 1024u;   // PS_COLLISION grid
static const uint AD_BLOCK = 512u;   // ACCUMULATE_DELTA block (accumulate)
static const uint AD_GRID = 32u;     // ACCUMULATE_DELTA grid

// PxMat33 (column-major sqrtInvInertia[9]) * PxVec3: column0*v.x + column1*v.y + column2*v.z.
float3 mat33MulVec3(float m[9], float3 v)
{
    return float3(m[0] * v.x + m[3] * v.y + m[6] * v.z,
                  m[1] * v.x + m[4] * v.y + m[7] * v.z,
                  m[2] * v.x + m[5] * v.y + m[8] * v.z);
}

// &getParticleMaterial<PxsParticleMaterialData>(materials, mi, stride).
uint64_t psMaterialElem(Ptr<PxgParticleSystemGpu> ps, uint particleId)
{
    const uint phase = Ptr<uint>(ps[0].mSortedPhaseArray)[particleId];
    const uint group = phase & PXG_PARTICLE_PHASE_GROUP_MASK;
    const uint mi = pxgLoadPhaseToMaterial(ps[0].mPhaseGroupToMaterialHandle, group);
    return ps[0].mParticleMaterials + uint64_t(mi) * uint64_t(ps[0].mParticleMaterialStride);
}

// ===========================================================================
// 1. ps_contactPrepareLaunch
//    CUDA ABI (PxgParticleSystemCore.cpp prepPrimitiveConstraint /
//    PxgPBDParticleSystemCore.cpp prepParticleConstraint):
//      (PxgParticleSystem* particleSystems, PxgDeformablePairInfo* sortedContactInfos,
//       const float4* sortedContactNormalPens, PxU32* numContacts,
//       PxgRigidParticleContactBlock* contactBlocks, PxgPrePrepDesc* preDesc,
//       PxgConstraintPrepareDesc* prepareDesc, float2* appliedForces, const PxReal dt,
//       const bool isTGS, float4* deltaVel, PxgSolverSharedDescBase* sharedDesc)
//    -> {8,8,8,8,8,8,8,8,4,1,8,8} = 88-byte blob (dt @64, isTGS byte @68, deltaVel @72).
//    dt and isTGS are not read by the body.
// ===========================================================================
struct PcPrepParams
{
    uint64_t particleSystems;         // @0
    uint64_t sortedContactInfos;      // @8
    uint64_t sortedContactNormalPens; // @16
    uint64_t numContacts;             // @24
    uint64_t contactBlocks;           // @32
    uint64_t preDesc;                 // @40
    uint64_t prepareDesc;             // @48
    uint64_t appliedForces;           // @56
    float dt;                         // @64 (unused by the body)
    uint isTGSWord;                   // @68 (bool byte0; unused by the body)
    uint64_t deltaVel;                // @72
    uint64_t sharedDesc;              // @80 -> 88
}
[[vk::push_constant]]
ConstantBuffer<PcPrepParams> gCp;

[shader("compute")]
[numthreads(PC_BLOCK, 1, 1)]
void ps_contactPrepareLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint tNumContacts = Ptr<uint>(gCp.numContacts)[0];

    const uint nbBlocksRequired = (tNumContacts + PC_BLOCK - 1u) / PC_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + PC_GRID - 1u) / PC_GRID;
    const uint idx = groupThreadId.x;

    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gCp.particleSystems);
    Ptr<PxgDeformablePairInfoGpu> contactInfos = Ptr<PxgDeformablePairInfoGpu>(gCp.sortedContactInfos);
    Ptr<float4> normalPens = Ptr<float4>(gCp.sortedContactNormalPens);
    Ptr<float2> appliedForces = Ptr<float2>(gCp.appliedForces);
    Ptr<PxgConstraintPrepareDescGpu> prepDesc = Ptr<PxgConstraintPrepareDescGpu>(gCp.prepareDesc);
    Ptr<PxgPrePrepDescGpu> preDesc = Ptr<PxgPrePrepDescGpu>(gCp.preDesc);

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * PC_BLOCK + idx + nbIterationsPerBlock * groupId.x * PC_BLOCK;
        if (workIndex >= tNumContacts)
            return;

        appliedForces[workIndex] = float2(0.0f, 0.0f);

        const PxgDeformablePairInfoGpu contact = contactInfos[workIndex];
        const uint64_t blockBase = gCp.contactBlocks + uint64_t(workIndex / 32u) * PXG_RPC_STRIDE;
        const uint tIdx = workIndex & 31u;

        const uint64_t particleRef = contact.ref1;
        const uint particleSystemId = PxgGetParticleSystemId(particleRef);
        const uint particleId = PxgGetParticleIndex(particleRef);

        const float4 pointInvM = Ptr<float4>(systems[particleSystemId].mSortedPositions_InvMass)[particleId];
        const float3 point = float3(pointInvM.x, pointInvM.y, pointInvM.z);

        const float4 normalPen = normalPens[workIndex];
        const float3 normal = float3(-normalPen.x, -normalPen.y, -normalPen.z);
        const float pen = normalPen.w;

        const uint64_t rigidId = contact.ref0;   // PxgGetRigidNode
        uint solverBodyIdx = 0u;
        if (!PxgNodeIsStaticBody(rigidId))
            solverBodyIdx = Ptr<uint>(preDesc[0].solverBodyIndices)[PxgNodeIndexOf(rigidId)];

        if (PxgNodeIsArticulation(rigidId))
        {
            Ptr<PxgSolverSharedDescPGSGpu> sharedDesc = Ptr<PxgSolverSharedDescPGSGpu>(gCp.sharedDesc);
            Ptr<PxgBodySim> bodySims = Ptr<PxgBodySim>(sharedDesc[0].mBodySimBufferDeviceData);
            const uint artiId = bodySims[PxgNodeIndexOf(rigidId)].articulationRemapId;
            const uint64_t articulation = sharedDesc[0].articulations + uint64_t(artiId) * PXG_ARTI_STRIDE;

            const uint linkID = PxgNodeArticulationLinkId(rigidId);
            // linkBody2Worlds[linkID]: PxTransform (28 B), p at float 4.
            Ptr<float> body2World = Ptr<float>(artiPtrField(articulation, PXG_ARTI_LINK_BODY2WORLDS)
                                               + uint64_t(linkID) * PXG_ARTI_TRANSFORM_SIZE);
            const float3 bodyFrame0p = float3(body2World[4], body2World[5], body2World[6]);

            const AbMatrix6x6 spatialResponse =
                asmLoad6x6Record(artiPtrField(articulation, PXG_ARTI_SPATIAL_RESPONSE_MATRIX_W), linkID);

            float3 t0, t1;
            dpComputeBasisVectors(normal, t0, t1);

            const float3 ra = float3(point.x - bodyFrame0p.x, point.y - bodyFrame0p.y, point.z - bodyFrame0p.z);
            const float3 raXn = dpCross3(ra, normal);
            const float3 raXF0 = dpCross3(ra, t0);
            const float3 raXF1 = dpCross3(ra, t1);

            // deltaV1 / deltaV2 (the tangent responses) are computed upstream and never
            // read: see the header.  They have no side effect, so they are not formed here.
            const AbSpatialVector deltaV0 = asmM6MulVec(spatialResponse, asmSVec(normal, raXn));

            const float resp0 = dpDot3(deltaV0.top, raXn) + dpDot3(deltaV0.bottom, normal);
            const float respF0 = dpDot3(deltaV0.top, raXF0) + dpDot3(deltaV0.bottom, t0);
            const float respF1 = dpDot3(deltaV0.top, raXF1) + dpDot3(deltaV0.bottom, t1);

            const float penBiasClampRigid = Ptr<float>(artiPtrField(articulation, PXG_ARTI_LINKS)
                                                       + uint64_t(linkID) * PXG_ARTI_LINK_SIZE
                                                       + PXG_ARTI_LINKREC_INITIAL_ANGVEL_PENBIAS)[3];

            Ptr<float4>(blockBase + PXG_RPC_raXn_resp)[tIdx] = float4(raXn.x, raXn.y, raXn.z, resp0);
            Ptr<float4>(blockBase + PXG_RPC_normal_errorW)[tIdx] = float4(normal.x, normal.y, normal.z, pen);
            // invMass0 is 1 so that the impulse rather than a deltaV is recorded.
            Ptr<float4>(blockBase + PXG_RPC_fricTan0_invMass0)[tIdx] = float4(t0.x, t0.y, t0.z, 1.0f);
            Ptr<float4>(blockBase + PXG_RPC_raXnF0_resp)[tIdx] = float4(raXF0.x, raXF0.y, raXF0.z, respF0);
            Ptr<float4>(blockBase + PXG_RPC_raXnF1_resp)[tIdx] = float4(raXF1.x, raXF1.y, raXF1.z, respF1);
            Ptr<float>(blockBase + PXG_RPC_penBiasClampRigid)[tIdx] = penBiasClampRigid;
        }
        else
        {
            Ptr<PxgSolverBodyData> solverBodyData = Ptr<PxgSolverBodyData>(prepDesc[0].solverBodyDataPool);
            Ptr<PxgSolverTxIData> solverTxI = Ptr<PxgSolverTxIData>(prepDesc[0].solverBodyTxIDataPool);
            Ptr<PxgAlignedTransform> bodyFrames = Ptr<PxgAlignedTransform>(prepDesc[0].body2WorldPool);

            const float4 linVel_invMass0 = solverBodyData[solverBodyIdx].initialLinVelXYZ_invMassW;
            const float invMass0 = linVel_invMass0.w;

            float invSqrtInertia0[9];
            float inertiaScale = 1.0f;
            if (solverBodyIdx != 0u && invMass0 == 0.0f)
            {
                // PxMat33(PxIdentity)
                invSqrtInertia0[0] = 1.0f; invSqrtInertia0[1] = 0.0f; invSqrtInertia0[2] = 0.0f;
                invSqrtInertia0[3] = 0.0f; invSqrtInertia0[4] = 1.0f; invSqrtInertia0[5] = 0.0f;
                invSqrtInertia0[6] = 0.0f; invSqrtInertia0[7] = 0.0f; invSqrtInertia0[8] = 1.0f;
                inertiaScale = 0.0f;
            }
            else
            {
                invSqrtInertia0 = solverTxI[solverBodyIdx].sqrtInvInertia;
            }

            const float4 bodyFrame0p4 = bodyFrames[solverBodyIdx].p;
            const float3 bodyFrame0p = float3(bodyFrame0p4.x, bodyFrame0p4.y, bodyFrame0p4.z);

            float3 t0, t1;
            dpComputeBasisVectors(normal, t0, t1);

            const float3 ra = float3(point.x - bodyFrame0p.x, point.y - bodyFrame0p.y, point.z - bodyFrame0p.z);
            const float3 raXn = dpCross3(ra, normal);
            const float3 raXF0 = dpCross3(ra, t0);
            const float3 raXF1 = dpCross3(ra, t1);

            const float3 raXnSqrtInertia = mat33MulVec3(invSqrtInertia0, raXn);
            const float resp0 = (dpDot3(raXnSqrtInertia, raXnSqrtInertia)) * inertiaScale + invMass0;

            const float3 raXF0SqrtInertia = mat33MulVec3(invSqrtInertia0, raXF0);
            const float3 raXF1SqrtInertia = mat33MulVec3(invSqrtInertia0, raXF1);

            const float respF0 = (dpDot3(raXF0SqrtInertia, raXF0SqrtInertia)) * inertiaScale + invMass0;
            const float respF1 = (dpDot3(raXF1SqrtInertia, raXF1SqrtInertia)) * inertiaScale + invMass0;

            Ptr<float4>(blockBase + PXG_RPC_raXn_resp)[tIdx] =
                float4(raXnSqrtInertia.x, raXnSqrtInertia.y, raXnSqrtInertia.z, resp0);
            Ptr<float4>(blockBase + PXG_RPC_normal_errorW)[tIdx] = float4(normal.x, normal.y, normal.z, pen);
            Ptr<float4>(blockBase + PXG_RPC_fricTan0_invMass0)[tIdx] = float4(t0.x, t0.y, t0.z, invMass0);
            Ptr<float4>(blockBase + PXG_RPC_raXnF0_resp)[tIdx] =
                float4(raXF0SqrtInertia.x, raXF0SqrtInertia.y, raXF0SqrtInertia.z, respF0);
            Ptr<float4>(blockBase + PXG_RPC_raXnF1_resp)[tIdx] =
                float4(raXF1SqrtInertia.x, raXF1SqrtInertia.y, raXF1SqrtInertia.z, respF1);
            // Static -> idx 0 (the world solver body, penBiasClamp -1e32 = unbounded).
            Ptr<float>(blockBase + PXG_RPC_penBiasClampRigid)[tIdx] = solverBodyData[solverBodyIdx].initialAngVelXYZ_penBiasClamp.w;
        }

        if (gCp.deltaVel != 0)
        {
            Ptr<float4> deltaVel = Ptr<float4>(gCp.deltaVel);
            deltaVel[workIndex] = float4(0.0f, 0.0f, 0.0f, 0.0f);
            deltaVel[workIndex + tNumContacts] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
}

// ===========================================================================
// 2. ps_queryPCContactReferenceCountLaunch (NEW in 5.12)
//    CUDA ABI (PxgParticleSystemCore.cpp solvePrimitiveCollisionForParticles):
//      (PxgParticleSystem* particleSystems, PxgDeformablePairInfo* sortedContactInfos,
//       PxgRigidParticleContactBlock* contactBlocks, PxU32* numContacts,
//       PxgPrePrepDesc* prePrepDesc, PxgSolverCoreDesc* solverCoreDesc,
//       float4* solverBodyVelPool, PxU32* rigidRefCount, PxReal dt, bool isTGS,
//       PxgArticulationCoreDesc* artiCoreDesc)
//    -> {8,8,8,8,8,8,8,8,4,1,8} = 80-byte blob (dt @64, isTGS byte @68,
//    artiCoreDesc @72).  particleSystems, solverBodyVelPool, dt and isTGS are not read.
//    The host zeroes rigidRefCount (numSolverBodies + numArticulations * maxLinks) first.
//    The order of the atomic increments is irrelevant: integer adds.
// ===========================================================================
struct PcQueryParams
{
    uint64_t particleSystems;     // @0  (unused)
    uint64_t sortedContactInfos;  // @8
    uint64_t contactBlocks;       // @16
    uint64_t numContacts;         // @24
    uint64_t prePrepDesc;         // @32
    uint64_t solverCoreDesc;      // @40
    uint64_t solverBodyVelPool;   // @48 (unused)
    uint64_t rigidRefCount;       // @56 output
    float dt;                     // @64 (unused)
    uint isTGSWord;               // @68 (unused)
    uint64_t artiCoreDesc;        // @72 -> 80
}
[[vk::push_constant]]
ConstantBuffer<PcQueryParams> gQr;

[shader("compute")]
[numthreads(PC_BLOCK, 1, 1)]
void ps_queryPCContactReferenceCountLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint numSolverBodies = Ptr<PxgSolverCoreDescGpu>(gQr.solverCoreDesc)[0].numSolverBodies;

    const uint tNumContacts = Ptr<uint>(gQr.numContacts)[0];

    const uint nbBlocksRequired = (tNumContacts + PC_BLOCK - 1u) / PC_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + PC_GRID - 1u) / PC_GRID;
    const uint idx = groupThreadId.x;

    Ptr<PxgDeformablePairInfoGpu> contactInfos = Ptr<PxgDeformablePairInfoGpu>(gQr.sortedContactInfos);

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * PC_BLOCK + idx + nbIterationsPerBlock * groupId.x * PC_BLOCK;
        if (workIndex >= tNumContacts)
            return;

        const uint tIdx = workIndex & 31u;
        const uint64_t blockBase = gQr.contactBlocks + uint64_t(workIndex / 32u) * PXG_RPC_STRIDE;

        const uint64_t rigidId = contactInfos[workIndex].ref0;

        PxgRigidPart rigid = PxgRigidPart();
        rigid.readContactPrep(blockBase, tIdx);

        // A kinematic (invMass 0) particle still pushes the rigid, so only the rigid's
        // invMass is asked about.
        const int globalRigidBodyId = getGlobalRigidBodyId(gQr.prePrepDesc, rigidId, numSolverBodies, gQr.artiCoreDesc);
        if (rigidIsRefCounted(globalRigidBodyId, rigid.invMass))
        {
            uint prev;
            InterlockedAdd(Ptr<uint>(gQr.rigidRefCount)[uint(globalRigidBodyId)], 1u, prev);
        }
    }
}

// ===========================================================================
// 3. ps_solvePCOutputParticleDeltaVLaunch
//    CUDA ABI: (PxgParticleSystem* particleSystems, PxgDeformablePairInfo*
//      sortedContactInfos, PxgRigidParticleContactBlock* contactBlocks,
//      PxU32* numContacts, PxgPrePrepDesc* prePrepDesc, PxgSolverCoreDesc* solverCoreDesc,
//      float4* solverBodyVelPool, float4* deltaPos, float2* appliedForces,
//      PxU32* rigidRefCount, PxReal dt, bool isVelocityIteration, bool isTGS,
//      PxgArticulationCoreDesc* artiCoreDesc)
//    -> {8x10, 4, 1, 1, 8} = 96-byte blob (dt @80, isVelocityIteration byte @84, isTGS
//    byte @85, artiCoreDesc @88).
// ===========================================================================
struct PcSolveParams
{
    uint64_t particleSystems;      // @0
    uint64_t sortedContactInfos;   // @8
    uint64_t contactBlocks;        // @16
    uint64_t numContacts;          // @24
    uint64_t prePrepDesc;          // @32
    uint64_t solverCoreDesc;       // @40
    uint64_t solverBodyVelPool;    // @48
    uint64_t deltaPos;             // @56 output
    uint64_t appliedForces;        // @64
    uint64_t rigidRefCount;        // @72
    float dt;                      // @80
    uint flagsWord;                // @84 byte0 = isVelocityIteration, byte1 = isTGS
    uint64_t artiCoreDesc;         // @88 -> 96
}
[[vk::push_constant]]
ConstantBuffer<PcSolveParams> gSv;

[shader("compute")]
[numthreads(PC_BLOCK, 1, 1)]
void ps_solvePCOutputParticleDeltaVLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const bool isVelocityIteration = (gSv.flagsWord & 0xffu) != 0u;
    const bool isTGS = ((gSv.flagsWord >> 8u) & 0xffu) != 0u;
    const float dt = gSv.dt;

    const uint numSolverBodies = Ptr<PxgSolverCoreDescGpu>(gSv.solverCoreDesc)[0].numSolverBodies;
    const PxgVelocityReaderGpu velocityReader = PxgMakeVelocityReader(gSv.prePrepDesc, gSv.solverCoreDesc,
                                                                      gSv.artiCoreDesc, gSv.solverBodyVelPool,
                                                                      numSolverBodies);

    const uint tNumContacts = Ptr<uint>(gSv.numContacts)[0];

    const uint nbBlocksRequired = (tNumContacts + PC_BLOCK - 1u) / PC_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + PC_GRID - 1u) / PC_GRID;
    const uint idx = groupThreadId.x;

    Ptr<PxgDeformablePairInfoGpu> contactInfos = Ptr<PxgDeformablePairInfoGpu>(gSv.sortedContactInfos);
    Ptr<float4> deltaPos = Ptr<float4>(gSv.deltaPos);
    Ptr<float2> appliedForces = Ptr<float2>(gSv.appliedForces);

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * PC_BLOCK + idx + nbIterationsPerBlock * groupId.x * PC_BLOCK;
        if (workIndex >= tNumContacts)
            return;

        const uint tIdx = workIndex & 31u;

        float2 appliedForce = appliedForces[workIndex];

        const uint64_t blockBase = gSv.contactBlocks + uint64_t(workIndex / 32u) * PXG_RPC_STRIDE;
        const uint64_t particleRef = contactInfos[workIndex].ref1;
        const uint particleSystemId = PxgGetParticleSystemId(particleRef);
        const uint particleId = PxgGetParticleIndex(particleRef);

        Ptr<PxgParticleSystemGpu> particleSystem = Ptr<PxgParticleSystemGpu>(gSv.particleSystems
                                                       + uint64_t(particleSystemId) * PXG_PARTICLE_SYSTEM_STRIDE);
        const uint64_t mat = psMaterialElem(particleSystem, particleId);

        const float4 deltaP_invMassW = Ptr<float4>(particleSystem[0].mSortedDeltaP)[particleId];
        const float invMass1 = deltaP_invMassW.w;

        // A particle of infinite mass takes no delta; the accumulate scan reads every
        // slot, so this one is cleared.
        if (invMass1 == 0.0f)
        {
            deltaPos[workIndex] = float4(0.0f, 0.0f, 0.0f, 0.0f);
            continue;
        }

        const uint64_t rigidId = contactInfos[workIndex].ref0;

        // Symmetric PC mass split: the rigid response is scaled by its PC contact count,
        // exactly as in ps_solvePCOutputRigidDeltaVLaunch.
        PxgRigidPart rigid = PxgRigidPart();
        const int globalRigidBodyId = getGlobalRigidBodyId(gSv.prePrepDesc, rigidId, numSolverBodies, gSv.artiCoreDesc);
        const float rigidInvMass0 = Ptr<float4>(blockBase + PXG_RPC_fricTan0_invMass0)[tIdx].w;
        rigid.readBodyProperties(rigidId, globalRigidBodyId, rigidInvMass0, gSv.rigidRefCount, 0);
        rigid.readContactPrep(blockBase, tIdx);
        rigid.readVelocity(velocityReader, rigidId, isTGS);

        PxgDeformablePart particle = PxgDeformablePart();
        particle.readParticle(particleSystem, particleId, deltaP_invMassW, mat,
                              /*inflateByRefCount*/ false, dt, isVelocityIteration);

        PxgDbContactPair contactPair = PxgDbContactPair();
        PxgDbSolveOutput solveOut = PxgDbSolveOutput();
        contactPair.readContactPrepRigidParticle(blockBase, particleSystem[0].mData_PenBiasClamp, tIdx);

        const bool hasApplied = solveRigidDbContact(rigid, particle, contactPair, solveOut,
                                                    appliedForce.x, appliedForce.y, dt, /*wasActive*/ true,
                                                    /*checkOnlyActivity*/ false, isVelocityIteration,
                                                    /*withAdhesion*/ true);

        if (hasApplied)
        {
            appliedForce.x = solveOut.accumulatedDeltaLambdaN;
            appliedForce.y += solveOut.deltaLambdaT;

            float3 delta;
            solveOut.computeDelta(contactPair, delta, dt);
            particle.writeParticle(gSv.deltaPos, workIndex, delta, solveOut);
            appliedForces[workIndex] = appliedForce;
        }
        else
        {
            deltaPos[workIndex] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
}

// ===========================================================================
// 4. ps_accumulateDeltaVParticleLaunch
//    CUDA ABI: (PxgParticleSystem* particleSystems, PxgDeformablePairInfo*
//      sortedContactInfos, PxU32* totalNumPairs, PxU32* rangeStart, PxU32* rangeEnd,
//      float4* deltaPos) -> 6 device pointers = 48-byte blob.
// ===========================================================================
struct PcAccumParams
{
    uint64_t particleSystems;    // @0  output (mAccumDeltaP)
    uint64_t sortedContactInfos; // @8  input
    uint64_t totalNumPairs;      // @16 input
    uint64_t rangeStart;         // @24 input
    uint64_t rangeEnd;           // @32 input
    uint64_t deltaPos;           // @40 input -> 48
}
[[vk::push_constant]]
ConstantBuffer<PcAccumParams> gAc;

[shader("compute")]
[numthreads(AD_BLOCK, 1, 1)]
void ps_accumulateDeltaVParticleLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint numTests = Ptr<uint>(gAc.totalNumPairs)[0];

    const uint nbBlocksRequired = (numTests + AD_BLOCK - 1u) / AD_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + AD_GRID - 1u) / AD_GRID;
    const uint idx = groupThreadId.x;

    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gAc.particleSystems);
    Ptr<PxgDeformablePairInfoGpu> contactInfos = Ptr<PxgDeformablePairInfoGpu>(gAc.sortedContactInfos);
    Ptr<uint> rangeStart = Ptr<uint>(gAc.rangeStart);
    Ptr<uint> rangeEnd = Ptr<uint>(gAc.rangeEnd);
    Ptr<float4> deltaPos = Ptr<float4>(gAc.deltaPos);

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * AD_BLOCK + idx + nbIterationsPerBlock * groupId.x * AD_BLOCK;
        if (workIndex >= numTests)
            return;

        const uint startIndex = rangeStart[workIndex];
        const uint endIndex = rangeEnd[workIndex];

        float4 accumulatedDeltaP = float4(0.0f, 0.0f, 0.0f, 0.0f);
        float fcount = 0.0f;
        for (uint j = startIndex; j < endIndex; ++j)
        {
            const float4 tDelta = deltaPos[j];
            accumulatedDeltaP.x += tDelta.x;
            accumulatedDeltaP.y += tDelta.y;
            accumulatedDeltaP.z += tDelta.z;
            fcount += tDelta.w;
        }

        const float recipCount = 1.0f / dpSelectMax(1.0f, fcount);
        accumulatedDeltaP.x *= recipCount;
        accumulatedDeltaP.y *= recipCount;
        accumulatedDeltaP.z *= recipCount;

        const uint64_t particleRef = contactInfos[startIndex].ref1;
        const uint tParticleSystemId = PxgGetParticleSystemId(particleRef);
        const uint tParticleId = PxgGetParticleIndex(particleRef);

        Ptr<float4> accumDeltaP = Ptr<float4>(systems[tParticleSystemId].mAccumDeltaP);
        float4 acc = accumDeltaP[tParticleId];
        acc.x += accumulatedDeltaP.x;
        acc.y += accumulatedDeltaP.y;
        acc.z += accumulatedDeltaP.z;
        acc.w += accumulatedDeltaP.w;
        accumDeltaP[tParticleId] = acc;
    }
}

// ===========================================================================
// 5. ps_accumulateFEMParticleDeltaVLaunch
//    The FEM twin of kernel 4.  CUDA ABI: (PxgParticleSystem* particleSystems,
//    PxgDeformablePairInfo* sortedContactInfo, PxU32* totalNumPairs, PxU32* rangeStart,
//    PxU32* rangeEnd, float4* deltaP) -> 6 pointers = 48-byte blob.  The key is ref0, the
//    particle side of a particle-deformable pair.
//
//    THE TWIN TRAP (unchanged in 5.12).  The reciprocal is NOT kernel 4's:
//        FEM      recipCount = fcount > 0.f ? 1.f / fcount : 0.f
//        particle recipCount = 1.f / PxMax(1.f, fcount)
//
//    5.12: after the accumulate, mAccumDeltaP[i].w = 0 resets the mass-split contact
//    count the deformable-particle pre-counts bumped (a plain store, one thread per
//    particle).  The += itself adds accumulatedDeltaP.w, which is never assigned after
//    its zero init (a faithful +0).
// ===========================================================================
struct PcFemAccumParams
{
    uint64_t particleSystems;   // @0  output (mAccumDeltaP)
    uint64_t sortedContactInfo; // @8  input
    uint64_t totalNumPairs;     // @16 input
    uint64_t rangeStart;        // @24 input
    uint64_t rangeEnd;          // @32 input
    uint64_t deltaP;            // @40 input -> 48
}
[[vk::push_constant]]
ConstantBuffer<PcFemAccumParams> gFAc;

[shader("compute")]
[numthreads(AD_BLOCK, 1, 1)]
void ps_accumulateFEMParticleDeltaVLaunch(uint3 groupThreadId : SV_GroupThreadID,
                                          uint3 groupId : SV_GroupID)
{
    const uint numTests = Ptr<uint>(gFAc.totalNumPairs)[0];

    const uint nbBlocksRequired = (numTests + AD_BLOCK - 1u) / AD_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + AD_GRID - 1u) / AD_GRID;
    const uint idx = groupThreadId.x;

    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gFAc.particleSystems);
    Ptr<PxgDeformablePairInfoGpu> infos = Ptr<PxgDeformablePairInfoGpu>(gFAc.sortedContactInfo);
    Ptr<uint> rangeStart = Ptr<uint>(gFAc.rangeStart);
    Ptr<uint> rangeEnd = Ptr<uint>(gFAc.rangeEnd);
    Ptr<float4> deltaP = Ptr<float4>(gFAc.deltaP);

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * AD_BLOCK + idx + nbIterationsPerBlock * groupId.x * AD_BLOCK;
        if (workIndex >= numTests)
            return;

        const uint startIndex = rangeStart[workIndex];
        const uint endIndex = rangeEnd[workIndex];

        float4 accumulatedDeltaP = float4(0.0f, 0.0f, 0.0f, 0.0f);
        float fcount = 0.0f;
        for (uint j = startIndex; j < endIndex; ++j)
        {
            const float4 tDelta = deltaP[j];
            accumulatedDeltaP.x += tDelta.x;
            accumulatedDeltaP.y += tDelta.y;
            accumulatedDeltaP.z += tDelta.z;
            fcount += tDelta.w;
        }

        const float recipCount = fcount > 0.0f ? 1.0f / fcount : 0.0f;
        accumulatedDeltaP.x *= recipCount;
        accumulatedDeltaP.y *= recipCount;
        accumulatedDeltaP.z *= recipCount;

        const uint64_t particleRef = infos[startIndex].ref0;
        const uint tParticleSystemId = PxgGetParticleSystemId(particleRef);
        const uint tParticleId = PxgGetParticleIndex(particleRef);

        Ptr<float4> accumDeltaP = Ptr<float4>(systems[tParticleSystemId].mAccumDeltaP);
        float4 acc = accumDeltaP[tParticleId];
        acc.x += accumulatedDeltaP.x;
        acc.y += accumulatedDeltaP.y;
        acc.z += accumulatedDeltaP.z;
        acc.w += accumulatedDeltaP.w;
        // The mass-split count reset: the store of .w follows the += store.
        acc.w = 0.0f;
        accumDeltaP[tParticleId] = acc;
    }
}
