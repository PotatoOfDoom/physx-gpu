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

// psContactSolveRigidBda.slang -- the RIGID side of the particle/rigid contact solve,
// ps_solvePCOutputRigidDeltaVLaunch (gpusimulationcontroller/src/CUDA/particlesystem.cu,
// PhysX 5.12).
//
// It walks the by-RIGID sorted contacts, solves each one with the same
// solveRigidDbContact (with adhesion) the particle half runs, and writes the per-contact
// rigid reaction through PxgRigidPart::writeContactDeltas: deltaVel[i] = (linear delta *
// refCount, count) and deltaVel[i + n] = (angular delta * refCount, 0), which
// accumulateDeltaVRigidFirst/SecondLaunch (rigidDeltaAccumBda) fold into the solver body.
// Launch PxgParticleSystemCore.cpp solvePrimitiveCollisionForRigids, block PS_COLLISION
// 256, grid PS_COLLISION 1024.
//
// 5.12 changes against 5.6.1:
//   * the TGS twin ps_solvePCOutputRigidDeltaVTGSLaunch is gone; isTGS selects the
//     velocity read;
//   * the contacts are the SoA PxgDeformablePairInfo stream, the block is
//     PxgRigidParticleContactBlock (unit responses, no velMultipliers, no ids);
//   * mass splitting: the rigid response is scaled by the rigid's PC contact count
//     (ps_queryPCContactReferenceCountLaunch) and the writeback re-inflates by it;
//   * the depenetration clamp is the pair max of penBiasClampRigid and the particle
//     system's mPenBiasClamp (setMaxDepenetrationVelocity now honoured);
//   * a STATIC rigid skips the lane without writing; a kinematic (invMass 0) writes
//     zeros into both deltaVel slots and the applied force.
//
// See psContactSolveBda for the value-neutral artiCoreDesc deviation.  In
// PXG_NOCONTRACT_SHADERS.

import pxgSlangParticleDesc;               // PxgParticleSystemGpu, PXG_PARTICLE_PHASE_GROUP_MASK
import pxgSlangSolverDesc;                 // PxgSolverCoreDescGpu
import pxgSlangDeformableRef;              // PxgDeformablePairInfoGpu, PXG_RPC_*
import pxgSlangVelocityReader;             // PxgMakeVelocityReader
import pxgSlangDeformableAndParticleUtils; // PxgRigidPart, PxgDeformablePart, solveRigidDbContact

static const uint PC_BLOCK = 256u;   // PS_COLLISION block
static const uint PC_GRID = 1024u;   // PS_COLLISION grid

// ===========================================================================
// ps_solvePCOutputRigidDeltaVLaunch
//    CUDA ABI: (PxgParticleSystem* particleSystems, PxgDeformablePairInfo*
//      sortedContactInfos, PxgRigidParticleContactBlock* contactBlocks,
//      PxU32* numContacts, PxgPrePrepDesc* prePrepDesc, PxgSolverCoreDesc* solverCoreDesc,
//      float4* solverBodyVelPool, float4* deltaVel, float2* appliedForces,
//      PxU32* rigidRefCount, PxReal dt, bool isVelocityIteration, bool isTGS,
//      PxgArticulationCoreDesc* artiCoreDesc)
//    -> {8x10, 4, 1, 1, 8} = 96-byte blob, the particle half's layout.
// ===========================================================================
struct PcRigidSolveParams
{
    uint64_t particleSystems;      // @0
    uint64_t sortedContactInfos;   // @8
    uint64_t contactBlocks;        // @16
    uint64_t numContacts;          // @24
    uint64_t prePrepDesc;          // @32
    uint64_t solverCoreDesc;       // @40
    uint64_t solverBodyVelPool;    // @48
    uint64_t deltaVel;             // @56 output
    uint64_t appliedForces;        // @64
    uint64_t rigidRefCount;        // @72
    float dt;                      // @80
    uint flagsWord;                // @84 byte0 = isVelocityIteration, byte1 = isTGS
    uint64_t artiCoreDesc;         // @88 -> 96
}
[[vk::push_constant]]
ConstantBuffer<PcRigidSolveParams> gRs;

[shader("compute")]
[numthreads(PC_BLOCK, 1, 1)]
void ps_solvePCOutputRigidDeltaVLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const bool isVelocityIteration = (gRs.flagsWord & 0xffu) != 0u;
    const bool isTGS = ((gRs.flagsWord >> 8u) & 0xffu) != 0u;
    const float dt = gRs.dt;

    const uint numSolverBodies = Ptr<PxgSolverCoreDescGpu>(gRs.solverCoreDesc)[0].numSolverBodies;
    const PxgVelocityReaderGpu velocityReader = PxgMakeVelocityReader(gRs.prePrepDesc, gRs.solverCoreDesc,
                                                                      gRs.artiCoreDesc, gRs.solverBodyVelPool,
                                                                      numSolverBodies);

    const uint tNumContacts = Ptr<uint>(gRs.numContacts)[0];

    const uint nbBlocksRequired = (tNumContacts + PC_BLOCK - 1u) / PC_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + PC_GRID - 1u) / PC_GRID;
    const uint idx = groupThreadId.x;

    Ptr<PxgDeformablePairInfoGpu> contactInfos = Ptr<PxgDeformablePairInfoGpu>(gRs.sortedContactInfos);
    Ptr<float4> deltaVel = Ptr<float4>(gRs.deltaVel);
    Ptr<float2> appliedForces = Ptr<float2>(gRs.appliedForces);

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * PC_BLOCK + idx + nbIterationsPerBlock * groupId.x * PC_BLOCK;
        if (workIndex >= tNumContacts)
            return;

        const uint64_t blockBase = gRs.contactBlocks + uint64_t(workIndex / 32u) * PXG_RPC_STRIDE;
        const uint tIdx = workIndex & 31u;

        const uint64_t rigidId = contactInfos[workIndex].ref0;

        // "TODO - need to figure out how to make this work for articulation links!" (upstream)
        if (PxgNodeIsStaticBody(rigidId))
            continue;

        const float rigidInvMass = Ptr<float4>(blockBase + PXG_RPC_fricTan0_invMass0)[tIdx].w;
        if (rigidInvMass == 0.0f)
        {
            deltaVel[workIndex] = float4(0.0f, 0.0f, 0.0f, 0.0f);
            deltaVel[workIndex + tNumContacts] = float4(0.0f, 0.0f, 0.0f, 0.0f);
            appliedForces[workIndex] = float2(0.0f, 0.0f);
            continue;
        }

        const uint64_t particleRef = contactInfos[workIndex].ref1;
        const uint particleSystemId = PxgGetParticleSystemId(particleRef);
        const uint particleId = PxgGetParticleIndex(particleRef);

        Ptr<PxgParticleSystemGpu> particleSystem = Ptr<PxgParticleSystemGpu>(gRs.particleSystems
                                                       + uint64_t(particleSystemId) * PXG_PARTICLE_SYSTEM_STRIDE);

        const uint phase = Ptr<uint>(particleSystem[0].mSortedPhaseArray)[particleId];
        const uint group = phase & PXG_PARTICLE_PHASE_GROUP_MASK;
        const uint mi = pxgLoadPhaseToMaterial(particleSystem[0].mPhaseGroupToMaterialHandle, group);
        const uint64_t mat = particleSystem[0].mParticleMaterials
                             + uint64_t(mi) * uint64_t(particleSystem[0].mParticleMaterialStride);

        const float4 delta4 = Ptr<float4>(particleSystem[0].mSortedDeltaP)[particleId];

        float2 appliedForce = appliedForces[workIndex];

        // The PC contact count of this rigid from ps_queryPCContactReferenceCountLaunch (1 for
        // static or kinematic); friction is the particle material's (rigid material NULL, so
        // combineScalars eMAX returns it).
        PxgRigidPart rigid = PxgRigidPart();
        const int globalRigidBodyId = getGlobalRigidBodyId(gRs.prePrepDesc, rigidId, numSolverBodies, gRs.artiCoreDesc);
        rigid.readBodyProperties(rigidId, globalRigidBodyId, rigidInvMass, gRs.rigidRefCount, 0);
        rigid.readContactPrep(blockBase, tIdx);
        rigid.readVelocity(velocityReader, rigidId, isTGS);

        // The raw particle invMass on the rigid side (no inflation).
        PxgDeformablePart particle = PxgDeformablePart();
        particle.readParticle(particleSystem, particleId, delta4, mat, /*inflateByRefCount*/ false, dt,
                              isVelocityIteration);

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
            appliedForces[workIndex] = appliedForce;
        }

        // Every slot of a dynamic rigid's range is written, applied or not.
        rigid.writeContactDeltas(gRs.deltaVel, rigidId, contactPair, solveOut, workIndex, workIndex + tNumContacts);
    }
}
