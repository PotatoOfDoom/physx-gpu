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

// softBodyRigidContactBda.slang -- the softbody-vs-rigid CONTACT chain of softBody.cu,
// PhysX 5.12:
//
//   sb_rigidContactPrepareLaunch                  turns each sorted contact into one lane of
//                                                 a PxgRigidDbContactBlock (prepareRigidDbContact).
//   sb_queryRigidSoftContactReferenceCountLaunch  the mass-splitting pre-count: decides each
//                                                 contact's activity (latched in the pair's
//                                                 ACTIVE flag), bumps mSimDelta.w of the
//                                                 touched tet vertices and the rigid's
//                                                 reference count.
//   sb_solveRigidSoftCollisionLaunch              solveRigidDbContact, the deformable scatter
//                                                 into mSimDelta and the rigid writeback into
//                                                 rigidDeltaVel.
//
// THE 5.12 DATA.  The contacts arrive as SoA streams: PxgDeformablePairInfo {ref0 = rigid
// PxNodeIndex, ref1 = tetrahedron PxgDeformableRef}, the PxU16 rigid material stream and
// the barycentric stream (the 5.6.1 PxgFemOtherContactInfo / packed tet id and the
// barycentric slot of PxgFemRigidConstraintBlock are gone).  A pair the remap could not
// resolve carries the INVALID flag and is skipped; the query latches activity in the
// ACTIVE flag, once activated it stays active (a conservative reference count), and the
// solve reads it back.  PGS and TGS share one kernel each (isTGS); the TGS twins and their
// PxgSolverSharedDesc argument are gone, the velocity reader takes solverBodyVelPool.
//
// SEMANTICS THAT ARE EASY TO LOSE:
//   - the prepare zeroes rigidAppliedForces (a float2 now) for every in-range contact,
//     before the validity guard;
//   - the normal handed to prepareRigidDbContact is the NEGATED narrowphase normal, and
//     the penetration has the rest offset (contacts_restW.w) subtracted;
//   - the solve writes BOTH rigidDeltaVel slots of every valid contact, inactive ones
//     included (writeContactDeltas' invMass == 0 sentinel writes zeros with count 0): the
//     rigid accumulate sums the whole range.  An INVALID pair writes nothing.
//
// DEVIATIONS (value-neutral):
//   - getGlobalRigidBodyId and the velocity reader read artiCoreDesc inside their
//     articulation arms only (the CUDA reads mMaxLinksPerArticulation at the call);
//   - the PxU16 rigid material index is read through its 32-bit word;
//   - the strided work index is clamped to tNumContacts - 1 after the per-lane guard (the
//     P9.23 speculated-load fix); every admitted lane keeps its own index.
//   - invDt and isTGS of the prepare are unread, as in the CUDA.
//
// In PXG_NOCONTRACT_SHADERS: the responses are dot-of-cross chains and the solve is
// compared bit-exactly against a CPU transcription of the 5.12 CUDA.

import pxgSlangSoftBodyDesc;                // PxgSoftBodyGpu
import pxgSlangSolverDesc;                  // PxgSolverCoreDescGpu
import pxgSlangPrepDesc;                    // PxgPrePrepDescGpu
import pxgSlangDeformableRef;               // PxgDeformablePairInfoGpu, PXG_RDBC_*, PxgGet*
import pxgSlangVelocityReader;              // PxgMakeVelocityReader
import pxgSlangDeformableUtils;             // prepareRigidDbContact, barycentricProjectTet
import pxgSlangDeformableAndParticleUtils;  // PxgRigidPart, PxgDeformablePart, solveRigidDbContact

// PxgSoftBodyKernelGridDim / BlockDim::SB_UPDATEROTATION.  The bridge forwards only the
// grid, so the stride is rebuilt from these constants; RC_BLOCK equals [numthreads].
static const uint RC_BLOCK = 256u;
static const uint RC_GRID = 1024u;

// PxsMaterialData stride (PxsMaterialCore.h).
static const uint64_t RIGID_MATERIAL_STRIDE = 24;

// ===========================================================================
// sb_rigidContactPrepareLaunch
//   (PxgSoftBody* softbodies, float4* contacts_restW, float4* normalPens,
//    float4* barycentrics, PxgDeformablePairInfo* contactInfos, PxU32* numContacts,
//    PxgRigidDbContactBlock* contactBlocks, PxgPrePrepDesc* preDesc,
//    PxgConstraintPrepareDesc* prepareDesc, float2* rigidAppliedForces, const PxReal invDt,
//    PxgSolverSharedDescBase* sharedDesc, const bool isTGS)
//   -> {8 x 10, 4, 8, 1} = 104-byte blob.
// ===========================================================================
struct SbPrepContactParams
{
    uint64_t softbodies;          // @0
    uint64_t contacts_restW;      // @8   xyz = point, w = rest offset
    uint64_t normalPens;          // @16  xyz = normal, w = penetration
    uint64_t barycentrics;        // @24
    uint64_t contactInfos;        // @32
    uint64_t numContacts;         // @40
    uint64_t contactBlocks;       // @48  output
    uint64_t preDesc;             // @56
    uint64_t prepareDesc;         // @64
    uint64_t rigidAppliedForces;  // @72  output (zeroed)
    float invDt;                  // @80  (unread, as in CUDA)
    uint _pad84;                  // @84
    uint64_t sharedDesc;          // @88  articulation arm only
    uint isTGS;                   // @96  bool byte0 (unread, as in CUDA)
    uint _pad100;                 // @100 -> 104
}
[[vk::push_constant]]
ConstantBuffer<SbPrepContactParams> gSbPrep;

[shader("compute")]
[numthreads(256, 1, 1)]   // == RC_BLOCK
void sb_rigidContactPrepareLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint tNumContacts = Ptr<uint>(gSbPrep.numContacts)[0];
    const uint nbBlocksRequired = (tNumContacts + RC_BLOCK - 1u) / RC_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + RC_GRID - 1u) / RC_GRID;
    const uint idx = groupThreadId.x;

    Ptr<PxgSoftBodyGpu> softbodies = Ptr<PxgSoftBodyGpu>(gSbPrep.softbodies);
    Ptr<PxgDeformablePairInfoGpu> contactInfos = Ptr<PxgDeformablePairInfoGpu>(gSbPrep.contactInfos);
    Ptr<float2> rigidAppliedForces = Ptr<float2>(gSbPrep.rigidAppliedForces);
    const uint64_t solverBodyIndicesPtr = Ptr<PxgPrePrepDescGpu>(gSbPrep.preDesc)[0].solverBodyIndices;

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint rawWorkIndex = i * RC_BLOCK + idx + nbIterationsPerBlock * groupId.x * RC_BLOCK;
        if (rawWorkIndex >= tNumContacts)
            return;
        // P9.23: the guard above is per-lane; the clamp keeps the speculated addresses of
        // the excluded lanes inside the allocation.  An admitted lane keeps its index.
        const uint workIndex = min(rawWorkIndex, tNumContacts - 1u);

        rigidAppliedForces[workIndex] = float2(0.0f, 0.0f);

        const PxgDeformablePairInfoGpu contactInfo = contactInfos[workIndex];
        const uint64_t block = gSbPrep.contactBlocks + uint64_t(workIndex / 32u) * PXG_RDBC_STRIDE;

        // First actor: rigid body.
        const uint64_t rigidId = contactInfo.ref0;   // PxgGetRigidNode

        if (!PxgIsValidPairInfo(contactInfo))
            continue;

        // Second actor: soft body.
        Ptr<PxgSoftBodyGpu> softbody = sbAt(gSbPrep.softbodies, PxgGetDeformableId(contactInfo.ref1));
        const uint tetInd = PxgGetDeformableFeatureId(contactInfo.ref1);

        const uint4 tetrahedronIdx = Ptr<uint4>(softbody[0].mSimTetIndices)[tetInd];

        const float4 barycentric = Ptr<float4>(gSbPrep.barycentrics)[workIndex];
        const float4 contact_restW = Ptr<float4>(gSbPrep.contacts_restW)[workIndex];
        const float4 normal_pen = Ptr<float4>(gSbPrep.normalPens)[workIndex];

        const float3 p = float3(contact_restW.x, contact_restW.y, contact_restW.z);
        const float rest = contact_restW.w;

        const float4 delta_invMass1 = barycentricProjectTet(tetrahedronIdx, softbody[0].mSimDeltaPos, barycentric);
        const float3 delta = float3(delta_invMass1.x, delta_invMass1.y, delta_invMass1.z);

        const float3 normal = float3(-normal_pen.x, -normal_pen.y, -normal_pen.z);
        const float pen = normal_pen.w - rest;

        prepareRigidDbContact(block, workIndex & 31u, normal, gSbPrep.sharedDesc, p, pen, delta, rigidId,
                              gSbPrep.prepareDesc, solverBodyIndicesPtr, softbody[0].mPenBiasClamp);
    }
}

// ===========================================================================
// sb_queryRigidSoftContactReferenceCountLaunch
//   (PxgSoftBody* softbodies, PxgDeformablePairInfo* contactInfos, const float4* barycentrics,
//    PxgRigidDbContactBlock* contactBlocks, PxU32* numContacts, PxgPrePrepDesc* prePrepDesc,
//    PxgSolverCoreDesc* solverCoreDesc, PxgArticulationCoreDesc* artiCoreDesc,
//    float4* solverBodyVelPool, const PxReal dt, float2* appliedForces,
//    PxU32* rigidBodyRefCounts, bool isTGS, bool isVelocityIteration)
//   -> {8 x 9, 4, 8, 8, 1, 1} = 104-byte blob (dt @72, appliedForces @80,
//   rigidBodyRefCounts @88, isTGS byte @96, isVelocityIteration byte @97).
// ===========================================================================
struct SbQueryContactParams
{
    uint64_t softbodies;          // @0
    uint64_t contactInfos;        // @8   in/out (ACTIVE flag)
    uint64_t barycentrics;        // @16
    uint64_t contactBlocks;       // @24
    uint64_t numContacts;         // @32
    uint64_t prePrepDesc;         // @40
    uint64_t solverCoreDesc;      // @48
    uint64_t artiCoreDesc;        // @56
    uint64_t solverBodyVelPool;   // @64
    float dt;                     // @72
    uint _pad76;                  // @76
    uint64_t appliedForces;       // @80
    uint64_t rigidBodyRefCounts;  // @88  output (atomic)
    uint isTGS_isVelocityIteration; // @96 byte0 = isTGS, byte1 = isVelocityIteration
    uint _pad100;                 // @100 -> 104
}
[[vk::push_constant]]
ConstantBuffer<SbQueryContactParams> gSbQuery;

[shader("compute")]
[numthreads(256, 1, 1)]   // == RC_BLOCK
void sb_queryRigidSoftContactReferenceCountLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint numSolverBodies = Ptr<PxgSolverCoreDescGpu>(gSbQuery.solverCoreDesc)[0].numSolverBodies;
    const uint tNumContacts = Ptr<uint>(gSbQuery.numContacts)[0];
    const uint nbBlocksRequired = (tNumContacts + RC_BLOCK - 1u) / RC_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + RC_GRID - 1u) / RC_GRID;
    const uint idx = groupThreadId.x;

    const bool isTGS = (gSbQuery.isTGS_isVelocityIteration & 0xffu) != 0u;
    const bool isVelocityIteration = ((gSbQuery.isTGS_isVelocityIteration >> 8u) & 0xffu) != 0u;
    const float dt = gSbQuery.dt;

    Ptr<PxgSoftBodyGpu> softbodies = Ptr<PxgSoftBodyGpu>(gSbQuery.softbodies);
    Ptr<PxgDeformablePairInfoGpu> contactInfos = Ptr<PxgDeformablePairInfoGpu>(gSbQuery.contactInfos);
    Ptr<float2> appliedForces = Ptr<float2>(gSbQuery.appliedForces);

    const PxgVelocityReaderGpu velocityReader = PxgMakeVelocityReader(gSbQuery.prePrepDesc, gSbQuery.solverCoreDesc,
                                                                      gSbQuery.artiCoreDesc, gSbQuery.solverBodyVelPool,
                                                                      numSolverBodies);

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint rawWorkIndex = i * RC_BLOCK + idx + nbIterationsPerBlock * groupId.x * RC_BLOCK;
        if (rawWorkIndex >= tNumContacts)
            return;
        const uint workIndex = min(rawWorkIndex, tNumContacts - 1u);   // P9.23, see the prepare

        const PxgDeformablePairInfoGpu contactInfo = contactInfos[workIndex];
        const bool wasActive = PxgIsActivePairInfo(contactInfo);

        const uint64_t block = gSbQuery.contactBlocks + uint64_t(workIndex / 32u) * PXG_RDBC_STRIDE;
        const uint threadIndexInWarp = workIndex & 31u;

        // First actor: rigid body.
        const uint64_t rigidId = contactInfo.ref0;

        if (PxgIsValidPairInfo(contactInfo))
        {
            // Only the invMass is wanted from the block here.
            const int globalRigidBodyId = getGlobalRigidBodyId(gSbQuery.prePrepDesc, rigidId, numSolverBodies,
                                                               gSbQuery.artiCoreDesc);
            const float rigidInvMass = Ptr<float4>(block + PXG_RDBC_fricTan0_invMass0)[threadIndexInWarp].w;

            const bool checkOnlyActivity = true;
            bool isActive = false;

            // Second actor: soft body.
            Ptr<PxgSoftBodyGpu> softbody = sbAt(gSbQuery.softbodies, PxgGetDeformableId(contactInfo.ref1));

            PxgRigidPart rigid = PxgRigidPart();
            PxgDeformablePart db = PxgDeformablePart();
            PxgDbContactPair contactPair = PxgDbContactPair();
            PxgDbSolveOutput solveOut = PxgDbSolveOutput();

            db.readSoftBody(softbody, contactInfo.ref1, gSbQuery.barycentrics, workIndex, 0UL,
                            checkOnlyActivity, dt, isVelocityIteration);

            if (wasActive)
            {
                // Once activated, keep the contact pair active to maintain a conservative
                // reference count.
                isActive = true;
            }
            else
            {
                rigid.readBodyProperties(rigidId, globalRigidBodyId, rigidInvMass, 0UL, 0UL);
                rigid.readContactPrep(block, threadIndexInWarp);
                rigid.readVelocity(velocityReader, rigidId, isTGS);
                contactPair.readContactPrepRigidDb(block, threadIndexInWarp);
                const float2 applied = appliedForces[workIndex];
                isActive = solveRigidDbContact(rigid, db, contactPair, solveOut, applied.x, applied.y, dt,
                                               wasActive, checkOnlyActivity, isVelocityIteration,
                                               /*withAdhesion*/ false);

                if (isActive)
                    Ptr<uint64_t>(gSbQuery.contactInfos + uint64_t(workIndex) * PXG_PAIRINFO_STRIDE
                                  + PXG_PAIRINFO_ref1)[0] = PxgSetActiveRef1(contactInfo.ref1, true);
            }

            if (isActive)
            {
                // Update soft body.
                db.bumpSoftBodyRefCountPart(softbody);

                // Update rigid body.
                if (rigidIsRefCounted(globalRigidBodyId, rigidInvMass))
                {
                    uint prev;
                    InterlockedAdd(Ptr<uint>(gSbQuery.rigidBodyRefCounts)[uint(globalRigidBodyId)], 1u, prev);
                }
            }
        }
    }
}

// ===========================================================================
// sb_solveRigidSoftCollisionLaunch
//   (PxgSoftBody* softbodies, PxgDeformablePairInfo* contactInfos,
//    const PxU16* contactRigidMaterials, const float4* barycentrics,
//    PxgRigidDbContactBlock* contactBlocks, PxU32* numContacts, PxgPrePrepDesc* prePrepDesc,
//    PxgSolverCoreDesc* solverCoreDesc, PxgArticulationCoreDesc* artiCoreDesc,
//    float4* solverBodyVelPool, float4* rigidDeltaVel, float2* appliedForces,
//    PxU32* rigidBodyRefCounts, const PxReal dt, PxsDeformableVolumeMaterialData* materials,
//    const PxsMaterialData* rigidBodyMaterials, bool isTGS, bool isVelocityIteration)
//   -> {8 x 13, 4, 8, 8, 1, 1} = 136-byte blob (dt @104, materials @112,
//   rigidBodyMaterials @120, isTGS byte @128, isVelocityIteration byte @129).
// ===========================================================================
struct SbSolveContactParams
{
    uint64_t softbodies;             // @0
    uint64_t contactInfos;           // @8
    uint64_t contactRigidMaterials;  // @16  PxU16*
    uint64_t barycentrics;           // @24
    uint64_t contactBlocks;          // @32
    uint64_t numContacts;            // @40
    uint64_t prePrepDesc;            // @48
    uint64_t solverCoreDesc;         // @56
    uint64_t artiCoreDesc;           // @64
    uint64_t solverBodyVelPool;      // @72
    uint64_t rigidDeltaVel;          // @80  output
    uint64_t appliedForces;          // @88  in/out
    uint64_t rigidBodyRefCounts;     // @96
    float dt;                        // @104
    uint _pad108;                    // @108
    uint64_t materials;              // @112 PxsDeformableVolumeMaterialData*
    uint64_t rigidBodyMaterials;     // @120 PxsMaterialData*
    uint isTGS_isVelocityIteration;  // @128 byte0 = isTGS, byte1 = isVelocityIteration
    uint _pad132;                    // @132 -> 136
}
[[vk::push_constant]]
ConstantBuffer<SbSolveContactParams> gSbSolve;

[shader("compute")]
[numthreads(256, 1, 1)]   // == RC_BLOCK
void sb_solveRigidSoftCollisionLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint numSolverBodies = Ptr<PxgSolverCoreDescGpu>(gSbSolve.solverCoreDesc)[0].numSolverBodies;
    const uint tNumContacts = Ptr<uint>(gSbSolve.numContacts)[0];
    const uint nbBlocksRequired = (tNumContacts + RC_BLOCK - 1u) / RC_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + RC_GRID - 1u) / RC_GRID;
    const uint idx = groupThreadId.x;

    const bool isTGS = (gSbSolve.isTGS_isVelocityIteration & 0xffu) != 0u;
    const bool isVelocityIteration = ((gSbSolve.isTGS_isVelocityIteration >> 8u) & 0xffu) != 0u;
    const float dt = gSbSolve.dt;

    Ptr<PxgSoftBodyGpu> softbodies = Ptr<PxgSoftBodyGpu>(gSbSolve.softbodies);
    Ptr<PxgDeformablePairInfoGpu> contactInfos = Ptr<PxgDeformablePairInfoGpu>(gSbSolve.contactInfos);
    Ptr<float2> appliedForces = Ptr<float2>(gSbSolve.appliedForces);

    const PxgVelocityReaderGpu velocityReader = PxgMakeVelocityReader(gSbSolve.prePrepDesc, gSbSolve.solverCoreDesc,
                                                                      gSbSolve.artiCoreDesc, gSbSolve.solverBodyVelPool,
                                                                      numSolverBodies);

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint rawWorkIndex = i * RC_BLOCK + idx + nbIterationsPerBlock * groupId.x * RC_BLOCK;
        if (rawWorkIndex >= tNumContacts)
            return;
        const uint workIndex = min(rawWorkIndex, tNumContacts - 1u);   // P9.23, see the prepare

        const PxgDeformablePairInfoGpu contactInfo = contactInfos[workIndex];
        const bool isActive = PxgIsActivePairInfo(contactInfo);

        const uint64_t block = gSbSolve.contactBlocks + uint64_t(workIndex / 32u) * PXG_RDBC_STRIDE;
        const uint threadIndexInWarp = workIndex & 31u;

        // First actor: rigid body.
        const uint64_t rigidId = contactInfo.ref0;

        if (PxgIsValidPairInfo(contactInfo))
        {
            const int globalRigidBodyId = getGlobalRigidBodyId(gSbSolve.prePrepDesc, rigidId, numSolverBodies,
                                                               gSbSolve.artiCoreDesc);
            const float rigidInvMass = Ptr<float4>(block + PXG_RDBC_fricTan0_invMass0)[threadIndexInWarp].w;

            const bool checkOnlyActivity = false;

            PxgRigidPart rigid = PxgRigidPart();
            PxgDeformablePart db = PxgDeformablePart();
            PxgDbContactPair contactPair = PxgDbContactPair();
            PxgDbSolveOutput solveOut = PxgDbSolveOutput();

            if (isActive)
            {
                // Second actor: soft body.
                Ptr<PxgSoftBodyGpu> softbody = sbAt(gSbSolve.softbodies, PxgGetDeformableId(contactInfo.ref1));

                const uint rigidMaterialIndex = PxgLoadU16(gSbSolve.contactRigidMaterials, workIndex);
                rigid.readBodyProperties(rigidId, globalRigidBodyId, rigidInvMass, gSbSolve.rigidBodyRefCounts,
                                         gSbSolve.rigidBodyMaterials + uint64_t(rigidMaterialIndex) * RIGID_MATERIAL_STRIDE);
                db.readSoftBody(softbody, contactInfo.ref1, gSbSolve.barycentrics, workIndex, gSbSolve.materials,
                                checkOnlyActivity, dt, isVelocityIteration);

                rigid.readContactPrep(block, threadIndexInWarp);
                rigid.readVelocity(velocityReader, rigidId, isTGS);
                contactPair.readContactPrepRigidDb(block, threadIndexInWarp);
                const float2 applied = appliedForces[workIndex];
                const bool hasApplied = solveRigidDbContact(rigid, db, contactPair, solveOut, applied.x, applied.y, dt,
                                                            isActive, checkOnlyActivity, isVelocityIteration,
                                                            /*withAdhesion*/ false);

                if (hasApplied)
                {
                    // delta is a position delta in position iterations and velocity * dt in
                    // velocity iterations; sb_gm_applyExternalDeltasLaunch applies it.
                    float3 delta;
                    solveOut.computeDelta(contactPair, delta, dt);
                    appliedForces[workIndex] = float2(solveOut.accumulatedDeltaLambdaN,
                                                      appliedForces[workIndex].y + solveOut.deltaLambdaT);

                    // Update soft body: .xyz weighted by the refCount-inflated vertexInvMasses.
                    db.writeSoftBody(softbody, delta);
                }
            }

            // Update rigid body.  Every valid slot is written every iteration, inactive
            // included: invMass is still zero on that path, the no-op sentinel.
            rigid.writeContactDeltas(gSbSolve.rigidDeltaVel, rigidId, contactPair, solveOut, workIndex,
                                     workIndex + tNumContacts);
        }
    }
}
