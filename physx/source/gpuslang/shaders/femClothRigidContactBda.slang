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

// femClothRigidContactBda.slang -- the cloth-vs-rigid CONTACT chain, PhysX 5.12:
//
//   cloth_rigidContactPrepareLaunch                   (FEMClothConstraintPrep.cu) turns
//                                                     each sorted contact into one lane of a
//                                                     PxgRigidDbContactBlock
//                                                     (prepareRigidDbContact).
//   cloth_queryRigidClothContactReferenceCountLaunch  (FEMClothExternalSolve.cu) the
//                                                     mass-splitting pre-count: decides each
//                                                     contact's activity (latched in the
//                                                     pair's ACTIVE flag), bumps mDeltaPos.w
//                                                     of the touched cloth vertices and the
//                                                     rigid's reference count.
//   cloth_solveRigidClothCollisionLaunch              (FEMClothExternalSolve.cu)
//                                                     solveRigidDbContact, the cloth scatter
//                                                     into mDeltaPos and the rigid writeback
//                                                     into rigidDeltaVel.
//
// THE 5.12 DATA.  The contacts arrive as SoA streams: PxgDeformablePairInfo {ref0 = rigid
// PxNodeIndex, ref1 = cloth PxgDeformableRef (a TRIANGLE, or a VERTEX for the vertex
// contact generators)}, the PxU16 rigid material stream and the barycentric stream.  The
// 5.6.1 PxgFemOtherContactInfo, the packed cloth id and PxgFemRigidConstraintBlock are
// gone, and with them prepareFEMContacts / FEMCollision (pxgSlangFemRigidContact).  A
// vertex ref is told by its feature type, not by barycentric.w as in 5.6.1.  A pair the
// remap could not resolve carries the INVALID flag and is skipped; the query latches
// activity in the ACTIVE flag (once active it stays active: a conservative reference
// count) and the solve reads it back.  PGS and TGS share one kernel each (isTGS); the TGS
// twins are gone, and the velocity reader takes solverBodyVelPool instead of the
// PxgSolverSharedDesc.
//
// SEMANTICS THAT ARE EASY TO LOSE:
//   - the prepare zeroes rigidLambdaNs (a float2 now) for every in-range contact, before
//     the validity guard;
//   - the normal handed to prepareRigidDbContact is the NEGATED narrowphase normal and
//     the penetration has the rest offset (contacts_restW.w) subtracted;
//   - the barycentric stream is read for a TRIANGLE ref only (prepare and readCloth);
//   - the solve writes BOTH rigidDeltaVel slots of every valid contact, inactive ones
//     included (writeContactDeltas' invMass == 0 sentinel writes zeros with count 0): the
//     rigid accumulate sums the whole range.  An INVALID pair writes nothing.
//
// DEVIATIONS (value-neutral):
//   - getGlobalRigidBodyId and the velocity reader read artiCoreDesc inside their
//     articulation arms only (the CUDA reads mMaxLinksPerArticulation at the call);
//   - the PxU16 rigid material index is read through its 32-bit word (PxgLoadU16);
//   - the strided work index is clamped to tNumContacts - 1 after the per-lane guard (the
//     P9.23 speculated-load fix); every admitted lane keeps its own index;
//   - invDt and isTGS of the prepare are unread, as in the CUDA.
//
// In PXG_NOCONTRACT_SHADERS: the responses are dot-of-cross chains and the solve is
// compared bit-exactly against a CPU transcription of the 5.12 CUDA (testFemClothSolve512).

import pxgSlangFEMClothDesc;                // PxgFEMClothGpu, clothAt
import pxgSlangSolverDesc;                  // PxgSolverCoreDescGpu
import pxgSlangPrepDesc;                    // PxgPrePrepDescGpu
import pxgSlangDeformableRef;               // PxgDeformablePairInfoGpu, PXG_RDBC_*, PxgGet*
import pxgSlangVelocityReader;              // PxgMakeVelocityReader
import pxgSlangDeformableUtils;             // prepareRigidDbContact, barycentricProjectTri
import pxgSlangDeformableAndParticleUtils;  // PxgRigidPart, PxgDeformablePart, solveRigidDbContact

// PxgSoftBodyKernelGridDim / BlockDim::SB_UPDATEROTATION (the host launches all three with
// them).  The bridge forwards only the grid, so the stride is rebuilt from these constants;
// RC_BLOCK equals [numthreads].
static const uint RC_BLOCK = 256u;
static const uint RC_GRID = 1024u;

// PxsMaterialData stride (PxsMaterialCore.h).
static const uint64_t RIGID_MATERIAL_STRIDE = 24;

// ===========================================================================
// cloth_rigidContactPrepareLaunch
//   (PxgFEMCloth* femClothes, float4* contacts_restW, float4* normalPens,
//    float4* barycentrics, const PxgDeformablePairInfo* contactInfos, PxU32* numContacts,
//    PxgRigidDbContactBlock* contactBlocks, PxgPrePrepDesc* preDesc,
//    PxgConstraintPrepareDesc* prepareDesc, float2* rigidLambdaNs, const PxReal invDt,
//    PxgSolverSharedDescBase* sharedDesc, bool isTGS)
//   -> {8 x 10, 4, 8, 1} = 104-byte blob.
// ===========================================================================
struct PrepContactParams
{
    uint64_t femClothes;          // @0
    uint64_t contacts_restW;      // @8   xyz = point, w = rest offset
    uint64_t normalPens;          // @16  xyz = normal, w = penetration
    uint64_t barycentrics;        // @24
    uint64_t contactInfos;        // @32
    uint64_t numContacts;         // @40
    uint64_t contactBlocks;       // @48  output
    uint64_t preDesc;             // @56
    uint64_t prepareDesc;         // @64
    uint64_t rigidLambdaNs;       // @72  output (zeroed)
    float invDt;                  // @80  (unread, as in CUDA)
    uint _pad84;                  // @84
    uint64_t sharedDesc;          // @88  articulation arm only
    uint isTGS;                   // @96  bool byte0 (unread, as in CUDA)
    uint _pad100;                 // @100 -> 104
}
[[vk::push_constant]]
ConstantBuffer<PrepContactParams> gPrep;

[shader("compute")]
[numthreads(256, 1, 1)]   // == RC_BLOCK
void cloth_rigidContactPrepareLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint tNumContacts = Ptr<uint>(gPrep.numContacts)[0];
    const uint nbBlocksRequired = (tNumContacts + RC_BLOCK - 1u) / RC_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + RC_GRID - 1u) / RC_GRID;
    const uint idx = groupThreadId.x;

    Ptr<PxgDeformablePairInfoGpu> contactInfos = Ptr<PxgDeformablePairInfoGpu>(gPrep.contactInfos);
    Ptr<float2> rigidLambdaNs = Ptr<float2>(gPrep.rigidLambdaNs);
    const uint64_t solverBodyIndicesPtr = Ptr<PxgPrePrepDescGpu>(gPrep.preDesc)[0].solverBodyIndices;

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint rawWorkIndex = i * RC_BLOCK + idx + nbIterationsPerBlock * groupId.x * RC_BLOCK;
        if (rawWorkIndex >= tNumContacts)
            return;
        // P9.23: the guard above is per-lane; the clamp keeps the speculated addresses of
        // the excluded lanes inside the allocation.  An admitted lane keeps its index.
        const uint workIndex = min(rawWorkIndex, tNumContacts - 1u);

        rigidLambdaNs[workIndex] = float2(0.0f, 0.0f);

        const PxgDeformablePairInfoGpu contactInfo = contactInfos[workIndex];
        const uint64_t block = gPrep.contactBlocks + uint64_t(workIndex / 32u) * PXG_RDBC_STRIDE;

        // First one is rigid body.
        const uint64_t rigidId = contactInfo.ref0;   // PxgGetRigidNode

        if (!PxgIsValidPairInfo(contactInfo))
            continue;

        // Second one is cloth.
        Ptr<PxgFEMClothGpu> cloth = clothAt(gPrep.femClothes, PxgGetDeformableId(contactInfo.ref1));
        const uint elementId = PxgGetDeformableFeatureId(contactInfo.ref1);
        const bool clothIsVertex = PxgGetDeformableFeatureType(contactInfo.ref1) == PXG_DREF_eVERTEX;

        const uint64_t accumDelta_invMass = cloth[0].mAccumulatedDeltaPos;

        const float4 contact_restW = Ptr<float4>(gPrep.contacts_restW)[workIndex];
        const float4 normal_pen = Ptr<float4>(gPrep.normalPens)[workIndex];

        const float3 p = float3(contact_restW.x, contact_restW.y, contact_restW.z);

        float4 deltaP;
        if (!clothIsVertex)
        {
            const float4 barycentric = Ptr<float4>(gPrep.barycentrics)[workIndex];
            const uint4 vertexIndices = Ptr<uint4>(cloth[0].mTriangleVertexIndices_ownedEdges)[elementId];
            deltaP = barycentricProjectTri(vertexIndices, accumDelta_invMass, barycentric);
        }
        else
        {
            deltaP = Ptr<float4>(accumDelta_invMass)[elementId];
        }

        const float3 normal = float3(-normal_pen.x, -normal_pen.y, -normal_pen.z);
        const float pen = normal_pen.w - contact_restW.w;

        const float3 delta = float3(deltaP.x, deltaP.y, deltaP.z);

        prepareRigidDbContact(block, workIndex & 31u, normal, gPrep.sharedDesc, p, pen, delta, rigidId,
                              gPrep.prepareDesc, solverBodyIndicesPtr, cloth[0].mPenBiasClamp);
    }
}

// ===========================================================================
// cloth_queryRigidClothContactReferenceCountLaunch
//   (PxgFEMCloth* femClothes, PxgDeformablePairInfo* contactInfos, const float4* barycentrics,
//    PxgRigidDbContactBlock* contactBlocks, PxU32* numContacts, PxgPrePrepDesc* prePrepDesc,
//    PxgSolverCoreDesc* solverCoreDesc, PxgArticulationCoreDesc* artiCoreDesc,
//    float4* solverBodyVelPool, const PxReal dt, float2* appliedForces,
//    PxU32* rigidBodyRefCounts, bool isTGS, bool isVelocityIteration)
//   -> {8 x 9, 4, 8, 8, 1, 1} = 104-byte blob (dt @72, appliedForces @80,
//   rigidBodyRefCounts @88, isTGS byte @96, isVelocityIteration byte @97).
// ===========================================================================
struct QueryContactParams
{
    uint64_t femClothes;          // @0
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
ConstantBuffer<QueryContactParams> gQuery;

[shader("compute")]
[numthreads(256, 1, 1)]   // == RC_BLOCK
void cloth_queryRigidClothContactReferenceCountLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint numSolverBodies = Ptr<PxgSolverCoreDescGpu>(gQuery.solverCoreDesc)[0].numSolverBodies;
    const uint tNumContacts = Ptr<uint>(gQuery.numContacts)[0];
    const uint nbBlocksRequired = (tNumContacts + RC_BLOCK - 1u) / RC_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + RC_GRID - 1u) / RC_GRID;
    const uint idx = groupThreadId.x;

    const bool isTGS = (gQuery.isTGS_isVelocityIteration & 0xffu) != 0u;
    const bool isVelocityIteration = ((gQuery.isTGS_isVelocityIteration >> 8u) & 0xffu) != 0u;
    const float dt = gQuery.dt;

    Ptr<PxgDeformablePairInfoGpu> contactInfos = Ptr<PxgDeformablePairInfoGpu>(gQuery.contactInfos);
    Ptr<float2> appliedForces = Ptr<float2>(gQuery.appliedForces);

    const PxgVelocityReaderGpu velocityReader = PxgMakeVelocityReader(gQuery.prePrepDesc, gQuery.solverCoreDesc,
                                                                      gQuery.artiCoreDesc, gQuery.solverBodyVelPool,
                                                                      numSolverBodies);

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint rawWorkIndex = i * RC_BLOCK + idx + nbIterationsPerBlock * groupId.x * RC_BLOCK;
        if (rawWorkIndex >= tNumContacts)
            return;
        const uint workIndex = min(rawWorkIndex, tNumContacts - 1u);   // P9.23, see the prepare

        const PxgDeformablePairInfoGpu contactInfo = contactInfos[workIndex];
        const bool wasActive = PxgIsActivePairInfo(contactInfo);

        const uint64_t block = gQuery.contactBlocks + uint64_t(workIndex / 32u) * PXG_RDBC_STRIDE;
        const uint threadIndexInWarp = workIndex & 31u;

        // First actor: rigid body.
        const uint64_t rigidId = contactInfo.ref0;

        if (PxgIsValidPairInfo(contactInfo))
        {
            // Only the invMass is wanted from the block here.
            const int globalRigidBodyId = getGlobalRigidBodyId(gQuery.prePrepDesc, rigidId, numSolverBodies,
                                                               gQuery.artiCoreDesc);
            const float rigidInvMass = Ptr<float4>(block + PXG_RDBC_fricTan0_invMass0)[threadIndexInWarp].w;

            const bool checkOnlyActivity = true;
            bool isActive = false;

            // Second actor: cloth.
            Ptr<PxgFEMClothGpu> cloth = clothAt(gQuery.femClothes, PxgGetDeformableId(contactInfo.ref1));

            PxgRigidPart rigid = PxgRigidPart();
            PxgDeformablePart db = PxgDeformablePart();
            PxgDbContactPair contactPair = PxgDbContactPair();
            PxgDbSolveOutput solveOut = PxgDbSolveOutput();

            db.readCloth(cloth, contactInfo.ref1, gQuery.barycentrics, workIndex, 0UL,
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
                    Ptr<uint64_t>(gQuery.contactInfos + uint64_t(workIndex) * PXG_PAIRINFO_STRIDE
                                  + PXG_PAIRINFO_ref1)[0] = PxgSetActiveRef1(contactInfo.ref1, true);
            }

            if (isActive)
            {
                // Update cloth.
                db.bumpClothRefCountPart(cloth);

                // Update rigid body.
                if (rigidIsRefCounted(globalRigidBodyId, rigidInvMass))
                {
                    uint prev;
                    InterlockedAdd(Ptr<uint>(gQuery.rigidBodyRefCounts)[uint(globalRigidBodyId)], 1u, prev);
                }
            }
        }
    }
}

// ===========================================================================
// cloth_solveRigidClothCollisionLaunch
//   (PxgFEMCloth* femClothes, const PxgDeformablePairInfo* contactInfos,
//    const PxU16* contactRigidMaterials, const float4* barycentrics,
//    PxgRigidDbContactBlock* contactBlocks, PxU32* numContacts, PxgPrePrepDesc* prePrepDesc,
//    PxgSolverCoreDesc* solverCoreDesc, PxgArticulationCoreDesc* artiCoreDesc,
//    float4* solverBodyVelPool, float4* rigidDeltaVel, float2* appliedForces,
//    PxU32* rigidBodyRefCounts, const PxReal dt, PxsDeformableSurfaceMaterialData* materials,
//    const PxsMaterialData* rigidBodyMaterials, bool isTGS, bool isVelocityIteration)
//   -> {8 x 13, 4, 8, 8, 1, 1} = 136-byte blob (dt @104, materials @112,
//   rigidBodyMaterials @120, isTGS byte @128, isVelocityIteration byte @129).
// ===========================================================================
struct SolveContactParams
{
    uint64_t femClothes;             // @0
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
    uint64_t materials;              // @112 PxsDeformableSurfaceMaterialData*
    uint64_t rigidBodyMaterials;     // @120 PxsMaterialData*
    uint isTGS_isVelocityIteration;  // @128 byte0 = isTGS, byte1 = isVelocityIteration
    uint _pad132;                    // @132 -> 136
}
[[vk::push_constant]]
ConstantBuffer<SolveContactParams> gSolve;

[shader("compute")]
[numthreads(256, 1, 1)]   // == RC_BLOCK
void cloth_solveRigidClothCollisionLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint numSolverBodies = Ptr<PxgSolverCoreDescGpu>(gSolve.solverCoreDesc)[0].numSolverBodies;
    const uint tNumContacts = Ptr<uint>(gSolve.numContacts)[0];
    const uint nbBlocksRequired = (tNumContacts + RC_BLOCK - 1u) / RC_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + RC_GRID - 1u) / RC_GRID;
    const uint idx = groupThreadId.x;

    const bool isTGS = (gSolve.isTGS_isVelocityIteration & 0xffu) != 0u;
    const bool isVelocityIteration = ((gSolve.isTGS_isVelocityIteration >> 8u) & 0xffu) != 0u;
    const float dt = gSolve.dt;

    Ptr<PxgDeformablePairInfoGpu> contactInfos = Ptr<PxgDeformablePairInfoGpu>(gSolve.contactInfos);
    Ptr<float2> appliedForces = Ptr<float2>(gSolve.appliedForces);

    const PxgVelocityReaderGpu velocityReader = PxgMakeVelocityReader(gSolve.prePrepDesc, gSolve.solverCoreDesc,
                                                                      gSolve.artiCoreDesc, gSolve.solverBodyVelPool,
                                                                      numSolverBodies);

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint rawWorkIndex = i * RC_BLOCK + idx + nbIterationsPerBlock * groupId.x * RC_BLOCK;
        if (rawWorkIndex >= tNumContacts)
            return;
        const uint workIndex = min(rawWorkIndex, tNumContacts - 1u);   // P9.23, see the prepare

        const PxgDeformablePairInfoGpu contactInfo = contactInfos[workIndex];
        const bool isActive = PxgIsActivePairInfo(contactInfo);

        const uint64_t block = gSolve.contactBlocks + uint64_t(workIndex / 32u) * PXG_RDBC_STRIDE;
        const uint threadIndexInWarp = workIndex & 31u;

        // First actor: rigid body.
        const uint64_t rigidId = contactInfo.ref0;

        if (PxgIsValidPairInfo(contactInfo))
        {
            const int globalRigidBodyId = getGlobalRigidBodyId(gSolve.prePrepDesc, rigidId, numSolverBodies,
                                                               gSolve.artiCoreDesc);
            const float rigidInvMass = Ptr<float4>(block + PXG_RDBC_fricTan0_invMass0)[threadIndexInWarp].w;

            const bool checkOnlyActivity = false;

            PxgRigidPart rigid = PxgRigidPart();
            PxgDeformablePart db = PxgDeformablePart();
            PxgDbContactPair contactPair = PxgDbContactPair();
            PxgDbSolveOutput solveOut = PxgDbSolveOutput();

            if (isActive)
            {
                // Second actor: cloth.
                Ptr<PxgFEMClothGpu> cloth = clothAt(gSolve.femClothes, PxgGetDeformableId(contactInfo.ref1));

                const uint rigidMaterialIndex = PxgLoadU16(gSolve.contactRigidMaterials, workIndex);
                rigid.readBodyProperties(rigidId, globalRigidBodyId, rigidInvMass, gSolve.rigidBodyRefCounts,
                                         gSolve.rigidBodyMaterials + uint64_t(rigidMaterialIndex) * RIGID_MATERIAL_STRIDE);
                db.readCloth(cloth, contactInfo.ref1, gSolve.barycentrics, workIndex, gSolve.materials,
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
                    // velocity iterations; cloth_applyExternalDeltasLaunch applies it.
                    float3 delta;
                    solveOut.computeDelta(contactPair, delta, dt);
                    appliedForces[workIndex] = float2(solveOut.accumulatedDeltaLambdaN,
                                                      appliedForces[workIndex].y + solveOut.deltaLambdaT);

                    // Update cloth: .xyz weighted by the refCount-inflated vertexInvMasses.
                    db.writeCloth(cloth, delta);
                }
            }

            // Update rigid body.  Every valid slot is written every iteration, inactive
            // included: invMass is still zero on that path, the no-op sentinel.
            rigid.writeContactDeltas(gSolve.rigidDeltaVel, rigidId, contactPair, solveOut, workIndex,
                                     workIndex + tNumContacts);
        }
    }
}
