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

// softBodyClothSolveBda -- the softbody-vs-cloth contact SOLVE of softBody.cu, PhysX 5.12:
//
//   sb_querySCContactReferenceCountLaunch  NEW in 5.12: the pre-count.  solveDbDbContact
//                                          with checkOnlyActivity decides the activity,
//                                          recorded on the pair; an active pair bumps
//                                          cloth.mDeltaPos.w and softbody.mSimDelta.w.
//   sb_solveOutputSCDeltaVLaunch           solveDbDbContact on PxgDeformablePart<PxVec3>
//                                          (cloth, side 0) and <PxVec4> (softbody, side 1),
//                                          +delta to the cloth, -delta to the softbody.  One
//                                          kernel for PGS and TGS.
//
// THE 5.12 DATA.  PxgDeformablePairInfo {ref0 = cloth vertex/triangle, ref1 = softbody
// tet}, the two barycentric streams and PxgDbDbContactBlock.  appliedForces is a float2
// (lambdaN, lambdaT); the 5.6.1 constraint block and lambdaNs buffer are gone.
//
// The strided work index is clamped to tNumContacts - 1 after the per-lane guard (P9.23,
// value-neutral).  In PXG_NOCONTRACT_SHADERS.

import pxgSlangSoftBodyDesc;                // PxgSoftBodyGpu, sbAt
import pxgSlangFEMClothDesc;                // PxgFEMClothGpu, clothAt
import pxgSlangDeformableRef;               // PxgDeformablePairInfoGpu, PXG_DBDBC_*
import pxgSlangDeformableAndParticleUtils;  // PxgDeformablePart, solveDbDbContact

// PxgSoftBodyKernelGridDim / BlockDim::SB_UPDATEROTATION.
static const uint SC_GRID = 1024u;
static const uint SC_BLOCK = 256u;

void scSetActive(uint64_t contactInfos, uint workIndex, uint64_t ref1, bool active)
{
    Ptr<uint64_t>(contactInfos + uint64_t(workIndex) * PXG_PAIRINFO_STRIDE + PXG_PAIRINFO_ref1)[0] =
        PxgSetActiveRef1(ref1, active);
}

// ===========================================================================
// sb_querySCContactReferenceCountLaunch (NEW in 5.12)
//   (PxgSoftBody* softbodies, PxgFEMCloth* clothes, PxgDeformablePairInfo* contactInfos,
//    const float4* barycentrics0, const float4* barycentrics1,
//    PxgDbDbContactBlock* contactBlocks, PxU32* numContacts, const PxReal dt,
//    bool isVelocityIteration)
//   -> {8 x 7, 4, 1} = 64-byte blob (dt @56, isVelocityIteration byte @60).
// ===========================================================================
struct SbScQueryParams
{
    uint64_t softbodies;          // @0
    uint64_t clothes;             // @8
    uint64_t contactInfos;        // @16  in/out (ACTIVE flag)
    uint64_t barycentrics0;       // @24
    uint64_t barycentrics1;       // @32
    uint64_t contactBlocks;       // @40
    uint64_t numContacts;         // @48
    float dt;                     // @56
    uint isVelocityIteration;     // @60  bool byte0 -> 64
}
[[vk::push_constant]]
ConstantBuffer<SbScQueryParams> gScQuery;

[shader("compute")]
[numthreads(256, 1, 1)]   // == SC_BLOCK
void sb_querySCContactReferenceCountLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint tNumContacts = Ptr<uint>(gScQuery.numContacts)[0];
    const uint nbBlocksRequired = (tNumContacts + SC_BLOCK - 1u) / SC_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + SC_GRID - 1u) / SC_GRID;
    const uint idx = groupThreadId.x;

    const float dt = gScQuery.dt;
    const bool isVelocityIteration = (gScQuery.isVelocityIteration & 0xffu) != 0u;

    Ptr<PxgDeformablePairInfoGpu> contactInfos = Ptr<PxgDeformablePairInfoGpu>(gScQuery.contactInfos);

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint rawWorkIndex = i * SC_BLOCK + idx + nbIterationsPerBlock * groupId.x * SC_BLOCK;
        if (rawWorkIndex >= tNumContacts)
            return;
        const uint workIndex = min(rawWorkIndex, tNumContacts - 1u);   // P9.23
        const uint threadIndexInWarp = workIndex & 31u;

        const PxgDeformablePairInfoGpu contactInfo = contactInfos[workIndex];
        const uint64_t block = gScQuery.contactBlocks + uint64_t(workIndex / 32u) * PXG_DBDBC_STRIDE;

        if (!PxgIsValidPairInfo(contactInfo))
        {
            scSetActive(gScQuery.contactInfos, workIndex, contactInfo.ref1, false);
            continue;
        }

        // Side 0: cloth.  Side 1: softbody tet.
        Ptr<PxgFEMClothGpu> cloth = clothAt(gScQuery.clothes, PxgGetDeformableId(contactInfo.ref0));
        Ptr<PxgSoftBodyGpu> softbody = sbAt(gScQuery.softbodies, PxgGetDeformableId(contactInfo.ref1));

        PxgDeformablePart dbCloth = PxgDeformablePart();
        PxgDeformablePart dbSb = PxgDeformablePart();
        dbCloth.readCloth(cloth, contactInfo.ref0, gScQuery.barycentrics0, workIndex, 0UL,
                          /*countReferenceOnly*/ true, dt, isVelocityIteration);
        dbSb.readSoftBody(softbody, contactInfo.ref1, gScQuery.barycentrics1, workIndex, 0UL,
                          /*checkOnlyActivity*/ true, dt, isVelocityIteration);

        PxgDbContactPair contactPair = PxgDbContactPair();
        PxgDbSolveOutput solveOut = PxgDbSolveOutput();
        contactPair.readContactPrepDbDb(block, PXG_DBDBC_maxPenBiasClamp, threadIndexInWarp);

        const bool isActive = solveDbDbContact(dbCloth, dbSb, contactPair, solveOut, 0.0f, 0.0f, 0.0f, dt,
                                               /*checkOnlyActivity*/ true, isVelocityIteration);

        scSetActive(gScQuery.contactInfos, workIndex, contactInfo.ref1, isActive);

        if (isActive)
        {
            // Integer-count refcount bump per side, gated |bc| > 1e-3.
            dbCloth.bumpClothRefCountPart(cloth);
            dbSb.bumpSoftBodyRefCountPart(softbody);
        }
    }
}

// ===========================================================================
// sb_solveOutputSCDeltaVLaunch
//   (PxgSoftBody* softbodies, PxgFEMCloth* clothes, PxgDeformablePairInfo* contactInfos,
//    const float4* barycentrics0, const float4* barycentrics1,
//    PxgDbDbContactBlock* contactBlocks, PxU32* numContacts, const PxReal dt,
//    float2* appliedForces, PxsDeformableVolumeMaterialData* sbMaterials,
//    PxsDeformableSurfaceMaterialData* clothMaterials, bool isVelocityIteration)
//   -> {8 x 7, 4, 8, 8, 8, 1} = 96-byte blob (dt @56, appliedForces @64, sbMaterials @72,
//   clothMaterials @80, isVelocityIteration byte @88).
// ===========================================================================
struct SbScSolveParams
{
    uint64_t softbodies;          // @0
    uint64_t clothes;             // @8
    uint64_t contactInfos;        // @16
    uint64_t barycentrics0;       // @24
    uint64_t barycentrics1;       // @32
    uint64_t contactBlocks;       // @40
    uint64_t numContacts;         // @48
    float dt;                     // @56
    uint _pad60;                  // @60
    uint64_t appliedForces;       // @64  in/out
    uint64_t sbMaterials;         // @72
    uint64_t clothMaterials;      // @80
    uint isVelocityIteration;     // @88  bool byte0
    uint _pad92;                  // @92 -> 96
}
[[vk::push_constant]]
ConstantBuffer<SbScSolveParams> gScSolve;

[shader("compute")]
[numthreads(256, 1, 1)]   // == SC_BLOCK
void sb_solveOutputSCDeltaVLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint tNumContacts = Ptr<uint>(gScSolve.numContacts)[0];
    const uint nbBlocksRequired = (tNumContacts + SC_BLOCK - 1u) / SC_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + SC_GRID - 1u) / SC_GRID;
    const uint idx = groupThreadId.x;

    const float dt = gScSolve.dt;
    const bool isVelocityIteration = (gScSolve.isVelocityIteration & 0xffu) != 0u;

    Ptr<PxgDeformablePairInfoGpu> contactInfos = Ptr<PxgDeformablePairInfoGpu>(gScSolve.contactInfos);
    Ptr<float2> appliedForces = Ptr<float2>(gScSolve.appliedForces);

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint rawWorkIndex = i * SC_BLOCK + idx + nbIterationsPerBlock * groupId.x * SC_BLOCK;
        if (rawWorkIndex >= tNumContacts)
            return;
        const uint workIndex = min(rawWorkIndex, tNumContacts - 1u);   // P9.23
        const uint threadIndexInWarp = workIndex & 31u;

        // The pre-count decided this on the same positions, this iteration.
        const PxgDeformablePairInfoGpu contactInfo = contactInfos[workIndex];
        if (!PxgIsActivePairInfo(contactInfo))
            continue;

        const uint64_t block = gScSolve.contactBlocks + uint64_t(workIndex / 32u) * PXG_DBDBC_STRIDE;

        Ptr<PxgFEMClothGpu> cloth = clothAt(gScSolve.clothes, PxgGetDeformableId(contactInfo.ref0));
        Ptr<PxgSoftBodyGpu> softbody = sbAt(gScSolve.softbodies, PxgGetDeformableId(contactInfo.ref1));

        PxgDeformablePart dbCloth = PxgDeformablePart();
        PxgDeformablePart dbSb = PxgDeformablePart();
        dbCloth.readCloth(cloth, contactInfo.ref0, gScSolve.barycentrics0, workIndex, gScSolve.clothMaterials,
                          /*countReferenceOnly*/ false, dt, isVelocityIteration);
        dbSb.readSoftBody(softbody, contactInfo.ref1, gScSolve.barycentrics1, workIndex, gScSolve.sbMaterials,
                          /*checkOnlyActivity*/ false, dt, isVelocityIteration);

        const float frictionCoefficient = (dbCloth.friction + dbSb.friction) * 0.5f;

        PxgDbContactPair contactPair = PxgDbContactPair();
        PxgDbSolveOutput solveOut = PxgDbSolveOutput();
        contactPair.readContactPrepDbDb(block, PXG_DBDBC_maxPenBiasClamp, threadIndexInWarp);

        float2 appliedForce = appliedForces[workIndex];   // (lambdaN_accum, lambdaT_accum)

        const bool hasApplied = solveDbDbContact(dbCloth, dbSb, contactPair, solveOut, appliedForce.x, appliedForce.y,
                                                 frictionCoefficient, dt, /*checkOnlyActivity*/ false,
                                                 isVelocityIteration);

        if (hasApplied)
        {
            float3 delta;
            solveOut.computeDelta(contactPair, delta, dt);

            // Scatter .xyz weighted by the refCount-inflated vertexInvMasses; the pre-count owns .w.
            dbCloth.writeCloth(cloth, delta);
            dbSb.writeSoftBody(softbody, float3(-delta.x, -delta.y, -delta.z));

            appliedForce.x = solveOut.accumulatedDeltaLambdaN;
            appliedForce.y += solveOut.deltaLambdaT;
            appliedForces[workIndex] = appliedForce;
        }
    }
}
