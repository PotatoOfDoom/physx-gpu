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

// softBodySelfSolveBda.slang -- the SOFTBODY-vs-SOFTBODY contact chain of softBody.cu,
// PhysX 5.12 (self collision and softbody pairs share it):
//
//   sb_softbodyContactPrepareLaunch       clamps the contact count to maxContacts (and
//                                         writes it back), zeroes the applied forces and
//                                         writes one PxgDbDbContactBlock lane: the NEGATED
//                                         narrowphase normal, the penetration re-referenced
//                                         to the start of the step, the pair max of the two
//                                         penBiasClamps.
//   sb_querySSContactReferenceCountLaunch NEW in 5.12: the pre-count.  Activity is
//                                         recomputed every iteration (solveDbDbContact with
//                                         checkOnlyActivity) and recorded on the pair; an
//                                         active pair bumps mSimDelta.w on both tets.
//   sb_solveOutputSSDeltaVLaunch          solveDbDbContact on the refCount-inflated inverse
//                                         masses, +delta to side 0, -delta to side 1.  One
//                                         kernel for PGS and TGS (5.12 deleted the TGS twin).
//
// THE 5.12 DATA.  PxgDeformablePairInfo {ref0, ref1} (both tetrahedron refs), the two
// barycentric streams and PxgDbDbContactBlock {normal_pen[32], maxPenBiasClamp[32]}.  The
// 5.6.1 PxgFemFemContactInfo, PxgSoftBodySoftBodyConstraintBlock (velMultiplier, friction,
// barycentrics) and the lambdaN accumulator are gone; appliedForces is a float2
// (lambdaN, lambdaT).
//
// DEVIATIONS (value-neutral): the strided work index is clamped to tNumContacts - 1 after
// the per-lane guard (P9.23).  `contacts` is unread, as in the CUDA.
//
// In PXG_NOCONTRACT_SHADERS: the barycentric blends and the solve are mul-feeding-add
// chains compared bit-exactly against a CPU transcription of the 5.12 CUDA.

import pxgSlangSoftBodyDesc;                // PxgSoftBodyGpu, sbAt
import pxgSlangDeformableRef;               // PxgDeformablePairInfoGpu, PXG_DBDBC_*
import pxgSlangDeformableUtils;             // barycentricProjectTet
import pxgSlangDeformableAndParticleUtils;  // PxgDeformablePart, solveDbDbContact

// PxgSoftBodyKernelGridDim / BlockDim::SB_UPDATEROTATION.
static const uint SS_GRID = 1024u;
static const uint SS_BLOCK = 256u;

void ssSetActive(uint64_t contactInfos, uint workIndex, uint64_t ref1, bool active)
{
    Ptr<uint64_t>(contactInfos + uint64_t(workIndex) * PXG_PAIRINFO_STRIDE + PXG_PAIRINFO_ref1)[0] =
        PxgSetActiveRef1(ref1, active);
}

// ===========================================================================
// sb_softbodyContactPrepareLaunch
//   (PxgSoftBody* softbodies, float4* contacts, float4* normalPens, float4* barycentrics0,
//    float4* barycentrics1, PxgDeformablePairInfo* contactInfos, PxU32* numContacts,
//    PxgDbDbContactBlock* contactBlocks, float2* softBodyAppliedForces, const PxU32 maxContacts)
//   -> {8 x 9, 4} = 80-byte blob.
// ===========================================================================
struct SbSsPrepParams
{
    uint64_t softbodies;             // @0
    uint64_t contacts;               // @8   (unread, as in CUDA)
    uint64_t normalPens;             // @16
    uint64_t barycentrics0;          // @24
    uint64_t barycentrics1;          // @32
    uint64_t contactInfos;           // @40
    uint64_t numContacts;            // @48  in/out (clamped)
    uint64_t contactBlocks;          // @56  output
    uint64_t softBodyAppliedForces;  // @64  output (zeroed)
    uint maxContacts;                // @72
    uint _pad76;                     // @76 -> 80
}
[[vk::push_constant]]
ConstantBuffer<SbSsPrepParams> gSsPrep;

[shader("compute")]
[numthreads(256, 1, 1)]   // == SS_BLOCK
void sb_softbodyContactPrepareLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint tNumContacts = min(gSsPrep.maxContacts, Ptr<uint>(gSsPrep.numContacts)[0]);

    if (groupThreadId.x == 0u && groupId.x == 0u)
        Ptr<uint>(gSsPrep.numContacts)[0] = tNumContacts;

    const uint nbBlocksRequired = (tNumContacts + SS_BLOCK - 1u) / SS_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + SS_GRID - 1u) / SS_GRID;
    const uint idx = groupThreadId.x;

    Ptr<PxgDeformablePairInfoGpu> contactInfos = Ptr<PxgDeformablePairInfoGpu>(gSsPrep.contactInfos);
    Ptr<float2> softBodyAppliedForces = Ptr<float2>(gSsPrep.softBodyAppliedForces);

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint rawWorkIndex = i * SS_BLOCK + idx + nbIterationsPerBlock * groupId.x * SS_BLOCK;
        if (rawWorkIndex >= tNumContacts)
            return;
        const uint workIndex = min(rawWorkIndex, tNumContacts - 1u);   // P9.23
        const uint threadIndexInWarp = workIndex & 31u;

        softBodyAppliedForces[workIndex] = float2(0.0f, 0.0f);

        // Side 1 is remapped from the collision mesh to the simulation mesh by a search that
        // can fail, which invalidates the whole pair.
        const PxgDeformablePairInfoGpu contactInfo = contactInfos[workIndex];
        if (!PxgIsValidPairInfo(contactInfo))
            continue;

        const uint64_t block = gSsPrep.contactBlocks + uint64_t(workIndex / 32u) * PXG_DBDBC_STRIDE;

        Ptr<PxgSoftBodyGpu> softbody0 = sbAt(gSsPrep.softbodies, PxgGetDeformableId(contactInfo.ref0));
        const uint tetInd0 = PxgGetDeformableFeatureId(contactInfo.ref0);

        Ptr<PxgSoftBodyGpu> softbody1 = sbAt(gSsPrep.softbodies, PxgGetDeformableId(contactInfo.ref1));
        const uint tetInd1 = PxgGetDeformableFeatureId(contactInfo.ref1);

        const float restDistance = softbody0[0].mRestOffset + softbody1[0].mRestOffset;

        const float4 normal_pen = Ptr<float4>(gSsPrep.normalPens)[workIndex];

        const float4 barycentric0 = Ptr<float4>(gSsPrep.barycentrics0)[workIndex];
        const float4 barycentric1 = Ptr<float4>(gSsPrep.barycentrics1)[workIndex];

        // The narrowphase reports pen at the predicted position; re-reference it to the
        // start-of-step configuration.  ssNormal is the negated narrowphase normal, so the
        // compensation adds (sb0 - sb1).ssNormal.
        const uint4 sb0TetVert = Ptr<uint4>(softbody0[0].mSimTetIndices)[tetInd0];
        const uint4 sb1TetVert = Ptr<uint4>(softbody1[0].mSimTetIndices)[tetInd1];
        const float4 sb0Delta = barycentricProjectTet(sb0TetVert, softbody0[0].mSimDeltaPos, barycentric0);
        const float4 sb1Delta = barycentricProjectTet(sb1TetVert, softbody1[0].mSimDeltaPos, barycentric1);
        const float3 ssNormal = float3(-normal_pen.x, -normal_pen.y, -normal_pen.z);
        const float3 rel = float3(sb0Delta.x - sb1Delta.x, sb0Delta.y - sb1Delta.y, sb0Delta.z - sb1Delta.z);
        const float ssPen = (normal_pen.w - restDistance) + dpDot3(rel, ssNormal);

        Ptr<float4>(block + PXG_DBDBC_normal_pen)[threadIndexInWarp] = float4(ssNormal.x, ssNormal.y, ssNormal.z, ssPen);
        Ptr<float>(block + PXG_DBDBC_maxPenBiasClamp)[threadIndexInWarp] =
            dpSelectMax(softbody0[0].mPenBiasClamp, softbody1[0].mPenBiasClamp);
    }
}

// ===========================================================================
// sb_querySSContactReferenceCountLaunch (NEW in 5.12)
//   (PxgSoftBody* softbodies, PxgDeformablePairInfo* contactInfos,
//    const float4* barycentrics0, const float4* barycentrics1,
//    PxgDbDbContactBlock* contactBlocks, PxU32* numContacts, const PxReal dt,
//    bool isVelocityIteration)
//   -> {8 x 6, 4, 1} = 56-byte blob (dt @48, isVelocityIteration byte @52).
// ===========================================================================
struct SbSsQueryParams
{
    uint64_t softbodies;          // @0
    uint64_t contactInfos;        // @8   in/out (ACTIVE flag)
    uint64_t barycentrics0;       // @16
    uint64_t barycentrics1;       // @24
    uint64_t contactBlocks;       // @32
    uint64_t numContacts;         // @40
    float dt;                     // @48
    uint isVelocityIteration;     // @52  bool byte0 -> 56
}
[[vk::push_constant]]
ConstantBuffer<SbSsQueryParams> gSsQuery;

[shader("compute")]
[numthreads(256, 1, 1)]   // == SS_BLOCK
void sb_querySSContactReferenceCountLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint tNumContacts = Ptr<uint>(gSsQuery.numContacts)[0];
    const uint nbBlocksRequired = (tNumContacts + SS_BLOCK - 1u) / SS_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + SS_GRID - 1u) / SS_GRID;
    const uint idx = groupThreadId.x;

    const float dt = gSsQuery.dt;
    const bool isVelocityIteration = (gSsQuery.isVelocityIteration & 0xffu) != 0u;

    Ptr<PxgDeformablePairInfoGpu> contactInfos = Ptr<PxgDeformablePairInfoGpu>(gSsQuery.contactInfos);

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint rawWorkIndex = i * SS_BLOCK + idx + nbIterationsPerBlock * groupId.x * SS_BLOCK;
        if (rawWorkIndex >= tNumContacts)
            return;
        const uint workIndex = min(rawWorkIndex, tNumContacts - 1u);   // P9.23
        const uint threadIndexInWarp = workIndex & 31u;

        const PxgDeformablePairInfoGpu contactInfo = contactInfos[workIndex];
        const uint64_t block = gSsQuery.contactBlocks + uint64_t(workIndex / 32u) * PXG_DBDBC_STRIDE;

        if (!PxgIsValidPairInfo(contactInfo))
        {
            ssSetActive(gSsQuery.contactInfos, workIndex, contactInfo.ref1, false);
            continue;
        }

        Ptr<PxgSoftBodyGpu> softbody0 = sbAt(gSsQuery.softbodies, PxgGetDeformableId(contactInfo.ref0));
        Ptr<PxgSoftBodyGpu> softbody1 = sbAt(gSsQuery.softbodies, PxgGetDeformableId(contactInfo.ref1));

        PxgDeformablePart db0 = PxgDeformablePart();
        PxgDeformablePart db1 = PxgDeformablePart();
        // checkOnlyActivity: no refcount multiply (the pre-count has not populated .w yet)
        // and no friction read.
        db0.readSoftBody(softbody0, contactInfo.ref0, gSsQuery.barycentrics0, workIndex, 0UL,
                         /*checkOnlyActivity*/ true, dt, isVelocityIteration);
        db1.readSoftBody(softbody1, contactInfo.ref1, gSsQuery.barycentrics1, workIndex, 0UL,
                         /*checkOnlyActivity*/ true, dt, isVelocityIteration);

        PxgDbContactPair contactPair = PxgDbContactPair();
        PxgDbSolveOutput solveOut = PxgDbSolveOutput();
        contactPair.readContactPrepDbDb(block, PXG_DBDBC_maxPenBiasClamp, threadIndexInWarp);

        const bool isActive = solveDbDbContact(db0, db1, contactPair, solveOut, 0.0f, 0.0f, 0.0f, dt,
                                               /*checkOnlyActivity*/ true, isVelocityIteration);

        ssSetActive(gSsQuery.contactInfos, workIndex, contactInfo.ref1, isActive);

        if (isActive)
        {
            // Integer-count refcount bump per touched tet vertex (gated |bc| > 1e-3).
            db0.bumpSoftBodyRefCountPart(softbody0);
            db1.bumpSoftBodyRefCountPart(softbody1);
        }
    }
}

// ===========================================================================
// sb_solveOutputSSDeltaVLaunch
//   (PxgSoftBody* softbodies, PxgDeformablePairInfo* contactInfos,
//    const float4* barycentrics0, const float4* barycentrics1,
//    PxgDbDbContactBlock* contactBlocks, PxU32* numContacts, const PxReal dt,
//    float2* appliedForces, PxsDeformableVolumeMaterialData* materials,
//    bool isVelocityIteration)
//   -> {8 x 6, 4, 8, 8, 1} = 80-byte blob (dt @48, appliedForces @56, materials @64,
//   isVelocityIteration byte @72).
// ===========================================================================
struct SbSsSolveParams
{
    uint64_t softbodies;          // @0
    uint64_t contactInfos;        // @8
    uint64_t barycentrics0;       // @16
    uint64_t barycentrics1;       // @24
    uint64_t contactBlocks;       // @32
    uint64_t numContacts;         // @40
    float dt;                     // @48
    uint _pad52;                  // @52
    uint64_t appliedForces;       // @56  in/out
    uint64_t materials;           // @64
    uint isVelocityIteration;     // @72  bool byte0
    uint _pad76;                  // @76 -> 80
}
[[vk::push_constant]]
ConstantBuffer<SbSsSolveParams> gSsSolve;

[shader("compute")]
[numthreads(256, 1, 1)]   // == SS_BLOCK
void sb_solveOutputSSDeltaVLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint tNumContacts = Ptr<uint>(gSsSolve.numContacts)[0];
    const uint nbBlocksRequired = (tNumContacts + SS_BLOCK - 1u) / SS_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + SS_GRID - 1u) / SS_GRID;
    const uint idx = groupThreadId.x;

    const float dt = gSsSolve.dt;
    const bool isVelocityIteration = (gSsSolve.isVelocityIteration & 0xffu) != 0u;

    Ptr<PxgDeformablePairInfoGpu> contactInfos = Ptr<PxgDeformablePairInfoGpu>(gSsSolve.contactInfos);
    Ptr<float2> appliedForces = Ptr<float2>(gSsSolve.appliedForces);

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint rawWorkIndex = i * SS_BLOCK + idx + nbIterationsPerBlock * groupId.x * SS_BLOCK;
        if (rawWorkIndex >= tNumContacts)
            return;
        const uint workIndex = min(rawWorkIndex, tNumContacts - 1u);   // P9.23
        const uint threadIndexInWarp = workIndex & 31u;

        // The pre-count decided this on the same positions, this iteration.
        const PxgDeformablePairInfoGpu contactInfo = contactInfos[workIndex];
        if (!PxgIsActivePairInfo(contactInfo))
            continue;

        const uint64_t block = gSsSolve.contactBlocks + uint64_t(workIndex / 32u) * PXG_DBDBC_STRIDE;

        Ptr<PxgSoftBodyGpu> softbody0 = sbAt(gSsSolve.softbodies, PxgGetDeformableId(contactInfo.ref0));
        Ptr<PxgSoftBodyGpu> softbody1 = sbAt(gSsSolve.softbodies, PxgGetDeformableId(contactInfo.ref1));

        PxgDeformablePart db0 = PxgDeformablePart();
        PxgDeformablePart db1 = PxgDeformablePart();
        db0.readSoftBody(softbody0, contactInfo.ref0, gSsSolve.barycentrics0, workIndex, gSsSolve.materials,
                         /*checkOnlyActivity*/ false, dt, isVelocityIteration);
        db1.readSoftBody(softbody1, contactInfo.ref1, gSsSolve.barycentrics1, workIndex, gSsSolve.materials,
                         /*checkOnlyActivity*/ false, dt, isVelocityIteration);

        const float frictionCoefficient = (db0.friction + db1.friction) * 0.5f;

        PxgDbContactPair contactPair = PxgDbContactPair();
        PxgDbSolveOutput solveOut = PxgDbSolveOutput();
        contactPair.readContactPrepDbDb(block, PXG_DBDBC_maxPenBiasClamp, threadIndexInWarp);

        float2 appliedForce = appliedForces[workIndex];   // (lambdaN_accum, lambdaT_accum)

        const bool hasApplied = solveDbDbContact(db0, db1, contactPair, solveOut, appliedForce.x, appliedForce.y,
                                                 frictionCoefficient, dt, /*checkOnlyActivity*/ false,
                                                 isVelocityIteration);

        if (hasApplied)
        {
            // A position delta in position iterations, velocity * dt in velocity iterations;
            // sb_gm_applyExternalDeltasLaunch applies it at iteration end.
            float3 delta;
            solveOut.computeDelta(contactPair, delta, dt);

            // Scatter .xyz weighted by the refCount-inflated vertexInvMasses; the pre-count owns .w.
            db0.writeSoftBody(softbody0, delta);
            db1.writeSoftBody(softbody1, float3(-delta.x, -delta.y, -delta.z));

            appliedForce.x = solveOut.accumulatedDeltaLambdaN;
            appliedForce.y += solveOut.deltaLambdaT;
            appliedForces[workIndex] = appliedForce;
        }
    }
}
