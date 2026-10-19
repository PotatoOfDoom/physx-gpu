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

// softBodyPsSolveBda.slang -- the softbody-vs-particle contact SOLVE of softBody.cu,
// PhysX 5.12.  Launches: grid 1024, block 256.
//
//   sb_querySBParticleContactReferenceCountLaunch  NEW in 5.12: the pre-count.  Activity
//                          from solveDbDbContact(checkOnlyActivity), recorded on the pair;
//                          an active pair bumps softbody.mSimDelta.w per touched tet vertex
//                          and the particle's mAccumDeltaP.w (the particle-side mass-split
//                          count).  An invalid pair or a kinematic particle (invMass 0) is
//                          set inactive.
//   sb_solveOutputSPDeltaVLaunch        the softbody half: solveDbDbContact with side 0 =
//                          softbody tet (PxVec4) and side 1 = the particle (PxVec3, bc
//                          (1,0,0)), +delta scattered into mSimDelta.
//   sb_solveOutputParticleDeltaVLaunch  the particle half: the same solve (both halves read
//                          the same refCount-inflated masses and friction, so the impulse
//                          stays equal and opposite), -delta * invMass written to the
//                          per-contact deltaP slot with .w flagging an applied impulse.
//                          Every slot of a non-kinematic particle is written (zero when
//                          inactive or not applied), and a kinematic particle's slot too:
//                          the accumulate scan reads them all.
//
// THE 5.12 DATA.  PxgDeformablePairInfo {ref0 = PxgParticleRef (system, SORTED index),
// ref1 = simulation tet ref} and PxgParticleDbContactBlock (normal_pen, barycentric,
// maxPenBiasClamp); readSoftBody reads the barycentric from the BLOCK lane.  appliedForces
// is a float2 (lambdaN, lambdaT), one buffer per half.  The 5.6.1 velMultiplier /
// biasCoefficient / relaxation terms are gone.
//
// DEVIATIONS (value-neutral): the PxU16 phase -> material handle is read through its 32-bit
// word; the strided work index is clamped to tNumContacts - 1 after the per-lane guard
// (P9.23).  In PXG_NOCONTRACT_SHADERS.

import pxgSlangSoftBodyDesc;                // PxgSoftBodyGpu, sbAt
import pxgSlangParticleDesc;                // PxgParticleSystemGpu, pxgLoadPhaseToMaterial
import pxgSlangDeformableRef;               // PxgDeformablePairInfoGpu, PXG_PDBC_*, PxgGetParticle*
import pxgSlangAtomicFloat;                 // pxgAtomicAddF4W
import pxgSlangDeformableAndParticleUtils;  // PxgDeformablePart, solveDbDbContact

static const uint SPS_GRID = 1024u;
static const uint SPS_BLOCK = 256u;

// &getParticleMaterial<PxsParticleMaterialData>(mParticleMaterials, mi, mParticleMaterialStride).
uint64_t spMaterialElem(Ptr<PxgParticleSystemGpu> ps, uint particleIndex)
{
    const uint phase = Ptr<uint>(ps[0].mSortedPhaseArray)[particleIndex];
    const uint group = phase & PXG_PARTICLE_PHASE_GROUP_MASK;
    const uint mi = pxgLoadPhaseToMaterial(ps[0].mPhaseGroupToMaterialHandle, group);
    return ps[0].mParticleMaterials + uint64_t(mi) * uint64_t(ps[0].mParticleMaterialStride);
}

Ptr<PxgParticleSystemGpu> spSystemAt(uint64_t particlesystems, uint64_t particleRef)
{
    return Ptr<PxgParticleSystemGpu>(particlesystems
                                     + uint64_t(PxgGetParticleSystemId(particleRef)) * PXG_PARTICLE_SYSTEM_STRIDE);
}

void spSetActive(uint64_t contactInfos, uint workIndex, uint64_t ref1, bool active)
{
    Ptr<uint64_t>(contactInfos + uint64_t(workIndex) * PXG_PAIRINFO_STRIDE + PXG_PAIRINFO_ref1)[0] =
        PxgSetActiveRef1(ref1, active);
}

// ===========================================================================
// sb_querySBParticleContactReferenceCountLaunch (NEW in 5.12)
//   (PxgSoftBody* softbodies, PxgParticleSystem* particlesystems,
//    PxgDeformablePairInfo* contactInfos, PxgParticleDbContactBlock* contactBlocks,
//    PxU32* numContacts, const PxReal dt, bool isVelocityIteration)
//   -> {8 x 5, 4, 1} = 48-byte blob (dt @40, isVelocityIteration byte @44).
// ===========================================================================
struct SbPsQueryParams
{
    uint64_t softbodies;          // @0
    uint64_t particlesystems;     // @8
    uint64_t contactInfos;        // @16  in/out (ACTIVE flag)
    uint64_t contactBlocks;       // @24
    uint64_t numContacts;         // @32
    float dt;                     // @40
    uint isVelocityIteration;     // @44  bool byte0 -> 48
}
[[vk::push_constant]] ConstantBuffer<SbPsQueryParams> gSpq;

[shader("compute")]
[numthreads(256, 1, 1)]
void sb_querySBParticleContactReferenceCountLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tNumContacts = Ptr<uint>(gSpq.numContacts)[0];
    const uint nbBlocksRequired = (tNumContacts + SPS_BLOCK - 1u) / SPS_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + SPS_GRID - 1u) / SPS_GRID;
    const uint idx = gtid.x;
    const uint threadIndexInWarp = gtid.x & 31u;

    const float dt = gSpq.dt;
    const bool isVelocityIteration = (gSpq.isVelocityIteration & 0xffu) != 0u;

    Ptr<PxgDeformablePairInfoGpu> contactInfos = Ptr<PxgDeformablePairInfoGpu>(gSpq.contactInfos);

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint rawWorkIndex = i * SPS_BLOCK + idx + nbIterationsPerBlock * gid.x * SPS_BLOCK;
        if (rawWorkIndex >= tNumContacts)
            return;
        const uint workIndex = min(rawWorkIndex, tNumContacts - 1u);   // P9.23

        const PxgDeformablePairInfoGpu contactInfo = contactInfos[workIndex];
        const uint64_t block = gSpq.contactBlocks + uint64_t(workIndex / 32u) * PXG_PDBC_STRIDE;

        if (!PxgIsValidPairInfo(contactInfo))
        {
            spSetActive(gSpq.contactInfos, workIndex, contactInfo.ref1, false);
            continue;
        }

        // ref0 = particle, ref1 = softbody tet.
        const uint64_t particleRef = contactInfo.ref0;
        Ptr<PxgParticleSystemGpu> particleSystem = spSystemAt(gSpq.particlesystems, particleRef);
        const uint particleIndex = PxgGetParticleIndex(particleRef);
        const float4 deltaP_invMass = Ptr<float4>(particleSystem[0].mSortedDeltaP)[particleIndex];
        const float invMass0 = deltaP_invMass.w;

        // A kinematic particle is inactive end-to-end (the SP solve skips its whole body).
        if (invMass0 == 0.0f)
        {
            spSetActive(gSpq.contactInfos, workIndex, contactInfo.ref1, false);
            continue;
        }

        Ptr<PxgSoftBodyGpu> softbody = sbAt(gSpq.softbodies, PxgGetDeformableId(contactInfo.ref1));

        // Side 0 = SB tet, side 1 = particle; the prepare stored normal = -narrowphase.
        PxgDeformablePart sbPart = PxgDeformablePart();
        sbPart.readSoftBody(softbody, contactInfo.ref1, block + PXG_PDBC_barycentric, threadIndexInWarp, 0UL,
                            /*checkOnlyActivity*/ true, dt, isVelocityIteration);

        PxgDeformablePart particlePart = PxgDeformablePart();
        particlePart.readParticle(particleSystem, particleIndex, deltaP_invMass, 0UL,
                                  /*inflateByRefCount*/ false, dt, isVelocityIteration);

        PxgDbContactPair contactPair = PxgDbContactPair();
        PxgDbSolveOutput solveOut = PxgDbSolveOutput();
        contactPair.readContactPrepDbDb(block, PXG_PDBC_maxPenBiasClamp, threadIndexInWarp);

        const bool isActive = solveDbDbContact(sbPart, particlePart, contactPair, solveOut, 0.0f, 0.0f, 0.0f, dt,
                                               /*checkOnlyActivity*/ true, isVelocityIteration);

        spSetActive(gSpq.contactInfos, workIndex, contactInfo.ref1, isActive);

        if (isActive)
        {
            // Integer-count refcount bump per touched tet vertex (gated |bc| > 1e-3).
            sbPart.bumpSoftBodyRefCountPart(softbody);

            // Particle-side mass-split count, the counterpart of the SB-vertex refCount.
            pxgAtomicAddF4W(particleSystem[0].mAccumDeltaP, particleIndex, 1.0f);
        }
    }
}

// ===========================================================================
// sb_solveOutputSPDeltaVLaunch
//   (PxgSoftBody* softbodies, PxgParticleSystem* particlesystems,
//    PxgDeformablePairInfo* contactInfos, PxgParticleDbContactBlock* contactBlocks,
//    PxU32* numContacts, float2* appliedForces, const PxReal dt,
//    PxsDeformableVolumeMaterialData* materials, bool isVelocityIteration)
//   -> {8 x 6, 4, 8, 1} = 72-byte blob (dt @48, materials @56, isVelocityIteration byte @64).
// ===========================================================================
struct SbPsSolveSbParams
{
    uint64_t softbodies;          // @0
    uint64_t particlesystems;     // @8
    uint64_t contactInfos;        // @16
    uint64_t contactBlocks;       // @24
    uint64_t numContacts;         // @32
    uint64_t appliedForces;       // @40  in/out
    float dt;                     // @48
    uint _pad52;                  // @52
    uint64_t materials;           // @56
    uint isVelocityIteration;     // @64  bool byte0
    uint _pad68;                  // @68 -> 72
}
[[vk::push_constant]] ConstantBuffer<SbPsSolveSbParams> gSps;

[shader("compute")]
[numthreads(256, 1, 1)]
void sb_solveOutputSPDeltaVLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tNumContacts = Ptr<uint>(gSps.numContacts)[0];
    const uint nbBlocksRequired = (tNumContacts + SPS_BLOCK - 1u) / SPS_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + SPS_GRID - 1u) / SPS_GRID;
    const uint idx = gtid.x;
    const uint threadIndexInWarp = gtid.x & 31u;

    const float dt = gSps.dt;
    const bool isVelocityIteration = (gSps.isVelocityIteration & 0xffu) != 0u;

    Ptr<PxgDeformablePairInfoGpu> contactInfos = Ptr<PxgDeformablePairInfoGpu>(gSps.contactInfos);
    Ptr<float2> appliedForces = Ptr<float2>(gSps.appliedForces);

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint rawWorkIndex = i * SPS_BLOCK + idx + nbIterationsPerBlock * gid.x * SPS_BLOCK;
        if (rawWorkIndex >= tNumContacts)
            return;
        const uint workIndex = min(rawWorkIndex, tNumContacts - 1u);   // P9.23

        // The pre-count decided this on the same positions, this iteration.
        const PxgDeformablePairInfoGpu contactInfo = contactInfos[workIndex];
        if (!PxgIsActivePairInfo(contactInfo))
            continue;

        const uint64_t block = gSps.contactBlocks + uint64_t(workIndex / 32u) * PXG_PDBC_STRIDE;

        const uint64_t particleRef = contactInfo.ref0;
        Ptr<PxgParticleSystemGpu> particleSystem = spSystemAt(gSps.particlesystems, particleRef);
        const uint particleIndex = PxgGetParticleIndex(particleRef);

        const float4 deltaP_invMass = Ptr<float4>(particleSystem[0].mSortedDeltaP)[particleIndex];
        const float invMass0 = deltaP_invMass.w;

        if (invMass0 != 0.0f)
        {
            const uint64_t psMat = spMaterialElem(particleSystem, particleIndex);

            Ptr<PxgSoftBodyGpu> softbody = sbAt(gSps.softbodies, PxgGetDeformableId(contactInfo.ref1));

            // SB side: checkOnlyActivity = false consumes the refCount in .w and reads the
            // material friction.
            PxgDeformablePart sbPart = PxgDeformablePart();
            sbPart.readSoftBody(softbody, contactInfo.ref1, block + PXG_PDBC_barycentric, threadIndexInWarp,
                                gSps.materials, /*checkOnlyActivity*/ false, dt, isVelocityIteration);

            PxgDeformablePart particlePart = PxgDeformablePart();
            particlePart.readParticle(particleSystem, particleIndex, deltaP_invMass, psMat,
                                      /*inflateByRefCount*/ true, dt, isVelocityIteration);

            PxgDbContactPair contactPair = PxgDbContactPair();
            PxgDbSolveOutput solveOut = PxgDbSolveOutput();
            contactPair.readContactPrepDbDb(block, PXG_PDBC_maxPenBiasClamp, threadIndexInWarp);

            const float frictionCoefficient = (sbPart.friction + particlePart.friction) * 0.5f;

            float2 appliedForce = appliedForces[workIndex];
            const bool hasApplied = solveDbDbContact(sbPart, particlePart, contactPair, solveOut, appliedForce.x,
                                                     appliedForce.y, frictionCoefficient, dt,
                                                     /*checkOnlyActivity*/ false, isVelocityIteration);

            if (hasApplied)
            {
                float3 delta;
                solveOut.computeDelta(contactPair, delta, dt);

                // .xyz weighted by the refCount-inflated vertexInvMasses; the pre-count owns .w.
                sbPart.writeSoftBody(softbody, delta);

                appliedForce.x = solveOut.accumulatedDeltaLambdaN;
                appliedForce.y += solveOut.deltaLambdaT;
                appliedForces[workIndex] = appliedForce;
            }
        }
    }
}

// ===========================================================================
// sb_solveOutputParticleDeltaVLaunch
//   (PxgSoftBody* softbodies, PxgParticleSystem* particlesystems,
//    PxgDeformablePairInfo* contactInfos, PxgParticleDbContactBlock* contactBlocks,
//    PxU32* numContacts, float4* deltaP, float2* appliedForces, const PxReal dt,
//    PxsDeformableVolumeMaterialData* materials, bool isVelocityIteration)
//   -> {8 x 7, 4, 8, 1} = 80-byte blob (dt @56, materials @64, isVelocityIteration byte @72).
// ===========================================================================
struct SbPsSolvePsParams
{
    uint64_t softbodies;          // @0
    uint64_t particlesystems;     // @8
    uint64_t contactInfos;        // @16
    uint64_t contactBlocks;       // @24
    uint64_t numContacts;         // @32
    uint64_t deltaP;              // @40  output, one float4 per contact
    uint64_t appliedForces;       // @48  in/out
    float dt;                     // @56
    uint _pad60;                  // @60
    uint64_t materials;           // @64
    uint isVelocityIteration;     // @72  bool byte0
    uint _pad76;                  // @76 -> 80
}
[[vk::push_constant]] ConstantBuffer<SbPsSolvePsParams> gSpp;

[shader("compute")]
[numthreads(256, 1, 1)]
void sb_solveOutputParticleDeltaVLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tNumContacts = Ptr<uint>(gSpp.numContacts)[0];
    const uint nbBlocksRequired = (tNumContacts + SPS_BLOCK - 1u) / SPS_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + SPS_GRID - 1u) / SPS_GRID;
    const uint idx = gtid.x;
    const uint threadIndexInWarp = gtid.x & 31u;

    const float dt = gSpp.dt;
    const bool isVelocityIteration = (gSpp.isVelocityIteration & 0xffu) != 0u;

    Ptr<PxgDeformablePairInfoGpu> contactInfos = Ptr<PxgDeformablePairInfoGpu>(gSpp.contactInfos);
    Ptr<float2> appliedForces = Ptr<float2>(gSpp.appliedForces);
    Ptr<float4> deltaP = Ptr<float4>(gSpp.deltaP);

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint rawWorkIndex = i * SPS_BLOCK + idx + nbIterationsPerBlock * gid.x * SPS_BLOCK;
        if (rawWorkIndex >= tNumContacts)
            return;
        const uint workIndex = min(rawWorkIndex, tNumContacts - 1u);   // P9.23

        const PxgDeformablePairInfoGpu contactInfo = contactInfos[workIndex];
        const uint64_t block = gSpp.contactBlocks + uint64_t(workIndex / 32u) * PXG_PDBC_STRIDE;

        const uint64_t particleRef = contactInfo.ref0;
        Ptr<PxgParticleSystemGpu> particleSystem = spSystemAt(gSpp.particlesystems, particleRef);
        const uint particleIndex = PxgGetParticleIndex(particleRef);

        const float4 deltaP_invMass = Ptr<float4>(particleSystem[0].mSortedDeltaP)[particleIndex];
        const float invMass0 = deltaP_invMass.w;

        if (invMass0 != 0.0f)
        {
            if (!PxgIsActivePairInfo(contactInfo))
            {
                // The accumulate scan reads every slot, so an inactive contact's is cleared.
                deltaP[workIndex] = float4(0.0f, 0.0f, 0.0f, 0.0f);
                continue;
            }

            const uint64_t psMat = spMaterialElem(particleSystem, particleIndex);

            Ptr<PxgSoftBodyGpu> softbody = sbAt(gSpp.softbodies, PxgGetDeformableId(contactInfo.ref1));

            // Read in full (not activity-only) so both halves see the same invMass and friction;
            // side-effect free here, this kernel scatters only to deltaP[].
            PxgDeformablePart sbPart = PxgDeformablePart();
            sbPart.readSoftBody(softbody, contactInfo.ref1, block + PXG_PDBC_barycentric, threadIndexInWarp,
                                gSpp.materials, /*checkOnlyActivity*/ false, dt, isVelocityIteration);

            PxgDeformablePart particlePart = PxgDeformablePart();
            particlePart.readParticle(particleSystem, particleIndex, deltaP_invMass, psMat,
                                      /*inflateByRefCount*/ true, dt, isVelocityIteration);

            PxgDbContactPair contactPair = PxgDbContactPair();
            PxgDbSolveOutput solveOut = PxgDbSolveOutput();
            contactPair.readContactPrepDbDb(block, PXG_PDBC_maxPenBiasClamp, threadIndexInWarp);

            const float frictionCoefficient = (sbPart.friction + particlePart.friction) * 0.5f;

            float2 appliedForce = appliedForces[workIndex];
            const bool hasApplied = solveDbDbContact(sbPart, particlePart, contactPair, solveOut, appliedForce.x,
                                                     appliedForce.y, frictionCoefficient, dt,
                                                     /*checkOnlyActivity*/ false, isVelocityIteration);

            if (hasApplied)
            {
                // Aggregated into mAccumDeltaP by the FEM accumulate (.w flags a live contact);
                // ps_updateParticleLaunch applies it at iteration end.
                float3 delta;
                solveOut.computeDelta(contactPair, delta, dt);
                particlePart.writeParticle(gSpp.deltaP, workIndex, float3(-delta.x, -delta.y, -delta.z), solveOut);

                appliedForce.x = solveOut.accumulatedDeltaLambdaN;
                appliedForce.y += solveOut.deltaLambdaT;
                appliedForces[workIndex] = appliedForce;
            }
            else
            {
                deltaP[workIndex] = float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        else
        {
            // A kinematic particle takes no delta; its slot is cleared for the accumulate scan.
            deltaP[workIndex] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
}
