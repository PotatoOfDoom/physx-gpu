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

// femClothPsSolveBda.slang -- the cloth-vs-particle contact SOLVE of FEMClothExternalSolve.cu,
// PhysX 5.12.  Launches: grid 1024, block 256.
//
//   cloth_queryCPContactReferenceCountLaunch  NEW in 5.12: the pre-count.  Activity from
//                          solveDbDbContact(checkOnlyActivity), recorded on the pair; an
//                          active pair bumps cloth.mDeltaPos.w per touched triangle vertex
//                          (|bc| > 1e-3) and the particle's mAccumDeltaP.w (the particle-side
//                          mass-split count).  An invalid pair or a kinematic particle
//                          (invMass 0) is set inactive.
//   cloth_solveCPOutputClothDeltaVLaunch     the cloth half: solveDbDbContact with side 0 =
//                          cloth triangle and side 1 = the particle (bc (1,0,0)), +delta
//                          scattered into mDeltaPos (writeCloth; the pre-count owns .w).
//   cloth_solveCPOutputParticleDeltaVLaunch  the particle half: the same solve (both halves
//                          read the same refCount-inflated masses and friction, so the
//                          impulse stays equal and opposite), -delta * invMass written to the
//                          per-contact deltaP slot with .w flagging an applied impulse.  Every
//                          slot of a non-kinematic particle is written (zero when inactive or
//                          not applied), and a kinematic particle's slot too: the accumulate
//                          scan (ps_accumulateFEMParticleDeltaVLaunch) reads them all.
//
// THE 5.12 DATA.  PxgDeformablePairInfo {ref0 = PxgParticleRef (system, SORTED index),
// ref1 = cloth triangle ref} and PxgParticleDbContactBlock (normal_pen, barycentric,
// maxPenBiasClamp); readCloth reads the barycentric from the BLOCK lane.  appliedForces is a
// float2 (lambdaN, lambdaT), one buffer per half.  The 5.6.1 velMultiplier / packed ids /
// PxgFemOtherContactInfo are gone.
//
// DEVIATIONS (value-neutral): the PxU16 phase -> material handle is read through its 32-bit
// word; the strided work index is clamped to tNumContacts - 1 after the per-lane guard
// (P9.23).  In PXG_NOCONTRACT_SHADERS.

import pxgSlangFEMClothDesc;                // PxgFEMClothGpu, clothAt
import pxgSlangParticleDesc;                // PxgParticleSystemGpu, pxgLoadPhaseToMaterial
import pxgSlangDeformableRef;               // PxgDeformablePairInfoGpu, PXG_PDBC_*, PxgGetParticle*
import pxgSlangAtomicFloat;                 // pxgAtomicAddF4W
import pxgSlangDeformableAndParticleUtils;  // PxgDeformablePart, solveDbDbContact

static const uint CPS_GRID = 1024u;
static const uint CPS_BLOCK = 256u;

// &getParticleMaterial<PxsParticleMaterialData>(mParticleMaterials, mi, mParticleMaterialStride).
uint64_t cpMaterialElem(Ptr<PxgParticleSystemGpu> ps, uint particleIndex)
{
    const uint phase = Ptr<uint>(ps[0].mSortedPhaseArray)[particleIndex];
    const uint group = phase & PXG_PARTICLE_PHASE_GROUP_MASK;
    const uint mi = pxgLoadPhaseToMaterial(ps[0].mPhaseGroupToMaterialHandle, group);
    return ps[0].mParticleMaterials + uint64_t(mi) * uint64_t(ps[0].mParticleMaterialStride);
}

Ptr<PxgParticleSystemGpu> cpSystemAt(uint64_t particlesystems, uint64_t particleRef)
{
    return Ptr<PxgParticleSystemGpu>(particlesystems
                                     + uint64_t(PxgGetParticleSystemId(particleRef)) * PXG_PARTICLE_SYSTEM_STRIDE);
}

void cpSetActive(uint64_t contactInfos, uint workIndex, uint64_t ref1, bool active)
{
    Ptr<uint64_t>(contactInfos + uint64_t(workIndex) * PXG_PAIRINFO_STRIDE + PXG_PAIRINFO_ref1)[0] =
        PxgSetActiveRef1(ref1, active);
}

// ===========================================================================
// cloth_queryCPContactReferenceCountLaunch (NEW in 5.12)
//   (PxgFEMCloth* clothes, PxgParticleSystem* particlesystems,
//    PxgDeformablePairInfo* contactInfos, PxgParticleDbContactBlock* contactBlocks,
//    PxU32* numContacts, const PxReal dt, bool isVelocityIteration)
//   -> {8 x 5, 4, 1} = 48-byte blob (dt @40, isVelocityIteration byte @44).
// ===========================================================================
struct ClothPsQueryParams
{
    uint64_t clothes;             // @0
    uint64_t particlesystems;     // @8
    uint64_t contactInfos;        // @16  in/out (ACTIVE flag)
    uint64_t contactBlocks;       // @24
    uint64_t numContacts;         // @32
    float dt;                     // @40
    uint isVelocityIteration;     // @44  bool byte0 -> 48
}
[[vk::push_constant]] ConstantBuffer<ClothPsQueryParams> gCpq;

[shader("compute")]
[numthreads(256, 1, 1)]
void cloth_queryCPContactReferenceCountLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tNumContacts = Ptr<uint>(gCpq.numContacts)[0];
    const uint nbBlocksRequired = (tNumContacts + CPS_BLOCK - 1u) / CPS_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + CPS_GRID - 1u) / CPS_GRID;
    const uint idx = gtid.x;
    const uint threadIndexInWarp = gtid.x & 31u;

    const float dt = gCpq.dt;
    const bool isVelocityIteration = (gCpq.isVelocityIteration & 0xffu) != 0u;

    Ptr<PxgDeformablePairInfoGpu> contactInfos = Ptr<PxgDeformablePairInfoGpu>(gCpq.contactInfos);

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint rawWorkIndex = i * CPS_BLOCK + idx + nbIterationsPerBlock * gid.x * CPS_BLOCK;
        if (rawWorkIndex >= tNumContacts)
            return;
        const uint workIndex = min(rawWorkIndex, tNumContacts - 1u);   // P9.23

        const PxgDeformablePairInfoGpu contactInfo = contactInfos[workIndex];
        const uint64_t block = gCpq.contactBlocks + uint64_t(workIndex / 32u) * PXG_PDBC_STRIDE;

        if (!PxgIsValidPairInfo(contactInfo))
        {
            cpSetActive(gCpq.contactInfos, workIndex, contactInfo.ref1, false);
            continue;
        }

        // ref0 = particle, ref1 = cloth triangle (matches the narrowphase).
        const uint64_t particleRef = contactInfo.ref0;
        Ptr<PxgParticleSystemGpu> particleSystem = cpSystemAt(gCpq.particlesystems, particleRef);
        const uint particleIndex = PxgGetParticleIndex(particleRef);
        const float4 deltaP_invMass = Ptr<float4>(particleSystem[0].mSortedDeltaP)[particleIndex];
        const float invMass0 = deltaP_invMass.w;

        // A kinematic particle is inactive end-to-end (the CP cloth solve skips its whole body).
        if (invMass0 == 0.0f)
        {
            cpSetActive(gCpq.contactInfos, workIndex, contactInfo.ref1, false);
            continue;
        }

        Ptr<PxgFEMClothGpu> cloth = clothAt(gCpq.clothes, PxgGetDeformableId(contactInfo.ref1));

        // Side 0 = cloth, side 1 = particle; the prepare stored normal = -narrowphase.
        PxgDeformablePart clothPart = PxgDeformablePart();
        clothPart.readCloth(cloth, contactInfo.ref1, block + PXG_PDBC_barycentric, threadIndexInWarp, 0UL,
                            /*countReferenceOnly*/ true, dt, isVelocityIteration);

        PxgDeformablePart particlePart = PxgDeformablePart();
        particlePart.readParticle(particleSystem, particleIndex, deltaP_invMass, 0UL,
                                  /*inflateByRefCount*/ false, dt, isVelocityIteration);

        PxgDbContactPair contactPair = PxgDbContactPair();
        PxgDbSolveOutput solveOut = PxgDbSolveOutput();
        contactPair.readContactPrepDbDb(block, PXG_PDBC_maxPenBiasClamp, threadIndexInWarp);

        const bool isActive = solveDbDbContact(clothPart, particlePart, contactPair, solveOut, 0.0f, 0.0f, 0.0f, dt,
                                               /*checkOnlyActivity*/ true, isVelocityIteration);

        cpSetActive(gCpq.contactInfos, workIndex, contactInfo.ref1, isActive);

        if (isActive)
        {
            // Integer-count refcount bump per touched triangle vertex (gated |bc| > 1e-3).
            clothPart.bumpClothRefCountPart(cloth);

            // Particle-side mass-split count, the counterpart of the cloth-vertex refCount.
            pxgAtomicAddF4W(particleSystem[0].mAccumDeltaP, particleIndex, 1.0f);
        }
    }
}

// ===========================================================================
// cloth_solveCPOutputClothDeltaVLaunch
//   (PxgFEMCloth* clothes, PxgParticleSystem* particlesystems,
//    const PxgDeformablePairInfo* contactInfos, PxgParticleDbContactBlock* contactBlocks,
//    PxU32* numContacts, float2* appliedForces, PxsDeformableSurfaceMaterialData* materials,
//    const PxReal dt, bool isVelocityIteration)
//   -> {8 x 7, 4, 1} = 64-byte blob (dt @56, isVelocityIteration byte @60).
// ===========================================================================
struct ClothPsSolveClothParams
{
    uint64_t clothes;             // @0
    uint64_t particlesystems;     // @8
    uint64_t contactInfos;        // @16
    uint64_t contactBlocks;       // @24
    uint64_t numContacts;         // @32
    uint64_t appliedForces;       // @40  in/out
    uint64_t materials;           // @48
    float dt;                     // @56
    uint isVelocityIteration;     // @60  bool byte0 -> 64
}
[[vk::push_constant]] ConstantBuffer<ClothPsSolveClothParams> gCps;

[shader("compute")]
[numthreads(256, 1, 1)]
void cloth_solveCPOutputClothDeltaVLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tNumContacts = Ptr<uint>(gCps.numContacts)[0];
    const uint nbBlocksRequired = (tNumContacts + CPS_BLOCK - 1u) / CPS_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + CPS_GRID - 1u) / CPS_GRID;
    const uint idx = gtid.x;
    const uint threadIndexInWarp = gtid.x & 31u;

    const float dt = gCps.dt;
    const bool isVelocityIteration = (gCps.isVelocityIteration & 0xffu) != 0u;

    Ptr<PxgDeformablePairInfoGpu> contactInfos = Ptr<PxgDeformablePairInfoGpu>(gCps.contactInfos);
    Ptr<float2> appliedForces = Ptr<float2>(gCps.appliedForces);

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint rawWorkIndex = i * CPS_BLOCK + idx + nbIterationsPerBlock * gid.x * CPS_BLOCK;
        if (rawWorkIndex >= tNumContacts)
            return;
        const uint workIndex = min(rawWorkIndex, tNumContacts - 1u);   // P9.23

        // The pre-count decided this on the same positions, this iteration.
        const PxgDeformablePairInfoGpu contactInfo = contactInfos[workIndex];
        if (!PxgIsActivePairInfo(contactInfo))
            continue;

        const uint64_t block = gCps.contactBlocks + uint64_t(workIndex / 32u) * PXG_PDBC_STRIDE;

        const uint64_t particleRef = contactInfo.ref0;
        Ptr<PxgParticleSystemGpu> particleSystem = cpSystemAt(gCps.particlesystems, particleRef);
        const uint particleIndex = PxgGetParticleIndex(particleRef);

        const float4 deltaP_invMass = Ptr<float4>(particleSystem[0].mSortedDeltaP)[particleIndex];
        const float invMass0 = deltaP_invMass.w;

        if (invMass0 != 0.0f)
        {
            const uint64_t psMat = cpMaterialElem(particleSystem, particleIndex);

            Ptr<PxgFEMClothGpu> cloth = clothAt(gCps.clothes, PxgGetDeformableId(contactInfo.ref1));

            // Cloth side: countReferenceOnly = false consumes the refCount in .w and reads the
            // material friction.
            PxgDeformablePart clothPart = PxgDeformablePart();
            clothPart.readCloth(cloth, contactInfo.ref1, block + PXG_PDBC_barycentric, threadIndexInWarp,
                                gCps.materials, /*countReferenceOnly*/ false, dt, isVelocityIteration);

            PxgDeformablePart particlePart = PxgDeformablePart();
            particlePart.readParticle(particleSystem, particleIndex, deltaP_invMass, psMat,
                                      /*inflateByRefCount*/ true, dt, isVelocityIteration);

            PxgDbContactPair contactPair = PxgDbContactPair();
            PxgDbSolveOutput solveOut = PxgDbSolveOutput();
            contactPair.readContactPrepDbDb(block, PXG_PDBC_maxPenBiasClamp, threadIndexInWarp);

            const float frictionCoefficient = (clothPart.friction + particlePart.friction) * 0.5f;

            float2 appliedForce = appliedForces[workIndex];
            const bool hasApplied = solveDbDbContact(clothPart, particlePart, contactPair, solveOut, appliedForce.x,
                                                     appliedForce.y, frictionCoefficient, dt,
                                                     /*checkOnlyActivity*/ false, isVelocityIteration);

            if (hasApplied)
            {
                float3 delta;
                solveOut.computeDelta(contactPair, delta, dt);

                // .xyz weighted by the refCount-inflated vertexInvMasses; the pre-count owns .w.
                clothPart.writeCloth(cloth, delta);

                appliedForce.x = solveOut.accumulatedDeltaLambdaN;
                appliedForce.y += solveOut.deltaLambdaT;
                appliedForces[workIndex] = appliedForce;
            }
        }
    }
}

// ===========================================================================
// cloth_solveCPOutputParticleDeltaVLaunch
//   (PxgFEMCloth* clothes, PxgParticleSystem* particlesystems,
//    const PxgDeformablePairInfo* contactInfos, PxgParticleDbContactBlock* contactBlocks,
//    PxU32* numContacts, float4* deltaP, float2* appliedForces,
//    PxsDeformableSurfaceMaterialData* materials, const PxReal dt, bool isVelocityIteration)
//   -> {8 x 8, 4, 1} = 72-byte blob (dt @64, isVelocityIteration byte @68).
// ===========================================================================
struct ClothPsSolvePsParams
{
    uint64_t clothes;             // @0
    uint64_t particlesystems;     // @8
    uint64_t contactInfos;        // @16
    uint64_t contactBlocks;       // @24
    uint64_t numContacts;         // @32
    uint64_t deltaP;              // @40  output, one float4 per contact
    uint64_t appliedForces;       // @48  in/out
    uint64_t materials;           // @56
    float dt;                     // @64
    uint isVelocityIteration;     // @68  bool byte0 -> 72
}
[[vk::push_constant]] ConstantBuffer<ClothPsSolvePsParams> gCpp;

[shader("compute")]
[numthreads(256, 1, 1)]
void cloth_solveCPOutputParticleDeltaVLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tNumContacts = Ptr<uint>(gCpp.numContacts)[0];
    const uint nbBlocksRequired = (tNumContacts + CPS_BLOCK - 1u) / CPS_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + CPS_GRID - 1u) / CPS_GRID;
    const uint idx = gtid.x;
    const uint threadIndexInWarp = gtid.x & 31u;

    const float dt = gCpp.dt;
    const bool isVelocityIteration = (gCpp.isVelocityIteration & 0xffu) != 0u;

    Ptr<PxgDeformablePairInfoGpu> contactInfos = Ptr<PxgDeformablePairInfoGpu>(gCpp.contactInfos);
    Ptr<float2> appliedForces = Ptr<float2>(gCpp.appliedForces);
    Ptr<float4> deltaP = Ptr<float4>(gCpp.deltaP);

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint rawWorkIndex = i * CPS_BLOCK + idx + nbIterationsPerBlock * gid.x * CPS_BLOCK;
        if (rawWorkIndex >= tNumContacts)
            return;
        const uint workIndex = min(rawWorkIndex, tNumContacts - 1u);   // P9.23

        const PxgDeformablePairInfoGpu contactInfo = contactInfos[workIndex];
        const uint64_t block = gCpp.contactBlocks + uint64_t(workIndex / 32u) * PXG_PDBC_STRIDE;

        const uint64_t particleRef = contactInfo.ref0;
        Ptr<PxgParticleSystemGpu> particleSystem = cpSystemAt(gCpp.particlesystems, particleRef);
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

            const uint64_t psMat = cpMaterialElem(particleSystem, particleIndex);

            Ptr<PxgFEMClothGpu> cloth = clothAt(gCpp.clothes, PxgGetDeformableId(contactInfo.ref1));

            // Read in full (not activity-only) so both halves see the same invMass and friction;
            // side-effect free here, this kernel scatters only to deltaP[], never to mDeltaPos.
            PxgDeformablePart clothPart = PxgDeformablePart();
            clothPart.readCloth(cloth, contactInfo.ref1, block + PXG_PDBC_barycentric, threadIndexInWarp,
                                gCpp.materials, /*countReferenceOnly*/ false, dt, isVelocityIteration);

            PxgDeformablePart particlePart = PxgDeformablePart();
            particlePart.readParticle(particleSystem, particleIndex, deltaP_invMass, psMat,
                                      /*inflateByRefCount*/ true, dt, isVelocityIteration);

            PxgDbContactPair contactPair = PxgDbContactPair();
            PxgDbSolveOutput solveOut = PxgDbSolveOutput();
            contactPair.readContactPrepDbDb(block, PXG_PDBC_maxPenBiasClamp, threadIndexInWarp);

            const float frictionCoefficient = (clothPart.friction + particlePart.friction) * 0.5f;

            float2 appliedForce = appliedForces[workIndex];
            const bool hasApplied = solveDbDbContact(clothPart, particlePart, contactPair, solveOut, appliedForce.x,
                                                     appliedForce.y, frictionCoefficient, dt,
                                                     /*checkOnlyActivity*/ false, isVelocityIteration);

            if (hasApplied)
            {
                // Aggregated into mAccumDeltaP by the FEM accumulate (.w flags a live contact);
                // ps_updateParticleLaunch applies it at iteration end.
                float3 delta;
                solveOut.computeDelta(contactPair, delta, dt);
                particlePart.writeParticle(gCpp.deltaP, workIndex, float3(-delta.x, -delta.y, -delta.z), solveOut);

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
