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

// femClothPsPrepareBda.slang -- cloth_particleContactPrepareLaunch (FEMClothConstraintPrep.cu,
// PhysX 5.12), the prepare of the cloth-vs-particle contact chain.  Launch: grid 1024,
// block 256.
//
// For every in-range contact it zeroes both applied-force streams (float2, before the
// validity guard); for every valid pair {ref0 = PxgParticleRef (system, SORTED particle
// index), ref1 = cloth TRIANGLE ref} it writes one PxgParticleDbContactBlock lane:
//
//   normal_pen       the NEGATED narrowphase normal and
//                    pen = normal_pen.w + normal.(clothDelta - particleDelta) - restDistance,
//                    the narrowphase raw distance re-referenced to the start-of-step
//                    configuration, restDistance = particleSystem.mRestOffset +
//                    cloth.mRestOffset (5.12: the narrowphase writes a raw distance, and the
//                    deformable rest offset is no longer counted twice);
//   barycentric      the contact's triangle barycentric (the solves read the triangle from it);
//   maxPenBiasClamp  the pair max of the cloth's and the particle system's clamps.
//
// 5.12 dropped the velMultiplier slot and the PxgFEMContactWriter-era contact layout.
// `contacts` is unread, as in the CUDA.  The cloth-particle narrowphase encodes every
// contact as a triangle, so there is no vertex branch.
//
// The strided work index is clamped to tNumContacts - 1 after the per-lane guard (P9.23,
// value-neutral).  In PXG_NOCONTRACT_SHADERS.

import pxgSlangFEMClothDesc;                // PxgFEMClothGpu, clothAt
import pxgSlangParticleDesc;                // PxgParticleSystemGpu
import pxgSlangDeformableRef;               // PxgDeformablePairInfoGpu, PXG_PDBC_*, PxgGetParticle*
import pxgSlangDeformableUtils;             // barycentricProjectTri
import pxgSlangDeformableAndParticleUtils;  // dpDot3, dpSelectMax

static const uint CPP_GRID = 1024u;
static const uint CPP_BLOCK = 256u;

// (PxgFEMCloth* clothes, PxgParticleSystem* particlesystems, float4* contacts,
//  float4* normalPens, float4* barycentrics, PxgDeformablePairInfo* contactInfos,
//  PxU32* numContacts, PxgParticleDbContactBlock* contactBlocks,
//  float2* clothAppliedForces, float2* particleAppliedForces) -> 80-byte blob.
struct ClothPsPrepareParams
{
    uint64_t clothes;                  // @0
    uint64_t particlesystems;          // @8
    uint64_t contacts;                 // @16  (unread, as in CUDA)
    uint64_t normalPens;               // @24
    uint64_t barycentrics;             // @32
    uint64_t contactInfos;             // @40
    uint64_t numContacts;              // @48
    uint64_t contactBlocks;            // @56  output
    uint64_t clothAppliedForces;       // @64  float2*, zeroed
    uint64_t particleAppliedForces;    // @72  float2*, zeroed -> 80
}

[[vk::push_constant]] ConstantBuffer<ClothPsPrepareParams> gCp;

[shader("compute")]
[numthreads(256, 1, 1)]
void cloth_particleContactPrepareLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tNumContacts = Ptr<uint>(gCp.numContacts)[0];
    const uint nbBlocksRequired = (tNumContacts + CPP_BLOCK - 1u) / CPP_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + CPP_GRID - 1u) / CPP_GRID;

    const uint idx = gtid.x;
    const uint threadIndexInWarp = gtid.x & 31u;

    Ptr<PxgParticleSystemGpu> particleSystems = Ptr<PxgParticleSystemGpu>(gCp.particlesystems);
    Ptr<PxgDeformablePairInfoGpu> contactInfos = Ptr<PxgDeformablePairInfoGpu>(gCp.contactInfos);

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint rawWorkIndex = i * CPP_BLOCK + idx + nbIterationsPerBlock * gid.x * CPP_BLOCK;
        if (rawWorkIndex >= tNumContacts)
            return;
        const uint workIndex = min(rawWorkIndex, tNumContacts - 1u);   // P9.23

        // Initialize the applied forces to zero.
        Ptr<float2>(gCp.clothAppliedForces)[workIndex] = float2(0.0f, 0.0f);
        Ptr<float2>(gCp.particleAppliedForces)[workIndex] = float2(0.0f, 0.0f);

        const PxgDeformablePairInfoGpu contactInfo = contactInfos[workIndex];
        const uint64_t block = gCp.contactBlocks + uint64_t(workIndex / 32u) * PXG_PDBC_STRIDE;

        if (!PxgIsValidPairInfo(contactInfo))
            continue;

        const uint64_t particleRef = contactInfo.ref0;
        const uint tParticleSystemId = PxgGetParticleSystemId(particleRef);
        const uint tParticleIndex = PxgGetParticleIndex(particleRef);

        // Second one is the cloth.
        Ptr<PxgFEMClothGpu> cloth = clothAt(gCp.clothes, PxgGetDeformableId(contactInfo.ref1));
        const uint triangleInd = PxgGetDeformableFeatureId(contactInfo.ref1);

        const uint4 triVertInd = Ptr<uint4>(cloth[0].mTriangleVertexIndices_ownedEdges)[triangleInd];

        const float4 normal_pen = Ptr<float4>(gCp.normalPens)[workIndex];
        const float4 barycentric = Ptr<float4>(gCp.barycentrics)[workIndex];

        // The narrowphase reports pen at the predicted position; re-reference it to the
        // start-of-step configuration, the frame the solver's live relLinDelta is measured from.
        const float4 delta1 = barycentricProjectTri(triVertInd, cloth[0].mAccumulatedDeltaPos, barycentric);

        const float3 normal = float3(-normal_pen.x, -normal_pen.y, -normal_pen.z);

        const float4 deltaP_invMass = Ptr<float4>(particleSystems[tParticleSystemId].mSortedDeltaP)[tParticleIndex];

        const float3 delta = float3(delta1.x - deltaP_invMass.x, delta1.y - deltaP_invMass.y, delta1.z - deltaP_invMass.z);

        // The rest distance as the SS/SC prepares compute it; the narrowphase writes a raw distance.
        const float restDistance = particleSystems[tParticleSystemId].mData_RestOffset + cloth[0].mRestOffset;
        const float pen = normal_pen.w + dpDot3(normal, delta) - restDistance;

        Ptr<float4>(block + PXG_PDBC_normal_pen)[threadIndexInWarp] = float4(normal.x, normal.y, normal.z, pen);
        Ptr<float4>(block + PXG_PDBC_barycentric)[threadIndexInWarp] = barycentric;
        Ptr<float>(block + PXG_PDBC_maxPenBiasClamp)[threadIndexInWarp] =
            dpSelectMax(cloth[0].mPenBiasClamp, particleSystems[tParticleSystemId].mData_PenBiasClamp);
    }
}
