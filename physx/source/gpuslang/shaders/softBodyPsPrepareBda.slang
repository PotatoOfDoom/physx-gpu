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

// softBodyPsPrepareBda.slang -- sb_particleContactPrepareLaunch (softBody.cu, PhysX 5.12),
// the prepare of the softbody-vs-particle contact chain.  Launch: grid 1024, block 256.
//
// For every valid pair {ref0 = PxgParticleRef (system, SORTED particle index),
// ref1 = simulation-mesh tetrahedron ref} it zeroes both applied-force streams (float2,
// before the validity guard) and writes one PxgParticleDbContactBlock lane:
//
//   normal_pen       the NEGATED narrowphase normal and
//                    pen = normal_pen.w - (particleDelta - tetDelta).normal - restDistance,
//                    the narrowphase raw distance re-referenced to the start of the step,
//                    restDistance = particleSystem.mRestOffset + softbody.mRestOffset;
//   barycentric      the contact's tet barycentric (the solves read the tet from it);
//   maxPenBiasClamp  the pair max of the softbody's and the particle system's clamps.
//
// 5.12 dropped the velMultiplier slot (the solves build the denominator from the refCount-
// inflated masses) and the 5.6.1 getSoftBodyInvMass L2-norm scale with it.  There is no
// maxContacts clamp and no write-back here, unlike the two sibling prepares.  `contacts`
// is unread, as in the CUDA.  Q-note kept from 5.6.1: the applied-force arrays are float2
// written densely at stride 8, two distinct buffers (softbody side, particle side).
//
// The strided work index is clamped to tNumContacts - 1 after the per-lane guard (P9.23,
// value-neutral).  In PXG_NOCONTRACT_SHADERS.

import pxgSlangSoftBodyDesc;                // PxgSoftBodyGpu, sbAt
import pxgSlangParticleDesc;                // PxgParticleSystemGpu
import pxgSlangDeformableRef;               // PxgDeformablePairInfoGpu, PXG_PDBC_*, PxgGetParticle*
import pxgSlangDeformableUtils;             // barycentricProjectTet
import pxgSlangDeformableAndParticleUtils;  // dpDot3, dpSelectMax

static const uint SPP_GRID = 1024u;
static const uint SPP_BLOCK = 256u;

// (PxgSoftBody* softbodies, PxgParticleSystem* particlesystems, float4* contacts,
//  float4* normalPens, float4* barycentrics, PxgDeformablePairInfo* contactInfos,
//  PxU32* numContacts, PxgParticleDbContactBlock* contactBlocks,
//  float2* softBodyAppliedForces, float2* particleAppliedForces) -> 80-byte blob.
struct SbPsPrepareParams
{
    uint64_t softbodies;               // @0
    uint64_t particlesystems;          // @8
    uint64_t contacts;                 // @16  (unread, as in CUDA)
    uint64_t normalPens;               // @24
    uint64_t barycentrics;             // @32
    uint64_t contactInfos;             // @40
    uint64_t numContacts;              // @48
    uint64_t contactBlocks;            // @56  output
    uint64_t softBodyAppliedForces;    // @64  float2*, zeroed
    uint64_t particleAppliedForces;    // @72  float2*, zeroed -> 80
}

[[vk::push_constant]] ConstantBuffer<SbPsPrepareParams> gSp;

[shader("compute")]
[numthreads(256, 1, 1)]
void sb_particleContactPrepareLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tNumContacts = Ptr<uint>(gSp.numContacts)[0];
    const uint nbBlocksRequired = (tNumContacts + SPP_BLOCK - 1u) / SPP_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + SPP_GRID - 1u) / SPP_GRID;

    const uint idx = gtid.x;
    const uint threadIndexInWarp = gtid.x & 31u;

    Ptr<PxgParticleSystemGpu> particleSystems = Ptr<PxgParticleSystemGpu>(gSp.particlesystems);
    Ptr<PxgDeformablePairInfoGpu> contactInfos = Ptr<PxgDeformablePairInfoGpu>(gSp.contactInfos);

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint rawWorkIndex = i * SPP_BLOCK + idx + nbIterationsPerBlock * gid.x * SPP_BLOCK;
        if (rawWorkIndex >= tNumContacts)
            return;
        const uint workIndex = min(rawWorkIndex, tNumContacts - 1u);   // P9.23

        Ptr<float2>(gSp.softBodyAppliedForces)[workIndex] = float2(0.0f, 0.0f);
        Ptr<float2>(gSp.particleAppliedForces)[workIndex] = float2(0.0f, 0.0f);

        const PxgDeformablePairInfoGpu contactInfo = contactInfos[workIndex];
        const uint64_t block = gSp.contactBlocks + uint64_t(workIndex / 32u) * PXG_PDBC_STRIDE;

        if (!PxgIsValidPairInfo(contactInfo))
            continue;

        const uint64_t particleRef = contactInfo.ref0;
        const uint tParticleSystemId = PxgGetParticleSystemId(particleRef);
        const uint tParticleIndex = PxgGetParticleIndex(particleRef);

        // Second one is the soft body.
        Ptr<PxgSoftBodyGpu> softbody = sbAt(gSp.softbodies, PxgGetDeformableId(contactInfo.ref1));
        const uint tetInd = PxgGetDeformableFeatureId(contactInfo.ref1);

        const uint4 tetrahedronIdx = Ptr<uint4>(softbody[0].mSimTetIndices)[tetInd];

        const float4 normal_pen = Ptr<float4>(gSp.normalPens)[workIndex];
        const float4 barycentric = Ptr<float4>(gSp.barycentrics)[workIndex];

        const float4 tetDelta4 = barycentricProjectTet(tetrahedronIdx, softbody[0].mSimDeltaPos, barycentric);
        const float3 tetDelta = float3(tetDelta4.x, tetDelta4.y, tetDelta4.z);

        const float3 normal = float3(-normal_pen.x, -normal_pen.y, -normal_pen.z);

        const float4 delta_invMass = Ptr<float4>(particleSystems[tParticleSystemId].mSortedDeltaP)[tParticleIndex];
        const float3 particleDelta = float3(delta_invMass.x, delta_invMass.y, delta_invMass.z);
        // The rest distance as the SS/SC prepares compute it; the narrowphase writes a raw distance.
        const float restDistance = particleSystems[tParticleSystemId].mData_RestOffset + softbody[0].mRestOffset;
        const float3 rel = float3(particleDelta.x - tetDelta.x, particleDelta.y - tetDelta.y,
                                  particleDelta.z - tetDelta.z);
        const float pen = normal_pen.w - dpDot3(rel, normal) - restDistance;

        Ptr<float4>(block + PXG_PDBC_normal_pen)[threadIndexInWarp] = float4(normal.x, normal.y, normal.z, pen);
        Ptr<float4>(block + PXG_PDBC_barycentric)[threadIndexInWarp] = barycentric;
        Ptr<float>(block + PXG_PDBC_maxPenBiasClamp)[threadIndexInWarp] =
            dpSelectMax(softbody[0].mPenBiasClamp, particleSystems[tParticleSystemId].mData_PenBiasClamp);
    }
}
