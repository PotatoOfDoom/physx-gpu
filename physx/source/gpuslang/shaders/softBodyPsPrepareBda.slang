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

// softBodyPsPrepareBda.slang
//
// P9.39d -- sb_particleContactPrepareLaunch
// gpusimulationcontroller/src/CUDA/softBody.cu:844 (wrapper), :640
// (barycentricProject), softBody.cuh:38 (getSoftBodyInvMass),
// PxgFEMCore.h:184 (PxgFEMParticleConstraintBlock),
// PxgSimulationCoreDesc.h:110/:118/:127/:150/:155 (the id encodings) -- all read in
// full.  Launch: PxgSoftBodyCore.cpp:2315, grid 1024, block 256.
//
// The FOURTH link of the softbody-vs-particle chain: P9.39a finds pairs, P9.39c
// writes contacts, sb_other_contact_remap_to_simLaunch (already mapped, P8.2) moves
// them onto the SIMULATION mesh, and this turns each into a solver constraint.
//
// The cloth twin is femClothPsPrepareBda (P9.31) and most of this file is that file
// with a tetrahedron where it has a triangle.  The differences are enumerated under
// Q3, Q4 and Q6 below, and they are the whole reason this is not a copy with three
// identifiers renamed.
//
// ---------------------------------------------------------------------------
// TRAPS
// ---------------------------------------------------------------------------
// Q1 -- THE TET INDEX IS A SIMULATION-MESH TET.  The contact gen encoded a
//     COLLISION-mesh tet (softbodyPrimitives.cu:2077) and the remap kernel
//     OVERWRITES pairInd1 and the barycentric with sim-mesh values before this runs.
//     So this kernel reads mSimTetIndices (@80) and mSimDeltaPos (@128), never
//     mTetIndices (@16) or mPosition_InvMass (@64).  Both index the same descriptor
//     and both compile.
//
// Q2 -- THE PARTICLE INDEX IS THE SORTED ONE.  pairInd0's low dword is the SORTED
//     index, which is what mSortedDeltaP wants.  Its sibling contact-gen kernel uses
//     BOTH kinds thirty lines apart -- the unsorted one only for the filter -- so
//     "the particle index" is not a single thing in this chain.
//
// Q3 -- getSoftBodyInvMass SCALES BY THE L2 NORM OF THE BARYCENTRIC VECTOR, and the
//     cloth twin has NO such scale: FEMClothConstraintPrep.cu:353 uses delta1.w raw
//     where softBody.cuh:38 returns `baryMass * sqrt(bx^2+by^2+bz^2+bw^2)`.  Dropping
//     it gives every softbody-particle contact the wrong stiffness, and nothing about
//     the two files' shape suggests the difference.
//     (The sqrt is written out, NOT `length(bary)` -- see the module.)
//
// Q4 -- barycentricProject IS FULL float4 ARITHMETIC and the .w that falls out is the
//     barycentrically blended INVERSE MASS, which is then Q3's input.  A port that
//     projected only .xyz would compile and lose the mass.
//     Association: ((a*bx + b*by) + c*bz) + d*bw, left to right.
//
// Q5 -- THE BLOCK INDEX AND THE LANE INDEX COME FROM DIFFERENT VARIABLES:
//     `spConstraints[workIndex / 32]` but `[threadIdx.x & 31]`.  They agree only
//     because the work-index formula keeps a block's threads contiguous.
//
// Q6 -- pen's FORM differs from the cloth twin while the value does not:
//         softbody: normal_pen.w - dot(normal, particleDelta - tetDelta) - restDist
//         cloth:    normal_pen.w + dot(normal, clothDelta - particleDelta) - restDist
//     Those are bit-identical -- negating each component before the dot and negating
//     the sum after are the same in IEEE -- but the ENGINE's spelling is the one
//     transcribed, because "equivalent" is a claim that has to be re-proved every
//     time the expression changes, and the transcription does not.
//
// Q7 -- `contacts` IS READ AND DISCARDED, exactly as in the cloth twin: the load
//     feeds a PxVec3 whose every use is commented out upstream.  The buffer stays in
//     the blob because dropping it would shift every pointer after it.
//
// Q8 -- NO maxContacts CLAMP AND NO WRITE-BACK, unlike the two sibling prepare
//     kernels in the same file (sb_softbodyContactPrepareLaunch at :941 and
//     sb_clothContactPrepareLaunch at :1018 both take maxContacts, clamp, and store
//     the clamp back through thread (0,0)).  This one does neither; transcribed as
//     written.
//
// Q9 -- NO tetInd == 0xfffff GUARD, although sb_rigidContactPrepareLaunch twenty
//     lines above (:817) has one.  Upstream asymmetry, documented and transcribed;
//     this port does not add a guard the engine does not have.
//
// Q10 -- THE APPLIED-FORCE ARRAYS ARE float2 IN float4-SIZED ALLOCATIONS.  The
//     kernel writes at stride 8 while the host allocates maxContacts float4s
//     (PxgFEMCore.cpp:160), so only the first half of each 16-byte slot is touched.
//     Writing at stride 16 would compile and silently spread the state out.

module softBodyPsPrepareBda;

import pxgSlangSoftBodyDesc;     // PxgSoftBodyGpu (mSimTetIndices@80, mSimDeltaPos@128,
                                 //                 mRestDistance@420)
import pxgSlangParticleDesc;     // PxgParticleSystemGpu (mSortedDeltaP)
import pxgSlangSoftBodySolve;    // P9.39d: getSoftBodyId / getSoftBodyElementIndex /
                                 //         getSoftBodyInvMass

static const uint SPP_GRID = 1024u;
static const uint SPP_BLOCK = 256u;

// PxgFEMParticleConstraintBlock (PxgFEMCore.h:184): float4 normal_pen[32],
// float4 barycentric[32], float velMultiplier[32] -- 1152 bytes, warp-blocked.
static const uint SPP_CONSTRAINT_STRIDE = 1152u;
static const uint SPP_OFF_NORMAL_PEN = 0u;      // float4[32]
static const uint SPP_OFF_BARYCENTRIC = 512u;   // float4[32]
static const uint SPP_OFF_VELMULT = 1024u;      // float[32]

struct SbPsPrepareParams
{
    uint64_t softbodies;               // @0
    uint64_t particleSystems;          // @8
    uint64_t contacts;                 // @16  Q7: read, never used
    uint64_t normalPens;               // @24
    uint64_t barycentrics;             // @32
    uint64_t contactInfos;             // @40  PxgFemOtherContactInfo* (16 B stride)
    uint64_t numContacts;              // @48
    uint64_t constraints;              // @56  PxgFEMParticleConstraintBlock*
    uint64_t softBodyAppliedForces;    // @64  float2*
    uint64_t particleAppliedForces;    // @72  float2* -> 80
}

[[vk::push_constant]] ConstantBuffer<SbPsPrepareParams> gSp;

// barycentricProject (softBody.cu:640) -- Q4: whole float4s, left to right.
float4 sppTetProject(uint64_t valsAddr, uint4 tetId, float4 bary)
{
    Ptr<float4> v = Ptr<float4>(valsAddr);
    const float4 a = v[tetId.x];
    const float4 b = v[tetId.y];
    const float4 c = v[tetId.z];
    const float4 d = v[tetId.w];
    return a * bary.x + b * bary.y + c * bary.z + d * bary.w;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sb_particleContactPrepareLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tNumContacts = Ptr<uint>(gSp.numContacts)[0];
    const uint nbBlocksRequired = (tNumContacts + SPP_BLOCK - 1u) / SPP_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + SPP_GRID - 1u) / SPP_GRID;

    const uint idx = gtid.x;
    const uint threadIndexInWarp = gtid.x & 31u;        // Q5

    Ptr<PxgSoftBodyGpu> softbodies = Ptr<PxgSoftBodyGpu>(gSp.softbodies);
    Ptr<PxgParticleSystemGpu> particleSystems = Ptr<PxgParticleSystemGpu>(gSp.particleSystems);

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * SPP_BLOCK + idx + nbIterationsPerBlock * gid.x * SPP_BLOCK;
        if (workIndex >= tNumContacts)
            return;

        // Q10: float2 stores into float4-sized slots.
        Ptr<float2>(gSp.softBodyAppliedForces)[workIndex] = float2(0.0f, 0.0f);
        Ptr<float2>(gSp.particleAppliedForces)[workIndex] = float2(0.0f, 0.0f);

        // PxgFemOtherContactInfo, 16 B: pairInd0 u64 @0, pairInd1 u32 @8.
        const uint64_t infoBase = gSp.contactInfos + uint64_t(workIndex) * 16UL;
        const uint2 pairInd0 = Ptr<uint2>(infoBase)[0];
        const uint pairInd1 = Ptr<uint>(infoBase + 8UL)[0];

        const uint64_t cBase = gSp.constraints + uint64_t(workIndex / 32u) * SPP_CONSTRAINT_STRIDE;

        // PxGetParticleSystemId / PxGetParticleIndex (PxgSimulationCoreDesc.h:150/:155).
        const uint tParticleSystemId = pairInd0.y;
        const uint tParticleIndex = pairInd0.x;          // Q2: the SORTED index

        const uint softbodyId = getSoftBodyId(pairInd1);
        const uint tetInd = getSoftBodyElementIndex(pairInd1);   // Q9: no 0xfffff guard

        const uint4 tetrahedronIdx = Ptr<uint4>(softbodies[softbodyId].mSimTetIndices)[tetInd];  // Q1

        // Q7: loaded and discarded, as upstream does.
        const float4 contact = Ptr<float4>(gSp.contacts)[workIndex];

        const float4 normal_pen = Ptr<float4>(gSp.normalPens)[workIndex];
        const float4 barycentric = Ptr<float4>(gSp.barycentrics)[workIndex];

        const float4 tetDelta_invMass =
            sppTetProject(softbodies[softbodyId].mSimDeltaPos, tetrahedronIdx, barycentric);  // Q1/Q4
        const float3 tetDelta = tetDelta_invMass.xyz;

        const float3 normal = float3(-normal_pen.x, -normal_pen.y, -normal_pen.z);

        const float4 delta_invMass =
            Ptr<float4>(particleSystems[tParticleSystemId].mSortedDeltaP)[tParticleIndex];
        const float3 particleDelta = delta_invMass.xyz;
        const float invMass0 = delta_invMass.w;

        // Q6: the engine's spelling, not the cloth twin's.
        const float pen = normal_pen.w - dot(particleDelta - tetDelta, normal)
                          - softbodies[softbodyId].mRestDistance;

        const float sbMass = getSoftBodyInvMass(tetDelta_invMass.w, barycentric);   // Q3

        const float unitResponse = invMass0 + sbMass;
        const float velMultiplier = (unitResponse > 0.0f) ? (1.0f / unitResponse) : 0.0f;

        Ptr<float4>(cBase + uint64_t(SPP_OFF_NORMAL_PEN))[threadIndexInWarp] =
            float4(normal.x, normal.y, normal.z, pen);
        Ptr<float4>(cBase + uint64_t(SPP_OFF_BARYCENTRIC))[threadIndexInWarp] = barycentric;
        Ptr<float>(cBase + uint64_t(SPP_OFF_VELMULT))[threadIndexInWarp] = velMultiplier;
    }
}
