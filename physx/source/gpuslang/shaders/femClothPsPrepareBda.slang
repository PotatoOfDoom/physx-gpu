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

// femClothPsPrepareBda.slang
//
// P9.31 -- cloth_particleContactPrepareLaunch
// gpusimulationcontroller/src/CUDA/FEMClothConstraintPrep.cu:283 (wrapper), :271
// (computeTriangleContact), PxgFEMCore.h:184 (PxgFEMParticleConstraintBlock),
// PxgSimulationCoreDesc.h:118/125/135/150/155 (the id encodings) -- all read in
// full.  Launch: PxgFEMClothCore.cpp:380, grid 1024, block 256.
//
// The THIRD link of the cloth-vs-particle chain P9.30 opened: the midphase
// finds pairs, the contact gen writes contacts, and this turns each contact
// into a solver constraint.
//
// ---------------------------------------------------------------------------
// TRAPS
// ---------------------------------------------------------------------------
// P1 -- THE WORK INDEX IS NOT A GRID STRIDE -- and the first version of this note
//     said that made a grid stride WRONG.  It does not, and the correction
//     belongs here because the claim was load-bearing.  The formula is
//         workIndex = i * blockDim.x + threadIdx.x
//                   + nbIterationsPerBlock * blockIdx.x * blockDim.x
//     so each BLOCK owns a CONTIGUOUS range of nbIterationsPerBlock * 256
//     indices.  A grid-stride loop visits the SAME set -- both cover [0, N)
//     exactly once -- and gives the SAME lane, because every term other than
//     threadIdx.x is a multiple of 256 and hence of 32, so `workIndex % 32 ==
//     threadIdx.x % 32` under both.  The substitution is therefore
//     OBSERVATION-EQUIVALENT, and the sweep says so by reporting that mutant as
//     a survivor; what the upstream formula buys is memory locality, not
//     correctness.
//
//     What IS observable is dropping the `nbIterationsPerBlock *` factor: the
//     per-block ranges then OVERLAP and the tail is never visited at all.  That
//     only shows up once nbIterationsPerBlock exceeds 1, i.e. past
//     gridDim * blockDim = 262144 contacts, which is why the fixture carries a
//     deliberately huge coverage run alongside its small exact one.
//
// P2 -- THE CONSTRAINT BLOCK IS WARP-BLOCKED.  `constraints[workIndex / 32]`
//     indexed by `threadIdx.x & 31` -- a struct of three lane-arrays
//     (float4[32], float4[32], float[32] = 1152 bytes), NOT an array of
//     per-contact structs.  Note the two indices come from DIFFERENT things:
//     the block from the work index, the lane from the thread id.  They agree
//     only because the work index formula keeps a block's threads contiguous.
//
// P3 -- `contacts` IS READ AND DISCARDED.  The kernel loads contacts[workIndex]
//     into `p` and every use of `p` is commented out upstream.  The load is
//     kept -- it is in the argument pack and a port that drops the buffer would
//     shift every pointer after it -- but nothing depends on the value.
//
// P4 -- THE NORMAL IS NEGATED AGAIN.  The contact gen stored -n; this reads
//     normal_pen and negates once more, so the constraint holds +n relative to
//     the original v = p - closestPt.
//
// P5 -- `pen` MIXES THREE TERMS: the stored penetration, the CURRENT relative
//     delta projected on the normal, and the cloth's rest distance:
//         pen = normal_pen.w + dot(normal, delta) - cloth.mRestDistance
//     `delta` is the cloth's accumulated delta MINUS the particle's, in that
//     order.
//
// P6 -- computeTriangleContact does FLOAT4 arithmetic: v0*b.x + v1*b.y + v2*b.z
//     over the whole float4, so the .w that falls out is the barycentrically
//     interpolated INVERSE MASS.  A port that interpolated only .xyz and then
//     fetched a mass separately gets a different number.
//
// P7 -- the early exit is a `return` INSIDE the loop, not a `break`.  There is
//     no code after the loop, so the two agree here -- but only here.
//
// P8 -- both applied-force arrays are zeroed for EVERY work index, before the
//     early exit can skip anything, and they are float2 while the host passes
//     float4-typed buffers.  The stride is 8 bytes, not 16.
// ===========================================================================

import pxgSlangFEMClothDesc;   // PxgFEMClothGpu
import pxgSlangParticleDesc;   // PxgParticleSystemGpu

static const uint PSP_GRID = 1024u;
static const uint PSP_BLOCK = 256u;

// PxgFEMParticleConstraintBlock (PxgFEMCore.h:184): three lane-arrays, 1152 B.
static const uint PSP_CONSTRAINT_STRIDE = 1152u;
static const uint PSP_OFF_NORMAL_PEN = 0u;      // float4[32]
static const uint PSP_OFF_BARYCENTRIC = 512u;   // float4[32]
static const uint PSP_OFF_VELMULT = 1024u;      // float[32]

// PXG_BITSHIFT_ELEMENT_ID / PX_MAX_NB_DEFORMABLE_SURFACE_TRI.
static const uint PXG_CLOTH_ID_SHIFT = 20u;
static const uint PXG_CLOTH_ELEM_MASK = 0x000fffffu;

// ---------------------------------------------------------------------------
// Push-constant blob -- BYTE-IDENTICAL to the CUDA pack
// (PxgFEMClothCore.cpp:382): ten pointers -> 80 bytes.  ABI row {8 x 10}.
// ---------------------------------------------------------------------------
struct ClothPsPrepareParams
{
    uint64_t clothes;                  // @0
    uint64_t particleSystems;          // @8
    uint64_t contacts;                 // @16  P3: read, never used
    uint64_t normalPens;               // @24
    uint64_t barycentrics;             // @32
    uint64_t contactInfos;             // @40  PxgFemOtherContactInfo* (16 B stride)
    uint64_t numContacts;              // @48
    uint64_t constraints;              // @56  PxgFEMParticleConstraintBlock*
    uint64_t clothAppliedForces;       // @64  float2*
    uint64_t particleAppliedForces;    // @72  float2* -> 80
}

[[vk::push_constant]] ConstantBuffer<ClothPsPrepareParams> gPp;

// computeTriangleContact (FEMClothConstraintPrep.cu:271) -- P6: float4 through.
float4 pspTriangleContact(uint64_t velsAddr, uint4 triVertId, float4 barycentric)
{
    Ptr<float4> vels = Ptr<float4>(velsAddr);
    const float4 v0 = vels[triVertId.x];
    const float4 v1 = vels[triVertId.y];
    const float4 v2 = vels[triVertId.z];
    return v0 * barycentric.x + v1 * barycentric.y + v2 * barycentric.z;
}

// ===========================================================================
[shader("compute")]
[numthreads(256, 1, 1)]
void cloth_particleContactPrepareLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tNumContacts = Ptr<uint>(gPp.numContacts)[0];

    const uint nbBlocksRequired = (tNumContacts + PSP_BLOCK - 1u) / PSP_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + PSP_GRID - 1u) / PSP_GRID;

    const uint idx = gtid.x;
    const uint threadIndexInWarp = gtid.x & 31u;

    Ptr<PxgFEMClothGpu> clothes = Ptr<PxgFEMClothGpu>(gPp.clothes);
    Ptr<PxgParticleSystemGpu> particleSystems = Ptr<PxgParticleSystemGpu>(gPp.particleSystems);

    for (uint i = 0u; i < nbIterationsPerBlock; ++i)
    {
        // P1: a contiguous per-block range, NOT a grid stride.
        const uint workIndex = i * PSP_BLOCK + idx + nbIterationsPerBlock * gid.x * PSP_BLOCK;

        if (workIndex >= tNumContacts)
            return;                                          // P7

        // P8: float2, and zeroed before anything else happens.
        Ptr<float2>(gPp.clothAppliedForces)[workIndex] = float2(0.0f, 0.0f);
        Ptr<float2>(gPp.particleAppliedForces)[workIndex] = float2(0.0f, 0.0f);

        // PxgFemOtherContactInfo (16 B): pairInd0 u64 @0, pairInd1 u32 @8.
        const uint64_t infoBase = gPp.contactInfos + uint64_t(workIndex) * 16UL;
        const uint2 pairInd0 = Ptr<uint2>(infoBase + 0UL)[0];
        const uint pairInd1 = Ptr<uint>(infoBase + 8UL)[0];

        // PxEncodeParticleIndex: (u64(systemId) << 32) | particleId.
        const uint tParticleSystemId = pairInd0.y;
        const uint tParticleIndex = pairInd0.x;

        const uint clothId = pairInd1 >> PXG_CLOTH_ID_SHIFT;
        const uint triangleInd = pairInd1 & PXG_CLOTH_ELEM_MASK;

        const uint4 triVertInd = Ptr<uint4>(clothes[clothId].mTriangleVertexIndices)[triangleInd];

        // P3: loaded because the CUDA loads it; every use of it is commented out.
        const float4 contact = Ptr<float4>(gPp.contacts)[workIndex];
        const float4 normal_pen = Ptr<float4>(gPp.normalPens)[workIndex];
        const float4 barycentric = Ptr<float4>(gPp.barycentrics)[workIndex];

        // P6: the whole float4, so .w is the interpolated inverse mass.
        const float4 delta1 = pspTriangleContact(clothes[clothId].mAccumulatedDeltaPos,
                                                 triVertInd, barycentric);
        const float invMass1 = delta1.w;

        const float3 normal = float3(-normal_pen.x, -normal_pen.y, -normal_pen.z);   // P4

        const float4 deltaP_invMass =
            Ptr<float4>(particleSystems[tParticleSystemId].mSortedDeltaP)[tParticleIndex];
        const float invMass0 = deltaP_invMass.w;

        // P5: cloth delta MINUS particle delta, in that order.
        const float3 delta = float3(delta1.x - deltaP_invMass.x,
                                    delta1.y - deltaP_invMass.y,
                                    delta1.z - deltaP_invMass.z);

        const float pen = normal_pen.w + dot(normal, delta) - clothes[clothId].mRestDistance;

        const float unitResponse = invMass0 + invMass1;
        const float velMultiplier = (unitResponse > 0.0f) ? (1.0f / unitResponse) : 0.0f;

        // P2: block from the WORK index, lane from the THREAD id.
        const uint64_t cBase = gPp.constraints +
                               uint64_t(workIndex / 32u) * uint64_t(PSP_CONSTRAINT_STRIDE);
        Ptr<float4>(cBase + uint64_t(PSP_OFF_NORMAL_PEN))[threadIndexInWarp] =
            float4(normal.x, normal.y, normal.z, pen);
        Ptr<float4>(cBase + uint64_t(PSP_OFF_BARYCENTRIC))[threadIndexInWarp] = barycentric;
        Ptr<float>(cBase + uint64_t(PSP_OFF_VELMULT))[threadIndexInWarp] = velMultiplier;
    }
}
