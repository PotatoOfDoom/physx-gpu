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

// npPsBoundaryDensity.slang
//
// P9.8 -- the PBD fluid BOUNDARY density pair:
//   ps_accumulateStaticDensity accumulates, per particle, the SPH density
//     contribution of its contacts with STATIC geometry into mStaticDensity;
//   ps_accumulateRigidDensity does the same for contacts with RIGID BODIES,
//     SEEDING its accumulator from what the static kernel just wrote.
//
// The pair is what makes a fluid feel a boundary: without it a fluid column has no
// density contribution from the walls it touches, so it behaves near a boundary as
// if the boundary were open.  Both are gated host-side by
// (mMaxParticles > 0 && mHasNonZeroFluidBoundaryScale)
// (PxgPBDParticleSystemCore.cpp:1118, :1141), and mHasNonZeroFluidBoundaryScale is
// set only when some PARTICLE SYSTEM carries fluidBoundaryDensityScale > 0 (:2167) --
// a per-system value off PxPBDParticleSystem::setFluidBoundaryDensityScale, NOT a
// material coefficient.  Its default is 0, so no scene launches this pair today.
//
// THE TWO ARE CHAINED, and that is the first thing to get right: the rigid kernel
// starts from `rho = mStaticDensity[tParticleId]` rather than from zero, so it must
// run after the static one and a fixture must exercise them in sequence.
//
// LAUNCHES, and they are NOT alike:
//   static: grid (ceil(mMaxParticles / PS_COLLISION), nbActiveSystems, 1),
//           block PS_COLLISION = 256, args (particleSystems*, active*) -> {8,8}
//   rigid:  grid (PxgParticleSystemKernelGridDim::PS_COLLISION = 1024, 1, 1),
//           block 256, args (particleSystems*, sortedContacts*, totalNumPairs*,
//           rangeStart*, rangeEnd*) -> {8,8,8,8,8}
//
// Traps, each with a mutant:
//  B1. The static kernel takes its system from blockIdx.Y, not blockIdx.Z.  Every
//      cloth kernel in this port uses Z; this one does not, and a port that carries
//      the convention over reads the wrong system.
//  B2. The contact loop is COLUMN-MAJOR with a numParticles stride:
//      `for (i = 0, index = gti; i < count; ++i, index += numParticles)`.
//      Row-major (gti * count + i) is wrong.
//  B3. The contact count is CLAMPED to MaxStaticContactsPerParticle = 12
//      (PxgParticleSystem.h:133) -- the stored count can exceed it.
//  B4. `W(x, kSpiky1, kInvRadius) = kSpiky1 * sqr(1 - x * kInvRadius)`
//      (particleSystem.cuh:60-63).  It is NOT clamped at x > radius, so a contact
//      beyond the radius contributes a POSITIVE value again (the square).
//  B5. The penetration is offset and one-sided: `PxMax(0, pen + restDistance)`.
//  B6. The store is one-sided too: `PxMin(rho, restDensityBoundary)`.
//  B7. Both kernels early-out on `restDensityBoundary > 0`, read from the
//      DESCRIPTOR, not from a material.
//  B8. The rigid kernel has NO shared staging and NO active-system indirection: it
//      recovers the system from the CONTACT, via the PxU64 `particleId` whose low
//      word is the particle index and whose high word is the system id (the same
//      packing psContactSolveBda.slang:633-635 already reads as a uint2).
//  B9. The rigid kernel's inner loop SHADOWS the outer loop's `i` in the CUDA.  C++
//      scoping makes that harmless, but it is transcribed with distinct names
//      because Slang does not allow the shadowing.
// B10. The guard inside the rigid kernel's outer loop is a `return`, not a
//      `continue` -- once a thread's workIndex passes the end it leaves entirely.
//      That is transcribed for fidelity, but it is PROVABLY EQUIVALENT to a
//      `continue` here and the tranche note says so rather than claiming a trap:
//      workIndex is strictly increasing in the loop counter (the only term that
//      varies is `i * 512`), so once one iteration is out of range every later one
//      is too.  A mutant that spells it `continue` is an equivalence, not a
//      survivor to be excused.
// B11. UPSTREAM CANDIDATE 16, transcribed as-is.  The rigid kernel's work index is
//          workIndex = i * ACCUMULATE_DELTA + idx + nbIterationsPerBlock * blockIdx.x * blockDim.x
//      but ACCUMULATE_DELTA is 512 while the kernel is LAUNCHED with blockDim.x =
//      PS_COLLISION = 256 (…KernelIndices.h:42 and :45).  The iteration stride is
//      therefore twice the block's width.  It is invisible whenever
//      nbIterationsPerBlock == 1 -- the `i * 512` term is multiplied by zero -- which
//      holds for every scene with at most 256 * 1024 = 262144 particle-rigid contact
//      pairs.  Above that, block bx iteration i covers
//          [512*bx + 512*i, 512*bx + 512*i + 256)
//      so every work index whose remainder mod 512 is at least 256 is covered by NO
//      block at all, while [512*(bx+1), 512*(bx+1) + 256) is covered by BOTH block
//      bx's second iteration and block bx+1's first.  BOTH halves are wrong, and
//      the second one is the part that is easy to get wrong on paper:
//        * the skipped half keeps only its static boundary density and never
//          receives the rigid contribution;
//        * the doubly-covered half is a genuine READ-MODIFY-WRITE RACE, because
//          `rho` is seeded from mStaticDensity[tParticleId] -- the very array the
//          kernel then stores to.  The two threads are in different blocks with no
//          ordering between them, so the result is either `static + sum` or
//          `static + 2*sum` depending on whether the second read happens before or
//          after the first write.  MEASURED, not reasoned: on llvmpipe the P9.8
//          fixture observes the doubled value on all 130817 doubly-covered indices,
//          at a ratio of exactly 2.0.  (On 550 of them the two admissible outcomes
//          coincide because the contribution falls below the accumulator's ULP, so
//          the fixture does not claim to distinguish those.)
//      This port reproduces the arithmetic exactly, including the 512, so the port
//      inherits the race as well.  Do NOT "fix" it.
//
// The module is in PXG_NOCONTRACT_SHADERS: W() is dense float work.

import pxgSlangParticleDesc;

// PxgParticleSystemKernelBlockDim (…KernelIndices.h:42, :45)
static const uint PS_COLLISION_BLOCK = 256u;
static const uint PS_ACCUMULATE_DELTA_BLOCK = 512u;   // B11: NOT the launch width

// PxgParticleSystemKernelGridDim::PS_COLLISION (…KernelIndices.h:59).  The rigid
// kernel is launched with a FIXED grid width, so gridDim.x is a compile-time
// constant on our side rather than something read from a dispatch parameter.
static const uint PS_COLLISION_GRID = 1024u;

// PxgParticleContactInfo::MaxStaticContactsPerParticle (PxgParticleSystem.h:133)
static const uint PS_MAX_STATIC_CONTACTS_PER_PARTICLE = 12u;

// sizeof(PxgParticlePrimitiveContact): PxVec4 + 2 * PxU64, align 16.
static const uint PXG_PRIMITIVE_CONTACT_STRIDE = 32u;

float bdSqr(float x) { return x * x; }
float bdMax(float a, float b) { return a > b ? a : b; }
float bdMin(float a, float b) { return a < b ? a : b; }

// B4: particleSystem.cuh:60-63, verbatim.  No radius clamp.
float bdW(float x, float kSpiky1, float kInvRadius)
{
    return kSpiky1 * bdSqr(1.0f - x * kInvRadius);
}

// ============================================================================
// ps_accumulateStaticDensity
// ============================================================================
struct PsStaticDensityParams
{
    uint64_t particleSystems;         // @0
    uint64_t activeParticleSystems;   // @8 -> 16
}
[[vk::push_constant]]
ConstantBuffer<PsStaticDensityParams> gSd;

[shader("compute")]
[numthreads(256, 1, 1)]
void ps_accumulateStaticDensity(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gSd.particleSystems);
    // B1: the system is on gridDim.Y here, not Z.
    const uint particleId = (Ptr<uint>(gSd.activeParticleSystems))[groupId.y];

    const float restDensityBoundary = systems[particleId].mData_RestDensityBoundary;

    if (restDensityBoundary > 0.0f)                                   // B7
    {
        const uint numParticles = systems[particleId].mCommonData_NumParticles;
        const uint groupThreadIdx = groupThreadId.x + groupId.x * PS_COLLISION_BLOCK;

        if (groupThreadIdx < numParticles)
        {
            // PxgParticleContactInfo is a bare float4 (PxgParticleSystem.h:131-136).
            Ptr<float4> contacts = Ptr<float4>(systems[particleId].mOneWayContactInfos);
            Ptr<uint> contactCounts = Ptr<uint>(systems[particleId].mOneWayContactCount);

            // B3: the stored count is clamped to twelve.
            const uint contactCount =
                min(contactCounts[groupThreadIdx], PS_MAX_STATIC_CONTACTS_PER_PARTICLE);

            const float spiky1 = systems[particleId].mData_Spiky1;
            const float particleContactDistanceInv =
                systems[particleId].mCommonData_ParticleContactDistanceInv;
            const float restDistance = systems[particleId].mData_FluidRestOffset;

            float rho = 0.0f;
            // B2: COLUMN-major, stride numParticles.
            uint index = groupThreadIdx;
            for (uint i = 0u; i < contactCount; ++i, index += numParticles)
            {
                const float4 contact = contacts[index];
                // B5: offset by the rest distance, then one-sided.
                rho += bdW(bdMax(0.0f, contact.w + restDistance), spiky1,
                           particleContactDistanceInv);
            }

            // B6: one-sided.
            (Ptr<float>(systems[particleId].mStaticDensity))[groupThreadIdx] =
                bdMin(rho, restDensityBoundary);
        }
    }
}

// ============================================================================
// ps_accumulateRigidDensity
// ============================================================================
struct PsRigidDensityParams
{
    uint64_t particleSystems;   // @0
    uint64_t sortedContacts;    // @8
    uint64_t totalNumPairs;     // @16
    uint64_t rangeStart;        // @24
    uint64_t rangeEnd;          // @32 -> 40
}
[[vk::push_constant]]
ConstantBuffer<PsRigidDensityParams> gRd;

[shader("compute")]
[numthreads(256, 1, 1)]
void ps_accumulateRigidDensity(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gRd.particleSystems);

    const uint numTests = (Ptr<uint>(gRd.totalNumPairs))[0];
    const uint nbBlocksRequired = (numTests + PS_COLLISION_BLOCK - 1u) / PS_COLLISION_BLOCK;
    // gridDim.x is PxgParticleSystemKernelGridDim::PS_COLLISION = 1024, a FIXED grid.
    const uint nbIterationsPerBlock =
        (nbBlocksRequired + PS_COLLISION_GRID - 1u) / PS_COLLISION_GRID;

    const uint idx = groupThreadId.x;

    Ptr<uint> rangeStart = Ptr<uint>(gRd.rangeStart);
    Ptr<uint> rangeEnd = Ptr<uint>(gRd.rangeEnd);

    for (uint it = 0u; it < nbIterationsPerBlock; ++it)
    {
        // B11: the stride is ACCUMULATE_DELTA (512) while the block is 256 wide.
        const uint workIndex = it * PS_ACCUMULATE_DELTA_BLOCK + idx
                             + nbIterationsPerBlock * groupId.x * PS_COLLISION_BLOCK;

        if (workIndex >= numTests)                                    // B10: a RETURN
            return;

        const uint startIndex = rangeStart[workIndex];
        const uint endIndex = rangeEnd[workIndex];

        // B8: the system comes from the CONTACT, not from an active list.  The
        // PxU64 particleId sits at byte 24 of the 32-byte record; low word is the
        // particle index, high word the system id.
        const uint64_t firstContact =
            gRd.sortedContacts + uint64_t(startIndex) * uint64_t(PXG_PRIMITIVE_CONTACT_STRIDE);
        const uint2 combinedParticleId = (Ptr<uint2>(firstContact + 24UL))[0];
        const uint tParticleSystemId = combinedParticleId.y;
        const uint tParticleId = combinedParticleId.x;

        const float restDensityBoundary = systems[tParticleSystemId].mData_RestDensityBoundary;

        if (restDensityBoundary > 0.0f)                               // B7
        {
            const float spiky1 = systems[tParticleSystemId].mData_Spiky1;
            const float particleContactDistanceInv =
                systems[tParticleSystemId].mCommonData_ParticleContactDistanceInv;
            const float restDistance = systems[tParticleSystemId].mData_FluidRestOffset;

            Ptr<float> staticDensity = Ptr<float>(systems[tParticleSystemId].mStaticDensity);

            // THE CHAIN: seeded from what ps_accumulateStaticDensity wrote.
            float rho = staticDensity[tParticleId];

            // B9: the CUDA shadows the outer `i` here; spelled distinctly.
            for (uint c = startIndex; c < endIndex; ++c)
            {
                const uint64_t rec =
                    gRd.sortedContacts + uint64_t(c) * uint64_t(PXG_PRIMITIVE_CONTACT_STRIDE);
                const float4 normalPen = (Ptr<float4>(rec))[0];
                rho += bdW(bdMax(0.0f, normalPen.w + restDistance), spiky1,
                           particleContactDistanceInv);
            }

            staticDensity[tParticleId] = bdMin(rho, restDensityBoundary);   // B6
        }
    }
}
