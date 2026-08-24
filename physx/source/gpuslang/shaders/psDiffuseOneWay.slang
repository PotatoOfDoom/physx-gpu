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

// psDiffuseOneWay.slang -- P9.13f, the CONSUMER of the diffuse one-way contact
// list.  Ported from
// gpusimulationcontroller/src/CUDA/diffuseParticles.cu:111-163
// (ps_diffuseParticleOneWayCollision), read in full.
//
// Per diffuse particle it walks that particle's static-contact list and pushes
// the particle's PREDICTED position out along each contact normal by that
// contact's penetration.  It is the only reader of mDiffuseOneWayContactInfos /
// mDiffuseOneWayContactCount; the two producers
// (ps_primitivesDiffuseCollisionLaunch, ps_convexDiffuseCollisionLaunch) are
// still unmapped, so until they land this kernel runs over a contact list that
// is always empty.  That is stated here rather than left for a reader to
// discover from a scene that does not move.
//
// ---------------------------------------------------------------------------
// TRAPS
// ---------------------------------------------------------------------------
// O1 -- grid (ceil(maxDiffuseParticles/256), nbActiveSystems, 1), block
//     (256,1,1) = PxgParticleSystemKernelBlockDim::PS_SOLVE.  The system comes
//     from blockIdx.**y** (PxgPBDParticleSystemCore.cpp:1201).  P9.13e's two
//     kernels take it from .y and .z respectively; this is a third launch shape
//     in the same chain and none of them may be copied from a neighbour.
//
// O2 -- THE THIRD ARGUMENT IS DEAD.  `count` is a host-side
//     `static PxU32 count; count++;` (PxgPBDParticleSystemCore.cpp:1184) passed
//     every frame and never read by the kernel body.  It is NOT the particle
//     count, and a port that uses it as the loop bound gets a frame counter.
//     It still occupies its 4 bytes in the push-constant blob, so the ABI row
//     carries it: kPsDiffCount { 8, 8, 4 }, already in the table.
//
// O3 -- the bound is `*mNumDiffuseParticles`, i.e. slot **[0]** of the two-slot
//     pair P9.13c documented: the PUBLISHED count, not the staging count.
//
// O4 -- the contact list is COLUMN-MAJOR with a stride of numParticles:
//     `for (c = 0, offset = pi; c < contactCount; ++c, offset += numParticles)`.
//     Contact c of particle pi lives at `pi + c * numParticles`, not at
//     `pi * maxContacts + c`.  Both index a valid element of the allocation, so
//     the wrong one reads a real contact belonging to another particle.
//
// O5 -- `contactCount = PxMin(MaxStaticContactsPerParticle, counts[pi])` and the
//     clamp is LOAD-BEARING, not defensive.  The producer's atomicAdd runs on
//     every intersection and only the WRITE is capped, so the stored count
//     routinely exceeds 12 while slots 12.. were never written.  Dropping the
//     clamp reads uninitialised memory.  (The same asymmetry as P9.13d's C6.)
//
// O6 -- `deltaP = -surfaceNormal * pen`, and `pen` is `distance - restDistance`,
//     which is NEGATIVE while the particle is inside.  Two negatives: the
//     correction moves ALONG +normal, i.e. out of the shape.  Dropping either
//     sign pushes the particle further in and still looks like a correction.
//
// O7 -- the accumulation is a strict left-to-right fold over the contacts, one
//     rounding per step, so contact ORDER is observable in the low bits.  The
//     order is the list order, which is itself the producer's atomic order --
//     a race.  This kernel must not reorder it; what may not be ASSERTED is
//     which order the producer produced (the P9.11 rule).
//
// O8 -- with contactCount == 0 the CUDA writes NOTHING.  The store is inside the
//     `if (contactCount)`, and this port keeps the early return so the same
//     addresses stay untouched.
//
//     But it is an OPTIMISATION, not a semantic, and the sweep proved it: the
//     mutant that removes the guard (o8-store-when-empty) SURVIVES, and it is
//     equivalent for every input, not merely for the fixture's.  With no
//     contacts `posCorr` is `p0.xyz` unmodified -- no arithmetic touches it --
//     so the store writes back exactly the bits the load returned, to a slot
//     one thread owns, in a dispatch where nothing else reads it.  A poison
//     comparison cannot see it either: the poisoned slack lies past
//     numParticles, and that region is guarded by the `pi >= numParticles`
//     return, not by the contact count.  Recorded as an equivalence with its
//     proof rather than chased.
//
// O9 -- the target is `mDiffuseSortedPos_LifeTime` -- the sorted PREDICTED
//     position, which is what P9.13e's Compact reads as `diffusePositions`.
//     NOT `mDiffuseSortedOriginPos_LifeTime`, the array P9.13e's UpdatePBF
//     overloads with velocity averages.  Both are float4, both are sorted, and
//     both are one field apart in the descriptor.
//
// O10 -- the `.w` (lifetime) passes through.  The CUDA re-reads `newPos[pi].w`
//     in the store expression, i.e. a SECOND load after the loop; nothing wrote
//     it in between and one thread owns pi, so the single load below is
//     equivalent.  Recorded rather than silently collapsed.
//
// Deviation, deliberate: the CUDA blockCopy's the whole PxgParticleSystem into
// shared memory and __syncthreads() before doing anything.  That is a CUDA
// caching device with no observable effect -- every field it copies is read
// straight through BDA here -- so the copy and its barrier are dropped.  The
// barrier is unconditional in the CUDA (it precedes the `pi >= numParticles`
// return), so dropping it cannot change which threads reach what.

import pxgSlangCommon;
import pxgSlangParticleDesc;

// PxgParticleContactInfo (PxgParticleSystem.h:131) is a single float4,
// 16 bytes, PX_ALIGN(16) -- mNormal_PenW and nothing else.  The array is read
// as float4 directly rather than through a one-field struct.
static const uint PS_MAX_STATIC_CONTACTS_PER_PARTICLE = 12u;  // ...::MaxStaticContactsPerParticle

// PxgParticleSystemKernelBlockDim::PS_SOLVE
static const uint PSO_SOLVE_BLOCK = 256u;

struct PsDiffOneWayPush
{
    uint64_t particleSystems;        // @0
    uint64_t activeParticleSystems;  // @8
    uint count;                      // @16 -> 24  (O2: DEAD)
    uint pad0;
}
[[vk::push_constant]]
ConstantBuffer<PsDiffOneWayPush> gOw;

[shader("compute")]
[numthreads(256, 1, 1)]
void ps_diffuseParticleOneWayCollision(uint3 dtid : SV_DispatchThreadID, uint3 gid : SV_GroupID)
{
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gOw.particleSystems);
    const uint id = Ptr<uint>(gOw.activeParticleSystems)[gid.y];          // O1: .y

    const uint pi = dtid.x;

    // O3: slot [0], the published count.
    const uint numParticles = uint(Ptr<int>(systems[id].mNumDiffuseParticles)[0]);
    if (pi >= numParticles)
        return;

    Ptr<uint> counts = Ptr<uint>(systems[id].mDiffuseOneWayContactCount);
    const uint rawCount = counts[pi];
    // O5: the producer overshoots this cap on purpose; the clamp is what keeps
    // the read inside the region that was written.
    const uint contactCount = min(PS_MAX_STATIC_CONTACTS_PER_PARTICLE, rawCount);

    // O8: no contacts -> no store at all.
    if (contactCount == 0u)
        return;

    Ptr<float4> newPos = Ptr<float4>(systems[id].mDiffuseSortedPos_LifeTime);   // O9
    Ptr<float4> contacts = Ptr<float4>(systems[id].mDiffuseOneWayContactInfos);

    const float4 p0 = newPos[pi];
    float3 posCorr = p0.xyz;

    // O4: column-major, stride numParticles.  O7: strict left-to-right fold.
    uint offset = pi;
    for (uint c = 0u; c < contactCount; ++c, offset += numParticles)
    {
        const float4 normalPen = contacts[offset];
        const float3 surfaceNormal = normalPen.xyz;
        // O6: pen is negative while penetrating, and the normal is negated, so
        // the correction moves the particle OUT.
        const float3 deltaP = -surfaceNormal * normalPen.w;
        posCorr += deltaP;
    }

    newPos[pi] = float4(posCorr, p0.w);   // O10: lifetime passes through
}
