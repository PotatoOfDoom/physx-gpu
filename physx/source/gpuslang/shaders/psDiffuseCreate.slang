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

// psDiffuseCreate.slang -- P9.13d, ps_diffuseParticleCreate
// (diffuseParticles.cu:386-500, read in full).  THE kernel of the diffuse chain:
// every other one moves, ages, sorts or collides diffuse particles, and this is
// the only one that makes any.
//
// ---------------------------------------------------------------------------
// IT ITERATES THE **PARTICLE** BUFFERS, NOT THE DIFFUSE ONES
// ---------------------------------------------------------------------------
// blockIdx.y indexes `mParticleSimBuffers` and is bounded by
// mCommonData.mNumParticleBuffers.  The diffuse buffer is then reached
// INDIRECTLY, through `buffer.mDiffuseParticleBufferIndex`, and that field is
// **0xffffffff for a particle buffer with no diffuse side** -- which is the
// normal case for every plain PxParticleBuffer in a scene that also has a
// diffuse one.  So the guard is not defensive; it is the common path.
//
// Everything else in the chain (P9.13c's three) walks mDiffuseSimBuffers
// directly and is bounded by mNumDiffuseBuffers.  The two index spaces are
// different sizes and are NOT in the same order.
//
// ---------------------------------------------------------------------------
// TRAPS
// ---------------------------------------------------------------------------
// C1 -- the grid is (ceil(maxParticlesPerBuffer/256), maxBuffersPerSystem,
//     nbActiveSystems): system from blockIdx.**z**, PARTICLE buffer from .y.
//     Note the y extent is mMaxBuffersPerSystem, not the diffuse one.
//
// C2 -- `pi` is the buffer-LOCAL particle index and everything downstream is
//     indexed by `sortedInd = mUnsortedToSortedMapping[pi + runsum[bufferIndex]]`.
//     Three different index spaces in two lines: buffer-local, flat-unsorted,
//     and sorted.  Getting any of them wrong still produces plausible output.
//
// C3 -- THE RANDOM TABLE IS INDEXED BY `sortedInd`, NOT BY THE THREAD.
//     `randomTable[(sortedInd + k) % randomTableSize]` for k = 0..3.  So the
//     randomness follows the particle's identity, and a port that uses `pi`
//     or the global thread index still spawns a plausible-looking spray.
//
// C4 -- the intensity is `pressure - divergence + kineticEnergy`, and the
//     DIVERGENCE IS SUBTRACTED.  All three weights are per-buffer parameters.
//     kineticEnergy is `dot3(v, v)` -- the SQUARED speed, unhalved and unmassed.
//
// C5 -- the spawn test is `r0 * intensity > threshold`: the random number
//     multiplies the intensity rather than being compared against a
//     probability.  A negative intensity therefore never spawns (r0 >= 0), and
//     doubling the threshold is NOT the same as halving the intensity in the
//     tail, because r0 is drawn per particle.
//
// C6 -- `atomicAdd(&numDiffuseParticles[1], 1)` ALWAYS runs for a qualifying
//     particle; only the WRITE is guarded by `newIndex < mMaxNumParticles`.
//     The staging counter therefore overshoots the capacity on a busy frame,
//     and it is P9.13c's ps_diffuseParticleCopy that clamps it back.  A port
//     that guards the atomic instead of the write keeps the counter tidy and
//     changes what Copy publishes.
//
// C7 -- that comparison is `int < PxU32` in the CUDA, so the int is converted
//     to unsigned.  Transcribed as an unsigned compare.
//
// C8 -- lifetime: `lifeMin` is a hard-coded 1.0f, `lifeScale =
//     min(intensity / threshold, 1) * r1`, `lifetime = lifeMin + lifeScale *
//     (lifeMax - lifeMin)`.  So r1 scales the WHOLE range, and a particle that
//     only just crosses the threshold still gets at least 1.0.
//
// C9 -- the spawn position is
//         q = xi - r2 * vi * dt + float3(r1, r2, r3) * restOffset * 0.25
//     and **r2 appears twice**: once as the velocity back-step scale and again
//     as the y component of the jitter.  Not a typo to tidy.
//
// C10 -- the new velocity is the PARENT's velocity with w = 0, while the new
//     position's w is the lifetime.  The two float4s carry different meanings
//     in the same slot, which is what P9.13b's predict already had to respect.
//
// C11 -- only FLUID particles spawn (`PxGetFluid(phase)` on the SORTED phase
//     array), and the phase is read at `sortedInd` like everything else.
//
// C12 -- `mMaxNumParticles == 0` returns before the potential is even read, and
//     the CUDA's `for (int i = 0; i < 5; ++i)` around the spawn is COMMENTED
//     OUT: exactly one attempt per qualifying particle.
//
// C13 -- the CUDA block-copies the whole PxgParticleSystem into shared memory
//     and then reads `mParticleBufferRunsum` from the GLOBAL descriptor anyway
//     (:441 against :435).  Both are the same bytes; the port reads fields
//     directly, as every other particle shader here does, and the inconsistency
//     is recorded rather than reproduced.

import pxgSlangCommon;
import pxgSlangParticleDesc;

// PxgParticleSystemKernelBlockDim::PS_SOLVE (PxgParticleSystemCoreKernelIndices.h:46)
static const uint PSDC_SOLVE = 256u;
static const uint PSDC_NO_DIFFUSE_BUFFER = 0xffffffffu;

// ps_diffuseParticleCreate(particleSystems, activeParticleSystems, randomTable,
//                          randomTableSize, dt) -> {8,8,8,4,4} = 32 bytes.
struct PsDiffCreatePush
{
    uint64_t particleSystems;        // @0
    uint64_t activeParticleSystems;  // @8
    uint64_t randomTable;            // @16
    uint randomTableSize;            // @24
    float dt;                        // @28 -> 32
}
[[vk::push_constant]]
ConstantBuffer<PsDiffCreatePush> gDc;

[shader("compute")]
[numthreads(256, 1, 1)]
void ps_diffuseParticleCreate(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gDc.particleSystems);
    const uint id = Ptr<uint>(gDc.activeParticleSystems)[gid.z];            // C1: .z

    const uint bufferIndex = gid.y;
    if (bufferIndex >= systems[id].mCommonData_NumParticleBuffers)          // the PARTICLE buffers
        return;

    Ptr<PxgParticleSimBufferGpu> bufs =
        Ptr<PxgParticleSimBufferGpu>(systems[id].mParticleSimBuffers);

    const uint diffuseParticleBufferIndex = bufs[bufferIndex].mDiffuseParticleBufferIndex;
    if (diffuseParticleBufferIndex == PSDC_NO_DIFFUSE_BUFFER)               // the common path
        return;

    const uint pi = gtid.x + gid.x * PSDC_SOLVE;
    const uint numParticles = bufs[bufferIndex].mNumActiveParticles;
    if (pi >= numParticles)
        return;

    Ptr<PxgParticleDiffuseSimBufferGpu> dbufs =
        Ptr<PxgParticleDiffuseSimBufferGpu>(systems[id].mDiffuseSimBuffers);
    const uint maxNumParticles = dbufs[diffuseParticleBufferIndex].mMaxNumParticles;
    if (maxNumParticles == 0u)                                             // C12
        return;

    // C2: buffer-local -> flat-unsorted -> sorted, in two steps.
    const uint offset = Ptr<uint>(systems[id].mParticleBufferRunsum)[bufferIndex];  // C13
    const uint sortedInd = Ptr<uint>(systems[id].mUnsortedToSortedMapping)[pi + offset];

    const uint phase = Ptr<uint>(systems[id].mSortedPhaseArray)[sortedInd];
    if (!psGetFluid(phase))                                                // C11
        return;

    // float2 potentials: .x is the divergence term, .y the pressure term.
    const uint64_t potAddr = systems[id].mDiffusePotentials + uint64_t(sortedInd) * 8UL;
    const float ptntsX = Ptr<float>(potAddr)[0];
    const float ptntsY = Ptr<float>(potAddr + 4UL)[0];

    const float threshold = dbufs[diffuseParticleBufferIndex].mParams_threshold;

    const float4 vi4 = Ptr<float4>(systems[id].mSortedVelocities)[sortedInd];

    // C4: dot3 of the velocity with itself, and the divergence SUBTRACTED.
    const float kineticEnergy = (vi4.x * vi4.x + vi4.y * vi4.y + vi4.z * vi4.z)
                              * dbufs[diffuseParticleBufferIndex].mParams_kineticEnergyWeight;
    const float divergence = dbufs[diffuseParticleBufferIndex].mParams_divergenceWeight * ptntsX;
    const float pressure = dbufs[diffuseParticleBufferIndex].mParams_pressureWeight * ptntsY;
    const float intensity = pressure - divergence + kineticEnergy;

    Ptr<float> randomTable = Ptr<float>(gDc.randomTable);
    const uint rts = gDc.randomTableSize;
    const float r0 = randomTable[(sortedInd + 0u) % rts];                  // C3

    if (r0 * intensity > threshold)                                        // C5
    {
        const float4 xi4 = Ptr<float4>(systems[id].mSortedPositions_InvMass)[sortedInd];

        // C6: the atomic ALWAYS runs; only the store is capacity-guarded.
        uint newIndex;
        InterlockedAdd(Ptr<uint>(dbufs[diffuseParticleBufferIndex].mNumDiffuseParticles)[1],
                       1u, newIndex);

        if (newIndex < maxNumParticles)                                    // C7: unsigned
        {
            const float3 xi = float3(xi4.x, xi4.y, xi4.z);
            const float3 vi = float3(vi4.x, vi4.y, vi4.z);

            const float r1 = randomTable[(sortedInd + 1u) % rts];
            const float r2 = randomTable[(sortedInd + 2u) % rts];
            const float r3 = randomTable[(sortedInd + 3u) % rts];

            const float lifeMin = 1.0f;                                    // C8
            const float lifeMax = dbufs[diffuseParticleBufferIndex].mParams_lifetime;
            const float lifeScale = min(intensity / threshold, 1.0f) * r1;
            const float lifetime = lifeMin + lifeScale * (lifeMax - lifeMin);

            // C9: r2 twice -- the velocity back-step AND the y jitter.
            const float restOffset = systems[id].mData_RestOffset;
            const float3 q = xi - r2 * vi * gDc.dt
                           + float3(r1, r2, r3) * restOffset * 0.25f;

            // C10: position.w is the lifetime, velocity.w is a hard zero.
            Ptr<float4>(dbufs[diffuseParticleBufferIndex].mDiffusePositions_LifeTime)[newIndex] =
                float4(q.x, q.y, q.z, lifetime);
            Ptr<float4>(dbufs[diffuseParticleBufferIndex].mDiffuseVelocities)[newIndex] =
                float4(vi.x, vi.y, vi.z, 0.0f);
        }
    }
}
