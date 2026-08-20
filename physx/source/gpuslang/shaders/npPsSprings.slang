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

// npPsSprings.slang
//
// P9.3a -- the first two kernels of the PBD CLOTH spring chain:
//   ps_initializeSpringsLaunch  (particlesystem.cu:4111-4135) zeroes the XPBD
//     lagrange multiplier of every spring, once per frame before the solve;
//   ps_updateRemapVertsLaunch   (particlesystem.cu:4043-4109) GATHERS each
//     spring's two endpoints out of the sorted particle arrays into the cloth
//     buffer's per-spring-endpoint scratch (mRemapPositions / mRemapVelocities),
//     which is the layout ps_solveSpringsLaunch then solves in place.
//
// A NEW GRID SHAPE for this port.  Every particle kernel so far has used a 2D
// grid with the system on gridDim.y.  The cloth chain is 3D:
//     blockIdx.z -> activeParticleSystems[.]   (the particle system)
//     blockIdx.y -> the CLOTH BUFFER within that system
//     blockIdx.x -> the work slice
// with block (UPDATEBOUND = 1024, 1, 1).  A port that keeps the 2D convention
// reads the wrong system and silently processes cloth buffer 0 only.
//
// LAUNCHES (PxgPBDParticleSystemCore.cpp:912-948 and :952-990), both {8,8}:
//   initializeSprings: grid (ceil(maxSprings     / 1024), maxClothBuffers, nbActive)
//   updateRemapVerts:  grid (ceil(maxSprings * 2 / 1024), maxClothBuffers, nbActive)
// Both are gated host-side by (getMaxSpringsPerBuffer() > 0), so a scene with no
// cloth buffer never dispatches them at all.
//
// Staging: ps_updateRemapVertsLaunch blockCopies the descriptor into __shared__
// and then reads SOME fields from the copy and others (mNumClothBuffers,
// mClothSimBuffers, mParticleBufferRunsum) straight from global.  Nothing writes
// the descriptor, so with the staging dropped both spellings are the same read
// and the mixture becomes moot.  ps_initializeSpringsLaunch has no staging at all.
//
// Traps, each with a mutant:
//  K1. Two threads per spring: workIndex = gti / 2 selects the SPRING and
//      index = gti & 1 selects which endpoint, so the guard is against
//      nbSprings * 2 and not nbSprings.
//  K2. The gathered endpoint lands at `workIndex + nbSprings * index` -- COLUMN
//      major with a nbSprings stride, the same convention mRemapOutput and the
//      solve both assume.  Row-major (workIndex * 2 + index) is wrong.
//  K3. The spring's indices are LOCAL to its particle buffer: the lookup is
//      `spring.ind0 + offset` with offset = mParticleBufferRunsum[
//      clothBuffer.mParticleBufferIndex], and only then through
//      mUnsortedToSortedMapping.  Two indirections, both needed.
//  K4. mRemapVelocities is filled from mSortedVelocities.  The CUDA has a
//      commented-out mSortedDeltaP alternative directly above it (:4103); the
//      live line is the velocity one.
//  K5. initializeSprings guards with `>= nbSprings -> return` while
//      updateRemapVerts guards with `< totalSprings -> do work`.  Same effect,
//      and both are transcribed as written rather than unified.
//  K6. The cloth-buffer guard is `bufferIndex < mNumClothBuffers`, per system --
//      the grid is sized by the ACROSS-system maximum, so this is the only thing
//      keeping a system with fewer cloth buffers inside its own array.
//
// The module is in PXG_NOCONTRACT_SHADERS: it moves float4s without arithmetic,
// but ps_solveSpringsLaunch will join it and that is dense float work.

import pxgSlangParticleDesc;

// PxgParticleSystemKernelBlockDim::UPDATEBOUND (PxgParticleSystemCoreKernelIndices.h:39)
static const uint PS_UPDATEBOUND_BLOCK = 1024u;

// sizeof(PxgParticleClothSimBuffer) / sizeof(PxParticleSpring), generated.
static const uint PXG_CLOTH_SIM_BUFFER_STRIDE = 112u;
static const uint PXG_PARTICLE_SPRING_STRIDE = 24u;

// argSizes for both: { particleSystems*, activeParticleSystems* } -> {8,8}.
struct PsSpringsParams
{
    uint64_t particleSystems;         // @0
    uint64_t activeParticleSystems;   // @8 -> 16
}
[[vk::push_constant]]
ConstantBuffer<PsSpringsParams> gSp;

// &mClothSimBuffers[bufferIndex]
uint64_t psClothBufferAddr(Ptr<PxgParticleSystemGpu> systems, uint id, uint bufferIndex)
{
    return systems[id].mClothSimBuffers + uint64_t(bufferIndex) * uint64_t(PXG_CLOTH_SIM_BUFFER_STRIDE);
}

[shader("compute")]
[numthreads(1024, 1, 1)]
void ps_initializeSpringsLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gSp.particleSystems);
    const uint particleId = (Ptr<uint>(gSp.activeParticleSystems))[groupId.z];   // gridDim.Z

    const uint bufferIndex = groupId.y;                                          // gridDim.Y

    if (bufferIndex < systems[particleId].mNumClothBuffers)                       // K6
    {
        Ptr<PxgParticleClothSimBufferGpu> cloth =
            Ptr<PxgParticleClothSimBufferGpu>(psClothBufferAddr(systems, particleId, bufferIndex));

        const uint groupThreadIdx = groupThreadId.x + groupId.x * PS_UPDATEBOUND_BLOCK;

        const uint nbSprings = cloth[0].mNumSprings;

        if (groupThreadIdx >= nbSprings)                                          // K5
            return;

        (Ptr<float>(cloth[0].mSpringLambda))[groupThreadIdx] = 0.0f;
    }
}

[shader("compute")]
[numthreads(1024, 1, 1)]
void ps_updateRemapVertsLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gSp.particleSystems);
    const uint particleId = (Ptr<uint>(gSp.activeParticleSystems))[groupId.z];

    const uint bufferIndex = groupId.y;

    if (bufferIndex < systems[particleId].mNumClothBuffers)                       // K6
    {
        Ptr<PxgParticleClothSimBufferGpu> cloth =
            Ptr<PxgParticleClothSimBufferGpu>(psClothBufferAddr(systems, particleId, bufferIndex));

        const uint groupThreadIdx = groupThreadId.x + groupId.x * PS_UPDATEBOUND_BLOCK;

        const uint nbSprings = cloth[0].mNumSprings;
        const uint totalSprings = nbSprings * 2u;                                 // K1

        if (groupThreadIdx < totalSprings)                                        // K5
        {
            const uint particleBufferIndex = cloth[0].mParticleBufferIndex;
            const uint offset =
                (Ptr<uint>(systems[particleId].mParticleBufferRunsum))[particleBufferIndex];

            Ptr<float4> positions = Ptr<float4>(systems[particleId].mSortedPositions_InvMass);
            Ptr<float4> velocities = Ptr<float4>(systems[particleId].mSortedVelocities);

            Ptr<float4> remapPoses = Ptr<float4>(cloth[0].mRemapPositions);
            Ptr<float4> remapVels = Ptr<float4>(cloth[0].mRemapVelocities);
            Ptr<uint> reverseLookup = Ptr<uint>(systems[particleId].mUnsortedToSortedMapping);

            const uint workIndex = groupThreadIdx / 2u;                           // K1
            const uint index = groupThreadIdx & 1u;

            Ptr<PxParticleSpringGpu> springs = Ptr<PxParticleSpringGpu>(cloth[0].mOrderedSprings);
            const PxParticleSpringGpu spring = springs[workIndex];

            // K3: the spring's indices are LOCAL to its particle buffer.
            const uint thisInd = (index == 0u) ? (spring.ind0 + offset) : (spring.ind1 + offset);
            const uint sortedInd = reverseLookup[thisInd];

            // K2: column major with a nbSprings stride.
            const uint dst = workIndex + nbSprings * index;
            remapPoses[dst] = positions[sortedInd];
            remapVels[dst] = velocities[sortedInd];                               // K4
        }
    }
}
