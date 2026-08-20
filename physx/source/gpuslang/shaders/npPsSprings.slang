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

// ============================================================================
// P9.3b -- ps_solveSpringsLaunch (particlesystem.cu:4137-4309)
// ============================================================================
// The XPBD spring solve, run once per PARTITION: the host loops partitionId over
// [0, maxPartitions) and dispatches this kernel each time
// (PxgPBDParticleSystemCore.cpp:992-1030), which is how a Jacobi kernel gets
// Gauss-Seidel convergence without write conflicts -- every spring inside one
// partition touches a disjoint pair of endpoints.
//
// LAUNCH.  args (particleSystems*, activeParticleSystems*, PxReal invDt,
// PxU32 partitionId, bool isTGS) -> {8,8,4,4,1}; grid
// (ceil(maxSpringsPerPartition / ACCUMULATE_DELTA), maxClothBuffers, nbActive),
// block (ACCUMULATE_DELTA = 512, 1, 1) -- a DIFFERENT block dim from the gather
// pair above, which uses UPDATEBOUND = 1024.
//
// Traps, each with a mutant:
//  S1. A NEGATIVE stiffness marks a ONE-SIDED constraint (a tether: compression
//      is free, only extension pulls).  It is spelled as `e = PxMin(e, 0);
//      k *= -1` -- the sign is consumed, not merely tested, so a port that only
//      takes the absolute value loses the tether behaviour entirely.
//  S2. `dt` is RECOMPUTED as 1.f / invDt.  The host has dt and passes its
//      reciprocal; the kernel inverts it back.  A reciprocal of a reciprocal is
//      not the identity on floats, so passing dt straight through would diverge.
//  S3. a = dt * b + d with b = dt * k, i.e. dt^2 * stiffness + dt * damping --
//      the compliance term is QUADRATIC in dt and the damping term is linear.
//  S4. velBias carries a leading MINUS: -x * a * dcdt.
//  S5. The PGS path (isTGS false) READS and WRITES springLambda with an `iMul`
//      warm start; the TGS path does NEITHER.  On TGS the lambda array must come
//      back exactly as ps_initializeSpringsLaunch left it, which is the only way
//      to tell the two apart from the outside.
//  S6. The magnitude limiter is an ASSIGNMENT, not a symmetric clamp: when
//      |dlambda| exceeds |errorBias| it becomes errorBias -- carrying errorBias's
//      SIGN, which can flip dlambda.  `PxClamp(dlambda, -|errorBias|, |errorBias|)`
//      is a different function.
//  S7. The two endpoints move in OPPOSITE directions (newPos0 += , newPos1 -=)
//      and each is weighted by its OWN inverse mass.
//  S8. The velocity correction is the position correction times invDt.
//  S9. The stores go to writeIndices[...] = mRemapOutput, NOT to workIndex.  That
//      scatter is what feeds the next partition's gather, and it happens even
//      when every early-out fired -- the unchanged values are written back.
// S10. remapPoses keeps the endpoint's inverse mass in .w; remapVels writes a
//      literal 0 there.
// S11. `xij / l` is PxVec3::operator/, a reciprocal multiply.
// S12. The degenerate guard is dSq > 1e-8f -- a different constant from
//      ps_solveDensityLaunch's 1e-10f.
// S13. The partition's range is [partitions[partitionId-1] or 0, partitions[
//      partitionId]), an exclusive prefix sum, and workIndex is startInd + gti.

// PxgParticleSystemKernelBlockDim::ACCUMULATE_DELTA (…KernelIndices.h:45)
static const uint PS_ACCUMULATE_DELTA_BLOCK = 512u;

float ssMin(float a, float b) { return a < b ? a : b; }
float ssAbs(float x) { return abs(x); }

// argSizes (PxgPBDParticleSystemCore.cpp:1003-1010):
// { particleSystems*, activeParticleSystems*, PxReal invDt, PxU32 partitionId,
//   bool isTGS } -> {8,8,4,4,1} = 25, rounded to the 8-alignment -> 32.
struct PsSolveSpringsParams
{
    uint64_t particleSystems;         // @0
    uint64_t activeParticleSystems;   // @8
    float invDt;                      // @16
    uint partitionId;                 // @20
    uint isTGSWord;                   // @24 bool byte 0 (bridge zero-pads 25..31)
    uint _pad28;                      // @28 -> 32
}
[[vk::push_constant]]
ConstantBuffer<PsSolveSpringsParams> gSs;

[shader("compute")]
[numthreads(512, 1, 1)]
void ps_solveSpringsLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gSs.particleSystems);
    const uint particleId = (Ptr<uint>(gSs.activeParticleSystems))[groupId.z];

    const uint bufferIndex = groupId.y;
    const uint groupThreadIdx = groupThreadId.x + groupId.x * PS_ACCUMULATE_DELTA_BLOCK;

    if (bufferIndex < systems[particleId].mNumClothBuffers)
    {
        Ptr<PxgParticleClothSimBufferGpu> cloth =
            Ptr<PxgParticleClothSimBufferGpu>(psClothBufferAddr(systems, particleId, bufferIndex));

        const uint nbSprings = cloth[0].mNumSprings;
        const uint nbPartitions = cloth[0].mNumPartitions;
        const uint partitionId = gSs.partitionId;

        if (nbSprings > 0u && partitionId < nbPartitions)
        {
            Ptr<float4> remapPoses = Ptr<float4>(cloth[0].mRemapPositions);
            Ptr<float4> remapVels = Ptr<float4>(cloth[0].mRemapVelocities);
            Ptr<float> springLambda = Ptr<float>(cloth[0].mSpringLambda);
            Ptr<uint> partitions = Ptr<uint>(cloth[0].mAccumulatedSpringsPerPartitions);
            Ptr<uint> writeIndices = Ptr<uint>(cloth[0].mRemapOutput);
            Ptr<PxParticleSpringGpu> springs = Ptr<PxParticleSpringGpu>(cloth[0].mOrderedSprings);

            // S13: an exclusive prefix sum over the partitions.
            const uint startInd = (partitionId > 0u) ? partitions[partitionId - 1u] : 0u;
            const uint endInd = partitions[partitionId];

            const uint workIndex = startInd + groupThreadIdx;

            if (workIndex < endInd)
            {
                const PxParticleSpringGpu spring = springs[workIndex];

                const float4 pos_im0 = remapPoses[workIndex];
                const float4 pos_im1 = remapPoses[workIndex + nbSprings];

                const float3 xi = float3(pos_im0.x, pos_im0.y, pos_im0.z);
                const float3 xj = float3(pos_im1.x, pos_im1.y, pos_im1.z);

                const float4 v0 = remapVels[workIndex];
                const float4 v1 = remapVels[workIndex + nbSprings];
                const float3 veli = float3(v0.x, v0.y, v0.z);
                const float3 velj = float3(v1.x, v1.y, v1.z);

                const float3 xij = xi - xj;
                const float wsum = pos_im0.w + pos_im1.w;

                float3 newPos0 = xi;
                float3 newPos1 = xj;
                float3 newVel0 = veli;
                float3 newVel1 = velj;

                // S9: the scatter targets, read BEFORE any early-out.
                const uint writeIndex0 = writeIndices[workIndex];
                const uint writeIndex1 = writeIndices[workIndex + nbSprings];

                if (wsum > 0.0f)
                {
                    const float dSq = xij.x * xij.x + xij.y * xij.y + xij.z * xij.z;

                    if (dSq > 1e-8f)                                   // S12
                    {
                        const float l = sqrt(dSq);

                        const float len = spring.length;
                        float e = len - l;
                        float k = spring.stiffness;

                        // S1: a negative stiffness is a ONE-SIDED tether.
                        if (k < 0.0f)
                        {
                            e = ssMin(e, 0.0f);
                            k *= -1.0f;
                        }

                        const float3 cgrad = xij * (1.0f / l);         // S11
                        const float3 dvel = veli - velj;
                        const float dcdt = cgrad.x * dvel.x + cgrad.y * dvel.y + cgrad.z * dvel.z;

                        const float dt = 1.0f / gSs.invDt;             // S2
                        const float b = dt * k;
                        const float d = dt * spring.damping;
                        const float a = dt * b + d;                    // S3

                        const float x = 1.0f / (1.0f + a * wsum);
                        const float bias = x * b * e;
                        const float velBias = -x * a * dcdt;           // S4

                        float dlambda = 0.0f;

                        const bool isTGS = (gSs.isTGSWord & 0xFFu) != 0u;
                        if (!isTGS)                                    // S5
                        {
                            const float iMul = 1.0f - x;
                            const float lambda = springLambda[workIndex];
                            dlambda = ((bias + velBias) * dt + iMul * lambda) - lambda;
                            springLambda[workIndex] = lambda + dlambda;
                        }
                        else
                        {
                            dlambda = (bias + velBias) * dt;
                        }

                        // S6: an ASSIGNMENT, so errorBias's SIGN is adopted.
                        const float errorBias = e / wsum;
                        if (ssAbs(dlambda) > ssAbs(errorBias))
                            dlambda = errorBias;

                        const float3 deltaV = dlambda * cgrad;
                        const float3 deltaV0 = deltaV * pos_im0.w;     // S7
                        const float3 deltaV1 = deltaV * pos_im1.w;

                        newPos0 += deltaV0;
                        newPos1 -= deltaV1;

                        newVel0 += deltaV0 * gSs.invDt;                // S8
                        newVel1 -= deltaV1 * gSs.invDt;
                    }
                }

                // S9/S10: unconditional scatter; the position keeps its inverse
                // mass in .w, the velocity gets a literal zero.
                remapPoses[writeIndex0] = float4(newPos0.x, newPos0.y, newPos0.z, pos_im0.w);
                remapPoses[writeIndex1] = float4(newPos1.x, newPos1.y, newPos1.z, pos_im1.w);
                remapVels[writeIndex0] = float4(newVel0.x, newVel0.y, newVel0.z, 0.0f);
                remapVels[writeIndex1] = float4(newVel1.x, newVel1.y, newVel1.z, 0.0f);
            }
        }
    }
}
