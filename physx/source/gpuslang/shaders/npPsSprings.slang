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

// ============================================================================
// P9.3c -- ps_averageVertsLaunch (particlesystem.cu:4312-4435)
// ============================================================================
// The GATHER that closes the cloth triple.  ps_updateRemapVertsLaunch scattered
// every spring endpoint into mRemapPositions; the per-partition
// ps_solveSpringsLaunch loop moved those duplicates; this kernel folds each
// particle's scattered copies back into ONE per-particle correction and ADDS it
// into mAccumDeltaP.  One thread per BUFFER-LOCAL particle of one cloth buffer
// of one active system.
//
// LAUNCH (PxgPBDParticleSystemCore.cpp:1031-1052): grid
// (ceil(mMaxParticlesPerBuffer / UPDATEBOUND), mMaxClothBuffersPerSystem,
// nbActive), block (UPDATEBOUND = 1024, 1, 1) -- a THIRD block dim in this
// module, after the gather pair's 1024 and the solve's 512.  Args
// (particleSystems*, activeParticleSystems*, PxReal invDt) -> {8,8,4} = 20
// bytes, rounded to the 8-alignment -> 24.  Dispatched ONCE per solver
// iteration, not once per partition, and gated host-side by
// getMaxSpringsPerBuffer() > 0 (:958).
//
// ZERO new ABI: every field is already mapped by P9.3a/P9.3b.  This is the first
// consumer of PxParticleClothGpu, which P9.3a defined and nothing used.
//
// Traps, each with a mutant:
//  A1. THREE index spaces in eight lines.  `index = groupThreadIdx + offset` is
//      GLOBAL; `sortedInd = reverseLookup[index]` is the sorted slot; but the
//      copy run is indexed by the BUFFER-LOCAL `groupThreadIdx`, and its
//      zero-sentinel tests `groupThreadIdx == 0`, not `index == 0`.  The runsum
//      array is exactly mNumActiveParticles long, so a global subscript is out
//      of range by construction.
//  A2. The store is a read-modify-write ACCUMULATE, and ALL FOUR lanes take
//      part.  cutil_math.h's float4 operator-, operator+= and operator*(float4,
//      float) each touch .w (:451-470), and the CUDA -- unlike its softbody twin
//      (softBodyGM.cu:1574) -- never zeroes diff.w.  A float3 port is
//      ACCIDENTALLY correct in the live pipeline, because ps_solveSpringsLaunch
//      writes pos_im.w into every copy so diff.w cancels to exactly 0.
//  A3. The binarySearch KEY is global while the searched array is BUFFER-LOCAL:
//      `sortedClothStartIndices[i] = cloths[i].startVertexIndex` is written
//      host-side with NO offset (NpParticleBuffer.cpp:587), yet the key is
//      `index`, and the very next line re-adds the offset to the same record's
//      startVertexIndex.  Transcribed as written -- see the RUNTIME_STATUS entry
//      for why this is an upstream defect the port must reproduce, not fix.
//  A4. binarySearch (reduction.cuh:726-752) is `largest i with data[i] <= value`
//      CLAMPED TO 0 -- an upper-bound-minus-one, not a lower_bound, and it has
//      no not-found sentinel: it returns 0 both for a genuine hit at 0 and for
//      data[0] > value.  The range test below is what disambiguates them.
//  A5. `springOffset = nbSprings * 2` -- the copy region starts AFTER both
//      endpoint planes.  The name invites reading it as "offset of the springs".
//  A6. Everything positional is addressed by `sortedInd`, and the line above the
//      live one names the OPPOSITE mapping in a comment.  mUnsortedToSortedMapping
//      (@48) and mSortedToUnsortedMapping (@56) are adjacent descriptor slots.
//  A7. Block dim is 1024 (the stale comment above the kernel says 256), the grid
//      is 3D with the system on Z, and `invDt` is a live push-constant field the
//      body must NEVER read -- the softbody twin does read its own.
//  A8. mAccumulatedCopiesPerParticles is an INCLUSIVE runsum (NpParticleBuffer.cpp
//      :494-495): entry i is the END of particle i's run.  The header's own
//      `//numSprings` annotation on the field is wrong; the array is
//      mNumActiveParticles long.
//  A9. The reduction PEELS the first element and the range is half-open:
//      `diff = poses[startInd] - pos` then `for j = startInd+1; j < endInd`.
//      Because the peel is a real load and not a literal zero, the P8.84
//      signed-zero fold does NOT apply here.
// A10. The cloth range test is STRICT `>`, offset-lifted, and has NO lower
//      bound -- it leans entirely on binarySearch's `<=` contract.  "Completing"
//      the interval changes behaviour exactly where the clamp-to-0 fired.
// A11. `if (numCopies)` is the divide-by-zero gate AND the empty-run gate: a
//      particle of a non-cloth type in the same buffer never went through the
//      partition table and has a flat run.  `deltaP` is loaded OUTSIDE it.
// A12. Two-level buffer indirection: only the CLOTH record is indexed by
//      blockIdx.y; the particle buffer and the runsum both go through
//      clothBuffer.mParticleBufferIndex, so nbParticles comes from the PARTICLE
//      buffer.  Several cloth buffers may share one particle buffer.
// A13. The `&&` short-circuit is the only thing that stops cloths[0] being
//      dereferenced against a ZERO-LENGTH allocation when mNumCloths == 0 (which
//      is also the only case where `clothIndex < mNumCloths` can be false, since
//      binarySearch can never return >= numElements).  Reproduced here as a real
//      BRANCH rather than a `&&`: per the P8.96c rule llvmpipe turns a guarded
//      uniform BDA load into an unconditional load plus a select, and only
//      branch-guarded loads were observed not to speculate.
// A14. PxParticleCloth's stride is 28, NOT 32 -- seven 4-byte members, align 4,
//      no trailing pad.  Loaded here by explicit byte address rather than
//      through Ptr<PxParticleClothGpu>, so the stride is spelled once and no
//      assumption is made about how Slang lays out a 4-aligned struct.
// A15. mNumCloths is re-read from the record twice and the array is exactly that
//      long.

// argSizes (PxgPBDParticleSystemCore.cpp:1038-1043):
// { particleSystems*, activeParticleSystems*, PxReal invDt } -> {8,8,4} = 20,
// rounded to the 8-alignment -> 24.
struct PsAverageVertsParams
{
    uint64_t particleSystems;         // @0
    uint64_t activeParticleSystems;   // @8
    float invDt;                      // @16  DEAD: the body must never read it
    uint _pad20;                      // @20 -> 24
}
[[vk::push_constant]]
ConstantBuffer<PsAverageVertsParams> gAv;

// sizeof(PxgParticleSimBuffer), generated and pinned by the fixture.
static const uint PXG_PARTICLE_SIM_BUFFER_STRIDE = 88u;
// A14: sizeof(PxParticleCloth) -- 28, not 32.
static const uint PXG_PARTICLE_CLOTH_STRIDE = 28u;

// A14: the three fields this kernel reads, by explicit byte address.
uint64_t avClothRecAddr(uint64_t base, uint i)
{
    return base + uint64_t(i) * uint64_t(PXG_PARTICLE_CLOTH_STRIDE);
}
uint avClothStartVertexIndex(uint64_t base, uint i) { return (Ptr<uint>(avClothRecAddr(base, i)))[0]; }
uint avClothNumVertices(uint64_t base, uint i)      { return (Ptr<uint>(avClothRecAddr(base, i)))[1]; }
float avClothBlendScale(uint64_t base, uint i)      { return (Ptr<float>(avClothRecAddr(base, i)))[2]; }

// A4: binarySearch, gpucommon/src/CUDA/reduction.cuh:726-752.  "The largest
// index into the (sorted!) data array s.t. data[index] <= value; if there is no
// such index (i.e. data[0] > value) returns 0."  Note the dead
// `//while((right - left) > 1)` directly above the live loop in the original --
// it is NOT equivalent and is not what runs.
uint avBinarySearch(Ptr<uint> data, uint numElements, uint value)
{
    uint left = 0u;
    uint right = numElements;
    while (left < right)
    {
        const uint pos = (left + right) / 2u;
        const uint element = data[pos];
        if (element <= value)
            left = pos + 1u;
        else
            right = pos;
    }
    return (left != 0u) ? (left - 1u) : 0u;
}

[shader("compute")]
[numthreads(1024, 1, 1)]
void ps_averageVertsLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gAv.particleSystems);
    const uint particleId = (Ptr<uint>(gAv.activeParticleSystems))[groupId.z];   // A7

    const uint bufferIndex = groupId.y;

    if (bufferIndex < systems[particleId].mNumClothBuffers)
    {
        Ptr<PxgParticleClothSimBufferGpu> cloth =
            Ptr<PxgParticleClothSimBufferGpu>(psClothBufferAddr(systems, particleId, bufferIndex));

        // A12: the particle buffer is NOT bufferIndex.
        const uint particleBufferIndex = cloth[0].mParticleBufferIndex;
        Ptr<PxgParticleSimBufferGpu> particleBuffers =
            Ptr<PxgParticleSimBufferGpu>(systems[particleId].mParticleSimBuffers);
        const uint offset =
            (Ptr<uint>(systems[particleId].mParticleBufferRunsum))[particleBufferIndex];

        const uint groupThreadIdx = groupThreadId.x + groupId.x * PS_UPDATEBOUND_BLOCK;

        const uint nbParticles = particleBuffers[particleBufferIndex].mNumActiveParticles;
        const uint nbSprings = cloth[0].mNumSprings;

        // A6: mUnsortedToSortedMapping (@48), not its neighbour at @56.
        Ptr<uint> reverseLookup = Ptr<uint>(systems[particleId].mUnsortedToSortedMapping);

        if (groupThreadIdx < nbParticles && nbSprings > 0u)
        {
            Ptr<float4> positions = Ptr<float4>(systems[particleId].mSortedPositions_InvMass);
            Ptr<uint> startIndices = Ptr<uint>(cloth[0].mSortedClothStartIndices);
            Ptr<uint> accumulatedCopies = Ptr<uint>(cloth[0].mAccumulatedCopiesPerParticles);
            const uint64_t cloths = cloth[0].mCloths;

            // A5: the copy region starts after BOTH endpoint planes.
            const uint springOffset = nbSprings * 2u;
            Ptr<float4> accumulatedPoses =
                Ptr<float4>(cloth[0].mRemapPositions + uint64_t(springOffset) * 16ull);

            Ptr<float4> accumDeltaP = Ptr<float4>(systems[particleId].mAccumDeltaP);

            const uint index = groupThreadIdx + offset;
            const uint sortedInd = reverseLookup[index];

            const float4 pos_invMass = positions[sortedInd];

            if (pos_invMass.w > 0.0f)
            {
                // A2: loaded BEFORE the numCopies test, and all four lanes.
                const float4 deltaP = accumDeltaP[sortedInd];

                // A1 + A8: buffer-local subscript, buffer-local sentinel,
                // INCLUSIVE runsum.
                const uint startInd = (groupThreadIdx == 0u) ? 0u
                                                             : accumulatedCopies[groupThreadIdx - 1u];
                const uint endInd = accumulatedCopies[groupThreadIdx];

                const uint numCopies = endInd - startInd;

                if (numCopies != 0u)                                        // A11
                {
                    // A9: peel the first, then a half-open ascending loop.
                    float4 diff = accumulatedPoses[startInd] - pos_invMass;

                    for (uint j = startInd + 1u; j < endInd; ++j)
                        diff += accumulatedPoses[j] - pos_invMass;

                    const uint numCloths = cloth[0].mNumCloths;             // A15

                    // A3: the GLOBAL key against the BUFFER-LOCAL array.
                    const uint clothIndex = avBinarySearch(startIndices, numCloths, index);

                    // A13: a real branch, not a `&&` -- see the header.
                    bool useBlend = false;
                    if (clothIndex < numCloths)
                    {
                        const uint startVertexIndex = avClothStartVertexIndex(cloths, clothIndex);
                        const uint numVertices = avClothNumVertices(cloths, clothIndex);
                        // A10: strict >, offset-lifted, no lower bound.
                        useBlend = ((startVertexIndex + offset) + numVertices) > index;
                    }

                    float scale;
                    if (useBlend)
                        scale = avClothBlendScale(cloths, clothIndex);
                    else
                        scale = 1.0f / float(numCopies);

                    const float4 averageDeltaP = diff * scale;

                    accumDeltaP[sortedInd] = deltaP + averageDeltaP;        // A2
                }
            }
        }
    }
}
