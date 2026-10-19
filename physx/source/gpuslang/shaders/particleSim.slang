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

// particleSim.slang
//
// P6.0 — the first PBD particle orchestration kernels on the Vulkan engine ABI
// (BDA), ported line-for-line from
// gpusimulationcontroller/src/CUDA/particlesystem.cu:
//   - ps_updateUnsortedArrayLaunch (particlesystem.cu:218): the INGEST — copies
//     the user PxParticleBuffer (positions with invMass, velocities, phases) into
//     the internal unsorted arrays.  REQUIRED for any observable: without it the
//     internal mUnsortedPositions_InvMass stays zero (invMass==0 => infinite
//     mass => preIntegrate's tPos.w!=0 guard skips every particle => no fall).
//     The launch site copyUserBufferToUnsortedArray issues PS_UPDATE_UNSORTED_ARRAY;
//     this is the mirror image of ps_updateUserBufferLaunch.
//   - ps_preIntegrateLaunch (particlesystem.cu:304): the gravity predict on the
//     UNSORTED arrays.  This is the free-fall driver: pos += vel*dt*fwdScale,
//     vel += gravity*dt (persisted), each frame.  Self-contained on unsorted
//     data — no dependency on any un-ported grid/sort/collision/solve kernel.
//   - ps_integrateLaunch (particlesystem.cu:3087): the post-solve writeback from
//     the SORTED/solved state into the unsorted arrays.  On the P6.0 contact-free
//     free-fall scene the un-ported reorder never fills the sorted buffers, which
//     stay zero-initialised (fresh vkAllocateMemory); the mSortedDeltaP==0 =>
//     locked => pos4=sortedOriginPos==0 => pos4.w==0 guard then makes the kernel
//     an inert no-op that neither corrupts nor faults, leaving the preIntegrate
//     fall intact.  A behaviour-preserving llvmpipe-safety bound guard
//     (sortedIndex < numParticles) hardens the one scatter write against the
//     un-ported-reorder garbage case (a valid reorder always yields
//     sortedIndex < numParticles, so it never changes correct behaviour).
//   - ps_updateUserBufferLaunch (particlesystem.cu:269): copy the solved unsorted
//     particle state back into the user PxParticleBuffer, so the host DtoH read
//     observes the fall.
//
// Differences from CUDA (recipe policy):
//   - the descriptor arrives as a device pointer to an array of PxgParticleSystem
//     read through Ptr<> (BDA), and its nested device-pointer members are walked
//     the same way; the CUDA ps_integrateLaunch block-copy of the whole
//     descriptor into shared memory is replaced by direct field loads
//     (semantically identical);
//   - the PxU16 phase->material lookup and mData.mLockFlags reads fold into 32-bit
//     loads (no 16-bit-storage capability required on llvmpipe);
//   - the ps_integrateLaunch material fetch is dead in the CUDA original (mat is
//     read but only data.mMaxVelocity is used), so it is dropped;
//   - P9.97: value-identical source-index clamps and a rest-pointer select in the two
//     buffer-copy kernels (UA:/UB: tags) -- hardening for back ends that hoist a
//     guarded load; llvmpipe issues none of the unclamped reads (measured).

import pxgSlangParticleDesc;

static const uint PS_BLOCK_DIM = 1024u; // PxgParticleSystemKernelBlockDim::UPDATEBOUND
static const uint PS_MAX_STATIC_CONTACTS_PER_PARTICLE = 12u; // PxgParticleContactInfo::MaxStaticContactsPerParticle

// PxParticleBufferFlag (PxParticleSystemFlag.h:51-54)
static const uint PS_BUFFER_FLAG_UPDATE_POSITION = 1u;
static const uint PS_BUFFER_FLAG_UPDATE_VELOCITY = 2u;
static const uint PS_BUFFER_FLAG_UPDATE_PHASE = 4u;
static const uint PS_BUFFER_FLAG_UPDATE_RESTPOSITION = 8u;

// PxsParticleMaterialData damping (byte 4) / gravityScale (byte 12), read with
// the runtime PxsPBDMaterialData stride.
float loadMaterialDamping(uint64_t materialsBase, uint mi, uint stride)
{
    let elem = materialsBase + uint64_t(mi) * uint64_t(stride);
    return (Ptr<float>(elem))[PXG_PARTICLE_MAT_DAMPING_OFFSET_FLOATS];
}
float loadMaterialGravityScale(uint64_t materialsBase, uint mi, uint stride)
{
    let elem = materialsBase + uint64_t(mi) * uint64_t(stride);
    return (Ptr<float>(elem))[PXG_PARTICLE_MAT_GRAVITYSCALE_OFFSET_FLOATS];
}

// phaseToMat[group] where phaseToMat is a PxU16 array; a 32-bit load + shift/mask
// avoids requiring 16-bit storage on the software Vulkan target.
uint loadPhaseToMaterial(uint64_t handleBase, uint group)
{
    let word = (Ptr<uint>(handleBase))[group >> 1];
    return (word >> ((group & 1u) * 16u)) & 0xffffu;
}

// ---------------------------------------------------------------------------
// ps_preIntegrateLaunch — gravity predict (particlesystem.cu:304-399)
// ---------------------------------------------------------------------------
// argSizes (CUDA pack): { particleSystems*, activeParticleSystems*, PxVec3
// gravity(12), PxReal dt(4), bool isTGS(1), bool extForces(1) } -> 40-byte blob.
struct PsPreIntegrateParams
{
    uint64_t particleSystems;       // @0
    uint64_t activeParticleSystems; // @8
    float gravityX;                 // @16
    float gravityY;                 // @20
    float gravityZ;                 // @24
    float dt;                       // @28
    uint  flagsWord;                // @32 byte0=isTGS, byte1=externalForcesEveryTgsIteration
    uint  _pad36;                   // @36 -> 40
}
[[vk::push_constant]]
ConstantBuffer<PsPreIntegrateParams> gPre;

[shader("compute")]
[numthreads(1024, 1, 1)]
void ps_preIntegrateLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gPre.particleSystems);
    const uint particleId = (Ptr<uint>(gPre.activeParticleSystems))[groupId.y];

    const uint numParticles = systems[particleId].mCommonData_NumParticles;
    const uint globalThreadIndex = groupThreadId.x + groupId.x * PS_BLOCK_DIM;

    if (globalThreadIndex >= numParticles)
        return;

    Ptr<float4> positions = Ptr<float4>(systems[particleId].mUnsortedPositions_InvMass);
    Ptr<float4> vels = Ptr<float4>(systems[particleId].mUnsortedVelocities);
    Ptr<float4> oriPositions = Ptr<float4>(systems[particleId].mOriginPos_InvMass);
    Ptr<float4> accumDeltaP = Ptr<float4>(systems[particleId].mAccumDeltaP);
    Ptr<uint> phases = Ptr<uint>(systems[particleId].mUnsortedPhaseArray);
    Ptr<uint> contactCounts = Ptr<uint>(systems[particleId].mOneWayContactCount);
    Ptr<float2> oneWayForces = Ptr<float2>(systems[particleId].mOneWayForces);

    const uint64_t phaseToMat = systems[particleId].mPhaseGroupToMaterialHandle;
    const uint64_t materials = systems[particleId].mParticleMaterials;
    const uint materialStride = systems[particleId].mParticleMaterialStride;

    const uint activeIndex = globalThreadIndex;
    const uint phase = phases[activeIndex];
    const uint group = phase & PXG_PARTICLE_PHASE_GROUP_MASK;
    const uint mi = loadPhaseToMaterial(phaseToMat, group);

    const float matDamping = loadMaterialDamping(materials, mi, materialStride);
    const float matGravityScale = loadMaterialGravityScale(materials, mi, materialStride);

    const float damping = 1.0f - min(1.0f, matDamping * gPre.dt);
    const float3 gravity = float3(gPre.gravityX, gPre.gravityY, gPre.gravityZ);
    const float3 tGravity = gravity * matGravityScale;

    const bool isTGS = (gPre.flagsWord & 0xffu) != 0u;
    const bool externalForcesEveryTgsIterationEnabled = ((gPre.flagsWord >> 8) & 0xffu) != 0u;

    const float4 tPos = positions[activeIndex];
    float3 vel = float3(0.0f, 0.0f, 0.0f);

    oriPositions[activeIndex] = tPos;

    if (tPos.w != 0.0f)
    {
        const float3 tVel = vels[activeIndex].xyz;
        const float3 oldPos = float3(tPos.x, tPos.y, tPos.z);

        vel = (tVel + tGravity * gPre.dt) * damping;

        // PARTICLE_FORWARD_PROJECTION_STEP_SCALE_TGS = 0.5, _PGS = 1.0
        const float forwardProjectionStepScale = isTGS ? 0.5f : 1.0f;
        const float3 delta = vel * gPre.dt * forwardProjectionStepScale;
        const float3 pos = oldPos + delta;

        positions[activeIndex] = float4(pos.x, pos.y, pos.z, tPos.w);
    }

    if (!externalForcesEveryTgsIterationEnabled)
        vels[activeIndex] = float4(vel.x, vel.y, vel.z, tPos.w);

    accumDeltaP[globalThreadIndex] = float4(0.0f, 0.0f, 0.0f, 0.0f);
    contactCounts[globalThreadIndex] = 0u;

    for (uint i = globalThreadIndex; i < (PS_MAX_STATIC_CONTACTS_PER_PARTICLE * numParticles); i += numParticles)
        oneWayForces[i] = float2(0.0f, 0.0f);
}

// ---------------------------------------------------------------------------
// ps_preDiffuseIntegrateLaunch — the DIFFUSE gravity predict
// (particlesystem.cu:403-448), P9.13b
// ---------------------------------------------------------------------------
// Sits here rather than in a module of its own because it is the diffuse twin of
// ps_preIntegrateLaunch above, and the two are only legible next to each other:
// the differences are the point.
//
// SIX TRAPS, each a mutant.
//
// D1 -- the count is a POINTER DEREFERENCE, not a descriptor field.
//     `*particleSystem.mNumDiffuseParticles` is the LIVE number of diffuse
//     particles, which changes every frame as they spawn and expire.  The
//     descriptor also carries `mCommonData_MaxDiffuseParticles`, which is the
//     ALLOCATION and is what the grid-hash kernel bounds itself by instead.
//     Reading the wrong one here integrates dead slots.
//
// D2 -- there is NO `tPos.w != 0` guard.  The solid kernel skips a particle whose
//     inverse mass is zero (an infinitely heavy, pinned particle); for a diffuse
//     particle the w channel is its remaining LIFETIME, not an inverse mass, so a
//     w of zero means "about to expire", not "immovable".  Copying the solid
//     guard across would freeze exactly the particles that are still alive on
//     their last frame.
//
// D3 -- no damping and no material lookup at all.  Diffuse particles have no
//     phase and no PBD material: `vel = tVel + gravity * dt` with the RAW
//     gravity, not `* matGravityScale`, and no `(1 - min(1, damping*dt))` factor.
//
// D4 -- no TGS forward-projection scale.  The solid kernel steps by
//     `vel * dt * (isTGS ? 0.5 : 1.0)`; this one always steps by `vel * dt`, and
//     it takes no isTGS argument to do otherwise.
//
// D5 -- THE VELOCITY'S w COMES FROM THE POSITION.  `vels[i] = (vel, tPos.w)` --
//     the lifetime is copied into the velocity array's fourth channel, not the
//     velocity's own previous w.  (The solid kernel does the same thing with the
//     inverse mass, so this is the engine's convention rather than a slip; it is
//     still a trap for a port that "tidies" it to `vels[i].w`.)
//
// D6 -- `mDiffuseOneWayContactCount` MAY BE NULL and the reset is guarded on it.
//     The CUDA's comment reads "If contactCounts is null, this means solver is
//     pbd", which is the opposite way round from what the code does -- the PBD
//     path is the one that HAS the array.  Transcribed as written: guard on the
//     pointer, do not reason from the comment.
//
// And the ordering that is load-bearing: `oriPositions[i] = tPos` stores the
// PRE-integration position (with its lifetime in w), and only then is `positions`
// overwritten.  ps_updateBoundFirstPassLaunch reads exactly that pair as the CCD
// sweep's endpoints.
//
// argSizes (launch site PxgParticleSystemCore.cpp:2716-2722): { particleSystems*,
// activeParticleSystems*, PxVec3 gravity(12), PxReal dt(4) } -> 32-byte blob.
// NOTE the two trailing bools the solid kernel has are ABSENT here.
struct PsPreDiffuseIntegrateParams
{
    uint64_t particleSystems;       // @0
    uint64_t activeParticleSystems; // @8
    float gravityX;                 // @16
    float gravityY;                 // @20
    float gravityZ;                 // @24
    float dt;                       // @28 -> 32
}
[[vk::push_constant]]
ConstantBuffer<PsPreDiffuseIntegrateParams> gPreD;

[shader("compute")]
[numthreads(1024, 1, 1)]
void ps_preDiffuseIntegrateLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gPreD.particleSystems);
    const uint particleId = (Ptr<uint>(gPreD.activeParticleSystems))[groupId.y];

    // D1: the LIVE count, dereferenced.
    const uint numParticles = (Ptr<uint>(systems[particleId].mNumDiffuseParticles))[0];

    const uint globalThreadIndex = groupThreadId.x + groupId.x * PS_BLOCK_DIM;

    Ptr<float4> positions = Ptr<float4>(systems[particleId].mDiffusePosition_LifeTime);
    Ptr<float4> vels = Ptr<float4>(systems[particleId].mDiffuseVelocity);
    Ptr<float4> oriPositions = Ptr<float4>(systems[particleId].mDiffuseOriginPos_LifeTime);
    const uint64_t contactCountsAddr = systems[particleId].mDiffuseOneWayContactCount;

    if (globalThreadIndex >= numParticles)
        return;

    const float4 tPos = positions[globalThreadIndex];

    oriPositions[globalThreadIndex] = tPos;

    const float3 tVel = vels[globalThreadIndex].xyz;
    const float3 oldPos = float3(tPos.x, tPos.y, tPos.z);

    // D3: raw gravity, no damping, no material.  D2: unconditional.
    const float3 vel = tVel + float3(gPreD.gravityX, gPreD.gravityY, gPreD.gravityZ) * gPreD.dt;
    // D4: no forward-projection scale.
    const float3 pos = oldPos + vel * gPreD.dt;

    positions[globalThreadIndex] = float4(pos.x, pos.y, pos.z, tPos.w);
    // D5: the LIFETIME, from the position.
    vels[globalThreadIndex] = float4(vel.x, vel.y, vel.z, tPos.w);

    // D6: guarded on the pointer, not on a solver mode.
    if (contactCountsAddr != 0UL)
        (Ptr<uint>(contactCountsAddr))[globalThreadIndex] = 0u;
}

// ---------------------------------------------------------------------------
// ps_integrateLaunch — post-solve vel/pos writeback (particlesystem.cu:3087-3218)
// ---------------------------------------------------------------------------
// argSizes: { particleSystems*, activeParticleSystems*, PxReal dt(4),
// PxReal epsTolerance(4) } -> 24-byte blob.
struct PsIntegrateParams
{
    uint64_t particleSystems;       // @0
    uint64_t activeParticleSystems; // @8
    float dt;                       // @16
    float epsTolerance;             // @20 -> 24
}
[[vk::push_constant]]
ConstantBuffer<PsIntegrateParams> gInt;

[shader("compute")]
[numthreads(1024, 1, 1)]
void ps_integrateLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gInt.particleSystems);
    const uint particleId = (Ptr<uint>(gInt.activeParticleSystems))[groupId.y];

    const uint numParticles = systems[particleId].mCommonData_NumParticles;
    const uint globalThreadIndex = groupThreadId.x + groupId.x * PS_BLOCK_DIM;

    if (globalThreadIndex >= numParticles)
        return;

    Ptr<float4> position = Ptr<float4>(systems[particleId].mUnsortedPositions_InvMass);
    Ptr<float4> newPos = Ptr<float4>(systems[particleId].mSortedPositions_InvMass);
    Ptr<float4> origPos = Ptr<float4>(systems[particleId].mSortedOriginPos_InvMass);
    Ptr<float4> deltaP = Ptr<float4>(systems[particleId].mSortedDeltaP);
    Ptr<float4> originalVels = Ptr<float4>(systems[particleId].mUnsortedVelocities);
    Ptr<float4> newVels = Ptr<float4>(systems[particleId].mSortedVelocities);
    Ptr<float4> deltas = Ptr<float4>(systems[particleId].mDelta);
    Ptr<uint> gridParticleIndex = Ptr<uint>(systems[particleId].mSortedToUnsortedMapping);

    const uint lockFlags = systems[particleId].mData_LockFlagsWord & 0xffffu;
    const float dataMaxVelocity = systems[particleId].mData_MaxVelocity;

    const uint sortedIndex = gridParticleIndex[globalThreadIndex];

    const float3 dta = deltaP[globalThreadIndex].xyz;
    const bool locked = dot(dta, dta) <= gInt.epsTolerance;
    const float4 origPos4 = origPos[globalThreadIndex];

    float4 pos4 = locked ? origPos4 : newPos[globalThreadIndex];

    if (pos4.w > 0.0f)
    {
        // (CUDA reads the material here but only data.mMaxVelocity is used, so the
        // material fetch is dropped as dead.)
        const float maxVelocity = locked ? 0.0f : dataMaxVelocity;

        const float4 newVel4 = newVels[globalThreadIndex];
        float3 newVel = float3(newVel4.x, newVel4.y, newVel4.z);

        const float4 j = deltas[globalThreadIndex];
        const float scale = 1.0f / max(j.w, 1.0f);
        newVel.x += j.x * scale;
        newVel.y += j.y * scale;
        newVel.z += j.z * scale;

        const float magSq = dot(newVel, newVel);
        if (magSq > (maxVelocity * maxVelocity))
        {
            const float ratio = maxVelocity * rsqrt(magSq);
            newVel *= ratio;
        }

        if (lockFlags != 0u)
        {
            if ((lockFlags & PXG_PARTICLE_LOCK_X) != 0u) { pos4.x = origPos4.x; newVel.x = 0.0f; }
            if ((lockFlags & PXG_PARTICLE_LOCK_Y) != 0u) { pos4.y = origPos4.y; newVel.y = 0.0f; }
            if ((lockFlags & PXG_PARTICLE_LOCK_Z) != 0u) { pos4.z = origPos4.z; newVel.z = 0.0f; }
        }

        // llvmpipe-safety bound guard on the scatter: a valid reorder always
        // yields sortedIndex < numParticles (behaviour-preserving); it neutralises
        // the un-ported-reorder garbage-index case on the P6.0 free-fall path.
        if (sortedIndex < numParticles)
        {
            position[sortedIndex] = pos4;
            originalVels[sortedIndex] = float4(newVel.x, newVel.y, newVel.z, 0.0f);
        }

        deltas[globalThreadIndex] = float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
}

// ---------------------------------------------------------------------------
// ps_updateUserBufferLaunch — copy solved state to user buffer (particlesystem.cu:269-301)
// ---------------------------------------------------------------------------
// CUDA __global__ takes 2 pointers; the launch site passes a 3rd PxU32 `count`
// (unused by the kernel) -> the marshalled pack is { particleSystems*,
// activeParticleSystems*, PxU32 count } -> 24-byte blob.
//
// P9.97 (P9.22 rule) -- HARDENING, NOT A DEFECT ON THIS PLATFORM: the per-lane SOURCE
// index into the unsorted arrays is clamped (UB:clamp, through the CUDA's own `ind`,
// :293).  The grid is (ceil(mMaxParticlesPerBuffer/1024), mMaxBuffersPerSystem,
// nbActive) (PxgParticleSystemCore.cpp:2520/2532).  Both maxima are taken over every
// buffer of every system and only grow, so a lane can sit past the end of the unsorted
// arrays (offset + x, typically for the LAST buffer; the arrays hold at least the
// summed capacities, grow-only and doubling, PxgCudaBuffer.cpp:47-64).  Only in-guard
// lanes store, so the clamp moves an address on a lane that stores nothing, never a
// value.  On llvmpipe 25.2.8 / LLVM 20.1.2 the unclamped reads are masked gathers and
// are never issued for a masked lane -- the particleSim*Guard fixtures measure this
// with every array ending at a PROT_NONE guard page; the clamp is for back ends that
// hoist a guarded load.  The per-workgroup reads at y >= nb (runsum[y], the sim-buffer
// record) are NOT clamped: they stay behind the CUDA's guard (:278), which llvmpipe
// compiles to a real branch because the guarded body is never a single block (it holds
// the per-lane `if`; lp_should_flatten_cf_list flattens only a single block of < 8
// instructions -- an edit that ever made this body one small block WOULD let those
// uniform reads be issued here).  For an empty system (nb == 0) the record and runsum
// pointers may be null, stale or uninitialised, and no index clamp could make them
// valid.
struct PsUpdateUserBufferParams
{
    uint64_t particleSystems;       // @0
    uint64_t activeParticleSystems; // @8
    uint  count;                    // @16 (ignored, matches the launch pack) -> 24
}
[[vk::push_constant]]
ConstantBuffer<PsUpdateUserBufferParams> gUpd;

[shader("compute")]
[numthreads(1024, 1, 1)]
void ps_updateUserBufferLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gUpd.particleSystems);
    const uint particleId = (Ptr<uint>(gUpd.activeParticleSystems))[groupId.z];             // UB:sys :272

    const uint bufferIndex = groupId.y;                                                       // UB:y :276
    const uint numParticleBuffers = systems[particleId].mCommonData_NumParticleBuffers;

    if (bufferIndex < numParticleBuffers)                                                     // UB:bufGuard :278
    {
        Ptr<float4> unsortedPositions = Ptr<float4>(systems[particleId].mUnsortedPositions_InvMass);
        Ptr<float4> unsortedVels = Ptr<float4>(systems[particleId].mUnsortedVelocities);

        Ptr<PxgParticleSimBufferGpu> simBuffers = Ptr<PxgParticleSimBufferGpu>(systems[particleId].mParticleSimBuffers);
        Ptr<uint> runsum = Ptr<uint>(systems[particleId].mParticleBufferRunsum);
        const uint offset = runsum[bufferIndex];                                                   // UB:offset :284

        Ptr<float4> particles = Ptr<float4>(simBuffers[bufferIndex].mPositionInvMasses);          // UB:dstPos :286
        Ptr<float4> velocities = Ptr<float4>(simBuffers[bufferIndex].mVelocities);                // UB:dstVel :287
        const uint numActiveParticles = simBuffers[bufferIndex].mNumActiveParticles;              // UB:count :291

        const uint globalThreadIndex = groupThreadId.x + groupId.x * PS_BLOCK_DIM;            // UB:gtid :289
        const uint ind = offset + globalThreadIndex;                                           // UB:ind :293 (above the guard: integer only)
        const uint srcInd = (globalThreadIndex < numActiveParticles) ? ind : 0u;                // UB:clamp (P9.97)

        if (globalThreadIndex < numActiveParticles)                                            // UB:guard :291
        {
            particles[globalThreadIndex] = unsortedPositions[srcInd];                          // UB:posCopy :294
            velocities[globalThreadIndex] = unsortedVels[srcInd];                              // UB:velCopy :295
        }

        if (globalThreadIndex == 0u)                                                           // UB:flagsGuard :298
            simBuffers[bufferIndex].mFlags = 0u;                                                    // UB:flagsClear :299
    }
}

// ---------------------------------------------------------------------------
// ps_updateUnsortedArrayLaunch — ingest user buffer -> unsorted (particlesystem.cu:218-267)
// ---------------------------------------------------------------------------
// argSizes: { particleSystems*, activeParticleSystems* } -> 16-byte blob.
//
// P9.97 (P9.22 rule) -- HARDENING, NOT A DEFECT ON THIS PLATFORM: the per-lane SOURCE
// index is clamped (UA:clamp, as the removed applyParticleBufferDataDEPRECATED's AP:clamp was), and the
// optionally-NULL rest pointer is selected (UA:restSel, P8.2c rule); the CUDA's own null
// test is kept (UA:restFlag), so a null rest buffer is still never copied.  The grid is
// (ceil(mMaxParticlesPerBuffer/1024), mMaxBuffersPerSystem, nbActive)
// (PxgParticleSystemCore.cpp:753/764).  Both maxima are taken over every buffer of
// every system and only grow, so a lane can sit past THIS buffer's user arrays.  Only
// in-guard lanes store, so the clamps move an address on a lane that stores nothing,
// never a value.  On llvmpipe 25.2.8 / LLVM 20.1.2 the unclamped reads are masked
// gathers and are never issued for a masked lane (the null test folds into the mask);
// the particleSim*Guard fixtures measure this with PROT_NONE guard pages.  The clamps
// are for back ends that hoist a guarded load.  NULL rest + REST flag is LIVE in the
// engine: initParticleSimBuffer leaves mRestPositions NULL for every non-cloth buffer
// (PxgPBDParticleSystemCore.cpp:1411) and the rigid-buffer update raises the REST flag
// (:1519-1521) -- PXG_SCENE=particlerigid ingests exactly that; without the null test
// AND without the select it segfaults at its first step.  The per-workgroup reads at
// y >= nb are NOT clamped: see ps_updateUserBufferLaunch (guard :227, a real branch on
// llvmpipe; nb == 0 pointers may be null, stale or uninitialised).
struct PsUpdateUnsortedParams
{
    uint64_t particleSystems;       // @0
    uint64_t activeParticleSystems; // @8 -> 16
}
[[vk::push_constant]]
ConstantBuffer<PsUpdateUnsortedParams> gIngest;

[shader("compute")]
[numthreads(1024, 1, 1)]
void ps_updateUnsortedArrayLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gIngest.particleSystems);
    const uint particleId = (Ptr<uint>(gIngest.activeParticleSystems))[groupId.z];          // UA:sys :221

    const uint bufferIndex = groupId.y;                                                       // UA:y :225
    const uint numParticleBuffers = systems[particleId].mCommonData_NumParticleBuffers;

    if (bufferIndex < numParticleBuffers)                                                     // UA:bufGuard :227
    {
        Ptr<float4> unsortedPositions = Ptr<float4>(systems[particleId].mUnsortedPositions_InvMass);
        Ptr<float4> unsortedVels = Ptr<float4>(systems[particleId].mUnsortedVelocities);
        Ptr<float4> unsortedRestArray = Ptr<float4>(systems[particleId].mRestArray);
        Ptr<uint> unsortedPhases = Ptr<uint>(systems[particleId].mUnsortedPhaseArray);

        Ptr<PxgParticleSimBufferGpu> simBuffers = Ptr<PxgParticleSimBufferGpu>(systems[particleId].mParticleSimBuffers);
        Ptr<uint> runsum = Ptr<uint>(systems[particleId].mParticleBufferRunsum);
        const uint offset = runsum[bufferIndex];                                                   // UA:offset :236

        const uint64_t particlesAddr = simBuffers[bufferIndex].mPositionInvMasses;                // UA:srcPos :238
        const uint64_t velocitiesAddr = simBuffers[bufferIndex].mVelocities;                      // UA:srcVel :239
        const uint64_t restAddr = simBuffers[bufferIndex].mRestPositions;                         // UA:srcRest :240
        const uint64_t phasesAddr = simBuffers[bufferIndex].mPhases;                              // UA:srcPhase :241
        const uint numActiveParticles = simBuffers[bufferIndex].mNumActiveParticles;              // UA:count :248
        const uint flags = simBuffers[bufferIndex].mFlags;                                        // UA:flags :246

        const uint globalThreadIndex = groupThreadId.x + groupId.x * PS_BLOCK_DIM;            // UA:gtid :244
        const uint srcIdx = (globalThreadIndex < numActiveParticles) ? globalThreadIndex : 0u;  // UA:clamp (P9.97)
        const uint64_t restSrc = (restAddr != 0UL) ? restAddr : particlesAddr;                // UA:restSel (P9.97)

        if (globalThreadIndex < numActiveParticles)                                            // UA:guard :248
        {
            const uint ind = offset + globalThreadIndex;                                       // UA:ind :250
            if ((flags & PS_BUFFER_FLAG_UPDATE_POSITION) != 0u)                                // UA:posFlag :251
                unsortedPositions[ind] = Ptr<float4>(particlesAddr)[srcIdx];                   // UA:posCopy :252
            if ((flags & PS_BUFFER_FLAG_UPDATE_VELOCITY) != 0u)                                // UA:velFlag :254
                unsortedVels[ind] = Ptr<float4>(velocitiesAddr)[srcIdx];                       // UA:velCopy :255
            if ((flags & PS_BUFFER_FLAG_UPDATE_PHASE) != 0u)                                   // UA:phaseFlag :257
                unsortedPhases[ind] = Ptr<uint>(phasesAddr)[srcIdx];                           // UA:phaseCopy :258
            if (((flags & PS_BUFFER_FLAG_UPDATE_RESTPOSITION) != 0u) && restAddr != 0UL)        // UA:restFlag :260
                unsortedRestArray[ind] = Ptr<float4>(restSrc)[srcIdx];                         // UA:restCopy :261
        }
    }
}

// (P9.95's applyParticleBufferDataDEPRECATED -- direct-GPU-API ingest by unique id --
// was removed upstream in PhysX 5.12 together with its host path.)
