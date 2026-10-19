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
//
// ===========================================================================
// femClothStepBda — cloth_stepLaunch (gpusimulationcontroller/src/CUDA/FEMCloth.cu:176,
// launch PxgFEMClothCore.cpp:1501, CLOTH_SIM_STEP), on the engine BDA ABI.
//
// *** THIS IS THE TGS CLOTH PATH'S INTEGRATOR, AND WITHOUT IT NO CLOTH MOVES. ***
// On PGS the fall comes from `cloth_preIntegrateLaunch` (already mapped, femClothSimBda), which
// does the gravity predict AND the position integration in one shot at the top of the step.  On
// TGS that kernel only predicts; the POSITION integration lives here and runs once per solver
// substep, called from `PxgFEMClothCore::step()` inside the `if (mIsTGS)` branch at
// PxgFEMClothCore.cpp:1380.  With CLOTH_SIM_STEP unmapped the engine refuses the launch and the
// cloth simply never advances -- which is exactly what the harness shows: `PXG_SCENE=clothdrape`
// on the default TGS solver reports `dropped 0.0000 over 3.00s` and FAILS, while the same binary
// with `PXG_SOLVER=pgs` settles and passes.  That pre-existing red scene is this port's proof.
//
// WHAT IT DOES, per active cloth (blockIdx.y) and per vertex:
//
//   1. THE CONTACT-PAIR BOOKKEEPING, done by ONE thread of the whole dispatch
//      (`!adaptive && globalThreadIndex == 0 && blockIdx.y == 0`): it publishes
//      `*updateContactPairs = forceUpdateClothContactPairs` and, when that is set, ZEROES the VT
//      and EE contact counts.  Those two stores are what makes the narrowphase re-run; dropping
//      them leaves the solver re-using last substep's contact set forever.
//   2. THE OPTIONAL GRAVITY + DAMPING PREDICT, only when
//      `externalForcesEveryTgsIterationEnabled` -- i.e. only when the app asked for external
//      forces on every TGS iteration.  Gated on `tPos.w != 0` (a pinned vertex has invMass 0 and
//      is left alone) and on `mActorFlags & eDISABLE_GRAVITY`.  Damping is
//      `1 - min(1, linearDamping * dt)`, a clamp that keeps a large damping from reversing the
//      velocity rather than killing it.
//   3. THE INTEGRATION: `delta = vel * dt`, added to BOTH the position and the accumulated
//      delta.  This is the observable -- the cloth vertex arrays alias the user buffer, so this
//      store is what a host read of getPositionInvMassBufferD() sees.
//   4. THE MAXIMUM-DISPLACEMENT CLAMP, against `mPrevPositionInRestOffset` rather than against
//      this substep's start, so it bounds the trajectory over the WHOLE step and not per
//      substep.  Three regimes, and the middle one is easy to miss:
//          mMaxLinearVelocity == PX_MAX_REAL  -> no limit at all (the whole block is skipped)
//          mMaxLinearVelocity <  0            -> clamp to 80% of mRestDistance
//          otherwise                          -> clamp to mMaxLinearVelocity * dt
//      The correction is applied to the position, the accumulated delta AND (scaled by 1/dt) to
//      the velocity, so a clamped vertex does not carry the excess into the next substep.
//      It only fires when `isPrevPosSet == 1`, i.e. once a contact-offset reference exists.
//   5. THE PREVIOUS-POSITION WRITES: `mPrevPositionInRestOffset` unconditionally, and
//      `mPrevPositionInContactOffset` only on a forced update -- with `.w` set to 1 to MARK the
//      reference as valid.  That flag is read back as `isPrevPosSet` above and by step 6.
//   6. THE ADAPTIVE INVALIDATION, when `adaptiveCollisionPairUpdate`: a vertex that has moved
//      further than `mOriginalContactOffset` from its contact-offset reference (or has no
//      reference yet) raises a per-thread flag; the flags are ORed BLOCK-WIDE through shared
//      memory and thread 0 publishes `*updateContactPairs = 1`.
//
// *** THE `__syncthreads()` BEFORE THE OR IS NOT DECORATIVE. ***  Upstream syncs, then ORs into
// the shared flag, then syncs again.  The first barrier is what makes the flag's initialisation
// visible; the second is what makes the OR visible to thread 0.  Both are reproduced, and BOTH
// are reached by every invocation -- the vertex-range test `globalThreadIndex < nbVerts` guards
// only the body, never a barrier.  A `return` for out-of-range threads would deadlock the group.
//
// DEVIATIONS from CUDA (recipe policy):
//   - `__shared__ PxU8 hasInvalidContacts` becomes a `groupshared uint` written with
//     InterlockedOr, since llvmpipe has no 8-bit shared storage in these shaders;
//   - `*updateContactPairs` is a PxU8*.  A one-byte store is not available either, so the
//     containing 32-bit word is read-modify-written -- and this is SAFE here in a way it would
//     not generally be.  CORRECTED IN P9.83a: this note used to say "the two writers in this
//     kernel are both single-threaded and separated by a barrier", and the second half of that
//     is false.  The !adaptive writer IS single-threaded across the whole dispatch
//     (globalThreadIndex == 0 && gid.y == 0).  The adaptive one fires on `gtid.x == 0` of
//     EVERY group that saw an invalid contact, which is many invocations.  What makes it safe
//     is IDEMPOTENCE, not exclusivity: every one of those writers stores the same value 1 into
//     the same byte, so concurrent read-modify-writes all read the same word, compute the same
//     result and store it.  The two arms are mutually exclusive on the kernel-wide `adaptive`
//     flag, and the buffer is a dedicated one-byte PxgCudaBuffer
//     (PxgFEMClothCore.cpp:1492) whose word's other three bytes are unused padding inside the
//     same allocation, so nothing else can lose an update to the collateral rewrite.
//     tools/check_subword_store.py carries the site with that reason.
//
// THE GRID: gridDim.x = ceil(maxClothVerts / CLOTH_STEP), gridDim.y = nbActiveFEMCloths, with
// blockDim CLOTH_STEP = 1024.  The bridge forwards gridDimX/Y and DROPS blockDim, so the group
// is declared [numthreads(1024,1,1)] to match -- the same treatment
// cloth_finalizeVelocitiesLaunch and cloth_sleeping already get in femClothSimBda.
//
// NoContraction-stamped (PXG_NOCONTRACT_SHADERS): the integration and the clamp correction are
// mul-feeding-add chains whose fusion would drift the position observable the scenes assert on.
// ===========================================================================
module femClothStepBda;

import pxgSlangFEMClothDesc;   // PxgFEMClothGpu / PXG_FEMCLOTH_ACTORFLAG_DISABLE_GRAVITY

// PxgFEMClothKernelBlockDim::CLOTH_STEP (PxgFEMClothCoreKernelIndices.h:40).
static const uint CLOTH_STEP = 1024u;
// PX_MAX_REAL (foundation/PxSimpleTypes) == FLT_MAX.
static const float PX_MAX_REAL = 3.40282347e+38f;

// ---------------------------------------------------------------------------
// Push-constant blob — BYTE-IDENTICAL to the CUDA pack (PxgFEMClothCore.cpp:1507):
// { PxgFEMCloth* femCloths, PxU32* activeId, PxReal dt, PxVec3 gravity,
//   bool externalForcesEveryTgsIterationEnabled, bool adaptiveCollisionPairUpdate,
//   bool forceUpdateClothContactPairs, PxU8* updateContactPairs, PxU32* VTContactCount,
//   PxU32* EEContactCount }
//
// paramAlign: PxVec3 is TWELVE bytes and 12 % 16 != 0, 12 % 8 != 0, 12 % 4 == 0, so it aligns
// to 4 and sits at @20 rather than being padded to 16.  The three bools are one byte each and
// pack at @32, @33, @34; the next pointer then realigns to @40.
// -> 8 @0, 8 @8, 4 @16, 12 @20, 1 @32, 1 @33, 1 @34, 8 @40, 8 @48, 8 @56 -> 64-byte blob.
// ABI row { 8,8,4,12,1,1,1,8,8,8 }.
// ---------------------------------------------------------------------------
struct ClothStepParams
{
    uint64_t femCloths;             // @0
    uint64_t activeId;              // @8
    float dt;                       // @16
    float gravityX;                 // @20
    float gravityY;                 // @24
    float gravityZ;                 // @28
    uint externalAndAdaptiveAndForce; // @32..@35 — the three bools plus one tail pad byte
    uint _pad36;                    // @36
    uint64_t updateContactPairs;    // @40  PxU8*
    uint64_t vtContactCount;        // @48  PxU32*
    uint64_t eeContactCount;        // @56  PxU32* -> 64
}

[[vk::push_constant]] ConstantBuffer<ClothStepParams> gSt;

// The three `bool` arguments occupy bytes @32, @33 and @34 of the blob.  Slang has no 8-bit
// push-constant member, so they are read out of the containing word: byte 0 is
// externalForcesEveryTgsIterationEnabled, byte 1 adaptiveCollisionPairUpdate, byte 2
// forceUpdateClothContactPairs.  A C++ `bool` is stored as 0 or 1, so any non-zero is true.
bool stExternalForces()  { return ((gSt.externalAndAdaptiveAndForce      ) & 0xffu) != 0u; }
bool stAdaptiveUpdate()  { return ((gSt.externalAndAdaptiveAndForce >>  8) & 0xffu) != 0u; }
bool stForceUpdate()     { return ((gSt.externalAndAdaptiveAndForce >> 16) & 0xffu) != 0u; }

// A single PxU8 store through the containing 32-bit word.  See the header on why the
// read-modify-write is safe for THIS buffer specifically.  NOTE the `& ~uint64_t(3)`: writing
// `& ~3UL` MISCOMPILES, because Slang evaluates the complement in 32 bits and clears the top
// half of the device address.
void storeByte(uint64_t base, uint value)
{
    const uint64_t wordAddr = base & ~uint64_t(3);
    const uint shift = uint(base & 3UL) * 8u;
    Ptr<uint> p = Ptr<uint>(wordAddr);
    p[0] = (p[0] & ~(0xffu << shift)) | ((value & 0xffu) << shift);
}

// ---------------------------------------------------------------------------
// The block-wide OR of the per-thread invalid-contact flags.
// ---------------------------------------------------------------------------
groupshared uint sHasInvalidContacts;

// ===========================================================================
[shader("compute")]
[numthreads(1024, 1, 1)]   // == CLOTH_STEP
void cloth_stepLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    if (gtid.x == 0u)
        sHasInvalidContacts = 0u;
    GroupMemoryBarrierWithGroupSync();

    Ptr<PxgFEMClothGpu> femCloths = Ptr<PxgFEMClothGpu>(gSt.femCloths);
    const uint id = Ptr<uint>(gSt.activeId)[gid.y];

    const uint nbVerts = femCloths[id].mNbVerts;

    Ptr<float4> velocities = Ptr<float4>(femCloths[id].mVelocity_InvMass);
    Ptr<float4> positions = Ptr<float4>(femCloths[id].mPosition_InvMass);
    Ptr<float4> posDelta = Ptr<float4>(femCloths[id].mAccumulatedDeltaPos);
    Ptr<float4> prevPosInContactOffset = Ptr<float4>(femCloths[id].mPrevPositionInContactOffset);
    Ptr<float4> prevPosInRestOffset = Ptr<float4>(femCloths[id].mPrevPositionInRestOffset);

    const float dt = gSt.dt;
    const float3 gravity = float3(gSt.gravityX, gSt.gravityY, gSt.gravityZ);
    const bool externalForces = stExternalForces();
    const bool adaptive = stAdaptiveUpdate();
    const bool forceUpdate = stForceUpdate();

    const uint globalThreadIndex = gtid.x + CLOTH_STEP * gid.x;

    // ONE thread of the whole dispatch publishes the contact-pair gate and, on a forced update,
    // zeroes both contact counts.  blockIdx.y == 0 as well, so it happens once per step and not
    // once per cloth.
    if (!adaptive && globalThreadIndex == 0u && gid.y == 0u)
    {
        storeByte(gSt.updateContactPairs, forceUpdate ? 1u : 0u);
        if (forceUpdate)
        {
            Ptr<uint>(gSt.vtContactCount)[0] = 0u;
            Ptr<uint>(gSt.eeContactCount)[0] = 0u;
        }
    }

    uint invalidContact = 0u;
    if (globalThreadIndex < nbVerts)
    {
        const float4 tPos = positions[globalThreadIndex];
        const float4 tVel = velocities[globalThreadIndex];
        const float4 tAccumDelta = posDelta[globalThreadIndex];

        const float4 prevPosContact4 = prevPosInContactOffset[globalThreadIndex];
        const float isPrevPosSet = prevPosContact4.w;
        const float3 prevPosContact = prevPosContact4.xyz;

        float3 vel = tVel.xyz;
        float3 pos = tPos.xyz;
        float3 accumDelta = tAccumDelta.xyz;

        if (externalForces)
        {
            // tPos.w is the INVERSE MASS: zero means pinned, and a pinned vertex takes neither
            // gravity nor damping.
            if (tPos.w != 0.0f)
            {
                const uint actorFlags = femCloths[id].mFlagsWord & 0xffu;
                if ((actorFlags & PXG_FEMCLOTH_ACTORFLAG_DISABLE_GRAVITY) == 0u)
                    vel += gravity * dt;

                const float velocityDamping = femCloths[id].mLinearDamping;
                // The min() is the point: a damping * dt above 1 would otherwise REVERSE the
                // velocity instead of killing it.
                const float damping = 1.0f - min(1.0f, velocityDamping * dt);
                vel *= damping;
            }
        }

        const float3 delta = vel * dt;
        pos += delta;
        accumDelta += delta;

        // The maximum-displacement clamp — three regimes, see the header.
        if (femCloths[id].mMaxLinearVelocity < PX_MAX_REAL)
        {
            const float invDt = 1.0f / dt;
            const float3 prevPosRest = prevPosInRestOffset[globalThreadIndex].xyz;

            const float maxLinVel = femCloths[id].mMaxLinearVelocity;
            const float maxDistance = (maxLinVel < 0.0f) ? 0.8f * femCloths[id].mRestOffset
                                                         : maxLinVel * dt;
            const float maxDistanceSq = maxDistance * maxDistance;

            // Measured against the REST-offset reference, so the bound is on the whole step's
            // trajectory rather than on this substep's.
            const float3 trajectory = pos - prevPosRest;
            const float displacementSq = dot(trajectory, trajectory);

            if (isPrevPosSet == 1.0f && displacementSq > maxDistanceSq)
            {
                const float correctionScale = sqrt(maxDistanceSq / displacementSq);
                float3 correction = (correctionScale - 1.0f) * trajectory;

                pos += correction;
                accumDelta += correction;

                correction *= invDt;
                vel += correction;
            }
        }

        float4 pos4 = float4(pos, tPos.w);

        positions[globalThreadIndex] = pos4;
        velocities[globalThreadIndex] = float4(vel, tVel.w);
        posDelta[globalThreadIndex] = float4(accumDelta, tAccumDelta.w);

        prevPosInRestOffset[globalThreadIndex] = pos4;
        if (forceUpdate)
        {
            pos4.w = 1.0f;   // MARK the contact-offset reference as set — read back as isPrevPosSet
            prevPosInContactOffset[globalThreadIndex] = pos4;
        }

        if (adaptive)
        {
            if (isPrevPosSet == 1.0f)
            {
                const float3 d = pos - prevPosContact;
                const float offset = femCloths[id].mOriginalContactOffset;
                if (dot(d, d) > offset * offset)
                    invalidContact = 1u;   // moved past the contact offset: pairs are stale
            }
            else
            {
                invalidContact = 1u;       // no reference yet: the first query of the simulation
            }
        }
    }

    // Both barriers are reached by EVERY invocation — the vertex-range test above guards only
    // the body.  See the header.
    GroupMemoryBarrierWithGroupSync();
    if (invalidContact == 1u)
        InterlockedOr(sHasInvalidContacts, 1u);
    GroupMemoryBarrierWithGroupSync();

    if (adaptive && sHasInvalidContacts == 1u && gtid.x == 0u)
        storeByte(gSt.updateContactPairs, 1u);
}
