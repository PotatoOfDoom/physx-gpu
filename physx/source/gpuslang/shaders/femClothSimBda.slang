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

// femClothSimBda — P7.0: the FIRST live FEMCloth frame on the Vulkan engine ABI
// (BDA).  The three cloth orchestration kernels that drive a triangle-mesh cloth
// FALLING under gravity, ported line-for-line from
// gpusimulationcontroller/src/CUDA/FEMCloth.cu (the exact analog of the completed
// PBD particle free-fall in particleSim.slang):
//   - cloth_preIntegrateLaunch (FEMCloth.cu:48): the free-fall driver.  Gravity
//     predict AND the position integration in one shot on the PGS path
//     (positions[i] = oldPos + vel*dt, the observable fall).  The cloth vertex
//     arrays ALIAS the user buffer (PxgSimulationCore.cpp:1850), so this write is
//     what a host DtoH read of getPositionInvMassBufferD() sees — no ingest/egress
//     kernel exists.  Both the TGS and PGS branches are ported.
//   - cloth_finalizeVelocitiesLaunch (FEMCloth.cu:1227): velocity settling/averaging
//     + sets mIsActive (block-wide OR -> device atomicOr).
//   - cloth_sleeping (FEMCloth.cu:1313): wake-counter bookkeeping.  REQUIRED — else
//     the cloth deactivates (wakeCounters -> 0 in ~24 frames) and the fall freezes.
//
// Deviations from the CUDA original (recipe policy; all preserve the free-fall
// semantics and are exercised by testFemClothSim.cpp):
//   - the descriptor arrives as a device pointer to an array of PxgFEMCloth read
//     through Ptr<> (BDA), reusing the byte-exact pxgSlangFEMClothDesc mirror;
//   - cloth_preIntegrateLaunch's adaptiveCollisionPairUpdate branch (its ONLY warp
//     op, warpReduction -> AtomicMax on speculativeCCDContactOffset) is OMITTED.
//     The P7.0 scene sets nbCollisionPairUpdatesPerTimestep != 0, so the engine
//     passes adaptiveCollisionPairUpdate = false and that branch is dead — this
//     keeps a subgroup op off the llvmpipe predictor entirely (matching the recipe
//     guard).  maxMagVelSq (fed only by that branch) is likewise dropped.
//   - the globalThreadIndex==0 bookkeeping writes the PxU32 VT/EE contact-count
//     resets (4-byte, faithful) AND the `*updateContactPairs = !isTGS` gate.  That
//     gate store was elided by the original P7.0 port (nothing mapped read it then);
//     P7.2 tranche 3b restored it, because every self-collision kernel early-outs on
//     it.  It is a 4-byte store, not the CUDA PxU8 (no 8-bit storage on llvmpipe) —
//     safe because the engine clears the same buffer with memsetD32Async and every
//     consumer reads `word & 0xFF`.

import pxgSlangFEMClothDesc;

static const uint CLOTH_WARP_SIZE = 32u;
// PxgFEMClothKernelBlockDim (PxgFEMClothCoreKernelIndices.h:39-40).
static const uint CLOTH_PREINTEGRATION = 512u;
static const uint CLOTH_STEP = 1024u;
static const uint CLOTH_STEP_NUM_WARPS = CLOTH_STEP / CLOTH_WARP_SIZE; // 32
// PX_MAX_REAL (foundation/PxSimpleTypes) == FLT_MAX.
static const float PX_MAX_REAL = 3.40282347e+38f;

// ---------------------------------------------------------------------------
// cloth_preIntegrateLaunch — gravity predict + position integration (FEMCloth.cu:48)
// ---------------------------------------------------------------------------
// CUDA arg pack (PxgFEMClothCore.cpp:177-190): { PxgFEMCloth* femCloths, const
// PxU32* activeIds, PxVec3 gravity(12), PxReal dt(4), bool isTGS(1), PxReal*
// speculativeCCDContactOffset, bool externalForcesEveryTgsIterationEnabled(1),
// bool adaptiveCollisionPairUpdate(1), PxU8* updateContactPairs, PxU32*
// VTContactCounts, PxU32* EEContactCounts } -> natural-alignment blob = 80 bytes.
struct PreIntegrateParams
{
    uint64_t femCloths;                 // @0
    uint64_t activeIds;                 // @8
    float gravityX;                     // @16
    float gravityY;                     // @20
    float gravityZ;                     // @24
    float dt;                           // @28
    uint isTGSWord;                     // @32  byte0 = isTGS
    uint _pad36;                        // @36  (isTGS's 7 tail-pad bytes -> next @40)
    uint64_t speculativeCCDContactOffset; // @40
    uint extFlagsWord;                  // @48  byte0 = extForces, byte1 = adaptivePairUpd
    uint _pad52;                        // @52  -> next @56
    uint64_t updateContactPairs;        // @56
    uint64_t VTContactCounts;           // @64
    uint64_t EEContactCounts;           // @72  -> 80
}
[[vk::push_constant]]
ConstantBuffer<PreIntegrateParams> gPre;

[shader("compute")]
[numthreads(32, 16, 1)] // (WARP_SIZE, CLOTH_PREINTEGRATION/WARP_SIZE) == (32,16) = 512
void cloth_preIntegrateLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgFEMClothGpu> femCloths = Ptr<PxgFEMClothGpu>(gPre.femCloths);
    const uint id = (Ptr<uint>(gPre.activeIds))[groupId.y];

    const uint nbVerts = femCloths[id].mNbVerts;

    Ptr<float4> velocities = Ptr<float4>(femCloths[id].mVelocity_InvMass);
    Ptr<float4> positions = Ptr<float4>(femCloths[id].mPosition_InvMass);
    Ptr<float4> posDelta = Ptr<float4>(femCloths[id].mAccumulatedDeltaPos);
    Ptr<float4> prevPosInContact = Ptr<float4>(femCloths[id].mPrevPositionInContactOffset);
    const float velocityDamping = femCloths[id].mLinearDamping;

    const uint threadIdxInWarp = groupThreadId.x;
    const uint warpIndex = groupThreadId.y;
    const uint globalThreadIndex =
        threadIdxInWarp + CLOTH_WARP_SIZE * warpIndex + CLOTH_PREINTEGRATION * groupId.x;

    const bool isTGS = (gPre.isTGSWord & 0xffu) != 0u;
    const bool externalForcesEveryTgsIterationEnabled = (gPre.extFlagsWord & 0xffu) != 0u;
    const bool adaptiveCollisionPairUpdate = ((gPre.extFlagsWord >> 8) & 0xffu) != 0u;

    if (globalThreadIndex == 0u)
    {
        femCloths[id].mIsActive = 0u; // reset each frame; finalize ORs it back

        if (groupId.y == 0u)
        {
            // Reset cloth contact counts if adaptive updates are not used (4-byte
            // PxU32 stores — faithful).
            if (!adaptiveCollisionPairUpdate)
            {
                (Ptr<uint>(gPre.VTContactCounts))[0] = 0u;
                (Ptr<uint>(gPre.EEContactCounts))[0] = 0u;
            }

            // For TGS, updateContactPairs is updated in "cloth_stepLaunch".
            // P7.2 tranche 3b: this store was ELIDED by the P7.0 port because no mapped
            // kernel consumed it back then.  It is now the GATE every self-collision
            // kernel reads first (cloth_selfCollisionMidphase{VT,EE}, cloth_refitBound,
            // cloth_clothContactPrepare all early-out on `*updateContactPairs == 0`), so
            // eliding it silently disabled the entire self-collision arc.  Written as a
            // 4-byte store rather than the CUDA PxU8 (8-bit storage is not available on
            // llvmpipe): the engine itself clears this buffer with memsetD32Async
            // (PxgFEMClothCore.cpp:1397), and every consumer reads it as `word & 0xFF`,
            // so the whole word is ours to own and the observable low byte is identical.
            (Ptr<uint>(gPre.updateContactPairs))[0] = isTGS ? 0u : 1u;
        }
    }

    if (globalThreadIndex < nbVerts)
    {
        const float4 tPos = positions[globalThreadIndex];
        const float4 oldVel = velocities[globalThreadIndex];

        float3 vel = float3(oldVel.x, oldVel.y, oldVel.z);
        const float3 gravity = float3(gPre.gravityX, gPre.gravityY, gPre.gravityZ);

        if (!externalForcesEveryTgsIterationEnabled && tPos.w != 0.0f)
        {
            const uint actorFlags = femCloths[id].mFlagsWord & 0xffu;
            if ((actorFlags & PXG_FEMCLOTH_ACTORFLAG_DISABLE_GRAVITY) == 0u)
            {
                vel += gravity * gPre.dt;
            }
            const float damping = 1.0f - min(1.0f, velocityDamping * gPre.dt);
            vel = vel * damping;
        }

        if (isTGS)
        {
            velocities[globalThreadIndex] = float4(vel.x, vel.y, vel.z, tPos.w);
            posDelta[globalThreadIndex] = float4(0.0f, 0.0f, 0.0f, tPos.w);
        }
        else
        {
            // PGS: clamp the per-vertex displacement per the input maxVelocity.
            const float maxLinVel = femCloths[id].mMaxLinearVelocity;
            const float restDistance = femCloths[id].mRestDistance;
            const float maxVel = (maxLinVel == PX_MAX_REAL) ? PX_MAX_REAL
                                 : (maxLinVel < 0.0f) ? (0.8f * restDistance / gPre.dt)
                                 : maxLinVel;

            const float velMagSq = vel.x * vel.x + vel.y * vel.y + vel.z * vel.z;
            vel = (maxVel != PX_MAX_REAL && maxVel * maxVel < velMagSq)
                      ? (maxVel / sqrt(velMagSq)) * vel
                      : vel;

            const float3 delta = vel * gPre.dt;

            velocities[globalThreadIndex] = float4(vel.x, vel.y, vel.z, tPos.w);
            posDelta[globalThreadIndex] = float4(delta.x, delta.y, delta.z, tPos.w);

            const float3 oldPos = float3(tPos.x, tPos.y, tPos.z);
            const float3 pos = oldPos + delta;
            positions[globalThreadIndex] = float4(pos.x, pos.y, pos.z, tPos.w);

            prevPosInContact[globalThreadIndex] = float4(pos.x, pos.y, pos.z, 1.0f);
        }
    }

    // The adaptiveCollisionPairUpdate speculative-CCD reduction is intentionally
    // omitted (dead on the non-adaptive P7.0 scene) — see the header.
}

// ---------------------------------------------------------------------------
// cloth_finalizeVelocitiesLaunch — velocity settling + wake flag (FEMCloth.cu:1227)
// ---------------------------------------------------------------------------
// CUDA arg pack (PxgFEMClothCore.cpp:1234-1236): { PxgFEMCloth* femCloths, const
// PxU32* activeId, PxReal invTotalDt(4), PxReal dt(4), bool
// alwaysRunVelocityAveraging(1) } -> natural-alignment blob = 32 bytes.  NOTE:
// invTotalDt is passed but unreferenced in the CUDA body (kept for ABI fidelity).
struct FinalizeParams
{
    uint64_t femCloths;                 // @0
    uint64_t activeId;                  // @8
    float invTotalDt;                   // @16 (unused, ABI placeholder)
    float dt;                           // @20
    uint alwaysAvgWord;                 // @24 byte0 = alwaysRunVelocityAveraging
    uint _pad28;                        // @28 -> 32
}
[[vk::push_constant]]
ConstantBuffer<FinalizeParams> gFin;

groupshared uint gAnyAwake;

[shader("compute")]
[numthreads(1024, 1, 1)] // CLOTH_STEP
void cloth_finalizeVelocitiesLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgFEMClothGpu> femCloths = Ptr<PxgFEMClothGpu>(gFin.femCloths);
    const uint id = (Ptr<uint>(gFin.activeId))[groupId.y];

    const uint nbVerts = femCloths[id].mNbVerts;
    Ptr<float4> vels = Ptr<float4>(femCloths[id].mVelocity_InvMass);
    Ptr<float4> posDelta = Ptr<float4>(femCloths[id].mAccumulatedDeltaPos);
    Ptr<float4> positions = Ptr<float4>(femCloths[id].mPosition_InvMass);

    const float dt = gFin.dt;
    const bool alwaysRunVelocityAveraging = (gFin.alwaysAvgWord & 0xffu) != 0u;

    const uint globalThreadIndex = groupThreadId.x + CLOTH_STEP * groupId.x;

    if (groupThreadId.x == 0u)
        gAnyAwake = 0u;
    GroupMemoryBarrierWithGroupSync();

    bool awake = false;

    if (globalThreadIndex < nbVerts)
    {
        const float settleTolerance = femCloths[id].mSettlingThreshold * dt;
        const float tolerance = femCloths[id].mSleepThreshold * dt;
        const float sleepDamping = 1.0f - min(1.0f, femCloths[id].mSettlingDamping * dt);

        const float4 delta = posDelta[globalThreadIndex];
        const float3 tDelta = float3(delta.x, delta.y, delta.z);
        // P9.36: FEMCloth.cu:1258 is `PxVec3 deltaVel = tDelta / dt;` on a PxVec3, i.e.
        // PxVec3::operator/ (PxVec3.h:219) -- ONE reciprocal, three multiplies.
        const float3 deltaVel = tDelta * (1.0f / dt);
        const float4 pos = positions[globalThreadIndex];

        if (pos.w != 0.0f) // skip kinematic / infinite-mass vertices
        {
            float velocityScaling = 1.0f;

            const float magSq = tDelta.x * tDelta.x + tDelta.y * tDelta.y + tDelta.z * tDelta.z;
            if (magSq < settleTolerance * settleTolerance)
            {
                awake = magSq >= tolerance * tolerance;
                velocityScaling = sleepDamping;
            }
            else
            {
                awake = true;
            }

            if (alwaysRunVelocityAveraging || velocityScaling != 1.0f)
            {
                const float4 vel = vels[globalThreadIndex];
                float3 tVel = float3(vel.x, vel.y, vel.z);

                const float deltaVelMagSqr =
                    deltaVel.x * deltaVel.x + deltaVel.y * deltaVel.y + deltaVel.z * deltaVel.z;
                const float velMagSqr = tVel.x * tVel.x + tVel.y * tVel.y + tVel.z * tVel.z;
                if (alwaysRunVelocityAveraging &&
                    deltaVelMagSqr < (tVel.x * tVel.x + tVel.y * tVel.y + tVel.z * tVel.z))
                {
                    tVel = tVel * sqrt(deltaVelMagSqr / velMagSqr);
                }

                vels[globalThreadIndex] =
                    float4(velocityScaling * tVel.x, velocityScaling * tVel.y, velocityScaling * tVel.z, vel.w);
            }
        }
    }

    // block-wide OR of `awake` (replaces the CUDA two-level warp __any_sync).
    if (awake)
        InterlockedOr(gAnyAwake, 1u);
    GroupMemoryBarrierWithGroupSync();

    if (groupThreadId.x == 0u && gAnyAwake != 0u)
        InterlockedOr(femCloths[id].mIsActive, 1u); // BDA device atomicOr
}

// ---------------------------------------------------------------------------
// cloth_sleeping — wake-counter bookkeeping (FEMCloth.cu:1313)
// ---------------------------------------------------------------------------
// CUDA arg pack (PxgFEMClothCore.cpp:1265-1269): { PxgFEMCloth* femCloths, PxU32
// numActiveCloths(4), const PxU32* activeId, PxReal dt(4), PxReal resetCounter(4),
// PxReal* wakeCounters, PxU32* stateChangedMask } -> natural-alignment blob = 48.
struct SleepingParams
{
    uint64_t femCloths;                 // @0
    uint numActiveCloths;               // @8
    uint _pad12;                        // @12 (activeId is 8-aligned -> @16)
    uint64_t activeId;                  // @16
    float dt;                           // @24
    float resetCounter;                 // @28
    uint64_t wakeCounters;              // @32
    uint64_t stateChangedMask;          // @40 -> 48
}
[[vk::push_constant]]
ConstantBuffer<SleepingParams> gSleep;

groupshared uint gSleepMasks[CLOTH_STEP_NUM_WARPS]; // 32

[shader("compute")]
[numthreads(1024, 1, 1)] // CLOTH_STEP
void cloth_sleeping(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgFEMClothGpu> femCloths = Ptr<PxgFEMClothGpu>(gSleep.femCloths);
    Ptr<uint> activeId = Ptr<uint>(gSleep.activeId);
    Ptr<float> wakeCounters = Ptr<float>(gSleep.wakeCounters);
    Ptr<uint> stateChangedMask = Ptr<uint>(gSleep.stateChangedMask);

    const uint globalThreadIdx = groupThreadId.x + groupId.x * CLOTH_STEP;
    const uint laneId = groupThreadId.x & 31u;
    const uint warpId = groupThreadId.x >> 5;

    // Restructured from the CUDA early-return + __syncthreads (UB in a partly
    // filled block) to an `if (gid < n)` guard so all threads reach both barriers.
    if (groupThreadId.x < CLOTH_STEP_NUM_WARPS)
        gSleepMasks[groupThreadId.x] = 0u;
    GroupMemoryBarrierWithGroupSync();

    if (globalThreadIdx < gSleep.numActiveCloths)
    {
        const uint id = activeId[globalThreadIdx];

        const bool reset = femCloths[id].mIsActive != 0u;
        float counter = wakeCounters[id];
        const bool wasActive = counter > 0.0f;

        if (reset)
            counter = gSleep.resetCounter;
        else
            counter = max(0.0f, counter - gSleep.dt);

        const bool isActive = counter > 0.0f;

        wakeCounters[id] = counter;

        // groupshared ballot (replaces the CUDA __ballot_sync): bit laneId of this
        // warp's word = (isActive XOR wasActive).
        if (isActive != wasActive)
            InterlockedOr(gSleepMasks[warpId], 1u << laneId);
    }
    GroupMemoryBarrierWithGroupSync();

    // CUDA gates this store on `(startIdx + threadIdx.x) < ((numActive + 31) / 32)` — only
    // the words a warp with at least one live cloth actually filled.  The P7.0 port dropped
    // that guard, which writes zeros PAST THE END of stateChangedMask: it is a PxgBitMap
    // sized to the body count, so a single active cloth means one live word while all 32
    // lanes were storing.  Restored while porting the softbody twin (sb_sleeping), whose
    // unit test asserts the neighbouring words stay untouched.
    const uint startIdx = (groupId.x * CLOTH_STEP) / 32u; // = 32 * groupId.x
    const uint nbWords = (gSleep.numActiveCloths + 31u) / 32u;
    if (groupThreadId.x < CLOTH_STEP_NUM_WARPS && (startIdx + groupThreadId.x) < nbWords)
        stateChangedMask[startIdx + groupThreadId.x] = gSleepMasks[groupThreadId.x];
}
