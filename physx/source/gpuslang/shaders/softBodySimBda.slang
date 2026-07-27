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

// softBodySimBda — P8.0: the FIRST live SOFTBODY (FEM-volume) frame on the Vulkan
// engine ABI (BDA).  The softbody free-fall predict/integrate driver, ported
// line-for-line from gpusimulationcontroller/src/CUDA/softBodyGM.cu (the exact
// softbody analog of the completed FEMCloth free-fall in femClothSimBda.slang):
//   - sb_gm_preIntegrateLaunch (softBodyGM.cu:72): the free-fall driver.  Gravity
//     predict AND (on the PGS path) the position integration in one shot
//     (positions[i] = oldPos + vel*dt, the observable fall).  The full-kinematic
//     branch (mBodyFlags & eKINEMATIC), the partially-kinematic sub-branch
//     (mVolumeFlags & ePARTIALLY_KINEMATIC), the gravity/damping predict, the
//     max-velocity clamp and both the TGS and PGS position paths are ported
//     faithfully, preserving every null/flag branch and the thread mapping.
//
// Deviations from the CUDA original (recipe policy; all preserve the free-fall
// semantics and are exercised by testSoftBodySim.cpp):
//   - the descriptor arrives as a device pointer to an array of PxgSoftBody read
//     through Ptr<> (BDA), using the byte-exact pxgSlangSoftBodyDesc mirror;
//   - sb_gm_preIntegrateLaunch's ONLY warp op — the speculative-CCD reduction
//     (warpReduction<MaxOpFloat> + __syncthreads + AtomicMax on
//     speculativeCCDContactOffset[id]) — is OMITTED.  It is gated by
//     `mBodyFlags & PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD`, which is an
//     OPT-IN body flag (PxDeformableBody::setDeformableBodyFlag), UNSET on the
//     default free-fall scene, so the whole block (and the `__shared__
//     maxMagnitudeVelSq[32]` it feeds) is dead — exactly the position the cloth
//     precedent took for its dead adaptiveCollisionPairUpdate reduction.  Keeping
//     it out means this shader has NO groupshared and NO barriers, so there is no
//     subgroup lowering nor uniform-barrier-control-flow hazard on llvmpipe (whose
//     subgroup size is 8, not the CUDA 32).  `maxMagVelSq` (fed only by that block)
//     is likewise dropped; `speculativeCCDContactOffset` stays in the push struct
//     for ABI parity but is unread.
//   - the two per-vertex `bool*` byte stores (`vertsDeformed[i] = false;
//     vertsCanNotDeform[i] = false;`) are ELIDED.  A faithful store would need
//     8-bit storage on llvmpipe (same reason the cloth precedent elided its PxU8
//     `updateContactPairs` write); no MAPPED kernel consumes mVertsAreDeformed /
//     mVertsCanNotDeform on the free-fall path (only the un-ported
//     deformation-detection path does), so the elision is unobservable.  The
//     `softbody.mAwake = false` reset at globalThreadIndex==0 IS kept — it is a
//     PxU32 (4-byte) store, faithful on llvmpipe.
//   - the max-velocity clamp's `1/PxSqrt(velMagSq)` transcendental is DEAD on the
//     default scene: mMaxLinearVelocity == PX_MAX_REAL, so `maxVel != PX_MAX_REAL`
//     short-circuits false and the clamp (hence the sqrt) never feeds the output.
//     The predict/integrate path is thus bit-exact with no transcendental seam
//     (verified in testSoftBodySim.cpp).  The shader is NoContraction-stamped
//     (PXG_NOCONTRACT_SHADERS) so the gravity/damping mul+add chains match the
//     -ffp-contract=off CPU reference bit-for-bit.

import pxgSlangSoftBodyDesc;

static const uint WARP_SIZE = 32u;
// PxgSoftBodyKernelBlockDim::SB_PREINTEGRATION (PxgSoftBodyCoreKernelIndices.h:39).
static const uint SB_PREINTEGRATION = 1024u;
// PX_MAX_REAL (foundation/PxSimpleTypes) == FLT_MAX.
static const float PX_MAX_REAL = 3.40282347e+38f;

// ---------------------------------------------------------------------------
// sb_gm_preIntegrateLaunch — gravity predict + position integration (softBodyGM.cu:72)
// ---------------------------------------------------------------------------
// CUDA arg pack (PxgSoftBodyCore.cpp:165-174): { PxgSoftBody* softbodies, const
// PxU32* activeId, PxVec3 gravity(12), PxReal dt(4), bool isTGS(1), PxReal*
// speculativeCCDContactOffset, bool externalForcesEveryTgsIterationEnabled(1) } ->
// natural-alignment blob = 56 bytes.
struct PreIntegrateParams
{
    uint64_t softbodies;                     // @0
    uint64_t activeId;                       // @8
    float gravityX;                          // @16
    float gravityY;                          // @20
    float gravityZ;                          // @24
    float dt;                                // @28
    uint isTGSWord;                          // @32  byte0 = isTGS
    uint _pad36;                             // @36  (isTGS's tail-pad -> next ptr @40)
    uint64_t speculativeCCDContactOffset;    // @40  (ABI parity; unread — CCD dead)
    uint extForcesWord;                      // @48  byte0 = externalForcesEveryTgsIterationEnabled
    uint _pad52;                             // @52  -> 56
}
[[vk::push_constant]]
ConstantBuffer<PreIntegrateParams> gPre;

[shader("compute")]
[numthreads(32, 32, 1)] // (WARP_SIZE, SB_PREINTEGRATION/WARP_SIZE) == (32,32) = 1024
void sb_gm_preIntegrateLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgSoftBodyGpu> softbodies = Ptr<PxgSoftBodyGpu>(gPre.softbodies);
    const uint id = (Ptr<uint>(gPre.activeId))[groupId.y];

    const uint nbVerts = softbodies[id].mNumVertsGM;
    Ptr<float4> positions = Ptr<float4>(softbodies[id].mSimPosition_InvMass);
    Ptr<float4> velocities = Ptr<float4>(softbodies[id].mSimVelocity_InvMass);
    // vertsDeformed / vertsCanNotDeform bool-array byte stores are elided (header).
    Ptr<float4> posDelta = Ptr<float4>(softbodies[id].mSimDeltaPos);

    const uint threadIdxInWarp = groupThreadId.x;
    const uint warpIndex = groupThreadId.y;
    const uint globalThreadIndex =
        threadIdxInWarp + WARP_SIZE * warpIndex + SB_PREINTEGRATION * groupId.x;

    if (globalThreadIndex == 0u)
    {
        softbodies[id].mAwake = 0u; // reset each frame (bool false -> PxU32 store)
    }

    // The @392 fold: byte0 = mActorFlags, byte1 = mBodyFlags, bytes2..3 = mVolumeFlags.
    const uint flagsWord = softbodies[id].mFlagsWord;
    const uint actorFlags = flagsWord & 0xffu;
    const uint bodyFlags = (flagsWord >> 8) & 0xffu;
    const uint volumeFlags = (flagsWord >> 16) & 0xffffu;

    const bool isTGS = (gPre.isTGSWord & 0xffu) != 0u;
    const bool externalForcesEveryTgsIterationEnabled = (gPre.extForcesWord & 0xffu) != 0u;

    const float3 gravity = float3(gPre.gravityX, gPre.gravityY, gPre.gravityZ);

    if ((bodyFlags & PXG_SOFTBODY_BODYFLAG_KINEMATIC) != 0u)
    {
        // Full-kinematic softbody: snap every vertex to its kinematic target and
        // derive the velocity from the displacement.  CUDA guards on a non-null
        // mSimKinematicTarget.
        const uint64_t kinTargetAddr = softbodies[id].mSimKinematicTarget;
        if (globalThreadIndex < nbVerts && kinTargetAddr != 0u)
        {
            Ptr<float4> kinematicTargets = Ptr<float4>(kinTargetAddr);
            const float4 tPos = positions[globalThreadIndex];
            const float3 pos = float3(tPos.x, tPos.y, tPos.z);
            const float4 tKinematicPos = kinematicTargets[globalThreadIndex];
            const float3 kinematicPos = float3(tKinematicPos.x, tKinematicPos.y, tKinematicPos.z);

            const float3 delta = kinematicPos - pos;
            posDelta[globalThreadIndex] = float4(delta.x, delta.y, delta.z, tPos.w);

            positions[globalThreadIndex] = tKinematicPos;

            const float4 tVel = velocities[globalThreadIndex];
            float3 vel = float3(tVel.x, tVel.y, tVel.z);
            vel += delta / gPre.dt;

            velocities[globalThreadIndex] = float4(vel.x, vel.y, vel.z, 0.0f);

            // vertsDeformed[i] = false; vertsCanNotDeform[i] = false; -- elided (header)
        }
    }
    else
    {
        if (globalThreadIndex < nbVerts)
        {
            float4 tPos = positions[globalThreadIndex];
            float3 vel = float3(0.0f, 0.0f, 0.0f);
            const float3 oldPos = float3(tPos.x, tPos.y, tPos.z);

            bool kinematic = false;
            if ((volumeFlags & PXG_SOFTBODY_VOLUMEFLAG_PARTIALLY_KINEMATIC) != 0u)
            {
                const uint64_t kinTargetAddr = softbodies[id].mSimKinematicTarget;
                if (kinTargetAddr != 0u)
                {
                    Ptr<float4> kinematicTargets = Ptr<float4>(kinTargetAddr);
                    const float4 tKinematicPos = kinematicTargets[globalThreadIndex];
                    if (tKinematicPos.w == 0.0f)
                    {
                        const float3 kinematicPos = float3(tKinematicPos.x, tKinematicPos.y, tKinematicPos.z);
                        vel = (kinematicPos - oldPos) / gPre.dt;
                        kinematic = true;
                        tPos.w = 0.0f;
                    }
                }
            }

            if (tPos.w != 0.0f && !kinematic)
            {
                const float4 oldVel = velocities[globalThreadIndex];
                vel = float3(oldVel.x, oldVel.y, oldVel.z);
                if (!externalForcesEveryTgsIterationEnabled)
                {
                    if ((actorFlags & PXG_SOFTBODY_ACTORFLAG_DISABLE_GRAVITY) == 0u)
                    {
                        vel += gravity * gPre.dt;
                    }
                    const float damping = 1.0f - min(gPre.dt * softbodies[id].mLinearDamping, 1.0f);
                    vel *= damping;
                }
            }

            // Max velocity clamping.
            const float maxVel = softbodies[id].mMaxLinearVelocity;
            const float velMagSq = vel.x * vel.x + vel.y * vel.y + vel.z * vel.z;

            vel = (maxVel != PX_MAX_REAL && maxVel * maxVel < velMagSq)
                      ? (maxVel / sqrt(velMagSq)) * vel
                      : vel;

            float3 delta = float3(0.0f, 0.0f, 0.0f);
            if (!isTGS)
            {
                delta += vel * gPre.dt;
            }

            const float3 pos = oldPos + delta;
            // The positions are still moved when gravity is applied every substep so
            // that refit bounds of the bvh and the following collision detection do
            // not miss potential contacts towards the end of the time step.
            positions[globalThreadIndex] = float4(pos.x, pos.y, pos.z, tPos.w);

            velocities[globalThreadIndex] = float4(vel.x, vel.y, vel.z, tPos.w);

            posDelta[globalThreadIndex] = float4(delta.x, delta.y, delta.z, tPos.w);

            // vertsDeformed[i] = false; vertsCanNotDeform[i] = false; -- elided (header)
        }
    }

    // The speculative-CCD warp reduction (AtomicMax on speculativeCCDContactOffset)
    // is intentionally omitted — dead on the default free-fall scene (the
    // eENABLE_SPECULATIVE_CCD body flag is opt-in and unset).  See the header.
}


// ===========================================================================
// P8.0 tranche 3 — the rest of the softbody FREE-FALL chain.
//
// sb_gm_preIntegrateLaunch (above) predicts the velocity and moves the vertices once
// per step; sb_refitBoundLaunch (softBodyRefitBda.slang) republishes the bounds.  These
// three close the loop so a softbody can fall, settle and go to sleep:
//
//   sb_gm_stepSoftbodyLaunch      (softBodyGM.cu:232, launch PxgSoftBodyCore.cpp:2632)
//       the SUBSTEP integrator — advances position by vel*dt and accumulates the
//       per-substep delta into mSimDeltaPos, optionally re-applying gravity/damping
//       first when external forces run every TGS iteration.
//   sb_gm_finalizeVelocitiesLaunch (softBodyGM.cu:3136, launch PxgSoftBodyCore.cpp:920)
//       turns the accumulated position delta back into a velocity, applies the settling
//       damping / velocity averaging, and ORs the body's awake flag.
//   sb_sleeping                    (softBodyGM.cu:3219, launch PxgSoftBodyCore.cpp:948)
//       the per-body wake-counter bookkeeping + the state-changed bitmap.
//
// These are the softbody twins of the cloth trio already living in femClothSimBda.slang
// (cloth_preIntegrate / cloth_finalizeVelocities / cloth_sleeping) and reuse its warp
// lowerings verbatim.  Softbody-specific differences, all straight from the .cu:
//   - every buffer is the SIMULATION mesh (mSimPosition_InvMass @96, mSimVelocity_InvMass
//     @88, mSimDeltaPos @128, mNumVertsGM @372) — not the collision mesh the refit walks;
//   - the awake flag is mAwake @408 (PxU32), not the cloth's mIsActive;
//   - finalize takes an explicit `velocityScale` and blends
//     `tVel*velocityScale + deltaVel*(1-velocityScale)`, where the cloth kernel rescales
//     by a sqrt ratio instead.  Different formulas — do NOT copy the cloth line here.
//
// DEVIATIONS from CUDA:
//   - __any_sync / __ballot_sync + the two-level shared-memory reduction become
//     groupshared InterlockedOr (order-independent, so bit-identical): a block-wide OR
//     for finalize's awake flag, and one 32-bit word per warp for sleeping's ballot;
//   - sb_sleeping's CUDA `if (globalThreadIdx >= numActiveSoftBodies) return;` sits
//     BEFORE a __syncthreads (UB in a partly filled block).  Restructured into an
//     `if (gid < n)` guard so every thread reaches both barriers.  Same stores;
//   - sb_gm_stepSoftbodyLaunch's commented-out `if (tPos.w != 0.f)` guard around the
//     velocity load is reproduced as CUDA leaves it — INACTIVE (the .cu comment says it
//     "doesn't look right and break[s] the collision of kinematic softbodies").
//
// PxSqrt in the max-velocity clamp is IEEE-754 correctly-rounded (unlike rsqrt / the
// trig family), so it stays bit-exact and needs no tolerance seam.  Every mul feeding an
// add/sub (gravity*dt into vel, vel*dt into pos and posDelta, the settle tolerances, the
// velocity blend) is NoContraction-stamped via PXG_NOCONTRACT_SHADERS.
// ===========================================================================

static const uint SB_STEP_NUM_WARPS = SB_PREINTEGRATION / WARP_SIZE;  // 32

// ---------------------------------------------------------------------------
// sb_gm_stepSoftbodyLaunch (softBodyGM.cu:232)
// ---------------------------------------------------------------------------
// CUDA arg pack (PxgSoftBodyCore.cpp:2637): { const PxgSoftBody* softbodies,
// const PxU32* activeId, const PxReal dt, const PxVec3 gravity(12),
// const bool externalForcesEveryTgsIterationEnabled(1) } -> PxVec3 has 4-byte
// alignment, so dt @16, gravity @20..32, the bool @32 -> 33, rounded up to the
// pointer alignment -> 40-byte blob.  (ABI row {8,8,4,12,1}.)
struct StepParams
{
    uint64_t softbodies;     // @0
    uint64_t activeId;       // @8
    float dt;                // @16
    float gravityX;          // @20
    float gravityY;          // @24
    float gravityZ;          // @28
    uint extForcesWord;      // @32  byte0 = externalForcesEveryTgsIterationEnabled
    uint _pad36;             // @36  -> 40
}
[[vk::push_constant]]
ConstantBuffer<StepParams> gStep;

[shader("compute")]
[numthreads(1024, 1, 1)] // SB_PREINTEGRATION; grid (nbBlocks, nbActiveSoftbodies, 1)
void sb_gm_stepSoftbodyLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgSoftBodyGpu> softbodies = Ptr<PxgSoftBodyGpu>(gStep.softbodies);
    const uint id = (Ptr<uint>(gStep.activeId))[groupId.y];

    const uint nbVerts = softbodies[id].mNumVertsGM;
    Ptr<float4> velocities = Ptr<float4>(softbodies[id].mSimVelocity_InvMass);
    Ptr<float4> positions = Ptr<float4>(softbodies[id].mSimPosition_InvMass);
    Ptr<float4> posDelta = Ptr<float4>(softbodies[id].mSimDeltaPos);

    const uint globalThreadIndex = groupThreadId.x + SB_PREINTEGRATION * groupId.x;

    const bool externalForcesEveryTgsIterationEnabled = (gStep.extForcesWord & 0xffu) != 0u;
    const float3 gravity = float3(gStep.gravityX, gStep.gravityY, gStep.gravityZ);
    const float dt = gStep.dt;

    if (globalThreadIndex < nbVerts)
    {
        const float4 tPos = positions[globalThreadIndex];
        const float4 pDelta = posDelta[globalThreadIndex];
        const float4 v = velocities[globalThreadIndex];
        const float3 oldPos = float3(tPos.x, tPos.y, tPos.z);
        // CUDA guards this load with a COMMENTED-OUT `if (tPos.w != 0.f)`; reproduced
        // as-is (inactive), so every vertex — kinematic included — picks up its velocity.
        float3 vel = float3(v.x, v.y, v.z);

        if (externalForcesEveryTgsIterationEnabled)
        {
            if (tPos.w != 0.0f)
            {
                // The @392 fold: byte0 = mActorFlags.
                const uint actorFlags = softbodies[id].mFlagsWord & 0xffu;
                if ((actorFlags & PXG_SOFTBODY_ACTORFLAG_DISABLE_GRAVITY) == 0u)
                {
                    vel += gravity * dt;
                }
                const float damping = 1.0f - min(dt * softbodies[id].mLinearDamping, 1.0f);
                vel *= damping;
            }

            // Max velocity clamping
            const float maxVel = softbodies[id].mMaxLinearVelocity;
            const float velMagSq = vel.x * vel.x + vel.y * vel.y + vel.z * vel.z;
            vel = (maxVel != PX_MAX_REAL && maxVel * maxVel < velMagSq)
                      ? (maxVel / sqrt(velMagSq)) * vel
                      : vel;

            // NOTE: w comes from the OLD VELOCITY here, not from tPos (CUDA writes v.w).
            velocities[globalThreadIndex] = float4(vel.x, vel.y, vel.z, v.w);
        }

        const float3 delta = vel * dt;

        const float3 pos = oldPos + delta;
        positions[globalThreadIndex] = float4(pos.x, pos.y, pos.z, tPos.w);
        posDelta[globalThreadIndex] =
            float4(pDelta.x + delta.x, pDelta.y + delta.y, pDelta.z + delta.z, tPos.w);
    }
}

// ---------------------------------------------------------------------------
// sb_gm_finalizeVelocitiesLaunch (softBodyGM.cu:3136)
// ---------------------------------------------------------------------------
// CUDA arg pack (PxgSoftBodyCore.cpp:922): { PxgSoftBody* gSoftbodies,
// const PxU32* activeSoftbodies, const PxReal invDt, const PxReal velocityScale,
// const PxReal dt, const bool alwaysRunVelocityAveraging(1) } -> 29, rounded up to the
// pointer alignment -> 32-byte blob.  (ABI row {8,8,4,4,4,1}.)
struct FinalizeParams
{
    uint64_t gSoftbodies;      // @0
    uint64_t activeSoftbodies; // @8
    float invDt;               // @16
    float velocityScale;       // @20
    float dt;                  // @24
    uint alwaysAvgWord;        // @28  byte0 = alwaysRunVelocityAveraging  -> 32
}
[[vk::push_constant]]
ConstantBuffer<FinalizeParams> gFin;

groupshared uint gSbAnyAwake;

[shader("compute")]
[numthreads(1024, 1, 1)] // SB_PREINTEGRATION; grid (nbBlocks, nbActiveSoftbodies, 1)
void sb_gm_finalizeVelocitiesLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgSoftBodyGpu> softbodies = Ptr<PxgSoftBodyGpu>(gFin.gSoftbodies);
    const uint softbodyId = (Ptr<uint>(gFin.activeSoftbodies))[groupId.y];

    Ptr<float4> deltaPos = Ptr<float4>(softbodies[softbodyId].mSimDeltaPos);
    Ptr<float4> vels = Ptr<float4>(softbodies[softbodyId].mSimVelocity_InvMass);
    Ptr<float4> positions = Ptr<float4>(softbodies[softbodyId].mSimPosition_InvMass);

    const uint vertIdx = groupThreadId.x + groupId.x * SB_PREINTEGRATION;
    const uint nbVerts = softbodies[softbodyId].mNumVertsGM;

    const float dt = gFin.dt;
    const float invDt = gFin.invDt;
    const float velocityScale = gFin.velocityScale;
    const bool alwaysRunVelocityAveraging = (gFin.alwaysAvgWord & 0xffu) != 0u;

    if (groupThreadId.x == 0u)
        gSbAnyAwake = 0u;
    GroupMemoryBarrierWithGroupSync();

    bool awake = false;

    if (vertIdx < nbVerts)
    {
        const float settleTolerance = softbodies[softbodyId].mSettlingThreshold * dt;
        const float tolerance = softbodies[softbodyId].mSleepThreshold * dt;
        const float sleepDamping = 1.0f - min(1.0f, softbodies[softbodyId].mSettlingDamping * dt);

        const float4 delta = deltaPos[vertIdx];
        const float3 tDelta = float3(delta.x, delta.y, delta.z);
        const float3 deltaVel = tDelta * invDt;
        const float4 pos = positions[vertIdx];

        if (pos.w != 0.0f) // skip kinematic and infinite mass particles
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
                const float4 vel = vels[vertIdx];
                float3 tVel = float3(vel.x, vel.y, vel.z);

                const float deltaVelMagSq =
                    deltaVel.x * deltaVel.x + deltaVel.y * deltaVel.y + deltaVel.z * deltaVel.z;
                const float velMagSq = tVel.x * tVel.x + tVel.y * tVel.y + tVel.z * tVel.z;
                if (alwaysRunVelocityAveraging && deltaVelMagSq < velMagSq)
                    tVel = tVel * velocityScale + deltaVel * (1.0f - velocityScale);

                vels[vertIdx] = float4(velocityScaling * tVel.x, velocityScaling * tVel.y,
                                       velocityScaling * tVel.z, vel.w);
            }
        }
    }

    // Block-wide OR of `awake` (replaces the CUDA two-level __any_sync + shared isAwake[]).
    if (awake)
        InterlockedOr(gSbAnyAwake, 1u);
    GroupMemoryBarrierWithGroupSync();

    if (groupThreadId.x == 0u && gSbAnyAwake != 0u)
        InterlockedOr(softbodies[softbodyId].mAwake, 1u); // BDA device atomicOr
}

// ---------------------------------------------------------------------------
// sb_sleeping (softBodyGM.cu:3219)
// ---------------------------------------------------------------------------
// CUDA arg pack (PxgSoftBodyCore.cpp:955): { PxgSoftBody* softBodies,
// const PxU32 numActiveSoftBodies, const PxU32* activeId, const PxReal dt,
// const PxReal resetCounter, PxReal* wakeCounters, PxU32* stateChangedMask }
// -> 48-byte blob.  (ABI row {8,4,8,4,4,8,8}.)
struct SleepParams
{
    uint64_t softBodies;         // @0
    uint numActiveSoftBodies;    // @8
    uint _pad12;                 // @12
    uint64_t activeId;           // @16
    float dt;                    // @24
    float resetCounter;          // @28
    uint64_t wakeCounters;       // @32
    uint64_t stateChangedMask;   // @40  -> 48
}
[[vk::push_constant]]
ConstantBuffer<SleepParams> gSleep;

groupshared uint gSbSleepMasks[SB_STEP_NUM_WARPS]; // 32

[shader("compute")]
[numthreads(1024, 1, 1)] // SB_PREINTEGRATION; grid (nbBlocks, 1, 1)
void sb_sleeping(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgSoftBodyGpu> softBodies = Ptr<PxgSoftBodyGpu>(gSleep.softBodies);
    Ptr<uint> activeId = Ptr<uint>(gSleep.activeId);
    Ptr<float> wakeCounters = Ptr<float>(gSleep.wakeCounters);
    Ptr<uint> stateChangedMask = Ptr<uint>(gSleep.stateChangedMask);

    const uint globalThreadIdx = groupThreadId.x + groupId.x * SB_PREINTEGRATION;
    const uint laneId = groupThreadId.x & 31u;
    const uint warpId = groupThreadId.x >> 5;

    // Restructured from the CUDA early-return + __syncthreads (UB in a partly filled
    // block) into an `if (gid < n)` guard so all threads reach both barriers.
    if (groupThreadId.x < SB_STEP_NUM_WARPS)
        gSbSleepMasks[groupThreadId.x] = 0u;
    GroupMemoryBarrierWithGroupSync();

    if (globalThreadIdx < gSleep.numActiveSoftBodies)
    {
        const uint id = activeId[globalThreadIdx];

        const bool reset = softBodies[id].mAwake != 0u;
        float counter = wakeCounters[id];
        const bool wasActive = counter > 0.0f;

        if (reset)
            counter = gSleep.resetCounter;
        else
            counter = max(0.0f, counter - gSleep.dt);

        const bool isActive = counter > 0.0f;

        wakeCounters[id] = counter;

        // groupshared ballot (replaces __ballot_sync): bit laneId of this warp's word
        // = (isActive XOR wasActive).
        if (isActive != wasActive)
            InterlockedOr(gSbSleepMasks[warpId], 1u << laneId);
    }
    GroupMemoryBarrierWithGroupSync();

    // CUDA gates the write on `(startIdx + threadIdx.x) < ((numActive + 31) / 32)` —
    // i.e. only the words that a warp with at least one live body actually filled.
    // The guard is load-bearing: stateChangedMask is a PxgBitMap sized for the body
    // count, so dropping it writes zeros past the end of the map.
    const uint startIdx = (groupId.x * SB_PREINTEGRATION) / 32u;
    const uint nbWords = (gSleep.numActiveSoftBodies + 31u) / 32u;
    if (groupThreadId.x < SB_STEP_NUM_WARPS && (startIdx + groupThreadId.x) < nbWords)
        stateChangedMask[startIdx + groupThreadId.x] = gSbSleepMasks[groupThreadId.x];
}
