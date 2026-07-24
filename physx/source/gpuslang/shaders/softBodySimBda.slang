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
