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

// psContactSolveTgsBda -- the two TGS-only particle kernels of
// gpusimulationcontroller/src/CUDA/particlesystem.cu, both textually unchanged in
// PhysX 5.12:
//
//   1. ps_stepParticlesLaunch -- the TGS substep PREDICTOR run at the top of each
//      position iteration: advances each particle's SORTED position by vel*dt (from the
//      sorted origin on the first iteration, else from the running sorted position +
//      running deltaP), and accumulates the running per-particle deltaP.  Block
//      PS_SOLVE=256, grid (ceil(mMaxParticles/256), nbActiveParticleSystems, 1).
//   2. ps_finalizeParticlesLaunch -- the TGS velocity-iteration FINALIZE: blends the
//      accumulated position delta (as a velocity deltaP*invTotalDt) into the sorted
//      velocity when the current velocity is larger.  Block PS_SOLVE=256, grid
//      (ceil(numParticles/256),1,1).
//
// 5.12 deleted ps_solvePCOutputParticleDeltaVTGSLaunch, the TGS twin of the particle
// contact solve: ps_solvePCOutputParticleDeltaVLaunch (psContactSolveBda) takes isTGS.
// Its port, with the solvePCOutputDeltaVTGS helper and the 5.6.1
// PxgParticlePrimitiveConstraintBlock offsets, is removed from this file.
//
// DEVIATIONS (flagged):
//   - ps_stepParticlesLaunch indexes particleSystems[blockIdx.y] DIRECTLY (the
//     activeParticleSystems* arg is UNUSED by the kernel body -- ABI parity only).
//   - The CUDA block-copies the whole PxgParticleSystem into shared memory; here every
//     field is read directly through Ptr<> (read-only descriptor).
//   - The PxU16 phase->material handle and the runtime-strided PxsParticleMaterialData
//     are read as 32-bit word + byte offset (no 16-bit storage capability needed).
//   - llvmpipe SAFETY: neither kernel uses a ballot/shuffle/warp-scan or a group
//     barrier, so the divergent early guard is safe.

import pxgSlangParticleDesc;   // PxgParticleSystemGpu, PXG_PARTICLE_PHASE_GROUP_MASK

// PxgParticleSystemKernelBlockDim / GridDim (PxgParticleSystemCoreKernelIndices.h)
static const uint PS_SOLVE_BLOCK = 256u;  // PS_SOLVE block (step + finalize)

// PxsParticleMaterialData float indices (PxParticleGpu.h): damping @1, gravityScale @3.
static const uint MAT_DAMPING = 1u;
static const uint MAT_GRAVITYSCALE = 3u;

// ---------------------------------------------------------------------------
// Shared helpers (transcribed; see file header for provenance)
// ---------------------------------------------------------------------------
float pmin(float a, float b) { return (a < b) ? a : b; }   // PxMin = selectMin

// phaseToMat[group] where phaseToMat is a PxU16 array (32-bit load + shift/mask).
uint loadPhaseToMaterial(uint64_t handleBase, uint group)
{
    let word = (Ptr<uint>(handleBase))[group >> 1];
    return (word >> ((group & 1u) * 16u)) & 0xffffu;
}

// getParticleMaterial (PxgParticleSystem.h:64) — raw byte-offset deref with the
// runtime material stride.  Floats: friction@0, damping@1, adhesion@2,
// gravityScale@3, adhesionRadiusScale@4.
float loadMaterialFloat(uint64_t materialsBase, uint mi, uint stride, uint floatIndex)
{
    let elem = materialsBase + uint64_t(mi) * uint64_t(stride);
    return (Ptr<float>(elem))[floatIndex];
}

// ===========================================================================
// 1. ps_stepParticlesLaunch (particlesystem.cu:120)
//    CUDA ABI (PxgParticleSystemCore.cpp:2460): (const PxgParticleSystem*
//      particleSystems, const PxU32* activeParticleSystems, const PxReal dt,
//      const PxReal invTotalDt, bool isFirstIteration, const bool isTGS,
//      const bool externalForcesEveryTgsIterationEnabled, const PxVec3 gravity)
//    -> {8,8,4,4,1,1,1,12} = 40-byte blob (three bool bytes @24/25/26, PxVec3
//    gravity @28 after the 4-align).  activeParticleSystems is ABI parity only.
// ===========================================================================
struct PcStepParams
{
    uint64_t particleSystems;       // @0
    uint64_t activeParticleSystems; // @8  (ABI parity; unused)
    float dt;                       // @16
    float invTotalDt;               // @20 (unused — CUDA totalDelta clamp is commented out)
    uint flagsWord;                 // @24 byte0=isFirstIteration, byte1=isTGS, byte2=extForces
    float gravityX;                 // @28
    float gravityY;                 // @32
    float gravityZ;                 // @36 -> 40
}
[[vk::push_constant]]
ConstantBuffer<PcStepParams> gSp;

[shader("compute")]
[numthreads(PS_SOLVE_BLOCK, 1, 1)]
void ps_stepParticlesLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gSp.particleSystems);
    // The .cu indexes particleSystems[blockIdx.y] DIRECTLY (activeParticleSystems
    // is unused); grid Y = nbActiveParticleSystems.
    const uint sysIndex = groupId.y;

    const uint numParticles = systems[sysIndex].mCommonData_NumParticles;
    const uint globalThreadIndex = groupThreadId.x + groupId.x * PS_SOLVE_BLOCK;

    const bool isFirstIteration = (gSp.flagsWord & 0xffu) != 0u;
    // isTGS (byte1) is only PX_ASSERT'd in the .cu; extForces (byte2) gates the
    // per-iteration external-force integration.
    const bool externalForcesEveryTgsIterationEnabled = ((gSp.flagsWord >> 16) & 0xffu) != 0u;

    const uint64_t srcPositions = isFirstIteration ? systems[sysIndex].mSortedOriginPos_InvMass
                                                    : systems[sysIndex].mSortedPositions_InvMass;
    Ptr<float4> positions = Ptr<float4>(systems[sysIndex].mSortedPositions_InvMass);
    Ptr<float4> vels = Ptr<float4>(systems[sysIndex].mSortedVelocities);
    Ptr<float4> deltaP = Ptr<float4>(systems[sysIndex].mSortedDeltaP);
    Ptr<uint> phases = Ptr<uint>(systems[sysIndex].mSortedPhaseArray);

    const uint64_t phaseToMat = systems[sysIndex].mPhaseGroupToMaterialHandle;
    const uint64_t materials = systems[sysIndex].mParticleMaterials;
    const uint materialStride = systems[sysIndex].mParticleMaterialStride;

    if (globalThreadIndex < numParticles)
    {
        const uint phase = phases[globalThreadIndex];
        const uint group = phase & PXG_PARTICLE_PHASE_GROUP_MASK;
        const uint mi = loadPhaseToMaterial(phaseToMat, group);

        const float cflVel = systems[sysIndex].mData_MaxVelocity;

        const float4 tPos = (Ptr<float4>(srcPositions))[globalThreadIndex];
        if (tPos.w != 0.0f)
        {
            float3 vel = vels[globalThreadIndex].xyz;

            bool velChanged = false;
            if (externalForcesEveryTgsIterationEnabled)
            {
                const float matDamping = loadMaterialFloat(materials, mi, materialStride, MAT_DAMPING);
                const float matGravityScale = loadMaterialFloat(materials, mi, materialStride, MAT_GRAVITYSCALE);
                const float damping = 1.0f - pmin(1.0f, matDamping * gSp.dt);
                const float3 gravity = float3(gSp.gravityX, gSp.gravityY, gSp.gravityZ);
                vel = (vel + gravity * matGravityScale * gSp.dt) * damping;
                velChanged = true;
            }

            const float velLimit = cflVel;
            if (dot(vel, vel) > velLimit * velLimit)
            {
                // PxVec3::getNormalized() = *this * (1/magnitude); then * velLimit.
                const float m = sqrt(dot(vel, vel));
                const float invM = 1.0f / m;
                vel = (vel * invM) * velLimit;
                velChanged = true;
            }

            const float3 dlta = isFirstIteration ? float3(0.0f, 0.0f, 0.0f)
                                                 : deltaP[globalThreadIndex].xyz;

            const float3 oldPos = float3(tPos.x, tPos.y, tPos.z);
            const float3 delta = vel * gSp.dt;
            const float3 pos = oldPos + delta;
            // P9.86f: on the first iteration dlta is the float3(0) select arm on the runtime isFirstIteration --
            // not folded by llvmpipe (p86SignedZero rows 23-25), so this is IEEE `delta + 0` as particlesystem.cu:187.
            const float3 totalDelta = delta + dlta;

            if (velChanged)
                vels[globalThreadIndex] = float4(vel.x, vel.y, vel.z, tPos.w);

            positions[globalThreadIndex] = float4(pos.x, pos.y, pos.z, tPos.w);
            deltaP[globalThreadIndex] = float4(totalDelta.x, totalDelta.y, totalDelta.z, tPos.w);
        }
    }
}

// ===========================================================================
// 3. ps_finalizeParticlesLaunch (particlesystem.cu:5731)
//    CUDA ABI (PxgPBDParticleSystemCore.cpp:498): (const PxgParticleSystem*
//      particleSystems, const PxU32 id, const PxReal invTotalDt, PxReal velocityScale)
//    -> {8,4,4,4} = 20 data bytes, rounded to the 8-alignment -> 24-byte blob.
// ===========================================================================
struct PcFinalizeParams
{
    uint64_t particleSystems; // @0
    uint id;                  // @8
    float invTotalDt;         // @12
    float velocityScale;      // @16 -> 20 (blob rounds to 24 on the 8-align)
}
[[vk::push_constant]]
ConstantBuffer<PcFinalizeParams> gFin;

[shader("compute")]
[numthreads(PS_SOLVE_BLOCK, 1, 1)]
void ps_finalizeParticlesLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gFin.particleSystems);
    const uint sysIndex = gFin.id;

    const uint numParticles = systems[sysIndex].mCommonData_NumParticles;
    const uint globalThreadIndex = groupThreadId.x + groupId.x * PS_SOLVE_BLOCK;

    Ptr<float4> vels = Ptr<float4>(systems[sysIndex].mSortedVelocities);
    Ptr<float4> deltaP = Ptr<float4>(systems[sysIndex].mSortedDeltaP);

    if (globalThreadIndex < numParticles)
    {
        const float4 tVel = vels[globalThreadIndex];
        if (tVel.w != 0.0f)
        {
            const float4 dlta = deltaP[globalThreadIndex];
            const float3 deltaVel = float3(dlta.x * gFin.invTotalDt, dlta.y * gFin.invTotalDt, dlta.z * gFin.invTotalDt);
            const float3 vel = float3(tVel.x, tVel.y, tVel.z);

            // Keep the position-inferred velocity unless the stored velocity is
            // larger (momentum conservation; avoids injecting energy — mirrors the
            // FEMCloth finalize).
            if (dot(vel, vel) > dot(deltaVel, deltaVel))
            {
                const float s = gFin.velocityScale;
                const float3 newVel = vel * s + (deltaVel * (1.0f - s));
                vels[globalThreadIndex] = float4(newVel.x, newVel.y, newVel.z, tVel.w);
            }
        }
    }
}
