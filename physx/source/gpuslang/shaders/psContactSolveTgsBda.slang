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

// psContactSolveTgsBda — P6.6: the THREE TGS-only kernels that complete the PBD
// particle/primitive contact SOLVE loop under the TGS solver, so the settling bag
// (P6.5 money shot, already SETTLING under PGS) ALSO settles under TGS instead of
// tunnelling.  Ported line-for-line from
// gpusimulationcontroller/src/CUDA/particlesystem.cu:
//
//   1. ps_stepParticlesLaunch (particlesystem.cu:120) — the TGS substep PREDICTOR
//      run at the top of each non-velocity solver iteration: advances each
//      particle's SORTED position by vel*dt (from the sorted-origin on the first
//      iteration, else from the running sorted position + running deltaP), and
//      accumulates the running per-particle deltaP.  Block PS_SOLVE=256, grid
//      (ceil(mMaxParticles/256), nbActiveParticleSystems, 1); launch
//      PxgParticleSystemCore.cpp:2460 (PS_STEP_PARTICLES).
//   2. ps_solvePCOutputParticleDeltaVTGSLaunch (particlesystem.cu:1690) — the TGS
//      twin of the already-mapped non-TGS ps_solvePCOutputParticleDeltaVLaunch
//      (particlesystem.cu:1531, psContactSolveBda.slang).  Same push pack (96
//      bytes), same constraint-block reads; the DIFFERENCE is the velocity read
//      (PxgVelocityReader::readVelocitiesTGS, which unpacks lin/ang VELOCITY AND
//      lin/ang DELTA from the 3-float4-per-body TGS iter pool) and the TGS solve
//      helper (solvePCOutputDeltaVTGS: folds normalVel*projectionDt +
//      normalDelta / tanDelta into the separation, projectionDt = velIter?0:dt).
//      Block PS_COLLISION=256, grid PS_COLLISION=1024; launch
//      PxgParticleSystemCore.cpp:1947 (isTGS ? PS_SOLVE_PC_PARTICLE_TGS : ...).
//   3. ps_finalizeParticlesLaunch (particlesystem.cu:5731) — the TGS velocity-
//      iteration FINALIZE: blends the accumulated position delta (as a velocity
//      deltaP*invTotalDt) into the sorted velocity when the current velocity is
//      larger, to conserve momentum after the TGS velocity iteration.  Block
//      PS_SOLVE=256, grid (ceil(numParticles/256),1,1); launch
//      PxgPBDParticleSystemCore.cpp:498 (PS_FINALIZE_PARTICLES).
//
// The non-TGS entrypoints (ps_contactPrepareLaunch / the non-TGS particle solve /
// the deltaV accumulate) stay in psContactSolveBda.slang, byte-untouched.  The
// contact PREPARE kernel is shared (isTGS-agnostic); the TGS solve consumes the
// SAME PxgParticlePrimitiveConstraintBlock ps_contactPrepareLaunch produced.
//
// Reused mirrors (verified against the engine headers; NO new descriptor fields
// were required — every field these three kernels touch already exists):
//   - PxgParticleSystemGpu (pxgSlangParticleDesc): mSortedOriginPos_InvMass @392,
//     mSortedPositions_InvMass @24, mSortedVelocities @32, mSortedDeltaP @160,
//     mSortedPhaseArray @40, mPhaseGroupToMaterialHandle @640, mParticleMaterials
//     @80, mParticleMaterialStride @632, mData_MaxVelocity @316, mData_RestOffset
//     @268, mCommonData_NumParticles @116 — all pre-existing.
//   - PxgSolverCoreDescGpu (@268 numSolverBodies, @276 accumulatedBodyDeltaVOffset)
//     + PxgSolverSharedDescPGSGpu (@160 iter_solverBodyVelPool) from
//     pxgSlangSolverDesc; PxgPrePrepDescGpu (@272 solverBodyIndices) from
//     pxgSlangPrepDesc — the SAME mirrors the non-TGS solve reuses.
//   - The 3072-byte PxgParticlePrimitiveConstraintBlock SoA byte offsets (CB_*)
//     and the material/phase byte helpers are transcribed identically to
//     psContactSolveBda.slang.
//
// PxNodeIndex decode (PxNodeIndex.h): rigidId is a PxU64 = { PxU32 mID (low), PxU32
// mLinkID (high) }.  index() = mID = rigidId.x; isStaticBody() = mID==0xffffffff;
// isArticulation() = mLinkID & 1 = rigidId.y & 1.
//
// CUDA DEVIATIONS (flagged):
//   - ARTICULATION branch DEFERRED (identical to the non-TGS port).  The CUDA TGS
//     solve builds a PxgVelocityReader whose CONSTRUCTOR reads artiCoreDesc->
//     {mMaxLinksPerArticulation, nbArticulations} and outArtiVelocity, then
//     branches per-lane on rigidId.isArticulation().  Here those articulation
//     reads are SUNK into the (dead) isArticulation branch, so a rigid/static
//     contact NEVER dereferences artiCoreDesc / outArtiVelocity — identical
//     observable result for the particle-vs-static-floor scenes (which generate
//     NO articulation contacts), and no BDA load is hoisted out of the guard.
//     artiCoreDesc is still carried in the push pack for ABI parity.  The rigid-
//     side TGS twin (ps_solvePCOutputRigidDeltaVTGSLaunch) stays DEFERRED: it is
//     inert for a static floor and refuses harmlessly, exactly like the non-TGS
//     rigid side under PGS.
//   - ps_stepParticlesLaunch indexes particleSystems[blockIdx.y] DIRECTLY (the
//     activeParticleSystems* arg is UNUSED by the kernel body — ABI parity only,
//     carried in the push pack; the .cu never dereferences it).
//   - The CUDA block-copies the whole PxgParticleSystem into shared memory in some
//     particle kernels (a read-fanout optimisation); here every field is read
//     directly through Ptr<> (semantically identical, read-only descriptor).
//   - The PxU16 phase->material handle and the runtime-strided PxsParticleMaterialData
//     are read as 32-bit word + byte offset (no 16-bit storage capability needed),
//     as particleSim / psContactSolveBda.
//   - llvmpipe SAFETY: none of the three kernels use a ballot/shuffle/warp-scan or
//     a group barrier (the step/finalize are per-particle, the TGS solve is
//     per-contact with a plain SoA indexed by workIndex&31), so the divergent
//     early `return` on the uniform-scalar guard (globalThreadIndex/workIndex >=
//     count) is safe and nothing speculatively hoists a BDA Ptr<> load out of a
//     zero-trip grid-stride loop.

import pxgSlangParticleDesc;   // PxgParticleSystemGpu, PXG_PARTICLE_PHASE_GROUP_MASK
import pxgSlangPrepDesc;       // PxgPrePrepDescGpu
import pxgSlangSolverDesc;     // PxgSolverCoreDescGpu, PxgSolverSharedDescPGSGpu

// PxgParticleSystemKernelBlockDim / GridDim (PxgParticleSystemCoreKernelIndices.h)
static const uint PS_SOLVE_BLOCK = 256u;  // PS_SOLVE block (step + finalize)
static const uint PC_BLOCK = 256u;        // PS_COLLISION block (TGS solve)
static const uint PC_GRID = 1024u;        // PS_COLLISION grid  (TGS solve)
static const uint STATIC_NODE = 0xffffffffu;  // PX_INVALID_NODE

// PxsParticleMaterialData float indices (PxParticleGpu.h:47): friction @0,
// damping @1, adhesion @2, gravityScale @3, adhesionRadiusScale @4.
static const uint MAT_FRICTION = 0u;
static const uint MAT_DAMPING = 1u;
static const uint MAT_ADHESION = 2u;
static const uint MAT_GRAVITYSCALE = 3u;
static const uint MAT_ADHESION_RADIUS_SCALE = 4u;

// PxgParticlePrimitiveConstraintBlock (PxgParticleSystemCore.h, 3072 bytes) — a
// SoA block of 32 lanes; per-field byte offsets within the block (identical to
// psContactSolveBda.slang).
static const uint CB_STRIDE = 3072u;
static const uint CB_RAXN_VM = 0u;      // float4 raXn_velMultiplierW[32]
static const uint CB_NORMAL_ERR = 512u; // float4 normal_errorW[32]
static const uint CB_FRIC0_IM0 = 1024u; // float4 fricTan0_invMass0[32]
static const uint CB_RAXNF0_VM = 1536u; // float4 raXnF0_velMultiplierW[32]
static const uint CB_RAXNF1_VM = 2048u; // float4 raXnF1_velMultiplierW[32]
static const uint CB_RIGIDID = 2560u;   // PxU64 rigidId[32]
static const uint CB_PARTICLEID = 2816u;// PxU64 particleId[32]

// ---------------------------------------------------------------------------
// Shared helpers (transcribed; see file header for provenance)
// ---------------------------------------------------------------------------
float pmin(float a, float b) { return (a < b) ? a : b; }   // PxMin = selectMin
float pmax(float a, float b) { return (a > b) ? a : b; }   // PxMax = selectMax
float quart(float x) { return x * x * x * x; }

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
            const float3 totalDelta = delta + dlta;

            if (velChanged)
                vels[globalThreadIndex] = float4(vel.x, vel.y, vel.z, tPos.w);

            positions[globalThreadIndex] = float4(pos.x, pos.y, pos.z, tPos.w);
            deltaP[globalThreadIndex] = float4(totalDelta.x, totalDelta.y, totalDelta.z, tPos.w);
        }
    }
}

// ===========================================================================
// 2. ps_solvePCOutputParticleDeltaVTGSLaunch (particlesystem.cu:1690)
//    CUDA ABI (PxgParticleSystemCore.cpp:1947): IDENTICAL push pack to the non-TGS
//    ps_solvePCOutputParticleDeltaVLaunch (the launch site packs the SAME array;
//    only the CUfunction differs):
//      (PxgParticleSystem* particleSystems, PxgParticlePrimitiveContact* sortedContacts,
//       PxgParticlePrimitiveConstraintBlock* primitiveConstraints, PxU32* numContacts,
//       PxgPrePrepDesc* prePrepDesc, PxgSolverCoreDesc* solverCoreDesc,
//       PxgSolverSharedDesc<IterativeSolveData>* sharedDesc, float4* deltaPos,
//       float2* appliedForces, PxReal dt, PxReal relaxationCoefficient,
//       bool isVelocityIteration, PxgArticulationCoreDesc* artiCoreDesc)
//    -> {8,8,8,8,8,8,8,8,8,4,4,1,8} = 96-byte blob (dt@72, coeff@76, bool@80,
//    artiCoreDesc@88).  sortedContacts is ABI parity only (particleId is read from
//    the constraint block, not the sorted contact).
// ===========================================================================

// solvePCOutputDeltaVTGS (particlesystem.cu:1473) — the TGS particle-contact solve
// core.  DIFF vs the non-TGS solvePCOutputDeltaV (particlesystem.cu:1421):
//   - projectionDt = isVelocityIteration ? 0 : dt;
//   - separation folds the running position DELTA and the projected velocity in:
//       separation = (error - normal.dot(delta) + normalDelta) + normalVel*projectionDt
//     (the non-TGS form is separation = error - normal.dot(delta), with the
//      -normalVel*dt applied inside deltaF instead);
//   - deltaF = max(clamp, -separation * velMultiplier)  (normalVel already folded);
//   - maxFriction uses |appliedForce.x| (appliedForce.x >= 0, so abs is a no-op);
//   - the tangential bias folds the tangential DELTA + projected velocity:
//       biasedF = -(fric.dot(delta) - tanDelta);
//       requiredFForce = (biasedF + tanVel*projectionDt) * vmF.
struct PcResult { float deltaF; float deltaFr0; float deltaFr1; float2 appliedForce; }
PcResult solvePCOutputDeltaVTGS(
    float3 normal, float error, float3 delta,
    float3 fric0, float3 fric1,
    float tanDelta0, float tanDelta1,
    float tanVel0, float tanVel1,
    float normalDelta,
    float normalVel,
    float velMultiplier, float vmF0, float vmF1,
    float frictionCoefficient, float restOffset,
    float adhesion, float adhesionRadiusScale,
    bool isVelocityIteration,
    float dt, float2 appliedForce)
{
    const float projectionDt = isVelocityIteration ? 0.0f : dt;
    const float separation = (error - dot(normal, delta) + normalDelta) + normalVel * projectionDt;
    float clamp = -appliedForce.x;

    if (adhesionRadiusScale > 0.0f && adhesion > 0.0f)
    {
        const float adhesionRadius = restOffset * adhesionRadiusScale - restOffset;
        if (separation < adhesionRadiusScale)
        {
            clamp = pmin(clamp, -pmax(0.0f, adhesion * quart(1.0f - separation / adhesionRadius)
                                             * pmax(0.0f, separation) * velMultiplier));
        }
    }

    PcResult r;
    r.deltaF = pmax(clamp, -separation * velMultiplier);
    appliedForce.x = pmax(0.0f, appliedForce.x + r.deltaF);

    const float maxFriction = abs(appliedForce.x) * frictionCoefficient;

    const float biasedF0 = -(dot(fric0, delta) - tanDelta0);
    const float biasedF1 = -(dot(fric1, delta) - tanDelta1);
    const float requiredF0Force = (biasedF0 + tanVel0 * projectionDt) * vmF0;
    const float requiredF1Force = (biasedF1 + tanVel1 * projectionDt) * vmF1;

    const float requiredForce = sqrt(requiredF0Force * requiredF0Force + requiredF1Force * requiredF1Force);
    const float recipRequiredForce = (requiredForce >= 1e-16f) ? (1.0f / requiredForce) : 0.0f;

    const float ratio0 = requiredF0Force * recipRequiredForce;
    const float ratio1 = requiredF1Force * recipRequiredForce;

    const float deltaFr = pmin(requiredForce + appliedForce.y, maxFriction) - appliedForce.y;
    appliedForce.y = appliedForce.y + deltaFr;

    r.deltaFr0 = deltaFr * ratio0;
    r.deltaFr1 = deltaFr * ratio1;
    r.appliedForce = appliedForce;
    return r;
}

struct PcSolveParams
{
    uint64_t particleSystems;      // @0
    uint64_t sortedContacts;       // @8  (ABI parity; unused)
    uint64_t primitiveConstraints; // @16
    uint64_t numContacts;          // @24
    uint64_t prePrepDesc;          // @32
    uint64_t solverCoreDesc;       // @40
    uint64_t sharedDesc;           // @48
    uint64_t deltaPos;             // @56 output
    uint64_t appliedForces;        // @64
    float dt;                      // @72
    float relaxationCoefficient;   // @76
    uint isVelIterWord;            // @80 (bool byte0)
    uint64_t artiCoreDesc;         // @88 -> 96
}
[[vk::push_constant]]
ConstantBuffer<PcSolveParams> gSv;

[shader("compute")]
[numthreads(PC_BLOCK, 1, 1)]
void ps_solvePCOutputParticleDeltaVTGSLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgSolverCoreDescGpu> solverCoreDesc = Ptr<PxgSolverCoreDescGpu>(gSv.solverCoreDesc);
    Ptr<PxgSolverSharedDescPGSGpu> sharedDesc = Ptr<PxgSolverSharedDescPGSGpu>(gSv.sharedDesc);
    Ptr<PxgPrePrepDescGpu> prePrepDesc = Ptr<PxgPrePrepDescGpu>(gSv.prePrepDesc);

    // PxgVelocityReader (dataReadWriteHelper.cuh:70) non-articulation state.  The
    // TGS iter pool is 3 float4 per body: v0/v1/v2 pack lin/ang VELOCITY + lin/ang
    // DELTA.  accOffset is a float4 index into iter_solverBodyVelPool.
    const uint numSolverBodies = solverCoreDesc[0].numSolverBodies;
    const uint accOffset = solverCoreDesc[0].accumulatedBodyDeltaVOffset;
    Ptr<float4> solverBodyVelPool = Ptr<float4>(sharedDesc[0].iter_solverBodyVelPool);

    const uint tNumContacts = Ptr<uint>(gSv.numContacts)[0];

    const uint nbBlocksRequired = (tNumContacts + PC_BLOCK - 1u) / PC_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + PC_GRID - 1u) / PC_GRID;
    const uint idx = groupThreadId.x;

    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gSv.particleSystems);
    Ptr<float4> deltaPos = Ptr<float4>(gSv.deltaPos);
    Ptr<float2> appliedForces = Ptr<float2>(gSv.appliedForces);

    const float dt = gSv.dt;
    const float relaxation = gSv.relaxationCoefficient;
    const bool isVelocityIteration = (gSv.isVelIterWord & 0xffu) != 0u;

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * PC_BLOCK + idx + nbIterationsPerBlock * groupId.x * PC_BLOCK;
        if (workIndex >= tNumContacts)
            return;

        const uint tIdx = workIndex & 31u;
        const uint64_t blockBase = gSv.primitiveConstraints + uint64_t(workIndex / 32u) * uint64_t(CB_STRIDE);

        float2 appliedForce = appliedForces[workIndex];

        const uint2 tParticleId = Ptr<uint2>(blockBase + CB_PARTICLEID)[tIdx];
        const uint particleSystemId = tParticleId.y;
        const uint particleId = tParticleId.x;

        Ptr<float4> sortedDeltaP_invMassW = Ptr<float4>(systems[particleSystemId].mSortedDeltaP);
        const uint64_t phasesBase = systems[particleSystemId].mSortedPhaseArray;
        const uint64_t phaseToMat = systems[particleSystemId].mPhaseGroupToMaterialHandle;
        const uint64_t materials = systems[particleSystemId].mParticleMaterials;
        const uint materialStride = systems[particleSystemId].mParticleMaterialStride;

        const uint phase = (Ptr<uint>(phasesBase))[particleId];
        const uint group = phase & PXG_PARTICLE_PHASE_GROUP_MASK;
        const uint mi = loadPhaseToMaterial(phaseToMat, group);
        const float frictionCoefficient = loadMaterialFloat(materials, mi, materialStride, MAT_FRICTION);

        const float4 deltaP_invMassW = sortedDeltaP_invMassW[particleId];
        const float invMass1 = deltaP_invMassW.w;

        // Infinite-mass particles do not respond (CUDA `continue`).
        if (invMass1 == 0.0f)
            continue;

        const float adhesionRadiusScale = loadMaterialFloat(materials, mi, materialStride, MAT_ADHESION_RADIUS_SCALE);
        const float adhesion = loadMaterialFloat(materials, mi, materialStride, MAT_ADHESION);
        const float restOffset = systems[particleSystemId].mData_RestOffset;

        const float3 delta = float3(deltaP_invMassW.x, deltaP_invMassW.y, deltaP_invMassW.z);

        const uint2 rigidId = Ptr<uint2>(blockBase + CB_RIGIDID)[tIdx];
        const bool isStaticBody = (rigidId.x == STATIC_NODE);
        const bool isArticulation = (rigidId.y & 1u) != 0u;

        // PxgVelocityReader::readVelocitiesTGS (dataReadWriteHelper.cuh:122).
        uint solverBodyIdx = 0u;
        if (!isStaticBody)
            solverBodyIdx = (Ptr<uint>(prePrepDesc[0].solverBodyIndices))[rigidId.x];

        float3 linVel, angVel, linDelta, angDelta;
        if (isArticulation)
        {
            // DEFERRED: articulation link velocity read (outArtiVelocity, maxLinks,
            // nbArticulations).  Never taken for rigid/static contacts; sinking the
            // artiCoreDesc / outArtiVelocity reads here keeps a rigid/static contact
            // from ever dereferencing them (see file header).
            linVel = float3(0.0f, 0.0f, 0.0f);
            angVel = float3(0.0f, 0.0f, 0.0f);
            linDelta = float3(0.0f, 0.0f, 0.0f);
            angDelta = float3(0.0f, 0.0f, 0.0f);
        }
        else
        {
            const float4 v0 = solverBodyVelPool[accOffset + solverBodyIdx];
            const float4 v1 = solverBodyVelPool[accOffset + solverBodyIdx + numSolverBodies];
            const float4 v2 = solverBodyVelPool[accOffset + solverBodyIdx + numSolverBodies + numSolverBodies];

            linVel = float3(v0.x, v0.y, v0.z);
            angVel = float3(v0.w, v1.x, v1.y);
            linDelta = float3(v1.z, v1.w, v2.x);
            angDelta = float3(v2.y, v2.z, v2.w);
        }

        const float4 raXn_vm = Ptr<float4>(blockBase + CB_RAXN_VM)[tIdx];
        const float4 normal_err = Ptr<float4>(blockBase + CB_NORMAL_ERR)[tIdx];
        const float4 fricTan0_invMass0 = Ptr<float4>(blockBase + CB_FRIC0_IM0)[tIdx];
        const float4 raXnF0_vm = Ptr<float4>(blockBase + CB_RAXNF0_VM)[tIdx];
        const float4 raXnF1_vm = Ptr<float4>(blockBase + CB_RAXNF1_VM)[tIdx];

        const float3 normal = float3(normal_err.x, normal_err.y, normal_err.z);
        const float3 fric0 = float3(fricTan0_invMass0.x, fricTan0_invMass0.y, fricTan0_invMass0.z);
        const float3 fric1 = cross(normal, fric0);
        const float error = normal_err.w;
        const float3 raXn = float3(raXn_vm.x, raXn_vm.y, raXn_vm.z);
        const float velMultiplier = raXn_vm.w;
        const float3 raXnF0 = float3(raXnF0_vm.x, raXnF0_vm.y, raXnF0_vm.z);
        const float vmF0 = raXnF0_vm.w;
        const float3 raXnF1 = float3(raXnF1_vm.x, raXnF1_vm.y, raXnF1_vm.z);
        const float vmF1 = raXnF1_vm.w;

        // Constraint normal/tangential velocity + running-delta projections.
        const float normalVel = dot(linVel, normal) + dot(angVel, raXn);
        const float normalDelta = dot(linDelta, normal) + dot(angDelta, raXn);

        const float tanVel0 = dot(linVel, fric0) + dot(angVel, raXnF0);
        const float tanDelta0 = dot(linDelta, fric0) + dot(angDelta, raXnF0);

        const float tanVel1 = dot(linVel, fric1) + dot(angVel, raXnF1);
        const float tanDelta1 = dot(linDelta, fric1) + dot(angDelta, raXnF1);

        PcResult res = solvePCOutputDeltaVTGS(
            normal, error, delta,
            fric0, fric1,
            tanDelta0, tanDelta1,
            tanVel0, tanVel1,
            normalDelta,
            normalVel,
            velMultiplier, vmF0, vmF1,
            frictionCoefficient, restOffset,
            adhesion, adhesionRadiusScale,
            isVelocityIteration,
            dt, appliedForce);

        const float3 deltaLinVel = (float3(-(normal.x * res.deltaF), -(normal.y * res.deltaF), -(normal.z * res.deltaF))
                                    + fric0 * res.deltaFr0 + fric1 * res.deltaFr1) * invMass1 * relaxation;
        const float scale = (res.deltaF != 0.0f) ? 1.0f : 0.0f;

        deltaPos[workIndex] = float4(deltaLinVel.x, deltaLinVel.y, deltaLinVel.z, scale);
        appliedForces[workIndex] = res.appliedForce;
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
