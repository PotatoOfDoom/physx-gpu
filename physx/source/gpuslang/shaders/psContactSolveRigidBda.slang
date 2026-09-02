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

// psContactSolveRigidBda.slang -- P9.44: the RIGID side of the particle/primitive
// contact solve, the two kernels of gpusimulationcontroller/src/CUDA/particlesystem.cu
// that the particle-side twins in psContactSolveBda / psContactSolveTgsBda left
// DEFERRED ("inert for a static floor"):
//
//   ps_solvePCOutputRigidDeltaVLaunch     (particlesystem.cu:1943-2121, PGS)
//   ps_solvePCOutputRigidDeltaVTGSLaunch  (particlesystem.cu:2123-2274, TGS)
//
// Both walk the by-RIGID sorted constraint blocks, re-solve the contact against the
// rigid body's current velocity with the SAME solvePCOutputDeltaV{,TGS} core the
// particle side uses (transcribed byte-for-byte from the two particle-side
// modules), and write the per-contact RIGID reaction: deltaVel[i] = linear delta
// with the relaxation weight in .w, deltaVel[i + n] = angular delta, plus the
// updated appliedForce.  accumulateDeltaVRigidFirst/SecondLaunch
// (rigidDeltaAccumBda.slang) then fold those into the solver body.  Launch
// PxgParticleSystemCore.cpp:2053-2130 (solvePrimitiveCollisionForRigids), block
// PS_COLLISION = 256, grid PS_COLLISION = 1024, the identical 96-byte pack the
// particle-side solve takes (only the CUfunction differs).  P9.40's guard found the
// TGS twin launched in all 10 particle scenes (x600..x2700).
//
// DIFFERENCES to the particle-side twins, all from the .cu:
//   * a STATIC rigid id skips the lane without writing (the prepare kernel zeroed
//     deltaVel already); invMass0 == 0 (kinematic) writes explicit zeros;
//   * the rigid velocity is read for the rigid body, not the particle: PGS as
//     initialVel + accumulated delta, TGS through the packed 3-float4 pool, and for
//     an ARTICULATION LINK from solverCoreDesc->outArtiVelocity at
//     solverBodyIndex * maxLinks + linkID (+ k * maxLinks * nbArticulations for
//     angular / TGS deltas) -- PxgVelocityReader (dataReadWriteHelper.cuh:70).  The
//     articulation branch is PORTED here, not deferred: the .cu hoists the
//     artiCoreDesc reads to the kernel top (the engine always passes the real
//     descriptor), and so does this port;
//   * the reaction is deltaLin = ((n*dF - f0*dFr0 - f1*dFr1) * invMass0) * relax,
//     deltaAng = (raXn*dF - raXnF0*dFr0 - raXnF1*dFr1) * relax with
//     relax = relaxationCoefficient / dt, only when some delta is nonzero AND
//     invMass0 > 0; the weight is max(invMass0 / max(invMass0, particleInvMass), 0.01).
//
// The particle-side twins are byte-untouched.  In PXG_NOCONTRACT_SHADERS like them.

import pxgSlangParticleDesc;      // PxgParticleSystemGpu, PXG_PARTICLE_PHASE_GROUP_MASK
import pxgSlangPrepDesc;          // PxgPrePrepDescGpu
import pxgSlangSolverDesc;        // PxgSolverCoreDescGpu, PxgSolverSharedDescPGSGpu
import pxgSlangArticulationDesc;  // PXG_ARTI_CORE_MAX_LINKS / NB_ARTICULATIONS, artiU32Field

static const uint PC_BLOCK = 256u;   // PS_COLLISION block
static const uint PC_GRID = 1024u;   // PS_COLLISION grid
static const uint STATIC_NODE = 0xffffffffu;   // PX_INVALID_NODE

// PxsParticleMaterialData float indices (PxParticleGpu.h:47).
static const uint MAT_FRICTION = 0u;
static const uint MAT_ADHESION = 2u;
static const uint MAT_ADHESION_RADIUS_SCALE = 4u;

// PxgParticlePrimitiveConstraintBlock SoA byte offsets (psContactSolveBda.slang).
static const uint CB_STRIDE = 3072u;
static const uint CB_RAXN_VM = 0u;
static const uint CB_NORMAL_ERR = 512u;
static const uint CB_FRIC0_IM0 = 1024u;
static const uint CB_RAXNF0_VM = 1536u;
static const uint CB_RAXNF1_VM = 2048u;
static const uint CB_RIGIDID = 2560u;
static const uint CB_PARTICLEID = 2816u;

// ---------------------------------------------------------------------------
// Shared helpers, transcribed from psContactSolveBda / psContactSolveTgsBda.
// ---------------------------------------------------------------------------
float pmin(float a, float b) { return (a < b) ? a : b; }   // PxMin = selectMin
float pmax(float a, float b) { return (a > b) ? a : b; }   // PxMax = selectMax
float quart(float x) { return x * x * x * x; }

uint loadPhaseToMaterial(uint64_t handleBase, uint group)
{
    let word = (Ptr<uint>(handleBase))[group >> 1];
    return (word >> ((group & 1u) * 16u)) & 0xffffu;
}

float loadMaterialFloat(uint64_t materialsBase, uint mi, uint stride, uint floatIndex)
{
    let elem = materialsBase + uint64_t(mi) * uint64_t(stride);
    return (Ptr<float>(elem))[floatIndex];
}

struct PcResult { float deltaF; float deltaFr0; float deltaFr1; float2 appliedForce; }

// solvePCOutputDeltaV (particlesystem.cu:1421) -- byte-identical to psContactSolveBda.slang.
PcResult solvePCOutputDeltaV(
    float3 normal, float error, float3 delta,
    float3 fric0, float3 fric1,
    float tanVel0, float tanVel1,
    float normalVel,
    float velMultiplier, float vmF0, float vmF1,
    float frictionCoefficient, float restOffset,
    float adhesion, float adhesionRadiusScale,
    float dt, float2 appliedForce)
{
    const float separation = (error - dot(normal, delta));
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
    r.deltaF = pmax(clamp, (-separation - normalVel * dt) * velMultiplier);
    appliedForce.x = pmax(0.0f, r.deltaF + appliedForce.x);

    const float friction = appliedForce.x * frictionCoefficient;

    const float biasedF0 = -(dot(fric0, delta));
    const float biasedF1 = -(dot(fric1, delta));
    const float requiredF0Force = (biasedF0 + tanVel0 * dt) * vmF0;
    const float requiredF1Force = (biasedF1 + tanVel1 * dt) * vmF1;

    const float requiredForce = sqrt(requiredF0Force * requiredF0Force + requiredF1Force * requiredF1Force);
    const float recipRequiredForce = (requiredForce >= 1e-16f) ? (1.0f / requiredForce) : 0.0f;

    const float ratio0 = requiredF0Force * recipRequiredForce;
    const float ratio1 = requiredF1Force * recipRequiredForce;

    const float deltaFr = pmin(requiredForce + appliedForce.y, friction) - appliedForce.y;
    appliedForce.y = appliedForce.y + deltaFr;

    r.deltaFr0 = deltaFr * ratio0;
    r.deltaFr1 = deltaFr * ratio1;
    r.appliedForce = appliedForce;
    return r;
}

// solvePCOutputDeltaVTGS (particlesystem.cu:1473) -- byte-identical to psContactSolveTgsBda.slang.
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

// The rigid reaction (.cu:2103-2118 / 2257-2271), shared by both twins.
struct RigidReaction { float4 lin; float4 ang; }
RigidReaction rigidReaction(float3 normal, float3 fric0, float3 fric1,
                            float3 raXn, float3 raXnF0, float3 raXnF1,
                            float invMass0, float particleInvMass,
                            PcResult res, float relaxationCoefficient, float dt)
{
    const float relaxation = relaxationCoefficient / dt;

    float count = 0.0f;
    float3 deltaAngVel0 = float3(0.0f, 0.0f, 0.0f);
    float3 deltaLinVel0 = float3(0.0f, 0.0f, 0.0f);

    // if rigid is kinematic or static, deltaVel must remain zero
    if ((res.deltaF != 0.0f || res.deltaFr0 != 0.0f || res.deltaFr1 != 0.0f) && invMass0 > 0.0f)
    {
        deltaAngVel0 = (raXn * res.deltaF - raXnF0 * res.deltaFr0 - raXnF1 * res.deltaFr1) * relaxation;
        deltaLinVel0 = ((normal * res.deltaF) - fric0 * res.deltaFr0 - fric1 * res.deltaFr1) * invMass0 * relaxation;
        count = pmax(invMass0 / pmax(invMass0, particleInvMass), 0.01f);
    }

    RigidReaction o;
    o.lin = float4(deltaLinVel0.x, deltaLinVel0.y, deltaLinVel0.z, count);
    o.ang = float4(deltaAngVel0.x, deltaAngVel0.y, deltaAngVel0.z, 0.0f);
    return o;
}

// ===========================================================================
// CUDA ABI (PxgParticleSystemCore.cpp:2074-2089), both twins: (PxgParticleSystem*
//   particleSystems, PxgParticlePrimitiveContact* sortedContacts,
//   PxgParticlePrimitiveConstraintBlock* primitiveConstraints, PxU32* numContacts,
//   PxgPrePrepDesc* prePrepDesc, PxgSolverCoreDesc* solverCoreDesc,
//   PxgSolverSharedDesc<IterativeSolveData>* sharedDesc, float4* deltaVel,
//   float2* appliedForces, PxReal dt, PxReal relaxationCoefficient,
//   bool isVelocityIteration, PxgArticulationCoreDesc* artiCoreDesc)
//   -> {8,8,8,8,8,8,8,8,8,4,4,1,8} = 96-byte blob (kPsSolvePCParticle).
//   sortedContacts is ABI parity only (the ids are read from the constraint block).
// ===========================================================================
struct PcSolveParams
{
    uint64_t particleSystems;      // @0
    uint64_t sortedContacts;       // @8  (ABI parity; unused)
    uint64_t primitiveConstraints; // @16
    uint64_t numContacts;          // @24
    uint64_t prePrepDesc;          // @32
    uint64_t solverCoreDesc;       // @40
    uint64_t sharedDesc;           // @48
    uint64_t deltaVel;             // @56 output
    uint64_t appliedForces;        // @64
    float dt;                      // @72
    float relaxationCoefficient;   // @76
    uint isVelIterWord;            // @80 (bool byte0)
    uint64_t artiCoreDesc;         // @88 -> 96
}
[[vk::push_constant]]
ConstantBuffer<PcSolveParams> gSv;

// The per-lane front half both twins share (.cu:1986-2029 / 2157-2200): the
// constraint block's ids, the early-outs, the particle's material and delta.
struct LaneFront
{
    bool skip;            // static rigid: continue without writing
    bool zeroOut;         // invMass0 == 0: write zeros and continue
    uint64_t blockBase;
    uint tIdx;
    uint2 rigidId;
    float4 fricTan0_invMass0;
    float4 delta4;
    float frictionCoefficient;
    float adhesion;
    float adhesionRadiusScale;
    float restOffset;
}

LaneFront laneFront(uint workIndex)
{
    LaneFront f;
    f.skip = false;
    f.zeroOut = false;
    f.tIdx = workIndex & 31u;
    f.blockBase = gSv.primitiveConstraints + uint64_t(workIndex / 32u) * uint64_t(CB_STRIDE);

    f.rigidId = Ptr<uint2>(f.blockBase + CB_RIGIDID)[f.tIdx];
    f.fricTan0_invMass0 = float4(0.0f, 0.0f, 0.0f, 0.0f);
    f.delta4 = float4(0.0f, 0.0f, 0.0f, 0.0f);
    f.frictionCoefficient = 0.0f;
    f.adhesion = 0.0f;
    f.adhesionRadiusScale = 0.0f;
    f.restOffset = 0.0f;

    // TODO in the .cu ("need to figure out how to make this work for articulation
    // links!") -- a STATIC body is skipped; an articulation link is NOT static and
    // goes through.
    if (f.rigidId.x == STATIC_NODE)
    {
        f.skip = true;
        return f;
    }

    f.fricTan0_invMass0 = Ptr<float4>(f.blockBase + CB_FRIC0_IM0)[f.tIdx];
    if (f.fricTan0_invMass0.w == 0.0f)
    {
        f.zeroOut = true;
        return f;
    }

    const uint2 tParticleId = Ptr<uint2>(f.blockBase + CB_PARTICLEID)[f.tIdx];
    const uint particleSystemId = tParticleId.y;   // PxGetParticleSystemId
    const uint particleId = tParticleId.x;         // PxGetParticleIndex

    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gSv.particleSystems);
    Ptr<float4> sortedDeltaP_invMassW = Ptr<float4>(systems[particleSystemId].mSortedDeltaP);
    const uint64_t phasesBase = systems[particleSystemId].mSortedPhaseArray;
    const uint64_t phaseToMat = systems[particleSystemId].mPhaseGroupToMaterialHandle;
    const uint64_t materials = systems[particleSystemId].mParticleMaterials;
    const uint materialStride = systems[particleSystemId].mParticleMaterialStride;

    const uint phase = (Ptr<uint>(phasesBase))[particleId];
    const uint group = phase & PXG_PARTICLE_PHASE_GROUP_MASK;
    const uint mi = loadPhaseToMaterial(phaseToMat, group);
    f.frictionCoefficient = loadMaterialFloat(materials, mi, materialStride, MAT_FRICTION);
    f.adhesionRadiusScale = loadMaterialFloat(materials, mi, materialStride, MAT_ADHESION_RADIUS_SCALE);
    f.adhesion = loadMaterialFloat(materials, mi, materialStride, MAT_ADHESION);
    f.restOffset = systems[particleSystemId].mData_RestOffset;

    f.delta4 = sortedDeltaP_invMassW[particleId];
    return f;
}

// The constraint geometry both twins read after the velocity (.cu:2053-2072).
struct LaneGeom
{
    float3 normal; float3 fric0; float3 fric1; float error;
    float3 raXn; float velMultiplier;
    float3 raXnF0; float vmF0;
    float3 raXnF1; float vmF1;
}
LaneGeom laneGeom(LaneFront f)
{
    const float4 raXn_vm = Ptr<float4>(f.blockBase + CB_RAXN_VM)[f.tIdx];
    const float4 normal_err = Ptr<float4>(f.blockBase + CB_NORMAL_ERR)[f.tIdx];
    const float4 raXnF0_vm = Ptr<float4>(f.blockBase + CB_RAXNF0_VM)[f.tIdx];
    const float4 raXnF1_vm = Ptr<float4>(f.blockBase + CB_RAXNF1_VM)[f.tIdx];

    LaneGeom g;
    g.normal = float3(normal_err.x, normal_err.y, normal_err.z);
    g.fric0 = float3(f.fricTan0_invMass0.x, f.fricTan0_invMass0.y, f.fricTan0_invMass0.z);
    g.fric1 = cross(g.normal, g.fric0);
    g.error = normal_err.w;
    g.raXn = float3(raXn_vm.x, raXn_vm.y, raXn_vm.z);
    g.velMultiplier = raXn_vm.w;
    g.raXnF0 = float3(raXnF0_vm.x, raXnF0_vm.y, raXnF0_vm.z);
    g.vmF0 = raXnF0_vm.w;
    g.raXnF1 = float3(raXnF1_vm.x, raXnF1_vm.y, raXnF1_vm.z);
    g.vmF1 = raXnF1_vm.w;
    return g;
}

// ===========================================================================
// 1. ps_solvePCOutputRigidDeltaVLaunch (particlesystem.cu:1943, PGS)
// ===========================================================================
[shader("compute")]
[numthreads(PC_BLOCK, 1, 1)]
void ps_solvePCOutputRigidDeltaVLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgSolverCoreDescGpu> solverCoreDesc = Ptr<PxgSolverCoreDescGpu>(gSv.solverCoreDesc);
    Ptr<PxgSolverSharedDescPGSGpu> sharedDesc = Ptr<PxgSolverSharedDescPGSGpu>(gSv.sharedDesc);
    Ptr<PxgPrePrepDescGpu> prePrepDesc = Ptr<PxgPrePrepDescGpu>(gSv.prePrepDesc);

    // .cu:1960-1976, hoisted as the .cu hoists them.
    const uint accOffset = solverCoreDesc[0].accumulatedBodyDeltaVOffset;
    Ptr<float4> solverBodyVelPool = Ptr<float4>(sharedDesc[0].iter_solverBodyVelPool);
    Ptr<float4> initialVel = Ptr<float4>(solverCoreDesc[0].outSolverVelocity);
    const uint numSolverBodies = solverCoreDesc[0].numSolverBodies;
    Ptr<uint> solverBodyIndices = Ptr<uint>(prePrepDesc[0].solverBodyIndices);

    const uint tNumContacts = Ptr<uint>(gSv.numContacts)[0];
    const uint nbBlocksRequired = (tNumContacts + PC_BLOCK - 1u) / PC_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + PC_GRID - 1u) / PC_GRID;
    const uint idx = groupThreadId.x;

    Ptr<float4> artiLinkVelocities = Ptr<float4>(solverCoreDesc[0].outArtiVelocity);
    const uint maxLinks = artiU32Field(gSv.artiCoreDesc, PXG_ARTI_CORE_MAX_LINKS);
    const uint nbArticulations = artiU32Field(gSv.artiCoreDesc, PXG_ARTI_CORE_NB_ARTICULATIONS);
    const uint artiOffset = maxLinks * nbArticulations;

    Ptr<float4> deltaVel = Ptr<float4>(gSv.deltaVel);
    Ptr<float2> appliedForces = Ptr<float2>(gSv.appliedForces);
    const float dt = gSv.dt;

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * PC_BLOCK + idx + nbIterationsPerBlock * groupId.x * PC_BLOCK;
        if (workIndex >= tNumContacts)
            return;

        const LaneFront f = laneFront(workIndex);
        if (f.skip)
            continue;
        if (f.zeroOut)
        {
            deltaVel[workIndex] = float4(0.0f, 0.0f, 0.0f, 0.0f);
            deltaVel[workIndex + tNumContacts] = float4(0.0f, 0.0f, 0.0f, 0.0f);
            appliedForces[workIndex] = float2(0.0f, 0.0f);
            continue;
        }

        const float3 delta = float3(f.delta4.x, f.delta4.y, f.delta4.z);
        float2 appliedForce = appliedForces[workIndex];

        // .cu:2032-2050
        const uint solverBodyIndex = solverBodyIndices[f.rigidId.x];
        float4 linearVelocity;
        float4 angularVelocity;
        if ((f.rigidId.y & 1u) != 0u)
        {
            const uint index = solverBodyIndex * maxLinks;
            const uint linkID = f.rigidId.y >> 1;
            linearVelocity = artiLinkVelocities[index + linkID];
            angularVelocity = artiLinkVelocities[index + linkID + artiOffset];
        }
        else
        {
            linearVelocity = initialVel[solverBodyIndex] + solverBodyVelPool[accOffset + solverBodyIndex];
            angularVelocity = initialVel[solverBodyIndex + numSolverBodies]
                            + solverBodyVelPool[accOffset + solverBodyIndex + numSolverBodies];
        }

        const LaneGeom g = laneGeom(f);
        const float invMass0 = f.fricTan0_invMass0.w;

        const float3 linVel0 = float3(linearVelocity.x, linearVelocity.y, linearVelocity.z);
        const float3 angVel0 = float3(angularVelocity.x, angularVelocity.y, angularVelocity.z);

        const float normalVel = dot(linVel0, g.normal) + dot(angVel0, g.raXn);
        const float tanVel0 = dot(linVel0, g.fric0) + dot(angVel0, g.raXnF0);
        const float tanVel1 = dot(linVel0, g.fric1) + dot(angVel0, g.raXnF1);

        const PcResult res = solvePCOutputDeltaV(
            g.normal, g.error, delta,
            g.fric0, g.fric1,
            tanVel0, tanVel1,
            normalVel,
            g.velMultiplier, g.vmF0, g.vmF1,
            f.frictionCoefficient, f.restOffset,
            f.adhesion, f.adhesionRadiusScale,
            dt, appliedForce);

        const RigidReaction o = rigidReaction(g.normal, g.fric0, g.fric1, g.raXn, g.raXnF0, g.raXnF1,
                                              invMass0, f.delta4.w, res, gSv.relaxationCoefficient, dt);
        deltaVel[workIndex] = o.lin;
        deltaVel[workIndex + tNumContacts] = o.ang;
        appliedForces[workIndex] = res.appliedForce;
    }
}

// ===========================================================================
// 2. ps_solvePCOutputRigidDeltaVTGSLaunch (particlesystem.cu:2123, TGS)
// ===========================================================================
[shader("compute")]
[numthreads(PC_BLOCK, 1, 1)]
void ps_solvePCOutputRigidDeltaVTGSLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgSolverCoreDescGpu> solverCoreDesc = Ptr<PxgSolverCoreDescGpu>(gSv.solverCoreDesc);
    Ptr<PxgSolverSharedDescPGSGpu> sharedDesc = Ptr<PxgSolverSharedDescPGSGpu>(gSv.sharedDesc);
    Ptr<PxgPrePrepDescGpu> prePrepDesc = Ptr<PxgPrePrepDescGpu>(gSv.prePrepDesc);

    const uint numSolverBodies = solverCoreDesc[0].numSolverBodies;
    const uint tNumContacts = Ptr<uint>(gSv.numContacts)[0];
    const uint nbBlocksRequired = (tNumContacts + PC_BLOCK - 1u) / PC_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + PC_GRID - 1u) / PC_GRID;
    const uint idx = groupThreadId.x;

    // PxgVelocityReader (dataReadWriteHelper.cuh:70), constructed at the top.
    Ptr<uint> solverBodyIndices = Ptr<uint>(prePrepDesc[0].solverBodyIndices);
    Ptr<float4> artiLinkVelocities = Ptr<float4>(solverCoreDesc[0].outArtiVelocity);
    const uint maxLinks = artiU32Field(gSv.artiCoreDesc, PXG_ARTI_CORE_MAX_LINKS);
    const uint nbArticulations = artiU32Field(gSv.artiCoreDesc, PXG_ARTI_CORE_NB_ARTICULATIONS);
    const uint artiOffset = maxLinks * nbArticulations;
    const uint accOffset = solverCoreDesc[0].accumulatedBodyDeltaVOffset;
    Ptr<float4> solverBodyVelPool = Ptr<float4>(sharedDesc[0].iter_solverBodyVelPool);

    Ptr<float4> deltaVel = Ptr<float4>(gSv.deltaVel);
    Ptr<float2> appliedForces = Ptr<float2>(gSv.appliedForces);
    const float dt = gSv.dt;
    const bool isVelocityIteration = (gSv.isVelIterWord & 0xffu) != 0u;

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * PC_BLOCK + idx + nbIterationsPerBlock * groupId.x * PC_BLOCK;
        if (workIndex >= tNumContacts)
            return;

        const LaneFront f = laneFront(workIndex);
        if (f.skip)
            continue;
        if (f.zeroOut)
        {
            deltaVel[workIndex] = float4(0.0f, 0.0f, 0.0f, 0.0f);
            deltaVel[workIndex + tNumContacts] = float4(0.0f, 0.0f, 0.0f, 0.0f);
            appliedForces[workIndex] = float2(0.0f, 0.0f);
            continue;
        }

        const float3 delta = float3(f.delta4.x, f.delta4.y, f.delta4.z);
        float2 appliedForce = appliedForces[workIndex];

        // readVelocitiesTGS (dataReadWriteHelper.cuh:122); the id is never static here.
        const uint solverBodyIdx = solverBodyIndices[f.rigidId.x];
        float3 linVel, angVel, linDelta, angDelta;
        if ((f.rigidId.y & 1u) != 0u)
        {
            const uint index = solverBodyIdx * maxLinks;
            const uint linkID = f.rigidId.y >> 1;
            const float4 linearVelocity = artiLinkVelocities[index + linkID];
            const float4 angularVelocity = artiLinkVelocities[index + linkID + artiOffset];
            const float4 lDelta = artiLinkVelocities[index + linkID + artiOffset + artiOffset];
            const float4 aDelta = artiLinkVelocities[index + linkID + artiOffset + artiOffset + artiOffset];
            linVel = float3(linearVelocity.x, linearVelocity.y, linearVelocity.z);
            angVel = float3(angularVelocity.x, angularVelocity.y, angularVelocity.z);
            linDelta = float3(lDelta.x, lDelta.y, lDelta.z);
            angDelta = float3(aDelta.x, aDelta.y, aDelta.z);
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

        const LaneGeom g = laneGeom(f);
        const float invMass0 = f.fricTan0_invMass0.w;

        const float normalVel = dot(linVel, g.normal) + dot(angVel, g.raXn);
        const float normalDelta = dot(linDelta, g.normal) + dot(angDelta, g.raXn);

        const float tanVel0 = dot(linVel, g.fric0) + dot(angVel, g.raXnF0);
        const float tanDelta0 = dot(linDelta, g.fric0) + dot(angDelta, g.raXnF0);

        const float tanVel1 = dot(linVel, g.fric1) + dot(angVel, g.raXnF1);
        const float tanDelta1 = dot(linDelta, g.fric1) + dot(angDelta, g.raXnF1);

        const PcResult res = solvePCOutputDeltaVTGS(
            g.normal, g.error, delta,
            g.fric0, g.fric1,
            tanDelta0, tanDelta1,
            tanVel0, tanVel1,
            normalDelta,
            normalVel,
            g.velMultiplier, g.vmF0, g.vmF1,
            f.frictionCoefficient, f.restOffset,
            f.adhesion, f.adhesionRadiusScale,
            isVelocityIteration,
            dt, appliedForce);

        const RigidReaction o = rigidReaction(g.normal, g.fric0, g.fric1, g.raXn, g.raXnF0, g.raXnF1,
                                              invMass0, f.delta4.w, res, gSv.relaxationCoefficient, dt);
        deltaVel[workIndex] = o.lin;
        deltaVel[workIndex + tNumContacts] = o.ang;
        appliedForces[workIndex] = res.appliedForce;
    }
}
