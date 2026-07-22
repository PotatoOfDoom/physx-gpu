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

// psContactSolveBda — P6.4b-2: the particle-SIDE contact SOLVE half of the PBD
// particle/primitive consume path on the BDA engine ABI.  Three kernels ported
// line-for-line from gpusimulationcontroller/src/CUDA/particlesystem.cu, running
// AFTER the contact SORT + reorder + range-find (P6.4b-1) has produced the
// by-particle sorted contacts and each particle's [start,end) range:
//
//   ps_contactPrepareLaunch (particlesystem.cu:1238) — prepares each primitive
//     contact CONSTRAINT: effective mass (velMultiplier = 1/(rigidResponse +
//     invMass1)), flipped contact normal, friction tangents, error/bias, written
//     into the SoA-per-warp PxgParticlePrimitiveConstraintBlock (workIndex/32 the
//     block, workIndex&31 the lane).  Zeroes appliedForces + (if non-null) the
//     rigid deltaV output.  block PS_COLLISION=256, grid PS_COLLISION=1024;
//     launch PxgParticleSystemCore.cpp:1609.
//   ps_solvePCOutputParticleDeltaVLaunch (particlesystem.cu:1531, the NON-TGS
//     variant; the ...TGSLaunch twin is skipped) + its device helper
//     solvePCOutputDeltaV (particlesystem.cu:1421) — reads the rigid body
//     velocity, computes the constraint normal/tangent velocities, and produces a
//     per-CONTACT position deltaV (deltaPos[workIndex] = deltaLinVel, w = scale)
//     plus the updated appliedForce.  block PS_COLLISION=256, grid 1024; launch
//     PxgParticleSystemCore.cpp:1948.
//   ps_accumulateDeltaVParticleLaunch (particlesystem.cu:2828) — reduces the
//     per-contact deltaPos over each particle's [start,end) range into that
//     particle's mAccumDeltaP (SOR-averaged by 1/max(1, sum(scale))).  block
//     ACCUMULATE_DELTA=512, grid ACCUMULATE_DELTA=32; launch
//     PxgParticleSystemCore.cpp:2005.
//
// Reused mirrors / cores (no new descriptor mirrors were required):
//   - PxgParticleSystemGpu (pxgSlangParticleDesc): mSortedPositions_InvMass @24,
//     mSortedDeltaP @160, mSortedPhaseArray @40, mPhaseGroupToMaterialHandle @640,
//     mParticleMaterials @80, mParticleMaterialStride @632, mData_RestOffset @268,
//     mAccumDeltaP @152 — all pre-existing, re-validated against the .cu field
//     reads; NO new fields were needed.
//   - PxgSolverBodyData (@16 initialLinVelXYZ_invMassW) / PxgSolverTxIData (@28
//     sqrtInvInertia[9], column-major) / PxgAlignedTransform (@16 p) from
//     pxgSlangCommon.
//   - PxgPrePrepDescGpu (@272 solverBodyIndices), PxgConstraintPrepareDescGpu
//     (@168 body2WorldPool, @64 solverBodyDataPool, @72 solverBodyTxIDataPool)
//     from pxgSlangPrepDesc.
//   - PxgSolverCoreDescGpu (@0 outSolverVelocity, @268 numSolverBodies, @276
//     accumulatedBodyDeltaVOffset) + PxgSolverSharedDescPGSGpu (@160
//     iter_solverBodyVelPool) from pxgSlangSolverDesc.
//   - solvePCOutputDeltaV math transcribed from npPsSolvePC.slang (bit-identical
//     to particlesystem.cu:1421); computeBasisVectors + the material/phase byte
//     helpers transcribed from particleSelfCollisionBda.slang; PxgParticlePrimitive-
//     Contact mirror (PsContact, 32 bytes) as in psContactSortBda/psPrimCollide.
//
// PxNodeIndex decode (PxNodeIndex.h): rigidId is a PxU64 = { PxU32 mID (low), PxU32
// mLinkID (high) }.  index() = mID = rigidId.x; isStaticBody() = mID==0xffffffff;
// isArticulation() = mLinkID & 1 = rigidId.y & 1; articulationLinkId() = mLinkID>>1.
//
// CUDA DEVIATIONS (flagged):
//   - ARTICULATION branch DEFERRED.  The .cu prepare/solve each branch on
//     rigidId.isArticulation(); that path multiplies through the articulation
//     spatial-response matrix (sharedDesc->articulations[...].spatialResponseMatrixW
//     / linkBody2Worlds) and reads artiCoreDesc->{mMaxLinksPerArticulation,
//     nbArticulations} + solverCoreDesc->outArtiVelocity.  A byte-exact
//     PxgArticulation / PxgArticulationCoreDesc mirror is out of scope for this
//     tranche (the rigid/articulation REACTION kernels are a separate later
//     tranche).  The branch is preserved structurally but produces no write
//     (guarded by the per-lane isArticulation() condition, which is FALSE for every
//     contact these particle-vs-static-floor scenes generate); the artiCoreDesc /
//     outArtiVelocity reads the .cu hoists to the top of the solve are likewise
//     sunk INTO the dead branch, so for a rigid/static contact the pointers are
//     never dereferenced — identical observable result, and nothing hoists a BDA
//     load out of the guard.  artiCoreDesc is still carried in the push pack for
//     ABI parity.
//   - The CUDA block-copies the whole PxgParticleSystem into shared memory in some
//     particle kernels (a read-fanout optimisation); here every field is read
//     directly through Ptr<> (semantically identical, read-only descriptor).
//   - llvmpipe SAFETY: no 32-lane-warp assumption exists in these three kernels
//     (no ballot/shuffle/warp-scan — the constraint block is a plain SoA indexed by
//     workIndex&31, the solve is per-contact, the accumulate is a serial range
//     reduction).  Every BDA Ptr<> load sits behind the uniform-scalar guard
//     `workIndex < tNumContacts` (an early `return`, and there are NO group barriers
//     in any of the three kernels, so the divergent return is safe) so nothing
//     speculatively hoists out of a zero-trip grid-stride loop.  The PxU16
//     phase->material handle and the runtime-strided PxsParticleMaterialData are
//     read as 32-bit word + byte offset (no 16-bit storage capability needed), as
//     particleSim / particleSelfCollisionBda.

import pxgSlangParticleDesc;   // PxgParticleSystemGpu, PXG_PARTICLE_PHASE_GROUP_MASK
import pxgSlangCommon;         // PxgSolverBodyData, PxgSolverTxIData, PxgAlignedTransform
import pxgSlangPrepDesc;       // PxgPrePrepDescGpu, PxgConstraintPrepareDescGpu
import pxgSlangSolverDesc;     // PxgSolverCoreDescGpu, PxgSolverSharedDescPGSGpu

// PxgParticleSystemKernelBlockDim / GridDim (PxgParticleSystemCoreKernelIndices.h)
static const uint PC_BLOCK = 256u;   // PS_COLLISION block   (prepare + solve)
static const uint PC_GRID = 1024u;   // PS_COLLISION grid
static const uint AD_BLOCK = 512u;   // ACCUMULATE_DELTA block (accumulate)
static const uint AD_GRID = 32u;     // ACCUMULATE_DELTA grid
static const uint STATIC_NODE = 0xffffffffu;   // PX_INVALID_NODE

// PxgParticlePrimitiveContact (PxgParticleSystem.h, PX_ALIGN(16), 32 bytes):
// float4 normal_pen @0, PxU64 rigidId @16, PxU64 particleId @24.
struct PsContact
{
    float4 normal_pen; // @0
    uint2 rigidId;     // @16
    uint2 particleId;  // @24 -> 32
}

// PxgParticlePrimitiveConstraintBlock (PxgParticleSystemCore.h, 3072 bytes) — a
// SoA block of 32 lanes.  Field byte offsets within the block; each is written
// through a per-field Ptr<float4>/Ptr<uint2> view at [tIdx] (byte-offset views
// avoid relying on lvalue-through-Ptr struct-field writes).
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
// gravityScale@3, adhesionRadiusScale@4 (PxParticleGpu.h:47).
float loadMaterialFloat(uint64_t materialsBase, uint mi, uint stride, uint floatIndex)
{
    let elem = materialsBase + uint64_t(mi) * uint64_t(stride);
    return (Ptr<float>(elem))[floatIndex];
}

// PxComputeBasisVectors (PxMathUtils.h:297) — two unit tangents to `dir`.
float3 normalize3(float3 v)
{
    const float m = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (m > 0.0f)
    {
        v.x = v.x / m;
        v.y = v.y / m;
        v.z = v.z / m;
    }
    return v;
}
void computeBasisVectors(float3 dir, out float3 right, out float3 up)
{
    if (abs(dir.y) <= 0.9999f)
    {
        right = normalize3(float3(dir.z, 0.0f, -dir.x));
        up = float3(dir.y * right.z, dir.z * right.x - dir.x * right.z, -dir.y * right.x);
    }
    else
    {
        right = float3(1.0f, 0.0f, 0.0f);
        up = normalize3(float3(0.0f, dir.z, -dir.y));
    }
}

// PxMat33 (column-major, sqrtInvInertia[0..2]=col0, [3..5]=col1, [6..8]=col2)
// times PxVec3: M*v = col0*v.x + col1*v.y + col2*v.z.
float3 mat33MulVec3(float m[9], float3 v)
{
    return float3(m[0] * v.x + m[3] * v.y + m[6] * v.z,
                  m[1] * v.x + m[4] * v.y + m[7] * v.z,
                  m[2] * v.x + m[5] * v.y + m[8] * v.z);
}

// solvePCOutputDeltaV (particlesystem.cu:1421) — non-TGS particle-contact solve;
// bit-identical to npPsSolvePC.slang.
struct PcResult { float deltaF; float deltaFr0; float deltaFr1; float2 appliedForce; }
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

// ===========================================================================
// 1. ps_contactPrepareLaunch (particlesystem.cu:1238)
//    CUDA ABI (PxgParticleSystemCore.cpp:1609): (PxgParticleSystem* particleSystems,
//      PxgParticlePrimitiveContact* sortedContacts, PxU32* numContacts,
//      PxgParticlePrimitiveConstraintBlock* primitiveConstraints,
//      PxgPrePrepDesc* preDesc, PxgConstraintPrepareDesc* prepareDesc,
//      float2* appliedForces, PxReal dt, bool isTGS, float4* deltaVel,
//      PxgSolverSharedDescBase* sharedDesc)
//    -> {8,8,8,8,8,8,8,4,1,8,8} = 80-byte blob (dt@56, isTGS byte@60, deltaVel@64).
// ===========================================================================
struct PcPrepParams
{
    uint64_t particleSystems;      // @0
    uint64_t sortedContacts;       // @8
    uint64_t numContacts;          // @16
    uint64_t primitiveConstraints; // @24
    uint64_t preDesc;              // @32
    uint64_t prepareDesc;          // @40
    uint64_t appliedForces;        // @48
    float dt;                      // @56
    uint isTGSWord;                // @60 (bool byte0)
    uint64_t deltaVel;             // @64
    uint64_t sharedDesc;           // @72 -> 80
}
[[vk::push_constant]]
ConstantBuffer<PcPrepParams> gCp;

[shader("compute")]
[numthreads(PC_BLOCK, 1, 1)]
void ps_contactPrepareLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint tNumContacts = Ptr<uint>(gCp.numContacts)[0];

    const uint nbBlocksRequired = (tNumContacts + PC_BLOCK - 1u) / PC_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + PC_GRID - 1u) / PC_GRID;
    const uint idx = groupThreadId.x;

    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gCp.particleSystems);
    Ptr<PsContact> contacts = Ptr<PsContact>(gCp.sortedContacts);
    Ptr<float2> appliedForces = Ptr<float2>(gCp.appliedForces);
    Ptr<PxgConstraintPrepareDescGpu> prepDesc = Ptr<PxgConstraintPrepareDescGpu>(gCp.prepareDesc);
    Ptr<PxgPrePrepDescGpu> preDesc = Ptr<PxgPrePrepDescGpu>(gCp.preDesc);

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * PC_BLOCK + idx + nbIterationsPerBlock * groupId.x * PC_BLOCK;
        if (workIndex >= tNumContacts)
            return;

        appliedForces[workIndex] = float2(0.0f, 0.0f);

        const PsContact contact = contacts[workIndex];
        const uint tIdx = workIndex & 31u;
        const uint64_t blockBase = gCp.primitiveConstraints + uint64_t(workIndex / 32u) * uint64_t(CB_STRIDE);

        const uint64_t compressedId = uint64_t(contact.particleId.x) | (uint64_t(contact.particleId.y) << 32);
        const uint particleSystemId = uint(compressedId >> 32);
        const uint particleId = uint(compressedId & 0xffffffffu);

        Ptr<float4> position_invmass = Ptr<float4>(systems[particleSystemId].mSortedPositions_InvMass);
        const float4 pointInvM = position_invmass[particleId];
        const float invMass1 = pointInvM.w;
        const float3 point = float3(pointInvM.x, pointInvM.y, pointInvM.z);

        // constraint normal is the FLIPPED contact normal.
        const float3 normal = float3(-contact.normal_pen.x, -contact.normal_pen.y, -contact.normal_pen.z);
        const float pen = contact.normal_pen.w;

        // PxNodeIndex decode.
        const uint2 rigidId = contact.rigidId;
        const bool isStaticBody = (rigidId.x == STATIC_NODE);
        const bool isArticulation = (rigidId.y & 1u) != 0u;

        uint solverBodyIdx = 0u;
        if (!isStaticBody)
            solverBodyIdx = (Ptr<uint>(preDesc[0].solverBodyIndices))[rigidId.x];

        if (isArticulation)
        {
            // DEFERRED: articulation spatial-response prep (see file header).  No
            // contact in the particle-vs-static-floor scenes takes this branch;
            // the constraint is left untouched for such a (nonexistent) lane.
        }
        else
        {
            Ptr<PxgSolverBodyData> solverBodyData = Ptr<PxgSolverBodyData>(prepDesc[0].solverBodyDataPool);
            Ptr<PxgSolverTxIData> solverTxI = Ptr<PxgSolverTxIData>(prepDesc[0].solverBodyTxIDataPool);
            Ptr<PxgAlignedTransform> bodyFrames = Ptr<PxgAlignedTransform>(prepDesc[0].body2WorldPool);

            const float4 linVel_invMass0 = solverBodyData[solverBodyIdx].initialLinVelXYZ_invMassW;
            const float invMass0 = linVel_invMass0.w;

            float invSqrtInertia0[9];
            float inertiaScale = 1.0f;
            if (solverBodyIdx != 0u && invMass0 == 0.0f)
            {
                // PxMat33(PxIdentity)
                invSqrtInertia0[0] = 1.0f; invSqrtInertia0[1] = 0.0f; invSqrtInertia0[2] = 0.0f;
                invSqrtInertia0[3] = 0.0f; invSqrtInertia0[4] = 1.0f; invSqrtInertia0[5] = 0.0f;
                invSqrtInertia0[6] = 0.0f; invSqrtInertia0[7] = 0.0f; invSqrtInertia0[8] = 1.0f;
                inertiaScale = 0.0f;
            }
            else
            {
                invSqrtInertia0 = solverTxI[solverBodyIdx].sqrtInvInertia;
            }

            const float4 bodyFrame0p4 = bodyFrames[solverBodyIdx].p;
            const float3 bodyFrame0p = float3(bodyFrame0p4.x, bodyFrame0p4.y, bodyFrame0p4.z);

            float3 t0, t1;
            computeBasisVectors(normal, t0, t1);

            const float3 ra = point - bodyFrame0p;
            const float3 raXn = cross(ra, normal);
            const float3 raXF0 = cross(ra, t0);
            const float3 raXF1 = cross(ra, t1);

            const float3 raXnSqrtInertia = mat33MulVec3(invSqrtInertia0, raXn);
            const float resp0 = (dot(raXnSqrtInertia, raXnSqrtInertia)) * inertiaScale + invMass0;

            const float3 raXF0SqrtInertia = mat33MulVec3(invSqrtInertia0, raXF0);
            const float3 raXF1SqrtInertia = mat33MulVec3(invSqrtInertia0, raXF1);

            const float respF0 = (dot(raXF0SqrtInertia, raXF0SqrtInertia)) * inertiaScale + invMass0;
            const float respF1 = (dot(raXF1SqrtInertia, raXF1SqrtInertia)) * inertiaScale + invMass0;

            const float unitResponse = resp0 + invMass1;
            const float unitResponseF0 = respF0 + invMass1;
            const float unitResponseF1 = respF1 + invMass1;
            const float velMultiplier = (unitResponse > 0.0f) ? (1.0f / unitResponse) : 0.0f;
            const float velMultiplierF0 = (unitResponseF0 > 0.0f) ? (1.0f / unitResponseF0) : 0.0f;
            const float velMultiplierF1 = (unitResponseF1 > 0.0f) ? (1.0f / unitResponseF1) : 0.0f;

            Ptr<float4>(blockBase + CB_RAXN_VM)[tIdx] =
                float4(raXnSqrtInertia.x, raXnSqrtInertia.y, raXnSqrtInertia.z, velMultiplier);
            Ptr<float4>(blockBase + CB_NORMAL_ERR)[tIdx] = float4(normal.x, normal.y, normal.z, pen);
            Ptr<float4>(blockBase + CB_FRIC0_IM0)[tIdx] = float4(t0.x, t0.y, t0.z, invMass0);
            Ptr<float4>(blockBase + CB_RAXNF0_VM)[tIdx] =
                float4(raXF0SqrtInertia.x, raXF0SqrtInertia.y, raXF0SqrtInertia.z, velMultiplierF0);
            Ptr<float4>(blockBase + CB_RAXNF1_VM)[tIdx] =
                float4(raXF1SqrtInertia.x, raXF1SqrtInertia.y, raXF1SqrtInertia.z, velMultiplierF1);
            Ptr<uint2>(blockBase + CB_RIGIDID)[tIdx] = contact.rigidId;
            Ptr<uint2>(blockBase + CB_PARTICLEID)[tIdx] = contact.particleId;
        }

        if (gCp.deltaVel != 0)
        {
            Ptr<float4> deltaVel = Ptr<float4>(gCp.deltaVel);
            deltaVel[workIndex] = float4(0.0f, 0.0f, 0.0f, 0.0f);
            deltaVel[workIndex + tNumContacts] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
}

// ===========================================================================
// 2. ps_solvePCOutputParticleDeltaVLaunch (particlesystem.cu:1531, NON-TGS)
//    CUDA ABI (PxgParticleSystemCore.cpp:1948): (PxgParticleSystem* particleSystems,
//      PxgParticlePrimitiveContact* sortedContacts,
//      PxgParticlePrimitiveConstraintBlock* primitiveConstraints,
//      PxU32* numContacts, PxgPrePrepDesc* prePrepDesc,
//      PxgSolverCoreDesc* solverCoreDesc, PxgSolverSharedDesc<IterativeSolveData>*
//      sharedDesc, float4* deltaPos, float2* appliedForces, PxReal dt,
//      PxReal relaxationCoefficient, bool isVelocityIteration,
//      PxgArticulationCoreDesc* artiCoreDesc)
//    -> {8,8,8,8,8,8,8,8,8,4,4,1,8} = 96-byte blob (dt@72, coeff@76, bool@80,
//    artiCoreDesc@88).  sortedContacts is ABI parity only (the particle id is read
//    from the constraint block, not the sorted contact).
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
void ps_solvePCOutputParticleDeltaVLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgSolverCoreDescGpu> solverCoreDesc = Ptr<PxgSolverCoreDescGpu>(gSv.solverCoreDesc);
    Ptr<PxgSolverSharedDescPGSGpu> sharedDesc = Ptr<PxgSolverSharedDescPGSGpu>(gSv.sharedDesc);
    Ptr<PxgPrePrepDescGpu> prePrepDesc = Ptr<PxgPrePrepDescGpu>(gSv.prePrepDesc);

    const uint accOffset = solverCoreDesc[0].accumulatedBodyDeltaVOffset;
    Ptr<float4> solverBodyVelPool = Ptr<float4>(sharedDesc[0].iter_solverBodyVelPool);
    Ptr<float4> initialVel = Ptr<float4>(solverCoreDesc[0].outSolverVelocity);
    const uint numSolverBodies = solverCoreDesc[0].numSolverBodies;

    const uint tNumContacts = Ptr<uint>(gSv.numContacts)[0];

    const uint nbBlocksRequired = (tNumContacts + PC_BLOCK - 1u) / PC_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + PC_GRID - 1u) / PC_GRID;
    const uint idx = groupThreadId.x;

    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gSv.particleSystems);
    Ptr<float4> deltaPos = Ptr<float4>(gSv.deltaPos);
    Ptr<float2> appliedForces = Ptr<float2>(gSv.appliedForces);

    const float dt = gSv.dt;
    const float relaxation = gSv.relaxationCoefficient;

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
        const float frictionCoefficient = loadMaterialFloat(materials, mi, materialStride, 0u); // friction

        const float4 deltaP_invMassW = sortedDeltaP_invMassW[particleId];
        const float invMass1 = deltaP_invMassW.w;

        // Infinite-mass particles do not respond (CUDA `continue`).
        if (invMass1 == 0.0f)
            continue;

        const float3 delta = float3(deltaP_invMassW.x, deltaP_invMassW.y, deltaP_invMassW.z);

        const uint2 rigidId = Ptr<uint2>(blockBase + CB_RIGIDID)[tIdx];
        const bool isStaticBody = (rigidId.x == STATIC_NODE);
        const bool isArticulation = (rigidId.y & 1u) != 0u;

        uint solverBodyIndex = 0u;
        if (!isStaticBody)
            solverBodyIndex = (Ptr<uint>(prePrepDesc[0].solverBodyIndices))[rigidId.x];

        float4 linearVelocity;
        float4 angularVelocity;
        if (isArticulation)
        {
            // DEFERRED: articulation link velocity read (outArtiVelocity, artiOffset
            // = maxLinks*nbArticulations).  Never taken for rigid/static contacts;
            // the artiCoreDesc + outArtiVelocity pointers are only dereferenced here,
            // so a rigid/static contact never touches them (see file header).
            linearVelocity = float4(0.0f, 0.0f, 0.0f, 0.0f);
            angularVelocity = float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        else
        {
            linearVelocity = initialVel[solverBodyIndex] + solverBodyVelPool[accOffset + solverBodyIndex];
            angularVelocity = initialVel[solverBodyIndex + numSolverBodies]
                            + solverBodyVelPool[accOffset + solverBodyIndex + numSolverBodies];
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

        const float3 linVel0 = float3(linearVelocity.x, linearVelocity.y, linearVelocity.z);
        const float3 angVel0 = float3(angularVelocity.x, angularVelocity.y, angularVelocity.z);

        const float normalVel = dot(linVel0, normal) + dot(angVel0, raXn);
        const float tanVel0 = dot(linVel0, fric0) + dot(angVel0, raXnF0);
        const float tanVel1 = dot(linVel0, fric1) + dot(angVel0, raXnF1);

        const float adhesionRadiusScale = loadMaterialFloat(materials, mi, materialStride, 4u);
        const float adhesion = loadMaterialFloat(materials, mi, materialStride, 2u);
        const float restOffset = systems[particleSystemId].mData_RestOffset;

        PcResult res = solvePCOutputDeltaV(
            normal, error, delta,
            fric0, fric1,
            tanVel0, tanVel1,
            normalVel,
            velMultiplier, vmF0, vmF1,
            frictionCoefficient, restOffset,
            adhesion, adhesionRadiusScale,
            dt, appliedForce);

        const float3 deltaLinVel = (float3(-(normal.x * res.deltaF), -(normal.y * res.deltaF), -(normal.z * res.deltaF))
                                    + fric0 * res.deltaFr0 + fric1 * res.deltaFr1) * invMass1 * relaxation;
        const float scale = (res.deltaF != 0.0f) ? 1.0f : 0.0f;

        deltaPos[workIndex] = float4(deltaLinVel.x, deltaLinVel.y, deltaLinVel.z, scale);
        appliedForces[workIndex] = res.appliedForce;
    }
}

// ===========================================================================
// 3. ps_accumulateDeltaVParticleLaunch (particlesystem.cu:2828)
//    CUDA ABI (PxgParticleSystemCore.cpp:2005): (PxgParticleSystem* particleSystems,
//      PxgParticlePrimitiveContact* sortedContacts, PxU32* totalNumPairs,
//      PxU32* rangeStart, PxU32* rangeEnd, float4* deltaPos)
//    -> {8,8,8,8,8,8} = 48-byte blob.
// ===========================================================================
struct PcAccumParams
{
    uint64_t particleSystems; // @0  output (mAccumDeltaP)
    uint64_t sortedContacts;  // @8  input
    uint64_t totalNumPairs;   // @16 input
    uint64_t rangeStart;      // @24 input
    uint64_t rangeEnd;        // @32 input
    uint64_t deltaPos;        // @40 input -> 48
}
[[vk::push_constant]]
ConstantBuffer<PcAccumParams> gAc;

[shader("compute")]
[numthreads(AD_BLOCK, 1, 1)]
void ps_accumulateDeltaVParticleLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint numTests = Ptr<uint>(gAc.totalNumPairs)[0];

    const uint nbBlocksRequired = (numTests + AD_BLOCK - 1u) / AD_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + AD_GRID - 1u) / AD_GRID;
    const uint idx = groupThreadId.x;

    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gAc.particleSystems);
    Ptr<PsContact> contacts = Ptr<PsContact>(gAc.sortedContacts);
    Ptr<uint> rangeStart = Ptr<uint>(gAc.rangeStart);
    Ptr<uint> rangeEnd = Ptr<uint>(gAc.rangeEnd);
    Ptr<float4> deltaPos = Ptr<float4>(gAc.deltaPos);

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * AD_BLOCK + idx + nbIterationsPerBlock * groupId.x * AD_BLOCK;
        if (workIndex >= numTests)
            return;

        const uint startIndex = rangeStart[workIndex];
        const uint endIndex = rangeEnd[workIndex];

        float4 accumulatedDeltaP = float4(0.0f, 0.0f, 0.0f, 0.0f);
        float fcount = 0.0f;
        for (uint j = startIndex; j < endIndex; ++j)
        {
            const float4 tDelta = deltaPos[j];
            accumulatedDeltaP.x += tDelta.x;
            accumulatedDeltaP.y += tDelta.y;
            accumulatedDeltaP.z += tDelta.z;
            fcount += tDelta.w;
        }

        const float recipCount = 1.0f / pmax(1.0f, fcount);
        accumulatedDeltaP.x *= recipCount;
        accumulatedDeltaP.y *= recipCount;
        accumulatedDeltaP.z *= recipCount;

        const uint2 combinedParticleId = contacts[startIndex].particleId;
        const uint tParticleSystemId = combinedParticleId.y;
        const uint tParticleId = combinedParticleId.x;

        Ptr<float4> accumDeltaP = Ptr<float4>(systems[tParticleSystemId].mAccumDeltaP);
        float4 acc = accumDeltaP[tParticleId];
        acc.x += accumulatedDeltaP.x;
        acc.y += accumulatedDeltaP.y;
        acc.z += accumulatedDeltaP.z;
        acc.w += accumulatedDeltaP.w;
        accumDeltaP[tParticleId] = acc;
    }
}
