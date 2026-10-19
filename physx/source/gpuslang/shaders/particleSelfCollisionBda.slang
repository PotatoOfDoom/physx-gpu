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

// particleSelfCollisionBda.slang
//
// P6.3 — the PBD particle self-collision + one-way-contact solve + delta-apply +
// velocity-recompute stage on the Vulkan engine ABI (BDA), ported line-for-line
// from gpusimulationcontroller/src/CUDA/particlesystem.cu:
//   - ps_selfCollisionLaunch (particlesystem.cu:1145) + its device helper
//     collideCell (particlesystem.cu:1049) and SelfCollideCache
//     (particlesystem.cu:1041): the 3x3x3 grid neighbour search.  Each particle
//     hashes its own sorted cell (gridCal.cuh calcGridPos/calcGridHash,
//     transcribed inline as in particleGridBda), walks the 27 surrounding cells
//     via mCellStart/mCellEnd, and records every in-range (distanceSq <=
//     cellWidthSq) non-self neighbour into mCollisionIndex + counts them into
//     mParticleSelfCollisionCount, zeroing mDensityCollisionImpulses.  blockDim
//     = 64 (PxgParticleSystemCore.cpp:1050), grid (numBlocks,1,1); the `id`
//     selects the system directly (no blockIdx.y).
//   - ps_solveOneWayContactDeltaVLaunch (particlesystem.cu, PhysX 5.12): solve the
//     accumulated per-particle one-way (particle-vs-static) contacts
//     (mOneWayContactInfos/mOneWayContactCount/mOneWayForces) with
//     solveRigidDbContact (pxgSlangDeformableAndParticleUtils) into a position delta
//     accumulated in mAccumDeltaP.  blockDim = PS_SOLVE = 256
//     (PxgParticleSystemCore.cpp:1370), grid (numBlocks,nbActive,1);
//     activeParticleSystems[blockIdx.y] selects the system.
//   - ps_applyDeltaLaunch (particlesystem.cu:3689): SOR-average the Jacobi delta
//     buffer mDelta into mAccumDeltaP and clear mDelta.  blockDim = 64
//     (PxgParticleSystemCore.cpp:1519), grid (numBlocks,nbActive,1).
//   - ps_updateParticleLaunch (particlesystem.cu, PhysX 5.12): recompute the sorted
//     velocity from mAccumDeltaP and clear it; outside velocity iterations also apply
//     it to the sorted positions and fold it into mSortedDeltaP.  blockDim = PS_SOLVE = 256 (PxgParticleSystemCore.cpp:1561),
//     grid (numBlocks,nbActive,1).
//
// Differences from CUDA (recipe policy):
//   - the descriptor arrives as a device pointer to an array of PxgParticleSystem
//     read through Ptr<> (BDA); its nested device-pointer members are walked the
//     same way.  The CUDA block-copy of the whole descriptor into shared memory
//     in the solve/apply/update kernels (a read-fanout optimisation) is replaced
//     by direct field loads (semantically identical — the copy is read-only);
//   - ps_selfCollisionLaunch's SelfCollideCache is a groupshared
//     [MaxCachedContacts=32][blockDim=64] uint scratch in which every thread
//     EXCLUSIVELY owns column threadIdx.x (collideCell writes
//     selfCollisions[k][threadIdx.x], the tail loop reads the same column) — there
//     is NO cross-thread sharing.  It is therefore lowered to a per-thread local
//     uint[32], and the accompanying __syncwarp() (which here only orders a
//     thread's own shared writes vs its own later reads, an ordering already
//     guaranteed for a local array) is dropped.  This also removes the sole
//     32-lane-warp assumption, so nothing depends on the llvmpipe subgroup width;
//   - collideCell's manual next-iteration prefetch (nextPhase/nextPos/
//     nextRestIndex) is a latency-hiding read that fetches the same values one
//     iteration early; it is transcribed as plain per-iteration loads
//     (bit-identical results);
//   - the PxU16 phase->material handle lookup folds into 32-bit loads (no
//     16-bit-storage capability required on llvmpipe), exactly as particleSim;
//   - the runtime-strided PxsParticleMaterialData is indexed by raw byte offset
//     (getParticleMaterial), matching the engine's stride-based deref.
//
// llvmpipe-safety: no extra bounds guards are added over CUDA.  calcGridHash
// always yields a hash in [0, numCells); cell contents are sorted particle
// indices in [0, numParticles); mSortedToUnsortedMapping values (used only under
// the selfCollideFilter branch) are a valid permutation in [0, numParticles);
// every per-particle solve/apply/update access is the thread's own index guarded
// by `< numParticles`.  The one-way-contact overflow scatter
// (mCollisionIndex[globalThreadIndex + k*numParticles], k < maxNeighborhood) is
// left unguarded exactly as CUDA — mCollisionIndex is engine-sized
// maxParticles*maxNeighborhood.

import pxgSlangParticleDesc;
import pxgSlangDeformableAndParticleUtils; // PxgRigidPart, PxgDeformablePart, solveRigidDbContact (one-way solve)

// PxgParticleSystemKernelBlockDim (PxgParticleSystemCoreKernelIndices.h):
static const uint PS_SELFCOLLISION_BLOCK = 64u;  // selfCollision + applyDelta launch block
static const uint PS_SOLVE = 256u;               // solveOneWay + updateParticle launch block

// SelfCollideCache::MaxCachedContacts (particlesystem.cu:1043).
static const uint PS_MAX_CACHED_CONTACTS = 32u;
// PxgParticleContactInfo::MaxStaticContactsPerParticle (PxgParticleSystem.h:133).
static const uint PS_MAX_STATIC_CONTACTS_PER_PARTICLE = 12u;

// PxParticlePhaseFlag (PxParticleSystemFlag.h:89-94).
static const uint PS_PHASE_SELF_COLLIDE = 0x00100000u;        // 1<<20
static const uint PS_PHASE_SELF_COLLIDE_FILTER = 0x00200000u; // 1<<21

// PxgParticleSystem.h:52 — #define EMPTY_CELL 0xffffffff.
static const uint PS_EMPTY_CELL = 0xffffffffu;

// ---------------------------------------------------------------------------
// gridCal.cuh transcriptions (identical to particleGridBda — see its header for
// the bit-identity rationale of the float floor + the asuint two's-complement
// wrap).
// ---------------------------------------------------------------------------
int3 calcGridPos(float4 particlePos, float cellWidth)
{
    int3 gridPos;
    gridPos.x = int(floor(particlePos.x / cellWidth));
    gridPos.y = int(floor(particlePos.y / cellWidth));
    gridPos.z = int(floor(particlePos.z / cellWidth));
    return gridPos;
}

uint calcGridHash(int3 gridPos, uint gridSizeX, uint gridSizeY, uint gridSizeZ)
{
    const uint gx = asuint(gridPos.x) & (gridSizeX - 1u);
    const uint gy = asuint(gridPos.y) & (gridSizeY - 1u);
    const uint gz = asuint(gridPos.z) & (gridSizeZ - 1u);
    return ((gz * gridSizeY) * gridSizeX) + (gy * gridSizeX) + gx;
}

// phaseToMat[group] where phaseToMat is a PxU16 array; a 32-bit load + shift/mask
// avoids requiring 16-bit storage on the software Vulkan target (as particleSim).
uint loadPhaseToMaterial(uint64_t handleBase, uint group)
{
    let word = (Ptr<uint>(handleBase))[group >> 1];
    return (word >> ((group & 1u) * 16u)) & 0xffffu;
}

// getParticleMaterial (PxgParticleSystem.h:64) — raw byte-offset deref with the
// runtime material stride.  PxsParticleMaterialData floats: friction@0,
// damping@1, adhesion@2, gravityScale@3, adhesionRadiusScale@4 (PxParticleGpu.h:47).
float loadMaterialFloat(uint64_t materialsBase, uint mi, uint stride, uint floatIndex)
{
    let elem = materialsBase + uint64_t(mi) * uint64_t(stride);
    return (Ptr<float>(elem))[floatIndex];
}

// ---------------------------------------------------------------------------
// ps_selfCollisionLaunch — 3x3x3 grid neighbour search (particlesystem.cu:1145)
// ---------------------------------------------------------------------------
// argSizes (launch site PxgParticleSystemCore.cpp:1043): { particleSystems*,
// PxU32 id } -> {8,4} = 12, rounded to the 8-alignment -> 16-byte blob (same
// shape as the P6.2 reorder pack).
struct PsSelfCollisionParams
{
    uint64_t particleSystems;       // @0
    uint id;                        // @8 -> 16
}
[[vk::push_constant]]
ConstantBuffer<PsSelfCollisionParams> gSelf;

// Per-thread lowering of SelfCollideCache column threadIdx.x (see header).
struct SelfCache { uint items[32]; }

// collideCell (particlesystem.cu:1049) — collide particle particleIndex0 against
// every particle in [startIndex, endIndex) of one grid cell, appending recorded
// neighbours to the cache (or, past MaxCachedContacts, straight to global
// mCollisionIndex).  Returns the running collided count.
uint collideCell(
    uint particleIndex0, uint startIndex, uint endIndex, uint maxNeighborhood,
    float contactDistanceSq, float filterDistanceSq, float3 p0,
    Ptr<float4> sortedPose, Ptr<float4> restPositions, uint stride,
    uint currentCount, Ptr<uint> collisionIndex, uint phase, Ptr<uint> sortedPhases,
    float3 restP, bool selfCollideFilter, Ptr<uint> particleIndices,
    inout SelfCache selfCache)
{
    uint numCollidedParticles = currentCount;

    if (startIndex != PS_EMPTY_CELL)
    {
        // Slot k of particle particleIndex0 lives at particleIndex0 + k * stride.  The
        // CUDA passes collisionIndex already offset by globalThreadIndex
        // (particlesystem.cu:1158) and starts at currentCount * stride (:1080); here
        // collisionIndex is the BASE pointer, so the column offset is added once, here.
        // Without it every neighbour past the 32 cached ones landed in particle 0's
        // column: dense fluid saw at most 32 neighbours, picked in the -z,-y,-x-first
        // cell order, under-counted its density on the +x/+y/+z side and drifted into
        // the +x/+z corner of SnippetPBFMultiMat's box -- >18000 particles in one grid
        // cell, which ps_primitivesCollisionLaunch and this kernel then walked serially.
        uint index = particleIndex0 + currentCount * stride;

        const bool selfCollide = (phase & PS_PHASE_SELF_COLLIDE) != 0u;

        for (uint particleIndex1 = startIndex;
             particleIndex1 < endIndex && numCollidedParticles < maxNeighborhood;
             particleIndex1++)
        {
            const uint phase2 = sortedPhases[particleIndex1];
            const float4 pos2 = sortedPose[particleIndex1];
            const uint particleId = selfCollideFilter ? particleIndices[particleIndex1] : 0u;

            if (particleIndex1 != particleIndex0)  // check not colliding with self
            {
                const bool sameGroup = (phase & PXG_PARTICLE_PHASE_GROUP_MASK)
                                    == (phase2 & PXG_PARTICLE_PHASE_GROUP_MASK);

                if (selfCollide || !sameGroup)
                {
                    const float3 p1 = float3(pos2.x, pos2.y, pos2.z);
                    const float3 _delta = p0 - p1;
                    const float distanceSq = dot(_delta, _delta);

                    if (distanceSq <= contactDistanceSq)
                    {
                        bool filtered = false;
                        if (selfCollideFilter && sameGroup)
                        {
                            const float4 rq = restPositions[particleId];
                            const float3 diff = float3(rq.x, rq.y, rq.z) - restP;
                            if (dot(diff, diff) <= filterDistanceSq)
                                filtered = true;
                        }

                        if (!filtered)
                        {
                            if (numCollidedParticles < PS_MAX_CACHED_CONTACTS)
                                selfCache.items[numCollidedParticles] = particleIndex1;
                            else
                                collisionIndex[index] = particleIndex1;
                            index += stride;
                            numCollidedParticles++;
                        }
                    }
                }
            }
        }
    }

    return numCollidedParticles;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ps_selfCollisionLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gSelf.particleSystems);
    const uint id = gSelf.id;

    const uint globalThreadIndex = groupThreadId.x + groupId.x * PS_SELFCOLLISION_BLOCK;

    const uint numParticles = systems[id].mCommonData_NumParticles;
    uint totalCollidedParticles = 0u;

    // collisionForces = mDensityCollisionImpulses + globalThreadIndex,
    // collisionIndices = mCollisionIndex + globalThreadIndex (base pointers; the
    // per-thread offset is folded into the index below).
    Ptr<float2> collisionForces = Ptr<float2>(systems[id].mDensityCollisionImpulses);
    Ptr<uint> collisionIndices = Ptr<uint>(systems[id].mCollisionIndex);

    SelfCache shContacts;

    if (globalThreadIndex < numParticles)
    {
        const float cellWidth = systems[id].mCommonData_GridCellWidth;
        const float cellWidthSq = cellWidth * cellWidth;

        Ptr<float4> sortedPose = Ptr<float4>(systems[id].mSortedPositions_InvMass);
        Ptr<uint> sortedPhase = Ptr<uint>(systems[id].mSortedPhaseArray);
        Ptr<float4> restPose = Ptr<float4>(systems[id].mRestArray);
        Ptr<uint> gridParticleIndex = Ptr<uint>(systems[id].mSortedToUnsortedMapping);
        Ptr<uint> particleSelfCollisionCounts = Ptr<uint>(systems[id].mParticleSelfCollisionCount);

        const uint phase = sortedPhase[globalThreadIndex];

        const float4 pos = sortedPose[globalThreadIndex];
        const float3 p0 = float3(pos.x, pos.y, pos.z);

        const int3 gridPos = calcGridPos(pos, cellWidth);

        const bool selfCollideFilter = (phase & PS_PHASE_SELF_COLLIDE_FILTER) != 0u;

        float3 restp = float3(0.0f, 0.0f, 0.0f);
        if (selfCollideFilter)
        {
            const float4 rp = restPose[gridParticleIndex[globalThreadIndex]];
            restp = float3(rp.x, rp.y, rp.z);
        }

        Ptr<uint> cellStart = Ptr<uint>(systems[id].mCellStart);
        Ptr<uint> cellEnd = Ptr<uint>(systems[id].mCellEnd);

        const float solidRestOffset = systems[id].mData_SolidRestOffset;
        const float filterDistanceSq = (2.01f * 2.01f) * solidRestOffset * solidRestOffset;
        const uint maxNeighborhood = systems[id].mCommonData_MaxNeighborhood;

        const uint gridSizeX = systems[id].mCommonData_GridSizeX;
        const uint gridSizeY = systems[id].mCommonData_GridSizeY;
        const uint gridSizeZ = systems[id].mCommonData_GridSizeZ;

        for (int z = -1; z <= 1; z++)
        {
            for (int y = -1; y <= 1; y++)
            {
                for (int x = -1; x <= 1; x++)
                {
                    const int3 neighbourPos = int3(gridPos.x + x, gridPos.y + y, gridPos.z + z);
                    const uint gridHash = calcGridHash(neighbourPos, gridSizeX, gridSizeY, gridSizeZ);

                    const uint startIndex = cellStart[gridHash];
                    if (startIndex != PS_EMPTY_CELL)
                    {
                        const uint endIndex = cellEnd[gridHash];
                        totalCollidedParticles = collideCell(
                            globalThreadIndex, startIndex, endIndex, maxNeighborhood, cellWidthSq,
                            filterDistanceSq, p0, sortedPose, restPose, numParticles,
                            totalCollidedParticles, collisionIndices, phase, sortedPhase, restp,
                            selfCollideFilter, gridParticleIndex, shContacts);
                    }
                }
            }
        }

        particleSelfCollisionCounts[globalThreadIndex] = totalCollidedParticles;
    }

    // (CUDA __syncwarp() dropped — see header; the cache is per-thread.)

    for (uint i = 0u, offset = 0u; i < totalCollidedParticles; ++i, offset += numParticles)
    {
        if (i < PS_MAX_CACHED_CONTACTS)
            collisionIndices[globalThreadIndex + offset] = shContacts.items[i];
        collisionForces[globalThreadIndex + offset] = float2(0.0f, 0.0f);
    }
}

float pmax(float a, float b) { return (a > b) ? a : b; }   // PxMax = selectMax

// ---------------------------------------------------------------------------
// ps_solveOneWayContactDeltaVLaunch -- one-way (particle vs static mesh) contact solve,
// PhysX 5.12 (particlesystem.cu).
// ---------------------------------------------------------------------------
// 5.12 routes every one-way contact through the deformable/particle solve core:
// solveRigidDbContact with adhesion against a NODELESS static rigid
// (PxgRigidPart::readNodelessStatic), the particle a single-vertex PxgDeformablePart.
// The 5.6.1 solvePCOutputDeltaVTGS helper and the biasCoefficient argument are gone;
// the pair's depenetration clamp is the particle system's mPenBiasClamp alone.  The
// per-particle delta is the scale-averaged sum of the contacts' position deltas (the
// velocity * dt equivalent in velocity iterations), added to mAccumDeltaP.xyz.
//
// argSizes (launch site PxgParticleSystemCore.cpp solveOneWayCollision):
// { particleSystems*, activeParticleSystems*, PxReal invDt, PxReal dt,
// bool isVelocityIteration } -> {8,8,4,4,1} -> 32-byte blob (bool at @24).
// Grid (numBlocks, nbActiveParticleSystems, 1), block PS_SOLVE.
struct PsSolveOneWayParams
{
    uint64_t particleSystems;       // @0
    uint64_t activeParticleSystems; // @8
    float invDt;                    // @16 (unused by the kernel body, kept for the pack)
    float dt;                       // @20
    uint isVelocityIterationWord;   // @24 (byte0 = bool) -> 32
}
[[vk::push_constant]]
ConstantBuffer<PsSolveOneWayParams> gSolve;

[shader("compute")]
[numthreads(256, 1, 1)]
void ps_solveOneWayContactDeltaVLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint particleId = (Ptr<uint>(gSolve.activeParticleSystems))[groupId.y];
    Ptr<PxgParticleSystemGpu> particleSystem = Ptr<PxgParticleSystemGpu>(gSolve.particleSystems
                                                   + uint64_t(particleId) * PXG_PARTICLE_SYSTEM_STRIDE);

    const float dt = gSolve.dt;
    const bool isVelocityIteration = (gSolve.isVelocityIterationWord & 0xffu) != 0u;

    const uint numParticles = particleSystem[0].mCommonData_NumParticles;
    const uint groupThreadIdx = groupThreadId.x + groupId.x * PS_SOLVE;

    if (groupThreadIdx >= numParticles)
        return;

    Ptr<float4> contacts = Ptr<float4>(particleSystem[0].mOneWayContactInfos); // PxgParticleContactInfo == float4 mNormal_PenW
    Ptr<uint> contactCounts = Ptr<uint>(particleSystem[0].mOneWayContactCount);

    Ptr<uint> phases = Ptr<uint>(particleSystem[0].mSortedPhaseArray);
    const uint64_t phaseToMat = particleSystem[0].mPhaseGroupToMaterialHandle;
    const uint64_t materials = particleSystem[0].mParticleMaterials;
    const uint materialStride = particleSystem[0].mParticleMaterialStride;

    const uint phase = phases[groupThreadIdx];
    const uint group = phase & PXG_PARTICLE_PHASE_GROUP_MASK;
    const uint mi = loadPhaseToMaterial(phaseToMat, group);

    const float frictionCoefficient = loadMaterialFloat(materials, mi, materialStride, PXG_PMAT_FRICTION);
    const float adhesionRadiusScale = loadMaterialFloat(materials, mi, materialStride, PXG_PMAT_ADHESION_RADIUS_SCALE);
    const float adhesion = loadMaterialFloat(materials, mi, materialStride, PXG_PMAT_ADHESION);
    const float restOffset = particleSystem[0].mData_RestOffset;

    Ptr<float2> appliedForces = Ptr<float2>(particleSystem[0].mOneWayForces);
    Ptr<float4> accumDeltaP = Ptr<float4>(particleSystem[0].mAccumDeltaP);
    Ptr<float4> deltaP = Ptr<float4>(particleSystem[0].mSortedDeltaP);

    const uint rawCount = contactCounts[groupThreadIdx];
    const uint contactCount = (PS_MAX_STATIC_CONTACTS_PER_PARTICLE < rawCount)
                            ? PS_MAX_STATIC_CONTACTS_PER_PARTICLE : rawCount;

    if (contactCount != 0u)
    {
        const float4 dp_ = deltaP[groupThreadIdx];
        const float3 posDelta = float3(dp_.x, dp_.y, dp_.z);

        // velocity * dt in velocity iterations, the position delta otherwise.
        const float3 relParticleDelta = readParticleLinDelta(particleSystem, groupThreadIdx, dp_, dt, isVelocityIteration);

        float denom = 0.0f;

        const float invMass = dp_.w;
        float3 thisDeltaP = float3(0.0f, 0.0f, 0.0f);
        for (uint c = 0u, offset = groupThreadIdx; c < contactCount; ++c, offset += numParticles)
        {
            const float4 normalPenW = contacts[offset];
            float2 appliedForce = appliedForces[offset];

            // normal is flipped to be the same as for constraints.
            const float3 normal = float3(-normalPenW.x, -normalPenW.y, -normalPenW.z);

            float3 tangent0, tangent1;
            dpComputeBasisVectors(normal, tangent0, tangent1);

            PxgRigidPart rigid = PxgRigidPart();
            rigid.readNodelessStatic();

            PxgDeformablePart particle = PxgDeformablePart();
            particle.dim = 3u;
            particle.linDelta = relParticleDelta;
            particle.posDelta = posDelta;
            particle.vertexInvMasses = float4(invMass, 0.0f, 0.0f, 0.0f);
            particle.bc = float4(1.0f, 0.0f, 0.0f, 0.0f);
            particle.friction = frictionCoefficient;
            particle.adhesion = adhesion;
            particle.adhesionRadius = restOffset * (adhesionRadiusScale - 1.0f);

            PxgDbContactPair contactPair = PxgDbContactPair();
            PxgDbSolveOutput solveOut = PxgDbSolveOutput();
            contactPair.normal = normal;
            contactPair.tangent0 = tangent0;
            contactPair.initPen = normalPenW.w;
            contactPair.maxPenBiasClamp = particleSystem[0].mData_PenBiasClamp;

            solveRigidDbContact(rigid, particle, contactPair, solveOut, appliedForce.x, appliedForce.y, dt,
                                /*wasActive*/ true, /*checkOnlyActivity*/ false, isVelocityIteration,
                                /*withAdhesion*/ true);
            appliedForce.x = solveOut.accumulatedDeltaLambdaN;
            appliedForce.y += solveOut.deltaLambdaT;

            float3 delta;
            solveOut.computeDelta(contactPair, delta, dt);
            const float scale = (solveOut.deltaLambdaN != 0.0f || solveOut.deltaLambdaT != 0.0f) ? 1.0f : 0.0f;

            thisDeltaP = float3(thisDeltaP.x + delta.x * invMass, thisDeltaP.y + delta.y * invMass,
                                thisDeltaP.z + delta.z * invMass);

            denom += scale;
            appliedForces[offset] = appliedForce;
        }

        if (denom != 0.0f)
        {
            const float r = 1.0f / denom;
            thisDeltaP = float3(thisDeltaP.x * r, thisDeltaP.y * r, thisDeltaP.z * r);
        }

        float4 delta = accumDeltaP[groupThreadIdx];
        delta.x += thisDeltaP.x;
        delta.y += thisDeltaP.y;
        delta.z += thisDeltaP.z;
        accumDeltaP[groupThreadIdx] = delta;
    }
}

// ---------------------------------------------------------------------------
// ps_applyDeltaLaunch — SOR-average mDelta into mAccumDeltaP (particlesystem.cu:3689)
// ---------------------------------------------------------------------------
// argSizes (launch site PxgParticleSystemCore.cpp:1526): { particleSystems*,
// activeParticleSystems*, PxReal dt } -> {8,8,4} -> 24-byte blob.
struct PsApplyDeltaParams
{
    uint64_t particleSystems;       // @0
    uint64_t activeParticleSystems; // @8
    float dt;                       // @16 (unused by the kernel body) -> 24
}
[[vk::push_constant]]
ConstantBuffer<PsApplyDeltaParams> gApply;

[shader("compute")]
[numthreads(64, 1, 1)]
void ps_applyDeltaLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const float damp = 1.0f;

    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gApply.particleSystems);
    const uint particleId = (Ptr<uint>(gApply.activeParticleSystems))[groupId.y];

    const float relaxation = systems[particleId].mData_RelaxationFactor;

    const uint numParticles = systems[particleId].mCommonData_NumParticles;
    Ptr<float4> deltas = Ptr<float4>(systems[particleId].mDelta);
    Ptr<float4> accumDeltaP = Ptr<float4>(systems[particleId].mAccumDeltaP);

    const uint p = groupId.x * PS_SELFCOLLISION_BLOCK + groupThreadId.x;

    if (p < numParticles)
    {
        const float4 j = deltas[p];
        const float4 accumDltaP = accumDeltaP[p];

        // Equal to (1/relaxation)/dt scaling: 1 / max(j.w*relaxation, damp).
        const float scale = 1.0f / pmax(j.w * relaxation, damp);

        const float3 deltaPos = float3(j.x, j.y, j.z);
        const float3 averagedPos = deltaPos * scale;

        accumDeltaP[p] = float4(accumDltaP.x + averagedPos.x, accumDltaP.y + averagedPos.y,
                                accumDltaP.z + averagedPos.z, 0.0f);
        deltas[p] = float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
}

// ---------------------------------------------------------------------------
// ps_updateParticleLaunch -- apply mAccumDeltaP, recompute the sorted velocity
// (particlesystem.cu, PhysX 5.12)
// ---------------------------------------------------------------------------
// 5.12 renames skipNewPositionAdjustment to isVelocityIteration and moves the
// mSortedDeltaP update under it: a velocity iteration writes only the velocity
// correction, the position and the accumulated delta are skipped together.
//
// argSizes (launch site PxgParticleSystemCore.cpp updateSortedVelocity):
// { particleSystems*, activeParticleSystems*, PxReal invDt, bool isVelocityIteration }
// -> {8,8,4,1} -> 24-byte blob (bool at @20).
struct PsUpdateParticleParams
{
    uint64_t particleSystems;       // @0
    uint64_t activeParticleSystems; // @8
    float invDt;                    // @16
    uint isVelocityIterationWord;   // @20 (byte0 = bool) -> 24
}
[[vk::push_constant]]
ConstantBuffer<PsUpdateParticleParams> gUpdate;

[shader("compute")]
[numthreads(256, 1, 1)]
void ps_updateParticleLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gUpdate.particleSystems);
    const uint particleId = (Ptr<uint>(gUpdate.activeParticleSystems))[groupId.y];

    const float invDt = gUpdate.invDt;
    const bool isVelocityIteration = (gUpdate.isVelocityIterationWord & 0xffu) != 0u;

    const uint numParticles = systems[particleId].mCommonData_NumParticles;
    const uint groupThreadIdx = groupThreadId.x + groupId.x * PS_SOLVE;

    if (groupThreadIdx < numParticles)
    {
        Ptr<float4> accumDeltaP = Ptr<float4>(systems[particleId].mAccumDeltaP);
        Ptr<float4> sortedVelocity = Ptr<float4>(systems[particleId].mSortedVelocities);
        Ptr<float4> sortedDeltaP = Ptr<float4>(systems[particleId].mSortedDeltaP);
        Ptr<float4> sortedNewPos = Ptr<float4>(systems[particleId].mSortedPositions_InvMass);

        const float4 aDeltaP = accumDeltaP[groupThreadIdx];
        const float4 oldVel = sortedVelocity[groupThreadIdx];
        const float4 newVel = oldVel + aDeltaP * invDt;

        const float4 dp = sortedDeltaP[groupThreadIdx];

        const float4 sortedPos = sortedNewPos[groupThreadIdx];
        float4 sortedNewP = sortedPos;

        if (!isVelocityIteration)
        {
            sortedNewP += aDeltaP;
            sortedDeltaP[groupThreadIdx] = dp + aDeltaP;
        }

        accumDeltaP[groupThreadIdx] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        sortedVelocity[groupThreadIdx] = newVel;
        sortedNewPos[groupThreadIdx] = float4(sortedNewP.x, sortedNewP.y, sortedNewP.z, sortedPos.w);
    }
}
