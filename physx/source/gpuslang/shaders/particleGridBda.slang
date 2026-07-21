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

// particleGridBda.slang
//
// P6.2 — the PBD particle spatial-hash grid build + reorder on the Vulkan
// engine ABI (BDA), ported line-for-line from
// gpusimulationcontroller/src/CUDA/particlesystem.cu:
//   - ps_calculateHashLaunch (particlesystem.cu:803): per-particle grid-cell
//     hash + identity sorted->unsorted seed.  Grid (numBlocks,
//     numActiveParticleSystems, 1); blockIdx.y selects the active system.  The
//     cell hash uses gridCal.cuh's calcGridPos (floor(pos/cellWidth)) and
//     calcGridHash (power-of-two wrap-around) transcribed inline below.
//   - ps_reorderDataAndFindCellStartLaunch (particlesystem.cu:898,
//     __launch_bounds__(1024,1)): gathers the sorted position/velocity/origin/
//     phase arrays through the permutation and computes mCellStart/mCellEnd via
//     a groupshared cell-boundary scan.  Grid (numBlocks, 1, 2); blockIdx.z==1
//     is the diffuse-particle slab (both paths ported faithfully).
//
// Differences from CUDA (recipe policy):
//   - the descriptor arrives as a device pointer to an array of
//     PxgParticleSystem read through Ptr<> (BDA) and its nested device-pointer
//     members are walked the same way; the CUDA blockCopy of the whole
//     descriptor into shared memory (a read-fanout optimisation) is replaced by
//     direct field loads (semantically identical — the copy is read-only);
//   - calcGridPos's floor is applied to the float pos.x/cellWidth ratio and
//     cast to int; CUDA promotes that same float ratio to double before floor,
//     which yields the IDENTICAL integer for every in-range position (the float
//     ratio is exactly representable in double), so this is bit-identical, not
//     a deviation;
//   - calcGridHash's `gridPos.x & (gridSize.x - 1)` (int & uint in CUDA, with
//     the int implicitly converted to uint) is reproduced with asuint(int) &
//     (uint), matching the two's-complement wrap for negative grid coordinates.
//
// No llvmpipe-safety guards are added: calcGridHash always yields a hash in
// [0, gridSizeX*gridSizeY*gridSizeZ) == [0, numCells), and the reorder gather
// index comes from a valid permutation (identity seed + radix sort, both now
// ported), so every array access is provably in range — matching CUDA exactly.

import pxgSlangParticleDesc;

// PxgParticleSystemKernelBlockDim::UPDATEGRID (PxgParticleSystemCoreKernelIndices.h)
static const uint PS_UPDATEGRID = 1024u;

// gridCal.cuh:34 calcGridPos — floor(pos / cellWidth) per axis, cast to int.
int3 calcGridPos(float4 particlePos, float cellWidth)
{
    int3 gridPos;
    gridPos.x = int(floor(particlePos.x / cellWidth));
    gridPos.y = int(floor(particlePos.y / cellWidth));
    gridPos.z = int(floor(particlePos.z / cellWidth));
    return gridPos;
}

// gridCal.cuh:46 calcGridHash — power-of-two wrap-around then linearise.
// gridSize.* are power-of-two, so `& (gridSize - 1)` wraps the (possibly
// negative) grid coordinate into [0, gridSize); asuint reproduces CUDA's
// implicit int->uint conversion (two's complement) before the mask.
uint calcGridHash(int3 gridPos, uint gridSizeX, uint gridSizeY, uint gridSizeZ)
{
    const uint gx = asuint(gridPos.x) & (gridSizeX - 1u);
    const uint gy = asuint(gridPos.y) & (gridSizeY - 1u);
    const uint gz = asuint(gridPos.z) & (gridSizeZ - 1u);
    return ((gz * gridSizeY) * gridSizeX) + (gy * gridSizeX) + gx;
}

// ---------------------------------------------------------------------------
// ps_calculateHashLaunch — cell hash + identity mapping (particlesystem.cu:803)
// ---------------------------------------------------------------------------
// argSizes (launch site PxgParticleSystemCore.cpp:636): { particleSystems*,
// activeParticleSystems* } -> {8,8} = 16-byte blob.
struct PsCalcHashParams
{
    uint64_t particleSystems;       // @0
    uint64_t activeParticleSystems; // @8 -> 16
}
[[vk::push_constant]]
ConstantBuffer<PsCalcHashParams> gHash;

[shader("compute")]
[numthreads(1024, 1, 1)]
void ps_calculateHashLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gHash.particleSystems);
    const uint particleId = (Ptr<uint>(gHash.activeParticleSystems))[groupId.y];

    const uint numParticles = systems[particleId].mCommonData_NumParticles;
    const float cellWidth = systems[particleId].mCommonData_GridCellWidth;

    Ptr<uint> hashArray = Ptr<uint>(systems[particleId].mGridParticleHash);
    Ptr<uint> particleIndex = Ptr<uint>(systems[particleId].mSortedToUnsortedMapping);

    const uint index = groupThreadId.x + groupId.x * PS_UPDATEGRID;

    if (index >= numParticles)
        return;

    Ptr<float4> pos_invmass = Ptr<float4>(systems[particleId].mUnsortedPositions_InvMass);

    const uint gridSizeX = systems[particleId].mCommonData_GridSizeX;
    const uint gridSizeY = systems[particleId].mCommonData_GridSizeY;
    const uint gridSizeZ = systems[particleId].mCommonData_GridSizeZ;

    const float4 pos = pos_invmass[index];

    const int3 gridPos = calcGridPos(pos, cellWidth);
    const uint hashValue = calcGridHash(gridPos, gridSizeX, gridSizeY, gridSizeZ);

    hashArray[index] = hashValue;
    particleIndex[index] = index;
}

// ---------------------------------------------------------------------------
// ps_reorderDataAndFindCellStartLaunch — gather + cell start/end
// (particlesystem.cu:898)
// ---------------------------------------------------------------------------
// argSizes (launch site PxgParticleSystemCore.cpp:681): { particleSystems*,
// PxU32 id } -> {8,4} = 12, rounded to 8-align -> 16-byte blob.
struct PsReorderParams
{
    uint64_t particleSystems;       // @0
    uint id;                        // @8 -> 16
}
[[vk::push_constant]]
ConstantBuffer<PsReorderParams> gReorder;

// __shared__ PxU32 sharedHash[UPDATEGRID + 1] — the cell-boundary scan window.
groupshared uint gsSharedHash[PS_UPDATEGRID + 1];

[shader("compute")]
[numthreads(1024, 1, 1)]
void ps_reorderDataAndFindCellStartLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gReorder.particleSystems);
    const uint id = gReorder.id;

    const uint index = groupThreadId.x + groupId.x * PS_UPDATEGRID;
    const bool isDiffuseParticlesThread = groupId.z == 1u;

    uint numParticles = systems[id].mCommonData_NumParticles;
    uint64_t gridParticleHashAddr = systems[id].mGridParticleHash;
    uint64_t gridParticleIndexAddr = systems[id].mSortedToUnsortedMapping;
    uint64_t cellStartAddr = systems[id].mCellStart;
    uint64_t cellEndAddr = systems[id].mCellEnd;
    uint64_t reverseLookupAddr = systems[id].mUnsortedToSortedMapping;

    if (isDiffuseParticlesThread)
    {
        // Early termination if there are no diffuse particles or is not using PBD
        if (systems[id].mCommonData_MaxDiffuseParticles == 0u)
            return;

        numParticles = uint((Ptr<int>(systems[id].mNumDiffuseParticles))[0]);
        gridParticleHashAddr = systems[id].mDiffuseGridParticleHash;
        gridParticleIndexAddr = systems[id].mDiffuseSortedToUnsortedMapping;
        cellStartAddr = systems[id].mDiffuseCellStart;
        cellEndAddr = systems[id].mDiffuseCellEnd;
        reverseLookupAddr = systems[id].mDiffuseUnsortedToSortedMapping;
    }

    Ptr<uint> gridParticleHash = Ptr<uint>(gridParticleHashAddr);
    Ptr<uint> gridParticleIndex = Ptr<uint>(gridParticleIndexAddr);
    Ptr<uint> cellStart = Ptr<uint>(cellStartAddr);
    Ptr<uint> cellEnd = Ptr<uint>(cellEndAddr);
    Ptr<uint> reverseLookup = Ptr<uint>(reverseLookupAddr);

    uint hash = 0u;

    if (index < numParticles)
    {
        hash = gridParticleHash[index];

        // Load hash data into shared memory so that we can look at neighbouring
        // particle's hash value without loading two hash values per thread.
        gsSharedHash[groupThreadId.x + 1] = hash;

        if (index > 0u && groupThreadId.x == 0u)
        {
            // first thread in block must load neighbour particle hash
            gsSharedHash[0] = gridParticleHash[index - 1u];
        }
    }

    GroupMemoryBarrierWithGroupSync();

    // common buffers
    uint64_t posAddr = systems[id].mUnsortedPositions_InvMass;
    uint64_t velAddr = systems[id].mUnsortedVelocities;
    uint64_t sortedPosAddr = systems[id].mSortedPositions_InvMass;
    uint64_t sortedVel0Addr = systems[id].mSortedVelocities;
    uint64_t originPosAddr = systems[id].mOriginPos_InvMass;
    uint64_t sortedOriginPosAddr = systems[id].mSortedOriginPos_InvMass;

    if (isDiffuseParticlesThread)
    {
        posAddr = systems[id].mDiffusePosition_LifeTime;
        velAddr = systems[id].mDiffuseVelocity;
        sortedPosAddr = systems[id].mDiffuseSortedPos_LifeTime;
        sortedVel0Addr = systems[id].mDiffuseSortedVel;
        originPosAddr = systems[id].mDiffuseOriginPos_LifeTime;
        sortedOriginPosAddr = systems[id].mDiffuseSortedOriginPos_LifeTime;
    }

    Ptr<float4> pos = Ptr<float4>(posAddr);
    Ptr<float4> vel = Ptr<float4>(velAddr);
    Ptr<float4> sortedPos = Ptr<float4>(sortedPosAddr);
    Ptr<float4> sortedVel0 = Ptr<float4>(sortedVel0Addr);
    Ptr<float4> origin_pos = Ptr<float4>(originPosAddr);
    Ptr<float4> sortedOriginPos = Ptr<float4>(sortedOriginPosAddr);

    // regular particles only
    Ptr<uint> phases = Ptr<uint>(systems[id].mUnsortedPhaseArray);
    Ptr<float4> sortedDeltaP = Ptr<float4>(systems[id].mSortedDeltaP);
    Ptr<uint> sortedPhases = Ptr<uint>(systems[id].mSortedPhaseArray);

    if (index < numParticles)
    {
        // Now use the sorted index to reorder the pos and vel data
        const uint sortedIndex = gridParticleIndex[index];

        const float4 p = pos[sortedIndex];
        const float4 v = vel[sortedIndex];

        const uint otherHash = gsSharedHash[groupThreadId.x];

        // If this particle has a different cell index to the previous particle
        // then it must be the first particle in the cell, so store the index of
        // this particle in the cell.  As it isn't the first particle, it must
        // also be the cell end of the previous particle's cell.
        if (index == 0u || hash != otherHash)
        {
            cellStart[hash] = index;

            if (index > 0u)
                cellEnd[otherHash] = index;
        }

        if (index == numParticles - 1u)
        {
            cellEnd[hash] = index + 1u;
        }

        sortedPos[index] = p;
        sortedVel0[index] = v;
        reverseLookup[sortedIndex] = index;

        const float4 orig = origin_pos[sortedIndex];
        sortedOriginPos[index] = orig;

        if (!isDiffuseParticlesThread)
        {
            sortedDeltaP[index] = float4(p.x - orig.x, p.y - orig.y, p.z - orig.z, p.w);
            sortedPhases[index] = phases[sortedIndex];
        }
    }
}
