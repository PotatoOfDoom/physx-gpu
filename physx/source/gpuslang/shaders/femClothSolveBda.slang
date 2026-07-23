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

// femClothSolveBda — P7.1 tranche 1: the FEM-cloth in-plane (membrane / ARAP +
// area) energy solve on the Vulkan engine ABI (BDA).  The two non-shared-triangle
// energy-solve kernels that make a hanging cloth resist stretching, ported
// line-for-line from gpusimulationcontroller/src/CUDA/FEMCloth.cu (the exact
// analog of the particle P6.2/6.3 solve tranches):
//   - cloth_solveNonSharedTriangleEnergyLaunch (FEMCloth.cu:669): one launch per
//     large ( >= CLOTH_SOLVESHELL ) partition; grid (numBlocks, nbActiveCloths, 1).
//   - cloth_solveNonSharedTriangleEnergyClusterLaunch (FEMCloth.cu:719): a single
//     launch that loops the small ( < CLOTH_SOLVESHELL ) "cluster" partitions with
//     a __syncthreads() between them; gridDim.x is always 1.
// Both drive the shared device fn updateNonSharedTriangle (FEMCloth.cu:610), which
// composes the in-plane leaf membraneSolve (= membraneEnergySolvePerTriangle,
// FEMClothUtil.cuh:437) imported from pxgSlangClothMembrane (the Phase-1-verified,
// bit-exact copy).
//
// Deviations from the CUDA original (recipe policy; each preserves the solve
// semantics and is exercised bit-exact by testFemClothSim.cpp):
//   - the descriptor arrives as a device pointer to an array of PxgFEMCloth read
//     through Ptr<> (BDA), reusing the byte-exact pxgSlangFEMClothDesc mirror;
//   - USE_BLOCK_COPY (the CUDA copies the whole PxgFEMCloth into shared memory,
//     then reads fields from that copy) is OMITTED: it is a pure read-cache of the
//     descriptor, which is immutable for the duration of these kernels (only the
//     pointed-to position/velocity/accumDelta/lambda buffers are mutated, and
//     those are read/written directly regardless).  Reading the pointer fields
//     straight through BDA is behaviourally identical (same precedent as the P7.0
//     femClothSimBda free-fall kernels);
//   - mMaterialIndices is a PxU16* — llvmpipe has no 16-bit storage, so the
//     per-triangle material handle is loaded by reading the containing 32-bit word
//     (buffer bases are >= 4-byte aligned) and extracting the correct half.  This
//     yields the identical PxU16 value;
//   - the cluster loop's __syncthreads() is emitted as GroupMemoryBarrierWith-
//     GroupSync().  Its trip count (mNbNonSharedTriPartitions, from the descriptor)
//     and start (clusterStartId push-constant) are uniform across the block, so the
//     barrier sits in uniform control flow — the per-thread `if (workIndex<endInd)`
//     work is done BEFORE the barrier exactly as in the CUDA.
//
// All float arithmetic is NoContraction-stamped (PXG_NOCONTRACT_SHADERS): the
// velocity update vel += delta*(1/dt) and the ARAP/area gradient sums would fuse
// to FMAs otherwise, breaking bitwise parity with the -ffp-contract=off reference.

import pxgSlangFEMClothDesc;   // PxgFEMClothGpu, PxsDeformableSurfaceMaterialGpu
import pxgSlangClothMembrane;  // membraneSolve (= membraneEnergySolvePerTriangle)

// PxgFEMClothKernelBlockDim::CLOTH_SOLVESHELL (PxgFEMClothCoreKernelIndices.h:41).
static const uint CLOTH_SOLVESHELL = 128u;

// The push-constant blob for BOTH kernels (identical param packs).  CUDA arg pack
// (PxgFEMClothCore.cpp:1556 / :1573): { PxgFEMCloth* femCloths, const PxU32*
// activeFEMCloths, const PxsDeformableSurfaceMaterialData* clothMaterials, PxReal
// dt, PxU32 partitionId|clusterStartId, bool isTGS } -> natural-alignment blob:
//   femCloths @0(8), activeFEMCloths @8(8), materials @16(8), dt @24(4),
//   partitionId @28(4), isTGS @32(1) -> rounded to the 8-alignment -> 40 bytes.
// (ABI row {8,8,8,4,4,1} in PxgVulkanKernelAbi.cpp.)
struct SolveNonSharedTriParams
{
    uint64_t femCloths;             // @0
    uint64_t activeFEMCloths;       // @8
    uint64_t materials;             // @16
    float dt;                       // @24
    uint partitionOrClusterId;      // @28  partitionId (single) / clusterStartId (cluster)
    uint isTGSWord;                 // @32  byte0 = isTGS
    uint _pad36;                    // @36  (isTGS's tail-pad -> struct rounds to 40)
}

[[vk::push_constant]]
ConstantBuffer<SolveNonSharedTriParams> gSolve;

// updateNonSharedTriangle (FEMCloth.cu:610): solve the in-plane membrane energy for
// the ordered non-shared triangle `index`, then fold the position change back into
// the velocity and accumulated-delta buffers.
void updateNonSharedTriangle(Ptr<PxgFEMClothGpu> femCloths, uint id, uint64_t materialsBase,
                             uint index, float dt, bool isTGS)
{
    Ptr<float4> positions = Ptr<float4>(femCloths[id].mPosition_InvMass);
    Ptr<float4> velocities = Ptr<float4>(femCloths[id].mVelocity_InvMass);
    Ptr<float4> accumulatedDeltaPos = Ptr<float4>(femCloths[id].mAccumulatedDeltaPos);

    const uint4 vertexIndex = Ptr<uint4>(femCloths[id].mOrderedNonSharedTriangleVertexIndices_triIndex)[index];

    // globalMaterialIndex = mMaterialIndices[vertexIndex.w] — a PxU16 load lowered
    // to a 32-bit word read + half extraction (no 16-bit storage on llvmpipe).
    const uint64_t matIdxBase = femCloths[id].mMaterialIndices;
    const uint byteOff = 2u * vertexIndex.w;
    const uint word = Ptr<uint>(matIdxBase)[byteOff >> 2];
    const uint globalMaterialIndex = ((byteOff >> 1) & 1u) != 0u ? (word >> 16) : (word & 0xffffu);

    Ptr<PxsDeformableSurfaceMaterialGpu> materials = Ptr<PxsDeformableSurfaceMaterialGpu>(materialsBase);
    const float youngs = materials[globalMaterialIndex].youngs;
    const float poissons = materials[globalMaterialIndex].poissons;
    const float thickness = materials[globalMaterialIndex].thickness;

    float4 x0 = positions[vertexIndex.x];
    float4 x1 = positions[vertexIndex.y];
    float4 x2 = positions[vertexIndex.z];

    const float4 xcp0 = x0;
    const float4 xcp1 = x1;
    const float4 xcp2 = x2;

    const float4 QInv = Ptr<float4>(femCloths[id].mOrderedNonSharedTriangleRestPoseInv)[index];

    // The per-triangle lambda slot (mOrderedNonSharedTriangleLambdas[index], a
    // float2) is read/written ONLY on the PGS path (!isTGS), exactly as the CUDA
    // membraneEnergySolvePerTriangle does.
    Ptr<float2> lambdas = Ptr<float2>(femCloths[id].mOrderedNonSharedTriangleLambdas);
    float2 lamSlot = float2(0.0f, 0.0f);
    if (!isTGS)
        lamSlot = lambdas[index];

    membraneSolve(x0, x1, x2, dt, youngs, poissons, thickness, QInv, 1.0f, 1.0f, 1.0f, lamSlot, isTGS);

    if (!isTGS)
        lambdas[index] = lamSlot;

    float4 delta0 = x0 - xcp0;
    float4 delta1 = x1 - xcp1;
    float4 delta2 = x2 - xcp2;

    float4 accumDelta0 = accumulatedDeltaPos[vertexIndex.x];
    float4 accumDelta1 = accumulatedDeltaPos[vertexIndex.y];
    float4 accumDelta2 = accumulatedDeltaPos[vertexIndex.z];

    float4 vel0 = velocities[vertexIndex.x];
    float4 vel1 = velocities[vertexIndex.y];
    float4 vel2 = velocities[vertexIndex.z];

    const float invDt = 1.0f / dt;
    vel0 += delta0 * invDt;
    vel1 += delta1 * invDt;
    vel2 += delta2 * invDt;

    accumDelta0 += delta0;
    accumDelta1 += delta1;
    accumDelta2 += delta2;

    positions[vertexIndex.x] = x0;
    positions[vertexIndex.y] = x1;
    positions[vertexIndex.z] = x2;

    accumulatedDeltaPos[vertexIndex.x] = accumDelta0;
    accumulatedDeltaPos[vertexIndex.y] = accumDelta1;
    accumulatedDeltaPos[vertexIndex.z] = accumDelta2;

    velocities[vertexIndex.x] = vel0;
    velocities[vertexIndex.y] = vel1;
    velocities[vertexIndex.z] = vel2;
}

// ---------------------------------------------------------------------------
// cloth_solveNonSharedTriangleEnergyLaunch — one large partition (FEMCloth.cu:669)
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(128, 1, 1)] // CLOTH_SOLVESHELL
void cloth_solveNonSharedTriangleEnergyLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgFEMClothGpu> femCloths = Ptr<PxgFEMClothGpu>(gSolve.femCloths);
    const uint id = Ptr<uint>(gSolve.activeFEMCloths)[groupId.y];
    const bool isTGS = (gSolve.isTGSWord & 0xffu) != 0u;
    const uint partitionId = gSolve.partitionOrClusterId;

    if (partitionId < femCloths[id].mNbNonSharedTriPartitions)
    {
        Ptr<uint> accum = Ptr<uint>(femCloths[id].mNonSharedTriAccumulatedPartitionsCP);
        const uint startInd = partitionId > 0u ? accum[partitionId - 1u] : 0u;
        const uint endInd = accum[partitionId];

        const uint groupThreadIdx = groupThreadId.x + groupId.x * CLOTH_SOLVESHELL;
        const uint workIndex = startInd + groupThreadIdx;

        if (workIndex < endInd)
        {
            updateNonSharedTriangle(femCloths, id, gSolve.materials, workIndex, gSolve.dt, isTGS);
        }
    }
}

// ---------------------------------------------------------------------------
// cloth_solveNonSharedTriangleEnergyClusterLaunch — the small partitions in one
// launch, looped with a barrier between them (FEMCloth.cu:719).  gridDim.x == 1.
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(128, 1, 1)] // CLOTH_SOLVESHELL
void cloth_solveNonSharedTriangleEnergyClusterLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgFEMClothGpu> femCloths = Ptr<PxgFEMClothGpu>(gSolve.femCloths);
    const uint id = Ptr<uint>(gSolve.activeFEMCloths)[groupId.y];
    const bool isTGS = (gSolve.isTGSWord & 0xffu) != 0u;
    const uint clusterStartId = gSolve.partitionOrClusterId;

    // Uniform loop bounds -> the barrier below is in uniform control flow.
    const uint nbPartitions = femCloths[id].mNbNonSharedTriPartitions;
    Ptr<uint> accum = Ptr<uint>(femCloths[id].mNonSharedTriAccumulatedPartitionsCP);

    for (uint partitionId = clusterStartId; partitionId < nbPartitions; ++partitionId)
    {
        const uint startInd = partitionId > 0u ? accum[partitionId - 1u] : 0u;
        const uint endInd = accum[partitionId];

        const uint groupThreadIdx = groupThreadId.x + groupId.x * CLOTH_SOLVESHELL;
        const uint workIndex = startInd + groupThreadIdx;

        if (workIndex < endInd)
        {
            updateNonSharedTriangle(femCloths, id, gSolve.materials, workIndex, gSolve.dt, isTGS);
        }

        GroupMemoryBarrierWithGroupSync();
    }
}
