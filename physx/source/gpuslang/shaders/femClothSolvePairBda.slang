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

// femClothSolvePairBda — P7.1 tranche 2: the FEM-cloth TRIANGLE-PAIR (shared shell
// energy + Discrete-Shells bending) solve on the Vulkan engine ABI (BDA).  The
// per-pair energy solve + the Jacobi averaging pass that make a hanging cloth
// resist BENDING, ported line-for-line from
// gpusimulationcontroller/src/CUDA/FEMCloth.cu (the direct sequel to the tranche-1
// non-shared in-plane solve in femClothSolveBda.slang):
//   - cloth_solveTrianglePairEnergyLaunch (FEMCloth.cu:861): one launch per large
//     ( >= CLOTH_SOLVESHELL ) shared-OR-non-shared tri-pair partition; grid
//     (numBlocks, nbActiveCloths, 1).
//   - cloth_solveTrianglePairEnergyClusterLaunch (FEMCloth.cu:1171): the single
//     launch that loops the small ( < CLOTH_SOLVESHELL ) "cluster" partitions with
//     a __syncthreads() between them; gridDim.x is always 1.
//   - cloth_averageTrianglePairVertsLaunch (FEMCloth.cu:522): the Jacobi averaging
//     pass; each shared vertex belongs to several pairs, each pair wrote its own
//     position COPY into the accumulated-poses region, and this pass folds them
//     back into the real position / velocity / accumulated-delta buffers by the
//     per-vertex reference count.  block CLOTH_STEP (1024).
// The first two drive the shared device fn updateTrianglePairs (FEMCloth.cu:768):
// the 4-vertex (x0,x1,x2,x3) COPIES/REMAP driver.  For a SHARED partition it calls
// clothSharedEnergySolvePerTrianglePair (FEMClothUtil.cuh:661 — membrane on BOTH
// triangles, imported bit-exact from pxgSlangClothMembrane, + bending); for a
// NON-shared partition it calls bendingEnergySolvePerTrianglePair
// (FEMClothUtil.cuh:543 — bending only), imported from pxgSlangClothBending.
//
// ------------------------------------------------------------------------------
// atan2 SEAM — what is bit-exact vs. tolerance (see pxgSlangClothBending header):
//   - the MEMBRANE half of the shared solve is FULLY bit-exact (imported verbatim
//     from the Phase-1-verified pxgSlangClothMembrane);
//   - the COPIES/REMAP assembly, vertex-reference-count gather, writeback, and the
//     Jacobi average are integer / strict-IEEE float and are bit-exact;
//   - the BENDING half evaluates atan2 ON DEVICE (no host to precompute the angle),
//     which is NOT cross-backend bit-exact.  Everything up to and including
//     sinAngle / cosAngle stays bit-exact; atan2 and everything downstream is a
//     TIGHT-TOLERANCE path.  The engine test compares the bending contribution
//     within a documented relative tolerance (and pins the whole downstream chain
//     bit-exact on a symmetric FLAT pair, where atan2(0,+x)==+0 is exact).
// ------------------------------------------------------------------------------
//
// Deviations from the CUDA original (recipe policy; each preserves the solve
// semantics and is exercised by testFemClothSim.cpp):
//   - the descriptor arrives as a device pointer to an array of PxgFEMCloth read
//     through Ptr<> (BDA), reusing the byte-exact pxgSlangFEMClothDesc mirror;
//   - USE_BLOCK_COPY (the CUDA read-caches the whole PxgFEMCloth into shared
//     memory) is OMITTED — the descriptor is immutable for these kernels (only the
//     pointed-to buffers are mutated); reading the pointer fields straight through
//     BDA is behaviourally identical (same precedent as femClothSolveBda);
//   - the two PxU16 material handles are packed inside restData1.y/.z as floats in
//     the descriptor (u32(restData1.y/z)) — no 16-bit load involved here;
//   - the CUDA writes only the .w half of a used copy slot back to zero
//     (`curTriPairPositions[i].w = 0`); Slang has no addressable component store
//     through a Ptr<>, so we write the FULL float4 with the already-loaded x/y/z
//     and w=0.  Each copy slot is owned by exactly one pair-vertex (the remap is a
//     permutation within the partition, and partitions are separated by a barrier),
//     so the read-modify-write of the unchanged x/y/z is identical to the partial
//     store;
//   - the per-pair lambda slots (membrane float2 @mOrderedSharedTriangleLambdas,
//     bending float @mShared/NonSharedBendingLambdas) are read up-front and written
//     back after the leaf.  On a leaf early-out (youngs<thr / kInv<=0 / x23==0) the
//     leaf leaves the slot unchanged, so the writeback stores the same bits it
//     read; the slot is per-pair-exclusive, so this is idempotent (identical
//     precedent to the tranche-1 non-shared membrane path);
//   - the cluster loop's __syncthreads() -> GroupMemoryBarrierWithGroupSync(); its
//     trip count (mNb{Shared,NonShared}TriPairPartitions) and the isSharedPartition
//     / hasActiveBending early-out are UNIFORM across the block, so the barrier sits
//     in uniform control flow exactly as the CUDA.
//
// All float arithmetic is NoContraction-stamped (PXG_NOCONTRACT_SHADERS): the
// membrane / bending gradient sums and the vel += delta*(1/dt) fuse to FMAs
// otherwise, breaking bitwise parity with the -ffp-contract=off reference.

import pxgSlangFEMClothDesc;   // PxgFEMClothGpu, PxsDeformableSurfaceMaterialGpu
import pxgSlangClothMembrane;  // membraneSolve (bit-exact in-plane leaf) + FEMCLOTH_THRESHOLD
import pxgSlangClothBending;   // bendingSolve (atan2-seam bending leaf)

// PxgFEMClothKernelBlockDim (PxgFEMClothCoreKernelIndices.h:40-41).
static const uint CLOTH_SOLVESHELL = 128u;
static const uint CLOTH_STEP = 1024u;

float femPabs(float a) { return a < 0.0f ? -a : a; }

// -------------------------------------------------------------------------------
// Push-constant blobs.
// -------------------------------------------------------------------------------
// The triangle-pair energy solve (BOTH launch + cluster).  CUDA arg pack
// (PxgFEMClothCore.cpp:1617 / :1637): { PxgFEMCloth* femCloths, const PxU32*
// activeFEMCloths, const PxsDeformableSurfaceMaterialData* clothMaterials, PxReal
// dt, PxU32 partitionId|firstClusterId, bool isSharedTrianglePair, bool isTGS } ->
// natural-alignment blob: femCloths @0(8), activeFEMCloths @8(8), materials @16(8),
// dt @24(4), partitionId @28(4), isSharedTrianglePair @32(1), isTGS @33(1) ->
// rounded to the 8-alignment -> 40 bytes.  (ABI row {8,8,8,4,4,1,1}.)
struct SolveTriPairParams
{
    uint64_t femCloths;             // @0
    uint64_t activeFEMCloths;       // @8
    uint64_t materials;             // @16
    float dt;                       // @24
    uint partitionOrClusterId;      // @28  partitionId (launch) / firstClusterId (cluster)
    uint flagsWord;                 // @32  byte0 = isSharedTrianglePair, byte1 = isTGS
    uint _pad36;                    // @36  (rounds the struct to 40)
}

// The Jacobi average pass.  CUDA arg pack (PxgFEMClothCore.cpp:1668):
// { PxgFEMCloth* gFEMCloths, const PxU32* activeFEMCloths, PxReal invDt,
//   bool isSharedPartition } -> femCloths @0(8), activeFEMCloths @8(8), dtInv @16(4),
// isSharedPartition @20(1) -> rounded to the 8-alignment -> 24 bytes.
// (ABI row {8,8,4,1}.)
struct AvgPairVertsParams
{
    uint64_t femCloths;             // @0
    uint64_t activeFEMCloths;       // @8
    float dtInv;                    // @16
    uint isSharedWord;              // @20  byte0 = isSharedPartition (rounds to 24)
}

[[vk::push_constant]]
ConstantBuffer<SolveTriPairParams> gSolve;   // used by the two solve entry points

[[vk::push_constant]]
ConstantBuffer<AvgPairVertsParams> gAvg;     // used by the average entry point

// -------------------------------------------------------------------------------
// clothSharedEnergySolvePerTrianglePair (FEMClothUtil.cuh:661): membrane on BOTH
// triangles (x2,x3,x0) and (x2,x3,x1) with QInv derived from the rest edges via
// cutil's reciprocal-multiply float4/scalar divide, threaded in place, then the
// bending term.  Positions are inout; the descriptor lambda plumbing lives here.
// -------------------------------------------------------------------------------
void bendingForPair(Ptr<PxgFEMClothGpu> femCloths, uint id, inout float4 x0, inout float4 x1,
                    inout float4 x2, inout float4 x3, float4 vrc, float dt, uint tpi,
                    bool isShared, bool isTGS)
{
    float4 rbfd;
    uint64_t lamBase;
    if (isShared)
    {
        rbfd = Ptr<float4>(femCloths[id].mOrderedSharedRestBendingAngle_flexuralStiffness_damping)[tpi];
        lamBase = femCloths[id].mSharedBendingLambdas;
    }
    else
    {
        rbfd = Ptr<float4>(femCloths[id].mOrderedNonSharedRestBendingAngle_flexuralStiffness_damping)[tpi];
        lamBase = femCloths[id].mNonSharedBendingLambdas;
    }
    const float restBendingAngle = rbfd.x;
    const float kInv = rbfd.y;

    Ptr<float> lambdas = Ptr<float>(lamBase);
    float lamSlot = 0.0f;
    if (!isTGS)
        lamSlot = lambdas[tpi];

    bendingSolve(x0, x1, x2, x3, vrc, restBendingAngle, kInv, dt, lamSlot, isTGS);

    if (!isTGS)
        lambdas[tpi] = lamSlot;
}

void clothSharedEnergySolvePerTrianglePair(Ptr<PxgFEMClothGpu> femCloths, uint id, uint64_t materialsBase,
                                           inout float4 x0, inout float4 x1, inout float4 x2, inout float4 x3,
                                           float4 vrc, float dt, uint tpi, bool isTGS)
{
    const float4 restData0 = Ptr<float4>(femCloths[id].mOrderedSharedRestEdge0_edge1)[tpi];
    const float4 restData1 = Ptr<float4>(femCloths[id].mOrderedSharedRestEdgeLength_material0_material1)[tpi];

    const uint globalMaterialIndex0 = uint(restData1.y);
    const uint globalMaterialIndex1 = uint(restData1.z);

    const float2 restEdge0 = float2(restData0.x, restData0.y);
    const float2 restEdge1 = float2(restData0.z, restData0.w);
    const float restSharedEdgeLength = restData1.x;

    const float det0 = restSharedEdgeLength * restEdge0.y;
    const float det1 = restSharedEdgeLength * restEdge1.y;

    Ptr<PxsDeformableSurfaceMaterialGpu> materials = Ptr<PxsDeformableSurfaceMaterialGpu>(materialsBase);
    Ptr<float2> sharedLambdas = Ptr<float2>(femCloths[id].mOrderedSharedTriangleLambdas);

    // In-plane constraint for triangle0 with vertex x2, x3, and x0.
    if (femPabs(det0) > FEMCLOTH_THRESHOLD)
    {
        const uint lambdaIndex = 2u * tpi;
        // cutil float4/scalar operator/: reciprocal-multiply per component.
        const float inv0 = 1.0f / det0;
        const float4 QInv0 = float4(restEdge0.y * inv0, 0.0f * inv0, -restEdge0.x * inv0, restSharedEdgeLength * inv0);

        const float youngs0 = materials[globalMaterialIndex0].youngs;
        const float poissons0 = materials[globalMaterialIndex0].poissons;
        const float thickness0 = materials[globalMaterialIndex0].thickness;

        float2 lamSlot = float2(0.0f, 0.0f);
        if (!isTGS)
            lamSlot = sharedLambdas[lambdaIndex];
        membraneSolve(x2, x3, x0, dt, youngs0, poissons0, thickness0, QInv0, vrc.z, vrc.w, vrc.x, lamSlot, isTGS);
        if (!isTGS)
            sharedLambdas[lambdaIndex] = lamSlot;
    }

    // In-plane constraint for triangle1 with vertex x2, x3, and x1.
    if (femPabs(det1) > FEMCLOTH_THRESHOLD)
    {
        const uint lambdaIndex = 2u * tpi + 1u;
        const float inv1 = 1.0f / det1;
        const float4 QInv1 = float4(restEdge1.y * inv1, 0.0f * inv1, -restEdge1.x * inv1, restSharedEdgeLength * inv1);

        const float youngs1 = materials[globalMaterialIndex1].youngs;
        const float poissons1 = materials[globalMaterialIndex1].poissons;
        const float thickness1 = materials[globalMaterialIndex1].thickness;

        float2 lamSlot = float2(0.0f, 0.0f);
        if (!isTGS)
            lamSlot = sharedLambdas[lambdaIndex];
        membraneSolve(x2, x3, x1, dt, youngs1, poissons1, thickness1, QInv1, vrc.z, vrc.w, vrc.y, lamSlot, isTGS);
        if (!isTGS)
            sharedLambdas[lambdaIndex] = lamSlot;
    }

    // Bending constraint for the triangle pair (shared partition).
    bendingForPair(femCloths, id, x0, x1, x2, x3, vrc, dt, tpi, true, isTGS);
}

// -------------------------------------------------------------------------------
// updateTrianglePairs (FEMCloth.cu:768): the 4-vertex copies/remap driver.
// -------------------------------------------------------------------------------
void updateTrianglePairs(Ptr<PxgFEMClothGpu> femCloths, uint id, uint64_t materialsBase,
                         uint workIndex, float dt, bool isShared, bool isTGS)
{
    uint64_t writeIndicesBase;
    uint64_t triPairAccumulatedCopiesBase;
    uint64_t vertexIndicesBase;
    uint nbTrianglePairs;

    if (isShared)
    {
        writeIndicesBase = femCloths[id].mSharedTriPairRemapOutputCP;
        triPairAccumulatedCopiesBase = femCloths[id].mSharedTriPairAccumulatedCopiesCP;
        vertexIndicesBase = femCloths[id].mOrderedSharedTrianglePairVertexIndices;
        nbTrianglePairs = femCloths[id].mNbSharedTrianglePairs;
    }
    else
    {
        writeIndicesBase = femCloths[id].mNonSharedTriPairRemapOutputCP;
        triPairAccumulatedCopiesBase = femCloths[id].mNonSharedTriPairAccumulatedCopiesCP;
        vertexIndicesBase = femCloths[id].mOrderedNonSharedTrianglePairVertexIndices;
        nbTrianglePairs = femCloths[id].mNbNonSharedTrianglePairs;
    }

    const uint4 vertexIndex = Ptr<uint4>(vertexIndicesBase)[workIndex];

    Ptr<float4> curTriPairPositions = Ptr<float4>(femCloths[id].mPosition_InvMassCP);
    Ptr<float4> positions = Ptr<float4>(femCloths[id].mPosition_InvMass);

    Ptr<uint> writeIndices = Ptr<uint>(writeIndicesBase);
    Ptr<uint> triPairAccumulatedCopies = Ptr<uint>(triPairAccumulatedCopiesBase);

    const uint remapIndex1 = workIndex + nbTrianglePairs;
    const uint remapIndex2 = remapIndex1 + nbTrianglePairs;
    const uint remapIndex3 = remapIndex2 + nbTrianglePairs;

    float4 x0 = curTriPairPositions[workIndex];
    float4 x1 = curTriPairPositions[remapIndex1];
    float4 x2 = curTriPairPositions[remapIndex2];
    float4 x3 = curTriPairPositions[remapIndex3];

    // If the w component of 'curTriPairPositions' is zero, the position values are
    // not yet assigned; use 'positions' for the first assignment.  If w is non-zero,
    // reset the used copy slot by writing its w back to zero (the CUDA writes only
    // .w; we write the full float4 with the already-loaded x/y/z + w=0 — see header).
    if (x0.w == 0.0f)
        x0 = positions[vertexIndex.x];
    else
        curTriPairPositions[workIndex] = float4(x0.x, x0.y, x0.z, 0.0f);

    if (x1.w == 0.0f)
        x1 = positions[vertexIndex.y];
    else
        curTriPairPositions[remapIndex1] = float4(x1.x, x1.y, x1.z, 0.0f);

    if (x2.w == 0.0f)
        x2 = positions[vertexIndex.z];
    else
        curTriPairPositions[remapIndex2] = float4(x2.x, x2.y, x2.z, 0.0f);

    if (x3.w == 0.0f)
        x3 = positions[vertexIndex.w];
    else
        curTriPairPositions[remapIndex3] = float4(x3.x, x3.y, x3.z, 0.0f);

    // vertexReferenceCounts[i] = number of pairs referencing vertexIndex[i].
    uint vi[4] = { vertexIndex.x, vertexIndex.y, vertexIndex.z, vertexIndex.w };
    float rc[4];
    for (uint i = 0; i < 4; ++i)
    {
        const uint triPairStartInd = vi[i] == 0u ? 0u : triPairAccumulatedCopies[vi[i] - 1u];
        const uint triPairEndInd = triPairAccumulatedCopies[vi[i]];
        rc[i] = float(triPairEndInd - triPairStartInd);
    }
    const float4 vertexReferenceCounts = float4(rc[0], rc[1], rc[2], rc[3]);

    if (isShared)
    {
        clothSharedEnergySolvePerTrianglePair(femCloths, id, materialsBase, x0, x1, x2, x3,
                                              vertexReferenceCounts, dt, workIndex, isTGS);
    }
    else
    {
        bendingForPair(femCloths, id, x0, x1, x2, x3, vertexReferenceCounts, dt, workIndex, false, isTGS);
    }

    const uint writeIndex0 = writeIndices[workIndex];
    const uint writeIndex1 = writeIndices[remapIndex1];
    const uint writeIndex2 = writeIndices[remapIndex2];
    const uint writeIndex3 = writeIndices[remapIndex3];

    curTriPairPositions[writeIndex0] = x0;
    curTriPairPositions[writeIndex1] = x1;
    curTriPairPositions[writeIndex2] = x2;
    curTriPairPositions[writeIndex3] = x3;
}

// -------------------------------------------------------------------------------
// cloth_solveTrianglePairEnergyLaunch — one large partition (FEMCloth.cu:861)
// -------------------------------------------------------------------------------
[shader("compute")]
[numthreads(128, 1, 1)] // CLOTH_SOLVESHELL
void cloth_solveTrianglePairEnergyLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgFEMClothGpu> femCloths = Ptr<PxgFEMClothGpu>(gSolve.femCloths);
    const uint id = Ptr<uint>(gSolve.activeFEMCloths)[groupId.y];
    const bool isSharedPartition = (gSolve.flagsWord & 0xffu) != 0u;
    const bool isTGS = ((gSolve.flagsWord >> 8u) & 0xffu) != 0u;
    const uint partitionId = gSolve.partitionOrClusterId;

    if (!isSharedPartition && femCloths[id].mNonSharedTriPair_hasActiveBendingWord == 0u)
        return;

    const uint maxPartitions = isSharedPartition ? femCloths[id].mNbSharedTriPairPartitions
                                                 : femCloths[id].mNbNonSharedTriPairPartitions;

    if (partitionId < maxPartitions)
    {
        Ptr<uint> accum = Ptr<uint>(isSharedPartition ? femCloths[id].mSharedTriPairAccumulatedPartitionsCP
                                                       : femCloths[id].mNonSharedTriPairAccumulatedPartitionsCP);
        const uint startInd = partitionId > 0u ? accum[partitionId - 1u] : 0u;
        const uint endInd = accum[partitionId];

        const uint groupThreadIdx = groupThreadId.x + groupId.x * CLOTH_SOLVESHELL;
        const uint workIndex = startInd + groupThreadIdx;

        if (workIndex < endInd)
        {
            updateTrianglePairs(femCloths, id, gSolve.materials, workIndex, gSolve.dt, isSharedPartition, isTGS);
        }
    }
}

// -------------------------------------------------------------------------------
// cloth_solveTrianglePairEnergyClusterLaunch — the small partitions in one launch,
// looped with a barrier between them (FEMCloth.cu:1171).  gridDim.x == 1.
// -------------------------------------------------------------------------------
[shader("compute")]
[numthreads(128, 1, 1)] // CLOTH_SOLVESHELL
void cloth_solveTrianglePairEnergyClusterLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgFEMClothGpu> femCloths = Ptr<PxgFEMClothGpu>(gSolve.femCloths);
    const uint id = Ptr<uint>(gSolve.activeFEMCloths)[groupId.y];
    const bool isSharedPartition = (gSolve.flagsWord & 0xffu) != 0u;
    const bool isTGS = ((gSolve.flagsWord >> 8u) & 0xffu) != 0u;
    const uint firstClusterId = gSolve.partitionOrClusterId;

    // Uniform early-out -> the barrier below stays in uniform control flow.
    if (!isSharedPartition && femCloths[id].mNonSharedTriPair_hasActiveBendingWord == 0u)
        return;

    const uint maxPartitions = isSharedPartition ? femCloths[id].mNbSharedTriPairPartitions
                                                 : femCloths[id].mNbNonSharedTriPairPartitions;
    Ptr<uint> accum = Ptr<uint>(isSharedPartition ? femCloths[id].mSharedTriPairAccumulatedPartitionsCP
                                                   : femCloths[id].mNonSharedTriPairAccumulatedPartitionsCP);

    for (uint partitionId = firstClusterId; partitionId < maxPartitions; ++partitionId)
    {
        const uint startInd = partitionId > 0u ? accum[partitionId - 1u] : 0u;
        const uint endInd = accum[partitionId];

        const uint groupThreadIdx = groupThreadId.x + groupId.x * CLOTH_SOLVESHELL;
        const uint workIndex = startInd + groupThreadIdx;

        if (workIndex < endInd)
        {
            updateTrianglePairs(femCloths, id, gSolve.materials, workIndex, gSolve.dt, isSharedPartition, isTGS);
        }

        GroupMemoryBarrierWithGroupSync();
    }
}

// -------------------------------------------------------------------------------
// cloth_averageTrianglePairVertsLaunch — the Jacobi averaging pass (FEMCloth.cu:522).
// block CLOTH_STEP (1024).
// -------------------------------------------------------------------------------
[shader("compute")]
[numthreads(1024, 1, 1)] // CLOTH_STEP
void cloth_averageTrianglePairVertsLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgFEMClothGpu> femCloths = Ptr<PxgFEMClothGpu>(gAvg.femCloths);
    const uint id = Ptr<uint>(gAvg.activeFEMCloths)[groupId.y];
    const bool isSharedPartition = (gAvg.isSharedWord & 0xffu) != 0u;
    const float invDt = gAvg.dtInv;

    if (!isSharedPartition && femCloths[id].mNonSharedTriPair_hasActiveBendingWord == 0u)
        return;

    const uint groupThreadIdx = groupThreadId.x + groupId.x * CLOTH_STEP;
    const uint nbVerts = femCloths[id].mNbVerts;

    if (groupThreadIdx < nbVerts)
    {
        Ptr<float4> positions = Ptr<float4>(femCloths[id].mPosition_InvMass);
        Ptr<float4> velocities = Ptr<float4>(femCloths[id].mVelocity_InvMass);
        Ptr<float4> accumulatedDeltaPos = Ptr<float4>(femCloths[id].mAccumulatedDeltaPos);

        Ptr<float4> triPairPositions = Ptr<float4>(femCloths[id].mPosition_InvMassCP);
        Ptr<uint> triPairAccumulatedCopies = Ptr<uint>(isSharedPartition ? femCloths[id].mSharedTriPairAccumulatedCopiesCP
                                                                         : femCloths[id].mNonSharedTriPairAccumulatedCopiesCP);

        const uint nbTrianglePairs = isSharedPartition ? femCloths[id].mNbSharedTrianglePairs
                                                       : femCloths[id].mNbNonSharedTrianglePairs;

        const uint triPairOffset = nbTrianglePairs * 4u;
        // triPairAccumulatedPoses = &triPairPositions[triPairOffset]; index into it
        // as triPairPositions[triPairOffset + j].

        const float4 pos_invMass = positions[groupThreadIdx];

        if (pos_invMass.w != 0.0f && nbTrianglePairs > 0u)
        {
            const uint triPairStartInd = groupThreadIdx == 0u ? 0u : triPairAccumulatedCopies[groupThreadIdx - 1u];
            const uint triPairEndInd = triPairAccumulatedCopies[groupThreadIdx];
            const uint triPairNbCopies = triPairEndInd - triPairStartInd;

            if (triPairNbCopies > 0u)
            {
                float4 slot0 = triPairPositions[triPairOffset + triPairStartInd];
                float4 triPairDiff = slot0 - pos_invMass;
                // Reset the used copy slot's w to zero (full float4 store; x/y/z
                // unchanged — see header note on the .w-only CUDA store).
                triPairPositions[triPairOffset + triPairStartInd] = float4(slot0.x, slot0.y, slot0.z, 0.0f);

                for (uint j = triPairStartInd + 1u; j < triPairEndInd; ++j)
                {
                    float4 slotj = triPairPositions[triPairOffset + j];
                    triPairDiff += slotj - pos_invMass;
                    triPairPositions[triPairOffset + j] = float4(slotj.x, slotj.y, slotj.z, 0.0f);
                }

                const float triPairScale = 1.0f / float(triPairNbCopies);
                float4 delta = triPairDiff * triPairScale;
                delta.w = 0.0f;

                const float4 pos = pos_invMass + delta;
                const float4 vel = velocities[groupThreadIdx] + delta * invDt;
                const float4 accumDelta = accumulatedDeltaPos[groupThreadIdx] + delta;

                positions[groupThreadIdx] = pos;
                velocities[groupThreadIdx] = vel;
                accumulatedDeltaPos[groupThreadIdx] = accumDelta;
            }
        }
    }
}
