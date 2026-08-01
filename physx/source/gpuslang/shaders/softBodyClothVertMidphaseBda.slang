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

// ===========================================================================
// softBodyClothVertMidphaseBda — sb_clothVertMidphaseGeneratePairsLaunch
// (softbodyMidPhase.cu:494, core at :411), on the engine BDA ABI.
//
// The OTHER half of softbody-vs-cloth midphase, and it runs the opposite way round from
// step 1: there a softbody TETRAHEDRON queried the cloth's triangle tree; here a cloth VERTEX
// queries the SOFTBODY's tetrahedron tree.  Both are needed because a tet-vs-triangle test
// misses a cloth vertex that has poked through into a tet's interior without any triangle of
// its coming near a tet face.
//
// It is much the simpler of the two.  The traverser is SoftbodyBoxTraverser
// (softbodyMidPhase.cu:253), whose whole job is AABB-vs-AABB:
//
//   * the query box is the cloth vertex inflated by the summed contact distance in every
//     direction -- pos +/- PxVec3(cDistance), a CUBE, not a sphere;
//   * an internal node's child is descended when its box overlaps that cube;
//   * a leaf primitive passes when isValidTet holds AND the tet's own AABB overlaps the cube.
//     The tet AABB is the plain min/max over its four corners, in WORLD space -- unlike
//     step 1, nothing is transformed anywhere, because both meshes are already world-space
//     and the tet tree is the softbody's own.
//
// EVERY SURVIVING PRIMITIVE IS WRITTEN, with no top-N reduction: pushOntoStackFullWarp
// (bv32Traversal.cuh:197) ballots the warp, takes ONE atomicAdd of the population count, and
// each qualifying lane stores at its scan offset, dropping the write (but NOT the count) when
// the stack is full.  Step 1's four-closest cap has no counterpart here.
//
// THE BROADPHASE EARLY-OUT IS TRANSCRIBED, INCLUDING ITS SIGN.  When the cloth's collision
// pairs are updated more than once per timestep the broadphase has already inflated the
// cloth's bounds, so this kernel DEFLATES them again by (n-1) * contactDistance -- minimum
// += , maximum -= , which shrinks the box -- and skips the whole pair when the shrunken box
// no longer meets the softbody's.  Reading that as an inflation inverts the test.
//
// GRID.  gridDim = (SB_SBMIDPHASE = 1024, numTests, 1), blockDim = (32, 2, 1)
// (PxgNarrowphaseCore.cpp:4761): blockIdx.y is the contact-manager index and the block's two
// warps are walked serially here, keeping the CUDA stride blockDim.y * gridDim.x.
// ===========================================================================
module softBodyClothVertMidphaseBda;

import pxgSlangSoftBodyDesc;
import pxgSlangFEMClothDesc;

static const uint MIDPHASE_WARPS_PER_BLOCK = 2u;
static const uint SB_SBMIDPHASE = 1024u;
static const uint SC_NB_WARPS = SB_SBMIDPHASE * MIDPHASE_WARPS_PER_BLOCK;

static const uint WARP = 32u;

// Gu::BV32DataPacked: 1168 B — mMin[32] @0, mMax[32] @512, mData[32] @1024, mNbNodes @1152.
static const uint64_t BV32_STRIDE = 1168UL;
static const uint64_t BV32_MMAX_OFFSET = 512UL;
static const uint64_t BV32_MDATA_OFFSET = 1024UL;
static const uint64_t BV32_MNBNODES_OFFSET = 1152UL;
static const uint GU_BV4_CHILD_OFFSET_SHIFT_COUNT = 11u;

static const uint STACK_SIZE = 192u;
static const uint GEOM_TETRAHEDRONMESH = 7u;

// ---------------------------------------------------------------------------
// Push-constant blob — BYTE-IDENTICAL to the CUDA pack (PxgNarrowphaseCore.cpp:4742):
// { PxReal toleranceLength, cmInputs*, transformCache*, bounds*, contactDistance*,
//   gpuShapes*, softbodies*, clothes*, PxU32 stackSizeBytes,
//   PxReal nbCollisionPairUpdatesPerTimestep, stackPtr*, midphasePairsNum* }.
// 4 @0 (padded to 8), seven pointers @8..@56, PxU32 @64, PxReal @68, then a pointer realigned
// to @72 and @80 -> 88-byte blob.  ABI row {4,8,8,8,8,8,8,8,4,4,8,8}.
//
// NOTE how this differs from step 1's pack, which is also 88 bytes: there the PxU32 at @64 is
// followed by a POINTER, so the pointer realigns to @72 and @68 is dead padding.  Here @68
// carries a real PxReal.  Two packs of the same size with different contents -- the ABI row is
// what tells them apart, and getting it wrong shifts every pointer after it.
// ---------------------------------------------------------------------------
struct SbClothVertMidParams
{
    float toleranceLength;          // @0   (ABI parity only)
    uint _pad4;                     // @4
    uint64_t cmInputs;              // @8
    uint64_t transformCache;        // @16  (ABI parity only — nothing is transformed here)
    uint64_t bounds;                // @24
    uint64_t contactDistance;       // @32
    uint64_t gpuShapes;             // @40
    uint64_t softbodies;            // @48
    uint64_t clothes;               // @56
    uint stackSizeBytes;            // @64
    float nbCollisionPairUpdatesPerTimestep; // @68
    uint64_t stackPtr;              // @72
    uint64_t midphasePairsNum;      // @80 -> 88
}

[[vk::push_constant]] ConstantBuffer<SbClothVertMidParams> gVm;

struct PxgContactManagerInput { uint shapeRef0, shapeRef1, transformCacheRef0, transformCacheRef1; }

struct PxgShape
{
    float scaleX, scaleY, scaleZ;
    float scaleRotX, scaleRotY, scaleRotZ, scaleRotW;
    uint materialIndex;
    uint2 hullOrMeshOffset;
    uint type;
    uint particleOrSoftbodyId;
}

// ---------------------------------------------------------------------------
// BV32DataPacked field readers.
// ---------------------------------------------------------------------------
uint64_t bvNodeBase(uint64_t treeAddr, uint nodeIdx) { return treeAddr + uint64_t(nodeIdx) * BV32_STRIDE; }
float3 bvMin(uint64_t treeAddr, uint nodeIdx, uint k)
{
    return Ptr<float4>(bvNodeBase(treeAddr, nodeIdx) + uint64_t(k) * 16UL)[0].xyz;
}
float3 bvMax(uint64_t treeAddr, uint nodeIdx, uint k)
{
    return Ptr<float4>(bvNodeBase(treeAddr, nodeIdx) + BV32_MMAX_OFFSET + uint64_t(k) * 16UL)[0].xyz;
}
uint bvData(uint64_t treeAddr, uint nodeIdx, uint k)
{
    return Ptr<uint>(bvNodeBase(treeAddr, nodeIdx) + BV32_MDATA_OFFSET + uint64_t(k) * 4UL)[0];
}
uint bvNbNodes(uint64_t treeAddr, uint nodeIdx)
{
    return Ptr<uint>(bvNodeBase(treeAddr, nodeIdx) + BV32_MNBNODES_OFFSET)[0];
}

// PxBounds3 (24 B): PxVec3 minimum @0, PxVec3 maximum @12.
void readBounds(uint64_t base, uint idx, out float3 mn, out float3 mx)
{
    Ptr<float> f = Ptr<float>(base + uint64_t(idx) * 24UL);
    mn = float3(f[0], f[1], f[2]);
    mx = float3(f[3], f[4], f[5]);
}

// PxBounds3::intersects — touching counts, so the comparisons are NOT strict.
bool boundsIntersect(float3 aMin, float3 aMax, float3 bMin, float3 bMax)
{
    return !(bMin.x > aMax.x || aMin.x > bMax.x ||
             bMin.y > aMax.y || aMin.y > bMax.y ||
             bMin.z > aMax.z || aMin.z > bMax.z);
}

bool isValidTet(float3 v0, float3 v1, float3 v2, float3 v3)
{
    return dot(v1 - v0, cross(v2 - v0, v3 - v0)) > 1e-9f;
}

void loadSoftbodyClothPair(uint cmIdx, out PxgShape sbShape, out uint sbCacheRef,
                           out PxgShape clothShape, out uint clothCacheRef)
{
    const PxgContactManagerInput cin = Ptr<PxgContactManagerInput>(gVm.cmInputs)[cmIdx];
    Ptr<PxgShape> shapes = Ptr<PxgShape>(gVm.gpuShapes);
    uint sRefSb = cin.shapeRef0, cRefSb = cin.transformCacheRef0;
    uint sRefCl = cin.shapeRef1, cRefCl = cin.transformCacheRef1;
    if (shapes[cin.shapeRef0].type != GEOM_TETRAHEDRONMESH)
    {
        sRefSb = cin.shapeRef1; cRefSb = cin.transformCacheRef1;
        sRefCl = cin.shapeRef0; cRefCl = cin.transformCacheRef0;
    }
    sbShape = shapes[sRefSb]; sbCacheRef = cRefSb;
    clothShape = shapes[sRefCl]; clothCacheRef = cRefCl;
}

// ---------------------------------------------------------------------------
groupshared int sVmStack[STACK_SIZE];
groupshared uint sVmBallot;
groupshared uint sVmIndex0;

uint vmBallot(bool pred, uint lane)
{
    if (lane == 0u) sVmBallot = 0u;
    GroupMemoryBarrierWithGroupSync();
    if (pred) InterlockedOr(sVmBallot, 1u << lane);
    GroupMemoryBarrierWithGroupSync();
    return sVmBallot;
}

// ===========================================================================
[shader("compute")]
[numthreads(32, 1, 1)]
void sb_clothVertMidphaseGeneratePairsLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;
    const uint cmIdx = gid.y;

    PxgShape sbShape, clothShape;
    uint sbCacheRef, clothCacheRef;
    loadSoftbodyClothPair(cmIdx, sbShape, sbCacheRef, clothShape, clothCacheRef);

    Ptr<PxgSoftBodyGpu> softbodies = Ptr<PxgSoftBodyGpu>(gVm.softbodies);
    Ptr<PxgFEMClothGpu> clothes = Ptr<PxgFEMClothGpu>(gVm.clothes);
    const uint softbodyId = sbShape.particleOrSoftbodyId;
    const uint clothId = clothShape.particleOrSoftbodyId;

    Ptr<float> contactDistance = Ptr<float>(gVm.contactDistance);

    // The broadphase early-out.  See the header on the SIGN: the cloth box SHRINKS.
    if (gVm.nbCollisionPairUpdatesPerTimestep > 1.0f)
    {
        float3 clothMin, clothMax, sbMin, sbMax;
        readBounds(gVm.bounds, clothCacheRef, clothMin, clothMax);
        readBounds(gVm.bounds, sbCacheRef, sbMin, sbMax);

        const float shrink = (gVm.nbCollisionPairUpdatesPerTimestep - 1.0f) *
                             contactDistance[clothCacheRef];
        clothMin += float3(shrink, shrink, shrink);
        clothMax -= float3(shrink, shrink, shrink);

        if (!boundsIntersect(clothMin, clothMax, sbMin, sbMax))
            return;
    }

    const float cDistance = contactDistance[sbCacheRef] + contactDistance[clothCacheRef];

    // The tree here is the SOFTBODY's tetrahedron BVH; mTetMeshData opens with
    // uint4(nbVerts, nbTets, maxDepth, nbBv32TreeNodes).
    const uint64_t treeAddr = softbodies[softbodyId].mTetMeshData + 16UL;
    Ptr<uint4> tetIndices = Ptr<uint4>(softbodies[softbodyId].mTetIndices);
    Ptr<float4> tetVerts = Ptr<float4>(softbodies[softbodyId].mPosition_InvMass);

    const uint nbVerts = clothes[clothId].mNbVerts;
    Ptr<float4> clothPos = Ptr<float4>(clothes[clothId].mPosition_InvMass);

    const uint stackCapacity = gVm.stackSizeBytes / 16u;

    for (uint sub = 0u; sub < MIDPHASE_WARPS_PER_BLOCK; ++sub)
    {
    const uint globalWarpIdx = gid.x * MIDPHASE_WARPS_PER_BLOCK + sub;

    for (uint vertIdx = globalWarpIdx; vertIdx < nbVerts; vertIdx += SC_NB_WARPS)
    {
        GroupMemoryBarrierWithGroupSync();   // groupshared scratch is reused per query

        const float3 pos = clothPos[vertIdx].xyz;
        // The query volume is a CUBE around the vertex, not a sphere.
        const float3 qMin = pos - float3(cDistance, cDistance, cDistance);
        const float3 qMax = pos + float3(cDistance, cDistance, cDistance);

        uint currentNodeIndex = 0u;
        bool hasNode = true;
        uint nbStack = 0u;

        while (hasNode)
        {
            const uint nodeIdx = currentNodeIndex >> 6u;

            if ((currentNodeIndex & 1u) != 0u)
            {
                const uint leafOffset = (currentNodeIndex >> 1u) & 31u;
                const uint data = bvData(treeAddr, nodeIdx, leafOffset);
                const uint nbPrimitives = (data >> 1u) & 63u;
                const uint primStartIndex = data >> 7u;
                uint primitiveIndex = 0xffffffffu;
                if (lane < nbPrimitives)
                    primitiveIndex = lane + primStartIndex;

                // SoftbodyBoxTraverser::intersectPrimitiveFullWarp.
                bool intersect = false;
                if (primitiveIndex != 0xffffffffu)
                {
                    const uint4 tetIdx = tetIndices[primitiveIndex];
                    const float3 w0 = tetVerts[tetIdx.x].xyz;
                    const float3 w1 = tetVerts[tetIdx.y].xyz;
                    const float3 w2 = tetVerts[tetIdx.z].xyz;
                    const float3 w3 = tetVerts[tetIdx.w].xyz;

                    if (isValidTet(w0, w1, w2, w3))
                    {
                        const float3 tMin = min(min(w0, w1), min(w2, w3));
                        const float3 tMax = max(max(w0, w1), max(w2, w3));
                        intersect = boundsIntersect(tMin, tMax, qMin, qMax);
                    }
                }

                // pushOntoStackFullWarp: ONE atomicAdd of the population count, then each
                // qualifying lane stores at its scan offset.  A full stack drops the WRITE
                // but not the COUNT, exactly as upstream -- the engine reads the counter back
                // to size the next allocation.
                const uint pmask = vmBallot(intersect, lane);
                const uint pcount = countbits(pmask);
                if (pcount != 0u)
                {
                    if (lane == 0u)
                    {
                        uint idx;
                        InterlockedAdd(Ptr<uint>(gVm.midphasePairsNum)[0], pcount, idx);
                        sVmIndex0 = idx;
                    }
                    GroupMemoryBarrierWithGroupSync();
                    const uint stackIndex = sVmIndex0 + countbits(pmask & ((1u << lane) - 1u));
                    if (intersect && stackIndex < stackCapacity)
                        Ptr<uint4>(gVm.stackPtr)[stackIndex] =
                            uint4(cmIdx, vertIdx, primitiveIndex, 0u);
                    GroupMemoryBarrierWithGroupSync();
                }
            }
            else
            {
                const uint nbChildren = bvNbNodes(treeAddr, nodeIdx);
                bool intersect = false;
                if (lane < nbChildren)
                {
                    const float3 mn = bvMin(treeAddr, nodeIdx, lane);
                    const float3 mx = bvMax(treeAddr, nodeIdx, lane);
                    intersect = boundsIntersect(qMin, qMax, mn, mx);
                }

                uint currentIndex = (currentNodeIndex & 0xFFFFFFC0u) | (lane << 1u) | 1u;
                bool isValidNode = false;
                if (lane < nbChildren)
                {
                    const uint data = bvData(treeAddr, nodeIdx, lane);
                    isValidNode = intersect;
                    if (isValidNode && (data & 1u) == 0u)
                        currentIndex = (data >> GU_BV4_CHILD_OFFSET_SHIFT_COUNT) << 6u;
                }

                const uint mask = vmBallot(isValidNode, lane);
                const uint offset = countbits(mask & ((1u << lane) - 1u));
                const uint validCount = countbits(mask);
                if (isValidNode)
                    sVmStack[nbStack + offset] = int(currentIndex);
                nbStack += validCount;
            }

            GroupMemoryBarrierWithGroupSync();
            if (nbStack > 0u)
            {
                nbStack -= 1u;
                currentNodeIndex = uint(sVmStack[nbStack]);
            }
            else
                hasNode = false;
        }
    }
    }
}
