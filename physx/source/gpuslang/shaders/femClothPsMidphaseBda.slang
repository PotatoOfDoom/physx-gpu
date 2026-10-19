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

// femClothPsMidphaseBda.slang
//
// P9.30 -- cloth_psMidphaseGeneratePairsLaunch
// gpunarrowphase/src/CUDA/femClothMidPhase.cu:395 (wrapper), :313
// (ClothParticleTreeTraverser), bv32Traversal.cuh (the walk and
// pushOntoStackFullWarp) -- all read in full.
//
// The FIRST half of the cloth-vs-PARTICLE chain, and the third BV32 point-query
// midphase in this port after the cloth-vs-mesh vertex query (P9.26) and the
// softbody-vs-cloth vertex query.  The query point here is a PARTICLE, the tree
// is the CLOTH's, and both live in WORLD space -- which is the one structural
// difference from its siblings and the reason there is no shape2Vertex here.
//
// ---------------------------------------------------------------------------
// TRAPS
// ---------------------------------------------------------------------------
// M1 -- THE PAIR IS (cmIdx, particleIndex, triangleIndex).  Every other cloth
//     midphase in this port writes the CLOTH element in .y; this one writes the
//     PARTICLE there and the triangle in .z.  A port that copies a sibling
//     swaps the two, and the contact gen then indexes triangles by particle id.
//
// M2 -- SENTINEL-GUARDED BDA READ, the P9.28 class.  `primitiveIndex` is
//     0xffffffff for the lanes past the leaf's primitive count, and the read of
//     the index triple sits under `!= 0xffffffff`.  That guard is a MASK on this
//     platform, so the read is issued for the excluded lanes too, at
//     base + 0xffffffff * 16.  The ADDRESS is clamped; the sentinel still
//     decides the VALUE, and it is the sentinel -- not the clamp -- that goes
//     into the pair.
//
// M3 -- `isMesh` IS DEAD.  The traverser takes the flag, the launch passes
//     `false`, and BOTH arms compute the same expression: the line that would
//     have distinguished them (a plane-distance test to drop back-facing
//     triangles) is commented out upstream.  Ported as the one expression, and
//     the parameter is not carried.
//
// M4 -- NO SCALING ANYWHERE.  The cloth's triangles are read straight from
//     mPosition_InvMass and the particles straight from
//     mSortedPositions_InvMass; neither transform cache is applied and neither
//     shape's scale is.  Both arrays are already world-space.  The mesh sibling
//     next door DOES transform, into the static mesh's vertex space, and
//     copying it here would move the cloth.
//
// M5 -- THE QUERY BOX IS A CUBE OF SIDE 2*cDistance, not a triangle bound
//     inflated by it: `PxBounds3(c - PxVec3(r), c + PxVec3(r))` where
//     r = contactDistance[particleCacheRef] + contactDistance[clothCacheRef].
//
// M6 -- THE SHAPE PAIR IS ROUTED BY TYPE, ePARTICLESYSTEM (6) first.  The cloth
//     sibling routes by "which one carries a body id", which is what you need
//     when both shapes are eTRIANGLEMESH; here the types differ, and
//     LoadShapePairWarp's own rule is the type test.
//
// M7 -- the CUDA warp-copies the WHOLE PxgParticleSystem (608 bytes in 5.12) into shared
//     memory before touching two of its fields.  That is a CUDA memory-traffic
//     device, not semantics: this port reads mSortedPositions_InvMass and
//     mCommonData_NumParticles through BDA where they are used.  Same values,
//     no shared-memory staging, and no 608-byte groupshared array.
//
// M8 -- the particle loop strides by blockDim.y * gridDim.x -- WARPS, not
//     threads, and NOT including blockDim.x.  Reproduced as a constant because
//     the bridge forwards only gridDim.
// ===========================================================================

import pxgSlangFEMClothDesc;   // PxgFEMClothGpu (mTriMeshData@0, mPosition_InvMass@16)
import pxgSlangParticleDesc;   // PxgParticleSystemGpu (mSortedPositions_InvMass@24)
import pxgSlangBv32;           // P8.25: the BV32 reader, shared

static const uint MIDPHASE_WARPS_PER_BLOCK = 2u;
static const uint PS_MIDPHASE_GRID = 1024u;
static const uint PS_NB_WARPS = PS_MIDPHASE_GRID * MIDPHASE_WARPS_PER_BLOCK;   // M8
static const uint WARP = 32u;
static const uint STACK_SIZE = 192u;    // femMidphaseScratch.sBv32Nodes[192]

// PxGeometryType (PxGeometry.h): ePARTICLESYSTEM is 6, eTRIANGLEMESH is 8.
static const uint GEOM_PARTICLESYSTEM = 6u;

// ---------------------------------------------------------------------------
// Push-constant blob -- BYTE-IDENTICAL to the CUDA pack
// (PxgNarrowphaseCore.cpp:3579): { PxU32 numWorkItems, PxReal toleranceLength,
// cmInputs*, gpuShapes*, transformCache*, bounds*, contactDistance*, femClothes*,
// particleSystems*, PxU32 stackSizeBytes, stackPtr*, midphasePairsNum* } -> 88 B.
// ABI row {4,4,8,8,8,8,8,8,8,4,8,8}.
//
// NOTE the argument ORDER against the cloth-vs-mesh vertex midphase, which takes
// eleven of the same twelve: there `transformCache` comes before `bounds` and
// `gpuShapes` after them, here `gpuShapes` comes THIRD.  Same names, different
// slots, and reusing that row would put every pointer past the third at the
// wrong offset.
// ---------------------------------------------------------------------------
struct ClothPsMidphaseParams
{
    uint numWorkItems;              // @0  (the CUDA gates on `blockIdx.y < numWorkItems`)
    float toleranceLength;          // @4  (ABI parity only)
    uint64_t cmInputs;              // @8
    uint64_t gpuShapes;             // @16
    uint64_t transformCache;        // @24 (ABI parity only -- M4: nothing is transformed)
    uint64_t bounds;                // @32 (ABI parity only)
    uint64_t contactDistance;       // @40
    uint64_t femClothes;            // @48
    uint64_t particleSystems;       // @56
    uint stackSizeBytes;            // @64
    uint _pad68;                    // @68
    uint64_t stackPtr;              // @72 output (uint4*)
    uint64_t midphasePairsNum;      // @80 output (PxU32*) -> 88
}

[[vk::push_constant]] ConstantBuffer<ClothPsMidphaseParams> gPm;

struct PxgContactManagerInput { uint shapeRef0, shapeRef1, transformCacheRef0, transformCacheRef1; }

struct PxgShape                 // PxgConvexConvexShape.h, 48 B
{
    float scaleX, scaleY, scaleZ;                     // @0
    float scaleRotX, scaleRotY, scaleRotZ, scaleRotW; // @12
    uint materialIndex;                               // @28
    uint2 hullOrMeshOffset;                           // @32
    uint type;                                        // @40
    uint particleOrSoftbodyId;                        // @44
}

// M6: LoadShapePairWarp<ePARTICLESYSTEM, eTRIANGLEMESH> -- shape0 is the one
// whose TYPE matches, and the pair is swapped if slot 0 is not it.
void loadParticleClothPair(uint64_t cmInputsAddr, uint64_t shapesAddr, uint cmIdx,
                           out PxgShape particleShape, out uint particleCacheRef,
                           out PxgShape clothShape, out uint clothCacheRef)
{
    const PxgContactManagerInput cin = Ptr<PxgContactManagerInput>(cmInputsAddr)[cmIdx];
    Ptr<PxgShape> shapes = Ptr<PxgShape>(shapesAddr);
    uint sRefP = cin.shapeRef0, cRefP = cin.transformCacheRef0;
    uint sRefC = cin.shapeRef1, cRefC = cin.transformCacheRef1;
    if (shapes[cin.shapeRef0].type != GEOM_PARTICLESYSTEM)
    {
        sRefP = cin.shapeRef1; cRefP = cin.transformCacheRef1;
        sRefC = cin.shapeRef0; cRefC = cin.transformCacheRef0;
    }
    particleShape = shapes[sRefP]; particleCacheRef = cRefP;
    clothShape = shapes[sRefC]; clothCacheRef = cRefC;
}

// ---------------------------------------------------------------------------
// Warp emulation scratch: one group == one warp, so a single slot each.
// ---------------------------------------------------------------------------
groupshared int sPmStack[STACK_SIZE];   // bv32 to-be-expanded node stack
groupshared uint sPmBallot;             // __ballot_sync mask
groupshared uint sPmBase;               // atomicAdd result broadcast

uint pmBallot(bool pred, uint lane)
{
    if (lane == 0u) sPmBallot = 0u;
    GroupMemoryBarrierWithGroupSync();
    if (pred) InterlockedOr(sPmBallot, 1u << lane);
    GroupMemoryBarrierWithGroupSync();
    return sPmBallot;
}

// pushOntoStackFullWarp (bv32Traversal.cuh:197): ONE atomicAdd of the popcount by
// lane 0, broadcast, and each pushing lane writes at base + its exclusive prefix.
void pmPushFullWarp(bool pushes, uint lane, uint stackCapacity, uint4 item)
{
    const uint mask = pmBallot(pushes, lane);
    const uint total = countbits(mask);
    if (lane == 0u)
    {
        uint base = 0u;
        if (total != 0u)
            InterlockedAdd(Ptr<uint>(gPm.midphasePairsNum)[0], total, base);
        sPmBase = base;
    }
    GroupMemoryBarrierWithGroupSync();
    const uint stackIndex = sPmBase + countbits(mask & ((1u << lane) - 1u));
    // "If we run out of stacksize, drop contacts" -- the counter still advanced,
    // which is what the contact gen's atomicMax high-water mark then reports.
    if (pushes && stackIndex < stackCapacity)
        Ptr<uint4>(gPm.stackPtr)[stackIndex] = item;
    GroupMemoryBarrierWithGroupSync();
}

// ===========================================================================
[shader("compute")]
[numthreads(32, 1, 1)]
void cloth_psMidphaseGeneratePairsLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;
    const uint cmIdx = gid.y;                      // one contact manager per y-block

    if (cmIdx >= gPm.numWorkItems)
        return;

    PxgShape particleShape, clothShape;
    uint particleCacheRef, clothCacheRef;
    loadParticleClothPair(gPm.cmInputs, gPm.gpuShapes, cmIdx,
                          particleShape, particleCacheRef, clothShape, clothCacheRef);

    Ptr<PxgFEMClothGpu> clothes = Ptr<PxgFEMClothGpu>(gPm.femClothes);
    const uint clothId = clothShape.particleOrSoftbodyId;

    // M7: read the two fields where they are used, not a 608-byte shared copy.
    Ptr<PxgParticleSystemGpu> particleSystems = Ptr<PxgParticleSystemGpu>(gPm.particleSystems);
    const uint particleSystemId = particleShape.particleOrSoftbodyId;
    Ptr<float4> sortedPose = Ptr<float4>(particleSystems[particleSystemId].mSortedPositions_InvMass);
    const uint nbParticles = particleSystems[particleSystemId].mCommonData_NumParticles;

    // The CLOTH's tree: mTriMeshData, then a uint4 header, then the packed nodes.
    const uint64_t meshData = clothes[clothId].mTriMeshData;
    const uint64_t treeAddr = meshData + 16UL;
    Ptr<float4> clothVerts = Ptr<float4>(clothes[clothId].mPosition_InvMass);
    Ptr<uint4> clothTris = Ptr<uint4>(clothes[clothId].mTriangleVertexIndices_ownedEdges);

    // M5: the query is a CUBE of half-side cDistance around the particle.
    const float cDistance = Ptr<float>(gPm.contactDistance)[particleCacheRef] +
                            Ptr<float>(gPm.contactDistance)[clothCacheRef];

    const uint stackCapacity = gPm.stackSizeBytes / 16u;

    // The block's two warps, walked serially: the CUDA block is (32, 2, 1) and
    // this group is one warp, so the group covers both of the block's warps in
    // turn.  Same arrangement, and same reason, as the mesh vertex sibling.
    for (uint sub = 0u; sub < MIDPHASE_WARPS_PER_BLOCK; ++sub)
    {
    const uint globalWarpIdx = gid.x * MIDPHASE_WARPS_PER_BLOCK + sub;

    for (uint particleIndex = globalWarpIdx; particleIndex < nbParticles; particleIndex += PS_NB_WARPS)
    {
        GroupMemoryBarrierWithGroupSync();   // groupshared scratch is reused per query

        // M4: world space on both sides -- no transform, no scale.
        // P9.49: this read FAULTED, reproducibly.  See the class note in the
        // header; the short form is that the barrier above forces this platform to
        // reach the body with an ALL-INACTIVE mask, the read is then issued as a
        // scalar one at lane 0, and lane 0's `particleIndex` starts at
        // `globalWarpIdx` -- a PERSISTENT grid's warp number, which has nothing to
        // do with nbParticles.  Clamping changes nothing for any lane the loop
        // condition admits.
        const float3 center = sortedPose[min(particleIndex, max(nbParticles, 1u) - 1u)].xyz;
        const float3 qMin = center - cDistance;
        const float3 qMax = center + cDistance;

        uint currentNodeIndex = 0u;   // root
        bool hasNode = true;
        uint nbStack = 0u;

        while (hasNode)
        {
            const uint nodeIdx = bvNodeIdxOf(currentNodeIndex);

            if (bvIsLeaf(currentNodeIndex))
            {
                const uint leafOffset = bvLeafSlotOf(currentNodeIndex);
                const uint data = bvData(treeAddr, nodeIdx, leafOffset);
                const uint nbPrimitives = bvDataNbPrimitives(data);
                const uint primStartIndex = bvDataPrimStart(data);
                // M2: the sentinel-guarded read -- clamp the ADDRESS, leave the
                // VALUE to the guard.
                uint primitiveIndex = 0xffffffffu;
                if (lane < nbPrimitives)
                    primitiveIndex = lane + primStartIndex;
                const uint safePrim = min(primitiveIndex,
                                          primStartIndex + max(nbPrimitives, 1u) - 1u);

                bool intersect = false;
                if (primitiveIndex != 0xffffffffu)
                {
                    const uint4 tri = clothTris[safePrim];
                    const float3 b0 = clothVerts[tri.x].xyz;
                    const float3 b1 = clothVerts[tri.y].xyz;
                    const float3 b2 = clothVerts[tri.z].xyz;
                    // M3: triBoundingBox(v0,v1,v2).intersects(mBox) -- the one
                    // expression BOTH arms of the CUDA's `if (isMesh)` compute.
                    const float3 tMin = min(min(b0, b1), b2);
                    const float3 tMax = max(max(b0, b1), b2);
                    intersect = !(tMin.x > qMax.x || qMin.x > tMax.x ||
                                  tMin.y > qMax.y || qMin.y > tMax.y ||
                                  tMin.z > qMax.z || qMin.z > tMax.z);
                }

                // M1: (cmIdx, PARTICLE, TRIANGLE) -- the particle in .y.
                pmPushFullWarp(intersect, lane, stackCapacity,
                               uint4(cmIdx, particleIndex, primitiveIndex, 0u));
            }
            else
            {
                const uint nbChildren = bvNbNodes(treeAddr, nodeIdx);
                float3 mn = float3(0, 0, 0), mx = float3(0, 0, 0);
                const bool hasBox = lane < nbChildren;
                if (hasBox)
                {
                    mn = bvMin(treeAddr, nodeIdx, lane);
                    mx = bvMax(treeAddr, nodeIdx, lane);
                }
                bool intersect = false;
                if (hasBox)
                    intersect = !(mn.x > qMax.x || qMin.x > mx.x ||
                                  mn.y > qMax.y || qMin.y > mx.y ||
                                  mn.z > qMax.z || qMin.z > mx.z);

                uint currentIndex = bvChildIndex(currentNodeIndex, lane);
                bool isValidNode = false;
                if (lane < nbChildren)
                {
                    const uint data = bvData(treeAddr, nodeIdx, lane);
                    isValidNode = intersect;
                    if (isValidNode && bvDataIsInternal(data))
                        currentIndex = bvDataChildNodeIndex(data);
                }

                const uint mask = pmBallot(isValidNode, lane);
                const uint offset = countbits(mask & ((1u << lane) - 1u));
                const uint validCount = countbits(mask);
                if (isValidNode)
                    sPmStack[nbStack + offset] = int(currentIndex);
                nbStack += validCount;
            }

            GroupMemoryBarrierWithGroupSync();
            if (nbStack > 0u)
            {
                nbStack -= 1u;
                currentNodeIndex = uint(sPmStack[nbStack]);
            }
            else
                hasNode = false;
        }
    }
    }
}
