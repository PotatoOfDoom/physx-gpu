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

// femClothMeshVertMidphaseBda.slang -- P9.26
//
// cloth_midphaseVertexMeshLaunch (femClothMidPhase.cu:515), with
// ClothParticleTreeTraverser (:313) and pushOntoStackFullWarp
// (bv32Traversal.cuh:197), all read in full.
//
// The VERTEX half of the cloth-vs-cooked-trimesh chain.  The TRIANGLE half --
// cloth_meshMidphaseGeneratePairsLaunch and cloth_meshContactGenLaunch -- landed
// in P8.17; this walks the rigid mesh's BV32 tree once per CLOTH VERTEX and
// pushes every triangle whose bound overlaps a sphere around it.
//
// ---------------------------------------------------------------------------
// HOW IT DIFFERS FROM THE TRIANGLE SIBLING, which is the file to read beside it
// ---------------------------------------------------------------------------
// V1 -- THE QUERY IS A SPHERE BOUND, not a triangle bound.  The traverser builds
//     `mBox(center - PxVec3(radius), center + PxVec3(radius))` from ONE cloth
//     vertex and the contact distance, where the sibling builds a bound over
//     three corners and fattens it.  Same AABB test at the nodes, different box.
//
// V2 -- IT PUSHES EVERY HIT.  The sibling runs a TriangleLeafBoundMinMaxTraverser
//     and keeps the best four by separation; this one has no reduction at all --
//     `pushOntoStackFullWarp` ballots the per-lane predicate, reserves the
//     popcount with ONE atomicAdd and writes at base + exclusive prefix.  A port
//     that borrows the sibling's top-4 silently drops pairs.
//
// V3 -- THE LEAF TEST IS THE TRIANGLE'S OWN BOUNDING BOX against mBox, and the
//     `isMesh` parameter that selects between the two branches SELECTS NOTHING:
//     both arms of `if (isMesh)` are the same expression, because the
//     "get rid of double-sided triangle" plane test between them is commented
//     out (femClothMidPhase.cu:366-378).  The parameter is still threaded through
//     the constructor and still passed `true` here and `false` by the particle
//     caller.  Ported as it stands -- restoring the commented test would change
//     which pairs reach contact gen.  Upstream candidate 34; documented, NOT
//     reported.
//
// V4 -- the grid is (SB_SBMIDPHASE, numTests, 1) x (WARP, MIDPHASE_WARPS_PER_BLOCK, 1),
//     so `cmIdx` is blockIdx.Y and the warp stride is blockDim.y * gridDim.x =
//     2 * 1024.  The port's launch site (PXG_NO_CUDA_TOOLKIT) launches one group per
//     CUDA warp, (2 * 1024, numTests, 1), so the warp index is gid.x and the stride
//     the group count.  Same shape as the sibling.
//
// V5 -- the vertex goes into the mesh's VERTEX space
//     (`shape2Vertex(trimeshToWorld.transformInv(p), scale, scaleRot)`) but the
//     RADIUS does not: the raw summed contact distance is used as the sphere
//     radius in vertex space, undivided by the scale.  The sibling's header
//     records the same asymmetry for its own fattening.
//
// ---------------------------------------------------------------------------
// WARP EMULATION: one dispatched group is ONE warp of 32 (the bridge forwards
// gridDim but drops blockDim), and the launch site launches one group per CUDA
// warp, exactly as for the sibling.  The ballot is groupshared +
// GroupMemoryBarrierWithGroupSync, so its control flow must be block-uniform --
// the loops below are over quantities uniform across the group (nbVerts, the
// node stack), and the only per-lane predicate is the one being balloted.
// ===========================================================================
module femClothMeshVertMidphaseBda;

import pxgSlangFEMClothDesc;   // PxgFEMClothGpu (mPosition_InvMass@16, mNbVerts@316)
import pxgSlangBv32;           // P8.25: the BV32 reader, shared

static const uint MIDPHASE_WARPS_PER_BLOCK = 2u;
static const uint SB_SBMIDPHASE = 1024u;
static const uint WARP = 32u;
static const uint STACK_SIZE = 192u;    // femMidphaseScratch.sBv32Nodes[192]

// ---------------------------------------------------------------------------
// Push-constant blob -- BYTE-IDENTICAL to the CUDA pack
// (PxgNarrowphaseCore.cpp:6723): { PxU32 numWorkItems, PxReal toleranceLength,
// cmInputs*, transformCache*, bounds*, contactDistance*, gpuShapes*, femClothes*,
// PxU32 stackSizeBytes, stackPtr*, midphasePairsNum* } -> 80 bytes.
// ABI row {4,4,8,8,8,8,8,8,4,8,8} -- byte-identical to kSbMidphasePrimitives
// (the SOFTBODY primitive midphase takes the same eleven arguments in the same
// order) and given its OWN row, kClothVertMeshMidphase, rather than aliased:
// the same call P8.17 made for kClothMeshMidphase, so an upstream divergence in
// one of two packs that agree today shows up as a row and not as a silent
// reinterpretation of the other kernel's arguments.
// ---------------------------------------------------------------------------
struct ClothMeshVertMidphaseParams
{
    uint numWorkItems;              // @0  (ABI parity: the CUDA's `if (cmIdx < numWorkItems)`
                                    //      is commented out and cmIdx comes from blockIdx.y)
    float toleranceLength;          // @4  (ABI parity only)
    uint64_t cmInputs;              // @8
    uint64_t transformCache;        // @16
    uint64_t bounds;                // @24 (ABI parity only)
    uint64_t contactDistance;       // @32
    uint64_t gpuShapes;             // @40
    uint64_t femClothes;            // @48
    uint stackSizeBytes;            // @56
    uint _pad60;                    // @60
    uint64_t stackPtr;              // @64 output (uint4*)
    uint64_t midphasePairsNum;      // @72 output (PxU32*) -> 80
}

[[vk::push_constant]] ConstantBuffer<ClothMeshVertMidphaseParams> gVm;

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

// ---------------------------------------------------------------------------
// PxQuat helpers (PxQuat.h), transcribed as the sibling does.
// ---------------------------------------------------------------------------
float3 vmQuatRotate(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d2 = (q.x * vx + q.y * vy + q.z * vz);
    return float3(vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d2,
                  vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d2,
                  vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d2);
}
float3 vmQuatRotateInv(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d2 = (q.x * vx + q.y * vy + q.z * vz);
    return float3(vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * d2,
                  vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * d2,
                  vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * d2);
}
// shape2Vertex (cudaNpCommon.h:285): rotate, divide by the scale, rotate back.
float3 vmShape2Vertex(float3 v, float3 scale, float4 rot)
{
    return vmQuatRotateInv(rot, vmQuatRotate(rot, v) / scale);
}

// femClothMidPhase.cu:527-568.  BOTH shapes are eTRIANGLEMESH, so the CLOTH is
// the one carrying a body id and the cooked static mesh carries 0xffffffff.
void loadClothTrimeshPair(uint64_t cmInputsAddr, uint64_t shapesAddr, uint cmIdx,
                          out PxgShape clothShape, out uint clothCacheRef,
                          out PxgShape triShape, out uint triCacheRef)
{
    const PxgContactManagerInput cin = Ptr<PxgContactManagerInput>(cmInputsAddr)[cmIdx];
    Ptr<PxgShape> shapes = Ptr<PxgShape>(shapesAddr);
    uint sRefCloth = cin.shapeRef0, cRefCloth = cin.transformCacheRef0;
    uint sRefTri = cin.shapeRef1, cRefTri = cin.transformCacheRef1;
    if (shapes[cin.shapeRef0].particleOrSoftbodyId == 0xffffffffu)
    {
        sRefCloth = cin.shapeRef1; cRefCloth = cin.transformCacheRef1;
        sRefTri = cin.shapeRef0; cRefTri = cin.transformCacheRef0;
    }
    clothShape = shapes[sRefCloth]; clothCacheRef = cRefCloth;
    triShape = shapes[sRefTri]; triCacheRef = cRefTri;
}

// ---------------------------------------------------------------------------
// Warp emulation scratch: one group == one warp, so a single slot each.
// ---------------------------------------------------------------------------
groupshared int sVmStack[STACK_SIZE];   // bv32 to-be-expanded node stack
groupshared uint sVmBallot;             // __ballot_sync mask
groupshared uint sVmBase;               // atomicAdd result broadcast

uint vmBallot(bool pred, uint lane)
{
    if (lane == 0u) sVmBallot = 0u;
    GroupMemoryBarrierWithGroupSync();
    if (pred) InterlockedOr(sVmBallot, 1u << lane);
    GroupMemoryBarrierWithGroupSync();
    return sVmBallot;
}

// pushOntoStackFullWarp (bv32Traversal.cuh:197) ->
// globalScanExclusiveSingleWarp: ONE atomicAdd of the popcount by lane 0,
// broadcast, and each pushing lane writes at base + its exclusive prefix.
// V2: every hit, no reduction.
void vmPushFullWarp(bool pushes, uint lane, uint stackCapacity, uint4 item)
{
    const uint mask = vmBallot(pushes, lane);
    const uint total = countbits(mask);
    if (lane == 0u)
    {
        uint base = 0u;
        if (total != 0u)
            InterlockedAdd(Ptr<uint>(gVm.midphasePairsNum)[0], total, base);
        sVmBase = base;
    }
    GroupMemoryBarrierWithGroupSync();
    const uint stackIndex = sVmBase + countbits(mask & ((1u << lane) - 1u));
    // "If we run out of stacksize, drop contacts" -- the counter still advanced.
    if (pushes && stackIndex < stackCapacity)
        Ptr<uint4>(gVm.stackPtr)[stackIndex] = item;
    GroupMemoryBarrierWithGroupSync();
}

// ===========================================================================
[shader("compute")]
[numthreads(32, 1, 1)]
void cloth_midphaseVertexMeshLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;
    const uint cmIdx = gid.y;                      // V4: blockIdx.y -- one pair per y-block

    PxgShape clothShape, triShape;
    uint clothCacheRef, triCacheRef;
    loadClothTrimeshPair(gVm.cmInputs, gVm.gpuShapes, cmIdx,
                         clothShape, clothCacheRef, triShape, triCacheRef);

    Ptr<PxgFEMClothGpu> clothes = Ptr<PxgFEMClothGpu>(gVm.femClothes);
    const uint clothId = clothShape.particleOrSoftbodyId;

    // PxsCachedTransform (32 B): PxQuat q @0, PxVec3 p @16.
    const uint64_t xf = gVm.transformCache + uint64_t(triCacheRef) * 32UL;
    const float4 triQ = Ptr<float4>(xf)[0];
    const float3 triP = float3(Ptr<float>(xf + 16UL)[0],
                               Ptr<float>(xf + 20UL)[0],
                               Ptr<float>(xf + 24UL)[0]);

    const float3 triScale = float3(triShape.scaleX, triShape.scaleY, triShape.scaleZ);
    const float4 triScaleRot = float4(triShape.scaleRotX, triShape.scaleRotY,
                                      triShape.scaleRotZ, triShape.scaleRotW);

    // readTriangleMesh (triangleMesh.cuh:120): header uint4, then the packed BV32
    // nodes, then the vertices, then the index triples.
    const uint64_t meshPtr = uint64_t(triShape.hullOrMeshOffset.x) |
                             (uint64_t(triShape.hullOrMeshOffset.y) << 32);
    const uint4 meshHeader = Ptr<uint4>(meshPtr)[0];
    const uint64_t treeAddr = meshPtr + 16UL;
    const uint64_t vertsAddr = treeAddr + uint64_t(meshHeader.w) * BV32_STRIDE;
    const uint64_t indicesAddr = vertsAddr + uint64_t(meshHeader.x) * 16UL;
    Ptr<float4> meshVerts = Ptr<float4>(vertsAddr);
    Ptr<uint4> meshIndices = Ptr<uint4>(indicesAddr);

    // V5: the raw SUM, used as a sphere radius in VERTEX space without being
    // divided by the scale.
    const float cDistance = Ptr<float>(gVm.contactDistance)[clothCacheRef] +
                            Ptr<float>(gVm.contactDistance)[triCacheRef];

    const uint numVerts = clothes[clothId].mNbVerts;
    Ptr<float4> clothVerts = Ptr<float4>(clothes[clothId].mPosition_InvMass);

    const uint stackCapacity = gVm.stackSizeBytes / 16u;

    // One group per CUDA warp (see the header on the grid): globalWarpIdx is the group
    // index and the stride the group count, i.e. blockDim.y * gridDim.x.
    {
    const uint globalWarpIdx = gid.x;

    for (uint vertexIdx = globalWarpIdx; vertexIdx < numVerts; vertexIdx += WorkgroupCount().x)
    {
        GroupMemoryBarrierWithGroupSync();   // groupshared scratch is reused per query

        // World -> mesh local -> mesh VERTEX space.
        // P9.49: one of TWO sites in this binary that were actually faulting (the
        // other is femClothPsMidphaseBda, which faulted more often of the two).
        // This platform executes the body of this loop with an
        // ALL-INACTIVE mask -- the barrier above is inside the loop, so the body has
        // to be reached even by a warp whose range is exhausted -- and it then issues
        // this read as a SCALAR one at "the first active lane, or lane 0 if there is
        // no active lane".  Lane 0 then carries `vertexIdx` PAST the bound the loop
        // condition just rejected.  Measured, not reasoned: the faulting address was
        // base + 0xe000 = index 3584 = 1536 + 2048, i.e. globalWarpIdx 1536 (numVerts
        // is 1100) after one `+= 2048`, the stride.  The value is discarded -- the mask is
        // empty -- so only the ADDRESS matters, and clamping it changes nothing for
        // any lane the condition admits.
        // `max(numVerts, 1u)` and not `numVerts`: at zero the subtraction wraps to
        // 0xffffffff and the clamp does nothing -- which is exactly the case where
        // the body runs on an empty mask alone.
        const float3 world = clothVerts[min(vertexIdx, max(numVerts, 1u) - 1u)].xyz;
        const float3 center = vmShape2Vertex(vmQuatRotateInv(triQ, world - triP),
                                             triScale, triScaleRot);
        // V1: the query is a SPHERE bound around one vertex.
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
                // P9.28: the `!= 0xffffffff` guard below is PER-LANE, so this platform turns it
                // into a mask and issues the reads under it for the EXCLUDED lanes too -- at
                // primStartIndex + 0xffffffff, sixty-four gigabytes out for a uint4 array.
                // `safePrim` is the ADDRESS those reads use; the sentinel still decides the
                // VALUE, so nothing observable changes.  Traced from a reproducible sporadic
                // SIGSEGV, not from reading.
                uint primitiveIndex = 0xffffffffu;
                if (lane < nbPrimitives)
                    primitiveIndex = lane + primStartIndex;
                const uint safePrim = min(primitiveIndex,
                                          primStartIndex + max(nbPrimitives, 1u) - 1u);

                bool intersect = false;
                if (primitiveIndex != 0xffffffffu)
                {
                    const uint4 mTri = meshIndices[safePrim];
                    const float3 b0 = meshVerts[mTri.x].xyz;
                    const float3 b1 = meshVerts[mTri.y].xyz;
                    const float3 b2 = meshVerts[mTri.z].xyz;
                    // V3: triBoundingBox(v0,v1,v2).intersects(mBox).  BOTH arms of
                    // the CUDA's `if (isMesh)` are this same expression.
                    const float3 tMin = min(min(b0, b1), b2);
                    const float3 tMax = max(max(b0, b1), b2);
                    intersect = !(tMin.x > qMax.x || qMin.x > tMax.x ||
                                  tMin.y > qMax.y || qMin.y > tMax.y ||
                                  tMin.z > qMax.z || qMin.z > tMax.z);
                }

                // V2: every hit, balloted and written at the exclusive prefix.
                vmPushFullWarp(intersect, lane, stackCapacity,
                               uint4(cmIdx, vertexIdx, primitiveIndex, 0u));
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
