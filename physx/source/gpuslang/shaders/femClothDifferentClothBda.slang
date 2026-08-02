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

// femClothDifferentClothBda.slang
//
// P8.16 — cloth-vs-DIFFERENT-cloth midphase on the engine ABI (BDA).  The sibling
// femClothSelfCollisionBda's header called out as "a SEPARATE arg pack / ABI, out of
// scope for this tranche"; this is that tranche.  Ported line-for-line from
// gpunarrowphase/src/CUDA/femClothClothMidPhase.cu:
//   - cloth_differentClothCollisionVTLaunch (:1388, launch PxgFEMClothCore.cpp:768,
//     CLOTH_DIFFERENTCLOTHCOLLISION_MIDPHASE_VT): one warp per cloth VERTEX of EITHER
//     cloth walks the OTHER cloth's BV32 triangle BVH (femDifferentClothCollisionCoreVT
//     + the ClothTreeVertexTraverser callback) and appends a vertex-triangle
//     PxgFemFemContactInfo per penetrating (vertex, triangle) pair;
//   - cloth_differentClothCollisionEELaunch (:1427, launch PxgFEMClothCore.cpp:768,
//     CLOTH_DIFFERENTCLOTHCOLLISION_MIDPHASE_EE): one warp per triangle-with-active-edges
//     of the SMALLER cloth walks the larger cloth's BVH (femDifferentClothCollisionCoreEE
//     + ClothTreeEdgeTraverser) and appends an edge-edge record per crossing edge pair.
//
// The two traversers are the SAME C++ structs the self-collision cores instantiate, so
// the leaf/box callbacks below are the proven femClothSelfCollisionBda transcriptions.
// Everything genuinely kernel-independent (BV32 readers, closestPtPointTriangle,
// closestPtLineLine, boundsIntersect, isType0EdgeActive, the auxInd bit layout,
// pxEncodeClothIndex) comes from pxgSlangClothMidphase, extracted at P8.15 precisely so
// this file would not fork a second copy.
//
// THE FIVE DIFFERENCES FROM THE SELF PATH (femClothClothMidPhase.cu:1240-1384):
//   1. The pair comes from the NARROWPHASE, not from an active-cloth list: cmIdx =
//      blockIdx.y indexes cmInputs; shapeRef0/1 index gpuShapes; the two cloth ids are
//      PxgShape::particleOrSoftbodyId.  There is no activeClothes array and NO
//      eDISABLE_SELF_COLLISION check on this path (that flag gates self-collision only).
//   2. contactDist is the SUM of the two shapes' cached contact distances
//      (contactDistance[transformCacheRef0] + contactDistance[transformCacheRef1]), where
//      the self path used ONE reference doubled.
//   3. VT covers BOTH directions in one launch: globalWarpIdx < numVerts0 + numVerts1,
//      the first numVerts0 warps querying cloth0's vertices against cloth1's tree and the
//      rest the mirror.  EE covers only min(numTri0, numTri1) queries, the SMALLER cloth
//      (by triangles-with-active-edges) always driving — `isCloth0 = numTri0 <= numTri1`.
//      Both use the branchless id select the CUDA writes verbatim.
//   4. Every `clothId0 != clothId1` guard is LIVE (and true), so the self path's three
//      skips are disabled: the VT vertex-is-a-corner-of-the-triangle test, the EE
//      `t0_index > primitiveIndex` half-pair dedup, and the EE connected-edge test.  The
//      corollary is markValidity(clothId0 != clothId1) -> VALID: a different-cloth pair is
//      final here, where a self pair is re-checked in cloth_clothContactPrepareLaunch.
//   5. The REST-POSE FILTER IS GONE.  The call sites pass NULL for both rest buffers and
//      0.0f for filterDistSq (:1298-1300, :1370-1371); distSqInRestPose returns
//      PX_MAX_REAL for a null first argument (:662-668, :685-691), so
//      `tempRestDistSq > filterDistSq` is unconditionally true.  Transcribing the
//      self-collision lookup here would dereference a null device pointer — the block is
//      OMITTED, which is what the CUDA collapses to, not a deviation from it.
//
// DEVIATIONS from CUDA (recipe policy; identical set to femClothSelfCollisionBda, each
// preserving the emitted contact SET and the running total):
//   - ONE 32-thread group per warp-query with the block's MIDPHASE_WARPS_PER_BLOCK warps
//     walked SERIALLY in a group-uniform `sub` loop, in place of the CUDA
//     (WARP_SIZE, MIDPHASE_WARPS_PER_BLOCK, 1) block.  The bv32 walk is warp-cooperative
//     with a per-warp groupshared stack and a data-dependent loop, so packing >1 warp per
//     group would put GroupMemoryBarrierWithGroupSync in control flow that is non-uniform
//     ACROSS warps; the serial loop keeps every barrier group-uniform.  The engine
//     launches ceil(2*maxClothVerts / 2) x numTests (VT) and
//     ceil(maxClothTrianglesWithActiveEdges / 2) x numTests (EE) blocks and the Vulkan
//     bridge drops blockDim, so the loop is what covers the second half of the range —
//     the same bug-and-fix femClothSelfCollisionBda records;
//   - __ballot_sync -> groupshared InterlockedOr mask; warpScanExclusive ->
//     countbits(mask & lanemask_lt) (VT, single-element) / a groupshared prefix sum over
//     the per-lane count (EE, multi-element);
//   - the per-warp femMidphaseScratch (other cloth's verts / indices / bv32 root cached by
//     lane 0 + __syncthreads) is dropped: every lane reads those addresses directly
//     through Ptr<>;
//   - PxgFemFemContactInfo's auxInd is zero-initialized before the mark*/setAux* sequence
//     (CUDA leaves the untouched bits indeterminate), pinning the otherwise-dead
//     InCollision bit 29 — and, for VT, the unused aux payload — to 0;
//   - clothClothPairs / numClothClothPairs are carried in the push blob for ABI parity but
//     NOT read: both traversers store them and apply no filter-pair lookup ("No filtering
//     applied yet" — femClothClothMidPhase.cu:768, :884).
//
// All float arithmetic is NoContraction-stamped (PXG_NOCONTRACT_SHADERS): the VT
// (vertex - closestPt) dot feeds the compared `< contactDistSq`, the EE closestPtLineLine
// chain feeds both `distSq < contactDistSq` and the barycentric s/t compared against the
// DEFORMABLE thresholds, and the box tests' clamp/dot feed the compared child overlap.
// No transcendental anywhere in the path — this tranche is FULLY bit-exact.

import pxgSlangFEMClothDesc;   // PxgFEMClothGpu (mTriMeshData@0, mPosition_InvMass@16, ...)
import pxgSlangClothMidphase;  // BV32 readers, closest-point routines, auxInd bit layout

// ---------------------------------------------------------------------------
// Push-constant blob — BYTE-IDENTICAL to the CUDA kernelParams pack
// (PxgFEMClothCore.cpp:768): { PxU32 mMaxContacts, PxReal toleranceLength,
// const PxgContactManagerInput* cmInputs, const PxsCachedTransform* transformCache,
// const PxReal* contactDistance, const PxgShape* gpuShapes, const PxgFEMCloth* femClothes,
// PxgNonRigidFilterPair* clothClothPairs, PxU32 numClothClothPairs,
// const PxU8* updateContactPairs, PxgFemFemContactInfo* outContactInfo,
// PxU32* totalContactCount } -> 88-byte natural-alignment blob.
// (ABI row {4,4,8,8,8,8,8,8,4,8,8,8}.)
struct DifferentClothParams
{
    uint maxContacts;             // @0
    float toleranceLength;        // @4   (unused; ABI parity — CUDA stores it and never reads it)
    uint64_t cmInputs;            // @8
    uint64_t transformCache;      // @16  (unused; ABI parity — the traversers work in world space)
    uint64_t contactDistance;     // @24
    uint64_t gpuShapes;           // @32
    uint64_t femClothes;          // @40
    uint64_t clothClothPairs;     // @48  (unused; ABI parity)
    uint numClothClothPairs;      // @56  (unused; ABI parity)
    uint _pad60;                  // @60
    uint64_t updateContactPairs;  // @64  PxU8*
    uint64_t outContactInfo;      // @72  output
    uint64_t totalContactCount;   // @80  output -> 88
}

[[vk::push_constant]] ConstantBuffer<DifferentClothParams> gD;

// PxgContactManagerInput (PxgContactManager.h): four PxU32.
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
// Groupshared state for the one-warp-per-query lowering (mirrors the self-collision
// shader's; see its header for why the barriers land where they do).
// ---------------------------------------------------------------------------
groupshared int sStack[STACK_SIZE];   // bv32 to-be-expanded node stack
groupshared uint sBallot;             // __ballot_sync mask
groupshared uint sScanStart;          // atomicAdd result broadcast
groupshared uint sScanCount[WARP];    // EE per-lane count for the exclusive prefix sum

// __ballot_sync over the 32-thread group.  The leading barrier orders the zero-write after
// every read of the previous ballot's mask.
uint ballotGroup(bool pred, uint lane)
{
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0u) sBallot = 0u;
    GroupMemoryBarrierWithGroupSync();
    if (pred) InterlockedOr(sBallot, 1u << lane);
    GroupMemoryBarrierWithGroupSync();
    return sBallot;
}

// PxgFemFemContactInfo store (16B: pairInd0 u64 @0, pairInd1 u32 @8, auxInd u32 @12).
void writeContactInfo(uint index, uint pairInd0, uint pairInd1, uint auxInd)
{
    const uint64_t infoBase = gD.outContactInfo + uint64_t(index) * 16UL;
    Ptr<uint2>(infoBase)[0] = uint2(pairInd0, 0u);   // pairInd0 is PxU64; encode() fits 32b -> high 0
    Ptr<uint>(infoBase + 8UL)[0] = pairInd1;
    Ptr<uint>(infoBase + 12UL)[0] = auxInd;
}

// The pair's two cloth ids and the summed contact distance, read once per group.
struct ClothPair
{
    uint clothId0;
    uint clothId1;
    float contactDist;
}

ClothPair loadClothPair(uint cmIdx)
{
    const PxgContactManagerInput cin = Ptr<PxgContactManagerInput>(gD.cmInputs)[cmIdx];
    Ptr<PxgShape> shapes = Ptr<PxgShape>(gD.gpuShapes);

    ClothPair p;
    p.clothId0 = shapes[cin.shapeRef0].particleOrSoftbodyId;
    p.clothId1 = shapes[cin.shapeRef1].particleOrSoftbodyId;
    // contactDistance[transformCacheRef0] + contactDistance[transformCacheRef1]
    p.contactDist = Ptr<float>(gD.contactDistance)[cin.transformCacheRef0] +
                    Ptr<float>(gD.contactDistance)[cin.transformCacheRef1];
    return p;
}

// ===========================================================================
// cloth_differentClothCollisionVTLaunch — one 32-thread group per cloth VERTEX of
// EITHER cloth; the first numVerts0 queries walk cloth1's tree, the rest cloth0's.
// ===========================================================================
[shader("compute")]
[numthreads(32, 1, 1)]
void cloth_differentClothCollisionVTLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    // Early exit if contact pairs are not updated (*updateContactPairs == 0; PxU8).
    if ((Ptr<uint>(gD.updateContactPairs)[0] & 0xFFu) == 0u)
        return;

    const uint lane = gtid.x;
    const uint cmIdx = gid.y;

    const ClothPair pair = loadClothPair(cmIdx);
    Ptr<PxgFEMClothGpu> clothes = Ptr<PxgFEMClothGpu>(gD.femClothes);

    const uint numVertices0 = clothes[pair.clothId0].mNbVerts;
    const uint numVertices1 = clothes[pair.clothId1].mNbVerts;
    const uint numQueries = numVertices0 + numVertices1;

    const float contactDist = pair.contactDist;
    const float contactDistSq = contactDist * contactDist;

    // See the header: the engine packs MIDPHASE_WARPS_PER_BLOCK warps into one block and
    // launches ceil(2 * maxClothVerts / MIDPHASE_WARPS_PER_BLOCK) of them, so the group
    // walks its block's warps serially.  `sub` and the break are group-uniform.
    for (uint sub = 0u; sub < MIDPHASE_WARPS_PER_BLOCK; ++sub)
    {
    GroupMemoryBarrierWithGroupSync();   // groupshared scratch is reused per sub-warp
    const uint globalWarpIdx = gid.x * MIDPHASE_WARPS_PER_BLOCK + sub;
    if (globalWarpIdx >= numQueries)
        break;

    const bool isCloth0 = globalWarpIdx < numVertices0;
    const uint vertexIdx = isCloth0 ? globalWarpIdx : globalWarpIdx - numVertices0;

    // Branchless id select, transcribed verbatim (femClothClothMidPhase.cu:1277-1279).
    const int selMask = isCloth0 ? 0 : 1;
    const uint myClothId = uint(selMask * (int(pair.clothId1) - int(pair.clothId0)) + int(pair.clothId0));
    const uint otherClothId = uint(selMask * (int(pair.clothId0) - int(pair.clothId1)) + int(pair.clothId1));

    // The query vertex is MINE; the tree, its vertices and its indices are the OTHER's.
    Ptr<float4> verts0 = Ptr<float4>(clothes[myClothId].mPosition_InvMass);
    Ptr<float4> meshVerts = Ptr<float4>(clothes[otherClothId].mPosition_InvMass);
    Ptr<uint4> triIndices = Ptr<uint4>(clothes[otherClothId].mTriangleVertexIndices);
    const uint64_t treeAddr = clothes[otherClothId].mTriMeshData + 16UL;   // + sizeof(uint4)

    const float3 vertex = verts0[vertexIdx].xyz;

    // markValidity(clothId0 != clothId1) — different cloths are valid immediately.
    const uint validityBits = (myClothId != otherClothId) ? VALIDITY_FLAG_MASK : 0u;

    uint currentNodeIndex = 0u;   // root
    bool hasNode = true;
    uint nbStack = 0u;

    while (hasNode)
    {
        const uint nodeIdx = currentNodeIndex >> 6u;

        if ((currentNodeIndex & 1u) != 0u)
        {
            // leaf: primitive range lives in the PARENT node's mData slot.
            const uint leafOffset = (currentNodeIndex >> 1u) & 31u;
            const uint data = bvData(treeAddr, nodeIdx, leafOffset);
            const uint nbPrimitives = (data >> 1u) & 63u;
            const uint primStartIndex = data >> 7u;
            uint primitiveIndex = 0xffffffffu;
            if (lane < nbPrimitives)
                primitiveIndex = lane + primStartIndex;

            // ClothTreeVertexTraverser::intersectPrimitiveFullWarp (different-cloth path).
            bool intersect = false;
            if (primitiveIndex != 0xffffffffu)
            {
                const uint4 triIdx1 = triIndices[primitiveIndex];
                // `clothId0 != clothId1 | !(vertexIdx == triIdx1.x|y|z)` — the corner skip
                // only bites on the self path, but it is transcribed, not folded away.
                const bool vtInTri = (vertexIdx == triIdx1.x) || (vertexIdx == triIdx1.y) ||
                                     (vertexIdx == triIdx1.z);
                if ((myClothId != otherClothId) || !vtInTri)
                {
                    const float3 x1 = meshVerts[triIdx1.x].xyz;
                    const float3 x2 = meshVerts[triIdx1.y].xyz;
                    const float3 x3 = meshVerts[triIdx1.z].xyz;

                    // AABB(x1,x2,x3).fattenFast(contactDist).contains(vertex)
                    const float3 bMin = min(min(x1, x2), x3) - contactDist;
                    const float3 bMax = max(max(x1, x2), x3) + contactDist;
                    const bool contains = !(vertex.x < bMin.x || vertex.x > bMax.x ||
                                            vertex.y < bMin.y || vertex.y > bMax.y ||
                                            vertex.z < bMin.z || vertex.z > bMax.z);
                    if (contains)
                    {
                        float s, t;
                        const float3 closestPt = closestPtPointTriangle(vertex, x1, x2, x3, s, t);
                        const float3 dir = vertex - closestPt;
                        const float distSq = dot(dir, dir);
                        // The rest-pose filter collapses here (NULL rest buffers, filterDistSq 0)
                        // — see the header, difference 5.
                        if (distSq < contactDistSq && distSq > 1.0e-14f)
                            intersect = true;
                    }
                }
            }

            // globalScanExclusiveSingleWarp(intersect, totalContactCount).
            const uint mask = ballotGroup(intersect, lane);
            const uint offset = countbits(mask & ((1u << lane) - 1u));
            const uint validCount = countbits(mask);
            if (lane == 0u)
            {
                uint start = 0u;
                if (validCount > 0u)
                    InterlockedAdd(Ptr<uint>(gD.totalContactCount)[0], validCount, start);
                sScanStart = start;
            }
            GroupMemoryBarrierWithGroupSync();
            const uint index = sScanStart + offset;
            if (intersect && index < gD.maxContacts)
            {
                // markVertexTrianglePair() clears bit 30 (already 0 on the zero-init).
                writeContactInfo(index, pxEncodeClothIndex(myClothId, vertexIdx),
                                 pxEncodeClothIndex(otherClothId, primitiveIndex), validityBits);
            }
        }
        else
        {
            // internal node.
            const uint nbChildren = bvNbNodes(treeAddr, nodeIdx);
            float3 mn = float3(0, 0, 0), mx = float3(0, 0, 0);
            const bool hasBox = lane < nbChildren;
            if (hasBox)
            {
                mn = bvMin(treeAddr, nodeIdx, lane);
                mx = bvMax(treeAddr, nodeIdx, lane);
            }
            // ClothTreeVertexTraverser::intersectBoxFullWarp: point-AABB distance.
            bool intersect = false;
            if (hasBox)
            {
                const float3 closest = max(mn, min(vertex, mx));
                const float3 dd = closest - vertex;
                intersect = dot(dd, dd) <= contactDistSq;
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

            const uint mask = ballotGroup(isValidNode, lane);
            const uint offset = countbits(mask & ((1u << lane) - 1u));
            const uint validCount = countbits(mask);
            if (isValidNode)
                sStack[nbStack + offset] = int(currentIndex);
            nbStack += validCount;
        }

        GroupMemoryBarrierWithGroupSync();
        if (nbStack > 0u)
        {
            nbStack -= 1u;
            currentNodeIndex = uint(sStack[nbStack]);
        }
        else
            hasNode = false;
    }
    }
}

// ===========================================================================
// cloth_differentClothCollisionEELaunch — one 32-thread group per triangle-with-
// active-edges of the SMALLER cloth, walking the larger cloth's tree.
// ===========================================================================
[shader("compute")]
[numthreads(32, 1, 1)]
void cloth_differentClothCollisionEELaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    if ((Ptr<uint>(gD.updateContactPairs)[0] & 0xFFu) == 0u)
        return;

    const uint lane = gtid.x;
    const uint cmIdx = gid.y;

    const ClothPair pair = loadClothPair(cmIdx);
    Ptr<PxgFEMClothGpu> clothes = Ptr<PxgFEMClothGpu>(gD.femClothes);

    const uint numTriangles0 = clothes[pair.clothId0].mNbTrianglesWithActiveEdges;
    const uint numTriangles1 = clothes[pair.clothId1].mNbTrianglesWithActiveEdges;
    const uint numQueries = min(numTriangles0, numTriangles1);

    const float contactDist = pair.contactDist;
    const float contactDistSq = contactDist * contactDist;

    // The SMALLER cloth (by triangles-with-active-edges) always drives the query; this is
    // loop-invariant, unlike the VT kernel's per-warp split.
    const bool isCloth0 = numTriangles0 <= numTriangles1;
    const int selMask = isCloth0 ? 0 : 1;
    const uint myClothId = uint(selMask * (int(pair.clothId1) - int(pair.clothId0)) + int(pair.clothId0));
    const uint otherClothId = uint(selMask * (int(pair.clothId0) - int(pair.clothId1)) + int(pair.clothId1));

    Ptr<float4> verts0 = Ptr<float4>(clothes[myClothId].mPosition_InvMass);
    Ptr<uint4> myTriIndices = Ptr<uint4>(clothes[myClothId].mTriangleVertexIndices);
    Ptr<uint> trisWithActiveEdges = Ptr<uint>(clothes[myClothId].mTrianglesWithActiveEdges);

    Ptr<float4> meshVerts = Ptr<float4>(clothes[otherClothId].mPosition_InvMass);
    Ptr<uint4> triIndices = Ptr<uint4>(clothes[otherClothId].mTriangleVertexIndices);
    const uint64_t treeAddr = clothes[otherClothId].mTriMeshData + 16UL;

    const uint validityBits = (myClothId != otherClothId) ? VALIDITY_FLAG_MASK : 0u;

    static const uint nextE_lookup[3] = { 1u, 2u, 0u };

    for (uint sub = 0u; sub < MIDPHASE_WARPS_PER_BLOCK; ++sub)
    {
    GroupMemoryBarrierWithGroupSync();   // groupshared scratch is reused per sub-warp
    const uint globalWarpIdx = gid.x * MIDPHASE_WARPS_PER_BLOCK + sub;
    if (globalWarpIdx >= numQueries)
        break;

    // setEdge0: the query triangle (t0) at globalWarpIdx, from MY cloth.
    const uint t0_index = trisWithActiveEdges[globalWarpIdx];
    const uint4 t0_tri = myTriIndices[t0_index];        // t0_triVertIndices0
    const uint t0_edgeAuthorship = t0_tri.w;
    const uint vIndices0[3] = { t0_tri.x, t0_tri.y, t0_tri.z };
    const float3 t0_points[3] = { verts0[t0_tri.x].xyz, verts0[t0_tri.y].xyz, verts0[t0_tri.z].xyz };
    const float3 t0_min = min(min(t0_points[0], t0_points[1]), t0_points[2]) - contactDist;  // fattenFast
    const float3 t0_max = max(max(t0_points[0], t0_points[1]), t0_points[2]) + contactDist;

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

            // ClothTreeEdgeTraverser::intersectPrimitiveFullWarp (different-cloth path).
            uint2 localEdgeIndices[9];   // up to 3x3 edge pairs
            uint intersectCount = 0u;

            // `primitiveIndex != 0xFFFFFFFF & (clothId0 != clothId1 | t0_index > primIdx)`:
            // the half-pair dedup is a SELF-path device (both triangles come from the same
            // BVH there); across two cloths every ordered pair is queried exactly once.
            if (primitiveIndex != 0xffffffffu &&
                ((myClothId != otherClothId) || (t0_index > primitiveIndex)))
            {
                const uint4 triVertexIdx1 = triIndices[primitiveIndex];
                const uint edgeAuthorship1 = triVertexIdx1.w;
                const uint vIndices1[3] = { triVertexIdx1.x, triVertexIdx1.y, triVertexIdx1.z };
                const float3 t1_points[3] = { meshVerts[vIndices1[0]].xyz, meshVerts[vIndices1[1]].xyz,
                                              meshVerts[vIndices1[2]].xyz };
                const float3 t1_min = min(min(t1_points[0], t1_points[1]), t1_points[2]);
                const float3 t1_max = max(max(t1_points[0], t1_points[1]), t1_points[2]);

                if (boundsIntersect(t0_min, t0_max, t1_min, t1_max))
                {
                    // Iterate through edges of triangle 1 (the found primitive).
                    for (uint e1 = 0u; e1 < 3u; ++e1)
                    {
                        if (!isType0EdgeActive(edgeAuthorship1, e1))
                            continue;

                        const uint nextE1 = nextE_lookup[e1];
                        const uint e1_v0 = vIndices1[e1];
                        const uint e1_v1 = vIndices1[nextE1];
                        const float3 e1_x0 = t1_points[e1];
                        const float3 e1_x1 = t1_points[nextE1];

                        // Iterate through edges of triangle 0 (the query).
                        for (uint e0 = 0u; e0 < 3u; ++e0)
                        {
                            if (!isType0EdgeActive(t0_edgeAuthorship, e0))
                                continue;

                            const uint nextE0 = nextE_lookup[e0];
                            const uint e0_v0 = vIndices0[e0];
                            const uint e0_v1 = vIndices0[nextE0];

                            // `(clothId0 == clothId1) & (shares an endpoint)` — two distinct
                            // cloths never share a vertex INDEX space, so this is dead here.
                            if ((myClothId == otherClothId) &&
                                ((e0_v0 == e1_v0) || (e0_v0 == e1_v1) || (e0_v1 == e1_v0) || (e0_v1 == e1_v1)))
                                continue;

                            const float3 e0_x0 = t0_points[e0];
                            const float3 e0_x1 = t0_points[nextE0];

                            float s, t, distSq;
                            closestPtLineLine(e0_x0, e0_x1, e1_x0, e1_x1, s, t, distSq);
                            const bool isOutside = (s < DEFORMABLE_BARYCENTRIC_THRESHOLD) ||
                                                   (s > DEFORMABLE_ONE_MINUS_BARYCENTRIC_THRESHOLD) ||
                                                   (t < DEFORMABLE_BARYCENTRIC_THRESHOLD) ||
                                                   (t > DEFORMABLE_ONE_MINUS_BARYCENTRIC_THRESHOLD);
                            // Vertex-edge collisions are handled in the vertex-triangle query.
                            if (isOutside)
                                continue;

                            // The rest-pose filter collapses here — see the header, difference 5.
                            if (distSq < contactDistSq && distSq > 1.0e-14f)
                            {
                                localEdgeIndices[intersectCount] = uint2(e0, e1);
                                ++intersectCount;
                            }
                        }
                    }
                }
            }

            // Exit early if no intersection anywhere in the warp.
            const uint mask = ballotGroup(intersectCount > 0u, lane);
            const uint validCount = countbits(mask);
            if (validCount != 0u)
            {
                // globalScanExclusiveSingleWarp(intersectCount, ...) — multi-element.
                sScanCount[lane] = intersectCount;
                GroupMemoryBarrierWithGroupSync();
                uint offset = 0u;
                for (uint j = 0u; j < lane; ++j)
                    offset += sScanCount[j];
                if (lane == 31u)
                {
                    uint start = 0u;
                    const uint total = offset + intersectCount;
                    if (total > 0u)
                        InterlockedAdd(Ptr<uint>(gD.totalContactCount)[0], total, start);
                    sScanStart = start;
                }
                GroupMemoryBarrierWithGroupSync();
                const uint index = sScanStart + offset;
                if (index < gD.maxContacts)
                {
                    const uint pairInd0 = pxEncodeClothIndex(myClothId, t0_index);
                    const uint pairInd1 = pxEncodeClothIndex(otherClothId, primitiveIndex);
                    for (uint li = 0u; li < intersectCount; ++li)
                    {
                        const uint2 aux = localEdgeIndices[li];
                        uint auxInd = PAIR_TYPE_FLAG_MASK | validityBits;   // markEdgeEdgePair + markValidity
                        auxInd = (auxInd & ~AUX_IND0_MASK) | (aux.x & AUX_IND0_MASK);              // setAuxInd0(e0)
                        auxInd = (auxInd & ~AUX_IND1_MASK) |
                                 ((aux.y & ((1u << AUX_IND1_BITS) - 1u)) << AUX_IND1_SHIFT);        // setAuxInd1(e1)
                        writeContactInfo(index + li, pairInd0, pairInd1, auxInd);
                    }
                }
            }
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
            // ClothTreeEdgeTraverser::intersectBoxFullWarp: t1_bounds.intersects(t0_bounds).
            bool intersect = false;
            if (hasBox)
                intersect = boundsIntersect(t0_min, t0_max, mn, mx);

            uint currentIndex = (currentNodeIndex & 0xFFFFFFC0u) | (lane << 1u) | 1u;
            bool isValidNode = false;
            if (lane < nbChildren)
            {
                const uint data = bvData(treeAddr, nodeIdx, lane);
                isValidNode = intersect;
                if (isValidNode && (data & 1u) == 0u)
                    currentIndex = (data >> GU_BV4_CHILD_OFFSET_SHIFT_COUNT) << 6u;
            }

            const uint mask = ballotGroup(isValidNode, lane);
            const uint offset = countbits(mask & ((1u << lane) - 1u));
            const uint validCount = countbits(mask);
            if (isValidNode)
                sStack[nbStack + offset] = int(currentIndex);
            nbStack += validCount;
        }

        GroupMemoryBarrierWithGroupSync();
        if (nbStack > 0u)
        {
            nbStack -= 1u;
            currentNodeIndex = uint(sStack[nbStack]);
        }
        else
            hasNode = false;
    }
    }
}
