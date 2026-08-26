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

// femClothSelfCollisionBda.slang
//
// P7.2 tranche 1 — cloth SELF-collision midphase + contact GENERATION (the first
// tranche of the cloth-fold-on-itself arc) on the Vulkan engine ABI (BDA).
// Ported line-for-line from gpunarrowphase/src/CUDA/femClothClothMidPhase.cu:
//   - cloth_selfCollisionMidphaseVTLaunch (:1163, launch PxgFEMClothCore.cpp:709,
//     CLOTH_SELFCOLLISION_MIDPHASE_VT): one warp per cloth VERTEX walks the cloth's
//     own BV32 triangle BVH (cloth_selfCollisionMidphaseCoreVT + the
//     ClothTreeVertexTraverser callback) and appends a vertex-triangle
//     PxgFemFemContactInfo for every penetrating (vertex, triangle) pair.
//   - cloth_selfCollisionMidphaseEELaunch (:1196, launch PxgFEMClothCore.cpp:710,
//     CLOTH_SELFCOLLISION_MIDPHASE_EE): one warp per triangle-with-active-edges
//     walks the same BVH (cloth_selfCollisionMidphaseCoreEE + ClothTreeEdgeTraverser)
//     and appends an edge-edge PxgFemFemContactInfo for every crossing edge pair.
//
// The BV32 warp-cooperative tree walk (bv32Traversal.cuh::bv32TreeTraversal) and the
// Gu::BV32DataPacked leaf/child decode are reused VERBATIM from the proven npBv32 /
// npClothCloth Slang ports (this file is their BDA sibling: Ptr<> reads + push
// constants instead of descriptor-set RWStructuredBuffers).  The vertex-triangle /
// edge-edge closest-point tests are the proven npClosest::closestPtPointTriangle /
// npEdgeEdge::closestPtLineLine transcriptions.
//
// Doctrine (identical to the prior cloth BDA tranches femClothContactGenBda /
// femClothRigidContactBda):
//   - the PxgFEMCloth descriptor arrives as a device-pointer ARRAY read through Ptr<>
//     (the byte-exact pxgSlangFEMClothDesc mirror); mTriMeshData @0, mPosition_InvMass
//     @16, mRestPosition @24, mTrianglesWithActiveEdges @64, mTriangleVertexIndices
//     @72, mSelfCollisionFilterDistance @312, mNbVerts @316, mNbTrianglesWithActiveEdges
//     @328, mElementIndex @368, mFlagsWord @376 (mBodyFlags = byte 1) are the fields
//     the two kernels touch;
//   - the BV32 tree hangs off mTriMeshData: bv32PackedNodes = (PxU8*)mTriMeshData +
//     sizeof(uint4) (the +16 header skip); Gu::BV32DataPacked is a 1168-byte node
//     (mMin[32] float4 @0, mMax[32] float4 @512, mData[32] uint @1024, mNbNodes @1152)
//     read field-by-field by explicit byte offset through Ptr<> (no whole-node copy);
//   - the emitted PxgFemFemContactInfo (16B: pairInd0 u64 @0, pairInd1 u32 @8, auxInd
//     u32 @12, PxgFEMCore.h:51) is written field-by-field, exactly as
//     femClothContactGenBda wrote PxgFemOtherContactInfo.
//
// SELF ONLY: this launch always has clothId0 == clothId1 (both == the active cloth).
// Every CUDA branch of shape `clothId0 != clothId1` therefore folds to a constant here
// (VT: `clothId0 != clothId1 | !(vtInTri)` -> `!(vtInTri)`;  EE primitive gate:
// `clothId0 != clothId1 | t0_index > primIdx` -> `t0_index > primIdx`;  EE connected-
// edge skip: `(clothId0 == clothId1) & (...)` -> `(...)`;  markValidity(clothId0 !=
// clothId1) -> markValidity(false)).  The different-cloth siblings
// (cloth_differentClothCollision{VT,EE}Launch) are a SEPARATE arg pack / ABI and are
// out of scope for this tranche.
//
// DEVIATIONS from CUDA (recipe policy; each preserves the emitted contact SET and the
// running total, and is exercised by testFemClothSim.cpp):
//   - NO 32-lane warp intrinsics and ONE 32-thread group per query (numthreads
//     (32,1,1)) in place of the CUDA (WARP_SIZE, MIDPHASE_WARPS_PER_BLOCK=2, 1) block.
//     The bv32 walk is warp-cooperative with a per-warp groupshared stack and a
//     variable-length (data-dependent) traversal loop, so packing >1 warp per block
//     would put the workgroup barrier in non-uniform control flow across warps.  With
//     one group == one warp == one query, currentNodeIndex / nbStack / the leaf-vs-
//     internal branch are all GROUP-UNIFORM, so every GroupMemoryBarrierWithGroupSync
//     is reached uniformly (the proven npBv32 / npClothCloth lowering).  Merging N
//     independent warps into one block vs N blocks is pure scheduling — identical
//     emitted contact set and atomic total.  Dispatch MUST be grid (numQueries,
//     nbActiveCloths, 1), block (32,1,1);
//   - __ballot_sync -> groupshared InterlockedOr mask (order-independent);
//     __shfl_sync(...,0/31) -> groupshared broadcast; warpScanExclusive ->
//     countbits(mask & lanemask_lt) (single-element, VT) or a groupshared prefix sum
//     over the per-lane count (multi-element, EE) — output-identical, same lowering
//     as psContactGenBda / npBv32;
//   - the per-warp femMidphaseScratch (meshVerts / indices / bv32 pointers cached by
//     lane 0 + __syncthreads) is dropped: every lane reads those addresses directly
//     through Ptr<> (semantically identical read-fanout; same as femClothContactGenBda);
//   - PxgFemFemContactInfo's local `contactInfo` is left with an INDETERMINATE auxInd
//     in CUDA outside the bits the mark*/setAux* calls touch (VT: everything but bits
//     30/31;  EE: everything but bits 0-28/30/31, i.e. the InCollision bit 29).  This
//     port zero-initializes auxInd so the emitted word is DETERMINISTIC.  Behavior-
//     preserving: every bit the downstream self-contact prepare/solve reads is written
//     explicitly by the same mark*/setAux* sequence; the zero-init only pins the
//     otherwise-dead InCollision bit (29) and — for VT — the unused aux payload (0-28)
//     to 0.  markInCollision is never called by the midphase (set later in the solve);
//   - distSqInRestPose's `restVerts == NULL -> PX_MAX_REAL` guard is the DIFFERENT-cloth
//     path's signal (it passes NULL for one rest buffer).  Self-collision always passes
//     cloth->mRestPosition (non-null), so the guard is dead here and omitted;
//   - the clothClothPairs / numClothClothPairs args are carried in the push blob for
//     ABI parity but NOT read: the CUDA ClothTree{Vertex,Edge}Traverser store them but
//     apply NO filter-pair lookup on the self path ("No filtering applied yet") — the
//     only filtering is the rest-pose distance test (distSqInRestPose > filterDistSq).
//
// All float arithmetic is NoContraction-stamped (PXG_NOCONTRACT_SHADERS): the VT
// (vertex - closestPt).dot feeds the compared `< contactDistSq`; the EE
// closestPtLineLine dot/cross/div chain feeds the compared `distSq < contactDistSq`
// and the barycentric `s/t` compared against the DEFORMABLE thresholds; the box tests'
// clamp/dot feed the compared child-overlap; the rest-pose dots feed the compared
// `> filterDistSq` — any of which would fuse to an FMA otherwise, breaking bitwise
// parity with the -ffp-contract=off CPU reference.  There is NO transcendental in the
// closest-point path (pure mul/add/sub/div/sqrt/min/max/abs) — this tranche is FULLY
// bit-exact, no tolerance seam.

import pxgSlangFEMClothDesc;   // PxgFEMClothGpu (mTriMeshData@0, mPosition_InvMass@16, ...)
// P8.15: the BV32 readers, the two closest-point routines, the bounds/edge predicates and the
// bit layouts moved out to a shared module when the cloth-vs-DIFFERENT-cloth midphase became a
// second consumer.  Only what is genuinely kernel-independent went; writeContactInfo and
// ballotGroup stay here because they touch this shader's push constants and groupshared state,
// and the rest-pose helpers stay because only the SELF path has rest buffers to read.
import pxgSlangClothMidphase;

// PxDeformableBodyFlag::eDISABLE_SELF_COLLISION (PxDeformableBodyFlag.h:47) = 1<<0,
// stored in mBodyFlags = byte 1 of mFlagsWord (@376; byte0 mActorFlags, byte1
// mBodyFlags, bytes2-3 mSurfaceFlags).
static const uint DISABLE_SELF_COLLISION = 1u;

// ---------------------------------------------------------------------------
// Push-constant blob — BYTE-IDENTICAL to the CUDA selfCollisionKernelParams pack
// (PxgFEMClothCore.cpp:698): { PxU32 maxContacts, const PxU32* activeClothes,
// const PxReal* contactDistance, const PxgFEMCloth* clothes,
// PxgNonRigidFilterPair* clothClothPairs, PxU32 numClothClothPairs,
// const PxU8* updateContactPairs, PxgFemFemContactInfo* outContactInfo,
// PxU32* totalContactCount } -> 72-byte natural-alignment blob.
// (ABI row {4,8,8,8,8,4,8,8,8}.)
struct SelfCollisionParams
{
    uint maxContacts;             // @0
    uint _pad4;                   // @4
    uint64_t activeClothes;       // @8
    uint64_t contactDistance;     // @16
    uint64_t clothes;             // @24
    uint64_t clothClothPairs;     // @32  (unused; ABI parity)
    uint numClothClothPairs;      // @40  (unused; ABI parity)
    uint _pad44;                  // @44
    uint64_t updateContactPairs;  // @48
    uint64_t outContactInfo;      // @56  output
    uint64_t totalContactCount;   // @64  output -> 72
}

[[vk::push_constant]] ConstantBuffer<SelfCollisionParams> gP;

// distSqInRestPose (femClothClothMidPhase.cu:662, VT overload) — restVerts0==restVerts1
// on the self path, so a single Ptr suffices.
float distSqInRestPoseVT(Ptr<float4> restVerts, uint restVertexIndex, uint4 triIndex)
{
    const float3 x1 = restVerts[triIndex.x].xyz;
    const float3 x2 = restVerts[triIndex.y].xyz;
    const float3 x3 = restVerts[triIndex.z].xyz;
    const float3 x0 = restVerts[restVertexIndex].xyz;

    const float d1 = dot(x1 - x0, x1 - x0);
    const float d2 = dot(x2 - x0, x2 - x0);
    const float d3 = dot(x3 - x0, x3 - x0);
    return min(min(d1, d2), d3);
}

// distSqInRestPose (femClothClothMidPhase.cu:685, EE overload).
float distSqInRestPoseEE(Ptr<float4> restVerts, uint e0v0, uint e0v1, uint e1v0, uint e1v1)
{
    const float3 a = restVerts[e0v0].xyz;
    const float3 b = restVerts[e0v1].xyz;
    const float3 c = restVerts[e1v0].xyz;
    const float3 d = restVerts[e1v1].xyz;

    const float d0 = dot(a - c, a - c);
    const float d1 = dot(a - d, a - d);
    const float d2 = dot(b - c, b - c);
    const float d3 = dot(b - d, b - d);
    return min(min(d0, d1), min(d2, d3));
}

// ---------------------------------------------------------------------------
// Groupshared state for the one-warp-per-query lowering.
// ---------------------------------------------------------------------------
groupshared int sStack[STACK_SIZE];   // bv32 to-be-expanded node stack
groupshared uint sBallot;             // __ballot_sync mask
groupshared uint sScanStart;          // atomicAdd result broadcast
groupshared uint sScanCount[WARP];    // EE per-lane count for the exclusive prefix sum

// __ballot_sync over the 32-thread group.  The leading barrier orders the zero-write
// after every read of the previous ballot's mask.
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
    const uint64_t infoBase = gP.outContactInfo + uint64_t(index) * 16UL;
    Ptr<uint2>(infoBase)[0] = uint2(pairInd0, 0u);   // pairInd0 is PxU64; encode() fits 32b -> high 0
    Ptr<uint>(infoBase + 8UL)[0] = pairInd1;
    Ptr<uint>(infoBase + 12UL)[0] = auxInd;
}

// ===========================================================================
// cloth_selfCollisionMidphaseVTLaunch — one 32-thread group per cloth VERTEX.
// ===========================================================================
[shader("compute")]
[numthreads(32, 1, 1)]
void cloth_selfCollisionMidphaseVTLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    // Early exit if contact pairs are not updated (*updateContactPairs == 0; PxU8).
    if ((Ptr<uint>(gP.updateContactPairs)[0] & 0xFFu) == 0u)
        return;

    const uint lane = gtid.x;
    const uint cmIdx = gid.y;
    const uint clothId = Ptr<uint>(gP.activeClothes)[cmIdx];
    Ptr<PxgFEMClothGpu> clothes = Ptr<PxgFEMClothGpu>(gP.clothes);

    // mBodyFlags & eDISABLE_SELF_COLLISION (mBodyFlags = byte 1 of mFlagsWord).
    const uint bodyFlags = (clothes[clothId].mFlagsWord >> 8u) & 0xFFu;
    if ((bodyFlags & DISABLE_SELF_COLLISION) != 0u)
        return;

    // *** GRID.  The engine packs MIDPHASE_WARPS_PER_BLOCK warps into ONE block and
    // launches only ceil(N / MIDPHASE_WARPS_PER_BLOCK) of them
    // (PxgFEMClothCore.cpp:714).  The Vulkan bridge DROPS blockDim and passes only
    // gridDim through (PxgVulkanCudaContext.cpp:293), so a [numthreads(32,1,1)] group
    // is one warp -- and indexing the query by gid.x alone covered just the FIRST HALF
    // of the cloth vertices.  It did exactly that until this loop.
    //
    // The one-group-per-warp lowering is KEPT (see the header: packing >1 warp into a
    // group would put the traversal's barriers in non-uniform control flow across
    // warps).  The block's warps are walked SERIALLY instead: `sub` and the break are
    // group-uniform, so every barrier below is still reached uniformly.  Same emitted
    // contact set as CUDA, different scheduling.
    //
    // Landing this DOUBLED the self-collision contact count (VT 76 -> 152, EE 34 -> 70)
    // and forced clothselfcollide to be re-tuned: the inherited per-pass overshoot in
    // the contact solve (mDeltaPos.w accumulates |coeff| rather than 1 -- see
    // femClothSimSelfContactManyTriangles) is absorbed by whichever endpoint acts as
    // the VERTEX side, and with correct coverage twice as many vertices do.
    const uint numVerts = clothes[clothId].mNbVerts;
    for (uint sub = 0u; sub < MIDPHASE_WARPS_PER_BLOCK; ++sub)
    {
    GroupMemoryBarrierWithGroupSync();   // groupshared scratch is reused per sub-warp
    const uint globalWarpIdx = gid.x * MIDPHASE_WARPS_PER_BLOCK + sub;
    if (globalWarpIdx >= numVerts)
        break;

    const uint elementInd = clothes[clothId].mElementIndex;
    const float contactDist = Ptr<float>(gP.contactDistance)[elementInd] * 2.0f;
    const float filterDistance = clothes[clothId].mSelfCollisionFilterDistance;
    const float filterDistSq = filterDistance * filterDistance;
    const float contactDistSq = contactDist * contactDist;

    Ptr<float4> meshVerts = Ptr<float4>(clothes[clothId].mPosition_InvMass);
    Ptr<uint4> triIndices = Ptr<uint4>(clothes[clothId].mTriangleVertexIndices);
    Ptr<float4> restVerts = Ptr<float4>(clothes[clothId].mRestPosition);
    const uint64_t treeAddr = clothes[clothId].mTriMeshData + 16UL;   // + sizeof(uint4)

    const uint vertexIdx = globalWarpIdx;
    const float3 vertex = meshVerts[vertexIdx].xyz;

    uint currentNodeIndex = 0u;   // root
    bool hasNode = true;
    uint nbStack = 0u;

    while (hasNode)
    {
        const uint nodeIdx = bvNodeIdxOf(currentNodeIndex);

        if (bvIsLeaf(currentNodeIndex))
        {
            // leaf: primitive range lives in the PARENT node's mData slot.
            const uint leafOffset = bvLeafSlotOf(currentNodeIndex);
            const uint data = bvData(treeAddr, nodeIdx, leafOffset);
            const uint nbPrimitives = bvDataNbPrimitives(data);
            const uint primStartIndex = bvDataPrimStart(data);
            uint primitiveIndex = 0xffffffffu;
            if (lane < nbPrimitives)
                primitiveIndex = lane + primStartIndex;

            // ClothTreeVertexTraverser::intersectPrimitiveFullWarp (self path).
            bool intersect = false;
            if (primitiveIndex != 0xffffffffu)
            {
                const uint4 triIdx1 = triIndices[primitiveIndex];
                const bool vtInTri = (vertexIdx == triIdx1.x) || (vertexIdx == triIdx1.y) || (vertexIdx == triIdx1.z);
                if (!vtInTri)
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
                        if (distSq < contactDistSq && distSq > 1.0e-14f)
                        {
                            const float tempRestDistSq = distSqInRestPoseVT(restVerts, vertexIdx, triIdx1);
                            if (tempRestDistSq > filterDistSq)
                                intersect = true;
                        }
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
                    InterlockedAdd(Ptr<uint>(gP.totalContactCount)[0], validCount, start);
                sScanStart = start;
            }
            GroupMemoryBarrierWithGroupSync();
            const uint index = sScanStart + offset;
            if (intersect && index < gP.maxContacts)
            {
                // markVertexTrianglePair() + markValidity(false) on zero-init -> auxInd 0.
                writeContactInfo(index, pxEncodeClothIndex(clothId, vertexIdx),
                                 pxEncodeClothIndex(clothId, primitiveIndex), 0u);
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

            uint currentIndex = bvChildIndex(currentNodeIndex, lane);
            bool isValidNode = false;
            if (lane < nbChildren)
            {
                const uint data = bvData(treeAddr, nodeIdx, lane);
                isValidNode = intersect;
                if (isValidNode && bvDataIsInternal(data))
                    currentIndex = bvDataChildNodeIndex(data);
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
// cloth_selfCollisionMidphaseEELaunch — one 32-thread group per triangle-with-
// active-edges.
// ===========================================================================
[shader("compute")]
[numthreads(32, 1, 1)]
void cloth_selfCollisionMidphaseEELaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    if ((Ptr<uint>(gP.updateContactPairs)[0] & 0xFFu) == 0u)
        return;

    const uint lane = gtid.x;
    const uint cmIdx = gid.y;
    const uint clothId = Ptr<uint>(gP.activeClothes)[cmIdx];
    Ptr<PxgFEMClothGpu> clothes = Ptr<PxgFEMClothGpu>(gP.clothes);

    const uint bodyFlags = (clothes[clothId].mFlagsWord >> 8u) & 0xFFu;
    if ((bodyFlags & DISABLE_SELF_COLLISION) != 0u)
        return;

    // *** GRID.  The engine packs MIDPHASE_WARPS_PER_BLOCK warps into ONE block and
    // launches only ceil(N / MIDPHASE_WARPS_PER_BLOCK) of them
    // (PxgFEMClothCore.cpp:714).  The Vulkan bridge DROPS blockDim and passes only
    // gridDim through (PxgVulkanCudaContext.cpp:293), so a [numthreads(32,1,1)] group
    // is one warp -- and indexing the query by gid.x alone covered just the FIRST HALF
    // of the triangles-with-active-edges.  It did exactly that until this loop.
    //
    // The one-group-per-warp lowering is KEPT (see the header: packing >1 warp into a
    // group would put the traversal's barriers in non-uniform control flow across
    // warps).  The block's warps are walked SERIALLY instead: `sub` and the break are
    // group-uniform, so every barrier below is still reached uniformly.  Same emitted
    // contact set as CUDA, different scheduling.
    //
    // Landing this DOUBLED the self-collision contact count (VT 76 -> 152, EE 34 -> 70)
    // and forced clothselfcollide to be re-tuned: the inherited per-pass overshoot in
    // the contact solve (mDeltaPos.w accumulates |coeff| rather than 1 -- see
    // femClothSimSelfContactManyTriangles) is absorbed by whichever endpoint acts as
    // the VERTEX side, and with correct coverage twice as many vertices do.
    const uint numTriangles = clothes[clothId].mNbTrianglesWithActiveEdges;
    for (uint sub = 0u; sub < MIDPHASE_WARPS_PER_BLOCK; ++sub)
    {
    GroupMemoryBarrierWithGroupSync();   // groupshared scratch is reused per sub-warp
    const uint globalWarpIdx = gid.x * MIDPHASE_WARPS_PER_BLOCK + sub;
    if (globalWarpIdx >= numTriangles)
        break;

    const uint elementInd = clothes[clothId].mElementIndex;
    const float contactDist = Ptr<float>(gP.contactDistance)[elementInd] * 2.0f;
    const float filterDistance = clothes[clothId].mSelfCollisionFilterDistance;
    const float filterDistSq = filterDistance * filterDistance;
    const float contactDistSq = contactDist * contactDist;

    Ptr<float4> meshVerts = Ptr<float4>(clothes[clothId].mPosition_InvMass);
    Ptr<uint4> triIndices = Ptr<uint4>(clothes[clothId].mTriangleVertexIndices);
    Ptr<float4> restVerts = Ptr<float4>(clothes[clothId].mRestPosition);
    Ptr<uint> trisWithActiveEdges = Ptr<uint>(clothes[clothId].mTrianglesWithActiveEdges);
    const uint64_t treeAddr = clothes[clothId].mTriMeshData + 16UL;

    // setEdge0: the query triangle (t0) at globalWarpIdx.
    const uint t0_index = trisWithActiveEdges[globalWarpIdx];
    const uint4 t0_tri = triIndices[t0_index];         // t0_triVertIndices0
    const uint t0_edgeAuthorship = t0_tri.w;
    const uint vIndices0[3] = { t0_tri.x, t0_tri.y, t0_tri.z };
    const float3 t0_points[3] = { meshVerts[t0_tri.x].xyz, meshVerts[t0_tri.y].xyz, meshVerts[t0_tri.z].xyz };
    const float3 t0_min = min(min(t0_points[0], t0_points[1]), t0_points[2]) - contactDist;  // fattenFast
    const float3 t0_max = max(max(t0_points[0], t0_points[1]), t0_points[2]) + contactDist;

    static const uint nextE_lookup[3] = { 1u, 2u, 0u };

    uint currentNodeIndex = 0u;
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
            uint primitiveIndex = 0xffffffffu;
            if (lane < nbPrimitives)
                primitiveIndex = lane + primStartIndex;

            // ClothTreeEdgeTraverser::intersectPrimitiveFullWarp (self path).
            uint2 localEdgeIndices[9];   // up to 3x3 edge pairs
            uint intersectCount = 0u;

            if (primitiveIndex != 0xffffffffu && (t0_index > primitiveIndex))
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

                            // Skip connected edges within the same cloth (self path).
                            if ((e0_v0 == e1_v0) || (e0_v0 == e1_v1) || (e0_v1 == e1_v0) || (e0_v1 == e1_v1))
                                continue;

                            const float3 e0_x0 = t0_points[e0];
                            const float3 e0_x1 = t0_points[nextE0];

                            float s, t, distSq;
                            closestPtLineLine(e0_x0, e0_x1, e1_x0, e1_x1, s, t, distSq);
                            const bool isOutside = (s < DEFORMABLE_BARYCENTRIC_THRESHOLD) ||
                                                   (s > DEFORMABLE_ONE_MINUS_BARYCENTRIC_THRESHOLD) ||
                                                   (t < DEFORMABLE_BARYCENTRIC_THRESHOLD) ||
                                                   (t > DEFORMABLE_ONE_MINUS_BARYCENTRIC_THRESHOLD);
                            if (isOutside)
                                continue;

                            if (distSq < contactDistSq && distSq > 1.0e-14f)
                            {
                                const float tempRestDistSq = distSqInRestPoseEE(restVerts, e0_v0, e0_v1, e1_v0, e1_v1);
                                if (tempRestDistSq > filterDistSq)
                                {
                                    localEdgeIndices[intersectCount] = uint2(e0, e1);
                                    ++intersectCount;
                                }
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
                        InterlockedAdd(Ptr<uint>(gP.totalContactCount)[0], total, start);
                    sScanStart = start;
                }
                GroupMemoryBarrierWithGroupSync();
                const uint index = sScanStart + offset;
                if (index < gP.maxContacts)
                {
                    const uint pairInd0 = pxEncodeClothIndex(clothId, t0_index);
                    const uint pairInd1 = pxEncodeClothIndex(clothId, primitiveIndex);
                    // markEdgeEdgePair() + markValidity(false) on zero-init -> PAIR_TYPE bit only.
                    for (uint li = 0u; li < intersectCount; ++li)
                    {
                        const uint2 aux = localEdgeIndices[li];
                        uint auxInd = PAIR_TYPE_FLAG_MASK;
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

            uint currentIndex = bvChildIndex(currentNodeIndex, lane);
            bool isValidNode = false;
            if (lane < nbChildren)
            {
                const uint data = bvData(treeAddr, nodeIdx, lane);
                isValidNode = intersect;
                if (isValidNode && bvDataIsInternal(data))
                    currentIndex = bvDataChildNodeIndex(data);
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
