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

// convexTrimeshMidphaseBda.slang
//
// `midphaseGeneratePairs` (convexMeshMidphase.cu:378) on the Vulkan engine ABI.
// The first of the six kernels behind rigid-vs-cooked-triangle-mesh collision,
// the gap P8.21 measured and P8.22 turned into a red scene (`PXG_SCENE=meshfloor`,
// where a dynamic box currently falls through the floor to y = -73.8 while still
// accelerating).
//
// Almost nothing here is new code.  The three pieces this kernel is made of were
// each extracted and certified on their own first, deliberately:
//   - `pxgSlangPcmManifold` (P8.23) — the persistent-manifold gate, 21/21 mutants,
//     with a SIX-DISPATCH fixture because a one-shot test cannot see a bug in it.
//   - `pxgSlangTriBox`      (P8.24) — Gu::intersectTriangleBox_RefImpl<true>.
//   - `pxgSlangBv32`        (P8.25) — the BV32DataPacked reader.
// What is left is the wiring, the pair emission, and the overflow clamp.
//
// ---------------------------------------------------------------------------
// GRID.  CUDA launches grid (numBlocks,1,1) x block (32, MIDPHASE_WARPS_PER_BLOCK=2, 1)
// and derives `cmIdx = threadIdx.y + blockIdx.x * blockDim.y`, i.e. ONE WARP PER
// CONTACT MANAGER, two per block.  The two warps never interact — separate
// s_warpScratch slots, separate cmIdx — so the port walks them SERIALLY inside a
// 32-thread group, the same shape softBodyMeshMidphaseBda uses and for the same
// reason: the engine picks the grid, and a 32-lane group is this port's virtual
// warp.
//
// STACK DEPTH IS 320, NOT 192.  The deformable midphases use 192 because their
// own `femMidphaseScratch` declares `int sBv32Nodes[192]`.  The convex path has a
// DIFFERENT scratch struct, `MidphaseScratch` (convexNpCommon.h:171), which
// declares `int sBv32Nodes[320]; //10 depth of the bv32 tree`.  Copying 192 from
// the deformable template would silently corrupt traversal on a deep mesh — see
// P8.22, where that exact confusion was raised against the shipped deformable
// ports and had to be checked before it could be dismissed.
//
// nbPrimitives IS CLAMPED TO 32, BUG FOR BUG.  The field is six bits (up to 63);
// `bv32Traversal.cuh` maps one lane per primitive and silently drops the rest.
// Writing the obvious base-stepped loop would emit triangles CUDA never emitted.
//
// ---------------------------------------------------------------------------
// TWO BDA ALIGNMENT TRAPS, both from odd-sized structs, both handled by reading
// and writing SCALARS:
//   - `PxBounds3` is 24 bytes, so `bounds[i]` sits at byte 24*i — 8-mod-16 for
//     odd i, which makes a Ptr<float4> read misaligned.
//   - `ConvexMeshPair` is 56 bytes, same problem at `cvxTrimeshPair[cmIdx]`.
//     Upstream writes it as a 14 x 4-byte scatter (dataReadWriteHelper.cuh:384)
//     for exactly this reason, and so does this port.
//
// `cmOutputs` is argument 10 of 15 and `midphaseCore` NEVER TOUCHES IT.  It is
// carried in the push constant because the CUDA signature has it and the blob
// must match byte for byte; it is deliberately unused below.

import pxgSlangCommon;
import pxgSlangBv32;
import pxgSlangTriBox;
import pxgSlangPcmManifold;

static const uint WARP = 32u;
static const uint MIDPHASE_WARPS_PER_BLOCK = 2u;
static const uint STACK_SIZE = 320u;          // MidphaseScratch::sBv32Nodes[320], 10 deep
static const uint PXG_GEOM_TRIANGLEMESH = 8u; // PxGeometryType::eTRIANGLEMESH

// cudaNpCommon.h:345/346
static const float MESH_MANIFOLD_EPSILON = 0.05f;
static const float BOX_MARGIN_RATIO      = 0.25f;

// ConvexMeshPair (convexNpCommon.h:70), 56 bytes:
//   aToB.q  @0 (float4)   aToB.p @16 (float3)   startIndex @28   count @32
//   cmIndex @36           roundedStartIndex @40  pad @44         materialIndices @48
static const uint64_t CMP_STRIDE = 56UL;
// CONVEX_TRIMESH_CACHED — the sentinel `count` for "the cached manifold stands"
// (convexNpCommon.h).  Written instead of a pair count when fullContactGen is false.
static const int CONVEX_TRIMESH_CACHED = -1;

struct MidphaseParams
{
    uint     numContactManagers;  // @0
    float    toleranceLength;     // @4
    uint64_t cmInputs;            // @8    const PxgContactManagerInput*
    uint64_t transformCache;      // @16   const PxsCachedTransform*
    uint64_t bounds;              // @24   const PxBounds3*
    uint64_t contactDistance;     // @32   const PxReal*
    uint64_t gpuShapes;           // @40   const PxgShape*
    uint64_t cvxTrimeshPair;      // @48   ConvexMeshPair*        (output)
    uint64_t multiManifolds;      // @56   PxgPersistentContactMultiManifold*
    uint64_t cmOutputs;           // @64   PxsContactManagerOutput*  (UNUSED, see header)
    uint64_t stackPtr;            // @72   PxU8*                  (output)
    uint64_t stackOffset;         // @80   PxU32*                 (atomic cursor)
    uint64_t midphasePairsNum;    // @88   PxU32*                 (atomic)
    uint64_t midphasePairsNumPadded; // @96 PxU32*                (atomic)
    uint     stackSizeBytes;      // @104
    uint     _pad108;             // @108 -> 112-byte blob
}

[[vk::push_constant]]
ConstantBuffer<MidphaseParams> gMp;

struct PxgContactManagerInput { uint shapeRef0, shapeRef1, transformCacheRef0, transformCacheRef1; }

// ---------------------------------------------------------------------------
// calculateMaxPairs (midphaseAllocate.cuh:119) and its two helpers.
//
//   memRequiredPerPair = sizeof(uint4) + sizeof(ConvexTriNormalAndIndex)
//                      + sizeof(ConvexTriContacts) + sizeof(PxReal)
//                      + sizeof(ConvexTriIntermediateData) + sizeof(PxU32)
//                      + sizeof(PxU32)*2
//                      = 16 + 16 + 4 + 4 + 4 + 4 + 8 = 56
//   additionalPadding  = 3*(memRequiredPerPair - 2*4) + numContactManagers*6*4
//   maxPairs           = (stackSizeBytes - additionalPadding) / memRequiredPerPair
//
// The subtraction is UNSIGNED and upstream does not guard it.  Transcribed as
// written; a stack too small for the padding would wrap in CUDA too.
// ---------------------------------------------------------------------------
static const uint MEM_REQUIRED_PER_PAIR = 56u;

uint calculateMaxPairs(uint stackSizeBytes, uint numContactManagers)
{
    const uint afterPairPadding = 3u * (MEM_REQUIRED_PER_PAIR - 2u * 4u);
    const uint afterTriangleIndexPadding = numContactManagers * 6u * 4u;
    const uint additionalPadding = afterPairPadding + afterTriangleIndexPadding;
    const uint memAvailableAfterPadding = stackSizeBytes - additionalPadding;
    return memAvailableAfterPadding / MEM_REQUIRED_PER_PAIR;
}

// shape2Vertex / vertex2Shape (cudaNpCommon.h:220).  shape2Vertex DIVIDES by the
// scale, vertex2Shape MULTIPLIES; both work in the scale rotation's frame.
float3 mpShape2Vertex(float3 v, float3 scale, float4 rot)
{
    return quatRotateInv(rot, quatRotate(rot, v) / scale);
}
float3 mpVertex2Shape(float3 v, float3 scale, float4 rot)
{
    return quatRotateInv(rot, quatRotate(rot, v) * scale);
}

// calculatePCMConvexMargin (cudaNpCommon.h:349): 25% of the smallest SCALED
// internal-AABB extent, capped at 5% of the tolerance length.
float calculatePCMConvexMargin(float4 extents_, float3 scale, float toleranceLength)
{
    const float3 extents = float3(extents_.x * scale.x, extents_.y * scale.y, extents_.z * scale.z);
    const float mn = min(min(extents.x, extents.y), extents.z);
    const float toleranceMargin = toleranceLength * MESH_MANIFOLD_EPSILON;
    return min(mn * BOX_MARGIN_RATIO, toleranceMargin);
}

// PxBounds3 is 24 bytes and lands 8-mod-16 on odd indices — scalars only.
void loadBounds(uint64_t base, uint i, out float3 mn, out float3 mx)
{
    const uint64_t a = base + uint64_t(i) * 24UL;
    mn = float3(Ptr<float>(a)[0], Ptr<float>(a + 4UL)[0], Ptr<float>(a + 8UL)[0]);
    mx = float3(Ptr<float>(a + 12UL)[0], Ptr<float>(a + 16UL)[0], Ptr<float>(a + 20UL)[0]);
}

// ConvexMeshPair_WriteWarp (dataReadWriteHelper.cuh:384) — a 14 x 4-byte scatter,
// which is what keeps a 56-byte stride safe at odd cmIdx.
void writeConvexMeshPair(uint64_t base, uint cmIdx, float4 aToBq, float3 aToBp,
                         int startIndex, int count, int cmIndex, int roundedStartIndex,
                         uint2 materialIndices)
{
    const uint64_t a = base + uint64_t(cmIdx) * CMP_STRIDE;
    Ptr<float>(a)[0] = aToBq.x;  Ptr<float>(a + 4UL)[0] = aToBq.y;
    Ptr<float>(a + 8UL)[0] = aToBq.z;  Ptr<float>(a + 12UL)[0] = aToBq.w;
    Ptr<float>(a + 16UL)[0] = aToBp.x; Ptr<float>(a + 20UL)[0] = aToBp.y;
    Ptr<float>(a + 24UL)[0] = aToBp.z;
    Ptr<int>(a + 28UL)[0] = startIndex;
    Ptr<int>(a + 32UL)[0] = count;
    Ptr<int>(a + 36UL)[0] = cmIndex;
    Ptr<int>(a + 40UL)[0] = roundedStartIndex;
    Ptr<int>(a + 44UL)[0] = 0;                  // pad, written for parity with the scatter
    Ptr<uint>(a + 48UL)[0] = materialIndices.x;
    Ptr<uint>(a + 52UL)[0] = materialIndices.y;
}

// ---------------------------------------------------------------------------
// Warp-emulation scratch.  One group == one virtual warp.
// ---------------------------------------------------------------------------
groupshared int sStack[STACK_SIZE];
groupshared uint sBallot;
groupshared uint sIndex0;    // atomicAdd result broadcast (__shfl_sync(.., 0))
groupshared uint sAnyLost;   // __any_sync for the manifold refresh

uint mpBallot(bool pred, uint lane)
{
    if (lane == 0u) sBallot = 0u;
    GroupMemoryBarrierWithGroupSync();
    if (pred) InterlockedOr(sBallot, 1u << lane);
    GroupMemoryBarrierWithGroupSync();
    return sBallot;
}

// ===========================================================================
[shader("compute")]
[numthreads(32, 1, 1)]
void midphaseGeneratePairs(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint lane = gtid.x;
    const uint maxPairs = calculateMaxPairs(gMp.stackSizeBytes, gMp.numContactManagers);
    const uint stackCapacity = gMp.stackSizeBytes / 16u;

    Ptr<PxgShape> shapes = Ptr<PxgShape>(gMp.gpuShapes);

    // The block's two warps, walked serially (see the header on the grid).
    for (uint sub = 0u; sub < MIDPHASE_WARPS_PER_BLOCK; ++sub)
    {
        const uint cmIdx = gid.x * MIDPHASE_WARPS_PER_BLOCK + sub;
        GroupMemoryBarrierWithGroupSync();      // scratch is reused per contact manager
        if (cmIdx >= gMp.numContactManagers)
            continue;                            // group-uniform

        const PxgContactManagerInput cin = Ptr<PxgContactManagerInput>(gMp.cmInputs)[cmIdx];
        uint transformCacheRef0 = cin.transformCacheRef0;
        uint transformCacheRef1 = cin.transformCacheRef1;
        uint shapeRef0 = cin.shapeRef0;
        uint shapeRef1 = cin.shapeRef1;

        // Slot 0 must end up as the convex/sphere/capsule and slot 1 as the mesh.
        if (shapes[shapeRef0].type == PXG_GEOM_TRIANGLEMESH)
        {
            const uint tc = transformCacheRef0; transformCacheRef0 = transformCacheRef1; transformCacheRef1 = tc;
            const uint sr = shapeRef0;          shapeRef0 = shapeRef1;                   shapeRef1 = sr;
        }

        const PxgShape shape = shapes[shapeRef0];
        const PxgShape trimeshShape = shapes[shapeRef1];

        // PxsCachedTransform (32 B): PxQuat q @0, PxVec3 p @16.
        const uint64_t xf0 = gMp.transformCache + uint64_t(transformCacheRef0) * 32UL;
        const uint64_t xf1 = gMp.transformCache + uint64_t(transformCacheRef1) * 32UL;
        const float4 q0 = Ptr<float4>(xf0)[0];
        const float3 p0 = float3(Ptr<float>(xf0 + 16UL)[0], Ptr<float>(xf0 + 20UL)[0], Ptr<float>(xf0 + 24UL)[0]);
        const float4 q1 = Ptr<float4>(xf1)[0];
        const float3 p1 = float3(Ptr<float>(xf1 + 16UL)[0], Ptr<float>(xf1 + 20UL)[0], Ptr<float>(xf1 + 24UL)[0]);

        // trimeshTransform.transformInv(shapeTransform): q = conj(q1)*q0, p = rotInv(q1, p0 - p1)
        const float4 aToBq = quatMul(quatConjugate(q1), q0);
        const float3 aToBp = quatRotateInv(q1, p0 - p1);

        const float3 shapeScale = float3(shape.scaleX, shape.scaleY, shape.scaleZ);
        const float4 shapeScaleRot = float4(shape.scaleRotX, shape.scaleRotY, shape.scaleRotZ, shape.scaleRotW);
        const float3 triScale = float3(trimeshShape.scaleX, trimeshShape.scaleY, trimeshShape.scaleZ);
        const float4 triScaleRot = float4(trimeshShape.scaleRotX, trimeshShape.scaleRotY,
                                          trimeshShape.scaleRotZ, trimeshShape.scaleRotW);

        const uint64_t hullPtr = uint64_t(shape.hullOrMeshOffset.x) |
                                 (uint64_t(shape.hullOrMeshOffset.y) << 32);

        // ---- margin / ratio, per shape type (convexMeshMidphase.cu:226) -------
        float minMargin, ratio, breakingThresholdRatio;
        if (shape.type == PXG_GEOM_SPHERE)
        {
            minMargin = shape.scaleX;            // sphere radius
            ratio = 0.02f; breakingThresholdRatio = 0.05f;
        }
        else if (shape.type == PXG_GEOM_CAPSULE)
        {
            minMargin = shape.scaleY;            // capsule radius
            ratio = 0.02f; breakingThresholdRatio = 0.05f;
        }
        else
        {
            // The convex hull's internal-AABB extents live at hullPtr + sizeof(float4)*2.
            const float4 extents4 = Ptr<float4>(hullPtr + 32UL)[0];
            minMargin = calculatePCMConvexMargin(extents4, shapeScale, gMp.toleranceLength);
            ratio = 0.2f; breakingThresholdRatio = 0.8f;
        }

        // ---- the PCM gate (P8.23) --------------------------------------------
        const uint64_t manifold = gMp.multiManifolds + uint64_t(cmIdx) * PCM_STRIDE;
        const bool invalidate = mmInvalidateManifold(manifold, aToBq, aToBp, minMargin, ratio);

        bool lostContacts = false;
        if (!invalidate)
        {
            // Warp-uniform: mmInvalidateManifold reads only the shared record, so
            // every lane agrees and the barriers below stay in uniform flow.
            const float projectBreakingThreshold = minMargin * breakingThresholdRatio;
            if (lane == 0u) sAnyLost = 0u;
            GroupMemoryBarrierWithGroupSync();
            const bool slotLost = mmRefreshSlot(manifold, aToBq, aToBp, projectBreakingThreshold, lane);
            if (slotLost) InterlockedOr(sAnyLost, 1u);
            GroupMemoryBarrierWithGroupSync();
            if (sAnyLost != 0u)
            {
                lostContacts = true;
                if (lane == 0u) Ptr<uint>(manifold + PCM_NBMANIFOLDS)[0] = 0u;
            }
        }

        const bool fullContactGen = invalidate || lostContacts;

        uint nbPairsPerCM = 0u;

        if (fullContactGen)
        {
            // ---- the convex's centre of mass, in mesh VERTEX space -------------
            float3 shapeCoMInTrimeshVertexSpace;
            if (shape.type == PXG_GEOM_SPHERE || shape.type == PXG_GEOM_CAPSULE)
            {
                shapeCoMInTrimeshVertexSpace =
                    mpShape2Vertex(quatRotateInv(q1, p0 - p1), triScale, triScaleRot);
            }
            else
            {
                const float4 cvxCoM = Ptr<float4>(hullPtr)[0];   // hull CoM at offset 0
                const float3 comShape = mpVertex2Shape(cvxCoM.xyz, shapeScale, shapeScaleRot);
                const float3 comInMesh = quatRotate(aToBq, comShape) + aToBp;
                shapeCoMInTrimeshVertexSpace = mpShape2Vertex(comInMesh, triScale, triScaleRot);
            }

            // ---- the query box ------------------------------------------------
            float3 aabbMin, aabbMax;
            loadBounds(gMp.bounds, transformCacheRef0, aabbMin, aabbMax);
            const float contactDist = Ptr<float>(gMp.contactDistance)[transformCacheRef0] +
                                      Ptr<float>(gMp.contactDistance)[transformCacheRef1];

            // The WORLD box the triangle test uses.  getExtents() is the HALF extent.
            const float3 center = (aabbMin + aabbMax) * 0.5f;
            const float3 inflatedExtents = (aabbMax - aabbMin) * 0.5f + float3(contactDist, contactDist, contactDist);

            // The VERTEX-space box the tree walk uses: translate by -meshPos, fatten,
            // then transform by inverse(meshScale) * conj(meshRot).  PxBounds3::transformFast
            // over a rotation is the centre/extent form below.
            float3 vMin = aabbMin - p1 - float3(contactDist, contactDist, contactDist);
            float3 vMax = aabbMax - p1 + float3(contactDist, contactDist, contactDist);
            {
                const float3 c = (vMin + vMax) * 0.5f;
                const float3 e = (vMax - vMin) * 0.5f;
                // worldToTrimeshVertex = meshScale.getInverse().toMat33() * Mat33(conj(q1)).
                // Applied to the centre directly and to the extents through |M|.
                const float3 cRot = quatRotateInv(q1, c);
                const float3 cV = mpShape2Vertex(cRot, triScale, triScaleRot);
                // Extent transform: the row-absolute sum of the same linear map.  Built by
                // mapping the three axis extents and taking |.| componentwise, which is the
                // same bound PxBounds3::transformFast computes.
                const float3 ex = abs(mpShape2Vertex(quatRotateInv(q1, float3(e.x, 0, 0)), triScale, triScaleRot));
                const float3 ey = abs(mpShape2Vertex(quatRotateInv(q1, float3(0, e.y, 0)), triScale, triScaleRot));
                const float3 ez = abs(mpShape2Vertex(quatRotateInv(q1, float3(0, 0, e.z)), triScale, triScaleRot));
                const float3 eV = ex + ey + ez;
                vMin = cV - eV;
                vMax = cV + eV;
            }

            // ---- the cooked mesh blob (readTriangleMesh, triangleMesh.cuh:120) --
            const uint64_t meshPtr = uint64_t(trimeshShape.hullOrMeshOffset.x) |
                                     (uint64_t(trimeshShape.hullOrMeshOffset.y) << 32);
            const uint4 meshHeader = Ptr<uint4>(meshPtr)[0];
            const uint64_t treeAddr = meshPtr + 16UL;
            // .w is nbPackedNodes and sizes the node ARRAY; .z is a different count.
            const uint64_t vertsAddr = treeAddr + uint64_t(meshHeader.w) * BV32_STRIDE;
            const uint64_t indicesAddr = vertsAddr + uint64_t(meshHeader.x) * 16UL;
            Ptr<float4> triVerts = Ptr<float4>(vertsAddr);
            Ptr<uint4> triIndices = Ptr<uint4>(indicesAddr);

            // ---- the walk ------------------------------------------------------
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
                    // Six-bit field, but one lane per primitive: CUDA drops past 32 and
                    // so does this.  See pxgSlangBv32's header.
                    const uint nbPrimitives = min((data >> 1u) & 63u, WARP);
                    const uint primStartIndex = data >> 7u;
                    uint primitiveIndex = 0xffffffffu;
                    if (lane < nbPrimitives)
                        primitiveIndex = lane + primStartIndex;

                    bool primitiveIntersects = false;
                    if (primitiveIndex != 0xffffffffu)
                    {
                        const uint4 triIdx = triIndices[primitiveIndex];
                        const float3 triV0 = triVerts[triIdx.x].xyz;
                        const float3 triV1 = triVerts[triIdx.y].xyz;
                        const float3 triV2 = triVerts[triIdx.z].xyz;

                        // PxPlane(triV0, triV1, triV2): n normalized, d = -n.dot(v0).
                        const float3 nRaw = cross(triV1 - triV0, triV2 - triV0);
                        const float nl = dot(nRaw, nRaw);
                        const float3 tn = nl > 0.0f ? nRaw * rsqrt(nl) : float3(0, 0, 0);
                        const float td = -dot(tn, triV0);

                        // The front-face gate: only triangles whose plane the convex's
                        // centre is on the positive side of.
                        if (dot(tn, shapeCoMInTrimeshVertexSpace) + td >= 0.0f)
                        {
                            const float3 w0 = quatRotate(q1, mpVertex2Shape(triV0, triScale, triScaleRot)) + p1;
                            const float3 w1 = quatRotate(q1, mpVertex2Shape(triV1, triScale, triScaleRot)) + p1;
                            const float3 w2 = quatRotate(q1, mpVertex2Shape(triV2, triScale, triScaleRot)) + p1;
                            primitiveIntersects = tbIntersectTriangleBox(center, inflatedExtents, w0, w1, w2);
                        }
                    }

                    // Pair emission.  The counter is bumped even when the write is
                    // dropped, exactly as the CUDA comments say ("even if we might
                    // overflow, we increase the stack offset. It's not written though.")
                    const uint mask = mpBallot(primitiveIntersects, lane);
                    const uint offset = countbits(mask & ((1u << lane) - 1u));
                    uint validCount = countbits(mask);

                    // The traverser's cursor is stackOffset (the 5th ctor arg,
                    // convexMeshMidphase.cu:349), NOT midphasePairsNum.  They are two
                    // different counters: stackOffset is the global uint4 write cursor
                    // into stackPtr, bumped per LEAF; midphasePairsNum is the per-manager
                    // allocation counter, bumped ONCE at the end with this manager's
                    // total.  Crossing them writes pairs at the wrong slot and hands the
                    // core kernel a startIndex that points at another manager's range.
                    if (lane == 0u)
                    {
                        sIndex0 = 0xffffffffu;
                        if (validCount > 0u)
                            InterlockedAdd(Ptr<uint>(gMp.stackOffset)[0], validCount, sIndex0);
                    }
                    GroupMemoryBarrierWithGroupSync();
                    const uint prevNbPairs = sIndex0;

                    if (primitiveIntersects && ((prevNbPairs + offset) < maxPairs) &&
                        ((prevNbPairs + offset) < stackCapacity))
                    {
                        Ptr<uint4>(gMp.stackPtr)[prevNbPairs + offset] =
                            uint4(cmIdx, primitiveIndex, nbPairsPerCM + offset, shapeRef1);
                    }

                    if ((validCount > 0u) && ((validCount + prevNbPairs) >= maxPairs))
                        validCount = max(maxPairs, prevNbPairs) - prevNbPairs;

                    nbPairsPerCM += validCount;
                }
                else
                {
                    const uint nbChildren = bvNbNodes(treeAddr, nodeIdx);
                    bool intersect = false;
                    if (lane < nbChildren)
                    {
                        const float3 mn = bvMin(treeAddr, nodeIdx, lane);
                        const float3 mx = bvMax(treeAddr, nodeIdx, lane);
                        intersect = !(mn.x > vMax.x || vMin.x > mx.x ||
                                      mn.y > vMax.y || vMin.y > mx.y ||
                                      mn.z > vMax.z || vMin.z > mx.z);
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

                    const uint mask = mpBallot(isValidNode, lane);
                    const uint offset = countbits(mask & ((1u << lane) - 1u));
                    const uint validCount = countbits(mask);
                    if (isValidNode && (nbStack + offset) < STACK_SIZE)
                        sStack[nbStack + offset] = int(currentIndex);
                    nbStack += min(validCount, STACK_SIZE - nbStack);
                }

                GroupMemoryBarrierWithGroupSync();
                if (nbStack > 0u)
                {
                    nbStack -= 1u;
                    currentNodeIndex = uint(sStack[nbStack]);
                }
                else
                    hasNode = false;
                GroupMemoryBarrierWithGroupSync();
            }
        }

        // ---- the per-pair record --------------------------------------------
        // Both counters are bumped by lane 0 only and broadcast, mirroring the
        // CUDA's `if (idxInWarp == 0 && nbPairsPerCM > 0)` + __shfl_sync pair.
        if (lane == 0u)
        {
            // convexMeshMidphase.cu:353 — both counters bumped once, by lane 0, only
            // when this manager found pairs.  0xFFFFFFFF is the "no pairs" sentinel and
            // is written into the record as-is when nbPairsPerCM is zero.
            uint prevOffset = 0xffffffffu;
            uint prevPaddedOffset = 0xffffffffu;
            if (nbPairsPerCM > 0u)
            {
                InterlockedAdd(Ptr<uint>(gMp.midphasePairsNum)[0], nbPairsPerCM, prevOffset);
                InterlockedAdd(Ptr<uint>(gMp.midphasePairsNumPadded)[0],
                               ((nbPairsPerCM + 3u) & ~3u) * 2u, prevPaddedOffset);
            }
            writeConvexMeshPair(gMp.cvxTrimeshPair, cmIdx, aToBq, aToBp,
                                int(prevOffset),
                                fullContactGen ? int(nbPairsPerCM) : CONVEX_TRIMESH_CACHED,
                                int(cmIdx), int(prevPaddedOffset),
                                uint2(shape.materialIndex, trimeshShape.materialIndex));
        }
    }
}
